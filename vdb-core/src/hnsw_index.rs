use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use std::collections::HashMap;
use hnsw_rs::prelude::{Hnsw, DistCosine, DistL2, DistL1};
use serde_json::Value;

// Define the distance type enum to handle different distance metrics
enum DistanceType {
    Cosine(Hnsw<'static, f32, DistCosine>),
    L2(Hnsw<'static, f32, DistL2>),
    L1(Hnsw<'static, f32, DistL1>),
}

// Structured results
#[derive(Debug, Clone)]
#[pyclass]
pub struct AddResult {
    #[pyo3(get)]
    pub total_inserted: usize,
    #[pyo3(get)]
    pub total_errors: usize,
    #[pyo3(get)]
    pub errors: Vec<String>,
    #[pyo3(get)]
    pub vector_shape: Option<(usize, usize)>, // (count, dimension)
}

#[pymethods]
impl AddResult {
    fn __repr__(&self) -> String {
        format!(
            "AddResult(inserted={}, errors={}, shape={:?})",
            self.total_inserted, self.total_errors, self.vector_shape
        )
    }

    pub fn is_success(&self) -> bool {
        self.total_errors == 0
    }

    pub fn summary(&self) -> String {
        format!("✅ {} inserted, ❌ {} errors", self.total_inserted, self.total_errors)
    }
}

#[pyclass]
pub struct HNSWIndex {
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,

    // Index-level metadata
    metadata: HashMap<String, String>,

    // Vector store
    vectors: HashMap<String, Vec<f32>>,
    vector_metadata: HashMap<String, HashMap<String, Value>>,

    hnsw: DistanceType,
    id_map: HashMap<String, usize>,     // Maps external ID → usize
    rev_map: HashMap<usize, String>,    // Maps usize → external ID
    id_counter: usize,
}

#[pymethods]
impl HNSWIndex {
    #[new]
    fn new(
        dim: usize, 
        space: String,
        m: usize, 
        ef_construction: usize,
        expected_size: usize
    ) -> PyResult<Self> {  // Return PyResult for validation
        // Validate parameters in Rust
        if dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dim must be positive"
            ));
        }
        if ef_construction == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ef_construction must be positive"
            ));
        }
        if expected_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "expected_size must be positive"
            ));
        }
        if m > 256 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "M must be less than or equal to 256"
            ));
        }

        // Normalize space to lowercase for case-insensitive matching
        let space_normalized = space.to_lowercase();
        // Choose the distance metric dynamically
        let max_layer = (expected_size as f32).log2().ceil() as usize;

        let hnsw = match space_normalized.as_str() {
            "cosine" => DistanceType::Cosine(Hnsw::new(m, expected_size, max_layer, ef_construction, DistCosine {})),
            "l2"     => DistanceType::L2(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL2 {})),
            "l1"     => DistanceType::L1(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL1 {})),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported space: '{}'. Must be 'cosine', 'l2', or 'l1' (case-insensitive)", space)
                ));
            }
        };
        
        Ok(HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            metadata: HashMap::new(),
            vectors: HashMap::new(),
            vector_metadata: HashMap::new(),
            hnsw,
            id_map: HashMap::new(),
            rev_map: HashMap::new(),
            id_counter: 0,
        })
    }

    /// Unified add method supporting all input formats
    pub fn add(&mut self, data: Bound<PyAny>) -> PyResult<AddResult> {
        let records = if let Ok(list) = data.downcast::<PyList>() {
            // Format 2: List of objects
            self.parse_list_format(&list)?
        } else if let Ok(dict) = data.downcast::<PyDict>() {
            if dict.contains("ids")? {
                // Format 3: Separate arrays
                self.parse_separate_arrays(&dict)?
            } else {
                // Format 1: Single object
                self.parse_single_object(&dict)?
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid format: expected dict, list, or object with 'id' field"
            ));
        };

        self.add_batch_internal(records)
    }

    /// Search for the k-nearest neighbors of a vector
    /// Returns actual Python dictionaries which most common for ML workflows
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false))]
    pub fn query(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
    ) -> PyResult<Vec<Py<PyDict>>> {
        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dim, vector.len()
            )));
        }

        let ef = ef_search.unwrap_or_else(|| std::cmp::max(2 * top_k, 100));

        let results = match &self.hnsw {
            DistanceType::Cosine(hnsw) => hnsw.search(&vector, top_k, ef),
            DistanceType::L2(hnsw) => hnsw.search(&vector, top_k, ef),
            DistanceType::L1(hnsw) => hnsw.search(&vector, top_k, ef),
        };

        let mut output = Vec::with_capacity(results.len());

        for neighbor in results {
            let score = neighbor.distance;
            let internal_id = neighbor.get_origin_id();

            if let Some(ext_id) = self.rev_map.get(&internal_id) {
                // Apply metadata filter if provided
                if let Some(filter_dict) = filter {
                    let filter_conditions = self.python_dict_to_value_map(filter_dict)?;
                    if let Some(meta) = self.vector_metadata.get(ext_id) {
                        if !self.matches_filter(meta, &filter_conditions)? {
                            continue;
                        }
                    } else {
                        continue; // no metadata to match against
                    }
                }

                let dict = PyDict::new(py);
                dict.set_item("id", ext_id)?;
                dict.set_item("score", score)?;

                let metadata = self.vector_metadata.get(ext_id).cloned().unwrap_or_default();
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    if let Some(vec) = self.vectors.get(ext_id) {
                        dict.set_item("vector", vec.clone())?;
                    }
                }

                output.push(dict.into());
            }
        }

        Ok(output)
    }


    /// Get one or more records by ID(s).
    /// Accepts a single ID or a list of IDs, and optionally returns vectors.
    ///
    /// Parameters:
    /// - `input`: `str` or `List[str]`
    /// - `return_vector`: if `True`, include the embedding vector in each result
    ///
    /// Returns:
    /// - List of dictionaries with fields: `id`, `metadata`, and optionally `vector`
    #[pyo3(signature = (input, return_vector = true))]
    pub fn get_records(&self, py: Python<'_>, input: &Bound<PyAny>, return_vector: bool) -> PyResult<Vec<Py<PyDict>>> {
        let ids: Vec<String> = if let Ok(id_str) = input.extract::<String>() {
            vec![id_str]
        } else if let Ok(id_list) = input.extract::<Vec<String>>() {
            id_list
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a string or a list of strings for ID(s)"
            ));
        };

        let mut records = Vec::with_capacity(ids.len());

        for id in ids {
            if let Some(vector) = self.vectors.get(&id) {
                let metadata = self.vector_metadata.get(&id).cloned().unwrap_or_default();

                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    dict.set_item("vector", vector.clone())?;
                }

                records.push(dict.into());
            }
        }

        Ok(records)
    }


    /// Get comprehensive statistics
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("total_vectors".to_string(), self.vectors.len().to_string());
        stats.insert("dimension".to_string(), self.dim.to_string());
        stats.insert("space".to_string(), self.space.clone());
        stats.insert("M".to_string(), self.m.to_string());
        stats.insert("ef_construction".to_string(), self.ef_construction.to_string());
        stats.insert("expected_size".to_string(), self.expected_size.to_string());
        stats.insert("index_type".to_string(), "HNSW".to_string());
        stats
    }

    /// List the first `number` records in the index (ID and metadata).
    #[pyo3(signature = (number=10))]
    pub fn list(&self, py: Python<'_>, number: usize) -> PyResult<Vec<(String, PyObject)>> {
        let mut results = Vec::new();
        for (id, _vec) in self.vectors.iter().take(number) {
            let metadata = self.vector_metadata.get(id).cloned().unwrap_or_default();
            let py_metadata = self.value_map_to_python(&metadata, py)?;
            results.push((id.clone(), py_metadata));
        }
        Ok(results)
    }
    
    /// Add multiple key-value pairs to index-level metadata
    pub fn add_metadata(&mut self, metadata: HashMap<String, String>) {
        for (key, value) in metadata {
            self.metadata.insert(key, value);
        }
    }

    /// Get a single index-level metadata value
    pub fn get_metadata(&self, key: String) -> Option<String> {
        self.metadata.get(&key).cloned()
    }

    /// Get all index-level metadata
    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    /// Returns basic info about the index
    pub fn info(&self) -> String {
        format!(
            "HNSWIndex(dim={}, space={}, M={}, ef_construction={}, expected_size={}, vectors={})",
            self.dim,
            self.space,
            self.m,
            self.ef_construction,
            self.expected_size,
            self.vectors.len()
        )
    }

    /// Check if vector ID exists
    pub fn contains(&self, id: String) -> bool {
        self.vectors.contains_key(&id)
    }

    /// Remove vector by ID
    /// Removes the vector and its metadata from all accessible mappings.
    /// The point will no longer appear in queries, contains() checks, or be
    /// retrievable by ID. 
    /// 
    /// Note: Due to HNSW algorithm limitations, the underlying graph structure
    /// retains stale nodes internally, but these are completely inaccessible
    /// to users and do not affect query results or performance.
    /// 
    /// Returns:
    ///   - `Ok(true)` if the vector was found and removed
    ///   - `Ok(false)` if the vector ID was not found
    pub fn remove_point(&mut self, id: String) -> PyResult<bool> {
        if let Some(internal_id) = self.id_map.remove(&id) {
            self.vectors.remove(&id);
            self.vector_metadata.remove(&id);
            self.rev_map.remove(&internal_id);
            // Note: HNSW doesn't support removal, so the graph still contains the point
            // but it won't be accessible via the mappings
            Ok(true)
        } else {
            Ok(false)
        }
    }
}


// Separate impl block for private helper methods (NO #[pymethods])
impl HNSWIndex {
    // Additional  Internal logic, private to Rust methods 

    /// INPUT DATA FORMAT 1
    /// Parse single object format: {"id": "doc1", "values": [0.1, 0.2], "metadata": {...}}
    fn parse_single_object(&self, dict: &Bound<PyDict>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>> {
        // Extract ID
        let id = dict.get_item("id")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing required field 'id'"))?
            .extract::<String>()?;

        // Extract vector - support both "values" and "vector" keys
        let vector = self.extract_vector_from_dict(dict, "object")?;

        // Extract metadata as Value to preserve Python types automatically
        let metadata = if let Some(meta_item) = dict.get_item("metadata")? {
            Some(self.python_dict_to_value_map(meta_item.downcast::<PyDict>()?)?)
        } else {
            None
        };

        Ok(vec![(id, vector, metadata)])
    }

    /// INPUT DATA FORMAT 2
    /// Parse list format: [{"id": "doc1", "values": [...]}, ...]
    fn parse_list_format(&self, list: &Bound<PyList>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>> {
        let mut records = Vec::with_capacity(list.len());
        
        for (i, item) in list.iter().enumerate() {
            let dict = item.downcast::<PyDict>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Item {}: expected dict object", i)
                ))?;

            // Extract ID
            let id = dict.get_item("id")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Item {}: missing required field 'id'", i)
                ))?
                .extract::<String>()?;

            // Extract vector
            let vector = self.extract_vector_from_dict(dict, &format!("item {}", i))?;

            // Extract metadata
            let metadata = if let Some(meta_item) = dict.get_item("metadata")? {
                Some(self.python_dict_to_value_map(meta_item.downcast::<PyDict>()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Item {}: metadata must be a dictionary", i)
                    ))?)?)
            } else {
                None
            };

            records.push((id, vector, metadata));
        }

        Ok(records)
    }

    /// INPUT DATA FORMAT 3
    /// Parse separate arrays format: {"ids": [...], "embeddings": [...], "metadatas": [...]}
    fn parse_separate_arrays(&self, dict: &Bound<PyDict>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>> {
        // Extract IDs
        let ids = dict.get_item("ids")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing required field 'ids'"))?
            .extract::<Vec<String>>()?;

        // Extract vectors - check for NumPy array first
        let vectors = self.extract_vectors_from_separate_arrays(dict)?;

        // Validate dimensions early
        if vectors.len() != ids.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Length mismatch: {} ids vs {} vectors", ids.len(), vectors.len())
            ));
        }

        // Extract metadatas (optional)
        let metadatas = if let Some(meta_item) = dict.get_item("metadatas")? {
            let meta_list = meta_item.downcast::<PyList>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Field 'metadatas' must be a list"
                ))?;

            let mut metas = Vec::with_capacity(meta_list.len());
            for (i, meta_item) in meta_list.iter().enumerate() {
                if meta_item.is_none() {
                    metas.push(None);
                } else {
                    let meta_dict = meta_item.downcast::<PyDict>()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("metadatas[{}] must be a dictionary or None", i)
                        ))?;
                    metas.push(Some(self.python_dict_to_value_map(&meta_dict)?));
                }
            }

            if metas.len() != ids.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Length mismatch: {} ids vs {} metadatas", ids.len(), metas.len())
                ));
            }
            metas
        } else {
            vec![None; ids.len()]
        };

        // Combine into records
        let records = ids.into_iter()
            .zip(vectors.into_iter())
            .zip(metadatas.into_iter())
            .map(|((id, vector), metadata)| (id, vector, metadata))
            .collect();

        Ok(records)
    }


    /// HELPER FUNCTION
    /// Extract vector from dict, supporting both "values" and "vector" keys
    fn extract_vector_from_dict(&self, dict: &Bound<PyDict>, context: &str) -> PyResult<Vec<f32>> {
        let vector_item = dict.get_item("values")?
            .or_else(|| dict.get_item("vector").ok().flatten())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("{}: missing required field 'values' or 'vector'", context)
            ))?;

        if let Ok(array1d) = vector_item.downcast::<PyArray1<f32>>() {
            Ok(array1d.readonly().as_slice()?.to_vec())
        } else if let Ok(array2d) = vector_item.downcast::<PyArray2<f32>>() {
            let readonly = array2d.readonly();
            let shape = readonly.shape();

            // Accept either (1, N) or (N, 1) for single vectors
            if (shape[0] == 1 && shape[1] > 0) || (shape[1] == 1 && shape[0] > 0) {
                Ok(readonly.as_slice()?.to_vec())
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("{}: expected single vector (1×N or N×1), got shape {:?}", context, shape)
                ));
            }
        } else {
            vector_item.extract::<Vec<f32>>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("{}: invalid vector format: {}", context, e)
                ))
        }
    }


    /// HELPER FUNCTION
    /// Extract vectors from separate arrays format with NumPy support
    fn extract_vectors_from_separate_arrays(&self, dict: &Bound<PyDict>) -> PyResult<Vec<Vec<f32>>> {
        let vectors_item = dict.get_item("embeddings")?
            .or_else(|| dict.get_item("values").ok().flatten())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Missing required field 'embeddings' or 'values'"
            ))?;

        // Try NumPy 2D array first (fastest path)
        if let Ok(array) = vectors_item.downcast::<PyArray2<f32>>() {
            let readonly = array.readonly();
            let shape = readonly.shape();
            
            if shape.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("NumPy array must be 2D, got {}D", shape.len())
                ));
            }

            let slice = readonly.as_slice()?;
            let (rows, cols) = (shape[0], shape[1]);
            
            // Convert to Vec<Vec<f32>>
            let mut vectors = Vec::with_capacity(rows);
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                vectors.push(slice[start..end].to_vec());
            }
            
            Ok(vectors)
        } else {
            // Fall back to Vec<Vec<f32>>
            vectors_item.extract::<Vec<Vec<f32>>>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid vectors format: {}", e)
                ))
        }
    }


    /// Internal batch processing with validation and structured results
    fn add_batch_internal(&mut self, records: Vec<(String, Vec<f32>, Option<HashMap<String, Value>>)>) -> PyResult<AddResult> {
        if records.is_empty() {
            return Ok(AddResult {
                total_inserted: 0,
                total_errors: 0,
                errors: vec![],
                vector_shape: Some((0, self.dim)),
            });
        }

        // Reserve capacity upfront for better performance
        let capacity = records.len();
        self.vectors.reserve(capacity);
        self.id_map.reserve(capacity);
        self.rev_map.reserve(capacity);
        self.vector_metadata.reserve(capacity);

        // Process records - each handles its own validation
        let mut errors = Vec::with_capacity(records.len());
        let mut success_count = 0;

        for (id, vector, metadata) in records.iter() {
            match self.add_point_internal(id.clone(), vector.clone(), metadata.clone()) {
                Ok(()) => success_count += 1,
                Err(e) => errors.push(format!("ID '{}': {}", id, e)),
            }
        }

        Ok(AddResult {
            total_inserted: success_count,
            total_errors: errors.len(),
            errors,
            vector_shape: Some((records.len(), self.dim)),
        })
    }


    /// Adds or updates a vector in the index.
    /// If the ID already exists, the vector and metadata are overwritten.
    /// Stale graph nodes are left in place but excluded from all queries.
    fn add_point_internal(&mut self, id: String, vector: Vec<f32>, metadata: Option<HashMap<String, Value>>) -> PyResult<()> {
        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Vector dimension mismatch: expected {}, got {} (id: '{}')", self.dim, vector.len(), id)
            ));
        }

        // Logical overwrite: remove any previous data
        if let Some(internal_id) = self.id_map.remove(&id) {
            self.rev_map.remove(&internal_id);
        }
        self.vectors.remove(&id);
        self.vector_metadata.remove(&id);

        let internal_id = self.id_counter;
        self.id_counter += 1;

        self.vectors.insert(id.clone(), vector);
        let stored_vec = self.vectors.get(&id).unwrap();

        // Dispatch to the correct HNSW variant
        match &mut self.hnsw {
            DistanceType::Cosine(hnsw) => hnsw.insert((stored_vec.as_slice(), internal_id)),
            DistanceType::L2(hnsw) => hnsw.insert((stored_vec.as_slice(), internal_id)),
            DistanceType::L1(hnsw) => hnsw.insert((stored_vec.as_slice(), internal_id)),
        };

        self.id_map.insert(id.clone(), internal_id);
        self.rev_map.insert(internal_id, id.clone());

        if let Some(meta) = metadata {
            self.vector_metadata.insert(id, meta);
        }

        Ok(())
    }

    
    /// HELPER FUNCTION
    /// Convert Python dict to HashMap<String, Value> for metadata storage
    fn python_dict_to_value_map(&self, py_dict: &Bound<PyDict>) -> PyResult<HashMap<String, Value>> {
        let mut map = HashMap::new();
        
        for (key, value) in py_dict.iter() {
            let string_key = key.extract::<String>()?;
            let json_value = self.python_object_to_value(&value)?;
            map.insert(string_key, json_value);
        }
        
        Ok(map)
    }

    /// HELPER FUNCTION  
    /// Convert Python object to serde_json::Value
    fn python_object_to_value(&self, py_obj: &Bound<PyAny>) -> PyResult<Value> {
        if py_obj.is_none() {
            Ok(Value::Null)
        } else if let Ok(b) = py_obj.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if let Ok(i) = py_obj.extract::<i64>() {
            Ok(Value::Number(serde_json::Number::from(i)))
        } else if let Ok(f) = py_obj.extract::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(f) {
                Ok(Value::Number(num))
            } else {
                Ok(Value::String(f.to_string()))
            }
        } else if let Ok(s) = py_obj.extract::<String>() {
            Ok(Value::String(s))
        } else if let Ok(py_list) = py_obj.downcast::<PyList>() {
            let mut vec = Vec::new();
            for item in py_list.iter() {
                vec.push(self.python_object_to_value(&item)?);
            }
            Ok(Value::Array(vec))
        } else if let Ok(py_dict) = py_obj.downcast::<PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, value) in py_dict.iter() {
                let string_key = key.extract::<String>()?;
                let json_value = self.python_object_to_value(&value)?;
                map.insert(string_key, json_value);
            }
            Ok(Value::Object(map))
        } else {
            // Fallback: convert to string representation
            Ok(Value::String(py_obj.to_string()))
        }
    }

    /// Filter Support Helpers
    // Simple but powerful filter matching using Value
    fn matches_filter(&self, metadata: &HashMap<String, Value>, filter: &HashMap<String, Value>) -> PyResult<bool> {
        for (field, condition) in filter {
            if !self.field_matches(metadata, field, condition)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn field_matches(&self, metadata: &HashMap<String, Value>, field: &str, condition: &Value) -> PyResult<bool> {
        let field_value = match metadata.get(field) {
            Some(value) => value,
            None => return Ok(false),
        };

        match condition {
            // Direct equality for simple values
            Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {
                Ok(field_value == condition)
            },
            
            // Complex conditions: {"gt": 5, "lt": 10}
            Value::Object(ops) => {
                self.evaluate_value_conditions(field_value, ops)
            },
            
            _ => Ok(false),
        }
    }

    fn evaluate_value_conditions(&self, field_value: &Value, operations: &serde_json::Map<String, Value>) -> PyResult<bool> {
        for (op, target_value) in operations {
            let matches = match op.as_str() {
                "eq" => field_value == target_value,
                "ne" => field_value != target_value,
                
                // Numeric comparisons - serde_json handles the conversion
                "gt" => self.compare_values(field_value, target_value, |a, b| a > b)?,
                "gte" => self.compare_values(field_value, target_value, |a, b| a >= b)?,
                "lt" => self.compare_values(field_value, target_value, |a, b| a < b)?,
                "lte" => self.compare_values(field_value, target_value, |a, b| a <= b)?,
                
                // String operations
                "contains" => self.value_contains(field_value, target_value)?,
                "startswith" => self.value_starts_with(field_value, target_value)?,
                "endswith" => self.value_ends_with(field_value, target_value)?,
                
                // Array operations
                "in" => self.value_in_array(field_value, target_value)?,
                
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Unknown filter operation: {}", op)
                    ));
                }
            };
            
            if !matches {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // Helper methods for Value comparisons
    fn compare_values<F>(&self, a: &Value, b: &Value, op: F) -> PyResult<bool>
    where
        F: Fn(f64, f64) -> bool,
    {
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                let f1 = n1.as_f64().unwrap_or(0.0);
                let f2 = n2.as_f64().unwrap_or(0.0);
                Ok(op(f1, f2))
            },
            _ => Ok(false),
        }
    }

    fn value_contains(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.contains(s2)),
            (Value::Array(arr), val) => Ok(arr.contains(val)),
            _ => Ok(false),
        }
    }

    fn value_starts_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.starts_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_ends_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.ends_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_in_array(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match target {
            Value::Array(arr) => Ok(arr.contains(field)),
            _ => Ok(false),
        }
    }

    // Convert Value HashMap back to Python objects
    fn value_map_to_python(&self, value_map: &HashMap<String, Value>, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for (key, value) in value_map {
            let py_value = self.value_to_python_object(value, py)?;
            dict.set_item(key, py_value)?;
        }
        
        //Ok(dict.into())
        Ok(dict.into_pyobject(py)?.to_owned().unbind().into_any())
    }

    // Convert individual Value to Python object
    fn value_to_python_object(&self, value: &Value, py: Python<'_>) -> PyResult<PyObject> {
        let py_obj = match value {
            Value::Null => py.None(),
            Value::Bool(b) => b.into_pyobject(py)?.to_owned().unbind().into_any(),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i.into_pyobject(py)?.to_owned().unbind().into_any()
                } else if let Some(f) = n.as_f64() {
                    f.into_pyobject(py)?.to_owned().unbind().into_any()
                } else {
                    n.to_string().into_pyobject(py)?.to_owned().unbind().into_any()
                }
            },
            Value::String(s) => s.clone().into_pyobject(py)?.unbind().into_any(),
            Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    py_list.append(self.value_to_python_object(item, py)?)?;
                }
                py_list.unbind().into_any() 
            },
            Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (k, v) in obj {
                    py_dict.set_item(k, self.value_to_python_object(v, py)?)?;
                }
                py_dict.unbind().into_any()
            }
        };
        
        Ok(py_obj)
    }

}
