use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use std::collections::HashMap;
use hnsw_rs::prelude::{Hnsw, DistCosine};

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
    vector_metadata: HashMap<String, HashMap<String, String>>,

    hnsw: Hnsw<'static, f32, DistCosine>,  // Actual graph
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
        if space != "cosine" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported space: {}. Only 'cosine' is supported", space)
            ));
        }

        // Calculate max_layer as log2(expected_size).ceil()
        let max_layer = (expected_size as f32).log2().ceil() as usize;
        let hnsw = Hnsw::<f32, DistCosine>::new(
            m,                // M
            expected_size,    // expected number of vectors
            max_layer,        // number of layers
            ef_construction,  // ef
            DistCosine {}
        );
        
        Ok(HNSWIndex {
            dim,
            space,
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

    /// INPUT DATA FORMAT 1
    /// Parse single object format: {"id": "doc1", "values": [0.1, 0.2], "metadata": {...}}
    fn parse_single_object(&self, dict: &Bound<PyDict>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>> {
        // Extract ID
        let id = dict.get_item("id")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing required field 'id'"))?
            .extract::<String>()?;

        // Extract vector - support both "values" and "vector" keys
        let vector = self.extract_vector_from_dict(dict, "object")?;

        // Extract metadata
        let metadata = dict.get_item("metadata")?
            .map(|m| m.extract::<HashMap<String, String>>())
            .transpose()?;

        Ok(vec![(id, vector, metadata)])
    }

    /// INPUT DATA FORMAT 2
    /// Parse list format: [{"id": "doc1", "values": [...]}, ...]
    fn parse_list_format(&self, list: &Bound<PyList>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>> {
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
            let metadata = dict.get_item("metadata")?
                .map(|m| m.extract::<HashMap<String, String>>())
                .transpose()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Item {}: invalid metadata: {}", i, e)
                ))?;

            records.push((id, vector, metadata));
        }

        Ok(records)
    }

    /// INPUT DATA FORMAT 3
    /// Parse separate arrays format: {"ids": [...], "embeddings": [...], "metadatas": [...]}
    fn parse_separate_arrays(&self, dict: &Bound<PyDict>) -> PyResult<Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>> {
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
            let metas = meta_item.extract::<Vec<Option<HashMap<String, String>>>>()?;
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
    fn add_batch_internal(&mut self, records: Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>) -> PyResult<AddResult> {
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

        // Validate vector dimensions early
        let expected_dim = self.dim;
        for (i, (id, vector, _)) in records.iter().enumerate() {
            if vector.len() != expected_dim {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!(
                        "Inconsistent vector dimensions at index {}: expected {}, got {} (id: '{}')",
                        i, expected_dim, vector.len(), id
                    )
                ));
            }
        }

        // Process records
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
            vector_shape: Some((records.len(), expected_dim)),
        })
    }


    // /// Internal add_point without external validation (already validated)
    // fn add_point_internal(&mut self, id: String, vector: Vec<f32>, metadata: Option<HashMap<String, String>>) -> PyResult<()> {
    //     // Check for duplicate ID first (cheapest operation)
    //     if self.vectors.contains_key(&id) {
    //         return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
    //             "Duplicate ID: '{}' already exists", id
    //         )));
    //     }

    //     // Assign internal index
    //     let internal_id = self.id_counter;
    //     self.id_counter += 1;

    //     // Store vector first to get stable memory location
    //     self.vectors.insert(id.clone(), vector);
    
    //     // Get stable reference to stored vector for HNSW
    //     let stored_vec = self.vectors.get(&id).unwrap();
    //     self.hnsw.insert((stored_vec.as_slice(), internal_id));
    
    //     // Store ID mappings (reduced cloning)
    //     self.id_map.insert(id.clone(), internal_id);
    //     self.rev_map.insert(internal_id, id.clone());
    
    //     // Store metadata with final move (no clone needed)
    //     if let Some(meta) = metadata {
    //         self.vector_metadata.insert(id, meta);
    //     }

    //     Ok(())
    // }


    /// Adds or updates a vector in the index.
    /// If the ID already exists, the vector and metadata are overwritten.
    /// Stale graph nodes are left in place but excluded from all queries.
    fn add_point_internal(&mut self, id: String, vector: Vec<f32>, metadata: Option<HashMap<String, String>>) -> PyResult<()> {
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
        self.hnsw.insert((stored_vec.as_slice(), internal_id));

        self.id_map.insert(id.clone(), internal_id);
        self.rev_map.insert(internal_id, id.clone());

        if let Some(meta) = metadata {
            self.vector_metadata.insert(id, meta);
        }

        Ok(())
    }




    /// Search for the k-nearest neighbors of a vector
    /// Returns actual Python dictionaries which most common for ML workflows
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false))]
    pub fn query(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        filter: Option<HashMap<String, String>>,
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
        let results = self.hnsw.search(&vector, top_k, ef);

        let mut output = Vec::with_capacity(results.len());

        for neighbor in results {
            let score = neighbor.distance;
            let internal_id = neighbor.get_origin_id();

            if let Some(ext_id) = self.rev_map.get(&internal_id) {
                // Apply optional metadata filter
                if let Some(ref filter_map) = filter {
                    if let Some(meta) = self.vector_metadata.get(ext_id) {
                        let matches = filter_map.iter().all(|(k, v)| meta.get(k) == Some(v));
                        if !matches {
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
                dict.set_item("metadata", metadata)?;

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





    /// Get vector by ID
    pub fn get_vector(&self, id: String) -> Option<Vec<f32>> {
        self.vectors.get(&id).cloned()
    }

    /// Get metadata by ID
    pub fn get_vector_metadata(&self, id: String) -> Option<HashMap<String, String>> {
        self.vector_metadata.get(&id).cloned()
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
    pub fn list(&self, number: usize) -> Vec<(String, Option<HashMap<String, String>>)> {
        self.vectors
            .iter()
            .take(number)
            .map(|(id, _vec)| {
                let meta = self.vector_metadata.get(id).cloned();
                (id.clone(), meta)
            })
            .collect()
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
    pub fn remove_point(&mut self, id: String) -> PyResult<bool> {
        if let Some(internal_id) = self.id_map.remove(&id) {
            self.vectors.remove(&id);
            self.vector_metadata.remove(&id);
            self.rev_map.remove(&internal_id);
            // Note: HNSW doesn't support removal, so the graph still contains the point
            // but it won't be accessible via our mappings
            Ok(true)
        } else {
            Ok(false)
        }
    }
} 
