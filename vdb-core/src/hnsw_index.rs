use pyo3::prelude::*;
use std::collections::HashMap;
use hnsw_rs::prelude::{Hnsw, DistCosine};

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

    /// Adds a vector to the index. Fails if the ID already exists.
    pub fn add_point(&mut self, id: String, vector: Vec<f32>, metadata: Option<HashMap<String, String>>) -> PyResult<()> {
        // Check vector dimension
        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim, vector.len()
            )));
        }

        // Check for duplicate ID
        if self.vectors.contains_key(&id) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Duplicate ID: '{}' already exists", id
            )));
        }

        // Assign internal index
        let internal_id = self.id_counter;
        self.id_counter += 1;

        // Store the vector and mappings
        self.vectors.insert(id.clone(), vector.clone());
        self.id_map.insert(id.clone(), internal_id);
        self.rev_map.insert(internal_id, id.clone());

        // Store metadata if provided
        if let Some(meta) = metadata {
            self.vector_metadata.insert(id.clone(), meta);
        }

        // Insert into HNSW using a reference to the stored vector
        let stored_vec = self.vectors.get(&id).unwrap();
        self.hnsw.insert((stored_vec.as_slice(), internal_id));

        Ok(())
    }

    /// Add multiple vectors in batch for better performance
    pub fn add_batch(&mut self, 
        points: Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>
    ) -> PyResult<HashMap<String, Vec<String>>> {
        let mut errors = Vec::new();
        let mut success_count = 0;

        for (id, vector, metadata) in points {
            match self.add_point(id.clone(), vector, metadata) {
                Ok(()) => success_count += 1,
                Err(e) => errors.push(format!("ID '{}': {}", id, e)),
            }
        }

        let mut result = HashMap::new();
        result.insert("success_count".to_string(), vec![success_count.to_string()]);
        result.insert("error_count".to_string(), vec![errors.len().to_string()]);
        result.insert("errors".to_string(), errors);

        Ok(result)
    }

    /// Query the index for the k-nearest neighbors of a vector
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None))]
    pub fn query(
        &self,
        vector: Vec<f32>,
        filter: Option<HashMap<String, String>>,
        top_k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<Vec<(String, f32)>> {
        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dim, vector.len()
            )));
        }

        // Get results from HNSW graph
        let ef = ef_search.unwrap_or_else(|| std::cmp::max(2 * top_k, 100));
        let results = self.hnsw.search(&vector, top_k, ef);

        let mut filtered_results = Vec::new();

        for neighbor in results {
            let score = neighbor.distance;
            let internal_id = neighbor.get_origin_id();
            
            if let Some(ext_id) = self.rev_map.get(&internal_id) {
                // Apply filtering if provided
                if let Some(ref filter_map) = filter {
                    if let Some(meta) = self.vector_metadata.get(ext_id) {
                        let mut matches = true;
                        for (k, v) in filter_map {
                            if meta.get(k) != Some(v) {
                                matches = false;
                                break;
                            }
                        }
                        if !matches {
                            continue;
                        }
                    } else {
                        // No metadata, but filter required - skip
                        continue;
                    }
                }
                filtered_results.push((ext_id.clone(), score));
            }
        }

        Ok(filtered_results)
    }

    /// Search with metadata included in results
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, include_metadata=false))]
    pub fn search_with_metadata(
        &self,
        vector: Vec<f32>,
        filter: Option<HashMap<String, String>>,
        top_k: usize,
        ef_search: Option<usize>,
        include_metadata: bool,
    ) -> PyResult<Vec<(String, f32, Option<HashMap<String, String>>)>> {
        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dim, vector.len()
            )));
        }

        let ef = ef_search.unwrap_or_else(|| std::cmp::max(2 * top_k, 100));
        let results = self.hnsw.search(&vector, top_k, ef);

        let mut filtered_results = Vec::new();

        for neighbor in results {
            let score = neighbor.distance;
            let internal_id = neighbor.get_origin_id();
            
            if let Some(ext_id) = self.rev_map.get(&internal_id) {
                // Apply filtering if provided
                if let Some(ref filter_map) = filter {
                    if let Some(meta) = self.vector_metadata.get(ext_id) {
                        let mut matches = true;
                        for (k, v) in filter_map {
                            if meta.get(k) != Some(v) {
                                matches = false;
                                break;
                            }
                        }
                        if !matches {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                let metadata = if include_metadata {
                    self.vector_metadata.get(ext_id).cloned()
                } else {
                    None
                };

                filtered_results.push((ext_id.clone(), score, metadata));
            }
        }

        Ok(filtered_results)
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

