use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock, Arc};
use hnsw_rs::prelude::{Hnsw, DistCosine, DistL2, DistL1, Distance};
use serde_json::Value;
use rayon::prelude::*;

// Import PQ module
use crate::pq::PQ;

// Quantization configuration structure
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub subvectors: usize,
    pub bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,
}

/// Custom distance function for Product Quantization using ADC
#[derive(Clone)]
pub struct DistPQ {
    /// Reference to the PQ instance for accessing centroids
    pq: Arc<PQ>,
    /// Pre-computed ADC lookup table for the current query
    /// Thread-safe storage since multiple threads may search simultaneously
    lut: Arc<RwLock<Option<Vec<Vec<f32>>>>>,
}

impl DistPQ {
    pub fn new(pq: Arc<PQ>) -> Self {
        DistPQ {
            pq,
            lut: Arc::new(RwLock::new(None)),
        }
    }
    
    pub fn set_query_lut(&self, query: &[f32]) -> Result<(), String> {
        if !self.pq.is_trained() {
            return Err("PQ must be trained before ADC computation".to_string());
        }
        
        let lut = self.pq.compute_adc_lut(query)?;
        {
            let mut lut_guard = self.lut.write().unwrap();
            *lut_guard = Some(lut);
        }
        Ok(())
    }
    
    pub fn clear_lut(&self) {
        let mut lut_guard = self.lut.write().unwrap();
        *lut_guard = None;
    }
}



impl Distance<Vec<u8>> for DistPQ {
    /// Compute distance between query (via LUT) and stored PQ codes
    /// The first parameter `_a` is ignored since we use the pre-computed LUT
    /// The second parameter `b` contains the PQ codes for the stored vector
    fn eval(&self, _a: &[Vec<u8>], b: &[Vec<u8>]) -> f32 {
        let lut_guard = self.lut.read().unwrap();
        match lut_guard.as_ref() {
            Some(lut) => {
                // Use the first vector in the batch for distance computation
                if let Some(codes) = b.first() {
                    self.pq.adc_distance(codes, lut).unwrap_or(f32::INFINITY)
                } else {
                    f32::INFINITY
                }
            }
            None => {
                // Fallback: this shouldn't happen in normal operation
                f32::INFINITY
            }
        }
    }
}




// Enhanced DistanceType enum to support PQ variants
enum DistanceType {
    // Raw vector variants
    Cosine(Hnsw<'static, f32, DistCosine>),
    L2(Hnsw<'static, f32, DistL2>),
    L1(Hnsw<'static, f32, DistL1>),
    
    // PQ variants
    CosinePQ(Hnsw<'static, Vec<u8>, DistPQ>),
    L2PQ(Hnsw<'static, Vec<u8>, DistPQ>),
    L1PQ(Hnsw<'static, Vec<u8>, DistPQ>),
}

impl DistanceType {
    fn new_raw(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        match space {
            "cosine" => DistanceType::Cosine(Hnsw::new(m, expected_size, max_layer, ef_construction, DistCosine {})),
            "l2" => DistanceType::L2(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL2 {})),
            "l1" => DistanceType::L1(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL1 {})),
            _ => panic!("Unsupported space: {}", space),
        }
    }
    

    fn new_pq(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
        pq: Arc<PQ>,
    ) -> Self {
        match space {
            "cosine" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::CosinePQ(Hnsw::new(m, expected_size, max_layer, ef_construction, dist_pq))
            }
            "l2" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::L2PQ(Hnsw::new(m, expected_size, max_layer, ef_construction, dist_pq))
            }
            "l1" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::L1PQ(Hnsw::new(m, expected_size, max_layer, ef_construction, dist_pq))
            }
            _ => panic!("Unsupported space: {}", space),
        }
    }


    
    fn set_query_lut(&self, query: &[f32]) -> Result<(), String> {
        match self {
            DistanceType::CosinePQ(hnsw) => hnsw.get_distance().set_query_lut(query),
            DistanceType::L2PQ(hnsw) => hnsw.get_distance().set_query_lut(query),
            DistanceType::L1PQ(hnsw) => hnsw.get_distance().set_query_lut(query),
            _ => Ok(()), // No-op for raw variants
        }
    }
    
    fn clear_lut(&self) {
        match self {
            DistanceType::CosinePQ(hnsw) => hnsw.get_distance().clear_lut(),
            DistanceType::L2PQ(hnsw) => hnsw.get_distance().clear_lut(),
            DistanceType::L1PQ(hnsw) => hnsw.get_distance().clear_lut(),
            _ => {}, // No-op for raw variants
        }
    }
    
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<hnsw_rs::prelude::Neighbour>, String> {
        match self {
            // Raw vector search
            DistanceType::Cosine(hnsw) => Ok(hnsw.search(query, k, ef)),
            DistanceType::L2(hnsw) => Ok(hnsw.search(query, k, ef)),
            DistanceType::L1(hnsw) => Ok(hnsw.search(query, k, ef)),
            
            // PQ-based search with ADC
            DistanceType::CosinePQ(hnsw) | 
            DistanceType::L2PQ(hnsw) | 
            DistanceType::L1PQ(hnsw) => {
                // Set the query LUT for ADC computation
                self.set_query_lut(query)?;
                
                // Create dummy query vector for HNSW traversal
                // The actual distance is computed via ADC using the stored LUT
                let dummy_query = vec![vec![0u8; self.get_code_size()]];
                
                // Perform search
                let results = hnsw.search(&dummy_query, k, ef);
                
                // Clear LUT after search for memory efficiency
                self.clear_lut();
                
                Ok(results)
            }
        }
    }
    
    fn get_code_size(&self) -> usize {
        match self {
            DistanceType::CosinePQ(hnsw) => hnsw.get_distance().pq.subvectors,
            DistanceType::L2PQ(hnsw) => hnsw.get_distance().pq.subvectors,
            DistanceType::L1PQ(hnsw) => hnsw.get_distance().pq.subvectors,
            _ => 0,
        }
    }
    
    fn is_quantized(&self) -> bool {
        matches!(self, 
            DistanceType::CosinePQ(_) | 
            DistanceType::L2PQ(_) | 
            DistanceType::L1PQ(_)
        )
    }
    
    fn insert(&mut self, vector: &[f32], id: usize) {
        match self {
            DistanceType::Cosine(hnsw) => hnsw.insert((vector, id)),
            DistanceType::L2(hnsw) => hnsw.insert((vector, id)),
            DistanceType::L1(hnsw) => hnsw.insert((vector, id)),
            _ => panic!("Cannot insert raw vectors into PQ index"),
        }
    }
    
    /// Insert PQ codes into the index
    #[allow(dead_code)]
    fn insert_pq_codes(&mut self, codes: &[u8], id: usize) {
        match self {
            DistanceType::CosinePQ(hnsw) => {
                let wrapped_codes = vec![codes.to_vec()];
                hnsw.insert((&wrapped_codes, id));
            },
            DistanceType::L2PQ(hnsw) => {
                let wrapped_codes = vec![codes.to_vec()];
                hnsw.insert((&wrapped_codes, id));
            },
            DistanceType::L1PQ(hnsw) => {
                let wrapped_codes = vec![codes.to_vec()];
                hnsw.insert((&wrapped_codes, id));
            },
            _ => panic!("Cannot insert PQ codes into raw index"),
        }
    }



    
    #[allow(dead_code)]
    fn insert_batch(&mut self, data: &[(&Vec<f32>, usize)]) {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        if data.len() >= threshold {
            match self {
                DistanceType::Cosine(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L2(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L1(hnsw) => hnsw.parallel_insert(data),
                _ => panic!("Cannot batch insert raw vectors into PQ index"),
            }
        } else {
            for (vector, id) in data {
                self.insert(vector.as_slice(), *id);
            }
        }
    }
    
    // fn insert_batch_pq(&mut self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
    //     let num_threads = rayon::current_num_threads();
    //     let threshold = 1000 * num_threads;

    //     match self {
    //         DistanceType::CosinePQ(hnsw) |
    //         DistanceType::L2PQ(hnsw) |
    //         DistanceType::L1PQ(hnsw) => {
    //             if data.len() >= threshold {
    //                 hnsw.parallel_insert(data);
    //             } else {
    //                 for (codes, id) in data {
    //                     hnsw.insert((codes.as_slice(), *id));
    //                 }
    //             }
    //             Ok(())
    //         }
    //         _ => Err("Cannot insert PQ codes into raw HNSW index".to_string()),
    //     }
    // }


    fn insert_batch_pq(&mut self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        match self {
            DistanceType::CosinePQ(hnsw) |
            DistanceType::L2PQ(hnsw) |
            DistanceType::L1PQ(hnsw) => {
                // Convert data format to match expected Vec<Vec<u8>>
                let converted_data: Vec<(Vec<Vec<u8>>, usize)> = data.iter()
                    .map(|(codes, id)| (vec![(*codes).clone()], *id))
                    .collect();
            
                let ref_data: Vec<(&Vec<Vec<u8>>, usize)> = converted_data.iter()
                    .map(|(codes, id)| (codes, *id))
                    .collect();

                if data.len() >= threshold {
                    hnsw.parallel_insert(&ref_data);
                } else {
                    for (codes, id) in &ref_data {
                        hnsw.insert((codes.as_slice(), *id));
                    }
                }
                Ok(())
            }
            _ => Err("Cannot insert PQ codes into raw HNSW index".to_string()),
        }
    }


}






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
    pub vector_shape: Option<(usize, usize)>,
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

    // Quantization configuration and PQ instance
    quantization_config: Option<QuantizationConfig>,
    pq: Option<Arc<PQ>>,
    pq_codes: RwLock<HashMap<String, Vec<u8>>>, // PQ codes storage

    // Index-level metadata (simple, infrequently accessed)
    metadata: Mutex<HashMap<String, String>>,

    // Thread-safe vector store with RwLock for concurrent reads
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    vector_metadata: RwLock<HashMap<String, HashMap<String, Value>>>,
    id_map: RwLock<HashMap<String, usize>>,
    rev_map: RwLock<HashMap<usize, String>>,
    
    // Mutex for write-only fields
    id_counter: Mutex<usize>,
    vector_count: Mutex<usize>, // Track total vectors for training trigger
    
    // Mutex for HNSW (not thread-safe for concurrent reads)
    hnsw: Mutex<DistanceType>,
}



#[pymethods]
impl HNSWIndex {
    #[new]
    fn new(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
        quantization_config: Option<&Bound<PyDict>>,
    ) -> PyResult<Self> {
        // Validation of parameters
        if dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("dim must be positive"));
        }
        if ef_construction == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("ef_construction must be positive"));
        }
        if expected_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("expected_size must be positive"));
        }
        if m > 256 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("m must be less than or equal to 256"));
        }

        let space_normalized = space.to_lowercase();
        
        // Extract quantization configuration
        let (quantization_params, pq_instance) = if let Some(config) = quantization_config {
            let qtype = config.get_item("type")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'type' in quantization_config"))?
                .extract::<String>()?;
            
            if qtype != "pq" {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported quantization type: '{}'. Only 'pq' is currently supported.", qtype)
                ));
            }
            
            // Extract PQ parameters
            let subvectors = config.get_item("subvectors")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'subvectors' in quantization_config"))?
                .extract::<usize>()?;
            
            let bits = config.get_item("bits")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'bits' in quantization_config"))?
                .extract::<usize>()?;
            
            let training_size = config.get_item("training_size")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'training_size' in quantization_config"))?
                .extract::<usize>()?;
            
            let max_training_vectors = config.get_item("max_training_vectors")?
                .map(|v| v.extract::<usize>())
                .transpose()?;
            
            // Validate PQ parameters
            if dim % subvectors != 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("subvectors ({}) must divide dimension ({}) evenly", subvectors, dim)
                ));
            }
            
            if bits < 1 || bits > 8 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("bits must be between 1 and 8, got {}", bits)
                ));
            }
            
            if training_size < 1000 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("training_size must be at least 1000, got {}", training_size)
                ));
            }
            
            let config = QuantizationConfig {
                subvectors,
                bits,
                training_size,
                max_training_vectors,
            };
            
            // Create PQ instance
            let pq = Arc::new(PQ::new(dim, subvectors, bits, training_size, max_training_vectors));
            
            (Some(config), Some(pq))
        } else {
            (None, None)
        };

        let max_layer = (expected_size as f32).log2().ceil() as usize;


        // Create initial raw HNSW index (will be rebuilt as PQ after training)
        let hnsw = DistanceType::new_raw(&space_normalized, m, expected_size, max_layer, ef_construction);


        // Initialize all fields with proper thread-safe wrappers
        Ok(HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            quantization_config: quantization_params,
            pq: pq_instance,
            pq_codes: RwLock::new(HashMap::new()),
            metadata: Mutex::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: Mutex::new(hnsw),
        })
    }

    /// Get quantization configuration and status
    pub fn get_quantization_info(&self) -> Option<PyObject> {
        Python::with_gil(|py| {
            if let Some(config) = &self.quantization_config {
                let dict = PyDict::new(py);
                dict.set_item("type", "pq").ok()?;
                dict.set_item("subvectors", config.subvectors).ok()?;
                dict.set_item("bits", config.bits).ok()?;
                dict.set_item("training_size", config.training_size).ok()?;
                
                if let Some(max_training) = config.max_training_vectors {
                    dict.set_item("max_training_vectors", max_training).ok()?;
                }
                
                if let Some(pq) = &self.pq {
                    dict.set_item("is_trained", pq.is_trained()).ok()?;
                    
                    // Use enhanced PQ methods
                    let (memory_mb, total_centroids) = pq.get_memory_stats();
                    dict.set_item("memory_mb", memory_mb).ok()?;
                    dict.set_item("total_centroids", total_centroids).ok()?;
                    
                    // Calculate compression ratio using cached values
                    let original_bytes = pq.dim * 4; // f32
                    let compressed_bytes = pq.subvectors; // u8 per subvector
                    let compression_ratio = original_bytes as f64 / compressed_bytes as f64;
                    dict.set_item("compression_ratio", compression_ratio).ok()?;
                }
                
                Some(dict.into())
            } else {
                None
            }
        })
    }

    /// Check if quantization is enabled
    pub fn has_quantization(&self) -> bool {
        self.quantization_config.is_some()
    }

    /// Get current vector count (for monitoring training trigger)
    pub fn get_vector_count(&self) -> usize {
        *self.vector_count.lock().unwrap()
    }

    /// Get next available internal ID
    fn get_next_id(&self) -> usize {
        let mut counter = self.id_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Rebuild the HNSW index to use PQ codes after training is complete
    pub fn rebuild_with_quantization(&mut self) -> PyResult<bool> {
        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => return Ok(false),
        };

        // Get all current vectors for quantization
        let vectors = self.vectors.read().unwrap();
        if vectors.is_empty() {
            return Ok(false);
        }

        // Quantize all existing vectors
        let vector_refs: Vec<&[f32]> = vectors.values().map(|v| v.as_slice()).collect();
        let quantized_codes = pq.quantize_batch(&vector_refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to quantize vectors: {}", e)
            ))?;

        // Create new PQ-based HNSW index
        let max_layer = (self.expected_size as f32).log2().ceil() as usize;
        let new_hnsw = DistanceType::new_pq(
            &self.space, 
            self.m, 
            self.expected_size, 
            max_layer, 
            self.ef_construction, 
            pq.clone()
        );

        // Store quantized codes
        {
            let mut pq_codes = self.pq_codes.write().unwrap();
            pq_codes.clear(); // Clear any existing codes
            
            for (i, (id, _)) in vectors.iter().enumerate() {
                if i < quantized_codes.len() {
                    pq_codes.insert(id.clone(), quantized_codes[i].clone());
                }
            }
        }

        // Replace the HNSW index
        {
            let mut hnsw_guard = self.hnsw.lock().unwrap();
            *hnsw_guard = new_hnsw;
        }

        // Insert quantized vectors into new index
        let pq_codes = self.pq_codes.read().unwrap();
        let id_map = self.id_map.read().unwrap();
        let mut batch_data = Vec::new();
        
        for (id, codes) in pq_codes.iter() {
            if let Some(&internal_id) = id_map.get(id) {
                batch_data.push((codes, internal_id));
            }
        }

        if !batch_data.is_empty() {
            let mut hnsw_guard = self.hnsw.lock().unwrap();
            hnsw_guard.insert_batch_pq(&batch_data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to insert quantized vectors: {}", e)
                ))?;
        }

        Ok(true)
    }


    /// Check if the index is using quantized search
    pub fn is_quantized(&self) -> bool {
        if let Some(pq) = &self.pq {
            if pq.is_trained() {
                let hnsw_guard = self.hnsw.lock().unwrap();
                return hnsw_guard.is_quantized();
            }
        }
        false
    }

    /// Check if quantization can be used (PQ is trained)
    pub fn can_use_quantization(&self) -> bool {
        if let Some(pq) = &self.pq {
            pq.is_trained()
        } else {
            false
        }
    }



    /// Add vectors to the index (placeholder for step 4)
    pub fn add(&mut self, data: Bound<PyAny>) -> PyResult<AddResult> {
        // Basic data type checking
        if data.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data cannot be None"
            ));
        }
        
        // Check if we should trigger training
        if let Some(config) = &self.quantization_config {
            if let Some(pq) = &self.pq {
                let current_count = self.get_vector_count();
                if current_count >= config.training_size && !pq.is_trained() {
                    println!("Training threshold reached: {} >= {}", current_count, config.training_size);
                    // TODO: Implement training logic in step 4
                }
            }
        }
        
        // TODO: Implement full add logic in step 4
        // For now, return placeholder
        Ok(AddResult {
            total_inserted: 0,
            total_errors: 1,
            errors: vec!["Add method implementation coming in step 4".to_string()],
            vector_shape: Some((0, self.dim)),
        })
    }




    /// Enhanced search method with automatic ADC usage
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false))]
    pub fn search(
        &self,
        py: Python<'_>,
        vector: Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
    ) -> PyResult<PyObject> {
        let ef = ef_search.unwrap_or_else(|| std::cmp::max(2 * top_k, 100));
        let filter_conditions = filter.map(|f| self.python_dict_to_value_map(f)).transpose()?;

        // Detect batch vs single query with comprehensive input support
        if let Ok(list_vec) = vector.extract::<Vec<Vec<f32>>>() {
            // Format: List of vectors [[0.1, 0.2], [0.3, 0.4]]
            let results = self.batch_search_internal(&list_vec, filter_conditions.as_ref(), top_k, ef, return_vector, py)?;
            Ok(PyList::new(py, results)?.into())
        } else if let Ok(np_array) = vector.downcast::<PyArray2<f32>>() {
            // Format: NumPy 2D array (N, dims)
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            if shape.len() != 2 || shape[1] != self.dim {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "NumPy array must have shape (N, {}), got {:?}", self.dim, shape
                )));
            }

            let flat = readonly.as_slice()?;
            let batch: Vec<Vec<f32>> = flat.chunks(self.dim).map(|chunk| chunk.to_vec()).collect();
            let results = self.batch_search_internal(&batch, filter_conditions.as_ref(), top_k, ef, return_vector, py)?;
            Ok(PyList::new(py, results)?.into())
        } else {
            // Single vector path - enhanced with NumPy 1D support
            let single_vec = if let Ok(array1d) = vector.downcast::<PyArray1<f32>>() {
                array1d.readonly().as_slice()?.to_vec()
            } else {
                vector.extract::<Vec<f32>>()?
            };

            if single_vec.len() != self.dim {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Search vector dimension mismatch: expected {}, got {}",
                    self.dim, single_vec.len()
                )));
            }

            
            let search_results = py.allow_threads(|| {
                // Check if we should use quantized search
                let use_quantized = self.is_quantized();

                let hnsw_results = {
                    let hnsw_guard = self.hnsw.lock().unwrap();

                    if use_quantized {
                        // Use ADC search for quantized index
                        hnsw_guard.search(&single_vec, top_k, ef)
                            .unwrap_or_else(|e| {
                                eprintln!("ADC search error: {}", e);
                                Vec::new()
                            })
                    } else {
                        // Use raw vector search
                        match hnsw_guard.search(&single_vec, top_k, ef) {
                            Ok(results) => results,
                            Err(e) => {
                                eprintln!("Raw search error: {}", e);
                                Vec::new()
                            }
                        }
                    }
                };


                // Process results with enhanced vector retrieval
                let vectors = self.vectors.read().unwrap();
                let pq_codes = self.pq_codes.read().unwrap();
                let vector_metadata = self.vector_metadata.read().unwrap();
                let rev_map = self.rev_map.read().unwrap();

                let mut results = Vec::with_capacity(hnsw_results.len());
                let has_filter = filter_conditions.is_some();

                for neighbor in hnsw_results {
                    let score = neighbor.distance;
                    let internal_id = neighbor.get_origin_id();

                    if let Some(ext_id) = rev_map.get(&internal_id) {
                        if has_filter {
                            if let Some(meta) = vector_metadata.get(ext_id) {
                                let filter_conds = filter_conditions.as_ref().unwrap();
                                if !self.matches_filter(meta, filter_conds).unwrap_or(false) {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let metadata = vector_metadata.get(ext_id).cloned().unwrap_or_default();
                        let vector_data = if return_vector {
                            // Try raw vector first, then PQ reconstruction
                            vectors.get(ext_id).cloned()
                                .or_else(|| {
                                    if let (Some(pq), Some(codes)) = (&self.pq, pq_codes.get(ext_id)) {
                                        pq.reconstruct(codes).ok()
                                    } else {
                                        None
                                    }
                                })
                        } else {
                            None
                        };

                        results.push((ext_id.clone(), score, metadata, vector_data));
                    }
                }

                results
            });

            // Convert to Python objects
            let mut output: Vec<Py<PyDict>> = Vec::with_capacity(search_results.len());
            for (id, score, metadata, vector_data) in search_results {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;
                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }
                output.push(dict.into());
            }

            Ok(PyList::new(py, output)?.into())
        }
    }



    /// Get records by ID(s) with PQ reconstruction support
    #[pyo3(signature = (input, return_vector = true))]
    pub fn get_records(&self, py: Python<'_>, input: &Bound<PyAny>, return_vector: bool) -> PyResult<Vec<Py<PyDict>>> {
        let ids: Vec<String> = if let Ok(id_str) = input.extract::<String>() {
            vec![id_str]
        } else if let Ok(id_list) = input.extract::<Vec<String>>() {
            id_list
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a string or a list of strings for ID(s)",
            ));
        };

        let mut records = Vec::with_capacity(ids.len());
        
        // Use read locks for concurrent access
        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        for id in ids {
            if let Some(vector) = vectors.get(&id) {
                let metadata = vector_metadata.get(&id).cloned().unwrap_or_default();

                let dict = PyDict::new(py);
                dict.set_item("id", id.clone())?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    // Try raw vector first, then PQ reconstruction
                    let vector_data = if !vector.is_empty() {
                        vector.clone()
                    } else if let (Some(pq), Some(codes)) = (&self.pq, pq_codes.get(&id)) {
                        pq.reconstruct(codes).unwrap_or_else(|_| vector.clone())
                    } else {
                        vector.clone()
                    };
                    
                    dict.set_item("vector", vector_data)?;
                }

                records.push(dict.into());
            }
        }

        Ok(records)
    }







    /// Get comprehensive statistics with quantization info
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        
        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        
        stats.insert("total_vectors".to_string(), vectors.len().to_string());
        stats.insert("dimension".to_string(), self.dim.to_string());
        stats.insert("space".to_string(), self.space.clone());
        stats.insert("m".to_string(), self.m.to_string());
        stats.insert("ef_construction".to_string(), self.ef_construction.to_string());
        stats.insert("expected_size".to_string(), self.expected_size.to_string());
        stats.insert("index_type".to_string(), "HNSW".to_string());
        
        // Add quantization info
        if let Some(config) = &self.quantization_config {
            stats.insert("quantization_type".to_string(), "pq".to_string());
            stats.insert("quantization_subvectors".to_string(), config.subvectors.to_string());
            stats.insert("quantization_bits".to_string(), config.bits.to_string());
            stats.insert("quantization_training_size".to_string(), config.training_size.to_string());
            
            if let Some(max_training) = config.max_training_vectors {
                stats.insert("quantization_max_training_vectors".to_string(), max_training.to_string());
            }
            
            if let Some(pq) = &self.pq {
                stats.insert("quantization_trained".to_string(), pq.is_trained().to_string());
                let (memory_mb, _) = pq.get_memory_stats();
                stats.insert("quantization_memory_mb".to_string(), format!("{:.2}", memory_mb));
                
                // Use cached values for efficiency
                let compression_ratio = (pq.dim * 4) as f64 / pq.subvectors as f64;
                stats.insert("quantization_compression_ratio".to_string(), format!("{:.1}x", compression_ratio));
                
                // Add quantization status
                stats.insert("quantization_active".to_string(), self.is_quantized().to_string());
                
                // Add storage info
                stats.insert("quantized_vectors".to_string(), pq_codes.len().to_string());
            }
        } else {
            stats.insert("quantization_type".to_string(), "none".to_string());
        }
        
        stats.insert("thread_safety".to_string(), "RwLock+Mutex".to_string());
        
        stats
    }




    /// List the first number of records in the index (ID and metadata)
    #[pyo3(signature = (number=10))]
    pub fn list(&self, py: Python<'_>, number: usize) -> PyResult<Vec<(String, PyObject)>> {
        let vectors = self.vectors.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        
        let mut results = Vec::new();
        for (id, _vec) in vectors.iter().take(number) {
            let metadata = vector_metadata.get(id).cloned().unwrap_or_default();
            let py_metadata = self.value_map_to_python(&metadata, py)?;
            results.push((id.clone(), py_metadata));
        }
        Ok(results)
    }

    /// Check if vector exists
    pub fn contains(&self, id: String) -> bool {
        let vectors = self.vectors.read().unwrap();
        vectors.contains_key(&id)
    }

    /// Add index-level metadata
    pub fn add_metadata(&mut self, metadata: HashMap<String, String>) {
        let mut meta_lock = self.metadata.lock().unwrap();
        for (key, value) in metadata {
            meta_lock.insert(key, value);
        }
    }

    /// Get index-level metadata value
    pub fn get_metadata(&self, key: String) -> Option<String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.get(&key).cloned()
    }

    /// Get all index-level metadata
    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.clone()
    }

    /// Get a human-readable info string
    pub fn info(&self) -> String {
        let vectors = self.vectors.read().unwrap();
        let base_info = format!(
            "HNSWIndex(dim={}, space={}, m={}, ef_construction={}, expected_size={}, vectors={}",
            self.dim, self.space, self.m, self.ef_construction, self.expected_size, vectors.len()
        );
        
        if let Some(config) = &self.quantization_config {
            let trained_status = self.pq.as_ref()
                .map(|pq| if pq.is_trained() { "trained" } else { "untrained" })
                .unwrap_or("unknown");

            let active_status = if self.is_quantized() { "active" } else { "inactive" };
            
            // Use cached compression ratio calculation
            let compression_info = self.pq.as_ref()
                .map(|pq| format!("{}x", (pq.dim * 4) / pq.subvectors))
                .unwrap_or_else(|| "unknown".to_string());
            
            format!(
                "{}, quantization=pq(subvectors={}, bits={}, {}, {}, compression={}))",
                base_info, config.subvectors, config.bits, trained_status, active_status, compression_info
            )
        } else {
            format!("{}, quantization=none)", base_info)
        }
    }

    /// Remove vector by ID
    pub fn remove_point(&mut self, id: String) -> PyResult<bool> {
        let mut vectors = self.vectors.write().unwrap();
        let mut vector_metadata = self.vector_metadata.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();
        let mut rev_map = self.rev_map.write().unwrap();
        
        if let Some(internal_id) = id_map.remove(&id) {
            vectors.remove(&id);
            vector_metadata.remove(&id);
            rev_map.remove(&internal_id);
            // Note: HNSW doesn't support removal, so the graph still contains the point
            // but it won't be accessible via the mappings
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get performance characteristics and limitations
    pub fn get_performance_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("search_speedup_expected".to_string(), "1.2x-2x".to_string());
        info.insert("insertion_speedup_expected".to_string(), "4x-8x_large_batches".to_string());
        info.insert("search_bottleneck".to_string(), "hnsw_mutex_serialization".to_string());
        info.insert("insertion_bottleneck".to_string(), "hnsw_mutex_for_large_batches".to_string());
        info.insert("benefits".to_string(), "gil_release_concurrent_metadata_processing_parallel_insert".to_string());
        info.insert("limitation".to_string(), "parallel_insert_threshold_1000x_threads".to_string());
        info.insert("recommendation".to_string(), "excellent_for_large_batch_workloads".to_string());
        
        // Add quantization performance info
        if let Some(config) = &self.quantization_config {
            let original_bytes = self.dim * 4; // f32
            let compressed_bytes = config.subvectors; // u8 per subvector
            let compression_ratio = original_bytes as f64 / compressed_bytes as f64;
            
            info.insert("quantization_compression".to_string(), format!("{:.1}x", compression_ratio));
            info.insert("quantization_memory_savings".to_string(), format!("{:.1}%", (1.0 - 1.0/compression_ratio) * 100.0));
            info.insert("quantization_accuracy_impact".to_string(), "slight_recall_reduction".to_string());
        }
        
        info
    }

    /// Concurrent benchmark for search performance
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_concurrent_reads(&self, query_count: usize, max_threads: Option<usize>) -> PyResult<HashMap<String, f64>> {
        use std::time::Instant;
        
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        let mut results = HashMap::new();

        // Sequential benchmark
        let start = Instant::now();
        for query in &queries {
            let _ = self.raw_search_no_gil(query);
        }
        let sequential_time = start.elapsed().as_secs_f64();
        results.insert("sequential_time".to_string(), sequential_time);
        results.insert("sequential_qps".to_string(), queries.len() as f64 / sequential_time);
        
        // Parallel benchmark
        let available_threads = rayon::current_num_threads();
        let num_threads = max_threads.unwrap_or(available_threads).min(available_threads);

        let start = Instant::now();
        let _: Vec<_> = queries
            .par_iter()
            .map(|query| {
                self.raw_search_no_gil(query)
            })
            .collect();

        let parallel_time = start.elapsed().as_secs_f64();
        results.insert("parallel_time".to_string(), parallel_time);
        results.insert("parallel_qps".to_string(), queries.len() as f64 / parallel_time);
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);
        
        Ok(results)
    }

    /// Raw performance benchmark
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_raw_concurrent_performance(&self, query_count: usize, max_threads: Option<usize>) -> HashMap<String, f64> {
        use std::time::Instant;
        
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        let mut results = HashMap::new();
        
        // Sequential benchmark
        let start = Instant::now();
        for query in &queries {
            let _ = self.raw_search_no_gil(query);
        }
        let sequential_time = start.elapsed().as_secs_f64();
        
        // Parallel benchmark
        let available_threads = rayon::current_num_threads();
        let num_threads = max_threads.unwrap_or(available_threads).min(available_threads);
        let chunk_size = (queries.len() + num_threads - 1) / num_threads;
        
        let start = Instant::now();
        let total_processed: usize = queries
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_count = 0;
                for query in chunk {
                    let _ = self.raw_search_no_gil(query);
                    local_count += 1;
                }
                local_count
            })
            .sum();
        
        let parallel_time = start.elapsed().as_secs_f64();
        
        results.insert("sequential_time_sec".to_string(), sequential_time);
        results.insert("parallel_time_sec".to_string(), parallel_time);
        results.insert("sequential_qps".to_string(), query_count as f64 / sequential_time);
        results.insert("parallel_qps".to_string(), total_processed as f64 / parallel_time);
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);
        results.insert("note".to_string(), "limited_by_hnsw_mutex".parse().unwrap_or(0.0));
        
        results
    }
}

impl HNSWIndex {
    /// Raw search without Python objects (for benchmarking)
    fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        // HNSW search with locking
        let hnsw_results = {
            let hnsw_guard = self.hnsw.lock().unwrap();
            //hnsw_guard.search(query, 10, 100)
            hnsw_guard.search(query, 10, 100).unwrap_or_else(|_| Vec::new())
        }; // Lock released immediately
        
        // Concurrent read access to ID mapping
        let rev_map = self.rev_map.read().unwrap();
        
        hnsw_results
            .into_iter()
            .filter_map(|neighbor| {
                rev_map.get(&neighbor.get_origin_id())
                    .map(|id| (id.clone(), neighbor.distance))
            })
            .collect()
    }

    // Helper methods for data conversion and filtering
    fn python_dict_to_value_map(&self, py_dict: &Bound<PyDict>) -> PyResult<HashMap<String, Value>> {
        let mut map = HashMap::new();
        
        for (key, value) in py_dict.iter() {
            let string_key = key.extract::<String>()?;
            let json_value = self.python_object_to_value(&value)?;
            map.insert(string_key, json_value);
        }
        
        Ok(map)
    }

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
            Ok(Value::String(py_obj.to_string()))
        }
    }

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
            Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {
                Ok(field_value == condition)
            },
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
                "gt" => self.compare_values(field_value, target_value, |a, b| a > b)?,
                "gte" => self.compare_values(field_value, target_value, |a, b| a >= b)?,
                "lt" => self.compare_values(field_value, target_value, |a, b| a < b)?,
                "lte" => self.compare_values(field_value, target_value, |a, b| a <= b)?,
                "contains" => self.value_contains(field_value, target_value)?,
                "startswith" => self.value_starts_with(field_value, target_value)?,
                "endswith" => self.value_ends_with(field_value, target_value)?,
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

    fn value_map_to_python(&self, value_map: &HashMap<String, Value>, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for (key, value) in value_map {
            let py_value = self.value_to_python_object(value, py)?;
            dict.set_item(key, py_value)?;
        }
        
        Ok(dict.into_pyobject(py)?.to_owned().unbind().into_any())
    }

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

    /// Internal batch search method for multiple query vectors
    fn batch_search_internal(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        top_k: usize,
        ef: usize,
        return_vector: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        // Validate all vectors have correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dim {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Vector {}: dimension mismatch: expected {}, got {}", 
                           i, self.dim, vector.len())));
            }
        }

        // Choose strategy based on batch size
        if vectors.len() <= 5 {
            self.batch_search_sequential(vectors, filter_conditions, top_k, ef, return_vector, py)
        } else {
            self.batch_search_parallel(vectors, filter_conditions, top_k, ef, return_vector, py)
        }
    }

    /// Sequential batch processing (for small batches)
    fn batch_search_sequential(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        top_k: usize,
        ef: usize,
        return_vector: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let rust_results = py.allow_threads(|| {
            let hnsw_guard = self.hnsw.lock().unwrap();
            let vector_store = self.vectors.read().unwrap();
            let metadata_store = self.vector_metadata.read().unwrap();
            let rev_map = self.rev_map.read().unwrap();
            
            let mut all_results = Vec::with_capacity(vectors.len());

            for vector in vectors {
                let neighbors = hnsw_guard.search(vector, top_k, ef).unwrap_or_else(|_| Vec::new());
                let mut query_results = Vec::with_capacity(neighbors.len());

                for neighbor in neighbors {
                    let internal_id = neighbor.get_origin_id();

                    if let Some(ext_id) = rev_map.get(&internal_id) {
                        // Apply filter if specified
                        if let Some(filter_conds) = filter_conditions {
                            if let Some(meta) = metadata_store.get(ext_id) {
                                if !self.matches_filter(meta, filter_conds).unwrap_or(false) {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let metadata = metadata_store.get(ext_id).cloned().unwrap_or_default();
                        let vector_data = if return_vector {
                            vector_store.get(ext_id).cloned()
                        } else {
                            None
                        };

                        query_results.push((ext_id.clone(), neighbor.distance, metadata, vector_data));
                    }
                }

                all_results.push(query_results);
            }
            
            all_results
        });

        // Convert to Python objects
        let mut output = Vec::with_capacity(rust_results.len());
        for batch_result in rust_results {
            let mut py_batch = Vec::with_capacity(batch_result.len());

            for (id, score, metadata, vector_data) in batch_result {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }

                py_batch.push(dict.into());
            }
            
            output.push(py_batch);
        }

        Ok(output)
    }

    /// Parallel batch processing (for larger batches)
    fn batch_search_parallel(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        top_k: usize,
        ef: usize,
        return_vector: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let rust_results = py.allow_threads(|| {
            let results: Vec<Vec<(String, f32, HashMap<String, Value>, Option<Vec<f32>>)>> = vectors
                .par_iter()
                .map(|vector| {
                    // Brief HNSW search (individual lock per query)
                    let neighbors = {
                        let hnsw_guard = self.hnsw.lock().unwrap();
                        hnsw_guard.search(vector, top_k, ef).unwrap_or_else(|_| Vec::new())
                    };

                    // Concurrent data lookup
                    let vector_store = self.vectors.read().unwrap();
                    let metadata_store = self.vector_metadata.read().unwrap();
                    let rev_map = self.rev_map.read().unwrap();

                    let mut query_results = Vec::with_capacity(neighbors.len());

                    for neighbor in neighbors {
                        let internal_id = neighbor.get_origin_id();

                        if let Some(ext_id) = rev_map.get(&internal_id) {
                            // Apply filter if specified
                            if let Some(filter_conds) = filter_conditions {
                                if let Some(meta) = metadata_store.get(ext_id) {
                                    if !self.matches_filter(meta, filter_conds).unwrap_or(false) {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            }

                            let metadata = metadata_store.get(ext_id).cloned().unwrap_or_default();
                            let vector_data = if return_vector {
                                vector_store.get(ext_id).cloned()
                            } else {
                                None
                            };

                            query_results.push((ext_id.clone(), neighbor.distance, metadata, vector_data));
                        }
                    }

                    query_results
                })
                .collect();
                
            results
        });

        // Convert to Python objects
        let mut output = Vec::with_capacity(rust_results.len());
        for batch_result in rust_results {
            let mut py_batch = Vec::with_capacity(batch_result.len());

            for (id, score, metadata, vector_data) in batch_result {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }

                py_batch.push(dict.into());
            }

            output.push(py_batch);
        }

        Ok(output)
    }
}
