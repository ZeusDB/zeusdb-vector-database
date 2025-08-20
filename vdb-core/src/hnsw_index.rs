use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock, Arc};
use std::sync::atomic::{AtomicBool, Ordering};
use hnsw_rs::prelude::{Hnsw, DistCosine, DistL2, DistL1, Distance};
use serde_json::Value;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use chrono::Utc;
use std::path::Path;
use hnsw_rs::api::AnnT;  // This provides the file_dump method
use std::time::Instant;

// ✅ ENTERPRISE: Structured logging imports
use tracing::{debug, info, warn, error, trace, instrument};

// Import PQ module
use crate::pq::PQ;

// ============================================================================
// VERSION COUNTER - MANUALLY INCREMENT TO TEST BUILD UPDATES
// ============================================================================

// 🔢 MANUAL VERSION COUNTER - Change this number after each code change
const CODE_VERSION_COUNTER: u32 = 1028;  // ← INCREMENT THIS MANUALLY
const CODE_VERSION_DESCRIPTION: &str = "Fixed overwrite bug - eliminates duplicate documents";

// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageMode {
    #[serde(rename = "quantized_only")]
    QuantizedOnly,

    #[serde(rename = "quantized_with_raw")]
    QuantizedWithRaw,
}

impl StorageMode {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s {
            "quantized_only" => Ok(StorageMode::QuantizedOnly),
            "quantized_with_raw" => Ok(StorageMode::QuantizedWithRaw),
            _ => Err(format!(
                "Invalid storage_mode: '{}'. Supported: quantized_only, quantized_with_raw",
                s
            ))
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            StorageMode::QuantizedOnly => "quantized_only",
            StorageMode::QuantizedWithRaw => "quantized_with_raw",
        }
    }
}

impl Default for StorageMode {
    fn default() -> Self {
        StorageMode::QuantizedOnly
    }
}

// Updated QuantizationConfig structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub subvectors: usize,
    pub bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,
    pub storage_mode: StorageMode,
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

impl Distance<u8> for DistPQ {
    /// Compute distance between query (via LUT) and stored PQ codes
    /// The first parameter `_a` is ignored since we use the pre-computed LUT
    /// The second parameter `b` contains the PQ codes for the stored vector (one u8 per subvector)
    fn eval(&self, _a: &[u8], b: &[u8]) -> f32 {
        let lut_guard = self.lut.read().unwrap();
        let lut = match lut_guard.as_ref() {
            Some(l) => l,
            None => return f32::INFINITY,
        };
        
        // b.len() should equal pq.subvectors
        let mut sum = 0.0f32;
        for (sv, &code) in b.iter().enumerate() {
            // lut[sv][code]
            let distance_component = lut.get(sv)
                .and_then(|row| row.get(code as usize))
                .copied()
                .unwrap_or(f32::INFINITY);
            sum += distance_component;
        }
        sum
    }
}

// Enhanced DistanceType enum to support PQ variants
enum DistanceType {
    // Raw vector variants
    Cosine(Hnsw<'static, f32, DistCosine>),
    L2(Hnsw<'static, f32, DistL2>),
    L1(Hnsw<'static, f32, DistL1>),
    
    // PQ variants - corrected to use u8 element type
    CosinePQ(Hnsw<'static, u8, DistPQ>),
    L2PQ(Hnsw<'static, u8, DistPQ>),
    L1PQ(Hnsw<'static, u8, DistPQ>),
}

impl DistanceType {
    fn new_raw(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        info!(
            operation = "hnsw_creation",
            space = space,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "raw",
            "Creating raw HNSW index"
        );

        match space {
            "cosine" => DistanceType::Cosine(Hnsw::new(m, expected_size, max_layer, ef_construction, DistCosine {})),
            "l2" => DistanceType::L2(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL2 {})),
            "l1" => DistanceType::L1(Hnsw::new(m, expected_size, max_layer, ef_construction, DistL1 {})),
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error
                error!(
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided"
                );
                // This is a programming error that should be caught earlier
                // For now, default to cosine to prevent panic
                warn!(
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                DistanceType::Cosine(Hnsw::new(m, expected_size, max_layer, ef_construction, DistCosine {}))
            }
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
        info!(
            operation = "hnsw_creation",
            space = space,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "quantized",
            subvectors = pq.subvectors,
            bits = pq.bits,
            "Creating PQ-enabled HNSW index"
        );

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
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error
                error!(
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided for PQ"
                );
                warn!(
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                let dist_pq = DistPQ::new(pq);
                DistanceType::CosinePQ(Hnsw::new(m, expected_size, max_layer, ef_construction, dist_pq))
            }
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
                
                // Create dummy query vector for HNSW traversal (flat u8 codes)
                let dummy_query = vec![0u8; self.get_code_size()];
                
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
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error logging
                error!(
                    operation = "vector_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_raw_vectors_into_pq_index",
                    "Cannot insert raw vectors into PQ index"
                );
            }
        }
    }
    
    /// Insert PQ codes into the index
    fn insert_pq_codes(&mut self, codes: &[u8], id: usize) {
        match self {
            DistanceType::CosinePQ(hnsw) => {
                hnsw.insert((codes, id));
            },
            DistanceType::L2PQ(hnsw) => {
                hnsw.insert((codes, id));
            },
            DistanceType::L1PQ(hnsw) => {
                hnsw.insert((codes, id));
            },
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error logging
                error!(
                    operation = "pq_codes_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_pq_codes_into_raw_index",
                    "Cannot insert PQ codes into raw index"
                );
            }
        }
    }

    #[allow(dead_code)]
    fn insert_batch(&mut self, data: &[(&Vec<f32>, usize)]) {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        debug!(
            operation = "batch_insert",
            batch_size = data.len(),
            num_threads = num_threads,
            threshold = threshold,
            parallel = data.len() >= threshold,
            "Starting batch insertion"
        );

        if data.len() >= threshold {
            match self {
                DistanceType::Cosine(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L2(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L1(hnsw) => hnsw.parallel_insert(data),
                _ => {
                    // ✅ ENTERPRISE: Replace panic with graceful error
                    error!(
                        operation = "batch_insert",
                        error = "invalid_operation",
                        reason = "cannot_batch_insert_raw_vectors_into_pq_index",
                        "Cannot batch insert raw vectors into PQ index"
                    );
                }
            }
        } else {
            for (vector, id) in data {
                self.insert(vector.as_slice(), *id);
            }
        }
    }
    
    fn insert_batch_pq(&mut self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        debug!(
            operation = "batch_insert_pq",
            batch_size = data.len(),
            num_threads = num_threads,
            threshold = threshold,
            parallel = data.len() >= threshold,
            "Starting PQ batch insertion"
        );

        match self {
            DistanceType::CosinePQ(hnsw) |
            DistanceType::L2PQ(hnsw) |
            DistanceType::L1PQ(hnsw) => {
                if data.len() >= threshold {
                    hnsw.parallel_insert(data);
                } else {
                    for (codes, id) in data {
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

    // ID-based training collection
    training_ids: RwLock<Vec<String>>,          // Just IDs, not vectors
    training_threshold_reached: AtomicBool,     // Atomic flag for safety

    // Timestamp when the index was created
    created_at: String,

    // NEW: Flag to prevent training ID collection during persistence rebuild
    pub rebuilding_from_persistence: AtomicBool,
}

#[pymethods]
impl HNSWIndex {
    #[new]
    #[pyo3(signature = (dim, space, m, ef_construction, expected_size, quantization_config = None))]
    #[instrument(level = "info", skip(quantization_config), fields(
        dim = dim,
        space = %space,
        m = m,
        ef_construction = ef_construction,
        expected_size = expected_size,
        has_quantization = quantization_config.is_some()
    ))]
    fn new(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
        quantization_config: Option<&Bound<PyDict>>,
    ) -> PyResult<Self> {
        let start_time = Instant::now();

        // Validation of parameters
        if dim == 0 {
            error!(operation = "validation", field = "dim", value = dim, "Invalid dimension");
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("dim must be positive"));
        }
        if ef_construction == 0 {
            error!(operation = "validation", field = "ef_construction", value = ef_construction, "Invalid ef_construction");
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("ef_construction must be positive"));
        }
        if expected_size == 0 {
            error!(operation = "validation", field = "expected_size", value = expected_size, "Invalid expected_size");
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("expected_size must be positive"));
        }
        if m > 256 {
            error!(operation = "validation", field = "m", value = m, max_allowed = 256, "m exceeds maximum");
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("m must be less than or equal to 256"));
        }

        // Early space validation with user-friendly error
        let space_normalized = space.to_lowercase();
        match space_normalized.as_str() {
            "cosine" | "l2" | "l1" => {
                debug!(operation = "validation", space = %space_normalized, "Distance space validated");
            }, 
            _ => {
                error!(operation = "validation", field = "space", value = %space, "Unsupported distance space");
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Unsupported space: '{}'. Supported spaces: 'cosine', 'l2', 'l1'", space)
                ));
            }
        }
        
        // Extract quantization configuration
        let (quantization_params, pq_instance) = if let Some(config) = quantization_config {
            let qtype = config.get_item("type")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'type' in quantization_config"))?
                .extract::<String>()?;
            
            if qtype != "pq" {
                error!(operation = "validation", field = "quantization_type", value = %qtype, "Unsupported quantization type");
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

            // Extract storage_mode
            let storage_mode_str = config.get_item("storage_mode")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "quantized_only".to_string());

            let storage_mode = StorageMode::from_string(&storage_mode_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

            // Validate PQ parameters
            if dim % subvectors != 0 {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    dim = dim,
                    subvectors = subvectors,
                    "Subvectors must divide dimension evenly"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("subvectors ({}) must divide dimension ({}) evenly", subvectors, dim)
                ));
            }
            
            if bits < 1 || bits > 8 {
                error!(operation = "validation", field = "bits", value = bits, min = 1, max = 8, "Bits out of range");
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("bits must be between 1 and 8, got {}", bits)
                ));
            }
            
            if training_size < 1000 {
                error!(operation = "validation", field = "training_size", value = training_size, min = 1000, "Training size too small");
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("training_size must be at least 1000, got {}", training_size)
                ));
            }
            
            let config = QuantizationConfig {
                subvectors,
                bits,
                training_size,
                max_training_vectors,
                storage_mode,
            };
            
            debug!(
                operation = "pq_configuration",
                subvectors = subvectors,
                bits = bits,
                training_size = training_size,
                storage_mode = %storage_mode_str,
                sub_dim = dim / subvectors,
                num_centroids = 1 << bits,
                "Product Quantization configured"
            );
            
            // Create PQ instance
            let pq = Arc::new(PQ::new(dim, subvectors, bits, training_size, max_training_vectors));
            
            (Some(config), Some(pq))
        } else {
            (None, None)
        };

        let max_layer = 16; // Always use NB_LAYER_MAX for hnsw-rs compatibility
        trace!(operation = "hnsw_config", max_layer = max_layer, reason = "hnsw-rs compatibility", "Using fixed max_layer");

        // Create initial raw HNSW index (will be rebuilt as PQ after training)
        let hnsw = DistanceType::new_raw(&space_normalized, m, expected_size, max_layer, ef_construction);

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "index_creation_complete",
            dim = dim,
            space = %space_normalized,
            m = m,
            ef_construction = ef_construction,
            expected_size = expected_size,
            has_quantization = quantization_params.is_some(),
            duration_ms = duration_ms,
            "HNSW index created successfully"
        );

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
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            created_at: Utc::now().to_rfc3339(),
            rebuilding_from_persistence: AtomicBool::new(false),
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

    /// Get the distance space configuration
    pub fn get_space(&self) -> String {
        self.space.clone()
    }

    /// Get next available internal ID
    fn get_next_id(&self) -> usize {
        let mut counter = self.id_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Rebuild the HNSW index to use PQ codes after training is complete
    #[instrument(level = "info", skip(self), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization()
    ), err)]
    pub fn rebuild_with_quantization(&mut self) -> PyResult<bool> {
        let start_time = Instant::now();

        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                warn!(operation = "rebuild_quantization", reason = "pq_not_trained", "Cannot rebuild: PQ not trained");
                return Ok(false);
            }
        };

        // Get all current vectors for quantization
        let vectors = self.vectors.read().unwrap();
        if vectors.is_empty() {
            warn!(operation = "rebuild_quantization", reason = "no_vectors", "Cannot rebuild: no vectors available");
            return Ok(false);
        }

        info!(operation = "quantization_rebuild_start", vector_count = vectors.len(), "Starting quantization rebuild");

        // Quantize all existing vectors
        let vector_refs: Vec<&[f32]> = vectors.values().map(|v| v.as_slice()).collect();
        let quantized_codes = pq.quantize_batch(&vector_refs)
            .map_err(|e| {
                error!(operation = "quantization_rebuild", error = %e, "Failed to quantize vectors");
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to quantize vectors: {}", e)
                )
            })?;

        // Create new PQ-based HNSW index
        let max_layer = 16; // Always use NB_LAYER_MAX for consistency
        trace!(operation = "rebuild_quantization", max_layer = max_layer, "Creating new PQ HNSW index");

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
            debug!(operation = "quantization_rebuild", codes_stored = pq_codes.len(), "Quantized codes stored");
        }

        // Replace the HNSW index
        {
            let mut hnsw_guard = self.hnsw.lock().unwrap();
            *hnsw_guard = new_hnsw;
        }

        // Insert quantized vectors into new index
        let pq_codes = self.pq_codes.read().unwrap();
        let id_map = self.id_map.read().unwrap();
        let mut batch_data: Vec<(&Vec<u8>, usize)> = Vec::new();
        
        for (id, codes) in pq_codes.iter() {
            if let Some(&internal_id) = id_map.get(id) {
                batch_data.push((codes, internal_id));
            }
        }

        if !batch_data.is_empty() {
            let mut hnsw_guard = self.hnsw.lock().unwrap();
            hnsw_guard.insert_batch_pq(&batch_data)
                .map_err(|e| {
                    error!(operation = "quantization_rebuild", error = %e, "Failed to insert quantized vectors");
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Failed to insert quantized vectors: {}", e)
                    )
                })?;
        }

        // ✅ ENTERPRISE: Add duration timing with fixed compression ratio calculation
        let duration_ms = start_time.elapsed().as_millis();
        let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
        info!(
            operation = "quantization_rebuild_complete",
            vector_count = vectors.len(),
            codes_inserted = batch_data.len(),
            compression_ratio = compression_ratio,
            duration_ms = duration_ms,
            "Quantization rebuild completed successfully"
        );

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



    /// Enhanced add method that properly handles PQ overwrite scenarios
    #[pyo3(signature = (data, overwrite = true))]
    #[instrument(level = "info", skip(self, data), fields(
        overwrite = overwrite,
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn add(&mut self, data: Bound<PyAny>, overwrite: bool) -> PyResult<AddResult> {
        let start_time = Instant::now();

        // Input validation
        if data.is_none() {
            error!(operation = "add_vectors", error = "data_is_none", "Data cannot be None");
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data cannot be None"
            ));
        }

        // Use error-collecting parsing
        let (parsed_data, parse_errors) = self.parse_input_data(&data);

        let mut total_inserted = 0;
        let mut total_errors = 0;
        let mut errors = Vec::new();

        // Add parse errors to the collection
        for parse_error in parse_errors {
            errors.push(parse_error);
            total_errors += 1;
        }

        if parsed_data.is_empty() && errors.is_empty() {
            trace!(operation = "add_vectors", result = "empty_input", "No vectors to process");
            return Ok(AddResult {
                total_inserted: 0,
                total_errors: 0,
                errors: vec![],
                vector_shape: Some((0, self.dim)),
            });
        }

        let total_input_count = parsed_data.len() + total_errors;
        let vector_shape = Some((total_input_count, self.dim));

        debug!(
            operation = "add_vectors_start",
            total_vectors = parsed_data.len(),
            parse_errors = total_errors,
            overwrite = overwrite,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            storage_mode = self.get_storage_mode(),
            "Starting vector addition"
        );

        // ENHANCED FIX: Handle overwrites properly for ALL paths (Raw, Training, PQ)
        if overwrite {
            // Phase 1: Batch identify and remove existing documents
            let (ids_to_remove, storage_analysis) = {
                let id_map = self.id_map.read().unwrap();
                let vectors = self.vectors.read().unwrap();
                let pq_codes = self.pq_codes.read().unwrap();

                let mut ids_to_remove = Vec::new();
                let mut has_raw = 0;
                let mut has_pq = 0;
                let mut has_both = 0;

                for (id, _, _) in &parsed_data {
                    if id_map.contains_key(id) {
                        ids_to_remove.push(id.clone());

                        // Analyze what's being replaced for logging
                        let has_raw_vector = vectors.contains_key(id);
                        let has_pq_codes = pq_codes.contains_key(id);

                        match (has_raw_vector, has_pq_codes) {
                            (true, true) => has_both += 1,
                            (true, false) => has_raw += 1,
                            (false, true) => has_pq += 1,
                            (false, false) => {} // Shouldn't happen, but handle gracefully
                        }
                    }
                }

                (ids_to_remove, (has_raw, has_pq, has_both))
            }; // Release all read locks here

            if !ids_to_remove.is_empty() {
                info!(
                    operation = "overwrite_preparation",
                    documents_to_remove = ids_to_remove.len(),
                    storage_analysis = format!("raw_only: {}, pq_only: {}, both: {}", 
                        storage_analysis.0, storage_analysis.1, storage_analysis.2),
                    "Removing existing documents for overwrite"
                );

                // Batch remove existing documents (handles both raw and PQ data)
                let mut removed_count = 0;
                let mut removal_errors = 0;

                for id in ids_to_remove {
                    match self.remove_point_internal(id.clone()) {
                        Ok(was_removed) => {
                            if was_removed {
                                removed_count += 1;
                                trace!(
                                    operation = "overwrite_removal",
                                    vector_id = %id,
                                    "Removed existing vector/codes for overwrite"
                                );
                            }
                        }
                        Err(e) => {
                            removal_errors += 1;
                            warn!(
                                operation = "overwrite_removal",
                                vector_id = %id,
                                error = %e,
                                "Failed to remove existing vector for overwrite"
                            );
                            errors.push(format!("Failed to remove existing {}: {}", id, e));
                            total_errors += 1;
                        }
                    }
                }

                info!(
                    operation = "overwrite_removal_complete",
                    removed_count = removed_count,
                    removal_errors = removal_errors,
                    "Completed removal phase for overwrite"
                );
            }
        }

        // Phase 2: Add new vectors using the correct path based on current PQ state
        debug!(
            operation = "add_vectors_insertion_phase",
            current_state = self.get_storage_mode(),
            "Starting insertion phase"
        );

        for (id, vector, metadata) in parsed_data {
            let id_for_error = id.clone();

            // Use overwrite=false since we already handled removals above
            // The add_single_vector method will route to the correct path based on current PQ state
            match self.add_single_vector(id, vector, metadata, false) {
                Ok(inserted_new) => {
                    total_inserted += 1;
                    if inserted_new {
                        let mut count = self.vector_count.lock().unwrap();
                        *count += 1;
                    }

                    // Check training trigger (graceful failure handling)
                    if let Err(training_error) = self.maybe_trigger_training() {
                        warn!(
                            operation = "training_trigger",
                            error = %training_error,
                            vector_id = %id_for_error,
                            "Training trigger failed"
                        );
                        errors.push(format!("Training failed: {}", training_error));
                    }
                }
                Err(e) => {
                    total_errors += 1;
                    errors.push(format!("Vector {}: {}", id_for_error, e));
                    trace!(
                        operation = "add_vector_error",
                        vector_id = %id_for_error,
                        error = %e,
                        "Vector addition failed"
                    );
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "add_vectors_complete",
            total_inserted = total_inserted,
            total_errors = total_errors,
            success_rate = if total_input_count > 0 { 
                total_inserted as f64 / total_input_count as f64 * 100.0
            } else {
                100.0
            },
            duration_ms = duration_ms,
            overwrite_mode = overwrite,
            final_storage_mode = self.get_storage_mode(),
            "Vector addition completed"
        );

        Ok(AddResult {
            total_inserted,
            total_errors,
            errors,
            vector_shape,
        })
    }




    pub fn get_training_progress(&self) -> f32 {
        if let Some(config) = &self.quantization_config {
            // If PQ is trained, always return 100%
            if let Some(pq) = &self.pq {
                if pq.is_trained() {
                    return 100.0;
                }
            }
            let training_ids = self.training_ids.read().unwrap();
            (training_ids.len() as f32 / config.training_size as f32 * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Get number of training vectors still needed
    pub fn training_vectors_needed(&self) -> usize {
        if let Some(config) = &self.quantization_config {
            if self.training_threshold_reached.load(Ordering::Acquire) {
                0
            } else {
                let training_ids = self.training_ids.read().unwrap();
                config.training_size.saturating_sub(training_ids.len())
            }
        } else {
            0
        }
    }

    /// Check if training is ready to be triggered
    pub fn is_training_ready(&self) -> bool {
        self.training_threshold_reached.load(Ordering::Acquire)
    }

    /// Get current storage mode description
    pub fn get_storage_mode(&self) -> String {
        if !self.has_quantization() {
            "raw_only".to_string()
        } else if !self.can_use_quantization() {
            if self.training_threshold_reached.load(Ordering::Acquire) {
                "raw_ready_for_training".to_string()
            } else {
                "raw_collecting_for_training".to_string()
            }
        } else if self.is_quantized() {
            "quantized_active".to_string()
        } else {
            "raw_trained_not_rebuilt".to_string()
        }
    }

    /// Enhanced search method with automatic ADC usage
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false))]
    #[instrument(level = "debug", skip(self, py, vector, filter), fields(
        top_k = top_k,
        ef_search = ef_search,
        return_vector = return_vector,
        is_quantized = self.is_quantized()
    ), err)]
    pub fn search(
        &self,
        py: Python<'_>,
        vector: Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
    ) -> PyResult<PyObject> {
        let start_time = Instant::now();

        let ef = ef_search.unwrap_or_else(|| {
            match self.space.to_lowercase().as_str() {
                "l1" | "l2" => std::cmp::max(2 * top_k, 150),
                _ => std::cmp::max(2 * top_k, 100),
            }
        });

        trace!(operation = "search_config", ef = ef, space = %self.space, "Search parameters configured");

        let filter_conditions = filter.map(|f| self.python_dict_to_value_map(f)).transpose()?;

        // Detect batch vs single query with comprehensive input support
        let result: PyObject = if let Ok(list_vec) = vector.extract::<Vec<Vec<f32>>>() {
            // Format: List of vectors [[0.1, 0.2], [0.3, 0.4]]

            // Validation for empty batch or empty vectors in batch
            if list_vec.is_empty() {
                error!(operation = "search", error = "empty_batch", "Batch cannot be empty");
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Batch cannot be empty"
                ));
            }

            // Check for empty vectors within the batch
            for (i, vec) in list_vec.iter().enumerate() {
                if vec.is_empty() {
                    error!(operation = "search", error = "empty_vector_in_batch", vector_index = i, "Vector in batch cannot be empty");
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Vector {} in batch cannot be empty", i)
                    ));
                }
            }

            debug!(operation = "batch_search", batch_size = list_vec.len(), "Starting batch search");
            let results = self.batch_search_internal(&list_vec, filter_conditions.as_ref(), top_k, ef, return_vector, py)?;
            PyList::new(py, results)?.into()
        } else if let Ok(np_array) = vector.downcast::<PyArray2<f32>>() {
            // Format: NumPy 2D array (N, dims)
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            if shape.len() != 2 || shape[1] != self.dim {
                error!(
                    operation = "search",
                    error = "shape_mismatch",
                    expected_shape = format!("(N, {})", self.dim),
                    actual_shape = format!("{:?}", shape),
                    "NumPy array shape mismatch"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "NumPy array must have shape (N, {}), got {:?}", self.dim, shape
                )));
            }

            let flat = readonly.as_slice()?;
            let batch: Vec<Vec<f32>> = flat.chunks(self.dim).map(|chunk| chunk.to_vec()).collect();
            debug!(operation = "batch_search_numpy", batch_size = batch.len(), "Starting NumPy batch search");
            let results = self.batch_search_internal(&batch, filter_conditions.as_ref(), top_k, ef, return_vector, py)?;
            PyList::new(py, results)?.into()
        } else {
            // Single vector path - enhanced with NumPy 1D support
            let query_vector = if let Ok(array1d) = vector.downcast::<PyArray1<f32>>() {
                array1d.readonly().as_slice()?.to_vec()
            } else {
                vector.extract::<Vec<f32>>()?
            };

            // PROCESS HERE using extract_single_vector logic
            let processed_query = self.validate_and_process_query_vector(query_vector)?;

            trace!(operation = "single_search", query_dim = processed_query.len(), "Starting single vector search");
            
            let search_results = py.allow_threads(|| {
                // Check if we should use quantized search
                let use_quantized = self.is_quantized();

                trace!(operation = "search_method", use_quantized = use_quantized, "Selected search method");

                let hnsw_results = {
                    let hnsw_guard = self.hnsw.lock().unwrap();

                    if use_quantized {
                        // Use ADC search for quantized index
                        hnsw_guard.search(&processed_query, top_k, ef)
                            .unwrap_or_else(|e| {
                                error!(operation = "adc_search", error = %e, "ADC search failed");
                                Vec::new()
                            })
                    } else {
                        // Use raw vector search
                        match hnsw_guard.search(&processed_query, top_k, ef) {
                            Ok(results) => results,
                            Err(e) => {
                                error!(operation = "raw_search", error = %e, "Raw search failed");
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

            PyList::new(py, output)?.into()
        };

        // ✅ ENTERPRISE: Add duration timing to hot path with actual result count
        let duration_ms = start_time.elapsed().as_millis();
        let results_count = {
            let any = result.bind(py); 
            match any.downcast::<PyList>() {
                Ok(list) => list.len(),
                Err(_) => 0,
            }
        };

        debug!(operation = "search_complete", results_count = results_count, duration_ms = duration_ms, "Search completed");

        Ok(result)
    }

    /// Enhanced Save method to include HNSW Graph
    #[instrument(level = "info", skip(self), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn save(&self, path: &str) -> PyResult<()> {
        let start_time = Instant::now();
        info!(operation = "save_start", path = path, "Starting index save");

        let path_buf = Path::new(path);

        // Phase 1: Save all ZeusDB components (already tested to work)
        debug!(operation = "save_phase1", "Saving ZeusDB components");
        crate::persistence::save_index(self, path)?;

        // Phase 2: Save HNSW graph using hnsw-rs native dump
        debug!(operation = "save_phase2", "Saving HNSW graph");
        self.save_hnsw_graph(path_buf)?;

        let duration_ms = start_time.elapsed().as_millis();
        info!(operation = "save_complete", path = path, duration_ms = duration_ms, "Index save completed successfully");
        Ok(())
    }


    /// Python property: `index.dim`
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }



    /// Get records by ID(s) with PQ reconstruction support and storage mode awareness
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

        trace!(operation = "get_records", record_count = ids.len(), return_vector = return_vector, "Retrieving records");

        let mut records = Vec::with_capacity(ids.len());

        // Use read locks for concurrent access
        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        for id in ids {
            // Check if this ID exists in either storage
            let exists = vectors.contains_key(&id) || pq_codes.contains_key(&id);

            if exists {
                let metadata = vector_metadata.get(&id).cloned().unwrap_or_default();

                let dict = PyDict::new(py);
                dict.set_item("id", id.clone())?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    // Priority: raw vector > PQ reconstruction
                    let vector_data = if let Some(raw_vector) = vectors.get(&id) {
                        // Case 1: Raw vector available (QuantizedWithRaw mode or non-quantized)
                        Some(raw_vector.clone())
                    } else if let (Some(pq), Some(codes)) = (&self.pq, pq_codes.get(&id)) {
                        // Case 2: Only quantized codes available (QuantizedOnly mode)
                        match pq.reconstruct(codes) {
                            Ok(reconstructed) => Some(reconstructed),
                            Err(e) => {
                                warn!(operation = "vector_reconstruction", vector_id = %id, error = %e, "Failed to reconstruct vector");
                                None
                            }
                        }
                    } else {
                        // Case 3: No vector data available
                        None
                    };

                    if let Some(vec) = vector_data {
                        dict.set_item("vector", vec)?;
                    }
                }

                records.push(dict.into());
            }
        }

        trace!(operation = "get_records_complete", found_records = records.len(), "Records retrieval completed");
        Ok(records)
    }

    /// Enhanced get_stats with storage mode information
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();

        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let vector_count = *self.vector_count.lock().unwrap();
        let training_ids = self.training_ids.read().unwrap();

        // Basic stats
        stats.insert("total_vectors".to_string(), vector_count.to_string());
        stats.insert("dimension".to_string(), self.dim.to_string());
        stats.insert("expected_size".to_string(), self.expected_size.to_string());
        stats.insert("space".to_string(), self.space.clone());
        stats.insert("index_type".to_string(), "HNSW".to_string());

        stats.insert("m".to_string(), self.m.to_string());
        stats.insert("ef_construction".to_string(), self.ef_construction.to_string());
        stats.insert("thread_safety".to_string(), "RwLock+Mutex".to_string());

        // Storage breakdown
        stats.insert("raw_vectors_stored".to_string(), vectors.len().to_string());
        stats.insert("quantized_codes_stored".to_string(), pq_codes.len().to_string());

        // Training info
        if let Some(config) = &self.quantization_config {
            stats.insert("quantization_type".to_string(), "pq".to_string());
            stats.insert("quantization_training_size".to_string(), config.training_size.to_string());

            // Storage mode information
            stats.insert("storage_mode".to_string(), config.storage_mode.to_string().to_string());

            // Calculate actual memory usage based on storage mode
            let raw_memory_mb = (vectors.len() * self.dim * 4) as f64 / (1024.0 * 1024.0);
            let quantized_memory_mb = (pq_codes.len() * config.subvectors) as f64 / (1024.0 * 1024.0);

            stats.insert("raw_vectors_memory_mb".to_string(), format!("{:.2}", raw_memory_mb));
            stats.insert("quantized_codes_memory_mb".to_string(), format!("{:.2}", quantized_memory_mb));

            match config.storage_mode {
                StorageMode::QuantizedOnly => {
                    stats.insert("storage_strategy".to_string(), "memory_optimized".to_string());
                    stats.insert("memory_savings".to_string(), "maximum".to_string());
                }
                StorageMode::QuantizedWithRaw => {
                    stats.insert("storage_strategy".to_string(), "quality_optimized".to_string());
                    stats.insert("memory_savings".to_string(), "raw_vectors_kept".to_string());
                }
            }

            let collected_count = training_ids.len();
            let progress = self.get_training_progress();
            stats.insert("training_progress".to_string(),
                format!("{}/{} ({:.1}%)", collected_count, config.training_size, progress));
            
            let vectors_needed = self.training_vectors_needed();
            stats.insert("training_vectors_needed".to_string(), vectors_needed.to_string());
            stats.insert("training_threshold_reached".to_string(),
                self.training_threshold_reached.load(Ordering::Acquire).to_string());

            if let Some(pq) = &self.pq {
                let is_trained = pq.is_trained();
                stats.insert("quantization_trained".to_string(), is_trained.to_string());
                stats.insert("quantization_active".to_string(), self.is_quantized().to_string());

                if is_trained {
                    let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
                    stats.insert("quantization_compression_ratio".to_string(), format!("{:.1}x", compression_ratio));
                }
            }
        } else {
            stats.insert("quantization_type".to_string(), "none".to_string());
            stats.insert("storage_mode".to_string(), "raw_only".to_string());
        }

        stats.insert("storage_mode_description".to_string(), self.get_storage_mode());

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
            
            // Use cached compression ratio calculation with proper float division
            let compression_info = self.pq.as_ref()
                .map(|pq| format!("{:.1}x", (pq.dim as f64 * 4.0) / pq.subvectors as f64))
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
    /// Public remove_point method (unchanged for API compatibility)
    /// This code delegates to remove_point_internal() which handles all the complex logic
    pub fn remove_point(&mut self, id: String) -> PyResult<bool> {
        match self.remove_point_internal(id) {
            Ok(result) => Ok(result),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
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
        use rand::random;  // Import for random number generation
        
        let start_time = Instant::now();
        
        debug!(
            operation = "benchmark_start",
            query_count = query_count,
            max_threads = max_threads,
            "Starting concurrent read benchmark"
        );
        
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| random::<f32>()).collect())
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
        
        let total_duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "benchmark_complete",
            sequential_qps = queries.len() as f64 / sequential_time,
            parallel_qps = queries.len() as f64 / parallel_time,
            speedup = sequential_time / parallel_time,
            duration_ms = total_duration_ms,
            "Benchmark completed"
        );
        
        Ok(results)
    }

    /// Raw performance benchmark
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_raw_concurrent_performance(&self, query_count: usize, max_threads: Option<usize>) -> HashMap<String, f64> {
        use rand::random;  // Import for random number generation
        
        let start_time = Instant::now();
        
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| random::<f32>()).collect())
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

        results.insert("sequential_time".to_string(), sequential_time);
        results.insert("parallel_time".to_string(), parallel_time);
        results.insert("sequential_qps".to_string(), queries.len() as f64 / sequential_time);
        results.insert("parallel_qps".to_string(), queries.len() as f64 / parallel_time);
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);
        results.insert("note".to_string(), "limited_by_hnsw_mutex".parse().unwrap_or(0.0));
        
        let total_duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "benchmark_complete",
            sequential_qps = queries.len() as f64 / sequential_time,
            parallel_qps = queries.len() as f64 / parallel_time,
            speedup = sequential_time / parallel_time,
            duration_ms = total_duration_ms,
            "Benchmark completed"
        );
        
        results
    }


    /// Get current code version counter to verify build updates
    pub fn get_code_version(&self) -> String {
        format!("Version: {}, Description: {}", CODE_VERSION_COUNTER, CODE_VERSION_DESCRIPTION)
    }

    /// Get just the version number for quick checking
    pub fn get_version_number(&self) -> u32 {
        CODE_VERSION_COUNTER
    }
}

// INTERNAL METHODS, HELPERS AND IMPLEMENTATIONS
impl HNSWIndex {

    /// Pure function for vector normalization
    fn normalize_vector(&self, vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vector  // Return unchanged for zero vectors
        }
    }

    /// Process vector according to distance space
    fn process_vector_for_space(&self, vector: Vec<f32>) -> Vec<f32> {
        match self.space.to_lowercase().as_str() {
            "cosine" => self.normalize_vector(vector),
            // Future extensions:
            // "l2" => self.preprocess_l2(vector),
            // "l1" => self.preprocess_l1(vector),
            _ => vector
        }
    }

    /// Helper for query processing (mirrors extract_single_vector validation)
    fn validate_and_process_query_vector(&self, vector: Vec<f32>) -> PyResult<Vec<f32>> {
        // Same validation as extract_single_vector
        if vector.is_empty() {
            error!(operation = "query_validation", error = "empty_vector", "Search vector cannot be empty");
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Search vector cannot be empty"
            ));
        }
        if vector.len() != self.dim {
            error!(
                operation = "query_validation",
                error = "dimension_mismatch",
                expected = self.dim,
                actual = vector.len(),
                "Search vector dimension mismatch"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Search vector dimension mismatch: expected {}, got {}", self.dim, vector.len()
            )));
        }
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                error!(
                    operation = "query_validation",
                    error = "invalid_value",
                    index = i,
                    value = val,
                    "Search vector contains invalid value"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Search vector contains invalid value at index {}: {}", i, val
                )));
            }
        }

        // Apply same processing as storage vectors
        Ok(self.process_vector_for_space(vector))
    }


    /// Internal remove_point method that can be called without Python bindings
    /// This is the core method that properly removes all traces of a document
    /// Enhanced internal remove_point method with comprehensive PQ support
    fn remove_point_internal(&mut self, id: String) -> Result<bool, String> {
        // Get all write locks in a consistent order to prevent deadlocks
        let mut vectors = self.vectors.write().unwrap();
        let mut vector_metadata = self.vector_metadata.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();
        let mut rev_map = self.rev_map.write().unwrap();
        let mut pq_codes = self.pq_codes.write().unwrap();

        // Check if the document exists
        if let Some(internal_id) = id_map.remove(&id) {
            // Track what we're removing for logging
            let had_raw_vector = vectors.contains_key(&id);
            let had_pq_codes = pq_codes.contains_key(&id);

            // Remove from all data structures
            vectors.remove(&id);           // Remove raw vectors (if present)
            vector_metadata.remove(&id);   // Remove metadata
            pq_codes.remove(&id);         // Remove PQ codes (if present)
            rev_map.remove(&internal_id);  // Remove ID mapping

            // Enhanced training state cleanup for quantization
            if self.has_quantization() {
                // Remove from training IDs if present and not yet trained
                if !self.can_use_quantization() {
                    let mut training_ids = self.training_ids.write().unwrap();
                    let original_len = training_ids.len();
                    training_ids.retain(|training_id| training_id != &id);

                    if training_ids.len() != original_len {
                        trace!(
                            operation = "training_cleanup",
                            vector_id = %id,
                            remaining_training_vectors = training_ids.len(),
                            "Removed vector from training set"
                        );

                        // Update threshold status if we dropped below training size
                        if let Some(config) = &self.quantization_config {
                            if training_ids.len() < config.training_size {
                                self.training_threshold_reached.store(false, std::sync::atomic::Ordering::Release);
                                debug!(
                                    operation = "training_threshold_reset",
                                    remaining_vectors = training_ids.len(),
                                    required = config.training_size,
                                    "Training threshold reset due to removal"
                                );
                            }
                        }
                    }
                }
            }

            // Decrement vector count since we removed a vector
            {
                let mut count = self.vector_count.lock().unwrap();
                if *count > 0 {
                    *count -= 1;
                }
            }

            debug!(
                operation = "remove_point_internal",
                vector_id = %id,
                internal_id = internal_id,
                had_raw_vector = had_raw_vector,
                had_pq_codes = had_pq_codes,
                storage_mode = self.get_storage_mode(),
                note = "hnsw_graph_entries_remain_unreachable",
                "Vector completely removed from index (HNSW graph entries become unreachable)"
            );
            Ok(true)
        } else {
            trace!(
                operation = "remove_point_internal",
                vector_id = %id,
                "Vector not found for removal"
            );
            Ok(false)
        }
    }




    // 1. CORE VECTOR OPERATIONS (6 methods)
    /// 3-PATH ARCHITECTURE - Main router
    fn add_single_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, Value>,
        overwrite: bool
    ) -> PyResult<bool> {
        // Check if this is a new vector or an overwrite
        let is_new = {
            let id_map = self.id_map.read().unwrap();
            !id_map.contains_key(&id)
        };

        if !overwrite && !is_new {
            warn!(
                operation = "add_single_vector",
                vector_id = %id,
                reason = "already_exists",
                "Vector already exists and overwrite=false"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Vector with ID '{}' already exists", id)
            ));
        }

        trace!(
            operation = "add_single_vector",
            vector_id = %id,
            is_new = is_new,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            "Routing vector addition"
        );

        // Clean 3-Path Architecture
        if !self.has_quantization() {
            // Path A: Raw storage (no quantization config)
            self.add_raw_vector(id, vector, metadata)?;
        } else if !self.is_quantized() {
            // Path B: Raw storage + ID collection for training
            self.add_with_id_collection(id, vector, metadata)?;
        } else {
            // Path C: Quantized storage (PQ trained and active)
            self.add_quantized_vector(id, vector, metadata)?;
        }

        Ok(is_new)
    }

    /// Path A: Raw storage (no quantization)
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "raw_storage"
    ))]
    fn add_raw_vector(
        &mut self,
        id: String,
        vector: Vec<f32>, // Already processed by extract_single_vector
        metadata: HashMap<String, Value>
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Store processed vector directly (no additional processing)
        {
            let mut vectors = self.vectors.write().unwrap();
            vectors.insert(id.clone(), vector.clone()); // Already normalized
        }

        // Insert processed vector into HNSW
        {
            let mut hnsw_guard = self.hnsw.lock().unwrap();
            hnsw_guard.insert(&vector, internal_id); // Already normalized
        }

        trace!(
            operation = "add_raw_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            "Raw vector added successfully"
        );

        Ok(())
    }

    /// Path B: ID collection for consistent training
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "id_collection"
    ))]
    fn add_with_id_collection(
        &mut self,
        id: String,
        vector: Vec<f32>,  // Already processed
        metadata: HashMap<String, Value>
    ) -> PyResult<()> {
        // 1. Store vector normally (single storage)
        self.add_raw_vector(id.clone(), vector, metadata)?;

        // SKIP TRAINING ID COLLECTION DURING PERSISTENCE REBUILD
        if self.rebuilding_from_persistence.load(std::sync::atomic::Ordering::Acquire) {
            trace!(
                operation = "add_with_id_collection",
                vector_id = %id,
                reason = "rebuilding_from_persistence",
                "Skipping training ID collection during rebuild"
            );
            return Ok(());
        }

        // 2. Collect ID for training (minimal memory overhead)
        if let Some(config) = &self.quantization_config {
            if !self.training_threshold_reached.load(Ordering::Acquire) {
                let mut training_ids = self.training_ids.write().unwrap();

                if training_ids.len() < config.training_size {
                    training_ids.push(id.clone());
                    let progress = (training_ids.len() as f32 / config.training_size as f32 * 100.0).min(100.0);

                    trace!(
                        operation = "training_id_collection",
                        vector_id = %id,
                        collected_count = training_ids.len(),
                        target_size = config.training_size,
                        progress_percent = progress,
                        "Training ID collected"
                    );

                    // Check if we've reached the threshold
                    if training_ids.len() >= config.training_size {
                        self.training_threshold_reached.store(true, Ordering::Release);
                        info!(
                            operation = "training_threshold_reached",
                            collected_count = training_ids.len(),
                            target_size = config.training_size,
                            "Training threshold reached - ready for PQ training"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Path C: Quantized storage with configurable raw vector retention
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "quantized_storage"
    ))]
    fn add_quantized_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,  // Already processed
        metadata: HashMap<String, Value>
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Quantize the vector
        let pq = self.pq.as_ref().unwrap();
        let codes = pq.quantize(&vector).map_err(|e| {
            error!(
                operation = "add_quantized_vector",
                vector_id = %id,
                error = %e,
                "Failed to quantize vector"
            );
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to quantize vector: {}", e)
            )
        })?;

        // Store quantized codes (always)
        {
            let mut pq_codes = self.pq_codes.write().unwrap();
            pq_codes.insert(id.clone(), codes.clone());
        }

        // Store raw vector only if configured to keep them
        if let Some(config) = &self.quantization_config {
            if config.storage_mode == StorageMode::QuantizedWithRaw {
                let mut vectors = self.vectors.write().unwrap();
                vectors.insert(id.clone(), vector.clone());
            }
            // If QuantizedOnly mode, we don't store raw vectors (saves memory)
        }

        // Insert codes into quantized HNSW
        {
            let mut hnsw_guard = self.hnsw.lock().unwrap();
            hnsw_guard.insert_pq_codes(&codes, internal_id);
        }

        trace!(
            operation = "add_quantized_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            codes_length = codes.len(),
            "Quantized vector added successfully"
        );

        Ok(())
    }

    /// TRAINING TRIGGER: Uses threshold flag for race condition safety
    #[instrument(level = "info", skip(self), fields(
        threshold_reached = self.training_threshold_reached.load(Ordering::Acquire),
        has_quantization = self.has_quantization()
    ))]
    fn maybe_trigger_training(&mut self) -> Result<(), String> {
        // Check atomic flag first (fast path)
        if !self.training_threshold_reached.load(Ordering::Acquire) {
            return Ok(());
        }

        // Only proceed if we have quantization config and aren't already trained
        if let Some(_config) = &self.quantization_config {
            if let Some(pq) = &self.pq {
                if !pq.is_trained() {
                    info!(operation = "training_trigger", "Training threshold reached - starting PQ training");
                    return self.train_quantization_from_ids();
                }
            }
        }

        Ok(())
    }

    /// TRAINING EXECUTION: Uses collected IDs for deterministic training set
    #[instrument(level = "info", skip(self), fields(
        has_pq = self.pq.is_some(),
        has_config = self.quantization_config.is_some()
    ))]
    fn train_quantization_from_ids(&mut self) -> Result<(), String> {
        let start_time = Instant::now();

        let pq = self.pq.as_ref().ok_or("PQ not available")?.clone();
        let config = self.quantization_config.as_ref().ok_or("Config not available")?.clone();
        
        // Get consistent training set using collected IDs
        let training_vectors = {
            let training_ids = self.training_ids.read().unwrap();

            // ADD EARLY CHECK:
            if training_ids.is_empty() {
                warn!(operation = "pq_training", reason = "no_training_ids", "No training IDs available");
                // Reset threshold to prevent repeated attempts
                self.training_threshold_reached.store(false, Ordering::Release);
                return Err("No training IDs available for training".to_string());
            }

            let vectors = self.vectors.read().unwrap();

            let mut training_data = Vec::new();
            let mut missing_vectors = 0;

            for id in training_ids.iter() {
                if let Some(vector) = vectors.get(id) {
                    training_data.push(vector.clone());
                } else {
                    missing_vectors += 1;
                }
            }

            if missing_vectors > 0 {
                warn!(
                    operation = "pq_training",
                    missing_vectors = missing_vectors,
                    available_vectors = training_data.len(),
                    "Some training vectors were removed before training"
                );
            }

            debug!(
                operation = "pq_training_dataset",
                collected_ids = training_ids.len(),
                available_vectors = training_data.len(),
                target_size = config.training_size,
                "Training dataset prepared"
            );

            training_data
        };

        if training_vectors.len() < config.training_size {
            error!(
                operation = "pq_training",
                available = training_vectors.len(),
                required = config.training_size,
                "Insufficient vectors for training"
            );
            return Err(format!("Insufficient vectors for training: need {}, have {} (some may have been removed)",
                config.training_size, training_vectors.len()));
        }

        // Respect max_training_vectors limit
        let final_training_set = if let Some(max_training) = config.max_training_vectors {
            if training_vectors.len() > max_training {
                // Take first max_training vectors (deterministic)
                debug!(
                    operation = "pq_training_limit",
                    available = training_vectors.len(),
                    using = max_training,
                    "Limiting training set size"
                );
                training_vectors.into_iter().take(max_training).collect()
            } else {
                training_vectors
            }
        } else {
            training_vectors
        };

        info!(
            operation = "pq_training_start",
            training_vectors = final_training_set.len(),
            subvectors = config.subvectors,
            bits = config.bits,
            "Starting PQ training"
        );

        // Train the PQ model
        let training_start = Instant::now();
        pq.train(&final_training_set)?;
        let training_duration = training_start.elapsed();

        info!(
            operation = "pq_training_complete",
            training_vectors = final_training_set.len(),
            duration_ms = training_duration.as_millis(),
            "PQ training completed successfully"
        );

        // Clear training IDs (no longer needed)
        {
            let mut training_ids = self.training_ids.write().unwrap();
            training_ids.clear();
        }

        // Rebuild index with quantization
        debug!(operation = "pq_rebuild_start", "Rebuilding index with quantization");
        let rebuild_start = Instant::now();
        let rebuild_success = self.rebuild_with_quantization()
            .map_err(|e| format!("Failed to rebuild with quantization: {}", e))?;
        let rebuild_duration = rebuild_start.elapsed();

        if rebuild_success {
            // Calculate compression info with proper float division
            let compression_ratio = (self.dim as f64 * 4.0) / pq.subvectors as f64;
            let memory_savings = (1.0 - (pq.subvectors as f64) / (self.dim as f64 * 4.0)) * 100.0;

            let total_duration_ms = start_time.elapsed().as_millis();
            info!(
                operation = "pq_complete",
                rebuild_duration_ms = rebuild_duration.as_millis(),
                compression_ratio = compression_ratio,
                memory_savings_percent = memory_savings,
                total_duration_ms = total_duration_ms,
                "Index successfully rebuilt with quantization"
            );
        } else {
            error!(operation = "pq_rebuild", "Index rebuild returned false");
            return Err("Index rebuild returned false".to_string());
        }

        Ok(())
    }

    // 2. SEARCH OPERATIONS (1 method)
    /// Raw search without Python objects (for benchmarking)
    fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        // HNSW search with locking
        let hnsw_results = {
            let hnsw_guard = self.hnsw.lock().unwrap();
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

    /// Parse input data into (id, vector, metadata) tuples with error collection
    fn parse_input_data(&self, data: &Bound<PyAny>) -> (Vec<(String, Vec<f32>, HashMap<String, Value>)>, Vec<String>) {
        let mut parsed_vectors = Vec::new();
        let mut errors = Vec::new();

        if let Ok(dict) = data.downcast::<PyDict>() {
            self.parse_dict_input_safe(dict, &mut parsed_vectors, &mut errors);
        } else if let Ok(list) = data.downcast::<PyList>() {
            self.parse_list_input_safe(list, &mut parsed_vectors, &mut errors);
        } else if let Ok(np_array) = data.downcast::<PyArray2<f32>>() {
            if let Err(e) = self.parse_numpy_input_safe(np_array, &mut parsed_vectors) {
                errors.push(format!("NumPy parsing error: {}", e));
            }
        } else {
            // Single vector
            match self.extract_single_vector_safe(data) {
                Ok(vector) => {
                    let id = self.generate_id();
                    parsed_vectors.push((id, vector, HashMap::new()));
                }
                Err(e) => {
                    errors.push(format!("Single vector error: {}", e));
                }
            }
        }

        (parsed_vectors, errors)
    }

    /// Safe dictionary parsing that collects errors
    fn parse_dict_input_safe(
        &self, 
        dict: &Bound<PyDict>, 
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>
    ) {
        // Check for single object format
        if dict.contains("id").unwrap_or(false) && 
            (dict.contains("values").unwrap_or(false) || dict.contains("vector").unwrap_or(false)) {

                // Single object format
                let vector_result = if let Ok(Some(values_item)) = dict.get_item("values") {
                    self.extract_single_vector_safe(&values_item)
                } else if let Ok(Some(vector_item)) = dict.get_item("vector") {
                    self.extract_single_vector_safe(&vector_item)
                } else {
                    Err("Missing 'vector' or 'values' key".to_string())
                };

                match vector_result {
                    Ok(vector) => {
                        let id = match dict.get_item("id") {
                            Ok(Some(id_item)) => id_item.extract::<String>().unwrap_or_else(|_| self.generate_id()),
                            _ => self.generate_id(),
                        };

                        let metadata = match dict.get_item("metadata") {
                            Ok(Some(meta_item)) => {
                                if let Ok(meta_dict) = meta_item.downcast::<PyDict>() {
                                    self.python_dict_to_value_map(meta_dict).unwrap_or_default()
                                } else {
                                    HashMap::new()
                                }
                            }
                            _ => HashMap::new(),
                        };

                        parsed_vectors.push((id, vector, metadata));
                    }
                    Err(e) => {
                        let id = dict.get_item("id")
                            .ok()
                            .flatten()
                            .and_then(|id_item| id_item.extract::<String>().ok())
                            .unwrap_or_else(|| "single_object".to_string());

                        errors.push(format!("Vector {}: {}", id, e));
                    }
                }
            } else {
                // Batch format - try the existing parse_batch_format
                if let Err(e) = self.parse_batch_format(dict, parsed_vectors) {
                    errors.push(format!("Batch parsing error: {}", e));
                }
            }
        }

    /// Handle Format 3 & 5: Batch format - WORKING SOLUTION
    fn parse_batch_format(&self, dict: &Bound<PyDict>, parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>) -> PyResult<()> {
        // Process each key path immediately without storing references

        // Try "vectors" key
        if let Some(vectors_item) = dict.get_item("vectors")? {
            if let Ok(list) = vectors_item.downcast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = vectors_item.downcast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return self.parse_numpy_with_context(np_array, dict, parsed_vectors);
            }
        }

        // Try "embeddings" key  
        if let Some(embeddings_item) = dict.get_item("embeddings")? {
            if let Ok(list) = embeddings_item.downcast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = embeddings_item.downcast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return self.parse_numpy_with_context(np_array, dict, parsed_vectors);
            }
        }

        // Try "values" key
        if let Some(values_item) = dict.get_item("values")? {
            if let Ok(list) = values_item.downcast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "values field must be a list in batch format"
                ));
            }
        }

        // No valid vector data found
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Missing vector data. Expected one of: 'vectors', 'embeddings', or 'values' key"
        ))
    }

    /// Helper method to process vector list (extracted to avoid code duplication)
    fn process_vector_list(
        &self, 
        vectors: &Bound<PyList>, 
        dict: &Bound<PyDict>, 
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>
    ) -> PyResult<()> {
        // Process each vector in the batch
        for (i, vector_item) in vectors.iter().enumerate() {
            let vector = self.extract_single_vector(&vector_item)?;

            // Extract ID from "ids" array
            let id = match dict.get_item("ids")? {
                Some(item) => {
                    let ids_list = item.downcast::<PyList>()?;
                    if i < ids_list.len() {
                        ids_list.get_item(i)?.extract::<String>()?
                    } else {
                        self.generate_id()
                    }
                }
                None => self.generate_id(),
            };

            // Extract metadata from "metadatas" or "metadata" arrays
            let meta = match dict.get_item("metadatas")?.or_else(|| dict.get_item("metadata").ok().flatten()) {
                Some(item) => {
                    if let Ok(meta_list) = item.downcast::<PyList>() {
                        if i < meta_list.len() {
                            let metadata_item = meta_list.get_item(i)?;
                            if let Ok(meta_dict) = metadata_item.downcast::<PyDict>() {
                                self.python_dict_to_value_map(meta_dict)?
                            } else if metadata_item.is_none() {
                                HashMap::new()
                            } else {
                                let mut map = HashMap::new();
                                let value = self.python_object_to_value(&metadata_item)?;
                                let key = if value.is_string() { "text" } else { "value" };
                                map.insert(key.to_string(), value);
                                map
                            }
                        } else {
                            HashMap::new()
                        }
                    } else {
                        HashMap::new()
                    }
                }
                None => HashMap::new(),
            };

            parsed_vectors.push((id, vector, meta));
        }

        Ok(())
    }

    /// Parse NumPy array with context (IDs and metadata from dict)
    fn parse_numpy_with_context(
        &self,
        np_array: &Bound<PyArray2<f32>>, 
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>
    ) -> PyResult<()> {
        let readonly = np_array.readonly();
        let shape = readonly.shape();

        trace!(operation = "parse_numpy_context", shape = ?shape, "Processing NumPy array with context");

        if shape.len() != 2 || shape[1] != self.dim {
            error!(
                operation = "parse_numpy_context",
                error = "shape_mismatch",
                expected_shape = format!("(N, {})", self.dim),
                actual_shape = format!("{:?}", shape),
                "NumPy array shape validation failed"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "NumPy array must have shape (N, {}), got {:?}", self.dim, shape
            )));
        }

        let flat = readonly.as_slice()?;
        let num_vectors = shape[0];

        // Extract IDs array
        let ids_list = dict.get_item("ids")?
            .and_then(|item| item.downcast::<PyList>().ok().map(|list| list.clone()));
        
        // Extract metadata array
        let metadatas_list = dict.get_item("metadatas")?
            .or_else(|| dict.get_item("metadata").ok().flatten())
            .and_then(|item| item.downcast::<PyList>().ok().map(|list| list.clone()));

        trace!(
            operation = "parse_numpy_context",
            num_vectors = num_vectors,
            has_ids = ids_list.is_some(),
            has_metadata = metadatas_list.is_some(),
            "Processing vectors with context"
        );

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = flat[start_idx..end_idx].to_vec();
            let processed_vector = self.process_vector_for_space(raw_vector);

            // Get ID from provided IDs or generate
            let id = if let Some(ids) = &ids_list {
                if i < ids.len() {
                    ids.get_item(i)?
                        .extract::<String>()
                        .unwrap_or_else(|_| self.generate_id())
                } else {
                    self.generate_id()
                }
            } else {
                self.generate_id()
            };

            // Get metadata from provided metadata or use empty
            let metadata = if let Some(metas) = &metadatas_list {
                if i < metas.len() {
                    let meta_item = metas.get_item(i)?;
                    if let Ok(meta_dict) = meta_item.downcast::<PyDict>() {
                        self.python_dict_to_value_map(meta_dict)?
                    } else {
                        HashMap::new()
                    }
                } else {
                    HashMap::new()
                }
            } else {
                HashMap::new()
            };

            trace!(
                operation = "parse_numpy_vector",
                vector_index = i,
                vector_id = %id,
                metadata_keys = metadata.keys().len(),
                "Parsed NumPy vector with context"
            );

            parsed_vectors.push((id, processed_vector, metadata));
        }

        trace!(operation = "parse_numpy_context_complete", parsed_count = num_vectors, "NumPy parsing completed");
        Ok(())
    }

    /// Safe list parsing that collects errors instead of failing immediately
    fn parse_list_input_safe(
        &self, 
        list: &Bound<PyList>, 
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>
    ) {
        for (item_index, item) in list.iter().enumerate() {
            if let Ok(item_dict) = item.downcast::<PyDict>() {
                // Extract vector safely
                let vector_result = if let Ok(Some(vector_item)) = item_dict.get_item("vector") {
                    self.extract_single_vector_safe(&vector_item)
                } else if let Ok(Some(values_item)) = item_dict.get_item("values") {
                    self.extract_single_vector_safe(&values_item)
                } else {
                    Err("Missing 'vector' or 'values' key in item".to_string())
                };

                match vector_result {
                    Ok(vector) => {
                        // Extract ID
                        let id = match item_dict.get_item("id") {
                            Ok(Some(id_item)) => {
                                id_item.extract::<String>().unwrap_or_else(|_| self.generate_id())
                            }
                            _ => self.generate_id(),
                        };

                        // Extract metadata
                        let metadata = match item_dict.get_item("metadata") {
                            Ok(Some(meta_item)) => {
                                if let Ok(meta_dict) = meta_item.downcast::<PyDict>() {
                                    self.python_dict_to_value_map(meta_dict).unwrap_or_default()
                                } else {
                                    // Handle non-dict metadata
                                    let mut map = HashMap::new();
                                    if let Ok(value) = self.python_object_to_value(&meta_item) {
                                        let key = if value.is_string() { "text" } else { "value" };
                                        map.insert(key.to_string(), value);
                                    }
                                    map
                                }
                            }
                            _ => HashMap::new(),
                        };

                        parsed_vectors.push((id, vector, metadata));
                    }
                    Err(e) => {
                        // Collect error with item index and ID for context
                        let id = item_dict.get_item("id")
                            .ok()
                            .flatten()
                            .and_then(|id_item| id_item.extract::<String>().ok())
                            .unwrap_or_else(|| format!("item_{}", item_index));

                        errors.push(format!("Vector {}: {}", id, e));
                    }
                }
            } else {
                // Direct vector item
                match self.extract_single_vector_safe(&item) {
                    Ok(vector) => {
                        let id = self.generate_id();
                        parsed_vectors.push((id, vector, HashMap::new()));
                    }
                    Err(e) => {
                        errors.push(format!("Item {}: {}", item_index, e));
                    }
                }
            }
        }
    }

    /// Safe NumPy parsing for error collection
    fn parse_numpy_input_safe(&self, np_array: &Bound<PyArray2<f32>>, parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>) -> Result<(), String> {
        // This is the same as your current parse_numpy_input but returns Result<(), String>
        let readonly = np_array.readonly();
        let shape = readonly.shape();

        if shape.len() != 2 || shape[1] != self.dim {
            return Err(format!("NumPy array must have shape (N, {}), got {:?}", self.dim, shape));
        }

        let flat = readonly.as_slice().map_err(|e| format!("NumPy access error: {}", e))?;
        let num_vectors = shape[0];

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = flat[start_idx..end_idx].to_vec();
            let processed_vector = self.process_vector_for_space(raw_vector);
            let id = self.generate_id();
            parsed_vectors.push((id, processed_vector, HashMap::new()));
        }

        Ok(())
    }

    /// Extract a single vector from various Python types (enhanced)
    fn extract_single_vector(&self, data: &Bound<PyAny>) -> PyResult<Vec<f32>> {
        let vector = if let Ok(array1d) = data.downcast::<PyArray1<f32>>() {
            // NumPy 1D array
            array1d.readonly().as_slice()?.to_vec()
        } else if let Ok(list) = data.downcast::<PyList>() {
            // Python list
            list.iter().map(|item| item.extract::<f32>()).collect::<PyResult<Vec<f32>>>()?
        } else {
            // Direct extraction (e.g., from other numeric arrays)
            data.extract::<Vec<f32>>()?
        };

        // Comprehensive validation
        if vector.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Vector cannot be empty"
            ));
        }

        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector dimension mismatch: expected {}, got {}", self.dim, vector.len()
            )));
        }

        // Check for invalid values
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Vector contains invalid value at index {}: {} (must be finite)", i, val
                )));
            }
        }

        // ✅ Apply space-specific processing
        Ok(self.process_vector_for_space(vector))
    }

    /// Generate a unique ID for a vector
    fn generate_id(&self) -> String {
        let id = self.get_next_id();
        format!("vec_{}", id)
    }

    /// Safe version of extract_single_vector that returns String errors instead of PyErr
    fn extract_single_vector_safe(&self, data: &Bound<PyAny>) -> Result<Vec<f32>, String> {
        let vector = if let Ok(array1d) = data.downcast::<PyArray1<f32>>() {
            array1d.readonly().as_slice()
                .map_err(|e| format!("NumPy access error: {}", e))?
                .to_vec()
        } else if let Ok(list) = data.downcast::<PyList>() {
            list.iter()
                .map(|item| item.extract::<f32>()
                    .map_err(|e| format!("List item error: {}", e)))
                .collect::<Result<Vec<f32>, String>>()?
        } else {
            data.extract::<Vec<f32>>()
                .map_err(|e| format!("Vector extraction error: {}", e))?
        };

        // Same validation as extract_single_vector, but with String errors
        if vector.is_empty() {
            return Err("Vector cannot be empty".to_string());
        }
        if vector.len() != self.dim {
            return Err(format!("Vector dimension mismatch: expected {}, got {}", self.dim, vector.len()));
        }
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("Vector contains invalid value at index {}: {}", i, val));
            }
        }

        Ok(self.process_vector_for_space(vector))
    }

    // 4. DATA CONVERSION & FILTERING (12 methods)
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

    // 5. BATCH SEARCH METHODS (3 methods)
    /// Internal batch search method for multiple query vectors
    #[instrument(level = "debug", skip(self, vectors, filter_conditions, py), fields(
        batch_size = vectors.len(),
        top_k = top_k,
        ef = ef,
        return_vector = return_vector,
        has_filter = filter_conditions.is_some()
    ), err)]
    fn batch_search_internal(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
        top_k: usize,
        ef: usize,
        return_vector: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let start_time = Instant::now();

        // Validate all vectors have correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dim {
                error!(
                    operation = "batch_search_validation",
                    vector_index = i,
                    expected_dim = self.dim,
                    actual_dim = vector.len(),
                    "Vector dimension mismatch in batch"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Vector {}: dimension mismatch: expected {}, got {}", 
                           i, self.dim, vector.len())));
            }
        }

        // Choose strategy based on batch size
        let result = if vectors.len() <= 5 {
            trace!(operation = "batch_search_strategy", strategy = "sequential", "Using sequential processing");
            self.batch_search_sequential(vectors, filter_conditions, top_k, ef, return_vector, py)
        } else {
            trace!(operation = "batch_search_strategy", strategy = "parallel", "Using parallel processing");
            self.batch_search_parallel(vectors, filter_conditions, top_k, ef, return_vector, py)
        };

        // ✅ ENTERPRISE: Add duration timing to hot path
        let duration_ms = start_time.elapsed().as_millis();
        debug!(
            operation = "batch_search_complete",
            batch_size = vectors.len(),
            duration_ms = duration_ms,
            "Batch search completed"
        );

        result
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
                // FIX: Process each query vector for space
                let processed_query = self.process_vector_for_space(vector.clone());

                let neighbors = hnsw_guard.search(&processed_query, top_k, ef).unwrap_or_else(|_| Vec::new());

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
        let span = tracing::Span::current();
        let rust_results = py.allow_threads(|| {
            let results: Vec<Vec<(String, f32, HashMap<String, Value>, Option<Vec<f32>>)>> = vectors
                .par_iter()
                .map(|vector| {
                    let _entered = span.clone().entered();
                    // FIX: Process each query vector for space
                    let processed_query = self.process_vector_for_space(vector.clone());

                    // Brief HNSW search (individual lock per query)
                    let neighbors = {
                        let hnsw_guard = self.hnsw.lock().unwrap();
                        hnsw_guard.search(&processed_query, top_k, ef).unwrap_or_else(|_| Vec::new())
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

    // 6. PERSISTENCE INTEGRATION METHODS (2 methods)

    /// Load an index from a .zdb directory structure (Phase 2)
    pub fn load(path: &str) -> PyResult<Self> {
        crate::persistence::load_index(path)
    }

    /// Save HNSW graph using hnsw-rs native file_dump
    #[instrument(level = "info", skip(self), fields(
        vector_count = self.get_vector_count(),
        path = %path.display()
    ))]
    fn save_hnsw_graph(&self, path: &Path) -> PyResult<()> {
        debug!(operation = "save_hnsw_graph_start", "Starting HNSW graph save");

        // EMPTY INDEX CHECK:
        let vector_count = self.get_vector_count();
        if vector_count == 0 {
            debug!(operation = "save_hnsw_graph", reason = "empty_index", "Skipping HNSW graph dump - index is empty");
            return Ok(());
        }

        let hnsw_guard = self.hnsw.lock().unwrap();

        let dump_result = match &*hnsw_guard {
            DistanceType::Cosine(hnsw) => {
                trace!(operation = "save_hnsw_graph", distance_type = "cosine", "Using Cosine distance HNSW");
                hnsw.file_dump(path, "hnsw_index")
            },
            DistanceType::L2(hnsw) => {
                trace!(operation = "save_hnsw_graph", distance_type = "l2", "Using L2 distance HNSW");
                hnsw.file_dump(path, "hnsw_index")
            },
            DistanceType::L1(hnsw) => {
                trace!(operation = "save_hnsw_graph", distance_type = "l1", "Using L1 distance HNSW");
                hnsw.file_dump(path, "hnsw_index")
            },
            DistanceType::CosinePQ(hnsw) => {
                trace!(operation = "save_hnsw_graph", distance_type = "cosine_pq", "Using Cosine-PQ distance HNSW");
                hnsw.file_dump(path, "hnsw_index")
            },
            DistanceType::L2PQ(hnsw) => {
                trace!(operation = "save_hnsw_graph", distance_type = "l2_pq", "Using L2-PQ distance HNSW");
                hnsw.file_dump(path, "hnsw_index")
            },
            DistanceType::L1PQ(hnsw) => {
                trace!(operation = "save_hnsw_graph", distance_type = "l1_pq", "Using L1-PQ distance HNSW");
                hnsw.file_dump(path, "hnsw_index")
            },
        };

        match dump_result {
            Ok(basename) => {
                debug!(
                    operation = "save_hnsw_graph_complete",
                    basename = %basename,
                    files_created = %["hnsw.graph", "hnsw.data"].iter()
                        .map(|ext| format!("{}.{}", basename, ext))
                        .collect::<Vec<_>>()
                        .join(", "),
                    "HNSW graph saved successfully"
                );
                Ok(())
            }
            Err(e) => {
                error!(operation = "save_hnsw_graph", error = %e, "HNSW graph dump failed");
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("HNSW graph dump failed: {}", e)
                ))
            }
        }
    }

    // ============================================================================
    // PERSISTENCE Minimal Empty Constructor and SETTERS
    // ============================================================================
    /// Minimal constructor for persistence loading - creates empty index with config
    /// No validation needed since config comes from trusted saved state
    pub fn new_empty(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
    ) -> Self {
        let space_normalized = space.to_lowercase();
        let max_layer = 16; // Always use NB_LAYER_MAX for consistency
        let hnsw = DistanceType::new_raw(&space_normalized, m, expected_size, max_layer, ef_construction);

        HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            quantization_config: None,
            pq: None,
            pq_codes: RwLock::new(HashMap::new()),
            metadata: Mutex::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: Mutex::new(hnsw),
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            created_at: chrono::Utc::now().to_rfc3339(),
            rebuilding_from_persistence: AtomicBool::new(false),
        }
    }

    /// Set vectors (for persistence loading only)
    pub(crate) fn set_vectors(&mut self, vectors: HashMap<String, Vec<f32>>) {
        *self.vectors.write().unwrap() = vectors;
    }

    /// Set vector metadata (for persistence loading only)
    pub(crate) fn set_vector_metadata(&mut self, metadata: HashMap<String, HashMap<String, serde_json::Value>>) {
        *self.vector_metadata.write().unwrap() = metadata;
    }

    /// Set ID mappings (for persistence loading only)
    pub(crate) fn set_id_mappings(&mut self, id_map: HashMap<String, usize>, rev_map: HashMap<usize, String>) {
        *self.id_map.write().unwrap() = id_map;
        *self.rev_map.write().unwrap() = rev_map;
    }

    /// Set counters (for persistence loading only)
    pub(crate) fn set_counters(&mut self, id_counter: usize, vector_count: usize) {
        *self.id_counter.lock().unwrap() = id_counter;
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Set quantization config (for persistence loading only)
    pub(crate) fn set_quantization_config(&mut self, config: Option<QuantizationConfig>) {
        self.quantization_config = config;
    }

    /// Set PQ instance (for persistence loading only)
    pub(crate) fn set_pq(&mut self, pq: Option<Arc<crate::pq::PQ>>) {
        self.pq = pq;
    }

    /// Set training threshold reached flag (for persistence loading only)
    pub(crate) fn set_training_threshold_reached(&mut self, value: bool) {
        self.training_threshold_reached.store(value, std::sync::atomic::Ordering::Release);
    }

    // ============================================================================
    // PERSISTENCE GETTERS - For accessing private fields from persistence module
    // ============================================================================

    /// Get the vector dimension
    pub fn get_dim(&self) -> usize {
        self.dim
    }

    /// Get the distance space (cosine, l2, l1) - changed to a more idiomatic getter
    pub fn space(&self) -> &str {  // Changed from get_space to space
        &self.space
    }

    /// Get the maximum number of bidirectional links per node
    pub fn get_m(&self) -> usize {
        self.m
    }

    /// Get the construction parameter ef_construction
    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Get the expected size parameter
    pub fn get_expected_size(&self) -> usize {
        self.expected_size
    }

    /// Get the current ID counter value (thread-safe)
    pub fn get_id_counter(&self) -> usize {
        *self.id_counter.lock().unwrap()
    }

    /// Get read access to the vectors HashMap (thread-safe)
    pub fn get_vectors(&self) -> std::sync::RwLockReadGuard<HashMap<String, Vec<f32>>> {
        self.vectors.read().unwrap()
    }

    /// Get read access to the PQ codes HashMap (thread-safe)
    pub fn get_pq_codes(&self) -> std::sync::RwLockReadGuard<HashMap<String, Vec<u8>>> {
        self.pq_codes.read().unwrap()
    }

    /// Get read access to the vector metadata HashMap (thread-safe)
    pub fn get_vector_metadata(&self) -> std::sync::RwLockReadGuard<HashMap<String, HashMap<String, Value>>> {
        self.vector_metadata.read().unwrap()
    }

    /// Get read access to the ID map (external ID -> internal ID)
    pub fn get_id_map(&self) -> std::sync::RwLockReadGuard<HashMap<String, usize>> {
        self.id_map.read().unwrap()
    }

    /// Get read access to the reverse ID map (internal ID -> external ID)
    pub fn get_rev_map(&self) -> std::sync::RwLockReadGuard<HashMap<usize, String>> {
        self.rev_map.read().unwrap()
    }

    /// Get reference to the quantization configuration
    pub fn get_quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    /// Get reference to the PQ instance
    pub fn get_pq(&self) -> Option<&Arc<crate::pq::PQ>> {
        self.pq.as_ref()
    }

    /// Helper to get quantization subvectors count
    pub fn get_quantization_subvectors(&self) -> usize {
        self.quantization_config
            .as_ref()
            .map(|config| config.subvectors)
            .unwrap_or(1)
    }

    /// Get the index creation timestamp
    pub fn get_created_at(&self) -> &str {
        &self.created_at
    }

    /// Get read access to training IDs (for persistence)
    pub fn get_training_ids(&self) -> std::sync::RwLockReadGuard<Vec<String>> {
        self.training_ids.read().unwrap()
    }

    /// Get training threshold reached flag (for persistence)
    pub fn get_training_threshold_reached(&self) -> bool {
        self.training_threshold_reached.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Set training IDs (for persistence loading only)
    pub(crate) fn set_training_ids(&mut self, ids: Vec<String>) {
        *self.training_ids.write().unwrap() = ids;
    }
}
