//! Building an index, and the bounds a declaration has to satisfy.
//!
//! `build` is the only construction path from Python and enforces every rule
//! that governs a valid index, which is what makes the Python factory and the
//! Rust constructor agree. `new_empty` is the loader's constructor and validates
//! nothing, because its configuration comes from a directory this crate wrote.

use super::{HNSWIndex, QuantizationConfig, StorageMode, MAX_LAYER};
use crate::graph::VectorGraph;
use crate::pq::PQ;
use chrono::Utc;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tracing::{debug, error, info, instrument, trace};
/// Largest `expected_size` a caller may declare.
///
/// `expected_size` reaches `PointIndexation::new` in the vendored graph crate,
/// which reserves capacity for it across the 16 layers at creation. Since the
/// layer reservation was corrected the reservation is one `Arc` slot per
/// declared record, measured at 8.02 bytes per declared record and flat across
/// declarations of 10 million through 4 billion. This bound therefore caps the
/// creation-time reservation at 764 MB, which was measured rather than derived.
///
/// The bound exists because the reservation is not fallible. `Vec::with_capacity`
/// aborts the process on allocation failure rather than unwinding, so a
/// declaration too large for the machine cannot be turned into a Python
/// exception after the fact. A declared 20 billion asks for 155 GB in the layer
/// zero reservation alone and the process dies with no traceback. Making that
/// path fallible means `try_reserve` inside the vendored crate, which is not a
/// change this package makes.
///
/// One hundred million is far above anything this index holds. A real 100,000
/// record build at 768 dimensions measured 10,617 bytes per record, so a hundred
/// million records is roughly a terabyte of process memory for the data alone.
/// Declaring less than the truth is safe, because a layer that receives more
/// points than reserved grows through the ordinary `Vec::push` path, so capping
/// the declaration costs a caller nothing it can observe.
///
/// The bound is not a guarantee. A machine whose commit limit is below the
/// reservation can still abort at a declaration under it.
const MAX_EXPECTED_SIZE: usize = 100_000_000;
impl HNSWIndex {
    #[instrument(level = "info", skip(quantization_config), fields(
        dim = dim,
        space = %space,
        m = m,
        ef_construction = ef_construction,
        expected_size = expected_size,
        has_quantization = quantization_config.is_some()
    ))]
    pub(crate) fn build(
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
            error!(
                operation = "validation",
                field = "dim",
                value = dim,
                "Invalid dimension"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dim must be positive",
            ));
        }
        if ef_construction == 0 {
            error!(
                operation = "validation",
                field = "ef_construction",
                value = ef_construction,
                "Invalid ef_construction"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ef_construction must be positive",
            ));
        }
        if expected_size == 0 {
            error!(
                operation = "validation",
                field = "expected_size",
                value = expected_size,
                "Invalid expected_size"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "expected_size must be positive",
            ));
        }
        if expected_size > MAX_EXPECTED_SIZE {
            error!(
                operation = "validation",
                field = "expected_size",
                value = expected_size,
                max_allowed = MAX_EXPECTED_SIZE,
                "expected_size exceeds maximum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "expected_size must be at most {}, got {}. The graph reserves one \
                 slot per declared record at creation, 8 bytes each, so this \
                 declaration would ask for {:.1} GB before a single record is \
                 added. That allocation is not fallible: above this bound the \
                 process aborts rather than raising. expected_size is a capacity \
                 hint and not a limit, and under-declaring only costs some \
                 reallocation, so declare what you expect to hold.",
                MAX_EXPECTED_SIZE,
                expected_size,
                (expected_size as f64 * 8.0) / 1_000_000_000.0
            )));
        }
        if m < 2 {
            error!(
                operation = "validation",
                field = "m",
                value = m,
                min_allowed = 2,
                "m below minimum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "m must be at least 2, got {}. Layer assignment samples from a \
                 scale of 1 / ln(m), which is infinity at m 1, so every point \
                 overflows the layer cap and is redispatched uniformly across all \
                 16 layers instead of following the exponential distribution the \
                 graph depends on. Measured on 3,000 records of 32 dimensions, \
                 recall at 10 was 0.0220 at m 1 against 0.6880 at m 2 and 1.0000 \
                 at m 16.",
                m
            )));
        }
        if m > 256 {
            error!(
                operation = "validation",
                field = "m",
                value = m,
                max_allowed = 256,
                "m exceeds maximum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "m must be less than or equal to 256",
            ));
        }

        // Early space validation with user-friendly error
        let space_normalized = space.to_lowercase();
        match space_normalized.as_str() {
            "cosine" | "l2" | "l1" => {
                debug!(operation = "validation", space = %space_normalized, "Distance space validated");
            }
            _ => {
                error!(operation = "validation", field = "space", value = %space, "Unsupported distance space");
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Unsupported space: '{}'. Supported spaces: 'cosine', 'l2', 'l1'",
                    space
                )));
            }
        }

        // Extract quantization configuration
        let (quantization_params, pq_instance) = if let Some(config) = quantization_config {
            let qtype = config
                .get_item("type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'type' in quantization_config",
                    )
                })?
                .extract::<String>()?;

            if qtype != "pq" {
                error!(operation = "validation", field = "quantization_type", value = %qtype, "Unsupported quantization type");
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported quantization type: '{}'. Only 'pq' is currently supported.",
                    qtype
                )));
            }

            // Extract PQ parameters
            let subvectors = config
                .get_item("subvectors")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'subvectors' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let bits = config
                .get_item("bits")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'bits' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let training_size = config
                .get_item("training_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'training_size' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let max_training_vectors = config
                .get_item("max_training_vectors")?
                .map(|v| v.extract::<usize>())
                .transpose()?;

            // Extract storage_mode
            let storage_mode_str = config
                .get_item("storage_mode")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "quantized_only".to_string());

            let storage_mode = StorageMode::from_string(&storage_mode_str)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            // Validate PQ parameters
            if subvectors == 0 {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    value = subvectors,
                    "Subvectors must be positive"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "subvectors must be a positive integer, got 0",
                ));
            }

            if subvectors > dim {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    dim = dim,
                    subvectors = subvectors,
                    "Subvectors exceed dimension"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "subvectors ({}) cannot exceed dimension ({})",
                    subvectors, dim
                )));
            }

            if !dim.is_multiple_of(subvectors) {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    dim = dim,
                    subvectors = subvectors,
                    "Subvectors must divide dimension evenly"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "subvectors ({}) must divide dimension ({}) evenly",
                    subvectors, dim
                )));
            }

            if !(1..=8).contains(&bits) {
                error!(
                    operation = "validation",
                    field = "bits",
                    value = bits,
                    min = 1,
                    max = 8,
                    "Bits out of range"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bits must be between 1 and 8, got {}",
                    bits
                )));
            }

            if training_size < 1000 {
                error!(
                    operation = "validation",
                    field = "training_size",
                    value = training_size,
                    min = 1000,
                    "Training size too small"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "training_size must be at least 1000, got {}",
                    training_size
                )));
            }

            // A max below the threshold produces an index that reaches its
            // training threshold and then fails training on every record from
            // then on, because the cap is already exceeded by the time the
            // trigger fires. Enforced here so it holds on every construction
            // path rather than only the Python factory.
            if let Some(max_training) = max_training_vectors {
                if max_training < training_size {
                    error!(
                        operation = "validation",
                        field = "max_training_vectors",
                        value = max_training,
                        training_size = training_size,
                        "max_training_vectors below training_size"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "max_training_vectors ({}) must be >= training_size ({})",
                        max_training, training_size
                    )));
                }
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
            let pq = Arc::new(PQ::new(
                dim,
                subvectors,
                bits,
                training_size,
                max_training_vectors,
            ));

            (Some(config), Some(pq))
        } else {
            (None, None)
        };

        trace!(
            operation = "hnsw_config",
            max_layer = MAX_LAYER,
            reason = "hnsw-rs compatibility",
            "Using fixed max_layer"
        );

        // Create initial raw HNSW index (will be rebuilt as PQ after training)
        let hnsw = VectorGraph::new_raw(
            &space_normalized,
            dim,
            m,
            expected_size,
            MAX_LAYER,
            ef_construction,
        );

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
            rerank_calibration: RwLock::new(None),
            metadata: Mutex::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: RwLock::new(hnsw),
            writers: Mutex::new(()),
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            training_completed_at: RwLock::new(None),
            created_at: RwLock::new(Utc::now().to_rfc3339()),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        })
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
        let hnsw = VectorGraph::new_raw(
            &space_normalized,
            dim,
            m,
            expected_size,
            MAX_LAYER,
            ef_construction,
        );

        HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            quantization_config: None,
            pq: None,
            pq_codes: RwLock::new(HashMap::new()),
            rerank_calibration: RwLock::new(None),
            metadata: Mutex::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: RwLock::new(hnsw),
            writers: Mutex::new(()),
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            // Both are overwritten by the loader from the saved directory, and
            // this is the only path that reaches here. See `set_created_at` and
            // `set_training_completed_at`.
            training_completed_at: RwLock::new(None),
            created_at: RwLock::new(chrono::Utc::now().to_rfc3339()),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        }
    }
}
