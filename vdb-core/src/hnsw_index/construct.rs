//! Building an index, and the bounds a declaration has to satisfy.
//!
//! `build` is the only construction path from Python and enforces every rule
//! that governs a valid index, which is what makes the Python factory and the
//! Rust constructor agree. `new_empty` is the loader's constructor and validates
//! nothing, because its configuration comes from a directory this crate wrote.

use super::locks::{MutexAt, RwLockAt};
use super::{HNSWIndex, QuantizationConfig, StorageMode, MAX_LAYER};
use crate::columns::{validate_indexed_fields, ColumnStore};
use crate::graph::VectorGraph;
use crate::pq::PQ;
use chrono::Utc;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
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

/// Every rule a valid index declaration has to satisfy, returning the
/// normalised space.
///
/// Extracted from `build` so that the loader enforces the same rules on the
/// same values rather than a copy of them. `build` takes its declaration from
/// a caller and `load_config` takes it from `config.json`, and a directory is
/// only trusted to the extent that something checked it: a config naming a
/// zero `dim` or a zero `m` used to reach `Backend::sized`, which clamps both
/// silently, so the index came back at a width or a degree the directory never
/// held. The messages are the ones `build` raised, because the invalid value is
/// the same value whichever door it came through.
///
/// `source` prefixes every message and is empty for `build`, whose caller is
/// looking at the argument they passed. The loader passes the path of the file
/// the value came out of, because a caller reading `dim must be positive` off a
/// `load()` has no argument of their own to look at.
pub(crate) fn validate_index_parameters(
    dim: usize,
    space: &str,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    source: &str,
) -> PyResult<String> {
    if dim == 0 {
        error!(
            operation = "validation",
            field = "dim",
            value = dim,
            "Invalid dimension"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}dim must be positive, got {}",
            source, dim
        )));
    }
    if ef_construction == 0 {
        error!(
            operation = "validation",
            field = "ef_construction",
            value = ef_construction,
            "Invalid ef_construction"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}ef_construction must be positive, got {}",
            source, ef_construction
        )));
    }
    if expected_size == 0 {
        error!(
            operation = "validation",
            field = "expected_size",
            value = expected_size,
            "Invalid expected_size"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}expected_size must be positive, got {}",
            source, expected_size
        )));
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
            "{}expected_size must be at most {}, got {}. The graph reserves one \n             slot per declared record at creation, 8 bytes each, so this \n             declaration would ask for {:.1} GB before a single record is \n             added. That allocation is not fallible: above this bound the \n             process aborts rather than raising. expected_size is a capacity \n             hint and not a limit, and under-declaring only costs some \n             reallocation, so declare what you expect to hold.",
            source,
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
            "{}m must be at least 2, got {}. Layer assignment samples from a \n             scale of 1 / ln(m), which is infinity at m 1, so every point \n             overflows the layer cap and is redispatched uniformly across all \n             16 layers instead of following the exponential distribution the \n             graph depends on. Measured on 3,000 records of 32 dimensions, \n             recall at 10 was 0.0220 at m 1 against 0.6880 at m 2 and 1.0000 \n             at m 16.",
            source, m
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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}m must be less than or equal to 256, got {}",
            source, m
        )));
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
                "{}Unsupported space: '{}'. Supported spaces: 'cosine', 'l2', 'l1'",
                source, space
            )));
        }
    }
    Ok(space_normalized)
}

/// Warn where `ef_construction` switches the neighbour selection heuristic off.
///
/// **The message is the one `VectorDatabase.create()` raises, word for word.**
/// The Python factory checks the pair a caller passes to `create()` and this
/// checks the pair a caller passes to `rebuild()`, so the same misconfiguration
/// reads the same either way.
/// `the_two_warnings_are_the_same_sentence` in `tests/test_index_lifecycle.py`
/// holds the two texts equal, which is what stops one drifting from the other.
///
/// The reasoning behind the threshold is in `_warn_if_selection_disabled` in
/// `vector_database.py` and is not repeated here. In short, `select_neighbours`
/// keeps every candidate rather than pruning once the candidate list is no
/// longer than the neighbour budget, that budget is `2 * m` at layer zero, and
/// the candidate list is `ef_construction` long, so the flip lands exactly on
/// `ef_construction <= 2 * m` and the warning carries no slack.
///
/// A pair the validation is going to reject returns without warning, so an
/// invalid `m` produces its real validation error and nothing else.
pub(crate) fn warn_if_selection_disabled(
    py: Python<'_>,
    m: usize,
    ef_construction: usize,
) -> PyResult<()> {
    if !(2..=256).contains(&m) || ef_construction < 1 || ef_construction > 2 * m {
        return Ok(());
    }
    let budget = 2 * m;
    // The largest m that leaves the heuristic running at this ef_construction.
    // Below the floor of 2 there is no such m, so the message offers the one
    // remedy that exists.
    let largest_m = (ef_construction - 1) / 2;
    let remedy = if largest_m >= 2 {
        format!(
            "Raise ef_construction above {}, or lower m to {} or below",
            budget, largest_m
        )
    } else {
        format!("Raise ef_construction above {}", budget)
    };
    let message = format!(
        "ef_construction={} is not greater than 2*m={}, so the neighbour selection \
         heuristic does not run. Layer zero insertion keeps every candidate the \
         construction search returns, in distance order, and prunes none of them. \
         {}, to run it.",
        ef_construction, budget, remedy
    );
    let text = std::ffi::CString::new(message)
        .expect("the message is built here from integers and carries no interior nul");
    PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyUserWarning>(),
        &text,
        1,
    )
}

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
        indexed_fields: Vec<String>,
    ) -> PyResult<Self> {
        let start_time = Instant::now();

        // Validation of parameters. The rules live in
        // `validate_index_parameters`, which the loader calls on the same
        // five values it reads out of `config.json`.
        let space_normalized =
            validate_index_parameters(dim, &space, m, ef_construction, expected_size, "")?;
        // The declaration is checked here for the same reason, and the loader
        // checks what `config.json` carried against the same rules.
        validate_indexed_fields(&indexed_fields, "")?;

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
            indexed_fields = indexed_fields.len(),
            duration_ms = duration_ms,
            "HNSW index created successfully"
        );

        // Initialize all fields with proper thread-safe wrappers
        Ok(HNSWIndex {
            dim,
            space: space_normalized,
            m: AtomicUsize::new(m),
            ef_construction: AtomicUsize::new(ef_construction),
            expected_size: AtomicUsize::new(expected_size),
            quantization_config: quantization_params,
            pq: pq_instance,
            pq_codes: RwLockAt::new(HashMap::new()),
            rerank_calibration: RwLockAt::new(None),
            metadata: MutexAt::new(HashMap::new()),
            vector_metadata: RwLockAt::new(HashMap::new()),
            columns: RwLockAt::new(ColumnStore::new(indexed_fields, expected_size)),
            undeclared_filter_warned: AtomicBool::new(false),
            id_map: RwLockAt::new(HashMap::new()),
            rev_map: RwLockAt::new(HashMap::new()),
            id_counter: MutexAt::new(0),
            vector_count: MutexAt::new(0),
            hnsw: RwLockAt::new(hnsw),
            writers: MutexAt::new(()),
            training_ids: RwLockAt::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            training_completed_at: RwLockAt::new(None),
            created_at: RwLockAt::new(Utc::now().to_rfc3339()),
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
        indexed_fields: Vec<String>,
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
            m: AtomicUsize::new(m),
            ef_construction: AtomicUsize::new(ef_construction),
            expected_size: AtomicUsize::new(expected_size),
            quantization_config: None,
            pq: None,
            pq_codes: RwLockAt::new(HashMap::new()),
            rerank_calibration: RwLockAt::new(None),
            metadata: MutexAt::new(HashMap::new()),
            vector_metadata: RwLockAt::new(HashMap::new()),
            columns: RwLockAt::new(ColumnStore::new(indexed_fields, expected_size)),
            undeclared_filter_warned: AtomicBool::new(false),
            id_map: RwLockAt::new(HashMap::new()),
            rev_map: RwLockAt::new(HashMap::new()),
            id_counter: MutexAt::new(0),
            vector_count: MutexAt::new(0),
            hnsw: RwLockAt::new(hnsw),
            writers: MutexAt::new(()),
            training_ids: RwLockAt::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            // Both are overwritten by the loader from the saved directory, and
            // this is the only path that reaches here. See `set_created_at` and
            // `set_training_completed_at`.
            training_completed_at: RwLockAt::new(None),
            created_at: RwLockAt::new(chrono::Utc::now().to_rfc3339()),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        }
    }
}
