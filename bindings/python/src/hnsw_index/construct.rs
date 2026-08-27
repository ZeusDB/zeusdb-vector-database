//! Parsing a declaration and its quantization mapping, and the one warning
//! that needs the interpreter.
//!
//! The rules a valid index has to satisfy live in `zeusdb_vector_hnsw`, on
//! `Declaration`. What is here is the reading of the quantization mapping's
//! keys, which is Python, interleaved with those rules in the order they have
//! always applied, so the message a caller reads is the one for the first
//! rule their declaration broke.

use super::HNSWIndex;
use crate::PyEngineError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tracing::{error, instrument};
use zeusdb_vector_core::Error;
use zeusdb_vector_hnsw::{Collection, Declaration, StorageMode};

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
pub(super) fn warn_if_selection_disabled(
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

/// Build an index from the arguments `_create_hnsw_index` receives.
///
/// The five values and the field declaration are validated first, then the
/// space is checked for quantization before any of the mapping is read, then
/// the mapping's keys are read one at a time and the product quantizer's own
/// rules run on what they held. Each rule runs once and in the order it
/// always did.
#[instrument(level = "info", skip(quantization_config), fields(
    dim = dim,
    space = %space,
    m = m,
    ef_construction = ef_construction,
    expected_size = expected_size,
    has_quantization = quantization_config.is_some()
))]
pub(super) fn build(
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    quantization_config: Option<&Bound<PyDict>>,
    indexed_fields: Vec<String>,
) -> Result<HNSWIndex, PyEngineError> {
    let declaration = Declaration::validate(
        dim,
        &space,
        m,
        ef_construction,
        expected_size,
        indexed_fields,
    )?;

    // Extract quantization configuration
    let quantization = match quantization_config {
        Some(config) => {
            // Before any of the config is read, so the message names the pair
            // rather than whichever PQ field happens to be checked first.
            declaration.quantizable()?;
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
                return Err(Error::UnsupportedQuantizationType { qtype }.into());
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

            let storage_mode =
                StorageMode::from_string(&storage_mode_str).map_err(Error::InvalidStorageMode)?;

            Some(declaration.quantization(
                subvectors,
                bits,
                training_size,
                max_training_vectors,
                storage_mode,
            )?)
        }
        None => None,
    };

    Ok(HNSWIndex::wrap(Collection::build(
        declaration,
        quantization,
    )))
}
