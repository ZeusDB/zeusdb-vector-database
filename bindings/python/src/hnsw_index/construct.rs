//! Parsing a declaration, its quantization mapping and its sparse space,
//! and the one warning that needs the interpreter.
//!
//! The rules a valid index has to satisfy live in `zeusdb_vector_hnsw`, on
//! `Declaration`. What is here is the reading of the quantization mapping's
//! keys, which is Python, interleaved with those rules in the order they have
//! always applied, so the message a caller reads is the one for the first
//! rule their declaration broke, and after them the reading of the sparse
//! space's mapping.

use super::HNSWIndex;
use crate::conversion::python_object_to_value;
use crate::tokenizer::tokenizer_from_python;
use crate::PyEngineError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{json, Value};
use tracing::{error, instrument};
use zeusdb_vector_core::Error;
use zeusdb_vector_hnsw::{Collection, Declaration, SparseConfig, StorageMode};

/// The keys a sparse space declaration may carry. `name` and `tokenizer`
/// are the binding's, and the other three are the fields of `SparseConfig`
/// as `config.json` writes them.
const SPARSE_KEYS: [&str; 5] = [
    "name",
    "unlink",
    "lazy_threshold_percent",
    "weighting",
    "tokenizer",
];

/// The three keys the sparse index's own declaration is read from.
const SPARSE_CONFIG_KEYS: [&str; 3] = ["unlink", "lazy_threshold_percent", "weighting"];

/// The name a sparse space takes where the declaration gives none.
const DEFAULT_SPARSE_NAME: &str = "sparse";

/// Declare the sparse space a `create()` mapping describes.
///
/// **One mapping, which is the sparse crate's own.** `unlink`, `weighting`
/// and the tokenizer's name are read by the serde derive that reads
/// `config.json`, so the spellings a caller writes are the spellings a
/// saved directory carries and there is no second table of them here. A
/// field left out takes its default, which is what the derive gives it,
/// with one exception made here: `weighting` left out beside a `tokenizer`
/// takes `bm25`, since a text layer stores term counts and the term
/// frequency rule is the one that reads a count as a count, where the
/// derive alone would give `dot`. Every other combination is read as
/// written, so a declaration without a tokenizer defaults to `dot` and
/// `dot` beside a tokenizer is what it says. A key that is not a field is
/// refused by name. A weighting given as a bare string is read as
/// `{"type": <string>}`, so `"bm25"` is the published defaults and
/// `{"type": "bm25", "k1": 1.5, "b": 0.6}` sets them.
///
/// `name` is the space's name, which is the directory its artefacts are
/// saved under; it defaults to `sparse`. `tokenizer` is `"simple"` for the
/// built-in tokenizer or a callable of the caller's own, and its presence
/// is what declares the text layer. A value of `None` under any key is the
/// key left out.
fn declare_sparse(
    declaration: Declaration,
    sparse: &Bound<PyDict>,
) -> Result<Declaration, PyEngineError> {
    for key in sparse.keys() {
        let key = key.extract::<String>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "sparse declaration keys must be strings",
            )
        })?;
        if !SPARSE_KEYS.contains(&key.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "sparse declares '{}', which is not a field of a sparse space. The fields are name, weighting, unlink, lazy_threshold_percent and tokenizer.",
                key
            ))
            .into());
        }
    }
    let present = |key: &str| -> PyResult<Option<Bound<PyAny>>> {
        Ok(sparse.get_item(key)?.filter(|value| !value.is_none()))
    };
    let name = match present("name")? {
        Some(value) => value.extract::<String>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "sparse['name'] must be a str, got {}",
                value
                    .get_type()
                    .name()
                    .map(|name| name.to_string())
                    .unwrap_or_default()
            ))
        })?,
        None => DEFAULT_SPARSE_NAME.to_string(),
    };
    let tokenizer = present("tokenizer")?;
    let mut fields = serde_json::Map::new();
    for key in SPARSE_CONFIG_KEYS {
        if let Some(value) = present(key)? {
            let mut value = python_object_to_value(&value)?;
            if key == "weighting" {
                if let Value::String(rule) = &value {
                    value = json!({ "type": rule });
                }
            }
            fields.insert(key.to_string(), value);
        }
    }
    // The one default the derive does not give. A text layer stores term
    // counts, and the term frequency rule reads a count as a count, so a
    // tokenizer with no weighting named takes it at its published
    // parameters. `config.json` names the weighting in full, so a saved
    // space reads what was written and never takes this default.
    if tokenizer.is_some() && !fields.contains_key("weighting") {
        fields.insert("weighting".to_string(), json!({ "type": "bm25" }));
    }
    let config: SparseConfig = serde_json::from_value(Value::Object(fields)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "The sparse space declaration could not be read: {}",
            e
        ))
    })?;
    match tokenizer {
        None => Ok(declaration.with_sparse(&name, config)?),
        Some(argument) => {
            let tokenizer = tokenizer_from_python(&argument, "sparse['tokenizer']")?;
            Ok(declaration.with_text(&name, config, tokenizer)?)
        }
    }
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
/// mapping's `type` is read. Under `pq` the space is checked for
/// quantization before any other key is read, then the keys are read one at
/// a time and the product quantizer's own rules run on what they held, each
/// rule once and in the order it always did. Under `int8` the keys the
/// scheme does not take are refused first, then `scale`, `training_size`,
/// `max_training_vectors` and `storage_mode` are read and the scalar rules
/// run on them; a scalar index takes all four spaces. The sparse space is
/// declared after all of them, so a caller reads the same message for the
/// same mistake whether or not one is declared.
#[instrument(level = "info", skip(quantization_config, sparse), fields(
    dim = dim,
    space = %space,
    m = m,
    ef_construction = ef_construction,
    expected_size = expected_size,
    has_quantization = quantization_config.is_some(),
    has_sparse = sparse.is_some()
))]
// The argument list is the Python signature's, one value per keyword.
#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    quantization_config: Option<&Bound<PyDict>>,
    indexed_fields: Vec<String>,
    sparse: Option<&Bound<PyDict>>,
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
            let qtype = config
                .get_item("type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'type' in quantization_config",
                    )
                })?
                .extract::<String>()?;

            if qtype == "int8" {
                return build_int8(declaration, config, sparse);
            }
            if qtype != "pq" {
                error!(operation = "validation", field = "quantization_type", value = %qtype, "Unsupported quantization type");
                return Err(Error::UnsupportedQuantizationType { qtype }.into());
            }

            // Before any of the product quantized keys is read, so the
            // message names the pair rather than whichever PQ field happens
            // to be checked first.
            declaration.quantizable()?;
            refuse_key(config, "scale", "pq", "int8")?;

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

    let declaration = match sparse {
        Some(sparse) => declare_sparse(declaration, sparse)?,
        None => declaration,
    };

    Ok(HNSWIndex::wrap(Collection::build(
        declaration,
        quantization,
    )))
}

/// Refuse a key of the other scheme, where the mapping carries it with a
/// value. A value of `None` is the key left out, as it is for every other
/// key of the mapping.
fn refuse_key(
    config: &Bound<PyDict>,
    key: &str,
    scheme: &str,
    belongs_to: &'static str,
) -> Result<(), PyEngineError> {
    if config.get_item(key)?.is_some_and(|value| !value.is_none()) {
        error!(
            operation = "validation",
            field = key,
            quantization_type = scheme,
            "Key of the other quantization scheme"
        );
        return Err(Error::QuantizationKeyNotForScheme {
            key: key.to_string(),
            scheme: scheme.to_string(),
            belongs_to,
        }
        .into());
    }
    Ok(())
}

/// The scalar half of [`build`], for a mapping whose `type` is `int8`.
///
/// `subvectors` and `bits` are refused first, since a caller who carried
/// them over from a product quantized declaration should hear that before
/// any rule about the keys the scheme does take. `scale` defaults to
/// `per_dimension`, `training_size` is required as it is under `pq`, and
/// `storage_mode` defaults to `quantized_only`, the one mode a scalar index
/// takes; see `Declaration::scalar_quantization` for the rules and the
/// reason.
fn build_int8(
    declaration: Declaration,
    config: &Bound<PyDict>,
    sparse: Option<&Bound<PyDict>>,
) -> Result<HNSWIndex, PyEngineError> {
    refuse_key(config, "subvectors", "int8", "pq")?;
    refuse_key(config, "bits", "int8", "pq")?;

    let scale = config
        .get_item("scale")?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "per_dimension".to_string());

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

    let storage_mode_str = config
        .get_item("storage_mode")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "quantized_only".to_string());
    let storage_mode =
        StorageMode::from_string(&storage_mode_str).map_err(Error::InvalidStorageMode)?;

    let quantization = declaration.scalar_quantization(
        &scale,
        training_size,
        max_training_vectors,
        storage_mode,
    )?;

    let declaration = match sparse {
        Some(sparse) => declare_sparse(declaration, sparse)?,
        None => declaration,
    };

    Ok(HNSWIndex::wrap(Collection::build(
        declaration,
        Some(quantization),
    )))
}
