//! Reading records out of Python.
//!
//! `add` accepts five input shapes and everything that turns any of them into
//! owned Rust lives here. What leaves this file is a list of [`Parsed`]
//! records, which hold no Python object, which is what lets insertion
//! release the interpreter lock. Every vector is processed for the space on
//! the way through, by the collection's own rule, so a record is stored as
//! the engine expects it. What a query vector is checked against once it is
//! out of Python is the engine's half of this module,
//! `Collection::validate_query`, and its records carry this module's target.
//!
//! # The sparse half of a record
//!
//! A record may fill the collection's sparse space as well as the dense
//! one, under one of two keys. `sparse` is a mapping `{"dims": [...],
//! "values": [...]}` of term ids and weights, parallel and sorted by id,
//! and `text` is a string the space's text layer counts into term ids. In
//! the single object and list shapes each record carries its own; in the
//! batch shapes `sparse` and `texts` are parallel arrays beside `ids`,
//! with `None` for a record that fills the dense space alone, and they are
//! held to the length rule the other parallel arrays are held to. A record
//! carries one or the other. Which of the two the space takes is the
//! space's rule and is applied in `add` after parsing, where a text is
//! tokenized with the interpreter lock held; see the crate's `tokenizer`
//! module. Every shape without either key parses as it always did.
//!
//! A record's vector is read by `extract_single_vector_safe`, which returns
//! the words `add` reports against the record it belongs to, so a bad vector
//! in any shape is that record's error and the records around it go in.
//!
//! # A record's metadata
//!
//! Every shape reads a record's metadata entry by one rule, `metadata_entry`.
//! A mapping is converted key by key, a string is kept under `text`, any
//! other value is kept under `value`, and `None` or an absent entry is no
//! metadata. A record whose entry does not convert, through a key that is not
//! a string or a value nested past `conversion::MAX_VALUE_DEPTH` levels, is
//! that record's error, named `Metadata error` with the conversion's own
//! words, and is found before the record takes a generated id.

use super::HNSWIndex;
use crate::conversion::{python_dict_to_value_map, python_object_to_value};
use crate::PyEngineError;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PySequence, PyString};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use tracing::{error, trace};
use zeusdb_vector_core::{Error, SparseVector};

/// A record's sparse half as it arrived: nothing, a vector of term ids and
/// weights, or a text the space's text layer counts.
pub(super) enum SparseInput {
    None,
    Vector(SparseVector),
    Text(String),
}

/// One record out of Python, with the vector processed for the space and
/// the sparse half not yet counted.
pub(super) struct Parsed {
    pub(super) id: String,
    pub(super) vector: Vec<f32>,
    pub(super) metadata: HashMap<String, Value>,
    pub(super) sparse: SparseInput,
}

impl Parsed {
    fn dense(id: String, vector: Vec<f32>, metadata: HashMap<String, Value>) -> Self {
        Parsed {
            id,
            vector,
            metadata,
            sparse: SparseInput::None,
        }
    }
}

/// The two keys a sparse vector mapping carries.
const SPARSE_VECTOR_KEYS: [&str; 2] = ["dims", "values"];

/// A sparse vector out of a mapping `{"dims": [...], "values": [...]}`.
///
/// The dims are read as integers and held to the width of a term id, the
/// values as numbers, from a list or a one dimensional array of either. The
/// rules the vector itself has to satisfy, being parallel slices, strictly
/// increasing dims and finite values, are the engine's and are applied
/// where the vector is used, so a malformed vector is refused with the
/// engine's message wherever it arrives.
pub(super) fn extract_sparse_vector(value: &Bound<PyAny>) -> Result<SparseVector, String> {
    let dict = value.cast::<PyDict>().map_err(|_| {
        format!(
            "'sparse' must be a mapping {{'dims': [...], 'values': [...]}}, got {}",
            value
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_default()
        )
    })?;
    for key in dict.keys() {
        let key = key
            .extract::<String>()
            .map_err(|_| "'sparse' keys must be strings".to_string())?;
        if !SPARSE_VECTOR_KEYS.contains(&key.as_str()) {
            return Err(format!(
                "'sparse' carries '{}', and a sparse vector is {{'dims': [...], 'values': [...]}}",
                key
            ));
        }
    }
    let dims = dict.get_item("dims").ok().flatten().ok_or_else(|| {
        "'sparse' is missing 'dims', the term ids in increasing order".to_string()
    })?;
    let values = dict
        .get_item("values")
        .ok()
        .flatten()
        .ok_or_else(|| "'sparse' is missing 'values', one weight per dim".to_string())?;
    let dims: Vec<i64> = dims
        .extract()
        .map_err(|_| "'sparse' dims must be a list of non-negative integers".to_string())?;
    let dims: Vec<u32> = dims
        .into_iter()
        .map(|dim| {
            u32::try_from(dim).map_err(|_| {
                format!(
                    "'sparse' dim {} is outside 0 to {}, which is the width of a term id",
                    dim,
                    u32::MAX
                )
            })
        })
        .collect::<Result<Vec<u32>, String>>()?;
    let values: Vec<f32> = values
        .extract()
        .map_err(|_| "'sparse' values must be a list of numbers".to_string())?;
    Ok(SparseVector { dims, values })
}

/// The sparse half a record carries under `sparse` and `text`, given the
/// two values where present. `None` under either is the key left out.
fn sparse_input(
    sparse: Option<&Bound<PyAny>>,
    text: Option<&Bound<PyAny>>,
) -> Result<SparseInput, String> {
    let sparse = sparse.filter(|value| !value.is_none());
    let text = text.filter(|value| !value.is_none());
    match (sparse, text) {
        (None, None) => Ok(SparseInput::None),
        (Some(_), Some(_)) => Err(
            "a record fills the sparse space with 'sparse' or with 'text', not both".to_string(),
        ),
        (Some(vector), None) => extract_sparse_vector(vector).map(SparseInput::Vector),
        (None, Some(text)) => text
            .extract::<String>()
            .map(SparseInput::Text)
            .map_err(|_| {
                format!(
                    "'text' must be a str, got {}",
                    text.get_type()
                        .name()
                        .map(|name| name.to_string())
                        .unwrap_or_default()
                )
            }),
    }
}

/// The sparse half a record mapping carries.
fn sparse_input_of_record(dict: &Bound<PyDict>) -> Result<SparseInput, String> {
    let sparse = dict.get_item("sparse").ok().flatten();
    let text = dict.get_item("text").ok().flatten();
    sparse_input(sparse.as_ref(), text.as_ref())
}

/// The sparse half at position `i` of a batch, from the `sparse` and
/// `texts` arrays where the batch carries them. Both are lists by the time
/// this runs, since `check_batch_lengths` refused anything else.
fn sparse_input_at(
    sparse_list: Option<&Bound<PyList>>,
    texts_list: Option<&Bound<PyList>>,
    i: usize,
) -> Result<SparseInput, String> {
    let sparse = sparse_list.and_then(|list| list.get_item(i).ok());
    let text = texts_list.and_then(|list| list.get_item(i).ok());
    sparse_input(sparse.as_ref(), text.as_ref())
}

/// A batch's parallel array under `key`, where it is a list.
fn parallel_list<'py>(dict: &Bound<'py, PyDict>, key: &str) -> Option<Bound<'py, PyList>> {
    dict.get_item(key)
        .ok()
        .flatten()
        .and_then(|item| item.cast::<PyList>().ok().cloned())
}

/// A two dimensional array of `f32` or of `f64`, as `add` reads one. An
/// `f64` array is narrowed to `f32` in one pass, as `search` narrows one, and
/// the narrowing is the rounding `extract::<f32>` applies to each element.
enum Rows<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

impl<'py> Rows<'py> {
    /// The array `value` is, where it is a two dimensional array of either.
    fn of(value: &Bound<'py, PyAny>) -> Option<Rows<'py>> {
        if let Ok(array) = value.cast::<PyArray2<f32>>() {
            return Some(Rows::F32(array.readonly()));
        }
        if let Ok(array) = value.cast::<PyArray2<f64>>() {
            return Some(Rows::F64(array.readonly()));
        }
        None
    }

    fn shape(&self) -> &[usize] {
        match self {
            Rows::F32(array) => array.shape(),
            Rows::F64(array) => array.shape(),
        }
    }

    /// Every value, row after row, refusing an array whose rows are not
    /// `dim` wide with `BatchArrayShape`, and one not contiguous in memory
    /// with the `TypeError` `search` raises for it.
    fn flat(&self, dim: usize) -> Result<Cow<'_, [f32]>, PyEngineError> {
        let shape = self.shape();
        if shape.len() != 2 || shape[1] != dim {
            error!(
                operation = "parse_numpy",
                error = "shape_mismatch",
                expected_shape = format!("(N, {})", dim),
                actual_shape = format!("{:?}", shape),
                "NumPy array shape validation failed"
            );
            return Err(Error::BatchArrayShape {
                dim,
                shape: shape.to_vec(),
            }
            .into());
        }
        Ok(match self {
            Rows::F32(array) => Cow::Borrowed(array.as_slice()?),
            Rows::F64(array) => Cow::Owned(
                array
                    .as_slice()?
                    .iter()
                    .map(|&value| value as f32)
                    .collect(),
            ),
        })
    }
}

/// Whether `value` is shaped as one vector: a sequence other than a string,
/// or a NumPy array. What `add` reads as a vector, so a value of this shape
/// that is not one is that record's error, and a value of any other shape is
/// not a record at all.
fn shaped_as_a_vector(value: &Bound<PyAny>) -> bool {
    if value.is_instance_of::<PyString>() || value.is_instance_of::<PyBytes>() {
        return false;
    }
    value.cast::<PySequence>().is_ok() || value.cast::<PyUntypedArray>().is_ok()
}

/// The name of `value`'s type, for a refusal.
fn type_name(value: &Bound<PyAny>) -> String {
    value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default()
}

/// The words `errors` names a record's metadata with where it does not
/// convert, after the record's label.
fn metadata_error(e: PyErr) -> String {
    format!("Metadata error: {}", e)
}

/// A record's metadata entry, read by the one rule every shape keeps: a
/// mapping converted key by key, a string kept under `text`, any other value
/// kept under `value`, and `None` as no metadata. An entry that does not
/// convert, through a key that is not a string or a value nested past
/// `conversion::MAX_VALUE_DEPTH` levels, is the record's error.
fn metadata_entry(entry: &Bound<PyAny>) -> Result<HashMap<String, Value>, String> {
    if entry.is_none() {
        return Ok(HashMap::new());
    }
    let converted = match entry.cast::<PyDict>() {
        Ok(mapping) => python_dict_to_value_map(mapping),
        Err(_) => python_object_to_value(entry).map(|value| {
            let key = if value.is_string() { "text" } else { "value" };
            HashMap::from([(key.to_string(), value)])
        }),
    };
    converted.map_err(metadata_error)
}

/// The metadata a record dict carries under `metadata`, read by
/// `metadata_entry`, and none where the key is absent.
fn record_dict_metadata(dict: &Bound<PyDict>) -> Result<HashMap<String, Value>, String> {
    match dict.get_item("metadata") {
        Ok(Some(entry)) => metadata_entry(&entry),
        _ => Ok(HashMap::new()),
    }
}

/// What a batch dict carries its vectors under, and as what.
enum BatchVectors<'py> {
    List(Bound<'py, PyList>),
    Rows(Rows<'py>),
}

impl BatchVectors<'_> {
    /// How many records the batch holds.
    fn count(&self) -> usize {
        match self {
            BatchVectors::List(list) => list.len(),
            BatchVectors::Rows(rows) => rows.shape().first().copied().unwrap_or(0),
        }
    }
}

impl HNSWIndex {
    /// Parse input data into records with error collection
    ///
    /// Two error channels, and which one a mistake takes is the distinction
    /// this function exists to draw. A bad record is collected into `errors`
    /// and reported through `AddResult`, because the other records in the
    /// batch are still what the caller asked for. Each refused record is
    /// named once, so the records parsed and the errors collected number the
    /// records the input holds, which is the count `vector_shape` reports. A
    /// malformed call raises before anything is inserted, because there is no
    /// record set to report against: an input that is none of the formats, a
    /// list holding an item that is neither a record dict nor a vector, a
    /// batch dict with no vector field or with one of the wrong type, a
    /// parallel array that is not a list or not the vectors' length, and a
    /// two dimensional array whose rows are not the index's width.
    pub(super) fn parse_input_data(
        &self,
        data: &Bound<PyAny>,
    ) -> PyResult<(Vec<Parsed>, Vec<String>)> {
        let mut parsed_vectors = Vec::new();
        let mut errors = Vec::new();

        if let Ok(dict) = data.cast::<PyDict>() {
            self.parse_dict_input_safe(dict, &mut parsed_vectors, &mut errors)?;
        } else if let Ok(list) = data.cast::<PyList>() {
            self.parse_list_input_safe(list, &mut parsed_vectors, &mut errors)?;
        } else if let Some(rows) = Rows::of(data) {
            self.parse_numpy_input_safe(&rows, &mut parsed_vectors, &mut errors)?;
        } else if shaped_as_a_vector(data) {
            // Single vector
            match self.extract_single_vector_safe(data) {
                Ok(vector) => {
                    let id = self.inner.generate_id();
                    parsed_vectors.push(Parsed::dense(id, vector, HashMap::new()));
                }
                Err(e) => {
                    errors.push(format!("Single vector error: {}", e));
                }
            }
        } else {
            error!(
                operation = "add_vectors",
                error = "unrecognised_input",
                input_type = %type_name(data),
                "Input is none of the formats add takes"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "add() takes a record dict, a list of records, a dict of parallel arrays, a \
                 two dimensional array or one vector, and got {}",
                type_name(data)
            )));
        }

        Ok((parsed_vectors, errors))
    }

    /// Safe dictionary parsing that collects errors
    ///
    /// Returns `Err` for the rules a batch dict is held to as a whole, being
    /// its vector field and its parallel arrays. Every failure of one record
    /// is collected.
    fn parse_dict_input_safe(
        &self,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<Parsed>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        // Check for single object format
        if dict.contains("id").unwrap_or(false)
            && (dict.contains("values").unwrap_or(false)
                || dict.contains("vector").unwrap_or(false))
        {
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
                    let provided = dict
                        .get_item("id")
                        .ok()
                        .flatten()
                        .and_then(|id_item| id_item.extract::<String>().ok());

                    // Read before the record takes a generated id, so a record
                    // refused for its metadata takes none.
                    let metadata = match record_dict_metadata(dict) {
                        Ok(metadata) => metadata,
                        Err(e) => {
                            let label = provided.unwrap_or_else(|| "single_object".to_string());
                            errors.push(format!("Vector {}: {}", label, e));
                            return Ok(());
                        }
                    };

                    let id = provided.unwrap_or_else(|| self.inner.generate_id());

                    match sparse_input_of_record(dict) {
                        Ok(sparse) => parsed_vectors.push(Parsed {
                            id,
                            vector,
                            metadata,
                            sparse,
                        }),
                        Err(e) => errors.push(format!("Vector {}: {}", id, e)),
                    }
                }
                Err(e) => {
                    let id = dict
                        .get_item("id")
                        .ok()
                        .flatten()
                        .and_then(|id_item| id_item.extract::<String>().ok())
                        .unwrap_or_else(|| "single_object".to_string());

                    errors.push(format!("Vector {}: {}", id, e));
                }
            }
        } else {
            // Batch format. The vector field and the parallel arrays are held
            // to their rules before any record is parsed, and a breach raises
            // rather than being collected. Everything else the batch can get
            // wrong is a property of one record and is reported per record.
            let (vector_field, vectors) = Self::batch_vectors(dict)?;
            self.check_batch_lengths(dict, vector_field, vectors.count())?;
            match vectors {
                BatchVectors::List(list) => {
                    self.process_vector_list(&list, dict, parsed_vectors, errors)?
                }
                BatchVectors::Rows(rows) => {
                    self.parse_numpy_with_context(&rows, dict, parsed_vectors, errors)?
                }
            }
        }

        Ok(())
    }

    /// The field a batch dict carries its vectors under, and the vectors.
    ///
    /// The first of `vectors`, `embeddings` and `values` the dict holds is the
    /// field. Under `vectors` or `embeddings` it is a list of vectors or a two
    /// dimensional array of `f32` or `f64`, and under `values` a list. A dict
    /// holding none of the three raises `ValueError`, a field of any other
    /// type under the first two raises `TypeError` naming the field, and one
    /// under `values` raises `ValueError`.
    fn batch_vectors<'py>(
        dict: &Bound<'py, PyDict>,
    ) -> PyResult<(&'static str, BatchVectors<'py>)> {
        for field in ["vectors", "embeddings", "values"] {
            let Some(item) = dict.get_item(field)? else {
                continue;
            };
            if let Ok(list) = item.cast::<PyList>() {
                return Ok((field, BatchVectors::List(list.clone())));
            }
            if field == "values" {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "values field must be a list in batch format",
                ));
            }
            if let Some(rows) = Rows::of(&item) {
                return Ok((field, BatchVectors::Rows(rows)));
            }
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "add expected '{}' to be a list of vectors or a two dimensional array of \
                 float32 or float64, got {}",
                field,
                type_name(&item)
            )));
        }
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Missing vector data. Expected one of: 'vectors', 'embeddings', or 'values' key",
        ))
    }

    /// The rule on a batch's parallel arrays, which raises rather than
    /// counting an error
    ///
    /// A batch dict pairs its arrays by position, so `ids[i]` names `vectors[i]`
    /// and `metadatas[i]` describes it. Nothing in the shape says how long each
    /// is, and until this check existed a disagreement was absorbed in both
    /// directions. Three ids against two vectors inserted two records under the
    /// first two ids and dropped the third, and two ids against three vectors
    /// inserted the third under a generated `vec_N`. Both reported
    /// `inserted=n, errors=0`.
    ///
    /// It raises rather than rejecting a record, because which record the caller
    /// meant is unrecoverable. A short `ids` does not say whether the trailing
    /// vectors were surplus or the ids were, and a per-record rejection would
    /// have to guess. Every other mistake `add` reports is a property of one
    /// record, and the surrounding records are still what the caller asked for.
    ///
    /// A parallel array that is not a list is the same loss by another route,
    /// and raises `TypeError` naming the field, under every field. Only a list
    /// keeps the pairing by position, so a tuple or an `ndarray` under any
    /// field is refused whole, before any record takes an id or loses its
    /// metadata.
    fn check_batch_lengths(
        &self,
        dict: &Bound<PyDict>,
        vector_field: &str,
        vector_count: usize,
    ) -> PyResult<()> {
        // 'metadata' is read as a spelling of 'metadatas' by both batch parsers,
        // and only where 'metadatas' is absent, so it is checked under the same
        // rule and only in the same case.
        let metadata_field = if dict.get_item("metadatas")?.is_some() {
            "metadatas"
        } else {
            "metadata"
        };

        // `sparse` and `texts` are the sparse half's parallel arrays, held to
        // the same rule.
        for (field, singular) in [
            ("ids", "id"),
            (metadata_field, "metadata mapping"),
            ("sparse", "sparse vector"),
            ("texts", "text"),
        ] {
            let Some(item) = dict.get_item(field)? else {
                continue;
            };
            let Ok(list) = item.cast::<PyList>() else {
                let tail = if field == "ids" {
                    "so any other type is discarded whole and every record takes a generated id"
                        .to_string()
                } else if field == metadata_field {
                    "with None for a record that carries no metadata".to_string()
                } else {
                    "with None for a record that fills the dense space alone".to_string()
                };
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "add expected '{}' to be a list, got {}. A batch pairs '{}' with '{}' \
                     by position and reads only a list that way, {}.",
                    field,
                    item.get_type().name()?,
                    field,
                    vector_field,
                    tail
                )));
            };
            if list.len() != vector_count {
                let short = if list.len() < vector_count {
                    field
                } else {
                    vector_field
                };
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "add received {} entries under '{}' and {} under '{}'. A batch pairs \
                     them by position, so the two must be the same length, and '{}' is \
                     the short one. Supply one {} per vector, or omit '{}' entirely.",
                    list.len(),
                    field,
                    vector_count,
                    vector_field,
                    short,
                    singular,
                    field
                )));
            }
        }

        Ok(())
    }

    /// The id a batch's `ids` give the record at position `i`, where they hold
    /// a string there. A record whose entry is not a string takes a
    /// generated id, as a record dict whose `id` is not a string does.
    fn provided_id(ids: Option<&Bound<PyList>>, i: usize) -> PyResult<Option<String>> {
        Ok(match ids {
            Some(ids) if i < ids.len() => ids.get_item(i)?.extract::<String>().ok(),
            _ => None,
        })
    }

    /// The batch's metadata list, under `metadatas` or else `metadata`, a
    /// list by the time this runs, since `check_batch_lengths` refused any
    /// other type.
    fn batch_metadatas<'py>(dict: &Bound<'py, PyDict>) -> PyResult<Option<Bound<'py, PyList>>> {
        Ok(dict
            .get_item("metadatas")?
            .or_else(|| dict.get_item("metadata").ok().flatten())
            .and_then(|item| item.cast::<PyList>().ok().cloned()))
    }

    /// The metadata a batch gives the record at position `i`, under its list
    /// of vectors or its array alike: its entry as `metadata_entry` reads it,
    /// and none for a record past the list.
    fn batch_metadata(
        metadatas: Option<&Bound<PyList>>,
        i: usize,
    ) -> Result<HashMap<String, Value>, String> {
        let Some(metadatas) = metadatas.filter(|metadatas| i < metadatas.len()) else {
            return Ok(HashMap::new());
        };
        metadata_entry(&metadatas.get_item(i).map_err(metadata_error)?)
    }

    /// Every record of a batch whose vectors are a list, each parsed on its
    /// own, so a refused record is named, by its id or by its position as
    /// `item_N`, and the records after it are parsed as the records before
    /// it were. A record takes a generated id once every check on it has
    /// passed.
    fn process_vector_list(
        &self,
        vectors: &Bound<PyList>,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<Parsed>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        let ids_list = parallel_list(dict, "ids");
        let metadatas_list = Self::batch_metadatas(dict)?;
        let sparse_list = parallel_list(dict, "sparse");
        let texts_list = parallel_list(dict, "texts");

        for (i, vector_item) in vectors.iter().enumerate() {
            let provided = Self::provided_id(ids_list.as_ref(), i)?;
            let label = || provided.clone().unwrap_or_else(|| format!("item_{}", i));
            let vector = match self.extract_single_vector_safe(&vector_item) {
                Ok(vector) => vector,
                Err(e) => {
                    errors.push(format!("Vector {}: {}", label(), e));
                    continue;
                }
            };
            let metadata = match Self::batch_metadata(metadatas_list.as_ref(), i) {
                Ok(metadata) => metadata,
                Err(e) => {
                    errors.push(format!("Vector {}: {}", label(), e));
                    continue;
                }
            };
            let sparse = match sparse_input_at(sparse_list.as_ref(), texts_list.as_ref(), i) {
                Ok(sparse) => sparse,
                Err(e) => {
                    errors.push(format!("Vector {}: {}", label(), e));
                    continue;
                }
            };
            let id = match provided {
                Some(id) => id,
                None => self.inner.generate_id(),
            };
            parsed_vectors.push(Parsed {
                id,
                vector,
                metadata,
                sparse,
            });
        }

        Ok(())
    }

    /// Every record of a batch whose vectors are a two dimensional array,
    /// with the ids, the metadata and the sparse half from the dict, each
    /// parsed on its own as a list batch's record is.
    fn parse_numpy_with_context(
        &self,
        rows: &Rows<'_>,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<Parsed>,
        errors: &mut Vec<String>,
    ) -> Result<(), PyEngineError> {
        let dim = self.inner.dim();
        let flat = rows.flat(dim)?;
        let num_vectors = rows.shape()[0];

        let ids_list = parallel_list(dict, "ids");
        let metadatas_list = Self::batch_metadatas(dict)?;
        // The sparse half's parallel arrays, where the batch carries them.
        let sparse_list = parallel_list(dict, "sparse");
        let texts_list = parallel_list(dict, "texts");

        trace!(
            operation = "parse_numpy_context",
            num_vectors = num_vectors,
            has_ids = ids_list.is_some(),
            has_metadata = metadatas_list.is_some(),
            "Processing vectors with context"
        );

        let mut rejected = 0usize;

        for (i, raw_vector) in flat.chunks(dim).enumerate() {
            // Resolve the caller's id before validating, so a rejected row can
            // be named without advancing the generated id counter for a record
            // that is never stored.
            let provided_id = Self::provided_id(ids_list.as_ref(), i)?;
            let label = || provided_id.clone().unwrap_or_else(|| format!("row_{}", i));

            if let Some((component, value)) = Self::first_non_finite(raw_vector) {
                error!(
                    operation = "parse_numpy_context",
                    error = "invalid_value",
                    vector_id = %label(),
                    index = component,
                    value = value,
                    "NumPy row contains a non-finite value"
                );
                errors.push(format!(
                    "Vector {}: contains invalid value at index {}: {} (must be finite)",
                    label(),
                    component,
                    value
                ));
                rejected += 1;
                continue;
            }

            // The record's metadata, read as a list batch reads it.
            let metadata = match Self::batch_metadata(metadatas_list.as_ref(), i) {
                Ok(metadata) => metadata,
                Err(e) => {
                    errors.push(format!("Vector {}: {}", label(), e));
                    rejected += 1;
                    continue;
                }
            };

            let processed_vector = self.inner.process_vector_for_space(raw_vector.to_vec());

            // Get ID from provided IDs or generate
            let id = match provided_id {
                Some(id) => id,
                None => self.inner.generate_id(),
            };

            trace!(
                operation = "parse_numpy_vector",
                vector_index = i,
                vector_id = %id,
                metadata_keys = metadata.keys().len(),
                "Parsed NumPy vector with context"
            );

            match sparse_input_at(sparse_list.as_ref(), texts_list.as_ref(), i) {
                Ok(sparse) => parsed_vectors.push(Parsed {
                    id,
                    vector: processed_vector,
                    metadata,
                    sparse,
                }),
                Err(e) => {
                    errors.push(format!("Vector {}: {}", id, e));
                    rejected += 1;
                }
            }
        }

        trace!(
            operation = "parse_numpy_context_complete",
            parsed_count = num_vectors - rejected,
            rejected_count = rejected,
            "NumPy parsing completed"
        );
        Ok(())
    }

    /// Safe list parsing that collects errors instead of failing immediately
    ///
    /// Each item is a record dict or a vector. An item of any other shape
    /// raises `TypeError` naming its position, before any item is parsed or
    /// takes a generated id.
    fn parse_list_input_safe(
        &self,
        list: &Bound<PyList>,
        parsed_vectors: &mut Vec<Parsed>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        for (item_index, item) in list.iter().enumerate() {
            if item.cast::<PyDict>().is_err() && !shaped_as_a_vector(&item) {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "add() takes a list whose items are record dicts or vectors, and item {} \
                     is {}",
                    item_index,
                    type_name(&item)
                )));
            }
        }
        for (item_index, item) in list.iter().enumerate() {
            if let Ok(item_dict) = item.cast::<PyDict>() {
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
                        let provided = item_dict
                            .get_item("id")
                            .ok()
                            .flatten()
                            .and_then(|id_item| id_item.extract::<String>().ok());

                        // Read before the record takes a generated id, so a
                        // record refused for its metadata takes none.
                        let metadata = match record_dict_metadata(item_dict) {
                            Ok(metadata) => metadata,
                            Err(e) => {
                                let label =
                                    provided.unwrap_or_else(|| format!("item_{}", item_index));
                                errors.push(format!("Vector {}: {}", label, e));
                                continue;
                            }
                        };

                        let id = provided.unwrap_or_else(|| self.inner.generate_id());

                        match sparse_input_of_record(item_dict) {
                            Ok(sparse) => parsed_vectors.push(Parsed {
                                id,
                                vector,
                                metadata,
                                sparse,
                            }),
                            Err(e) => errors.push(format!("Vector {}: {}", id, e)),
                        }
                    }
                    Err(e) => {
                        // Collect error with item index and ID for context
                        let id = item_dict
                            .get_item("id")
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
                        let id = self.inner.generate_id();
                        parsed_vectors.push(Parsed::dense(id, vector, HashMap::new()));
                    }
                    Err(e) => {
                        errors.push(format!("Item {}: {}", item_index, e));
                    }
                }
            }
        }
        Ok(())
    }

    /// A bare two dimensional array, one record a row, each under a generated
    /// id. A row holding a value that is not finite is that row's error,
    /// named by its position, and takes no id.
    fn parse_numpy_input_safe(
        &self,
        rows: &Rows<'_>,
        parsed_vectors: &mut Vec<Parsed>,
        errors: &mut Vec<String>,
    ) -> Result<(), PyEngineError> {
        let dim = self.inner.dim();
        let flat = rows.flat(dim)?;

        for (i, raw_vector) in flat.chunks(dim).enumerate() {
            // A bare array carries no ids, so a rejected row is named by its
            // position and no id is generated for it.
            if let Some((component, value)) = Self::first_non_finite(raw_vector) {
                error!(
                    operation = "parse_numpy",
                    error = "invalid_value",
                    row = i,
                    index = component,
                    value = value,
                    "NumPy row contains a non-finite value"
                );
                errors.push(format!(
                    "Vector row_{}: contains invalid value at index {}: {} (must be finite)",
                    i, component, value
                ));
                continue;
            }

            let processed_vector = self.inner.process_vector_for_space(raw_vector.to_vec());
            let id = self.inner.generate_id();
            parsed_vectors.push(Parsed::dense(id, processed_vector, HashMap::new()));
        }

        Ok(())
    }

    /// Report the first non-finite component of a vector, if there is one
    ///
    /// The two NumPy branches read their rows straight out of the buffer, so
    /// they skip the per-value check that `extract_single_vector_safe` runs.
    /// A NaN that reaches the graph degrades every later query rather than
    /// only the one that carried it, so both branches route through this.
    fn first_non_finite(vector: &[f32]) -> Option<(usize, f32)> {
        vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
            .map(|(index, value)| (index, *value))
    }

    /// One record's vector, checked and processed for the space, with a
    /// failure returned as words `add` reports against the record.
    fn extract_single_vector_safe(&self, data: &Bound<PyAny>) -> Result<Vec<f32>, String> {
        let vector = if let Ok(array1d) = data.cast::<PyArray1<f32>>() {
            array1d
                .readonly()
                .as_slice()
                .map_err(|e| format!("NumPy access error: {}", e))?
                .to_vec()
        } else if let Ok(list) = data.cast::<PyList>() {
            list.iter()
                .map(|item| {
                    item.extract::<f32>()
                        .map_err(|e| format!("List item error: {}", e))
                })
                .collect::<Result<Vec<f32>, String>>()?
        } else {
            data.extract::<Vec<f32>>()
                .map_err(|e| format!("Vector extraction error: {}", e))?
        };

        if vector.is_empty() {
            return Err("Vector cannot be empty".to_string());
        }
        if vector.len() != self.inner.dim() {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                vector.len()
            ));
        }
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "Vector contains invalid value at index {}: {}",
                    i, val
                ));
            }
        }

        Ok(self.inner.process_vector_for_space(vector))
    }
}
