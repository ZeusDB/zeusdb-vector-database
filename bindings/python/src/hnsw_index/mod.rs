//! The Python surface of the index: the `#[pyclass]` that wraps a
//! `Collection`, the one `#[pymethods]` block, and `AddResult`.
//!
//! # What lives here and what does not
//!
//! Every method here does four things and nothing else: it parses its
//! arguments into owned Rust, it releases the interpreter lock, it calls one
//! operation on the collection, and it converts what comes back. The
//! collection and every operation on it are in `zeusdb_vector_hnsw`, which
//! names nothing of Python, so what is left here is the Python-facing
//! contract, being the signatures, the docstrings and the exception classes.
//! The `#[pymethods]` block cannot be split: PyO3 accepts a second one only
//! under its `multiple-pymethods` feature, which would add a dependency this
//! crate has removed.
//!
//! | module | what it covers |
//! |---|---|
//! | `construct` | reading a declaration, its quantization mapping and its sparse space, and the neighbour selection warning |
//! | `input` | turning Python input into records and query vectors |
//! | `query` | reading a query over one or more arms, and writing its page and its plan |
//!
//! # Why the module keeps its name
//!
//! The module path is the target of every log record an entry point emits,
//! and `zeusdb_vector_database::hnsw_index` is what a filter directive has
//! always matched. The operations that moved into the index crate name the
//! same targets explicitly, so a record carries the target it always did
//! whichever side of the boundary emits it.
//!
//! # Interior mutability
//!
//! The wrapper holds the collection by value and every method takes `&self`.
//! `rebuild`, `compact` and `clear` replace the whole graph, and they do so
//! through `&self` by swapping guarded fields under the mutation lock, so
//! nothing here needs `&mut` and PyO3's exclusive borrow is never taken. That
//! is what lets a search run while a write is in flight.

mod construct;
mod input;
mod query;

use crate::conversion::{
    batch_hits_to_python, hits_to_python, python_dict_to_value_map, value_map_to_python,
};
use crate::PyEngineError;
use input::{Parsed, SparseInput};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use query::{page_to_python, plan_to_python};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, error, instrument, trace};
use zeusdb_vector_core::{compile_filter, Error};
use zeusdb_vector_hnsw::{Added, Collection, ParsedRecord, SparseHalf};

/// `skip_from_py_object` because nothing extracts an `AddResult`. It is the
/// return type of `add` and appears in no argument position, in this crate or
/// in the Python layer. PyO3 0.29 derives `FromPyObject` for a `#[pyclass]`
/// that is `Clone` and warns that the derive becomes opt-in, so the choice has
/// to be stated. Opting in would generate an extraction path no caller reaches.
///
/// It lives in the binding rather than beside the operation that produces it,
/// because it is a Python class: its five attributes and three methods are
/// the Python surface, and the index crate names nothing of Python. What the
/// index returns is [`Added`], the same five values as owned Rust, and the
/// conversion below is the whole of the difference.
#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
pub struct AddResult {
    #[pyo3(get)]
    pub total_inserted: usize,
    #[pyo3(get)]
    pub total_errors: usize,
    #[pyo3(get)]
    pub errors: Vec<String>,
    #[pyo3(get)]
    pub vector_shape: Option<(usize, usize)>,

    /// The id of every record this call put in the index, in insertion order.
    ///
    /// **It lines up with `total_inserted` and with nothing else.**
    /// `len(ids) == total_inserted` on every result this crate produces. A
    /// record rejected by parsing contributes no id, because none was ever
    /// allocated for it, and a record rejected by insertion contributes none
    /// either, because it is not in the index. So this is not positionally
    /// aligned with the input, and a caller who needs that alignment reads
    /// `errors`, which names each rejection by its id or by its position.
    ///
    /// Why it exists. An `add` that is given no ids generates them, as
    /// `vec_{counter}`, and until this field the generated ids were
    /// unreachable: the caller had to list the index and guess which entries
    /// were new. The llama-index adapter probes for exactly this attribute and,
    /// not finding it, returns the ids it supplied, which on a batch where any
    /// node lacked an id is a shorter list of the wrong values.
    ///
    /// An overwrite reports the id it accepted, since a replacement is an
    /// insertion as far as this counts.
    #[pyo3(get)]
    pub ids: Vec<String>,
}

impl From<Added> for AddResult {
    fn from(added: Added) -> Self {
        AddResult {
            total_inserted: added.inserted.len(),
            total_errors: added.total_errors,
            errors: added.errors,
            vector_shape: added.vector_shape,
            ids: added.inserted,
        }
    }
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

    /// One line human-readable summary of the insertion
    ///
    /// ASCII only, deliberately. This used to open with a check mark and carry a
    /// cross before the error count, and it is the first thing the README and the
    /// documentation site tell a new user to print. `print()` encodes through the
    /// console's code page, so on a Windows console still using the legacy one
    /// that first statement raised `UnicodeEncodeError` before the reader had
    /// added a second record.
    ///
    /// The counts stay available as `total_inserted` and `total_errors`, so the
    /// alternative of returning them as structured data would only duplicate two
    /// attributes that already exist while breaking every caller that prints
    /// this. The numbers and the words around them are unchanged, so a substring
    /// test or a `(\d+) inserted` match still holds. What no longer holds is a
    /// parse keyed on the emoji themselves or on a fixed character offset.
    pub fn summary(&self) -> String {
        format!(
            "{} inserted, {} errors",
            self.total_inserted, self.total_errors
        )
    }
}

/// The index as Python sees it.
///
/// A `Collection` held by value. The pyclass cannot hold a reference, and it
/// needs no `&mut`: every mutating operation replaces guarded fields under
/// the collection's own mutation lock, so every method here takes `&self` and
/// searches are never serialised against writes by PyO3's borrow flag. The
/// lock acquisition order, the registry that checks it and the reasoning
/// behind both are documented on `Collection`.
#[pyclass]
pub struct HNSWIndex {
    inner: Collection,
}
/// Build an `HNSWIndex`
///
/// The only way to construct an index from Python other than loading one from
/// disk. `HNSWIndex` carries no `#[new]`, so the class is importable for
/// `isinstance` checks and type annotations while direct construction raises
/// `TypeError`. Every rule that governs a valid index is enforced here, which
/// is what makes the Python factory and this function agree.
// The argument list is the Python signature, one value per keyword.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "_create_hnsw_index")]
#[pyo3(signature = (dim, space, m, ef_construction, expected_size, quantization_config = None, indexed_fields = None, sparse = None))]
pub fn create_hnsw_index(
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    quantization_config: Option<&Bound<PyDict>>,
    indexed_fields: Option<Vec<String>>,
    sparse: Option<&Bound<PyDict>>,
) -> Result<HNSWIndex, PyEngineError> {
    construct::build(
        dim,
        space,
        m,
        ef_construction,
        expected_size,
        quantization_config,
        indexed_fields.unwrap_or_default(),
        sparse,
    )
}
impl HNSWIndex {
    /// Wrap a collection the index crate built or loaded.
    pub(crate) fn wrap(inner: Collection) -> Self {
        HNSWIndex { inner }
    }

    /// The shape rule for a 2-D query array
    ///
    /// Shared by the `f32` and `f64` batch arms because a query is wrong in the
    /// same way whatever it is made of. Both arms only became reachable when
    /// `cast` was moved above `extract`, so until then the list arm below them
    /// answered for every array and these messages were never seen. They are
    /// worded to match it: an empty batch is refused in the same words, and a
    /// row of the wrong width says "dimension mismatch" rather than describing
    /// a shape, because that is what a caller who passed a list would read.
    fn validate_batch_array_shape(shape: &[usize], dim: usize) -> Result<(), Error> {
        if shape.len() != 2 {
            error!(
                operation = "search",
                error = "shape_mismatch",
                expected_shape = format!("(N, {})", dim),
                actual_shape = format!("{:?}", shape),
                "NumPy array shape mismatch"
            );
            return Err(Error::BatchArrayShape {
                dim,
                shape: shape.to_vec(),
            });
        }

        if shape[0] == 0 {
            error!(
                operation = "search",
                error = "empty_batch",
                "Batch cannot be empty"
            );
            return Err(Error::BatchEmpty);
        }

        if shape[1] != dim {
            error!(
                operation = "search",
                error = "dimension_mismatch",
                expected = dim,
                actual = shape[1],
                "NumPy batch row width does not match the index"
            );
            return Err(Error::SearchVectorDimension {
                expected: dim,
                got: shape[1],
            });
        }

        Ok(())
    }
}
#[pymethods]
impl HNSWIndex {
    /// Get quantization configuration and status
    pub fn get_quantization_info(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let report = self.inner.quantization_report()?;
        let dict = PyDict::new(py);
        dict.set_item("type", "pq").ok()?;
        dict.set_item("subvectors", report.subvectors).ok()?;
        dict.set_item("bits", report.bits).ok()?;
        dict.set_item("training_size", report.training_size).ok()?;

        if let Some(max_training) = report.max_training_vectors {
            dict.set_item("max_training_vectors", max_training).ok()?;
        }

        if let Some(quantizer) = report.quantizer {
            dict.set_item("is_trained", quantizer.is_trained).ok()?;
            dict.set_item("memory_mb", quantizer.memory_mb).ok()?;
            dict.set_item("total_centroids", quantizer.total_centroids)
                .ok()?;
            dict.set_item("sdc_memory_mb", quantizer.sdc_memory_mb)
                .ok()?;
            dict.set_item("centroid_norm_memory_mb", quantizer.centroid_norm_memory_mb)
                .ok()?;
            dict.set_item("compression_ratio", quantizer.compression_ratio)
                .ok()?;
        }

        Some(dict.into())
    }

    /// Check if quantization is enabled
    pub fn has_quantization(&self) -> bool {
        self.inner.has_quantization()
    }

    /// Get current vector count (for monitoring training trigger)
    pub fn get_vector_count(&self) -> usize {
        self.inner.vector_count()
    }

    /// Get the distance space configuration
    pub fn get_space(&self) -> String {
        self.inner.metric().to_string()
    }

    /// Rebuild the HNSW index to use PQ codes after training is complete
    ///
    /// Re-encodes whatever raw vectors the index still holds through the
    /// trained codebook and rebuilds the graph from the stored codes. It never
    /// retrains the codebook; training runs exactly once, on the `add` that
    /// reaches `training_size`. A trained `quantized_only` index holds no raw
    /// vectors, so there the rebuild proceeds from the codes alone, and under
    /// either mode nothing is lost by calling it. Returns false when there is
    /// no trained quantizer or nothing stored to rebuild from.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.inner.vector_count(),
        has_quantization = self.inner.has_quantization()
    ), err)]
    pub fn rebuild_with_quantization(&self, py: Python<'_>) -> Result<bool, PyEngineError> {
        // The whole rebuild runs with the interpreter lock released, the mutation
        // guard included. Waiting for another writer while holding the lock would
        // stall every Python thread in the process for the length of that writer,
        // which is the failure `add` releasing the lock would otherwise create.
        Ok(py.detach(|| self.inner.rebuild_with_quantization())?)
    }

    /// Check if the index is using quantized search
    pub fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
    }

    /// Check if quantization can be used (PQ is trained)
    pub fn can_use_quantization(&self) -> bool {
        self.inner.can_use_quantization()
    }

    /// Add records, in any of the five shapes, each filling the sparse
    /// space as well where the index declares one.
    ///
    /// A record fills the sparse space under the key the space takes.
    /// `sparse` is a mapping `{"dims": [...], "values": [...]}` of term ids
    /// and weights, parallel, the dims strictly increasing and the values
    /// finite, and whole numbers above zero on a space weighted by term
    /// frequency; a space that takes term ids alone takes it, and a space
    /// with a text layer refuses it, since that space's term ids are its
    /// dictionary's to issue. `text` is a string the space's text layer
    /// splits with the tokenizer it was declared with and counts into term
    /// ids and term frequencies, issuing an id to every term it has not
    /// seen; a space without a text layer refuses it. A record carries one
    /// or the other, or neither and fills the dense space alone. In the
    /// single object and list shapes each record carries its own key; in
    /// the batch shapes `sparse` and `texts` are parallel arrays beside
    /// `ids`, one entry per vector with `None` where a record fills the
    /// dense space alone, and held to the same length rule as `ids` and
    /// `metadatas`. The text itself is not stored; put it in the metadata
    /// to read it back.
    ///
    /// A record refused for its sparse half is counted and named in
    /// `errors`, and the records around it are inserted, as a record with a
    /// malformed vector is treated. That covers a malformed mapping, a
    /// sparse vector or a text on an index whose space does not take it,
    /// the engine's rules on the vector, and an exception the tokenizer
    /// raised, which is named with its class in the record's error.
    ///
    /// # Where the interpreter lock is held
    ///
    /// Parsing holds it, as it always has. Every text is then tokenized
    /// with it still held and no engine guard taken, since the tokenizer
    /// may be a Python callable; see the crate's `tokenizer` module. The
    /// lock is released after that, and the terms are counted into ids and
    /// the records inserted under the engine's mutation guard and the
    /// spaces' guards without it, so the interning and the postings insert
    /// never hold the lock and a search never waits on a Python callable.
    #[pyo3(signature = (data, overwrite = true))]
    #[instrument(level = "info", skip(self, data), fields(
        overwrite = overwrite,
        has_quantization = self.inner.has_quantization(),
        is_quantized = self.inner.is_quantized()
    ), err)]
    pub fn add(&self, data: Bound<PyAny>, overwrite: bool) -> Result<AddResult, PyEngineError> {
        // Input validation
        if data.is_none() {
            error!(
                operation = "add_vectors",
                error = "data_is_none",
                "Data cannot be None"
            );
            return Err(
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Data cannot be None").into(),
            );
        }

        // Use error-collecting parsing
        let (parsed, mut errors) = self.parse_input_data(&data)?;

        // The texts, tokenized with the lock held and no guard taken. A
        // tokenizer that fails on a record is that record's error, and so
        // is a supplied vector on a space with a text layer, whose term
        // ids are the dictionary's to issue. Whether the space has one is
        // read once per call, through an accessor that takes no guard
        // where there is none and the dictionary's read guard alone, for
        // a length, where there is.
        let takes_text_alone = self.inner.term_count().is_some();
        let mut records: Vec<ParsedRecord> = Vec::with_capacity(parsed.len());
        for Parsed {
            id,
            vector,
            metadata,
            sparse,
        } in parsed
        {
            let sparse = match sparse {
                SparseInput::None => None,
                SparseInput::Vector(vector) => {
                    if takes_text_alone {
                        // The vector's own rules first, as the engine
                        // applies them, so a malformed vector reads the
                        // same message on either kind of space, and then
                        // the space's.
                        let refused = match vector.as_ref().validate() {
                            Err(rule) => rule,
                            Ok(()) => Error::SparseVectorOnTextSpace,
                        };
                        errors.push(format!(
                            "Vector {}: {}: {}",
                            id,
                            refused.exception().name(),
                            refused
                        ));
                        continue;
                    }
                    Some(SparseHalf::Vector(vector))
                }
                SparseInput::Text(text) => match self.inner.tokenize(&text) {
                    Ok(terms) => Some(SparseHalf::Terms(terms)),
                    Err(e) => {
                        errors.push(format!("Vector {}: {}: {}", id, e.exception().name(), e));
                        continue;
                    }
                },
            };
            records.push(ParsedRecord {
                id,
                vector,
                sparse,
                metadata,
            });
        }

        // Parsing and tokenizing are the whole of what reads Python objects,
        // and both are done. Everything below works on owned Rust, so the
        // interning and the insertion run with the interpreter lock
        // released. The mutation guard is taken inside that region rather
        // than above it, so a caller waiting for another writer waits
        // without the lock. Holding it while waiting would stall every
        // Python thread in the process for the length of the writer ahead.
        //
        // `Collection::add_records` carries the proof that nothing inside
        // touches Python. A record's terms travel as strings and are counted
        // into ids there, under the mutation guard, so nothing this thread
        // holds across the boundary is an id the dictionary issued.
        let py = data.py();
        let added: Added = py.detach(|| self.inner.add_records(records, errors, overwrite));

        Ok(AddResult::from(added))
    }

    pub fn get_training_progress(&self) -> f32 {
        self.inner.training_progress()
    }

    /// Get number of training vectors still needed
    pub fn training_vectors_needed(&self) -> usize {
        self.inner.training_vectors_needed()
    }

    /// Check if training is ready to be triggered
    pub fn is_training_ready(&self) -> bool {
        self.inner.is_training_ready()
    }

    /// Get current storage mode description
    pub fn get_storage_mode(&self) -> String {
        self.inner.storage_mode()
    }

    /// Enhanced search method with automatic ADC usage
    ///
    /// `rerank` controls how far a quantized search over-fetches before it
    /// rescores the candidates against raw vectors. Omitted, the fetch is
    /// derived from the live record count; see `SearchParams::fetch_k`. An
    /// integer of 1 or more pulls that many candidates per requested result,
    /// which is a fixed multiple of the page and does not move with the corpus.
    /// Zero turns rerank off and restores the ADC scores and ordering. It has
    /// no effect on a raw index or on a `quantized_only` one, both of which
    /// never rerank; see `rerank_plan`.
    ///
    /// `ef_search` has no effect on a reranked quantized search. Below the
    /// fetch it is discarded, because `Hnsw::search_filter` raises the
    /// traversal width to the number of neighbours asked for and cannot return
    /// more results than its candidate list holds. Above the fetch it buys no
    /// recall, because the candidates a fetch returns are limited by the ADC
    /// ordering rather than by the traversal, and quadrupling it moves recall
    /// at 10 by at most 0.008. The default fetch is at least 250 and the
    /// default `ef_search` is 100, so changing `ef_search` alone changes
    /// nothing on a reranked search at the defaults.
    // The argument list is the Python signature, so it is not free to be
    // bundled the way the internal batch paths bundle theirs.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false, rerank=None))]
    #[instrument(level = "debug", skip(self, py, vector, filter), fields(
        top_k = top_k,
        ef_search = ef_search,
        return_vector = return_vector,
        rerank = rerank,
        is_quantized = self.inner.is_quantized()
    ), err)]
    pub fn search(
        &self,
        py: Python<'_>,
        vector: Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
        rerank: Option<usize>,
    ) -> Result<Py<PyAny>, PyEngineError> {
        let start_time = Instant::now();

        // The bounds on both arguments, the `ef` default and the rerank plan,
        // resolved once here rather than per query. See `search_params`.
        let params = self
            .inner
            .search_params(top_k, ef_search, return_vector, rerank)?;

        // Compiled once per search, however many queries the call carries, and
        // before any record is examined. Rejecting an unrecognised operator or a
        // malformed group per record would make the error depend on the data,
        // because a record that lacks the field never reaches the operator at
        // all. What comes back cannot fail against any record, which is why the
        // traversal predicate has no error channel.
        let filter_conditions = filter
            .map(python_dict_to_value_map)
            .transpose()?
            .as_ref()
            .map(compile_filter)
            .transpose()?;
        let dim = self.inner.dim();

        // Detect batch vs single query with comprehensive input support.
        //
        // # Why `cast` runs before `extract`
        //
        // `extract::<Vec<Vec<f32>>>` succeeds on a 2-D NumPy array, because an
        // array satisfies the sequence protocol and each row satisfies it in
        // turn. Tried first, it therefore consumed exactly the input the
        // zero-copy arm below it was written for, and that arm was unreachable
        // for every array of every dtype. Measured on a one record index at
        // dimension 1,536 with a batch of 32, where the traversal is negligible
        // and the marshalling is the whole cost, it took 101.96 microseconds a
        // query against 21.54 for a list of lists, so the general path was
        // paying 4.7 times the list path to read the one input it could read
        // without copying at all.
        //
        // `cast` is a type check rather than a conversion attempt, so it
        // cannot claim an input that is not an array of that dtype, and it is
        // safe above `extract` in a way `extract` is not above it. `add`
        // already dispatches this way at every arm.
        //
        // The `f64` arm sits between them. NumPy's default dtype is `f64`, so
        // `np.random.rand(32, 1536)` is the common shape rather than an unusual
        // one, and without an arm of its own it falls to `extract` and is read
        // one Python float at a time. The conversion is the same rounding
        // `extract::<f32>` performs per element, so the values are identical.
        let result: Py<PyAny> = if let Ok(np_array) = vector.cast::<PyArray2<f32>>() {
            // Format: NumPy 2-D array (N, dims), read without copying.
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            Self::validate_batch_array_shape(shape, dim)?;

            let flat = readonly.as_slice()?;
            let batch: Vec<Vec<f32>> = flat.chunks(dim).map(|chunk| chunk.to_vec()).collect();
            debug!(
                operation = "batch_search_numpy",
                batch_size = batch.len(),
                "Starting NumPy batch search"
            );
            let hits = py.detach(|| {
                self.inner
                    .search_batch(&batch, filter_conditions.as_ref(), params)
            })?;
            PyList::new(py, batch_hits_to_python(hits, py)?)?.into()
        } else if let Ok(np_array) = vector.cast::<PyArray2<f64>>() {
            // Format: NumPy 2-D array of f64, narrowed in one pass.
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            Self::validate_batch_array_shape(shape, dim)?;

            let flat = readonly.as_slice()?;
            let batch: Vec<Vec<f32>> = flat
                .chunks(dim)
                .map(|chunk| chunk.iter().map(|&value| value as f32).collect())
                .collect();
            debug!(
                operation = "batch_search_numpy",
                batch_size = batch.len(),
                dtype = "f64",
                "Starting NumPy batch search"
            );
            let hits = py.detach(|| {
                self.inner
                    .search_batch(&batch, filter_conditions.as_ref(), params)
            })?;
            PyList::new(py, batch_hits_to_python(hits, py)?)?.into()
        } else if let Ok(list_vec) = vector.extract::<Vec<Vec<f32>>>() {
            // Format: List of vectors [[0.1, 0.2], [0.3, 0.4]]

            // Validation for empty batch or empty vectors in batch
            if list_vec.is_empty() {
                error!(
                    operation = "search",
                    error = "empty_batch",
                    "Batch cannot be empty"
                );
                return Err(Error::BatchEmpty.into());
            }

            // Check for empty vectors within the batch
            for (i, vec) in list_vec.iter().enumerate() {
                if vec.is_empty() {
                    error!(
                        operation = "search",
                        error = "empty_vector_in_batch",
                        vector_index = i,
                        "Vector in batch cannot be empty"
                    );
                    return Err(Error::BatchVectorEmpty { position: i }.into());
                }
            }

            debug!(
                operation = "batch_search",
                batch_size = list_vec.len(),
                "Starting batch search"
            );
            let hits = py.detach(|| {
                self.inner
                    .search_batch(&list_vec, filter_conditions.as_ref(), params)
            })?;
            PyList::new(py, batch_hits_to_python(hits, py)?)?.into()
        } else {
            // Single vector path - enhanced with NumPy 1D support.
            //
            // Already `cast` before `extract`, so there is no reordering to
            // make here. The `f64` arm is the same addition as on the batch
            // path and for the same reason: without it a `float64` query, which
            // is what NumPy hands back by default, was read one Python float at
            // a time. Measured on a one record index at dimension 1,536 that
            // was 84.41 microseconds a query against 8.43 for `float32`.
            let query_vector = if let Ok(array1d) = vector.cast::<PyArray1<f32>>() {
                array1d.readonly().as_slice()?.to_vec()
            } else if let Ok(array1d) = vector.cast::<PyArray1<f64>>() {
                array1d
                    .readonly()
                    .as_slice()?
                    .iter()
                    .map(|&value| value as f32)
                    .collect()
            } else {
                vector.extract::<Vec<f32>>()?
            };

            // The width and value checks, then the processing every stored
            // vector had. See `Collection::validate_query`.
            let processed_query = self.inner.validate_query(query_vector)?;

            trace!(
                operation = "single_search",
                query_dim = processed_query.len(),
                "Starting single vector search"
            );

            let hits = py.detach(|| {
                self.inner
                    .search_one(&processed_query, filter_conditions.as_ref(), params)
            })?;
            PyList::new(py, hits_to_python(hits, py)?)?.into()
        };

        // ✅ ENTERPRISE: Add duration timing to hot path with actual result count
        let duration_ms = start_time.elapsed().as_millis();
        let results_count = {
            let any = result.bind(py);
            match any.cast::<PyList>() {
                Ok(list) => list.len(),
                Err(_) => 0,
            }
        };

        debug!(
            operation = "search_complete",
            results_count = results_count,
            duration_ms = duration_ms,
            "Search completed"
        );

        Ok(result)
    }

    /// Search one or more spaces with one query and return one page.
    ///
    /// `arms` is a list of mappings, one per arm, each asking one space with
    /// one query. `{"vector": [...]}` asks the dense space, with the
    /// vector on the caller's scale as `search` takes it, and takes
    /// `ef_search` and `rerank` as `search` does. `{"sparse": {"dims":
    /// [...], "values": [...]}}` asks a sparse space that takes term ids
    /// alone, with term ids and weights; a space with a text layer refuses
    /// it, since that space's term ids are its dictionary's. `{"text":
    /// "..."}` asks a space with a text layer, with a string the layer
    /// splits and counts as it counted the records; a space without one
    /// refuses it. Either takes `idf`: `"corpus"`, the default, weights a
    /// term by its rarity over the records the filter admits, and
    /// `"global"` over every live record. A query names one to eight arms.
    ///
    /// Every arm runs under the one admit set the `filter` decides, which
    /// is the filter `search` takes and selects the same records. `top_k`
    /// is the page. `fetch` is the candidates each arm contributes to the
    /// fusion, which is the page for one arm and five times it for several
    /// unless set; a record just outside one arm's page cannot be lifted by
    /// its rank on another, so a deeper fetch gives the fusion more to work
    /// with, and the dense arm's traversal widens once the fetch passes
    /// half its default width. `fusion` is how the arms' pages become one,
    /// `"rrf"` or `{"type": "rrf", "k": 60.0}`, being reciprocal rank
    /// fusion: a record's fused score is the sum over the pages it appears
    /// on of `1 / (k + rank)`, rank counted from one, reading no score,
    /// because a dense distance and a sparse similarity are not comparable.
    /// It is read where there is more than one arm.
    ///
    /// The page is a list of dicts, best first and cut to `top_k`, and may
    /// be shorter, since a sparse arm returns no record sharing no term
    /// with the query and a filter may admit fewer records than asked for.
    /// Each carries `id`, `score`, `metadata` and `contributions`. For one
    /// arm the score is the arm's own, being the dense distance or the
    /// sparse similarity, and a one arm dense query is `search` id for id
    /// and score for score. For several it is the fused score, higher
    /// better and comparable only with the other fused scores of the same
    /// query, and `contributions` is what it was made from: one entry per
    /// arm's page the record appeared on, in arm order, each carrying the
    /// arm's position in `arms`, the record's `rank` on that page from one,
    /// and the `score` that arm gave it. Among equal fused scores the id
    /// orders the page, so a query returns the same page from one run to
    /// the next.
    ///
    /// Every rule is checked before any guard is taken, so a refused query
    /// changes nothing. `explain` returns the plan the same query runs
    /// under without running it.
    ///
    /// # Where the interpreter lock is held
    ///
    /// Reading the arms holds it, and a text arm's tokenizer runs then,
    /// with no engine guard taken. The lock is released for the search,
    /// which takes the spaces' guards without it, and taken back to build
    /// the page.
    #[pyo3(signature = (arms, filter=None, top_k=10, fetch=None, fusion=None))]
    #[instrument(level = "debug", skip(self, py, arms, filter, fusion), fields(
        top_k = top_k,
        fetch = fetch
    ), err)]
    pub fn query(
        &self,
        py: Python<'_>,
        arms: &Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        fetch: Option<usize>,
        fusion: Option<&Bound<PyAny>>,
    ) -> Result<Py<PyAny>, PyEngineError> {
        let start_time = Instant::now();
        let parsed = self.parse_query(arms, filter, top_k, fetch, fusion)?;
        let page = py.detach(|| {
            let arms = parsed.arms();
            self.inner.query(&parsed.query(&arms))
        })?;
        let results_count = page.hits.len();
        let result = page_to_python(page.hits, py)?;
        debug!(
            operation = "query_complete",
            arms = page.plan.arms.len(),
            results_count = results_count,
            duration_ms = start_time.elapsed().as_millis(),
            "Query completed"
        );
        Ok(result)
    }

    /// The plan a query would run under, without running it.
    ///
    /// Takes what `query` takes, holds the query to every rule `query`
    /// applies, decides the admit set every arm would run under, prices
    /// every arm under it, and returns that as a dict, which is the plan
    /// `query` would produce beside the same page. It costs the admit set,
    /// which for a filter over undeclared fields is a walk of the
    /// metadata, and not the search.
    ///
    /// `admit` is the shape of the admit set: `{"shape": "all"}` for no
    /// filter or a filter admitting every live record above the dense scan
    /// threshold, `{"shape": "bitmap", "admitted": n}` for a filter every
    /// field of which is declared, `{"shape": "sorted", "admitted": n}` for
    /// a metadata walk that finished, `{"shape": "bounded", "bound": n}`
    /// for a walk that gave up inside the bound the declared fields left,
    /// and `{"shape": "predicate"}` for one that gave up over every record.
    ///
    /// `arms` is one dict per arm, in arm order: `space` and `kind` name
    /// the space the arm asks, `fetch` is the candidates it was asked for,
    /// `exact` says whether its page is exact by construction, being a
    /// scan rather than a graph traversal, and `cost_ns` is the arm's own
    /// estimate of its work, in nanoseconds.
    ///
    /// **`cost_ns` is an estimate of the arm's own work and nothing more.**
    /// It is priced from unit costs the space timed on itself when it was
    /// built or loaded, and it leaves out what the collection pays around
    /// the arm, being the filter's evaluation over the columns and the
    /// page's assembly, which measured near seventy microseconds a query at
    /// fifty thousand records under a filter admitting every record. A
    /// wall time measured under a filter exceeds it by that much, a dense
    /// arm on a freshly loaded index of low width may be priced up to one
    /// and a half times what it runs at from a warm cache, and a sparse
    /// arm's price is within about fifteen percent where it was measured.
    /// It is a figure to compare arms of one query by, not a promise.
    ///
    /// `fusion` is the fusion `query` would apply, `{"type": "rrf", "k":
    /// 60.0}`, or `None` for a one arm query, whose page is the arm's own.
    #[pyo3(signature = (arms, filter=None, top_k=10, fetch=None, fusion=None))]
    pub fn explain(
        &self,
        py: Python<'_>,
        arms: &Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        fetch: Option<usize>,
        fusion: Option<&Bound<PyAny>>,
    ) -> Result<Py<PyDict>, PyEngineError> {
        let parsed = self.parse_query(arms, filter, top_k, fetch, fusion)?;
        let plan = py.detach(|| {
            let arms = parsed.arms();
            self.inner.explain(&parsed.query(&arms))
        })?;
        plan_to_python(&plan, py)
    }

    /// Enhanced Save method to include HNSW Graph
    ///
    /// The whole save runs with the interpreter lock released. `save` reaches
    /// `save_config`, `save_mappings`, `save_metadata`,
    /// `save_quantization_config`, `save_pq_centroids`, `save_pq_codes` and
    /// `save_vectors`, and every one of them speaks only to `serde_json`,
    /// `bincode` and `std::fs`. `save_hnsw_graph` reaches `graph::dump::write_dump`,
    /// which names PyO3 nowhere, and `save_manifest` and `StagingDir::commit`
    /// after it speak only to `serde_json` and `std::fs`. Nothing in the index
    /// crate names Python at all.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.inner.vector_count(),
        has_quantization = self.inner.has_quantization(),
        is_quantized = self.inner.is_quantized()
    ), err)]
    pub fn save(&self, py: Python<'_>, path: &str) -> Result<(), PyEngineError> {
        Ok(py.detach(|| self.inner.save(path))?)
    }

    /// Python property: `index.dim`
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Python property: `index.space`
    ///
    /// The same value `get_space()` returns, which stays. The two exist because
    /// `dim` is a property and `space` was not, so anything reading the index's
    /// configuration by attribute found half of it. The langchain adapter reads
    /// `getattr(index, "space", None)` to choose how it normalises a distance
    /// into a relevance score, and finding nothing it took the literal default
    /// of cosine, so an L2 or L1 index was scored by the cosine rule.
    ///
    /// It is a property rather than a second method because the caller that
    /// needed it probes for an attribute and calls it only if it turns out to be
    /// callable, so a property satisfies it and a method reads as configuration
    /// rather than as an action.
    ///
    /// `m`, `ef_construction` and `expected_size` are properties too, added
    /// after this one and for a different reason. Neither adapter reads them;
    /// their gap was that a caller wanting them had to parse `get_stats()`,
    /// which returns them as text.
    ///
    /// Named `space_property` in Rust because PyO3 derives the symbol it
    /// generates from the Rust name, and a getter's symbol takes a `get_`
    /// prefix, so a getter written as `space` collides with the existing
    /// `get_space` method. `#[getter(space)]` names the Python property.
    #[getter(space)]
    pub fn space_property(&self) -> String {
        self.inner.metric().to_string()
    }

    /// Python property: `index.m`
    ///
    /// The graph degree, and the one creation parameter `rebuild` can change.
    ///
    /// Reachable before this only as `int(index.get_stats()["m"])`, which is a
    /// number formatted into a string and parsed back. A caller reads the
    /// construction parameters as typed properties instead.
    ///
    /// Read-only, because `m` is what the graph was built with. Assigning it
    /// would describe a graph that does not exist. Changing it for real is
    /// `rebuild(m=...)`, which builds the graph again at the new degree.
    #[getter]
    pub fn m(&self) -> usize {
        self.inner.m()
    }

    /// Python property: `index.ef_construction`
    ///
    /// The candidate width each insertion searched at. It has no effect on a
    /// search, so a caller wanting a wider search passes `ef_search`.
    ///
    /// Read-only, because it describes work already done. Changing it for real
    /// is `rebuild(ef_construction=...)`, which re-inserts every record at the
    /// new width and so makes the number true again.
    #[getter]
    pub fn ef_construction(&self) -> usize {
        self.inner.ef_construction()
    }

    /// Python property: `index.expected_size`
    ///
    /// The record count declared at creation. A capacity hint rather than a
    /// cap, so an index that grows past it grows the graph rather than
    /// raising. It selected the default `m` and it
    /// sized the initial reservation, which is why it is worth reading back.
    ///
    /// `len(index)` is the actual count and this is the declaration. The two
    /// disagreeing is ordinary.
    #[getter]
    pub fn expected_size(&self) -> usize {
        self.inner.expected_size()
    }

    /// `len(index)`, the number of live records.
    ///
    /// Reads `id_map`, which is the record set: every insertion path writes it,
    /// removal keys on it, and `contains`, `list` and `count` all read the same
    /// map, so none of them can disagree with this. It equals
    /// `get_vector_count()` and `get_stats()["total_vectors"]`, which are
    /// maintained separately as a counter.
    ///
    /// Zero on an empty index, which is the only edge case a length has.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// `id in index`, which is `contains(id)`.
    ///
    /// `contains()` stays, because removing it would break every caller using
    /// it. This is the same read of the same map and cannot answer differently.
    pub fn __contains__(&self, id: String) -> bool {
        self.inner.contains(&id)
    }

    /// Live records matching a filter, or every live record when none is given.
    ///
    /// **Exact, and therefore a complete walk.** With a filter this evaluates
    /// every record's metadata and counts the matches. It cannot stop early:
    /// a count is a statement about the whole index, so the first record it
    /// skipped would make the answer a lower bound rather than a count.
    /// `scan_candidates` stops at `FULL_SCAN_THRESHOLD` because a search only
    /// needs to know whether the matching set is small enough to rank directly,
    /// which is a question an early exit answers and this one is not.
    ///
    /// What it reuses is `matches_filter` and `validate_filter_conditions`,
    /// which are the whole of the filter language. What it does not reuse is
    /// the scan's give-up point, its distance evaluation and its sort, none of
    /// which a count reads.
    ///
    /// Without a filter it is `len(index)` and reads one map length.
    ///
    /// An unknown operator raises `ValueError` before any record is examined,
    /// which is what a search does with the same filter. An empty index counts
    /// zero, and a filter matching nothing counts zero.
    #[pyo3(signature = (filter=None))]
    pub fn count(
        &self,
        py: Python<'_>,
        filter: Option<&Bound<PyDict>>,
    ) -> Result<usize, PyEngineError> {
        let conditions = filter
            .map(python_dict_to_value_map)
            .transpose()?
            .as_ref()
            .map(compile_filter)
            .transpose()?;

        // The walk runs with the interpreter lock released. It reads every
        // record, which at 100,000 records is tens of milliseconds, and holding
        // the lock for that would stall every Python thread in the process.
        Ok(py.detach(|| self.inner.count(conditions.as_ref())))
    }

    /// The metadata fields this index built a column for, in the order they
    /// were declared.
    ///
    /// Empty on an index created without `indexed_fields` and on every
    /// directory saved before the declaration existed. A filter naming a field
    /// not in this list still works and still returns the same records; what it
    /// costs is a walk of every record's metadata.
    #[getter]
    pub fn indexed_fields(&self) -> Vec<String> {
        self.inner.indexed_fields()
    }

    /// Return the graph's spare buffer capacity to the allocator.
    ///
    /// A graph built by insertion grows its arenas geometrically, so the last
    /// growth leaves the largest of them holding close to twice what it uses. A
    /// graph read back from a saved dump has none of that slack, because the
    /// node count is known before the first write. That is the whole of why the
    /// same index reports a smaller graph after a save and load round trip than
    /// it did when it was built.
    ///
    /// `compact()` does not reclaim it. Compaction rebuilds by inserting, so the
    /// replacement graph regrows exactly the same slack, which is why this is
    /// its own operation. `compact()` calls it on the graph it built, so a
    /// caller who runs compaction gets both.
    ///
    /// **No node, edge or distance is touched.** Fourteen buffers are
    /// reallocated at their current lengths and their contents copied, so the
    /// topology after the call is the topology before it and every search
    /// returns the same page with the same scores.
    ///
    /// **The index stays writable.** Every buffer is a growable vector, so the
    /// next `add()` reallocates the per node arenas once and then proceeds as
    /// before. Shrinking an index that is still being written to therefore
    /// trades one regrowth for the memory, which is why this is never automatic.
    /// On an index that is finished, or one about to be searched for a long
    /// time, there is nothing to trade.
    ///
    /// Returns the bytes released, which is zero only when the buffers are
    /// already tight, in which case nothing is reallocated either.
    ///
    /// **On an empty index it releases the whole creation reservation**, which
    /// is what `expected_size` bought. Calling it before inserting is therefore
    /// not free: it hands back the pre-allocation and every subsequent
    /// insertion regrows the arenas from nothing. Call it on an index that
    /// holds its records, not on one about to receive them.
    pub fn shrink_to_fit(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.shrink_to_fit())
    }

    /// Get records by ID(s) with PQ reconstruction support and storage mode awareness
    ///
    /// Looks the ids up in the union of the raw vectors and the quantized codes,
    /// so it already saw every record before the other accessors did.
    ///
    /// **An absent id is dropped by default and raised with `strict=True`.** The
    /// default is silence because that is what every existing caller is written
    /// against: asking for ten ids and receiving nine has always been how this
    /// reports an id the index does not hold, and changing the return shape
    /// would break every one of them. What the default cannot do is say *which*
    /// nine, since the result is not aligned with the input. `strict=True`
    /// raises a `KeyError` naming the absent ids, which is the form a caller who
    /// cares can act on, and it is recommended over the alternative of returning
    /// a structure carrying the misses: a caller who passes ids it believes are
    /// present wants the failure, and one who does not can filter with
    /// `contains`.
    ///
    /// `return_vector` is served from the raw vector where one exists and from a
    /// reconstruction of the code where one does not. Under `quantized_only`
    /// that is every record once training completes, the training records
    /// included, since the rebuild releases their raw vectors the moment their
    /// codes are stored. The returned value is then an approximation rather
    /// than the value supplied. Measured on 16 dimensional data with 4
    /// subvectors and 8 bits, a reconstructed vector differed from the stored
    /// unit vector by 0.066 at the worst component and sat at cosine similarity
    /// 0.991 to it. Under `quantized_with_raw` every record keeps its raw
    /// vector and returns exactly. `get_stats()["raw_vectors_stored"]` is what
    /// tells the two apart in aggregate.
    #[pyo3(signature = (input, return_vector = true, strict = false))]
    pub fn get_records(
        &self,
        py: Python<'_>,
        input: &Bound<PyAny>,
        return_vector: bool,
        strict: bool,
    ) -> Result<Vec<Py<PyDict>>, PyEngineError> {
        let ids: Vec<String> = if let Ok(id_str) = input.extract::<String>() {
            vec![id_str]
        } else if let Ok(id_list) = input.extract::<Vec<String>>() {
            id_list
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a string or a list of strings for ID(s)",
            )
            .into());
        };

        let records = py.detach(|| self.inner.records(ids, return_vector, strict))?;

        let mut output = Vec::with_capacity(records.len());
        for record in records {
            let dict = PyDict::new(py);
            dict.set_item("id", record.id)?;
            dict.set_item("metadata", value_map_to_python(&record.metadata, py)?)?;
            if let Some(vec) = record.vector {
                // A list, matching `search`.
                dict.set_item("vector", vec)?;
            }
            output.push(dict.into());
        }
        Ok(output)
    }

    /// Enhanced get_stats with storage mode information
    ///
    /// The figures, and why `total_memory_mb` is not the resident set, are on
    /// `Collection::stats`.
    pub fn get_stats(&self) -> HashMap<String, String> {
        self.inner.stats()
    }

    /// One page of records, as (id, metadata), in the order they were added.
    ///
    /// Enumerates `id_map`, which holds every live record. It used to enumerate
    /// `vectors`, which under `quantized_only` holds only the records collected
    /// before training, so every record added afterwards was missing from the
    /// listing while search still returned it.
    ///
    /// # The order, which is what makes `offset` mean anything
    ///
    /// **Ascending internal id, which is arrival order.** This used to hand back
    /// `id_map.keys()` directly, and a `HashMap` iterates in an order its hasher
    /// reseeds in every process, so two calls in one process agreed and two
    /// processes did not. An offset over an order like that is not a page: it
    /// can return a record twice and miss another entirely, which is worse than
    /// having no paging at all. The scan path meets the same problem, where two
    /// equally distant records come back in hasher order, and pins a tie break
    /// for it.
    ///
    /// Arrival order rather than the external id's own order, for two reasons.
    /// A record added while a caller is paging appends at the end, so it cannot
    /// push a record the caller has not reached yet across a page boundary,
    /// which sorting by external id would do for any id that sorts low. And it
    /// is the order `compact` already rebuilds in, for the same reason: a
    /// property of the data rather than of where a hasher put a key. Internal
    /// ids are unique and are never reissued, so the order is total, and they
    /// survive a save and load, so it is the same order in the next process.
    ///
    /// # Paging by cursor rather than by count
    ///
    /// **`offset` shifts under deletion and `after` does not.** Removing a
    /// record ahead of an offset moves everything behind it up by one, so the
    /// next page skips a record, and no ordering fixes that: it is what paging
    /// by a count means. `after` names the last record the caller saw, and the
    /// next page is every record whose internal id is above that one's, so a
    /// deletion anywhere ahead of the cursor changes nothing about where the
    /// page starts.
    ///
    /// It is cheap because internal ids are what the order already rests on.
    /// They are unique, never reissued, and survive `compact`, `rebuild` and a
    /// save and load, so a cursor taken in one process is a cursor in the next.
    ///
    /// The one case it cannot absorb is the cursor record itself being removed,
    /// since its internal id goes with it. That raises, naming the id, rather
    /// than silently returning a page from somewhere else.
    ///
    /// `offset` stays as a convenience and the two are not combined: passing
    /// both raises, because a caller mixing them has two ideas about where the
    /// page starts.
    ///
    /// An offset at or past the record count returns an empty list rather than
    /// raising, and `number` of zero returns an empty list. Neither is an error.
    #[pyo3(signature = (number=10, offset=0, after=None))]
    pub fn list(
        &self,
        py: Python<'_>,
        number: usize,
        offset: usize,
        after: Option<String>,
    ) -> Result<Vec<(String, Py<PyAny>)>, PyEngineError> {
        let page = py.detach(|| self.inner.list(number, offset, after.as_deref()))?;

        let mut results = Vec::with_capacity(page.len());
        for (id, metadata) in page {
            results.push((id, value_map_to_python(&metadata, py)?));
        }
        Ok(results)
    }

    /// Check whether a record with this id is in the index
    ///
    /// Reads `id_map`, which is the record set. Every insertion path writes it,
    /// `remove_point_internal` keys its removal on it, `add(overwrite=True)`
    /// keys its collision test on it, and `compact` rebuilds the graph from it.
    /// It used to read `vectors`, which under `quantized_only` holds only the
    /// records collected before training, so this returned `false` for a record
    /// that search returned and `remove_point` removed.
    pub fn contains(&self, id: String) -> bool {
        self.inner.contains(&id)
    }

    /// Add index-level metadata, merging the pairs into what is held.
    ///
    /// The engine takes its mutation guard for this, as it does for every
    /// other change a save records, so the interpreter lock is released
    /// around the call: a caller waiting for another writer waits without
    /// it, as `add` and `clear` arrange.
    pub fn add_metadata(
        &self,
        py: Python<'_>,
        metadata: HashMap<String, String>,
    ) -> Result<(), PyEngineError> {
        Ok(py.detach(|| self.inner.add_metadata(metadata))?)
    }

    /// Get index-level metadata value
    pub fn get_metadata(&self, key: String) -> Option<String> {
        self.inner.metadata(&key)
    }

    /// Get all index-level metadata
    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        self.inner.all_metadata()
    }

    /// Get a human-readable info string
    ///
    /// `vectors=` is the live record count in every storage mode. See
    /// `Collection::info`.
    pub fn info(&self) -> String {
        self.inner.info()
    }

    /// Remove vector by ID
    /// Public remove_point method (unchanged for API compatibility)
    /// This code delegates to remove_point_internal() which handles all the complex logic
    pub fn remove_point(&self, py: Python<'_>, id: String) -> Result<bool, PyEngineError> {
        // `id` arrives already converted, and the removal is entirely Rust. The
        // removal itself is short, but the wait for the mutation guard is not,
        // because `add` can hold it for a long insert with the lock released.
        // Waiting here with the lock held would stall every Python thread.
        Ok(py.detach(|| self.inner.remove_point(id))?)
    }

    /// Remove a batch of records, taking the mutation lock once.
    ///
    /// **Returns the ids that were not in the index**, in the order they were
    /// given, so an empty list means every id was removed. That is what the
    /// caller needs and a count is not: both shipped adapters loop
    /// `remove_point` today precisely to learn which ids failed, and both then
    /// throw away everything except a total.
    ///
    /// A repeated id is removed on its first occurrence and skipped afterwards,
    /// so it is never reported missing. The caller asked for the record to be
    /// gone and it is.
    ///
    /// An empty list of ids removes nothing and returns an empty list. Every id
    /// being absent returns all of them, which is not an error: removing a
    /// record that is not there is the state the caller asked for.
    ///
    /// What it saves against the loop is the locking. One acquisition of the
    /// mutation guard and one of each of the five storage guards for the whole
    /// batch, rather than one of each per id. It also makes the batch atomic
    /// against every search, where the loop let a search land between any two
    /// removals.
    ///
    /// Stranded graph nodes are left exactly as `remove_point` leaves them, one
    /// per record removed. Search already excludes them. `compact()` reclaims
    /// them and this does not call it, because compaction costs a full rebuild
    /// and the caller is the one who knows whether the debris is worth it.
    pub fn remove_points(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
    ) -> Result<Vec<String>, PyEngineError> {
        Ok(py.detach(|| self.inner.remove_points(&ids))?)
    }

    /// Remove records by id or by filter, and report how many went.
    ///
    /// An alias. `delete(ids=...)` is `remove_points`, `delete(where=...)` is
    /// `remove_where`, and both of those stay. It exists because `delete` is
    /// the ordinary name for the operation, so a caller reaches for
    /// `index.delete(...)`, gets an `AttributeError`, and has to go looking.
    /// The existing family compounds that: `remove_point` and `remove_points`
    /// differ by one character.
    ///
    /// **Both arguments is an error.** Two selections do not compose into one
    /// without a rule, and either rule is a guess. The union deletes records
    /// the filter did not choose and the intersection deletes fewer records
    /// than the ids name, so neither is what a caller writing both meant.
    ///
    /// **Neither argument is an error.** A `delete()` that deleted everything
    /// is the hazard `remove_where({})` already refuses, arrived at by leaving
    /// two optional arguments unset rather than by passing an empty mapping,
    /// which is easier to do by accident and not harder.
    ///
    /// **Returns the number of records removed**, in both cases, so the return
    /// type does not depend on which argument was given. `remove_points`
    /// returns the ids it could not find, which is more than a count and is
    /// what a caller needing that detail should keep calling. A repeated id
    /// counts once, because one record was removed.
    ///
    /// `ids` takes a single string or a list of strings, matching
    /// `get_records`. An empty list removes nothing and returns zero, which is
    /// not an error: it is a delete of nothing, not a delete of everything.
    #[pyo3(signature = (ids=None, r#where=None))]
    pub fn delete(
        &self,
        py: Python<'_>,
        ids: Option<&Bound<PyAny>>,
        r#where: Option<&Bound<PyDict>>,
    ) -> Result<usize, PyEngineError> {
        match (ids, r#where) {
            (Some(_), Some(_)) => Err(Error::DeleteBothSelectors.into()),
            (None, None) => Err(Error::DeleteNoSelector.into()),
            (Some(ids), None) => {
                let requested: Vec<String> = if let Ok(single) = ids.extract::<String>() {
                    vec![single]
                } else if let Ok(many) = ids.extract::<Vec<String>>() {
                    many
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "delete(ids=) expects a string or a list of strings",
                    )
                    .into());
                };

                Ok(py.detach(|| self.inner.delete_ids(&requested))?)
            }
            (None, Some(filter)) => self.remove_where(py, filter),
        }
    }

    /// Remove every record whose metadata matches the filter, and report how
    /// many were removed.
    ///
    /// The filter is the language `search` takes, evaluated by the same
    /// function against the same metadata, so a filter that selects a set here
    /// selects the same set there. An unknown operator raises `ValueError`
    /// before any record is examined.
    ///
    /// A filter matching nothing removes nothing and returns zero, which is not
    /// an error. An empty index returns zero.
    ///
    /// **An empty filter is refused.** Everywhere else in this language an empty
    /// filter matches every record, and `search(filter={})` returns the whole
    /// index for exactly that reason. This is the one method where following
    /// that rule destroys every record, and an empty mapping reaches it far more
    /// often from a caller that built its filter and got nothing than from a
    /// caller that meant it. Consistency is worth less here than the failure it
    /// would permit, because a search is repeatable and this is not. A caller
    /// who does mean it names the records, through `remove_points`.
    ///
    /// Stranded graph nodes are left behind, one per record removed, exactly as
    /// `remove_point` leaves them. This does not call `compact()` either; see
    /// `remove_points`.
    pub fn remove_where(
        &self,
        py: Python<'_>,
        filter: &Bound<PyDict>,
    ) -> Result<usize, PyEngineError> {
        let conditions = compile_filter(&python_dict_to_value_map(filter)?)?;
        Ok(py.detach(|| self.inner.remove_where(&conditions))?)
    }

    /// Empty the index, keeping its configuration, and report how many
    /// records went.
    ///
    /// **A fresh graph and empty maps, not `remove_points` over every id.**
    /// Both were considered. Removing every record one at a time is linear in
    /// the record count and leaves one stranded graph node per record, so an
    /// index of a million records would spend a long time producing a graph
    /// holding a million dead nodes and no live ones, which `compact()` would
    /// then have to walk again. Replacing the graph reclaims all of it at once
    /// and leaves `get_stats()["stranded_graph_nodes"]` at zero, which is what
    /// an empty index should report.
    ///
    /// What it keeps is the index: `dim`, `space`, `m`, `ef_construction`,
    /// `expected_size`, the index-level metadata `add_metadata` wrote, and the
    /// quantization configuration including a fitted codebook. **Training is
    /// not undone.** A codebook is fitted from data that is now gone and cannot
    /// be refitted from an empty index, so a trained quantized index stays
    /// trained and its replacement graph is a quantized graph. An untrained one
    /// returns to collecting, since the records it had collected are gone.
    ///
    /// The internal id counter restarts, because nothing is left for a reissued
    /// id to collide with and restarting keeps the internal ids in step with
    /// the fresh graph's node indices, which is the invariant `list()`'s
    /// ordering and `compact()` both rest on.
    ///
    /// Clearing an index that is already empty is not an error and returns
    /// zero. It still replaces the graph, so it also returns the arena.
    ///
    /// llama-index probes `hasattr(index, "clear")` and raised
    /// `NotImplementedError` when it found nothing.
    pub fn clear(&self, py: Python<'_>) -> Result<usize, PyEngineError> {
        Ok(py.detach(|| self.inner.clear())?)
    }

    /// Replace one record's metadata without resupplying its vector.
    ///
    /// **Wholesale, not a merge.** The supplied mapping becomes the record's
    /// metadata and any key not in it is gone. That is what `add(overwrite=True)`
    /// already does, since it removes the record outright before inserting the
    /// replacement, so the two ways of re-tagging a record agree.
    ///
    /// Returns `false` for an id the index does not hold, and writes nothing in
    /// that case. `true` when the metadata was replaced. Passing an empty
    /// mapping clears the record's metadata, which is a write and returns
    /// `true`.
    ///
    /// **It touches `vector_metadata` and nothing else.** No graph work, no
    /// vector work, no id allocation, no training. The record keeps its internal
    /// id, its graph node, its stored vector and its quantized codes.
    ///
    /// That is the reason it exists. Re-tagging a document used to mean reading
    /// it back with `get_records` and adding it again with `overwrite=True`, and
    /// under `quantized_only` `get_records` returns a reconstruction rather than
    /// the vector supplied, so the round trip silently replaced the record's
    /// vector with an approximation of itself. It also stranded a graph node and
    /// re-ran the insertion. None of that happens here.
    pub fn update_metadata(
        &self,
        py: Python<'_>,
        id: String,
        metadata: &Bound<PyDict>,
    ) -> Result<bool, PyEngineError> {
        let fields = python_dict_to_value_map(metadata)?;
        Ok(py.detach(|| self.inner.update_metadata(&id, fields))?)
    }

    /// Rebuild the graph at a new degree, in place.
    ///
    /// **`m` is the one creation parameter nothing else can correct.** It is
    /// chosen from `expected_size` at `create()` and fixed there, so an index
    /// declared for 10,000 records and given a million runs at a degree meant
    /// for the smaller one, and no search width recovers the recall that costs.
    /// `dim` and `space` cannot be wrong that way, since an index at the wrong
    /// width or the wrong metric rejects or misranks its records outright, and
    /// `ef_construction` describes work already done.
    ///
    /// At least one of the three has to be given. Rebuilding the graph as it
    /// stands is `compact()`, which reclaims the nodes removals and overwrites
    /// leave behind.
    ///
    /// `expected_size` moves with `m` because the two are related. It selects
    /// the default `m`, it sizes the replacement graph's reservation, and it is
    /// what the overgrowth warning compares against, so a caller correcting one
    /// usually wants the other. Passing it also re-arms that warning.
    ///
    /// `ef_construction` moves with them because **the warning below names it as
    /// one of two remedies and a caller has to be able to take either.** Raising
    /// `m` past half of `ef_construction` switches the neighbour selection
    /// heuristic off, and the message says to raise `ef_construction` or lower
    /// `m`. It is honest to report afterwards for the same reason the rebuild is
    /// honest at all: every record really was re-inserted at the new width.
    ///
    /// **Everything except the graph survives untouched.** Each record is
    /// re-inserted under the internal id it already holds, so `id_map`,
    /// `rev_map`, the metadata store and every declared field's column stay
    /// correct without being rewritten, and the record any given id resolves to
    /// is the same before and after. A quantized index is rebuilt from its
    /// stored codes rather than re-encoded, so the codebook is not retrained and
    /// no record's code changes; a `quantized_with_raw` index carries its raw
    /// store over node by node.
    ///
    /// The three are held to the rules `create()` applies, on the same five
    /// values, and an invalid one raises the message `create()` raises for it.
    /// The `ef_construction` against `2 * m` pair is checked too, and warns where
    /// the pair puts the neighbour selection heuristic out of reach.
    ///
    /// Returns the node count of the graph it built, which equals the live
    /// record count.
    ///
    /// **The cost is a full rebuild by insertion**, the same cost as
    /// `compact()`, proportional to the live record count. The replacement is
    /// built in full before the old graph is dropped, so peak memory holds both
    /// and a failure part way through leaves the index exactly as it was.
    #[pyo3(signature = (m=None, expected_size=None, ef_construction=None))]
    pub fn rebuild(
        &self,
        py: Python<'_>,
        m: Option<usize>,
        expected_size: Option<usize>,
        ef_construction: Option<usize>,
    ) -> Result<usize, PyEngineError> {
        let plan = self.inner.plan_rebuild(m, expected_size, ef_construction)?;
        // Before the interpreter lock is released, because a Python warning
        // needs it. The pair is what decides it, exactly as at `create()`, and
        // **both of the remedies it names are reachable from here**: this takes
        // `ef_construction` as well as `m`, so a caller told to raise one or
        // lower the other can do either.
        construct::warn_if_selection_disabled(py, plan.m, plan.ef_construction)?;
        Ok(py.detach(|| self.inner.rebuild(plan))?)
    }

    /// Rebuild the graph in memory and reclaim the nodes removal and overwrite strand.
    ///
    /// `remove_point` clears a record from every storage map but cannot delete its
    /// graph node, and `add(overwrite=True)` is a removal followed by an insertion,
    /// so both leave behind a node that still holds a copy of the vector and both
    /// directions of adjacency while resolving to no record. Search already excludes
    /// those nodes, so this is a resource operation and not a correctness one. What it
    /// reclaims is their memory, their edge slots in live neighbour lists, and the
    /// traversal steps they cost.
    ///
    /// Returns the number of nodes reclaimed. Zero means the graph held no stranded
    /// nodes, in which case nothing is rebuilt and the call is a no-op.
    ///
    /// The cost is a full sequential rebuild, proportional to the live record count
    /// rather than to the amount of debris. Nothing outside the graph is touched.
    /// Internal ids, external ids, metadata, stored vectors, quantized codes, PQ
    /// training state and the id counter all survive unchanged, so the record any
    /// given id resolves to is the same before and after.
    ///
    /// The replacement graph is built in full before the old one is dropped, so peak
    /// memory holds both for the duration and a failure part way through leaves the
    /// index exactly as it was.
    ///
    /// This is never automatic. Calling it is a decision a deployment can schedule.
    ///
    /// The rebuild runs with the interpreter lock released, and every function
    /// it reaches is in the index crate, which names nothing of Python.
    pub fn compact(&self, py: Python<'_>) -> Result<usize, PyEngineError> {
        Ok(py.detach(|| self.inner.compact())?)
    }

    /// Get performance characteristics and limitations
    pub fn get_performance_info(&self) -> HashMap<String, String> {
        self.inner.performance_info()
    }

    /// Concurrent benchmark for search performance
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_concurrent_reads(
        &self,
        py: Python<'_>,
        query_count: usize,
        max_threads: Option<usize>,
    ) -> Result<HashMap<String, f64>, PyEngineError> {
        Ok(py.detach(|| self.inner.benchmark_reads(query_count, max_threads))?)
    }
}
