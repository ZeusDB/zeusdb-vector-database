//! Reading records and query vectors out of Python.
//!
//! `add` accepts five input shapes and `search` accepts three, and everything
//! that turns any of them into owned Rust lives here. What leaves this file is
//! `ParsedRecords` and validated query vectors, neither of which holds a Python
//! object, which is what lets insertion and search release the interpreter lock.
//!
//! Two validators that look alike are deliberately not one.
//! `extract_single_vector` raises a `PyErr` and `extract_single_vector_safe`
//! returns a `String` that `add` reports against the record it belongs to, and
//! their messages differ.

use super::{HNSWIndex, ParsedRecords};
use crate::conversion::{python_dict_to_value_map, python_object_to_value};
use crate::PyEngineError;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{error, trace};
use zeusdb_vector_core::Error;
impl HNSWIndex {
    /// Pure function for vector normalization
    fn normalize_vector(&self, vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vector // Return unchanged for zero vectors
        }
    }

    /// Process vector according to distance space
    pub(super) fn process_vector_for_space(&self, vector: Vec<f32>) -> Vec<f32> {
        match self.space.to_lowercase().as_str() {
            "cosine" => self.normalize_vector(vector),
            // Future extensions:
            // "l2" => self.preprocess_l2(vector),
            // "l1" => self.preprocess_l1(vector),
            _ => vector,
        }
    }

    /// Helper for query processing (mirrors extract_single_vector validation)
    pub(super) fn validate_and_process_query_vector(
        &self,
        vector: Vec<f32>,
    ) -> Result<Vec<f32>, Error> {
        // Same validation as extract_single_vector
        if vector.is_empty() {
            error!(
                operation = "query_validation",
                error = "empty_vector",
                "Search vector cannot be empty"
            );
            return Err(Error::SearchVectorEmpty);
        }
        if vector.len() != self.dim {
            error!(
                operation = "query_validation",
                error = "dimension_mismatch",
                expected = self.dim,
                actual = vector.len(),
                "Search vector dimension mismatch"
            );
            return Err(Error::SearchVectorDimension {
                expected: self.dim,
                got: vector.len(),
            });
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
                return Err(Error::SearchVectorNotFinite {
                    index: i,
                    value: val,
                });
            }
        }

        // Apply same processing as storage vectors
        Ok(self.process_vector_for_space(vector))
    }

    /// Parse input data into (id, vector, metadata) tuples with error collection
    ///
    /// Two error channels, and which one a mistake takes is the distinction
    /// this function exists to draw. A bad record is collected into `errors`
    /// and reported through `AddResult`, because the other records in the batch
    /// are still what the caller asked for. A malformed call raises, because
    /// there is no record set to report against. The only mistakes in the
    /// second class are the ones `check_batch_lengths` names.
    pub(super) fn parse_input_data(
        &self,
        data: &Bound<PyAny>,
    ) -> PyResult<(ParsedRecords, Vec<String>)> {
        let mut parsed_vectors = Vec::new();
        let mut errors = Vec::new();

        if let Ok(dict) = data.cast::<PyDict>() {
            self.parse_dict_input_safe(dict, &mut parsed_vectors, &mut errors)?;
        } else if let Ok(list) = data.cast::<PyList>() {
            self.parse_list_input_safe(list, &mut parsed_vectors, &mut errors);
        } else if let Ok(np_array) = data.cast::<PyArray2<f32>>() {
            if let Err(e) = self.parse_numpy_input_safe(np_array, &mut parsed_vectors, &mut errors)
            {
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

        Ok((parsed_vectors, errors))
    }

    /// Safe dictionary parsing that collects errors
    ///
    /// Returns `Err` only for the length and type rule on the parallel arrays.
    /// Every other failure is collected.
    fn parse_dict_input_safe(
        &self,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
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
                    let id = match dict.get_item("id") {
                        Ok(Some(id_item)) => id_item
                            .extract::<String>()
                            .unwrap_or_else(|_| self.generate_id()),
                        _ => self.generate_id(),
                    };

                    let metadata = match dict.get_item("metadata") {
                        Ok(Some(meta_item)) => {
                            if let Ok(meta_dict) = meta_item.cast::<PyDict>() {
                                python_dict_to_value_map(meta_dict).unwrap_or_default()
                            } else {
                                HashMap::new()
                            }
                        }
                        _ => HashMap::new(),
                    };

                    parsed_vectors.push((id, vector, metadata));
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
            // Batch format. The parallel arrays are checked against the vector
            // count before anything is parsed, and a disagreement raises rather
            // than being collected. Everything else the batch can get wrong is
            // a property of one record and is still reported per record.
            self.check_batch_lengths(dict)?;
            if let Err(e) = self.parse_batch_format(dict, parsed_vectors, errors) {
                errors.push(format!("Batch parsing error: {}", e));
            }
        }

        Ok(())
    }

    /// How many vectors a batch dict carries, or `None` if it carries no
    /// recognised vector field.
    ///
    /// Mirrors `parse_batch_format`'s dispatch exactly, key for key and cast
    /// for cast, so the count checked here is the count that loop will run to.
    /// An unrecognised or wrongly typed vector field returns `None` and is left
    /// for `parse_batch_format` to report in its own words.
    fn batch_vector_count(dict: &Bound<PyDict>) -> PyResult<Option<usize>> {
        for key in ["vectors", "embeddings"] {
            if let Some(item) = dict.get_item(key)? {
                if let Ok(list) = item.cast::<PyList>() {
                    return Ok(Some(list.len()));
                } else if let Ok(np_array) = item.cast::<PyArray2<f32>>() {
                    let count = np_array.readonly().shape().first().copied();
                    return Ok(count);
                }
                return Ok(None);
            }
        }

        if let Some(item) = dict.get_item("values")? {
            if let Ok(list) = item.cast::<PyList>() {
                return Ok(Some(list.len()));
            }
            return Ok(None);
        }

        Ok(None)
    }

    /// The one rule that raises rather than counting an error
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
    /// A parallel array that is not a list is the same loss by another route.
    /// The NumPy branch resolved `ids` with `cast::<PyList>().ok()`, so a tuple
    /// or an `ndarray` of ids was discarded whole and every record took a
    /// generated id, while the list branch raised on the same input. Both now
    /// say so.
    fn check_batch_lengths(&self, dict: &Bound<PyDict>) -> PyResult<()> {
        let Some(vector_count) = Self::batch_vector_count(dict)? else {
            return Ok(());
        };

        let vector_field = ["vectors", "embeddings", "values"]
            .into_iter()
            .find(|key| dict.contains(key).unwrap_or(false))
            .unwrap_or("vectors");

        // 'metadata' is read as a spelling of 'metadatas' by both batch parsers,
        // and only where 'metadatas' is absent, so it is checked under the same
        // rule and only in the same case.
        let metadata_field = if dict.get_item("metadatas")?.is_some() {
            "metadatas"
        } else {
            "metadata"
        };

        for (field, singular) in [("ids", "id"), (metadata_field, "metadata mapping")] {
            let Some(item) = dict.get_item(field)? else {
                continue;
            };
            let Ok(list) = item.cast::<PyList>() else {
                if field != "ids" {
                    // A non-list under a metadata key is ignored by both
                    // parsers today and carries no id, so it is left alone.
                    continue;
                }
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "add expected '{}' to be a list, got {}. A batch pairs '{}' with \
                     '{}' by position and reads only a list that way, so any other type \
                     is discarded whole and every record takes a generated id.",
                    field,
                    item.get_type().name()?,
                    field,
                    vector_field
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

    /// Handle Format 3 & 5: Batch format - WORKING SOLUTION
    fn parse_batch_format(
        &self,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        // Process each key path immediately without storing references

        // Try "vectors" key
        if let Some(vectors_item) = dict.get_item("vectors")? {
            if let Ok(list) = vectors_item.cast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = vectors_item.cast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return Ok(self.parse_numpy_with_context(
                    np_array,
                    dict,
                    parsed_vectors,
                    errors,
                )?);
            }
        }

        // Try "embeddings" key
        if let Some(embeddings_item) = dict.get_item("embeddings")? {
            if let Ok(list) = embeddings_item.cast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = embeddings_item.cast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return Ok(self.parse_numpy_with_context(
                    np_array,
                    dict,
                    parsed_vectors,
                    errors,
                )?);
            }
        }

        // Try "values" key
        if let Some(values_item) = dict.get_item("values")? {
            if let Ok(list) = values_item.cast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "values field must be a list in batch format",
                ));
            }
        }

        // No valid vector data found
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Missing vector data. Expected one of: 'vectors', 'embeddings', or 'values' key",
        ))
    }

    /// Helper method to process vector list (extracted to avoid code duplication)
    fn process_vector_list(
        &self,
        vectors: &Bound<PyList>,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
    ) -> PyResult<()> {
        // Process each vector in the batch
        for (i, vector_item) in vectors.iter().enumerate() {
            let vector = self.extract_single_vector(&vector_item)?;

            // Extract ID from "ids" array
            let id = match dict.get_item("ids")? {
                Some(item) => {
                    let ids_list = item.cast::<PyList>()?;
                    if i < ids_list.len() {
                        ids_list.get_item(i)?.extract::<String>()?
                    } else {
                        self.generate_id()
                    }
                }
                None => self.generate_id(),
            };

            // Extract metadata from "metadatas" or "metadata" arrays
            let meta = match dict
                .get_item("metadatas")?
                .or_else(|| dict.get_item("metadata").ok().flatten())
            {
                Some(item) => {
                    if let Ok(meta_list) = item.cast::<PyList>() {
                        if i < meta_list.len() {
                            let metadata_item = meta_list.get_item(i)?;
                            if let Ok(meta_dict) = metadata_item.cast::<PyDict>() {
                                python_dict_to_value_map(meta_dict)?
                            } else if metadata_item.is_none() {
                                HashMap::new()
                            } else {
                                let mut map = HashMap::new();
                                let value = python_object_to_value(&metadata_item)?;
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
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> Result<(), PyEngineError> {
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
            return Err(Error::BatchArrayShape {
                dim: self.dim,
                shape: shape.to_vec(),
            }
            .into());
        }

        let flat = readonly.as_slice()?;
        let num_vectors = shape[0];

        // Extract IDs array
        let ids_list = dict
            .get_item("ids")?
            .and_then(|item| item.cast::<PyList>().ok().cloned());

        // Extract metadata array
        let metadatas_list = dict
            .get_item("metadatas")?
            .or_else(|| dict.get_item("metadata").ok().flatten())
            .and_then(|item| item.cast::<PyList>().ok().cloned());

        trace!(
            operation = "parse_numpy_context",
            num_vectors = num_vectors,
            has_ids = ids_list.is_some(),
            has_metadata = metadatas_list.is_some(),
            "Processing vectors with context"
        );

        let mut rejected = 0usize;

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = &flat[start_idx..end_idx];

            // Resolve the caller's id before validating, so a rejected row can
            // be named without advancing the internal id counter for a record
            // that is never stored.
            let provided_id = match &ids_list {
                Some(ids) if i < ids.len() => ids.get_item(i)?.extract::<String>().ok(),
                _ => None,
            };

            if let Some((component, value)) = Self::first_non_finite(raw_vector) {
                let label = provided_id.clone().unwrap_or_else(|| format!("row_{}", i));
                error!(
                    operation = "parse_numpy_context",
                    error = "invalid_value",
                    vector_id = %label,
                    index = component,
                    value = value,
                    "NumPy row contains a non-finite value"
                );
                errors.push(format!(
                    "Vector {}: contains invalid value at index {}: {} (must be finite)",
                    label, component, value
                ));
                rejected += 1;
                continue;
            }

            let processed_vector = self.process_vector_for_space(raw_vector.to_vec());

            // Get ID from provided IDs or generate
            let id = match provided_id {
                Some(id) => id,
                None => self.generate_id(),
            };

            // Get metadata from provided metadata or use empty
            let metadata = if let Some(metas) = &metadatas_list {
                if i < metas.len() {
                    let meta_item = metas.get_item(i)?;
                    if let Ok(meta_dict) = meta_item.cast::<PyDict>() {
                        python_dict_to_value_map(meta_dict)?
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

        trace!(
            operation = "parse_numpy_context_complete",
            parsed_count = num_vectors - rejected,
            rejected_count = rejected,
            "NumPy parsing completed"
        );
        Ok(())
    }

    /// Safe list parsing that collects errors instead of failing immediately
    fn parse_list_input_safe(
        &self,
        list: &Bound<PyList>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) {
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
                        // Extract ID
                        let id = match item_dict.get_item("id") {
                            Ok(Some(id_item)) => id_item
                                .extract::<String>()
                                .unwrap_or_else(|_| self.generate_id()),
                            _ => self.generate_id(),
                        };

                        // Extract metadata
                        let metadata = match item_dict.get_item("metadata") {
                            Ok(Some(meta_item)) => {
                                if let Ok(meta_dict) = meta_item.cast::<PyDict>() {
                                    python_dict_to_value_map(meta_dict).unwrap_or_default()
                                } else {
                                    // Handle non-dict metadata
                                    let mut map = HashMap::new();
                                    if let Ok(value) = python_object_to_value(&meta_item) {
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
    fn parse_numpy_input_safe(
        &self,
        np_array: &Bound<PyArray2<f32>>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> Result<(), String> {
        // This is the same as your current parse_numpy_input but returns Result<(), String>
        let readonly = np_array.readonly();
        let shape = readonly.shape();

        if shape.len() != 2 || shape[1] != self.dim {
            return Err(format!(
                "NumPy array must have shape (N, {}), got {:?}",
                self.dim, shape
            ));
        }

        let flat = readonly
            .as_slice()
            .map_err(|e| format!("NumPy access error: {}", e))?;
        let num_vectors = shape[0];

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = &flat[start_idx..end_idx];

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

            let processed_vector = self.process_vector_for_space(raw_vector.to_vec());
            let id = self.generate_id();
            parsed_vectors.push((id, processed_vector, HashMap::new()));
        }

        Ok(())
    }

    /// Report the first non-finite component of a vector, if there is one
    ///
    /// The two NumPy branches read their rows straight out of the buffer, so
    /// they skip the per-value check that `extract_single_vector` runs. A NaN
    /// that reaches the graph degrades every later query rather than only the
    /// one that carried it, so both branches route through this.
    fn first_non_finite(vector: &[f32]) -> Option<(usize, f32)> {
        vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
            .map(|(index, value)| (index, *value))
    }

    /// Extract a single vector from various Python types (enhanced)
    fn extract_single_vector(&self, data: &Bound<PyAny>) -> Result<Vec<f32>, PyEngineError> {
        let vector = if let Ok(array1d) = data.cast::<PyArray1<f32>>() {
            // NumPy 1D array
            array1d.readonly().as_slice()?.to_vec()
        } else if let Ok(list) = data.cast::<PyList>() {
            // Python list
            list.iter()
                .map(|item| item.extract::<f32>())
                .collect::<PyResult<Vec<f32>>>()?
        } else {
            // Direct extraction (e.g., from other numeric arrays)
            data.extract::<Vec<f32>>()?
        };

        // Comprehensive validation
        if vector.is_empty() {
            return Err(Error::VectorEmpty.into());
        }

        if vector.len() != self.dim {
            return Err(Error::VectorDimension {
                expected: self.dim,
                got: vector.len(),
            }
            .into());
        }

        // Check for invalid values
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(Error::VectorNotFinite {
                    index: i,
                    value: val,
                }
                .into());
            }
        }

        // ✅ Apply space-specific processing
        Ok(self.process_vector_for_space(vector))
    }

    /// Generate a unique ID for a vector
    ///
    /// From a counter of its own rather than from the internal id counter. See
    /// `HNSWIndex::generated_ids` for what each of the two has to guarantee and
    /// why one counter could not do both.
    fn generate_id(&self) -> String {
        let mut counter = self.generated_ids.lock().unwrap();
        *counter += 1;
        format!("vec_{}", *counter)
    }

    /// Safe version of extract_single_vector that returns String errors instead of PyErr
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

        // Same validation as extract_single_vector, but with String errors
        if vector.is_empty() {
            return Err("Vector cannot be empty".to_string());
        }
        if vector.len() != self.dim {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
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

        Ok(self.process_vector_for_space(vector))
    }
}
