//! What a vector becomes once it is out of Python.
//!
//! The binding turns five input shapes into records and three into query
//! vectors, and what it hands over is owned Rust. The two rules that apply
//! after that are here: how a vector is processed for the space it will be
//! stored in, and how a query is checked before it is processed the same way.
//! Both are the space's, because both depend on the metric and the width, and
//! the collection delegates to them for the binding's sake.
//!
//! The parsing itself stays in the binding, in `hnsw_index/input.rs`. This is
//! the engine's half of that module and its records keep that module's target.

use super::{Collection, Space};
use tracing::error;
use zeusdb_vector_core::Error;

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::input";

impl Space {
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
        match self.metric.to_lowercase().as_str() {
            "cosine" => self.normalize_vector(vector),
            // Future extensions:
            // "l2" => self.preprocess_l2(vector),
            // "l1" => self.preprocess_l1(vector),
            _ => vector,
        }
    }

    /// Helper for query processing (mirrors extract_single_vector validation)
    pub(super) fn validate_query(&self, vector: Vec<f32>) -> Result<Vec<f32>, Error> {
        // Same validation as extract_single_vector
        if vector.is_empty() {
            error!(
                target: LOG_TARGET,
                operation = "query_validation",
                error = "empty_vector",
                "Search vector cannot be empty"
            );
            return Err(Error::SearchVectorEmpty);
        }
        if vector.len() != self.dim {
            error!(
                target: LOG_TARGET,
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
                    target: LOG_TARGET,
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
}

impl Collection {
    /// A vector as the space stores it, which on a cosine index is the unit
    /// vector. The binding applies this to every parsed record.
    pub fn process_vector_for_space(&self, vector: Vec<f32>) -> Vec<f32> {
        self.space.process_vector_for_space(vector)
    }

    /// A query vector checked for its width and for non-finite values, then
    /// processed as a stored vector is.
    pub fn validate_query(&self, vector: Vec<f32>) -> Result<Vec<f32>, Error> {
        self.space.validate_query(vector)
    }
}
