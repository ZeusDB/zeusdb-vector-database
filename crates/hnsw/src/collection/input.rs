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

use super::{Collection, DenseSpace};
use tracing::error;
use zeusdb_vector_core::{norm_f64, screen, Error, Screen};

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::input";

impl DenseSpace {
    /// The unit vector of a finite vector, and an all-zero vector unchanged.
    ///
    /// The norm is taken in `f32` and divides every value. Where that norm is
    /// not finite, its squares having overflowed, or is zero for a vector that
    /// is not all zeros, its squares having underflowed, the norm and the
    /// division are taken in `f64` instead, where the square of every `f32` is
    /// exact, so the vector is stored at unit length rather than as the zero
    /// vector or unchanged. Every other vector takes the `f32` norm and
    /// division alone.
    fn normalize_vector(&self, vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 && norm.is_finite() {
            vector.iter().map(|x| x / norm).collect()
        } else if vector.iter().all(|&x| x == 0.0) {
            vector // Return unchanged for zero vectors
        } else {
            normalized_in_f64(&vector)
        }
    }

    /// Whether the bound on a vector's norm applies to this space, which it
    /// does under every metric but cosine. See `NORM_BOUND`.
    pub(super) fn bounded(&self) -> bool {
        self.metric != "cosine"
    }

    /// The values of `vector` held finite and, under l2, dot or l1, its
    /// Euclidean norm held to `NORM_BOUND`.
    pub(super) fn screen(&self, vector: &[f32]) -> Screen {
        screen(vector, self.bounded())
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
        match self.screen(&vector) {
            Screen::Admitted => {}
            Screen::NotFinite { index, value } => {
                error!(
                    target: LOG_TARGET,
                    operation = "query_validation",
                    error = "invalid_value",
                    index = index,
                    value = value,
                    "Search vector contains invalid value"
                );
                return Err(Error::SearchVectorNotFinite { index, value });
            }
            Screen::PastBound { norm } => {
                error!(
                    target: LOG_TARGET,
                    operation = "query_validation",
                    error = "past_bound",
                    norm = norm,
                    "Search vector's norm is past the bound"
                );
                return Err(Error::SearchVectorPastBound { norm });
            }
        }

        // Apply same processing as storage vectors
        Ok(self.process_vector_for_space(vector))
    }
}

/// The unit vector of a finite vector that is not all zeros, its norm and
/// every division taken in `f64`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a value of a unit vector, at most one in magnitude, narrowed to the f32 the \
              index stores"
)]
fn normalized_in_f64(vector: &[f32]) -> Vec<f32> {
    let norm = norm_f64(vector);
    vector
        .iter()
        .map(|&x| (f64::from(x) / norm) as f32)
        .collect()
}

impl Collection {
    /// A vector as the space stores it, which on a cosine index is the unit
    /// vector. The binding applies this to every parsed record.
    pub fn process_vector_for_space(&self, vector: Vec<f32>) -> Vec<f32> {
        self.dense().process_vector_for_space(vector)
    }

    /// A query vector checked for its width, for non-finite values and,
    /// under l2, dot or l1, for a Euclidean norm past `NORM_BOUND`, then
    /// processed as a stored vector is.
    pub fn validate_query(&self, vector: Vec<f32>) -> Result<Vec<f32>, Error> {
        self.dense().validate_query(vector)
    }

    /// A record's vector held finite and, under l2, dot or l1, its Euclidean
    /// norm held to `NORM_BOUND`, which the binding applies to every parsed
    /// record before it processes it.
    pub fn screen_vector(&self, vector: &[f32]) -> Screen {
        self.dense().screen(vector)
    }
}
