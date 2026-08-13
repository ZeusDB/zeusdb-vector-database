//! Metadata filter evaluation.
//!
//! Two entry points. `matches_filter` judges one record's metadata against a
//! filter, and `validate_filter_conditions` rejects an operator the engine does
//! not implement before any record is examined. Everything else below is
//! reached only from those two.
//!
//! These were fifteen methods on `HNSWIndex`. Every one took `&self` and not one
//! read a field of it, so they are free functions here, which is what they
//! already were.

use pyo3::prelude::*;
use serde_json::Value;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
/// A JSON number as either an exact integer or a float. `serde_json` stores an
/// integer and a float in different variants, and comparing those variants is
/// what made 10 and 10.0 unequal under some operators and equal under others.
enum NumericValue {
    Integer(i128),
    Float(f64),
}
// 4. DATA CONVERSION & FILTERING (12 methods)
pub(crate) fn matches_filter(
    metadata: &HashMap<String, Value>,
    filter: &HashMap<String, Value>,
) -> PyResult<bool> {
    for (field, condition) in filter {
        if !field_matches(metadata, field, condition)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn field_matches(
    metadata: &HashMap<String, Value>,
    field: &str,
    condition: &Value,
) -> PyResult<bool> {
    let field_value = match metadata.get(field) {
        Some(value) => value,
        None => return Ok(false),
    };

    match condition {
        // A map is always the operator form. Direct equality against a
        // nested object has no syntax of its own, because the two forms
        // would be indistinguishable, so it is written {"eq": {...}}.
        Value::Object(ops) => evaluate_value_conditions(field_value, ops),
        _ => Ok(values_equal(field_value, condition)),
    }
}

/// Reject an operator the engine does not implement, before any record is
/// examined. Checking during evaluation is not enough on its own, because a
/// record that lacks the field never reaches the operator and a filter that
/// fails an earlier field short circuits, so whether the typo is noticed
/// would depend on the data. `evaluate_operator` is the only list of
/// operator names, so validation cannot disagree with dispatch about what
/// is known. The field value here is a placeholder, which is sound because
/// every operator helper is total and the unknown operator arm is the one
/// error the dispatch can produce.
pub(crate) fn validate_filter_conditions(filter: &HashMap<String, Value>) -> PyResult<()> {
    for condition in filter.values() {
        if let Value::Object(operations) = condition {
            for (op, target_value) in operations {
                evaluate_operator(&Value::Null, op, target_value)?;
            }
        }
    }
    Ok(())
}

fn evaluate_value_conditions(
    field_value: &Value,
    operations: &serde_json::Map<String, Value>,
) -> PyResult<bool> {
    for (op, target_value) in operations {
        if !evaluate_operator(field_value, op, target_value)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn evaluate_operator(field_value: &Value, op: &str, target_value: &Value) -> PyResult<bool> {
    match op {
        "eq" => Ok(values_equal(field_value, target_value)),
        "ne" => Ok(!values_equal(field_value, target_value)),
        "gt" => compare_values(field_value, target_value, CmpOrdering::is_gt),
        "gte" => compare_values(field_value, target_value, CmpOrdering::is_ge),
        "lt" => compare_values(field_value, target_value, CmpOrdering::is_lt),
        "lte" => compare_values(field_value, target_value, CmpOrdering::is_le),
        "contains" => value_contains(field_value, target_value),
        "startswith" => value_starts_with(field_value, target_value),
        "endswith" => value_ends_with(field_value, target_value),
        "in" => value_in_array(field_value, target_value),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown filter operation: {}",
            op
        ))),
    }
}

/// Equality over the whole value tree. Numbers compare by magnitude, so a
/// stored integer matches an equal float, and arrays and objects compare
/// element by element so their numbers do too. Every other pairing keeps
/// `serde_json` equality, which is why a boolean is not equal to a number
/// and a numeric string is not equal to a number.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(left), Value::Number(right)) => {
            compare_numbers(left, right) == Some(CmpOrdering::Equal)
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right.iter())
                    .all(|(item, other)| values_equal(item, other))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, item)| {
                    right
                        .get(key)
                        .is_some_and(|other| values_equal(item, other))
                })
        }
        _ => a == b,
    }
}

/// Order two JSON numbers by magnitude. Integers compare as integers, so
/// two values above 2^53 that share an f64 representation stay distinct,
/// and a mixed pair compares exactly rather than through a lossy cast.
fn compare_numbers(a: &serde_json::Number, b: &serde_json::Number) -> Option<CmpOrdering> {
    match (numeric_value(a)?, numeric_value(b)?) {
        (NumericValue::Integer(left), NumericValue::Integer(right)) => Some(left.cmp(&right)),
        (NumericValue::Float(left), NumericValue::Float(right)) => left.partial_cmp(&right),
        (NumericValue::Integer(left), NumericValue::Float(right)) => {
            compare_integer_to_float(left, right)
        }
        (NumericValue::Float(left), NumericValue::Integer(right)) => {
            compare_integer_to_float(right, left).map(CmpOrdering::reverse)
        }
    }
}

/// `i128` holds every `serde_json` integer, which is an `i64` or a `u64`,
/// so the widening is lossless.
fn numeric_value(number: &serde_json::Number) -> Option<NumericValue> {
    if let Some(value) = number.as_i64() {
        Some(NumericValue::Integer(value as i128))
    } else if let Some(value) = number.as_u64() {
        Some(NumericValue::Integer(value as i128))
    } else {
        number.as_f64().map(NumericValue::Float)
    }
}

/// Order an integer against a float without casting the integer to f64.
/// The float splits into a truncated part, which converts to an integer
/// exactly, and a fraction that breaks the tie when the integer parts are
/// equal. A float outside the `i128` range saturates on conversion, and
/// the comparison still lands on the correct side because every integer
/// reaching this point fits in a `u64`.
fn compare_integer_to_float(integer: i128, float: f64) -> Option<CmpOrdering> {
    if float.is_nan() {
        return None;
    }
    if float.is_infinite() {
        return Some(if float.is_sign_positive() {
            CmpOrdering::Less
        } else {
            CmpOrdering::Greater
        });
    }

    let truncated = float.trunc();
    let integer_part = truncated as i128;
    Some(match integer.cmp(&integer_part) {
        CmpOrdering::Equal => truncated.partial_cmp(&float)?,
        ordering => ordering,
    })
}

fn compare_values<F>(a: &Value, b: &Value, op: F) -> PyResult<bool>
where
    F: Fn(CmpOrdering) -> bool,
{
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => Ok(compare_numbers(n1, n2).is_some_and(op)),
        _ => Ok(false),
    }
}

fn value_contains(field: &Value, target: &Value) -> PyResult<bool> {
    match (field, target) {
        (Value::String(s1), Value::String(s2)) => Ok(s1.contains(s2)),
        (Value::Array(arr), val) => Ok(arr.iter().any(|item| values_equal(item, val))),
        _ => Ok(false),
    }
}

fn value_starts_with(field: &Value, target: &Value) -> PyResult<bool> {
    match (field, target) {
        (Value::String(s1), Value::String(s2)) => Ok(s1.starts_with(s2)),
        _ => Ok(false),
    }
}

fn value_ends_with(field: &Value, target: &Value) -> PyResult<bool> {
    match (field, target) {
        (Value::String(s1), Value::String(s2)) => Ok(s1.ends_with(s2)),
        _ => Ok(false),
    }
}

fn value_in_array(field: &Value, target: &Value) -> PyResult<bool> {
    match target {
        Value::Array(arr) => Ok(arr.iter().any(|item| values_equal(item, field))),
        _ => Ok(false),
    }
}
