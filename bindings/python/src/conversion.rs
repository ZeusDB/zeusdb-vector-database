//! Conversion between Python objects and the `serde_json::Value` tree the
//! index stores metadata in.
//!
//! Both directions live here because both are needed in two places. The index
//! reads Python metadata on the way in and writes it back on the way out, and
//! the persistence loader writes it back too when it replays records through
//! `add`. `persistence.rs` used to carry its own copy of the outward direction
//! under the name `convert_json_value_to_python`, which is what this replaces.
//!
//! These are free functions rather than methods. Nothing here reads any field
//! of an index, so nothing here needs one.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use std::collections::HashMap;

/// How deeply a Python object may nest on the way in.
///
/// **This is a stack guard and not a schema rule.** The conversion below
/// recurses once per level of a nested mapping or list, and it used to do so
/// without a bound. A filter nested about four thousand deep killed the process
/// outright with a stack overflow, which is a crash rather than an error and
/// takes the interpreter with it. Deeper input now raises `ValueError` from the
/// level that would have been the next frame.
///
/// The number is `serde_json`'s own default recursion limit, which is what
/// reads `metadata.json` back. Matching it means anything this accepts can be
/// reloaded: a value one level past what the loader would parse is refused at
/// the point it is written rather than on the load that fails to read it.
///
/// The filter language has a second, much smaller limit,
/// `filter::MAX_FILTER_DEPTH`, which bounds how far `$and`, `$or` and `$not`
/// nest. That one is a contract about the language. This one is about the
/// stack, and it applies to metadata values too, since they arrive through the
/// same function.
pub(crate) const MAX_VALUE_DEPTH: usize = 128;

/// A Python mapping as a metadata map.
pub(crate) fn python_dict_to_value_map(
    py_dict: &Bound<PyDict>,
) -> PyResult<HashMap<String, Value>> {
    let mut map = HashMap::new();

    for (key, value) in py_dict.iter() {
        let string_key = key.extract::<String>()?;
        // The mapping itself is level one, so its values start at two.
        let json_value = convert_value(&value, 2)?;
        map.insert(string_key, json_value);
    }

    Ok(map)
}

/// One Python object as a `Value`.
///
/// The order the arms are tried in is the behaviour. `bool` comes before the
/// integer arm because Python's `bool` extracts as an integer, and the integer
/// arm comes before the float arm so a whole number stays exact rather than
/// becoming an f64. Anything that matches none of them keeps its `str()`.
pub(crate) fn python_object_to_value(py_obj: &Bound<PyAny>) -> PyResult<Value> {
    convert_value(py_obj, 1)
}

/// The recursion the two entry points above share, carrying the level it is at.
fn convert_value(py_obj: &Bound<PyAny>, depth: usize) -> PyResult<Value> {
    if depth > MAX_VALUE_DEPTH {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Nested value is deeper than {} levels, which is as far as mappings and lists nest.",
            MAX_VALUE_DEPTH
        )));
    }

    if py_obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = py_obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(i) = py_obj.extract::<i64>() {
        Ok(Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = py_obj.extract::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(f) {
            Ok(Value::Number(num))
        } else {
            Ok(Value::String(f.to_string()))
        }
    } else if let Ok(s) = py_obj.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(py_list) = py_obj.cast::<PyList>() {
        let mut vec = Vec::new();
        for item in py_list.iter() {
            vec.push(convert_value(&item, depth + 1)?);
        }
        Ok(Value::Array(vec))
    } else if let Ok(py_dict) = py_obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in py_dict.iter() {
            let string_key = key.extract::<String>()?;
            let json_value = convert_value(&value, depth + 1)?;
            map.insert(string_key, json_value);
        }
        Ok(Value::Object(map))
    } else {
        Ok(Value::String(py_obj.to_string()))
    }
}

/// A metadata map as a Python dict.
pub(crate) fn value_map_to_python(
    value_map: &HashMap<String, Value>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    for (key, value) in value_map {
        let py_value = value_to_python_object(value, py)?;
        dict.set_item(key, py_value)?;
    }

    Ok(dict.into_pyobject(py)?.to_owned().unbind().into_any())
}

/// One `Value` as a Python object.
///
/// A number that is neither an `i64` nor an `f64` comes back as its decimal
/// string, which is the only lossless answer for an integer wider than the two
/// and is what both copies of this did before they were merged.
pub(crate) fn value_to_python_object(value: &Value, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let py_obj = match value {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().unbind().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.to_owned().unbind().into_any()
            } else if let Some(f) = n.as_f64() {
                f.into_pyobject(py)?.to_owned().unbind().into_any()
            } else {
                n.to_string()
                    .into_pyobject(py)?
                    .to_owned()
                    .unbind()
                    .into_any()
            }
        }
        Value::String(s) => s.clone().into_pyobject(py)?.unbind().into_any(),
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(value_to_python_object(item, py)?)?;
            }
            py_list.unbind().into_any()
        }
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (k, v) in obj {
                py_dict.set_item(k, value_to_python_object(v, py)?)?;
            }
            py_dict.unbind().into_any()
        }
    };

    Ok(py_obj)
}
