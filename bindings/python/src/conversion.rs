//! Conversion between Python objects and the `serde_json::Value` tree the
//! index stores metadata in, and of a search page into the list of dicts
//! Python receives.
//!
//! Both directions live here because both are needed in two places. The index
//! reads Python metadata on the way in and writes it back on the way out, and
//! the persistence loader writes it back too when it replays records through
//! `add`. `persistence.rs` used to carry its own copy of the outward direction
//! under the name `convert_json_value_to_python`, which is what this replaces.
//!
//! These are free functions rather than methods. Nothing here reads any field
//! of an index, so nothing here needs one.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use zeusdb_vector_core::Error;
use zeusdb_vector_hnsw::{FieldRef, QueryHits};

/// A count or a size a call takes, read from a Python integer, with a
/// negative one refused naming the argument. The signatures take `i64`,
/// which is the index-sized integer Python itself reads, so PyO3 refuses an
/// integer past what it holds with `OverflowError` before this runs.
pub(crate) fn count(value: i64, name: &'static str) -> Result<usize, Error> {
    if value < 0 {
        return Err(Error::ArgumentNegative { name, value });
    }
    Ok(usize::try_from(value).unwrap_or(usize::MAX))
}

/// [`count`] for an argument that may be left out.
pub(crate) fn count_if_given(
    value: Option<i64>,
    name: &'static str,
) -> Result<Option<usize>, Error> {
    value.map(|value| count(value, name)).transpose()
}

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

/// A Python mapping as a metadata map, held to the metadata rule.
///
/// A record's metadata, the metadata `update_metadata` replaces and every
/// filter come through here. Each value is a metadata value, by
/// [`metadata_value`]'s rule, and a refusal names the key it sits under.
pub(crate) fn python_dict_to_value_map(
    py_dict: &Bound<PyDict>,
) -> PyResult<HashMap<String, Value>> {
    let mut map = HashMap::new();

    for (key, value) in py_dict.iter() {
        let string_key = key.extract::<String>()?;
        // The mapping itself is level one, so its values start at two.
        let json_value = convert_value(&value, 2, Rule::Metadata)
            .map_err(|refused| refused.under(Step::Key(string_key.clone())).into_error())?;
        map.insert(string_key, json_value);
    }

    Ok(map)
}

/// One metadata value as a `Value`, a record's metadata entry that is not
/// a mapping being one.
///
/// A metadata value is `None`, a `bool`, an integer from -2^63 to 2^63 - 1,
/// a finite float, a `str`, a list of metadata values or a mapping of `str`
/// keys to metadata values, nested at most [`MAX_VALUE_DEPTH`] levels. Each
/// comes back as the value it was, so any other value is refused, naming
/// where it sits: a type JSON does not hold with `TypeError`, and a float
/// that is not finite or an integer outside that range with `ValueError`.
/// A `numpy` integer, float or bool and a `str` subclass are read by their
/// value, which comes back exact.
pub(crate) fn metadata_value(py_obj: &Bound<PyAny>) -> PyResult<Value> {
    convert_value(py_obj, 1, Rule::Metadata).map_err(Refused::into_error)
}

/// One field of a declaration as a `Value`, for the parser of the
/// declaration to hold to its own rules.
///
/// The order the arms are tried in is the behaviour. `bool` comes before the
/// integer arm because Python's `bool` extracts as an integer, and the integer
/// arm comes before the float arm so a whole number stays exact rather than
/// becoming an f64. Anything that matches none of them keeps its `str()`,
/// which the declaration's parser then refuses in its own words.
pub(crate) fn python_object_to_value(py_obj: &Bound<PyAny>) -> PyResult<Value> {
    convert_value(py_obj, 1, Rule::Declaration).map_err(Refused::into_error)
}

/// Which values a conversion keeps: the metadata rule of [`metadata_value`],
/// or a declaration's, which keeps every value as the arms of
/// [`python_object_to_value`] read it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    Metadata,
    Declaration,
}

/// One step into what the caller handed over, from the outside in.
enum Step {
    Key(String),
    Item(usize),
}

/// A value the metadata rule refuses, and why.
enum Kind {
    /// A type JSON does not hold, by its Python name.
    Type(String),
    /// A float that is not finite, as Python writes it.
    NotFinite(String),
    /// An integer outside -2^63 to 2^63 - 1, as Python writes it.
    OutOfRange(String),
}

/// Why a conversion stopped: an error whose words are settled, or a value
/// the metadata rule refuses, with the steps to it gathered on the way out.
enum Refused {
    Settled(PyErr),
    Value { kind: Kind, steps: Vec<Step> },
}

impl From<PyErr> for Refused {
    fn from(error: PyErr) -> Self {
        Refused::Settled(error)
    }
}

impl Refused {
    /// The refusal one step further out.
    fn under(mut self, step: Step) -> Self {
        if let Refused::Value { steps, .. } = &mut self {
            steps.push(step);
        }
        self
    }

    /// The exception a caller reads, naming where the value sits.
    fn into_error(self) -> PyErr {
        let (kind, steps) = match self {
            Refused::Settled(error) => return error,
            Refused::Value { kind, steps } => (kind, steps),
        };
        let place = if steps.is_empty() {
            "the entry".to_string()
        } else {
            steps
                .iter()
                .rev()
                .map(|step| match step {
                    Step::Key(key) => format!("key '{}'", key),
                    Step::Item(item) => format!("item {}", item),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        match kind {
            Kind::Type(name) => PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "the value at {} is a {}, and a metadata value is None, a bool, an int, a \
                 float, a str, a list or a dict, so convert it to one of those.",
                place, name
            )),
            Kind::NotFinite(value) => PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "the value at {} is {}, a float that is not finite, and a metadata value is a \
                 finite float, so give None or a string in its place.",
                place, value
            )),
            Kind::OutOfRange(value) => PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "the value at {} is {}, outside the integers from -2^63 to 2^63 - 1 that a \
                 metadata value holds, so give it as a string.",
                place, value
            )),
        }
    }
}

/// A value the metadata rule refuses, of `kind`, with no step taken yet.
fn refused(kind: Kind) -> Refused {
    Refused::Value {
        kind,
        steps: Vec::new(),
    }
}

/// Whether `py_obj` is a float of a type other than Python's own, being one
/// `numbers.Real` holds and `numbers.Rational` does not, which is how
/// `numpy`'s float types register, and which leaves out `Fraction`.
fn is_other_float(py_obj: &Bound<PyAny>) -> PyResult<bool> {
    let numbers = py_obj.py().import(intern!(py_obj.py(), "numbers"))?;
    Ok(
        py_obj.is_instance(&numbers.getattr(intern!(py_obj.py(), "Real"))?)?
            && !py_obj.is_instance(&numbers.getattr(intern!(py_obj.py(), "Rational"))?)?,
    )
}

/// A float as a `Value` under `rule`. A float that is not finite is
/// refused under the metadata rule, and kept as its text under a
/// declaration's.
fn float_value(py_obj: &Bound<PyAny>, f: f64, rule: Rule) -> Result<Value, Refused> {
    match serde_json::Number::from_f64(f) {
        Some(num) => Ok(Value::Number(num)),
        None if rule == Rule::Metadata => Err(refused(Kind::NotFinite(py_obj.str()?.to_string()))),
        None => Ok(Value::String(f.to_string())),
    }
}

/// The recursion the entry points above share, carrying the level it is at
/// and the rule it keeps.
fn convert_value(py_obj: &Bound<PyAny>, depth: usize, rule: Rule) -> Result<Value, Refused> {
    if depth > MAX_VALUE_DEPTH {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Nested value is deeper than {} levels, which is as far as mappings and lists nest.",
            MAX_VALUE_DEPTH
        ))
        .into());
    }

    if py_obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = py_obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    match py_obj.extract::<i64>() {
        Ok(i) => return Ok(Value::Number(serde_json::Number::from(i))),
        // An integer past 64 bits, which the metadata rule refuses.
        Err(error)
            if rule == Rule::Metadata
                && error.is_instance_of::<pyo3::exceptions::PyOverflowError>(py_obj.py()) =>
        {
            return Err(refused(Kind::OutOfRange(py_obj.str()?.to_string())));
        }
        Err(_) => {}
    }
    match rule {
        Rule::Metadata => {
            if py_obj.is_instance_of::<PyFloat>() {
                return float_value(py_obj, py_obj.extract::<f64>()?, rule);
            }
        }
        Rule::Declaration => {
            if let Ok(f) = py_obj.extract::<f64>() {
                return float_value(py_obj, f, rule);
            }
        }
    }
    if let Ok(s) = py_obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(py_list) = py_obj.cast::<PyList>() {
        let mut vec = Vec::new();
        for (item, value) in py_list.iter().enumerate() {
            vec.push(
                convert_value(&value, depth + 1, rule).map_err(|r| r.under(Step::Item(item)))?,
            );
        }
        return Ok(Value::Array(vec));
    }
    if let Ok(py_dict) = py_obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in py_dict.iter() {
            let string_key = key.extract::<String>()?;
            let json_value = convert_value(&value, depth + 1, rule)
                .map_err(|r| r.under(Step::Key(string_key.clone())))?;
            map.insert(string_key, json_value);
        }
        return Ok(Value::Object(map));
    }
    match rule {
        Rule::Metadata if is_other_float(py_obj)? => {
            float_value(py_obj, py_obj.extract::<f64>()?, rule)
        }
        Rule::Metadata => Err(refused(Kind::Type(py_obj.get_type().name()?.to_string()))),
        Rule::Declaration => Ok(Value::String(py_obj.to_string())),
    }
}

/// A metadata map as a Python dict, its keys in the map's order, which is
/// code point order.
pub(crate) fn value_map_to_python(
    value_map: &BTreeMap<String, Value>,
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

/// One query's hits as the list of dicts Python receives.
///
/// Each dict carries `id`, `score`, `metadata` and, where the search returned
/// vectors, `vector`, in that order. The four keys are interned for the
/// process and each field name is made once for the page, so a hit makes
/// Python strings for its id and its string values alone. A value that is not
/// a string goes through `value_to_python_object`, which is what
/// `value_map_to_python` applies to the same value.
///
/// Each hit's metadata keys go in in code point order, as every other call
/// hands metadata over. The page's names are ranked once, and each hit's
/// fields are put in rank order before its dict is built.
pub(crate) fn hits_to_python(hits: QueryHits, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    let id_key = intern!(py, "id");
    let score_key = intern!(py, "score");
    let metadata_key = intern!(py, "metadata");
    let vector_key = intern!(py, "vector");
    let page_names: Vec<&str> = hits.names().collect();
    let names: Vec<Bound<'_, PyString>> = page_names
        .iter()
        .map(|name| PyString::new(py, name))
        .collect();
    // Each name's place in code point order, by the page's number for it.
    let mut by_name: Vec<usize> = (0..page_names.len()).collect();
    by_name.sort_unstable_by_key(|&number| page_names[number]);
    let mut rank = vec![0usize; page_names.len()];
    for (place, &number) in by_name.iter().enumerate() {
        rank[number] = place;
    }
    let mut fields: Vec<(usize, FieldRef<'_>)> = Vec::new();
    let mut output = Vec::with_capacity(hits.len());
    for hit in hits.iter() {
        fields.clear();
        fields.extend(hit.fields());
        fields.sort_unstable_by_key(|&(name, _)| rank[name]);
        let metadata = PyDict::new(py);
        for &(name, value) in &fields {
            match value {
                FieldRef::Text(text) => metadata.set_item(&names[name], PyString::new(py, text))?,
                FieldRef::Other(value) => {
                    metadata.set_item(&names[name], value_to_python_object(value, py)?)?
                }
            }
        }
        let dict = PyDict::new(py);
        dict.set_item(id_key, hit.id())?;
        dict.set_item(score_key, hit.score())?;
        dict.set_item(metadata_key, metadata)?;
        if let Some(vector) = hit.vector() {
            // A list of Python floats, as every release has returned. Both
            // adapters read it as a list, and one of them tests for it.
            dict.set_item(vector_key, vector)?;
        }
        output.push(dict.unbind());
    }
    Ok(output)
}

/// A batch's hits as one list of dicts per query, in query order.
pub(crate) fn batch_hits_to_python(
    batches: Vec<QueryHits>,
    py: Python<'_>,
) -> PyResult<Vec<Vec<Py<PyDict>>>> {
    let mut output = Vec::with_capacity(batches.len());
    for hits in batches {
        output.push(hits_to_python(hits, py)?);
    }
    Ok(output)
}
