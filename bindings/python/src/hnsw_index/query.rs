//! Reading a query out of Python, and writing its page and its plan back.
//!
//! A query is a list of arms, each a mapping naming one query on one space,
//! a filter every arm shares, the page size, the candidates each arm
//! contributes and the fusion. What leaves `parse_query` is owned Rust
//! holding no Python object, so the search runs with the interpreter lock
//! released. A text arm's tokenizer has already run by then, with the lock
//! held and no engine guard taken, and what the arm carries into the engine
//! is its terms; see the crate's `tokenizer` module.

use super::input::extract_sparse_vector;
use super::HNSWIndex;
use crate::conversion::{python_dict_to_value_map, value_map_to_python};
use crate::PyEngineError;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use zeusdb_vector_core::{compile_filter, Error, Filter, Fusion, IdfScope, SparseVector};
use zeusdb_vector_hnsw::{AdmitShape, Arm, Plan, Query, QueryHit, DEFAULT_RRF_K};

/// The keys an arm may carry, by the query it names.
const DENSE_KEYS: [&str; 3] = ["vector", "ef_search", "rerank"];
const SPARSE_KEYS: [&str; 2] = ["sparse", "idf"];
const TEXT_KEYS: [&str; 2] = ["text", "idf"];
const FUSION_KEYS: [&str; 2] = ["type", "k"];

/// The one fusion this build has, as a query names it.
const RRF: &str = "rrf";

/// One arm, owned.
pub(super) enum ParsedArm {
    Dense {
        vector: Vec<f32>,
        ef: Option<usize>,
        rerank: Option<usize>,
    },
    Sparse {
        vector: SparseVector,
        idf: IdfScope,
    },
    /// A text arm, tokenized already.
    Terms {
        terms: Vec<String>,
        idf: IdfScope,
    },
}

/// A query, owned, holding no Python object.
pub(super) struct ParsedQuery {
    arms: Vec<ParsedArm>,
    filter: Option<Filter>,
    k: usize,
    fetch: Option<usize>,
    fusion: Fusion,
}

impl ParsedQuery {
    /// The arms as the engine borrows them.
    pub(super) fn arms(&self) -> Vec<Arm<'_>> {
        self.arms
            .iter()
            .map(|arm| match arm {
                ParsedArm::Dense { vector, ef, rerank } => Arm::Dense {
                    vector: vector.as_slice(),
                    ef: *ef,
                    rerank: *rerank,
                },
                ParsedArm::Sparse { vector, idf } => Arm::Sparse {
                    vector: vector.as_ref(),
                    idf: *idf,
                },
                ParsedArm::Terms { terms, idf } => Arm::Terms {
                    terms: terms.as_slice(),
                    idf: *idf,
                },
            })
            .collect()
    }

    /// The query as the engine takes it, over arms borrowed above.
    pub(super) fn query<'a>(&'a self, arms: &'a [Arm<'a>]) -> Query<'a> {
        Query {
            arms,
            filter: self.filter.as_ref(),
            k: self.k,
            fetch: self.fetch,
            fusion: self.fusion,
        }
    }
}

fn value_error(message: String) -> PyEngineError {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message).into()
}

fn type_error(message: String) -> PyEngineError {
    PyErr::new::<pyo3::exceptions::PyTypeError, _>(message).into()
}

/// The Python type's name, for a message.
fn type_name(value: &Bound<PyAny>) -> String {
    value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "an object".to_string())
}

/// The keys a mapping carries, each as a string.
fn keys_of(dict: &Bound<PyDict>) -> PyResult<Vec<String>> {
    dict.keys()
        .iter()
        .map(|key| key.extract::<String>())
        .collect()
}

/// A value under `key`, treating `None` as absent.
fn present<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
}

impl HNSWIndex {
    /// Read every argument of `query` and `explain` into owned Rust.
    ///
    /// A text arm is tokenized here, with the interpreter lock held and no
    /// engine guard taken, so the callable a caller declared runs under the
    /// lock it already holds and the search that follows runs without it.
    pub(super) fn parse_query(
        &self,
        arms: &Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        fetch: Option<usize>,
        fusion: Option<&Bound<PyAny>>,
    ) -> Result<ParsedQuery, PyEngineError> {
        let arms: Vec<Bound<PyAny>> = arms.extract().map_err(|_| {
            type_error(format!(
                "arms must be a list of mappings, one per arm, got {}",
                type_name(arms)
            ))
        })?;
        let arms = arms
            .iter()
            .enumerate()
            .map(|(position, arm)| self.parse_arm(arm, position))
            .collect::<Result<Vec<ParsedArm>, PyEngineError>>()?;
        // Compiled once, before any record is examined, as `search` does.
        let filter = filter
            .map(python_dict_to_value_map)
            .transpose()?
            .as_ref()
            .map(compile_filter)
            .transpose()?;
        Ok(ParsedQuery {
            arms,
            filter,
            k: top_k,
            fetch,
            fusion: parse_fusion(fusion)?,
        })
    }

    /// One arm: a mapping naming exactly one of `vector`, `sparse` or
    /// `text`, with the options that query takes and nothing else.
    fn parse_arm(&self, arm: &Bound<PyAny>, position: usize) -> Result<ParsedArm, PyEngineError> {
        let dict = arm.cast::<PyDict>().map_err(|_| {
            type_error(format!(
                "arms[{}] must be a mapping naming one of 'vector', 'sparse' or 'text', \
                 got {}",
                position,
                type_name(arm)
            ))
        })?;
        let named: Vec<&str> = ["vector", "sparse", "text"]
            .into_iter()
            .filter(|key| present(dict, key).ok().flatten().is_some())
            .collect();
        let kind = match named.as_slice() {
            [kind] => *kind,
            [] => {
                return Err(value_error(format!(
                    "arms[{}] names none of 'vector', 'sparse' or 'text'. An arm asks one \
                     space with one query: a dense vector, a sparse vector of term ids and \
                     weights, or a text.",
                    position
                )))
            }
            several => {
                return Err(value_error(format!(
                    "arms[{}] names {}, and an arm asks one space with one query. Put each \
                     in an arm of its own.",
                    position,
                    several
                        .iter()
                        .map(|key| format!("'{}'", key))
                        .collect::<Vec<String>>()
                        .join(" and ")
                )))
            }
        };
        let allowed: &[&str] = match kind {
            "vector" => &DENSE_KEYS,
            "sparse" => &SPARSE_KEYS,
            _ => &TEXT_KEYS,
        };
        for key in keys_of(dict)? {
            if !allowed.contains(&key.as_str()) {
                return Err(value_error(format!(
                    "arms[{}] carries '{}', which a '{}' arm does not take. It takes {}.",
                    position,
                    key,
                    kind,
                    allowed
                        .iter()
                        .map(|key| format!("'{}'", key))
                        .collect::<Vec<String>>()
                        .join(", ")
                )));
            }
        }
        let value = present(dict, kind)?.expect("the arm names the key");
        match kind {
            "vector" => Ok(ParsedArm::Dense {
                vector: dense_vector(&value, position)?,
                ef: present(dict, "ef_search")?
                    .map(|ef| ef.extract::<usize>())
                    .transpose()?,
                rerank: present(dict, "rerank")?
                    .map(|rerank| rerank.extract::<usize>())
                    .transpose()?,
            }),
            "sparse" => {
                let vector = extract_sparse_vector(&value)
                    .map_err(|detail| value_error(format!("arms[{}] {}", position, detail)))?;
                let idf = parse_idf(dict, position)?;
                // A space with a text layer takes text alone, since its
                // term ids are the dictionary's to issue. The arm's own
                // rules come first, the vector's as the engine applies
                // them, so a malformed arm reads the same message on
                // either kind of space, and then the space's, through an
                // accessor that takes no guard where there is no layer.
                if self.inner.term_count().is_some() {
                    vector.as_ref().validate()?;
                    return Err(Error::SparseVectorOnTextSpace.into());
                }
                Ok(ParsedArm::Sparse { vector, idf })
            }
            _ => {
                let text = value.extract::<String>().map_err(|_| {
                    type_error(format!(
                        "arms[{}]['text'] must be a str, got {}",
                        position,
                        type_name(&value)
                    ))
                })?;
                // The tokenizer runs here, under the interpreter lock the
                // caller holds and under no engine guard. Its own failure
                // comes back as the exception it raised.
                let terms = self.inner.tokenize(&text)?;
                Ok(ParsedArm::Terms {
                    terms,
                    idf: parse_idf(dict, position)?,
                })
            }
        }
    }
}

/// A dense arm's vector: a list of numbers or a one dimensional array of
/// either float width.
fn dense_vector(value: &Bound<PyAny>, position: usize) -> Result<Vec<f32>, PyEngineError> {
    if let Ok(array) = value.cast::<PyArray1<f32>>() {
        return Ok(array.readonly().as_slice()?.to_vec());
    }
    if let Ok(array) = value.cast::<PyArray1<f64>>() {
        return Ok(array
            .readonly()
            .as_slice()?
            .iter()
            .map(|&component| component as f32)
            .collect());
    }
    value.extract::<Vec<f32>>().map_err(|_| {
        type_error(format!(
            "arms[{}]['vector'] must be a list of numbers or a one dimensional array, got {}",
            position,
            type_name(value)
        ))
    })
}

/// The corpus a term's rarity is measured over: the admitted records, which
/// is the default, or every live record.
fn parse_idf(dict: &Bound<PyDict>, position: usize) -> Result<IdfScope, PyEngineError> {
    let Some(value) = present(dict, "idf")? else {
        return Ok(IdfScope::Corpus);
    };
    let spelled = value.extract::<String>().map_err(|_| {
        type_error(format!(
            "arms[{}]['idf'] must be 'corpus' or 'global', got {}",
            position,
            type_name(&value)
        ))
    })?;
    match spelled.as_str() {
        "corpus" => Ok(IdfScope::Corpus),
        "global" => Ok(IdfScope::Global),
        other => Err(value_error(format!(
            "arms[{}]['idf'] is '{}', and it is 'corpus' to weight a term by its rarity \
             over the records the filter admits, or 'global' to weight it over every live \
             record.",
            position, other
        ))),
    }
}

/// The fusion a query names: nothing, the name, or a mapping with the name
/// and the constant.
fn parse_fusion(value: Option<&Bound<PyAny>>) -> Result<Fusion, PyEngineError> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(Fusion::default());
    };
    let (kind, k) = if let Ok(name) = value.extract::<String>() {
        (name, None)
    } else if let Ok(dict) = value.cast::<PyDict>() {
        for key in keys_of(dict)? {
            if !FUSION_KEYS.contains(&key.as_str()) {
                return Err(value_error(format!(
                    "fusion carries '{}', and a fusion is {{'type': 'rrf', 'k': 60.0}}.",
                    key
                )));
            }
        }
        let kind = present(dict, "type")?
            .ok_or_else(|| {
                value_error("fusion is missing 'type', which names the rule; 'rrf' is the one this build has.".to_string())
            })?
            .extract::<String>()
            .map_err(|_| type_error("fusion['type'] must be a str".to_string()))?;
        let k = present(dict, "k")?
            .map(|k| k.extract::<f32>())
            .transpose()
            .map_err(|_| type_error("fusion['k'] must be a number".to_string()))?;
        (kind, k)
    } else {
        return Err(type_error(format!(
            "fusion must be 'rrf' or a mapping {{'type': 'rrf', 'k': 60.0}}, got {}",
            type_name(value)
        )));
    };
    if kind != RRF {
        return Err(value_error(format!(
            "fusion type '{}' is not one this build has. 'rrf' fuses the arms' pages by \
             reciprocal rank, being 1 / (k + rank) summed over the pages a record appears on.",
            kind
        )));
    }
    Ok(Fusion::ReciprocalRank {
        k: k.unwrap_or(DEFAULT_RRF_K),
    })
}

/// A page as the list of dicts Python receives, each hit carrying its id,
/// its score, its metadata and its place on every arm's page.
pub(super) fn page_to_python(
    hits: Vec<QueryHit>,
    py: Python<'_>,
) -> Result<Py<PyAny>, PyEngineError> {
    let page = PyList::empty(py);
    for hit in hits {
        let dict = PyDict::new(py);
        dict.set_item("id", hit.id)?;
        dict.set_item("score", hit.score)?;
        dict.set_item("metadata", value_map_to_python(&hit.metadata, py)?)?;
        let contributions = PyList::empty(py);
        for contribution in hit.contributions {
            let entry = PyDict::new(py);
            entry.set_item("arm", contribution.arm)?;
            entry.set_item("rank", contribution.rank)?;
            entry.set_item("score", contribution.score)?;
            contributions.append(entry)?;
        }
        dict.set_item("contributions", contributions)?;
        page.append(dict)?;
    }
    Ok(page.into())
}

/// A plan as the dict Python receives.
pub(super) fn plan_to_python(plan: &Plan, py: Python<'_>) -> Result<Py<PyDict>, PyEngineError> {
    let dict = PyDict::new(py);
    let admit = PyDict::new(py);
    match plan.admit {
        AdmitShape::All => admit.set_item("shape", "all")?,
        AdmitShape::Bitmap { admitted } => {
            admit.set_item("shape", "bitmap")?;
            admit.set_item("admitted", admitted)?;
        }
        AdmitShape::Sorted { admitted } => {
            admit.set_item("shape", "sorted")?;
            admit.set_item("admitted", admitted)?;
        }
        AdmitShape::Bounded { bound } => {
            admit.set_item("shape", "bounded")?;
            admit.set_item("bound", bound)?;
        }
        AdmitShape::Predicate => admit.set_item("shape", "predicate")?,
    }
    dict.set_item("admit", admit)?;
    let arms = PyList::empty(py);
    for arm in &plan.arms {
        let entry = PyDict::new(py);
        entry.set_item("space", arm.space.as_str())?;
        entry.set_item("kind", arm.kind.name())?;
        entry.set_item("fetch", arm.fetch)?;
        entry.set_item("cost_ns", arm.cost.work_ns)?;
        entry.set_item("exact", arm.cost.exact)?;
        arms.append(entry)?;
    }
    dict.set_item("arms", arms)?;
    match plan.fusion {
        Some(fusion) => dict.set_item("fusion", fusion_to_python(fusion, py)?)?,
        None => dict.set_item("fusion", py.None())?,
    }
    Ok(dict.unbind())
}

fn fusion_to_python(fusion: Fusion, py: Python<'_>) -> Result<Bound<'_, PyDict>, PyEngineError> {
    let dict = PyDict::new(py);
    match fusion {
        Fusion::ReciprocalRank { k } => {
            dict.set_item("type", RRF)?;
            dict.set_item("k", k)?;
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "The plan carries a fusion this binding does not name",
            )
            .into())
        }
    }
    Ok(dict)
}
