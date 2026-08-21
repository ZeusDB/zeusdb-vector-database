//! Metadata filter compilation and evaluation.
//!
//! Two entry points. `compile_filter` turns the mapping a caller handed in into
//! a [`Filter`] tree, rejecting everything the engine cannot evaluate before any
//! record is examined, and `matches_filter` judges one record's metadata against
//! that tree. Everything else below is reached only from those two.
//!
//! **The compile step is what makes evaluation total.** It used to be
//! `validate_filter_conditions`, which walked the caller's map, proved that
//! every operator name was one the dispatch knew, and then threw the walk away
//! so that evaluation re-read the same map per record and re-matched the same
//! operator strings. Four call sites carried a `debug_assert!` explaining why
//! the error arm they were forced to write could not fire. The tree removes the
//! arm rather than explaining it: an operator name is resolved once into [`Op`],
//! a group's shape is checked once, and `matches_filter` returns `bool` because
//! there is no longer anything for it to fail on.
//!
//! These were fifteen methods on `HNSWIndex`. Every one took `&self` and not one
//! read a field of it, so they are free functions here, which is what they
//! already were.

use pyo3::prelude::*;
use serde_json::Value;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;

/// How deeply `$and`, `$or` and `$not` may nest.
///
/// Depth 1 is the mapping the caller passed. Each group adds one, so
/// `{"$or": [{"$not": {"a": 1}}]}` is depth 3.
///
/// The reason for a limit is not the stack. A filter ten groups deep has up to
/// 2^10 leaves, which is past anything a person writes or an adapter generates
/// from a query, so the number is chosen to be unreachable by accident and
/// stated rather than left open. A public API that accepts arbitrary user input
/// should not have "as deep as you like" in its contract.
///
/// The stack is guarded a layer earlier and by a different number. A nested
/// mapping is converted from Python before any of this runs, and that recursion
/// is capped by `conversion::MAX_VALUE_DEPTH`.
pub(crate) const MAX_FILTER_DEPTH: usize = 10;

/// The three keys that name a group rather than a metadata field.
///
/// **The `$` namespace is not reserved, these three names are.** A field called
/// `$price` still filters, because reserving the prefix would break more
/// filters today and break another one every time an operator is added. A field
/// called `$or` no longer filters, and that is the whole cost of this form.
const GROUP_AND: &str = "$and";
const GROUP_OR: &str = "$or";
const GROUP_NOT: &str = "$not";

/// A JSON number as either an exact integer or a float. `serde_json` stores an
/// integer and a float in different variants, and comparing those variants is
/// what made 10 and 10.0 unequal under some operators and equal under others.
enum NumericValue {
    Integer(i128),
    Float(f64),
}

/// A compiled filter.
///
/// `All` is the conjunction a mapping of fields has always meant, so a flat
/// filter compiles to `All` over its fields and nothing else about it changes.
/// `Any` and `Not` have no form in the flat language and are reachable only
/// through `$or` and `$not`.
pub(crate) enum Filter {
    All(Vec<Filter>),
    Any(Vec<Filter>),
    Not(Box<Filter>),
    Field { name: String, test: FieldTest },
}

/// What one field is asked.
///
/// The two arms are the two forms a condition can take. A plain value is
/// equality and a mapping is operators, all of which must hold.
pub(crate) enum FieldTest {
    Equals(Value),
    Operators(Vec<(Op, Value)>),
}

/// An operator, resolved from its name once at compile time.
///
/// The dispatch below is the only list of these, and `parse_op` is the only
/// place a name becomes one, so validation cannot disagree with evaluation
/// about what is known.
#[derive(Clone, Copy)]
pub(crate) enum Op {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
    StartsWith,
    EndsWith,
    In,
    Nin,
    Any,
    All,
}

// ============================================================================
// COMPILATION
// ============================================================================

/// Compile a caller's filter, rejecting anything the engine cannot evaluate.
///
/// Everything that can go wrong with a filter goes wrong here, before any
/// record is read: an operator name the dispatch does not know, a group whose
/// value is not the shape that group takes, and a tree past
/// [`MAX_FILTER_DEPTH`]. What comes back evaluates against every record without
/// failing, which is why `matches_filter` has no error channel and the
/// traversal predicate needs none either.
pub(crate) fn compile_filter(filter: &HashMap<String, Value>) -> PyResult<Filter> {
    // The caller's mapping is a `HashMap`, whose iteration order varies per
    // process. Sorting fixes the order the compiled conjunction is evaluated
    // in, so two runs of one search short circuit at the same field. Nothing
    // depends on which order it is, only on it being the same one twice.
    let mut entries: Vec<(&String, &Value)> = filter.iter().collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    compile_entries(entries.into_iter(), 1)
}

/// One mapping's entries as a conjunction, at a known depth.
///
/// Every entry is either a group or a field, decided by the key alone. Deciding
/// it by the value's type instead would make `{"$or": [...]}` mean one thing
/// for a user who has a field named `$or` holding a list and another for a user
/// who does not, which is a rule that depends on the data.
fn compile_entries<'a>(
    entries: impl Iterator<Item = (&'a String, &'a Value)>,
    depth: usize,
) -> PyResult<Filter> {
    let mut children = Vec::new();

    for (key, condition) in entries {
        children.push(match key.as_str() {
            GROUP_AND => Filter::All(compile_group(GROUP_AND, condition, depth)?),
            GROUP_OR => Filter::Any(compile_group(GROUP_OR, condition, depth)?),
            GROUP_NOT => Filter::Not(Box::new(compile_negation(condition, depth)?)),
            _ => Filter::Field {
                name: key.clone(),
                test: compile_field(condition)?,
            },
        });
    }

    Ok(Filter::All(children))
}

/// The branches of an `$and` or an `$or`.
///
/// An empty list is allowed and is the identity of the operation it names, so
/// `{"$and": []}` matches every record and `{"$or": []}` matches none. That is
/// what `all` and `any` already do with an empty target array, so the language
/// answers the empty case the same way in both places.
fn compile_group(key: &str, condition: &Value, depth: usize) -> PyResult<Vec<Filter>> {
    let Value::Array(branches) = condition else {
        return Err(reserved_key_error(key, condition));
    };
    check_depth(key, depth)?;

    let mut compiled = Vec::with_capacity(branches.len());
    for branch in branches {
        let Value::Object(map) = branch else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Every entry of \"{}\" must be a filter mapping, for example \
                 {{\"lang\": \"en\"}}, but one of them is {}.",
                key,
                describe(branch)
            )));
        };
        compiled.push(compile_entries(map.iter(), depth + 1)?);
    }
    Ok(compiled)
}

/// The one filter a `$not` negates.
///
/// One mapping and not a list, so the tree has a single node shape and there is
/// no convention to invent about whether a list under `$not` is negated
/// together or separately. `{"$not": {"$or": [...]}}` is the second of those
/// and is written out.
fn compile_negation(condition: &Value, depth: usize) -> PyResult<Filter> {
    let Value::Object(map) = condition else {
        return Err(reserved_key_error(GROUP_NOT, condition));
    };
    check_depth(GROUP_NOT, depth)?;
    compile_entries(map.iter(), depth + 1)
}

/// One field's condition.
///
/// A mapping is always the operator form. Direct equality against a nested
/// object has no syntax of its own, because the two forms would be
/// indistinguishable, so it is written `{"eq": {...}}`.
fn compile_field(condition: &Value) -> PyResult<FieldTest> {
    match condition {
        Value::Object(operations) => {
            let mut tests = Vec::with_capacity(operations.len());
            for (op, target) in operations {
                tests.push((parse_op(op)?, target.clone()));
            }
            Ok(FieldTest::Operators(tests))
        }
        plain => Ok(FieldTest::Equals(plain.clone())),
    }
}

/// An operator name as an [`Op`].
///
/// The message names the operator and nothing else, which is what it said
/// before this became a compile step and is what the README quotes.
fn parse_op(op: &str) -> PyResult<Op> {
    Ok(match op {
        "eq" => Op::Eq,
        "ne" => Op::Ne,
        "gt" => Op::Gt,
        "gte" => Op::Gte,
        "lt" => Op::Lt,
        "lte" => Op::Lte,
        "contains" => Op::Contains,
        "startswith" => Op::StartsWith,
        "endswith" => Op::EndsWith,
        "in" => Op::In,
        "nin" => Op::Nin,
        "any" => Op::Any,
        "all" => Op::All,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown filter operation: {}",
                op
            )))
        }
    })
}

fn check_depth(key: &str, depth: usize) -> PyResult<()> {
    if depth >= MAX_FILTER_DEPTH {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Filter groups nest to {} levels and \"{}\" would open level {}.",
            MAX_FILTER_DEPTH,
            key,
            depth + 1
        )));
    }
    Ok(())
}

/// The error a reserved key carrying the wrong shape raises.
///
/// This is also the error a user with a metadata field named `$or` sees, and it
/// is why the collision is loud. Their filter stops working, which is the cost
/// of the reservation, but it stops working with a message that names the key
/// rather than by quietly selecting the wrong records.
fn reserved_key_error(key: &str, condition: &Value) -> PyErr {
    let shape = if key == GROUP_NOT {
        "one filter mapping, for example {\"$not\": {\"lang\": \"en\"}}"
    } else {
        "a list of filter mappings, for example {\"$or\": [{\"lang\": \"en\"}, {\"lang\": \"es\"}]}"
    };
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "\"{}\" is a reserved filter key and takes {}, but it was given {}. A \
         metadata field named \"{}\" cannot be filtered on.",
        key,
        shape,
        describe(condition),
        key
    ))
}

/// What a value is, for an error message. The value itself is not printed,
/// because a filter can carry a caller's data and an error is not the place to
/// echo it back.
fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a mapping",
    }
}

// ============================================================================
// EVALUATION
// ============================================================================

/// Judge one record's metadata against a compiled filter.
///
/// It short circuits in both directions. A conjunction stops at the first
/// branch that fails and a disjunction at the first that holds, which is what
/// `Iterator::all` and `Iterator::any` do.
pub(crate) fn matches_filter(metadata: &HashMap<String, Value>, filter: &Filter) -> bool {
    match filter {
        Filter::All(branches) => branches
            .iter()
            .all(|branch| matches_filter(metadata, branch)),
        Filter::Any(branches) => branches
            .iter()
            .any(|branch| matches_filter(metadata, branch)),
        Filter::Not(inner) => !matches_filter(metadata, inner),
        Filter::Field { name, test } => field_matches(metadata, name, test),
    }
}

impl Filter {
    /// Whether this admits every record whatever its metadata, decided from the
    /// tree's shape alone.
    ///
    /// It exists for `remove_where`, which refuses a filter that would delete
    /// the whole index. Testing the caller's mapping for emptiness was enough
    /// while `{}` was the only way to write "everything"; `{"$and": []}` and
    /// `{"$not": {"$or": []}}` are two more, so the question is now asked of the
    /// tree.
    ///
    /// A field test never admits a record that lacks the field, so no leaf is
    /// unconditional and the recursion bottoms out at `false`.
    pub(crate) fn matches_every_record(&self) -> bool {
        match self {
            Filter::All(branches) => branches.iter().all(Filter::matches_every_record),
            Filter::Any(branches) => branches.iter().any(Filter::matches_every_record),
            Filter::Not(inner) => inner.matches_no_record(),
            Filter::Field { .. } => false,
        }
    }

    /// Whether this admits no record at all, decided from the tree's shape
    /// alone. The other half of the pair above, since negating one is the
    /// other.
    ///
    /// Conservative at the leaves in the safe direction. `{"tags": {"any": []}}`
    /// does match nothing and is answered `false` here, which only means a
    /// `$not` wrapping it is not recognised as unconditional, so a delete that
    /// would have been refused runs and removes everything the filter names.
    fn matches_no_record(&self) -> bool {
        match self {
            Filter::All(branches) => branches.iter().any(Filter::matches_no_record),
            Filter::Any(branches) => branches.iter().all(Filter::matches_no_record),
            Filter::Not(inner) => inner.matches_every_record(),
            Filter::Field { .. } => false,
        }
    }
}

/// Judge one field of one record.
///
/// **A record that does not carry the field never reaches an operator.** The
/// absence is answered here, once, for all thirteen of them, so `ne`, `nin` and
/// `all` exclude such a record exactly as `eq`, `in` and `any` do. That is a
/// rule about the record rather than about the operator: this engine holds that
/// a field it has no value for cannot be judged, so nothing is asserted of it
/// either way.
///
/// The alternative, where a negating operator admits a record the field is
/// missing from, makes the language incoherent at its own boundary. `nin`
/// against a one element list is `ne` against that element, and `ne` has
/// answered false for an absent field since relay 44 fixed it there. Splitting
/// them would mean `{"tag": {"ne": "x"}}` and `{"tag": {"nin": ["x"]}}` return
/// different record sets while meaning the same thing.
///
/// What it costs is that "the field is absent" has no operator that expresses
/// it. `$not` expresses it instead, without touching this rule:
/// `{"tag": {"all": []}}` is the empty conjunction and so holds for every value
/// the field can carry, which makes it "the field is present", and
/// `{"$not": {"tag": {"all": []}}}` is its complement.
fn field_matches(metadata: &HashMap<String, Value>, field: &str, test: &FieldTest) -> bool {
    let Some(field_value) = metadata.get(field) else {
        return false;
    };
    field_test_matches(field_value, test)
}

/// Judge one value against one field's condition.
///
/// **This is the whole of what an operator decides, and it is called from
/// exactly two places.** [`field_matches`] calls it with the value it found in
/// a record's metadata, and [`crate::columns::ColumnStore::select`] calls it
/// with the value a column holds. A column and the metadata walk therefore
/// cannot disagree about whether a value matches, because there is one function
/// and not two.
///
/// It takes the value rather than the record, so the absence rule stays where
/// it was. A record that does not carry the field never reaches here, and a
/// column that holds no value for a slot never calls it.
pub(crate) fn field_test_matches(field_value: &Value, test: &FieldTest) -> bool {
    match test {
        FieldTest::Equals(target) => values_equal(field_value, target),
        FieldTest::Operators(operations) => operations
            .iter()
            .all(|(op, target)| evaluate_operator(field_value, *op, target)),
    }
}

fn evaluate_operator(field_value: &Value, op: Op, target_value: &Value) -> bool {
    match op {
        Op::Eq => values_equal(field_value, target_value),
        Op::Ne => !values_equal(field_value, target_value),
        Op::Gt => compare_values(field_value, target_value, CmpOrdering::is_gt),
        Op::Gte => compare_values(field_value, target_value, CmpOrdering::is_ge),
        Op::Lt => compare_values(field_value, target_value, CmpOrdering::is_lt),
        Op::Lte => compare_values(field_value, target_value, CmpOrdering::is_le),
        Op::Contains => value_contains(field_value, target_value),
        Op::StartsWith => value_starts_with(field_value, target_value),
        Op::EndsWith => value_ends_with(field_value, target_value),
        Op::In => value_in_array(field_value, target_value),
        Op::Nin => !value_in_array(field_value, target_value),
        Op::Any => value_intersects_any(field_value, target_value),
        Op::All => value_contains_all(field_value, target_value),
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

fn compare_values<F>(a: &Value, b: &Value, op: F) -> bool
where
    F: Fn(CmpOrdering) -> bool,
{
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => compare_numbers(n1, n2).is_some_and(op),
        _ => false,
    }
}

fn value_contains(field: &Value, target: &Value) -> bool {
    match (field, target) {
        (Value::String(s1), Value::String(s2)) => s1.contains(s2),
        (Value::Array(arr), val) => arr.iter().any(|item| values_equal(item, val)),
        _ => false,
    }
}

fn value_starts_with(field: &Value, target: &Value) -> bool {
    match (field, target) {
        (Value::String(s1), Value::String(s2)) => s1.starts_with(s2),
        _ => false,
    }
}

fn value_ends_with(field: &Value, target: &Value) -> bool {
    match (field, target) {
        (Value::String(s1), Value::String(s2)) => s1.ends_with(s2),
        _ => false,
    }
}

fn value_in_array(field: &Value, target: &Value) -> bool {
    match target {
        Value::Array(arr) => arr.iter().any(|item| values_equal(item, field)),
        _ => false,
    }
}

/// Whether an array field shares at least one element with the target array.
///
/// `contains` asks whether an array field holds one named value, and a field
/// maps to one condition mapping which cannot carry an operator twice, so
/// asking whether it holds any of several needed an operator of its own. A
/// non-array field is compared as a single element, which makes this the
/// array-valued generalisation of `in` and keeps the two agreeing on a scalar
/// field.
///
/// An empty target array matches nothing, which is what an empty disjunction
/// is, and it agrees with `{"$or": []}`.
fn value_intersects_any(field: &Value, target: &Value) -> bool {
    let Value::Array(wanted) = target else {
        return false;
    };
    wanted.iter().any(|want| match field {
        Value::Array(held) => held.iter().any(|item| values_equal(item, want)),
        _ => values_equal(field, want),
    })
}

/// Whether an array field holds every element of the target array.
///
/// A field maps to one condition mapping, and a condition mapping cannot carry
/// the same operator twice, so a conjunction of `contains` over one field had no
/// syntax. This is that conjunction.
///
/// An empty target array matches every record that carries the field, which is
/// what an empty conjunction is and agrees with `{"$and": []}`. It is the only
/// case where this admits a record whose field holds nothing, and it is what
/// makes `{"$not": {"field": {"all": []}}}` mean "the field is absent".
fn value_contains_all(field: &Value, target: &Value) -> bool {
    let Value::Array(wanted) = target else {
        return false;
    };
    wanted.iter().all(|want| match field {
        Value::Array(held) => held.iter().any(|item| values_equal(item, want)),
        _ => values_equal(field, want),
    })
}
