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
    Field {
        name: String,
        test: FieldTest,
    },
    /// A question about whether a field is there at all, rather than about the
    /// value it holds. See [`Presence`].
    Presence {
        name: String,
        want: Presence,
    },
}

/// What `exists`, `is_missing` and `is_null` ask of one field.
///
/// # Why these are their own node rather than operators
///
/// Every other operator is judged against a value, and `field_matches` answers
/// the absent case before any of them runs: a record that does not carry the
/// field matches nothing, so `ne`, `nin` and `all` exclude it exactly as `eq`,
/// `in` and `any` do. That rule is what makes the language coherent, and it is
/// also what left absence with no way to say it. Until this existed the only
/// spelling was `{"$not": {"field": {"all": []}}}`, which nobody finds.
///
/// So these three are decided before the value is looked up rather than after,
/// which is a different kind of question and gets a different node.
///
/// # Each one is total, and each one has a complement
///
/// `exists: false` matches a record the field is absent from, and `is_null:
/// false` matches a record the field is absent from as well, because it is the
/// complement of `is_null: true` rather than a claim that a value is present
/// and not null. Anything else would repeat the trap `ne` fell into, where the
/// operator and its negation are not complements and a user has to know which
/// of the two rules they are under.
///
/// Write "present and not null" as `{"field": {"exists": true, "is_null":
/// false}}`, which is the conjunction it looks like.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Presence {
    /// The record carries the field, whatever value it holds.
    Present,
    /// The record does not carry the field.
    Absent,
    /// The record carries the field and the value is null.
    Null,
    /// Anything else, which is the complement of `Null` and therefore includes
    /// a record the field is absent from.
    NotNull,
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
            _ => compile_field(key, condition)?,
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

/// One field's condition, as one node or as a conjunction of several.
///
/// A mapping is always the operator form. Direct equality against a nested
/// object has no syntax of its own, because the two forms would be
/// indistinguishable, so it is written `{"eq": {...}}`.
///
/// The three presence operators are split out here rather than carried through
/// as operators, because they are answered before the value is looked up and
/// every other operator is answered after. A mapping holding both kinds
/// compiles to a conjunction of a `Presence` node per presence operator and one
/// `Field` node carrying the rest, which is what the mapping already meant.
fn compile_field(name: &str, condition: &Value) -> PyResult<Filter> {
    let Value::Object(operations) = condition else {
        return Ok(Filter::Field {
            name: name.to_string(),
            test: FieldTest::Equals(condition.clone()),
        });
    };

    let mut nodes: Vec<Filter> = Vec::new();
    let mut tests = Vec::with_capacity(operations.len());
    for (op, target) in operations {
        match presence_op(op) {
            Some(kinds) => nodes.push(Filter::Presence {
                name: name.to_string(),
                want: kinds[usize::from(!presence_target(name, op, target)?)],
            }),
            None => tests.push((parse_op(op)?, target.clone())),
        }
    }

    if !tests.is_empty() {
        nodes.push(Filter::Field {
            name: name.to_string(),
            test: FieldTest::Operators(tests),
        });
    }
    // An operator mapping is a conjunction of its entries, which is what the
    // single `Field` node meant when it held all of them.
    Ok(if nodes.len() == 1 {
        nodes.pop().expect("just checked the length")
    } else {
        Filter::All(nodes)
    })
}

/// The pair one presence operator selects between, true first
fn presence_op(op: &str) -> Option<[Presence; 2]> {
    match op {
        "exists" => Some([Presence::Present, Presence::Absent]),
        "is_missing" => Some([Presence::Absent, Presence::Present]),
        "is_null" => Some([Presence::Null, Presence::NotNull]),
        _ => None,
    }
}

/// The boolean a presence operator takes, and nothing else
///
/// A string or a number here is a filter that means nothing, and reading it as
/// truthy would silently pick one of the two answers. The three operators are
/// new, so refusing is free.
fn presence_target(name: &str, op: &str, target: &Value) -> PyResult<bool> {
    target.as_bool().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "\"{}\" takes true or false, and {{\"{}\": {{\"{}\": ...}}}} was given {}.",
            op,
            name,
            op,
            describe(target)
        ))
    })
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
        Filter::Presence { name, want } => want.holds(metadata.get(name)),
    }
}

impl Presence {
    /// Judge one field's presence, given whatever the record holds for it
    ///
    /// `None` is a record that does not carry the field. `Some(Value::Null)` is
    /// a record that carries it and holds a null, which is what a Python `None`
    /// in a metadata mapping becomes and what a null in `metadata.json` reads
    /// back as. The two are distinct in the storage format and this is what
    /// tells them apart.
    pub(crate) fn holds(self, found: Option<&Value>) -> bool {
        match self {
            Presence::Present => found.is_some(),
            Presence::Absent => found.is_none(),
            Presence::Null => matches!(found, Some(Value::Null)),
            Presence::NotNull => !matches!(found, Some(Value::Null)),
        }
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
            // Conservative in the safe direction, as the field leaf is.
            // `{"x": {"is_null": false}}` does hold for every record with no
            // `x`, and answering false here only means a delete that would have
            // been refused runs instead.
            Filter::Presence { .. } => false,
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
            Filter::Presence { .. } => false,
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
/// answered false for an absent field. Splitting
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

// ============================================================================
// THE NUMERIC ORDER
// ============================================================================

/// `compare_integer_to_float` against exact arithmetic, and the order it induces.
///
/// # What is under test
///
/// Every filter comparison between an integer and a float goes through
/// [`compare_integer_to_float`], and it had no test at all. It saturates an
/// `i128` out of an `f64` and breaks the tie on the fraction, which are two
/// places an ordering can be wrong without any caller noticing: a filter does
/// not report a comparison, it reports a page.
///
/// # The oracle
///
/// [`exact_order`] shares no line with the function it checks. It takes the
/// float apart into the sign, mantissa and exponent IEEE-754 actually stores,
/// so the value it works with is `mantissa * 2^exponent` exactly, and it
/// compares that against the integer by integer arithmetic alone. No cast from
/// a float to an integer happens anywhere in it, which is the operation the
/// function under test depends on.
///
/// # The domain
///
/// [`numeric_value`] is the only producer of `NumericValue::Integer`, and it
/// widens an `i64` or a `u64`, so the integer reaching the function is always
/// within `i64::MIN ..= u64::MAX`. That is what makes the saturating cast
/// sound: a float too large for an `i128` saturates to `i128::MAX`, which is
/// still above every integer that can arrive, so the comparison lands on the
/// correct side. [`the_integer_domain_is_i64_through_u64`] holds that
/// precondition, because widening `numeric_value` would silently invalidate the
/// saturation rather than fail anywhere near it.
///
/// # The budget and the deeper run
///
/// The randomised property runs [`CASES`] pairs and costs milliseconds.
/// `ZEUSDB_ORDER_CASES` raises it for a soak by hand and is absent in CI, which
/// is what `ZEUSDB_FUZZ_CASES` does for the dump fuzzer.
#[cfg(test)]
mod order_tests {
    use super::*;
    use serde_json::json;
    use serde_json::Number;

    /// Randomised pairs the committed test draws.
    ///
    /// A budget rather than a target. Measured on the machine it was written
    /// on, five million pairs cost 0.25 s in a debug build, so this is under
    /// a tenth of a second against a `cargo test` that already takes minutes.
    /// The deep runs are the soak: `ZEUSDB_ORDER_CASES=200000000` is ten
    /// seconds and found nothing.
    const CASES: usize = 2_000_000;

    /// splitmix64, matching `graph::fuzz`. The crate's seeded generator is
    /// `rand_chacha` and is pinned because every product draw runs on it; this
    /// draws test inputs, so coupling the two would tie together two things
    /// that have no reason to move together.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    // ------------------------------------------------------------------
    // THE ORACLE
    // ------------------------------------------------------------------

    /// A finite `f64` as `(negative, mantissa, exponent)`, exactly.
    ///
    /// The value is `mantissa * 2^exponent`, negated when `negative`. A
    /// subnormal and a zero share the biased exponent of zero and carry no
    /// implicit leading bit, which is the only case that needs separating.
    fn decompose(value: f64) -> (bool, u64, i32) {
        let bits = value.to_bits();
        let negative = bits >> 63 == 1;
        let biased = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        if biased == 0 {
            (negative, fraction, -1074)
        } else {
            (negative, fraction | (1u64 << 52), biased - 1075)
        }
    }

    /// `magnitude` against `mantissa * 2^exponent`, both non-negative, exactly.
    ///
    /// Neither side is ever rounded. Where the float's magnitude cannot fit in
    /// a `u128` it is above every `u128` and the answer is settled without
    /// computing it, and the same in the other direction for the integer's.
    fn compare_magnitude(magnitude: u128, mantissa: u64, exponent: i32) -> CmpOrdering {
        let mantissa = mantissa as u128;
        if exponent >= 0 {
            let width = 128 - mantissa.leading_zeros();
            if width + exponent as u32 > 128 {
                return CmpOrdering::Less;
            }
            magnitude.cmp(&(mantissa << exponent))
        } else {
            // The value is mantissa / 2^shift. Multiplying both sides by
            // 2^shift keeps the ordering and leaves two integers.
            let shift = exponent.unsigned_abs();
            let width = 128 - magnitude.leading_zeros();
            if width + shift > 128 {
                return CmpOrdering::Greater;
            }
            (magnitude << shift).cmp(&mantissa)
        }
    }

    /// The ordering the function under test is supposed to produce.
    fn exact_order(integer: i128, float: f64) -> Option<CmpOrdering> {
        if float.is_nan() {
            return None;
        }
        if float == f64::INFINITY {
            return Some(CmpOrdering::Less);
        }
        if float == f64::NEG_INFINITY {
            return Some(CmpOrdering::Greater);
        }
        let (negative, mantissa, exponent) = decompose(float);
        if mantissa == 0 {
            // Both zeros are zero, so the sign of the float does not enter.
            return Some(integer.cmp(&0));
        }
        if integer <= 0 && !negative {
            return Some(CmpOrdering::Less);
        }
        if integer >= 0 && negative {
            return Some(CmpOrdering::Greater);
        }
        let ordering = compare_magnitude(integer.unsigned_abs(), mantissa, exponent);
        Some(if negative {
            ordering.reverse()
        } else {
            ordering
        })
    }

    /// Every integer the filter can hold, which is what bounds the domain.
    fn domain_integers() -> Vec<i128> {
        let mut values = vec![
            0,
            1,
            -1,
            2,
            -2,
            i64::MIN as i128,
            i64::MIN as i128 + 1,
            i64::MAX as i128,
            i64::MAX as i128 - 1,
            u64::MAX as i128,
            u64::MAX as i128 - 1,
            i64::MAX as i128 + 1,
        ];
        // Both sides of every power of two an f64 stops being able to name
        // consecutive integers at, and of the two integer widths.
        for exponent in [52u32, 53, 54, 62, 63, 64] {
            let base = 1i128 << exponent;
            for offset in -3i128..=3 {
                values.push(base + offset);
                if base + offset <= -(i64::MIN as i128) {
                    values.push(-(base + offset));
                }
            }
        }
        values.retain(|v| *v >= i64::MIN as i128 && *v <= u64::MAX as i128);
        values.sort_unstable();
        values.dedup();
        values
    }

    /// Floats worth naming, being the ones a cast, a truncation or a sign test
    /// can go wrong on.
    fn interesting_floats() -> Vec<f64> {
        let mut values = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            SMALLEST_SUBNORMAL,
            -SMALLEST_SUBNORMAL,
            LARGEST_SUBNORMAL,
            f64::EPSILON,
            1e300,
            -1e300,
            // Beyond i128, so the cast in the function under test saturates.
            1e39,
            -1e39,
            170_141_183_460_469_231_731_687_303_715_884_105_728.0, // 2^127
            -170_141_183_460_469_231_731_687_303_715_884_105_728.0,
        ];
        for exponent in [52u32, 53, 54, 62, 63, 64, 65] {
            let base = (1u128 << exponent) as f64;
            for offset in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
                values.push(base + offset);
                values.push(-(base + offset));
            }
        }
        for whole in -4i32..=4 {
            for fraction in [0.0, 0.25, 0.5, 0.75, 0.999_999_999] {
                values.push(whole as f64 + fraction);
                values.push(whole as f64 - fraction);
            }
        }
        values
    }

    fn check(integer: i128, float: f64) {
        let got = compare_integer_to_float(integer, float);
        let want = exact_order(integer, float);
        assert_eq!(
            got, want,
            "compare_integer_to_float({}, {:?}) gave {:?}, exact arithmetic gives {:?}",
            integer, float, got, want
        );
    }

    // ------------------------------------------------------------------
    // AGREEMENT WITH EXACT ARITHMETIC
    // ------------------------------------------------------------------

    #[test]
    fn it_agrees_with_exact_arithmetic_on_every_named_pair() {
        let integers = domain_integers();
        let floats = interesting_floats();
        for integer in &integers {
            for float in &floats {
                check(*integer, *float);
            }
        }
    }

    /// The smallest positive `f64`, which is a subnormal with one bit set.
    const SMALLEST_SUBNORMAL: f64 = f64::from_bits(1);

    /// The largest subnormal, one unit below `f64::MIN_POSITIVE`.
    const LARGEST_SUBNORMAL: f64 = f64::from_bits((1u64 << 52) - 1);

    /// Floats a uniform bit pattern almost never produces.
    ///
    /// A random 64-bit pattern is NaN about one time in a thousand and lands
    /// on a power of two never, so the shapes the function actually branches
    /// on have to be drawn deliberately.
    /// [`the_random_pairs_reach_every_shape_the_function_branches_on`] is what
    /// holds that, and it is what this table exists to satisfy.
    const SPECIAL_FLOATS: [f64; 23] = [
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        // Subnormals, which carry no implicit leading bit and so are the one
        // case the oracle's decomposition separates.
        SMALLEST_SUBNORMAL,
        -SMALLEST_SUBNORMAL,
        LARGEST_SUBNORMAL,
        f64::MAX,
        f64::MIN,
        9_007_199_254_740_992.0, // 2^53
        -9_007_199_254_740_992.0,
        9_223_372_036_854_775_808.0,  // 2^63
        18_446_744_073_709_551_616.0, // 2^64
        1e39,                         // beyond i128, so the cast saturates
        -1e39,
    ];

    /// The pairs both randomised tests draw, seeded and reproducible from here.
    ///
    /// The integer shape and the float shape are drawn independently, so the
    /// cross product of the two is covered rather than four fixed pairings.
    /// The small values a filter really carries are drawn as often as the
    /// extremes; a raw bit pattern reaches subnormals and the whole exponent
    /// range; the shapes that land near an integer are what reach the tie
    /// break; and [`SPECIAL_FLOATS`] supplies what none of those produce often
    /// enough to test.
    fn random_pairs(count: usize) -> impl Iterator<Item = (i128, f64)> {
        let mut rng = Rng(0x0ade_0000_0001_face);
        (0..count).map(move |_| {
            let raw = rng.next();
            let integer = match raw % 5 {
                0 => (raw % 2_001) as i128 - 1_000,
                1 => raw as i64 as i128,
                2 => raw as i128,
                3 => (1i128 << 53) + (raw % 17) as i128 - 8,
                _ => (raw % (1u64 << 55)) as i128 - (1i128 << 54),
            };
            let bits = rng.next();
            let float = match bits % 8 {
                0 | 1 => f64::from_bits(bits),
                2 => SPECIAL_FLOATS[(bits >> 8) as usize % SPECIAL_FLOATS.len()],
                3 => (bits % 2_001) as f64 - 1_000.0 + ((bits >> 40) % 4) as f64 / 4.0,
                4 | 5 => integer as f64 + ((bits % 9) as f64 - 4.0) / 4.0,
                6 => (integer + (bits % 3) as i128 - 1) as f64,
                _ => (bits as i64 as f64) / 2.0_f64.powi((bits % 80) as i32 - 40),
            };
            (integer, float)
        })
    }

    #[test]
    fn it_agrees_with_exact_arithmetic_on_random_pairs() {
        let budget: usize = std::env::var("ZEUSDB_ORDER_CASES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(CASES);
        for (integer, float) in random_pairs(budget) {
            check(integer, float);
        }
    }

    #[test]
    fn the_random_pairs_reach_every_shape_the_function_branches_on() {
        // The fuzzer in `graph::fuzz` measures that its mutations carry past
        // the checksums rather than assuming it, and this is the same idea. A
        // generator that stopped producing NaN, or stopped landing on the tie
        // break, would leave `it_agrees_with_exact_arithmetic_on_random_pairs`
        // passing while testing nothing, and no assertion in it would notice.
        let mut nan = 0usize;
        let mut infinite = 0usize;
        let mut subnormal = 0usize;
        let mut equal = 0usize;
        let mut fractional_tie = 0usize;
        let mut above_two_to_the_fifty_third = 0usize;
        let mut both_signs = [0usize; 2];
        for (integer, float) in random_pairs(200_000) {
            if float.is_nan() {
                nan += 1;
                continue;
            }
            if float.is_infinite() {
                infinite += 1;
                continue;
            }
            if float != 0.0 && float.abs() < f64::MIN_POSITIVE {
                subnormal += 1;
            }
            if float.abs() >= 9_007_199_254_740_992.0 {
                above_two_to_the_fifty_third += 1;
            }
            both_signs[usize::from(float.is_sign_negative())] += 1;
            match compare_integer_to_float(integer, float) {
                Some(CmpOrdering::Equal) => equal += 1,
                Some(_) => {
                    // The tie break decided it, rather than the integer parts.
                    if integer == float.trunc() as i128 {
                        fractional_tie += 1;
                    }
                }
                None => unreachable!("a finite float orders"),
            }
        }
        for (name, count, floor) in [
            ("NaN", nan, 100),
            ("infinite", infinite, 100),
            ("subnormal", subnormal, 100),
            ("exactly equal", equal, 100),
            ("decided by the fraction", fractional_tie, 100),
            ("above 2^53", above_two_to_the_fifty_third, 1_000),
            ("positive", both_signs[0], 10_000),
            ("negative", both_signs[1], 10_000),
        ] {
            assert!(
                count >= floor,
                "the generator produced {} {} pairs in 200,000, below the floor \
                 of {}, so the property test is no longer reaching that branch",
                count,
                name,
                floor
            );
        }
    }

    #[test]
    fn it_agrees_across_every_consecutive_integer_around_two_to_the_fifty_third() {
        // Above 2^53 an f64 cannot name consecutive integers, so 2^53 + 1 is
        // stored as 2^53 and two distinct integers share one float. The
        // comparison has to stay exact through that, which it can only do by
        // never casting the integer.
        let start = (1i128 << 53) - 8;
        for integer in start..=(1i128 << 53) + 8 {
            for step in -8i128..=8 {
                let float = ((1i128 << 53) + step) as f64;
                check(integer, float);
                check(integer, float + 0.5);
                check(integer, float - 0.5);
            }
        }
        // The pair that motivates it. 2^53 + 1 is not representable, so the
        // literal rounds down to 2^53 and the integer is strictly greater.
        let float: f64 = 9_007_199_254_740_993.0;
        assert_eq!(float, 9_007_199_254_740_992.0, "the literal did not round");
        assert_eq!(
            compare_integer_to_float(9_007_199_254_740_993, float),
            Some(CmpOrdering::Greater)
        );
        assert_eq!(
            compare_integer_to_float(9_007_199_254_740_992, float),
            Some(CmpOrdering::Equal)
        );
    }

    #[test]
    fn the_widest_integers_compare_against_floats_beyond_them() {
        for integer in [
            i64::MIN as i128,
            i64::MAX as i128,
            u64::MAX as i128,
            0,
            -1,
            1,
        ] {
            for float in [
                1e39,
                -1e39,
                f64::MAX,
                f64::MIN,
                1.8446744073709552e19, // u64::MAX rounded up, so above it
                -9.223372036854776e18, // i64::MIN exactly
                9.223372036854776e18,  // i64::MAX rounded up, so above it
            ] {
                check(integer, float);
            }
        }
        // i64::MIN is a power of two and converts exactly, so this is equality
        // rather than a near miss.
        assert_eq!(
            compare_integer_to_float(i64::MIN as i128, -9_223_372_036_854_775_808.0),
            Some(CmpOrdering::Equal)
        );
        // i64::MAX is not, so the nearest f64 is one above it.
        assert_eq!(
            compare_integer_to_float(i64::MAX as i128, 9_223_372_036_854_775_807.0),
            Some(CmpOrdering::Less)
        );
        // u64::MAX likewise.
        assert_eq!(
            compare_integer_to_float(u64::MAX as i128, 18_446_744_073_709_551_615.0),
            Some(CmpOrdering::Less)
        );
    }

    // ------------------------------------------------------------------
    // THE VALUES THAT ARE NOT ORDINARY NUMBERS
    // ------------------------------------------------------------------

    #[test]
    fn a_nan_orders_against_nothing() {
        for integer in domain_integers() {
            assert_eq!(compare_integer_to_float(integer, f64::NAN), None);
            // The negative quiet NaN and a signalling bit pattern too, since
            // `is_nan` is the only thing separating them from a real value.
            assert_eq!(compare_integer_to_float(integer, -f64::NAN), None);
            assert_eq!(
                compare_integer_to_float(integer, f64::from_bits(0x7ff0_0000_0000_0001)),
                None
            );
        }
    }

    #[test]
    fn both_infinities_order_every_integer() {
        for integer in domain_integers() {
            assert_eq!(
                compare_integer_to_float(integer, f64::INFINITY),
                Some(CmpOrdering::Less),
                "{} against +inf",
                integer
            );
            assert_eq!(
                compare_integer_to_float(integer, f64::NEG_INFINITY),
                Some(CmpOrdering::Greater),
                "{} against -inf",
                integer
            );
        }
    }

    #[test]
    fn negative_zero_equals_zero_and_orders_like_it() {
        assert_eq!(
            compare_integer_to_float(0, -0.0),
            Some(CmpOrdering::Equal),
            "IEEE holds -0.0 equal to 0.0, so an integer zero equals both"
        );
        assert_eq!(compare_integer_to_float(0, 0.0), Some(CmpOrdering::Equal));
        assert_eq!(
            compare_integer_to_float(1, -0.0),
            Some(CmpOrdering::Greater)
        );
        assert_eq!(compare_integer_to_float(-1, -0.0), Some(CmpOrdering::Less));
        // A fraction that truncates to negative zero is still below zero, which
        // is the case a `trunc` alone would get wrong.
        assert_eq!(
            compare_integer_to_float(0, -0.5),
            Some(CmpOrdering::Greater)
        );
        assert_eq!(compare_integer_to_float(0, 0.5), Some(CmpOrdering::Less));
    }

    // ------------------------------------------------------------------
    // THE ORDER'S OWN PROPERTIES
    // ------------------------------------------------------------------

    #[test]
    fn the_order_is_antisymmetric() {
        // Directly, on the function, and through `compare_numbers`, which is
        // where a filter reaches it and which reverses one of the two arms.
        let mut rng = Rng(0xa571_5eed_ce77_c000);
        for integer in domain_integers() {
            for float in interesting_floats() {
                let forward = compare_integer_to_float(integer, float);
                let backward = compare_integer_to_float(integer, float).map(|o| o.reverse());
                assert_eq!(forward.map(|o| o.reverse()), backward);
                let Some(left) = number_from_integer(integer) else {
                    continue;
                };
                let Some(right) = Number::from_f64(float) else {
                    continue;
                };
                assert_eq!(
                    compare_numbers(&left, &right),
                    compare_numbers(&right, &left).map(|o| o.reverse()),
                    "{} against {}",
                    left,
                    right
                );
            }
        }
        for _ in 0..20_000 {
            let integer = rng.next() as i64 as i128;
            let float = f64::from_bits(rng.next());
            let forward = compare_integer_to_float(integer, float);
            let reversed = exact_order(integer, float).map(|o| o.reverse());
            assert_eq!(forward.map(|o| o.reverse()), reversed);
        }
    }

    #[test]
    fn the_order_is_transitive_over_a_mixed_pool() {
        // Integers and floats together, because transitivity across the two
        // variants is what a filter's ordering rests on and what a lossy cast
        // in one arm would break.
        let mut pool: Vec<Number> = Vec::new();
        for integer in [
            -3i128,
            -1,
            0,
            1,
            2,
            3,
            (1i128 << 53) - 1,
            1i128 << 53,
            (1i128 << 53) + 1,
            i64::MAX as i128,
            u64::MAX as i128,
            i64::MIN as i128,
        ] {
            if let Some(number) = number_from_integer(integer) {
                pool.push(number);
            }
        }
        for float in [
            -3.5f64,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            2.5,
            3.0,
            9_007_199_254_740_992.0,
            9_223_372_036_854_775_808.0,
            18_446_744_073_709_551_616.0,
            1e300,
            -1e300,
        ] {
            if let Some(number) = Number::from_f64(float) {
                pool.push(number);
            }
        }
        for a in &pool {
            for b in &pool {
                for c in &pool {
                    let (Some(ab), Some(bc), Some(ac)) = (
                        compare_numbers(a, b),
                        compare_numbers(b, c),
                        compare_numbers(a, c),
                    ) else {
                        continue;
                    };
                    if ab != CmpOrdering::Greater && bc != CmpOrdering::Greater {
                        assert_ne!(
                            ac,
                            CmpOrdering::Greater,
                            "{} <= {} and {} <= {} but {} > {}",
                            a,
                            b,
                            b,
                            c,
                            a,
                            c
                        );
                    }
                    if ab == CmpOrdering::Equal && bc == CmpOrdering::Equal {
                        assert_eq!(ac, CmpOrdering::Equal, "{} = {} = {}", a, b, c);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // WHAT A FILTER SEES
    // ------------------------------------------------------------------

    #[test]
    fn the_integer_domain_is_i64_through_u64() {
        // The precondition the saturating cast rests on. A float above
        // `i128::MAX` saturates to `i128::MAX`, which is still above every
        // integer `numeric_value` can produce, so the comparison lands on the
        // correct side. Widening this producer would invalidate that silently,
        // so it is held here rather than reasoned about in a comment.
        for candidate in [
            json!(0),
            json!(-1),
            json!(i64::MIN),
            json!(i64::MAX),
            json!(u64::MAX),
            json!(1.5),
            json!(1e308),
        ] {
            let Value::Number(number) = candidate else {
                unreachable!()
            };
            match numeric_value(&number) {
                Some(NumericValue::Integer(value)) => assert!(
                    (i64::MIN as i128..=u64::MAX as i128).contains(&value),
                    "numeric_value produced {}, which is outside the range the \
                     saturating cast in compare_integer_to_float assumes",
                    value
                ),
                Some(NumericValue::Float(_)) | None => {}
            }
        }
    }

    #[test]
    fn the_six_comparison_operators_agree_with_the_order() {
        // `field_test_matches` is the entry every filter uses, and each operator
        // is a predicate over the same ordering. A pairing the order refuses
        // makes every one of them false, including `gte` and `lte`, which is
        // what stops a NaN matching by default.
        let integers = [
            json!(0),
            json!(3),
            json!(-3),
            json!(i64::MAX),
            json!(u64::MAX),
        ];
        let floats = [
            json!(0.0),
            json!(2.5),
            json!(3.0),
            json!(-3.5),
            json!(1e300),
        ];
        for left in integers.iter().chain(floats.iter()) {
            for right in integers.iter().chain(floats.iter()) {
                let (Value::Number(a), Value::Number(b)) = (left, right) else {
                    unreachable!()
                };
                let order = compare_numbers(a, b);
                let test = |op: Op| {
                    field_test_matches(left, &FieldTest::Operators(vec![(op, right.clone())]))
                };
                assert_eq!(test(Op::Gt), order == Some(CmpOrdering::Greater));
                assert_eq!(test(Op::Lt), order == Some(CmpOrdering::Less));
                assert_eq!(test(Op::Eq), order == Some(CmpOrdering::Equal));
                assert_eq!(test(Op::Ne), order != Some(CmpOrdering::Equal));
                assert_eq!(
                    test(Op::Gte),
                    matches!(order, Some(CmpOrdering::Greater) | Some(CmpOrdering::Equal))
                );
                assert_eq!(
                    test(Op::Lte),
                    matches!(order, Some(CmpOrdering::Less) | Some(CmpOrdering::Equal))
                );
            }
        }
    }

    #[test]
    fn a_filter_orders_an_integer_against_a_float_the_way_exact_arithmetic_does() {
        // The whole path, from a compiled filter down to the comparison, on the
        // values a cast is most likely to get wrong.
        let cases: [(i64, f64, bool); 6] = [
            (9_007_199_254_740_993, 9_007_199_254_740_992.0, true),
            (9_007_199_254_740_992, 9_007_199_254_740_992.0, false),
            (i64::MAX, 9_223_372_036_854_775_807.0, false),
            (0, -0.5, true),
            (-1, -0.5, false),
            (3, 2.999_999_999, true),
        ];
        for (stored, target, expect_greater) in cases {
            let mut metadata = HashMap::new();
            metadata.insert("n".to_string(), json!(stored));
            let filter = compile_filter(&HashMap::from([("n".to_string(), json!({"gt": target}))]))
                .expect("the filter compiles");
            assert_eq!(
                matches_filter(&metadata, &filter),
                expect_greater,
                "{} gt {}",
                stored,
                target
            );
        }
    }

    /// A `serde_json::Number` from an `i128` in the filter's integer domain.
    fn number_from_integer(value: i128) -> Option<Number> {
        if let Ok(signed) = i64::try_from(value) {
            Some(Number::from(signed))
        } else {
            u64::try_from(value).ok().map(Number::from)
        }
    }
}
