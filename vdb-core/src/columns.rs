//! A column per declared filterable field, and the bitmap a filter selects
//! into.
//!
//! # What this replaces
//!
//! A filtered search used to answer "which records match" by walking
//! `vector_metadata`, which is a `HashMap<String, HashMap<String, Value>>`.
//! That walk costs about 250 nanoseconds a record, so a filter matching ten
//! records over 100,000 cost 26 to 39 milliseconds where an unfiltered search
//! cost 0.3 to 1.2. Relay 90 measured the walk at 73 to 100 percent of a
//! filtered search and measured a column per field walking 224 times faster.
//!
//! A column here is addressed by internal id, so the same bitmap answers both
//! paths a filtered search can take. The exact scan reads the set bits and
//! scores those records. The graph traversal tests one bit per node it reaches,
//! in place of the node, `rev_map`, `vector_metadata`, field lookup chain that
//! relay 90 measured at 154 nanoseconds a probe.
//!
//! # Every operator is served, because the leaf is the same function
//!
//! Nothing here reimplements an operator. A leaf calls
//! [`crate::filter::field_test_matches`], which is the function
//! `crate::filter::field_matches` calls once it has found the field, so a
//! column and the walk cannot disagree about what a value matches. What the
//! column changes is how often that function runs and what it costs to reach
//! the value, not what it answers.
//!
//! Boolean composition is bitmap algebra. `$and` intersects, `$or` unions and
//! `$not` complements within the live set, which is why the store carries
//! [`ColumnStore::live`]. An empty `$and` is the whole live set and an empty
//! `$or` is nothing, which is what `matches_filter` answers for the same two
//! shapes.
//!
//! # What a column cannot serve
//!
//! A field that was not declared at `create()`. There is no column to read, so
//! [`ColumnStore::select`] returns the field's name and the caller walks the
//! metadata store exactly as it did before. That is the only case, and it is a
//! property of the declaration rather than of the filter language.

use crate::filter::{field_test_matches, FieldTest, Filter};
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

/// Declared fields one index may carry.
///
/// Each one costs four bytes a record in the common case and a full pass of its
/// codes per filter that names it, so the bound is about keeping a declaration
/// honest rather than about any limit in the structure. A filter naming more
/// than thirty two fields is past anything written by hand or generated from a
/// query.
pub(crate) const MAX_INDEXED_FIELDS: usize = 32;

/// A dictionary below this many entries is never traded for a plain column,
/// whatever its cardinality, because at that size the codes are the whole cost
/// and the dictionary itself rounds to nothing.
const DICTIONARY_FLOOR: usize = 4_096;

/// The code a slot carries when it holds no value for this field, either
/// because the record does not carry the field or because no record occupies
/// the slot.
const ABSENT: u32 = u32::MAX;

/// The three keys that name a group rather than a field, repeated here so that
/// a declaration naming one is refused at `create()` rather than building a
/// column no filter can ever reach.
const GROUP_KEYS: [&str; 3] = ["$and", "$or", "$not"];

// ============================================================================
// THE BITMAP
// ============================================================================

/// A set of internal ids, one bit each.
///
/// Sized to the store's slot count rather than to the record count, because a
/// slot is an internal id and internal ids are never reused. At 100,000 records
/// that is 12.5 kilobytes, so a filter of a dozen leaves allocates less than
/// the page it returns.
#[derive(Clone)]
pub(crate) struct Bitmap {
    words: Vec<u64>,
}

impl Bitmap {
    fn zeros(slots: usize) -> Self {
        Bitmap {
            words: vec![0; slots.div_ceil(64)],
        }
    }

    #[inline]
    fn set(&mut self, slot: usize) {
        self.words[slot >> 6] |= 1u64 << (slot & 63);
    }

    /// Whether an internal id is in the set.
    ///
    /// Total over every `usize`, so the traversal predicate can ask it about a
    /// node the store has never heard of and get `false` rather than a panic.
    #[inline]
    pub(crate) fn contains(&self, slot: usize) -> bool {
        self.words
            .get(slot >> 6)
            .is_some_and(|word| word >> (slot & 63) & 1 == 1)
    }

    pub(crate) fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    fn intersect(&mut self, other: &Bitmap) {
        for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
            *mine &= *theirs;
        }
    }

    fn union(&mut self, other: &Bitmap) {
        for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
            *mine |= *theirs;
        }
    }

    /// `live` without `self`.
    ///
    /// The complement is taken within the live set rather than over the whole
    /// word range, so a `$not` never selects a slot that holds no record. Every
    /// other bitmap here is already a subset of the live set, because a slot
    /// with no record carries [`ABSENT`] in every column and no leaf matches
    /// that.
    fn complement_within(&self, live: &Bitmap) -> Bitmap {
        let mut out = Bitmap {
            words: vec![0; self.words.len()],
        };
        for (index, word) in out.words.iter_mut().enumerate() {
            *word = live.words.get(index).copied().unwrap_or(0) & !self.words[index];
        }
        out
    }

    /// Every internal id in the set, in increasing order.
    pub(crate) fn for_each<F: FnMut(usize)>(&self, mut visit: F) {
        for (index, mut word) in self.words.iter().copied().enumerate() {
            while word != 0 {
                visit(index * 64 + word.trailing_zeros() as usize);
                word &= word - 1;
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        self.words.capacity() * 8
    }
}

// ============================================================================
// ONE COLUMN
// ============================================================================

/// One declared field's values, addressed by internal id.
///
/// Two representations, chosen by cardinality and never by declaration. A
/// column starts as a dictionary and is traded for a plain one the first time
/// it holds [`DICTIONARY_FLOOR`] distinct values across no more than twice that
/// many slots, which is the point where the dictionary stops saving anything
/// and starts costing.
enum Column {
    /// Low cardinality. Four bytes a record, plus one copy of each distinct
    /// value however many records carry it.
    ///
    /// A filter runs its predicate once per distinct value and then walks the
    /// codes, so a `contains` over a field holding five distinct strings does
    /// five substring searches rather than one per record.
    Dictionary {
        codes: Vec<u32>,
        dict: Vec<Value>,
        /// Live records per dictionary entry. Zero means the entry is free and
        /// its code is on `free`. Without this a field whose values change on
        /// every `update_metadata` would grow its dictionary without bound.
        refs: Vec<u32>,
        free: Vec<u32>,
        lookup: HashMap<Value, u32>,
    },
    /// High cardinality. One value a record, which is a second copy of that
    /// value, the first being the one `vector_metadata` holds.
    ///
    /// Measured smaller and faster than a dictionary once nearly every record
    /// holds a distinct value, because there the dictionary holds one entry per
    /// record as well as one code per record and its lookup table holds a third.
    Plain { values: Vec<Option<Value>> },
}

impl Column {
    fn new(expected_size: usize) -> Self {
        Column::Dictionary {
            codes: Vec::with_capacity(expected_size.saturating_add(1)),
            dict: Vec::new(),
            refs: Vec::new(),
            free: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Put one value, or its absence, at one slot.
    fn write(&mut self, slot: usize, value: Option<&Value>) {
        match self {
            Column::Plain { values } => {
                if values.len() <= slot {
                    values.resize(slot + 1, None);
                }
                values[slot] = value.cloned();
                return;
            }
            Column::Dictionary {
                codes,
                dict,
                refs,
                free,
                lookup,
            } => {
                if codes.len() <= slot {
                    codes.resize(slot + 1, ABSENT);
                }
                let previous = codes[slot];
                codes[slot] = match value {
                    None => ABSENT,
                    Some(value) => match lookup.get(value) {
                        Some(&code) => {
                            refs[code as usize] += 1;
                            code
                        }
                        None => {
                            let code = match free.pop() {
                                Some(code) => {
                                    dict[code as usize] = value.clone();
                                    refs[code as usize] = 1;
                                    code
                                }
                                None => {
                                    dict.push(value.clone());
                                    refs.push(1);
                                    (dict.len() - 1) as u32
                                }
                            };
                            lookup.insert(value.clone(), code);
                            code
                        }
                    },
                };
                if previous != ABSENT {
                    refs[previous as usize] -= 1;
                    if refs[previous as usize] == 0 {
                        let released = std::mem::replace(&mut dict[previous as usize], Value::Null);
                        lookup.remove(&released);
                        free.push(previous);
                    }
                }
            }
        }

        if self.dictionary_has_stopped_paying() {
            self.trade_for_plain();
        }
    }

    fn dictionary_has_stopped_paying(&self) -> bool {
        match self {
            Column::Plain { .. } => false,
            Column::Dictionary { codes, dict, .. } => {
                dict.len() >= DICTIONARY_FLOOR && dict.len() * 2 >= codes.len()
            }
        }
    }

    fn trade_for_plain(&mut self) {
        let Column::Dictionary { codes, dict, .. } = self else {
            return;
        };
        let mut values = Vec::with_capacity(codes.len());
        for &code in codes.iter() {
            values.push(if code == ABSENT {
                None
            } else {
                Some(dict[code as usize].clone())
            });
        }
        *self = Column::Plain { values };
    }

    /// What this slot holds, for the debug assertion that ties a column to a
    /// record.
    fn read(&self, slot: usize) -> Option<&Value> {
        match self {
            Column::Plain { values } => values.get(slot).and_then(Option::as_ref),
            Column::Dictionary { codes, dict, .. } => match codes.get(slot).copied() {
                Some(code) if code != ABSENT => dict.get(code as usize),
                _ => None,
            },
        }
    }

    /// One leaf of a filter, as the set of internal ids whose value matches.
    fn select(&self, test: &FieldTest, slots: usize) -> Bitmap {
        let mut out = Bitmap::zeros(slots);
        match self {
            Column::Plain { values } => {
                for (slot, value) in values.iter().enumerate().take(slots) {
                    if value
                        .as_ref()
                        .is_some_and(|value| field_test_matches(value, test))
                    {
                        out.set(slot);
                    }
                }
            }
            Column::Dictionary {
                codes, dict, refs, ..
            } => {
                // The predicate runs once per distinct value rather than once
                // per record, which is the whole of why an operator as
                // expensive as `contains` costs the same here as `eq`.
                let mut matching = vec![false; dict.len()];
                let mut any = false;
                for (code, value) in dict.iter().enumerate() {
                    if refs[code] > 0 && field_test_matches(value, test) {
                        matching[code] = true;
                        any = true;
                    }
                }
                if !any {
                    return out;
                }
                for (slot, &code) in codes.iter().enumerate().take(slots) {
                    if code != ABSENT && matching[code as usize] {
                        out.set(slot);
                    }
                }
            }
        }
        out
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Column::Plain { values } => {
                values.capacity() * std::mem::size_of::<Option<Value>>()
                    + values.iter().flatten().map(value_payload).sum::<usize>()
            }
            Column::Dictionary {
                codes,
                dict,
                refs,
                free,
                lookup,
            } => {
                codes.capacity() * 4
                    + dict.capacity() * std::mem::size_of::<Value>()
                    + refs.capacity() * 4
                    + free.capacity() * 4
                    // A bucket holds the key, the code and a control byte, and
                    // the key owns the same payload the dictionary entry does.
                    + lookup.capacity() * (std::mem::size_of::<Value>() + 4 + 8)
                    + dict.iter().map(value_payload).sum::<usize>() * 2
            }
        }
    }
}

/// The first field a filter tree names, in evaluation order.
///
/// Used only to say which declaration is missing. A tree that names no field at
/// all is `{}` or a nest of empty groups, and there is nothing to report about
/// those.
fn first_field_name(filter: &Filter) -> Option<&str> {
    match filter {
        Filter::Field { name, .. } => Some(name.as_str()),
        Filter::All(branches) | Filter::Any(branches) => branches.iter().find_map(first_field_name),
        Filter::Not(inner) => first_field_name(inner),
    }
}

/// What a value owns on the heap beyond the 32 bytes of the `Value` itself.
fn value_payload(value: &Value) -> usize {
    match value {
        Value::String(text) => text.capacity(),
        Value::Array(items) => {
            items.capacity() * std::mem::size_of::<Value>()
                + items.iter().map(value_payload).sum::<usize>()
        }
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| key.capacity() + 48 + value_payload(value))
            .sum(),
        _ => 0,
    }
}

// ============================================================================
// THE STORE
// ============================================================================

/// Every declared field's column, and which slots hold a record.
///
/// A store with no declared fields is what an index created without
/// `indexed_fields` carries and what every directory saved before this existed
/// loads as. It answers [`ColumnStore::select`] with the first field the filter
/// names, so every caller falls back to the metadata walk and the index behaves
/// exactly as it did.
pub(crate) struct ColumnStore {
    names: Vec<String>,
    columns: Vec<Column>,
    index_of: HashMap<String, usize>,
    /// One bit per internal id, set while a record occupies the slot.
    ///
    /// It exists for `$not`, which is the only shape that has to name the
    /// records a filter does not select. Every other bitmap is a subset of this
    /// one by construction.
    live: Bitmap,
    /// Highest occupied internal id plus one, which is how far any walk goes.
    slots: usize,
    records: usize,
}

impl ColumnStore {
    /// A store for the declared fields, with each column reserved for the
    /// declared size.
    pub(crate) fn new(names: Vec<String>, expected_size: usize) -> Self {
        let columns = names.iter().map(|_| Column::new(expected_size)).collect();
        let index_of = names
            .iter()
            .enumerate()
            .map(|(position, name)| (name.clone(), position))
            .collect();
        ColumnStore {
            names,
            columns,
            index_of,
            live: Bitmap::zeros(expected_size.saturating_add(1)),
            slots: 0,
            records: 0,
        }
    }

    pub(crate) fn declared(&self) -> &[String] {
        &self.names
    }

    pub(crate) fn is_declared(&self) -> bool {
        !self.names.is_empty()
    }

    /// How many records the store holds. The tests below are the only readers;
    /// every other caller asks [`ColumnStore::tracks`] instead, which answers
    /// the question the invariant actually poses.
    #[cfg(test)]
    pub(crate) fn record_count(&self) -> usize {
        self.records
    }

    /// Record one insertion, or replace one record's values in place.
    ///
    /// Called by every path that writes a record's metadata, which is the three
    /// insertion paths, `update_metadata` and the loader. A field the record
    /// does not carry is written as absent rather than left alone, so an
    /// `update_metadata` that drops a key drops it from the column too.
    pub(crate) fn write(&mut self, slot: usize, metadata: &HashMap<String, Value>) {
        if self.names.is_empty() {
            return;
        }
        self.reserve(slot);
        for (position, name) in self.names.iter().enumerate() {
            self.columns[position].write(slot, metadata.get(name));
        }
        if !self.live.contains(slot) {
            self.live.set(slot);
            self.records += 1;
        }
    }

    /// Forget the record at one internal id.
    pub(crate) fn erase(&mut self, slot: usize) {
        if self.names.is_empty() || !self.live.contains(slot) {
            return;
        }
        for column in self.columns.iter_mut() {
            column.write(slot, None);
        }
        self.live.words[slot >> 6] &= !(1u64 << (slot & 63));
        self.records -= 1;
    }

    /// Drop every record, keeping the declaration.
    pub(crate) fn clear(&mut self, expected_size: usize) {
        let names = std::mem::take(&mut self.names);
        *self = ColumnStore::new(names, expected_size);
    }

    fn reserve(&mut self, slot: usize) {
        if slot >= self.slots {
            self.slots = slot + 1;
        }
        let needed = self.slots.div_ceil(64);
        if self.live.words.len() < needed {
            self.live.words.resize(needed.next_power_of_two(), 0);
        }
    }

    /// The internal ids a filter selects, or the first field it names that has
    /// no column.
    ///
    /// The error is the field name so that the caller can say which declaration
    /// is missing. It is deliberately the first one found rather than all of
    /// them, because one is enough to send the whole filter down the walk.
    pub(crate) fn select<'f>(&self, filter: &'f Filter) -> Result<Bitmap, &'f str> {
        // An index that declared nothing answers nothing, whatever the filter.
        //
        // Without this a filter carrying no field leaf at all is served from an
        // empty store and comes back empty. `{}` compiles to an empty
        // conjunction, which holds for every record, and an empty store has no
        // records to hold it, so `search(filter={})` returned no results and
        // `remove_where` had nothing to refuse. The name is what the tree
        // carries, or the empty string where it names no field, which the
        // warning already declines to print on an index with no declaration.
        if self.names.is_empty() {
            return Err(first_field_name(filter).unwrap_or(""));
        }
        match filter {
            Filter::Field { name, test } => {
                let position = *self.index_of.get(name).ok_or(name.as_str())?;
                Ok(self.columns[position].select(test, self.slots))
            }
            Filter::All(branches) => {
                let mut accumulated: Option<Bitmap> = None;
                for branch in branches {
                    let selected = self.select(branch)?;
                    accumulated = Some(match accumulated {
                        None => selected,
                        Some(mut running) => {
                            running.intersect(&selected);
                            running
                        }
                    });
                    // Nothing below can put a bit back, but every remaining
                    // branch would still walk its whole column to find that
                    // out.
                    if accumulated.as_ref().is_some_and(Bitmap::is_empty) {
                        break;
                    }
                }
                // An empty conjunction holds for every record, which is what
                // `Iterator::all` answers over no branches.
                Ok(accumulated.unwrap_or_else(|| self.live_set()))
            }
            Filter::Any(branches) => {
                // An empty disjunction holds for no record, which is what
                // `Iterator::any` answers over no branches.
                let mut accumulated = Bitmap::zeros(self.slots);
                for branch in branches {
                    let selected = self.select(branch)?;
                    accumulated.union(&selected);
                }
                Ok(accumulated)
            }
            Filter::Not(inner) => Ok(self.select(inner)?.complement_within(&self.live_set())),
        }
    }

    fn live_set(&self) -> Bitmap {
        let mut out = Bitmap::zeros(self.slots);
        let shared = out.words.len().min(self.live.words.len());
        out.words[..shared].copy_from_slice(&self.live.words[..shared]);
        out
    }

    /// **The invariant that ties a column to a record**, asserted by every path
    /// that writes one.
    ///
    /// A record at internal id `slot` occupies the slot, and every declared
    /// field reads back the value `vector_metadata` holds for that record, with
    /// a field the record does not carry reading absent. Checking it needs both
    /// stores, so the call sites are in `hnsw_index` and this is the half only
    /// the column can answer.
    ///
    /// **True for a store with no declaration, whatever the slot.** Such a
    /// store holds nothing and tracks nothing, so there is nothing to be out of
    /// step with. Requiring the live bit here instead is what the first version
    /// of this did, and it fired on the first debug build against every index
    /// created without `indexed_fields`, which is most of them.
    ///
    /// Compiled in every profile and called from every one, because
    /// `debug_assert!` type checks its expression in a release build even
    /// though it does not run it. The attribute is conditioned the same way the
    /// call is, so a debug build still reports it if the assertion is ever
    /// deleted.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn agrees_with(&self, slot: usize, metadata: &HashMap<String, Value>) -> bool {
        if self.names.is_empty() {
            return true;
        }
        self.live.contains(slot)
            && self
                .names
                .iter()
                .enumerate()
                .all(|(position, name)| self.columns[position].read(slot) == metadata.get(name))
    }

    /// Whether the store holds one entry per record the index holds.
    ///
    /// The other half of the invariant, asked once per removal and once per
    /// load rather than per field. True for a store with no declaration, for
    /// the reason above.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn tracks(&self, records: usize) -> bool {
        self.names.is_empty() || self.records == records
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.columns.iter().map(Column::heap_bytes).sum::<usize>()
            + self.live.heap_bytes()
            + self
                .names
                .iter()
                .map(|name| name.capacity() * 2 + 48)
                .sum::<usize>()
    }
}

// ============================================================================
// THE DECLARATION
// ============================================================================

/// Every rule a declared field list has to satisfy.
///
/// Called by `build` on what a caller passed and by the loader on what
/// `config.json` carried, so a directory holding a declaration this build
/// would refuse fails the load rather than producing an index whose columns
/// nothing can reach.
///
/// `source` prefixes the message and is empty for `build`, matching what
/// `validate_index_parameters` does with the same argument.
pub(crate) fn validate_indexed_fields(names: &[String], source: &str) -> PyResult<()> {
    if names.len() > MAX_INDEXED_FIELDS {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}indexed_fields names {} fields and the limit is {}. Declare the fields \
             you filter on; every other field is still stored and still filterable, \
             it just costs a walk of the metadata store.",
            source,
            names.len(),
            MAX_INDEXED_FIELDS
        )));
    }

    let mut seen: Vec<&str> = Vec::with_capacity(names.len());
    for name in names {
        if name.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{}indexed_fields contains an empty name. Every entry has to be the \
                 name of a metadata field.",
                source
            )));
        }
        if GROUP_KEYS.contains(&name.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{}indexed_fields names \"{}\", which is a reserved filter key rather \
                 than a metadata field. A field with that name cannot be filtered on, \
                 so a column for it could never be read.",
                source, name
            )));
        }
        if seen.contains(&name.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{}indexed_fields names \"{}\" twice. Each field is declared once.",
                source, name
            )));
        }
        seen.push(name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect()
    }

    fn store_of(names: &[&str], records: &[(usize, HashMap<String, Value>)]) -> ColumnStore {
        let mut store = ColumnStore::new(
            names.iter().map(|name| name.to_string()).collect(),
            records.len(),
        );
        for (slot, metadata) in records {
            store.write(*slot, metadata);
        }
        store
    }

    fn selected(store: &ColumnStore, filter: &Filter) -> Vec<usize> {
        let mut out = Vec::new();
        store
            .select(filter)
            .unwrap()
            .for_each(|slot| out.push(slot));
        out
    }

    fn field(name: &str, test: FieldTest) -> Filter {
        Filter::Field {
            name: name.to_string(),
            test,
        }
    }

    #[test]
    fn an_undeclared_field_names_itself() {
        let store = store_of(&["cat"], &[(1, record(&[("cat", json!("a"))]))]);
        let filter = field("lang", FieldTest::Equals(json!("en")));
        assert_eq!(store.select(&filter).err(), Some("lang"));
    }

    #[test]
    fn a_store_with_no_declaration_serves_nothing() {
        let store = ColumnStore::new(Vec::new(), 4);
        let filter = field("cat", FieldTest::Equals(json!("a")));
        assert_eq!(store.select(&filter).err(), Some("cat"));
        // Including a filter that names no field. An empty conjunction holds
        // for every record and an undeclared store knows of none, so answering
        // it here would return an empty page for `search(filter={})`.
        assert_eq!(store.select(&Filter::All(Vec::new())).err(), Some(""));
        assert_eq!(store.select(&Filter::Any(Vec::new())).err(), Some(""));
    }

    #[test]
    fn an_absent_field_never_matches() {
        let store = store_of(
            &["cat"],
            &[
                (1, record(&[("cat", json!("a"))])),
                (2, record(&[("other", json!("a"))])),
            ],
        );
        assert_eq!(
            selected(&store, &field("cat", FieldTest::Equals(json!("a")))),
            vec![1]
        );
        // `ne` excludes it too, which is the rule `field_matches` states.
        assert_eq!(
            selected(
                &store,
                &field(
                    "cat",
                    FieldTest::Operators(vec![(crate::filter::Op::Ne, json!("a"))])
                )
            ),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn not_selects_only_live_slots() {
        let mut store = store_of(
            &["cat"],
            &[
                (1, record(&[("cat", json!("a"))])),
                (2, record(&[("cat", json!("b"))])),
                (3, record(&[("cat", json!("a"))])),
            ],
        );
        store.erase(3);
        let filter = Filter::Not(Box::new(field("cat", FieldTest::Equals(json!("a")))));
        assert_eq!(selected(&store, &filter), vec![2]);
        assert_eq!(store.record_count(), 2);
    }

    #[test]
    fn an_empty_and_is_every_record_and_an_empty_or_is_none() {
        let store = store_of(
            &["cat"],
            &[
                (1, record(&[("cat", json!("a"))])),
                (2, record(&[("cat", json!("b"))])),
            ],
        );
        assert_eq!(selected(&store, &Filter::All(Vec::new())), vec![1, 2]);
        assert_eq!(
            selected(&store, &Filter::Any(Vec::new())),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn a_rewritten_record_loses_the_value_it_held() {
        let mut store = store_of(&["cat"], &[(1, record(&[("cat", json!("a"))]))]);
        store.write(1, &record(&[("cat", json!("b"))]));
        assert_eq!(
            selected(&store, &field("cat", FieldTest::Equals(json!("a")))),
            Vec::<usize>::new()
        );
        assert_eq!(
            selected(&store, &field("cat", FieldTest::Equals(json!("b")))),
            vec![1]
        );
        assert_eq!(store.record_count(), 1);
    }

    #[test]
    fn a_dropped_key_reads_absent() {
        let mut store = store_of(&["cat"], &[(1, record(&[("cat", json!("a"))]))]);
        store.write(1, &record(&[("other", json!("a"))]));
        assert_eq!(
            selected(&store, &field("cat", FieldTest::Equals(json!("a")))),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn a_high_cardinality_column_is_traded_for_a_plain_one_and_still_agrees() {
        let count = DICTIONARY_FLOOR * 2;
        let records: Vec<(usize, HashMap<String, Value>)> = (0..count)
            .map(|i| (i + 1, record(&[("rank", json!(i))])))
            .collect();
        let store = store_of(&["rank"], &records);
        assert!(matches!(store.columns[0], Column::Plain { .. }));
        assert_eq!(
            selected(
                &store,
                &field(
                    "rank",
                    FieldTest::Operators(vec![(crate::filter::Op::Lt, json!(3))])
                )
            ),
            vec![1, 2, 3]
        );
        for (slot, metadata) in &records {
            assert!(store.agrees_with(*slot, metadata));
        }
    }

    #[test]
    fn a_dictionary_entry_is_reclaimed_when_its_last_record_goes() {
        let mut store = store_of(
            &["cat"],
            &[
                (1, record(&[("cat", json!("a"))])),
                (2, record(&[("cat", json!("a"))])),
            ],
        );
        store.write(1, &record(&[("cat", json!("b"))]));
        store.write(2, &record(&[("cat", json!("b"))]));
        let Column::Dictionary { free, lookup, .. } = &store.columns[0] else {
            panic!("a two value column stays a dictionary");
        };
        assert_eq!(free.len(), 1);
        assert_eq!(lookup.len(), 1);
    }

    #[test]
    fn the_declaration_is_validated() {
        assert!(validate_indexed_fields(&["cat".into(), "lang".into()], "").is_ok());
        assert!(validate_indexed_fields(&["".into()], "").is_err());
        assert!(validate_indexed_fields(&["$or".into()], "").is_err());
        assert!(validate_indexed_fields(&["cat".into(), "cat".into()], "").is_err());
        let too_many: Vec<String> = (0..=MAX_INDEXED_FIELDS).map(|i| i.to_string()).collect();
        assert!(validate_indexed_fields(&too_many, "").is_err());
    }

    #[test]
    fn clear_keeps_the_declaration() {
        let mut store = store_of(&["cat"], &[(1, record(&[("cat", json!("a"))]))]);
        store.clear(4);
        assert_eq!(store.declared(), &["cat".to_string()]);
        assert_eq!(store.record_count(), 0);
        assert_eq!(
            selected(&store, &field("cat", FieldTest::Equals(json!("a")))),
            Vec::<usize>::new()
        );
    }
}
