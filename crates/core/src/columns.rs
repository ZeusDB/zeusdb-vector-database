//! A column per declared filterable field, and the bitmap a filter selects
//! into.
//!
//! # What this replaces
//!
//! A filtered search used to answer "which records match" by walking
//! `vector_metadata`, which is a `HashMap<String, HashMap<String, Value>>`.
//! That walk costs about 250 nanoseconds a record, so a filter matching ten
//! records over 100,000 cost 26 to 39 milliseconds where an unfiltered search
//! cost 0.3 to 1.2. The walk is 73 to 100 percent of a filtered search, and a
//! column per field was measured walking 224 times faster.
//!
//! A column here is addressed by internal id, so the same bitmap answers both
//! paths a filtered search can take. The exact scan reads the set bits and
//! scores those records. The graph traversal tests one bit per node it reaches,
//! in place of the node, `rev_map`, `vector_metadata`, field lookup chain that
//! measured at 154 nanoseconds a probe.
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
//! # What a column cannot serve, and what it can still bound
//!
//! A field that was not declared at `create()`. There is no column to read, so
//! the filter cannot be answered here, and that is a property of the
//! declaration rather than of the filter language.
//!
//! It does not follow that the declared fields are useless to such a filter. A
//! record has to satisfy every branch of a conjunction, so a conjunction naming
//! one declared field and one undeclared one matches a subset of what the
//! declared branch matches, and that subset is a candidate set the caller reads
//! instead of reading every record. [`ColumnStore::bound`] computes it, and it
//! carries a lower bound as well as an upper one because a negation needs the
//! two swapped. What comes back is [`Selection`], which says whether the answer
//! is the matching set, a superset of it, or nothing better than every record.
//!
//! A disjunction with an undeclared branch is the last of those, since that
//! branch could match anything and a union with the live set is the live set.

use crate::error::Error;
use crate::filter::{field_test_matches, FieldTest, Filter, Presence};
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
#[derive(Clone, Default)]
pub struct Bitmap {
    words: Vec<u64>,
}

impl Bitmap {
    fn zeros(slots: usize) -> Self {
        Bitmap {
            words: vec![0; slots.div_ceil(64)],
        }
    }

    /// An empty set whose words already reach `slots`, so a test of any id
    /// below that reads a word rather than falling off the end. What a
    /// measurement of the bit test's cost wants, since the set that admits
    /// nothing and holds no words answers without a read.
    pub fn with_slots(slots: usize) -> Self {
        Self::zeros(slots)
    }

    #[inline]
    fn set(&mut self, slot: usize) {
        self.words[slot >> 6] |= 1u64 << (slot & 63);
    }

    /// Put an internal id in the set, growing the words to reach it.
    ///
    /// The three methods below are what a set maintained beside a map needs,
    /// being the live record set the collection keeps under its reverse map.
    /// The columns never call them: every bitmap a column builds is sized to
    /// the store at once by [`Bitmap::zeros`] and filled by [`Bitmap::set`].
    pub fn insert(&mut self, slot: usize) {
        let index = slot >> 6;
        if index >= self.words.len() {
            self.words.resize(index + 1, 0);
        }
        self.words[index] |= 1u64 << (slot & 63);
    }

    /// Take an internal id out of the set. An id beyond the words was never in
    /// it, so there is nothing to clear.
    pub fn remove(&mut self, slot: usize) {
        if let Some(word) = self.words.get_mut(slot >> 6) {
            *word &= !(1u64 << (slot & 63));
        }
    }

    /// Empty the set, keeping its allocation for the ids about to refill it.
    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|word| *word = 0);
    }

    /// Whether an internal id is in the set.
    ///
    /// Total over every `usize`, so the traversal predicate can ask it about a
    /// node the store has never heard of and get `false` rather than a panic.
    #[inline]
    pub fn contains(&self, slot: usize) -> bool {
        self.words
            .get(slot >> 6)
            .is_some_and(|word| word >> (slot & 63) & 1 == 1)
    }

    pub fn count(&self) -> usize {
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
    pub fn for_each<F: FnMut(usize)>(&self, mut visit: F) {
        self.for_each_while(|slot| {
            visit(slot);
            true
        });
    }

    /// The same walk, stopping at the first slot the visitor declines.
    ///
    /// The bounded scan needs it. That scan gives up once too many records have
    /// matched, and a bound holding every slot in the store would otherwise be
    /// walked to the end after the give-up had already been decided.
    pub fn for_each_while<F: FnMut(usize) -> bool>(&self, mut visit: F) {
        for (index, mut word) in self.words.iter().copied().enumerate() {
            while word != 0 {
                if !visit(index * 64 + word.trailing_zeros() as usize) {
                    return;
                }
                word &= word - 1;
            }
        }
    }

    /// Bytes the words ask the allocator for.
    pub fn heap_bytes(&self) -> usize {
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

    /// Every slot this column holds a value for, whatever that value is
    ///
    /// This is what answers `exists` and `is_missing`. It reports a slot no
    /// record occupies as absent too, which is why the caller intersects with
    /// the live set rather than trusting it alone.
    fn present(&self, slots: usize) -> Bitmap {
        let mut out = Bitmap::zeros(slots);
        match self {
            Column::Plain { values } => {
                for (slot, value) in values.iter().enumerate().take(slots) {
                    if value.is_some() {
                        out.set(slot);
                    }
                }
            }
            Column::Dictionary { codes, .. } => {
                for (slot, &code) in codes.iter().enumerate().take(slots) {
                    if code != ABSENT {
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

/// How much of the corpus a bound has to remove before it is worth reporting.
///
/// # Why a bound that reads fewer records can still be slower
///
/// The walk iterates `vector_metadata` in the map's own layout and reads each
/// entry where it lies. A caller reading a bound takes a slot, looks it up in
/// `rev_map` to get an external id, and looks that id up in `vector_metadata`,
/// so it pays two hash lookups a candidate and an access pattern the map's
/// layout has nothing to do with. Measured at 100,000 records on sift-128, the
/// walk costs 0.345 microseconds a record and the bounded read 0.4 to 1.1 a
/// candidate.
///
/// A bound holding four fifths of the corpus therefore reads a fifth fewer
/// records at up to three times the cost each. Measured, a mixed filter whose
/// bound held 80,000 of 100,000 records cost 4.12 milliseconds through the
/// bound against 3.66 through the walk, which is 12 percent worse.
///
/// # Why one third
///
/// The upper end of that per-record ratio is 3.1, so a bound that removes two
/// thirds of the corpus breaks even in the worst case measured and wins in
/// every other. The filters this actually pays for are far below the line,
/// their bounds holding 1 to 20,000 records out of 100,000.
///
/// Between one third and the whole corpus the two are within a factor the
/// machine's own load moves them by. Two runs of the same cell put a bound of
/// half the corpus at 0.9 and 1.7 times the walk, so no direction is claimed
/// there and no bound is reported, which leaves those filters costing exactly
/// what they cost before.
///
/// # What it costs to decide
///
/// One pass of each declared column the filter names, thrown away where the
/// bound turns out not to pay. That is about 4 bytes a record a column, which
/// at 100,000 records is a tenth of a millisecond against a walk of 30.
const BOUND_PAYS_BELOW: usize = 3;

/// Whether reading the bound beats walking the metadata store. See
/// [`BOUND_PAYS_BELOW`].
fn bound_pays(candidates: usize, live: usize) -> bool {
    candidates.saturating_mul(BOUND_PAYS_BELOW) <= live
}

// ============================================================================
// WHAT THE COLUMNS CAN SAY ABOUT A FILTER
// ============================================================================

/// The answer [`ColumnStore::select`] gives, in three arms.
///
/// A filter every field of which is declared is answered outright. A filter
/// naming one field with no column is answered as far as the declared ones
/// reach, which for a conjunction is a candidate set the caller narrows and for
/// a disjunction is usually nothing at all. See [`ColumnStore::bound`] for
/// which shapes yield which.
pub enum Selection<'f> {
    /// Exactly the records the filter matches. Nothing else has to be read.
    Exact(Bitmap),
    /// A superset of the records the filter matches, and the first field it
    /// names that has no column. The caller reads each candidate's metadata and
    /// judges the whole filter on it, which is the same work the walk does over
    /// fewer records. **At most a third of the store**, since a bound that
    /// removes less than that is reported as `Whole` instead; see
    /// [`BOUND_PAYS_BELOW`].
    Narrowed(Bitmap, &'f str),
    /// The declared fields bound nothing, so the caller reads every record.
    /// Carries the first field with no column, for the warning.
    Whole(&'f str),
}

/// A matching set bracketed from both sides.
///
/// `Exact` is the two sides having met, which is every node of a fully declared
/// tree, and it is kept as its own arm so that such a tree allocates one bitmap
/// per node rather than two.
enum Bounds {
    Exact(Bitmap),
    /// `lower ⊆ matches ⊆ upper`.
    Range {
        lower: Bitmap,
        upper: Bitmap,
    },
}

impl Bounds {
    fn upper(&self) -> &Bitmap {
        match self {
            Bounds::Exact(bits) => bits,
            Bounds::Range { upper, .. } => upper,
        }
    }

    /// The two sides as owned bitmaps, which for `Exact` costs one clone.
    ///
    /// Paid once per combining step that has an inexact side, and never on a
    /// fully declared tree.
    fn split(self) -> (Bitmap, Bitmap) {
        match self {
            Bounds::Exact(bits) => (bits.clone(), bits),
            Bounds::Range { lower, upper } => (lower, upper),
        }
    }

    /// An empty upper bound means nothing matches, which is exact however it
    /// was arrived at. It is what lets a conjunction whose declared branch
    /// matches no record answer without a walk.
    fn normalised(self) -> Bounds {
        match self {
            Bounds::Range { upper, .. } if upper.is_empty() => Bounds::Exact(upper),
            other => other,
        }
    }

    fn intersected(self, other: Bounds) -> Bounds {
        match (self, other) {
            (Bounds::Exact(mut mine), Bounds::Exact(theirs)) => {
                mine.intersect(&theirs);
                Bounds::Exact(mine)
            }
            (mine, theirs) => {
                let (mut lower, mut upper) = mine.split();
                let (their_lower, their_upper) = theirs.split();
                lower.intersect(&their_lower);
                upper.intersect(&their_upper);
                Bounds::Range { lower, upper }.normalised()
            }
        }
    }

    fn united(self, other: Bounds) -> Bounds {
        match (self, other) {
            (Bounds::Exact(mut mine), Bounds::Exact(theirs)) => {
                mine.union(&theirs);
                Bounds::Exact(mine)
            }
            (mine, theirs) => {
                let (mut lower, mut upper) = mine.split();
                let (their_lower, their_upper) = theirs.split();
                lower.union(&their_lower);
                upper.union(&their_upper);
                Bounds::Range { lower, upper }.normalised()
            }
        }
    }

    /// `live` without this, which swaps the two sides.
    ///
    /// **The swap is the whole reason both sides are carried.** Complementing
    /// an upper bound gives a lower one, so a rule that propagated only upper
    /// bounds would hand a negation a subset of its matches and drop the rest.
    fn complemented(self, live: &Bitmap) -> Bounds {
        match self {
            Bounds::Exact(bits) => Bounds::Exact(bits.complement_within(live)),
            Bounds::Range { lower, upper } => Bounds::Range {
                lower: upper.complement_within(live),
                upper: lower.complement_within(live),
            }
            .normalised(),
        }
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
pub struct ColumnStore {
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
    pub fn new(names: Vec<String>, expected_size: usize) -> Self {
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

    pub fn declared(&self) -> &[String] {
        &self.names
    }

    pub fn is_declared(&self) -> bool {
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
    pub fn write(&mut self, slot: usize, metadata: &HashMap<String, Value>) {
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
    pub fn erase(&mut self, slot: usize) {
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
    pub fn clear(&mut self, expected_size: usize) {
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

    /// What a filter's declared fields can say about which records match.
    ///
    /// Three answers, and the caller does different work for each. See
    /// [`Selection`].
    ///
    /// **An index that declared nothing answers nothing, whatever the filter.**
    /// Its live bitmap is empty and its slot count is zero, so the bound below
    /// would compute an empty upper bound and drop every record. It is also
    /// what makes a filter carrying no field leaf at all behave: `{}` compiles
    /// to an empty conjunction, which holds for every record, and an empty
    /// store has no records to hold it, so answering it here returned no
    /// results for `search(filter={})` and gave `remove_where` nothing to
    /// refuse. The name is what the tree carries, or the empty string where it
    /// names no field, which the warning already declines to print on an index
    /// with no declaration.
    pub fn select<'f>(&self, filter: &'f Filter) -> Selection<'f> {
        if self.names.is_empty() {
            return Selection::Whole(self.first_undeclared(filter).unwrap_or(""));
        }
        match self.bound(filter) {
            Bounds::Exact(selected) => Selection::Exact(selected),
            Bounds::Range { upper, .. } => {
                let undeclared = self.first_undeclared(filter).unwrap_or("");
                // Every bitmap in the algebra above is a subset of the live
                // set, so a bound holding as many records as the store holds is
                // the live set and says nothing. One that says a little is not
                // worth reading either; see [`BOUND_PAYS_BELOW`].
                if bound_pays(upper.count(), self.records) {
                    Selection::Narrowed(upper, undeclared)
                } else {
                    Selection::Whole(undeclared)
                }
            }
        }
    }

    /// The first field the tree names that has no column, in evaluation order.
    ///
    /// Used only to say which declaration is missing, so it is deliberately the
    /// first one found rather than all of them. A tree that names no undeclared
    /// field at all is `{}`, a nest of empty groups, or a fully declared filter,
    /// and there is nothing to report about any of those.
    fn first_undeclared<'f>(&self, filter: &'f Filter) -> Option<&'f str> {
        match filter {
            Filter::Field { name, .. } | Filter::Presence { name, .. } => {
                (!self.index_of.contains_key(name)).then_some(name.as_str())
            }
            Filter::All(branches) | Filter::Any(branches) => branches
                .iter()
                .find_map(|branch| self.first_undeclared(branch)),
            Filter::Not(inner) => self.first_undeclared(inner),
        }
    }

    /// Bracket the matching set between what the columns prove matches and what
    /// they leave possible.
    ///
    /// # Why both sides
    ///
    /// An upper bound alone is enough for `$and` and `$or` and wrong for
    /// `$not`. A negation matches `live` without what its inner matches, so
    /// bounding it above needs a bound **below** the inner: from an upper bound
    /// on the inner, all that follows is `live \ upper ⊆ matches`, which is a
    /// subset and would drop matching records. Carrying both sides makes the
    /// complement a swap, which is what the `Not` arm does.
    ///
    /// # Why it is sound
    ///
    /// A structural induction on the tree, with `L ⊆ M ⊆ U` at every node.
    ///
    /// A declared leaf is exact, because [`Column::select`] and
    /// `crate::filter::field_matches` both end in `field_test_matches`, a
    /// record that does not carry the field holds [`ABSENT`] and never matches,
    /// and a slot holding no record holds `ABSENT` in every column.
    ///
    /// An undeclared leaf is bracketed by nothing and by the live set, which
    /// holds for any filter whatever.
    ///
    /// A conjunction intersects and a disjunction unions, and both preserve
    /// containment in both directions. A negation complements within the live
    /// set and swaps the sides, which turns `L ⊆ M` into `live \ M ⊆ live \ L`
    /// and the same for the other side.
    ///
    /// # What it costs
    ///
    /// A fully declared tree never leaves the `Exact` arm, so it allocates one
    /// bitmap per node exactly as it did before this existed. A tree with an
    /// undeclared leaf carries two bitmaps from that leaf upwards, which at
    /// 100,000 records is 12.5 kilobytes each.
    fn bound(&self, filter: &Filter) -> Bounds {
        match filter {
            Filter::Field { name, test } => match self.index_of.get(name) {
                Some(&position) => Bounds::Exact(self.columns[position].select(test, self.slots)),
                None => Bounds::Range {
                    lower: Bitmap::zeros(self.slots),
                    upper: self.live_set(),
                },
            },
            // A declared column answers presence exactly, because it holds a
            // value for a slot exactly when that record carries the field. Both
            // answers are taken inside the live set, so a slot no record
            // occupies never appears in either.
            Filter::Presence { name, want } => match self.index_of.get(name) {
                Some(&position) => {
                    let live = self.live_set();
                    let held = match want {
                        Presence::Present | Presence::Absent => {
                            self.columns[position].present(self.slots)
                        }
                        Presence::Null | Presence::NotNull => self.columns[position]
                            .select(&FieldTest::Equals(Value::Null), self.slots),
                    };
                    let mut out = live;
                    match want {
                        Presence::Present | Presence::Null => out.intersect(&held),
                        Presence::Absent | Presence::NotNull => out = held.complement_within(&out),
                    }
                    Bounds::Exact(out)
                }
                None => Bounds::Range {
                    lower: Bitmap::zeros(self.slots),
                    upper: self.live_set(),
                },
            },
            Filter::All(branches) => {
                let mut accumulated: Option<Bounds> = None;
                for branch in branches {
                    let bounded = self.bound(branch);
                    accumulated = Some(match accumulated {
                        None => bounded,
                        Some(running) => running.intersected(bounded),
                    });
                    // Nothing below can put a bit back, but every remaining
                    // branch would still walk its whole column to find that
                    // out.
                    if accumulated
                        .as_ref()
                        .is_some_and(|bounds| bounds.upper().is_empty())
                    {
                        break;
                    }
                }
                // An empty conjunction holds for every record, which is what
                // `Iterator::all` answers over no branches.
                accumulated.unwrap_or_else(|| Bounds::Exact(self.live_set()))
            }
            Filter::Any(branches) => {
                // An empty disjunction holds for no record, which is what
                // `Iterator::any` answers over no branches.
                let mut accumulated = Bounds::Exact(Bitmap::zeros(self.slots));
                for branch in branches {
                    accumulated = accumulated.united(self.bound(branch));
                }
                accumulated
            }
            Filter::Not(inner) => self.bound(inner).complemented(&self.live_set()),
        }
    }

    /// The bound the tree's shape yields, with the payment rule not applied.
    ///
    /// **The two rules are separate and are tested separately.** [`Self::bound`]
    /// answers the soundness question, which is about the shape of the tree,
    /// and [`bound_pays`] answers the cost question, which is about how many
    /// records the bound turned out to hold. Testing them through `select`
    /// alone would mean every shape case had to be built on a corpus the cost
    /// rule happens to accept, and a change to the constant would rewrite tests
    /// about soundness.
    ///
    /// `None` where the tree is answered exactly.
    #[cfg(test)]
    fn shape_bound(&self, filter: &Filter) -> Option<Bitmap> {
        match self.bound(filter) {
            Bounds::Exact(_) => None,
            Bounds::Range { upper, .. } => Some(upper),
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
    /// stores, so the call sites are in the index and this is the half only
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
    pub fn agrees_with(&self, slot: usize, metadata: &HashMap<String, Value>) -> bool {
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
    pub fn tracks(&self, records: usize) -> bool {
        self.names.is_empty() || self.records == records
    }

    pub fn heap_bytes(&self) -> usize {
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
pub fn validate_indexed_fields(names: &[String], source: &str) -> Result<(), Error> {
    if names.len() > MAX_INDEXED_FIELDS {
        return Err(Error::IndexedFieldsTooMany {
            source: source.to_string(),
            count: names.len(),
            max: MAX_INDEXED_FIELDS,
        });
    }

    let mut seen: Vec<&str> = Vec::with_capacity(names.len());
    for name in names {
        if name.is_empty() {
            return Err(Error::IndexedFieldEmpty {
                source: source.to_string(),
            });
        }
        if GROUP_KEYS.contains(&name.as_str()) {
            return Err(Error::IndexedFieldReserved {
                source: source.to_string(),
                name: name.clone(),
            });
        }
        if seen.contains(&name.as_str()) {
            return Err(Error::IndexedFieldRepeated {
                source: source.to_string(),
                name: name.clone(),
            });
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

    /// Slots paired with the metadata written to them, which every helper here
    /// takes and the bound sweep walks.
    type Records = Vec<(usize, HashMap<String, Value>)>;

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

    fn bits(bitmap: &Bitmap) -> Vec<usize> {
        let mut out = Vec::new();
        bitmap.for_each(|slot| out.push(slot));
        out
    }

    /// The records an exactly answered filter selects, failing the test where
    /// the filter was not answered exactly.
    fn selected(store: &ColumnStore, filter: &Filter) -> Vec<usize> {
        match store.select(filter) {
            Selection::Exact(bitmap) => bits(&bitmap),
            Selection::Narrowed(_, name) => {
                panic!("expected an exact answer, got a bound on {name}")
            }
            Selection::Whole(name) => panic!("expected an exact answer, got no bound on {name}"),
        }
    }

    /// The field with no column, and the records the tree's shape bounds the
    /// search to, or `None` where the shape bounds nothing.
    ///
    /// The cost rule is not applied, so these cases are about soundness alone.
    /// `answered` below is what a caller actually gets.
    fn partial(store: &ColumnStore, filter: &Filter) -> (String, Option<Vec<usize>>) {
        let name = store.first_undeclared(filter).unwrap_or("").to_string();
        match store.shape_bound(filter) {
            None => panic!("expected a partial answer, got an exact one"),
            // A bound holding every live record is no bound.
            Some(bitmap) if bitmap.count() == store.records => (name, None),
            Some(bitmap) => (name, Some(bits(&bitmap))),
        }
    }

    /// Which arm `select` returns, which is the shape rule and the cost rule
    /// together.
    fn answered(store: &ColumnStore, filter: &Filter) -> &'static str {
        match store.select(filter) {
            Selection::Exact(_) => "exact",
            Selection::Narrowed(..) => "narrowed",
            Selection::Whole(_) => "whole",
        }
    }

    /// Every live record the filter matches, by the walk rather than by a
    /// column, which is the answer a bound has to be a superset of.
    fn walked(store: &ColumnStore, records: &Records, filter: &Filter) -> Vec<usize> {
        records
            .iter()
            .filter(|(slot, _)| store.live.contains(*slot))
            .filter(|(_, metadata)| crate::filter::matches_filter(metadata, filter))
            .map(|(slot, _)| *slot)
            .collect()
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
        assert!(matches!(store.select(&filter), Selection::Whole("lang")));
    }

    #[test]
    fn a_store_with_no_declaration_serves_nothing() {
        let store = ColumnStore::new(Vec::new(), 4);
        let filter = field("cat", FieldTest::Equals(json!("a")));
        assert!(matches!(store.select(&filter), Selection::Whole("cat")));
        // Including a filter that names no field. An empty conjunction holds
        // for every record and an undeclared store knows of none, so answering
        // it here would return an empty page for `search(filter={})`.
        assert!(matches!(
            store.select(&Filter::All(Vec::new())),
            Selection::Whole("")
        ));
        assert!(matches!(
            store.select(&Filter::Any(Vec::new())),
            Selection::Whole("")
        ));
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

    // ------------------------------------------------------------------
    // A FILTER MIXING A DECLARED FIELD WITH ONE THAT HAS NO COLUMN
    // ------------------------------------------------------------------

    /// Four records over two fields, one declared and one not.
    ///
    /// `cat` is declared and splits them two and two. `price` is not declared
    /// and splits them differently, so no bound derived from `cat` alone can be
    /// the matching set and every assertion below is about containment rather
    /// than equality.
    fn mixed() -> (ColumnStore, Records) {
        let records = vec![
            (1, record(&[("cat", json!("a")), ("price", json!(10))])),
            (2, record(&[("cat", json!("a")), ("price", json!(30))])),
            (3, record(&[("cat", json!("b")), ("price", json!(10))])),
            (4, record(&[("cat", json!("b")), ("price", json!(30))])),
        ];
        (store_of(&["cat"], &records), records)
    }

    fn cat(value: &str) -> Filter {
        field("cat", FieldTest::Equals(json!(value)))
    }

    fn cheap() -> Filter {
        field(
            "price",
            FieldTest::Operators(vec![(crate::filter::Op::Lt, json!(20))]),
        )
    }

    /// The one shape that pays. A conjunction bounds by its declared branch,
    /// because a record has to satisfy every branch.
    #[test]
    fn a_conjunction_is_bounded_by_its_declared_branch() {
        let (store, records) = mixed();
        let filter = Filter::All(vec![cat("a"), cheap()]);
        let (name, bound) = partial(&store, &filter);
        assert_eq!(name, "price");
        assert_eq!(bound, Some(vec![1, 2]));
        // The bound is a superset of what the walk answers, which is the whole
        // requirement.
        assert_eq!(walked(&store, &records, &filter), vec![1]);
    }

    /// A disjunction bounds nothing, because the undeclared branch could match
    /// any record and a union with the live set is the live set.
    #[test]
    fn a_disjunction_with_an_undeclared_branch_bounds_nothing() {
        let (store, _) = mixed();
        let filter = Filter::Any(vec![cat("a"), cheap()]);
        assert_eq!(partial(&store, &filter), ("price".to_string(), None));
    }

    /// A negation of an undeclared leaf bounds nothing, for the same reason
    /// read the other way round.
    #[test]
    fn a_negated_undeclared_leaf_bounds_nothing() {
        let (store, _) = mixed();
        let filter = Filter::Not(Box::new(cheap()));
        assert_eq!(partial(&store, &filter), ("price".to_string(), None));
    }

    /// A negated disjunction bounds, because it distributes into a conjunction
    /// of negations and the declared one of those still holds.
    #[test]
    fn a_negated_disjunction_is_bounded_by_its_declared_branch() {
        let (store, records) = mixed();
        let filter = Filter::Not(Box::new(Filter::Any(vec![cat("a"), cheap()])));
        let (name, bound) = partial(&store, &filter);
        assert_eq!(name, "price");
        // Not cat a, which is the complement of the declared branch.
        assert_eq!(bound, Some(vec![3, 4]));
        assert_eq!(walked(&store, &records, &filter), vec![4]);
    }

    /// A negated conjunction bounds nothing, because it distributes into a
    /// disjunction of negations and one of those is unbounded.
    #[test]
    fn a_negated_conjunction_mixing_the_two_bounds_nothing() {
        let (store, _) = mixed();
        let filter = Filter::Not(Box::new(Filter::All(vec![cat("a"), cheap()])));
        assert_eq!(partial(&store, &filter), ("price".to_string(), None));
    }

    /// A conjunction carrying a negated undeclared leaf still bounds, since the
    /// negation contributes the live set and the intersection keeps the
    /// declared branch.
    #[test]
    fn a_conjunction_with_a_negated_undeclared_branch_still_bounds() {
        let (store, records) = mixed();
        let filter = Filter::All(vec![cat("b"), Filter::Not(Box::new(cheap()))]);
        let (name, bound) = partial(&store, &filter);
        assert_eq!(name, "price");
        assert_eq!(bound, Some(vec![3, 4]));
        assert_eq!(walked(&store, &records, &filter), vec![4]);
    }

    /// A conjunction whose declared branch matches nothing is answered
    /// outright, because an empty upper bound proves the matching set is empty.
    #[test]
    fn an_empty_bound_is_an_exact_answer() {
        let (store, _) = mixed();
        let filter = Filter::All(vec![cat("nothing"), cheap()]);
        assert_eq!(selected(&store, &filter), Vec::<usize>::new());
    }

    /// A conjunction of undeclared leaves bounds nothing, which is what the
    /// index did before any of this existed.
    #[test]
    fn a_conjunction_of_undeclared_leaves_bounds_nothing() {
        let (store, _) = mixed();
        let filter = Filter::All(vec![cheap(), field("lang", FieldTest::Equals(json!("en")))]);
        assert_eq!(partial(&store, &filter), ("price".to_string(), None));
    }

    /// A bound that turns out to hold every live record bounds nothing, since
    /// reading it would cost a pass and save nothing.
    #[test]
    fn a_bound_holding_every_record_is_reported_as_none() {
        let (store, _) = mixed();
        let unconditional = field(
            "cat",
            FieldTest::Operators(vec![(crate::filter::Op::All, json!([]))]),
        );
        let filter = Filter::All(vec![unconditional, cheap()]);
        assert_eq!(partial(&store, &filter), ("price".to_string(), None));
    }

    // ------------------------------------------------------------------
    // THE COST RULE, WHICH IS NOT THE SOUNDNESS RULE
    // ------------------------------------------------------------------

    /// A bound is only reported where reading it beats the walk.
    ///
    /// Twelve records over a declared `cat` that splits them four ways. Three
    /// records is a quarter of the store, which pays; six is a half, which does
    /// not. The shape is the same conjunction in both cases, so what differs
    /// here is nothing but how many records the declared branch matched.
    #[test]
    fn a_bound_that_removes_too_little_is_not_reported() {
        let records: Records = (0..12)
            .map(|i| {
                (
                    i + 1,
                    record(&[("cat", json!(format!("c{}", i % 4))), ("price", json!(i))]),
                )
            })
            .collect();
        let store = store_of(&["cat"], &records);

        // One category in four, which is three of twelve.
        let narrow = Filter::All(vec![cat("c0"), cheap()]);
        assert_eq!(answered(&store, &narrow), "narrowed");

        // Two categories in four, which is six of twelve.
        let wide = Filter::All(vec![Filter::Any(vec![cat("c0"), cat("c1")]), cheap()]);
        assert_eq!(answered(&store, &wide), "whole");
        // And the shape still bounds it, which is what says the two rules are
        // separate rather than one rule stated twice.
        assert_eq!(
            partial(&store, &wide),
            ("price".to_string(), Some(vec![1, 2, 5, 6, 9, 10]))
        );
    }

    /// An empty bound is exact whatever its size rule says, because nothing
    /// matching is an answer rather than a bound.
    #[test]
    fn an_empty_bound_is_never_traded_for_the_walk() {
        let records: Records = (0..12)
            .map(|i| (i + 1, record(&[("cat", json!("c0")), ("price", json!(i))])))
            .collect();
        let store = store_of(&["cat"], &records);
        let filter = Filter::All(vec![cat("nothing"), cheap()]);
        assert_eq!(answered(&store, &filter), "exact");
    }

    /// **The containment property itself**, over every shape above and both of
    /// the two fields' conditions.
    ///
    /// A bound that dropped one matching record would make a filtered search
    /// return a short page, so this asserts the one thing the whole design
    /// rests on rather than trusting the shape by shape cases to cover it.
    #[test]
    fn every_bound_contains_every_record_the_walk_finds() {
        let (store, records) = mixed();
        let declared = [cat("a"), cat("b"), Filter::Not(Box::new(cat("a")))];
        let undeclared = [cheap(), Filter::Not(Box::new(cheap()))];
        for left in &declared {
            for right in &undeclared {
                let shapes = [
                    Filter::All(vec![clone_filter(left), clone_filter(right)]),
                    Filter::Any(vec![clone_filter(left), clone_filter(right)]),
                    Filter::Not(Box::new(Filter::All(vec![
                        clone_filter(left),
                        clone_filter(right),
                    ]))),
                    Filter::Not(Box::new(Filter::Any(vec![
                        clone_filter(left),
                        clone_filter(right),
                    ]))),
                ];
                for shape in shapes {
                    let expected = walked(&store, &records, &shape);
                    let bound = match store.select(&shape) {
                        Selection::Exact(bitmap) => bits(&bitmap),
                        Selection::Narrowed(bitmap, _) => bits(&bitmap),
                        // No bound is the live set, which contains everything.
                        Selection::Whole(_) => vec![1, 2, 3, 4],
                    };
                    for slot in expected {
                        assert!(
                            bound.contains(&slot),
                            "bound {bound:?} dropped record {slot}"
                        );
                    }
                }
            }
        }
    }

    /// `Filter` is not `Clone`, and the sweep above needs one tree per shape.
    fn clone_filter(filter: &Filter) -> Filter {
        match filter {
            Filter::Field { name, test } => Filter::Field {
                name: name.clone(),
                test: match test {
                    FieldTest::Equals(value) => FieldTest::Equals(value.clone()),
                    FieldTest::Operators(operations) => FieldTest::Operators(operations.clone()),
                },
            },
            Filter::All(branches) => Filter::All(branches.iter().map(clone_filter).collect()),
            Filter::Any(branches) => Filter::Any(branches.iter().map(clone_filter).collect()),
            Filter::Not(inner) => Filter::Not(Box::new(clone_filter(inner))),
            Filter::Presence { name, want } => Filter::Presence {
                name: name.clone(),
                want: *want,
            },
        }
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
