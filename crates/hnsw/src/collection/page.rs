//! One dense search's page, owned, in the shape the binding reads.
//!
//! The page is built under the search's read guards and read after they are
//! released. The binding turns it into Python objects with the interpreter
//! lock held, and no engine guard is held while that lock is taken, so
//! nothing on the page borrows from what the guards protect.
//!
//! It is held in a handful of arrays rather than in an allocation per value.
//! Every external id and every string value is appended to one text buffer,
//! every metadata field to one list and every returned vector to one list of
//! floats, and a hit names its ranges in them. A field name is held once for
//! the page, however many hits carry it, keyed by the symbol the metadata
//! store interned it under.
//!
//! A hit reads back as the record it names: its external id, its score, the
//! mapping its metadata was inserted as, and the vector the search returned
//! for it.

use serde_json::Value;
use std::collections::HashMap;
use std::ops::Range;
use zeusdb_vector_core::RecordFields;

/// The most bytes the first hit reserves in any one array for the hits after
/// it.
///
/// A page of records of one shape then grows each array once. A page whose
/// records differ from its first grows past this by doubling, so one record
/// with a long value or many fields cannot reserve room for a whole page of
/// records like it.
const RESERVE_CAP_BYTES: usize = 1 << 20;

/// One metadata value on a page.
#[derive(Debug)]
enum Cell {
    /// A string, as a range of the page's text.
    Text(Range<usize>),
    /// Any other value, cloned. A null, a boolean and a number clone without
    /// allocating, and an array or an object clones whole.
    Other(Value),
}

/// One hit, as ranges of the page's arrays.
#[derive(Debug)]
struct Entry {
    id: Range<usize>,
    score: f32,
    fields: Range<usize>,
    vector: Option<Range<usize>>,
}

/// One dense search's hits, best first.
///
/// Read through [`QueryHits::iter`], which yields a [`HitRef`] a hit.
#[derive(Debug, Default)]
pub struct QueryHits {
    entries: Vec<Entry>,
    /// Every id, string value and field name on the page, end to end.
    text: String,
    /// Each field name the page carries, as a range of `text`, at the
    /// position the page numbers it by.
    names: Vec<Range<usize>>,
    /// The store's symbol for each name on the page, sorted, beside the
    /// page's number for it.
    symbols: Vec<(u32, u32)>,
    /// Every hit's fields, end to end, as the page's number for the name and
    /// the value.
    fields: Vec<(u32, Cell)>,
    /// Every returned vector, end to end.
    vectors: Vec<f32>,
}

impl QueryHits {
    /// An empty page with room for `hits` hits.
    pub(super) fn with_capacity(hits: usize) -> Self {
        QueryHits {
            entries: Vec::with_capacity(hits),
            ..QueryHits::default()
        }
    }

    /// Append a hit.
    ///
    /// `fields` is the record's metadata entry, or `None` where it holds none,
    /// which reads back as an empty mapping as a record inserted with no
    /// fields does. The vector is copied.
    pub(super) fn push(
        &mut self,
        id: &str,
        score: f32,
        fields: Option<RecordFields<'_>>,
        vector: Option<&[f32]>,
    ) {
        if self.entries.is_empty() {
            self.reserve_like(id, fields, vector);
        }
        let id = self.append(id);
        let first = self.fields.len();
        if let Some(fields) = fields {
            for (symbol, name, value) in fields.entries() {
                let name = self.number(symbol, name);
                let cell = match value {
                    Value::String(text) => Cell::Text(self.append(text)),
                    other => Cell::Other(other.clone()),
                };
                self.fields.push((name, cell));
            }
        }
        let vector = vector.map(|values| {
            let start = self.vectors.len();
            self.vectors.extend_from_slice(values);
            start..self.vectors.len()
        });
        self.entries.push(Entry {
            id,
            score,
            fields: first..self.fields.len(),
            vector,
        });
    }

    /// Reserve each array for as many hits the size of the first as the page
    /// was made for, under [`RESERVE_CAP_BYTES`] an array.
    fn reserve_like(&mut self, id: &str, fields: Option<RecordFields<'_>>, vector: Option<&[f32]>) {
        let hits = self.entries.capacity().max(1);
        let (count, names, values) = fields.map_or((0, 0, 0), |fields| {
            fields
                .iter()
                .fold((0, 0, 0), |(count, names, values), (name, value)| {
                    (
                        count + 1,
                        names + name.len(),
                        values + value.as_str().map_or(0, str::len),
                    )
                })
        });
        let text = hits
            .saturating_mul(id.len() + values)
            .saturating_add(names)
            .min(RESERVE_CAP_BYTES);
        let fields = hits
            .saturating_mul(count)
            .min(RESERVE_CAP_BYTES / std::mem::size_of::<(u32, Cell)>());
        let floats = hits
            .saturating_mul(vector.map_or(0, <[f32]>::len))
            .min(RESERVE_CAP_BYTES / std::mem::size_of::<f32>());
        self.text.reserve(text);
        self.fields.reserve(fields);
        self.vectors.reserve(floats);
    }

    /// Append `text` to the page's text, returning its range.
    fn append(&mut self, text: &str) -> Range<usize> {
        let start = self.text.len();
        self.text.push_str(text);
        start..self.text.len()
    }

    /// The page's number for the field name the store interned as `symbol`,
    /// taking the name onto the page the first time a hit carries it.
    fn number(&mut self, symbol: u32, name: &str) -> u32 {
        match self
            .symbols
            .binary_search_by_key(&symbol, |&(held, _)| held)
        {
            Ok(at) => self.symbols[at].1,
            Err(at) => {
                let number = u32::try_from(self.names.len())
                    .expect("a page holds no more names than the store has symbols");
                let range = self.append(name);
                self.names.push(range);
                self.symbols.insert(at, (symbol, number));
                number
            }
        }
    }

    /// How many hits the page holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the page holds no hit.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every field name the page's hits carry, once each, in the order
    /// [`HitRef::fields`] numbers them by.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        self.names.iter().map(|range| &self.text[range.clone()])
    }

    /// The hit at `position`, counted from the best.
    pub fn get(&self, position: usize) -> Option<HitRef<'_>> {
        self.entries
            .get(position)
            .map(|entry| HitRef { page: self, entry })
    }

    /// Every hit, best first.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = HitRef<'_>> + '_ {
        self.entries
            .iter()
            .map(|entry| HitRef { page: self, entry })
    }
}

/// One hit on a page, borrowed from it.
#[derive(Clone, Copy, Debug)]
pub struct HitRef<'a> {
    page: &'a QueryHits,
    entry: &'a Entry,
}

impl<'a> HitRef<'a> {
    /// The record's external id.
    pub fn id(self) -> &'a str {
        &self.page.text[self.entry.id.clone()]
    }

    /// The score the search ranked the record by.
    pub fn score(self) -> f32 {
        self.entry.score
    }

    /// The record's metadata fields, each as the page's number for its name
    /// and its value, in the order the metadata store holds them.
    pub fn fields(self) -> impl ExactSizeIterator<Item = (usize, FieldRef<'a>)> + 'a {
        let page = self.page;
        page.fields[self.entry.fields.clone()]
            .iter()
            .map(move |(name, cell)| {
                let value = match cell {
                    Cell::Text(range) => FieldRef::Text(&page.text[range.clone()]),
                    Cell::Other(value) => FieldRef::Other(value),
                };
                (*name as usize, value)
            })
    }

    /// The record's metadata as the mapping it was inserted as.
    pub fn metadata(self) -> HashMap<String, Value> {
        let page = self.page;
        self.fields()
            .map(|(name, value)| {
                let value = match value {
                    FieldRef::Text(text) => Value::String(text.to_string()),
                    FieldRef::Other(value) => value.clone(),
                };
                (page.text[page.names[name].clone()].to_string(), value)
            })
            .collect()
    }

    /// The vector the search returned for the record, where the search asked
    /// for vectors and the index holds or can reconstruct one.
    pub fn vector(self) -> Option<&'a [f32]> {
        self.entry
            .vector
            .clone()
            .map(|range| &self.page.vectors[range])
    }
}

/// One metadata value, borrowed from a page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldRef<'a> {
    /// A string.
    Text(&'a str),
    /// Any value that is not a string.
    Other(&'a Value),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zeusdb_vector_core::MetadataStore;

    fn mapping(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect()
    }

    /// Every hit reads back as the id, the score, the mapping and the vector
    /// it was pushed with, whatever kind each value is, and a record holding
    /// no entry reads back as an empty mapping as an empty entry does.
    #[test]
    fn a_page_reads_back_what_each_hit_was_pushed_with() {
        let mut store = MetadataStore::new(8);
        store.insert(
            1,
            mapping(&[
                ("cat", json!("alpha")),
                ("rank", json!(7)),
                ("flag", json!(true)),
            ]),
        );
        store.insert(
            2,
            mapping(&[
                ("rank", json!(-3)),
                ("ratio", json!(0.25)),
                ("none", Value::Null),
                ("big", json!(u64::MAX)),
                ("cat", json!("")),
            ]),
        );
        store.insert(
            3,
            mapping(&[
                ("tags", json!(["a", "b", 3])),
                ("nested", json!({"k": "v", "n": [1.5, null]})),
                ("cat", json!("\u{e9}\u{4e2d}")),
            ]),
        );
        store.insert(4, HashMap::new());

        let first = [1.0f32, -0.0, f32::MIN_POSITIVE];
        let third = [3.5f32, f32::MAX, -1.25];
        let pushed: [(&str, f32, usize, Option<&[f32]>); 5] = [
            ("r1", 0.5, 1, Some(&first)),
            ("r2", -0.0, 2, None),
            ("", 1.25, 3, Some(&third)),
            ("r4", 2.0, 4, None),
            ("r5", 3.0, 5, None),
        ];

        // Room for two, so the page grows past what its first hit reserved.
        let mut page = QueryHits::with_capacity(2);
        for (id, score, slot, vector) in pushed {
            page.push(id, score, store.get(slot), vector);
        }

        assert_eq!(page.len(), 5);
        assert!(!page.is_empty());
        for ((id, score, slot, vector), hit) in pushed.iter().zip(page.iter()) {
            assert_eq!(hit.id(), *id);
            assert_eq!(hit.score().to_bits(), score.to_bits());
            let want = store
                .get(*slot)
                .map(|fields| fields.to_map())
                .unwrap_or_default();
            assert_eq!(hit.metadata(), want, "slot {slot}");
            let got = hit
                .vector()
                .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            let wanted = vector.map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(got, wanted, "slot {slot}");
            // The store's order, and a string as text and nothing else.
            let names: Vec<&str> = hit
                .fields()
                .map(|(name, _)| page.names().nth(name).unwrap())
                .collect();
            let held: Vec<&str> = store
                .get(*slot)
                .map(|fields| fields.iter().map(|(name, _)| name).collect())
                .unwrap_or_default();
            assert_eq!(names, held, "slot {slot}");
            for (_, value) in hit.fields() {
                match value {
                    FieldRef::Text(_) => {}
                    FieldRef::Other(other) => assert!(!other.is_string()),
                }
            }
        }

        // Each name once, however many hits carry it.
        let mut names: Vec<&str> = page.names().collect();
        assert_eq!(names.len(), 8);
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 8);
        assert!(page.get(5).is_none());
    }

    /// An empty page holds nothing and reserves nothing past its entries.
    #[test]
    fn an_empty_page_is_empty() {
        let page = QueryHits::with_capacity(10);
        assert!(page.is_empty());
        assert_eq!(page.len(), 0);
        assert_eq!(page.names().len(), 0);
        assert!(page.iter().next().is_none());
        assert_eq!(page.text.capacity(), 0);
        assert_eq!(page.fields.capacity(), 0);
        assert_eq!(page.vectors.capacity(), 0);
    }

    /// A first hit with a long value reserves no more than the cap past it,
    /// on a page made for the largest page a search may return.
    #[test]
    fn a_long_first_hit_does_not_reserve_for_a_page_of_its_like() {
        let mut store = MetadataStore::new(2);
        let long = "x".repeat(2 << 20);
        store.insert(1, mapping(&[("text", json!(long))]));
        let wide = vec![0.5f32; 4096];
        let mut page = QueryHits::with_capacity(65_536);
        page.push("r1", 0.0, store.get(1), Some(&wide));
        assert!(page.text.capacity() <= (2 << 20) + 2 * RESERVE_CAP_BYTES);
        assert!(page.vectors.capacity() * std::mem::size_of::<f32>() <= 2 * RESERVE_CAP_BYTES);
        assert_eq!(
            page.get(0).unwrap().metadata(),
            store.get(1).unwrap().to_map()
        );
        assert_eq!(page.get(0).unwrap().vector(), Some(&wide[..]));
    }
}
