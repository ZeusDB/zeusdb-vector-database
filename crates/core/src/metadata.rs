//! The per record metadata, held by internal id.
//!
//! One entry per record, by internal id, and a record's fields in one block
//! behind it. A record that carries no metadata costs its entry and nothing
//! else, which is sixteen bytes, and a record that carries fields costs one
//! block of forty bytes a field beside the text of its string values. Field
//! names are interned once for the whole store, so a record holds a four
//! byte symbol for each name rather than its own copy of it.
//!
//! This replaced a `HashMap<String, HashMap<String, Value>>` keyed by external
//! id. That map held a third copy of every id and a 72 byte bucket per record
//! whether the record carried anything or not, and a record with two small
//! fields paid a four bucket inner table of 56 byte buckets, 244 bytes, for
//! sixteen bytes of payload. Every reader of a record's metadata already held
//! the internal id or reached the forward map first, so nothing needed the
//! external key.
//!
//! **What comes back is the same mapping.** [`RecordFields::to_map`] rebuilds
//! the `HashMap<String, Value>` a caller handed in, and [`FieldLookup`] lets a
//! filter read a field without rebuilding anything. `metadata.json` is written
//! from and read into the same shape it always was.
//!
//! Internal ids are never reused, and `compact` re-inserts every record under
//! the id it already holds, so the ids the store holds spread over a range
//! far larger than its records. The entries sit in an [`IdMap`], flat while
//! the ids are dense, which is the vector indexed by id the store always
//! held, and in pages once they are sparse, so the store costs what its
//! records cost however far the ids spread. The entries are reserved for the
//! declared record count, under a cap, and grow by doubling past it, which is
//! the rule the graph's per node arrays follow.

use crate::filter::FieldLookup;
use crate::idmap::IdMap;
use serde_json::Value;
use std::collections::HashMap;

/// The most entries the store reserves at creation, whatever the declaration.
///
/// Sixteen bytes an entry, so 16 MiB. A declaration past this grows by
/// doubling, and a declaration under it costs exactly what it declares.
const RESERVE_CAP: usize = 1 << 20;

/// One field of one record: the symbol of its name and its value.
///
/// Forty bytes, being a `Value` at thirty-two and a symbol padded to eight.
struct Field {
    key: u32,
    value: Value,
}

/// Every record's metadata, indexed by internal id.
pub struct MetadataStore {
    /// Field names in the order first seen, indexed by symbol.
    names: Vec<String>,
    /// Field name to symbol.
    symbols: HashMap<String, u32>,
    /// One entry per record, by internal id. `None` is the map's absent
    /// value, being a removed record or one no insertion has reached.
    /// `Some` of an empty block is a record inserted with no fields, which
    /// is distinct: a filter judges an empty mapping and never judges an
    /// absent one.
    records: IdMap<Option<Box<[Field]>>>,
}

impl MetadataStore {
    /// An empty store reserved for `expected_size` records, under the cap.
    ///
    /// One slot more than the declaration, because internal ids are issued
    /// from one and the slot is the id, so a declaration filled exactly
    /// reaches slot `expected_size` and would double the vector for it.
    pub fn new(expected_size: usize) -> Self {
        MetadataStore {
            names: Vec::new(),
            symbols: HashMap::new(),
            records: IdMap::with_capacity(expected_size.saturating_add(1).min(RESERVE_CAP)),
        }
    }

    /// The symbol for a field name, interning it on first sight.
    fn symbol(&mut self, name: String) -> u32 {
        if let Some(&symbol) = self.symbols.get(&name) {
            return symbol;
        }
        let symbol = u32::try_from(self.names.len()).expect("fewer than 2^32 field names");
        self.names.push(name.clone());
        self.symbols.insert(name, symbol);
        symbol
    }

    /// Set the record at `slot` to exactly `metadata`, replacing whatever it
    /// held. An empty mapping is held as an entry with no fields.
    pub fn insert(&mut self, slot: usize, metadata: HashMap<String, Value>) {
        let mut fields: Vec<Field> = metadata
            .into_iter()
            .map(|(name, value)| Field {
                key: self.symbol(name),
                value,
            })
            .collect();
        // Symbol order, so a lookup can bisect and two records with the same
        // fields lay them out the same way.
        fields.sort_unstable_by_key(|field| field.key);
        self.records.insert(slot, Some(fields.into_boxed_slice()));
    }

    /// Forget the record at `slot`, reporting whether it held an entry.
    pub fn remove(&mut self, slot: usize) -> bool {
        self.records.remove(slot).is_some()
    }

    /// Forget every record and every field name, keeping a reservation for
    /// `expected_size` records.
    pub fn clear(&mut self, expected_size: usize) {
        *self = MetadataStore::new(expected_size);
    }

    /// Decide the form for a run of `records` insertions whose largest
    /// internal id is `highest`, from the run's final shape rather than the
    /// order the run comes in, which a load's run does not keep.
    pub fn plan(&mut self, records: usize, highest: usize) {
        self.records.plan(records, highest);
    }

    /// Settle the entries once a load has written every record, so a paged
    /// store holds the same bytes whatever order the records came in. A flat
    /// store is left as it is.
    pub fn settle(&mut self) {
        self.records.settle();
    }

    /// Whether the entries are flat, one slot per internal id up to the
    /// largest, rather than paged; see [`IdMap`].
    pub fn is_flat(&self) -> bool {
        self.records.is_flat()
    }

    /// The record at `slot`, or `None` where it holds no entry.
    #[inline]
    pub fn get(&self, slot: usize) -> Option<RecordFields<'_>> {
        let fields = self.records.get(slot)?.as_deref()?;
        Some(RecordFields {
            store: self,
            fields,
        })
    }

    /// How many records hold an entry.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no record holds an entry.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Every record holding an entry, in increasing internal id order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, RecordFields<'_>)> + '_ {
        self.records.iter().filter_map(|(slot, entry)| {
            entry.as_deref().map(|fields| {
                (
                    slot,
                    RecordFields {
                        store: self,
                        fields,
                    },
                )
            })
        })
    }

    /// The field name table, for the memory report to price as the hash
    /// table it is. The names' text is priced by [`MetadataStore::heap_bytes`].
    pub fn key_table(&self) -> &HashMap<String, u32> {
        &self.symbols
    }

    /// Bytes the store asked the allocator for, apart from the key table.
    ///
    /// The entries at their capacity, every record's block at its length, the
    /// text of every string value and the name list with its text. A `Value`
    /// is thirty-two bytes wherever it sits and a string is the one variant
    /// that also owns text.
    pub fn heap_bytes(&self) -> usize {
        let entries = self.records.heap_bytes();
        let blocks: usize = self
            .records
            .iter()
            .filter_map(|(_, entry)| entry.as_deref())
            .map(|fields| {
                std::mem::size_of_val(fields)
                    + fields
                        .iter()
                        .filter_map(|field| field.value.as_str())
                        .map(str::len)
                        .sum::<usize>()
            })
            .sum();
        let names = self.names.capacity() * std::mem::size_of::<String>()
            + self.names.iter().map(String::len).sum::<usize>();
        entries + blocks + names
    }
}

/// One record's fields, borrowed from the store.
#[derive(Clone, Copy)]
pub struct RecordFields<'a> {
    store: &'a MetadataStore,
    fields: &'a [Field],
}

impl<'a> RecordFields<'a> {
    /// How many fields the record carries.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the record carries no field.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The fields as name and value, in symbol order.
    pub fn iter(&self) -> impl Iterator<Item = (&'a str, &'a Value)> + 'a {
        let names = &self.store.names;
        self.fields
            .iter()
            .map(move |field| (names[field.key as usize].as_str(), &field.value))
    }

    /// The fields as the symbol of the name, the name and the value, in
    /// symbol order. A symbol stands for one name until the store is
    /// cleared, so a reader holding the store's guard may key a name on it.
    pub fn entries(&self) -> impl Iterator<Item = (u32, &'a str, &'a Value)> + 'a {
        let names = &self.store.names;
        self.fields
            .iter()
            .map(move |field| (field.key, names[field.key as usize].as_str(), &field.value))
    }

    /// The mapping the record was inserted with.
    pub fn to_map(&self) -> HashMap<String, Value> {
        self.iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect()
    }
}

impl FieldLookup for RecordFields<'_> {
    #[inline]
    fn field(&self, name: &str) -> Option<&Value> {
        let symbol = *self.store.symbols.get(name)?;
        self.fields
            .binary_search_by_key(&symbol, |field| field.key)
            .ok()
            .map(|at| &self.fields[at].value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect()
    }

    /// What goes in comes back, field for field, whatever order the names
    /// were first seen in.
    #[test]
    fn a_record_reads_back_as_the_mapping_it_was_inserted_with() {
        let mut store = MetadataStore::new(4);
        let first = record(&[("year", json!(1999)), ("category", json!("c3"))]);
        let second = record(&[("category", json!("c4")), ("flag", json!(true))]);
        store.insert(0, first.clone());
        store.insert(2, second.clone());
        assert_eq!(store.get(0).unwrap().to_map(), first);
        assert_eq!(store.get(2).unwrap().to_map(), second);
        assert_eq!(store.get(2).unwrap().field("category"), Some(&json!("c4")));
        assert_eq!(store.get(2).unwrap().field("year"), None);
        assert_eq!(store.get(2).unwrap().field("absent"), None);
        assert!(store.get(1).is_none());
        assert!(store.get(3).is_none());
        assert_eq!(store.len(), 2);
        let slots: Vec<usize> = store.iter().map(|(slot, _)| slot).collect();
        assert_eq!(slots, vec![0, 2]);
    }

    /// An empty mapping is an entry with no fields, and an absent entry is
    /// not, since a filter judges the one and never the other.
    #[test]
    fn an_empty_mapping_is_held_and_an_absent_one_is_not() {
        let mut store = MetadataStore::new(2);
        store.insert(1, HashMap::new());
        assert!(store.get(0).is_none());
        let fields = store.get(1).expect("an empty mapping is an entry");
        assert!(fields.is_empty());
        assert_eq!(fields.to_map(), HashMap::new());
        assert_eq!(store.len(), 1);
        assert!(store.remove(1));
        assert!(!store.remove(1));
        assert!(!store.remove(7));
        assert!(store.get(1).is_none());
        assert_eq!(store.len(), 0);
    }

    /// Inserting at a slot that holds an entry replaces it whole.
    #[test]
    fn an_insert_replaces_the_whole_record() {
        let mut store = MetadataStore::new(1);
        store.insert(0, record(&[("a", json!(1)), ("b", json!(2))]));
        store.insert(0, record(&[("b", json!(3))]));
        assert_eq!(store.get(0).unwrap().to_map(), record(&[("b", json!(3))]));
        assert_eq!(store.len(), 1);
    }

    /// The fields come out in symbol order, which is the order their names
    /// were first seen by the store, so two records carrying the same names
    /// lay them out the same way.
    #[test]
    fn the_fields_iterate_in_symbol_order() {
        let mut store = MetadataStore::new(2);
        store.insert(
            0,
            record(&[("year", json!(1999)), ("category", json!("c3"))]),
        );
        store.insert(
            1,
            record(&[("category", json!("c4")), ("year", json!(2000))]),
        );
        let first: Vec<&str> = store.get(0).unwrap().iter().map(|(name, _)| name).collect();
        let second: Vec<&str> = store.get(1).unwrap().iter().map(|(name, _)| name).collect();
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);

        // The symbols rise, and one symbol is one name in both records.
        let entries: Vec<(u32, &str)> = store
            .get(0)
            .unwrap()
            .entries()
            .map(|(symbol, name, _)| (symbol, name))
            .collect();
        assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(
            entries.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
            first
        );
        let again: Vec<(u32, &str)> = store
            .get(1)
            .unwrap()
            .entries()
            .map(|(symbol, name, _)| (symbol, name))
            .collect();
        assert_eq!(entries, again);
    }

    /// A record without fields costs its entry alone, and a record with
    /// fields costs one block of forty bytes a field beside its text.
    #[test]
    fn the_report_prices_an_entry_and_a_block() {
        let mut store = MetadataStore::new(7);
        let empty = store.heap_bytes();
        assert_eq!(empty, 8 * 16, "seven declared records reserve eight slots");
        store.insert(0, HashMap::new());
        assert_eq!(
            store.heap_bytes(),
            empty,
            "an empty mapping allocates nothing"
        );
        store.insert(
            1,
            record(&[("category", json!("c3")), ("year", json!(1999))]),
        );
        let names = 2 * std::mem::size_of::<String>()
            + "category".len()
            + "year".len()
            + (store.names.capacity() - 2) * std::mem::size_of::<String>();
        assert_eq!(store.heap_bytes(), empty + 2 * 40 + "c3".len() + names);
        store.clear(7);
        assert_eq!(store.heap_bytes(), empty);
        assert!(store.is_empty());
    }

    /// A store whose records sit under ids spread far apart costs what its
    /// records cost, being a page's header and an entry a record beside the
    /// table, rather than sixteen bytes for every id below the largest, and
    /// reads back every record as a store over dense ids does.
    #[test]
    fn a_store_over_scattered_ids_costs_what_its_records_cost() {
        let slots: Vec<usize> = (0..300usize).map(|k| 3 + k * 70_001).collect();
        let largest = *slots.last().unwrap();
        let mut store = MetadataStore::new(16);
        store.plan(slots.len(), largest);
        for &slot in slots.iter().rev() {
            store.insert(slot, record(&[("n", json!(slot))]));
        }
        store.settle();
        assert!(!store.records.is_flat());
        let pages = largest / crate::idmap::PAGE_IDS + 1;
        let blocks = 300 * 40;
        let bound = 4 * pages + 300 * 128 + 300 * (2 + 16) + blocks + 64;
        let entries = store.records.heap_bytes();
        assert!(entries < bound, "{entries} >= {bound}");
        assert_eq!(store.len(), 300);
        for &slot in &slots {
            assert_eq!(store.get(slot).unwrap().field("n"), Some(&json!(slot)));
            assert!(store.get(slot + 1).is_none());
        }
        let walked: Vec<usize> = store.iter().map(|(slot, _)| slot).collect();
        assert_eq!(walked, slots);
        assert!(store.remove(slots[7]));
        assert!(!store.remove(slots[7]));
        assert_eq!(store.len(), 299);
    }
}
