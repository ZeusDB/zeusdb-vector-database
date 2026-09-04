//! The per record metadata, held by internal id.
//!
//! One entry per internal id the collection has issued, and a record's fields
//! in one block behind it. A record that carries no metadata costs its entry
//! and nothing else, which is sixteen bytes, and a record that carries fields
//! costs one block of forty bytes a field beside the text of its string
//! values. Field names are interned once for the whole store, so a record
//! holds a four byte symbol for each name rather than its own copy of it.
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
//! the id it already holds, so the entry vector grows with the id counter and
//! a removed record leaves a sixteen byte hole until `clear`. The vector is
//! reserved for the declared record count, under a cap, and grows by doubling
//! past it, which is the rule the graph's per node arrays follow.

use crate::filter::FieldLookup;
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
    /// One slot per internal id issued. `None` is a record that holds no
    /// entry, which is a removed record or one no insertion has reached.
    /// `Some` of an empty block is a record inserted with no fields, which
    /// is distinct: a filter judges an empty mapping and never judges an
    /// absent one.
    records: Vec<Option<Box<[Field]>>>,
    /// How many slots hold an entry.
    held: usize,
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
            records: Vec::with_capacity(expected_size.saturating_add(1).min(RESERVE_CAP)),
            held: 0,
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
        if slot >= self.records.len() {
            self.records.resize_with(slot + 1, || None);
        }
        if self.records[slot].is_none() {
            self.held += 1;
        }
        self.records[slot] = Some(fields.into_boxed_slice());
    }

    /// Forget the record at `slot`, reporting whether it held an entry.
    pub fn remove(&mut self, slot: usize) -> bool {
        match self.records.get_mut(slot) {
            Some(entry) if entry.is_some() => {
                *entry = None;
                self.held -= 1;
                true
            }
            _ => false,
        }
    }

    /// Forget every record and every field name, keeping a reservation for
    /// `expected_size` records.
    pub fn clear(&mut self, expected_size: usize) {
        *self = MetadataStore::new(expected_size);
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
        self.held
    }

    /// Whether no record holds an entry.
    pub fn is_empty(&self) -> bool {
        self.held == 0
    }

    /// Every record holding an entry, in increasing internal id order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, RecordFields<'_>)> + '_ {
        self.records.iter().enumerate().filter_map(|(slot, entry)| {
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
    /// The entry vector at capacity, every record's block at its length, the
    /// text of every string value and the name list with its text. A `Value`
    /// is thirty-two bytes wherever it sits and a string is the one variant
    /// that also owns text.
    pub fn heap_bytes(&self) -> usize {
        let entries = self.records.capacity() * std::mem::size_of::<Option<Box<[Field]>>>();
        let blocks: usize = self
            .records
            .iter()
            .filter_map(|entry| entry.as_deref())
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
}
