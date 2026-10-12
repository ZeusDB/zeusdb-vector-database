//! The per record metadata, held by position.
//!
//! One entry per position assigned since the last close-up, and a record's
//! fields in one block behind it. A record that carries no metadata costs its entry
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
//! the record's position or reached the forward map first, so nothing needed
//! the external key.
//!
//! **What comes back is the same mapping.** [`RecordFields::to_map`] rebuilds
//! the `HashMap<String, Value>` a caller handed in, and [`FieldLookup`] lets a
//! filter read a field without rebuilding anything. `metadata.json` is written
//! from and read into the same shape it always was.
//!
//! A record takes the next position when it is inserted, so the entry vector
//! grows with the records inserted since the last close-up and a removed
//! record leaves a sixteen byte hole until the next one, which
//! [`MetadataStore::closed_up`] builds off to the side and
//! [`MetadataStore::swap_parts`] puts in. The vector is reserved for the
//! declared record count, under a cap, and grows by doubling past it, which
//! is the rule the graph's per node arrays follow.

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
#[derive(Clone)]
struct Field {
    key: u32,
    value: Value,
}

/// Every record's metadata, indexed by position.
pub struct MetadataStore {
    /// Field names in the order first seen, indexed by symbol.
    names: Vec<String>,
    /// Field name to symbol.
    symbols: HashMap<String, u32>,
    /// One slot per position. `None` is a position that holds no entry,
    /// which is a hole a removal left or one no insertion has reached.
    /// `Some` of an empty block is a record inserted with no fields, which
    /// is distinct: a filter judges an empty mapping and never judges an
    /// absent one.
    records: Vec<Option<Box<[Field]>>>,
    /// How many slots hold an entry.
    held: usize,
}

/// What a close-up replaces in a metadata store, built off to the side by
/// [`MetadataStore::closed_up`] and swapped in by
/// [`MetadataStore::swap_parts`]. The swap trades the parts with the store's
/// own, so once it has run the value holds the store's old entries, for the
/// caller to drop with no guard held.
pub struct MetadataParts {
    records: Vec<Option<Box<[Field]>>>,
}

impl MetadataStore {
    /// An empty store reserved for `expected_size` records, under the cap.
    ///
    /// One slot more than the declaration, which is the reservation the
    /// store took while a slot was an internal id issued from one, kept so
    /// that a declaration costs what it always cost.
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

    /// Move every record held to the front, keeping their order, so the
    /// record at `old_of_new[k]` is at `k` and the store reaches no further
    /// than the records it holds. `old_of_new` lists the positions held in
    /// increasing order. Each block moves where it is; nothing is copied.
    /// The vector then gives back what it holds past the records or the
    /// reservation for `expected_size`, whichever is more.
    ///
    /// The close-up a collection runs is [`MetadataStore::closed_up`] and
    /// [`MetadataStore::swap_parts`]. This is the same pass made in place,
    /// kept for the tests that hold the two to the same result.
    #[cfg(any(test, feature = "test-support"))]
    pub fn close_up(&mut self, old_of_new: &[u32], expected_size: usize) {
        for (k, &old) in old_of_new.iter().enumerate() {
            // A record past the slots is one no insertion reached, which
            // holds no entry, and nothing is past it either.
            if k >= self.records.len() {
                break;
            }
            let entry = self.records.get_mut(old as usize).and_then(Option::take);
            self.records[k] = entry;
        }
        self.records.truncate(old_of_new.len());
        self.records.shrink_to(
            old_of_new
                .len()
                .max(expected_size.saturating_add(1).min(RESERVE_CAP)),
        );
    }

    /// The entries [`MetadataStore::close_up`] leaves, built from this store
    /// and leaving it as it is, so a reader goes on reading the store while
    /// they are made.
    ///
    /// The entry at `old_of_new[k]` is copied to `k`, block and all, as far
    /// as the store reaches, and a record past the slots holds no entry.
    /// The vector takes the capacity the pass in place leaves it: what it
    /// holds, or the records or the reservation for `expected_size` where
    /// that is less. Both copies of every block exist until the caller
    /// drops the parts the swap hands back.
    pub fn closed_up(&self, old_of_new: &[u32], expected_size: usize) -> MetadataParts {
        let held = old_of_new.len();
        let kept = held.max(expected_size.saturating_add(1).min(RESERVE_CAP));
        let mut records = Vec::with_capacity(self.records.capacity().min(kept));
        records.extend(
            old_of_new[..held.min(self.records.len())]
                .iter()
                .map(|&old| self.records.get(old as usize).cloned().flatten()),
        );
        MetadataParts { records }
    }

    /// Trade this store's entries for `parts`, which
    /// [`MetadataStore::closed_up`] built from it, so the store holds the
    /// closed-up entries and `parts` the store's old ones. A move alone.
    pub fn swap_parts(&mut self, parts: &mut MetadataParts) {
        std::mem::swap(&mut self.records, &mut parts.records);
    }

    /// Every part of the store, each with its capacity, for a test that
    /// holds two stores to be the same entries. Absent from every build
    /// that is not a test.
    #[cfg(any(test, feature = "test-support"))]
    pub fn layout(&self) -> Vec<(&'static str, usize, String)> {
        let records: Vec<Option<Vec<(u32, String)>>> = self
            .records
            .iter()
            .map(|entry| {
                entry.as_deref().map(|fields| {
                    fields
                        .iter()
                        .map(|field| (field.key, field.value.to_string()))
                        .collect()
                })
            })
            .collect();
        vec![
            ("records", self.records.capacity(), format!("{records:?}")),
            ("names", self.names.capacity(), format!("{:?}", self.names)),
            ("held", 0, self.held.to_string()),
        ]
    }

    /// One past the last position the store holds a slot for.
    pub fn extent(&self) -> usize {
        self.records.len()
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

    /// Every record holding an entry, in increasing position, which is
    /// internal id order.
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

    /// A store most of whose records were removed keeps them at scattered
    /// positions until the close-up, which moves every record's block to the
    /// front in its order and leaves the entries costing what the records
    /// cost, sixteen bytes a record beside its block, rather than sixteen
    /// for every position the store reached. Every record reads back at its
    /// new position, the walk visits them in order and a removal answers as
    /// before.
    #[test]
    fn a_store_over_scattered_ids_costs_what_its_records_cost() {
        let total = 21_000usize;
        let mut store = MetadataStore::new(16);
        for slot in 0..total {
            store.insert(slot, record(&[("n", json!(slot))]));
        }
        let kept: Vec<usize> = (0..total).filter(|slot| slot % 70 == 3).collect();
        for slot in (0..total).filter(|slot| slot % 70 != 3) {
            assert!(store.remove(slot));
        }
        assert_eq!(store.len(), kept.len());
        assert_eq!(store.extent(), total, "a removal leaves a hole");

        let old_of_new: Vec<u32> = kept.iter().map(|&slot| slot as u32).collect();
        store.close_up(&old_of_new, 16);
        assert_eq!(store.extent(), kept.len());
        assert_eq!(store.records.capacity(), kept.len());
        let blocks = kept.len() * 40;
        let names = store.names.capacity() * std::mem::size_of::<String>() + "n".len();
        assert_eq!(store.heap_bytes(), kept.len() * 16 + blocks + names);

        for (k, &old) in kept.iter().enumerate() {
            assert_eq!(store.get(k).unwrap().field("n"), Some(&json!(old)));
        }
        assert!(store.get(kept.len()).is_none());
        let walked: Vec<usize> = store.iter().map(|(slot, _)| slot).collect();
        assert_eq!(walked, (0..kept.len()).collect::<Vec<_>>());
        assert!(store.remove(7));
        assert!(!store.remove(7));
        assert_eq!(store.len(), kept.len() - 1);
    }

    /// A close-up keeps the reservation the declaration made, and a record
    /// past the last one any insertion reached holds no entry to move.
    #[test]
    fn a_close_up_keeps_the_declared_reservation() {
        let mut store = MetadataStore::new(100);
        for slot in 0..3_000usize {
            store.insert(slot, record(&[("n", json!(slot))]));
        }
        for slot in 5..3_000usize {
            assert!(store.remove(slot));
        }
        store.close_up(&[0, 1, 2, 3, 4], 100);
        assert_eq!(store.extent(), 5);
        assert_eq!(store.records.capacity(), 101);
        assert_eq!(store.get(4).unwrap().field("n"), Some(&json!(4)));
    }
}
