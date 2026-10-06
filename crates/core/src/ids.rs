//! Every record's external id, held once.
//!
//! The text of every id sits end to end in one arena. An entry per internal
//! id names where its text starts and how long it is, so resolving a node to
//! its id is two array reads and a slice. A forward table finds the internal
//! id of a name: open addressing over `u32` buckets, each holding an internal
//! id, with one byte of the hash beside each bucket so a probe compares the
//! text only where the byte agrees. Beside them is the live set as bits,
//! which is what a filtered search checks its selection against.
//!
//! This replaced two hash maps, one from the id to the internal id and one
//! back, each holding its own copy of the text in its own heap block and a
//! 32 byte bucket in a power of two table. The forward table here holds a
//! five byte bucket, the reverse map is the entry vector, and the text is one
//! arena with no block per id. Every reader of the two maps already held the
//! internal id or the name, so nothing needed both maps.
//!
//! **Bucket counts are hashbrown's.** The table grows when an insertion would
//! take it past seven eighths full, to the next power of two, which is the
//! rule the standard map follows, so a table over a given record count holds
//! the bucket count the map held and the saving is the bucket's width. A miss
//! probes until it meets an empty bucket, and at seven eighths full that is a
//! few dozen sequential tag bytes.
//!
//! The hasher is the standard library's, seeded afresh for every store, so a
//! caller's ids hash as they did under the two maps.
//!
//! Internal ids are never reused and `compact` re-inserts every record under
//! the id it already holds, so the ids the store holds spread over a range
//! far larger than its records. The entries sit in an [`IdMap`], flat while
//! the ids are dense, which is the vector indexed by id the store always
//! held, and in pages once they are sparse, so the entries cost what the
//! records cost however far the ids spread. A removed record's text stays
//! in the arena until [`IdStore::compact_text`] rewrites the arena without
//! it, which the collection runs when it rebuilds the graph. The entries are
//! reserved for the declared record count, under a cap, and grow by doubling
//! past it, which is the rule the metadata store and the graph's per node
//! arrays follow.

use crate::bitmap::Bitmap;
use crate::error::Error;
use crate::idmap::{lookup_ratio, IdMap};
use std::hash::{BuildHasher, RandomState};

/// Bits of an entry that hold the text's offset. Sixty-four gibibytes of id
/// text, which no index a `u32` node index can hold reaches.
const OFFSET_BITS: u32 = 36;
/// Bits of an entry that hold the text's length. An id of 256 mebibytes.
const LEN_BITS: u32 = 28;
/// The arena's ceiling in bytes, two short of what the offset can name so
/// that no entry is ever all ones.
const MAX_TEXT: u64 = (1u64 << OFFSET_BITS) - 2;
/// The longest id an entry can describe.
const MAX_ID_LEN: u64 = (1u64 << LEN_BITS) - 1;
/// An entry for a slot that holds no record.
const ABSENT: u64 = u64::MAX;
/// A bucket that holds no internal id.
const EMPTY: u32 = u32::MAX;

/// Ids of its extent the entries hold flat for each record: the ratio of a
/// map a search looks up, for their eight-byte entries.
const ENTRY_RATIO: usize = lookup_ratio(std::mem::size_of::<u64>());
/// The last internal id the store holds. A slot is a `u32` and the store
/// keeps `u32::MAX` for the empty bucket, so the last id is the one below it.
/// `add` issues no id past this one and a replay installs none, so neither
/// hands the store an id it refuses.
pub const MAX_INTERNAL_ID: usize = EMPTY as usize - 1;
/// The most entries and the most buckets the store reserves at creation,
/// whatever the declaration. Eight bytes an entry and five a bucket.
const RESERVE_CAP: usize = 1 << 20;

/// Buckets a table holds for `records` entries at seven eighths full, which
/// is the count the standard map holds for the same entries.
fn buckets_for(records: usize) -> usize {
    if records == 0 {
        return 0;
    }
    (records.saturating_mul(8))
        .div_ceil(7)
        .next_power_of_two()
        .max(8)
}

#[inline]
fn pack(offset: usize, len: usize) -> u64 {
    (offset as u64) | ((len as u64) << OFFSET_BITS)
}

#[inline]
fn unpack(entry: u64) -> (usize, usize) {
    let offset = entry & ((1u64 << OFFSET_BITS) - 1);
    let len = entry >> OFFSET_BITS;
    (offset as usize, len as usize)
}

#[inline]
fn tag_of(hash: u64) -> u8 {
    (hash >> 56) as u8
}

/// Every record's external id, held once, with the live set as bits.
pub struct IdStore {
    /// The text of every id, end to end. Each id was appended whole, so
    /// every offset an entry names is a character boundary.
    text: String,
    /// One entry per record, by internal id: the offset and the length of
    /// its text. [`ABSENT`] is the map's absent value. It stays flat to
    /// [`ENTRY_RATIO`] ids a record.
    entries: IdMap<u64, ENTRY_RATIO>,
    /// The forward table, a power of two of buckets each holding an internal
    /// id or [`EMPTY`], probed linearly.
    buckets: Vec<u32>,
    /// One byte of the hash per bucket, read before the text is.
    tags: Vec<u8>,
    hasher: RandomState,
    /// The set of internal ids that hold a record.
    live: Bitmap,
    /// How many entries hold a record.
    held: usize,
    /// Bytes of the arena that belong to removed records.
    dead_text: usize,
}

impl IdStore {
    /// An empty store reserved for `expected_size` records, under the cap.
    ///
    /// One entry more than the declaration, because internal ids are issued
    /// from one and the entry is the id, so a declaration filled exactly
    /// reaches entry `expected_size` and would double the vector for it.
    pub fn new(expected_size: usize) -> Self {
        let entries = expected_size.saturating_add(1).min(RESERVE_CAP);
        let buckets = buckets_for(expected_size.min(RESERVE_CAP));
        IdStore {
            text: String::new(),
            entries: IdMap::with_capacity(entries),
            buckets: vec![EMPTY; buckets],
            tags: vec![0; buckets],
            hasher: RandomState::new(),
            live: Bitmap::default(),
            held: 0,
            dead_text: 0,
        }
    }

    /// Make room for `records` more entries and for the slot `highest`
    /// before a run of insertions, which is what the loader knows in
    /// advance. The entries and the live set take the form the run's final
    /// shape gives, flat with room up to `highest` where the ids are dense
    /// and paged where they are not, whatever order the run comes in.
    pub fn reserve(&mut self, records: usize, highest: usize) {
        self.entries.reserve(records, highest);
        self.live.plan(records, highest);
        let wanted = buckets_for(self.held.saturating_add(records));
        if wanted > self.buckets.len() {
            self.rehash(wanted);
        }
    }

    /// Settle the entries and the live set once a load has written every
    /// record, so a paged store holds the same bytes whatever order the
    /// records came in. A flat store is left as it is.
    pub fn settle(&mut self) {
        self.entries.settle();
        self.live.settle();
    }

    /// Whether the entries are flat, one slot per internal id up to the
    /// largest, rather than paged; see [`IdMap`]. The live set reports its
    /// own form through [`IdStore::live`].
    pub fn is_flat(&self) -> bool {
        self.entries.is_flat()
    }

    /// The entry at a slot a bucket names, which holds a record.
    #[inline]
    fn entry(&self, slot: usize) -> u64 {
        *self
            .entries
            .get(slot)
            .expect("every bucket names a slot that holds a record")
    }

    #[inline]
    fn hash(&self, id: &str) -> u64 {
        self.hasher.hash_one(id)
    }

    /// The text an entry names. The entry is present.
    #[inline]
    fn text_of(&self, entry: u64) -> &str {
        let (offset, len) = unpack(entry);
        &self.text[offset..offset + len]
    }

    /// The bucket holding the internal id of `id`, if the store holds it.
    fn find(&self, id: &str) -> Option<usize> {
        if self.buckets.is_empty() {
            return None;
        }
        let hash = self.hash(id);
        let tag = tag_of(hash);
        let mask = self.buckets.len() - 1;
        let mut at = (hash as usize) & mask;
        loop {
            let slot = self.buckets[at];
            if slot == EMPTY {
                return None;
            }
            if self.tags[at] == tag && self.text_of(self.entry(slot as usize)) == id {
                return Some(at);
            }
            at = (at + 1) & mask;
        }
    }

    /// Put `slot` into the first empty bucket on its probe sequence. The
    /// table has room, which the caller ensured.
    fn place(buckets: &mut [u32], tags: &mut [u8], slot: u32, hash: u64) {
        let mask = buckets.len() - 1;
        let mut at = (hash as usize) & mask;
        while buckets[at] != EMPTY {
            at = (at + 1) & mask;
        }
        buckets[at] = slot;
        tags[at] = tag_of(hash);
    }

    /// Rebuild the table at `buckets` buckets from the live entries.
    fn rehash(&mut self, buckets: usize) {
        let mut new_buckets = vec![EMPTY; buckets];
        let mut new_tags = vec![0u8; buckets];
        for (slot, &entry) in self.entries.iter() {
            let hash = self.hash(self.text_of(entry));
            Self::place(&mut new_buckets, &mut new_tags, slot as u32, hash);
        }
        self.buckets = new_buckets;
        self.tags = new_tags;
    }

    /// Grow the table if one more entry would take it past seven eighths.
    fn grow_if_needed(&mut self) {
        let need = self.held + 1;
        if self.buckets.is_empty() || need > self.buckets.len() / 8 * 7 {
            self.rehash(buckets_for(need).max(self.buckets.len() * 2));
        }
    }

    /// Take the entry at bucket `at` out of the table, shifting the entries
    /// after it back where their probe sequences allow, so the table holds
    /// no tombstone and a probe still stops at the first empty bucket.
    fn erase_bucket(&mut self, mut at: usize) {
        let mask = self.buckets.len() - 1;
        let mut next = at;
        loop {
            next = (next + 1) & mask;
            let slot = self.buckets[next];
            if slot == EMPTY {
                break;
            }
            let ideal = (self.hash(self.text_of(self.entry(slot as usize))) as usize) & mask;
            // The entry at `next` may fill the hole at `at` unless its own
            // bucket lies cyclically in (at, next], where a probe for it
            // would still find it.
            let stays = if at <= next {
                at < ideal && ideal <= next
            } else {
                at < ideal || ideal <= next
            };
            if !stays {
                self.buckets[at] = slot;
                self.tags[at] = self.tags[next];
                at = next;
            }
        }
        self.buckets[at] = EMPTY;
    }

    /// Forget the entry at `slot`, which is present.
    fn forget(&mut self, slot: usize) {
        let entry = self
            .entries
            .remove(slot)
            .expect("a slot is forgotten only where it holds a record");
        let (_, len) = unpack(entry);
        self.dead_text += len;
        self.live.release(slot);
        self.held -= 1;
    }

    /// Append `id` to the arena, growing it by a quarter at a time rather
    /// than doubling, so the capacity nothing has written stays small.
    fn push_text(&mut self, id: &str) -> usize {
        let offset = self.text.len();
        let spare = self.text.capacity() - self.text.len();
        if spare < id.len() {
            let grow = id.len().max(self.text.capacity() / 4).max(1024);
            self.text.reserve_exact(grow);
        }
        self.text.push_str(id);
        offset
    }

    /// Record that `slot` resolves to `id` and `id` to `slot`.
    ///
    /// A name the store already holds is re-bound to `slot` and its old slot
    /// forgotten, and a slot that already holds a name is freed first, so a
    /// name and a slot always pair one to one. Neither happens on any path
    /// the collection takes, since every insertion checks the name first and
    /// no internal id is issued twice.
    ///
    /// Refused where the slot is past [`MAX_INTERNAL_ID`], or where the id
    /// or the arena would pass what an entry can name.
    pub fn insert(&mut self, slot: usize, id: &str) -> Result<(), Error> {
        let slot32 = u32::try_from(slot)
            .ok()
            .filter(|_| slot <= MAX_INTERNAL_ID)
            .ok_or_else(|| {
                Error::Engine(format!(
                    "internal id {} is above what the id store can hold",
                    slot
                ))
            })?;
        if id.len() as u64 > MAX_ID_LEN {
            return Err(Error::Engine(format!(
                "an id of {} bytes is longer than the id store can hold",
                id.len()
            )));
        }
        if self.text.len() as u64 + id.len() as u64 > MAX_TEXT {
            return Err(Error::Engine(
                "the id store's text arena is full".to_string(),
            ));
        }
        if let Some(at) = self.find(id) {
            let old = self.buckets[at] as usize;
            self.erase_bucket(at);
            self.forget(old);
        }
        if self.entries.contains(slot) {
            self.remove_slot(slot);
        }
        self.grow_if_needed();
        let offset = self.push_text(id);
        let entry = pack(offset, id.len());
        debug_assert_ne!(entry, ABSENT, "an entry is never all ones");
        self.entries.insert(slot, entry);
        let hash = self.hash(id);
        Self::place(&mut self.buckets, &mut self.tags, slot32, hash);
        self.live.hold(slot);
        self.held += 1;
        Ok(())
    }

    /// Forget the record named `id`, returning the internal id it held.
    pub fn remove_name(&mut self, id: &str) -> Option<usize> {
        let at = self.find(id)?;
        let slot = self.buckets[at] as usize;
        self.erase_bucket(at);
        self.forget(slot);
        Some(slot)
    }

    /// Forget the record at `slot`, reporting whether it held one.
    pub fn remove_slot(&mut self, slot: usize) -> bool {
        let Some(&entry) = self.entries.get(slot) else {
            return false;
        };
        let at = self
            .find(self.text_of(entry))
            .expect("every present entry is in the table");
        self.erase_bucket(at);
        self.forget(slot);
        true
    }

    /// Forget every record, keeping a reservation for `expected_size`.
    pub fn clear(&mut self, expected_size: usize) {
        *self = IdStore::new(expected_size);
    }

    /// The internal id of `id`, where the store holds it.
    #[inline]
    pub fn slot_of(&self, id: &str) -> Option<usize> {
        self.find(id).map(|at| self.buckets[at] as usize)
    }

    /// Whether the store holds a record named `id`.
    #[inline]
    pub fn contains_name(&self, id: &str) -> bool {
        self.find(id).is_some()
    }

    /// The external id of `slot`, where it holds a record.
    #[inline]
    pub fn name(&self, slot: usize) -> Option<&str> {
        let entry = *self.entries.get(slot)?;
        Some(self.text_of(entry))
    }

    /// Whether `slot` holds a record. Total over every `usize`.
    #[inline]
    pub fn contains_slot(&self, slot: usize) -> bool {
        self.live.contains(slot)
    }

    /// How many records the store holds.
    pub fn len(&self) -> usize {
        self.held
    }

    /// Whether the store holds no record.
    pub fn is_empty(&self) -> bool {
        self.held == 0
    }

    /// Every record as its internal id and its external id, in increasing
    /// internal id order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &str)> + '_ {
        self.entries
            .iter()
            .map(|(slot, &entry)| (slot, self.text_of(entry)))
    }

    /// Every record's internal id, in increasing order.
    pub fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().map(|(slot, _)| slot)
    }

    /// The largest internal id that holds a record.
    pub fn highest_slot(&self) -> Option<usize> {
        self.entries.highest()
    }

    /// The set of internal ids that hold a record, as bits.
    pub fn live(&self) -> &Bitmap {
        &self.live
    }

    /// Whether `other` admits every live record, by a word walk of the
    /// intersection against the live count. What decides that a filter's
    /// bitmap is no filter at all.
    pub fn admits_every_live(&self, other: &Bitmap) -> bool {
        other.count_and(&self.live) == self.held
    }

    /// Whether `other` holds exactly the ids this store holds.
    pub fn agrees_with(&self, other: &Bitmap) -> bool {
        if self.live.count() != other.count() {
            return false;
        }
        let mut agrees = true;
        self.live.for_each_while(|slot| {
            agrees = other.contains(slot);
            agrees
        });
        agrees
    }

    /// Rewrite the arena without the text of removed records, returning the
    /// bytes it let go of. Every entry keeps its internal id; only its offset
    /// moves. Costs one pass over the entries and one copy of the live text.
    pub fn compact_text(&mut self) -> usize {
        if self.dead_text == 0 {
            return 0;
        }
        let before = self.text.len();
        let mut text = String::with_capacity(before - self.dead_text);
        for (_, entry) in self.entries.iter_mut() {
            let (offset, len) = unpack(*entry);
            let start = text.len();
            text.push_str(&self.text[offset..offset + len]);
            *entry = pack(start, len);
        }
        self.text = text;
        self.dead_text = 0;
        before - self.text.len()
    }

    /// Bytes of the arena that belong to removed records.
    pub fn dead_text(&self) -> usize {
        self.dead_text
    }

    /// Bytes of id text the records the store holds carry, end to end.
    pub fn text_bytes(&self) -> usize {
        self.text.len() - self.dead_text
    }

    /// Bytes the store asked the allocator for: the arena at its capacity,
    /// the entries at theirs, the table's buckets and tags, and the live
    /// set.
    pub fn heap_bytes(&self) -> usize {
        self.text.capacity()
            + self.entries.heap_bytes()
            + self.buckets.capacity() * std::mem::size_of::<u32>()
            + self.tags.capacity()
            + self.live.heap_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idmap::PAGE_IDS;

    /// The entries stay flat while the store holds one record in eight ids or
    /// more past the floor, grown in increasing order and by a run's plan,
    /// and page out below that.
    #[test]
    fn the_entries_stay_flat_to_their_ratio() {
        assert_eq!(ENTRY_RATIO, 8);
        for (gap, flat) in [
            (ENTRY_RATIO - 1, true),
            (ENTRY_RATIO, true),
            (ENTRY_RATIO + 1, false),
        ] {
            let mut grown = IdStore::new(10);
            let mut planned = IdStore::new(10);
            planned.reserve(2_000, 1_999 * gap);
            for k in 0..2_000usize {
                grown.insert(k * gap, &format!("r{k}")).unwrap();
                planned.insert(k * gap, &format!("r{k}")).unwrap();
            }
            assert_eq!(grown.entries.is_flat(), flat, "grown, one id in {gap}");
            assert_eq!(planned.entries.is_flat(), flat, "planned, one id in {gap}");
            assert_eq!(grown.len(), 2_000);
        }
    }

    /// What goes in comes back both ways, the empty id included, and a
    /// name the store does not hold resolves to nothing.
    #[test]
    fn a_name_and_a_slot_resolve_to_each_other() {
        let mut store = IdStore::new(4);
        assert!(store.is_empty());
        assert_eq!(store.slot_of("a"), None);
        assert_eq!(store.name(1), None);
        store.insert(1, "a").unwrap();
        store.insert(2, "").unwrap();
        store.insert(9, "vec_9").unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.slot_of("a"), Some(1));
        assert_eq!(store.slot_of(""), Some(2));
        assert_eq!(store.slot_of("vec_9"), Some(9));
        assert_eq!(store.slot_of("vec_"), None);
        assert_eq!(store.name(1), Some("a"));
        assert_eq!(store.name(2), Some(""));
        assert_eq!(store.name(9), Some("vec_9"));
        assert_eq!(store.name(3), None);
        assert_eq!(store.name(10), None);
        assert!(store.contains_name("a") && !store.contains_name("b"));
        assert!(store.contains_slot(9) && !store.contains_slot(8));
        assert_eq!(store.highest_slot(), Some(9));
        let pairs: Vec<(usize, &str)> = store.iter().collect();
        assert_eq!(pairs, vec![(1, "a"), (2, ""), (9, "vec_9")]);
        let slots: Vec<usize> = store.slots().collect();
        assert_eq!(slots, vec![1, 2, 9]);
    }

    /// The table grows past the reservation and every name still resolves,
    /// which holds the probe, the growth and the rehash together.
    #[test]
    fn every_name_resolves_through_growth() {
        let mut store = IdStore::new(0);
        for slot in 1..=5000usize {
            store.insert(slot, &format!("record-{slot}")).unwrap();
        }
        assert_eq!(store.len(), 5000);
        for slot in 1..=5000usize {
            let name = format!("record-{slot}");
            assert_eq!(store.slot_of(&name), Some(slot), "{name}");
            assert_eq!(store.name(slot), Some(name.as_str()));
        }
        assert_eq!(store.slot_of("record-5001"), None);
        assert_eq!(store.slot_of("record-0"), None);
        assert!(store.buckets.len() * 7 / 8 >= 5000);
    }

    /// Removal shifts the entries after the hole back where their probe
    /// sequences allow, so no tombstone is left and every other name still
    /// resolves, whichever names went.
    #[test]
    fn removal_leaves_no_tombstone_and_the_rest_resolve() {
        let mut store = IdStore::new(2000);
        for slot in 1..=2000usize {
            store.insert(slot, &format!("{slot}")).unwrap();
        }
        let mut removed = 0;
        for slot in (1..=2000usize).step_by(3) {
            assert_eq!(store.remove_name(&format!("{slot}")), Some(slot));
            assert_eq!(store.remove_name(&format!("{slot}")), None);
            removed += 1;
        }
        assert_eq!(store.len(), 2000 - removed);
        for slot in 1..=2000usize {
            let expected = if (slot - 1) % 3 == 0 {
                None
            } else {
                Some(slot)
            };
            assert_eq!(store.slot_of(&format!("{slot}")), expected, "{slot}");
            assert_eq!(store.name(slot).is_some(), expected.is_some());
        }
        let occupied = store.buckets.iter().filter(|&&b| b != EMPTY).count();
        assert_eq!(
            occupied,
            store.len(),
            "a removed entry leaves nothing behind"
        );
        assert!(store.remove_slot(2));
        assert!(!store.remove_slot(2));
        assert!(!store.remove_slot(7000));
        assert_eq!(store.slot_of("2"), None);
    }

    /// A removed record's text stays until the arena is compacted, and the
    /// compaction moves every live id's text without changing its name or
    /// its internal id.
    #[test]
    fn compaction_reclaims_the_text_of_removed_records() {
        let mut store = IdStore::new(100);
        for slot in 1..=100usize {
            store.insert(slot, &format!("name-{slot:03}")).unwrap();
        }
        let full = store.text.len();
        for slot in 1..=50usize {
            store.remove_name(&format!("name-{slot:03}")).unwrap();
        }
        assert_eq!(
            store.text.len(),
            full,
            "removal leaves the text where it is"
        );
        assert_eq!(store.dead_text(), 50 * 8);
        assert_eq!(store.compact_text(), 50 * 8);
        assert_eq!(store.text.len(), full - 50 * 8);
        assert_eq!(store.dead_text(), 0);
        assert_eq!(store.compact_text(), 0);
        for slot in 51..=100usize {
            let name = format!("name-{slot:03}");
            assert_eq!(store.name(slot), Some(name.as_str()));
            assert_eq!(store.slot_of(&name), Some(slot));
        }
        store.insert(101, "after").unwrap();
        assert_eq!(store.slot_of("after"), Some(101));
        assert_eq!(store.name(101), Some("after"));
    }

    /// The bitmap admits exactly the slots that hold a record, after every
    /// kind of write, which is what makes a filter's selection comparable
    /// with the live set.
    #[test]
    fn the_live_set_agrees_with_the_entries_through_every_write() {
        let agrees = |store: &IdStore| {
            (0..256).all(|slot| store.live().contains(slot) == store.name(slot).is_some())
        };
        let mut store = IdStore::new(0);
        assert!(agrees(&store));
        for slot in [1usize, 2, 3, 64, 65, 130] {
            store.insert(slot, &format!("r{slot}")).unwrap();
        }
        assert!(agrees(&store));
        assert_eq!(store.live().count(), 6);
        assert_eq!(store.remove_name("r64"), Some(64));
        assert!(agrees(&store));
        assert!(!store.live().contains(64));
        let mut every = Bitmap::default();
        for slot in [1usize, 2, 3, 65, 130] {
            every.insert(slot);
        }
        assert!(store.agrees_with(&every));
        assert!(store.admits_every_live(&every));
        every.insert(64);
        assert!(!store.agrees_with(&every));
        assert!(store.admits_every_live(&every));
        every.remove(3);
        assert!(!store.admits_every_live(&every));
        store.clear(4);
        assert!(agrees(&store));
        assert!(store.is_empty());
        assert_eq!(store.live().count(), 0);
    }

    /// A name inserted a second time under a new slot pairs with the new
    /// slot alone, and a slot given a second name holds the second alone.
    #[test]
    fn a_name_and_a_slot_pair_one_to_one() {
        let mut store = IdStore::new(4);
        store.insert(1, "x").unwrap();
        store.insert(2, "x").unwrap();
        assert_eq!(store.slot_of("x"), Some(2));
        assert_eq!(store.name(1), None);
        assert_eq!(store.len(), 1);
        store.insert(2, "y").unwrap();
        assert_eq!(store.slot_of("x"), None);
        assert_eq!(store.slot_of("y"), Some(2));
        assert_eq!(store.len(), 1);
        let occupied = store.buckets.iter().filter(|&&b| b != EMPTY).count();
        assert_eq!(occupied, 1);
    }

    /// The report prices the arena, the entries, the buckets with their
    /// tags and the live words, and a declaration reserves one entry more
    /// than it names.
    #[test]
    fn the_report_prices_the_arena_the_entries_and_the_table() {
        let store = IdStore::new(7);
        assert_eq!(store.entries.heap_bytes(), 8 * 8);
        assert_eq!(store.buckets.len(), 8);
        assert_eq!(store.heap_bytes(), 8 * 8 + 8 * 4 + 8);
        let mut store = IdStore::new(100_000);
        assert_eq!(store.buckets.len(), 131_072);
        store.insert(1, "12345").unwrap();
        assert_eq!(
            store.heap_bytes(),
            store.text.capacity() + 100_001 * 8 + 131_072 * 5 + store.live.heap_bytes()
        );
        assert!(store.text.capacity() >= 5);
        assert_eq!(buckets_for(114_688), 131_072);
        assert_eq!(buckets_for(114_689), 262_144);
    }

    /// The reservation before a run of insertions sizes the table once.
    #[test]
    fn a_reservation_sizes_the_table_for_the_run() {
        let mut store = IdStore::new(0);
        store.reserve(1000, 1500);
        let buckets = store.buckets.len();
        assert!(buckets * 7 / 8 >= 1000);
        assert!(store.entries.is_flat());
        assert!(store.entries.heap_bytes() >= 1501 * 8);
        for slot in 1..=1000usize {
            store.insert(slot + 500, &format!("{slot}")).unwrap();
        }
        assert_eq!(store.buckets.len(), buckets, "no rehash inside the run");
        assert_eq!(store.slot_of("1000"), Some(1500));
    }

    /// The last internal id is the one below the empty bucket, and every
    /// slot past it is refused with the store left as it was.
    #[test]
    fn no_slot_past_the_last_internal_id_is_held() {
        assert_eq!(MAX_INTERNAL_ID, 4_294_967_294);
        let mut store = IdStore::new(4);
        store.insert(1, "held").unwrap();
        for slot in [MAX_INTERNAL_ID + 1, u32::MAX as usize + 1, usize::MAX] {
            let err = store.insert(slot, "past").unwrap_err();
            assert!(
                err.to_string()
                    .contains("is above what the id store can hold"),
                "{err}"
            );
        }
        assert_eq!(store.len(), 1);
        assert!(!store.contains_name("past"));
        assert_eq!(store.highest_slot(), Some(1));
        assert_eq!(
            store.entries.extent(),
            2,
            "no entry was made for a refused slot"
        );
    }

    /// A store whose records sit under ids spread far apart pages its
    /// entries and its live set out, costs what its records cost rather than
    /// a slot for every id below the largest, and answers every name, slot,
    /// walk and removal as a store over dense ids does.
    #[test]
    fn a_store_over_scattered_ids_costs_what_its_records_cost() {
        let slots: Vec<usize> = (0..500usize).map(|k| 7 + k * 40_009).collect();
        let largest = *slots.last().unwrap();
        let mut store = IdStore::new(16);
        store.reserve(slots.len(), largest);
        for &slot in &slots {
            store.insert(slot, &format!("r{slot}")).unwrap();
        }
        store.settle();
        assert!(!store.entries.is_flat());
        assert!(!store.live.is_flat());
        let pages = largest / PAGE_IDS + 1;
        // A table entry a page of the range, a header a page and two arrays a
        // page, against the 160 megabytes eight bytes an id below the largest
        // would cost.
        let bound = 4 * pages + 2 * pages * 128 + 500 * (2 + 8 + 2);
        let ids_bytes = store.entries.heap_bytes() + store.live.heap_bytes();
        assert!(ids_bytes < bound, "{ids_bytes} >= {bound}");
        for &slot in &slots {
            let name = format!("r{slot}");
            assert_eq!(store.slot_of(&name), Some(slot));
            assert_eq!(store.name(slot), Some(name.as_str()));
            assert!(store.contains_slot(slot));
            assert!(!store.contains_slot(slot + 1));
        }
        assert_eq!(store.highest_slot(), Some(largest));
        assert_eq!(store.slots().collect::<Vec<_>>(), slots);
        for &slot in slots.iter().step_by(2) {
            assert!(store.remove_slot(slot));
        }
        assert_eq!(store.len(), 250);
        assert_eq!(store.live().count(), 250);
        let kept: Vec<usize> = slots.iter().copied().skip(1).step_by(2).collect();
        assert_eq!(store.slots().collect::<Vec<_>>(), kept);
        assert!(store.compact_text() > 0);
        for &slot in &kept {
            assert_eq!(store.name(slot), Some(format!("r{slot}").as_str()));
        }
        store.insert(largest + 1, "after").unwrap();
        assert_eq!(store.slot_of("after"), Some(largest + 1));
        assert_eq!(store.highest_slot(), Some(largest + 1));
    }

    /// A store whose ids are dense keeps the flat form, so its entries and its
    /// live set are the vector and the words the store always held.
    #[test]
    fn a_store_over_dense_ids_keeps_its_flat_form() {
        let mut store = IdStore::new(1_000);
        for slot in 1..=20_000usize {
            store.insert(slot, &format!("r{slot}")).unwrap();
        }
        for slot in (1..=20_000usize).step_by(3) {
            store.remove_slot(slot);
        }
        assert!(store.entries.is_flat());
        assert!(store.live.is_flat());
        let mut entries: Vec<u64> = Vec::with_capacity(1_001);
        entries.resize(20_001, ABSENT);
        assert_eq!(
            store.entries.as_flat().map(<[u64]>::len),
            Some(entries.len())
        );
    }
}
