//! Every record's external id, held once.
//!
//! The text of every id sits end to end in one arena. An entry per position
//! names where its text starts and how long it is, so resolving a record to
//! its id is two array reads and a slice. A forward table finds the position
//! of a name: open addressing over `u32` buckets, each holding a position,
//! with one byte of the hash beside each bucket so a probe compares the text
//! only where the byte agrees. Beside them is the live set as bits, which is
//! what a filtered search checks its selection against.
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
//! A record takes the next position when it is inserted, and positions follow
//! internal id order, so the entry vector grows with the records inserted
//! since the last close-up and a removed record leaves an eight byte hole
//! until the next one. [`IdStore::closed_up`] builds, in one pass and with
//! the store left as it is, the entries with every record moved to the front,
//! the table with its positions rewritten through the same map and an arena
//! without the text of removed records, and [`IdStore::swap_parts`] puts
//! them in. Until then a removed record's text stays in the arena. The
//! entries give back what they hold past the records or the reservation. The
//! vector is reserved for the declared record count, under a cap, and grows
//! by doubling past it, which is the rule the metadata store and the graph's
//! per node arrays follow.

use crate::bitmap::Bitmap;
use crate::error::Error;
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
/// A bucket that holds no position.
const EMPTY: u32 = u32::MAX;
/// The last internal id a collection issues. A slot is a `u32` and the store
/// keeps `u32::MAX` for the empty bucket, so the last slot is the one below
/// it, and an internal id is held to the same ceiling so that every number
/// naming a record fits the same width. `add` issues no id past this one, a
/// replay installs none, and a load refuses one, whether the mappings name it
/// or the rebuild fallback would issue it. A record's position is below its
/// internal id, so the store is never handed a slot it refuses.
pub const MAX_INTERNAL_ID: usize = EMPTY as usize - 1;
/// The most entries and the most buckets the store reserves at creation,
/// whatever the declaration. Eight bytes an entry and five a bucket.
const RESERVE_CAP: usize = 1 << 20;

/// Entries the store reserves for `expected_size` records: one more than the
/// declaration, under the cap.
fn reserved_entries(expected_size: usize) -> usize {
    expected_size.saturating_add(1).min(RESERVE_CAP)
}

// The longest id and the largest arena this thread's checks hold a record
// to, which a test lowers to reach the two refusals without gigabytes of
// text. Nothing outside the tests sees it.
#[cfg(any(test, feature = "test-support"))]
thread_local! {
    pub static ID_CEILINGS: std::cell::Cell<(u64, u64)> =
        const { std::cell::Cell::new((MAX_ID_LEN, MAX_TEXT)) };
}

/// The longest id and the largest arena a store takes.
#[cfg(any(test, feature = "test-support"))]
fn ceilings() -> (u64, u64) {
    ID_CEILINGS.with(std::cell::Cell::get)
}

/// The longest id and the largest arena a store takes.
#[cfg(not(any(test, feature = "test-support")))]
fn ceilings() -> (u64, u64) {
    (MAX_ID_LEN, MAX_TEXT)
}

/// Refuse `id` where a store whose arena holds `arena` bytes could not take
/// it as its next record: an id longer than an entry can describe, or one
/// that would take the arena past what an entry can name. The words are
/// the ones [`IdStore::insert`] gives. A collection runs this where it
/// admits a record, before anything is issued or recorded for it, with the
/// text of the records it has admitted and not yet named counted in
/// `arena`.
pub fn refuse_unnamable(id: &str, arena: usize) -> Result<(), Error> {
    let (longest, largest) = ceilings();
    if id.len() as u64 > longest {
        return Err(Error::Engine(format!(
            "an id of {} bytes is longer than the id store can hold",
            id.len()
        )));
    }
    if arena as u64 + id.len() as u64 > largest {
        return Err(Error::Engine(
            "the id store's text arena is full".to_string(),
        ));
    }
    Ok(())
}

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
    /// One entry per position assigned since the last close-up: the offset
    /// and the length of its text, or [`ABSENT`] at a hole. Its length is the
    /// position the next record takes.
    entries: Vec<u64>,
    /// The forward table, a power of two of buckets each holding a position
    /// or [`EMPTY`], probed linearly.
    buckets: Vec<u32>,
    /// One byte of the hash per bucket, read before the text is.
    tags: Vec<u8>,
    hasher: RandomState,
    /// The set of positions that hold a record.
    live: Bitmap,
    /// How many entries hold a record.
    held: usize,
    /// Bytes of the arena that belong to removed records.
    dead_text: usize,
}

/// What a close-up replaces in an id store, built off to the side by
/// [`IdStore::closed_up`] and swapped in by [`IdStore::swap_parts`]. The
/// swap trades the parts with the store's own, so once it has run the value
/// holds the store's old parts, for the caller to drop with no guard held.
/// The table's tags, the hasher and the record count do not change at a
/// close-up and are not among them.
pub struct IdParts {
    text: String,
    entries: Vec<u64>,
    buckets: Vec<u32>,
    live: Bitmap,
    dead_text: usize,
}

impl IdParts {
    /// Bytes of the arena that belong to removed records. Once swapped in,
    /// these are the store's old parts, and the figure is the id text the
    /// close-up let go of, which [`IdStore::close_up`] returns.
    pub fn removed_text(&self) -> usize {
        self.dead_text
    }
}

impl IdStore {
    /// An empty store reserved for `expected_size` records, under the cap.
    ///
    /// One entry more than the declaration, which is the reservation the
    /// store took while an entry was an internal id issued from one, kept so
    /// that a declaration costs what it always cost.
    pub fn new(expected_size: usize) -> Self {
        let entries = reserved_entries(expected_size);
        let buckets = buckets_for(expected_size.min(RESERVE_CAP));
        IdStore {
            text: String::new(),
            entries: Vec::with_capacity(entries),
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
    /// advance.
    ///
    /// The loader files the records at consecutive positions from zero, so
    /// `highest` is one below the record count.
    pub fn reserve(&mut self, records: usize, highest: usize) {
        let entries = highest.saturating_add(1);
        if entries > self.entries.len() {
            self.entries.reserve(entries - self.entries.len());
        }
        let wanted = buckets_for(self.held.saturating_add(records));
        if wanted > self.buckets.len() {
            self.rehash(wanted);
        }
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
            if self.tags[at] == tag && self.text_of(self.entries[slot as usize]) == id {
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
        for (slot, &entry) in self.entries.iter().enumerate() {
            if entry == ABSENT {
                continue;
            }
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
            let ideal = (self.hash(self.text_of(self.entries[slot as usize])) as usize) & mask;
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
        let (_, len) = unpack(self.entries[slot]);
        self.entries[slot] = ABSENT;
        self.dead_text += len;
        self.live.remove(slot);
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
    /// takes the next position, being [`IdStore::extent`].
    ///
    /// Refused where the slot is past [`MAX_INTERNAL_ID`], or where the id
    /// or the arena would pass what an entry can name.
    pub fn insert(&mut self, slot: usize, id: &str) -> Result<(), Error> {
        if u32::try_from(slot)
            .ok()
            .filter(|_| slot <= MAX_INTERNAL_ID)
            .is_none()
        {
            return Err(Error::Engine(format!(
                "internal id {} is above what the id store can hold",
                slot
            )));
        }
        refuse_unnamable(id, self.text.len())?;
        self.insert_admitted(slot, id);
        Ok(())
    }

    /// The text arena's length, the text of removed records included, which
    /// is what [`refuse_unnamable`] holds a record's id against.
    pub fn arena_len(&self) -> usize {
        self.text.len()
    }

    /// [`IdStore::insert`] for a record already held to the three refusals
    /// it runs, being the slot's ceiling, the id's length and the arena's.
    /// The collection's insertion path calls this, having run them where a
    /// record is admitted, where a refusal moves nothing. A debug build
    /// holds the three as an assertion.
    pub fn insert_admitted(&mut self, slot: usize, id: &str) {
        debug_assert!(
            slot <= MAX_INTERNAL_ID,
            "slot {slot} is above what the id store can hold"
        );
        debug_assert!(
            refuse_unnamable(id, self.text.len()).is_ok(),
            "the record was held to the id's and the arena's ceilings"
        );
        let slot32 = slot as u32;
        if let Some(at) = self.find(id) {
            let old = self.buckets[at] as usize;
            self.erase_bucket(at);
            self.forget(old);
        }
        if self.entries.get(slot).is_some_and(|&entry| entry != ABSENT) {
            self.remove_slot(slot);
        }
        self.grow_if_needed();
        let offset = self.push_text(id);
        if slot >= self.entries.len() {
            self.entries.resize(slot + 1, ABSENT);
        }
        self.entries[slot] = pack(offset, id.len());
        let hash = self.hash(id);
        Self::place(&mut self.buckets, &mut self.tags, slot32, hash);
        self.live.insert(slot);
        self.held += 1;
    }

    /// Forget the record named `id`, returning the position it held, which
    /// becomes a hole.
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
        if entry == ABSENT {
            return false;
        }
        let at = self
            .find(self.text_of(entry))
            .expect("every present entry is in the table");
        self.erase_bucket(at);
        self.forget(slot);
        true
    }

    /// Forget the record the last insertion filed at `slot`, being the
    /// position before [`IdStore::extent`], and take the position back, so
    /// the next insertion files its record there. What an insertion does
    /// when the index refuses the record it has just named, so a refusal
    /// leaves no hole. Its text stays in the arena as removed text, as a
    /// removal's does.
    pub fn retract(&mut self, slot: usize) {
        if self.remove_slot(slot) && slot + 1 == self.entries.len() {
            self.entries.truncate(slot);
        }
    }

    /// Forget every record, keeping a reservation for `expected_size`. What
    /// the store held is let go before the reservation is made, so a clear
    /// never holds both.
    pub fn clear(&mut self, expected_size: usize) {
        self.text = String::new();
        self.entries = Vec::new();
        self.buckets = Vec::new();
        self.tags = Vec::new();
        self.live = Bitmap::default();
        *self = IdStore::new(expected_size);
    }

    /// The position the next record takes, being one past the last position
    /// assigned since the last close-up, holes included.
    #[inline]
    pub fn extent(&self) -> usize {
        self.entries.len()
    }

    /// Move every record held to the front, keeping their order, so the
    /// record at `old_of_new[k]` is at `k` and the store reaches no further
    /// than the records it holds.
    ///
    /// `old_of_new` lists the positions held in increasing order and
    /// `new_of_old` is its inverse, each hole there holding
    /// [`crate::Position::NONE`]. Each entry moves with its text where it
    /// is, each bucket's position is rewritten through `new_of_old`, and the
    /// live set becomes the positions below the record count. No name is
    /// hashed again, since a bucket's place follows its text and not its
    /// position. The entries give back what they hold past the records or
    /// the reservation for `expected_size`, whichever is more, and the arena
    /// is rewritten without the removed text, which returns the bytes it let
    /// go of; see [`IdStore::compact_text`].
    ///
    /// The close-up a collection runs is [`IdStore::closed_up`] and
    /// [`IdStore::swap_parts`]. This is the same pass made in place, kept
    /// for the tests that hold the two to the same result.
    #[cfg(any(test, feature = "test-support"))]
    pub fn close_up(
        &mut self,
        old_of_new: &[u32],
        new_of_old: &[u32],
        expected_size: usize,
    ) -> usize {
        debug_assert_eq!(old_of_new.len(), self.held);
        for (k, &old) in old_of_new.iter().enumerate() {
            self.entries[k] = self.entries[old as usize];
        }
        self.entries.truncate(old_of_new.len());
        self.entries
            .shrink_to(old_of_new.len().max(reserved_entries(expected_size)));
        for bucket in self.buckets.iter_mut() {
            if *bucket != EMPTY {
                *bucket = new_of_old[*bucket as usize];
            }
        }
        self.live.hold_first(old_of_new.len(), 0);
        self.compact_text()
    }

    /// The parts [`IdStore::close_up`] leaves, built from this store and
    /// leaving it as it is, so a reader goes on reading the store while they
    /// are made.
    ///
    /// The entry at `old_of_new[k]` goes to `k`. Where the arena holds
    /// removed text, each record's text is copied into a new arena without
    /// it and its entry names the new offset; otherwise the arena is copied
    /// as it is. Each bucket's position is rewritten through `new_of_old`,
    /// and the live set is the positions below the record count. Each part
    /// takes the capacity the pass in place leaves it: the entries keep what
    /// they hold, or the records or the reservation for `expected_size`
    /// where that is less, a rewritten arena holds the live text and no
    /// more, and the buckets and an arena copied as it is keep what they
    /// hold.
    pub fn closed_up(
        &self,
        old_of_new: &[u32],
        new_of_old: &[u32],
        expected_size: usize,
    ) -> IdParts {
        debug_assert_eq!(old_of_new.len(), self.held);
        let held = old_of_new.len();
        let kept = held.max(reserved_entries(expected_size));
        let mut entries = Vec::with_capacity(self.entries.capacity().min(kept));
        let text = if self.dead_text == 0 {
            entries.extend(old_of_new.iter().map(|&old| self.entries[old as usize]));
            let mut text = String::with_capacity(self.text.capacity());
            text.push_str(&self.text);
            text
        } else {
            let mut text = String::with_capacity(self.text.len() - self.dead_text);
            for &old in old_of_new {
                let (offset, len) = unpack(self.entries[old as usize]);
                entries.push(pack(text.len(), len));
                text.push_str(&self.text[offset..offset + len]);
            }
            text
        };
        // Eight buckets at a time. A group whose eight are all empty, which
        // the AND of the eight shows, is copied as it is, and every other
        // bucket takes its record's new position. An empty bucket's marker
        // lies past every position, so the bound of `new_of_old` answers for
        // it.
        let rewrite = |bucket: u32| {
            debug_assert!(bucket == EMPTY || (bucket as usize) < new_of_old.len());
            new_of_old.get(bucket as usize).copied().unwrap_or(EMPTY)
        };
        let mut buckets = Vec::with_capacity(self.buckets.capacity());
        let mut groups = self.buckets.chunks_exact(8);
        for group in &mut groups {
            if group.iter().fold(EMPTY, |all, &bucket| all & bucket) == EMPTY {
                buckets.extend_from_slice(group);
            } else {
                buckets.extend(group.iter().map(|&bucket| rewrite(bucket)));
            }
        }
        buckets.extend(groups.remainder().iter().map(|&bucket| rewrite(bucket)));
        IdParts {
            text,
            entries,
            buckets,
            live: Bitmap::first(held, 0),
            dead_text: 0,
        }
    }

    /// Trade this store's parts for `parts`, which [`IdStore::closed_up`]
    /// built from it, so the store holds the closed-up parts and `parts`
    /// holds the store's old ones. Moves alone, so a caller holding the
    /// store's write guard holds it for the swap and nothing else.
    pub fn swap_parts(&mut self, parts: &mut IdParts) {
        std::mem::swap(&mut self.text, &mut parts.text);
        std::mem::swap(&mut self.entries, &mut parts.entries);
        std::mem::swap(&mut self.buckets, &mut parts.buckets);
        std::mem::swap(&mut self.live, &mut parts.live);
        std::mem::swap(&mut self.dead_text, &mut parts.dead_text);
    }

    /// Every record's position, in increasing order, read from the live set
    /// a word at a time into a vector sized for the records held. These are
    /// the positions [`IdStore::slots`] reads from the entries, since every
    /// write keeps the two in step; a close-up lists them first.
    pub fn held_positions(&self) -> Vec<u32> {
        let mut held = Vec::with_capacity(self.held);
        self.live.for_each(|slot| held.push(slot as u32));
        debug_assert!(
            held.iter().map(|&at| at as usize).eq(self.slots()),
            "the live set holds exactly the positions whose entries are present"
        );
        held
    }

    /// Every part of the store, each with its capacity, for a test that
    /// holds two stores to be the same bytes. Absent from every build that
    /// is not a test.
    #[cfg(any(test, feature = "test-support"))]
    pub fn layout(&self) -> Vec<(&'static str, usize, String)> {
        vec![
            (
                "text",
                self.text.capacity(),
                format!("{:?}", self.text.as_bytes()),
            ),
            (
                "entries",
                self.entries.capacity(),
                format!("{:?}", self.entries),
            ),
            (
                "buckets",
                self.buckets.capacity(),
                format!("{:?}", self.buckets),
            ),
            ("tags", self.tags.capacity(), format!("{:?}", self.tags)),
            (
                "live",
                self.live.words.capacity(),
                format!("{:?}", self.live.words),
            ),
            ("held", 0, self.held.to_string()),
            ("dead_text", 0, self.dead_text.to_string()),
        ]
    }

    /// The position of `id`, where the store holds it.
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
        if entry == ABSENT {
            return None;
        }
        Some(self.text_of(entry))
    }

    /// The external id of `slot`, where it holds a record, for a loader
    /// naming every record it reads. The text is sliced by a check that
    /// answers `None` where the range is not one, which no entry's range
    /// is, rather than one that panics, so the read carries no panic path.
    /// [`IdStore::name`] is the reader the searches name their hits with.
    pub fn read_name(&self, slot: usize) -> Option<&str> {
        let entry = *self.entries.get(slot)?;
        if entry == ABSENT {
            return None;
        }
        let (offset, len) = unpack(entry);
        self.text.get(offset..offset + len)
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

    /// Every record as its position and its external id, in increasing
    /// position, which is internal id order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &str)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, &entry)| entry != ABSENT)
            .map(|(slot, &entry)| (slot, self.text_of(entry)))
    }

    /// Every record's position, in increasing order.
    pub fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, &entry)| entry != ABSENT)
            .map(|(slot, _)| slot)
    }

    /// The largest position that holds a record.
    pub fn highest_slot(&self) -> Option<usize> {
        self.entries.iter().rposition(|&entry| entry != ABSENT)
    }

    /// The set of positions that hold a record, as bits.
    pub fn live(&self) -> &Bitmap {
        &self.live
    }

    /// Whether the forward table names every record held exactly once and
    /// names nothing else: each bucket holds a position that holds a record,
    /// found again by its own text, and the buckets in use are the records
    /// held. A debug check, since every write keeps it true.
    pub fn table_agrees(&self) -> bool {
        let mut named = 0usize;
        for (at, &slot) in self.buckets.iter().enumerate() {
            if slot == EMPTY {
                continue;
            }
            named += 1;
            let Some(&entry) = self.entries.get(slot as usize) else {
                return false;
            };
            if entry == ABSENT || self.find(self.text_of(entry)) != Some(at) {
                return false;
            }
        }
        named == self.held && self.live.count() == self.held
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
    /// bytes it let go of. Every entry keeps its position; only its offset
    /// moves. Costs one pass over the entries and one copy of the live text.
    pub fn compact_text(&mut self) -> usize {
        if self.dead_text == 0 {
            return 0;
        }
        let before = self.text.len();
        let mut text = String::with_capacity(before - self.dead_text);
        for entry in self.entries.iter_mut() {
            if *entry == ABSENT {
                continue;
            }
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
    /// set's words.
    pub fn heap_bytes(&self) -> usize {
        self.text.capacity()
            + self.entries.capacity() * std::mem::size_of::<u64>()
            + self.buckets.capacity() * std::mem::size_of::<u32>()
            + self.tags.capacity()
            + self.live.heap_bytes()
    }

    /// Bytes of that request no record has been written into: the arena
    /// past its text, the entries past the positions assigned and the live
    /// set's words past them, which [`IdStore::shrink_to_fit`] gives back.
    /// The table gives nothing, since it is sized from the records it holds.
    pub fn reserved_bytes(&self) -> usize {
        (self.text.capacity() - self.text.len())
            + (self.entries.capacity() - self.entries.len()) * std::mem::size_of::<u64>()
            + self.live.reserved_past(self.entries.len())
    }

    /// Give back what [`IdStore::reserved_bytes`] prices, reporting the
    /// bytes released. No record moves.
    pub fn shrink_to_fit(&mut self) -> usize {
        let before = self.heap_bytes();
        self.text.shrink_to_fit();
        self.entries.shrink_to_fit();
        self.live.shrink_past(self.entries.len());
        before - self.heap_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table a close-up builds aside is the table the pass in place
    /// leaves and the table a rewrite of each bucket through the map gives,
    /// on tables from nearly empty to nearly full, which hold groups of eight
    /// empty buckets and groups holding a record, and every name resolves to
    /// its new position once the store is closed up.
    #[test]
    fn the_table_built_aside_is_the_table_each_bucket_rewritten_gives() {
        let (mut empty_groups, mut held_groups) = (0, 0);
        for (reserved, records, step) in [
            (1_000usize, 10usize, 3usize),
            (1_000, 300, 2),
            (0, 700, 5),
            (64, 56, 7),
            (0, 2, 2),
        ] {
            let mut store = IdStore::new(reserved);
            for at in 0..records {
                store.insert(at, &format!("r{at}")).unwrap();
            }
            for at in (1..records).step_by(step) {
                assert_eq!(store.remove_name(&format!("r{at}")), Some(at));
            }
            let old_of_new = store.held_positions();
            let mut new_of_old = vec![EMPTY; store.extent()];
            for (k, &old) in old_of_new.iter().enumerate() {
                new_of_old[old as usize] = k as u32;
            }
            for group in store.buckets.chunks_exact(8) {
                if group.iter().all(|&bucket| bucket == EMPTY) {
                    empty_groups += 1;
                } else {
                    held_groups += 1;
                }
            }
            let each: Vec<u32> = store
                .buckets
                .iter()
                .map(|&bucket| {
                    if bucket == EMPTY {
                        EMPTY
                    } else {
                        new_of_old[bucket as usize]
                    }
                })
                .collect();
            let parts = store.closed_up(&old_of_new, &new_of_old, reserved);
            assert_eq!(parts.buckets, each, "{reserved}, {records}");
            assert_eq!(parts.buckets.capacity(), store.buckets.capacity());
            store.close_up(&old_of_new, &new_of_old, reserved);
            assert_eq!(store.buckets, parts.buckets, "{reserved}, {records}");
            for (k, &old) in old_of_new.iter().enumerate() {
                assert_eq!(
                    store.slot_of(&format!("r{old}")),
                    Some(k),
                    "{reserved}, {records}"
                );
            }
        }
        assert!(
            empty_groups > 0 && held_groups > 0,
            "{empty_groups}, {held_groups}"
        );
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
        assert_eq!(store.entries.capacity(), 8);
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
        assert!(store.entries.capacity() >= 1501);
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
            store.entries.len(),
            2,
            "no entry was made for a refused slot"
        );
    }

    /// The positions held in increasing order and the map from each old
    /// position to its new one, as a close-up is handed them.
    fn closing(store: &IdStore) -> (Vec<u32>, Vec<u32>) {
        let old_of_new: Vec<u32> = store.slots().map(|slot| slot as u32).collect();
        let mut new_of_old = vec![crate::Position::NONE; store.extent()];
        for (k, &old) in old_of_new.iter().enumerate() {
            new_of_old[old as usize] = k as u32;
        }
        (old_of_new, new_of_old)
    }

    /// A store most of whose records were removed keeps them at scattered
    /// positions until the close-up, which moves every record to the front
    /// in its order, rewrites the table to the new positions, drops the
    /// removed text and leaves the entries and the live set costing what the
    /// records cost rather than a slot for every position the store reached.
    /// Every name, position, walk and removal then answers as a store built
    /// from those records does, and the next record takes the position after
    /// the last.
    #[test]
    fn a_store_over_scattered_ids_costs_what_its_records_cost() {
        let total = 20_000usize;
        let mut store = IdStore::new(16);
        for slot in 0..total {
            store.insert(slot, &format!("r{slot}")).unwrap();
        }
        let kept: Vec<usize> = (0..total).filter(|slot| slot % 40 == 7).collect();
        for slot in (0..total).filter(|slot| slot % 40 != 7) {
            assert!(store.remove_slot(slot));
        }
        assert_eq!(store.len(), kept.len());
        assert_eq!(store.extent(), total, "a removal leaves a hole");
        assert!(store.dead_text() > 0);

        let (old_of_new, new_of_old) = closing(&store);
        let reclaimed = store.close_up(&old_of_new, &new_of_old, 16);
        assert!(reclaimed > 0, "the close-up drops the removed text");
        assert_eq!(store.dead_text(), 0);
        assert_eq!(store.extent(), kept.len());
        assert!(store.table_agrees());

        // Eight bytes an entry and a bit a position, for the records held.
        assert_eq!(store.entries.capacity(), kept.len());
        assert_eq!(store.live.heap_bytes(), kept.len().div_ceil(64) * 8);
        assert_eq!(store.text.capacity(), store.text_bytes());

        for (k, &old) in kept.iter().enumerate() {
            let name = format!("r{old}");
            assert_eq!(store.slot_of(&name), Some(k));
            assert_eq!(store.name(k), Some(name.as_str()));
            assert!(store.contains_slot(k));
        }
        assert!(!store.contains_slot(kept.len()));
        assert_eq!(store.name(kept.len()), None);
        assert_eq!(
            store.slots().collect::<Vec<_>>(),
            (0..kept.len()).collect::<Vec<_>>()
        );
        assert_eq!(store.highest_slot(), Some(kept.len() - 1));

        for k in (0..kept.len()).step_by(2) {
            assert!(store.remove_slot(k));
        }
        assert_eq!(store.len(), kept.len() / 2);
        assert_eq!(store.live().count(), kept.len() / 2);
        assert!(store.table_agrees());
        store.insert(store.extent(), "after").unwrap();
        assert_eq!(store.slot_of("after"), Some(kept.len()));
    }

    /// A close-up keeps the reservation the declaration made, however few
    /// records it leaves, and gives back only what lies past it.
    #[test]
    fn a_close_up_keeps_the_declared_reservation() {
        let mut store = IdStore::new(1_000);
        for slot in 0..5_000usize {
            store.insert(slot, &format!("r{slot}")).unwrap();
        }
        for slot in 10..5_000usize {
            assert!(store.remove_slot(slot));
        }
        let (old_of_new, new_of_old) = closing(&store);
        store.close_up(&old_of_new, &new_of_old, 1_000);
        assert_eq!(store.extent(), 10);
        assert_eq!(store.entries.capacity(), 1_001);
        assert!(store.table_agrees());
    }

    /// A position taken back after the insertion that filed it leaves the
    /// store as it was before the insertion, and the next record takes the
    /// same position.
    #[test]
    fn a_retracted_position_is_taken_again() {
        let mut store = IdStore::new(4);
        store.insert(0, "a").unwrap();
        store.insert(1, "b").unwrap();
        assert!(store.remove_slot(0));
        store.insert(2, "refused").unwrap();
        store.retract(2);
        assert_eq!(store.extent(), 2);
        assert_eq!(store.len(), 1);
        assert!(!store.contains_name("refused"));
        assert!(!store.contains_slot(2));
        assert!(store.table_agrees());
        store.insert(store.extent(), "next").unwrap();
        assert_eq!(store.slot_of("next"), Some(2));
        assert_eq!(store.slot_of("b"), Some(1));
    }

    /// What the store reserved and no record has been written into is what
    /// `reserved_bytes` prices and what `shrink_to_fit` gives back, at the
    /// declaration, past it and after removals, after which the store prices
    /// nothing and resolves every record as before. A clear holds a fresh
    /// store's reservation and nothing else.
    #[test]
    fn the_unwritten_reservation_is_what_a_shrink_gives_back() {
        for (declared, records) in [(1_000usize, 10usize), (100, 3_000), (2_000_000, 500)] {
            let mut store = IdStore::new(declared);
            assert_eq!(
                store.reserved_bytes(),
                reserved_entries(declared) * 8,
                "{declared}: nothing is written"
            );
            for at in 0..records {
                store.insert(at, &format!("record-{at}")).unwrap();
            }
            for at in (0..records).step_by(3) {
                assert!(store.remove_slot(at));
            }
            let priced = store.reserved_bytes();
            let before = store.heap_bytes();
            let released = store.shrink_to_fit();
            assert_eq!(released, priced, "{declared}");
            assert_eq!(store.heap_bytes(), before - released, "{declared}");
            assert_eq!(store.reserved_bytes(), 0, "{declared}");
            assert_eq!(store.shrink_to_fit(), 0, "{declared}");
            for at in 0..records {
                let held = at % 3 != 0;
                assert_eq!(store.contains_slot(at), held, "{declared}: {at}");
                assert_eq!(
                    store.slot_of(&format!("record-{at}")),
                    held.then_some(at),
                    "{declared}: {at}"
                );
            }
            assert!(store.table_agrees());
            store.insert(store.extent(), "next").unwrap();
            assert_eq!(store.slot_of("next"), Some(records));

            store.clear(declared);
            let fresh = IdStore::new(declared);
            assert_eq!(store.heap_bytes(), fresh.heap_bytes(), "{declared}");
            assert_eq!(store.reserved_bytes(), fresh.reserved_bytes());
            assert_eq!(store.len(), 0);
            assert_eq!(store.extent(), 0);
        }
    }

    /// The loader's read of a name answers what the searches' read answers
    /// at every slot, a hole's and the slots past the extent included,
    /// with ids of one byte a character and of several, and after a
    /// close-up.
    #[test]
    fn a_loaders_read_of_a_name_answers_what_name_answers() {
        let mut store = IdStore::new(8);
        let ids: Vec<String> = (0..40)
            .map(|k| match k % 3 {
                0 => format!("{k}"),
                1 => format!("récord-{k}"),
                _ => "x".repeat(k + 1),
            })
            .collect();
        for (at, id) in ids.iter().enumerate() {
            store.insert(at, id).unwrap();
        }
        for at in (0..40).step_by(4) {
            assert!(store.remove_slot(at));
        }
        let agree = |store: &IdStore| {
            for slot in 0..store.extent() + 3 {
                assert_eq!(store.read_name(slot), store.name(slot), "{slot}");
            }
        };
        agree(&store);
        let (old_of_new, new_of_old) = closing(&store);
        store.close_up(&old_of_new, &new_of_old, 8);
        agree(&store);
    }
}
