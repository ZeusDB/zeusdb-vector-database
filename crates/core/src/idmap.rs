//! A value for each internal id a structure holds, in memory that follows the
//! entries rather than the largest id.
//!
//! Internal ids are issued in increasing order and never reused, so the ids a
//! long-lived collection holds spread over a range far larger than its
//! records. A vector indexed by id costs one entry for every id up to the
//! largest, whatever it holds. [`IdMap`] keeps the same entries in one of two
//! forms, and moves from the first to the second when the ids spread.
//!
//! # Flat
//!
//! One vector indexed by the id, the absent value in each slot that holds no
//! entry. It is the vector every structure keyed by internal id held before
//! this type: the same reservation, the same growth and the same capacity. A
//! map whose ids are dense stays flat, and costs and reads what that vector
//! did.
//!
//! # Paged
//!
//! The id range in pages of [`PAGE_IDS`] ids. A table indexed by the page
//! number holds the page's place in an arena, or [`NO_PAGE`], so the table
//! costs four bytes for each page of the range, 256 KiB at most below the id
//! ceiling. The arena holds the pages that hold an entry, in page order. A
//! page holds its entries in one of three kinds:
//!
//! - **dense**, a vector indexed by the id's offset in the page, the absent
//!   value where it holds none;
//! - **short**, at most [`SHORT_LIMIT`] entries, the offsets in increasing
//!   order with the values beside them;
//! - **ranked**, more entries than a short page holds, the values in offset
//!   order beside a rank index; see below.
//!
//! A sparse page, short or ranked, turns dense once its span, one past its
//! largest offset, is at most [`PAGE_RATIO`] times its entries, and a dense
//! page turns sparse once its span passes twice that, so a page between the
//! two keeps the kind it has. A short page turns ranked once it holds more
//! than [`SHORT_LIMIT`] entries, and a ranked page turns short once it holds
//! half that or fewer, so a page fed entries at either limit changes kind at
//! most once. A page that empties is freed.
//!
//! # The rank index
//!
//! A ranked page divides its offsets into groups of 64. Its summary is
//! [`SUMMARY_WORDS`] words, one bit for each of the page's groups, set where
//! the group holds an entry, and beside each summary word the count of held
//! groups before it. Each held group has one word, a bit for each of its
//! offsets that holds an entry, and the count of entries before it. A lookup
//! reads a summary word and its count, which give the group's place among
//! the held groups by one population count, then that group's word and
//! count, which give the value's place by another, then the value: three
//! dependent reads, whatever the page holds and in any order. The index costs
//! 160 bytes a page and 10 bytes a held group. A map's page keeps each word
//! beside its count, a record of five 16-bit slots, in the vector a short
//! page keeps its offsets in, so a page's header is the same in every kind.
//!
//! # When a flat map pages out
//!
//! The map's extent is one past the largest id it has written. A growth past
//! the map's capacity, its reservation, a planned run's extent and
//! [`FLAT_FLOOR`] stays flat while the extent is at most [`FLAT_RATIO`] times
//! the entries. A removal pages the map out where its extent is past the floor
//! and the reservation and is more than twice that ratio times the entries,
//! which leaves a margin between the two rules. A paged map stays paged until
//! its owner replaces it.
//!
//! # A lookup
//!
//! A lookup reads the flat vector first. A flat map answers there with the one
//! read the vector took. Only an id past the vector goes on to the table and
//! the page. A dense page answers in one read, a short page by a bisection of
//! at most [`SHORT_LIMIT`] offsets, and a ranked page in three reads.

/// Bits of an internal id below its page number.
const PAGE_SHIFT: u32 = 16;

/// Internal ids one page covers.
pub const PAGE_IDS: usize = 1 << PAGE_SHIFT;

/// The bits of an internal id that give its offset in its page.
const OFFSET_MASK: usize = PAGE_IDS - 1;

/// A table entry that names no page.
const NO_PAGE: u32 = u32::MAX;

/// The extent at or below which a flat map never pages out.
pub(crate) const FLAT_FLOOR: usize = 4_096;

/// Ids of its extent a flat map may hold for each entry when it grows.
pub(crate) const FLAT_RATIO: usize = 4;

/// Offsets of its span a dense page may hold for each entry.
pub(crate) const PAGE_RATIO: usize = 4;

/// Entries a short page holds at most. A sparse page that holds more is
/// ranked.
pub(crate) const SHORT_LIMIT: usize = 32;

/// Words of a ranked page's summary, one bit for each of a page's groups of
/// 64 offsets.
pub(crate) const SUMMARY_WORDS: usize = PAGE_IDS / 64 / 64;

/// The value a flat slot or a dense page's slot holds where the map holds no
/// entry.
pub trait Vacant {
    /// The absent value.
    fn vacant() -> Self;

    /// Whether this is the absent value.
    fn is_vacant(&self) -> bool;
}

impl Vacant for u32 {
    #[inline]
    fn vacant() -> Self {
        u32::MAX
    }

    #[inline]
    fn is_vacant(&self) -> bool {
        *self == u32::MAX
    }
}

impl Vacant for u64 {
    #[inline]
    fn vacant() -> Self {
        u64::MAX
    }

    #[inline]
    fn is_vacant(&self) -> bool {
        *self == u64::MAX
    }
}

/// NaN, which no length or score an entry holds is.
impl Vacant for f32 {
    #[inline]
    fn vacant() -> Self {
        f32::NAN
    }

    #[inline]
    fn is_vacant(&self) -> bool {
        self.is_nan()
    }
}

impl<T> Vacant for Option<T> {
    #[inline]
    fn vacant() -> Self {
        None
    }

    #[inline]
    fn is_vacant(&self) -> bool {
        self.is_none()
    }
}

/// The set bits of a word, lowest first.
#[derive(Clone, Copy)]
pub(crate) struct Bits(pub(crate) u64);

impl Iterator for Bits {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

/// A count a rank index keeps, being held groups before a summary word or
/// entries before a group. A page holds at most [`PAGE_IDS`] entries, so
/// every such count fits.
pub(crate) fn count(n: usize) -> u16 {
    u16::try_from(n).expect("a count within one page fits in a u16")
}

/// Slots of a ranked map page's `offsets` that one rank record takes: its
/// word as four, lowest bits first, then its count.
const RECORD: usize = 5;

/// The slots of a rank record holding `word` and `count`.
fn record_slots(word: u64, count: u16) -> [u16; RECORD] {
    [
        word as u16,
        (word >> 16) as u16,
        (word >> 32) as u16,
        (word >> 48) as u16,
        count,
    ]
}

/// Rank record `index` of a ranked map page, its word and its count, read
/// as one run of slots.
#[inline]
fn record(slots: &[u16], index: usize) -> (u64, u16) {
    let at = index * RECORD;
    let run: &[u16; RECORD] = slots[at..at + RECORD]
        .try_into()
        .expect("a rank record is one run of slots");
    let word = u64::from(run[0])
        | u64::from(run[1]) << 16
        | u64::from(run[2]) << 32
        | u64::from(run[3]) << 48;
    (word, run[4])
}

/// Write rank record `index`'s word, leaving its count.
fn set_word(slots: &mut [u16], index: usize, word: u64) {
    let at = index * RECORD;
    slots[at..at + RECORD - 1].copy_from_slice(&record_slots(word, 0)[..RECORD - 1]);
}

/// Count one more in each rank record from `from` up to `to`.
fn count_up(slots: &mut [u16], from: usize, to: usize) {
    for index in from..to {
        slots[index * RECORD + RECORD - 1] += 1;
    }
}

/// Count one fewer in each rank record from `from` up to `to`.
fn count_down(slots: &mut [u16], from: usize, to: usize) {
    for index in from..to {
        slots[index * RECORD + RECORD - 1] -= 1;
    }
}

/// The rank records of a ranked map page over `offsets`, which increase:
/// the summary's [`SUMMARY_WORDS`], each counting the held groups before it,
/// then one for each held group in group order, counting the entries before
/// it. At their exact size.
fn rank_records(offsets: impl Iterator<Item = usize> + Clone) -> Vec<u16> {
    let mut groups = 0;
    let mut last = usize::MAX;
    for offset in offsets.clone() {
        if offset >> 6 != last {
            last = offset >> 6;
            groups += 1;
        }
    }
    let mut slots = Vec::with_capacity((SUMMARY_WORDS + groups) * RECORD);
    slots.resize(SUMMARY_WORDS * RECORD, 0);
    let mut summary = [0u64; SUMMARY_WORDS];
    let (mut last, mut word, mut before) = (usize::MAX, 0u64, 0usize);
    for (at, offset) in offsets.enumerate() {
        let group = offset >> 6;
        if group != last {
            if last != usize::MAX {
                slots.extend(record_slots(word, count(before)));
            }
            summary[group >> 6] |= 1u64 << (group & 63);
            (last, word, before) = (group, 0, at);
        }
        word |= 1u64 << (offset & 63);
    }
    if last != usize::MAX {
        slots.extend(record_slots(word, count(before)));
    }
    let mut held = 0;
    for (index, &summary_word) in summary.iter().enumerate() {
        slots[index * RECORD..(index + 1) * RECORD]
            .copy_from_slice(&record_slots(summary_word, count(held)));
        held += summary_word.count_ones() as usize;
    }
    slots
}

/// Every offset a ranked map page holds, in increasing order, from its rank
/// records.
fn record_offsets(slots: &[u16]) -> impl Iterator<Item = usize> + '_ {
    (0..SUMMARY_WORDS)
        .flat_map(move |index| Bits(record(slots, index).0).map(move |bit| index * 64 + bit))
        .zip(SUMMARY_WORDS..)
        .flat_map(move |(group, index)| {
            Bits(record(slots, index).0).map(move |bit| group * 64 + bit)
        })
}

/// One past the largest offset a ranked map page holds, from its rank
/// records. Its last record belongs to its largest group.
fn record_span(slots: &[u16]) -> usize {
    let Some(index) = (0..SUMMARY_WORDS)
        .rev()
        .find(|&index| record(slots, index).0 != 0)
    else {
        return 0;
    };
    let group = index * 64 + 63 - record(slots, index).0.leading_zeros() as usize;
    let last = record(slots, slots.len() / RECORD - 1).0;
    group * 64 + 64 - last.leading_zeros() as usize
}

/// One page of a paged map.
#[derive(Clone)]
struct Page<T> {
    /// Which page of the id range this is.
    number: u32,
    /// Entries the page holds, above zero for every page in the arena.
    held: u32,
    /// Whether `values` is indexed by offset.
    dense: bool,
    /// Whether the page is sparse and keeps the rank index in `offsets`.
    ranked: bool,
    /// A short page's offsets, strictly increasing. A ranked page's rank
    /// records, [`RECORD`] slots each. Empty on a dense page.
    offsets: Vec<u16>,
    /// A dense page's values by offset, the absent value where it holds none.
    /// A sparse page's values, in the order of their offsets.
    values: Vec<T>,
}

impl<T: Vacant> Page<T> {
    fn new(number: u32) -> Self {
        Page {
            number,
            held: 0,
            dense: false,
            ranked: false,
            offsets: Vec::new(),
            values: Vec::new(),
        }
    }

    /// The first id the page covers.
    fn base(&self) -> usize {
        (self.number as usize) << PAGE_SHIFT
    }

    /// One past the largest offset the page holds an entry at.
    fn span(&self) -> usize {
        if self.dense {
            self.values
                .iter()
                .rposition(|value| !value.is_vacant())
                .map_or(0, |offset| offset + 1)
        } else if self.ranked {
            record_span(&self.offsets)
        } else {
            self.offsets.last().map_or(0, |&offset| offset as usize + 1)
        }
    }

    /// Whether a page of `held` entries over a span of `span` turns dense.
    fn dense_for(span: usize, held: usize) -> bool {
        span <= PAGE_RATIO.saturating_mul(held)
    }

    /// Whether a dense page of `held` entries over a span of `span` stays
    /// dense, which is within twice the ratio a sparse page turns dense at.
    fn stays_dense(span: usize, held: usize) -> bool {
        span <= (2 * PAGE_RATIO).saturating_mul(held)
    }

    /// The place among a ranked page's values of the entry at `offset`,
    /// where the page holds one. Two dependent reads, the summary word's
    /// record and then the group's, so the value is the third.
    #[inline]
    fn ranked_place(&self, offset: usize) -> Option<usize> {
        let group = offset >> 6;
        let (summary, held) = record(&self.offsets, group >> 6);
        let bit = 1u64 << (group & 63);
        if summary & bit == 0 {
            return None;
        }
        let index = SUMMARY_WORDS + usize::from(held) + (summary & (bit - 1)).count_ones() as usize;
        let (members, before) = record(&self.offsets, index);
        let low = 1u64 << (offset & 63);
        if members & low == 0 {
            return None;
        }
        Some(usize::from(before) + (members & (low - 1)).count_ones() as usize)
    }

    fn get(&self, offset: usize) -> Option<&T> {
        if self.dense {
            self.values.get(offset).filter(|value| !value.is_vacant())
        } else if self.ranked {
            self.ranked_place(offset).map(|at| &self.values[at])
        } else {
            let at = self.offsets.binary_search(&(offset as u16)).ok()?;
            Some(&self.values[at])
        }
    }

    fn get_mut(&mut self, offset: usize) -> Option<&mut T> {
        if self.dense {
            self.values
                .get_mut(offset)
                .filter(|value| !value.is_vacant())
        } else if self.ranked {
            let at = self.ranked_place(offset)?;
            Some(&mut self.values[at])
        } else {
            let at = self.offsets.binary_search(&(offset as u16)).ok()?;
            Some(&mut self.values[at])
        }
    }

    /// Put `value` at `offset`, handing back what it replaced. A dense page
    /// that would grow past twice its ratio turns sparse first. After the
    /// insert a sparse page whose span the ratio covers turns dense, and a
    /// short page holding more than [`SHORT_LIMIT`] entries turns ranked.
    fn insert(&mut self, offset: usize, value: T) -> Option<T> {
        let held = self.held as usize;
        if self.dense && offset >= self.values.len() && !Self::stays_dense(offset + 1, held + 1) {
            self.make_sparse();
        }
        let previous = if self.dense {
            if offset >= self.values.len() {
                self.values.resize_with(offset + 1, T::vacant);
            }
            let old = std::mem::replace(&mut self.values[offset], value);
            (!old.is_vacant()).then_some(old)
        } else if self.ranked {
            self.ranked_insert(offset, value)
        } else {
            let key = offset as u16;
            match self.offsets.binary_search(&key) {
                Ok(at) => Some(std::mem::replace(&mut self.values[at], value)),
                Err(at) => {
                    self.offsets.insert(at, key);
                    self.values.insert(at, value);
                    None
                }
            }
        };
        if previous.is_none() {
            self.held += 1;
        }
        if !self.dense {
            if Self::dense_for(self.span(), self.held as usize) {
                self.make_dense();
            } else if !self.ranked && self.held as usize > SHORT_LIMIT {
                self.make_ranked();
            }
        }
        previous
    }

    /// [`Page::insert`] on a ranked page: the group's record opened where
    /// the page holds no entry in it, the value put at its place, and every
    /// later group's count moved past it.
    fn ranked_insert(&mut self, offset: usize, value: T) -> Option<T> {
        let group = offset >> 6;
        let (summary, held) = record(&self.offsets, group >> 6);
        let bit = 1u64 << (group & 63);
        let index = SUMMARY_WORDS + usize::from(held) + (summary & (bit - 1)).count_ones() as usize;
        if summary & bit == 0 {
            let before = match self.offsets.get(index * RECORD..(index + 1) * RECORD) {
                Some(run) => run[RECORD - 1],
                None => count(self.values.len()),
            };
            let at = index * RECORD;
            self.offsets.splice(at..at, record_slots(0, before));
            set_word(&mut self.offsets, group >> 6, summary | bit);
            count_up(&mut self.offsets, (group >> 6) + 1, SUMMARY_WORDS);
        }
        let (members, before) = record(&self.offsets, index);
        let low = 1u64 << (offset & 63);
        let at = usize::from(before) + (members & (low - 1)).count_ones() as usize;
        if members & low != 0 {
            return Some(std::mem::replace(&mut self.values[at], value));
        }
        set_word(&mut self.offsets, index, members | low);
        self.values.insert(at, value);
        let records = self.offsets.len() / RECORD;
        count_up(&mut self.offsets, index + 1, records);
        None
    }

    /// Take the entry at `offset` out. A dense page left holding fewer than
    /// one entry in twice its ratio turns sparse, and a ranked page left
    /// holding half of [`SHORT_LIMIT`] entries or fewer turns short.
    fn remove(&mut self, offset: usize) -> Option<T> {
        let old = if self.dense {
            let slot = self.values.get_mut(offset)?;
            if slot.is_vacant() {
                return None;
            }
            std::mem::replace(slot, T::vacant())
        } else if self.ranked {
            self.ranked_remove(offset)?
        } else {
            let at = self.offsets.binary_search(&(offset as u16)).ok()?;
            self.offsets.remove(at);
            self.values.remove(at)
        };
        self.held -= 1;
        if self.dense && self.held > 0 && !Self::stays_dense(self.values.len(), self.held as usize)
        {
            self.make_sparse();
        } else if self.ranked && self.held as usize <= SHORT_LIMIT / 2 {
            self.make_short();
        }
        Some(old)
    }

    /// [`Page::remove`] on a ranked page: the value taken from its place,
    /// every later group's count moved back, and the group's record closed
    /// where the group empties.
    fn ranked_remove(&mut self, offset: usize) -> Option<T> {
        let group = offset >> 6;
        let (summary, held) = record(&self.offsets, group >> 6);
        let bit = 1u64 << (group & 63);
        if summary & bit == 0 {
            return None;
        }
        let index = SUMMARY_WORDS + usize::from(held) + (summary & (bit - 1)).count_ones() as usize;
        let (members, before) = record(&self.offsets, index);
        let low = 1u64 << (offset & 63);
        if members & low == 0 {
            return None;
        }
        let old = self
            .values
            .remove(usize::from(before) + (members & (low - 1)).count_ones() as usize);
        let records = self.offsets.len() / RECORD;
        count_down(&mut self.offsets, index + 1, records);
        if members == low {
            self.offsets.drain(index * RECORD..(index + 1) * RECORD);
            set_word(&mut self.offsets, group >> 6, summary & !bit);
            count_down(&mut self.offsets, (group >> 6) + 1, SUMMARY_WORDS);
        } else {
            set_word(&mut self.offsets, index, members & !low);
        }
        Some(old)
    }

    /// Turn a dense page sparse: short where it holds at most
    /// [`SHORT_LIMIT`] entries, ranked where it holds more.
    fn make_sparse(&mut self) {
        let values = std::mem::take(&mut self.values);
        let mut offsets = Vec::with_capacity(self.held as usize);
        let mut kept = Vec::with_capacity(self.held as usize);
        for (offset, value) in values.into_iter().enumerate() {
            if !value.is_vacant() {
                offsets.push(offset as u16);
                kept.push(value);
            }
        }
        self.offsets = offsets;
        self.values = kept;
        self.dense = false;
        self.ranked = false;
        if self.held as usize > SHORT_LIMIT {
            self.make_ranked();
        }
    }

    /// Turn a sparse page dense, its values by offset.
    fn make_dense(&mut self) {
        let span = self.span();
        let values = std::mem::take(&mut self.values);
        let mut dense = Vec::with_capacity(span);
        dense.resize_with(span, T::vacant);
        if self.ranked {
            for (offset, value) in record_offsets(&self.offsets).zip(values) {
                dense[offset] = value;
            }
        } else {
            for (&offset, value) in self.offsets.iter().zip(values) {
                dense[offset as usize] = value;
            }
        }
        self.offsets = Vec::new();
        self.values = dense;
        self.dense = true;
        self.ranked = false;
    }

    /// Turn a short page ranked, its offsets becoming rank records.
    fn make_ranked(&mut self) {
        self.offsets = rank_records(self.offsets.iter().map(|&offset| offset as usize));
        self.ranked = true;
    }

    /// Turn a ranked page short, its rank records becoming its offsets.
    fn make_short(&mut self) {
        let mut offsets = Vec::with_capacity(self.held as usize);
        offsets.extend(record_offsets(&self.offsets).map(|offset| offset as u16));
        self.offsets = offsets;
        self.ranked = false;
    }

    /// The kind the page's content gives, at its exact capacity: dense where
    /// the ratio covers its span, otherwise short or ranked by its entries.
    fn settle(&mut self) {
        let dense = Self::dense_for(self.span(), self.held as usize);
        if self.dense && !dense {
            self.make_sparse();
        } else if !self.dense && dense {
            self.make_dense();
        }
        if self.dense {
            let span = self.span();
            self.values.truncate(span);
        } else if self.ranked && self.held as usize <= SHORT_LIMIT {
            self.make_short();
        } else if !self.ranked && self.held as usize > SHORT_LIMIT {
            self.make_ranked();
        }
        self.offsets.shrink_to_fit();
        self.values.shrink_to_fit();
    }

    fn heap_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u16>()
            + self.values.capacity() * std::mem::size_of::<T>()
    }

    fn spare_bytes(&self) -> usize {
        (self.offsets.capacity() - self.offsets.len()) * std::mem::size_of::<u16>()
            + (self.values.capacity() - self.values.len()) * std::mem::size_of::<T>()
    }

    /// The page's entries as their ids and values, in increasing id order.
    fn iter(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        let base = self.base();
        let dense = self.dense.then(|| {
            self.values
                .iter()
                .enumerate()
                .filter(|(_, value)| !value.is_vacant())
                .map(move |(offset, value)| (base + offset, value))
        });
        let short = (!self.dense && !self.ranked).then(|| {
            self.offsets
                .iter()
                .zip(&self.values)
                .map(move |(&offset, value)| (base + offset as usize, value))
        });
        let ranked = self.ranked.then(|| {
            record_offsets(&self.offsets)
                .zip(&self.values)
                .map(move |(offset, value)| (base + offset, value))
        });
        dense
            .into_iter()
            .flatten()
            .chain(short.into_iter().flatten())
            .chain(ranked.into_iter().flatten())
    }

    fn iter_mut(&mut self) -> Box<dyn Iterator<Item = (usize, &mut T)> + '_> {
        let base = self.base();
        let Page {
            dense,
            ranked,
            offsets,
            values,
            ..
        } = self;
        if *dense {
            Box::new(
                values
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, value)| !value.is_vacant())
                    .map(move |(offset, value)| (base + offset, value)),
            )
        } else if *ranked {
            Box::new(
                record_offsets(offsets)
                    .zip(values.iter_mut())
                    .filter(|(_, value)| !value.is_vacant())
                    .map(move |(offset, value)| (base + offset, value)),
            )
        } else {
            Box::new(
                offsets
                    .iter()
                    .zip(values.iter_mut())
                    .filter(|(_, value)| !value.is_vacant())
                    .map(move |(&offset, value)| (base + offset as usize, value)),
            )
        }
    }
}

/// The paged form: the table, the arena and the extent.
#[derive(Clone)]
struct Pages<T> {
    /// The arena index of each page of the id range, or [`NO_PAGE`].
    table: Vec<u32>,
    /// Every page that holds an entry, in increasing page number.
    pages: Vec<Page<T>>,
    /// One past the largest id the map has written.
    extent: usize,
}

impl<T: Vacant> Pages<T> {
    fn new(extent: usize) -> Self {
        Pages {
            table: Vec::new(),
            pages: Vec::new(),
            extent,
        }
    }

    /// Page `number`, where the map holds one. [`NO_PAGE`] is past every
    /// place an arena can hold, so the arena's own bound answers for it.
    fn page(&self, number: usize) -> Option<&Page<T>> {
        let at = *self.table.get(number)?;
        self.pages.get(at as usize)
    }

    fn page_mut(&mut self, number: usize) -> Option<&mut Page<T>> {
        let at = *self.table.get(number)?;
        self.pages.get_mut(at as usize)
    }

    /// The arena index of page `number`, opening the page where it holds
    /// nothing yet. The arena stays in page order, so every page after the
    /// new one moves up one place and its table entry with it.
    fn open(&mut self, number: usize) -> usize {
        if number >= self.table.len() {
            self.table.reserve_exact(number + 1 - self.table.len());
            self.table.resize(number + 1, NO_PAGE);
        }
        let at = self.table[number];
        if at != NO_PAGE {
            return at as usize;
        }
        let place = self
            .pages
            .partition_point(|page| (page.number as usize) < number);
        let page_number = u32::try_from(number).expect("a page number fits in a u32");
        self.pages.insert(place, Page::new(page_number));
        for page in &self.pages[place + 1..] {
            self.table[page.number as usize] += 1;
        }
        self.table[number] = place as u32;
        place
    }

    fn insert(&mut self, id: usize, value: T) -> Option<T> {
        let at = self.open(id >> PAGE_SHIFT);
        let previous = self.pages[at].insert(id & OFFSET_MASK, value);
        self.extent = self.extent.max(id + 1);
        previous
    }

    /// Take the entry at `id` out, freeing its page where the page empties.
    fn remove(&mut self, id: usize) -> Option<T> {
        let number = id >> PAGE_SHIFT;
        let at = *self.table.get(number)?;
        if at == NO_PAGE {
            return None;
        }
        let at = at as usize;
        let old = self.pages[at].remove(id & OFFSET_MASK)?;
        if self.pages[at].held == 0 {
            self.pages.remove(at);
            self.table[number] = NO_PAGE;
            for page in &self.pages[at..] {
                self.table[page.number as usize] -= 1;
            }
        }
        Some(old)
    }

    fn settle(&mut self) {
        for page in &mut self.pages {
            page.settle();
        }
        self.pages.shrink_to_fit();
        self.table.shrink_to_fit();
    }

    fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.table.capacity() * std::mem::size_of::<u32>()
            + self.pages.capacity() * std::mem::size_of::<Page<T>>()
            + self.pages.iter().map(Page::heap_bytes).sum::<usize>()
    }

    fn spare_bytes(&self) -> usize {
        (self.table.capacity() - self.table.len()) * std::mem::size_of::<u32>()
            + (self.pages.capacity() - self.pages.len()) * std::mem::size_of::<Page<T>>()
            + self.pages.iter().map(Page::spare_bytes).sum::<usize>()
    }
}

/// A value for each internal id a structure holds. See the module.
#[derive(Clone)]
pub struct IdMap<T> {
    /// Every entry by id while the map is flat. Empty once it is paged.
    flat: Vec<T>,
    /// The pages, once the map is paged.
    paged: Option<Box<Pages<T>>>,
    /// Entries the map holds.
    held: usize,
    /// The extent a reservation asked for, at or below which the map stays
    /// flat.
    reserved: usize,
    /// The extent a planned run of insertions reaches, at or below which a
    /// growth stays flat.
    planned: usize,
}

impl<T> Default for IdMap<T> {
    fn default() -> Self {
        IdMap {
            flat: Vec::new(),
            paged: None,
            held: 0,
            reserved: 0,
            planned: 0,
        }
    }
}

impl<T: Vacant> IdMap<T> {
    /// An empty flat map that has asked the allocator for nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty flat map with room for ids below `extent`, which stays flat
    /// for as long as its extent is within it.
    pub fn with_capacity(extent: usize) -> Self {
        IdMap {
            flat: Vec::with_capacity(extent),
            paged: None,
            held: 0,
            reserved: extent,
            planned: 0,
        }
    }

    /// Reserve room for ids below `extent` and no more, on a flat map, and
    /// keep the map flat within it.
    pub fn reserve_exact(&mut self, extent: usize) {
        if self.paged.is_none() {
            self.flat
                .reserve_exact(extent.saturating_sub(self.flat.len()));
            self.reserved = self.reserved.max(extent);
        }
    }

    /// Decide the form for a run of insertions of `records` entries whose
    /// largest id is `highest`, from the run's final shape rather than its
    /// order. A run dense enough to stay flat grows flat to its end; one that
    /// is not pages the map out before it starts.
    pub fn plan(&mut self, records: usize, highest: usize) {
        if self.paged.is_some() {
            return;
        }
        let extent = highest.saturating_add(1);
        let held = self.held.saturating_add(records);
        if extent <= FLAT_FLOOR.max(self.reserved).max(self.flat.capacity())
            || extent <= FLAT_RATIO.saturating_mul(held)
        {
            self.planned = self.planned.max(extent);
        } else {
            self.page_out();
        }
    }

    /// [`IdMap::plan`], and on a map left flat room for ids up to
    /// `highest`.
    pub fn reserve(&mut self, records: usize, highest: usize) {
        self.plan(records, highest);
        if self.paged.is_none() {
            let extent = highest.saturating_add(1);
            if extent > self.flat.len() {
                self.flat.reserve(extent - self.flat.len());
            }
        }
    }

    /// Entries the map holds.
    pub fn len(&self) -> usize {
        self.held
    }

    /// Whether the map holds no entry.
    pub fn is_empty(&self) -> bool {
        self.held == 0
    }

    /// Whether the map is flat.
    pub fn is_flat(&self) -> bool {
        self.paged.is_none()
    }

    /// The flat vector, the absent value in each slot with no entry, while
    /// the map is flat.
    pub fn as_flat(&self) -> Option<&[T]> {
        self.paged.is_none().then_some(self.flat.as_slice())
    }

    /// One past the largest id the map has written.
    pub fn extent(&self) -> usize {
        match self.paged.as_deref() {
            None => self.flat.len(),
            Some(pages) => pages.extent,
        }
    }

    /// The largest id the map holds an entry at.
    pub fn highest(&self) -> Option<usize> {
        match self.paged.as_deref() {
            None => self.flat.iter().rposition(|value| !value.is_vacant()),
            Some(pages) => pages.pages.last().map(|page| page.base() + page.span() - 1),
        }
    }

    /// The entry at `id`. Total over every `usize`.
    #[inline]
    pub fn get(&self, id: usize) -> Option<&T> {
        match self.flat.get(id) {
            Some(value) => (!value.is_vacant()).then_some(value),
            None => self.paged_get(id),
        }
    }

    #[inline(never)]
    fn paged_get(&self, id: usize) -> Option<&T> {
        self.paged
            .as_deref()?
            .page(id >> PAGE_SHIFT)?
            .get(id & OFFSET_MASK)
    }

    /// The entry at `id`, to change in place.
    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        if self.paged.is_none() {
            return self.flat.get_mut(id).filter(|value| !value.is_vacant());
        }
        self.paged
            .as_deref_mut()?
            .page_mut(id >> PAGE_SHIFT)?
            .get_mut(id & OFFSET_MASK)
    }

    /// Whether the map holds an entry at `id`.
    #[inline]
    pub fn contains(&self, id: usize) -> bool {
        self.get(id).is_some()
    }

    /// Put `value` at `id`, handing back the entry it replaced. `value` is
    /// not the absent value, and `id` fits in a `u32`, which every internal
    /// id does.
    pub fn insert(&mut self, id: usize, value: T) -> Option<T> {
        debug_assert!(!value.is_vacant(), "an entry is never the absent value");
        assert!(
            u32::try_from(id).is_ok(),
            "an internal id fits in a u32, and {} does not",
            id
        );
        if self.paged.is_none() {
            if let Some(slot) = self.flat.get_mut(id) {
                let old = std::mem::replace(slot, value);
                if old.is_vacant() {
                    self.held += 1;
                    return None;
                }
                return Some(old);
            }
            let extent = id + 1;
            if self.flat_may_reach(extent) {
                self.flat.resize_with(extent, T::vacant);
                self.flat[id] = value;
                self.held += 1;
                return None;
            }
            self.page_out();
        }
        let pages = self
            .paged
            .as_deref_mut()
            .expect("the map is paged past this point");
        let previous = pages.insert(id, value);
        if previous.is_none() {
            self.held += 1;
        }
        previous
    }

    /// Whether a flat map may grow to `extent`: inside its allocation, its
    /// reservation, a planned run or the floor, or while it holds at least
    /// one entry in [`FLAT_RATIO`] of the extent once the entry is in.
    fn flat_may_reach(&self, extent: usize) -> bool {
        extent <= self.flat.capacity()
            || extent <= FLAT_FLOOR.max(self.reserved).max(self.planned)
            || extent <= FLAT_RATIO.saturating_mul(self.held + 1)
    }

    /// Take the entry at `id` out and hand it back.
    pub fn remove(&mut self, id: usize) -> Option<T> {
        if self.paged.is_none() {
            let slot = self.flat.get_mut(id)?;
            if slot.is_vacant() {
                return None;
            }
            let old = std::mem::replace(slot, T::vacant());
            self.held -= 1;
            let extent = self.flat.len();
            if extent > FLAT_FLOOR.max(self.reserved)
                && (2 * FLAT_RATIO).saturating_mul(self.held) < extent
            {
                self.page_out();
            }
            return Some(old);
        }
        let old = self.paged.as_deref_mut()?.remove(id)?;
        self.held -= 1;
        Some(old)
    }

    /// Move a flat map's entries into pages and free the vector. The extent
    /// is kept.
    pub fn page_out(&mut self) {
        if self.paged.is_some() {
            return;
        }
        let flat = std::mem::take(&mut self.flat);
        let mut pages = Pages::new(flat.len());
        for (id, value) in flat.into_iter().enumerate() {
            if !value.is_vacant() {
                pages.insert(id, value);
            }
        }
        pages.settle();
        self.paged = Some(Box::new(pages));
    }

    /// Put each page in the kind its content gives, at its exact capacity.
    /// What a load does once it has written every entry, so a paged map
    /// holds the same bytes whatever order its entries came in. A flat map
    /// is left as it is, its capacity being the vector's own.
    pub fn settle(&mut self) {
        if let Some(pages) = self.paged.as_deref_mut() {
            pages.settle();
        }
    }

    /// Return the spare capacity to the allocator: the vector's on a flat
    /// map, and each page's, the arena's and the table's on a paged one.
    pub fn shrink_to_fit(&mut self) {
        match self.paged.as_deref_mut() {
            None => self.flat.shrink_to_fit(),
            Some(pages) => pages.settle(),
        }
    }

    /// Bytes the map asked the allocator for.
    pub fn heap_bytes(&self) -> usize {
        self.flat.capacity() * std::mem::size_of::<T>()
            + self.paged.as_deref().map_or(0, Pages::heap_bytes)
    }

    /// Bytes of that request that hold no slot yet.
    pub fn spare_bytes(&self) -> usize {
        (self.flat.capacity() - self.flat.len()) * std::mem::size_of::<T>()
            + self.paged.as_deref().map_or(0, Pages::spare_bytes)
    }

    /// Every entry as its id and its value, in increasing id order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        let flat = self
            .flat
            .iter()
            .enumerate()
            .filter(|(_, value)| !value.is_vacant());
        let paged = self
            .paged
            .as_deref()
            .into_iter()
            .flat_map(|pages| pages.pages.iter().flat_map(Page::iter));
        flat.chain(paged)
    }

    /// Every entry as its id and its value to change in place, in increasing
    /// id order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> + '_ {
        let IdMap { flat, paged, .. } = self;
        let flat = flat
            .iter_mut()
            .enumerate()
            .filter(|(_, value)| !value.is_vacant());
        let paged = paged
            .as_deref_mut()
            .into_iter()
            .flat_map(|pages| pages.pages.iter_mut().flat_map(Page::iter_mut));
        flat.chain(paged)
    }

    /// A map of the same ids in the same form, each value through `f`. A
    /// flat map's vector is allocated at its length, which is what a vector
    /// built by pushing each mapped slot holds.
    pub fn map<U: Vacant>(&self, mut f: impl FnMut(&T) -> U) -> IdMap<U> {
        let mut flat = Vec::with_capacity(self.flat.len());
        for value in &self.flat {
            flat.push(if value.is_vacant() {
                U::vacant()
            } else {
                f(value)
            });
        }
        let paged = self.paged.as_deref().map(|pages| {
            Box::new(Pages {
                table: pages.table.clone(),
                pages: pages
                    .pages
                    .iter()
                    .map(|page| Page {
                        number: page.number,
                        held: page.held,
                        dense: page.dense,
                        offsets: page.offsets.clone(),
                        ranked: page.ranked,
                        values: page
                            .values
                            .iter()
                            .map(|value| {
                                if value.is_vacant() {
                                    U::vacant()
                                } else {
                                    f(value)
                                }
                            })
                            .collect(),
                    })
                    .collect(),
                extent: pages.extent,
            })
        });
        IdMap {
            flat,
            paged,
            held: self.held,
            reserved: self.reserved,
            planned: self.planned,
        }
    }

    /// The form as plain values, for a test that holds two maps to being
    /// one map: the flat vector's capacity and its slots, or the paged
    /// form's bytes and each page's number, kind and entries.
    #[cfg(test)]
    pub(crate) fn snapshot(&self, mut widen: impl FnMut(&T) -> u64) -> (usize, Vec<u64>) {
        match self.paged.as_deref() {
            None => (
                self.flat.capacity(),
                self.flat.iter().map(&mut widen).collect(),
            ),
            Some(pages) => {
                let mut out = vec![u64::MAX, pages.extent as u64];
                for page in &pages.pages {
                    out.push(page.number as u64);
                    out.push(page.dense as u64);
                    out.push(page.ranked as u64);
                    for (id, value) in page.iter() {
                        out.push(id as u64);
                        out.push(widen(value));
                    }
                }
                (self.heap_bytes(), out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plain vector indexed by id, the model every map is held to.
    struct Model(Vec<Option<u64>>);

    impl Model {
        fn insert(&mut self, id: usize, value: u64) -> Option<u64> {
            if id >= self.0.len() {
                self.0.resize(id + 1, None);
            }
            self.0[id].replace(value)
        }

        fn remove(&mut self, id: usize) -> Option<u64> {
            self.0.get_mut(id).and_then(Option::take)
        }

        fn entries(&self) -> Vec<(usize, u64)> {
            self.0
                .iter()
                .enumerate()
                .filter_map(|(id, value)| value.map(|v| (id, v)))
                .collect()
        }
    }

    fn entries(map: &IdMap<u64>) -> Vec<(usize, u64)> {
        map.iter().map(|(id, &value)| (id, value)).collect()
    }

    /// Every question the map answers, against the model.
    fn agrees(map: &IdMap<u64>, model: &Model, probe: &[usize]) {
        let expected = model.entries();
        assert_eq!(entries(map), expected);
        assert_eq!(map.len(), expected.len());
        assert_eq!(map.highest(), expected.last().map(|&(id, _)| id));
        assert_eq!(map.extent(), model.0.len());
        for &id in probe {
            assert_eq!(
                map.get(id).copied(),
                model.0.get(id).copied().flatten(),
                "{id}"
            );
            assert_eq!(
                map.contains(id),
                model.0.get(id).copied().flatten().is_some()
            );
        }
    }

    /// A deterministic stream of numbers, so a failing case repeats.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Dense ids keep the flat form for the map's life, and the vector is the
    /// one a plain `Vec` indexed by id holds: the same length, the same
    /// growth, the same capacity and the same bytes.
    #[test]
    fn a_dense_map_is_the_vector_it_replaced() {
        let mut map: IdMap<u64> = IdMap::with_capacity(1_001);
        let mut vector: Vec<u64> = Vec::with_capacity(1_001);
        for id in 1..=20_000usize {
            map.insert(id, id as u64 * 3);
            if id >= vector.len() {
                vector.resize(id + 1, u64::MAX);
            }
            vector[id] = id as u64 * 3;
        }
        assert!(map.is_flat());
        assert_eq!(map.as_flat(), Some(vector.as_slice()));
        assert_eq!(map.heap_bytes(), vector.capacity() * 8);
        assert_eq!(map.spare_bytes(), (vector.capacity() - vector.len()) * 8);
        for id in (1..=20_000usize).step_by(3) {
            assert_eq!(map.remove(id), Some(id as u64 * 3));
            vector[id] = u64::MAX;
        }
        assert!(map.is_flat(), "two thirds held is dense");
        assert_eq!(map.as_flat(), Some(vector.as_slice()));
        map.shrink_to_fit();
        vector.shrink_to_fit();
        assert_eq!(map.heap_bytes(), vector.capacity() * 8);
    }

    /// Ids scattered one to a page and a run under a large base page the
    /// map out, and it then costs what its entries cost: the table, a header
    /// a page and ten bytes an entry, never a slot for an id it does not hold.
    #[test]
    fn a_scattered_map_costs_what_its_entries_cost() {
        let mut map: IdMap<u64> = IdMap::new();
        for k in 0..64usize {
            map.insert(k * 1_000_003 + 7, k as u64);
        }
        assert!(!map.is_flat());
        map.settle();
        let pages = map.paged.as_deref().unwrap();
        assert_eq!(pages.pages.len(), 64);
        assert!(pages.pages.iter().all(|page| !page.dense && page.held == 1));
        let table = pages.table.len() * 4;
        assert_eq!(pages.table.len(), (63 * 1_000_003 + 7) / PAGE_IDS + 1);
        let headers = 64 * std::mem::size_of::<Page<u64>>();
        let bound = std::mem::size_of::<Pages<u64>>() + table + headers + 64 * (2 + 8) * 4;
        assert!(
            map.heap_bytes() <= bound,
            "{} > {}",
            map.heap_bytes(),
            bound
        );
        for k in 0..64usize {
            assert_eq!(map.get(k * 1_000_003 + 7), Some(&(k as u64)));
            assert_eq!(map.get(k * 1_000_003 + 8), None);
        }

        let mut run: IdMap<u64> = IdMap::new();
        for id in (1usize << 30)..(1usize << 30) + 5_000 {
            run.insert(id, 1);
        }
        assert!(!run.is_flat());
        run.settle();
        let pages = run.paged.as_deref().unwrap();
        assert!(pages.pages.iter().all(|page| page.dense));
        assert!(run.heap_bytes() < 4 * ((1usize << 30) / PAGE_IDS + 1) + 2 * 5_000 * 8 + 1_024);
        assert_eq!(run.extent(), (1usize << 30) + 5_000);
        assert_eq!(run.highest(), Some((1usize << 30) + 4_999));
    }

    /// The model test. Insertions, overwrites and removals over ids drawn to
    /// cross every rule: a dense run, a scattered tail, a page emptied and
    /// refilled, a dense page thinned to sparse and a sparse page filled to
    /// dense. After every step the map answers what the model answers.
    #[test]
    fn the_map_answers_what_a_vector_answers_through_every_form() {
        for seed in 1..=6u64 {
            let mut stream = Stream(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut map: IdMap<u64> = IdMap::new();
            let mut model = Model(Vec::new());
            let mut probe: Vec<usize> = vec![0, 1, 4_095, 4_096, 65_535, 65_536, 1 << 20];
            for step in 0..6_000u64 {
                let draw = stream.next();
                let id = match draw % 7 {
                    0..=2 => (step as usize) % 9_000,
                    3 => 200_000 + (draw as usize >> 8) % 3,
                    4 => (draw as usize >> 8) % (1 << 22),
                    5 => 70_000 + ((draw as usize >> 8) % 30_000),
                    _ => model.0.len().saturating_sub(1 + (draw as usize >> 8) % 50),
                };
                if draw.is_multiple_of(5) {
                    assert_eq!(map.remove(id), model.remove(id), "remove {id} step {step}");
                } else {
                    assert_eq!(map.insert(id, draw), model.insert(id, draw), "insert {id}");
                }
                if step % 97 == 0 {
                    probe.push(id);
                    probe.push(id + 1);
                    agrees(&map, &model, &probe);
                }
                if step % 1_013 == 0 {
                    map.settle();
                }
            }
            agrees(&map, &model, &probe);
            assert!(!map.is_flat(), "every seed pages the map out");
            for (id, value) in map.iter_mut() {
                *value = value.wrapping_add(id as u64);
            }
            for (id, value) in model.0.iter_mut().enumerate() {
                if let Some(value) = value {
                    *value = value.wrapping_add(id as u64);
                }
            }
            agrees(&map, &model, &probe);
            let doubled = map.map(|value| value.wrapping_mul(2));
            let expected: Vec<(usize, u64)> = model
                .entries()
                .into_iter()
                .map(|(id, value)| (id, value.wrapping_mul(2)))
                .collect();
            assert_eq!(entries(&doubled), expected);
            assert_eq!(doubled.is_flat(), map.is_flat());
        }
    }

    /// A planned run lays out from its final shape, so the same entries
    /// inserted in increasing and in scrambled order end in the same form,
    /// and once settled in the same bytes.
    #[test]
    fn a_planned_run_lays_out_the_same_in_any_order() {
        let dense: Vec<usize> = (1..=30_000).collect();
        let sparse: Vec<usize> = (1..=3_000).map(|k| k * 97).collect();
        for ids in [dense, sparse] {
            let highest = *ids.iter().max().unwrap();
            let mut scrambled = ids.clone();
            let mut stream = Stream(77);
            for i in (1..scrambled.len()).rev() {
                let j = (stream.next() as usize) % (i + 1);
                scrambled.swap(i, j);
            }
            let mut maps = Vec::new();
            for order in [&ids, &scrambled] {
                let mut map: IdMap<u64> = IdMap::with_capacity(1_001);
                map.plan(order.len(), highest);
                for &id in order.iter() {
                    map.insert(id, id as u64);
                }
                map.settle();
                maps.push(map);
            }
            assert_eq!(maps[0].is_flat(), maps[1].is_flat());
            assert_eq!(entries(&maps[0]), entries(&maps[1]));
            if !maps[0].is_flat() {
                assert_eq!(maps[0].snapshot(|&v| v), maps[1].snapshot(|&v| v));
            }
        }
    }

    /// A reservation keeps the map flat within it however few entries it
    /// holds, and a removal past it pages the map out.
    #[test]
    fn a_reservation_keeps_the_map_flat_within_it() {
        let mut map: IdMap<u64> = IdMap::with_capacity(50_001);
        map.insert(50_000, 1);
        map.insert(3, 1);
        assert!(map.is_flat());
        map.remove(3);
        assert!(map.is_flat(), "within the reservation");
        map.insert(60_000, 1);
        assert!(!map.is_flat(), "two entries cannot hold 60,001 ids flat");
        assert_eq!(map.get(50_000), Some(&1));
        assert_eq!(map.get(60_000), Some(&1));
        assert_eq!(map.extent(), 60_001);
    }

    /// An id past the table reads absent, as an id past a vector did.
    #[test]
    fn an_id_past_every_page_reads_absent() {
        let mut map: IdMap<u64> = IdMap::new();
        map.insert(1 << 20, 5);
        assert!(!map.is_flat());
        for id in [0usize, (1 << 20) - 1, (1 << 20) + 1, 1 << 31, usize::MAX] {
            assert_eq!(map.get(id), None);
            assert_eq!(map.remove(id), None);
        }
        assert_eq!(map.len(), 1);
    }

    /// A page changes kind as its entries come and go: dense while its span
    /// is at most four times its entries, sparse once a removal leaves it an
    /// eighth full, dense again once its entries fill it, and freed with its
    /// table entry once it empties, the arena staying in page order.
    #[test]
    fn a_page_changes_kind_as_its_entries_come_and_go() {
        let mut map: IdMap<u64> = IdMap::new();
        map.page_out();
        let base = 3 * PAGE_IDS;
        for offset in 0..4_000usize {
            map.insert(base + offset, offset as u64);
        }
        map.insert(PAGE_IDS + 9, 1);
        map.insert(7 * PAGE_IDS + 9, 1);
        let kind = |map: &IdMap<u64>, number: usize| {
            let pages = map.paged.as_deref().unwrap();
            pages.page(number).map(|page| (page.dense, page.held))
        };
        assert_eq!(kind(&map, 3), Some((true, 4_000)));
        for offset in 0..3_600usize {
            map.remove(base + offset);
        }
        assert_eq!(
            kind(&map, 3),
            Some((false, 400)),
            "an eighth full is sparse"
        );
        for offset in 0..3_600usize {
            map.insert(base + offset, 2);
        }
        assert_eq!(kind(&map, 3), Some((true, 4_000)), "refilled is dense");
        for offset in 0..4_000usize {
            map.remove(base + offset);
        }
        assert_eq!(kind(&map, 3), None, "an empty page is freed");
        let pages = map.paged.as_deref().unwrap();
        let numbers: Vec<u32> = pages.pages.iter().map(|page| page.number).collect();
        assert_eq!(numbers, vec![1, 7]);
        assert_eq!(pages.table[1], 0);
        assert_eq!(pages.table[7], 1);
        assert_eq!(pages.table[3], NO_PAGE);
        assert_eq!(map.len(), 2);
        assert_eq!(map.extent(), 7 * PAGE_IDS + 10);
        map.insert(5 * PAGE_IDS, 3);
        let pages = map.paged.as_deref().unwrap();
        let numbers: Vec<u32> = pages.pages.iter().map(|page| page.number).collect();
        assert_eq!(
            numbers,
            vec![1, 5, 7],
            "a page opened between two keeps the order"
        );
        assert_eq!((pages.table[1], pages.table[5], pages.table[7]), (0, 1, 2));
        assert_eq!(
            entries(&map),
            vec![(PAGE_IDS + 9, 1), (5 * PAGE_IDS, 3), (7 * PAGE_IDS + 9, 1)]
        );
    }

    /// A page fed offsets in increasing order whose gaps straddle its ratio,
    /// one above and one below in turn, from every start, keeps its kind
    /// rather than changing it at every few entries, so a run of insertions
    /// costs what its entries cost.
    #[test]
    fn a_page_fed_gaps_about_its_ratio_keeps_its_kind() {
        for start in 0..2 * PAGE_RATIO {
            let mut map: IdMap<u64> = IdMap::new();
            map.page_out();
            let mut offset = start;
            let mut changes = 0;
            let mut last = None;
            for k in 0..4_000usize {
                map.insert(offset, k as u64);
                let dense = map.paged.as_deref().unwrap().pages[0].dense;
                changes += usize::from(last.is_some_and(|was| was != dense));
                last = Some(dense);
                offset += if k % 2 == 0 {
                    PAGE_RATIO + 1
                } else {
                    PAGE_RATIO - 1
                };
            }
            assert!(changes <= 1, "from {start}, {changes} changes of kind");
            assert_eq!(map.len(), 4_000);
        }
    }

    /// The kind of the page holding `id`: 0 dense, 1 short, 2 ranked.
    fn kind_of<T: Vacant>(map: &IdMap<T>, id: usize) -> Option<u8> {
        let page = map.paged.as_deref()?.page(id >> PAGE_SHIFT)?;
        Some(if page.dense {
            0
        } else if page.ranked {
            2
        } else {
            1
        })
    }

    /// `ids` in an order a seed fixes.
    fn scrambled(ids: &[usize], seed: u64) -> Vec<usize> {
        let mut out = ids.to_vec();
        let mut stream = Stream(seed);
        for i in (1..out.len()).rev() {
            let j = (stream.next() as usize) % (i + 1);
            out.swap(i, j);
        }
        out
    }

    /// Ids that put a page in each kind once settled: page 3 ranked, offsets
    /// seven apart with a whole group of 64 and the page's last group whole;
    /// page 5 short, its first and last offsets among 20; page 6 dense, 3,000
    /// in a row; page 9 ranked, the first and last offset of every summary
    /// word's run of 4,096 among 72.
    fn every_kind() -> Vec<usize> {
        let mut ids: Vec<usize> = Vec::new();
        ids.extend((0..2_000).map(|k| 3 * PAGE_IDS + k * 7));
        ids.extend((0..64).map(|k| 3 * PAGE_IDS + 30_016 + k));
        ids.extend((0..64).map(|k| 3 * PAGE_IDS + 65_472 + k));
        ids.extend([5 * PAGE_IDS, 5 * PAGE_IDS + 65_535]);
        ids.extend((1..19).map(|k| 5 * PAGE_IDS + k * 3_001));
        ids.extend((0..3_000).map(|k| 6 * PAGE_IDS + 100 + k));
        ids.extend(
            (0..16)
                .flat_map(|w| [w * 4_096, w * 4_096 + 4_095])
                .map(|o| 9 * PAGE_IDS + o),
        );
        ids.extend((0..40).map(|k| 9 * PAGE_IDS + 777 + k * 1_500));
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Every page kind answers what a plain vector answers: entries inserted
    /// in increasing and in scrambled order, overwritten, changed in place,
    /// thinned until a ranked page turns short and a dense page sparse, and
    /// put back, with a settle between each step and none after the last.
    #[test]
    fn each_page_kind_answers_what_a_vector_answers_in_any_order() {
        let ids = every_kind();
        let mut probe: Vec<usize> = ids
            .iter()
            .flat_map(|&id| [id, id + 1, id.saturating_sub(1)])
            .collect();
        probe.extend([0, PAGE_IDS, 4 * PAGE_IDS, 10 * PAGE_IDS, 1 << 30]);
        for seed in [None, Some(11u64), Some(12)] {
            let feed = match seed {
                Some(seed) => scrambled(&ids, seed),
                None => ids.clone(),
            };
            let mut map: IdMap<u64> = IdMap::new();
            map.page_out();
            let mut model = Model(Vec::new());
            for &id in &feed {
                assert_eq!(map.insert(id, id as u64), model.insert(id, id as u64));
            }
            agrees(&map, &model, &probe);
            map.settle();
            agrees(&map, &model, &probe);
            let kinds: Vec<Option<u8>> = [3, 5, 6, 9]
                .iter()
                .map(|&page| kind_of(&map, page * PAGE_IDS))
                .collect();
            assert_eq!(kinds, vec![Some(2), Some(1), Some(0), Some(2)], "{seed:?}");

            for &id in feed.iter().step_by(3) {
                let value = 3 * id as u64 + 1;
                assert_eq!(map.insert(id, value), model.insert(id, value));
            }
            for &id in feed.iter().skip(1).step_by(4) {
                *map.get_mut(id).unwrap() += 5;
                *model.0[id].as_mut().unwrap() += 5;
            }
            agrees(&map, &model, &probe);
            map.settle();

            let thin: Vec<usize> = feed
                .iter()
                .copied()
                .filter(|&id| id % 10 != 3 || (5 * PAGE_IDS..6 * PAGE_IDS).contains(&id))
                .collect();
            for &id in &thin {
                assert_eq!(map.remove(id), model.remove(id), "remove {id}");
                assert_eq!(map.remove(id), None);
            }
            agrees(&map, &model, &probe);
            assert_eq!(
                kind_of(&map, 5 * PAGE_IDS),
                None,
                "an emptied page is freed"
            );
            assert_eq!(
                (kind_of(&map, 6 * PAGE_IDS), kind_of(&map, 9 * PAGE_IDS)),
                (Some(2), Some(1)),
                "a dense page thinned turns sparse and a ranked page short, {seed:?}"
            );
            map.settle();
            agrees(&map, &model, &probe);

            for &id in thin.iter().rev() {
                assert_eq!(
                    map.insert(id, 1 + id as u64),
                    model.insert(id, 1 + id as u64)
                );
            }
            agrees(&map, &model, &probe);
        }
    }

    /// A sparse page settles short at the short limit and below it and
    /// ranked above it: 31, 32, 33 and 34 entries spread over a page. It
    /// settles dense where its span is four times its entries and ranked one
    /// offset past that. A ranked page thinned to 17 entries stays ranked
    /// and turns short at 16, and refilled to 32 it stays short and turns
    /// ranked at 33.
    #[test]
    fn a_sparse_page_takes_its_kind_at_and_about_the_short_limit() {
        for (held, kind) in [(31usize, 1u8), (32, 1), (33, 2), (34, 2)] {
            let mut map: IdMap<u64> = IdMap::new();
            map.page_out();
            for k in 0..held {
                map.insert(k * 1_000, k as u64);
            }
            map.settle();
            assert_eq!(kind_of(&map, 0), Some(kind), "{held} entries");
            assert_eq!(map.len(), held);
        }
        for (last, kind) in [(399usize, 0u8), (400, 2)] {
            let mut map: IdMap<u64> = IdMap::new();
            map.page_out();
            for offset in 0..99 {
                map.insert(offset, 1);
            }
            map.insert(last, 1);
            map.settle();
            assert_eq!(
                kind_of(&map, 0),
                Some(kind),
                "100 entries over {}",
                last + 1
            );
            assert_eq!(map.get(last), Some(&1));
            assert_eq!(map.get(99), None);
        }
        let mut map: IdMap<u64> = IdMap::new();
        map.page_out();
        for k in 0..33 {
            map.insert(k * 1_000, k as u64);
        }
        assert_eq!(kind_of(&map, 0), Some(2), "33 entries are ranked");
        for k in 0..16 {
            map.remove(k * 1_000);
        }
        assert_eq!(kind_of(&map, 16_000), Some(2), "17 entries stay ranked");
        map.remove(16_000);
        assert_eq!(kind_of(&map, 17_000), Some(1), "16 entries turn short");
        for k in 0..16 {
            map.insert(k * 1_000, k as u64);
        }
        assert_eq!(kind_of(&map, 0), Some(1), "32 entries stay short");
        map.insert(16_000, 16);
        assert_eq!(kind_of(&map, 0), Some(2), "33 entries turn ranked");
        let held: Vec<(usize, u64)> = (0..33).map(|k| (k * 1_000, k as u64)).collect();
        assert_eq!(entries(&map), held);
    }

    /// A page fed entries at either limit changes kind at most once, in
    /// increasing and in scrambled offsets: insertions and removals in turn
    /// about 32 and 33 entries, from a short page, and about 16 and 17, from
    /// a ranked one.
    #[test]
    fn a_page_fed_at_the_short_limit_changes_kind_at_most_once() {
        let offsets: Vec<usize> = (0..300).map(|k| k * 211).collect();
        for seed in [None, Some(5u64), Some(6)] {
            let feed = match seed {
                Some(seed) => scrambled(&offsets, seed),
                None => offsets.clone(),
            };
            let anchor = feed[0];
            let mut map: IdMap<u64> = IdMap::new();
            map.page_out();
            for &offset in &feed[..32] {
                map.insert(offset, 1);
            }
            let mut kinds = vec![kind_of(&map, anchor)];
            for &offset in &feed[32..232] {
                map.insert(offset, 2);
                kinds.push(kind_of(&map, anchor));
                map.remove(offset);
                kinds.push(kind_of(&map, anchor));
            }
            let changes = kinds.windows(2).filter(|pair| pair[0] != pair[1]).count();
            assert_eq!(kinds[0], Some(1), "{seed:?}");
            assert!(
                changes <= 1,
                "about 32 and 33 from {seed:?}: {changes} changes"
            );

            for &offset in &feed[1..16] {
                map.remove(offset);
            }
            assert_eq!(map.len(), 17);
            let mut kinds = vec![kind_of(&map, anchor)];
            for &offset in &feed[232..300] {
                map.remove(feed[16]);
                kinds.push(kind_of(&map, anchor));
                map.insert(feed[16], 3);
                kinds.push(kind_of(&map, anchor));
                map.insert(offset, 4);
                kinds.push(kind_of(&map, anchor));
                map.remove(offset);
                kinds.push(kind_of(&map, anchor));
            }
            let changes = kinds.windows(2).filter(|pair| pair[0] != pair[1]).count();
            assert_eq!(kinds[0], Some(2), "{seed:?}");
            assert!(
                changes <= 1,
                "about 16 and 17 from {seed:?}: {changes} changes"
            );
            assert_eq!(map.len(), 17);
        }
    }

    /// The bytes of each settled page and their sum, with the headers, the
    /// table and the paged form's block, as the formulas give them.
    fn costs_its_formula<T: Vacant + Clone>(value: T) {
        let e = std::mem::size_of::<T>();
        assert!(
            std::mem::size_of::<Page<T>>() <= 64,
            "a page's header holds the same fields in every kind"
        );
        let mut map: IdMap<T> = IdMap::new();
        map.page_out();
        for offset in (0..1_000).step_by(2) {
            map.insert(offset, value.clone());
        }
        let short: Vec<usize> = (0..20).map(|k| PAGE_IDS + k * 3_001).collect();
        for &id in &short {
            map.insert(id, value.clone());
        }
        let ranked: Vec<usize> = (0..1_000)
            .map(|k| 2 * PAGE_IDS + k * 13 + k % 3)
            .chain((0..64).map(|k| 2 * PAGE_IDS + 40_000 + k))
            .collect();
        for &id in &ranked {
            map.insert(id, value.clone());
        }
        map.settle();
        let groups = {
            let mut groups: Vec<usize> = ranked.iter().map(|&id| (id & OFFSET_MASK) >> 6).collect();
            groups.dedup();
            groups.len()
        };
        let pages = map.paged.as_deref().unwrap();
        let kinds: Vec<u8> = (0..3)
            .map(|page| kind_of(&map, page * PAGE_IDS).unwrap())
            .collect();
        assert_eq!(kinds, vec![0, 1, 2]);
        let formulas = [
            e * 999,
            (2 + e) * short.len(),
            160 + 10 * groups + e * ranked.len(),
        ];
        let bytes: Vec<usize> = pages.pages.iter().map(Page::heap_bytes).collect();
        assert_eq!(bytes, formulas, "{e}-byte entries, {groups} groups");
        assert_eq!(
            map.heap_bytes(),
            std::mem::size_of::<Pages<T>>()
                + pages.table.capacity() * 4
                + pages.pages.capacity() * std::mem::size_of::<Page<T>>()
                + formulas.iter().sum::<usize>()
        );
        assert_eq!(map.spare_bytes(), 0);
    }

    /// A settled dense page costs `e s` bytes, a short page `(2 + e) n` and
    /// a ranked page `160 + 10 G + e n`, `G` being the groups of 64 offsets
    /// it holds an entry in, for entries of four and eight bytes.
    #[test]
    fn each_page_kind_costs_its_formula() {
        costs_its_formula(7u32);
        costs_its_formula(7u64);
        costs_its_formula(Some(Box::new(7u8)));
    }
}
