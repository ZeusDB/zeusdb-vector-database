//! A set of internal ids, one bit each while the ids are dense and in pages
//! once they are sparse.
//!
//! # Flat
//!
//! One bit per internal id in a vector of words, as the set has always been
//! held: the same growth, the same capacity, the same reads. Every operation
//! below runs the code it ran before on a flat set, and a flat set never
//! changes form through [`Bitmap::insert`], [`Bitmap::remove`] or the
//! algebra.
//!
//! # Paged
//!
//! The id range in pages of [`PAGE_IDS`] ids, as [`crate::idmap`] lays it
//! out: a table of four bytes a page of the range naming each page's place in
//! an arena kept in page order. A page holds its ids as words or as its
//! offsets in increasing order. A page of offsets turns to words once its
//! span is at most [`BIT_PAGE_RATIO`] times its ids, where the words cost no
//! more than two bytes an id, and a page of words turns to offsets once its
//! span passes twice that, so a page between the two keeps its kind.
//!
//! # Which sets page out
//!
//! A record set, being a set that holds every record its owner holds, pages
//! out on its own count through [`Bitmap::hold`] and [`Bitmap::release`]: a
//! growth past its allocation, its reservation, a planned run and one page
//! stays flat while its extent is at most [`BIT_RATIO`] times its ids, and a
//! removal pages it out where it holds fewer than one id in twice that. A set
//! that holds part of its owner's records says nothing about its owner's
//! range by its own count, so it takes its owner's form instead: its owner
//! builds it with [`Bitmap::paged`] or calls [`Bitmap::page_out`].
//!
//! # The algebra
//!
//! Two flat sets combine word by word as they always have. A paged set
//! combines page by page, each page through a scratch of one page's words,
//! and the result is paged.

use crate::idmap::PAGE_IDS;

/// Bits of an internal id below its page number.
const PAGE_SHIFT: u32 = PAGE_IDS.trailing_zeros();

/// Words one dense page holds at most.
const PAGE_WORDS: usize = PAGE_IDS / 64;

/// A table entry that names no page.
const NO_PAGE: u32 = u32::MAX;

/// The extent in ids at or below which a record set never pages out.
pub(crate) const BIT_FLOOR: usize = PAGE_IDS;

/// Ids of its extent a flat record set may hold for each id when it grows.
pub(crate) const BIT_RATIO: usize = 64;

/// Bits of its span a page held as words may hold for each id.
pub(crate) const BIT_PAGE_RATIO: usize = 16;

/// The extent in ids of the words that reach `ids` ids, being `ids` rounded
/// up to a whole number of words. The growth and removal rules read a flat
/// set's extent from its words, so a reservation and a plan are held in the
/// same unit.
fn whole_words(ids: usize) -> usize {
    ids.div_ceil(64).saturating_mul(64)
}

/// One page of a paged set.
#[derive(Clone)]
struct BitPage {
    /// Which page of the id range this is.
    number: u32,
    /// Ids the page holds, above zero for every page in the arena.
    held: u32,
    /// Whether the page holds words rather than offsets.
    dense: bool,
    /// A dense page's words, the first `words.len() * 64` offsets.
    words: Vec<u64>,
    /// A sparse page's offsets, strictly increasing.
    offsets: Vec<u16>,
}

impl BitPage {
    fn new(number: u32) -> Self {
        BitPage {
            number,
            held: 0,
            dense: false,
            words: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn base(&self) -> usize {
        (self.number as usize) << PAGE_SHIFT
    }

    /// Words a dense form of the page would hold, being up to its largest
    /// offset.
    fn span_words(&self) -> usize {
        if self.dense {
            self.words
                .iter()
                .rposition(|&word| word != 0)
                .map_or(0, |at| at + 1)
        } else {
            self.offsets
                .last()
                .map_or(0, |&offset| offset as usize / 64 + 1)
        }
    }

    /// Whether a page of `held` ids over `words` words turns to words.
    fn dense_for(words: usize, held: usize) -> bool {
        words * 64 <= BIT_PAGE_RATIO.saturating_mul(held)
    }

    /// Whether a page of words holding `held` ids over `words` words stays
    /// words, which is within twice the ratio a page of offsets turns to
    /// words at.
    fn stays_dense(words: usize, held: usize) -> bool {
        words * 64 <= (2 * BIT_PAGE_RATIO).saturating_mul(held)
    }

    fn contains(&self, offset: usize) -> bool {
        if self.dense {
            self.words
                .get(offset >> 6)
                .is_some_and(|word| word >> (offset & 63) & 1 == 1)
        } else {
            self.offsets.binary_search(&(offset as u16)).is_ok()
        }
    }

    /// Put an offset in, reporting whether it was new. A page of words that
    /// would grow past twice its ratio turns to offsets first, and a page of
    /// offsets whose span the ratio covers after the insert turns to words.
    fn insert(&mut self, offset: usize) -> bool {
        let index = offset >> 6;
        if self.dense
            && index >= self.words.len()
            && !Self::stays_dense(index + 1, self.held as usize + 1)
        {
            self.make_sparse();
        }
        let added = if self.dense {
            if index >= self.words.len() {
                self.words.resize(index + 1, 0);
            }
            let mask = 1u64 << (offset & 63);
            let was = self.words[index] & mask != 0;
            self.words[index] |= mask;
            !was
        } else {
            match self.offsets.binary_search(&(offset as u16)) {
                Ok(_) => false,
                Err(at) => {
                    self.offsets.insert(at, offset as u16);
                    true
                }
            }
        };
        if added {
            self.held += 1;
        }
        if !self.dense && Self::dense_for(self.span_words(), self.held as usize) {
            self.make_dense();
        }
        added
    }

    /// Take an offset out, reporting whether it was there.
    fn remove(&mut self, offset: usize) -> bool {
        let removed = if self.dense {
            match self.words.get_mut(offset >> 6) {
                Some(word) => {
                    let mask = 1u64 << (offset & 63);
                    let was = *word & mask != 0;
                    *word &= !mask;
                    was
                }
                None => false,
            }
        } else {
            match self.offsets.binary_search(&(offset as u16)) {
                Ok(at) => {
                    self.offsets.remove(at);
                    true
                }
                Err(_) => false,
            }
        };
        if removed {
            self.held -= 1;
            if self.dense
                && self.held > 0
                && !Self::stays_dense(self.words.len(), self.held as usize)
            {
                self.make_sparse();
            }
        }
        removed
    }

    fn make_sparse(&mut self) {
        let mut offsets = Vec::with_capacity(self.held as usize);
        for (index, &word) in self.words.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                offsets.push((index * 64 + word.trailing_zeros() as usize) as u16);
                word &= word - 1;
            }
        }
        self.words = Vec::new();
        self.offsets = offsets;
        self.dense = false;
    }

    fn make_dense(&mut self) {
        let mut words = vec![0u64; self.span_words()];
        for &offset in &self.offsets {
            words[offset as usize >> 6] |= 1u64 << (offset & 63);
        }
        self.words = words;
        self.offsets = Vec::new();
        self.dense = true;
    }

    /// The kind the page's content gives, at its exact capacity.
    fn settle(&mut self) {
        let dense = Self::dense_for(self.span_words(), self.held as usize);
        if self.dense && !dense {
            self.make_sparse();
        } else if !self.dense && dense {
            self.make_dense();
        }
        if self.dense {
            let span = self.span_words();
            self.words.truncate(span);
        }
        self.words.shrink_to_fit();
        self.offsets.shrink_to_fit();
    }

    /// The page as one page of words.
    fn expand(&self, into: &mut [u64; PAGE_WORDS]) {
        into.fill(0);
        if self.dense {
            into[..self.words.len()].copy_from_slice(&self.words);
        } else {
            for &offset in &self.offsets {
                into[offset as usize >> 6] |= 1u64 << (offset & 63);
            }
        }
    }

    /// A page built from one page of words, in the kind its content gives,
    /// or `None` where it holds nothing.
    fn from_words(number: u32, words: &[u64; PAGE_WORDS]) -> Option<BitPage> {
        let held: usize = words.iter().map(|word| word.count_ones() as usize).sum();
        if held == 0 {
            return None;
        }
        let span = words
            .iter()
            .rposition(|&word| word != 0)
            .map_or(0, |at| at + 1);
        let mut page = BitPage {
            number,
            held: held as u32,
            dense: true,
            words: words[..span].to_vec(),
            offsets: Vec::new(),
        };
        if !Self::dense_for(span, held) {
            page.make_sparse();
        }
        Some(page)
    }

    fn for_each_while<F: FnMut(usize) -> bool>(&self, visit: &mut F) -> bool {
        let base = self.base();
        if self.dense {
            for (index, &word) in self.words.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    if !visit(base + index * 64 + word.trailing_zeros() as usize) {
                        return false;
                    }
                    word &= word - 1;
                }
            }
        } else {
            for &offset in &self.offsets {
                if !visit(base + offset as usize) {
                    return false;
                }
            }
        }
        true
    }

    fn heap_bytes(&self) -> usize {
        self.words.capacity() * 8 + self.offsets.capacity() * 2
    }
}

/// The paged form: the table and the arena.
#[derive(Clone, Default)]
struct BitPages {
    /// The arena index of each page of the id range, or [`NO_PAGE`].
    table: Vec<u32>,
    /// Every page that holds an id, in increasing page number.
    pages: Vec<BitPage>,
}

impl BitPages {
    fn page(&self, number: usize) -> Option<&BitPage> {
        let at = *self.table.get(number)?;
        (at != NO_PAGE).then(|| &self.pages[at as usize])
    }

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
        self.pages.insert(place, BitPage::new(page_number));
        for page in &self.pages[place + 1..] {
            self.table[page.number as usize] += 1;
        }
        self.table[number] = place as u32;
        place
    }

    /// Append a page built in increasing page order.
    fn push(&mut self, page: BitPage) {
        let number = page.number as usize;
        if number >= self.table.len() {
            self.table.resize(number + 1, NO_PAGE);
        }
        self.table[number] = self.pages.len() as u32;
        self.pages.push(page);
    }

    fn insert(&mut self, slot: usize) -> bool {
        let at = self.open(slot >> PAGE_SHIFT);
        self.pages[at].insert(slot & (PAGE_IDS - 1))
    }

    fn remove(&mut self, slot: usize) -> bool {
        let number = slot >> PAGE_SHIFT;
        let Some(&at) = self.table.get(number) else {
            return false;
        };
        if at == NO_PAGE {
            return false;
        }
        let at = at as usize;
        let removed = self.pages[at].remove(slot & (PAGE_IDS - 1));
        if removed && self.pages[at].held == 0 {
            self.pages.remove(at);
            self.table[number] = NO_PAGE;
            for page in &self.pages[at..] {
                self.table[page.number as usize] -= 1;
            }
        }
        removed
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
            + self.pages.capacity() * std::mem::size_of::<BitPage>()
            + self.pages.iter().map(BitPage::heap_bytes).sum::<usize>()
    }
}

/// One page of a set as the algebra reads it.
enum View<'a> {
    Empty,
    Words(&'a [u64]),
    Page(&'a BitPage),
}

impl View<'_> {
    fn expand(&self, into: &mut [u64; PAGE_WORDS]) {
        match self {
            View::Empty => into.fill(0),
            View::Words(words) => {
                into.fill(0);
                into[..words.len()].copy_from_slice(words);
            }
            View::Page(page) => page.expand(into),
        }
    }
}

/// A set of internal ids, one bit each. See the module.
#[derive(Clone, Default)]
pub struct Bitmap {
    /// Every word while the set is flat. Empty once it is paged.
    words: Vec<u64>,
    /// The pages, once the set is paged.
    paged: Option<Box<BitPages>>,
    /// Ids the set holds.
    held: usize,
    /// The extent in ids the words of a reservation reach, a whole number of
    /// words, at or below which a record set stays flat.
    reserved: usize,
    /// The extent in ids the words of a planned run reach, a whole number of
    /// words, at or below which a growth stays flat.
    planned: usize,
}

impl Bitmap {
    pub(crate) fn zeros(slots: usize) -> Self {
        Bitmap {
            words: vec![0; slots.div_ceil(64)],
            ..Bitmap::default()
        }
    }

    /// An empty set whose words already reach `slots`, so a test of any id
    /// below that reads a word rather than falling off the end. What a
    /// measurement of the bit test's cost wants, since the set that admits
    /// nothing and holds no words answers without a read.
    pub fn with_slots(slots: usize) -> Self {
        Self::zeros(slots)
    }

    /// An empty record set whose words reach `slots`, and which stays flat
    /// for as long as its extent is within them.
    pub fn reserved(slots: usize) -> Self {
        Bitmap {
            reserved: whole_words(slots),
            ..Self::zeros(slots)
        }
    }

    /// An empty set in the paged form, for a set that takes the form of an
    /// owner whose ids are sparse.
    pub fn paged() -> Self {
        Bitmap {
            paged: Some(Box::default()),
            ..Bitmap::default()
        }
    }

    /// An empty set in the form of `other`: words reaching `slots` where
    /// `other` is flat, the paged form where it is paged.
    pub(crate) fn blank_like(other: &Bitmap, slots: usize) -> Self {
        if other.is_flat() {
            Self::zeros(slots)
        } else {
            Self::paged()
        }
    }

    /// Whether the set is flat.
    pub fn is_flat(&self) -> bool {
        self.paged.is_none()
    }

    /// Put an id in a set built at its full size by [`Bitmap::zeros`], or in
    /// a paged set.
    #[inline]
    pub(crate) fn set(&mut self, slot: usize) {
        match self.paged.as_deref_mut() {
            None => {
                let word = &mut self.words[slot >> 6];
                let mask = 1u64 << (slot & 63);
                self.held += (*word & mask == 0) as usize;
                *word |= mask;
            }
            Some(pages) => {
                if pages.insert(slot) {
                    self.held += 1;
                }
            }
        }
    }

    /// Put an internal id in the set, growing the words to reach it. A flat
    /// set stays flat, as a set maintained beside a map always has.
    pub fn insert(&mut self, slot: usize) {
        match self.paged.as_deref_mut() {
            None => {
                let index = slot >> 6;
                if index >= self.words.len() {
                    self.words.resize(index + 1, 0);
                }
                let mask = 1u64 << (slot & 63);
                self.held += (self.words[index] & mask == 0) as usize;
                self.words[index] |= mask;
            }
            Some(pages) => {
                if pages.insert(slot) {
                    self.held += 1;
                }
            }
        }
    }

    /// Take an internal id out of the set. An id beyond the words was never in
    /// it, so there is nothing to clear.
    pub fn remove(&mut self, slot: usize) {
        match self.paged.as_deref_mut() {
            None => {
                if let Some(word) = self.words.get_mut(slot >> 6) {
                    let mask = 1u64 << (slot & 63);
                    self.held -= (*word & mask != 0) as usize;
                    *word &= !mask;
                }
            }
            Some(pages) => {
                if pages.remove(slot) {
                    self.held -= 1;
                }
            }
        }
    }

    /// Put a record's id in a record set. A flat set grows to reach it as
    /// [`Bitmap::insert`] does, unless the growth is past its allocation, its
    /// reservation, a planned run and one page and would leave it holding
    /// fewer than one id in [`BIT_RATIO`] of its extent, in which case it
    /// pages out first.
    pub fn hold(&mut self, slot: usize) {
        self.hold_with(slot, |words| words);
    }

    /// [`Bitmap::hold`], a flat set growing to the words `grow` gives for the
    /// words it needs.
    pub fn hold_with(&mut self, slot: usize, grow: impl FnOnce(usize) -> usize) {
        if self.paged.is_none() {
            let needed = (slot >> 6) + 1;
            if needed > self.words.len() {
                let extent = needed.saturating_mul(64);
                let stays = needed <= self.words.capacity()
                    || extent <= BIT_FLOOR.max(self.reserved).max(self.planned)
                    || extent <= BIT_RATIO.saturating_mul(self.held + 1);
                if stays {
                    self.words.resize(grow(needed).max(needed), 0);
                } else {
                    self.page_out();
                }
            }
        }
        self.set(slot);
    }

    /// Take a record's id out of a record set, paging a flat set out where its
    /// extent is past one page and its reservation and it holds fewer than
    /// one id in twice [`BIT_RATIO`] of its extent.
    pub fn release(&mut self, slot: usize) {
        self.remove(slot);
        if self.paged.is_none() {
            let extent = self.words.len().saturating_mul(64);
            if extent > BIT_FLOOR.max(self.reserved)
                && (2 * BIT_RATIO).saturating_mul(self.held) < extent
            {
                self.page_out();
            }
        }
    }

    /// Decide a record set's form for a run of insertions of `records` ids
    /// whose largest is `highest`, from the run's final shape rather than
    /// its order.
    pub fn plan(&mut self, records: usize, highest: usize) {
        if self.paged.is_some() {
            return;
        }
        let extent = whole_words(highest.saturating_add(1));
        let allocated = self.words.capacity().saturating_mul(64);
        if extent <= BIT_FLOOR.max(self.reserved).max(allocated)
            || extent <= BIT_RATIO.saturating_mul(self.held.saturating_add(records))
        {
            self.planned = self.planned.max(extent);
        } else {
            self.page_out();
        }
    }

    /// Move a flat set's ids into pages and free its words.
    pub fn page_out(&mut self) {
        if self.paged.is_some() {
            return;
        }
        let words = std::mem::take(&mut self.words);
        let mut pages = BitPages::default();
        let mut scratch = [0u64; PAGE_WORDS];
        for (number, chunk) in words.chunks(PAGE_WORDS).enumerate() {
            scratch.fill(0);
            scratch[..chunk.len()].copy_from_slice(chunk);
            let page_number = u32::try_from(number).expect("a page number fits in a u32");
            if let Some(page) = BitPage::from_words(page_number, &scratch) {
                pages.push(page);
            }
        }
        pages.table.shrink_to_fit();
        self.paged = Some(Box::new(pages));
    }

    /// Put each page in the kind its content gives, at its exact capacity.
    /// A flat set is left as it is.
    pub fn settle(&mut self) {
        if let Some(pages) = self.paged.as_deref_mut() {
            pages.settle();
        }
    }

    /// Empty the set, keeping its form.
    pub fn clear(&mut self) {
        match self.paged.as_deref_mut() {
            None => self.words.iter_mut().for_each(|word| *word = 0),
            Some(pages) => *pages = BitPages::default(),
        }
        self.held = 0;
    }

    /// Whether an internal id is in the set.
    ///
    /// Total over every `usize`, so the traversal predicate can ask it about a
    /// node the store has never heard of and get `false` rather than a panic.
    #[inline]
    pub fn contains(&self, slot: usize) -> bool {
        match self.words.get(slot >> 6) {
            Some(word) => word >> (slot & 63) & 1 == 1,
            None => self.paged_contains(slot),
        }
    }

    #[inline(never)]
    fn paged_contains(&self, slot: usize) -> bool {
        self.paged.as_deref().is_some_and(|pages| {
            pages
                .page(slot >> PAGE_SHIFT)
                .is_some_and(|page| page.contains(slot & (PAGE_IDS - 1)))
        })
    }

    pub fn count(&self) -> usize {
        match self.paged.as_deref() {
            None => self.words.iter().map(|w| w.count_ones() as usize).sum(),
            Some(pages) => pages.pages.iter().map(|page| page.held as usize).sum(),
        }
    }

    /// The page numbers the set's ids sit in, in increasing order.
    fn page_numbers(&self) -> Vec<usize> {
        match self.paged.as_deref() {
            None => (0..self.words.len().div_ceil(PAGE_WORDS)).collect(),
            Some(pages) => pages
                .pages
                .iter()
                .map(|page| page.number as usize)
                .collect(),
        }
    }

    /// One page of the set as the algebra reads it.
    fn view(&self, number: usize) -> View<'_> {
        match self.paged.as_deref() {
            None => {
                let start = number.saturating_mul(PAGE_WORDS);
                if start >= self.words.len() {
                    View::Empty
                } else {
                    let end = (start + PAGE_WORDS).min(self.words.len());
                    View::Words(&self.words[start..end])
                }
            }
            Some(pages) => pages.page(number).map_or(View::Empty, View::Page),
        }
    }

    /// How many internal ids are in both sets, by a walk over the shorter
    /// word range. A postings index counts the records a filter admits
    /// among those it holds with this, rather than by testing each admitted
    /// id.
    pub fn count_and(&self, other: &Bitmap) -> usize {
        if self.is_flat() && other.is_flat() {
            return self
                .words
                .iter()
                .zip(&other.words)
                .map(|(mine, theirs)| (mine & theirs).count_ones() as usize)
                .sum();
        }
        let mut mine = [0u64; PAGE_WORDS];
        let mut theirs = [0u64; PAGE_WORDS];
        let mut total = 0;
        for number in self.page_numbers() {
            let other_view = other.view(number);
            if matches!(other_view, View::Empty) {
                continue;
            }
            self.view(number).expand(&mut mine);
            other_view.expand(&mut theirs);
            total += mine
                .iter()
                .zip(&theirs)
                .map(|(a, b)| (a & b).count_ones() as usize)
                .sum::<usize>();
        }
        total
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self.paged.as_deref() {
            None => self.words.iter().all(|word| *word == 0),
            Some(pages) => pages.pages.is_empty(),
        }
    }

    /// Recount the ids after the words changed wholesale.
    fn recount(&mut self) {
        self.held = self.count();
    }

    /// Combine with `other` page by page into a paged result, through `op`
    /// over one page's words, over the page numbers `numbers` gives.
    fn combine(
        numbers: Vec<usize>,
        left: &Bitmap,
        right: &Bitmap,
        op: impl Fn(u64, u64) -> u64,
    ) -> Bitmap {
        let mut out = BitPages::default();
        let mut a = [0u64; PAGE_WORDS];
        let mut b = [0u64; PAGE_WORDS];
        for number in numbers {
            left.view(number).expand(&mut a);
            right.view(number).expand(&mut b);
            for (x, y) in a.iter_mut().zip(&b) {
                *x = op(*x, *y);
            }
            let page_number = u32::try_from(number).expect("a page number fits in a u32");
            if let Some(page) = BitPage::from_words(page_number, &a) {
                out.push(page);
            }
        }
        let mut set = Bitmap {
            paged: Some(Box::new(out)),
            ..Bitmap::default()
        };
        set.recount();
        set
    }

    pub(crate) fn intersect(&mut self, other: &Bitmap) {
        if self.is_flat() && other.is_flat() {
            for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
                *mine &= *theirs;
            }
            self.recount();
            return;
        }
        let numbers = self.page_numbers();
        *self = Self::combine(numbers, self, other, |a, b| a & b);
    }

    pub(crate) fn union(&mut self, other: &Bitmap) {
        if self.is_flat() && other.is_flat() {
            for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
                *mine |= *theirs;
            }
            self.recount();
            return;
        }
        let mut numbers = self.page_numbers();
        numbers.extend(other.page_numbers());
        numbers.sort_unstable();
        numbers.dedup();
        *self = Self::combine(numbers, self, other, |a, b| a | b);
    }

    /// `live` without `self`.
    ///
    /// The complement is taken within the live set rather than over the whole
    /// word range, so a `$not` never selects a slot that holds no record. Every
    /// other bitmap here is already a subset of the live set, because a slot
    /// with no record carries the absent value in every column and no leaf
    /// matches that.
    pub(crate) fn complement_within(&self, live: &Bitmap) -> Bitmap {
        if self.is_flat() && live.is_flat() {
            let mut out = Bitmap {
                words: vec![0; self.words.len()],
                ..Bitmap::default()
            };
            for (index, word) in out.words.iter_mut().enumerate() {
                *word = live.words.get(index).copied().unwrap_or(0) & !self.words[index];
            }
            out.recount();
            return out;
        }
        Self::combine(live.page_numbers(), live, self, |a, b| a & !b)
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
        match self.paged.as_deref() {
            None => {
                for (index, mut word) in self.words.iter().copied().enumerate() {
                    while word != 0 {
                        if !visit(index * 64 + word.trailing_zeros() as usize) {
                            return;
                        }
                        word &= word - 1;
                    }
                }
            }
            Some(pages) => {
                for page in &pages.pages {
                    if !page.for_each_while(&mut visit) {
                        return;
                    }
                }
            }
        }
    }

    /// Bytes the set asks the allocator for.
    pub fn heap_bytes(&self) -> usize {
        self.words.capacity() * 8 + self.paged.as_deref().map_or(0, BitPages::heap_bytes)
    }

    /// A flat set whose words reach `slots`, holding this flat set's ids
    /// below them, which is the copy of its live set the column store hands
    /// a filter.
    pub(crate) fn prefix(&self, slots: usize) -> Bitmap {
        let mut out = Self::zeros(slots);
        let shared = out.words.len().min(self.words.len());
        out.words[..shared].copy_from_slice(&self.words[..shared]);
        out.recount();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn ids(set: &Bitmap) -> Vec<usize> {
        let mut out = Vec::new();
        set.for_each(|slot| out.push(slot));
        out
    }

    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A flat set is the words it always was: the same growth, the same
    /// capacity, the same bytes, whatever is held or released.
    #[test]
    fn a_flat_set_is_the_words_it_was() {
        let mut held = Bitmap::default();
        let mut plain = Bitmap::default();
        let mut words: Vec<u64> = Vec::new();
        for slot in (1..50_001usize).filter(|slot| slot % 7 != 3) {
            held.hold(slot);
            plain.insert(slot);
            let index = slot >> 6;
            if index >= words.len() {
                words.resize(index + 1, 0);
            }
            words[index] |= 1 << (slot & 63);
        }
        for set in [&held, &plain] {
            assert!(set.is_flat());
            assert_eq!(set.words, words);
            assert_eq!(set.heap_bytes(), words.capacity() * 8);
            assert_eq!(set.count(), set.held);
        }
        for slot in (1..50_001usize).step_by(5) {
            held.release(slot);
        }
        assert!(held.is_flat(), "within one page a record set stays flat");
    }

    /// A record set over scattered ids pages out and costs two bytes an id
    /// and a header a page beside the table, and answers every question the
    /// model answers.
    #[test]
    fn a_scattered_record_set_costs_what_its_ids_cost() {
        let mut set = Bitmap::default();
        let mut model = BTreeSet::new();
        for k in 0..2_000usize {
            let slot = k * 30_011 + 5;
            set.hold(slot);
            model.insert(slot);
        }
        assert!(!set.is_flat());
        set.settle();
        let pages = set.paged.as_deref().unwrap();
        let bound = std::mem::size_of::<BitPages>()
            + pages.table.len() * 4
            + pages.pages.len() * std::mem::size_of::<BitPage>()
            + 2_000 * 2;
        assert!(
            set.heap_bytes() <= bound,
            "{} > {}",
            set.heap_bytes(),
            bound
        );
        assert_eq!(ids(&set), model.iter().copied().collect::<Vec<_>>());
        assert_eq!(set.count(), 2_000);
        for probe in [0usize, 5, 6, 30_016, 30_017, 1 << 40] {
            assert_eq!(set.contains(probe), model.contains(&probe), "{probe}");
        }
    }

    /// The model test: holds, releases, inserts and removes over ids drawn to
    /// cross every rule, against a sorted set.
    #[test]
    fn the_set_answers_what_a_sorted_set_answers_through_every_form() {
        for seed in 1..=6u64 {
            let mut stream = Stream(seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut set = Bitmap::default();
            let mut model = BTreeSet::new();
            for step in 0..8_000usize {
                let draw = stream.next();
                let slot = match draw % 6 {
                    0 | 1 => step % 70_000,
                    2 => 1_000_000 + (draw as usize >> 9) % 4_000,
                    3 => (draw as usize >> 9) % (1 << 24),
                    _ => 300_000 + (draw as usize >> 9) % 200,
                };
                if draw.is_multiple_of(4) {
                    set.release(slot);
                    model.remove(&slot);
                } else {
                    set.hold(slot);
                    model.insert(slot);
                }
                if step % 211 == 0 {
                    assert_eq!(ids(&set), model.iter().copied().collect::<Vec<_>>());
                    assert_eq!(set.count(), model.len());
                    assert_eq!(set.held, model.len());
                    for &probe in [slot, slot + 1, slot.saturating_sub(1)].iter() {
                        assert_eq!(set.contains(probe), model.contains(&probe));
                    }
                }
                if step % 1_999 == 0 {
                    set.settle();
                }
            }
            assert_eq!(ids(&set), model.iter().copied().collect::<Vec<_>>());
        }
    }

    /// A set over `slots`: flat at the full size of `range`, as every set the
    /// column store combines is, or paged.
    fn set_of(slots: &[usize], paged: bool, range: usize) -> Bitmap {
        let mut set = if paged {
            Bitmap::paged()
        } else {
            Bitmap::zeros(range)
        };
        for &slot in slots {
            set.set(slot);
        }
        set
    }

    /// The algebra over flat, paged and mixed operands gives what the sorted
    /// sets give, and two flat operands give a flat result.
    #[test]
    fn the_algebra_answers_in_every_form() {
        let mut stream = Stream(0x1234_5678);
        let draw = |stream: &mut Stream, n: usize, range: usize| -> Vec<usize> {
            (0..n).map(|_| (stream.next() as usize) % range).collect()
        };
        for range in [5_000usize, 200_000, 3_000_000] {
            let a = draw(&mut stream, 3_000, range);
            let b = draw(&mut stream, 3_000, range);
            let live: Vec<usize> = a.iter().chain(&b).copied().filter(|s| s % 3 != 0).collect();
            let model = |v: &[usize]| v.iter().copied().collect::<BTreeSet<usize>>();
            let (ma, mb, ml) = (model(&a), model(&b), model(&live));
            for (pa, pb) in [(false, false), (true, true), (false, true), (true, false)] {
                let both = (pa, pb);
                let mut and = set_of(&a, pa, range);
                and.intersect(&set_of(&b, pb, range));
                assert_eq!(
                    ids(&and),
                    ma.intersection(&mb).copied().collect::<Vec<_>>(),
                    "{both:?}"
                );
                assert_eq!(and.count(), ma.intersection(&mb).count());
                assert_eq!(and.held, and.count());
                let mut or = set_of(&a, pa, range);
                or.union(&set_of(&b, pb, range));
                assert_eq!(
                    ids(&or),
                    ma.union(&mb).copied().collect::<Vec<_>>(),
                    "{both:?}"
                );
                let not = set_of(&a, pa, range).complement_within(&set_of(&live, pb, range));
                assert_eq!(
                    ids(&not),
                    ml.difference(&ma).copied().collect::<Vec<_>>(),
                    "{both:?}"
                );
                assert_eq!(
                    set_of(&a, pa, range).count_and(&set_of(&b, pb, range)),
                    ma.intersection(&mb).count()
                );
                assert_eq!(set_of(&a, pa, range).is_empty(), ma.is_empty());
                if !pa && !pb {
                    assert!(and.is_flat() && or.is_flat() && not.is_flat());
                }
                let live_set = set_of(&live, pb, range);
                let copy = if live_set.is_flat() {
                    live_set.prefix(range)
                } else {
                    live_set.clone()
                };
                assert_eq!(ids(&copy), ml.iter().copied().collect::<Vec<_>>());
            }
        }
        let empty = Bitmap::paged();
        assert!(empty.is_empty());
        assert_eq!(empty.count(), 0);
        assert!(!empty.contains(0));
    }

    /// A planned run lays out from its final shape whatever its order.
    #[test]
    fn a_planned_record_set_lays_out_the_same_in_any_order() {
        let slots: Vec<usize> = (0..5_000usize).map(|k| k * 1_009 + 3).collect();
        let highest = *slots.iter().max().unwrap();
        let mut scrambled = slots.clone();
        let mut stream = Stream(9);
        for i in (1..scrambled.len()).rev() {
            let j = (stream.next() as usize) % (i + 1);
            scrambled.swap(i, j);
        }
        let mut sets = Vec::new();
        for order in [&slots, &scrambled] {
            let mut set = Bitmap::default();
            set.plan(order.len(), highest);
            for &slot in order.iter() {
                set.hold(slot);
            }
            set.settle();
            sets.push(set);
        }
        assert!(!sets[0].is_flat() && !sets[1].is_flat());
        assert_eq!(ids(&sets[0]), ids(&sets[1]));
        assert_eq!(sets[0].heap_bytes(), sets[1].heap_bytes());
    }

    /// A page changes kind as its ids come and go: words while its span is
    /// at most sixteen bits an id, offsets once a removal leaves it a
    /// thirty-second full, words again once it fills, and freed once empty.
    #[test]
    fn a_page_changes_kind_as_its_ids_come_and_go() {
        let mut set = Bitmap::paged();
        let base = 2 * PAGE_IDS;
        for offset in 0..8_192usize {
            set.insert(base + offset);
        }
        set.insert(9);
        let kind = |set: &Bitmap, number: usize| {
            let pages = set.paged.as_deref().unwrap();
            pages.page(number).map(|page| (page.dense, page.held))
        };
        assert_eq!(kind(&set, 2), Some((true, 8_192)));
        for offset in 0..8_000usize {
            set.remove(base + offset);
        }
        assert_eq!(
            kind(&set, 2),
            Some((false, 192)),
            "a thirty-second full is sparse"
        );
        for offset in 0..8_000usize {
            set.insert(base + offset);
        }
        assert_eq!(kind(&set, 2), Some((true, 8_192)), "refilled is dense");
        for offset in 0..8_192usize {
            set.remove(base + offset);
        }
        assert_eq!(kind(&set, 2), None, "an empty page is freed");
        assert_eq!(ids(&set), vec![9]);
        assert_eq!(set.count(), 1);
        assert_eq!(set.held, 1);
    }

    /// A page fed ids in increasing order whose gaps sit at its ratio or
    /// straddle it, from every start in a word, keeps its kind rather than
    /// changing it at every few ids.
    #[test]
    fn a_page_fed_gaps_about_its_ratio_keeps_its_kind() {
        let ratio = BIT_PAGE_RATIO;
        for gaps in [
            [ratio, ratio],
            [ratio + 1, ratio - 1],
            [ratio - 1, ratio + 1],
        ] {
            for start in 0..64usize {
                let mut set = Bitmap::paged();
                let mut offset = start;
                let mut changes = 0;
                let mut last = None;
                for k in 0..3_000usize {
                    set.insert(offset);
                    let dense = set.paged.as_deref().unwrap().pages[0].dense;
                    changes += usize::from(last.is_some_and(|was| was != dense));
                    last = Some(dense);
                    offset += gaps[k % 2];
                }
                assert!(
                    changes <= 1,
                    "gaps {gaps:?} from {start}, {changes} changes of kind"
                );
                assert_eq!(set.count(), 3_000);
            }
        }
    }

    /// A record set reserved past one page stays flat as records leave it
    /// for as long as its words are the reservation's, whatever the
    /// reservation's remainder over a word, and pages out once it grows past
    /// them holding few.
    #[test]
    fn a_reserved_record_set_stays_flat_within_its_reservation() {
        for reserved in [70_000usize, 100_001, 100_032] {
            let mut set = Bitmap::reserved(reserved);
            for slot in 1..=500usize {
                set.hold(slot);
            }
            for slot in 1..500usize {
                set.release(slot);
            }
            assert!(set.is_flat(), "reserved {reserved}");
            assert_eq!(ids(&set), vec![500]);
            set.hold(4 * reserved);
            assert!(!set.is_flat(), "reserved {reserved}, grown past it");
            assert_eq!(ids(&set), vec![500, 4 * reserved]);
        }
    }

    /// A planned run of dense ids stays flat in any order, its highest id
    /// first included, as it does in increasing order.
    #[test]
    fn a_planned_dense_run_stays_flat_in_any_order() {
        for highest in [70_000usize, 100_000, 100_031] {
            let ascending: Vec<usize> = (0..=highest).collect();
            let descending: Vec<usize> = (0..=highest).rev().collect();
            for order in [&ascending, &descending] {
                let mut set = Bitmap::default();
                set.plan(order.len(), highest);
                for &slot in order.iter() {
                    set.hold(slot);
                }
                set.settle();
                assert!(set.is_flat(), "highest {highest}");
                assert_eq!(set.count(), highest + 1);
            }
        }
    }
}
