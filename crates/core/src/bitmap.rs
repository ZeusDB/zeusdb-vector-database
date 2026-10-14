//! A set of positions, one bit each.
//!
//! One home for the set every structure uses: the column store's live set and
//! every filter's bitmap, the id store's live set, and the sparse space's dead
//! and live sets. Each is a vector of words over positions, read with the same
//! loops.

// ============================================================================
// THE BITMAP
// ============================================================================

/// A set of positions, one bit each.
///
/// Sized to the store's slot count rather than to the record count. At
/// 100,000 records that is 12.5 kilobytes, so a filter of a dozen leaves
/// allocates less than the page it returns.
#[derive(Clone, Default)]
pub struct Bitmap {
    /// One bit a position. The column store grows and clears the words of
    /// its live set in place.
    pub(crate) words: Vec<u64>,
}

impl Bitmap {
    pub(crate) fn zeros(slots: usize) -> Self {
        Bitmap {
            words: vec![0; slots.div_ceil(64)],
        }
    }

    /// An empty set whose words already reach `slots`, so a test of any
    /// position below that reads a word rather than falling off the end. What
    /// a measurement of the bit test's cost wants, since the set that admits
    /// nothing and holds no words answers without a read.
    pub fn with_slots(slots: usize) -> Self {
        Self::zeros(slots)
    }

    #[inline]
    pub(crate) fn set(&mut self, slot: usize) {
        self.words[slot >> 6] |= 1u64 << (slot & 63);
    }

    /// Put a position in the set, growing the words to reach it.
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

    /// Take a position out of the set. A position beyond the words was never
    /// in it, so there is nothing to clear.
    pub fn remove(&mut self, slot: usize) {
        if let Some(word) = self.words.get_mut(slot >> 6) {
            *word &= !(1u64 << (slot & 63));
        }
    }

    /// Empty the set, keeping its allocation for the positions about to refill
    /// it.
    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|word| *word = 0);
    }

    /// A set of exactly the positions below `count`, in words that reach
    /// `count` or `floor`, whichever is further, and no further. What a
    /// close-up built off to the side holds, being the set
    /// [`Bitmap::hold_first`] leaves.
    pub(crate) fn first(count: usize, floor: usize) -> Bitmap {
        let mut set = Bitmap::default();
        set.hold_first(count, floor);
        set
    }

    /// Make the set exactly the positions below `count`, in words that reach
    /// `count` or `floor`, whichever is further, and no further, and give
    /// back the words past them. What a close-up leaves, since it moves every
    /// record held to the front, and `floor` is the reservation the set was
    /// made with.
    pub(crate) fn hold_first(&mut self, count: usize, floor: usize) {
        let needed = count.max(floor).div_ceil(64);
        self.words.resize(needed, 0);
        self.words.shrink_to_fit();
        let full = count / 64;
        let partial = count % 64;
        for (index, word) in self.words.iter_mut().enumerate() {
            *word = if index < full {
                u64::MAX
            } else if index == full && partial != 0 {
                (1u64 << partial) - 1
            } else {
                0
            };
        }
    }

    /// Whether a position is in the set.
    ///
    /// Total over every `usize`, so the traversal predicate can ask it about a
    /// position the store has never heard of, the empty marker included, and
    /// get `false` rather than a panic.
    #[inline]
    pub fn contains(&self, slot: usize) -> bool {
        self.words
            .get(slot >> 6)
            .is_some_and(|word| word >> (slot & 63) & 1 == 1)
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// How many positions are in both sets, by a walk over the shorter word
    /// range. A postings index counts the records a filter admits among those
    /// it holds with this, rather than by testing each admitted position.
    pub fn count_and(&self, other: &Bitmap) -> usize {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(mine, theirs)| (mine & theirs).count_ones() as usize)
            .sum()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    pub(crate) fn intersect(&mut self, other: &Bitmap) {
        for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
            *mine &= *theirs;
        }
    }

    pub(crate) fn union(&mut self, other: &Bitmap) {
        for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
            *mine |= *theirs;
        }
    }

    /// `live` without `self`.
    ///
    /// The complement is taken within the live set rather than over the whole
    /// word range, so a `$not` never selects a slot that holds no record. Every
    /// other bitmap the column store builds is already a subset of the live
    /// set, because a slot with no record carries no value in any column and
    /// no leaf matches that.
    pub(crate) fn complement_within(&self, live: &Bitmap) -> Bitmap {
        let mut out = Bitmap {
            words: vec![0; self.words.len()],
        };
        for (index, word) in out.words.iter_mut().enumerate() {
            *word = live.words.get(index).copied().unwrap_or(0) & !self.words[index];
        }
        out
    }

    /// Every position in the set, in increasing order.
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

    /// Bytes of the words past those that reach `slots`, whether the words
    /// hold them or only their capacity does. A store whose set holds no
    /// position past `slots` reserved them, and no record has been written
    /// into them.
    pub(crate) fn reserved_past(&self, slots: usize) -> usize {
        (self.words.capacity() - self.words.len().min(slots.div_ceil(64))) * 8
    }

    /// Give back the words [`Bitmap::reserved_past`] prices, reporting the
    /// bytes released. The set holds no position past `slots`, so it is the
    /// same set.
    pub(crate) fn shrink_past(&mut self, slots: usize) -> usize {
        let before = self.heap_bytes();
        self.words.truncate(slots.div_ceil(64));
        self.words.shrink_to_fit();
        before - self.heap_bytes()
    }

    /// The words and their capacity, for a test that holds two sets to be
    /// the same words. Absent from every build that is not a test.
    #[cfg(any(test, feature = "test-support"))]
    pub fn words_held(&self) -> (&[u64], usize) {
        (&self.words, self.words.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::Position;
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

    /// A set is its words. An insert grows them as a vector of words grows,
    /// to the same capacity and the same bytes, and a removal clears its bit
    /// and keeps them.
    #[test]
    fn a_set_is_the_words_its_inserts_and_removals_write() {
        let mut set = Bitmap::default();
        let mut words: Vec<u64> = Vec::new();
        for slot in (1..50_001usize).filter(|slot| slot % 7 != 3) {
            set.insert(slot);
            let index = slot >> 6;
            if index >= words.len() {
                words.resize(index + 1, 0);
            }
            words[index] |= 1 << (slot & 63);
        }
        assert_eq!(set.words, words);
        assert_eq!(set.heap_bytes(), words.capacity() * 8);
        for slot in (1..50_001usize).step_by(5) {
            set.remove(slot);
            words[slot >> 6] &= !(1 << (slot & 63));
        }
        assert_eq!(
            set.words, words,
            "a removal clears its bit and keeps the words"
        );
    }

    /// The model test: inserts and removes over positions drawn across a wide
    /// range, against a sorted set.
    #[test]
    fn the_set_answers_what_a_sorted_set_answers() {
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
                    set.remove(slot);
                    model.remove(&slot);
                } else {
                    set.insert(slot);
                    model.insert(slot);
                }
                if step % 211 == 0 {
                    assert_eq!(ids(&set), model.iter().copied().collect::<Vec<_>>());
                    assert_eq!(set.count(), model.len());
                    for &probe in [slot, slot + 1, slot.saturating_sub(1)].iter() {
                        assert_eq!(set.contains(probe), model.contains(&probe));
                    }
                }
            }
            assert_eq!(ids(&set), model.iter().copied().collect::<Vec<_>>());
        }
    }

    /// A set over `slots` at the full size of `range`, as every set the
    /// column store combines is.
    fn set_of(slots: &[usize], range: usize) -> Bitmap {
        let mut set = Bitmap::zeros(range);
        for &slot in slots {
            set.set(slot);
        }
        set
    }

    /// The algebra gives what the sorted sets give.
    #[test]
    fn the_algebra_answers_what_the_sorted_sets_answer() {
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
            let mut and = set_of(&a, range);
            and.intersect(&set_of(&b, range));
            assert_eq!(ids(&and), ma.intersection(&mb).copied().collect::<Vec<_>>());
            assert_eq!(and.count(), ma.intersection(&mb).count());
            let mut or = set_of(&a, range);
            or.union(&set_of(&b, range));
            assert_eq!(ids(&or), ma.union(&mb).copied().collect::<Vec<_>>());
            let not = set_of(&a, range).complement_within(&set_of(&live, range));
            assert_eq!(ids(&not), ml.difference(&ma).copied().collect::<Vec<_>>());
            assert_eq!(
                set_of(&a, range).count_and(&set_of(&b, range)),
                ma.intersection(&mb).count()
            );
            assert_eq!(set_of(&a, range).is_empty(), ma.is_empty());
        }
        let empty = Bitmap::default();
        assert!(empty.is_empty());
        assert_eq!(empty.count(), 0);
        assert!(!empty.contains(0));
    }

    /// The words of a set answer what the set answers at every position,
    /// across word boundaries, past the words' extent and at the empty
    /// marker, which no set holds.
    #[test]
    fn the_words_of_a_set_answer_what_the_set_answers() {
        let held = [0usize, 1, 63, 64, 65, 127, 128, 1_000, 4_095];
        let mut set = Bitmap::default();
        for &slot in &held {
            set.insert(slot);
        }
        let mut asked: BTreeSet<usize> =
            BTreeSet::from([4_096, 4_097, 1 << 20, Position::NONE as usize, usize::MAX]);
        for &slot in &held {
            asked.extend(slot.saturating_sub(1)..=slot + 1);
        }
        for slot in asked {
            let word = set.words.get(slot >> 6).copied().unwrap_or(0);
            assert_eq!(
                word >> (slot & 63) & 1 == 1,
                set.contains(slot),
                "at {slot}"
            );
            assert_eq!(set.contains(slot), held.contains(&slot), "at {slot}");
        }

        // A set whose words reach the last position a slot can name still
        // answers no for the empty marker, whose bit no insert ever sets.
        let mut wide = Bitmap::default();
        wide.insert(Position::NONE as usize - 1);
        assert!(wide.contains(Position::NONE as usize - 1));
        assert!(!wide.contains(Position::NONE as usize));
    }

    /// A set a close-up leaves holds exactly the positions below the record
    /// count, in words that reach the count or the reservation and no
    /// further, whatever it held before.
    #[test]
    fn a_closed_up_set_holds_the_first_positions_in_its_words() {
        for (count, floor) in [
            (0usize, 0usize),
            (1, 0),
            (64, 0),
            (65, 0),
            (130, 0),
            (5, 1_001),
            (2_000, 1_001),
        ] {
            let mut set = Bitmap::default();
            for slot in (0..10_000).step_by(3) {
                set.insert(slot);
            }
            set.hold_first(count, floor);
            assert_eq!(set.words.len(), count.max(floor).div_ceil(64));
            assert_eq!(set.heap_bytes(), set.words.len() * 8);
            assert_eq!(set.count(), count);
            assert!((0..count).all(|slot| set.contains(slot)));
            assert!((count..10_100).all(|slot| !set.contains(slot)));
        }
    }
}
