//! The predicate every arm accepts.
//!
//! A search is asked under an [`Admit`] set, which says which records it may
//! return. The collection builds one from the live set, from a filter's
//! bitmap, from a previous arm's page, or from a conjunction of those, and
//! the index it hands it to never sees a metadata store or a column.
//!
//! # Two doors, so the hot loop pays for neither
//!
//! `admits` is one indirect call per candidate, which a graph traversal asks
//! once per six to eight distance evaluations and a postings scan asks once
//! per posting. Measured on a postings scan the call is 28 to 33 percent of
//! the whole loop, so an admit set that is a bitmap hands it over through
//! `as_bitmap` and the index runs a loop monomorphised over the bit test.
//! `enumerate` is the other door. An admit set that can walk itself in
//! position order lets an index score the members directly rather than
//! traverse or scan, which is what the exact scan under a selective filter
//! does.
//!
//! Every set names records by their position.

use crate::bitmap::Bitmap;
use crate::space::{Hits, Position};

/// Which records a search may return.
pub trait Admit: Sync {
    fn admits(&self, at: Position) -> bool;

    /// How many positions this admits, where known. `None` means unknown,
    /// which is what a predicate over metadata answers.
    fn len_hint(&self) -> Option<usize>;

    /// Whether this set admits every record, which a caller with no filter
    /// passes. An index then runs under its own live set alone, which is the
    /// traversal an unfiltered search has always run.
    fn admits_all(&self) -> bool {
        false
    }

    /// The set as a bitmap, where it is one, so an index can test bits in a
    /// loop of its own rather than call through the table.
    fn as_bitmap(&self) -> Option<&Bitmap> {
        None
    }

    /// Walk every admitted position in increasing order, stopping when
    /// `visit` returns false. Returns false if this set cannot enumerate
    /// itself, in which case `visit` was never called.
    fn enumerate(&self, visit: &mut dyn FnMut(Position) -> bool) -> bool {
        let _ = visit;
        false
    }
}

impl Admit for Bitmap {
    #[inline]
    fn admits(&self, at: Position) -> bool {
        self.contains(at.slot())
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.count())
    }

    fn as_bitmap(&self) -> Option<&Bitmap> {
        Some(self)
    }

    fn enumerate(&self, visit: &mut dyn FnMut(Position) -> bool) -> bool {
        self.for_each_while(|slot| visit(Position::from_slot(slot)));
        true
    }
}

/// A filter's answer, or a previous arm's page turned into an admit set.
#[derive(Clone, Default)]
pub enum Candidates {
    /// Every record the index holds. A caller with no filter passes this and
    /// the index applies its own live set.
    #[default]
    All,
    Bitmap(Bitmap),
    /// Strictly increasing positions.
    Sorted(Vec<Position>),
}

impl Candidates {
    /// Turn a page into a candidate set, which is how one arm feeds the next.
    pub fn from_hits(hits: &Hits) -> Self {
        let mut positions: Vec<Position> = hits.items.iter().map(|h| h.position).collect();
        positions.sort_unstable();
        positions.dedup();
        Candidates::Sorted(positions)
    }
}

impl Admit for Candidates {
    fn admits(&self, at: Position) -> bool {
        match self {
            Candidates::All => true,
            Candidates::Bitmap(b) => b.contains(at.slot()),
            Candidates::Sorted(positions) => positions.binary_search(&at).is_ok(),
        }
    }

    fn admits_all(&self) -> bool {
        matches!(self, Candidates::All)
    }

    fn len_hint(&self) -> Option<usize> {
        match self {
            Candidates::All => None,
            Candidates::Bitmap(b) => Some(b.count()),
            Candidates::Sorted(positions) => Some(positions.len()),
        }
    }

    fn as_bitmap(&self) -> Option<&Bitmap> {
        match self {
            Candidates::Bitmap(b) => Some(b),
            _ => None,
        }
    }

    fn enumerate(&self, visit: &mut dyn FnMut(Position) -> bool) -> bool {
        match self {
            Candidates::All => false,
            Candidates::Bitmap(b) => b.enumerate(visit),
            Candidates::Sorted(positions) => {
                for &at in positions {
                    if !visit(at) {
                        break;
                    }
                }
                true
            }
        }
    }
}

/// Both must admit. The collection conjoins a filter's bound with the
/// predicate that finishes it, and a chained arm's candidates with a filter.
pub struct And<'a>(pub &'a dyn Admit, pub &'a dyn Admit);

impl Admit for And<'_> {
    fn admits(&self, at: Position) -> bool {
        self.0.admits(at) && self.1.admits(at)
    }

    fn admits_all(&self) -> bool {
        self.0.admits_all() && self.1.admits_all()
    }

    /// The smaller of the two hints, which is an upper bound on the
    /// conjunction rather than its size.
    fn len_hint(&self) -> Option<usize> {
        match (self.0.len_hint(), self.1.len_hint()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Drive from whichever side is smaller and can enumerate, and test the
    /// other. Returns false only when neither side can enumerate.
    fn enumerate(&self, visit: &mut dyn FnMut(Position) -> bool) -> bool {
        let (drive, test): (&dyn Admit, &dyn Admit) = match (self.0.len_hint(), self.1.len_hint()) {
            (Some(a), Some(b)) if b < a => (self.1, self.0),
            (None, Some(_)) => (self.1, self.0),
            _ => (self.0, self.1),
        };
        if drive.enumerate(&mut |at| !test.admits(at) || visit(at)) {
            return true;
        }
        test.enumerate(&mut |at| !drive.admits(at) || visit(at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(admit: &dyn Admit) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        admit
            .enumerate(&mut |at| {
                out.push(at.0);
                true
            })
            .then_some(out)
    }

    struct Even;

    impl Admit for Even {
        fn admits(&self, at: Position) -> bool {
            at.0.is_multiple_of(2)
        }
        fn len_hint(&self) -> Option<usize> {
            None
        }
    }

    /// A bitmap admits what it holds, counts itself, hands itself over, and
    /// walks in increasing order across a word boundary.
    #[test]
    fn a_bitmap_is_an_admit_set() {
        let mut bitmap = Bitmap::default();
        for slot in [3usize, 64, 65, 200] {
            bitmap.insert(slot);
        }
        assert!(bitmap.admits(Position(64)));
        assert!(!bitmap.admits(Position(4)));
        assert_eq!(bitmap.len_hint(), Some(4));
        assert!(bitmap.as_bitmap().is_some());
        assert_eq!(ids(&bitmap), Some(vec![3, 64, 65, 200]));
    }

    /// Every candidate shape answers the four questions the way its name
    /// says, and a page becomes a sorted, deduplicated set.
    #[test]
    fn candidates_answer_by_shape() {
        assert!(Candidates::All.admits(Position(9)));
        assert!(Candidates::All.admits_all());
        assert_eq!(Candidates::All.len_hint(), None);
        assert_eq!(ids(&Candidates::All), None);

        let sorted = Candidates::Sorted(vec![Position(2), Position(5), Position(9)]);
        assert!(sorted.admits(Position(5)));
        assert!(!sorted.admits(Position(6)));
        assert!(!sorted.admits_all());
        assert_eq!(sorted.len_hint(), Some(3));
        assert!(sorted.as_bitmap().is_none());
        assert_eq!(ids(&sorted), Some(vec![2, 5, 9]));

        let hits = Hits {
            items: [9u32, 2, 9, 5]
                .iter()
                .map(|&at| crate::space::Hit {
                    position: Position(at),
                    score: 0.0,
                })
                .collect(),
            kind: crate::space::ScoreKind::Distance,
            exact: true,
        };
        assert_eq!(ids(&Candidates::from_hits(&hits)), Some(vec![2, 5, 9]));
    }

    /// A conjunction drives from the side that can enumerate and tests the
    /// other, stops when the visitor declines, and admits only what both do.
    #[test]
    fn a_conjunction_drives_from_the_enumerable_side() {
        let mut bitmap = Bitmap::default();
        for slot in 0..10usize {
            bitmap.insert(slot);
        }
        let both = And(&Even, &bitmap);
        assert_eq!(ids(&both), Some(vec![0, 2, 4, 6, 8]));
        assert_eq!(both.len_hint(), Some(10));
        assert!(both.admits(Position(4)));
        assert!(!both.admits(Position(5)));
        assert!(!both.admits(Position(12)));

        let mut seen = Vec::new();
        assert!(both.enumerate(&mut |at| {
            seen.push(at.0);
            seen.len() < 2
        }));
        assert_eq!(seen, vec![0, 2]);

        assert_eq!(ids(&And(&Even, &Candidates::All)), None);
    }
}
