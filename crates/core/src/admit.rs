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
//! `enumerate` is the other door. An admit set that can walk itself in id
//! order lets an index score the members directly rather than traverse or
//! scan, which is what the exact scan under a selective filter does.

use crate::columns::Bitmap;
use crate::space::{Hits, RecordId};

/// Which records a search may return.
pub trait Admit: Sync {
    fn admits(&self, id: RecordId) -> bool;

    /// How many ids this admits, where known. `None` means unknown, which is
    /// what a predicate over metadata answers.
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

    /// Walk every admitted id in increasing order, stopping when `visit`
    /// returns false. Returns false if this set cannot enumerate itself, in
    /// which case `visit` was never called.
    fn enumerate(&self, visit: &mut dyn FnMut(RecordId) -> bool) -> bool {
        let _ = visit;
        false
    }
}

impl Admit for Bitmap {
    #[inline]
    fn admits(&self, id: RecordId) -> bool {
        self.contains(id.slot())
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.count())
    }

    fn as_bitmap(&self) -> Option<&Bitmap> {
        Some(self)
    }

    fn enumerate(&self, visit: &mut dyn FnMut(RecordId) -> bool) -> bool {
        self.for_each_while(|slot| visit(RecordId::from_slot(slot)));
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
    /// Strictly increasing record ids.
    Sorted(Vec<RecordId>),
}

impl Candidates {
    /// Turn a page into a candidate set, which is how one arm feeds the next.
    pub fn from_hits(hits: &Hits) -> Self {
        let mut ids: Vec<RecordId> = hits.items.iter().map(|h| h.id).collect();
        ids.sort_unstable();
        ids.dedup();
        Candidates::Sorted(ids)
    }
}

impl Admit for Candidates {
    fn admits(&self, id: RecordId) -> bool {
        match self {
            Candidates::All => true,
            Candidates::Bitmap(b) => b.contains(id.slot()),
            Candidates::Sorted(ids) => ids.binary_search(&id).is_ok(),
        }
    }

    fn admits_all(&self) -> bool {
        matches!(self, Candidates::All)
    }

    fn len_hint(&self) -> Option<usize> {
        match self {
            Candidates::All => None,
            Candidates::Bitmap(b) => Some(b.count()),
            Candidates::Sorted(ids) => Some(ids.len()),
        }
    }

    fn as_bitmap(&self) -> Option<&Bitmap> {
        match self {
            Candidates::Bitmap(b) => Some(b),
            _ => None,
        }
    }

    fn enumerate(&self, visit: &mut dyn FnMut(RecordId) -> bool) -> bool {
        match self {
            Candidates::All => false,
            Candidates::Bitmap(b) => b.enumerate(visit),
            Candidates::Sorted(ids) => {
                for &id in ids {
                    if !visit(id) {
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
    fn admits(&self, id: RecordId) -> bool {
        self.0.admits(id) && self.1.admits(id)
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
    fn enumerate(&self, visit: &mut dyn FnMut(RecordId) -> bool) -> bool {
        let (drive, test): (&dyn Admit, &dyn Admit) = match (self.0.len_hint(), self.1.len_hint()) {
            (Some(a), Some(b)) if b < a => (self.1, self.0),
            (None, Some(_)) => (self.1, self.0),
            _ => (self.0, self.1),
        };
        if drive.enumerate(&mut |id| !test.admits(id) || visit(id)) {
            return true;
        }
        test.enumerate(&mut |id| !drive.admits(id) || visit(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(admit: &dyn Admit) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        admit
            .enumerate(&mut |id| {
                out.push(id.0);
                true
            })
            .then_some(out)
    }

    struct Even;

    impl Admit for Even {
        fn admits(&self, id: RecordId) -> bool {
            id.0.is_multiple_of(2)
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
        assert!(bitmap.admits(RecordId(64)));
        assert!(!bitmap.admits(RecordId(4)));
        assert_eq!(bitmap.len_hint(), Some(4));
        assert!(bitmap.as_bitmap().is_some());
        assert_eq!(ids(&bitmap), Some(vec![3, 64, 65, 200]));
    }

    /// Every candidate shape answers the four questions the way its name
    /// says, and a page becomes a sorted, deduplicated set.
    #[test]
    fn candidates_answer_by_shape() {
        assert!(Candidates::All.admits(RecordId(9)));
        assert!(Candidates::All.admits_all());
        assert_eq!(Candidates::All.len_hint(), None);
        assert_eq!(ids(&Candidates::All), None);

        let sorted = Candidates::Sorted(vec![RecordId(2), RecordId(5), RecordId(9)]);
        assert!(sorted.admits(RecordId(5)));
        assert!(!sorted.admits(RecordId(6)));
        assert!(!sorted.admits_all());
        assert_eq!(sorted.len_hint(), Some(3));
        assert!(sorted.as_bitmap().is_none());
        assert_eq!(ids(&sorted), Some(vec![2, 5, 9]));

        let hits = Hits {
            items: [9u32, 2, 9, 5]
                .iter()
                .map(|&id| crate::space::Hit {
                    id: RecordId(id),
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
        assert!(both.admits(RecordId(4)));
        assert!(!both.admits(RecordId(5)));
        assert!(!both.admits(RecordId(12)));

        let mut seen = Vec::new();
        assert!(both.enumerate(&mut |id| {
            seen.push(id.0);
            seen.len() < 2
        }));
        assert_eq!(seen, vec![0, 2]);

        assert_eq!(ids(&And(&Even, &Candidates::All)), None);
    }
}
