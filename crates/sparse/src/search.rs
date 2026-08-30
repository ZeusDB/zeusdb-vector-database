//! The search loops, the top-k selection, and the rule that chooses a loop.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use zeusdb_vector_core::{Admit, Bitmap, Error, Hit, Hits, RecordId, ScoreKind, SparseRef};

use crate::index::PostingsIndex;

/// Which loop a search runs. `Auto` is what the trait method uses. The rest
/// exist so a measurement can name each arm and a test can check every one
/// against brute force.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Choose between the scan and the enumerate-driven path by cost.
    Auto,
    /// Term-at-a-time scan, `admit` asked once per posting through the table.
    PerPosting,
    /// Term-at-a-time scan, `admit` asked once per touched record after the
    /// accumulation.
    PerCandidate,
    /// Term-at-a-time scan, `admit.as_bitmap()` taken once and the loop
    /// monomorphised over it, asked once per posting. Falls back to the
    /// table where the admit set is not a bitmap.
    BitmapPerPosting,
    /// The admit set drives: every admitted id is scored from the forward
    /// arena against the query. `admits` is never asked. Falls back to the
    /// per-candidate scan where the set cannot enumerate itself.
    Enumerate,
    /// The floor for a measurement. No predicate and no live test, so it is
    /// correct only when nothing has been removed and no filter applies.
    Floor,
}

/// Admitted records at or under which the enumerate-driven path runs
/// without being priced.
///
/// Pricing a search reads the length of every list the query names, which
/// is one hash lookup per query dimension, and on a query of forty
/// dimensions that is a few microseconds against a search over twenty
/// records that takes twelve. Below this many records no scan of any list
/// the index holds is cheaper than scoring them from the arena, so the
/// rule's answer is known before it is asked.
const ENUMERATE_OUTRIGHT: usize = 32;

/// The dense accumulator and the touched list, kept per thread so a search
/// does not allocate and zero a buffer the size of the record table.
///
/// The accumulator is cleared by walking the touched list rather than the
/// whole buffer, so the cost of clearing is the cost of the scan's own
/// footprint. A `RefCell` in a thread local rather than a lock, because the
/// buffer is never shared between threads.
struct Scratch {
    acc: Vec<f32>,
    touched: Vec<u32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const {
        RefCell::new(Scratch {
            acc: Vec::new(),
            touched: Vec::new(),
        })
    };
}

impl Scratch {
    fn ready(&mut self, slots: usize) {
        if self.acc.len() < slots {
            self.acc.resize(slots, 0.0);
        }
        self.touched.clear();
    }

    /// Zero what the scan touched.
    ///
    /// Entry by entry where the scan touched few records, since a query of
    /// rare terms touches a few hundred of fifty thousand slots. As one
    /// contiguous fill once it touched more than a sixteenth of them, since
    /// a scattered write per touched slot costs more than a sweep of the
    /// whole buffer past that point: measured against a fresh zeroed buffer
    /// per search, the scattered reset alone made the whole scan a tenth
    /// slower on both regimes.
    fn reset(&mut self, slots: usize) {
        if self.touched.len() > slots / 16 {
            self.acc[..slots].fill(0.0);
        } else {
            for &id in &self.touched {
                self.acc[id as usize] = 0.0;
            }
        }
        self.touched.clear();
    }
}

impl PostingsIndex {
    /// The two arms' estimated cost in nanoseconds for a scan visiting
    /// `scan_postings` postings under an admit set of `admitted` records,
    /// and for scoring `admitted` records from the arena against a query of
    /// `query_nnz` nonzeros.
    ///
    /// The scan's per-posting cost depends on the share admitted. A posting
    /// the bitmap rejects costs the bit test alone, and one it admits costs
    /// the accumulate as well, so the estimate is the mix, plus the
    /// misprediction a test pays when its outcome cannot be guessed from the
    /// last one's, which for a share `p` admitted at random is `2p(1 - p)`
    /// of the tests. A set that is not a bitmap is tested through the table,
    /// which is priced as one more posting visit per posting on top of the
    /// accumulate. The enumerate-driven path pays a fixed cost per admitted
    /// record and a merge of the record against the query.
    pub fn arm_costs(
        &self,
        scan_postings: usize,
        admitted: usize,
        bitmap: bool,
        query_nnz: usize,
    ) -> (f64, f64) {
        let units = self.units;
        let records = self.live.max(1) as f64;
        let frac = (admitted as f64 / records).min(1.0);
        let per_posting = if bitmap {
            frac * units.posting_ns
                + (1.0 - frac) * units.reject_ns
                + 2.0 * frac * (1.0 - frac) * units.mispredict_ns
        } else {
            units.posting_ns * 2.0
        };
        let scan_ns = scan_postings as f64 * per_posting;
        let enumerate_ns = admitted as f64
            * (units.record_ns + (self.mean_nnz() + query_nnz as f64) * units.merge_ns);
        (scan_ns, enumerate_ns)
    }

    /// Term-at-a-time accumulation with a per-posting predicate. Generic so
    /// the monomorphised, closure and trait-object arms share one body.
    fn scan_per_posting<P: Fn(u32) -> bool>(
        &self,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admits: P,
    ) -> Vec<Hit> {
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.ready(self.slots());
            let Scratch { acc, touched } = &mut *scratch;
            for (d, &qw) in query.dims.iter().zip(query.values) {
                let Some(&slot) = self.slots_by_dim.get(d) else {
                    continue;
                };
                for p in &self.lists[slot as usize].postings {
                    if !admits(p.id) {
                        continue;
                    }
                    let a = &mut acc[p.id as usize];
                    if *a == 0.0 {
                        touched.push(p.id);
                    }
                    *a += p.weight * qw;
                }
            }
            let page = select(acc, touched, k, boundary_ties, |_| true);
            scratch.reset(self.slots());
            page
        })
    }

    /// Term-at-a-time accumulation, the predicate asked once per touched
    /// record after the accumulation.
    fn scan_per_candidate(
        &self,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admit: &dyn Admit,
    ) -> Vec<Hit> {
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.ready(self.slots());
            let Scratch { acc, touched } = &mut *scratch;
            for (d, &qw) in query.dims.iter().zip(query.values) {
                let Some(&slot) = self.slots_by_dim.get(d) else {
                    continue;
                };
                for p in &self.lists[slot as usize].postings {
                    let a = &mut acc[p.id as usize];
                    if *a == 0.0 {
                        touched.push(p.id);
                    }
                    *a += p.weight * qw;
                }
            }
            let dead = &self.dead;
            let page = select(acc, touched, k, boundary_ties, |id| {
                !dead.contains(id as usize) && admit.admits(RecordId(id))
            });
            scratch.reset(self.slots());
            page
        })
    }

    /// The admit set drives. `None` if it cannot enumerate itself.
    fn enumerate_driven(
        &self,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admit: &dyn Admit,
    ) -> Option<Vec<Hit>> {
        let mut scored: Vec<(f32, u32)> = Vec::new();
        let walked = admit.enumerate(&mut |id| {
            if let Some(slot) = self.slot_of(id) {
                let score = self.forward(slot).dot(query);
                if score != 0.0 {
                    scored.push((score, id.0));
                }
            }
            true
        });
        if !walked {
            return None;
        }
        Some(cut(scored, k, boundary_ties))
    }

    /// Run one named loop. What the trait's `search` calls with `Mode::Auto`.
    pub fn search_mode(
        &self,
        mode: Mode,
        query: SparseRef<'_>,
        k: usize,
        admit: &dyn Admit,
        boundary_ties: bool,
    ) -> Result<Hits, Error> {
        query.validate()?;
        let has_dead = self.dead_records > 0;
        let dead = &self.dead;
        let items = match mode {
            Mode::Floor => self.scan_per_posting(query, k, boundary_ties, |_| true),
            Mode::PerPosting => self.scan_per_posting(query, k, boundary_ties, |id| {
                !dead.contains(id as usize) && admit.admits(RecordId(id))
            }),
            Mode::BitmapPerPosting => self.scan_bitmap(query, k, boundary_ties, admit, has_dead),
            Mode::PerCandidate => self.scan_per_candidate(query, k, boundary_ties, admit),
            Mode::Enumerate => match self.enumerate_driven(query, k, boundary_ties, admit) {
                Some(items) => items,
                None => self.scan_per_candidate(query, k, boundary_ties, admit),
            },
            Mode::Auto => self.auto(query, k, boundary_ties, admit, has_dead),
        };
        Ok(Hits {
            items,
            kind: ScoreKind::Similarity,
            exact: true,
        })
    }

    /// The bitmap-monomorphised scan, with the dead test only where a record
    /// has been removed since the last compaction. Falls back to the table
    /// where the admit set is not a bitmap.
    fn scan_bitmap(
        &self,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admit: &dyn Admit,
        has_dead: bool,
    ) -> Vec<Hit> {
        let dead: &Bitmap = &self.dead;
        match (admit.as_bitmap(), has_dead) {
            (Some(bitmap), false) => {
                self.scan_per_posting(query, k, boundary_ties, |id| bitmap.contains(id as usize))
            }
            (Some(bitmap), true) => self.scan_per_posting(query, k, boundary_ties, |id| {
                bitmap.contains(id as usize) && !dead.contains(id as usize)
            }),
            (None, false) => {
                self.scan_per_posting(query, k, boundary_ties, |id| admit.admits(RecordId(id)))
            }
            (None, true) => self.scan_per_posting(query, k, boundary_ties, |id| {
                !dead.contains(id as usize) && admit.admits(RecordId(id))
            }),
        }
    }

    /// The rule. A set the index can enumerate drives the search when
    /// scoring its members from the arena is estimated cheaper than the
    /// scan under it, and the scan runs otherwise, monomorphised over the
    /// bitmap where the set is one. A set admitting everything, which
    /// answers no hint, is the scan with no predicate but the dead test.
    fn auto(
        &self,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admit: &dyn Admit,
        has_dead: bool,
    ) -> Vec<Hit> {
        let Some(admitted) = admit.len_hint() else {
            let dead = &self.dead;
            return match (admit.as_bitmap(), has_dead) {
                (None, false) => {
                    self.scan_per_posting(query, k, boundary_ties, |id| admit.admits(RecordId(id)))
                }
                _ => self.scan_bitmap(query, k, boundary_ties, admit, has_dead || dead.count() > 0),
            };
        };
        let enumerate = admitted <= ENUMERATE_OUTRIGHT || {
            let scan = self.scan_postings(query);
            let (scan_ns, enumerate_ns) =
                self.arm_costs(scan, admitted, admit.as_bitmap().is_some(), query.nnz());
            enumerate_ns < scan_ns
        };
        if enumerate {
            if let Some(items) = self.enumerate_driven(query, k, boundary_ties, admit) {
                return items;
            }
        }
        self.scan_bitmap(query, k, boundary_ties, admit, has_dead)
    }
}

// ---------------------------------------------------------------------------
// Top-k. Higher score wins, then lower id.
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
struct Cand {
    score: f32,
    id: u32,
}

impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cand {
    /// Greater is better: higher score, then lower id.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.id.cmp(&self.id))
    }
}

/// Bounded top-k over a stream of candidates.
struct TopK {
    k: usize,
    heap: BinaryHeap<std::cmp::Reverse<Cand>>,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    #[inline]
    fn offer(&mut self, score: f32, id: u32) {
        if self.k == 0 {
            return;
        }
        let cand = Cand { score, id };
        if self.heap.len() < self.k {
            self.heap.push(std::cmp::Reverse(cand));
        } else if cand > self.heap.peek().expect("the heap holds k entries").0 {
            self.heap.pop();
            self.heap.push(std::cmp::Reverse(cand));
        }
    }

    /// The page, best first, and the score of its last member where the page
    /// is full.
    fn finish(self) -> (Vec<Cand>, Option<f32>) {
        let full = self.heap.len() == self.k && self.k > 0;
        let mut out: Vec<Cand> = self.heap.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        let boundary = full.then(|| out.last().map(|c| c.score)).flatten();
        (out, boundary)
    }
}

/// Select the page from the accumulator over the touched records that pass
/// `admits`, keeping the boundary tie group where asked.
fn select<P: Fn(u32) -> bool>(
    acc: &[f32],
    touched: &[u32],
    k: usize,
    boundary_ties: bool,
    admits: P,
) -> Vec<Hit> {
    let mut top = TopK::new(k);
    for &id in touched {
        let score = acc[id as usize];
        if score != 0.0 && admits(id) {
            top.offer(score, id);
        }
    }
    let (mut page, boundary) = top.finish();
    if let (true, Some(boundary)) = (boundary_ties, boundary) {
        // Every record tied at the boundary score that the heap cut, in id
        // order after the page's own members, so the page stays ordered by
        // score and then by id.
        let last_id = page.last().map(|c| c.id).unwrap_or(0);
        let mut extra: Vec<u32> = touched
            .iter()
            .copied()
            .filter(|&id| acc[id as usize] == boundary && id > last_id && admits(id))
            .collect();
        extra.sort_unstable();
        page.extend(extra.into_iter().map(|id| Cand {
            score: boundary,
            id,
        }));
    }
    page.into_iter()
        .map(|c| Hit {
            id: RecordId(c.id),
            score: c.score,
        })
        .collect()
}

/// Order a fully scored candidate list and cut it, keeping the boundary tie
/// group where asked.
fn cut(mut scored: Vec<(f32, u32)>, k: usize, boundary_ties: bool) -> Vec<Hit> {
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut end = k.min(scored.len());
    if boundary_ties && end > 0 && end < scored.len() {
        let boundary = scored[end - 1].0;
        while end < scored.len() && scored[end].0 == boundary {
            end += 1;
        }
    }
    scored.truncate(end);
    scored
        .into_iter()
        .map(|(score, id)| Hit {
            id: RecordId(id),
            score,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{SparseConfig, Unlink};
    use zeusdb_vector_core::{Candidates, Prepared, SparseVector, VectorIndex};

    fn index_of(rows: &[(u32, &[u32], &[f32])], unlink: Unlink) -> PostingsIndex {
        let mut index = PostingsIndex::new(SparseConfig {
            unlink,
            ..SparseConfig::default()
        });
        for &(id, dims, values) in rows {
            let v = SparseVector {
                dims: dims.to_vec(),
                values: values.to_vec(),
            };
            index
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        index
    }

    /// Ties are broken by lower id, a page is shorter than `k` when fewer
    /// records share a term with the query, and the boundary tie group is
    /// kept only when asked.
    #[test]
    fn ties_go_to_the_lower_id_and_the_boundary_group_is_kept_on_request() {
        let index = index_of(
            &[
                (1, &[7], &[2.0]),
                (2, &[7], &[3.0]),
                (3, &[7], &[3.0]),
                (4, &[7], &[3.0]),
                (5, &[7], &[1.0]),
                (6, &[8], &[9.0]),
            ],
            Unlink::Lazy,
        );
        let q = SparseVector {
            dims: vec![7],
            values: vec![1.0],
        };
        for mode in [
            Mode::Auto,
            Mode::PerPosting,
            Mode::PerCandidate,
            Mode::BitmapPerPosting,
            Mode::Enumerate,
        ] {
            let page = index
                .search_mode(mode, q.as_ref(), 2, &Candidates::All, false)
                .unwrap();
            let ids: Vec<u32> = page.items.iter().map(|h| h.id.0).collect();
            assert_eq!(ids, vec![2, 3], "{mode:?}");

            let page = index
                .search_mode(mode, q.as_ref(), 2, &Candidates::All, true)
                .unwrap();
            let ids: Vec<u32> = page.items.iter().map(|h| h.id.0).collect();
            assert_eq!(ids, vec![2, 3, 4], "{mode:?} with the boundary group");

            let page = index
                .search_mode(mode, q.as_ref(), 10, &Candidates::All, true)
                .unwrap();
            assert_eq!(page.items.len(), 5, "{mode:?} short page");
            assert!(page.exact);
        }
    }

    /// A removed record leaves every page under every loop, whether or not
    /// the caller's admit set knows about it.
    #[test]
    fn a_removed_record_leaves_every_loop() {
        let mut index = index_of(
            &[
                (1, &[7], &[2.0]),
                (2, &[7], &[3.0]),
                (3, &[7, 9], &[1.0, 1.0]),
            ],
            Unlink::Strand,
        );
        index.remove(RecordId(2)).unwrap();
        let q = SparseVector {
            dims: vec![7],
            values: vec![1.0],
        };
        let mut everything = Bitmap::default();
        for id in 1..=3usize {
            everything.insert(id);
        }
        for mode in [
            Mode::Auto,
            Mode::PerPosting,
            Mode::PerCandidate,
            Mode::BitmapPerPosting,
            Mode::Enumerate,
        ] {
            for admit in [&Candidates::All as &dyn Admit, &everything] {
                let page = index
                    .search_mode(mode, q.as_ref(), 10, admit, false)
                    .unwrap();
                let ids: Vec<u32> = page.items.iter().map(|h| h.id.0).collect();
                assert_eq!(ids, vec![1, 3], "{mode:?}");
            }
        }
    }

    /// A malformed query is refused before any list is read.
    #[test]
    fn a_malformed_query_is_refused() {
        let index = index_of(&[(1, &[7], &[2.0])], Unlink::Lazy);
        let bad = SparseRef {
            dims: &[9, 7],
            values: &[1.0, 1.0],
        };
        assert!(matches!(
            index.search(
                bad,
                10,
                &Candidates::All,
                &zeusdb_vector_core::Budget::default()
            ),
            Err(Error::SparseDimsNotIncreasing { position: 1 })
        ));
    }
}
