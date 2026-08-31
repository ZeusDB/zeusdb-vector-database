//! The search loops, the scoring rules, the corpus statistics, the top-k
//! selection, and the rule that chooses a loop.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use zeusdb_vector_core::{
    Admit, Bitmap, CorpusStats, Error, Hit, Hits, IdfScope, RecordId, ScoreKind, SparseRef,
};

use crate::index::{PostingsIndex, Weighting};

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
/// does not allocate and clear a buffer the size of the record table.
///
/// A slot no posting of the current scan has touched holds NaN rather than
/// zero, so the first contribution to a record is told from a later one by
/// the slot's own value and the record is put on the touched list exactly
/// once. Zero would not do, because a signed contribution can bring a
/// touched record's sum back to exactly zero and the next contribution would
/// then list it twice, which put a record on a page twice.
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
            self.acc.resize(slots, f32::NAN);
        }
        self.touched.clear();
    }

    /// Clear what the scan touched.
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
            self.acc[..slots].fill(f32::NAN);
        } else {
            for &id in &self.touched {
                self.acc[id as usize] = f32::NAN;
            }
        }
        self.touched.clear();
    }
}

// ---------------------------------------------------------------------------
// Scoring rules. One posting, or one matched element of a record, at a time.
// ---------------------------------------------------------------------------

/// What one stored value contributes against one query value, for the
/// record it belongs to. The scan and the arena merge both call it, so the
/// two paths agree bit for bit under every rule.
trait Scorer {
    fn score(&self, id: u32, stored: f32, query: f32) -> f32;
}

/// The product, which is the sparse dot product summed.
struct DotScorer;

impl Scorer for DotScorer {
    #[inline(always)]
    fn score(&self, _id: u32, stored: f32, query: f32) -> f32 {
        stored * query
    }
}

/// The saturated, length-normalised term frequency, with the query value
/// already carrying the term's rarity and the `k1 + 1` numerator, so the
/// per-posting work is one gather of the record's length, one multiply-add
/// and one division.
struct Bm25Scorer<'a> {
    lengths: &'a [f32],
    /// `k1 * (1 - b)`.
    c0: f32,
    /// `k1 * b / mean_length`.
    c1: f32,
}

impl Scorer for Bm25Scorer<'_> {
    #[inline(always)]
    fn score(&self, id: u32, tf: f32, query: f32) -> f32 {
        query * tf / (tf + self.c0 + self.c1 * self.lengths[id as usize])
    }
}

/// A record scored from the arena against the query by a merge over the
/// dimensions the two share, accumulated in ascending dimension order, which
/// is the order the term-at-a-time scan adds the same contributions in.
fn score_record<S: Scorer>(
    scorer: &S,
    id: u32,
    record: SparseRef<'_>,
    query: SparseRef<'_>,
) -> f32 {
    let (a, b) = (record, query);
    let (mut i, mut j, mut sum) = (0usize, 0usize, 0f32);
    while i < a.dims.len() && j < b.dims.len() {
        match a.dims[i].cmp(&b.dims[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                sum += scorer.score(id, a.values[i], b.values[j]);
                i += 1;
                j += 1;
            }
        }
    }
    sum
}

/// A query with the term weighting applied, being the query's dimensions
/// with each value multiplied by the term's rarity, the dimensions the
/// corpus lacks dropped since they can contribute nothing, and the two
/// per-search constants of the length normalisation.
struct Weighted {
    dims: Vec<u32>,
    values: Vec<f32>,
    c0: f32,
    c1: f32,
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
    /// the monomorphised, closure and trait-object arms share one body, and
    /// over the scoring rule.
    fn scan_per_posting<S: Scorer, P: Fn(u32) -> bool>(
        &self,
        scorer: &S,
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
                    let s = scorer.score(p.id, p.weight, qw);
                    if a.is_nan() {
                        *a = s;
                        touched.push(p.id);
                    } else {
                        *a += s;
                    }
                }
            }
            let page = select(acc, touched, k, boundary_ties, |_| true);
            scratch.reset(self.slots());
            page
        })
    }

    /// Term-at-a-time accumulation, the predicate asked once per touched
    /// record after the accumulation.
    fn scan_per_candidate<S: Scorer>(
        &self,
        scorer: &S,
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
                    let s = scorer.score(p.id, p.weight, qw);
                    if a.is_nan() {
                        *a = s;
                        touched.push(p.id);
                    } else {
                        *a += s;
                    }
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
    fn enumerate_driven<S: Scorer>(
        &self,
        scorer: &S,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admit: &dyn Admit,
    ) -> Option<Vec<Hit>> {
        let mut scored: Vec<(f32, u32)> = Vec::new();
        let walked = admit.enumerate(&mut |id| {
            if let Some(slot) = self.slot_of(id) {
                let score = score_record(scorer, id.0, self.forward(slot), query);
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

    /// Run one named loop with the corpus-scoped weighting. What the
    /// verifier and the measurements call.
    pub fn search_mode(
        &self,
        mode: Mode,
        query: SparseRef<'_>,
        k: usize,
        admit: &dyn Admit,
        boundary_ties: bool,
    ) -> Result<Hits, Error> {
        self.search_scoped(mode, query, k, admit, boundary_ties, IdfScope::Corpus)
    }

    /// Run one named loop under the configured scoring rule. What the
    /// trait's `search` calls with `Mode::Auto`.
    ///
    /// Under the dot product the query is scored as given. Under term
    /// frequency weighting the query is first weighted by each term's
    /// rarity over the corpus `idf` names, which is one pass over the
    /// query's postings under the admit set where that corpus is the
    /// admitted records, and the loop then runs on the weighted query with
    /// the length normalisation applied per posting.
    pub fn search_scoped(
        &self,
        mode: Mode,
        query: SparseRef<'_>,
        k: usize,
        admit: &dyn Admit,
        boundary_ties: bool,
        idf: IdfScope,
    ) -> Result<Hits, Error> {
        query.validate()?;
        let items = match self.config.weighting {
            Weighting::Dot => self.run(&DotScorer, mode, query, k, admit, boundary_ties),
            Weighting::Bm25 { k1, b } => {
                let weighted = self.weigh(query, admit, idf, k1, b);
                let scorer = Bm25Scorer {
                    lengths: &self.lengths,
                    c0: weighted.c0,
                    c1: weighted.c1,
                };
                let query = SparseRef {
                    dims: &weighted.dims,
                    values: &weighted.values,
                };
                self.run(&scorer, mode, query, k, admit, boundary_ties)
            }
        };
        Ok(Hits {
            items,
            kind: ScoreKind::Similarity,
            exact: true,
        })
    }

    fn run<S: Scorer>(
        &self,
        scorer: &S,
        mode: Mode,
        query: SparseRef<'_>,
        k: usize,
        admit: &dyn Admit,
        boundary_ties: bool,
    ) -> Vec<Hit> {
        let has_dead = self.dead_records > 0;
        let dead = &self.dead;
        match mode {
            Mode::Floor => self.scan_per_posting(scorer, query, k, boundary_ties, |_| true),
            Mode::PerPosting => self.scan_per_posting(scorer, query, k, boundary_ties, |id| {
                !dead.contains(id as usize) && admit.admits(RecordId(id))
            }),
            Mode::BitmapPerPosting => {
                self.scan_bitmap(scorer, query, k, boundary_ties, admit, has_dead)
            }
            Mode::PerCandidate => self.scan_per_candidate(scorer, query, k, boundary_ties, admit),
            Mode::Enumerate => {
                match self.enumerate_driven(scorer, query, k, boundary_ties, admit) {
                    Some(items) => items,
                    None => self.scan_per_candidate(scorer, query, k, boundary_ties, admit),
                }
            }
            Mode::Auto => self.auto(scorer, query, k, boundary_ties, admit, has_dead),
        }
    }

    /// The bitmap-monomorphised scan, with the dead test only where a record
    /// has been removed since the last compaction. Falls back to the table
    /// where the admit set is not a bitmap.
    fn scan_bitmap<S: Scorer>(
        &self,
        scorer: &S,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admit: &dyn Admit,
        has_dead: bool,
    ) -> Vec<Hit> {
        let dead: &Bitmap = &self.dead;
        match (admit.as_bitmap(), has_dead) {
            (Some(bitmap), false) => self.scan_per_posting(scorer, query, k, boundary_ties, |id| {
                bitmap.contains(id as usize)
            }),
            (Some(bitmap), true) => self.scan_per_posting(scorer, query, k, boundary_ties, |id| {
                bitmap.contains(id as usize) && !dead.contains(id as usize)
            }),
            (None, false) => self.scan_per_posting(scorer, query, k, boundary_ties, |id| {
                admit.admits(RecordId(id))
            }),
            (None, true) => self.scan_per_posting(scorer, query, k, boundary_ties, |id| {
                !dead.contains(id as usize) && admit.admits(RecordId(id))
            }),
        }
    }

    /// Whether a set of `admitted` records is cheaper scored from the arena
    /// than scanned for, which is the same answer for a search and for the
    /// count of the query's postings under the set.
    fn prefers_enumerate(&self, dims: &[u32], admitted: usize, bitmap: bool) -> bool {
        admitted <= ENUMERATE_OUTRIGHT || {
            let scan = self.scan_postings(SparseRef { dims, values: &[] });
            let (scan_ns, enumerate_ns) = self.arm_costs(scan, admitted, bitmap, dims.len());
            enumerate_ns < scan_ns
        }
    }

    /// The rule. A set the index can enumerate drives the search when
    /// scoring its members from the arena is estimated cheaper than the
    /// scan under it, and the scan runs otherwise, monomorphised over the
    /// bitmap where the set is one. A set admitting everything, which
    /// answers no hint, is the scan with no predicate but the dead test.
    fn auto<S: Scorer>(
        &self,
        scorer: &S,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        admit: &dyn Admit,
        has_dead: bool,
    ) -> Vec<Hit> {
        let Some(admitted) = admit.len_hint() else {
            let dead = &self.dead;
            return match (admit.as_bitmap(), has_dead) {
                (None, false) => self.scan_per_posting(scorer, query, k, boundary_ties, |id| {
                    admit.admits(RecordId(id))
                }),
                _ => self.scan_bitmap(
                    scorer,
                    query,
                    k,
                    boundary_ties,
                    admit,
                    has_dead || dead.count() > 0,
                ),
            };
        };
        if self.prefers_enumerate(query.dims, admitted, admit.as_bitmap().is_some()) {
            if let Some(items) = self.enumerate_driven(scorer, query, k, boundary_ties, admit) {
                return items;
            }
        }
        self.scan_bitmap(scorer, query, k, boundary_ties, admit, has_dead)
    }

    // -----------------------------------------------------------------------
    // Corpus statistics.
    // -----------------------------------------------------------------------

    /// Document frequencies over every live record, and the live count.
    pub(crate) fn global_stats(&self, dims: &[u32]) -> CorpusStats {
        CorpusStats {
            documents: self.live,
            df: dims.iter().map(|&d| self.df(d)).collect(),
        }
    }

    /// Document frequencies over the records `admit` admits, by whichever
    /// of the two walks the search itself would take under that set, or
    /// `None` where the set can neither be tested as a bitmap nor
    /// enumerated. A set admitting everything answers from the lists alone.
    pub(crate) fn stats_under(&self, dims: &[u32], admit: &dyn Admit) -> Option<CorpusStats> {
        if admit.admits_all() {
            return Some(self.global_stats(dims));
        }
        let bitmap = admit.as_bitmap();
        let enumerate_first = admit
            .len_hint()
            .is_some_and(|admitted| self.prefers_enumerate(dims, admitted, bitmap.is_some()));
        if enumerate_first {
            if let Some(stats) = self.stats_by_enumerate(dims, admit) {
                return Some(stats);
            }
        }
        if let Some(bitmap) = bitmap {
            return Some(self.stats_by_walk(dims, bitmap));
        }
        self.stats_by_enumerate(dims, admit)
    }

    /// Each named list walked under the bitmap, counting the postings it
    /// admits, with the dead test only where a record has been removed
    /// since the last compaction. The document count is the intersection
    /// of the bitmap with the live set, taken a word at a time.
    fn stats_by_walk(&self, dims: &[u32], bitmap: &Bitmap) -> CorpusStats {
        let documents = bitmap.count_and(&self.live_set);
        let has_dead = self.dead_records > 0;
        let dead = &self.dead;
        let df = dims
            .iter()
            .map(|d| {
                let Some(&slot) = self.slots_by_dim.get(d) else {
                    return 0;
                };
                let postings = &self.lists[slot as usize].postings;
                if has_dead {
                    postings
                        .iter()
                        .filter(|p| bitmap.contains(p.id as usize) && !dead.contains(p.id as usize))
                        .count()
                } else {
                    postings
                        .iter()
                        .filter(|p| bitmap.contains(p.id as usize))
                        .count()
                }
            })
            .collect();
        CorpusStats { documents, df }
    }

    /// The admit set drives: every admitted record the index holds is
    /// merged against the dimensions from the arena. `None` if the set
    /// cannot enumerate itself. The dimensions are merged in sorted order,
    /// so an unsorted request is sorted first and its counts put back.
    fn stats_by_enumerate(&self, dims: &[u32], admit: &dyn Admit) -> Option<CorpusStats> {
        let sorted = dims.windows(2).all(|w| w[0] < w[1]);
        let order: Vec<usize> = if sorted {
            (0..dims.len()).collect()
        } else {
            let mut order: Vec<usize> = (0..dims.len()).collect();
            order.sort_by_key(|&i| dims[i]);
            order.dedup_by_key(|i| dims[*i]);
            order
        };
        let keys: Vec<u32> = order.iter().map(|&i| dims[i]).collect();
        let mut counts = vec![0usize; keys.len()];
        let mut documents = 0usize;
        let walked = admit.enumerate(&mut |id| {
            if let Some(slot) = self.slot_of(id) {
                documents += 1;
                let record = self.forward(slot);
                let (mut i, mut j) = (0usize, 0usize);
                while i < record.dims.len() && j < keys.len() {
                    match record.dims[i].cmp(&keys[j]) {
                        Ordering::Less => i += 1,
                        Ordering::Greater => j += 1,
                        Ordering::Equal => {
                            counts[j] += 1;
                            i += 1;
                            j += 1;
                        }
                    }
                }
            }
            true
        });
        if !walked {
            return None;
        }
        let df = if sorted {
            counts
        } else {
            dims.iter()
                .map(|d| counts[keys.binary_search(d).expect("every dimension was keyed")])
                .collect()
        };
        Some(CorpusStats { documents, df })
    }

    /// The query weighted for a term frequency space. See [`Weighted`].
    ///
    /// The rarity of a term over `n` documents of which `df` carry it is
    /// `ln(1 + (n - df + 0.5) / (df + 0.5))`, which is above zero for every
    /// `df` up to `n`, so a term every document carries still counts for a
    /// little rather than for nothing or for less than nothing. Computed in
    /// double precision and folded into the query value with the `k1 + 1`
    /// numerator, so the scan's own arithmetic is single precision alone.
    /// The mean length is over every live record whichever corpus the
    /// rarity is counted over, since that is what was measured.
    fn weigh(
        &self,
        query: SparseRef<'_>,
        admit: &dyn Admit,
        idf: IdfScope,
        k1: f32,
        b: f32,
    ) -> Weighted {
        let stats = match idf {
            IdfScope::Global => self.global_stats(query.dims),
            IdfScope::Corpus => self
                .stats_under(query.dims, admit)
                .unwrap_or_else(|| self.global_stats(query.dims)),
        };
        let n = stats.documents as f64;
        let numerator = k1 as f64 + 1.0;
        let mut dims = Vec::with_capacity(query.nnz());
        let mut values = Vec::with_capacity(query.nnz());
        for ((&d, &qw), &df) in query.dims.iter().zip(query.values).zip(&stats.df) {
            if df == 0 {
                continue;
            }
            let df = df as f64;
            let rarity = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            dims.push(d);
            values.push((qw as f64 * rarity * numerator) as f32);
        }
        let mean = self.mean_length();
        let mean = if mean > 0.0 { mean } else { 1.0 };
        Weighted {
            dims,
            values,
            c0: k1 * (1.0 - b),
            c1: (k1 as f64 * b as f64 / mean) as f32,
        }
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

    const MODES: [Mode; 5] = [
        Mode::Auto,
        Mode::PerPosting,
        Mode::PerCandidate,
        Mode::BitmapPerPosting,
        Mode::Enumerate,
    ];

    fn index_with(
        rows: &[(u32, &[u32], &[f32])],
        unlink: Unlink,
        weighting: Weighting,
    ) -> PostingsIndex {
        let mut index = PostingsIndex::new(SparseConfig {
            unlink,
            weighting,
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

    fn index_of(rows: &[(u32, &[u32], &[f32])], unlink: Unlink) -> PostingsIndex {
        index_with(rows, unlink, Weighting::Dot)
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
        for mode in MODES {
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
        for mode in MODES {
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

    /// A signed query can bring a record's accumulator back to exactly zero
    /// and then move it again, and the record appears on the page once.
    #[test]
    fn a_record_whose_score_returns_to_zero_is_listed_once() {
        let index = index_of(
            &[
                (1, &[1, 2, 3], &[1.0, 1.0, 1.0]),
                (2, &[1, 2, 3], &[1.0, 1.0, 2.0]),
                (3, &[3], &[0.5]),
            ],
            Unlink::Lazy,
        );
        let q = SparseVector {
            dims: vec![1, 2, 3],
            values: vec![1.0, -1.0, 1.0],
        };
        for mode in [Mode::Floor, Mode::PerPosting, Mode::PerCandidate] {
            let page = index
                .search_mode(mode, q.as_ref(), 10, &Candidates::All, true)
                .unwrap();
            let ids: Vec<u32> = page.items.iter().map(|h| h.id.0).collect();
            assert_eq!(ids, vec![2, 1, 3], "{mode:?}");
        }
    }

    /// The term frequency weighting reproduces the formula by hand on a
    /// corpus of three, every loop agreeing, and a value at or below zero
    /// is refused at insert.
    #[test]
    fn term_frequency_weighting_reproduces_the_formula_and_refuses_a_zero() {
        let (k1, b) = (1.2f32, 0.75f32);
        let index = index_with(
            &[
                (1, &[1, 2], &[2.0, 1.0]),
                (2, &[1], &[1.0]),
                (3, &[2, 3], &[1.0, 5.0]),
            ],
            Unlink::Lazy,
            Weighting::Bm25 { k1, b },
        );
        assert_eq!(index.mean_length(), 10.0 / 3.0);
        let q = SparseVector {
            dims: vec![1, 2],
            values: vec![1.0, 1.0],
        };
        // By hand, in double precision.
        let n = 3.0f64;
        let idf = |df: f64| (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
        let mean = 10.0 / 3.0;
        let part = |tf: f64, len: f64| {
            tf * (k1 as f64 + 1.0) / (tf + k1 as f64 * (1.0 - b as f64 + b as f64 * len / mean))
        };
        // The shortest record ranks above the longest at the same frequency.
        let expected = [
            (1u32, idf(2.0) * part(2.0, 3.0) + idf(2.0) * part(1.0, 3.0)),
            (2, idf(2.0) * part(1.0, 1.0)),
            (3, idf(2.0) * part(1.0, 6.0)),
        ];
        for mode in MODES {
            let page = index
                .search_mode(mode, q.as_ref(), 10, &Candidates::All, false)
                .unwrap();
            assert_eq!(page.items.len(), 3, "{mode:?}");
            for (hit, (id, score)) in page.items.iter().zip(expected) {
                assert_eq!(hit.id.0, id, "{mode:?}");
                assert!(
                    ((hit.score as f64 - score) / score).abs() < 1e-5,
                    "{mode:?} record {} scored {} against {}",
                    id,
                    hit.score,
                    score
                );
            }
        }

        let mut index = index;
        let zero = SparseVector {
            dims: vec![1, 4],
            values: vec![1.0, 0.0],
        };
        assert!(matches!(
            index.insert(RecordId(4), zero.as_ref(), Prepared::none()),
            Err(Error::SparseValueNotPositive { index: 1, .. })
        ));
        assert_eq!(index.len(), 3);
    }

    /// Corpus statistics count admitted live records and their postings by
    /// every walk, agree with the global count under a set admitting
    /// everything, and answer `None` for a set that is only a predicate.
    #[test]
    fn corpus_statistics_count_under_every_shape_of_admit_set() {
        let mut index = index_of(
            &[
                (1, &[1, 2], &[1.0, 1.0]),
                (2, &[1], &[1.0]),
                (3, &[2, 3], &[1.0, 1.0]),
                (4, &[1, 3], &[1.0, 1.0]),
                (5, &[1, 2, 3], &[1.0, 1.0, 1.0]),
            ],
            Unlink::Strand,
        );
        index.remove(RecordId(4)).unwrap();
        let dims = [1u32, 2, 3, 9];
        assert_eq!(
            index.corpus_stats(&dims, &Candidates::All),
            Some(CorpusStats {
                documents: 4,
                df: vec![3, 3, 2, 0]
            })
        );
        // A bitmap admitting 1, 4 and 5, of which 4 is dead. The bitmap
        // walk and the arena walk both answer it.
        let mut bitmap = Bitmap::default();
        for slot in [1usize, 4, 5] {
            bitmap.insert(slot);
        }
        let expected = Some(CorpusStats {
            documents: 2,
            df: vec![2, 2, 1, 0],
        });
        assert_eq!(index.corpus_stats(&dims, &bitmap), expected);
        assert_eq!(
            index.stats_by_walk(&dims, &bitmap),
            expected.clone().unwrap()
        );
        let sorted = Candidates::Sorted(vec![RecordId(1), RecordId(4), RecordId(5)]);
        assert_eq!(index.corpus_stats(&dims, &sorted), expected);
        // Unsorted and repeated dimensions are answered in the order asked.
        assert_eq!(
            index.stats_by_enumerate(&[3, 1, 3], &sorted),
            Some(CorpusStats {
                documents: 2,
                df: vec![1, 2, 1]
            })
        );

        struct Odd;
        impl Admit for Odd {
            fn admits(&self, id: RecordId) -> bool {
                id.0 % 2 == 1
            }
            fn len_hint(&self) -> Option<usize> {
                None
            }
        }
        assert_eq!(index.corpus_stats(&dims, &Odd), None);
    }

    /// A score is a function of the corpus at the moment of the query and
    /// nothing is saved, so a removal that shares no term with the query
    /// still moves the mean length, and two records that differ in length
    /// and frequency can change places across it.
    #[test]
    fn an_unrelated_removal_can_reorder_a_page() {
        let mut index = index_with(
            &[
                (1, &[1], &[1.0]),
                (2, &[1, 2], &[2.0, 100.0]),
                (3, &[3], &[100_000.0]),
            ],
            Unlink::Lazy,
            Weighting::BM25,
        );
        let q = SparseVector {
            dims: vec![1],
            values: vec![1.0],
        };
        let before = index
            .search_mode(Mode::Auto, q.as_ref(), 10, &Candidates::All, false)
            .unwrap();
        let ids: Vec<u32> = before.items.iter().map(|h| h.id.0).collect();
        assert_eq!(ids, vec![2, 1], "the mean is huge, so length barely counts");
        index.remove(RecordId(3)).unwrap();
        let after = index
            .search_mode(Mode::Auto, q.as_ref(), 10, &Candidates::All, false)
            .unwrap();
        let ids: Vec<u32> = after.items.iter().map(|h| h.id.0).collect();
        assert_eq!(
            ids,
            vec![1, 2],
            "the mean fell, so the long record is penalised"
        );
        assert!(after.items[0].score != before.items[1].score);
    }

    /// Under a filter the rarity is counted over the admitted records by
    /// default and over every record on request, and the two rank a page
    /// differently where a term is common inside the filter and rare
    /// outside it.
    #[test]
    fn the_corpus_scope_changes_the_weighting_under_a_filter() {
        let mut rows: Vec<(u32, Vec<u32>, Vec<f32>)> = Vec::new();
        // Records 1 to 10 all carry term 1; only record 1 carries term 2.
        // Records 11 to 100 carry term 3 alone.
        for id in 1..=10u32 {
            let dims = if id == 1 { vec![1, 2] } else { vec![1] };
            let values = vec![1.0; dims.len()];
            rows.push((id, dims, values));
        }
        for id in 11..=100u32 {
            rows.push((id, vec![3], vec![1.0]));
        }
        let borrowed: Vec<(u32, &[u32], &[f32])> = rows
            .iter()
            .map(|(id, d, v)| (*id, d.as_slice(), v.as_slice()))
            .collect();
        let index = index_with(&borrowed, Unlink::Lazy, Weighting::BM25);
        let mut filter = Bitmap::default();
        for slot in 1..=10usize {
            filter.insert(slot);
        }
        let q = SparseVector {
            dims: vec![1, 2],
            values: vec![1.0, 1.0],
        };
        let corpus = index
            .search_scoped(Mode::Auto, q.as_ref(), 10, &filter, false, IdfScope::Corpus)
            .unwrap();
        let global = index
            .search_scoped(Mode::Auto, q.as_ref(), 10, &filter, false, IdfScope::Global)
            .unwrap();
        // Record 1 leads under both, but term 1's weight is near nothing
        // inside the filter and large across the index.
        assert_eq!(corpus.items[0].id, RecordId(1));
        assert_eq!(global.items[0].id, RecordId(1));
        assert!(corpus.items[1].score < global.items[1].score);
        let ratio = corpus.items[1].score / global.items[1].score;
        assert!(ratio < 0.2, "ratio {ratio}");
    }
}
