//! The search loops, the scoring rules, the corpus statistics, the top-k
//! selection, and the rule that chooses a loop.
//!
//! Every loop reads its working state by the key each posting carries; see
//! `PostingsIndex`. The sums, the dead set and the lengths are flat vectors
//! over the keys, and so is the set of admitted records a filtered scan
//! builds. Each is one read by key, whether the ids are dense or spread far
//! apart. A loop selects a page by key and names each hit by its id at the
//! end. Keys run in the order of the ids they stand for, so the two orders
//! are the same.

use std::cell::{OnceCell, RefCell};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use zeusdb_vector_core::{
    word_holds, Admit, Bitmap, CorpusStats, Error, Hit, Hits, IdfScope, RecordId, ScoreKind,
    SparseRef,
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
/// does not allocate and clear a buffer the size of the record table. One
/// slot for each key the index holds.
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
    fn ready(&mut self, keys: usize) {
        if self.acc.len() < keys {
            self.acc.resize(keys, f32::NAN);
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
    fn reset(&mut self, keys: usize) {
        if self.touched.len() > keys / 16 {
            self.acc[..keys].fill(f32::NAN);
        } else {
            for &key in &self.touched {
                self.acc[key as usize] = f32::NAN;
            }
        }
        self.touched.clear();
    }
}

/// An admit set as a search reads it by key.
///
/// A search tests a bitmap admit set as the words of a flat set over the
/// keys. These are the bitmap's own words where keys are ids and the bitmap
/// is flat. Otherwise they are the keys of the live records the bitmap
/// admits, built once for the search where a loop or a count finds them
/// worth building. The scan and the corpus count of one search share that
/// set. They also share the admitted count, since a bitmap counts its ids by
/// a walk.
struct Keyed<'a> {
    admit: &'a dyn Admit,
    admitted: OnceCell<Option<usize>>,
    built: OnceCell<Vec<u64>>,
}

impl<'a> Keyed<'a> {
    fn new(admit: &'a dyn Admit) -> Self {
        Keyed {
            admit,
            admitted: OnceCell::new(),
            built: OnceCell::new(),
        }
    }

    /// The set's `len_hint`, asked once.
    fn admitted(&self) -> Option<usize> {
        *self.admitted.get_or_init(|| self.admit.len_hint())
    }
}

/// The words a loop under a bitmap admit set tests each posting's key
/// against, and whether the dead set is already folded into them.
struct AdmitWords<'a> {
    words: &'a [u64],
    live: bool,
}

// ---------------------------------------------------------------------------
// Scoring rules. One posting, or one matched element of a record, at a time.
// ---------------------------------------------------------------------------

/// What one stored value contributes against one query value, for the
/// record it belongs to, named by its key. The scan and the arena merge both
/// call it, so the two paths agree bit for bit under every rule.
trait Scorer {
    fn score(&self, key: u32, stored: f32, query: f32) -> f32;
}

/// The product, which is the sparse dot product summed.
struct DotScorer;

impl Scorer for DotScorer {
    #[inline(always)]
    fn score(&self, _key: u32, stored: f32, query: f32) -> f32 {
        stored * query
    }
}

/// The saturated, length-normalised term frequency, with the query value
/// already carrying the term's rarity and the `k1 + 1` numerator, so the
/// per-posting work is one gather of the record's length by key, one
/// multiply-add and one division.
struct Bm25Scorer<'a> {
    lengths: &'a [f32],
    /// `k1 * (1 - b)`.
    c0: f32,
    /// `k1 * b / mean_length`.
    c1: f32,
}

impl Scorer for Bm25Scorer<'_> {
    #[inline(always)]
    fn score(&self, key: u32, tf: f32, query: f32) -> f32 {
        query * tf / (tf + self.c0 + self.c1 * self.lengths[key as usize])
    }
}

/// A record scored from the arena against the query by a merge over the
/// dimensions the two share, accumulated in ascending dimension order, which
/// is the order the term-at-a-time scan adds the same contributions in.
fn score_record<S: Scorer>(
    scorer: &S,
    key: u32,
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
                sum += scorer.score(key, a.values[i], b.values[j]);
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

/// The number of leading ids in `ids`, which increase, that are below `id`.
/// Doubling steps from the front and then a bisection, so ids asked in
/// increasing order and close together cost a step or two each.
fn below(ids: &[u32], id: usize) -> usize {
    let mut at = 0usize;
    let mut step = 1usize;
    while at < ids.len() && (ids[at] as usize) < id {
        let next = at + step;
        if next >= ids.len() || ids[next] as usize >= id {
            let end = next.min(ids.len());
            return at + 1 + ids[at + 1..end].partition_point(|&held| (held as usize) < id);
        }
        at = next;
        step *= 2;
    }
    at
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
    /// record, one unit per nonzero of the record and one per nonzero of
    /// the query.
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
            * (units.record_ns
                + self.mean_nnz() * units.merge_ns
                + query_nnz as f64 * units.query_ns);
        (scan_ns, enumerate_ns)
    }

    /// The dead set's words. The dead set is keyed by key, and it is flat in
    /// every form of the index.
    fn dead_words(&self) -> &[u64] {
        self.dead.as_flat().expect("the dead set is flat")
    }

    /// The words that a loop under `bitmap`, the set `keyed` holds, tests
    /// each key against. These are the bitmap's own words while keys are ids
    /// and the bitmap is flat, and the loop then tests the dead set beside
    /// them. Otherwise they are the keys of the live records the bitmap
    /// admits, built the first time they are asked for where they are worth
    /// building. `None` where they are not, and the loop then asks the bitmap
    /// about the id of each posting.
    ///
    /// The build takes one step for each id of the shorter of the bitmap and
    /// the keys, and zeroes one word for every 64 keys. The build is worth it
    /// where those steps and words are at most the postings that the lists of
    /// `dims` hold, since a test of the id of each posting costs about that.
    fn admit_words<'a>(
        &'a self,
        bitmap: &'a Bitmap,
        keyed: &'a Keyed<'_>,
        dims: &[u32],
    ) -> Option<AdmitWords<'a>> {
        if self.ranks.is_none() {
            if let Some(words) = bitmap.as_flat() {
                return Some(AdmitWords { words, live: false });
            }
        }
        let live = |words: &'a Vec<u64>| AdmitWords { words, live: true };
        if let Some(words) = keyed.built.get() {
            return Some(live(words));
        }
        let keys = self.keys();
        let admitted = keyed.admitted().unwrap_or(usize::MAX);
        let steps = admitted.min(keys).saturating_add(keys / 64);
        (steps <= self.scan_postings(SparseRef { dims, values: &[] })).then(|| {
            live(
                keyed
                    .built
                    .get_or_init(|| self.admitted_keys(bitmap, admitted)),
            )
        })
    }

    /// The keys of the live records `bitmap` admits, as the words of a flat
    /// set over the keys. `admitted` is how many ids the bitmap holds.
    ///
    /// While keys are ids, each id the bitmap holds is kept where the record
    /// table holds a record under it and the dead set does not. Once keys are
    /// ranks, the shorter of the two runs in increasing id order is walked:
    /// each id the bitmap holds is found among the ids by rank by a search
    /// forward from the last one found, or each rank's id is read from the
    /// bitmap through a cursor. A rank is kept where the dead set does not
    /// hold it.
    fn admitted_keys(&self, bitmap: &Bitmap, admitted: usize) -> Vec<u64> {
        let keys = self.keys();
        let mut words = vec![0u64; keys.div_ceil(64)];
        let dead = self.dead_words();
        match self.rank_ids() {
            None => {
                let records = self.records.flat_slots();
                bitmap.for_each(|id| {
                    if records.get(id).is_some_and(|slot| slot.held()) && !word_holds(dead, id) {
                        words[id >> 6] |= 1u64 << (id & 63);
                    }
                });
            }
            Some(ids) if admitted <= ids.len() => {
                let mut at = 0usize;
                bitmap.for_each(|id| {
                    at += below(&ids[at..], id);
                    if at < ids.len() && ids[at] as usize == id && !word_holds(dead, at) {
                        words[at >> 6] |= 1u64 << (at & 63);
                    }
                });
            }
            Some(ids) => {
                let mut held = bitmap.cursor();
                for (rank, &id) in ids.iter().enumerate() {
                    if held.contains(id as usize) && !word_holds(dead, rank) {
                        words[rank >> 6] |= 1u64 << (rank & 63);
                    }
                }
            }
        }
        words
    }

    /// Term-at-a-time accumulation with a per-posting predicate on the
    /// posting's key. Generic so the monomorphised, closure and trait-object
    /// arms share one body, and over the scoring rule.
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
            scratch.ready(self.keys());
            let Scratch { acc, touched } = &mut *scratch;
            for (d, &qw) in query.dims.iter().zip(query.values) {
                let Some(&slot) = self.slots_by_dim.get(d) else {
                    continue;
                };
                for p in &self.lists[slot as usize].postings {
                    if !admits(p.key) {
                        continue;
                    }
                    let a = &mut acc[p.key as usize];
                    let s = scorer.score(p.key, p.weight, qw);
                    if a.is_nan() {
                        *a = s;
                        touched.push(p.key);
                    } else {
                        *a += s;
                    }
                }
            }
            let page = select(acc, touched, k, boundary_ties, |_| true, self.rank_ids());
            scratch.reset(self.keys());
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
            scratch.ready(self.keys());
            let Scratch { acc, touched } = &mut *scratch;
            for (d, &qw) in query.dims.iter().zip(query.values) {
                let Some(&slot) = self.slots_by_dim.get(d) else {
                    continue;
                };
                for p in &self.lists[slot as usize].postings {
                    let a = &mut acc[p.key as usize];
                    let s = scorer.score(p.key, p.weight, qw);
                    if a.is_nan() {
                        *a = s;
                        touched.push(p.key);
                    } else {
                        *a += s;
                    }
                }
            }
            let dead = self.dead_words();
            let page = select(
                acc,
                touched,
                k,
                boundary_ties,
                |key| !word_holds(dead, key as usize) && admit.admits(RecordId(self.id_at(key))),
                self.rank_ids(),
            );
            scratch.reset(self.keys());
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
            if let Some((key, slot)) = self.live_slot(id) {
                let score = score_record(scorer, key as u32, self.forward(slot), query);
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
        let keyed = Keyed::new(admit);
        let items = match self.config.weighting {
            Weighting::Dot => self.run(&DotScorer, mode, query, k, &keyed, boundary_ties),
            Weighting::Bm25 { k1, b } => {
                let weighted = self.weigh(query, &keyed, idf, k1, b);
                let scorer = Bm25Scorer {
                    lengths: self.lengths.as_flat().expect("the lengths are flat"),
                    c0: weighted.c0,
                    c1: weighted.c1,
                };
                let query = SparseRef {
                    dims: &weighted.dims,
                    values: &weighted.values,
                };
                self.run(&scorer, mode, query, k, &keyed, boundary_ties)
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
        keyed: &Keyed<'_>,
        boundary_ties: bool,
    ) -> Vec<Hit> {
        let has_dead = self.dead_records > 0;
        let dead = self.dead_words();
        let admit = keyed.admit;
        match mode {
            Mode::Floor => self.scan_per_posting(scorer, query, k, boundary_ties, |_| true),
            Mode::PerPosting => self.scan_per_posting(scorer, query, k, boundary_ties, |key| {
                !word_holds(dead, key as usize) && admit.admits(RecordId(self.id_at(key)))
            }),
            Mode::BitmapPerPosting => {
                self.scan_bitmap(scorer, query, k, boundary_ties, keyed, has_dead)
            }
            Mode::PerCandidate => self.scan_per_candidate(scorer, query, k, boundary_ties, admit),
            Mode::Enumerate => {
                match self.enumerate_driven(scorer, query, k, boundary_ties, admit) {
                    Some(items) => items,
                    None => self.scan_per_candidate(scorer, query, k, boundary_ties, admit),
                }
            }
            Mode::Auto => self.auto(scorer, query, k, boundary_ties, keyed, has_dead),
        }
    }

    /// The bitmap-monomorphised scan, with the dead test only where a record
    /// has been removed since the last compaction and the bitmap's own words
    /// are read. Falls back to the table where the admit set is not a
    /// bitmap, or where a set of its keys is not worth building.
    fn scan_bitmap<S: Scorer>(
        &self,
        scorer: &S,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        keyed: &Keyed<'_>,
        has_dead: bool,
    ) -> Vec<Hit> {
        let dead = self.dead_words();
        let admit = keyed.admit;
        let words = admit
            .as_bitmap()
            .and_then(|bitmap| self.admit_words(bitmap, keyed, query.dims));
        match words {
            Some(AdmitWords { words, live }) if live || !has_dead => {
                self.scan_per_posting(scorer, query, k, boundary_ties, |key| {
                    word_holds(words, key as usize)
                })
            }
            Some(AdmitWords { words, .. }) => {
                self.scan_per_posting(scorer, query, k, boundary_ties, |key| {
                    word_holds(words, key as usize) && !word_holds(dead, key as usize)
                })
            }
            None if !has_dead => self.scan_per_posting(scorer, query, k, boundary_ties, |key| {
                admit.admits(RecordId(self.id_at(key)))
            }),
            None => self.scan_per_posting(scorer, query, k, boundary_ties, |key| {
                !word_holds(dead, key as usize) && admit.admits(RecordId(self.id_at(key)))
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
    /// answers no hint, is the scan with no predicate but the dead test,
    /// and the dead test only where a record has been removed since the
    /// last compaction. Asked through the table instead, as a set that
    /// answers no hint and is not a bitmap is, the scan paid one indirect
    /// call per posting for a predicate that is always true, and ran
    /// slower under no filter than under a bitmap admitting everything.
    fn auto<S: Scorer>(
        &self,
        scorer: &S,
        query: SparseRef<'_>,
        k: usize,
        boundary_ties: bool,
        keyed: &Keyed<'_>,
        has_dead: bool,
    ) -> Vec<Hit> {
        let admit = keyed.admit;
        if admit.admits_all() {
            let dead = self.dead_words();
            return if has_dead {
                self.scan_per_posting(scorer, query, k, boundary_ties, |key| {
                    !word_holds(dead, key as usize)
                })
            } else {
                self.scan_per_posting(scorer, query, k, boundary_ties, |_| true)
            };
        }
        let Some(admitted) = keyed.admitted() else {
            return match (admit.as_bitmap(), has_dead) {
                (None, false) => self.scan_per_posting(scorer, query, k, boundary_ties, |key| {
                    admit.admits(RecordId(self.id_at(key)))
                }),
                _ => self.scan_bitmap(
                    scorer,
                    query,
                    k,
                    boundary_ties,
                    keyed,
                    has_dead || self.dead.count() > 0,
                ),
            };
        };
        if self.prefers_enumerate(query.dims, admitted, admit.as_bitmap().is_some()) {
            if let Some(items) = self.enumerate_driven(scorer, query, k, boundary_ties, admit) {
                return items;
            }
        }
        self.scan_bitmap(scorer, query, k, boundary_ties, keyed, has_dead)
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
        self.stats_keyed(dims, &Keyed::new(admit))
    }

    /// [`PostingsIndex::stats_under`] on a set the search reads by key, so
    /// the set of admitted keys it builds serves the search after it.
    fn stats_keyed(&self, dims: &[u32], keyed: &Keyed<'_>) -> Option<CorpusStats> {
        let admit = keyed.admit;
        if admit.admits_all() {
            return Some(self.global_stats(dims));
        }
        let bitmap = admit.as_bitmap();
        let enumerate_first = keyed
            .admitted()
            .is_some_and(|admitted| self.prefers_enumerate(dims, admitted, bitmap.is_some()));
        if enumerate_first {
            if let Some(stats) = self.stats_by_enumerate(dims, admit) {
                return Some(stats);
            }
        }
        if let Some(bitmap) = bitmap {
            return Some(self.stats_by_walk(dims, bitmap, keyed));
        }
        self.stats_by_enumerate(dims, admit)
    }

    /// Each named list walked under the bitmap, the set `keyed` holds,
    /// counting the postings it admits by key, with the dead test only where
    /// a record has been removed since the last compaction and the bitmap's
    /// own words are read, or by the id at each posting's key where a set of
    /// its keys is not worth building. The document count is the
    /// intersection of the bitmap with the live set, taken a word at a time,
    /// or the count of the admitted keys. A set of keys built here serves
    /// the search that follows.
    fn stats_by_walk(&self, dims: &[u32], bitmap: &Bitmap, keyed: &Keyed<'_>) -> CorpusStats {
        let words = self.admit_words(bitmap, keyed, dims);
        let documents = match &words {
            Some(AdmitWords { words, live: true }) => {
                words.iter().map(|word| word.count_ones() as usize).sum()
            }
            _ => bitmap.count_and(&self.live_set),
        };
        let has_dead = self.dead_records > 0;
        let dead = self.dead_words();
        let df = dims
            .iter()
            .map(|d| {
                let Some(&slot) = self.slots_by_dim.get(d) else {
                    return 0;
                };
                let postings = self.lists[slot as usize].postings.iter();
                match &words {
                    Some(AdmitWords { words, live }) if *live || !has_dead => postings
                        .filter(|p| word_holds(words, p.key as usize))
                        .count(),
                    Some(AdmitWords { words, .. }) => postings
                        .filter(|p| {
                            word_holds(words, p.key as usize) && !word_holds(dead, p.key as usize)
                        })
                        .count(),
                    None => postings
                        .filter(|p| {
                            !word_holds(dead, p.key as usize)
                                && bitmap.contains(self.id_at(p.key) as usize)
                        })
                        .count(),
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
        keyed: &Keyed<'_>,
        idf: IdfScope,
        k1: f32,
        b: f32,
    ) -> Weighted {
        let stats = match idf {
            IdfScope::Global => self.global_stats(query.dims),
            IdfScope::Corpus => self
                .stats_keyed(query.dims, keyed)
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
// Top-k. Higher score wins, then lower key, which is lower id.
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
struct Cand {
    score: f32,
    key: u32,
}

impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cand {
    /// Greater is better: higher score, then lower key.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.key.cmp(&self.key))
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
    fn offer(&mut self, score: f32, key: u32) {
        if self.k == 0 {
            return;
        }
        let cand = Cand { score, key };
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

/// Select the page from the accumulator over the touched keys that pass
/// `admits`, keeping the boundary tie group where asked, and name each by
/// its id: the key itself while keys are ids, and the id at the rank in
/// `ids` once keys are ranks.
fn select<P: Fn(u32) -> bool>(
    acc: &[f32],
    touched: &[u32],
    k: usize,
    boundary_ties: bool,
    admits: P,
    ids: Option<&[u32]>,
) -> Vec<Hit> {
    let mut top = TopK::new(k);
    for &key in touched {
        let score = acc[key as usize];
        if score != 0.0 && admits(key) {
            top.offer(score, key);
        }
    }
    let (mut page, boundary) = top.finish();
    if let (true, Some(boundary)) = (boundary_ties, boundary) {
        // Every record tied at the boundary score that the heap cut, in key
        // order after the page's own members, so the page stays ordered by
        // score and then by key.
        let last_key = page.last().map(|c| c.key).unwrap_or(0);
        let mut extra: Vec<u32> = touched
            .iter()
            .copied()
            .filter(|&key| acc[key as usize] == boundary && key > last_key && admits(key))
            .collect();
        extra.sort_unstable();
        page.extend(extra.into_iter().map(|key| Cand {
            score: boundary,
            key,
        }));
    }
    page.into_iter()
        .map(|c| Hit {
            id: RecordId(ids.map_or(c.key, |ids| ids[c.key as usize])),
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
            index.stats_by_walk(&dims, &bitmap, &Keyed::new(&bitmap)),
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

    /// A space whose records sit under ids spread far apart holds its record
    /// table paged, its keys on ranks and its lengths and its dead set flat by
    /// rank, and answers in every mode, under every admit shape, both
    /// weightings and every unlink policy, the page the same records under
    /// dense ids answer, each id mapped across and the score bits equal,
    /// before and after removals and a compaction.
    #[test]
    fn a_space_over_scattered_ids_answers_what_dense_ids_answer() {
        let far = |id: u32| 7 + (id - 1) * 97_003;
        let record = |i: u32| -> SparseVector {
            let mut dims: Vec<u32> = (0..6).map(|j| (i * 7 + j * 11) % 60).collect();
            dims.sort_unstable();
            dims.dedup();
            let values = dims.iter().map(|&d| 1.0 + ((d + i) % 4) as f32).collect();
            SparseVector { dims, values }
        };
        let queries: Vec<SparseVector> = (0..12u32)
            .map(|q| {
                let dims = record(q * 13 + 5).dims;
                SparseVector {
                    values: vec![1.0; dims.len()],
                    dims,
                }
            })
            .collect();
        for weighting in [Weighting::Dot, Weighting::BM25] {
            for unlink in [Unlink::Strand, Unlink::Lazy, Unlink::Eager] {
                let config = SparseConfig {
                    unlink,
                    weighting,
                    ..SparseConfig::default()
                };
                let mut dense = PostingsIndex::new(config.clone());
                let mut scattered = PostingsIndex::new(config);
                for id in 1..=400u32 {
                    let v = record(id);
                    dense
                        .insert(RecordId(id), v.as_ref(), Prepared::none())
                        .unwrap();
                    scattered
                        .insert(RecordId(far(id)), v.as_ref(), Prepared::none())
                        .unwrap();
                }
                assert!(dense.records.is_flat());
                assert!(!scattered.records.is_flat() && scattered.ranks.is_some());
                assert!(scattered.lengths.is_flat());
                let check = |dense: &PostingsIndex, scattered: &PostingsIndex, label: &str| {
                    let held: Vec<u32> = (1..=400u32)
                        .filter(|&id| dense.holds(RecordId(id)))
                        .collect();
                    let mut third = Bitmap::default();
                    let mut third_far = Bitmap::paged();
                    for &id in held.iter().filter(|&&id| id % 3 != 1) {
                        third.insert(id as usize);
                        third_far.insert(far(id) as usize);
                    }
                    let fifty =
                        Candidates::Sorted(held.iter().take(50).map(|&id| RecordId(id)).collect());
                    let fifty_far = Candidates::Sorted(
                        held.iter().take(50).map(|&id| RecordId(far(id))).collect(),
                    );
                    let admits: [(&dyn Admit, &dyn Admit); 3] = [
                        (&Candidates::All, &Candidates::All),
                        (&third, &third_far),
                        (&fifty, &fifty_far),
                    ];
                    for (admit, admit_far) in admits {
                        for mode in MODES {
                            for q in &queries {
                                let want: Vec<(u32, u32)> = dense
                                    .search_mode(mode, q.as_ref(), 10, admit, true)
                                    .unwrap()
                                    .items
                                    .iter()
                                    .map(|hit| (far(hit.id.0), hit.score.to_bits()))
                                    .collect();
                                let got: Vec<(u32, u32)> = scattered
                                    .search_mode(mode, q.as_ref(), 10, admit_far, true)
                                    .unwrap()
                                    .items
                                    .iter()
                                    .map(|hit| (hit.id.0, hit.score.to_bits()))
                                    .collect();
                                assert_eq!(got, want, "{label} {weighting:?} {unlink:?} {mode:?}");
                            }
                        }
                    }
                };
                check(&dense, &scattered, "as built");
                for id in (1..=400u32).step_by(7) {
                    dense.remove(RecordId(id)).unwrap();
                    scattered.remove(RecordId(far(id))).unwrap();
                }
                assert!(scattered.dead.is_flat(), "the dead set is flat by rank");
                check(&dense, &scattered, "after removals");
                dense.compact();
                scattered.compact();
                check(&dense, &scattered, "after a compaction");
                assert_eq!(scattered.slots(), far(400) as usize + 1);
                assert_eq!(dense.slots(), 401);
                // Each record sits alone in its page, so each costs a page's
                // header and its entry, beside four bytes a page of the range,
                // where a flat table would cost twelve bytes an id below the
                // largest.
                let held = scattered.len();
                let pages = far(400) as usize / zeusdb_vector_core::PAGE_IDS + 1;
                let paged = scattered.heap_bytes();
                assert!(
                    paged.records <= 4 * pages + held * (80 + 2 + 8) + 256,
                    "{paged:?}"
                );
                assert!(
                    paged.lengths <= 4 * pages + held * (80 + 2 + 4) + 256,
                    "{paged:?}"
                );
                assert!(paged.records + paged.lengths < scattered.slots() / 100);
            }
        }
    }

    /// A record whose sum turns NaN partway through a scan counts as not yet
    /// started at its next contribution and is listed as touched again,
    /// under the slot per id and under the slot per rank alike, so the
    /// page over scattered ids is the page over dense ids, the record twice on
    /// it in both. Two finite products overflow to infinities of opposite
    /// sign, and their sum is NaN.
    #[test]
    fn a_sum_that_turns_nan_restarts_alike_in_both_accumulators() {
        let rows: [(&[u32], &[f32]); 3] = [
            (&[1, 2, 3], &[3.0e38, 3.0e38, 1.0]),
            (&[1, 3], &[1.0, 2.0]),
            (&[2, 3], &[1.0, 0.5]),
        ];
        let query = SparseVector {
            dims: vec![1, 2, 3],
            values: vec![2.0, -2.0, 1.0],
        };
        let far = |id: u32| 9 + (id - 1) * 1_000_003;
        let mut dense = PostingsIndex::new(SparseConfig::default());
        let mut scattered = PostingsIndex::new(SparseConfig::default());
        for (k, (dims, values)) in rows.iter().enumerate() {
            let v = SparseVector {
                dims: dims.to_vec(),
                values: values.to_vec(),
            };
            let id = k as u32 + 1;
            dense
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
            scattered
                .insert(RecordId(far(id)), v.as_ref(), Prepared::none())
                .unwrap();
        }
        assert!(!scattered.records.is_flat());
        for mode in [
            Mode::Auto,
            Mode::PerPosting,
            Mode::PerCandidate,
            Mode::BitmapPerPosting,
            Mode::Floor,
        ] {
            let want: Vec<(u32, u32)> = dense
                .search_mode(mode, query.as_ref(), 10, &Candidates::All, false)
                .unwrap()
                .items
                .iter()
                .map(|hit| (far(hit.id.0), hit.score.to_bits()))
                .collect();
            let got: Vec<(u32, u32)> = scattered
                .search_mode(mode, query.as_ref(), 10, &Candidates::All, false)
                .unwrap()
                .items
                .iter()
                .map(|hit| (hit.id.0, hit.score.to_bits()))
                .collect();
            assert_eq!(got, want, "{mode:?}");
            assert_eq!(
                got.iter().filter(|(id, _)| *id == far(1)).count(),
                2,
                "{mode:?}: the record whose sum restarted is listed twice"
            );
        }
    }

    /// A set that admits the ids it holds, asked one id at a time through
    /// the table, which cannot count or enumerate itself.
    struct Within(std::collections::HashSet<u32>);

    impl Admit for Within {
        fn admits(&self, id: RecordId) -> bool {
            self.0.contains(&id.0)
        }

        fn len_hint(&self) -> Option<usize> {
            None
        }
    }

    /// A space whose record table is paged, so its keys are ranks, answers
    /// what the same records under dense ids answer, each id mapped across
    /// and the score bits equal, in every mode, under both weightings and
    /// both corpus scopes, with and without the boundary tie group: with no
    /// filter, under a filter bitmap held flat and one held paged, each also
    /// naming ids no record holds, under a sorted list and under a predicate
    /// asked through the table, as built, with removed records and after a
    /// compaction. At one record in twenty ids a flat bitmap meets a paged
    /// table, and far apart both page. A space whose table is flat answers
    /// the same under a paged bitmap as under a flat one.
    #[test]
    fn a_scan_by_rank_answers_what_a_scan_by_id_answers_under_every_admit_form() {
        let record = |i: u32| -> SparseVector {
            let mut dims: Vec<u32> = (0..7).map(|j| (i * 5 + j * 9) % 48).collect();
            dims.sort_unstable();
            dims.dedup();
            let values = dims.iter().map(|&d| 1.0 + ((d + i) % 5) as f32).collect();
            SparseVector { dims, values }
        };
        let mut queries: Vec<SparseVector> = (0..10u32)
            .map(|q| {
                let dims = record(q * 11 + 3).dims;
                SparseVector {
                    values: dims.iter().map(|&d| 0.5 + (d % 3) as f32).collect(),
                    dims,
                }
            })
            .collect();
        // One list, whose postings are too few to build a set of keys for,
        // and every list, whose postings are enough.
        queries.push(SparseVector {
            dims: vec![9],
            values: vec![1.0],
        });
        queries.push(SparseVector {
            dims: (0..48).collect(),
            values: vec![1.0; 48],
        });
        let twentieth: fn(u32) -> u32 = |id| 5 + (id - 1) * 20;
        let apart: fn(u32) -> u32 = |id| 11 + (id - 1) * 50_021;
        for (label, far) in [("one in twenty", twentieth), ("far apart", apart)] {
            for weighting in [Weighting::Dot, Weighting::BM25] {
                let config = SparseConfig {
                    unlink: Unlink::Strand,
                    weighting,
                    ..SparseConfig::default()
                };
                let mut dense = PostingsIndex::new(config.clone());
                let mut ranked = PostingsIndex::new(config);
                for id in 1..=500u32 {
                    let v = record(id);
                    dense
                        .insert(RecordId(id), v.as_ref(), Prepared::none())
                        .unwrap();
                    ranked
                        .insert(RecordId(far(id)), v.as_ref(), Prepared::none())
                        .unwrap();
                }
                assert!(dense.ranks.is_none() && ranked.ranks.is_some(), "{label}");
                let check = |dense: &PostingsIndex, ranked: &PostingsIndex, step: &str| {
                    assert!(dense.keys_agree() && ranked.keys_agree(), "{label} {step}");
                    let picked: Vec<u32> = (1..=500u32)
                        .filter(|&id| id % 3 != 1 && dense.holds(RecordId(id)))
                        .collect();
                    let mut flat = Bitmap::default();
                    let mut paged = Bitmap::paged();
                    let mut flat_far = Bitmap::default();
                    let mut paged_far = Bitmap::paged();
                    for &id in &picked {
                        flat.insert(id as usize);
                        paged.insert(id as usize);
                        flat_far.insert(far(id) as usize);
                        paged_far.insert(far(id) as usize);
                    }
                    // Ids beside every record's, which no record holds.
                    for id in 1..=500u32 {
                        flat_far.insert(far(id) as usize + 1);
                        paged_far.insert(far(id) as usize + 1);
                    }
                    let sorted = Candidates::Sorted(
                        picked.iter().take(60).map(|&id| RecordId(id)).collect(),
                    );
                    let sorted_far = Candidates::Sorted(
                        picked
                            .iter()
                            .take(60)
                            .map(|&id| RecordId(far(id)))
                            .collect(),
                    );
                    let within = Within(picked.iter().copied().collect());
                    let within_far = Within(picked.iter().map(|&id| far(id)).collect());
                    let admits: [(&dyn Admit, &dyn Admit, &str); 6] = [
                        (&Candidates::All, &Candidates::All, "everything"),
                        (&flat, &flat_far, "a flat bitmap"),
                        (&flat, &paged_far, "a paged bitmap"),
                        (&paged, &flat_far, "a paged bitmap over the flat table"),
                        (&sorted, &sorted_far, "a sorted list"),
                        (&within, &within_far, "a predicate"),
                    ];
                    for (admit, admit_far, shape) in admits {
                        for mode in MODES {
                            for scope in [IdfScope::Corpus, IdfScope::Global] {
                                for ties in [true, false] {
                                    for q in &queries {
                                        let want: Vec<(u32, u32)> = dense
                                            .search_scoped(mode, q.as_ref(), 10, admit, ties, scope)
                                            .unwrap()
                                            .items
                                            .iter()
                                            .map(|hit| (far(hit.id.0), hit.score.to_bits()))
                                            .collect();
                                        let got: Vec<(u32, u32)> = ranked
                                            .search_scoped(
                                                mode,
                                                q.as_ref(),
                                                10,
                                                admit_far,
                                                ties,
                                                scope,
                                            )
                                            .unwrap()
                                            .items
                                            .iter()
                                            .map(|hit| (hit.id.0, hit.score.to_bits()))
                                            .collect();
                                        assert_eq!(
                                            got, want,
                                            "{label} {weighting:?} {step} {shape} {mode:?} {scope:?} {ties}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                };
                check(&dense, &ranked, "as built");
                for id in (1..500u32).step_by(7) {
                    dense.remove(RecordId(id)).unwrap();
                    ranked.remove(RecordId(far(id))).unwrap();
                }
                check(&dense, &ranked, "with removed records");
                dense.compact();
                ranked.compact();
                check(&dense, &ranked, "compacted");
            }
        }
    }

    /// Once a space's record table is paged, the ids by rank cost four bytes
    /// a record and the rank map its own bytes, each record alone in its page
    /// costing a page's header, its offset and its rank beside four bytes a
    /// page of the range; the lengths cost four bytes a rank and the dead set
    /// a bit a rank. A restore holds the ids, the lengths and the rank map at
    /// exactly its records. A scan's accumulator, on a thread of its own,
    /// holds one slot a record and not one an id of the range, and the set of
    /// admitted keys a filtered scan builds holds one bit a record.
    #[test]
    fn the_ranks_and_the_scan_state_cost_what_the_records_cost() {
        let far = |k: u32| 7 + k * 100_003;
        let mut index = PostingsIndex::new(SparseConfig {
            unlink: Unlink::Strand,
            ..SparseConfig::default()
        });
        for k in 0..300u32 {
            let v = SparseVector {
                dims: vec![k % 11, 20 + k % 7],
                values: vec![1.0, 1.0 + k as f32],
            };
            index
                .insert(RecordId(far(k)), v.as_ref(), Prepared::none())
                .unwrap();
        }
        for k in (0..299u32).step_by(10) {
            index.remove(RecordId(far(k))).unwrap();
        }
        let held = index.records.len();
        let pages = far(299) as usize / zeusdb_vector_core::PAGE_IDS + 1;
        let ranks = index.ranks.as_ref().unwrap();
        let heap = index.heap_bytes();
        assert_eq!(heap.ranks, 4 * ranks.ids.capacity() + ranks.of.heap_bytes());
        assert!(ranks.ids.capacity() <= 2 * held, "{heap:?}");
        assert!(
            index.dead.heap_bytes() <= 16 * held.div_ceil(64),
            "{heap:?}"
        );
        assert!(heap.lengths <= 8 * held, "{heap:?}");

        let bytes = crate::persist::encode(&index);
        let bounds = zeusdb_vector_core::Bounds {
            min_records: 0,
            max_records: far(299) as usize,
            max_bytes: 1 << 30,
        };
        let restored = crate::persist::decode(&bytes, index.config(), &bounds, "postings").unwrap();
        let live = restored.len();
        let ranks = restored.ranks.as_ref().unwrap();
        assert_eq!(ranks.ids.capacity(), live);
        assert_eq!(restored.lengths.heap_bytes(), 4 * live);
        assert_eq!(restored.dead.heap_bytes(), 0);
        let heap = restored.heap_bytes();
        assert_eq!(heap.ranks, 4 * live + ranks.of.heap_bytes());
        assert!(
            ranks.of.heap_bytes() <= 4 * pages + live * (80 + 2 + 4) + 256,
            "{heap:?}"
        );
        assert!(
            heap.records + heap.lengths + heap.ranks < 64 * 1024,
            "{heap:?}"
        );

        let query = SparseVector {
            dims: vec![3, 21],
            values: vec![1.0, 1.0],
        };
        let mut filter = Bitmap::paged();
        for k in (1..300u32).step_by(2) {
            filter.insert(far(k) as usize);
        }
        assert_eq!(
            restored.admitted_keys(&filter, filter.count()).len(),
            live.div_ceil(64)
        );
        let slots = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    restored
                        .search_mode(Mode::Auto, query.as_ref(), 10, &Candidates::All, false)
                        .unwrap();
                    restored
                        .search_mode(Mode::Auto, query.as_ref(), 10, &filter, false)
                        .unwrap();
                    SCRATCH.with(|scratch| scratch.borrow().acc.len())
                })
                .join()
                .unwrap()
        });
        assert_eq!(slots, live);
    }

    /// The keys a filter bitmap admits are the live records it holds, keyed
    /// by id while the record table is flat and by rank once it is paged, by
    /// either walk the build can take and over a flat and a paged bitmap,
    /// with ids no record holds and removed records left out. The set is
    /// built where its steps are at most the postings the query's lists
    /// hold, once a search, and not built where they are more.
    #[test]
    fn the_admitted_keys_are_the_live_records_a_bitmap_admits_by_either_walk() {
        let far = |k: u32| 3 + k * 70_001;
        let mut ranked = PostingsIndex::new(SparseConfig::default());
        let mut flat = PostingsIndex::new(SparseConfig::default());
        for k in 0..400u32 {
            let v = SparseVector {
                dims: vec![k % 13, 13 + k % 17],
                values: vec![1.0, 2.0],
            };
            ranked
                .insert(RecordId(far(k)), v.as_ref(), Prepared::none())
                .unwrap();
            flat.insert(RecordId(k + 1), v.as_ref(), Prepared::none())
                .unwrap();
        }
        for k in (0..399u32).step_by(9) {
            ranked.remove(RecordId(far(k))).unwrap();
            flat.remove(RecordId(k + 1)).unwrap();
        }
        assert!(ranked.ranks.is_some() && flat.ranks.is_none());
        let expected = |index: &PostingsIndex, bitmap: &Bitmap| -> Vec<usize> {
            (0..index.keys())
                .filter(|&key| {
                    let id = index.id_at(key as u32) as usize;
                    index.records.contains(id) && !index.dead.contains(key) && bitmap.contains(id)
                })
                .collect()
        };
        let set = |words: &[u64]| -> Vec<usize> {
            (0..words.len() * 64)
                .filter(|&key| word_holds(words, key))
                .collect()
        };
        for few in [true, false] {
            for paged in [false, true] {
                let mut bitmap = if paged {
                    Bitmap::paged()
                } else {
                    Bitmap::default()
                };
                // Id 0, which no record holds, below the slot count.
                let mut by_id = Bitmap::paged();
                by_id.insert(0);
                for k in (0..400u32).filter(|&k| if few { k % 50 == 7 } else { k % 4 != 0 }) {
                    bitmap.insert(far(k) as usize);
                    by_id.insert(k as usize + 1);
                }
                if !few {
                    // Ids past every record's, which no record holds.
                    for k in 0..2_000usize {
                        bitmap.insert(far(399) as usize + 1 + k);
                        by_id.insert(1_000 + k);
                    }
                }
                let label = format!("few {few} paged {paged}");
                let want = expected(&ranked, &bitmap);
                assert!(!want.is_empty(), "{label}");
                assert_eq!(
                    set(&ranked.admitted_keys(&bitmap, bitmap.count())),
                    want,
                    "{label}"
                );
                for admitted in [0, usize::MAX] {
                    assert_eq!(
                        set(&ranked.admitted_keys(&bitmap, admitted)),
                        want,
                        "{label}, the walk for {admitted}"
                    );
                }
                assert_eq!(
                    set(&flat.admitted_keys(&by_id, by_id.count())),
                    expected(&flat, &by_id),
                    "{label}, keys by id"
                );
            }
        }
        let mut bitmap = Bitmap::paged();
        for k in 0..400u32 {
            bitmap.insert(far(k) as usize);
        }
        let keyed = Keyed::new(&bitmap);
        let narrow = [0u32];
        let wide: Vec<u32> = (0..30).collect();
        assert!(
            ranked.scan_postings(SparseRef {
                dims: &narrow,
                values: &[]
            }) < 406
        );
        assert!(
            ranked.scan_postings(SparseRef {
                dims: &wide,
                values: &[]
            }) >= 406
        );
        assert!(ranked.admit_words(&bitmap, &keyed, &narrow).is_none());
        assert!(keyed.built.get().is_none());
        assert!(ranked
            .admit_words(&bitmap, &keyed, &wide)
            .is_some_and(|words| words.live));
        assert!(keyed.built.get().is_some());
        assert!(
            ranked.admit_words(&bitmap, &keyed, &narrow).is_some(),
            "a set built once serves the rest of the search"
        );
    }
}
