//! Running a search, on one query or on a batch.
//!
//! Four paths reach the graph: one query, a batch of five or fewer run in turn,
//! a larger batch fanned across rayon, and the benchmark. They differ only in
//! how long they hold the storage guards, so what each does with a candidate is
//! in `collect_hits` and what each hands back to Python is in `hits_to_python`.
//!
//! # A filter now decides the page rather than trimming it
//!
//! Until relay 86 the filter ran in `collect_hits`, which is after the graph
//! has already cut to `top_k`. A filter matching one record in a hundred
//! therefore discarded most of a ten result page and returned what was left,
//! and one matching one record in a thousand returned nothing at all. Relay 85
//! measured that as post-filter recall of 0.0090 at one match in a hundred and
//! 0.0000 below it, on all three real sets.
//!
//! Two paths replace it, and [`HNSWIndex::search_candidates`] picks between
//! them per search.
//!
//! [`HNSWIndex::scan_candidates`] walks the metadata once and scores every
//! record that matches, which is exact and has no recall question. It gives up
//! and returns `None` the moment the match count passes
//! [`FULL_SCAN_THRESHOLD`], so a broad filter pays a bounded walk rather than a
//! full one.
//!
//! Where the scan gives up, the traversal runs with the filter conjoined into
//! the predicate it already carried for liveness. That is the correct answer
//! for a broad filter and a latency pathology for a selective one, which is why
//! the scan is not optional. A filter matching one record in 100,000 costs 128
//! to 359 milliseconds through the traversal and 26 through the scan, both
//! measured on this code.
//!
//! **A filter over declared fields does not walk anything.** The fields named
//! at `create(indexed_fields=[...])` each carry a column addressed by internal
//! id, so a filter over them compiles to a bitmap and both paths read it: the
//! exact scan takes the set bits and the traversal tests one bit per node. A
//! filter naming a field that was not declared has no column to read and takes
//! the walk described below, which is what every filtered search did before the
//! columns existed.
//!
//! **The walk is what made a filtered search expensive.** It costs about 250
//! nanoseconds a record, so a selective filter over 100,000 records cost about
//! 25 milliseconds where an unfiltered search costs 0.3 to 1.2. See
//! [`FULL_SCAN_THRESHOLD`] for the measurements and
//! [`crate::columns`] for what replaces it.
//!
//! **Lock order.** Every path takes `rev_map` before the graph and the storage
//! maps after it, which is the order declared on `HNSWIndex` in the parent
//! module. A search holds `rev_map` for its whole traversal, so a mutation
//! taking `vectors` before `rev_map` deadlocks against it, and that is the
//! inversion `remove_point_internal` used to carry.
//!
//! The filter predicate reads `vector_metadata`, so that guard is now held
//! across the traversal too, and so are `vectors` and `pq_codes` because the
//! scan runs before the traversal and scores against them. The declared order
//! is `hnsw < vectors < pq_codes < vector_metadata`, so taking all four before
//! traversing is in order rather than against it, and it is the shape
//! `batch_search_sequential` already ran in. What it costs is duration. The
//! window a writer waits on grows from the post-traversal phase to the whole
//! search, and `std::sync::RwLock` queues readers behind a waiting writer, so a
//! long read hold delays the next reader as well as the writer.

use super::{HNSWIndex, StorageMode, VectorGraph};
use crate::columns::{Bitmap, ColumnStore};
use crate::conversion::value_map_to_python;
use crate::filter::{matches_filter, Filter};
use crate::graph::GraphHit;
use crate::rerank::{
    raw_distance_fn, rescore_candidate, take_best, RawVectors, RerankPlan, SearchParams,
};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error, instrument, trace, warn};
/// Records a filtered search may match before it stops scanning and traverses
/// the graph instead.
///
/// # What the two paths actually cost
///
/// Relay 86 built the crate twice, once with this at zero so every filtered
/// search traverses and once with it above the corpus so every one scans, and
/// measured both at each selectivity on all three real sets at 100,000 records.
/// Microseconds per query, minimum of two passes over thirty queries.
///
/// | Matched | Traversal, sift | Scan, sift | Traversal, glove | Scan, glove | Traversal, dbpedia | Scan, dbpedia |
/// |---:|---:|---:|---:|---:|---:|---:|
/// | 50,000 | 1,141 | 45,664 | 1,166 | 44,543 | 4,658 | 79,511 |
/// | 10,000 | 4,514 | 31,782 | 4,786 | 30,406 | 15,726 | 37,801 |
/// | 1,000 | 36,023 | 25,954 | 35,641 | 26,113 | 74,143 | 27,175 |
/// | 100 | 182,224 | 25,121 | 235,577 | 25,760 | 384,844 | 25,738 |
/// | 10 | 77,422 | 27,954 | 303,811 | 26,241 | 313,549 | 26,612 |
/// | 1 | 128,110 | 26,509 | 296,585 | 26,117 | 358,556 | 26,333 |
///
/// **The crossover is between 1,000 and 10,000 matches, on all three sets.**
/// Below it the scan wins by 1.4 to 15 times and is exact. Above it the
/// traversal wins by 2.4 to 6.7 times. This sits inside that bracket.
///
/// # Why the upper half of the bracket
///
/// The scan's cost is the metadata walk and almost nothing else. Every scan row
/// above sits at 25 to 28 milliseconds whatever matched, except the two broad
/// ones where the matched records' own distances add to it, and 25 milliseconds
/// is 100,000 records at about 250 nanoseconds each. **That walk, and not the
/// distances, is what a filtered search pays for.** Relay 85 measured the same
/// per-record figure and named the same cause.
///
/// A filter matching just over the threshold is therefore the worst case, since
/// it walks the whole corpus, gives up, and then traverses as well. It pays the
/// full walk plus `traversal(threshold)`, so the regret against having scanned
/// is `traversal(threshold)` itself. At 5,000 that is roughly 9 milliseconds on
/// top of 25. At 1,000 it would be 36 to 74 milliseconds on top of 25, because
/// the traversal is far worse at 1,000 matches than at 5,000. **A lower
/// threshold is worse, not safer**, which is the opposite of what the shape of
/// the problem suggests.
///
/// Nothing above the bracket is chosen because a broad filter is where the
/// give-up saves most. A filter matching half the corpus reaches 5,001 matches
/// after about 10,000 records, being 2.5 milliseconds, and then traverses in
/// another 1.1 to 4.7. Doubling the threshold doubles that walk.
///
/// # What this does not settle
///
/// Every figure above is at 100,000 records. The walk is linear in the corpus
/// and the traversal is not, so the crossover moves with corpus size and this
/// relay did not measure where. Relay 85 recommended 5,000 from the crossover
/// between a scan and an **unfiltered** search, which is not the comparison
/// that matters, and then said so itself. The number it named survives on this
/// evidence rather than on that reasoning.
///
/// It is not settable per search. Nothing measured here wants a different
/// number, and a knob nobody has a reason to turn is a knob that goes wrong.
pub(super) const FULL_SCAN_THRESHOLD: usize = 5_000;

/// One search's candidates, as borrowed external ids paired with the distance
/// whichever path found them scored them at.
///
/// Both paths produce this. The traversal resolves each graph hit through
/// `rev_map` and borrows the id it finds there, and the scan borrows the id
/// from the metadata store's own key. Nothing is cloned until the page is cut,
/// so an over-fetched page pays to clone the metadata and the vector of the
/// results it returns rather than of every candidate it considered.
type Candidates<'a> = Vec<(&'a String, f32)>;

/// Everything a filtered search reads that an unfiltered one does not.
///
/// The conditions, the columns that answer them, the metadata store they are
/// judged against where a column cannot, and the two stores the exact scan
/// scores a matching record from. They travel together because a search carries
/// either all of them or none, and because that is what decides which guards a
/// path has to take: `raw_search_no_gil` passes `None` and takes neither the
/// metadata guard, the columns guard nor the two storage guards.
#[derive(Clone, Copy)]
struct Filtered<'a> {
    conditions: &'a Filter,
    columns: &'a ColumnStore,
    metadata: &'a HashMap<String, HashMap<String, Value>>,
    vectors: RawVectors<'a>,
    pq_codes: &'a HashMap<String, Vec<u8>>,
}

impl<'a> Filtered<'a> {
    /// Whether the record this external id names exists and matches.
    ///
    /// A record with no metadata entry does not match, which is the rule
    /// `collect_hits` applied when the filter ran there.
    #[inline]
    fn admits(&self, ext_id: &String) -> bool {
        match self.metadata.get(ext_id) {
            Some(meta) => self.judge(meta),
            None => false,
        }
    }

    /// Whether one record's metadata matches, with no lookup.
    ///
    /// **There is no error channel and none is needed.** What reaches here is a
    /// compiled [`Filter`], which `compile_filter` built from the caller's
    /// mapping once per search and before any record was examined, rejecting
    /// every operator name and every group shape the engine cannot evaluate.
    /// Nothing below `matches_filter` can fail, so it returns `bool`. This was
    /// a `match` on a `PyResult` whose `Err` arm carried a debug assertion
    /// explaining why it could not fire.
    #[inline]
    fn judge(&self, meta: &HashMap<String, Value>) -> bool {
        matches_filter(meta, self.conditions)
    }
}

/// Search hits for one query vector, as (external id, distance, metadata,
/// optional raw vector). The raw vector is present only when the caller asked
/// for it and the index still holds one.
pub(super) type QueryHits = Vec<(String, f32, HashMap<String, Value>, Option<Vec<f32>>)>;
impl HNSWIndex {
    // 2. SEARCH OPERATIONS (2 methods)

    /// Decide whether a search reranks, and how far it over-fetches
    ///
    /// Rerank rescores the candidates the graph returns against raw vectors,
    /// so it needs a raw vector for every candidate. Three cases resolve to no
    /// rerank.
    ///
    /// A raw index already ranks by the raw distance, so over-fetching and
    /// rescoring would return the same page at a higher cost.
    ///
    /// A `quantized_only` index holds no raw vectors once trained, the
    /// training records included, so the only thing available to rescore any
    /// candidate against is its reconstruction, and that carries exactly the
    /// information the ADC distance already used. Measured at 10,000 records
    /// of dimension 768, recall at `top_k` 10 over code held records moved
    /// from 0.1320 to 0.1330 across one data seed and from 0.1440 to 0.1400
    /// across another, which is noise in both directions.
    ///
    /// `rerank = 0` from the caller turns it off and restores the ADC scores.
    pub(super) fn rerank_plan(&self, rerank: Option<usize>) -> Option<RerankPlan> {
        if rerank == Some(0) || !self.is_quantized() {
            return None;
        }

        let keeps_raw = self
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        if !keeps_raw {
            return None;
        }

        Some(RerankPlan {
            factor: rerank.map(|factor| factor.max(1)),
            calibration: self.get_rerank_calibration(),
            distance: raw_distance_fn(&self.space),
        })
    }

    // 5. BATCH SEARCH METHODS (3 methods)

    /// The candidates a filtered search's exact scan found, or `None` where
    /// there were too many of them
    ///
    /// One walk of the metadata store, evaluating the filter on each record and
    /// keeping the ones that match. It gives up and returns `None` the moment
    /// the match count passes [`FULL_SCAN_THRESHOLD`], so the walk **stops**
    /// rather than completing: a filter matching half the corpus reaches the
    /// give-up point after roughly twice the threshold in records examined,
    /// whatever the corpus size, and the caller then traverses.
    ///
    /// **The distances are computed after the walk and not during it.** Every
    /// distance a fused walk evaluated before giving up would be thrown away,
    /// and at 1,536 dimensions that is thousands of six kilobyte reads charged
    /// to every broad filtered search. Deferring them costs one extra pass over
    /// a list of at most `FULL_SCAN_THRESHOLD` borrowed ids and makes the
    /// give-up path free of distance work entirely. The relay that specified
    /// this described one fused pass; the results are identical either way and
    /// this is the cheaper of the two.
    ///
    /// **What it scores against.** The stored raw vector where the index holds
    /// one and the reconstruction from the record's codes where it does not,
    /// which is `rescore_candidate`'s rule and therefore the same scale a
    /// reranked search orders on. Three cases follow. A raw index is scored on
    /// its raw vectors, so the scan's scores are the numbers the traversal
    /// would have reported. A `quantized_with_raw` index is scored on its raw
    /// vectors, which is exact, and matches what rerank produces on the default
    /// path; a caller who passed `rerank=0` asked for ADC scores and gets raw
    /// ones, which is the more accurate of the two and is the one deliberate
    /// difference here. A `quantized_only` index holds no raw vectors, so every
    /// record is scored against its reconstruction, which carries exactly the
    /// information the ADC distance uses. The scan is exact over what that
    /// index retains rather than over the vectors it was given, and no path in
    /// this crate can be exact over those once the raws are gone.
    ///
    /// **Fewer than `top_k` matches is a result and not a failure.** The scan
    /// returns every record that matched, ranked, so a filter matching three
    /// records returns three results to a caller who asked for ten, and a
    /// filter matching none returns an empty page. That is what the search
    /// already did and it is what an exact answer to the question is.
    fn scan_candidates<'a>(
        &self,
        query: &[f32],
        filter: Filtered<'a>,
        fetch_k: usize,
    ) -> Option<Candidates<'a>> {
        let mut matched: Vec<&'a String> = Vec::new();
        for (ext_id, meta) in filter.metadata.iter() {
            if filter.judge(meta) {
                if matched.len() == FULL_SCAN_THRESHOLD {
                    // One past the threshold, so the walk stops here and the
                    // caller traverses. Nothing has been scored yet.
                    return None;
                }
                matched.push(ext_id);
            }
        }

        self.score_matched(query, matched, filter, fetch_k)
    }

    /// The candidates a selected bitmap names, scored and ranked.
    ///
    /// The counterpart of [`Self::scan_candidates`] for a filter every field of
    /// which is declared. The bitmap already holds the whole matching set, so
    /// this resolves each set bit through `rev_map` and hands the ids to the
    /// same scorer. **A slot the bitmap holds always resolves**, because
    /// `ColumnStore::erase` clears a slot in the same write that removes it from
    /// `rev_map`; a slot that somehow did not resolve is dropped, which is the
    /// liveness rule every other path applies.
    fn scan_selected<'a>(
        &self,
        query: &[f32],
        selected: &Bitmap,
        rev_map: &'a HashMap<usize, String>,
        filter: Filtered<'a>,
        fetch_k: usize,
    ) -> Candidates<'a> {
        let mut matched: Vec<&'a String> = Vec::with_capacity(selected.count());
        selected.for_each(|slot| {
            if let Some(ext_id) = rev_map.get(&slot) {
                matched.push(ext_id);
            }
        });
        // Infallible here. `score_matched` returns an `Option` only so that the
        // walk above can express its give-up point, and the caller has already
        // decided this set is small enough to score.
        self.score_matched(query, matched, filter, fetch_k)
            .unwrap_or_default()
    }

    /// Score a matched set and cut it to the page.
    ///
    /// Both paths that produce an exact answer end here, so the ranking and the
    /// tie break are stated once and neither path can drift from the other.
    fn score_matched<'a>(
        &self,
        query: &[f32],
        matched: Vec<&'a String>,
        filter: Filtered<'a>,
        fetch_k: usize,
    ) -> Option<Candidates<'a>> {
        let distance = raw_distance_fn(&self.space);
        let mut scored: Candidates<'a> = Vec::with_capacity(matched.len());
        for ext_id in matched {
            let score = match filter.vectors.get(ext_id.as_str()) {
                Some(stored) => distance(query, stored),
                None => match filter
                    .pq_codes
                    .get(ext_id)
                    .and_then(|codes| self.pq.as_ref()?.reconstruct(codes).ok())
                {
                    Some(reconstructed) => distance(query, &reconstructed),
                    // A record holding neither a raw vector nor codes cannot
                    // exist, since every insertion writes one or the other.
                    // Sorting it last is what `rescore_candidate`'s callers do
                    // with the same impossibility.
                    None => f32::INFINITY,
                },
            };
            scored.push((ext_id, score));
        }
        // By distance, and by external id where two distances are equal.
        //
        // The tie break is what makes this page reproducible. The walk takes
        // the metadata store in `HashMap` iteration order, which `RandomState`
        // seeds afresh in every process, so a stable sort on distance alone
        // would hand back two equally distant records in an order that differed
        // run to run. The graph path has no such problem because a traversal of
        // a fixed graph is deterministic. Two exactly equal distances are
        // unordered by distance, so ordering them by id is a choice rather than
        // a correction, and it is the only one available that does not depend on
        // where a hasher happened to put a key.
        //
        // **It is also what makes the columns and the walk return the same
        // page.** The bitmap arrives in increasing internal id order and the
        // walk arrives in hash order, and external ids are unique, so this key
        // is total and both inputs sort to the same output.
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        scored.truncate(fetch_k);
        Some(scored)
    }

    /// One query's candidates, by whichever of the two paths suits its filter
    ///
    /// An unfiltered search traverses with the liveness predicate it has always
    /// carried, and nothing about it changes. A filtered search tries the exact
    /// scan first and traverses with the filter conjoined into that predicate
    /// where the scan gives up.
    ///
    /// **The combined predicate is the liveness check and `matches_filter`,
    /// conjoined.** `rev_map.get` answering `Some` is the liveness check, since
    /// a node stranded by a removal or an overwrite no longer resolves, and
    /// `FilterView::admits` is the filter. A node either check rejects still
    /// routes the search and never consumes a result slot, which is what the
    /// traversal's predicate has always done.
    ///
    /// Both closures are monomorphised into the traversal by
    /// `VectorGraph::search`'s generic parameter, so the filtered arm and the
    /// unfiltered arm compile to two specialisations and neither pays for an
    /// indirect call. That is why the branch is written out rather than folded
    /// into one closure with an `Option` inside it.
    fn search_candidates<'a>(
        &self,
        hnsw: &VectorGraph,
        query: &[f32],
        rev_map: &'a HashMap<usize, String>,
        filter: Option<Filtered<'a>>,
        fetch_k: usize,
        ef: usize,
    ) -> Candidates<'a> {
        // Asked of the guard this function was handed, not of
        // `HNSWIndex::is_quantized`, which takes the graph read lock itself.
        // Calling that from here took `hnsw` a second time on a thread already
        // holding it, which the lock order forbids for exactly the reason it
        // then demonstrated: the standard library queues readers behind a
        // waiting writer, so the second read blocked forever the moment a
        // concurrent insert landed between the two. It deadlocked the
        // concurrency suite on the first run.
        let operation = if hnsw.is_quantized() {
            "adc_search"
        } else {
            "raw_search"
        };

        let graph_hits = match filter {
            Some(filter) => match filter.columns.select(filter.conditions) {
                Ok(selected) => {
                    let matched = selected.count();
                    if matched <= FULL_SCAN_THRESHOLD {
                        trace!(
                            operation = "filtered_columns",
                            matched = matched,
                            "Filtered search answered from the columns"
                        );
                        return self.scan_selected(query, &selected, rev_map, filter, fetch_k);
                    }
                    // Above the threshold the traversal runs, and the bitmap is
                    // the whole predicate. One bit test replaces the node,
                    // `rev_map`, metadata, field lookup chain the walk had to
                    // make per node, and it subsumes the liveness check because
                    // a slot holding no record holds no bit.
                    trace!(
                        operation = "filtered_columns",
                        matched = matched,
                        "Filtered search traverses with a bitmap predicate"
                    );
                    let admits = |internal_id: &usize| selected.contains(*internal_id);
                    hnsw.search(query, fetch_k, ef, Some(&admits))
                }
                Err(undeclared) => {
                    self.warn_undeclared_filter_field(filter.columns, undeclared);
                    if let Some(scanned) = self.scan_candidates(query, filter, fetch_k) {
                        trace!(
                            operation = "filtered_scan",
                            matched = scanned.len(),
                            "Filtered search answered by the exact scan"
                        );
                        return scanned;
                    }
                    let admits = |internal_id: &usize| match rev_map.get(internal_id) {
                        Some(ext_id) => filter.admits(ext_id),
                        None => false,
                    };
                    hnsw.search(query, fetch_k, ef, Some(&admits))
                }
            },
            None => {
                let live = |internal_id: &usize| rev_map.contains_key(internal_id);
                hnsw.search(query, fetch_k, ef, Some(&live))
            }
        }
        .unwrap_or_else(|e| {
            error!(operation = operation, error = %e, "Graph search failed");
            Vec::new()
        });

        Self::resolve(graph_hits, rev_map)
    }

    /// Say once that a filter named a field this index did not declare.
    ///
    /// Silent on an index that declared nothing, because there the walk is what
    /// the index has always done and a warning on every filtered search would
    /// be noise for every user who never asked for columns. On an index that
    /// declared some fields it is worth saying, because the user has met the
    /// declaration surface and the cost they are paying is invisible otherwise.
    ///
    /// Fires once per index, claimed with a compare and exchange so a batch
    /// fanned across rayon produces one line and not one per worker.
    ///
    /// **Whether anything is declared is asked of the guard the caller was
    /// handed, not of `self.columns`.** Every search path takes the columns read
    /// guard before it reaches here and holds it across the whole search, so
    /// reading the lock again on the same thread is the second acquisition the
    /// declared lock order forbids: the standard library queues readers behind a
    /// waiting writer, so it blocks forever the moment a concurrent insert lands
    /// between the two. That is the deadlock relay 86 found when
    /// `search_candidates` called `HNSWIndex::is_quantized`, and the first
    /// version of this function reintroduced it.
    fn warn_undeclared_filter_field(&self, columns: &ColumnStore, field: &str) {
        if !columns.is_declared() || self.undeclared_filter_warned.load(Ordering::Acquire) {
            return;
        }
        if self
            .undeclared_filter_warned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        warn!(
            operation = "filter_field_not_indexed",
            field = %field,
            "Filter names \"{}\", which this index did not declare in indexed_fields, so \
             the search walks every record's metadata to find the matches. The answer is \
             the same either way; what differs is that the walk costs about 250 \
             nanoseconds a record. Add the field to indexed_fields at create() if you \
             filter on it. This warning fires once.",
            field
        );
    }

    /// Graph hits as candidates, dropping any whose node no longer resolves
    ///
    /// The traversal's predicate has already rejected those, so this drops
    /// nothing in practice. It is kept because `rev_map` is what turns a node
    /// into a record and the resolution has to happen somewhere.
    fn resolve<'a>(hits: Vec<GraphHit>, rev_map: &'a HashMap<usize, String>) -> Candidates<'a> {
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            if let Some(ext_id) = rev_map.get(&hit.internal_id) {
                out.push((ext_id, hit.distance));
            }
        }
        out
    }

    /// Score and cut one query's candidates
    ///
    /// The single query path and the two batch paths each held their own copy
    /// of this. The three copies agreed, and a rule that has to hold across
    /// every search path is one that must not be stated three times: the
    /// reconstruction fallback for `return_vector` was added to the batch
    /// copies in a later relay than the single one, and for a while the three
    /// disagreed about what a `quantized_only` index hands back.
    ///
    /// The guards are taken by the caller and passed in, because each path
    /// holds them for a different span. The single query path and the
    /// sequential batch path hold one set across every query, the parallel path
    /// takes its own per worker, and none of that is this function's business.
    ///
    /// **The filter is not applied here.** It used to be, which is what made a
    /// selective filter return an empty page, and it now runs inside
    /// `search_candidates` before the page is cut. Every candidate arriving
    /// here has already been admitted by whichever path produced it.
    fn collect_hits(
        &self,
        candidates: Candidates<'_>,
        query: &[f32],
        vectors: RawVectors<'_>,
        pq_codes: &HashMap<String, Vec<u8>>,
        vector_metadata: &HashMap<String, HashMap<String, Value>>,
        params: SearchParams,
    ) -> PyResult<QueryHits> {
        let mut scored = candidates;

        if let Some(plan) = params.rerank.as_ref() {
            for entry in scored.iter_mut() {
                entry.1 =
                    rescore_candidate(plan, query, entry.0, vectors, self.pq.as_ref(), pq_codes)
                        .unwrap_or(f32::INFINITY);
            }
            take_best(&mut scored, params.top_k);
        }

        let mut results = Vec::with_capacity(scored.len());
        for (ext_id, score) in scored {
            let metadata = vector_metadata.get(ext_id).cloned().unwrap_or_default();
            // The raw vector where one exists and the reconstruction from the
            // codes where none does. Under `quantized_only` every record is
            // code held once training completes, so without the fallback a
            // search returns no vectors at all.
            let vector_data = if params.return_vector {
                vectors
                    .get(ext_id.as_str())
                    .map(<[f32]>::to_vec)
                    .or_else(|| {
                        let codes = pq_codes.get(ext_id)?;
                        self.pq.as_ref()?.reconstruct(codes).ok()
                    })
            } else {
                None
            };

            results.push((ext_id.clone(), score, metadata, vector_data));
        }

        Ok(results)
    }

    /// One query's hits as the list of dicts Python receives.
    fn hits_to_python(&self, hits: QueryHits, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut output = Vec::with_capacity(hits.len());
        for (id, score, metadata, vector_data) in hits {
            let dict = PyDict::new(py);
            dict.set_item("id", id)?;
            dict.set_item("score", score)?;
            dict.set_item("metadata", value_map_to_python(&metadata, py)?)?;
            if let Some(vec) = vector_data {
                // A NumPy array rather than a list of Python floats.
                //
                // `set_item("vector", vec)` on a `Vec<f32>` builds a `PyList`
                // and one Python float object per component. At `top_k` 10 and
                // dimension 1,536 that is 15,360 allocations a page, and on a
                // batch of 32 it is 491,520. Measured at dimension 1,536 it
                // nearly doubled the per query cost of a batch search.
                //
                // `PyArray1::from_vec` writes the same `f32` values into one
                // buffer, so the values are unchanged and only their container
                // is different. Both adapters ask for the vector on every MMR
                // query and then run pure Python cosine loops over what comes
                // back, which an array lets them do in NumPy instead.
                dict.set_item("vector", PyArray1::from_vec(py, vec))?;
            }
            output.push(dict.into());
        }
        Ok(output)
    }

    /// A batch's hits as one list of dicts per query, in query order.
    fn batch_hits_to_python(
        &self,
        batches: Vec<QueryHits>,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let mut output = Vec::with_capacity(batches.len());
        for hits in batches {
            output.push(self.hits_to_python(hits, py)?);
        }
        Ok(output)
    }

    /// The single query search path
    ///
    /// Lifted out of the `search` entry point so that all four paths, this one,
    /// the sequential batch, the parallel batch and the benchmark, sit together
    /// and take their guards in the one documented order.
    pub(super) fn single_search_internal(
        &self,
        processed_query: &[f32],
        filter_conditions: Option<&Filter>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let search_results = py.detach(|| -> PyResult<QueryHits> {
            // Six read guards, held for the whole search, in the order every
            // path in the crate takes them: `id_map < rev_map < hnsw <
            // pq_codes < vector_metadata < columns`. `id_map` is here because the raw
            // vectors are addressed by node index now and it is what turns an
            // external id into one; it is taken before `rev_map` because a
            // removal holds `id_map` and then takes `rev_map`, and the reverse
            // order here would deadlock against it. The raw vector map that
            // used to sit between `hnsw` and `pq_codes` is gone.
            let id_map = self.id_map.read().unwrap();
            let rev_map = self.rev_map.read().unwrap();
            let hnsw_guard = self.hnsw.read().unwrap();
            let pq_codes = self.pq_codes.read().unwrap();
            let vector_metadata = self.vector_metadata.read().unwrap();
            let columns = self.columns.read().unwrap();
            let vectors = RawVectors {
                id_map: &id_map,
                graph: &hnsw_guard,
            };

            let filter = filter_conditions.map(|conditions| Filtered {
                conditions,
                columns: &columns,
                metadata: &vector_metadata,
                vectors,
                pq_codes: &pq_codes,
            });
            let fetch_k = params.fetch_k(rev_map.len());

            let candidates = self.search_candidates(
                &hnsw_guard,
                processed_query,
                &rev_map,
                filter,
                fetch_k,
                params.ef,
            );

            self.collect_hits(
                candidates,
                processed_query,
                vectors,
                &pq_codes,
                &vector_metadata,
                params,
            )
        })?;

        self.hits_to_python(search_results, py)
    }

    /// Internal batch search method for multiple query vectors
    #[instrument(level = "debug", skip(self, vectors, filter_conditions, params, py), fields(
        batch_size = vectors.len(),
        top_k = params.top_k,
        ef = params.ef,
        return_vector = params.return_vector,
        has_filter = filter_conditions.is_some(),
        rerank_factor = params.rerank.and_then(|plan| plan.factor)
    ), err)]
    pub(super) fn batch_search_internal(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&Filter>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let start_time = Instant::now();

        // Validate all vectors have correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dim {
                error!(
                    operation = "batch_search_validation",
                    vector_index = i,
                    expected_dim = self.dim,
                    actual_dim = vector.len(),
                    "Vector dimension mismatch in batch"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Vector {}: dimension mismatch: expected {}, got {}",
                    i,
                    self.dim,
                    vector.len()
                )));
            }

            // The same value check the single query path applies. A non-finite
            // component survives normalization, because the norm of a vector
            // containing one is not greater than zero, and the search then
            // returns hits whose scores carry no distance information. The
            // message names the batch entry as well as the component, so one
            // bad vector is findable in a batch of thousands.
            for (component, &value) in vector.iter().enumerate() {
                if !value.is_finite() {
                    error!(
                        operation = "batch_search_validation",
                        vector_index = i,
                        value_index = component,
                        value = value,
                        "Vector in batch contains invalid value"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Vector {} in batch contains invalid value at index {}: {} (must be finite)",
                        i, component, value
                    )));
                }
            }
        }

        // Choose strategy based on batch size
        let result = if vectors.len() <= 5 {
            trace!(
                operation = "batch_search_strategy",
                strategy = "sequential",
                "Using sequential processing"
            );
            self.batch_search_sequential(vectors, filter_conditions, params, py)
        } else {
            trace!(
                operation = "batch_search_strategy",
                strategy = "parallel",
                "Using parallel processing"
            );
            self.batch_search_parallel(vectors, filter_conditions, params, py)
        };

        // ✅ ENTERPRISE: Add duration timing to hot path
        let duration_ms = start_time.elapsed().as_millis();
        debug!(
            operation = "batch_search_complete",
            batch_size = vectors.len(),
            duration_ms = duration_ms,
            "Batch search completed"
        );

        result
    }

    /// Sequential batch processing (for small batches)
    fn batch_search_sequential(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&Filter>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let rust_results = py.detach(|| -> PyResult<Vec<QueryHits>> {
            // Six read guards, in the one documented order, held across every
            // query in the batch. See `single_search_internal` for why `id_map`
            // is first and where the raw vector map went.
            let id_map = self.id_map.read().unwrap();
            let rev_map = self.rev_map.read().unwrap();
            let hnsw_guard = self.hnsw.read().unwrap();
            let code_store = self.pq_codes.read().unwrap();
            let metadata_store = self.vector_metadata.read().unwrap();
            let column_store = self.columns.read().unwrap();
            let vector_store = RawVectors {
                id_map: &id_map,
                graph: &hnsw_guard,
            };

            let filter = filter_conditions.map(|conditions| Filtered {
                conditions,
                columns: &column_store,
                metadata: &metadata_store,
                vectors: vector_store,
                pq_codes: &code_store,
            });

            // The same over-fetch the single query path applies, so a batch of
            // one query returns what that query returns on its own.
            let fetch_k = params.fetch_k(rev_map.len());

            let mut all_results = Vec::with_capacity(vectors.len());

            for vector in vectors {
                // FIX: Process each query vector for space
                let processed_query = self.process_vector_for_space(vector.clone());

                let candidates = self.search_candidates(
                    &hnsw_guard,
                    &processed_query,
                    &rev_map,
                    filter,
                    fetch_k,
                    params.ef,
                );

                all_results.push(self.collect_hits(
                    candidates,
                    &processed_query,
                    vector_store,
                    &code_store,
                    &metadata_store,
                    params,
                )?);
            }

            Ok(all_results)
        })?;

        self.batch_hits_to_python(rust_results, py)
    }

    /// Parallel batch processing (for larger batches)
    fn batch_search_parallel(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&Filter>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let span = tracing::Span::current();
        let rust_results = py.detach(|| -> PyResult<Vec<QueryHits>> {
            let results: PyResult<Vec<QueryHits>> = vectors
                .par_iter()
                .map(|vector| -> PyResult<QueryHits> {
                    let _entered = span.clone().entered();
                    // FIX: Process each query vector for space
                    let processed_query = self.process_vector_for_space(vector.clone());

                    // Six read guards per worker, in the one documented order
                    // and held across the traversal. Every guard is a read, so
                    // the rule that no path forks to rayon holding a write
                    // guard is untouched. See `single_search_internal` for why
                    // `id_map` is first and where the raw vector map went.
                    let id_map = self.id_map.read().unwrap();
                    let rev_map = self.rev_map.read().unwrap();
                    let hnsw_guard = self.hnsw.read().unwrap();
                    let code_store = self.pq_codes.read().unwrap();
                    let metadata_store = self.vector_metadata.read().unwrap();
                    let column_store = self.columns.read().unwrap();
                    let vector_store = RawVectors {
                        id_map: &id_map,
                        graph: &hnsw_guard,
                    };

                    let filter = filter_conditions.map(|conditions| Filtered {
                        conditions,
                        columns: &column_store,
                        metadata: &metadata_store,
                        vectors: vector_store,
                        pq_codes: &code_store,
                    });

                    // The same over-fetch the other two search paths apply.
                    let fetch_k = params.fetch_k(rev_map.len());

                    let candidates = self.search_candidates(
                        &hnsw_guard,
                        &processed_query,
                        &rev_map,
                        filter,
                        fetch_k,
                        params.ef,
                    );

                    self.collect_hits(
                        candidates,
                        &processed_query,
                        vector_store,
                        &code_store,
                        &metadata_store,
                        params,
                    )
                })
                .collect();

            results
        })?;

        self.batch_hits_to_python(rust_results, py)
    }

    /// Raw search without Python objects (for benchmarking)
    ///
    /// Two read guards in declared order, `rev_map` then `hnsw`, and the graph
    /// guard is now held across the resolution as well as the traversal because
    /// it is `search_candidates` that owns both. This path takes no filter, so
    /// it takes no storage guards and its predicate is the liveness check
    /// alone.
    pub(super) fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        let rev_map = self.rev_map.read().unwrap();
        let hnsw_guard = self.hnsw.read().unwrap();

        self.search_candidates(&hnsw_guard, query, &rev_map, None, 10, 100)
            .into_iter()
            .map(|(ext_id, distance)| (ext_id.clone(), distance))
            .collect()
    }
}
