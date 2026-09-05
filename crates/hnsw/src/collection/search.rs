//! Running a search, on one query or on a batch.
//!
//! Four paths reach the dense index: one query, a batch of five or fewer run
//! in turn, a larger batch fanned across rayon, and the benchmark. They
//! differ only in how long they hold the storage guards, so what each does
//! with a candidate is in `collect_hits`. What each hands back is a
//! [`QueryHits`] page of owned Rust, which the binding turns into the list
//! of dicts Python receives. The two sparse searches at the end of this file
//! are one arm queries, and a query over several arms is `query.rs`, which
//! builds its admit set and its dense page with the same pieces these four
//! paths use.
//!
//! # What the collection decides and what the index decides
//!
//! The collection owns the metadata and the columns, so it decides which
//! records a search may return, and it hands the index that decision as an
//! admit set. A filter over declared fields is a bitmap. A filter mixing a
//! declared field with an undeclared one is the bound the declared fields
//! leave, walked here with the whole filter judged on each candidate. A
//! filter over undeclared fields alone is a walk of every record's metadata.
//! Where a walk finishes under [`FULL_SCAN_THRESHOLD`] its matches are the
//! admit set, and where it gives up the admit set is the bound conjoined
//! with the metadata predicate, or the predicate alone.
//!
//! The index decides the path. An admit set it can enumerate whose count is
//! at or under the threshold is scored exactly, and everything else
//! traverses the graph under the set as a predicate. The threshold is stated
//! once, in the index, and the walks here stop at the same figure so that a
//! walk that gives up hands the index a set it will traverse.
//!
//! # The page says which path produced it
//!
//! [`Scored`] carries `exact`, set by the index for a scored page and not for
//! a traversal. The two order a tie differently. The scan takes its
//! candidates in the metadata store's hash order or in bitmap order, so two
//! records at exactly equal distance have to be ordered by something that is
//! not where a hasher put a key, and that is the external id string. The
//! traversal's order among equal distances is the heap's, which is a
//! function of the graph, and it is returned as it comes: a traversal of a
//! fixed graph is deterministic, and 1,208 of 64,800 recorded pages hold a
//! tie the string order would reverse. So the tie break applies to exact
//! pages and to nothing else, in one place, `Scored::cut`. The index keeps
//! the whole tie group at an exact page's boundary so that rule can decide
//! the cut.
//!
//! # A filter decides the page rather than trimming it
//!
//! The filter used to run in `collect_hits`, which is after the graph
//! has already cut to `top_k`. A filter matching one record in a hundred
//! therefore discarded most of a ten result page and returned what was left,
//! and one matching one record in a thousand returned nothing at all. That
//! measured as post-filter recall of 0.0090 at one match in a hundred and 0.0000
//! below it, on all three real sets.
//!
//! The exact scan under a selective filter is what replaced it, and it gives
//! up once the match count passes [`FULL_SCAN_THRESHOLD`], so a broad filter
//! pays a bounded walk rather than a full one. Where the scan gives up, the
//! traversal runs with the filter conjoined into the predicate it already
//! carried for liveness. That is the correct answer for a broad filter and a
//! latency pathology for a selective one, which is why the scan is not
//! optional. A filter matching one record in 100,000 costs 128 to 359
//! milliseconds through the traversal and 26 through the scan, both measured
//! on this code.
//!
//! **A filter over declared fields does not walk anything.** The fields named
//! at `create(indexed_fields=[...])` each carry a column addressed by internal
//! id, so a filter over them compiles to a bitmap and both paths read it: the
//! exact scan takes the set bits and the traversal tests one bit per node.
//!
//! **The walk is what made a filtered search expensive.** It costs about 250
//! nanoseconds a record, so a selective filter over 100,000 records cost about
//! 25 milliseconds where an unfiltered search costs 0.3 to 1.2. See
//! [`FULL_SCAN_THRESHOLD`] for the measurements and the column store in
//! `zeusdb_vector_core` for what replaces it.
//!
//! **Lock order.** Every path takes `rev_map` before the index and the
//! storage maps after it, which is the order declared on `Collection` in the
//! parent module. A search holds `rev_map` for its whole traversal, so a
//! mutation taking a storage map before `rev_map` deadlocks against it, and
//! that is the inversion `remove_point_internal` used to carry.

use super::{Arm, Collection, LiveRecords, Query, StorageMode};
use crate::{
    raw_distance_fn, reconstruction_needs_unit, rescore_candidate, take_best, RawVectors,
    RerankPlan, SearchParams,
};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, error, instrument, trace, warn};
use zeusdb_vector_core::{
    matches_filter, Admit, And, Bitmap, Budget, Candidates, ColumnStore, Error, FieldLookup,
    Filter, Fusion, Hits, IdfScope, MetadataStore, RecordId, Selection, SparseRef, VectorIndex,
};

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::search";

/// Largest `top_k` a search may ask for.
///
/// `top_k` sizes the candidate search through the default `ef_search` of
/// twice `top_k`, and `search_layer` sizes its two candidate heaps from that
/// width, 8 bytes a slot, before it visits a node. The allocation is not
/// fallible, so `search(top_k=2**40)` asked for 17,592,186,044,416 bytes and
/// **aborted the process** with exit status 3221226505, and `top_k=2**33`
/// died the same way asking for 137 GB. Nothing checked either argument.
///
/// The ceiling is far above any page a caller has reason to ask for, so no
/// real caller is refused. At the ceiling the heaps are 2 MiB and the result
/// list is 65,536 Python dicts, which is slow and is what was asked for.
pub(super) const MAX_TOP_K: usize = 65_536;

/// Largest `ef_search` a search may pass, being the default `ef_search` at
/// the largest `top_k`.
///
/// The same two heaps as `MAX_TOP_K`, reached directly. `search(ef_search=2**40)`
/// asked for 8,796,093,022,208 bytes and aborted. At the ceiling the heaps are
/// 2 MiB, and a search at that width on a corpus smaller than it is an
/// exhaustive scan, which is the slowest thing a search can be and is bounded
/// by the corpus.
pub(super) const MAX_EF_SEARCH: usize = 2 * MAX_TOP_K;
/// Records a filtered search may match before it stops scanning and traverses
/// the graph instead.
///
/// # What the two paths actually cost
///
/// The crate was built twice, once with this at zero so every filtered
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
/// distances, is what a filtered search pays for.**
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
/// and the traversal is not, so the crossover moves with corpus size and where
/// it moves to is not measured here. The threshold's value was first derived
/// from the crossover between a scan and an **unfiltered** search, which is not
/// the comparison that matters. It survives on the evidence above rather than on
/// that derivation.
///
/// It is not settable per search. Nothing measured here wants a different
/// number, and a knob nobody has a reason to turn is a knob that goes wrong.
///
/// The dense index reads it to decide between the exact scan and the
/// traversal, and the two metadata walks here stop at it, so a walk that
/// gives up hands the index a set it will traverse.
pub(super) const FULL_SCAN_THRESHOLD: usize = 5_000;

/// One search's candidates, as borrowed external ids paired with the score
/// whichever path found them scored them at, and whether that path was exact.
///
/// The index returns a page of internal ids and this is that page resolved
/// through `rev_map`, borrowing the id it finds there. Nothing is cloned until
/// the page is cut, so an over-fetched page pays to clone the metadata and
/// the vector of the results it returns rather than of every candidate it
/// considered.
///
/// `exact` is true when every admitted record was scored, which is the scan
/// and never the traversal. It decides the tie break; see [`Scored::cut`] and
/// the module documentation.
pub(super) struct Scored<'a> {
    pub(super) items: Vec<(&'a String, f32)>,
    pub(super) exact: bool,
}

impl<'a> Scored<'a> {
    /// A page every admitted record was scored for.
    #[cfg(test)]
    fn exact(items: Vec<(&'a String, f32)>) -> Self {
        Scored { items, exact: true }
    }

    /// A page in the traversal's own order.
    #[cfg(test)]
    fn traversed(items: Vec<(&'a String, f32)>) -> Self {
        Scored {
            items,
            exact: false,
        }
    }

    /// The index's page with every id resolved to the external id it names.
    ///
    /// An id that no longer resolves is dropped, which is the liveness rule
    /// every path applies. The index searched under the live set, so this
    /// drops nothing in practice, and `rev_map` is what turns a node into a
    /// record so the resolution has to happen somewhere.
    pub(super) fn resolve(hits: Hits, rev_map: &'a LiveRecords) -> Self {
        let mut items = Vec::with_capacity(hits.items.len());
        for hit in hits.items {
            if let Some(ext_id) = rev_map.get(&hit.id.slot()) {
                items.push((ext_id, hit.score));
            }
        }
        Scored {
            items,
            exact: hits.exact,
        }
    }

    /// Cut the page to `fetch_k`, ordering an exact page and leaving a
    /// traversal's alone.
    ///
    /// An exact page is ordered by distance, and by external id where two
    /// distances are equal. The tie break is what makes it reproducible. The
    /// walk takes the metadata store in `HashMap` iteration order, which
    /// `RandomState` seeds afresh in every process, so a stable sort on
    /// distance alone would hand back two equally distant records in an order
    /// that differed run to run. Two exactly equal distances are unordered by
    /// distance, so ordering them by id is a choice rather than a correction,
    /// and it is the only one available that does not depend on where a
    /// hasher happened to put a key. It is also what makes the columns and
    /// the walk return the same page: the bitmap arrives in increasing
    /// internal id order and the walk arrives in hash order, and external ids
    /// are unique, so this key is total and both inputs sort to the same
    /// output.
    ///
    /// A traversal's page is returned as the traversal produced it, already
    /// at most `fetch_k` long. Its order among equal distances is the heap's,
    /// which is a function of the graph and is deterministic, and re-sorting
    /// it by string would move a page that has never moved.
    pub(super) fn cut(mut self, fetch_k: usize) -> Self {
        if self.exact {
            self.items
                .sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(b.0)));
            self.items.truncate(fetch_k);
        }
        self
    }
}

/// The metadata predicate, as an admit set.
///
/// What a filter that cannot be answered from the columns alone reads. It
/// reads the node's entry in the store by internal id, which is the liveness
/// check since a removed record holds none, and judges the filter on it. A
/// record with no metadata entry does not match, which is the rule
/// `collect_hits` applied when the filter ran there. It cannot count itself
/// and cannot enumerate itself, so the index asks it one node at a time.
pub(super) struct MetadataAdmit<'a> {
    conditions: &'a Filter,
    metadata: &'a MetadataStore,
}

impl MetadataAdmit<'_> {
    /// Whether one record's metadata matches, with no lookup.
    ///
    /// **There is no error channel and none is needed.** What reaches here is a
    /// compiled [`Filter`], which `compile_filter` built from the caller's
    /// mapping once per search and before any record was examined, rejecting
    /// every operator name and every group shape the engine cannot evaluate.
    /// Nothing below `matches_filter` can fail, so it returns `bool`.
    #[inline]
    fn judge<M: FieldLookup + ?Sized>(&self, meta: &M) -> bool {
        matches_filter(meta, self.conditions)
    }
}

impl Admit for MetadataAdmit<'_> {
    #[inline]
    fn admits(&self, id: RecordId) -> bool {
        match self.metadata.get(id.slot()) {
            Some(fields) => self.judge(&fields),
            None => false,
        }
    }

    fn len_hint(&self) -> Option<usize> {
        None
    }
}

/// What the collection decided a search may return, before the index is
/// asked.
///
/// Four shapes. Everything, for a search with no filter, which the index
/// runs under its own live set. A bitmap, for a filter every field of which
/// is declared. The matches of a walk that finished, in increasing internal
/// id order. Or the walk's give-up, being the bound the declared fields left
/// conjoined with the metadata predicate, or the predicate alone where they
/// left none, which the index traverses under.
pub(super) enum AdmitPlan<'a> {
    All,
    Bitmap(Bitmap),
    Matched(Vec<RecordId>),
    Bounded(Bitmap, MetadataAdmit<'a>),
    Predicate(MetadataAdmit<'a>),
}

impl AdmitPlan<'_> {
    /// Run `search` under the set, building the conjunction where there is
    /// one.
    pub(super) fn run<R>(&self, search: impl FnOnce(&dyn Admit) -> R) -> R {
        match self {
            AdmitPlan::All => search(&Candidates::All),
            AdmitPlan::Bitmap(bitmap) => search(bitmap),
            AdmitPlan::Matched(ids) => search(&Candidates::Sorted(ids.clone())),
            AdmitPlan::Bounded(bound, predicate) => search(&And(bound, predicate)),
            AdmitPlan::Predicate(predicate) => search(predicate),
        }
    }
}

/// Search hits for one query vector, as (external id, distance, metadata,
/// optional raw vector). The raw vector is present only when the caller asked
/// for it and the index still holds one, or could reconstruct one from its
/// codes. Owned Rust, which the binding converts to a list of dicts.
pub type QueryHits = Vec<(String, f32, HashMap<String, Value>, Option<Vec<f32>>)>;

/// Search hits for one sparse query, as (external id, score), best first.
pub type SparseHits = Vec<(String, f32)>;

impl Collection {
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
            .dense()
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        if !keeps_raw {
            return None;
        }

        Some(RerankPlan {
            factor: rerank.map(|factor| factor.max(1)),
            calibration: self.dense().rerank_calibration(),
            distance: raw_distance_fn(&self.dense().metric),
            unit_reconstruction: reconstruction_needs_unit(&self.dense().metric),
        })
    }

    /// What one search may return, decided from its filter.
    ///
    /// An unfiltered search admits everything and the index runs it under
    /// its own live set. A filtered one is decided by the shape the columns
    /// leave.
    ///
    /// **A filter every field of which is declared is a bitmap**, and the
    /// index scans it where it is small and traverses under it where it is
    /// not, which is the threshold rule stated once in the index.
    ///
    /// **A filter mixing a declared field with one that has no column** walks
    /// the bound the declared fields leave, judging the whole filter on each
    /// candidate's metadata. The bound is a superset of the matching set, so
    /// what the columns bought is how few candidates there are. The walk
    /// gives up at [`FULL_SCAN_THRESHOLD`] matches and the index then
    /// traverses under the bound conjoined with the predicate, which admits
    /// exactly what the predicate alone admits since a record the filter
    /// matches is inside the bound by construction. The bit test comes first
    /// in that conjunction, so a node outside the bound costs one word read.
    ///
    /// **A filter naming no declared field** walks every record's metadata,
    /// which is what every filtered search did before the columns existed,
    /// and gives up at the same figure, after which the index traverses under
    /// the predicate.
    ///
    /// **The distances are computed after the walk and not during it.** Every
    /// distance a fused walk evaluated before giving up would be thrown away,
    /// and at 1,536 dimensions that is thousands of six kilobyte reads charged
    /// to every broad filtered search. The walk collects internal ids alone,
    /// the index scores them, and the give-up path is free of distance work
    /// entirely.
    ///
    /// **Fewer than `top_k` matches is a result and not a failure.** A filter
    /// matching three records returns three results to a caller who asked for
    /// ten, and a filter matching none returns an empty page.
    pub(super) fn admit_plan<'a>(
        &self,
        conditions: Option<&'a Filter>,
        columns: &ColumnStore,
        rev_map: &'a LiveRecords,
        metadata: &'a MetadataStore,
    ) -> AdmitPlan<'a> {
        let Some(conditions) = conditions else {
            return AdmitPlan::All;
        };
        let predicate = MetadataAdmit {
            conditions,
            metadata,
        };
        match columns.select(conditions) {
            Selection::Exact(selected) => {
                let matched = selected.count();
                if matched <= FULL_SCAN_THRESHOLD {
                    trace!(
                        target: LOG_TARGET,
                        operation = "filtered_columns",
                        matched = matched,
                        "Filtered search answered from the columns"
                    );
                    AdmitPlan::Bitmap(selected)
                } else if rev_map.admits_every_live(&selected) {
                    // Every live record, so the filter is no filter and the
                    // index runs under its own live set, which is the
                    // unfiltered search. Handed on as a bitmap instead, a
                    // term weighted sparse scan would walk the query's
                    // lists to count a corpus it can read from its own
                    // totals, a fifth of such a query, a dot product scan
                    // would test a bit per posting, an eighth, and a dense
                    // traversal would test a bit per node, which measures
                    // as nothing. Only above the threshold, because below
                    // it the exact scan over the bitmap is the cheaper and
                    // the better page, and it would become a traversal.
                    // One word walk of the intersection against the live
                    // count decides it.
                    trace!(
                        target: LOG_TARGET,
                        operation = "filtered_columns",
                        matched = matched,
                        "Filtered search admits every live record and runs unfiltered"
                    );
                    AdmitPlan::All
                } else {
                    // Above the threshold the traversal runs, and the bitmap
                    // is the whole predicate. One bit test replaces the
                    // entry, field lookup chain the walk had to make per
                    // node, and it subsumes the liveness check because a slot
                    // holding no record holds no bit.
                    trace!(
                        target: LOG_TARGET,
                        operation = "filtered_columns",
                        matched = matched,
                        "Filtered search traverses with a bitmap predicate"
                    );
                    AdmitPlan::Bitmap(selected)
                }
            }
            Selection::Narrowed(bound, undeclared) => {
                let candidates = bound.count();
                self.warn_undeclared_filter_field(columns, undeclared, Some(candidates));
                let mut matched: Vec<RecordId> = Vec::new();
                let mut gave_up = false;
                bound.for_each_while(|slot| {
                    let Some(fields) = metadata.get(slot) else {
                        return true;
                    };
                    if predicate.judge(&fields) {
                        if matched.len() == FULL_SCAN_THRESHOLD {
                            // One past the threshold, so the walk stops here
                            // and the index traverses. Nothing has been
                            // scored yet.
                            gave_up = true;
                            return false;
                        }
                        matched.push(RecordId::from_slot(slot));
                    }
                    true
                });
                if gave_up {
                    trace!(
                        target: LOG_TARGET,
                        operation = "filtered_bounded_traversal",
                        candidates = candidates,
                        "Filtered search traverses inside a bitmap bound"
                    );
                    AdmitPlan::Bounded(bound, predicate)
                } else {
                    trace!(
                        target: LOG_TARGET,
                        operation = "filtered_bounded_scan",
                        candidates = candidates,
                        matched = matched.len(),
                        "Filtered search answered by a bounded scan"
                    );
                    AdmitPlan::Matched(matched)
                }
            }
            Selection::Whole(undeclared) => {
                self.warn_undeclared_filter_field(columns, undeclared, None);
                let mut matched: Vec<RecordId> = Vec::new();
                for (slot, fields) in metadata.iter() {
                    if predicate.judge(&fields) {
                        if matched.len() == FULL_SCAN_THRESHOLD {
                            // One past the threshold, so the walk stops here
                            // and the index traverses. Nothing has been
                            // scored yet.
                            return AdmitPlan::Predicate(predicate);
                        }
                        matched.push(RecordId::from_slot(slot));
                    }
                }
                // The walk arrives in increasing internal id order, which is
                // the sorted set the index expects. It used to arrive in a
                // hash map's order and be sorted here, and the page is the
                // same either way, since the index sorts an exact page by
                // distance and the collection breaks ties by external id.
                debug_assert!(
                    matched.windows(2).all(|pair| pair[0] < pair[1]),
                    "the store walks in increasing internal id order"
                );
                trace!(
                    target: LOG_TARGET,
                    operation = "filtered_scan",
                    matched = matched.len(),
                    "Filtered search answered by the exact scan"
                );
                AdmitPlan::Matched(matched)
            }
        }
    }

    /// Say once that a filter named a field this index did not declare.
    ///
    /// `narrowed_to` is the number of records the declared fields bounded the
    /// search to, and `None` where they bounded nothing. The two cases cost
    /// different amounts and the message says which one happened, because a
    /// filter reading a hundred metadata entries and one reading a hundred
    /// thousand are not the same report.
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
    /// between the two.
    fn warn_undeclared_filter_field(
        &self,
        columns: &ColumnStore,
        field: &str,
        narrowed_to: Option<usize>,
    ) {
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
        match narrowed_to {
            Some(candidates) => warn!(target: LOG_TARGET, operation = "filter_field_not_indexed",
                field = %field,
                candidates = candidates,
                "Filter names \"{}\", which this index did not declare in indexed_fields. \
                 The fields it did declare narrowed the search to {} records, and it read \
                 the metadata of those to finish the answer. A filter naming only declared \
                 fields reads none. Add the field to indexed_fields at create() if you \
                 filter on it. This warning fires once.",
                field,
                candidates
            ),
            None => warn!(target: LOG_TARGET, operation = "filter_field_not_indexed",
                field = %field,
                "Filter names \"{}\", which this index did not declare in indexed_fields, so \
                 the search walks every record's metadata to find the matches. The answer is \
                 the same either way; what differs is that the walk costs about 250 \
                 nanoseconds a record. Add the field to indexed_fields at create() if you \
                 filter on it. This warning fires once.",
                field
            ),
        }
    }

    /// Score and cut one query's candidates
    ///
    /// The single query path and the two batch paths each held their own copy
    /// of this. The three copies agreed, and a rule that has to hold across
    /// every search path is one that must not be stated three times.
    ///
    /// The guards are taken by the caller and passed in, because each path
    /// holds them for a different span. The single query path and the
    /// sequential batch path hold one set across every query, the parallel path
    /// takes its own per worker, and none of that is this function's business.
    ///
    /// **The filter is not applied here.** It used to be, which is what made a
    /// selective filter return an empty page, and it now decides the admit set
    /// the index searched under. Every candidate arriving here has already
    /// been admitted by whichever path produced it.
    fn collect_hits(
        &self,
        candidates: Scored<'_>,
        query: &[f32],
        vectors: RawVectors<'_>,
        pq_codes: &HashMap<String, Vec<u8>>,
        vector_metadata: &MetadataStore,
        params: SearchParams,
    ) -> Result<QueryHits, Error> {
        let mut scored = candidates.items;
        self.rescore_page(&mut scored, query, vectors, pq_codes, &params);

        let mut results = Vec::with_capacity(scored.len());
        for (ext_id, score) in scored {
            let metadata = vectors
                .id_map
                .get(ext_id)
                .and_then(|&slot| vector_metadata.get(slot))
                .map(|fields| fields.to_map())
                .unwrap_or_default();
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
                        self.dense().pq.as_ref()?.reconstruct(codes).ok()
                    })
                    .or_else(|| {
                        let slot = *vectors.id_map.get(ext_id)?;
                        vectors.graph.int8_reconstruct(slot)
                    })
            } else {
                None
            };

            results.push((ext_id.clone(), score, metadata, vector_data));
        }

        Ok(results)
    }

    /// Rescore a page against the raw vectors where the plan reranks, and
    /// cut it to the page size.
    ///
    /// Nothing happens without a rerank plan. With one, every candidate the
    /// graph over-fetched is rescored against its raw vector, or against the
    /// reconstruction of its codes where the index keeps no raw vector, a
    /// candidate that has neither sorts last, and the best `top_k` are kept.
    /// The single space search and the query path both apply it, so a dense
    /// arm inside a query returns the page the single space search returns.
    pub(super) fn rescore_page(
        &self,
        scored: &mut Vec<(&String, f32)>,
        query: &[f32],
        vectors: RawVectors<'_>,
        pq_codes: &HashMap<String, Vec<u8>>,
        params: &SearchParams,
    ) {
        if let Some(plan) = params.rerank.as_ref() {
            for entry in scored.iter_mut() {
                entry.1 = rescore_candidate(
                    plan,
                    query,
                    entry.0,
                    vectors,
                    self.dense().pq.as_ref(),
                    pq_codes,
                )
                .unwrap_or(f32::INFINITY);
            }
            take_best(scored, params.top_k);
        }
    }

    /// Resolve the search arguments into the parameters every path reads.
    ///
    /// Both bounds size the candidate heaps the traversal allocates before it
    /// visits a node, and neither allocation is fallible; see `MAX_TOP_K`.
    /// Checked first, so a bad argument is a ValueError and not a dead
    /// interpreter, and before `ef` is derived so the derivation cannot
    /// overflow on a value the check would have refused.
    ///
    /// The rerank plan is resolved once here rather than per query, because it
    /// locks the graph to read whether the index is quantized and the search
    /// paths take that lock themselves.
    pub fn search_params(
        &self,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
        rerank: Option<usize>,
    ) -> Result<SearchParams, Error> {
        if top_k > MAX_TOP_K {
            return Err(Error::TopKTooLarge {
                max: MAX_TOP_K,
                top_k,
            });
        }
        if let Some(requested) = ef_search {
            if requested > MAX_EF_SEARCH {
                return Err(Error::EfSearchTooLarge {
                    max: MAX_EF_SEARCH,
                    ef_search: requested,
                });
            }
        }

        let ef = ef_search.unwrap_or_else(|| match self.dense().metric.to_lowercase().as_str() {
            "l1" | "l2" => std::cmp::max(2 * top_k, 150),
            _ => std::cmp::max(2 * top_k, 100),
        });

        let params = SearchParams {
            top_k,
            ef,
            return_vector,
            rerank: self.rerank_plan(rerank),
        };

        // The record the entry point wrote, under the entry point's target.
        trace!(
            target: super::LOG_TARGET,
            operation = "search_config",
            ef = ef,
            space = %self.dense().metric,
            rerank_factor = params.rerank.and_then(|plan| plan.factor),
            "Search parameters configured"
        );

        Ok(params)
    }

    /// The budget every dense search hands the index.
    ///
    /// The traversal width the parameters resolved, and the boundary tie
    /// group, because the collection orders an exact page's ties by external
    /// id and needs the whole group at the cut to do so.
    pub(super) fn dense_budget(params: &SearchParams) -> Budget {
        Budget {
            ef: Some(params.ef),
            boundary_ties: true,
            ..Budget::default()
        }
    }

    /// The single query search path
    ///
    /// The query arrives validated and processed for the space; see
    /// `validate_query`. All four paths, this one, the sequential batch, the
    /// parallel batch and the benchmark, sit together and take their guards in
    /// the one documented order.
    pub fn search_one(
        &self,
        processed_query: &[f32],
        filter_conditions: Option<&Filter>,
        params: SearchParams,
    ) -> Result<QueryHits, Error> {
        // Six read guards, held for the whole search, in the order every
        // path in the crate takes them: `id_map < rev_map < index <
        // pq_codes < vector_metadata < columns`. `id_map` is here because the raw
        // vectors are addressed by node index now and it is what turns an
        // external id into one; it is taken before `rev_map` because a
        // removal holds `id_map` and then takes `rev_map`, and the reverse
        // order here would deadlock against it. The raw vector map that
        // used to sit between the index and `pq_codes` is gone.
        let id_map = self.id_map.read().unwrap();
        let rev_map = self.rev_map.read().unwrap();
        let index = self.dense().index.read().unwrap();
        let pq_codes = self.dense().pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        let columns = self.columns.read().unwrap();
        let vectors = RawVectors {
            id_map: &id_map,
            graph: index.graph(),
        };

        let plan = self.admit_plan(filter_conditions, &columns, &rev_map, &vector_metadata);
        let fetch_k = params.fetch_k(rev_map.len());
        let budget = Self::dense_budget(&params);
        let hits = plan.run(|admit| index.search(processed_query, fetch_k, admit, &budget))?;
        let candidates = Scored::resolve(hits, &rev_map).cut(fetch_k);

        self.collect_hits(
            candidates,
            processed_query,
            vectors,
            &pq_codes,
            &vector_metadata,
            params,
        )
    }

    /// The batch search path, for query vectors that are not yet processed
    /// for the space.
    ///
    /// Every vector is checked for its width and for non-finite values before
    /// any is searched, so one bad vector is findable in a batch of thousands
    /// and refuses the whole call. A batch of five or fewer runs in turn under
    /// one set of guards and a larger one fans across rayon.
    #[instrument(target = LOG_TARGET, level = "debug", skip(self, vectors, filter_conditions, params), fields(
        batch_size = vectors.len(),
        top_k = params.top_k,
        ef = params.ef,
        return_vector = params.return_vector,
        has_filter = filter_conditions.is_some(),
        rerank_factor = params.rerank.and_then(|plan| plan.factor)
    ), err)]
    pub fn search_batch(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&Filter>,
        params: SearchParams,
    ) -> Result<Vec<QueryHits>, Error> {
        let start_time = Instant::now();

        // Validate all vectors have correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dense().dim {
                error!(
                    target: LOG_TARGET,
                    operation = "batch_search_validation",
                    vector_index = i,
                    expected_dim = self.dense().dim,
                    actual_dim = vector.len(),
                    "Vector dimension mismatch in batch"
                );
                return Err(Error::BatchVectorDimension {
                    position: i,
                    expected: self.dense().dim,
                    got: vector.len(),
                });
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
                        target: LOG_TARGET,
                        operation = "batch_search_validation",
                        vector_index = i,
                        value_index = component,
                        value = value,
                        "Vector in batch contains invalid value"
                    );
                    return Err(Error::BatchVectorNotFinite {
                        position: i,
                        index: component,
                        value,
                    });
                }
            }
        }

        // Choose strategy based on batch size
        let result = if vectors.len() <= 5 {
            trace!(
                target: LOG_TARGET,
                operation = "batch_search_strategy",
                strategy = "sequential",
                "Using sequential processing"
            );
            self.batch_search_sequential(vectors, filter_conditions, params)
        } else {
            trace!(
                target: LOG_TARGET,
                operation = "batch_search_strategy",
                strategy = "parallel",
                "Using parallel processing"
            );
            self.batch_search_parallel(vectors, filter_conditions, params)
        };

        // ✅ ENTERPRISE: Add duration timing to hot path
        let duration_ms = start_time.elapsed().as_millis();
        debug!(
            target: LOG_TARGET,
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
    ) -> Result<Vec<QueryHits>, Error> {
        // Six read guards, in the one documented order, held across every
        // query in the batch. See `search_one` for why `id_map` is first and
        // where the raw vector map went.
        let id_map = self.id_map.read().unwrap();
        let rev_map = self.rev_map.read().unwrap();
        let index = self.dense().index.read().unwrap();
        let code_store = self.dense().pq_codes.read().unwrap();
        let metadata_store = self.vector_metadata.read().unwrap();
        let column_store = self.columns.read().unwrap();
        let vector_store = RawVectors {
            id_map: &id_map,
            graph: index.graph(),
        };

        // The admit set is the filter's, so it is decided once for the batch.
        let plan = self.admit_plan(filter_conditions, &column_store, &rev_map, &metadata_store);

        // The same over-fetch the single query path applies, so a batch of
        // one query returns what that query returns on its own.
        let fetch_k = params.fetch_k(rev_map.len());
        let budget = Self::dense_budget(&params);

        let mut all_results = Vec::with_capacity(vectors.len());

        for vector in vectors {
            // FIX: Process each query vector for space
            let processed_query = self.dense().process_vector_for_space(vector.clone());

            let hits = plan.run(|admit| index.search(&processed_query, fetch_k, admit, &budget))?;
            let candidates = Scored::resolve(hits, &rev_map).cut(fetch_k);

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
    }

    /// Parallel batch processing (for larger batches)
    fn batch_search_parallel(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&Filter>,
        params: SearchParams,
    ) -> Result<Vec<QueryHits>, Error> {
        let span = tracing::Span::current();
        vectors
            .par_iter()
            .map(|vector| -> Result<QueryHits, Error> {
                let _entered = span.clone().entered();
                // FIX: Process each query vector for space
                let processed_query = self.dense().process_vector_for_space(vector.clone());

                // Six read guards per worker, in the one documented order
                // and held across the traversal. Every guard is a read, so
                // the rule that no path forks to rayon holding a write
                // guard is untouched. See `search_one` for why `id_map` is
                // first and where the raw vector map went.
                let id_map = self.id_map.read().unwrap();
                let rev_map = self.rev_map.read().unwrap();
                let index = self.dense().index.read().unwrap();
                let code_store = self.dense().pq_codes.read().unwrap();
                let metadata_store = self.vector_metadata.read().unwrap();
                let column_store = self.columns.read().unwrap();
                let vector_store = RawVectors {
                    id_map: &id_map,
                    graph: index.graph(),
                };

                let plan =
                    self.admit_plan(filter_conditions, &column_store, &rev_map, &metadata_store);

                // The same over-fetch the other two search paths apply.
                let fetch_k = params.fetch_k(rev_map.len());
                let budget = Self::dense_budget(&params);

                let hits =
                    plan.run(|admit| index.search(&processed_query, fetch_k, admit, &budget))?;
                let candidates = Scored::resolve(hits, &rev_map).cut(fetch_k);

                self.collect_hits(
                    candidates,
                    &processed_query,
                    vector_store,
                    &code_store,
                    &metadata_store,
                    params,
                )
            })
            .collect()
    }

    /// Raw search with no page building (for benchmarking)
    ///
    /// Two read guards in declared order, `rev_map` then the index, and the
    /// index guard is held across the resolution as well as the traversal.
    /// This path takes no filter, so it takes no storage guards and the
    /// index runs under its live set alone.
    pub(super) fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        let rev_map = self.rev_map.read().unwrap();
        let index = self.dense().index.read().unwrap();

        let budget = Budget {
            ef: Some(100),
            ..Budget::default()
        };
        let hits = index
            .search(query, 10, &Candidates::All, &budget)
            .unwrap_or_else(|e| {
                error!(target: LOG_TARGET, operation = "raw_search", error = %e, "Search failed");
                Hits {
                    items: Vec::new(),
                    kind: zeusdb_vector_core::ScoreKind::Distance,
                    exact: false,
                }
            });
        Scored::resolve(hits, &rev_map)
            .items
            .into_iter()
            .map(|(ext_id, distance)| (ext_id.clone(), distance))
            .collect()
    }

    /// Search the sparse space, where the collection declares one.
    ///
    /// A one arm query, reachable from Rust alone. It takes the same admit
    /// set a dense search takes under the same filter, so a filter selects
    /// the same records whichever space is asked. The page is best first,
    /// by score and then by external id among equal scores, which is the
    /// rule an exact dense page follows. Scores are the sparse space's own,
    /// higher better, and a record sharing no dimension with the query
    /// never appears, so the page may be shorter than `top_k`. See
    /// [`Collection::query`] for the query it is.
    pub fn search_sparse(
        &self,
        query: SparseRef<'_>,
        filter_conditions: Option<&Filter>,
        top_k: usize,
        idf: IdfScope,
    ) -> Result<SparseHits, Error> {
        let arms = [Arm::Sparse { vector: query, idf }];
        self.one_arm(&arms, filter_conditions, top_k)
    }

    /// Search the sparse space with a text, where the space has a text
    /// layer.
    ///
    /// A one arm query. The text is counted into term ids as the records
    /// were, under the dictionary's read guard taken after the sparse
    /// index's and held with it through the search, and a term no record
    /// has carried is dropped. The rest is `search_sparse`.
    pub fn search_text(
        &self,
        text: &str,
        filter_conditions: Option<&Filter>,
        top_k: usize,
        idf: IdfScope,
    ) -> Result<SparseHits, Error> {
        let arms = [Arm::Text { text, idf }];
        self.one_arm(&arms, filter_conditions, top_k)
    }

    /// One arm's page as (external id, score), best first.
    fn one_arm(
        &self,
        arms: &[Arm<'_>],
        filter_conditions: Option<&Filter>,
        top_k: usize,
    ) -> Result<SparseHits, Error> {
        let page = self.query(&Query {
            arms,
            filter: filter_conditions,
            k: top_k,
            fetch: None,
            fusion: Fusion::default(),
        })?;
        Ok(page
            .hits
            .into_iter()
            .map(|hit| (hit.id, hit.score))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::Scored;

    /// An exact page is ordered by distance and then by external id, and cut.
    /// A traversal's page is returned as the traversal produced it, so a tie
    /// the string order would reverse stays in the heap's order.
    #[test]
    fn the_tie_break_applies_to_exact_pages_only() {
        let a = "a".to_string();
        let b = "b".to_string();
        let c = "c".to_string();
        // `b` before `a` at the same distance, which is the traversal's order
        // on the 12,002 record l2 corpus the tie-break probe records.
        let items = vec![(&b, 1.0f32), (&a, 1.0), (&c, 0.5)];

        let exact = Scored::exact(items.clone()).cut(2);
        assert!(exact.exact);
        assert_eq!(exact.items, vec![(&c, 0.5), (&a, 1.0)]);

        let traversed = Scored::traversed(items.clone()).cut(2);
        assert!(!traversed.exact);
        assert_eq!(traversed.items, items);
    }
}
