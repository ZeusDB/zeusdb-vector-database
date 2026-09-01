//! The dense index, being the value under a dense space's graph guard and
//! the first implementor of the seam in `zeusdb-vector-core`.
//!
//! # What it is
//!
//! The graph, the live set the graph is searched under, and what an insertion
//! or a search needs from the space's declaration, being the metric, the
//! width, the degree, the quantizer where the space has one and whether the
//! space keeps a raw vector beside every code. The guard around it is the
//! space's `index` field, so a search runs under a read guard on this value
//! and an installation under a write guard, which is the two-phase insertion
//! the graph was built for.
//!
//! # The live set
//!
//! The index keeps its own bitmap of the records it holds. It is the same
//! set the collection keeps beside its reverse map, maintained on the same
//! writes, and it is what lets the index answer `len`, `holds`, `stranded`
//! and a removal of an id it never held without reaching into the
//! collection. A search under a set admitting everything runs under this
//! bitmap alone, which is the traversal an unfiltered search has always run.
//!
//! # The two paths
//!
//! An admit set the index can enumerate whose count is at or under
//! [`FULL_SCAN_THRESHOLD`] is scored exactly, record by record, from the
//! graph's own store or from the reconstruction of the record's codes where
//! the store holds none. Everything else traverses the graph under the admit
//! set as a predicate. The collection decides what the admit set is, being a
//! filter's bitmap, the matches of a metadata walk, or a conjunction of a
//! bound and a predicate, and the threshold is applied here, so the rule is
//! stated once whatever built the set.
//!
//! # The tie rule
//!
//! An exact page is ordered by distance and then by internal id, and it is
//! cut to `k` keeping every record tied at the distance of the last member
//! where the caller asked for that. The collection orders equal distances by
//! external id string, a rule that predates this seam and that every recorded
//! page holds, and it can apply that rule at the boundary only if the whole
//! tie group is in the page. A traversal's page is returned in the
//! traversal's own order and holds at most `k`.
//!
//! # Cost
//!
//! Two units, both timed on the graph itself when the index is opened,
//! again whenever the graph is replaced, and again each time the live count
//! doubles past the smallest graph worth timing, so a collection built by
//! insertion prices its searches on what it holds rather than on the floor
//! it started from. Neither is persisted, since both move with the machine
//! and the build. A traversal is priced per unit of
//! `ef` from a whole search timed on the graph, because a traversal's time
//! is the memory it touches and a kernel timed in a loop runs from cache:
//! at width 100 a search at `ef` 200 cost as much as eleven thousand kernel
//! evaluations and visited about a thousand nodes. Both units are timed
//! over records scattered across the store, under a bitmap test per
//! candidate as every search the collection runs pays, and the search is
//! asked for a vector the graph does not hold. An exact scan is priced per
//! record from the kernel. A filtered traversal grows as the admitted share
//! falls, from two measured points. An index too small to time takes a
//! compiled-in floor for each.
//!
//! What the figure prices is the arm's own work. At 50,000 records of width
//! 100 the traversal measured 106 microseconds and a filter admitting every
//! record added 70 more, being the filter's evaluation over the column
//! codes and the page's assembly, which no arm's cost includes and which
//! `ArmPlan::cost` says. A caller reading the plan against a filtered
//! query's wall time sees that difference, not a unit that is wrong.

use std::path::Path;
use std::sync::Arc;

use tracing::error;
use zeusdb_vector_core::{
    restore_graph, Admit, ArtefactRecord, Bitmap, Bounds, Budget, Cost, Dense, DumpBounds, Error,
    Hit, Hits, Inventory, Ledger, Persist, Planned, Prepared, Record, RecordId, Restore, ScoreKind,
    Selectivity, VectorGraph, VectorIndex, DUMP_FILENAME, PQ,
};

use super::search::FULL_SCAN_THRESHOLD;
use crate::{prepare_reconstruction, raw_distance_fn, reconstruction_needs_unit};

/// The target every record this file emits carries, being the search
/// module's, since the records here are the ones the traversal emitted from
/// there.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::search";

/// What the floor prices one unit of `ef` at, in kernel evaluations, where
/// the graph is too small to time a search on.
///
/// From one measured point, being 1,192 evaluations for a search at `ef` 200
/// and `k` 10 over 50,000 records at `m` 16. Linear in `ef` is an assumption
/// past that point, and it is the floor alone that uses it: an index large
/// enough to time prices a traversal from a search it ran on itself.
const FLOOR_VISITS_PER_EF: f64 = 6.0;

/// The search timed at open, being `k`, `ef` and how many searches the
/// median is taken over. A search at these settings on 50,000 points takes
/// a quarter of a millisecond at width 100 and a millisecond at 1,536, so
/// the nine together cost a few milliseconds when an index is opened.
const CALIBRATION_SEARCH_K: usize = 10;
const CALIBRATION_SEARCH_EF: usize = 100;
const CALIBRATION_SEARCHES: usize = 9;

/// How the traversal's cost grows as the admitted share falls.
///
/// A filtered traversal visits nodes the predicate rejects, and the more it
/// rejects the further it walks for each result. Fitted to two measured
/// points on 100,000 records, being four times the unfiltered cost at a tenth
/// admitted and thirty two times at a hundredth, which an exponent of three
/// quarters on the inverse of the share reproduces to within a factor of 1.4
/// at both. An assumption beyond those two points.
const SELECTIVITY_EXPONENT: f64 = 0.75;

/// Nodes the store holds before the unit cost is timed rather than taken
/// from the floor. Below this the evaluations cycle through so few vectors
/// that the figure is a cache figure rather than a kernel figure.
const CALIBRATION_MIN_NODES: usize = 64;

/// Kernel evaluations timed per round, and rounds, at open. Ten thousand
/// rather than fewer because a shorter run stays in cache and reads a
/// figure a scan over scattered records never sees: at width 100 the
/// figure moved from 14 to 58 nanoseconds between five hundred and ten
/// thousand evaluations.
const CALIBRATION_EVALUATIONS: usize = 10_000;
const CALIBRATION_ROUNDS: usize = 5;

/// The two units, in nanoseconds: one kernel evaluation over a scattered
/// record, and one unit of `ef` in a traversal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DenseUnits {
    pub(crate) distance_ns: f64,
    pub(crate) ef_ns: f64,
    pub(crate) measured: bool,
}

/// What phase one hands to phase two: the codes a quantized graph installs
/// and the plan the graph made for them, or the plan for a raw vector.
struct DensePlan {
    codes: Option<Vec<u8>>,
    planned: Option<Planned>,
}

/// The value under a dense space's graph guard.
pub(crate) struct DenseIndex {
    graph: VectorGraph,
    /// The records this index holds a node for and has not removed.
    live: Bitmap,
    live_count: usize,
    metric: String,
    dim: usize,
    pq: Option<Arc<PQ>>,
    keeps_raw: bool,
    /// What a scan and a traversal cost per unit on this graph.
    units: DenseUnits,
}

impl DenseIndex {
    /// Wrap a graph. The live set starts empty and the unit cost is timed on
    /// the graph's store, or taken from the floor where the store is too
    /// small.
    pub(crate) fn new(
        graph: VectorGraph,
        metric: &str,
        dim: usize,
        pq: Option<Arc<PQ>>,
        keeps_raw: bool,
    ) -> Self {
        let mut index = DenseIndex {
            graph,
            live: Bitmap::default(),
            live_count: 0,
            metric: metric.to_string(),
            dim,
            pq,
            keeps_raw,
            units: DenseUnits {
                distance_ns: 0.0,
                ef_ns: 0.0,
                measured: false,
            },
        };
        index.calibrate();
        index
    }

    pub(crate) fn graph(&self) -> &VectorGraph {
        &self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut VectorGraph {
        &mut self.graph
    }

    /// The live set, for a caller that wants to check it against the
    /// collection's.
    pub(crate) fn live_set(&self) -> &Bitmap {
        &self.live
    }

    /// Replace the live set with the ids the collection holds, which the
    /// loader does once the id mappings are back.
    pub(crate) fn set_live<I: IntoIterator<Item = usize>>(&mut self, ids: I) {
        let mut live = Bitmap::default();
        let mut count = 0;
        for id in ids {
            live.insert(id);
            count += 1;
        }
        self.live = live;
        self.live_count = count;
    }

    /// Swap the graph and keep everything else, which is what a compaction,
    /// a rebuild and the training transition do.
    ///
    /// `timed` is what [`DenseIndex::time_graph`] measured on the replacement
    /// before the caller took the write guard, so the guard is held for the
    /// swap alone and never across a timing run. `None` takes the floor.
    pub(crate) fn replace_graph(
        &mut self,
        graph: VectorGraph,
        timed: Option<(f64, f64)>,
    ) -> VectorGraph {
        let old = std::mem::replace(&mut self.graph, graph);
        self.set_units(timed);
        old
    }

    /// Time the two units on a graph that is not yet installed, so a
    /// replacement is measured with no guard held. The kernel over records
    /// scattered across the store, and a whole search per unit of `ef`,
    /// both under a bitmap test admitting every node, which is the
    /// predicate's cost without its outcome.
    pub(crate) fn time_graph(graph: &VectorGraph) -> Option<(f64, f64)> {
        let nodes = graph.nb_points();
        if nodes < CALIBRATION_MIN_NODES {
            return None;
        }
        let mut every = Bitmap::with_slots(nodes);
        for node in 0..nodes {
            every.insert(node);
        }
        let admits = |id: usize| every.contains(id);
        let distance_ns = graph
            .time_distance_ns(CALIBRATION_EVALUATIONS, CALIBRATION_ROUNDS, &admits)
            .filter(|ns| ns.is_finite() && *ns > 0.0)?;
        let search_ns = graph
            .time_search_ns(
                CALIBRATION_SEARCH_K,
                CALIBRATION_SEARCH_EF,
                CALIBRATION_SEARCHES,
                &admits,
            )
            .filter(|ns| ns.is_finite() && *ns > 0.0)?;
        Some((distance_ns, search_ns / CALIBRATION_SEARCH_EF as f64))
    }

    /// Install a quantizer, which the loader does before it restores a
    /// quantized graph.
    pub(crate) fn set_pq(&mut self, pq: Option<Arc<PQ>>) {
        self.pq = pq;
    }

    pub(crate) fn set_keeps_raw(&mut self, keeps_raw: bool) {
        self.keeps_raw = keeps_raw;
    }

    /// The two units this index prices a search with, and whether they were
    /// timed or taken from the floor.
    #[cfg(test)]
    pub(crate) fn units(&self) -> DenseUnits {
        self.units
    }

    /// Time the two units on the graph's own store, or take the floor where
    /// the store is too small.
    ///
    /// The kernel's floor is a line through two measured points on a raw
    /// graph, 174 nanoseconds at width 100 and 1,026 at width 1,536, and one
    /// point on a quantized graph, which reads one table entry per
    /// subvector. The traversal's floor is that figure times the visits one
    /// unit of `ef` was measured to cost. Both are assumptions about this
    /// machine that a timed figure replaces.
    pub(crate) fn calibrate(&mut self) {
        let timed = Self::time_graph(&self.graph);
        self.set_units(timed);
    }

    /// Whether the index has just reached a size at which the units are
    /// timed again, being each doubling of the live count from the smallest
    /// graph worth timing. The collection times the graph under a read
    /// guard and then adopts the figures under the write guard, so no
    /// search waits on a timing run.
    pub(crate) fn due_for_timing(&self) -> bool {
        self.live_count >= CALIBRATION_MIN_NODES && self.live_count.is_power_of_two()
    }

    /// Keep the timed units, or the floor where there are none.
    pub(crate) fn set_units(&mut self, timed: Option<(f64, f64)>) {
        match timed {
            Some((distance_ns, ef_ns)) => {
                self.units = DenseUnits {
                    distance_ns,
                    ef_ns,
                    measured: true,
                };
            }
            None => {
                let distance_ns = if self.graph.is_quantized() {
                    10.0 + 1.5 * self.pq.as_ref().map_or(8, |pq| pq.subvectors()) as f64
                } else {
                    115.0 + 0.6 * self.dim as f64
                };
                self.units = DenseUnits {
                    distance_ns,
                    ef_ns: distance_ns * FLOOR_VISITS_PER_EF,
                    measured: false,
                };
            }
        }
    }

    /// The traversal width a search takes when the caller names none.
    fn default_ef(&self, k: usize) -> usize {
        match self.metric.as_str() {
            "l1" | "l2" => std::cmp::max(2 * k, 150),
            _ => std::cmp::max(2 * k, 100),
        }
    }

    /// The distance from `query` to one record, on the raw scale.
    ///
    /// The stored raw vector where the graph holds one and the reconstruction
    /// from the record's codes where it does not, normalised first where the
    /// space requires it. A record holding neither cannot exist, since every
    /// insertion writes one or the other, and is sorted last.
    fn exact_distance(
        &self,
        query: &[f32],
        id: usize,
        distance: fn(&[f32], &[f32]) -> f32,
        needs_unit: bool,
    ) -> f32 {
        if let Some(stored) = self.graph.raw_vector(id) {
            return distance(query, stored);
        }
        let reconstructed = self
            .graph
            .codes_of(id)
            .and_then(|codes| self.pq.as_ref()?.reconstruct(codes).ok());
        match reconstructed {
            Some(reconstructed) => {
                distance(query, &prepare_reconstruction(needs_unit, reconstructed))
            }
            None => f32::INFINITY,
        }
    }

    /// Score every enumerated id the index holds and cut the page.
    fn score_exact(&self, query: &[f32], ids: &[RecordId], k: usize, boundary_ties: bool) -> Hits {
        let distance = raw_distance_fn(&self.metric);
        let needs_unit = reconstruction_needs_unit(&self.metric);
        let mut scored: Vec<(f32, u32)> = ids
            .iter()
            .filter(|id| self.live.contains(id.slot()))
            .map(|id| {
                (
                    self.exact_distance(query, id.slot(), distance, needs_unit),
                    id.0,
                )
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut end = k.min(scored.len());
        if boundary_ties && end > 0 && end < scored.len() {
            let boundary = scored[end - 1].0;
            while end < scored.len() && scored[end].0 == boundary {
                end += 1;
            }
        }
        scored.truncate(end);
        Hits {
            items: scored
                .into_iter()
                .map(|(score, id)| Hit {
                    id: RecordId(id),
                    score,
                })
                .collect(),
            kind: ScoreKind::Distance,
            exact: true,
        }
    }

    /// Run the traversal under one predicate.
    ///
    /// Three predicates, each monomorphised into the traversal by the
    /// graph's generic parameter. A set admitting everything is the live
    /// bit alone, which is the unfiltered search. A bitmap is its bit and
    /// then the live bit, in that order so a rejected node costs one word
    /// read. Anything else is the live bit and then the table call.
    fn traverse(&self, query: &[f32], k: usize, ef: usize, admit: &dyn Admit) -> Hits {
        let quantized = self.graph.is_quantized();
        let operation = if quantized {
            "adc_search"
        } else {
            "raw_search"
        };
        let live = &self.live;
        let searched = if admit.admits_all() {
            let admits = |id: &usize| live.contains(*id);
            self.graph.search(query, k, ef, Some(&admits))
        } else if let Some(bitmap) = admit.as_bitmap() {
            let admits = |id: &usize| bitmap.contains(*id) && live.contains(*id);
            self.graph.search(query, k, ef, Some(&admits))
        } else {
            let admits = |id: &usize| live.contains(*id) && admit.admits(RecordId::from_slot(*id));
            self.graph.search(query, k, ef, Some(&admits))
        };
        let graph_hits = searched.unwrap_or_else(|e| {
            error!(target: LOG_TARGET, operation = operation, error = %e, "Graph search failed");
            Vec::new()
        });
        let mut items: Vec<Hit> = graph_hits
            .into_iter()
            .map(|hit| Hit {
                id: RecordId::from_slot(hit.internal_id),
                score: hit.distance,
            })
            .collect();
        // Put an l2 index's approximate scores back on the scale it reports.
        // The quantized l2 scorer sums a table of squared distances and takes
        // no root, because the root is monotone and would cost one per
        // evaluation to change nothing, so the root is taken once per
        // returned candidate here. Cosine arrives already on its own scale,
        // since its conversion is not monotone in the sum and the scorer has
        // to make it.
        if quantized && self.metric == "l2" {
            for item in &mut items {
                item.score = item.score.max(0.0).sqrt();
            }
        }
        Hits {
            items,
            kind: ScoreKind::Distance,
            exact: false,
        }
    }

    /// Phase one for a vector, whatever the graph holds.
    fn plan(&self, id: RecordId, vector: &[f32]) -> Result<DensePlan, Error> {
        if self.graph.is_quantized() {
            let pq = self.pq.as_ref().ok_or(Error::NoQuantizer)?;
            let codes = pq.quantize(vector).map_err(|e| {
                error!(target: "zeusdb_vector_database::hnsw_index::insert", operation = "add_quantized_vector",
                    internal_id = id.0,
                    error = %e,
                    "Failed to quantize vector"
                );
                Error::QuantizeFailed(e)
            })?;
            let planned = self.graph.plan(Record::Codes {
                codes: &codes,
                raw: self.keeps_raw.then_some(vector),
            });
            Ok(DensePlan {
                codes: Some(codes),
                planned,
            })
        } else {
            Ok(DensePlan {
                codes: None,
                planned: self.graph.plan(Record::Raw(vector)),
            })
        }
    }
}

impl VectorIndex<Dense> for DenseIndex {
    fn len(&self) -> usize {
        self.live_count
    }

    fn holds(&self, id: RecordId) -> bool {
        self.live.contains(id.slot())
    }

    /// Phase one. Draws the level, descends and chooses the neighbour lists
    /// under the read guard, and quantizes first where the graph holds
    /// codes.
    fn prepare(&self, id: RecordId, vector: &[f32]) -> Result<Prepared, Error> {
        Ok(Prepared::new(self.plan(id, vector)?))
    }

    /// Phase two. Appends the node and installs its lists under the write
    /// guard. A caller that skipped `prepare` has both phases run here.
    ///
    /// A plan the graph refused, because the record's element type is not
    /// the graph's, installs nothing and is logged by the graph, which is
    /// what the two-phase insertion has always done with it; the record is
    /// still counted held, since the collection holds it.
    fn insert(&mut self, id: RecordId, vector: &[f32], prepared: Prepared) -> Result<(), Error> {
        if self.live.contains(id.slot()) {
            return Err(Error::RecordAlreadyHeld { id: id.0 });
        }
        let plan = match prepared.take::<DensePlan>() {
            Some(plan) => plan,
            None => self.plan(id, vector)?,
        };
        if let Some(planned) = plan.planned {
            let record = match &plan.codes {
                Some(codes) => Record::Codes {
                    codes,
                    raw: self.keeps_raw.then_some(vector),
                },
                None => Record::Raw(vector),
            };
            self.graph.install(record, id.slot(), planned);
        }
        self.live.insert(id.slot());
        self.live_count += 1;
        Ok(())
    }

    /// Strands the node. The graph keeps it, since a stranded node still
    /// routes a traversal, and `compact` rebuilds the graph without it.
    fn remove(&mut self, id: RecordId) -> Result<(), Error> {
        if !self.live.contains(id.slot()) {
            return Err(Error::RecordNotHeld { id: id.0 });
        }
        self.live.remove(id.slot());
        self.live_count -= 1;
        Ok(())
    }

    /// Nodes the graph holds beyond the live records, which is what
    /// removal and overwrite strand and what a compaction reclaims.
    fn stranded(&self) -> usize {
        self.graph.nb_points().saturating_sub(self.live_count)
    }

    fn vector(&self, id: RecordId) -> Option<&[f32]> {
        if !self.live.contains(id.slot()) {
            return None;
        }
        self.graph.raw_vector(id.slot())
    }

    /// The raw vector where the graph holds one and the reconstruction from
    /// the record's codes where it does not.
    fn recover(&self, id: RecordId) -> Option<Vec<f32>> {
        if let Some(raw) = self.vector(id) {
            return Some(raw.to_vec());
        }
        if !self.live.contains(id.slot()) {
            return None;
        }
        let codes = self.graph.codes_of(id.slot())?;
        self.pq.as_ref()?.reconstruct(codes).ok()
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        admit: &dyn Admit,
        budget: &Budget,
    ) -> Result<Hits, Error> {
        let ef = budget.ef.unwrap_or_else(|| self.default_ef(k));
        if let Some(admitted) = admit.len_hint() {
            if admitted <= FULL_SCAN_THRESHOLD {
                let mut ids = Vec::with_capacity(admitted);
                if admit.enumerate(&mut |id| {
                    ids.push(id);
                    true
                }) {
                    return Ok(self.score_exact(query, &ids, k, budget.boundary_ties));
                }
            }
        }
        Ok(self.traverse(query, k, ef, admit))
    }

    /// Priced from `ef` and the admitted share, and the query is ignored,
    /// since a traversal's work does not depend on which vector is asked.
    fn cost(&self, _query: &[f32], k: usize, admitted: Option<&Selectivity>) -> Cost {
        let ef = self.default_ef(k) as f64;
        match admitted {
            Some(selectivity) if (selectivity.expected as usize) <= FULL_SCAN_THRESHOLD => Cost {
                work_ns: selectivity.expected as f64 * self.units.distance_ns,
                exact: true,
            },
            Some(selectivity) => {
                let live = self.live_count.max(1) as f64;
                let share = (selectivity.expected as f64 / live).clamp(1e-3, 1.0);
                Cost {
                    work_ns: ef * self.units.ef_ns / share.powf(SELECTIVITY_EXPONENT),
                    exact: false,
                }
            }
            None => Cost {
                work_ns: ef * self.units.ef_ns,
                exact: false,
            },
        }
    }
}

/// The dump's name under a prefix.
fn dump_name(prefix: &str) -> String {
    format!("{prefix}{DUMP_FILENAME}")
}

impl Persist for DenseIndex {
    /// Write the graph dump under `prefix` and record its length.
    ///
    /// Its length rather than a digest, because the dump streams itself and
    /// seeks back to write its header, so hashing it whole would mean reading
    /// the largest artefact in the directory back off the disk for a
    /// guarantee its own two checksums already give.
    fn write(&self, prefix: &str, dir: &Path, ledger: &mut dyn Ledger) -> Result<(), Error> {
        let target = if prefix.is_empty() {
            dir.to_path_buf()
        } else {
            let target = dir.join(prefix);
            std::fs::create_dir_all(&target).map_err(|e| Error::ArtefactCreateFailed {
                name: prefix.to_string(),
                error: e.to_string(),
            })?;
            target
        };
        let filename = self.graph.dump(&target).map_err(Error::GraphDumpFailed)?;
        let bytes = std::fs::metadata(target.join(&filename))
            .map(|meta| meta.len())
            .map_err(|e| Error::DumpLengthUnreadable(e.to_string()))?;
        ledger.record(
            &dump_name(prefix),
            ArtefactRecord {
                bytes,
                checksum: None,
            },
        );
        Ok(())
    }

    fn artefact_names(&self, prefix: &str) -> Vec<String> {
        vec![dump_name(prefix)]
    }
}

/// What the loader knows before it opens a dense index from its dump.
pub(crate) struct DenseOpen {
    pub(crate) metric: String,
    pub(crate) dim: usize,
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    /// The trained quantizer, where the saved graph was a quantized one.
    pub(crate) pq: Option<Arc<PQ>>,
    pub(crate) keeps_raw: bool,
}

impl Restore for DenseIndex {
    type Config = DenseOpen;

    /// Restore the graph the save wrote.
    ///
    /// The inventory's recorded length is checked before the dump is read,
    /// because a self-consistent dump from a different save of an index of
    /// the same shape would otherwise pass the dump's own checksums. Every
    /// error is a reason to rebuild rather than to fail, which the caller
    /// decides. The live set comes back empty and the caller fills it from
    /// the id mappings, which the dump does not carry.
    fn restore(
        config: &DenseOpen,
        prefix: &str,
        dir: &Path,
        inventory: &dyn Inventory,
        bounds: &Bounds,
    ) -> Result<Self, Error> {
        let target = if prefix.is_empty() {
            dir.to_path_buf()
        } else {
            dir.join(prefix)
        };
        if let Some(recorded) = inventory.recorded(&dump_name(prefix)) {
            let found = std::fs::metadata(target.join(DUMP_FILENAME))
                .map(|meta| meta.len())
                .unwrap_or(0);
            if found != recorded.bytes {
                return Err(Error::Engine(format!(
                    "the graph dump is {} bytes and manifest.json records it as {}",
                    found, recorded.bytes
                )));
            }
        }
        let pq = config.pq.as_ref().filter(|pq| pq.is_trained()).cloned();
        let (graph, _nodes) = restore_graph(
            &target,
            &config.metric,
            config.m,
            config.ef_construction,
            config.dim,
            pq,
            DumpBounds {
                min_nodes: bounds.min_records,
                max_origin_id: bounds.max_records,
            },
        )
        .map_err(Error::Engine)?;
        Ok(DenseIndex::new(
            graph,
            &config.metric,
            config.dim,
            config.pq.clone(),
            config.keeps_raw,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeusdb_vector_core::{Candidates, NB_LAYER_MAX};

    fn index_of(vectors: &[Vec<f32>]) -> DenseIndex {
        let graph = VectorGraph::new_raw("l2", 2, 4, 64, NB_LAYER_MAX as usize, 50);
        let mut index = DenseIndex::new(graph, "l2", 2, None, false);
        for (i, v) in vectors.iter().enumerate() {
            let id = RecordId(i as u32 + 1);
            let prepared = index.prepare(id, v).unwrap();
            index.insert(id, v, prepared).unwrap();
        }
        index
    }

    /// A held id is refused, an unknown id cannot be removed, and the live
    /// and stranded counts follow every write.
    #[test]
    fn held_ids_are_refused_and_the_counts_follow_every_write() {
        let mut index = index_of(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(index.len(), 3);
        assert_eq!(index.stranded(), 0);
        assert!(matches!(
            index.insert(RecordId(2), &[5.0, 5.0], Prepared::none()),
            Err(Error::RecordAlreadyHeld { id: 2 })
        ));
        assert!(matches!(
            index.remove(RecordId(9)),
            Err(Error::RecordNotHeld { id: 9 })
        ));
        index.remove(RecordId(2)).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.stranded(), 1);
        assert!(!index.holds(RecordId(2)));
        assert!(index.vector(RecordId(2)).is_none());
        assert_eq!(index.recover(RecordId(3)), Some(vec![0.0, 1.0]));
        // The node stays, so the graph still holds three.
        assert_eq!(index.graph().nb_points(), 3);
    }

    /// A small enumerable admit set is scored exactly with the boundary tie
    /// group kept on request, and a set admitting everything traverses
    /// under the live set alone.
    #[test]
    fn a_small_admit_set_is_scored_exactly_and_everything_traverses() {
        let mut index = index_of(&[
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![3.0, 0.0],
        ]);
        index.remove(RecordId(5)).unwrap();
        let query = [0.0f32, 0.0];

        let sorted = Candidates::Sorted(vec![RecordId(1), RecordId(2), RecordId(3), RecordId(5)]);
        let page = index
            .search(&query, 2, &sorted, &Budget::default())
            .unwrap();
        assert!(page.exact);
        let ids: Vec<u32> = page.items.iter().map(|h| h.id.0).collect();
        assert_eq!(ids, vec![1, 2]);

        let page = index
            .search(
                &query,
                2,
                &sorted,
                &Budget {
                    boundary_ties: true,
                    ..Budget::default()
                },
            )
            .unwrap();
        let ids: Vec<u32> = page.items.iter().map(|h| h.id.0).collect();
        assert_eq!(ids, vec![1, 2, 3], "the removed record 5 is never scored");

        let page = index
            .search(&query, 10, &Candidates::All, &Budget::default())
            .unwrap();
        assert!(!page.exact);
        let mut ids: Vec<u32> = page.items.iter().map(|h| h.id.0).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4],
            "the stranded node routes and is never returned"
        );
    }

    /// The cost is priced in nanoseconds from the two units, is exact under
    /// a small admit set, and grows as the admitted share falls.
    #[test]
    fn cost_is_priced_from_the_units_and_the_admitted_share() {
        let index = index_of(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let units = index.units();
        assert!(!units.measured, "two nodes is below the calibration floor");
        assert!(units.distance_ns > 0.0 && units.ef_ns > units.distance_ns);
        let whole = index.cost(&[0.0, 0.0], 10, None);
        assert!(!whole.exact);
        assert!((whole.work_ns - 150.0 * units.ef_ns).abs() < 1e-6);
        let narrow = index.cost(&[0.0, 0.0], 10, Some(&Selectivity::exact(1)));
        assert!(narrow.exact);
        assert!((narrow.work_ns - units.distance_ns).abs() < 1e-6);
        let broad = index.cost(&[0.0, 0.0], 10, Some(&Selectivity::exact(6_000)));
        assert!(!broad.exact);
        assert!(broad.work_ns >= whole.work_ns);
    }

    /// A graph large enough to time reports measured units, and the
    /// traversal's unit is above the kernel's, since a traversal visits
    /// several nodes per unit of `ef`.
    #[test]
    fn a_large_enough_graph_times_both_units() {
        let vectors: Vec<Vec<f32>> = (0..400)
            .map(|i| vec![(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        let mut index = index_of(&vectors);
        assert!(
            !index.units().measured,
            "built by insertion, so the floor until it is timed"
        );
        assert!(index.due_for_timing() || index.len() != 256);
        index.calibrate();
        let units = index.units();
        assert!(units.measured);
        assert!(units.distance_ns > 0.0 && units.distance_ns.is_finite());
        assert!(units.ef_ns > units.distance_ns);
    }
}
