//! The structure, and the trait it implements.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{debug, trace};
use zeusdb_vector_core::{
    Admit, Bitmap, Budget, CorpusStats, Cost, Error, Hits, Inventory, Ledger, Persist, Prepared,
    RecordId, Restore, Selectivity, Sparse, SparseRef, SparseVector, VectorIndex,
};

use crate::calibrate::UnitCosts;
use crate::search::Mode;
use crate::LOG_TARGET;

/// The saturation parameter a term frequency weighted space applies unless
/// told otherwise. The value the weighting is most often published with.
pub const DEFAULT_BM25_K1: f32 = 1.2;

/// The length normalisation parameter a term frequency weighted space
/// applies unless told otherwise. Zero applies none and one applies it in
/// full.
pub const DEFAULT_BM25_B: f32 = 0.75;

/// How a stored value and a query value combine into a score.
///
/// Written into `config.json` by value as `{"type": "dot"}` or
/// `{"type": "bm25", "k1": 1.2, "b": 0.75}`, so a saved space is scored as it
/// was declared.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Weighting {
    /// The product of the stored value and the query value, summed over the
    /// dimensions the two share. The values mean whatever the caller's
    /// encoder decided.
    Dot,
    /// Every stored value is a term frequency, and a query term contributes
    /// its query value times the rarity of the term over the corpus times
    /// the saturated, length-normalised frequency
    ///
    /// ```text
    /// tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / mean_length))
    /// ```
    ///
    /// where `length` is the record's total term frequency, `mean_length`
    /// is the mean over the live records, and rarity is
    /// `ln(1 + (n - df + 0.5) / (df + 0.5))` over a corpus of `n` records
    /// of which `df` carry the term. The corpus is the admitted records by
    /// default; see `IdfScope`. Nothing derived from a corpus statistic is
    /// stored, so every query is scored under the statistics of the moment
    /// and a record leaving or arriving moves no stored weight.
    Bm25 { k1: f32, b: f32 },
}

impl Weighting {
    /// Term frequency weighting at the published defaults.
    pub const BM25: Weighting = Weighting::Bm25 {
        k1: DEFAULT_BM25_K1,
        b: DEFAULT_BM25_B,
    };

    /// `k1` is finite and at least zero, and `b` is between zero and one.
    pub fn validate(&self) -> Result<(), Error> {
        if let Weighting::Bm25 { k1, b } = *self {
            if !(k1.is_finite() && k1 >= 0.0) {
                return Err(Error::SparseWeightingInvalid {
                    parameter: "k1",
                    value: k1,
                    rule: "finite and at least zero",
                });
            }
            if !(b.is_finite() && (0.0..=1.0).contains(&b)) {
                return Err(Error::SparseWeightingInvalid {
                    parameter: "b",
                    value: b,
                    rule: "between zero and one",
                });
            }
        }
        Ok(())
    }

    /// Whether every stored value is read as a term frequency, which is
    /// what makes a value at or below zero a refusal at insert.
    pub fn reads_term_frequency(&self) -> bool {
        matches!(self, Weighting::Bm25 { .. })
    }
}

/// Dead share of a list, in percent of its length, above which a lazy unlink
/// rewrites the list.
///
/// Measured at 50,000 records on two synthetic regimes, with half the corpus
/// removed in a shuffled order. At 25 percent the scan after half removed
/// held to 1.2 times a compacted index's, and a removal cost 4 to 14
/// microseconds. At 10 percent the removal cost and the scan at every removal
/// level are what decided the figure here; see the crate's measurement record.
pub const DEFAULT_LAZY_THRESHOLD_PERCENT: u32 = 10;

/// What `remove` does to the record's postings.
///
/// Written into `config.json` as `"strand"`, `"lazy"` or `"eager"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Unlink {
    /// Leave every posting where it is, counting it dead on its list. The
    /// dead bitmap hides the record and `compact` reclaims the postings. This
    /// is what a graph does with a node.
    Strand,
    /// Count the dead posting on each list the record sits in, and rewrite a
    /// list once its dead share crosses the threshold.
    Lazy,
    /// Remove the posting from every list the record sits in, now. A
    /// `Vec::remove` on a common term's list moves most of the list, which is
    /// where its cost goes.
    Eager,
}

/// How a sparse space is declared.
///
/// Written into `config.json` by value, every field named, so a directory
/// records the declaration itself rather than a name for it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseConfig {
    pub unlink: Unlink,
    /// Dead share, in percent of a list's length, above which a lazy unlink
    /// rewrites the list. Read under `Unlink::Lazy` alone.
    pub lazy_threshold_percent: u32,
    /// The scoring rule.
    pub weighting: Weighting,
}

impl Default for SparseConfig {
    fn default() -> Self {
        SparseConfig {
            unlink: Unlink::Lazy,
            lazy_threshold_percent: DEFAULT_LAZY_THRESHOLD_PERCENT,
            weighting: Weighting::Dot,
        }
    }
}

impl SparseConfig {
    /// The rules a declaration has to satisfy before a space is built on it.
    pub fn validate(&self) -> Result<(), Error> {
        self.weighting.validate()
    }
}

/// One posting. Eight bytes. The weight is the value as given.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub(crate) struct Posting {
    pub(crate) id: u32,
    pub(crate) weight: f32,
}

/// One dimension's postings, sorted by record id.
#[derive(Clone, Debug, Default)]
pub(crate) struct PostingList {
    pub(crate) postings: Vec<Posting>,
    /// Postings whose record has been removed and not yet rewritten out.
    /// Maintained under `Unlink::Lazy` alone.
    pub(crate) dead: u32,
}

/// Where one record's vector sits in the forward arena.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slot {
    pub(crate) start: u32,
    pub(crate) len: u32,
}

const NEVER_HELD: Slot = Slot {
    start: 0,
    len: u32::MAX,
};

impl Slot {
    #[inline]
    pub(crate) fn held(&self) -> bool {
        self.len != u32::MAX
    }
}

/// Heap bytes the structure holds, by capacity, split by part. The
/// allocator's own overhead sits outside it.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeapBytes {
    pub lists_postings: usize,
    pub lists_headers: usize,
    pub map: usize,
    pub forward: usize,
    pub records: usize,
    /// The length per record slot, four bytes each.
    pub lengths: usize,
    /// The dead set and the live set together.
    pub dead: usize,
}

impl HeapBytes {
    pub fn total(&self) -> usize {
        self.lists_postings
            + self.lists_headers
            + self.map
            + self.forward
            + self.records
            + self.lengths
            + self.dead
    }
}

/// The index.
pub struct PostingsIndex {
    pub(crate) config: SparseConfig,
    /// Dimension to slot in `lists`.
    pub(crate) slots_by_dim: HashMap<u32, u32>,
    pub(crate) lists: Vec<PostingList>,
    /// The forward arena. A removed record's span is stranded until `compact`.
    pub(crate) fwd_dims: Vec<u32>,
    pub(crate) fwd_values: Vec<f32>,
    /// By record id. Grows with the id counter, not the live count.
    pub(crate) records: Vec<Slot>,
    /// Each record's length, being the sum of its values, by record id and
    /// parallel to `records`. What a term frequency weighting normalises
    /// by, read once per posting the scan admits. Kept under every
    /// weighting, since it costs four bytes a slot and one sum an insert.
    pub(crate) lengths: Vec<f32>,
    /// Records removed and not yet compacted out.
    pub(crate) dead: Bitmap,
    /// Records held and not removed, being what `holds` answers, as a set a
    /// filter's bitmap can be intersected with in a walk over words.
    pub(crate) live_set: Bitmap,
    pub(crate) dead_records: usize,
    pub(crate) live: usize,
    pub(crate) live_nnz: usize,
    /// The sum of every live record's length, so the mean is a division.
    /// Recomputed from the records at compaction, which bounds the drift a
    /// running sum of single-precision lengths accumulates.
    pub(crate) live_length: f64,
    /// Postings in every list, live and dead.
    pub(crate) postings_total: usize,
    pub(crate) dead_postings: usize,
    /// What a posting visit, a rejected posting and a merged element cost on
    /// this machine, timed on this index or taken from the floor.
    pub(crate) units: UnitCosts,
}

impl PostingsIndex {
    /// An empty index. Its unit costs are the compiled-in floor until it is
    /// large enough to time; see [`PostingsIndex::calibrate`].
    pub fn new(config: SparseConfig) -> Self {
        PostingsIndex {
            config,
            slots_by_dim: HashMap::new(),
            lists: Vec::new(),
            fwd_dims: Vec::new(),
            fwd_values: Vec::new(),
            records: Vec::new(),
            lengths: Vec::new(),
            dead: Bitmap::default(),
            live_set: Bitmap::default(),
            dead_records: 0,
            live: 0,
            live_nnz: 0,
            live_length: 0.0,
            postings_total: 0,
            dead_postings: 0,
            units: UnitCosts::FLOOR,
        }
    }

    /// An empty index with the forward arena reserved for `records` records
    /// of `nnz_per_record` nonzeros, so the arena is not doubled into being.
    pub fn with_capacity(config: SparseConfig, records: usize, nnz_per_record: usize) -> Self {
        let mut index = Self::new(config);
        let nnz = records.saturating_mul(nnz_per_record);
        index.fwd_dims.reserve(nnz);
        index.fwd_values.reserve(nnz);
        index.records.reserve(records.saturating_add(1));
        index.lengths.reserve(records.saturating_add(1));
        index
    }

    pub fn config(&self) -> &SparseConfig {
        &self.config
    }

    /// Mean length of the live records, being the mean sum of values. Zero
    /// for an empty index.
    pub fn mean_length(&self) -> f64 {
        if self.live == 0 {
            0.0
        } else {
            self.live_length / self.live as f64
        }
    }

    /// The length of one live record, being the sum of its values.
    pub fn length_of(&self, id: RecordId) -> Option<f32> {
        self.slot_of(id).map(|_| self.lengths[id.slot()])
    }

    /// The live set, being every record `holds` answers true for.
    pub fn live_set(&self) -> &Bitmap {
        &self.live_set
    }

    /// Document frequency of a dimension, being the live postings on its
    /// list. Inherent rather than on the trait, because a corpus statistic is
    /// this crate's own.
    pub fn df(&self, dim: u32) -> usize {
        self.slots_by_dim.get(&dim).map_or(0, |&s| {
            let list = &self.lists[s as usize];
            list.postings.len() - list.dead as usize
        })
    }

    pub fn distinct_dims(&self) -> usize {
        self.lists.len()
    }

    /// The largest dimension any list is kept for, or `None` on an index
    /// that has seen no posting. What a text layer's dictionary is checked
    /// against, since every dimension of a text space is a term id the
    /// dictionary issued.
    pub fn max_dim(&self) -> Option<u32> {
        self.slots_by_dim.keys().copied().max()
    }

    pub fn postings_total(&self) -> usize {
        self.postings_total
    }

    pub fn dead_postings(&self) -> usize {
        self.dead_postings
    }

    pub fn dead_records(&self) -> usize {
        self.dead_records
    }

    /// Mean live nonzeros per live record.
    pub fn mean_nnz(&self) -> f64 {
        if self.live == 0 {
            0.0
        } else {
            self.live_nnz as f64 / self.live as f64
        }
    }

    /// The unit costs the index prices a search with.
    pub fn unit_costs(&self) -> UnitCosts {
        self.units
    }

    /// Slots the record table holds, being one past the largest id ever
    /// inserted. The scratch accumulator is sized to it.
    pub(crate) fn slots(&self) -> usize {
        self.records.len()
    }

    /// Heap bytes the structure holds, by capacity, split by part.
    pub fn heap_bytes(&self) -> HeapBytes {
        let lists_postings: usize = self
            .lists
            .iter()
            .map(|l| l.postings.capacity() * std::mem::size_of::<Posting>())
            .sum();
        let lists_headers = self.lists.capacity() * std::mem::size_of::<PostingList>();
        // A std HashMap<u32, u32> stores (K, V) pairs plus one control byte
        // per bucket, at a load factor of 7/8.
        let map = self.slots_by_dim.capacity() * (std::mem::size_of::<(u32, u32)>() + 1);
        let forward = self.fwd_dims.capacity() * 4 + self.fwd_values.capacity() * 4;
        let records = self.records.capacity() * std::mem::size_of::<Slot>();
        let lengths = self.lengths.capacity() * 4;
        let dead = self.dead.heap_bytes() + self.live_set.heap_bytes();
        HeapBytes {
            lists_postings,
            lists_headers,
            map,
            forward,
            records,
            lengths,
            dead,
        }
    }

    /// Postings a scan for `query` would visit, dead ones included.
    pub fn scan_postings(&self, query: SparseRef<'_>) -> usize {
        query
            .dims
            .iter()
            .filter_map(|d| self.slots_by_dim.get(d))
            .map(|&s| self.lists[s as usize].postings.len())
            .sum()
    }

    pub(crate) fn slot_of(&self, id: RecordId) -> Option<Slot> {
        let slot = *self.records.get(id.slot())?;
        if slot.held() && !self.dead.contains(id.slot()) {
            Some(slot)
        } else {
            None
        }
    }

    pub(crate) fn forward(&self, slot: Slot) -> SparseRef<'_> {
        let (s, e) = (slot.start as usize, (slot.start + slot.len) as usize);
        SparseRef {
            dims: &self.fwd_dims[s..e],
            values: &self.fwd_values[s..e],
        }
    }

    /// Rewrite one list without its dead postings.
    fn compact_list(&mut self, slot: u32) {
        let dead = &self.dead;
        let list = &mut self.lists[slot as usize];
        let before = list.postings.len();
        list.postings.retain(|p| !dead.contains(p.id as usize));
        let removed = before - list.postings.len();
        list.dead = 0;
        self.postings_total -= removed;
        self.dead_postings -= removed;
        trace!(
            target: LOG_TARGET,
            operation = "list_rewrite",
            list = slot,
            removed = removed,
            remaining = list.postings.len(),
            "Rewrote a posting list without its dead postings"
        );
    }

    /// Rewrite every list and the forward arena without the dead records,
    /// keeping every live record's id, and return every buffer's spare
    /// capacity to the allocator.
    ///
    /// What the collection's `compact` calls for a sparse space. A third of
    /// the memory a built index holds is `Vec` slack, which the shrink
    /// reclaims. The unit costs are timed again afterwards, since the lists
    /// the scan reads have just changed shape.
    pub fn compact(&mut self) {
        let postings_before = self.postings_total;
        let dead_before = self.dead_records;
        for slot in 0..self.lists.len() as u32 {
            if self.lists[slot as usize].dead > 0 {
                self.compact_list(slot);
            }
            self.lists[slot as usize].postings.shrink_to_fit();
        }
        // The forward arena, rebuilt in id order, and the length total
        // summed again from the records that survive.
        let mut dims = Vec::with_capacity(self.live_nnz);
        let mut values = Vec::with_capacity(self.live_nnz);
        let mut live_length = 0f64;
        for (id, slot) in self.records.iter_mut().enumerate() {
            if !slot.held() {
                continue;
            }
            if self.dead.contains(id) {
                *slot = NEVER_HELD;
                self.lengths[id] = 0.0;
                continue;
            }
            let (s, e) = (slot.start as usize, (slot.start + slot.len) as usize);
            let start = dims.len() as u32;
            dims.extend_from_slice(&self.fwd_dims[s..e]);
            values.extend_from_slice(&self.fwd_values[s..e]);
            slot.start = start;
            live_length += self.lengths[id] as f64;
        }
        self.fwd_dims = dims;
        self.fwd_values = values;
        self.live_length = live_length;
        self.dead = Bitmap::default();
        self.dead_records = 0;
        self.lists.shrink_to_fit();
        self.records.shrink_to_fit();
        self.lengths.shrink_to_fit();
        self.slots_by_dim.shrink_to_fit();
        debug_assert_eq!(self.dead_postings, 0);
        debug!(
            target: LOG_TARGET,
            operation = "compact",
            postings_before = postings_before,
            postings_after = self.postings_total,
            dead_records = dead_before,
            live_records = self.live,
            "Compacted the sparse index"
        );
        self.calibrate();
    }

    /// Insert one record, with the check every caller wants.
    pub(crate) fn insert_record(
        &mut self,
        id: RecordId,
        vector: SparseRef<'_>,
    ) -> Result<(), Error> {
        vector.validate()?;
        if self.config.weighting.reads_term_frequency() {
            if let Some((index, &value)) = vector
                .values
                .iter()
                .enumerate()
                .find(|(_, value)| **value <= 0.0)
            {
                return Err(Error::SparseValueNotPositive { index, value });
            }
        }
        let slot_index = id.slot();
        if slot_index >= self.records.len() {
            self.records.resize(slot_index + 1, NEVER_HELD);
            self.lengths.resize(slot_index + 1, 0.0);
        }
        if self.records[slot_index].held() && !self.dead.contains(slot_index) {
            return Err(Error::RecordAlreadyHeld { id: id.0 });
        }
        if self.dead.contains(slot_index) {
            // A removed id being re-inserted. The engine never does this,
            // since it never reuses an id, but the structure stays correct if
            // a caller does: the old span and its postings are already
            // counted dead, and the record starts again.
            self.dead.remove(slot_index);
            self.dead_records -= 1;
        }
        let nnz = vector.dims.len();
        let start = self.fwd_dims.len() as u32;
        self.fwd_dims.extend_from_slice(vector.dims);
        self.fwd_values.extend_from_slice(vector.values);
        self.records[slot_index] = Slot {
            start,
            len: nnz as u32,
        };
        let length = vector.values.iter().map(|&v| v as f64).sum::<f64>() as f32;
        self.lengths[slot_index] = length;
        self.live_length += length as f64;
        self.live_set.insert(slot_index);

        for (&d, &w) in vector.dims.iter().zip(vector.values) {
            let slot = match self.slots_by_dim.get(&d) {
                Some(&s) => s,
                None => {
                    let s = self.lists.len() as u32;
                    self.lists.push(PostingList::default());
                    self.slots_by_dim.insert(d, s);
                    s
                }
            };
            let list = &mut self.lists[slot as usize].postings;
            match list.last() {
                Some(last) if last.id >= id.0 => {
                    // The engine never takes this path. It exists so the
                    // sorted-by-id property is the structure's and not the
                    // caller's.
                    match list.binary_search_by_key(&id.0, |p| p.id) {
                        Ok(at) => {
                            list[at] = Posting {
                                id: id.0,
                                weight: w,
                            }
                        }
                        Err(at) => list.insert(
                            at,
                            Posting {
                                id: id.0,
                                weight: w,
                            },
                        ),
                    }
                }
                _ => list.push(Posting {
                    id: id.0,
                    weight: w,
                }),
            }
        }
        self.live += 1;
        self.live_nnz += nnz;
        self.postings_total += nnz;
        Ok(())
    }

    /// The whole of a removal, under the configured policy.
    fn remove_record(&mut self, id: RecordId) -> Result<(), Error> {
        let Some(slot) = self.slot_of(id) else {
            return Err(Error::RecordNotHeld { id: id.0 });
        };
        self.dead.insert(id.slot());
        self.live_set.remove(id.slot());
        self.dead_records += 1;
        self.live -= 1;
        self.live_nnz -= slot.len as usize;
        self.live_length -= self.lengths[id.slot()] as f64;
        let (s, e) = (slot.start as usize, (slot.start + slot.len) as usize);
        match self.config.unlink {
            Unlink::Strand => {
                // The posting stays and is counted on its list, so `df`
                // reports live postings under every policy. What stranding
                // saves against the lazy policy is the rewrite alone.
                self.dead_postings += slot.len as usize;
                for i in s..e {
                    let d = self.fwd_dims[i];
                    let list_slot = self.slots_by_dim[&d];
                    self.lists[list_slot as usize].dead += 1;
                }
            }
            Unlink::Lazy => {
                self.dead_postings += slot.len as usize;
                let threshold = self.config.lazy_threshold_percent as usize;
                let mut to_compact: Vec<u32> = Vec::new();
                for i in s..e {
                    let d = self.fwd_dims[i];
                    let list_slot = self.slots_by_dim[&d];
                    let list = &mut self.lists[list_slot as usize];
                    list.dead += 1;
                    if list.dead as usize * 100 > list.postings.len() * threshold {
                        to_compact.push(list_slot);
                    }
                }
                for list_slot in to_compact {
                    self.compact_list(list_slot);
                }
            }
            Unlink::Eager => {
                for i in s..e {
                    let d = self.fwd_dims[i];
                    let list_slot = self.slots_by_dim[&d];
                    let list = &mut self.lists[list_slot as usize].postings;
                    if let Ok(pos) = list.binary_search_by_key(&id.0, |p| p.id) {
                        list.remove(pos);
                        self.postings_total -= 1;
                    }
                }
            }
        }
        Ok(())
    }
}

impl VectorIndex<Sparse> for PostingsIndex {
    fn len(&self) -> usize {
        self.live
    }

    fn holds(&self, id: RecordId) -> bool {
        self.slot_of(id).is_some()
    }

    /// Nothing to plan. A postings insert reads nothing from the index before
    /// it writes, so the whole of it runs under the write guard, and at five
    /// to fourteen microseconds that is a small fraction of a graph insert on
    /// the same record.
    ///
    /// Each time the live count doubles past the smallest index worth
    /// timing, the unit costs are timed again, so an index built by
    /// insertion prices its searches on what it holds rather than on the
    /// floor it started from. A restore replays through `insert_record`
    /// instead and times itself once at the end.
    fn insert(
        &mut self,
        id: RecordId,
        vector: SparseRef<'_>,
        _prepared: Prepared,
    ) -> Result<(), Error> {
        self.insert_record(id, vector)?;
        if self.live.is_power_of_two() && self.live >= crate::calibrate::CALIBRATION_MIN_RECORDS {
            self.calibrate();
        }
        Ok(())
    }

    fn remove(&mut self, id: RecordId) -> Result<(), Error> {
        self.remove_record(id)
    }

    /// Dead postings, which is what a compaction reclaims here.
    fn stranded(&self) -> usize {
        self.dead_postings
    }

    fn vector(&self, id: RecordId) -> Option<SparseRef<'_>> {
        self.slot_of(id).map(|slot| self.forward(slot))
    }

    fn recover(&self, id: RecordId) -> Option<SparseVector> {
        self.vector(id).map(|v| SparseVector {
            dims: v.dims.to_vec(),
            values: v.values.to_vec(),
        })
    }

    fn search(
        &self,
        query: SparseRef<'_>,
        k: usize,
        admit: &dyn Admit,
        budget: &Budget,
    ) -> Result<Hits, Error> {
        // `ef`, `fetch` and `rerank` name nothing a postings scan has. The
        // two knobs read are the boundary tie rule and the corpus the term
        // weighting counts over.
        self.search_scoped(
            Mode::Auto,
            query,
            k,
            admit,
            budget.boundary_ties,
            budget.idf,
        )
    }

    fn cost(&self, query: SparseRef<'_>, k: usize, admitted: Option<&Selectivity>) -> Cost {
        let _ = k;
        let scan = self.scan_postings(query);
        let work_ns = match admitted {
            Some(sel) => {
                // The admit set a planner describes is one the collection
                // builds, and every one of those is a bitmap or a sorted
                // list, so the two arms the search would choose between are
                // the bitmap scan and the enumerate-driven path.
                let (scan_ns, enumerate_ns) =
                    self.arm_costs(scan, sel.expected as usize, true, query.nnz());
                if self.config.weighting.reads_term_frequency() {
                    // A term weighting under an admit set first counts the
                    // query's postings under it, which is the scan's
                    // predicate loop without its accumulate on the bitmap
                    // arm and a second merge pass on the enumerate arm.
                    let (walk_ns, _) = self.arm_costs(scan, 0, true, query.nnz());
                    let frac = (sel.expected as f64 / self.live.max(1) as f64).min(1.0);
                    let walk_ns = walk_ns
                        + scan as f64 * 2.0 * frac * (1.0 - frac) * self.units.mispredict_ns;
                    (scan_ns + walk_ns).min(2.0 * enumerate_ns)
                } else {
                    scan_ns.min(enumerate_ns)
                }
            }
            None => scan as f64 * self.units.posting_ns,
        };
        Cost {
            work_ns,
            exact: true,
        }
    }

    fn corpus_stats(&self, dims: &[u32], admit: &dyn Admit) -> Option<CorpusStats> {
        self.stats_under(dims, admit)
    }
}

// ---------------------------------------------------------------------------
// Persistence. One artefact, live records only, ids kept.
// ---------------------------------------------------------------------------

impl Persist for PostingsIndex {
    fn write(&self, prefix: &str, dir: &Path, ledger: &mut dyn Ledger) -> Result<(), Error> {
        crate::persist::write(self, prefix, dir, ledger)
    }

    fn artefact_names(&self, prefix: &str) -> Vec<String> {
        vec![crate::persist::artefact_name(prefix)]
    }
}

impl Restore for PostingsIndex {
    type Config = SparseConfig;

    fn restore(
        config: &SparseConfig,
        prefix: &str,
        dir: &Path,
        inventory: &dyn Inventory,
        bounds: &zeusdb_vector_core::Bounds,
    ) -> Result<Self, Error> {
        crate::persist::restore(config, prefix, dir, inventory, bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeusdb_vector_core::Candidates;

    fn sparse(dims: &[u32], values: &[f32]) -> SparseVector {
        SparseVector {
            dims: dims.to_vec(),
            values: values.to_vec(),
        }
    }

    /// An id already held is refused, an id never held cannot be removed,
    /// and the live count and the stranded count follow every write.
    #[test]
    fn held_ids_are_refused_and_unknown_ids_cannot_be_removed() {
        let mut index = PostingsIndex::new(SparseConfig::default());
        let a = sparse(&[1, 4, 9], &[1.0, 2.0, 3.0]);
        index
            .insert(RecordId(1), a.as_ref(), Prepared::none())
            .unwrap();
        assert!(matches!(
            index.insert(RecordId(1), a.as_ref(), Prepared::none()),
            Err(Error::RecordAlreadyHeld { id: 1 })
        ));
        assert!(matches!(
            index.remove(RecordId(2)),
            Err(Error::RecordNotHeld { id: 2 })
        ));
        assert!(index.holds(RecordId(1)));
        assert!(!index.holds(RecordId(2)));
        assert_eq!(index.len(), 1);
        assert_eq!(index.stranded(), 0);

        index.remove(RecordId(1)).unwrap();
        assert!(!index.holds(RecordId(1)));
        assert_eq!(index.len(), 0);
        // Under the lazy policy a single record's three postings sit on three
        // lists of one posting each, so every list crosses the threshold at
        // once and is rewritten on the spot.
        assert_eq!(index.stranded(), 0);
        assert!(matches!(
            index.remove(RecordId(1)),
            Err(Error::RecordNotHeld { id: 1 })
        ));
        assert!(index.vector(RecordId(1)).is_none());
    }

    /// Stranding leaves the postings and counts them, and a compaction takes
    /// them back and keeps every live record's id and vector.
    #[test]
    fn stranding_counts_dead_postings_and_compaction_reclaims_them() {
        let mut index = PostingsIndex::new(SparseConfig {
            unlink: Unlink::Strand,
            ..SparseConfig::default()
        });
        for id in 1..=10u32 {
            let v = sparse(&[1, id + 10], &[1.0, id as f32]);
            index
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        for id in [2u32, 5, 9] {
            index.remove(RecordId(id)).unwrap();
        }
        assert_eq!(index.len(), 7);
        assert_eq!(index.stranded(), 6);
        assert_eq!(index.postings_total(), 20);
        assert_eq!(index.df(1), 7);

        index.compact();
        assert_eq!(index.stranded(), 0);
        assert_eq!(index.postings_total(), 14);
        assert_eq!(index.dead_records(), 0);
        for id in 1..=10u32 {
            let removed = [2u32, 5, 9].contains(&id);
            assert_eq!(index.holds(RecordId(id)), !removed);
            if !removed {
                let v = index.recover(RecordId(id)).unwrap();
                assert_eq!(v, sparse(&[1, id + 10], &[1.0, id as f32]));
            }
        }
        // Search still answers from the compacted lists.
        let q = sparse(&[1, 13], &[1.0, 1.0]);
        let hits = index
            .search(q.as_ref(), 3, &Candidates::All, &Budget::default())
            .unwrap();
        assert!(hits.exact);
        assert_eq!(hits.items[0].id, RecordId(3));
        assert_eq!(hits.items[0].score, 4.0);
    }

    /// The list stays sorted when an id arrives out of order, which the
    /// engine never does and the structure still handles.
    #[test]
    fn an_out_of_order_id_keeps_the_list_sorted() {
        let mut index = PostingsIndex::new(SparseConfig::default());
        for id in [5u32, 2, 9, 1] {
            let v = sparse(&[7], &[id as f32]);
            index
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        let slot = index.slots_by_dim[&7] as usize;
        let ids: Vec<u32> = index.lists[slot].postings.iter().map(|p| p.id).collect();
        assert_eq!(ids, vec![1, 2, 5, 9]);
    }

    /// The cost is the cheaper of the two arms and grows with the query's
    /// lists.
    #[test]
    fn cost_takes_the_cheaper_arm_and_reads_the_query() {
        let mut index = PostingsIndex::new(SparseConfig::default());
        for id in 1..=2000u32 {
            let v = sparse(&[1, 2 + id % 50], &[1.0, 1.0]);
            index
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        let common = sparse(&[1], &[1.0]);
        let rare = sparse(&[3], &[1.0]);
        let whole = index.cost(common.as_ref(), 10, None);
        assert!(whole.exact);
        assert!(whole.work_ns > index.cost(rare.as_ref(), 10, None).work_ns);
        // Ten admitted records are cheaper to score from the arena than a
        // scan of the common list.
        let narrow = index.cost(common.as_ref(), 10, Some(&Selectivity::exact(10)));
        assert!(narrow.work_ns < whole.work_ns);
    }
}
