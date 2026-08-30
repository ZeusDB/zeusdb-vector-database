//! The unit costs the index prices a search with, timed on the index itself.
//!
//! A search's cost is a count of posting visits, or a count of records
//! scored from the arena, multiplied by what one of those costs on this
//! machine in this build. Both figures move with the machine and with the
//! build, so they are measured at open rather than tabulated, and never
//! persisted. An index too small to time takes the compiled-in floor.

use std::time::Instant;

use tracing::debug;
use zeusdb_vector_core::{Bitmap, SparseRef, SparseVector};

use crate::index::PostingsIndex;
use crate::search::Mode;
use crate::LOG_TARGET;

/// What one unit of each kind of work costs, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitCosts {
    /// One posting visited and accumulated by the scan.
    pub posting_ns: f64,
    /// One posting the scan tests against a bitmap and rejects, where the
    /// test's outcome is the same as the last one's.
    pub reject_ns: f64,
    /// What a bitmap test costs on top of that when its outcome cannot be
    /// predicted from the last one's, which is the case in the middle of
    /// the selectivity range. A test admitting half the postings at random
    /// costs several times a test admitting all or none, and a scan priced
    /// without this term came out four times too cheap at half admitted.
    pub mispredict_ns: f64,
    /// One element merged when a record is scored from the arena.
    pub merge_ns: f64,
    /// What scoring one admitted record costs beyond its merge, being the
    /// walk of the admit set, the slot lookup and the arena read.
    pub record_ns: f64,
    /// Whether the figures were timed on this index or are the floor.
    pub measured: bool,
}

impl UnitCosts {
    /// The compiled-in floor, for an index too small to time.
    ///
    /// The figures are what each unit measured at 50,000 records on two
    /// synthetic regimes on one developer machine, rounded up, so an untimed
    /// index errs towards the scan being dear and the enumerate path being
    /// chosen. A timed index replaces them.
    pub const FLOOR: UnitCosts = UnitCosts {
        posting_ns: 3.5,
        reject_ns: 1.0,
        mispredict_ns: 8.0,
        merge_ns: 3.0,
        record_ns: 80.0,
        measured: false,
    };
}

/// Postings the index holds before it is timed rather than given the floor.
///
/// Below this a round is too short to read against the timer's resolution,
/// and the floor is close enough for an index this size to choose its path
/// by.
pub(crate) const CALIBRATION_MIN_POSTINGS: usize = 20_000;

/// Live records at which an index built by insertion first times itself,
/// and at every doubling after. Five hundred records at the regimes this
/// was measured on hold the postings above, and a timing run at that size
/// is a fraction of a millisecond.
pub(crate) const CALIBRATION_MIN_RECORDS: usize = 512;

/// Records whose dimensions become the timing queries.
const SAMPLE_RECORDS: usize = 8;

/// Dimensions each timing query carries.
///
/// A query is a handful of terms where a record is dozens, and the cost of
/// a posting visit depends on the shape of the scan that visits it: a scan
/// over forty lists amortises its fixed costs and runs longer sequential
/// stretches than one over eight, and measured per posting it came out at
/// under half the figure a real query pays. Eight is the query length the
/// regimes this was measured on centre around.
const QUERY_DIMS: usize = 8;

/// Rounds each measurement runs, of which the median is kept.
const ROUNDS: usize = 5;

/// Records the admit set holds when the enumerate-driven path is timed.
const ENUMERATE_SAMPLE: usize = 256;

impl PostingsIndex {
    /// Time the three unit costs on this index and keep them.
    ///
    /// Called at open, by `compact`, each time the live count doubles past
    /// [`CALIBRATION_MIN_RECORDS`], and by a caller that has just filled an
    /// index in bulk and wants its searches priced on what it now holds.
    /// Runs about twenty milliseconds at 50,000 records and nothing at all
    /// below [`CALIBRATION_MIN_POSTINGS`], where the floor is kept.
    pub fn calibrate(&mut self) {
        if self.postings_total < CALIBRATION_MIN_POSTINGS || self.live < 2 {
            self.units = UnitCosts::FLOOR;
            return;
        }
        // Queries drawn from stored records spaced across the table, so they
        // name lists the index really holds at the lengths it holds them,
        // cut to a query's length by taking every nth dimension of the
        // record, so a query carries the record's spread of common and rare
        // terms rather than its first few.
        let mut queries: Vec<SparseVector> = Vec::with_capacity(SAMPLE_RECORDS);
        let step = (self.records.len() / SAMPLE_RECORDS).max(1);
        let mut at = 0usize;
        while queries.len() < SAMPLE_RECORDS && at < self.records.len() {
            if let Some(v) = self.slot_of(zeusdb_vector_core::RecordId::from_slot(at)) {
                let v = self.forward(v);
                let stride = (v.dims.len() / QUERY_DIMS).max(1);
                let dims: Vec<u32> = v
                    .dims
                    .iter()
                    .copied()
                    .step_by(stride)
                    .take(QUERY_DIMS)
                    .collect();
                queries.push(SparseVector {
                    values: vec![1.0; dims.len()],
                    dims,
                });
            }
            at += step;
        }
        if queries.len() < 2 {
            self.units = UnitCosts::FLOOR;
            return;
        }
        let postings: usize = queries.iter().map(|q| self.scan_postings(q.as_ref())).sum();
        if postings == 0 {
            self.units = UnitCosts::FLOOR;
            return;
        }
        // An empty set whose words reach every slot, so a rejected posting
        // costs the word read a real bitmap costs and not the bounds check
        // an unsized set answers with.
        let nothing = Bitmap::with_slots(self.records.len());
        let all = zeusdb_vector_core::Candidates::All;
        // A set admitting half the slots at random, so the bit test's
        // outcome cannot be predicted from the last one's, and a set of a
        // few hundred slots at random for the enumerate-driven path.
        let (half, few) = {
            let mut half = Bitmap::with_slots(self.records.len());
            let mut few = Bitmap::default();
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            let stride = (self.records.len() / ENUMERATE_SAMPLE).max(1);
            for slot in 0..self.records.len() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state & 1 == 1 {
                    half.insert(slot);
                }
                if slot.is_multiple_of(stride) && few.count() < ENUMERATE_SAMPLE {
                    few.insert(slot);
                }
            }
            (half, few)
        };
        let few_count = few.count().max(1);

        // The scan with no predicate, per posting.
        let posting_ns = median(ROUNDS, || {
            for q in &queries {
                let _ = std::hint::black_box(self.search_mode(
                    Mode::Floor,
                    q.as_ref(),
                    10,
                    &all,
                    false,
                ));
            }
            postings
        });
        // The bitmap scan with an empty bitmap, so every posting is tested
        // and rejected, per posting.
        let reject_ns = median(ROUNDS, || {
            for q in &queries {
                let _ = std::hint::black_box(self.search_mode(
                    Mode::BitmapPerPosting,
                    q.as_ref(),
                    10,
                    &nothing,
                    false,
                ));
            }
            postings
        });
        // The merge of every sampled record against every sampled query, per
        // element merged, which is what the enumerate-driven path pays per
        // admitted record.
        let records: Vec<SparseRef<'_>> = (0..self.records.len())
            .step_by(step)
            .filter_map(|at| self.slot_of(zeusdb_vector_core::RecordId::from_slot(at)))
            .take(SAMPLE_RECORDS)
            .map(|slot| self.forward(slot))
            .collect();
        let merge_elements: usize = records
            .iter()
            .flat_map(|a| queries.iter().map(move |b| a.dims.len() + b.dims.len()))
            .sum();
        let merge_ns = median(ROUNDS, || {
            let mut acc = 0f32;
            for a in &records {
                for b in &queries {
                    acc += a.dot(b.as_ref());
                }
            }
            std::hint::black_box(acc);
            merge_elements
        });
        // The bitmap scan under the random half, per posting, less what the
        // admitted and rejected halves would cost were every test
        // predicted. Half the tests mispredict there, so the remainder is
        // half the misprediction cost.
        let half_ns = median(ROUNDS, || {
            for q in &queries {
                let _ = std::hint::black_box(self.search_mode(
                    Mode::BitmapPerPosting,
                    q.as_ref(),
                    10,
                    &half,
                    false,
                ));
            }
            postings
        });
        let mispredict_ns = ((half_ns - 0.5 * posting_ns - 0.5 * reject_ns) * 2.0).max(0.0);
        // The enumerate-driven path over the few, per record, less the
        // merge each record costs at the sampled query length.
        let mean_nnz = self.mean_nnz();
        let mean_query =
            queries.iter().map(|q| q.dims.len()).sum::<usize>() as f64 / queries.len() as f64;
        let enumerate_ns = median(ROUNDS, || {
            for q in &queries {
                let _ = std::hint::black_box(self.search_mode(
                    Mode::Enumerate,
                    q.as_ref(),
                    10,
                    &few,
                    false,
                ));
            }
            few_count * queries.len()
        });
        let record_ns = (enumerate_ns - merge_ns * (mean_nnz + mean_query)).max(0.0);
        self.units = UnitCosts {
            posting_ns,
            reject_ns,
            mispredict_ns,
            merge_ns,
            record_ns,
            measured: true,
        };
        debug!(
            target: LOG_TARGET,
            operation = "calibrate",
            posting_ns = posting_ns,
            reject_ns = reject_ns,
            mispredict_ns = mispredict_ns,
            merge_ns = merge_ns,
            record_ns = record_ns,
            postings_timed = postings,
            "Timed the unit costs on the index"
        );
    }
}

/// Run `work` for `rounds` rounds and return the median nanoseconds per unit,
/// where each round reports the units it performed.
fn median<F: FnMut() -> usize>(rounds: usize, mut work: F) -> f64 {
    let mut samples: Vec<f64> = (0..rounds)
        .map(|_| {
            let started = Instant::now();
            let units = work();
            started.elapsed().as_secs_f64() * 1e9 / units.max(1) as f64
        })
        .collect();
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

/// The scan rule reads the query's lists, so an index timed on itself reports
/// a cost that follows them. Kept here so the trait's `cost` has one place
/// that says what its unit is.
#[allow(dead_code)]
pub(crate) fn describe(units: &UnitCosts) -> String {
    format!(
        "posting {:.2} ns, reject {:.2} ns, mispredict {:.2} ns, merge {:.2} ns, record {:.2} ns{}",
        units.posting_ns,
        units.reject_ns,
        units.mispredict_ns,
        units.merge_ns,
        units.record_ns,
        if units.measured { "" } else { " (floor)" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::SparseConfig;
    use zeusdb_vector_core::{Prepared, RecordId, VectorIndex};

    /// A small index keeps the floor, and a large one times itself and
    /// reports figures that are positive and finite.
    #[test]
    fn a_small_index_keeps_the_floor_and_a_large_one_times_itself() {
        let mut small = PostingsIndex::new(SparseConfig::default());
        for id in 1..=20u32 {
            let v = SparseVector {
                dims: vec![1, 2, 3],
                values: vec![1.0, 1.0, 1.0],
            };
            small
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        small.calibrate();
        assert_eq!(small.unit_costs(), UnitCosts::FLOOR);

        let mut large = PostingsIndex::new(SparseConfig::default());
        for id in 1..=2000u32 {
            let dims: Vec<u32> = (0..20).map(|j| (id + j * 7) % 300).collect();
            let mut dims: Vec<u32> = dims;
            dims.sort_unstable();
            dims.dedup();
            let values = vec![1.0; dims.len()];
            let v = SparseVector { dims, values };
            large
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        assert!(large.postings_total() >= CALIBRATION_MIN_POSTINGS);
        large.calibrate();
        let units = large.unit_costs();
        assert!(units.measured);
        assert!(units.posting_ns > 0.0 && units.posting_ns.is_finite());
        assert!(units.reject_ns > 0.0 && units.reject_ns.is_finite());
        assert!(units.mispredict_ns >= 0.0 && units.mispredict_ns.is_finite());
        assert!(units.merge_ns > 0.0 && units.merge_ns.is_finite());
        assert!(units.record_ns >= 0.0 && units.record_ns.is_finite());
        assert!(!describe(&units).contains("floor"));
    }
}
