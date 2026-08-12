use crate::distance::{CosineDist, L1Dist, L2Dist};
use chrono::Utc;
use hnsw_rs::api::AnnT; // This provides the file_dump method
use hnsw_rs::hnsw::Point;
use hnsw_rs::prelude::{Distance, FilterT, Hnsw};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
// Aliased because `std::sync::atomic::Ordering` already holds the bare name.
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// ✅ ENTERPRISE: Structured logging imports
use tracing::{debug, error, info, instrument, trace, warn};

// Import PQ module
use crate::pq::PQ;

/// Largest `expected_size` a caller may declare.
///
/// `expected_size` reaches `PointIndexation::new` in the vendored graph crate,
/// which reserves capacity for it across the 16 layers at creation. Since the
/// layer reservation was corrected the reservation is one `Arc` slot per
/// declared record, measured at 8.02 bytes per declared record and flat across
/// declarations of 10 million through 4 billion. This bound therefore caps the
/// creation-time reservation at 764 MB, which was measured rather than derived.
///
/// The bound exists because the reservation is not fallible. `Vec::with_capacity`
/// aborts the process on allocation failure rather than unwinding, so a
/// declaration too large for the machine cannot be turned into a Python
/// exception after the fact. A declared 20 billion asks for 155 GB in the layer
/// zero reservation alone and the process dies with no traceback. Making that
/// path fallible means `try_reserve` inside the vendored crate, which is not a
/// change this package makes.
///
/// One hundred million is far above anything this index holds. A real 100,000
/// record build at 768 dimensions measured 10,617 bytes per record, so a hundred
/// million records is roughly a terabyte of process memory for the data alone.
/// Declaring less than the truth is safe, because a layer that receives more
/// points than reserved grows through the ordinary `Vec::push` path, so capping
/// the declaration costs a caller nothing it can observe.
///
/// The bound is not a guarantee. A machine whose commit limit is below the
/// reservation can still abort at a declaration under it.
const MAX_EXPECTED_SIZE: usize = 100_000_000;

/// Multiple of `expected_size` at which an index warns that it has outgrown its
/// declaration. Fires once per index.
const EXPECTED_SIZE_OVERGROWTH_FACTOR: usize = 2;

/// Seed the training sample is shuffled with before it is used
///
/// The records the sample holds are fixed and cannot be sampled. Training fires
/// on the record that reaches `training_size`, so the index holds exactly
/// `training_size` records at that moment and every one of them is in the
/// sample. What can be drawn randomly is the order, and the order is what every
/// subset of the sample is taken by: the codebook sees the records in this
/// order, the calibration takes its queries by striding it, and the calibration
/// takes each fitting fraction as a prefix of it. Without the shuffle all three
/// are slices of insertion order, and a corpus that arrives in a meaningful
/// order makes a prefix measure something other than the whole. On ada-002
/// embeddings in DBpedia article order the first half of the sample measured a
/// fetch of 120 to 135 candidates where the second half measured 165 to 178 and
/// a random half measured 109 to 156, over three codebook draws.
///
/// It is a fixed seed rather than an entropy draw, so two builds over the same
/// records in the same order produce the same shuffle and the same calibration.
/// The k-means the codebook is fitted with is unseeded and remains the source
/// of run to run variation.
const TRAINING_SAMPLE_SEED: u64 = 0x5A_EE_5D_B0_5E_ED_57_01;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageMode {
    #[default]
    #[serde(rename = "quantized_only")]
    QuantizedOnly,

    #[serde(rename = "quantized_with_raw")]
    QuantizedWithRaw,
}

impl StorageMode {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s {
            "quantized_only" => Ok(StorageMode::QuantizedOnly),
            "quantized_with_raw" => Ok(StorageMode::QuantizedWithRaw),
            _ => Err(format!(
                "Invalid storage_mode: '{}'. Supported: quantized_only, quantized_with_raw",
                s
            )),
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            StorageMode::QuantizedOnly => "quantized_only",
            StorageMode::QuantizedWithRaw => "quantized_with_raw",
        }
    }
}

// Updated QuantizationConfig structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub subvectors: usize,
    pub bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,
    pub storage_mode: StorageMode,
}

/// The three terms of the default rerank fetch.
///
/// A rerank fetch has to be deep enough to contain the true neighbours in the
/// ADC ordering. What sets that depth is the number of records the codes
/// cannot tell apart from the query, and that count is a property of the data
/// rather than of the index. On clustered data it is the size of the query's
/// own cluster, because a 128x code resolves which cluster a record belongs to
/// and resolves very little inside it.
///
/// The 90th percentile depth of the deepest true neighbour, at dimension 768
/// and the default subvectors, 100 queries per cell, over seven cells that
/// vary the cluster count and the record count independently.
///
///   clusters   records   records per cluster   depth   share of corpus
///        500    25,000                    50      60            0.24%
///        200    25,000                   125     119            0.48%
///        100    20,000                   200     197            0.99%
///        500   100,000                   200     190            0.19%
///         50    25,000                   500     469            1.88%
///        200   100,000                   500     461            0.46%
///         50   100,000                 2,000   1,863            1.86%
///
/// The depth column tracks the records per cluster and not the record count,
/// across a fortyfold range and at roughly 0.93 times it. Two pairs of rows
/// make the point without arithmetic. 500 records to a cluster returns 469 at
/// 25,000 records and 461 at 100,000, being the same rank at four times the
/// corpus. 200 records to a cluster returns 197 and 190 at 20,000 and 100,000.
/// The depth is the cluster size in records and it is not a share of anything.
///
/// An earlier version of this comment claimed the depth held at 2 percent of
/// the corpus across eight sizes. That measurement varied the record count
/// while holding the generator at fifty clusters, which forced the cluster size
/// to be one fiftieth of the corpus, and one fiftieth is 2 percent. The claim
/// restated the generator rather than measuring the codes.
///
/// Two data models with no cluster structure to find. An anisotropic corpus
/// with a power law covariance spectrum and a multi-scale topic hierarchy
/// returns 713 at 25,000 records and 1,506 at 100,000. Uniform points on the
/// sphere, and 5,000 clusters over 25,000 records where a cluster holds five
/// records, both put the deepest true neighbour beyond the 12,500th candidate
/// on more than half of the queries.
///
/// The fetch is the largest of three terms, capped at the live record count.
///
/// `DEFAULT_RERANK_CORPUS_DIVISOR` is the corpus term. It is a bound rather
/// than a law, because how a corpus's indistinguishable groups grow with the
/// record count is a property of the corpus and no index measurement settles
/// it. Recall at 10 at 100,000 records, derived from the ADC ordering over 100
/// queries, at four data models where recall 0.99 is reachable at all. Every
/// column is at a training set of 1,000 vectors, and the shipped default of
/// 10,000 moves the last column by at most 0.008 at any of these fetches.
///
///   fetch     50 clusters  200 clusters  500 clusters  embedding-like
///     100         0.234        0.448         0.761          0.645
///     200         0.316        0.675         0.996          0.759
///     500         0.535        1.000         1.000          0.897
///   1,000         0.793        1.000         1.000          0.960
///   2,000         0.999        1.000         1.000          0.991
///
/// 2,000 candidates is what a divisor of 50 produces at that size and it is the
/// smallest value in the grid that clears 0.99 on every model. A constant
/// multiple of `top_k` in the range the field uses, being 10 to 40 candidates
/// at `top_k` 10, returns between 0.10 and 0.20 on the hardest of them, which
/// is why this term is corpus proportional where every other vector database
/// uses a multiple of the page.
///
/// # Where no fetch works
///
/// Once the group of records the codes cannot separate is smaller than
/// `top_k`, the true top ten span groups, the distances between groups differ
/// in the fourth decimal, and no fetch reaches them. At 5,000 clusters over
/// 25,000 records, being five records to a cluster, recall at 10 reaches 0.917
/// at a fetch of half the corpus. Uniform points on the sphere reach 0.859 at
/// the same fetch. The defaults return 0.473 and 0.191 on those two. That is
/// what a 128x code does to data with no resolvable structure and no rerank
/// value repairs it.
///
/// `DEFAULT_RERANK_MIN_CANDIDATES` is the floor, which holds the fetch up
/// where the corpus term is too small to reach the neighbours. It governs up
/// to `DEFAULT_RERANK_CORPUS_DIVISOR * DEFAULT_RERANK_MIN_CANDIDATES` records.
/// At 10,000 records on fifty clusters the fetch that reaches 0.99 is 195
/// candidates, and over ten independent codebook draws a fetch of 200 returns
/// a mean of 0.9930 with a worst case of 0.9905 where a fetch of 250 returns
/// 0.9997 with a worst case of 0.9990. On the embedding-like corpus at the same
/// size the fetch reaching 0.99 is 250.
///
/// It costs nothing anywhere the criteria can see. It changes only corpus
/// sizes below 12,500 records, where a reranked quantized search is already
/// faster than an unquantized one, and at dimension 768 it reads 0.82 times the
/// unquantized query time at 3,000 records and 0.91 at 5,000.
///
/// `DEFAULT_RERANK_PAGE_FACTOR` is the page term, because the hundredth true
/// neighbour sits deeper than the tenth. It is small, since the candidate
/// count that saturates recall is a property of the corpus and not of the page
/// size, and at `top_k` 100 on 10,000 records recall reaches 0.986 at a fetch
/// of 200 and 1.000 at 500.
///
/// On a calibrated index this term is a floor and nothing else. The page is
/// handled by the page exponent the calibration measures, and five times
/// `top_k` sits below the calibrated fetch at every corpus size measured, so it
/// never governs there. See `RERANK_CALIBRATION_PAGES`.
///
/// # Why the fetch is the whole cost, and why nothing narrows it
///
/// `Hnsw::search_filter` raises the traversal width to the number of
/// neighbours asked for at `hnsw.rs:1650`, and then cuts the result to
/// `knbn.min(ef)` at `hnsw.rs:1666`. Removing the first would not decouple the
/// two, because the second caps the page at `ef` regardless. The candidate
/// heap `search_layer` returns holds at most `ef` entries, so a fetch of F
/// candidates cannot be served by a traversal narrower than F. That is a
/// property of the algorithm rather than a choice, and it is why an
/// `ef_search` below the fetch is discarded.
///
/// A traversal wider than the fetch buys nothing. Quadrupling `ef_search` at a
/// fixed fetch moves recall at 10 by at most 0.008 and by nothing at all in
/// fourteen of twenty measured pairs, because the candidates a fetch returns
/// are limited by the ADC ordering rather than by the traversal. It costs
/// query time in every case.
///
/// A lower compression ratio mostly does not buy query time either. It puts
/// the true neighbours shallower, so the fetch shrinks, and it lengthens the
/// code, so each candidate costs more to score. Between 384x and 32x the two
/// cancel. Measured at dimension 768 and 100,000 records, the fetch that
/// reaches recall at 10 of 0.99 runs 1,995 candidates at 384x, 1,921 at 128x
/// and 960 at 32x, and the query costs 9.40, 13.08 and 9.34 ms. Only at 16x
/// does the fetch collapse, to 222 candidates and 4.32 ms, and that ratio
/// holds more memory than an unquantized index at 10,000 records and builds
/// five times slower at 100,000, so it is documented rather than defaulted to.
///
/// A staged fetch is not ruled out, and an earlier version of this comment said
/// it was. It needs a long thin tail of hard queries. On fifty clusters there
/// is none, the median and the 90th percentile depth reading 1,530 and 1,863 at
/// 100,000 records, and at 10,000 records the share of queries whose true top
/// ten sit inside the first F candidates runs 0.060 at F of 100, 0.355 at 150
/// and 0.935 at 200. On the embedding-like corpus at 100,000 records the same
/// two percentiles read 314 and 1,506, which is the tail a staged fetch wants.
/// Which of the two a real corpus resembles is not settled here.
///
/// # The cost, stated where a user reads it
///
/// The fetch grows with the corpus and the traversal grows with the fetch, so
/// a reranked quantized search costs time proportional to the record count
/// where an unquantized one costs time proportional to its logarithm. The two
/// cross once, and above the crossing this default is slower than an
/// unquantized search while holding roughly 60 percent of the memory. No value
/// of `subvectors` and no rerank expression that holds recall moves that
/// crossing. See the README table.
pub const DEFAULT_RERANK_CORPUS_DIVISOR: usize = 50;
pub const DEFAULT_RERANK_MIN_CANDIDATES: usize = 250;
pub const DEFAULT_RERANK_PAGE_FACTOR: usize = 5;

/// The calibrated rerank fetch, measured on the index's own data
///
/// The three terms above are a formula in the record count, and no formula in
/// the record count fits real data. At 100,000 records the fetch that reaches
/// mean recall at 10 of 0.99 is 494 candidates on OpenAI ada-002 embeddings,
/// 426 on SIFT descriptors and 5,143 on GloVe word vectors, being 0.49, 0.43
/// and 5.14 percent of the same corpus. The corpus term produces 2,000 for all
/// three, which is four times what two of them need and two fifths of what the
/// third needs.
///
/// Training completion is where the index can measure it instead. It holds
/// `training_size` raw vectors and a codebook fitted to them, so exact
/// distances over that sample are affordable and the ADC ordering the search
/// will use is already defined. `calibrate_rerank` measures the fetch that
/// reaches recall `RERANK_CALIBRATION_TARGET` on the sample, and the search
/// scales that measurement to the live record count.
///
/// # The measurement
///
/// The training sample is held in a seeded random order; see
/// `TRAINING_SAMPLE_SEED`. Queries are drawn from it by striding, and a subset
/// of it is its own prefix, so both are random draws over the sample rather
/// than slices of insertion order. A query is also a record of the corpus it is
/// being searched against, and it is its own exact nearest neighbour. It is
/// removed from both the true neighbour list and the ADC ordering, so the
/// measurement is leave one out and the self match contributes nothing.
///
/// True neighbours come from exact distances over the sample, computed with
/// the same `raw_distance_fn` a rerank rescores with. The depth of a true
/// neighbour is its rank in the ADC ordering over the sample. The statistic is
/// the `RERANK_CALIBRATION_TARGET` percentile of every true neighbour rank
/// pooled across queries, because mean recall at 10 over a fetch of F is the
/// share of pooled ranks at or below F, so that percentile is by construction
/// the fetch reaching that recall.
///
/// The percentile of the pooled ranks and the percentile of the per query
/// deepest neighbour are different statistics and the second is far larger.
/// On ada-002 embeddings at 100,000 records they read 494 and 2,060. Mean
/// recall is what the pooled statistic answers.
///
/// # Why the measurement does not transfer as a share
///
/// The depth grows with the record count, and how fast it grows is a property
/// of the corpus rather than of the codes. Measured on three real datasets at
/// three codebook draws each, holding the codebook fixed and taking prefixes
/// of a permuted corpus. The fetch reaching recall 0.99, and the same figure
/// as a share of the corpus.
///
///   records    ada-002        GloVe          SIFT
///    10,000    151, 1.51%     948,  9.48%    135, 1.35%
///    25,000    263, 1.05%   2,175,  8.70%    210, 0.84%
///    50,000    418, 0.84%   3,244,  6.49%    313, 0.63%
///   100,000    494, 0.49%   5,143,  5.14%    426, 0.43%
///
/// The share falls by 3.1, 1.8 and 3.2 times over a tenfold range in the
/// record count, so a share does not transfer. What transfers is a power law,
/// and fitting the exponent over that range returns 0.515, 0.734 and 0.499.
///
/// **That exponent is not a constant either.** Those three corpora are random
/// subsamples of a fixed source, so the number of groups the codes cannot
/// separate is fixed by the source and each group grows as a root of the
/// record count. A corpus whose group count is fixed instead grows each group
/// linearly. Two generators at dimension 256, 50 and 200 clusters at every
/// size, return exponents of 0.987 and 0.952 over the same tenfold range. One
/// constant cannot serve 0.499 and 0.987, and a constant chosen for the real
/// datasets loses recall on the generators. At 0.60 the fifty cluster corpus
/// returns 0.727 at 25,000 records where the corpus term returns 0.999.
///
/// # The exponent is measured, not assumed
///
/// The calibration measures the fetch at each fraction of the training sample
/// in `RERANK_CALIBRATION_FIT_FRACTIONS` and fits the exponent as the least
/// squares slope of the log fetch on the log record count. A doubling of the
/// records that doubles the fetch is an exponent of one.
///
/// A two point fit over the sample and its first half was measured first and
/// it is not good enough. Against the held out exponent over 10,000 to 100,000
/// records, mean error over three codebook draws and three subset draws, with
/// the standard deviation of the estimate beside it:
///
///   estimator                              ada-002   GloVe    SIFT
///   two points, insertion order prefix       +0.261  -0.186  -0.101
///   two points, random half                  +0.349  -0.152  -0.030
///   four fractions, least squares            +0.100  -0.177  -0.094
///
///   spread of the estimate, standard deviation
///   two points, random half                   0.094   0.073   0.078
///   four fractions, least squares             0.088   0.039   0.031
///
/// The two point fit reads 0.26 to 0.35 too steep on ada-002, which is the
/// corpus whose true exponent is shallowest, and it swings more widely on the
/// two corpora whose exponent it gets right. The information is in the short
/// fractions: a fit over the sample and its half spans a factor of two, and one
/// over a quarter to the whole spans a factor of four. Dropping the quarter
/// point returns the two point error, at +0.328 on ada-002.
///
/// The fit still reads low on the two corpora whose exponent is steep, because
/// a subset is small enough for the leave one out geometry to compress its
/// ranks. `RERANK_CALIBRATION_EXPONENT_BIAS` is that correction. The result is
/// clamped between `RERANK_CALIBRATION_EXPONENT_MIN` and
/// `RERANK_CALIBRATION_EXPONENT_MAX`, so it can never grow faster than linear
/// and never slower than the floor.
///
/// The floor was 0.60 and it is 0.40, because 0.60 sat above the true exponent
/// of two of the three real datasets and governed the fetch on both of them
/// once the fit stopped reading steep. Measured true exponents over 10,000 to
/// 100,000 records are 0.27 to 0.32 on ada-002, 0.48 to 0.58 on SIFT and 0.64
/// to 0.74 on GloVe, against 0.95 to 0.99 on the generators. The floor is a
/// guard against a fit that collapses, not a term that should govern.
///
/// A subset that measures a deeper fetch than the whole sample sits below the
/// size at which the codes resolve the data at all, which is relay 54's case
/// of a group smaller than the page. The fit carries no signal there, its
/// slope is not positive, and the exponent takes the maximum. That fires on
/// fifty clusters at a `training_size` of 1,000, where a quarter of the sample
/// holds five records to a cluster.
///
/// `RERANK_CALIBRATION_SAFETY` is 1.75 and it was 1.5. A fetch equal to the
/// measured percentile lands at recall 0.99 in expectation, so it lands below
/// it on about half of the draws, and the in sample measurement is taken with
/// queries drawn from the corpus where a caller's queries are not. The
/// multiplier covers both. It went up rather than down because the exponent fit
/// that replaced the two point one removed the surplus that used to hide inside
/// the extrapolation: at 1.5 the fetch on ada-002 reads 665 candidates where
/// 750 are needed.
///
/// # What the whole rule delivers
///
/// Mean recall at 10 measured on built indexes at 100,000 records, one query at
/// a time through the ordinary search path. The requirement is the smallest
/// fetch on the same index reaching 0.99, read off a sweep of the explicit
/// `rerank` argument over that index.
///
///   corpus     calibrated   recall   requirement   corpus term   its recall
///   ada-002           776   0.9905           750         2,000       0.9954
///   GloVe           7,744   0.9962         4,500         2,000       0.9723
///   SIFT              596   0.9941           450         2,000       1.0000
///
/// A requirement has to be read off the index the fetch will run on, and not
/// off the codes. Ordering the codes exactly over the same corpus asks for the
/// same fetch on GloVe and SIFT, at 1.01 and 0.98 times these, and three fifths
/// of it on ada-002, at 0.63. A fetch is served by a traversal of the graph over
/// the codes and a traversal of width F does not return the F nearest by code.
///
/// A generator holding a fixed number of clusters at every size is the case
/// that breaks a constant exponent, and it is checked rather than assumed. At
/// fifty clusters and dimension 256 the calibration reads recall 0.9972 at
/// 25,000 records and 0.9972 at 100,000, where the corpus term reads 0.9864 and
/// 0.9924. At a `training_size` of 1,000 on the same corpus neither reaches the
/// target, the calibration reading 0.968 and 0.942 against the corpus term's
/// 0.744 and 0.794, because a codebook fitted to twenty records per cluster
/// does not order those clusters and no fetch repairs that.
///
/// # The page
///
/// The measurement above is taken at one page, being `RERANK_CALIBRATION_TOP_K`,
/// and a deeper page needs a deeper fetch. How much deeper is measured at
/// training as well and it is fitted as an exponent of the page in the same way
/// the record term is fitted as an exponent of the record count. See
/// `RERANK_CALIBRATION_PAGES`.
///
/// # The floor and the cap
///
/// The floor is `DEFAULT_RERANK_MIN_CANDIDATES` and the page term, unchanged,
/// so a calibration that measures a depth of two cannot produce a fetch of
/// two.
///
/// `RERANK_CALIBRATION_CAP_DIVISOR` is the cap, at one quarter of the live
/// record count. The deepest calibrated fetch measured is 17.8 percent of its
/// corpus, on a fifty cluster generator trained on 1,000 records, so the cap
/// sits above every measured cell and below a full scan. It exists for the data
/// relay 54 recorded, where the codes resolve nothing and the depth the
/// calibration measures is most of the sample. It bounds that case rather than
/// repairing it, and no fetch repairs it.
///
/// # What it costs and where it is absent
///
/// The calibration runs once, inside training, and `get_stats` reports what it
/// measured and what it cost. `quantized_only` never reranks, so it is not
/// calibrated and pays nothing.
///
/// An index trained before this existed carries no calibration, and its
/// `quantization.json` has no field for one. It falls back to the three corpus
/// terms above, which is what it was built against.
pub const RERANK_CALIBRATION_TARGET: f64 = 0.99;
pub const RERANK_CALIBRATION_SAFETY: f64 = 1.75;
pub const RERANK_CALIBRATION_EXPONENT_BIAS: f64 = 0.15;
pub const RERANK_CALIBRATION_EXPONENT_MIN: f64 = 0.40;
pub const RERANK_CALIBRATION_EXPONENT_MAX: f64 = 1.00;
pub const RERANK_CALIBRATION_CAP_DIVISOR: usize = 4;

/// Fractions of the training sample the exponent is fitted over
///
/// The sample is held in a seeded random order, so a fraction of it is a random
/// draw over it and a prefix is the cheapest way to take one. The whole sample
/// has to be one of them, since its fetch is what the search scales from. The
/// measurement costs the sum of these in units of one pass over the sample, so
/// this set costs 2.5 against the 1.5 the two point fit cost.
pub const RERANK_CALIBRATION_FIT_FRACTIONS: [f64; 4] = [0.25, 0.50, 0.75, 1.00];

pub const RERANK_CALIBRATION_QUERIES: usize = 512;

/// Page size the calibration measures the depth for
///
/// The fetch a page of 10 needs. It is the reference page, so the scaling in
/// `RerankCalibration::fetch_at` is exactly one at this page and a search at
/// `top_k` 10 fetches what it fetched before the page term was calibrated.
pub const RERANK_CALIBRATION_TOP_K: usize = 10;

/// Pages the calibration measures the depth at, ascending
///
/// The reference page has to be one of them, since the page exponent is the
/// slope through these points and the fetch the search scales from is measured
/// at the reference.
///
/// # Why a page term is calibrated at all
///
/// The fetch used to be measured for a page of ten and nothing scaled it, so a
/// search at `top_k` 100 fetched exactly what a search at `top_k` 10 fetched.
/// `DEFAULT_RERANK_PAGE_FACTOR` was the only page term and at five times
/// `top_k` it is 500 at a page of 100, which sits below the calibrated fetch at
/// every corpus this was measured on and therefore never governed. Measured
/// recall at 100 on `quantized_with_raw` read 0.9243 at 50,000 dbpedia-openai
/// records against 0.9940 for the same records unquantized, and it got worse as
/// the corpus shrank, because a smaller corpus calibrates a smaller fetch.
///
/// # The page requirement is sublinear
///
/// The smallest fetch reaching mean recall 0.99 at a page, read off built
/// indexes by sweeping the explicit `rerank` argument, 200 queries against
/// exact ground truth to depth 1,000.
///
/// See the report for the full table. The requirement rises with the page and
/// it rises far more slowly than the page does, because what buries a true
/// neighbour in the ADC ordering is the number of records the codes cannot
/// separate from the query and that count does not move when a caller asks for
/// more results. Only once the page approaches that count does the requirement
/// track it, which is why the exponent is fitted rather than fixed and why it
/// is clamped below one.
pub const RERANK_CALIBRATION_PAGES: [usize; 3] = [1, 10, 100];

/// Bounds on the fitted page exponent
///
/// Zero is a fetch that ignores the page, which is what shipped before this was
/// measured. One is a fetch proportional to the page, which is what every
/// constant multiple of `top_k` assumes and which no measured corpus needs.
pub const RERANK_CALIBRATION_PAGE_EXPONENT_MIN: f64 = 0.0;
pub const RERANK_CALIBRATION_PAGE_EXPONENT_MAX: f64 = 1.0;

/// The page exponent an index that never measured one falls back to
///
/// An index trained before the page term existed carries a `quantization.json`
/// with no field for it, and a sample too small to compare two pages measures
/// no slope. Both take this value rather than a flat fetch.
///
/// It is the median of the six exponents the calibration fitted on the three
/// real datasets at 50,000 and 100,000 records, being 0.346 and 0.381 on
/// glove-100, 0.474 and 0.499 on sift-128 and 0.551 and 0.567 on
/// dbpedia-openai. It is a default and not a calibration, since it is not
/// anything the index it applies to measured.
///
/// Applying it to an index that did not measure its own pages cannot cost
/// recall at the reference page, because the scaling is exactly one there
/// whatever the exponent is. Above the reference page it can only deepen the
/// fetch.
pub const RERANK_CALIBRATION_DEFAULT_PAGE_EXPONENT: f64 = 0.49;

/// What the calibration measured, and on what
///
/// Stored with the index and written to `quantization.json`, so it survives a
/// save and a load and is not recomputed. `sample_records` is the corpus the
/// fetch was measured over, and it is the denominator the search scales from.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RerankCalibration {
    /// Candidates reaching mean recall at 10 of `target` on the whole sample.
    pub fetch: usize,
    /// The same figure at each of `RERANK_CALIBRATION_FIT_FRACTIONS`, which is
    /// what `exponent` is fitted from. The last entry is `fetch`. Zero where a
    /// fraction was too small to measure, and all zero on a calibration read
    /// back from a directory written before the fit was recorded.
    #[serde(default)]
    pub fit_fetches: [usize; RERANK_CALIBRATION_FIT_FRACTIONS.len()],
    /// The power of the record count the fetch is scaled by. The least squares
    /// slope of the log fetch on the log record count over `fit_fetches`,
    /// corrected and clamped.
    pub exponent: f64,
    /// The fetch at each of `RERANK_CALIBRATION_PAGES`, measured over the whole
    /// sample, which is what `page_exponent` is fitted from. Zero where the
    /// sample was too small for that page, and all zero on a calibration read
    /// back from a directory written before the page term existed.
    #[serde(default)]
    pub page_fetches: [usize; RERANK_CALIBRATION_PAGES.len()],
    /// The power of the requested page the fetch is scaled by. The least
    /// squares slope of the log fetch on the log page over `page_fetches`,
    /// clamped. `RERANK_CALIBRATION_DEFAULT_PAGE_EXPONENT` on a calibration
    /// that did not measure it.
    #[serde(default = "default_page_exponent")]
    pub page_exponent: f64,
    /// Records `fetch` was measured over.
    pub sample_records: usize,
    /// Queries the ranks were pooled across.
    pub queries: usize,
    /// The recall `fetch` reaches on the sample.
    pub target: f64,
    /// Wall clock milliseconds both measurements took.
    pub millis: u64,
}

/// Serde's fallback for a calibration written before the page term existed.
fn default_page_exponent() -> f64 {
    RERANK_CALIBRATION_DEFAULT_PAGE_EXPONENT
}

/// `y` at `x`, by straight lines through `points`
///
/// `points` is ascending in `x`. Outside the measured range the nearest
/// segment's slope carries on, which is an extrapolation and is bounded by the
/// caller rather than trusted. Fewer than two points leave no line to read, and
/// the single value or zero is the answer rather than an index out of bounds.
fn interpolate(points: &[(f64, f64)], x: f64) -> f64 {
    match points.len() {
        0 => return 0.0,
        1 => return points[0].1,
        _ => {}
    }
    let first = points[0];
    let last = points[points.len() - 1];
    if x <= first.0 {
        let next = points[1];
        let run = next.0 - first.0;
        if run.abs() < f64::EPSILON {
            return first.1;
        }
        return first.1 + (next.1 - first.1) / run * (x - first.0);
    }
    if x >= last.0 {
        let prev = points[points.len() - 2];
        let run = last.0 - prev.0;
        if run.abs() < f64::EPSILON {
            return last.1;
        }
        return last.1 + (last.1 - prev.1) / run * (x - last.0);
    }
    for pair in points.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        if x >= x0 && x <= x1 {
            let run = x1 - x0;
            if run.abs() < f64::EPSILON {
                return y0;
            }
            return y0 + (y1 - y0) * (x - x0) / run;
        }
    }
    last.1
}

impl RerankCalibration {
    /// What the requested page multiplies the reference fetch by
    ///
    /// The measured pages are interpolated rather than fitted to one exponent,
    /// because the relation is convex in log space and a single slope through
    /// it is wrong at both ends. On dbpedia-openai the calibration measures 60,
    /// 162 and 777 candidates at pages 1, 10 and 100, which is a slope of 0.431
    /// over the first decade and 0.681 over the second, and the least squares
    /// line through all three reads 0.556. Against the requirement read off a
    /// built index at 50,000 records, the second decade is the one that matters
    /// and it needs 0.652. The line asks for 2,062 candidates at a page of 100
    /// where interpolating asks for 2,748 and the index needs 2,962.
    ///
    /// Interpolation passes exactly through the reference page, so the
    /// multiplier there is exactly one and a search at `RERANK_CALIBRATION_TOP_K`
    /// asks for precisely what it asked for before any page term existed.
    ///
    /// Outside the measured pages the nearest segment's slope carries on.
    ///
    /// **The result is never below one, so the page term only ever deepens the
    /// fetch.** The measurement says a page below the reference needs less, and
    /// acting on that costs recall. Measured on glove-100 at 50,000 records, the
    /// calibration reads 310 and 811 candidates at pages 1 and 10 over its
    /// training sample, which is a ratio of 0.382, where the requirement read
    /// off the built index is 1,673 against 3,468, a ratio of 0.482. Scaling the
    /// live fetch of 3,490 by the sample's ratio asks for 1,334 where the index
    /// needs 1,673, and recall at a page of one fell from 1.000 to 0.988. The
    /// ratio between two pages measured on a tenth of the records does not carry
    /// the safety factor the reference measurement carries, so it is trusted
    /// upward and not downward.
    ///
    /// It is also held at or below the page ratio, so no extrapolation can ask
    /// for more than a fetch proportional to the page.
    ///
    /// A calibration with fewer than two measured pages takes `page_exponent`,
    /// which is the shipped default on an index trained before the page term
    /// existed.
    fn page_scale(&self, top_k: usize) -> f64 {
        let reference = RERANK_CALIBRATION_TOP_K.max(1) as f64;
        let page = top_k.max(1) as f64;
        let ratio = page / reference;

        let measured: Vec<(f64, f64)> = RERANK_CALIBRATION_PAGES
            .iter()
            .zip(self.page_fetches.iter())
            .filter(|(_, &fetch)| fetch > 0)
            .map(|(&p, &fetch)| ((p as f64).ln(), (fetch as f64).ln()))
            .collect();

        let scale = if measured.len() < 2 {
            let exponent = self.page_exponent.clamp(
                RERANK_CALIBRATION_PAGE_EXPONENT_MIN,
                RERANK_CALIBRATION_PAGE_EXPONENT_MAX,
            );
            ratio.powf(exponent)
        } else {
            (interpolate(&measured, page.ln()) - interpolate(&measured, reference.ln())).exp()
        };

        if !scale.is_finite() || scale <= 0.0 {
            return 1.0;
        }
        scale.clamp(1.0, ratio.max(1.0))
    }

    /// The fetch this calibration asks for at a live record count
    ///
    /// The measured fetch scaled by the record ratio raised to the fitted
    /// `exponent` and by what the requested page asks for, multiplied by the
    /// safety factor, held between the floor terms and the cap. The caller
    /// applies the live record count as the final bound.
    ///
    /// `page_scale` is exactly one at `RERANK_CALIBRATION_TOP_K`, which is the
    /// page `fetch` was measured at, so a search there asks for exactly what it
    /// asked for before the page term existed.
    fn fetch_at(&self, live_records: usize, top_k: usize) -> usize {
        let floor =
            DEFAULT_RERANK_MIN_CANDIDATES.max(top_k.saturating_mul(DEFAULT_RERANK_PAGE_FACTOR));
        if self.sample_records == 0 {
            return floor;
        }

        let ratio = live_records as f64 / self.sample_records as f64;
        let exponent = self.exponent.clamp(
            RERANK_CALIBRATION_EXPONENT_MIN,
            RERANK_CALIBRATION_EXPONENT_MAX,
        );
        let scaled = RERANK_CALIBRATION_SAFETY
            * self.fetch as f64
            * ratio.powf(exponent)
            * self.page_scale(top_k);

        // A non-finite or negative product cannot come from a stored
        // calibration this crate wrote, and the floor answers it rather than a
        // cast that saturates somewhere surprising.
        let wanted = if scaled.is_finite() && scaled > 0.0 {
            scaled.round() as usize
        } else {
            floor
        };

        let cap = (live_records / RERANK_CALIBRATION_CAP_DIVISOR).max(floor);
        wanted.clamp(floor, cap)
    }
}

/// The raw vector distance for a space
///
/// These are the same `crate::distance` implementations `DistanceType::new_raw`
/// hands to a raw graph, so a rescored score is the number a raw index would
/// have reported for the same pair rather than a second implementation of the
/// same formula.
fn raw_distance_fn(space: &str) -> fn(&[f32], &[f32]) -> f32 {
    match space {
        "l2" => |a: &[f32], b: &[f32]| L2Dist {}.eval(a, b),
        "l1" => |a: &[f32], b: &[f32]| L1Dist {}.eval(a, b),
        _ => |a: &[f32], b: &[f32]| CosineDist {}.eval(a, b),
    }
}

/// The fetch a sample of `records` needs at each of `pages`, measured over that
/// sample
///
/// The 0.99 percentile of every true neighbour rank pooled across queries, the
/// query itself removed from both the true neighbour list and the ADC ordering.
/// `codes` holds the whole sample and only its first `records` entries are
/// scored, so the measurements `calibrate_rerank_from_sample` takes share one
/// quantization pass.
///
/// Every page is answered from one pass. The exact neighbours are found once to
/// the deepest page asked for and their ranks are kept in true neighbour order,
/// so the pool for a shallower page is a prefix of the pool for a deeper one.
/// The deepest page therefore costs a longer running best list and nothing
/// else, and the distances and the ADC ordering are computed once whatever is
/// asked for.
///
/// `pages` must be ascending. The returned vector is one fetch per page.
/// `None` where the sample has no room for a rank distribution at the shallowest
/// page, being fewer records than twice that page. A page the sample is too
/// small for returns zero in its slot.
fn measure_rerank_fetches(
    pq: &PQ,
    sample: &[Vec<f32>],
    codes: &[Vec<u8>],
    distance: fn(&[f32], &[f32]) -> f32,
    records: usize,
    pages: &[usize],
) -> Option<Vec<usize>> {
    let top_k = *pages.last()?;
    let shallowest = *pages.first()?;
    if records < shallowest * 2 + 1 || records > sample.len() {
        return None;
    }
    // The deepest page the sample can carry a rank distribution for.
    let top_k = top_k.min((records - 1) / 2).max(shallowest);

    // Spread the queries over insertion order rather than taking a prefix,
    // since a prefix of a corpus inserted in a meaningful order is a narrower
    // slice of it than the sample as a whole.
    let stride = (records / RERANK_CALIBRATION_QUERIES).max(1);
    let query_ids: Vec<usize> = (0..records)
        .step_by(stride)
        .take(RERANK_CALIBRATION_QUERIES)
        .collect();
    if query_ids.is_empty() {
        return None;
    }

    let subvectors = pq.subvectors;

    // One rank list per query, in true neighbour order, so that the pool for a
    // page is a prefix of each of them.
    let per_query: Vec<Vec<usize>> = query_ids
        .par_iter()
        .map(|&qi| {
            let query = &sample[qi];

            // The exact top_k, the query itself excluded. A running list of the
            // best few costs one comparison per record and no sort of the
            // sample.
            let mut best: Vec<(f32, usize)> = Vec::with_capacity(top_k + 1);
            for (i, vector) in sample.iter().enumerate().take(records) {
                if i == qi {
                    continue;
                }
                let d = distance(query, vector);
                if best.len() < top_k {
                    best.push((d, i));
                    if best.len() == top_k {
                        best.sort_by(|a, b| a.0.total_cmp(&b.0));
                    }
                } else if d < best[top_k - 1].0 {
                    let at = best.partition_point(|entry| entry.0 <= d);
                    best.insert(at, (d, i));
                    best.truncate(top_k);
                }
            }
            if best.len() < top_k {
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
            }

            // The ADC ordering the search would traverse, over the same records
            // and with the query excluded from it as well.
            let lut = match pq.compute_adc_lut(query) {
                Ok(lut) => lut,
                Err(_) => return Vec::new(),
            };
            let adc: Vec<f32> = codes
                .iter()
                .take(records)
                .enumerate()
                .map(|(i, code)| {
                    if i == qi {
                        return f32::INFINITY;
                    }
                    let mut sum = 0.0f32;
                    for s in 0..subvectors {
                        sum += lut[s][code[s] as usize];
                    }
                    sum
                })
                .collect();

            // Ranks come off a sorted copy rather than a scan per neighbour.
            // The scan is one pass over the sample for every true neighbour,
            // which is affordable at a page of ten and is not at a page of a
            // hundred, and one sort serves every page.
            let mut ordered = adc.clone();
            ordered.sort_unstable_by(|a, b| a.total_cmp(b));

            best.iter()
                .map(|&(_, neighbour)| {
                    let target = adc[neighbour];
                    1 + ordered.partition_point(|&d| d < target)
                })
                .collect::<Vec<usize>>()
        })
        .collect();

    if per_query.iter().all(|ranks| ranks.is_empty()) {
        return None;
    }

    // Mean recall at a page of P over a fetch of F is the share of the pooled
    // ranks of the first P true neighbours that sit at or below F, so the fetch
    // reaching the target is that percentile of the pool.
    let mut out = Vec::with_capacity(pages.len());
    for &page in pages {
        if page > top_k {
            out.push(0);
            continue;
        }
        let mut pool: Vec<usize> = Vec::with_capacity(per_query.len() * page);
        for ranks in &per_query {
            pool.extend(ranks.iter().take(page).copied());
        }
        if pool.is_empty() {
            out.push(0);
            continue;
        }
        pool.sort_unstable();
        let position =
            ((RERANK_CALIBRATION_TARGET * pool.len() as f64).ceil() as usize).clamp(1, pool.len());
        out.push(pool[position - 1]);
    }
    Some(out)
}

/// The rerank calibration, with no index around it
///
/// Split from `HNSWIndex::calibrate_rerank` so the measurement is reachable
/// without an index, which is what lets it be tested directly. The method
/// decides whether to run this, and this decides what it produces.
///
/// One measurement per fraction in `RERANK_CALIBRATION_FIT_FRACTIONS`. The one
/// over the whole sample is the fetch the sample itself needs, and the set of
/// them is what fixes the record exponent. That last measurement also answers
/// every page in `RERANK_CALIBRATION_PAGES` from the same pass, which is what
/// fixes the page exponent; see `RerankCalibration`.
fn calibrate_rerank_from_sample(
    pq: &PQ,
    sample: &[Vec<f32>],
    distance: fn(&[f32], &[f32]) -> f32,
) -> Option<RerankCalibration> {
    let total = sample.len();
    if total < RERANK_CALIBRATION_TOP_K * 4 + 2 {
        return None;
    }

    let started = Instant::now();

    let refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
    let codes = pq.quantize_batch(&refs).ok()?;
    if codes.len() != total {
        return None;
    }

    // The sample is in a seeded random order, so its first `records` entries
    // are a random draw over it and no separate sampling pass is needed.
    //
    // Every fraction but the last measures the reference page alone, since the
    // record exponent is fitted from that one page. The last one is the whole
    // sample and it measures every page, because the pages have to be compared
    // at one record count and the whole sample is the largest available.
    let reference = [RERANK_CALIBRATION_TOP_K];
    let mut fit_fetches = [0usize; RERANK_CALIBRATION_FIT_FRACTIONS.len()];
    let mut page_fetches = [0usize; RERANK_CALIBRATION_PAGES.len()];
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(fit_fetches.len());
    let last = fit_fetches.len() - 1;
    for (i, fraction) in RERANK_CALIBRATION_FIT_FRACTIONS.iter().enumerate() {
        let records = ((total as f64 * fraction).round() as usize).min(total);
        let pages: &[usize] = if i == last {
            &RERANK_CALIBRATION_PAGES
        } else {
            &reference
        };
        let Some(measured) = measure_rerank_fetches(pq, sample, &codes, distance, records, pages)
        else {
            continue;
        };
        let at_reference = pages
            .iter()
            .position(|&p| p == RERANK_CALIBRATION_TOP_K)
            .and_then(|slot| measured.get(slot).copied())
            .unwrap_or(0);
        if i == last && measured.len() == page_fetches.len() {
            page_fetches.copy_from_slice(&measured);
        }
        if at_reference > 0 {
            fit_fetches[i] = at_reference;
            points.push(((records as f64).ln(), (at_reference as f64).ln()));
        }
    }

    // The whole sample is the last fraction and its fetch is what the search
    // scales from, so a calibration without it measures nothing usable.
    let fetch = *fit_fetches.last()?;
    if fetch == 0 {
        return None;
    }

    // A subset measuring a deeper fetch than the whole sample sits below the
    // size at which the codes resolve this data at all, so a slope that is not
    // positive carries no signal and the safe bound is linear growth.
    let raw = least_squares_slope(&points).unwrap_or(0.0);
    let exponent = if raw <= 0.0 {
        RERANK_CALIBRATION_EXPONENT_MAX
    } else {
        (raw + RERANK_CALIBRATION_EXPONENT_BIAS).clamp(
            RERANK_CALIBRATION_EXPONENT_MIN,
            RERANK_CALIBRATION_EXPONENT_MAX,
        )
    };

    // The page exponent, over whichever pages the sample was large enough to
    // measure. A sample that could only answer one page carries no slope, and
    // the shipped default is the honest answer there rather than a flat fetch.
    let page_points: Vec<(f64, f64)> = RERANK_CALIBRATION_PAGES
        .iter()
        .zip(page_fetches.iter())
        .filter(|(_, &f)| f > 0)
        .map(|(&p, &f)| ((p as f64).ln(), (f as f64).ln()))
        .collect();
    let page_exponent = match least_squares_slope(&page_points) {
        Some(slope) => slope.clamp(
            RERANK_CALIBRATION_PAGE_EXPONENT_MIN,
            RERANK_CALIBRATION_PAGE_EXPONENT_MAX,
        ),
        None => RERANK_CALIBRATION_DEFAULT_PAGE_EXPONENT,
    };

    Some(RerankCalibration {
        fetch,
        fit_fetches,
        exponent,
        page_fetches,
        page_exponent,
        sample_records: total,
        queries: (0..total)
            .step_by((total / RERANK_CALIBRATION_QUERIES).max(1))
            .take(RERANK_CALIBRATION_QUERIES)
            .count(),
        target: RERANK_CALIBRATION_TARGET,
        millis: started.elapsed().as_millis() as u64,
    })
}

/// The least squares slope of `y` on `x`
///
/// `None` where there are fewer than two points or every point shares one `x`,
/// both of which leave the slope undefined rather than large.
fn least_squares_slope(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance = 0.0;
    for (x, y) in points {
        covariance += (x - mean_x) * (y - mean_y);
        variance += (x - mean_x) * (x - mean_x);
    }
    if variance <= 0.0 || !covariance.is_finite() {
        return None;
    }
    Some(covariance / variance)
}

/// How a quantized search over-fetches and rescores
///
/// Present only when the index is quantized, its storage mode keeps raw
/// vectors, and the caller has not turned rerank off. `HNSWIndex::rerank_plan`
/// is the single place that decides, and all three search paths take it from
/// there.
#[derive(Clone, Copy)]
struct RerankPlan {
    /// Candidates to pull from the graph per requested result, when the caller
    /// named a factor. `None` means the caller named none and the fetch comes
    /// from the calibration, or from the live record count where there is no
    /// calibration; see `SearchParams::fetch_k`.
    factor: Option<usize>,
    /// What training measured on this index's own data, where it ran. `None`
    /// for an index trained before the calibration existed.
    calibration: Option<RerankCalibration>,
    /// The space's raw vector distance.
    distance: fn(&[f32], &[f32]) -> f32,
}

/// The settings a search carries once its input has been parsed
///
/// Bundled rather than threaded through one at a time, because the three batch
/// entry points forward every one of them unchanged and the page size, the
/// traversal breadth and the rerank plan are read together wherever they are
/// read at all.
#[derive(Clone, Copy)]
struct SearchParams {
    top_k: usize,
    ef: usize,
    return_vector: bool,
    rerank: Option<RerankPlan>,
}

impl SearchParams {
    /// Candidates to ask the graph for
    ///
    /// The requested page unless the page is going to be reordered, and the
    /// over-fetch capped at the live record count when it is. The cap means a
    /// large factor degrades to a full scan rather than to an allocation the
    /// size of the factor, and it is also what keeps the default from
    /// over-fetching more than a small index holds.
    ///
    /// A caller's own factor is a multiple of the page, unchanged. An unset
    /// factor takes what training measured on this index's own data, scaled to
    /// the live record count; see `RerankCalibration`. Where there is no
    /// calibration it takes the largest of the corpus term, the floor and the
    /// page term, for the reasons recorded on those three constants.
    fn fetch_k(&self, live_records: usize) -> usize {
        match self.rerank {
            Some(plan) => {
                let wanted = match (plan.factor, plan.calibration) {
                    (Some(factor), _) => self.top_k.saturating_mul(factor),
                    (None, Some(calibration)) => calibration.fetch_at(live_records, self.top_k),
                    (None, None) => (live_records / DEFAULT_RERANK_CORPUS_DIVISOR)
                        .max(DEFAULT_RERANK_MIN_CANDIDATES)
                        .max(self.top_k.saturating_mul(DEFAULT_RERANK_PAGE_FACTOR)),
                };
                wanted.min(live_records.max(self.top_k))
            }
            None => self.top_k,
        }
    }
}

/// Score one candidate against the query on the raw vector scale
///
/// The stored raw vector where the index holds one, which under
/// `quantized_with_raw` is every record, and the reconstruction from its codes
/// otherwise. Both are vectors of the index's own dimension, so a page mixing
/// them is still ordered by one distance.
///
/// `None` means the candidate holds neither a raw vector nor codes, which no
/// record resolving through `rev_map` can be. The callers sort such a candidate
/// last rather than letting an unscored one displace a scored one.
fn rescore_candidate(
    plan: &RerankPlan,
    query: &[f32],
    ext_id: &str,
    vectors: &HashMap<String, Vec<f32>>,
    pq: Option<&Arc<PQ>>,
    pq_codes: &HashMap<String, Vec<u8>>,
) -> Option<f32> {
    if let Some(stored) = vectors.get(ext_id) {
        return Some((plan.distance)(query, stored));
    }
    let reconstructed = pq?.reconstruct(pq_codes.get(ext_id)?).ok()?;
    Some((plan.distance)(query, &reconstructed))
}

/// Order a rescored page and cut it to the requested size
///
/// `total_cmp` rather than `partial_cmp`, so a non-finite score orders rather
/// than panicking the sort.
fn take_best<T>(scored: &mut Vec<(T, f32)>, top_k: usize) {
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(top_k);
}

thread_local! {
    /// The ADC lookup table for the query the calling thread is running.
    ///
    /// The table belongs to a query, not to an index. It used to live on
    /// `DistPQ`, one per index, which meant two searches overlapping would each
    /// overwrite the other's table and score candidates against a query they
    /// were never given. An exclusive lock on the graph was the only thing
    /// preventing that, so the table had to move before the lock could be
    /// relaxed.
    ///
    /// `Distance::eval` takes `&self` and has no parameter to carry per query
    /// state, so the table cannot be threaded through as an argument. Thread
    /// local storage is the way to give it to `eval` without giving it to the
    /// index, and it needs no change to the vendored crate.
    ///
    /// The invariant this rests on is that one query's traversal runs entirely
    /// on the thread that installed its table. `Hnsw::search` is sequential
    /// within a single query, so that holds. `batch_search_parallel` splits
    /// across queries rather than within one, and each query installs its own
    /// table on the worker that runs it. Adopting `Hnsw::parallel_search`, which
    /// would fan one query's distance evaluations out across the pool, would
    /// break this and must not be done without replacing the mechanism.
    static QUERY_LUT: RefCell<Option<Vec<Vec<f32>>>> = const { RefCell::new(None) };
}

/// Holds a query's ADC table on the calling thread and removes it on drop.
///
/// Drop rather than an explicit clear, so an early return or a panic inside the
/// traversal cannot leave a stale table behind for the next query this thread
/// runs. A leftover table would be read as if it belonged to that next query.
struct QueryLut;

impl Drop for QueryLut {
    fn drop(&mut self) {
        QUERY_LUT.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Custom distance function for Product Quantization using ADC
#[derive(Clone)]
pub struct DistPQ {
    /// Reference to the PQ instance for accessing centroids
    pq: Arc<PQ>,
}

impl DistPQ {
    pub fn new(pq: Arc<PQ>) -> Self {
        DistPQ { pq }
    }

    /// Compute this query's ADC table and install it for the calling thread.
    ///
    /// The returned guard must be held for the whole traversal. Dropping it
    /// early returns the thread to graph construction mode, where `eval` reads
    /// the codebook's symmetric table instead.
    fn install_query_lut(&self, query: &[f32]) -> Result<QueryLut, String> {
        if !self.pq.is_trained() {
            return Err("PQ must be trained before ADC computation".to_string());
        }

        let lut = self.pq.compute_adc_lut(query)?;
        QUERY_LUT.with(|slot| *slot.borrow_mut() = Some(lut));
        Ok(QueryLut)
    }
}

impl Distance<u8> for DistPQ {
    /// Distance between two points the graph holds, both of which are PQ codes
    ///
    /// A query table on this thread means a search is running. `a` is then the
    /// dummy code vector `DistanceType::search` passes, the real query lives in
    /// the table, and the distance is asymmetric: query subvector against stored
    /// centroid.
    ///
    /// No query table means graph construction, where there is no query and
    /// both `a` and `b` are stored codes. The distance is then symmetric,
    /// centroid against centroid, read from the table the codebook carries.
    /// Returning infinity here, which is what this did until the symmetric
    /// table existed, made every candidate tie in the neighbour selection
    /// heuristic and left the graph with one edge per node.
    ///
    /// Both branches return a sum of squared L2 distances, so they are on the
    /// same scale and neither takes a square root.
    ///
    /// Choosing the branch on the table rather than on `a` is deliberate. The
    /// dummy query is a valid code slice and cannot be told apart from real
    /// codes by inspection. It is sound because the table is thread local, so an
    /// insertion can never observe a query table it did not install itself, no
    /// matter what any other thread is doing at the time. That used to depend on
    /// the graph mutex serialising searches against insertions, which is the
    /// dependency this removes.
    fn eval(&self, a: &[u8], b: &[u8]) -> f32 {
        QUERY_LUT.with(|slot| {
            let slot = slot.borrow();
            let Some(lut) = slot.as_ref() else {
                return self.pq.symmetric_distance(a, b);
            };

            // b.len() should equal pq.subvectors
            let mut sum = 0.0f32;
            for (sv, &code) in b.iter().enumerate() {
                // lut[sv][code]
                let distance_component = lut
                    .get(sv)
                    .and_then(|row| row.get(code as usize))
                    .copied()
                    .unwrap_or(f32::INFINITY);
                sum += distance_component;
            }
            sum
        })
    }
}

/// Bytes an `Arc<T>` allocation carries beyond `T`, being the strong and the
/// weak count.
const ARC_COUNTS_BYTES: usize = 2 * std::mem::size_of::<usize>();

/// Bytes a `Vec<T>` header occupies, being a pointer, a capacity and a length.
const VEC_HEADER_BYTES: usize = 3 * std::mem::size_of::<usize>();

/// Bytes `parking_lot::RwLock<()>` occupies, being one `AtomicUsize`.
const PARKING_LOT_LOCK_BYTES: usize = std::mem::size_of::<usize>();

/// The capacity `Vec::push` gives a buffer it has just allocated for the first
/// time. `RawVec::MIN_NON_ZERO_CAP` is 4 for an element of 8 bytes.
const MIN_VEC_CAP: usize = 4;

/// Points whose neighbour lists the graph memory figure is measured over.
///
/// The adjacency count is a property of the data rather than of `m`, so it is
/// sampled rather than derived; see `graph_memory_bytes`. The sample is taken
/// by striding the point enumeration, which is insertion order within a layer,
/// because a prefix would be all early records and an early record has taken
/// more reverse links than a late one.
const GRAPH_SAMPLE_POINTS: usize = 4096;

/// Layer indices `graph_memory_bytes` asks the graph about.
///
/// The vendored crate fixes the layer count at `NB_LAYER_MAX`, which is 16 and
/// is `pub(crate)`, and `get_layer_nb_point` answers zero for an index it does
/// not have. Probing past the end therefore costs one lock and no correctness.
const GRAPH_LAYER_PROBE: usize = 32;

/// Layer `Vec` headers a point carries when nothing was sampled to count them.
///
/// Only reachable on a graph that reports points and holds none in any layer,
/// which no path produces. It is the crate's `NB_LAYER_MAX`.
const GRAPH_LAYERS_FALLBACK: usize = 16;

/// What the HNSW graph holds, in bytes it has asked the allocator for
///
/// `get_stats` used to report the storage maps and the two quantization tables
/// and stop there, which on a trained `quantized_only` index at 50,000 records
/// of dimension 1,536 named 9.77 MB against a measured 231 MiB resident. The
/// graph is the rest of it and this is what it holds.
///
/// # Per point
///
/// The graph owns a second copy of every point, separate from the storage map,
/// and it is `dim * 4` bytes in a raw graph and `subvectors` bytes in a
/// quantized one. That copy is one allocation. Around it the vendored crate
/// carries five more, all of them fixed and none of them proportional to the
/// dimension.
///
/// ```text
///   Arc<Point<T>>                              16 + size_of::<Point<T>>()
///   the point's own data vector                dim * 4, or subvectors
///   Arc<RwLock<Vec<Vec<Arc<PointWithOrder>>>>>  16 + 8 + 24
///   sixteen layer Vec headers                  16 * 24
///   its Arc slot in points_by_layer            8
/// ```
///
/// `Point` is 112 bytes on a 64 bit target, being a 24 byte `PointData` enum,
/// a `DataId`, a `PointId`, the `Arc` to the neighbour lists and a 64 byte
/// `[AtomicU32; 16]` of in-degree counters. `size_of` is taken rather than
/// written down. The sixteen layer headers are allocated for every point
/// whatever level it was drawn at, because `Point::new` fills the outer `Vec`
/// to `NB_LAYER_MAX` before it knows anything about the point.
///
/// # Per adjacency entry
///
/// Every entry in a neighbour list is an `Arc<PointWithOrder>`, which is 16
/// bytes of `Arc` counts around a pointer to the target and an `f32` distance,
/// and a pointer slot in the list itself.
///
/// **The number of entries is a property of the data and not of `m`.** Layer
/// zero caps a list at `2 * m` and the crate does fill it on data with no
/// structure, measured at exactly 32.000 entries per point at `m` 16 and
/// exactly 64.000 at `m` 32 over 40,000 uniform points on the sphere. Real
/// embeddings do not fill it, because `select_neighbours` prunes a candidate
/// that sits closer to an already chosen neighbour than to the query and
/// clustered data gives it far more to prune. The same measurement over 50,000
/// dbpedia-openai records at `m` 32 reads 29.95 at the full 1,536 dimensions
/// and 36.75 over their first 128, and over 10,000 of them at `m` 16 it reads
/// 24.81. **A count derived from `m` alone is 2.03 times the truth** at 50,000
/// records of dimension 1,536, being 3,401,398 entries against the 1,677,300
/// the saved graph dump holds. So the entry count is measured over
/// `GRAPH_SAMPLE_POINTS` points and scaled.
///
/// A list holds more slots than entries. It is filled once by `clone_from`,
/// which sizes it exactly, and grown afterwards by the reverse link updates,
/// which double it. `2 * len` is the capacity that produces, and it is what the
/// measurement on the uniform corpus asks for: at `m` 32 the live bytes exceed
/// a length based count by 506 per point where doubling a full layer zero list
/// predicts 512.
///
/// # What it does not cover
///
/// The allocator. Every block above carries a header and is rounded up, and the
/// process commits more than the sum of the blocks. Measured on this platform a
/// 32 byte request occupies 52 bytes of commit and a 512 byte request occupies
/// 551, and the whole graph commits between 1.4 and 1.7 times what this figure
/// names. That is allocator behaviour rather than something the graph holds,
/// and folding a platform factor into a reported number would state it as
/// though the structure carried it.
fn graph_memory_bytes<T, D>(hnsw: &Hnsw<'_, T, D>) -> usize
where
    T: Clone + Send + Sync + 'static,
    D: Distance<T> + Send + Sync,
{
    let indexation = hnsw.get_point_indexation();
    let nb_point = indexation.get_nb_point();
    if nb_point == 0 {
        return 0;
    }

    let element_bytes = indexation.get_data_dimension() * std::mem::size_of::<T>();
    let point_bytes = ARC_COUNTS_BYTES + std::mem::size_of::<Point<'static, T>>();
    let neighbour_cell_bytes = ARC_COUNTS_BYTES + PARKING_LOT_LOCK_BYTES + VEC_HEADER_BYTES;
    // `PointWithOrder` is a pointer to the target and an `f32` distance, and it
    // is padded to the pointer's alignment, so it is two words rather than one
    // and a half. It is `pub(crate)` in the vendored crate, so its size is
    // written out rather than taken.
    let entry_bytes = ARC_COUNTS_BYTES + 2 * std::mem::size_of::<usize>();
    let slot_bytes = std::mem::size_of::<usize>();

    // The adjacency, over a strided sample. `get_neighborhood_id` is the only
    // way out of the crate and it reallocates, so it is not called on every
    // point of a large graph. The stride runs across the concatenation of the
    // layers rather than within one, so the sample holds upper layer points in
    // the proportion the graph does, and a point at an upper layer carries more
    // adjacency than one at layer zero.
    //
    // One layer at a time, and never two iterators at once. Each iterator holds
    // a read guard on `points_by_layer` for its whole life, `parking_lot` does
    // not admit a recursive read while a writer is queued, and a concurrent
    // `add` queues exactly that writer. `get_layer_nb_point` takes the same
    // guard, so the counts are read before the first iterator exists.
    // Every layer is probed rather than stopping at the first empty one,
    // because a level is drawn independently per point and an empty layer below
    // an occupied one is legal. A layer index the graph does not have returns
    // zero rather than raising, so the probe is bounded by the crate's own
    // `NB_LAYER_MAX` without naming it.
    let layer_counts: Vec<usize> = (0..GRAPH_LAYER_PROBE)
        .map(|layer| indexation.get_layer_nb_point(layer))
        .collect();

    let stride = nb_point.div_ceil(GRAPH_SAMPLE_POINTS).max(1);
    let mut seen = 0usize;
    let mut sampled = 0usize;
    let mut adjacency = 0usize;
    let mut layers = 0usize;
    for (index, count) in layer_counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        for point in indexation.get_layer_iterator(index) {
            let take = seen.is_multiple_of(stride);
            seen += 1;
            if !take {
                continue;
            }
            let neighbourhood = point.get_neighborhood_id();
            layers = layers.max(neighbourhood.len());
            for list in &neighbourhood {
                if list.is_empty() {
                    continue;
                }
                let capacity = (2 * list.len()).max(MIN_VEC_CAP);
                adjacency += capacity * slot_bytes + list.len() * entry_bytes;
            }
            sampled += 1;
        }
    }

    if layers == 0 {
        layers = GRAPH_LAYERS_FALLBACK;
    }
    let fixed =
        point_bytes + element_bytes + neighbour_cell_bytes + layers * VEC_HEADER_BYTES + slot_bytes;
    let mut total = nb_point * fixed;
    if sampled > 0 {
        total += ((adjacency as f64 / sampled as f64) * nb_point as f64).round() as usize;
    }
    total
}

// Enhanced DistanceType enum to support PQ variants
enum DistanceType {
    // Raw vector variants
    Cosine(Hnsw<'static, f32, CosineDist>),
    L2(Hnsw<'static, f32, L2Dist>),
    L1(Hnsw<'static, f32, L1Dist>),

    // PQ variants - corrected to use u8 element type
    CosinePQ(Hnsw<'static, u8, DistPQ>),
    L2PQ(Hnsw<'static, u8, DistPQ>),
    L1PQ(Hnsw<'static, u8, DistPQ>),
}

impl DistanceType {
    fn new_raw(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        info!(
            operation = "hnsw_creation",
            space = space,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "raw",
            "Creating raw HNSW index"
        );

        match space {
            "cosine" => DistanceType::Cosine(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                CosineDist {},
            )),
            "l2" => DistanceType::L2(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                L2Dist {},
            )),
            "l1" => DistanceType::L1(Hnsw::new(
                m,
                expected_size,
                max_layer,
                ef_construction,
                L1Dist {},
            )),
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error
                error!(
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided"
                );
                // This is a programming error that should be caught earlier
                // For now, default to cosine to prevent panic
                warn!(
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                DistanceType::Cosine(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    CosineDist {},
                ))
            }
        }
    }

    fn new_pq(
        space: &str,
        m: usize,
        expected_size: usize,
        max_layer: usize,
        ef_construction: usize,
        pq: Arc<PQ>,
    ) -> Self {
        info!(
            operation = "hnsw_creation",
            space = space,
            m = m,
            expected_size = expected_size,
            max_layer = max_layer,
            ef_construction = ef_construction,
            variant = "quantized",
            subvectors = pq.subvectors,
            bits = pq.bits,
            "Creating PQ-enabled HNSW index"
        );

        match space {
            "cosine" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::CosinePQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            "l2" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::L2PQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            "l1" => {
                let dist_pq = DistPQ::new(pq);
                DistanceType::L1PQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error
                error!(
                    operation = "hnsw_creation",
                    space = space,
                    error = "invalid_space",
                    "Invalid distance space provided for PQ"
                );
                warn!(
                    operation = "hnsw_creation",
                    space = space,
                    fallback = "cosine",
                    "Defaulting to cosine distance due to invalid space"
                );
                let dist_pq = DistPQ::new(pq);
                DistanceType::CosinePQ(Hnsw::new(
                    m,
                    expected_size,
                    max_layer,
                    ef_construction,
                    dist_pq,
                ))
            }
        }
    }

    /// Search the graph, admitting only the internal ids the filter accepts.
    ///
    /// The filter runs inside the traversal, before the fixed `top_k` cut, so a
    /// node the caller rejects routes the search but never consumes a result
    /// slot. Removal and overwrite both leave a node behind that no longer
    /// resolves to a record, and without the filter each such node inside a
    /// query's `top_k` costs one result. Passing `None` restores the previous
    /// unfiltered behaviour.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn FilterT>,
    ) -> Result<Vec<hnsw_rs::prelude::Neighbour>, String> {
        match self {
            // Raw vector search
            DistanceType::Cosine(hnsw) => Ok(hnsw.search_filter(query, k, ef, filter)),
            DistanceType::L2(hnsw) => Ok(hnsw.search_filter(query, k, ef, filter)),
            DistanceType::L1(hnsw) => Ok(hnsw.search_filter(query, k, ef, filter)),

            // PQ-based search with ADC
            DistanceType::CosinePQ(hnsw) | DistanceType::L2PQ(hnsw) | DistanceType::L1PQ(hnsw) => {
                // This query's ADC table, installed on this thread alone. The
                // guard is named so it lives to the end of the arm rather than
                // dropping at the end of the statement, and it releases the
                // table once the traversal is done.
                let _query_lut = hnsw.get_distance().install_query_lut(query)?;

                // Create dummy query vector for HNSW traversal (flat u8 codes)
                let dummy_query = vec![0u8; self.get_code_size()];

                Ok(hnsw.search_filter(&dummy_query, k, ef, filter))
            }
        }
    }

    /// Number of nodes the graph holds, which is the number of insertions it has
    /// taken. It exceeds the live record count by exactly the number of nodes
    /// that removal and overwrite have stranded.
    fn nb_points(&self) -> usize {
        match self {
            DistanceType::Cosine(hnsw) => hnsw.get_nb_point(),
            DistanceType::L2(hnsw) => hnsw.get_nb_point(),
            DistanceType::L1(hnsw) => hnsw.get_nb_point(),
            DistanceType::CosinePQ(hnsw) => hnsw.get_nb_point(),
            DistanceType::L2PQ(hnsw) => hnsw.get_nb_point(),
            DistanceType::L1PQ(hnsw) => hnsw.get_nb_point(),
        }
    }

    fn get_code_size(&self) -> usize {
        match self {
            DistanceType::CosinePQ(hnsw) => hnsw.get_distance().pq.subvectors,
            DistanceType::L2PQ(hnsw) => hnsw.get_distance().pq.subvectors,
            DistanceType::L1PQ(hnsw) => hnsw.get_distance().pq.subvectors,
            _ => 0,
        }
    }

    /// Bytes the graph asks the allocator for. See `graph_memory_bytes`.
    fn memory_bytes(&self) -> usize {
        match self {
            DistanceType::Cosine(hnsw) => graph_memory_bytes(hnsw),
            DistanceType::L2(hnsw) => graph_memory_bytes(hnsw),
            DistanceType::L1(hnsw) => graph_memory_bytes(hnsw),
            DistanceType::CosinePQ(hnsw) => graph_memory_bytes(hnsw),
            DistanceType::L2PQ(hnsw) => graph_memory_bytes(hnsw),
            DistanceType::L1PQ(hnsw) => graph_memory_bytes(hnsw),
        }
    }

    fn is_quantized(&self) -> bool {
        matches!(
            self,
            DistanceType::CosinePQ(_) | DistanceType::L2PQ(_) | DistanceType::L1PQ(_)
        )
    }

    fn insert(&self, vector: &[f32], id: usize) {
        match self {
            DistanceType::Cosine(hnsw) => hnsw.insert((vector, id)),
            DistanceType::L2(hnsw) => hnsw.insert((vector, id)),
            DistanceType::L1(hnsw) => hnsw.insert((vector, id)),
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error logging
                error!(
                    operation = "vector_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_raw_vectors_into_pq_index",
                    "Cannot insert raw vectors into PQ index"
                );
            }
        }
    }

    /// Insert PQ codes into the index
    fn insert_pq_codes(&self, codes: &[u8], id: usize) {
        match self {
            DistanceType::CosinePQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            DistanceType::L2PQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            DistanceType::L1PQ(hnsw) => {
                hnsw.insert((codes, id));
            }
            _ => {
                // ✅ ENTERPRISE: Replace panic with graceful error logging
                error!(
                    operation = "pq_codes_insert",
                    error = "invalid_operation",
                    reason = "cannot_insert_pq_codes_into_raw_index",
                    "Cannot insert PQ codes into raw index"
                );
            }
        }
    }

    #[allow(dead_code)]
    fn insert_batch(&self, data: &[(&Vec<f32>, usize)]) {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        debug!(
            operation = "batch_insert",
            batch_size = data.len(),
            num_threads = num_threads,
            threshold = threshold,
            parallel = data.len() >= threshold,
            "Starting batch insertion"
        );

        if data.len() >= threshold {
            match self {
                DistanceType::Cosine(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L2(hnsw) => hnsw.parallel_insert(data),
                DistanceType::L1(hnsw) => hnsw.parallel_insert(data),
                _ => {
                    // ✅ ENTERPRISE: Replace panic with graceful error
                    error!(
                        operation = "batch_insert",
                        error = "invalid_operation",
                        reason = "cannot_batch_insert_raw_vectors_into_pq_index",
                        "Cannot batch insert raw vectors into PQ index"
                    );
                }
            }
        } else {
            for (vector, id) in data {
                self.insert(vector.as_slice(), *id);
            }
        }
    }

    fn insert_batch_pq(&self, data: &[(&Vec<u8>, usize)]) -> Result<(), String> {
        let num_threads = rayon::current_num_threads();
        let threshold = 1000 * num_threads;

        debug!(
            operation = "batch_insert_pq",
            batch_size = data.len(),
            num_threads = num_threads,
            threshold = threshold,
            parallel = data.len() >= threshold,
            "Starting PQ batch insertion"
        );

        match self {
            DistanceType::CosinePQ(hnsw) | DistanceType::L2PQ(hnsw) | DistanceType::L1PQ(hnsw) => {
                if data.len() >= threshold {
                    hnsw.parallel_insert(data);
                } else {
                    for (codes, id) in data {
                        hnsw.insert((codes.as_slice(), *id));
                    }
                }

                Ok(())
            }
            _ => Err("Cannot insert PQ codes into raw HNSW index".to_string()),
        }
    }

    /// Match a freshly constructed graph's insertion settings.
    ///
    /// `Hnsw::new` starts with `extend_candidates` false and the vendored
    /// reload sets it true, so a restored graph would build the neighbourhood
    /// of every record added after the load differently from the same record
    /// added before the save. Nothing else `load_hnsw_with_dist` fills in
    /// differs from what `new` sets.
    fn settle_after_reload(&mut self) {
        match self {
            DistanceType::Cosine(hnsw) => hnsw.set_extend_candidates(false),
            DistanceType::L2(hnsw) => hnsw.set_extend_candidates(false),
            DistanceType::L1(hnsw) => hnsw.set_extend_candidates(false),
            DistanceType::CosinePQ(hnsw) => hnsw.set_extend_candidates(false),
            DistanceType::L2PQ(hnsw) => hnsw.set_extend_candidates(false),
            DistanceType::L1PQ(hnsw) => hnsw.set_extend_candidates(false),
        }
    }
}

// ============================================================================
// RESTORING THE SAVED GRAPH
// ============================================================================

/// Basename `save_hnsw_graph` dumps under and the loader reads back.
const HNSW_DUMP_BASENAME: &str = "hnsw_index";

/// Header of the dumped data file, being one magic and the data dimension.
const DUMP_DATA_HEADER_BYTES: usize = 4 + 8;

/// What the dumped data file spends per point before the vector itself, being
/// one magic, the origin id and the serialized byte length.
const DUMP_DATA_POINT_BYTES: usize = 4 + 8 + 8;

/// Layers the vendored crate always dumps, being its `NB_LAYER_MAX`.
const DUMP_LAYERS: u8 = 16;

/// Set to any non-empty value other than `0` to skip the saved graph and
/// rebuild it by re-inserting every record.
///
/// The rebuild is what upgrades an index whose graph was built by a release
/// carrying a defect the vendored patches have since fixed, since restoring the
/// dump restores the graph exactly as it was written, defects included. Without
/// this there is no way to ask for that upgrade on a directory whose dump is
/// perfectly readable.
const REBUILD_ENV: &str = "ZEUSDB_LOAD_REBUILD_GRAPH";

/// Whether the caller has asked for the rebuild rather than the saved graph
fn rebuild_requested() -> bool {
    match std::env::var(REBUILD_ENV) {
        Ok(value) => !value.is_empty() && value != "0",
        Err(_) => false,
    }
}

/// Read the dump's own description and judge it against what this index expects
///
/// Everything here runs before the vendored reload is entered, because that
/// reload reaches `std::process::exit(1)` when the data file is short. The data
/// file's length is fully determined by the point count and the dimension, so
/// an exact size comparison closes that path. Every other malformed dump the
/// vendored reader meets raises a panic it can unwind from, which the caller
/// catches.
///
/// Returns the node count the dump declares.
#[allow(clippy::too_many_arguments)]
fn inspect_graph_dump(
    dir: &Path,
    dimension: usize,
    element_bytes: usize,
    t_name: &str,
    dist_name: &str,
    m: usize,
    ef_construction: usize,
    min_nodes: usize,
) -> Result<usize, String> {
    let graph_path = dir.join(format!("{}.hnsw.graph", HNSW_DUMP_BASENAME));
    let data_path = dir.join(format!("{}.hnsw.data", HNSW_DUMP_BASENAME));

    if !graph_path.exists() || !data_path.exists() {
        return Err("the directory holds no HNSW graph dump".to_string());
    }

    let file = std::fs::File::open(&graph_path)
        .map_err(|e| format!("the graph dump could not be opened: {}", e))?;
    let mut reader = std::io::BufReader::new(file);

    // `load_description` unwraps a UTF-8 conversion on the two names it reads,
    // so a dump whose header is garbage panics here rather than returning.
    let described = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hnsw_rs::hnswio::load_description(&mut reader)
    }));
    let descr = match described {
        Ok(Ok(descr)) => descr,
        Ok(Err(e)) => return Err(format!("the graph dump has no readable header: {}", e)),
        Err(_) => return Err("the graph dump has an unreadable header".to_string()),
    };

    if descr.t_name != t_name {
        return Err(format!(
            "the dump stores {} points where this index holds {}",
            descr.t_name, t_name
        ));
    }
    if descr.distname != dist_name {
        return Err(format!(
            "the dump was written under distance {} and this build uses {}",
            descr.distname, dist_name
        ));
    }
    if descr.dimension != dimension {
        return Err(format!(
            "the dump stores {} values per point where this index expects {}",
            descr.dimension, dimension
        ));
    }
    if descr.nb_layer != DUMP_LAYERS {
        return Err(format!(
            "the dump declares {} layers where this build uses {}",
            descr.nb_layer, DUMP_LAYERS
        ));
    }
    if descr.max_nb_connection as usize != m {
        return Err(format!(
            "the dump was written at m {} and config.json declares {}",
            descr.max_nb_connection, m
        ));
    }
    if descr.ef != ef_construction {
        return Err(format!(
            "the dump was written at ef_construction {} and config.json declares {}",
            descr.ef, ef_construction
        ));
    }
    if descr.nb_point < min_nodes {
        return Err(format!(
            "the dump holds {} graph nodes and the index holds {} records",
            descr.nb_point, min_nodes
        ));
    }

    let expected = DUMP_DATA_HEADER_BYTES
        + descr
            .nb_point
            .saturating_mul(DUMP_DATA_POINT_BYTES + dimension * element_bytes);
    let actual = std::fs::metadata(&data_path)
        .map_err(|e| format!("the data dump could not be measured: {}", e))?
        .len();
    if actual != expected as u64 {
        return Err(format!(
            "the data dump is {} bytes where {} nodes of {} values need {}",
            actual, descr.nb_point, dimension, expected
        ));
    }

    Ok(descr.nb_point)
}

/// Reload one graph of a known element type and distance
///
/// `load_hnsw_with_dist` rather than `load_hnsw`, for two reasons. It takes the
/// distance by value, which is the only way to restore a PQ graph, since
/// `DistPQ` carries the codebook and cannot be produced by `Default`. And it
/// leaves `datamap_opt` false, where `load_hnsw` sets it true and a later
/// `file_dump` then refuses to overwrite its own files and writes
/// `hnsw_index-4173.hnsw.graph` beside them instead. Measured on the vendored
/// crate.
///
/// The reader is leaked because the vendored signature ties the returned graph's
/// lifetime to it, so that a graph reading a memory mapped data file cannot
/// outlive the mapping. Nothing is mapped here, since neither the default
/// options nor this entry point ever construct a `DataMap`, so the leak is 280
/// bytes per successful load and holds no file open.
fn reload_graph<T, D>(dir: &Path, dist: D) -> Result<Hnsw<'static, T, D>, String>
where
    T: 'static
        + Serialize
        + serde::de::DeserializeOwned
        + Clone
        + Sized
        + Send
        + Sync
        + std::fmt::Debug,
    D: Distance<T> + Send + Sync,
{
    let reader = Box::leak(Box::new(hnsw_rs::hnswio::HnswIo::new(
        dir,
        HNSW_DUMP_BASENAME,
    )));

    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reader.load_hnsw_with_dist::<T, D>(dist)
    }));

    match loaded {
        Ok(Ok(hnsw)) => Ok(hnsw),
        Ok(Err(e)) => Err(format!("the graph dump could not be read: {}", e)),
        Err(_) => Err("the graph dump is malformed and reading it panicked".to_string()),
    }
}

/// Restore the saved graph for one index configuration
///
/// `pq` present means the saved graph was a quantized one, which is exactly the
/// condition the loader branches on, since training is what replaces the raw
/// graph with a PQ graph. A directory whose dump disagrees is caught by the
/// element type and distance name in `inspect_graph_dump` and falls back.
fn restore_graph(
    dir: &Path,
    space: &str,
    m: usize,
    ef_construction: usize,
    dim: usize,
    pq: Option<Arc<PQ>>,
    min_nodes: usize,
) -> Result<(DistanceType, usize), String> {
    let mut graph = match pq {
        Some(pq) => {
            let nodes = inspect_graph_dump(
                dir,
                pq.subvectors,
                std::mem::size_of::<u8>(),
                std::any::type_name::<u8>(),
                std::any::type_name::<DistPQ>(),
                m,
                ef_construction,
                min_nodes,
            )?;
            let hnsw = reload_graph::<u8, DistPQ>(dir, DistPQ::new(pq))?;
            let restored = hnsw.get_nb_point();
            if restored != nodes {
                return Err(format!(
                    "the dump declares {} graph nodes and yielded {}",
                    nodes, restored
                ));
            }
            match space {
                "l2" => DistanceType::L2PQ(hnsw),
                "l1" => DistanceType::L1PQ(hnsw),
                _ => DistanceType::CosinePQ(hnsw),
            }
        }
        None => {
            // The raw graphs differ only in their distance type, so each arm
            // states the name the dump must carry and the value the reload
            // needs, and nothing else about them differs.
            macro_rules! raw {
                ($dist:ty, $value:expr, $variant:ident) => {{
                    let nodes = inspect_graph_dump(
                        dir,
                        dim,
                        std::mem::size_of::<f32>(),
                        std::any::type_name::<f32>(),
                        std::any::type_name::<$dist>(),
                        m,
                        ef_construction,
                        min_nodes,
                    )?;
                    let hnsw = reload_graph::<f32, $dist>(dir, $value)?;
                    let restored = hnsw.get_nb_point();
                    if restored != nodes {
                        return Err(format!(
                            "the dump declares {} graph nodes and yielded {}",
                            nodes, restored
                        ));
                    }
                    DistanceType::$variant(hnsw)
                }};
            }
            match space {
                "l2" => raw!(L2Dist, L2Dist {}, L2),
                "l1" => raw!(L1Dist, L1Dist {}, L1),
                // `new_raw` also falls back to cosine on an unrecognised space,
                // so the two construction paths agree on what a bad space means.
                _ => raw!(CosineDist, CosineDist {}, Cosine),
            }
        }
    };

    graph.settle_after_reload();
    let nodes = graph.nb_points();
    Ok((graph, nodes))
}

/// `skip_from_py_object` because nothing extracts an `AddResult`. It is the
/// return type of `add` and appears in no argument position, in this crate or
/// in the Python layer. PyO3 0.29 derives `FromPyObject` for a `#[pyclass]`
/// that is `Clone` and warns that the derive becomes opt-in, so the choice has
/// to be stated. Opting in would generate an extraction path no caller reaches.
#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
pub struct AddResult {
    #[pyo3(get)]
    pub total_inserted: usize,
    #[pyo3(get)]
    pub total_errors: usize,
    #[pyo3(get)]
    pub errors: Vec<String>,
    #[pyo3(get)]
    pub vector_shape: Option<(usize, usize)>,
}

#[pymethods]
impl AddResult {
    fn __repr__(&self) -> String {
        format!(
            "AddResult(inserted={}, errors={}, shape={:?})",
            self.total_inserted, self.total_errors, self.vector_shape
        )
    }

    pub fn is_success(&self) -> bool {
        self.total_errors == 0
    }

    /// One line human-readable summary of the insertion
    ///
    /// ASCII only, deliberately. This used to open with a check mark and carry a
    /// cross before the error count, and it is the first thing the README and the
    /// documentation site tell a new user to print. `print()` encodes through the
    /// console's code page, so on a Windows console still using the legacy one
    /// that first statement raised `UnicodeEncodeError` before the reader had
    /// added a second record.
    ///
    /// The counts stay available as `total_inserted` and `total_errors`, so the
    /// alternative of returning them as structured data would only duplicate two
    /// attributes that already exist while breaking every caller that prints
    /// this. The numbers and the words around them are unchanged, so a substring
    /// test or a `(\d+) inserted` match still holds. What no longer holds is a
    /// parse keyed on the emoji themselves or on a fixed character offset.
    pub fn summary(&self) -> String {
        format!(
            "{} inserted, {} errors",
            self.total_inserted, self.total_errors
        )
    }
}

/// Lock acquisition order for `HNSWIndex`
///
/// Every path that holds two of these guards at once acquires them in this
/// order, top to bottom. Releasing may happen in any order.
///
/// ```text
/// id_map < rev_map < hnsw < vectors < pq_codes < vector_metadata
///        < training_ids < metadata < id_counter < vector_count
/// ```
///
/// This exists because search and mutation now overlap. Until the receivers
/// were relaxed, PyO3's exclusive borrow kept every mutating method away from
/// every search, so no reader and no writer were ever in flight together and
/// the acquisition order could not matter. It matters now. A search holds
/// `rev_map` for its whole traversal and takes `vectors` afterwards, so a
/// removal taking `vectors` before `rev_map`, which is what it used to do,
/// deadlocks against it on the first interleaving that lands.
///
/// One further rule, which the order alone does not express. No path forks to
/// rayon while holding a write guard. Mutations are serialised against each
/// other by `writers`, so a read guard held across a fork can only ever be
/// blocked by that one writer, and a fork under a write guard is exactly the
/// case where the pool's workers can all end up waiting on the forking thread.
#[pyclass]
pub struct HNSWIndex {
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,

    // Quantization configuration and PQ instance
    quantization_config: Option<QuantizationConfig>,
    pq: Option<Arc<PQ>>,
    pq_codes: RwLock<HashMap<String, Vec<u8>>>, // PQ codes storage

    /// What training measured about how deep this index's codes bury a true
    /// neighbour, which is what the default rerank fetch is derived from.
    ///
    /// Written once by `calibrate_rerank` at training completion and by the
    /// loader from `quantization.json`. `None` on an unquantized index, on a
    /// `quantized_only` one, before training, and on an index trained before
    /// the calibration existed. See `RerankCalibration`.
    rerank_calibration: RwLock<Option<RerankCalibration>>,

    // Index-level metadata (simple, infrequently accessed)
    metadata: Mutex<HashMap<String, String>>,

    /// The raw vector store.
    ///
    /// Holds every record for an unquantized index and under
    /// `quantized_with_raw`. Under `quantized_only` it holds the records
    /// collected before training and nothing after: the quantization rebuild
    /// releases them once their codes are stored, and the loader drops them
    /// from a directory written before that was true. A trained
    /// `quantized_only` index therefore holds no raw vector anywhere.
    vectors: RwLock<HashMap<String, Vec<f32>>>,
    vector_metadata: RwLock<HashMap<String, HashMap<String, Value>>>,
    id_map: RwLock<HashMap<String, usize>>,
    rev_map: RwLock<HashMap<usize, String>>,

    // Mutex for write-only fields
    id_counter: Mutex<usize>,
    vector_count: Mutex<usize>, // Track total vectors for training trigger

    /// The graph. A read guard covers a traversal and a single record insertion,
    /// because `hnsw_rs` takes `&self` on both and does its own interior locking.
    /// A write guard covers replacing the whole backend, which `compact`,
    /// `rebuild_with_quantization` and the persistence rebuild each do once.
    hnsw: RwLock<DistanceType>,

    /// Serialises the mutating operations against each other, not against reads.
    ///
    /// `add`, `remove_point`, `compact` and `rebuild_with_quantization` were
    /// serialised against everything by PyO3's exclusive borrow. Relaxing the
    /// receivers removes that, and their internals are not written to interleave
    /// with each other. Id allocation, the training trigger and the overwrite
    /// path each read state and then act on it, so two of them in flight would
    /// race. This restores exactly the mutual exclusion the borrow flag gave
    /// them and nothing more, which leaves searches free to run throughout.
    ///
    /// Held by the Python entry points only. An internal caller reaching a
    /// mutating helper is already inside the guard, so the helpers never take
    /// it and cannot deadlock against the caller that owns it.
    writers: Mutex<()>,

    // ID-based training collection
    training_ids: RwLock<Vec<String>>,      // Just IDs, not vectors
    training_threshold_reached: AtomicBool, // Atomic flag for safety

    // Timestamp when the index was created
    created_at: String,

    // NEW: Flag to prevent training ID collection during persistence rebuild
    pub rebuilding_from_persistence: AtomicBool,

    /// Set once the index has warned that it holds materially more records than
    /// `expected_size` declared, so the warning fires once rather than on every
    /// subsequent `add`.
    overgrowth_warned: AtomicBool,
}

/// Build an `HNSWIndex`
///
/// The only way to construct an index from Python other than loading one from
/// disk. `HNSWIndex` carries no `#[new]`, so the class is importable for
/// `isinstance` checks and type annotations while direct construction raises
/// `TypeError`. Every rule that governs a valid index is enforced here, which
/// is what makes the Python factory and this function agree.
#[pyfunction]
#[pyo3(name = "_create_hnsw_index")]
#[pyo3(signature = (dim, space, m, ef_construction, expected_size, quantization_config = None))]
pub fn create_hnsw_index(
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    quantization_config: Option<&Bound<PyDict>>,
) -> PyResult<HNSWIndex> {
    HNSWIndex::build(
        dim,
        space,
        m,
        ef_construction,
        expected_size,
        quantization_config,
    )
}

impl HNSWIndex {
    #[instrument(level = "info", skip(quantization_config), fields(
        dim = dim,
        space = %space,
        m = m,
        ef_construction = ef_construction,
        expected_size = expected_size,
        has_quantization = quantization_config.is_some()
    ))]
    pub(crate) fn build(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
        quantization_config: Option<&Bound<PyDict>>,
    ) -> PyResult<Self> {
        let start_time = Instant::now();

        // Validation of parameters
        if dim == 0 {
            error!(
                operation = "validation",
                field = "dim",
                value = dim,
                "Invalid dimension"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dim must be positive",
            ));
        }
        if ef_construction == 0 {
            error!(
                operation = "validation",
                field = "ef_construction",
                value = ef_construction,
                "Invalid ef_construction"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ef_construction must be positive",
            ));
        }
        if expected_size == 0 {
            error!(
                operation = "validation",
                field = "expected_size",
                value = expected_size,
                "Invalid expected_size"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "expected_size must be positive",
            ));
        }
        if expected_size > MAX_EXPECTED_SIZE {
            error!(
                operation = "validation",
                field = "expected_size",
                value = expected_size,
                max_allowed = MAX_EXPECTED_SIZE,
                "expected_size exceeds maximum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "expected_size must be at most {}, got {}. The graph reserves one \
                 slot per declared record at creation, 8 bytes each, so this \
                 declaration would ask for {:.1} GB before a single record is \
                 added. That allocation is not fallible: above this bound the \
                 process aborts rather than raising. expected_size is a capacity \
                 hint and not a limit, and under-declaring only costs some \
                 reallocation, so declare what you expect to hold.",
                MAX_EXPECTED_SIZE,
                expected_size,
                (expected_size as f64 * 8.0) / 1_000_000_000.0
            )));
        }
        if m < 2 {
            error!(
                operation = "validation",
                field = "m",
                value = m,
                min_allowed = 2,
                "m below minimum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "m must be at least 2, got {}. Layer assignment samples from a \
                 scale of 1 / ln(m), which is infinity at m 1, so every point \
                 overflows the layer cap and is redispatched uniformly across all \
                 16 layers instead of following the exponential distribution the \
                 graph depends on. Measured on 3,000 records of 32 dimensions, \
                 recall at 10 was 0.0220 at m 1 against 0.6880 at m 2 and 1.0000 \
                 at m 16.",
                m
            )));
        }
        if m > 256 {
            error!(
                operation = "validation",
                field = "m",
                value = m,
                max_allowed = 256,
                "m exceeds maximum"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "m must be less than or equal to 256",
            ));
        }

        // Early space validation with user-friendly error
        let space_normalized = space.to_lowercase();
        match space_normalized.as_str() {
            "cosine" | "l2" | "l1" => {
                debug!(operation = "validation", space = %space_normalized, "Distance space validated");
            }
            _ => {
                error!(operation = "validation", field = "space", value = %space, "Unsupported distance space");
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Unsupported space: '{}'. Supported spaces: 'cosine', 'l2', 'l1'",
                    space
                )));
            }
        }

        // Extract quantization configuration
        let (quantization_params, pq_instance) = if let Some(config) = quantization_config {
            let qtype = config
                .get_item("type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'type' in quantization_config",
                    )
                })?
                .extract::<String>()?;

            if qtype != "pq" {
                error!(operation = "validation", field = "quantization_type", value = %qtype, "Unsupported quantization type");
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported quantization type: '{}'. Only 'pq' is currently supported.",
                    qtype
                )));
            }

            // Extract PQ parameters
            let subvectors = config
                .get_item("subvectors")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'subvectors' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let bits = config
                .get_item("bits")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'bits' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let training_size = config
                .get_item("training_size")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'training_size' in quantization_config",
                    )
                })?
                .extract::<usize>()?;

            let max_training_vectors = config
                .get_item("max_training_vectors")?
                .map(|v| v.extract::<usize>())
                .transpose()?;

            // Extract storage_mode
            let storage_mode_str = config
                .get_item("storage_mode")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "quantized_only".to_string());

            let storage_mode = StorageMode::from_string(&storage_mode_str)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            // Validate PQ parameters
            if subvectors == 0 {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    value = subvectors,
                    "Subvectors must be positive"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "subvectors must be a positive integer, got 0",
                ));
            }

            if subvectors > dim {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    dim = dim,
                    subvectors = subvectors,
                    "Subvectors exceed dimension"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "subvectors ({}) cannot exceed dimension ({})",
                    subvectors, dim
                )));
            }

            if !dim.is_multiple_of(subvectors) {
                error!(
                    operation = "validation",
                    field = "subvectors",
                    dim = dim,
                    subvectors = subvectors,
                    "Subvectors must divide dimension evenly"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "subvectors ({}) must divide dimension ({}) evenly",
                    subvectors, dim
                )));
            }

            if !(1..=8).contains(&bits) {
                error!(
                    operation = "validation",
                    field = "bits",
                    value = bits,
                    min = 1,
                    max = 8,
                    "Bits out of range"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bits must be between 1 and 8, got {}",
                    bits
                )));
            }

            if training_size < 1000 {
                error!(
                    operation = "validation",
                    field = "training_size",
                    value = training_size,
                    min = 1000,
                    "Training size too small"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "training_size must be at least 1000, got {}",
                    training_size
                )));
            }

            // A max below the threshold produces an index that reaches its
            // training threshold and then fails training on every record from
            // then on, because the cap is already exceeded by the time the
            // trigger fires. Enforced here so it holds on every construction
            // path rather than only the Python factory.
            if let Some(max_training) = max_training_vectors {
                if max_training < training_size {
                    error!(
                        operation = "validation",
                        field = "max_training_vectors",
                        value = max_training,
                        training_size = training_size,
                        "max_training_vectors below training_size"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "max_training_vectors ({}) must be >= training_size ({})",
                        max_training, training_size
                    )));
                }
            }

            let config = QuantizationConfig {
                subvectors,
                bits,
                training_size,
                max_training_vectors,
                storage_mode,
            };

            debug!(
                operation = "pq_configuration",
                subvectors = subvectors,
                bits = bits,
                training_size = training_size,
                storage_mode = %storage_mode_str,
                sub_dim = dim / subvectors,
                num_centroids = 1 << bits,
                "Product Quantization configured"
            );

            // Create PQ instance
            let pq = Arc::new(PQ::new(
                dim,
                subvectors,
                bits,
                training_size,
                max_training_vectors,
            ));

            (Some(config), Some(pq))
        } else {
            (None, None)
        };

        let max_layer = 16; // Always use NB_LAYER_MAX for hnsw-rs compatibility
        trace!(
            operation = "hnsw_config",
            max_layer = max_layer,
            reason = "hnsw-rs compatibility",
            "Using fixed max_layer"
        );

        // Create initial raw HNSW index (will be rebuilt as PQ after training)
        let hnsw = DistanceType::new_raw(
            &space_normalized,
            m,
            expected_size,
            max_layer,
            ef_construction,
        );

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "index_creation_complete",
            dim = dim,
            space = %space_normalized,
            m = m,
            ef_construction = ef_construction,
            expected_size = expected_size,
            has_quantization = quantization_params.is_some(),
            duration_ms = duration_ms,
            "HNSW index created successfully"
        );

        // Initialize all fields with proper thread-safe wrappers
        Ok(HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            quantization_config: quantization_params,
            pq: pq_instance,
            pq_codes: RwLock::new(HashMap::new()),
            rerank_calibration: RwLock::new(None),
            metadata: Mutex::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: RwLock::new(hnsw),
            writers: Mutex::new(()),
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            created_at: Utc::now().to_rfc3339(),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        })
    }
}

#[pymethods]
impl HNSWIndex {
    /// Get quantization configuration and status
    pub fn get_quantization_info(&self) -> Option<Py<PyAny>> {
        Python::attach(|py| {
            if let Some(config) = &self.quantization_config {
                let dict = PyDict::new(py);
                dict.set_item("type", "pq").ok()?;
                dict.set_item("subvectors", config.subvectors).ok()?;
                dict.set_item("bits", config.bits).ok()?;
                dict.set_item("training_size", config.training_size).ok()?;

                if let Some(max_training) = config.max_training_vectors {
                    dict.set_item("max_training_vectors", max_training).ok()?;
                }

                if let Some(pq) = &self.pq {
                    dict.set_item("is_trained", pq.is_trained()).ok()?;

                    // Use enhanced PQ methods
                    let (memory_mb, total_centroids) = pq.get_memory_stats();
                    dict.set_item("memory_mb", memory_mb).ok()?;
                    dict.set_item("total_centroids", total_centroids).ok()?;

                    // The symmetric distance table graph construction reads.
                    // Reported separately because it is derived from the
                    // codebook rather than part of it, and because it scales
                    // with subvectors and bits alone while memory_mb scales
                    // with the dimension too.
                    dict.set_item(
                        "sdc_memory_mb",
                        pq.sdc_memory_bytes() as f64 / (1024.0 * 1024.0),
                    )
                    .ok()?;

                    // Calculate compression ratio using cached values
                    let original_bytes = pq.dim * 4; // f32
                    let compressed_bytes = pq.subvectors; // u8 per subvector
                    let compression_ratio = original_bytes as f64 / compressed_bytes as f64;
                    dict.set_item("compression_ratio", compression_ratio).ok()?;
                }

                Some(dict.into())
            } else {
                None
            }
        })
    }

    /// Check if quantization is enabled
    pub fn has_quantization(&self) -> bool {
        self.quantization_config.is_some()
    }

    /// Get current vector count (for monitoring training trigger)
    pub fn get_vector_count(&self) -> usize {
        *self.vector_count.lock().unwrap()
    }

    /// Get the distance space configuration
    pub fn get_space(&self) -> String {
        self.space.clone()
    }

    /// Rebuild the HNSW index to use PQ codes after training is complete
    ///
    /// Re-encodes whatever raw vectors the index still holds through the
    /// trained codebook and rebuilds the graph from the stored codes. It never
    /// retrains the codebook; training runs exactly once, on the `add` that
    /// reaches `training_size`. A trained `quantized_only` index holds no raw
    /// vectors, so there the rebuild proceeds from the codes alone, and under
    /// either mode nothing is lost by calling it. Returns false when there is
    /// no trained quantizer or nothing stored to rebuild from.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization()
    ), err)]
    pub fn rebuild_with_quantization(&self, py: Python<'_>) -> PyResult<bool> {
        // The whole rebuild runs with the interpreter lock released, the mutation
        // guard included. Waiting for another writer while holding the lock would
        // stall every Python thread in the process for the length of that writer,
        // which is the failure `add` releasing the lock would otherwise create.
        py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            self.rebuild_with_quantization_locked()
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Check if the index is using quantized search
    pub fn is_quantized(&self) -> bool {
        if let Some(pq) = &self.pq {
            if pq.is_trained() {
                let hnsw_guard = self.hnsw.read().unwrap();
                return hnsw_guard.is_quantized();
            }
        }
        false
    }

    /// Check if quantization can be used (PQ is trained)
    pub fn can_use_quantization(&self) -> bool {
        if let Some(pq) = &self.pq {
            pq.is_trained()
        } else {
            false
        }
    }

    /// Enhanced add method that properly handles PQ overwrite scenarios
    #[pyo3(signature = (data, overwrite = true))]
    #[instrument(level = "info", skip(self, data), fields(
        overwrite = overwrite,
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn add(&self, data: Bound<PyAny>, overwrite: bool) -> PyResult<AddResult> {
        let start_time = Instant::now();

        // Input validation
        if data.is_none() {
            error!(
                operation = "add_vectors",
                error = "data_is_none",
                "Data cannot be None"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data cannot be None",
            ));
        }

        // Use error-collecting parsing
        let (parsed_data, parse_errors) = self.parse_input_data(&data);

        let mut total_inserted = 0;
        let mut total_errors = 0;
        let mut errors = Vec::new();

        // Add parse errors to the collection
        for parse_error in parse_errors {
            errors.push(parse_error);
            total_errors += 1;
        }

        if parsed_data.is_empty() && errors.is_empty() {
            trace!(
                operation = "add_vectors",
                result = "empty_input",
                "No vectors to process"
            );
            return Ok(AddResult {
                total_inserted: 0,
                total_errors: 0,
                errors: vec![],
                vector_shape: Some((0, self.dim)),
            });
        }

        let total_input_count = parsed_data.len() + total_errors;
        let vector_shape = Some((total_input_count, self.dim));

        debug!(
            operation = "add_vectors_start",
            total_vectors = parsed_data.len(),
            parse_errors = total_errors,
            overwrite = overwrite,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            storage_mode = self.get_storage_mode(),
            "Starting vector addition"
        );

        // Parsing is the whole of what reads Python objects, and it is done.
        // Everything below works on `parsed_data`, which is owned Rust, so the
        // insertion phase runs with the interpreter lock released. The mutation
        // guard is taken inside that region rather than above it, so a caller
        // waiting for another writer waits without the lock. Holding it while
        // waiting would stall every Python thread in the process for the length
        // of the writer ahead, which is the failure this change would otherwise
        // introduce in place of the one it removes.
        //
        // `insert_parsed_records` carries the proof that nothing inside touches
        // Python.
        let py = data.py();
        let (inserted, insert_errors) = py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            self.insert_parsed_records(parsed_data, overwrite)
        });
        total_inserted += inserted;

        // The errors come back in the order they happened. Two of the three
        // variants carry a message Rust already built. The third carries a
        // `PyErr`, which is formatted here because `PyErr`'s `Display` acquires
        // the interpreter lock and so could not run above.
        for insert_error in insert_errors {
            match insert_error {
                InsertError::Counted(message) => {
                    errors.push(message);
                    total_errors += 1;
                }
                InsertError::Training(message) => {
                    errors.push(message);
                }
                InsertError::Vector { id, err } => {
                    trace!(
                        operation = "add_vector_error",
                        vector_id = %id,
                        error = %err,
                        "Vector addition failed"
                    );
                    errors.push(format!("Vector {}: {}", id, err));
                    total_errors += 1;
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "add_vectors_complete",
            total_inserted = total_inserted,
            total_errors = total_errors,
            success_rate = if total_input_count > 0 {
                total_inserted as f64 / total_input_count as f64 * 100.0
            } else {
                100.0
            },
            duration_ms = duration_ms,
            overwrite_mode = overwrite,
            final_storage_mode = self.get_storage_mode(),
            "Vector addition completed"
        );

        self.warn_if_outgrown_expected_size();

        Ok(AddResult {
            total_inserted,
            total_errors,
            errors,
            vector_shape,
        })
    }

    pub fn get_training_progress(&self) -> f32 {
        if let Some(config) = &self.quantization_config {
            // If PQ is trained, always return 100%
            if let Some(pq) = &self.pq {
                if pq.is_trained() {
                    return 100.0;
                }
            }
            let training_ids = self.training_ids.read().unwrap();
            (training_ids.len() as f32 / config.training_size as f32 * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Get number of training vectors still needed
    pub fn training_vectors_needed(&self) -> usize {
        if let Some(config) = &self.quantization_config {
            if self.training_threshold_reached.load(Ordering::Acquire) {
                0
            } else {
                let training_ids = self.training_ids.read().unwrap();
                config.training_size.saturating_sub(training_ids.len())
            }
        } else {
            0
        }
    }

    /// Check if training is ready to be triggered
    pub fn is_training_ready(&self) -> bool {
        self.training_threshold_reached.load(Ordering::Acquire)
    }

    /// Get current storage mode description
    pub fn get_storage_mode(&self) -> String {
        if !self.has_quantization() {
            "raw_only".to_string()
        } else if !self.can_use_quantization() {
            if self.training_threshold_reached.load(Ordering::Acquire) {
                "raw_ready_for_training".to_string()
            } else {
                "raw_collecting_for_training".to_string()
            }
        } else if self.is_quantized() {
            "quantized_active".to_string()
        } else {
            "raw_trained_not_rebuilt".to_string()
        }
    }

    /// Enhanced search method with automatic ADC usage
    ///
    /// `rerank` controls how far a quantized search over-fetches before it
    /// rescores the candidates against raw vectors. Omitted, the fetch is
    /// derived from the live record count; see `SearchParams::fetch_k`. An
    /// integer of 1 or more pulls that many candidates per requested result,
    /// which is a fixed multiple of the page and does not move with the corpus.
    /// Zero turns rerank off and restores the ADC scores and ordering. It has
    /// no effect on a raw index or on a `quantized_only` one, both of which
    /// never rerank; see `rerank_plan`.
    ///
    /// `ef_search` has no effect on a reranked quantized search. Below the
    /// fetch it is discarded, because `Hnsw::search_filter` raises the
    /// traversal width to the number of neighbours asked for and cannot return
    /// more results than its candidate list holds. Above the fetch it buys no
    /// recall, because the candidates a fetch returns are limited by the ADC
    /// ordering rather than by the traversal, and quadrupling it moves recall
    /// at 10 by at most 0.008. The default fetch is at least 250 and the
    /// default `ef_search` is 100, so changing `ef_search` alone changes
    /// nothing on a reranked search at the defaults.
    // The argument list is the Python signature, so it is not free to be
    // bundled the way the internal batch paths bundle theirs.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (vector, filter=None, top_k=10, ef_search=None, return_vector=false, rerank=None))]
    #[instrument(level = "debug", skip(self, py, vector, filter), fields(
        top_k = top_k,
        ef_search = ef_search,
        return_vector = return_vector,
        rerank = rerank,
        is_quantized = self.is_quantized()
    ), err)]
    pub fn search(
        &self,
        py: Python<'_>,
        vector: Bound<PyAny>,
        filter: Option<&Bound<PyDict>>,
        top_k: usize,
        ef_search: Option<usize>,
        return_vector: bool,
        rerank: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let start_time = Instant::now();

        let ef = ef_search.unwrap_or_else(|| match self.space.to_lowercase().as_str() {
            "l1" | "l2" => std::cmp::max(2 * top_k, 150),
            _ => std::cmp::max(2 * top_k, 100),
        });

        // Resolved once here rather than per query, because it locks the graph
        // to read whether the index is quantized and the batch paths take that
        // lock themselves.
        let params = SearchParams {
            top_k,
            ef,
            return_vector,
            rerank: self.rerank_plan(rerank),
        };

        trace!(
            operation = "search_config",
            ef = ef,
            space = %self.space,
            rerank_factor = params.rerank.and_then(|plan| plan.factor),
            "Search parameters configured"
        );

        let filter_conditions = filter
            .map(|f| self.python_dict_to_value_map(f))
            .transpose()?;

        // Reject an unrecognised operator before the search runs. Checking it
        // per record would make the error depend on the data, because a record
        // that lacks the field never reaches the operator at all.
        if let Some(conditions) = filter_conditions.as_ref() {
            self.validate_filter_conditions(conditions)?;
        }

        // Detect batch vs single query with comprehensive input support
        let result: Py<PyAny> = if let Ok(list_vec) = vector.extract::<Vec<Vec<f32>>>() {
            // Format: List of vectors [[0.1, 0.2], [0.3, 0.4]]

            // Validation for empty batch or empty vectors in batch
            if list_vec.is_empty() {
                error!(
                    operation = "search",
                    error = "empty_batch",
                    "Batch cannot be empty"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Batch cannot be empty",
                ));
            }

            // Check for empty vectors within the batch
            for (i, vec) in list_vec.iter().enumerate() {
                if vec.is_empty() {
                    error!(
                        operation = "search",
                        error = "empty_vector_in_batch",
                        vector_index = i,
                        "Vector in batch cannot be empty"
                    );
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Vector {} in batch cannot be empty",
                        i
                    )));
                }
            }

            debug!(
                operation = "batch_search",
                batch_size = list_vec.len(),
                "Starting batch search"
            );
            let results =
                self.batch_search_internal(&list_vec, filter_conditions.as_ref(), params, py)?;
            PyList::new(py, results)?.into()
        } else if let Ok(np_array) = vector.cast::<PyArray2<f32>>() {
            // Format: NumPy 2D array (N, dims)
            let readonly = np_array.readonly();
            let shape = readonly.shape();

            if shape.len() != 2 || shape[1] != self.dim {
                error!(
                    operation = "search",
                    error = "shape_mismatch",
                    expected_shape = format!("(N, {})", self.dim),
                    actual_shape = format!("{:?}", shape),
                    "NumPy array shape mismatch"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "NumPy array must have shape (N, {}), got {:?}",
                    self.dim, shape
                )));
            }

            let flat = readonly.as_slice()?;
            let batch: Vec<Vec<f32>> = flat.chunks(self.dim).map(|chunk| chunk.to_vec()).collect();
            debug!(
                operation = "batch_search_numpy",
                batch_size = batch.len(),
                "Starting NumPy batch search"
            );
            let results =
                self.batch_search_internal(&batch, filter_conditions.as_ref(), params, py)?;
            PyList::new(py, results)?.into()
        } else {
            // Single vector path - enhanced with NumPy 1D support
            let query_vector = if let Ok(array1d) = vector.cast::<PyArray1<f32>>() {
                array1d.readonly().as_slice()?.to_vec()
            } else {
                vector.extract::<Vec<f32>>()?
            };

            // PROCESS HERE using extract_single_vector logic
            let processed_query = self.validate_and_process_query_vector(query_vector)?;

            trace!(
                operation = "single_search",
                query_dim = processed_query.len(),
                "Starting single vector search"
            );

            let search_results = py.detach(|| -> PyResult<QueryHits> {
                // Check if we should use quantized search
                let use_quantized = self.is_quantized();

                trace!(
                    operation = "search_method",
                    use_quantized = use_quantized,
                    "Selected search method"
                );

                // One read guard for the whole search, taken before the graph lock
                // and held across it. The predicate runs once per candidate the
                // traversal visits, so acquiring the guard inside it would put a
                // lock acquisition on the hot path. See the same pattern in the
                // two batch paths.
                let rev_map = self.rev_map.read().unwrap();
                let live = |internal_id: &usize| rev_map.contains_key(internal_id);

                let fetch_k = params.fetch_k(rev_map.len());

                let hnsw_results = {
                    let hnsw_guard = self.hnsw.read().unwrap();

                    if use_quantized {
                        // Use ADC search for quantized index
                        hnsw_guard
                            .search(&processed_query, fetch_k, params.ef, Some(&live))
                            .unwrap_or_else(|e| {
                                error!(operation = "adc_search", error = %e, "ADC search failed");
                                Vec::new()
                            })
                    } else {
                        // Use raw vector search
                        match hnsw_guard.search(&processed_query, fetch_k, params.ef, Some(&live)) {
                            Ok(results) => results,
                            Err(e) => {
                                error!(operation = "raw_search", error = %e, "Raw search failed");
                                Vec::new()
                            }
                        }
                    }
                };

                // Process results with enhanced vector retrieval
                let vectors = self.vectors.read().unwrap();
                let pq_codes = self.pq_codes.read().unwrap();
                let vector_metadata = self.vector_metadata.read().unwrap();

                // Resolve, filter and score first, holding only a borrowed id
                // and a float per candidate. Metadata and vectors are cloned
                // after the cut, so an over-fetched page pays for the results
                // it returns rather than for every candidate it considered.
                let mut scored: Vec<(&String, f32)> = Vec::with_capacity(hnsw_results.len());
                let has_filter = filter_conditions.is_some();

                for neighbor in hnsw_results {
                    let internal_id = neighbor.get_origin_id();

                    if let Some(ext_id) = rev_map.get(&internal_id) {
                        if has_filter {
                            if let Some(meta) = vector_metadata.get(ext_id) {
                                let filter_conds = filter_conditions.as_ref().unwrap();
                                if !self.matches_filter(meta, filter_conds)? {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let score = match params.rerank.as_ref() {
                            Some(plan) => rescore_candidate(
                                plan,
                                &processed_query,
                                ext_id,
                                &vectors,
                                self.pq.as_ref(),
                                &pq_codes,
                            )
                            .unwrap_or(f32::INFINITY),
                            None => neighbor.distance,
                        };

                        scored.push((ext_id, score));
                    }
                }

                if params.rerank.is_some() {
                    take_best(&mut scored, top_k);
                }

                let mut results = Vec::with_capacity(scored.len());
                for (ext_id, score) in scored {
                    let metadata = vector_metadata.get(ext_id).cloned().unwrap_or_default();
                    let vector_data = if return_vector {
                        // Try raw vector first, then PQ reconstruction
                        vectors.get(ext_id).cloned().or_else(|| {
                            if let (Some(pq), Some(codes)) = (&self.pq, pq_codes.get(ext_id)) {
                                pq.reconstruct(codes).ok()
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    };

                    results.push((ext_id.clone(), score, metadata, vector_data));
                }

                Ok(results)
            })?;

            // Convert to Python objects
            let mut output: Vec<Py<PyDict>> = Vec::with_capacity(search_results.len());
            for (id, score, metadata, vector_data) in search_results {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;
                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }
                output.push(dict.into());
            }

            PyList::new(py, output)?.into()
        };

        // ✅ ENTERPRISE: Add duration timing to hot path with actual result count
        let duration_ms = start_time.elapsed().as_millis();
        let results_count = {
            let any = result.bind(py);
            match any.cast::<PyList>() {
                Ok(list) => list.len(),
                Err(_) => 0,
            }
        };

        debug!(
            operation = "search_complete",
            results_count = results_count,
            duration_ms = duration_ms,
            "Search completed"
        );

        Ok(result)
    }

    /// Enhanced Save method to include HNSW Graph
    ///
    /// The whole save runs with the interpreter lock released. `save_index`
    /// reaches `save_config`, `save_mappings`, `save_metadata`,
    /// `save_quantization_config`, `save_pq_centroids`, `save_pq_codes`,
    /// `save_vectors` and `save_manifest`, and every one of them speaks only to
    /// `serde_json`, `bincode` and `std::fs`. Every Python token in
    /// `persistence.rs` sits in the load path, in `rebuild_using_add_method` and
    /// `convert_json_value_to_python`. `save_hnsw_graph` calls the vendored
    /// crate's `file_dump`, which names PyO3 nowhere.
    #[instrument(level = "info", skip(self, py), fields(
        vector_count = self.get_vector_count(),
        has_quantization = self.has_quantization(),
        is_quantized = self.is_quantized()
    ), err)]
    pub fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.save_locked(path))
    }

    /// Python property: `index.dim`
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get records by ID(s) with PQ reconstruction support and storage mode awareness
    ///
    /// Looks the ids up in the union of the raw vectors and the quantized codes,
    /// so it already saw every record before the other accessors did. An id that
    /// resolves to no record is dropped from the result rather than reported, so
    /// the returned list can be shorter than the list of ids asked for.
    ///
    /// `return_vector` is served from the raw vector where one exists and from a
    /// reconstruction of the code where one does not. Under `quantized_only`
    /// that is every record once training completes, the training records
    /// included, since the rebuild releases their raw vectors the moment their
    /// codes are stored. The returned value is then an approximation rather
    /// than the value supplied. Measured on 16 dimensional data with 4
    /// subvectors and 8 bits, a reconstructed vector differed from the stored
    /// unit vector by 0.066 at the worst component and sat at cosine similarity
    /// 0.991 to it. Under `quantized_with_raw` every record keeps its raw
    /// vector and returns exactly. `get_stats()["raw_vectors_stored"]` is what
    /// tells the two apart in aggregate.
    #[pyo3(signature = (input, return_vector = true))]
    pub fn get_records(
        &self,
        py: Python<'_>,
        input: &Bound<PyAny>,
        return_vector: bool,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let ids: Vec<String> = if let Ok(id_str) = input.extract::<String>() {
            vec![id_str]
        } else if let Ok(id_list) = input.extract::<Vec<String>>() {
            id_list
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a string or a list of strings for ID(s)",
            ));
        };

        trace!(
            operation = "get_records",
            record_count = ids.len(),
            return_vector = return_vector,
            "Retrieving records"
        );

        let mut records = Vec::with_capacity(ids.len());

        // Use read locks for concurrent access
        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        for id in ids {
            // Check if this ID exists in either storage
            let exists = vectors.contains_key(&id) || pq_codes.contains_key(&id);

            if exists {
                let metadata = vector_metadata.get(&id).cloned().unwrap_or_default();

                let dict = PyDict::new(py);
                dict.set_item("id", id.clone())?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if return_vector {
                    // Priority: raw vector > PQ reconstruction
                    let vector_data = if let Some(raw_vector) = vectors.get(&id) {
                        // Case 1: Raw vector available (QuantizedWithRaw mode or non-quantized)
                        Some(raw_vector.clone())
                    } else if let (Some(pq), Some(codes)) = (&self.pq, pq_codes.get(&id)) {
                        // Case 2: Only quantized codes available (QuantizedOnly mode)
                        match pq.reconstruct(codes) {
                            Ok(reconstructed) => Some(reconstructed),
                            Err(e) => {
                                warn!(operation = "vector_reconstruction", vector_id = %id, error = %e, "Failed to reconstruct vector");
                                None
                            }
                        }
                    } else {
                        // Case 3: No vector data available
                        None
                    };

                    if let Some(vec) = vector_data {
                        dict.set_item("vector", vec)?;
                    }
                }

                records.push(dict.into());
            }
        }

        trace!(
            operation = "get_records_complete",
            found_records = records.len(),
            "Records retrieval completed"
        );
        Ok(records)
    }

    /// Enhanced get_stats with storage mode information
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();

        // Nodes the graph holds, which exceeds the live record count by exactly the
        // number of nodes removal and overwrite have stranded. `compact` reclaims the
        // difference. Read first, because the declared lock order puts the graph
        // above every map read below it.
        let (graph_nodes, graph_memory_mb) = {
            let hnsw = self.hnsw.read().unwrap();
            (
                hnsw.nb_points(),
                hnsw.memory_bytes() as f64 / (1024.0 * 1024.0),
            )
        };

        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let training_ids = self.training_ids.read().unwrap();
        let vector_count = *self.vector_count.lock().unwrap();

        // Basic stats
        stats.insert("total_vectors".to_string(), vector_count.to_string());
        stats.insert("dimension".to_string(), self.dim.to_string());
        stats.insert("expected_size".to_string(), self.expected_size.to_string());
        stats.insert("space".to_string(), self.space.clone());
        stats.insert("index_type".to_string(), "HNSW".to_string());

        stats.insert("m".to_string(), self.m.to_string());
        stats.insert(
            "ef_construction".to_string(),
            self.ef_construction.to_string(),
        );
        stats.insert("thread_safety".to_string(), "RwLock+Mutex".to_string());

        stats.insert("graph_nodes".to_string(), graph_nodes.to_string());
        stats.insert(
            "stranded_graph_nodes".to_string(),
            graph_nodes.saturating_sub(vector_count).to_string(),
        );

        // The memory keys, reported on every index rather than only on a
        // quantized one. `raw_vectors_memory_mb` used to sit inside the
        // quantization branch, so an unquantized index reported no memory at
        // all, and `graph_memory_mb` did not exist, so no index reported the
        // largest thing it holds. Both are additions. Nothing that reads a key
        // this call already returned sees a different value, which matters
        // because the langchain adapter forwards `memory_mb` from
        // `get_quantization_info` verbatim and that key is untouched.
        let raw_memory_mb = (vectors.len() * self.dim * 4) as f64 / (1024.0 * 1024.0);
        let mut total_memory_mb = graph_memory_mb + raw_memory_mb;
        stats.insert(
            "graph_memory_mb".to_string(),
            format!("{:.2}", graph_memory_mb),
        );
        stats.insert(
            "raw_vectors_memory_mb".to_string(),
            format!("{:.2}", raw_memory_mb),
        );

        // Storage breakdown
        stats.insert("raw_vectors_stored".to_string(), vectors.len().to_string());
        stats.insert(
            "quantized_codes_stored".to_string(),
            pq_codes.len().to_string(),
        );

        // Training info
        if let Some(config) = &self.quantization_config {
            stats.insert("quantization_type".to_string(), "pq".to_string());
            stats.insert(
                "quantization_training_size".to_string(),
                config.training_size.to_string(),
            );

            // Storage mode information
            stats.insert(
                "storage_mode".to_string(),
                config.storage_mode.to_string().to_string(),
            );

            // Calculate actual memory usage based on storage mode. The raw
            // vector figure is reported above, on every index rather than only
            // on this branch.
            let quantized_memory_mb =
                (pq_codes.len() * config.subvectors) as f64 / (1024.0 * 1024.0);
            total_memory_mb += quantized_memory_mb;

            stats.insert(
                "quantized_codes_memory_mb".to_string(),
                format!("{:.2}", quantized_memory_mb),
            );

            // `memory_savings` used to sit beside `storage_strategy`, reading
            // "maximum" under QuantizedOnly. That mode is the smaller of the two
            // and it is not a maximum of anything. Measured at 3,000 records of
            // 64 dimensions it held more resident memory than the same index
            // unquantized, because the centroid distance table is a fixed 1 MB.
            // What replaces it is the fact the figures above can be checked
            // against, being which records still have a raw vector. Under
            // QuantizedOnly that is every record while the index is still
            // collecting for training and none from the moment training
            // completes, since the rebuild releases the training records once
            // their codes are stored.
            match config.storage_mode {
                StorageMode::QuantizedOnly => {
                    stats.insert(
                        "storage_strategy".to_string(),
                        "memory_optimized".to_string(),
                    );
                    stats.insert(
                        "raw_vectors_retained".to_string(),
                        "none_once_trained".to_string(),
                    );
                }
                StorageMode::QuantizedWithRaw => {
                    stats.insert(
                        "storage_strategy".to_string(),
                        "quality_optimized".to_string(),
                    );
                    stats.insert(
                        "raw_vectors_retained".to_string(),
                        "all_records".to_string(),
                    );
                }
            }

            let collected_count = training_ids.len();
            let progress = self.get_training_progress();
            stats.insert(
                "training_progress".to_string(),
                format!(
                    "{}/{} ({:.1}%)",
                    collected_count, config.training_size, progress
                ),
            );

            let vectors_needed = self.training_vectors_needed();
            stats.insert(
                "training_vectors_needed".to_string(),
                vectors_needed.to_string(),
            );
            stats.insert(
                "training_threshold_reached".to_string(),
                self.training_threshold_reached
                    .load(Ordering::Acquire)
                    .to_string(),
            );

            if let Some(pq) = &self.pq {
                let is_trained = pq.is_trained();
                stats.insert("quantization_trained".to_string(), is_trained.to_string());
                stats.insert(
                    "quantization_active".to_string(),
                    self.is_quantized().to_string(),
                );

                // The two fixed costs, reported here so that the whole memory
                // question can be answered from one call. Both are independent
                // of the record count, and at small record counts the table is
                // the largest single thing a quantized index holds. They were
                // only on `get_quantization_info`, which is where a caller
                // reading the storage breakdown above would not look.
                let (centroid_mb, _) = pq.get_memory_stats();
                let sdc_mb = pq.sdc_memory_bytes() as f64 / (1024.0 * 1024.0);
                total_memory_mb += centroid_mb + sdc_mb;
                stats.insert(
                    "codebook_memory_mb".to_string(),
                    format!("{:.2}", centroid_mb),
                );
                stats.insert("sdc_table_memory_mb".to_string(), format!("{:.2}", sdc_mb));

                if is_trained {
                    let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
                    stats.insert(
                        "quantization_compression_ratio".to_string(),
                        format!("{:.1}x", compression_ratio),
                    );
                }

                // What the default rerank fetch is derived from, and what it
                // resolves to at the record count the index holds now, so a
                // caller can see the number their searches are paying for
                // rather than deriving it. See `RerankCalibration`.
                match self.get_rerank_calibration() {
                    Some(calibration) => {
                        let live = self.id_map.read().unwrap().len();
                        stats.insert("rerank_calibrated".to_string(), "true".to_string());
                        stats.insert(
                            "rerank_calibration_fetch".to_string(),
                            calibration.fetch.to_string(),
                        );
                        stats.insert(
                            "rerank_calibration_records".to_string(),
                            calibration.sample_records.to_string(),
                        );
                        stats.insert(
                            "rerank_calibration_queries".to_string(),
                            calibration.queries.to_string(),
                        );
                        stats.insert(
                            "rerank_calibration_target_recall".to_string(),
                            format!("{:.3}", calibration.target),
                        );
                        stats.insert(
                            "rerank_calibration_fit_fetches".to_string(),
                            calibration
                                .fit_fetches
                                .iter()
                                .map(|f| f.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                        stats.insert(
                            "rerank_calibration_exponent".to_string(),
                            format!("{:.3}", calibration.exponent),
                        );
                        stats.insert(
                            "rerank_calibration_page_fetches".to_string(),
                            calibration
                                .page_fetches
                                .iter()
                                .map(|f| f.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                        stats.insert(
                            "rerank_calibration_pages".to_string(),
                            RERANK_CALIBRATION_PAGES
                                .iter()
                                .map(|p| p.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        );
                        stats.insert(
                            "rerank_calibration_page_exponent".to_string(),
                            format!("{:.3}", calibration.page_exponent),
                        );
                        stats.insert(
                            "rerank_calibration_ms".to_string(),
                            calibration.millis.to_string(),
                        );
                        stats.insert(
                            "rerank_default_fetch".to_string(),
                            calibration
                                .fetch_at(live, RERANK_CALIBRATION_TOP_K)
                                .min(live.max(RERANK_CALIBRATION_TOP_K))
                                .to_string(),
                        );
                    }
                    None => {
                        let live = self.id_map.read().unwrap().len();
                        stats.insert("rerank_calibrated".to_string(), "false".to_string());
                        stats.insert(
                            "rerank_default_fetch".to_string(),
                            (live / DEFAULT_RERANK_CORPUS_DIVISOR)
                                .max(DEFAULT_RERANK_MIN_CANDIDATES)
                                .max(RERANK_CALIBRATION_TOP_K * DEFAULT_RERANK_PAGE_FACTOR)
                                .min(live.max(RERANK_CALIBRATION_TOP_K))
                                .to_string(),
                        );
                    }
                }
            }
        } else {
            stats.insert("quantization_type".to_string(), "none".to_string());
            stats.insert("storage_mode".to_string(), "raw_only".to_string());
        }

        stats.insert(
            "storage_mode_description".to_string(),
            self.get_storage_mode(),
        );

        // The sum of the five memory keys above. It is what the index holds in
        // the structures this call can price, being the graph, the raw vector
        // store, the codes, the codebook and the centroid distance table.
        //
        // It is not the resident set. The id maps, the metadata map, the hash
        // table slots and the allocator's own headers and fragmentation sit
        // outside it. Measured on three loaded indexes of 50,000
        // dbpedia-openai records at dimension 1,536, the process held 805.9,
        // 474.8 and 181.4 MiB where this reports 692.4, 401.2 and 107.8, being
        // 1.16, 1.18 and 1.68 times. The share it misses is roughly 1,500 bytes
        // per record and it does not move with the dimension, so it dominates
        // the ratio on the mode that holds least.
        stats.insert(
            "total_memory_mb".to_string(),
            format!("{:.2}", total_memory_mb),
        );

        stats
    }

    /// List the first number of records in the index (ID and metadata)
    ///
    /// Enumerates `id_map`, which holds every live record. It used to enumerate
    /// `vectors`, which under `quantized_only` holds only the records collected
    /// before training, so every record added afterwards was missing from the
    /// listing while search still returned it.
    ///
    /// Iteration order is a hash map's and is not stable between calls, so
    /// `number` takes an arbitrary N rather than a defined page.
    #[pyo3(signature = (number=10))]
    pub fn list(&self, py: Python<'_>, number: usize) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let id_map = self.id_map.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();

        let mut results = Vec::new();
        for id in id_map.keys().take(number) {
            let metadata = vector_metadata.get(id).cloned().unwrap_or_default();
            let py_metadata = self.value_map_to_python(&metadata, py)?;
            results.push((id.clone(), py_metadata));
        }
        Ok(results)
    }

    /// Check whether a record with this id is in the index
    ///
    /// Reads `id_map`, which is the record set. Every insertion path writes it,
    /// `remove_point_internal` keys its removal on it, `add(overwrite=True)`
    /// keys its collision test on it, and `compact` rebuilds the graph from it.
    /// It used to read `vectors`, which under `quantized_only` holds only the
    /// records collected before training, so this returned `false` for a record
    /// that search returned and `remove_point` removed.
    pub fn contains(&self, id: String) -> bool {
        let id_map = self.id_map.read().unwrap();
        id_map.contains_key(&id)
    }

    /// Add index-level metadata
    pub fn add_metadata(&self, metadata: HashMap<String, String>) {
        let mut meta_lock = self.metadata.lock().unwrap();
        for (key, value) in metadata {
            meta_lock.insert(key, value);
        }
    }

    /// Get index-level metadata value
    pub fn get_metadata(&self, key: String) -> Option<String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.get(&key).cloned()
    }

    /// Get all index-level metadata
    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        let meta_lock = self.metadata.lock().unwrap();
        meta_lock.clone()
    }

    /// Get a human-readable info string
    ///
    /// `vectors=` is the live record count, taken from `id_map`. It used to be
    /// `vectors.len()`, which under `quantized_only` counts only the records
    /// that still hold a raw vector and therefore reported fewer records than
    /// the index contains.
    ///
    /// The count is read into a local and the guard dropped before anything
    /// else is touched. `is_quantized()` below takes the graph lock, and the
    /// declared order puts the graph above `vectors`, so holding that guard
    /// across the call was an inversion.
    pub fn info(&self) -> String {
        let record_count = self.id_map.read().unwrap().len();
        let base_info = format!(
            "HNSWIndex(dim={}, space={}, m={}, ef_construction={}, expected_size={}, vectors={}",
            self.dim, self.space, self.m, self.ef_construction, self.expected_size, record_count
        );

        if let Some(config) = &self.quantization_config {
            let trained_status = self
                .pq
                .as_ref()
                .map(|pq| {
                    if pq.is_trained() {
                        "trained"
                    } else {
                        "untrained"
                    }
                })
                .unwrap_or("unknown");

            let active_status = if self.is_quantized() {
                "active"
            } else {
                "inactive"
            };

            // Use cached compression ratio calculation with proper float division
            let compression_info = self
                .pq
                .as_ref()
                .map(|pq| format!("{:.1}x", (pq.dim as f64 * 4.0) / pq.subvectors as f64))
                .unwrap_or_else(|| "unknown".to_string());

            format!(
                "{}, quantization=pq(subvectors={}, bits={}, {}, {}, compression={}))",
                base_info,
                config.subvectors,
                config.bits,
                trained_status,
                active_status,
                compression_info
            )
        } else {
            format!("{}, quantization=none)", base_info)
        }
    }

    /// Remove vector by ID
    /// Public remove_point method (unchanged for API compatibility)
    /// This code delegates to remove_point_internal() which handles all the complex logic
    pub fn remove_point(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        // `id` arrives already converted, and `remove_point_internal` is in the
        // set `insert_parsed_records` verifies, so the whole body is Rust. The
        // removal itself is short, but the wait for the mutation guard is not,
        // because `add` can now hold it for a long insert with the lock released.
        // Waiting here with the lock held would stall every Python thread.
        py.detach(|| {
            let _writers = self.writers.lock().unwrap();
            self.remove_point_internal(id)
        })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Rebuild the graph in memory and reclaim the nodes removal and overwrite strand.
    ///
    /// `remove_point` clears a record from every storage map but cannot delete its
    /// graph node, and `add(overwrite=True)` is a removal followed by an insertion,
    /// so both leave behind a node that still holds a copy of the vector and both
    /// directions of adjacency while resolving to no record. Search already excludes
    /// those nodes, so this is a resource operation and not a correctness one. What it
    /// reclaims is their memory, their edge slots in live neighbour lists, and the
    /// traversal steps they cost.
    ///
    /// Returns the number of nodes reclaimed. Zero means the graph held no stranded
    /// nodes, in which case nothing is rebuilt and the call is a no-op.
    ///
    /// The cost is a full sequential rebuild, proportional to the live record count
    /// rather than to the amount of debris. Nothing outside the graph is touched.
    /// Internal ids, external ids, metadata, stored vectors, quantized codes, PQ
    /// training state and the id counter all survive unchanged, so the record any
    /// given id resolves to is the same before and after.
    ///
    /// The replacement graph is built in full before the old one is dropped, so peak
    /// memory holds both for the duration and a failure part way through leaves the
    /// index exactly as it was.
    ///
    /// This is never automatic. Calling it is a decision a deployment can schedule.
    ///
    /// The rebuild runs with the interpreter lock released. Every function it
    /// reaches is in the set `insert_parsed_records` verifies, plus
    /// `DistanceType::new_raw` and `insert_batch`, which are the same shape as
    /// the quantized pair already listed there.
    pub fn compact(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.compact_locked())
    }

    /// Get performance characteristics and limitations
    ///
    /// Reports only what the code does. Insertion runs one record at a time
    /// through `add_single_vector`, so the fields that described a parallel
    /// insert path were removed rather than corrected.
    pub fn get_performance_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("search_speedup_expected".to_string(), "1.2x-2x".to_string());
        info.insert(
            "search_bottleneck".to_string(),
            "hnsw_mutex_serialization".to_string(),
        );
        info.insert(
            "benefits".to_string(),
            "gil_release_concurrent_metadata_processing_batched_search".to_string(),
        );
        info.insert("insertion_path".to_string(), "sequential".to_string());

        // Add quantization performance info
        //
        // `quantization_memory_savings` used to sit here, reporting
        // 1 - 1/compression_ratio as a percentage. That is the share of a
        // vector a code replaces, not the memory an index saves, and under
        // QuantizedWithRaw the index saves nothing on the vectors because it
        // keeps every one of them. It carried no information
        // `quantization_compression` does not, so it is gone rather than
        // qualified. Memory belongs to `get_stats`, which measures rather than
        // projects.
        if let Some(config) = &self.quantization_config {
            let original_bytes = self.dim * 4; // f32
            let compressed_bytes = config.subvectors; // u8 per subvector
            let compression_ratio = original_bytes as f64 / compressed_bytes as f64;

            info.insert(
                "quantization_compression".to_string(),
                format!("{:.1}x", compression_ratio),
            );
            // Measured at 0.16 recall at 10 against 1.00 for the same data
            // unquantized, so "slight" was wrong by a factor the word cannot
            // carry. Only QuantizedWithRaw can rerank, so the two modes get
            // different answers. Rerank recovers most of the loss rather than
            // all of it, and how much depends on the fetch depth, which is why
            // the default fetch is derived from the live record count rather
            // than fixed; see `DEFAULT_RERANK_CORPUS_DIVISOR`.
            info.insert(
                "quantization_accuracy_impact".to_string(),
                match config.storage_mode {
                    StorageMode::QuantizedOnly => "large_recall_loss_no_rerank_available",
                    StorageMode::QuantizedWithRaw => "large_recall_loss_unless_reranked",
                }
                .to_string(),
            );
        }

        info
    }

    /// Concurrent benchmark for search performance
    #[pyo3(signature = (query_count, max_threads=None))]
    pub fn benchmark_concurrent_reads(
        &self,
        query_count: usize,
        max_threads: Option<usize>,
    ) -> PyResult<HashMap<String, f64>> {
        use rand::random; // Import for random number generation

        let start_time = Instant::now();

        debug!(
            operation = "benchmark_start",
            query_count = query_count,
            max_threads = max_threads,
            "Starting concurrent read benchmark"
        );

        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..self.dim).map(|_| random::<f32>()).collect())
            .collect();

        let mut results = HashMap::new();

        // Sequential benchmark
        let start = Instant::now();
        for query in &queries {
            let _ = self.raw_search_no_gil(query);
        }
        let sequential_time = start.elapsed().as_secs_f64();
        results.insert("sequential_time".to_string(), sequential_time);
        results.insert(
            "sequential_qps".to_string(),
            queries.len() as f64 / sequential_time,
        );

        // Parallel benchmark
        let available_threads = rayon::current_num_threads();
        let num_threads = max_threads
            .unwrap_or(available_threads)
            .min(available_threads);

        let start = Instant::now();
        let _: Vec<_> = queries
            .par_iter()
            .map(|query| self.raw_search_no_gil(query))
            .collect();

        let parallel_time = start.elapsed().as_secs_f64();
        results.insert("parallel_time".to_string(), parallel_time);
        results.insert(
            "parallel_qps".to_string(),
            queries.len() as f64 / parallel_time,
        );
        results.insert("speedup".to_string(), sequential_time / parallel_time);
        results.insert("threads_used".to_string(), num_threads as f64);

        let total_duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "benchmark_complete",
            sequential_qps = queries.len() as f64 / sequential_time,
            parallel_qps = queries.len() as f64 / parallel_time,
            speedup = sequential_time / parallel_time,
            duration_ms = total_duration_ms,
            "Benchmark completed"
        );

        Ok(results)
    }
}

/// Records accepted by `add`, after parsing and before insertion, as
/// (external id, vector, metadata). The vector is still in its input form and
/// has not been normalized for the index space yet.
type ParsedRecords = Vec<(String, Vec<f32>, HashMap<String, Value>)>;

/// An error raised inside `add`'s insertion phase, carried out to be recorded
///
/// The insertion phase runs with the interpreter lock released, so it cannot
/// build the message for an error that arrives as a `PyErr`. `PyErr`'s
/// `Display` implementation calls `Python::attach`, which would reacquire the
/// lock in the middle of the region that exists to have released it, and would
/// do so while the mutation guard and possibly a storage guard are held.
/// `add` formats those once the lock is back.
enum InsertError {
    /// A message Rust already holds, counted against `total_errors`
    Counted(String),

    /// A training failure. Recorded but not counted, which is what the training
    /// path has always done, because a training failure is not a rejected record
    Training(String),

    /// A `PyErr` from one of the three insert paths, with the id it belongs to.
    /// Counted against `total_errors` once formatted
    Vector { id: String, err: PyErr },
}

/// Search hits for one query vector, as (external id, distance, metadata,
/// optional raw vector). The raw vector is present only when the caller asked
/// for it and the index still holds one.
type QueryHits = Vec<(String, f32, HashMap<String, Value>, Option<Vec<f32>>)>;

/// A JSON number as either an exact integer or a float. `serde_json` stores an
/// integer and a float in different variants, and comparing those variants is
/// what made 10 and 10.0 unequal under some operators and equal under others.
enum NumericValue {
    Integer(i128),
    Float(f64),
}

// INTERNAL METHODS, HELPERS AND IMPLEMENTATIONS
impl HNSWIndex {
    /// Count the records the index actually holds
    ///
    /// The union of the raw vectors and the PQ codes, because `quantized_only`
    /// keeps a record added after training in the codes alone. This is derived
    /// from the stored data rather than from the counter, so it is what the
    /// counter is checked against after a load. Not exposed to Python, since
    /// the only caller is the load path in `persistence`.
    pub fn count_stored_records(&self) -> usize {
        let vectors = self.vectors.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let code_only = pq_codes
            .keys()
            .filter(|id| !vectors.contains_key(*id))
            .count();
        vectors.len() + code_only
    }

    /// Warn once when the index holds materially more records than it declared
    ///
    /// `expected_size` is a capacity hint rather than a limit, so exceeding it is
    /// legal and the index keeps working. What it costs is not the reservation,
    /// which grows through the ordinary `Vec::push` path, but the graph degree.
    /// The Python factory derives the default `m` from `expected_size`, and `m`
    /// is fixed at construction, so an index that has outgrown its declaration by
    /// a wide margin is running at a degree chosen for a smaller index and no
    /// later `add` revises it. Nothing else tells a caller that.
    ///
    /// A warning rather than an error, because the index is correct and the only
    /// remedy is to rebuild at an honest declaration, which is the caller's call.
    ///
    /// Fires once per index. The flag is claimed with a compare and exchange, so
    /// two writers crossing the threshold together produce one line and not two.
    fn warn_if_outgrown_expected_size(&self) {
        if self.overgrowth_warned.load(Ordering::Acquire) {
            return;
        }

        let threshold = self
            .expected_size
            .saturating_mul(EXPECTED_SIZE_OVERGROWTH_FACTOR);
        let live_records = self.id_map.read().unwrap().len();
        if live_records <= threshold {
            return;
        }

        if self
            .overgrowth_warned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        warn!(
            operation = "expected_size_exceeded",
            live_records = live_records,
            expected_size = self.expected_size,
            m = self.m,
            "Index holds more than {}x the records its expected_size declared. \
             expected_size is a hint and not a limit, so nothing is broken, but m \
             is fixed at construction and was sized for the declaration. Recreate \
             the index with an expected_size matching what it actually holds if \
             recall matters. This warning fires once.",
            EXPECTED_SIZE_OVERGROWTH_FACTOR
        );
    }

    /// Get next available internal ID
    ///
    /// Not exposed to Python. Every call takes the counter mutex and
    /// increments it, so a call from outside the insertion path burns an
    /// internal id that no record will ever hold.
    fn get_next_id(&self) -> usize {
        let mut counter = self.id_counter.lock().unwrap();
        *counter += 1;
        *counter
    }

    /// Pure function for vector normalization
    fn normalize_vector(&self, vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vector // Return unchanged for zero vectors
        }
    }

    /// Process vector according to distance space
    fn process_vector_for_space(&self, vector: Vec<f32>) -> Vec<f32> {
        match self.space.to_lowercase().as_str() {
            "cosine" => self.normalize_vector(vector),
            // Future extensions:
            // "l2" => self.preprocess_l2(vector),
            // "l1" => self.preprocess_l1(vector),
            _ => vector,
        }
    }

    /// Helper for query processing (mirrors extract_single_vector validation)
    fn validate_and_process_query_vector(&self, vector: Vec<f32>) -> PyResult<Vec<f32>> {
        // Same validation as extract_single_vector
        if vector.is_empty() {
            error!(
                operation = "query_validation",
                error = "empty_vector",
                "Search vector cannot be empty"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Search vector cannot be empty",
            ));
        }
        if vector.len() != self.dim {
            error!(
                operation = "query_validation",
                error = "dimension_mismatch",
                expected = self.dim,
                actual = vector.len(),
                "Search vector dimension mismatch"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Search vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                error!(
                    operation = "query_validation",
                    error = "invalid_value",
                    index = i,
                    value = val,
                    "Search vector contains invalid value"
                );
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Search vector contains invalid value at index {}: {}",
                    i, val
                )));
            }
        }

        // Apply same processing as storage vectors
        Ok(self.process_vector_for_space(vector))
    }

    /// Internal remove_point method that can be called without Python bindings
    /// This is the core method that properly removes all traces of a document
    /// Enhanced internal remove_point method with comprehensive PQ support
    fn remove_point_internal(&self, id: String) -> Result<bool, String> {
        // Read before the guards are taken, because it reaches the graph lock and
        // the declared order puts the graph above every map held below.
        let storage_mode = self.get_storage_mode();

        // Every write guard, in the order declared on `HNSWIndex`. This used to
        // take `vectors` before `rev_map`, which is the reverse of what a search
        // does, and a search holds `rev_map` for its whole traversal. That pair
        // could not overlap while the receivers were exclusive. It can now.
        let mut id_map = self.id_map.write().unwrap();
        let mut rev_map = self.rev_map.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        let mut pq_codes = self.pq_codes.write().unwrap();
        let mut vector_metadata = self.vector_metadata.write().unwrap();

        // Check if the document exists
        if let Some(internal_id) = id_map.remove(&id) {
            // Track what we're removing for logging
            let had_raw_vector = vectors.contains_key(&id);
            let had_pq_codes = pq_codes.contains_key(&id);

            // Remove from all data structures
            vectors.remove(&id); // Remove raw vectors (if present)
            vector_metadata.remove(&id); // Remove metadata
            pq_codes.remove(&id); // Remove PQ codes (if present)
            rev_map.remove(&internal_id); // Remove ID mapping

            // Enhanced training state cleanup for quantization
            if self.has_quantization() {
                // Remove from training IDs if present and not yet trained
                if !self.can_use_quantization() {
                    let mut training_ids = self.training_ids.write().unwrap();
                    let original_len = training_ids.len();
                    training_ids.retain(|training_id| training_id != &id);

                    if training_ids.len() != original_len {
                        trace!(
                            operation = "training_cleanup",
                            vector_id = %id,
                            remaining_training_vectors = training_ids.len(),
                            "Removed vector from training set"
                        );

                        // Update threshold status if we dropped below training size
                        if let Some(config) = &self.quantization_config {
                            if training_ids.len() < config.training_size {
                                self.training_threshold_reached
                                    .store(false, std::sync::atomic::Ordering::Release);
                                debug!(
                                    operation = "training_threshold_reset",
                                    remaining_vectors = training_ids.len(),
                                    required = config.training_size,
                                    "Training threshold reset due to removal"
                                );
                            }
                        }
                    }
                }
            }

            // Decrement vector count since we removed a vector
            {
                let mut count = self.vector_count.lock().unwrap();
                if *count > 0 {
                    *count -= 1;
                }
            }

            debug!(
                operation = "remove_point_internal",
                vector_id = %id,
                internal_id = internal_id,
                had_raw_vector = had_raw_vector,
                had_pq_codes = had_pq_codes,
                storage_mode = storage_mode,
                note = "hnsw_graph_entries_remain_unreachable",
                "Vector completely removed from index (HNSW graph entries become unreachable)"
            );
            Ok(true)
        } else {
            trace!(
                operation = "remove_point_internal",
                vector_id = %id,
                "Vector not found for removal"
            );
            Ok(false)
        }
    }

    /// The body of `rebuild_with_quantization`, with the writers guard already held
    ///
    /// Training reaches this from inside `add`, which owns the guard for the whole
    /// call, so the two entry points are separate rather than one taking the guard
    /// twice and deadlocking on itself.
    ///
    /// Errors are `String` rather than `PyErr` because both callers reach this with
    /// the interpreter lock released, and `PyErr`'s `Display` acquires it. The
    /// entry point above turns the message back into the `PyRuntimeError` it always
    /// raised.
    fn rebuild_with_quantization_locked(&self) -> Result<bool, String> {
        let start_time = Instant::now();

        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                warn!(
                    operation = "rebuild_quantization",
                    reason = "pq_not_trained",
                    "Cannot rebuild: PQ not trained"
                );
                return Ok(false);
            }
        };

        // Create new PQ-based HNSW index
        let max_layer = 16; // Always use NB_LAYER_MAX for consistency
        trace!(
            operation = "rebuild_quantization",
            max_layer = max_layer,
            "Creating new PQ HNSW index"
        );

        // Quantize every stored raw vector and record the codes, then release
        // the storage guards. Nothing below this block holds one, which is what
        // lets the graph work take its own guards in the declared order rather
        // than under a `vectors` guard taken first.
        //
        // An empty raw store is not an error. Under QuantizedOnly the raw
        // vectors are released the moment training completes, so a trained
        // index in that mode holds codes alone, and the rebuild proceeds from
        // those stored codes exactly as `compact` does. Only an index with
        // neither raw vectors nor codes has nothing to rebuild from.
        let (vector_count, retained) = {
            let vectors = self.vectors.read().unwrap();
            if vectors.is_empty() {
                let code_count = self.pq_codes.read().unwrap().len();
                if code_count == 0 {
                    warn!(
                        operation = "rebuild_quantization",
                        reason = "no_vectors_or_codes",
                        "Cannot rebuild: no vectors or codes available"
                    );
                    return Ok(false);
                }
                info!(
                    operation = "quantization_rebuild_start",
                    vector_count = 0,
                    codes_retained = code_count,
                    "Starting quantization rebuild from stored codes"
                );
                (0, code_count)
            } else {
                info!(
                    operation = "quantization_rebuild_start",
                    vector_count = vectors.len(),
                    "Starting quantization rebuild"
                );

                let vector_refs: Vec<&[f32]> = vectors.values().map(|v| v.as_slice()).collect();
                let quantized_codes = pq.quantize_batch(&vector_refs).map_err(|e| {
                    error!(operation = "quantization_rebuild", error = %e, "Failed to quantize vectors");
                    format!("Failed to quantize vectors: {}", e)
                })?;

                // Store quantized codes. Codes for records that have no raw vector
                // are kept rather than cleared, because under QuantizedOnly they
                // are the only copy of every record added after training completed
                // and there is nothing left to re-quantize them from. Clearing
                // dropped those records from the index outright. Removal already
                // deletes an id's codes, so nothing stale can survive here.
                let mut pq_codes = self.pq_codes.write().unwrap();
                let retained = pq_codes
                    .keys()
                    .filter(|id| !vectors.contains_key(*id))
                    .count();

                for (i, (id, _)) in vectors.iter().enumerate() {
                    if i < quantized_codes.len() {
                        pq_codes.insert(id.clone(), quantized_codes[i].clone());
                    }
                }
                debug!(
                    operation = "quantization_rebuild",
                    codes_stored = pq_codes.len(),
                    codes_retained = retained,
                    "Quantized codes stored"
                );
                (vectors.len(), retained)
            }
        };

        // The codes are copied out so the graph is built with no lock held at
        // all. A large batch insert forks to rayon, and a fork under the graph's
        // write guard can leave every worker in the pool waiting on the thread
        // that holds it. Copying costs one byte per subvector per record.
        let batch_data: Vec<(Vec<u8>, usize)> = {
            let id_map = self.id_map.read().unwrap();
            let pq_codes = self.pq_codes.read().unwrap();
            pq_codes
                .iter()
                .filter_map(|(id, codes)| {
                    id_map
                        .get(id)
                        .map(|&internal_id| (codes.clone(), internal_id))
                })
                .collect()
        };

        let new_hnsw = DistanceType::new_pq(
            &self.space,
            self.m,
            self.expected_size,
            max_layer,
            self.ef_construction,
            pq.clone(),
        );

        if !batch_data.is_empty() {
            let batch: Vec<(&Vec<u8>, usize)> = batch_data
                .iter()
                .map(|(codes, internal_id)| (codes, *internal_id))
                .collect();
            new_hnsw.insert_batch_pq(&batch)
                .map_err(|e| {
                    error!(operation = "quantization_rebuild", error = %e, "Failed to insert quantized vectors");
                    format!("Failed to insert quantized vectors: {}", e)
                })?;
        }

        // The replacement is built in full before it is installed, so a search
        // running alongside this sees the old graph or the new one and never a
        // partly filled one. It used to see the empty new graph for as long as
        // the insertions took.
        //
        // The old graph is moved out and dropped after the guard is released.
        // See `replace_graph`.
        self.replace_graph(new_hnsw);

        // Release the raw vectors QuantizedOnly no longer needs. Every one of
        // them was encoded above and its codes stored before the graph was
        // built, so from here the codes are the record and the raw copies are
        // dead weight. This runs only after the new graph is installed, so a
        // failed rebuild leaves the raw store untouched. The map is replaced
        // rather than cleared so its allocation is returned as well. Training
        // completion is the only path that reaches here with a populated raw
        // store under QuantizedOnly, which is what makes this the single point
        // where the mode sheds its training records.
        let released = if vector_count > 0
            && self
                .quantization_config
                .as_ref()
                .is_some_and(|config| config.storage_mode == StorageMode::QuantizedOnly)
        {
            let mut vectors = self.vectors.write().unwrap();
            let released = vectors.len();
            *vectors = HashMap::new();
            released
        } else {
            0
        };

        // ✅ ENTERPRISE: Add duration timing with fixed compression ratio calculation
        let duration_ms = start_time.elapsed().as_millis();
        let compression_ratio = (pq.dim as f64 * 4.0) / pq.subvectors as f64;
        info!(
            operation = "quantization_rebuild_complete",
            vector_count = vector_count,
            codes_inserted = batch_data.len(),
            codes_retained = retained,
            raw_vectors_released = released,
            compression_ratio = compression_ratio,
            duration_ms = duration_ms,
            "Quantization rebuild completed successfully"
        );

        Ok(true)
    }

    /// The body of `compact`, with the interpreter lock already released
    fn compact_locked(&self) -> PyResult<usize> {
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();

        let live_count = self.id_map.read().unwrap().len();
        let nodes_before = self.hnsw.read().unwrap().nb_points();

        if nodes_before <= live_count {
            debug!(
                operation = "compact",
                graph_nodes = nodes_before,
                live_records = live_count,
                "No stranded nodes, compact is a no-op"
            );
            return Ok(0);
        }

        let quantized = self.is_quantized();
        // NB_LAYER_MAX, matching every other construction site in this file.
        let max_layer = 16;

        let new_hnsw = if quantized {
            let pq = self.pq.as_ref().cloned().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Index reports a quantized graph but holds no product quantizer",
                )
            })?;
            DistanceType::new_pq(
                &self.space,
                self.m,
                self.expected_size,
                max_layer,
                self.ef_construction,
                pq,
            )
        } else {
            DistanceType::new_raw(
                &self.space,
                self.m,
                self.expected_size,
                max_layer,
                self.ef_construction,
            )
        };

        // Re-insert every live record under the internal id it already holds, so the
        // two id maps stay correct without being rewritten. A record whose source data
        // is missing is collected rather than skipped, because skipping it would drop
        // it from the index silently.
        let missing: Vec<String> = {
            let id_map = self.id_map.read().unwrap();

            // Internal id order, which is arrival order, rather than the order a
            // hash map hands its entries out. Two compactions of the same index
            // in two processes otherwise wire the replacement graph differently
            // and answer the same query differently.
            let mut live: Vec<(&String, usize)> = id_map
                .iter()
                .map(|(id, &internal)| (id, internal))
                .collect();
            live.sort_by_key(|&(_, internal_id)| internal_id);

            if quantized {
                let pq_codes = self.pq_codes.read().unwrap();
                let mut batch: Vec<(&Vec<u8>, usize)> = Vec::with_capacity(id_map.len());
                let mut missing = Vec::new();

                for (ext_id, internal_id) in live {
                    match pq_codes.get(ext_id) {
                        Some(codes) => batch.push((codes, internal_id)),
                        None => missing.push(ext_id.clone()),
                    }
                }

                if missing.is_empty() && !batch.is_empty() {
                    new_hnsw.insert_batch_pq(&batch).map_err(|e| {
                        error!(operation = "compact", error = %e, "Failed to re-insert quantized codes");
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to re-insert quantized codes during compact: {}",
                            e
                        ))
                    })?;
                }

                missing
            } else {
                let vectors = self.vectors.read().unwrap();
                let mut missing = Vec::new();

                for (ext_id, internal_id) in live {
                    match vectors.get(ext_id) {
                        Some(vector) => new_hnsw.insert(vector, internal_id),
                        None => missing.push(ext_id.clone()),
                    }
                }

                missing
            }
        };

        if !missing.is_empty() {
            error!(
                operation = "compact",
                missing_records = missing.len(),
                live_records = live_count,
                quantized = quantized,
                "Refusing to compact, some live records have no source data to rebuild from"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Refusing to compact: {} of {} live records have no stored {} to rebuild \
                 the graph from, so compacting would drop them. The index is unchanged.",
                missing.len(),
                live_count,
                if quantized {
                    "quantized codes"
                } else {
                    "vector"
                }
            )));
        }

        let nodes_after = new_hnsw.nb_points();
        self.replace_graph(new_hnsw);

        let reclaimed = nodes_before - nodes_after;
        info!(
            operation = "compact_complete",
            nodes_before = nodes_before,
            nodes_after = nodes_after,
            nodes_reclaimed = reclaimed,
            live_records = live_count,
            quantized = quantized,
            duration_ms = start_time.elapsed().as_millis(),
            "Graph compacted"
        );

        Ok(reclaimed)
    }

    /// The body of `save`, with the interpreter lock already released
    fn save_locked(&self, path: &str) -> PyResult<()> {
        // A save reads the mappings, the metadata, the codes, the vectors and
        // the graph in five separate passes, so it needs the index to hold
        // still. PyO3's exclusive borrow used to guarantee that by keeping every
        // mutating method away from it. Relaxing the receivers removes that, and
        // a save overlapping an add would write a directory whose mappings and
        // vectors came from different instants. This takes the mutation lock
        // instead, which blocks a concurrent write and no reader at all.
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();
        info!(operation = "save_start", path = path, "Starting index save");

        let path_buf = Path::new(path);

        // Phase 1: Save all ZeusDB components (already tested to work)
        debug!(operation = "save_phase1", "Saving ZeusDB components");
        crate::persistence::save_index(self, path)?;

        // Phase 2: Save HNSW graph using hnsw-rs native dump
        debug!(operation = "save_phase2", "Saving HNSW graph");
        self.save_hnsw_graph(path_buf)?;

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "save_complete",
            path = path,
            duration_ms = duration_ms,
            "Index save completed successfully"
        );
        Ok(())
    }

    /// Install a replacement graph, and drop the old one outside the guard
    ///
    /// The three paths that replace the whole backend, `compact`, the
    /// quantization rebuild and the persistence rebuild, all used to write
    /// `*hnsw_guard = new_hnsw` directly. That assignment drops the old graph
    /// while the write guard is still held, and dropping a graph is not a quiet
    /// operation. `PointIndexation::drop` in the vendored crate clears each
    /// layer with `into_par_iter().for_each(...)`, so the drop forks to rayon.
    ///
    /// A rayon fork under the graph's write guard deadlocks whenever the pool is
    /// occupied by search tasks. `batch_search_parallel` fans a batch of more
    /// than five queries across the pool and each task takes a read guard, so
    /// once a writer is queued every worker blocks behind it. The fork then has
    /// no worker to run on and the writer never reaches the point of releasing.
    /// Relay 36 wrote the rule that no path forks to rayon while holding a write
    /// guard, and this is the fork that rule missed, because it is hidden inside
    /// an assignment rather than written as a call.
    ///
    /// Moving the old value out and dropping it after the guard is released
    /// keeps the swap to a pointer move under the guard.
    fn replace_graph(&self, new_hnsw: DistanceType) {
        let old = {
            let mut hnsw_guard = self.hnsw.write().unwrap();
            std::mem::replace(&mut *hnsw_guard, new_hnsw)
        };
        drop(old);
    }

    /// The insertion phase of `add`, run with the interpreter lock released
    ///
    /// Everything here operates on `ParsedRecords`, which is
    /// `Vec<(String, Vec<f32>, HashMap<String, Value>)>` and holds no Python
    /// object, no `Py<T>` and no borrow of anything Python owns.
    /// The caller holds the mutation guard.
    ///
    /// The complete set of functions reachable from here, verified by reading
    /// each one rather than by inference:
    ///
    /// - `remove_point_internal`, and through it `get_storage_mode`,
    ///   `has_quantization` and `can_use_quantization`
    /// - `add_single_vector`, and the three paths below it, `add_raw_vector`,
    ///   `add_with_id_collection` and `add_quantized_vector`
    /// - `get_next_id`, `is_quantized`
    /// - `maybe_trigger_training`, `train_quantization_from_ids` and
    ///   `rebuild_with_quantization_locked`
    /// - `PQ::is_trained`, `quantize`, `quantize_batch` and `train`, plus the
    ///   k-means below `train`
    /// - `DistanceType::insert`, `insert_pq_codes`, `insert_batch_pq`,
    ///   `new_pq` and `nb_points`, and the vendored `hnsw_rs` graph below them
    ///
    /// None of them takes a `Python` token, and none of them calls into the
    /// interpreter. `pq.rs`, `distance.rs` and the vendored crate name PyO3
    /// nowhere at all. The two places that did reach Python were both removed
    /// rather than worked around. `rebuild_with_quantization_locked` returned a
    /// `PyResult` whose error the training path formatted into a message, and it
    /// now returns `Result<bool, String>`. The per-record errors are carried out
    /// as `InsertError` values instead of being formatted here.
    ///
    /// Training completing mid-insert is the longest thing this can run, since it
    /// fires k-means and then rebuilds the whole graph from quantized codes, and
    /// it is entirely Rust.
    ///
    /// Logging is safe. The `tracing` subscriber this crate installs writes to
    /// stdout, to stderr, or to a rotating file through `tracing-appender`. No
    /// layer bridges to Python's `logging`, and the Python layer only ever sets
    /// environment variables that the Rust initialiser reads at import.
    ///
    /// A panic in here is safe too. `Python::detach` restores the
    /// interpreter lock from a `Drop` guard, so an unwind reacquires it before it
    /// reaches PyO3's boundary.
    fn insert_parsed_records(
        &self,
        parsed_data: ParsedRecords,
        overwrite: bool,
    ) -> (usize, Vec<InsertError>) {
        let mut total_inserted = 0;
        let mut errors: Vec<InsertError> = Vec::new();

        // ENHANCED FIX: Handle overwrites properly for ALL paths (Raw, Training, PQ)
        if overwrite {
            // Phase 1: Batch identify and remove existing documents
            let (ids_to_remove, storage_analysis) = {
                let id_map = self.id_map.read().unwrap();
                let vectors = self.vectors.read().unwrap();
                let pq_codes = self.pq_codes.read().unwrap();

                let mut ids_to_remove = Vec::new();
                let mut has_raw = 0;
                let mut has_pq = 0;
                let mut has_both = 0;

                for (id, _, _) in &parsed_data {
                    if id_map.contains_key(id) {
                        ids_to_remove.push(id.clone());

                        // Analyze what's being replaced for logging
                        let has_raw_vector = vectors.contains_key(id);
                        let has_pq_codes = pq_codes.contains_key(id);

                        match (has_raw_vector, has_pq_codes) {
                            (true, true) => has_both += 1,
                            (true, false) => has_raw += 1,
                            (false, true) => has_pq += 1,
                            (false, false) => {} // Shouldn't happen, but handle gracefully
                        }
                    }
                }

                (ids_to_remove, (has_raw, has_pq, has_both))
            }; // Release all read locks here

            if !ids_to_remove.is_empty() {
                info!(
                    operation = "overwrite_preparation",
                    documents_to_remove = ids_to_remove.len(),
                    storage_analysis = format!(
                        "raw_only: {}, pq_only: {}, both: {}",
                        storage_analysis.0, storage_analysis.1, storage_analysis.2
                    ),
                    "Removing existing documents for overwrite"
                );

                // Batch remove existing documents (handles both raw and PQ data)
                let mut removed_count = 0;
                let mut removal_errors = 0;

                for id in ids_to_remove {
                    match self.remove_point_internal(id.clone()) {
                        Ok(was_removed) => {
                            if was_removed {
                                removed_count += 1;
                                trace!(
                                    operation = "overwrite_removal",
                                    vector_id = %id,
                                    "Removed existing vector/codes for overwrite"
                                );
                            }
                        }
                        Err(e) => {
                            removal_errors += 1;
                            warn!(
                                operation = "overwrite_removal",
                                vector_id = %id,
                                error = %e,
                                "Failed to remove existing vector for overwrite"
                            );
                            errors.push(InsertError::Counted(format!(
                                "Failed to remove existing {}: {}",
                                id, e
                            )));
                        }
                    }
                }

                info!(
                    operation = "overwrite_removal_complete",
                    removed_count = removed_count,
                    removal_errors = removal_errors,
                    "Completed removal phase for overwrite"
                );
            }
        }

        // Phase 2: Add new vectors using the correct path based on current PQ state
        debug!(
            operation = "add_vectors_insertion_phase",
            current_state = self.get_storage_mode(),
            "Starting insertion phase"
        );

        for (id, vector, metadata) in parsed_data {
            let id_for_error = id.clone();

            // Use overwrite=false since we already handled removals above
            // The add_single_vector method will route to the correct path based on current PQ state
            match self.add_single_vector(id, vector, metadata, false) {
                Ok(inserted_new) => {
                    total_inserted += 1;
                    if inserted_new {
                        let mut count = self.vector_count.lock().unwrap();
                        *count += 1;
                    }

                    // Check training trigger (graceful failure handling)
                    if let Err(training_error) = self.maybe_trigger_training() {
                        warn!(
                            operation = "training_trigger",
                            error = %training_error,
                            vector_id = %id_for_error,
                            "Training trigger failed"
                        );
                        errors.push(InsertError::Training(format!(
                            "Training failed: {}",
                            training_error
                        )));
                    }
                }
                Err(e) => {
                    errors.push(InsertError::Vector {
                        id: id_for_error,
                        err: e,
                    });
                }
            }
        }

        (total_inserted, errors)
    }

    // 1. CORE VECTOR OPERATIONS (6 methods)
    /// 3-PATH ARCHITECTURE - Main router
    fn add_single_vector(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, Value>,
        overwrite: bool,
    ) -> PyResult<bool> {
        // Check if this is a new vector or an overwrite
        let is_new = {
            let id_map = self.id_map.read().unwrap();
            !id_map.contains_key(&id)
        };

        if !overwrite && !is_new {
            warn!(
                operation = "add_single_vector",
                vector_id = %id,
                reason = "already_exists",
                "Vector already exists and overwrite=false"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector with ID '{}' already exists",
                id
            )));
        }

        trace!(
            operation = "add_single_vector",
            vector_id = %id,
            is_new = is_new,
            has_quantization = self.has_quantization(),
            is_quantized = self.is_quantized(),
            "Routing vector addition"
        );

        // Clean 3-Path Architecture
        if !self.has_quantization() {
            // Path A: Raw storage (no quantization config)
            self.add_raw_vector(id, vector, metadata)?;
        } else if !self.is_quantized() {
            // Path B: Raw storage + ID collection for training
            self.add_with_id_collection(id, vector, metadata)?;
        } else {
            // Path C: Quantized storage (PQ trained and active)
            self.add_quantized_vector(id, vector, metadata)?;
        }

        Ok(is_new)
    }

    /// Path A: Raw storage (no quantization)
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "raw_storage"
    ))]
    fn add_raw_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed by extract_single_vector
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Store processed vector directly (no additional processing)
        {
            let mut vectors = self.vectors.write().unwrap();
            vectors.insert(id.clone(), vector.clone()); // Already normalized
        }

        // Insert processed vector into HNSW
        {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard.insert(&vector, internal_id); // Already normalized
        }

        trace!(
            operation = "add_raw_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            "Raw vector added successfully"
        );

        Ok(())
    }

    /// Path B: ID collection for consistent training
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "id_collection"
    ))]
    fn add_with_id_collection(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        // 1. Store vector normally (single storage)
        self.add_raw_vector(id.clone(), vector, metadata)?;

        // SKIP TRAINING ID COLLECTION DURING PERSISTENCE REBUILD
        if self
            .rebuilding_from_persistence
            .load(std::sync::atomic::Ordering::Acquire)
        {
            trace!(
                operation = "add_with_id_collection",
                vector_id = %id,
                reason = "rebuilding_from_persistence",
                "Skipping training ID collection during rebuild"
            );
            return Ok(());
        }

        // 2. Collect ID for training (minimal memory overhead)
        //
        // The training set is the first `training_size` records to arrive, and
        // which records it holds cannot be drawn randomly. Training fires on the
        // record that reaches `training_size`, so the index holds exactly
        // `training_size` records at that moment and any sample of the records
        // present is the whole of them. Drawing from a wider pool would mean
        // deferring the trigger and holding more records raw, which is a change
        // of shape rather than a change of sampling.
        //
        // What the membership would buy was measured, by feeding the current
        // design its worst case instead of changing it. Three corpora at 25,000
        // records and dim 768, each built twice, once inserted in generation
        // order and once sorted so the training set is one segment.
        //
        //   corpus                              in order   sorted
        //   50 Gaussian clusters                  0.996      0.993
        //   8 sources, disjoint 48-dim subspaces  0.887      0.930
        //   8 sources, disjoint variance blocks   0.268      0.347
        //
        // Sorted trains on 2 clusters of 50 in the first row and 1 source of 8
        // in the other two, and it is no worse in any of them. The reason is
        // that a codebook is fitted per contiguous coordinate slice, so a
        // segment only looks different to it if its per-coordinate marginals
        // differ, and those are far more stable across content than the joint
        // distribution is.
        //
        // What is drawn randomly is the order the sample is held in, which is
        // what every subset of it is taken by. `train_quantization_from_ids`
        // shuffles it under a fixed seed; see `TRAINING_SAMPLE_SEED`.
        if let Some(config) = &self.quantization_config {
            if !self.training_threshold_reached.load(Ordering::Acquire) {
                let mut training_ids = self.training_ids.write().unwrap();

                if training_ids.len() < config.training_size {
                    training_ids.push(id.clone());
                    let progress = (training_ids.len() as f32 / config.training_size as f32
                        * 100.0)
                        .min(100.0);

                    trace!(
                        operation = "training_id_collection",
                        vector_id = %id,
                        collected_count = training_ids.len(),
                        target_size = config.training_size,
                        progress_percent = progress,
                        "Training ID collected"
                    );

                    // Check if we've reached the threshold
                    if training_ids.len() >= config.training_size {
                        self.training_threshold_reached
                            .store(true, Ordering::Release);
                        info!(
                            operation = "training_threshold_reached",
                            collected_count = training_ids.len(),
                            target_size = config.training_size,
                            "Training threshold reached - ready for PQ training"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Path C: Quantized storage with configurable raw vector retention
    #[instrument(level = "trace", skip(self, vector, metadata), fields(
        vector_id = %id,
        path = "quantized_storage"
    ))]
    fn add_quantized_vector(
        &self,
        id: String,
        vector: Vec<f32>, // Already processed
        metadata: HashMap<String, Value>,
    ) -> PyResult<()> {
        let internal_id = self.get_next_id();

        // Store metadata
        {
            let mut vector_metadata = self.vector_metadata.write().unwrap();
            vector_metadata.insert(id.clone(), metadata);
        }

        // Update ID mappings
        {
            let mut id_map = self.id_map.write().unwrap();
            let mut rev_map = self.rev_map.write().unwrap();

            id_map.insert(id.clone(), internal_id);
            rev_map.insert(internal_id, id.clone());
        }

        // Quantize the vector
        let pq = self.pq.as_ref().unwrap();
        let codes = pq.quantize(&vector).map_err(|e| {
            error!(
                operation = "add_quantized_vector",
                vector_id = %id,
                error = %e,
                "Failed to quantize vector"
            );
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to quantize vector: {}",
                e
            ))
        })?;

        // Store quantized codes (always)
        {
            let mut pq_codes = self.pq_codes.write().unwrap();
            pq_codes.insert(id.clone(), codes.clone());
        }

        // Store raw vector only if configured to keep them
        if let Some(config) = &self.quantization_config {
            if config.storage_mode == StorageMode::QuantizedWithRaw {
                let mut vectors = self.vectors.write().unwrap();
                vectors.insert(id.clone(), vector.clone());
            }
            // If QuantizedOnly mode, we don't store raw vectors (saves memory)
        }

        // Insert codes into quantized HNSW
        {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard.insert_pq_codes(&codes, internal_id);
        }

        trace!(
            operation = "add_quantized_vector_complete",
            vector_id = %id,
            internal_id = internal_id,
            codes_length = codes.len(),
            "Quantized vector added successfully"
        );

        Ok(())
    }

    /// TRAINING TRIGGER: Uses threshold flag for race condition safety
    #[instrument(level = "info", skip(self), fields(
        threshold_reached = self.training_threshold_reached.load(Ordering::Acquire),
        has_quantization = self.has_quantization()
    ))]
    fn maybe_trigger_training(&self) -> Result<(), String> {
        // Check atomic flag first (fast path)
        if !self.training_threshold_reached.load(Ordering::Acquire) {
            return Ok(());
        }

        // Only proceed if we have quantization config and aren't already trained
        if let Some(_config) = &self.quantization_config {
            if let Some(pq) = &self.pq {
                if !pq.is_trained() {
                    info!(
                        operation = "training_trigger",
                        "Training threshold reached - starting PQ training"
                    );
                    return self.train_quantization_from_ids();
                }
            }
        }

        Ok(())
    }

    /// TRAINING EXECUTION: Uses collected IDs for deterministic training set
    #[instrument(level = "info", skip(self), fields(
        has_pq = self.pq.is_some(),
        has_config = self.quantization_config.is_some()
    ))]
    fn train_quantization_from_ids(&self) -> Result<(), String> {
        let start_time = Instant::now();

        let pq = self.pq.as_ref().ok_or("PQ not available")?.clone();
        let config = self
            .quantization_config
            .as_ref()
            .ok_or("Config not available")?
            .clone();

        // Get consistent training set using collected IDs. `vectors` is taken
        // first because the declared lock order puts it above `training_ids`,
        // and every reader that holds both takes them that way round.
        let training_vectors = {
            let vectors = self.vectors.read().unwrap();
            let training_ids = self.training_ids.read().unwrap();

            // ADD EARLY CHECK:
            if training_ids.is_empty() {
                warn!(
                    operation = "pq_training",
                    reason = "no_training_ids",
                    "No training IDs available"
                );
                // Reset threshold to prevent repeated attempts
                self.training_threshold_reached
                    .store(false, Ordering::Release);
                return Err("No training IDs available for training".to_string());
            }

            let mut training_data = Vec::new();
            let mut missing_vectors = 0;

            for id in training_ids.iter() {
                if let Some(vector) = vectors.get(id) {
                    training_data.push(vector.clone());
                } else {
                    missing_vectors += 1;
                }
            }

            if missing_vectors > 0 {
                warn!(
                    operation = "pq_training",
                    missing_vectors = missing_vectors,
                    available_vectors = training_data.len(),
                    "Some training vectors were removed before training"
                );
            }

            debug!(
                operation = "pq_training_dataset",
                collected_ids = training_ids.len(),
                available_vectors = training_data.len(),
                target_size = config.training_size,
                "Training dataset prepared"
            );

            training_data
        };

        if training_vectors.len() < config.training_size {
            error!(
                operation = "pq_training",
                available = training_vectors.len(),
                required = config.training_size,
                "Insufficient vectors for training"
            );
            return Err(format!(
                "Insufficient vectors for training: need {}, have {} (some may have been removed)",
                config.training_size,
                training_vectors.len()
            ));
        }

        // Draw the sample in a seeded random order before anything reads it, so
        // that the codebook, the calibration's queries and every fraction the
        // calibration fits over are random draws rather than slices of
        // insertion order. See `TRAINING_SAMPLE_SEED`.
        let mut training_vectors = training_vectors;
        let mut sample_rng = rand::rngs::StdRng::seed_from_u64(TRAINING_SAMPLE_SEED);
        training_vectors.shuffle(&mut sample_rng);

        // Respect max_training_vectors limit
        let final_training_set = if let Some(max_training) = config.max_training_vectors {
            if training_vectors.len() > max_training {
                debug!(
                    operation = "pq_training_limit",
                    available = training_vectors.len(),
                    using = max_training,
                    "Limiting training set size"
                );
                training_vectors.into_iter().take(max_training).collect()
            } else {
                training_vectors
            }
        } else {
            training_vectors
        };

        info!(
            operation = "pq_training_start",
            training_vectors = final_training_set.len(),
            subvectors = config.subvectors,
            bits = config.bits,
            "Starting PQ training"
        );

        // Train the PQ model
        let training_start = Instant::now();
        pq.train(&final_training_set)?;
        let training_duration = training_start.elapsed();

        info!(
            operation = "pq_training_complete",
            training_vectors = final_training_set.len(),
            duration_ms = training_duration.as_millis(),
            "PQ training completed successfully"
        );

        // Measure the rerank fetch on the data the codebook was just fitted to,
        // while that data and the codebook are both in hand. See
        // `RerankCalibration`. The training set is released with this function's
        // frame, so this is the only point where the measurement is free of a
        // second pass over the records.
        if let Some(calibration) = self.calibrate_rerank(&pq, &final_training_set) {
            info!(
                operation = "rerank_calibration",
                fetch = calibration.fetch,
                sample_records = calibration.sample_records,
                queries = calibration.queries,
                duration_ms = calibration.millis,
                "Rerank fetch calibrated from the training sample"
            );
            *self.rerank_calibration.write().unwrap() = Some(calibration);
        }

        // Clear training IDs (no longer needed)
        {
            let mut training_ids = self.training_ids.write().unwrap();
            training_ids.clear();
        }

        // Rebuild index with quantization
        debug!(
            operation = "pq_rebuild_start",
            "Rebuilding index with quantization"
        );
        let rebuild_start = Instant::now();
        let rebuild_success = self
            .rebuild_with_quantization_locked()
            .map_err(|e| format!("Failed to rebuild with quantization: {}", e))?;
        let rebuild_duration = rebuild_start.elapsed();

        if rebuild_success {
            // The code size against the vector size. A `memory_savings_percent`
            // field used to sit beside it, carrying 1 - 1/compression_ratio,
            // which is the same number in another form and was labelled as a
            // saving the index does not make.
            let compression_ratio = (self.dim as f64 * 4.0) / pq.subvectors as f64;

            let total_duration_ms = start_time.elapsed().as_millis();
            info!(
                operation = "pq_complete",
                rebuild_duration_ms = rebuild_duration.as_millis(),
                compression_ratio = compression_ratio,
                total_duration_ms = total_duration_ms,
                "Index successfully rebuilt with quantization"
            );
        } else {
            error!(operation = "pq_rebuild", "Index rebuild returned false");
            return Err("Index rebuild returned false".to_string());
        }

        Ok(())
    }

    /// Measure how deep this index's codes bury a true neighbour
    ///
    /// Runs once, at training completion, over the training sample and the
    /// codebook just fitted to it. What it measures, why the queries come from
    /// the sample itself, and how the search scales the result to a larger
    /// corpus are all recorded on `RerankCalibration`.
    ///
    /// Returns `None` where the measurement would be spent for nothing.
    /// `quantized_only` never reranks, so it is not calibrated.
    fn calibrate_rerank(&self, pq: &PQ, sample: &[Vec<f32>]) -> Option<RerankCalibration> {
        let keeps_raw = self
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        if !keeps_raw {
            return None;
        }

        calibrate_rerank_from_sample(pq, sample, raw_distance_fn(&self.space))
    }

    /// What training measured, where it ran
    pub fn get_rerank_calibration(&self) -> Option<RerankCalibration> {
        *self.rerank_calibration.read().unwrap()
    }

    /// Install a calibration read back from a saved index
    pub fn set_rerank_calibration(&self, calibration: Option<RerankCalibration>) {
        *self.rerank_calibration.write().unwrap() = calibration;
    }

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
    fn rerank_plan(&self, rerank: Option<usize>) -> Option<RerankPlan> {
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

    /// Raw search without Python objects (for benchmarking)
    fn raw_search_no_gil(&self, query: &[f32]) -> Vec<(String, f32)> {
        // Concurrent read access to ID mapping, taken before the graph lock so the
        // traversal predicate can consult it without acquiring anything itself.
        let rev_map = self.rev_map.read().unwrap();
        let live = |internal_id: &usize| rev_map.contains_key(internal_id);

        // HNSW search with locking
        let hnsw_results = {
            let hnsw_guard = self.hnsw.read().unwrap();
            hnsw_guard
                .search(query, 10, 100, Some(&live))
                .unwrap_or_else(|_| Vec::new())
        }; // Lock released immediately

        hnsw_results
            .into_iter()
            .filter_map(|neighbor| {
                rev_map
                    .get(&neighbor.get_origin_id())
                    .map(|id| (id.clone(), neighbor.distance))
            })
            .collect()
    }

    /// Parse input data into (id, vector, metadata) tuples with error collection
    fn parse_input_data(&self, data: &Bound<PyAny>) -> (ParsedRecords, Vec<String>) {
        let mut parsed_vectors = Vec::new();
        let mut errors = Vec::new();

        if let Ok(dict) = data.cast::<PyDict>() {
            self.parse_dict_input_safe(dict, &mut parsed_vectors, &mut errors);
        } else if let Ok(list) = data.cast::<PyList>() {
            self.parse_list_input_safe(list, &mut parsed_vectors, &mut errors);
        } else if let Ok(np_array) = data.cast::<PyArray2<f32>>() {
            if let Err(e) = self.parse_numpy_input_safe(np_array, &mut parsed_vectors, &mut errors)
            {
                errors.push(format!("NumPy parsing error: {}", e));
            }
        } else {
            // Single vector
            match self.extract_single_vector_safe(data) {
                Ok(vector) => {
                    let id = self.generate_id();
                    parsed_vectors.push((id, vector, HashMap::new()));
                }
                Err(e) => {
                    errors.push(format!("Single vector error: {}", e));
                }
            }
        }

        (parsed_vectors, errors)
    }

    /// Safe dictionary parsing that collects errors
    fn parse_dict_input_safe(
        &self,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) {
        // Check for single object format
        if dict.contains("id").unwrap_or(false)
            && (dict.contains("values").unwrap_or(false)
                || dict.contains("vector").unwrap_or(false))
        {
            // Single object format
            let vector_result = if let Ok(Some(values_item)) = dict.get_item("values") {
                self.extract_single_vector_safe(&values_item)
            } else if let Ok(Some(vector_item)) = dict.get_item("vector") {
                self.extract_single_vector_safe(&vector_item)
            } else {
                Err("Missing 'vector' or 'values' key".to_string())
            };

            match vector_result {
                Ok(vector) => {
                    let id = match dict.get_item("id") {
                        Ok(Some(id_item)) => id_item
                            .extract::<String>()
                            .unwrap_or_else(|_| self.generate_id()),
                        _ => self.generate_id(),
                    };

                    let metadata = match dict.get_item("metadata") {
                        Ok(Some(meta_item)) => {
                            if let Ok(meta_dict) = meta_item.cast::<PyDict>() {
                                self.python_dict_to_value_map(meta_dict).unwrap_or_default()
                            } else {
                                HashMap::new()
                            }
                        }
                        _ => HashMap::new(),
                    };

                    parsed_vectors.push((id, vector, metadata));
                }
                Err(e) => {
                    let id = dict
                        .get_item("id")
                        .ok()
                        .flatten()
                        .and_then(|id_item| id_item.extract::<String>().ok())
                        .unwrap_or_else(|| "single_object".to_string());

                    errors.push(format!("Vector {}: {}", id, e));
                }
            }
        } else {
            // Batch format - try the existing parse_batch_format
            if let Err(e) = self.parse_batch_format(dict, parsed_vectors, errors) {
                errors.push(format!("Batch parsing error: {}", e));
            }
        }
    }

    /// Handle Format 3 & 5: Batch format - WORKING SOLUTION
    fn parse_batch_format(
        &self,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        // Process each key path immediately without storing references

        // Try "vectors" key
        if let Some(vectors_item) = dict.get_item("vectors")? {
            if let Ok(list) = vectors_item.cast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = vectors_item.cast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return self.parse_numpy_with_context(np_array, dict, parsed_vectors, errors);
            }
        }

        // Try "embeddings" key
        if let Some(embeddings_item) = dict.get_item("embeddings")? {
            if let Ok(list) = embeddings_item.cast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else if let Ok(np_array) = embeddings_item.cast::<PyArray2<f32>>() {
                // FIX: Handle NumPy with IDs and metadata
                return self.parse_numpy_with_context(np_array, dict, parsed_vectors, errors);
            }
        }

        // Try "values" key
        if let Some(values_item) = dict.get_item("values")? {
            if let Ok(list) = values_item.cast::<PyList>() {
                return self.process_vector_list(list, dict, parsed_vectors);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "values field must be a list in batch format",
                ));
            }
        }

        // No valid vector data found
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Missing vector data. Expected one of: 'vectors', 'embeddings', or 'values' key",
        ))
    }

    /// Helper method to process vector list (extracted to avoid code duplication)
    fn process_vector_list(
        &self,
        vectors: &Bound<PyList>,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
    ) -> PyResult<()> {
        // Process each vector in the batch
        for (i, vector_item) in vectors.iter().enumerate() {
            let vector = self.extract_single_vector(&vector_item)?;

            // Extract ID from "ids" array
            let id = match dict.get_item("ids")? {
                Some(item) => {
                    let ids_list = item.cast::<PyList>()?;
                    if i < ids_list.len() {
                        ids_list.get_item(i)?.extract::<String>()?
                    } else {
                        self.generate_id()
                    }
                }
                None => self.generate_id(),
            };

            // Extract metadata from "metadatas" or "metadata" arrays
            let meta = match dict
                .get_item("metadatas")?
                .or_else(|| dict.get_item("metadata").ok().flatten())
            {
                Some(item) => {
                    if let Ok(meta_list) = item.cast::<PyList>() {
                        if i < meta_list.len() {
                            let metadata_item = meta_list.get_item(i)?;
                            if let Ok(meta_dict) = metadata_item.cast::<PyDict>() {
                                self.python_dict_to_value_map(meta_dict)?
                            } else if metadata_item.is_none() {
                                HashMap::new()
                            } else {
                                let mut map = HashMap::new();
                                let value = Self::python_object_to_value(&metadata_item)?;
                                let key = if value.is_string() { "text" } else { "value" };
                                map.insert(key.to_string(), value);
                                map
                            }
                        } else {
                            HashMap::new()
                        }
                    } else {
                        HashMap::new()
                    }
                }
                None => HashMap::new(),
            };

            parsed_vectors.push((id, vector, meta));
        }

        Ok(())
    }

    /// Parse NumPy array with context (IDs and metadata from dict)
    fn parse_numpy_with_context(
        &self,
        np_array: &Bound<PyArray2<f32>>,
        dict: &Bound<PyDict>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> PyResult<()> {
        let readonly = np_array.readonly();
        let shape = readonly.shape();

        trace!(operation = "parse_numpy_context", shape = ?shape, "Processing NumPy array with context");

        if shape.len() != 2 || shape[1] != self.dim {
            error!(
                operation = "parse_numpy_context",
                error = "shape_mismatch",
                expected_shape = format!("(N, {})", self.dim),
                actual_shape = format!("{:?}", shape),
                "NumPy array shape validation failed"
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "NumPy array must have shape (N, {}), got {:?}",
                self.dim, shape
            )));
        }

        let flat = readonly.as_slice()?;
        let num_vectors = shape[0];

        // Extract IDs array
        let ids_list = dict
            .get_item("ids")?
            .and_then(|item| item.cast::<PyList>().ok().cloned());

        // Extract metadata array
        let metadatas_list = dict
            .get_item("metadatas")?
            .or_else(|| dict.get_item("metadata").ok().flatten())
            .and_then(|item| item.cast::<PyList>().ok().cloned());

        trace!(
            operation = "parse_numpy_context",
            num_vectors = num_vectors,
            has_ids = ids_list.is_some(),
            has_metadata = metadatas_list.is_some(),
            "Processing vectors with context"
        );

        let mut rejected = 0usize;

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = &flat[start_idx..end_idx];

            // Resolve the caller's id before validating, so a rejected row can
            // be named without advancing the internal id counter for a record
            // that is never stored.
            let provided_id = match &ids_list {
                Some(ids) if i < ids.len() => ids.get_item(i)?.extract::<String>().ok(),
                _ => None,
            };

            if let Some((component, value)) = Self::first_non_finite(raw_vector) {
                let label = provided_id.clone().unwrap_or_else(|| format!("row_{}", i));
                error!(
                    operation = "parse_numpy_context",
                    error = "invalid_value",
                    vector_id = %label,
                    index = component,
                    value = value,
                    "NumPy row contains a non-finite value"
                );
                errors.push(format!(
                    "Vector {}: contains invalid value at index {}: {} (must be finite)",
                    label, component, value
                ));
                rejected += 1;
                continue;
            }

            let processed_vector = self.process_vector_for_space(raw_vector.to_vec());

            // Get ID from provided IDs or generate
            let id = match provided_id {
                Some(id) => id,
                None => self.generate_id(),
            };

            // Get metadata from provided metadata or use empty
            let metadata = if let Some(metas) = &metadatas_list {
                if i < metas.len() {
                    let meta_item = metas.get_item(i)?;
                    if let Ok(meta_dict) = meta_item.cast::<PyDict>() {
                        self.python_dict_to_value_map(meta_dict)?
                    } else {
                        HashMap::new()
                    }
                } else {
                    HashMap::new()
                }
            } else {
                HashMap::new()
            };

            trace!(
                operation = "parse_numpy_vector",
                vector_index = i,
                vector_id = %id,
                metadata_keys = metadata.keys().len(),
                "Parsed NumPy vector with context"
            );

            parsed_vectors.push((id, processed_vector, metadata));
        }

        trace!(
            operation = "parse_numpy_context_complete",
            parsed_count = num_vectors - rejected,
            rejected_count = rejected,
            "NumPy parsing completed"
        );
        Ok(())
    }

    /// Safe list parsing that collects errors instead of failing immediately
    fn parse_list_input_safe(
        &self,
        list: &Bound<PyList>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) {
        for (item_index, item) in list.iter().enumerate() {
            if let Ok(item_dict) = item.cast::<PyDict>() {
                // Extract vector safely
                let vector_result = if let Ok(Some(vector_item)) = item_dict.get_item("vector") {
                    self.extract_single_vector_safe(&vector_item)
                } else if let Ok(Some(values_item)) = item_dict.get_item("values") {
                    self.extract_single_vector_safe(&values_item)
                } else {
                    Err("Missing 'vector' or 'values' key in item".to_string())
                };

                match vector_result {
                    Ok(vector) => {
                        // Extract ID
                        let id = match item_dict.get_item("id") {
                            Ok(Some(id_item)) => id_item
                                .extract::<String>()
                                .unwrap_or_else(|_| self.generate_id()),
                            _ => self.generate_id(),
                        };

                        // Extract metadata
                        let metadata = match item_dict.get_item("metadata") {
                            Ok(Some(meta_item)) => {
                                if let Ok(meta_dict) = meta_item.cast::<PyDict>() {
                                    self.python_dict_to_value_map(meta_dict).unwrap_or_default()
                                } else {
                                    // Handle non-dict metadata
                                    let mut map = HashMap::new();
                                    if let Ok(value) = Self::python_object_to_value(&meta_item) {
                                        let key = if value.is_string() { "text" } else { "value" };
                                        map.insert(key.to_string(), value);
                                    }
                                    map
                                }
                            }
                            _ => HashMap::new(),
                        };

                        parsed_vectors.push((id, vector, metadata));
                    }
                    Err(e) => {
                        // Collect error with item index and ID for context
                        let id = item_dict
                            .get_item("id")
                            .ok()
                            .flatten()
                            .and_then(|id_item| id_item.extract::<String>().ok())
                            .unwrap_or_else(|| format!("item_{}", item_index));

                        errors.push(format!("Vector {}: {}", id, e));
                    }
                }
            } else {
                // Direct vector item
                match self.extract_single_vector_safe(&item) {
                    Ok(vector) => {
                        let id = self.generate_id();
                        parsed_vectors.push((id, vector, HashMap::new()));
                    }
                    Err(e) => {
                        errors.push(format!("Item {}: {}", item_index, e));
                    }
                }
            }
        }
    }

    /// Safe NumPy parsing for error collection
    fn parse_numpy_input_safe(
        &self,
        np_array: &Bound<PyArray2<f32>>,
        parsed_vectors: &mut Vec<(String, Vec<f32>, HashMap<String, Value>)>,
        errors: &mut Vec<String>,
    ) -> Result<(), String> {
        // This is the same as your current parse_numpy_input but returns Result<(), String>
        let readonly = np_array.readonly();
        let shape = readonly.shape();

        if shape.len() != 2 || shape[1] != self.dim {
            return Err(format!(
                "NumPy array must have shape (N, {}), got {:?}",
                self.dim, shape
            ));
        }

        let flat = readonly
            .as_slice()
            .map_err(|e| format!("NumPy access error: {}", e))?;
        let num_vectors = shape[0];

        for i in 0..num_vectors {
            let start_idx = i * self.dim;
            let end_idx = start_idx + self.dim;
            let raw_vector = &flat[start_idx..end_idx];

            // A bare array carries no ids, so a rejected row is named by its
            // position and no id is generated for it.
            if let Some((component, value)) = Self::first_non_finite(raw_vector) {
                error!(
                    operation = "parse_numpy",
                    error = "invalid_value",
                    row = i,
                    index = component,
                    value = value,
                    "NumPy row contains a non-finite value"
                );
                errors.push(format!(
                    "Vector row_{}: contains invalid value at index {}: {} (must be finite)",
                    i, component, value
                ));
                continue;
            }

            let processed_vector = self.process_vector_for_space(raw_vector.to_vec());
            let id = self.generate_id();
            parsed_vectors.push((id, processed_vector, HashMap::new()));
        }

        Ok(())
    }

    /// Report the first non-finite component of a vector, if there is one
    ///
    /// The two NumPy branches read their rows straight out of the buffer, so
    /// they skip the per-value check that `extract_single_vector` runs. A NaN
    /// that reaches the graph degrades every later query rather than only the
    /// one that carried it, so both branches route through this.
    fn first_non_finite(vector: &[f32]) -> Option<(usize, f32)> {
        vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
            .map(|(index, value)| (index, *value))
    }

    /// Extract a single vector from various Python types (enhanced)
    fn extract_single_vector(&self, data: &Bound<PyAny>) -> PyResult<Vec<f32>> {
        let vector = if let Ok(array1d) = data.cast::<PyArray1<f32>>() {
            // NumPy 1D array
            array1d.readonly().as_slice()?.to_vec()
        } else if let Ok(list) = data.cast::<PyList>() {
            // Python list
            list.iter()
                .map(|item| item.extract::<f32>())
                .collect::<PyResult<Vec<f32>>>()?
        } else {
            // Direct extraction (e.g., from other numeric arrays)
            data.extract::<Vec<f32>>()?
        };

        // Comprehensive validation
        if vector.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Vector cannot be empty",
            ));
        }

        if vector.len() != self.dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }

        // Check for invalid values
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Vector contains invalid value at index {}: {} (must be finite)",
                    i, val
                )));
            }
        }

        // ✅ Apply space-specific processing
        Ok(self.process_vector_for_space(vector))
    }

    /// Generate a unique ID for a vector
    fn generate_id(&self) -> String {
        let id = self.get_next_id();
        format!("vec_{}", id)
    }

    /// Safe version of extract_single_vector that returns String errors instead of PyErr
    fn extract_single_vector_safe(&self, data: &Bound<PyAny>) -> Result<Vec<f32>, String> {
        let vector = if let Ok(array1d) = data.cast::<PyArray1<f32>>() {
            array1d
                .readonly()
                .as_slice()
                .map_err(|e| format!("NumPy access error: {}", e))?
                .to_vec()
        } else if let Ok(list) = data.cast::<PyList>() {
            list.iter()
                .map(|item| {
                    item.extract::<f32>()
                        .map_err(|e| format!("List item error: {}", e))
                })
                .collect::<Result<Vec<f32>, String>>()?
        } else {
            data.extract::<Vec<f32>>()
                .map_err(|e| format!("Vector extraction error: {}", e))?
        };

        // Same validation as extract_single_vector, but with String errors
        if vector.is_empty() {
            return Err("Vector cannot be empty".to_string());
        }
        if vector.len() != self.dim {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }
        for (i, &val) in vector.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "Vector contains invalid value at index {}: {}",
                    i, val
                ));
            }
        }

        Ok(self.process_vector_for_space(vector))
    }

    // 4. DATA CONVERSION & FILTERING (12 methods)
    // Helper methods for data conversion and filtering
    fn python_dict_to_value_map(
        &self,
        py_dict: &Bound<PyDict>,
    ) -> PyResult<HashMap<String, Value>> {
        let mut map = HashMap::new();

        for (key, value) in py_dict.iter() {
            let string_key = key.extract::<String>()?;
            let json_value = Self::python_object_to_value(&value)?;
            map.insert(string_key, json_value);
        }

        Ok(map)
    }

    fn python_object_to_value(py_obj: &Bound<PyAny>) -> PyResult<Value> {
        if py_obj.is_none() {
            Ok(Value::Null)
        } else if let Ok(b) = py_obj.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if let Ok(i) = py_obj.extract::<i64>() {
            Ok(Value::Number(serde_json::Number::from(i)))
        } else if let Ok(f) = py_obj.extract::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(f) {
                Ok(Value::Number(num))
            } else {
                Ok(Value::String(f.to_string()))
            }
        } else if let Ok(s) = py_obj.extract::<String>() {
            Ok(Value::String(s))
        } else if let Ok(py_list) = py_obj.cast::<PyList>() {
            let mut vec = Vec::new();
            for item in py_list.iter() {
                vec.push(Self::python_object_to_value(&item)?);
            }
            Ok(Value::Array(vec))
        } else if let Ok(py_dict) = py_obj.cast::<PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, value) in py_dict.iter() {
                let string_key = key.extract::<String>()?;
                let json_value = Self::python_object_to_value(&value)?;
                map.insert(string_key, json_value);
            }
            Ok(Value::Object(map))
        } else {
            Ok(Value::String(py_obj.to_string()))
        }
    }

    fn matches_filter(
        &self,
        metadata: &HashMap<String, Value>,
        filter: &HashMap<String, Value>,
    ) -> PyResult<bool> {
        for (field, condition) in filter {
            if !self.field_matches(metadata, field, condition)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn field_matches(
        &self,
        metadata: &HashMap<String, Value>,
        field: &str,
        condition: &Value,
    ) -> PyResult<bool> {
        let field_value = match metadata.get(field) {
            Some(value) => value,
            None => return Ok(false),
        };

        match condition {
            // A map is always the operator form. Direct equality against a
            // nested object has no syntax of its own, because the two forms
            // would be indistinguishable, so it is written {"eq": {...}}.
            Value::Object(ops) => self.evaluate_value_conditions(field_value, ops),
            _ => Ok(Self::values_equal(field_value, condition)),
        }
    }

    /// Reject an operator the engine does not implement, before any record is
    /// examined. Checking during evaluation is not enough on its own, because a
    /// record that lacks the field never reaches the operator and a filter that
    /// fails an earlier field short circuits, so whether the typo is noticed
    /// would depend on the data. `evaluate_operator` is the only list of
    /// operator names, so validation cannot disagree with dispatch about what
    /// is known. The field value here is a placeholder, which is sound because
    /// every operator helper is total and the unknown operator arm is the one
    /// error the dispatch can produce.
    fn validate_filter_conditions(&self, filter: &HashMap<String, Value>) -> PyResult<()> {
        for condition in filter.values() {
            if let Value::Object(operations) = condition {
                for (op, target_value) in operations {
                    self.evaluate_operator(&Value::Null, op, target_value)?;
                }
            }
        }
        Ok(())
    }

    fn evaluate_value_conditions(
        &self,
        field_value: &Value,
        operations: &serde_json::Map<String, Value>,
    ) -> PyResult<bool> {
        for (op, target_value) in operations {
            if !self.evaluate_operator(field_value, op, target_value)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn evaluate_operator(
        &self,
        field_value: &Value,
        op: &str,
        target_value: &Value,
    ) -> PyResult<bool> {
        match op {
            "eq" => Ok(Self::values_equal(field_value, target_value)),
            "ne" => Ok(!Self::values_equal(field_value, target_value)),
            "gt" => self.compare_values(field_value, target_value, CmpOrdering::is_gt),
            "gte" => self.compare_values(field_value, target_value, CmpOrdering::is_ge),
            "lt" => self.compare_values(field_value, target_value, CmpOrdering::is_lt),
            "lte" => self.compare_values(field_value, target_value, CmpOrdering::is_le),
            "contains" => self.value_contains(field_value, target_value),
            "startswith" => self.value_starts_with(field_value, target_value),
            "endswith" => self.value_ends_with(field_value, target_value),
            "in" => self.value_in_array(field_value, target_value),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown filter operation: {}",
                op
            ))),
        }
    }

    /// Equality over the whole value tree. Numbers compare by magnitude, so a
    /// stored integer matches an equal float, and arrays and objects compare
    /// element by element so their numbers do too. Every other pairing keeps
    /// `serde_json` equality, which is why a boolean is not equal to a number
    /// and a numeric string is not equal to a number.
    fn values_equal(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Number(left), Value::Number(right)) => {
                Self::compare_numbers(left, right) == Some(CmpOrdering::Equal)
            }
            (Value::Array(left), Value::Array(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right.iter())
                        .all(|(item, other)| Self::values_equal(item, other))
            }
            (Value::Object(left), Value::Object(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(key, item)| {
                        right
                            .get(key)
                            .is_some_and(|other| Self::values_equal(item, other))
                    })
            }
            _ => a == b,
        }
    }

    /// Order two JSON numbers by magnitude. Integers compare as integers, so
    /// two values above 2^53 that share an f64 representation stay distinct,
    /// and a mixed pair compares exactly rather than through a lossy cast.
    fn compare_numbers(a: &serde_json::Number, b: &serde_json::Number) -> Option<CmpOrdering> {
        match (Self::numeric_value(a)?, Self::numeric_value(b)?) {
            (NumericValue::Integer(left), NumericValue::Integer(right)) => Some(left.cmp(&right)),
            (NumericValue::Float(left), NumericValue::Float(right)) => left.partial_cmp(&right),
            (NumericValue::Integer(left), NumericValue::Float(right)) => {
                Self::compare_integer_to_float(left, right)
            }
            (NumericValue::Float(left), NumericValue::Integer(right)) => {
                Self::compare_integer_to_float(right, left).map(CmpOrdering::reverse)
            }
        }
    }

    /// `i128` holds every `serde_json` integer, which is an `i64` or a `u64`,
    /// so the widening is lossless.
    fn numeric_value(number: &serde_json::Number) -> Option<NumericValue> {
        if let Some(value) = number.as_i64() {
            Some(NumericValue::Integer(value as i128))
        } else if let Some(value) = number.as_u64() {
            Some(NumericValue::Integer(value as i128))
        } else {
            number.as_f64().map(NumericValue::Float)
        }
    }

    /// Order an integer against a float without casting the integer to f64.
    /// The float splits into a truncated part, which converts to an integer
    /// exactly, and a fraction that breaks the tie when the integer parts are
    /// equal. A float outside the `i128` range saturates on conversion, and
    /// the comparison still lands on the correct side because every integer
    /// reaching this point fits in a `u64`.
    fn compare_integer_to_float(integer: i128, float: f64) -> Option<CmpOrdering> {
        if float.is_nan() {
            return None;
        }
        if float.is_infinite() {
            return Some(if float.is_sign_positive() {
                CmpOrdering::Less
            } else {
                CmpOrdering::Greater
            });
        }

        let truncated = float.trunc();
        let integer_part = truncated as i128;
        Some(match integer.cmp(&integer_part) {
            CmpOrdering::Equal => truncated.partial_cmp(&float)?,
            ordering => ordering,
        })
    }

    fn compare_values<F>(&self, a: &Value, b: &Value, op: F) -> PyResult<bool>
    where
        F: Fn(CmpOrdering) -> bool,
    {
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                Ok(Self::compare_numbers(n1, n2).is_some_and(op))
            }
            _ => Ok(false),
        }
    }

    fn value_contains(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.contains(s2)),
            (Value::Array(arr), val) => Ok(arr.iter().any(|item| Self::values_equal(item, val))),
            _ => Ok(false),
        }
    }

    fn value_starts_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.starts_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_ends_with(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match (field, target) {
            (Value::String(s1), Value::String(s2)) => Ok(s1.ends_with(s2)),
            _ => Ok(false),
        }
    }

    fn value_in_array(&self, field: &Value, target: &Value) -> PyResult<bool> {
        match target {
            Value::Array(arr) => Ok(arr.iter().any(|item| Self::values_equal(item, field))),
            _ => Ok(false),
        }
    }

    fn value_map_to_python(
        &self,
        value_map: &HashMap<String, Value>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        for (key, value) in value_map {
            let py_value = Self::value_to_python_object(value, py)?;
            dict.set_item(key, py_value)?;
        }

        Ok(dict.into_pyobject(py)?.to_owned().unbind().into_any())
    }

    fn value_to_python_object(value: &Value, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let py_obj = match value {
            Value::Null => py.None(),
            Value::Bool(b) => b.into_pyobject(py)?.to_owned().unbind().into_any(),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i.into_pyobject(py)?.to_owned().unbind().into_any()
                } else if let Some(f) = n.as_f64() {
                    f.into_pyobject(py)?.to_owned().unbind().into_any()
                } else {
                    n.to_string()
                        .into_pyobject(py)?
                        .to_owned()
                        .unbind()
                        .into_any()
                }
            }
            Value::String(s) => s.clone().into_pyobject(py)?.unbind().into_any(),
            Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    py_list.append(Self::value_to_python_object(item, py)?)?;
                }
                py_list.unbind().into_any()
            }
            Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (k, v) in obj {
                    py_dict.set_item(k, Self::value_to_python_object(v, py)?)?;
                }
                py_dict.unbind().into_any()
            }
        };

        Ok(py_obj)
    }

    // 5. BATCH SEARCH METHODS (3 methods)
    /// Internal batch search method for multiple query vectors
    #[instrument(level = "debug", skip(self, vectors, filter_conditions, params, py), fields(
        batch_size = vectors.len(),
        top_k = params.top_k,
        ef = params.ef,
        return_vector = params.return_vector,
        has_filter = filter_conditions.is_some(),
        rerank_factor = params.rerank.and_then(|plan| plan.factor)
    ), err)]
    fn batch_search_internal(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
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
        filter_conditions: Option<&HashMap<String, Value>>,
        params: SearchParams,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        let rust_results = py.detach(|| -> PyResult<Vec<QueryHits>> {
            // The read guard is taken before the graph lock and held across every
            // query in the batch, so the traversal predicate below is a hash lookup
            // rather than a lock acquisition.
            let rev_map = self.rev_map.read().unwrap();
            let live = |internal_id: &usize| rev_map.contains_key(internal_id);

            let hnsw_guard = self.hnsw.read().unwrap();
            let vector_store = self.vectors.read().unwrap();
            let code_store = self.pq_codes.read().unwrap();
            let metadata_store = self.vector_metadata.read().unwrap();

            // The same over-fetch the single query path applies, so a batch of
            // one query returns what that query returns on its own.
            let fetch_k = params.fetch_k(rev_map.len());

            let mut all_results = Vec::with_capacity(vectors.len());

            for vector in vectors {
                // FIX: Process each query vector for space
                let processed_query = self.process_vector_for_space(vector.clone());

                let neighbors = hnsw_guard
                    .search(&processed_query, fetch_k, params.ef, Some(&live))
                    .unwrap_or_else(|_| Vec::new());

                let mut scored: Vec<(&String, f32)> = Vec::with_capacity(neighbors.len());

                for neighbor in neighbors {
                    let internal_id = neighbor.get_origin_id();

                    if let Some(ext_id) = rev_map.get(&internal_id) {
                        // Apply filter if specified
                        if let Some(filter_conds) = filter_conditions {
                            if let Some(meta) = metadata_store.get(ext_id) {
                                if !self.matches_filter(meta, filter_conds)? {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let score = match params.rerank.as_ref() {
                            Some(plan) => rescore_candidate(
                                plan,
                                &processed_query,
                                ext_id,
                                &vector_store,
                                self.pq.as_ref(),
                                &code_store,
                            )
                            .unwrap_or(f32::INFINITY),
                            None => neighbor.distance,
                        };

                        scored.push((ext_id, score));
                    }
                }

                if params.rerank.is_some() {
                    take_best(&mut scored, params.top_k);
                }

                let mut query_results = Vec::with_capacity(scored.len());
                for (ext_id, score) in scored {
                    let metadata = metadata_store.get(ext_id).cloned().unwrap_or_default();
                    // The raw vector where one exists and the reconstruction
                    // from the codes where none does, exactly as the single
                    // query path serves it. Under quantized_only every record
                    // is code held once training completes, so without the
                    // fallback a batch search returned no vectors at all.
                    let vector_data = if params.return_vector {
                        vector_store.get(ext_id).cloned().or_else(|| {
                            let codes = code_store.get(ext_id)?;
                            self.pq.as_ref()?.reconstruct(codes).ok()
                        })
                    } else {
                        None
                    };

                    query_results.push((ext_id.clone(), score, metadata, vector_data));
                }

                all_results.push(query_results);
            }

            Ok(all_results)
        })?;

        // Convert to Python objects
        let mut output = Vec::with_capacity(rust_results.len());
        for batch_result in rust_results {
            let mut py_batch = Vec::with_capacity(batch_result.len());

            for (id, score, metadata, vector_data) in batch_result {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }

                py_batch.push(dict.into());
            }

            output.push(py_batch);
        }

        Ok(output)
    }

    /// Parallel batch processing (for larger batches)
    fn batch_search_parallel(
        &self,
        vectors: &[Vec<f32>],
        filter_conditions: Option<&HashMap<String, Value>>,
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

                    // Taken before the graph lock, as in the other two search paths,
                    // so every path acquires these two locks in the same order.
                    let rev_map = self.rev_map.read().unwrap();
                    let live = |internal_id: &usize| rev_map.contains_key(internal_id);

                    // The same over-fetch the other two search paths apply.
                    let fetch_k = params.fetch_k(rev_map.len());

                    // Brief HNSW search (individual lock per query)
                    let neighbors = {
                        let hnsw_guard = self.hnsw.read().unwrap();
                        hnsw_guard
                            .search(&processed_query, fetch_k, params.ef, Some(&live))
                            .unwrap_or_else(|_| Vec::new())
                    };

                    // Concurrent data lookup
                    let vector_store = self.vectors.read().unwrap();
                    let code_store = self.pq_codes.read().unwrap();
                    let metadata_store = self.vector_metadata.read().unwrap();

                    let mut scored: Vec<(&String, f32)> = Vec::with_capacity(neighbors.len());

                    for neighbor in neighbors {
                        let internal_id = neighbor.get_origin_id();

                        if let Some(ext_id) = rev_map.get(&internal_id) {
                            // Apply filter if specified
                            if let Some(filter_conds) = filter_conditions {
                                if let Some(meta) = metadata_store.get(ext_id) {
                                    if !self.matches_filter(meta, filter_conds)? {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            }

                            let score = match params.rerank.as_ref() {
                                Some(plan) => rescore_candidate(
                                    plan,
                                    &processed_query,
                                    ext_id,
                                    &vector_store,
                                    self.pq.as_ref(),
                                    &code_store,
                                )
                                .unwrap_or(f32::INFINITY),
                                None => neighbor.distance,
                            };

                            scored.push((ext_id, score));
                        }
                    }

                    if params.rerank.is_some() {
                        take_best(&mut scored, params.top_k);
                    }

                    let mut query_results = Vec::with_capacity(scored.len());
                    for (ext_id, score) in scored {
                        let metadata = metadata_store.get(ext_id).cloned().unwrap_or_default();
                        // The same raw-then-reconstruction service as the
                        // single query and sequential batch paths.
                        let vector_data = if params.return_vector {
                            vector_store.get(ext_id).cloned().or_else(|| {
                                let codes = code_store.get(ext_id)?;
                                self.pq.as_ref()?.reconstruct(codes).ok()
                            })
                        } else {
                            None
                        };

                        query_results.push((ext_id.clone(), score, metadata, vector_data));
                    }

                    Ok(query_results)
                })
                .collect();

            results
        })?;

        // Convert to Python objects
        let mut output = Vec::with_capacity(rust_results.len());
        for batch_result in rust_results {
            let mut py_batch = Vec::with_capacity(batch_result.len());

            for (id, score, metadata, vector_data) in batch_result {
                let dict = PyDict::new(py);
                dict.set_item("id", id)?;
                dict.set_item("score", score)?;
                dict.set_item("metadata", self.value_map_to_python(&metadata, py)?)?;

                if let Some(vec) = vector_data {
                    dict.set_item("vector", vec)?;
                }

                py_batch.push(dict.into());
            }

            output.push(py_batch);
        }

        Ok(output)
    }

    // 6. PERSISTENCE INTEGRATION METHODS (2 methods)

    /// Load an index from a .zdb directory structure (Phase 2)
    pub fn load(path: &str) -> PyResult<Self> {
        crate::persistence::load_index(path)
    }

    /// Save HNSW graph using hnsw-rs native file_dump
    #[instrument(level = "info", skip(self), fields(
        vector_count = self.get_vector_count(),
        path = %path.display()
    ))]
    fn save_hnsw_graph(&self, path: &Path) -> PyResult<()> {
        debug!(
            operation = "save_hnsw_graph_start",
            "Starting HNSW graph save"
        );

        // EMPTY INDEX CHECK:
        let vector_count = self.get_vector_count();
        if vector_count == 0 {
            debug!(
                operation = "save_hnsw_graph",
                reason = "empty_index",
                "Skipping HNSW graph dump - index is empty"
            );
            return Ok(());
        }

        let hnsw_guard = self.hnsw.read().unwrap();

        let dump_result = match &*hnsw_guard {
            DistanceType::Cosine(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "cosine",
                    "Using Cosine distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L2(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l2",
                    "Using L2 distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L1(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l1",
                    "Using L1 distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::CosinePQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "cosine_pq",
                    "Using Cosine-PQ distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L2PQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l2_pq",
                    "Using L2-PQ distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
            DistanceType::L1PQ(hnsw) => {
                trace!(
                    operation = "save_hnsw_graph",
                    distance_type = "l1_pq",
                    "Using L1-PQ distance HNSW"
                );
                hnsw.file_dump(path, "hnsw_index")
            }
        };

        match dump_result {
            Ok(basename) => {
                debug!(
                    operation = "save_hnsw_graph_complete",
                    basename = %basename,
                    files_created = %["hnsw.graph", "hnsw.data"].iter()
                        .map(|ext| format!("{}.{}", basename, ext))
                        .collect::<Vec<_>>()
                        .join(", "),
                    "HNSW graph saved successfully"
                );
                Ok(())
            }
            Err(e) => {
                error!(operation = "save_hnsw_graph", error = %e, "HNSW graph dump failed");
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "HNSW graph dump failed: {}",
                    e
                )))
            }
        }
    }

    // ============================================================================
    // PERSISTENCE Minimal Empty Constructor and SETTERS
    // ============================================================================
    /// Minimal constructor for persistence loading - creates empty index with config
    /// No validation needed since config comes from trusted saved state
    pub fn new_empty(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
    ) -> Self {
        let space_normalized = space.to_lowercase();
        let max_layer = 16; // Always use NB_LAYER_MAX for consistency
        let hnsw = DistanceType::new_raw(
            &space_normalized,
            m,
            expected_size,
            max_layer,
            ef_construction,
        );

        HNSWIndex {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            quantization_config: None,
            pq: None,
            pq_codes: RwLock::new(HashMap::new()),
            rerank_calibration: RwLock::new(None),
            metadata: Mutex::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            vector_metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
            id_counter: Mutex::new(0),
            vector_count: Mutex::new(0),
            hnsw: RwLock::new(hnsw),
            writers: Mutex::new(()),
            training_ids: RwLock::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            created_at: chrono::Utc::now().to_rfc3339(),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        }
    }

    /// Set ID mappings (for persistence loading only)
    pub(crate) fn set_id_mappings(
        &mut self,
        id_map: HashMap<String, usize>,
        rev_map: HashMap<usize, String>,
    ) {
        *self.id_map.write().unwrap() = id_map;
        *self.rev_map.write().unwrap() = rev_map;
    }

    /// Set counters (for persistence loading only)
    pub(crate) fn set_counters(&mut self, id_counter: usize, vector_count: usize) {
        *self.id_counter.lock().unwrap() = id_counter;
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Set the vector count alone (for persistence loading only)
    ///
    /// Separate from `set_counters` because the id counter must keep whatever
    /// the graph rebuild advanced it to. Rewinding it would hand out internal
    /// ids the rebuild has already used.
    pub(crate) fn set_vector_count(&mut self, vector_count: usize) {
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Replace the stored record data with what was read from disk
    ///
    /// For persistence loading only, and only after the graph rebuild. The
    /// rebuild routes every record through add(), which stores whatever vector
    /// it was handed, so a record that was reconstructed from PQ codes would
    /// otherwise be kept at full width. Writing the three maps back leaves the
    /// loaded index holding exactly what was saved.
    pub(crate) fn restore_storage_maps(
        &mut self,
        vectors: HashMap<String, Vec<f32>>,
        pq_codes: HashMap<String, Vec<u8>>,
        vector_metadata: HashMap<String, HashMap<String, Value>>,
    ) {
        *self.vectors.write().unwrap() = vectors;
        *self.pq_codes.write().unwrap() = pq_codes;
        *self.vector_metadata.write().unwrap() = vector_metadata;
    }

    /// Restore the graph the save wrote, instead of rebuilding it
    ///
    /// For persistence loading only, and only after `restore_data_fields` has
    /// installed the id mappings and the product quantizer, because both decide
    /// what the dump is checked against.
    ///
    /// Returns the number of graph nodes restored, or the reason the dump
    /// cannot be used. Every reason is a fallback rather than a failure: the
    /// caller rebuilds instead, so a directory whose dump is absent, was written
    /// by a release whose distance types were named differently, or is damaged
    /// still loads.
    pub(crate) fn restore_graph_from_dump(&mut self, dir: &Path) -> Result<usize, String> {
        if rebuild_requested() {
            return Err(format!("{} asked for the rebuild", REBUILD_ENV));
        }

        // A save skips the graph dump entirely when the index holds nothing, so
        // any dump left in an empty index's directory belongs to an earlier
        // save and describes records this one no longer holds.
        let live = self.id_map.read().unwrap().len();
        if live == 0 {
            return Err("the index holds no records".to_string());
        }

        // A trained product quantizer is what makes the saved graph a quantized
        // one, so it is what decides which element type the dump must carry.
        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => Some(pq.clone()),
            _ => None,
        };

        let (graph, nodes) = restore_graph(
            dir,
            &self.space,
            self.m,
            self.ef_construction,
            self.dim,
            pq,
            live,
        )?;

        self.replace_graph(graph);
        Ok(nodes)
    }

    /// Rebuild the graph from the stored PQ codes (for persistence loading only)
    ///
    /// Requires a trained product quantizer, which the loader installs before
    /// any rebuild runs. Replaces the raw graph `new_empty` built with a fresh
    /// PQ graph and inserts every record's codes under the internal id restored
    /// from mappings.bin, so the loaded index is quantized exactly as the saved
    /// one was and no vector is reconstructed to full width on the way.
    ///
    /// A record that has a raw vector but no stored codes is quantized through
    /// the loaded codebook rather than dropped. An intact directory saved by a
    /// trained index holds codes for every record, so that path only runs on a
    /// directory that lost pq_codes.bin while keeping its raw vectors. A record
    /// missing from mappings.bin is assigned a fresh internal id for the same
    /// reason: every record must come back.
    ///
    /// Returns (records inserted, quantized from raw, remapped).
    pub(crate) fn rebuild_graph_from_codes(
        &mut self,
        pq_codes: &HashMap<String, Vec<u8>>,
        vectors: &HashMap<String, Vec<f32>>,
    ) -> Result<(usize, usize, usize), String> {
        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                return Err(
                    "the quantized graph rebuild requires a trained product quantizer".to_string(),
                )
            }
        };

        let mut extra: Vec<(String, Vec<u8>)> = Vec::new();
        for (id, vector) in vectors {
            if !pq_codes.contains_key(id) {
                let codes = pq.quantize(vector).map_err(|e| {
                    format!(
                        "record '{}' has a raw vector but no stored PQ codes, and quantizing \
                         it through the loaded codebook failed: {}",
                        id, e
                    )
                })?;
                extra.push((id.clone(), codes));
            }
        }
        // Sorted so the internal ids the missing records are about to be handed
        // are handed in a fixed order rather than in hash map order.
        extra.sort_by(|a, b| a.0.cmp(&b.0));

        // NB_LAYER_MAX, matching every other construction site in this file.
        let max_layer = 16;
        let new_hnsw = DistanceType::new_pq(
            &self.space,
            self.m,
            self.expected_size,
            max_layer,
            self.ef_construction,
            pq,
        );

        let mut batch: Vec<(&Vec<u8>, usize)> = Vec::with_capacity(pq_codes.len() + extra.len());
        let mut lost: Vec<(&String, &Vec<u8>)> = Vec::new();
        {
            let id_map = self.id_map.read().unwrap();
            for (id, codes) in pq_codes
                .iter()
                .chain(extra.iter().map(|(id, codes)| (id, codes)))
            {
                match id_map.get(id) {
                    Some(&internal_id) => batch.push((codes, internal_id)),
                    None => lost.push((id, codes)),
                }
            }
        }
        let remapped = lost.len();
        lost.sort_by(|a, b| a.0.cmp(b.0));
        for (id, codes) in lost {
            let internal_id = self.get_next_id();
            self.id_map.write().unwrap().insert(id.clone(), internal_id);
            self.rev_map
                .write()
                .unwrap()
                .insert(internal_id, id.clone());
            batch.push((codes, internal_id));
        }

        // Insert in internal id order, which is arrival order, rather than in
        // the order a hash map hands the codes out. Two rebuilds of one
        // directory otherwise wire the graph differently in each process.
        batch.sort_by_key(|&(_, internal_id)| internal_id);

        // Filled before it is installed, and installed under one write guard, so
        // the graph the index holds is never a partly rebuilt one. The batch
        // insert forks to rayon when it is large, which must not happen while
        // the graph's write guard is held.
        if !batch.is_empty() {
            new_hnsw.insert_batch_pq(&batch)?;
        }
        self.replace_graph(new_hnsw);

        Ok((batch.len(), extra.len(), remapped))
    }

    /// Set quantization config (for persistence loading only)
    pub(crate) fn set_quantization_config(&mut self, config: Option<QuantizationConfig>) {
        self.quantization_config = config;
    }

    /// Set PQ instance (for persistence loading only)
    pub(crate) fn set_pq(&mut self, pq: Option<Arc<crate::pq::PQ>>) {
        self.pq = pq;
    }

    /// Set training threshold reached flag (for persistence loading only)
    pub(crate) fn set_training_threshold_reached(&mut self, value: bool) {
        self.training_threshold_reached
            .store(value, std::sync::atomic::Ordering::Release);
    }

    // ============================================================================
    // PERSISTENCE GETTERS - For accessing private fields from persistence module
    // ============================================================================

    /// Get the vector dimension
    pub fn get_dim(&self) -> usize {
        self.dim
    }

    /// Get the distance space (cosine, l2, l1) - changed to a more idiomatic getter
    pub fn space(&self) -> &str {
        // Changed from get_space to space
        &self.space
    }

    /// Get the maximum number of bidirectional links per node
    pub fn get_m(&self) -> usize {
        self.m
    }

    /// Get the construction parameter ef_construction
    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Get the expected size parameter
    pub fn get_expected_size(&self) -> usize {
        self.expected_size
    }

    /// Get the current ID counter value (thread-safe)
    pub fn get_id_counter(&self) -> usize {
        *self.id_counter.lock().unwrap()
    }

    /// Get read access to the vectors HashMap (thread-safe)
    pub fn get_vectors(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Vec<f32>>> {
        self.vectors.read().unwrap()
    }

    /// Get read access to the PQ codes HashMap (thread-safe)
    pub fn get_pq_codes(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Vec<u8>>> {
        self.pq_codes.read().unwrap()
    }

    /// Get read access to the vector metadata HashMap (thread-safe)
    pub fn get_vector_metadata(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, HashMap<String, HashMap<String, Value>>> {
        self.vector_metadata.read().unwrap()
    }

    /// Get read access to the ID map (external ID -> internal ID)
    pub fn get_id_map(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, usize>> {
        self.id_map.read().unwrap()
    }

    /// Get read access to the reverse ID map (internal ID -> external ID)
    pub fn get_rev_map(&self) -> std::sync::RwLockReadGuard<'_, HashMap<usize, String>> {
        self.rev_map.read().unwrap()
    }

    /// Get reference to the quantization configuration
    pub fn get_quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    /// Get reference to the PQ instance
    pub fn get_pq(&self) -> Option<&Arc<crate::pq::PQ>> {
        self.pq.as_ref()
    }

    /// Helper to get quantization subvectors count
    pub fn get_quantization_subvectors(&self) -> usize {
        self.quantization_config
            .as_ref()
            .map(|config| config.subvectors)
            .unwrap_or(1)
    }

    /// Get the index creation timestamp
    pub fn get_created_at(&self) -> &str {
        &self.created_at
    }

    /// Get read access to training IDs (for persistence)
    pub fn get_training_ids(&self) -> std::sync::RwLockReadGuard<'_, Vec<String>> {
        self.training_ids.read().unwrap()
    }

    /// Get training threshold reached flag (for persistence)
    pub fn get_training_threshold_reached(&self) -> bool {
        self.training_threshold_reached
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Set training IDs (for persistence loading only)
    pub(crate) fn set_training_ids(&mut self, ids: Vec<String>) {
        *self.training_ids.write().unwrap() = ids;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        calibrate_rerank_from_sample, interpolate, least_squares_slope, raw_distance_fn,
        rescore_candidate, take_best, DistPQ, RerankCalibration, RerankPlan, SearchParams,
        DEFAULT_RERANK_MIN_CANDIDATES, RERANK_CALIBRATION_CAP_DIVISOR,
        RERANK_CALIBRATION_EXPONENT_MAX, RERANK_CALIBRATION_EXPONENT_MIN,
        RERANK_CALIBRATION_FIT_FRACTIONS, RERANK_CALIBRATION_PAGES, RERANK_CALIBRATION_QUERIES,
        RERANK_CALIBRATION_TARGET, RERANK_CALIBRATION_TOP_K, TRAINING_SAMPLE_SEED,
    };
    use crate::distance::{CosineDist, L1Dist, L2Dist};
    use crate::pq::PQ;
    use rand::seq::SliceRandom;
    // `DistCosine` is the `anndists` implementation these distances replaced.
    // The two graph guard tests keep it on purpose. They guard patches in the
    // vendored crate rather than anything about the distance, their data is
    // deliberately unnormalised, and holding the distance fixed keeps the
    // orphan counts their comments record comparable across relays.
    use hnsw_rs::prelude::{DistCosine, Distance, Hnsw};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, OnceLock};

    // Scale for the quantized graph tests. Small enough for CI and large
    // enough that the neighbour selection heuristic runs, which needs
    // `search_layer` to return more than `2 * M` candidates.
    //
    // Eight bits is the setting the README recommends and it is not
    // negotiable here for speed. Six was tried, and on data with this cluster
    // structure a 64 centroid codebook is coarse enough that every record in a
    // cluster quantizes to the same codes in every subvector. Their distance is
    // then genuinely zero, the diversity heuristic ties for real, and roughly
    // 45 percent of nodes come out with one neighbour. That is the quantizer
    // being too coarse for the data rather than the defect these tests guard,
    // but it is indistinguishable from it at the assertion, so the tests run at
    // the width real indexes use. k-means over 256 centroids is what makes them
    // the slowest in the crate.
    const PQ_N: usize = 1200;
    const PQ_NQ: usize = 100;
    const PQ_DIM: usize = 32;
    const PQ_SUBVECTORS: usize = 8;
    const PQ_BITS: usize = 8;
    const PQ_M: usize = 16;
    const PQ_EF_C: usize = 200;

    /// Clustered unit vectors. Fifty Gaussian centres, points drawn as a centre
    /// plus 0.15 times a Gaussian perturbation, then L2 normalised, with the
    /// centres deliberately left unnormalised. This is the shape real
    /// embeddings have and the specification every quantized measurement in
    /// this project uses, so the figures in the relay reports and the
    /// thresholds below describe the same data.
    ///
    /// Uniform noise was tried and rejected. Its spread is too small relative
    /// to the centre separation, so records within a cluster quantize to the
    /// same codes, their distance is genuinely zero, and the graph partially
    /// collapses for a reason that has nothing to do with what these tests
    /// assert.
    fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let gauss = |rng: &mut StdRng| {
            let u: f32 = rng.random::<f32>().max(1e-12);
            let v: f32 = rng.random::<f32>();
            (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        };
        let centres: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| gauss(&mut rng)).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % 50];
                let mut v: Vec<f32> = (0..dim).map(|d| c[d] + 0.15 * gauss(&mut rng)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect()
    }

    /// The rescored score has to be the number a raw index would report, not a
    /// second implementation of the same formula that happens to agree today.
    #[test]
    fn rerank_scores_come_from_the_raw_distances() {
        let a = vec![0.6f32, 0.8, 0.0, -0.5];
        let b = vec![-0.2f32, 0.3, 0.9, 0.4];

        assert_eq!(
            raw_distance_fn("cosine")(&a, &b),
            CosineDist {}.eval(&a, &b)
        );
        assert_eq!(raw_distance_fn("l2")(&a, &b), L2Dist {}.eval(&a, &b));
        assert_eq!(raw_distance_fn("l1")(&a, &b), L1Dist {}.eval(&a, &b));

        // An unrecognised space falls back the way `DistanceType::new_raw`
        // does, so the score still matches the graph that was built.
        assert_eq!(
            raw_distance_fn("nonsense")(&a, &b),
            CosineDist {}.eval(&a, &b)
        );
    }

    fn plan(factor: usize) -> RerankPlan {
        RerankPlan {
            factor: Some(factor),
            calibration: None,
            distance: raw_distance_fn("cosine"),
        }
    }

    /// The plan a caller who named no factor gets on an index with no
    /// calibration, which is one trained before the calibration existed.
    fn auto_plan() -> RerankPlan {
        RerankPlan {
            factor: None,
            calibration: None,
            distance: raw_distance_fn("cosine"),
        }
    }

    /// The plan a caller who named no factor gets on a calibrated index.
    ///
    /// No page was measured and the page exponent is zero, so the page term is
    /// exactly one at every page. That is the behaviour this relay changed, and
    /// keeping it here is what lets the tests below compare against it.
    fn calibrated_plan(fetch: usize, sample_records: usize, exponent: f64) -> RerankPlan {
        RerankPlan {
            factor: None,
            calibration: Some(RerankCalibration {
                fetch,
                fit_fetches: [fetch; RERANK_CALIBRATION_FIT_FRACTIONS.len()],
                exponent,
                page_fetches: [0; RERANK_CALIBRATION_PAGES.len()],
                page_exponent: 0.0,
                sample_records,
                queries: RERANK_CALIBRATION_QUERIES,
                target: RERANK_CALIBRATION_TARGET,
                millis: 0,
            }),
            distance: raw_distance_fn("cosine"),
        }
    }

    fn params(top_k: usize, rerank: Option<RerankPlan>) -> SearchParams {
        SearchParams {
            top_k,
            ef: 100,
            return_vector: false,
            rerank,
        }
    }

    /// The over-fetch is the factor the caller asked for, and it cannot ask the
    /// graph for more nodes than the index holds.
    #[test]
    fn fetch_k_over_fetches_and_stays_bounded() {
        assert_eq!(params(10, None).fetch_k(10_000), 10);
        assert_eq!(params(10, Some(plan(1))).fetch_k(10_000), 10);
        assert_eq!(params(10, Some(plan(20))).fetch_k(10_000), 200);

        // Capped at the live record count rather than at the product.
        assert_eq!(params(10, Some(plan(20))).fetch_k(50), 50);

        // A page larger than the index still asks for the page, so a short
        // result stays short rather than being cut further.
        assert_eq!(params(500, Some(plan(20))).fetch_k(50), 500);

        // A factor big enough to overflow the multiply degrades to a full scan.
        assert_eq!(params(usize::MAX, Some(plan(2))).fetch_k(50), usize::MAX);
        assert_eq!(params(10, Some(plan(usize::MAX))).fetch_k(50), 50);
    }

    /// The default fetch is a share of the corpus, so it grows with the index
    /// rather than staying at a fixed multiple of the page. That share is a
    /// bound chosen to cover the coarsest structure measured rather than a
    /// property of the codes; see `DEFAULT_RERANK_CORPUS_DIVISOR`.
    #[test]
    fn default_fetch_k_scales_with_the_corpus() {
        // The floor holds below 12,500 records, being where the corpus term
        // reaches it, and 10,000 records is the size the floor is set from.
        assert_eq!(params(10, Some(auto_plan())).fetch_k(10_000), 250);
        assert_eq!(params(10, Some(auto_plan())).fetch_k(5_000), 250);
        assert_eq!(params(10, Some(auto_plan())).fetch_k(12_500), 250);
        assert_eq!(params(10, Some(auto_plan())).fetch_k(15_000), 300);

        // Above that the corpus term carries it.
        assert_eq!(params(10, Some(auto_plan())).fetch_k(100_000), 2_000);
        assert_eq!(params(10, Some(auto_plan())).fetch_k(1_000_000), 20_000);

        // The page term takes over for a large page on a small corpus, where
        // the old default would have fetched twenty times the page.
        assert_eq!(params(100, Some(auto_plan())).fetch_k(10_000), 500);
        assert_eq!(params(100, Some(auto_plan())).fetch_k(1_000_000), 20_000);

        // A corpus smaller than the fetch degrades to a full scan rather than
        // asking the graph for more nodes than it holds.
        assert_eq!(params(10, Some(auto_plan())).fetch_k(120), 120);
        assert_eq!(params(10, Some(auto_plan())).fetch_k(4), 10);
    }

    /// A calibrated index takes the fetch from what training measured, scaled
    /// to the live record count by the exponent training fitted, rather than
    /// from the corpus term. The three arms below are the figures the three
    /// real datasets calibrate to on 10,000 training records.
    #[test]
    fn a_calibration_governs_the_default_fetch() {
        // ada-002 embeddings, which measure a fetch of 164 candidates and fit
        // an exponent of 0.432.
        let ada = calibrated_plan(164, 10_000, 0.432);
        assert_eq!(params(10, Some(ada)).fetch_k(10_000), 287);
        assert_eq!(params(10, Some(ada)).fetch_k(100_000), 776);
        // The corpus term would have asked for 2,000 at that size.
        assert_eq!(params(10, Some(auto_plan())).fetch_k(100_000), 2_000);

        // GloVe word vectors, which measure 904 at a fitted 0.690 and need
        // more than the corpus term gives, not less.
        let glove = calibrated_plan(904, 10_000, 0.690);
        assert_eq!(params(10, Some(glove)).fetch_k(10_000), 1_582);
        assert_eq!(params(10, Some(glove)).fetch_k(100_000), 7_748);

        // SIFT descriptors, which measure 111 at a fitted 0.487.
        let sift = calibrated_plan(111, 10_000, 0.487);
        assert_eq!(params(10, Some(sift)).fetch_k(100_000), 596);

        // A steeper exponent asks for more at the same measurement, which is
        // what a corpus holding a fixed number of groups gets.
        let linear = calibrated_plan(164, 10_000, 1.00);
        assert!(params(10, Some(linear)).fetch_k(100_000) > params(10, Some(ada)).fetch_k(100_000));
    }

    /// The same plan carrying a page exponent and no measured page, which is
    /// what an index trained before the page term existed looks like once the
    /// loader has filled the missing field in.
    fn paged_plan(
        fetch: usize,
        sample_records: usize,
        exponent: f64,
        page_exponent: f64,
    ) -> RerankPlan {
        let mut plan = calibrated_plan(fetch, sample_records, exponent);
        if let Some(calibration) = plan.calibration.as_mut() {
            calibration.page_exponent = page_exponent;
        }
        plan
    }

    /// A plan carrying the pages the calibration measured, which is what an
    /// index trained by this build carries.
    fn measured_page_plan(
        sample_records: usize,
        exponent: f64,
        page_fetches: [usize; RERANK_CALIBRATION_PAGES.len()],
    ) -> RerankPlan {
        let reference = RERANK_CALIBRATION_PAGES
            .iter()
            .position(|&p| p == RERANK_CALIBRATION_TOP_K)
            .expect("the reference page is measured");
        let mut plan = calibrated_plan(page_fetches[reference], sample_records, exponent);
        if let Some(calibration) = plan.calibration.as_mut() {
            calibration.page_fetches = page_fetches;
            calibration.page_exponent = 0.0;
        }
        plan
    }

    /// The measured pages are interpolated, not fitted to one slope.
    ///
    /// The three page fetches below are what the calibration measures on
    /// dbpedia-openai over 10,000 training records, and the relation through
    /// them is convex: 0.431 over the first decade and 0.681 over the second.
    #[test]
    fn the_measured_pages_are_interpolated() {
        let dbpedia = measured_page_plan(10_000, 0.437, [60, 162, 777]);
        let flat = calibrated_plan(162, 10_000, 0.437);

        // The reference page is exactly what it was.
        assert_eq!(
            params(RERANK_CALIBRATION_TOP_K, Some(dbpedia)).fetch_k(50_000),
            params(RERANK_CALIBRATION_TOP_K, Some(flat)).fetch_k(50_000),
        );

        // A page of 100 asks for the measured ratio between the two pages,
        // which is 777 over 162 rather than a line through all three.
        let reference = params(10, Some(dbpedia)).fetch_k(50_000) as f64;
        let hundred = params(100, Some(dbpedia)).fetch_k(50_000) as f64;
        assert!(
            (hundred / reference - 777.0 / 162.0).abs() < 0.02,
            "the page of 100 scaled by {:.4} where the measurement says {:.4}",
            hundred / reference,
            777.0 / 162.0
        );

        // A least squares line through all three would ask for less, which is
        // the reason for interpolating.
        let fitted = paged_plan(162, 10_000, 0.437, 0.556);
        assert!(
            params(100, Some(fitted)).fetch_k(50_000) < hundred as usize,
            "the line asked for at least as much as the interpolation"
        );

        // A page between two measured ones lands between the two fetches.
        let between = params(30, Some(dbpedia)).fetch_k(50_000) as f64 / reference;
        assert!(
            between > 1.0 && between < 777.0 / 162.0,
            "a page of 30 scaled by {:.4}",
            between
        );

        // At a page below the reference the measurement asks for less and the
        // clamp refuses it, so the fetch is the reference fetch.
        assert_eq!(
            params(1, Some(dbpedia)).fetch_k(50_000),
            params(10, Some(dbpedia)).fetch_k(50_000),
        );
    }

    /// The straight lines the page term is read off, on points whose answer is
    /// known.
    #[test]
    fn the_interpolation_is_piecewise_linear() {
        let points = [(0.0, 0.0), (1.0, 2.0), (2.0, 10.0)];

        // On a knot, and between two of them.
        assert!((interpolate(&points, 0.0) - 0.0).abs() < 1e-12);
        assert!((interpolate(&points, 1.0) - 2.0).abs() < 1e-12);
        assert!((interpolate(&points, 0.5) - 1.0).abs() < 1e-12);
        assert!((interpolate(&points, 1.5) - 6.0).abs() < 1e-12);

        // Outside, the nearest segment's slope carries on.
        assert!((interpolate(&points, -1.0) - -2.0).abs() < 1e-12);
        assert!((interpolate(&points, 3.0) - 18.0).abs() < 1e-12);

        // Two points that share an x leave the slope undefined, and the value
        // rather than an infinity is the answer.
        let flat = [(1.0, 5.0), (1.0, 7.0)];
        assert!(interpolate(&flat, 0.0).is_finite());
        assert!(interpolate(&flat, 2.0).is_finite());

        // Fewer than two points leave no line at all.
        assert_eq!(interpolate(&[(3.0, 9.0)], 100.0), 9.0);
        assert_eq!(interpolate(&[], 100.0), 0.0);
    }

    /// The page term scales the fetch and leaves the reference page alone.
    ///
    /// The reference page is what the calibration measured, so a search there
    /// has to ask for exactly what it asked for before the page term existed.
    /// That is the whole guarantee that recall at ten cannot regress.
    #[test]
    fn the_page_term_leaves_the_reference_page_alone() {
        let flat = calibrated_plan(164, 10_000, 0.432);
        let paged = paged_plan(164, 10_000, 0.432, 0.45);

        assert_eq!(
            params(RERANK_CALIBRATION_TOP_K, Some(flat)).fetch_k(50_000),
            params(RERANK_CALIBRATION_TOP_K, Some(paged)).fetch_k(50_000),
        );

        // Below the reference page the fetch does not move. The measurement
        // says a shallower page needs less, and acting on that costs recall;
        // see `page_scale`.
        assert_eq!(
            params(1, Some(paged)).fetch_k(50_000),
            params(10, Some(paged)).fetch_k(50_000),
        );

        // Above it the fetch rises, and it rises less than the page does.
        let at_ten = params(10, Some(paged)).fetch_k(50_000);
        let at_hundred = params(100, Some(paged)).fetch_k(50_000);
        assert!(at_hundred > at_ten, "{} against {}", at_hundred, at_ten);
        assert!(
            at_hundred < 10 * at_ten,
            "{} against {}",
            at_hundred,
            at_ten
        );

        // A page exponent of zero is the behaviour before this was measured,
        // and a page exponent of one is a fetch proportional to the page.
        let ignores_page = paged_plan(164, 10_000, 0.432, 0.0);
        assert_eq!(
            params(100, Some(ignores_page)).fetch_k(50_000),
            params(10, Some(ignores_page)).fetch_k(50_000),
        );
        let linear = paged_plan(164, 10_000, 0.432, 1.0);
        let ten = params(10, Some(linear)).fetch_k(50_000);
        let hundred = params(100, Some(linear)).fetch_k(50_000);
        assert!(
            hundred.abs_diff(10 * ten) <= 10,
            "{} against {}",
            hundred,
            10 * ten
        );

        // The cap still binds, so a page term cannot ask for more than a
        // quarter of the records.
        assert_eq!(
            params(500, Some(linear)).fetch_k(50_000),
            50_000 / RERANK_CALIBRATION_CAP_DIVISOR,
        );
    }

    /// A calibration that measured no page exponent takes the shipped default,
    /// which is what a directory written before the page term existed carries.
    #[test]
    fn a_calibration_without_a_page_exponent_takes_the_default() {
        let json = r#"{"fetch":164,"fit_fetches":[109,124,152,164],"exponent":0.432,
                       "sample_records":10000,"queries":512,"target":0.99,"millis":3164}"#;
        let restored: RerankCalibration = serde_json::from_str(json).expect("older calibration");
        assert_eq!(restored.fetch, 164);
        assert_eq!(restored.page_fetches, [0; RERANK_CALIBRATION_PAGES.len()]);
        assert_eq!(
            restored.page_exponent,
            super::RERANK_CALIBRATION_DEFAULT_PAGE_EXPONENT
        );

        // It reaches the reference page at exactly the fetch it always did, and
        // it deepens above it.
        let plan = RerankPlan {
            factor: None,
            calibration: Some(restored),
            distance: raw_distance_fn("cosine"),
        };
        let flat = calibrated_plan(164, 10_000, 0.432);
        assert_eq!(
            params(10, Some(plan)).fetch_k(50_000),
            params(10, Some(flat)).fetch_k(50_000),
        );
        assert!(params(100, Some(plan)).fetch_k(50_000) > params(100, Some(flat)).fetch_k(50_000));
    }

    /// The slope the exponent is fitted with, on points whose answer is known.
    #[test]
    fn the_least_squares_slope_is_the_slope() {
        let logged = |points: &[(f64, f64)]| -> Vec<(f64, f64)> {
            points.iter().map(|(x, y)| (x.ln(), y.ln())).collect()
        };

        // Doubling the records doubles the fetch, which is an exponent of one.
        let doubling = logged(&[(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0)]);
        let fitted = least_squares_slope(&doubling).expect("four points fit");
        assert!((fitted - 1.0).abs() < 1e-9, "fitted {}", fitted);

        // A fetch that does not move with the records has a slope of zero, and
        // that is what sends the caller to the maximum exponent.
        let flat = logged(&[(1.0, 5.0), (2.0, 5.0), (4.0, 5.0)]);
        assert!(least_squares_slope(&flat).unwrap().abs() < 1e-9);

        // A fetch that falls as the records grow fits negative, which is the
        // signal that the codes resolve nothing at this size.
        let falling = logged(&[(1.0, 8.0), (2.0, 4.0), (4.0, 2.0)]);
        assert!(least_squares_slope(&falling).unwrap() < 0.0);

        // Fewer than two points, and points that all share one record count,
        // both leave the slope undefined rather than enormous.
        assert!(least_squares_slope(&[(1.0, 1.0)]).is_none());
        assert!(least_squares_slope(&[(1.0, 1.0), (1.0, 2.0)]).is_none());
        assert!(least_squares_slope(&[]).is_none());
    }

    /// The shuffle the training sample is drawn in is fixed by its seed, so two
    /// builds over the same records produce the same sample order and two
    /// calibrations over the same codebook produce the same numbers.
    #[test]
    fn the_training_sample_shuffle_is_reproducible() {
        let sample = clustered(500, 32, 909);

        let shuffled = |seed: u64| {
            let mut copy = sample.clone();
            copy.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
            copy
        };

        // The same seed twice is the same order, and it is not the order the
        // records arrived in.
        assert_eq!(
            shuffled(TRAINING_SAMPLE_SEED),
            shuffled(TRAINING_SAMPLE_SEED)
        );
        assert_ne!(shuffled(TRAINING_SAMPLE_SEED), sample);

        // A different seed is a different order, so the fixed seed is doing the
        // work rather than the shuffle being a no-op.
        assert_ne!(
            shuffled(TRAINING_SAMPLE_SEED),
            shuffled(TRAINING_SAMPLE_SEED ^ 1)
        );

        // Every record survives it. A shuffle that dropped or duplicated one
        // would change the codebook as well as the order.
        let mut before: Vec<Vec<f32>> = sample.clone();
        let mut after = shuffled(TRAINING_SAMPLE_SEED);
        let key = |v: &Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        before.sort_by_key(key);
        after.sort_by_key(key);
        assert_eq!(before, after);

        // And the calibration over the shuffled sample is reproducible, given
        // the codebook. The codebook itself is fitted by unseeded k-means, so
        // it is trained once and both calibrations read it.
        let pq = PQ::new(32, 8, 6, 500, None);
        pq.train(&after).unwrap();
        let first = calibrate_rerank_from_sample(&pq, &after, raw_distance_fn("cosine")).unwrap();
        let second = calibrate_rerank_from_sample(&pq, &after, raw_distance_fn("cosine")).unwrap();
        assert_eq!(first.fetch, second.fetch);
        assert_eq!(first.fit_fetches, second.fit_fetches);
        assert_eq!(first.exponent, second.exponent);
    }

    /// A sample the codes resolve completely fits no exponent at all, which is
    /// the guard the fifty cluster generator exists to exercise. Every fraction
    /// of it measures the same depth, the slope is flat rather than positive,
    /// and the calibration takes the maximum exponent rather than a fit that
    /// carries no signal.
    #[test]
    fn a_sample_whose_depth_does_not_grow_takes_the_maximum_exponent() {
        let sample = clustered(1_000, 32, 31337);
        let pq = PQ::new(32, 8, 6, 1_000, None);
        pq.train(&sample).unwrap();

        let calibration = calibrate_rerank_from_sample(&pq, &sample, raw_distance_fn("cosine"))
            .expect("1,000 records calibrate");

        // Fifty clusters of twenty records, and the true ten of a query are its
        // own cluster whatever fraction of the sample is present, so the depth
        // is the cluster and it does not move.
        let deepest = *calibration.fit_fetches.iter().max().unwrap();
        let shallowest = *calibration.fit_fetches.iter().min().unwrap();
        assert!(
            deepest - shallowest <= deepest / 2,
            "fit fetches {:?}",
            calibration.fit_fetches
        );
        assert_eq!(
            calibration.exponent, RERANK_CALIBRATION_EXPONENT_MAX,
            "fit fetches {:?}",
            calibration.fit_fetches
        );
    }

    /// The floor, the page term and the cap all bind a calibration that would
    /// otherwise produce a fetch of two or a fetch of the whole corpus.
    #[test]
    fn the_calibrated_fetch_is_held_between_a_floor_and_a_cap() {
        // A calibration measuring a single candidate cannot fetch one.
        let shallow = calibrated_plan(1, 10_000, 1.00);
        assert_eq!(
            params(10, Some(shallow)).fetch_k(100_000),
            DEFAULT_RERANK_MIN_CANDIDATES
        );

        // The page term still governs a large page on a small corpus.
        assert_eq!(params(100, Some(shallow)).fetch_k(10_000), 500);

        // A calibration measuring the whole sample is capped at a quarter of
        // the live records rather than scanning them.
        let pathological = calibrated_plan(10_000, 10_000, 1.00);
        assert_eq!(
            params(10, Some(pathological)).fetch_k(100_000),
            100_000 / RERANK_CALIBRATION_CAP_DIVISOR
        );

        // The cap is a quarter of the corpus wherever that sits above the
        // floor, and it never cuts below the floor where it does not.
        assert_eq!(params(10, Some(pathological)).fetch_k(2_000), 500);
        assert_eq!(params(10, Some(pathological)).fetch_k(400), 250);

        // The live record count is still the final bound.
        assert_eq!(params(10, Some(pathological)).fetch_k(120), 120);

        // A stored exponent outside the clamp cannot escape it, whichever way
        // it points.
        let steep = calibrated_plan(164, 10_000, 4.0);
        let linear = calibrated_plan(164, 10_000, 1.0);
        assert_eq!(
            params(10, Some(steep)).fetch_k(100_000),
            params(10, Some(linear)).fetch_k(100_000)
        );
        let flat = calibrated_plan(164, 10_000, 0.0);
        let lowest = calibrated_plan(164, 10_000, RERANK_CALIBRATION_EXPONENT_MIN);
        assert_eq!(
            params(10, Some(flat)).fetch_k(100_000),
            params(10, Some(lowest)).fetch_k(100_000)
        );
    }

    /// An explicit factor from the caller overrides the calibration, and zero
    /// still turns rerank off before a plan is ever built.
    #[test]
    fn an_explicit_rerank_overrides_the_calibration() {
        let mut explicit = calibrated_plan(164, 10_000, 0.637);
        explicit.factor = Some(3);
        assert_eq!(params(10, Some(explicit)).fetch_k(100_000), 30);

        // No plan at all is what `rerank = 0` and an unquantized index both
        // produce, and then the fetch is the page.
        assert_eq!(params(10, None).fetch_k(100_000), 10);
    }

    /// The measurement itself, run over a clustered sample the way training
    /// runs it. The rank of a true neighbour is at least one and no deeper
    /// than the sample, the fetch is the target percentile of the pool, and
    /// the exponent is fitted over the sample's fractions.
    #[test]
    fn the_calibration_measures_a_fetch_and_an_exponent() {
        let sample = clustered(400, 64, 4242);
        let pq = PQ::new(64, 8, 6, 400, None);
        pq.train(&sample).unwrap();

        let calibration = calibrate_rerank_from_sample(&pq, &sample, raw_distance_fn("cosine"))
            .expect("a sample of 400 records calibrates");

        assert_eq!(calibration.sample_records, 400);
        assert_eq!(calibration.queries, 400);
        assert_eq!(calibration.target, RERANK_CALIBRATION_TARGET);
        assert!(calibration.fetch >= 1);
        assert!(calibration.fetch <= 400);

        // One fetch per fitting fraction, each no deeper than the records it
        // was measured over, and the last one is the fetch itself.
        assert_eq!(
            *calibration.fit_fetches.last().unwrap(),
            calibration.fetch,
            "fit fetches {:?}",
            calibration.fit_fetches
        );
        for (measured, fraction) in calibration
            .fit_fetches
            .iter()
            .zip(RERANK_CALIBRATION_FIT_FRACTIONS)
        {
            assert!(*measured >= 1, "fit fetches {:?}", calibration.fit_fetches);
            assert!(
                *measured <= (400.0 * fraction) as usize,
                "fetch {} over {} records",
                measured,
                (400.0 * fraction) as usize
            );
        }

        // The exponent is always inside the clamp, whatever the two
        // measurements were.
        assert!(calibration.exponent >= RERANK_CALIBRATION_EXPONENT_MIN);
        assert!(calibration.exponent <= RERANK_CALIBRATION_EXPONENT_MAX);

        // On data a codebook can resolve, the fetch is a small share of the
        // sample rather than most of it.
        assert!(
            calibration.fetch < 200,
            "fetch {} on 400 clustered records",
            calibration.fetch
        );

        // A sample with no room for two measurements is not calibrated.
        assert!(
            calibrate_rerank_from_sample(&pq, &sample[..8], raw_distance_fn("cosine")).is_none()
        );
    }

    /// The page comes back ascending and cut to size, and a non-finite score
    /// orders rather than panicking the sort.
    #[test]
    fn take_best_orders_ascending_and_cuts() {
        let mut scored = vec![
            ("d", 0.9f32),
            ("a", 0.1),
            ("e", f32::INFINITY),
            ("c", 0.5),
            ("b", 0.2),
        ];
        take_best(&mut scored, 3);
        assert_eq!(
            scored.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );

        let mut short = vec![("a", 2.0f32), ("b", 1.0)];
        take_best(&mut short, 10);
        assert_eq!(short.len(), 2);
        assert_eq!(short[0].0, "b");
    }

    /// A candidate is scored from its raw vector where the index kept one and
    /// from its reconstruction otherwise, and both land on the space's own
    /// distance scale.
    #[test]
    fn rescore_prefers_the_raw_vector_and_falls_back_to_the_codes() {
        let data = clustered(64, 8, 99);
        let pq = Arc::new(PQ::new(8, 4, 4, 16, None));
        pq.train(&data).unwrap();

        let query = data[0].clone();
        let stored = data[7].clone();
        let codes = pq.quantize(&stored).unwrap();

        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut pq_codes: HashMap<String, Vec<u8>> = HashMap::new();
        vectors.insert("kept".to_string(), stored.clone());
        pq_codes.insert("kept".to_string(), codes.clone());
        pq_codes.insert("coded".to_string(), codes.clone());

        let p = plan(1);

        // The raw vector wins where there is one, so the score is exact.
        let exact = rescore_candidate(&p, &query, "kept", &vectors, Some(&pq), &pq_codes);
        assert_eq!(exact, Some(CosineDist {}.eval(&query, &stored)));

        // Codes alone still score, against the reconstruction.
        let approximate =
            rescore_candidate(&p, &query, "coded", &vectors, Some(&pq), &pq_codes).unwrap();
        let reconstructed = pq.reconstruct(&codes).unwrap();
        assert_eq!(approximate, CosineDist {}.eval(&query, &reconstructed));

        // Neither is unscoreable rather than silently zero, which is what keeps
        // an unscored candidate from displacing a scored one.
        assert!(rescore_candidate(&p, &query, "absent", &vectors, Some(&pq), &pq_codes).is_none());
        assert!(rescore_candidate(&p, &query, "coded", &vectors, None, &pq_codes).is_none());
    }

    struct PqFixture {
        data: Vec<Vec<f32>>,
        queries: Vec<Vec<f32>>,
        pq: Arc<PQ>,
        codes: Vec<Vec<u8>>,
    }

    /// One trained codebook shared by every quantized graph test, because
    /// k-means over 256 centroids is the expensive part and it is the same
    /// codebook each time. The graphs themselves are built per test, since
    /// that is what is under assertion.
    fn fixture() -> &'static PqFixture {
        static FIXTURE: OnceLock<PqFixture> = OnceLock::new();
        FIXTURE.get_or_init(|| {
            let all = clustered(PQ_N + PQ_NQ, PQ_DIM, 42);
            let data: Vec<Vec<f32>> = all[..PQ_N].to_vec();
            let queries: Vec<Vec<f32>> = all[PQ_N..].to_vec();

            let pq = Arc::new(PQ::new(PQ_DIM, PQ_SUBVECTORS, PQ_BITS, 1000, None));
            pq.train(&data).expect("pq training");
            let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
            let codes = pq.quantize_batch(&refs).expect("quantization");

            PqFixture {
                data,
                queries,
                pq,
                codes,
            }
        })
    }

    /// Build the quantized graph exactly as `insert_pq_codes` does, one code
    /// vector at a time with no query table set.
    fn build_pq_graph(pq: Arc<PQ>, codes: &[Vec<u8>]) -> Hnsw<'static, u8, DistPQ> {
        let hnsw = Hnsw::new(PQ_M, codes.len(), 16, PQ_EF_C, DistPQ::new(pq));
        for (i, c) in codes.iter().enumerate() {
            hnsw.insert((c.as_slice(), i));
        }
        hnsw
    }

    /// Layer zero adjacency keyed by origin id, each list sorted.
    fn layer_zero_adjacency(hnsw: &Hnsw<'static, u8, DistPQ>) -> Vec<(usize, Vec<usize>)> {
        let mut adj: Vec<(usize, Vec<usize>)> = hnsw
            .get_point_indexation()
            .into_iter()
            .map(|p| {
                let mut v: Vec<usize> = p.get_neighborhood_id()[0].iter().map(|x| x.d_id).collect();
                v.sort_unstable();
                (p.get_origin_id(), v)
            })
            .collect();
        adj.sort_unstable_by_key(|(id, _)| *id);
        adj
    }

    /// The strongest assertion available about the quantized graph: that the
    /// data it is built from has any effect on it at all.
    ///
    /// Build the same graph twice, in the same insertion order so the level
    /// sequence is identical, once with each id holding its own codes and once
    /// with every id holding a different record's codes. A graph built on real
    /// distances comes out different. A graph built on a constant comes out
    /// byte for byte identical, which is what `DistPQ::eval` produced for every
    /// release that shipped quantization: it returned infinity whenever no
    /// query table was set, and no insertion path sets one.
    ///
    /// Measured on the shipped v0.4.1 code at 10,000 records, layer zero
    /// adjacency was identical for 10,000 of 10,000 nodes.
    #[test]
    fn quantized_graph_depends_on_the_data() {
        let f = fixture();

        let own = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &f.codes));

        let mut shuffled = f.codes.clone();
        shuffled.reverse();
        let other = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &shuffled));

        assert_eq!(own.len(), PQ_N);
        assert_eq!(other.len(), PQ_N);

        let identical = own
            .iter()
            .zip(other.iter())
            .filter(|((id_a, a), (id_b, b))| id_a == id_b && a == b)
            .count();

        assert!(
            identical * 20 < PQ_N,
            "layer zero adjacency is identical for {} of {} nodes when every id is given a \
             different record's codes, so the graph is not being built on the codes. \
             DistPQ::eval is returning a constant on the insertion path.",
            identical,
            PQ_N
        );
    }

    /// A graph whose distances all tie leaves every node with one neighbour,
    /// because the diversity heuristic in `select_neighbours` rejects a
    /// candidate that is at least as close to an already chosen neighbour as it
    /// is to the new point, and under a total tie that is every candidate after
    /// the first. Measured on the shipped code, layer zero out-degree was
    /// exactly one for 99.64 percent of nodes and a traversal reached 33 of
    /// 10,000.
    #[test]
    fn quantized_graph_layer_zero_out_degree() {
        let f = fixture();
        let adj = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &f.codes));

        let degenerate = adj.iter().filter(|(_, n)| n.len() <= 1).count();
        assert!(
            degenerate * 100 < PQ_N,
            "{} of {} nodes have layer zero out-degree of one or less; the quantized graph \
             has collapsed to a star",
            degenerate,
            PQ_N
        );

        let total: usize = adj.iter().map(|(_, n)| n.len()).sum();
        let mean = total as f64 / PQ_N as f64;
        assert!(
            mean > (PQ_M as f64) / 2.0,
            "mean layer zero out-degree is {:.2}, expected well above {} for m = {}",
            mean,
            PQ_M / 2,
            PQ_M
        );
    }

    /// Quantized search has to find the right answers, not merely return the
    /// right number of them. Measured on the shipped code at this scale the
    /// graph reached 33 nodes of 1,200 and recall was under one percent, so
    /// the threshold below fails against the old behaviour by a wide margin.
    ///
    /// The ceiling is the quantizer rather than the graph, so this asserts a
    /// floor and not equality with the raw path.
    #[test]
    fn quantized_graph_recall_against_brute_force() {
        const K: usize = 10;

        let f = fixture();
        let (data, queries) = (&f.data, &f.queries);
        let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

        let mut hits = 0usize;
        let mut returned = usize::MAX;
        for q in queries.iter() {
            let found = {
                let _lut = hnsw.get_distance().install_query_lut(q).expect("query lut");
                let dummy = vec![0u8; PQ_SUBVECTORS];
                hnsw.search(&dummy, K, 100)
            };
            returned = returned.min(found.len());

            let mut truth: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    (
                        v.iter().zip(q.iter()).map(|(x, y)| (x - y) * (x - y)).sum(),
                        j,
                    )
                })
                .collect();
            truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let truth: HashSet<usize> = truth[..K].iter().map(|x| x.1).collect();

            hits += found.iter().filter(|n| truth.contains(&n.d_id)).count();
        }

        assert_eq!(returned, K, "a top {} request came back short", K);

        let recall = hits as f64 / (K * PQ_NQ) as f64;
        assert!(
            recall > 0.30,
            "quantized recall at top {} is {:.4}, which is far below what these codes support",
            K,
            recall
        );
    }

    /// The whole graph has to be reachable, which is the property the shipped
    /// code lost most visibly. Asking for more results than the graph can reach
    /// is what exposed it: a request for 1,000 came back with 34.
    #[test]
    fn quantized_graph_is_fully_reachable() {
        let f = fixture();
        let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

        let found = {
            let _lut = hnsw
                .get_distance()
                .install_query_lut(&f.data[0])
                .expect("query lut");
            let dummy = vec![0u8; PQ_SUBVECTORS];
            hnsw.search(&dummy, PQ_N, PQ_N)
        };

        assert_eq!(
            found.len(),
            PQ_N,
            "a request for all {} records returned {}, so the traversal cannot reach the \
             whole graph",
            PQ_N,
            found.len()
        );
    }

    /// The symmetric distance must not leak into search. With a query table set
    /// the distance has to be the asymmetric one, byte for byte as before, so
    /// a quantized graph and a raw graph rank a query's own record the same way.
    #[test]
    fn quantized_search_still_uses_the_query_table() {
        let f = fixture();
        let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

        // The ADC distance from a query to a stored code, computed directly.
        let query = &f.data[7];
        let lut = f.pq.compute_adc_lut(query).expect("adc lut");
        let expected: f32 = f.codes[7]
            .iter()
            .enumerate()
            .map(|(sv, &c)| lut[sv][c as usize])
            .sum();

        let found = {
            let _lut = hnsw
                .get_distance()
                .install_query_lut(query)
                .expect("query lut");
            let dummy = vec![0u8; PQ_SUBVECTORS];
            hnsw.search(&dummy, 10, 200)
        };

        assert!(
            found.iter().any(|n| n.d_id == 7),
            "a record could not find itself"
        );

        // Every distance the search reports must be the asymmetric one. The
        // symmetric table would give a different number here, since it compares
        // the dummy query's all-zero codes rather than the query itself.
        for n in found.iter() {
            let adc: f32 = f.codes[n.d_id]
                .iter()
                .enumerate()
                .map(|(sv, &c)| lut[sv][c as usize])
                .sum();
            assert!(
                (n.distance - adc).abs() <= 1e-4 * adc.max(1.0),
                "search reported {} for record {} where its asymmetric distance is {}, \
                 so the query path is no longer using the query table",
                n.distance,
                n.d_id,
                adc
            );
        }

        // The record's own ADC distance is the smallest of the ten returned,
        // which is what the ranking has to produce.
        assert!(
            found[0].distance <= expected + 1e-4,
            "the top hit scored {} against the query's own record at {}",
            found[0].distance,
            expected
        );
    }

    /// Guards the vendored hnsw_rs patch that files reverse links at the
    /// layer being processed instead of at the inserting point's own top
    /// layer. Without the patch, points assigned a level above zero lose
    /// their layer-zero inbound adjacency and can become unreachable to
    /// similarity search, and at this index size roughly one to two
    /// percent of self-queries fail. A failure here means the patch was
    /// lost, most likely during an hnsw_rs upgrade. See
    /// vendor/hnsw_rs/ZEUSDB-PATCH.md.
    ///
    /// Insertion is sequential on purpose. `parallel_insert` assigns levels
    /// in whatever order threads reach the level generator, so the graph
    /// varies between runs even under the fixed seed and the test was
    /// intermittently red. Building one vector at a time makes the graph a
    /// function of the data and the parameters alone, which is also the
    /// path every non-quantized index takes through `add()`. The defect
    /// this test guards is not specific to the parallel path.
    #[test]
    fn self_query_reachability() {
        const N: usize = 3000;
        const DIM: usize = 32;

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let hnsw = Hnsw::new(16, N, 16, 200, DistCosine {});
        for (i, v) in data.iter().enumerate() {
            hnsw.insert((v.as_slice(), i));
        }

        let failures: Vec<usize> = (0..N)
            .filter(|&i| hnsw.search(&data[i], 1, 64).first().map(|n| n.d_id) != Some(i))
            .collect();

        assert!(
            failures.is_empty(),
            "{} of {} points cannot find themselves by self-query (first: {:?}); \
             the hnsw_rs reverse link layer patch is missing",
            failures.len(),
            N,
            &failures[..failures.len().min(10)]
        );
    }

    /// Guards the vendored hnsw_rs patch that stops the layer-zero overflow
    /// pop from evicting a point's last inbound link. Without the patch the
    /// pop always discards the farthest entry, so a point whose only inbound
    /// link happens to be that entry is left with no layer-zero in-edge and
    /// becomes an orphan that no search can reach through the graph. A
    /// failure here means the patch was lost, most likely during an hnsw_rs
    /// upgrade. See vendor/hnsw_rs/ZEUSDB-PATCH.md.
    ///
    /// In-degree is counted from the adjacency lists rather than from the
    /// patch's own counters, so the assertion holds against the graph itself
    /// and stays meaningful if the counters are ever wrong.
    ///
    /// `M` is 4 rather than the shipped 16 on purpose. The layer-zero
    /// neighbour cap is `2 * M`, so a small `M` fills lists early and makes
    /// the overflow pop frequent, which is the only site that can strand a
    /// point. Whether the shipped `M` of 16 strands points depends on the
    /// data model rather than on index size. On uniform vectors like these
    /// it strands none up to 30,000 points, while on clustered data, 50
    /// Gaussian clusters at sigma 0.15 in 768 dimensions, it strands 6 of
    /// 10,000. A small `M` lets this uniform generator fail fast instead,
    /// and the unpatched crate strands 24 of these 5,000 points.
    ///
    /// Insertion is sequential for the same reason as `self_query_reachability`.
    #[test]
    fn layer_zero_in_degree() {
        const N: usize = 5000;
        const DIM: usize = 128;
        const M: usize = 4;

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let hnsw = Hnsw::new(M, N, 16, 200, DistCosine {});
        for (i, v) in data.iter().enumerate() {
            hnsw.insert((v.as_slice(), i));
        }

        let mut in_degree = vec![0usize; N];
        let mut nb_seen = 0usize;
        for point in hnsw.get_point_indexation() {
            nb_seen += 1;
            for neighbour in &point.get_neighborhood_id()[0] {
                in_degree[neighbour.d_id] += 1;
            }
        }
        assert_eq!(nb_seen, N, "walked {} points, expected {}", nb_seen, N);

        let orphans: Vec<usize> = (0..N).filter(|&i| in_degree[i] == 0).collect();

        assert!(
            orphans.is_empty(),
            "{} of {} points have zero layer-zero in-degree (first: {:?}); \
             the hnsw_rs overflow pop guard is missing",
            orphans.len(),
            N,
            &orphans[..orphans.len().min(10)]
        );
    }
}
