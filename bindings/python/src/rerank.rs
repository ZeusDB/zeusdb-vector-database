//! The rerank fetch: how far a quantized search over-fetches before it rescores
//! against raw vectors, and how the depth it needs is measured on the index's
//! own data when training completes.
//!
//! Everything here is a free function or a plain value. Nothing reads an index,
//! which is what lets the whole rule be exercised without building one, and it
//! is why this sits beside `hnsw_index` rather than inside it. The index decides
//! whether a search reranks at all and hands the training sample to the
//! calibration; see `hnsw_index::search::rerank_plan` and
//! `hnsw_index::training::calibrate_rerank`.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use zeusdb_vector_core::{CosineDist, Distance, DotDist, L1Dist, L2Dist, VectorGraph, PQ};
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
/// size at which the codes resolve the data at all, which is the case of a
/// group smaller than the page. The fit carries no signal there, its
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
/// Mean recall at 10 measured on built indexes, one query at a time through the
/// ordinary search path, 1,000 queries. The requirement is the smallest fetch on
/// the same index reaching 0.99, read off a sweep of the explicit `rerank`
/// argument over that index at 300 queries.
///
///   records   corpus     calibrated   requirement   ratio   recall
///    50,000   ada-002           554           860    0.64   0.9883
///    50,000   GloVe           6,534         3,060    2.14   0.9982
///    50,000   SIFT              465           370    1.26   0.9953
///   100,000   ada-002           747           620    1.20   0.9907
///   100,000   GloVe          11,439         4,620    2.48   0.9985
///   100,000   SIFT              656           450    1.46   0.9968
///
/// A requirement has to be read off the index the fetch will run on, and not
/// off the codes. A fetch is served by a traversal of the graph over the codes
/// and a traversal of width F does not return the F nearest by code.
///
/// # Why the ratio is not the same on the three
///
/// The rule is `SAFETY * fetch * ratio.powf(slope + BIAS)`, and everything in it
/// except `fetch` and `slope` is the same constant on every index. Taking the
/// raw least squares slope alone, with no safety factor and no bias, the
/// extrapolation to 100,000 records reads 4,626 candidates on GloVe against a
/// requirement of 4,620, being 1.001 times what the index needs. On ada-002 it
/// reads 302 against 620 and on SIFT 265 against 450, being 0.49 and 0.59 times.
/// **The uncorrected extrapolation is exact on GloVe and about half of what the
/// other two need**, and the correction those constants apply is 2.47 at a
/// tenfold ratio, being 1.75 for the safety factor and 1.41 for the bias.
///
/// So the margin on GloVe is not a defect in GloVe's calibration. It is the
/// correction sized for the dataset whose calibration is least accurate,
/// applied to the one whose calibration needs none. The correction the three
/// need spans 2.05 to 1.00 at 100,000 records and 3.46 to 1.06 at 50,000, and
/// one constant cannot serve that.
///
/// Neither constant can come down. At 100,000 records the constant that would
/// bring GloVe to a 1.20 ratio, being what ada-002 has there, is 0.48 of the
/// present one, and it takes the ada-002 fetch from 747 to 359 where a sweep of
/// that index reads recall 0.9787 at 360 candidates. ada-002 also sits below the
/// target at three of the four sizes measured, reading 0.9878, 0.9800 and 0.9883
/// at 10,000, 25,000 and 50,000 records, so it has nothing to give back.
///
/// What separates them is the ratio between the requirement read off a built
/// index and the depth this calibration measures in the exact code ordering over
/// the training sample. At 10,000 records those read 340 against 158 on ada-002,
/// 880 against 1,017 on GloVe and 140 against 119 on SIFT, so the sample
/// understates the index by 2.15 times on one corpus and overstates it by 1.16
/// on another. Nothing in the training sample measures that ratio, because the
/// graph the fetch will traverse does not exist until the rebuild that follows
/// the calibration. Measuring it there is the change that would let the safety
/// factor come down, and it is a change to what training measures rather than to
/// a constant.
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
/// sits above every measured cell and below a full scan. It exists for the case
/// where the codes resolve nothing and the depth the calibration measures is
/// most of the sample. It bounds that case rather than
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
pub(crate) fn interpolate(points: &[(f64, f64)], x: f64) -> f64 {
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

/// Whether this space's raw distance requires a unit vector
///
/// One place decides it, so the exact scan, which has the space in hand, and
/// the rerank plan, which carries the answer past it, cannot disagree about
/// which spaces are in the set.
pub(crate) fn reconstruction_needs_unit(space: &str) -> bool {
    space == "cosine"
}

/// Put a reconstruction on the footing the space's raw distance expects
///
/// A reconstruction is the concatenation of the nearest centroid per subvector
/// and nothing renormalises it, so it is not a unit vector even where the
/// record it approximates was. Measured norms run 0.85 to 0.96 at the shipped
/// dbpedia default and down to 0.41 at two subvectors and four bits.
///
/// `CosineDist::eval` is `1 - dot` and assumes both arguments are unit, so
/// handing it a raw reconstruction returns neither a cosine distance nor a
/// squared L2. On one `quantized_only` cosine index at 25,000 records that put
/// the same record at 0.19118 through a narrow filter's exact scan and 0.19678
/// through the traversal, with a true cosine distance of 0.10381, and the ratio
/// between the two paths ran with the distance rather than being a constant.
///
/// Normalising here is what `add` would have done to the same vector. The
/// doctrine `rescore_candidate` states is that a rescored score is the number a
/// raw index holding that vector would report, and a raw cosine index
/// normalises on the way in, so this is that doctrine applied rather than an
/// exception to it.
///
/// Every other space takes the vector as it is. `l2` and `l1` are defined on
/// unnormalised input and a `dot` index is never quantized, so nothing else
/// reaches here with a reconstruction at all.
pub(crate) fn prepare_reconstruction(needs_unit: bool, mut reconstructed: Vec<f32>) -> Vec<f32> {
    if !needs_unit {
        return reconstructed;
    }
    let norm: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
    // A zero reconstruction stays zero, which `cosine_normalized` answers as
    // distance one, and dividing by zero would answer it as NaN.
    if norm > 0.0 {
        for x in reconstructed.iter_mut() {
            *x /= norm;
        }
    }
    reconstructed
}

/// The raw vector distance for a space
///
/// These are the same `zeusdb_vector_core` distances `VectorGraph::new_raw`
/// hands to a raw graph, so a rescored score is the number a raw index would
/// have reported for the same pair rather than a second implementation of the
/// same formula.
pub(crate) fn raw_distance_fn(space: &str) -> fn(&[f32], &[f32]) -> f32 {
    match space {
        "l2" => |a: &[f32], b: &[f32]| L2Dist {}.eval(a, b),
        "l1" => |a: &[f32], b: &[f32]| L1Dist {}.eval(a, b),
        // A dot index is never quantized, so nothing rescores against this. The
        // filtered exact scan does score with it, and scoring that page with
        // cosine while the traversal scored with the inner product would have
        // been two different orderings from one query.
        "dot" => |a: &[f32], b: &[f32]| DotDist {}.eval(a, b),
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
pub(crate) fn measure_rerank_fetches(
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

    let subvectors = pq.subvectors();

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
pub(crate) fn calibrate_rerank_from_sample(
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
pub(crate) fn least_squares_slope(points: &[(f64, f64)]) -> Option<f64> {
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
pub(crate) struct RerankPlan {
    /// Candidates to pull from the graph per requested result, when the caller
    /// named a factor. `None` means the caller named none and the fetch comes
    /// from the calibration, or from the live record count where there is no
    /// calibration; see `SearchParams::fetch_k`.
    pub(crate) factor: Option<usize>,
    /// What training measured on this index's own data, where it ran. `None`
    /// for an index trained before the calibration existed.
    pub(crate) calibration: Option<RerankCalibration>,
    /// The space's raw vector distance.
    pub(crate) distance: fn(&[f32], &[f32]) -> f32,
    /// Whether `distance` requires a unit vector, which decides what happens to
    /// a reconstruction before it is scored. See `prepare_reconstruction`.
    pub(crate) unit_reconstruction: bool,
}

/// The settings a search carries once its input has been parsed
///
/// Bundled rather than threaded through one at a time, because the three batch
/// entry points forward every one of them unchanged and the page size, the
/// traversal breadth and the rerank plan are read together wherever they are
/// read at all.
#[derive(Clone, Copy)]
pub(crate) struct SearchParams {
    pub(crate) top_k: usize,
    pub(crate) ef: usize,
    pub(crate) return_vector: bool,
    pub(crate) rerank: Option<RerankPlan>,
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
    pub(crate) fn fetch_k(&self, live_records: usize) -> usize {
        match self.rerank {
            Some(plan) => match plan.factor {
                Some(factor) => self
                    .top_k
                    .saturating_mul(factor)
                    .min(live_records.max(self.top_k)),
                None => default_rerank_fetch(plan.calibration, live_records, self.top_k),
            },
            None => self.top_k,
        }
    }
}

/// The fetch a search asks for when the caller named no factor
///
/// One rule, read by `SearchParams::fetch_k` on the search path and by
/// `get_stats` when it reports `rerank_default_fetch`. The two used to state it
/// separately, so the figure `get_stats` reported was a second copy of the rule
/// rather than a reading of it, and a change to either would have left the
/// reported number describing a fetch no search performs.
///
/// A calibrated index takes what training measured on its own data, scaled to
/// the live record count and to the requested page; see `RerankCalibration`.
/// An index with no calibration takes the largest of the corpus term, the floor
/// and the page term, for the reasons recorded on those three constants. Either
/// way the live record count is the final bound, since the graph cannot return
/// more nodes than it holds.
pub(crate) fn default_rerank_fetch(
    calibration: Option<RerankCalibration>,
    live_records: usize,
    top_k: usize,
) -> usize {
    let wanted = match calibration {
        Some(calibration) => calibration.fetch_at(live_records, top_k),
        None => (live_records / DEFAULT_RERANK_CORPUS_DIVISOR)
            .max(DEFAULT_RERANK_MIN_CANDIDATES)
            .max(top_k.saturating_mul(DEFAULT_RERANK_PAGE_FACTOR)),
    };
    wanted.min(live_records.max(top_k))
}

/// Where a raw vector is reached, now that there is no map of them.
///
/// Two array reads and no hashing beyond the one `id_map` lookup every caller
/// here was already doing: the external id gives the internal id, the graph
/// turns that into a node index, and the store is addressed by node. It is a
/// borrow of both rather than a copy of either, so a caller holds the two read
/// guards it already held and passes this down.
///
/// The two are always taken `id_map` first and the graph second, which is the
/// order every path in the crate takes them in.
#[derive(Clone, Copy)]
pub(crate) struct RawVectors<'a> {
    /// The external id to internal id map, which is the record set.
    pub(crate) id_map: &'a HashMap<String, usize>,
    /// The graph, which owns the store the vectors live in.
    pub(crate) graph: &'a VectorGraph,
}

impl RawVectors<'_> {
    /// One record's raw vector, or `None` where the index keeps none for it.
    #[inline]
    pub(crate) fn get(&self, ext_id: &str) -> Option<&[f32]> {
        self.graph.raw_vector(*self.id_map.get(ext_id)?)
    }

    /// Whether the index keeps a raw vector for this record.
    #[inline]
    pub(crate) fn contains(&self, ext_id: &str) -> bool {
        self.get(ext_id).is_some()
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
pub(crate) fn rescore_candidate(
    plan: &RerankPlan,
    query: &[f32],
    ext_id: &str,
    vectors: RawVectors<'_>,
    pq: Option<&Arc<PQ>>,
    pq_codes: &HashMap<String, Vec<u8>>,
) -> Option<f32> {
    if let Some(stored) = vectors.get(ext_id) {
        return Some((plan.distance)(query, stored));
    }
    let reconstructed = pq?.reconstruct(pq_codes.get(ext_id)?).ok()?;
    let reconstructed = prepare_reconstruction(plan.unit_reconstruction, reconstructed);
    Some((plan.distance)(query, &reconstructed))
}

/// Order a rescored page and cut it to the requested size
///
/// `total_cmp` rather than `partial_cmp`, so a non-finite score orders rather
/// than panicking the sort.
pub(crate) fn take_best<T>(scored: &mut Vec<(T, f32)>, top_k: usize) {
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(top_k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use zeusdb_vector_core::{test_support::clustered, CosineDist, PQ};

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

        // An unrecognised space falls back the way `VectorGraph::new_raw`
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
            unit_reconstruction: reconstruction_needs_unit("cosine"),
        }
    }

    /// The plan a caller who named no factor gets on an index with no
    /// calibration, which is one trained before the calibration existed.
    fn auto_plan() -> RerankPlan {
        RerankPlan {
            factor: None,
            calibration: None,
            distance: raw_distance_fn("cosine"),
            unit_reconstruction: reconstruction_needs_unit("cosine"),
        }
    }

    /// The plan a caller who named no factor gets on a calibrated index.
    ///
    /// No page was measured and the page exponent is zero, so the page term is
    /// exactly one at every page. That is the behaviour before the page term,
    /// and keeping it here is what lets the tests below compare against it.
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
            unit_reconstruction: reconstruction_needs_unit("cosine"),
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
            RERANK_CALIBRATION_DEFAULT_PAGE_EXPONENT
        );

        // It reaches the reference page at exactly the fetch it always did, and
        // it deepens above it.
        let plan = RerankPlan {
            factor: None,
            calibration: Some(restored),
            distance: raw_distance_fn("cosine"),
            unit_reconstruction: reconstruction_needs_unit("cosine"),
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

        // The raw vectors live in the graph now, so the fixture is a real one
        // holding the one record that keeps its vector. "coded" has an internal
        // id and no node, which is the shape of a record a quantized_only index
        // holds, and "absent" has no id at all.
        let mut id_map: HashMap<String, usize> = HashMap::new();
        id_map.insert("kept".to_string(), 1);
        id_map.insert("coded".to_string(), 2);
        let mut graph = VectorGraph::new_raw("cosine", 8, 16, 4, 16, 64);
        graph.insert(&stored, 1);
        let vectors = RawVectors {
            id_map: &id_map,
            graph: &graph,
        };

        let mut pq_codes: HashMap<String, Vec<u8>> = HashMap::new();
        pq_codes.insert("kept".to_string(), codes.clone());
        pq_codes.insert("coded".to_string(), codes.clone());

        let p = plan(1);

        // The raw vector wins where there is one, so the score is exact.
        let exact = rescore_candidate(&p, &query, "kept", vectors, Some(&pq), &pq_codes);
        assert_eq!(exact, Some(CosineDist {}.eval(&query, &stored)));

        // Codes alone still score, against the reconstruction, and under cosine
        // the reconstruction is normalised first. It arrives from the codebook
        // at whatever length the centroids give it, and `CosineDist` is
        // `1 - dot` on a pair assumed to be unit, so handing it the raw
        // reconstruction returned a third quantity that was neither a cosine
        // distance nor a squared L2. See `prepare_reconstruction`.
        let approximate =
            rescore_candidate(&p, &query, "coded", vectors, Some(&pq), &pq_codes).unwrap();
        let reconstructed = pq.reconstruct(&codes).unwrap();
        let length: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (length - 1.0).abs() > 1e-3,
            "the fixture must produce a reconstruction that is not unit, got {length}"
        );
        let unit: Vec<f32> = reconstructed.iter().map(|x| x / length).collect();
        assert_eq!(approximate, CosineDist {}.eval(&query, &unit));
        assert_ne!(approximate, CosineDist {}.eval(&query, &reconstructed));

        // Neither is unscoreable rather than silently zero, which is what keeps
        // an unscored candidate from displacing a scored one.
        assert!(rescore_candidate(&p, &query, "absent", vectors, Some(&pq), &pq_codes).is_none());
        assert!(rescore_candidate(&p, &query, "coded", vectors, None, &pq_codes).is_none());
    }
}
