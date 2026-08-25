//! Building an index, and the bounds a declaration has to satisfy.
//!
//! `build` is the only construction path from Python and enforces every rule
//! that governs a valid index, which is what makes the Python factory and the
//! Rust constructor agree. `new_empty` is the loader's constructor and validates
//! nothing, because its configuration comes from a directory this crate wrote.

use super::locks::{MutexAt, RwLockAt};
use super::{HNSWIndex, QuantizationConfig, StorageMode, MAX_LAYER};
use crate::columns::{validate_indexed_fields, ColumnStore};
use crate::graph::VectorGraph;
use crate::pq::PQ;
use chrono::Utc;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, instrument, trace};
/// Widest `dim` a caller may declare, and the widest a saved index may name.
///
/// `dim` is the width of one vector buffer, so sizing that buffer from the
/// declared width is the first allocation either creation or loading makes.
/// `create(dim=2**40)` asked the allocator for 4,398,046,511,104 bytes and
/// **aborted the process** with exit status 3221226505. An allocation failure
/// does not unwind, so no `catch_unwind` sees one and a Python caller gets a
/// dead interpreter with no traceback.
///
/// The loader carried this bound first, on the reasoning that bounding it here
/// would change `create()`'s documented contract. It changes it, and the README
/// row now says so, which is what makes the two doors refuse the same values.
///
/// The ceiling is derived rather than measured. One vector at this width is
/// 262,144 bytes, and the widest embedding any published model produces is an
/// order of magnitude below it, so the bound refuses nothing an embedding model
/// can generate. What it costs at the ceiling is that width per record: a
/// thousand records at 65,536 values is 262 MB, which is an allocation the
/// process can refuse rather than one it dies on.
pub(crate) const MAX_DIM: usize = 65_536;

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

/// Largest `ef_construction` a caller may declare, and the largest a saved
/// index may name.
///
/// `ef_construction` is the width of the candidate search every insertion
/// runs, and `search_layer` in `graph/traverse.rs` sizes its two candidate
/// heaps from that width before it visits a node, 8 bytes a slot. The
/// allocation is not fallible and it happens on the first `add()` rather than
/// at creation, so `create(ef_construction=2**40)` succeeded and the first
/// insertion asked for 8,796,093,022,208 bytes and **aborted the process**
/// with exit status 3221226505. The same width is reached by `rebuild()` and
/// by a `config.json` naming it, which is why the bound lives in
/// `validate_index_parameters` with the other three rather than in `build`.
///
/// The ceiling is reasoned from `m` rather than invented. The neighbour
/// selection heuristic runs only while `ef_construction` is above the
/// neighbour budget of `2 * m`, that budget is 512 at the largest `m` of 256,
/// and the default of 200 sits 6.25 times above the budget at the default
/// `m` of 16. Carrying that margin to the largest `m` is 3,200, and 4,096 is
/// the first power of two above it, so every `m` the index allows has the
/// default's headroom available and nothing a real build asks for is
/// refused. The highest comparable ceiling in the field is Lucene's beam
/// width of 3,200, and pgvector stops at 1,000.
///
/// What it costs at the ceiling, measured on 20,000 records of 64
/// dimensions. 2.28 s to build at 200 and 30.0 s at 4,096, thirteen times the
/// default, against 47.4 s at 8,192. The README's 50,000 record build of
/// 1,536 dimensions, linear above 100 at 0.31 s per unit of width, would run
/// about 21 minutes at the ceiling. At the ceiling the two heaps are 64 KiB,
/// which is an allocation the process can refuse rather than one it dies on.
pub(crate) const MAX_EF_CONSTRUCTION: usize = 4_096;

/// Every rule a valid index declaration has to satisfy, returning the
/// normalised space.
///
/// Extracted from `build` so that the loader enforces the same rules on the
/// same values rather than a copy of them. `build` takes its declaration from
/// a caller and `load_config` takes it from `config.json`, and a directory is
/// only trusted to the extent that something checked it: a config naming a
/// zero `dim` or a zero `m` used to reach `Backend::sized`, which clamps both
/// silently, so the index came back at a width or a degree the directory never
/// held. The messages are the ones `build` raised, because the invalid value is
/// the same value whichever door it came through.
///
/// `source` prefixes every message and is empty for `build`, whose caller is
/// looking at the argument they passed. The loader passes the path of the file
/// the value came out of, because a caller reading `dim must be positive` off a
/// `load()` has no argument of their own to look at.
pub(crate) fn validate_index_parameters(
    dim: usize,
    space: &str,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    source: &str,
) -> PyResult<String> {
    if dim == 0 {
        error!(
            operation = "validation",
            field = "dim",
            value = dim,
            "Invalid dimension"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}dim must be positive, got {}",
            source, dim
        )));
    }
    if dim > MAX_DIM {
        error!(
            operation = "validation",
            field = "dim",
            value = dim,
            max_allowed = MAX_DIM,
            "dim exceeds maximum"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}dim must be at most {}, got {}. dim is the width of one vector \
             buffer, so sizing that buffer from the declared width is the \
             first allocation this index makes. That allocation is not \
             fallible: above this bound the process aborts rather than \
             raising. One vector at the ceiling is {} bytes, an order of \
             magnitude above the widest embedding any published model \
             produces.",
            source,
            MAX_DIM,
            dim,
            MAX_DIM * 4
        )));
    }
    if ef_construction == 0 {
        error!(
            operation = "validation",
            field = "ef_construction",
            value = ef_construction,
            "Invalid ef_construction"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}ef_construction must be positive, got {}",
            source, ef_construction
        )));
    }
    if ef_construction > MAX_EF_CONSTRUCTION {
        error!(
            operation = "validation",
            field = "ef_construction",
            value = ef_construction,
            max_allowed = MAX_EF_CONSTRUCTION,
            "ef_construction exceeds maximum"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}ef_construction must be at most {}, got {}. It is the width of the \
             candidate search every insertion runs, and the graph sizes two \
             candidate heaps from it, 8 bytes a slot, before each insertion visits \
             a node. That allocation is not fallible: a value of 2^40 asks for 8 TB \
             on the first add() and the process aborts rather than raising. The \
             ceiling is eight times the neighbour budget at the largest m, being \
             2 * 256, so the default's margin over that budget is available at \
             every m the index allows, and a build at the ceiling runs about \
             thirteen times longer than one at the default.",
            source, MAX_EF_CONSTRUCTION, ef_construction
        )));
    }
    if expected_size == 0 {
        error!(
            operation = "validation",
            field = "expected_size",
            value = expected_size,
            "Invalid expected_size"
        );
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}expected_size must be positive, got {}",
            source, expected_size
        )));
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
            "{}expected_size must be at most {}, got {}. The graph reserves one \
 slot per \
             declared record at creation, 8 bytes each, so this \
 declaration would ask for \
             {:.1} GB before a single record is \
 added. That allocation is not fallible: \
             above this bound the \
 process aborts rather than raising. expected_size is a \
             capacity \
 hint and not a limit, and under-declaring only costs some \
 \
             reallocation, so declare what you expect to hold.",
            source,
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
            "{}m must be at least 2, got {}. Layer assignment samples from a \
 scale of 1 / \
             ln(m), which is infinity at m 1, so every point \
 overflows the layer cap and \
             is redispatched uniformly across all \
 16 layers instead of following the \
             exponential distribution the \
 graph depends on. Measured on 3,000 records of \
             32 dimensions, \
 recall at 10 was 0.0220 at m 1 against 0.6880 at m 2 and \
             1.0000 \
 at m 16.",
            source, m
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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{}m must be less than or equal to 256, got {}",
            source, m
        )));
    }

    // Early space validation with user-friendly error
    let space_normalized = space.to_lowercase();
    match space_normalized.as_str() {
        "cosine" | "l2" | "l1" | "dot" => {
            debug!(operation = "validation", space = %space_normalized, "Distance space validated");
        }
        _ => {
            error!(operation = "validation", field = "space", value = %space, "Unsupported distance space");
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "{}Unsupported space: '{}'. Supported spaces: 'cosine', 'l2', 'l1', 'dot'",
                source, space
            )));
        }
    }
    Ok(space_normalized)
}

/// Refuse the pairs of a space and quantization this build cannot serve
///
/// A quantized graph scores every candidate against a lookup table of squared
/// L2 distances between the query's subvectors and the codebook's centroids,
/// and that table is the same table whatever the space is declared to be.
/// `DistPQ::eval` sums it on the search path and reads the codebook's symmetric
/// table of the same quantity on the construction path.
///
/// **Two spaces can be served from that sum and two cannot.** What separates
/// them is whether the sum can be turned into the declared distance using only
/// what the codes already carry.
///
/// For l2 the sum is the distance itself before the root, so the ordering is
/// the declared one and only the reported number needs converting;
/// `sqrt_adc_page` does that. For cosine the conversion needs each record's own
/// reconstruction length, which is a second sum over the same codes against a
/// table of squared centroid norms, so `PqMetric::Cosine` accumulates both and
/// returns the cosine distance from the scorer. That moves the ordering as well
/// as the number, which it has to, since dividing by a length that varies from
/// record to record is not a monotone map of the sum.
///
/// The two spaces below are refused, each for its own reason.
///
/// **The inner product.** The squared L2 between a query and a stored vector is
/// the query's squared length plus the stored vector's squared length less
/// twice their inner product, and a dot index does not normalise, so the middle
/// term varies from record to record and the ordering it produces is not the
/// ordering by inner product. A short vector pointing the right way would
/// outrank a long one pointing further the same way.
///
/// The norm table `PqMetric::Cosine` reads makes that middle term recoverable,
/// and exactly: over a real trained codebook,
/// `(norm(q)^2 + norm(c)^2 - adc) / 2` task-measured within 9e-5 of `|q||c|`
/// against the directly computed inner product to the reconstruction, on
/// vectors whose lengths span three orders of magnitude. The scorer is
/// therefore not the obstacle. The codebook is. k-means places its centroids
/// to minimise squared L2, and ranking by the recovered inner product was
/// task-measured by brute force over the codebook's own reconstructions,
/// before any graph loss, on sift-128, glove-100 and dbpedia-openai-1536 at
/// the shipped defaults, both as the corpora come and with stored lengths
/// rescaled to controlled spreads of one to three orders of magnitude. Recall
/// at 10 against an exact inner product ranking never exceeded 0.37 and sat
/// 0.35 to 0.82 below an unquantized dot index on the same data. Where
/// lengths are near uniform the recovered product ranked below the plain ADC
/// sum, 0.18 against 0.40 on sift-128, because the true lengths carry no
/// signal there while the recovered term faithfully adds the reconstruction's
/// own norm error as ranking noise. Where lengths spread the length signal is
/// real and recall still plateaued near 0.3, because the squared L2 objective
/// spends its centroids on the longest vectors and resolves no vector's
/// length finely enough to rank by. The two regimes fail in opposite
/// directions, so the pair is refused on the codebook rather than on a
/// missing scorer. Serving it would take a codebook fitted to an inner
/// product objective, which is a training change with its own measurement.
///
/// **Manhattan distance.** L1 and squared L2 do not induce the same order, and
/// the counterexample needs no approximation to reach. Against the query
/// `[0, 0]`, the point `[2, 0]` is at L1 2.0 and squared L2 4.0 while
/// `[1.1, 1.1]` is at L1 2.2 and squared L2 2.42, so the two rank the pair in
/// opposite orders. `the_l1_counterexample_is_ordered_by_squared_l2` in
/// `distance.rs` holds that arithmetic against the live scorer. A quantized l1
/// index therefore ranked by a quantity it never declared and reported a score
/// on it as well, and an l1 and an l2 index over one corpus returned the same
/// page as each other.
///
/// Serving it properly means an L1 codebook, and that is more than an L1 table.
/// Lloyd's algorithm assigns by squared L2 and takes a mean, which is the
/// minimiser of squared L2 and not of L1, so an L1 quantizer needs k-medians
/// rather than a second distance in the same loop. That is a training change
/// with its own measurement. Until it exists the pair is refused rather than
/// served wrongly.
///
/// Refusing costs a configuration that measures well. Measured on
/// SIFT-128 at 25,000 records, `quantized_with_raw` with the default rerank
/// reached recall at 10 of 0.9885 against an exact L1 ranking, because rerank
/// rescores against the raw vectors and hides the graph's ordering. The pair is
/// still refused, because the graph underneath it is ordered by the wrong
/// quantity and `rerank=0` exposes that at any time.
///
/// Called at `create()` and again at load, because a hand edited `config.json`
/// reaches the same constructors.
pub(crate) fn validate_space_supports_quantization(space: &str, source: &str) -> PyResult<()> {
    // The remedy differs by space, so each pair carries its own middle clause.
    // The opening sentence and the closing offer are shared, because the reason
    // is one reason and the way out of it is the same way out.
    let reason = match space {
        "dot" => "the codebook those tables are computed from is fitted by squared L2 and cannot rank by the inner product. Measured by brute force over its own reconstructions at the shipped defaults, recall at 10 against an exact inner product ranking never exceeded 0.37 and sat at least 0.35 below an unquantized dot index on every corpus and every stored length spread measured, so the index would return the wrong records. Use space='cosine' with normalised vectors where only direction should count, which quantizes correctly and ranks identically to an inner product on normalised input",
        "l1" => "Manhattan distance is not one of the two those tables can be turned into, so the index would return the wrong records and report a score on a quantity it never declared. Use space='l2', which the same codebook quantizes correctly, if squared distance suits your data",
        _ => return Ok(()),
    };
    error!(
        operation = "validation",
        field = "space",
        value = space,
        "Quantization is not available for this distance space"
    );
    // At load the caller is a directory rather than a keyword argument, so the
    // offer above is advice for the next index rather than for this one. The
    // extra sentence says what became of the directory in hand. `source` is
    // empty at `create()` and carries the path at load, which is the only thing
    // that tells the two doors apart, and one check still decides both.
    let recovery = if source.is_empty() {
        ""
    } else {
        " A directory saved with this pair by an earlier release cannot be opened by this build and has to be rebuilt from the vectors it was given."
    };
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "{}space='{}' cannot be quantized. A quantized graph scores every candidate from tables of squared L2 distances to the codebook, and {}, or drop quantization_config.{}",
        source, space, reason, recovery
    )))
}

/// Warn where `ef_construction` switches the neighbour selection heuristic off.
///
/// **The message is the one `VectorDatabase.create()` raises, word for word.**
/// The Python factory checks the pair a caller passes to `create()` and this
/// checks the pair a caller passes to `rebuild()`, so the same misconfiguration
/// reads the same either way.
/// `the_two_warnings_are_the_same_sentence` in `tests/test_index_lifecycle.py`
/// holds the two texts equal, which is what stops one drifting from the other.
///
/// The reasoning behind the threshold is in `_warn_if_selection_disabled` in
/// `vector_database.py` and is not repeated here. In short, `select_neighbours`
/// keeps every candidate rather than pruning once the candidate list is no
/// longer than the neighbour budget, that budget is `2 * m` at layer zero, and
/// the candidate list is `ef_construction` long, so the flip lands exactly on
/// `ef_construction <= 2 * m` and the warning carries no slack.
///
/// A pair the validation is going to reject returns without warning, so an
/// invalid `m` produces its real validation error and nothing else.
pub(crate) fn warn_if_selection_disabled(
    py: Python<'_>,
    m: usize,
    ef_construction: usize,
) -> PyResult<()> {
    if !(2..=256).contains(&m) || ef_construction < 1 || ef_construction > 2 * m {
        return Ok(());
    }
    let budget = 2 * m;
    // The largest m that leaves the heuristic running at this ef_construction.
    // Below the floor of 2 there is no such m, so the message offers the one
    // remedy that exists.
    let largest_m = (ef_construction - 1) / 2;
    let remedy = if largest_m >= 2 {
        format!(
            "Raise ef_construction above {}, or lower m to {} or below",
            budget, largest_m
        )
    } else {
        format!("Raise ef_construction above {}", budget)
    };
    let message = format!(
        "ef_construction={} is not greater than 2*m={}, so the neighbour selection \
         heuristic does not run. Layer zero insertion keeps every candidate the \
         construction search returns, in distance order, and prunes none of them. \
         {}, to run it.",
        ef_construction, budget, remedy
    );
    let text = std::ffi::CString::new(message)
        .expect("the message is built here from integers and carries no interior nul");
    PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyUserWarning>(),
        &text,
        1,
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
        indexed_fields: Vec<String>,
    ) -> PyResult<Self> {
        let start_time = Instant::now();

        // Validation of parameters. The rules live in
        // `validate_index_parameters`, which the loader calls on the same
        // five values it reads out of `config.json`.
        let space_normalized =
            validate_index_parameters(dim, &space, m, ef_construction, expected_size, "")?;
        // The declaration is checked here for the same reason, and the loader
        // checks what `config.json` carried against the same rules.
        validate_indexed_fields(&indexed_fields, "")?;

        // Extract quantization configuration
        let (quantization_params, pq_instance) = if let Some(config) = quantization_config {
            // Before any of the config is read, so the message names the pair
            // rather than whichever PQ field happens to be checked first.
            validate_space_supports_quantization(&space_normalized, "")?;
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

        trace!(
            operation = "hnsw_config",
            max_layer = MAX_LAYER,
            reason = "hnsw-rs compatibility",
            "Using fixed max_layer"
        );

        // Create initial raw HNSW index (will be rebuilt as PQ after training)
        let hnsw = VectorGraph::new_raw(
            &space_normalized,
            dim,
            m,
            expected_size,
            MAX_LAYER,
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
            indexed_fields = indexed_fields.len(),
            duration_ms = duration_ms,
            "HNSW index created successfully"
        );

        // Initialize all fields with proper thread-safe wrappers
        Ok(HNSWIndex {
            dim,
            space: space_normalized,
            m: AtomicUsize::new(m),
            ef_construction: AtomicUsize::new(ef_construction),
            expected_size: AtomicUsize::new(expected_size),
            quantization_config: quantization_params,
            pq: pq_instance,
            pq_codes: RwLockAt::new(HashMap::new()),
            rerank_calibration: RwLockAt::new(None),
            metadata: MutexAt::new(HashMap::new()),
            vector_metadata: RwLockAt::new(HashMap::new()),
            columns: RwLockAt::new(ColumnStore::new(indexed_fields, expected_size)),
            undeclared_filter_warned: AtomicBool::new(false),
            id_map: RwLockAt::new(HashMap::new()),
            rev_map: RwLockAt::new(HashMap::new()),
            id_counter: MutexAt::new(0),
            generated_ids: MutexAt::new(0),
            vector_count: MutexAt::new(0),
            hnsw: RwLockAt::new(hnsw),
            writers: MutexAt::new(()),
            training_ids: RwLockAt::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            training_completed_at: RwLockAt::new(None),
            created_at: RwLockAt::new(Utc::now().to_rfc3339()),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        })
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
        indexed_fields: Vec<String>,
    ) -> Self {
        let space_normalized = space.to_lowercase();
        let hnsw = VectorGraph::new_raw(
            &space_normalized,
            dim,
            m,
            expected_size,
            MAX_LAYER,
            ef_construction,
        );

        HNSWIndex {
            dim,
            space: space_normalized,
            m: AtomicUsize::new(m),
            ef_construction: AtomicUsize::new(ef_construction),
            expected_size: AtomicUsize::new(expected_size),
            quantization_config: None,
            pq: None,
            pq_codes: RwLockAt::new(HashMap::new()),
            rerank_calibration: RwLockAt::new(None),
            metadata: MutexAt::new(HashMap::new()),
            vector_metadata: RwLockAt::new(HashMap::new()),
            columns: RwLockAt::new(ColumnStore::new(indexed_fields, expected_size)),
            undeclared_filter_warned: AtomicBool::new(false),
            id_map: RwLockAt::new(HashMap::new()),
            rev_map: RwLockAt::new(HashMap::new()),
            id_counter: MutexAt::new(0),
            generated_ids: MutexAt::new(0),
            vector_count: MutexAt::new(0),
            hnsw: RwLockAt::new(hnsw),
            writers: MutexAt::new(()),
            training_ids: RwLockAt::new(Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            // Both are overwritten by the loader from the saved directory, and
            // this is the only path that reaches here. See `set_created_at` and
            // `set_training_completed_at`.
            training_completed_at: RwLockAt::new(None),
            created_at: RwLockAt::new(chrono::Utc::now().to_rfc3339()),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        }
    }
}
