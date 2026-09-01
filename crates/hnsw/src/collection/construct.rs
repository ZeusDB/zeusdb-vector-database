//! Building a collection, and the bounds a declaration has to satisfy.
//!
//! [`Declaration::validate`] enforces every rule that governs a valid index
//! and [`Collection::build`] builds from what it validated, which is the only
//! construction path from Python and is what makes the Python factory and the
//! Rust constructor agree. `new_empty` is the loader's constructor and
//! validates nothing, because its configuration comes from a directory this
//! crate wrote and the loader holds it to the same rules on the way in.

use super::{
    Collection, DenseIndex, DenseSpace, LiveRecords, NamedSpace, QuantizationConfig, Space,
    SparseSpace, StorageMode, TextLayer, DEFAULT_SPACE, MAX_LAYER,
};
use crate::locks::{order, MutexAt, RwLockAt};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, trace};
use zeusdb_vector_core::{validate_indexed_fields, ColumnStore, Error, SpaceName, VectorGraph, PQ};
use zeusdb_vector_sparse::{PostingsIndex, SparseConfig};
use zeusdb_vector_text::{TermDictionary, Tokenizer};

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::construct";
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
/// refused. The ceiling sits far above any beam width a graph search
/// benefits from, so it bounds a mistake rather than a workload.
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
) -> Result<String, Error> {
    if dim == 0 {
        error!(target: LOG_TARGET, operation = "validation",
            field = "dim",
            value = dim,
            "Invalid dimension"
        );
        return Err(Error::DimZero {
            source: source.to_string(),
            dim,
        });
    }
    if dim > MAX_DIM {
        error!(target: LOG_TARGET, operation = "validation",
            field = "dim",
            value = dim,
            max_allowed = MAX_DIM,
            "dim exceeds maximum"
        );
        return Err(Error::DimTooLarge {
            source: source.to_string(),
            dim,
            max: MAX_DIM,
        });
    }
    if ef_construction == 0 {
        error!(target: LOG_TARGET, operation = "validation",
            field = "ef_construction",
            value = ef_construction,
            "Invalid ef_construction"
        );
        return Err(Error::EfConstructionZero {
            source: source.to_string(),
            value: ef_construction,
        });
    }
    if ef_construction > MAX_EF_CONSTRUCTION {
        error!(target: LOG_TARGET, operation = "validation",
            field = "ef_construction",
            value = ef_construction,
            max_allowed = MAX_EF_CONSTRUCTION,
            "ef_construction exceeds maximum"
        );
        return Err(Error::EfConstructionTooLarge {
            source: source.to_string(),
            value: ef_construction,
            max: MAX_EF_CONSTRUCTION,
        });
    }
    if expected_size == 0 {
        error!(target: LOG_TARGET, operation = "validation",
            field = "expected_size",
            value = expected_size,
            "Invalid expected_size"
        );
        return Err(Error::ExpectedSizeZero {
            source: source.to_string(),
            value: expected_size,
        });
    }
    if expected_size > MAX_EXPECTED_SIZE {
        error!(target: LOG_TARGET, operation = "validation",
            field = "expected_size",
            value = expected_size,
            max_allowed = MAX_EXPECTED_SIZE,
            "expected_size exceeds maximum"
        );
        return Err(Error::ExpectedSizeTooLarge {
            source: source.to_string(),
            value: expected_size,
            max: MAX_EXPECTED_SIZE,
        });
    }
    if m < 2 {
        error!(target: LOG_TARGET, operation = "validation",
            field = "m",
            value = m,
            min_allowed = 2,
            "m below minimum"
        );
        return Err(Error::MBelowMinimum {
            source: source.to_string(),
            m,
        });
    }
    if m > 256 {
        error!(target: LOG_TARGET, operation = "validation",
            field = "m",
            value = m,
            max_allowed = 256,
            "m exceeds maximum"
        );
        return Err(Error::MTooLarge {
            source: source.to_string(),
            m,
        });
    }

    // Early space validation with user-friendly error
    let space_normalized = space.to_lowercase();
    match space_normalized.as_str() {
        "cosine" | "l2" | "l1" | "dot" => {
            debug!(target: LOG_TARGET, operation = "validation", space = %space_normalized, "Distance space validated");
        }
        _ => {
            error!(target: LOG_TARGET, operation = "validation", field = "space", value = %space, "Unsupported distance space");
            return Err(Error::UnsupportedSpace {
                source: source.to_string(),
                space: space.to_string(),
            });
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
/// page as each other, 1,000 of 1,000 pages identical on SIFT-128 at 25,000
/// and at 100,000 records.
///
/// The tables are not the obstacle. L1 is separable across subvectors exactly
/// as squared L2 is, so an L1 query table and an L1 symmetric table are the
/// same shape as the ones this build fills. The codebook is not the obstacle
/// either. Lloyd's algorithm assigns by squared L2 and takes a mean, which
/// minimises squared error, and the k-medians that minimises absolute error,
/// assigning by L1 and updating to the per dimension median under the same
/// seeded initialisation, was task-measured beside it. By brute force over
/// each codebook's own reconstructions, on sift-128, glove-100 and
/// dbpedia-openai-1536 at 100,000 records, at the default `subvectors` and at
/// twice it, ranking by L1 to a k-medians reconstruction reached recall at 10
/// against an exact L1 ranking of 0.17 to 0.65, which is 0.03 to 0.08 below
/// what the shipped quantized l2 mode reaches against exact L2 on the same
/// records and code length, and at the default `subvectors` it ranked below
/// the squared L2 ordering it would replace on every corpus, by 0.007, 0.014
/// and 0.025. The codebook objective moved recall by under 0.01 against L1
/// tables over the shipped k-means codebook at every cell, although the two
/// codebooks differ, the k-medians centroids sitting 0.5 to 0.8 of a centroid
/// spacing from their nearest k-means centroid and lowering L1 distortion by
/// 3 to 7 percent. An L1 table over a reconstruction estimates the L1
/// distance to the record with a bias the squared L2 table does not carry,
/// and in the grid the L1 ordering won only at subvectors of 5 and 8 values
/// and lost at 10, 16 and 32. So the pair stays refused on the measured
/// ordering rather than on a missing table.
///
/// Refusing costs a configuration that measures well. Measured on SIFT-128 at
/// 25,000 records, `quantized_with_raw` with the default rerank reached recall
/// at 10 of 0.9924 against an exact L1 ranking over 1,000 held-out queries,
/// because rerank rescores against the raw vectors and hides the graph's
/// ordering. The pair is still refused, because the graph underneath it is
/// ordered by the wrong quantity and `rerank=0` exposes that at any time, and
/// because holding that recall on glove-100 at 100,000 records took a
/// calibrated fetch of 12,336 candidates, 12 percent of the corpus.
///
/// Called at `create()` and again at load, because a hand edited `config.json`
/// reaches the same constructors.
pub(crate) fn validate_space_supports_quantization(space: &str, source: &str) -> Result<(), Error> {
    // The remedy differs by space, so each pair carries its own middle clause.
    // The opening sentence and the closing offer are shared, because the reason
    // is one reason and the way out of it is the same way out.
    let reason = match space {
        "dot" => "the codebook those tables are computed from is fitted by squared L2 and cannot rank by the inner product. Measured by brute force over its own reconstructions at the shipped defaults, recall at 10 against an exact inner product ranking never exceeded 0.37 and sat at least 0.35 below an unquantized dot index on every corpus and every stored length spread measured, so the index would return the wrong records. Use space='cosine' with normalised vectors where only direction should count, which quantizes correctly and ranks identically to an inner product on normalised input",
        "l1" => "Manhattan distance does not order the same way. L1 tables are buildable, since L1 is separable across subvectors, and were measured over that codebook and over a k-medians codebook fitted to absolute error. By brute force over their own reconstructions at 100,000 records on three corpora, recall at 10 against an exact L1 ranking sat 0.03 to 0.08 below what quantized l2 reaches against exact L2 on the same records and code length, and at the default subvectors below the squared L2 ordering it would replace, so the index would return the wrong records. Use space='l2', which the same codebook quantizes correctly, if squared distance suits your data",
        _ => return Ok(()),
    };
    error!(target: LOG_TARGET, operation = "validation",
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
    Err(Error::SpaceCannotBeQuantized {
        source: source.to_string(),
        space: space.to_string(),
        reason,
        recovery,
    })
}

/// A validated declaration, being the five values `create()` takes held to
/// every rule a valid index has to satisfy, with the space normalised.
///
/// Built by [`Declaration::validate`] and consumed by [`Collection::build`].
/// The two are separate so that the binding can parse a quantization mapping
/// between them, in the order the rules are applied: the five values first,
/// then whether the space can be quantized at all, then the mapping's keys one
/// at a time, then [`Declaration::quantization`] holding the parsed values to
/// the product quantizer's own rules. Each rule runs exactly once, and the
/// message a caller reads is the one for the first rule their declaration
/// broke, which is the order `build` applied them in when it read the mapping
/// itself.
#[derive(Debug, Clone)]
pub struct Declaration {
    dim: usize,
    space: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    indexed_fields: Vec<String>,
    /// A sparse space beside the dense one, by name, with a tokenizer
    /// where it takes text. Declared through [`Declaration::with_sparse`]
    /// or [`Declaration::with_text`], which nothing in the binding calls.
    sparse: Option<SparseDeclaration>,
}

/// A sparse space as declared.
#[derive(Debug, Clone)]
pub(crate) struct SparseDeclaration {
    pub(crate) name: SpaceName,
    pub(crate) config: SparseConfig,
    pub(crate) tokenizer: Option<Arc<dyn Tokenizer>>,
}

impl Declaration {
    /// Hold the five values and the field declaration to the rules `create()`
    /// applies. The rules live in `validate_index_parameters`, which the loader
    /// calls on the same five values it reads out of `config.json`, and in
    /// `validate_indexed_fields`, which it calls on the declaration.
    pub fn validate(
        dim: usize,
        space: &str,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
        indexed_fields: Vec<String>,
    ) -> Result<Self, Error> {
        let space = validate_index_parameters(dim, space, m, ef_construction, expected_size, "")?;
        validate_indexed_fields(&indexed_fields, "")?;
        Ok(Declaration {
            dim,
            space,
            m,
            ef_construction,
            expected_size,
            indexed_fields,
            sparse: None,
        })
    }

    /// Declare a sparse space beside the dense one.
    ///
    /// The name is refused where it is empty or is the dense space's, and a
    /// second sparse space is refused, since a collection holds one dense
    /// space and at most one sparse space.
    pub fn with_sparse(self, name: &str, config: SparseConfig) -> Result<Self, Error> {
        self.declare_sparse(name, config, None)
    }

    /// Declare a sparse space with a text layer beside the dense one.
    ///
    /// The same rules as [`Declaration::with_sparse`], and the space takes
    /// text through `Collection::vectorize_texts` and `search_text` as well
    /// as term ids. The tokenizer is the caller's to keep: an index that
    /// used one the engine cannot write down must be handed the same
    /// implementation when it is opened.
    pub fn with_text(
        self,
        name: &str,
        config: SparseConfig,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Result<Self, Error> {
        self.declare_sparse(name, config, Some(tokenizer))
    }

    fn declare_sparse(
        mut self,
        name: &str,
        config: SparseConfig,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> Result<Self, Error> {
        let name = SpaceName::new(name)?;
        if name.as_str() == DEFAULT_SPACE {
            return Err(Error::SpaceDeclaredTwice {
                name: name.as_str().to_string(),
            });
        }
        if self.sparse.is_some() {
            return Err(Error::SpacesTooMany { max: 2 });
        }
        config.validate()?;
        self.sparse = Some(SparseDeclaration {
            name,
            config,
            tokenizer,
        });
        Ok(self)
    }

    /// Refuse the space if this build cannot quantize it. Checked before any
    /// of a quantization mapping is read, so the message names the pair rather
    /// than whichever PQ field happens to be checked first. See
    /// `validate_space_supports_quantization`.
    pub fn quantizable(&self) -> Result<(), Error> {
        validate_space_supports_quantization(&self.space, "")
    }

    /// The product quantizer's parameters, held to its rules against this
    /// declaration's width.
    pub fn quantization(
        &self,
        subvectors: usize,
        bits: usize,
        training_size: usize,
        max_training_vectors: Option<usize>,
        storage_mode: StorageMode,
    ) -> Result<QuantizationConfig, Error> {
        let dim = self.dim;

        // Validate PQ parameters
        if subvectors == 0 {
            error!(
                target: LOG_TARGET,
                operation = "validation",
                field = "subvectors",
                value = subvectors,
                "Subvectors must be positive"
            );
            return Err(Error::SubvectorsZero);
        }

        if subvectors > dim {
            error!(
                target: LOG_TARGET,
                operation = "validation",
                field = "subvectors",
                dim = dim,
                subvectors = subvectors,
                "Subvectors exceed dimension"
            );
            return Err(Error::SubvectorsExceedDim { subvectors, dim });
        }

        if !dim.is_multiple_of(subvectors) {
            error!(
                target: LOG_TARGET,
                operation = "validation",
                field = "subvectors",
                dim = dim,
                subvectors = subvectors,
                "Subvectors must divide dimension evenly"
            );
            return Err(Error::SubvectorsDoNotDivideDim { subvectors, dim });
        }

        if !(1..=8).contains(&bits) {
            error!(
                target: LOG_TARGET,
                operation = "validation",
                field = "bits",
                value = bits,
                min = 1,
                max = 8,
                "Bits out of range"
            );
            return Err(Error::BitsOutOfRange { bits });
        }

        if training_size < 1000 {
            error!(
                target: LOG_TARGET,
                operation = "validation",
                field = "training_size",
                value = training_size,
                min = 1000,
                "Training size too small"
            );
            return Err(Error::TrainingSizeTooSmall { training_size });
        }

        // A max below the threshold produces an index that reaches its
        // training threshold and then fails training on every record from
        // then on, because the cap is already exceeded by the time the
        // trigger fires. Enforced here so it holds on every construction
        // path rather than only the Python factory.
        if let Some(max_training) = max_training_vectors {
            if max_training < training_size {
                error!(
                    target: LOG_TARGET,
                    operation = "validation",
                    field = "max_training_vectors",
                    value = max_training,
                    training_size = training_size,
                    "max_training_vectors below training_size"
                );
                return Err(Error::MaxTrainingBelowTrainingSize {
                    max_training,
                    training_size,
                });
            }
        }

        Ok(QuantizationConfig {
            subvectors,
            bits,
            training_size,
            max_training_vectors,
            storage_mode,
        })
    }
}

impl Collection {
    /// Build a collection from a validated declaration.
    ///
    /// Every rule has already run, in `Declaration::validate` and, for a
    /// quantized space, in `Declaration::quantization`, so nothing here can
    /// fail. What happens here is the quantizer instance, the empty raw graph
    /// and the assembly of the two structs.
    pub fn build(declaration: Declaration, quantization: Option<QuantizationConfig>) -> Self {
        let start_time = Instant::now();
        let Declaration {
            dim,
            space: space_normalized,
            m,
            ef_construction,
            expected_size,
            indexed_fields,
            sparse,
        } = declaration;

        let pq_instance = quantization.as_ref().map(|config| {
            debug!(
                target: LOG_TARGET,
                operation = "pq_configuration",
                subvectors = config.subvectors,
                bits = config.bits,
                training_size = config.training_size,
                storage_mode = %config.storage_mode.to_string(),
                sub_dim = dim / config.subvectors,
                num_centroids = 1 << config.bits,
                "Product Quantization configured"
            );

            // Create PQ instance
            Arc::new(PQ::new(
                dim,
                config.subvectors,
                config.bits,
                config.training_size,
                config.max_training_vectors,
            ))
        });

        trace!(
            target: LOG_TARGET,
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
            target: LOG_TARGET,
            operation = "index_creation_complete",
            dim = dim,
            space = %space_normalized,
            m = m,
            ef_construction = ef_construction,
            expected_size = expected_size,
            has_quantization = quantization.is_some(),
            indexed_fields = indexed_fields.len(),
            duration_ms = duration_ms,
            "HNSW index created successfully"
        );

        Collection::assemble(
            dim,
            space_normalized,
            m,
            ef_construction,
            expected_size,
            indexed_fields,
            quantization,
            pq_instance,
            hnsw,
            sparse,
        )
    }

    // ============================================================================
    // PERSISTENCE Minimal Empty Constructor and SETTERS
    // ============================================================================
    /// The constructor the loader uses, being an empty collection under the
    /// configuration `config.json` recorded. The loader validates every
    /// value before it reaches here, through the same rules `validate`
    /// applies, and builds the sparse declaration from the space
    /// `config.json` recorded and the tokenizer handed to `load`.
    pub(crate) fn new_empty(
        dim: usize,
        space: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
        indexed_fields: Vec<String>,
        sparse: Option<SparseDeclaration>,
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

        // The quantizer, the calibration and both timestamps are written by
        // the loader from the saved directory, and this is the only path that
        // reaches here. See `set_pq`, `set_created_at` and
        // `set_training_completed_at`.
        Collection::assemble(
            dim,
            space_normalized,
            m,
            ef_construction,
            expected_size,
            indexed_fields,
            None,
            None,
            hnsw,
            sparse,
        )
    }

    /// The one place the structs are put together, so the two constructors
    /// cannot disagree about a field's starting value.
    ///
    /// The dense space is first, under [`DEFAULT_SPACE`], and takes the
    /// first space's four ranks. A sparse space, where one was declared,
    /// is second and takes the second space's index rank. Every lock is
    /// given its rank here, beside the field it guards.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        dim: usize,
        metric: String,
        m: usize,
        ef_construction: usize,
        expected_size: usize,
        indexed_fields: Vec<String>,
        quantization_config: Option<QuantizationConfig>,
        pq: Option<Arc<PQ>>,
        hnsw: VectorGraph,
        sparse: Option<SparseDeclaration>,
    ) -> Self {
        let keeps_raw = quantization_config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        let index = DenseIndex::new(hnsw, &metric, dim, pq.clone(), keeps_raw);
        let mut spaces = vec![NamedSpace {
            name: SpaceName::new(DEFAULT_SPACE).expect("the default space name is not empty"),
            space: Space::Dense(DenseSpace {
                dim,
                metric,
                m: AtomicUsize::new(m),
                ef_construction: AtomicUsize::new(ef_construction),
                quantization_config,
                pq,
                pq_codes: RwLockAt::new(order::space_codes(0), HashMap::new()),
                rerank_calibration: RwLockAt::new(order::space_calibration(0), None),
                index: RwLockAt::new(order::space_index(0), index),
                training_completed_at: RwLockAt::new(order::space_trained_at(0), None),
            }),
        }];
        if let Some(SparseDeclaration {
            name,
            config,
            tokenizer,
        }) = sparse
        {
            let position = spaces.len();
            spaces.push(NamedSpace {
                name,
                space: Space::Sparse(SparseSpace {
                    index: RwLockAt::new(
                        order::space_index(position),
                        PostingsIndex::new(config.clone()),
                    ),
                    config,
                    text: tokenizer.map(|tokenizer| TextLayer {
                        tokenizer,
                        dictionary: RwLockAt::new(
                            order::space_codes(position),
                            TermDictionary::new(),
                        ),
                    }),
                }),
            });
        }
        // Initialize all fields with proper thread-safe wrappers
        Collection {
            spaces,
            expected_size: AtomicUsize::new(expected_size),
            metadata: MutexAt::new(order::METADATA, HashMap::new()),
            vector_metadata: RwLockAt::new(order::VECTOR_METADATA, HashMap::new()),
            columns: RwLockAt::new(
                order::COLUMNS,
                ColumnStore::new(indexed_fields, expected_size),
            ),
            undeclared_filter_warned: AtomicBool::new(false),
            id_map: RwLockAt::new(order::ID_MAP, HashMap::new()),
            rev_map: RwLockAt::new(order::REV_MAP, LiveRecords::new()),
            id_counter: MutexAt::new(order::ID_COUNTER, 0),
            generated_ids: MutexAt::new(order::GENERATED_IDS, 0),
            vector_count: MutexAt::new(order::VECTOR_COUNT, 0),
            writers: MutexAt::new(order::WRITERS, ()),
            training_ids: RwLockAt::new(order::TRAINING_IDS, Vec::new()),
            training_threshold_reached: AtomicBool::new(false),
            created_at: RwLockAt::new(order::CREATED_AT, Utc::now().to_rfc3339()),
            rebuilding_from_persistence: AtomicBool::new(false),
            overgrowth_warned: AtomicBool::new(false),
        }
    }
}
