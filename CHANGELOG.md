# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2026-08-10

This release repairs the index graph. Search results change on upgrade for every index type, in each case for the better, so any test that pins exact result sets against a 0.4.x index needs its expectations refreshed.

### Breaking

- A search on a `quantized_with_raw` index reranks its candidates against the stored raw vectors by default, and the returned score is the raw distance rather than the quantized approximation. Scores and result order both change. Pass `rerank=0` to `search()` to restore the previous scoring.
- Under `quantized_only`, raw vectors are released once training completes. `return_vector=True` and `get_records()` then return a reconstruction from the stored codes rather than the value supplied, for every record including the training records. Saving a trained `quantized_only` index writes no `vectors.bin`, and loading a quantized directory written by an earlier release drops any retained raw vector for a record that carries codes. Use `quantized_with_raw` where the exact vector has to come back.
- The default `subvectors` is derived from the dimension as `dim / 32`, clamped to 8 through 192 and snapped to a divisor of `dim`. It was a fixed 8. A quantized index created without an explicit `subvectors` gets different codes, memory and recall. Pass `subvectors=8` to keep the previous configuration. A saved index keeps the configuration it was built with.
- The default `m` scales with `expected_size`: 16 up to 25,000 and 32 above. An index created above 25,000 records on defaults uses more memory and returns better results. Pass `m=16` to keep the previous density. An explicit `m` is unaffected and a saved index keeps the `m` it was built with.
- An unknown filter operator raises `ValueError`. It previously matched nothing and returned an empty result set, so a misspelled operator now surfaces instead of silently filtering everything out.
- Numbers in filter conditions compare by magnitude, so an integer and an equal float match under `eq`, `ne`, `in` and direct equality. A filter that relied on `10` and `10.0` being distinct values returns different results.
- An array filter condition such as `{"tags": ["ai", "science"]}` matches by exact element-by-element equality, in order. It previously matched nothing at all.
- Non-finite values (NaN or infinity) are rejected everywhere: `add()` reports them per record through `AddResult`, naming the record and component, single and batch `search()` raise naming the entry and component, and a load that finds one in a saved index fails.
- `m` has a floor of 2 and `expected_size` a ceiling of 100,000,000. Values outside these bounds raise instead of building a degenerate or unallocatable graph.
- `HNSWIndex` can no longer be constructed directly and `HNSWIndex(...)` raises `TypeError`. An index comes from `VectorDatabase.create()` or `VectorDatabase.load()`. The class stays importable for `isinstance` checks and annotations.
- `get_next_id()`, `benchmark_raw_concurrent_performance()`, `get_code_version()`, `get_version_number()`, `count_stored_records()` and `load_index` are removed from the Python surface. Use `VectorDatabase.load()` in place of `load_index` and `zeusdb_vector_database.__version__` in place of the version getters.
- `MemoryInfo`, `auto_configure_logging` and `JSONFormatter` are renamed `_MemoryInfo`, `_auto_configure_logging` and `_JSONFormatter`. `VectorDatabase._index_constructors` is replaced by `_index_types`, a name to description map that hands out no constructor.
- `get_stats()` no longer reports `memory_savings`, and `get_performance_info()` no longer reports `quantization_memory_savings`, `insertion_speedup_expected`, `insertion_bottleneck` or `limitation`, and its `benefits` list no longer claims `parallel_insert`. Read the new memory keys listed under Added instead.
- `ZEUSDB_LOG_FILE` writes exactly the file named. `ZEUSDB_LOG_FILE=app.log` used to write `app.log.YYYY-MM-DD` and leave `app.log` empty. Set `ZEUSDB_LOG_ROTATION=daily` to keep the dated files.
- Wheels are no longer published for macOS x86_64, 32-bit x86, armv7, s390x or ppc64le. The wheel targets are Linux x86_64 and aarch64 (glibc and musl), Windows x64 and macOS arm64. Any other platform installs from the sdist, which needs a Rust toolchain.

### Fixed

- Quantized search returned meaningless results in every release that offered it. The graph of a quantized index was built on distances that all evaluated as infinite, so its structure carried no information about the data. Measured at 10,000 records of dimension 768 at `top_k` 10, recall against exact search was 0.0035 where an unquantized index reached 0.9995, and a large result request returned 34 results out of 1,000 asked for. The graph is now built on real distances between codes and reranked against raw vectors where they are stored. Upgrading fixes this. A quantized directory saved by an earlier release is rebuilt on load with the corrected construction, so loading it once and saving it again persists the repaired graph.
- Between 1 and 2 percent of records were unreachable by any query, including a search for the record's own vector, measured at 1,000 and 10,000 points. Two graph construction defects caused this, one assigning reverse links to the wrong layers and one evicting a point's last inbound link. Both are fixed. Newly built graphs no longer strand records, and a directory saved by an earlier release gets the corrected construction when its graph is rebuilt on load.
- A deleted or overwritten record kept its node in the graph, where it consumed result slots and degraded search results as churn accumulated. Removed records are now excluded from traversal, and `compact()` rebuilds the graph to reclaim the stranded nodes when their memory matters.
- Loading a `quantized_only` directory dropped every record added after training, and restored the codebook as all zeroes, so quantized distances after a reload were meaningless. Both are fixed. A directory saved once by an earlier release recovers in full when loaded under this release. A directory that an earlier release loaded and then saved again lost those records permanently, and no upgrade can recover them.
- A quantized index came back from `load()` in a degraded state that kept full-width vectors in memory. It now loads back as quantized and the memory saving survives the reload.
- Creating an index reserved graph capacity for hundreds of neighbours per expected record. It now reserves one slot per record, and memory at creation drops accordingly.
- Index builds are reproducible. The same records added in the same order produce the same graph, and save followed by load returns identical results to the index that was saved.
- Metadata added through `add_metadata()` survives a save and load.
- The documented `ZEUSDB_DISABLE_AUTO_LOGGING` variable now reaches the Rust layer, which previously read only the undocumented `ZEUSDB_DISABLE_AUTOLOG`. Both layers accept the old name as a deprecated alias and require `true`, `1` or `yes`.
- Loading an index whose graph rebuild refuses a record fails with the rejected records named, instead of returning an index that reports a vector count no query can reach.

### Added

- `rerank` argument on `search()`. An integer sets the over-fetch factor for the raw-vector rescoring pass on `quantized_with_raw`, and `0` turns rescoring off. When unset, the fetch comes from a calibration measured on the index's own data at training completion, stored in `quantization.json` and scaled to the live record count and the requested page.
- `compact()` rebuilds the graph in memory, reclaiming the nodes stranded by `remove_point()` and `add(overwrite=True)`. It returns the number of nodes reclaimed and is never automatic.
- `HNSWIndex`, `AddResult`, `init_logging`, `init_file_logging` and `is_logging_initialized` are exported from `zeusdb_vector_database`, and `__all__` names them explicitly. The documented programmatic logging recipe previously raised `AttributeError` at package level.
- Memory reporting in `get_stats()`: `graph_memory_mb`, `raw_vectors_memory_mb` and `total_memory_mb` on every index, plus `codebook_memory_mb`, `sdc_table_memory_mb`, `raw_vectors_retained` and the rerank calibration figures on a quantized one. `get_performance_info()` reports the measured recall loss for each quantized storage mode and an `insertion_path` key reading `sequential`.
- `create()` warns when a quantization configuration cannot repay its fixed memory cost at the declared `expected_size`, and when `expected_size` is below `training_size`, since such an index never trains. An index warns when its record count outgrows the declared `expected_size`.
- `ZEUSDB_LOG_ROTATION` accepts `daily` or `never`, default `never`. `daily` routes the file target to a rolling appender that appends the UTC date to the file name. The resolved log path is logged at startup.
- `ZEUSDB_LOAD_REBUILD_GRAPH` forces `load()` to rebuild the graph from the stored records instead of restoring the saved dump.
- `max_training_vectors >= training_size`, `subvectors > 0` and `subvectors <= dim` are enforced in the Rust builder as well as the Python factory.
- Six worked examples under `examples/`, each carrying the transcript it reproduces, and test suites that execute the examples and every README example against a freshly built module.

### Changed

- `load()` restores the saved graph instead of rebuilding it. Loading a 50,000 record index of 1,536 dimensional embeddings drops from roughly three minutes to under two seconds. A directory whose dump is absent, damaged or written by an earlier release falls back to the rebuild, which produces a working index from any intact directory.
- Searches from different threads run concurrently, and a search no longer blocks behind an add. The interpreter lock is released during `add()`, `remove_point()`, `compact()`, `save()` and `rebuild_with_quantization()`, so other Python threads make progress during mutation.
- The `hnsw_rs` graph crate is vendored into the source tree at 0.3.4 with six patches, recorded in `vendor/hnsw_rs/ZEUSDB-PATCH.md`. The sdist ships the vendored tree and its licences, and building from source no longer pulls `hnsw_rs` from the registry.
- The centroid distance table stores only its strict upper triangle, halving its memory on a quantized index.
- `AddResult.summary()` returns plain ASCII.
- The saved directory manifest declares format version 1.1.0. Directories written by earlier releases still load.

---

## [0.4.1] - 2025-08-20

### Added
- Comprehensive test suite for overwrite behavior verification (`test_overwrite_fix.py`)
- Product Quantization (PQ) overwrite testing across all storage modes (`test_pq_overwrite_comprehensive.py`)
- Enhanced logging and storage analysis for overwrite operations
- Training state cleanup during document removal operations

### Changed
- Overwrite operations now use two-phase process (remove then add) to prevent duplicates
- `remove_point()` method now delegates to internal `remove_point_internal()` for better code reuse
- Enhanced `add()` method with comprehensive PQ support and storage mode awareness
- Improved error handling and logging throughout overwrite operations

### Fixed
- Critical: Fixed duplicate document bug where `overwrite=True` created multiple entries instead of replacing existing ones
- Memory leak from accumulated duplicate vectors in HNSW graph during overwrites
- Product Quantization codes and training state not properly cleaned up during document removal
- Vector count inconsistencies when removing documents during overwrite operations

### Removed
- Legacy overwrite behavior that created duplicates instead of proper replacements

---

## [0.4.0] - 2025-08-13

### Added
- Enterprise-grade structured logging with Python+Rust coordination
- Smart environment detection (production/development/testing/jupyter/CI)
- Automatic logging configuration with graceful fallbacks
- JSON and human-readable log formats with configurable targets (console/file)
- File logging with daily rotation and intelligent path handling
- Performance timing instrumentation on all hot paths (add, search, training)
- Comprehensive error context logging with field standardization
- Cross-platform logging support (Windows, macOS, Linux)
- Environment variable configuration for all logging aspects
- Production-ready observability for operations teams

### Changed
- Replaced debug println! statements with structured tracing throughout codebase
- Enhanced error handling with rich logging context instead of panic conditions
- Improved vector addition pipeline with detailed operation tracking
- Updated quantization training process with progress logging and timing metrics
- Modernized persistence operations with comprehensive save/load logging

### Fixed
- Eliminated potential panic conditions in distance space validation
- Improved error propagation with proper logging context
- Enhanced thread safety in concurrent logging scenarios
- Resolved cross-platform path handling inconsistencies
 
---

## [0.3.0] - 2025-08-06

### Added
- `save()` method to `HNSWIndex` for persisting index state to disk via Python and Rust.
- New `persistence.rs` module implementing index save/load logic, including manifest and file structure generation.
- PyO3 bindings for persistence-related methods, exposing them to Python.
- Internal unit tests for the `save` function to ensure correct file output and manifest validation.
- HNSW graph structure persistence via native hnsw-rs file_dump() integration
- Enhanced save workflow with Phase 2 graph serialization support
- Comprehensive Phase 2 integration test suite for full persistence validation
- Complete component loading infrastructure with helper functions for all ZeusDB file types
- load() method to VectorDatabase class for loading saved indexes from disk
- Comprehensive component validation and data consistency checking in load workflow
- Python API integration for load_index function with proper PyO3 bindings
- End-to-end test suite for component loading validation and error handling
- Complete HNSW graph loading functionality using NoData pattern from hnsw-rs
- anndists dependency for NoDist distance type compatibility
- Phase 2 graph structure loading with validation and error handling
- Full persistence roundtrip capability: save and load HNSW graph structures
- Empty index handling with conditional graph file creation for zero-vector scenarios
- Training state preservation with ID collection tracking during persistence
- Storage mode awareness in persistence (quantized_only vs quantized_with_raw handling)
- PQ centroids and codes serialization for complete quantization state preservation
- Compression statistics and memory usage reporting in manifest files
- Directory size calculation and file inventory tracking in manifest generation
- rebuilding_from_persistence flag to prevent training ID contamination during reconstruction
- Smart reconstruction approach using existing add() logic instead of complex graph deserialization
- Thread-safe data access patterns during save operations with proper lock management

### Changed
- Refactored `hnsw_index.rs` to integrate persistence logic and support serialization.
- Updated `lib.rs` to register the persistence module and ensure all new methods are exposed to Python.
- Enhanced error handling and docstrings for persistence operations.
- Modified HNSW initialization to use fixed max_layer=16 for hnsw-rs dump compatibility
- Updated manifest generation to include HNSW graph files (.hnsw.graph) and exclude data files (.hnsw.data)
- Enhanced save_manifest() with graph file tracking and size calculation
- Replaced placeholder load_index() with complete component loading implementation
- Enhanced lib.rs module exports to include load_index function for Python access
- Updated persistence.rs with comprehensive file loading and validation infrastructure
- Extended persistence.rs with complete HNSW graph loading using HnswIo and ReloadOptions
- Updated test suite to recognize and validate HNSW graph loading success
- Enhanced quantization config validation to include training state and storage mode persistence
- Modified PQ implementation to support set_trained() for persistence restoration
- Updated index reconstruction to use "Simple Reconstruction" pattern for reliability
- Refactored training threshold calculation to be self-healing during load operations
- Enhanced error collection and reporting throughout persistence workflow

### Fixed
- Improved reliability of index serialization and file output.
- Addressed edge cases in directory creation and file writing during persistence.
- Resolved critical "nb_layer != NB_MAX_LAYER" error preventing HNSW graph dumps
- Fixed layer count compatibility issue between ZeusDB and hnsw-rs library requirements
- Enabled successful HNSW graph structure serialization for graph files
- Resolved Python binding compilation error for load_index function export
- Fixed missing #[pyfunction] annotation preventing Python module integration
- Established proper API consistency between save and load methods
- Resolved anndists dependency issues for NoDist import compatibility
- Fixed HNSW graph loading import paths for hnsw-rs v0.3.0+ compatibility
- Resolved training ID loss during graph reconstruction by adding persistence rebuild flag
- Fixed PQ training state restoration ensuring loaded instances are properly marked as trained
- Corrected training progress calculation inconsistencies between save/load cycles
- Addressed quantization state contamination during index reconstruction
- Resolved thread safety issues in concurrent data access during persistence operations
- Fixed storage mode detection and raw vector preservation based on configuration
- Prevented training ID re-collection during persistence rebuild operations

---

## [0.2.1] - 2025-07-30

### Added
- Storage mode configuration for product quantization: New storage_mode parameter in quantization config allows users to choose between:
  - '"quantized_only"' (default): Maximum memory efficiency by discarding raw vectors after quantization
  - '"quantized_with_raw"': Keep both quantized codes and raw vectors for exact reconstruction
- Case-insensitive storage mode validation: Accepts variations like "Quantized_Only", "QUANTIZED_WITH_RAW"
- Automatic memory usage warnings: Users are warned when `quantized_with_raw` mode will use significantly more memory
- Enhanced subvector divisor suggestions: `_suggest_subvector_divisors()` now returns `list[int]` for programmatic use
- StorageMode enum: Rust backend support for `quantized_only` and `quantized_with_raw` storage modes with JSON serialization
- Storage mode parsing: Complete quantization config parsing in HNSWIndex constructor with proper error handling
- Intelligent vector retrieval: `get_records()` method now prioritizes raw vectors over PQ reconstruction when available
- Enhanced statistics: `get_stats()` now reports storage mode, memory usage breakdown, and storage strategy information
- Memory usage tracking: Real-time memory usage calculations for both raw vectors and quantized codes

### Changed
- Quantization config validation: Now includes comprehensive validation and normalization of all parameters
- Error messages: Improved clarity for storage mode validation with sorted mode suggestions
- Defensive programming: Added final safety checks to ensure complete configuration before passing to Rust backend
- QuantizationConfig struct: Now includes `storage_mode` field with backward-compatible defaults
- add_quantized_vector logic: Respects storage mode configuration to conditionally store raw vectors
- get_stats output: Enhanced with storage strategy descriptions ("memory_optimized" vs "quality_optimized")
- Vector storage behavior: `quantized_only` mode stops storing raw vectors after PQ training for maximum memory efficiency

### Fixed
- Configuration completeness: All quantization parameters now have guaranteed defaults to prevent missing key errors
- None value handling: Python config cleaning now properly removes `None` values before passing to Rust backend
- Constructor parameter validation: Improved error handling for missing or invalid quantization parameters
- Memory statistics accuracy: Corrected memory usage calculations based on actual storage mode behavior

---

## [0.2.0] - 2025-07-28

### Added
- Product Quantization (PQ) Support
  - Quantized vector storage with configurable compression ratios (4x-256x)
  - Automatic training pipeline with intelligent threshold detection
  - 3-path storage architecture for optimal memory usage:
    - Path A: Raw storage (no quantization)
    - Path B: Raw storage + ID collection (pre-training)
    - Path C: Quantized storage (post-training)

- Quantized Search API
  - Unified search interface supports both raw and quantized vectors transparently.
- Automatic fallback to raw search if quantization is not yet trained.
- Quantization-aware batch addition for efficient ingestion at scale.
- Detailed quantization diagnostics via get_quantization_info() (e.g., codebook stats, compression ratio, memory footprint).
- Debug logging macro (ZEUSDB_DEBUG) for controlled diagnostic output in Rust backend.
- Thread safety diagnostics in get_stats() (e.g., "thread_safety": "RwLock+Mutex").
- Improved test coverage for quantized and raw modes, including edge cases and error handling.

- Asymmetric Distance Computation (ADC) for fast quantized search
- Memory-efficient k-means clustering for codebook generation
- Configurable quantization parameters:
  - `subvectors`: Number of vector subspaces (divisor of dimension)
  - `bits`: Bits per quantized code (1-8)
  - `training_size`: Vectors needed for training (minimum 1000)
  - `max_training_vectors`: Maximum vectors used for training

- Enhanced Vector Database API
- Quantization configuration support in create() method
- Training progress monitoring with get_training_progress()
- Storage mode detection with get_storage_mode()
- Quantization status methods:
  - `has_quantization()`: Check if quantization is configured
  - `can_use_quantization()`: Check if PQ model is trained
  - `is_quantized()`: Check if index is using quantized storage
- Quantization info retrieval with `get_quantization_info()`
- Training readiness check with `is_training_ready()`
- Training vectors needed with `training_vectors_needed()`

- Performance Monitoring
  - Compression ratio calculation and reporting
  - Memory usage estimation for raw vs compressed storage
  - Training time measurement and optimization
  - Search performance metrics for quantized vs raw modes
  - Detailed statistics in 'get_stats()' method

- Input Handling
 - Enhanced dictionary input parsing with comprehensive error handling
 - Flexible metadata support for various Python object types
 - Automatic type detection and conversion for metadata
 - Graceful handling of None values and edge cases
 - Comprehensive input validation with descriptive error messages

- Performance Optimizations
 - Batch processing for large-scale vector additions
 - Optimized memory allocation during training and storage
 - Efficient vector reconstruction from quantized codes
 - Fast ADC search implementation with SIMD optimizations
 - Automatic performance scaling post-training (up to 8x faster additions)

### Changed
- Vector Addition Behavior
 - Automatic training trigger when threshold is reached during vector addition
 - Dynamic storage mode switching from raw to quantized seamlessly
 - Enhanced error reporting with detailed failure information in AddResult
 - Improved batch processing with better memory management

- Search Performance
 - Adaptive search strategy based on storage mode (raw vs quantized)
 - Optimized distance calculations for quantized vectors
 - Enhanced result quality with proper score normalization

- Index Architecture
 - 3-path storage system replaces simple raw storage
 - Intelligent memory management with automatic cleanup
 - Robust state transitions between storage modes
 - Enhanced concurrency handling with proper lock management

- Statistics and Monitoring
 - Extended statistics including quantization metrics
 - Real-time progress tracking during training operations
 - Enhanced memory usage reporting with compression analysis
 - Detailed timing information for performance optimization

- Default search parameters tuned for quantized and L1/L2 spaces (e.g., higher default ef_search for L1/L2).
- Improved error messages for quantization-related failures and configuration issues.
- Consistent handling of vector normalization (cosine) vs. raw (L1/L2) in all input/output paths.

### Fixed
- Memory Management
 - Fixed temporary value lifetime issues in PyO3 integration
 - Resolved borrow checker conflicts in quantization pipeline
 - Corrected memory leaks during large-scale operations
 - Fixed reference counting for Python object handling

- Vector Processing
 - Fixed input format parsing for edge cases and invalid data
 - Resolved metadata conversion issues for complex Python objects
 - Corrected vector dimension validation with proper error messages
 - Fixed batch processing memory allocation issues

- Performance Issues
 - Optimized training memory usage to prevent out-of-memory errors
 - Fixed search performance degradation in large indexes
 - Resolved training stability issues with improved k-means initialization
 - Corrected distance calculation accuracy in quantized mode

- Error Handling
 - Enhanced validation for quantization configuration parameters
 - Improved error propagation from Rust to Python
 - Fixed panic conditions in edge cases
 - Better handling of invalid input combinations

- Fixed rare edge case where quantization training could stall with duplicate vectors.
- Resolved non-deterministic search results in small datasets with L1/L2 metrics by tuning search parameters.
- Fixed debug output leaking to production logs (now controlled by environment variable).

### Removed
- Removed legacy single-path storage logic (now fully 3-path).
- Deprecated or removed any old quantization/test hooks that are no longer needed.

---

## [0.1.2] - 2025-07-17

### Added
- **Intelligent Batch Search**: Automatic batch processing for multiple query vectors
  - Transparent optimization: users get performance gains without API changes
  - Smart strategy selection: sequential processing for ≤5 queries, parallel for 6+ queries
  - Multiple input format support:
    - `List[List[f32]]` - Native Python lists of vectors
    - `NumPy 2D arrays (N, dims)` - Automatic batch detection
    - `NumPy 1D arrays (dims,)` - Single vector fallback
    - `List[f32]` - Traditional single vector (unchanged)
- Added comprehensive batch search test suite

### Changed
- Optimized GIL release patterns for better concurrent performance
- Reduced lock contention through intelligent batching strategies

---

## [0.1.1] - 2025-07-15

### Added
- Parallel batch insertion using `rayon` for large datasets (`insert_batch`).
- GIL-optimized `add_batch_parallel_gil_optimized()` path for inserts ≥ 50 items. (Removed in 0.2.0. `add()` has run one record at a time since.)
- Thread-safe locking using `RwLock` and `Mutex` for all core maps (`vectors`, `id_map`, etc.).
- `benchmark_concurrent_reads()` and `benchmark_raw_concurrent_performance()` for performance diagnostics. (`benchmark_raw_concurrent_performance()` removed as a duplicate of `benchmark_concurrent_reads()`, see Unreleased.)
- `get_performance_info()` for runtime introspection of bottlenecks and recommendations.
- Added `normalize_vector()` helper function to match Rust implementation behavior
- Added `assert_vectors_close()` utility for normalized vector comparison with tolerance
- Added additional tests for parallel batch processing validation, thread safety verification, and performance benchmarking.

### Changed
- `add()` now selects between sequential and parallel batch paths based on batch size.
- `search()` releases the Python GIL and performs fast concurrent metadata filtering and conversion.
- All internal maps (`vectors`, `metadata`, etc.) are now thread-safe for concurrent reads.
- Cosine vector normalization is now always applied consistently across all input formats.

### Fixed
- Prevented deadlocks and data races by isolating all shared state behind locks.
- Ensured proper ID overwrite handling across HNSW and reverse mappings with lock safety.
- Fixed HNSW test suite to properly account for cosine space vector normalization. Replace exact floating-point comparisons with normalized vector assertions. The HNSW implementation was working correctly from the start. The tests 
were actually validating that cosine normalization was properly implemented. 
- Fixed comprehensive search test expectations for HNSW approximation behavior

### Removed
- Legacy single-threaded insertion behavior (now delegated via `add_batch_*` paths).

---

## [0.1.0] - 2025-07-13

### Added
- **Generic `create()` method** for extensible vector index creation
  - Registry-based architecture supporting multiple index types
  - Case-insensitive index type matching: `create("HNSW")` or `create("hnsw")`
  - Comprehensive parameter defaults with Rust backend validation
  - Self-updating error messages showing all available index types
  - Supports case-insensitive index types (e.g. "HNSW" and "hnsw")
- **`available_index_types()`** class method for programmatic type discovery
- Future-ready architecture for IVF, LSH, Annoy, and Flat index types

### Changed
- ⚠️ **Breaking Change**: Replaced index-specific factory methods with generic `create()`
  - Migration: `VectorDatabase().create_index_hnsw(dim=768)` → `VectorDatabase().create("hnsw", dim=768)`
  - All HNSW parameters now default to best-practice values; dim is the only commonly customized field. Most of the settings like `m`, `ef_construction`, `expected_size`, and `space` already have good defaults, so users typically don't change them. The only one they usually set themselves is `dim`, since it must match the shape of their data.
  - Improved error messages with dynamic type listing

### Fixed
- Updated all internal testing files to use the new .create()` API

### Removed
- Index-specific factory methods (replaced by unified `create()` interface)

---

## [0.0.9] - 2025-07-10

### Added
- `search()` is a more accurate and industry-standard term for vector similarity retrieval.

### Changed
- ⚠️ Breaking Changes - Renamed `HNSWIndex.query()` → `HNSWIndex.search()` to better reflect its role as a k-nearest neighbor (KNN) similarity search method.
- Updated all internal references, tests, and examples to reflect the new `.search()` method name.

### Removed
- All usages of `.query()` must be replaced with `.search()`.

---

## [0.0.8] - 2025-07-10

### Added
- **Metadata filtering** support for HNSW vector indexes
  - Filters can be applied during `query()` using Python dictionaries
  - Supported operators:
    - Basic equality: `"field": value`
    - Comparison: `{"gt": val}`, `{"gte": val}`, `{"lt": val}`, `{"lte": val}`
    - String ops: `{"contains": "x"}`, `{"startswith": "x"}`, `{"endswith": "x"}`
    - Array ops: `{"in": [a, b, c]}`
  - Filters can be combined across fields using AND logic
  - Supports `None` for null value matching
- **serde** and **serde_json** dependencies:
  - Enables typed serialization and deserialization of metadata
  - Powers the new metadata filtering and storage system using `serde_json::Value`
- Comprehensive test suite for metadata filtering:
  - Covers string, numeric, boolean, array, and null filters
  - Includes multi-condition queries and invalid filter error handling
  - Validates type fidelity in round-trip metadata storage and retrieval

### Changed
- Vector metadata is now stored as `HashMap<String, Value>` for flexible typing

### Fixed
- Improved type extraction and conversion between Python and Rust for metadata fields

---

## [0.0.7] - 2025-07-08

### Added
- Support for multiple distance metrics in HNSW index creation:
  - `"cosine"` (default): cosine distance
  - `"L2"`: Euclidean distance
  - `"L1"`: Manhattan distance
- Metric selection is now configurable via the `space` argument in `VectorDatabase.create_index_hnsw()`
- Internal Rust implementation uses an enum-based dispatch for safe and performant metric switching
- Comprehensive test coverage added for all three metrics using shared query and add APIs

### Changed
- Distance metric names (`space` parameter) are now case-insensitive:
  - Accepts "L1", "l1", "L2", "l2", "Cosine", "cosine", etc.
- Internally stores normalized lowercase form (e.g., "l1") for consistency
- Error messages preserve original user input for clarity

---

## [0.0.6] - 2025-07-07

### Added
- `get_records()` method for retrieving one or more indexed records by ID.
 - Accepts either a single string ("doc1") or a list of strings (["doc1", "doc2"]).
 - Optional return_vector parameter (default: True) controls whether embedding vectors are included in the output.
 - Returns a list of Python dictionaries matching the query() response format
 - Missing IDs are silently skipped for graceful partial batch access.
 - Supports efficient batch usage with preallocation and avoids unnecessary `.clone()` calls.
 - Exposed with PyO3 signature binding for clean Python defaults.

### Changed
- `add()` now always performs an upsert by default: existing vectors with the same ID are overwritten.
- Removed distinction between "insert" and "overwrite" modes — no `overwrite` flag is needed.
- `AddResult` still reports all errors; successful overwrites are counted as successful additions.
- Old HNSW graph entries are logically removed by clearing internal ID mappings (`rev_map`, `id_map`) — queries will not return outdated vectors.
- `add()` now fully supports partial success: invalid records (e.g. bad vector shape) no longer abort the entire batch.
- `AddResult.vector_shape` now reflects total attempted records, even if some fail.
- Error messages now clearly indicate the failed record by ID and reason, improving debugging and retry workflows.

### Removed
- Removed early vector dimension validation in `add_batch_internal()` in favor of per-record validation inside `add_point_internal()`.

---

## [0.0.5] - 2025-07-06

### Changed
- Renamed BatchResult → AddResult to improve semantic clarity in both Rust and Python layers.
- Updated unit tests for `create_index_hnsw`, `query` and `search_with_metadata` methods to improve clarity and maintain edge case coverage. (Recorded as `create_index` and `similarity_search`. No `similarity_search` has ever existed in this package. The 0.0.5 test file exercised `query`, renamed to `search` in 0.0.9, and `search_with_metadata`.)
- Refactored test structure for better readability and maintainability.
- Expanded the README with clearer descriptions of the core 3-step workflow.
- Improved formatting and language for better readability and developer onboarding.

---

## [0.0.4] - 2025-07-03

### Added
- `return_vector: bool = False` parameter added to the `.query()` method.
  - When set to `True`, the returned results include the full embedding vector for each match.
  - Useful for downstream workflows such as LLM context injection, reranking, or embedding inspection.

### Changed
- `.query()` method now returns results as a list of Python dictionaries instead of tuples.
  - Old format: `[("doc_1", 0.87), ("doc_2", 0.91)]`
  - New format:
    ```python
    [
      {"id": "doc_1", "score": 0.87, "metadata": {...}},
      {"id": "doc_2", "score": 0.91, "metadata": {...}}
    ]
    ```
  - This change improves compatibility with modern machine learning workflows, LLM frameworks, and JSON-based APIs.
- Metadata filtering is still applied after ANN search and before result construction.
- Added `LICENSES/` directory to store third-party license files
- Included `hnsw_rs-Apache-2.0.txt` containing the full Apache License 2.0 text from the `hnsw_rs` crate (https://crates.io/crates/hnsw_rs)
- Updated `NOTICE` file to include proper attribution for `hnsw_rs`

### Removed
- `.search_with_metadata()` method has been removed. All functionality has been consolidated into the enhanced `.query()` interface.

---

## [0.0.3] - 2025-07-02

### Added
- Integrated `numpy = "0.25.0"` crate to support NumPy interoperability for Python bindings in `zeusdb-vector` via PyO3.
- Registered `BatchResult` class in the Python bindings for `zeusdb_vector_database`, making it accessible from Python alongside `HNSWIndex`.
- Internal test scripts for manual validation and experimentation. These are not integrated with `pytest` and are intended for ad hoc or exploratory testing.
- Introduced `BatchResult` class with structured summary of vector insertion, including total inserted, error count, and shape.
- Implemented a unified `add()` method in `HNSWIndex` supporting three common input formats:
  - Single object: `{"id": ..., "values": ..., "metadata": ...}`
  - List of objects: `[{"id": ..., "values": ...}, ...]`
  - Separate arrays: `{"ids": [...], "embeddings": [...], "metadatas": [...]}`
- Added robust input parsing and validation for each format, with detailed error handling.
- Enabled support for NumPy arrays (1D and 2D) in all input styles for seamless integration with Python scientific workflows.
- Extended internal batch insertion logic to track successes and errors, improving diagnostics and debugging.

---

## [0.0.2] - 2025-06-27

### Added
- `search_with_metadata` method on `HNSWIndex` for querying vectors with metadata in the results.
- Support for per-vector and index-level metadata (add/get/get_all) within `HNSWIndex`.
- Parameter validation in the `HNSWIndex` constructor to enforce safe index creation.
- `get_stats` and `info` methods on `HNSWIndex` for index statistics and summaries.
- Methods on `HNSWIndex` to list vectors, check for existence, and remove vectors by ID.
- `info()` method on `VectorDatabase` for usage guidance and available index types.
- Comprehensive test coverage for all HNSWIndex methods based on benchmark files
- Error handling tests for parameter validation and edge cases
- Tests for metadata functionality (both vector-level and index-level)
- Tests for utility methods (get_vector, get_vector_metadata, list, contains, remove_point)
- Tests for search functionality with and without metadata filtering

### Changed
- Rust module renamed from `create_index_hnsw.rs` to `hnsw_index.rs` for clarity and alignment with API naming.
- `VectorDatabase` is now a **pure stateless factory** — all index creation is handled here, but all vector operations are performed directly on `HNSWIndex`.
- Improved error handling and parameter validation in the Rust implementation.
- Enhanced docstrings and usage examples in Python for clearer developer experience.
- Updated maturin dependency requirement from >=1.8.7 to >=1.9.0 for both development dependencies and build system requirements.

### Fixed
- Fixed and clarified the code example in the README.
- Updated test suite to work with new stateless factory pattern API
- Fixed floating-point precision issues in vector comparison tests using approximate equality
- Updated test batch format from dictionary to tuple format to match Rust implementation

### Removed
- Removed `create_index_hnsw.py` from the Python package; logic is now part of the `VectorDatabase` factory.
- Removed `self.index` and all delegation methods (`add_point`, `query`, `add_batch`, etc.) from `VectorDatabase`; users now operate directly on the returned `HNSWIndex`.
- Removed info() static method from VectorDatabase class


---

## [0.0.1] - 2025-06-17

### Added
- Initial implementation of the ZeusDB Vector Database Python package with Rust backend.
- Support for HNSW (Hierarchical Navigable Small World) indexing.
- Python bindings via PyO3 for creating and querying vector indexes.
- HNSWIndex class with support for:
  - Vector insertion with optional metadata.
  - Top-K nearest neighbor search with optional metadata filtering.
  - Support for multiple distance metrics: cosine
- Metadata storage and retrieval at both index and vector level.
- VectorDatabase Python wrapper class to manage index creation and usage.
- Example benchmarking script for inserting and querying vectors.
- Introduced a new GitHub Actions workflow (`Check Package (without publishing to PyPI)`) for verifying ZeusDB builds across all supported platforms without pushing to PyPI. Includes metadata validation and artifact inspection.

### Fixed
- Replaced deprecated `license-file` metadata with SPDX-compliant `license = "Apache-2.0"` in `pyproject.toml` to fix PyPI distribution validation errors.
- CI workflow now correctly uploads built wheels to PyPI by using the `dist/*` pattern in the maturin upload step, resolving an issue where no files were found for publishing.

### Changed
- CI workflow now explicitly installs and upgrades `pkginfo` alongside `twine` and `wheel` during artifact inspection. This prevents metadata validation errors (such as "Metadata is missing required fields: Name, Version") caused by outdated versions of `pkginfo`.

---

## [Unreleased]

### Added
<!-- Add new features here -->

### Changed
<!-- Add changed behavior here -->

### Fixed
<!-- Add bug fixes here -->

### Removed
<!-- Add removals/deprecations here -->

---