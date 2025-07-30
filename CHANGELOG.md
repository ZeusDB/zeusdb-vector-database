# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.2] - 2025-08

### Added
<!-- Add new features here -->

### Changed
<!-- Add changed behavior here -->

### Fixed
<!-- Add bug fixes here -->

### Removed
<!-- Add removals/deprecations here -->

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
- GIL-optimized `add_batch_parallel_gil_optimized()` path for inserts ≥ 50 items.
- Thread-safe locking using `RwLock` and `Mutex` for all core maps (`vectors`, `id_map`, etc.).
- `benchmark_concurrent_reads()` and `benchmark_raw_concurrent_performance()` for performance diagnostics.
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
- Updated unit tests for `create_index` and `similarity_search` methods to improve clarity and maintain edge case coverage.
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