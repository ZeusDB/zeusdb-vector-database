# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2025-

### Added
<!-- Add new features here -->

### Changed
<!-- Add changed behavior here -->

### Fixed
<!-- Add bug fixes here -->

### Removed
<!-- Add removals/deprecations here -->

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