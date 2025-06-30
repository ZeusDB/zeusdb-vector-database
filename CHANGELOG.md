# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.3]

### Added
<!-- Add new features here -->

### Changed
<!-- Add changed behavior here -->

### Fixed
<!-- Add bug fixes here -->

### Removed
<!-- Add removals/deprecations here -->

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