# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

