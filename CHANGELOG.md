# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.0] - 2026-08-25

The filter language gains boolean composition, presence tests and columns for the fields you declare, and a filtered search returns the nearest of the records the filter matches rather than whatever survived an unfiltered page. The index gains the verbs a store expects, `len()`, `in`, `count()`, `delete()`, `clear()`, `update_metadata()` and `rebuild()` among them, and `dot` as a fourth metric. A save is atomic and verified, a quantized index is scored on the metric it declared, and an unquantized index holds about half the memory it did. `dim` is now required, `add()` raises on a malformed batch and `l1` can no longer be quantized, so read the breaking list before upgrading. A directory saved by 0.7.0 opens unchanged unless it pairs `l1` with quantization or was created outside the new bounds.

### Breaking

- **`dim` is required on `create()`.** It defaulted to 1,536, which built an index sized for one vendor's model family and reported the mistake at the first `add()`, as a dimension mismatch on every vector of any other width. `create("hnsw")` without it raises `TypeError`. Pass the width your embedding model produces, for example `dim=1536` for OpenAI text-embedding-3-small or `dim=768` for most sentence-transformers models.
- **`add()` raises `ValueError` when the parallel arrays of a batch dict disagree in length**, naming both lengths and the short field. It used to insert a prefix in one direction and store the surplus records under generated ids in the other, so one id against two vectors reported `2 inserted, 0 errors`. Make `ids`, `metadatas` and the vector array the same length. Omitting `ids` entirely is not a disagreement and still generates one per record.
- **A parallel array that is not a `list` raises `TypeError`.** A tuple or an ndarray of `ids` or `metadatas` was counted as one error per record on the list path and discarded whole on the NumPy path, where every record then took a generated id. Pass a list, which for an array is `ids.tolist()`.
- **A metadata filter decides which records are ranked, not which results survive.** The traversal admits only matching records, so a filtered search returns the `top_k` nearest of the records the filter matches. It used to find the `top_k` nearest of every record and then discard the ones the filter rejected, so a selective filter returned a short or empty page. A filter matching 5 of 300 records at `top_k=5` returns all 5 where it returned none. Code that raised `top_k` to compensate can stop, and a test that pinned a short page under a filter needs its expectation refreshed. Where the filter matches fewer than 5,000 records those records are scored exactly rather than traversed, and that page is ordered by distance and then by id.
- **`l1` cannot be quantized.** `create(space="l1", quantization_config=...)` raises `ValueError`. A quantized graph scores every candidate from tables of squared L2 distances to the codebook, and Manhattan distance cannot be recovered from them, so such an index returned the wrong records and reported a score on a quantity it never declared. Use `space="l2"` with the same configuration, or `l1` without quantization.
- **A directory pairing `l1` with quantization no longer opens**, whatever release saved it. `load()` raises the same `ValueError`, naming the path. Open it under the release that saved it, read the records back with `get_records()`, and add them to an index created under `l2` or without quantization. Every other directory saved by 0.7.0 inside the bounds below opens without any action.
- **A quantized `cosine` index ranks and scores by cosine distance to the reconstruction.** It ranked and scored by the squared L2 distance to it and reported that sum as the score, which read 1.86 times the cosine distance at 25,000 records. Result order changes, in each measured case for the better. Recall at 10 rose by 0.0015 to 0.0518 across three corpora at two subvector counts each, with every 95 percent half width at or under 0.001. The exact scan path normalises the reconstruction before scoring, as `add()` does, where it scored against a reconstruction of length 0.90. A search costs 1 to 3 percent more and a build 3 to 8 percent more. A test that pinned scores or exact result sets from a quantized cosine index needs its expectations refreshed. A quantized cosine directory saved by an earlier release opens and is scored by cosine distance from then on. Its graph keeps the neighbour lists it was saved with. Loading it once with `ZEUSDB_LOAD_REBUILD_GRAPH=1` set and saving it again wires it on the new ordering.
- **A quantized `l2` index reports the rooted distance on every path.** The traversal returned the squared sum and the exact scan the rooted distance, so one index reported 159.416 and 12.626 for the same record depending on how selective the filter was. The page is rooted once per candidate, which does not move the order. A threshold on the score of a quantized `l2` index that was derived from the traversal path needs re-deriving on the rooted scale. An unquantized index is unchanged.
- **`dim`, `ef_construction`, `top_k` and `ef_search` have ceilings.** `dim` above 65,536 and `ef_construction` above 4,096 raise at `create()`, `rebuild()` and `load()`, and `top_k` above 65,536 and `ef_search` above 131,072 raise at `search()`. None had one, and `create(dim=2**40)` ended the process with an allocation failure. A directory saved with a value outside these bounds does not open under this release. Open it under the release that saved it, read the records back with `get_records()`, and add them to an index created inside the bounds.
- **`save()` and `load()` print nothing.** Both wrote progress lines to stdout, ending in a line reporting that the save or load had completed. A script that read stdout for those lines has nothing to read. `save()` returns `None` and raises on failure, `load()` returns the index, and the log records are unchanged.

### Fixed

- Records queued in the log file's writer at process exit were lost, because the appender's worker guard was never dropped and its `flush()` is a no-op. The guard is dropped at exit through an `atexit` hook registered on import, which waits for the writer to drain. `os._exit()` and an abort run no hook.
- A trained `quantized_with_raw` index saved while holding no records reopened without its raw vector store, so every record added after the load lost its raw vector permanently and `get_records()` returned a reconstruction. Reached by `clear()` or by removing every record before a save. The store is opened whenever the storage mode expects one.
- A `quantized_with_raw` directory whose `vectors.bin` was lost opened as a complete index built from reconstructions with the top page moved, one whose `quantization.json` was lost opened as unquantized, and one whose `pq_codes.bin` was lost reported itself quantized while holding no codes. A raw index saved over a quantized directory reopened carrying the previous save's codebook and codes. `load()` checks `files_included` in `manifest.json` against the directory in both directions before it parses anything, reads each quantization artefact only when the manifest names it, and refuses a directory with a file missing or with a file the manifest does not name, naming the file.
- `total_size_mb` in `manifest.json` counted the previous save's graph dump or none at all, `training_progress` in `get_stats()` read `0/1000 (100.0%)` on a trained index, `created_at` recorded the load rather than the creation when a loaded index was saved again, and `training_completed_at` was stamped at save rather than when the codebook was fitted. All four are corrected, and the two timestamps survive a save and load.
- `ZEUSDB_LOG_LEVEL=warning` was accepted by the Python layer and refused by the Rust layer, which printed `ignoring zeusdb_vector_database=warning` to stderr and then filtered nothing, and `warn` failed the other way round. Both layers accept `warn`, `warning`, `err`, `error`, `fatal` and `critical`.
- A filter nested about four thousand levels deep overflowed the stack and ended the process. Conversion is capped at 128 levels and group nesting at 10, and either raises `ValueError` naming the depth.
- `load()` validates what it reads. `config.json` and `quantization.json` are held to the rules `create()` applies to `dim`, `space`, `m`, `ef_construction`, `expected_size`, `bits` and `subvectors`, where an unrecognised `space` used to fall back to `cosine` in silence, `bits` of 40 ended the process, `bits` of 64 masked every code to zero and `subvectors` of 0 divided by zero. Every container length in `mappings.bin`, `vectors.bin`, `pq_codes.bin` and `pq_centroids.bin` is decoded under a budget of 64 bytes per byte the file holds, where ten of them went to the allocator unexamined and a declared length of 2^40 ended the process. A graph dump naming an origin id above what `config.json` counts is refused, where it sized the graph's id table from that id and ended the process. A damaged or hand edited directory raises where it used to end the process or open wrongly.
- The `quantized_with_raw` warning, the `create()` docstring and the README said the mode held less memory than an unquantized index above a break even. It holds more at every record count, 1.08 times at 50,000 records of dimension 1,536 and 1.14 times at dimension 128, because it keeps every raw vector and adds the codes and the trained tables to it. The warning describes the mode as the accuracy mode and quotes only figures exact from the configuration.
- Eight error messages carried runs of spaces where a line continuation had been lost.

### Added

- `space="dot"`, the inner product, as a fourth metric. `search()` reports `1 - dot` so that lower stays better, vectors are stored as given rather than normalised, and `index.space` reads the metric back. It cannot be combined with `quantization_config`, which raises `ValueError` at `create()`. The codebook is fitted by squared L2 and cannot rank by the inner product, with recall at 10 never above 0.37 by brute force over its own reconstructions across three corpora and stored length spreads up to three orders of magnitude, against an unquantized `dot` index at least 0.35 higher on the same data. Use `cosine` on normalised vectors where only direction should count, or `dot` without quantization where length must count.
- Boolean composition in the filter language. `$and`, `$or` and `$not` compose whole filters, where a mapping was only ever a conjunction of its fields. The three names are reserved as keys, so a field named `$and`, `$or` or `$not` raises `ValueError` naming the key, and any other name beginning with `$` still filters. Groups nest to 10 levels.
- Filter operators `nin`, `any`, `all`, `exists`, `is_missing` and `is_null`. `nin`, `any` and `all` exclude a record lacking the field, as the existing operators do, and `exists`, `is_missing` and `is_null` take `true` or `false` and refuse anything else.
- `create(indexed_fields=[...])` builds a column for each metadata field named, so a filter naming only declared fields is answered from the columns rather than by reading every record's metadata. Up to 32 names. At 100,000 records a filter matching one record costs 0.09 to 0.15 milliseconds where it cost 28 to 74, and eight declared columns cost 6.69 MB. A filter naming an undeclared field returns the same records, reads every record to find them and logs one warning naming the field, and one naming a declared field beside an undeclared one is bounded by the declared branch where that removes at least two thirds of the records. The declaration is carried in `config.json` and the columns are rebuilt from `metadata.json` on load, so no file is added and a directory saved before this opens unchanged. `index.indexed_fields` reads the declaration back.
- `len(index)`, `id in index`, and `count(filter=None)`, which walks every record with the interpreter lock released.
- `remove_points(ids)`, returning the ids that were not present, `remove_where(filter)`, returning how many it removed, and `delete(ids=..., where=...)`, which dispatches to them, returns a count, and raises when given both arguments or neither. `remove_where({})` is refused, because an empty filter matches every record everywhere else in the language and here that would empty the index.
- `clear()`, which replaces the graph rather than removing record by record, keeps the configuration including a fitted codebook, and returns the count removed. It does not reset the generated id counter, so ids generated after it continue the sequence.
- `update_metadata(id, metadata)`, which replaces one record's metadata wholesale, as `add(overwrite=True)` does, leaves the vector, the codes and the graph node alone, and returns whether the id was present.
- `rebuild(m=..., expected_size=..., ef_construction=...)`, which builds the graph again at a new configuration, in place. Every record keeps its vector, its metadata, its external id and its internal id, and a quantized index is rebuilt from its stored codes rather than re-encoded. The three arguments are held to the rules `create()` applies. `m` and `ef_construction` were fixed at construction before this.
- `shrink_to_fit()`, which returns the graph's spare buffer capacity to the allocator and reports the bytes released. `compact()` calls it.
- `list(after=...)`, a cursor naming the last id of the previous page, beside `offset`. A deletion ahead of an offset shifts the next page by one and a cursor does not. The two cannot be combined, and a cursor naming a removed record raises `KeyError` rather than returning a page from somewhere else.
- `get_records(strict=True)`, which raises `KeyError` naming every id the index does not hold. The default still skips them.
- `AddResult.ids`, every id the call put in the index, in insertion order.
- `index.space`, `index.m`, `index.ef_construction` and `index.expected_size` as read-only properties beside `index.dim`.
- `shutdown_logging()`, exported at package level, which drains the log file and closes it. It is registered with `atexit` on import, so a normally exiting process needs no call. Records emitted after an explicit call are discarded.
- `centroid_norm_memory_mb` in `get_stats()` on a quantized index, pricing the table of squared centroid norms the cosine scorer reads. It is folded into `total_memory_mb`.
- A CycloneDX SBOM of the crates the wheel links, attached to each GitHub Release, and dependency scanning with cargo-deny on pull requests touching the manifests and daily, reporting advisories to code scanning.
- A fuzzer over the graph dump reader, a randomised operation sequence checked after every step against a model of what the index holds, subprocess tests for every allocation bound, a comparison of every search page against a brute force ranking, and a lock rank registry that asserts the declared lock order on every acquisition in a debug build.

### Changed

- **An unquantized index holds about half the memory it did.** Every raw vector was held twice, once in a map keyed by external id and once inside the graph, byte for byte. It is held once, in a store addressed by node index that the graph reads rather than owns. At 50,000 records of dimension 1,536 the index commits 325.3 MiB where it committed 632.4. Search latency did not move and the on-disk format is unchanged. `graph_memory_mb` reports everything the graph holds apart from the raw vectors, and `total_memory_mb` no longer counts a raw vector twice, so both read lower on an index that has not changed.
- A save writes a sibling directory and renames it into place, so a reader sees the previous index or the new one and never a mixture, and no artefact of an earlier save survives. `manifest.json` is written last and records a length and a SHA-256 digest for every other artefact, which `load()` checks before parsing them. A directory saved by an earlier release carries no digests and opens as before. `format_version` is unchanged at 1.1.0.
- `list()` returns records in arrival order, ascending by internal id, where it returned hash map iteration order, which differed from one process to the next. The order survives `save()` and `load()`, and `offset` pages over it.
- A 2-D NumPy array of `float32` or `float64` given to `add()` or to batch `search()`, and a 1-D `float64` array given to single `search()`, is read directly rather than through the sequence protocol, which every such array fell through to before. The list path's empty batch check and dimension mismatch message apply on the array paths as well.
- Neighbour vectors are prefetched during a traversal on x86-64.
- The rebuild `load()` falls back to when a graph dump is absent or damaged runs inside the extension rather than through Python `add()` calls, and wires the same graph byte for byte.
- The `quantized_with_raw` warning is suppressed where the configuration never trains, the low dimension warning is removed, the `quantized_only` break even counts both codes a record carries, which moves the figure from 4,476 to 4,626 records at dimension 64, and every quantization warning attributes to the caller's `create()` line rather than to library internals.
- Every GitHub Actions step is pinned to a commit SHA, each wheel's SHA-256 is recorded at build and verified before publishing, a check-version job holds the tag, `pyproject.toml`, `vdb-core/Cargo.toml` and `__version__` to one canonical version, every job carries a timeout, the lock order suite runs on a pull request touching a module that takes a lock, and the compiler is pinned to 1.97.1 in `rust-toolchain.toml`. The Miri job is removed, and the two tests comparing the AVX kernels against the scalar path bit for bit run on every push.
- The README documents the new verbs, the filter composition and columns, what a filtered search costs, what each storage mode holds, and `is_training_ready`, `training_vectors_needed`, `rebuild_with_quantization`, `get_performance_info` and `benchmark_concurrent_reads`, and corrects what it said about batch atomicity, cosine normalisation and `quantized_only` reconstruction.

---

## [0.7.0] - 2026-08-18

The graph is ZeusDB's own. The structure, the traversal, the insert, the level stream, the distance trait and the on-disk format are all in this package, and the vendored `hnsw_rs` crate is gone. Builds, searches and searches taken while another thread inserts are all faster, a loaded index holds less memory, and the graph is saved in a new single file format. A directory saved by 0.6.0 or earlier still opens, rebuilds its graph once and writes the new file on the next save.

### Breaking

- **The saved graph is one file, `hnsw_index.zdbgraph`, in place of `hnsw_index.hnsw.graph` and `hnsw_index.hnsw.data`.** A directory saved by 0.6.0 or earlier opens without any action. Nothing reads the old pair, so the first load rebuilds the graph from the stored records and the next `save()` writes the single file. At 50,000 records of 1,536 dimensional embeddings that rebuild takes between two and three and a half minutes, against under two seconds for a restore, and it happens once per directory. A rebuilt graph is wired from the records rather than restored edge for edge, so exact result sets from such a directory can differ from the ones it returned before the upgrade. Code that lists the directory, copies named files or reads `files_included` in `manifest.json` needs the new name. `format_version` in the manifest is unchanged at 1.1.0.

### Fixed

- A directory saved at `m` 256 rebuilt its graph on every load, for ever. The old dump header held `m` in a single byte, so 256 was written as 0, the loader compared that against the 256 in `config.json` and fell back to the rebuild every time the directory was opened. The new header holds `m` in eight bytes and such a directory restores.
- A damaged or truncated graph file raises instead of ending the process. The old reader panicked on a malformed header and called `std::process::exit(1)` on a short data file. The new reader validates every length before it allocates and returns an error, which the load path treats as a reason to rebuild from the stored records.
- A graph file records the byte order and pointer width it was written with, and one written on a machine that disagrees is refused rather than read as noise. The old format recorded neither.
- `benchmark_concurrent_reads()` built its queries without processing them for the index's space, so on a cosine index it measured the traversal an unnormalised query takes rather than the traversal a real query takes. Its queries are now processed exactly as a caller's query is, and the throughput and speedup figures it reports move accordingly.

### Added

- `index_bookkeeping_memory_mb` in `get_stats()`, on every index. It prices the hash tables that find a record rather than the ones that hold one, and it is folded into `total_memory_mb`. It is set by the record count and not by the dimension, running at 265 bytes per record with no quantization, 334 under `quantized_with_raw` and 265 under `quantized_only`.
- A GitHub Release is created for each stable tag, titled `Version X.Y.Z`, with its body taken from the matching section of this file.

### Changed

- **The index runs on ZeusDB's graph.** Building, searching, saving, loading, `compact()`, `remove_point()` and overwrite all go through it. The replacement was compared against the vendored path over 155,790 cells of result ids, exact score bits, recall and re-saved dump bytes, with none differing, so an index already in use answers as it did.
- Index building through `add()` is 1.60x to 4.16x faster, measured at 10,000 and 50,000 records on 1,536 dimensional embeddings and on 128 dimensional SIFT vectors.
- Search is 1.46x to 3.11x faster on mean latency and 1.30x to 3.09x faster at p95, measured on four loaded 50,000 record indexes at `top_k` 10 across three metrics and all three storage modes.
- A loaded 50,000 record index holds 100 to 107 MiB less. A `quantized_only` index of 1,536 dimensional embeddings is 2.42x smaller and an unquantized 128 dimensional one is 2.17x smaller.
- Search under a concurrent insert is faster on both counts. Throughput rose by 1.44x to 2.61x and p95 latency fell by 1.46x to 2.48x, in every cell measured at 1, 4 and 16 searching threads. Throughput as a share of a no-insert baseline at the same thread count fell in five of those six cells, from a range of 73.3 to 94.2 percent to a range of 66.0 to 93.0 percent, because the baseline rose faster than the figure measured against it. The install phase of an insert takes the graph write guard, where the previous insert locked each neighbour list on its own and never blocked a search, so a search now waits briefly once per inserted record.
- The saved graph is smaller and loads faster. At 50,000 records the file is 5.5 percent smaller on an unquantized index and 48.8 percent smaller on a quantized one, and a load is 1.5 to 2.7 times faster.
- `graph_memory_mb` is exact arithmetic over the capacity of every buffer the graph holds rather than a sample of its adjacency, and `total_memory_mb` follows it. The new figure reads 370 bytes per node of structure across three metrics and both element types, where the sampled one varied from 1,261 to 2,085 bytes on those same four graphs. **An index holding no record no longer reports zero**, because the capacity `expected_size` reserves at creation is committed whether a record lands in it or not. A `dim` 32 index declaring 100 records reads 0.04 MB where it read 0.00.
- The capacity reserved at creation is capped at 128 MiB whatever `expected_size` declares, and the graph grows geometrically past it. An index declaring the documented maximum of 100,000,000 records committed 764 MB at creation before.
- Neighbour lists order equal distances by insertion order, because the list sorts are stable. On data where distances tie, which in practice means integer valued embeddings, a newly built index can differ from one built by 0.6.0 by a handful of edges. Measured recall is unchanged on three real datasets at 10,000 and 50,000 records, and a saved graph carries no sort, so a restored index is unaffected.
- The level generator and both of the quantizer's draws name `rand_chacha` directly rather than reaching it through `rand`'s `StdRng`, which is documented as free to change algorithm in any release. A routine `rand` update can no longer change the graph a given set of records builds. The stream itself is unchanged, so no graph moved.
- The distance kernels dispatch at run time to an AVX path on an x86-64 processor that offers one, and to the previous code on one that does not. Measured on their own the kernels are 1.20x to 1.55x faster and an index build 1.10x, and search latency did not move. Every distance the AVX path computes is bit-identical to the one the previous path computes, over 22,176 compared values, 24,000 orderings, three built graphs and 250 search pages.
- The Rust dependency tree fell from 157 crates to 118, the source distribution is 29.8 percent smaller and no longer ships a vendored crate, and a clean release build of the extension is 1.14x faster.
- The README records what `total_memory_mb` prices and what it leaves out, with the unpriced share measured against the resident set on all three storage modes, and its rerank calibration table is replaced by twelve measured cells covering three datasets at 50,000 and 100,000 records.

---

## [0.6.0] - 2026-08-15

This release changes which interpreters receive a wheel. The extension is built against CPython's stable ABI, so one wheel per platform serves every CPython from 3.10 up, and PyPy and the free-threaded builds install from the source distribution instead. A saved index is unaffected and a directory written by 0.5.0 loads unchanged.

### Breaking

- **PyPy and free-threaded CPython no longer receive a wheel.** Neither can load a stable ABI wheel, so `pip install zeusdb-vector-database` builds the extension from the source distribution on both. That needs a Rust toolchain on the machine and takes about seventy seconds where a wheel install takes a few. The install succeeds and the package behaves as it did, so this is a slower install rather than a broken one. Ordinary CPython still gets a wheel, on Linux x86_64 and aarch64 for both glibc and musl, on Windows x64 and on macOS arm64. That set of platforms is unchanged from 0.5.0, as is the minimum Python of 3.10.
- The published wheel set is 6 files rather than 41, one per platform, each tagged `cp310-abi3` and loading on every CPython from 3.10 up including releases that do not exist yet. A pinned wheel filename or a download URL carrying an interpreter tag such as `cp312-cp312` no longer resolves.
- A quantized index trained under this release returns slightly different results, because k-means training now draws from a fixed seed rather than an unseeded generator. Measured on 50,000 records of 1,536 dimensional embeddings at `top_k` 10, recall went from 0.4872 to 0.4861 without reranking and from 0.9907 to 0.9883 reranked. The earlier figures were single draws from an unseeded distribution rather than fixed values, and two unseeded builds of the same data differed from each other by 0.0005 on the same measure. A saved index keeps the codebook it was trained with and answers exactly as before.
- Two `TypeError` messages are reworded, because PyO3 owns them and changed them at 0.29. `HNSWIndex(...)` reports `cannot create 'builtins.HNSWIndex' instances` where it reported `No constructor defined`, and `search(None)` reports `'None' is not an instance of 'Sequence'` where it named `NoneType`. Both raise the same exception from the same place. Code matching on the message text needs updating.

### Fixed

- `get_stats()` could hang the calling thread forever on a quantized index when another thread was adding or removing records. Two lock ordering defects caused it, one taking the record map while three storage locks were held and one re-entering the training id lock it already held. Every figure `get_stats()` reports is unchanged.

### Added

- Quantization training is reproducible. The same records trained with the same configuration produce the same codebook, the same codes and the same graph on every run of the same build. Unquantized index builds were already reproducible.
- `create()` warns when `ef_construction` is not greater than `2 * m`. At or below that the neighbour selection heuristic does not run, and layer zero insertion keeps every candidate the construction search returns rather than pruning them. The warning names the budget and the largest `m` that leaves the heuristic running. No default combination reaches it, since the default `ef_construction` of 200 clears the budget of 32 at `m` 16 and 64 at `m` 32.

### Changed

- `compact()` is slower on a quantized index. Its graph rebuild now inserts sequentially so that two rebuilds of the same records wire the same graph, and at 50,000 records it takes about 78 seconds against about 27. `add()`, `search()` and a normal `load()` are unaffected, as is `compact()` on an unquantized index. The same cost applies to the rebuild that `load()` falls back to when a saved graph dump is absent or damaged.
- Index building is about 1.1 times faster at 768 and 1,536 dimensions, from accumulating the distance kernels into a vector register. The distances the new kernels compute are bit-identical to the ones 0.5.0 computed, so this change moves no result list and no recall figure. Search latency did not move measurably.
- The Rust dependency tree fell from 185 crates to 157, which shortens a build from source, and the unmaintained `atty` crate carrying RUSTSEC-2021-0145 is gone. The built extension is under half the size it was and exports one symbol, `PyInit_zeusdb_vector_database`, where it exported 65.
- The README entry for `ef_construction` is corrected. It claimed larger values increase memory, which they do not, since a node holds at most `2 * m` neighbours at layer zero whatever width found them. It now records what the parameter governs, the measured build time, the recall plateau per dataset and the `2 * m` threshold. The `create()` docstring matches it. Neither default changed.

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