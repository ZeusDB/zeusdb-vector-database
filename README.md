<p align="center" width="100%">
  <img src="https://github.com/user-attachments/assets/ad21baec-6f4c-445c-b423-88a081ca2b97" alt="zeusdb-vector-database-logo-cropped" />
  <h1 align="center">ZeusDB Vector Database</h1>
</p>

<!-- <h2 align="center">Fast, Rust-powered vector database for similarity search</h2> -->
<!--**Fast, Rust-powered vector database for similarity search** -->

<!-- badges: start -->

<div align="center">
  <table>
    <tr>
      <td><strong>Meta</strong></td>
      <td>
        <a href="https://pypi.org/project/zeusdb-vector-database/"><img src="https://img.shields.io/pypi/v/zeusdb-vector-database?label=PyPI&color=blue"></a>&nbsp;
        <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13%7C3.14-blue?logo=python&logoColor=ffdd54"></a>&nbsp;
        <a href="https://github.com/zeusdb/zeusdb-vector-database/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>&nbsp;
        <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/Powered%20by-Rust-black?logo=rust&logoColor=white" alt="Powered by Rust"></a>&nbsp;
        <a href="https://github.com/ZeusDB"><img src="https://github.com/user-attachments/assets/e140d900-1160-4eaa-85c0-2b3507a5f0f5" alt="ZeusDB"></a>&nbsp;
      </td>
    </tr>
  </table>
</div>

<!-- badges: end -->

<br />

## ℹ️ What is ZeusDB Vector Database?

ZeusDB Vector Database is a high-performance, Rust-powered vector database designed for fast similarity search across high-dimensional data. It enables efficient approximate nearest neighbor (ANN) search, ideal for use cases like document retrieval, semantic search, recommendation systems, and AI-powered assistants.

ZeusDB leverages the HNSW (Hierarchical Navigable Small World) algorithm for speed and accuracy, with native Python bindings for easy integration into data science and machine learning workflows. Whether you're indexing millions of vectors or running low-latency queries in production, ZeusDB offers a lightweight, extensible foundation for scalable vector search.

<br/>

## ⭐ Features

🐍 User-friendly Python API for adding vectors and running similarity searches

🔥 High-performance Rust backend optimized for speed and concurrency

🔍 Approximate Nearest Neighbor (ANN) search using HNSW for fast, accurate results

📦 Product Quantization (PQ) for compact storage and faster distance computations

📥 Flexible input formats, including native Python types and NumPy arrays

🗂️ Metadata-aware filtering for precise and contextual querying

💾 Save and load complete indexes to disk

<br/>

## ✅ Supported Distance Metrics

ZeusDB Vector Database supports the following metrics for vector similarity search. All metric names are case-insensitive, so "cosine", "COSINE", and "Cosine" are treated identically.

| Metric | Description                          | Accepted Values (case-insensitive)  |
|--------|--------------------------------------|--------|
| cosine | Cosine Distance (1 - Cosine Similarity) | "cosine", "COSINE", "Cosine" |
| l1     | Manhattan distance                   | "l1", "L1" |
| l2     | Euclidean distance                 | "l2", "L2" |

### 📏 Scores vs Distances

All distance metrics in ZeusDB Vector Database return distance values, not similarity scores:

 - Lower values = more similar
 - A vector identical to the query scores 0.0, or a value within floating point error of it

This applies to all distance types, including cosine.

Under `cosine`, vectors are normalized to unit length when they are stored. A vector you read back with `return_vector=True` or `get_records()` is therefore the normalized form, not the values you supplied. Under `l1` and `l2` the values are stored unchanged.

A zero vector has no direction, so under `cosine` it sits at distance 1.0 from everything, including itself.

<br/>

## 📦 Installation

You can install ZeusDB Vector Database with 'uv' or alternatively using 'pip'.

### Recommended (with uv):
```bash
uv pip install zeusdb-vector-database
```

### Alternatively (using pip):
```bash
pip install zeusdb-vector-database
```

<br/>

## 🔥 Quick Start Example

```python
# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Initialize and set up the database resources
index = vdb.create(index_type="hnsw", dim=8)

# Vector embeddings with accompanying ID's and Metadata
records = [
    {"id": "doc_001", "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], "metadata": {"author": "Alice"}},
    {"id": "doc_002", "values": [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], "metadata": {"author": "Bob"}},
    {"id": "doc_003", "values": [0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], "metadata": {"author": "Alice"}},
    {"id": "doc_004", "values": [0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], "metadata": {"author": "Bob"}},
    {"id": "doc_005", "values": [0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], "metadata": {"author": "Alice"}},
]

# Upload records using the `add()` method
add_result = index.add(records)
print(add_result.summary())

# Perform a similarity search and print the top 2 results
query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

results = index.search(vector=query_vector, filter=None, top_k=2)

for i, res in enumerate(results, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.6f}, Metadata: {res['metadata']}")
```

*Results Output:*
```
5 inserted, 0 errors
1. ID: doc_001, Score: 0.000000, Metadata: {'author': 'Alice'}
2. ID: doc_003, Score: 0.000988, Metadata: {'author': 'Alice'}
```

`add_result.summary()` returns a plain ASCII string, so it prints on any console encoding. The same counts are on `add_result.total_inserted` and `add_result.total_errors` if you want the numbers rather than the sentence.

<br/>

## ✨ Usage

ZeusDB Vector Database makes it easy to work with high-dimensional vector data using a fast, memory-efficient HNSW index. Whether you're building semantic search, recommendation engines, or embedding-based clustering, the workflow is simple and intuitive.

**Three simple steps**

1. **Create an index** using `.create()`
2. **Add data** using `.add(...)`
3. **Conduct a similarity search** using `.search(...)`

Each step is covered below.

<br/>

### 1️⃣ Create an Index

To get started, first initialize a VectorDatabase and create an HNSWIndex. You can configure the vector dimension, distance metric, and graph construction parameters.

```python
# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Initialize and set up the database resources
index = vdb.create(
    index_type="hnsw",
    dim=8,
    space="cosine",
    m=16,
    ef_construction=200,
    expected_size=5,
)
print(index.info())
```

*Output*
```
HNSWIndex(dim=8, space=cosine, m=16, ef_construction=200, expected_size=5, vectors=0, quantization=none)
```

<br/>

#### 📘 Parameters - `create()`

| Parameter        | Type   | Default   | Description                                                                 |
|------------------|--------|-----------|-----------------------------------------------------------------------------|
| `index_type`     | `str`  | `"hnsw"`  | The type of vector index to create. Currently only `"hnsw"` is supported. Case-insensitive. |
| `dim`            | `int`  | *required* | Dimensionality of the vectors to be indexed. Each vector must have this length. Must be positive. **Required as of 0.8.0**, see below. |
| `space`          | `str`  | `"cosine"`| Distance metric used for similarity search. One of `"cosine"`, `"l1"`, `"l2"`. Case-insensitive. |
| `m`              | `int`  | `16` or `32`, see below | Number of bi-directional connections created for each new node, from 2 to 256. Higher `m` improves recall but increases index size and build time. |
| `ef_construction`| `int`  | `200`     | Width of the candidate search each insertion runs. Must be positive. It costs build time and buys graph quality, and it changes neither search latency nor the size of the finished index. See below. |
| `expected_size`  | `int`  | `10000`   | Estimated number of records to be inserted, from 1 to 100,000,000. Used for preallocating internal data structures and for choosing the default `m`. Not a hard limit, see below. |
| `quantization_config` | `dict` | `None` | Product Quantization configuration for memory-efficient vector compression. See [Product Quantization](#️-product-quantization). |

**`dim` is required.** There is no default, because `dim` has to equal the width your embedding model produces and an index built at any other width rejects every vector you add to it. Omitting it raises `TypeError`.

```python
try:
    vdb.create("hnsw")
except TypeError as error:
    print(error)
```

*Output*
```
create() requires 'dim', the width of the vectors this index will hold. There is no default because dim has to equal the width your embedding model produces, and an index built at any other width rejects every vector you add to it. Read it off one embedding with len(vector), or pass the width your model documents, for example dim=1536 for OpenAI text-embedding-3-small or dim=768 for most sentence-transformers models.
```

**The default `m` depends on `expected_size`.** It is 16 for an `expected_size` of 25,000 or less, and 32 above that. `m` is fixed once the index is created and a graph too sparse for the number of records loses recall that no search width recovers, so declare `expected_size` honestly or set `m` yourself. Passing `m` explicitly always wins.

```python
vdb.create("hnsw", dim=8, expected_size=25_000).get_stats()["m"]   # '16'
vdb.create("hnsw", dim=8, expected_size=25_001).get_stats()["m"]   # '32'
```

**`expected_size` is a hint and not a limit.** An index accepts more records than it declared and the graph grows to fit them. What it does not change is `m`. Passing twice the declared size logs a warning once, on the `add()` that crosses it.

**`ef_construction` costs build time and buys graph quality.** It changes neither search latency nor the size of the finished index. Build time is linear in it above 100, so 50,000 records of `dim=1536` build in 76.9 s at the default and 262.1 s at 800. Recall stops improving at or near the default on most data, and where it keeps climbing a larger `ef_search` buys more for less, so raise `ef_search` before raising this.

**Keep `ef_construction` above `2 × m`.** At or below the neighbour budget the graph keeps every candidate the insertion search returned and prunes none of them. `create()` warns when the pair reaches that point. The defaults are clear of it.

<br/>

### 2️⃣ Add Data to the Index

ZeusDB provides a flexible `.add(...)` method that supports multiple input formats for inserting or updating vectors in the index. Whether you're adding a single record, a list of documents, or structured arrays, the API is designed to be both intuitive and robust. Each record can include optional metadata for filtering or downstream use.

All formats return an `AddResult` containing `total_inserted`, `total_errors`, `errors`, `vector_shape` and `ids`.

#### ✅ Format 1 – Single Object

```python
index = vdb.create("hnsw", dim=2)

add_result = index.add({
    "id": "doc1",
    "values": [0.1, 0.2],
    "metadata": {"text": "hello"}
})

print(add_result.total_inserted, add_result.total_errors)
print(add_result.is_success())
```

*Output*
```
1 0
True
```

#### ✅ Format 2 – List of Objects

```python
index = vdb.create("hnsw", dim=2)

add_result = index.add([
    {"id": "doc1", "values": [0.1, 0.2], "metadata": {"text": "hello"}},
    {"id": "doc2", "values": [0.3, 0.4], "metadata": {"text": "world"}},
])

print(add_result.total_inserted, add_result.total_errors)
print(add_result.vector_shape)
print(add_result.errors)
```

*Output*
```
2 0
(2, 2)
[]
```

#### ✅ Format 3 – Separate Arrays

```python
index = vdb.create("hnsw", dim=2)

add_result = index.add({
    "ids": ["doc1", "doc2"],
    "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    "metadatas": [{"text": "hello"}, {"text": "world"}],
})
print(add_result)
```

*Output*
```
AddResult(inserted=2, errors=0, shape=Some((2, 2)))
```

The `Some(...)` wrapper appears only in the printed form. `add_result.vector_shape` is the plain tuple `(2, 2)`.

#### ✅ Format 4 – Using NumPy Arrays

ZeusDB also supports NumPy arrays as input for seamless integration with scientific and ML workflows.

```python
import numpy as np

index = vdb.create("hnsw", dim=4)

data = [
    {"id": "doc2", "values": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), "metadata": {"type": "blog"}},
    {"id": "doc3", "values": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32), "metadata": {"type": "news"}},
]

result = index.add(data)

print(result.total_inserted, result.total_errors)
```

*Output*
```
2 0
```

#### ✅ Format 5 – Separate Arrays with NumPy

```python
index = vdb.create("hnsw", dim=2)

add_result = index.add({
    "ids": ["doc1", "doc2"],
    "embeddings": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    "metadatas": [{"text": "hello"}, {"text": "world"}],
})
print(add_result)
```

*Output*
```
AddResult(inserted=2, errors=0, shape=Some((2, 2)))
```

Each format is parsed and validated automatically. Invalid records are skipped rather than aborting the call, and the reason for each is returned in `errors`. A record whose vector contains `NaN` or an infinity is rejected this way.

<br/>

#### ⚠️ Adding an ID that already exists

`add()` upserts by default. Re-adding an existing ID **replaces the whole record**, metadata included. Metadata is not merged, so a key you leave out of the new record is gone.

```python
index = vdb.create("hnsw", dim=2)
index.add({"id": "doc1", "values": [0.1, 0.2], "metadata": {"text": "hello", "lang": "en"}})

# "lang" is not carried over
index.add({"id": "doc1", "values": [0.3, 0.4], "metadata": {"text": "goodbye"}})
print(index.get_records("doc1", return_vector=False))

# overwrite=False rejects the record instead, and counts it as an error
rejected = index.add({"id": "doc1", "values": [0.5, 0.6]}, overwrite=False)
print(rejected.total_inserted, rejected.total_errors)
print(rejected.errors)
```

*Output*
```
[{'id': 'doc1', 'metadata': {'text': 'goodbye'}}]
0 1
["Vector doc1: ValueError: Vector with ID 'doc1' already exists"]
```

A rejected record is reported in the `AddResult`. It does not raise. The rejection is also logged at WARNING level, which is visible on stderr under the default development settings.

Every overwrite leaves a node behind in the graph. See [`compact()`](#️-reclaim-space-left-by-removals-and-overwrites).

<br/>

#### 📘 Parameters - `add()`

The `add()` method inserts or replaces one or more vectors in the index.

| Parameter | Type                                | Default | Description |
|-----------|-------------------------------------|---------|-------------|
| `data`    | `dict`, `list[dict]`, `dict` of arrays, or `np.ndarray` | *required* | Input records to upsert into the index. Supports the five formats above. |
| `overwrite` | `bool`                            | `True`  | Whether an ID already in the index is replaced. With `False`, a colliding record is skipped and counted as an error. |

**The parallel arrays must be the same length.** Formats 3 and 5 pair `ids[i]` with `vectors[i]` and `metadatas[i]` by position, so a disagreement in length is a caller error and raises `ValueError` naming both lengths and which field is short. Nothing is inserted before the raise, so the call is safe to retry.

```python
two_wide = vdb.create("hnsw", dim=2)
try:
    two_wide.add({"ids": ["c", "d", "e"], "embeddings": [[0.1, 0.2], [0.3, 0.4]]})
except ValueError as error:
    print(error)
print(len(two_wide))
```

*Output*
```
add received 3 entries under 'ids' and 2 under 'embeddings'. A batch pairs them by position, so the two must be the same length, and 'embeddings' is the short one. Supply one id per vector, or omit 'ids' entirely.
0
```

The rule covers `ids`, `metadatas` and `metadata`, under every spelling of the vector key, on both the list and the NumPy branch. Omitting `ids` entirely is not a disagreement and still generates one per record. A parallel array must be a `list`; a `tuple` or an `ndarray` raises `TypeError`.

**Returns:**
`AddResult` with:
- `total_inserted`: number of records successfully inserted or replaced
- `total_errors`: number of failed records
- `errors`: list of error messages
- `vector_shape`: the shape of the processed batch, as `(rows, dim)`
- `ids`: the ID of every record that was inserted or replaced, in insertion order
- `is_success()`: `True` when `total_errors` is zero
- `summary()`: a one-line string of the two counts

`ids` is how you learn the IDs the index generated for records you supplied without one. It lines up with `total_inserted` and with nothing else, so a rejected record contributes no ID and `errors` is what names it.

```python
index = vdb.create("hnsw", dim=2)
generated = index.add({"vectors": [[0.1, 0.2], [0.3, 0.4]]})
print(generated.ids)

supplied = index.add({"ids": ["a", "b"], "embeddings": [[0.5, 0.6], [0.7, 0.8]]})
print(supplied.ids)

partial = index.add({"ids": ["ok", "bad"], "embeddings": [[0.1, 0.2], [0.1]]})
print(partial.ids, partial.total_inserted, partial.total_errors)
```

*Output*
```
['vec_1', 'vec_2']
['a', 'b']
['ok'] 1 1
```

<br/>

### 3️⃣ Conduct a Similarity Search

Query the index using a new vector and retrieve the top-k nearest neighbors. You can also filter by metadata or return the stored vectors.

The examples below all run against this index:

```python
index = vdb.create(index_type="hnsw", dim=8)
index.add([
    {"id": "doc_001", "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], "metadata": {"author": "Alice"}},
    {"id": "doc_002", "values": [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], "metadata": {"author": "Bob"}},
    {"id": "doc_003", "values": [0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], "metadata": {"author": "Alice"}},
    {"id": "doc_004", "values": [0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], "metadata": {"author": "Bob"}},
    {"id": "doc_005", "values": [0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], "metadata": {"author": "Alice"}},
])
query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]
```

#### 🔍 Search Example 1 - Basic (Returning Top 2 most similar)

```python
results = index.search(vector=query_vector, top_k=2)
for res in results:
    print(res["id"], round(res["score"], 6), res["metadata"])
```

*Output*
```
doc_001 0.0 {'author': 'Alice'}
doc_003 0.000988 {'author': 'Alice'}
```

#### 🔍 Search Example 2 - Query with metadata filter

```python
results = index.search(vector=query_vector, filter={"author": "Alice"}, top_k=5)
for res in results:
    print(res["id"], round(res["score"], 6), res["metadata"])
```

*Output*
```
doc_001 0.0 {'author': 'Alice'}
doc_003 0.000988 {'author': 'Alice'}
doc_005 0.001143 {'author': 'Alice'}
```

**The filter decides which records are ranked, not which results survive.** A search asking for five results with a filter matching a hundred records returns the five nearest of those hundred. `top_k` is the page size and nothing else, so there is no need to raise it when you filter. See [Metadata Filtering](#️-metadata-filtering) for what that costs.

#### 🔍 Search Example 3 - Search results include vectors

Set `return_vector=True` to get the stored embedding alongside the metadata and score. Under `cosine` this is the normalized vector, not the values you supplied.

The vector comes back as a **`numpy.ndarray` of `float32`**, from both `search` and `get_records`. Anything that concatenates it with `+`, calls `.append` on it, tests it with `if vector:` or passes it to `json.dumps` needs `.tolist()` first.

```python
results = index.search(vector=query_vector, top_k=1, return_vector=True)
print(results[0]["id"], round(results[0]["score"], 6))
print([round(float(v), 4) for v in results[0]["vector"]])
```

*Output*
```
doc_001 0.0
[0.0913, 0.1826, 0.2739, 0.0913, 0.3651, 0.1826, 0.5477, 0.639]
```

#### 🔍 Search Example 4 - Batch Search with a list of vectors

Perform a similarity search on multiple query vectors at once. The result is a list of result lists, one per query, in the order the queries were given.

```python
batch = [
    [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7],
    [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9],
]
results = index.search(vector=batch, top_k=2)
for q, hits in enumerate(results):
    print(f"query {q}:", [(h["id"], round(h["score"], 6)) for h in hits])
```

*Output*
```
query 0: [('doc_001', 0.0), ('doc_003', 0.000988)]
query 1: [('doc_002', 0.0), ('doc_004', 0.002238)]
```

#### 🔍 Search Example 5 - Batch Search with NumPy Array

```python
query_batch = np.array(batch, dtype=np.float32)

results = index.search(vector=query_batch, top_k=2)
for q, hits in enumerate(results):
    print(f"query {q}:", [h["id"] for h in hits])
```

*Output*
```
query 0: ['doc_001', 'doc_003']
query 1: ['doc_002', 'doc_004']
```

#### 🔍 Search Example 6 - Batch Search with metadata filter

The same filter is applied to every query in the batch. Each query gets the two nearest of Alice's documents, which for the second query are not among its two nearest documents overall.

```python
results = index.search(batch, filter={"author": "Alice"}, top_k=2)
for q, hits in enumerate(results):
    print(f"query {q}:", [h["id"] for h in hits])
```

*Output*
```
query 0: ['doc_001', 'doc_003']
query 1: ['doc_005', 'doc_003']
```

<br/>

#### 📘 Parameters - `search()`

The `search()` method retrieves the top-k most similar vectors from the index given an input query vector. Results include the vector ID, distance score, metadata, and optionally the stored vector.

| Parameter         | Type                            | Default   | Description                                                                 |
|------------------|----------------------------------|-----------|-----------------------------------------------------------------------------|
| `vector`         | `List[float]`, `List[List[float]]`, or `np.ndarray`  | *required* | The query vector (single: `List[float]` or 1D `np.ndarray`) or batch of query vectors (`List[List[float]]` or 2D `np.ndarray`). Must match the index dimension and contain only finite values. |
| `filter`         | `Dict[str, Any] \| None`         | `None`    | Optional metadata filter. Values may be a plain value for equality or a dict of operators, and `$and`, `$or` and `$not` compose them. See [Filter Operators](#-filter-operators-reference) and [Boolean composition](#-boolean-composition). |
| `top_k`          | `int`                            | `10`      | Number of nearest neighbors to return. |
| `ef_search`      | `int \| None`                    | see below | Search complexity parameter. Higher values improve accuracy at the cost of speed. |
| `return_vector`  | `bool`                           | `False`   | If `True`, each result includes the stored embedding vector under a `vector` key. |
| `rerank`         | `int \| None`                    | derived from the record count | Candidates fetched per requested result before rescoring against raw vectors. Only applies to a quantized index whose `storage_mode` is `quantized_with_raw`. See [Quantized search accuracy](#-quantized-search-accuracy). |

**The default `ef_search` depends on the distance metric.** It is `max(2 × top_k, 100)` for `cosine` and `max(2 × top_k, 150)` for `l1` and `l2`.

A query vector containing `NaN` or an infinity raises `ValueError` rather than returning meaningless distances.

<br/>

### 🧰 Additional functionality

ZeusDB Vector Database includes a suite of utility functions to help you inspect, manage, and maintain your index. You can view index configuration, attach custom metadata, list stored records, and remove vectors by ID.

#### ☑️ Check the details of your HNSW index

```python
print(index.info())
```
*Output*
```
HNSWIndex(dim=8, space=cosine, m=16, ef_construction=200, expected_size=10000, vectors=5, quantization=none)
```

The `vectors=` field is the live record count, in every storage mode. `get_vector_count()` returns the same number. `get_stats()["raw_vectors_stored"]` is the one that counts raw vectors specifically, and on a trained `quantized_only` index it is zero.

Other single-value accessors: `index.dim`, `index.space`, `index.m`, `index.ef_construction`, `index.expected_size`, `index.get_space()`, `len(index)`, `index.get_vector_count()`, `index.has_quantization()`, `index.can_use_quantization()`, and `VectorDatabase.available_index_types()`.

`index.dim`, `index.space`, `index.m`, `index.ef_construction` and `index.expected_size` are read-only properties. `get_space()` is the same value as a method and is kept for callers already using it.

```python
print(index.m, index.ef_construction, index.expected_size)
```
*Output*
```
16 200 10000
```

<br/>

#### ☑️ Add index level metadata

Index level metadata is a flat `str` to `str` map, separate from the per-record metadata used for filtering. It is preserved by `save()` and `load()`.

```python
index.add_metadata({
    "creator": "John Smith",
    "version": "0.1",
    "created_at": "2024-01-28T11:35:55Z",
    "embedding_model": "openai/text-embedding-ada-002",
    "environment": "production",
})

# View index level metadata by key
print(index.get_metadata("creator"))

# View all index level metadata
for key, value in sorted(index.get_all_metadata().items()):
    print(f"{key}: {value}")
```
*Output*
```
John Smith
created_at: 2024-01-28T11:35:55Z
creator: John Smith
embedding_model: openai/text-embedding-ada-002
environment: production
version: 0.1
```

`get_all_metadata()` returns a `dict` whose iteration order is not stable, which is why the example sorts it.

<br/>

#### ☑️ List records in the index

```python
for record_id, metadata in index.list(number=5):
    print(record_id, metadata)
```
*Output*
```
doc_001 {'author': 'Alice'}
doc_002 {'author': 'Bob'}
doc_003 {'author': 'Alice'}
doc_004 {'author': 'Bob'}
doc_005 {'author': 'Alice'}
```

`list()` returns `(id, metadata)` tuples **in the order the records were added**, and `offset` pages through them. It lists every record, in every storage mode, and the order survives `save()` and `load()`.

```python
print(index.list(number=2, offset=0))
print(index.list(number=2, offset=2))
print(index.list(number=2, offset=4))
print(index.list(number=2, offset=99))
```
*Output*
```
[('doc_001', {'author': 'Alice'}), ('doc_002', {'author': 'Bob'})]
[('doc_003', {'author': 'Alice'}), ('doc_004', {'author': 'Bob'})]
[('doc_005', {'author': 'Alice'})]
[]
```

An offset past the end returns an empty list rather than raising.

**Deleting while you page still shifts the pages.** Removing a record ahead of your cursor moves everything behind it up by one. If you cannot tolerate that, page by remembering the last ID you saw instead of a count.

<br/>

#### ☑️ Inspect index statistics

```python
stats = index.get_stats()
for key in ["total_vectors", "graph_nodes", "stranded_graph_nodes", "storage_mode_description"]:
    print(f"{key}: {stats[key]}")
```
*Output*
```
total_vectors: 5
graph_nodes: 5
stranded_graph_nodes: 0
storage_mode_description: raw_only
```

`get_stats()` returns a `str` to `str` map. Every key it carries:

| Key | Holds |
| --- | --- |
| `dimension`, `space`, `m`, `ef_construction`, `expected_size`, `index_type` | The configuration the index was created with |
| `total_vectors` | The live record count |
| `graph_nodes`, `stranded_graph_nodes` | Nodes in the HNSW graph, and how many of them no record uses |
| `raw_vectors_stored`, `quantized_codes_stored` | Records held at full width, and records held as codes |
| `storage_mode`, `storage_mode_description`, `storage_strategy` | What the index is storing and serving |
| `thread_safety` | The locking the index uses |
| `graph_memory_mb` | The HNSW graph |
| `raw_vectors_memory_mb` | The raw vector store |
| `quantized_codes_memory_mb` | The codes, which grow with the record count |
| `codebook_memory_mb`, `sdc_table_memory_mb` | The trained tables, fixed by `dim`, `subvectors` and `bits` |
| `index_bookkeeping_memory_mb` | The hash tables that find a record |
| `total_memory_mb` | The sum of the six figures above |
| `quantization_type` | `pq`, or `none` on an unquantized index |
| `raw_vectors_retained` | The storage mode's policy, `none_once_trained` or `all_records`. Quantized indexes only |

On a quantized index it also carries `quantization_active`, `quantization_trained`, `quantization_compression_ratio`, `quantization_training_size`, `training_progress`, `training_threshold_reached`, `training_vectors_needed`, `raw_vectors_retained` and the rerank calibration keys below.

**`total_memory_mb` is not the resident set and it does not claim to be.** It prices the six structures the index asked the allocator for, and leaves out what the allocator takes to hold them. Measured on three loaded indexes of 50,000 real 1,536-dimensional embeddings, it accounts for 0.88 of resident memory with no quantization, 0.88 under `quantized_with_raw` and 0.66 under `quantized_only`. Size infrastructure from the resident figure rather than from this one.

`index_bookkeeping_memory_mb` is proportional to the record count and independent of the dimension. It is not independent of the metadata, because the per-record metadata map is one of the tables it counts.

It also reports what a quantized search will fetch. `rerank_default_fetch` is the number of candidates a search at `top_k=10` fetches and rescores at the record count the index holds now, and a search at a larger `top_k` fetches more than it reports. `rerank_calibrated` is `true` on a trained `quantized_with_raw` index and `false` on every other one, including an index saved before the calibration existed. When it is `true`, these report what training measured:

| Key | Holds |
| --- | --- |
| `rerank_calibration_fetch` | The fetch measured on the training sample |
| `rerank_calibration_records` | The records it was measured over |
| `rerank_calibration_queries` | The queries it used |
| `rerank_calibration_target_recall` | The recall it was measured to reach |
| `rerank_calibration_exponent` | How the fetch is scaled as the index grows |
| `rerank_calibration_fit_fetches` | The fetches the exponent was fitted from, comma separated |
| `rerank_calibration_pages` | The page sizes the fetch was measured at |
| `rerank_calibration_page_fetches` | The fetch at each of those pages |
| `rerank_calibration_page_exponent` | The slope through those pages |
| `rerank_calibration_ms` | What the calibration cost |

<br/>

#### ☑️ Remove Records

Remove a vector and its metadata with `.remove_point(id)`. This performs a <u>logical deletion</u>:
- The vector is deleted from internal storage.
- The metadata is removed.
- The vector ID is no longer returned by `.contains()`, `.get_records()`, or `.search()`.

```python
index.remove_point("doc_001")
print("doc_001 present:", index.contains("doc_001"))
print("records remaining:", index.get_vector_count())
```
*Output*
```
doc_001 present: False
records remaining: 4
```

**⚠️ Please Note:** Due to the nature of HNSW, the underlying graph node remains in memory after a point is removed. Searches never return it, but it still occupies memory and edge slots. `compact()` reclaims those nodes.

`remove_points(ids)` and `remove_where(filter)` remove a batch and a filtered set. Both are below, at the end of this section.

<br/>

#### ♻️ Reclaim space left by removals and overwrites

Both `remove_point()` and an overwriting `add()` leave a node behind in the graph. `compact()` rebuilds the graph in memory and returns the number of nodes it reclaimed. IDs, metadata, stored vectors, quantized codes and PQ training state all survive.

```python
print("stranded graph nodes:", index.get_stats()["stranded_graph_nodes"])
print("reclaimed:", index.compact())
print("stranded graph nodes:", index.get_stats()["stranded_graph_nodes"])
```
*Output*
```
stranded graph nodes: 1
reclaimed: 1
stranded graph nodes: 0
```

`compact()` costs a full rebuild, proportional to the number of live records rather than to the amount of debris, and it holds both graphs in memory while it runs. It returns 0 and does nothing when there is nothing to reclaim. It is never automatic, so schedule it when your workload has accumulated deletions.

<br/>

#### ☑️ Retrieve records by ID

Use `get_records()` to fetch one or more records by ID, with optional vector inclusion. It returns a list of dicts with `id`, `metadata`, and, when `return_vector` is true, `vector`.

```python
# Single record
print(index.get_records("doc_002", return_vector=False))

# Multiple records
print(index.get_records(["doc_002", "doc_003"], return_vector=False))

# Missing IDs are silently skipped
print(index.get_records(["doc_002", "missing_id"], return_vector=False))

# Vectors are included by default
record = index.get_records("doc_002")[0]
print(sorted(record.keys()), len(record["vector"]))
```

*Output*
```
[{'id': 'doc_002', 'metadata': {'author': 'Bob'}}]
[{'id': 'doc_002', 'metadata': {'author': 'Bob'}}, {'id': 'doc_003', 'metadata': {'author': 'Alice'}}]
[{'id': 'doc_002', 'metadata': {'author': 'Bob'}}]
['id', 'metadata', 'vector'] 8
```

⚠️ `get_records()` only returns results for IDs that exist in the index. Missing IDs are silently skipped, so a shorter list than you asked for is how a missing ID is reported.

<br/>

#### ☑️ Count and test membership

`len(index)` is the live record count. `id in index` tests membership. `count(filter)` counts the records a metadata filter matches, and `count()` with no filter is `len(index)`.

```python
print(len(index))
print("doc_002" in index, "doc_001" in index)
print(index.count())
print(index.count({"author": "Alice"}))
print(index.count({"author": "Nobody"}))
```
*Output*
```
4
True False
4
2
0
```

`count()` is exact and therefore reads every record's metadata, so it costs what a filtered search costs. `contains(id)` is the same test as `in` and is kept for callers already using it.

<br/>

#### ☑️ Change a record's metadata

`update_metadata(id, metadata)` replaces one record's metadata without resupplying its vector. The record keeps its vector, its quantized codes and its graph node, and no node is stranded.

```python
print(index.get_records("doc_002", return_vector=False)[0]["metadata"])
print(index.update_metadata("doc_002", {"author": "Bob", "status": "reviewed"}))
print(sorted(index.get_records("doc_002", return_vector=False)[0]["metadata"].items()))
print(index.update_metadata("no_such_id", {"author": "Nobody"}))
print(index.get_stats()["stranded_graph_nodes"])
```
*Output*
```
{'author': 'Bob'}
True
[('author', 'Bob'), ('status', 'reviewed')]
False
0
```

The example sorts the second result because a record's metadata comes back as a `dict` whose key order is not stable between processes. Read metadata by key rather than by position.

**The replacement is wholesale, not a merge.** Any key you leave out is gone, which is what `add(overwrite=True)` already does. It returns `False` for an ID the index does not hold, and writes nothing in that case.

Use this rather than reading a record back with `get_records()` and adding it again. Measured at 20,000 records the round trip costs 486.5 microseconds against 1.57 for this, and it strands one graph node per update.

<br/>

#### ☑️ Remove several records at once

`remove_points(ids)` takes the lock once for the whole batch instead of once per ID. It returns the IDs that were **not** in the index, so an empty list means every one was removed. A repeated ID is removed on its first occurrence and is never reported missing.

```python
print(index.remove_points(["doc_004", "no_such_id"]))
print(len(index))
```
*Output*
```
['no_such_id']
3
```

`remove_where(filter)` removes every record a metadata filter matches, using the same filter language as `search()`, and returns how many it removed.

```python
print(index.remove_where({"author": "Alice"}))
print(len(index), index.count({"author": "Alice"}))
print(index.remove_where({"author": "Nobody"}))
```
*Output*
```
2
1 0
0
```

An unrecognised operator raises `ValueError` before any record is removed. A filter matching nothing removes nothing and returns `0`.

**`remove_where({})` is refused.** An empty filter matches every record everywhere else in this language, and here that would destroy the index. Name the records with `remove_points(ids)` if that is what you want, or use `clear()`.

Both leave one stranded graph node per record removed, exactly as `remove_point()` does, and neither calls `compact()`.

<br/>

#### ☑️ `delete()`, the shorter name for both

`delete(ids=...)` dispatches to `remove_points` and `delete(where=...)` to `remove_where`. Both of those stay.

```python
deletable = vdb.create("hnsw", dim=2, expected_size=10)
deletable.add({
    "ids": ["doc_1", "doc_2", "doc_3", "doc_4"],
    "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
    "metadatas": [{"author": "Alice"}, {"author": "Bob"},
                  {"author": "Bob"}, {"author": "Alice"}],
})

print(deletable.delete(ids="doc_1"))
print(deletable.delete(ids=["doc_2", "no_such_id"]))
print(deletable.delete(where={"author": "Bob"}))
print(len(deletable))
```
*Output*
```
1
1
1
1
```

It returns **the number of records removed**, an `int`, whichever argument was given. `ids` takes a string or a list of strings. A repeated ID counts once. An ID that was not there counts zero rather than raising.

`remove_points` still returns the IDs it could not find, which is more than a count, so keep calling it where you need that.

**Passing both arguments raises, and so does passing neither.** Use `clear()` when emptying the index is what you mean.

<br/>

#### ☑️ Empty the index with `clear()`

`clear()` drops every record and returns how many went. It replaces the graph rather than removing records one at a time, so `stranded_graph_nodes` reads `0` afterwards.

```python
clearable = vdb.create("hnsw", dim=4, expected_size=10)
clearable.add({
    "ids": ["a", "b", "c", "d", "e"],
    "embeddings": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0],
                   [0, 0, 0, 1.0], [1.0, 1.0, 0, 0]],
})
clearable.remove_point("a")

print(clearable.clear())
print(len(clearable), clearable.get_stats()["stranded_graph_nodes"])
print(clearable.clear())
```
*Output*
```
4
0 0
0
```

**It keeps the index and drops the records.** `dim`, `space`, `m`, `ef_construction`, `expected_size`, the index level metadata and the quantization configuration all survive, and a fitted PQ codebook survives with them, so a trained quantized index can be refilled and searched without retraining. An index still collecting for training starts collecting again.

Clearing an empty index returns `0` and is not an error. The internal ID counter restarts, so the first generated ID after a clear is `vec_1` again.

<br/>

#### ♻️ Return the graph's spare capacity

An index built by inserting grows its graph buffers geometrically, so the last growth leaves the largest of them holding close to twice what they use. `shrink_to_fit()` returns that slack to the allocator and reports the bytes it released.

```python
fresh = vdb.create("hnsw", dim=8, expected_size=300)
fresh.add({
    "ids": [f"v{i}" for i in range(500)],
    "embeddings": [[float(i % 7) + j * 0.1 for j in range(8)] for i in range(500)],
})

before = float(fresh.get_stats()["graph_memory_mb"])
freed = fresh.shrink_to_fit()
after = float(fresh.get_stats()["graph_memory_mb"])
print(freed > 0, after < before)
print(fresh.shrink_to_fit())
```
*Output*
```
True True
0
```

The index above declared 300 records and was given 500, so its graph grew and left slack behind. `index`, the index used throughout this section, returns `0` instead, because `compact()` was called on it earlier and compaction already shrinks the graph it rebuilds.

**No node, edge or distance is touched**, so every search returns the same page with the same scores. Measured on 50,000 records of `dim=1536` at `m=32`, it released 188.28 MB in one millisecond and search latency did not move, so the slack was buying nothing.

**Call it on an index that holds its records, not on one about to receive them.** On an empty index it hands back the whole creation reservation that `expected_size` bought, so every later insertion regrows the arenas from nothing.

**The index stays writable.** The buffers grow again on the next `add()`, which costs one reallocation. That is why it is never automatic.

<br/>

#### ☑️ Quantization status and performance reporting

Five further accessors report state that `get_stats()` also carries.

| Method | Returns |
| --- | --- |
| `is_training_ready()` | Whether the training threshold has been reached. `False` on an index with no quantization configuration |
| `training_vectors_needed()` | Records still to collect before training triggers. `0` on an index with no quantization configuration |
| `rebuild_with_quantization()` | Rebuilds the graph against the quantized codes and returns whether it did. Training and `load()` already do this, so calling it is normally redundant |
| `get_performance_info()` | A `str` to `str` map describing the search and insertion paths |
| `benchmark_concurrent_reads(query_count, max_threads)` | Times sequential against threaded searches over random queries, returning `sequential_qps`, `parallel_qps`, `speedup`, `sequential_time`, `parallel_time` and `threads_used` |

```python
ready = vdb.create("hnsw", dim=8, expected_size=1200, quantization_config={
    "type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000,
})
print(ready.is_training_ready(), ready.training_vectors_needed())
print(sorted(ready.get_performance_info()))
```
*Output*
```
False 1000
['benefits', 'insertion_path', 'quantization_accuracy_impact', 'quantization_compression', 'search_bottleneck', 'search_speedup_expected']
```

<br />

## 🗜️ Product Quantization

Product Quantization (PQ) is a vector compression technique that reduces memory usage by dividing each vector into subvectors and quantizing them independently. A record's compressed form is one byte per subvector, whatever the dimension, so an index over 1536-dimensional vectors with 8 subvectors stores 8 bytes per code in place of 6144 bytes of float32.

ZeusDB Vector Database's PQ implementation features:

✅ Automatic training, triggered on the `add()` call that reaches the configured threshold

✅ Compact codes, one byte per subvector per record

✅ Asymmetric Distance Computation (ADC) for fast search against the codes

✅ Automatic switch from raw to quantized storage once training completes

Compression is not free, and the accuracy cost is much larger than the memory saving suggests. Read [Quantized search accuracy](#-quantized-search-accuracy) before choosing a storage mode.

<br />

### 📘 Quantization Configuration Parameters

To enable PQ, pass a `quantization_config` dictionary to the `.create()` index method:

| Parameter | Type | Description | Valid Range | Default |
|-----------|------|-------------|-------------|---------|
| `type` | `str` | Quantization algorithm type | `"pq"` | *required* |
| `subvectors` | `int` | Number of vector subspaces. Must divide `dim` evenly | 1 to `dim` | derived from `dim`, see below |
| `bits` | `int` | Bits per quantized code, which sets the centroids per subvector to 2^bits | 1 to 8 | `8` |
| `training_size` | `int` | Records collected before training is triggered | ≥ 1000 | `10000` |
| `max_training_vectors` | `int \| None` | Maximum records used during training | ≥ `training_size` | `None` |
| `storage_mode` | `str` | `"quantized_only"` or `"quantized_with_raw"` | see below | `"quantized_only"` |

**Compression ratio is `dim × 4 / subvectors`.** More subvectors means a longer code, so it lowers the compression ratio and raises accuracy. Fewer subvectors means the opposite. At `dim=1536`, 8 subvectors gives 768x and 16 subvectors gives 384x.

**`subvectors` defaults to `dim / 32`, clamped to between 8 and 192, snapped to a divisor of `dim`.** That holds the compression ratio at 128x, which is the quantity accuracy follows.

| `dim` | default `subvectors` | compression |
|-------|----------------------|-------------|
| 64 | 8 | 32x |
| 128 | 8 | 64x |
| 256 | 8 | 128x |
| 768 | 24 | 128x |
| 1536 | 48 | 128x |
| 3072 | 96 | 128x |

Going lower than 128x costs memory and build time and buys nothing on recall until the ratio reaches about 16x, where the fetch collapses and query time falls instead. See [Quantized search accuracy](#-quantized-search-accuracy). Below `dim=256` the floor of 8 subvectors binds. Pass `subvectors` explicitly for a cheaper, less accurate setting.

`bits` does not change the size of a record's code, which is always one byte per subvector, so lowering it saves no memory per record. It sets the number of centroids per subvector to 2^bits, which sizes the codebook and the centroid distance table. Lowering it costs recall and shortens the build. Leave it at 8 unless the fixed cost or the build time is the constraint.

`create()` emits a `UserWarning` when the configuration looks unbalanced, for example when the compression ratio exceeds 50x, and another when `storage_mode` is `quantized_with_raw`. The ratio warning does not fire on a `subvectors` the library derived, only on one you passed.

It also warns when the configuration cannot repay its fixed memory at the `expected_size` you declared, naming the record count above which quantization starts saving. Raise `expected_size` if your estimate was low, or drop `quantization_config`. A separate warning fires when `expected_size` is below `training_size`, because an index that never reaches its training threshold never trains.

<br/>

### 🔧 Usage Example 1

```python
from zeusdb_vector_database import VectorDatabase
import numpy as np

vdb = VectorDatabase()

quantization_config = {
    "type": "pq",                        # `pq` for Product Quantization
    "subvectors": 8,                     # 8 subvectors of 192 dims each
    "bits": 8,                           # 256 centroids per subvector (2^8)
    "training_size": 1000,               # Train once 1,000 records are collected
    "storage_mode": "quantized_with_raw" # Keep raw vectors so results can be reranked
}

index = vdb.create(
    index_type="hnsw",
    dim=1536,                                # OpenAI `text-embedding-3-small` dimension
    expected_size=2500,
    quantization_config=quantization_config
)

# Add vectors. Training triggers automatically at the threshold.
rng = np.random.default_rng(0)
documents = {
    "ids": [f"doc_{i}" for i in range(2500)],
    "embeddings": rng.random((2500, 1536), dtype=np.float32),
    "metadatas": [{"category": "tech", "year": 2026} for _ in range(2500)],
}

result = index.add(documents)
print("inserted:", result.total_inserted)

# Check quantization status
print("training progress:", f"{index.get_training_progress():.1f}%")
print("storage mode:", index.get_storage_mode())
print("is quantized:", index.is_quantized())

# Get compression statistics
quant_info = index.get_quantization_info()
print("compression ratio:", f"{quant_info['compression_ratio']:.1f}x")
print("codebook memory:", f"{quant_info['memory_mb']:.1f} MB")

# Search works the same way on a quantized index
query_vector = rng.random(1536, dtype=np.float32)
results = index.search(vector=query_vector, top_k=3)
print("results:", len(results), "| keys:", sorted(results[0].keys()))
```

*Output*
```
inserted: 2500
training progress: 100.0%
storage mode: quantized_active
is quantized: True
compression ratio: 768.0x
codebook memory: 1.5 MB
results: 3 | keys: ['id', 'metadata', 'score']
```

The result IDs and scores depend on the data, so they are not shown. Production indexes use a much larger `training_size`; 1,000 is the minimum the validator accepts and keeps this example quick.

`index.info()` reports the quantization state as well:

```python
print(index.info())
```

*Output*
```
HNSWIndex(dim=1536, space=cosine, m=16, ef_construction=200, expected_size=2500, vectors=2500, quantization=pq(subvectors=8, bits=8, trained, active, compression=768.0x))
```

<br />

### 🔧 Usage Example 2 - with explicit storage mode

```python
from zeusdb_vector_database import VectorDatabase

vdb = VectorDatabase()

quantization_config = {
    "type": "pq",
    "subvectors": 8,
    "bits": 8,
    "training_size": 10000,
    "max_training_vectors": 50000,
    "storage_mode": "quantized_only"    # Drop raw vectors once training completes
}

index = vdb.create(
    index_type="hnsw",
    dim=3072,                           # OpenAI `text-embedding-3-large` dimension
    expected_size=100000,
    quantization_config=quantization_config
)
```

<br/>

### 📦 Storage modes

| Mode | What it stores | Rerank available | Memory |
|------|----------------|------------------|--------|
| `quantized_only` | Codes for every record; the raw vectors collected for training are released when training completes | No | Lowest of the three |
| `quantized_with_raw` | Codes and raw vectors for every record | Yes | Between the two. Measured at 0.69x an unquantized index at 10,000 records of `dim=768` and 0.59x at 100,000, at the default `subvectors` |

Two consequences of `quantized_only` are worth knowing before you pick it.

**The training records are held at full width only until training completes.** Records collected before the threshold is reached are stored raw so the quantizer has something to train on. The moment training completes they are encoded and their raw copies are released, so a trained index in this mode holds no raw vector for any record.

**Once training completes every record exists only as a code, so the vector you read back is an approximation.** Every accessor sees every record. `get_records(..., return_vector=True)` and `search(..., return_vector=True)` reconstruct the vector from its code. Only `quantized_with_raw` reads back exactly, and `get_stats()["raw_vectors_stored"]` reading zero is how you confirm the release happened.

```python
only = vdb.create("hnsw", dim=1536, expected_size=2500, quantization_config={
    "type": "pq",
    "subvectors": 8,
    "bits": 8,
    "training_size": 1000,
    "storage_mode": "quantized_only",
})
only.add(documents)   # the same 2,500 records used in Usage Example 1

print("storage mode:", only.get_storage_mode())
stats = only.get_stats()
print("raw vectors kept:", stats["raw_vectors_stored"])
print("quantized codes:", stats["quantized_codes_stored"])
print("records:", only.get_vector_count())
print("contains doc_0 (added before training):", only.contains("doc_0"))
print("contains doc_2000 (added after training):", only.contains("doc_2000"))
print("get_records doc_2000 returns:", len(only.get_records("doc_2000")), "record")
```

*Output*
```
storage mode: quantized_active
raw vectors kept: 0
quantized codes: 2500
records: 2500
contains doc_0 (added before training): True
contains doc_2000 (added after training): True
get_records doc_2000 returns: 1 record
```

**How much quantization saves is set by the dimension, and below `dim=256` it is not much.** Measured resident, one dimension per process, 25,000 records at `m=32`, an unquantized index and a `quantized_with_raw` one built over the same records:

| dim | unquantized | `quantized_with_raw` | ratio | saving |
|---:|---:|---:|---:|---:|
| 64 | 75.7 MiB | 73.4 MiB | 0.97x | 3% |
| 96 | 83.4 | 73.8 | 0.88x | 12% |
| 128 | 89.7 | 75.0 | 0.84x | 16% |
| 192 | 102.6 | 82.7 | 0.81x | 19% |
| 256 | 115.5 | 88.2 | 0.76x | 24% |
| 384 | 140.5 | 105.8 | 0.75x | 25% |
| 768 | 216.1 | 151.5 | 0.70x | 30% |
| 1,536 | 369.6 | 236.0 | 0.64x | 36% |

`create()` warns when the saving falls below a fifth of what an unquantized index holds, which is `dim=235` for `quantized_with_raw` and `dim=88` for `quantized_only`. It also warns below the record count where quantization repays its fixed tables, which is a few thousand records at `dim=256` and under two thousand at `dim=768`. Below that, quantization costs memory rather than saving it.

<br/>

### 🎯 Quantized search accuracy

**Quantized search is far less accurate than raw search, and `quantized_only` cannot be repaired by tuning.** ADC scores candidates against the codes, and a code discards most of the information in a vector. Rerank fixes this by over-fetching candidates and rescoring them against raw vectors, which is only possible when the raw vectors are still there.

Measured on 6,000 clustered 128-dimensional vectors with 8 subvectors and 8 bits, recall at 10 against exact cosine search:

| Configuration | Recall@10 |
|---------------|-----------|
| No quantization | 1.00 |
| `quantized_only` | 0.16 |
| `quantized_with_raw`, `rerank=0` | 0.15 |
| `quantized_with_raw`, default rerank | 1.00 |

The exact figures depend on your data, but the shape does not. If you need quantization and you need accuracy, use `quantized_with_raw` and leave rerank on.

**How deep a search has to fetch to hold that recall depends on your data, not on the record count.** Measured on three real datasets at 100,000 records with the default `subvectors`, the fetch that reaches mean recall at 10 of 0.99:

| dataset | dim | compression | fetch for 0.99 | share of corpus |
|---|---:|---:|---:|---:|
| dbpedia-openai (ada-002) | 1,536 | 128x | 494 | 0.49% |
| sift-128 | 128 | 64x | 426 | 0.43% |
| glove-100 | 100 | 40x | 5,143 | 5.14% |

No formula in the record count fits those three, so ZeusDB measures the fetch on your data instead. A `quantized_with_raw` index measures it when training completes, and scales what it measured with the record count and with the page size you ask for. `get_stats()["rerank_default_fetch"]` reports what a search at `top_k=10` will fetch on the index as it stands.

**On data with no resolvable structure no fetch works.** Once the group the codes cannot separate is smaller than `top_k`, the true top ten span groups and nothing reaches them. Measure recall on your own data before you rely on quantization.

What `rerank` does:

| `rerank` | Effect |
| --- | --- |
| omitted | Uses the calibrated fetch. It is the only setting that holds recall across corpus sizes and across datasets |
| `N` of 1 or more | Fetches `top_k × N` candidates, a fixed multiple of the page that does not move with the corpus. Use it to override the default deliberately |
| `0` | Turns reranking off and returns the ADC scores and ordering |

A page below ten fetches what a page of ten fetches, so pass `rerank` explicitly if you want a shallower page to cost less. `rerank` has no effect on an unquantized index or on a `quantized_only` one, and both ignore it. With rerank on the scores you get back are raw-vector distances, and with it off they are ADC estimates, which are not comparable.

An index trained before the calibration existed, and any index loaded from a directory saved by one, carries no calibration and falls back to a fixed fetch of 2% of the record count. `get_stats()["rerank_calibrated"]` reads `false` for it. Rebuild the index to calibrate it.

**Above roughly 10,000 records a reranked quantized search is slower than an unquantized one, and the gap widens as the index grows.** That is the price of the default holding recall. On dbpedia-openai at `dim=1,536`, paired against an unquantized index over the same records, 200 queries one each in turn:

| records | calibrated fetch | unquantized | quantized, default rerank | ratio |
|---:|---:|---:|---:|---:|
| 10,000 | 277 | 0.75 ms | 0.71 ms | 0.95 |
| 25,000 | 411 | 0.79 ms | 0.97 ms | 1.23 |
| 50,000 | 554 | 1.17 ms | 1.54 ms | 1.32 |
| 100,000 | 747 | 1.18 ms | 2.12 ms | 1.79 |

Each row is one process building both indexes over the same records, so the ratio is the figure to read. Quantization remains a memory decision. At 100,000 records of `dim=768` the default holds 585 MB against 994 MB unquantized and answers in 13.4 ms against 3.05 ms. Lower `rerank` explicitly if query time matters more to you than recall, and measure what it costs you.

**`ef_search` does nothing on a reranked quantized search.** The graph traversal widens to the number of candidates the fetch asks for, which is already wider than any `ef_search` a caller is likely to set, and setting it smaller is discarded. Change `rerank` instead. On an unquantized search, and on a quantized search with `rerank=0`, `ef_search` applies normally.

### 📊 Performance Characteristics

- **Training**: happens once, on the `add()` call that reaches `training_size`. That call takes noticeably longer than the others. On `quantized_with_raw` it also calibrates the rerank fetch, which `get_stats()["rerank_calibration_ms"]` prices.
- **Memory**: a record's code is `subvectors` bytes against `dim × 4` for a raw vector, and the graph holds a second copy of every point that shrinks by the same factor. How much that saves is set by the dimension, and the table in Storage modes prices it from `dim=64` to `dim=1536`.
- **Search speed**: an unreranked quantized search is faster than a raw search. A reranked one is slower above roughly 10,000 records, and the table above prices it.
- **Build speed**: a quantized build is faster than an unquantized one, and it slows as `subvectors` rises. At 100,000 records of `dim=768` it is 137 s against 231 s at the default `subvectors`.
- **Accuracy**: see the tables above. Treat quantization as a memory decision that costs accuracy and query time, not as a free win.

<br/>

## 💾 Persistence

ZeusDB Vector Database can save and restore complete indexes on disk, which lets you preserve your work, move indexes between systems, and back up production deployments.

The persistence system supports:

✅ **Complete state preservation** for vectors, per-record metadata, index level metadata, ID mappings and quantization models
✅ **Hybrid storage format**, binary encoding for vectors with human-readable JSON for metadata
✅ **Quantization support**, both raw and quantized storage modes, including the trained codebook
✅ **Training state recovery**, so an index saved mid-collection resumes collecting
✅ **Format versioning**, so a directory this build cannot interpret is refused rather than misread

**`save()` and `load()` print progress to stdout.** Every step writes a line. This is not configurable, so redirect stdout if it is a problem in your application.

<br/>

### 💾 Saving an Index - .save()

Use the `.save()` method to persist your index to a `.zdb` directory:

```python
from zeusdb_vector_database import VectorDatabase
import numpy as np
import os

vdb = VectorDatabase()
index = vdb.create("hnsw", dim=1536, space="cosine", expected_size=1000)

rng = np.random.default_rng(1)
vectors = rng.random((1000, 1536), dtype=np.float32)
index.add({
    "ids": [f"doc_{i}" for i in range(1000)],
    "embeddings": vectors,
    "metadatas": [{"category": f"cat_{i % 5}", "index": i} for i in range(1000)],
})

index.save("my_index.zdb")
print("saved:", sorted(os.listdir("my_index.zdb")))
```

*Output, with the progress lines omitted*
```
saved: ['config.json', 'hnsw_index.zdbgraph', 'manifest.json', 'mappings.bin', 'metadata.json', 'vectors.bin']
```

<br />

### 📂 Loading an Index - .load()

Use the `.load()` method to restore a previously saved index:

```python
vdb = VectorDatabase()
loaded_index = vdb.load("my_index.zdb")

print("vectors:", loaded_index.get_vector_count())
print(loaded_index.info())

results = loaded_index.search(vectors[0].tolist(), top_k=3)
print("top hit:", results[0]["id"])
```

*Output, with the progress lines omitted*
```
vectors: 1000
HNSWIndex(dim=1536, space=cosine, m=16, ef_construction=200, expected_size=1000, vectors=1000, quantization=none)
top hit: doc_0
```

**Loading reads the saved graph back rather than rebuilding it**, so a reloaded index returns the same result pages as the index that was saved, with the same IDs and the same scores. Load time is proportional to the size of the directory rather than to the cost of building the index: 50,000 records at 1,536 dimensions load in 1.1 seconds against a 156 second build.

The graph is rebuilt by re-inserting every record only when the saved graph cannot be used, which covers a directory whose graph files were lost or damaged and one written by a release too old for this build to interpret. Set `ZEUSDB_LOAD_REBUILD_GRAPH=1` to ask for that rebuild on a directory whose graph is perfectly readable, which is how an index built by an earlier release picks up graph improvements made since.

<br />

### 🗜️ Persistence with Product Quantization

A quantized index comes back quantized, with its codebook and training state intact:

```python
quantization_config = {
    "type": "pq",
    "subvectors": 8,
    "bits": 8,
    "training_size": 1000,
    "storage_mode": "quantized_with_raw",
}

vdb = VectorDatabase()
index = vdb.create("hnsw", dim=1536, expected_size=2000,
                   quantization_config=quantization_config)

rng = np.random.default_rng(2)
index.add({
    "ids": [f"vec_{i}" for i in range(2000)],
    "embeddings": rng.random((2000, 1536), dtype=np.float32),
})

print("quantization active:", index.is_quantized())
index.save("quantized_index.zdb")

loaded_index = vdb.load("quantized_index.zdb")
print("quantization active after load:", loaded_index.is_quantized())
print("storage mode after load:", loaded_index.get_storage_mode())
print("saved:", sorted(os.listdir("quantized_index.zdb")))
```

*Output, with the progress lines omitted*
```
quantization active: True
quantization active after load: True
storage mode after load: quantized_active
saved: ['config.json', 'hnsw_index.zdbgraph', 'manifest.json', 'mappings.bin', 'metadata.json', 'pq_centroids.bin', 'pq_codes.bin', 'quantization.json', 'vectors.bin']
```

<br/>

### 📁 Index Directory Structure
The `.save()` method creates a directory containing all index components:

```
my_index.zdb/
├── manifest.json           # Index metadata and file inventory
├── config.json             # HNSW configuration and index level metadata
├── mappings.bin            # ID mappings (binary format)
├── metadata.json           # Per-record metadata (JSON format)
├── vectors.bin             # Raw vectors (whenever the index holds any)
├── quantization.json       # PQ configuration (if enabled)
├── pq_centroids.bin        # Trained centroids (if PQ trained)
├── pq_codes.bin            # Quantized codes (if PQ active)
└── hnsw_index.zdbgraph     # HNSW graph structure and payload
```

`manifest.json` lists every file the save wrote under `files_included`, and it is written after all of them except the graph. It is the inventory of what the directory should hold.

A directory saved by 0.6.0 or earlier holds `hnsw_index.hnsw.graph` and `hnsw_index.hnsw.data` in place of `hnsw_index.zdbgraph`. Opening it still works: the graph is rebuilt once from the stored records, and the next `.save()` writes the single file.

**`load()` refuses a directory that does not hold what its manifest names.** It checks `files_included` before it reads anything, and the graph dump is the one exempt artefact, because every record carries what the graph is built from. A directory missing any other file will not open, and the refusal names the file and says what it held. Restore it from a copy; the missing file cannot be rebuilt from the ones that remain. A file the manifest does not name is neither read nor complained about.

A file that is present and does not parse is a different failure with a different message, of the form `Failed to parse config.json` or `Failed to deserialize mappings.bin`.

<br/>

### 🔄 Complete Save/Load Workflow
A full persistence lifecycle with integrity checks:

```python
from zeusdb_vector_database import VectorDatabase
import numpy as np

# === PHASE 1: CREATE AND POPULATE INDEX ===
vdb = VectorDatabase()
original_index = vdb.create("hnsw", dim=1536, space="cosine", expected_size=500)

rng = np.random.default_rng(42)
vectors = rng.random((500, 1536), dtype=np.float32)

original_index.add({
    "ids": [f"doc_{i:03d}" for i in range(500)],
    "embeddings": vectors,
    "metadatas": [
        {
            "category": ["science", "tech", "health", "finance"][i % 4],
            "priority": i % 10,
            "published": i % 2 == 0,
            "tags": ["important", "featured"] if i % 5 == 0 else ["standard"],
        }
        for i in range(500)
    ],
})

original_index.add_metadata({
    "dataset": "demo_collection",
    "created_by": "data_team",
    "version": "1.0",
})

query_vector = vectors[0].tolist()
original_results = original_index.search(query_vector, top_k=3)

# === PHASE 2: SAVE, THEN LOAD ===
original_index.save("demo_index.zdb")
loaded_index = vdb.load("demo_index.zdb")

# === PHASE 3: VERIFY INTEGRITY ===
assert loaded_index.get_vector_count() == original_index.get_vector_count()
assert loaded_index.info() == original_index.info()
assert loaded_index.get_all_metadata() == original_index.get_all_metadata()

loaded_results = loaded_index.search(query_vector, top_k=3)
assert [r["id"] for r in loaded_results] == [r["id"] for r in original_results]

filtered = loaded_index.search(
    query_vector,
    filter={"category": "science", "published": True},
    top_k=20,
)

print("records:", loaded_index.get_vector_count())
print("index metadata fields:", len(loaded_index.get_all_metadata()))
print("filtered hits:", len(filtered))
print("all checks passed")
```

*Output, with the progress lines omitted*
```
records: 500
index metadata fields: 3
filtered hits: 20
all checks passed
```

### ⚠️ Important Notes on Persistence

- **Directory, not a file.** `.save()` creates a directory. You need write permission for the target location.

- **Not atomic.** Files are written one at a time into the target directory. An interrupted save leaves a partial directory behind, and a later `load()` of it fails rather than returning a truncated index. Save to a new path and move it into place if you need an atomic swap.

- **Overwriting is not clean either.** Saving over an existing directory replaces files individually and does not remove ones that no longer apply. The leftovers are inert but still occupy the disk. Save to a fresh directory.

- **Version compatibility.** The manifest records a format version. This build writes 1.1.0 and reads any 1.x. A different major version is refused.

- **Integrity checks on load.** Three run, in this order: the format version, then `files_included` against the directory, then the restored record count against the count in `config.json`.

<br />

## 🏷️ Metadata Filtering

ZeusDB supports rich metadata with full type fidelity. Your metadata preserves the original Python data types, so integers stay integers and floats stay floats.

### 📘 Supported Types

| Type | Python Example | Notes |
|------|----------------|-------|
| **String** | `"Alice"` | Text data, IDs, categories |
| **Integer** | `42`, `2024` | Counts, years, IDs |
| **Float** | `4.5`, `29.99` | Ratings, prices, scores |
| **Boolean** | `True`, `False` | Flags, status indicators |
| **Null** | `None` | Missing or empty values |
| **Array** | `["ai", "science"]` | Tags, categories, lists |
| **Nested Object** | `{"key": "value"}` | Structured data |

Integers and floats compare by magnitude, so a stored integer `10` matches `{"eq": 10.0}` and `{"gte": 10.0}` alike. Booleans and strings do not cross into numbers.

<br/>

### 📘 Filter Operators Reference

A filter is a dict whose keys are field names, and all of them must hold. A field maps either to a plain value, which means equality, or to a dict of operators, all of which must hold. Three reserved keys, `$and`, `$or` and `$not`, compose whole filters rather than naming a field; see [Boolean composition](#-boolean-composition).

| Operator | Usage | Example | Description |
|----------|-------|---------|-------------|
| **Direct equality** | `{"field": value}` | `{"author": "Alice"}` | Equality for strings, numbers, booleans, null and arrays |
| `eq` | `{"eq": value}` | `{"source": {"eq": {"kind": "web"}}}` | Equality, including for nested objects |
| `ne` | `{"ne": value}` | `{"author": {"ne": "Alice"}}` | Not equal |
| `gt` | `{"gt": value}` | `{"rating": {"gt": 4.0}}` | Greater than (numeric) |
| `gte` | `{"gte": value}` | `{"year": {"gte": 2024}}` | Greater than or equal (numeric) |
| `lt` | `{"lt": value}` | `{"price": {"lt": 30}}` | Less than (numeric) |
| `lte` | `{"lte": value}` | `{"pages": {"lte": 100}}` | Less than or equal (numeric) |
| `contains` | `{"contains": value}` | `{"tags": {"contains": "ai"}}` | String contains substring, or array contains value |
| `startswith` | `{"startswith": value}` | `{"title": {"startswith": "The"}}` | String starts with substring |
| `endswith` | `{"endswith": value}` | `{"file": {"endswith": ".pdf"}}` | String ends with substring |
| `in` | `{"in": [values]}` | `{"lang": {"in": ["en", "es"]}}` | Value is in the provided array |
| `nin` | `{"nin": [values]}` | `{"lang": {"nin": ["en", "es"]}}` | Value is not in the provided array |
| `any` | `{"any": [values]}` | `{"tags": {"any": ["ai", "ml"]}}` | Array field shares at least one element with the provided array |
| `all` | `{"all": [values]}` | `{"tags": {"all": ["ai", "ml"]}}` | Array field holds every element of the provided array |

`any` and `all` exist because a field maps to one condition object, so it cannot carry `contains` twice. They ask their question of one field's array, where `$or` and `$and` compose whole filters across fields. On a field holding a plain value rather than an array, both read it as an array of one.

Three behaviours are worth knowing.

**A record that lacks the field never matches, whatever the operator.** That includes `ne` and `nin`. `{"lang": {"ne": "en"}}` and `{"lang": {"nin": ["en"]}}` do not match a record with no `lang` at all, and they agree because `nin` against a one-element array means what `ne` means. So an operator never selects a record missing a field, and `$not` is what does: `{"$not": {"lang": {"all": []}}}` selects exactly the records with no `lang`, because `{"all": []}` is the empty conjunction and holds for every value the field can carry.

**A dict value is always read as operators.** Direct equality against a nested object has no plain form, because the two would be indistinguishable, so write it as `{"source": {"eq": {"kind": "web"}}}`. Writing `{"source": {"kind": "web"}}` raises `ValueError: Unknown filter operation: kind`.

**An unrecognised operator raises `ValueError` before the search runs**, rather than quietly matching nothing.

<br/>

### 🧩 Boolean composition

A filter is a conjunction of its keys. Three reserved keys compose whole filters instead of naming a field.

| Key | Takes | Means |
|-----|-------|-------|
| `$and` | a list of filters | every one of them holds |
| `$or` | a list of filters | at least one of them holds |
| `$not` | one filter | that filter does not hold |

**Precedence.** There is none to remember, because the structure is explicit. A mapping is an AND of everything in it, fields and groups alike, so `{"a": 1, "$or": [...]}` means `a == 1` AND the disjunction. A group's branches are each a whole filter, so a branch carrying two fields conjoins them.

**Nesting.** Groups nest to 10 levels, counting the filter itself as level one. A filter deeper than that raises `ValueError`.

**Reserved keys.** Exactly `$and`, `$or` and `$not`. The `$` prefix is not reserved, so a field named `$price` still filters. A field literally named `$or`, `$and` or `$not` cannot be filtered on, and a filter naming it raises.

**The empty cases.** `{"$and": []}` matches every record and `{"$or": []}` matches none, which is what `all` and `any` already do with an empty array.

```python
from zeusdb_vector_database import VectorDatabase

vdb = VectorDatabase()
index = vdb.create("hnsw", dim=4, space="l2")
index.add([
    {"id": "d1", "values": [0.1, 0.1, 0.1, 0.1],
     "metadata": {"lang": "en", "tier": "gold", "year": 2024}},
    {"id": "d2", "values": [0.2, 0.2, 0.2, 0.2],
     "metadata": {"lang": "es", "tier": "free", "year": 2023}},
    {"id": "d3", "values": [0.3, 0.3, 0.3, 0.3],
     "metadata": {"lang": "fr", "tier": "gold", "year": 2026}},
    {"id": "d4", "values": [0.4, 0.4, 0.4, 0.4],
     "metadata": {"tier": "free", "year": 2025}},
])
q = [0.1, 0.1, 0.1, 0.1]


def matched(filter):
    return sorted(hit["id"] for hit in index.search(vector=q, filter=filter, top_k=10))


# Either language. A flat filter cannot ask this, because one field maps to one
# condition and two conditions on it are conjoined.
print(matched({"$or": [{"lang": "en"}, {"lang": "es"}]}))

# Gold tier, and either recent or English. A branch is a whole filter, so the
# second one carries two fields and conjoins them.
print(matched({"tier": "gold",
               "$or": [{"year": {"gte": 2026}}, {"lang": "en"}]}))

# Not free tier. This is what `ne` does here, since every record has a tier.
print(matched({"$not": {"tier": "free"}}))

# Records with no lang field at all, which no operator can select.
print(matched({"$not": {"lang": {"all": []}}}))

# None of these, which is Qdrant's must_not over a group.
print(matched({"$not": {"$or": [{"lang": "es"}, {"tier": "gold"}]}}))
```

*Output*
```
['d1', 'd2']
['d1', 'd3']
['d1', 'd3']
['d4']
['d4']
```

<br/>

### ⏱️ What a filtered search costs

A filtered search costs a great deal more than an unfiltered one. There is no index on metadata fields, so finding out which records match means reading every record's metadata.

Two paths serve a filter and the index chooses between them per search. At or below 5,000 matching records it walks the whole metadata store, scores every record that matched and ranks them, which is exact. Above 5,000 the graph traversal runs instead with the filter tested at every node it reaches, and recall there is the graph's own, measured at 0.96 and above on three real 100,000 record sets.

Measured on three real 100,000 record sets, milliseconds per query, minimum of several passes:

| Records matched | Path | sift, 128d | glove, 100d | dbpedia, 1536d |
| --- | --- | --- | --- | --- |
| no filter | graph | 0.30 | 0.29 | 1.16 |
| 50,000 | graph | 3.2 | 3.7 | 5.0 |
| 10,000 | graph | 18.0 | 27.2 | 31.6 |
| 1,000 | exact | 33.1 | 39.4 | 38.4 |
| 100 | exact | 36.1 | 30.8 | 30.8 |
| 1 | exact | 38.3 | 31.6 | 34.7 |

**Expect a filtered search over 100,000 records to cost tens of milliseconds where an unfiltered one costs a fraction of one**, and expect it to grow in proportion to the record count. Filtering on a field that few records carry does not reduce it, because the walk visits every record either way. Filtering less often does, and so does keeping separate indexes for partitions you always filter by.

<br/>

### 💡 Practical Filter Examples

The examples below all run against this index:

```python
from zeusdb_vector_database import VectorDatabase

vdb = VectorDatabase()
index = vdb.create("hnsw", dim=4, space="l2")
index.add([
    {"id": "doc_1", "values": [0.1, 0.1, 0.1, 0.1], "metadata": {
        "author": "Alice", "rating": 4.5, "year": 2024, "price": 29.99,
        "published": True, "tags": ["ai", "science"], "title": "The Guide",
        "filename": "report.pdf", "lang": "en"}},
    {"id": "doc_2", "values": [0.2, 0.2, 0.2, 0.2], "metadata": {
        "author": "Bob", "rating": 3.0, "year": 2023, "price": 45.00,
        "published": False, "tags": ["cooking"], "title": "A Book",
        "filename": "notes.txt", "lang": "es"}},
    {"id": "doc_3", "values": [0.3, 0.3, 0.3, 0.3], "metadata": {
        "author": "Charlie", "rating": 5.0, "year": 2026, "price": 25.00,
        "published": True, "tags": ["ai"], "title": "Theory",
        "filename": "paper.pdf", "lang": "fr"}},
])
query_embedding = [0.1, 0.1, 0.1, 0.1]
```

#### ✔️ The filter chooses what is ranked, so `top_k` is just the page size

```python
def matched(filter, top_k=10):
    return [hit["id"] for hit in index.search(vector=query_embedding, filter=filter, top_k=top_k)]

# doc_3 is the furthest of the three from the query, and it is still the only
# thing the filter admits, so it is what a page of one holds
print(matched({"author": "Charlie"}, top_k=1))
print(matched({"author": "Charlie"}, top_k=10))
```

*Output*
```
['doc_3']
['doc_3']
```

A filter matching fewer records than `top_k` returns that many results, and one matching none returns an empty list. Neither is a truncation.

#### ✔️ Common filters

```python
# Find high-quality recent documents
print(matched({"published": True, "rating": {"gte": 4.0}, "year": {"gte": 2024}}))

# Find documents by specific authors
print(matched({"author": {"in": ["Alice", "Bob"]}}))

# Find AI-related content
print(matched({"tags": {"contains": "ai"}}))

# Find documents in a price range
print(matched({"price": {"gte": 20.0, "lte": 40.0}}))

# Find documents with a specific file type
print(matched({"filename": {"endswith": ".pdf"}}))

# Match on a title prefix
print(matched({"title": {"startswith": "The"}}))

# Exclude an author
print(matched({"author": {"ne": "Alice"}}))

# Match a whole array
print(matched({"tags": ["ai"]}))

# Either a top rating or a recent year, which needs a disjunction
print(matched({"$or": [{"rating": {"gte": 5.0}}, {"year": {"gte": 2024}}]}))

# Published, and either English or cheap
print(matched({"published": True,
               "$or": [{"lang": "en"}, {"price": {"lt": 26.0}}]}))

# Everything except Bob's, including any record with no author at all
print(matched({"$not": {"author": "Bob"}}))
```

*Output*
```
['doc_1', 'doc_3']
['doc_1', 'doc_2']
['doc_1', 'doc_3']
['doc_1', 'doc_3']
['doc_1', 'doc_3']
['doc_1', 'doc_3']
['doc_2', 'doc_3']
['doc_3']
['doc_1', 'doc_3']
['doc_1', 'doc_3']
['doc_1', 'doc_3']
```

<br />

## 📝 Logging

ZeusDB Vector Database includes structured logging that works automatically out of the box while providing customization for advanced users.

### 🚀 Basic Usage - it just works!

**For most users, logging works automatically with sensible defaults:**

<!-- zeusdb:skip -->
```python
from zeusdb_vector_database import VectorDatabase
# Logging is automatically configured, no setup required

vdb = VectorDatabase()
index = vdb.create("hnsw", dim=1536)

# Operations are automatically logged with structured data
result = index.add({"ids": ids, "embeddings": vectors})
results = index.search(query_vector, top_k=5)
```

**What you get automatically:**
- ✅ **Quiet by default**, only warnings and errors outside development
- ✅ **Environment detection**, appropriate defaults for dev, prod, testing, CI and notebooks
- ✅ **Structured JSON logs** in production environments
- ✅ **Human-readable logs** in development environments
- ✅ **Operation timing** on index creation, additions, searches and saves
- ✅ **Cross-platform compatibility**

Note that `save()` and `load()` print progress directly to stdout. That output is not part of the logging system and is not affected by any of the settings below.

### ⚙️ Intermediate Usage (Environment Variables)

**Control logging behavior with environment variables:**

#### Quick Development Debugging
```bash
export ZEUSDB_LOG_LEVEL=debug
python your_app.py
```

#### Production JSON Logging
```bash
export ZEUSDB_LOG_LEVEL=error
export ZEUSDB_LOG_FORMAT=json
export ZEUSDB_LOG_TARGET=file
export ZEUSDB_LOG_FILE=/var/log/zeusdb/app.log
python your_app.py
```

#### Environment Variables Reference

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `ZEUSDB_LOG_LEVEL` | `trace`, `debug`, `info`, `error` | `warning` (dev), `error` (prod) | Controls log verbosity |
| `ZEUSDB_LOG_FORMAT` | `human`, `json` | `human` (dev), `json` (prod) | Output format |
| `ZEUSDB_LOG_TARGET` | `stdout`, `stderr`, `file` | `stderr` | Where logs go |
| `ZEUSDB_LOG_FILE` | `/path/to/file.log` | `zeusdb.log` | Log file path, written exactly as given (if target=file) |
| `ZEUSDB_LOG_ROTATION` | `daily`, `never` | `never` | With `daily`, a UTC date is appended to the file name |
| `ZEUSDB_LOG_CONSOLE` | `true`, `false` | Auto-detected | Force console output |
| `ZEUSDB_DISABLE_AUTO_LOGGING` | `true`, `1`, `yes` | unset | Skip automatic configuration entirely |
| `RUST_LOG` | standard `env_logger` syntax | unset | Overrides `ZEUSDB_LOG_LEVEL` for the Rust layer |

**⚠️ `warning` and `critical` are not accepted level names.** The Python layer accepts them, but the Rust layer rejects them and prints `ignoring 'zeusdb_vector_database=warning': invalid filter directive`. The bare `warn` is the opposite, accepted by Rust and rejected by Python. Use `trace`, `debug`, `info` or `error`, which both layers accept.

Under `ZEUSDB_LOG_ROTATION=daily` with `ZEUSDB_LOG_FILE=logs/app.log`, two files appear: `logs/app.log` and a dated `logs/app.log.2026-08-05`. Rotation applies to the Rust layer, which writes the dated one.

#### Smart Environment Detection
The system detects your environment and applies appropriate defaults:

- **🏭 Production** (`ENVIRONMENT=production`, or Kubernetes or Docker markers): ERROR level, JSON format, file output
- **💻 Development** (default): WARNING level, human format, console output
- **🧪 Testing** (`ENVIRONMENT=testing`, `PYTEST_CURRENT_TEST`, or `pytest` imported): CRITICAL level, minimal output
- **📓 Jupyter** (`JUPYTER_SERVER_ROOT`, `JPY_PARENT_PID`, or IPython imported): INFO level, human format
- **🔄 CI/CD** (`CI`, `GITHUB_ACTIONS`, `GITLAB_CI`): WARNING level, human format for readability

Environment variables always override the detected defaults.

### 🔧 Advanced Usage (Programmatic Control)

**For enterprise environments with existing logging infrastructure:**

#### Option 1: Disable Auto-Configuration
<!-- zeusdb:skip -->
```python
import os
os.environ["ZEUSDB_DISABLE_AUTO_LOGGING"] = "1"

# Now configure your own logging before importing ZeusDB
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from zeusdb_vector_database import VectorDatabase  # Will respect your existing logging setup
```

#### Option 2: Programmatic Initialization
<!-- zeusdb:skip -->
```python
import os
os.environ["ZEUSDB_DISABLE_AUTO_LOGGING"] = "1"

import zeusdb_vector_database

# JSON to stdout
success = zeusdb_vector_database.init_logging(level="info")

# OR JSON to a directory of daily rotating files. Pick one, not both.
# success = zeusdb_vector_database.init_file_logging(
#     log_dir="/var/log/myapp",
#     level="debug",
#     file_prefix="zeusdb"
# )

print("initialized:", success)

vdb = zeusdb_vector_database.VectorDatabase()
```

**Only the first initializer to run takes effect.** Both functions return `True` if they installed the subscriber and `False` if one was already installed, so calling both leaves the second with no effect and a `False` return. `zeusdb_vector_database.is_logging_initialized()` reports whether either has run.

#### Option 3: Custom Logger Integration
<!-- zeusdb:skip -->
```python
import logging
import os

# Disable auto-configuration
os.environ["ZEUSDB_DISABLE_AUTO_LOGGING"] = "1"

# Set up your own logger first
logger = logging.getLogger("myapp.zeusdb")
logger.setLevel(logging.INFO)

# Configure Rust logging to match
os.environ["ZEUSDB_LOG_LEVEL"] = "info"
os.environ["ZEUSDB_LOG_FORMAT"] = "json"

from zeusdb_vector_database import VectorDatabase
# ZeusDB will integrate with your logging setup
```

### 📊 Log Output Examples

#### Human-Readable (Development)
```
2026-08-05T12:19:39.261318Z  INFO build: HNSW index created successfully operation="index_creation_complete" dim=8 space=cosine m=16 ef_construction=200 expected_size=10000 has_quantization=false duration_ms=0
2026-08-05T12:19:39.3491294Z  INFO add: Vector addition completed operation="add_vectors_complete" total_inserted=2 total_errors=0 success_rate=100.0 duration_ms=87 overwrite_mode=true final_storage_mode="raw_only"
```

#### Structured JSON (Production)
```json
{"timestamp":"2026-08-05T12:19:39.4853862Z","level":"INFO","fields":{"message":"HNSW index created successfully","operation":"index_creation_complete","dim":8,"space":"cosine","m":16,"ef_construction":200,"expected_size":10000,"has_quantization":false,"duration_ms":"0"},"target":"zeusdb_vector_database::hnsw_index","filename":"src\\hnsw_index.rs","line_number":1068,"threadId":"ThreadId(1)"}
```

### 🔍 Monitoring and Observability

#### Key Fields to Monitor
- **`operation`**: the operation name, for example `index_creation_complete`, `add_vectors_complete`, `search_complete`, `pq_training_complete`, `save_complete`, `compact_complete`
- **`duration_ms`**: timing on index creation, additions, searches, saves and compaction
- **`total_inserted`**, **`total_errors`**, **`success_rate`**: outcome of each `add()`
- **`final_storage_mode`**: whether an index is serving raw or quantized results
- **`results_count`**: results returned by a search

#### Production Alerting Examples
```bash
# Monitor error rates
grep '"level":"ERROR"' /var/log/zeusdb/app.log | wc -l

# Track search latency
grep '"operation":"search_complete"' /var/log/zeusdb/app.log | jq '.fields.duration_ms'

# Watch quantization training
grep '"operation":"pq_training' /var/log/zeusdb/app.log
```

### 🛠️ Troubleshooting

#### Common Issues

**Logs not appearing?**
```bash
# Check if auto-logging is disabled
echo $ZEUSDB_DISABLE_AUTO_LOGGING

# Verify the level is one both layers accept
ZEUSDB_LOG_LEVEL=debug python -c "import zeusdb_vector_database as z; print(z.is_logging_initialized())"
```

**File logging not working?**
```bash
# Check permissions
ls -la /path/to/log/directory

# Test with console first
ZEUSDB_LOG_TARGET=stderr ZEUSDB_LOG_LEVEL=info python your_app.py
```

**Want to see Rust logs specifically?**
```bash
# Enable trace level to see all Rust operations
ZEUSDB_LOG_LEVEL=trace python your_app.py
```

#### Performance Notes
- File logging is non-blocking: records are handed to a background writer rather than written on the calling thread.
- `trace` and `debug` are verbose enough to dominate runtime on a hot loop. Leave production at `error`.

### 🎯 Best Practices

#### Development
```bash
export ZEUSDB_LOG_LEVEL=debug
export ZEUSDB_LOG_FORMAT=human
```

#### Staging
```bash
export ZEUSDB_LOG_LEVEL=info
export ZEUSDB_LOG_FORMAT=json
export ZEUSDB_LOG_TARGET=file
export ZEUSDB_LOG_FILE=logs/zeusdb-staging.log
export ZEUSDB_LOG_ROTATION=daily
```

#### Production
```bash
export ENVIRONMENT=production
export ZEUSDB_LOG_LEVEL=error
export ZEUSDB_LOG_FORMAT=json
export ZEUSDB_LOG_TARGET=file
export ZEUSDB_LOG_FILE=/var/log/zeusdb/production.log
export ZEUSDB_LOG_ROTATION=daily
```

<br/>

## 📄 License

This project is licensed under the Apache License 2.0.
