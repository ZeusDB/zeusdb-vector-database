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
        <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-blue?logo=python&logoColor=ffdd54"></a>&nbsp;
        <a href="https://github.com/zeusdb/zeusdb-vector-database/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>&nbsp;
      </td>
    </tr>
  </table>
</div>

<!-- badges: end -->

<br/>

## What is ZeusDB Vector Database?

ZeusDB Vector Database is a high-performance, Rust-powered vector database designed for blazing-fast similarity search across high-dimensional data. It enables efficient approximate nearest neighbor (ANN) search, ideal for use cases like document retrieval, semantic search, recommendation systems, and AI-powered assistants. 

ZeusDB leverages the HNSW (Hierarchical Navigable Small World) algorithm for speed and accuracy, with native Python bindings for easy integration into data science and machine learning workflows. Whether you're indexing millions of vectors or running low-latency queries in production, ZeusDB offers a lightweight, extensible foundation for scalable vector search.

<br/>

## Features

🔍 Approximate Nearest Neighbor (ANN) search with HNSW

<!-- 🧠 Supports multiple distance metrics: `cosine`, `l2`, `dot` -->

🔥 High-performance Rust backend 

📥 Supports multiple input formats using a single, easy-to-use Python method

🗂️ Metadata-aware filtering at query time

🐍 Simple and intuitive Python API




<br/>

## ✅ Supported Distance Metrics

| Metric | Description                          |
|--------|--------------------------------------|
| cosine | Cosine Distance (1 - Cosine Similiarity) |
<!--
| l2     | Euclidean distance                   |
| dot    | Dot product                 |

-->

Scores vs Distances: 
- Similarity Scores (higher = more similar)
- Distances (lower = more similar)

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
index = vdb.create_index_hnsw(dim = 8, space = "cosine", M = 16, ef_construction = 200, expected_size=5)

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
print("\n--- Add Results Summary ---")
print(add_result.summary())

# Perform a similarity search and print the top 2 results
# Query Vector
query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

# Query with no filter (all documents)
results = index.query(vector=query_vector, filter=None, top_k=2)
print("\n--- Query Results Output - Raw ---")
print(results)

print("\n--- Query Results Output - Formatted ---")
for i, res in enumerate(results, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")
```

*Results Output:*
```
--- Add Results Summary ---
✅ 5 inserted, ❌ 0 errors

--- Raw Results Format ---
[{'id': 'doc_001', 'score': 0.0, 'metadata': {'author': 'Alice'}}, {'id': 'doc_003', 'score': 0.0009883458260446787, 'metadata': {'author': 'Alice'}}]

--- Formatted Results ---
1. ID: doc_001, Score: 0.0000, Metadata: {'author': 'Alice'}
2. ID: doc_003, Score: 0.0010, Metadata: {'author': 'Alice'}
```

<br/>

## ✨ Usage

ZeusDB makes it easy to work with high-dimensional vector data using a fast, memory-efficient HNSW index. Whether you're building semantic search, recommendation engines, or embedding-based clustering, the workflow is simple and intuitive:

Three simple steps:

  1. Create an index

  2. Add data to the index

  3. Conduct a similarity search

Each step is covered below.

### 1️⃣ Create an Index

To get started, first initialize a VectorDatabase and create an HNSWIndex. You can configure the vector dimension, distance metric, and graph construction parameters.

```python
# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Initialize and set up the database resources
index = vdb.create_index_hnsw(
  dim = 8, 
  space = "cosine", 
  M = 16, 
  ef_construction = 200, 
  expected_size=5
  )
```

#### 📘 `create_index_hnsw()` Parameters

| Parameter        | Type   | Default   | Description                                                                 |
|------------------|--------|-----------|-----------------------------------------------------------------------------|
| `dim`            | `int`  | `1536`    | Dimensionality of the vectors to be indexed. Each vector must have this length. The default dim=1536 is chosen to match the output dimensionality of OpenAI’s text-embedding-ada-002 model. |
| `space`          | `str`  | `"cosine"`| Distance metric used for similarity search. Options include `"cosine"`. Additional metrics such as `"l2"`, and `"dot"` will be added in future versions. |
| `M`              | `int`  | `16`      | Number of bi-directional connections created for each new node. Higher `M` improves recall but increases index size and build time. |
| `ef_construction`| `int`  | `200`     | Size of the dynamic list used during index construction. Larger values increase indexing time and memory, but improve quality. |
| `expected_size`  | `int`  | `10000`   | Estimated number of elements to be inserted. Used for preallocating internal data structures. Not a hard limit. |

<br/>


### 2️⃣ Add Data to the Index

ZeusDB provides a flexible `.add(...)` method that supports multiple input formats for inserting vectors into the index. Whether you're adding a single record, a list of documents, or structured arrays, the API is designed to be both intuitive and robust. Each record can include optional metadata for filtering or downstream use.

All formats return an AddResult containing total_inserted, total_errors, and detailed error messages for any invalid entries.

#### ✅ Format 1 – Single Object

```python
add_result = index.add({
    "id": "doc1",
    "values": [0.1, 0.2],
    "metadata": {"text": "hello"}
})

print(add_result.summary())     # ✅ 1 inserted, ❌ 0 errors
print(add_result.is_success())  # True
```

#### ✅ Format 2 – List of Objects

```python
add_result = index.add([
    {"id": "doc1", "values": [0.1, 0.2], "metadata": {"text": "hello"}},
    {"id": "doc2", "values": [0.3, 0.4], "metadata": {"text": "world"}}
])

print(add_result.summary())       # ✅ 2 inserted, ❌ 0 errors
print(add_result.vector_shape)    # (2, 2)
print(add_result.errors)          # []
```

#### ✅ Format 3 – Separate Arrays

```python
add_result = index.add({
    "ids": ["doc1", "doc2"],
    "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    "metadatas": [{"text": "hello"}, {"text": "world"}]
})
print(add_result)  # AddResult(inserted=2, errors=0, shape=(2, 2))
```

#### ✅ Format 4 – Using NumPy Arrays

ZeusDB also supports NumPy arrays as input for seamless integration with scientific and ML workflows.

```python
import numpy as np

data = [
    {"id": "doc2", "values": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), "metadata": {"type": "blog"}},
    {"id": "doc3", "values": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32), "metadata": {"type": "news"}},
]

result = index.add(data)

print(result.summary())   # ✅ 2 inserted, ❌ 0 errors
```

#### ✅ Format 5 – Separate Arrays with NumPy

This format is highly performant and leverages NumPy's internal memory layout for efficient transfer of data.

```python
add_result = index.add({
    "ids": ["doc1", "doc2"],
    "embeddings": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    "metadatas": [{"text": "hello"}, {"text": "world"}]
})
print(add_result)  # AddResult(inserted=2, errors=0, shape=(2, 2))
```

Each format is automatically parsed and validated internally, including support for NumPy arrays (np.ndarray). Errors and successes are returned in a structured AddResult object for easy debugging and logging.

#### 📘 `add()` Parameters

The `add()` method inserts one or more vectors into the index. Multiple data formats are supported to accommodate different workflows, including native Python types and NumPy arrays.

| Parameter | Type                                | Default | Description |
|-----------|-------------------------------------|---------|-------------|
| `data`    | `dict`, `list[dict]`, or `dict of arrays` | *required* | Input records to be added to the index. Supports multiple formats |

**Returns:**  
[`AddResult`](#) – an object with `total_inserted`, `total_errors`, `errors`, and `vector_shape`. Helpful for validation, logging, and debugging.

<br/>

### 3️⃣ Conduct a Similarity Search

Query the index using a new vector and retrieve the top-k nearest neighbors. You can also filter by metadata or return the original stored vectors.

#### 🔍 Basic Search (Returning Top 2 most similar)

```python
print("\n--- Query returning two most similar results ---")
results = index.query(vector=query_vector, top_k=2)
print(results)
```

*Output*
```
[
  {'id': 'doc_37', 'score': 0.016932480037212372, 'metadata': {'index': '37', 'split': 'test'}}, 
  {'id': 'doc_33', 'score': 0.019877362996339798, 'metadata': {'split': 'test', 'index': '33'}}
]
```

#### 🔍 Query with metadata filter

This filters on the given metadata after conducting the similarity search.

```python
print("\n--- Querying with filter: author = 'Alice' ---")
results = index.query(vector=query_vector, filter={"author": "Alice"}, top_k=5)
print(results)
```

*Output*
```
[
  {'id': 'doc_001', 'score': 0.0, 'metadata': {'author': 'Alice'}}, 
  {'id': 'doc_003', 'score': 0.0009883458260446787, 'metadata': {'author': 'Alice'}}, 
  {'id': 'doc_005', 'score': 0.0011433829786255956, 'metadata': {'author': 'Alice'}}
]
```

#### 🔍 Include Vector in Similarity Results

You can optionally return the stored embedding vectors alongside metadata and similarity scores by setting `return_vector=True`. This is useful when you need access to the raw vectors for downstream tasks such as re-ranking, inspection, or hybrid scoring.

```python
print("\n--- Querying with filter and returning embedding vectors ---")
results = index.query(vector=query_vector, filter={"split": "test"}, top_k=2, return_vector=True)
print(results)
```

*Output*
```
[
  {'id': 'doc_37', 'score': 0.016932480037212372, 'metadata': {'index': '37', 'split': 'test'}, 'vector': [0.36544516682624817, 0.11984539777040482, 0.7143614292144775, 0.8995016813278198]}, 
  {'id': 'doc_33', 'score': 0.019877362996339798, 'metadata': {'split': 'test', 'index': '33'}, 'vector': [0.8367619514465332, 0.6394991874694824, 0.9291712641716003, 0.9777664542198181]}
]
```

#### 📘 `query()` Parameters

The `query()` method retrieves the top-k most similar vectors from the index given an input query vector. Results include the vector ID, similarity score, metadata, and (optionally) the stored vector itself.

| Parameter         | Type                            | Default   | Description                                                                 |
|------------------|----------------------------------|-----------|-----------------------------------------------------------------------------|
| `vector`         | `List[float]` or `np.ndarray`    | *required* | The query vector to compare against the index. Must match the index dimension. |
| `filter`         | `Dict[str, str] \| None`         | `None`    | Optional metadata filter. Only vectors with matching key-value metadata pairs will be considered in the search. |
| `top_k`          | `int`                            | `10`      | Number of nearest neighbors to return. |
| `ef_search`      | `int \| None`                    | `max(2 × top_k, 100)` | Search complexity parameter. Higher values improve accuracy at the cost of speed. |
| `return_vector`  | `bool`                           | `False`   | If `True`, the result objects will include the original embedding vector. Useful for downstream processing like re-ranking or hybrid search. |

<br/>

### 🧰 Additional functionality

#### Check the details of your HNSW index 

```python
print(index.info()) 
```
*Output*
```
HNSWIndex(dim=8, space=cosine, M=16, ef_construction=200, expected_size=5, vectors=5)
```

<br/>


#### Add index level metadata

```python
index.add_metadata({
  "creator": "John Smith",
  "version": "0.1",
  "created_at": "2024-01-28T11:35:55Z",
  "index_type": "HNSW",
  "embedding_model": "openai/text-embedding-ada-002",
  "dataset": "docs_corpus_v2",
  "environment": "production",
  "description": "Knowledge base index for customer support articles",
  "num_documents": "15000",
  "tags": "['support', 'docs', '2024']"
})

# View index level metadata by key
print(index.get_metadata("creator"))  

# View all index level metadata 
print(index.get_all_metadata())       
```
*Output*
```
John Smith
{'description': 'Knowledge base index for customer support articles', 'environment': 'production', 'embedding_model': 'openai/text-embedding-ada-002', 'creator': 'John Smith', 'tags': "['support', 'docs', '2024']", 'num_documents': '15000', 'version': '0.1', 'index_type': 'HNSW', 'dataset': 'docs_corpus_v2', 'created_at': '2024-01-28T11:35:55Z'}
```

<br/>


#### List records in the index

```python
print("\n--- Index Shows first 5 records ---")
print(index.list(number=5)) # Shows first 5 records
```
*Output*
```
[('doc_004', {'author': 'Bob'}), ('doc_003', {'author': 'Alice'}), ('doc_005', {'author': 'Alice'}), ('doc_002', {'author': 'Bob'}), ('doc_001', {'author': 'Alice'})]
```

<br/>


## 📄 License

This project is licensed under the Apache License 2.0.
