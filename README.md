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
        <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/Powered%20by-Rust-black?logo=rust&logoColor=white" alt="Powered by Rust"></a>&nbsp;
        <a href="https://github.com/ZeusDB"><img src="https://github.com/user-attachments/assets/e140d900-1160-4eaa-85c0-2b3507a5f0f5" alt="ZeusDB"></a>&nbsp;
      </td>
    </tr>
  </table>
</div>

<!-- badges: end -->

<br/>

## ℹ️ What is ZeusDB Vector Database?

ZeusDB Vector Database is a high-performance, Rust-powered vector database designed for blazing-fast similarity search across high-dimensional data. It enables efficient approximate nearest neighbor (ANN) search, ideal for use cases like document retrieval, semantic search, recommendation systems, and AI-powered assistants. 

ZeusDB leverages the HNSW (Hierarchical Navigable Small World) algorithm for speed and accuracy, with native Python bindings for easy integration into data science and machine learning workflows. Whether you're indexing millions of vectors or running low-latency queries in production, ZeusDB offers a lightweight, extensible foundation for scalable vector search.

<br/>

## ⭐ Features

🐍 User-friendly Python API for adding vectors and running similarity searches

🔥 High-performance Rust backend optimized for speed and concurrency

🔍 Approximate Nearest Neighbor (ANN) search using HNSW for fast, accurate results

📦 Product Quantization (PQ) for compact storage, faster distance computations, and scalability for Big Data

📥 Flexible input formats, including native Python types and zero-copy NumPy arrays

🗂️ Metadata-aware filtering for precise and contextual querying




<!-- 
📋 Supports multiple distance metrics: `cosine`, `L1`, `L2` 

📥 Supports multiple input formats using a single, easy-to-use Python method

⚡ Smart multi-threaded inserts that automatically speed up large batch uploads

🚀 Fast, concurrent searches so you can run multiple queries at the same time
-->

<br/>

## ✅ Supported Distance Metrics

ZeusDB Vector Database supports the following metrics for vector similarity search. All metric names are case-insensitive, so "cosine", "COSINE", and "Cosine" are treated identically.

| Metric | Description                          | Accepted Values (case-insensitive)  |
|--------|--------------------------------------|--------|
| cosine | Cosine Distance (1 - Cosine Similiarity) | "cosine", "COSINE", "Cosine" |
| l1     | Manhattan distance                   | "l1", "L1" |
| l2     | Euclidean distance                 | "l2", "L2" |


### 📏 Scores vs Distances 

All distance metrics in ZeusDB Vector Database return distance values, not similarity scores:

 - Lower values = more similar
 - A score of 0.0 means a perfect match

This applies to all distance types, including cosine.



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
print("\n--- Add Results Summary ---")
print(add_result.summary())

# Perform a similarity search and print the top 2 results
# Query Vector
query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

# Query with no filter (all documents)
results = index.search(vector=query_vector, filter=None, top_k=2)
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
  index_type = "hnsw",
  dim = 8, 
  space = "cosine", 
  m = 16, 
  ef_construction = 200, 
  expected_size = 5
  )
```

<br/>

#### 📘 Parameters - `create()`

| Parameter        | Type   | Default   | Description                                                                 |
|------------------|--------|-----------|-----------------------------------------------------------------------------|
| `index_type`     | `str`  | `"hnsw"`  | The type of vector index to create. Currently supports `"hnsw"`. Future options include `"ivf"`, `"flat"`, etc. Case-insensitive. |
| `dim`            | `int`  | `1536`    | Dimensionality of the vectors to be indexed. Each vector must have this length. The default dim=1536 is chosen to match the output dimensionality of OpenAI’s text-embedding-ada-002 model. |
| `space`          | `str`  | `"cosine"`| Distance metric used for similarity search. Options include `"cosine"`, `"L1"` and `"L2"`.|
| `m`              | `int`  | `16`      | Number of bi-directional connections created for each new node. Higher `m` improves recall but increases index size and build time. |
| `ef_construction`| `int`  | `200`     | Size of the dynamic list used during index construction. Larger values increase indexing time and memory, but improve quality. |
| `expected_size`  | `int`  | `10000`   | Estimated number of elements to be inserted. Used for preallocating internal data structures. Not a hard limit. |
| `quantization_config` | `dict` | `None` | Product Quantization configuration for memory-efficient vector compression. |

<br/>


### 2️⃣ Add Data to the Index

ZeusDB provides a flexible `.add(...)` method that supports multiple input formats for inserting or updating vectors in the index. Whether you're adding a single record, a list of documents, or structured arrays, the API is designed to be both intuitive and robust. Each record can include optional metadata for filtering or downstream use.

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

Each format is parsed and validated automatically. Invalid records are skipped, and detailed error messages are returned to help with debugging and retry workflows.

<br/>

#### 📘 Parameters - `add()`

The `add()` method inserts one or more vectors into the index. Multiple data formats are supported to accommodate different workflows, including native Python types and NumPy arrays.

| Parameter | Type                                | Default | Description |
|-----------|-------------------------------------|---------|-------------|
| `data`    | `dict`, `list[dict]`, or `dict of arrays` | *required* | Input records to upsert into the index. Supports multiple formats |

**Returns:**  
`AddResult` includes: – 
- `total_success`: number of vectors successfully inserted or updated
- `total_errors`: number of failed records
- `errors`: list of error messages
- `vector_shape`: the shape of the processed vector batch

Helpful for validation, logging, and debugging.

<br/>

### 3️⃣ Conduct a Similarity Search

Query the index using a new vector and retrieve the top-k nearest neighbors. You can also filter by metadata or return the original stored vectors.

#### 🔍 Search Example 1 - Basic (Returning Top 2 most similar)

```python
results = index.search(vector=query_vector, top_k=2)
print(results)
```

*Output*
```
[
  {'id': 'doc_37', 'score': 0.016932480037212372, 'metadata': {'index': '37', 'split': 'test'}}, 
  {'id': 'doc_33', 'score': 0.019877362996339798, 'metadata': {'split': 'test', 'index': '33'}}
]
```

#### 🔍 Search Example 2 - Query with metadata filter

This filters on the given metadata after conducting the similarity search.

```python
query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]
results = index.search(vector=query_vector, filter={"author": "Alice"}, top_k=5)
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

#### 🔍 Search Example 3 - Search results include vectors

You can optionally return the stored embedding vectors alongside metadata and similarity scores by setting `return_vector=True`. This is useful when you need access to the raw vectors for downstream tasks such as re-ranking, inspection, or hybrid scoring.

```python
results = index.search(vector=query_vector, filter={"split": "test"}, top_k=2, return_vector=True)
print(results)
```

*Output*
```
[
  {'id': 'doc_37', 'score': 0.016932480037212372, 'metadata': {'index': '37', 'split': 'test'}, 'vector': [0.36544516682624817, 0.11984539777040482, 0.7143614292144775, 0.8995016813278198]}, 
  {'id': 'doc_33', 'score': 0.019877362996339798, 'metadata': {'split': 'test', 'index': '33'}, 'vector': [0.8367619514465332, 0.6394991874694824, 0.9291712641716003, 0.9777664542198181]}
]
```

#### 🔍 Search Example 4 - Batch Search with a list of vectors

Perform a similarity search on multiple query vectors simultaneously, returning results for each query.

```python
query_vector =
[
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
]
results = index.search(vector=query_vector, top_k=3)
print(results)
```

*Output*
```
[
[{'id': 'a', 'score': 4.999447078546382e-09, 'metadata': {'category': 'A'}}, {'id': 'b', 'score': 0.02536815218627453, 'metadata': {'category': 'B'}}, {'id': 'c', 'score': 0.04058804363012314, 'metadata': {'category': 'A'}}],
[{'id': 'b', 'score': 4.591760305316939e-09, 'metadata': {'category': 'B'}}, {'id': 'c', 'score': 0.0018091063247993588, 'metadata': {'category': 'A'}}, {'id': 'a', 'score': 0.025368161499500275, 'metadata': {'category': 'A'}}]
]
```

#### 🔍 Search Example 5 - Batch Search with NumPy Array

Perform a similarity search on multiple query vectors from a NumPy array, returning results for each query.

```python
query_vector = np.array(
[
    [0.1, 0.2, 0.3],
    [0.7, 0.8, 0.9]
], dtype=np.float32)

results = index.search(vector=query_vector, top_k=3)
print(results)
```

#### 🔍 Search Example 6 - Batch Search with metadata filter

Performs similarity search on multiple query vectors with metadata filtering, returning filtered results for each query.

```python
results = index.search(
    [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
    filter={"category": "A"},
    top_k=3
)
print(results)
```


<br/>

#### 📘 Parameters - `search()` 

The `search()` method retrieves the top-k most similar vectors from the index given an input query vector. Results include the vector ID, similarity score, metadata, and (optionally) the stored vector itself.

| Parameter         | Type                            | Default   | Description                                                                 |
|------------------|----------------------------------|-----------|-----------------------------------------------------------------------------|
| `vector`         | `List[float]` or `List[List[float]]` or `np.ndarray`  | *required* | The query vector (single: `List[float]`) or batch of query vectors (`List[List[float]]` or 2D `np.ndarray`) to compare against the index. Must match the index dimension. |
| `filter`         | `Dict[str, str] \| None`         | `None`    | Optional metadata filter. Only vectors with matching key-value metadata pairs will be considered in the search. |
| `top_k`          | `int`                            | `10`      | Number of nearest neighbors to return. |
| `ef_search`      | `int \| None`                    | `max(2 × top_k, 100)` | Search complexity parameter. Higher values improve accuracy at the cost of speed. |
| `return_vector`  | `bool`                           | `False`   | If `True`, the result objects will include the original embedding vector. Useful for downstream processing like re-ranking or hybrid search. |

<br/>

### 🧰 Additional functionality

ZeusDB Vector Database includes a suite of utility functions to help you inspect, manage, and maintain your index. You can view index configuration, attach custom metadata, list stored records, and remove vectors by ID. These tools make it easy to monitor and evolve your index over time,  whether you are experimenting locally or deploying in production.

#### ☑️ Check the details of your HNSW index 

```python
print(index.info()) 
```
*Output*
```
HNSWIndex(dim=8, space=cosine, m=16, ef_construction=200, expected_size=5, vectors=5)
```

<br/>


#### ☑️ Add index level metadata

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


#### ☑️ List records in the index

```python
print("\n--- Index Shows first 5 records ---")
print(index.list(number=5)) # Shows first 5 records
```
*Output*
```
[('doc_004', {'author': 'Bob'}), ('doc_003', {'author': 'Alice'}), ('doc_005', {'author': 'Alice'}), ('doc_002', {'author': 'Bob'}), ('doc_001', {'author': 'Alice'})]
```

<br/>

#### ☑️ Remove Records 

ZeusDB allows you to remove a vector and its associated metadata from the index using the .remove_point(id) method. This performs a <u>logical deletion</u>, meaning:
- The vector is deleted from internal storage.
- The metadata is removed.
- The vector ID is no longer accessible via .contains(), .get_vector(), or .search().

```python
# Remove the point using its ID
index.remove_point("doc1")  # "doc1" is the unique vector ID

print("\n--- Check Removal ---")
exists = index.contains("doc1")
print(f"Point 'doc1' {'found' if exists else 'not found'} in index")
```
*Output*
```
--- Check Removal ---
Point 'doc1' not found in index
```

**⚠️ Please Note:** Due to the nature of HNSW, the underlying graph node remains in memory, even after removing a point. This is common for HNSW implementations. To fully remove stale graph entries, consider rebuilding the index.

<br/>

#### ☑️ Retrieve records by ID

Use `get_records()` to fetch one or more records by ID, with optional vector inclusion.

```python
# Single record
print("\n--- Get Single Record ---")
rec = index.get_records("doc1")
print(rec)

# Multiple records
print("\n--- Get Multiple Records ---")
batch = index.get_records(["doc1", "doc3"])
print(batch)

# Metadata only
print("\n--- Get Metadata only ---")
meta_only = index.get_records(["doc1", "doc2"], return_vector=False)
print(meta_only)

# Missing ID silently ignored
print("\n--- Partial only ---")
partial = index.get_records(["doc1", "missing_id"])
print(partial)
```

⚠️ `get_records()` only returns results for IDs that exist in the index. Missing IDs are silently skipped.

<br />


## 🗜️ Product Quantization

Product Quantization (PQ) is a vector compression technique that significantly reduces memory usage while preserving high search accuracy. Commonly used in HNSW-based vector databases, PQ works by dividing each vector into subvectors and quantizing them independently. This enables compression ratios of 4× to 256×, making it ideal for large-scale, high-dimensional datasets.

ZeusDB Vector Database’s PQ implementation features:

✅ Intelligent Training – PQ model trains automatically at defined thresholds

✅ Efficient Memory Use – Store 4× to 256× more vectors in the same RAM footprint

✅ Fast Approximate Search – Uses Asymmetric Distance Computation (ADC) for high-speed search computation

✅ Seamless Operation – Index automatically switches from raw to quantized storage modes

<br />

### 📘 Quantization Configuration Parameters

To enable PQ, pass a `quantization_config` dictionary to the `.create()` index method:

| Parameter | Type | Description | Valid Range | Default |
|-----------|------|-------------|-------------|---------|
| `type` | `str` | Quantization algorithm type | `"pq"` | *required* |
| `subvectors` | `int` | Number of vector subspaces (must divide dimension evenly) | 1 to dimension | `8` |
| `bits` | `int` | Bits per quantized code (controls centroids per subvector) | 1-8 | `8` |
| `training_size` | `int` | Minimum vectors needed for stable k-means clustering | ≥ 1000 | 1000 |
| `max_training_vectors` | `int` | Maximum vectors used during training (optional limit) | ≥ training_size | `None` |
| `storage_mode` | `str` | Storage strategy: "quantized_only" (memory optimized) or "quantized_with_raw" (keep raw vectors for exact reconstruction) | "quantized_only", "quantized_with_raw" | `"quantized_only"` |


<br/>


### 🔧 Usage Example 1

```python
from zeusdb_vector_database import VectorDatabase
import numpy as np

# Create index with product quantization
vdb = VectorDatabase()

# Configure quantization for memory efficiency
quantization_config = {
    'type': 'pq',                  # `pq` for Product Quantization
    'subvectors': 8,               # Divide 1536-dim vectors into 8 subvectors of 192 dims each
    'bits': 8,                     # 256 centroids per subvector (2^8)
    'training_size': 10000,        # Train when 10k vectors are collected
    'max_training_vectors': 50000  # Use max 50k vectors for training
}

# Create index with quantization
# This will automatically handle training when enough vectors are added
index = vdb.create(
    index_type="hnsw",
    dim=1536,                                  # OpenAI `text-embedding-3-small` dimension
    quantization_config=quantization_config    # Add the compression configuration
)

# Add vectors - training triggers automatically at threshold
documents = [
    {
        "id": f"doc_{i}", 
        "values": np.random.rand(1536).astype(float).tolist(),
        "metadata": {"category": "tech", "year": 2026}
    }
    for i in range(15000) 
]

# Training will trigger automatically when 10k vectors are added
result = index.add(documents)
print(f"Added {result.total_inserted} vectors")

# Check quantization status
print(f"Training progress: {index.get_training_progress():.1f}%")
print(f"Storage mode: {index.get_storage_mode()}")
print(f"Is quantized: {index.is_quantized()}")

# Get compression statistics
quant_info = index.get_quantization_info()
if quant_info:
    print(f"Compression ratio: {quant_info['compression_ratio']:.1f}x")
    print(f"Memory usage: {quant_info['memory_mb']:.1f} MB")

# Search works seamlessly with quantized storage
query_vector = np.random.rand(1536).astype(float).tolist()
results = index.search(vector=query_vector, top_k=3)

# Simply print raw results
print(results)
```

Results
```python
[
{'id': 'doc_9719', 'score': 0.5133496522903442, 'metadata': {'category': 'tech', 'year': 2026}},
{'id': 'doc_8148', 'score': 0.5139288306236267, 'metadata': {'category': 'tech', 'year': 2026}}, 
{'id': 'doc_7822', 'score': 0.5151920914649963, 'metadata': {'category': 'tech', 'year': 2026}}, 
]
```
<br />

### 🔧 Usage Example 2 - with explicit storage mode

```python
from zeusdb_vector_database import VectorDatabase
import numpy as np

# Create index with product quantization
vdb = VectorDatabase()

# Configure quantization for memory efficiency
quantization_config = {
    'type': 'pq',                  # `pq` for Product Quantization
    'subvectors': 8,               # Divide 1536-dim vectors into 8 subvectors of 192 dims each
    'bits': 8,                     # 256 centroids per subvector (2^8)
    'training_size': 10000,        # Train when 10k vectors are collected
    'max_training_vectors': 50000,  # Use max 50k vectors for training
    'storage_mode': 'quantized_only'  # Explicitly set storage mode to only keep quantized values
}

# Create index with quantization
# This will automatically handle training when enough vectors are added
index = vdb.create(
    index_type="hnsw",
    dim=3072,                                  # OpenAI `text-embedding-3-large` dimension
    quantization_config=quantization_config    # Add the compression configuration
)

```

<br />

### ⚙️ Configuration Guidelines

For Balanced Memory & Accuracy (Recommended to start with)
```python
quantization_config = {
    'type': 'pq',
    'subvectors': 8,      # Balanced: moderate compression, good accuracy
    'bits': 8,            # 256 centroids per subvector (high precision)
    'training_size': 10000,  # Or higher for large datasets
    'storage_mode': 'quantized_only'  # Default, memory efficient
}
# Achieves ~16x–32x compression with strong recall for most applications
```


For Memory Optimization:
```python
quantization_config = {
    'type': 'pq',
    'subvectors': 16,      # More subvectors = better compression
    'bits': 6,             # Fewer bits = less memory per centroid
    'training_size': 20000,
    'storage_mode': 'quantized_only'
}
# Achieves ~32x compression ratio
```

For Accuracy Optimization:
```python
quantization_config = {
    'type': 'pq',
    'subvectors': 4,       # Fewer subvectors = better accuracy
    'bits': 8,             # More bits = more precise quantization
    'training_size': 50000 # More training data = better centroids
    'storage_mode': 'quantized_with_raw'  # Keep raw vectors for exact recall
}
# Achieves ~4x compression ratio with minimal accuracy loss
```

### 📊 Performance Characteristics

- Training: Occurs once when threshold is reached (typically 1-5 minutes for 50k vectors)
- Memory Reduction: 4x-256x depending on configuration
- Search Speed: Comparable or faster than raw vectors due to ADC optimization
- Accuracy Impact: Typically 1-5% recall reduction with proper tuning

Quantization is ideal for production deployments with large vector datasets (100k+ vectors) where memory efficiency is critical.

`"quantized_only"` is recommended for most use cases and maximizes memory savings.

`"quantized_with_raw"` keeps both quantized and raw vectors for exact reconstruction, but uses more memory.


<br/>

## 🏷️ Metadata Filtering

ZeusDB supports rich metadata with full type fidelity. This means your metadata preserves the original Python data types (integers stay integers, floats stay floats, etc.) and enables powerful filtering capabilities.

### 📘 Supported Types

The following Python types are supported for metadata and preserved during filtering and retrieval.

| Type | Python Example | Stored As | Notes |
|------|----------------|-----------|-------|
| **String** | `"Alice"` | `Value::String` | Text data, IDs, categories |
| **Integer** | `42`, `2024` | `Value::Number` | Counts, years, IDs |
| **Float** | `4.5`, `29.99` | `Value::Number` | Ratings, prices, scores |
| **Boolean** | `True`, `False` | `Value::Bool` | Flags, status indicators |
| **Null** | `None` | `Value::Null` | Missing/empty values |
| **Array** | `["ai", "science"]` | `Value::Array` | Tags, categories, lists |
| **Nested Object** | `{"key": "value"}` | `Value::Object` | Structured data |

<br/>

### 📘 Filter Operators Reference

These operators can be used in metadata filters:

| Operator | Usage | Example | Description |
|----------|-------|---------|-------------|
| **Direct equality** | `{"field": value}` | `{"author": "Alice"}` | Exact equality for any type |
| `gt` | `{"gt": value}` | `{"rating": {"gt": 4.0}}` | Greater than (numeric) |
| `gte` | `{"gte": value}` | `{"year": {"gte": 2024}}` | Greater than or equal (numeric) |
| `lt` | `{"lt": value}` | `{"price": {"lt": 30}}` | Less than (numeric) |
| `lte` | `{"lte": value}` | `{"pages": {"lte": 100}}` | Less than or equal (numeric) |
| `contains` | `{"contains": value}` | `{"tags": {"contains": "ai"}}` | String contains substring or array contains value |
| `startswith` | `{"startswith": value}` | `{"title": {"startswith": "The"}}` | String starts with substring |
| `endswith` | `{"endswith": value}` | `{"file": {"endswith": ".pdf"}}` | String ends with substring |
| `in` | `{"in": [values]}` | `{"lang": {"in": ["en", "es"]}}` | Value is in the provided array |

<br/>

### 💡 Practical Filter Examples

Below are common real-world examples of how to apply metadata filters using ZeusDB's metadata filtering:

#### ✔️ Find high-quality recent documents
```python
filter = {
    "published": True,
    "rating": {"gte": 4.0},
    "year": {"gte": 2024}
}

results = index.search(vector=query_embedding, filter=filter, top_k=5)
```

#### ✔️ Find documents by specific authors
```python
filter = {"author": {"in": ["Alice", "Bob", "Charlie"]}}
results = index.search(vector=query_embedding, filter=filter, top_k=5)
```

#### ✔️ Find AI-related content
```python
filter = {"tags": {"contains": "ai"}}
results = index.search(vector=query_embedding, filter=filter, top_k=5)
```

#### ✔️ Find documents in price range
```python
filter = {"price": {"gte": 20.0, "lte": 40.0}}
results = index.search(vector=query_embedding, filter=filter, top_k=5)
```

#### ✔️ Find documents with specific file types
```python
filter = {"filename": {"endswith": ".pdf"}}
results = index.search(vector=query_embedding, filter=filter, top_k=5)
```


<br/>

## 📄 License

This project is licensed under the Apache License 2.0.
