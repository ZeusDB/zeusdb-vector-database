# ZeusDB Vector Database

**Fast, Rust-powered vector database for similarity search**  

<!-- badges: start -->

<div align="left">
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

## What is ZeusDB Vectr Database

ZeusDB Vector Database is a high-performance, Rust-powered vector database designed for blazing-fast similarity search across high-dimensional data. It enables efficient approximate nearest neighbor (ANN) search, ideal for use cases like document retrieval, semantic search, recommendation systems, and AI-powered assistants. 

ZeusDB leverages the HNSW (Hierarchical Navigable Small World) algorithm for speed and accuracy, with native Python bindings for easy integration into data science and machine learning workflows. Whether you're indexing millions of vectors or running low-latency queries in production, ZeusDB offers a lightweight, extensible foundation for scalable vector search.

---

<br/>

## ⚡️ Features

 🔍 Approximate Nearest Neighbor (ANN) search with HNSW
 🧠 Supports multiple distance metrics: `cosine`, `l2`, `dot`
 🔥 High-performance Rust backend 
 🗂️ Metadata-aware filtering at query time
 🐍 Simple and intuitive Python API


---

<br/>

## ✅ Supported Distance Metrics

| Metric | Description                          |
|--------|--------------------------------------|
| cosine | Cosine distance (1 - cosine similarity) |
| l2     | Euclidean distance                   |
| dot    | Dot product                 |

---

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

---

<br/>


## ✨ Usage

### 🔥 Quick Start Example 

```python
# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Initialize and set up the database resources
index = vdb.create_index_hnsw(dim = 8, space = "cosine", M = 16, ef_construction = 200, expected_size=5)

# Upload vector records
vectors = {
    "doc_001": ([0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], {"author": "Alice"}),
    "doc_002": ([0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], {"author": "Bob"}),
    "doc_003": ([0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], {"author": "Alice"}),
    "doc_004": ([0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], {"author": "Bob"}),
    "doc_005": ([0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], {"author": "Alice"}),
}

for doc_id, (vec, meta) in vectors.items():
    index.add_point(doc_id, vec, metadata=meta)

# Perform a similarity search and print the top 2 results
# Query Vector
query_vec = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

# Step 5: Query with no filter (all documents)
print("\n--- Querying without filter (all documents) ---")
results = index.query(vector=query_vec, filter=None, top_k=2)
for doc_id, score in results_all:
    print(f"{doc_id} (score={score:.4f})")
```

<br/>


### 🧰 Additional functionality

#### Check the details of your HNSW index 

```python
print(index.info()) 
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

<br/>


#### List records in the index

```python
print("\n--- Index Shows first 5 records ---")
print(index.list(number=5)) # Shows first 5 records
```

<br/>


#### Query with metadata filter (only Alice documents)
```python
print("\n--- Querying with filter: author = 'Alice' ---")
results = index.query(vector=query_vec, filter={"author": "Alice"}, top_k=5)
for doc_id, score in results:
    print(f"{doc_id} (score={score:.4f})")
```

---

<br/>

## 📄 License

This project is licensed under the Apache License 2.0.