from zeusdb_vector_database import VectorDatabase

# Step 1: Create the index
vdb = VectorDatabase()
index = vdb.create_index_hnsw(dim=8, space="cosine", M=16, ef_construction=200)

# Step 2: Add data points
vectors = {
    "doc_001": ([0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], {"author": "Alice"}),
    "doc_002": ([0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], {"author": "Bob"}),
    "doc_003": ([0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], {"author": "Alice"}),
    "doc_004": ([0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], {"author": "Bob"}),
    "doc_005": ([0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], {"author": "Alice"}),
}

for doc_id, (vec, meta) in vectors.items():
    index.add_point(doc_id, vec, metadata=meta)

# Step 3: Perform query vector (similar to Alice docs)
query_vec = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

# Step 4: Query with metadata filter (only Alice documents)
print("\n--- Querying with filter: author = 'Alice' ---")
results = index.query(vector=query_vec, filter={"author": "Alice"}, top_k=5)
#for doc_id, score in results:
#    print(f"{doc_id} (score={score:.4f})")
print(results)
