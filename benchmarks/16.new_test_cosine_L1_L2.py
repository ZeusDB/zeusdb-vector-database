from zeusdb_vector_database import VectorDatabase

records = [
    {"id": "doc_001", "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], "metadata": {"author": "Alice"}},
    {"id": "doc_002", "values": [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], "metadata": {"author": "Bob"}},
    {"id": "doc_003", "values": [0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], "metadata": {"author": "Alice"}},
    {"id": "doc_004", "values": [0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], "metadata": {"author": "Bob"}},
    {"id": "doc_005", "values": [0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], "metadata": {"author": "Alice"}},
]

query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

# -------------------- COSINE --------------------
print("\n==================== Testing space = 'cosine' ====================\n")

vdb_cos = VectorDatabase()
index_cos = vdb_cos.create(index_type="hnsw", dim=8, space="cosine", m=16, ef_construction=200, expected_size=5)

add_result_cos = index_cos.add(records)
print("--- Add Results Summary ---")
print(add_result_cos.summary())

results_cos = index_cos.search(vector=query_vector, filter=None, top_k=2)
print("\n--- Top 2 Neighbors (No Filter) ---")
for i, res in enumerate(results_cos, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")

results_cos_filtered = index_cos.search(vector=query_vector, filter={"author": "Alice"}, top_k=5)
print("\n--- Top 5 Neighbors (Filter: author = 'Alice') ---")
for i, res in enumerate(results_cos_filtered, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")

# -------------------- L2 --------------------
print("\n==================== Testing space = 'L2' ====================\n")

vdb_l2 = VectorDatabase()
index_l2 = vdb_l2.create(index_type="hnsw", dim=8, space="L2", m=16, ef_construction=200, expected_size=5)


add_result_l2 = index_l2.add(records)
print("--- Add Results Summary ---")
print(add_result_l2.summary())

results_l2 = index_l2.search(vector=query_vector, filter=None, top_k=2)
print("\n--- Top 2 Neighbors (No Filter) ---")
for i, res in enumerate(results_l2, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")

results_l2_filtered = index_l2.search(vector=query_vector, filter={"author": "Alice"}, top_k=5)
print("\n--- Top 5 Neighbors (Filter: author = 'Alice') ---")
for i, res in enumerate(results_l2_filtered, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")

# -------------------- L1 --------------------
print("\n==================== Testing space = 'L1' ====================\n")

vdb_l1 = VectorDatabase()
index_l1 = vdb_l1.create(index_type="hnsw", dim=8, space="L1", m=16, ef_construction=200, expected_size=5)

add_result_l1 = index_l1.add(records)
print("--- Add Results Summary ---")
print(add_result_l1.summary())

results_l1 = index_l1.search(vector=query_vector, filter=None, top_k=2)
print("\n--- Top 2 Neighbors (No Filter) ---")
for i, res in enumerate(results_l1, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")

results_l1_filtered = index_l1.search(vector=query_vector, filter={"author": "Alice"}, top_k=5)
print("\n--- Top 5 Neighbors (Filter: author = 'Alice') ---")
for i, res in enumerate(results_l1_filtered, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")
