from zeusdb_vector_database import VectorDatabase

print("\n=== ZeusDB Metadata Filter Test ===")

# Create the index
vdb = VectorDatabase()
index = vdb.create_index_hnsw(dim=4, space="cosine", expected_size=10)

# Add records with metadata of varying structure
records = [
    {"id": "v1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"author": "Alice", "score": 95}},
    {"id": "v2", "values": [0.9, 0.8, 0.7, 0.6], "metadata": {"author": "Bob", "score": 80}},
    {"id": "v3", "values": [0.15, 0.25, 0.35, 0.45], "metadata": {"author": "Alice", "score": 85}},
    {"id": "v4", "values": [0.92, 0.82, 0.72, 0.62], "metadata": {"author": "Charlie", "score": 78}},
    {"id": "v5", "values": [0.12, 0.22, 0.32, 0.42], "metadata": {"tags": ["ml", "ai"], "score": 91}},
]

result = index.add(records)
print("\n--- Added Records ---")
print(result.summary())

# Define the query vector
query = [0.1, 0.2, 0.3, 0.4]

print("\n--- No Filter ---")
res_all = index.query(vector=query, filter=None, top_k=10)
for r in res_all:
    print(f"{r['id']} → {r['metadata']}")

print("\n--- Filter: author = 'Alice' ---")
res_alice = index.query(vector=query, filter={"author": "Alice"}, top_k=10)
for r in res_alice:
    print(f"{r['id']} → {r['metadata']}")

print("\n--- Filter: author != 'Bob' ---")
res_not_bob = index.query(vector=query, filter={"author": {"ne": "Bob"}}, top_k=10)
for r in res_not_bob:
    print(f"{r['id']} → {r['metadata']}")

print("\n--- Filter: score > 90 ---")
res_score = index.query(vector=query, filter={"score": {"gt": 90}}, top_k=10)
for r in res_score:
    print(f"{r['id']} → {r['metadata']}")

print("\n--- Filter: tags contains 'ai' ---")
res_tags = index.query(vector=query, filter={"tags": {"contains": "ai"}}, top_k=10)
for r in res_tags:
    print(f"{r['id']} → {r['metadata']}")

print("\n--- Filter: author = 'DoesNotExist' ---")
res_none = index.query(vector=query, filter={"author": "DoesNotExist"}, top_k=10)
print(f"Returned {len(res_none)} results (expected: 0)")
