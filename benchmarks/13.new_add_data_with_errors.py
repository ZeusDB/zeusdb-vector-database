# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Step 1: Set up index with dim=8
index = vdb.create(index_type="hnsw", dim=8, space="cosine", m=16, ef_construction=200, expected_size=5)

# Step 2: Add initial valid records
records = [
    {"id": "doc_001", "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], "metadata": {"author": "Alice"}},
    {"id": "doc_002", "values": [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], "metadata": {"author": "Bob"}},
    {"id": "doc_003", "values": [0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], "metadata": {"author": "Alice"}},
    {"id": "doc_004", "values": [0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], "metadata": {"author": "Bob"}},
    {"id": "doc_005", "values": [0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], "metadata": {"author": "Alice"}},
]

add_result_1 = index.add(records)
print("\n--- First Add Summary ---")
print(add_result_1.summary())  # Expect: ✅ 5 inserted, ❌ 0 errors

# Step 3: Add batch with 1 error and 1 overwrite
# - "doc_007" is new (valid)
# - "doc_002" already exists (should overwrite - valid)
# - "doc_006" has invalid vector length (only 6 values) - error
error_records = [
    {"id": "doc_007", "values": [0.5, 0.4, 0.3, 0.7, 0.8, 0.6, 0.3, 0.9], "metadata": {"author": "Eve"}},   # Valid - new
    {"id": "doc_002", "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], "metadata": {"author": "Mallory"}}, # Valid - overwrite
    {"id": "doc_006", "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "metadata": {"author": "Zoe"}},  # Invalid dimension
]

add_result_2 = index.add(error_records)
print("\n--- Second Add Summary ---")
print(add_result_2.summary())  # Expect: ✅ 2 inserted, ❌ 1 errors

print("\n--- Detailed Errors ---")
for err in add_result_2.errors:
    print(f"❌ {err}")

# Verify the overwrite worked
print("\n--- Verify Overwrite ---")
doc_002_metadata = index.get_records(["doc_002"], return_vector=False)[0]["metadata"]
print(f"doc_002 author after overwrite: {doc_002_metadata.get('author')}")  # Should be "Mallory"