from zeusdb_vector_database import VectorDatabase

# Step 1: Create the index
vdb = VectorDatabase()
index = vdb.create_index_hnsw(dim=8, space="cosine", M=16, ef_construction=200, expected_size=5)

# Step 2: Add data points
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

# Outputs the details of the HNSW index
print("\n--- Index Summary Information ---")
print(index.info())  

print("\n--- Index Shows first 10 records ---")
print(index.list())         # Shows first 10 records

print("\n--- Index Shows first 2 records ---")
print(index.list(number=2)) # Shows first 2 records