# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Initialize and set up the database resources
index = vdb.create_index_hnsw(dim = 2, space = "cosine", M = 16, ef_construction = 200, expected_size=5)

# Upload vector records using the unified `add()` method
index.add([
        {"id": "doc1", "values": [0.1, 0.2], "metadata": {"tag": "alpha"}},
        {"id": "doc2", "values": [0.3, 0.4], "metadata": {"tag": "beta"}},
        {"id": "doc3", "values": [0.5, 0.6], "metadata": {"tag": "gamma"}},
    ])

# Get records by ID using the unified `get_records()` method

# Single record
print("\n--- Get Single Record ---")
rec = index.get_records("doc1")
print(rec)

# Batch records
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
