from zeusdb_vector_database import VectorDatabase

vdb = VectorDatabase()

# Create an HNSW index with specified dimension and expected size
index = vdb.create_index_hnsw(dim=2, expected_size=5)

# Add points in batch with metadata
# Each point has an ID, a vector, and metadata
# points = [
#     {"id": "a", "vector": [1.0, 0.0], "metadata": {"type": "x"}},
#     {"id": "b", "vector": [0.0, 1.0], "metadata": {"type": "y"}},
#     {"id": "c", "vector": [1.0, 1.0], "metadata": {"type": "x"}},
#     ]

points = [
    ("a", [1.0, 0.0], {"type": "x"}),
    ("b", [0.0, 1.0], {"type": "y"}),
    ("c", [1.0, 1.0], {"type": "x"}),
    ]

# Add points in batch
index.add_batch(points)

# Query with filter
results = index.query([1.0, 0.0], top_k=3, filter={"type": "x"})

# Check results
for r in results:
    meta = index.get_vector_metadata(r[0])
    assert meta is not None
    assert meta["type"] == "x"

# Print results
print("Results with filter 'type = x':" + str(results))