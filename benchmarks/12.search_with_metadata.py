from zeusdb_vector_database import VectorDatabase


# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create an HNSW index with specified dimension and expected size
index = vdb.create_index_hnsw(dim=2, expected_size=5)

# Add a point to the index - with metadata
index.add_point("foo", [0.1, 0.2], {"cat": "bar"})

# Verify that the point was added
results = index.search_with_metadata([0.1, 0.2], top_k=1)
print("Results with metadata:", results)

# Check results
assert len(results) == 1, "There should be one result"
assert results[0][0] == "foo"

# Extract metadata from the result
meta = results[0][2]

# Check metadata
print("Metadata:", meta)


# use the include_metadata parameter to get metadata
results2 = index.search_with_metadata([0.1, 0.2], top_k=1, include_metadata=True)
print("New Results with metadata:", results2)
# Extract metadata from the result
meta2 = results2[0][2]
# Check metadata
print("Metadata:", meta2)
    