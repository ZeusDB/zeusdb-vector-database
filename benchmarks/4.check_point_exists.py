# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create an HNSW index with specified dimension and expected size
index = vdb.create(index_type="hnsw", dim=2, expected_size=5)

# Add a point to the index - Blank metadata
result = index.add({
    "id": "doc1", 
    "values": [0.5, 0.5],
    "metadata": {}
})

# Verify that the point was added
print("\n--- Check 1 ---")
if index.contains("doc1"):
    print("✓ Point 'doc1' exists in the index")
else:
    print("✗ Point 'doc1' not found in the index")

print("\n--- Check 2 ---")
exists = index.contains("doc1")
print(f"Point 'doc1' {'found' if exists else 'not found'} in index")

# Verify that the point exists in the index
assert exists, "Point should exist in index"
