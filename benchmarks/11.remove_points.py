from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create an HNSW index with specified dimension and expected size
index = vdb.create_index_hnsw(dim=2, expected_size=5)

# Add a point to the index - Blank metadata
index.add_point("to_remove", [0.5, 0.5], {})

# Verify that the point was added
print("\n--- Check 1 ---")
if index.contains("to_remove"):
    print("✓ Point 'to_remove' exists in the index")
else:
    print("✗ Point 'to_remove' not found in the index")

print("\n--- Check 2 ---")
exists = index.contains("to_remove")
print(f"Point 'to_remove' {'found' if exists else 'not found'} in index")

# Verify that the point exists in the index
assert exists, "Point should exist in index"

# Now, let's remove the point from the index
print("\n--- Removing Point ---")
# Remove the point from the index
index.remove_point("to_remove")

print("\n--- Check 3 ---")
exists = index.contains("to_remove")
print(f"Point 'to_remove' {'found' if exists else 'not found'} in index")

# Verify that the point was removed
assert not index.contains("to_remove"), "Point should not exist in index after removal"

