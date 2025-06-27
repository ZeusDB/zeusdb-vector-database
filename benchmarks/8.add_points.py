from zeusdb_vector_database import VectorDatabase

vdb = VectorDatabase()

index = vdb.create_index_hnsw(dim=4, expected_size=10)

index.add_point("vec1", [0.1, 0.2, 0.3, 0.4], {"label": "A"})
index.add_point("vec2", [0.2, 0.1, 0.4, 0.3], {"label": "B"})    
    
results = index.query([0.1, 0.2, 0.3, 0.4], top_k=2)
print(f"Results: {results}")

ids = [r[0] for r in results]
print(f"IDs: {ids}")
assert "vec1" in ids, "vec1 should be in the results"
