from zeusdb_vector_database import VectorDatabase

print("\n✅ Testing lowercase distance metric names")

metrics = ["cosine", "l1", "l2", "Cosine", "L1", "L2"]

for metric in metrics:
    print(f"\n--- Creating index with space = '{metric}' ---")
    try:
        vdb = VectorDatabase()
        index = vdb.create(index_type="hnsw", dim=4, space=metric, m=8, ef_construction=100, expected_size=10)
        print(f"✔️  Successfully created index with space = '{metric}'")
    except Exception as e:
        print(f"❌ Failed to create index with space = '{metric}': {e}")


