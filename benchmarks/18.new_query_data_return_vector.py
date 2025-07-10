import numpy as np
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create index with dim=4
index = vdb.create_index_hnsw(dim=4, expected_size=100)

# Show index info
print("\n--- Shows Initial Index Information ---")
print(index.info()) 

# Create 100 records with dim=4
ids = [f"doc_{i}" for i in range(100)]
vectors = np.random.rand(100, 4).astype(np.float32)  # NumPy 2D array
metadatas = [{"batch": "test", "index": str(i)} for i in range(100)]

# Insert using Format 3
result = index.add({
    "ids": ids,
    "embeddings": vectors,
    "metadatas": metadatas
})

print("\n--- Shows Data Insertion Results ---")
print(f"Add result: {result}")

# Show updated index info
print("\n--- Shows Index Information ---")
print(index.info()) 

# Query with top_k=2 and return_vector=True
query_vector = np.random.rand(4).tolist()
results = index.search(query_vector, top_k=2, return_vector=True)

print("\n--- Top 2 Results (with vectors) - raw ---")
print(results)

print("\n--- Top 2 Results (with vectors) ---")
for i, res in enumerate(results, 1):
    print(f"{i}. ID: {res['id']}, Score: {res['score']:.4f}, Vector: {res['vector']}, Metadata: {res['metadata']}")
