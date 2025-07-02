import numpy as np
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create index
vdb = VectorDatabase()
index = vdb.create_index_hnsw(dim=384, expected_size=10000)

# Outputs the details of the HNSW index
print("\n--- Shows Initial Index Information ---")
print(index.info()) 

# Format 3: Separate arrays with NumPy (fastest for large batches)
ids = [f"doc_{i}" for i in range(1000)]
vectors = np.random.rand(1000, 384).astype(np.float32)  # NumPy 2D array
#metadatas = [{"batch": "large", "index": i} for i in range(1000)] # ONLY SUPPORTS STRING IN METADATA AT THIS STAGE
metadatas = [{"batch": "large", "index": str(i)} for i in range(1000)]

result = index.add({
    "ids": ids,
    "embeddings": vectors,  # Zero-copy NumPy access!
    "metadatas": metadatas
})

print("\n--- Shows Data Insertion Results ---")
print(f"Batch result: {result}")
# Output: BatchResult(inserted=1000, errors=0, shape=(1000, 384))


# Outputs the details of the HNSW index
print("\n--- Shows Index Information ---")
print(index.info()) 



# Query
query_vector = np.random.rand(384).tolist()
results = index.query(query_vector, top_k=5)
print("\n--- Top K results ---")
print(results)

