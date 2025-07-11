import numpy as np

# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create index
vdb = VectorDatabase()
index = vdb.create(index_type="hnsw", dim=384, expected_size=10000)

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

# print("\n--- Index Shows first 5 records ---")
# print(index.list(number=5)) # Shows first 5 records

# # Add another single record
# index.add({
#     "id": "doc4", 
#     "values": [0.4, 0.5, 0.6] * 128,  # 384-dim
#     "metadata": {"source": "web", "type": "article"}
# })

# # Outputs the details of the HNSW index
# print("\n--- Shows Index Information after new row added---")
# print(index.info()) 

# print("\n--- Shows Index records ---")
# print(index.list(number=5)) # Shows first 5 records




