import numpy as np
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create index
index = vdb.create_index_hnsw(dim=384, expected_size=10000)

# Outputs the details of the HNSW index
print("\n--- Shows Initial Index Information ---")
print(index.info())

# Format 3: Separate arrays with NumPy (fastest for large batches)
ids = [f"doc_{i}" for i in range(1000)]
vectors = np.random.rand(1000, 384).astype(np.float32)  # NumPy 2D array

# Create mixed metadata: 500 "large" + 500 "small"
metadatas = []
for i in range(1000):
    if i < 500:
        metadatas.append({"batch": "large", "index": str(i)})
    else:
        metadatas.append({"batch": "small", "index": str(i)})

result = index.add({
    "ids": ids,
    "embeddings": vectors,  # Zero-copy NumPy access!
    "metadatas": metadatas
})

print("\n--- Shows Data Insertion Results ---")
print(f"Add result: {result}")
# Output: AddResult(inserted=1000, errors=0, shape=(1000, 384))

# Outputs the details of the HNSW index
print("\n--- Shows Index Information ---")
print(index.info())

# Query all records
query_vector = np.random.rand(384).tolist()
all_results = index.search(query_vector, top_k=10)
print("\n--- Top 10 results (raw no filter) ---")
print(all_results)
print(type(all_results))
print("\n--- Top 10 results (no filter) ---")
for i, result in enumerate(all_results):
    print(f"{i+1}. ID: {result['id']}, Score: {result['score']:.4f}, Batch: {result['metadata']['batch']}")

# Query only "large" batch records
large_results = index.search(query_vector, filter={"batch": "large"}, top_k=5)
print("\n--- Top 5 'large' batch results - raw ---")
print(large_results)
print("\n--- Top 5 'large' batch results ---")
for i, result in enumerate(large_results):
    print(f"{i+1}. ID: {result['id']}, Score: {result['score']:.4f}, Batch: {result['metadata']['batch']}")

# Query only "small" batch records  
small_results = index.search(query_vector, filter={"batch": "small"}, top_k=5)
print("\n--- Top 5 'small' batch results ---")
for i, result in enumerate(small_results):
    print(f"{i+1}. ID: {result['id']}, Score: {result['score']:.4f}, Batch: {result['metadata']['batch']}")

# Verify the split
print("\n--- Verification ---")
print(f"Total records: {len(ids)}")
print(f"Large batch records: {sum(1 for m in metadatas if m['batch'] == 'large')}")
print(f"Small batch records: {sum(1 for m in metadatas if m['batch'] == 'small')}")