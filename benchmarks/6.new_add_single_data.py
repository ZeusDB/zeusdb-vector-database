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

# Format 1: Single object
result = index.add({
    "id": "doc1", 
    "values": [0.1, 0.2, 0.3] * 128,  # 384-dim
    "metadata": {"source": "web", "type": "article"}
})

print("\n--- Shows Data Insertion Results ---")
print(f"Added: {result.total_inserted}, Errors: {result.total_errors}")

# Outputs the details of the HNSW index
print("\n--- Shows Index Information ---")
print(index.info()) 

print("\n--- Index Shows first 5 records ---")
print(index.list(number=5)) # Shows first 5 records

# Add another single record
index.add({
    "id": "doc2", 
    "values": [0.4, 0.5, 0.6] * 128,  # 384-dim
    "metadata": {"source": "web", "type": "article"}
})

# Outputs the details of the HNSW index
print("\n--- Shows Index Information after new row added---")
print(index.info()) 

print("\n--- Shows Index records ---")
print(index.list(number=5)) # Shows first 5 records




