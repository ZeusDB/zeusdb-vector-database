import time
import random
from random import randint

# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create index with large capacity
index = vdb.create(index_type="hnsw", dim=8, expected_size=100000)

# Insert 10,000 random records
data = [{"id": f"doc_{i}", "values": [random.random() for _ in range(8)]} for i in range(10000)]
index.add(data)

# Now simulate overwrite churn
overwrite_data = [{"id": f"doc_{randint(0, 9999)}", "values": [random.random() for _ in range(8)]} for _ in range(2000)]

start = time.perf_counter()
index.add(overwrite_data)
end = time.perf_counter()

print(f"Overwrite batch of 2000 took {end - start:.3f} sec")
