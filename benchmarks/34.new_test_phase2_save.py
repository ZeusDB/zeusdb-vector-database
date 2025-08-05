from zeusdb_vector_database import VectorDatabase
import numpy as np
import os

# Create test data
vdb = VectorDatabase()
index = vdb.create("hnsw", dim=128)

# Add some test vectors
vectors = np.random.random((50, 128)).astype(np.float32)
test_data = {
    'vectors': vectors.tolist(),
    'ids': [f'test_{i}' for i in range(50)]
}

index.add(test_data)

# Test Phase 2 enhanced save
index.save("test_phase2.zdb")

# Check files were created
#import os
files = os.listdir("test_phase2.zdb")
print("Files created:", files)

# Should see: hnsw_index.hnsw.graph and hnsw_index.hnsw.data
assert "hnsw_index.hnsw.graph" in files
print("✅ Phase 2 save enhancement working!")