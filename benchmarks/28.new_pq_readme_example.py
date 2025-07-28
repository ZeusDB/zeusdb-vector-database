from zeusdb_vector_database import VectorDatabase
import numpy as np

# Create index with product quantization
vdb = VectorDatabase()

# Configure quantization for memory efficiency
quantization_config = {
    'type': 'pq',                  # `pq` for Product Quantization
    'subvectors': 8,               # Divide 1536-dim vectors into 8 subvectors of 192 dims each
    'bits': 8,                     # 256 centroids per subvector (2^8)
    'training_size': 10000,        # Train when 10k vectors are collected
    'max_training_vectors': 50000  # Use max 50k vectors for training
}

# Create index with quantization
# This will automatically handle training when enough vectors are added
index = vdb.create(
    index_type="hnsw",
    dim=1536,           # OpenAI embedding dimension for `text-embedding-3-small`
    quantization_config=quantization_config
)

# Add vectors - training triggers automatically at threshold
documents = [
    {
        "id": f"doc_{i}", 
        "values": np.random.rand(1536).astype(float).tolist(),
        "metadata": {"category": "tech", "year": 2024}
    }
    for i in range(15000) 
]

# Training will trigger automatically when 10k vectors are added
result = index.add(documents)
print(f"Added {result.total_inserted} vectors")

# Check quantization status
print(f"Training progress: {index.get_training_progress():.1f}%")
print(f"Storage mode: {index.get_storage_mode()}")
print(f"Is quantized: {index.is_quantized()}")

# Get compression statistics
quant_info = index.get_quantization_info()
if quant_info:
    print(f"Compression ratio: {quant_info['compression_ratio']:.1f}x")
    print(f"Memory usage: {quant_info['memory_mb']:.1f} MB")

# Search works seamlessly with quantized storage
query_vector = np.random.rand(1536).astype(float).tolist()
results = index.search(vector=query_vector, top_k=3)

# Simply print raw results
print(results)