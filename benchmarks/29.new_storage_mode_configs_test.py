# Usage Examples for Storage Mode Configuration

from zeusdb_vector_database import VectorDatabase
import numpy as np

# Example 1: Default (Memory Efficient) - quantized_only
vdb = VectorDatabase()
index_memory_efficient = vdb.create(
    "hnsw", 
    dim=768,
    quantization_config={
        "type": "pq",
        "subvectors": 8,
        "bits": 8,
        "training_size": 10000
        # storage_mode defaults to "quantized_only"
    }
)

# Example 2: Explicit quantized_only mode
index_explicit = vdb.create(
    "hnsw", 
    dim=768,
    quantization_config={
        "type": "pq",
        "subvectors": 8,
        "bits": 8,
        "training_size": 10000,
        "storage_mode": "quantized_only"
    }
)

# Example 3: Keep raw vectors for exact reconstruction
index_with_raw = vdb.create(
    "hnsw", 
    dim=768,
    quantization_config={
        "type": "pq",
        "subvectors": 8,
        "bits": 8,
        "training_size": 10000,
        "storage_mode": "quantized_with_raw"  # Keep both quantized + raw
    }
)
# This will show a warning about increased memory usage

# Testing the different modes
def test_storage_modes():
    # Generate test data
    vectors = np.random.random((15000, 768)).astype(np.float32)
    
    # Test quantized_only mode
    print("=== Testing quantized_only mode ===")
    index1 = vdb.create("hnsw", dim=768, quantization_config={
        "type": "pq", "subvectors": 8, "bits": 8, 
        "training_size": 10000, "storage_mode": "quantized_only"
    })
    
    # Add vectors (will trigger training)
    result1 = index1.add(vectors.tolist())
    print(f"Added: {result1.total_inserted}, Errors: {result1.total_errors}")
    
    # Check stats
    stats1 = index1.get_stats()
    print(f"Storage mode: {stats1['storage_mode']}")
    print(f"Raw vectors stored: {stats1['raw_vectors_stored']}")
    print(f"Quantized codes stored: {stats1['quantized_codes_stored']}")
    
    # Get records (will use PQ reconstruction)
    records1 = index1.get_records(["vec_1"], return_vector=True)
    print(f"Vector available: {'vector' in records1[0] if records1 else False}")

    if records1 and 'vector' in records1[0]:
        print(f"Vector shape: {len(records1[0]['vector'])}")
    
    print("\n=== Testing quantized_with_raw mode ===")
    index2 = vdb.create("hnsw", dim=768, quantization_config={
        "type": "pq", "subvectors": 8, "bits": 8, 
        "training_size": 10000, "storage_mode": "quantized_with_raw"
    })
    
    # Add vectors (will trigger training)
    result2 = index2.add(vectors.tolist())
    print(f"Added: {result2.total_inserted}, Errors: {result2.total_errors}")
    
    # Check stats
    stats2 = index2.get_stats()
    print(f"Storage mode: {stats2['storage_mode']}")
    print(f"Raw vectors stored: {stats2['raw_vectors_stored']}")
    print(f"Quantized codes stored: {stats2['quantized_codes_stored']}")
    
    # Get records (will use exact raw vectors)
    records2 = index2.get_records(["vec_1"], return_vector=True)
    print(f"Vector available: {'vector' in records2[0] if records2 else False}")

    if records2 and 'vector' in records2[0]:
        print(f"Vector shape: {len(records2[0]['vector'])}")
    
    # Compare memory usage
    print("\nMemory comparison:")
    print(f"quantized_only - Raw vectors: {stats1['raw_vectors_stored']}")
    print(f"quantized_with_raw - Raw vectors: {stats2['raw_vectors_stored']}")

# Error handling test
def test_invalid_storage_mode():
    try:
        vdb = VectorDatabase()
        vdb.create("hnsw", dim=768, quantization_config={
            "type": "pq", 
            "subvectors": 8, 
            "bits": 8, 
            "training_size": 10000,
            "storage_mode": "invalid_mode"  # This should fail
        })
    except ValueError as e:
        print(f"Expected error: {e}")

if __name__ == "__main__":
    test_storage_modes()
    test_invalid_storage_mode()