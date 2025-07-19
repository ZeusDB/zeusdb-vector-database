# Step 3 Comprehensive Tests - Run these after compiling your Rust code
# These tests verify your ADC implementation and quantization integration

import numpy as np
import random
#import time
from zeusdb_vector_database import VectorDatabase  

# Test Configuration - FIXED: Updated training_size to meet minimum requirement
DIM = 128
NUM_VECTORS = 1000
SUBVECTORS = 8
BITS = 8
TRAINING_SIZE = 1000  # Changed from 500 to 1000 to meet minimum requirement

print("🚀 Starting Step 3 Comprehensive Tests")
print("=" * 60)

# Test 1: Basic Quantization Configuration
print("\n📋 Test 1: Basic Quantization Configuration")
try:
    vdb = VectorDatabase()
    
    # Test valid quantization config
    quantization_config = {
        'type': 'pq',
        'subvectors': SUBVECTORS,
        'bits': BITS,
        'training_size': TRAINING_SIZE,  # Now uses 1000 instead of 500
        'max_training_vectors': 2000
    }
    
    index = vdb.create("hnsw", dim=DIM, quantization_config=quantization_config)
    print("✅ Successfully created index with quantization config")
    
    # Test quantization info
    info = index.get_quantization_info()
    print(f"📊 Quantization info: {info}")
    
    # Test status methods
    print(f"📈 Has quantization: {index.has_quantization()}")
    print(f"📈 Can use quantization: {index.can_use_quantization()}")
    print(f"📈 Is quantized: {index.is_quantized()}")
    
except Exception as e:
    print(f"❌ Test 1 failed: {e}")
    exit(1)

# Test 2: Index Creation Without Quantization (Backward Compatibility)
print("\n📋 Test 2: Backward Compatibility (No Quantization)")
try:
    vdb2 = VectorDatabase()
    index_raw = vdb2.create("hnsw", dim=DIM)  # No quantization
    
    print("✅ Successfully created raw index")
    print(f"📈 Has quantization: {index_raw.has_quantization()}")
    print(f"📈 Is quantized: {index_raw.is_quantized()}")
    
    # Should return None for quantization info
    info_raw = index_raw.get_quantization_info()
    print(f"📊 Raw index quantization info: {info_raw}")
    
except Exception as e:
    print(f"❌ Test 2 failed: {e}")
    exit(1)

# Test 3: Parameter Validation
print("\n📋 Test 3: Parameter Validation")
test_cases = [
    # Valid cases
    ({'type': 'pq', 'subvectors': 4, 'bits': 8, 'training_size': 1000}, True),
    ({'type': 'pq', 'subvectors': 8, 'bits': 4, 'training_size': 1500}, True),
    
    # Invalid cases
    ({'type': 'pq', 'subvectors': 7, 'bits': 8, 'training_size': 1000}, False),  # 128 % 7 != 0
    ({'type': 'pq', 'subvectors': 8, 'bits': 0, 'training_size': 1000}, False),  # bits < 1
    ({'type': 'pq', 'subvectors': 8, 'bits': 9, 'training_size': 1000}, False),  # bits > 8
    ({'type': 'pq', 'subvectors': 8, 'bits': 8, 'training_size': 500}, False),   # training_size < 1000 (FIXED: This test case now correctly expects failure)
    ({'type': 'invalid', 'subvectors': 8, 'bits': 8, 'training_size': 1000}, False),  # invalid type
]

for i, (config, should_succeed) in enumerate(test_cases):
    try:
        vdb_test = VectorDatabase()
        test_index = vdb_test.create("hnsw", dim=DIM, quantization_config=config)
        
        if should_succeed:
            print(f"✅ Test case {i+1}: Valid config accepted")
        else:
            print(f"❌ Test case {i+1}: Invalid config should have been rejected")
            
    except Exception as e:
        if not should_succeed:
            print(f"✅ Test case {i+1}: Invalid config correctly rejected: {str(e)[:50]}...")
        else:
            print(f"❌ Test case {i+1}: Valid config incorrectly rejected: {e}")

# Test 4: Statistics and Info Methods
print("\n📋 Test 4: Statistics and Info Methods")
try:
    # Test with quantization
    stats = index.get_stats()
    print("📊 Statistics with quantization:")
    for key, value in stats.items():
        if 'quantization' in key:
            print(f"  {key}: {value}")
    
    # Test info string
    info_str = index.info()
    print(f"📋 Info string: {info_str}")
    
    # Test performance info
    perf_info = index.get_performance_info()
    print("🚀 Performance info:")
    for key, value in perf_info.items():
        if 'quantization' in key:
            print(f"  {key}: {value}")
    
except Exception as e:
    print(f"❌ Test 4 failed: {e}")

# Test 5: Basic Search Functionality (Before Training)
print("\n📋 Test 5: Basic Search (Before Training)")
try:
    # Generate test vectors
    print("🔄 Generating test vectors...")
    test_vectors = []
    for i in range(50):  # Small number for initial testing
        vector = [random.random() for _ in range(DIM)]
        test_vectors.append(vector)
    
    # Try search on empty index
    query_vector = [random.random() for _ in range(DIM)]
    results = index.search(query_vector, top_k=5)
    print(f"✅ Search on empty index returned {len(results)} results")
    
    # Test different search input formats
    # List format
    results_list = index.search(query_vector, top_k=5)
    print(f"✅ List format search: {len(results_list)} results")
    
    # NumPy array format
    np_query = np.array(query_vector, dtype=np.float32)
    results_numpy = index.search(np_query, top_k=5)
    print(f"✅ NumPy format search: {len(results_numpy)} results")
    
except Exception as e:
    print(f"❌ Test 5 failed: {e}")

# Test 6: Vector Count Tracking
print("\n📋 Test 6: Vector Count Tracking")
try:
    initial_count = index.get_vector_count()
    print(f"📊 Initial vector count: {initial_count}")
    
    # Vector count should be 0 since add() is not fully implemented yet
    assert initial_count == 0, f"Expected 0 vectors, got {initial_count}"
    print("✅ Vector count tracking works correctly")
    
except Exception as e:
    print(f"❌ Test 6 failed: {e}")

# Test 7: Thread Safety - Basic Concurrent Operations
print("\n📋 Test 7: Thread Safety (Basic)")
try:
    import threading
    #import time
    
    results = []
    errors = []
    
    def concurrent_search(index, query, thread_id):
        try:
            result = index.search(query, top_k=5)
            results.append(f"Thread {thread_id}: {len(result)} results")
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    # Create threads for concurrent searches
    threads = []
    query = [random.random() for _ in range(DIM)]
    
    for i in range(5):
        thread = threading.Thread(target=concurrent_search, args=(index, query, i))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    print(f"✅ Concurrent searches completed: {len(results)} successful, {len(errors)} errors")
    if errors:
        print("⚠️  Errors:", errors)
    
except Exception as e:
    print(f"❌ Test 7 failed: {e}")

# Test 8: Rebuild Method (Should Return False - Not Trained Yet)
print("\n📋 Test 8: Rebuild Method")
try:
    # Should return False since PQ is not trained yet
    rebuild_result = index.rebuild_with_quantization()
    print(f"📊 Rebuild result (should be False): {rebuild_result}")
    
    if not rebuild_result:
        print("✅ Rebuild correctly returned False (not trained yet)")
    else:
        print("❌ Rebuild should have returned False")
    
except Exception as e:
    print(f"❌ Test 8 failed: {e}")

# Test 9: Memory and Performance Benchmarks
print("\n📋 Test 9: Memory and Performance")
try:
    # Test benchmark methods
    print("🔄 Running concurrent read benchmark...")
    benchmark_results = index.benchmark_concurrent_reads(query_count=100, max_threads=4)
    
    print("📊 Benchmark results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Test raw performance benchmark
    print("🔄 Running raw performance benchmark...")
    raw_benchmark = index.benchmark_raw_concurrent_performance(query_count=50, max_threads=2)
    
    print("📊 Raw benchmark results:")
    for key, value in raw_benchmark.items():
        print(f"  {key}: {value:.4f}")
    
except Exception as e:
    print(f"❌ Test 9 failed: {e}")

# Test 10: Edge Cases
print("\n📋 Test 10: Edge Cases")
try:
    # Test with None input
    try:
        index.search(None, top_k=5)
        print("❌ Should have rejected None input")
    except Exception:
        print("✅ Correctly rejected None input")
    
    # Test with wrong dimensions
    try:
        wrong_dim_query = [random.random() for _ in range(DIM + 10)]
        index.search(wrong_dim_query, top_k=5)
        print("❌ Should have rejected wrong dimension")
    except Exception:
        print("✅ Correctly rejected wrong dimension")
    
    # Test with empty list
    try:
        index.search([], top_k=5)
        print("❌ Should have rejected empty list")
    except Exception:
        print("✅ Correctly rejected empty list")
    
except Exception as e:
    print(f"❌ Test 10 failed: {e}")

# Test 11: Additional Training Size Edge Cases
print("\n📋 Test 11: Training Size Validation")
try:
    # Test minimum boundary (should succeed)
    try:
        vdb_min = VectorDatabase()
        config_min = {
            'type': 'pq',
            'subvectors': 8,
            'bits': 8,
            'training_size': 1000  # Exactly at minimum
        }
        index_min = vdb_min.create("hnsw", dim=DIM, quantization_config=config_min)
        print("✅ Minimum training_size (1000) accepted")
    except Exception as e:
        print(f"❌ Minimum training_size should be accepted: {e}")
    
    # Test below minimum (should fail)
    try:
        vdb_low = VectorDatabase()
        config_low = {
            'type': 'pq',
            'subvectors': 8,
            'bits': 8,
            'training_size': 999  # Below minimum
        }
        index_low = vdb_low.create("hnsw", dim=DIM, quantization_config=config_low)
        print("❌ Below minimum training_size should be rejected")
    except Exception:
        print("✅ Below minimum training_size correctly rejected")
    
except Exception as e:
    print(f"❌ Test 11 failed: {e}")

# Summary
print("\n" + "=" * 60)
print("🎉 Step 3 Testing Complete!")
print("=" * 60)
print("✅ If all tests passed, your Step 3 implementation is working correctly!")
print("✅ You're ready to move on to Step 4 (Full Add Method Implementation)")
print("✅ The quantization infrastructure is properly integrated")
print("✅ ADC search capability is ready for activation after training")
print("\n📋 Next Steps:")
print("1. Verify all tests passed")
print("2. Check compilation succeeded without errors")
print("3. Move to Step 4 - Implement full training and add() method")
print("4. Test the complete quantization pipeline")
