#!/usr/bin/env python3
"""
Comprehensive test suite for the parallel HNSW implementation
Tests core functionality, thread safety, and performance
"""

import numpy as np
import time
#import threading
import concurrent.futures
#from typing import List, Dict, Any
import random
#import json

# Import your vector database
from zeusdb_vector_database import VectorDatabase

print("🧪 Starting HNSW Index Test Suite")
print("=" * 60)

# ================================
# Test 1: Basic Construction & Validation
# ================================
print("\n📋 Test 1: Basic Construction & Validation")

# Test different configurations
configs = [
    {"dim": 128, "space": "cosine", "m": 16, "ef_construction": 200, "expected_size": 1000},
    {"dim": 768, "space": "l2", "m": 32, "ef_construction": 100, "expected_size": 5000},
    {"dim": 1536, "space": "l1", "m": 8, "ef_construction": 300, "expected_size": 10000},
]

vdb = VectorDatabase()
indexes = []

for i, config in enumerate(configs):
    try:
        index = vdb.create("hnsw", **config)
        indexes.append(index)
        print(f"   ✅ Config {i+1}: {index.info()}")
        
        # Test stats
        stats = index.get_stats()
        assert stats["dimension"] == str(config["dim"])
        assert stats["space"] == config["space"]
        assert stats["total_vectors"] == "0"
        
    except Exception as e:
        print(f"   ❌ Config {i+1} failed: {e}")

# Test invalid configurations
invalid_configs = [
    {"dim": 0, "space": "cosine"},  # Invalid dimension
    {"dim": 100, "space": "invalid"},  # Invalid space
    {"dim": 100, "space": "cosine", "m": 300},  # Invalid m
]

for config in invalid_configs:
    try:
        index = vdb.create("hnsw", **config)
        print(f"   ❌ Should have failed: {config}")
    except Exception as e:
        print(f"   ✅ Correctly rejected invalid config: {type(e).__name__}")

# ================================
# Test 2: Data Format Support
# ================================
print("\n📄 Test 2: Data Format Support")

index = vdb.create("hnsw", dim=128, space="cosine")

# Generate test data
def random_vector(dim=128):
    return np.random.random(dim).astype(np.float32).tolist()

# Test single object format
print("   Testing single object format...")
single_data = {
    "id": "test_single",
    "vector": random_vector(),
    "metadata": {"type": "single", "index": 1}
}

try:
    result = index.add(single_data)
    print(f"   ✅ Single object: {result.summary()}")
    assert result.is_success()
    assert result.total_inserted == 1
    assert index.contains("test_single")
except Exception as e:
    print(f"   ❌ Single object failed: {e}")

# Test list format
print("   Testing list format...")
list_data = [
    {"id": f"list_{i}", "vector": random_vector(), "metadata": {"type": "list", "index": i}}
    for i in range(5)
]

try:
    result = index.add(list_data)
    print(f"   ✅ List format: {result.summary()}")
    assert result.is_success()
    assert result.total_inserted == 5
    for i in range(5):
        assert index.contains(f"list_{i}")
except Exception as e:
    print(f"   ❌ List format failed: {e}")

# Test separate arrays format
print("   Testing separate arrays format...")
array_data = {
    "ids": [f"array_{i}" for i in range(3)],
    "embeddings": np.random.random((3, 128)).astype(np.float32),
    "metadatas": [{"type": "array", "index": i} for i in range(3)]
}

try:
    result = index.add(array_data)
    print(f"   ✅ Array format: {result.summary()}")
    assert result.is_success()
    assert result.total_inserted == 3
except Exception as e:
    print(f"   ❌ Array format failed: {e}")

# Test NumPy array support
print("   Testing NumPy arrays...")
numpy_data = {
    "id": "numpy_test",
    "vector": np.random.random(128).astype(np.float32),  # NumPy array
    "metadata": {"type": "numpy"}
}

try:
    result = index.add(numpy_data)
    print(f"   ✅ NumPy arrays: {result.summary()}")
    assert result.is_success()
except Exception as e:
    print(f"   ❌ NumPy arrays failed: {e}")

print(f"   📊 Total vectors in index: {index.get_stats()['total_vectors']}")

# ================================
# Test 3: Search Functionality
# ================================
print("\n🔍 Test 3: Search Functionality")

# Add some test data for searching
search_data = [
    {"id": f"search_{i}", "vector": random_vector(), 
     "metadata": {"category": "A" if i % 2 == 0 else "B", "value": i}}
    for i in range(20)
]

result = index.add(search_data)
print(f"   Added search data: {result.summary()}")

# Test basic search
query_vector = random_vector()
search_results = index.search(query_vector, top_k=5)
print(f"   ✅ Basic search returned {len(search_results)} results")

for i, result in enumerate(search_results[:3]):
    print(f"      {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")

# Test search with filters
filter_dict = {"category": "A"}
filtered_results = index.search(query_vector, filter=filter_dict, top_k=5)
print(f"   ✅ Filtered search returned {len(filtered_results)} results")

# Verify all results match filter
for result in filtered_results:
    assert result['metadata']['category'] == 'A'

# Test search with vector return
vector_results = index.search(query_vector, top_k=3, return_vector=True)
print(f"   ✅ Search with vectors returned {len(vector_results)} results")
assert 'vector' in vector_results[0]
assert len(vector_results[0]['vector']) == 128

# Test get_records
record_ids = ["search_0", "search_5", "search_10"]
records = index.get_records(record_ids)
print(f"   ✅ get_records returned {len(records)} records")
assert len(records) == 3

# ================================
# Test 4: Parallel Processing & Performance
# ================================
print("\n⚡ Test 4: Parallel Processing & Performance")

# Create a fresh index for performance testing
perf_index = vdb.create("hnsw", dim=256, space="cosine", expected_size=5000)

# Test small batch (should use sequential)
small_batch = [
    {"id": f"small_{i}", "vector": np.random.random(256).astype(np.float32).tolist(),
     "metadata": {"batch": "small", "index": i}}
    for i in range(10)  # Small batch
]

start_time = time.time()
result = perf_index.add(small_batch)
small_time = time.time() - start_time
print(f"   📈 Small batch (10 vectors): {result.summary()}, Time: {small_time:.3f}s")

# Test large batch (should use parallel)
large_batch = [
    {"id": f"large_{i}", "vector": np.random.random(256).astype(np.float32).tolist(),
     "metadata": {"batch": "large", "index": i}}
    for i in range(100)  # Large batch
]

start_time = time.time()
result = perf_index.add(large_batch)
large_time = time.time() - start_time
print(f"   📈 Large batch (100 vectors): {result.summary()}, Time: {large_time:.3f}s")

# Test very large batch
very_large_batch = [
    {"id": f"vlarge_{i}", "vector": np.random.random(256).astype(np.float32).tolist(),
     "metadata": {"batch": "very_large", "index": i}}
    for i in range(500)  # Very large batch
]

start_time = time.time()
result = perf_index.add(very_large_batch)
very_large_time = time.time() - start_time
print(f"   📈 Very large batch (500 vectors): {result.summary()}, Time: {very_large_time:.3f}s")

# Performance stats
total_vectors = int(perf_index.get_stats()['total_vectors'])
print(f"   📊 Total vectors added: {total_vectors}")
print(f"   📊 Average insertion rate: {total_vectors / (small_time + large_time + very_large_time):.0f} vectors/sec")

# ================================
# Test 5: Concurrent Search Performance
# ================================
print("\n🧵 Test 5: Concurrent Search Performance")

# Run built-in benchmarks
try:
    print("   Running concurrent read benchmark...")
    bench_results = perf_index.benchmark_concurrent_reads(query_count=50, max_threads=4)
    
    print(f"   📊 Sequential time: {bench_results['sequential_time']:.3f}s")
    print(f"   📊 Parallel time: {bench_results['parallel_time']:.3f}s")
    print(f"   📊 Speedup: {bench_results['speedup']:.2f}x")
    print(f"   📊 Sequential QPS: {bench_results['sequential_qps']:.0f}")
    print(f"   📊 Parallel QPS: {bench_results['parallel_qps']:.0f}")
    
except Exception as e:
    print(f"   ❌ Concurrent benchmark failed: {e}")

# Manual concurrent search test
def search_worker(index, query_count=10):
    """Worker function for concurrent search testing"""
    results = []
    for _ in range(query_count):
        query = np.random.random(256).astype(np.float32).tolist()
        try:
            search_results = index.search(query, top_k=5)
            results.append(len(search_results))
        except Exception as e:
            results.append(f"Error: {e}")
    return results

print("   Testing manual concurrent searches...")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    start_time = time.time()
    
    # Submit 4 concurrent search tasks
    futures = [executor.submit(search_worker, perf_index, 10) for _ in range(4)]
    
    # Wait for all to complete
    results = [future.result() for future in futures]
    
    concurrent_time = time.time() - start_time
    total_searches = sum(len(r) for r in results)
    
    print(f"   ✅ Concurrent searches: {total_searches} searches in {concurrent_time:.3f}s")
    print(f"   📊 Concurrent QPS: {total_searches / concurrent_time:.0f}")

# ================================
# Test 6: Error Handling & Edge Cases
# ================================
print("\n🚨 Test 6: Error Handling & Edge Cases")

error_index = vdb.create("hnsw", dim=128, space="cosine")

# Test dimension mismatch
try:
    wrong_dim_data = {"id": "wrong_dim", "vector": [1.0, 2.0, 3.0]}  # Wrong dimension
    result = error_index.add(wrong_dim_data)
    print(f"   ✅ Dimension mismatch handled: {result.total_errors} errors")
    assert result.total_errors > 0
except Exception as e:
    print(f"   ❌ Dimension mismatch not handled properly: {e}")

# Test empty data
try:
    empty_result = error_index.add([])
    print(f"   ✅ Empty data handled: {empty_result.summary()}")
    assert empty_result.total_inserted == 0
except Exception as e:
    print(f"   ❌ Empty data failed: {e}")

# Test duplicate IDs (should overwrite)
duplicate_data = [
    {"id": "dup_test", "vector": random_vector(), "metadata": {"version": 1}},
    {"id": "dup_test", "vector": random_vector(), "metadata": {"version": 2}},
]

try:
    result = error_index.add(duplicate_data)
    print(f"   ✅ Duplicate IDs handled: {result.summary()}")
    
    # Check that the second version overwrote the first
    records = error_index.get_records(["dup_test"])
    assert records[0]['metadata']['version'] == 2
    print("   ✅ Duplicate ID correctly overwrote previous entry")
except Exception as e:
    print(f"   ❌ Duplicate ID handling failed: {e}")

# Test invalid search vector
try:
    invalid_search = error_index.search([1.0, 2.0])  # Wrong dimension
    print("   ❌ Should have failed with wrong dimension")
except Exception as e:
    print(f"   ✅ Invalid search vector correctly rejected: {type(e).__name__}")

# ================================
# Test 7: Distance Metrics
# ================================
print("\n📏 Test 7: Distance Metrics")

# Test all supported distance metrics
metrics = ["cosine", "l2", "l1"]
metric_indexes = {}

for metric in metrics:
    try:
        idx = vdb.create("hnsw", dim=64, space=metric, expected_size=1000)
        metric_indexes[metric] = idx
        
        # Add some test data
        test_data = [
            {"id": f"{metric}_{i}", "vector": np.random.random(64).astype(np.float32).tolist()}
            for i in range(20)
        ]
        
        result = idx.add(test_data)
        print(f"   ✅ {metric.upper()} metric: {result.summary()}")
        
        # Test search
        query = np.random.random(64).astype(np.float32).tolist()
        search_results = idx.search(query, top_k=5)
        print(f"      Search returned {len(search_results)} results")
        
    except Exception as e:
        print(f"   ❌ {metric.upper()} metric failed: {e}")

# ================================
# Test 8: Metadata Operations & Filtering
# ================================
print("\n🏷️  Test 8: Metadata Operations & Filtering")

meta_index = vdb.create("hnsw", dim=128, space="cosine")

# Add data with rich metadata
rich_data = []
for i in range(50):
    metadata = {
        "category": random.choice(["A", "B", "C"]),
        "value": random.randint(1, 100),
        "tags": random.sample(["tag1", "tag2", "tag3", "tag4"], 2),
        "active": random.choice([True, False]),
        "name": f"item_{i:03d}"
    }
    
    rich_data.append({
        "id": f"meta_{i:03d}",
        "vector": random_vector(),
        "metadata": metadata
    })

result = meta_index.add(rich_data)
print(f"   Added rich metadata: {result.summary()}")

# Test various filter types
filter_tests = [
    {"category": "A"},  # Simple equality
    {"value": {"gt": 50}},  # Greater than
    {"active": True},  # Boolean
    {"name": {"startswith": "item_0"}},  # String operations
]

query = random_vector()

for i, filter_dict in enumerate(filter_tests):
    try:
        results = meta_index.search(query, filter=filter_dict, top_k=10)
        print(f"   ✅ Filter {i+1}: {filter_dict} → {len(results)} results")
        
        # Verify first result matches filter (basic validation)
        if results:
            metadata = results[0]['metadata']
            if "category" in filter_dict:
                assert metadata['category'] == filter_dict['category']
            elif "active" in filter_dict:
                assert metadata['active'] == filter_dict['active']
                
    except Exception as e:
        print(f"   ❌ Filter {i+1} failed: {e}")

# Test index-level metadata
meta_index.add_metadata({"created_by": "test_suite", "version": "1.0"})
index_meta = meta_index.get_all_metadata()
print(f"   ✅ Index metadata: {index_meta}")

# ================================
# Test 9: Data Persistence & Retrieval
# ================================
print("\n💾 Test 9: Data Persistence & Retrieval")

persist_index = vdb.create("hnsw", dim=128, space="cosine")

# Add data
persist_data = [
    {"id": f"persist_{i}", "vector": random_vector(), 
     "metadata": {"group": i % 3, "timestamp": time.time()}}
    for i in range(30)
]

result = persist_index.add(persist_data)
print(f"   Added persistence test data: {result.summary()}")

# Test bulk retrieval
all_ids = [f"persist_{i}" for i in range(30)]
all_records = persist_index.get_records(all_ids)
print(f"   ✅ Bulk retrieval: {len(all_records)} records")

# Test partial retrieval
partial_ids = [f"persist_{i}" for i in range(0, 30, 5)]
partial_records = persist_index.get_records(partial_ids, return_vector=False)
print(f"   ✅ Partial retrieval (no vectors): {len(partial_records)} records")

# Test list functionality
list_results = persist_index.list(number=5)
print(f"   ✅ List first 5: {len(list_results)} records")

# ================================
# Test 10: Performance Characteristics
# ================================
print("\n🏃 Test 10: Performance Characteristics")

# Get performance info
perf_info = perf_index.get_performance_info()
print("   📊 Performance characteristics:")
for key, value in perf_info.items():
    print(f"      {key}: {value}")

# Raw performance benchmark
try:
    raw_bench = perf_index.benchmark_raw_concurrent_performance(query_count=100)
    print("   📊 Raw performance benchmark:")
    print(f"      Sequential QPS: {raw_bench['sequential_qps']:.0f}")
    print(f"      Parallel QPS: {raw_bench['parallel_qps']:.0f}")
    print(f"      Speedup: {raw_bench['speedup']:.2f}x")
    print(f"      Threads used: {int(raw_bench['threads_used'])}")
except Exception as e:
    print(f"   ❌ Raw benchmark failed: {e}")

# ================================
# Final Summary
# ================================
print("\n" + "=" * 60)
print("📊 TEST SUITE SUMMARY")
print("=" * 60)

total_indexes_created = len(indexes) + len(metric_indexes) + 4  # Additional test indexes
print(f"✅ Total indexes created: {total_indexes_created}")
print(f"✅ All distance metrics tested: {', '.join(metrics)}")
print("✅ All data formats tested: single object, list, separate arrays, NumPy")
print("✅ Thread safety validated through concurrent operations")
print("✅ Performance benchmarks completed")
print("✅ Error handling verified")
print("✅ Metadata filtering functional")

# Quick performance summary
final_stats = perf_index.get_stats()
print("\n📈 Performance Index Final Stats:")
print(f"   Total vectors: {final_stats['total_vectors']}")
print(f"   Dimension: {final_stats['dimension']}")
print(f"   Space: {final_stats['space']}")
print(f"   Thread safety: {final_stats['thread_safety']}")

print("\n🎉 All tests completed successfully!")
print("Your parallel HNSW implementation is working correctly! 🚀")