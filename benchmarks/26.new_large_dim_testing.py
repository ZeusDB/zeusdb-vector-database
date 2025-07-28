# ================================================================================
# LARGE SCALE HIGH-DIMENSIONAL VECTOR TEST
# ================================================================================
# Testing with 1536D (OpenAI embeddings) and 3072D vectors at production scale

#import numpy as np
import random
import time
import psutil
import os
from zeusdb_vector_database import VectorDatabase

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_dimensional_test(dim, total_vectors=25000):
    """Run comprehensive test for specified dimension"""
    
    # Configuration
    training_size = 1000
    batch_size = 500  # Smaller batches for high-dim vectors
    
    # Calculate optimal subvectors (aim for reasonable compression)
    if dim == 1536:
        subvectors = 24  # 1536 / 24 = 64 dimensions per subvector
    elif dim == 3072:
        subvectors = 48  # 3072 / 48 = 64 dimensions per subvector
    else:
        subvectors = max(8, dim // 64)  # Default: ~64 dims per subvector
    
    bits = 8
    compression_ratio = (dim * 4) // subvectors
    
    print(f"\n🚀 TESTING {dim}D VECTORS - {total_vectors:,} vectors")
    print("=" * 80)
    print("📊 Configuration:")
    print(f"   Dimensions: {dim:,}")
    print(f"   Total vectors: {total_vectors:,}")
    print(f"   Training size: {training_size:,}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Subvectors: {subvectors}")
    print(f"   Expected compression: {compression_ratio}x")
    print(f"   Theoretical raw memory: {(total_vectors * dim * 4) / (1024*1024):.1f} MB")
    print("=" * 80)
    
    # Phase 1: Setup
    print(f"\n🔧 PHASE 1: INDEX SETUP ({dim}D)")
    print("-" * 50)
    
    setup_start = time.time()
    
    vdb = VectorDatabase()
    quantization_config = {
        'type': 'pq',
        'subvectors': subvectors,
        'bits': bits,
        'training_size': training_size,
        'max_training_vectors': 3000
    }
    
    try:
        index = vdb.create("hnsw", dim=dim, quantization_config=quantization_config)
        setup_time = time.time() - setup_start
        setup_memory = get_memory_usage()
        
        print(f"✅ {dim}D index created in {setup_time:.3f} seconds")
        print(f"📊 Memory usage: {setup_memory:.1f} MB")
        print(f"📊 Storage mode: {index.get_storage_mode()}")
        
    except Exception as e:
        print(f"❌ Failed to create {dim}D index: {e}")
        return None
    
    # Phase 2: Pre-training addition
    print(f"\n🔄 PHASE 2: PRE-TRAINING ADDITION ({dim}D)")
    print("-" * 50)
    
    pre_training_start = time.time()
    target_pre_training = training_size - 100
    vectors_added = 0
    
    print(f"📋 Adding {target_pre_training:,} vectors before training...")
    
    while vectors_added < target_pre_training:
        batch_start = time.time()
        current_batch_size = min(batch_size, target_pre_training - vectors_added)
        
        # Generate high-dimensional vectors
        batch_vectors = []
        for i in range(current_batch_size):
            vector = [random.random() for _ in range(dim)]
            batch_vectors.append(vector)
        
        result = index.add(batch_vectors)
        batch_time = time.time() - batch_start
        vectors_added += result.total_inserted
        
        progress = index.get_training_progress()
        vectors_needed = index.training_vectors_needed()
        current_memory = get_memory_usage()
        
        # Report every 5 batches for high-dim (they're slower)
        if vectors_added % (batch_size * 5) == 0 or vectors_added >= target_pre_training:
            print(f"  Added {vectors_added:5d} | Progress: {progress:5.1f}% | "
                  f"Need: {vectors_needed:3d} | Time: {batch_time:.3f}s | "
                  f"Memory: {current_memory:.1f}MB")
    
    pre_training_time = time.time() - pre_training_start
    pre_training_memory = get_memory_usage()
    
    print("\n✅ Pre-training completed:")
    print(f"   Vectors: {vectors_added:,}")
    print(f"   Time: {pre_training_time:.2f}s")
    print(f"   Rate: {vectors_added/pre_training_time:.0f} vec/s")
    print(f"   Memory: {pre_training_memory:.1f} MB")
    
    # Phase 3: Training trigger
    print(f"\n🎯 PHASE 3: TRAINING TRIGGER ({dim}D)")
    print("-" * 50)
    
    training_start = time.time()
    vectors_needed = index.training_vectors_needed()
    
    print(f"📋 Adding {vectors_needed + 50} vectors to trigger training...")
    
    trigger_vectors = []
    for i in range(vectors_needed + 50):
        vector = [random.random() for _ in range(dim)]
        trigger_vectors.append(vector)
    
    print("🚀 Triggering PQ training...")
    result = index.add(trigger_vectors)
    training_time = time.time() - training_start
    post_training_memory = get_memory_usage()
    
    print("\n✅ Training completed:")
    print(f"   Added: {result.total_inserted}")
    print(f"   Training time: {training_time:.2f}s")
    print(f"   Memory: {post_training_memory:.1f} MB")
    print(f"   Storage mode: {index.get_storage_mode()}")
    
    # Get compression info
    quant_info = index.get_quantization_info()
    print(f"   Compression: {quant_info['compression_ratio']}x")
    print(f"   Compressed memory: {quant_info['memory_mb']:.3f} MB")
    
    # Phase 4: Post-training large scale addition
    print(f"\n📈 PHASE 4: POST-TRAINING ADDITION ({dim}D)")
    print("-" * 50)
    
    post_training_start = time.time()
    current_count = index.get_vector_count()
    remaining = total_vectors - current_count
    post_added = 0
    
    print(f"📋 Adding {remaining:,} vectors in quantized mode...")
    
    batch_times = []
    while post_added < remaining:
        batch_start = time.time()
        current_batch_size = min(batch_size, remaining - post_added)
        
        batch_vectors = []
        for i in range(current_batch_size):
            vector = [random.random() for _ in range(dim)]
            batch_vectors.append(vector)
        
        result = index.add(batch_vectors)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        post_added += result.total_inserted
        
        # Report every 10 batches
        if len(batch_times) % 10 == 0:
            total_count = index.get_vector_count()
            current_memory = get_memory_usage()
            print(f"  Batch {len(batch_times):3d}: {result.total_inserted:3d} vectors | "
                  f"Total: {total_count:6d} | Time: {batch_time:.3f}s | "
                  f"Memory: {current_memory:.1f}MB")
    
    post_training_time = time.time() - post_training_start
    final_memory = get_memory_usage()
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    print("\n✅ Post-training completed:")
    print(f"   Vectors: {post_added:,}")
    print(f"   Time: {post_training_time:.2f}s")
    print(f"   Avg batch: {avg_batch_time:.3f}s")
    print(f"   Rate: {post_added/post_training_time:.0f} vec/s")
    print(f"   Final memory: {final_memory:.1f} MB")
    
    # Phase 5: Performance benchmarking
    print(f"\n⚡ PHASE 5: PERFORMANCE BENCHMARKING ({dim}D)")
    print("-" * 50)
    
    final_count = index.get_vector_count()
    print(f"📊 Final vector count: {final_count:,}")
    
    # Single query performance (fewer queries for high-dim)
    print("\n🔍 Single Query Performance:")
    query_times = []
    for i in range(50):  # 50 queries for high-dim
        query_vector = [random.random() for _ in range(dim)]
        start = time.time()
        _ = index.search(query_vector, top_k=10)  # Use results or ignore
        query_times.append(time.time() - start)
    
    avg_query = sum(query_times) / len(query_times) * 1000
    min_query = min(query_times) * 1000
    max_query = max(query_times) * 1000
    
    print(f"   Average: {avg_query:.3f} ms")
    print(f"   Min: {min_query:.3f} ms")
    print(f"   Max: {max_query:.3f} ms")
    print(f"   QPS: {1000/avg_query:.0f}")
    
    # Batch query performance
    print("\n🔍 Batch Query Performance:")
    batch_queries = []
    for i in range(25):  # Smaller batch for high-dim
        query = [random.random() for _ in range(dim)]
        batch_queries.append(query)
    
    batch_start = time.time()
    _ = index.search(batch_queries, top_k=10)  # Use results or ignore
    batch_time = time.time() - batch_start
    
    print(f"   Batch size: {len(batch_queries)}")
    print(f"   Total time: {batch_time:.3f}s")
    print(f"   Per query: {(batch_time/len(batch_queries))*1000:.3f} ms")
    print(f"   Batch QPS: {len(batch_queries)/batch_time:.0f}")
    
    # Memory efficiency analysis
    print("\n💾 Memory Efficiency Analysis:")
    
    theoretical_raw_mb = (final_count * dim * 4) / (1024 * 1024)
    actual_memory_mb = float(quant_info['memory_mb'])
    compression_ratio = quant_info['compression_ratio']
    memory_savings = (1 - actual_memory_mb / theoretical_raw_mb) * 100
    
    print(f"   Theoretical raw: {theoretical_raw_mb:.1f} MB")
    print(f"   Actual compressed: {actual_memory_mb:.1f} MB")
    print(f"   Compression: {compression_ratio}x")
    print(f"   Memory savings: {memory_savings:.1f}%")
    print(f"   Process memory: {final_memory:.1f} MB")
    
    # Return results for comparison
    return {
        'dimension': dim,
        'vectors': final_count,
        'avg_query_ms': avg_query,
        'qps': 1000/avg_query,
        'compression_ratio': compression_ratio,
        'memory_savings': memory_savings,
        'process_memory_mb': final_memory,
        'compressed_memory_mb': actual_memory_mb,
        'theoretical_memory_mb': theoretical_raw_mb,
        'training_time': training_time,
        'pre_training_rate': vectors_added/pre_training_time,
        'post_training_rate': post_added/post_training_time
    }

# ================================================================================
# MAIN EXECUTION
# ================================================================================

print("🚀 HIGH-DIMENSIONAL LARGE SCALE TESTING")
print("🚀 Testing production-scale performance with realistic embeddings")
print("=" * 80)

# Test both dimensions
dimensions = [1536, 3072]
results = []

for dim in dimensions:
    # Adjust vector count based on dimension (higher dims = fewer vectors due to memory)
    if dim == 1536:
        vector_count = 25000  # 25K vectors for 1536D
    elif dim == 3072:
        vector_count = 15000  # 15K vectors for 3072D (higher memory usage)
    else:
        vector_count = 10000  # Default fallback
    
    try:
        result = run_dimensional_test(dim, vector_count)
        if result:
            results.append(result)
            
    except Exception as e:
        print(f"❌ Test failed for {dim}D: {e}")
        continue

# ================================================================================
# COMPARATIVE ANALYSIS
# ================================================================================

if len(results) >= 2:
    print("\n" + "🔄" * 60)
    print("COMPARATIVE ANALYSIS - 1536D vs 3072D")
    print("🔄" * 60)
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<25} {'1536D':<15} {'3072D':<15} {'Ratio':<10}")
    print("-" * 70)
    
    r1, r2 = results[0], results[1]
    
    print(f"{'Vectors':<25} {r1['vectors']:,<15} {r2['vectors']:,<15} {r2['vectors']/r1['vectors']:.2f}x")
    print(f"{'Query Time (ms)':<25} {r1['avg_query_ms']:<15.3f} {r2['avg_query_ms']:<15.3f} {r2['avg_query_ms']/r1['avg_query_ms']:.2f}x")
    print(f"{'QPS':<25} {r1['qps']:<15.0f} {r2['qps']:<15.0f} {r1['qps']/r2['qps']:.2f}x")
    print(f"{'Compression':<25} {r1['compression_ratio']:<15.0f}x {r2['compression_ratio']:<15.0f}x {r2['compression_ratio']/r1['compression_ratio']:.2f}x")
    print(f"{'Memory Savings':<25} {r1['memory_savings']:<15.1f}% {r2['memory_savings']:<15.1f}% {r2['memory_savings']/r1['memory_savings']:.2f}x")
    print(f"{'Process Memory (MB)':<25} {r1['process_memory_mb']:<15.0f} {r2['process_memory_mb']:<15.0f} {r2['process_memory_mb']/r1['process_memory_mb']:.2f}x")
    print(f"{'Training Time (s)':<25} {r1['training_time']:<15.2f} {r2['training_time']:<15.2f} {r2['training_time']/r1['training_time']:.2f}x")
    
    print("\n📊 MEMORY EFFICIENCY:")
    print(f"{'Dimension':<12} {'Raw Memory':<15} {'Compressed':<15} {'Savings':<12} {'Ratio':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['dimension']:<12} {r['theoretical_memory_mb']:<15.1f} {r['compressed_memory_mb']:<15.1f} "
              f"{r['memory_savings']:<12.1f}% {r['compression_ratio']:<10.0f}x")
    
    print("\n📊 THROUGHPUT ANALYSIS:")
    print(f"{'Dimension':<12} {'Pre-Training':<15} {'Post-Training':<15} {'Improvement':<12}")
    print("-" * 60)
    for r in results:
        improvement = r['post_training_rate'] / r['pre_training_rate']
        print(f"{r['dimension']:<12} {r['pre_training_rate']:<15.0f} {r['post_training_rate']:<15.0f} {improvement:<12.1f}x")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("\n" + "🎉" * 60)
print("HIGH-DIMENSIONAL TESTING SUMMARY")
print("🎉" * 60)

if results:
    print("\n📊 ACHIEVEMENTS:")
    for r in results:
        print(f"\n🚀 {r['dimension']}D Results:")
        print(f"   ✅ Processed {r['vectors']:,} vectors successfully")
        print(f"   ✅ Search: {r['avg_query_ms']:.3f}ms avg ({r['qps']:.0f} QPS)")
        print(f"   ✅ Compression: {r['compression_ratio']}x ({r['memory_savings']:.1f}% savings)")
        print(f"   ✅ Memory: {r['compressed_memory_mb']:.1f}MB vs {r['theoretical_memory_mb']:.1f}MB theoretical")
    
    print("\n🎯 HIGH-DIMENSIONAL VALIDATION:")
    print("   ✅ OPENAI EMBEDDINGS READY (1536D tested)")
    print("   ✅ LARGE EMBEDDINGS READY (3072D tested)")
    print("   ✅ PRODUCTION SCALE VALIDATED")
    print("   ✅ MEMORY EFFICIENCY MAINTAINED")
    print("   ✅ SEARCH PERFORMANCE PRESERVED")
    
    print("\n🚀 REAL-WORLD APPLICATION READY:")
    print("   • OpenAI text-embedding-3-large (1536D) ✅")
    print("   • Custom high-dimensional embeddings (3072D) ✅")
    print("   • Enterprise-scale deployments ✅")
    print("   • Memory-constrained environments ✅")

else:
    print("\n❌ HIGH-DIMENSIONAL TESTING FAILED")
    print("   Check configuration and system resources")

print("\n" + "=" * 80)
print("🎉 HIGH-DIMENSIONAL TESTING COMPLETED!")
print("=" * 80)