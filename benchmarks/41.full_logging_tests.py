#!/usr/bin/env python3
"""
ZeusDB Vector Database - Comprehensive Production Testing Suite
Tests logging integration, core functionality, and production scenarios.
"""

import os
# import sys
# import json
import tempfile
# import shutil
# import logging
# import time
# import subprocess
from pathlib import Path
# from typing import Dict, List, Any
import numpy as np

LOG_FILE_PATH = os.path.join(tempfile.gettempdir(), "test.log")

# Test the logging configuration BEFORE importing ZeusDB
def test_logging_coordination():
    """Test Python/Rust logging coordination."""
    print("🔧 Testing Logging Coordination...")
    
    # Test 1: Environment variable setting
    test_envs = [
        {"ZEUSDB_LOG_LEVEL": "debug", "ZEUSDB_LOG_FORMAT": "json"},
        {"ZEUSDB_LOG_LEVEL": "trace", "ZEUSDB_LOG_FORMAT": "human"},
        {"ZEUSDB_LOG_LEVEL": "error", "ZEUSDB_LOG_TARGET": "file", "ZEUSDB_LOG_FILE": LOG_FILE_PATH},
    ]
    
    for i, env_vars in enumerate(test_envs):
        print(f"  Test {i+1}: {env_vars}")
        
        # Set environment
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            # Import should trigger logging initialization
            import zeusdb_vector_database as zdb
            
            # Create index to trigger Rust logging
            vdb = zdb.VectorDatabase()
            index = vdb.create("hnsw", dim=128)
            
            # Add some vectors to trigger operations
            vectors = np.random.random((10, 128)).astype(np.float32)
            result = index.add({"vectors": vectors.tolist()})
            
            print(f"    ✅ Success: {result.summary()}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        finally:
            # Clean up environment
            for key in env_vars:
                os.environ.pop(key, None)
            
            # Clean up test file
            if LOG_FILE_PATH in str(env_vars.get("ZEUSDB_LOG_FILE", "")):
                try:
                    os.remove(LOG_FILE_PATH)
                except OSError:
                    pass

def test_core_functionality():
    """Test core vector database operations with logging."""
    print("\n📊 Testing Core Functionality...")
    
    try:
        import zeusdb_vector_database as zdb
        
        # Test 1: Basic HNSW creation
        print("  Test 1: Basic HNSW Index Creation")
        vdb = zdb.VectorDatabase()
        index = vdb.create("hnsw", dim=384, space="cosine", m=16, ef_construction=200)
        print(f"    ✅ Index created: {index.info()}")
        
        # Test 2: Vector addition with metadata
        print("  Test 2: Vector Addition with Metadata")
        test_data = {
            "vectors": np.random.random((100, 384)).astype(np.float32).tolist(),
            "ids": [f"doc_{i}" for i in range(100)],
            "metadatas": [{"category": f"cat_{i%5}", "value": i} for i in range(100)]
        }
        
        result = index.add(test_data)
        print(f"    ✅ Added vectors: {result.summary()}")
        
        # Test 3: Search operations
        print("  Test 3: Search Operations")
        query_vector = np.random.random(384).astype(np.float32).tolist()
        
        # Basic search
        results = index.search(query_vector, top_k=5)
        print(f"    ✅ Basic search: {len(results)} results")
        
        # Filtered search
        filter_condition = {"category": {"eq": "cat_1"}}
        filtered_results = index.search(query_vector, filter=filter_condition, top_k=3)
        print(f"    ✅ Filtered search: {len(filtered_results)} results")
        
        # Batch search
        batch_queries = np.random.random((5, 384)).astype(np.float32)
        batch_results = index.search(batch_queries, top_k=3)
        print(f"    ✅ Batch search: {len(batch_results)} result sets")
        
        # Test 4: Index statistics
        print("  Test 4: Index Statistics")
        stats = index.get_stats()
        print(f"    ✅ Stats: {stats['total_vectors']} vectors, space={stats['space']}")
        
        return index
        
    except Exception as e:
        print(f"    ❌ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_quantization_features(base_index):
    """Test Product Quantization features."""
    print("\n🗜️  Testing Quantization Features...")
    
    try:
        import zeusdb_vector_database as zdb
        
        # Test 1: PQ index creation
        print("  Test 1: PQ Index Creation")
        vdb = zdb.VectorDatabase()
        
        quantization_config = {
            'type': 'pq',
            'subvectors': 8,
            'bits': 8,
            'training_size': 5000,
            'storage_mode': 'quantized_only'
        }
        
        pq_index = vdb.create(
            "hnsw", 
            dim=384, 
            quantization_config=quantization_config,
            expected_size=10000
        )
        print(f"    ✅ PQ Index created: {pq_index.info()}")
        
        # Test 2: Training trigger
        print("  Test 2: PQ Training Process")
        training_data = {
            "vectors": np.random.random((6000, 384)).astype(np.float32).tolist(),
            "ids": [f"train_{i}" for i in range(6000)],
            "metadatas": [{"source": "training"} for _ in range(6000)]
        }
        
        print(f"    Initial training progress: {pq_index.get_training_progress():.1f}%")
        result = pq_index.add(training_data)
        print(f"    ✅ Training data added: {result.summary()}")
        print(f"    Final training progress: {pq_index.get_training_progress():.1f}%")
        print(f"    Is quantized: {pq_index.is_quantized()}")
        
        # Test 3: Quantized search
        if pq_index.is_quantized():
            print("  Test 3: Quantized Search")
            query = np.random.random(384).astype(np.float32).tolist()
            results = pq_index.search(query, top_k=5)
            print(f"    ✅ Quantized search: {len(results)} results")
            
            # Test quantization info
            quant_info = pq_index.get_quantization_info()
            if quant_info:
                print(f"    ✅ Compression ratio: {quant_info.get('compression_ratio', 'N/A')}")
        
        return pq_index
        
    except Exception as e:
        print(f"    ❌ Quantization test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_persistence_operations(index):
    """Test save/load operations."""
    print("\n💾 Testing Persistence Operations...")
    
    if not index:
        print("    ⚠️  Skipping persistence tests (no index provided)")
        return
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_index.zdb"
            
            # Test 1: Save index
            print("  Test 1: Index Save")
            index.save(str(save_path))
            print(f"    ✅ Index saved to: {save_path}")
            
            # Verify files exist
            expected_files = ["manifest.json", "config.json", "mappings.bin", "metadata.json"]
            for filename in expected_files:
                file_path = save_path / filename
                if file_path.exists():
                    print(f"    ✅ {filename}: {file_path.stat().st_size} bytes")
                else:
                    print(f"    ❌ Missing: {filename}")
            
            # Test 2: Load index
            print("  Test 2: Index Load")
            import zeusdb_vector_database as zdb
            vdb = zdb.VectorDatabase()
            loaded_index = vdb.load(str(save_path))
            print(f"    ✅ Index loaded: {loaded_index.info()}")
            
            # Test 3: Verify loaded index works
            print("  Test 3: Loaded Index Functionality")
            query = np.random.random(loaded_index.dim).astype(np.float32).tolist()
            results = loaded_index.search(query, top_k=3)
            print(f"    ✅ Search on loaded index: {len(results)} results")
            
            # Compare stats
            original_stats = index.get_stats()
            loaded_stats = loaded_index.get_stats()
            print(f"    ✅ Vector count: {original_stats['total_vectors']} -> {loaded_stats['total_vectors']}")
            
    except Exception as e:
        print(f"    ❌ Persistence test failed: {e}")
        import traceback
        traceback.print_exc()

def test_performance_benchmarks():
    """Test concurrent performance features."""
    print("\n⚡ Testing Performance Features...")
    
    try:
        import zeusdb_vector_database as zdb
        
        # Create index with test data
        vdb = zdb.VectorDatabase()
        index = vdb.create("hnsw", dim=128, expected_size=5000)
        
        # Add test vectors
        test_vectors = np.random.random((1000, 128)).astype(np.float32)
        result = index.add({"vectors": test_vectors.tolist()})
        print(f"  Setup: {result.summary()}")
        
        # Test concurrent read benchmark
        print("  Test: Concurrent Read Performance")
        bench_results = index.benchmark_concurrent_reads(query_count=100, max_threads=4)
        
        print(f"    ✅ Sequential QPS: {bench_results['sequential_qps']:.1f}")
        print(f"    ✅ Parallel QPS: {bench_results['parallel_qps']:.1f}")
        print(f"    ✅ Speedup: {bench_results['speedup']:.2f}x")
        
    except Exception as e:
        print(f"    ❌ Performance test failed: {e}")

def test_error_conditions():
    """Test error handling and edge cases."""
    print("\n🛡️  Testing Error Handling...")
    
    try:
        import zeusdb_vector_database as zdb
        vdb = zdb.VectorDatabase()
        
        # Test 1: Invalid configurations
        print("  Test 1: Invalid Configurations")
        
        test_cases = [
            ("Invalid dimension", {"dim": 0}),
            ("Invalid space", {"dim": 128, "space": "invalid"}),
            ("Invalid m", {"dim": 128, "m": 1000}),
            ("Invalid quantization", {"dim": 128, "quantization_config": {"type": "invalid"}}),
        ]
        
        for name, config in test_cases:
            try:
                index = vdb.create("hnsw", **config)
                print(f"    ❌ {name}: Should have failed but didn't")
            except (ValueError, RuntimeError) as e:
                print(f"    ✅ {name}: Properly caught - {type(e).__name__}")
            except Exception as e:
                print(f"    ⚠️  {name}: Unexpected error - {e}")
        
        # Test 2: Invalid vector operations
        print("  Test 2: Invalid Vector Operations")
        index = vdb.create("hnsw", dim=128)
        
        invalid_ops = [
            ("Empty vector", []),
            ("Wrong dimension", [1.0, 2.0, 3.0]),  # Only 3 dims, need 128
            ("Invalid values", [float('inf')] * 128),
        ]
        
        for name, vector in invalid_ops:
            try:
                result = index.add({"vectors": [vector]})
                if result.total_errors > 0:
                    print(f"    ✅ {name}: Properly handled in result.errors")
                else:
                    print(f"    ❌ {name}: Should have produced errors")
            except Exception as e:
                print(f"    ✅ {name}: Exception caught - {type(e).__name__}")
        
    except Exception as e:
        print(f"    ❌ Error handling test failed: {e}")

def test_logging_output():
    """Test actual logging output."""
    print("\n📝 Testing Logging Output...")
    
    # Set debug level to see more logs
    os.environ["ZEUSDB_LOG_LEVEL"] = "debug"
    os.environ["ZEUSDB_LOG_FORMAT"] = "human"
    
    try:
        import zeusdb_vector_database as zdb
        
        print("  Performing operations to generate logs...")
        vdb = zdb.VectorDatabase()
        index = vdb.create("hnsw", dim=64)
        
        # This should generate various log messages
        vectors = np.random.random((50, 64)).astype(np.float32)
        result = index.add({"vectors": vectors.tolist()})
        
        query = np.random.random(64).astype(np.float32)
        results = index.search(query, top_k=5)
        
        print(f"  ✅ Operations completed: {result.summary()}, {len(results)} search results")
        print("  Check console output above for structured log messages")
        
    finally:
        # Clean up
        os.environ.pop("ZEUSDB_LOG_LEVEL", None)
        os.environ.pop("ZEUSDB_LOG_FORMAT", None)

def main():
    """Run the complete test suite."""
    print("🚀 ZeusDB Vector Database - Production Test Suite")
    print("=" * 60)
    
    # Test logging coordination first (before imports)
    test_logging_coordination()
    
    # Test core functionality
    index = test_core_functionality()
    
    # Test advanced features
    pq_index = test_quantization_features(index)
    
    # Test persistence
    test_persistence_operations(index or pq_index)
    
    # Test performance
    test_performance_benchmarks()
    
    # Test error handling
    test_error_conditions()
    
    # Test logging output
    test_logging_output()
    
    print("\n" + "=" * 60)
    print("🎯 Test Suite Complete!")
    print("\nNext Steps:")
    print("1. Run this test suite: python test_zeusdb.py")
    print("2. Check for any failures or warnings")
    print("3. Monitor log output for proper structured logging")
    print("4. Test in production-like environment")

if __name__ == "__main__":
    main()