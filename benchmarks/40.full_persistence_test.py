#!/usr/bin/env python3
"""
Enhanced Comprehensive Persistence Test Suite for ZeusDB
Tests multiple scenarios, scales, and configurations
"""

from zeusdb_vector_database import VectorDatabase
import numpy as np
import os
import shutil
import time
#from typing import Dict, Any, List

class PersistenceTestSuite:
    def __init__(self, base_dir: str = "persistence_tests"):
        self.base_dir = base_dir
        self.results = []
        
    def setup(self):
        """Clean up and prepare test environment"""
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"🧹 Test environment prepared: {self.base_dir}")
    
    def run_test(self, test_name: str, test_func, **kwargs):
        """Run a single test with error handling and timing"""
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = test_func(**kwargs)
            duration = time.time() - start_time
            self.results.append({
                'test': test_name,
                'status': 'PASS',
                'duration': duration,
                'details': result
            })
            print(f"✅ PASSED in {duration:.2f}s")
            return True
        except Exception as e:
            duration = time.time() - start_time

            # 🔥 ENHANCED ERROR REPORTING:
            error_msg = str(e)
            print(f"❌ FAILED in {duration:.2f}s")
            print(f"💥 Error: {error_msg}")

            # Show full traceback for debugging
            import traceback
            print("📋 Full traceback:")
            traceback.print_exc()


            self.results.append({
                'test': test_name,
                'status': 'FAIL',
                'duration': duration,
                # 'error': str(e)
                'error': error_msg
            })
            # print(f"❌ FAILED in {duration:.2f}s: {e}")
            # import traceback
            # traceback.print_exc()
            return False
    
    def test_basic_persistence(self, num_vectors=100, dim=64):
        """Basic persistence test (your current test)"""
        test_dir = os.path.join(self.base_dir, "basic_test.zdb")
        
        # Create and populate index
        vdb = VectorDatabase()
        index = vdb.create("hnsw", dim=dim, space="cosine", m=16, ef_construction=200)
        
        np.random.seed(42)
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
        test_data = {
            'vectors': vectors.tolist(),
            'ids': [f'vec_{i}' for i in range(num_vectors)],
            'metadatas': [{'index': i, 'category': f'cat_{i % 5}'} for i in range(num_vectors)]
        }
        
        add_result = index.add(test_data)
        original_stats = index.get_stats()
        
        # Test search
        test_query = vectors[0]
        original_results = index.search(test_query.tolist(), top_k=5)
        
        # Save
        index.save(test_dir)
        
        # Load and verify
        loaded_index = vdb.load(test_dir)
        loaded_stats = loaded_index.get_stats()
        loaded_results = loaded_index.search(test_query.tolist(), top_k=5)
        
        # Verify
        assert loaded_stats['total_vectors'] == original_stats['total_vectors']
        assert len(loaded_results) == len(original_results)
        
        return {
            'vectors': num_vectors,
            'original_vectors': original_stats['total_vectors'],
            'loaded_vectors': loaded_stats['total_vectors'],
            'search_results': len(loaded_results)
        }
    
    def test_pq_untrained_persistence(self, num_vectors=800, dim=64):
        """Test persistence of PQ index before training threshold"""
        test_dir = os.path.join(self.base_dir, "pq_untrained.zdb")
        
        vdb = VectorDatabase()
        quantization_config = {
            'type': 'pq',
            'subvectors': 8,
            'bits': 8,
            'training_size': 1000  # Higher than num_vectors
        }
        index = vdb.create("hnsw", dim=dim, quantization_config=quantization_config)
        
        # Add vectors (less than training_size)
        np.random.seed(42)
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
        test_data = {
            'vectors': vectors.tolist(),
            'ids': [f'vec_{i}' for i in range(num_vectors)],
            'metadatas': [{'index': i} for i in range(num_vectors)]
        }
        
        index.add(test_data)
        training_progress = index.get_training_progress()
        assert training_progress < 100.0, "Training should not be complete"
        
        # Save and load
        index.save(test_dir)
        loaded_index = vdb.load(test_dir)
        
        # Verify
        loaded_progress = loaded_index.get_training_progress()

        # 🔥 ADD THESE DEBUG LINES:
        print(f"  🔍 ORIGINAL training progress: {training_progress:.3f}%")
        print(f"  🔍 LOADED training progress: {loaded_progress:.3f}%")
        print(f"  🔍 Difference: {abs(loaded_progress - training_progress):.3f}")
        print(f"  🔍 Test passes? {abs(loaded_progress - training_progress) < 1.0}")

        assert abs(loaded_progress - training_progress) < 1.0
        assert not loaded_index.can_use_quantization()
        
        return {
            'training_progress': training_progress,
            'loaded_progress': loaded_progress,
            'quantization_active': loaded_index.can_use_quantization()
        }
    
    def test_pq_trained_persistence(self, num_vectors=2000, dim=64):
        """Test persistence of fully trained PQ index"""
        test_dir = os.path.join(self.base_dir, "pq_trained.zdb")
        
        vdb = VectorDatabase()
        quantization_config = {
            'type': 'pq',
            'subvectors': 8,
            'bits': 8,
            'training_size': 1000,
            'storage_mode': 'quantized_only'
        }
        index = vdb.create("hnsw", dim=dim, quantization_config=quantization_config)
        
        # Add enough vectors to trigger training
        np.random.seed(42)
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
        test_data = {
            'vectors': vectors.tolist(),
            'ids': [f'vec_{i}' for i in range(num_vectors)],
            'metadatas': [{'index': i} for i in range(num_vectors)]
        }
        
        index.add(test_data)
        
        # Verify training completed
        assert index.can_use_quantization(), "PQ should be trained"
        assert index.is_quantized(), "Index should be using quantization"
        
        # Test search before save
        test_query = vectors[0]
        original_results = index.search(test_query.tolist(), top_k=5)
        
        # Save and load
        index.save(test_dir)
        loaded_index = vdb.load(test_dir)
        
        # Verify quantization state
        assert loaded_index.can_use_quantization()
        
        # Test search after load
        loaded_results = loaded_index.search(test_query.tolist(), top_k=5)
        
        return {
            'vectors': num_vectors,
            'quantization_trained': loaded_index.can_use_quantization(),
            'quantization_active': loaded_index.is_quantized(),
            'search_results': len(loaded_results)
        }
    
    def test_different_distance_metrics(self, num_vectors=500, dim=32):
        """Test persistence with different distance metrics"""
        results = {}
        
        for space in ['cosine', 'l2', 'l1']:
            test_dir = os.path.join(self.base_dir, f"distance_{space}.zdb")
            
            vdb = VectorDatabase()
            index = vdb.create("hnsw", dim=dim, space=space)
            
            np.random.seed(42)
            vectors = np.random.random((num_vectors, dim)).astype(np.float32)
            test_data = {
                'vectors': vectors.tolist(),
                'ids': [f'vec_{i}' for i in range(num_vectors)]
            }
            
            index.add(test_data)
            index.save(test_dir)
            
            # Load and verify
            loaded_index = vdb.load(test_dir)
            assert loaded_index.get_space() == space
            
            results[space] = {
                'space': loaded_index.get_space(),
                'vectors': loaded_index.get_vector_count()
            }
        
        return results
    
    def test_large_scale_persistence(self, num_vectors=10000, dim=128):
        """Test with larger scale data"""
        test_dir = os.path.join(self.base_dir, "large_scale.zdb")
        
        vdb = VectorDatabase()
        quantization_config = {
            'type': 'pq',
            'subvectors': 16,
            'bits': 8,
            'training_size': 2000
        }
        index = vdb.create("hnsw", dim=dim, quantization_config=quantization_config)
        
        # Add vectors in batches to test incremental building
        batch_size = 1000
        np.random.seed(42)
        
        for batch_start in range(0, num_vectors, batch_size):
            batch_end = min(batch_start + batch_size, num_vectors)
            batch_vectors = np.random.random((batch_end - batch_start, dim)).astype(np.float32)
            
            test_data = {
                'vectors': batch_vectors.tolist(),
                'ids': [f'vec_{i}' for i in range(batch_start, batch_end)],
                'metadatas': [{'batch': batch_start // batch_size, 'index': i} 
                            for i in range(batch_start, batch_end)]
            }
            
            index.add(test_data)
            print(f"  Added batch {batch_start//batch_size + 1}/{(num_vectors-1)//batch_size + 1}")
        
        original_stats = index.get_stats()
        
        # Save and load
        print("  Saving large index...")
        save_start = time.time()
        index.save(test_dir)
        save_time = time.time() - save_start
        
        print("  Loading large index...")
        load_start = time.time()
        loaded_index = vdb.load(test_dir)
        load_time = time.time() - load_start
        
        loaded_stats = loaded_index.get_stats()
        
        return {
            'vectors': num_vectors,
            'save_time': save_time,
            'load_time': load_time,
            'original_vectors': original_stats['total_vectors'],
            'loaded_vectors': loaded_stats['total_vectors'],
            'quantization_trained': loaded_index.can_use_quantization()
        }
    
    def test_edge_cases(self):
        """Test various edge cases"""
        results = {}
        
        # Test 1: Empty index
        test_dir = os.path.join(self.base_dir, "empty.zdb")
        vdb = VectorDatabase()
        index = vdb.create("hnsw", dim=32)
        index.save(test_dir)
        loaded_index = vdb.load(test_dir)
        results['empty'] = loaded_index.get_vector_count() == 0
        
        # Test 2: Single vector
        test_dir = os.path.join(self.base_dir, "single.zdb")
        index = vdb.create("hnsw", dim=32)
        index.add({'vectors': [[1.0] * 32], 'ids': ['single']})
        index.save(test_dir)
        loaded_index = vdb.load(test_dir)
        results['single'] = loaded_index.get_vector_count() == 1
        
        return results
    
    def test_storage_modes(self, num_vectors=1500, dim=64):
        """Test different PQ storage modes"""
        results = {}
        
        for storage_mode in ['quantized_only', 'quantized_with_raw']:
            test_dir = os.path.join(self.base_dir, f"storage_{storage_mode}.zdb")
            
            vdb = VectorDatabase()
            quantization_config = {
                'type': 'pq',
                'subvectors': 8,
                'bits': 8,
                'training_size': 1000,
                'storage_mode': storage_mode
            }
            index = vdb.create("hnsw", dim=dim, quantization_config=quantization_config)
            
            np.random.seed(42)
            vectors = np.random.random((num_vectors, dim)).astype(np.float32)
            test_data = {
                'vectors': vectors.tolist(),
                'ids': [f'vec_{i}' for i in range(num_vectors)]
            }
            
            index.add(test_data)
            
            # Test vector retrieval before save
            original_record = index.get_records(['vec_0'], return_vector=True)[0]
            has_original_vector = 'vector' in original_record
            
            index.save(test_dir)
            loaded_index = vdb.load(test_dir)
            
            # Test vector retrieval after load
            loaded_record = loaded_index.get_records(['vec_0'], return_vector=True)[0]
            has_loaded_vector = 'vector' in loaded_record
            
            results[storage_mode] = {
                'trained': loaded_index.can_use_quantization(),
                'original_has_vector': has_original_vector,
                'loaded_has_vector': has_loaded_vector,
                'vectors_match': (has_original_vector and has_loaded_vector and 
                                np.allclose(original_record['vector'], loaded_record['vector'], rtol=1e-3))
            }
        
        return results
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting Enhanced Persistence Test Suite")
        print(f"📁 Test directory: {self.base_dir}")
        
        self.setup()
        
        # Run all tests
        tests = [
            ("Basic Persistence", self.test_basic_persistence, {}),
            ("PQ Untrained State", self.test_pq_untrained_persistence, {}),
            ("PQ Trained State", self.test_pq_trained_persistence, {}),
            ("Distance Metrics", self.test_different_distance_metrics, {}),
            ("Storage Modes", self.test_storage_modes, {}),
            ("Edge Cases", self.test_edge_cases, {}),
            ("Large Scale", self.test_large_scale_persistence, {"num_vectors": 5000})  # Reduced for faster testing
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func, kwargs in tests:
            if self.run_test(test_name, test_func, **kwargs):
                passed += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for result in self.results:
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {result['test']}: {result['status']} ({result['duration']:.2f}s)")
            if result['status'] == 'FAIL':
                print(f"    Error: {result['error']}")
        
        return passed == total

if __name__ == "__main__":
    suite = PersistenceTestSuite()
    success = suite.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Persistence system is production ready!")
    else:
        print("\n⚠️  Some tests failed. Review results above.")
        exit(1)