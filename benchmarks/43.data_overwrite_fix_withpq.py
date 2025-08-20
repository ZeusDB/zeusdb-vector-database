#!/usr/bin/env python3
"""
Comprehensive test script to verify that the ZeusDB overwrite bug fix works 
correctly with Product Quantization (PQ) in all storage modes.

Tests all PQ scenarios:
1. Raw storage (no PQ)
2. Raw storage + training collection
3. PQ quantized_only mode
4. PQ quantized_with_raw mode
"""

import zeusdb_vector_database as zdb
import numpy as np
import time


def test_pq_overwrite_scenarios():
    """Test overwrite behavior across all PQ states and storage modes."""
    print("🧪 Testing ZeusDB PQ overwrite scenarios...")
    
    # Test scenarios in order of complexity
    scenarios = [
        ("No PQ (Raw only)", None),
        ("PQ quantized_only", {
            'type': 'pq',
            'subvectors': 4,
            'bits': 8,
            'storage_mode': 'quantized_only'
        }),
        ("PQ quantized_with_raw", {
            'type': 'pq', 
            'subvectors': 4,
            'bits': 8,
            'storage_mode': 'quantized_with_raw'
        })
    ]
    
    for scenario_name, quantization_config in scenarios:
        print(f"\n{'='*60}")
        print(f"🎯 TESTING: {scenario_name}")
        print(f"{'='*60}")
        
        test_single_pq_scenario(scenario_name, quantization_config)
    
    print("\n🎉 ALL PQ SCENARIOS PASSED!")


def test_single_pq_scenario(scenario_name, quantization_config):
    """Test overwrite behavior for a single PQ scenario."""
    
    # Create index with PQ configuration
    vdb = zdb.VectorDatabase()
    index = vdb.create(
        index_type="hnsw",
        dim=8,  # Use 8 dimensions to work with 4 subvectors
        space="cosine",
        m=16,
        ef_construction=200,
        expected_size=1000,
        quantization_config=quantization_config
    )
    
    print(f"\n📋 Index Info: {index.info()}")
    
    # Test vectors (need enough for PQ training if applicable)
    base_vectors = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    
    # Add training vectors if PQ is enabled
    if quantization_config:
        print("\n📚 Adding training vectors for PQ...")
        training_vectors = []
        training_ids = []
        training_metadata = []
        
        # Generate enough vectors for training (need 5000+ for PQ)
        np.random.seed(42)  # For reproducible results
        for i in range(5500):
            # Create diverse training vectors
            vector = np.random.normal(0, 1, 8).tolist()
            training_vectors.append(vector)
            training_ids.append(f"training_{i}")
            training_metadata.append({"type": "training", "index": i})
        
        # Add training vectors
        result = index.add({
            "vectors": training_vectors,
            "ids": training_ids,
            "metadatas": training_metadata
        }, overwrite=True)
        
        print(f"   Added {result.total_inserted} training vectors")
        
        # Wait for PQ training to complete
        print("   Waiting for PQ training...")
        max_wait = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if index.can_use_quantization():
                print(f"   ✅ PQ training completed after {time.time() - start_time:.1f}s")
                break
            time.sleep(0.5)
        else:
            print(f"   ⚠️  PQ training not completed after {max_wait}s")
        
        # Show quantization info
        quant_info = index.get_quantization_info()
        if quant_info:
            print(f"   PQ Info: {quant_info}")
    
    print(f"\n📊 Current storage mode: {index.get_storage_mode()}")
    print(f"   Vector count: {index.get_vector_count()}")
    print(f"   Can use quantization: {index.can_use_quantization()}")
    print(f"   Is quantized: {index.is_quantized()}")
    
    # Now test the overwrite behavior
    print("\n🔄 Testing overwrite behavior...")
    
    # Step 1: Add initial test documents
    print("   Step 1: Adding initial test documents...")
    result1 = index.add({
        "vectors": base_vectors[:3],
        "ids": ["test1", "test2", "test3"],
        "metadatas": [
            {"text": "first doc", "version": 1},
            {"text": "second doc", "version": 1}, 
            {"text": "third doc", "version": 1}
        ]
    }, overwrite=True)
    
    print(f"      Initial add: ✅ {result1.total_inserted} inserted, ❌ {result1.total_errors} errors")
    
    # Step 2: Search to establish baseline
    print("   Step 2: Baseline search...")
    search_results = index.search(base_vectors[0], top_k=10)
    test_doc_results = [r for r in search_results if r['id'].startswith('test')]
    print(f"      Found {len(test_doc_results)} test documents")
    
    initial_ids = set(r['id'] for r in test_doc_results)
    print(f"      Initial test IDs: {sorted(initial_ids)}")
    
    # Step 3: Overwrite one document
    print("   Step 3: Overwriting test1...")
    updated_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Different vector
    result2 = index.add({
        "vectors": [updated_vector],
        "ids": ["test1"],
        "metadatas": [{"text": "UPDATED first doc", "version": 2}]
    }, overwrite=True)
    
    print(f"      Overwrite result: ✅ {result2.total_inserted} inserted, ❌ {result2.total_errors} errors")
    
    # Step 4: Search after overwrite and check for duplicates
    print("   Step 4: Post-overwrite duplicate check...")
    search_results_after = index.search(base_vectors[0], top_k=20)  # Search more broadly
    test_doc_results_after = [r for r in search_results_after if r['id'].startswith('test')]
    
    print(f"      Found {len(test_doc_results_after)} test documents after overwrite")
    
    # Count occurrences of each test ID
    id_counts = {}
    for result in test_doc_results_after:
        id_counts[result['id']] = id_counts.get(result['id'], 0) + 1
    
    print(f"      Test ID occurrence counts: {id_counts}")
    
    # Step 5: Verify no duplicates
    print("   Step 5: Duplicate verification...")
    duplicate_ids = [id_val for id_val, count in id_counts.items() if count > 1]
    
    if duplicate_ids:
        print(f"      ❌ FOUND DUPLICATES: {duplicate_ids}")
        print("      Search results:")
        for i, result in enumerate(test_doc_results_after, 1):
            print(f"         {i}. ID: {result['id']}, Score: {result['score']:.4f}")
        raise AssertionError(f"Duplicate IDs found in {scenario_name}: {duplicate_ids}")
    else:
        print("      ✅ No duplicates found")
    
    # Step 6: Verify updated content
    print("   Step 6: Content verification...")
    test1_records = index.get_records("test1", return_vector=True)
    if test1_records:
        metadata = test1_records[0]['metadata']
        if metadata.get('version') == 2 and 'UPDATED' in metadata.get('text', ''):
            print("      ✅ test1 content properly updated")
        else:
            print(f"      ❌ test1 content not updated: {metadata}")
            raise AssertionError(f"Content not updated in {scenario_name}")
    else:
        print("      ❌ test1 not found after update")
        raise AssertionError(f"Document not found after update in {scenario_name}")
    
    # Step 7: Multiple overwrites test
    print("   Step 7: Multiple overwrites test...")
    for i in range(3):
        result = index.add({
            "vectors": [base_vectors[4]],  # Use 5th vector
            "ids": ["multi_test"],
            "metadatas": [{"text": f"multi version {i}", "iteration": i}]
        }, overwrite=True)
        
        if result.total_errors > 0:
            raise AssertionError(f"Error in multiple overwrite {i}: {result.errors}")
    
    # Check that only one copy exists
    search_results_multi = index.search(base_vectors[4], top_k=10)
    multi_results = [r for r in search_results_multi if r['id'] == 'multi_test']
    
    if len(multi_results) != 1:
        print(f"      ❌ Multiple overwrite failed: found {len(multi_results)} copies")
        raise AssertionError(f"Multiple overwrite failed in {scenario_name}")
    else:
        print("      ✅ Multiple overwrites work correctly")
    
    print(f"\n✅ {scenario_name} - ALL TESTS PASSED!")
    
    # Cleanup
    del index
    del vdb


def test_pq_storage_mode_transitions():
    """Test overwrite behavior during PQ storage mode transitions."""
    print(f"\n{'='*60}")
    print("🔄 TESTING: PQ Storage Mode Transitions")
    print(f"{'='*60}")
    
    # This test verifies that overwrites work correctly as the index
    # transitions from raw -> training -> quantized states
    
    vdb = zdb.VectorDatabase()
    index = vdb.create(
        dim=8,
        quantization_config={
            'type': 'pq',
            'subvectors': 4,
            'bits': 8,
            'storage_mode': 'quantized_only'
        }
    )
    
    # Add a test document in raw mode
    print("   Phase 1: Adding document in raw mode...")
    result1 = index.add({
        "vectors": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "ids": ["transition_test"],
        "metadatas": [{"phase": "raw", "version": 1}]
    }, overwrite=True)
    
    print(f"      Storage mode: {index.get_storage_mode()}")
    
    # Overwrite in raw mode
    print("   Phase 1b: Overwrite in raw mode...")
    result1b = index.add({
        "vectors": [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "ids": ["transition_test"],
        "metadatas": [{"phase": "raw_updated", "version": 2}]
    }, overwrite=True)
    
    # Add training vectors to trigger PQ training
    print("   Phase 2: Adding training vectors...")
    np.random.seed(42)
    training_vectors = [np.random.normal(0, 1, 8).tolist() for _ in range(5500)]
    training_ids = [f"train_{i}" for i in range(5500)]
    training_metadata = [{"type": "training"} for _ in range(5500)]
    
    result2 = index.add({
        "vectors": training_vectors,
        "ids": training_ids,
        "metadatas": training_metadata
    }, overwrite=True)
    
    # Wait for quantization
    print("   Waiting for PQ training...")
    max_wait = 30
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if index.can_use_quantization():
            break
        time.sleep(0.5)
    
    print(f"      Storage mode after training: {index.get_storage_mode()}")
    
    # Overwrite in quantized mode
    print("   Phase 3: Overwrite in quantized mode...")
    result3 = index.add({
        "vectors": [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "ids": ["transition_test"],
        "metadatas": [{"phase": "quantized", "version": 3}]
    }, overwrite=True)
    
    # Verify final state
    print("   Phase 4: Final verification...")
    search_results = index.search([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=10)
    transition_results = [r for r in search_results if r['id'] == 'transition_test']
    
    if len(transition_results) != 1:
        raise AssertionError(f"Transition test failed: found {len(transition_results)} copies")
    
    final_metadata = transition_results[0]['metadata']
    if final_metadata.get('phase') != 'quantized' or final_metadata.get('version') != 3:
        raise AssertionError(f"Transition content not updated: {final_metadata}")
    
    print("   ✅ Storage mode transitions work correctly!")


def test_edge_case_scenarios():
    """Test edge cases for PQ overwrite behavior."""
    print(f"\n{'='*60}")
    print("🎭 TESTING: Edge Case Scenarios")
    print(f"{'='*60}")
    
    # Test 1: Overwrite during training collection phase
    print("   Edge Case 1: Overwrite during training collection...")
    vdb = zdb.VectorDatabase()
    index = vdb.create(
        dim=8,
        quantization_config={
            'type': 'pq',
            'subvectors': 4,
            'bits': 8,
            'storage_mode': 'quantized_only'
        }
    )
    
    # Add some vectors (not enough for training)
    vectors = [[float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(10)]
    ids = [f"edge_test_{i}" for i in range(10)]
    metadatas = [{"type": "edge", "version": 1} for _ in range(10)]
    
    result1 = index.add({
        "vectors": vectors,
        "ids": ids,
        "metadatas": metadatas
    }, overwrite=True)
    
    # Overwrite some of them
    new_vectors = [[0.0, float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(5)]
    new_ids = [f"edge_test_{i}" for i in range(5)]
    new_metadatas = [{"type": "edge", "version": 2} for _ in range(5)]
    
    result2 = index.add({
        "vectors": new_vectors,
        "ids": new_ids,
        "metadatas": new_metadatas
    }, overwrite=True)
    
    # Verify no duplicates in training collection phase
    search_results = index.search([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=20)
    edge_results = [r for r in search_results if r['id'].startswith('edge_test_')]
    
    edge_id_counts = {}
    for result in edge_results:
        edge_id_counts[result['id']] = edge_id_counts.get(result['id'], 0) + 1
    
    duplicate_edge_ids = [id_val for id_val, count in edge_id_counts.items() if count > 1]
    if duplicate_edge_ids:
        raise AssertionError(f"Edge case 1 failed: duplicate IDs {duplicate_edge_ids}")
    
    print("      ✅ No duplicates during training collection phase")
    
    # Test 2: Rapid successive overwrites
    print("   Edge Case 2: Rapid successive overwrites...")
    for i in range(10):
        rapid_result = index.add({
            "vectors": [[0.0, 0.0, float(i), 0.0, 0.0, 0.0, 0.0, 0.0]],
            "ids": ["rapid_test"],
            "metadatas": [{"type": "rapid", "iteration": i}]
        }, overwrite=True)
        
        if rapid_result.total_errors > 0:
            raise AssertionError(f"Rapid overwrite {i} failed: {rapid_result.errors}")
    
    # Verify only final version exists
    rapid_search = index.search([0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=10)
    rapid_results = [r for r in rapid_search if r['id'] == 'rapid_test']
    
    if len(rapid_results) != 1:
        raise AssertionError(f"Rapid overwrite test failed: found {len(rapid_results)} copies")
    
    if rapid_results[0]['metadata']['iteration'] != 9:
        raise AssertionError(f"Rapid overwrite content wrong: {rapid_results[0]['metadata']}")
    
    print("      ✅ Rapid successive overwrites work correctly")
    
    print("   ✅ All edge cases passed!")


if __name__ == "__main__":
    try:
        test_pq_overwrite_scenarios()
        test_pq_storage_mode_transitions()
        test_edge_case_scenarios()
        
        print("\n🏆 ALL PQ OVERWRITE TESTS PASSED!")
        print("   The overwrite bug fix correctly handles all PQ scenarios:")
        print("   ✅ Raw storage (no quantization)")
        print("   ✅ PQ quantized_only mode") 
        print("   ✅ PQ quantized_with_raw mode")
        print("   ✅ Storage mode transitions during training")
        print("   ✅ Edge cases and rapid overwrites")
        print("   ✅ No duplicate documents in any scenario")
        
    except Exception as e:
        print(f"\n❌ PQ OVERWRITE TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)