"""Product quantization configuration, training, quantized search and storage modes."""

import warnings
import pytest
import numpy as np
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 30: PQ Basic Configuration and Creation
# ------------------------------------------------------------
def test_pq_basic_configuration():
    vdb = VectorDatabase()
    
    # Test creating index with PQ configuration
    quantization_config = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000
    }
    
    # ✅ EXPECT the compression warning (1536÷8 = 192x compression > 50x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*768.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw", 
            dim=1536, 
            quantization_config=quantization_config,
            expected_size=5000
        )
    
    assert index is not None
    assert index.has_quantization()
    assert not index.can_use_quantization()  # Not trained yet
    assert not index.is_quantized()  # Not using quantized search yet
    
    # Check quantization info
    quant_info = index.get_quantization_info()
    assert quant_info is not None
    assert quant_info['type'] == 'pq'
    assert quant_info['subvectors'] == 8
    assert quant_info['bits'] == 8
    assert quant_info['training_size'] == 1000
    assert not quant_info['is_trained']

# ------------------------------------------------------------
# Test 31: PQ Configuration Validation - FIXED
# ------------------------------------------------------------
def test_pq_configuration_validation():
    vdb = VectorDatabase()
    
    # Test invalid subvectors (doesn't divide dimension)
    with pytest.raises(ValueError, match="subvectors.*must divide dimension.*evenly"):
        invalid_config = {'type': 'pq', 'subvectors': 7, 'bits': 8, 'training_size': 1000}
        vdb.create("hnsw", dim=1536, quantization_config=invalid_config)
    
    # Test invalid bits
    with pytest.raises(ValueError, match="bits must be an integer between 1 and 8"):
        invalid_config = {'type': 'pq', 'subvectors': 8, 'bits': 9, 'training_size': 1000}
        vdb.create("hnsw", dim=1536, quantization_config=invalid_config)
    
    # Test invalid training size
    with pytest.raises(ValueError, match="training_size must be at least 1000"):
        invalid_config = {'type': 'pq', 'subvectors': 8, 'bits': 8, 'training_size': 500}
        vdb.create("hnsw", dim=1536, quantization_config=invalid_config)
    
    # Test unsupported quantization type
    with pytest.raises(ValueError, match="Unsupported quantization type"):
        invalid_config = {'type': 'ivf', 'subvectors': 8, 'bits': 8, 'training_size': 1000}
        vdb.create("hnsw", dim=1536, quantization_config=invalid_config)

    # ✅ FIXED: Update expected compression ratio from 96.0x to 192.0x
    with pytest.warns(UserWarning, match="Very high compression ratio.*192.0x.*may significantly impact recall quality"):
        warning_config = {'type': 'pq', 'subvectors': 32, 'bits': 4, 'training_size': 1000}
        index = vdb.create("hnsw", dim=1536, quantization_config=warning_config)
        assert index is not None  # Should still create successfully
    
    # ✅ Test that reasonable configs don't warn (8x compression < 50x threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        reasonable_config = {'type': 'pq', 'subvectors': 8, 'bits': 8, 'training_size': 1000}
        index = vdb.create("hnsw", dim=64, quantization_config=reasonable_config)  # 64÷8=8x compression
        assert index is not None

# ------------------------------------------------------------
# Test 32: PQ Training Trigger and Progress
# ------------------------------------------------------------
def test_pq_training_trigger_and_progress():
    vdb = VectorDatabase()
    
    # Use minimum valid training size
    quantization_config = {
        'type': 'pq',
        'subvectors': 4,
        'bits': 6,
        'training_size': 1000
    }
    
    # ✅ EXPECT the compression warning (128÷4 = 32x compression, but 4 bytes per float makes it 128x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*128.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw", 
            dim=128, 
            quantization_config=quantization_config,
            expected_size=2000
        )
    
    # Check initial state
    assert index.get_training_progress() == 0.0
    assert index.training_vectors_needed() == 1000
    assert not index.is_training_ready()
    assert index.get_storage_mode() == "raw_collecting_for_training"
    
    # Add partial batch and check progress
    partial_batch = []
    for i in range(500):  # Half the training size
        partial_batch.append({
            "id": f"train_{i}",
            "vector": np.random.rand(128).astype(np.float32).tolist(),
            "metadata": {"batch": "training", "index": i}
        })
    
    result = index.add(partial_batch)
    assert result.is_success()
    
    # Should be 50% progress
    progress = index.get_training_progress()
    assert abs(progress - 50.0) < 5.0  # Allow some tolerance
    assert index.training_vectors_needed() == 500
    assert not index.is_training_ready()
    
    # Add remaining vectors to trigger training
    remaining_batch = []
    for i in range(500, 1000):
        remaining_batch.append({
            "id": f"train_{i}",
            "vector": np.random.rand(128).astype(np.float32).tolist(),
            "metadata": {"batch": "training", "index": i}
        })
    
    result = index.add(remaining_batch)
    assert result.is_success()
    
    # Check training was triggered
    assert index.get_training_progress() == 100.0
    assert index.training_vectors_needed() == 0
    assert index.is_training_ready()
    assert index.can_use_quantization()

# ------------------------------------------------------------
# Test 33: PQ Memory Usage and Compression
# ------------------------------------------------------------
def test_pq_memory_usage_and_compression():
    vdb = VectorDatabase()
    
    # Configuration with high compression ratio
    quantization_config = {
        'type': 'pq',
        'subvectors': 16,
        'bits': 8,
        'training_size': 1000
    }
    
    # ✅ EXPECT the compression warning (1536÷16 = 96x compression > 50x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*384.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw", 
            dim=1536,
            quantization_config=quantization_config,
            expected_size=2000
        )
    
    # Add training data
    training_data = []
    for i in range(1000):
        training_data.append({
            "id": f"train_{i}",
            "vector": np.random.rand(1536).astype(np.float32).tolist(),
            "metadata": {"type": "training"}
        })
    
    result = index.add(training_data)
    assert result.is_success()
    assert result.total_inserted == 1000
    
    # Check quantization info after training
    quant_info = index.get_quantization_info()
    assert quant_info['is_trained']
    assert 'compression_ratio' in quant_info
    assert 'memory_mb' in quant_info
    assert 'total_centroids' in quant_info
    
    # Verify compression ratio calculation
    expected_compression = (1536 * 4) / 16  # original bytes / compressed bytes
    actual_compression = quant_info['compression_ratio']
    assert abs(actual_compression - expected_compression) < 1.0
    
    # Memory usage should be reasonable
    memory_mb = quant_info['memory_mb']
    assert memory_mb > 0
    assert memory_mb < 100  # Should be less than 100MB for this config

# ------------------------------------------------------------
# Test 34: PQ Quantized Search, Large Batches and Batch Search
# ------------------------------------------------------------
def test_pq_quantized_search_and_batch_operations():
    """Search behaviour on a quantized index, built from one oversized batch.

    This carries the distinguishing content of two earlier tests that each
    built their own PQ index, added 1000 to 1500 vectors, asserted
    is_quantized() and then asserted a search returned results. The shared
    setup is now a single index and a single k-means run, and every clause that
    distinguished the two is asserted below exactly once.
    """
    vdb = VectorDatabase()

    quantization_config = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000
    }

    # ✅ EXPECT the compression warning (256÷8 = 32x, but 4 bytes = 128x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*128.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw",
            dim=256,
            quantization_config=quantization_config,
            expected_size=2000
        )

    # The same subvector count at dim 384 produces a different ratio. The
    # warning text asserts the dim * 4 / subvectors arithmetic, so the second
    # pairing is kept even though this test does its work at dim 256. No
    # vectors are added, so this costs a construction and nothing else.
    with pytest.warns(UserWarning, match="Very high compression ratio.*192.0x.*may significantly impact recall quality"):
        vdb.create("hnsw", dim=384, quantization_config=quantization_config, expected_size=10)

    # A single batch larger than training_size, added through Format 5, which
    # both triggers training and populates the index in one call.
    batch_size = 1500  # Larger than training_size
    large_batch = {
        "ids": [f"doc_{i}" for i in range(batch_size)],
        "embeddings": np.random.rand(batch_size, 256).astype(np.float32),
        "metadatas": [{"category": "A" if i % 2 == 0 else "B", "batch": "large", "index": i}
                      for i in range(batch_size)]
    }

    result = index.add(large_batch)
    assert result.is_success()
    assert result.total_inserted == batch_size
    assert index.is_quantized()  # Should be using quantized search

    # Test search on quantized index
    query_vector = np.random.rand(256).astype(np.float32).tolist()
    search_results = index.search(query_vector, top_k=5)

    assert len(search_results) == 5
    for result in search_results:
        assert "id" in result
        assert "score" in result
        assert "metadata" in result
        assert result["score"] >= 0.0

    # Test filtered search on quantized index
    filtered_results = index.search(
        query_vector,
        filter={"category": "A"},
        top_k=10
    )

    assert len(filtered_results) >= 1
    for result in filtered_results:
        assert result["metadata"]["category"] == "A"

    # Test search with vector return (should work with quantized index)
    vector_results = index.search(query_vector, top_k=3, return_vector=True)
    assert len(vector_results) == 3
    for result in vector_results:
        assert "vector" in result
        assert len(result["vector"]) == 256

    # Test batch search on quantized index
    num_queries = 10
    query_batch = np.random.rand(num_queries, 256).astype(np.float32)

    batch_results = index.search(query_batch, top_k=5)
    assert len(batch_results) == num_queries

    for query_results in batch_results:
        assert len(query_results) == 5
        for result in query_results:
            assert result["metadata"]["batch"] == "large"
            assert isinstance(result["metadata"]["index"], int)

# ------------------------------------------------------------
# Test 35: PQ Different Configurations Performance
# ------------------------------------------------------------
def test_pq_different_configurations():
    vdb = VectorDatabase()
    
    configs = [
        # High compression, lower quality
        {'subvectors': 32, 'bits': 4, 'name': 'high_compression', 'expected_ratio': 64.0},
        # Balanced
        {'subvectors': 16, 'bits': 8, 'name': 'balanced', 'expected_ratio': 128.0},
        # Lower compression, higher quality  
        {'subvectors': 8, 'bits': 8, 'name': 'high_quality', 'expected_ratio': 256.0},
    ]
    
    indexes = {}

    # A local Generator keeps the draws reproducible without touching the global
    # numpy random state, so this test cannot perturb any other test.
    rng = np.random.default_rng(20260801)

    for config in configs:
        quantization_config = {
            'type': 'pq',
            'subvectors': config['subvectors'],
            'bits': config['bits'],
            'training_size': 1000
        }
        
        # ✅ EXPECT compression warnings for all configs (all > 50x)
        with pytest.warns(UserWarning, match=f"Very high compression ratio.*{config['expected_ratio']}x.*may significantly impact recall quality"):
            index = vdb.create(
                "hnsw",
                dim=512,  # Divisible by all subvector counts
                quantization_config=quantization_config,
                expected_size=1500
            )
        
        # Add training data
        training_data = []
        for i in range(1000):
            training_data.append({
                "id": f"{config['name']}_doc_{i}",
                "vector": rng.random(512).astype(np.float32).tolist(),
                "metadata": {"config": config['name'], "index": i}
            })
        
        result = index.add(training_data)
        assert result.is_success()
        assert index.is_quantized()
        
        indexes[config['name']] = index
        
        # Check compression ratios
        quant_info = index.get_quantization_info()
        expected_ratio = (512 * 4) / config['subvectors']
        actual_ratio = quant_info['compression_ratio']
        assert abs(actual_ratio - expected_ratio) < 1.0
    
    # Test search quality across different configurations
    query_vector = rng.random(512).astype(np.float32).tolist()

    for name, index in indexes.items():
        results = index.search(query_vector, top_k=5)
        # Approximate search may return fewer than top_k results, so an
        # assertion on an exact result count is invalid here.
        assert 0 < len(results) <= 5

        # All configurations should return valid results
        for result in results:
            assert result["metadata"]["config"] == name
            assert isinstance(result["score"], float)
            assert result["score"] >= 0.0

# ------------------------------------------------------------
# Test 37: PQ Training Size Limits and Max Training Vectors
# ------------------------------------------------------------
def test_pq_training_size_limits():
    vdb = VectorDatabase()
    
    quantization_config = {
        'type': 'pq',
        'subvectors': 4,
        'bits': 8,
        'training_size': 1000,
        'max_training_vectors': 1200  # Limit training data
    }
    
    # ✅ EXPECT the compression warning (128÷4 = 32x, but 4 bytes = 128x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*128.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw",
            dim=128,
            quantization_config=quantization_config,
            expected_size=2000
        )
    
    # Add more vectors than max_training_vectors
    training_data = []
    for i in range(1500):  # More than max_training_vectors
        training_data.append({
            "id": f"train_{i}",
            "vector": np.random.rand(128).astype(np.float32).tolist(),
            "metadata": {"index": i}
        })
    
    result = index.add(training_data)
    assert result.is_success()
    assert result.total_inserted == 1500
    
    # Should still be trained (max_training_vectors limits training data, not total vectors)
    assert index.can_use_quantization()
    
    # Test search works
    query = np.random.rand(128).astype(np.float32).tolist()
    results = index.search(query, top_k=5)
    assert len(results) == 5

# ------------------------------------------------------------
# Test 38: PQ Error Handling in Quantized Mode
# ------------------------------------------------------------
def test_pq_error_handling_quantized():
    vdb = VectorDatabase()
    
    quantization_config = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000
    }
    
    # ✅ EXPECT the compression warning (256÷8 = 32x, but 4 bytes = 128x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*128.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw",
            dim=256,
            quantization_config=quantization_config
        )
    
    # Add training data
    training_data = []
    for i in range(1000):
        training_data.append({
            "id": f"train_{i}",
            "vector": np.random.rand(256).astype(np.float32).tolist(),
            "metadata": {"type": "training"}
        })
    
    result = index.add(training_data)
    assert result.is_success()
    assert index.is_quantized()
    
    # Test error handling with invalid vectors after quantization is active
    error_data = [
        {"id": "valid", "vector": np.random.rand(256).astype(np.float32).tolist(), "metadata": {"type": "valid"}},
        {"id": "invalid", "vector": [1.0, 2.0], "metadata": {"type": "invalid"}},  # Wrong dimension
        {"id": "valid2", "vector": np.random.rand(256).astype(np.float32).tolist(), "metadata": {"type": "valid"}},
    ]
    
    result = index.add(error_data)
    assert result.total_inserted == 2  # Two valid vectors
    assert result.total_errors == 1    # One invalid vector
    assert len(result.errors) == 1
    assert "invalid" in result.errors[0]
    assert "dimension mismatch" in result.errors[0]

# ------------------------------------------------------------
# Test 39: PQ Stats and Information
# ------------------------------------------------------------
def test_pq_stats_and_information():
    vdb = VectorDatabase()
    
    quantization_config = {
        'type': 'pq',
        'subvectors': 16,
        'bits': 8,
        'training_size': 1000
    }
    
    # ✅ EXPECT the compression warning (1024÷16 = 64x, but 4 bytes = 256x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*256.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw",
            dim=1024,
            quantization_config=quantization_config
        )
    
    # Check stats before training
    stats = index.get_stats()
    assert stats["quantization_type"] == "pq"
    assert "training_progress" in stats
    assert stats["quantization_trained"] == "false"
    assert stats["quantization_active"] == "false"
    
    # Add training data
    training_data = []
    for i in range(1000):
        training_data.append({
            "id": f"doc_{i}",
            "vector": np.random.rand(1024).astype(np.float32).tolist(),
            "metadata": {"index": i}
        })
    
    result = index.add(training_data)
    assert result.is_success()
    
    # Check stats after training
    stats_after = index.get_stats()
    assert stats_after["quantization_trained"] == "true"
    assert stats_after["quantization_active"] == "true"
    assert "quantization_compression_ratio" in stats_after
    
    # Check storage mode
    storage_mode = index.get_storage_mode()
    assert storage_mode == "quantized_active"
    
    # Check info string includes quantization info
    info_str = index.info()
    assert "quantization=pq" in info_str
    assert "trained" in info_str
    assert "active" in info_str
    assert "compression=" in info_str

# ------------------------------------------------------------
# Test 40: PQ Vector Reconstruction and Get Records
# ------------------------------------------------------------
def test_pq_vector_reconstruction():
    vdb = VectorDatabase()
    
    quantization_config = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000
    }
    
    # ✅ EXPECT the compression warning (128÷8 = 16x, but 4 bytes = 64x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*64.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw",
            dim=128,
            quantization_config=quantization_config
        )
    
    # Add specific test vectors
    test_vectors = [
        {"id": "test_1", "vector": [0.1] * 128, "metadata": {"type": "uniform"}},
        {"id": "test_2", "vector": list(np.linspace(0, 1, 128)), "metadata": {"type": "sequence"}},
    ]
    
    # Add more training data
    training_data = test_vectors.copy()
    for i in range(998):  # 998 + 2 test vectors = 1000 total
        training_data.append({
            "id": f"train_{i}",
            "vector": np.random.rand(128).astype(np.float32).tolist(),
            "metadata": {"type": "random"}
        })
    
    result = index.add(training_data)
    assert result.is_success()
    assert index.is_quantized()
    
    # Test get_records with vector reconstruction
    records = index.get_records(["test_1", "test_2"], return_vector=True)
    assert len(records) == 2
    
    for record in records:
        assert "vector" in record
        assert len(record["vector"]) == 128
        assert isinstance(record["vector"], list)
        
        # Vectors should be approximately reconstructed (not exact due to quantization)
        vector = record["vector"]
        assert all(isinstance(v, float) for v in vector)
    
    # Test get_records without vectors
    records_no_vec = index.get_records(["test_1", "test_2"], return_vector=False)
    assert len(records_no_vec) == 2
    for record in records_no_vec:
        assert "vector" not in record
        assert "metadata" in record

# ------------------------------------------------------------
# Test 41: PQ Auto-calculated Training Size
# ------------------------------------------------------------  
def test_pq_auto_calculated_training_size():
    vdb = VectorDatabase()
    
    # Test auto-calculation with different subvector/bits combinations
    test_configs = [
        {'subvectors': 8, 'bits': 8, 'expected_ratio': 256.0},   # Should calculate reasonable training size
        {'subvectors': 16, 'bits': 6, 'expected_ratio': 128.0},  # Different calculation
        {'subvectors': 4, 'bits': 8, 'expected_ratio': 512.0},   # Another variation
    ]
    
    for config in test_configs:
        quantization_config = {
            'type': 'pq',
            'subvectors': config['subvectors'],
            'bits': config['bits'],
            # No training_size specified - should be auto-calculated
        }
        
        # ✅ EXPECT compression warnings for all configs
        with pytest.warns(UserWarning, match=f"Very high compression ratio.*{config['expected_ratio']}x.*may significantly impact recall quality"):
            index = vdb.create(
                "hnsw",
                dim=512,  # Divisible by all subvector counts
                quantization_config=quantization_config
            )
        
        quant_info = index.get_quantization_info()
        training_size = quant_info['training_size']
        
        # Should be auto-calculated to reasonable value
        assert training_size >= 10000  # Minimum reasonable size
        assert training_size <= 200000  # Maximum reasonable size
        
        # Should be related to the number of centroids
        centroids_per_subvector = 2 ** config['bits']
        expected_min = centroids_per_subvector * 20  # 20 samples per centroid minimum
        assert training_size >= expected_min

# ------------------------------------------------------------
# Test 42: Storage Mode Configuration and Behavior
# ------------------------------------------------------------
def test_storage_mode_configuration():
    """Test both storage modes and their memory/quality tradeoffs"""
    vdb = VectorDatabase()
    
    # Test 1: quantized_only mode (default)
    quantization_config_only = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000,
        'storage_mode': 'quantized_only'  # Explicit default
    }
    
    with pytest.warns(UserWarning, match="Very high compression ratio.*128.0x.*may significantly impact recall quality"):
        index_only = vdb.create(
            "hnsw",
            dim=256,
            quantization_config=quantization_config_only,
            expected_size=1500
        )
    
    # Test 2: quantized_with_raw mode
    quantization_config_with_raw = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000,
        'storage_mode': 'quantized_with_raw'  # Keep both
    }
    
    # Should warn about both compression AND storage mode
    with pytest.warns(UserWarning) as warning_info:
        index_with_raw = vdb.create(
            "hnsw",
            dim=256,
            quantization_config=quantization_config_with_raw,
            expected_size=1500
        )
    
    # Verify we got both warnings
    warning_messages = [str(w.message) for w in warning_info.list]
    assert any("Very high compression ratio" in msg for msg in warning_messages)
    assert any("storage_mode='quantized_with_raw' will use ~128.0x more memory" in msg for msg in warning_messages)
    
    # Add identical training data to both indexes.
    # A local Generator keeps the draws reproducible without touching the global
    # numpy random state, so this test cannot perturb any other test.
    rng = np.random.default_rng(20260801)

    training_data = []
    for i in range(1200):  # More than training_size
        training_data.append({
            "id": f"doc_{i}",
            "vector": rng.random(256).astype(np.float32).tolist(),
            "metadata": {"category": "A" if i % 2 == 0 else "B", "index": i}
        })
    
    result1 = index_only.add(training_data)
    result2 = index_with_raw.add(training_data)
    
    assert result1.is_success() and result2.is_success()
    assert index_only.is_quantized() and index_with_raw.is_quantized()
    
    # Test storage behavior differences
    stats1 = index_only.get_stats()
    stats2 = index_with_raw.get_stats()
    
    # Both should have same quantized codes
    assert stats1["quantized_codes_stored"] == stats2["quantized_codes_stored"] == "1200"
    
    # Different raw vector storage behavior
    raw_stored_only = int(stats1["raw_vectors_stored"])
    raw_stored_with_raw = int(stats2["raw_vectors_stored"])
    
    # quantized_only: should only store vectors up to training (1000)
    assert raw_stored_only == 1000
    
    # quantized_with_raw: should store ALL vectors (1200)
    assert raw_stored_with_raw == 1200
    
    # Test storage mode reporting
    assert stats1["storage_mode"] == "quantized_only"
    assert stats2["storage_mode"] == "quantized_with_raw"
    
    # Test memory usage reporting
    assert "raw_vectors_memory_mb" in stats1 and "raw_vectors_memory_mb" in stats2
    raw_memory_only = float(stats1["raw_vectors_memory_mb"])
    raw_memory_with_raw = float(stats2["raw_vectors_memory_mb"])
    assert raw_memory_with_raw > raw_memory_only  # quantized_with_raw uses more memory
    
    # Test vector retrieval behavior
    # Both should be able to retrieve vectors (different mechanisms)
    test_id = "doc_1100"  # Added after training
    
    records1 = index_only.get_records([test_id], return_vector=True)
    records2 = index_with_raw.get_records([test_id], return_vector=True)
    
    assert len(records1) == 1 and len(records2) == 1
    assert "vector" in records1[0] and "vector" in records2[0]
    assert len(records1[0]["vector"]) == len(records2[0]["vector"]) == 256
    
    # quantized_with_raw should have exact vector, quantized_only should have reconstructed
    # (We can't easily test for exactness due to floating point precision, but both should work)
    
    # Test search functionality works identically
    query_vector = rng.random(256).astype(np.float32).tolist()

    search1 = index_only.search(query_vector, top_k=5)
    search2 = index_with_raw.search(query_vector, top_k=5)

    # Approximate search may return fewer than top_k results, so an assertion on
    # an exact result count is invalid here, and the two storage modes search
    # different representations so their counts need not agree either.
    for hits in (search1, search2):
        assert 0 < len(hits) <= 5
        assert all(r["id"].startswith("doc_") for r in hits)
        scores = [r["score"] for r in hits]
        assert all(np.isfinite(s) for s in scores)
        assert scores == sorted(scores)

    # Test filtered search
    filtered1 = index_only.search(query_vector, filter={"category": "A"}, top_k=3)
    filtered2 = index_with_raw.search(query_vector, filter={"category": "A"}, top_k=3)

    # The filter is applied to the candidates the graph returned rather than
    # driving the traversal, so a filtered search can legitimately return
    # nothing and any assertion on the result count is invalid here. What holds
    # is that every result matches the filter.
    assert len(filtered1) <= 3 and len(filtered2) <= 3
    for result in filtered1 + filtered2:
        assert result["metadata"]["category"] == "A"

# ------------------------------------------------------------
# Test 43: Storage Mode Error Handling and Edge Cases
# ------------------------------------------------------------
def test_storage_mode_error_handling():
    """Test storage mode validation and edge cases"""
    vdb = VectorDatabase()
    
    # Test 1: Invalid storage mode
    with pytest.raises(ValueError, match="Invalid storage_mode.*Supported modes: quantized_only, quantized_with_raw"):
        invalid_config = {
            'type': 'pq',
            'subvectors': 8,
            'bits': 8,
            'training_size': 1000,
            'storage_mode': 'invalid_mode'
        }
        vdb.create("hnsw", dim=256, quantization_config=invalid_config)
    
    # Test 2: Case insensitive storage mode (should work)
    case_variants = ['QUANTIZED_ONLY', 'Quantized_With_Raw', 'quantized_ONLY']
    
    for variant in case_variants:
        try:
            config = {
                'type': 'pq',
                'subvectors': 8,
                'bits': 8,
                'training_size': 1000,
                'storage_mode': variant
            }

            with pytest.warns(UserWarning):  # Expect compression warning
                index = vdb.create("hnsw", dim=256, quantization_config=config)

            assert index is not None
            # Storage mode should be normalized to lowercase
            quant_info = index.get_quantization_info()
            assert quant_info is not None
            assert index.get_stats()["storage_mode"] == variant.lower()

        except Exception as e:
            pytest.fail(f"Case insensitive storage mode '{variant}' should work, but got: {e}")

    # Test 3: Default storage mode behavior (no storage_mode specified)
    default_config = {
        'type': 'pq',
        'subvectors': 4,
        'bits': 8,
        'training_size': 1000
        # No storage_mode specified - should default to quantized_only
    }

    with pytest.warns(UserWarning, match="Very high compression ratio.*128.0x"):
        default_index = vdb.create("hnsw", dim=128, quantization_config=default_config)

    # get_stats reports the configured storage mode from construction, while
    # get_storage_mode reports the current lifecycle state, so the default is
    # observable without adding any data. What quantized_only then does at
    # runtime, storing raw vectors up to training_size and codes for every
    # vector, is asserted for the explicit mode in test_storage_mode_configuration.
    assert default_index.get_stats()["storage_mode"] == "quantized_only"
    assert default_index.get_storage_mode() == "raw_collecting_for_training"

    # Test 4: Storage mode with backward compatibility (no quantization)
    no_quant_index = vdb.create("hnsw", dim=64)  # No quantization config
    
    no_quant_stats = no_quant_index.get_stats()
    assert no_quant_stats["quantization_type"] == "none"
    assert no_quant_stats["storage_mode"] == "raw_only"
    
    # Add some data
    no_quant_data = [
        {"id": "raw_1", "vector": np.random.rand(64).tolist(), "metadata": {"type": "raw"}},
        {"id": "raw_2", "vector": np.random.rand(64).tolist(), "metadata": {"type": "raw"}},
    ]
    
    result = no_quant_index.add(no_quant_data)
    assert result.is_success()
    
    # Should store everything as raw vectors
    updated_stats = no_quant_index.get_stats()
    assert int(updated_stats["raw_vectors_stored"]) == 2
    assert int(updated_stats["quantized_codes_stored"]) == 0
    
    # Search should still work
    query = np.random.rand(64).tolist()
    search_results = no_quant_index.search(query, top_k=2)
    assert len(search_results) == 2

    # The lifecycle transition that used to close this test, a
    # raw_collecting_for_training to quantized_active walk at dim 192 with
    # quantized_with_raw, is covered without a second k-means run. The
    # transition itself is asserted phase by phase in
    # test_pq_storage_mode_transition_completes_training, the pre-training
    # state in test_pq_training_trigger_and_progress, the quantized_active
    # state in test_pq_stats_and_information, and the quantized_with_raw
    # storage counters in test_storage_mode_configuration and
    # test_pq_overwrite_after_training_quantized_with_raw.

# ------------------------------------------------------------
# Shared setup for the ported overwrite and rebuild coverage
# ------------------------------------------------------------
# Every test below uses dim 8 over 4 subvectors with training_size set
# explicitly. Benchmark 43 omitted training_size, so
# _calculate_smart_training_size(4, 8) returned 10,000 against 5,500 added
# vectors, training never completed, and every overwrite it claimed to test in
# a quantized state actually ran in raw_collecting_for_training. Setting
# training_size to its 1000 minimum and adding 1,010 vectors reaches the
# quantized state for the cost of one k-means run at sub dimension 2.
#
# 8 * 4 / 4 is 8x compression, below the 50x threshold in _check_memory_usage,
# so quantized_only emits no warning. quantized_with_raw warns unconditionally
# and that warning is matched where it is used.
PQ_DIM = 8
PQ_TRAINING_SIZE = 1000
PQ_TOTAL = 1010


def _overwrite_pq_config(storage_mode):
    return {
        "type": "pq",
        "subvectors": 4,
        "bits": 8,
        "training_size": PQ_TRAINING_SIZE,
        "storage_mode": storage_mode,
    }


def _make_trained_pq_index(storage_mode, seed=20260802):
    """Build a PQ index whose training has actually completed."""
    vdb = VectorDatabase()
    if storage_mode == "quantized_with_raw":
        with pytest.warns(UserWarning,
                          match=r"storage_mode='quantized_with_raw' will use ~8\.0x more memory"):
            index = vdb.create("hnsw", dim=PQ_DIM,
                               quantization_config=_overwrite_pq_config(storage_mode),
                               expected_size=2000)
    else:
        index = vdb.create("hnsw", dim=PQ_DIM,
                           quantization_config=_overwrite_pq_config(storage_mode),
                           expected_size=2000)

    # A local Generator keeps the draws reproducible without touching the
    # global numpy random state.
    rng = np.random.default_rng(seed)
    result = index.add({
        "ids": [f"train_{i}" for i in range(PQ_TOTAL)],
        "embeddings": rng.random((PQ_TOTAL, PQ_DIM)).astype(np.float32),
        "metadatas": [{"type": "training", "index": i} for i in range(PQ_TOTAL)],
    })
    assert result.is_success()
    assert index.is_quantized(), "training must complete before the overwrite is exercised"
    return index


def _duplicate_ids(hits):
    seen = {}
    for hit in hits:
        seen[hit["id"]] = seen.get(hit["id"], 0) + 1
    return sorted(k for k, v in seen.items() if v > 1)

# ------------------------------------------------------------
# Test 78: Overwrite while still collecting vectors for training
# ------------------------------------------------------------
def test_pq_overwrite_while_collecting_for_training():
    """The state benchmark 43 actually exercised, which was itself untested."""
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=PQ_DIM,
                       quantization_config=_overwrite_pq_config("quantized_only"),
                       expected_size=1000)

    assert index.get_storage_mode() == "raw_collecting_for_training"
    assert not index.is_quantized()

    originals = [{"id": f"doc_{i}",
                  "vector": [float(i + 1)] + [0.0] * (PQ_DIM - 1),
                  "metadata": {"version": 1, "text": f"original {i}"}}
                 for i in range(10)]
    assert index.add(originals).is_success()
    assert index.get_vector_count() == 10
    assert index.get_storage_mode() == "raw_collecting_for_training"

    # Overwrite half of them while the index is still collecting.
    updated_vector = [0.0, 1.0] + [0.0] * (PQ_DIM - 2)
    result = index.add([{"id": f"doc_{i}", "vector": updated_vector,
                         "metadata": {"version": 2, "text": f"updated {i}"}}
                        for i in range(5)])
    assert result.total_inserted == 5
    assert result.total_errors == 0

    # No record was duplicated and none was lost.
    assert index.get_vector_count() == 10
    assert len(index.list(number=100)) == 10
    for i in range(10):
        assert len(index.get_records(f"doc_{i}", return_vector=False)) == 1

    # Content is the second version for the overwritten half only.
    for i in range(5):
        assert index.get_records(f"doc_{i}", return_vector=False)[0]["metadata"]["version"] == 2
    for i in range(5, 10):
        assert index.get_records(f"doc_{i}", return_vector=False)[0]["metadata"]["version"] == 1

    # Raw search still sees every record and returns each id once.
    hits = index.search(updated_vector, top_k=20)
    assert _duplicate_ids(hits) == []
    assert {h["id"] for h in hits} == {f"doc_{i}" for i in range(10)}

    # Training progress counts records, not add calls, so the five overwrites
    # did not inflate it.
    assert index.training_vectors_needed() == PQ_TRAINING_SIZE - 10
    assert not index.is_training_ready()

    # Rapid successive overwrites of one id leave exactly one record behind,
    # and the last write wins.
    for i in range(10):
        rapid = index.add({"id": "rapid", "vector": [0.0, 0.0, float(i + 1)] + [0.0] * 5,
                           "metadata": {"iteration": i}})
        assert rapid.total_errors == 0
    assert index.get_vector_count() == 11
    assert len(index.get_records("rapid", return_vector=False)) == 1
    assert index.get_records("rapid", return_vector=False)[0]["metadata"]["iteration"] == 9
    rapid_hits = index.search([0.0, 0.0, 1.0] + [0.0] * 5, top_k=20)
    assert sum(1 for h in rapid_hits if h["id"] == "rapid") == 1
    assert _duplicate_ids(rapid_hits) == []

# ------------------------------------------------------------
# Test 79: Overwrite after training completes, quantized_only
# ------------------------------------------------------------
def test_pq_overwrite_after_training_quantized_only():
    index = _make_trained_pq_index("quantized_only")
    assert index.get_storage_mode() == "quantized_active"

    baseline = index.get_stats()
    assert int(baseline["raw_vectors_stored"]) == PQ_TRAINING_SIZE
    assert int(baseline["quantized_codes_stored"]) == PQ_TOTAL

    target = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert index.add({"id": "target", "vector": target,
                      "metadata": {"version": 1}}).is_success()
    assert index.get_vector_count() == PQ_TOTAL + 1

    replacement = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = index.add({"id": "target", "vector": replacement,
                        "metadata": {"version": 2}})
    assert result.total_inserted == 1
    assert result.total_errors == 0

    # The count does not grow and the id resolves to exactly one record.
    assert index.get_vector_count() == PQ_TOTAL + 1
    records = index.get_records("target", return_vector=True)
    assert len(records) == 1
    assert records[0]["metadata"] == {"version": 2}

    # Storage accounting is unchanged by the overwrite. quantized_only stops
    # storing raw vectors once training completes, so the post training record
    # contributes a code and no raw vector.
    after = index.get_stats()
    assert int(after["raw_vectors_stored"]) == PQ_TRAINING_SIZE
    assert int(after["quantized_codes_stored"]) == PQ_TOTAL + 1
    assert after["storage_mode"] == "quantized_only"

    # Current behaviour, asserted rather than expected. contains and list both
    # read the raw vector map, so under quantized_only they are blind to any
    # record added after training even though get_records reconstructs it from
    # its code. The expectation this violates is that the three agree on which
    # ids the index holds.
    assert not index.contains("target")
    assert len(index.list(number=PQ_TOTAL + 100)) == PQ_TRAINING_SIZE

    # Whatever quantized search returns, it returns each id at most once.
    assert _duplicate_ids(index.search(replacement, top_k=25)) == []

# ------------------------------------------------------------
# Test 80: Overwrite after training completes, quantized_with_raw
# ------------------------------------------------------------
def test_pq_overwrite_after_training_quantized_with_raw():
    index = _make_trained_pq_index("quantized_with_raw")
    assert index.get_storage_mode() == "quantized_active"

    baseline = index.get_stats()
    assert int(baseline["raw_vectors_stored"]) == PQ_TOTAL
    assert int(baseline["quantized_codes_stored"]) == PQ_TOTAL

    target = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert index.add({"id": "target", "vector": target,
                      "metadata": {"version": 1}}).is_success()

    replacement = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = index.add({"id": "target", "vector": replacement,
                        "metadata": {"version": 2}})
    assert result.total_inserted == 1
    assert result.total_errors == 0

    assert index.get_vector_count() == PQ_TOTAL + 1
    records = index.get_records("target", return_vector=True)
    assert len(records) == 1
    assert records[0]["metadata"] == {"version": 2}

    # quantized_with_raw keeps every raw vector, so the replacement is exact
    # after cosine normalization rather than reconstructed from its code.
    assert records[0]["vector"] == pytest.approx(replacement, abs=1e-6)

    # Both counters grow by one, and neither grows by two.
    after = index.get_stats()
    assert int(after["raw_vectors_stored"]) == PQ_TOTAL + 1
    assert int(after["quantized_codes_stored"]) == PQ_TOTAL + 1
    assert after["storage_mode"] == "quantized_with_raw"

    # Unlike quantized_only, the raw map keeps the record visible everywhere.
    assert index.contains("target")
    assert len(index.list(number=PQ_TOTAL + 100)) == PQ_TOTAL + 1

    assert _duplicate_ids(index.search(replacement, top_k=25)) == []

    # Repeated overwrites of one id leave one record behind.
    for i in range(3):
        rapid = index.add({"id": "target", "vector": [0.0, float(i + 1)] + [0.0] * 6,
                           "metadata": {"version": 3, "iteration": i}})
        assert rapid.total_errors == 0
    assert index.get_vector_count() == PQ_TOTAL + 1
    assert len(index.get_records("target", return_vector=False)) == 1
    assert index.get_records("target", return_vector=False)[0]["metadata"]["iteration"] == 2

# ------------------------------------------------------------
# Test 81: The storage mode transition through all three phases
# ------------------------------------------------------------
def test_pq_storage_mode_transition_completes_training():
    """The lifecycle benchmark 43 named but never reached, at 1010 vectors."""
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=PQ_DIM,
                       quantization_config=_overwrite_pq_config("quantized_only"),
                       expected_size=2000)
    rng = np.random.default_rng(20260802)

    # Phase 1, empty and collecting.
    assert index.get_storage_mode() == "raw_collecting_for_training"
    assert not index.is_quantized()
    assert not index.can_use_quantization()
    assert index.get_training_progress() == 0.0
    assert index.training_vectors_needed() == PQ_TRAINING_SIZE

    # An overwrite in this phase is still a plain raw overwrite.
    index.add({"id": "tracker", "vector": [1.0] + [0.0] * 7, "metadata": {"phase": "collecting"}})
    index.add({"id": "tracker", "vector": [0.0, 1.0] + [0.0] * 6, "metadata": {"phase": "collecting_2"}})
    assert index.get_vector_count() == 1
    assert index.get_records("tracker", return_vector=False)[0]["metadata"]["phase"] == "collecting_2"

    # Phase 2, half way to the training threshold.
    half = PQ_TRAINING_SIZE // 2
    assert index.add({"ids": [f"pre_{i}" for i in range(half - 1)],
                      "embeddings": rng.random((half - 1, PQ_DIM)).astype(np.float32)}).is_success()
    assert index.get_vector_count() == half
    assert index.get_storage_mode() == "raw_collecting_for_training"
    assert index.get_training_progress() == 50.0
    assert index.training_vectors_needed() == half
    assert not index.is_training_ready()
    assert not index.is_quantized()

    # Phase 3, the threshold is crossed and training runs.
    assert index.add({"ids": [f"post_{i}" for i in range(half)],
                      "embeddings": rng.random((half, PQ_DIM)).astype(np.float32)}).is_success()
    assert index.get_training_progress() == 100.0
    assert index.training_vectors_needed() == 0
    assert index.is_training_ready()
    assert index.can_use_quantization()
    assert index.is_quantized()
    assert index.get_storage_mode() == "quantized_active"

    trained = index.get_stats()
    assert int(trained["raw_vectors_stored"]) == PQ_TRAINING_SIZE
    assert int(trained["quantized_codes_stored"]) == PQ_TRAINING_SIZE

    # Phase 4, records added after training are quantized but not stored raw.
    assert index.add({"ids": [f"late_{i}" for i in range(10)],
                      "embeddings": rng.random((10, PQ_DIM)).astype(np.float32)}).is_success()
    final = index.get_stats()
    assert index.get_storage_mode() == "quantized_active"
    assert int(final["raw_vectors_stored"]) == PQ_TRAINING_SIZE
    assert int(final["quantized_codes_stored"]) == PQ_TRAINING_SIZE + 10
    assert int(final["total_vectors"]) == PQ_TRAINING_SIZE + 10

    # An overwrite after training still replaces rather than duplicating.
    index.add({"id": "tracker", "vector": [0.0, 0.0, 1.0] + [0.0] * 5,
               "metadata": {"phase": "quantized"}})
    assert index.get_vector_count() == PQ_TRAINING_SIZE + 10
    assert len(index.get_records("tracker", return_vector=False)) == 1
    assert index.get_records("tracker", return_vector=False)[0]["metadata"]["phase"] == "quantized"

# ------------------------------------------------------------
# Test 82: rebuild_with_quantization
# ------------------------------------------------------------
def test_pq_rebuild_with_quantization():
    """The only call site anywhere was benchmark 23, which is otherwise obsolete."""
    vdb = VectorDatabase()

    # An index with no quantization configured returns False rather than raising.
    plain = vdb.create("hnsw", dim=PQ_DIM, expected_size=100)
    plain.add({"id": "a", "vector": [0.1] * PQ_DIM, "metadata": {}})
    assert plain.rebuild_with_quantization() is False

    # A configured but untrained index also returns False. This is the case
    # benchmark 23 asserted.
    untrained = vdb.create("hnsw", dim=PQ_DIM,
                           quantization_config=_overwrite_pq_config("quantized_only"),
                           expected_size=2000)
    untrained.add({"id": "a", "vector": [0.1] * PQ_DIM, "metadata": {}})
    assert untrained.rebuild_with_quantization() is False
    assert not untrained.is_quantized()

    # A trained index rebuilds, and rebuilding an already quantized index is
    # safe to repeat.
    index = _make_trained_pq_index("quantized_with_raw")
    assert index.is_quantized()
    assert index.rebuild_with_quantization() is True
    assert index.is_quantized()
    assert index.get_storage_mode() == "quantized_active"
    assert int(index.get_stats()["quantized_codes_stored"]) == PQ_TOTAL
    assert index.get_vector_count() == PQ_TOTAL

    # The rebuild does not disturb the records.
    record = index.get_records("train_5", return_vector=True)[0]
    assert record["metadata"] == {"type": "training", "index": 5}
    assert len(record["vector"]) == PQ_DIM

# ------------------------------------------------------------
# Shared fixture for the persistence facing quantization coverage
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def saved_pq_index(tmp_path_factory):
    """A trained quantized_only index saved once, with its live reconstructions.

    Training runs k-means over PQ_TRAINING_SIZE vectors, so the index and the
    directory are built once and shared. The records past PQ_TRAINING_SIZE hold
    no raw vector, which makes them the ones the loader has to reconstruct.
    """
    index = _make_trained_pq_index("quantized_only")
    rng = np.random.default_rng(20260802)
    vectors = rng.random((PQ_TOTAL, PQ_DIM)).astype(np.float32)

    ids = [f"train_{i}" for i in range(PQ_TOTAL)]
    code_only = ids[PQ_TRAINING_SIZE:]

    # What the live index returns for a record that exists only as codes. It
    # reconstructs through the codebook, which is the same path the loader now
    # uses, so this is the reference the reload must match.
    before = {r["id"]: np.asarray(r["vector"], dtype=np.float64)
              for r in index.get_records(code_only, return_vector=True)}
    assert len(before) == PQ_TOTAL - PQ_TRAINING_SIZE

    save_dir = tmp_path_factory.mktemp("pq_saved") / "pq.zdb"
    index.save(str(save_dir))

    return {"path": save_dir, "ids": ids, "code_only": code_only,
            "vectors": vectors, "before": before}

# ------------------------------------------------------------
# Test 83: PQ codebook identity across a save and load
# ------------------------------------------------------------
def test_pq_codebook_survives_save_and_load(tmp_path, saved_pq_index):
    """The restored codebook is the one that was written, byte for byte.

    pq_centroids.bin used to be written and never read, so a reloaded index
    held the zeros a fresh PQ starts with while reporting itself trained. Saving
    that index wrote the zeros back, which destroyed the only thing that made
    its coded records readable. The file is a Vec<Vec<Vec<f32>>>, whose encoding
    is ordered, so comparing bytes across the two saves is meaningful.
    """
    vdb = VectorDatabase()
    first = (saved_pq_index["path"] / "pq_centroids.bin").read_bytes()
    assert first != bytes(len(first)), "the saved codebook must not be all zeros"

    loaded = vdb.load(str(saved_pq_index["path"]))
    assert loaded.can_use_quantization()
    assert loaded.get_quantization_info()["is_trained"] is True

    second_dir = tmp_path / "resaved.zdb"
    loaded.save(str(second_dir))
    second = (second_dir / "pq_centroids.bin").read_bytes()
    assert second == first

    # A third cycle is what used to be unrecoverable, so it is checked too.
    third_dir = tmp_path / "resaved_again.zdb"
    vdb.load(str(second_dir)).save(str(third_dir))
    assert (third_dir / "pq_centroids.bin").read_bytes() == first

    # The codes ride along with the codebook, so the records that exist only as
    # codes are still there after two further saves.
    final = vdb.load(str(third_dir))
    assert final.get_vector_count() == PQ_TOTAL
    assert len(final.get_records(saved_pq_index["ids"])) == PQ_TOTAL

# ------------------------------------------------------------
# Test 84: PQ reconstruction fidelity across a round trip
# ------------------------------------------------------------
def test_pq_reconstructed_record_fidelity(saved_pq_index):
    """A reconstructed record is exactly what the live index returned.

    A record with no raw vector is rebuilt through the codebook on load. Its
    stored codes are put back as written rather than recomputed from the
    reconstruction, which is why the reload adds no error at all rather than
    compounding the quantization loss. The similarity against the original input
    is asserted as a bound because the codebook comes from k-means, which is
    seeded afresh on every training run.
    """
    vdb = VectorDatabase()
    loaded = vdb.load(str(saved_pq_index["path"]))

    after = {r["id"]: np.asarray(r["vector"], dtype=np.float64)
             for r in loaded.get_records(saved_pq_index["code_only"], return_vector=True)}
    assert set(after) == set(saved_pq_index["code_only"])

    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    similarities = []
    for record_id in saved_pq_index["code_only"]:
        # Identical to the live reconstruction, so the round trip is lossless
        # on top of the loss the storage mode already carries.
        assert np.array_equal(after[record_id], saved_pq_index["before"][record_id])

        original = saved_pq_index["vectors"][int(record_id.split("_")[1])]
        similarities.append(cosine(after[record_id], original.astype(np.float64)))

    # Measured at 0.9984 to 0.9993 over three training seeds at these settings,
    # being 256 centroids over a two dimensional subvector. The bound leaves
    # room for k-means variation while still failing on a wrong codebook, which
    # scores near zero, or an all-zero one, which cannot be normalized at all.
    assert min(similarities) > 0.95
    assert sum(similarities) / len(similarities) > 0.98

    # A record that kept its raw vector is returned from that vector, so it is
    # unchanged by the round trip rather than approximated.
    kept = loaded.get_records(["train_0", "train_500"], return_vector=True)
    kept_by_id = {r["id"]: np.asarray(r["vector"], dtype=np.float64) for r in kept}
    for record_id in ("train_0", "train_500"):
        original = saved_pq_index["vectors"][int(record_id.split("_")[1])]
        assert cosine(kept_by_id[record_id], original.astype(np.float64)) > 0.999999

# ------------------------------------------------------------
# Shared setup for the quantized graph quality coverage
# ------------------------------------------------------------
# Until the symmetric distance table landed, DistPQ::eval returned infinity
# whenever no query lookup table was set, and no insertion path sets one. Every
# distance the graph builder saw was therefore the same value, the diversity
# heuristic in select_neighbours rejected every candidate after the first, and
# layer zero out-degree was exactly one for 99.64 percent of nodes. A traversal
# reached 33 nodes of 10,000 whatever ef_search was set to, and recall at top_k
# 10 was 0.0035 against 0.9995 for the raw path on the same data.
#
# The tests below are the public API half of that guard. The graph structure
# itself is not reachable from Python, so the shuffle test that proves the
# adjacency depends on the codes lives in vdb-core's own test module.
#
# 16 * 4 / 8 is 8x compression, below the 50x threshold in
# _check_memory_usage, so neither mode warns about the ratio. Clustered unit
# vectors are used rather than uniform ones because uniform vectors are close
# to equidistant, which would let a broken graph score well.
#
# Eight subvectors rather than four. At four the codebook is coarse enough
# relative to this data that a handful of records quantize to identical codes,
# and a record whose codes another record shares cannot be told apart from it
# by any quantized search. That is the quantizer rather than the graph, but it
# makes the self query assertion flap. Measured over three data seeds, four
# subvectors gave 1 to 4 colliding records and recall 0.58 to 0.60, while eight
# gave none and recall 0.78 to 0.79.
GRAPH_DIM = 16
GRAPH_SUBVECTORS = 8
GRAPH_TRAINING_SIZE = 1000
GRAPH_TOTAL = 1500
GRAPH_QUERIES = 50


def _graph_vectors(n, dim, seed):
    """Twenty Gaussian centres, a small perturbation, then L2 normalised."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((20, dim))
    points = centres[rng.integers(0, 20, size=n)] + 0.15 * rng.standard_normal((n, dim))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module", params=["quantized_only", "quantized_with_raw"])
def quantized_graph(request):
    """A trained quantized index over clustered data, plus its ground truth.

    Module scoped and parametrised, so both storage modes are covered for the
    cost of two k-means runs. Both route through insert_pq_codes, so both were
    affected identically and both have to be asserted.
    """
    all_vectors = _graph_vectors(GRAPH_TOTAL + GRAPH_QUERIES, GRAPH_DIM, 20260804)
    data = all_vectors[:GRAPH_TOTAL]
    queries = all_vectors[GRAPH_TOTAL:]
    ids = [f"g_{i}" for i in range(GRAPH_TOTAL)]

    config = {
        "type": "pq",
        "subvectors": GRAPH_SUBVECTORS,
        "bits": 8,
        "training_size": GRAPH_TRAINING_SIZE,
        "storage_mode": request.param,
    }
    with warnings.catch_warnings():
        # quantized_with_raw warns unconditionally about its memory use, which
        # is asserted where it is the subject rather than here.
        warnings.simplefilter("ignore")
        index = VectorDatabase().create(
            "hnsw", dim=GRAPH_DIM, expected_size=GRAPH_TOTAL,
            quantization_config=config,
        )

    assert index.add({"ids": ids, "embeddings": data}).is_success()
    assert index.is_quantized(), "training did not complete"

    return {
        "index": index,
        "ids": ids,
        "data": data,
        "queries": queries,
        # Brute force cosine over the original vectors, which is what the
        # quantized path is being scored against.
        "truth": np.argsort(-(queries @ data.T), axis=1)[:, :10],
        "mode": request.param,
    }


def _recall_at_10(index, fixture, ef_search=100):
    hits = 0
    for qi, query in enumerate(fixture["queries"]):
        found = {r["id"] for r in index.search(query.tolist(), top_k=10,
                                               ef_search=ef_search)}
        hits += len(found & {fixture["ids"][j] for j in fixture["truth"][qi]})
    return hits / (10 * len(fixture["queries"]))

# ------------------------------------------------------------
# Test 85: the quantized traversal reaches the whole graph
# ------------------------------------------------------------
def test_quantized_graph_reaches_every_record(quantized_graph):
    """A request for every record returns every record.

    This is the cheapest proxy for a sound graph and it fails hardest against
    the old behaviour. The star the collapsed heuristic produced was reachable
    only 33 nodes deep, so this request came back with 34 results regardless of
    index size or ef_search. Any test that asked for a single page of ten saw
    nothing wrong.
    """
    index = quantized_graph["index"]
    for query in quantized_graph["queries"][:10]:
        found = index.search(query.tolist(), top_k=GRAPH_TOTAL,
                             ef_search=GRAPH_TOTAL)
        assert len(found) == GRAPH_TOTAL, (
            f"asked for {GRAPH_TOTAL} records and got {len(found)}; the quantized "
            f"traversal cannot reach the whole graph"
        )

# ------------------------------------------------------------
# Test 86: quantized search finds the right records
# ------------------------------------------------------------
def test_quantized_search_recall(quantized_graph):
    """Recall against brute force over the original vectors.

    The threshold is a floor rather than a match against the raw path, because
    the remaining gap is quantization loss and not a graph defect. Measured at
    0.78 to 0.79 at these settings over three data seeds. The old behaviour
    scored under 0.01, so the margin is wide in the direction that matters.
    """
    recall = _recall_at_10(quantized_graph["index"], quantized_graph)
    assert recall > 0.60, (
        f"quantized recall at top_k 10 is {recall:.4f} for "
        f"{quantized_graph['mode']}, far below what these codes support"
    )

# ------------------------------------------------------------
# Test 87: a quantized record can be found by its own vector
# ------------------------------------------------------------
def test_quantized_self_query(quantized_graph):
    """Handing a record's own vector back returns that record.

    On the collapsed graph this succeeded 4 times in 500. It is the plainest
    statement of whether the index works at all.

    A small allowance is left rather than demanding a clean sweep. Two records
    that quantize to the same codes are indistinguishable to any quantized
    search, so whichever the ranking puts first is arbitrary, and the codebook
    comes from k-means, which is seeded afresh on every run. Measured at zero
    misses over three data seeds at these settings.
    """
    index = quantized_graph["index"]
    data = quantized_graph["data"]
    ids = quantized_graph["ids"]

    checked = list(range(0, GRAPH_TOTAL, 15))
    misses = []
    for i in checked:
        found = [r["id"] for r in index.search(data[i].tolist(), top_k=10,
                                               ef_search=100)]
        if not found or found[0] != ids[i]:
            misses.append(ids[i])

    assert len(misses) <= 3, (
        f"{len(misses)} of {len(checked)} records cannot find themselves by "
        f"their own vector (first: {misses[:5]})"
    )

# ------------------------------------------------------------
# Test 88: rebuild_with_quantization produces a sound graph
# ------------------------------------------------------------
def test_rebuild_with_quantization_produces_a_sound_graph(quantized_graph):
    """The documented rebuild call has to build the same graph as training did.

    It is the one place a quantized graph is built in bulk rather than one
    record at a time, and it used to rebuild the degenerate star: calling it on
    a healthy index took recall from 0.1220 back to 0.0065. A user reaching for
    it to restore quantized search got the opposite.

    The index is rebuilt in place, which is safe because every later test takes
    its own copy of the results rather than the index state.
    """
    index = quantized_graph["index"]
    before = _recall_at_10(index, quantized_graph)

    assert index.rebuild_with_quantization() is True
    assert index.is_quantized()
    assert index.get_storage_mode() == "quantized_active"

    after = _recall_at_10(index, quantized_graph)
    assert after > 0.60, (
        f"recall fell to {after:.4f} after rebuild_with_quantization() for "
        f"{quantized_graph['mode']}"
    )
    # The rebuild quantizes from the same codebook, so it should land where it
    # started rather than merely above the floor. Measured identical to four
    # decimal places over three data seeds, and the bound leaves room for the
    # insertion order to differ.
    assert abs(after - before) < 0.05, (
        f"rebuild_with_quantization() moved recall from {before:.4f} to {after:.4f}"
    )

    for query in quantized_graph["queries"][:5]:
        found = index.search(query.tolist(), top_k=GRAPH_TOTAL,
                             ef_search=GRAPH_TOTAL)
        assert len(found) == GRAPH_TOTAL

# ------------------------------------------------------------
# Test 89: the symmetric table is reported and sized correctly
# ------------------------------------------------------------
def test_quantized_symmetric_table_memory(quantized_graph):
    """The centroid pair table is subvectors * 2^bits * 2^bits floats.

    It is reported apart from the codebook because it scales with subvectors
    and bits alone while the codebook also scales with the dimension. At the
    configuration the README recommends, 16 subvectors at 8 bits, it is 4 MiB
    whatever the dimension.
    """
    info = quantized_graph["index"].get_quantization_info()
    expected_mb = GRAPH_SUBVECTORS * (2 ** 8) * (2 ** 8) * 4 / (1024 * 1024)
    assert info["sdc_memory_mb"] == pytest.approx(expected_mb)
    assert info["is_trained"] is True

# ------------------------------------------------------------
# Shared setup for the rerank coverage
# ------------------------------------------------------------
# Relay 33 left the quantized graph sound and the codes lossy. Recall at top_k
# 10 was 0.1235 against 0.9995 for the raw path on the same data, and the graph
# was measured returning exactly what an exhaustive scan of every code returns,
# so the shortfall was the 64x compression rather than the traversal.
#
# Rerank closes it by over-fetching candidates with the codes and rescoring
# them against the raw vectors the index kept. It applies to
# quantized_with_raw, which holds a raw vector for every record, and not to
# quantized_only, which holds one only for the records collected before
# training. Rescoring a code held record against its reconstruction gains
# nothing, because the reconstruction carries exactly the information the ADC
# distance already used.
#
# Dimension 64 over 4 subvectors is 64x compression, the ratio every relay 33
# and relay 34 measurement used, which puts recall without rerank near 0.35 and
# leaves the bounds below a wide margin. Twenty clusters and the same 0.15
# perturbation as the graph fixture above.
RERANK_DIM = 64
RERANK_SUBVECTORS = 4
RERANK_TRAINING_SIZE = 1000
RERANK_TOTAL = 1500
RERANK_QUERIES = 50


def _rerank_fixture(mode, seed=20260804):
    everything = _graph_vectors(RERANK_TOTAL + RERANK_QUERIES, RERANK_DIM, seed)
    data = everything[:RERANK_TOTAL]
    queries = everything[RERANK_TOTAL:]
    ids = [f"r_{i}" for i in range(RERANK_TOTAL)]

    config = {
        "type": "pq",
        "subvectors": RERANK_SUBVECTORS,
        "bits": 8,
        "training_size": RERANK_TRAINING_SIZE,
        "storage_mode": mode,
    }
    with warnings.catch_warnings():
        # 64x compression trips the ratio warning, and quantized_with_raw warns
        # about its memory unconditionally. Both are asserted where they are the
        # subject rather than here.
        warnings.simplefilter("ignore")
        index = VectorDatabase().create(
            "hnsw", dim=RERANK_DIM, expected_size=RERANK_TOTAL,
            quantization_config=config,
        )

    assert index.add({
        "ids": ids,
        "embeddings": data,
        "metadatas": [{"band": i % 4} for i in range(RERANK_TOTAL)],
    }).is_success()
    assert index.is_quantized(), "training did not complete"

    return {
        "index": index,
        "ids": ids,
        "data": data,
        "queries": queries,
        "truth": np.argsort(-(queries @ data.T), axis=1)[:, :10],
        "mode": mode,
    }


@pytest.fixture(scope="module")
def rerank_index():
    """quantized_with_raw, the mode rerank applies to."""
    return _rerank_fixture("quantized_with_raw")


@pytest.fixture(scope="module")
def rerank_index_code_only():
    """quantized_only, the mode rerank leaves alone."""
    return _rerank_fixture("quantized_only")


def _rerank_recall(fixture, **kwargs):
    index = fixture["index"]
    hits = 0
    for qi, query in enumerate(fixture["queries"]):
        found = {r["id"] for r in index.search(query.tolist(), top_k=10, **kwargs)}
        hits += len(found & {fixture["ids"][j] for j in fixture["truth"][qi]})
    return hits / (10 * len(fixture["queries"]))

# ------------------------------------------------------------
# Test 90: rerank lifts recall on the mode that keeps raw vectors
# ------------------------------------------------------------
def test_rerank_lifts_recall(rerank_index):
    """The whole point of the feature, asserted as a gap rather than a level.

    Measured over two data seeds at these settings: 0.346 and 0.360 with rerank
    off, 0.948 and 0.920 at factor 5, and 1.000 at the default factor of 20.
    The pre-rerank behaviour is the rerank off arm, so the bound of 0.90 on the
    default fails against it by more than half.
    """
    off = _rerank_recall(rerank_index, rerank=0)
    on = _rerank_recall(rerank_index)

    assert off < 0.60, (
        f"recall without rerank is {off:.4f}, too high for the bound below to "
        f"mean anything; the fixture no longer loses enough to quantization"
    )
    assert on > 0.90, f"recall with the default rerank is only {on:.4f}"
    assert on - off > 0.40, (
        f"rerank moved recall from {off:.4f} to {on:.4f}, a gain of {on - off:.4f}"
    )

# ------------------------------------------------------------
# Test 91: a reranked score is the raw distance, and the page is ordered
# ------------------------------------------------------------
def test_rerank_returns_ordered_raw_distances(rerank_index):
    """What comes back is the cosine distance to the stored vector.

    This is the behaviour change a caller feels. Without rerank the score is the
    ADC distance, a sum of squared subvector distances against the codebook.
    With it the score is 1 minus the cosine similarity against the raw vector,
    which is exactly what a raw index reports for the same pair.
    """
    index = rerank_index["index"]
    data = rerank_index["data"]

    for query in rerank_index["queries"][:10]:
        hits = index.search(query.tolist(), top_k=10)
        assert len(hits) == 10

        scores = [h["score"] for h in hits]
        assert scores == sorted(scores), f"page is not ordered: {scores}"

        for hit in hits:
            stored = data[int(hit["id"].split("_")[1])]
            expected = 1.0 - float(np.dot(query, stored))
            assert hit["score"] == pytest.approx(expected, abs=1e-5), (
                f"{hit['id']} scored {hit['score']}, which is not the cosine "
                f"distance {expected}"
            )

    # And the ADC score for the same query is a different number, so the change
    # is real rather than the two happening to agree.
    query = rerank_index["queries"][0].tolist()
    reranked = index.search(query, top_k=10)[0]
    adc = index.search(query, top_k=10, rerank=0)[0]
    assert adc["score"] != pytest.approx(reranked["score"], abs=1e-5)

# ------------------------------------------------------------
# Test 92: quantized_only is left alone
# ------------------------------------------------------------
def test_quantized_only_does_not_rerank(rerank_index_code_only):
    """Every rerank setting returns the same page, including an exhaustive one.

    The mode drops raw vectors for every record added after training, so the
    only thing available to rescore a code held record against is its
    reconstruction, and that carries exactly the information the ADC distance
    already used. Measured at 10,000 records of dimension 768, recall at top_k
    10 over the code held records moved from 0.1320 to 0.1330 on one data draw
    and from 0.1440 to 0.1400 on another, which is noise in both directions.
    """
    index = rerank_index_code_only["index"]
    stats = index.get_stats()
    assert stats["storage_mode"] == "quantized_only"

    # The collection phase kept raw vectors and nothing clears them, so the mode
    # holds a raw vector for exactly the records it trained on. That mixed pool
    # is why the decision is made per index rather than per candidate.
    assert int(stats["raw_vectors_stored"]) == RERANK_TRAINING_SIZE
    assert int(stats["quantized_codes_stored"]) == RERANK_TOTAL

    for query in rerank_index_code_only["queries"][:10]:
        baseline = [(h["id"], h["score"]) for h in
                    index.search(query.tolist(), top_k=10, rerank=0)]
        for setting in (None, 1, 20, RERANK_TOTAL):
            page = [(h["id"], h["score"]) for h in
                    index.search(query.tolist(), top_k=10, rerank=setting)]
            assert page == baseline, (
                f"rerank={setting} changed the page on a quantized_only index"
            )

    # The scores are still ADC distances rather than cosine distances.
    query = rerank_index_code_only["queries"][0]
    hit = index.search(query.tolist(), top_k=10)[0]
    stored = rerank_index_code_only["data"][int(hit["id"].split("_")[1])]
    assert hit["score"] != pytest.approx(1.0 - float(np.dot(query, stored)), abs=1e-5)

# ------------------------------------------------------------
# Test 93: the over-fetch factor is honoured
# ------------------------------------------------------------
def test_rerank_factor_is_honoured(rerank_index):
    """Three checks that pin what the factor does to the candidate pool.

    At factor 1 the pool is the requested page, so rescoring can reorder it but
    cannot bring in a record the ADC ordering missed. At a factor that covers
    the whole index the pool is every record, so the rescore is exhaustive and
    the page has to be the brute force page. In between, recall rises with the
    factor.
    """
    index = rerank_index["index"]
    query = rerank_index["queries"][0].tolist()

    # Factor 1 fetches exactly top_k, so the ids match the unreranked page and
    # only the order and the scores move.
    unreranked = index.search(query, top_k=10, rerank=0)
    factor_one = index.search(query, top_k=10, rerank=1)
    assert {h["id"] for h in factor_one} == {h["id"] for h in unreranked}
    assert [h["id"] for h in factor_one] != [h["id"] for h in unreranked]

    # More candidates, more of the true neighbours.
    by_factor = [_rerank_recall(rerank_index, rerank=f) for f in (1, 5, 20)]
    assert by_factor[0] < by_factor[1] < by_factor[2], (
        f"recall did not rise with the factor: {by_factor}"
    )

    # A factor covering the index rescores every record, so the page is the
    # brute force page. The cap keeps the request at the record count rather
    # than at ten times it.
    for qi, held_out in enumerate(rerank_index["queries"][:10]):
        page = {h["id"] for h in index.search(held_out.tolist(), top_k=10,
                                              rerank=RERANK_TOTAL)}
        assert page == {rerank_index["ids"][j] for j in rerank_index["truth"][qi]}, (
            "an exhaustive rescore did not return the brute force page"
        )

# ------------------------------------------------------------
# Test 94: metadata filters still hold, and the pages get longer
# ------------------------------------------------------------
def test_rerank_with_metadata_filter(rerank_index):
    """The filter runs on the over-fetched pool, before the rescore.

    That ordering costs nothing, because a candidate the filter drops never
    needs a raw distance, and it fixes a shortfall that predates rerank. The
    filter used to run on a page of exactly top_k candidates, so a selective
    filter left a short page. Measured at 3,000 records of dimension 768 with a
    filter admitting a quarter of them, a request for ten came back with 2.74
    results on average without rerank and 10.00 with it.
    """
    index = rerank_index["index"]

    lengths = {}
    for factor in (0, 20):
        pages = [index.search(q.tolist(), filter={"band": 1}, top_k=10, rerank=factor)
                 for q in rerank_index["queries"]]
        for page in pages:
            for hit in page:
                assert hit["metadata"]["band"] == 1, "a filtered page leaked a record"
                assert int(hit["id"].split("_")[1]) % 4 == 1
        lengths[factor] = sum(len(page) for page in pages) / len(pages)

    assert lengths[20] > lengths[0], (
        f"over-fetching did not lengthen the filtered pages: {lengths}"
    )
    assert lengths[20] == pytest.approx(10.0), (
        f"a filter admitting a quarter of the index still returns short pages "
        f"with rerank on: mean {lengths[20]:.2f} of 10"
    )

# ------------------------------------------------------------
# Test 95: batch search matches single search
# ------------------------------------------------------------
def test_rerank_batch_matches_single(rerank_index):
    """Both batch paths rerank, and they agree with the single query path.

    batch_search_internal switches to the parallel path above five queries, so
    four and eight cover both. All three paths take the same plan, which is
    resolved once in search rather than per query.
    """
    index = rerank_index["index"]
    queries = [q.tolist() for q in rerank_index["queries"][:8]]

    single = [[(h["id"], h["score"]) for h in index.search(q, top_k=10)]
              for q in queries]
    assert all(len(page) == 10 for page in single)

    sequential = [[(h["id"], h["score"]) for h in page]
                  for page in index.search(queries[:4], top_k=10)]
    assert sequential == single[:4]

    parallel = [[(h["id"], h["score"]) for h in page]
                for page in index.search(queries, top_k=10)]
    assert parallel == single

    # An explicit factor reaches both paths too.
    single_five = [[(h["id"], h["score"]) for h in index.search(q, top_k=10, rerank=5)]
                   for q in queries]
    batch_five = [[(h["id"], h["score"]) for h in page]
                  for page in index.search(queries, top_k=10, rerank=5)]
    assert batch_five == single_five

# ------------------------------------------------------------
# Test 96: rerank does not disturb a raw index
# ------------------------------------------------------------
def test_rerank_leaves_the_raw_path_alone():
    """A raw index already ranks by the raw distance, so the parameter is inert.

    It is accepted rather than rejected so that one call site can serve both
    kinds of index, and it must not change the page, the scores or the order.
    """
    rng = np.random.default_rng(20260804)
    data = rng.standard_normal((300, RERANK_DIM)).astype(np.float32)
    index = VectorDatabase().create("hnsw", dim=RERANK_DIM, expected_size=300)
    assert index.add({"ids": [f"p_{i}" for i in range(300)],
                      "embeddings": data}).is_success()
    assert not index.is_quantized()

    query = data[7].tolist()
    baseline = [(h["id"], h["score"]) for h in index.search(query, top_k=10)]
    assert baseline[0][0] == "p_7"

    for setting in (0, 1, 20, 100000):
        assert [(h["id"], h["score"]) for h in
                index.search(query, top_k=10, rerank=setting)] == baseline, (
            f"rerank={setting} changed a raw index result"
        )

# ------------------------------------------------------------
# Test 97: removed records stay out of a reranked page
# ------------------------------------------------------------
def test_rerank_excludes_removed_records(rerank_index):
    """The live record predicate runs inside the traversal, ahead of everything.

    Relay 31 added it so a stranded graph node routes the search without
    consuming a result slot. Over-fetching multiplies the slots, so a leak here
    would be twenty times as visible as it was.

    This test removes records from the shared index, so it is last in the module
    and nothing after it may rely on that fixture being whole.
    """
    index = rerank_index["index"]
    removed = [f"r_{i}" for i in range(400)]
    for record_id in removed:
        assert index.remove_point(record_id) is True
    assert index.get_vector_count() == RERANK_TOTAL - len(removed)

    gone = set(removed)
    for query in rerank_index["queries"]:
        hits = index.search(query.tolist(), top_k=10)
        assert len(hits) == 10, f"a reranked page came back short: {len(hits)}"
        assert not (gone & {h["id"] for h in hits}), "a removed record was returned"

    # And the page is still the brute force page over what is left.
    survivors = np.arange(len(removed), RERANK_TOTAL)
    data = rerank_index["data"]
    for query in rerank_index["queries"][:10]:
        similarity = data[survivors] @ query
        want = {f"r_{survivors[j]}" for j in np.argsort(-similarity)[:10]}
        got = {h["id"] for h in index.search(query.tolist(), top_k=10,
                                             rerank=RERANK_TOTAL)}
        assert got == want
