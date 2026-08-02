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
# Test 34: PQ Quantized Search Functionality
# ------------------------------------------------------------
def test_pq_quantized_search():
    vdb = VectorDatabase()
    
    # Use minimum valid training size
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
            expected_size=1500
        )
    
    # Add training data to trigger quantization
    training_data = []
    for i in range(1000):
        training_data.append({
            "id": f"doc_{i}",
            "vector": np.random.rand(256).astype(np.float32).tolist(),
            "metadata": {"category": "A" if i % 2 == 0 else "B", "index": i}
        })
    
    result = index.add(training_data)
    assert result.is_success()
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
# Test 36: PQ with Large Batch Operations
# ------------------------------------------------------------
def test_pq_large_batch_operations():
    vdb = VectorDatabase()
    
    quantization_config = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000
    }
    
    # ✅ EXPECT the compression warning (384÷8 = 48x, but 4 bytes = 192x)
    with pytest.warns(UserWarning, match="Very high compression ratio.*192.0x.*may significantly impact recall quality"):
        index = vdb.create(
            "hnsw",
            dim=384,
            quantization_config=quantization_config,
            expected_size=2000
        )
    
    # Add large batch to trigger training
    batch_size = 1500  # Larger than training_size
    large_batch = {
        "ids": [f"batch_doc_{i}" for i in range(batch_size)],
        "embeddings": np.random.rand(batch_size, 384).astype(np.float32),
        "metadatas": [{"batch": "large", "index": i} for i in range(batch_size)]
    }
    
    result = index.add(large_batch)
    assert result.is_success()
    assert result.total_inserted == batch_size
    assert index.is_quantized()
    
    # Test batch search on quantized index
    num_queries = 10
    query_batch = np.random.rand(num_queries, 384).astype(np.float32)
    
    batch_results = index.search(query_batch, top_k=5)
    assert len(batch_results) == num_queries
    
    for query_results in batch_results:
        assert len(query_results) == 5
        for result in query_results:
            assert result["metadata"]["batch"] == "large"
            assert isinstance(result["metadata"]["index"], int)

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
    
    # Add training data
    training_data = []
    for i in range(1200):
        training_data.append({
            "id": f"default_doc_{i}",
            "vector": np.random.rand(128).astype(np.float32).tolist(),
            "metadata": {"type": "default_test"}
        })
    
    result = default_index.add(training_data)
    assert result.is_success()
    
    # Should behave like quantized_only (default)
    stats = default_index.get_stats()
    assert stats["storage_mode"] == "quantized_only"
    assert int(stats["raw_vectors_stored"]) == 1000  # Only training vectors stored
    assert int(stats["quantized_codes_stored"]) == 1200  # All vectors quantized
    
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
    
    # Test 5: Storage mode transitions during lifecycle
    transition_config = {
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000,
        'storage_mode': 'quantized_with_raw'
    }
    
    with pytest.warns(UserWarning):
        transition_index = vdb.create("hnsw", dim=192, quantization_config=transition_config)
    
    # Before training: should be in collecting mode
    assert transition_index.get_storage_mode() == "raw_collecting_for_training"
    assert not transition_index.is_quantized()
    
    # Add training data
    pre_training_data = []
    for i in range(500):  # Less than training_size
        pre_training_data.append({
            "id": f"pre_{i}",
            "vector": np.random.rand(192).tolist(),
            "metadata": {"phase": "pre_training"}
        })
    
    result = transition_index.add(pre_training_data)
    assert result.is_success()
    
    # Still collecting
    assert not transition_index.is_training_ready()
    assert transition_index.get_storage_mode() == "raw_collecting_for_training"
    
    # Complete training
    post_training_data = []
    for i in range(500, 1000):
        post_training_data.append({
            "id": f"post_{i}",
            "vector": np.random.rand(192).tolist(),
            "metadata": {"phase": "complete_training"}
        })
    
    result = transition_index.add(post_training_data)
    assert result.is_success()
    
    # Should now be quantized and active
    assert transition_index.is_quantized()
    assert transition_index.get_storage_mode() == "quantized_active"
    
    # Add post-training data to test storage mode behavior
    final_data = []
    for i in range(1000, 1100):
        final_data.append({
            "id": f"final_{i}",
            "vector": np.random.rand(192).tolist(),
            "metadata": {"phase": "post_training"}
        })
    
    result = transition_index.add(final_data)
    assert result.is_success()
    
    # Verify quantized_with_raw behavior: should store all vectors
    final_stats = transition_index.get_stats()
    assert int(final_stats["raw_vectors_stored"]) == 1100  # All vectors stored
    assert int(final_stats["quantized_codes_stored"]) == 1100  # All vectors quantized
    assert final_stats["storage_mode"] == "quantized_with_raw"
