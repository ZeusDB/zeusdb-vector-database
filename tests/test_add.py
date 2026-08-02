"""Vector insertion across the five input formats, overwrite, and add error handling."""

import numpy as np
from zeusdb_vector_database import VectorDatabase
from helpers import assert_vectors_close

# ------------------------------------------------------------
# Test 3: Format 1 - Single Object
# ------------------------------------------------------------
def test_add_format_1_single_object():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=10)
    
    # Format 1: Single Object
    add_result = index.add({
        "id": "doc1",
        "values": [0.1, 0.2],
        "metadata": {"text": "hello"}
    })
    
    # Verify AddResult properties
    assert add_result.total_inserted == 1
    assert add_result.total_errors == 0
    assert add_result.is_success()
    assert "1 inserted" in add_result.summary()
    assert "0 errors" in add_result.summary()
    assert add_result.vector_shape == (1, 2)
    assert len(add_result.errors) == 0
    
    # Verify the record was added correctly
    records = index.get_records("doc1")
    assert len(records) == 1
    assert records[0]["id"] == "doc1"
    assert records[0]["metadata"]["text"] == "hello"
    
    # ✅ FIXED: Account for cosine normalization
    assert_vectors_close(records[0]["vector"], [0.1, 0.2], space="cosine")

# ------------------------------------------------------------
# Test 4: Format 2 - List of Objects
# ------------------------------------------------------------
def test_add_format_2_list_of_objects():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=10)
    
    # Format 2: List of Objects
    add_result = index.add([
        {"id": "doc1", "values": [0.1, 0.2], "metadata": {"text": "hello"}},
        {"id": "doc2", "values": [0.3, 0.4], "metadata": {"text": "world"}}
    ])
    
    # Verify AddResult properties
    assert add_result.total_inserted == 2
    assert add_result.total_errors == 0
    assert add_result.is_success()
    assert "2 inserted" in add_result.summary()
    assert "0 errors" in add_result.summary()
    assert add_result.vector_shape == (2, 2)
    assert len(add_result.errors) == 0
    
    # Verify both records were added correctly
    records = index.get_records(["doc1", "doc2"])
    assert len(records) == 2
    
    # Check first record
    doc1 = next(r for r in records if r["id"] == "doc1")
    assert doc1["metadata"]["text"] == "hello"
    # ✅ FIXED: Account for cosine normalization
    assert_vectors_close(doc1["vector"], [0.1, 0.2], space="cosine")
    
    # Check second record
    doc2 = next(r for r in records if r["id"] == "doc2")
    assert doc2["metadata"]["text"] == "world"
    # ✅ FIXED: Account for cosine normalization
    assert_vectors_close(doc2["vector"], [0.3, 0.4], space="cosine")

# ------------------------------------------------------------
# Test 5: Format 3 - Separate Arrays (Python lists)
# ------------------------------------------------------------
def test_add_format_3_separate_arrays_lists():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=10)
    
    # Format 3: Separate Arrays with Python lists
    add_result = index.add({
        "ids": ["doc1", "doc2"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "metadatas": [{"text": "hello"}, {"text": "world"}]
    })
    
    # Verify AddResult properties
    assert add_result.total_inserted == 2
    assert add_result.total_errors == 0
    assert add_result.is_success()
    assert add_result.vector_shape == (2, 2)
    assert len(add_result.errors) == 0
    
    # Verify repr format matches expected (adjust for actual format)
    repr_str = repr(add_result)
    # The actual format might be "Some((2, 2))" instead of "(2, 2)"
    assert "inserted=2" in repr_str
    assert "errors=0" in repr_str
    assert "(2, 2)" in repr_str
    
    # Verify records were added correctly
    records = index.get_records(["doc1", "doc2"])
    assert len(records) == 2

# ------------------------------------------------------------
# Test 6: Format 4 - List of Objects with NumPy Arrays
# ------------------------------------------------------------
def test_add_format_4_list_with_numpy():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=10)
    
    # Format 4: List of Objects with NumPy arrays
    data = [
        {"id": "doc2", "values": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), "metadata": {"type": "blog"}},
        {"id": "doc3", "values": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32), "metadata": {"type": "news"}},
    ]
    
    result = index.add(data)
    
    # Verify AddResult properties
    assert result.total_inserted == 2
    assert result.total_errors == 0
    assert result.is_success()
    assert "2 inserted" in result.summary()
    assert "0 errors" in result.summary()
    assert result.vector_shape == (2, 4)
    
    # Verify records were added correctly
    records = index.get_records(["doc2", "doc3"])
    assert len(records) == 2
    
    # Check that NumPy arrays were converted properly
    doc2 = next(r for r in records if r["id"] == "doc2")
    assert doc2["metadata"]["type"] == "blog"
    # ✅ FIXED: Account for cosine normalization
    assert_vectors_close(doc2["vector"], [0.1, 0.2, 0.3, 0.4], space="cosine")
    
    doc3 = next(r for r in records if r["id"] == "doc3")
    assert doc3["metadata"]["type"] == "news"
    # ✅ FIXED: Account for cosine normalization
    assert_vectors_close(doc3["vector"], [0.5, 0.6, 0.7, 0.8], space="cosine")

# ------------------------------------------------------------
# Test 7: Format 5 - Separate Arrays with NumPy (High Performance)
# ------------------------------------------------------------
def test_add_format_5_separate_arrays_numpy():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=10)
    
    # Format 5: Separate Arrays with NumPy (most performant)
    add_result = index.add({
        "ids": ["doc1", "doc2"],
        "embeddings": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "metadatas": [{"text": "hello"}, {"text": "world"}]
    })
    
    # Verify AddResult properties
    assert add_result.total_inserted == 2
    assert add_result.total_errors == 0
    assert add_result.is_success()
    assert add_result.vector_shape == (2, 2)
    
    # Verify repr format matches expected (adjust for actual format)
    repr_str = repr(add_result)
    assert "inserted=2" in repr_str
    assert "errors=0" in repr_str
    assert "(2, 2)" in repr_str
    
    # Verify records were added correctly
    records = index.get_records(["doc1", "doc2"])
    assert len(records) == 2
    
    # Verify NumPy data was processed correctly
    doc1 = next(r for r in records if r["id"] == "doc1")
    assert doc1["metadata"]["text"] == "hello"
    # ✅ FIXED: Account for cosine normalization
    assert_vectors_close(doc1["vector"], [0.1, 0.2], space="cosine")
    
    doc2 = next(r for r in records if r["id"] == "doc2")
    assert doc2["metadata"]["text"] == "world"
    # ✅ FIXED: Account for cosine normalization  
    assert_vectors_close(doc2["vector"], [0.3, 0.4], space="cosine")

# ------------------------------------------------------------
# Test 8: Large Scale NumPy Performance Test
# ------------------------------------------------------------
def test_add_large_scale_numpy_performance():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=128, expected_size=1000)
    
    # Create large batch using NumPy for performance
    batch_size = 500
    ids = [f"doc_{i}" for i in range(batch_size)]
    vectors = np.random.rand(batch_size, 128).astype(np.float32)
    metadatas = [{"batch": "large", "index": str(i)} for i in range(batch_size)]
    
    # Format 5: Large scale separate arrays with NumPy
    result = index.add({
        "ids": ids,
        "embeddings": vectors,  # NumPy 2D array for efficiency
        "metadatas": metadatas
    })
    
    # Verify large batch results
    assert result.total_inserted == batch_size
    assert result.total_errors == 0
    assert result.is_success()
    assert result.vector_shape == (batch_size, 128)
    
    # Verify search functionality works with large dataset
    query = np.random.rand(128).tolist()
    results = index.search(query, top_k=5)
    assert len(results) == 5
    
    # Test filtered search on large dataset
    filtered_results = index.search(query, filter={"batch": "large"}, top_k=10)
    assert len(filtered_results) == 10

# ------------------------------------------------------------
# Test 9: Mixed Format Error Handling
# ------------------------------------------------------------
def test_mixed_format_error_handling():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=10)
    
    # Test Format 2 with one valid and one invalid record
    add_result = index.add([
        {"id": "valid1", "values": [0.1, 0.2], "metadata": {"status": "good"}},
        {"id": "invalid", "values": [0.1], "metadata": {"status": "bad"}},  # Wrong dimension
        {"id": "valid2", "values": [0.3, 0.4], "metadata": {"status": "good"}},
    ])
    
    # Verify partial success
    assert add_result.total_inserted == 2
    assert add_result.total_errors == 1
    assert not add_result.is_success()  # Has errors
    assert "2 inserted" in add_result.summary()
    assert "1 errors" in add_result.summary()
    assert len(add_result.errors) == 1
    assert "invalid" in add_result.errors[0]  # Error should mention the problematic ID
    
    # Verify valid records were still added
    records = index.get_records(["valid1", "valid2"])
    assert len(records) == 2

# ------------------------------------------------------------
# Test 17: Test error handling with AddResult
# ------------------------------------------------------------
def test_error_handling_add_result():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=5)
    
    # Test batch with errors
    error_records = [
        {"id": "valid1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"type": "valid"}},
        {"id": "invalid", "values": [0.1, 0.2], "metadata": {"type": "invalid"}},  # Wrong dimension
        {"id": "valid2", "values": [0.5, 0.6, 0.7, 0.8], "metadata": {"type": "valid"}},
    ]
    
    result = index.add(error_records)
    assert result.total_inserted == 2  # 2 valid records
    assert result.total_errors == 1    # 1 invalid record
    assert len(result.errors) == 1
    assert not result.is_success()
    assert "2 inserted" in result.summary()
    assert "1 errors" in result.summary()

# ------------------------------------------------------------
# Test 23: Test overwrite functionality
# ------------------------------------------------------------
def test_overwrite_functionality():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)
    
    # Add initial record
    result1 = index.add({
        "id": "doc1", 
        "values": [0.1, 0.2], 
        "metadata": {"version": "v1"}
    })
    assert result1.is_success()
    
    # Verify initial record
    records = index.get_records("doc1", return_vector=False)
    assert records[0]["metadata"]["version"] == "v1"
    
    # Overwrite with new data
    result2 = index.add({
        "id": "doc1", 
        "values": [0.3, 0.4], 
        "metadata": {"version": "v2"}
    })
    assert result2.is_success()
    
    # Verify overwrite
    updated_records = index.get_records("doc1", return_vector=True)
    assert updated_records[0]["metadata"]["version"] == "v2"
    # ✅ FIXED: Account for cosine normalization
    assert_vectors_close(updated_records[0]["vector"], [0.3, 0.4], space="cosine")
