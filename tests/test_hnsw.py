import pytest
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 1: Test the creation of an HNSW index with default parameters
# ------------------------------------------------------------
def test_create_index_hnsw_default():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw()
    assert index is not None
    assert index.info() is not None

# ------------------------------------------------------------
# Test 2: Test the creation of an HNSW index with custom parameters
# ------------------------------------------------------------
def test_create_index_hnsw_custom():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=4, expected_size=10)
    assert index is not None
    stats = index.get_stats()
    assert stats["dimension"] == "4"
    assert stats["expected_size"] == "10"

# ------------------------------------------------------------
# Test 3: Test adding vectors and querying for nearest neighbors
# ------------------------------------------------------------
def test_add_and_query():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=4, expected_size=10)
    
    index.add_point("vec1", [0.1, 0.2, 0.3, 0.4], {"label": "A"})
    index.add_point("vec2", [0.2, 0.1, 0.4, 0.3], {"label": "B"})
    
    results = index.query([0.1, 0.2, 0.3, 0.4], top_k=2)
    assert len(results) == 2
    ids = [r[0] for r in results]
    assert "vec1" in ids

# ------------------------------------------------------------
# Test 4: Test batch adding vectors and querying with metadata filter
# ------------------------------------------------------------
def test_add_batch_and_filter():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=2, expected_size=5)
    
    # Use tuple format for batch adding
    points = [
        ("a", [1.0, 0.0], {"type": "x"}),
        ("b", [0.0, 1.0], {"type": "y"}),
        ("c", [1.0, 1.0], {"type": "x"}),
    ]
    result = index.add_batch(points)
    assert result["success_count"] == ["3"]
    assert result["error_count"] == ["0"]
    
    # Query with filter
    results = index.query([1.0, 0.0], top_k=3, filter={"type": "x"})
    assert len(results) >= 1  # Should find at least one match
    for r in results:
        meta = index.get_vector_metadata(r[0])
        assert meta is not None
        assert meta["type"] == "x"

# ------------------------------------------------------------
# Test 5: Test removing a vector and checking for its existence
# ------------------------------------------------------------
def test_remove_and_contains():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=2, expected_size=5)
    
    index.add_point("to_remove", [0.5, 0.5], {})
    assert index.contains("to_remove")
    
    removed = index.remove_point("to_remove")
    assert removed is True
    assert not index.contains("to_remove")

# ------------------------------------------------------------
# Test 6: Test searching with metadata included in the results
# ------------------------------------------------------------
def test_search_with_metadata():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=2, expected_size=5)
    
    index.add_point("foo", [0.1, 0.2], {"cat": "bar"})
    results = index.search_with_metadata([0.1, 0.2], top_k=1, include_metadata=True)
    
    assert len(results) == 1
    assert results[0][0] == "foo"
    meta = results[0][2]
    assert meta is not None
    assert meta["cat"] == "bar"

# ------------------------------------------------------------
# Test 7: Test comprehensive search functionality (from benchmark 6)
# ------------------------------------------------------------
def test_comprehensive_search():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=8, space="cosine", M=16, ef_construction=200)
    
    # Add test data
    vectors = {
        "doc_001": ([0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], {"author": "Alice"}),
        "doc_002": ([0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], {"author": "Bob"}),
        "doc_003": ([0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], {"author": "Alice"}),
        "doc_004": ([0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], {"author": "Bob"}),
        "doc_005": ([0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], {"author": "Alice"}),
    }
    
    for doc_id, (vec, meta) in vectors.items():
        index.add_point(doc_id, vec, metadata=meta)
    
    query_vec = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]
    
    # Test filtered search
    alice_results = index.query(vector=query_vec, filter={"author": "Alice"}, top_k=5)
    assert len(alice_results) == 3  # Should find 3 Alice documents
    for doc_id, score in alice_results:
        meta = index.get_vector_metadata(doc_id)
        assert meta["author"] == "Alice"
    
    # Test unfiltered search
    all_results = index.query(vector=query_vec, filter=None, top_k=5)
    assert len(all_results) == 5  # Should find all 5 documents
    
    # Test high ef_search
    high_ef_results = index.query(vector=query_vec, filter={"author": "Alice"}, top_k=5, ef_search=400)
    assert len(high_ef_results) == 3

# ------------------------------------------------------------
# Test 8: Test index metadata functionality (from benchmark 2)
# ------------------------------------------------------------
def test_index_metadata():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=128, space="cosine", M=32, ef_construction=100)
    
    # Add index metadata
    metadata = {
        "creator": "Ross Armstrong",
        "version": "0.1",
        "created_at": "2024-01-28T11:35:55Z",
        "index_type": "HNSW",
        "embedding_model": "openai/text-embedding-ada-002",
        "dataset": "docs_corpus_v2",
        "environment": "production",
        "description": "Knowledge base index for customer support articles",
        "num_documents": "15000",
        "tags": "['support', 'docs', '2024']"
    }
    
    index.add_metadata(metadata)
    
    # Test individual metadata retrieval
    assert index.get_metadata("creator") == "Ross Armstrong"
    assert index.get_metadata("version") == "0.1"
    assert index.get_metadata("nonexistent") is None
    
    # Test all metadata retrieval
    all_meta = index.get_all_metadata()
    assert len(all_meta) == len(metadata)
    assert all_meta["creator"] == "Ross Armstrong"
    assert all_meta["embedding_model"] == "openai/text-embedding-ada-002"

# ------------------------------------------------------------
# Test 9: Test list functionality (from benchmark 5)
# ------------------------------------------------------------
def test_list_records():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=8, space="cosine", M=16, ef_construction=200, expected_size=5)
    
    # Add test data
    vectors = {
        "doc_001": ([0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], {"author": "Alice"}),
        "doc_002": ([0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], {"author": "Bob"}),
        "doc_003": ([0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], {"author": "Alice"}),
    }
    
    for doc_id, (vec, meta) in vectors.items():
        index.add_point(doc_id, vec, metadata=meta)
    
    # Test default list (10 records)
    records = index.list()
    assert len(records) == 3  # We only added 3 records
    
    # Test custom number
    records_2 = index.list(number=2)
    assert len(records_2) == 2
    
    # Verify structure
    for doc_id, metadata in records:
        assert isinstance(doc_id, str)
        assert isinstance(metadata, dict)
        assert "author" in metadata

# ------------------------------------------------------------
# Test 10: Test get_vector functionality
# ------------------------------------------------------------
def test_get_vector():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=4, expected_size=5)
    
    test_vector = [0.1, 0.2, 0.3, 0.4]
    index.add_point("test_id", test_vector, {"type": "test"})
    
    # Test successful retrieval
    retrieved = index.get_vector("test_id")
    assert retrieved is not None
    assert len(retrieved) == len(test_vector)
    
    # Use approximate equality for floating-point comparison
    for i, (expected, actual) in enumerate(zip(test_vector, retrieved)):
        assert abs(expected - actual) < 1e-6, f"Vector element {i}: expected {expected}, got {actual}"
    
    # Test non-existent ID
    assert index.get_vector("nonexistent") is None

# ------------------------------------------------------------
# Test 11: Test get_vector_metadata functionality
# ------------------------------------------------------------
def test_get_vector_metadata():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=4, expected_size=5)
    
    test_metadata = {"type": "test", "category": "example"}
    index.add_point("test_id", [0.1, 0.2, 0.3, 0.4], test_metadata)
    
    # Test successful retrieval
    retrieved = index.get_vector_metadata("test_id")
    assert retrieved == test_metadata
    
    # Test non-existent ID
    assert index.get_vector_metadata("nonexistent") is None

# ------------------------------------------------------------
# Test 12: Test error handling
# ------------------------------------------------------------
def test_error_handling():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=4, expected_size=5)
    
    # Test dimension mismatch
    with pytest.raises(Exception):  # Should raise PyValueError from Rust
        index.add_point("test", [0.1, 0.2], {})  # Wrong dimension
    
    # Test duplicate ID
    index.add_point("duplicate", [0.1, 0.2, 0.3, 0.4], {})
    with pytest.raises(Exception):  # Should raise PyValueError from Rust
        index.add_point("duplicate", [0.1, 0.2, 0.3, 0.4], {})
    
    # Test query dimension mismatch
    with pytest.raises(Exception):  # Should raise PyValueError from Rust
        index.query([0.1, 0.2], top_k=1)  # Wrong dimension

# ------------------------------------------------------------
# Test 13: Test parameter validation during index creation
# ------------------------------------------------------------
def test_index_creation_validation():
    vdb = VectorDatabase()
    
    # Test invalid dimension
    with pytest.raises(RuntimeError):
        vdb.create_index_hnsw(dim=0)
    
    # Test invalid ef_construction
    with pytest.raises(RuntimeError):
        vdb.create_index_hnsw(ef_construction=0)
    
    # Test invalid expected_size
    with pytest.raises(RuntimeError):
        vdb.create_index_hnsw(expected_size=0)
    
    # Test invalid M
    with pytest.raises(RuntimeError):
        vdb.create_index_hnsw(M=300)  # > 256
    
    # Test invalid space
    with pytest.raises(RuntimeError):
        vdb.create_index_hnsw(space="invalid")

# ------------------------------------------------------------
# Test 14: Test search without metadata
# ------------------------------------------------------------
def test_search_without_metadata():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=2, expected_size=5)
    
    index.add_point("foo", [0.1, 0.2], {"cat": "bar"})
    results = index.search_with_metadata([0.1, 0.2], top_k=1, include_metadata=False)
    
    assert len(results) == 1
    assert results[0][0] == "foo"
    assert results[0][2] is None  # No metadata included

# ------------------------------------------------------------
# Test 15: Test removing non-existent point
# ------------------------------------------------------------
def test_remove_nonexistent():
    vdb = VectorDatabase()
    index = vdb.create_index_hnsw(dim=2, expected_size=5)
    
    # Try to remove a point that doesn't exist
    removed = index.remove_point("nonexistent")
    assert removed is False
