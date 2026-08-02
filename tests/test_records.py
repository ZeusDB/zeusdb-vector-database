"""Record retrieval, removal, listing and index level metadata."""

from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 11: Test removing a vector and checking for its existence
# ------------------------------------------------------------
def test_remove_and_contains():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)
    
    index.add({
        "id": "to_remove", 
        "values": [0.5, 0.5], 
        "metadata": {}
    })
    assert index.contains("to_remove")
    
    removed = index.remove_point("to_remove")
    assert removed is True
    assert not index.contains("to_remove")
    
    # Test removing non-existent point
    removed_again = index.remove_point("nonexistent")
    assert removed_again is False

# ------------------------------------------------------------
# Test 12: Test get_records functionality
# ------------------------------------------------------------
def test_get_records():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)
    
    # Add test data
    index.add([
        {"id": "doc1", "values": [0.1, 0.2], "metadata": {"tag": "alpha"}},
        {"id": "doc2", "values": [0.3, 0.4], "metadata": {"tag": "beta"}},
        {"id": "doc3", "values": [0.5, 0.6], "metadata": {"tag": "gamma"}},
    ])
    
    # Single record
    rec = index.get_records("doc1")
    assert len(rec) == 1
    assert rec[0]["id"] == "doc1"
    assert rec[0]["metadata"]["tag"] == "alpha"
    assert "vector" in rec[0]
    
    # Multiple records
    batch = index.get_records(["doc1", "doc3"])
    assert len(batch) == 2
    
    # Metadata only
    meta_only = index.get_records(["doc1", "doc2"], return_vector=False)
    assert len(meta_only) == 2
    assert "vector" not in meta_only[0]
    
    # Missing ID silently ignored
    partial = index.get_records(["doc1", "missing_id"])
    assert len(partial) == 1
    assert partial[0]["id"] == "doc1"

# ------------------------------------------------------------
# Test 14: Test index metadata functionality
# ------------------------------------------------------------
def test_index_metadata():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=128, space="cosine", m=32, ef_construction=100)
    
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
# Test 15: Test list functionality
# ------------------------------------------------------------
def test_list_records():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, space="cosine", m=16, ef_construction=200, expected_size=5)
    
    # Add test data
    records = [
        {"id": "doc_001", "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], "metadata": {"author": "Alice"}},
        {"id": "doc_002", "values": [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], "metadata": {"author": "Bob"}},
        {"id": "doc_003", "values": [0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], "metadata": {"author": "Alice"}},
    ]
    
    result = index.add(records)
    assert result.total_inserted == 3
    
    # Test default list (10 records)
    records_list = index.list()
    assert len(records_list) == 3  # We only added 3 records
    
    # Test custom number
    records_2 = index.list(number=2)
    assert len(records_2) == 2
    
    # Verify structure (returns tuples of (id, metadata))
    for doc_id, metadata in records_list:
        assert isinstance(doc_id, str)
        assert isinstance(metadata, dict)
        assert "author" in metadata
