"""Metadata filter evaluation on search."""

from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 21: Test metadata filtering (basic)
# ------------------------------------------------------------
def test_metadata_filtering_basic():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="cosine", expected_size=10)
    
    records = [
        {"id": "v1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"author": "Alice", "score": 95}},
        {"id": "v2", "values": [0.9, 0.8, 0.7, 0.6], "metadata": {"author": "Bob", "score": 80}},
        {"id": "v3", "values": [0.15, 0.25, 0.35, 0.45], "metadata": {"author": "Alice", "score": 85}},
        {"id": "v4", "values": [0.92, 0.82, 0.72, 0.62], "metadata": {"author": "Charlie", "score": 78}},
    ]
    
    result = index.add(records)
    assert result.is_success()
    
    query = [0.1, 0.2, 0.3, 0.4]
    
    # Test equality filter
    alice_results = index.search(vector=query, filter={"author": "Alice"}, top_k=10)
    assert len(alice_results) == 2
    for r in alice_results:
        assert r['metadata']['author'] == "Alice"
    
    # Test numeric filter
    high_score_results = index.search(vector=query, filter={"score": {"gt": 90}}, top_k=10)
    assert len(high_score_results) == 1
    assert high_score_results[0]['metadata']['score'] == 95

# ------------------------------------------------------------
# Test 22: Test advanced metadata filtering
# ------------------------------------------------------------
def test_metadata_filtering_advanced():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)
    
    records = [
        {
            "id": "doc1", 
            "values": [0.1, 0.2, 0.3, 0.4], 
            "metadata": {
                "author": "Alice",
                "year": 2024,
                "rating": 4.5,
                "published": True,
                "tags": ["science", "ai"],
                "price": 29.99
            }
        },
        {
            "id": "doc2", 
            "values": [0.9, 0.8, 0.7, 0.6], 
            "metadata": {
                "author": "Bob",
                "year": 2023,
                "rating": 3.8,
                "published": False,
                "tags": ["technology"],
                "price": 19.99
            }
        }
    ]
    
    result = index.add(records)
    assert result.is_success()
    
    query = [0.1, 0.2, 0.3, 0.4]
    
    # Test multiple conditions
    complex_results = index.search(
        vector=query,
        filter={"published": True, "rating": {"gte": 4.0}, "year": {"gte": 2024}},
        top_k=10
    )
    assert len(complex_results) == 1
    assert complex_results[0]['id'] == 'doc1'
    
    # Test array contains
    ai_results = index.search(vector=query, filter={"tags": {"contains": "ai"}}, top_k=10)
    assert len(ai_results) == 1
    assert ai_results[0]['id'] == 'doc1'
    
    # Test string operations
    author_contains = index.search(vector=query, filter={"author": {"contains": "A"}}, top_k=10)
    assert len(author_contains) == 1
    assert author_contains[0]['metadata']['author'] == "Alice"
