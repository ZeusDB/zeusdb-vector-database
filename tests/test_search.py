"""Single vector and batch search, ranking, distance metrics and search edge cases."""

import json

import pytest
import struct

import numpy as np
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 10: All Formats Search Functionality
# ------------------------------------------------------------
def test_all_formats_search_functionality():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=20)
    
    # Add data using all different formats
    
    # Format 1: Single Object
    index.add({
        "id": "single", 
        "values": [0.1, 0.2, 0.3, 0.4], 
        "metadata": {"format": "single", "type": "test"}
    })
    
    # Format 2: List of Objects
    index.add([
        {"id": "list1", "values": [0.2, 0.3, 0.4, 0.5], "metadata": {"format": "list", "type": "test"}},
        {"id": "list2", "values": [0.3, 0.4, 0.5, 0.6], "metadata": {"format": "list", "type": "test"}},
    ])
    
    # Format 3: Separate Arrays (lists)
    index.add({
        "ids": ["sep1", "sep2"],
        "embeddings": [[0.4, 0.5, 0.6, 0.7], [0.5, 0.6, 0.7, 0.8]],
        "metadatas": [{"format": "separate", "type": "test"}, {"format": "separate", "type": "test"}]
    })
    
    # Format 4: List with NumPy
    index.add([
        {"id": "numpy1", "values": np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float32), "metadata": {"format": "numpy_list", "type": "test"}},
    ])
    
    # Format 5: Separate Arrays with NumPy
    index.add({
        "ids": ["numpy_sep1"],
        "embeddings": np.array([[0.7, 0.8, 0.9, 1.0]], dtype=np.float32),
        "metadatas": [{"format": "numpy_separate", "type": "test"}]
    })
    
    # Verify all records were added (7 total)
    stats = index.get_stats()
    total_vectors = int(stats["total_vectors"])
    # Debug: print actual count if assertion fails
    if total_vectors != 7:
        print(f"Expected 7 vectors, got {total_vectors}")
        # List all records to debug
        all_records = index.list(number=20)
        print(f"All records: {[r[0] for r in all_records]}")
    assert total_vectors == 7
    
    # Test search functionality across all formats
    query_vector = [0.1, 0.2, 0.3, 0.4]
    expected_ids = {"single", "list1", "list2", "sep1", "sep2", "numpy1", "numpy_sep1"}

    # Every record reached the store, checked exactly. contains reads the id map
    # directly rather than traversing the approximate graph, so it is the right
    # place to assert that all five input formats landed.
    for record_id in expected_ids:
        assert index.contains(record_id)

    # Seven records, literal vectors, top_k above the record count and a
    # sequential build with a seeded level generator, so the search is exhaustive
    # here and the count is exact. This was relaxed to a range while the graph
    # varied between runs.
    all_results = index.search(query_vector, top_k=10)
    assert len(all_results) == 7
    all_ids = {r["id"] for r in all_results}
    assert all_ids == expected_ids
    assert len(all_ids) == len(all_results)  # no id returned twice
    scores = [r["score"] for r in all_results]
    assert all(np.isfinite(s) for s in scores)
    assert scores == sorted(scores)

    # The filter is applied to the candidates the graph returned rather than
    # driving the traversal, so in general a filtered search yields fewer results
    # than there are matching records. On this index the unfiltered search
    # already returns all seven, so every matching record survives the filter and
    # the counts are exact.
    filtered_results = index.search(query_vector, filter={"type": "test"}, top_k=10)
    assert len(filtered_results) == 7
    assert all(r["metadata"]["type"] == "test" for r in filtered_results)

    single_format = index.search(query_vector, filter={"format": "single"}, top_k=10)
    assert len(single_format) == 1
    assert all(r["metadata"]["format"] == "single" for r in single_format)
    assert {r["id"] for r in single_format} == {"single"}

    list_format = index.search(query_vector, filter={"format": "list"}, top_k=10)
    assert len(list_format) == 2
    assert all(r["metadata"]["format"] == "list" for r in list_format)
    assert {r["id"] for r in list_format} == {"list1", "list2"}

# ------------------------------------------------------------
# Test 13: Test comprehensive search functionality with filters
# ------------------------------------------------------------
def test_comprehensive_search():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, space="cosine", m=16, ef_construction=200)
    
    # Add test data
    records = [
        {"id": "doc_001", "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], "metadata": {"author": "Alice"}},
        {"id": "doc_002", "values": [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], "metadata": {"author": "Bob"}},
        {"id": "doc_003", "values": [0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], "metadata": {"author": "Alice"}},
        {"id": "doc_004", "values": [0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], "metadata": {"author": "Bob"}},
        {"id": "doc_005", "values": [0.12, 0.22, 0.33, 0.13, 0.45, 0.23, 0.65, 0.71], "metadata": {"author": "Alice"}},
    ]
    
    result = index.add(records)
    assert result.total_inserted == 5
    
    query_vec = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]
    
    # Test filtered search
    alice_results = index.search(vector=query_vec, filter={"author": "Alice"}, top_k=5)
    alice_count = len(alice_results)
    if alice_count != 3:
        print(f"Expected 3 Alice results, got {alice_count}")
        print(f"Alice results: {[r['id'] for r in alice_results]}")
        # Check all results to see what's there
        all_results = index.search(vector=query_vec, top_k=10)
        print(f"All results: {[(r['id'], r['metadata']['author']) for r in all_results]}")
    # Three Alice records exist and all three come back. Five literal vectors and
    # a sequential build make this exact. It was relaxed to >= 2 while the graph
    # varied between runs.
    assert alice_count == 3
    for result in alice_results:
        assert result["metadata"]["author"] == "Alice"

    # top_k above the record count over five records, so the search is
    # exhaustive. Relaxed to >= 3 for the same reason.
    all_results = index.search(vector=query_vec, filter=None, top_k=10)
    assert len(all_results) == 5

    # A wider ef_search cannot find fewer, so this is the same three.
    high_ef_results = index.search(vector=query_vec, filter={"author": "Alice"}, top_k=5, ef_search=400)
    assert len(high_ef_results) == 3

# ------------------------------------------------------------
# Test 16: Test search with return_vector option
# ------------------------------------------------------------
def test_search_with_return_vector():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=5)
    
    test_vector = [0.1, 0.2, 0.3, 0.4]
    index.add({
        "id": "test_id", 
        "values": test_vector, 
        "metadata": {"type": "test"}
    })
    
    # Test with return_vector=True
    results_with_vector = index.search([0.1, 0.2, 0.3, 0.4], top_k=1, return_vector=True)
    assert len(results_with_vector) == 1
    assert "vector" in results_with_vector[0]
    assert len(results_with_vector[0]["vector"]) == 4
    
    # Test with return_vector=False (default)
    results_without_vector = index.search([0.1, 0.2, 0.3, 0.4], top_k=1, return_vector=False)
    assert len(results_without_vector) == 1
    assert "vector" not in results_without_vector[0]

# ------------------------------------------------------------
# Test 19: Test different distance metrics (cosine, L1, L2)
# ------------------------------------------------------------
def test_distance_metrics():
    records = [
        {"id": "doc1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"type": "test"}},
        {"id": "doc2", "values": [0.9, 0.8, 0.7, 0.6], "metadata": {"type": "test"}},
    ]
    query_vector = [0.1, 0.2, 0.3, 0.4]
    expected_ids = {"doc1", "doc2"}

    # Test cosine
    vdb_cos = VectorDatabase()
    index_cos = vdb_cos.create("hnsw", dim=4, space="cosine")
    result_cos = index_cos.add(records)
    assert result_cos.is_success()
    results_cos = index_cos.search(query_vector, top_k=2)
    # Two records and top_k of 2, so both come back. An unseeded parallel build
    # could return one. The build is sequential and the level generator is
    # seeded, and the reverse link fix files layer zero adjacency for both
    # points, so neither point can be unreachable.
    assert len(results_cos) == 2
    assert {r["id"] for r in results_cos} == expected_ids
    assert all(np.isfinite(r["score"]) for r in results_cos)
    assert [r["score"] for r in results_cos] == sorted(r["score"] for r in results_cos)

    # Test L2
    vdb_l2 = VectorDatabase()
    index_l2 = vdb_l2.create("hnsw", dim=4, space="L2")
    result_l2 = index_l2.add(records)
    assert result_l2.is_success()
    results_l2 = index_l2.search(query_vector, top_k=2, ef_search=150)
    assert len(results_l2) == 2
    assert {r["id"] for r in results_l2} == expected_ids
    assert all(np.isfinite(r["score"]) for r in results_l2)
    assert [r["score"] for r in results_l2] == sorted(r["score"] for r in results_l2)

    # Test L1
    vdb_l1 = VectorDatabase()
    index_l1 = vdb_l1.create("hnsw", dim=4, space="L1")
    result_l1 = index_l1.add(records)
    assert result_l1.is_success()
    results_l1 = index_l1.search(query_vector, top_k=2, ef_search=150)
    assert len(results_l1) == 2
    assert {r["id"] for r in results_l1} == expected_ids
    assert all(np.isfinite(r["score"]) for r in results_l1)
    assert [r["score"] for r in results_l1] == sorted(r["score"] for r in results_l1)

# ------------------------------------------------------------
# Test 24: Test edge cases with proper debug output
# ------------------------------------------------------------
def test_edge_cases():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)
    
    # Test empty metadata
    result = index.add({
        "id": "empty_meta",
        "values": [0.1, 0.2],
        "metadata": {}
    })
    assert result.is_success()
    
    # Test no metadata - completely omit metadata field
    result2 = index.add({
        "id": "no_meta", 
        "values": [0.3, 0.4]
        # Completely omit metadata to test if this is supported
    })
    
    # If the add failed, let's try with empty metadata instead
    if not result2.is_success():
        print("Adding without metadata failed, trying with empty metadata")
        result2 = index.add({
            "id": "no_meta", 
            "values": [0.3, 0.4],
            "metadata": {}
        })
    assert result2.is_success()
    
    # Debug: Check what records actually exist
    all_records = index.list(number=10)
    print(f"All records in index: {[r[0] for r in all_records]}")
    
    # Test search with empty filter
    results = index.search([0.1, 0.2], filter={}, top_k=5)
    actual_count = len(results)
    print(f"Search with empty filter found {actual_count} results: {[r['id'] for r in results]}")
    
    # Try search with no filter at all
    no_filter_results = index.search([0.1, 0.2], filter=None, top_k=5)
    print(f"Search with no filter found {len(no_filter_results)} results: {[r['id'] for r in no_filter_results]}")
    
    # An empty filter imposes no conditions, so every record satisfies it and it
    # is equivalent to passing no filter. Both find both records, including the
    # one added without a metadata field. This used to be an if/elif/else that
    # accepted three different outcomes and so could not fail on the behaviour it
    # names.
    assert len(no_filter_results) == 2
    assert actual_count == 2
    assert {r["id"] for r in results} == {"empty_meta", "no_meta"}

    # Test very small top_k
    results_small = index.search([0.1, 0.2], top_k=1)
    assert len(results_small) == 1

    # A top_k above the record count returns the whole index and no more.
    results_large = index.search([0.1, 0.2], top_k=100)
    assert len(results_large) == 2

# ------------------------------------------------------------
# Test 27: Batch Search with List of Vectors
# ------------------------------------------------------------
def test_batch_search_list_vectors():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="cosine", expected_size=20)
    
    # Add test data
    records = [
        {"id": "doc1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"category": "A", "priority": 1}},
        {"id": "doc2", "values": [0.5, 0.6, 0.7, 0.8], "metadata": {"category": "B", "priority": 2}},
        {"id": "doc3", "values": [0.2, 0.3, 0.4, 0.5], "metadata": {"category": "A", "priority": 3}},
        {"id": "doc4", "values": [0.8, 0.7, 0.6, 0.5], "metadata": {"category": "B", "priority": 1}},
        {"id": "doc5", "values": [0.1, 0.1, 0.2, 0.2], "metadata": {"category": "C", "priority": 2}},
        {"id": "doc6", "values": [0.9, 0.8, 0.7, 0.6], "metadata": {"category": "C", "priority": 3}},
    ]
    
    result = index.add(records)
    assert result.is_success()
    assert result.total_inserted == 6
    
    # Test batch search with list of vectors
    query_vectors = [
        [0.1, 0.2, 0.3, 0.4],  # Similar to doc1
        [0.5, 0.6, 0.7, 0.8],  # Similar to doc2
        [0.9, 0.8, 0.7, 0.6],  # Similar to doc6
    ]
    
    batch_results = index.search(query_vectors, top_k=3)
    
    # Verify batch results structure
    assert isinstance(batch_results, list)
    assert len(batch_results) == 3  # One result set per query
    
    # Verify each query result
    for i, query_results in enumerate(batch_results):
        assert isinstance(query_results, list)
        assert len(query_results) <= 3  # top_k=3
        assert len(query_results) >= 1  # Should find at least one result
        
        # Verify result structure
        for result in query_results:
            assert "id" in result
            assert "score" in result
            assert "metadata" in result
            assert isinstance(result["score"], float)
            assert result["score"] >= 0.0  # Distance should be non-negative
    
    # Test batch search with return_vector=True
    batch_results_with_vectors = index.search(query_vectors, top_k=2, return_vector=True)
    assert len(batch_results_with_vectors) == 3
    
    for query_results in batch_results_with_vectors:
        for result in query_results:
            assert "vector" in result
            assert len(result["vector"]) == 4  # Dimension should match
            vector = result["vector"]
            # An ndarray of float32 rather than a list of Python floats. The
            # values are the same and only the container is different, which
            # is what a caller reading it by index or by iteration sees.
            assert isinstance(vector, np.ndarray)
            assert vector.dtype == np.float32

# ------------------------------------------------------------
# Test 28: Batch Search with 2D NumPy Array
# ------------------------------------------------------------
def test_batch_search_numpy_array():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, space="cosine", expected_size=50)
    
    # Add test data using NumPy for efficiency
    np.random.seed(42)  # For reproducible results
    num_docs = 30
    
    # Create document vectors
    doc_vectors = np.random.rand(num_docs, 8).astype(np.float32)
    doc_ids = [f"doc_{i:03d}" for i in range(num_docs)]
    doc_metadatas = [{"type": "document", "index": i, "batch": i % 3} for i in range(num_docs)]
    
    # Add documents using NumPy format
    add_result = index.add({
        "ids": doc_ids,
        "embeddings": doc_vectors,
        "metadatas": doc_metadatas
    })
    
    assert add_result.is_success()
    assert add_result.total_inserted == num_docs
    
    # Create query vectors using NumPy 2D array
    num_queries = 5
    query_vectors = np.random.rand(num_queries, 8).astype(np.float32)
    
    # Test batch search with NumPy 2D array
    batch_results = index.search(query_vectors, top_k=5)
    
    # Verify batch results structure
    assert isinstance(batch_results, list)
    assert len(batch_results) == num_queries
    
    # Verify each query result
    for i, query_results in enumerate(batch_results):
        assert isinstance(query_results, list)
        assert len(query_results) <= 5  # top_k=5
        assert len(query_results) >= 1  # Should find at least one result
        
        # Verify results are sorted by score (ascending for cosine distance)
        scores = [r["score"] for r in query_results]
        assert scores == sorted(scores), f"Query {i} results not sorted by score"
        
        # Verify result structure
        for result in query_results:
            assert "id" in result
            assert "score" in result
            assert "metadata" in result
            assert result["id"] in doc_ids
            assert result["metadata"]["type"] == "document"
            assert isinstance(result["metadata"]["index"], int)
            assert isinstance(result["metadata"]["batch"], int)
    
    # Test with different ef_search parameter
    batch_results_high_ef = index.search(query_vectors, top_k=3, ef_search=200)
    assert len(batch_results_high_ef) == num_queries
    
    for query_results in batch_results_high_ef:
        assert len(query_results) <= 3  # top_k=3
        assert len(query_results) >= 1
    
    # Test error handling: wrong NumPy array shape
    wrong_shape_queries = np.random.rand(3, 4).astype(np.float32)  # Wrong dimension
    
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.search(wrong_shape_queries, top_k=3)

# ------------------------------------------------------------
# Test 29: Batch Search with Metadata Filter
# ------------------------------------------------------------
def test_batch_search_with_metadata_filter():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=6, space="cosine", expected_size=40)
    
    # Add diverse test data with rich metadata
    records = [
        {"id": "article_001", "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
         "metadata": {"type": "article", "author": "Alice", "year": 2024, "published": True, "score": 8.5}},
        {"id": "article_002", "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 
         "metadata": {"type": "article", "author": "Bob", "year": 2023, "published": True, "score": 7.2}},
        {"id": "article_003", "values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
         "metadata": {"type": "article", "author": "Alice", "year": 2024, "published": False, "score": 9.1}},
        {"id": "blog_001", "values": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
         "metadata": {"type": "blog", "author": "Charlie", "year": 2024, "published": True, "score": 6.8}},
        {"id": "blog_002", "values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
         "metadata": {"type": "blog", "author": "Alice", "year": 2023, "published": True, "score": 8.0}},
        {"id": "news_001", "values": [0.6, 0.7, 0.8, 0.9, 1.0, 0.1], 
         "metadata": {"type": "news", "author": "David", "year": 2024, "published": True, "score": 5.5}},
        {"id": "news_002", "values": [0.7, 0.8, 0.9, 1.0, 0.1, 0.2], 
         "metadata": {"type": "news", "author": "Bob", "year": 2024, "published": False, "score": 7.8}},
        {"id": "draft_001", "values": [0.8, 0.9, 1.0, 0.1, 0.2, 0.3], 
         "metadata": {"type": "article", "author": "Alice", "year": 2024, "published": False, "score": 9.5}},
    ]
    
    result = index.add(records)
    assert result.is_success()
    assert result.total_inserted == 8
    
    # Create batch queries
    query_vectors = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Similar to article_001
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Similar to blog_001
        [0.7, 0.8, 0.9, 1.0, 0.1, 0.2],  # Similar to news_002
    ]
    
    # Test 1: Filter by type
    article_results = index.search(
        query_vectors, 
        filter={"type": "article"}, 
        top_k=5
    )
    
    assert len(article_results) == 3  # Three queries
    for query_results in article_results:
        assert len(query_results) >= 1  # Should find at least one article
        for result in query_results:
            assert result["metadata"]["type"] == "article"
    
    # Test 2: Filter by author
    alice_results = index.search(
        query_vectors, 
        filter={"author": "Alice"}, 
        top_k=5
    )
    
    assert len(alice_results) == 3  # Three queries
    for query_results in alice_results:
        assert len(query_results) >= 1  # Should find at least one Alice document
        for result in query_results:
            assert result["metadata"]["author"] == "Alice"
    
    # Test 3: Filter by multiple conditions
    published_alice_2024 = index.search(
        query_vectors,
        filter={"author": "Alice", "year": 2024, "published": True},
        top_k=5
    )
    
    assert len(published_alice_2024) == 3  # Three queries
    for query_results in published_alice_2024:
        # May not find results for all queries due to strict filter
        for result in query_results:
            assert result["metadata"]["author"] == "Alice"
            assert result["metadata"]["year"] == 2024
            assert result["metadata"]["published"] is True
    
    # Test 4: Filter with numeric conditions
    high_score_results = index.search(
        query_vectors,
        filter={"score": {"gte": 8.0}},
        top_k=5
    )
    
    assert len(high_score_results) == 3  # Three queries
    for query_results in high_score_results:
        for result in query_results:
            assert result["metadata"]["score"] >= 8.0
    
    # Test 5: Filter with range conditions
    recent_high_quality = index.search(
        query_vectors,
        filter={"year": {"gte": 2024}, "score": {"gt": 7.0}, "published": True},
        top_k=3
    )
    
    assert len(recent_high_quality) == 3  # Three queries
    for query_results in recent_high_quality:
        for result in query_results:
            assert result["metadata"]["year"] >= 2024
            assert result["metadata"]["score"] > 7.0
            assert result["metadata"]["published"] is True
    
    # Test 6: Empty filter results (should still return structure)
    impossible_filter = index.search(
        query_vectors,
        filter={"type": "nonexistent"},
        top_k=5
    )
    
    assert len(impossible_filter) == 3  # Three queries
    for query_results in impossible_filter:
        assert len(query_results) == 0  # No results should match
    
    # Test 7: Batch search with filter and return_vector=True
    filtered_with_vectors = index.search(
        query_vectors,
        filter={"type": "article"},
        top_k=2,
        return_vector=True
    )
    
    assert len(filtered_with_vectors) == 3  # Three queries
    for query_results in filtered_with_vectors:
        for result in query_results:
            assert "vector" in result
            assert len(result["vector"]) == 6  # Dimension should match
            assert result["metadata"]["type"] == "article"
    
    # Test 8: Compare filtered vs unfiltered results
    unfiltered_results = index.search(query_vectors, top_k=8)
    filtered_results = index.search(query_vectors, filter={"published": True}, top_k=8)
    
    assert len(unfiltered_results) == 3
    assert len(filtered_results) == 3
    
    # Filtered results should be a subset (or equal) for each query
    for i in range(3):
        unfiltered_count = len(unfiltered_results[i])
        filtered_count = len(filtered_results[i])
        assert filtered_count <= unfiltered_count
        
        # All filtered results should have published=True
        for result in filtered_results[i]:
            assert result["metadata"]["published"] is True

# ------------------------------------------------------------
# Helper for the input validation tests below
# ------------------------------------------------------------
def build_validation_index(dim=4):
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=dim, space="cosine", expected_size=10)
    index.add([
        {"id": "s1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"type": "test"}},
        {"id": "s2", "values": [0.5, 0.6, 0.7, 0.8], "metadata": {"type": "test"}},
    ])
    return index

# ------------------------------------------------------------
# Test 61: A single query vector of the wrong dimension
# ------------------------------------------------------------
def test_search_single_query_wrong_dimension():
    index = build_validation_index()

    # The message names both the expected and the actual length.
    with pytest.raises(ValueError, match="dimension mismatch: expected 4, got 3"):
        index.search([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="dimension mismatch: expected 4, got 5"):
        index.search([0.1, 0.2, 0.3, 0.4, 0.5])

    # A 1D NumPy query of the wrong length takes the same path.
    with pytest.raises(ValueError, match="dimension mismatch: expected 4, got 3"):
        index.search(np.array([0.1, 0.2, 0.3], dtype=np.float32))

    # A batch names the offending position instead.
    with pytest.raises(ValueError, match="Vector 1: dimension mismatch: expected 4, got 2"):
        index.search([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6]])

# ------------------------------------------------------------
# Test 62: An empty list as a query
# ------------------------------------------------------------
def test_search_empty_list_query():
    index = build_validation_index()

    # Current behaviour, asserted rather than expected. An empty list extracts
    # as an empty batch before the single vector path is reached, so a caller
    # who passed one empty query vector is told the batch is empty. The
    # expectation this violates is that the message describes what was passed,
    # which here would be an empty search vector.
    with pytest.raises(ValueError, match="Batch cannot be empty"):
        index.search([])

# ------------------------------------------------------------
# Test 63: None as a query
# ------------------------------------------------------------
def test_search_none_query():
    index = build_validation_index()

    # Current behaviour, asserted rather than expected. None fails every
    # extraction attempt and surfaces as the PyO3 conversion TypeError rather
    # than as a ValueError raised by the index. The message names the offending
    # value but not the parameter. The expectation this violates is that search
    # validates its query argument the way add validates its data argument,
    # which rejects None with an explicit message.
    #
    # PyO3 owns this wording and changed it at 0.29. It used to name the type,
    # so this matched NoneType, and it now names the value. The rejection is
    # the same one from the same line, the extract to Vec<f32> that follows the
    # failed cast to a one dimensional array.
    with pytest.raises(TypeError, match="'None' is not an instance of 'Sequence'"):
        index.search(None)

    # For contrast, add does reject None explicitly.
    with pytest.raises(ValueError, match="Data cannot be None"):
        index.add(None)

# ------------------------------------------------------------
# Test 64: An empty batch
# ------------------------------------------------------------
def test_search_empty_batch():
    index = build_validation_index()

    # A batch with no queries in it is the same object as the empty single
    # query of test 62, and the two are indistinguishable to the index.
    empty_batch = []
    with pytest.raises(ValueError, match="Batch cannot be empty"):
        index.search(empty_batch)

    # A NumPy array with no rows is rejected the same way.
    with pytest.raises(ValueError, match="Batch cannot be empty"):
        index.search(np.zeros((0, 4), dtype=np.float32))

# ------------------------------------------------------------
# Test 65: A batch containing an empty vector
# ------------------------------------------------------------
def test_search_batch_with_empty_vector():
    index = build_validation_index()

    # The empty vector check runs before the dimension check, and the message
    # names the position in the batch.
    with pytest.raises(ValueError, match="Vector 0 in batch cannot be empty"):
        index.search([[]])

    with pytest.raises(ValueError, match="Vector 1 in batch cannot be empty"):
        index.search([[0.1, 0.2, 0.3, 0.4], []])

    with pytest.raises(ValueError, match="Vector 2 in batch cannot be empty"):
        index.search([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], []])

# ------------------------------------------------------------
# Test 66: A query containing a non-finite value
# ------------------------------------------------------------
def test_search_non_finite_query_values():
    index = build_validation_index()

    # The single vector path rejects NaN and both infinities, and the message
    # names the offending index and value.
    with pytest.raises(ValueError, match="invalid value at index 0: NaN"):
        index.search([float("nan"), 0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="invalid value at index 0: inf"):
        index.search([float("inf"), 0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="invalid value at index 3: -inf"):
        index.search([0.1, 0.2, 0.3, float("-inf")])

    # A 1D NumPy query is validated identically.
    with pytest.raises(ValueError, match="invalid value at index 0: NaN"):
        index.search(np.array([np.nan, 0.2, 0.3, 0.4], dtype=np.float32))

    # The batch path applies the same value check, and the message names the
    # entry in the batch as well as the component within it. Without the check
    # the vector is normalized to itself, because the norm of a vector
    # containing NaN is not greater than zero, and the search returns hits
    # whose scores carry no distance information.
    with pytest.raises(
        ValueError, match=r"Vector 0 in batch contains invalid value at index 0: NaN"
    ):
        index.search([[float("nan"), 0.2, 0.3, 0.4]], top_k=2)

    with pytest.raises(
        ValueError, match=r"Vector 0 in batch contains invalid value at index 0: inf"
    ):
        index.search([[float("inf"), 0.2, 0.3, 0.4]], top_k=2)

    # The failing entry is named by its position in the batch, which is the
    # point of the check for a batch large enough that finding it by hand is
    # not practical.
    with pytest.raises(
        ValueError, match=r"Vector 2 in batch contains invalid value at index 3: -inf"
    ):
        index.search(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.1, 0.2, 0.3, float("-inf")],
            ],
            top_k=2,
        )

    # Batches above five queries take the parallel path, and the check runs
    # before either path is chosen.
    with pytest.raises(
        ValueError, match=r"Vector 7 in batch contains invalid value at index 1: NaN"
    ):
        index.search([[0.1, 0.2, 0.3, 0.4]] * 7 + [[0.1, float("nan"), 0.3, 0.4]], top_k=2)

    # A NumPy 2D batch reaches the same check.
    with pytest.raises(
        ValueError, match=r"Vector 1 in batch contains invalid value at index 2: NaN"
    ):
        index.search(
            np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, np.nan, 0.4]], dtype=np.float32),
            top_k=2,
        )

    # A valid batch is unaffected.
    assert len(index.search([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], top_k=2)) == 2

# ------------------------------------------------------------
# Test 67: Searching an empty index
# ------------------------------------------------------------
def test_search_empty_index():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="cosine", expected_size=10)

    assert int(index.get_stats()["total_vectors"]) == 0
    assert not index.contains("anything")

    # An empty index is not an error condition. Every search shape returns an
    # empty result rather than raising, and a batch still returns one result
    # set per query.
    assert index.search([0.1, 0.2, 0.3, 0.4], top_k=5) == []
    assert index.search([0.1, 0.2, 0.3, 0.4], top_k=5, return_vector=True) == []
    assert index.search([0.1, 0.2, 0.3, 0.4], filter={"type": "test"}, top_k=5) == []
    assert index.search([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], top_k=5) == [[], []]

    # Input validation still applies on an empty index.
    with pytest.raises(ValueError, match="dimension mismatch: expected 4, got 2"):
        index.search([0.1, 0.2])


# ------------------------------------------------------------
# Test 104: return_vector hands back an array, not a list of Python floats
# ------------------------------------------------------------
def test_return_vector_is_a_float32_array():
    """The container changed in 0.8.0 and the values did not.

    set_item("vector", vec) on a Vec<f32> built a PyList and one Python float
    per component, which at top_k 10 and dimension 1,536 is 15,360 allocations
    a page. PyArray1::from_vec writes the same f32 values into one buffer.
    """
    index = build_validation_index()

    hit = index.search([1.0, 0.0, 0.0, 0.0], top_k=1, return_vector=True)[0]
    assert isinstance(hit["vector"], np.ndarray)
    assert hit["vector"].dtype == np.float32
    assert hit["vector"].shape == (4,)

    # get_records agrees with search, so a caller does not have to remember
    # which one hands back which.
    record = index.get_records(hit["id"], return_vector=True)[0]
    assert isinstance(record["vector"], np.ndarray)
    assert record["vector"].dtype == np.float32
    assert np.array_equal(record["vector"], hit["vector"])

    # Every element is readable by index and by iteration, which is what an
    # unchanged caller does. Under cosine the stored vector is the unit length
    # form of what was supplied.
    supplied = {"s1": [0.1, 0.2, 0.3, 0.4], "s2": [0.5, 0.6, 0.7, 0.8]}[hit["id"]]
    expected = np.asarray(supplied) / np.linalg.norm(supplied)
    assert len(hit["vector"]) == 4
    assert float(hit["vector"][0]) == pytest.approx(expected[0], abs=1e-6)
    assert [float(v) for v in hit["vector"]] == pytest.approx(list(expected), abs=1e-6)

    # A batch page carries the same type in every hit.
    batch = index.search([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                         top_k=2, return_vector=True)
    for page in batch:
        for entry in page:
            assert isinstance(entry["vector"], np.ndarray)
            assert entry["vector"].dtype == np.float32

    # return_vector=False still omits the key entirely.
    assert "vector" not in index.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0]

    # An empty page carries no vectors and does not raise.
    empty = VectorDatabase().create("hnsw", dim=4, expected_size=10)
    assert empty.search([1.0, 0.0, 0.0, 0.0], top_k=5, return_vector=True) == []


# ------------------------------------------------------------
# Test 105: the batch dispatch reads an array through the array branch
# ------------------------------------------------------------
def test_batch_dispatch_reads_arrays_without_the_sequence_protocol():
    """cast now runs before extract, so the zero copy branch is reachable.

    extract::<Vec<Vec<f32>>> succeeds on a 2-D array through the sequence
    protocol, so tried first it consumed exactly the input the array branch
    below it was written for. What is asserted here is that the three input
    shapes still answer identically, since a reordered dispatch reads the same
    values through a different path.
    """
    index = build_validation_index()

    queries = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    from_f32 = index.search(queries, top_k=3)
    from_f64 = index.search(queries.astype(np.float64), top_k=3)
    from_list = index.search(queries.tolist(), top_k=3)

    def page_bits(batch):
        return [[(h["id"], struct.pack("<f", h["score"])) for h in page] for page in batch]

    assert page_bits(from_f32) == page_bits(from_list)
    assert page_bits(from_f64) == page_bits(from_list)

    # A 1-D float64 query, which is what NumPy hands back by default, agrees
    # with the float32 form and with the list form.
    single_bits = [
        [(h["id"], struct.pack("<f", h["score"])) for h in index.search(q, top_k=3)]
        for q in (queries[0], queries[0].astype(np.float64), queries[0].tolist())
    ]
    assert single_bits[0] == single_bits[1] == single_bits[2]

    # The checks the list branch ran are still run on an array. An array with
    # no rows is an empty batch, whatever its dtype.
    for dtype in (np.float32, np.float64):
        with pytest.raises(ValueError, match="Batch cannot be empty"):
            index.search(np.empty((0, 4), dtype=dtype))

        # A row of the wrong width is a dimension mismatch in the same words a
        # list of lists gets.
        with pytest.raises(ValueError, match="dimension mismatch: expected 4, got 2"):
            index.search(np.zeros((3, 2), dtype=dtype))

        # Three dimensions is not a batch of queries at all. It fails the
        # array cast, which checks the rank as well as the dtype, and falls
        # through every arm to the single vector one, exactly as before.
        with pytest.raises(TypeError):
            index.search(np.zeros((2, 2, 4), dtype=dtype))

    # A filter applies the same way through the array branch.
    filtered_array = index.search(queries, top_k=3, filter={"type": "test"})
    filtered_list = index.search(queries.tolist(), top_k=3, filter={"type": "test"})
    assert page_bits(filtered_array) == page_bits(filtered_list)


# ============================================================================
# THE INNER PRODUCT SPACE
# ============================================================================
#
# `space="dot"` is the fourth metric. It is the only one whose score can be
# negative, because it returns `1 - dot` and an inner product above one gives
# one, and the only one that does not normalise what it is given.


def _dot_corpus(seed=31, n=240, dim=16, scale=3.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, dim)).astype(np.float32) * scale)


def _dot_index(corpus, **kwargs):
    n, dim = corpus.shape
    index = VectorDatabase().create("hnsw", dim=dim, space="dot",
                                    expected_size=max(n, 100), **kwargs)
    assert index.add({
        "ids": [f"d{i}" for i in range(n)],
        "embeddings": corpus,
        "metadatas": [{"g": i % 3} for i in range(n)],
    }).is_success()
    return index


def test_dot_is_accepted_and_reported_the_way_the_other_three_are():
    corpus = _dot_corpus()
    index = _dot_index(corpus)
    assert index.space == "dot"
    assert index.get_space() == "dot"
    assert "space=dot" in index.info()
    assert index.get_stats()["space"] == "dot"
    # Case insensitive, as every other space name is.
    assert VectorDatabase().create("hnsw", dim=4, space="DOT").space == "dot"


def test_an_unsupported_space_names_dot_among_the_four():
    with pytest.raises(RuntimeError, match=r"'cosine', 'l2', 'l1', 'dot'"):
        VectorDatabase().create("hnsw", dim=4, space="inner_product")


def test_dot_does_not_normalise_what_it_stores():
    """The one difference between this space and cosine, and it is the point.

    Under cosine a stored vector comes back as the unit vector, so the length a
    caller supplied is gone. Under dot the length is part of the score, so it is
    kept and `get_records` returns what was inserted.
    """
    corpus = _dot_corpus(n=40)
    index = _dot_index(corpus)
    back = np.asarray(index.get_records("d0", return_vector=True)[0]["vector"],
                      dtype=np.float32)
    assert np.allclose(back, corpus[0], atol=0, rtol=0)
    assert np.linalg.norm(back) > 2.0, "the corpus is scaled, so this is not a unit vector"


def test_a_dot_score_is_one_minus_the_inner_product():
    """What the score field means, which is what the README has to say."""
    corpus = _dot_corpus(n=120)
    index = _dot_index(corpus)
    query = corpus[7]
    page = index.search(query.tolist(), top_k=8, ef_search=120)
    truth = 1.0 - (corpus.astype(np.float64) @ query.astype(np.float64))
    for hit in page:
        want = truth[int(hit["id"][1:])]
        assert abs(hit["score"] - want) <= 1e-4 + 1e-5 * abs(want), hit

    # Negative scores are ordinary here and are ordered like any other.
    assert min(hit["score"] for hit in page) < 0.0
    scores = [hit["score"] for hit in page]
    assert scores == sorted(scores)


def test_a_dot_index_round_trips_through_a_save_and_a_load(tmp_path):
    corpus = _dot_corpus(n=200)
    index = _dot_index(corpus)
    rng = np.random.default_rng(99)
    queries = rng.standard_normal((5, corpus.shape[1])).astype(np.float32)
    before = [[(h["id"], h["score"]) for h in index.search(q.tolist(), top_k=10,
                                                           ef_search=200)]
              for q in queries]

    path = tmp_path / "dot.zdb"
    index.save(str(path))
    assert json.loads((path / "config.json").read_text(encoding="utf-8"))["space"] == "dot"

    loaded = VectorDatabase().load(str(path))
    assert loaded.space == "dot"
    after = [[(h["id"], h["score"]) for h in loaded.search(q.tolist(), top_k=10,
                                                           ef_search=200)]
             for q in queries]
    assert after == before

    # The graph came back from the dump rather than being rebuilt, which is what
    # a new GraphKind discriminant has to get right.
    assert np.allclose(
        np.asarray(loaded.get_records("d0", return_vector=True)[0]["vector"]),
        corpus[0], atol=0, rtol=0)


def test_a_filtered_dot_search_scores_with_the_inner_product():
    """The exact scan path, which reads `raw_distance_fn` rather than the graph.

    Without a `dot` arm there it scored with cosine, so a filtered search and an
    unfiltered one on the same index ranked by two different quantities.
    """
    corpus = _dot_corpus(n=180)
    index = _dot_index(corpus)
    query = corpus[3]
    page = index.search(query.tolist(), filter={"g": 0}, top_k=6)
    truth = 1.0 - (corpus.astype(np.float64) @ query.astype(np.float64))
    matching = [i for i in range(len(corpus)) if i % 3 == 0]
    want = sorted(matching, key=lambda i: (truth[i], f"d{i}"))[:6]
    assert [h["id"] for h in page] == [f"d{i}" for i in want]
    for hit in page:
        expected = truth[int(hit["id"][1:])]
        assert abs(hit["score"] - expected) <= 1e-4 + 1e-5 * abs(expected)


def test_dot_cannot_be_quantized():
    """Named rather than served wrongly.

    A quantized graph scores against a table of squared L2 distances to the
    codebook whatever the space. For cosine that orders identically, because
    every vector is a unit vector. For an inner product it does not, because the
    stored vector's own length enters the L2 and not the inner product.
    """
    with pytest.raises(RuntimeError, match=r"space='dot' cannot be quantized"):
        VectorDatabase().create(
            "hnsw", dim=16, space="dot", expected_size=20000,
            quantization_config={"type": "pq", "subvectors": 4, "bits": 8,
                                 "training_size": 1000},
        )


def test_a_hand_assembled_dot_directory_claiming_quantization_is_refused(tmp_path):
    """The same rule at load, because a config.json is not a caller."""
    corpus = _dot_corpus(n=60)
    index = _dot_index(corpus)
    path = tmp_path / "forged.zdb"
    index.save(str(path))

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files_included"].append("quantization.json")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (path / "quantization.json").write_text(json.dumps({
        "type": "pq", "subvectors": 4, "bits": 8, "training_size": 1000,
        "max_training_vectors": None, "storage_mode": "quantized_only",
        "is_trained": False, "training_completed_at": None, "memory_stats": None,
        "pq_config": {"dim": 16, "sub_dim": 4, "num_centroids": 256},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match=r"space='dot' cannot be quantized"):
        VectorDatabase().load(str(path))


def test_l1_cannot_be_quantized():
    """Refused for the same reason `dot` is, and measured rather than argued.

    A quantized graph ranks by a squared L2 distance to the codebook whatever
    space is declared. Against the query `[0, 0]` the point `[2, 0]` is at L1
    2.0 and squared L2 4.0 while `[1.1, 1.1]` is at L1 2.2 and squared L2 2.42,
    so the two rank that pair in opposite orders and no rescaling joins them.
    `the_l1_counterexample_is_ordered_by_squared_l2` in `distance.rs` holds the
    arithmetic against the live scorer.
    """
    with pytest.raises(RuntimeError, match=r"space='l1' cannot be quantized"):
        VectorDatabase().create(
            "hnsw", dim=16, space="l1", expected_size=20000,
            quantization_config={"type": "pq", "subvectors": 4, "bits": 8,
                                 "training_size": 1000},
        )


def test_a_hand_assembled_l1_directory_claiming_quantization_is_refused(tmp_path):
    """The same rule at load, because a config.json is not a caller.

    A directory saved by a release that allowed the pair reaches this too, and
    the message names the remedy rather than only the refusal.
    """
    rng = np.random.default_rng(1105)
    corpus = rng.normal(size=(60, 16)).astype(np.float32)
    index = VectorDatabase().create("hnsw", dim=16, space="l1", expected_size=100)
    index.add({"ids": [f"d{i}" for i in range(len(corpus))],
               "vectors": corpus.tolist()})
    path = tmp_path / "forged.zdb"
    index.save(str(path))

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files_included"].append("quantization.json")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (path / "quantization.json").write_text(json.dumps({
        "type": "pq", "subvectors": 4, "bits": 8, "training_size": 1000,
        "max_training_vectors": None, "storage_mode": "quantized_only",
        "is_trained": False, "training_completed_at": None, "memory_stats": None,
        "pq_config": {"dim": 16, "sub_dim": 4, "num_centroids": 256},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match=r"space='l1' cannot be quantized"):
        VectorDatabase().load(str(path))


def test_the_refusal_message_names_the_remedy():
    """Both refused pairs say what to do instead, not only what is refused."""
    for space, remedy in (("dot", "space='cosine'"), ("l1", "space='l2'")):
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            VectorDatabase().create(
                "hnsw", dim=16, space=space, expected_size=20000,
                quantization_config={"type": "pq", "subvectors": 4, "bits": 8,
                                     "training_size": 1000},
            )
        text = str(excinfo.value)
        assert remedy in text
        assert "drop quantization_config" in text


def test_an_unquantized_l1_index_is_untouched():
    """Only the pair is refused. l1 on its own is unchanged."""
    rng = np.random.default_rng(4)
    corpus = rng.normal(size=(200, 16)).astype(np.float32)
    index = VectorDatabase().create("hnsw", dim=16, space="l1", expected_size=200)
    index.add({"ids": [f"d{i}" for i in range(len(corpus))],
               "vectors": corpus.tolist()})
    query = corpus[7]
    page = index.search(query.tolist(), top_k=5)
    truth = np.abs(corpus - query).sum(axis=1)
    assert [h["id"] for h in page] == [f"d{i}" for i in np.argsort(truth)[:5]]
    for hit in page:
        assert abs(hit["score"] - truth[int(hit["id"][1:])]) <= 1e-3


def test_dot_and_cosine_agree_on_normalised_input():
    """`1 - dot` is the cosine distance once the input is a unit vector.

    This is why the constant is there rather than plain negation: a caller who
    has already normalised sees the same number under either space.
    """
    rng = np.random.default_rng(17)
    raw = rng.standard_normal((150, 24)).astype(np.float32)
    unit = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)

    pages = {}
    for space in ("cosine", "dot"):
        index = VectorDatabase().create("hnsw", dim=24, space=space, expected_size=300)
        assert index.add({"ids": [f"u{i}" for i in range(150)],
                          "embeddings": unit}).is_success()
        pages[space] = [(h["id"], h["score"])
                        for h in index.search(unit[5].tolist(), top_k=10, ef_search=150)]

    assert [i for i, _ in pages["cosine"]] == [i for i, _ in pages["dot"]]
    for (_, a), (_, b) in zip(pages["cosine"], pages["dot"]):
        assert abs(a - b) < 1e-5, (a, b)


# ------------------------------------------------------------
# top_k and ef_search ceilings
# ------------------------------------------------------------
def test_top_k_and_ef_search_have_ceilings():
    """Both sized the candidate heaps with no upper bound, and 2**40 killed the interpreter.

    The traversal allocates two heaps of 8 bytes a slot from the search width
    before it visits a node, and top_k reaches the same width through the
    default ef_search of twice top_k. search(top_k=2**40) asked for 16 TB,
    search(ef_search=2**40) for 8 TB, and top_k=2**33 for 137 GB, each ending
    the process with exit status 3221226505 on a healthy index. No subprocess:
    the bound is checked before anything is allocated, so a broken bound would
    return results or die rather than raise, and either fails the assertions.
    """
    index = VectorDatabase().create("hnsw", dim=4, expected_size=4)
    index.add({"ids": ["a", "b"], "embeddings": [[1.0, 0, 0, 0], [0, 1.0, 0, 0]]})
    query = np.array([1.0, 0, 0, 0], dtype=np.float32)
    with pytest.raises(ValueError, match="top_k must be at most 65536, got 65537"):
        index.search(query, top_k=65_537)
    with pytest.raises(ValueError, match="top_k must be at most 65536, got 1099511627776"):
        index.search(query, top_k=1 << 40)
    with pytest.raises(ValueError, match="ef_search must be at most 131072, got 131073"):
        index.search(query, top_k=2, ef_search=131_073)
    with pytest.raises(ValueError, match="ef_search must be at most 131072, got 1099511627776"):
        index.search(query, top_k=2, ef_search=1 << 40)
    # The ceilings are inclusive, and a page wider than the index is a result.
    assert len(index.search(query, top_k=65_536)) == 2
    assert len(index.search(query, top_k=2, ef_search=131_072)) == 2
    # Zero stays accepted, as test_compaction.py already holds.
    assert index.search(query, top_k=0, ef_search=0) == []
