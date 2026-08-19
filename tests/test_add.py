"""Vector insertion across the five input formats, overwrite, and add error handling."""

import numpy as np
import pytest
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

# ------------------------------------------------------------
# Test 56: overwrite=False against an id that already exists
# ------------------------------------------------------------
def test_add_overwrite_false_existing_id_reports_an_error():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)

    index.add({"id": "doc1", "values": [0.1, 0.2], "metadata": {"version": "v1"}})

    # Current behaviour, asserted rather than expected. add_single_vector does
    # raise a ValueError naming the id, but add() catches every per record
    # error and folds it into AddResult, so nothing reaches the caller as an
    # exception. The expectation this violates is that a single record add
    # which inserts nothing raises rather than returning a result object the
    # caller has to inspect.
    result = index.add(
        {"id": "doc1", "values": [0.9, 0.9], "metadata": {"version": "v2"}},
        overwrite=False,
    )

    assert result.total_inserted == 0
    assert result.total_errors == 1
    assert not result.is_success()
    assert len(result.errors) == 1
    assert "doc1" in result.errors[0]
    assert "already exists" in result.errors[0]
    assert "ValueError" in result.errors[0]

    # vector_shape reports the input shape, not the inserted shape.
    assert result.vector_shape == (1, 2)

    # The stored record is untouched.
    assert int(index.get_stats()["total_vectors"]) == 1
    stored = index.get_records("doc1", return_vector=True)
    assert len(stored) == 1
    assert stored[0]["metadata"]["version"] == "v1"
    assert_vectors_close(stored[0]["vector"], [0.1, 0.2], space="cosine")

# ------------------------------------------------------------
# Test 57: overwrite=False against an id that does not exist
# ------------------------------------------------------------
def test_add_overwrite_false_new_id_inserts_normally():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)

    index.add({"id": "doc1", "values": [0.1, 0.2], "metadata": {"version": "v1"}})

    result = index.add(
        {"id": "doc2", "values": [0.3, 0.4], "metadata": {"version": "v1"}},
        overwrite=False,
    )

    assert result.total_inserted == 1
    assert result.total_errors == 0
    assert result.is_success()
    assert len(result.errors) == 0
    assert result.vector_shape == (1, 2)

    assert int(index.get_stats()["total_vectors"]) == 2
    assert index.contains("doc2")
    stored = index.get_records("doc2", return_vector=True)
    assert stored[0]["metadata"]["version"] == "v1"
    assert_vectors_close(stored[0]["vector"], [0.3, 0.4], space="cosine")

# ------------------------------------------------------------
# Test 58: overwrite=True passed explicitly
# ------------------------------------------------------------
def test_add_overwrite_true_replaces_without_duplicating():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)

    index.add({"id": "doc1", "values": [0.1, 0.2], "metadata": {"version": "v1"}})
    assert int(index.get_stats()["total_vectors"]) == 1

    result = index.add(
        {"id": "doc1", "values": [0.9, 0.1], "metadata": {"version": "v2"}},
        overwrite=True,
    )

    assert result.total_inserted == 1
    assert result.total_errors == 0
    assert result.is_success()

    # The existing record is removed before the new one is inserted, so the
    # count does not grow and the id resolves to exactly one record.
    assert int(index.get_stats()["total_vectors"]) == 1
    assert len(index.get_records("doc1", return_vector=False)) == 1
    assert len(index.list(number=10)) == 1

    stored = index.get_records("doc1", return_vector=True)
    assert stored[0]["metadata"]["version"] == "v2"
    assert_vectors_close(stored[0]["vector"], [0.9, 0.1], space="cosine")

# ------------------------------------------------------------
# Test 59: A mixed batch under overwrite=False
# ------------------------------------------------------------
def test_add_overwrite_false_mixed_batch_reports_per_record():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=10)

    index.add([
        {"id": "existing1", "values": [0.1, 0.2], "metadata": {"version": "v1"}},
        {"id": "existing2", "values": [0.3, 0.4], "metadata": {"version": "v1"}},
    ])

    # The batch is not rejected as a whole. Each record is attempted in turn
    # and the two collisions are reported through AddResult while the new
    # record is inserted.
    result = index.add([
        {"id": "existing1", "values": [0.5, 0.5], "metadata": {"version": "v2"}},
        {"id": "brand_new", "values": [0.6, 0.7], "metadata": {"version": "v1"}},
        {"id": "existing2", "values": [0.8, 0.9], "metadata": {"version": "v2"}},
    ], overwrite=False)

    assert result.total_inserted == 1
    assert result.total_errors == 2
    assert not result.is_success()
    assert len(result.errors) == 2
    assert "1 inserted" in result.summary()
    assert "2 errors" in result.summary()
    assert result.vector_shape == (3, 2)

    error_text = " ".join(result.errors)
    assert "existing1" in error_text
    assert "existing2" in error_text
    assert "brand_new" not in error_text

    # The new record landed and neither existing record was replaced.
    assert int(index.get_stats()["total_vectors"]) == 3
    assert index.contains("brand_new")
    assert index.get_records("existing1", return_vector=False)[0]["metadata"]["version"] == "v1"
    assert index.get_records("existing2", return_vector=False)[0]["metadata"]["version"] == "v1"

# ------------------------------------------------------------
# Test 60: Metadata is replaced rather than merged on overwrite
# ------------------------------------------------------------
def test_add_overwrite_replaces_metadata_wholesale():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)

    index.add({
        "id": "doc1",
        "values": [0.1, 0.2],
        "metadata": {"version": "v1", "only_in_first": "kept?"},
    })

    # The record is removed and reinserted, so the second call decides the
    # whole metadata map. A key present only in the first call does not
    # survive, which makes add an overwrite rather than an upsert of fields.
    index.add({
        "id": "doc1",
        "values": [0.3, 0.4],
        "metadata": {"version": "v2", "only_in_second": "added"},
    })

    metadata = index.get_records("doc1", return_vector=False)[0]["metadata"]
    assert metadata == {"version": "v2", "only_in_second": "added"}
    assert "only_in_first" not in metadata

    # An overwrite carrying empty metadata clears the map entirely.
    index.add({"id": "doc1", "values": [0.5, 0.6], "metadata": {}})
    assert index.get_records("doc1", return_vector=False)[0]["metadata"] == {}

# ------------------------------------------------------------
# Test 93: vector_shape on a batch that contains errors
# ------------------------------------------------------------
def test_add_result_vector_shape_counts_errors():
    """vector_shape describes the input, not what was inserted.

    The Rust computes it as parsed records plus parse errors, so a batch that
    partly failed still reports its own length. Every existing assertion on
    vector_shape is against a clean batch, where the two are the same number.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=10)

    result = index.add([
        {"id": "ok1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {}},
        {"id": "bad", "values": [0.1, 0.2], "metadata": {}},  # Wrong dimension
        {"id": "ok2", "values": [0.5, 0.6, 0.7, 0.8], "metadata": {}},
    ])

    assert result.total_inserted == 2
    assert result.total_errors == 1
    # Three rows, not two, and the second element is the index dimension rather
    # than the dimension of any input vector.
    assert result.vector_shape == (3, 4)
    assert "(3, 4)" in repr(result)

    # Every record failing still reports the input shape.
    all_bad = index.add([
        {"id": "bad1", "values": [0.1], "metadata": {}},
        {"id": "bad2", "values": [0.1, 0.2, 0.3, 0.4, 0.5], "metadata": {}},
    ])
    assert all_bad.total_inserted == 0
    assert all_bad.total_errors == 2
    assert all_bad.vector_shape == (2, 4)

    # An add carrying no records at all reports a zero row shape.
    empty = index.add([])
    assert empty.total_inserted == 0
    assert empty.total_errors == 0
    assert empty.vector_shape == (0, 4)
    assert empty.is_success()

# ------------------------------------------------------------
# Test 98: non-finite values are rejected on every add path
# ------------------------------------------------------------
def test_add_rejects_non_finite_values():
    """A NaN in the graph degrades every later query, not only its own.

    The search path validated finiteness and add did not, on either NumPy
    branch, so a NaN vector inserted with zero errors. add reports per record
    for every other error, so it reports per record for this one too, and the
    surrounding good records are still inserted.
    """
    vdb = VectorDatabase()

    for bad in (float("nan"), float("inf"), float("-inf")):
        index = vdb.create("hnsw", dim=4, expected_size=10)

        # Bare NumPy array. Rows are named by position, since none carry an id.
        result = index.add(
            np.array(
                [[0.1, 0.2, 0.3, 0.4], [bad, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                dtype=np.float32,
            )
        )
        assert result.total_inserted == 2
        assert result.total_errors == 1
        assert "Vector row_1" in result.errors[0]
        assert "invalid value at index 0" in result.errors[0]
        assert index.get_vector_count() == 2

        # NumPy inside a batch dict. The offending record is named by its id.
        index = vdb.create("hnsw", dim=4, expected_size=10)
        result = index.add({
            "vectors": np.array(
                [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, bad]], dtype=np.float32
            ),
            "ids": ["good", "poisoned"],
            "metadatas": [{"n": 1}, {"n": 2}],
        })
        assert result.total_inserted == 1
        assert result.total_errors == 1
        assert "Vector poisoned" in result.errors[0]
        assert "invalid value at index 3" in result.errors[0]
        assert index.contains("good")
        assert not index.contains("poisoned")

        # The list and dict paths already validated, and still do.
        index = vdb.create("hnsw", dim=4, expected_size=10)
        result = index.add([
            {"id": "ok", "values": [0.1, 0.2, 0.3, 0.4]},
            {"id": "bad", "values": [0.1, bad, 0.3, 0.4]},
        ])
        assert result.total_inserted == 1
        assert result.total_errors == 1
        assert "Vector bad" in result.errors[0]

        single = index.add({"id": "solo", "values": [bad, 0.2, 0.3, 0.4]})
        assert single.total_inserted == 0
        assert single.total_errors == 1


# ------------------------------------------------------------
# Test 99: a rejected NumPy row does not burn an internal id
# ------------------------------------------------------------
def test_rejected_numpy_row_keeps_generated_ids_contiguous():
    """A row that is never stored should not advance the id counter."""
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=10)

    result = index.add(
        np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [float("nan"), 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ],
            dtype=np.float32,
        )
    )

    assert result.total_inserted == 2
    stored = sorted(record_id for record_id, _ in index.list(10))
    assert stored == ["vec_1", "vec_2"]


# ------------------------------------------------------------
# Test 100: a batch whose parallel arrays disagree in length raises
# ------------------------------------------------------------
def test_add_batch_length_disagreement_raises():
    """Three ids and two vectors used to insert two records and drop an id.

    add({"ids": ["c", "d", "e"], "embeddings": <two rows>}) returned
    AddResult(inserted=2, errors=0). Nothing said the third id was gone, and
    nothing could say which record the caller meant, so it raises rather than
    rejecting a record. The reverse shape was worse: two ids and three vectors
    stored the third under a generated vec_N, which no caller can look up.
    """
    vdb = VectorDatabase()

    rows2 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    rows3 = rows2 + [[0.0, 0.0, 1.0, 0.0]]
    np2 = np.array(rows2, dtype=np.float32)
    np3 = np.array(rows3, dtype=np.float32)

    # Surplus ids, on both the list and the NumPy branch and under every
    # spelling of the vector key.
    for payload in (
        {"ids": ["c", "d", "e"], "embeddings": rows2},
        {"ids": ["c", "d", "e"], "embeddings": np2},
        {"ids": ["c", "d", "e"], "vectors": rows2},
        {"ids": ["c", "d", "e"], "vectors": np2},
        {"ids": ["c", "d", "e"], "values": rows2},
    ):
        index = vdb.create("hnsw", dim=4, expected_size=10)
        with pytest.raises(ValueError) as excinfo:
            index.add(payload)
        message = str(excinfo.value)
        # Both lengths and the short field are named.
        assert "3 entries under 'ids'" in message
        assert "2 under" in message
        assert "is the short one" in message
        # Nothing was inserted, so a caller can retry the whole call.
        assert len(index) == 0

    # Surplus vectors, where the short field is 'ids'.
    for payload in (
        {"ids": ["c", "d"], "embeddings": rows3},
        {"ids": ["c", "d"], "embeddings": np3},
    ):
        index = vdb.create("hnsw", dim=4, expected_size=10)
        with pytest.raises(ValueError) as excinfo:
            index.add(payload)
        assert "'ids' is the short one" in str(excinfo.value)
        assert len(index) == 0


# ------------------------------------------------------------
# Test 101: the metadata arrays are held to the same length rule
# ------------------------------------------------------------
def test_add_batch_metadata_length_disagreement_raises():
    """A short metadatas list silently gave the trailing records no metadata.

    Both spellings are checked, and 'metadata' only where 'metadatas' is
    absent, which is exactly when the parsers read it.
    """
    vdb = VectorDatabase()

    rows3 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    np3 = np.array(rows3, dtype=np.float32)
    ids = ["c", "d", "e"]

    for rows in (rows3, np3):
        for key in ("metadatas", "metadata"):
            index = vdb.create("hnsw", dim=4, expected_size=10)
            with pytest.raises(ValueError) as excinfo:
                index.add({"ids": ids, "embeddings": rows, key: [{"a": 1}, {"a": 2}]})
            message = str(excinfo.value)
            assert f"2 entries under '{key}'" in message
            assert f"'{key}' is the short one" in message
            assert len(index) == 0

            # A surplus metadata entry is the same disagreement the other way.
            index = vdb.create("hnsw", dim=4, expected_size=10)
            with pytest.raises(ValueError):
                index.add({"ids": ids, "embeddings": rows,
                           key: [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]})
            assert len(index) == 0

    # 'metadatas' wins where both are present, and the ignored 'metadata' is
    # not held to the rule, because no parser reads it in that case.
    index = vdb.create("hnsw", dim=4, expected_size=10)
    result = index.add({
        "ids": ids,
        "embeddings": np3,
        "metadatas": [{"a": 1}, {"a": 2}, {"a": 3}],
        "metadata": [{"b": 9}],
    })
    assert result.total_inserted == 3
    assert index.get_records("c", return_vector=False)[0]["metadata"] == {"a": 1}


# ------------------------------------------------------------
# Test 102: a parallel array that is not a list raises
# ------------------------------------------------------------
def test_add_batch_non_list_ids_raises():
    """A tuple or an ndarray of ids was discarded whole on the NumPy branch.

    parse_numpy_with_context resolved ids with cast::<PyList>().ok(), so any
    other sequence type became None and every record took a generated id. The
    list branch raised on the same input, so one mistake had two behaviours.
    """
    vdb = VectorDatabase()

    rows3 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    np3 = np.array(rows3, dtype=np.float32)

    for rows in (rows3, np3):
        for ids in (("a", "b", "c"), np.array(["a", "b", "c"])):
            index = vdb.create("hnsw", dim=4, expected_size=10)
            with pytest.raises(TypeError) as excinfo:
                index.add({"ids": ids, "embeddings": rows})
            assert "'ids' to be a list" in str(excinfo.value)
            assert len(index) == 0


# ------------------------------------------------------------
# Test 103: agreeing lengths and the other four input shapes are unchanged
# ------------------------------------------------------------
def test_add_batch_length_rule_leaves_valid_input_alone():
    """The rule fires only on a disagreement, and only on the batch dict.

    The empty case, the ordinary case and the four shapes that carry no
    parallel arrays at all are all as they were.
    """
    vdb = VectorDatabase()

    rows3 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    np3 = np.array(rows3, dtype=np.float32)
    ids = ["c", "d", "e"]

    # Equal lengths, both branches, with and without metadata.
    for rows in (rows3, np3):
        index = vdb.create("hnsw", dim=4, expected_size=10)
        assert index.add({"ids": ids, "embeddings": rows}).total_inserted == 3
        index = vdb.create("hnsw", dim=4, expected_size=10)
        result = index.add({"ids": ids, "embeddings": rows,
                            "metadatas": [{"a": 1}, {"a": 2}, {"a": 3}]})
        assert result.total_inserted == 3
        assert result.total_errors == 0

    # No ids at all is not a disagreement. Every record takes a generated id,
    # which is what a caller who omits them asked for.
    index = vdb.create("hnsw", dim=4, expected_size=10)
    assert index.add({"embeddings": np3}).total_inserted == 3
    assert sorted(record_id for record_id, _ in index.list(10)) == ["vec_1", "vec_2", "vec_3"]

    # An empty batch agrees with itself at zero.
    index = vdb.create("hnsw", dim=4, expected_size=10)
    empty = index.add({"ids": [], "embeddings": []})
    assert empty.total_inserted == 0
    assert empty.total_errors == 0

    # A batch carrying no recognised vector field is still reported through
    # AddResult rather than raised, because it names no records to lose.
    index = vdb.create("hnsw", dim=4, expected_size=10)
    missing = index.add({"ids": ["a"], "junk": 1})
    assert missing.total_inserted == 0
    assert missing.total_errors == 1
    assert "Missing vector data" in missing.errors[0]

    # The four shapes with no parallel arrays are untouched, including the
    # per-record rejection a bad vector still gets inside a list batch.
    index = vdb.create("hnsw", dim=4, expected_size=10)
    assert index.add([{"id": "a", "values": rows3[0]},
                      {"id": "b", "values": rows3[1]}]).total_inserted == 2
    index = vdb.create("hnsw", dim=4, expected_size=10)
    assert index.add(np3).total_inserted == 3
    index = vdb.create("hnsw", dim=4, expected_size=10)
    assert index.add({"id": "a", "values": rows3[0], "metadata": {"x": 1}}).total_inserted == 1
    index = vdb.create("hnsw", dim=4, expected_size=10)
    assert index.add(np.array(rows3[0], dtype=np.float32)).total_inserted == 1
    index = vdb.create("hnsw", dim=4, expected_size=10)
    mixed = index.add([{"id": "ok", "values": rows3[0]}, {"id": "bad", "values": [1.0, 2.0]}])
    assert mixed.total_inserted == 1
    assert mixed.total_errors == 1
