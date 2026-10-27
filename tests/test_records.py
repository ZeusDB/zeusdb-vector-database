"""Record retrieval, removal, listing and index level metadata."""

import numpy as np
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
        "creator": "data-platform-team",
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
    assert index.get_metadata("creator") == "data-platform-team"
    assert index.get_metadata("version") == "0.1"
    assert index.get_metadata("nonexistent") is None
    
    # Test all metadata retrieval
    all_meta = index.get_all_metadata()
    assert len(all_meta) == len(metadata)
    assert all_meta["creator"] == "data-platform-team"
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

# ------------------------------------------------------------
# Test 91: A removed point stops appearing in search results
# ------------------------------------------------------------
def test_remove_point_removes_from_search_results():
    """Removal was only ever checked through contains, never through search."""
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=16, expected_size=500)

    # A local Generator keeps the draws reproducible without touching the
    # global numpy random state.
    rng = np.random.default_rng(20260802)
    count = 200
    vectors = rng.random((count, 16)).astype(np.float32)
    ids = [f"doc_{i}" for i in range(count)]
    assert index.add({
        "ids": ids,
        "embeddings": vectors,
        "metadatas": [{"i": i} for i in range(count)],
    }).is_success()

    # The exact match for its own vector is the first hit before removal.
    query = vectors[7].tolist()
    before = index.search(query, top_k=10)
    assert before[0]["id"] == "doc_7"
    assert len(before) == 10

    assert index.remove_point("doc_7") is True
    assert not index.contains("doc_7")
    assert index.get_vector_count() == count - 1
    assert index.get_records("doc_7") == []

    # The removed id is gone from the ranking. This is the part that holds.
    after = index.search(query, top_k=10)
    assert all(hit["id"] != "doc_7" for hit in after)

    # The page is full. remove_point_internal still cannot delete the graph
    # node, so the removed point keeps its vector and both directions of its
    # adjacency, but search now passes a live-record predicate into the
    # traversal, so that node routes the search without consuming a result
    # slot. A search over 199 remaining records returns a full page of 10.
    # Without that predicate the page would come back one short.
    assert len(after) == 9 + 1
    assert len(index.search(query, top_k=5)) == 5
    assert len(index.search(query, top_k=20)) == 20

    # ef_search is no longer load bearing here, and was never able to backfill
    # the slot when the shortfall existed.
    assert len(index.search(query, top_k=10, ef_search=200)) == 10

    # A query whose candidate window never reaches the removed node was
    # unaffected before the fix, which is what made the shortfall easy to miss.
    assert len(index.search(vectors[150].tolist(), top_k=10)) == 10

    # Re-adding the same id still does not reclaim the dead node, which is what
    # compact() is for, but the record comes back, is returned, and the page
    # stays full.
    index.add({"id": "doc_7", "values": query, "metadata": {"i": 7}})
    assert index.contains("doc_7")
    revived = index.search(query, top_k=10)
    assert any(hit["id"] == "doc_7" for hit in revived)
    assert len(revived) == 10
    assert index.remove_point("doc_7") is True

    # A wider removal is not visible in a wider search either.
    removed = {f"doc_{i}" for i in range(10, 30)}
    for doc_id in sorted(removed):
        assert index.remove_point(doc_id) is True

    assert index.get_vector_count() == count - 1 - len(removed)
    assert len(index.list(number=count)) == count - 1 - len(removed)

    wide = index.search(vectors[15].tolist(), top_k=50)
    assert removed.isdisjoint({hit["id"] for hit in wide})
    # A wider removal is not visible in a wider search either, and now it is not
    # visible because nothing is lost rather than because the loss is hidden.
    # The dead nodes inside the candidate window no longer take slots, so the
    # page is full. Without that, the page was merely non-empty.
    assert len(wide) == 50

    # The batch path agrees with the single path.
    batch = index.search(np.array([vectors[15], vectors[7]], dtype=np.float32), top_k=25)
    assert len(batch) == 2
    returned = {hit["id"] for result in batch for hit in result}
    assert returned.isdisjoint(removed | {"doc_7"})

    # A filtered search cannot resurrect them either.
    filtered = index.search(vectors[15].tolist(), filter={"i": {"gte": 10, "lt": 30}}, top_k=25)
    assert all(hit["id"] not in removed for hit in filtered)

# ------------------------------------------------------------
# Test 92: add_metadata merges on a second call
# ------------------------------------------------------------
def test_add_metadata_merges_on_a_second_call():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)

    index.add_metadata({"owner": "Alice", "version": "1"})
    index.add_metadata({"version": "2", "dataset": "docs_v2"})

    # The second call inserts key by key rather than replacing the map, so a
    # key present only in the first call survives and an overlapping key takes
    # the newer value. This is the opposite of what add() does to per record
    # metadata on an overwrite, which replaces the map wholesale.
    assert index.get_all_metadata() == {"owner": "Alice", "version": "2", "dataset": "docs_v2"}
    assert index.get_metadata("owner") == "Alice"
    assert index.get_metadata("version") == "2"
    assert index.get_metadata("dataset") == "docs_v2"

    # An empty call is a no op rather than a clear.
    index.add_metadata({})
    assert index.get_all_metadata() == {"owner": "Alice", "version": "2", "dataset": "docs_v2"}

    # Explicitly writing an empty string overwrites the value.
    index.add_metadata({"owner": ""})
    assert index.get_metadata("owner") == ""

    # The map returned is a copy, so mutating it does not reach the index.
    snapshot = index.get_all_metadata()
    snapshot["injected"] = "no"
    assert index.get_metadata("injected") is None


# ------------------------------------------------------------
# Metadata keys come back in code point order
# ------------------------------------------------------------
_ORDER_NAMES = ["zeta", "Alpha", "mid", "été", "beta", "_under"]


def _order_index(n=60):
    """An index whose records each name their fields in a rotated order, so
    no one order is the order the store first saw the names in."""
    index = VectorDatabase().create("hnsw", dim=4, expected_size=n)
    rng = np.random.default_rng(222)
    vectors = rng.random((n, 4), dtype=np.float32)
    records = []
    for i in range(n):
        shift = i % len(_ORDER_NAMES)
        rotated = _ORDER_NAMES[shift:] + _ORDER_NAMES[:shift]
        metadata = {name: i for name in rotated}
        metadata["nested"] = {"z": 1, "a": 2}
        records.append({"id": f"r{i:03d}", "vector": vectors[i].tolist(), "metadata": metadata})
    assert index.add(records).is_success()
    return index, vectors


def _orders(index, vectors, n=60):
    """The key order of every record's metadata, by the call that handed it."""
    ids = [f"r{i:03d}" for i in range(n)]
    query = vectors[0].tolist()
    pages = {
        "search": [h["metadata"] for h in index.search(query, top_k=n)],
        "search of a batch": [h["metadata"] for h in index.search([query], top_k=n)[0]],
        "query": [h["metadata"] for h in index.query([{"vector": query}], top_k=n)],
        "get_records": [r["metadata"] for r in index.get_records(ids, return_vector=False)],
        "list": [metadata for _, metadata in index.list(number=n)],
    }
    return {call: [list(metadata) for metadata in dicts] for call, dicts in pages.items()}


def test_every_call_hands_metadata_keys_over_in_code_point_order():
    """`search`, `query`, `get_records` and `list` hand every record's keys over
    in one order, being code point order, which is the order a nested mapping's
    keys come in, whatever order the store first saw the names in."""
    index, vectors = _order_index()
    expected = sorted(_ORDER_NAMES + ["nested"])
    assert expected == ["Alpha", "_under", "beta", "mid", "nested", "zeta", "été"]
    for call, orders in _orders(index, vectors).items():
        assert len(orders) == 60, call
        assert all(order == expected for order in orders), call
    record = index.get_records("r000", return_vector=False)[0]
    assert list(record["metadata"]["nested"]) == ["a", "z"]


def test_the_reports_hand_their_keys_over_in_code_point_order():
    index, _ = _order_index(20)
    index.add_metadata({"zulu": "1", "alpha": "2", "Mike": "3"})
    stats = index.get_stats()
    assert list(stats) == sorted(stats)
    assert list(index.get_all_metadata()) == ["Mike", "alpha", "zulu"]
    performance = index.get_performance_info()
    assert list(performance) == sorted(performance)
    benchmark = index.benchmark_concurrent_reads(4, max_threads=1)
    assert list(benchmark) == sorted(benchmark)


def test_the_key_order_is_the_same_in_another_process():
    """A second process, whose hash maps are seeded afresh and whose store sees
    the names first in another order, hands the keys over in the same order."""
    import json
    import subprocess
    import sys

    script = r"""
import json
import numpy as np
from zeusdb_vector_database import VectorDatabase
names = ["mid", "_under", "zeta", "été", "Alpha", "beta"]
index = VectorDatabase().create("hnsw", dim=4)
vectors = np.random.default_rng(7).random((5, 4), dtype=np.float32)
index.add([{"id": f"p{i}", "vector": vectors[i].tolist(),
            "metadata": {name: i for name in names[i:] + names[:i]}} for i in range(5)])
print(json.dumps({
    "search": [list(h["metadata"]) for h in index.search(vectors[0].tolist(), top_k=5)],
    "get_records": [list(r["metadata"]) for r in index.get_records([f"p{i}" for i in range(5)])],
    "list": [list(m) for _, m in index.list(5)],
    "stats": list(index.get_stats())[:6],
}))
"""
    out = subprocess.run([sys.executable, "-c", script], check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    seen = json.loads(out.strip().splitlines()[-1])
    expected = sorted(_ORDER_NAMES)
    for call in ("search", "get_records", "list"):
        assert all(order == expected for order in seen[call]), call
    index, _ = _order_index(5)
    assert seen["stats"] == list(index.get_stats())[:6]
