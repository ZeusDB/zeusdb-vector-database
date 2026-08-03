"""Removal and overwrite leave a node in the graph. These tests cover the two
halves of the fix, excluding those nodes from traversal and reclaiming them.

A removed record clears from every storage map but its graph node stays, keeping
its vector and both directions of adjacency. `add(overwrite=True)` is a removal
followed by an insertion, so an update leaves the same node behind. Search passes
a live-record predicate into the traversal so those nodes never consume a result
slot, and `compact()` rebuilds the graph to reclaim them.
"""

import numpy as np
import pytest

from zeusdb_vector_database import VectorDatabase

DIM = 16
N = 400
TOP_K = 10

# A trained product quantizer needs at least this many records, so the quantized
# tests are built once at this size and the raw tests stay small.
PQ_DIM = 32
PQ_N = 1200
PQ_CONFIG = {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000}


def unit(vectors):
    """Cosine is the default space and the index normalises on the way in, so the
    fixtures normalise too and brute force over them agrees with the index."""
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def clustered(n, dim, seed):
    """Ten Gaussian clusters. Uniform noise gives every point roughly the same
    distance to every other, which hides exactly the neighbourhood effects here."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((10, dim))
    points = centres[rng.integers(0, 10, size=n)] + 0.15 * rng.standard_normal((n, dim))
    return unit(points)


def build_raw(n=N, dim=DIM, seed=1):
    vectors = clustered(n, dim, seed)
    index = VectorDatabase().create("hnsw", dim=dim, expected_size=n)
    result = index.add({"ids": [f"doc_{i}" for i in range(n)], "embeddings": vectors})
    assert result.is_success()
    return index, vectors


def build_quantized(seed=2):
    vectors = clustered(PQ_N, PQ_DIM, seed)
    index = VectorDatabase().create(
        "hnsw", dim=PQ_DIM, expected_size=PQ_N, quantization_config=PQ_CONFIG
    )
    result = index.add({"ids": [f"doc_{i}" for i in range(PQ_N)], "embeddings": vectors})
    assert result.is_success()
    assert index.is_quantized(), "training must complete before these tests mean anything"
    return index, vectors


def nearest(vectors, query, count):
    """Indices of the `count` nearest rows, by cosine on unit vectors."""
    return np.argsort(-(vectors @ query))[:count]


def ids_of(results):
    return [r["id"] for r in results]


def graph_nodes(index):
    return int(index.get_stats()["graph_nodes"])


@pytest.fixture(scope="module")
def quantized_fixture():
    """One trained quantized index shared by the read-only quantized tests. The
    tests that mutate it build their own."""
    return build_quantized()


# ------------------------------------------------------------
# Deleted records do not reduce the results returned
# ------------------------------------------------------------
def test_search_after_deletes_returns_full_top_k():
    index, vectors = build_raw()
    query = vectors[0]

    # Delete the whole neighbourhood the query lands in, so every stranded node
    # sits exactly where the traversal will find it.
    doomed = nearest(vectors, query, 40)
    for i in doomed:
        assert index.remove_point(f"doc_{i}")

    results = index.search(query.tolist(), top_k=TOP_K)
    assert len(results) == TOP_K
    returned = set(ids_of(results))
    assert returned.isdisjoint({f"doc_{i}" for i in doomed})

    # The same claim at half the index removed, which is where relay 29 measured
    # 4.83 results of 10.
    for i in range(N // 2, N):
        index.remove_point(f"doc_{i}")
    for probe in range(20):
        assert len(index.search(vectors[probe].tolist(), top_k=TOP_K)) == TOP_K


def test_deleted_records_never_appear_in_results():
    index, vectors = build_raw(n=200, seed=3)
    removed = {f"doc_{i}" for i in range(0, 200, 2)}
    for doc_id in removed:
        index.remove_point(doc_id)

    for probe in range(15):
        results = index.search(vectors[probe].tolist(), top_k=TOP_K)
        assert len(results) == TOP_K
        assert set(ids_of(results)).isdisjoint(removed)


# ------------------------------------------------------------
# An overwritten record returns exactly once
# ------------------------------------------------------------
def test_overwritten_record_returns_exactly_once():
    index, vectors = build_raw()
    replacement = unit(np.random.default_rng(11).standard_normal((1, DIM)))[0]
    index.add({"id": "doc_5", "values": replacement.tolist()}, overwrite=True)

    # Asking for the whole index makes the count exact rather than probable.
    results = index.search(replacement.tolist(), top_k=N)
    assert ids_of(results).count("doc_5") == 1

    # And it is found where its new vector is, not where its old one was.
    assert "doc_5" in ids_of(index.search(replacement.tolist(), top_k=TOP_K))


def test_overwrite_does_not_shorten_results():
    index, vectors = build_raw()
    rng = np.random.default_rng(12)
    for i in range(0, N, 2):
        new_vector = unit(rng.standard_normal((1, DIM)))[0]
        index.add({"id": f"doc_{i}", "values": new_vector.tolist()}, overwrite=True)

    assert index.get_vector_count() == N
    for probe in range(20):
        assert len(index.search(vectors[probe].tolist(), top_k=TOP_K)) == TOP_K


def test_repeatedly_overwritten_record_still_returns():
    """Relay 29 isolated this case. Twelve tight overwrites stack twelve stranded
    copies on one location, and a query at that location returned nothing at all."""
    index, vectors = build_raw()
    rng = np.random.default_rng(13)
    original = vectors[7].copy()
    current = original.copy()

    for _ in range(12):
        drift = rng.standard_normal(DIM).astype(np.float32)
        current = unit((current + 0.05 * drift / np.linalg.norm(drift))[None, :])[0]
        index.add({"id": "doc_7", "values": current.tolist()}, overwrite=True)

    # A query at the original location is the one that used to return zero.
    for query in (original, current):
        results = index.search(query.tolist(), top_k=TOP_K)
        assert len(results) == TOP_K
        assert ids_of(results).count("doc_7") <= 1

    assert "doc_7" in ids_of(index.search(current.tolist(), top_k=TOP_K))


# ------------------------------------------------------------
# compact()
# ------------------------------------------------------------
def test_compact_reclaims_stranded_nodes():
    index, vectors = build_raw()
    rng = np.random.default_rng(14)

    for i in range(0, 100):
        index.remove_point(f"doc_{i}")
    for i in range(100, 200):
        new_vector = unit(rng.standard_normal((1, DIM)))[0]
        index.add({"id": f"doc_{i}", "values": new_vector.tolist()}, overwrite=True)

    live = index.get_vector_count()
    assert live == N - 100
    stranded = graph_nodes(index) - live
    assert stranded == 200, "100 removals and 100 overwrites strand one node each"
    assert int(index.get_stats()["stranded_graph_nodes"]) == stranded

    assert index.compact() == stranded
    assert graph_nodes(index) == live
    assert int(index.get_stats()["stranded_graph_nodes"]) == 0


def test_compact_is_a_no_op_on_a_clean_index():
    index, vectors = build_raw(n=150, seed=4)
    assert graph_nodes(index) == index.get_vector_count()

    before = ids_of(index.search(vectors[0].tolist(), top_k=TOP_K))
    assert index.compact() == 0
    assert graph_nodes(index) == index.get_vector_count()
    assert ids_of(index.search(vectors[0].tolist(), top_k=TOP_K)) == before

    # Twice in a row is also a no-op, and the second call after a real compaction
    # is the case a scheduled job actually hits.
    assert index.compact() == 0


def test_compact_preserves_every_live_record():
    index, vectors = build_raw()
    rng = np.random.default_rng(15)
    removed = {f"doc_{i}" for i in range(0, N, 4)}
    for doc_id in removed:
        index.remove_point(doc_id)

    current = {}
    for i in range(1, N, 4):
        new_vector = unit(rng.standard_normal((1, DIM)))[0]
        index.add({"id": f"doc_{i}", "values": new_vector.tolist()}, overwrite=True)
        current[f"doc_{i}"] = new_vector
    for i in range(N):
        doc_id = f"doc_{i}"
        if doc_id not in removed and doc_id not in current:
            current[doc_id] = vectors[i]

    assert index.compact() > 0

    assert index.get_vector_count() == len(current)
    for doc_id, vector in current.items():
        assert index.contains(doc_id)
        # Every live record is still reachable by its own vector, which is the
        # property a rebuild could plausibly break.
        assert doc_id in ids_of(index.search(vector.tolist(), top_k=TOP_K))
    for doc_id in removed:
        assert not index.contains(doc_id)


def test_compact_leaves_search_quality_intact():
    index, vectors = build_raw()
    for i in range(0, N, 3):
        index.remove_point(f"doc_{i}")
    live = [i for i in range(N) if i % 3 != 0]
    live_matrix = vectors[live]

    index.compact()

    for probe in (1, 2, 4, 5, 7, 8):
        query = vectors[probe]
        truth = {f"doc_{live[j]}" for j in nearest(live_matrix, query, TOP_K)}
        got = set(ids_of(index.search(query.tolist(), top_k=TOP_K)))
        assert len(got) == TOP_K
        assert len(got & truth) >= TOP_K - 1


# ------------------------------------------------------------
# The quantized path
#
# The predicate reaches the quantized search through the same call sites, and
# these tests hold it to the same invariants as the raw path. They stop short of
# the raw path's full-page assertions under churn, and the reason is a separate
# pre-existing limitation rather than anything this change did. On a clean
# quantized index of 10,000 records a query asking for 1,000 results gets 34,
# unchanged by `ef_search` at any value up to 4,000, so the ADC traversal reaches
# only a few dozen nodes whatever the search is told to do. That caps how much of
# a churned page the predicate can refill. Measured at 10,000 records, dimension
# 256 and 50 percent delete churn, results returned rose from 4.12 of 10 to 7.98
# of 10, and `compact()` then took it to 10.0 with a minimum of 10. So the tests
# below assert the invariants under churn and the full page after compaction.
# ------------------------------------------------------------
def test_quantized_clean_index_returns_full_top_k(quantized_fixture):
    index, vectors = quantized_fixture
    assert len(index.search(vectors[0].tolist(), top_k=TOP_K)) == TOP_K


def test_quantized_deletes_never_return_a_removed_record():
    index, vectors = build_quantized(seed=5)
    removed = {f"doc_{i}" for i in range(0, PQ_N, 2)}
    for doc_id in removed:
        index.remove_point(doc_id)

    for probe in range(1, 40, 2):
        results = index.search(vectors[probe].tolist(), top_k=TOP_K)
        assert 0 < len(results) <= TOP_K
        assert set(ids_of(results)).isdisjoint(removed)
        assert len(set(ids_of(results))) == len(results)


def test_quantized_compact_restores_full_pages():
    index, vectors = build_quantized(seed=5)
    for i in range(0, PQ_N, 2):
        index.remove_point(f"doc_{i}")

    assert index.compact() == PQ_N // 2
    for probe in range(1, 40, 2):
        assert len(index.search(vectors[probe].tolist(), top_k=TOP_K)) == TOP_K


def test_quantized_overwrite_returns_no_duplicates():
    index, vectors = build_quantized(seed=6)
    rng = np.random.default_rng(16)

    for i in range(0, 300):
        new_vector = unit(rng.standard_normal((1, PQ_DIM)))[0]
        index.add({"id": f"doc_{i}", "values": new_vector.tolist()}, overwrite=True)

    assert index.get_vector_count() == PQ_N
    for probe in range(400, 420):
        ids = ids_of(index.search(vectors[probe].tolist(), top_k=TOP_K))
        assert 0 < len(ids) <= TOP_K
        assert len(set(ids)) == len(ids)

    # The overwritten record itself is intact, which is the half of the claim the
    # quantized search is not reliable enough to answer.
    assert len(index.get_records("doc_5", return_vector=False)) == 1


def test_quantized_repeatedly_overwritten_record_does_not_duplicate():
    index, vectors = build_quantized(seed=7)
    rng = np.random.default_rng(17)
    original = vectors[7].copy()
    current = original.copy()

    for _ in range(12):
        drift = rng.standard_normal(PQ_DIM).astype(np.float32)
        current = unit((current + 0.05 * drift / np.linalg.norm(drift))[None, :])[0]
        index.add({"id": "doc_7", "values": current.tolist()}, overwrite=True)

    assert index.get_vector_count() == PQ_N
    for query in (original, current):
        ids = ids_of(index.search(query.tolist(), top_k=TOP_K))
        assert 0 < len(ids) <= TOP_K
        assert ids.count("doc_7") <= 1
    assert len(index.get_records("doc_7", return_vector=False)) == 1


def test_quantized_compact_reclaims_and_preserves():
    index, vectors = build_quantized(seed=8)
    removed = {f"doc_{i}" for i in range(0, 100)}
    for doc_id in removed:
        index.remove_point(doc_id)
    rng = np.random.default_rng(18)
    current = {}
    for i in range(100, 200):
        new_vector = unit(rng.standard_normal((1, PQ_DIM)))[0]
        index.add({"id": f"doc_{i}", "values": new_vector.tolist()}, overwrite=True)
        current[f"doc_{i}"] = new_vector

    live = index.get_vector_count()
    stranded = graph_nodes(index) - live
    assert stranded == 200

    # Quantization survives the rebuild. Relay 24 found the other rebuild path
    # cleared the codes it depends on.
    assert index.compact() == stranded
    assert index.is_quantized()
    assert graph_nodes(index) == live
    assert int(index.get_stats()["quantized_codes_stored"]) == live

    for probe in range(300, 315):
        assert len(index.search(vectors[probe].tolist(), top_k=TOP_K)) == TOP_K
    # `contains` reports on the raw vector store, and under this storage mode a
    # record written after training has codes but no raw vector, so the record
    # itself is checked through get_records.
    for doc_id in current:
        assert len(index.get_records(doc_id, return_vector=False)) == 1
    for doc_id in removed:
        assert index.get_records(doc_id, return_vector=False) == []


def test_quantized_compact_is_a_no_op_on_a_clean_index(quantized_fixture):
    index, _ = quantized_fixture
    assert graph_nodes(index) == index.get_vector_count()
    assert index.compact() == 0


# ------------------------------------------------------------
# Boundary arguments the filter made reachable
# ------------------------------------------------------------
def test_degenerate_search_widths_do_not_crash():
    """The filtered path in the vendored graph panicked at a resolved search width
    of zero and returned nothing at a width of one. Both are legal arguments."""
    index, vectors = build_raw(n=200, seed=9)
    for i in range(0, 200, 2):
        index.remove_point(f"doc_{i}")
    query = vectors[1].tolist()

    assert index.search(query, top_k=0, ef_search=0) == []
    assert len(index.search(query, top_k=1, ef_search=1)) == 1
    assert len(index.search(query, top_k=1, ef_search=0)) == 1
    assert len(index.search(query, top_k=5, ef_search=1)) == 5


def test_batch_search_paths_exclude_stranded_nodes():
    """The sequential batch path runs at five queries or fewer and the parallel
    path above that, so both thresholds are crossed here."""
    index, vectors = build_raw()
    for i in range(0, N, 2):
        index.remove_point(f"doc_{i}")
    removed = {f"doc_{i}" for i in range(0, N, 2)}

    for batch_size in (3, 20):
        queries = [vectors[i].tolist() for i in range(1, 2 * batch_size, 2)]
        batches = index.search(queries, top_k=TOP_K)
        assert len(batches) == batch_size
        for results in batches:
            assert len(results) == TOP_K
            assert set(ids_of(results)).isdisjoint(removed)
