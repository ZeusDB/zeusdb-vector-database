"""Layer capacity reservation.

Regression tests for the creation-time reservation, which the graph takes
against expected_size and caps in bytes. The vendored crate this replaced
reserved every layer at the full declared size, which committed hundreds of
times the memory an index needed.

Memory is read as psutil memory_info().vms rather than rss. The reservation is
committed but never written, so on Windows rss reports almost nothing and on
Linux the untouched pages never enter the resident set. vms is private commit on
Windows and virtual size on Linux, and both track a reservation the moment it is
made.
"""

import gc

import numpy as np
import psutil
import pytest

from zeusdb_vector_database import VectorDatabase

MB = 1024 * 1024


def _vms_delta(build):
    """Return the process virtual memory a callable commits, in bytes."""
    proc = psutil.Process()
    gc.collect()
    before = proc.memory_info().vms
    held = build()
    gc.collect()
    after = proc.memory_info().vms
    assert held is not None
    return after - before


# ------------------------------------------------------------
# Test 1: An empty index at a large declared size stays small
# ------------------------------------------------------------
@pytest.mark.parametrize("m", [8, 16, 32, 64])
def test_empty_index_at_a_large_declared_size_stays_under_the_bound(m):
    """A declared size costs about one pointer per record and nothing more.

    Each point lives in exactly one layer, the layer of its own top level, so
    the per-layer reservations partition the declared size and sum to it. The
    total is therefore one Arc slot per declared record, 8 bytes on a 64 bit
    target, and it does not depend on m.

    Without patch 5 this reserves 136 * ln(m) * 8 bytes per declared record,
    which for the size used here is 14,455 MB at m 16 and 21,657 MB at m 64, so
    the bound below is missed by more than two orders of magnitude.
    """
    declared = 5_000_000
    vdb = VectorDatabase()
    delta = _vms_delta(
        lambda: vdb.create(
            index_type="hnsw",
            dim=8,
            m=m,
            ef_construction=200,
            expected_size=declared,
        )
    )

    assert delta < 256 * MB, (
        f"m={m} declared={declared} committed {delta / MB:.1f} MB, "
        "which is far above one pointer per declared record"
    )
    assert delta / declared < 64, "reservation is more than 8 slots per record"


# ------------------------------------------------------------
# Test 2: A declared size that used to abort the process
# ------------------------------------------------------------
def test_declared_size_that_previously_aborted_the_process():
    """expected_size 100,000,000 must not kill the interpreter.

    Before patch 5 this reserved roughly 302 GB across the 16 layers and the
    process died with 0xC0000409 on the failed allocation, with no Python
    exception raised and no chance for a caller to handle it. A Python level
    error is an acceptable outcome here, a process abort is not, and reaching
    the assertions at all proves the abort is gone.
    """
    declared = 100_000_000
    vdb = VectorDatabase()
    try:
        index = vdb.create(
            index_type="hnsw",
            dim=8,
            m=16,
            ef_construction=200,
            expected_size=declared,
        )
    except (MemoryError, ValueError, RuntimeError):
        # A host too small to commit 800 MB may still refuse. That is a
        # catchable Python error, which is the behaviour being asserted.
        return

    assert index.get_vector_count() == 0
    index.add([{"id": "a", "values": [1.0] * 8}])
    assert index.get_vector_count() == 1
    assert index.search(vector=[1.0] * 8, top_k=1)[0]["id"] == "a"


# ------------------------------------------------------------
# Test 2b: The declared size has an upper bound
# ------------------------------------------------------------
def test_declared_size_above_the_bound_raises_rather_than_aborting():
    """Above the bound the caller gets a ValueError, not a dead process.

    The reservation is one Arc slot per declared record and it is not fallible.
    Vec::with_capacity aborts on allocation failure rather than unwinding, so a
    declaration too large for the machine cannot be caught after the fact. A
    declared 20,000,000,000 asks for 155 GB in the layer zero reservation alone
    and the process exits with no traceback. The bound converts that into an
    error, and the message carries the size the declaration would have asked
    for.
    """
    vdb = VectorDatabase()
    bound = 100_000_000

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        vdb.create(index_type="hnsw", dim=8, expected_size=bound + 1)
    message = str(excinfo.value)
    assert "expected_size must be at most 100000000" in message
    assert "0.8 GB" in message

    with pytest.raises((ValueError, RuntimeError), match="expected_size must be positive"):
        vdb.create(index_type="hnsw", dim=8, expected_size=0)

    # The bound itself is accepted, which Test 2 above already exercises.


# ------------------------------------------------------------
# Test 3: An index that outgrows its declared size
# ------------------------------------------------------------
def test_index_exceeding_its_declared_size_still_works():
    """The reservation is a hint, so the layer Vec must simply grow.

    Patch 5 makes the reservation tight rather than generous, which only stays
    safe because a layer that receives more points than reserved reallocates.
    """
    declared = 10
    records = 500
    vdb = VectorDatabase()
    index = vdb.create(
        index_type="hnsw",
        dim=8,
        m=16,
        ef_construction=200,
        expected_size=declared,
    )

    rng = np.random.default_rng(11)
    vectors = rng.standard_normal((records, 8)).astype(np.float32)
    result = index.add(
        [{"id": str(i), "values": vectors[i].tolist()} for i in range(records)]
    )

    assert result.total_inserted == records
    assert index.get_vector_count() == records

    hits = index.search(vector=vectors[0].tolist(), top_k=10)
    assert len(hits) == 10
    assert hits[0]["id"] == "0"


# ------------------------------------------------------------
# Test 4: Recall at a fixed configuration
# ------------------------------------------------------------
def test_recall_at_a_fixed_configuration_is_unchanged():
    """A capacity hint cannot reach the graph, so recall must not move.

    m is 4 rather than the shipped default on purpose. At 16 this configuration
    returns 1.0000 and would absorb a real regression without moving, while at 4
    it sits at 0.8890 and any change to the graph shows up.
    """
    n, dim, nq, top_k = 3_000, 32, 100, 10

    rng = np.random.default_rng(7)
    points = rng.standard_normal((n + nq, dim)).astype(np.float32)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    base, queries = points[:n], points[n:]
    truth = np.argsort(-(queries @ base.T), axis=1)[:, :top_k]

    vdb = VectorDatabase()
    index = vdb.create(
        index_type="hnsw",
        dim=dim,
        space="cosine",
        m=4,
        ef_construction=100,
        expected_size=n,
    )
    index.add([{"id": str(i), "values": base[i].tolist()} for i in range(n)])

    hits = 0
    for q in range(nq):
        found = {int(r["id"]) for r in index.search(vector=queries[q].tolist(), top_k=top_k)}
        hits += len(found & {int(x) for x in truth[q]})
    recall = hits / (nq * top_k)

    assert recall == pytest.approx(0.8890, abs=0.005), (
        f"recall at 10 moved to {recall:.4f} from the pinned 0.8890"
    )
