"""What `get_stats()` says an index holds, against what it actually holds.

`get_stats()` used to report the storage maps and the two quantization tables
and stop there. The graph is the largest thing an index holds and it was left
out, so on a trained `quantized_only` index at 50,000 records of dimension 1,536
the reported figure was 9.77 MB against 231 MiB resident, under five percent of
the truth. These tests cover the graph figure being reported at all, its shape
against the parameters that drive it, and the reported total against the
resident set of a process that holds one index and nothing else.

Memory is read as `psutil` `memory_info().rss`, which is the working set on
Windows and the pages faulted in and still held on Linux. The figure is a delta
across the build with the source array already allocated on both sides, so what
it measures is the index. It is not the same quantity as the reported total and
it does not bound it in either direction; see
`TOTAL_AGAINST_RESIDENT_CEILING`.
"""

import warnings

import numpy as np
import pytest

from zeusdb_vector_database import VectorDatabase


def _corpus(records, dim, seed):
    """Clustered unit vectors, so the graph prunes the way real data makes it."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((40, dim))
    points = centres[rng.integers(0, 40, records)] + rng.standard_normal((records, dim))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def _build(records, dim, storage_mode=None, seed=20260810, training_size=1000):
    data = _corpus(records, dim, seed)
    ids = [f"m_{i}" for i in range(records)]
    config = None
    if storage_mode is not None:
        config = {"type": "pq", "training_size": training_size,
                  "storage_mode": storage_mode}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create("hnsw", dim=dim, expected_size=records,
                                        quantization_config=config)
    assert index.add({"ids": ids, "embeddings": data}).is_success()
    return index, data


# ------------------------------------------------------------
# The graph is reported
# ------------------------------------------------------------
def test_graph_memory_is_reported_on_every_index():
    """Every index reports a graph figure and a total, quantized or not.

    `raw_vectors_memory_mb` used to appear only on a quantized index, so an
    unquantized one reported no memory at all.
    """
    for storage_mode in (None, "quantized_only", "quantized_with_raw"):
        index, _ = _build(2000, 128, storage_mode)
        stats = index.get_stats()
        for key in ("graph_memory_mb", "raw_vectors_memory_mb", "total_memory_mb"):
            assert key in stats, f"{key} missing at storage_mode {storage_mode}"
        graph = float(stats["graph_memory_mb"])
        assert graph > 0.0, f"graph memory is {graph} at storage_mode {storage_mode}"
        assert float(stats["total_memory_mb"]) >= graph


def test_graph_memory_is_empty_before_any_record_arrives():
    """A graph holding no point holds no memory."""
    index = VectorDatabase().create("hnsw", dim=32, expected_size=100)
    assert float(index.get_stats()["graph_memory_mb"]) == 0.0


def test_graph_memory_carries_a_copy_of_every_point():
    """Raising the dimension adds `records * dim * 4` bytes to the graph.

    The graph owns a second copy of every point, separate from the storage map.
    Everything else it holds is set by `m`, the record count and how hard the
    data prunes, and only the last of those moves with the dimension, so the
    jump below is eightfold to put the copy well above that drift.
    """
    records = 3000
    small, _ = _build(records, 64, None, seed=7)
    large, _ = _build(records, 512, None, seed=7)

    grew = (float(large.get_stats()["graph_memory_mb"])
            - float(small.get_stats()["graph_memory_mb"]))
    expected = records * (512 - 64) * 4 / (1024 * 1024)
    assert grew == pytest.approx(expected, rel=0.15), (
        f"the graph grew by {grew:.3f} MB where the second copy of every point "
        f"is {expected:.3f} MB")


def test_the_quantized_graph_is_smaller_than_the_raw_one():
    """Quantization replaces the graph's copy with a code, in both storage modes.

    It does not make the graph negligible. The neighbour lists, the sixteen
    layer headers and the counters around them are there either way, and the
    adjacency itself moves because the codes prune differently from the vectors,
    so the saving is bounded rather than pinned.
    """
    records, dim = 3000, 256
    raw, _ = _build(records, dim, None)
    with_raw, _ = _build(records, dim, "quantized_with_raw")
    only, _ = _build(records, dim, "quantized_only")

    raw_mb = float(raw.get_stats()["graph_memory_mb"])
    modes = [float(i.get_stats()["graph_memory_mb"]) for i in (with_raw, only)]
    assert all(m < raw_mb for m in modes), (
        f"the quantized graph is {modes} MB against {raw_mb:.2f} unquantized")
    assert modes[0] == pytest.approx(modes[1], rel=0.15), (
        "the two storage modes build the same graph, so they hold the same one")

    # The saving is at least the copy it replaced and it can be more, because
    # the codes prune harder than the vectors at this compression so the
    # adjacency shrinks as well. It is not the whole graph.
    copy_mb = records * dim * 4 / (1024 * 1024)
    saved = raw_mb - modes[0]
    assert 0.5 * copy_mb <= saved <= 2.5 * copy_mb, (
        f"the quantized graph saved {saved:.2f} MB where the copy it replaced "
        f"is {copy_mb:.2f} MB")
    assert modes[0] > copy_mb, (
        "the quantized graph reads as smaller than the copy it dropped, so it "
        "is not carrying the neighbour lists")


def test_graph_memory_rises_with_the_graph_degree():
    """A denser graph holds more, because `m` sets the adjacency cap."""
    records, dim = 4000, 64
    data = _corpus(records, dim, 11)
    ids = [f"m_{i}" for i in range(records)]
    held = {}
    for m in (8, 32):
        index = VectorDatabase().create("hnsw", dim=dim, m=m, expected_size=records)
        assert index.add({"ids": ids, "embeddings": data}).is_success()
        held[m] = float(index.get_stats()["graph_memory_mb"])
    assert held[32] > held[8], f"graph memory did not rise with m: {held}"


# ------------------------------------------------------------
# The total against the resident set
# ------------------------------------------------------------
# `total_memory_mb` and the resident set are not the same quantity, and neither
# one bounds the other.
#
# The report counts bytes the index asked the allocator for. The resident set
# counts pages the process has touched and still holds. They diverge in both
# directions and the bounds below are asymmetric for that reason.
#
# Downward, the process holds more than the report names. The allocator's own
# block headers and its rounding, its fragmentation, and the id maps, the
# metadata map and the hash table slots that `total_memory_mb` does not price
# all sit outside the report. Measured on Windows over the relay 60 grid, being
# three dimensions, two record counts and three storage modes over real
# dbpedia-openai embeddings, the report ran 0.62 to 0.89 of the resident delta.
#
# Upward, the report names more than the process holds. Capacity that was
# requested and never written never faults in, so it never enters the resident
# set at all. The layer reservation `expected_size` makes at creation and the
# slack in the neighbour list buffers are the largest of those, and their share
# is fixed by the declared size rather than by the bytes the records occupy, so
# it dominates at the 8,000 records this test builds and washes out by 50,000.
# Measured on Linux at this size the report ran 1.196, 1.229 and 1.579 of the
# resident delta across the three storage modes.
#
# The ceiling therefore sits well above one. It is 2.00 rather than just above
# the 1.579 seen, so that a different allocator or a different kernel's fault-in
# behaviour does not fail a test that is about neither.
#
# The floor is unchanged. Under-reporting is the defect this test exists to
# catch, and nothing about page accounting makes a report that misses half of
# what an index holds acceptable.
TOTAL_AGAINST_RESIDENT_FLOOR = 0.45
TOTAL_AGAINST_RESIDENT_CEILING = 2.00


_RESIDENT_PROBE = '''
import gc, json, sys, warnings
import numpy as np, psutil
from zeusdb_vector_database import VectorDatabase

records, dim, storage_mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(20260811)
centres = rng.standard_normal((40, dim))
points = centres[rng.integers(0, 40, records)] + rng.standard_normal((records, dim))
data = (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)
del points, centres
ids = [f"m_{i}" for i in range(records)]

config = None
if storage_mode != "none":
    config = {"type": "pq", "training_size": 1000, "storage_mode": storage_mode}

gc.collect()
before = psutil.Process().memory_info().rss
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    index = VectorDatabase().create("hnsw", dim=dim, expected_size=records,
                                    quantization_config=config)
assert index.add({"ids": ids, "embeddings": data}).is_success()
gc.collect()
after = psutil.Process().memory_info().rss
print(json.dumps({"resident": after - before,
                  "reported": float(index.get_stats()["total_memory_mb"])}))
'''


@pytest.mark.parametrize("storage_mode", ["none", "quantized_with_raw", "quantized_only"])
def test_the_reported_total_tracks_the_resident_set(storage_mode):
    """What the index says it holds, against what the process gained.

    The report can land either side of the resident delta, and the bounds are
    asymmetric for reasons recorded on `TOTAL_AGAINST_RESIDENT_CEILING`. What
    this test is for is the floor. A report that misses most of what an index
    holds is the defect the graph figure was added to fix, and it would be
    caught here whatever the pages say.

    One index per process. In a shared process the second build reuses the
    pages the first one freed, so the resident delta stops measuring the index
    and the comparison stops meaning anything.
    """
    import json
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", _RESIDENT_PROBE, "8000", "256", storage_mode],
        capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    measured = json.loads(completed.stdout.strip().splitlines()[-1])

    resident_mb = measured["resident"] / (1024 * 1024)
    reported_mb = measured["reported"]

    # A process that gained nothing measurable cannot grade the report. Page
    # accounting is not a clean instrument and this leaves the assertion to the
    # runs where it is.
    if resident_mb < 8.0:
        pytest.skip(f"the build added only {resident_mb:.1f} MiB of resident set")

    share = reported_mb / resident_mb
    assert TOTAL_AGAINST_RESIDENT_FLOOR <= share <= TOTAL_AGAINST_RESIDENT_CEILING, (
        f"at storage_mode {storage_mode} the index reports {reported_mb:.1f} MB "
        f"against {resident_mb:.1f} MiB resident, a share of {share:.3f}")


def test_the_total_is_the_sum_of_the_parts():
    """`total_memory_mb` adds up the six figures beside it and nothing else."""
    index, _ = _build(3000, 128, "quantized_with_raw")
    stats = index.get_stats()
    parts = sum(float(stats[k]) for k in (
        "graph_memory_mb", "raw_vectors_memory_mb", "quantized_codes_memory_mb",
        "codebook_memory_mb", "sdc_table_memory_mb", "index_bookkeeping_memory_mb"))
    assert float(stats["total_memory_mb"]) == pytest.approx(parts, abs=0.05)


def test_the_bookkeeping_is_reported_on_every_index():
    """The tables that find a record are priced, quantized or not."""
    for storage_mode in (None, "quantized_only", "quantized_with_raw"):
        index, _ = _build(2000, 128, storage_mode)
        book = float(index.get_stats()["index_bookkeeping_memory_mb"])
        assert book > 0.0, f"the bookkeeping is {book} at storage_mode {storage_mode}"


def test_the_bookkeeping_tracks_the_records_and_not_the_dimension():
    """Five hash tables and two id copies per record, and no vector in any of them.

    An eightfold dimension moves the raw vector store eightfold and must leave
    this figure where it was. Doubling the records doubles it, within the step
    a power of two bucket array takes: 3,000 and 6,000 records both sit in the
    same half of their table, so the ratio lands near two rather than exactly on
    it.
    """
    small, _ = _build(3000, 64, None, seed=7)
    wide, _ = _build(3000, 512, None, seed=7)
    many, _ = _build(6000, 64, None, seed=7)

    def book(index):
        return float(index.get_stats()["index_bookkeeping_memory_mb"])

    assert book(wide) == pytest.approx(book(small), rel=0.01), (
        f"an eightfold dimension moved the bookkeeping from {book(small):.3f} "
        f"to {book(wide):.3f} MB")
    assert book(many) / book(small) == pytest.approx(2.0, rel=0.25), (
        f"doubling the records took the bookkeeping from {book(small):.3f} to "
        f"{book(many):.3f} MB")


def test_the_graph_figure_survives_a_save_and_a_load(tmp_path):
    """A restored graph reports what the graph it restored reported."""
    index, _ = _build(2000, 128, None)
    before = float(index.get_stats()["graph_memory_mb"])
    directory = str(tmp_path / "graph_memory.zdb")
    index.save(directory)

    loaded = VectorDatabase().load(directory)
    after = float(loaded.get_stats()["graph_memory_mb"])
    assert after == pytest.approx(before, rel=0.02), (
        f"the loaded graph reports {after:.2f} MB where the saved one reported "
        f"{before:.2f} MB")
