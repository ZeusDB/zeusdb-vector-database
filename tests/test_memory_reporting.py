"""What `get_stats()` says an index holds, against what it actually holds.

`get_stats()` used to report the storage maps and the two quantization tables
and stop there. The graph is the largest thing an index holds and it was left
out, so on a trained `quantized_only` index at 50,000 records of dimension 1,536
the reported figure was 9.77 MB against 231 MiB resident, under five percent of
the truth. These tests cover the graph figure being reported at all, its shape
against the parameters that drive it, and the reported total against the
resident set of a process that holds one index and nothing else.

Memory is read as `psutil` `memory_info().rss`, which on Windows is the working
set. The figure is a delta across the build with the source array already
allocated on both sides, so what it measures is the index.
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
# The bound is measured rather than chosen. On the relay 60 grid, being three
# dimensions, two record counts and three storage modes over real
# dbpedia-openai embeddings, the reported total ran between 0.62 and 0.89 of
# the resident set the build added, and the rest is the id maps, the metadata
# map, the hash table slots and the allocator's own headers and fragmentation.
# The assertion below is deliberately looser than that band, because the
# allocator's share is a property of the machine.
TOTAL_AGAINST_RESIDENT_FLOOR = 0.45
TOTAL_AGAINST_RESIDENT_CEILING = 1.00


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
    """What the index says it holds is most of what the process gained.

    It cannot be all of it. `total_memory_mb` prices the structures this call
    can price and names what it leaves out, and no figure derived from the
    structures can account for the allocator's own headers or its
    fragmentation.

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

    # A process that gained nothing measurable cannot grade the report. The
    # working set is not a clean instrument and this leaves the assertion to
    # the runs where it is.
    if resident_mb < 8.0:
        pytest.skip(f"the build added only {resident_mb:.1f} MiB of working set")

    share = reported_mb / resident_mb
    assert TOTAL_AGAINST_RESIDENT_FLOOR <= share <= TOTAL_AGAINST_RESIDENT_CEILING, (
        f"at storage_mode {storage_mode} the index reports {reported_mb:.1f} MB "
        f"against {resident_mb:.1f} MiB resident, a share of {share:.3f}")


def test_the_total_is_the_sum_of_the_parts():
    """`total_memory_mb` adds up the five figures beside it and nothing else."""
    index, _ = _build(3000, 128, "quantized_with_raw")
    stats = index.get_stats()
    parts = sum(float(stats[k]) for k in (
        "graph_memory_mb", "raw_vectors_memory_mb", "quantized_codes_memory_mb",
        "codebook_memory_mb", "sdc_table_memory_mb"))
    assert float(stats["total_memory_mb"]) == pytest.approx(parts, abs=0.05)


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
