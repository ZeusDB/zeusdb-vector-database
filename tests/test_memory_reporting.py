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


def test_graph_memory_reports_the_reservation_before_any_record_arrives():
    """A graph holding no point reports what its reservation asked for.

    This used to read exactly zero, because the figure was derived from a
    structure that allocated per point and reported nothing until a point
    arrived. It hid a reservation that was real: `expected_size` sizes the
    arenas at creation and those bytes are committed whether or not a record
    ever lands in them, which is why `expected_size` is capped at all.

    The figure is now the reservation, so it is small rather than zero on a
    small declaration and it is bounded on a large one. What bounds it is a
    byte budget rather than the declaration; see
    `test_empty_index_at_a_large_declared_size_stays_under_the_bound`, which is
    where that bound is held.
    """
    index = VectorDatabase().create("hnsw", dim=32, expected_size=100)
    reported = float(index.get_stats()["graph_memory_mb"])
    assert 0.0 < reported < 1.0, (
        f"an empty index at expected_size 100 reports {reported} MB")

    # And it rises with the declaration, since that is what it is reserving
    # against.
    larger = VectorDatabase().create("hnsw", dim=32, expected_size=100_000)
    assert float(larger.get_stats()["graph_memory_mb"]) > reported


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
    # The graph is more than the codes it holds. What it carries beside them is
    # a fixed capacity neighbour slab per point at layer zero, which is
    # `(2m + 1)` targets and the same number of distances, so the figure has to
    # clear that whatever the element type. This used to be stated as clearing
    # the raw copy it dropped, which held while a point cost roughly 2,000 bytes
    # of structure around its vector. A point now costs about 400, so the
    # quantized graph is genuinely smaller than the copy it replaced and the
    # slab is the honest floor.
    m = int(with_raw.get_stats()["m"])
    slab_mb = records * (2 * m + 1) * (4 + 4) / (1024 * 1024)
    assert modes[0] > slab_mb, (
        f"the quantized graph reads as {modes[0]:.2f} MB where its layer zero "
        f"slabs alone are {slab_mb:.2f} MB, so it is not carrying the "
        f"neighbour lists")


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
# set at all. That mechanism is real and it is measurable: at 8,000 records of
# dimension 256 declared as 100,000, the report reads 138.93 MB against 25.97
# MiB of resident delta, a share of 5.35, and the creation itself adds 0.16 MiB
# of resident set for 128 MB of reservation.
#
# **It does not fire here, because the probe declares what it builds.** At
# `expected_size` equal to the record count the arenas are written by the
# records that arrive: creation adds 0.12 MiB of resident set, and the report
# then runs 0.935 of the delta. A declaration equal to the truth leaves nothing
# untouched to diverge over.
#
# The ceiling therefore sits above one for the allocator's sake rather than for
# the reservation's, and 2.00 is what it has been since relay 60. The Linux
# figures of 1.196, 1.229 and 1.579 that used to be recorded here were measured
# through a probe whose baseline was wrong; see `_RESIDENT_PROBE`. They are not
# evidence of anything and they are not carried forward.
#
# The floor is unchanged. Under-reporting is the defect this test exists to
# catch, and nothing about page accounting makes a report that misses half of
# what an index holds acceptable.
TOTAL_AGAINST_RESIDENT_FLOOR = 0.45
TOTAL_AGAINST_RESIDENT_CEILING = 2.00


# The corpus is generated at the width it is stored at and normalised in place,
# which is what makes the baseline below mean anything.
#
# It used to draw `standard_normal` at its default float64 and narrow at the
# end, so building 7.81 MiB of corpus allocated and freed 62.51 MiB of
# temporaries first, being the picked centres, the noise, their sum and the
# division's result. All of that was freed before the baseline was taken, and a
# freed block is not necessarily a returned one: glibc raises its mmap threshold
# as large mapped blocks are released, after which large allocations come from
# the heap and freed heap stays resident. The index was then built into a pool
# the process already held and the delta stopped measuring it.
#
# What that looked like: on CI the delta read 9.2 MiB for an unquantized index
# of 8,000 records at dimension 256, whose raw vector store is 7.81 MiB and
# whose graph holds a second copy of every vector at another 7.81 MiB. Every one
# of those 15.62 MiB is written, so every page holding them is faulted in. A
# delta below the bytes the index demonstrably wrote is a broken denominator
# rather than an over-reporting numerator.
#
# Generating in blocks at float32 takes the allocated-and-freed figure from
# 62.51 MiB to 0.65 MiB. The corpus keeps its shape, being 40 Gaussian centres
# with unit-normalised points drawn around them, and its values move, which
# nothing here depends on.
_RESIDENT_PROBE = '''
import gc, json, sys, warnings
import numpy as np, psutil
from zeusdb_vector_database import VectorDatabase

records, dim, storage_mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(20260811)
centres = rng.standard_normal((40, dim), dtype=np.float32)
pick = rng.integers(0, 40, records)
data = np.empty((records, dim), dtype=np.float32)
for at in range(0, records, 512):
    block = pick[at:at + 512]
    data[at:at + 512] = centres[block]
    data[at:at + 512] += rng.standard_normal((len(block), dim), dtype=np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)
del centres, pick
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
created = psutil.Process().memory_info().rss
assert index.add({"ids": ids, "embeddings": data}).is_success()
gc.collect()
after = psutil.Process().memory_info().rss
stats = index.get_stats()
print(json.dumps({"resident": after - before,
                  "resident_at_create": created - before,
                  "written_floor": (float(stats["raw_vectors_memory_mb"])
                                    + float(stats.get("quantized_codes_memory_mb", 0))),
                  "reported": float(stats["total_memory_mb"])}))
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

    # The denominator has to be sound before the ratio means anything. The
    # stored records are bytes the index wrote, so their pages are faulted in
    # and resident, and a delta below them says the process was already holding
    # pages the index then reused rather than saying the report is too large.
    # That is a broken instrument and it is reported as one, because failing on
    # the ratio would point at the wrong thing.
    floor_mb = measured["written_floor"]
    assert resident_mb >= floor_mb, (
        f"at storage_mode {storage_mode} the process gained {resident_mb:.1f} "
        f"MiB where the index wrote {floor_mb:.1f} MiB of stored records, so "
        f"the baseline is measuring a pool the process already held rather than "
        f"the index. Creation alone added "
        f"{measured['resident_at_create'] / (1024 * 1024):.2f} MiB.")

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
    # Close rather than equal, and the gap has a direction. A built graph
    # reserves against `expected_size` and against an estimate of how many upper
    # lists a graph that size will own, so it holds slack. A loaded one is sized
    # from the file and trimmed to fit, so it holds none. Measured at 3.6
    # percent on this fixture, with the loaded figure the smaller of the two.
    assert after <= before
    assert after == pytest.approx(before, rel=0.10), (
        f"the loaded graph reports {after:.2f} MB where the saved one reported "
        f"{before:.2f} MB")
