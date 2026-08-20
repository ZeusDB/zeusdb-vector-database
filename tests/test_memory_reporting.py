"""What `get_stats()` says an index holds, against what it actually holds.

`get_stats()` used to report the storage maps and the two quantization tables
and stop there. The graph is the largest thing an index holds and it was left
out, so on a trained `quantized_only` index at 50,000 records of dimension 1,536
the reported figure was 9.77 MB against 231 MiB resident, under five percent of
the truth. These tests cover the graph figure being reported at all, its shape
against the parameters that drive it, and the reported total against the bytes
the structure cannot avoid holding.

Nothing here reads the resident set. It was read until this relay, as a delta
across a build in a process of its own, and it turned out to be confounded by
more than it measured. What replaced it and what the two confounders came to
are recorded above `_structure_floor_mb`.
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


def test_the_dimension_moves_the_vector_store_and_not_the_graph():
    """Raising the dimension adds `records * dim * 4` bytes, once.

    There used to be two copies of every raw vector, one in a map keyed by
    external id and one in the graph, and `graph_memory_mb` carried the second
    while `raw_vectors_memory_mb` carried the first. There is one copy now and
    it is priced under `raw_vectors_memory_mb` alone, so an eightfold dimension
    moves that key by the vectors and leaves the graph figure where it was.

    What `graph_memory_mb` still holds is set by `m`, the record count and how
    hard the data prunes, and only the last of those moves with the dimension,
    so it drifts rather than staying fixed.
    """
    records = 3000
    small, _ = _build(records, 64, None, seed=7)
    large, _ = _build(records, 512, None, seed=7)

    expected = records * (512 - 64) * 4 / (1024 * 1024)
    vectors_grew = (float(large.get_stats()["raw_vectors_memory_mb"])
                    - float(small.get_stats()["raw_vectors_memory_mb"]))
    assert vectors_grew == pytest.approx(expected, rel=0.15), (
        f"the vector store grew by {vectors_grew:.3f} MB where the vectors "
        f"themselves are {expected:.3f} MB")

    graph_grew = (float(large.get_stats()["graph_memory_mb"])
                  - float(small.get_stats()["graph_memory_mb"]))
    assert abs(graph_grew) < 0.25 * expected, (
        f"the graph figure moved by {graph_grew:.3f} MB on a dimension change, "
        f"so it is still carrying a copy of the vectors")


def test_the_quantized_graph_scores_against_codes_rather_than_vectors():
    """Quantization replaces what the graph scores against with a code.

    `graph_memory_mb` is everything the graph holds apart from the raw vectors,
    which on a quantized graph includes the codes it is addressed against and on
    a raw one includes nothing of the sort, because there the raw vectors are
    reported under their own key. So the quantized figure is the raw one plus
    one byte per subvector per record rather than less than it, and the saving
    the mode buys shows up in the total.

    It does not make the graph negligible either way. The neighbour lists, the
    sixteen layer headers and the counters around them are there whatever the
    element type.
    """
    records, dim = 3000, 256
    raw, _ = _build(records, dim, None)
    with_raw, _ = _build(records, dim, "quantized_with_raw")
    only, _ = _build(records, dim, "quantized_only")

    raw_mb = float(raw.get_stats()["graph_memory_mb"])
    modes = [float(i.get_stats()["graph_memory_mb"]) for i in (with_raw, only)]
    assert modes[0] == pytest.approx(modes[1], rel=0.15), (
        "the two storage modes build the same graph, so they hold the same one")

    # A raw index reports no vector inside the graph figure, so the two differ
    # by the codes and the adjacency drift alone rather than by a vector copy.
    copy_mb = records * dim * 4 / (1024 * 1024)
    assert abs(modes[0] - raw_mb) < 0.5 * copy_mb, (
        f"the quantized graph is {modes[0]:.2f} MB against {raw_mb:.2f} raw, "
        f"and neither should carry a raw vector copy of {copy_mb:.2f} MB")

    # What the mode actually buys is in the total, where a quantized index no
    # longer holds a vector per record on the search path.
    raw_total = float(raw.get_stats()["total_memory_mb"])
    only_total = float(only.get_stats()["total_memory_mb"])
    assert only_total < raw_total, (
        f"quantized_only reports {only_total:.2f} MB against {raw_total:.2f} raw")

    # The graph is more than the codes it holds. What it carries beside them is
    # a fixed capacity neighbour slab per point at layer zero, which is
    # `(2m + 1)` targets and the same number of distances, so the figure has to
    # clear that whatever the element type.
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
# The total against what the structure must hold
# ------------------------------------------------------------
# This used to compare `total_memory_mb` against the resident set delta of a
# subprocess that built one index. That comparison is gone and the reason is
# that the resident delta is not a usable denominator here. It is confounded in
# both directions, each confounder is larger than the signal, and both were
# measured rather than argued.
#
# Downward, a build allocates and frees. On glibc a freed block is not a
# returned one, so the transient stays resident and inflates the delta. At
# 8,000 records of dimension 256 under `quantized_only`, the build peaks at
# 21.87 MB of private commit and settles at 13.69, of which 8.18 MB is
# decommitted and a further 6.65 MB is retained past what the index holds. The
# index reports 7.53 MB. CI read a delta of 20.6 MB and a share of 0.365
# against a floor of 0.45, and 20.6 is the peak of the build rather than the
# footprint of the index.
#
# `quantized_only` is the cell that shows it because it is the cell that frees
# most. Training rebuilds the graph, so the raw graph is dropped after the
# quantized one is built, and the mode then releases every raw vector. The same
# index loaded from a directory, which trains nothing, settles at 7.04 MB of
# commit against a report of 7.21, a share of 1.024. The report tracks the
# index. It is the measurement of the index that did not.
#
# Upward, capacity that is requested and never written never faults in. Declare
# 8,000 records as 100,000 and the report reads 138.93 MB against 25.97 MB of
# resident delta, a share of 5.35.
#
# So the report is compared against what the structure must hold, derived from
# the record count, the dimension, the degree and the code width. Every term is
# a fact about the layout rather than a measurement, so no allocator, kernel,
# runner or corpus can move it.
#
# Three terms, all of them written and none of them optional.
#
#   the raw vector store   `records * dim * 4`, where the mode keeps one
#   the graph's own copy   `records * dim * 4` raw, `records * subvectors`
#                          quantized, which the graph holds separately from
#                          the store
#   the layer zero slabs   `records * (2m + 1) * 8`, a fixed capacity
#                          neighbour list per point at layer zero, being that
#                          many targets and the same many distances
#
# What this catches is the defect the graph figure was added to fix. A trained
# `quantized_only` index of 50,000 records at dimension 1,536 reported 9.77 MB
# before the graph was priced, where these three terms name 17.2 MB, so it
# fails here.
#
# What it does not catch is an omission nobody enumerated. That is covered
# instead by `test_the_total_is_the_sum_of_the_parts`, which pins the total to
# its six components, and by the test each component has of its own.


def _structure_floor_mb(index, records, dim):
    """What the index must hold, from its own parameters and the layout.

    Derived and not measured. See the note above.
    """
    stats = index.get_stats()
    m = int(stats["m"])
    quantized = stats["storage_mode_description"] == "quantized_active"
    if quantized:
        ratio = float(stats["quantization_compression_ratio"].rstrip("x"))
        element_bytes = round(dim * 4 / ratio)
    else:
        element_bytes = dim * 4

    # One copy of every raw vector, not two. The index used to hold a map of
    # them beside the graph's own arena; relay 95 removed the map, so the floor
    # counts the store once. The graph's own element is a code on a quantized
    # index and is the store itself on a raw one, so it is only added where the
    # two are different things.
    store = records * dim * 4 if int(stats["raw_vectors_stored"]) else 0
    graph_copy = records * element_bytes if quantized else 0
    slabs = records * (2 * m + 1) * (4 + 4)
    return (store + graph_copy + slabs) / (1024 * 1024)


@pytest.mark.parametrize("storage_mode", [None, "quantized_with_raw", "quantized_only"])
def test_the_reported_total_covers_what_the_structure_holds(storage_mode):
    """The report clears the bytes the index cannot avoid holding.

    A report that misses most of what an index holds is the defect the graph
    figure was added to fix, and this is where that is caught. Measured on this
    fixture the report clears the floor by 1.23, 1.63 and 3.63 times across the
    three storage modes, and the margin is the structure the floor does not
    enumerate, being the upper layer lists, the counters, the codebook, the
    centroid distance table and the bookkeeping.
    """
    records, dim = 8000, 256
    index, _ = _build(records, dim, storage_mode)
    reported_mb = float(index.get_stats()["total_memory_mb"])
    floor_mb = _structure_floor_mb(index, records, dim)

    assert reported_mb >= floor_mb, (
        f"at storage_mode {storage_mode} the index reports {reported_mb:.2f} MB "
        f"where its stored records, the graph's own copy of them and its layer "
        f"zero slabs come to {floor_mb:.2f} MB, so the report is missing a "
        f"structure the index cannot be without")


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
