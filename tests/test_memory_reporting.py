"""What `get_stats()` says an index holds, against what it actually holds.

`get_stats()` used to report the storage maps and the two quantization tables
and stop there. The graph is the largest thing an index holds and it was left
out, so on a trained `quantized_only` index at 50,000 records of dimension 1,536
the reported figure was 9.77 MB against 231 MiB resident, under five percent of
the truth. These tests cover the graph figure being reported at all, its shape
against the parameters that drive it, and the reported total against the bytes
the structure cannot avoid holding.

Nothing here reads the resident set. Reading it as a delta across a build in a
process of its own is confounded by more than it measures. What replaces it, and
what the two confounders come to, are recorded above `_structure_floor_mb`.
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
    # `(2m + 1)` targets of four bytes each and no stored distance, so the
    # figure has to clear that whatever the element type.
    m = int(with_raw.get_stats()["m"])
    slab_mb = records * (2 * m + 1) * 4 / (1024 * 1024)
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
    # them beside the graph's own arena; the map is gone, so the floor counts
    # the store once. The graph's own element is a code on a quantized
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
    """`total_memory_mb` adds up the seven figures beside it and nothing else."""
    index, _ = _build(3000, 128, "quantized_with_raw")
    stats = index.get_stats()
    parts = sum(float(stats[k]) for k in (
        "graph_memory_mb", "raw_vectors_memory_mb", "quantized_codes_memory_mb",
        "codebook_memory_mb", "sdc_table_memory_mb", "centroid_norm_memory_mb",
        "index_bookkeeping_memory_mb"))
    assert float(stats["total_memory_mb"]) == pytest.approx(parts, abs=0.05)


def test_the_bookkeeping_is_reported_on_every_index():
    """The tables that find a record are priced, quantized or not."""
    for storage_mode in (None, "quantized_only", "quantized_with_raw"):
        index, _ = _build(2000, 128, storage_mode)
        book = float(index.get_stats()["index_bookkeeping_memory_mb"])
        assert book > 0.0, f"the bookkeeping is {book} at storage_mode {storage_mode}"


def test_a_record_pays_for_the_metadata_it_carries_and_no_table_for_it():
    """A record without metadata costs the bookkeeping sixteen bytes, and a
    record with two small fields costs one forty byte block per field beside
    its string text, not a hash table of its own.

    Two indexes over the same records, one with no metadata and one with a
    string field of twenty distinct values and an integer field, isolate the
    metadata. Everything else the bookkeeping counts is set by the record
    count and the configuration. Measured against a counting allocator at
    100,000 records the difference is 82.5 bytes a record; it was 259 when
    every record held a `HashMap` of its own, at a four bucket table of 56
    byte buckets for sixteen bytes of payload.
    """
    records, dim = 6000, 32
    data = _corpus(records, dim, 20260904)
    ids = [f"m_{i}" for i in range(records)]
    metas = [{"category": f"c{i % 20}", "year": 1990 + i % 30} for i in range(records)]

    def bookkeeping_bytes(metadatas):
        index = VectorDatabase().create("hnsw", dim=dim, expected_size=records)
        batch = {"ids": ids, "embeddings": data}
        if metadatas is not None:
            batch["metadatas"] = metadatas
        assert index.add(batch).is_success()
        return float(index.get_stats()["index_bookkeeping_memory_mb"]) * 1024 * 1024

    bare = bookkeeping_bytes(None)
    tagged = bookkeeping_bytes(metas)
    per_record = (tagged - bare) / records
    assert 70.0 <= per_record <= 100.0, (
        f"two small fields cost {per_record:.1f} bytes a record of bookkeeping, "
        f"which is not one block of two forty byte fields and a short string")


def test_metadata_json_names_every_record_and_holds_an_empty_object_for_a_bare_one(tmp_path):
    """The saved file is one object per live record under its id, and a record
    added without metadata is written as an empty object, which is what the
    file held before the store was indexed by internal id."""
    import json

    records, dim = 200, 8
    data = _corpus(records, dim, 20260905)
    ids = [f"m_{i}" for i in range(records)]
    metas = [{"category": f"c{i % 5}"} if i % 2 else {} for i in range(records)]
    index = VectorDatabase().create("hnsw", dim=dim, expected_size=records)
    assert index.add({"ids": ids, "embeddings": data, "metadatas": metas}).is_success()
    assert index.remove_point("m_3")
    index.save(str(tmp_path / "meta.zdb"))
    written = json.loads((tmp_path / "meta.zdb" / "metadata.json").read_text(encoding="utf-8"))
    expected = {ids[i]: metas[i] for i in range(records) if i != 3}
    assert written == expected

    loaded = VectorDatabase().load(str(tmp_path / "meta.zdb"))
    got = {r["id"]: r["metadata"] for r in loaded.get_records(list(expected))}
    assert got == expected
    assert loaded.count({"category": "c1"}) == sum(1 for m in expected.values() if m.get("category") == "c1")
    assert loaded.count({"$not": {"category": "c1"}}) == len(expected) - loaded.count({"category": "c1"})


def test_the_bookkeeping_tracks_the_records_and_not_the_dimension():
    """Two hash tables, two id copies and a metadata entry per record, and no
    vector in any of them.

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


# ------------------------------------------------------------
# One convention: every key prices the capacity it asked for
# ------------------------------------------------------------
def test_the_raw_vector_key_prices_the_block_and_not_the_payload():
    """`raw_vectors_memory_mb` is the store, slack included.

    It used to be `live records * dim * 4`, the payload, while every key beside
    it priced a capacity, so the sum was neither the request nor the resident
    set. The store is a growable block: it starts at the creation-time
    reservation and a build past that grows it geometrically, so a build that
    outgrows its declaration holds a store larger than its vectors.

    The payload is not lost by the change. It is `raw_vectors_stored` times
    `dimension` times four, and both keys are reported beside this one.
    """
    records, dim = 6000, 64
    # An honest declaration, where the reservation covers every record and the
    # block is exactly its payload.
    honest, _ = _build(records, dim, None)
    stats = honest.get_stats()
    payload_mb = (int(stats["raw_vectors_stored"]) * int(stats["dimension"]) * 4
                  / (1024 * 1024))
    assert float(stats["raw_vectors_memory_mb"]) == pytest.approx(payload_mb, abs=0.01)
    assert int(stats["raw_vectors_stored"]) == records

    # An understated one, where the block has doubled past the declaration and
    # carries what the last doubling left.
    data = _corpus(records, dim, 20260810)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        short = VectorDatabase().create("hnsw", dim=dim, expected_size=500)
    assert short.add({"ids": [f"s_{i}" for i in range(records)],
                      "embeddings": data}).is_success()
    stats = short.get_stats()
    held = float(stats["raw_vectors_memory_mb"])
    assert held > payload_mb, (
        f"the store reports {held:.3f} MB for {payload_mb:.3f} MB of vectors, so "
        "it is still being priced at its payload")
    assert float(stats["reserved_memory_mb"]) > 0.0


def test_the_reserved_figure_is_what_a_shrink_returns():
    """`reserved_memory_mb` names the bytes `shrink_to_fit()` hands back.

    Both are the graph's fifteen buffers and the vector stores priced at the gap
    between their capacity and their length, so the report says in advance what
    the call would release, and the index reports no slack once it has run.
    """
    records, dim = 6000, 64
    data = _corpus(records, dim, 20260810)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create("hnsw", dim=dim, expected_size=500)
    assert index.add({"ids": [f"r_{i}" for i in range(records)],
                      "embeddings": data}).is_success()

    reserved_mb = float(index.get_stats()["reserved_memory_mb"])
    assert reserved_mb > 0.0, "a build past its declaration holds no slack"

    released_mb = index.shrink_to_fit() / (1024 * 1024)
    assert released_mb == pytest.approx(reserved_mb, abs=0.02), (
        f"the report named {reserved_mb:.3f} MB of slack and the shrink returned "
        f"{released_mb:.3f} MB")

    after = index.get_stats()
    assert float(after["reserved_memory_mb"]) == pytest.approx(0.0, abs=0.02)
    # And the search still answers, because nothing but capacity moved.
    assert len(index.search(data[0], top_k=5)) == 5


def test_the_reserved_figure_is_inside_the_total_rather_than_beside_it():
    """The sum is unchanged by the key that says how much of it is untouched.

    `total_memory_mb` is the request. `reserved_memory_mb` is the part of that
    request no record has been written into, so it is bounded by the total and
    is not one of its terms.
    """
    for storage_mode in (None, "quantized_only", "quantized_with_raw"):
        index, _ = _build(3000, 64, storage_mode)
        stats = index.get_stats()
        parts = sum(float(stats[k]) for k in (
            "graph_memory_mb", "raw_vectors_memory_mb", "quantized_codes_memory_mb",
            "codebook_memory_mb", "sdc_table_memory_mb", "centroid_norm_memory_mb",
            "index_bookkeeping_memory_mb") if k in stats)
        total = float(stats["total_memory_mb"])
        assert total == pytest.approx(parts, abs=0.05)
        assert 0.0 <= float(stats["reserved_memory_mb"]) <= total


def test_the_training_buffer_is_released_once_the_codebook_is_fitted():
    """A trained index holds no capacity for the ids it collected to train on.

    The buffer was cleared and kept its slots, at a 24 byte `String` header
    each, for the life of the index. Collection stops for good once the
    threshold is reached, so nothing pushes into it again and the slots were
    dead weight inside `index_bookkeeping_memory_mb`.

    Two indexes over the same records differing only in `training_size` are what
    isolates it. Everything else the bookkeeping counts is set by the record
    count and by the configuration, and neither moves here, so a buffer that
    survived training would be the whole of the difference between them. A `Vec`
    grown by pushing holds the smallest power of two at or above its length, so
    at 1,000 and 4,000 the two buffers would be 1,024 and 4,096 slots, being
    0.070 MB apart. Measured before this was fixed the two read 1.63 and 1.70 MB.
    """
    records, dim = 6000, 32
    reports = []
    for training_size in (1000, 4000):
        index, _ = _build(records, dim, "quantized_only", training_size=training_size)
        assert index.is_quantized(), (
            f"training_size {training_size} did not train at {records} records")
        reports.append(float(index.get_stats()["index_bookkeeping_memory_mb"]))

    small, large = reports
    assert large == pytest.approx(small, abs=0.005), (
        f"a training_size of 4,000 reports {large:.3f} MB of bookkeeping against "
        f"{small:.3f} MB at 1,000, so the buffer survived the codebook")


def test_the_raw_store_a_training_rebuild_opens_is_sized_for_the_declaration():
    """A `quantized_with_raw` index holds its vectors in a block it asked for.

    The store that carries the raw vectors across the training transition used
    to be opened at the record count present at that moment, which is the
    training sample. Every record added afterwards pushed into a block sized for
    a fraction of the index, and the block doubled its way up, so an index that
    declared its size honestly still ended holding close to twice the vectors it
    had. It is reserved for the declared size now, so the block is the payload
    where the declaration covers the records.

    The unquantized arm is the control. Its store has always been reserved at
    creation and has always been exact here.
    """
    records, dim, training = 4000, 64, 1000
    payload_mb = records * dim * 4 / (1024 * 1024)

    raw, _ = _build(records, dim, None)
    assert float(raw.get_stats()["raw_vectors_memory_mb"]) == pytest.approx(
        payload_mb, abs=0.01)

    with_raw, _ = _build(records, dim, "quantized_with_raw", training_size=training)
    assert with_raw.is_quantized(), "training did not complete"
    stats = with_raw.get_stats()
    assert int(stats["raw_vectors_stored"]) == records
    held = float(stats["raw_vectors_memory_mb"])
    assert held == pytest.approx(payload_mb, abs=0.01), (
        f"the store holds {held:.3f} MB for {payload_mb:.3f} MB of vectors, so "
        "it was reserved from the training sample rather than the declaration")
