"""`rebuild(m=...)`, the one creation parameter nothing else can correct.

`m` is chosen from `expected_size` at `create()` and was fixed there, so an
index declared for a smaller corpus than it received ran at a degree meant for
that smaller one. Every test here holds the same bar: the graph is the only
thing that changes. Every record, its vector, its metadata, its external id, its
internal id, its column entry and every filter's answer survive the rebuild
unchanged, and what moves is recall.

Recall is measured against exact ground truth computed in numpy from the corpus
itself, so neither the index before the rebuild nor the index after it is the
authority on what the right answer was.
"""

import contextlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

DIM = 32
SIZE = 3000
SEED = 20260821

CATS = ["alpha", "beta", "gamma", "delta", "epsilon"]

FILTERS = [
    {"cat": "beta"},
    {"rank": {"lt": 500}},
    {"cat": "gamma", "rank": {"gte": 1000}},
    {"$or": [{"cat": "alpha"}, {"rank": {"lt": 50}}]},
    {"$not": {"cat": "beta"}},
    {"flag": True},
]


def corpus(size=SIZE, dim=DIM, seed=SEED):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((size, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"r{i:05d}" for i in range(size)]
    metadata = [
        {"cat": CATS[i % 5], "rank": i, "flag": i % 3 == 0, "name": f"doc-{i:04d}.pdf"}
        for i in range(size)
    ]
    return ids, vectors, metadata


@contextlib.contextmanager
def _nothing():
    yield


def build(m=None, expected_size=SIZE, quantization_config=None,
          indexed_fields=("cat", "rank", "flag"), size=SIZE):
    ids, vectors, metadata = corpus(size)
    kwargs = {}
    if m is not None:
        kwargs["m"] = m
    index = VectorDatabase().create(
        "hnsw", dim=DIM, space="cosine", expected_size=expected_size,
        indexed_fields=list(indexed_fields) if indexed_fields else None,
        quantization_config=quantization_config, **kwargs,
    )
    result = index.add({"ids": ids, "embeddings": vectors, "metadatas": metadata})
    assert result.is_success(), result.errors
    return index, ids, vectors, metadata


def snapshot(index, ids):
    """Every record as the index reports it, keyed by external id."""
    out = {}
    for record in index.get_records(ids, return_vector=True):
        vector = record.get("vector")
        out[record["id"]] = (
            None if vector is None else b"".join(struct.pack("<f", v) for v in vector),
            record["metadata"],
        )
    return out


def page(index, query, filter=None, top_k=10):
    kwargs = {} if filter is None else {"filter": filter}
    results = index.search(vector=query, top_k=top_k, ef_search=200, **kwargs)
    return [(r["id"], struct.pack("<f", r["score"])) for r in results]


def matched(index, filter):
    return sorted(r["id"] for r in index.search(
        vector=np.zeros(DIM, dtype=np.float32), filter=filter, top_k=SIZE, ef_search=400))


def exact_neighbours(vectors, queries, k=10):
    """Ground truth by brute force, which neither index produced."""
    out = []
    for query in queries:
        scores = vectors @ query
        out.append(set(np.argsort(-scores)[:k].tolist()))
    return out


def recall_at(index, vectors, queries, truth, k=10):
    hits = 0
    for query, want in zip(queries, truth):
        got = {int(r["id"][1:]) for r in index.search(
            vector=query, top_k=k, ef_search=100)}
        hits += len(got & want)
    return hits / (k * len(queries))


# ---------------------------------------------------------------------------
# What survives
# ---------------------------------------------------------------------------

def test_every_record_survives_with_its_vector_and_metadata():
    index, ids, _, _ = build(m=8)
    before = snapshot(index, ids)
    assert len(before) == SIZE

    assert index.rebuild(m=24) == SIZE

    after = snapshot(index, ids)
    assert set(after) == set(before)
    for record_id in before:
        assert after[record_id][0] == before[record_id][0], record_id
        assert after[record_id][1] == before[record_id][1], record_id
    assert len(index) == SIZE


def test_the_internal_ids_are_the_ones_the_records_already_held():
    """What makes the columns and both id maps correct without being rewritten."""
    index, ids, _, _ = build(m=8)
    before = index.list(number=SIZE)
    index.rebuild(m=24)
    assert index.list(number=SIZE) == before
    # `list` walks the metadata store, so this is the record set. The graph
    # having the same node count is the other half.
    assert index.get_stats()["stranded_graph_nodes"] == "0"


def test_every_filter_returns_the_same_records():
    index, _, _, _ = build(m=8)
    before = {json.dumps(f, sort_keys=True): matched(index, f) for f in FILTERS}
    counts = {json.dumps(f, sort_keys=True): index.count(f) for f in FILTERS}

    index.rebuild(m=24)

    for f in FILTERS:
        key = json.dumps(f, sort_keys=True)
        assert matched(index, f) == before[key], key
        assert index.count(f) == counts[key], key


def test_the_columns_are_correct_after_a_rebuild():
    """The declaration and the columns' answers both survive.

    Checked against an index that declared nothing and therefore walks, so a
    column left stale would show up as a disagreement rather than as a wrong
    answer both sides shared.
    """
    declared, _, _, _ = build(m=8)
    plain, _, _, _ = build(m=8, indexed_fields=None)

    declared.rebuild(m=24)

    assert declared.indexed_fields == ["cat", "rank", "flag"]
    for f in FILTERS:
        assert matched(declared, f) == matched(plain, f), f


# ---------------------------------------------------------------------------
# What changes
# ---------------------------------------------------------------------------

def test_the_declared_configuration_reads_back_changed():
    index, _, _, _ = build(m=8, expected_size=1000)
    assert index.m == 8
    assert index.expected_size == 1000

    index.rebuild(m=24, expected_size=50_000)

    assert index.m == 24
    assert index.expected_size == 50_000
    assert index.get_stats()["m"] == "24"
    assert index.get_stats()["expected_size"] == "50000"
    assert "m=24" in index.info()
    assert "expected_size=50000" in index.info()


def test_either_parameter_alone_leaves_the_other_where_it_was():
    index, _, _, _ = build(m=8, expected_size=1000)
    index.rebuild(m=24)
    assert (index.m, index.expected_size) == (24, 1000)
    index.rebuild(expected_size=9000)
    assert (index.m, index.expected_size) == (24, 9000)


def test_recall_rises_at_a_raised_degree_and_falls_at_a_lowered_one():
    """Measured against exact ground truth, which is the point of the operation.

    A raised `m` is what a caller who under-declared `expected_size` needs, and
    it is the direction the requirement runs in. A lowered `m` costs recall,
    which is not a defect: it is the operation doing what it was asked.
    """
    ids, vectors, _ = corpus()
    rng = np.random.default_rng(SEED + 1)
    queries = rng.standard_normal((40, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    truth = exact_neighbours(vectors, queries)

    index, _, _, _ = build(m=4)
    at_four = recall_at(index, vectors, queries, truth)

    index.rebuild(m=32)
    at_thirty_two = recall_at(index, vectors, queries, truth)

    index.rebuild(m=4)
    back_at_four = recall_at(index, vectors, queries, truth)

    assert at_thirty_two > at_four, (at_four, at_thirty_two)
    assert at_thirty_two >= 0.95, at_thirty_two
    assert back_at_four < at_thirty_two, (back_at_four, at_thirty_two)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_a_save_after_a_rebuild_carries_the_new_degree(tmp_path):
    index, _, vectors, _ = build(m=8, expected_size=1000)
    index.rebuild(m=24, expected_size=50_000)

    path = str(tmp_path / "rebuilt.zdb")
    index.save(path)

    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    assert config["m"] == 24
    assert config["expected_size"] == 50_000

    loaded = VectorDatabase().load(path)
    assert loaded.m == 24
    assert loaded.expected_size == 50_000
    assert len(loaded) == SIZE

    query = vectors[7]
    assert page(loaded, query) == page(index, query)
    for f in FILTERS:
        assert matched(loaded, f) == matched(index, f), f


def test_two_saves_of_a_rebuilt_index_hold_identical_pages(tmp_path):
    """The round trip, page for page and score bit for score bit."""
    index, _, vectors, _ = build(m=8)
    index.rebuild(m=24)

    first = str(tmp_path / "once.zdb")
    index.save(first)
    loaded = VectorDatabase().load(first)
    second = str(tmp_path / "twice.zdb")
    loaded.save(second)

    again = VectorDatabase().load(second)
    for query in vectors[:5]:
        assert page(again, query) == page(loaded, query)


# ---------------------------------------------------------------------------
# Storage modes
# ---------------------------------------------------------------------------

QUANTIZATION = [
    None,
    {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000,
     "storage_mode": "quantized_only"},
    {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000,
     "storage_mode": "quantized_with_raw"},
]


@pytest.mark.parametrize("quantization_config", QUANTIZATION,
                         ids=["raw", "quantized_only", "quantized_with_raw"])
def test_every_storage_mode_rebuilds(quantization_config, tmp_path):
    """A quantized graph holds codes, and the rebuild preserves them.

    Nothing is re-encoded and the codebook is not retrained, so `get_records`
    returns what it returned before, bit for bit, in every mode. On
    `quantized_only` that is the reconstruction from the record's codes, which
    is what the index retains rather than what it was given.
    """
    with pytest.warns(UserWarning) if quantization_config else _nothing():
        index, ids, vectors, _ = build(m=8, quantization_config=quantization_config)

    mode_before = index.get_storage_mode()
    before = snapshot(index, ids)

    assert index.rebuild(m=24) == SIZE

    assert index.get_storage_mode() == mode_before
    assert index.m == 24
    after = snapshot(index, ids)
    assert set(after) == set(before)
    for record_id in before:
        assert after[record_id] == before[record_id], record_id

    # Still searchable, still filterable, and still saveable.
    assert len(page(index, vectors[3])) == 10
    for f in FILTERS:
        assert len(matched(index, f)) == index.count(f), f

    path = str(tmp_path / "mode.zdb")
    index.save(path)
    loaded = VectorDatabase().load(path)
    assert loaded.m == 24
    assert page(loaded, vectors[3]) == page(index, vectors[3])


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,fragment", [
    (1, "m must be at least 2"),
    (0, "m must be at least 2"),
    (257, "m must be less than or equal to 256"),
])
def test_an_invalid_m_raises_the_message_create_raises(m, fragment):
    """The same rules on the same values, so the same sentence comes back."""
    index, _, _, _ = build(m=8, size=50)

    with pytest.raises(ValueError) as rebuilt:
        index.rebuild(m=m)
    with pytest.raises(RuntimeError) as created:
        VectorDatabase().create("hnsw", dim=DIM, m=m)

    assert fragment in str(rebuilt.value)
    # `create()` wraps the same ValueError in a RuntimeError naming the factory,
    # so the rebuild's message is the tail of the creation's.
    assert str(created.value).endswith(str(rebuilt.value))
    # And the index is untouched.
    assert index.m == 8
    assert len(index) == 50


def test_an_invalid_expected_size_raises_the_same_way():
    index, _, _, _ = build(m=8, size=50)
    with pytest.raises(ValueError, match="expected_size must be positive"):
        index.rebuild(expected_size=0)
    with pytest.raises(ValueError, match="expected_size must be at most"):
        index.rebuild(expected_size=200_000_000)
    assert index.expected_size == SIZE


def test_rebuild_with_no_parameter_at_all_points_at_compact():
    index, _, _, _ = build(m=8, size=50)
    with pytest.raises(ValueError) as excinfo:
        index.rebuild()
    assert "compact()" in str(excinfo.value)


def test_the_two_warnings_are_the_same_sentence():
    """`rebuild` and `create` say the same thing about the same pair.

    The check lives in two places, the Python factory for `create()` and Rust
    for `rebuild()`, and two copies of a sentence are two chances to drift.
    """
    index, _, _, _ = build(m=8, size=50)

    with pytest.warns(UserWarning) as from_rebuild:
        index.rebuild(m=100)
    with pytest.warns(UserWarning) as from_create:
        VectorDatabase().create("hnsw", dim=DIM, m=100)

    assert str(from_rebuild[0].message) == str(from_create[0].message)
    assert "lower m to 99 or below" in str(from_rebuild[0].message)
    # The rebuild ran; the warning is about graph quality and not a refusal.
    assert index.m == 100


def test_no_warning_where_the_pair_clears_the_budget():
    import warnings

    index, _, _, _ = build(m=8, size=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        index.rebuild(m=99)
    assert index.m == 99


def test_the_warning_points_at_the_callers_line():
    index, _, _, _ = build(m=8, size=50)
    with pytest.warns(UserWarning) as caught:
        index.rebuild(m=100)
    assert caught[0].filename == __file__


# ---------------------------------------------------------------------------
# Interaction with the rest of the surface
# ---------------------------------------------------------------------------

def test_a_rebuild_reclaims_what_compact_would_have():
    """A rebuild rebuilds, so stranded nodes go with it."""
    index, ids, _, _ = build(m=8, size=200)
    assert index.remove_point(ids[0])
    assert int(index.get_stats()["stranded_graph_nodes"]) == 1

    assert index.rebuild(m=16) == 199
    assert index.get_stats()["stranded_graph_nodes"] == "0"
    assert len(index) == 199


def test_the_index_stays_writable_after_a_rebuild():
    index, _, _, _ = build(m=8, size=200)
    index.rebuild(m=16)
    result = index.add({"id": "extra", "values": [0.1] * DIM,
                        "metadata": {"cat": "alpha", "rank": 9999, "flag": False}})
    assert result.is_success(), result.errors
    assert len(index) == 201
    assert matched(index, {"rank": 9999}) == ["extra"]


def test_an_empty_index_rebuilds():
    index = VectorDatabase().create("hnsw", dim=DIM, m=8, expected_size=100)
    assert index.rebuild(m=16) == 0
    assert index.m == 16


def test_raising_expected_size_re_arms_the_overgrowth_warning(tmp_path):
    """The warning fires once against a declaration, and a new one is new.

    Without the re-arm a caller who took the advice, raised `expected_size` and
    then overgrew the new declaration would never hear about it again. The
    warning is a tracing line rather than a Python one, so this runs a
    subprocess with file logging on, exactly as
    `test_expected_size_overgrowth_warns_exactly_once` does.
    """
    log_file = tmp_path / "rearm.log"
    code = r"""
import numpy as np
import zeusdb_vector_database as zdb
idx = zdb.VectorDatabase().create('hnsw', dim=8, expected_size=10)
rng = np.random.default_rng(4242)
idx.add({'vectors': rng.random((40, 8)).astype('float32').tolist()})
idx.add({'vectors': rng.random((40, 8)).astype('float32').tolist()})
idx.rebuild(expected_size=90)
idx.add({'vectors': rng.random((120, 8)).astype('float32').tolist()})
assert idx.get_vector_count() == 200
assert idx.expected_size == 90
"""
    env = os.environ.copy()
    env.update({
        "ZEUSDB_LOG_LEVEL": "info",
        "ZEUSDB_LOG_FORMAT": "json",
        "ZEUSDB_LOG_TARGET": "file",
        "ZEUSDB_LOG_FILE": str(log_file),
    })
    subprocess.run([sys.executable, "-c", code], env=env, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    text = log_file.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if "expected_size_exceeded" in ln]
    # One before the rebuild and one after it. Without the re-arm there is one.
    assert len(lines) == 2, f"expected two warnings, got {len(lines)}"
    assert json.loads(lines[0])["fields"]["expected_size"] == 10
    assert json.loads(lines[1])["fields"]["expected_size"] == 90


# ---------------------------------------------------------------------------
# ef_construction, which the warning names as a remedy
# ---------------------------------------------------------------------------

def test_ef_construction_changes_and_survives_a_round_trip(tmp_path):
    index, _, vectors, _ = build(m=8, size=200)
    assert index.ef_construction == 200

    assert index.rebuild(ef_construction=64) == 200

    assert index.ef_construction == 64
    assert index.get_stats()["ef_construction"] == "64"
    assert "ef_construction=64" in index.info()

    path = str(tmp_path / "ef.zdb")
    index.save(path)
    assert json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))[
        "ef_construction"] == 64
    loaded = VectorDatabase().load(path)
    assert loaded.ef_construction == 64
    assert page(loaded, vectors[3]) == page(index, vectors[3])


def test_all_three_move_together_and_none_disturbs_the_others():
    index, _, _, _ = build(m=8, expected_size=1000, size=200)
    index.rebuild(m=24, expected_size=5000, ef_construction=300)
    assert (index.m, index.expected_size, index.ef_construction) == (24, 5000, 300)
    index.rebuild(ef_construction=250)
    assert (index.m, index.expected_size, index.ef_construction) == (24, 5000, 250)


def test_an_invalid_ef_construction_raises_the_message_create_raises():
    index, _, _, _ = build(m=8, size=50)
    with pytest.raises(ValueError) as rebuilt:
        index.rebuild(ef_construction=0)
    with pytest.raises(RuntimeError) as created:
        VectorDatabase().create("hnsw", dim=DIM, ef_construction=0)
    assert "ef_construction must be positive" in str(rebuilt.value)
    assert str(created.value).endswith(str(rebuilt.value))
    assert index.ef_construction == 200


def test_both_remedies_the_warning_names_are_reachable():
    """The warning offers two ways out and `rebuild` takes the pair.

    Raising `m` past half of `ef_construction` fires it. Lowering `m` again is
    one remedy and raising `ef_construction` is the other, and this asserts that
    both silence it, since a message naming an option the caller cannot take is
    worse than no message.
    """
    import warnings

    index, _, _, _ = build(m=8, size=50)
    with pytest.warns(UserWarning, match=r"2\*m=200"):
        index.rebuild(m=100)
    assert index.m == 100

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        index.rebuild(ef_construction=201)
    assert (index.m, index.ef_construction) == (100, 201)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        index.rebuild(m=99, ef_construction=200)
    assert (index.m, index.ef_construction) == (99, 200)


def test_an_ef_construction_above_the_ceiling_raises_the_message_create_raises():
    """The ceiling is checked by the validator create() uses, so the text is the same."""
    index, _, _, _ = build(m=8, size=50)
    with pytest.raises(ValueError) as rebuilt:
        index.rebuild(ef_construction=4097)
    with pytest.raises(RuntimeError) as created:
        VectorDatabase().create("hnsw", dim=DIM, ef_construction=4097)
    assert "ef_construction must be at most 4096, got 4097" in str(rebuilt.value)
    assert str(created.value).endswith(str(rebuilt.value))
    assert index.ef_construction == 200
