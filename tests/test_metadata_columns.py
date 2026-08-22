"""The columns a filtered search reads, and the proof they answer what the walk answers.

Every test here compares two indexes built over the identical corpus, one
declaring its filterable fields and one declaring none. The declared one answers
from its columns and the undeclared one walks every record's metadata, so any
disagreement between them is a defect in the columns rather than a change of
behaviour a test could be rewritten to accept.

A third answer is computed in Python from the corpus itself, so the pair is
checked against something neither of them produced.
"""

import json
import struct

import numpy as np
import pytest
from helpers import repair_manifest
from zeusdb_vector_database import VectorDatabase

DIM = 8
QUERY = np.zeros(DIM, dtype=np.float32)
QUERY[0] = 1.0

# Every field the corpus below carries, which is what the declared index
# declares. Two are strings of low cardinality, one is a distinct integer per
# record, one is an array, one is a float, one is a boolean and one is absent
# from a tenth of the records.
DECLARED = ["cat", "flag", "rank", "ratio", "tags", "name", "sometimes"]

CATS = ["alpha", "beta", "gamma", "delta", "epsilon"]
TAG_SETS = [["ai", "science"], ["tech"], [], ["ai"], ["science", "tech"]]


def corpus(size=600, seed=20260820):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((size, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"r{i:05d}" for i in range(size)]
    metadata = []
    for i in range(size):
        record = {
            "cat": CATS[i % 5],
            "flag": i % 3 == 0,
            "rank": i,
            "ratio": round(i / 7.0, 4),
            "tags": TAG_SETS[i % 5],
            "name": f"{CATS[i % 5].capitalize()}-{i:04d}.pdf",
        }
        # A field a tenth of the records do not carry, so every operator is
        # asked what it does about an absent field on real data.
        if i % 10 != 0:
            record["sometimes"] = "here" if i % 2 == 0 else "there"
        metadata.append(record)
    return ids, vectors, metadata


def build(indexed_fields, size=600, **create_kwargs):
    ids, vectors, metadata = corpus(size)
    index = VectorDatabase().create(
        "hnsw", dim=DIM, space="cosine", expected_size=size,
        indexed_fields=indexed_fields, **create_kwargs,
    )
    result = index.add({"ids": ids, "embeddings": vectors, "metadatas": metadata})
    assert result.is_success(), result.errors
    return index, vectors, metadata


@pytest.fixture(scope="module")
def pair():
    """The same corpus twice, once with columns and once without."""
    with_columns, vectors, metadata = build(DECLARED)
    without_columns, _, _ = build(None)
    return with_columns, without_columns, vectors, metadata


def page(index, filter, top_k=25):
    """(id, score bits) for one filtered search, in the order returned."""
    results = index.search(vector=QUERY, filter=filter, top_k=top_k, ef_search=200)
    return [(r["id"], struct.pack("<f", r["score"])) for r in results]


def ids_of(index, filter, top_k=1000):
    return sorted(r["id"] for r in index.search(
        vector=QUERY, filter=filter, top_k=top_k, ef_search=400))


# ---------------------------------------------------------------------------
# Every filter this relay could think of, evaluated three ways
# ---------------------------------------------------------------------------

# Each entry is (label, filter, predicate over one record's metadata dict).
# The predicate is the independent answer, written in Python against the corpus.
FILTERS = [
    ("eq string", {"cat": "beta"}, lambda m: m.get("cat") == "beta"),
    ("eq bool", {"flag": True}, lambda m: m.get("flag") is True),
    ("eq int", {"rank": 42}, lambda m: m.get("rank") == 42),
    ("eq float", {"ratio": {"eq": 1.0}}, lambda m: m.get("ratio") == 1.0),
    ("eq array", {"tags": ["ai", "science"]}, lambda m: m.get("tags") == ["ai", "science"]),
    ("eq empty array", {"tags": []}, lambda m: m.get("tags") == []),
    ("ne", {"cat": {"ne": "beta"}},
     lambda m: "cat" in m and m["cat"] != "beta"),
    ("ne on absent field", {"sometimes": {"ne": "here"}},
     lambda m: "sometimes" in m and m["sometimes"] != "here"),
    ("gt", {"rank": {"gt": 500}}, lambda m: m.get("rank", -1) > 500),
    ("gte", {"rank": {"gte": 500}}, lambda m: m.get("rank", -1) >= 500),
    ("lt", {"rank": {"lt": 25}}, lambda m: m.get("rank", 10**9) < 25),
    ("lte", {"rank": {"lte": 25}}, lambda m: m.get("rank", 10**9) <= 25),
    ("range", {"rank": {"gte": 100, "lt": 110}},
     lambda m: 100 <= m.get("rank", -1) < 110),
    ("gt on a float", {"ratio": {"gt": 80.0}}, lambda m: m.get("ratio", -1) > 80.0),
    ("contains on a string", {"name": {"contains": "Beta"}},
     lambda m: "Beta" in m.get("name", "")),
    ("contains on an array", {"tags": {"contains": "ai"}},
     lambda m: "ai" in m.get("tags", [])),
    ("startswith", {"name": {"startswith": "Gamma"}},
     lambda m: m.get("name", "").startswith("Gamma")),
    ("endswith", {"name": {"endswith": "7.pdf"}},
     lambda m: m.get("name", "").endswith("7.pdf")),
    ("in", {"cat": {"in": ["beta", "delta"]}},
     lambda m: m.get("cat") in ("beta", "delta")),
    ("in on a number", {"rank": {"in": [1, 2, 3]}}, lambda m: m.get("rank") in (1, 2, 3)),
    ("nin", {"cat": {"nin": ["beta", "delta"]}},
     lambda m: "cat" in m and m["cat"] not in ("beta", "delta")),
    ("any", {"tags": {"any": ["ai", "tech"]}},
     lambda m: bool(set(m.get("tags", [])) & {"ai", "tech"})),
    ("any on empty target", {"tags": {"any": []}}, lambda m: False),
    ("all", {"tags": {"all": ["ai", "science"]}},
     lambda m: {"ai", "science"} <= set(m.get("tags", []))),
    ("all on empty target", {"tags": {"all": []}}, lambda m: "tags" in m),
    ("two operators on one field", {"name": {"startswith": "Al", "endswith": ".pdf"}},
     lambda m: m.get("name", "").startswith("Al") and m.get("name", "").endswith(".pdf")),
    ("two fields", {"cat": "beta", "flag": True},
     lambda m: m.get("cat") == "beta" and m.get("flag") is True),
    ("matches nothing", {"cat": "nosuchvalue"}, lambda m: False),
    ("matches everything", {"rank": {"gte": 0}}, lambda m: True),
    ("absent field named directly", {"sometimes": "here"},
     lambda m: m.get("sometimes") == "here"),
    # Boolean composition, at four depths.
    ("or", {"$or": [{"cat": "beta"}, {"cat": "delta"}]},
     lambda m: m.get("cat") in ("beta", "delta")),
    ("or empty", {"$or": []}, lambda m: False),
    ("and explicit", {"$and": [{"cat": "beta"}, {"flag": True}]},
     lambda m: m.get("cat") == "beta" and m.get("flag") is True),
    ("and empty", {"$and": []}, lambda m: True),
    ("not", {"$not": {"cat": "beta"}}, lambda m: m.get("cat") != "beta"),
    ("not of an absent field", {"$not": {"sometimes": {"all": []}}},
     lambda m: "sometimes" not in m),
    ("and of or and not", {"$and": [
        {"$or": [{"cat": "beta"}, {"cat": "gamma"}]},
        {"$not": {"flag": True}},
    ]}, lambda m: m.get("cat") in ("beta", "gamma") and m.get("flag") is not True),
    ("or of and and not", {"$or": [
        {"$and": [{"cat": "alpha"}, {"rank": {"lt": 50}}]},
        {"$not": {"$or": [{"cat": "alpha"}, {"cat": "beta"}, {"cat": "gamma"},
                          {"cat": "delta"}]}},
    ]}, lambda m: (m.get("cat") == "alpha" and m.get("rank", 10**9) < 50)
        or m.get("cat") not in ("alpha", "beta", "gamma", "delta")),
    ("four deep", {"$not": {"$or": [{"$and": [{"$not": {"cat": "beta"}}]}]}},
     lambda m: m.get("cat") == "beta"),
    ("group beside a field", {"cat": "beta", "$not": {"flag": True}},
     lambda m: m.get("cat") == "beta" and m.get("flag") is not True),
]


@pytest.mark.parametrize("label,filter,predicate", FILTERS, ids=[f[0] for f in FILTERS])
def test_the_columns_and_the_walk_return_the_same_records(pair, label, filter, predicate):
    """The whole bar this relay is held to, one filter at a time."""
    with_columns, without_columns, _, metadata = pair
    ids, _, _ = corpus()

    expected = sorted(rid for rid, meta in zip(ids, metadata) if predicate(meta))

    from_columns = ids_of(with_columns, filter)
    from_walk = ids_of(without_columns, filter)

    assert from_walk == expected, f"{label}: the walk disagrees with Python"
    assert from_columns == expected, f"{label}: the columns disagree with Python"


@pytest.mark.parametrize("label,filter,predicate", FILTERS, ids=[f[0] for f in FILTERS])
def test_the_columns_and_the_walk_count_the_same(pair, label, filter, predicate):
    with_columns, without_columns, _, metadata = pair
    ids, _, _ = corpus()
    expected = sum(1 for meta in metadata if predicate(meta))
    assert without_columns.count(filter) == expected
    assert with_columns.count(filter) == expected


@pytest.mark.parametrize("label,filter,predicate", FILTERS, ids=[f[0] for f in FILTERS])
def test_the_filtered_page_is_identical_in_ids_and_score_bits(pair, label, filter, predicate):
    """Not merely the same records. The same page, ranked the same way, to the bit."""
    with_columns, without_columns, _, _ = pair
    assert page(with_columns, filter) == page(without_columns, filter)


def test_the_exact_scan_and_the_filtered_traversal_agree(pair):
    """Both paths through a filtered search read the same bitmap.

    A filter matching more than FULL_SCAN_THRESHOLD records is served by the
    graph traversal and one matching fewer by the exact scan. This corpus is
    smaller than the threshold, so the scan serves everything here; the two
    paths are compared against each other on a corpus large enough to cross it
    in `test_both_paths_agree_across_the_scan_threshold`.
    """
    with_columns, without_columns, vectors, metadata = pair
    ids, _, _ = corpus()
    # Every record matches, so the scan is the only path that could serve it at
    # this size, and the answer is the whole corpus ranked by distance.
    every = ids_of(with_columns, {"rank": {"gte": 0}})
    assert every == sorted(ids)
    assert every == ids_of(without_columns, {"rank": {"gte": 0}})


def test_both_paths_agree_across_the_scan_threshold():
    """A corpus larger than the threshold, so both paths actually run."""
    size = 12000
    with_columns, _, _ = build(["rank", "cat"], size=size)
    without_columns, _, _ = build(None, size=size)

    for filter in [
        {"rank": {"lt": 10}},          # scan
        {"rank": {"lt": 4999}},        # scan, just under
        {"rank": {"lt": 5001}},        # traversal, just over
        {"cat": "beta"},               # traversal, 2400 of 12000
        {"rank": {"gte": 0}},          # traversal, everything
    ]:
        assert page(with_columns, filter, top_k=10) == page(
            without_columns, filter, top_k=10), filter


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

def test_the_declaration_is_reported_back():
    index, _, _ = build(["cat", "rank"])
    assert index.indexed_fields == ["cat", "rank"]
    plain, _, _ = build(None)
    assert plain.indexed_fields == []


@pytest.mark.parametrize("declaration,fragment", [
    (["cat", "cat"], "twice"),
    ([""], "empty name"),
    (["$or"], "reserved filter key"),
    ([str(i) for i in range(33)], "limit is 32"),
    ([1, 2], "strings"),
    ("cat", "not a single string"),
    (17, "list of metadata field names"),
])
def test_a_bad_declaration_is_refused_with_a_message_naming_what_is_wrong(
        declaration, fragment):
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        VectorDatabase().create("hnsw", dim=4, indexed_fields=declaration)
    assert fragment in str(excinfo.value)


def test_a_filter_on_an_undeclared_field_returns_the_same_records():
    """It falls back to the walk. It does not raise and it does not change the answer."""
    declared, _, metadata = build(["cat"])
    plain, _, _ = build(None)
    ids, _, _ = corpus()

    filter = {"name": {"startswith": "Gamma"}}
    expected = sorted(rid for rid, meta in zip(ids, metadata)
                      if meta["name"].startswith("Gamma"))
    assert ids_of(declared, filter) == expected
    assert ids_of(plain, filter) == expected

    # And a filter mixing a declared field with an undeclared one, which is the
    # shape a partial declaration produces. The declared branch bounds which
    # records are read and the metadata decides among them.
    mixed = {"cat": "gamma", "name": {"endswith": "3.pdf"}}
    expected_mixed = sorted(
        rid for rid, meta in zip(ids, metadata)
        if meta["cat"] == "gamma" and meta["name"].endswith("3.pdf"))
    assert ids_of(declared, mixed) == expected_mixed
    assert ids_of(plain, mixed) == expected_mixed


# ---------------------------------------------------------------------------
# A filter mixing a declared field with one that has no column
# ---------------------------------------------------------------------------
#
# The declared fields cannot answer such a filter, and for some tree shapes they
# can still say which records could possibly match. Every test below compares
# the partly declared index against the undeclared one, which walks, so a bound
# that dropped a record shows up as a short page rather than as a slow one.

# Two fields of the seven, chosen so that every filter in FILTERS naming
# anything else becomes mixed.
PARTIAL = ["cat", "flag"]


@pytest.fixture(scope="module")
def partly_declared():
    """The same corpus again, declaring two of its seven fields."""
    index, _, _ = build(PARTIAL)
    return index


@pytest.mark.parametrize("label,filter,predicate", FILTERS, ids=[f[0] for f in FILTERS])
def test_a_partial_declaration_answers_what_the_walk_answers(
        pair, partly_declared, label, filter, predicate):
    """Every filter in the suite again, over an index declaring two fields.

    Most of them are now mixed, since they name a field with no column beside
    one that has. The page has to be identical to the walk's in ids, in order
    and in score bits.
    """
    _, without_columns, _, _ = pair
    assert page(partly_declared, filter) == page(without_columns, filter), label
    assert ids_of(partly_declared, filter) == ids_of(without_columns, filter), label
    assert partly_declared.count(filter) == without_columns.count(filter), label


# One filter per tree shape a mixed declaration can produce, with the shape
# named. `cat` and `flag` have columns; `rank`, `name` and `tags` do not.
MIXED_SHAPES = [
    ("conjunction, flat",
     {"cat": "beta", "rank": {"lt": 300}},
     lambda m: m["cat"] == "beta" and m["rank"] < 300),
    ("conjunction, explicit",
     {"$and": [{"cat": "beta"}, {"name": {"endswith": "3.pdf"}}]},
     lambda m: m["cat"] == "beta" and m["name"].endswith("3.pdf")),
    ("conjunction of three, two declared",
     {"cat": "beta", "flag": True, "rank": {"gte": 100}},
     lambda m: m["cat"] == "beta" and m["flag"] is True and m["rank"] >= 100),
    ("disjunction",
     {"$or": [{"cat": "beta"}, {"rank": {"lt": 20}}]},
     lambda m: m["cat"] == "beta" or m["rank"] < 20),
    ("negation of an undeclared leaf",
     {"$not": {"rank": {"lt": 300}}},
     lambda m: not m["rank"] < 300),
    ("negation of a mixed conjunction",
     {"$not": {"$and": [{"cat": "beta"}, {"rank": {"lt": 300}}]}},
     lambda m: not (m["cat"] == "beta" and m["rank"] < 300)),
    ("negation of a mixed disjunction",
     {"$not": {"$or": [{"cat": "beta"}, {"rank": {"lt": 300}}]}},
     lambda m: not (m["cat"] == "beta" or m["rank"] < 300)),
    ("conjunction carrying a negated undeclared branch",
     {"cat": "gamma", "$not": {"rank": {"lt": 300}}},
     lambda m: m["cat"] == "gamma" and not m["rank"] < 300),
    ("conjunction whose declared branch matches nothing",
     {"cat": "nosuchvalue", "rank": {"lt": 300}},
     lambda m: False),
    ("conjunction whose declared branch matches everything",
     {"$and": [{"cat": {"nin": []}}, {"rank": {"lt": 20}}]},
     lambda m: m["rank"] < 20),
    ("disjunction inside a conjunction",
     {"$and": [{"$or": [{"cat": "beta"}, {"cat": "delta"}]},
               {"name": {"endswith": "4.pdf"}}]},
     lambda m: m["cat"] in ("beta", "delta") and m["name"].endswith("4.pdf")),
    ("conjunction inside a disjunction",
     {"$or": [{"$and": [{"cat": "beta"}, {"flag": True}]},
              {"rank": {"lt": 5}}]},
     lambda m: (m["cat"] == "beta" and m["flag"] is True) or m["rank"] < 5),
    ("four deep, mixed at the bottom",
     {"$not": {"$or": [{"$and": [{"cat": "beta"}, {"$not": {"rank": {"lt": 300}}}]}]}},
     lambda m: not (m["cat"] == "beta" and not m["rank"] < 300)),
    ("array operator undeclared beside a declared field",
     {"cat": "alpha", "tags": {"any": ["ai", "tech"]}},
     lambda m: m["cat"] == "alpha" and bool(set(m["tags"]) & {"ai", "tech"})),
    ("undeclared field absent from a tenth of the records",
     {"flag": True, "sometimes": "here"},
     lambda m: m["flag"] is True and m.get("sometimes") == "here"),
]


@pytest.mark.parametrize("label,filter,predicate", MIXED_SHAPES,
                         ids=[s[0] for s in MIXED_SHAPES])
def test_every_mixed_tree_shape_returns_the_records_the_walk_returns(
        pair, partly_declared, label, filter, predicate):
    """The bar for the bound: the same records, in the same order, per shape.

    The Python predicate is the third answer, computed from the corpus itself,
    so a bound and a walk that agreed with each other and not with the data
    would still fail here.
    """
    _, without_columns, _, metadata = pair
    ids, _, _ = corpus()

    expected = sorted(rid for rid, meta in zip(ids, metadata) if predicate(meta))
    assert ids_of(without_columns, filter) == expected, f"{label}: the walk disagrees"
    assert ids_of(partly_declared, filter) == expected, f"{label}: the bound disagrees"
    assert page(partly_declared, filter) == page(without_columns, filter), label
    assert partly_declared.count(filter) == len(expected), label


def test_a_mixed_filter_agrees_across_the_scan_threshold():
    """A corpus larger than the threshold, so the bounded traversal runs too."""
    size = 12000
    partial, _, _ = build(["cat"], size=size)
    plain, _, _ = build(None, size=size)

    for filter in [
        {"cat": "beta", "rank": {"lt": 10}},        # bounded scan, few matches
        {"cat": "beta", "rank": {"lt": 4999}},      # bounded scan, just under
        {"cat": "beta", "rank": {"gte": 0}},        # 2,400 matches, bounded scan
        {"rank": {"lt": 8000}, "$not": {"cat": "nosuchvalue"}},  # 8,000, traversal
        {"$or": [{"cat": "beta"}, {"rank": {"lt": 8000}}]},      # no bound, traversal
    ]:
        assert page(partial, filter, top_k=10) == page(
            plain, filter, top_k=10), filter
        assert partial.count(filter) == plain.count(filter), filter


def test_remove_where_on_a_mixed_filter_removes_the_same_records():
    def mutate(index):
        assert index.remove_where({"cat": "beta", "rank": {"lt": 100}}) == 20

    declared, plain = _pair_after(mutate, declaration=PARTIAL)

    assert len(declared) == len(plain) == 180
    assert ids_of(declared, {"cat": "beta"}) == ids_of(plain, {"cat": "beta"})
    assert ids_of(declared, {"rank": {"lt": 100}}) == ids_of(plain, {"rank": {"lt": 100}})


# ---------------------------------------------------------------------------
# The columns stay in step with every mutation
# ---------------------------------------------------------------------------

def _pair_after(mutate, declaration=None):
    """Apply `mutate` to a declared index and an undeclared one, and return both."""
    declared, _, _ = build(declaration or DECLARED, size=200)
    plain, _, _ = build(None, size=200)
    mutate(declared)
    mutate(plain)
    return declared, plain


def test_update_metadata_moves_a_record_between_filters():
    def mutate(index):
        assert index.update_metadata("r00007", {"cat": "omega", "rank": 7})

    declared, plain = _pair_after(mutate)

    assert ids_of(declared, {"cat": "omega"}) == ["r00007"]
    assert ids_of(declared, {"cat": "omega"}) == ids_of(plain, {"cat": "omega"})
    # The value it used to hold no longer selects it.
    assert "r00007" not in ids_of(declared, {"cat": "gamma"})
    assert ids_of(declared, {"cat": "gamma"}) == ids_of(plain, {"cat": "gamma"})
    # A key the update left out is gone from the column as well as from the map.
    assert ids_of(declared, {"tags": {"contains": "ai"}}) == ids_of(
        plain, {"tags": {"contains": "ai"}})
    assert declared.get_records("r00007")[0]["metadata"] == {"cat": "omega", "rank": 7}


def test_remove_points_takes_the_records_out_of_every_filter():
    doomed = ["r00000", "r00001", "r00002", "r00005"]

    def mutate(index):
        assert index.remove_points(doomed) == []

    declared, plain = _pair_after(mutate)
    for filter in [{"cat": "alpha"}, {"rank": {"lt": 10}}, {"$not": {"cat": "alpha"}},
                   {"flag": True}]:
        assert ids_of(declared, filter) == ids_of(plain, filter), filter
        assert not set(doomed) & set(ids_of(declared, filter))
    assert declared.count({"rank": {"gte": 0}}) == 196


def test_remove_where_uses_the_columns_and_removes_the_same_records():
    def mutate(index):
        assert index.remove_where({"cat": "beta"}) == 40

    declared, plain = _pair_after(mutate)
    assert ids_of(declared, {"cat": "beta"}) == []
    assert len(declared) == 160
    for filter in [{"rank": {"gte": 0}}, {"$not": {"cat": "alpha"}}]:
        assert ids_of(declared, filter) == ids_of(plain, filter), filter


def test_an_overwrite_moves_the_record_to_a_new_slot():
    def mutate(index):
        result = index.add(
            {"id": "r00003", "values": [0.0] * DIM, "metadata": {"cat": "omega", "rank": 3}},
            overwrite=True,
        )
        assert result.is_success()

    declared, plain = _pair_after(mutate)
    assert ids_of(declared, {"cat": "omega"}) == ["r00003"]
    assert "r00003" not in ids_of(declared, {"cat": "delta"})
    assert len(declared) == 200
    for filter in [{"cat": "delta"}, {"rank": {"gte": 0}}, {"$not": {"cat": "omega"}}]:
        assert ids_of(declared, filter) == ids_of(plain, filter), filter


def test_clear_empties_the_columns_and_keeps_the_declaration():
    declared, _, _ = build(DECLARED, size=200)
    assert declared.clear() == 200
    assert declared.indexed_fields == DECLARED
    assert declared.count({"cat": "beta"}) == 0
    assert ids_of(declared, {"$not": {"cat": "beta"}}) == []

    # And the index fills again, with the columns tracking the new records.
    declared.add({"id": "fresh", "values": [1.0] + [0.0] * (DIM - 1),
                  "metadata": {"cat": "beta", "rank": 0}})
    assert ids_of(declared, {"cat": "beta"}) == ["fresh"]


def test_compact_leaves_the_columns_alone():
    """Compaction rebuilds the graph and keeps every internal id, so it must not
    disturb a store addressed by internal id."""
    declared, _, _ = build(DECLARED, size=200)
    plain, _, _ = build(None, size=200)
    for index in (declared, plain):
        assert index.remove_points(["r00000", "r00010", "r00020"]) == []

    before = ids_of(declared, {"cat": "alpha"})
    reclaimed = declared.compact()
    assert reclaimed == 3
    plain.compact()

    assert ids_of(declared, {"cat": "alpha"}) == before
    for filter in [{"cat": "alpha"}, {"rank": {"gte": 0}}, {"$not": {"flag": True}}]:
        assert ids_of(declared, filter) == ids_of(plain, filter), filter


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quantization_config", [
    None,
    {"type": "pq", "subvectors": 4, "bits": 8, "training_size": 1000,
     "storage_mode": "quantized_with_raw"},
    {"type": "pq", "subvectors": 4, "bits": 8, "training_size": 1000,
     "storage_mode": "quantized_only"},
])
def test_every_storage_mode_round_trips_with_identical_filtered_pages(
        tmp_path, quantization_config):
    # Above the 1,000 record training floor, so a quantized index actually
    # trains and the round trip covers a trained codebook rather than a
    # collecting one.
    index, _, _ = build(DECLARED, size=1500, quantization_config=quantization_config)

    filters = [{"cat": "beta"}, {"rank": {"lt": 20}}, {"$not": {"flag": True}},
               {"tags": {"any": ["ai"]}}, {"name": {"startswith": "Delta"}}]
    before = {json.dumps(f, sort_keys=True): page(index, f) for f in filters}

    directory = tmp_path / "columns.zdb"
    index.save(str(directory))
    loaded = VectorDatabase().load(str(directory))

    assert loaded.indexed_fields == DECLARED
    after = {json.dumps(f, sort_keys=True): page(loaded, f) for f in filters}
    assert after == before


def test_a_directory_saved_without_a_declaration_still_opens(tmp_path):
    """Which is every directory written before the columns existed."""
    index, _, metadata = build(None, size=300)
    directory = tmp_path / "nodeclaration.zdb"
    index.save(str(directory))

    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert config["indexed_fields"] == []
    # Strip the key outright, which is what a directory written by an older
    # release actually holds.
    del config["indexed_fields"]
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    repair_manifest(directory, "config.json")

    loaded = VectorDatabase().load(str(directory))
    assert loaded.indexed_fields == []
    ids, _, _ = corpus(300)
    expected = sorted(rid for rid, meta in zip(ids, metadata) if meta["cat"] == "beta")
    assert ids_of(loaded, {"cat": "beta"}) == expected


def test_a_saved_declaration_that_this_build_would_refuse_fails_the_load(tmp_path):
    index, _, _ = build(["cat"], size=50)
    directory = tmp_path / "baddeclaration.zdb"
    index.save(str(directory))

    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config["indexed_fields"] = ["cat", "cat"]
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    repair_manifest(directory, "config.json")

    with pytest.raises(ValueError, match="twice"):
        VectorDatabase().load(str(directory))


# ---------------------------------------------------------------------------
# What the columns cost
# ---------------------------------------------------------------------------

def test_the_columns_are_reported_in_the_memory_accounting():
    """A declared index reports more bookkeeping than an undeclared one, and the
    difference is the columns rather than anything else."""
    declared, _, _ = build(["cat", "flag"], size=2000)
    plain, _, _ = build(None, size=2000)
    declared_bytes = float(declared.get_stats()["index_bookkeeping_memory_mb"])
    plain_bytes = float(plain.get_stats()["index_bookkeeping_memory_mb"])
    assert declared_bytes > plain_bytes


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_a_declared_index_survives_a_concurrent_insert():
    """Every search path now holds a sixth read guard, so this is where a
    deadlock would show.

    The three filters below take the three paths a filtered search can take.
    ``rank gte 0`` admits every record and so runs above the scan threshold with
    a bitmap predicate, ``cat=beta`` runs below it and is answered from the
    columns outright, and ``name startswith`` names a field this index did not
    declare and so takes the walk and the warning.

    **The warning is what this test exists for.** It asks whether anything is
    declared, and the first version asked ``self.columns`` rather than the guard
    the caller already held, which is a second read on a thread holding one.
    ``std::sync::RwLock`` queues readers behind a waiting writer, so it blocks
    forever the moment the inserter below lands between the two.
    """
    import threading
    import time

    index, vectors, _ = build(["cat", "rank", "flag"], size=2000)
    errors = []
    stop = threading.Event()

    def search_forever(filter_):
        try:
            while not stop.is_set():
                for i in range(40):
                    index.search(vector=vectors[i], filter=filter_, top_k=5)
        except Exception as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    def insert_forever():
        try:
            i = 0
            while not stop.is_set() and i < 400:
                index.add({
                    "id": f"live{i:04d}", "values": vectors[i % 2000],
                    "metadata": {"cat": "omega", "rank": 900000 + i, "flag": False},
                })
                i += 1
        except Exception as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    workers = [
        threading.Thread(target=search_forever, args=({"rank": {"gte": 0}},)),
        threading.Thread(target=search_forever, args=({"cat": "beta"},)),
        threading.Thread(target=search_forever, args=({"name": {"startswith": "Beta"}},)),
        threading.Thread(target=search_forever, args=(None,)),
        threading.Thread(target=insert_forever),
        threading.Thread(target=insert_forever),
    ]
    for worker in workers:
        worker.start()
    time.sleep(2.0)
    stop.set()
    for worker in workers:
        worker.join(timeout=60)
        assert not worker.is_alive(), "a worker did not finish, which is a deadlock"
    assert not errors, errors

    # The inserted records all carry cat omega, so the selective page is unchanged.
    assert ids_of(index, {"cat": "beta"}) == sorted(
        f"r{i:05d}" for i in range(2000) if CATS[i % 5] == "beta")
