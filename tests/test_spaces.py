"""The sparse space, the text layer and the query surface.

`create(sparse=...)`, the `sparse` and `text` keys of every `add` shape,
`query`, `explain`, `load(path, tokenizer=...)`, and the five legacy `add`
shapes held explicitly beside them. Every method under every shape it
accepts and every rule it refuses, and a round trip of every space shape
through a save and a load, the external tokenizer included.

A one arm dense `query` is held equal to `search` id for id and score bit
for bit, a one arm sparse page to the dot product by hand, a text page under
term frequency weighting to the weighting by hand, and a fused page to the
reciprocal rank fusion by hand of the two arms' own pages.
"""

import json
import math
import os
import shutil
import threading

import numpy as np
import pytest
from helpers import repair_manifest
from zeusdb_vector_database import VectorDatabase, _engine

DIM = 4
CREATE = _engine._create_hnsw_index


def vec(seed, dim=DIM):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32).tolist()


def dense_only(**kw):
    kw.setdefault("expected_size", 100)
    return VectorDatabase().create("hnsw", dim=DIM, space="l2", **kw)


def with_sparse(weighting="dot", **kw):
    kw.setdefault("expected_size", 100)
    return VectorDatabase().create(
        "hnsw", dim=DIM, space="l2", sparse={"name": "terms", "weighting": weighting}, **kw
    )


def with_text(weighting="bm25", tokenizer="simple", **kw):
    kw.setdefault("expected_size", 100)
    return VectorDatabase().create(
        "hnsw", dim=DIM, space="l2",
        sparse={"name": "text", "weighting": weighting, "tokenizer": tokenizer}, **kw
    )


def whitespace(text):
    """A tokenizer of the caller's own: whitespace, case kept."""
    return text.split()


def sparse_keys(index):
    return {k: v for k, v in index.get_stats().items()
            if k.startswith(("sparse_", "term_count", "dictionary_"))}


def ids(page):
    return [hit["id"] for hit in page]


# ============================================================================
# CREATE
# ============================================================================


def test_create_declares_a_sparse_space_and_reports_it():
    index = with_sparse()
    stats = sparse_keys(index)
    assert stats == {
        "sparse_space": "terms",
        "sparse_weighting": "dot",
        "sparse_records": "0",
        "sparse_postings": "0",
        "sparse_dead_postings": "0",
        "sparse_memory_mb": "0.00",
    }
    # The dense declaration is untouched.
    assert index.dim == DIM
    assert index.space == "l2"
    assert index.info().startswith("HNSWIndex(dim=4, space=l2")


def test_create_declares_a_text_layer_and_reports_it():
    index = with_text()
    stats = sparse_keys(index)
    assert stats["sparse_space"] == "text"
    assert stats["sparse_weighting"] == "bm25"
    assert stats["sparse_tokenizer"] == "simple"
    assert stats["term_count"] == "0"
    assert stats["dictionary_memory_mb"] == "0.00"
    external = with_text(tokenizer=whitespace)
    assert sparse_keys(external)["sparse_tokenizer"] == "external"


def test_a_dense_only_index_reports_nothing_sparse():
    index = dense_only()
    assert sparse_keys(index) == {}
    index.add({"id": "a", "vector": vec(1)})
    assert sparse_keys(index) == {}


def test_create_sparse_takes_its_defaults_by_name(tmp_path):
    """Every field left out takes the value config.json records for it."""
    index = VectorDatabase().create("hnsw", dim=DIM, space="l2", expected_size=50, sparse={})
    assert sparse_keys(index)["sparse_space"] == "sparse"
    assert sparse_keys(index)["sparse_weighting"] == "dot"
    index.add({"id": "a", "vector": vec(1), "sparse": {"dims": [1], "values": [1.0]}})
    path = tmp_path / "defaults.zdb"
    index.save(str(path))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["spaces"] == [{
        "name": "sparse",
        "kind": "sparse",
        "index": {"unlink": "lazy", "lazy_threshold_percent": 10,
                  "weighting": {"type": "dot"}},
    }]

    # The weighting as a bare string is its published defaults, and as a
    # mapping it sets them. None under a key is the key left out.
    for weighting, expected in [
        ("bm25", {"type": "bm25", "k1": 1.2, "b": 0.75}),
        ({"type": "bm25"}, {"type": "bm25", "k1": 1.2, "b": 0.75}),
        ({"type": "bm25", "k1": 1.5, "b": 0.6}, {"type": "bm25", "k1": 1.5, "b": 0.6}),
        ({"type": "bm25", "k1": 2}, {"type": "bm25", "k1": 2.0, "b": 0.75}),
        ({"type": "dot"}, {"type": "dot"}),
        (None, {"type": "dot"}),
    ]:
        index = VectorDatabase().create(
            "hnsw", dim=DIM, space="l2", expected_size=50,
            sparse={"name": "s", "weighting": weighting, "unlink": "eager",
                    "lazy_threshold_percent": 25, "tokenizer": None},
        )
        target = tmp_path / f"w-{abs(hash(str(weighting)))}.zdb"
        index.save(str(target))
        config = json.loads((target / "config.json").read_text(encoding="utf-8"))
        assert config["spaces"][0]["index"] == {
            "unlink": "eager", "lazy_threshold_percent": 25, "weighting": expected,
        }
        assert "tokenizer" not in config["spaces"][0]


def test_create_with_a_tokenizer_weights_by_term_frequency_by_default(tmp_path):
    """A text layer stores term counts, so a declaration naming a tokenizer
    and no weighting takes bm25 at its published parameters. Every other
    combination is read as written, and config.json names the weighting in
    full either way, so a saved space never takes the default."""
    bm25 = {"type": "bm25", "k1": 1.2, "b": 0.75}
    cases = [
        ({"tokenizer": "simple"}, "bm25", bm25),
        ({"tokenizer": whitespace}, "bm25", bm25),
        ({"tokenizer": "simple", "weighting": None}, "bm25", bm25),
        ({"name": "t", "tokenizer": "simple", "unlink": "eager"}, "bm25", bm25),
        ({"tokenizer": "simple", "weighting": "dot"}, "dot", {"type": "dot"}),
        ({"tokenizer": "simple", "weighting": {"type": "bm25", "k1": 1.5, "b": 0.6}}, "bm25",
         {"type": "bm25", "k1": 1.5, "b": 0.6}),
        ({}, "dot", {"type": "dot"}),
        ({"tokenizer": None}, "dot", {"type": "dot"}),
        ({"name": "terms"}, "dot", {"type": "dot"}),
    ]
    for i, (sparse, name, written) in enumerate(cases):
        index = VectorDatabase().create("hnsw", dim=DIM, space="l2", expected_size=50, sparse=sparse)
        assert sparse_keys(index)["sparse_weighting"] == name, sparse
        path = tmp_path / f"default-{i}.zdb"
        index.save(str(path))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        assert config["spaces"][0]["index"]["weighting"] == written, sparse
    # The default scores as the rule declared by name, and not as the dot
    # product of two counts.
    records = [{"id": "a", "vector": vec(1), "text": "dog"},
               {"id": "b", "vector": vec(2), "text": "dog dog dog dog dog dog dog dog"}]
    defaulted = VectorDatabase().create("hnsw", dim=DIM, space="l2", expected_size=50,
                                        sparse={"tokenizer": "simple"})
    defaulted.add(records)
    named = with_text(weighting="bm25")
    named.add(records)
    dot = with_text(weighting="dot")
    dot.add(records)
    page = defaulted.query(arms=[{"text": "dog"}], top_k=5)
    assert page == named.query(arms=[{"text": "dog"}], top_k=5)
    assert page != dot.query(arms=[{"text": "dog"}], top_k=5)
    assert [(h["id"], h["score"]) for h in dot.query(arms=[{"text": "dog"}], top_k=5)] == [
        ("b", 8.0), ("a", 1.0)]


@pytest.mark.parametrize("sparse,klass,fragment", [
    ({"nope": 1}, ValueError, "declares 'nope', which is not a field"),
    ({"name": ""}, ValueError, "A space name must not be empty"),
    ({"name": "default"}, ValueError, "Space 'default' is declared twice"),
    ({"name": 3}, TypeError, "sparse['name'] must be a str"),
    ({"weighting": "bm26"}, ValueError, "unknown variant `bm26`, expected `dot` or `bm25`"),
    ({"weighting": {"type": "bm25", "b": 1.5}}, ValueError,
     "Term weighting parameter b is 1.5, and it must be between zero and one"),
    ({"weighting": {"type": "bm25", "k1": -1}}, ValueError,
     "Term weighting parameter k1 is -1, and it must be finite and at least zero"),
    ({"weighting": {"type": "bm25", "k1": "x"}}, ValueError, "expected f32"),
    ({"weighting": {"k1": 1.0}}, ValueError, "missing field `type`"),
    ({"weighting": 7}, ValueError, "could not be read"),
    ({"unlink": "sometimes"}, ValueError,
     "unknown variant `sometimes`, expected one of `strand`, `lazy`, `eager`"),
    ({"lazy_threshold_percent": -1}, ValueError, "expected u32"),
    ({"lazy_threshold_percent": "ten"}, ValueError, "expected u32"),
    ({"tokenizer": "external"}, ValueError, "without supplying it"),
    ({"tokenizer": "banana"}, ValueError, "names a tokenizer this build does not have"),
    ({"tokenizer": 3}, TypeError, "must be 'simple' or a callable"),
])
def test_create_refuses_a_bad_sparse_declaration(sparse, klass, fragment):
    """Through the engine's own door the class is the binding's; through
    create() every backend refusal is wrapped in a RuntimeError, as every
    other backend validation has always been."""
    with pytest.raises(klass) as raised:
        CREATE(dim=DIM, space="l2", m=8, ef_construction=50, expected_size=50, sparse=sparse)
    assert fragment in str(raised.value)
    with pytest.raises(RuntimeError, match="Failed to create HNSW index") as wrapped:
        VectorDatabase().create("hnsw", dim=DIM, space="l2", expected_size=50, sparse=sparse)
    assert fragment in str(wrapped.value)


def test_create_refuses_a_sparse_declaration_that_is_not_a_mapping():
    with pytest.raises(TypeError):
        CREATE(dim=DIM, space="l2", m=8, ef_construction=50, expected_size=50, sparse="bm25")


def test_create_refuses_the_dense_rules_before_reading_the_sparse_declaration():
    """The message a caller reads is the one for the first rule broken, in
    the order the rules have always applied."""
    with pytest.raises(ValueError, match="dim must be positive"):
        CREATE(dim=0, space="l2", m=8, ef_construction=50, expected_size=50,
               sparse={"nope": 1})
    with pytest.raises(ValueError, match="cannot be quantized"):
        CREATE(dim=DIM, space="dot", m=8, ef_construction=50, expected_size=50,
               quantization_config={"type": "pq", "subvectors": 2, "bits": 4,
                                    "training_size": 1000},
               sparse={"nope": 1})


def test_a_sparse_space_sits_beside_a_quantized_dense_space():
    index = VectorDatabase().create(
        "hnsw", dim=8, space="cosine", expected_size=2000,
        quantization_config={"type": "pq", "subvectors": 2, "bits": 4, "training_size": 1000,
                             "storage_mode": "quantized_with_raw"},
        sparse={"name": "terms"},
    )
    assert index.has_quantization()
    assert sparse_keys(index)["sparse_space"] == "terms"


# ============================================================================
# ADD, THE SPARSE HALF ON EVERY SHAPE
# ============================================================================


def sparse_vector(*pairs):
    return {"dims": [d for d, _ in pairs], "values": [v for _, v in pairs]}


def test_add_single_object_with_a_sparse_vector():
    index = with_sparse()
    result = index.add({"id": "a", "vector": vec(1), "sparse": sparse_vector((1, 0.5), (7, 2.0)),
                        "metadata": {"k": 1}})
    assert result.is_success() and result.ids == ["a"]
    assert sparse_keys(index)["sparse_records"] == "1"
    assert sparse_keys(index)["sparse_postings"] == "2"
    page = index.query(arms=[{"sparse": sparse_vector((7, 1.0))}], top_k=5)
    assert ids(page) == ["a"]
    assert page[0]["score"] == 2.0
    assert page[0]["metadata"] == {"k": 1}


def test_add_single_object_with_a_text():
    index = with_text()
    result = index.add({"id": "a", "values": vec(1), "text": "The quick brown fox"})
    assert result.is_success()
    assert sparse_keys(index)["term_count"] == "4"
    assert sparse_keys(index)["sparse_postings"] == "4"
    assert ids(index.query(arms=[{"text": "FOX"}], top_k=5)) == ["a"]


def test_add_list_of_objects_with_text_and_neither_refuses_the_sparse_one():
    """A space with a text layer takes text alone, since its term ids are
    the dictionary's, so in a list carrying a text, a supplied vector and
    neither, the supplied vector is that record's error and the records
    around it are inserted."""
    index = with_text()
    result = index.add([
        {"id": "a", "vector": vec(1), "text": "alpha beta"},
        {"id": "b", "values": vec(2), "sparse": sparse_vector((0, 3.0))},
        {"id": "c", "vector": np.array(vec(3), dtype=np.float32)},
        {"id": "d", "vector": vec(4), "text": None, "sparse": None},
    ])
    assert result.total_inserted == 3 and result.ids == ["a", "c", "d"]
    assert result.errors == ["Vector b: ValueError: This collection's sparse space takes text alone"]
    assert "b" not in index
    stats = sparse_keys(index)
    assert stats["sparse_records"] == "1"
    assert stats["term_count"] == "2"
    # Term id 0 is "alpha", and no supplied vector sits on it.
    assert ids(index.query(arms=[{"text": "alpha"}], top_k=5)) == ["a"]


def test_add_batch_dict_with_parallel_sparse_and_texts():
    """The two parallel arrays beside ids, None where a record fills the
    dense space alone. On a space with a text layer an entry under sparse
    is that record's error, and the records around it are inserted."""
    index = with_text()
    result = index.add({
        "ids": ["a", "b", "c", "d"],
        "vectors": [vec(1), vec(2), vec(3), vec(4)],
        "metadatas": [{"i": 1}, {"i": 2}, {"i": 3}, {"i": 4}],
        "sparse": [sparse_vector((0, 1.0)), None, None, None],
        "texts": [None, "beta gamma", None, ""],
    })
    assert result.total_inserted == 3 and result.ids == ["b", "c", "d"]
    assert result.errors == ["Vector a: ValueError: This collection's sparse space takes text alone"]
    stats = sparse_keys(index)
    # b by text, d by an empty text, c not at all, a refused.
    assert stats["sparse_records"] == "2"
    assert stats["term_count"] == "2"
    assert ids(index.query(arms=[{"text": "gamma"}], top_k=5)) == ["b"]
    assert index.get_records("d", return_vector=False)[0]["metadata"] == {"i": 4}


def test_add_batch_dict_with_a_numpy_array_and_the_sparse_arrays():
    """The NumPy batch shape with both arrays beside it. The entry under
    sparse, its dims and values arrays themselves, is read before the
    space's rule refuses it, so its error is the rule's and not a parse
    error, and the texts around it are counted."""
    index = with_text()
    vectors = np.array([vec(1), vec(2), vec(3)], dtype=np.float32)
    result = index.add({
        "ids": ["a", "b", "c"],
        "embeddings": vectors,
        "sparse": [None, {"dims": np.array([2, 5], dtype=np.int64),
                         "values": np.array([1.0, 2.0], dtype=np.float32)}, None],
        "texts": ["one two", None, "two three"],
    })
    assert result.total_inserted == 2 and result.ids == ["a", "c"]
    assert result.errors == ["Vector b: ValueError: This collection's sparse space takes text alone"]
    assert sparse_keys(index)["sparse_records"] == "2"
    assert set(ids(index.query(arms=[{"text": "two"}], top_k=5))) == {"a", "c"}
    with pytest.raises(ValueError, match="This collection's sparse space takes text alone"):
        index.query(arms=[{"sparse": sparse_vector((5, 1.0))}], top_k=5)


def test_add_batch_dict_under_the_values_spelling_takes_the_sparse_arrays():
    index = with_sparse()
    result = index.add({"ids": ["a", "b"], "values": [vec(1), vec(2)],
                        "sparse": [sparse_vector((3, 1.0)), sparse_vector((3, 2.0))]})
    assert result.is_success()
    page = index.query(arms=[{"sparse": sparse_vector((3, 1.0))}], top_k=5)
    assert ids(page) == ["b", "a"]


def test_add_bare_shapes_fill_the_dense_space_alone():
    # A bare vector is a tuple or a one dimensional array, since a bare list
    # has always been read as a list of records; a bare two dimensional
    # array is a batch; a list of vectors is a list of records.
    index = with_sparse()
    assert index.add(tuple(vec(1))).is_success()
    assert index.add(np.array(vec(2), dtype=np.float32)).is_success()
    assert index.add(np.array([vec(3), vec(4)], dtype=np.float32)).is_success()
    assert index.add([vec(5)]).is_success()
    assert len(index) == 5
    assert sparse_keys(index)["sparse_records"] == "0"


def test_a_generated_id_carries_its_sparse_half():
    index = with_sparse()
    result = index.add({"vectors": [vec(1), vec(2)],
                        "sparse": [None, sparse_vector((9, 4.0))]})
    assert result.ids == ["vec_1", "vec_2"]
    assert ids(index.query(arms=[{"sparse": sparse_vector((9, 1.0))}], top_k=5)) == ["vec_2"]


def test_an_overwrite_replaces_the_sparse_half():
    index = with_sparse()
    index.add({"id": "a", "vector": vec(1), "sparse": sparse_vector((1, 1.0))})
    index.add({"id": "a", "vector": vec(1), "sparse": sparse_vector((2, 1.0))})
    assert ids(index.query(arms=[{"sparse": sparse_vector((1, 1.0))}], top_k=5)) == []
    assert ids(index.query(arms=[{"sparse": sparse_vector((2, 1.0))}], top_k=5)) == ["a"]
    # Overwritten without a sparse half, the record leaves the space.
    index.add({"id": "a", "vector": vec(1)})
    assert ids(index.query(arms=[{"sparse": sparse_vector((2, 1.0))}], top_k=5)) == []
    assert len(index) == 1


def test_the_text_is_not_stored_unless_the_caller_stores_it():
    index = with_text()
    index.add({"id": "a", "vector": vec(1), "text": "kept nowhere"})
    index.add({"id": "b", "vector": vec(2), "text": "kept here", "metadata": {"text": "kept here"}})
    assert index.get_records("a", return_vector=False)[0]["metadata"] == {}
    assert index.get_records("b", return_vector=False)[0]["metadata"] == {"text": "kept here"}


# ---------------------------------------------------------------- refusals


def one_error(index, record):
    result = index.add(record)
    assert result.total_inserted == 0
    assert result.total_errors == 1
    return result.errors[0]


@pytest.mark.parametrize("sparse,fragment", [
    ("nope", "'sparse' must be a mapping {'dims': [...], 'values': [...]}, got str"),
    ({"values": [1.0]}, "'sparse' is missing 'dims'"),
    ({"dims": [1]}, "'sparse' is missing 'values'"),
    ({"dims": [1], "values": [1.0], "extra": 1},
     "'sparse' carries 'extra', and a sparse vector is {'dims': [...], 'values': [...]}"),
    ({"dims": [-1], "values": [1.0]}, "'sparse' dim -1 is outside 0 to 4294967295"),
    ({"dims": [1 << 40], "values": [1.0]}, "is outside 0 to 4294967295"),
    ({"dims": [1.5], "values": [1.0]}, "'sparse' dims must be a list of non-negative integers"),
    ({"dims": [1], "values": ["x"]}, "'sparse' values must be a list of numbers"),
    ({"dims": [1, 2], "values": [1.0]},
     "ValueError: Sparse vector has 2 dims and 1 values"),
    ({"dims": [2, 1], "values": [1.0, 1.0]},
     "ValueError: Sparse vector dims must be strictly increasing"),
    ({"dims": [1, 1], "values": [1.0, 1.0]},
     "ValueError: Sparse vector dims must be strictly increasing"),
    ({"dims": [1], "values": [math.nan]},
     "ValueError: Sparse vector contains invalid value at index 0"),
])
def test_add_refuses_a_malformed_sparse_vector_per_record(sparse, fragment):
    index = with_sparse()
    message = one_error(index, {"id": "bad", "vector": vec(1), "sparse": sparse})
    assert message.startswith("Vector bad: ")
    assert fragment in message
    assert "bad" not in index
    assert sparse_keys(index)["sparse_records"] == "0"


def test_add_refuses_a_non_positive_weight_under_term_frequency_weighting():
    index = with_sparse(weighting="bm25")
    message = one_error(index, {"id": "z", "vector": vec(1), "sparse": sparse_vector((1, 0.0))})
    assert message == ("Vector z: ValueError: Sparse vector value at index 0 is 0, and a "
                       "space weighted by term frequency takes values above zero alone")
    # Nothing was left behind in the dense space either.
    assert "z" not in index
    assert len(index) == 0
    # Under the dot product a zero and a negative weight are values.
    assert with_sparse().add({"id": "z", "vector": vec(1),
                              "sparse": sparse_vector((1, 0.0), (2, -1.0))}).is_success()


def test_add_refuses_a_value_that_is_not_a_whole_number_under_term_frequency_weighting(tmp_path):
    """A term frequency is a count. The rule names the first value that
    breaks it, the positivity rule first, refuses it at insert and at
    load, and the dot product takes the same values."""
    index = with_sparse(weighting="bm25")
    message = one_error(index, {"id": "z", "vector": vec(1), "sparse": sparse_vector((1, 2.0), (2, 0.5))})
    assert message == ("Vector z: ValueError: Sparse vector value at index 1 is 0.5, and a "
                       "space weighted by term frequency takes whole numbers alone")
    assert "z" not in index and len(index) == 0
    assert sparse_keys(index)["sparse_records"] == "0"
    message = one_error(index, {"id": "n", "vector": vec(1), "sparse": sparse_vector((1, -0.5))})
    assert message == ("Vector n: ValueError: Sparse vector value at index 0 is -0.5, and a "
                       "space weighted by term frequency takes values above zero alone")
    # Whole numbers of any size, as floats or ints, are counts.
    assert index.add({"id": "w", "vector": vec(2),
                      "sparse": sparse_vector((1, 3.0), (2, 1), (3, 1e6))}).is_success()
    assert index.query(arms=[{"sparse": sparse_vector((3, 1.0))}], top_k=5)[0]["id"] == "w"
    # The dot product takes the fraction, and a directory holding it,
    # redeclared bm25 by hand, is refused at load by the same rule.
    dot = with_sparse()
    assert dot.add({"id": "z", "vector": vec(1), "sparse": sparse_vector((1, 2.0), (2, 0.5))}).is_success()
    path = tmp_path / "fraction.zdb"
    dot.save(str(path))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config["spaces"][0]["index"]["weighting"] = {"type": "bm25", "k1": 1.2, "b": 0.75}
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    repair_manifest(path, "config.json")
    with pytest.raises(RuntimeError, match=(
            r"Failed to deserialize spaces/terms/postings\.zdbsparse: record \d+: Sparse vector "
            r"value at index 1 is 0\.5, and a space weighted by term frequency takes whole "
            r"numbers alone")):
        VectorDatabase().load(str(path))


def test_add_refuses_a_sparse_half_where_there_is_no_space_or_no_layer():
    dense = dense_only()
    assert one_error(dense, {"id": "a", "vector": vec(1), "sparse": sparse_vector((1, 1.0))}) == \
        "Vector a: ValueError: This collection declares no sparse space"
    assert one_error(dense, {"id": "b", "vector": vec(1), "text": "x"}) == \
        "Vector b: ValueError: This collection declares no sparse space"
    ids_only = with_sparse()
    assert one_error(ids_only, {"id": "c", "vector": vec(1), "text": "x"}) == \
        "Vector c: ValueError: This collection's sparse space takes no text"
    assert len(dense) == 0 and len(ids_only) == 0


def test_add_refuses_a_sparse_vector_where_the_space_takes_text():
    """A text layer issues the space's term ids, so a supplied vector on it
    is refused per record under either weighting, after the mapping's own
    rules and leaving nothing behind. An ids-only space takes the same
    vector."""
    for index in (with_text(), with_text(weighting="dot")):
        message = one_error(index, {"id": "s", "vector": vec(1), "sparse": sparse_vector((0, 1.0))})
        assert message == "Vector s: ValueError: This collection's sparse space takes text alone"
        assert "s" not in index and len(index) == 0
        assert sparse_keys(index)["sparse_records"] == "0"
        assert sparse_keys(index)["term_count"] == "0"
        message = one_error(index, {"id": "m", "vector": vec(3), "sparse": {"dims": [1]}})
        assert message == "Vector m: 'sparse' is missing 'values', one weight per dim"
        message = one_error(index, {"id": "v", "vector": vec(3), "sparse": sparse_vector((2, 1.0), (1, 1.0))})
        assert message.startswith("Vector v: ValueError: Sparse vector dims must be strictly increasing")
        assert index.add({"id": "t", "vector": vec(2), "text": "alpha"}).is_success()
    assert with_sparse().add({"id": "s", "vector": vec(1), "sparse": sparse_vector((0, 1.0))}).is_success()


def test_add_refuses_a_record_naming_both_and_a_text_that_is_not_a_str():
    index = with_text()
    assert one_error(index, {"id": "a", "vector": vec(1), "text": "x",
                             "sparse": sparse_vector((1, 1.0))}) == \
        "Vector a: a record fills the sparse space with 'sparse' or with 'text', not both"
    assert one_error(index, {"id": "b", "vector": vec(1), "text": 42}) == \
        "Vector b: 'text' must be a str, got int"


def test_add_reports_a_bad_vector_before_a_bad_sparse_half():
    """The vector's rule comes first, as it did before the sparse half existed."""
    index = with_sparse()
    message = one_error(index, {"id": "a", "vector": [1.0], "sparse": "nope"})
    assert message == "Vector a: Vector dimension mismatch: expected 4, got 1"


def test_add_batch_refuses_sparse_and_texts_arrays_of_the_wrong_length_or_type():
    index = with_text()
    with pytest.raises(ValueError, match="2 entries under 'sparse' and 3 under 'vectors'"):
        index.add({"ids": ["a", "b", "c"], "vectors": [vec(1), vec(2), vec(3)],
                   "sparse": [None, None]})
    with pytest.raises(ValueError, match="4 entries under 'texts' and 3 under 'embeddings'"):
        index.add({"ids": ["a", "b", "c"], "embeddings": [vec(1), vec(2), vec(3)],
                   "texts": ["a", "b", "c", "d"]})
    with pytest.raises(TypeError, match="add expected 'sparse' to be a list, got dict"):
        index.add({"ids": ["a"], "vectors": [vec(1)], "sparse": sparse_vector((1, 1.0))})
    with pytest.raises(TypeError, match="add expected 'texts' to be a list, got str"):
        index.add({"ids": ["a"], "vectors": [vec(1)], "texts": "one text"})
    # The batch rules hold on the NumPy shape as well.
    with pytest.raises(ValueError, match="1 entries under 'texts' and 2 under 'vectors'"):
        index.add({"ids": ["a", "b"], "vectors": np.array([vec(1), vec(2)], dtype=np.float32),
                   "texts": ["only one"]})
    assert len(index) == 0


def test_add_names_a_tokenizer_failure_per_record():
    def picky(text):
        if "bad" in text:
            raise KeyError("no such word")
        return text.split()

    index = with_text(tokenizer=picky)
    result = index.add({
        "ids": ["a", "b", "c"],
        "vectors": [vec(1), vec(2), vec(3)],
        "texts": ["good words", "a bad word", "more good words"],
    })
    assert result.total_inserted == 2
    assert result.ids == ["a", "c"]
    assert result.errors == ["Vector b: RuntimeError: The tokenizer raised KeyError: 'no such word'"]
    assert "b" not in index

    def not_iterable(text):
        return 7

    def wrong_items(text):
        return [1, 2]

    assert one_error(with_text(tokenizer=not_iterable),
                     {"id": "x", "vector": vec(1), "text": "t"}).startswith(
        "Vector x: RuntimeError: The tokenizer raised TypeError:")
    assert one_error(with_text(tokenizer=wrong_items),
                     {"id": "y", "vector": vec(1), "text": "t"}) == \
        ("Vector y: RuntimeError: The tokenizer raised TypeError: The tokenizer returned "
         "int where a str was expected. A tokenizer returns an iterable of str, one per term.")


def test_a_callable_tokenizer_counts_as_it_returns_and_a_generator_will_do():
    def shouty(text):
        for word in text.split():
            yield word.upper()

    index = with_text(tokenizer=shouty)
    index.add([{"id": "a", "vector": vec(1), "text": "fox fox dog"},
               {"id": "b", "vector": vec(2), "text": "dog"}])
    assert sparse_keys(index)["term_count"] == "2"
    # The query is tokenized the same way, so case does not matter and the
    # repeated term counts twice in the record.
    page = index.query(arms=[{"text": "FOX"}], top_k=5)
    assert ids(page) == ["a"]


# ============================================================================
# THE FIVE LEGACY SHAPES, UNCHANGED
# ============================================================================


@pytest.fixture(params=["dense", "sparse", "text"])
def any_index(request):
    return {"dense": dense_only, "sparse": with_sparse, "text": with_text}[request.param]()


def stored(index, record_ids):
    return {r["id"]: (r["vector"], r["metadata"]) for r in index.get_records(record_ids)}


def test_legacy_shape_1_single_object(any_index):
    result = any_index.add({"id": "doc1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"t": "h"}})
    assert (result.total_inserted, result.total_errors, result.ids, result.vector_shape) == \
        (1, 0, ["doc1"], (1, DIM))
    assert stored(any_index, ["doc1"])["doc1"][1] == {"t": "h"}
    assert sparse_keys(any_index).get("sparse_records", "0") == "0"


def test_legacy_shape_2_list_of_objects(any_index):
    result = any_index.add([
        {"id": "doc1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"t": "h"}},
        {"id": "doc2", "vector": [0.3, 0.4, 0.5, 0.6], "metadata": "plain"},
    ])
    assert (result.total_inserted, result.total_errors, result.ids) == (2, 0, ["doc1", "doc2"])
    assert stored(any_index, ["doc2"])["doc2"][1] == {"text": "plain"}
    assert sparse_keys(any_index).get("sparse_records", "0") == "0"


def test_legacy_shape_3_separate_arrays(any_index):
    result = any_index.add({
        "ids": ["doc1", "doc2"],
        "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6]],
        "metadatas": [{"t": "h"}, {"t": "w"}],
    })
    assert (result.total_inserted, result.total_errors, result.ids) == (2, 0, ["doc1", "doc2"])
    assert stored(any_index, ["doc1", "doc2"])["doc2"][1] == {"t": "w"}
    assert sparse_keys(any_index).get("sparse_records", "0") == "0"


def test_legacy_shape_4_list_with_numpy(any_index):
    result = any_index.add([
        {"id": "doc2", "values": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), "metadata": {"type": "blog"}},
        {"id": "doc3", "values": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32), "metadata": {"type": "news"}},
    ])
    assert (result.total_inserted, result.total_errors, result.ids) == (2, 0, ["doc2", "doc3"])
    assert stored(any_index, ["doc3"])["doc3"][1] == {"type": "news"}


def test_legacy_shape_5_separate_arrays_with_numpy(any_index):
    result = any_index.add({
        "ids": ["doc1", "doc2"],
        "embeddings": np.array([[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
        "metadatas": [{"t": "h"}, {"t": "w"}],
    })
    assert (result.total_inserted, result.total_errors, result.ids, result.vector_shape) == \
        (2, 0, ["doc1", "doc2"], (2, DIM))
    assert stored(any_index, ["doc1"])["doc1"][1] == {"t": "h"}


def test_legacy_shapes_report_the_same_record_errors(any_index):
    result = any_index.add([
        {"id": "ok", "values": [0.1, 0.2, 0.3, 0.4]},
        {"id": "short", "values": [0.1]},
        {"id": "nan", "values": [math.nan, 0.0, 0.0, 0.0]},
    ])
    assert result.total_inserted == 1 and result.total_errors == 2
    assert result.errors == [
        "Vector short: Vector dimension mismatch: expected 4, got 1",
        "Vector nan: Vector contains invalid value at index 0: NaN",
    ]
    with pytest.raises(ValueError, match="2 entries under 'ids' and 1 under 'vectors'"):
        any_index.add({"ids": ["a", "b"], "vectors": [[0.1, 0.2, 0.3, 0.4]]})


# ============================================================================
# QUERY
# ============================================================================


def corpus(index, n=60, cats=3, seed=5, whole=False):
    """Dense vectors, a sparse vector over forty term ids, and a declared field
    `cat` in rotation, on every record. The weights are fractions above one,
    or counts of one to three where the space is weighted by term frequency."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        count = 2 + int(rng.integers(0, 4))
        dims = sorted(set(int(d) for d in rng.integers(0, 40, size=count)))
        values = [1.0 + float(rng.random()) for _ in dims]
        if whole:
            values = [float(1 + int(3.0 * (value - 1.0))) for value in values]
        records.append({
            "id": f"r{i:03d}",
            "vector": rng.standard_normal(DIM).astype(np.float32).tolist(),
            "sparse": {"dims": dims, "values": values},
            "metadata": {"cat": f"c{i % cats}", "rank": i},
        })
    assert index.add(records).is_success()
    return records


def brute_dot(records, query, admit=lambda r: True):
    page = []
    for record in records:
        if not admit(record):
            continue
        weights = dict(zip(record["sparse"]["dims"], record["sparse"]["values"]))
        score = np.float32(0.0)
        for d, v in zip(query["dims"], query["values"]):
            if d in weights:
                score = np.float32(score + np.float32(weights[d]) * np.float32(v))
        if score != 0.0:
            page.append((record["id"], float(score)))
    page.sort(key=lambda hit: (-hit[1], hit[0]))
    return page


def same_page(query_page, search_page):
    assert len(query_page) == len(search_page)
    for q, s in zip(query_page, search_page):
        assert q["id"] == s["id"]
        assert q["score"] == s["score"], (q["id"], q["score"], s["score"])
        assert q["metadata"] == s["metadata"]


@pytest.mark.parametrize("filter", [
    None,
    {"cat": "c1"},
    {"cat": "c0", "rank": {"lt": 30}},
    {"rank": {"gte": 20}},
    {"$or": [{"cat": "c2"}, {"rank": {"lt": 3}}]},
    {"cat": "nobody"},
])
def test_a_one_arm_dense_query_is_the_search(filter):
    """Id for id and score bit for bit, unfiltered and under a declared, a
    mixed, an undeclared and a composed filter, and under one admitting
    nothing."""
    index = with_sparse(indexed_fields=["cat"])
    corpus(index, n=80)
    for seed in range(5):
        q = vec(100 + seed)
        for top_k in (1, 7, 20):
            page = index.query(arms=[{"vector": q}], filter=filter, top_k=top_k)
            same_page(page, index.search(q, filter=filter, top_k=top_k))
            for position, hit in enumerate(page):
                assert hit["contributions"] == [{"arm": 0, "rank": position + 1, "score": hit["score"]}]
                assert set(hit) == {"id", "score", "metadata", "contributions"}


def test_a_one_arm_dense_query_is_the_search_with_its_options():
    index = with_sparse()
    corpus(index)
    q = vec(9)
    same_page(index.query(arms=[{"vector": q, "ef_search": 300}], top_k=5),
              index.search(q, ef_search=300, top_k=5))
    same_page(index.query(arms=[{"vector": np.array(q, dtype=np.float32)}], top_k=5),
              index.search(np.array(q, dtype=np.float32), top_k=5))
    same_page(index.query(arms=[{"vector": np.array(q, dtype=np.float64)}], top_k=5),
              index.search(np.array(q, dtype=np.float64), top_k=5))
    assert index.query(arms=[{"vector": q}], top_k=0) == []


@pytest.mark.parametrize("mode", ["quantized_only", "quantized_with_raw"])
def test_a_one_arm_dense_query_is_the_search_on_a_quantized_index(mode):
    index = VectorDatabase().create(
        "hnsw", dim=16, space="cosine", expected_size=2000, indexed_fields=["cat"],
        quantization_config={"type": "pq", "subvectors": 4, "bits": 4, "training_size": 1000,
                             "storage_mode": mode},
        sparse={"name": "terms"},
    )
    rng = np.random.default_rng(3)
    n = 1500
    index.add({"ids": [f"q{i}" for i in range(n)],
               "embeddings": rng.standard_normal((n, 16)).astype(np.float32),
               "metadatas": [{"cat": "a" if i % 2 else "b"} for i in range(n)]})
    assert index.is_quantized()
    for seed in range(3):
        q = rng.standard_normal(16).astype(np.float32).tolist()
        for filter in (None, {"cat": "a"}):
            for rerank in (None, 0, 3):
                same_page(index.query(arms=[{"vector": q, "rerank": rerank}], filter=filter, top_k=10),
                          index.search(q, filter=filter, top_k=10, rerank=rerank))


def test_a_one_arm_sparse_query_is_the_dot_product_by_hand():
    index = with_sparse(indexed_fields=["cat"])
    records = corpus(index, n=80)
    query = {"dims": [3, 7, 11, 19, 25], "values": [1.0, 0.5, 2.0, 1.0, 0.25]}
    for filter, admit in [
        (None, lambda r: True),
        ({"cat": "c1"}, lambda r: r["metadata"]["cat"] == "c1"),
        ({"rank": {"lt": 40}}, lambda r: r["metadata"]["rank"] < 40),
        ({"cat": "c2", "rank": {"gte": 10}}, lambda r: r["metadata"]["cat"] == "c2" and r["metadata"]["rank"] >= 10),
    ]:
        expected = brute_dot(records, query, admit)
        page = index.query(arms=[{"sparse": query}], filter=filter, top_k=100)
        assert [(h["id"], h["score"]) for h in page] == expected, filter
        assert ids(index.query(arms=[{"sparse": query}], filter=filter, top_k=3)) == \
            [i for i, _ in expected[:3]]
        for position, hit in enumerate(page):
            assert hit["contributions"] == [{"arm": 0, "rank": position + 1, "score": hit["score"]}]
    # A query sharing no term with any record is an empty page, not an error.
    assert index.query(arms=[{"sparse": {"dims": [999], "values": [1.0]}}], top_k=5) == []
    # And the dims and values may be arrays.
    page = index.query(arms=[{"sparse": {"dims": np.array([3, 7]), "values": np.array([1.0, 0.5])}}],
                       top_k=5)
    assert page == index.query(arms=[{"sparse": {"dims": [3, 7], "values": [1.0, 0.5]}}], top_k=5)


def bm25_by_hand(docs, query_terms, k1=1.2, b=0.75):
    """The weighting as the sparse crate documents it, over term counts."""
    n = len(docs)
    lengths = {doc_id: sum(counts.values()) for doc_id, counts in docs.items()}
    mean_length = sum(lengths.values()) / n
    query_counts = {}
    for term in query_terms:
        query_counts[term] = query_counts.get(term, 0) + 1
    page = []
    for doc_id, counts in docs.items():
        score = 0.0
        for term, query_value in query_counts.items():
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = sum(1 for other in docs.values() if term in other)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            saturated = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * lengths[doc_id] / mean_length))
            score += query_value * idf * saturated
        if score != 0.0:
            page.append((doc_id, score))
    page.sort(key=lambda hit: (-hit[1], hit[0]))
    return page


def test_a_text_query_is_the_term_frequency_weighting_by_hand():
    texts = {
        "t0": "the quick brown fox jumps over the lazy dog",
        "t1": "a fox is a small wild dog",
        "t2": "the dog sleeps",
        "t3": "nothing here about animals",
        "t4": "quick quick slow",
    }
    index = with_text(weighting={"type": "bm25", "k1": 1.5, "b": 0.6})
    index.add([{"id": doc_id, "vector": vec(i), "text": text}
               for i, (doc_id, text) in enumerate(texts.items())])
    docs = {}
    for doc_id, text in texts.items():
        counts = {}
        for term in text.split():
            counts[term] = counts.get(term, 0) + 1
        docs[doc_id] = counts
    for query in ["Fox, DOG!", "quick", "the the", "dog fox quick", "zebra", "dog zebra"]:
        query_terms = [t for t in "".join(c if c.isalnum() else " " for c in query.lower()).split()]
        expected = bm25_by_hand(docs, query_terms, k1=1.5, b=0.6)
        page = index.query(arms=[{"text": query}], top_k=10)
        assert ids(page) == [i for i, _ in expected], query
        for hit, (_, score) in zip(page, expected):
            assert hit["score"] == pytest.approx(score, rel=1e-5), query


def test_idf_scope_reads_the_admitted_corpus_by_default_and_the_whole_index_on_request():
    index = with_text(indexed_fields=["cat"])
    texts = ["dog dog", "dog cat", "cat", "dog bird", "bird bird bird", "cat cat"]
    index.add([{"id": f"t{i}", "vector": vec(i), "text": text,
                "metadata": {"cat": "a" if i < 3 else "b"}}
               for i, text in enumerate(texts)])
    unfiltered_corpus = index.query(arms=[{"text": "dog", "idf": "corpus"}], top_k=10)
    unfiltered_global = index.query(arms=[{"text": "dog", "idf": "global"}], top_k=10)
    assert unfiltered_corpus == unfiltered_global
    # Under a filter the rarity is counted over the admitted records by
    # default, and over every live record on request, so the scores differ.
    filtered_corpus = index.query(arms=[{"text": "dog"}], filter={"cat": "a"}, top_k=10)
    filtered_global = index.query(arms=[{"text": "dog", "idf": "global"}], filter={"cat": "a"}, top_k=10)
    assert ids(filtered_corpus) == ids(filtered_global) == ["t0", "t1"]
    assert filtered_corpus[0]["score"] != filtered_global[0]["score"]


def rrf_by_hand(pages, k=60.0):
    fused = {}
    for arm, page in enumerate(pages):
        for position, hit in enumerate(page):
            entry = fused.setdefault(hit["id"], [0.0, []])
            entry[0] += 1.0 / (k + position + 1)
            entry[1].append({"arm": arm, "rank": position + 1, "score": hit["score"]})
    return sorted(((i, np.float32(s), c) for i, (s, c) in fused.items()),
                  key=lambda entry: (-entry[1], entry[0]))


def test_two_arms_fuse_by_reciprocal_rank_and_carry_their_contributions():
    index = with_sparse(indexed_fields=["cat"])
    corpus(index, n=80)
    sparse = {"dims": [3, 7, 11, 19], "values": [1.0, 0.5, 2.0, 1.0]}
    for seed, filter, fetch, k in [
        (1, None, None, None), (2, None, 25, None), (3, {"cat": "c2"}, None, None),
        (4, {"cat": "c2"}, 40, 10.0), (5, {"rank": {"lt": 50}}, None, 0.0),
    ]:
        q = vec(seed)
        depth = fetch if fetch is not None else 5 * 5
        dense_page = index.query(arms=[{"vector": q}], filter=filter, top_k=depth)
        sparse_page = index.query(arms=[{"sparse": sparse}], filter=filter, top_k=depth)
        expected = rrf_by_hand([dense_page, sparse_page], k=60.0 if k is None else k)
        fusion = None if k is None else {"type": "rrf", "k": k}
        page = index.query(arms=[{"vector": q}, {"sparse": sparse}], filter=filter, top_k=5,
                           fetch=fetch, fusion=fusion)
        assert len(page) == 5
        for hit, (doc_id, score, contributions) in zip(page, expected):
            assert hit["id"] == doc_id, (seed, filter)
            assert np.float32(hit["score"]) == score
            assert hit["contributions"] == contributions
        # Best first, ties by id, and the same page again.
        assert all(a["score"] > b["score"] or (a["score"] == b["score"] and a["id"] < b["id"])
                   for a, b in zip(page, page[1:]))
        assert page == index.query(arms=[{"vector": q}, {"sparse": sparse}], filter=filter,
                                   top_k=5, fetch=fetch, fusion=fusion)
    # The fusion by name is the fusion at the published constant, and a
    # query over a text arm and a dense arm fuses the same way.
    q = vec(1)
    assert index.query(arms=[{"vector": q}, {"sparse": sparse}], top_k=5, fusion="rrf") == \
        index.query(arms=[{"vector": q}, {"sparse": sparse}], top_k=5)
    text_index = with_text()
    text_index.add([{"id": f"t{i}", "vector": vec(i), "text": t}
                    for i, t in enumerate(["fox dog", "dog", "cat", "fox"])])
    fused = text_index.query(arms=[{"vector": vec(0)}, {"text": "fox"}], top_k=4)
    assert fused[0]["id"] in ("t0", "t3")
    assert any(len(h["contributions"]) == 2 for h in fused)


def test_explain_reports_the_plan_without_running_the_query():
    index = with_text(indexed_fields=["cat"])
    index.add([{"id": f"t{i}", "vector": vec(i), "text": f"word{i % 4} common",
                "metadata": {"cat": f"c{i % 2}", "rank": i}} for i in range(30)])
    q = vec(99)
    plan = index.explain(arms=[{"vector": q}, {"text": "common word1"}], top_k=4)
    assert plan["admit"] == {"shape": "all"}
    assert plan["fusion"] == {"type": "rrf", "k": 60.0}
    assert [arm["space"] for arm in plan["arms"]] == ["default", "text"]
    assert [arm["kind"] for arm in plan["arms"]] == ["dense", "sparse"]
    assert [arm["fetch"] for arm in plan["arms"]] == [20, 20]
    for arm in plan["arms"]:
        assert set(arm) == {"space", "kind", "fetch", "cost_ns", "exact"}
        assert arm["cost_ns"] > 0.0
        assert isinstance(arm["exact"], bool)
    # A one arm query has no fusion and fetches its page; a declared filter
    # is a bitmap the dense arm scans exactly, an undeclared one a walk.
    one = index.explain(arms=[{"vector": q}], filter={"cat": "c1"}, top_k=4)
    assert one["fusion"] is None
    assert one["admit"] == {"shape": "bitmap", "admitted": 15}
    assert one["arms"] == [dict(one["arms"][0], fetch=4)]
    assert one["arms"][0]["exact"] is True
    walk = index.explain(arms=[{"text": "common"}], filter={"rank": {"lt": 10}}, top_k=4)
    assert walk["admit"] == {"shape": "sorted", "admitted": 10}
    assert index.explain(arms=[{"vector": q}], top_k=4, fetch=7)["arms"][0]["fetch"] == 7
    assert index.explain(arms=[{"vector": q}, {"text": "x"}], top_k=4,
                         fusion={"type": "rrf", "k": 5})["fusion"] == {"type": "rrf", "k": 5.0}
    # Nothing ran, so the dictionary issued no id for the query's terms.
    assert sparse_keys(index)["term_count"] == "5"
    assert index.explain(arms=[{"text": "zebra"}], top_k=4)["admit"] == {"shape": "all"}
    assert sparse_keys(index)["term_count"] == "5"


@pytest.mark.parametrize("arms,klass,fragment", [
    ("nope", TypeError, "arms must be a list of mappings"),
    ({"vector": [1.0] * DIM}, TypeError, "arms must be a list of mappings"),
    ([], ValueError, "A query needs at least one arm"),
    ([{"vector": [1.0] * DIM}] * 9, ValueError, "A query names at most 8 arms, got 9"),
    ([[1.0] * DIM], TypeError, "arms[0] must be a mapping naming one of 'vector', 'sparse' or 'text', got list"),
    ([{}], ValueError, "arms[0] names none of 'vector', 'sparse' or 'text'"),
    ([{"vector": None, "text": None}], ValueError, "arms[0] names none of"),
    ([{"vector": [1.0] * DIM, "text": "x"}], ValueError, "arms[0] names 'vector' and 'text'"),
    ([{"vector": [1.0] * DIM}, {"sparse": {"dims": [1], "values": [1.0]}, "text": "x"}], ValueError,
     "arms[1] names 'sparse' and 'text'"),
    ([{"vector": [1.0] * DIM, "idf": "corpus"}], ValueError,
     "arms[0] carries 'idf', which a 'vector' arm does not take. It takes 'vector', 'ef_search', 'rerank'."),
    ([{"text": "x", "ef_search": 5}], ValueError,
     "arms[0] carries 'ef_search', which a 'text' arm does not take. It takes 'text', 'idf'."),
    ([{"sparse": {"dims": [1], "values": [1.0]}, "rerank": 2}], ValueError,
     "which a 'sparse' arm does not take. It takes 'sparse', 'idf'."),
    ([{"vector": [1.0, 2.0]}], ValueError, "Search vector dimension mismatch: expected 4, got 2"),
    ([{"vector": []}], ValueError, "Search vector cannot be empty"),
    ([{"vector": [math.nan] + [0.0] * (DIM - 1)}], ValueError, "Search vector contains invalid value at index 0"),
    ([{"vector": "abc"}], TypeError, "arms[0]['vector'] must be a list of numbers or a one dimensional array, got str"),
    ([{"vector": [1.0] * DIM, "ef_search": 131_073}], ValueError, "ef_search must be at most 131072"),
    ([{"sparse": {"dims": [2, 1], "values": [1.0, 1.0]}}], ValueError,
     "Sparse vector dims must be strictly increasing"),
    ([{"sparse": {"dims": [1], "values": [1.0, 2.0]}}], ValueError, "Sparse vector has 1 dims and 2 values"),
    ([{"sparse": {"dims": [1], "values": [math.inf]}}], ValueError, "Sparse vector contains invalid value"),
    ([{"sparse": [1, 2]}], ValueError, "arms[0] 'sparse' must be a mapping {'dims': [...], 'values': [...]}, got list"),
    ([{"sparse": {"dims": [-1], "values": [1.0]}}], ValueError, "arms[0] 'sparse' dim -1 is outside"),
    ([{"text": 5}], TypeError, "arms[0]['text'] must be a str, got int"),
    ([{"text": "x", "idf": "sometimes"}], ValueError, "arms[0]['idf'] is 'sometimes', and it is 'corpus'"),
    ([{"text": "x", "idf": 3}], TypeError, "arms[0]['idf'] must be 'corpus' or 'global', got int"),
])
def test_query_and_explain_refuse_a_malformed_query(arms, klass, fragment):
    index = with_text()
    index.add({"id": "a", "vector": vec(1), "text": "x y"})
    for method in (index.query, index.explain):
        with pytest.raises(klass) as raised:
            method(arms=arms)
        assert fragment in str(raised.value), str(raised.value)
    # Nothing ran and nothing changed.
    assert len(index) == 1
    assert sparse_keys(index)["term_count"] == "2"


@pytest.mark.parametrize("kwargs,klass,fragment", [
    ({"top_k": 65_537}, ValueError, "top_k must be at most 65536"),
    ({"fetch": 65_537}, ValueError, "fetch must be at most 65536, got 65537"),
    ({"fusion": {"type": "rrf", "k": -1.0}}, ValueError,
     "Reciprocal rank constant is -1, and it must be finite and at least zero"),
    ({"fusion": {"type": "rrf", "k": math.nan}}, ValueError, "Reciprocal rank constant is NaN"),
    ({"fusion": "borda"}, ValueError, "fusion type 'borda' is not one this build has"),
    ({"fusion": {"type": "rrf", "weights": [1, 2]}}, ValueError,
     "fusion carries 'weights', and a fusion is {'type': 'rrf', 'k': 60.0}"),
    ({"fusion": {"k": 3}}, ValueError, "fusion is missing 'type'"),
    ({"fusion": {"type": "rrf", "k": "x"}}, TypeError, "fusion['k'] must be a number"),
    ({"fusion": 60}, TypeError, "fusion must be 'rrf' or a mapping"),
    ({"filter": {"cat": {"regex": "x"}}}, ValueError, "Unknown filter operation: regex"),
])
def test_query_and_explain_refuse_bad_query_arguments(kwargs, klass, fragment):
    index = with_sparse()
    index.add({"id": "a", "vector": vec(1), "sparse": {"dims": [1], "values": [1.0]}})
    arms = [{"vector": vec(2)}, {"sparse": {"dims": [1], "values": [1.0]}}]
    for method in (index.query, index.explain):
        with pytest.raises(klass) as raised:
            method(arms=arms, **kwargs)
        assert fragment in str(raised.value), str(raised.value)


def test_query_refuses_an_arm_the_index_has_no_space_or_layer_for():
    dense = dense_only()
    dense.add({"id": "a", "vector": vec(1)})
    with pytest.raises(ValueError, match="This collection declares no sparse space"):
        dense.query(arms=[{"sparse": {"dims": [1], "values": [1.0]}}])
    with pytest.raises(ValueError, match="This collection declares no sparse space"):
        dense.query(arms=[{"vector": vec(1)}, {"text": "x"}])
    ids_only = with_sparse()
    with pytest.raises(ValueError, match="This collection's sparse space takes no text"):
        ids_only.explain(arms=[{"text": "x"}])
    # A dense arm alone still answers on both.
    assert ids(dense.query(arms=[{"vector": vec(1)}], top_k=1)) == ["a"]


def test_query_refuses_a_sparse_arm_where_the_space_takes_text():
    """The pair of the refusal above: a sparse arm on a space with a text
    layer is refused at the door by query and explain, after the mapping's
    own rules, and the other arms still answer."""
    index = with_text()
    index.add({"id": "a", "vector": vec(1), "text": "alpha beta"})
    arm = {"sparse": sparse_vector((0, 1.0))}
    for method in (index.query, index.explain):
        with pytest.raises(ValueError, match="This collection's sparse space takes text alone"):
            method(arms=[arm])
        with pytest.raises(ValueError, match="This collection's sparse space takes text alone"):
            method(arms=[{"vector": vec(1)}, arm], top_k=5)
    with pytest.raises(ValueError, match=r"arms\[0\] 'sparse' is missing 'values'"):
        index.query(arms=[{"sparse": {"dims": [0]}}])
    assert ids(index.query(arms=[{"text": "beta"}], top_k=5)) == ["a"]
    assert ids(index.query(arms=[{"vector": vec(1)}, {"text": "alpha"}], top_k=5)) == ["a"]
    ids_only = with_sparse()
    ids_only.add({"id": "a", "vector": vec(1), "sparse": sparse_vector((0, 1.0))})
    assert ids(ids_only.query(arms=[arm], top_k=5)) == ["a"]


def test_a_tokenizer_failure_in_a_query_is_the_callers_own_exception():
    class Special(Exception):
        pass

    def picky(text):
        if "bad" in text:
            raise Special("cannot split this")
        return text.split()

    index = with_text(tokenizer=picky)
    index.add({"id": "a", "vector": vec(1), "text": "good"})
    with pytest.raises(Special, match="cannot split this"):
        index.query(arms=[{"text": "a bad query"}])
    with pytest.raises(Special):
        index.explain(arms=[{"text": "bad"}])
    with pytest.raises(TypeError, match="returned int where a str was expected"):
        with_text(tokenizer=lambda t: [1]).query(arms=[{"text": "x"}])
    with pytest.raises(TypeError):
        with_text(tokenizer=lambda t: 5).query(arms=[{"text": "x"}])
    # The index is untouched and still answers.
    assert ids(index.query(arms=[{"text": "good"}], top_k=5)) == ["a"]


def test_search_is_untouched_by_the_sparse_space():
    """`search` takes what it took and returns the dict it returned."""
    index = with_text()
    index.add({"id": "a", "vector": vec(1), "text": "x"})
    page = index.search(vec(1), top_k=1)
    assert set(page[0]) == {"id", "score", "metadata"}
    with_vector = index.search(vec(1), top_k=1, return_vector=True)
    assert set(with_vector[0]) == {"id", "score", "metadata", "vector"}
    batch = index.search([vec(1), vec(2)], top_k=1)
    assert len(batch) == 2


# ============================================================================
# EVERY SPACE SHAPE THROUGH A SAVE AND A LOAD
# ============================================================================


def pages_of(index, records):
    q = vec(77)
    sparse = {"dims": [1, 14, 27, 33], "values": [1.0, 2.0, 1.0, 0.5]}
    return (
        index.query(arms=[{"vector": q}], top_k=8),
        index.query(arms=[{"sparse": sparse}], top_k=20),
        index.query(arms=[{"sparse": sparse}], filter={"cat": "c1"}, top_k=20),
        index.query(arms=[{"vector": q}, {"sparse": sparse}], top_k=8),
    )


def test_a_dense_only_directory_stays_what_it_was(tmp_path):
    index = dense_only(indexed_fields=["cat"])
    records = corpus(with_sparse(indexed_fields=["cat"]))
    assert index.add([{k: v for k, v in r.items() if k != "sparse"} for r in records]).is_success()
    before = index.query(arms=[{"vector": vec(77)}], filter={"cat": "c1"}, top_k=8)
    path = tmp_path / "dense.zdb"
    index.save(str(path))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "1.1.0"
    assert not (path / "spaces").exists()
    assert "spaces" not in json.loads((path / "config.json").read_text(encoding="utf-8"))
    loaded = VectorDatabase().load(str(path))
    assert loaded.query(arms=[{"vector": vec(77)}], filter={"cat": "c1"}, top_k=8) == before
    assert sparse_keys(loaded) == {}


def test_a_dense_and_sparse_directory_round_trips(tmp_path):
    index = with_sparse(weighting="bm25", indexed_fields=["cat"])
    records = corpus(index, n=60, whole=True)
    assert index.remove_point("r004")
    before = pages_of(index, records)
    assert all(before)
    path = tmp_path / "both.zdb"
    index.save(str(path))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "2.0.0"
    assert "spaces/terms/postings.zdbsparse" in manifest["files_included"]
    assert "checksum" not in manifest["file_digests"]["spaces/terms/postings.zdbsparse"]
    assert (path / "spaces" / "terms" / "postings.zdbsparse").exists()
    assert not (path / "spaces" / "terms" / "terms.zdbdict").exists()
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["spaces"][0]["name"] == "terms"
    assert config["spaces"][0]["index"]["weighting"] == {"type": "bm25", "k1": 1.2, "b": 0.75}

    loaded = VectorDatabase().load(str(path))
    assert len(loaded) == 59
    assert pages_of(loaded, records) == before
    assert sparse_keys(loaded)["sparse_records"] == sparse_keys(index)["sparse_records"] == "59"
    assert sparse_keys(loaded)["sparse_weighting"] == "bm25"
    # A save of the loaded index is the same postings artefact again.
    again = tmp_path / "again.zdb"
    loaded.save(str(again))
    assert (again / "spaces" / "terms" / "postings.zdbsparse").read_bytes() == \
        (path / "spaces" / "terms" / "postings.zdbsparse").read_bytes()
    # And the loaded index keeps taking records into both spaces.
    assert loaded.add({"id": "new", "vector": vec(500), "sparse": {"dims": [1], "values": [9.0]}}).is_success()
    assert ids(loaded.query(arms=[{"sparse": {"dims": [1], "values": [1.0]}}], top_k=1)) == ["new"]


def test_a_text_directory_round_trips_with_its_dictionary(tmp_path):
    index = with_text(indexed_fields=["cat"])
    texts = ["the quick brown fox", "a lazy dog sleeps", "the fox and the dog", "quick quick slow",
             "nothing in common here"]
    index.add([{"id": f"t{i}", "vector": vec(i), "text": text, "metadata": {"cat": f"c{i % 2}"}}
               for i, text in enumerate(texts)])
    before = index.query(arms=[{"text": "quick fox"}], top_k=5)
    fused_before = index.query(arms=[{"vector": vec(1)}, {"text": "dog"}], filter={"cat": "c0"}, top_k=5)
    terms = sparse_keys(index)["term_count"]
    path = tmp_path / "text.zdb"
    index.save(str(path))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "2.0.0"
    assert "spaces/text/terms.zdbdict" in manifest["files_included"]
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["spaces"][0]["tokenizer"] == "simple"

    # No tokenizer handed, since the directory rebuilds the built-in one, and
    # "simple" handed, which matches.
    for loaded in (VectorDatabase().load(str(path)), VectorDatabase().load(str(path), tokenizer="simple")):
        assert loaded.query(arms=[{"text": "quick fox"}], top_k=5) == before
        assert loaded.query(arms=[{"vector": vec(1)}, {"text": "dog"}], filter={"cat": "c0"}, top_k=5) == fused_before
        assert sparse_keys(loaded)["term_count"] == terms
        assert sparse_keys(loaded)["sparse_tokenizer"] == "simple"
    # A new term after the load takes the next id, so the dictionary came back whole.
    loaded = VectorDatabase().load(str(path))
    loaded.add({"id": "z", "vector": vec(9), "text": "zebra"})
    assert int(sparse_keys(loaded)["term_count"]) == int(terms) + 1
    assert ids(loaded.query(arms=[{"text": "zebra"}], top_k=5)) == ["z"]


def test_an_external_tokenizer_directory_opens_with_the_callable_and_not_otherwise(tmp_path):
    index = with_text(tokenizer=whitespace)
    index.add([{"id": "a", "vector": vec(1), "text": "Alpha beta"},
               {"id": "b", "vector": vec(2), "text": "beta GAMMA"}])
    before = index.query(arms=[{"text": "GAMMA"}], top_k=5)
    assert ids(before) == ["b"]
    path = tmp_path / "external.zdb"
    index.save(str(path))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["spaces"][0]["tokenizer"] == "external"

    with pytest.raises(RuntimeError) as refused:
        VectorDatabase().load(str(path))
    assert str(refused.value) == (
        "The sparse space 'text' was declared with a tokenizer of the caller's own, which the "
        "directory records as external and cannot reproduce. Open it with the same implementation "
        "handed to load.")
    with pytest.raises(RuntimeError, match="recorded its tokenizer as external and the one handed to load declares itself simple"):
        VectorDatabase().load(str(path), tokenizer="simple")
    with pytest.raises(ValueError, match="without supplying it"):
        VectorDatabase().load(str(path), tokenizer="external")
    with pytest.raises(ValueError, match="names a tokenizer this build does not have"):
        VectorDatabase().load(str(path), tokenizer="banana")
    with pytest.raises(TypeError, match="tokenizer must be 'simple' or a callable"):
        VectorDatabase().load(str(path), tokenizer=7)

    loaded = VectorDatabase().load(str(path), tokenizer=whitespace)
    assert loaded.query(arms=[{"text": "GAMMA"}], top_k=5) == before
    assert sparse_keys(loaded)["sparse_tokenizer"] == "external"
    # The engine cannot tell one callable from another, so a different
    # callable opens the directory too, and tokenizes the query its own way.
    other = VectorDatabase().load(str(path), tokenizer=lambda text: text.lower().split())
    assert other.query(arms=[{"text": "GAMMA"}], top_k=5) == []

    # A callable handed to a directory recording the built-in tokenizer, and
    # any tokenizer handed to one with no text layer, are refused.
    simple = with_text()
    simple.add({"id": "a", "vector": vec(1), "text": "x"})
    simple_path = tmp_path / "simple.zdb"
    simple.save(str(simple_path))
    with pytest.raises(RuntimeError, match="recorded its tokenizer as simple and the one handed to load declares itself external"):
        VectorDatabase().load(str(simple_path), tokenizer=whitespace)
    plain = dense_only()
    plain.add({"id": "a", "vector": vec(1)})
    plain_path = tmp_path / "plain.zdb"
    plain.save(str(plain_path))
    with pytest.raises(RuntimeError, match="A tokenizer was handed to load and no space in the directory takes text"):
        VectorDatabase().load(str(plain_path), tokenizer="simple")
    with pytest.raises(RuntimeError, match="no space in the directory takes text"):
        VectorDatabase().load(str(plain_path), tokenizer=whitespace)


def test_a_quantized_index_with_a_sparse_space_round_trips(tmp_path):
    index = VectorDatabase().create(
        "hnsw", dim=16, space="cosine", expected_size=2000, indexed_fields=["cat"],
        quantization_config={"type": "pq", "subvectors": 4, "bits": 4, "training_size": 1000,
                             "storage_mode": "quantized_with_raw"},
        sparse={"name": "terms", "weighting": "bm25"},
    )
    rng = np.random.default_rng(8)
    n = 1500
    index.add({
        "ids": [f"q{i}" for i in range(n)],
        "embeddings": rng.standard_normal((n, 16)).astype(np.float32),
        "metadatas": [{"cat": "a" if i % 2 else "b"} for i in range(n)],
        "sparse": [{"dims": sorted(set(int(d) for d in rng.integers(0, 50, size=3))),
                    "values": [1.0, 2.0, 3.0][:len(set(int(d) for d in rng.integers(0, 50, size=3)))]}
                   if i % 3 else None for i in range(n)],
    })
    assert index.is_quantized()
    q = rng.standard_normal(16).astype(np.float32).tolist()
    sparse = {"dims": [3, 17, 29], "values": [1.0, 1.0, 1.0]}
    before = (index.query(arms=[{"vector": q}], top_k=10),
              index.query(arms=[{"sparse": sparse}], filter={"cat": "a"}, top_k=10),
              index.query(arms=[{"vector": q}, {"sparse": sparse}], top_k=10))
    path = tmp_path / "quantized.zdb"
    index.save(str(path))
    loaded = VectorDatabase().load(str(path))
    assert loaded.is_quantized()
    assert (loaded.query(arms=[{"vector": q}], top_k=10),
            loaded.query(arms=[{"sparse": sparse}], filter={"cat": "a"}, top_k=10),
            loaded.query(arms=[{"vector": q}, {"sparse": sparse}], top_k=10)) == before


def test_the_graph_rebuild_fallback_carries_the_sparse_space(tmp_path):
    index = with_sparse(indexed_fields=["cat"])
    records = corpus(index, n=60)
    before = pages_of(index, records)
    path = tmp_path / "rebuild.zdb"
    index.save(str(path))
    (path / "hnsw_index.zdbgraph").unlink()
    loaded = VectorDatabase().load(str(path))
    assert len(loaded) == 60
    after = pages_of(loaded, records)
    # The sparse pages are exact and identical; the dense page is the rebuilt
    # graph's own and holds the same records to within its recall.
    assert after[1] == before[1]
    assert after[2] == before[2]
    assert len(set(ids(after[0])) & set(ids(before[0]))) >= 6


@pytest.mark.parametrize("artefact,fragment", [
    ("spaces/terms/postings.zdbsparse", "does not hold it. spaces/terms/postings.zdbsparse holds the postings of a sparse space"),
])
def test_a_sparse_artefact_the_manifest_names_and_the_directory_lacks_is_refused(tmp_path, artefact, fragment):
    index = with_sparse()
    corpus(index, n=10)
    path = tmp_path / "missing.zdb"
    index.save(str(path))
    (path / artefact).unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json names spaces/terms/postings.zdbsparse") as raised:
        VectorDatabase().load(str(path))
    assert fragment in str(raised.value)


def test_a_damaged_sparse_artefact_is_refused_by_its_frame(tmp_path):
    index = with_text()
    index.add([{"id": f"t{i}", "vector": vec(i), "text": f"word{i} common"} for i in range(5)])
    path = tmp_path / "damaged.zdb"
    index.save(str(path))
    artefact = path / "spaces" / "text" / "terms.zdbdict"
    raw = bytearray(artefact.read_bytes())
    raw[100] ^= 0x40
    artefact.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="terms.zdbdict"):
        VectorDatabase().load(str(path))
    # A length that no longer matches the manifest is refused before the frame.
    with open(artefact, "ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(RuntimeError, match="manifest.json records it as"):
        VectorDatabase().load(str(path))


def test_a_first_major_manifest_over_a_sparse_directory_is_refused(tmp_path):
    index = with_sparse()
    corpus(index, n=10)
    path = tmp_path / "downgraded.zdb"
    index.save(str(path))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["format_version"] = "1.1.0"
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config.json declares a sparse space, which no release writing that format holds"):
        VectorDatabase().load(str(path))


def test_a_hand_edited_space_declaration_is_held_to_the_rules(tmp_path):
    index = with_sparse()
    corpus(index, n=10)
    path = tmp_path / "edited.zdb"
    index.save(str(path))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config["spaces"][0]["name"] = "default"
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    repair_manifest(path, "config.json")
    with pytest.raises(RuntimeError, match="the spaces declared are invalid: space 'default' takes the dense space's name"):
        VectorDatabase().load(str(path))


def test_clear_compact_and_removal_keep_the_sparse_space_correct(tmp_path):
    index = with_text()
    index.add([{"id": f"t{i}", "vector": vec(i), "text": f"word{i} word{i % 5} common"} for i in range(40)])
    terms = sparse_keys(index)["term_count"]
    for i in range(10):
        assert index.remove_point(f"t{i}")
    assert index.delete(ids=["t10", "t11"]) == 2
    # word9 and word11 each named one record, and both are gone; word12 stays.
    assert ids(index.query(arms=[{"text": "word9 word11"}], top_k=5)) == []
    assert ids(index.query(arms=[{"text": "word12"}], top_k=5)) == ["t12"]
    assert sparse_keys(index)["sparse_records"] == "28"
    page = index.query(arms=[{"text": "word3 common"}], top_k=5)
    assert index.compact() == 12
    assert index.query(arms=[{"text": "word3 common"}], top_k=5) == page
    assert sparse_keys(index)["term_count"] == terms
    assert sparse_keys(index)["sparse_dead_postings"] == "0"
    path = tmp_path / "compacted.zdb"
    index.save(str(path))
    assert VectorDatabase().load(str(path)).query(arms=[{"text": "word3 common"}], top_k=5) == page

    assert index.clear() == 28
    assert sparse_keys(index)["sparse_records"] == "0"
    assert sparse_keys(index)["term_count"] == "0"
    assert index.query(arms=[{"text": "common"}], top_k=5) == []
    cleared = tmp_path / "cleared.zdb"
    index.save(str(cleared))
    manifest = json.loads((cleared / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "2.0.0"
    loaded = VectorDatabase().load(str(cleared))
    assert len(loaded) == 0
    loaded.add({"id": "fresh", "vector": vec(1), "text": "fresh start"})
    assert ids(loaded.query(arms=[{"text": "fresh"}], top_k=5)) == ["fresh"]
    assert sparse_keys(loaded)["term_count"] == "2"


def test_update_metadata_and_remove_where_see_the_sparse_records(tmp_path):
    index = with_sparse(indexed_fields=["cat"])
    corpus(index, n=30)
    sparse = {"dims": [3, 7, 11, 19], "values": [1.0, 1.0, 1.0, 1.0]}
    before = ids(index.query(arms=[{"sparse": sparse}], filter={"cat": "c0"}, top_k=30))
    assert before
    assert index.update_metadata(before[0], {"cat": "moved"})
    after = ids(index.query(arms=[{"sparse": sparse}], filter={"cat": "c0"}, top_k=30))
    assert after == before[1:]
    remaining = index.count({"cat": "c0"})
    removed = index.remove_where({"cat": "c0"})
    assert removed == remaining == 9
    assert ids(index.query(arms=[{"sparse": sparse}], filter={"cat": "c0"}, top_k=30)) == []
    assert ids(index.query(arms=[{"sparse": sparse}], filter={"cat": "moved"}, top_k=30)) == [before[0]]


# ============================================================================
# THE TOKENIZER BOUNDARY UNDER CONTENTION
# ============================================================================


def test_a_python_tokenizer_never_deadlocks_against_readers_and_writers():
    """Adds with texts, text queries, counts and stats from several threads
    under a callable tokenizer, which is where a tokenizer run under a guard
    would wait forever against a thread holding the interpreter lock."""
    def slow_split(text):
        return [w for w in text.lower().split() if w]

    index = with_text(tokenizer=slow_split, expected_size=5000)
    index.add([{"id": f"seed{i}", "vector": vec(i), "text": f"seed word{i % 7} common"}
               for i in range(50)])
    failures = []
    stop = threading.Event()

    def writer(tag):
        try:
            for i in range(60):
                index.add({"id": f"{tag}-{i}", "vector": vec(i + 1000),
                           "text": f"{tag} word{i % 7} common extra{i}"})
                index.remove_point(f"{tag}-{i - 5}") if i >= 5 else None
        except BaseException as e:  # noqa: BLE001
            failures.append(e)

    def reader():
        try:
            while not stop.is_set():
                index.query(arms=[{"text": "common word3"}], top_k=5)
                index.query(arms=[{"vector": vec(3)}, {"text": "seed extra1"}], top_k=5)
                index.count()
                index.get_stats()
                index.explain(arms=[{"text": "common"}], top_k=5)
        except BaseException as e:  # noqa: BLE001
            failures.append(e)

    threads = [threading.Thread(target=writer, args=(f"w{t}",)) for t in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads[:2]:
        thread.join(timeout=120)
    stop.set()
    for thread in threads[2:]:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a thread hung"
    assert failures == []
    assert len(index) == 50 + 2 * 5
    assert ids(index.query(arms=[{"text": "seed"}], top_k=1))[0].startswith("seed")


def test_a_text_query_never_returns_a_record_that_never_held_the_term_under_concurrent_reissues():
    """A text query counts its terms into ids under the guards its search
    holds, so a clear and an insert on another thread cannot reissue those
    ids between the count and the search. Each round the other thread clears
    and inserts one record carrying a term of its own, so every clear
    reissues id 0 to a new term, and a query for a term returns the one
    record that carried it or nothing."""
    index = with_text(expected_size=1000)
    index.add({"id": "r0", "vector": vec(0), "text": "only0"})
    generation = [0]
    stop = threading.Event()
    failures = []

    def mutator():
        try:
            rounds = 0
            while not stop.is_set():
                index.clear()
                rounds += 1
                index.add({"id": f"r{rounds}", "vector": vec(rounds), "text": f"only{rounds}"})
                generation[0] = rounds
        except BaseException as e:  # noqa: BLE001
            failures.append(e)

    thread = threading.Thread(target=mutator)
    thread.start()
    wrong = []
    hits = 0
    for _ in range(1500):
        current = generation[0]
        for hit in index.query(arms=[{"text": f"only{current}"}], top_k=5):
            hits += 1
            if hit["id"] != f"r{current}":
                wrong.append((current, hit["id"]))
    stop.set()
    thread.join(timeout=60)
    assert not thread.is_alive(), "the mutating thread hung"
    assert failures == []
    assert wrong == [], f"records returned for a term they never held: {wrong}"
    assert generation[0] > 0, "the mutating thread ran"
    assert hits > 0, "some query found the record its term named"
