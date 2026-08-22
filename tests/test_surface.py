"""The verbs added in the 0.8 surface pass.

`space` as a property, `AddResult.ids`, `__len__`, `__contains__`, `count`,
`list` with an offset, `remove_points`, `remove_where`, `update_metadata`,
`shrink_to_fit`, and the three filter operators `nin`, `any` and `all`.

Every method here has a case for an empty index, a case for the error it can
raise or the falsehood it can report, and a case for ordinary use.
"""

import json
import os
import tempfile

import numpy as np
import pytest
from helpers import repair_manifest
from zeusdb_vector_database import VectorDatabase


def build(n=0, dim=4, space="cosine", expected_size=100, metadatas=None):
    """An index holding `n` records, ids `r0` upward, orthogonal-ish vectors."""
    index = VectorDatabase().create(
        "hnsw", dim=dim, space=space, expected_size=expected_size
    )
    if n:
        rng = np.random.default_rng(11)
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        payload = {"ids": [f"r{i}" for i in range(n)], "embeddings": vectors}
        if metadatas is not None:
            payload["metadatas"] = metadatas
        assert index.add(payload).is_success()
    return index


# ------------------------------------------------------------
# space, as a property beside dim
# ------------------------------------------------------------
@pytest.mark.parametrize("space", ["cosine", "l2", "l1"])
def test_space_property_matches_get_space(space):
    index = build(space=space)
    assert index.space == space
    assert index.space == index.get_space()
    assert index.get_stats()["space"] == space


def test_space_property_is_present_on_an_empty_index():
    # The langchain adapter reads getattr(index, "space", None) before any
    # record exists, so an empty index has to answer.
    index = build()
    assert getattr(index, "space", None) == "cosine"


def test_space_property_is_read_only():
    index = build()
    with pytest.raises(AttributeError):
        index.space = "l2"


# ------------------------------------------------------------
# AddResult.ids
# ------------------------------------------------------------
def test_add_result_ids_are_empty_for_an_empty_input():
    index = build()
    result = index.add({"ids": [], "embeddings": []})
    assert result.ids == []
    assert result.total_inserted == 0


def test_add_result_ids_carry_the_supplied_ids_in_order():
    index = build()
    result = index.add(
        {"ids": ["a", "b", "c"], "embeddings": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}
    )
    assert result.ids == ["a", "b", "c"]
    assert len(result.ids) == result.total_inserted


def test_add_result_ids_carry_the_generated_ids():
    # Without this the generated ids were unreachable to the caller.
    index = build()
    result = index.add({"vectors": [[1, 0, 0, 0], [0, 1, 0, 0]]})
    assert len(result.ids) == 2
    assert all(i.startswith("vec_") for i in result.ids)
    for generated in result.ids:
        assert generated in index


def test_add_result_ids_omit_a_rejected_record():
    index = build()
    result = index.add(
        {
            "ids": ["good", "wrong_width"],
            "embeddings": [[1, 0, 0, 0], [1, 0]],
        }
    )
    assert result.total_errors == 1
    assert "good" in result.ids
    assert "wrong_width" not in result.ids
    # The list lines up with the count and with nothing else.
    assert len(result.ids) == result.total_inserted


def test_add_result_ids_report_an_overwrite():
    index = build()
    index.add({"ids": ["a"], "embeddings": [[1, 0, 0, 0]]})
    result = index.add({"ids": ["a"], "embeddings": [[0, 1, 0, 0]]}, overwrite=True)
    assert result.ids == ["a"]


# ------------------------------------------------------------
# __len__ and __contains__
# ------------------------------------------------------------
def test_len_is_zero_on_an_empty_index():
    assert len(build()) == 0


def test_len_tracks_the_live_record_count():
    index = build(12)
    assert len(index) == 12
    assert len(index) == index.get_vector_count()
    assert len(index) == int(index.get_stats()["total_vectors"])
    index.remove_point("r0")
    assert len(index) == 11
    assert len(index) == index.get_vector_count()


def test_contains_on_an_empty_index_is_false():
    index = build()
    assert "anything" not in index


def test_contains_agrees_with_the_contains_method():
    index = build(5)
    assert "r3" in index
    assert index.contains("r3")
    assert "missing" not in index
    assert not index.contains("missing")
    index.remove_point("r3")
    assert "r3" not in index


def test_contains_rejects_a_non_string_key():
    index = build(2)
    with pytest.raises(TypeError):
        3 in index  # noqa: B015


# ------------------------------------------------------------
# count
# ------------------------------------------------------------
def test_count_on_an_empty_index_is_zero():
    index = build()
    assert index.count() == 0
    assert index.count({"tier": "a"}) == 0


def test_count_without_a_filter_is_the_record_count():
    index = build(9)
    assert index.count() == 9
    assert index.count() == len(index)


def test_count_with_a_filter_is_exact():
    metas = [{"tier": "a" if i % 3 == 0 else "b", "n": i} for i in range(30)]
    index = build(30, metadatas=metas)
    assert index.count({"tier": "a"}) == 10
    assert index.count({"tier": "b"}) == 20
    assert index.count({"n": {"gte": 25}}) == 5
    assert index.count({"tier": "a", "n": {"lt": 10}}) == 4
    assert index.count({"tier": "nobody"}) == 0


def test_count_counts_past_the_scan_threshold():
    # The search path gives up at FULL_SCAN_THRESHOLD matches and traverses
    # instead. A count cannot give up, so a filter matching more than a search
    # would scan still returns the true total.
    n = 6000
    metas = [{"tier": "a"} for _ in range(n)]
    index = build(n, expected_size=n, metadatas=metas)
    assert index.count({"tier": "a"}) == n
    assert len(index.search([1, 0, 0, 0], filter={"tier": "a"}, top_k=5)) == 5


def test_count_rejects_an_unknown_operator():
    index = build(3, metadatas=[{"tier": "a"} for _ in range(3)])
    with pytest.raises(ValueError, match="Unknown filter operation"):
        index.count({"tier": {"nonsense": 1}})


# ------------------------------------------------------------
# list, and the paging order
# ------------------------------------------------------------
def test_list_on_an_empty_index_is_empty():
    index = build()
    assert index.list() == []
    assert index.list(number=5, offset=3) == []


def test_list_is_in_arrival_order():
    index = build(20)
    got = [i for i, _ in index.list(number=20)]
    assert got == [f"r{i}" for i in range(20)]


def test_list_pages_do_not_overlap_or_skip():
    index = build(37)
    seen = []
    offset = 0
    while True:
        page = index.list(number=7, offset=offset)
        if not page:
            break
        seen.extend(i for i, _ in page)
        offset += 7
    assert seen == [f"r{i}" for i in range(37)]
    assert len(set(seen)) == 37


def test_list_offset_past_the_end_is_empty_rather_than_an_error():
    index = build(4)
    assert index.list(number=10, offset=4) == []
    assert index.list(number=10, offset=1000) == []
    assert index.list(number=0) == []


def test_list_carries_the_metadata():
    metas = [{"tier": chr(ord("a") + i)} for i in range(5)]
    index = build(5, metadatas=metas)
    page = index.list(number=5)
    assert [m["tier"] for _, m in page] == ["a", "b", "c", "d", "e"]


def test_list_order_survives_a_save_and_load(tmp_path):
    index = build(15)
    before = [i for i, _ in index.list(number=15)]
    path = str(tmp_path / "ordered.zdb")
    index.save(path)
    loaded = VectorDatabase().load(path)
    assert [i for i, _ in loaded.list(number=15)] == before


def test_list_keeps_its_single_argument_call():
    # The langchain adapter calls list(number=k * 2) positionally by keyword.
    index = build(6)
    assert len(index.list(number=3)) == 3


# ------------------------------------------------------------
# remove_points
# ------------------------------------------------------------
def test_remove_points_on_an_empty_index_reports_every_id_missing():
    index = build()
    assert index.remove_points(["a", "b"]) == ["a", "b"]
    assert index.remove_points([]) == []


def test_remove_points_removes_a_batch_and_reports_the_misses():
    index = build(6)
    missing = index.remove_points(["r1", "nope", "r4"])
    assert missing == ["nope"]
    assert len(index) == 4
    assert "r1" not in index
    assert "r4" not in index
    assert "r0" in index


def test_remove_points_handles_a_repeated_id_once():
    index = build(3)
    assert index.remove_points(["r0", "r0"]) == []
    assert len(index) == 2


def test_remove_points_matches_the_loop_it_replaces():
    one = build(8)
    many = build(8)
    ids = ["r2", "r5", "absent"]
    looped = [i for i in ids if not one.remove_point(i)]
    assert many.remove_points(ids) == looped
    assert len(one) == len(many)
    assert sorted(i for i, _ in one.list(number=99)) == sorted(
        i for i, _ in many.list(number=99)
    )


def test_remove_points_strands_one_graph_node_per_record():
    index = build(6)
    before = int(index.get_stats()["graph_nodes"])
    index.remove_points(["r0", "r1"])
    assert int(index.get_stats()["graph_nodes"]) == before
    assert int(index.get_stats()["stranded_graph_nodes"]) == 2
    assert index.compact() == 2


# ------------------------------------------------------------
# remove_where
# ------------------------------------------------------------
def test_remove_where_on_an_empty_index_removes_nothing():
    index = build()
    assert index.remove_where({"tier": "a"}) == 0


def test_remove_where_removes_the_matching_set():
    metas = [{"tier": "a" if i % 2 else "b", "n": i} for i in range(10)]
    index = build(10, metadatas=metas)
    assert index.remove_where({"tier": "a"}) == 5
    assert len(index) == 5
    assert index.count({"tier": "a"}) == 0
    assert index.count({"tier": "b"}) == 5


def test_remove_where_matching_nothing_returns_zero():
    index = build(4, metadatas=[{"tier": "a"} for _ in range(4)])
    assert index.remove_where({"tier": "z"}) == 0
    assert len(index) == 4


def test_remove_where_refuses_an_empty_filter():
    # Everywhere else an empty filter matches every record. This is the one
    # method where following that rule destroys the index, so it is refused.
    index = build(5, metadatas=[{"tier": "a"} for _ in range(5)])
    with pytest.raises(ValueError, match="empty filter"):
        index.remove_where({})
    assert len(index) == 5
    # The rule itself is unchanged everywhere it is safe.
    assert index.count({}) == 5
    assert len(index.search([1, 0, 0, 0], filter={}, top_k=5)) == 5


def test_remove_where_rejects_an_unknown_operator():
    index = build(3, metadatas=[{"tier": "a"} for _ in range(3)])
    with pytest.raises(ValueError, match="Unknown filter operation"):
        index.remove_where({"tier": {"nonsense": 1}})
    assert len(index) == 3


def test_remove_where_selects_what_search_selects():
    metas = [{"tier": "a" if i % 3 == 0 else "b"} for i in range(12)]
    index = build(12, metadatas=metas)
    matched = {h["id"] for h in index.search([1, 0, 0, 0], filter={"tier": "a"}, top_k=99)}
    assert index.remove_where({"tier": "a"}) == len(matched)
    for gone in matched:
        assert gone not in index


def test_remove_where_by_a_reference_key_is_what_the_adapter_needs():
    # llama-index writes ref_doc_id into every record's metadata and raised
    # NotImplementedError for delete(ref_doc_id).
    metas = [{"ref_doc_id": "doc1" if i < 4 else "doc2"} for i in range(10)]
    index = build(10, metadatas=metas)
    assert index.remove_where({"ref_doc_id": "doc1"}) == 4
    assert index.count({"ref_doc_id": "doc2"}) == 6


# ------------------------------------------------------------
# update_metadata
# ------------------------------------------------------------
def test_update_metadata_on_an_empty_index_is_false():
    index = build()
    assert index.update_metadata("nobody", {"tier": "a"}) is False


def test_update_metadata_for_an_unknown_id_writes_nothing():
    index = build(3, metadatas=[{"tier": "a"} for _ in range(3)])
    assert index.update_metadata("missing", {"tier": "z"}) is False
    assert index.count({"tier": "a"}) == 3
    assert index.count({"tier": "z"}) == 0


def test_update_metadata_replaces_wholesale():
    index = build(2, metadatas=[{"tier": "a", "keep": 1}, {"tier": "b"}])
    assert index.update_metadata("r0", {"tier": "c"}) is True
    record = index.get_records("r0", return_vector=False)[0]
    assert record["metadata"] == {"tier": "c"}
    assert "keep" not in record["metadata"]


def test_update_metadata_agrees_with_add_overwrite():
    one = build(2, metadatas=[{"tier": "a", "keep": 1}, {"tier": "b"}])
    two = build(2, metadatas=[{"tier": "a", "keep": 1}, {"tier": "b"}])
    one.update_metadata("r0", {"tier": "c"})
    vector = two.get_records("r0", return_vector=True)[0]["vector"]
    two.add({"ids": ["r0"], "embeddings": [vector], "metadatas": [{"tier": "c"}]},
            overwrite=True)
    assert (one.get_records("r0", return_vector=False)[0]["metadata"]
            == two.get_records("r0", return_vector=False)[0]["metadata"])


def test_update_metadata_leaves_the_vector_and_the_graph_alone():
    index = build(5)
    before_vector = index.get_records("r2", return_vector=True)[0]["vector"]
    before_nodes = int(index.get_stats()["graph_nodes"])
    before_page = [h["id"] for h in index.search([1, 0, 0, 0], top_k=5)]
    assert index.update_metadata("r2", {"tier": "new"}) is True
    assert np.array_equal(
        index.get_records("r2", return_vector=True)[0]["vector"], before_vector
    )
    assert int(index.get_stats()["graph_nodes"]) == before_nodes
    assert int(index.get_stats()["stranded_graph_nodes"]) == 0
    assert [h["id"] for h in index.search([1, 0, 0, 0], top_k=5)] == before_page
    assert len(index) == 5


def test_update_metadata_to_an_empty_mapping_clears_it():
    index = build(1, metadatas=[{"tier": "a"}])
    assert index.update_metadata("r0", {}) is True
    assert index.get_records("r0", return_vector=False)[0]["metadata"] == {}


def test_update_metadata_is_visible_to_the_filter():
    metas = [{"tier": "a"} for _ in range(4)]
    index = build(4, metadatas=metas)
    index.update_metadata("r1", {"tier": "b"})
    assert index.count({"tier": "a"}) == 3
    assert index.count({"tier": "b"}) == 1
    hits = index.search([1, 0, 0, 0], filter={"tier": "b"}, top_k=4)
    assert [h["id"] for h in hits] == ["r1"]


def test_update_metadata_survives_a_save_and_load(tmp_path):
    index = build(3, metadatas=[{"tier": "a"} for _ in range(3)])
    index.update_metadata("r1", {"tier": "b"})
    path = str(tmp_path / "updated.zdb")
    index.save(path)
    loaded = VectorDatabase().load(path)
    assert loaded.count({"tier": "b"}) == 1
    assert loaded.get_records("r1", return_vector=False)[0]["metadata"] == {"tier": "b"}


# ------------------------------------------------------------
# shrink_to_fit
# ------------------------------------------------------------
def test_shrink_to_fit_on_an_empty_index_releases_the_reservation():
    # An empty index is not an index with nothing to release. It holds the
    # creation reservation that expected_size bought, and this hands it back.
    index = build(expected_size=1000)
    before = float(index.get_stats()["graph_memory_mb"])
    freed = index.shrink_to_fit()
    assert freed > 0
    assert float(index.get_stats()["graph_memory_mb"]) < before
    assert index.shrink_to_fit() == 0
    # It stays usable. It simply regrows from nothing.
    assert index.add({"ids": ["a"], "embeddings": [[1.0, 0.0, 0.0, 0.0]]}).is_success()
    assert len(index) == 1
    assert "a" in index


def test_shrink_to_fit_is_idempotent():
    index = build(400, expected_size=10, dim=32)
    index.shrink_to_fit()
    assert index.shrink_to_fit() == 0


def test_shrink_to_fit_reduces_the_reported_graph_memory():
    index = build(400, expected_size=10, dim=32)
    before = float(index.get_stats()["graph_memory_mb"])
    freed = index.shrink_to_fit()
    after = float(index.get_stats()["graph_memory_mb"])
    assert freed > 0
    assert after < before


def test_shrink_to_fit_returns_the_same_page():
    index = build(400, expected_size=10, dim=32)
    rng = np.random.default_rng(3)
    queries = rng.standard_normal((10, 32)).astype(np.float32)
    before = [[(h["id"], h["score"]) for h in index.search(q, top_k=10)] for q in queries]
    index.shrink_to_fit()
    after = [[(h["id"], h["score"]) for h in index.search(q, top_k=10)] for q in queries]
    assert before == after


def test_the_index_is_still_writable_after_a_shrink():
    index = build(200, expected_size=10, dim=16)
    index.shrink_to_fit()
    assert index.add({"ids": ["later"], "embeddings": [[1.0] * 16]}).is_success()
    assert "later" in index
    assert len(index) == 201
    assert [h["id"] for h in index.search([1.0] * 16, top_k=1)] == ["later"]


def test_compact_shrinks_the_graph_it_rebuilds():
    index = build(300, expected_size=10, dim=32)
    index.remove_points([f"r{i}" for i in range(50)])
    before = float(index.get_stats()["graph_memory_mb"])
    assert index.compact() == 50
    after = float(index.get_stats()["graph_memory_mb"])
    assert after < before
    assert index.shrink_to_fit() == 0


# ------------------------------------------------------------
# nin, any and all
# ------------------------------------------------------------
def tagged():
    metas = [
        {"tier": "a", "tags": ["x", "y"]},
        {"tier": "b", "tags": ["y", "z"]},
        {"tier": "c", "tags": ["z"]},
        {"tier": "a", "tags": []},
    ]
    return build(4, metadatas=metas)


def test_nin_on_an_empty_index_matches_nothing():
    index = build()
    assert index.count({"tier": {"nin": ["a"]}}) == 0


def test_nin_excludes_the_listed_values():
    index = tagged()
    assert index.count({"tier": {"nin": ["a"]}}) == 2
    assert index.count({"tier": {"nin": ["a", "b", "c"]}}) == 0
    assert index.count({"tier": {"nin": []}}) == 4


def test_nin_and_ne_agree_on_an_absent_field():
    # relay 44 fixed `ne` to answer false where the field is missing. `nin`
    # against a one element list means the same thing, so it answers the same.
    index = build(2, metadatas=[{"tier": "a"}, {"other": 1}])
    assert index.count({"tier": {"ne": "a"}}) == 0
    assert index.count({"tier": {"nin": ["a"]}}) == 0
    assert index.count({"tier": {"nin": ["zzz"]}}) == 1


def test_nin_is_the_negation_of_in_over_present_fields():
    index = tagged()
    assert index.count({"tier": {"in": ["a", "b"]}}) == 3
    assert index.count({"tier": {"nin": ["a", "b"]}}) == 1


def test_nin_on_search_returns_the_complement():
    index = tagged()
    hits = index.search([1, 0, 0, 0], filter={"tier": {"nin": ["a"]}}, top_k=4)
    assert sorted(h["id"] for h in hits) == ["r1", "r2"]


def test_any_matches_an_intersection():
    index = tagged()
    assert index.count({"tags": {"any": ["x"]}}) == 1
    assert index.count({"tags": {"any": ["x", "z"]}}) == 3
    assert index.count({"tags": {"any": []}}) == 0
    assert index.count({"tags": {"any": ["nothing"]}}) == 0


def test_all_matches_a_conjunction():
    index = tagged()
    assert index.count({"tags": {"all": ["y", "z"]}}) == 1
    assert index.count({"tags": {"all": ["x", "y"]}}) == 1
    assert index.count({"tags": {"all": ["x", "z"]}}) == 0
    # An empty conjunction holds of every record carrying the field.
    assert index.count({"tags": {"all": []}}) == 4


def test_any_and_all_on_a_scalar_field_read_it_as_one_element():
    index = tagged()
    assert index.count({"tier": {"any": ["a", "c"]}}) == 3
    assert index.count({"tier": {"all": ["a"]}}) == 2
    assert index.count({"tier": {"all": ["a", "b"]}}) == 0


def test_any_and_all_reject_a_non_array_target():
    index = tagged()
    assert index.count({"tags": {"any": "x"}}) == 0
    assert index.count({"tags": {"all": "x"}}) == 0


def test_the_new_operators_are_validated_like_the_others():
    index = tagged()
    for op in ("nin", "any", "all"):
        assert index.count({"tags": {op: ["x"]}}) >= 0
    with pytest.raises(ValueError, match="Unknown filter operation"):
        index.count({"tags": {"nin_typo": ["x"]}})


def test_text_match_still_raises_and_contains_is_what_it_wants():
    # llama-index maps TEXT_MATCH onto "text_match", which this engine does not
    # implement. `contains` is the substring operator it is asking for.
    index = build(2, metadatas=[{"body": "hello world"}, {"body": "goodbye"}])
    with pytest.raises(ValueError, match="Unknown filter operation"):
        index.count({"body": {"text_match": "world"}})
    assert index.count({"body": {"contains": "world"}}) == 1


# ------------------------------------------------------------
# The absent field, under the two new array operators
# ------------------------------------------------------------
def test_any_and_all_exclude_a_record_missing_the_field():
    # field_matches answers absence once, before dispatch, for every operator.
    index = build(2, metadatas=[{"tags": ["x"]}, {"other": 1}])
    assert index.count({"tags": {"any": ["x"]}}) == 1
    assert index.count({"tags": {"any": ["zzz"]}}) == 0
    assert index.count({"tags": {"all": []}}) == 1


# ------------------------------------------------------------
# The quantized paths, which is where the guard hoist could break
# ------------------------------------------------------------
def quantized(n=1600, dim=16, mode="quantized_only"):
    index = VectorDatabase().create(
        "hnsw",
        dim=dim,
        space="cosine",
        expected_size=n,
        quantization_config={
            "type": "pq",
            "storage_mode": mode,
            "training_size": 1000,
            "subvectors": 4,
        },
    )
    rng = np.random.default_rng(23)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    assert index.add(
        {
            "ids": [f"q{i}" for i in range(n)],
            "embeddings": vectors,
            "metadatas": [{"tier": "a" if i % 2 else "b"} for i in range(n)],
        }
    ).is_success()
    assert index.is_quantized()
    return index


def test_remove_points_on_a_quantized_index_clears_the_codes():
    index = quantized()
    before = int(index.get_stats()["quantized_codes_stored"])
    missing = index.remove_points(["q0", "q1", "absent"])
    assert missing == ["absent"]
    assert len(index) == 1598
    assert int(index.get_stats()["quantized_codes_stored"]) == before - 2
    assert "q0" not in index
    assert index.get_records(["q0", "q1"], return_vector=False) == []


def test_remove_where_on_a_quantized_index():
    index = quantized()
    assert index.count({"tier": "a"}) == 800
    assert index.remove_where({"tier": "a"}) == 800
    assert len(index) == 800
    assert index.count({"tier": "a"}) == 0
    assert int(index.get_stats()["quantized_codes_stored"]) == 800


def test_update_metadata_on_a_quantized_index_leaves_the_codes_alone():
    index = quantized()
    codes_before = index.get_stats()["quantized_codes_stored"]
    vector_before = index.get_records("q5", return_vector=True)[0]["vector"]
    query = index.get_records("q5", return_vector=True)[0]["vector"]
    page_before = [h["id"] for h in index.search(query, top_k=10)]

    assert index.update_metadata("q5", {"tier": "c"}) is True

    assert index.get_stats()["quantized_codes_stored"] == codes_before
    assert np.array_equal(
        index.get_records("q5", return_vector=True)[0]["vector"], vector_before
    )
    assert [h["id"] for h in index.search(query, top_k=10)] == page_before
    assert index.count({"tier": "c"}) == 1


def test_quantized_index_still_searches_after_a_shrink():
    index = quantized()
    rng = np.random.default_rng(77)
    queries = rng.standard_normal((5, 16)).astype(np.float32)
    before = [[(h["id"], h["score"]) for h in index.search(q, top_k=10)] for q in queries]
    index.shrink_to_fit()
    after = [[(h["id"], h["score"]) for h in index.search(q, top_k=10)] for q in queries]
    assert before == after


def tiers(n):
    """One metadata mapping per record, alternating tier."""
    return [{"tier": "gold" if i % 2 == 0 else "silver"} for i in range(n)]


# ------------------------------------------------------------
# delete, the name four of five comparators use
# ------------------------------------------------------------
def test_delete_dispatches_to_the_two_existing_methods():
    """An alias, and the existing names stay.

    A caller arriving from hnswlib, ChromaDB, Qdrant or LanceDB reaches for
    index.delete(...) and got an AttributeError.
    """
    index = build(6, metadatas=tiers(6))

    # A single string, matching get_records, and a list of strings.
    assert index.delete(ids="r1") == 1
    assert index.delete(ids=["r2", "r3"]) == 2
    assert not index.contains("r1")
    assert len(index) == 3

    # Absent ids count zero rather than raising. Removing a record that is not
    # there is the state the caller asked for.
    assert index.delete(ids=["nothing", "nowhere"]) == 0
    assert len(index) == 3

    # A repeated id names one record, so it counts once.
    assert index.delete(ids=["r4", "r4", "r4"]) == 1
    assert len(index) == 2

    # r0 and r5 are left, being tier gold and tier silver.
    assert sorted(record_id for record_id, _ in index.list(10)) == ["r0", "r5"]
    assert index.delete(where={"tier": "gold"}) == 1
    assert sorted(record_id for record_id, _ in index.list(10)) == ["r5"]


def test_delete_refuses_both_arguments_and_neither():
    """Two selections do not compose, and no selection is not everything."""
    index = build(4, metadatas=tiers(4))

    with pytest.raises(ValueError, match="not both"):
        index.delete(ids=["r1"], where={"tier": "gold"})

    with pytest.raises(ValueError, match="requires 'ids' or 'where'"):
        index.delete()

    # Neither call touched the index.
    assert len(index) == 4

    # A wrong type for ids says so rather than being read as a filter.
    with pytest.raises(TypeError, match="string or a list of strings"):
        index.delete(ids=17)

    # An empty list is a delete of nothing, not a delete of everything.
    assert index.delete(ids=[]) == 0
    assert len(index) == 4

    # An empty filter is refused by remove_where and delete inherits that.
    with pytest.raises(ValueError, match="requires a filter that selects records"):
        index.delete(where={})
    assert len(index) == 4

    # An empty index is not a special case for either argument.
    empty = build(0)
    assert empty.delete(ids=["absent"]) == 0
    assert empty.delete(where={"tier": "gold"}) == 0


def test_delete_agrees_with_the_methods_it_aliases():
    """The same selection through either name leaves the same index."""
    by_alias = build(8, metadatas=tiers(8))
    by_method = build(8, metadatas=tiers(8))

    assert by_alias.delete(ids=["r2", "r5"]) == 2
    assert by_method.remove_points(["r2", "r5"]) == []
    assert sorted(i for i, _ in by_alias.list(20)) == sorted(
        i for i, _ in by_method.list(20)
    )

    assert by_alias.delete(where={"tier": "gold"}) == by_method.remove_where(
        {"tier": "gold"}
    )
    assert sorted(i for i, _ in by_alias.list(20)) == sorted(
        i for i, _ in by_method.list(20)
    )

    # remove_points still reports which ids were missing, which is the detail
    # delete's count does not carry.
    assert by_method.remove_points(["r1", "absent"]) == ["absent"]


# ------------------------------------------------------------
# clear, which llama-index probes for and did not find
# ------------------------------------------------------------
def test_clear_empties_the_index_and_reclaims_the_graph():
    """A fresh graph rather than remove_points over every id.

    Removing every record one at a time leaves one stranded node per record, so
    an emptied index would report a graph full of dead nodes and no live ones.
    """
    index = build(6, metadatas=tiers(6))

    # Some debris first, so there is something for the replacement to reclaim.
    index.remove_points(["r1", "r2"])
    assert int(index.get_stats()["stranded_graph_nodes"]) == 2

    assert index.clear() == 4
    assert len(index) == 0
    assert index.get_vector_count() == 0
    assert int(index.get_stats()["graph_nodes"]) == 0
    assert int(index.get_stats()["stranded_graph_nodes"]) == 0
    assert int(index.get_stats()["total_vectors"]) == 0
    assert index.list(10) == []
    assert index.count() == 0
    assert not index.contains("r3")
    assert index.get_records(["r3", "r4"]) == []

    # A search on an emptied index is an empty page rather than an error.
    assert index.search([1.0, 0.0, 0.0, 0.0], top_k=5) == []
    assert index.count({"tier": "gold"}) == 0

    # The configuration survived.
    assert index.dim == 4
    assert index.space == "cosine"
    assert index.m == 16
    assert index.ef_construction == 200

    # And it is usable again, with the internal ids restarted so a generated id
    # is vec_1 rather than continuing from where the cleared records stopped.
    index.add({"embeddings": [[1.0, 0.0, 0.0, 0.0]]})
    assert [record_id for record_id, _ in index.list(10)] == ["vec_1"]
    assert index.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0]["id"] == "vec_1"


def test_clear_on_an_empty_index_is_not_an_error():
    """Nothing to remove returns zero and still replaces the graph."""
    index = build(0)
    assert index.clear() == 0
    assert len(index) == 0
    assert index.clear() == 0

    # Clearing twice in a row is the same as clearing once.
    index.add({"ids": ["a"], "embeddings": [[1.0, 0.0, 0.0, 0.0]]})
    assert index.clear() == 1
    assert index.clear() == 0
    assert len(index) == 0


def test_clear_keeps_a_fitted_codebook_and_drops_the_records():
    """Training is not undone, because it cannot be refitted from nothing.

    A trained quantized index stays trained and its replacement graph is a
    quantized graph. An untrained one returns to collecting, since what it had
    collected is gone.
    """
    rng = np.random.default_rng(11)
    vectors = rng.standard_normal((2000, 64)).astype(np.float32)
    ids = [str(k) for k in range(2000)]

    for mode in ("quantized_with_raw", "quantized_only"):
        index = VectorDatabase().create(
            "hnsw", dim=64, expected_size=20000,
            quantization_config={"type": "pq", "subvectors": 8, "bits": 8,
                                 "training_size": 2000, "storage_mode": mode},
        )
        assert index.add({"ids": ids, "embeddings": vectors}).is_success()
        assert index.is_quantized()

        assert index.clear() == 2000
        assert len(index) == 0
        # Still trained, and holding no records under either storage.
        assert index.is_quantized()
        assert int(index.get_stats()["quantized_codes_stored"]) == 0
        assert int(index.get_stats()["raw_vectors_stored"]) == 0
        assert int(index.get_stats()["graph_nodes"]) == 0

        # The replacement graph is a quantized graph, so refilling and
        # searching works without retraining.
        index.add({"ids": ["x", "y"], "embeddings": vectors[:2]})
        page = index.search(vectors[0], top_k=2)
        assert [hit["id"] for hit in page] == ["x", "y"]

    # An index still collecting starts collecting again.
    index = VectorDatabase().create(
        "hnsw", dim=64, expected_size=20000,
        quantization_config={"type": "pq", "subvectors": 8, "bits": 8,
                             "training_size": 2000, "storage_mode": "quantized_only"},
    )
    index.add({"ids": ids[:500], "embeddings": vectors[:500]})
    assert not index.is_quantized()
    assert index.training_vectors_needed() == 1500

    assert index.clear() == 500
    assert index.training_vectors_needed() == 2000
    assert index.get_training_progress() == 0.0


def test_clear_survives_a_save_and_load():
    """An emptied index saves and loads as an empty index."""
    index = build(5, metadatas=tiers(5))
    assert index.clear() == 5

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "cleared.zdb")
        index.save(path)
        loaded = VectorDatabase().load(path)

        assert len(loaded) == 0
        assert loaded.dim == 4
        assert loaded.m == 16
        assert loaded.search([1.0, 0.0, 0.0, 0.0], top_k=3) == []

        loaded.add({"ids": ["fresh"], "embeddings": [[1.0, 0.0, 0.0, 0.0]]})
        assert loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0]["id"] == "fresh"


# ------------------------------------------------------------
# m, ef_construction and expected_size as typed properties
# ------------------------------------------------------------
def test_construction_parameters_are_typed_properties():
    """int(index.get_stats()['m']) is what a caller wrote before this.

    hnswlib exposes M, ef_construction and max_elements as read-only
    properties, and every comparator exposes the equivalent typed.
    """
    index = VectorDatabase().create(
        "hnsw", dim=8, space="l2", m=24, ef_construction=120, expected_size=777
    )

    assert index.m == 24
    assert index.ef_construction == 120
    assert index.expected_size == 777
    assert isinstance(index.m, int)
    assert isinstance(index.ef_construction, int)
    assert isinstance(index.expected_size, int)

    # They agree with the text get_stats reports, which is where they were only
    # reachable before.
    stats = index.get_stats()
    assert int(stats["m"]) == index.m
    assert int(stats["ef_construction"]) == index.ef_construction
    assert int(stats["expected_size"]) == index.expected_size

    # Read-only. They describe a graph already built, so assigning would name a
    # graph that does not exist.
    for name in ("m", "ef_construction", "expected_size"):
        with pytest.raises(AttributeError):
            setattr(index, name, 99)

    # They hold on an empty index, since they are the declaration rather than a
    # measurement, and expected_size is a hint that len may exceed.
    assert len(index) == 0
    assert index.expected_size == 777
    index.add({"embeddings": [[1.0] + [0.0] * 7]})
    assert index.expected_size == 777
    assert len(index) == 1

    # The defaults are reachable the same way, including the m ladder.
    small = VectorDatabase().create("hnsw", dim=4)
    assert small.m == 16
    assert small.ef_construction == 200
    assert small.expected_size == 10000
    large = VectorDatabase().create("hnsw", dim=4, expected_size=50000)
    assert large.m == 32

    # A loaded index reports what it was built with.
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "props.zdb")
        index.save(path)
        loaded = VectorDatabase().load(path)
        assert loaded.m == 24
        assert loaded.ef_construction == 120
        assert loaded.expected_size == 777

    # Clearing the index does not change what it was built with.
    index.clear()
    assert (index.m, index.ef_construction, index.expected_size) == (24, 120, 777)


# ============================================================================
# THE SURFACE GAPS RELAY 102 CLOSED
# ============================================================================


def _presence_index(declared):
    """Three records: one with a value, one with a stored null, one with no key."""
    index = VectorDatabase().create(
        "hnsw", dim=4, space="l2", expected_size=50,
        indexed_fields=["lang"] if declared else [],
    )
    assert index.add([
        {"id": "has", "vector": [1.0, 0, 0, 0], "metadata": {"lang": "en", "n": 1}},
        {"id": "null", "vector": [0, 1.0, 0, 0], "metadata": {"lang": None, "n": 2}},
        {"id": "none", "vector": [0, 0, 1.0, 0], "metadata": {"n": 3}},
    ]).is_success()
    return index


@pytest.mark.parametrize("declared", [False, True],
                         ids=["undeclared", "declared"])
@pytest.mark.parametrize("condition,expected", [
    ({"lang": {"exists": True}}, ["has", "null"]),
    ({"lang": {"exists": False}}, ["none"]),
    ({"lang": {"is_missing": True}}, ["none"]),
    ({"lang": {"is_missing": False}}, ["has", "null"]),
    ({"lang": {"is_null": True}}, ["null"]),
    ({"lang": {"is_null": False}}, ["has", "none"]),
    ({"lang": {"exists": True, "is_null": False}}, ["has"]),
    ({"lang": {"exists": True, "eq": "en"}}, ["has"]),
    ({"$not": {"lang": {"exists": True}}}, ["none"]),
    ({"$or": [{"lang": {"is_null": True}}, {"lang": {"is_missing": True}}]},
     ["none", "null"]),
    ({"lang": {"is_missing": True}, "n": {"gt": 2}}, ["none"]),
    ({"lang": {"is_missing": True}, "n": {"lt": 2}}, []),
])
def test_the_presence_operators_agree_on_both_filter_paths(declared, condition, expected):
    """A declared column and the metadata walk answer these identically.

    The column path is a bitmap of the slots that hold a value, taken inside the
    live set. The walk asks the record. Declaring the field is what decides
    which runs, so both are parametrized here rather than trusted to agree.
    """
    index = _presence_index(declared)
    page = index.search([1.0, 0, 0, 0], filter=condition, top_k=9)
    assert sorted(hit["id"] for hit in page) == expected
    assert index.count(condition) == len(expected)


def test_a_stored_null_and_a_missing_field_are_different_things():
    """They are distinct in the storage format, which is what these rest on."""
    index = _presence_index(False)
    stored = {r["id"]: r["metadata"] for r in index.get_records(["has", "null", "none"])}
    assert stored["null"]["lang"] is None
    assert "lang" in stored["null"]
    assert "lang" not in stored["none"]
    # `eq` against null was already the way to ask, and `is_null` agrees with it.
    assert index.count({"lang": None}) == index.count({"lang": {"is_null": True}}) == 1


@pytest.mark.parametrize("operator", ["exists", "is_missing", "is_null"])
def test_a_presence_operator_takes_a_boolean_and_nothing_else(operator):
    index = _presence_index(False)
    with pytest.raises(ValueError, match=rf'"{operator}" takes true or false'):
        index.count({"lang": {operator: "yes"}})


def test_a_presence_filter_does_not_delete_the_whole_index():
    """`remove_where` refuses an unconditional filter, and these are not one."""
    index = _presence_index(False)
    assert index.remove_where({"lang": {"is_missing": True}}) == 1
    assert len(index) == 2


# ---------------------------------------------------------------- get_records


def test_get_records_is_lenient_by_default_and_strict_on_request():
    index = _presence_index(False)
    present = index.get_records(["has", "gone", "also_gone"], return_vector=False)
    assert [r["id"] for r in present] == ["has"]

    with pytest.raises(KeyError) as raised:
        index.get_records(["has", "gone", "also_gone"], return_vector=False, strict=True)
    message = str(raised.value)
    # Every absent id, sorted, so the message does not depend on the ask order.
    assert "also_gone, gone" in message
    assert "2 ids" in message
    assert "  " not in message, "the message carries a run of spaces"


def test_get_records_strict_returns_normally_when_every_id_is_present():
    index = _presence_index(False)
    got = index.get_records(["none", "has"], return_vector=False, strict=True)
    assert sorted(r["id"] for r in got) == ["has", "none"]


# ---------------------------------------------------------------- generated ids


def test_a_generated_id_is_issued_once_in_the_life_of_an_index(tmp_path):
    """`clear` used to reset the counter, so `vec_1` was handed out again.

    An external reference to the first record then named a different one and
    nothing said so. The counter behind a generated id is now separate from the
    internal id counter, which still resets, because that one sizes a dense
    array the graph indexes by.
    """
    index = VectorDatabase().create("hnsw", dim=4, space="l2", expected_size=50)
    index.add({"vectors": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]})
    assert sorted(rid for rid, _ in index.list(10)) == ["vec_1", "vec_2", "vec_3"]

    index.clear()
    index.add({"vectors": [[0, 0, 0, 1.0]]})
    assert [rid for rid, _ in index.list(10)] == ["vec_4"]

    # And it survives a save and load, so the next process does not reissue.
    path = tmp_path / "generated.zdb"
    index.save(str(path))
    loaded = VectorDatabase().load(str(path))
    loaded.add({"vectors": [[1.0, 1.0, 0, 0]]})
    assert sorted(rid for rid, _ in loaded.list(10)) == ["vec_4", "vec_5"]


def test_a_generated_id_is_dense_rather_than_burning_internal_ids():
    """It used to draw from the internal id counter, which insertion also drew
    from, so three records with no ids of their own left the fourth at `vec_7`."""
    index = VectorDatabase().create("hnsw", dim=4, space="l2", expected_size=50)
    first = index.add({"vectors": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]})
    assert first.ids == ["vec_1", "vec_2", "vec_3"]
    assert index.add({"vectors": [[0, 0, 0, 1.0]]}).ids == ["vec_4"]


def test_a_directory_with_no_generated_counter_takes_its_floor_from_the_records(tmp_path):
    """An old directory carries no counter, so the records supply one."""
    index = VectorDatabase().create("hnsw", dim=4, space="l2", expected_size=50)
    index.add({"vectors": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]})
    path = tmp_path / "old.zdb"
    index.save(str(path))

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config.pop("generated_ids") == 3
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    repair_manifest(path, "config.json")

    loaded = VectorDatabase().load(str(path))
    assert loaded.add({"vectors": [[0, 0, 0, 1.0]]}).ids == ["vec_4"]


def test_an_id_a_caller_chose_still_raises_the_floor(tmp_path):
    """A record called `vec_9` is counted, so nothing generated collides with it."""
    index = VectorDatabase().create("hnsw", dim=4, space="l2", expected_size=50)
    index.add({"ids": ["vec_9"], "embeddings": [[1.0, 0, 0, 0]]})
    path = tmp_path / "chosen.zdb"
    index.save(str(path))
    loaded = VectorDatabase().load(str(path))
    assert loaded.add({"vectors": [[0, 1.0, 0, 0]]}).ids == ["vec_10"]


# ---------------------------------------------------------------- list cursor


def _paged(n=8):
    index = VectorDatabase().create("hnsw", dim=4, space="l2", expected_size=50)
    assert index.add({
        "ids": [f"k{i}" for i in range(n)],
        "embeddings": [[float(i), 0, 0, 0] for i in range(n)],
    }).is_success()
    return index


def test_a_cursor_page_is_stable_under_a_deletion_and_an_offset_page_is_not():
    index = _paged()
    first = [rid for rid, _ in index.list(3)]
    assert first == ["k0", "k1", "k2"]

    index.remove_point("k0")

    assert [rid for rid, _ in index.list(3, after="k2")] == ["k3", "k4", "k5"]
    # The same page by offset skips one, because a record ahead of it is gone.
    assert [rid for rid, _ in index.list(3, offset=3)] == ["k4", "k5", "k6"]


def test_a_cursor_pages_the_whole_index_exactly_once():
    index = _paged(11)
    seen, cursor = [], None
    while True:
        page = index.list(3, after=cursor) if cursor else index.list(3)
        if not page:
            break
        seen.extend(rid for rid, _ in page)
        cursor = page[-1][0]
    assert seen == [f"k{i}" for i in range(11)]


def test_a_cursor_survives_a_save_and_a_load(tmp_path):
    """Internal ids are what the order rests on and they survive the round trip."""
    index = _paged()
    path = tmp_path / "cursor.zdb"
    index.save(str(path))
    loaded = VectorDatabase().load(str(path))
    assert [rid for rid, _ in loaded.list(3, after="k2")] == ["k3", "k4", "k5"]


def test_a_cursor_naming_a_removed_record_raises():
    index = _paged()
    index.remove_point("k4")
    with pytest.raises(KeyError, match=r"list\(after='k4'\) names a record"):
        index.list(3, after="k4")


def test_a_cursor_and_an_offset_together_raise():
    index = _paged()
    with pytest.raises(ValueError, match=r"takes after or offset, not both"):
        index.list(3, offset=2, after="k1")


def test_a_cursor_at_the_last_record_returns_an_empty_page():
    index = _paged()
    assert index.list(3, after="k7") == []


# ---------------------------------------------------------------- stdout


def test_a_save_and_a_load_write_nothing_to_stdout(tmp_path, capfd):
    """A library should not write unsolicited progress to stdout.

    Every line the two used to print is a `debug` record now, and the default
    level is `warn`, so nothing appears unless it is asked for.
    """
    index = _paged()
    path = tmp_path / "quiet.zdb"
    index.save(str(path))
    VectorDatabase().load(str(path))
    captured = capfd.readouterr()
    assert captured.out == "", captured.out[:400]
