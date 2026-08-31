"""Metadata filter evaluation on search."""

import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 21: Test metadata filtering (basic)
# ------------------------------------------------------------
def test_metadata_filtering_basic():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="cosine", expected_size=10)
    
    records = [
        {"id": "v1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"author": "Alice", "score": 95}},
        {"id": "v2", "values": [0.9, 0.8, 0.7, 0.6], "metadata": {"author": "Bob", "score": 80}},
        {"id": "v3", "values": [0.15, 0.25, 0.35, 0.45], "metadata": {"author": "Alice", "score": 85}},
        {"id": "v4", "values": [0.92, 0.82, 0.72, 0.62], "metadata": {"author": "Charlie", "score": 78}},
    ]
    
    result = index.add(records)
    assert result.is_success()
    
    query = [0.1, 0.2, 0.3, 0.4]
    
    # Test equality filter
    alice_results = index.search(vector=query, filter={"author": "Alice"}, top_k=10)
    assert len(alice_results) == 2
    for r in alice_results:
        assert r['metadata']['author'] == "Alice"
    
    # Test numeric filter
    high_score_results = index.search(vector=query, filter={"score": {"gt": 90}}, top_k=10)
    assert len(high_score_results) == 1
    assert high_score_results[0]['metadata']['score'] == 95

# ------------------------------------------------------------
# Test 22: Test advanced metadata filtering
# ------------------------------------------------------------
def test_metadata_filtering_advanced():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)
    
    records = [
        {
            "id": "doc1", 
            "values": [0.1, 0.2, 0.3, 0.4], 
            "metadata": {
                "author": "Alice",
                "year": 2024,
                "rating": 4.5,
                "published": True,
                "tags": ["science", "ai"],
                "price": 29.99
            }
        },
        {
            "id": "doc2", 
            "values": [0.9, 0.8, 0.7, 0.6], 
            "metadata": {
                "author": "Bob",
                "year": 2023,
                "rating": 3.8,
                "published": False,
                "tags": ["technology"],
                "price": 19.99
            }
        }
    ]
    
    result = index.add(records)
    assert result.is_success()
    
    query = [0.1, 0.2, 0.3, 0.4]
    
    # Test multiple conditions
    complex_results = index.search(
        vector=query,
        filter={"published": True, "rating": {"gte": 4.0}, "year": {"gte": 2024}},
        top_k=10
    )
    assert len(complex_results) == 1
    assert complex_results[0]['id'] == 'doc1'
    
    # Test array contains
    ai_results = index.search(vector=query, filter={"tags": {"contains": "ai"}}, top_k=10)
    assert len(ai_results) == 1
    assert ai_results[0]['id'] == 'doc1'
    
    # Test string operations
    author_contains = index.search(vector=query, filter={"author": {"contains": "A"}}, top_k=10)
    assert len(author_contains) == 1
    assert author_contains[0]['metadata']['author'] == "Alice"

# ------------------------------------------------------------
# Shared index for the operator tests below
#
# A filter is applied to the candidates the graph already returned, so a
# filtered search sees exactly the candidate set an unfiltered search at the
# same top_k and ef_search would return. OPERATOR_QUERY and
# OPERATOR_SEARCH_KWARGS pin those parameters and the fixture asserts full
# recall, which is what makes the exact result sets below valid assertions
# rather than approximations.
# ------------------------------------------------------------
OPERATOR_QUERY = [1.0, 0.0, 0.0, 0.0]
OPERATOR_SEARCH_KWARGS = {"top_k": 10, "ef_search": 200}
OPERATOR_ALL_IDS = ["r01", "r02", "r03", "r04"]


@pytest.fixture(scope="module")
def operator_index():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="cosine", expected_size=16)

    records = [
        {
            "id": "r01",
            "values": [1.0, 0.0, 0.0, 0.0],
            "metadata": {
                "name": "Alpha.pdf",
                "count": 10,
                "ratio": 1.5,
                "flag": True,
                "tags": ["ai", "science"],
                "nullable": None,
                "nested": {"key": "value", "n": 1},
            },
        },
        {
            "id": "r02",
            "values": [0.9, 0.1, 0.0, 0.0],
            "metadata": {
                "name": "Beta.txt",
                "count": 20,
                "ratio": 2.5,
                "flag": False,
                "tags": ["tech"],
                "nullable": "present",
            },
        },
        {
            "id": "r03",
            "values": [0.8, 0.2, 0.0, 0.0],
            "metadata": {
                "name": "gamma.pdf",
                "count": 30,
                "ratio": 3.0,
                "flag": True,
                "tags": [],
            },
        },
        # r04 carries only name and count, so it is the record that pins what
        # each operator does when the field is absent from the metadata.
        {
            "id": "r04",
            "values": [0.7, 0.3, 0.0, 0.0],
            "metadata": {"name": "Delta.md", "count": 10},
        },
    ]

    result = index.add(records)
    assert result.is_success()

    baseline = index.search(vector=OPERATOR_QUERY, **OPERATOR_SEARCH_KWARGS)
    assert sorted(r["id"] for r in baseline) == OPERATOR_ALL_IDS

    return index


def filtered_ids(index, filter):
    """Sorted ids matching a filter, using the pinned search parameters."""
    results = index.search(vector=OPERATOR_QUERY, filter=filter, **OPERATOR_SEARCH_KWARGS)
    return sorted(r["id"] for r in results)

# ------------------------------------------------------------
# Test 48: The eq operator
# ------------------------------------------------------------
def test_filter_operator_eq(operator_index):
    index = operator_index

    # Equality on each scalar type the metadata carries
    assert filtered_ids(index, {"name": {"eq": "Alpha.pdf"}}) == ["r01"]
    assert filtered_ids(index, {"count": {"eq": 10}}) == ["r01", "r04"]
    assert filtered_ids(index, {"ratio": {"eq": 1.5}}) == ["r01"]
    assert filtered_ids(index, {"flag": {"eq": True}}) == ["r01", "r03"]
    assert filtered_ids(index, {"flag": {"eq": False}}) == ["r02"]

    # eq compares numbers by magnitude, so an integer field matches an equal
    # float target. count is the integer 10 for r01 and r04, and 10.0 selects
    # both, exactly as lt and gte already did.
    assert filtered_ids(index, {"count": {"eq": 10.0}}) == ["r01", "r04"]

    # Numeric equality does not extend to booleans. A boolean field is not
    # equal to the integer 1.
    assert filtered_ids(index, {"flag": {"eq": 1}}) == []

    # A cross type comparison returns no match rather than raising.
    assert filtered_ids(index, {"count": {"eq": "10"}}) == []
    assert filtered_ids(index, {"name": {"eq": 10}}) == []

    # An absent field never matches, so eq cannot be used to find the records
    # that lack a field.
    assert filtered_ids(index, {"missing_field": {"eq": 1}}) == []

# ------------------------------------------------------------
# Test 49: The ne operator
# ------------------------------------------------------------
def test_filter_operator_ne(operator_index):
    index = operator_index

    assert filtered_ids(index, {"name": {"ne": "Alpha.pdf"}}) == ["r02", "r03", "r04"]
    assert filtered_ids(index, {"count": {"ne": 10}}) == ["r02", "r03"]
    assert filtered_ids(index, {"flag": {"ne": True}}) == ["r02"]

    # ne is not a negation of the condition. field_matches returns false when
    # the field is missing from the metadata and it returns that before the
    # operator is evaluated, so a record without the field is excluded by ne
    # exactly as it is by eq. r03 and r04 have no nullable field and neither
    # appears below, and r04 has no flag field and is absent above.
    assert filtered_ids(index, {"nullable": {"ne": "present"}}) == ["r01"]

    # A field no record carries yields nothing at all under ne.
    assert filtered_ids(index, {"missing_field": {"ne": 1}}) == []

    # ne is the negation of eq over the records that carry the field, so the
    # numeric equality of test 48 applies here too. r01 and r04 hold the
    # integer 10 and are excluded by a float target of 10.0.
    assert filtered_ids(index, {"count": {"ne": 10.0}}) == ["r02", "r03"]

# ------------------------------------------------------------
# Test 50: The lt and lte operators
# ------------------------------------------------------------
def test_filter_operators_lt_and_lte(operator_index):
    index = operator_index

    assert filtered_ids(index, {"count": {"lt": 20}}) == ["r01", "r04"]
    assert filtered_ids(index, {"count": {"lte": 20}}) == ["r01", "r02", "r04"]
    assert filtered_ids(index, {"count": {"lt": 10}}) == []
    assert filtered_ids(index, {"count": {"lte": 10}}) == ["r01", "r04"]

    # Ordered comparison converts both sides through f64, so an integer field
    # and a float bound compare as numbers. That is the opposite of eq and ne,
    # which compare the Value variants.
    assert filtered_ids(index, {"count": {"lt": 10.5}}) == ["r01", "r04"]
    assert filtered_ids(index, {"ratio": {"lt": 2.6}}) == ["r01", "r02"]
    assert filtered_ids(index, {"ratio": {"lte": 2.5}}) == ["r01", "r02"]

    # compare_values matches only a pair of numbers. Every other pairing
    # returns no match rather than raising, including a numeric field against a
    # string bound, a string field against any bound, and a boolean field.
    assert filtered_ids(index, {"count": {"lt": "20"}}) == []
    assert filtered_ids(index, {"name": {"lt": 5}}) == []
    assert filtered_ids(index, {"name": {"lte": "Zulu"}}) == []
    assert filtered_ids(index, {"flag": {"lt": 5}}) == []

    # Two bounds on one field intersect.
    assert filtered_ids(index, {"count": {"gte": 10, "lte": 20}}) == ["r01", "r02", "r04"]

    assert filtered_ids(index, {"missing_field": {"lt": 1}}) == []

# ------------------------------------------------------------
# Test 51: The startswith and endswith operators
# ------------------------------------------------------------
def test_filter_operators_startswith_and_endswith(operator_index):
    index = operator_index

    assert filtered_ids(index, {"name": {"startswith": "A"}}) == ["r01"]
    assert filtered_ids(index, {"name": {"startswith": "Alpha"}}) == ["r01"]
    assert filtered_ids(index, {"name": {"endswith": ".pdf"}}) == ["r01", "r03"]
    assert filtered_ids(index, {"name": {"endswith": ".md"}}) == ["r04"]

    # Both operators delegate to the Rust str methods, which are case
    # sensitive. Alpha.pdf is not matched by a lowercase a and gamma.pdf is not
    # matched by an uppercase G.
    assert filtered_ids(index, {"name": {"startswith": "a"}}) == []
    assert filtered_ids(index, {"name": {"startswith": "G"}}) == []

    # An empty prefix or suffix matches every record that has the field.
    assert filtered_ids(index, {"name": {"startswith": ""}}) == OPERATOR_ALL_IDS
    assert filtered_ids(index, {"name": {"endswith": ""}}) == OPERATOR_ALL_IDS

    # Both operators require a string on each side. A numeric field, a numeric
    # target and an array field all return no match rather than raising.
    assert filtered_ids(index, {"count": {"startswith": "1"}}) == []
    assert filtered_ids(index, {"name": {"endswith": 5}}) == []
    assert filtered_ids(index, {"tags": {"startswith": "ai"}}) == []

    assert filtered_ids(index, {"missing_field": {"startswith": "x"}}) == []

# ------------------------------------------------------------
# Test 52: The in operator
# ------------------------------------------------------------
def test_filter_operator_in(operator_index):
    index = operator_index

    assert filtered_ids(index, {"name": {"in": ["Alpha.pdf", "Beta.txt"]}}) == ["r01", "r02"]
    assert filtered_ids(index, {"count": {"in": [10, 30]}}) == ["r01", "r03", "r04"]
    assert filtered_ids(index, {"flag": {"in": [False]}}) == ["r02"]

    # A candidate absent from the array simply does not match.
    assert filtered_ids(index, {"name": {"in": ["nothing", "here"]}}) == []

    # in tests array membership with the same equality eq uses, so it inherits
    # the numeric comparison. count is the integer 10 for r01 and r04, and a
    # float member of 10.0 selects both.
    assert filtered_ids(index, {"count": {"in": [10.0]}}) == ["r01", "r04"]

    # An empty array matches nothing, which follows from membership rather than
    # being a special case.
    assert filtered_ids(index, {"count": {"in": []}}) == []

    # value_in_array matches only an array target. A scalar target returns no
    # match rather than raising, even when it equals the field value.
    assert filtered_ids(index, {"name": {"in": "Alpha.pdf"}}) == []
    assert filtered_ids(index, {"count": {"in": 10}}) == []

    # The field value is compared whole, so an array field is a member only of
    # an array of arrays.
    assert filtered_ids(index, {"tags": {"in": [["tech"]]}}) == ["r02"]
    assert filtered_ids(index, {"tags": {"in": ["tech"]}}) == []

    assert filtered_ids(index, {"missing_field": {"in": [1]}}) == []

# ------------------------------------------------------------
# Test 53: Null metadata values under the filter operators
# ------------------------------------------------------------
def test_filter_null_values(operator_index):
    index = operator_index

    # A stored None becomes Value::Null, and Null is one of the four variants
    # field_matches compares directly, so a bare None filter matches it. r01 is
    # the only record whose nullable field is None.
    assert filtered_ids(index, {"nullable": None}) == ["r01"]
    assert filtered_ids(index, {"nullable": {"eq": None}}) == ["r01"]
    assert filtered_ids(index, {"nullable": {"in": [None]}}) == ["r01"]

    # A stored null is a value and not an absent field, so the two are
    # distinguishable. r03 and r04 never carried a nullable field and are
    # excluded from every result here, the ne one included.
    assert filtered_ids(index, {"nullable": {"ne": None}}) == ["r02"]

    # Null is not ordered, so a comparison operator against it returns no match
    # rather than raising.
    assert filtered_ids(index, {"nullable": {"gt": None}}) == []
    assert filtered_ids(index, {"nullable": {"lt": 1}}) == []
    assert filtered_ids(index, {"count": {"eq": None}}) == []

# ------------------------------------------------------------
# Test 54: Direct equality against an array or an object value
# ------------------------------------------------------------
def test_filter_direct_equality_compares_arrays(operator_index):
    index = operator_index

    # An array filter value written without an operator is exact equality, the
    # same comparison an explicit eq performs. The README describes direct
    # equality as exact equality for any type and this is now that.
    assert filtered_ids(index, {"tags": ["tech"]}) == ["r02"]
    assert filtered_ids(index, {"tags": {"eq": ["tech"]}}) == ["r02"]
    assert filtered_ids(index, {"tags": ["ai", "science"]}) == ["r01"]

    # Equality, not membership. in already covers membership, so an array is
    # compared element by element and in order.
    assert filtered_ids(index, {"tags": ["science", "ai"]}) == []
    assert filtered_ids(index, {"tags": ["ai"]}) == []

    # An empty array is a value like any other and matches the record whose
    # tags are empty rather than matching everything or nothing.
    assert filtered_ids(index, {"tags": []}) == ["r03"]

    # A float filter value written without an operator behaves as it does under
    # eq, so it matches an equal integer field.
    assert filtered_ids(index, {"count": 10}) == ["r01", "r04"]
    assert filtered_ids(index, {"count": 10.0}) == ["r01", "r04"]

    # An empty filter matches every record, so it is not equivalent to a filter
    # no record satisfies.
    assert filtered_ids(index, {}) == OPERATOR_ALL_IDS


# ------------------------------------------------------------
# Test 94: A dict filter value is always the operator form
# ------------------------------------------------------------
def test_filter_object_condition_is_always_the_operator_form(operator_index):
    index = operator_index

    # A dict filter value is the operator map, and there is no separate syntax
    # for direct equality against a nested object because the two would be
    # indistinguishable. The keys of an object written as a bare condition are
    # read as operator names, so this raises rather than comparing.
    with pytest.raises(ValueError, match="Unknown filter operation: key"):
        filtered_ids(index, {"nested": {"key": "value"}})

    # eq is how equality against a nested object is written, and it compares
    # the whole object including its numbers.
    assert filtered_ids(index, {"nested": {"eq": {"key": "value", "n": 1}}}) == ["r01"]
    assert filtered_ids(index, {"nested": {"eq": {"key": "value", "n": 1.0}}}) == ["r01"]

    # Equality is over the whole object, so a subset of the keys does not
    # match and neither does a different value.
    assert filtered_ids(index, {"nested": {"eq": {"key": "value"}}}) == []
    assert filtered_ids(index, {"nested": {"eq": {"key": "other", "n": 1}}}) == []

# ------------------------------------------------------------
# Test 55: An unrecognised filter operator
# ------------------------------------------------------------
def test_filter_unknown_operator_raises(operator_index):
    index = operator_index

    # A filter naming an operator the engine does not implement is a mistake in
    # the query rather than a condition of the data, so it fails the search
    # with a ValueError naming the operator.
    with pytest.raises(ValueError, match="Unknown filter operation: not_an_operator"):
        filtered_ids(index, {"count": {"not_an_operator": 10}})

    with pytest.raises(ValueError, match="Unknown filter operation: regex"):
        filtered_ids(index, {"name": {"regex": "Alpha"}})

    # A satisfied known operator alongside an unknown one does not suppress it.
    with pytest.raises(ValueError, match="Unknown filter operation: not_an_operator"):
        filtered_ids(index, {"count": {"gt": 0, "not_an_operator": 1}})

    # The filter is checked before any record is examined, so a field no record
    # carries raises exactly as a populated one does. Checking during
    # evaluation alone would not, because a record without the field never
    # reaches the operator.
    with pytest.raises(ValueError, match="Unknown filter operation: not_an_operator"):
        filtered_ids(index, {"missing_field": {"not_an_operator": 1}})

    # The batch path raises for the whole call rather than returning one empty
    # result set per query.
    with pytest.raises(ValueError, match="Unknown filter operation: not_an_operator"):
        index.search(
            vector=[OPERATOR_QUERY, OPERATOR_QUERY],
            filter={"count": {"not_an_operator": 10}},
            **OPERATOR_SEARCH_KWARGS,
        )

    # Batches above five queries take the parallel path, which raises the same
    # way rather than losing the error in a worker thread.
    with pytest.raises(ValueError, match="Unknown filter operation: not_an_operator"):
        index.search(
            vector=[OPERATOR_QUERY] * 8,
            filter={"count": {"not_an_operator": 10}},
            **OPERATOR_SEARCH_KWARGS,
        )

    # Every documented operator survives the check, so validation and dispatch
    # agree about what is known.
    for condition in (
        {"count": {"eq": 10}},
        {"count": {"ne": 10}},
        {"count": {"gt": 0}},
        {"count": {"gte": 0}},
        {"count": {"lt": 99}},
        {"count": {"lte": 99}},
        {"tags": {"contains": "ai"}},
        {"name": {"startswith": "A"}},
        {"name": {"endswith": ".pdf"}},
        {"count": {"in": [10]}},
    ):
        filtered_ids(index, condition)

# ------------------------------------------------------------
# Test 95: Numbers compare by magnitude under every operator
# ------------------------------------------------------------
def test_filter_numeric_comparison_is_consistent_across_operators():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="cosine", expected_size=16)

    # 9007199254740993 is the first integer above 2^53 that no f64 can hold, so
    # it and its predecessor share one float representation. They are the pair
    # that separates an exact integer comparison from a comparison that casts
    # both sides to f64 first.
    records = [
        {
            "id": "m1",
            "values": [1.0, 0.0, 0.0, 0.0],
            "metadata": {"count": 10, "ratio": 10.0, "big": 9007199254740993},
        },
        {
            "id": "m2",
            "values": [0.9, 0.1, 0.0, 0.0],
            "metadata": {"count": 20, "ratio": 20.5, "big": 9007199254740992},
        },
    ]
    assert index.add(records).is_success()

    def ids(filter):
        results = index.search(vector=[1.0, 0.0, 0.0, 0.0], filter=filter, top_k=10, ef_search=200)
        return sorted(r["id"] for r in results)

    assert ids({}) == ["m1", "m2"]

    # An integer field and an equal float target match under every operator
    # that compares values, and a float field and an equal integer target do
    # the same in the other direction.
    assert ids({"count": {"eq": 10.0}}) == ["m1"]
    assert ids({"count": 10.0}) == ["m1"]
    assert ids({"count": {"in": [10.0]}}) == ["m1"]
    assert ids({"count": {"ne": 10.0}}) == ["m2"]
    assert ids({"count": {"gte": 10.0, "lte": 10.0}}) == ["m1"]
    assert ids({"ratio": {"eq": 10}}) == ["m1"]
    assert ids({"ratio": 10}) == ["m1"]
    assert ids({"ratio": {"in": [10]}}) == ["m1"]

    # Two integers above 2^53 that share an f64 representation stay distinct
    # under the ordered operators as well as under equality.
    assert ids({"big": {"eq": 9007199254740993}}) == ["m1"]
    assert ids({"big": {"eq": 9007199254740992}}) == ["m2"]
    assert ids({"big": {"lte": 9007199254740992}}) == ["m2"]
    assert ids({"big": {"gt": 9007199254740992}}) == ["m1"]
    assert ids({"big": {"gte": 9007199254740993}}) == ["m1"]

    # A boolean is not a number and a numeric string is not a number, under
    # equality and under ordering alike.
    assert ids({"count": {"eq": True}}) == []
    assert ids({"count": {"eq": "10"}}) == []
    assert ids({"count": {"in": ["10", True]}}) == []
    assert ids({"count": {"gte": "10"}}) == []


# ------------------------------------------------------------
# The filtered page is the nearest matching records
#
# The filter used to run after the graph had cut to top_k, so a selective
# filter returned whatever survived of a page it was never given a say in.
# Measured on three real 100,000 record sets, post-filter recall tracked
# selectivity exactly: 0.5010 at one match in two, 0.0090 at one in a hundred
# and 0.0000 below that. It now decides which records are ranked, by one of two
# paths.
#
# At or below FULL_SCAN_THRESHOLD matches the index walks the metadata, scores
# every match and ranks them, which is exact. Above it the walk stops and the
# graph traversal runs with the filter conjoined into the liveness predicate it
# already carried, which is the graph's own recall. The tests below hold the
# exact path to a brute force ranking and the graph path to a high overlap with
# one.
# ------------------------------------------------------------

# Mirrors FULL_SCAN_THRESHOLD in crates/hnsw/src/collection/search.rs. The Rust
# constant is the one that decides; this is what the boundary tests aim at.
FULL_SCAN_THRESHOLD = 5000

FILTER_CORPUS = 6000
FILTER_DIM = 8


def _filter_corpus(size=FILTER_CORPUS, dim=FILTER_DIM, seed=20260818):
    """A corpus whose metadata makes any selectivity exactly expressible.

    ``bucket`` names one record in ``n`` for each n in the powers of ten, and
    ``rank`` is the record's own index so a range filter matches an exact count.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((size, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"r{i:05d}" for i in range(size)]
    metadata = [
        {
            "s2": "y" if i % 2 == 0 else "n",
            "s10": "y" if i % 10 == 0 else "n",
            "s100": "y" if i % 100 == 0 else "n",
            "s1000": "y" if i % 1000 == 0 else "n",
            "one": "y" if i == 4321 else "n",
            "rank": i,
        }
        for i in range(size)
    ]
    return ids, vectors, metadata


def _filter_index(space="cosine", quantization_config=None, **kwargs):
    ids, vectors, metadata = _filter_corpus(**kwargs)
    index = VectorDatabase().create(
        "hnsw", dim=FILTER_DIM, space=space, expected_size=len(ids),
        quantization_config=quantization_config,
    )
    result = index.add({"ids": ids, "embeddings": vectors, "metadatas": metadata})
    assert result.is_success(), result.errors
    return index, vectors


def _exact_page(vectors, matching, query, k):
    """The k nearest of `matching` under cosine, computed exactly."""
    subset = np.asarray(sorted(matching))
    order = np.argsort(-(vectors[subset] @ query))[:k]
    return [f"r{subset[i]:05d}" for i in order]


def test_a_selective_filter_returns_the_exact_nearest_matching_records():
    """The scan path is exact, so its page is the brute force page."""
    index, vectors = _filter_index()
    query = vectors[7]

    for field, rate in (("s100", 100), ("s1000", 1000)):
        matching = [i for i in range(FILTER_CORPUS) if i % rate == 0]
        assert len(matching) <= FULL_SCAN_THRESHOLD
        page = index.search(vector=query, filter={field: "y"}, top_k=10)
        assert [hit["id"] for hit in page] == _exact_page(vectors, matching, query, 10), (
            f"the exact path did not return the nearest matching records for {field}"
        )
        for hit in page:
            assert hit["metadata"][field] == "y"


def test_a_broad_filter_keeps_the_graph_recall():
    """Above the threshold the traversal runs, and it is nearly exact."""
    index, vectors = _filter_index()
    query = vectors[11]

    # Every record carries a rank, so this admits all of them and is well above
    # the threshold. Half the corpus would not be, since the corpus is 6,000.
    matching = list(range(FILTER_CORPUS))
    assert len(matching) > FULL_SCAN_THRESHOLD
    page = index.search(vector=query, filter={"rank": {"gte": 0}}, top_k=10)
    assert len(page) == 10
    for hit in page:
        assert hit["metadata"]["rank"] >= 0, "a filtered page leaked a record"
    exact = set(_exact_page(vectors, matching, query, 10))
    overlap = len({hit["id"] for hit in page} & exact)
    assert overlap >= 9, f"graph recall on a broad filter fell to {overlap} of 10"


def test_a_filter_matching_one_record_returns_that_record():
    """The case the exact path exists for.

    Through the traversal alone this costs 128 to 359 milliseconds on a 100,000
    record set, because the graph has to walk almost every node before it can
    conclude there is nothing else to admit. Through the scan it is one walk of
    the metadata.
    """
    index, _ = _filter_index()
    page = index.search(vector=[1.0] * FILTER_DIM, filter={"one": "y"}, top_k=10)
    assert [hit["id"] for hit in page] == ["r04321"]


def test_top_k_is_the_page_size_and_not_the_pool():
    """Raising top_k under a filter adds results rather than rescuing them."""
    index, _ = _filter_index()
    query = [1.0] + [0.0] * (FILTER_DIM - 1)

    seen = {}
    for top_k in (1, 5, 10, 50, 60, 100):
        page = index.search(vector=query, filter={"s100": "y"}, top_k=top_k)
        seen[top_k] = [hit["id"] for hit in page]
        assert len(page) == min(top_k, 60), f"top_k={top_k} returned {len(page)}"
    # Every shorter page is a prefix of every longer one.
    for top_k in (1, 5, 10, 50):
        assert seen[top_k] == seen[100][:top_k]


def test_a_filter_matching_fewer_than_top_k_returns_what_matched():
    index, _ = _filter_index()
    query = [1.0] * FILTER_DIM
    page = index.search(vector=query, filter={"rank": {"lt": 3}}, top_k=10)
    assert sorted(hit["id"] for hit in page) == ["r00000", "r00001", "r00002"]
    assert index.search(vector=query, filter={"rank": {"lt": 0}}, top_k=10) == []
    assert index.search(vector=query, filter={"s2": "absent"}, top_k=10) == []


def test_the_scan_threshold_boundary_returns_one_page():
    """One below, on, and one above the threshold agree on the page.

    The path changes between the second and the third of these, because the walk
    gives up once it has counted one match more than the threshold. Both paths
    are asked the same question and both answer it.
    """
    index, vectors = _filter_index()
    query = vectors[3]

    pages = {}
    for matched in (FULL_SCAN_THRESHOLD - 1, FULL_SCAN_THRESHOLD, FULL_SCAN_THRESHOLD + 1):
        page = index.search(vector=query, filter={"rank": {"lt": matched}}, top_k=10)
        assert len(page) == 10
        for hit in page:
            assert hit["metadata"]["rank"] < matched
        pages[matched] = [hit["id"] for hit in page]

    # The three filters admit almost the same records, so the three pages should
    # be the same page. The one above the threshold is served by the traversal.
    assert pages[FULL_SCAN_THRESHOLD - 1] == pages[FULL_SCAN_THRESHOLD]
    assert pages[FULL_SCAN_THRESHOLD] == pages[FULL_SCAN_THRESHOLD + 1]
    assert pages[FULL_SCAN_THRESHOLD] == _exact_page(
        vectors, range(FULL_SCAN_THRESHOLD), query, 10
    )


def test_the_filtered_page_does_not_depend_on_hash_order():
    """A filtered page is the same in a fresh interpreter.

    The scan walks the metadata store in ``HashMap`` order, which the standard
    library seeds afresh in every process, so two equally distant records would
    come back in an order that varied run to run without a tie break on the
    external id. This runs the same query in three subprocesses.
    """
    script = textwrap.dedent(
        """
        import json
        import numpy as np
        from zeusdb_vector_database import VectorDatabase

        rng = np.random.default_rng(4242)
        # Deliberately duplicated vectors, so ties are guaranteed.
        base = rng.standard_normal((20, 4)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        vectors = np.repeat(base, 5, axis=0)
        index = VectorDatabase().create("hnsw", dim=4, expected_size=100)
        index.add({
            "ids": [f"t{i:03d}" for i in range(100)],
            "embeddings": vectors,
            "metadatas": [{"keep": "y"} for _ in range(100)],
        })
        page = index.search(vector=base[0], filter={"keep": "y"}, top_k=25)
        print(json.dumps([h["id"] for h in page]))
        """
    )
    seen = set()
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip().splitlines()[-1])
    assert len(seen) == 1, f"the filtered page varied across processes: {seen}"


def test_the_batch_paths_filter_the_same_way_as_the_single_path():
    """Sequential batches, parallel batches and single queries agree.

    The batch split is at five queries, so the two batches below take different
    code paths through the guards.
    """
    index, vectors = _filter_index()
    queries = [vectors[i].tolist() for i in range(7)]
    filter_ = {"s100": "y"}

    single = [
        [hit["id"] for hit in index.search(vector=q, filter=filter_, top_k=10)]
        for q in queries
    ]
    sequential = index.search(queries[:3], filter=filter_, top_k=10)
    parallel = index.search(queries, filter=filter_, top_k=10)

    assert [[hit["id"] for hit in page] for page in sequential] == single[:3]
    assert [[hit["id"] for hit in page] for page in parallel] == single


@pytest.mark.parametrize("storage_mode", ["quantized_with_raw", "quantized_only"])
def test_a_quantized_index_filters_on_both_paths(storage_mode):
    """The scan scores from raw vectors where there are any and codes otherwise.

    Under ``quantized_only`` the raw vectors are gone once training completes, so
    the scan scores each match against its reconstruction. That is exact over
    what the index still holds rather than over the vectors it was given, which
    is the same limit every other read of a ``quantized_only`` index carries.
    """
    index, _ = _filter_index(
        quantization_config={
            "type": "pq", "subvectors": 4, "bits": 4,
            "training_size": 1000, "storage_mode": storage_mode,
        },
    )
    assert index.is_quantized(), "the corpus should have trained the quantizer"

    query = [1.0] + [0.0] * (FILTER_DIM - 1)
    selective = index.search(vector=query, filter={"s100": "y"}, top_k=10)
    assert len(selective) == 10
    for hit in selective:
        assert hit["metadata"]["s100"] == "y"

    broad = index.search(vector=query, filter={"s2": "y"}, top_k=10)
    assert len(broad) == 10
    for hit in broad:
        assert hit["metadata"]["s2"] == "y"

    one = index.search(vector=query, filter={"one": "y"}, top_k=10)
    assert [hit["id"] for hit in one] == ["r04321"]


def test_a_filtered_search_survives_a_concurrent_insert():
    """The guards are held longer now, so this is where a deadlock would show.

    The filter predicate reads the metadata store, so a filtered search holds
    ``vector_metadata`` across the whole traversal rather than only afterwards,
    and it holds the graph guard beside it. Both paths are exercised here, since
    the rank filter admits every record and so runs above the threshold, and
    ``s1000`` admits six and so runs below it.
    """
    index, vectors = _filter_index()
    errors = []
    stop = threading.Event()

    def search_forever(filter_):
        try:
            while not stop.is_set():
                for i in range(0, 40):
                    page = index.search(vector=vectors[i], filter=filter_, top_k=5)
                    for hit in page:
                        assert hit["metadata"], "a hit came back with no metadata"
        except Exception as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    def insert_forever():
        try:
            rng = np.random.default_rng(99)
            i = 0
            while not stop.is_set() and i < 400:
                vector = rng.standard_normal(FILTER_DIM).astype(np.float32)
                vector /= np.linalg.norm(vector)
                index.add({
                    "id": f"live_{i:04d}", "values": vector.tolist(),
                    "metadata": {"s2": "n", "s10": "n", "s100": "n",
                                 "s1000": "n", "one": "n", "rank": 900000 + i},
                })
                i += 1
        except Exception as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    workers = [
        threading.Thread(target=search_forever, args=({"rank": {"gte": 0}},)),
        threading.Thread(target=search_forever, args=({"s1000": "y"},)),
        threading.Thread(target=search_forever, args=(None,)),
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

    # The inserted records carry s1000 "n", so the selective page is unchanged.
    page = index.search(vector=vectors[0], filter={"one": "y"}, top_k=5)
    assert [hit["id"] for hit in page] == ["r04321"]


# ------------------------------------------------------------
# Boolean composition
#
# `$and`, `$or` and `$not` are the three reserved keys. A mapping is still a
# conjunction of its entries, and a group is one entry of that conjunction, so
# every filter written before this existed means what it always did.
# ------------------------------------------------------------

# Mirrors MAX_FILTER_DEPTH in crates/core/src/filter.rs.
MAX_FILTER_DEPTH = 10


def _nest(depth):
    """A filter whose groups nest exactly `depth` levels.

    Depth 1 is the mapping itself, so depth 2 is one group and depth n is
    n - 1 of them.
    """
    nested = {"count": 10}
    for _ in range(depth - 1):
        nested = {"$or": [nested]}
    return nested


def test_or_selects_the_union_of_its_branches(operator_index):
    index = operator_index

    # Neither branch alone selects both records, and the flat language could
    # not ask for their union because a mapping is a conjunction and a field
    # maps to one condition.
    assert filtered_ids(index, {"name": "Alpha.pdf"}) == ["r01"]
    assert filtered_ids(index, {"name": "Beta.txt"}) == ["r02"]
    assert filtered_ids(
        index, {"$or": [{"name": "Alpha.pdf"}, {"name": "Beta.txt"}]}
    ) == ["r01", "r02"]

    # A branch is a whole filter, so it may carry several fields, and they are
    # conjoined inside it.
    assert filtered_ids(
        index,
        {"$or": [{"name": "Alpha.pdf", "count": 20}, {"count": 30}]},
    ) == ["r03"]

    # An empty disjunction matches nothing, which is what `any` against an
    # empty array already does.
    assert filtered_ids(index, {"$or": []}) == []

    # A group is one entry of the conjunction it sits in, so a field beside it
    # narrows the union rather than widening it.
    assert filtered_ids(
        index,
        {"flag": True, "$or": [{"count": 10}, {"count": 20}]},
    ) == ["r01"]


def test_and_is_explicit_as_well_as_implicit(operator_index):
    index = operator_index

    # The flat form is the conjunction and `$and` is the same conjunction
    # written out, so the two select the same records.
    assert filtered_ids(index, {"count": 10, "flag": True}) == ["r01"]
    assert filtered_ids(index, {"$and": [{"count": 10}, {"flag": True}]}) == ["r01"]

    # It earns its place where a disjunction has to sit inside a conjunction.
    assert filtered_ids(
        index,
        {"$and": [{"count": {"gte": 10}}, {"$or": [{"flag": False}, {"count": 30}]}]},
    ) == ["r02", "r03"]

    # An empty conjunction matches every record, which is what `all` against an
    # empty array already does and what an empty filter does.
    assert filtered_ids(index, {"$and": []}) == OPERATOR_ALL_IDS


def test_not_negates_a_whole_filter(operator_index):
    index = operator_index

    # `ne` is not negation. It excludes a record that lacks the field, so r04,
    # which carries no flag at all, is in neither result set.
    assert filtered_ids(index, {"flag": {"ne": True}}) == ["r02"]

    # `$not` is negation, so the same question asked of the group admits r04.
    assert filtered_ids(index, {"$not": {"flag": True}}) == ["r02", "r04"]

    # Negating a disjunction is written out rather than given a key of
    # its own.
    assert filtered_ids(
        index, {"$not": {"$or": [{"count": 10}, {"count": 20}]}}
    ) == ["r03"]

    # Two negations cancel.
    assert filtered_ids(index, {"$not": {"$not": {"count": 30}}}) == ["r03"]

    # Negating a filter that matches everything matches nothing, and negating
    # one that matches nothing matches everything.
    assert filtered_ids(index, {"$not": {}}) == []
    assert filtered_ids(index, {"$not": {"$or": []}}) == OPERATOR_ALL_IDS


def test_not_makes_an_absent_field_expressible(operator_index):
    index = operator_index

    # `{"field": {"all": []}}` is the empty conjunction, so it holds for every
    # value the field can carry and fails only where the field is missing.
    # That makes it "the field is present", and its negation is the filter for
    # a missing field, which the language had no way to write before.
    assert filtered_ids(index, {"nested": {"all": []}}) == ["r01"]
    assert filtered_ids(index, {"$not": {"nested": {"all": []}}}) == ["r02", "r03", "r04"]

    # r03 carries tags as an empty array and r04 carries no tags at all, so the
    # two questions have different answers and both are now askable.
    assert filtered_ids(index, {"$not": {"tags": {"all": []}}}) == ["r04"]
    assert filtered_ids(index, {"tags": []}) == ["r03"]


def test_reserved_keys_are_three_names_and_not_the_dollar_namespace():
    """A field whose name begins with a dollar still filters."""
    index = VectorDatabase().create("hnsw", dim=4, expected_size=8)
    index.add({
        "ids": ["a", "b"],
        "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "metadatas": [{"$price": 10, "$or": "kept"}, {"$price": 20, "$or": "other"}],
    })

    # Only $and, $or and $not are reserved. Every other dollar prefixed key is
    # an ordinary field name and is unaffected.
    hits = index.search(vector=[1.0, 0.0, 0.0, 0.0], filter={"$price": {"lt": 15}}, top_k=5)
    assert [hit["id"] for hit in hits] == ["a"]

    # A field genuinely named $or is no longer filterable, and it fails loudly
    # rather than selecting the wrong records. The metadata itself is intact.
    with pytest.raises(ValueError, match="reserved filter key"):
        index.search(vector=[1.0, 0.0, 0.0, 0.0], filter={"$or": "kept"}, top_k=5)
    assert index.get_records("a")[0]["metadata"]["$or"] == "kept"


def test_a_malformed_group_is_refused_before_any_record_is_read(operator_index):
    index = operator_index

    # A group takes a list of filters, so anything else names the key and says
    # what it takes.
    with pytest.raises(ValueError, match="reserved filter key"):
        filtered_ids(index, {"$or": {"count": 10}})
    with pytest.raises(ValueError, match="reserved filter key"):
        filtered_ids(index, {"$and": 3})

    # A branch of a group is a filter mapping, not a bare value.
    with pytest.raises(ValueError, match="must be a filter mapping"):
        filtered_ids(index, {"$or": ["count", 10]})

    # `$not` takes one filter mapping rather than a list, so there is no
    # convention to guess about how a list under it would combine.
    with pytest.raises(ValueError, match="reserved filter key"):
        filtered_ids(index, {"$not": [{"count": 10}]})

    # An unknown operator inside a group is rejected exactly as one at the top
    # level is, and before any record is examined.
    with pytest.raises(ValueError, match="Unknown filter operation: not_an_operator"):
        filtered_ids(index, {"$or": [{"count": {"not_an_operator": 1}}]})
    with pytest.raises(ValueError, match="Unknown filter operation: regex"):
        filtered_ids(index, {"$not": {"$and": [{"name": {"regex": "Alpha"}}]}})


def test_group_nesting_is_bounded(operator_index):
    index = operator_index

    # The limit counts the mapping itself as level one, so the deepest filter
    # that compiles carries MAX_FILTER_DEPTH - 1 groups.
    assert filtered_ids(index, _nest(MAX_FILTER_DEPTH)) == ["r01", "r04"]

    with pytest.raises(ValueError, match="nest to 10 levels"):
        filtered_ids(index, _nest(MAX_FILTER_DEPTH + 1))

    # It is refused wherever a filter is accepted, not only on search.
    with pytest.raises(ValueError, match="nest to 10 levels"):
        index.count(_nest(MAX_FILTER_DEPTH + 1))
    with pytest.raises(ValueError, match="nest to 10 levels"):
        index.remove_where(_nest(MAX_FILTER_DEPTH + 1))


def test_a_deeply_nested_value_is_refused_rather_than_overflowing_the_stack():
    """A pathological filter raises where it used to kill the process.

    Nesting is converted out of Python before any filter code sees it, and that
    recursion had no bound. About four thousand levels overflowed the stack,
    which takes the interpreter with it rather than raising.
    """
    index = VectorDatabase().create("hnsw", dim=4, expected_size=8)
    index.add({"ids": ["a"], "vectors": [[1.0, 0.0, 0.0, 0.0]], "metadatas": [{"x": 1}]})

    deep = {"x": 1}
    for _ in range(5000):
        deep = {"eq": deep}
    with pytest.raises(ValueError, match="deeper than 128 levels"):
        index.search(vector=[1.0, 0.0, 0.0, 0.0], filter={"x": deep}, top_k=1)

    # Metadata arrives through the same conversion and is bounded the same way.
    # `add` collects parse errors rather than raising them, which is what it
    # already did for every other malformed record, so the depth shows up as a
    # refused record and the index is left holding what it held.
    result = index.add({"ids": ["b"], "vectors": [[0.0, 1.0, 0.0, 0.0]],
                        "metadatas": [{"x": deep}]})
    assert result.total_inserted == 0
    assert result.total_errors == 1
    assert "deeper than 128 levels" in result.errors[0]
    assert "b" not in index
    assert len(index) == 1

    # And a value just inside the limit is accepted, so the guard bounds the
    # recursion rather than refusing nesting.
    ok = {"x": 1}
    for _ in range(120):
        ok = {"inner": ok}
    assert index.add({"ids": ["c"], "vectors": [[0.0, 0.0, 1.0, 0.0]],
                      "metadatas": [{"deep": ok}]}).is_success()


def test_remove_where_refuses_a_composed_filter_that_matches_everything():
    """The refusal is asked of the tree, not of the mapping being empty."""
    index = VectorDatabase().create("hnsw", dim=4, expected_size=8)
    index.add({
        "ids": ["a", "b"],
        "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "metadatas": [{"tier": "a"}, {"tier": "b"}],
    })

    unconditional = ({}, {"$and": []}, {"$not": {"$or": []}}, {"$and": [{"$and": []}]})
    for filter in unconditional:
        with pytest.raises(ValueError, match="empty filter"):
            index.remove_where(filter)
    assert len(index) == 2

    # A composed filter that does select records is removed as any other is.
    assert index.remove_where({"$or": [{"tier": "a"}, {"tier": "z"}]}) == 1
    assert len(index) == 1


def test_a_disjunction_crosses_the_scan_threshold_neither_branch_crosses():
    """The path is chosen by what the whole filter matches, not by a branch.

    Each branch here matches half the corpus, which is under the threshold, and
    their union is the whole of it, which is over. The scan therefore gives up
    on a filter whose branches would each have been answered exactly, and the
    traversal returns the union rather than either half.
    """
    index, vectors = _filter_index()
    query = vectors[13]

    half = FILTER_CORPUS // 2
    low = {"rank": {"lt": half}}
    high = {"rank": {"gte": half}}
    assert index.count(low) == half <= FULL_SCAN_THRESHOLD
    assert index.count(high) == half <= FULL_SCAN_THRESHOLD

    union = {"$or": [low, high]}
    assert index.count(union) == FILTER_CORPUS > FULL_SCAN_THRESHOLD

    page = index.search(vector=query, filter=union, top_k=10)
    assert len(page) == 10
    # The union admits every record, so the page is the one the traversal
    # returns unfiltered, and not either branch's exact page.
    unfiltered = index.search(vector=query, top_k=10)
    assert [hit["id"] for hit in page] == [hit["id"] for hit in unfiltered]

    exact = set(_exact_page(vectors, list(range(FILTER_CORPUS)), query, 10))
    assert len({hit["id"] for hit in page} & exact) >= 9

    # Below the threshold the disjunction is answered by the scan and is exact,
    # so a union of two selective branches is the exact union.
    selective = {"$or": [{"rank": {"lt": 700}}, {"rank": {"gte": FILTER_CORPUS - 700}}]}
    matching = list(range(700)) + list(range(FILTER_CORPUS - 700, FILTER_CORPUS))
    assert index.count(selective) == len(matching) <= FULL_SCAN_THRESHOLD
    page = index.search(vector=query, filter=selective, top_k=10)
    assert [hit["id"] for hit in page] == _exact_page(vectors, matching, query, 10)


def test_every_filter_valid_before_composition_selects_the_same_records(operator_index):
    """The regression set for the flat language, held against fixed answers.

    Every shape below was valid before `$and`, `$or` and `$not` existed. The
    expected ids were taken from the build immediately before the change and
    are asserted rather than recomputed, so a change in evaluation shows up
    here rather than being absorbed by a helper that changed with it.
    """
    index = operator_index

    legacy = [
        ({}, ["r01", "r02", "r03", "r04"]),
        ({"name": "Alpha.pdf"}, ["r01"]),
        ({"count": 10}, ["r01", "r04"]),
        ({"count": 10.0}, ["r01", "r04"]),
        ({"flag": True}, ["r01", "r03"]),
        ({"nullable": None}, ["r01"]),
        ({"tags": []}, ["r03"]),
        ({"nested": {"eq": {"key": "value", "n": 1}}}, ["r01"]),
        ({"count": {"eq": 20}}, ["r02"]),
        ({"count": {"ne": 10}}, ["r02", "r03"]),
        ({"count": {"gt": 10}}, ["r02", "r03"]),
        ({"count": {"gte": 20}}, ["r02", "r03"]),
        ({"count": {"lt": 20}}, ["r01", "r04"]),
        ({"count": {"lte": 20}}, ["r01", "r02", "r04"]),
        ({"count": {"gte": 10, "lte": 20}}, ["r01", "r02", "r04"]),
        ({"ratio": {"gt": 1.5}}, ["r02", "r03"]),
        ({"name": {"contains": "lph"}}, ["r01"]),
        ({"tags": {"contains": "ai"}}, ["r01"]),
        ({"name": {"startswith": "A"}}, ["r01"]),
        ({"name": {"endswith": ".pdf"}}, ["r01", "r03"]),
        ({"count": {"in": [10, 30]}}, ["r01", "r03", "r04"]),
        ({"count": {"nin": [10, 30]}}, ["r02"]),
        ({"count": {"nin": []}}, ["r01", "r02", "r03", "r04"]),
        ({"tags": {"any": ["ai", "tech"]}}, ["r01", "r02"]),
        ({"tags": {"any": []}}, []),
        ({"tags": {"all": ["ai", "science"]}}, ["r01"]),
        ({"tags": {"all": []}}, ["r01", "r02", "r03"]),
        ({"missing_field": "x"}, []),
        ({"name": {"endswith": ".pdf"}, "flag": True}, ["r01", "r03"]),
        ({"count": {"gte": 10}, "ratio": {"lt": 3.0}, "flag": True}, ["r01"]),
    ]
    for filter, expected in legacy:
        assert filtered_ids(index, filter) == expected, filter
        assert index.count(filter) == len(expected), filter
