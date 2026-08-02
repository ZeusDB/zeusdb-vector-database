"""Metadata filter evaluation on search."""

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

    # eq compares serde_json Values, and an integer Value never equals a float
    # Value even when the two numbers are mathematically equal. count is the
    # integer 10 for r01 and r04, and 10.0 matches nothing.
    assert filtered_ids(index, {"count": {"eq": 10.0}}) == []

    # A boolean field is not equal to the integer 1 either.
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

    # The integer against float behaviour noted in test 48 inverts here. Every
    # record with a count is unequal to 10.0, including the two whose count is
    # the integer 10.
    assert filtered_ids(index, {"count": {"ne": 10.0}}) == OPERATOR_ALL_IDS

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

    # in tests array membership by Value equality, so it inherits the integer
    # against float behaviour of eq. count is the integer 10 for r01 and r04,
    # and 10.0 is not a member.
    assert filtered_ids(index, {"count": {"in": [10.0]}}) == []

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
# Test 54: Direct equality against an array value
# ------------------------------------------------------------
def test_filter_direct_equality_ignores_array_values(operator_index):
    index = operator_index

    # Current behaviour, asserted rather than expected. field_matches compares
    # a condition directly only when it is a String, Number, Bool or Null, and
    # every other variant falls through to a bare false. An array filter value
    # written without an operator is therefore dropped, while the same array
    # under an explicit eq matches. The README describes direct equality as
    # exact equality for any type, which this contradicts for arrays.
    assert filtered_ids(index, {"tags": ["tech"]}) == []
    assert filtered_ids(index, {"tags": {"eq": ["tech"]}}) == ["r02"]

    # A float filter value written without an operator behaves as it does under
    # eq, which is the integer against float mismatch again.
    assert filtered_ids(index, {"count": 10}) == ["r01", "r04"]
    assert filtered_ids(index, {"count": 10.0}) == []

    # An empty filter matches every record, so it is not equivalent to a filter
    # no record satisfies.
    assert filtered_ids(index, {}) == OPERATOR_ALL_IDS

# ------------------------------------------------------------
# Test 55: An unrecognised filter operator
# ------------------------------------------------------------
def test_filter_unknown_operator_is_silently_dropped(operator_index):
    index = operator_index

    # Current behaviour, asserted rather than expected. evaluate_value_conditions
    # builds a ValueError naming the operator, but every call site reaches it
    # through matches_filter(...).unwrap_or(false), so the error is discarded
    # and the record is treated as a non match. The expectation this violates is
    # that a filter naming an operator the engine does not implement reaches the
    # caller as a ValueError rather than as an empty result set.
    assert filtered_ids(index, {"count": {"not_an_operator": 10}}) == []
    assert filtered_ids(index, {"name": {"regex": "Alpha"}}) == []

    # A satisfied known operator alongside an unknown one does not rescue the
    # record either.
    assert filtered_ids(index, {"count": {"gt": 0, "not_an_operator": 1}}) == []

    # An unknown operator on a field no record carries is short circuited by the
    # missing field check, so it is indistinguishable from the cases above.
    assert filtered_ids(index, {"missing_field": {"not_an_operator": 1}}) == []

    # The batch path discards the error the same way, returning one empty
    # result set per query rather than raising.
    batch = index.search(
        vector=[OPERATOR_QUERY, OPERATOR_QUERY],
        filter={"count": {"not_an_operator": 10}},
        **OPERATOR_SEARCH_KWARGS,
    )
    assert batch == [[], []]
