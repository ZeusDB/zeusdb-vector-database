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
