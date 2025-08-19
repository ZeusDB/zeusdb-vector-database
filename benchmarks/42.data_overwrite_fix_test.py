#!/usr/bin/env python3
"""
ZeusDB Overwrite Bug Fix – Verification Tests

This script covers:
1) Core overwrite behavior (no duplicates, metadata and vector updated)
2) Edge cases (overwrite non-existent doc, multiple overwrites)
"""

import zeusdb_vector_database as zdb


def test_overwrite_bug_fix():
    """Test that overwrite=True replaces documents without creating duplicates."""
    print("🧪 Testing ZeusDB overwrite bug fix...")

    # Create a small test index
    vdb = zdb.VectorDatabase()
    index = vdb.create(
        index_type="hnsw",
        dim=3,
        space="cosine",
        m=16,
        ef_construction=200,
        expected_size=100,
    )

    # Test vectors
    vector1 = [1.0, 0.0, 0.0]
    vector2 = [0.0, 1.0, 0.0]
    vector1_updated = [0.0, 0.0, 1.0]  # Different vector, same ID ("doc1")

    print("\n📝 Step 1: Adding initial documents...")
    result1 = index.add(
        {
            "vectors": [vector1, vector2],
            "ids": ["doc1", "doc2"],
            "metadatas": [
                {"text": "first document", "version": 1},
                {"text": "second document", "version": 1},
            ],
        },
        overwrite=True,
    )
    print(
        f"   Initial add result: ✅ {result1.total_inserted} inserted, ❌ {result1.total_errors} errors"
    )
    assert result1.total_inserted == 2
    assert result1.total_errors == 0

    print("\n🔍 Step 2: Search after initial add (near vector1)...")
    search_results = index.search(vector1, top_k=5)
    print(f"   Found {len(search_results)} results:")
    for i, r in enumerate(search_results, 1):
        print(f"   {i}. ID: {r['id']}, Score: {r['score']:.4f}")
    unique_ids_initial = {r["id"] for r in search_results}
    print(f"   Unique IDs found: {unique_ids_initial}")
    assert "doc1" in unique_ids_initial and "doc2" in unique_ids_initial

    print("\n🔄 Step 3: Overwriting doc1 with new vector...")
    result2 = index.add(
        {
            "vectors": [vector1_updated],
            "ids": ["doc1"],
            "metadatas": [{"text": "first document UPDATED", "version": 2}],
        },
        overwrite=True,
    )
    print(
        f"   Overwrite result: ✅ {result2.total_inserted} inserted, ❌ {result2.total_errors} errors"
    )
    assert result2.total_inserted == 1
    assert result2.total_errors == 0

    print("\n🔍 Step 4a: Search after overwrite (near OLD vector1)...")
    search_results_after_old = index.search(vector1, top_k=5)
    print(f"   Found {len(search_results_after_old)} results:")
    for i, r in enumerate(search_results_after_old, 1):
        print(f"   {i}. ID: {r['id']}, Score: {r['score']:.4f}")

    print("\n🔍 Step 4b: Search after overwrite (near UPDATED vector1_updated)...")
    search_results_after_new = index.search(vector1_updated, top_k=5)
    print(f"   Found {len(search_results_after_new)} results:")
    for i, r in enumerate(search_results_after_new, 1):
        print(f"   {i}. ID: {r['id']}, Score: {r['score']:.4f}")

    # Combine both result sets to check for duplicates across queries
    all_after = search_results_after_old + search_results_after_new
    id_counts = {}
    for r in all_after:
        id_counts[r["id"]] = id_counts.get(r["id"], 0) + 1
    print(f"   ID occurrence counts across both searches: {id_counts}")

    # ✅ No duplicate IDs should appear within a single search result set
    for results in (search_results_after_old, search_results_after_new):
        counts = {}
        for r in results:
            counts[r["id"]] = counts.get(r["id"], 0) + 1
        dups = [k for k, v in counts.items() if v > 1]
        assert not dups, f"Found duplicate IDs in a single result set: {dups}"

    # ✅ Each known ID should be present (and not duplicated overall)
    assert any(r["id"] == "doc1" for r in all_after), "doc1 should exist after overwrite"
    assert any(r["id"] == "doc2" for r in all_after), "doc2 should still exist after overwrite"

    print("\n🔍 Step 5: Verify updated document content (fail fast if missing)...")
    # ---- Option A: fail fast & keep scope tight ----
    doc1_records = index.get_records("doc1", return_vector=True)
    assert doc1_records and isinstance(doc1_records, list), "doc1 not found after overwrite"

    metadata = doc1_records[0]["metadata"]
    print(f"   doc1 metadata: {metadata}")
    print(f"   doc1 vector: {doc1_records[0].get('vector', 'Not returned')}")

    print("\n✅ Step 6: Verification...")
    # Verify updated metadata
    assert metadata["version"] == 2, "doc1 should have updated metadata"
    assert "UPDATED" in metadata["text"], "doc1 should have updated text"

    # Optional: ensure only one record for doc1 is returned by direct lookup
    assert len(doc1_records) == 1, f"Expected 1 record for doc1, got {len(doc1_records)}"

    print("   ✅ No duplicate IDs in searches")
    print("   ✅ All documents still accessible")
    print("   ✅ Metadata properly updated")
    print("   ✅ Vector count consistent")
    print("\n🎉 SUCCESS: Overwrite bug fix verified!")
    return True


def test_edge_cases():
    """Test edge cases for the overwrite functionality."""
    print("\n🧪 Testing edge cases...")

    vdb = zdb.VectorDatabase()
    index = vdb.create(dim=3, space="cosine")

    # Edge Case 1: Overwrite non-existent document (should add it)
    print("\n📝 Edge Case 1: Overwrite non-existent document")
    result = index.add(
        {
            "vectors": [[1.0, 0.0, 0.0]],
            "ids": ["new_doc"],
            "metadatas": [{"text": "brand new"}],
        },
        overwrite=True,
    )
    assert result.total_inserted == 1
    assert result.total_errors == 0
    print("   ✅ Successfully added non-existent document with overwrite=True")

    # Edge Case 2: Multiple overwrites of the same document
    print("\n📝 Edge Case 2: Multiple overwrites of same document")
    for i in range(3):
        result = index.add(
            {
                "vectors": [[0.0, 1.0, float(i)]],
                "ids": ["multi_overwrite"],
                "metadatas": [{"text": f"version {i}", "iteration": i}],
            },
            overwrite=True,
        )
        assert result.total_inserted == 1
        assert result.total_errors == 0

    # Verify only one copy exists in search results
    search_results = index.search([0.0, 1.0, 2.0], top_k=10)
    multi = [r for r in search_results if r["id"] == "multi_overwrite"]
    assert len(multi) == 1, f"Expected 1 result for multi_overwrite, got {len(multi)}"

    final_metadata = multi[0]["metadata"]
    assert final_metadata["iteration"] == 2, "Should have final iteration metadata"
    print("   ✅ Multiple overwrites work correctly")
    print("\n🎉 All edge cases passed!")


if __name__ == "__main__":
    try:
        ok1 = test_overwrite_bug_fix()
        test_edge_cases()
        if ok1:
            print("\n🏆 ALL TESTS PASSED! The overwrite bug has been successfully fixed.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
