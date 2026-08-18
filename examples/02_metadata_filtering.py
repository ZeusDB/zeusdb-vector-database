"""Filter a catalogue of 12,000 products, and see what filtering costs.

A filter decides which records the search ranks. Ask for ten results with a
filter matching two hundred products and you get the ten nearest of those two
hundred, not whatever survives of the ten nearest products overall. This file
checks that against a brute force ranking, on both of the paths the index uses.

Which path runs depends on how many records match. At or below 5,000 the index
walks every record's metadata, scores the ones that matched and ranks them,
which is exact. Above 5,000 it stops the walk and searches the graph with the
filter applied at every node it reaches, which is fast and very slightly lossy.
Both paths are shown below with their agreement against brute force.

It also covers the three ways a filter surprises people, which are a record that
lacks the field, a nested object written without `eq`, and an operator name the
index does not recognise.

Run it with:

    python 02_metadata_filtering.py
"""

import numpy as np

from zeusdb_vector_database import VectorDatabase

DIM = 16
CATALOGUE_SIZE = 12000
FULL_SCAN_THRESHOLD = 5000
CATEGORIES = ["audio", "camera", "laptop", "monitor", "phone", "printer", "tablet", "watch"]
BRANDS = ["acme", "borealis", "cinder", "dovetail"]


def build_catalogue():
    """12,000 products in eight categories, each with a vector and metadata."""
    rng = np.random.default_rng(20260806)

    # Ten clusters of vectors, so that "nearest" means something. Real
    # applications put an embedding of the product text here instead.
    centres = rng.standard_normal((10, DIM))
    vectors = centres[rng.integers(0, 10, CATALOGUE_SIZE)]
    vectors += 0.35 * rng.standard_normal((CATALOGUE_SIZE, DIM))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)

    metadata = []
    for i in range(CATALOGUE_SIZE):
        metadata.append(
            {
                "category": CATEGORIES[i % len(CATEGORIES)],
                "brand": BRANDS[i % len(BRANDS)],
                "price": round(5.0 + (i % 400) * 1.25, 2),
                "rating": round(1.0 + (i % 9) * 0.5, 1),
                "year": 2019 + (i % 7),
                "in_stock": i % 3 != 0,
                "tags": ["sale"] if i % 5 == 0 else ["standard"],
                "sku": f"SKU-{i:05d}.pdf" if i % 2 == 0 else f"SKU-{i:05d}.txt",
            }
        )

    index = VectorDatabase().create("hnsw", dim=DIM, expected_size=CATALOGUE_SIZE)
    result = index.add(
        {
            "ids": [f"item_{i:05d}" for i in range(CATALOGUE_SIZE)],
            "embeddings": vectors,
            "metadatas": metadata,
        }
    )
    assert result.is_success(), result.errors
    return index, vectors, metadata


def brute_force(vectors, matching, query, k):
    """The k nearest of `matching`, computed exactly. Cosine, so 1 - dot."""
    if not matching:
        return []
    subset = np.array(sorted(matching))
    scores = vectors[subset] @ query
    order = np.argsort(-scores)[:k]
    return [f"item_{subset[i]:05d}" for i in order]


def main():
    index, vectors, metadata = build_catalogue()
    query = vectors[0]

    def hits(filter_, top_k=10):
        return index.search(vector=query, filter=filter_, top_k=top_k)

    def matches(predicate):
        return [i for i, m in enumerate(metadata) if predicate(m)]

    # ------------------------------------------------------------------
    # A selective filter, which the exact path serves
    # ------------------------------------------------------------------
    selective = {"category": "watch", "in_stock": True, "tags": {"contains": "sale"}}
    matching = matches(
        lambda m: m["category"] == "watch" and m["in_stock"] and "sale" in m["tags"]
    )
    page = hits(selective)
    print(f"selective filter matches {len(matching)} of {CATALOGUE_SIZE} products")
    print(f"  at or below {FULL_SCAN_THRESHOLD}, so the exact path serves it")
    print(f"  asked for 10, got {len(page)}")
    exact = brute_force(vectors, matching, query, 10)
    print(f"  page equals the brute force ranking: {[h['id'] for h in page] == exact}")
    print()

    # top_k is the page size and nothing more. It does not widen or narrow what
    # the filter admits, so the page is min(top_k, matching) every time.
    print("results returned, by top_k")
    for top_k in (1, 5, 10, 50, 200, 1000):
        print(f"  top_k={top_k:<5d} -> {len(hits(selective, top_k)):>3d}")
    print()

    # ------------------------------------------------------------------
    # A broad filter, which the graph path serves
    # ------------------------------------------------------------------
    broad = {"in_stock": True}
    matching = matches(lambda m: m["in_stock"])
    page = hits(broad)
    exact = brute_force(vectors, matching, query, 10)
    overlap = len(set(h["id"] for h in page) & set(exact))
    print(f"broad filter matches {len(matching)} of {CATALOGUE_SIZE} products")
    print(f"  above {FULL_SCAN_THRESHOLD}, so the graph path serves it")
    print(f"  asked for 10, got {len(page)}, of which {overlap} are the true nearest 10")
    print()

    # ------------------------------------------------------------------
    # Fewer matches than the page, and none at all
    # ------------------------------------------------------------------
    few = hits({"sku": {"endswith": "00007.txt"}})
    print(f"a filter matching one product returns {len(few)}: {[h['id'] for h in few]}")
    none = hits({"category": "submarine"})
    print(f"a filter matching nothing returns {len(none)}, which is not an error")
    print()

    # ------------------------------------------------------------------
    # The operators, on one question
    # ------------------------------------------------------------------
    # A field maps either to a plain value, which means equality, or to a dict
    # of operators, all of which must hold.
    shopping = {
        "in_stock": True,
        "rating": {"gte": 4.0},
        "price": {"lt": 120.0},
        "year": {"gte": 2023},
        "brand": {"in": ["acme", "cinder"]},
        "tags": {"contains": "sale"},
        "sku": {"endswith": ".pdf"},
    }
    found = hits(shopping, top_k=5)
    print(f"all seven conditions together, top 5 of the matching products")
    for hit in found:
        meta = hit["metadata"]
        print(
            f"  {hit['id']}  {meta['brand']:<9s} {meta['rating']}  "
            f"{meta['price']:>7.2f}  {meta['year']}"
        )
    print()

    # ------------------------------------------------------------------
    # Three behaviours that surprise people
    # ------------------------------------------------------------------
    # A record that lacks the field never matches, whatever the operator. That
    # includes `ne`, so "not equal to acme" does not find records with no brand.
    index.add(
        {
            "id": "item_no_brand",
            "values": query.tolist(),
            "metadata": {"category": "watch"},
        }
    )
    absent = [h["id"] for h in hits({"brand": {"ne": "acme"}}, top_k=5)]
    print("item_no_brand has no brand at all, so brand != acme excludes it:",
          "item_no_brand" not in absent)
    print()

    # A dict value is always read as operators, so equality against a nested
    # object has to be spelled with `eq`.
    index.add(
        {
            "id": "item_nested",
            "values": query.tolist(),
            "metadata": {"source": {"kind": "web", "trusted": True}},
        }
    )
    ok = [h["id"] for h in hits({"source": {"eq": {"kind": "web", "trusted": True}}}, top_k=5)]
    print("nested object matched with eq:", ok)
    try:
        hits({"source": {"kind": "web"}})
    except ValueError as exc:
        print("written without eq:", exc)
    print()

    # An unrecognised operator raises before the search runs, rather than
    # quietly matching nothing. Both rejections above are also logged at ERROR
    # level, which is what the two lines on stderr are.
    try:
        hits({"price": {"less_than": 50}})
    except ValueError as exc:
        print("unknown operator:", exc)


# The transcript this file prints.
EXPECTED_OUTPUT = """\
selective filter matches 200 of 12000 products
  at or below 5000, so the exact path serves it
  asked for 10, got 10
  page equals the brute force ranking: True

results returned, by top_k
  top_k=1     ->   1
  top_k=5     ->   5
  top_k=10    ->  10
  top_k=50    ->  50
  top_k=200   -> 200
  top_k=1000  -> 200

broad filter matches 8000 of 12000 products
  above 5000, so the graph path serves it
  asked for 10, got 10, of which 10 are the true nearest 10

a filter matching one product returns 1: ['item_00007']
a filter matching nothing returns 0, which is not an error

all seven conditions together, top 5 of the matching products
  item_09610  cinder    4.5    17.50  2025
  item_08000  acme      5.0     5.00  2025
  item_08810  cinder    5.0    17.50  2023
  item_08440  acme      4.5    55.00  2024
  item_01600  acme      4.5     5.00  2023

item_no_brand has no brand at all, so brand != acme excludes it: True

nested object matched with eq: ['item_nested']
written without eq: Unknown filter operation: kind

unknown operator: Unknown filter operation: less_than
"""

if __name__ == "__main__":
    main()
