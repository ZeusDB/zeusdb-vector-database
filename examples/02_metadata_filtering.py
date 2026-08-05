"""Filter a catalogue of 4,000 products, and see where filtering goes wrong.

The filter is applied after the graph search, not during it. The index finds the
top_k nearest vectors and then throws away the ones the filter rejects, so a
selective filter over a large index needs a top_k far larger than the number of
results you want. This file measures how much larger.

It also covers the three ways a filter fails quietly or loudly, which are a
record that lacks the field, a nested object written without `eq`, and an
operator name the index does not recognise.

Run it with:

    python 02_metadata_filtering.py
"""

import numpy as np

from zeusdb_vector_database import VectorDatabase

DIM = 16
CATALOGUE_SIZE = 4000
CATEGORIES = ["audio", "camera", "laptop", "monitor", "phone", "printer", "tablet", "watch"]
BRANDS = ["acme", "borealis", "cinder", "dovetail"]


def build_catalogue():
    """4,000 products in eight categories, each with a vector and metadata."""
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
    return index, vectors[0]


def main():
    index, query = build_catalogue()

    def hits(filter_, top_k=10):
        return index.search(vector=query, filter=filter_, top_k=top_k)

    # ------------------------------------------------------------------
    # A selective filter needs a large top_k
    # ------------------------------------------------------------------
    # One product in 60 is a discounted watch that is in stock, so a top_k of 10
    # almost never contains one.
    selective = {"category": "watch", "in_stock": True, "tags": {"contains": "sale"}}
    matching = sum(
        1
        for i in range(CATALOGUE_SIZE)
        if CATEGORIES[i % 8] == "watch" and i % 3 != 0 and i % 5 == 0
    )
    print(f"products in the catalogue matching the filter: {matching} of {CATALOGUE_SIZE}")
    print("matches returned, by top_k")
    for top_k in (10, 50, 200, 1000, 4000):
        print(f"  top_k={top_k:<5d} -> {len(hits(selective, top_k)):>2d} of {matching}")
    print()
    print("Ask for a page of 10 and you get none. The filter never widens the")
    print("search, it only removes from what the graph already returned.")
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
    found = hits(shopping, top_k=CATALOGUE_SIZE)
    print(f"all seven conditions together: {len(found)} results")
    for hit in found[:3]:
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
    print("top_k=5 with brand != acme kept:", absent)
    print("item_no_brand has no brand at all, so it is absent:", "item_no_brand" not in absent)
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
products in the catalogue matching the filter: 66 of 4000
matches returned, by top_k
  top_k=10    ->  0 of 66
  top_k=50    ->  1 of 66
  top_k=200   ->  4 of 66
  top_k=1000  -> 14 of 66
  top_k=4000  -> 66 of 66

Ask for a page of 10 and you get none. The filter never widens the
search, it only removes from what the graph already returned.

all seven conditions together: 8 results
  item_01600  acme      4.5     5.00  2023
  item_02420  acme      5.0    30.00  2024
  item_02050  cinder    4.5    67.50  2025

top_k=5 with brand != acme kept: ['item_03770', 'item_02106', 'item_02779']
item_no_brand has no brand at all, so it is absent: True

nested object matched with eq: ['item_nested']
written without eq: Unknown filter operation: kind

unknown operator: Unknown filter operation: less_than
"""

if __name__ == "__main__":
    main()
