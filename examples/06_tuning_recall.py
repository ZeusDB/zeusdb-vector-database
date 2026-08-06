"""Measure what `m`, `ef_search` and `expected_size` actually do to recall.

These three parameters are the ones people guess at. This file stops guessing.
It builds the same 5,000 vectors at five graph degrees, measures recall against
exact search at three search widths, and prints the grid.

The result that matters is the shape of the grid rather than the numbers. `m` is
fixed when the index is created and sets a ceiling that no amount of `ef_search`
lifts. `ef_search` is per query and trades time for recall underneath that
ceiling. `expected_size` matters mostly because it is what picks `m` when you do
not.

Run it with:

    python 06_tuning_recall.py
"""

import time

import numpy as np

from zeusdb_vector_database import VectorDatabase

DIM = 32
RECORDS = 5000
QUERIES = 100
TOP_K = 10
DEGREES = (2, 4, 8, 16, 32)
WIDTHS = (10, 50, 200)


def unit(vectors):
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def dataset():
    rng = np.random.default_rng(99)
    centres = rng.standard_normal((20, DIM))
    vectors = unit(centres[rng.integers(0, 20, RECORDS)] + 0.4 * rng.standard_normal((RECORDS, DIM)))
    queries = vectors[rng.choice(RECORDS, QUERIES, replace=False)]
    return vectors, queries


def main():
    vectors, queries = dataset()
    ids = [f"vec_{i:05d}" for i in range(RECORDS)]

    # Exact cosine ranking, which is what the index is graded against.
    truth = [{ids[j] for j in np.argsort(-row)[:TOP_K]} for row in queries @ vectors.T]

    def recall(index, ef_search):
        pages = index.search(queries, top_k=TOP_K, ef_search=ef_search)
        found = sum(len({h["id"] for h in page} & truth[i]) for i, page in enumerate(pages))
        return found / (TOP_K * QUERIES)

    # ------------------------------------------------------------------
    # The grid
    # ------------------------------------------------------------------
    # m is the number of bi-directional links each node gets. It is fixed at
    # creation and cannot be changed afterwards, so it is the one parameter here
    # you have to get right up front.
    print(f"recall@{TOP_K} over {QUERIES} queries against {RECORDS} vectors")
    print(" " * 8 + "".join(f"ef_search={width}".ljust(14) for width in WIDTHS).rstrip())
    for degree in DEGREES:
        index = VectorDatabase().create("hnsw", dim=DIM, expected_size=RECORDS, m=degree)
        index.add({"ids": ids, "embeddings": vectors})
        row = "".join(f"{recall(index, width):.3f}".ljust(14) for width in WIDTHS)
        print(f"m={degree}".ljust(8) + row.rstrip())
    print()
    print("Reading down a column, a graph that is too sparse loses recall that no")
    print("search width recovers. m=2 does not reach m=8's worst result even at")
    print("ef_search=200. Reading across a row, ef_search buys back the last few")
    print("percent, and only up to the ceiling m set.")
    print()

    # ------------------------------------------------------------------
    # What ef_search costs
    # ------------------------------------------------------------------
    # ef_search is the size of the candidate list a query keeps as it descends,
    # so search time grows with it. The default is max(2 * top_k, 100) for cosine
    # and max(2 * top_k, 150) for l1 and l2.
    index = VectorDatabase().create("hnsw", dim=DIM, expected_size=RECORDS, m=16)
    index.add({"ids": ids, "embeddings": vectors})
    baseline = None
    print("search time relative to ef_search=10, at m=16")
    for width in (10, 50, 200, 800):
        started = time.perf_counter()
        for _ in range(3):
            index.search(queries, top_k=TOP_K, ef_search=width)
        elapsed = time.perf_counter() - started
        baseline = baseline or elapsed
        print(f"  ef_search={width:<4d} recall {recall(index, width):.3f}   "
              f"{elapsed / baseline:.1f}x the time")
    print()

    # ------------------------------------------------------------------
    # expected_size
    # ------------------------------------------------------------------
    # expected_size preallocates, and it picks the default m. That is 16 up to
    # 25,000 and 32 above it. Passing m explicitly always wins.
    vdb = VectorDatabase()
    print("default m by expected_size")
    for size in (1000, 25_000, 25_001, 1_000_000):
        chosen = vdb.create("hnsw", dim=DIM, expected_size=size).get_stats()["m"]
        print(f"  expected_size={size:<9d} m={chosen}")
    explicit = vdb.create("hnsw", dim=DIM, expected_size=1_000_000, m=4).get_stats()["m"]
    print(f"  passing m=4 alongside expected_size=1000000 wins: m={explicit}")
    print()

    # expected_size is a hint and not a limit. An index accepts more records than
    # it declared and the graph grows to fit them. What does not grow is m, which
    # was already chosen from the declaration, so an index that badly outgrows its
    # declaration is running at a degree meant for a smaller one.
    small = VectorDatabase().create("hnsw", dim=DIM, expected_size=100)
    small.add({"ids": ids[:1000], "embeddings": vectors[:1000]})
    print("declared 100 records, inserted", small.get_vector_count())
    print("m it was given:", small.get_stats()["m"])
    print("Crossing twice the declared size logs a warning once, on stderr.")


# The transcript this file prints. A "..." stands for a figure that moves
# between runs, which here is wall clock timing.
EXPECTED_OUTPUT = """\
recall@10 over 100 queries against 5000 vectors
        ef_search=10  ef_search=50  ef_search=200
m=2     0.425         0.716         0.918
m=4     0.785         0.953         0.999
m=8     0.966         1.000         1.000
m=16    0.987         1.000         1.000
m=32    0.985         1.000         1.000

Reading down a column, a graph that is too sparse loses recall that no
search width recovers. m=2 does not reach m=8's worst result even at
ef_search=200. Reading across a row, ef_search buys back the last few
percent, and only up to the ceiling m set.

search time relative to ef_search=10, at m=16
  ef_search=10   recall 0.987   1.0x the time
  ef_search=50   recall 1.000   ...x the time
  ef_search=200  recall 1.000   ...x the time
  ef_search=800  recall 1.000   ...x the time

default m by expected_size
  expected_size=1000      m=16
  expected_size=25000     m=16
  expected_size=25001     m=32
  expected_size=1000000   m=32
  passing m=4 alongside expected_size=1000000 wins: m=4

declared 100 records, inserted 1000
m it was given: 16
Crossing twice the declared size logs a warning once, on stderr.
"""

if __name__ == "__main__":
    main()
