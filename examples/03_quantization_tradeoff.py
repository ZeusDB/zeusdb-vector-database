"""Measure what product quantization costs you, on your own machine.

Quantization is a memory decision that costs accuracy. The size of that cost is
much larger than most people expect, and one of the two storage modes cannot
recover it at all. This file builds the same 3,000 vectors three ways, measures
the memory each holds and the recall each returns against exact search, and
prints the comparison.

The short version, which the numbers below reproduce every run. Use
`quantized_with_raw` and leave rerank on, or do not quantize.

Run it with:

    python 03_quantization_tradeoff.py
"""

import warnings

import numpy as np

from zeusdb_vector_database import VectorDatabase

DIM = 64
RECORDS = 3000
QUERIES = 200
TOP_K = 10

# subvectors sets the code length, at one byte per subvector per record, so the
# compression ratio is dim * 4 / subvectors. Here that is 32x.
# training_size is the record count that triggers training. 1,000 is the minimum
# the validator accepts and it keeps this file quick; production indexes use far
# more.
PQ = {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000}


def dataset():
    """Clustered vectors and queries drawn near them, so neighbours are real."""
    rng = np.random.default_rng(20260806)
    centres = rng.standard_normal((15, DIM))
    vectors = centres[rng.integers(0, 15, RECORDS)] + 0.3 * rng.standard_normal((RECORDS, DIM))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)

    picks = rng.choice(RECORDS, QUERIES, replace=False)
    queries = vectors[picks] + 0.08 * rng.standard_normal((QUERIES, DIM))
    queries = (queries / np.linalg.norm(queries, axis=1, keepdims=True)).astype(np.float32)
    return vectors, queries


def exact_neighbours(vectors, queries, ids):
    """Brute force cosine ranking, which is the answer the index is graded against."""
    similarity = queries @ vectors.T
    return [{ids[j] for j in np.argsort(-row)[:TOP_K]} for row in similarity]


def build(vectors, ids, storage_mode):
    """Create an index, optionally quantized, and load every record into it."""
    config = None
    if storage_mode is not None:
        config = dict(PQ, storage_mode=storage_mode)

    # create() warns when a configuration looks unbalanced, and quantized_with_raw
    # always warns because it holds raw vectors as well as codes. The warnings are
    # silenced here so that the measurements below are the only output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create(
            "hnsw", dim=DIM, expected_size=RECORDS, quantization_config=config
        )

    # Training fires automatically on the add() that reaches training_size. That
    # one call runs k-means and then rebuilds the graph from the codes, so it
    # takes noticeably longer than the others.
    assert index.add({"ids": ids, "embeddings": vectors}).is_success()
    return index


def recall(index, queries, truth, **search_kwargs):
    """Share of the true nearest neighbours the index actually returned."""
    pages = index.search(queries, top_k=TOP_K, **search_kwargs)
    found = sum(len({hit["id"] for hit in page} & truth[i]) for i, page in enumerate(pages))
    return found / (TOP_K * QUERIES)


def megabytes(stats, key):
    return float(stats.get(key, "0"))


def verdict(measured):
    """A stable word for a number that moves.

    Product quantization trains with an unseeded k-means, so a quantized index
    is not reproducible and every recall figure below shifts by a few hundredths
    from run to run. The verdict does not, which is the part worth reading.
    """
    return "good" if measured >= 0.90 else "poor"


def main():
    vectors, queries = dataset()
    ids = [f"vec_{i:05d}" for i in range(RECORDS)]
    truth = exact_neighbours(vectors, queries, ids)

    raw = build(vectors, ids, None)
    only = build(vectors, ids, "quantized_only")
    with_raw = build(vectors, ids, "quantized_with_raw")

    print(f"{RECORDS} vectors of {DIM} dimensions, recall@{TOP_K} over {QUERIES} queries")
    print(f"compression ratio: {only.get_quantization_info()['compression_ratio']:.0f}x")
    print()

    # ------------------------------------------------------------------
    # What each mode stores
    # ------------------------------------------------------------------
    # quantized_only keeps the raw vectors of every record collected before
    # training and never discards them, so an index whose training_size is a
    # large share of its total size saves much less than the ratio suggests.
    print("stored per mode")
    print(f"  {'mode':<20s} {'raw':>6s} {'codes':>6s} {'raw MB':>7s} {'code MB':>8s} {'total':>6s}")
    unquantized_mb = RECORDS * DIM * 4 / 1024 / 1024
    print(f"  {'no quantization':<20s} {RECORDS:>6d} {0:>6d} "
          f"{unquantized_mb:>7.2f} {0.0:>8.2f} {unquantized_mb:>6.2f}")
    for label, index in (("quantized_only", only), ("quantized_with_raw", with_raw)):
        stats = index.get_stats()
        raw_mb = megabytes(stats, "raw_vectors_memory_mb")
        code_mb = megabytes(stats, "quantized_codes_memory_mb")
        print(
            f"  {label:<20s} {stats['raw_vectors_stored']:>6s} "
            f"{stats['quantized_codes_stored']:>6s} "
            f"{raw_mb:>7.2f} {code_mb:>8.2f} {raw_mb + code_mb:>6.2f}"
        )
    print()
    print("quantized_only saves far less than the 32x ratio, because a third of")
    print("the records were collected before training and kept their raw vectors.")
    print("quantized_with_raw stores more than an unquantized index, not less.")
    print()

    # ------------------------------------------------------------------
    # What each mode returns
    # ------------------------------------------------------------------
    # rerank over-fetches candidates and rescores them against raw vectors, so it
    # only works where raw vectors survive. On quantized_only it is ignored.
    print("recall@10 against exact cosine search")
    measurements = [
        ("no quantization", recall(raw, queries, truth)),
        ("quantized_only, default rerank", recall(only, queries, truth)),
        ("quantized_only, rerank=50", recall(only, queries, truth, rerank=50)),
        ("quantized_with_raw, rerank=0", recall(with_raw, queries, truth, rerank=0)),
        ("quantized_with_raw, default rerank", recall(with_raw, queries, truth)),
    ]
    for label, measured in measurements:
        print(f"  {label:<34s} {verdict(measured):<5s} {measured:.2f}")
    print()
    print("rerank changes nothing on quantized_only, because the raw vectors it")
    print("would rescore against are gone. That mode cannot be tuned back up.")
    print()

    # ------------------------------------------------------------------
    # A record added after training is stored only as a code
    # ------------------------------------------------------------------
    # Reading it back reconstructs the vector from the code, so what you get is
    # close to what you supplied rather than equal to it. A record collected
    # before training still holds its raw vector and reads back exactly.
    before = only.get_records(ids[0], return_vector=True)[0]["vector"]
    after = only.get_records(ids[-1], return_vector=True)[0]["vector"]
    before_error = np.abs(np.array(before) - vectors[0]).max()
    after_error = np.abs(np.array(after) - vectors[-1]).max()
    print("largest error in a vector read back from quantized_only")
    print(f"  collected before training  {'exact' if before_error < 1e-6 else 'lossy':<5s} "
          f"{before_error:.6f}")
    print(f"  added after training       {'exact' if after_error < 1e-6 else 'lossy':<5s} "
          f"{after_error:.6f}")
    print()

    # With rerank on, a score is a raw vector distance. With rerank=0 it is an
    # ADC estimate off the codes. The two are on different scales and comparing
    # them across a configuration change is meaningless.
    reranked = with_raw.search(queries[0], top_k=1)[0]["score"]
    estimated = with_raw.search(queries[0], top_k=1, rerank=0)[0]["score"]
    print(f"same query, same index, scores on different scales: "
          f"{'differ' if abs(reranked - estimated) > 1e-6 else 'agree'}")
    print(f"  rerank on, a raw vector distance:  {reranked:.4f}")
    print(f"  rerank off, an ADC estimate:       {estimated:.4f}")


# The transcript this file prints. A "..." stands for a figure that moves
# between runs, which here is everything downstream of quantizer training,
# because that trains with an unseeded k-means. The verdict beside each one
# does not move.
EXPECTED_OUTPUT = """\
3000 vectors of 64 dimensions, recall@10 over 200 queries
compression ratio: 32x

stored per mode
  mode                    raw  codes  raw MB  code MB  total
  no quantization        3000      0    0.73     0.00   0.73
  quantized_only         1000   3000    0.24     0.02   0.26
  quantized_with_raw     3000   3000    0.73     0.02   0.75

quantized_only saves far less than the 32x ratio, because a third of
the records were collected before training and kept their raw vectors.
quantized_with_raw stores more than an unquantized index, not less.

recall@10 against exact cosine search
  no quantization                    good  1.00
  quantized_only, default rerank     poor  0...
  quantized_only, rerank=50          poor  0...
  quantized_with_raw, rerank=0       poor  0...
  quantized_with_raw, default rerank good  ...

rerank changes nothing on quantized_only, because the raw vectors it
would rescore against are gone. That mode cannot be tuned back up.

largest error in a vector read back from quantized_only
  collected before training  exact 0.000000
  added after training       lossy 0...

same query, same index, scores on different scales: differ
  rerank on, a raw vector distance:  ...
  rerank off, an ADC estimate:       ...
"""

if __name__ == "__main__":
    main()
