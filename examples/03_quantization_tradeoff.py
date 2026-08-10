"""Measure what product quantization costs you, on your own machine.

Quantization is a memory decision that costs accuracy, and above roughly 10,000
records it costs query time as well. The size of the accuracy cost is much
larger than most people expect, and one of the two storage modes cannot recover
it at all. This file builds the same 3,000 vectors three ways, measures the
memory each holds and the recall each returns against exact search, and prints
the comparison.

At 3,000 records the reranked quantized search is still faster than an
unquantized one. It stops being faster somewhere above that, because the rerank
fetch grows with the corpus and the graph traversal has to be as wide as the
fetch. Where it stops depends on your data. On clustered vectors of dimension
768 it is between 10,000 and 15,000 records, and on real OpenAI embeddings it is
near 11,000. See the README for the measured tables.

How deep the fetch has to go is measured on your own data when training
completes, not set by a formula, so `get_stats()["rerank_default_fetch"]` tells
you what each search is paying for at a page of ten. The measurement runs over a
seeded random draw of the training sample rather than the order your records
arrived in, and it is repeated at a quarter, a half and three quarters of that
sample so the fetch can be scaled to a corpus larger than the one it was
measured on. It is also taken at pages of 1, 10 and 100, because the hundredth
true neighbour sits deeper than the tenth, and a search interpolates between
those three for whatever page you ask for. This file runs at dimension 64, where
`create()` warns that quantization removes only a few percent of what an
unquantized index holds. That warning is the point of the file rather than a
problem with it.

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
    # quantized_only holds codes alone once training completes. The records
    # collected for training are held at full width only until that moment,
    # then released, so a trained index in this mode has no raw vector left.
    # The codebook and the centroid distance table are fixed by dim, subvectors
    # and bits. They do not move as records arrive, and at this record count
    # they are larger than everything the records themselves hold, which is why
    # they are in the total rather than left out of it.
    # get_stats() reports the codes, the raw vectors, the codebook, the table
    # and the graph, and total_memory_mb is the sum of those five. The graph
    # owns a second copy of every point on top of its neighbour lists, and that
    # copy is dim * 4 bytes in an unquantized index and subvectors bytes in a
    # quantized one, in both storage modes. It used to be left out of this
    # table, which is what made an earlier version of this file claim that
    # quantized_with_raw stores more than an unquantized index. It does not,
    # once the graph is counted and the record count clears the fixed cost.
    print("stored per mode")
    print(f"  {'mode':<20s} {'raw':>6s} {'codes':>6s} {'raw MB':>7s} {'code MB':>8s} "
          f"{'fixed MB':>9s} {'graph MB':>9s} {'total':>6s}")
    for label, index in (("no quantization", raw),
                         ("quantized_only", only),
                         ("quantized_with_raw", with_raw)):
        stats = index.get_stats()
        raw_mb = megabytes(stats, "raw_vectors_memory_mb")
        code_mb = megabytes(stats, "quantized_codes_memory_mb")
        fixed_mb = (megabytes(stats, "codebook_memory_mb")
                    + megabytes(stats, "sdc_table_memory_mb"))
        graph_mb = megabytes(stats, "graph_memory_mb")
        print(
            f"  {label:<20s} {stats['raw_vectors_stored']:>6s} "
            f"{stats['quantized_codes_stored']:>6s} "
            f"{raw_mb:>7.2f} {code_mb:>8.2f} {fixed_mb:>9.2f} {graph_mb:>9.2f} "
            f"{megabytes(stats, 'total_memory_mb'):>6.2f}"
        )
    print()
    print("Both modes drop the graph's full width copy of every point, so both")
    print("save there. The graph column holds more than that copy. The neighbour")
    print("lists, the sixteen layer headers every point carries and the counters")
    print("around them do not shrink when the copy does, so the quantized graph")
    print("is smaller rather than negligible. The fixed table is 1.06 MB at this")
    print("configuration and neither mode has repaid it at 3,000 records. The")
    print("fixed cost does not grow with the record count and the saving does, so")
    print("both cross into saving as the index grows, quantized_only first")
    print("because it drops the raw vectors as well.")
    print()
    print("total_memory_mb is what the index holds in the structures get_stats()")
    print("can price. The id maps, the metadata map and the allocator's own")
    print("headers sit outside it, so a process holds rather more than this.")
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
    # Every record is stored only as a code once training completes
    # ------------------------------------------------------------------
    # Reading any record back reconstructs its vector from the code, the
    # training records included, since their raw copies are released the moment
    # training completes. What you get is close to what you supplied rather
    # than equal to it. Only quantized_with_raw reads back exactly.
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
  mode                    raw  codes  raw MB  code MB  fixed MB  graph MB  total
  no quantization        3000      0    0.73     0.00      0.00      5.14   5.87
  quantized_only            0   3000    0.00     0.02      1.06      4...   5...
  quantized_with_raw     3000   3000    0.73     0.02      1.06      4...   6...

Both modes drop the graph's full width copy of every point, so both
save there. The graph column holds more than that copy. The neighbour
lists, the sixteen layer headers every point carries and the counters
around them do not shrink when the copy does, so the quantized graph
is smaller rather than negligible. The fixed table is 1.06 MB at this
configuration and neither mode has repaid it at 3,000 records. The
fixed cost does not grow with the record count and the saving does, so
both cross into saving as the index grows, quantized_only first
because it drops the raw vectors as well.

total_memory_mb is what the index holds in the structures get_stats()
can price. The id maps, the metadata map and the allocator's own
headers sit outside it, so a process holds rather more than this.

recall@10 against exact cosine search
  no quantization                    good  1.00
  quantized_only, default rerank     poor  0...
  quantized_only, rerank=50          poor  0...
  quantized_with_raw, rerank=0       poor  0...
  quantized_with_raw, default rerank good  ...

rerank changes nothing on quantized_only, because the raw vectors it
would rescore against are gone. That mode cannot be tuned back up.

largest error in a vector read back from quantized_only
  collected before training  lossy 0...
  added after training       lossy 0...

same query, same index, scores on different scales: differ
  rerank on, a raw vector distance:  ...
  rerank off, an ADC estimate:       ...
"""

if __name__ == "__main__":
    main()
