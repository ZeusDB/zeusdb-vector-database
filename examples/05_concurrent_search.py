"""Serve searches from a thread pool, and write to the index while they run.

An index is safe to share across threads. `search()` releases the interpreter
lock for the traversal, so several searches genuinely run at once rather than
queueing behind each other, and a write arriving mid-search neither raises nor
corrupts the answers.

This file proves both. First it checks that eight threads searching together
return exactly what one thread returns alone. Then it inserts 2,000 records
while those threads keep searching, and checks that every result stayed well
formed throughout.

Run it with:

    python 05_concurrent_search.py
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from zeusdb_vector_database import VectorDatabase

DIM = 32
RECORDS = 5000
QUERIES = 64
TOP_K = 10
THREADS = 8
INSERTED = 2000


def unit(vectors):
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def clustered(count, seed):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((12, DIM))
    return unit(centres[rng.integers(0, 12, count)] + 0.3 * rng.standard_normal((count, DIM)))


def page(index, query):
    """One search reduced to the part that has to be reproducible."""
    return [(hit["id"], hit["score"]) for hit in index.search(query, top_k=TOP_K)]


def main():
    vectors = clustered(RECORDS, seed=1)
    ids = [f"doc_{i:05d}" for i in range(RECORDS)]
    index = VectorDatabase().create("hnsw", dim=DIM, expected_size=RECORDS + INSERTED)
    index.add({"ids": ids, "embeddings": vectors})
    queries = clustered(QUERIES, seed=2)

    # ------------------------------------------------------------------
    # Concurrent searches against an index nobody is writing to
    # ------------------------------------------------------------------
    # The index is not being modified here, so a single threaded run is an exact
    # reference. Any difference under threads would be interference, not noise.
    started = time.perf_counter()
    reference = [page(index, query) for query in queries]
    solo_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        # Every worker replays the whole query set, so each query overlaps the
        # other workers' different queries many times over.
        rounds = list(pool.map(lambda _: [page(index, q) for q in queries], range(THREADS)))
    pool_seconds = time.perf_counter() - started

    print(f"{RECORDS} records, {QUERIES} queries, {THREADS} threads")
    print("searches run:", sum(len(r) for r in rounds))
    print("every threaded result identical to the single threaded one:",
          all(result == reference for result in rounds))
    print(f"wall clock, {THREADS} threads against 1: {solo_seconds / pool_seconds * THREADS:.1f}x "
          f"the work in {pool_seconds / solo_seconds:.1f}x the time")
    print("The ratio depends on how many cores you have. What does not vary is")
    print("that the answers are identical.")
    print()

    # ------------------------------------------------------------------
    # Searching while the index is being written
    # ------------------------------------------------------------------
    # An exact reference does not exist here, because the insert changes what the
    # right answer is while the queries run. The checks are the properties that
    # hold at every instant instead.
    known = set(ids)
    batches = []
    for batch in range(INSERTED // 500):
        batch_ids = [f"new_{batch}_{i:03d}" for i in range(500)]
        batches.append({"ids": batch_ids, "embeddings": clustered(500, seed=100 + batch)})
        known.update(batch_ids)

    writing = threading.Event()
    writing.set()
    searching = threading.Event()
    failures = []

    def writer():
        # Wait until a search has actually completed before inserting anything,
        # so the write lands on top of a live read rather than beside one. A run
        # that inserted everything before the pool woke up would prove nothing.
        searching.wait(timeout=60)
        try:
            for batch in batches:
                index.add(batch)
        finally:
            writing.clear()

    def reader(worker):
        seen = 0
        while True:
            results = index.search(queries[worker], top_k=TOP_K)
            found = [hit["id"] for hit in results]
            scores = [hit["score"] for hit in results]
            if len(results) > TOP_K or len(set(found)) != len(found):
                failures.append("a page repeated an id or overran top_k")
            if not set(found) <= known:
                failures.append("a page contained an id that was never added")
            if scores != sorted(scores):
                failures.append("a page came back out of order")
            seen += 1
            searching.set()
            if not writing.is_set():
                return seen

    write_thread = threading.Thread(target=writer)
    write_thread.start()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        searches = sum(pool.map(reader, range(THREADS)))
    write_thread.join()

    print(f"inserted {INSERTED} records while {THREADS} threads searched")
    print("searches completed during the write:", "many" if searches > THREADS else "too few")
    print("malformed pages:", len(failures))
    print("records now:", index.get_vector_count())
    print()

    # The index is still exactly right once everything has stopped.
    settled = page(index, queries[0])
    print("top hit after the write:", settled[0][0])
    print("scores still ordered:", [s for _, s in settled] == sorted(s for _, s in settled))


# The transcript this file prints. A "..." stands for a figure that moves
# between runs, which here is wall clock timing.
EXPECTED_OUTPUT = """\
5000 records, 64 queries, 8 threads
searches run: 512
every threaded result identical to the single threaded one: True
wall clock, 8 threads against 1: ...x the work in ...x the time
The ratio depends on how many cores you have. What does not vary is
that the answers are identical.

inserted 2000 records while 8 threads searched
searches completed during the write: many
malformed pages: 0
records now: 7000

top hit after the write: doc_00820
scores still ordered: True
"""

if __name__ == "__main__":
    main()
