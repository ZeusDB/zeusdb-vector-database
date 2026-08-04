"""Concurrent search, and search running alongside a write.

Two separate defects sat on this path. Search held an exclusive lock on the
graph, so two searches never ran at once. Above that, the mutating methods took
`&mut self`, which PyO3 enforces as an exclusive borrow of the whole object, so a
write arriving while a search held its shared borrow across `allow_threads`
raised `RuntimeError: Already borrowed` before any lock was reached. These tests
hold the line on both, which is that concurrent searches return exactly what the
same queries return alone and that a write in flight never makes a search raise.

Every test here is built to fail for a reason rather than on timing.

Where the index is frozen, single threaded search over it is deterministic, so
the reference results are exact and any difference under concurrency is real
interference rather than scheduling noise. Where the index is being written, an
exact reference does not exist, so the assertions drop to invariants that hold at
every instant of the write instead.

Overlap is measured rather than assumed. Readers maintain an in flight counter
around each search, the writer samples it as it enters, and the test asserts
afterwards that the write genuinely landed on top of a read. A run that failed to
construct the overlap fails the test rather than passing on an empty exercise.
"""

import threading
import time

import numpy as np
import pytest

from zeusdb_vector_database import VectorDatabase

DIM = 32
N = 2000
TOP_K = 10
QUERIES = 24

# A trained product quantizer needs at least this many records.
PQ_DIM = 32
PQ_N = 2000
PQ_CONFIG = {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000}

# Wide enough to contend on a developer machine, small enough to stay quick.
THREADS = 8
ROUNDS = 6

# A batch search holds its shared borrow of the index for the whole batch, which
# is what gives a writer a window wide enough to land in reliably.
READ_BATCH = 128

# How many times a write must be observed starting on top of a live read before
# the overlap counts as constructed.
MIN_OVERLAPS = 5
OVERLAP_TIMEOUT_S = 60.0


def unit(vectors):
    """Cosine is the default space and the index normalises on the way in."""
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def clustered(n, dim, seed):
    """Ten Gaussian clusters, so neighbourhoods are real rather than uniform noise."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((10, dim))
    points = centres[rng.integers(0, 10, size=n)] + 0.15 * rng.standard_normal((n, dim))
    return unit(points)


def build_raw(n=N, dim=DIM, seed=11):
    vectors = clustered(n, dim, seed)
    index = VectorDatabase().create("hnsw", dim=dim, expected_size=n * 8)
    result = index.add({"ids": [f"doc_{i}" for i in range(n)], "embeddings": vectors})
    assert result.is_success()
    return index, [f"doc_{i}" for i in range(n)]


def build_quantized(seed=12):
    vectors = clustered(PQ_N, PQ_DIM, seed)
    index = VectorDatabase().create(
        "hnsw",
        dim=PQ_DIM,
        expected_size=PQ_N * 8,
        quantization_config=PQ_CONFIG,
    )
    result = index.add({"ids": [f"doc_{i}" for i in range(PQ_N)], "embeddings": vectors})
    assert result.is_success()
    assert index.is_quantized(), "training must complete or these tests mean nothing"
    return index, [f"doc_{i}" for i in range(PQ_N)]


def query_set(dim, count, seed):
    return clustered(count, dim, seed)


def hits(index, query, top_k=TOP_K):
    """A search result reduced to the part that must be reproducible."""
    return [(hit["id"], hit["score"]) for hit in index.search(query, top_k=top_k)]


def run_on_threads(work, threads=THREADS):
    """Run `work(worker_index)` on every thread at once and collect the results.

    The barrier starts the threads together rather than in creation order, which
    is what puts the searches on top of each other instead of beside each other.
    An exception on any worker is re-raised on the calling thread, so a
    `RuntimeError` inside a worker fails the test rather than being printed by
    the threading module and otherwise ignored.
    """
    barrier = threading.Barrier(threads)
    results = [None] * threads
    errors = [None] * threads

    def target(index):
        try:
            barrier.wait()
            results[index] = work(index)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors[index] = exc
            barrier.abort()

    workers = [threading.Thread(target=target, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    for error in errors:
        if error is not None:
            raise error
    return results


class ReadLoad:
    """Background threads searching continuously, with a count of reads in flight.

    A reader raises its slot of `in_flight` immediately before entering `search`
    and lowers it immediately after. A writer that samples a non zero total as it
    starts is therefore starting on top of a read, which is exactly the pairing
    that used to raise. Reads run a batch so the shared borrow is held for the
    whole batch rather than for one traversal.
    """

    def __init__(self, index, batch, threads=4):
        self.index = index
        self.batch = batch
        self.reads = 0
        self.errors = []
        self._in_flight = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._run) for _ in range(threads)]

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                self._in_flight += 1
            try:
                self.index.search(self.batch, top_k=TOP_K)
            except BaseException as exc:  # noqa: BLE001 - surfaced by check()
                self.errors.append(exc)
                return
            finally:
                with self._lock:
                    self._in_flight -= 1
                    self.reads += 1

    def in_flight(self):
        with self._lock:
            return self._in_flight

    def __enter__(self):
        for thread in self._threads:
            thread.start()
        # Do not proceed until the readers are actually running, so the first
        # write is not raced against an empty pool.
        deadline = time.monotonic() + 30.0
        while self.reads == 0 and not self.errors and time.monotonic() < deadline:
            time.sleep(0.001)
        return self

    def __exit__(self, *_):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=120)
        return False

    def check(self):
        if self.errors:
            raise self.errors[0]


def drive_writes(load, write, min_overlaps=MIN_OVERLAPS):
    """Call `write` repeatedly until enough calls have started on top of a read.

    Returns the number of writes that began while at least one search was in
    flight. The caller asserts on it, so a run that could not construct the
    overlap fails rather than reporting a pass it did not earn.
    """
    overlaps = 0
    attempts = 0
    deadline = time.monotonic() + OVERLAP_TIMEOUT_S
    while overlaps < min_overlaps and time.monotonic() < deadline:
        overlapping = load.in_flight() > 0
        write(attempts)
        attempts += 1
        if overlapping:
            overlaps += 1
        load.check()
    return overlaps


# ------------------------------------------------------------
# Concurrent searches against a frozen index
# ------------------------------------------------------------


@pytest.mark.parametrize("build", [build_raw, build_quantized], ids=["raw", "quantized"])
def test_concurrent_searches_match_single_threaded(build):
    """Concurrent searches return exactly what the same queries return alone.

    The index is not mutated during the concurrent phase, so the single threaded
    reference is exact. Every worker replays the whole query set several times,
    so any given query overlaps other workers' different queries many times over.
    """
    index, _ = build()
    queries = query_set(index.dim, QUERIES, seed=101)
    reference = [hits(index, query) for query in queries]

    def work(_worker):
        return [[hits(index, query) for query in queries] for _round in range(ROUNDS)]

    for worker_results in run_on_threads(work):
        for round_results in worker_results:
            assert round_results == reference


def test_concurrent_quantized_queries_do_not_share_a_lookup_table():
    """Each quantized search carries its own ADC table.

    The regression this pins is specific. The lookup table the quantized distance
    reads used to live on the index, and a search set it, traversed, then cleared
    it. Two searches overlapping would each overwrite the other's table and score
    candidates against a query they were never given. An exclusive lock on the
    graph was the only thing preventing that, so the table had to move before the
    lock could be relaxed.

    Every worker owns a different query and asserts against that query's own
    single threaded result, so a table belonging to another worker shows up as
    different scores here. The queries are checked to be distinguishable first,
    because references that agreed would make the whole test vacuous.
    """
    index, _ = build_quantized()
    queries = query_set(PQ_DIM, THREADS, seed=202)
    reference = [hits(index, query) for query in queries]

    distinct = {tuple(ident for ident, _ in row) for row in reference}
    assert len(distinct) == THREADS, "the queries must have distinct answers"

    def work(worker):
        query = queries[worker]
        return [hits(index, query) for _round in range(ROUNDS * 4)]

    for worker, worker_results in enumerate(run_on_threads(work)):
        for round_result in worker_results:
            assert round_result == reference[worker]


# ------------------------------------------------------------
# Writing while searches are in flight
# ------------------------------------------------------------


@pytest.mark.parametrize("build", [build_raw, build_quantized], ids=["raw", "quantized"])
def test_add_during_search_does_not_raise(build):
    """An add arriving while a search is in flight completes rather than raising.

    This is the user facing symptom. `add` took `&mut self` and `search` holds a
    shared borrow across the traversal it runs with the GIL released, so the add
    raised `RuntimeError: Already borrowed` on arrival. Nothing about it was
    intermittent once the pairing was constructed, and this constructs it.
    """
    index, _ = build()
    read_batch = query_set(index.dim, READ_BATCH, seed=303)
    write_vectors = clustered(64, index.dim, seed=404)

    with ReadLoad(index, read_batch) as load:
        def write(attempt):
            index.add(
                {
                    "ids": [f"new_{attempt}_{i}" for i in range(64)],
                    "embeddings": write_vectors,
                }
            )

        overlaps = drive_writes(load, write)
        load.check()

    assert overlaps >= MIN_OVERLAPS, "could not get an add to start on top of a read"


def test_remove_metadata_and_compact_during_search_do_not_raise():
    """The other three mutating methods release the object too.

    `remove_point`, `add_metadata` and `compact` took the same exclusive borrow
    `add` did, so each raised in the same pairing. Removal runs first so that
    `compact` has stranded nodes to reclaim and is a real rebuild rather than the
    no-op it returns on a clean graph.
    """
    index, ids = build_raw()
    read_batch = query_set(DIM, READ_BATCH, seed=505)

    with ReadLoad(index, read_batch) as load:
        removals = drive_writes(load, lambda attempt: index.remove_point(ids[attempt]))
        load.check()
        metadata = drive_writes(
            load, lambda attempt: index.add_metadata({"phase": str(attempt)})
        )
        load.check()
        compactions = drive_writes(load, lambda _attempt: index.compact())
        load.check()

    assert removals >= MIN_OVERLAPS, "no removal started on top of a read"
    assert metadata >= MIN_OVERLAPS, "no metadata write started on top of a read"
    assert compactions >= MIN_OVERLAPS, "no compact started on top of a read"
    assert not index.contains(ids[0])


@pytest.mark.parametrize("build", [build_raw, build_quantized], ids=["raw", "quantized"])
def test_concurrent_searches_during_add_stay_correct(build):
    """Searches running through an add return well formed, live results.

    An exact reference does not exist here, because the add changes what the
    right answer is while the queries run. The assertions are the properties that
    hold at every instant instead. Every returned id is one the index was given,
    no id repeats within a result, the result respects `top_k`, and the scores
    come back in non decreasing order. A search reading another query's state
    produces scores that do not order and ids drawn from the wrong neighbourhood,
    which is what this catches.
    """
    index, ids = build()
    known = set(ids)
    queries = query_set(index.dim, THREADS, seed=606)
    batches = [
        {
            "ids": [f"new_{round_}_{i}" for i in range(64)],
            "embeddings": clustered(64, index.dim, seed=700 + round_),
        }
        for round_ in range(40)
    ]
    for batch in batches:
        known.update(batch["ids"])

    writing = threading.Event()
    writing.set()

    def read(worker):
        query = queries[worker]
        seen = 0
        while writing.is_set():
            results = index.search(query, top_k=TOP_K)
            assert len(results) <= TOP_K
            found = [hit["id"] for hit in results]
            assert len(set(found)) == len(found)
            assert set(found) <= known
            scores = [hit["score"] for hit in results]
            assert scores == sorted(scores)
            seen += 1
        return seen

    failure = []

    def writer():
        # The flag is cleared in a finally, so a writer that raises releases the
        # readers rather than leaving them spinning on an event nobody will clear.
        try:
            for batch in batches:
                index.add(batch)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            failure.append(exc)
        finally:
            writing.clear()

    write_thread = threading.Thread(target=writer)
    write_thread.start()
    try:
        counts = run_on_threads(read)
    finally:
        writing.clear()
        write_thread.join(timeout=120)
        assert not write_thread.is_alive(), "writer did not finish"

    if failure:
        raise failure[0]
    assert sum(counts) > 0, "no search ran during the add, so nothing was proved"
