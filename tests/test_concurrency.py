"""Concurrent search, and search running alongside a write.

Three separate defects sat on this path. Search held an exclusive lock on the
graph, so two searches never ran at once. Above that, the mutating methods took
`&mut self`, which PyO3 enforces as an exclusive borrow of the whole object, so a
write arriving while a search held its shared borrow across `allow_threads`
raised `RuntimeError: Already borrowed` before any lock was reached. Below both,
`add` held the interpreter lock for its entire duration, so a write in flight
stopped every Python thread in the process rather than only the ones touching the
index. These tests hold the line on all three, which is that concurrent searches
return exactly what the same queries return alone, that a write in flight never
makes a search raise, and that searches keep running at a real rate while an
insert is underway.

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

import os
import subprocess
import sys
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

# The interpreter lock tests below. `GIL_INSERT` is sized so the single `add`
# call lasts long enough to sample a rate against, on any machine that runs the
# rest of this file in seconds.
GIL_DIM = 32
GIL_BASE = 1500
GIL_INSERT = 6000
SOLO_WINDOW_S = 0.30
MIN_ADD_WINDOW_S = 0.10

# The search rate during an insert, as a share of the rate the same thread
# reaches with no insert running, measured moments earlier in the same process.
# A share rather than a count, so the bound is a fraction of what this machine
# actually does rather than a number tuned to one machine.
#
# Relay 36 measured 2.2 to 2.4 percent for a build where `add` never released
# the lock. Releasing it puts the two threads in ordinary competition for the
# CPU, which is a half share on a single core and close to a full share on
# anything wider. A fifth sits an order of magnitude above the first and well
# below the second.
MIN_SEARCH_SHARE = 0.20

# The sustained mixed workload. Short enough for the suite, and its assertions
# are counts and known answers rather than durations, so a slow machine runs
# fewer rounds rather than failing.
SUSTAINED_SECONDS = 2.0
SUSTAINED_BASE = 800
SUSTAINED_BATCH = 100
EXACT_SCORE_TOLERANCE = 1e-5

# Training fires on a record count, so a run where it did not fire is a failed
# test rather than a quiet pass. Sized to cross the threshold part way through
# the writer's batches.
TRAIN_DIM = 32
TRAIN_SIZE = 1000
TRAIN_PRELOAD = 600
TRAIN_BATCH = 200
TRAIN_BATCHES = 6

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


# ------------------------------------------------------------
# The interpreter lock during a write
# ------------------------------------------------------------


class SearchCounter:
    """One thread issuing single queries in a loop, counting completed calls.

    Single queries rather than batches, because a batch releases the lock once
    for the whole batch and would hide the thing being measured.
    """

    def __init__(self, index, queries):
        self.index = index
        self.queries = queries
        self.count = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        cursor = 0
        size = len(self.queries)
        try:
            while not self._stop.is_set():
                self.index.search(self.queries[cursor], top_k=TOP_K)
                cursor = cursor + 1 if cursor + 1 < size else 0
                self.count += 1
        except BaseException as exc:  # noqa: BLE001 - surfaced by the caller
            self.error = exc

    def __enter__(self):
        self._thread.start()
        # Do not sample until the thread has actually issued a search, so the
        # first window is a steady state rather than thread start-up.
        deadline = time.monotonic() + 30.0
        while self.count == 0 and self.error is None and time.monotonic() < deadline:
            time.sleep(0.001)
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=120)
        return False

    def rate_over(self, seconds):
        started = time.monotonic()
        before = self.count
        time.sleep(seconds)
        return (self.count - before) / (time.monotonic() - started)


def test_search_keeps_its_rate_while_one_insert_runs():
    """A search thread keeps running at a real rate through a long insert.

    `add` used to hold the interpreter lock for its whole duration, so no other
    Python thread could start a search at all while one ran. Relay 36 measured
    the collapse at 97.6 to 98.4 percent of the solo rate, and the cause was the
    lock rather than any lock inside the index.

    The bound is a share of the rate this same thread reaches with no insert
    running, sampled seconds earlier in the same process, so it does not depend
    on the machine being fast. A build that holds the lock cannot reach a fifth
    of its solo rate on any machine, and a build that releases it cannot fall
    below a half share on any machine, because the two threads then simply
    compete for the CPU.

    The insert is one call rather than a loop, so the window being measured is
    exactly one `add` and there are no gaps between calls to hide in. The test
    asserts the window was long enough to sample, so a machine that finished the
    insert too fast to measure fails rather than passing on three samples.
    """
    index, _ = build_raw(n=GIL_BASE, dim=GIL_DIM, seed=17)
    queries = query_set(GIL_DIM, 64, seed=808)
    batch = {
        "ids": [f"bulk_{i}" for i in range(GIL_INSERT)],
        "embeddings": clustered(GIL_INSERT, GIL_DIM, seed=909),
    }

    with SearchCounter(index, queries) as reader:
        solo_rate = reader.rate_over(SOLO_WINDOW_S)

        before = reader.count
        started = time.monotonic()
        result = index.add(batch)
        window = time.monotonic() - started
        during_rate = (reader.count - before) / window

    assert reader.error is None, reader.error
    assert result.total_inserted == GIL_INSERT
    assert window >= MIN_ADD_WINDOW_S, (
        f"the insert took {window:.3f}s, too short to sample a rate against"
    )
    assert solo_rate > 0, "the reader never completed a search on its own"
    assert during_rate >= MIN_SEARCH_SHARE * solo_rate, (
        f"searches ran at {during_rate:.0f}/s during the insert against "
        f"{solo_rate:.0f}/s alone, a share of {during_rate / solo_rate:.3f}"
    )


# ------------------------------------------------------------
# A sustained mixed workload
# ------------------------------------------------------------


def half_space(n, dim, seed, upper):
    """Unit vectors confined to one half of the axes.

    Two sets built on opposite halves are orthogonal, so their cosine distance
    is exactly 1. That is what lets a query keep a known answer while records
    are being inserted, since nothing arriving in the other half can ever come
    closer than a record's own vector at distance 0.
    """
    rng = np.random.default_rng(seed)
    points = np.zeros((n, dim), dtype=np.float64)
    half = dim // 2
    block = np.abs(rng.standard_normal((n, half))) + 0.1
    if upper:
        points[:, half:] = block
    else:
        points[:, :half] = block
    return unit(points)


def test_sustained_mixed_workload_leaves_the_index_exact():
    """Search and insert together for a while, then check the index is right.

    The known answer is a record's own vector. Every base record lives in one
    half of the axes and every inserted record in the other, so the two sets are
    orthogonal and an inserted record is at distance 1 from any base query while
    the base record itself is at distance 0. No amount of insertion can change
    which record answers a base query, which is what makes this a correctness
    assertion rather than an invariant.

    The premise is checked rather than assumed. Before the concurrent phase
    starts, every query is run alone and asserted to return its own record first
    at distance 0, with the runner up far enough behind that a tie is not
    possible. A query set that did not have that property would make the whole
    test vacuous.

    The count assertion is exact and independent of timing. The writer inserts a
    fixed number of batches with fixed ids, the readers stop when it finishes,
    and the final record count must equal the base plus every id written.
    """
    dim = 32
    base = half_space(SUSTAINED_BASE, dim, seed=21, upper=False)
    ids = [f"base_{i}" for i in range(SUSTAINED_BASE)]
    index = VectorDatabase().create("hnsw", dim=dim, expected_size=SUSTAINED_BASE * 8)
    assert index.add({"ids": ids, "embeddings": base}).is_success()

    probes = list(range(0, SUSTAINED_BASE, SUSTAINED_BASE // THREADS))[:THREADS]
    for probe in probes:
        results = index.search(base[probe], top_k=TOP_K)
        assert results[0]["id"] == ids[probe]
        assert results[0]["score"] < EXACT_SCORE_TOLERANCE
        assert results[1]["score"] > 100 * EXACT_SCORE_TOLERANCE, (
            "the runner up is too close for the first hit to be a known answer"
        )

    writing = threading.Event()
    writing.set()
    written = []
    failure = []

    def writer():
        try:
            round_ = 0
            deadline = time.monotonic() + SUSTAINED_SECONDS
            while time.monotonic() < deadline:
                batch_ids = [f"other_{round_}_{i}" for i in range(SUSTAINED_BATCH)]
                index.add(
                    {
                        "ids": batch_ids,
                        "embeddings": half_space(
                            SUSTAINED_BATCH, dim, seed=3000 + round_, upper=True
                        ),
                    }
                )
                written.extend(batch_ids)
                round_ += 1
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            failure.append(exc)
        finally:
            writing.clear()

    def read(worker):
        probe = probes[worker]
        expected = ids[probe]
        query = base[probe]
        seen = 0
        while writing.is_set():
            results = index.search(query, top_k=TOP_K)
            assert results[0]["id"] == expected
            assert results[0]["score"] < EXACT_SCORE_TOLERANCE
            found = [hit["id"] for hit in results]
            assert len(set(found)) == len(found)
            scores = [hit["score"] for hit in results]
            assert scores == sorted(scores)
            seen += 1
        return seen

    write_thread = threading.Thread(target=writer)
    write_thread.start()
    try:
        counts = run_on_threads(read)
    finally:
        writing.clear()
        write_thread.join(timeout=180)
        assert not write_thread.is_alive(), "writer did not finish"

    if failure:
        raise failure[0]
    assert sum(counts) > 0, "no search ran during the workload"
    assert written, "no batch was written"

    assert index.get_vector_count() == SUSTAINED_BASE + len(written)
    for written_id in (written[0], written[len(written) // 2], written[-1]):
        assert index.contains(written_id)

    # The same queries once the index is still, against the same known answer.
    for worker, probe in enumerate(probes):
        results = index.search(base[probe], top_k=TOP_K)
        assert results[0]["id"] == ids[probe]
        assert results[0]["score"] < EXACT_SCORE_TOLERANCE


# ------------------------------------------------------------
# Training completing under load
# ------------------------------------------------------------


def test_training_completes_under_a_concurrent_search_load():
    """Product quantization training fires mid-workload without breaking it.

    Training is the longest thing an insert can do. It runs k-means over the
    collected vectors and then rebuilds the whole graph from quantized codes, and
    all of it now happens with the interpreter lock released. This checks that
    the transition is safe while searches are in flight, and that searches keep
    completing through it.

    Nothing here depends on a duration. Training fires on a record count, so the
    test asserts the index was not quantized before the crossing batch and is
    quantized after it. A run where training did not fire fails on that pair
    rather than passing without exercising anything.

    The overlap is proved by counting. The reader pool's completed reads are
    sampled either side of the batch that crosses the threshold, and the test
    asserts reads landed inside that window. A build that held the interpreter
    lock through training records none.
    """
    vectors = clustered(TRAIN_PRELOAD, TRAIN_DIM, seed=31)
    index = VectorDatabase().create(
        "hnsw",
        dim=TRAIN_DIM,
        expected_size=20000,
        quantization_config={
            "type": "pq",
            "subvectors": 8,
            "bits": 8,
            "training_size": TRAIN_SIZE,
        },
    )
    assert index.add(
        {"ids": [f"pre_{i}" for i in range(TRAIN_PRELOAD)], "embeddings": vectors}
    ).is_success()
    assert not index.is_quantized(), "training must not have fired yet"

    read_batch = query_set(TRAIN_DIM, READ_BATCH, seed=707)
    total = TRAIN_PRELOAD
    crossing_reads = 0
    quantized_before_crossing = None

    with ReadLoad(index, read_batch) as load:
        for round_ in range(TRAIN_BATCHES):
            quantized_before = index.is_quantized()
            before = load.reads
            index.add(
                {
                    "ids": [f"t_{round_}_{i}" for i in range(TRAIN_BATCH)],
                    "embeddings": clustered(TRAIN_BATCH, TRAIN_DIM, seed=800 + round_),
                }
            )
            total += TRAIN_BATCH
            load.check()
            if not quantized_before and index.is_quantized():
                quantized_before_crossing = quantized_before
                crossing_reads = load.reads - before
        load.check()

    assert quantized_before_crossing is False, (
        "training never completed, so the transition was not exercised"
    )
    assert index.is_quantized()
    assert crossing_reads > 0, "no search completed while training ran"
    assert index.get_vector_count() == total

    # The index still answers after the rebuild the training triggered.
    for query in query_set(TRAIN_DIM, 4, seed=1212):
        results = index.search(query, top_k=TOP_K)
        assert 0 < len(results) <= TOP_K
        assert [hit["score"] for hit in results] == sorted(
            hit["score"] for hit in results
        )


# ------------------------------------------------------------
# get_stats against concurrent mutation
# ------------------------------------------------------------

# get_stats used to hold the vectors read guard and then take id_map, while
# remove_point_internal holds id_map and then takes vectors, so the two in
# flight together deadlocked with the stats thread holding the interpreter
# lock and the whole process froze. It also re-read training_ids inside its
# own training_ids hold, which deadlocks the moment a writer queues between
# the two reads, because the standard library queues readers behind waiting
# writers. Each mode below isolates one mechanism: "removal" runs on a
# trained index, where the recursive read cannot happen, and
# "training_collection" runs pure adds against an unreachable training_size,
# where no remove ever holds two guards.
#
# The probe runs in a subprocess because the failure mode is a frozen
# interpreter. No assertion can run inside the deadlocked process, so the
# parent's timeout is the detector. On a build where every stats guard is
# taken alone the workload finishes in a few seconds regardless of
# scheduling; the wide budget absorbs a loaded machine rather than tuning a
# race, and a regression hangs the child outright rather than making it slow,
# so this cannot flake red on good code. What one pass cannot promise is
# sensitivity, since a single clean run proves the loop completed rather than
# that no window exists. The window itself froze seven of seven twenty second
# runs on the build that carried it.

PROBE_CHURN_SECONDS = 8.0
PROBE_TIMEOUT_S = 150.0

PROBE_SCRIPT = '''
import sys
import threading
import time
from collections import deque

import numpy as np
from zeusdb_vector_database import VectorDatabase

mode = sys.argv[1]
seconds = float(sys.argv[2])
DIM = 16
BATCH = 200
rng = np.random.default_rng(11)

index = VectorDatabase().create(
    "hnsw",
    dim=DIM,
    expected_size=20000,
    quantization_config={
        "type": "pq",
        "subvectors": 8,
        "bits": 8,
        "training_size": 100000 if mode == "training_collection" else 1000,
        "storage_mode": "quantized_with_raw",
    },
)
seed_count = 1400 if mode == "removal" else 500
seeds = rng.standard_normal((seed_count, DIM)).astype(np.float32)
index.add({"ids": [f"seed_{i}" for i in range(seed_count)], "embeddings": seeds})
assert index.is_quantized() == (mode == "removal"), index.get_storage_mode()

stop = threading.Event()
counts = {"stats": 0, "adds": 0, "removes": 0}
pending = deque()
pending_lock = threading.Lock()


def stats_loop():
    while not stop.is_set():
        index.get_stats()
        counts["stats"] += 1


def add_loop():
    # Long detached batch inserts keep the writers mutex busy, so the remove
    # thread wakes from the writers queue at a point uncorrelated with the
    # stats loop. The two thread version of this probe never reproduced the
    # deadlock, because a remove entered at a GIL hand-off always cleared its
    # guard window before get_stats reached the vulnerable span.
    batch = rng.standard_normal((BATCH, DIM)).astype(np.float32)
    n = 0
    while not stop.is_set():
        ids = [f"churn_{n}_{i}" for i in range(BATCH)]
        index.add({"ids": ids, "embeddings": batch})
        with pending_lock:
            pending.extend(ids)
        counts["adds"] += 1
        n += 1


def remove_loop():
    while not stop.is_set():
        with pending_lock:
            rid = pending.popleft() if pending else None
        if rid is None:
            time.sleep(0.001)
            continue
        index.remove_point(rid)
        counts["removes"] += 1


threads = [
    threading.Thread(target=stats_loop, daemon=True),
    threading.Thread(target=add_loop, daemon=True),
]
if mode == "removal":
    threads.append(threading.Thread(target=remove_loop, daemon=True))
for t in threads:
    t.start()
deadline = time.monotonic() + seconds
while time.monotonic() < deadline:
    time.sleep(0.2)
stop.set()
for t in threads:
    t.join(timeout=30)
assert not any(t.is_alive() for t in threads), "a worker never came back"
print(f"OK {counts['stats']} {counts['adds']} {counts['removes']}")
'''


@pytest.mark.parametrize("probe_mode", ["removal", "training_collection"])
def test_get_stats_never_deadlocks_against_mutation(tmp_path, probe_mode):
    """get_stats loops beside mutation without freezing the process."""
    script = tmp_path / "stats_deadlock_probe.py"
    script.write_text(PROBE_SCRIPT, encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(script), probe_mode, str(PROBE_CHURN_SECONDS)],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"get_stats deadlocked against {probe_mode}: the probe froze and "
            f"was killed after {PROBE_TIMEOUT_S:.0f}s"
        )
    assert result.returncode == 0, result.stderr

    # The run must have exercised both sides, not idled to a quiet pass.
    stats_calls, add_calls, remove_calls = map(int, result.stdout.split()[1:4])
    assert stats_calls > 100, "the stats loop barely ran"
    assert add_calls > 1, "the churn loop barely ran"
    if probe_mode == "removal":
        assert remove_calls > 10, "no removal pressure was generated"


# ============================================================================
# save() against concurrent mutation
# ============================================================================
#
# Nothing in this file mentioned `save` at all, and a lock order inversion lived
# in that path once: `save_manifest` held the `vectors` and `pq_codes` read
# guards across a `get_storage_mode` call, which takes the graph's read guard,
# acquiring the three in the reverse of the documented `hnsw < vectors <
# pq_codes` order. It could not deadlock, because a save holds the mutation lock
# and so does every path taking the graph's write guard, but that is the same
# reasoning that failed for the three inversions found before it, and nothing in
# the suite would have caught it if the mutation lock ever stopped covering one
# of them.
#
# The shape is the get_stats probe's: a subprocess with a parent timeout, since
# the failure mode is a frozen interpreter that no in-process assertion can
# report. Saves run in a loop beside adds, removes and searches.
#
# # Why the assertions count overlaps rather than operations
#
# The three mutating loops share one mutex, so they cannot be given independent
# throughput floors. Measured on one machine over 8 seconds at 9,600 records,
# `save` costs 37.8 ms uncontended and `remove_point` costs 0.001 ms, so a round
# of the mutation lock is dominated by the save and each round serves one save,
# one add and one removal. The three counts come out at 190, 176 and 203 in raw
# mode and 95, 94 and 124 in quantized mode, pinned to a 1:1:1 ratio, and a
# contended `remove_point` measures 39.3 ms against 0.001 ms of work, so it is
# 99.997 percent lock wait.
#
# A floor of 10 on removals was therefore a floor of 10 on **saves**, which is a
# statement about how many rounds of that mutex the machine completes in 8
# seconds and not about whether removal happened. A slower machine completing 5
# rounds reads 5 removals, 5 saves and 5 adds, and only the removal floor fails,
# because the other two were set at 1. The counts were consistent and the floor
# was the wrong kind of assertion.
#
# What the test needs is that every loop ran and that the interleaving it exists
# to create actually occurred. `save` holds the mutation lock for essentially the
# whole run, so a removal requested during a save is a removal queued behind it,
# which is the ordering an inversion between the two paths needs. The remove loop
# waits for a save to be in flight before removing, so that this is deliberate
# rather than left to how many rounds the machine got through, and a search
# completing during a save is the assertion that a save takes the mutation lock
# and no reader lock. Neither count scales with machine speed, and both go to
# zero if their loop stops.

SAVE_PROBE_SECONDS = 8.0
SAVE_PROBE_TIMEOUT_S = 180.0

SAVE_PROBE_SCRIPT = '''
import os
import sys
import threading
import time
from collections import deque

import numpy as np
from zeusdb_vector_database import VectorDatabase

mode = sys.argv[1]
seconds = float(sys.argv[2])
root = sys.argv[3]
DIM = 16
BATCH = 150
rng = np.random.default_rng(23)

quantization = None
if mode == "quantized":
    quantization = {
        "type": "pq",
        "subvectors": 8,
        "bits": 8,
        "training_size": 1000,
        "storage_mode": "quantized_with_raw",
    }

index = VectorDatabase().create(
    "hnsw", dim=DIM, expected_size=20000, quantization_config=quantization,
)
seed_count = 1400 if mode == "quantized" else 600
seeds = rng.standard_normal((seed_count, DIM)).astype(np.float32)
index.add({"ids": [f"seed_{i}" for i in range(seed_count)], "embeddings": seeds})
assert index.is_quantized() == (mode == "quantized"), index.get_storage_mode()

stop = threading.Event()
counts = {"saves": 0, "adds": 0, "removes": 0, "searches": 0,
          "removes_during_save": 0, "searches_during_save": 0}
failures = []
pending = deque()
pending_lock = threading.Lock()
queries = rng.standard_normal((32, DIM)).astype(np.float32)
# Set while index.save() is running, which is how the other two loops learn
# that their operation met one.
save_active = threading.Event()


def guarded(name, body):
    def run():
        try:
            body()
        except BaseException as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            stop.set()
    return run


def save_loop():
    n = 0
    while not stop.is_set():
        # Alternating targets, so a save is never writing over the directory the
        # previous one wrote and a directory a save is still filling is not
        # mistaken afterwards for one it finished.
        save_active.set()
        try:
            index.save(os.path.join(root, "snap_%d.zdb" % (n % 2)))
        finally:
            save_active.clear()
        counts["saves"] += 1
        n += 1


def add_loop():
    batch = rng.standard_normal((BATCH, DIM)).astype(np.float32)
    n = 0
    while not stop.is_set():
        ids = [f"churn_{n}_{i}" for i in range(BATCH)]
        index.add({"ids": ids, "embeddings": batch})
        with pending_lock:
            pending.extend(ids)
        counts["adds"] += 1
        n += 1


def remove_loop():
    while not stop.is_set():
        with pending_lock:
            rid = pending.popleft() if pending else None
        if rid is None:
            time.sleep(0.001)
            continue
        # Line the removal up behind a running save on purpose. The two share
        # the mutation lock, so they never execute at the same instant, and what
        # an inversion between them needs is one waiting on the other. The add
        # loop keeps this queue oversupplied by two orders of magnitude, at
        # 26,400 ids queued against 203 removed, so the wait here is on the save
        # and never on the queue. Bounded, so a save loop that has died leaves
        # this counting removals rather than spinning.
        during = save_active.wait(timeout=1.0)
        index.remove_point(rid)
        counts["removes"] += 1
        if during:
            counts["removes_during_save"] += 1


def search_loop():
    # A save holds the mutation lock and no reader lock, so searches must keep
    # answering throughout. This also puts a reader on the storage guards while
    # the save is reading them.
    n = 0
    while not stop.is_set():
        during = save_active.is_set()
        page = index.search(queries[n % len(queries)], top_k=5)
        assert isinstance(page, list)
        counts["searches"] += 1
        if during and save_active.is_set():
            # Entered and left with a save still in flight, so this one was
            # answered while a save held the mutation lock.
            counts["searches_during_save"] += 1
        n += 1


threads = [
    threading.Thread(target=guarded("save", save_loop), daemon=True),
    threading.Thread(target=guarded("add", add_loop), daemon=True),
    threading.Thread(target=guarded("remove", remove_loop), daemon=True),
    threading.Thread(target=guarded("search", search_loop), daemon=True),
]
for t in threads:
    t.start()
deadline = time.monotonic() + seconds
while time.monotonic() < deadline and not stop.is_set():
    time.sleep(0.2)
stop.set()
for t in threads:
    t.join(timeout=60)
assert not any(t.is_alive() for t in threads), "a worker never came back"
assert not failures, "; ".join(failures)

# Every directory a save finished writing must load, and what it holds must be
# one instant of the index rather than a mixture of two. A save takes the
# mutation lock, so the mappings and the stores cannot come from either side of
# an insertion.
checked = 0
for n in (0, 1):
    path = os.path.join(root, "snap_%d.zdb" % n)
    if not os.path.isdir(path):
        continue
    loaded = VectorDatabase().load(path)
    assert len(loaded) > 0, path
    page = loaded.search(queries[0], top_k=5)
    assert isinstance(page, list), path
    stats = loaded.get_stats()
    assert int(stats["total_vectors"]) == len(loaded), (path, stats["total_vectors"])
    # Every record the id map names has metadata and a vector or a code behind
    # it, which is what a torn save would break.
    listed = loaded.list(number=50)
    fetched = loaded.get_records([rid for rid, _ in listed], return_vector=True)
    assert len(fetched) == len(listed), (path, len(fetched), len(listed))
    checked += 1
assert checked > 0, "no snapshot was written"

print("OK " + " ".join(str(counts[k]) for k in
                       ("saves", "adds", "removes", "searches",
                        "removes_during_save", "searches_during_save")))
'''


@pytest.mark.parametrize("probe_mode", ["raw", "quantized"])
def test_save_never_deadlocks_or_tears_against_mutation(tmp_path, probe_mode):
    """save loops beside adds, removes and searches without freezing or tearing.

    Two failures are in scope. A lock order inversion in the save path would
    hang the child, which the parent timeout detects. A save reading the
    mappings and the stores at different instants would write a directory whose
    id map names records the stores do not hold, which the child's reload
    assertions detect.
    """
    script = tmp_path / "save_concurrency_probe.py"
    script.write_text(SAVE_PROBE_SCRIPT, encoding="utf-8")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    try:
        # save() prints a progress banner carrying non-ASCII, and the default
        # child encoding on Windows cannot decode it, so the stream is read as
        # UTF-8 with replacement rather than left to the console codepage.
        child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run(
            [sys.executable, str(script), probe_mode,
             str(SAVE_PROBE_SECONDS), str(snapshots)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=SAVE_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"save deadlocked against mutation in {probe_mode} mode: the probe "
            f"froze and was killed after {SAVE_PROBE_TIMEOUT_S:.0f}s"
        )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    # The counts are on the one line beginning OK, since save writes a banner
    # to the same stream.
    summary = [line for line in result.stdout.splitlines() if line.startswith("OK ")]
    assert len(summary) == 1, result.stdout[-2000:]
    saves, adds, removes, searches, removes_during_save, searches_during_save = map(
        int, summary[0].split()[1:7]
    )
    seen = (f"saves={saves} adds={adds} removes={removes} searches={searches} "
            f"removes_during_save={removes_during_save} "
            f"searches_during_save={searches_during_save}")

    # Liveness. Every loop got at least one operation through. These are not
    # throughput floors. The three mutating loops share one mutex and run at a
    # 1:1:1 ratio, so any number above one would be a floor on how many rounds
    # of that mutex the machine completes, which is what the note above the
    # probe records.
    assert saves >= 1, f"the save loop never completed a save. {seen}"
    assert adds >= 1, f"the add loop never completed an add. {seen}"
    assert removes >= 1, f"the remove loop never removed a record. {seen}"
    assert searches >= 1, f"the search loop never completed a search. {seen}"

    # The interleaving this test exists to create. A removal requested while a
    # save held the mutation lock is a removal queued behind that save. Zero
    # here means the two never met, so a clean run proved nothing about them.
    assert removes_during_save >= 1, (
        f"no removal overlapped a save, so the interleaving under test never "
        f"occurred. {seen}"
    )

    # A save takes the mutation lock and no reader lock, so a search has to be
    # answerable while one runs. Zero here means saves are excluding readers,
    # which is a regression whatever the totals say.
    assert searches_during_save >= 1, (
        f"no search completed while a save held the mutation lock. {seen}"
    )
