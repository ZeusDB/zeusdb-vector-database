"""Concurrent search, and search running alongside a write.

Every method of the index takes `&self`, so a write that arrives while a
search is in flight takes no exclusive borrow of the object. These tests hold
that concurrent searches return exactly what the same queries return alone,
and that a write in flight never makes a search raise. That a call releases
the interpreter for its work, so other Python threads run beside it, is held
in `tests/test_interpreter.py`, and that a write made while a search is held
in flight completes is held in `tests/test_overlap.py`.

Every test here is built to fail for a reason rather than on timing.

Where the index is frozen, single threaded search over it is deterministic, so
the reference results are exact and any difference under concurrency is real
interference rather than scheduling noise. Where the index is being written, an
exact reference does not exist, so the assertions drop to invariants that hold at
every instant of the write instead.

Every count a test asserts is one its own loops run until they reach, so a slow
machine takes longer and fails nothing. Every wait ends at a limit far above what
any run needs, and reaching it fails the test with words that name a hang. Where
a rule needs one operation to overlap another, the overlap is made at a pause, in
`tests/test_overlap.py` and in the index crate's own tests, and not counted here.
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

# The sustained mixed workload. Its assertions are counts and known answers
# rather than durations, and the writer is bounded by batches rather than by a
# clock. Bounded by a clock, the eight readers had to be created and scheduled
# inside the writer's window. On four cores they were not: every reader
# returned nothing and the run failed with the index exactly right.
SUSTAINED_ROUNDS = 40
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

# The queries a read thread searches as one batch, so its shared borrow of the
# index is held for the whole batch.
READ_BATCH = 128


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
    """Background threads searching continuously.

    Reads run a batch, so each thread's shared borrow of the index is held for
    the whole batch rather than for one traversal. A reader that raises stops
    and keeps what it raised, which `check()` raises on the calling thread.
    """

    def __init__(self, index, batch, threads=4):
        self.index = index
        self.batch = batch
        self.reads = 0
        self.errors = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._run) for _ in range(threads)]

    def _run(self):
        while not self._stop.is_set():
            try:
                self.index.search(self.batch, top_k=TOP_K)
            except BaseException as exc:  # noqa: BLE001 - surfaced by check()
                self.errors.append(exc)
                return
            finally:
                with self._lock:
                    self.reads += 1

    def __enter__(self):
        for thread in self._threads:
            thread.start()
        # Do not proceed until the readers are actually running, so the first
        # write is not raced against an empty pool. The bound is a stuck pool
        # detector and not a race bound: a pool that has completed no read in
        # thirty seconds is not slow, and saying so here is better than
        # letting the caller spend its own budget discovering that what it
        # raced was empty.
        deadline = time.monotonic() + 30.0
        while self.reads == 0 and not self.errors and time.monotonic() < deadline:
            time.sleep(0.001)
        self.check()
        assert self.reads > 0, "the reader pool completed no read in thirty seconds"
        return self

    def __exit__(self, *_):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=120)
        return False

    def check(self):
        if self.errors:
            raise self.errors[0]


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
    # Every reader and the writer meet here before the first add, so the adds
    # cannot all complete before a reader has been scheduled. On a machine
    # where the writer wins the start, forty batches finished before the first
    # search and the test failed for having proved nothing.
    readers_up = threading.Barrier(THREADS + 1)

    def read(worker):
        query = queries[worker]
        seen = 0
        readers_up.wait(timeout=30)
        # One search on every reader before the flag is read, then as many as
        # the write lasts for. The count below is the reader count whatever
        # the scheduler does, and the barrier above is what puts those
        # searches on top of the write rather than after it.
        while True:
            results = index.search(query, top_k=TOP_K)
            assert len(results) <= TOP_K
            found = [hit["id"] for hit in results]
            assert len(set(found)) == len(found)
            assert set(found) <= known
            scores = [hit["score"] for hit in results]
            assert scores == sorted(scores)
            seen += 1
            if not writing.is_set():
                return seen

    failure = []

    def writer():
        # The flag is cleared in a finally, so a writer that raises releases the
        # readers rather than leaving them spinning on an event nobody will clear.
        try:
            readers_up.wait(timeout=30)
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
    assert sum(counts) >= THREADS, (
        f"every reader runs a search before it reads the flag, so {sum(counts)} "
        f"searches across {THREADS} readers means a reader never ran"
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

    Nothing here is bounded by a clock. The writer inserts a fixed number of
    batches with fixed ids, the readers stop when it finishes, and the final
    record count must equal the base plus every id written. The readers and
    the writer meet at a barrier before the first batch and every reader runs
    a search before it reads the flag, so the count below is the reader count
    on a machine of any speed. Bounded by a clock instead, the readers had to
    be created and scheduled inside the writer's window, and on four cores
    they were not.
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
    # Every reader and the writer meet here before the first batch, so the
    # readers are running when the writing starts rather than being created
    # inside a window the writer is already spending.
    readers_up = threading.Barrier(THREADS + 1)

    def writer():
        try:
            readers_up.wait(timeout=30)
            for round_ in range(SUSTAINED_ROUNDS):
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
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            failure.append(exc)
        finally:
            writing.clear()

    def read(worker):
        probe = probes[worker]
        expected = ids[probe]
        query = base[probe]
        seen = 0
        readers_up.wait(timeout=30)
        while True:
            results = index.search(query, top_k=TOP_K)
            assert results[0]["id"] == expected
            assert results[0]["score"] < EXACT_SCORE_TOLERANCE
            found = [hit["id"] for hit in results]
            assert len(set(found)) == len(found)
            scores = [hit["score"] for hit in results]
            assert scores == sorted(scores)
            seen += 1
            if not writing.is_set():
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
    assert sum(counts) >= THREADS, (
        f"every reader runs a search before it reads the flag, so {sum(counts)} "
        f"searches across {THREADS} readers means a reader never ran"
    )
    assert len(written) == SUSTAINED_ROUNDS * SUSTAINED_BATCH

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

    Training runs k-means over the collected vectors and then rebuilds the
    whole graph from quantized codes. Four threads search the index through
    every batch, the one that trains among them, and a search that raises
    fails the test. That the training runs with the interpreter released is
    held in `tests/test_interpreter.py`.

    Nothing here depends on a duration. Training fires on a record count, so the
    test asserts the index was not quantized before the crossing batch and is
    quantized after it. A run where training did not fire fails on that pair
    rather than passing without exercising anything.
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
    quantized_before_crossing = None

    with ReadLoad(index, read_batch) as load:
        for round_ in range(TRAIN_BATCHES):
            quantized_before = index.is_quantized()
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
        load.check()

    assert quantized_before_crossing is False, (
        "training never completed, so the transition was not exercised"
    )
    assert index.is_quantized()
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

# get_stats takes every guard alone and in the declared order, so it holds no
# guard while it waits for another, and it reads training_ids once. Each mode
# below runs it beside one kind of mutation. "removal" runs on a trained index
# beside adds and removals, which take the id store and then the index, and
# "training_collection" runs beside adds against an unreachable training_size,
# which collect training ids. The lock rank registry of a build with debug
# assertions refuses an inverted or repeated acquisition at once on every path
# the probe runs, and the child reports it as a failure of its loop.
#
# The probe runs in a subprocess, since on a build without the registry the
# failure is a frozen interpreter, and no assertion can run inside a deadlocked
# process. The parent's timeout is the detector, and its words name a hang.
# The child churns until it has run for PROBE_CHURN_SECONDS and every count has
# reached its target, so it does the same work on a machine of any speed.

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
# Every loop runs until the churn has lasted `seconds` and every count has
# reached its target.
TARGETS = {"stats": 101, "adds": 2, "removes": 11 if mode == "removal" else 0}

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
failures = []
pending = deque()
pending_lock = threading.Lock()


def guarded(name, body):
    def run():
        try:
            body()
        except BaseException as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            stop.set()
    return run


def stats_loop():
    while not stop.is_set():
        index.get_stats()
        counts["stats"] += 1


def add_loop():
    # Batch inserts with the interpreter released keep the mutation lock
    # busy, so the remove thread wakes from its wait for that lock at a point
    # the stats loop does not choose.
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
    threading.Thread(target=guarded("stats", stats_loop), daemon=True),
    threading.Thread(target=guarded("add", add_loop), daemon=True),
]
if mode == "removal":
    threads.append(threading.Thread(target=guarded("remove", remove_loop), daemon=True))
for t in threads:
    t.start()
started = time.monotonic()
while not stop.is_set() and (
    time.monotonic() - started < seconds
    or any(counts[name] < target for name, target in TARGETS.items())
):
    time.sleep(0.2)
stop.set()
for t in threads:
    t.join(timeout=30)
assert not any(t.is_alive() for t in threads), "a worker never came back"
assert not failures, "; ".join(failures)
assert all(counts[name] >= target for name, target in TARGETS.items()), counts
print(f"OK {counts['stats']} {counts['adds']} {counts['removes']}")
'''


@pytest.mark.parametrize("probe_mode", ["removal", "training_collection"])
def test_get_stats_never_deadlocks_against_mutation(tmp_path, probe_mode):
    """get_stats loops beside mutation without freezing the process.

    The child churns until each loop has reached its target, and a failure
    of any loop fails the child, so a pass is a run of the whole load.
    """
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
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    # The child's own targets, which it runs until it reaches.
    stats_calls, add_calls, remove_calls = map(int, result.stdout.split()[1:4])
    assert stats_calls > 100, "the stats loop stopped short of its target"
    assert add_calls > 1, "the churn loop stopped short of its target"
    if probe_mode == "removal":
        assert remove_calls > 10, "the remove loop stopped short of its target"


# ============================================================================
# The two live sets, read from outside the mutation lock
# ============================================================================
#
# The dense index keeps a bitmap of the records it holds, which is what an
# unfiltered traversal runs its predicate against, and the collection keeps the
# same set in its id store. They are one set held twice, and `get_stats` checks
# them against each other on a build with debug assertions.
#
# The insertion writes both under one acquisition of its four write guards, and
# the clear trades both under one acquisition, so a reader that takes no
# mutation lock, which `get_stats` is, finds them agree wherever it lands. The
# index crate's tests hold a reader at every place an insertion or a clear lets
# its guards go.
#
# Here two readers run `get_stats` beside a writer on the wheel's own paths. The
# writer makes LIVE_SET_WRITES writes, and after each one it waits until each
# reader has completed a call begun after the write began, so the reads keep
# pace with the writes on a machine of any speed. LIVE_SET_LIMIT_S bounds the
# whole run, and reaching it fails the arm with words that name a hang.

LIVE_SET_WRITES = 40
LIVE_SET_LIMIT_S = 120.0


@pytest.mark.parametrize("path", ["insert", "clear"])
def test_the_two_live_sets_agree_under_a_concurrent_reader(path):
    """get_stats beside a write never sees the two live sets disagree.

    Each reader runs in its own thread and takes no mutation lock. On a build
    with debug assertions a disagreement raises out of `get_stats` into the
    reader, and the arm reports it rather than letting the traceback pass by.
    After each write the writer waits until every reader has completed a call
    begun after the write began.
    """
    dim = 8
    fill = 100
    readers = 2
    rng = np.random.default_rng(136)
    vectors = rng.standard_normal((fill, dim)).astype(np.float32)
    ids = [f"live_{i}" for i in range(fill)]
    index = VectorDatabase().create("hnsw", dim=dim, expected_size=fill * 4)
    if path == "insert":
        # A seed, so the first phase of each insertion has a graph to descend.
        index.add({"ids": ids, "embeddings": vectors})

    stop = threading.Event()
    # How many calls each reader has begun, and the number of the last call
    # it has finished.
    begun = [0] * readers
    done = [0] * readers
    raised = []

    def reader(k):
        while not stop.is_set():
            begun[k] += 1
            try:
                index.get_stats()
            except BaseException as exc:  # noqa: BLE001
                raised.append(f"{type(exc).__name__}: {exc}")
                stop.set()
                return
            done[k] = begun[k]

    threads = [
        threading.Thread(target=reader, args=(k,), daemon=True) for k in range(readers)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + LIVE_SET_LIMIT_S
    writes = 0
    try:
        for n in range(LIVE_SET_WRITES):
            marks = list(begun)
            if path == "insert":
                batch = [f"churn_{n}_{i}" for i in range(fill)]
                index.add({"ids": batch, "embeddings": vectors})
            else:
                index.add({"ids": ids, "embeddings": vectors})
                index.clear()
            writes += 1
            while not stop.is_set() and any(done[k] <= marks[k] for k in range(readers)):
                assert time.monotonic() < deadline, (
                    f"the readers completed no call beside write {n} within "
                    f"{LIVE_SET_LIMIT_S:.0f} seconds, so a reader hung"
                )
                time.sleep(0.001)
            if stop.is_set():
                break
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads), "a reader hung"
    assert not raised, (
        f"get_stats saw the two live sets disagree during a {path}: {raised[0]}"
    )
    assert writes == LIVE_SET_WRITES
    assert min(done) >= LIVE_SET_WRITES, (
        f"each reader completes a call beside each write, and they completed {done}"
    )


# ============================================================================
# save() against concurrent mutation
# ============================================================================
#
# A save holds the mutation lock across its four phases and takes no guard a
# reader waits for, so every mutation of what it writes is ordered against it
# and a search answers while it runs. Each guard it reads is taken in the
# declared order, which the lock rank registry of a build with debug
# assertions checks at every acquisition on every path the probe runs.
#
# The shape is the get_stats probe's: a subprocess with a parent timeout, since
# on a build without the registry the failure is a frozen interpreter that no
# in-process assertion can report. Saves run in a loop beside adds, removes and
# searches, and every directory a save finished writing must load as one
# instant of the index.
#
# The child churns until it has run for SAVE_PROBE_SECONDS and every loop has
# completed an operation. The three mutating loops share the mutation lock, so
# no loop is held to a count above one. The remove loop waits, for a second at
# most, until a save is in flight before it removes, so its removals queue
# behind saves. That a search answers inside a held save, and that a removal
# and an add wait for it, is held at a pause in `collection::persist_tests` in
# the index crate.

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
counts = {"saves": 0, "adds": 0, "removes": 0, "searches": 0}
failures = []
pending = deque()
pending_lock = threading.Lock()
queries = rng.standard_normal((32, DIM)).astype(np.float32)
# Set while index.save() is running, which is what the remove loop waits for.
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
        # The removal lines up behind a running save. The two share the
        # mutation lock, so they never run at the same instant, and what an
        # inversion between them needs is one waiting on the other. Bounded,
        # so a save loop that has stopped leaves this removing.
        save_active.wait(timeout=1.0)
        index.remove_point(rid)
        counts["removes"] += 1


def search_loop():
    # A save holds the mutation lock and no reader lock, so searches keep
    # answering throughout, and a search reads the storage guards beside the
    # save's reads of them.
    n = 0
    while not stop.is_set():
        page = index.search(queries[n % len(queries)], top_k=5)
        assert isinstance(page, list)
        counts["searches"] += 1
        n += 1


threads = [
    threading.Thread(target=guarded("save", save_loop), daemon=True),
    threading.Thread(target=guarded("add", add_loop), daemon=True),
    threading.Thread(target=guarded("remove", remove_loop), daemon=True),
    threading.Thread(target=guarded("search", search_loop), daemon=True),
]
for t in threads:
    t.start()
# Every loop runs until the churn has lasted `seconds` and each has completed
# an operation.
started = time.monotonic()
while not stop.is_set() and (
    time.monotonic() - started < seconds or min(counts.values()) < 1
):
    time.sleep(0.2)
stop.set()
for t in threads:
    t.join(timeout=60)
assert not any(t.is_alive() for t in threads), "a worker never came back"
assert not failures, "; ".join(failures)
assert min(counts.values()) >= 1, counts

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
                       ("saves", "adds", "removes", "searches")))
'''


@pytest.mark.parametrize("probe_mode", ["raw", "quantized"])
def test_save_never_deadlocks_or_tears_against_mutation(tmp_path, probe_mode):
    """save loops beside adds, removes and searches without freezing or tearing.

    Two failures are in scope. A lock order inversion in the save path would
    hang the child, which the parent timeout detects, and the registry of a
    build with debug assertions refuses at once. A save reading the mappings
    and the stores at different instants would write a directory whose id
    map names records the stores do not hold, which the child's reload
    assertions detect. The child churns until every loop has completed an
    operation.
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
    saves, adds, removes, searches = map(int, summary[0].split()[1:5])
    seen = f"saves={saves} adds={adds} removes={removes} searches={searches}"

    # The child's own targets, which it runs until it reaches. The three
    # mutating loops share one mutex, so none is held to more than one.
    assert saves >= 1, f"the save loop never completed a save. {seen}"
    assert adds >= 1, f"the add loop never completed an add. {seen}"
    assert removes >= 1, f"the remove loop never removed a record. {seen}"
    assert searches >= 1, f"the search loop never completed a search. {seen}"


# ============================================================================
# The columns and the bound, against a concurrent writer
# ============================================================================
#
# Nothing above this line declares `indexed_fields`, so every probe in this file
# ran against an index whose column store holds nothing and whose filtered
# search takes the metadata walk. That put the whole column and bound machinery
# outside the suite.
#
# A filtered search on a declared field takes the columns guard, and a second
# acquisition of that guard on the same thread is the recursive acquisition the
# declared lock order forbids. A probe that declares no field never reaches it.
#
# So these four probes take the three selection paths and the rebuild, each
# against a writer.
#
# | Probe | Filter | What the search takes |
# | --- | --- | --- |
# | `declared` | every field declared | the exact bitmap, no metadata read |
# | `bounded` | one declared, one not | the declared bound, then the metadata of what it names |
# | `undeclared` | no declared field | the warning path, then the whole metadata walk |
# | `rebuild` | every field declared | the same as `declared`, while the graph is replaced under it |
#
# The shape is the `get_stats` probe's, for the same reason: a recursive read
# freezes the interpreter, and no assertion can run inside a frozen one, so the
# parent's timeout is the detector and a hang is reported with a location rather
# than as a bare timeout.
#
# # Why the assertions are invariants rather than an exact page
#
# The index is being written throughout, so no fixed reference page exists. What
# does hold at every instant is that **every record a filtered search returns
# matches the filter**, whichever of the three paths answered it. The three paths
# are three different pieces of code deciding the same predicate, so a bound that
# admitted a record the metadata rejects, or a bitmap that went stale against a
# concurrent insert, shows up here as a returned record whose own metadata fails
# the test. A record removed between the search and the fetch simply does not
# come back, which is not a failure.
#
# The child churns until it has run for COLUMN_PROBE_SECONDS and every loop has
# completed an operation, a filtered search having returned a page and a record
# of it having been checked against the filter. That a filtered search answers
# inside a held write, under each of the three paths, and that every record it
# returns matches its filter, is held at a pause in `collection::close_up_tests`
# in the index crate.

COLUMN_PROBE_SECONDS = 8.0
COLUMN_PROBE_TIMEOUT_S = 180.0

COLUMN_PROBE_SCRIPT = '''
import sys
import threading
import time
from collections import deque

import numpy as np
from zeusdb_vector_database import VectorDatabase

mode = sys.argv[1]
seconds = float(sys.argv[2])
DIM = 16
BATCH = 120
SEED_COUNT = 1200
CATS = ["alpha", "beta", "gamma"]
rng = np.random.default_rng(9799)

# Two fields declared and one left out, which is what makes all three selection
# paths reachable on one index.
index = VectorDatabase().create(
    "hnsw", dim=DIM, expected_size=40000, indexed_fields=["cat", "flag"],
)
assert list(index.indexed_fields) == ["cat", "flag"], index.indexed_fields


def meta(n):
    return {"cat": CATS[n % 3], "flag": (n % 2 == 0), "rank": n % 500}


seeds = rng.standard_normal((SEED_COUNT, DIM)).astype(np.float32)
index.add({
    "ids": ["seed_%d" % i for i in range(SEED_COUNT)],
    "embeddings": seeds,
    "metadatas": [meta(i) for i in range(SEED_COUNT)],
})

FILTERS = {
    # Selection::Exact. Both fields have a column, so the answer comes from the
    # bitmaps and no record's metadata is read to decide it.
    "declared": {"cat": "beta", "flag": True},
    # Selection::Narrowed. `cat` bounds the candidates and `rank`, which has no
    # column, decides among them.
    "bounded": {"cat": "beta", "rank": {"lt": 100}},
    # Selection::Whole. No field in the filter has a column, so this is the
    # warning path and the walk.
    "undeclared": {"rank": {"lt": 100}},
    "rebuild": {"cat": "beta", "flag": True},
}


def admits(record):
    if mode == "bounded":
        return record.get("cat") == "beta" and record.get("rank", 10**9) < 100
    if mode == "undeclared":
        return record.get("rank", 10**9) < 100
    return record.get("cat") == "beta" and record.get("flag") is True


stop = threading.Event()
counts = {"searches": 0, "pages": 0, "hits": 0, "adds": 0, "removes": 0,
          "rebuilds": 0}
# The counts each loop runs until it reaches, the writer's being its own.
TARGETS = ["searches", "pages", "hits"] + (
    ["rebuilds"] if mode == "rebuild" else ["adds", "removes"]
)
failures = []
pending = deque()
pending_lock = threading.Lock()
queries = rng.standard_normal((32, DIM)).astype(np.float32)


def guarded(name, body):
    def run():
        try:
            body()
        except BaseException as exc:
            failures.append("%s: %s: %s" % (name, type(exc).__name__, exc))
            stop.set()
    return run


def search_loop():
    n = 0
    while not stop.is_set():
        query = queries[n % len(queries)]
        n += 1
        page = index.search(query, top_k=10, filter=FILTERS[mode])
        counts["searches"] += 1
        if not page:
            continue
        counts["pages"] += 1
        ids = [hit["id"] for hit in page]
        # A record a concurrent removal took between the search and this fetch
        # is simply absent, which is not a failure. What would be one is a
        # record that is still here and does not match the filter that named it.
        for record in index.get_records(ids, return_vector=False):
            counts["hits"] += 1
            if not admits(record["metadata"]):
                failures.append(
                    "%s returned %s with %r" % (mode, record["id"], record["metadata"])
                )
                stop.set()
                return


def add_loop():
    n = 0
    while not stop.is_set():
        base = SEED_COUNT + n * BATCH
        ids = ["churn_%d" % (base + i) for i in range(BATCH)]
        vectors = rng.standard_normal((BATCH, DIM)).astype(np.float32)
        index.add({
            "ids": ids,
            "embeddings": vectors,
            "metadatas": [meta(base + i) for i in range(BATCH)],
        })
        with pending_lock:
            pending.extend(ids)
        counts["adds"] += 1
        n += 1


def remove_loop():
    # `remove_point` writes the columns under the same guard that writes
    # `vector_metadata`, so this is the write side of what the search reads.
    while not stop.is_set():
        with pending_lock:
            rid = pending.popleft() if pending else None
        if rid is None:
            time.sleep(0.001)
            continue
        index.remove_point(rid)
        counts["removes"] += 1


def rebuild_loop():
    # Replaces the graph outright under the mutation lock while searches read
    # it. `rebuild` offers no concurrency of its own, by construction.
    degree = 12
    while not stop.is_set():
        degree = 16 if degree == 12 else 12
        index.rebuild(m=degree)
        counts["rebuilds"] += 1


threads = [threading.Thread(target=guarded("search", search_loop), daemon=True)]
if mode == "rebuild":
    threads.append(threading.Thread(target=guarded("rebuild", rebuild_loop), daemon=True))
else:
    threads.append(threading.Thread(target=guarded("add", add_loop), daemon=True))
    threads.append(threading.Thread(target=guarded("remove", remove_loop), daemon=True))
for t in threads:
    t.start()
# Every loop runs until the churn has lasted `seconds` and each of its counts
# has reached one.
started = time.monotonic()
while not stop.is_set() and (
    time.monotonic() - started < seconds or min(counts[k] for k in TARGETS) < 1
):
    time.sleep(0.2)
stop.set()
for t in threads:
    t.join(timeout=30)
assert not any(t.is_alive() for t in threads), "a worker never came back"
assert not failures, failures[:4]
assert min(counts[k] for k in TARGETS) >= 1, counts

print("OK " + " ".join(str(counts[k]) for k in
                       ("searches", "pages", "hits", "adds", "removes",
                        "rebuilds")))
'''


@pytest.mark.parametrize(
    "probe_mode", ["declared", "bounded", "undeclared", "rebuild"]
)
def test_a_filtered_search_never_deadlocks_against_a_writer(tmp_path, probe_mode):
    """Each selection path loops beside a writer without freezing or lying.

    Two failures are in scope. A recursive read on the columns guard, or an
    inversion between it and the guards around it, hangs the child, which the
    parent timeout detects. A bound or a bitmap that goes stale against a
    concurrent insert returns a record the filter does not match, which the
    child's own invariant detects.
    """
    script = tmp_path / "column_concurrency_probe.py"
    script.write_text(COLUMN_PROBE_SCRIPT, encoding="utf-8")
    try:
        child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run(
            [sys.executable, str(script), probe_mode, str(COLUMN_PROBE_SECONDS)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=COLUMN_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"a {probe_mode} filtered search deadlocked against a writer: the "
            f"probe froze and was killed after {COLUMN_PROBE_TIMEOUT_S:.0f}s"
        )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    summary = [line for line in result.stdout.splitlines() if line.startswith("OK ")]
    assert len(summary) == 1, result.stdout[-2000:]
    searches, pages, hits, adds, removes, rebuilds = map(
        int, summary[0].split()[1:7]
    )
    seen = (f"searches={searches} pages={pages} hits={hits} adds={adds} "
            f"removes={removes} rebuilds={rebuilds}")

    # The child's own targets, which it runs until it reaches. The writer holds
    # one mutex and the search loop runs beside it, so none is held to more
    # than one.
    assert searches >= 1, f"the search loop never completed a search. {seen}"
    assert pages >= 1, f"no filtered search returned anything. {seen}"
    assert hits >= 1, f"no returned record was checked against the filter. {seen}"
    if probe_mode == "rebuild":
        assert rebuilds >= 1, f"the rebuild loop never rebuilt. {seen}"
    else:
        assert adds >= 1, f"the add loop never completed an add. {seen}"
        assert removes >= 1, f"the remove loop never removed a record. {seen}"
