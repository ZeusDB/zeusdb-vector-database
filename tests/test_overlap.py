"""A write made while a search is in flight completes and does not raise.

Every method of the index takes `&self`, so a write that arrives while a search
holds its borrow of the index takes no exclusive borrow of the Python object.
Each check stops a search at the start of the released part of its call with
`_engine._Pause`, holding that borrow, and makes the write from the main thread
while the search waits there. The pause still holds the search once the write
returns, so the write ran whole inside the search's call, and the search then
answers. Every wait in this module ends at a pause's limit or at a join's, so
no check hangs the suite.

`_Pause` exists only in an extension built with the binding's `test-support`
feature,

    maturin develop --release --features test-support

and on any other build every check here fails with words that say so.
"""

import threading

import numpy as np
import pytest

from zeusdb_vector_database import VectorDatabase, _engine

Pause = getattr(_engine, "_Pause", None)

MISSING = (
    "zeusdb_vector_database._engine has no _Pause, so this extension was built "
    "without the binding's test-support feature. Build it with "
    "`maturin develop --release --features test-support` to run these checks."
)

# The longest a pause holds its search, in seconds, and the longest a check
# waits for its search thread to end.
LIMIT = 30.0
JOIN_LIMIT = 3 * LIMIT

DIM = 32
N = 2000
TOP_K = 10
# The queries a held search carries as one batch.
READ_BATCH = 128
# Writes made inside a held search, for each kind of write.
ROUNDS = 5
PQ_CONFIG = {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000}


def new_pause(limit=LIMIT):
    """A `_Pause` with `limit`, or a failure naming the feature where the
    extension has no `_Pause`."""
    if Pause is None:
        pytest.fail(MISSING, pytrace=False)
    return Pause(limit)


@pytest.fixture(autouse=True)
def disarmed():
    """The main thread is armed nowhere once a check ends, whatever it
    armed."""
    yield
    if Pause is not None:
        Pause.disarm()


def clustered(n, dim, seed):
    """Ten Gaussian clusters of unit vectors."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((10, dim))
    points = centres[rng.integers(0, 10, size=n)] + 0.15 * rng.standard_normal((n, dim))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def build_raw():
    index = VectorDatabase().create("hnsw", dim=DIM, expected_size=N * 8)
    ids = [f"doc_{i}" for i in range(N)]
    assert index.add({"ids": ids, "embeddings": clustered(N, DIM, seed=11)}).is_success()
    return index, ids


def build_quantized():
    index = VectorDatabase().create(
        "hnsw", dim=DIM, expected_size=N * 8, quantization_config=PQ_CONFIG,
    )
    ids = [f"doc_{i}" for i in range(N)]
    assert index.add({"ids": ids, "embeddings": clustered(N, DIM, seed=12)}).is_success()
    assert index.is_quantized(), "the index trains on its first batch"
    return index, ids


class Worker:
    """A thread that runs `work` once and keeps its result or its exception.

    `finish` waits for it within `JOIN_LIMIT` and raises nothing, for a
    `finally` block. `result` then fails where the thread is still running
    and raises what `work` raised, so a failure on the thread fails the check.
    """

    def __init__(self, work):
        self._work = work
        self._result = None
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self._result = self._work()
        except BaseException as exc:  # noqa: BLE001 - raised again by result()
            self._error = exc

    def start(self):
        self._thread.start()
        return self

    def finish(self):
        self._thread.join(timeout=JOIN_LIMIT)

    def result(self):
        assert not self._thread.is_alive(), (
            f"a thread of the check was still running after {JOIN_LIMIT} seconds"
        )
        if self._error is not None:
            raise self._error
        return self._result


def write_inside_a_held_search(index, queries, write):
    """Hold a search of `queries` at the start of the released part of its
    call on a second thread, make `write` from this thread while it waits,
    and release it.

    Gives what reached the pause, whether the pause still held the search once
    the write returned, and the search's pages. A write that raises raises
    here, after the search is released.
    """
    pause = new_pause()

    def search():
        pause.arm_next_call()
        try:
            return index.search(queries, top_k=TOP_K)
        finally:
            Pause.disarm()

    searcher = Worker(search).start()
    reached = None
    held_through = False
    try:
        reached = pause.wait_reached()
        if reached is not None:
            write()
            held_through = pause.state == "waiting"
    finally:
        pause.release()
        searcher.finish()
    return reached, held_through, searcher.result()


def check_round(name, reached, held_through, pages):
    assert reached == "search()", (
        f"the search never reached the released part of its call, and the pause "
        f"saw {reached!r}"
    )
    assert held_through, (
        f"the search was not waiting at its pause once the {name} returned, so the "
        f"{name} did not run inside the search's call"
    )
    assert len(pages) == READ_BATCH
    for page in pages:
        assert 0 < len(page) <= TOP_K


@pytest.mark.parametrize("build", [build_raw, build_quantized], ids=["raw", "quantized"])
def test_add_during_search_does_not_raise(build):
    """An add made while a search is in flight completes rather than raising.

    The search holds its borrow of the index for its whole call, and the add
    takes no exclusive borrow, so the add runs whole inside the search's
    call, and both the add and the search return what they were asked for.
    """
    index, ids = build()
    queries = clustered(READ_BATCH, DIM, seed=303)
    vectors = clustered(64, DIM, seed=404)
    for attempt in range(ROUNDS):
        batch = {"ids": [f"new_{attempt}_{i}" for i in range(64)], "embeddings": vectors}
        added = []
        reached, held_through, pages = write_inside_a_held_search(
            index, queries, lambda: added.append(index.add(batch))
        )
        check_round("add", reached, held_through, pages)
        assert added[0].total_inserted == 64, added[0].errors
    assert len(index) == len(ids) + ROUNDS * 64


def test_remove_metadata_and_compact_during_search_do_not_raise():
    """The other three mutating methods take no exclusive borrow either.

    `remove_point`, `add_metadata` and `compact` each run whole inside a
    search's call. Removal runs first so that `compact` has stranded nodes to
    reclaim and is a real rebuild rather than the no-op it returns on a clean
    graph.
    """
    index, ids = build_raw()
    queries = clustered(READ_BATCH, DIM, seed=505)
    writes = [
        ("remove_point", lambda attempt: index.remove_point(ids[attempt])),
        ("add_metadata", lambda attempt: index.add_metadata({"phase": str(attempt)})),
        ("compact", lambda _attempt: index.compact()),
    ]
    for name, write in writes:
        for attempt in range(ROUNDS):
            reached, held_through, pages = write_inside_a_held_search(
                index, queries, lambda: write(attempt)
            )
            check_round(name, reached, held_through, pages)
    assert not index.contains(ids[0])
    assert len(index) == len(ids) - ROUNDS
    assert index.get_metadata("phase") == str(ROUNDS - 1)
