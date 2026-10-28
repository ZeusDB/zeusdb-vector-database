"""The interpreter rule every call on an index keeps.

A call that does more than a constant amount of work, or that may wait for a
writer, runs that part with the interpreter released, so other Python threads
run while it works or waits. A read of a guard or two keeps the interpreter
while every guard it takes is free, and releases it only to wait where a
writer holds one. A call of constant work keeps it throughout.

Each check stops a thread at a named point with `_engine._Pause` and has
another Python thread release it there. Every method of a Python class runs
with the interpreter held, so a release that finds the stopped thread waiting
shows that the call it stopped did not hold the interpreter at that point.
Where the call holds it, the releasing thread cannot run until the pause gives
up at its limit, and the release then finds no thread waiting. Every wait in
this module ends at a pause's limit or at a join's, so no check hangs the
suite.

`_Pause` exists only in an extension built with the binding's `test-support`
feature,

    maturin develop --release --features test-support

and on any other build every check here fails with words that say so.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from zeusdb_vector_database import VectorDatabase, _engine

Pause = getattr(_engine, "_Pause", None)

MISSING = (
    "zeusdb_vector_database._engine has no _Pause, so this extension was built "
    "without the binding's test-support feature. Build it with "
    "`maturin develop --release --features test-support` to run these checks."
)

# The longest a pause holds its thread and the longest a wait for one lasts,
# in seconds. It bounds how long a check takes on a build that holds the
# interpreter where the rule says to release it.
LIMIT = 30.0
# The longest a check waits for one of its threads to end, in seconds. Each
# thread's own waits end at a pause's limit, and no thread waits on more than
# two pauses.
JOIN_LIMIT = 3 * LIMIT

DIM = 8
RECORDS = 200
TOP_K = 10
TRAINING_SIZE = 1000
PQ = {"type": "pq", "subvectors": 4, "bits": 4, "training_size": TRAINING_SIZE}
NEW_ID = "written"


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


def records(prefix, n, seed):
    """`n` records with ids `prefix0` upwards, the same vectors for the same
    seed, and an integer `kind` in their metadata."""
    vectors = np.random.default_rng(seed).random((n, DIM), dtype=np.float32)
    return {
        "ids": [f"{prefix}{i}" for i in range(n)],
        "vectors": vectors,
        "metadatas": [{"kind": i % 5} for i in range(n)],
    }


def plain_index():
    """An index of `RECORDS` records with `kind` declared as an indexed field
    and one index-level key."""
    index = VectorDatabase().create(
        "hnsw", dim=DIM, expected_size=4000, m=8, ef_construction=40,
        indexed_fields=["kind"],
    )
    result = index.add(records("r", RECORDS, seed=1))
    assert result.total_inserted == RECORDS, result.errors
    index.add_metadata({"owner": "checks"})
    return index


def quantized_index(collected):
    """A product-quantized index holding `collected` records, with `kind`
    declared as an indexed field and one index-level key. It is trained and
    quantized where `collected` reaches `TRAINING_SIZE`."""
    index = VectorDatabase().create(
        "hnsw", dim=DIM, expected_size=4000, m=8, ef_construction=40,
        indexed_fields=["kind"], quantization_config=PQ,
    )
    result = index.add(records("q", collected, seed=2))
    assert result.total_inserted == collected, result.errors
    index.add_metadata({"owner": "checks"})
    return index


def release_once_reached(pause):
    """Wait for `pause` to be reached, release it, and give the name of what
    reached it and whether the release found a thread waiting."""
    reached = pause.wait_reached()
    return reached, pause.release()


# ------------------------------------------------------------
# A call releases the interpreter for its work
# ------------------------------------------------------------


def test_get_stats_walks_with_the_interpreter_released():
    """`get_stats()` walks the index in the released part of its call. The
    main thread stops there, and a second Python thread runs and releases it
    while it waits, then the call returns its report."""
    pause = new_pause()
    index = plain_index()
    releaser = Worker(lambda: release_once_reached(pause)).start()
    try:
        pause.arm_next_call()
        stats = index.get_stats()
    finally:
        Pause.disarm()
        pause.release()
        releaser.finish()
    reached, released = releaser.result()
    assert reached == "get_stats()", (
        f"get_stats() never reached the released part of its call, and the "
        f"pause saw {reached!r}"
    )
    assert released, (
        "get_stats() held the interpreter at its pause, so no other Python "
        f"thread ran before the pause gave up after {LIMIT} seconds"
    )
    assert stats["total_vectors"] == str(RECORDS)


def stop_an_add(index, batch, point):
    """Add `batch` on a second thread stopped at `point`, search the index
    from this thread while it waits there, and release it.

    Gives what reached the pause, whether the release found the add waiting,
    the search's page, and the add's result.
    """
    pause = new_pause()

    def add():
        pause.arm_at(point)
        try:
            return index.add(batch)
        finally:
            Pause.disarm()

    adder = Worker(add).start()
    page = None
    released = False
    try:
        reached = pause.wait_reached()
        if reached is not None:
            page = index.search(batch["vectors"][0], top_k=TOP_K)
        released = pause.release()
    finally:
        pause.release()
        adder.finish()
    return reached, released, page, adder.result()


def test_add_inserts_with_the_interpreter_released():
    """`add()` inserts with the interpreter released. Stopped after its commit
    and before its installs, it lets this thread search the index and release
    it, the page holds none of the batch, and the add then inserts the whole
    batch."""
    index = plain_index()
    batch = records("b", 50, seed=3)
    reached, released, page, result = stop_an_add(index, batch, "add:before_installs")
    assert reached == "add:before_installs", (
        f"add() never reached add:before_installs, and the pause saw {reached!r}"
    )
    assert released, (
        "add() held the interpreter at its pause, so no other Python thread "
        f"ran before the pause gave up after {LIMIT} seconds"
    )
    assert 0 < len(page) <= TOP_K
    assert not {hit["id"] for hit in page} & set(batch["ids"]), (
        "the search returned a record the stopped add had not installed"
    )
    assert result.total_inserted == len(batch["ids"]), result.errors
    assert len(index) == RECORDS + len(batch["ids"])


def test_a_training_inside_add_runs_with_the_interpreter_released():
    """A training that `add()` sets off runs with the interpreter released.
    Stopped after its codebook is installed and before the graph is rebuilt
    over the codes, it lets this thread search the index and release it, and
    the index is quantized once the add returns."""
    index = quantized_index(TRAINING_SIZE - 50)
    assert not index.is_quantized(), "the index trains inside the add below"
    batch = records("t", 50, seed=4)
    reached, released, page, result = stop_an_add(
        index, batch, "training:before_rebuild"
    )
    assert reached == "training:before_rebuild", (
        "the training inside add() never reached training:before_rebuild, and "
        f"the pause saw {reached!r}"
    )
    assert released, (
        "the training inside add() held the interpreter at its pause, so no "
        f"other Python thread ran before the pause gave up after {LIMIT} seconds"
    )
    assert 0 < len(page) <= TOP_K
    assert result.total_inserted == len(batch["ids"]), result.errors
    assert index.is_quantized()


# ------------------------------------------------------------
# A cheap read releases the interpreter only to wait for a writer
# ------------------------------------------------------------

# The reads whose guard a writer stopped at `insert:after_name` holds, being
# the id store or the columns, and on a trained index the dense space. Each
# names the call as it reaches its pause, the read, and what the read gives
# once the writer's record is in, from the count of records before it.
WAITING_READS = [
    SimpleNamespace(
        name="len", trained=False, operation="len()",
        read=lambda index: len(index),
        holds=lambda value, count: value == count + 1,
    ),
    SimpleNamespace(
        name="contains", trained=False, operation="contains()",
        read=lambda index: index.contains(NEW_ID),
        holds=lambda value, count: value is True,
    ),
    SimpleNamespace(
        name="in", trained=False, operation="contains()",
        read=lambda index: NEW_ID in index,
        holds=lambda value, count: value is True,
    ),
    SimpleNamespace(
        name="info", trained=False, operation="info()",
        read=lambda index: index.info(),
        holds=lambda value, count: f"vectors={count + 1}" in value,
    ),
    SimpleNamespace(
        name="indexed_fields", trained=False, operation="indexed_fields",
        read=lambda index: index.indexed_fields,
        holds=lambda value, count: value == ["kind"],
    ),
    SimpleNamespace(
        name="is_quantized", trained=True, operation="is_quantized()",
        read=lambda index: index.is_quantized(),
        holds=lambda value, count: value is True,
    ),
    SimpleNamespace(
        name="get_storage_mode", trained=True, operation="get_storage_mode()",
        read=lambda index: index.get_storage_mode(),
        holds=lambda value, count: value == "quantized_active",
    ),
]


@pytest.mark.parametrize("case", WAITING_READS, ids=[case.name for case in WAITING_READS])
def test_a_cheap_read_releases_the_interpreter_to_wait_for_a_writer(case):
    """A read whose guard a writer holds waits for the writer with the
    interpreter released.

    A writer thread stops at `insert:after_name` under its four write guards.
    The main thread makes the read, which finds its guard held and stops at
    the start of the released part of its call. A third Python thread
    releases the read and then the writer, and the writer's release finds it
    still waiting, so the read waited for the writer and never held the
    interpreter. The read then returns what the index gives once the
    writer's record is in.
    """
    writer_pause = new_pause()
    read_pause = new_pause()
    index = quantized_index(TRAINING_SIZE) if case.trained else plain_index()
    if case.trained:
        assert index.is_quantized(), "the read needs a trained index"
    count = len(index)

    def write():
        writer_pause.arm_at("insert:after_name")
        try:
            return index.add(
                {"ids": [NEW_ID], "vectors": records("w", 1, seed=5)["vectors"],
                 "metadatas": [{"kind": 0}]}
            )
        finally:
            Pause.disarm()

    def release_both():
        reached = read_pause.wait_reached()
        read_released = read_pause.release()
        writer_released = writer_pause.release()
        return reached, read_released, writer_released

    writer = Worker(write).start()
    releaser = None
    value = None
    try:
        writer_reached = writer_pause.wait_reached()
        if writer_reached is not None:
            releaser = Worker(release_both).start()
            read_pause.arm_next_call()
            value = case.read(index)
    finally:
        Pause.disarm()
        read_pause.release()
        writer_pause.release()
        if releaser is not None:
            releaser.finish()
        writer.finish()
    assert writer_reached == "insert:after_name", (
        f"the writer never stopped under its write guards, and its pause saw "
        f"{writer_reached!r}"
    )
    reached, read_released, writer_released = releaser.result()
    added = writer.result()
    assert reached == case.operation, (
        f"{case.name} never reached the released part of its call while a "
        f"writer held its guard, and the pause saw {reached!r}"
    )
    assert read_released, (
        f"{case.name} held the interpreter at the pause where it waits for the "
        f"writer, so no other Python thread ran before the pause gave up after "
        f"{LIMIT} seconds"
    )
    assert writer_released, (
        f"the writer was not waiting once {case.name} was released, so "
        f"{case.name} held the interpreter while it waited for the writer"
    )
    assert case.holds(value, count), (
        f"{case.name} returned {value!r}, which is not what the index gives "
        f"once the writer's record is in"
    )
    assert added.total_inserted == 1, added.errors


# Every read that keeps the interpreter while its guards are free, and every
# call that keeps it throughout, as (case id, the call).
FREE_READS = [
    ("get_quantization_info", lambda index: index.get_quantization_info()),
    ("get_vector_count", lambda index: index.get_vector_count()),
    ("is_quantized", lambda index: index.is_quantized()),
    ("can_use_quantization", lambda index: index.can_use_quantization()),
    ("get_training_progress", lambda index: index.get_training_progress()),
    ("training_vectors_needed", lambda index: index.training_vectors_needed()),
    ("get_storage_mode", lambda index: index.get_storage_mode()),
    ("journal_path", lambda index: index.journal_path),
    ("identity", lambda index: index.identity),
    ("len", lambda index: len(index)),
    ("in", lambda index: "q0" in index),
    ("indexed_fields", lambda index: index.indexed_fields),
    ("contains", lambda index: index.contains("q0")),
    ("get_metadata", lambda index: index.get_metadata("owner")),
    ("get_all_metadata", lambda index: index.get_all_metadata()),
    ("info", lambda index: index.info()),
]
HELD_CALLS = [
    ("has_quantization", lambda index: index.has_quantization()),
    ("get_space", lambda index: index.get_space()),
    ("is_training_ready", lambda index: index.is_training_ready()),
    ("dim", lambda index: index.dim),
    ("space", lambda index: index.space),
    ("m", lambda index: index.m),
    ("ef_construction", lambda index: index.ef_construction),
    ("expected_size", lambda index: index.expected_size),
    ("recovery", lambda index: index.recovery),
    ("get_performance_info", lambda index: index.get_performance_info()),
]


@pytest.fixture(scope="module")
def still_index():
    """A trained index with an indexed field and an index-level key, which no
    check writes to."""
    index = quantized_index(TRAINING_SIZE)
    assert index.is_quantized()
    return index


@pytest.mark.parametrize(
    "name, call", FREE_READS + HELD_CALLS,
    ids=[name for name, _ in FREE_READS + HELD_CALLS],
)
def test_a_call_with_nothing_writing_keeps_the_interpreter(still_index, name, call):
    """With nothing writing, a read of free guards and a call of constant work
    reach no released part of a call, so each keeps the interpreter. The
    pause's limit is 0, so a call that reached it would not wait, and the
    check rests on no timing."""
    pause = new_pause(limit=0)
    pause.arm_next_call()
    try:
        call(still_index)
    finally:
        Pause.disarm()
    assert pause.reached is None, (
        f"{name} reached the released part of its call, as {pause.reached!r}, "
        f"with nothing writing"
    )
