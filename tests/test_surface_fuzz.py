"""No call on the Python surface panics the engine or stops an index.

A seeded sequence of calls on small indexes of each kind: raw under the four
metrics, one with a declared field and a sparse space and one with a text
space, a product quantized one and a scalar quantized one, each trained before
the sequence. Each call is drawn from `add()` in its five shapes, `search()`,
`query()`, `get_records()`, `list()`, `count()`, the removals,
`update_metadata()` and `add_metadata()`, and each argument is drawn as a valid
value, a value at an edge, or a value of another type: widths, `top_k`,
`ef_search`, empty and large lists, NaN and infinity, and metadata of every JSON
type nested to and past the conversion's limit.

Three calls in four draw every argument valid, and in the rest each argument
is valid three times in four, so most calls run through the engine and the
rest meet the binding's refusals, one argument or several. Every call returns
or raises
`ValueError`, `TypeError`, `KeyError`, `RuntimeError` or `OverflowError`, the
classes the binding and its argument conversion raise, and a `RuntimeError`
never carries the words of a panic, stopped or not. After every call the index
answers `len()` and a search, and an index a removal has left with fewer than
100 records takes 300 more outside the count. Where `add()` returns, every id
it reports is held, and an id it does not report that was not held before the
call is not held after it. Every kind of call returns at least once, and at
least one is refused.

Each kind's sequence runs in a process of its own, whose committed memory the
system holds to 1 GiB beyond what it held when the sequence began, so a call
that asks for more is refused before anything is reserved and ends that
process, which fails the test with its words.

`ZEUSDB_SURFACE_CALLS` sets the calls a run makes over every kind together,
for a deep run.
"""
import math
import os
import random
import subprocess
import sys

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

DIM = 8

#: The calls the run makes over every kind together.
CALLS = int(os.environ.get("ZEUSDB_SURFACE_CALLS", "1000"))

#: The exception classes a call may raise.
ALLOWED = (ValueError, TypeError, KeyError, RuntimeError, OverflowError)

#: The words of a panic inside the engine, in a call's refusal.
PANIC_WORDS = ("failed inside the engine", "This index stopped")

#: Each index kind, as the keyword arguments `create()` takes beside `dim`.
KINDS = {
    "raw-l2-sparse": {"space": "l2", "indexed_fields": ["cat"], "sparse": {"name": "terms"}},
    "raw-cosine-text": {"space": "cosine", "sparse": {"tokenizer": "simple"}},
    "raw-dot": {"space": "dot"},
    "raw-l1": {"space": "l1", "indexed_fields": ["cat", "price"]},
    "pq-l2": {
        "space": "l2",
        "quantization_config": {"type": "pq", "subvectors": 4, "bits": 4, "training_size": 1000},
    },
    "int8-cosine": {
        "space": "cosine",
        "quantization_config": {"type": "int8", "training_size": 1000},
    },
}

#: Numbers at the edges of every width the conversion takes.
EDGE_NUMBERS = [0, -1, 1, 2**31 - 1, 2**31, 2**63 - 1, 2**63, 2**64, -(2**63) - 1, 10**30,
                0.5, -0.0, 1e308, -1e308, 1e-320, float("nan"), float("inf"), float("-inf")]

#: Strings sharing prefixes, ASCII and not, and a long one.
STRINGS = ["", "a", "ab", "b", "ü", "日本", "x" * 5000, "\u0000", "$and"]


def deep(depth):
    """A mapping nested `depth` times."""
    value = {"leaf": 1}
    for _ in range(depth):
        value = {"in": value}
    return value


def value(rng, depth=2):
    """A value of any JSON type, or of a type JSON has no form for."""
    roll = rng.randrange(12)
    if roll == 0:
        return None
    if roll == 1:
        return rng.random() < 0.5
    if roll in (2, 3):
        return rng.choice(EDGE_NUMBERS)
    if roll == 4:
        return rng.choice(STRINGS)
    if roll == 5 and depth > 0:
        return [value(rng, depth - 1) for _ in range(rng.randrange(4))]
    if roll == 6 and depth > 0:
        return {rng.choice(["cat", "price", "tags", "x"]): value(rng, depth - 1)
                for _ in range(rng.randrange(3))}
    if roll == 7:
        return deep(rng.choice([1, 127, 128, 129, 400]))
    if roll == 8:
        return rng.choice([b"bytes", (1, 2), {1, 2}, object(), 3 + 4j])
    return rng.randrange(-5, 50)


#: Whether the call being drawn takes every argument valid.
WHOLE = {"valid": False}


def valid(rng):
    """Whether an argument is drawn valid: always in a wholly valid call, and
    three times in four in any other."""
    return rng.random() < 0.75 or WHOLE["valid"]


#: The keys a record's metadata takes, a group key and one that is not ASCII
#: among them.
KEYS = ["cat", "price", "tags", "text", "x", "$or", "ü"]


def metadata(rng):
    """Metadata for one record: three times in four valid values under the
    keys above, and otherwise a value that is not a mapping, a key that is
    not a string, the empty key, or values of any type at any depth."""
    if valid(rng):
        return {key: rng.choice(["a", "b", 3, 2.5, True, None, ["x", "y"]])
                for key in rng.sample(KEYS, rng.randrange(4))}
    roll = rng.randrange(4)
    if roll == 0:
        return value(rng)
    if roll == 1:
        return {1: "a key that is not a string"}
    if roll == 2:
        return {"": value(rng)}
    return {key: value(rng) for key in rng.sample(KEYS, 1 + rng.randrange(3))}


def vector(rng, dim=DIM):
    """A vector of the declared width three times in four, and otherwise of
    another width, holding a value that is not finite, or not a vector."""
    if valid(rng):
        return [rng.uniform(-1, 1) for _ in range(dim)]
    roll = rng.randrange(5)
    if roll == 0:
        dim = rng.choice([0, 1, DIM - 1, DIM + 1, 4096])
    if roll == 1:
        values = [rng.uniform(-1, 1) for _ in range(dim)]
        values[rng.randrange(max(dim, 1))] = rng.choice([math.nan, math.inf, -math.inf])
        return values
    if roll == 2:
        return rng.choice([None, "text", 3, [[0.1] * dim], ["a"] * dim, {}])
    if roll == 3:
        return np.zeros(dim, dtype=rng.choice([np.float64, np.int32, np.float16]))
    if roll == 4:
        return [0.0] * dim
    return [rng.uniform(-1, 1) for _ in range(dim)]


def sparse(rng):
    """A sparse half: valid three times in four, and otherwise with
    dimensions out of order, repeated or negative, values that are not
    finite, lists of unequal length, or not a mapping."""
    dims = sorted(rng.sample(range(50), rng.randrange(1, 5)))
    values = [float(rng.randrange(1, 4)) for _ in dims]
    if valid(rng):
        return {"dims": dims, "values": values}
    roll = rng.randrange(5)
    if roll == 0:
        dims = list(reversed(dims)) + dims[:1]
    elif roll == 1:
        values[0] = rng.choice([math.nan, math.inf, -1.0, 0.0])
    elif roll == 2:
        values = values[:-1]
    elif roll == 3:
        dims[0] = rng.choice([-1, 2**32, 2**64])
    elif roll == 4:
        return rng.choice([None, "dims", [1, 2], {"dims": "x"}])
    return {"dims": dims, "values": values}


def text(rng):
    """A record's or a query's text, or a value that is not one."""
    words = ["the", "quick", "Brown", "fox", "ü", "日本語", "a-b", "", "x" * 300]
    if rng.randrange(10) == 0:
        return rng.choice([None, 3, ["words"], b"bytes"])
    return " ".join(rng.choice(words) for _ in range(rng.randrange(0, 6)))


def record_id(rng, held):
    """An id the index holds, a new one, or one that is not a string."""
    roll = rng.randrange(10)
    if roll < 3 and held:
        return rng.choice(held)
    if roll == 3:
        return rng.choice([None, 7, "", "x" * 3000, "ü", ["id"]])
    return f"r{rng.randrange(10**9)}"


#: Conditions a filter's leaf takes, each with a target of its operator's
#: type.
VALID_CONDITIONS = [("eq", "a"), ("ne", "b"), ("gt", 3), ("lte", 50), ("in", ["a", "c"]),
                    ("nin", [1, 2]), ("contains", "a"), ("startswith", "a"), ("any", ["x"]),
                    ("exists", True), ("is_null", False)]


def filter_(rng):
    """A filter: three times in four none or a valid leaf, alone or under
    `$or` or `$not`, and otherwise any shape the language takes or refuses,
    with a target of any type."""
    if valid(rng):
        roll = rng.randrange(4)
        if roll == 0:
            return None
        op, target = rng.choice(VALID_CONDITIONS)
        leaf = {rng.choice(["cat", "price", "tags", "x", "$price"]): {op: target}}
        if roll == 1:
            return {"$or": [leaf, {"cat": "a"}]}
        if roll == 2:
            return {"$not": leaf}
        return leaf
    roll = rng.randrange(10)
    if roll < 2:
        return rng.choice([{}, {"$and": []}, {"$or": "x"}, {"cat": {"between": 1}}, [1], "cat"])
    field = rng.choice(["cat", "price", "tags", "x", "$price"])
    op = rng.choice(["eq", "ne", "gt", "gte", "lt", "lte", "contains", "startswith",
                     "endswith", "in", "nin", "any", "all", "exists", "is_null"])
    target = value(rng, 1)
    if op in ("in", "nin", "any", "all") and rng.random() < 0.7:
        target = [value(rng, 0) for _ in range(rng.randrange(4))]
    leaf = {field: {op: target}}
    if roll == 2:
        return {"$or": [leaf, {"cat": rng.choice(STRINGS)}]}
    if roll == 3:
        return {"$not": leaf}
    return leaf


def records_payload(rng, kind, held, count):
    """The records of one `add()`, in one of its five shapes, and the ids it
    names where it names them."""
    texts = KINDS[kind].get("sparse", {}).get("tokenizer") is not None
    sparse_kind = "sparse" in KINDS[kind] and not texts
    shape = rng.randrange(5)

    def extra(entry):
        if texts and rng.random() < 0.7:
            entry["text"] = text(rng)
        if sparse_kind and rng.random() < 0.7:
            entry["sparse"] = sparse(rng)
        if rng.random() < 0.05:
            entry[rng.choice(["sparse", "text"])] = rng.choice([sparse(rng), text(rng)])

    if shape == 0:
        entry = {"id": record_id(rng, held), rng.choice(["values", "vector"]): vector(rng),
                 "metadata": metadata(rng)}
        extra(entry)
        return entry, [entry["id"]]
    if shape == 1:
        ids = [record_id(rng, held) for _ in range(count)]
        entry = {"ids": ids, rng.choice(["embeddings", "vectors"]): [vector(rng) for _ in ids],
                 "metadatas": [metadata(rng) for _ in ids]}
        if rng.random() < 0.1:
            entry["metadatas"] = entry["metadatas"][:-1]
        if texts and rng.random() < 0.5:
            entry["texts"] = [text(rng) for _ in ids]
        return entry, ids
    if shape == 2:
        entries = []
        for _ in range(count):
            entry = {"id": record_id(rng, held), rng.choice(["values", "vector"]): vector(rng),
                     "metadata": metadata(rng)}
            extra(entry)
            entries.append(entry)
        return entries, [entry["id"] for entry in entries]
    if shape == 3:
        rows = rng.choice([count, 0])
        width = DIM if rng.random() < 0.9 else DIM + 1
        array = np.asarray([[rng.uniform(-1, 1) for _ in range(width)] for _ in range(rows)],
                           dtype=np.float32).reshape(rows, width)
        if rows and rng.random() < 0.1:
            array[0, 0] = np.nan
        return array, None
    single = np.asarray(vector(rng) if rng.random() < 0.2 else
                        [rng.uniform(-1, 1) for _ in range(DIM)])
    return single, None


def query_arms(rng, kind):
    """The arms of one `query()`: valid ones of the kinds the index holds,
    and arms it refuses."""
    texts = KINDS[kind].get("sparse", {}).get("tokenizer") is not None
    sparse_kind = "sparse" in KINDS[kind] and not texts
    roll = rng.randrange(8)
    if roll == 0:
        return rng.choice([[], None, "vector", [{}], [{"vector": vector(rng)}] * 9,
                           [{"bogus": 1}], [{"text": text(rng)}], [{"sparse": sparse(rng)}]])
    arms = [{"vector": vector(rng)}]
    if texts:
        arms.append({"text": text(rng)})
    if sparse_kind:
        arms.append({"sparse": sparse(rng)})
    if (texts or sparse_kind) and rng.random() < 0.3:
        arms.append({"text": text(rng)} if texts else {"sparse": sparse(rng)})
    return arms


def edge_count(rng):
    """A count argument: valid three times in four, and otherwise at an
    edge or of another type."""
    if valid(rng):
        return rng.choice([1, 3, 10, 50])
    return rng.choice([0, -1, 2**31, 2**63, 10**30, None, "5", 2.5, float("nan")])


def either(rng, good, bad):
    """One of `good` three times in four, and otherwise one of `bad`."""
    return rng.choice(good) if valid(rng) else rng.choice(bad)


def held_ids(index):
    """Every id the index holds, as `list()` reports them."""
    return [entry[0] if isinstance(entry, (list, tuple)) else entry
            for entry in index.list(number=len(index) + 1)]


def call(rng, index, kind, held):
    """One call drawn at random, wholly valid three times in four, with its
    arguments. Returns what it was."""
    WHOLE["valid"] = rng.random() < 0.75
    roll = rng.randrange(100)
    if roll < 26:
        payload, ids = records_payload(rng, kind, held, rng.choice([1, 2, 3, 40]))
        overwrite = rng.random() < 0.5
        before = {i: index.contains(i) for i in ids or [] if isinstance(i, str)}
        result = index.add(payload, overwrite=overwrite)
        reported = set(result.ids)
        for i in reported:
            assert index.contains(i), f"add() reported {i!r} and does not hold it"
        for i, was in before.items():
            if i not in reported and not was:
                assert not index.contains(i), f"add() refused {i!r} and holds it"
        return "add"
    if roll < 41:
        index.search(vector(rng), filter=filter_(rng), top_k=edge_count(rng),
                     ef_search=either(rng, [None, 10, 50, 200], [0, 1, -1, 2**40, "x"]),
                     return_vector=rng.random() < 0.5,
                     rerank=either(rng, [None, True, False], ["yes", 2]))
        return "search"
    if roll < 52:
        index.query(query_arms(rng, kind), filter=filter_(rng), top_k=edge_count(rng),
                    fetch=either(rng, [None, 50, 100], [0, 1, -1, 10**12]),
                    fusion=either(rng, [None, "rrf", {"type": "rrf", "k": 60}],
                                  [{"type": "rrf", "k": -1}, "linear", 3]))
        return "query"
    if roll < 60:
        ids = rng.choice([[record_id(rng, held) for _ in range(rng.randrange(5))],
                          record_id(rng, held), None, [], ["r"] * 2000])
        index.get_records(ids, return_vector=rng.random() < 0.5, strict=rng.random() < 0.3)
        return "get_records"
    if roll < 68:
        index.list(number=edge_count(rng), offset=either(rng, [0, 1, 5], [-1, 2**63, "a"]),
                   after=either(rng, [None, None, record_id(rng, held)], [3, b"id"]))
        return "list"
    if roll < 76:
        index.count(filter_(rng))
        return "count"
    if roll < 88:
        which = rng.randrange(4)
        if which == 0:
            index.remove_point(record_id(rng, held))
        elif which == 1:
            index.remove_points([record_id(rng, held) for _ in range(rng.randrange(6))])
        elif which == 2:
            index.delete(ids=rng.choice([None, [record_id(rng, held)], record_id(rng, held)]),
                         where=rng.choice([None, filter_(rng)]))
        else:
            index.remove_where(filter_(rng))
        return "remove"
    if roll < 95:
        index.update_metadata(record_id(rng, held), metadata(rng))
        return "update_metadata"
    index.add_metadata(either(rng, [{"index": "meta"}, {"owner": "ü"}],
                              [metadata(rng), {"k": value(rng)}]))
    return "add_metadata"


def base_records(kind, prefix, count):
    """`count` valid records named from `prefix`, each with a field and a
    sparse or text half where the space takes one."""
    options = KINDS[kind]
    rng = random.Random(f"{prefix}-{kind}")
    texts = options.get("sparse", {}).get("tokenizer") is not None
    sparse_kind = "sparse" in options and not texts
    records = []
    for i in range(count):
        entry = {"id": f"{prefix}{i}", "values": [rng.uniform(-1, 1) for _ in range(DIM)],
                 "metadata": {"cat": rng.choice(["a", "b", "c"]), "price": rng.randrange(100)}}
        if texts:
            entry["text"] = " ".join(rng.choice(["the", "quick", "fox", "dog"]) for _ in range(3))
        if sparse_kind:
            dims = sorted(rng.sample(range(50), 3))
            entry["sparse"] = {"dims": dims, "values": [1.0, 2.0, 3.0]}
        records.append(entry)
    return records


def a_trained_index(kind):
    """An index of `kind` holding a thousand records, which trains it where
    it declares a quantizer."""
    options = KINDS[kind]
    index = VectorDatabase().create("hnsw", dim=DIM, expected_size=4000, **options)
    result = index.add(base_records(kind, "base", 1000))
    assert result.total_errors == 0, result.errors[:3]
    if "quantization_config" in options:
        assert index.is_quantized(), f"{kind}: the base trains the index"
    return index


#: The memory a sequence's process may commit beyond what it held when the
#: sequence began, 1 GiB, held by the system's own limit on the process.
CAP = 1 << 30

#: The seconds a sequence's process may take, after which the test fails.
TIMEOUT = 120 + CALLS // 50


def hold_to_cap():
    """Hold this process's committed memory to CAP beyond what it holds now,
    by a job object on Windows and by RLIMIT_DATA elsewhere. A request past
    it is refused by the system before anything is reserved. Returns whether
    the limit is in force, which it is not on a system honouring neither."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                    "PrivateUsage")]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic),
                        ("IoInfo", ctypes.c_ulonglong * 6)] + [
                (name, ctypes.c_size_t) for name in (
                    "ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed",
                    "PeakJobMemoryUsed")]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        counters = Counters(cb=ctypes.sizeof(Counters))
        process = kernel32.GetCurrentProcess()
        assert kernel32.K32GetProcessMemoryInfo(process, ctypes.byref(counters),
                                                counters.cb), ctypes.get_last_error()
        job = kernel32.CreateJobObjectW(None, None)
        assert job, ctypes.get_last_error()
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        limits.ProcessMemoryLimit = counters.PrivateUsage + CAP
        assert kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits),
                                                ctypes.sizeof(limits)), ctypes.get_last_error()
        assert kernel32.AssignProcessToJobObject(job, process), ctypes.get_last_error()
        return True
    try:
        import resource
    except ImportError:
        return False
    with open("/proc/self/status", encoding="ascii") as status:
        data = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmData:"))
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    soft = data + CAP if hard == resource.RLIM_INFINITY else min(data + CAP, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))
    return True


def run_sequence(kind):
    """The seeded sequence of calls on a trained index of `kind`, raising
    AssertionError where a call breaks the module's rules."""
    index = a_trained_index(kind)
    rng = random.Random(f"surface-{kind}")
    calls = max(1, CALLS // len(KINDS))
    raised = {}
    returned = set()
    held = held_ids(index)
    probe = [0.1] * DIM
    refills = 0
    for step in range(calls):
        try:
            returned.add(call(rng, index, kind, held))
        except ALLOWED as error:
            message = str(error)
            assert not any(words in message for words in PANIC_WORDS), (
                f"{kind}, call {step}: a panic reached the caller: {message}"
            )
            raised[type(error).__name__] = raised.get(type(error).__name__, 0) + 1
        if len(index) < 100:
            refills += 1
            added = index.add(base_records(kind, f"refill{refills}-", 300))
            assert added.total_errors == 0, added.errors[:3]
        held = held_ids(index)
        index.search(probe, top_k=3)
    every = {"add", "search", "query", "get_records", "list", "count", "remove",
             "update_metadata", "add_metadata"}
    assert returned == every, f"{kind}: no call of {sorted(every - returned)} returned"
    assert raised, f"{kind}: no call was refused, so the edges were not reached"


@pytest.mark.parametrize("kind", list(KINDS))
def test_no_call_on_the_surface_panics_or_stops_the_index(kind):
    """The sequence of `kind`, in a process of its own held to CAP, which
    passes where that process ends with nothing refused and nothing broken."""
    child = subprocess.run([sys.executable, __file__, kind], capture_output=True, text=True,
                           timeout=TIMEOUT)
    assert child.returncode == 0, (
        f"{kind}: the sequence's process ended with {child.returncode}\n"
        f"{child.stdout[-4000:]}\n{child.stderr[-4000:]}")
    assert "held to the cap" in child.stdout or sys.platform not in ("win32", "linux"), (
        child.stdout[-4000:])


if __name__ == "__main__":
    held = hold_to_cap()
    run_sequence(sys.argv[1])
    print("the sequence ran held to the cap" if held else "the sequence ran with no cap")
