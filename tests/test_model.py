"""A random operation sequence against a model that says what the index holds.

A mutation path defect is a disagreement between what the index holds and what
the operations performed on it say it should hold. The general form of the check
is a random sequence of operations against an independent statement of the
expected contents, which is what this file is.

The model is a plain Python dictionary and a list. It holds the record set, each
record's vector, each record's metadata, the index level metadata, the ordering
`list()` pages through, and the counter generated ids come from. After every
operation the index is asked for each of those and must agree.

What the model does not hold is ranking. Search is approximate, so no model can
say which records come back or in what order. What it does say about a search is
that every hit is a live record, that no id repeats, that scores do not decrease
and that under a filter every hit satisfies the model's own evaluation of that
filter. Those hold whatever the traversal did.

The generator is seeded and inline, so a failure reproduces from [`MODEL_SEED`]
and the printed step alone. `ZEUSDB_MODEL_SEQUENCES` and `ZEUSDB_MODEL_STEPS`
raise the budget for a soak run, matching what the graph dump fuzzer does with
`ZEUSDB_FUZZ_CASES`.
"""

import os
import tempfile

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

# ============================================================================
# THE BUDGET
# ============================================================================

# The generator's seed, which with the step number is the whole of a failure's
# reproduction. Every sequence draws from `MODEL_SEED ^ sequence_index`.
MODEL_SEED = 0x5EED_0100_C0DE_1CAF

# Sequences per configuration, and steps per sequence.
#
# A budget rather than a target. This runs in the ordinary gate on every commit,
# and a gate nobody can afford to run finds nothing. The four configurations at
# these numbers cost about five seconds, most of it in the two quantized
# configurations, which have to carry more than a thousand records to reach the
# training threshold at all.
#
# **The committed budget is a regression net rather than a search.** Anything it
# finds gets its own named test, so the net does not depend on a later draw
# reaching the same sequence again. Deeper draws are the search, and the soak is
# `ZEUSDB_MODEL_SEQUENCES=10 ZEUSDB_MODEL_STEPS=300` and upwards.
SEQUENCES = int(os.environ.get("ZEUSDB_MODEL_SEQUENCES", "2"))
STEPS = int(os.environ.get("ZEUSDB_MODEL_STEPS", "120"))

# Announce every step before driving it, for a failure that does not raise.
TRACE = os.environ.get("ZEUSDB_MODEL_TRACE", "0") != "0"


# ============================================================================
# THE GENERATOR
# ============================================================================


class Rng:
    """splitmix64, which is eleven lines and no dependency.

    The same generator the graph dump fuzzer draws from, for the same reason.
    The crate's own generator is `rand_chacha` and is pinned because every
    seeded product draw runs on it; this one draws test inputs and has no reason
    to move with it.
    """

    MASK = (1 << 64) - 1

    def __init__(self, seed):
        self.state = seed & self.MASK

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def below(self, bound):
        """A draw in `0..bound`, which is zero for a bound of zero."""
        if bound <= 0:
            return 0
        return self.next_u64() % bound

    def between(self, low, high):
        """A draw in `low..=high`."""
        return low + self.below(high - low + 1)

    def choice(self, items):
        return items[self.below(len(items))]

    def sample(self, items, count):
        """`count` distinct entries of `items`, or all of them if fewer."""
        pool = list(items)
        count = min(count, len(pool))
        picked = []
        for _ in range(count):
            picked.append(pool.pop(self.below(len(pool))))
        return picked


# ============================================================================
# THE CORPUS
# ============================================================================

CATEGORIES = ["alpha", "beta", "gamma", "delta"]
TAGS = ["ai", "science", "tech", "ops"]

# The pages `list()` is asked for beside the whole set. The last two ask past
# the end, which is where the offset arithmetic has to saturate rather than
# panic.
PAGES = [(0, 3), (1, 4), (2, 1), (5, 10), (0, 0), (40, 5), (10_000, 3)]

# The eight filters every step is checked against. Four name a declared field,
# three name an undeclared one and one mixes the two, which is the pairing a
# stale column shows up in: the declared side answers from the bitmap and the
# undeclared side walks the metadata, so a column that drifted from the metadata
# makes the two disagree.
FILTERS = [
    {"category": "alpha"},
    {"rank": {"gte": 50}},
    {"category": {"in": ["beta", "gamma"]}},
    {"$not": {"category": "delta"}},
    {"tag": "ai"},
    {"flag": True},
    {"$or": [{"tag": "ops"}, {"tag": "tech"}]},
    {"$and": [{"category": "alpha"}, {"tag": {"in": ["ai", "science"]}}]},
]


def draw_metadata(rng):
    """Two declared fields and two undeclared ones, from small domains.

    Small so that every filter in the pool selects a non-trivial subset rather
    than nothing, which is what makes a count disagreement visible.
    """
    return {
        "category": rng.choice(CATEGORIES),
        "rank": rng.below(100),
        "tag": rng.choice(TAGS),
        "flag": rng.below(2) == 0,
    }


def draw_vector(rng, dim):
    """A vector of exactly representable values.

    Every component is a small integer over eight, so the value the model holds
    and the value the index stores are the same float32 with no rounding to
    reason about. Cosine normalisation is then the only transformation between
    them, and the model applies the same one.
    """
    return [rng.between(-40, 40) / 8.0 for _ in range(dim)]


def normalize(vector, space):
    if space != "cosine":
        return list(vector)
    as_f32 = np.asarray(vector, dtype=np.float32)
    norm = float(np.sqrt(np.sum(as_f32.astype(np.float32) ** 2, dtype=np.float32)))
    if norm > 0.0:
        return [float(np.float32(x) / np.float32(norm)) for x in as_f32]
    return [float(x) for x in as_f32]


# ============================================================================
# THE FILTER LANGUAGE, EVALUATED IN PYTHON
# ============================================================================
#
# An independent implementation rather than a call back into the index, so a
# filter the index gets wrong is a disagreement rather than two wrong answers
# that match. It covers only the operators the pool above uses.


def field_matches(metadata, name, test):
    if isinstance(test, dict):
        for operator, target in test.items():
            value = metadata.get(name)
            if operator == "gte":
                if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and value >= target):
                    return False
            elif operator == "in":
                if value not in target:
                    return False
            else:  # pragma: no cover - the pool uses no other operator
                raise AssertionError(f"the model does not implement {operator!r}")
        return True
    return name in metadata and metadata[name] == test


def matches(metadata, condition):
    for key, value in condition.items():
        if key == "$and":
            if not all(matches(metadata, branch) for branch in value):
                return False
        elif key == "$or":
            if not any(matches(metadata, branch) for branch in value):
                return False
        elif key == "$not":
            if matches(metadata, value):
                return False
        elif not field_matches(metadata, key, value):
            return False
    return True


# ============================================================================
# THE MODEL
# ============================================================================


class Model:
    """What the index should hold, maintained beside it.

    `order` is the sequence `list()` pages through. It is held as relative order
    rather than as internal id values, because the values are not otherwise
    observable. A record joins at the end, an overwrite moves it to the end
    because an overwrite is a removal followed by a fresh insertion, and
    `compact`, `rebuild` and a save and load round trip leave it alone.

    `generated` mirrors the counter behind a generated external id. It is one
    per generated id, and it does not reset on `clear`, so `vec_1` is issued once
    in the life of an index. It used to be the internal id counter, which minted
    both and reset on `clear`, so a generated id was reissued after one and a
    record added without an id of its own could take a name an earlier record
    still held elsewhere.

    `vectors_exact` goes false when a `quantized_only` index trains, because the
    raw vectors are released at that point and `get_records` returns a
    reconstruction. Nothing else in the model becomes unknowable.
    """

    def __init__(self, config):
        self.space = config["space"]
        self.dim = config["dim"]
        self.declared = list(config["indexed_fields"] or [])
        self.has_quantization = config["quantization"] is not None
        self.vectors = {}
        self.metadata = {}
        self.order = []
        self.generated = 0
        self.index_metadata = {}
        self.vectors_exact = True
        # The three `rebuild` can move, held because a load has to bring back
        # what the last rebuild set rather than what `create` was given.
        self.m = None
        self.ef_construction = None
        self.expected_size = None
        # Whether training has fired, which decides the storage mode the index
        # should report and whether it should call itself quantized.
        self.trained = False
        # Whether the index has trained, which is what stops the batches
        # growing. Separate from `vectors_exact` because `quantized_with_raw`
        # crosses the same threshold and keeps its vectors.
        self.vectors_exact_threshold_crossed = False

    def generated_id(self):
        self.generated += 1
        return f"vec_{self.generated}"

    def put(self, record_id, vector, metadata):
        if record_id in self.vectors:
            self.order.remove(record_id)
        self.vectors[record_id] = normalize(vector, self.space)
        self.metadata[record_id] = dict(metadata)
        self.order.append(record_id)

    def drop(self, record_id):
        if record_id not in self.vectors:
            return False
        del self.vectors[record_id]
        del self.metadata[record_id]
        self.order.remove(record_id)
        return True

    def clear(self):
        removed = len(self.vectors)
        self.vectors.clear()
        self.metadata.clear()
        self.order.clear()
        # `generated` is deliberately not reset. The index's internal id counter
        # is, and the two were one counter until they had to differ here.
        return removed

    def selected(self, condition):
        return {rid for rid, meta in self.metadata.items() if matches(meta, condition)}

    def ids(self):
        return set(self.vectors)


# ============================================================================
# THE CHECK
# ============================================================================


def check(index, model, where):
    """Everything the model holds, asked of the index.

    `where` names the configuration and step, so a failure reproduces without
    reading the traceback.
    """
    assert len(index) == len(model.vectors), f"{where}: count"
    assert index.get_vector_count() == len(model.vectors), f"{where}: get_vector_count"
    assert index.count() == len(model.vectors), f"{where}: count()"

    listed = index.list(number=len(model.order) + 8)
    assert [rid for rid, _ in listed] == model.order, f"{where}: list ordering"

    for record_id, metadata in listed:
        assert metadata == model.metadata[record_id], f"{where}: list metadata {record_id}"

    # A page rather than the whole set. `list` partitions the tail away and
    # sorts only the prefix, so an offset past the sorted window is a different
    # path from the one the call above takes.
    for offset, number in PAGES:
        page = index.list(number=number, offset=offset)
        assert [rid for rid, _ in page] == model.order[offset:offset + number], (
            f"{where}: list(number={number}, offset={offset})"
        )

    # The creation parameters, which `rebuild` moves and a save and load has to
    # carry. `dim` and `space` are fixed at creation and nothing can move them.
    assert index.dim == model.dim, f"{where}: dim"
    assert index.space == model.space, f"{where}: space"
    assert index.m == model.m, f"{where}: m"
    assert index.ef_construction == model.ef_construction, f"{where}: ef_construction"
    assert index.expected_size == model.expected_size, f"{where}: expected_size"
    assert index.indexed_fields == model.declared, f"{where}: indexed_fields"

    # Quantization state. A trained index stays trained through a clear, because
    # the codebook was fitted from data a clear cannot bring back.
    assert index.has_quantization() == model.has_quantization, f"{where}: has_quantization"
    assert index.is_quantized() == model.trained, f"{where}: is_quantized"
    assert index.can_use_quantization() == model.trained, f"{where}: can_use_quantization"

    ids = sorted(model.vectors)
    fetched = index.get_records(ids, return_vector=True)
    assert len(fetched) == len(ids), f"{where}: get_records returned {len(fetched)}"
    by_id = {record["id"]: record for record in fetched}
    assert set(by_id) == set(ids), f"{where}: get_records ids"
    for record_id in ids:
        record = by_id[record_id]
        assert record["metadata"] == model.metadata[record_id], (
            f"{where}: metadata of {record_id}"
        )
        assert index.contains(record_id), f"{where}: contains {record_id}"
        if model.vectors_exact:
            stored = [float(x) for x in record["vector"]]
            expected = model.vectors[record_id]
            assert len(stored) == len(expected), f"{where}: vector width of {record_id}"
            for i, (got, want) in enumerate(zip(stored, expected)):
                assert abs(got - want) < 1e-6, (
                    f"{where}: vector of {record_id} at {i}: {got} against {want}"
                )

    assert index.get_all_metadata() == model.index_metadata, f"{where}: index metadata"

    for condition in FILTERS:
        expected = model.selected(condition)
        assert index.count(condition) == len(expected), (
            f"{where}: count{condition} gave {index.count(condition)} "
            f"against {len(expected)}"
        )


def check_search(index, model, rng, dim, where):
    """What a model can say about an approximate search.

    Not which records come back, which is what makes it approximate. That every
    hit is a live record, that no id repeats, that the page is no longer than
    the records that could fill it, that scores do not decrease, and that under
    a filter every hit satisfies the model's own evaluation of it.
    """
    if not model.vectors:
        return
    query = np.asarray(draw_vector(rng, dim), dtype=np.float32)
    top_k = rng.between(1, 12)

    condition = rng.choice([None] + FILTERS)
    if condition is None:
        results = index.search(query, top_k=top_k)
        eligible = model.ids()
    else:
        results = index.search(query, filter=condition, top_k=top_k)
        eligible = model.selected(condition)

    found = [hit["id"] for hit in results]
    assert len(found) == len(set(found)), f"{where}: search repeated an id"
    assert len(found) <= min(top_k, len(eligible)), f"{where}: search page too long"
    for record_id in found:
        assert record_id in eligible, (
            f"{where}: search returned {record_id}, which the model does not select"
        )
    scores = [hit["score"] for hit in results]
    assert scores == sorted(scores), f"{where}: search scores are not ordered"


# ============================================================================
# THE OPERATIONS
# ============================================================================
#
# Each returns a short label for the trace. Weights are beside the table at the
# bottom of this section; a uniform draw over seventeen would spend most of a
# sequence adding, because records have to exist before anything can remove or
# re-tag them, so the rare ones are weighted up to make them actually occur.


def batch_size(rng, model, config):
    """How many records one add carries.

    Mostly small. A configuration that has to cross the training threshold draws
    a large batch half the time until it has crossed, and small ones after.

    The growth has to be deliberate because the removals outrun a gentle one.
    `training_size` cannot go below 1000, and with `remove_where` and `delete`
    each taking a filter that selects between a quarter and three quarters of
    the index, a first attempt drawing 120 to 240 a quarter of the time peaked
    at 652 records over 240 steps and never trained at all. Sizing the batch
    against the threshold is what makes the crossing a property of the harness
    rather than of the draw.
    """
    if config["crosses_training"] and not model.vectors_exact_threshold_crossed:
        if rng.below(2) == 0:
            return rng.between(200, 400)
    return rng.between(1, 5)


def fresh_ids(rng, model, count):
    """Explicit ids that cannot collide with a generated one.

    Generated ids are `vec_N`, so nothing here starts with that prefix and the
    model never has to decide which of two records with one id survived a batch.
    """
    ids = []
    while len(ids) < count:
        candidate = f"r{rng.next_u64() % 1_000_000_000:09d}"
        if candidate not in model.vectors and candidate not in ids:
            ids.append(candidate)
    return ids


def op_add_parallel(index, model, rng, config):
    count = batch_size(rng, model, config)
    ids = fresh_ids(rng, model, count)
    vectors = [draw_vector(rng, config["dim"]) for _ in range(count)]
    metadatas = [draw_metadata(rng) for _ in range(count)]
    result = index.add({"ids": ids, "embeddings": vectors, "metadatas": metadatas})
    assert result.total_errors == 0, result.errors
    for record_id, vector, metadata in zip(ids, vectors, metadatas):
        model.put(record_id, vector, metadata)
    return f"add_parallel({count})"


def op_add_single_object(index, model, rng, config):
    record_id = fresh_ids(rng, model, 1)[0]
    vector = draw_vector(rng, config["dim"])
    metadata = draw_metadata(rng)
    result = index.add({"id": record_id, "values": vector, "metadata": metadata})
    assert result.total_errors == 0, result.errors
    model.put(record_id, vector, metadata)
    return "add_single_object"


def op_add_list_of_dicts(index, model, rng, config):
    count = rng.between(1, 4)
    ids = fresh_ids(rng, model, count)
    payload = []
    drawn = []
    for record_id in ids:
        vector = draw_vector(rng, config["dim"])
        metadata = draw_metadata(rng)
        payload.append({"id": record_id, "vector": vector, "metadata": metadata})
        drawn.append((record_id, vector, metadata))
    result = index.add(payload)
    assert result.total_errors == 0, result.errors
    for record_id, vector, metadata in drawn:
        model.put(record_id, vector, metadata)
    return f"add_list_of_dicts({count})"


def op_add_numpy_2d(index, model, rng, config):
    count = batch_size(rng, model, config)
    vectors = [draw_vector(rng, config["dim"]) for _ in range(count)]
    array = np.asarray(vectors, dtype=np.float32)
    # Every id is minted during parsing, before any record is inserted, so the
    # model mints all of them first too.
    ids = [model.generated_id() for _ in range(count)]
    result = index.add(array)
    assert result.total_errors == 0, result.errors
    for record_id, vector in zip(ids, vectors):
        model.put(record_id, vector, {})
    return f"add_numpy_2d({count})"


def op_add_single_vector(index, model, rng, config):
    vector = draw_vector(rng, config["dim"])
    record_id = model.generated_id()
    result = index.add(np.asarray(vector, dtype=np.float32))
    assert result.total_errors == 0, result.errors
    model.put(record_id, vector, {})
    return "add_single_vector"


def op_add_overwrite(index, model, rng, config):
    if not model.vectors:
        return op_add_parallel(index, model, rng, config)
    count = rng.between(1, min(3, len(model.vectors)))
    ids = rng.sample(sorted(model.vectors), count)
    vectors = [draw_vector(rng, config["dim"]) for _ in range(count)]
    metadatas = [draw_metadata(rng) for _ in range(count)]
    result = index.add(
        {"ids": ids, "embeddings": vectors, "metadatas": metadatas}, overwrite=True
    )
    assert result.total_errors == 0, result.errors
    for record_id, vector, metadata in zip(ids, vectors, metadatas):
        model.put(record_id, vector, metadata)
    return f"add_overwrite({count})"


def op_add_collision(index, model, rng, config):
    """A batch where one id already exists and overwrite is off.

    The colliding record is refused and reported, and the others are inserted.
    The model applies exactly that, which is what makes a batch non atomic here.
    """
    if not model.vectors:
        return op_add_parallel(index, model, rng, config)
    existing = rng.choice(sorted(model.vectors))
    count = rng.between(1, 3)
    ids = [existing] + fresh_ids(rng, model, count)
    vectors = [draw_vector(rng, config["dim"]) for _ in range(count + 1)]
    metadatas = [draw_metadata(rng) for _ in range(count + 1)]
    result = index.add(
        {"ids": ids, "embeddings": vectors, "metadatas": metadatas}, overwrite=False
    )
    assert result.total_errors == 1, result.errors
    assert result.total_inserted == count
    # The refused record still consumed nothing, and the accepted ones each
    # consumed one tick in the order they appear.
    for record_id, vector, metadata in zip(ids[1:], vectors[1:], metadatas[1:]):
        model.put(record_id, vector, metadata)
    return f"add_collision({count})"


def op_remove_point(index, model, rng, config):
    if not model.vectors:
        return "remove_point(empty)"
    record_id = rng.choice(sorted(model.vectors))
    assert index.remove_point(record_id) is True
    model.drop(record_id)
    # An id the index does not hold is refused rather than silently accepted.
    assert index.remove_point(record_id) is False
    return "remove_point"


def op_remove_points(index, model, rng, config):
    if not model.vectors:
        return "remove_points(empty)"
    count = rng.between(1, min(4, len(model.vectors)))
    ids = rng.sample(sorted(model.vectors), count)
    absent = f"absent_{rng.next_u64() % 100000}"
    # A repeat and an absent id in the same call, which is what the return
    # contract has to be right about.
    requested = ids + [absent, ids[0]]
    missing = index.remove_points(requested)
    assert missing == [absent], missing
    for record_id in ids:
        model.drop(record_id)
    return f"remove_points({count})"


def op_remove_where(index, model, rng, config):
    condition = rng.choice(FILTERS)
    expected = model.selected(condition)
    removed = index.remove_where(condition)
    assert removed == len(expected), f"remove_where removed {removed} of {len(expected)}"
    for record_id in expected:
        model.drop(record_id)
    return f"remove_where({len(expected)})"


def op_delete(index, model, rng, config):
    if rng.below(2) == 0 or not model.vectors:
        condition = rng.choice(FILTERS)
        expected = model.selected(condition)
        if not expected:
            # `remove_where` refuses a filter that matches everything, and
            # `delete(where=)` is that method, so an empty selection is still a
            # legal call returning zero.
            assert index.delete(where=condition) == 0
            return "delete(where, none)"
        assert index.delete(where=condition) == len(expected)
        for record_id in expected:
            model.drop(record_id)
        return f"delete(where, {len(expected)})"

    count = rng.between(1, min(3, len(model.vectors)))
    ids = rng.sample(sorted(model.vectors), count)
    absent = f"absent_{rng.next_u64() % 100000}"
    # A repeat and an absent id together, because `delete` counts distinct ids
    # and subtracts the ones it could not find.
    removed = index.delete(ids=ids + [ids[0], absent])
    assert removed == count, f"delete removed {removed} of {count}"
    for record_id in ids:
        model.drop(record_id)
    return f"delete(ids, {count})"


def op_update_metadata(index, model, rng, config):
    if not model.vectors:
        return "update_metadata(empty)"
    record_id = rng.choice(sorted(model.vectors))
    metadata = draw_metadata(rng)
    if rng.below(6) == 0:
        # An empty mapping clears the record's metadata, which is a write.
        metadata = {}
    assert index.update_metadata(record_id, metadata) is True
    model.metadata[record_id] = dict(metadata)
    assert index.update_metadata(f"absent_{rng.next_u64() % 1000}", metadata) is False
    return "update_metadata"


def op_add_metadata(index, model, rng, config):
    key = f"k{rng.below(4)}"
    value = f"v{rng.below(100)}"
    index.add_metadata({key: value})
    model.index_metadata[key] = value
    assert index.get_metadata(key) == value
    return "add_metadata"


def op_compact(index, model, rng, config):
    reclaimed = index.compact()
    assert reclaimed >= 0
    return f"compact({reclaimed})"


def op_shrink_to_fit(index, model, rng, config):
    index.shrink_to_fit()
    return "shrink_to_fit"


def op_clear(index, model, rng, config):
    removed = index.clear()
    assert removed == len(model.vectors), f"clear removed {removed}"
    model.clear()
    return f"clear({removed})"


def op_rebuild(index, model, rng, config):
    # `m` alone, or `m` with a raised declaration. Rebuilding at the values it
    # already holds is `compact`, which `rebuild` refuses, so at least one has
    # to move.
    new_m = rng.between(4, 24)
    if rng.below(2) == 0:
        nodes = index.rebuild(m=new_m)
        model.m = new_m
    else:
        new_size = max(1000, len(model.vectors) * 2)
        nodes = index.rebuild(m=new_m, expected_size=new_size)
        model.m = new_m
        model.expected_size = new_size
    assert nodes == len(model.vectors), f"rebuild reported {nodes} nodes"
    return f"rebuild(m={new_m})"


def op_save_load(index, model, rng, config):
    """Save the index and reopen it, then keep driving the reopened one.

    The whole point of putting it in the sequence rather than at the end is that
    everything after it runs against the index that came back off disk, so a
    field a load does not restore shows up on the operation that reads it rather
    than on the load itself.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "index")
        index.save(path)
        reopened = VectorDatabase().load(path)
    return reopened, "save_load"


OPERATIONS = [
    (10, op_add_parallel),
    (4, op_add_single_object),
    (4, op_add_list_of_dicts),
    (4, op_add_numpy_2d),
    (3, op_add_single_vector),
    (6, op_add_overwrite),
    (3, op_add_collision),
    (5, op_remove_point),
    (4, op_remove_points),
    (4, op_remove_where),
    (3, op_delete),
    (5, op_update_metadata),
    (2, op_add_metadata),
    (3, op_compact),
    (2, op_shrink_to_fit),
    (3, op_save_load),
    (1, op_clear),
    (1, op_rebuild),
]

TOTAL_WEIGHT = sum(weight for weight, _ in OPERATIONS)


def draw_operation(rng):
    ticket = rng.below(TOTAL_WEIGHT)
    for weight, operation in OPERATIONS:
        if ticket < weight:
            return operation
        ticket -= weight
    raise AssertionError("the weight table does not cover its own total")


# ============================================================================
# THE CONFIGURATIONS
# ============================================================================

CONFIGURATIONS = [
    {
        "name": "raw_cosine",
        "dim": 8,
        "space": "cosine",
        "indexed_fields": None,
        "quantization": None,
        "crosses_training": False,
    },
    {
        "name": "raw_l2_declared",
        "dim": 6,
        "space": "l2",
        "indexed_fields": ["category", "rank"],
        "quantization": None,
        "crosses_training": False,
    },
    {
        "name": "quantized_with_raw",
        "dim": 8,
        "space": "cosine",
        "indexed_fields": ["category", "rank"],
        "quantization": {
            "type": "pq",
            "subvectors": 4,
            "bits": 4,
            "training_size": 1000,
            "storage_mode": "quantized_with_raw",
        },
        "crosses_training": True,
    },
    {
        "name": "quantized_only",
        "dim": 8,
        "space": "cosine",
        "indexed_fields": ["category"],
        "quantization": {
            "type": "pq",
            "subvectors": 4,
            "bits": 4,
            "training_size": 1000,
            "storage_mode": "quantized_only",
        },
        "crosses_training": True,
    },
]


def build(config):
    kwargs = {
        "dim": config["dim"],
        "space": config["space"],
        "expected_size": 4000,
    }
    if config["indexed_fields"] is not None:
        kwargs["indexed_fields"] = config["indexed_fields"]
    if config["quantization"] is not None:
        kwargs["quantization_config"] = config["quantization"]
    return VectorDatabase().create("hnsw", **kwargs)


def run_sequence(config, sequence, steps):
    """One sequence, checked after every step."""
    rng = Rng(MODEL_SEED ^ (sequence * 0x9E3779B9) ^ hash_name(config["name"]))
    index = build(config)
    model = Model(config)
    # Read back rather than assumed. `m` is derived from `expected_size` at
    # creation by a rule the model has no reason to duplicate, and reading it
    # once is what lets `rebuild` be checked against a value the model tracks.
    model.m = index.m
    model.ef_construction = index.ef_construction
    model.expected_size = index.expected_size

    for step in range(steps):
        operation = draw_operation(rng)
        where = f"{config['name']} seed={MODEL_SEED:#x} sequence={sequence} step={step}"
        if TRACE:
            print(f"{where} {operation.__name__}", flush=True)

        outcome = operation(index, model, rng, config)
        if isinstance(outcome, tuple):
            index, label = outcome
        else:
            label = outcome

        # A `quantized_only` index releases its raw vectors when training fires,
        # so from that point `get_records` returns a reconstruction and the
        # model stops asserting the vector. Nothing else it holds changes.
        if config["quantization"] is not None and index.is_quantized():
            model.trained = True
            model.vectors_exact_threshold_crossed = True
            if config["quantization"]["storage_mode"] == "quantized_only":
                model.vectors_exact = False

        check(index, model, f"{where} after {label}")
        check_search(index, model, rng, config["dim"], f"{where} after {label}")

    return index, model


def hash_name(name):
    """A stable name hash, because PYTHONHASHSEED randomises the built in one.

    A sequence has to draw the same operations on every run for a failure to
    reproduce from the seed, and `hash(str)` in CPython does not.
    """
    value = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return value


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=lambda c: c["name"])
def test_a_random_operation_sequence_agrees_with_the_model(config):
    """Every operation, in a random order, against a statement of what should hold.

    A failure names the configuration, the seed, the sequence and the step, and
    reruns identically from those.
    """
    for sequence in range(SEQUENCES):
        index, model = run_sequence(config, sequence, STEPS)
        # The index the sequence finished on, checked once more after a compact,
        # because the debris a sequence leaves is what a later compact has to
        # reclaim without changing anything the model holds.
        index.compact()
        check(index, model, f"{config['name']} sequence={sequence} final compact")


def test_the_generator_is_reproducible():
    """The same seed draws the same operations.

    Without this a failing sequence could not be replayed, which is the whole
    reason the generator is seeded rather than drawn from entropy.
    """
    left = Rng(MODEL_SEED)
    right = Rng(MODEL_SEED)
    assert [draw_operation(left).__name__ for _ in range(200)] == [
        draw_operation(right).__name__ for _ in range(200)
    ]
    # And a different seed draws a different sequence, so the seed is load
    # bearing rather than decorative.
    other = Rng(MODEL_SEED ^ 1)
    assert [draw_operation(Rng(MODEL_SEED)).__name__ for _ in range(50)] != [
        draw_operation(other).__name__ for _ in range(50)
    ]


def test_the_model_filter_agrees_with_the_index_on_a_fixed_corpus():
    """The Python filter evaluator is checked against the index on known data.

    The model's filter implementation is independent of the index's, which is
    what makes a disagreement meaningful. That argument only holds if the model's
    own implementation is right, so it is pinned here against a corpus whose
    answers are counted by hand.
    """
    metadata = [
        {"category": "alpha", "rank": 10, "tag": "ai", "flag": True},
        {"category": "beta", "rank": 60, "tag": "ops", "flag": False},
        {"category": "delta", "rank": 99, "tag": "ai", "flag": True},
        {"category": "alpha", "rank": 50, "tag": "science", "flag": False},
        {"category": "gamma", "rank": 0, "tag": "tech", "flag": True},
    ]
    expected = [
        ({"category": "alpha"}, 2),
        ({"rank": {"gte": 50}}, 3),
        ({"category": {"in": ["beta", "gamma"]}}, 2),
        ({"$not": {"category": "delta"}}, 4),
        ({"tag": "ai"}, 2),
        ({"flag": True}, 3),
        ({"$or": [{"tag": "ops"}, {"tag": "tech"}]}, 2),
        ({"$and": [{"category": "alpha"}, {"tag": {"in": ["ai", "science"]}}]}, 2),
    ]
    assert [condition for condition, _ in expected] == FILTERS

    index = VectorDatabase().create(
        "hnsw", dim=4, space="cosine", expected_size=64,
        indexed_fields=["category", "rank"],
    )
    vectors = [[1.0, 0.0, 0.0, float(i)] for i in range(len(metadata))]
    index.add({
        "ids": [f"r{i}" for i in range(len(metadata))],
        "embeddings": vectors,
        "metadatas": metadata,
    })

    for condition, count in expected:
        by_hand = sum(1 for record in metadata if matches(record, condition))
        assert by_hand == count, f"{condition} counted {by_hand} by the model"
        assert index.count(condition) == count, f"{condition} counted by the index"
