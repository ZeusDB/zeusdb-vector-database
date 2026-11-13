"""Saved directories older releases wrote, and the format this build writes.

The fixtures under tests/fixtures/formats are journaled directories real
releases wrote: a raw index at 3.0.0 and one trained with scalar quantization
at 3.1.0, both by 0.11.0, and one with a text space at 4.0.0 by 0.13.0. Each
holds records above its checkpoint in a journal of format version 1 that
ends in a torn tail, and an expected.json naming every record a recovery
must hold, with its vector as the hexadecimal of its float32 bytes. A load
replays the journal and writes the directory again at 5.0.0, so every test
loads a copy.

Fifteen more are unjournaled directories, one at each format an earlier
release wrote and one at 4.0.0 for each graph kind 0.13.0 writes: a raw
index at 1.0.0 by 0.4.1, whose two-file dump no build reads, product and
scalar quantized ones at 1.1.0 and 1.2.0 and text spaces at 2.0.0 and 2.1.0
by 0.11.0, and the four raw spaces, product quantized cosine and l2 and the
four scalar spaces at 4.0.0 by 0.13.0. Each holds `index.zdb` and an
expected.json recording its records, every one of them for a fixture of up
to 120 records and a digest of them with one in fifty in full for a fixture
of 1,000. A load reads each version 1 dump rather than rebuilding it, and a
save writes the directory at 5.0.0, each row in its dump once.

A 5.0.0 manifest also keeps the fields every release from 0.5.0 to 0.13.0
parses before its version check, so those releases refuse the directory with
their version message rather than a parse error.
"""

import gc
import hashlib
import json
import re
import shutil
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest
from helpers import checksum, reframe, repair_manifest, unframe
from zeusdb_vector_database import VectorDatabase

FIXTURES = Path(__file__).parent / "fixtures" / "formats"
NAMES = ["v3.0.0-journaled", "v3.1.0-journaled-int8", "v4.0.0-journaled-text"]


def copy_of(name, tmp_path):
    """A copy of the fixture `name`, and the directory in it to load."""
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target)
    return target / "journaled.zdb"


def expected_of(name):
    return json.loads((FIXTURES / name / "expected.json").read_text(encoding="utf-8"))


def manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def holds(index, expected):
    """Every recorded record held with its vector bit for bit and its
    metadata, nothing absent held, nothing else held, and the term count."""
    records = expected["records"]
    names = sorted(records, key=lambda name: int(name[1:]))
    held = index.get_records(names, return_vector=True)
    assert len(held) == len(names)
    for record in held:
        want = records[record["id"]]
        assert np.asarray(record["vector"], dtype="<f4").tobytes().hex() == want["vector"], record["id"]
        assert record["metadata"] == want["metadata"], record["id"]
    for absent in expected["absent"]:
        assert absent not in index
    assert len(index) == len(names)
    stats = index.get_stats()
    if expected["term_count"] is None:
        assert "term_count" not in stats
    else:
        assert int(stats["term_count"]) == expected["term_count"]


@pytest.mark.parametrize("name", NAMES)
def test_every_fixture_upgrades_with_every_record_it_recorded(tmp_path, name):
    """The load replays the file at journal format version 1, writes the
    directory again at 5.0.0 with a journal of segments carrying the sequence
    on, and deletes the file. A second load replays nothing and holds the
    same, and a record added after the upgrade survives a third."""
    path = copy_of(name, tmp_path)
    expected = expected_of(name)
    legacy = path.parent / "journaled.zdb.zdbwal"
    journal = path.parent / "journaled.zdb.zdbjournal"
    assert legacy.is_file() and not journal.exists()

    index = VectorDatabase().load(str(path))
    holds(index, expected)
    m = manifest(path)
    assert m["format_version"] == "5.0.0"
    assert m["journal"]["file"] == "journaled.zdb.zdbjournal"
    reached = expected["torn_sequence"] - 1
    assert m["journal"]["sequence"] == reached
    assert m["identity"]["sequence"] == reached
    assert not legacy.exists(), "the file at journal format version 1 went"
    assert [p.name for p in journal.iterdir()] == [f"{reached + 1:020d}.zdbseg"]
    assert index.get_stats()["journal_records"] == "0"
    del index
    gc.collect()

    again = VectorDatabase().load(str(path))
    holds(again, expected)
    rng = np.random.default_rng(219)
    added = again.add({"ids": ["after"], "embeddings": rng.standard_normal((1, expected["dim"])).astype(np.float32)})
    assert added.is_success(), added.errors
    del again
    gc.collect()
    third = VectorDatabase().load(str(path))
    assert "after" in third
    assert len(third) == len(expected["records"]) + 1


@pytest.mark.parametrize("name", NAMES)
def test_a_fixtures_checkpoint_alone_upgrades_nothing(tmp_path, name):
    """checkpoint_only=True opens the directory as it stood at its last save
    and changes no file of it or its journal."""
    path = copy_of(name, tmp_path)
    folder = path.parent

    def files():
        return sorted((p.relative_to(folder).as_posix(), p.read_bytes())
                      for p in folder.rglob("*") if p.is_file() and p.suffix != ".zdblock")

    before = files()
    index = VectorDatabase().load(str(path), checkpoint_only=True)
    assert len(index) == expected_of(name)["checkpoint_records"]
    del index
    gc.collect()
    assert files() == before


def test_the_fixtures_are_as_the_releases_wrote_them():
    """No load has written to a fixture: each holds its file at journal format
    version 1 of the recorded length, no journal of segments and no lock
    file, and declares the version its release wrote."""
    for name in NAMES:
        folder = FIXTURES / name
        expected = expected_of(name)
        assert (folder / "journaled.zdb.zdbwal").stat().st_size == expected["journal_bytes"]
        assert not (folder / "journaled.zdb.zdbjournal").exists()
        assert not (folder / "journaled.zdb.zdblock").exists()
        assert manifest(folder / "journaled.zdb")["format_version"] == expected["format_version"]


# The fields every release from 0.5.0 to 0.13.0 parses before its version
# check, and their types.
OLDER_FIELDS = {
    "format_version": str, "zeusdb_version": str, "created_at": str, "saved_at": str,
    "total_vectors": int, "index_type": str, "has_quantization": bool,
    "quantization_trained": bool, "storage_mode": str, "files_included": list,
    "files_excluded": list, "total_size_mb": (int, float),
}


def test_a_5_0_0_manifest_keeps_what_older_releases_parse(tmp_path):
    """The twelve fields older releases need to reach their version check,
    the journal record 0.11.0 to 0.13.0 parse, and the identity 0.12.0 and
    0.13.0 parse, each in the names and types those releases read."""
    rng = np.random.default_rng(5)
    plain = VectorDatabase().create("hnsw", dim=8, expected_size=50)
    assert plain.add({"ids": [f"r{i}" for i in range(5)],
                      "embeddings": rng.standard_normal((5, 8)).astype(np.float32)}).is_success()
    unjournaled = tmp_path / "plain.zdb"
    plain.save(str(unjournaled))
    journaled = tmp_path / "journaled.zdb"
    plain.journal_to(str(journaled))
    for path in (unjournaled, journaled):
        m = manifest(path)
        for field, kind in OLDER_FIELDS.items():
            assert isinstance(m[field], kind), (path.name, field)
        assert m["format_version"] == "5.0.0"
        identity = m["identity"]
        assert isinstance(identity["collection_id"], str) and len(identity["collection_id"]) == 32
        assert isinstance(identity["generation"], int) and identity["generation"] >= 1
        assert isinstance(identity["snapshot"], str) and len(identity["snapshot"]) == 32
        assert isinstance(identity["sequence"], int)
    record = manifest(journaled)["journal"]
    assert isinstance(record["file"], str) and record["file"] == "journaled.zdb.zdbjournal"
    assert isinstance(record["sequence"], int)
    assert record["collection_id"] == manifest(journaled)["identity"]["collection_id"]
    assert "journal" not in manifest(unjournaled)


# ============================================================================
# THE UNJOURNALED FIXTURES
# ============================================================================

DIRECTORIES = [
    "v1.0.0-raw", "v1.1.0-pq", "v1.2.0-int8", "v2.0.0-text", "v2.1.0-text-int8",
    "v4.0.0-raw-cosine", "v4.0.0-raw-l2", "v4.0.0-raw-l1", "v4.0.0-raw-dot",
    "v4.0.0-pq-cosine", "v4.0.0-pq-l2-raw",
    "v4.0.0-int8-cosine", "v4.0.0-int8-l2", "v4.0.0-int8-l1", "v4.0.0-int8-dot",
]
# The fixtures whose graph keeps raw vectors beside its codes.
KEEPS_RAW = {"v1.1.0-pq", "v4.0.0-pq-l2-raw"}


def index_copy(name, tmp_path):
    """A copy of the fixture `name`'s directory, to load."""
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target)
    return target / "index.zdb"


def holds_recorded(index, expected):
    """Every record held, none absent held, and every record expected.json
    records held with its vector bit for bit and its metadata, or, for a
    fixture of 1,000 records, a digest of every vector and every record's
    metadata in id order, and the term count."""
    count, absent = expected["records"], expected["absent"]
    ids = [f"r{i}" for i in range(count + len(absent)) if f"r{i}" not in absent]
    held = {r["id"]: r for r in index.get_records(ids, return_vector=True)}
    assert len(held) == count
    assert len(index) == count
    for name in absent:
        assert name not in index
    hex_of = lambda vector: np.asarray(vector, dtype="<f4").tobytes().hex()
    for name, want in (expected.get("every") or expected["sampled"]).items():
        assert hex_of(held[name]["vector"]) == want["vector"], name
        assert held[name]["metadata"] == want["metadata"], name
    if "vectors_sha256" in expected:
        vectors = b"".join(np.asarray(held[i]["vector"], dtype="<f4").tobytes() for i in ids)
        metadata = "\n".join(json.dumps(held[i]["metadata"], sort_keys=True, separators=(",", ":"))
                              for i in ids)
        assert hashlib.sha256(vectors).hexdigest() == expected["vectors_sha256"]
        assert hashlib.sha256(metadata.encode("utf-8")).hexdigest() == expected["metadata_sha256"]
    stats = index.get_stats()
    if expected["term_count"] is None:
        assert "term_count" not in stats
    else:
        assert int(stats["term_count"]) == expected["term_count"]


def pages(index):
    """The pages of a fixed set of queries, each as an id and a score per hit."""
    queries = np.random.default_rng(2190).standard_normal((8, 8)).astype(np.float32)
    return [[(hit["id"], hit["score"]) for hit in index.search(query.tolist(), top_k=10)]
            for query in queries]


@pytest.mark.parametrize("name", DIRECTORIES)
def test_every_directory_fixture_opens_with_every_record_it_recorded(tmp_path, name):
    """Each fixture opens at the format its release wrote with every record
    it recorded. Its version 1 dump is read rather than rebuilt, but for the
    1.0.0 fixture's two-file dump, which no build reads."""
    expected = expected_of(name)
    index = VectorDatabase().load(str(index_copy(name, tmp_path)))
    holds_recorded(index, expected)
    assert index.recovery["format_version"] == expected["format_version"]
    assert index.recovery["graph_rebuilt"] is (name == "v1.0.0-raw")


@pytest.mark.parametrize("name", DIRECTORIES)
def test_every_directory_fixture_saves_at_5_0_0_with_the_same_pages(tmp_path, name):
    """A save of the opened fixture writes the directory at 5.0.0, each row
    in its version 2 dump alone, beside vectors.bin only where the graph
    keeps raw vectors, and the directory reopens, its dump read, with every
    record and the pages the fixture gave before the save."""
    expected = expected_of(name)
    index = VectorDatabase().load(str(index_copy(name, tmp_path)))
    before = pages(index)
    saved = tmp_path / f"{name}-saved.zdb"
    index.save(str(saved))
    del index
    gc.collect()
    m = manifest(saved)
    assert m["format_version"] == "5.0.0"
    for absent in ("pq_codes.bin", "int8_rows.zdbint8", "hnsw_index.hnsw.graph",
                   "hnsw_index.hnsw.data"):
        assert absent not in m["files_included"], absent
        assert not (saved / absent).exists(), absent
    assert ("vectors.bin" in m["files_included"]) is (name in KEEPS_RAW)
    assert (saved / "hnsw_index.zdbgraph").read_bytes()[8:12] == (2).to_bytes(4, "little")
    reopened = VectorDatabase().load(str(saved))
    assert reopened.recovery["graph_rebuilt"] is False
    holds_recorded(reopened, expected)
    assert pages(reopened) == before


def test_the_directory_fixtures_are_as_the_releases_wrote_them():
    """No load has written to a fixture: each declares the version its
    release wrote, every file its manifest names is present at the length the
    manifest records, and nothing sits beside the directory."""
    for name in DIRECTORIES:
        folder = FIXTURES / name
        assert sorted(p.name for p in folder.iterdir()) == ["expected.json", "index.zdb"], name
        m = manifest(folder / "index.zdb")
        assert m["format_version"] == expected_of(name)["format_version"], name
        for file, entry in m.get("file_digests", {}).items():
            assert (folder / "index.zdb" / file).stat().st_size == entry["bytes"], (name, file)
        for file in m["files_included"]:
            if file not in ("hnsw_index.hnsw.graph", "hnsw_index.hnsw.data"):
                assert (folder / "index.zdb" / file).is_file(), (name, file)


# ============================================================================
# EVERY FIXTURE SAVED AT 5.0.0
# ============================================================================

def opened_copy(name, tmp_path):
    """A copy of the fixture `name`, journaled or not, and the directory in it
    to load."""
    return copy_of(name, tmp_path) if name in NAMES else index_copy(name, tmp_path)


def everything(index):
    """Every record the index holds, by id, as its vector's float32 bytes in
    hexadecimal and its metadata."""
    ids = [record_id for record_id, _ in index.list(number=len(index) + 1)]
    assert len(ids) == len(index)
    return {record["id"]: (np.asarray(record["vector"], dtype="<f4").tobytes().hex(),
                           record["metadata"])
            for record in index.get_records(ids, return_vector=True)}


@pytest.mark.parametrize("name", NAMES + DIRECTORIES)
def test_every_fixture_saved_at_5_0_0_opens_again_with_the_same_records(tmp_path, name):
    """Each fixture, loaded and written into a new folder, by save(), or by
    backup() for a journaled one, whose save is its own checkpoint, is written
    at 5.0.0, and the copy opens with every record the load held, each with
    its vector bit for bit and its metadata, and with no other record."""
    index = VectorDatabase().load(str(opened_copy(name, tmp_path)))
    held = everything(index)
    saved = tmp_path / f"{name}-saved.zdb"
    if name in NAMES:
        index.backup(str(saved))
    else:
        index.save(str(saved))
    del index
    gc.collect()
    assert manifest(saved)["format_version"] == "5.0.0"
    reopened = VectorDatabase().load(str(saved))
    assert everything(reopened) == held


# ============================================================================
# A VALUE A LOAD OF 5.0.0 REFUSES
# ============================================================================

# A created_at that is not an RFC 3339 time.
NOT_A_TIME = "9999999999999999999999999999999999999999"


def loaded_with_warnings(path):
    """The index a load of `path` opens, and the words of every
    RuntimeWarning the load raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = VectorDatabase().load(str(path))
    return index, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("name,saved_at", [
    ("v4.0.0-raw-l2", None), ("v4.0.0-raw-l2", "yesterday"), ("v3.0.0-journaled", None),
])
def test_a_created_at_that_is_not_an_rfc_3339_time_is_replaced_at_the_load(tmp_path, name,
                                                                          saved_at):
    """A directory below 5.0.0 whose created_at is not an RFC 3339 time opens
    with its saved_at in its place, or with the time of the load where its
    saved_at is not one either. A save of it, and the upgrade a journaled
    directory takes at its load, write a manifest the next load opens with
    nothing replaced. The load's RuntimeWarning and index.recovery each name
    the value recorded and the value taken."""
    path = opened_copy(name, tmp_path)
    original = manifest(path)
    rewritten = dict(original, created_at=NOT_A_TIME)
    if saved_at is not None:
        rewritten["saved_at"] = saved_at
    (path / "manifest.json").write_text(json.dumps(rewritten, indent=2), encoding="utf-8",
                                        newline="\n")

    index, messages = loaded_with_warnings(path)
    first = index.recovery
    count = len(index)
    if name in NAMES:
        written = path
    else:
        written = tmp_path / f"{name}-saved.zdb"
        index.save(str(written))
    del index
    gc.collect()
    reopened, again = loaded_with_warnings(written)
    assert len(reopened) == count
    assert again == []
    assert reopened.recovery["replaced"] == []

    taken = manifest(written)["created_at"]
    if saved_at is None:
        assert taken == original["saved_at"]
    else:
        assert taken not in (NOT_A_TIME, saved_at)
    [entry] = first["replaced"]
    assert (entry["file"], entry["field"], entry["recorded"], entry["taken"]) == (
        "manifest.json", "created_at", NOT_A_TIME, taken)
    assert NOT_A_TIME in entry["detail"] and taken in entry["detail"]
    [message] = messages
    assert NOT_A_TIME in message and taken in message


# ============================================================================
# THE LARGEST GENERATION
# ============================================================================

# What every save of a directory at the largest generation is refused with.
AT_THE_LARGEST = (
    "This collection was read from a directory recording generation 18446744073709551615, the "
    "largest a manifest can hold, so no save can record the one after it. No release writes that "
    "generation, so the directory's manifest was edited.")


def files_of(folder):
    """Every file under `folder` but the lock, with its bytes."""
    return sorted((p.relative_to(folder).as_posix(), p.read_bytes())
                  for p in folder.rglob("*") if p.is_file() and p.suffix != ".zdblock")


@pytest.mark.parametrize("name", ["v4.0.0-journaled-text", "v4.0.0-raw-l2"])
def test_a_directory_at_the_largest_generation_refuses_every_save(tmp_path, name):
    """A directory below 5.0.0 whose identity records the largest generation a
    manifest can hold refuses every save in the words that name it. A
    journaled one, whose load would upgrade it by a save, opens read only: its
    journal is replayed in memory, nothing on disk changes, a RuntimeWarning
    gives those words, and every change, checkpoint() and save is refused in
    them. An unjournaled one opens as it did and takes changes."""
    path = opened_copy(name, tmp_path)
    m = manifest(path)
    m["identity"]["generation"] = 2 ** 64 - 1
    (path / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8", newline="\n")
    before = files_of(path.parent)

    index, messages = loaded_with_warnings(path)
    copy = tmp_path / "copy.zdb"
    if name in NAMES:
        holds(index, expected_of(name))
        assert index.recovery["read_only"] is True
        assert index.recovery["read_only_reason"] == AT_THE_LARGEST
        [message] = messages
        assert AT_THE_LARGEST in message and "opened read only" in message
        refused = index.add({"ids": ["new"], "embeddings": np.zeros((1, 8), dtype=np.float32)})
        assert refused.total_errors == 1 and AT_THE_LARGEST in refused.errors[0], refused.errors
        for change in (lambda: index.remove_point("r0"),
                       lambda: index.checkpoint(),
                       lambda: index.save(str(path)),
                       lambda: index.save(str(copy)),
                       lambda: index.backup(str(copy))):
            with pytest.raises(RuntimeError) as refusal:
                change()
            assert AT_THE_LARGEST in str(refusal.value)
    else:
        assert index.recovery["read_only"] is False
        assert messages == []
        added = index.add({"ids": ["new"], "embeddings": np.zeros((1, 8), dtype=np.float32)})
        assert added.is_success(), added.errors
        with pytest.raises(RuntimeError) as refusal:
            index.save(str(copy))
        assert AT_THE_LARGEST in str(refusal.value)
    del index
    gc.collect()
    assert not copy.exists()
    assert files_of(path.parent) == before


# ============================================================================
# A VALUE PAST THE BOUND ON A VECTOR'S NORM
# ============================================================================

# What a load's reason for a value past the bound says after the value.
PAST_THE_BOUND = " This build writes no such value, so the collection is read only."
# The bound, 2^62, in the words of a stored vector or a codebook value.
BOUND_WORDS = "2^62 (4.611686018427388e18)"


def set_float(directory, name, at, value):
    """The four bytes at `at` of the framed artefact `name` set to `value`,
    framed again, and the manifest's record of the file made to agree."""
    path = directory / name
    _, _, payload = unframe(path.read_bytes())
    payload = bytearray(payload)
    payload[at:at + 4] = struct.pack("<f", value)
    reframe(path, payload=bytes(payload))
    repair_manifest(directory, name)


def set_in_version_1_row(directory, node, at, value):
    """The four bytes at `at` of node `node`'s row in the version 1 dump of
    `directory` set to `value`, the dump's checksum and the manifest's record
    of it made to agree. The rows follow the 96 byte header, the layer table
    of 16 counts, every node's eight byte origin id and the adjacency."""
    path = directory / "hnsw_index.zdbgraph"
    blob = bytearray(path.read_bytes())
    width = int.from_bytes(blob[20:24], "little") * (4 if blob[17] == 1 else 1)
    rows = (96 + 64 + 8 * int.from_bytes(blob[40:48], "little")
            + int.from_bytes(blob[64:72], "little"))
    start = rows + node * width + at
    blob[start:start + 4] = struct.pack("<f", value)
    end = len(blob) - 16
    blob[end:end + 8] = checksum(bytes(blob[96:end])).to_bytes(8, "little")
    path.write_bytes(bytes(blob))
    repair_manifest(directory, "hnsw_index.zdbgraph")


def a_scale_of_1e30(path):
    set_float(path, "int8_scales.zdbint8", 0, 1e30)
    return (r"int8_scales\.zdbint8 holds a scale of (\S+) at dimension 0, past (\S+), the largest "
            r"scale vectors whose norm is within 2\^62 fit\.",
            [float(np.float32(1e30)), float(np.float32(2.0 ** 62) / np.float32(127))])


def a_codebook_value_of_5e18(path):
    _, _, payload = unframe((path / "pq_centroids.bin").read_bytes())
    width = int.from_bytes(payload[4:8], "little")
    set_float(path, "pq_centroids.bin", 8 + 4 * width, 5e18)
    return (r"pq_centroids\.bin holds (\S+) at value 0 of centroid 1 of subvector 0, past "
            + re.escape(BOUND_WORDS) + r", the largest value a codebook fitted to vectors whose "
            r"norm is within 2\^62 holds\.",
            [float(np.float32(5e18))])


def a_raw_vector_of_norm_1e19(path):
    set_float(path, "vectors.bin", 4 + 4 + 4, 1e19)
    return (r"vectors\.bin holds the vector of 'r\d+', whose Euclidean norm is (\S+), past "
            + re.escape(BOUND_WORDS) + r", the largest norm an index under l2 takes\.",
            None)


def a_dump_row_of_norm_1e19(path):
    set_in_version_1_row(path, 2, 4, 1e19)
    return (r"hnsw_index\.zdbgraph holds the vector of 'r\d+', whose Euclidean norm is (\S+), "
            r"past " + re.escape(BOUND_WORDS) + r", the largest norm an index under l2 takes\.",
            None)


@pytest.mark.parametrize("name,edit", [
    ("v4.0.0-int8-l2", a_scale_of_1e30),
    ("v4.0.0-pq-l2-raw", a_codebook_value_of_5e18),
    ("v4.0.0-pq-l2-raw", a_raw_vector_of_norm_1e19),
    ("v4.0.0-journaled-text", a_dump_row_of_norm_1e19),
])
def test_a_directory_holding_a_value_past_the_bound_opens_read_only(tmp_path, name, edit):
    """A directory 0.13.0 writes from vectors past the bound on a vector's
    norm, here holding a scale, a codebook value or a raw vector past what
    vectors within it give, every value finite, opens read only, whether or
    not it is journaled: nothing on disk changes, a RuntimeWarning names the
    value, the file and the bound, and every change, checkpoint() and save
    is refused in the same words. Its searches run."""
    path = opened_copy(name, tmp_path)
    pattern, values = edit(path)
    before = files_of(path.parent)

    index, messages = loaded_with_warnings(path)
    assert index.recovery["read_only"] is True
    reason = index.recovery["read_only_reason"]
    assert reason.endswith(PAST_THE_BOUND), reason
    found = re.fullmatch(pattern + re.escape(PAST_THE_BOUND), reason)
    assert found, reason
    named = [float(group) for group in found.groups()]
    if values is None:
        assert 9.9e18 < named[0] < 1.1e19, reason
    else:
        # A codebook value and a scale are float32 values, named as such.
        assert [np.float32(n) for n in named] == [np.float32(v) for v in values], reason
    [message] = messages
    assert reason in message and "opened read only" in message, message

    hits = index.search([0.25] * index.dim, top_k=5)
    assert len(hits) == 5
    copy = tmp_path / "copy.zdb"
    first = index.list(number=1)[0][0]
    refused = index.add({"ids": ["new"], "embeddings": np.zeros((1, index.dim), dtype=np.float32)})
    assert refused.total_errors == 1 and reason in refused.errors[0], refused.errors
    for change in (lambda: index.remove_point(first),
                   lambda: index.checkpoint(),
                   lambda: index.save(str(path)),
                   lambda: index.save(str(copy)),
                   lambda: index.backup(str(copy))):
        with pytest.raises(RuntimeError) as refusal:
            change()
        assert reason in str(refusal.value)
    del index
    gc.collect()
    assert not copy.exists()
    assert files_of(path.parent) == before


# ============================================================================
# A SPARSE VALUE PAST THE BOUND ON ITS MAGNITUDE
# ============================================================================

# The bound on a sparse value and on a term weighting k1, 2^40, in the words
# of a load's reason.
SPARSE_BOUND_WORDS = "2^40 (1.099511627776e12)"


def a_sparse_value_of_the_largest_float(path):
    """The first value of the first record the text space's postings hold
    set to the largest float32. The payload holds the slot count and the
    live count, then each record's id, its count, its dims and its values."""
    name = "spaces/text/postings.zdbsparse"
    _, _, payload = unframe((path / name).read_bytes())
    count = int.from_bytes(payload[12:16], "little")
    largest = float(np.finfo(np.float32).max)
    set_float(path, name, 16 + 4 * count, largest)
    return (r"spaces/text/postings\.zdbsparse holds the sparse value (\S+) at index 0 of "
            r"'([^']+)', past " + re.escape(SPARSE_BOUND_WORDS)
            + r", the largest magnitude a sparse space takes\.", largest)


def a_term_weighting_k1_of_2e12(path):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config["spaces"][0]["index"]["weighting"]["k1"] = 2e12
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8",
                                      newline="\n")
    repair_manifest(path, "config.json")
    return (r"config\.json holds a term weighting k1 of (\S+), past "
            + re.escape(SPARSE_BOUND_WORDS) + r", the largest k1 a sparse space takes\.", 2e12)


@pytest.mark.parametrize("name,edit", [
    ("v2.0.0-text", a_sparse_value_of_the_largest_float),
    ("v4.0.0-journaled-text", a_sparse_value_of_the_largest_float),
    ("v2.0.0-text", a_term_weighting_k1_of_2e12),
    ("v4.0.0-journaled-text", a_term_weighting_k1_of_2e12),
])
def test_a_directory_holding_a_sparse_value_past_the_bound_opens_read_only(tmp_path, name, edit):
    """A directory 0.13.0 writes holding a sparse value whose magnitude
    passes 2^40, or a term weighting k1 past 2^40, every value finite, opens
    read only, whether or not it is journaled: nothing on disk changes, a
    RuntimeWarning names the value, the file and the bound, and every change,
    checkpoint() and save is refused in the same words. Its searches run, a
    text arm among them."""
    path = opened_copy(name, tmp_path)
    pattern, value = edit(path)
    before = files_of(path.parent)

    index, messages = loaded_with_warnings(path)
    assert index.recovery["read_only"] is True
    reason = index.recovery["read_only_reason"]
    found = re.fullmatch(pattern + re.escape(PAST_THE_BOUND), reason)
    assert found, reason
    assert np.float32(float(found.group(1))) == np.float32(value), reason
    if found.re.groups == 2:
        assert found.group(2) in index, reason
    [message] = messages
    assert reason in message and "opened read only" in message, message

    assert len(index.search([0.25] * index.dim, top_k=5)) == 5
    assert len(index.query(arms=[{"vector": [0.25] * index.dim}, {"text": "absent"}],
                           top_k=5)) == 5
    copy = tmp_path / "copy.zdb"
    first = index.list(number=1)[0][0]
    refused = index.add({"ids": ["new"], "embeddings": np.zeros((1, index.dim), dtype=np.float32)})
    assert refused.total_errors == 1 and reason in refused.errors[0], refused.errors
    for change in (lambda: index.remove_point(first),
                   lambda: index.checkpoint(),
                   lambda: index.save(str(path)),
                   lambda: index.save(str(copy)),
                   lambda: index.backup(str(copy))):
        with pytest.raises(RuntimeError) as refusal:
            change()
        assert reason in str(refusal.value)
    del index
    gc.collect()
    assert not copy.exists()
    assert files_of(path.parent) == before
