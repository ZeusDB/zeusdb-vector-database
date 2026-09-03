"""The journal beside a directory, from Python.

`journal_to`, `checkpoint`, the `journal_path` property, the three
durabilities, the three arguments `load` gained, and the keys a journaled
index reports. The engine's own tests hold the crash matrix, the interval
thread's lifetime and what a failed commit leaves; what is here is the
surface a caller reaches, and the one rule a caller acts on most, being that
a directory saved without a journal is the directory it always was.
"""

import gc
import json
import os

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

DIM = 8
JOURNAL_HEADER_BYTES = 64


def build(n=0, **kw):
    kw.setdefault("expected_size", 200)
    index = VectorDatabase().create("hnsw", dim=DIM, **kw)
    if n:
        add(index, 0, n)
    return index


def add(index, start, stop):
    rng = np.random.default_rng(start + 1)
    result = index.add({
        "ids": [f"r{i}" for i in range(start, stop)],
        "embeddings": rng.standard_normal((stop - start, DIM)).astype(np.float32),
        "metadatas": [{"i": i} for i in range(start, stop)],
    })
    assert result.is_success(), result.errors
    return result


def manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def journal_keys(index):
    return {k: v for k, v in index.get_stats().items() if k.startswith("journal_")}


def journal_of(path):
    return path.parent / (path.name + ".zdbwal")


def drop(index):
    """Let go of the index so the journal it holds open is closed."""
    del index
    gc.collect()


# ------------------------------------------------------------
# The journal is opt-in
# ------------------------------------------------------------
def test_a_directory_saved_without_a_journal_is_the_directory_it_always_was(tmp_path):
    """create() and save() write what they wrote before the journal existed.

    No journal file, no journal record in the manifest, the format version
    an earlier release reads, and nothing about a journal in the stats or
    the info line.
    """
    index = build(5)
    path = tmp_path / "plain.zdb"
    index.save(str(path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plain.zdb"]
    m = manifest(path)
    assert m["format_version"] == "1.1.0"
    assert "journal" not in m
    assert index.journal_path is None
    assert journal_keys(index) == {}
    assert "journal" not in index.info()
    loaded = VectorDatabase().load(str(path))
    assert loaded.journal_path is None
    assert journal_keys(loaded) == {}


# ------------------------------------------------------------
# journal_to
# ------------------------------------------------------------
def test_journal_to_writes_the_checkpoint_and_the_sibling(tmp_path):
    index = build(3)
    path = tmp_path / "j.zdb"
    wal = journal_of(path)
    assert index.journal_to(str(path)) is None
    assert path.is_dir() and wal.is_file()
    assert wal.stat().st_size == JOURNAL_HEADER_BYTES
    m = manifest(path)
    assert m["format_version"] == "3.0.0"
    assert m["journal"]["file"] == "j.zdb.zdbwal"
    assert m["journal"]["sequence"] == 0
    assert len(m["journal"]["collection_id"]) == 32
    assert os.path.normcase(index.journal_path) == os.path.normcase(str(wal))
    assert journal_keys(index) == {
        "journal_durability": "call",
        "journal_sequence": "0",
        "journal_records": "0",
        "journal_bytes": str(JOURNAL_HEADER_BYTES),
    }
    assert "journal=call" in index.info()

    # Every mutation from now on is a record in the file.
    add(index, 3, 7)
    keys = journal_keys(index)
    assert keys["journal_records"] == "4"
    assert keys["journal_sequence"] == "0"
    assert int(keys["journal_bytes"]) > JOURNAL_HEADER_BYTES
    assert wal.stat().st_size == int(keys["journal_bytes"])

    with pytest.raises(RuntimeError, match="journaled once"):
        index.journal_to(str(tmp_path / "second.zdb"))
    assert not (tmp_path / "second.zdb").exists()


def test_a_journaled_index_reopens_with_what_the_journal_holds(tmp_path):
    """Everything since the checkpoint is replayed by load, and a checkpoint
    empties the journal."""
    index = build(2)
    path = tmp_path / "replay.zdb"
    index.journal_to(str(path))
    add(index, 2, 6)
    assert index.remove_point("r0")
    assert index.update_metadata("r1", {"i": 100, "tag": "moved"})
    expected = index.get_records(["r1"], return_vector=True)
    drop(index)

    loaded = VectorDatabase().load(str(path))
    assert len(loaded) == 5
    assert "r0" not in loaded
    assert loaded.get_records(["r1"], return_vector=True) == expected
    keys = journal_keys(loaded)
    assert keys["journal_durability"] == "call"
    assert keys["journal_sequence"] == "0"
    assert keys["journal_records"] == "6", "four inserts, a removal and a replacement"

    assert loaded.checkpoint() is None
    keys = journal_keys(loaded)
    assert keys["journal_sequence"] == "6"
    assert keys["journal_records"] == "0"
    assert keys["journal_bytes"] == str(JOURNAL_HEADER_BYTES)
    assert manifest(path)["journal"]["sequence"] == 6
    drop(loaded)

    again = VectorDatabase().load(str(path))
    assert len(again) == 5
    assert journal_keys(again)["journal_records"] == "0"


def test_checkpoint_needs_a_journal_and_save_takes_the_journals_directory(tmp_path):
    plain = build(2)
    with pytest.raises(RuntimeError, match="no journal"):
        plain.checkpoint()

    index = build(2)
    path = tmp_path / "home.zdb"
    index.journal_to(str(path))
    with pytest.raises(RuntimeError, match="journal"):
        index.save(str(tmp_path / "elsewhere.zdb"))
    assert not (tmp_path / "elsewhere.zdb").exists()
    add(index, 2, 4)
    index.save(str(path))
    assert journal_keys(index)["journal_sequence"] == "2"


# ------------------------------------------------------------
# The three durabilities
# ------------------------------------------------------------
@pytest.mark.parametrize("durability, interval_ms, expected", [
    ("call", None, {"journal_durability": "call"}),
    ("interval", None, {"journal_durability": "interval", "journal_interval_ms": "10"}),
    ("interval", 25, {"journal_durability": "interval", "journal_interval_ms": "25"}),
    ("none", None, {"journal_durability": "none"}),
])
def test_journal_to_takes_one_of_three_durabilities(tmp_path, durability, interval_ms, expected):
    index = build(2)
    path = tmp_path / f"{durability}.zdb"
    kwargs = {"durability": durability}
    if interval_ms is not None:
        kwargs["interval_ms"] = interval_ms
    index.journal_to(str(path), **kwargs)
    keys = journal_keys(index)
    for key, value in expected.items():
        assert keys[key] == value
    assert ("journal_interval_ms" in keys) == (durability == "interval")
    assert f"journal={durability}" in index.info()
    add(index, 2, 12)
    drop(index)
    # Whatever the policy, a process that stops loses nothing a call
    # returned from.
    loaded = VectorDatabase().load(str(path), **kwargs)
    assert len(loaded) == 12
    for key, value in expected.items():
        assert journal_keys(loaded)[key] == value


def test_journal_to_refuses_what_is_not_a_durability(tmp_path):
    index = build(1)
    path = str(tmp_path / "refused.zdb")
    with pytest.raises(ValueError, match="durability must be 'call', 'interval' or 'none', got 'sync'"):
        index.journal_to(path, durability="sync")
    with pytest.raises(TypeError):
        index.journal_to(path, durability=3)
    with pytest.raises(ValueError, match="interval_ms applies to durability='interval' alone"):
        index.journal_to(path, durability="call", interval_ms=5)
    with pytest.raises(ValueError, match="interval_ms applies to durability='interval' alone"):
        index.journal_to(path, durability="none", interval_ms=5)
    with pytest.raises(ValueError, match="at least 1"):
        index.journal_to(path, durability="interval", interval_ms=0)
    with pytest.raises(OverflowError):
        index.journal_to(path, durability="interval", interval_ms=-1)
    with pytest.raises(TypeError):
        index.journal_to(path, durability="interval", interval_ms=2.5)
    # Nothing was written by any refusal.
    assert list(tmp_path.iterdir()) == []
    assert index.journal_path is None


# ------------------------------------------------------------
# load's three arguments
# ------------------------------------------------------------
def test_load_takes_a_durability_only_where_there_is_a_journal(tmp_path):
    index = build(3)
    journaled = tmp_path / "journaled.zdb"
    index.journal_to(str(journaled))
    drop(index)
    plain = build(3)
    unjournaled = tmp_path / "plain.zdb"
    plain.save(str(unjournaled))

    loaded = VectorDatabase().load(str(journaled), durability="none")
    assert journal_keys(loaded)["journal_durability"] == "none"
    drop(loaded)
    loaded = VectorDatabase().load(str(journaled), durability="interval", interval_ms=50)
    assert journal_keys(loaded)["journal_interval_ms"] == "50"
    drop(loaded)

    with pytest.raises(ValueError, match="has no journal"):
        VectorDatabase().load(str(unjournaled), durability="call")
    with pytest.raises(ValueError, match="interval_ms applies to durability='interval' alone"):
        VectorDatabase().load(str(unjournaled), interval_ms=5)
    with pytest.raises(ValueError, match="durability must be"):
        VectorDatabase().load(str(journaled), durability="fsync")
    with pytest.raises(ValueError, match="checkpoint_only"):
        VectorDatabase().load(str(journaled), checkpoint_only=True, durability="none")
    with pytest.raises(ValueError, match="checkpoint_only"):
        VectorDatabase().load(str(journaled), checkpoint_only=True, interval_ms=5)
    # A refusal after the directory was read left it as it was.
    assert len(VectorDatabase().load(str(unjournaled))) == 3


def test_checkpoint_only_opens_the_checkpoint_and_leaves_the_journal_alone(tmp_path):
    index = build(2)
    path = tmp_path / "ckpt.zdb"
    wal = journal_of(path)
    index.journal_to(str(path))
    add(index, 2, 5)
    drop(index)

    loaded = VectorDatabase().load(str(path), checkpoint_only=True)
    assert len(loaded) == 2, "the three records in the journal were not applied"
    assert loaded.journal_path is None
    assert journal_keys(loaded) == {}
    # The file was not opened, so it can go while the index is alive, and
    # the index records nothing until a journal is opened on it.
    os.remove(wal)
    add(loaded, 10, 12)
    assert len(loaded) == 4
    # A directory whose journal is gone refuses to open the ordinary way,
    # and still opens as the checkpoint.
    with pytest.raises(FileNotFoundError):
        VectorDatabase().load(str(path))
    assert len(VectorDatabase().load(str(path), checkpoint_only=True)) == 2
    # And it can be journaled afresh, which writes a checkpoint of what it
    # now holds.
    loaded.journal_to(str(path))
    assert wal.is_file()
    assert manifest(path)["journal"]["sequence"] == 0
    drop(loaded)
    assert len(VectorDatabase().load(str(path))) == 4


def test_the_journal_closes_when_the_index_is_dropped(tmp_path):
    """load holds the journal open; dropping the index lets it go."""
    index = build(2)
    path = tmp_path / "held.zdb"
    wal = journal_of(path)
    index.journal_to(str(path))
    drop(index)
    loaded = VectorDatabase().load(str(path))
    assert os.path.normcase(loaded.journal_path) == os.path.normcase(str(wal))
    drop(loaded)
    os.remove(wal)
    with pytest.raises(FileNotFoundError):
        VectorDatabase().load(str(path))


# ------------------------------------------------------------
# A text layer with an external tokenizer
# ------------------------------------------------------------
def whitespace(text):
    """A tokenizer of the caller's own: whitespace, case kept."""
    return text.split()


def test_an_external_tokenizer_is_handed_to_load_and_the_replayed_records_are_found(tmp_path):
    index = VectorDatabase().create(
        "hnsw", dim=DIM, space="l2", expected_size=100,
        sparse={"name": "text", "tokenizer": whitespace},
    )
    path = tmp_path / "external.zdb"
    index.journal_to(str(path))
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((2, DIM)).astype(np.float32)
    # Past the checkpoint, so the records and their terms live in the
    # journal alone.
    result = index.add([{"id": "a", "vector": vectors[0], "text": "Alpha beta"},
                        {"id": "b", "vector": vectors[1], "text": "beta GAMMA"}])
    assert result.is_success(), result.errors
    before = index.query(arms=[{"text": "GAMMA"}], top_k=5)
    assert [hit["id"] for hit in before] == ["b"]
    drop(index)

    with pytest.raises(RuntimeError, match="tokenizer of the caller's own"):
        VectorDatabase().load(str(path))
    with pytest.raises(RuntimeError, match="declares itself simple"):
        VectorDatabase().load(str(path), tokenizer="simple")

    loaded = VectorDatabase().load(str(path), tokenizer=whitespace)
    assert journal_keys(loaded)["journal_records"] == "5", "two inserts and three interned terms"
    assert loaded.query(arms=[{"text": "GAMMA"}], top_k=5) == before
    assert loaded.query(arms=[{"text": "gamma"}], top_k=5) == [], "the caller's tokenizer keeps case"
    assert loaded.get_stats()["sparse_tokenizer"] == "external"
    result = loaded.add({"id": "c", "vector": vectors[0], "text": "GAMMA delta"})
    assert result.is_success(), result.errors
    assert sorted(hit["id"] for hit in loaded.query(arms=[{"text": "GAMMA"}], top_k=5)) == ["b", "c"]


# ------------------------------------------------------------
# What a journaled index refuses that an unjournaled one accepts
# ------------------------------------------------------------
def test_the_journal_ceiling_applies_to_a_journaled_index_alone(tmp_path):
    """A record whose journal entry would exceed 65 MiB is refused by a
    journaled index and accepted by one without a journal."""
    big = {"ids": ["big"], "embeddings": [[0.5] * DIM],
           "metadatas": [{"blob": "x" * (65 * 1024 * 1024)}]}
    plain = build()
    assert plain.add(big).is_success()
    assert len(plain) == 1

    journaled = build()
    journaled.journal_to(str(tmp_path / "ceiling.zdb"))
    result = journaled.add(big)
    assert result.total_errors == 1 and result.total_inserted == 0
    assert result.errors[0].startswith("Vector big: ValueError:"), result.errors
    assert len(journaled) == 0
    # The refusal moved nothing: the next record takes the first id.
    add(journaled, 0, 1)
    assert len(journaled) == 1
    assert journal_keys(journaled)["journal_records"] == "1"
