"""What a saved directory records about what it is, from Python.

The `identity` property, what every save records under `identity` in
`manifest.json`, and what each part of it is for: telling a copy from the
directory saved after it, keeping one id through a load and a save, naming
one collection in a journaled directory's manifest and its journal's header,
and naming at its next save a directory that predates the record.
"""

import gc
import json
import shutil

import numpy as np
import pytest
from helpers import as_old_wire
from zeusdb_vector_database import VectorDatabase

DIM = 8


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


def recorded(path):
    """The identity a directory's manifest records, as the property spells it."""
    identity = manifest(path)["identity"]
    return {
        "collection_id": identity["collection_id"],
        "generation": identity["generation"],
        "snapshot": identity["snapshot"],
        "parent": identity.get("parent"),
    }


def is_id(value):
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def load(path):
    return VectorDatabase().load(str(path))


def drop(index):
    """Let go of the index so a journal it holds open is closed."""
    del index
    gc.collect()


# ------------------------------------------------------------
# The property
# ------------------------------------------------------------
def test_a_new_index_has_an_id_and_no_snapshot():
    index = build()
    identity = index.identity
    assert sorted(identity) == ["collection_id", "generation", "parent", "snapshot"]
    assert is_id(identity["collection_id"])
    assert identity["generation"] == 0
    assert identity["snapshot"] is None
    assert identity["parent"] is None
    assert index.identity == identity, "the same index reports the same identity"
    assert build().identity["collection_id"] != identity["collection_id"]


def test_the_identity_is_read_only():
    index = build()
    with pytest.raises(AttributeError):
        index.identity = {}


def test_get_stats_carries_no_identity():
    index = build(5)
    stats = index.get_stats()
    assert not any("identity" in key or "generation" in key or "snapshot" in key for key in stats)
    assert index.identity["collection_id"] not in stats.values()


# ------------------------------------------------------------
# What a save records
# ------------------------------------------------------------
def test_every_save_records_what_the_index_then_reports(tmp_path):
    index = build(10)
    path = tmp_path / "saved.zdb"
    index.save(str(path))
    first = recorded(path)
    assert first["collection_id"] == index.identity["collection_id"]
    assert first["generation"] == 1
    assert is_id(first["snapshot"])
    assert first["parent"] is None
    assert "parent" not in manifest(path)["identity"], "absent rather than null"
    assert index.identity == first

    index.save(str(path))
    second = recorded(path)
    assert second["collection_id"] == first["collection_id"]
    assert second["generation"] == 2
    assert second["parent"] == first["snapshot"]
    assert second["snapshot"] != first["snapshot"]
    assert index.identity == second


def test_a_save_that_fails_moves_nothing(tmp_path):
    index = build(5)
    path = tmp_path / "kept.zdb"
    index.save(str(path))
    before = index.identity
    (tmp_path / "afile").write_text("x")
    with pytest.raises(RuntimeError, match="staging directory"):
        index.save(str(tmp_path / "afile" / "sub.zdb"))
    assert index.identity == before
    index.save(str(path))
    assert recorded(path)["generation"] == 2


# ------------------------------------------------------------
# What it is for
# ------------------------------------------------------------
def test_a_copy_is_told_from_the_directory_saved_after_it(tmp_path):
    """Save, copy, save again: the copy is one generation behind and is the
    snapshot the directory saved after it names as its parent. The copy
    opened and saved where it is becomes a second child of that snapshot."""
    index = build(10)
    current = tmp_path / "current.zdb"
    copy = tmp_path / "copy.zdb"
    index.save(str(current))
    shutil.copytree(current, copy)
    assert recorded(copy) == recorded(current)

    add(index, 10, 15)
    index.save(str(current))
    old, new = recorded(copy), recorded(current)
    assert old["collection_id"] == new["collection_id"]
    assert new["generation"] == old["generation"] + 1
    assert new["parent"] == old["snapshot"]

    forked = load(copy)
    forked.save(str(copy))
    fork = recorded(copy)
    assert fork["collection_id"] == new["collection_id"]
    assert fork["generation"] == new["generation"]
    assert fork["parent"] == new["parent"]
    assert fork["snapshot"] != new["snapshot"]


def test_a_directory_loaded_twice_is_one_index_and_saving_it_moves_one_generation(tmp_path):
    index = build(10)
    path = tmp_path / "plain.zdb"
    index.save(str(path))
    written = recorded(path)
    drop(index)

    first, second = load(path), load(path)
    assert first.identity == written
    assert second.identity == written
    first.save(str(path))
    again = recorded(path)
    assert again["collection_id"] == written["collection_id"]
    assert again["generation"] == 2
    assert again["parent"] == written["snapshot"]


def test_a_journaled_directory_names_one_id_three_times(tmp_path):
    """The manifest's identity, its journal record and the journal's header."""
    index = build(5)
    path = tmp_path / "j.zdb"
    wal = path.parent / (path.name + ".zdbwal")
    index.journal_to(str(path))

    def agree():
        m = manifest(path)
        header = int.from_bytes(wal.read_bytes()[24:40], "little")
        assert m["identity"]["collection_id"] == m["journal"]["collection_id"]
        assert m["identity"]["collection_id"] == format(header, "032x")
        assert m["identity"]["collection_id"] == index.identity["collection_id"]

    agree()
    assert recorded(path)["generation"] == 1
    add(index, 5, 9)
    index.checkpoint()
    agree()
    two = recorded(path)
    assert two["generation"] == 2
    add(index, 9, 12)
    drop(index)

    reopened = load(path)
    assert reopened.identity == two, "a replay is not a save"
    assert len(reopened) == 12


def test_clear_keeps_the_identity(tmp_path):
    index = build(10)
    path = tmp_path / "cleared.zdb"
    index.save(str(path))
    before = index.identity
    assert index.clear() == 10
    assert index.identity == before
    index.save(str(path))
    after = recorded(path)
    assert after["collection_id"] == before["collection_id"]
    assert after["generation"] == 2
    assert after["parent"] == before["snapshot"]


# ------------------------------------------------------------
# A directory saved before the record existed
# ------------------------------------------------------------
@pytest.mark.parametrize("version", ["1.1.0", "4.0.0"])
def test_a_directory_recording_no_identity_is_named_at_its_next_save(tmp_path, version):
    """1.1.0 in the old wire, as 0.11.0 wrote it, and 4.0.0 framed, as the
    build before the record wrote it. Each opens at generation 0 under an id
    drawn for it, and its next save records that id at generation 1."""
    index = build(10)
    path = tmp_path / "older.zdb"
    index.save(str(path))
    drop(index)
    if version != "4.0.0":
        as_old_wire(path, version)
    m = manifest(path)
    del m["identity"]
    (path / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    first, second = load(path), load(path)
    assert first.identity["collection_id"] != second.identity["collection_id"]
    assert first.identity["generation"] == 0
    assert first.identity["snapshot"] is None
    assert len(first) == 10

    first.save(str(path))
    named = recorded(path)
    assert named["collection_id"] == first.identity["collection_id"]
    assert named["generation"] == 1
    assert named["parent"] is None
    assert load(path).identity == named
