"""A backup, from Python.

`backup(path)`, what the copy it writes is, and the targets it refuses, with
the refusal a save gives a path that is not an index directory. The engine's
own tests hold a backup taken while other threads write and search, the
kill points inside it and a sync that fails before it; what is here is the
surface a caller reaches.
"""

import gc
import json

import numpy as np
import pytest
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


def everything(index):
    """Every record the index holds, with its vector and its metadata."""
    ids = sorted(record_id for record_id, _ in index.list(10_000))
    return index.get_records(ids, return_vector=True)


def test_a_backup_is_a_standalone_copy_and_moves_nothing(tmp_path):
    """The copy is a directory at 5.0.0 naming no journal, one generation on
    under a snapshot of its own with the index's snapshot as its parent and
    the sequence the journal reached. The index's identity and journal do
    not move, and the copy loads holding every record."""
    index = build(6)
    path = tmp_path / "j.zdb"
    index.journal_to(str(path))
    add(index, 6, 10)
    before = index.identity
    stats = {k: v for k, v in index.get_stats().items() if k.startswith("journal_")}
    target = tmp_path / "copy.zdb"

    copy = index.backup(str(target))
    assert sorted(copy) == [
        "collection_id", "generation", "parent", "saved_at", "sequence", "snapshot",
    ]
    assert copy["collection_id"] == before["collection_id"]
    assert copy["generation"] == before["generation"] + 1
    assert copy["parent"] == before["snapshot"]
    assert copy["snapshot"] != before["snapshot"]
    assert copy["sequence"] == 5, "four inserts and their commit"
    m = manifest(target)
    assert m["format_version"] == "5.0.0"
    assert "journal" not in m
    assert "journal" not in m["features"]
    assert m["identity"]["snapshot"] == copy["snapshot"]
    assert m["saved_at"] == copy["saved_at"]
    assert not (tmp_path / "copy.zdb.zdbjournal").exists()

    assert index.identity == before, "the identity stays"
    assert {k: v for k, v in index.get_stats().items() if k.startswith("journal_")} == stats

    loaded = VectorDatabase().load(str(target))
    assert loaded.journal_path is None
    assert loaded.identity == copy
    assert everything(loaded) == everything(index)
    assert loaded.recovery["journaled"] is False

    index.checkpoint()
    assert index.identity["parent"] == before["snapshot"], "the next checkpoint names the same parent"


def test_a_backup_of_an_index_with_no_journal(tmp_path):
    index = build(5)
    copy = index.backup(str(tmp_path / "copy.zdb"))
    assert copy["generation"] == 1
    assert copy["parent"] is None
    assert copy["sequence"] == 0
    assert index.identity["generation"] == 0, "a backup is not the index's save"
    assert len(VectorDatabase().load(str(tmp_path / "copy.zdb"))) == 5
    # A second backup into the same path replaces the first.
    add(index, 5, 7)
    again = index.backup(str(tmp_path / "copy.zdb"))
    assert again["snapshot"] != copy["snapshot"]
    assert len(VectorDatabase().load(str(tmp_path / "copy.zdb"))) == 7


def test_a_backup_refuses_the_indexs_own_directory(tmp_path):
    index = build(3)
    path = tmp_path / "j.zdb"
    index.journal_to(str(path))
    written = manifest(path)
    with pytest.raises(RuntimeError, match="the collection's own"):
        index.backup(str(path))
    assert manifest(path) == written


def test_a_backup_refuses_a_directory_with_a_journal_beside_it(tmp_path):
    other = build(2)
    target = tmp_path / "other.zdb"
    other.journal_to(str(target))
    del other
    gc.collect()
    written = manifest(target)
    with pytest.raises(RuntimeError, match="A journal sits beside"):
        build(3).backup(str(target))
    assert manifest(target) == written


def test_a_backup_and_a_save_refuse_a_path_that_is_not_an_index(tmp_path):
    index = build(3)
    afile = tmp_path / "notes.zdb"
    afile.write_text("mine")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine")
    for target in (afile, folder):
        with pytest.raises(RuntimeError, match="not an index directory"):
            index.backup(str(target))
        with pytest.raises(RuntimeError, match="not an index directory"):
            index.save(str(target))
        with pytest.raises(RuntimeError, match="not an index directory"):
            index.journal_to(str(target))
    assert afile.read_text() == "mine"
    assert (folder / "notes.txt").read_text() == "mine"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["folder", "notes.zdb"]
    assert index.journal_path is None

    empty = tmp_path / "empty.zdb"
    empty.mkdir()
    index.save(str(empty))
    assert manifest(empty)["format_version"] == "5.0.0"
