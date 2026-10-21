"""Saved directories older releases wrote, and the format this build writes.

The fixtures under tests/fixtures/formats are journaled directories real
releases wrote: a raw index at 3.0.0 and one trained with scalar quantization
at 3.1.0, both by 0.11.0, and one with a text space at 4.0.0 by 0.13.0. Each
holds records above its checkpoint in a journal of format version 1 that
ends in a torn tail, and an expected.json naming every record a recovery
must hold, with its vector as the hexadecimal of its float32 bytes. A load
replays the journal and writes the directory again at 5.0.0, so every test
loads a copy.

A 5.0.0 manifest also keeps the fields every release from 0.5.0 to 0.13.0
parses before its version check, so those releases refuse the directory with
their version message rather than a parse error.
"""

import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
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
