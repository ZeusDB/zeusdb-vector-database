"""A journal whose own history is broken, from Python.

`load(path, repair=True)`, the warning it raises, what `index.recovery`
reports of it, the refusal a load without it gives, and the arguments it is
refused beside. The engine's own tests hold each class of damage, a record
that will not apply, the kill points inside a repair and the bytes it sets
aside; what is here is the surface a caller reaches.
"""

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

DIM = 8
SEGMENT_HEADER_BYTES = 128
CALL = 4


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


def journal_of(path):
    return path.parent / (path.name + ".zdbjournal")


def journaled(tmp_path, calls=5):
    """A journaled directory holding `calls` calls of four records each in
    its journal, with the index that wrote it let go."""
    index = build()
    path = tmp_path / "damaged.zdb"
    index.journal_to(str(path))
    for call in range(calls):
        add(index, CALL * call, CALL * call + CALL)
    del index
    gc.collect()
    return path


def damage(path):
    """Flip one byte a third of the way into the journal's segment, inside a
    record with whole records after it, so the journal holds a corrupt
    middle."""
    (segment,) = [p for p in journal_of(path).iterdir() if p.is_file()]
    data = bytearray(segment.read_bytes())
    data[SEGMENT_HEADER_BYTES + (len(data) - SEGMENT_HEADER_BYTES) // 3] ^= 0x40
    segment.write_bytes(bytes(data))


def test_a_damaged_journal_is_refused_naming_repair(tmp_path):
    path = journaled(tmp_path)
    damage(path)
    with pytest.raises(RuntimeError, match="repair=True"):
        VectorDatabase().load(str(path))


def test_repair_keeps_every_call_before_the_damage_and_says_so(tmp_path):
    path = journaled(tmp_path)
    damage(path)
    with pytest.warns(RuntimeWarning, match="set aside"):
        index = VectorDatabase().load(str(path), repair=True)

    recovery = index.recovery
    assert recovery["checkpoint_taken"] == "repair"
    assert recovery["damage"]["kind"] == "corrupt"
    assert recovery["damage"]["segment"].endswith(".zdbseg")
    assert isinstance(recovery["damage"]["detail"], str)
    repair = recovery["repair"]
    assert sorted(repair) == [
        "bytes_discarded", "discarded_from", "kept_through", "quarantine", "records_discarded",
    ]
    assert repair["kept_through"] == repair["discarded_from"] - 1
    assert repair["records_discarded"] > 0
    assert repair["bytes_discarded"] > 0
    assert 0 < len(index) < 5 * CALL
    assert len(index) % CALL == 0, "whole calls are kept"
    assert repair["discarded_from"] == len(index) // CALL * (CALL + 1) + 1

    folder = Path(repair["quarantine"])
    assert folder.parent == journal_of(path) / "quarantine"
    assert folder.name.endswith(f"-{repair['discarded_from']}")
    assert ":" not in folder.name
    assert not (folder / "UNSEALED").exists()
    report = json.loads((folder / "repair.json").read_text(encoding="utf-8"))
    assert report["cut"]["sequence"] == repair["discarded_from"]
    assert report["records_discarded"] == repair["records_discarded"]
    aside = sum(p.stat().st_size for p in folder.iterdir() if p.name != "repair.json")
    assert aside == repair["bytes_discarded"]

    kept = len(index)
    del index
    gc.collect()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = VectorDatabase().load(str(path))
    assert len(again) == kept
    assert again.recovery["damage"] is None
    assert again.recovery["repair"] is None


def test_a_stale_segment_whose_header_does_not_read_is_set_aside_alone(tmp_path):
    """A segment a checkpoint deleted, put back as a crash inside that
    deletion leaves it, with its header damaged: named at or below the
    checkpoint's sequence, it is stale by its name. A load refuses naming
    repair=True, and a repair sets that segment aside whole and keeps every
    record."""
    index = build()
    path = tmp_path / "stale.zdb"
    index.journal_to(str(path))
    add(index, 0, 8)
    (first,) = [p for p in journal_of(path).iterdir() if p.is_file()]
    stale = bytearray(first.read_bytes())
    index.checkpoint()
    add(index, 8, 12)
    del index
    gc.collect()
    assert not first.exists()
    stale[40] ^= 0x40
    first.write_bytes(bytes(stale))

    with pytest.raises(RuntimeError, match="repair=True"):
        VectorDatabase().load(str(path))
    with pytest.warns(RuntimeWarning, match="set aside"):
        index = VectorDatabase().load(str(path), repair=True)
    assert len(index) == 12
    recovery = index.recovery
    assert recovery["checkpoint_taken"] == "repair"
    assert recovery["damage"]["kind"] == "header"
    repair = recovery["repair"]
    assert repair["discarded_from"] is None
    assert repair["records_discarded"] == 0
    assert repair["bytes_discarded"] == len(stale)
    assert (Path(repair["quarantine"]) / first.name).read_bytes() == bytes(stale)
    assert not first.exists()


def test_repair_of_an_undamaged_journal_is_a_plain_load(tmp_path):
    path = journaled(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        index = VectorDatabase().load(str(path), repair=True)
    assert len(index) == 5 * CALL
    assert index.recovery["repair"] is None
    assert index.recovery["damage"] is None
    assert index.recovery["checkpoint_taken"] is None
    assert not (journal_of(path) / "quarantine").exists()


def test_repair_of_a_directory_with_no_journal_is_a_plain_load(tmp_path):
    path = tmp_path / "plain.zdb"
    build(5).save(str(path))
    index = VectorDatabase().load(str(path), repair=True)
    assert len(index) == 5
    assert index.recovery["journaled"] is False
    assert index.recovery["repair"] is None


def test_repair_takes_a_durability(tmp_path):
    path = journaled(tmp_path)
    damage(path)
    with pytest.warns(RuntimeWarning):
        index = VectorDatabase().load(str(path), repair=True, durability="none")
    assert index.get_stats()["journal_durability"] == "none"


def test_repair_is_refused_beside_checkpoint_only(tmp_path):
    path = journaled(tmp_path)
    with pytest.raises(ValueError, match="checkpoint_only"):
        VectorDatabase().load(str(path), checkpoint_only=True, repair=True)
