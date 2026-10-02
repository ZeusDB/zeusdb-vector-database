"""The features a saved directory lists, from Python.

What every kind of save lists under `features` in `manifest.json`, a feature
this build does not know refused by its name unless the directory marks it
compatible, one marked compatible opened and dropped by the next save, a
record that does not describe its directory refused, and the directories
that list no features, below 4.0.0 and the 4.0.0 ones saved before the record
existed, opening as they did.
"""

import gc
import json
import shutil

import numpy as np
import pytest
from helpers import as_old_wire
from zeusdb_vector_database import VectorDatabase

DIM = 8
PQ = {"type": "pq", "subvectors": 4, "bits": 4, "training_size": 1000}
INT8 = {"type": "int8", "training_size": 1000}


def create(**kw):
    kw.setdefault("expected_size", 2000)
    return VectorDatabase().create("hnsw", dim=DIM, space="l2", **kw)


def filled(index, n, sparse=None, text=None):
    rng = np.random.default_rng(n)
    batch = {"ids": [f"r{i}" for i in range(n)],
             "embeddings": rng.standard_normal((n, DIM)).astype(np.float32)}
    if sparse:
        batch["sparse"] = [{"dims": [i % 7, 7 + i % 5], "values": [1.0, 2.0]} for i in range(n)]
    if text:
        batch["texts"] = [f"word{i % 11} common" for i in range(n)]
    result = index.add(batch)
    assert result.is_success(), result.errors
    return index


def manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def rewrite(path, edit):
    m = manifest(path)
    edit(m)
    (path / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return path


def forged(source, target, edit):
    shutil.copytree(source, target)
    return rewrite(target, edit)


def load(path):
    return VectorDatabase().load(str(path))


def drop(index):
    """Let go of the index so a journal it holds open is closed."""
    del index
    gc.collect()


@pytest.fixture
def dense_dir(tmp_path):
    path = tmp_path / "dense.zdb"
    filled(create(), 30).save(str(path))
    return path


SHAPES = {
    "dense": (lambda: filled(create(), 30), {"identity": "compatible"}),
    "pq_quantized_with_raw": (
        lambda: filled(create(quantization_config={**PQ, "storage_mode": "quantized_with_raw"}), 1100),
        {"identity": "compatible", "pq": "incompatible"}),
    "pq_quantized_only": (
        lambda: filled(create(quantization_config={**PQ, "storage_mode": "quantized_only"}), 1100),
        {"identity": "compatible", "pq": "incompatible"}),
    "pq_collecting": (
        lambda: filled(create(quantization_config=dict(PQ)), 30),
        {"identity": "compatible", "pq": "incompatible"}),
    "int8": (lambda: filled(create(quantization_config=dict(INT8)), 1100),
             {"identity": "compatible", "int8": "incompatible"}),
    "sparse": (lambda: filled(create(sparse={"name": "terms"}), 30, sparse=True),
               {"identity": "compatible", "sparse": "incompatible"}),
    "text": (lambda: filled(create(sparse={"name": "text", "tokenizer": "simple"}), 30, text=True),
             {"identity": "compatible", "sparse": "incompatible", "text": "incompatible"}),
}


# ------------------------------------------------------------
# What a save lists
# ------------------------------------------------------------
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_every_save_lists_the_features_it_holds(tmp_path, shape):
    make, expected = SHAPES[shape]
    index = make()
    path = tmp_path / f"{shape}.zdb"
    index.save(str(path))
    m = manifest(path)
    assert m["features"] == expected
    assert list(m)[:3] == ["format_version", "features", "identity"]
    assert m["format_version"] == "4.0.0"
    assert len(load(path)) == len(index)


def test_a_journaled_directory_lists_its_journal(tmp_path):
    index = filled(create(sparse={"name": "text", "tokenizer": "simple"}), 20, text=True)
    path = tmp_path / "journaled.zdb"
    index.journal_to(str(path))
    assert manifest(path)["features"] == {"identity": "compatible", "journal": "incompatible",
                                          "sparse": "incompatible", "text": "incompatible"}
    drop(index)
    assert len(load(path)) == 20


# ------------------------------------------------------------
# A feature this build does not know
# ------------------------------------------------------------
def test_a_feature_this_build_does_not_know_is_refused_by_name(dense_dir, tmp_path):
    one = forged(dense_dir, tmp_path / "one.zdb",
                 lambda m: m["features"].__setitem__("a_later_feature", "incompatible"))
    with pytest.raises(RuntimeError) as refused:
        load(one)
    assert str(refused.value) == (
        "The directory holds the feature 'a_later_feature', which this build does not know. "
        "Its manifest does not mark it compatible, so a build without it cannot open the "
        "directory. This build knows identity, int8, journal, pq, sparse and text. The "
        "directory was written by a newer release of zeusdb-vector-database, so upgrade the "
        "package to open it.")

    two = forged(dense_dir, tmp_path / "two.zdb", lambda m: m["features"].update(
        {"zz_later": "incompatible", "a_later_feature": "read_only", "a_later_note": "compatible"}))
    with pytest.raises(RuntimeError, match=r"the features 'a_later_feature' and 'zz_later', which "
                                           r"this build does not know\. Its manifest does not mark "
                                           r"them compatible, so a build without them cannot open"):
        load(two)


def test_a_compatible_feature_this_build_does_not_know_opens_and_is_not_kept(dense_dir, tmp_path):
    noted = forged(dense_dir, tmp_path / "noted.zdb",
                   lambda m: m["features"].__setitem__("a_later_note", "compatible"))
    index = load(noted)
    assert len(index) == 30
    again = tmp_path / "again.zdb"
    index.save(str(again))
    assert manifest(again)["features"] == {"identity": "compatible"}


# ------------------------------------------------------------
# A record that does not describe its directory
# ------------------------------------------------------------
INVALID = "manifest.json lists features that do not describe the directory: "


@pytest.mark.parametrize("edit, detail", [
    (lambda f: f.__setitem__("identity", "incompatible"), "it marks identity incompatible"),
    (lambda f: f.pop("identity"), "it lists no feature, and the directory holds identity"),
    (lambda f: f.__setitem__("journal", "incompatible"),
     "it lists identity and journal, and the directory holds identity"),
    (lambda f: f.__setitem__("pq", "incompatible"),
     "it lists identity and pq, and the directory holds identity"),
], ids=["identity_under_another_mark", "identity_unlisted", "journal_not_held", "pq_not_held"])
def test_a_record_that_does_not_describe_its_directory_is_refused(dense_dir, tmp_path, edit, detail):
    path = forged(dense_dir, tmp_path / "forged.zdb", lambda m: edit(m["features"]))
    with pytest.raises(RuntimeError) as refused:
        load(path)
    assert str(refused.value) == (
        INVALID + detail + ". A saved directory lists every feature it holds and no other, and "
        "this build marks identity compatible and int8, journal, pq, sparse and text incompatible.")


def test_a_held_feature_left_unlisted_is_refused(tmp_path):
    source = tmp_path / "text.zdb"
    filled(create(sparse={"name": "text", "tokenizer": "simple"}), 20, text=True).save(str(source))
    path = forged(source, tmp_path / "no-text.zdb", lambda m: m["features"].pop("text"))
    with pytest.raises(RuntimeError, match=INVALID + "it lists identity and sparse, and the "
                                           "directory holds identity, sparse and text"):
        load(path)


# ------------------------------------------------------------
# Directories that list no features
# ------------------------------------------------------------
@pytest.mark.parametrize("version", ["1.1.0", "4.0.0"])
def test_a_directory_listing_no_features_opens_and_lists_them_at_its_next_save(tmp_path, version):
    """1.1.0 in the old wire, as 0.11.0 wrote it, and 4.0.0 framed, as the
    builds before the record wrote it. Each opens as it did, and its next save
    lists what it holds."""
    path = tmp_path / "older.zdb"
    filled(create(quantization_config={**PQ, "storage_mode": "quantized_only"}), 1100).save(str(path))
    if version != "4.0.0":
        as_old_wire(path, version)
    rewrite(path, lambda m: m.pop("features"))
    index = load(path)
    assert len(index) == 1100 and index.is_quantized()
    again = tmp_path / "again.zdb"
    index.save(str(again))
    assert manifest(again)["features"] == {"identity": "compatible", "pq": "incompatible"}


def test_below_the_fourth_major_the_version_decides(dense_dir, tmp_path):
    """No release before 4.0.0 listed features, so below it a record is read
    for a feature this build does not know and is not held to the directory."""
    def third(edit):
        def stamp(m):
            m["format_version"] = "3.0.0"
            edit(m["features"])
        return stamp

    listed = forged(dense_dir, tmp_path / "listed.zdb",
                    third(lambda f: f.__setitem__("journal", "incompatible")))
    assert len(load(listed)) == 30
    later = forged(dense_dir, tmp_path / "later.zdb",
                   third(lambda f: f.__setitem__("a_later_feature", "incompatible")))
    with pytest.raises(RuntimeError, match="The directory holds the feature 'a_later_feature'"):
        load(later)
