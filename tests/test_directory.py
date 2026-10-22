"""A directory at format 5.0.0, from Python.

Each record's row is held once, in the graph dump, beside `vectors.bin` only
where a product quantized index keeps raw vectors. `metadata.json` is compact
and names only the records that hold a field. `quantization.json` and the
manifest carry the fields a load reads. A dump refused on its header, its
identity or its rows refuses the load, naming the check, and a dump whose
topology fails is rebuilt from the rows it holds, as
`ZEUSDB_LOAD_REBUILD_GRAPH` asks. A directory holding a feature this build
does not know, which its manifest marks read_only, opens read only.
"""

import json
import shutil
import struct

import numpy as np
import pytest
from helpers import artefact_digest, checksum, repair_manifest
from zeusdb_vector_database import VectorDatabase

DIM = 8
PQ_ONLY = {"type": "pq", "subvectors": 4, "bits": 4, "training_size": 1000,
           "storage_mode": "quantized_only"}
PQ_RAW = {"type": "pq", "subvectors": 4, "bits": 4, "training_size": 1000,
          "storage_mode": "quantized_with_raw"}
INT8 = {"type": "int8", "training_size": 1000}
DUMP = "hnsw_index.zdbgraph"


def build(n, quantization=None, space="l2"):
    kw = {"quantization_config": quantization} if quantization else {}
    if quantization and quantization.get("storage_mode") == "quantized_with_raw":
        with pytest.warns(UserWarning):
            index = VectorDatabase().create("hnsw", dim=DIM, space=space, expected_size=2000, **kw)
    else:
        index = VectorDatabase().create("hnsw", dim=DIM, space=space, expected_size=2000, **kw)
    rng = np.random.default_rng(n)
    result = index.add({
        "ids": [f"r{i}" for i in range(n)],
        "embeddings": rng.standard_normal((n, DIM)).astype(np.float32),
        "metadatas": [{"i": i} if i % 10 == 0 else {} for i in range(n)],
    })
    assert result.is_success(), result.errors
    return index


def manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def records(index, n):
    found = index.get_records([f"r{i}" for i in range(n)], return_vector=True)
    return {r["id"]: (r["metadata"], np.asarray(r["vector"], dtype=np.float32).tobytes())
            for r in found}


def regions(blob):
    """Where each region of a version 2 dump starts: the identity, the rows,
    the topology and the trailer."""
    identity = 128
    rows = identity + struct.unpack_from("<Q", blob, 48)[0]
    topology = rows + struct.unpack_from("<Q", blob, 56)[0]
    trailer = topology + struct.unpack_from("<Q", blob, 64)[0]
    return identity, rows, topology, trailer


def damaged(source, target, damage):
    """A copy of `source` whose dump `damage` changes in place."""
    shutil.copytree(source, target)
    blob = bytearray((target / DUMP).read_bytes())
    damage(blob)
    (target / DUMP).write_bytes(bytes(blob))
    return target


@pytest.mark.parametrize("label, quantization, kept", [
    ("raw", None, ()),
    ("pq-only", PQ_ONLY, ("pq_centroids.bin",)),
    ("pq-raw", PQ_RAW, ("pq_centroids.bin", "vectors.bin")),
    ("int8", INT8, ("int8_scales.zdbint8",)),
])
def test_each_row_is_held_once_in_the_dump(tmp_path, label, quantization, kept):
    """A save writes each record's row into the dump alone: no pq_codes.bin,
    no int8_rows.zdbint8, and vectors.bin only beside a product quantized
    graph that keeps raw vectors. The directory opens with every record, its
    graph read and not rebuilt, and a second save writes the same files."""
    n = 1100 if quantization else 200
    index = build(n, quantization)
    path = tmp_path / f"{label}.zdb"
    index.save(str(path))
    m = manifest(path)
    assert m["format_version"] == "5.0.0"
    for absent in ("pq_codes.bin", "int8_rows.zdbint8"):
        assert absent not in m["files_included"]
        assert not (path / absent).exists()
    assert ("vectors.bin" in m["files_included"]) is ("vectors.bin" in kept)
    for name in kept:
        assert name in m["files_included"], name
    assert "compression_info" not in m
    blob = (path / DUMP).read_bytes()
    assert struct.unpack_from("<I", blob, 8)[0] == 2
    assert m["total_size_mb"] * 1024 * 1024 == pytest.approx(
        sum(entry["bytes"] for entry in m["file_digests"].values()))

    loaded = VectorDatabase().load(str(path))
    assert loaded.recovery["graph_rebuilt"] is False
    assert records(loaded, n) == records(index, n)
    again = tmp_path / f"{label}-again.zdb"
    loaded.save(str(again))
    for name in m["files_included"]:
        assert (path / name).read_bytes() == (again / name).read_bytes(), name


def test_metadata_is_compact_and_names_the_records_holding_a_field(tmp_path):
    """metadata.json names only the records that hold a field, compact, and
    a load gives every other record its empty entry."""
    index = build(50)
    path = tmp_path / "compact.zdb"
    index.save(str(path))
    text = (path / "metadata.json").read_text(encoding="utf-8")
    assert "\n" not in text
    assert sorted(json.loads(text)) == sorted(f"r{i}" for i in range(0, 50, 10))
    loaded = VectorDatabase().load(str(path))
    assert records(loaded, 50) == records(index, 50)
    assert loaded.get_records(["r1"])[0]["metadata"] == {}


def test_quantization_json_carries_no_derived_fields(tmp_path):
    """quantization.json carries no memory_stats and no pq_config, and the
    manifest no compression_info. A file still holding the two is read with
    both ignored."""
    index = build(1100, PQ_ONLY)
    path = tmp_path / "pq.zdb"
    index.save(str(path))
    quantization = json.loads((path / "quantization.json").read_text(encoding="utf-8"))
    assert "memory_stats" not in quantization
    assert "pq_config" not in quantization
    quantization["memory_stats"] = {"centroid_storage_mb": 0.0, "compression_ratio": 8.0,
                                    "centroids_per_subvector": 16, "total_centroids": 64}
    quantization["pq_config"] = {"dim": 8, "sub_dim": 2, "num_centroids": 16}
    (path / "quantization.json").write_text(json.dumps(quantization, indent=2), encoding="utf-8")
    m = manifest(path)
    data = (path / "quantization.json").read_bytes()
    m["file_digests"]["quantization.json"] = {"bytes": len(data), "checksum": artefact_digest(data)}
    m["total_size_mb"] = sum(e["bytes"] for e in m["file_digests"].values()) / (1024 * 1024)
    (path / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    loaded = VectorDatabase().load(str(path))
    assert records(loaded, 1100) == records(index, 1100)


def test_the_training_fields_are_held_to_create(tmp_path):
    """A product quantized quantization.json whose max_training_vectors is
    below its training_size is refused as create() refuses it."""
    index = build(1100, PQ_ONLY)
    path = tmp_path / "pq.zdb"
    index.save(str(path))
    quantization = json.loads((path / "quantization.json").read_text(encoding="utf-8"))
    quantization["max_training_vectors"] = 900
    (path / "quantization.json").write_text(json.dumps(quantization, indent=2), encoding="utf-8")
    m = manifest(path)
    data = (path / "quantization.json").read_bytes()
    m["file_digests"]["quantization.json"] = {"bytes": len(data), "checksum": artefact_digest(data)}
    m["total_size_mb"] = sum(e["bytes"] for e in m["file_digests"].values()) / (1024 * 1024)
    (path / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    with pytest.raises(ValueError) as refused:
        VectorDatabase().load(str(path))
    assert str(refused.value).endswith(
        "quantization.json: max_training_vectors is 900 and training_size is 1000, and "
        "create() requires the first to be at or above the second, so this file does not "
        "describe an index this build can rebuild.")


def test_a_manifest_field_that_does_not_describe_the_directory_is_refused(tmp_path):
    index = build(30)
    path = tmp_path / "plain.zdb"
    index.save(str(path))
    m = manifest(path)
    m["total_vectors"] = 12
    (path / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError) as refused:
        VectorDatabase().load(str(path))
    assert str(refused.value) == (
        "manifest.json records total_vectors as 12, and config.json records vector_count 30. "
        "A save writes every field of manifest.json from the directory it describes, so this "
        "manifest does not describe the directory. Restore the directory from a copy.")


def test_a_dump_refused_on_its_rows_refuses_the_load(tmp_path):
    index = build(200)
    path = tmp_path / "source.zdb"
    index.save(str(path))

    def flip_a_row(blob):
        _, rows, _, _ = regions(blob)
        blob[rows + 5] ^= 0x10

    target = damaged(path, tmp_path / "rows.zdb", flip_a_row)
    with pytest.raises(RuntimeError) as refused:
        VectorDatabase().load(str(target))
    assert str(refused.value) == (
        "The graph dump hnsw_index.zdbgraph cannot be used, since its rows region fails its "
        "checksum. At format 5.0.0 the dump holds the only copy of each record's stored "
        "vector or code, so the directory cannot open without it. Restore the directory from "
        "a copy, such as one backup() wrote.")


@pytest.mark.parametrize("quantization", [None, PQ_ONLY, INT8])
def test_a_dump_whose_topology_fails_is_rebuilt_from_its_rows(tmp_path, quantization):
    n = 1100 if quantization else 200
    index = build(n, quantization)
    path = tmp_path / "source.zdb"
    index.save(str(path))

    def flip_the_topology(blob):
        _, _, topology, _ = regions(blob)
        blob[topology] ^= 0x01

    target = damaged(path, tmp_path / "topology.zdb", flip_the_topology)
    loaded = VectorDatabase().load(str(target))
    assert loaded.recovery["graph_rebuilt"] is True
    assert records(loaded, n) == records(index, n)


def test_the_rebuild_asked_for_reads_the_rows_and_refuses_damaged_ones(tmp_path, monkeypatch):
    """ZEUSDB_LOAD_REBUILD_GRAPH rebuilds the graph from the dump's rows,
    which it reads and verifies, so a dump whose rows are damaged is refused
    with the variable set too."""
    index = build(200)
    path = tmp_path / "source.zdb"
    index.save(str(path))
    monkeypatch.setenv("ZEUSDB_LOAD_REBUILD_GRAPH", "1")
    loaded = VectorDatabase().load(str(path))
    assert loaded.recovery["graph_rebuilt"] is True
    assert records(loaded, 200) == records(index, 200)

    def flip_a_row(blob):
        _, rows, _, _ = regions(blob)
        blob[rows + 5] ^= 0x10

    target = damaged(path, tmp_path / "rows.zdb", flip_a_row)
    with pytest.raises(RuntimeError, match="its rows region fails its checksum"):
        VectorDatabase().load(str(target))


def test_a_feature_marked_read_only_opens_the_directory_read_only(tmp_path):
    """A directory listing a feature this build does not know, marked
    read_only, opens read only with a RuntimeWarning naming the feature, and
    every change, save, checkpoint, journal and backup raises RuntimeError."""
    index = build(40)
    path = tmp_path / "probe.zdb"
    index.save(str(path))
    m = manifest(path)
    m["features"]["probe"] = "read_only"
    (path / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    reason = ("it holds the feature 'probe', which this build does not know and its "
              "manifest marks read_only")
    with pytest.warns(RuntimeWarning) as caught:
        loaded = VectorDatabase().load(str(path))
    assert any(str(w.message) == (
        f"The directory {path} was opened read only, since {reason}. It stays at format 5.0.0. "
        "Every change, save, checkpoint, journal and backup is refused. Upgrade the package "
        "to change it.") for w in caught), [str(w.message) for w in caught]
    assert loaded.recovery["read_only"] is True
    assert loaded.recovery["read_only_reason"] == reason
    assert records(loaded, 40) == records(index, 40)
    refusal = (f"The directory {path} holds the feature 'probe', which this build does not "
               "know. Its manifest marks the feature read_only, so this build reads every "
               "record the directory holds and refuses every change, every save, a checkpoint, "
               "a journal and a backup, since a directory this build wrote would not hold the "
               "feature. Upgrade the package to change the collection.")
    for call in (
        lambda: loaded.remove_point("r1"),
        lambda: loaded.save(str(path)),
        lambda: loaded.save(str(tmp_path / "elsewhere.zdb")),
        lambda: loaded.backup(str(tmp_path / "copy.zdb")),
        lambda: loaded.checkpoint(),
        lambda: loaded.journal_to(str(tmp_path / "journaled.zdb")),
    ):
        with pytest.raises(RuntimeError) as refused:
            call()
        assert str(refused.value) == refusal
    added = loaded.add({"ids": ["new"], "embeddings": [[0.5] * DIM]})
    assert not added.is_success()
    assert not (tmp_path / "elsewhere.zdb").exists()
    assert not (tmp_path / "copy.zdb").exists()
    assert manifest(path)["features"]["probe"] == "read_only"


def test_the_checksum_helper_reads_a_dump_as_the_crate_wrote_it(tmp_path):
    """The dump's header checksum, taken here by the helper, is the one the
    crate wrote, so the tests that repair a dump repair it as a save does."""
    index = build(20)
    path = tmp_path / "plain.zdb"
    index.save(str(path))
    blob = (path / DUMP).read_bytes()
    assert struct.unpack_from("<Q", blob, 120)[0] == checksum(blob[:120])
    identity, rows, topology, trailer = regions(blob)
    assert struct.unpack_from("<Q", blob, 80)[0] == checksum(blob[identity:rows])
    assert struct.unpack_from("<Q", blob, 88)[0] == checksum(blob[rows:topology])
    assert struct.unpack_from("<Q", blob, 96)[0] == checksum(blob[topology:trailer])
    repair_manifest(path)
    assert VectorDatabase().load(str(path)).get_vector_count() == 20
