"""Scalar quantization, `type: 'int8'`, from the Python surface.

The declaration and every refusal a caller can reach, training on the
`add()` that fills the sample, what `get_stats()`, `get_quantization_info()`
and `info()` report, the page against a raw index under every space, the
directory a save writes and its version, the two artefacts held to their
bounds, the journal, and the mutations that keep the scales.
"""
import json
import shutil
import struct
import warnings

import numpy as np
import pytest
from helpers import artefact_digest

from zeusdb_vector_database import VectorDatabase, _engine

DIM = 16
TRAINING = 1000
N = 1300


def _vectors(n=N, dim=DIM, seed=157):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((12, dim)) * 3.0
    return (centres[rng.integers(0, 12, n)] + rng.standard_normal((n, dim))).astype(np.float32)


def _config(**overrides):
    config = {"type": "int8", "training_size": TRAINING}
    config.update(overrides)
    return config


def _create(space="l2", **kwargs):
    vdb = VectorDatabase()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vdb.create("hnsw", dim=DIM, space=space, m=8, ef_construction=60,
                          expected_size=2000, quantization_config=_config(**kwargs))


def _trained(space="l2", n=N, seed=157):
    index = _create(space)
    data = _vectors(n, seed=seed)
    result = index.add({"ids": [f"r{i}" for i in range(n)], "vectors": data})
    assert result.is_success(), result.errors
    assert index.is_quantized(), index.get_storage_mode()
    return index, data


def _brute(space, query, data):
    if space == "l2":
        return np.linalg.norm(data - query, axis=1)
    if space == "l1":
        return np.abs(data - query).sum(axis=1)
    if space == "dot":
        return 1.0 - data @ query
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    return 1.0 - unit @ (query / np.linalg.norm(query))


def _recall_at_5(index, space, data, queries):
    hits = 0
    for q in queries:
        truth = set(np.argsort(_brute(space, q, data), kind="stable")[:5].tolist())
        page = {int(h["id"][1:]) for h in index.search(q, top_k=5)}
        hits += len(truth & page)
    return hits / (5 * len(queries))


# ------------------------------------------------------------------
# The declaration and its refusals
# ------------------------------------------------------------------

def test_the_scalar_declaration_is_read_and_defaulted():
    index = _create()
    assert index.has_quantization()
    assert not index.can_use_quantization()
    assert not index.is_quantized()
    assert index.get_storage_mode() == "raw_collecting_for_training"
    stats = index.get_stats()
    assert stats["quantization_type"] == "int8"
    assert stats["quantization_scale"] == "per_dimension"
    assert stats["quantization_training_size"] == str(TRAINING)
    assert stats["storage_mode"] == "quantized_only"
    assert stats["quantization_trained"] == "false"
    assert stats["quantization_active"] == "false"
    assert "quantization=int8(scale=per_dimension, untrained, inactive, compression=4.0x)" in index.info()
    info = index.get_quantization_info()
    assert info["type"] == "int8"
    assert info["scale"] == "per_dimension"
    assert info["training_size"] == TRAINING
    assert info["is_trained"] is False
    assert "subvectors" not in info and "bits" not in info and "total_centroids" not in info


def test_training_size_defaults_to_ten_thousand():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=DIM, expected_size=20_000, quantization_config={"type": "int8"})
    assert index.get_quantization_info()["training_size"] == 10_000
    assert index.get_stats()["quantization_training_size"] == "10000"


@pytest.mark.parametrize("space", ["cosine", "l2", "l1", "dot"])
def test_every_space_admits_the_scalar_scheme(space):
    index = _create(space)
    assert index.get_space() == space
    assert index.get_stats()["quantization_type"] == "int8"


@pytest.mark.parametrize("overrides, fragment", [
    ({"subvectors": 4}, "names 'subvectors' under type 'int8'"),
    ({"bits": 8}, "names 'bits' under type 'int8'"),
    ({"scale": "per_vector"}, "scale must be 'per_dimension', got 'per_vector'"),
    ({"training_size": 999}, "training_size must be at least 1000"),
    ({"max_training_vectors": 999}, "max_training_vectors (999) must be >= training_size (1000)"),
    ({"storage_mode": "quantized_with_raw"}, "storage_mode='quantized_with_raw' is not available under type 'int8'"),
    ({"storage_mode": "sideways"}, "Invalid storage_mode: 'sideways'"),
])
def test_every_scalar_refusal_names_its_rule(overrides, fragment):
    with pytest.raises(ValueError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        _create(**overrides)


def test_the_engine_refuses_the_same_declarations():
    def create(config):
        return _engine._create_hnsw_index(dim=DIM, space="l2", m=8, ef_construction=60,
                                          expected_size=2000, quantization_config=config)

    for config, fragment in [
        ({"type": "int8", "subvectors": 4, "training_size": TRAINING}, "names 'subvectors' under type 'int8'"),
        ({"type": "int8", "bits": 4, "training_size": TRAINING}, "names 'bits' under type 'int8'"),
        ({"type": "int8", "scale": "per_vector", "training_size": TRAINING}, "scale must be 'per_dimension'"),
        ({"type": "int8", "training_size": 10}, "training_size must be at least 1000"),
        ({"type": "int8", "training_size": TRAINING, "storage_mode": "quantized_with_raw"},
         "not available under type 'int8'"),
        ({"type": "pq", "subvectors": 4, "bits": 8, "training_size": TRAINING, "scale": "per_dimension"},
         "names 'scale' under type 'pq'"),
        ({"type": "opq", "training_size": TRAINING}, "Supported types: 'pq' and 'int8'"),
    ]:
        with pytest.raises(ValueError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
            create(config)
    # A key set to None is the key left out, under both schemes.
    assert create({"type": "int8", "subvectors": None, "training_size": TRAINING}).has_quantization()
    assert create({"type": "pq", "subvectors": 4, "bits": 8, "training_size": TRAINING,
                   "scale": None}).has_quantization()


def test_scale_is_refused_under_pq_from_the_factory():
    vdb = VectorDatabase()
    with pytest.raises(ValueError, match="names 'scale' under type 'pq'"):
        vdb.create("hnsw", dim=DIM, expected_size=2000,
                   quantization_config={"type": "pq", "subvectors": 4, "training_size": TRAINING,
                                        "scale": "per_dimension"})


def test_a_declared_size_below_the_training_size_warns():
    vdb = VectorDatabase()
    with pytest.warns(UserWarning, match="training will never trigger"):
        vdb.create("hnsw", dim=DIM, expected_size=500,
                   quantization_config={"type": "int8", "training_size": TRAINING})


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def test_training_fires_on_the_record_that_fills_the_sample():
    index = _create()
    data = _vectors()
    index.add({"ids": [f"r{i}" for i in range(TRAINING - 1)], "vectors": data[:TRAINING - 1]})
    assert not index.is_quantized()
    assert index.training_vectors_needed() == 1
    assert index.get_stats()["training_progress"] == f"{TRAINING - 1}/{TRAINING} (99.9%)"
    index.add({"id": f"r{TRAINING - 1}", "vector": data[TRAINING - 1]})
    assert index.is_quantized()
    assert index.can_use_quantization()
    assert index.get_storage_mode() == "quantized_active"
    assert index.training_vectors_needed() == 0
    assert index.get_stats()["quantization_saturated_values"] == "0"
    index.add({"ids": [f"r{i}" for i in range(TRAINING, N)], "vectors": data[TRAINING:]})
    stats = index.get_stats()
    assert stats["quantization_trained"] == "true"
    assert stats["quantization_active"] == "true"
    assert stats["quantized_codes_stored"] == str(N)
    assert stats["raw_vectors_stored"] == "0"
    assert stats["raw_vectors_memory_mb"] == "0.00"
    assert stats["quantization_compression_ratio"] == "4.0x"
    assert stats["storage_strategy"] == "memory_optimized"
    assert stats["raw_vectors_retained"] == "none_once_trained"
    assert stats["training_progress"] == f"{TRAINING}/{TRAINING} (100.0%)"
    assert int(stats["quantization_saturated_values"]) > 0
    assert "codebook_memory_mb" not in stats
    assert "rerank_calibrated" not in stats
    assert "scale_memory_mb" in stats
    assert index.info().endswith("quantization=int8(scale=per_dimension, trained, active, compression=4.0x))")
    info = index.get_quantization_info()
    assert info["is_trained"] is True
    assert info["compression_ratio"] == 4.0
    assert info["memory_mb"] > 0


def test_the_total_is_the_sum_of_the_scalar_parts():
    index, _ = _trained()
    stats = index.get_stats()
    parts = sum(float(stats[k]) for k in (
        "graph_memory_mb", "raw_vectors_memory_mb", "quantized_codes_memory_mb",
        "scale_memory_mb", "index_bookkeeping_memory_mb"))
    assert float(stats["total_memory_mb"]) == pytest.approx(parts, abs=0.05)


def test_a_value_past_the_sample_range_is_clipped_not_refused():
    index, data = _trained()
    before = int(index.get_stats()["quantization_saturated_values"])
    wild = (np.sign(data[0]) * np.abs(data).max(axis=0) * 10).astype(np.float32)
    result = index.add({"id": "wild", "vector": wild})
    assert result.is_success(), result.errors
    assert int(index.get_stats()["quantization_saturated_values"]) == before + DIM
    assert index.search(wild, top_k=1)[0]["id"] == "wild"


# ------------------------------------------------------------------
# The page
# ------------------------------------------------------------------

@pytest.mark.parametrize("space", ["cosine", "l2", "l1", "dot"])
def test_the_scalar_page_finds_what_brute_force_finds(space):
    index, data = _trained(space)
    rng = np.random.default_rng(99)
    queries = data[-50:] + 0.1 * rng.standard_normal((50, DIM)).astype(np.float32)
    recall = _recall_at_5(index, space, data, queries)
    assert recall >= 0.8, f"{space}: {recall}"


def test_a_scalar_l2_score_is_a_distance_to_the_decoded_record():
    index, data = _trained("l2")
    for i in range(0, 200, 20):
        hits = index.search(data[i], top_k=3, return_vector=True)
        for hit in hits:
            decoded = np.asarray(hit["vector"], dtype=np.float32)
            assert hit["score"] == pytest.approx(float(np.linalg.norm(data[i] - decoded)), abs=1e-4)
            record = int(hit["id"][1:])
            assert hit["score"] == pytest.approx(float(np.linalg.norm(data[i] - data[record])), abs=0.3)


def test_a_scalar_cosine_record_decodes_at_unit_length():
    index, data = _trained("cosine")
    records = index.get_records(["r3", "r4"], return_vector=True)
    for record in records:
        vector = np.asarray(record["vector"], dtype=np.float32)
        assert len(vector) == DIM
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)
    hit = index.search(data[3], top_k=1, return_vector=True)[0]
    assert hit["id"] == "r3"
    assert 0.0 <= hit["score"] <= 0.05
    assert float(np.linalg.norm(hit["vector"])) == pytest.approx(1.0, abs=1e-5)


def test_rerank_and_ef_search_leave_a_scalar_page_alone():
    index, data = _trained("l2")
    plain = [(h["id"], h["score"]) for h in index.search(data[7], top_k=5)]
    reranked = [(h["id"], h["score"]) for h in index.search(data[7], top_k=5, rerank=4)]
    assert plain == reranked
    assert index.search(data[7], top_k=5, ef_search=300)[0]["id"] == "r7"


def test_a_filtered_scalar_search_scans_the_admitted_records():
    index = _create("l2")
    data = _vectors()
    index.add({"ids": [f"r{i}" for i in range(N)], "vectors": data,
               "metadatas": [{"cat": "a" if i % 2 == 0 else "b"} for i in range(N)]})
    assert index.is_quantized()
    hits = index.search(data[10], top_k=5, filter={"cat": "b"})
    assert all(int(h["id"][1:]) % 2 == 1 for h in hits)
    truth = np.argsort(np.linalg.norm(data - data[10], axis=1), kind="stable")
    truth = [i for i in truth.tolist() if i % 2 == 1][:5]
    assert [int(h["id"][1:]) for h in hits] == truth


def test_query_and_explain_run_over_a_scalar_arm():
    index, data = _trained("l2")
    page = index.query(arms=[{"vector": data[11]}], top_k=3)
    assert page[0]["id"] == "r11"
    plan = index.explain(arms=[{"vector": data[11]}], top_k=3)
    assert plan["arms"][0]["kind"] == "dense"


# ------------------------------------------------------------------
# The mutations
# ------------------------------------------------------------------

def test_every_mutation_keeps_the_scales():
    index, data = _trained("l2")
    assert index.remove_point("r10")
    assert index.compact() > 0
    assert index.is_quantized()
    assert index.search(data[20], top_k=1)[0]["id"] == "r20"
    index.add({"id": "r20", "vector": data[N - 1]}, overwrite=True)
    assert "r20" in {h["id"] for h in index.search(data[N - 1], top_k=2)}
    index.rebuild(m=6)
    assert index.is_quantized()
    assert index.rebuild_with_quantization() is True
    assert index.clear() == N - 1
    assert index.is_quantized()
    assert index.get_storage_mode() == "quantized_active"
    assert index.get_stats()["quantization_saturated_values"] == "0"
    index.add({"ids": ["z1", "z2"], "vectors": data[:2]})
    assert index.search(data[1], top_k=1)[0]["id"] == "z2"
    assert index.get_stats()["quantized_codes_stored"] == "2"


# ------------------------------------------------------------------
# The directory
# ------------------------------------------------------------------

def test_a_scalar_directory_carries_its_artefacts_at_the_scalar_minor(tmp_path):
    index, data = _trained("cosine")
    path = tmp_path / "scalar.zdb"
    index.save(str(path))
    names = sorted(p.name for p in path.iterdir())
    assert names == ["config.json", "hnsw_index.zdbgraph", "int8_rows.zdbint8", "int8_scales.zdbint8",
                     "manifest.json", "mappings.bin", "metadata.json", "quantization.json"]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "1.2.0"
    assert manifest["quantization_trained"] is True
    assert "int8_scales.zdbint8" in manifest["files_included"]
    assert "int8_rows.zdbint8" in manifest["files_included"]
    assert "int8_scales.zdbint8" in manifest["file_digests"]
    assert manifest["compression_info"]["compression_ratio"] == pytest.approx(DIM * 4 / (DIM + 4))
    quantization = json.loads((path / "quantization.json").read_text(encoding="utf-8"))
    assert quantization["type"] == "int8"
    assert quantization["scale"] == "per_dimension"
    assert quantization["is_trained"] is True
    assert quantization["saturated_values"] == int(index.get_stats()["quantization_saturated_values"])
    for key in ("subvectors", "bits", "pq_config", "memory_stats", "rerank_calibration"):
        assert key not in quantization
    # The scales artefact is dim floats under an 80 byte frame.
    assert (path / "int8_scales.zdbint8").stat().st_size == 80 + DIM * 4
    # The rows artefact is one u32 id and one row of dim + 4 bytes a record.
    assert (path / "int8_rows.zdbint8").stat().st_size == 80 + N * (4 + DIM + 4)

    loaded = VectorDatabase().load(str(path))
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"
    assert loaded.get_vector_count() == N
    # Every quantization key survives the round trip. `graph_memory_mb` is
    # left out because a graph built by insertion carries the slack of its
    # last doubling and a graph restored from the dump carries none.
    for key in ("quantization_type", "quantization_scale", "quantization_trained", "quantization_active",
                "quantized_codes_stored", "quantization_compression_ratio", "quantization_saturated_values",
                "training_progress", "scale_memory_mb"):
        assert loaded.get_stats()[key] == index.get_stats()[key], key
    for i in range(0, 100, 10):
        want = [(h["id"], h["score"]) for h in index.search(data[i], top_k=5)]
        got = [(h["id"], h["score"]) for h in loaded.search(data[i], top_k=5)]
        assert want == got
    assert loaded.get_records(["r7"], return_vector=True)[0]["vector"] == \
        index.get_records(["r7"], return_vector=True)[0]["vector"]

    again = tmp_path / "again.zdb"
    loaded.save(str(again))
    for name in ("hnsw_index.zdbgraph", "int8_scales.zdbint8", "int8_rows.zdbint8"):
        assert (again / name).read_bytes() == (path / name).read_bytes(), name


def test_a_collecting_scalar_directory_trains_after_it_is_loaded(tmp_path):
    index = _create("l2")
    data = _vectors()
    index.add({"ids": [f"r{i}" for i in range(600)], "vectors": data[:600]})
    path = tmp_path / "collecting.zdb"
    index.save(str(path))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "1.2.0"
    assert "vectors.bin" in manifest["files_included"]
    assert "int8_scales.zdbint8" not in manifest["files_included"]
    loaded = VectorDatabase().load(str(path))
    assert loaded.get_storage_mode() == "raw_collecting_for_training"
    assert loaded.training_vectors_needed() == 400
    loaded.add({"ids": [f"r{i}" for i in range(600, N)], "vectors": data[600:]})
    assert loaded.is_quantized()
    assert loaded.get_records(["r5"], return_vector=True)[0]["vector"] == \
        _trained("l2")[0].get_records(["r5"], return_vector=True)[0]["vector"]


def test_the_rows_rebuild_the_graph_when_the_dump_is_refused(tmp_path):
    index, data = _trained("l2")
    path = tmp_path / "scalar.zdb"
    index.save(str(path))
    rows = (path / "int8_rows.zdbint8").read_bytes()
    dump = path / "hnsw_index.zdbgraph"
    dump.write_bytes(dump.read_bytes()[:-1])
    loaded = VectorDatabase().load(str(path))
    assert loaded.is_quantized()
    assert loaded.get_vector_count() == N
    assert loaded.get_stats()["graph_nodes"] == str(N)
    for i in range(0, N, 100):
        assert loaded.search(data[i], top_k=1)[0]["id"] == f"r{i}"
    again = tmp_path / "rebuilt.zdb"
    loaded.save(str(again))
    assert (again / "int8_rows.zdbint8").read_bytes() == rows


def _relabel(path, version):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = version
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_the_scalar_minor_sits_on_each_major(tmp_path):
    vdb = VectorDatabase()
    data = _vectors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spaced = vdb.create("hnsw", dim=DIM, space="l2", m=8, ef_construction=60, expected_size=2000,
                            quantization_config=_config(), sparse={"name": "terms"})
    spaced.add({"ids": [f"r{i}" for i in range(N)], "vectors": data})
    spaced.save(str(tmp_path / "spaced.zdb"))
    manifest = json.loads((tmp_path / "spaced.zdb" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "2.1.0"
    assert vdb.load(str(tmp_path / "spaced.zdb")).is_quantized()

    journaled, _ = _trained("l2")
    journaled.journal_to(str(tmp_path / "journaled.zdb"))
    manifest = json.loads((tmp_path / "journaled.zdb" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "3.1.0"

    # This build reads any 1.x, so the older minor over a scalar directory
    # opens here; the older reader's refusal is held outside the suite.
    plain, _ = _trained("l2")
    plain.save(str(tmp_path / "plain.zdb"))
    _relabel(tmp_path / "plain.zdb", "1.1.0")
    assert vdb.load(str(tmp_path / "plain.zdb")).is_quantized()
    _relabel(tmp_path / "plain.zdb", "4.0.0")
    with pytest.raises(RuntimeError, match="format version 4.0.0 cannot be opened"):
        vdb.load(str(tmp_path / "plain.zdb"))


@pytest.mark.parametrize("name, mutate, fragment", [
    ("int8_scales.zdbint8", lambda b: b[:40] + struct.pack("<Q", DIM + 1) + b[48:], "17 scales"),
    ("int8_scales.zdbint8", lambda b: b[:64] + struct.pack("<f", 0.0) + b[68:], "scale 0 is 0"),
    ("int8_rows.zdbint8", lambda b: b[:64] + struct.pack("<I", 0) + b[68:], "internal id 0"),
    ("int8_rows.zdbint8", lambda b: b[:40] + struct.pack("<Q", N - 1) + b[48:], f"{N - 1} rows"),
])
def test_a_scalar_artefact_outside_its_bounds_is_refused_by_name(tmp_path, name, mutate, fragment):
    index, _ = _trained("l2")
    path = tmp_path / "scalar.zdb"
    index.save(str(path))
    file = path / name
    damaged = bytearray(mutate(file.read_bytes()))
    # The frame's header and trailer checksums cover what was changed, so
    # they are recomputed as the engine's own fuzzer does, over the same
    # ranges: the header up to its checksum, and the payload for the trailer.
    damaged = _restamp(damaged)
    file.write_bytes(bytes(damaged))
    with pytest.raises(RuntimeError, match=fragment):
        VectorDatabase().load(str(path))


def _restamp(blob):
    """Recompute the frame checksums after an edit, with the checksum the
    frame module states: the header's over its first 56 bytes and the
    trailer's over the payload. `artefact_digest` is the engine's checksum
    as the manifest repairer reproduces it, as sixteen hex digits."""
    header_sum = int(artefact_digest(bytes(blob[:56])), 16)
    blob[56:64] = struct.pack("<Q", header_sum)
    end = len(blob) - 16
    payload_sum = int(artefact_digest(bytes(blob[64:end])), 16)
    blob[end:end + 8] = struct.pack("<Q", payload_sum)
    return blob


def test_a_scalar_directory_without_its_scales_is_refused(tmp_path):
    index, _ = _trained("l2")
    path = tmp_path / "scalar.zdb"
    index.save(str(path))
    (path / "int8_scales.zdbint8").unlink()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files_included"].remove("int8_scales.zdbint8")
    del manifest["file_digests"]["int8_scales.zdbint8"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # The same class the missing codebook raises: the file is what is absent.
    with pytest.raises(FileNotFoundError, match="int8_scales.zdbint8 is missing"):
        VectorDatabase().load(str(path))


def test_the_journal_replays_a_scalar_training(tmp_path):
    index = _create("l2")
    data = _vectors()
    index.add({"ids": [f"r{i}" for i in range(600)], "vectors": data[:600]})
    index.journal_to(str(tmp_path / "journal.zdb"))
    index.add({"ids": [f"r{i}" for i in range(600, N)], "vectors": data[600:]})
    assert index.is_quantized()
    assert index.remove_point("r700")
    saturated = index.get_stats()["quantization_saturated_values"]
    pages = [[(h["id"], h["score"]) for h in index.search(data[i], top_k=5)] for i in range(0, N, 130)]
    del index
    recovered = VectorDatabase().load(str(tmp_path / "journal.zdb"))
    assert recovered.is_quantized()
    assert recovered.get_vector_count() == N - 1
    assert recovered.get_stats()["quantization_saturated_values"] == saturated
    assert [[(h["id"], h["score"]) for h in recovered.search(data[i], top_k=5)]
            for i in range(0, N, 130)] == pages
    manifest = json.loads((tmp_path / "journal.zdb" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "3.1.0"


def test_a_scalar_index_loads_in_the_hostile_test_layout(tmp_path):
    """The directory a scalar save writes is what the loader's presence
    check names, and a copy with nothing changed opens."""
    index, _ = _trained("l2")
    path = tmp_path / "scalar.zdb"
    index.save(str(path))
    copy = tmp_path / "copy.zdb"
    shutil.copytree(path, copy)
    assert VectorDatabase().load(str(copy)).get_vector_count() == N
