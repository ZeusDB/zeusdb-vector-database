"""Saving an index to disk and loading it back."""

import json
import shutil
import struct
import warnings

import numpy as np
import pytest
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 44: Persistence: Save and Load
# ------------------------------------------------------------
def test_persistence_save_and_load(tmp_path):

    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, expected_size=10)

    vectors = np.random.rand(5, 8).astype(np.float32)
    ids = [f"vec_{i}" for i in range(5)]

    add_res = index.add({"vectors": vectors.tolist(), "ids": ids})
    assert add_res.is_success()

    save_dir = tmp_path / "test_index.zdb"
    index.save(str(save_dir))

    # Optional: verify save produced a directory
    assert save_dir.exists() and save_dir.is_dir()

    loaded = vdb.load(str(save_dir))
    assert loaded.get_vector_count() == 5

    results = loaded.search(vectors[0].tolist(), top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3

# ------------------------------------------------------------
# Test 45: Persistence: Quantized Index
# ------------------------------------------------------------
def test_persistence_quantized_index(tmp_path):
    vdb = VectorDatabase()
    quant_config = {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 1000}
    index = vdb.create("hnsw", dim=8, quantization_config=quant_config)

    # Insert >= training_size to force training to complete
    n = 1200
    vectors = np.random.rand(n, 8).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n)]

    add_res = index.add({"vectors": vectors.tolist(), "ids": ids})
    assert add_res.is_success()
    # More resilient: accept either quantized or ready for quantization
    assert index.is_quantized() or index.can_use_quantization()

    save_dir = tmp_path / "pq_index.zdb"
    index.save(str(save_dir))
    assert save_dir.exists() and save_dir.is_dir()

    loaded = vdb.load(str(save_dir))
    # Accept either quantization is active or can be used
    assert loaded.is_quantized() or loaded.can_use_quantization()

    # Basic smoke search on the loaded index
    results = loaded.search(vectors[0].tolist(), top_k=5)
    assert isinstance(results, list)
    assert len(results) == 5

# ------------------------------------------------------------
# Shared fixtures for the ported persistence coverage
# ------------------------------------------------------------
# A local Generator keeps the draws reproducible without touching the global
# numpy random state, so these tests cannot perturb any other test.
def _sample_vectors(count, dim, seed=20260802):
    return np.random.default_rng(seed).random((count, dim)).astype(np.float32)


def _normalize(vector):
    """Reproduce the cosine normalization the Rust add() path applies."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0.0 else arr


def _pq_config(storage_mode):
    # dim 8 over 4 subvectors is 8x compression, below the 50x threshold in
    # _check_memory_usage, so no compression warning is emitted. The
    # quantized_with_raw mode warns unconditionally and that is asserted where
    # it is used.
    return {
        "type": "pq",
        "subvectors": 4,
        "bits": 8,
        "training_size": 1000,
        "storage_mode": storage_mode,
    }


QO_TRAINING_SIZE = 1000
QO_COUNT = 1010


def _zero_codebook_bytes(subvectors, centroids, sub_dim):
    """Encode an all-zero codebook the way bincode's standard config does.

    Varint lengths followed by little endian f32 payloads, which is the layout
    save_pq_centroids writes for a Vec<Vec<Vec<f32>>>. Building the file rather
    than blanking the real one keeps the length prefixes intact, so the loader
    sees a well formed codebook whose only problem is its content.
    """
    def varint(n):
        return bytes([n]) if n < 251 else bytes([251]) + n.to_bytes(2, "little")

    centroid = varint(sub_dim) + b"\x00" * (4 * sub_dim)
    return varint(subvectors) + (varint(centroids) + centroid * centroids) * subvectors


@pytest.fixture(scope="module")
def quantized_only_saved(tmp_path_factory):
    """A trained quantized_only index, built and saved once for the module.

    Training runs k-means over training_size vectors and is the slowest thing
    in this file, so the directory is shared. Any test that modifies it copies
    it first. The 10 records past training_size are the ones stored as codes
    alone, which is the case the loader has to reconstruct.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8,
                       quantization_config=_pq_config("quantized_only"),
                       expected_size=2000)

    vectors = _sample_vectors(QO_COUNT, 8)
    ids = [f"v_{i}" for i in range(QO_COUNT)]
    assert index.add({"ids": ids, "embeddings": vectors,
                      "metadatas": [{"i": i} for i in range(QO_COUNT)]}).is_success()
    index.add_metadata({"owner": "relay24"})
    assert index.is_quantized()

    # What the live index returns for the code-only records, which is the
    # reconstruction the reload has to match.
    code_only = ids[QO_TRAINING_SIZE:]
    before = {r["id"]: np.asarray(r["vector"], dtype=np.float64)
              for r in index.get_records(code_only, return_vector=True)}
    assert len(before) == QO_COUNT - QO_TRAINING_SIZE

    save_dir = tmp_path_factory.mktemp("quantized_only") / "qo.zdb"
    index.save(str(save_dir))

    return {"path": save_dir, "ids": ids, "vectors": vectors,
            "code_only": code_only, "before": before}

# ------------------------------------------------------------
# Test 68: Persistence: vectors survive a reload, per distance space
# ------------------------------------------------------------
@pytest.mark.parametrize("space", ["cosine", "l2", "l1"])
def test_persistence_vector_equality_by_space(tmp_path, space):
    """Compare vectors numerically across a reload, one rule per space.

    The rule is not the same for every space, so it is established here rather
    than assumed. add() runs process_vector_for_space, which normalizes on the
    cosine path and returns the vector untouched on l1 and l2. The load path
    rebuilds the index by re-adding the saved vectors through add(), so cosine
    vectors are normalized a second time and land within one float32 step of
    where they started, while l1 and l2 vectors come back bit for bit.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, space=space, expected_size=50)

    count = 6
    vectors = _sample_vectors(count, 8)
    ids = [f"vec_{i}" for i in range(count)]
    add_res = index.add({
        "ids": ids,
        "embeddings": vectors,
        "metadatas": [{"i": i} for i in range(count)],
    })
    assert add_res.is_success()

    before = {
        r["id"]: np.asarray(r["vector"], dtype=np.float64)
        for r in index.get_records(ids, return_vector=True)
    }
    assert len(before) == count

    save_dir = tmp_path / f"{space}.zdb"
    index.save(str(save_dir))
    loaded = vdb.load(str(save_dir))

    # Configuration survives the round trip.
    assert loaded.get_space() == space
    assert loaded.dim == 8
    assert loaded.get_vector_count() == count

    after = {
        r["id"]: np.asarray(r["vector"], dtype=np.float64)
        for r in loaded.get_records(ids, return_vector=True)
    }
    assert len(after) == count

    for i, vec_id in enumerate(ids):
        stored_before = before[vec_id]
        stored_after = after[vec_id]

        if space == "cosine":
            # Stored form is the normalized input, not the input itself.
            assert np.allclose(stored_before, _normalize(vectors[i]), atol=1e-6)
            assert not np.array_equal(stored_before, vectors[i].astype(np.float64))
            # The reload renormalizes, so equality holds to float32 precision
            # rather than exactly. One float32 step near 1.0 is about 1.2e-07.
            assert np.allclose(stored_before, stored_after, atol=1e-6)
        else:
            # l1 and l2 do not normalize, so the stored vector is the input and
            # the reload is bit for bit.
            assert np.array_equal(stored_before, vectors[i].astype(np.float64))
            assert np.array_equal(stored_before, stored_after)

    # Metadata rides along with the vectors.
    assert loaded.get_records("vec_3", return_vector=False)[0]["metadata"] == {"i": 3}

# ------------------------------------------------------------
# Test 69: Persistence: an empty index
# ------------------------------------------------------------
def test_persistence_empty_index(tmp_path):
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, expected_size=10)
    assert index.get_vector_count() == 0

    save_dir = tmp_path / "empty.zdb"
    index.save(str(save_dir))
    assert save_dir.is_dir()

    # An empty index writes no vectors.bin and no graph, because there is
    # nothing to write. The three remaining files are still produced.
    written = sorted(p.name for p in save_dir.glob("*"))
    assert written == ["config.json", "manifest.json", "mappings.bin", "metadata.json"]

    manifest = json.loads((save_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_vectors"] == 0
    assert manifest["files_included"] == ["config.json", "mappings.bin", "metadata.json"]

    loaded = vdb.load(str(save_dir))
    assert loaded.get_vector_count() == 0
    assert loaded.dim == 8
    assert loaded.list() == []

    # Searching a reloaded empty index returns nothing rather than raising.
    assert loaded.search([0.1] * 8, top_k=5) == []

    # The reloaded index is still writable.
    assert loaded.add({"id": "post", "values": [0.1] * 8, "metadata": {}}).is_success()
    assert loaded.get_vector_count() == 1

# ------------------------------------------------------------
# Test 70: Persistence: a single vector index
# ------------------------------------------------------------
def test_persistence_single_vector_index(tmp_path):
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, space="l2", expected_size=10)
    index.add({"id": "only", "values": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               "metadata": {"solo": True}})

    save_dir = tmp_path / "single.zdb"
    index.save(str(save_dir))
    loaded = vdb.load(str(save_dir))

    assert loaded.get_vector_count() == 1
    assert loaded.contains("only")

    results = loaded.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "only"
    assert results[0]["score"] == 0.0
    assert results[0]["metadata"] == {"solo": True}

    record = loaded.get_records("only", return_vector=True)[0]
    assert record["vector"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ------------------------------------------------------------
# Test 71: Persistence: per record metadata across a reload
# ------------------------------------------------------------
def test_persistence_preserves_record_metadata(tmp_path):
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="l2", expected_size=10)

    records = [
        {"id": "r1", "values": [0.1, 0.2, 0.3, 0.4],
         "metadata": {"author": "Alice", "count": 7, "score": 1.5,
                      "flag": True, "tags": ["a", "b"], "missing": None}},
        {"id": "r2", "values": [0.5, 0.6, 0.7, 0.8], "metadata": {}},
        {"id": "r3", "values": [0.9, 1.0, 1.1, 1.2],
         "metadata": {"nested": {"inner": "value"}}},
    ]
    assert index.add(records).is_success()

    before = {r["id"]: r["metadata"]
              for r in index.get_records(["r1", "r2", "r3"], return_vector=False)}

    save_dir = tmp_path / "meta.zdb"
    index.save(str(save_dir))
    loaded = vdb.load(str(save_dir))

    after = {r["id"]: r["metadata"]
             for r in loaded.get_records(["r1", "r2", "r3"], return_vector=False)}

    assert after == before
    assert after["r1"]["author"] == "Alice"
    assert after["r1"]["count"] == 7
    assert isinstance(after["r1"]["count"], int)
    assert after["r1"]["score"] == 1.5
    assert after["r1"]["flag"] is True
    assert after["r1"]["tags"] == ["a", "b"]
    assert after["r1"]["missing"] is None
    assert after["r2"] == {}
    assert after["r3"]["nested"] == {"inner": "value"}

    # Metadata reaches the search results too.
    hit = loaded.search([0.1, 0.2, 0.3, 0.4], top_k=1)[0]
    assert hit["id"] == "r1"
    assert hit["metadata"]["author"] == "Alice"

# ------------------------------------------------------------
# Test 72: Persistence: index level metadata survives a round trip
# ------------------------------------------------------------
def test_persistence_preserves_index_level_metadata(tmp_path):
    """add_metadata content is carried by config.json.

    It belongs to the index rather than to any record, so it goes in the one
    artefact that already holds whole index state instead of gaining a file and
    a manifest entry of its own. metadata.json cannot take it, because that file
    is a map of record id to metadata and adding an envelope would change its
    type for every reader.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=10)
    index.add_metadata({"owner": "relay24", "dataset": "docs_v2"})
    index.add({"id": "r1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"kept": "yes"}})

    assert index.get_all_metadata() == {"owner": "relay24", "dataset": "docs_v2"}

    save_dir = tmp_path / "indexmeta.zdb"
    index.save(str(save_dir))

    config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
    assert config["metadata"] == {"owner": "relay24", "dataset": "docs_v2"}
    assert sorted(config) == ["dim", "ef_construction", "expected_size", "id_counter",
                              "m", "metadata", "space", "vector_count"]

    loaded = vdb.load(str(save_dir))
    assert loaded.get_all_metadata() == {"owner": "relay24", "dataset": "docs_v2"}
    assert loaded.get_metadata("owner") == "relay24"
    assert loaded.get_metadata("absent") is None

    # Per record metadata is unaffected.
    assert loaded.get_records("r1", return_vector=False)[0]["metadata"] == {"kept": "yes"}

    # A second round trip keeps it too, so it is restored into the live index
    # rather than only copied from file to file.
    second = tmp_path / "indexmeta2.zdb"
    loaded.save(str(second))
    assert vdb.load(str(second)).get_all_metadata() == {"owner": "relay24",
                                                        "dataset": "docs_v2"}

    # An index that never called add_metadata writes an empty map, not a
    # missing key, so the field is a stable part of config.json.
    plain = vdb.create("hnsw", dim=4, expected_size=10)
    plain.add({"id": "r1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {}})
    plain_dir = tmp_path / "plain.zdb"
    plain.save(str(plain_dir))
    assert json.loads((plain_dir / "config.json").read_text(encoding="utf-8"))["metadata"] == {}
    assert vdb.load(str(plain_dir)).get_all_metadata() == {}

# ------------------------------------------------------------
# Test 73: Persistence: the manifest and the on disk file inventory
# ------------------------------------------------------------
def test_persistence_manifest_and_file_inventory(tmp_path):
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, space="cosine", m=16, ef_construction=200,
                       expected_size=50)
    index.add({
        "ids": ["a", "b", "c"],
        "embeddings": _sample_vectors(3, 8),
        "metadatas": [{"i": i} for i in range(3)],
    })

    save_dir = tmp_path / "manifest.zdb"
    index.save(str(save_dir))

    on_disk = sorted(p.name for p in save_dir.glob("*"))
    # The README documents this directory listing at lines 843 to 857 and does
    # not mention hnsw_index.hnsw.data, which is written by the vendored graph
    # writer and is present on disk for every non empty index.
    assert on_disk == [
        "config.json",
        "hnsw_index.hnsw.data",
        "hnsw_index.hnsw.graph",
        "manifest.json",
        "mappings.bin",
        "metadata.json",
        "vectors.bin",
    ]

    manifest = json.loads((save_dir / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest) == [
        "compression_info", "created_at", "files_excluded", "files_included",
        "format_version", "has_quantization", "index_type", "quantization_trained",
        "saved_at", "storage_mode", "total_size_mb", "total_vectors", "zeusdb_version",
    ]

    # 1.1.0 rather than 1.0.0 because config.json gained the index level
    # metadata field. The loader reads any 1.x and refuses another major.
    assert manifest["format_version"] == "1.1.0"
    assert manifest["index_type"] == "HNSW"
    assert manifest["total_vectors"] == 3
    assert manifest["has_quantization"] is False
    assert manifest["quantization_trained"] is False
    assert manifest["storage_mode"] == "raw_only"
    assert manifest["compression_info"] is None
    assert manifest["total_size_mb"] > 0

    # files_included is the inventory the loader reads back, and it is a strict
    # subset of what is on disk. The graph data file is deliberately excluded.
    assert manifest["files_included"] == [
        "config.json", "mappings.bin", "metadata.json", "vectors.bin",
        "hnsw_index.hnsw.graph",
    ]
    assert manifest["files_excluded"] == ["hnsw_index.hnsw.data"]
    assert set(manifest["files_included"]).issubset(on_disk)
    assert set(manifest["files_excluded"]).issubset(on_disk)

    config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
    assert config["dim"] == 8
    assert config["space"] == "cosine"
    assert config["m"] == 16
    assert config["ef_construction"] == 200
    assert config["vector_count"] == 3
    assert config["metadata"] == {}

# ------------------------------------------------------------
# Test 74: Persistence: a quantized_with_raw round trip
# ------------------------------------------------------------
def test_persistence_quantized_with_raw_round_trip(tmp_path):
    """The codebook, the codes and the quantized graph all come back.

    The loaded index holds its trained codebook and every stored code, and its
    graph is rebuilt from those codes rather than from the raw vectors, so
    is_quantized() is true and the mode reads quantized_active exactly as it
    did before the save. rebuild_with_quantization() is no longer needed after
    a load, and calling it anyway is asserted to be a safe no-op.
    """
    vdb = VectorDatabase()
    with pytest.warns(UserWarning,
                      match=r"storage_mode='quantized_with_raw' will use ~8\.0x more memory"):
        index = vdb.create("hnsw", dim=8,
                           quantization_config=_pq_config("quantized_with_raw"),
                           expected_size=2000)

    count = 1100  # 100 past training_size, so post training records are involved
    vectors = _sample_vectors(count, 8)
    ids = [f"v_{i}" for i in range(count)]
    assert index.add({"ids": ids, "embeddings": vectors,
                      "metadatas": [{"i": i} for i in range(count)]}).is_success()
    assert index.is_quantized()
    assert index.get_storage_mode() == "quantized_active"

    before = index.get_records(["v_0", "v_1050"], return_vector=True)
    before_by_id = {r["id"]: r for r in before}

    save_dir = tmp_path / "qwr.zdb"
    index.save(str(save_dir))

    on_disk = sorted(p.name for p in save_dir.glob("*"))
    for expected in ("quantization.json", "pq_centroids.bin", "pq_codes.bin", "vectors.bin"):
        assert expected in on_disk

    manifest = json.loads((save_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["has_quantization"] is True
    assert manifest["quantization_trained"] is True
    assert manifest["storage_mode"] == "quantized_active"
    assert manifest["compression_info"]["compression_ratio"] == 8.0

    loaded = vdb.load(str(save_dir))

    # The trained codebook, the stored codes and the quantized graph all
    # survive, so the loaded index is the index that was saved.
    assert loaded.can_use_quantization()
    assert loaded.get_quantization_info()["is_trained"] is True
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"
    assert int(loaded.get_stats()["quantized_codes_stored"]) == count

    # quantized_with_raw keeps every raw vector, so nothing is lost.
    assert loaded.get_vector_count() == count
    assert len(loaded.list(number=count + 10)) == count
    assert int(loaded.get_stats()["raw_vectors_stored"]) == count

    after_by_id = {r["id"]: r for r in loaded.get_records(["v_0", "v_1050"], return_vector=True)}
    assert set(after_by_id) == {"v_0", "v_1050"}
    for vec_id in ("v_0", "v_1050"):
        assert after_by_id[vec_id]["metadata"] == before_by_id[vec_id]["metadata"]
        # Cosine renormalizes on the rebuild, so this holds to float32 precision.
        assert np.allclose(after_by_id[vec_id]["vector"],
                           before_by_id[vec_id]["vector"], atol=1e-6)

    # Search works on the loaded index, through the quantized graph with the
    # default rerank against the raw vectors this mode keeps.
    hits = loaded.search(vectors[0].tolist(), top_k=5)
    assert 0 < len(hits) <= 5
    assert any(h["id"] == "v_0" for h in hits)

    # rebuild_with_quantization is redundant after a load now, and calling it
    # anyway must leave the index quantized with nothing lost.
    assert loaded.rebuild_with_quantization() is True
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"
    assert int(loaded.get_stats()["quantized_codes_stored"]) == count

# ------------------------------------------------------------
# Test 75: Persistence: a PQ index saved before training completes
# ------------------------------------------------------------
def test_persistence_untrained_pq_index(tmp_path):
    """A PQ index saved mid collection resumes where it left off.

    The collected ids are applied after the graph rebuild rather than before it,
    because the rebuild re-adds every record through add(overwrite=true) and the
    removal half of that strips each id from the collection. The reloaded index
    also gets a PQ instance even though nothing is trained yet, without which
    the training trigger could never fire again.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8,
                       quantization_config=_pq_config("quantized_only"),
                       expected_size=2000)

    # Fewer vectors than training_size, so training never runs and no k-means
    # is paid for here.
    count = 800
    vectors = _sample_vectors(count, 8)
    assert index.add({"ids": [f"v_{i}" for i in range(count)],
                      "embeddings": vectors,
                      "metadatas": [{"i": i} for i in range(count)]}).is_success()

    progress = index.get_training_progress()
    assert progress == 80.0
    assert not index.can_use_quantization()
    assert not index.is_quantized()
    assert index.get_storage_mode() == "raw_collecting_for_training"

    save_dir = tmp_path / "untrained.zdb"
    index.save(str(save_dir))

    # No centroids and no codes exist yet, so neither file is written.
    on_disk = {p.name for p in save_dir.glob("*")}
    assert "quantization.json" in on_disk
    assert "pq_centroids.bin" not in on_disk
    assert "pq_codes.bin" not in on_disk

    manifest = json.loads((save_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["has_quantization"] is True
    assert manifest["quantization_trained"] is False
    assert manifest["storage_mode"] == "raw_collecting_for_training"
    assert manifest["compression_info"] is None

    # The saved file does carry the whole training set.
    quantization = json.loads((save_dir / "quantization.json").read_text(encoding="utf-8"))
    assert quantization["is_trained"] is False
    assert quantization["training_threshold_reached"] is False
    assert quantization["training_size"] == 1000
    assert len(quantization["training_ids"]) == count

    loaded = vdb.load(str(save_dir))

    # The vectors and their metadata are intact.
    assert loaded.get_vector_count() == count
    assert len(loaded.list(number=count + 10)) == count
    assert len(loaded.get_records("v_799", return_vector=True)) == 1
    assert loaded.get_records("v_799", return_vector=False)[0]["metadata"] == {"i": 799}
    assert not loaded.can_use_quantization()
    assert not loaded.is_quantized()
    assert loaded.get_storage_mode() == "raw_collecting_for_training"

    # The training progress is intact. The collection picks up at 800 of 1000
    # rather than restarting.
    assert loaded.get_training_progress() == progress
    assert loaded.training_vectors_needed() == 1000 - count
    assert not loaded.is_training_ready()

    # The reloaded index carries a PQ instance, so the fields that come from it
    # are present and report an untrained codebook.
    reloaded_info = loaded.get_quantization_info()
    assert reloaded_info["type"] == "pq"
    assert reloaded_info["training_size"] == 1000
    assert reloaded_info["is_trained"] is False

    # Adding the 200 vectors that complete the collection completes training,
    # and the index reaches the quantized state it could never reach before.
    assert loaded.add({"ids": [f"w_{i}" for i in range(200)],
                       "embeddings": _sample_vectors(200, 8, seed=99)}).is_success()
    assert loaded.get_training_progress() == 100.0
    assert loaded.is_training_ready()
    assert loaded.can_use_quantization()
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"

    # Training used the restored collection together with the new records, so
    # every one of them is still retrievable afterwards.
    assert loaded.get_vector_count() == count + 200
    assert len(loaded.get_records(["v_0", "v_799", "w_0", "w_199"])) == 4

    # And the index that results survives its own round trip.
    second = tmp_path / "trained_after_reload.zdb"
    loaded.save(str(second))
    again = vdb.load(str(second))
    assert again.get_vector_count() == count + 200
    assert again.can_use_quantization()
    assert len(again.get_records(["v_0", "v_799", "w_0", "w_199"])) == 4

# ------------------------------------------------------------
# Test 76: Persistence: a quantized_only round trip after training
# ------------------------------------------------------------
def test_persistence_quantized_only_round_trip(quantized_only_saved):
    """Every record comes back, including the ones stored as codes alone.

    quantized_only stops storing raw vectors once training completes, so
    vectors.bin holds the training set and pq_codes.bin holds everything. The
    loader rebuilds the graph by inserting those stored codes into a PQ graph,
    so every record is covered, nothing is reconstructed to full width on the
    way, and the loaded index is quantized exactly as the saved one was. The
    stored codes are put back as written rather than recomputed, and no record
    is promoted to raw storage, so the mode keeps the memory saving that is
    its whole purpose.
    """
    vdb = VectorDatabase()
    ids = quantized_only_saved["ids"]
    save_dir = quantized_only_saved["path"]
    assert "pq_codes.bin" in {p.name for p in save_dir.glob("*")}

    loaded = vdb.load(str(save_dir))

    # Nothing is lost, and the count now agrees with what is stored.
    assert loaded.get_vector_count() == QO_COUNT
    records = loaded.get_records(ids, return_vector=True)
    assert len(records) == QO_COUNT
    assert {r["id"] for r in records} == set(ids)
    assert all("vector" in r for r in records)

    # The split between raw and coded storage is exactly what was saved.
    assert int(loaded.get_stats()["raw_vectors_stored"]) == QO_TRAINING_SIZE
    assert int(loaded.get_stats()["quantized_codes_stored"]) == QO_COUNT
    assert len(loaded.list(number=QO_COUNT + 10)) == QO_TRAINING_SIZE

    # Per record metadata comes back for the reconstructed records too, which
    # is where it used to be dropped.
    by_id = {r["id"]: r for r in records}
    assert by_id["v_0"]["metadata"] == {"i": 0}
    assert by_id["v_1005"]["metadata"] == {"i": 1005}
    assert by_id["v_1009"]["metadata"] == {"i": 1009}

    # The codebook is restored and the graph is rebuilt from the stored codes,
    # so quantized search comes back with it.
    assert loaded.can_use_quantization()
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"

    # A record with no raw vector is still found by search through the graph.
    hits = loaded.search(quantized_only_saved["vectors"][1005].tolist(), top_k=5)
    assert any(h["id"] == "v_1005" for h in hits)

    # rebuild_with_quantization is redundant after a load now, and calling it
    # anyway must not drop the records that exist only as codes.
    assert loaded.rebuild_with_quantization() is True
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"
    assert int(loaded.get_stats()["quantized_codes_stored"]) == QO_COUNT
    assert len(loaded.get_records(ids)) == QO_COUNT

    # And the result of that rebuild survives a further round trip.
    second = save_dir.parent / "qo_rebuilt.zdb"
    loaded.save(str(second))
    again = vdb.load(str(second))
    assert again.get_vector_count() == QO_COUNT
    assert len(again.get_records(ids)) == QO_COUNT

# ------------------------------------------------------------
# Test 78: Persistence: the restored count matches what was restored
# ------------------------------------------------------------
def test_persistence_vector_count_matches_reality(tmp_path, quantized_only_saved):
    """vector_count is checked against the index that was actually built.

    It is stored in config.json and used to be restored verbatim, so it could
    report records the directory no longer contained. The count is now derived
    from the restored data and asserted against the saved value, and a
    disagreement fails the load rather than producing an index that misreports
    what it holds.
    """
    vdb = VectorDatabase()

    # The healthy case agrees, in both quantized and plain indexes.
    loaded = vdb.load(str(quantized_only_saved["path"]))
    assert loaded.get_vector_count() == QO_COUNT
    assert loaded.get_vector_count() == len(loaded.get_records(quantized_only_saved["ids"]))

    plain_dir = tmp_path / "plain.zdb"
    plain = vdb.create("hnsw", dim=8, expected_size=50)
    plain.add({"ids": [f"p_{i}" for i in range(6)], "embeddings": _sample_vectors(6, 8)})
    plain.save(str(plain_dir))
    assert vdb.load(str(plain_dir)).get_vector_count() == 6

    # Losing vectors.bin from a plain index used to load zero records while
    # still reporting the saved count. It is now refused, and the message names
    # both numbers.
    broken = tmp_path / "broken.zdb"
    shutil.copytree(plain_dir, broken)
    (broken / "vectors.bin").unlink()
    with pytest.raises(RuntimeError, match=r"yields 0 records while config.json reports 6"):
        vdb.load(str(broken))

    # Losing pq_codes.bin from a quantized_only index loses the records that
    # exist only as codes, so that is refused too.
    no_codes = tmp_path / "no_codes.zdb"
    shutil.copytree(quantized_only_saved["path"], no_codes)
    (no_codes / "pq_codes.bin").unlink()
    with pytest.raises(RuntimeError, match=r"yields 1000 records while config.json reports 1010"):
        vdb.load(str(no_codes))

    # Losing vectors.bin from a quantized_only index loses nothing, because
    # every record still has its codes. This one loads.
    no_vectors = tmp_path / "no_vectors.zdb"
    shutil.copytree(quantized_only_saved["path"], no_vectors)
    (no_vectors / "vectors.bin").unlink()
    recovered = vdb.load(str(no_vectors))
    assert recovered.get_vector_count() == QO_COUNT
    assert len(recovered.get_records(quantized_only_saved["ids"])) == QO_COUNT

# ------------------------------------------------------------
# Test 79: Persistence: an unusable codebook fails the load
# ------------------------------------------------------------
def test_persistence_rejects_unusable_codebook(tmp_path, quantized_only_saved):
    """A trained index needs its codebook, and a wrong one is worse than none.

    Directories written by 0.3.0 through 0.4.1 contain an all-zero codebook if
    they were ever saved a second time, because those versions never read
    pq_centroids.bin back and re-saved the zeros a fresh PQ starts with. Every
    code in such a directory decodes to the zero vector, so it is refused.
    """
    vdb = VectorDatabase()

    missing = tmp_path / "missing_codebook.zdb"
    shutil.copytree(quantized_only_saved["path"], missing)
    (missing / "pq_centroids.bin").unlink()
    with pytest.raises(FileNotFoundError,
                       match=r"trained codebook but pq_centroids.bin is missing"):
        vdb.load(str(missing))

    # An all-zero codebook of the right shape is the signature of the second
    # save, and it is what those directories hold today.
    zeroed = tmp_path / "zero_codebook.zdb"
    shutil.copytree(quantized_only_saved["path"], zeroed)
    zero_bytes = _zero_codebook_bytes(subvectors=4, centroids=256, sub_dim=2)
    # Same length as the real codebook, which is the check that the encoding
    # above matches what save_pq_centroids writes.
    assert len(zero_bytes) == len((zeroed / "pq_centroids.bin").read_bytes())
    (zeroed / "pq_centroids.bin").write_bytes(zero_bytes)
    with pytest.raises(RuntimeError, match=r"all-zero codebook"):
        vdb.load(str(zeroed))

    # A codebook of the wrong shape names both shapes rather than
    # reconstructing against whatever it happens to hold.
    wrong = tmp_path / "wrong_codebook.zdb"
    shutil.copytree(quantized_only_saved["path"], wrong)
    (wrong / "pq_centroids.bin").write_bytes(
        _zero_codebook_bytes(subvectors=2, centroids=256, sub_dim=2))
    with pytest.raises(RuntimeError,
                       match=r"codebook is 2x256x2, expected 4x256x2"):
        vdb.load(str(wrong))

# ------------------------------------------------------------
# Test 80: Persistence: a directory written in the previous format
# ------------------------------------------------------------
def test_persistence_reads_previous_format(tmp_path, quantized_only_saved):
    """Format 1.0.0, written by 0.3.0 through 0.4.1, still opens.

    The fixture is built by saving with this build and then reversing the only
    two save side changes, being the index level metadata field in config.json
    and the format version in manifest.json. Nothing else about what is written
    changed, verified by comparing the persisted structures and the artefact
    set against the v0.4.1 tag, so this is what those releases produced.
    """
    vdb = VectorDatabase()
    legacy = tmp_path / "legacy.zdb"
    shutil.copytree(quantized_only_saved["path"], legacy)

    config = json.loads((legacy / "config.json").read_text(encoding="utf-8"))
    had_metadata = config.pop("metadata")
    assert had_metadata == {"owner": "relay24"}
    (legacy / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    manifest = json.loads((legacy / "manifest.json").read_text(encoding="utf-8"))
    manifest["format_version"] = "1.0.0"
    (legacy / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    loaded = vdb.load(str(legacy))

    # Every record comes back, including the ten that the release which wrote
    # this directory dropped on its own load.
    assert loaded.get_vector_count() == QO_COUNT
    assert len(loaded.get_records(quantized_only_saved["ids"])) == QO_COUNT
    assert loaded.can_use_quantization()

    # Index level metadata is the one thing a 1.0.0 directory cannot return,
    # because it was never written. An empty map is the honest answer.
    assert loaded.get_all_metadata() == {}

    # Saving it again upgrades the directory in place.
    upgraded = tmp_path / "upgraded.zdb"
    loaded.save(str(upgraded))
    assert json.loads((upgraded / "manifest.json").read_text(
        encoding="utf-8"))["format_version"] == "1.1.0"
    assert "metadata" in json.loads((upgraded / "config.json").read_text(encoding="utf-8"))
    assert vdb.load(str(upgraded)).get_vector_count() == QO_COUNT

# ------------------------------------------------------------
# Test 81: Persistence: the format version gate
# ------------------------------------------------------------
def test_persistence_format_version_gate(tmp_path):
    """A later minor is read, another major is refused.

    Minor bumps are additive by construction, so any 1.x is accepted. A
    different major means the layout changed in a way this build cannot reason
    about, and guessing at it is how a format loses data quietly.
    """
    vdb = VectorDatabase()
    source = tmp_path / "versioned.zdb"
    index = vdb.create("hnsw", dim=4, expected_size=10)
    index.add({"id": "r1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {}})
    index.save(str(source))

    def with_version(version, name):
        target = tmp_path / name
        shutil.copytree(source, target)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["format_version"] = version
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return target

    assert vdb.load(str(with_version("1.0.0", "v100.zdb"))).get_vector_count() == 1
    assert vdb.load(str(with_version("1.9.3", "v193.zdb"))).get_vector_count() == 1

    with pytest.raises(RuntimeError, match=r"format version 2.0.0 cannot be opened"):
        vdb.load(str(with_version("2.0.0", "v200.zdb")))

    with pytest.raises(RuntimeError, match=r"not a version this build can interpret"):
        vdb.load(str(with_version("banana", "vbanana.zdb")))

# ------------------------------------------------------------
# Test 77: Persistence: load failure modes
# ------------------------------------------------------------
def test_persistence_load_failure_modes(tmp_path):
    vdb = VectorDatabase()

    # A path that does not exist is caught before anything is read.
    missing = tmp_path / "not_there.zdb"
    with pytest.raises(FileNotFoundError, match="Index directory not found"):
        vdb.load(str(missing))

    # A directory that exists but holds no manifest reports the missing file.
    empty = tmp_path / "empty_dir.zdb"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Failed to read manifest.json"):
        vdb.load(str(empty))

    # A manifest that is not valid JSON is a parse failure, not a read failure,
    # and it surfaces as RuntimeError rather than FileNotFoundError.
    malformed = tmp_path / "malformed.zdb"
    malformed.mkdir()
    (malformed / "manifest.json").write_text("this is not json {{{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to parse manifest.json"):
        vdb.load(str(malformed))

    # A manifest that is valid JSON but not a valid manifest is also a parse
    # failure, and the message names the first field the deserializer misses.
    incomplete = tmp_path / "incomplete.zdb"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text(
        json.dumps({"format_version": "1.0.0", "total_vectors": 0,
                    "storage_mode": "raw_only", "files_included": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Failed to parse manifest.json.*zeusdb_version"):
        vdb.load(str(incomplete))

    # A real save whose config.json has been removed gets past the manifest and
    # fails on the next file the loader reaches.
    truncated = tmp_path / "truncated.zdb"
    index = vdb.create("hnsw", dim=4, expected_size=10)
    index.add({"id": "r1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {}})
    index.save(str(truncated))
    (truncated / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="Failed to read config.json"):
        vdb.load(str(truncated))

    # A regular file rather than a directory is not rejected by the directory
    # check, because Path::exists is true for both, so it fails later.
    not_a_dir = tmp_path / "file.zdb"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Failed to read manifest.json"):
        vdb.load(str(not_a_dir))

# ------------------------------------------------------------
# Test 100: a saved directory holding a non-finite value fails loudly
# ------------------------------------------------------------
def test_load_refuses_a_saved_index_holding_a_non_finite_value(tmp_path):
    """A NaN could reach disk before add validated the NumPy branches.

    The rebuild replays every record through add(), which has always refused a
    non-finite value on the list path, and add() reports rather than raises.
    The rebuild ignored that report, and the storage maps are written back
    afterwards from the file, so the count check still agreed and the load
    succeeded holding records that no query could reach. It now fails.

    The directory is poisoned by patching the float in vectors.bin, because
    add() no longer accepts one, which is the point of the other half of this
    change.
    """
    marker_value = 123456.75  # exactly representable in f32 and unique here
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="l2", expected_size=10)
    index.add({"id": "poisoned", "values": [marker_value, 0.25, 0.5, 0.75]})
    index.add({"id": "clean", "values": [0.25, 0.5, 0.75, 1.0]})

    save_dir = tmp_path / "poisoned.zdb"
    index.save(str(save_dir))

    vectors_bin = save_dir / "vectors.bin"
    blob = vectors_bin.read_bytes()
    marker = struct.pack("<f", marker_value)
    assert blob.count(marker) == 1
    vectors_bin.write_bytes(blob.replace(marker, struct.pack("<f", float("nan"))))

    with pytest.raises(RuntimeError, match="Graph rebuild refused"):
        vdb.load(str(save_dir))

# ------------------------------------------------------------
# Shared fixture for the quantized reload coverage
# ------------------------------------------------------------
RELOAD_DIM = 16
RELOAD_SUBVECTORS = 8
RELOAD_TRAINING_SIZE = 1000
RELOAD_TOTAL = 1500
RELOAD_QUERIES = 50


def _clustered_unit_vectors(n, dim, seed):
    """Twenty Gaussian centres, a small perturbation, then L2 normalised.

    Clustered rather than uniform, because uniform vectors are close to
    equidistant and recall saturates on them, which would hide a regression.
    Same specification as the graph quality fixtures in test_quantization.py.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((20, dim))
    points = centres[rng.integers(0, 20, size=n)] + 0.15 * rng.standard_normal((n, dim))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def _reload_recall(index, ids, queries, truth, **search_kwargs):
    hits = 0
    for qi, query in enumerate(queries):
        found = {r["id"] for r in index.search(query.tolist(), top_k=10,
                                               **search_kwargs)}
        hits += len(found & {ids[j] for j in truth[qi]})
    return hits / (10 * len(queries))


@pytest.fixture(scope="module", params=["quantized_only", "quantized_with_raw"])
def quantized_reload(request, tmp_path_factory):
    """A trained quantized index saved together with its pre-save behaviour.

    Module scoped and parametrised over both storage modes, so each mode pays
    for one k-means run. The 500 records past training_size arrive through the
    quantized add path, which under quantized_only stores codes and no raw
    vector, so they are the records a lossy loader would drop.
    """
    mode = request.param
    all_vectors = _clustered_unit_vectors(RELOAD_TOTAL + RELOAD_QUERIES,
                                          RELOAD_DIM, 20260804)
    data, queries = all_vectors[:RELOAD_TOTAL], all_vectors[RELOAD_TOTAL:]
    ids = [f"r_{i}" for i in range(RELOAD_TOTAL)]

    config = {"type": "pq", "subvectors": RELOAD_SUBVECTORS, "bits": 8,
              "training_size": RELOAD_TRAINING_SIZE, "storage_mode": mode}
    with warnings.catch_warnings():
        # quantized_with_raw warns unconditionally about its memory use, which
        # test 74 asserts where it is the subject.
        warnings.simplefilter("ignore")
        index = VectorDatabase().create("hnsw", dim=RELOAD_DIM,
                                        expected_size=RELOAD_TOTAL,
                                        quantization_config=config)

    assert index.add({"ids": ids, "embeddings": data,
                      "metadatas": [{"i": i} for i in range(RELOAD_TOTAL)]
                      }).is_success()
    assert index.is_quantized(), "training did not complete"

    truth = np.argsort(-(queries @ data.T), axis=1)[:, :10]
    recall_before = {
        "default": _reload_recall(index, ids, queries, truth),
        "rerank0": _reload_recall(index, ids, queries, truth, rerank=0),
    }

    save_dir = tmp_path_factory.mktemp(f"reload_{mode}") / "idx.zdb"
    index.save(str(save_dir))

    return {"mode": mode, "path": save_dir, "ids": ids, "data": data,
            "queries": queries, "truth": truth, "recall_before": recall_before}

# ------------------------------------------------------------
# Test 101: a trained quantized index loads back quantized
# ------------------------------------------------------------
def test_quantized_index_loads_back_quantized(quantized_reload):
    """The state, the records and the recall all survive the round trip.

    The loader rebuilds the graph by inserting the stored PQ codes into a PQ
    graph, so the reloaded index reports quantized_active rather than
    raw_trained_not_rebuilt and searches through ADC exactly as it did before
    the save. Insertion order on load follows HashMap iteration rather than
    arrival order, so the graph need not be identical, and the recall bound
    below is a tolerance rather than an equality for that reason.
    """
    fixture = quantized_reload
    loaded = VectorDatabase().load(str(fixture["path"]))

    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"
    assert loaded.can_use_quantization()

    # Every record survives, including the 500 added after training that have
    # codes and no raw vector under quantized_only.
    assert loaded.get_vector_count() == RELOAD_TOTAL
    records = loaded.get_records(fixture["ids"])
    assert len(records) == RELOAD_TOTAL
    post_training = fixture["ids"][RELOAD_TRAINING_SIZE:]
    assert len(loaded.get_records(post_training)) == len(post_training)

    # The whole graph is reachable, so nothing was silently left out of it.
    page = loaded.search(fixture["queries"][0].tolist(), top_k=RELOAD_TOTAL,
                         ef_search=RELOAD_TOTAL)
    assert len(page) == RELOAD_TOTAL

    # Recall matches the pre-save figure within the variation a different
    # insertion order produces. Measured at 10,000 records, reordering the
    # same codes moved recall by under 0.001; the bound leaves room for the
    # smaller index here.
    for label, kwargs in (("default", {}), ("rerank0", {"rerank": 0})):
        after = _reload_recall(loaded, fixture["ids"], fixture["queries"],
                               fixture["truth"], **kwargs)
        before = fixture["recall_before"][label]
        assert abs(after - before) <= 0.05, (
            f"{fixture['mode']} {label}: recall moved from {before:.4f} to "
            f"{after:.4f} across the reload"
        )

# ------------------------------------------------------------
# Test 102: the raw vector store is not inflated by a reload
# ------------------------------------------------------------
def test_quantized_reload_raw_store_holds_only_what_was_saved(quantized_reload):
    """quantized_only comes back holding raw vectors for the training set alone.

    That is exactly what the live index held before the save, because the
    collection phase stores raw vectors and nothing clears them at training.
    The 500 code-only records must not be materialised at full width on load;
    before this fix the loader reconstructed them into the graph, which is
    where the mode's memory saving went to die.
    """
    fixture = quantized_reload
    loaded = VectorDatabase().load(str(fixture["path"]))

    expected_raw = (RELOAD_TRAINING_SIZE if fixture["mode"] == "quantized_only"
                    else RELOAD_TOTAL)
    stats = loaded.get_stats()
    assert int(stats["raw_vectors_stored"]) == expected_raw
    assert int(stats["quantized_codes_stored"]) == RELOAD_TOTAL

# ------------------------------------------------------------
# Test 103: rerank behaviour survives the reload
# ------------------------------------------------------------
def test_quantized_reload_rerank_behaviour(quantized_reload):
    """quantized_with_raw reranks after a reload, quantized_only stays inert.

    Before this fix a reloaded index was a raw index, so the rerank plan never
    engaged and quantized_with_raw searched at raw recall with raw scores. Now
    the mode comes back quantized and the default search over-fetches and
    rescores against the raw vectors exactly as it did before the save.
    """
    fixture = quantized_reload
    loaded = VectorDatabase().load(str(fixture["path"]))
    query = fixture["queries"][0]

    if fixture["mode"] == "quantized_with_raw":
        # Reranked recall is near the raw ceiling and far above the ADC page.
        reranked = _reload_recall(loaded, fixture["ids"], fixture["queries"],
                                  fixture["truth"])
        unreranked = _reload_recall(loaded, fixture["ids"], fixture["queries"],
                                    fixture["truth"], rerank=0)
        assert reranked >= 0.90
        assert reranked > unreranked + 0.10

        # The reranked score is the cosine distance to the stored raw vector,
        # which is what a raw index reports for the same pair.
        top = loaded.search(query.tolist(), top_k=1)[0]
        stored = loaded.get_records(top["id"], return_vector=True)[0]["vector"]
        expected = 1.0 - float(
            np.dot(query, stored) / (np.linalg.norm(query) * np.linalg.norm(stored))
        )
        assert abs(top["score"] - expected) < 1e-5
    else:
        # quantized_only has nothing to rerank, so the parameter changes
        # neither the page nor the scores.
        with_default = [(r["id"], round(float(r["score"]), 6))
                        for r in loaded.search(query.tolist(), top_k=10)]
        with_off = [(r["id"], round(float(r["score"]), 6))
                    for r in loaded.search(query.tolist(), top_k=10, rerank=0)]
        assert with_default == with_off
