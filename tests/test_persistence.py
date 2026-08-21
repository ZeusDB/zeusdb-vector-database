"""Saving an index to disk and loading it back."""

import json
import os
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
    # Both hold once training has run: the quantizer is trained and the graph
    # has been rebuilt onto the codes. This was a disjunction, which pinned
    # neither state.
    assert index.can_use_quantization()
    assert index.is_quantized()

    save_dir = tmp_path / "pq_index.zdb"
    index.save(str(save_dir))
    assert save_dir.exists() and save_dir.is_dir()

    loaded = vdb.load(str(save_dir))
    # A quantized index loads back quantized rather than in a state that merely
    # could be. Same disjunction, same reason for collapsing it.
    assert loaded.can_use_quantization()
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"

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
    # it is used. quantized_only warns that it cannot repay its fixed cost at
    # these sizes, which is true and is asserted in test_quantization.py.
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
    it first. Every record is stored as codes alone, since the mode releases
    the training records' raw vectors at training completion, so the directory
    carries no vectors.bin and the codes are the only copy of every record.
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
    cosine path and returns the vector untouched on l1 and l2. What the load
    path returns is the stored vector as it was written, bit for bit, in every
    space, because vectors.bin is restored verbatim.
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
            # And the stored form comes back exactly, because vectors.bin is
            # restored as written and nothing normalizes it a second time.
            assert np.array_equal(stored_before, stored_after)
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
    assert record["vector"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
                              "indexed_fields", "m", "metadata", "space", "vector_count"]
    # The declared filterable fields ride here too, for the same reason. The
    # columns themselves are derived from metadata.json on load, so nothing else
    # in the directory records which fields a user chose. This index declared
    # none, which is what an empty list means.
    assert config["indexed_fields"] == []

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
    # One graph file rather than the two the vendored writer used to leave. The
    # split existed so the points could be memory mapped, which this build never
    # asks for, and it meant the topology and the points could disagree.
    assert on_disk == [
        "config.json",
        "hnsw_index.zdbgraph",
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
    # subset of what is on disk. The graph file is in it, because the loader
    # restores the saved graph and cannot do that without the points.
    assert manifest["files_included"] == [
        "config.json", "mappings.bin", "metadata.json", "vectors.bin",
        "hnsw_index.zdbgraph",
    ]
    assert manifest["files_excluded"] == []
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
    # The warning used to quote the compression ratio as a memory multiplier,
    # and this asserted the 8.0x it printed. The ratio between the two storage
    # modes is a different quantity that depends on the record count, so the
    # warning no longer quotes one. It now opens by naming what the mode is for.
    with pytest.warns(UserWarning,
                      match=r"storage_mode='quantized_with_raw' is the accuracy "
                            r"mode rather than the memory mode"):
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
    # dim 8 at 4 subvectors, so a code is 4 bytes against 32 for the vector.
    # This mode keeps a raw vector for every coded record, so the manifest read
    # 8.0 before the numerator was changed to count the coded records too, and
    # it reads 8.0 after. Under quantized_only it did not; see test 105.
    assert manifest["compression_info"]["compression_ratio"] == 8.0
    assert manifest["compression_info"]["original_size_mb"] == pytest.approx(
        count * 8 * 4 / (1024 * 1024))

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
        # The reconstruction a loaded index returns is the reconstruction the
        # live index returned, since the codes are restored as written.
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

    The collected ids are applied after the graph step rather than before it,
    because the fallback rebuild re-adds every record through add(overwrite=true)
    and the removal half of that strips each id from the collection. The reloaded
    index also gets a PQ instance even though nothing is trained yet, without
    which the training trigger could never fire again.
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

    A trained quantized_only index holds no raw vectors, so its directory has
    pq_codes.bin and no vectors.bin at all. The loader restores the PQ graph the
    save dumped, so every record is covered, nothing is reconstructed to full
    width on the way, and the loaded index is quantized exactly as the saved one
    was. The stored codes are put back as written rather than recomputed, and no
    record is promoted to raw storage,
    so the mode keeps the memory saving that is its whole purpose.
    """
    vdb = VectorDatabase()
    ids = quantized_only_saved["ids"]
    save_dir = quantized_only_saved["path"]
    on_disk = {p.name for p in save_dir.glob("*")}
    assert "pq_codes.bin" in on_disk
    assert "vectors.bin" not in on_disk

    loaded = vdb.load(str(save_dir))

    # Nothing is lost, and the count now agrees with what is stored.
    assert loaded.get_vector_count() == QO_COUNT
    records = loaded.get_records(ids, return_vector=True)
    assert len(records) == QO_COUNT
    assert {r["id"] for r in records} == set(ids)
    assert all("vector" in r for r in records)

    # The split between raw and coded storage is exactly what was saved, which
    # for this mode is codes for everything and no raw vector anywhere. Only
    # get_stats reports the split. list() enumerates the id map, so it returns
    # every record regardless of which store holds its vector, and a reloaded
    # index is no different from a live one here.
    assert int(loaded.get_stats()["raw_vectors_stored"]) == 0
    assert int(loaded.get_stats()["quantized_codes_stored"]) == QO_COUNT
    assert len(loaded.list(number=QO_COUNT + 10)) == QO_COUNT
    assert {rid for rid, _ in loaded.list(number=QO_COUNT + 10)} == set(ids)
    assert all(loaded.contains(record_id) for record_id in ids)

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

    # And that state survives a further round trip.
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

    # A file the manifest names and the directory lacks is refused before any
    # artefact is read, so losing vectors.bin from a plain index is caught by
    # the completeness check rather than by the count. Both files below are
    # named under files_included, which is what makes the earlier check the one
    # that fires.
    broken = tmp_path / "broken.zdb"
    shutil.copytree(plain_dir, broken)
    (broken / "vectors.bin").unlink()
    with pytest.raises(FileNotFoundError,
                       match=r"manifest.json names vectors.bin under files_included"):
        vdb.load(str(broken))

    # Losing pq_codes.bin from a quantized_only index loses every record,
    # because the codes are the only copy the mode keeps. The directory carries
    # no vectors.bin to fall back on: a trained index in this mode saved none,
    # having released the training records' raw vectors when training
    # completed.
    assert not (quantized_only_saved["path"] / "vectors.bin").exists()
    no_codes = tmp_path / "no_codes.zdb"
    shutil.copytree(quantized_only_saved["path"], no_codes)
    (no_codes / "pq_codes.bin").unlink()
    with pytest.raises(FileNotFoundError,
                       match=r"manifest.json names pq_codes.bin under files_included"):
        vdb.load(str(no_codes))

    # The count check itself still stands behind that one. A directory whose
    # files are all present and all parse, but whose config.json reports a
    # count the records cannot produce, is refused with both numbers named.
    miscounted = tmp_path / "miscounted.zdb"
    shutil.copytree(plain_dir, miscounted)
    config = json.loads((miscounted / "config.json").read_text(encoding="utf-8"))
    config["vector_count"] = 9
    (miscounted / "config.json").write_text(json.dumps(config, indent=2),
                                            encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"yields 6 records while config.json reports 9"):
        vdb.load(str(miscounted))

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

    # Deleting it is caught by the completeness check, because the manifest
    # names it. The message says what the codebook is for, so a reader knows
    # the file cannot be rebuilt from the others.
    missing = tmp_path / "missing_codebook.zdb"
    shutil.copytree(quantized_only_saved["path"], missing)
    (missing / "pq_centroids.bin").unlink()
    with pytest.raises(FileNotFoundError,
                       match=r"pq_centroids.bin holds the trained PQ codebook"):
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

    The fixture is built by saving with this build and then reversing two of
    the save side changes since, being the index level metadata field in
    config.json and the format version in manifest.json. The third difference
    is that those releases also wrote the training records to vectors.bin for
    a trained quantized_only index; that file's handling is covered by
    test_persistence_drops_raw_vectors_from_an_old_quantized_only_directory,
    which loads a directory that carries one.
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
# Test 106: Persistence: an old quantized_only directory sheds its raw vectors
# ------------------------------------------------------------
def test_persistence_drops_raw_vectors_from_an_old_quantized_only_directory(tmp_path):
    """A directory that carries vectors.bin loads, and the raw vectors go.

    Releases up to 0.4.1 kept the training records at full width under
    quantized_only and wrote them to vectors.bin, so their directories carry
    raw vectors this build no longer keeps. The loader drops every raw vector
    whose record has stored codes, which in an intact directory is all of
    them, so an old directory sheds its training records on load exactly as a
    live index sheds them at training. Nothing is lost: the codes are the
    record, and the count check still passes because the union of raws and
    codes is unchanged.

    The fixture is a quantized_with_raw save, whose vectors.bin holds a raw
    vector for every record alongside full codes, relabelled quantized_only in
    quantization.json. That is a superset of what 0.4.1 wrote for
    quantized_only, where vectors.bin held the training records alone.
    """
    vdb = VectorDatabase()
    count = 1100
    with pytest.warns(UserWarning, match="keeps a raw vector for every record"):
        index = vdb.create("hnsw", dim=8,
                           quantization_config=_pq_config("quantized_with_raw"),
                           expected_size=2000)
    vectors = _sample_vectors(count, 8)
    ids = [f"v_{i}" for i in range(count)]
    assert index.add({"ids": ids, "embeddings": vectors,
                      "metadatas": [{"i": i} for i in range(count)]}).is_success()
    assert index.is_quantized()
    # The live quantized_with_raw index serves these exactly, from raw storage.
    live_raw = {
        r["id"]: r["vector"]
        for r in index.get_records(ids[:5] + ids[-5:], return_vector=True)
    }

    old_style = tmp_path / "old_quantized_only.zdb"
    index.save(str(old_style))
    assert (old_style / "vectors.bin").exists()

    quantization = json.loads(
        (old_style / "quantization.json").read_text(encoding="utf-8"))
    assert quantization["storage_mode"] == "quantized_with_raw"
    quantization["storage_mode"] = "quantized_only"
    (old_style / "quantization.json").write_text(
        json.dumps(quantization, indent=2), encoding="utf-8")

    loaded = vdb.load(str(old_style))

    # Every record is present, served from its codes, and the raw store is
    # empty despite vectors.bin holding a copy of every record.
    assert loaded.get_vector_count() == count
    stats = loaded.get_stats()
    assert int(stats["raw_vectors_stored"]) == 0
    assert int(stats["quantized_codes_stored"]) == count
    assert stats["storage_mode"] == "quantized_only"
    records = loaded.get_records(ids, return_vector=True)
    assert len(records) == count
    assert all("vector" in r for r in records)

    # What comes back is the reconstruction from the stored codes rather than
    # the raw copy vectors.bin carried: close to the vector the raw store
    # held, not equal to it.
    by_id = {r["id"]: r["vector"] for r in loaded.get_records(
        ids[:5] + ids[-5:], return_vector=True)}
    for record_id, raw_vector in live_raw.items():
        reconstructed = np.asarray(by_id[record_id], dtype=np.float64)
        raw = np.asarray(raw_vector, dtype=np.float64)
        assert not np.allclose(reconstructed, raw, atol=1e-6)
        cosine = float(reconstructed @ raw
                       / (np.linalg.norm(reconstructed) * np.linalg.norm(raw)))
        assert cosine > 0.90

    # A save after the load writes the new shape, with no vectors.bin.
    resaved = tmp_path / "resaved.zdb"
    loaded.save(str(resaved))
    assert not (resaved / "vectors.bin").exists()
    again = vdb.load(str(resaved))
    assert again.get_vector_count() == count
    assert len(again.get_records(ids)) == count

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

    # A real save whose config.json has been removed is refused by the
    # completeness check, which runs on the manifest before any artefact is
    # read. A file the manifest names and the directory lacks and a file that
    # is there and does not parse are different failures with different
    # messages.
    truncated = tmp_path / "truncated.zdb"
    index = vdb.create("hnsw", dim=4, expected_size=10)
    index.add({"id": "r1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {}})
    index.save(str(truncated))
    gone = tmp_path / "config_gone.zdb"
    shutil.copytree(truncated, gone)
    (gone / "config.json").unlink()
    with pytest.raises(FileNotFoundError,
                       match=r"manifest.json names config.json under files_included"):
        vdb.load(str(gone))

    # Present and unreadable is the other half, and it still names the parse.
    (truncated / "config.json").write_text("not json at all {", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to parse config.json"):
        vdb.load(str(truncated))

    # A regular file rather than a directory is not rejected by the directory
    # check, because Path::exists is true for both, so it fails later.
    not_a_dir = tmp_path / "file.zdb"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Failed to read manifest.json"):
        vdb.load(str(not_a_dir))

# ------------------------------------------------------------
# Test 120: config.json is validated at the load
# ------------------------------------------------------------
def test_load_validates_the_declaration_in_config_json(tmp_path):
    """The five values create() validates are validated on the way back in too.

    Parsing proved config.json was JSON of the right shape and stopped there.
    dim, m, ef_construction, expected_size and space went from the file into
    new_empty, which validates nothing, and on into Backend::sized, which
    clamps dim up to 1, m into 2 to 256 and expected_size up to 1 in silence.
    A directory naming m 0 came back as an index at m 2, and one naming an
    unknown space came back scoring cosine whatever it was saved with. A zero
    dim was refused, but by a later check comparing a record against the
    declared width, so the message named a record rather than the config.

    Every message is the one create() raises for the same value, with the path
    of the file in front of it.
    """
    vdb = VectorDatabase()
    good = tmp_path / "good.zdb"
    index = vdb.create("hnsw", dim=4, space="l2", m=8, ef_construction=64,
                       expected_size=100)
    for i in range(5):
        index.add({"id": f"r{i}", "values": [float(i), 0.5, 0.25, 0.125],
                   "metadata": {}})
    index.save(str(good))

    # The directory as written still opens, and opens as what it was saved as.
    reopened = vdb.load(str(good))
    stats = reopened.get_stats()
    assert (int(stats["dimension"]), stats["space"], int(stats["m"]),
            int(stats["ef_construction"]), int(stats["expected_size"])) == (
        4, "l2", 8, 64, 100)
    assert reopened.get_vector_count() == 5

    def corrupted(name, field, value):
        target = tmp_path / name
        shutil.copytree(good, target)
        config = json.loads((target / "config.json").read_text(encoding="utf-8"))
        config[field] = value
        (target / "config.json").write_text(json.dumps(config, indent=2),
                                            encoding="utf-8")
        return str(target)

    # A zero in any of the three that must be positive.
    with pytest.raises(ValueError, match=r"config\.json: dim must be positive, got 0"):
        vdb.load(corrupted("dim0.zdb", "dim", 0))
    with pytest.raises(ValueError,
                       match=r"config\.json: ef_construction must be positive, got 0"):
        vdb.load(corrupted("ef0.zdb", "ef_construction", 0))
    with pytest.raises(ValueError,
                       match=r"config\.json: expected_size must be positive, got 0"):
        vdb.load(corrupted("es0.zdb", "expected_size", 0))

    # m outside the range the graph accepts, at both ends. Backend::sized
    # clamped both without saying so.
    with pytest.raises(ValueError, match=r"config\.json: m must be at least 2, got 1"):
        vdb.load(corrupted("m1.zdb", "m", 1))
    with pytest.raises(ValueError,
                       match=r"config\.json: m must be less than or equal to 256, got 300"):
        vdb.load(corrupted("m300.zdb", "m", 300))

    # An expected_size above the cap, which create() refuses because the layer
    # reservation is not fallible.
    with pytest.raises(ValueError,
                       match=r"config\.json: expected_size must be at most 100000000"):
        vdb.load(corrupted("esbig.zdb", "expected_size", 200_000_000))

    # A space this build does not know. new_raw fell back to cosine and logged
    # it, so an l2 index came back answering cosine distances.
    with pytest.raises(RuntimeError,
                       match=r"config\.json: Unsupported space: 'euclidean'"):
        vdb.load(corrupted("space.zdb", "space", "euclidean"))

    # id_counter above what a node index can name. This one is not a behaviour
    # that came back wrong: the internal id is the index into the graph's
    # id-to-node array, so a config declaring 2^40 loaded and then **aborted the
    # process** on the next add, asking the allocator for 4.4 TB. An allocation
    # failure does not unwind, so nothing catches it and the interpreter dies
    # with no traceback.
    with pytest.raises(ValueError, match=r"config\.json: id_counter is 1099511627776"):
        vdb.load(corrupted("idbig.zdb", "id_counter", 1 << 40))

    # And the add that used to abort now runs, on a directory whose id_counter
    # is merely larger than the ids it holds rather than impossible.
    grown = vdb.load(corrupted("idgrown.zdb", "id_counter", 4_000_000))
    grown.add({"id": "extra", "values": [1.0, 0.5, 0.25, 0.125], "metadata": {}})
    assert grown.get_vector_count() == 6

    # A valid value that merely differs still opens, so the check refuses bad
    # declarations rather than unfamiliar ones.
    reopened = vdb.load(corrupted("l1.zdb", "space", "l1"))
    assert reopened.get_stats()["space"] == "l1"
    assert reopened.get_vector_count() == 5

# ------------------------------------------------------------
# Test 100: a saved directory holding a non-finite value fails loudly
# ------------------------------------------------------------
def test_load_refuses_a_saved_index_holding_a_non_finite_value(tmp_path):
    """A NaN could reach disk before add validated the NumPy branches.

    The guard is in the loader rather than in the graph rebuild. The rebuild
    used to catch this by replaying every record through add(), which refuses a
    non-finite value, but the loader now restores the saved graph and the
    rebuild does not run. vectors.bin is checked as it is read, so the refusal
    holds whichever graph path the directory takes.

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

    with pytest.raises(RuntimeError, match="holds a NaN or an infinity"):
        vdb.load(str(save_dir))

    # The same refusal when the graph is rebuilt rather than restored, so the
    # guard does not depend on which path the load takes.
    os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"] = "1"
    try:
        with pytest.raises(RuntimeError, match="holds a NaN or an infinity"):
            vdb.load(str(save_dir))
    finally:
        del os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"]

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

    pages_before = _pages(index, queries)

    save_dir = tmp_path_factory.mktemp(f"reload_{mode}") / "idx.zdb"
    index.save(str(save_dir))

    return {"mode": mode, "path": save_dir, "ids": ids, "data": data,
            "queries": queries, "truth": truth, "recall_before": recall_before,
            "pages_before": pages_before}


def _pages(index, queries, top_k=10, **search_kwargs):
    """Result pages as ids paired with the exact float32 bits of each score."""
    return [
        [(r["id"], np.float32(r["score"]).tobytes())
         for r in index.search(q.tolist(), top_k=top_k, **search_kwargs)]
        for q in queries
    ]


def _page_differences(left, right):
    """(pages whose ids differ, pages whose scores differ) between two runs."""
    assert len(left) == len(right)
    ids_differ = sum(1 for a, b in zip(left, right)
                     if [x[0] for x in a] != [x[0] for x in b])
    scores_differ = sum(1 for a, b in zip(left, right)
                        if [x[0] for x in a] == [x[0] for x in b] and a != b)
    return ids_differ, scores_differ

# ------------------------------------------------------------
# Test 101: a trained quantized index loads back quantized
# ------------------------------------------------------------
def test_quantized_index_loads_back_quantized(quantized_reload):
    """The state, the records and the recall all survive the round trip.

    The loader restores the PQ graph the save dumped, so the reloaded index
    reports quantized_active rather than raw_trained_not_rebuilt and searches
    through ADC exactly as it did before the save. The recall bound below is a
    tolerance rather than an equality because it also has to hold on the
    fallback rebuild, which wires a different graph over the same codes. Test
    106 asserts the equality that the restored path actually delivers.
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
    """The raw store comes back exactly as the live index held it.

    For quantized_only that is empty, since training releases the collected
    raw vectors the moment their codes are stored, and for quantized_with_raw
    it is every record. No code-only record may be materialised at full width
    on load; before relay 35's fix the loader reconstructed them into the
    graph, which is where the mode's memory saving went to die.
    """
    fixture = quantized_reload
    loaded = VectorDatabase().load(str(fixture["path"]))

    expected_raw = 0 if fixture["mode"] == "quantized_only" else RELOAD_TOTAL
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

# ------------------------------------------------------------
# Test 104: a saved index keeps the m it was built with
# ------------------------------------------------------------
def test_reload_preserves_m_against_a_changed_default(tmp_path):
    """The default m now scales with expected_size, so a saved index has to
    carry its own m rather than pick one up at load time.

    An index declared at 30,000 records built before the change holds m 16.
    Loading it must carry 16 forward, because taking the new default of 32
    would change the index under a user who only asked to open it, and would
    cost memory and load time they did not agree to. It is also what the graph
    dump is checked against, so an index that lost its m would rebuild rather
    than restore.
    """
    vdb = VectorDatabase()

    # m 16 at an expected_size the scaled default would now answer with 32.
    index = vdb.create("hnsw", dim=8, space="cosine", m=16,
                       ef_construction=200, expected_size=30_000)
    assert VectorDatabase._default_m(30_000) == 32, "the default has to differ"
    index.add({
        "ids": [f"r{i}" for i in range(20)],
        "embeddings": _sample_vectors(20, 8),
    })

    save_dir = tmp_path / "old_default.zdb"
    index.save(str(save_dir))
    assert json.loads((save_dir / "config.json").read_text(encoding="utf-8"))["m"] == 16

    loaded = vdb.load(str(save_dir))
    assert loaded.get_stats()["m"] == "16"
    assert loaded.get_stats()["expected_size"] == "30000"
    assert loaded.get_vector_count() == 20

    # It survives a second round trip, so the value is restored into the live
    # index rather than only copied from file to file.
    again = tmp_path / "old_default2.zdb"
    loaded.save(str(again))
    assert vdb.load(str(again)).get_stats()["m"] == "16"

    # And the reloaded graph works, so 16 was used to build it rather than
    # merely recorded.
    hits = loaded.search(_sample_vectors(1, 8)[0].tolist(), top_k=5)
    assert len(hits) == 5

# ------------------------------------------------------------
# Test 105: the manifest compression ratio is the compression ratio
# ------------------------------------------------------------
def test_manifest_compression_ratio_is_mode_independent(quantized_reload):
    """Both sizes are taken over the coded records, so the ratio is dim * 4 / subvectors.

    The fixture is parametrised over both storage modes, so this runs twice and
    the two runs assert the same number.

    The numerator used to count the raw vectors the index still holds. Under
    quantized_with_raw that is every coded record and the ratio came out right.
    Under quantized_only it is only the training records, so with 1,000 of the
    1,500 collected before training the manifest read 5.3x where the codes are
    8x smaller than the vectors. The figure now matches the ratio the index
    reports live, in both modes.
    """
    expected = RELOAD_DIM * 4 / RELOAD_SUBVECTORS
    manifest = json.loads(
        (quantized_reload["path"] / "manifest.json").read_text(encoding="utf-8"))
    info = manifest["compression_info"]

    assert info["compression_ratio"] == pytest.approx(expected)
    assert info["original_size_mb"] == pytest.approx(
        RELOAD_TOTAL * RELOAD_DIM * 4 / (1024 * 1024))
    assert info["compressed_size_mb"] == pytest.approx(
        RELOAD_TOTAL * RELOAD_SUBVECTORS / (1024 * 1024))

    # And it agrees with what the live index reports, which is the point of
    # having it in the manifest at all.
    loaded = VectorDatabase().load(str(quantized_reload["path"]))
    assert loaded.get_quantization_info()["compression_ratio"] == pytest.approx(expected)

# ------------------------------------------------------------
# Test 106: the round trip is exact, not merely close
# ------------------------------------------------------------
def test_round_trip_returns_identical_ids_and_scores(quantized_reload):
    """Every page comes back with the same ids and the same score bits.

    The loader restores the graph the save dumped instead of rebuilding it by
    re-inserting every record, so the reloaded index is the saved index rather
    than another index over the same data. Scores are compared as raw float32
    bytes, because the defect this replaces moved them by one unit in the last
    place while leaving the ids alone.
    """
    fixture = quantized_reload
    loaded = VectorDatabase().load(str(fixture["path"]))
    after = _pages(loaded, fixture["queries"])

    ids_differ, scores_differ = _page_differences(fixture["pages_before"], after)
    assert ids_differ == 0, f"{ids_differ} pages returned different ids"
    assert scores_differ == 0, f"{scores_differ} pages returned different scores"
    assert after == fixture["pages_before"]


def test_round_trip_is_exact_without_quantization(tmp_path):
    """The same equality on a raw index, where the drift used to appear.

    An unquantized search scores against the graph's own copy of each vector.
    The rebuild put that copy through add(), which normalises, so a vector that
    was already unit length was normalised twice and every score moved by one
    unit in the last place. Restoring the dump takes the copy the save wrote.
    """
    vectors = _clustered_unit_vectors(600, 32, 20260809)
    queries = _clustered_unit_vectors(40, 32, 917)
    ids = [f"v_{i}" for i in range(len(vectors))]

    index = VectorDatabase().create("hnsw", dim=32, expected_size=600)
    assert index.add({"ids": ids, "embeddings": vectors}).is_success()
    before = _pages(index, queries)

    save_dir = tmp_path / "exact.zdb"
    index.save(str(save_dir))
    loaded = VectorDatabase().load(str(save_dir))

    assert _pages(loaded, queries) == before
    assert loaded.get_stats()["graph_nodes"] == index.get_stats()["graph_nodes"]

# ------------------------------------------------------------
# Test 107: two loads of one directory agree with each other
# ------------------------------------------------------------
def test_two_loads_of_one_directory_agree(quantized_reload):
    """Loading twice gives one answer rather than two.

    The rebuild inserted records in hash map order, which varies between
    processes and between two calls in one process, so two loads of the same
    directory used to disagree on some pages. Nothing iterates a map to insert
    any more.
    """
    fixture = quantized_reload
    first = _pages(VectorDatabase().load(str(fixture["path"])), fixture["queries"])
    second = _pages(VectorDatabase().load(str(fixture["path"])), fixture["queries"])

    assert _page_differences(first, second) == (0, 0)
    assert first == second


def test_save_after_load_rewrites_the_same_graph(quantized_reload, tmp_path):
    """A load, a save and a second load return the same pages again.

    The vendored dump refused to overwrite its own files when the graph it held
    may have been memory mapped, and minted a basename with a random suffix
    instead, so the loader had to take the entry point that left that flag
    clear. Nothing here maps anything and the file is replaced outright, so a
    saved index keeps one graph file however many times it is saved.
    """
    fixture = quantized_reload
    first = VectorDatabase().load(str(fixture["path"]))
    again = tmp_path / "again.zdb"
    first.save(str(again))

    names = sorted(p.name for p in again.iterdir() if p.suffix == ".zdbgraph")
    assert names == ["hnsw_index.zdbgraph"]

    second = VectorDatabase().load(str(again))
    assert _pages(second, fixture["queries"]) == fixture["pages_before"]

    # The graph dump itself is reproduced byte for byte, which the JSON and the
    # bincode maps around it cannot be, since they carry a save timestamp and a
    # hash map's iteration order.
    for name in names:
        assert (again / name).read_bytes() == (fixture["path"] / name).read_bytes()

# ------------------------------------------------------------
# Test 108: a restored index still mutates
# ------------------------------------------------------------
def test_restored_graph_accepts_inserts_removals_and_compaction(tmp_path):
    """A loaded graph is a working graph, not a frozen one."""
    vectors = _clustered_unit_vectors(500, 24, 5150)
    ids = [f"v_{i}" for i in range(len(vectors))]
    index = VectorDatabase().create("hnsw", dim=24, expected_size=500)
    assert index.add({"ids": ids, "embeddings": vectors}).is_success()

    save_dir = tmp_path / "mutable.zdb"
    index.save(str(save_dir))
    loaded = VectorDatabase().load(str(save_dir))

    # Insertion after load, and every new record findable by its own vector.
    fresh = _clustered_unit_vectors(30, 24, 6161)
    fresh_ids = [f"n_{i}" for i in range(len(fresh))]
    result = loaded.add({"ids": fresh_ids, "embeddings": fresh})
    assert result.total_inserted == 30 and result.total_errors == 0
    assert loaded.get_vector_count() == 530
    for wanted, vector in zip(fresh_ids, fresh):
        assert loaded.search(vector.tolist(), top_k=1)[0]["id"] == wanted

    # Every record that was there before the save is still findable.
    for wanted, vector in list(zip(ids, vectors))[:50]:
        assert loaded.search(vector.tolist(), top_k=1)[0]["id"] == wanted

    # Removal after load, and the record leaves every accessor.
    assert loaded.remove_point("v_0") is True
    assert loaded.contains("v_0") is False
    assert loaded.get_vector_count() == 529

    # Compaction after load reclaims exactly the nodes the removal stranded and
    # leaves the ids on the pages it does not touch alone.
    queries = _clustered_unit_vectors(20, 24, 7272)
    before = _pages(loaded, queries)
    stranded = int(loaded.get_stats()["stranded_graph_nodes"])
    assert loaded.compact() == stranded
    assert loaded.get_stats()["stranded_graph_nodes"] == "0"
    assert loaded.get_vector_count() == 529
    assert [[x[0] for x in page] for page in _pages(loaded, queries)] == \
           [[x[0] for x in page] for page in before]

    # And the whole thing survives a second round trip.
    again = tmp_path / "mutable2.zdb"
    loaded.save(str(again))
    reloaded = VectorDatabase().load(str(again))
    assert reloaded.get_vector_count() == 529
    assert _pages(reloaded, queries) == _pages(loaded, queries)

# ------------------------------------------------------------
# Test 109: the rebuild is still there when the dump cannot be used
# ------------------------------------------------------------
@pytest.mark.parametrize("damage", [
    "absent",
    "empty",
    "header_only",
    "truncated_early",
    "truncated_half",
    "truncated_by_one",
    "truncated_trailer",
    "extra_bytes",
    "wrong_magic",
    "wrong_version",
    "corrupt_header",
    "huge_nb_point",
    "corrupt_body",
    "corrupt_trailer_magic",
    "wrong_m",
    "vendored_dump",
])
def test_load_falls_back_to_the_rebuild_when_the_dump_is_unusable(tmp_path, damage):
    """Every record comes back whatever state the dump is in.

    None of these may panic, exit the process, or size a buffer from a length
    the file has not earned. That is the class of failure the vendored reader
    had: it panicked on a malformed header and reached std::process::exit(1) on
    a short data file, which is why the loader used to measure the data file
    beforehand and wrap the reload in two catch_unwind calls.

    A dump written by 0.6.0 or earlier is one of these cases rather than a
    special one. There is deliberately no reader for the vendored format, so
    such a directory reaches the rebuild, comes back whole, and writes the new
    format on its next save.
    """
    vectors = _clustered_unit_vectors(400, 16, 31415)
    ids = [f"v_{i}" for i in range(len(vectors))]
    index = VectorDatabase().create("hnsw", dim=16, expected_size=400)
    assert index.add({"ids": ids, "embeddings": vectors}).is_success()

    save_dir = tmp_path / "damaged.zdb"
    index.save(str(save_dir))
    dump = save_dir / "hnsw_index.zdbgraph"
    blob = dump.read_bytes()

    if damage == "absent":
        dump.unlink()
    elif damage == "empty":
        dump.write_bytes(b"")
    elif damage == "header_only":
        dump.write_bytes(blob[:96])
    elif damage == "truncated_early":
        dump.write_bytes(blob[:200])
    elif damage == "truncated_half":
        dump.write_bytes(blob[: len(blob) // 2])
    elif damage == "truncated_by_one":
        dump.write_bytes(blob[:-1])
    elif damage == "truncated_trailer":
        dump.write_bytes(blob[:-16])
    elif damage == "extra_bytes":
        dump.write_bytes(blob + b"\x00" * 32)
    elif damage == "wrong_magic":
        dump.write_bytes(b"NOTZEUSD" + blob[8:])
    elif damage == "wrong_version":
        dump.write_bytes(blob[:8] + struct.pack("<I", 99) + blob[12:])
    elif damage == "corrupt_header":
        # One flipped bit inside the header, which the header checksum catches
        # before any field is believed.
        dump.write_bytes(blob[:24] + bytes([blob[24] ^ 0x01]) + blob[25:])
    elif damage == "huge_nb_point":
        # A header claiming a billion points. Rewriting the field alone breaks
        # the header checksum, so this case is caught there. The case where the
        # checksum agrees and the count is still absurd is a Rust unit test,
        # since only the writer can produce a consistent header.
        dump.write_bytes(blob[:40] + struct.pack("<Q", 1_000_000_000) + blob[48:])
    elif damage == "corrupt_body":
        # A flipped bit in the vector region, which only the payload checksum
        # can see: every length in the file still agrees.
        at = len(blob) - 64
        dump.write_bytes(blob[:at] + bytes([blob[at] ^ 0x08]) + blob[at + 1:])
    elif damage == "corrupt_trailer_magic":
        dump.write_bytes(blob[:-8] + b"XXXXXXXX")
    elif damage == "wrong_m":
        # A dump written at one m against a config declaring another is what a
        # directory assembled from two indexes looks like.
        config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        config["m"] = 17
        (save_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    elif damage == "vendored_dump":
        # What a directory saved by 0.6.0 or earlier holds. The vendored magic
        # is MAGICDESCR_4, and the two files sit beside a name this build does
        # not write.
        dump.unlink()
        (save_dir / "hnsw_index.hnsw.graph").write_bytes(
            struct.pack("<I", 0x002A6779) + blob[8:2048])
        (save_dir / "hnsw_index.hnsw.data").write_bytes(
            struct.pack("<I", 0xA67F0000) + blob[8:4096])

    loaded = VectorDatabase().load(str(save_dir))
    assert loaded.get_vector_count() == 400
    assert all(loaded.contains(i) for i in ids)
    assert len(loaded.search(vectors[0].tolist(), top_k=10)) == 10

    # And the rebuilt index saves in the new format, so the directory is only
    # ever wrong once.
    again = tmp_path / "repaired.zdb"
    loaded.save(str(again))
    assert (again / "hnsw_index.zdbgraph").exists()
    assert not (again / "hnsw_index.hnsw.graph").exists()
    assert _pages(VectorDatabase().load(str(again)), vectors[3:6]) == \
           _pages(loaded, vectors[3:6])

# ------------------------------------------------------------
# Test 110: the rebuild can be asked for on an intact directory
# ------------------------------------------------------------
def test_the_rebuild_can_be_requested_on_an_intact_directory(tmp_path):
    """The escape hatch that makes a graph defect recoverable by upgrading.

    Restoring the dump restores the graph exactly as it was written, so an
    index whose graph was built by a release carrying a defect keeps it. The
    environment variable is what asks for the graph to be built again by the
    current build instead.
    """
    vectors = _clustered_unit_vectors(400, 16, 2718)
    ids = [f"v_{i}" for i in range(len(vectors))]
    index = VectorDatabase().create("hnsw", dim=16, expected_size=400)
    assert index.add({"ids": ids, "embeddings": vectors}).is_success()

    save_dir = tmp_path / "rebuildable.zdb"
    index.save(str(save_dir))

    restored = VectorDatabase().load(str(save_dir))
    os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"] = "1"
    try:
        rebuilt = VectorDatabase().load(str(save_dir))
        rebuilt_twice = VectorDatabase().load(str(save_dir))
    finally:
        del os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"]

    assert rebuilt.get_vector_count() == restored.get_vector_count() == 400
    assert all(rebuilt.contains(i) for i in ids)

    # Two rebuilds of one directory agree with each other, since neither
    # iterates a hash map to decide the insertion order. This holds below the
    # batch size at which insertion forks to rayon, being 1,000 times the
    # thread count, above which thread scheduling reorders the work and two
    # rebuilds diverge again. 400 records is below it on any machine.
    queries = _clustered_unit_vectors(30, 16, 1618)
    assert _pages(rebuilt, queries) == _pages(rebuilt_twice, queries)

# ------------------------------------------------------------
# Test 111: an index at the top of the m range restores
# ------------------------------------------------------------
@pytest.mark.parametrize("m", [16, 255, 256])
def test_an_index_at_the_top_of_the_m_range_restores(tmp_path, m):
    """m 256 used to rebuild on every load, for ever.

    The vendored header stored max_nb_connection as a u8 while the graph admits
    256, so a dump written at 256 declared 0, the loader compared 0 against the
    256 in config.json, and the directory fell back to the rebuild every single
    time it was opened. m is a u64 in ZeusDB's header.

    The evidence that the dump was read rather than rebuilt is the score bits.
    A rebuild wires a different graph over the same records and would not
    reproduce them exactly.
    """
    vectors = _clustered_unit_vectors(300, 16, 4096)
    ids = [f"v_{i}" for i in range(len(vectors))]
    index = VectorDatabase().create("hnsw", dim=16, m=m, expected_size=300)
    assert index.add({"ids": ids, "embeddings": vectors}).is_success()

    save_dir = tmp_path / f"m{m}.zdb"
    index.save(str(save_dir))

    with open(save_dir / "hnsw_index.zdbgraph", "rb") as handle:
        header = handle.read(96)
    assert struct.unpack_from("<Q", header, 24)[0] == m

    queries = _clustered_unit_vectors(40, 16, 8192)
    loaded = VectorDatabase().load(str(save_dir))
    assert loaded.get_vector_count() == 300
    assert _pages(loaded, queries) == _pages(index, queries)


# ------------------------------------------------------------
# Test 112: the rebuild writes one graph, byte for byte
# ------------------------------------------------------------
def test_the_rebuild_is_byte_deterministic(tmp_path):
    """The fallback stopped going through add(), and nothing about it moved.

    The rebuild used to marshal every record into a PyDict holding three
    PyLists and call add(), which parsed them straight back into the owned Rust
    the loader already had. It now hands that data to the insertion phase
    directly, with the interpreter lock released.

    Two things had to survive, and this is what pins them. The vectors are
    processed for the index space a second time, because a stored vector is
    already normalised for a cosine index and add() applied that processing
    again, and the records are inserted in ascending internal id, which is
    arrival order. Either one changing would wire the graph differently, so
    the dump the rebuild writes is compared byte for byte rather than by
    recall.

    **The rebuilt graph is not the graph the dump holds and never was.** The
    original was built from the vectors as supplied and the rebuild builds from
    the stored ones, processed once more, under internal ids that continue from
    the saved counter. What has to hold is that the rebuild produces one answer
    every time.
    """
    vectors = _clustered_unit_vectors(600, 16, 987654)
    ids = [f"v_{i}" for i in range(len(vectors))]
    index = VectorDatabase().create("hnsw", dim=16, expected_size=600)
    assert index.add({"ids": ids, "embeddings": vectors}).is_success()

    save_dir = tmp_path / "source.zdb"
    index.save(str(save_dir))

    os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"] = "1"
    try:
        first = VectorDatabase().load(str(save_dir))
        second = VectorDatabase().load(str(save_dir))
    finally:
        del os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"]

    dumps = []
    for name, rebuilt in (("first", first), ("second", second)):
        out = tmp_path / f"{name}.zdb"
        rebuilt.save(str(out))
        dumps.append((out / "hnsw_index.zdbgraph").read_bytes())

    assert dumps[0] == dumps[1], (
        "two rebuilds of one directory wired the graph differently"
    )
    assert first.get_vector_count() == second.get_vector_count() == 600
    assert all(first.contains(i) for i in ids)

    queries = _clustered_unit_vectors(20, 16, 24680)
    assert _pages(first, queries) == _pages(second, queries)

    # The rebuild carries the metadata through, so a filtered search over a
    # rebuilt index answers as one over the index it was built from.
    assert first.count({"any_field": "absent"}) == 0


# ------------------------------------------------------------
# Test 113: the rebuild answers what the records say, filters included
# ------------------------------------------------------------
def test_a_rebuilt_index_filters_on_its_restored_metadata(tmp_path):
    """Metadata reaches the rebuilt index without a trip through Python.

    The round trip through add() converted every metadata value into a Python
    object and straight back. The storage maps are written back verbatim after
    the rebuild either way, so this pins that the rebuilt index filters on the
    values the directory holds and not on whatever the trip produced.
    """
    vectors = _clustered_unit_vectors(300, 16, 555)
    ids = [f"v_{i}" for i in range(len(vectors))]
    metadata = [
        {
            "tier": "abc"[i % 3],
            "rank": i,
            "big": 9007199254740993,  # wider than an f64 carries exactly
            "tags": ["x"] if i % 2 else ["y", "z"],
            "empty": None,
        }
        for i in range(len(vectors))
    ]
    index = VectorDatabase().create("hnsw", dim=16, expected_size=300)
    assert index.add({"ids": ids, "embeddings": vectors, "metadatas": metadata}).is_success()

    save_dir = tmp_path / "meta.zdb"
    index.save(str(save_dir))

    os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"] = "1"
    try:
        rebuilt = VectorDatabase().load(str(save_dir))
    finally:
        del os.environ["ZEUSDB_LOAD_REBUILD_GRAPH"]

    assert rebuilt.count({"tier": "a"}) == index.count({"tier": "a"})
    assert rebuilt.count({"rank": {"lt": 50}}) == 50
    assert rebuilt.count({"big": 9007199254740993}) == 300
    assert rebuilt.count({"tags": {"any": ["z"]}}) == index.count({"tags": {"any": ["z"]}})
    assert rebuilt.count({"empty": None}) == 300
    assert rebuilt.get_records("v_7")[0]["metadata"] == index.get_records("v_7")[0]["metadata"]

    # And the composed forms, which reach the rebuilt index like any other.
    assert rebuilt.count({"$or": [{"tier": "a"}, {"tier": "b"}]}) == 200
    assert rebuilt.count({"$not": {"tier": "c"}}) == 200


# ============================================================================
# load() against a half-written directory
# ============================================================================
#
# Every artefact is written with `fs::write` straight into the target directory
# and `manifest.json` is written last, so a save that fails part way leaves a
# mixture of files under no manifest, and a save whose last write failed leaves a
# manifest describing files that are not all there. There is no temporary
# directory and no atomic rename.
#
# What these assert is that such a directory produces an error rather than an
# index that answers queries wrongly. Two damage kinds per artefact, a truncation
# to nothing and a deletion, which are the two shapes an interrupted write
# leaves.

HALF_WRITTEN_N = 1200
HALF_WRITTEN_DIM = 8


def half_written_source(mode):
    """An index of HALF_WRITTEN_N records in one of the three storage modes."""
    quantization = None
    if mode != "raw":
        quantization = {
            "type": "pq", "subvectors": 4, "bits": 8, "training_size": 1000,
            "storage_mode": ("quantized_only" if mode == "quantized_only"
                             else "quantized_with_raw"),
        }
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((HALF_WRITTEN_N, HALF_WRITTEN_DIM)).astype(np.float32)
    ids = [f"r{i}" for i in range(HALF_WRITTEN_N)]
    metadatas = [{"tier": "gold" if i % 2 == 0 else "silver", "n": i}
                 for i in range(HALF_WRITTEN_N)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        index = VectorDatabase().create(
            "hnsw", dim=HALF_WRITTEN_DIM, expected_size=5000,
            quantization_config=quantization,
        )
    assert index.add({"ids": ids, "embeddings": vectors,
                      "metadatas": metadatas}).is_success()
    assert index.is_quantized() == (mode != "raw"), index.get_storage_mode()
    return index, vectors


def damage(directory, name, kind):
    """Truncate a file to nothing or delete it. False if it is not there."""
    target = os.path.join(directory, name)
    if not os.path.exists(target):
        return False
    if kind == "truncate":
        with open(target, "r+b") as handle:
            handle.truncate(0)
    elif kind == "half":
        size = os.path.getsize(target)
        with open(target, "r+b") as handle:
            handle.truncate(max(1, size // 2))
    else:
        os.remove(target)
    return True


# Every artefact whose loss or truncation must be reported rather than absorbed,
# per storage mode. The graph dump is deliberately absent from these lists: it is
# derived data and the loader rebuilds it, which is covered separately below.
#
# These are now every file each mode's directory holds bar the graph. Three
# entries in the quantized_with_raw row used to be missing, being vectors.bin,
# quantization.json and pq_codes.bin, because deleting any of the three loaded
# rather than raised. The loader now checks files_included against the
# directory, so they are here.
MUST_RAISE = {
    "raw": ["config.json", "mappings.bin", "metadata.json", "vectors.bin",
            "manifest.json"],
    "quantized_with_raw": ["config.json", "mappings.bin", "metadata.json",
                           "quantization.json", "pq_centroids.bin",
                           "pq_codes.bin", "vectors.bin", "manifest.json"],
    "quantized_only": ["config.json", "mappings.bin", "metadata.json",
                       "pq_centroids.bin", "pq_codes.bin", "quantization.json",
                       "manifest.json"],
}


@pytest.mark.parametrize("mode", ["raw", "quantized_with_raw", "quantized_only"])
@pytest.mark.parametrize("kind", ["truncate", "half", "delete"])
def test_load_refuses_a_half_written_directory(tmp_path, mode, kind):
    """A damaged artefact is an error, not an index that answers wrongly.

    A truncated file is what an interrupted write leaves and a missing file is
    what a save that never reached that artefact leaves. Either way the
    directory does not describe one index, and load has to say so rather than
    building whatever the surviving files support.
    """
    index, vectors = half_written_source(mode)
    pristine = tmp_path / f"pristine-{mode}.zdb"
    index.save(str(pristine))

    # The undamaged directory is the control. Every assertion below is against
    # an index that loads and answers correctly from these same files.
    control = VectorDatabase().load(str(pristine))
    want = [hit["id"] for hit in control.search(vectors[0], top_k=5)]
    assert len(control) == HALF_WRITTEN_N

    for name in MUST_RAISE[mode]:
        work = tmp_path / f"work-{mode}-{kind}-{name}.zdb"
        shutil.copytree(pristine, work)
        assert damage(str(work), name, kind), f"{name} was not in the directory"

        with pytest.raises(Exception) as excinfo:
            VectorDatabase().load(str(work))
        # An OSError for a file that is gone, a RuntimeError for one that is
        # there and does not parse. Neither is a silent load.
        assert isinstance(excinfo.value, (OSError, RuntimeError, ValueError)), (
            name, kind, type(excinfo.value).__name__
        )
        message = str(excinfo.value)
        assert message, f"{name} {kind} raised with no message"

    # Nothing above touched the pristine directory, which still loads.
    again = VectorDatabase().load(str(pristine))
    assert [hit["id"] for hit in again.search(vectors[0], top_k=5)] == want


@pytest.mark.parametrize("mode", ["raw", "quantized_with_raw", "quantized_only"])
@pytest.mark.parametrize("kind", ["truncate", "half", "delete"])
def test_load_rebuilds_a_damaged_graph_dump_rather_than_refusing(tmp_path, mode, kind):
    """The graph is derived data, so a damaged dump is recoverable and recovered.

    This is the one artefact whose loss is not an error. The records carry
    everything the graph is built from, so the loader rebuilds rather than
    refusing, and it says on stdout which of the three conditions it detected.
    The page has to be the page the intact directory answers, since the records
    are the same records.
    """
    index, vectors = half_written_source(mode)
    pristine = tmp_path / f"graph-{mode}.zdb"
    index.save(str(pristine))

    control = VectorDatabase().load(str(pristine))
    want = [hit["id"] for hit in control.search(vectors[0], top_k=5)]

    work = tmp_path / f"graph-work-{mode}-{kind}.zdb"
    shutil.copytree(pristine, work)
    assert damage(str(work), "hnsw_index.zdbgraph", kind)

    rebuilt = VectorDatabase().load(str(work))
    assert len(rebuilt) == HALF_WRITTEN_N
    assert int(rebuilt.get_stats()["graph_nodes"]) == HALF_WRITTEN_N
    assert [hit["id"] for hit in rebuilt.search(vectors[0], top_k=5)] == want
    assert rebuilt.get_records("r0", return_vector=False)[0]["metadata"]["n"] == 0


def test_load_refuses_a_missing_vectors_bin_under_quantized_with_raw(tmp_path):
    """The manifest inventory is checked against the directory, so this fails.

    Under quantized_with_raw the codes alone yield the full record count, so a
    directory whose vectors.bin was lost used to load as a complete 1,200 record
    index with no error and no warning. What it held was a reconstruction of
    every vector. Measured on the directory this test builds, the worst
    component error against the values supplied was 0.034680 where an intact
    load returns 0.000000, the cosine between the two was 0.998447, and the
    top-5 page moved. raw_vectors_stored read 0, and no caller reads it.

    That is what the refusal prevents. The same check covers the second face,
    a missing quantization.json under the same mode, which used to load as an
    unquantized index, and a missing pq_codes.bin, which used to load as an
    index reporting itself quantized while storing no codes.
    """
    index, vectors = half_written_source("quantized_with_raw")
    pristine = tmp_path / "qraw-refusal.zdb"
    index.save(str(pristine))

    manifest = json.loads((pristine / "manifest.json").read_text(encoding="utf-8"))
    for name in ("vectors.bin", "quantization.json", "pq_codes.bin"):
        assert name in manifest["files_included"], name

    # The intact directory is the control, and it returns what was supplied
    # rather than a reconstruction of it.
    control = VectorDatabase().load(str(pristine))
    assert len(control) == HALF_WRITTEN_N
    exact = np.asarray(control.get_records("r0", return_vector=True)[0]["vector"],
                       dtype=np.float64)
    live = np.asarray(index.get_records("r0", return_vector=True)[0]["vector"],
                      dtype=np.float64)
    assert float(np.abs(exact - live).max()) == 0.0
    # The stored vector is the supplied one normalized, this being a cosine
    # index, so it points the same way.
    supplied = np.asarray(vectors[0], dtype=np.float64)
    assert float(exact @ supplied / (np.linalg.norm(exact) * np.linalg.norm(supplied))) > 0.999999

    # Each of the three names a different file and says what that file holds.
    expected = {
        "vectors.bin": "vectors.bin holds the raw vector of every record",
        "quantization.json": ("quantization.json holds the product quantization "
                              "configuration"),
        "pq_codes.bin": "pq_codes.bin holds the quantized code of every record",
    }
    for name, phrase in expected.items():
        work = tmp_path / f"qraw-refusal-{name}.zdb"
        shutil.copytree(pristine, work)
        os.remove(work / name)
        with pytest.raises(FileNotFoundError) as excinfo:
            VectorDatabase().load(str(work))
        message = str(excinfo.value)
        assert f"manifest.json names {name} under files_included" in message, message
        assert phrase in message, message
        # A missing file and an unparseable one are different failures.
        assert "Failed to parse" not in message

    # A directory missing two reports the first the manifest names rather than
    # whichever one a partial load would have reached.
    two = tmp_path / "qraw-refusal-two.zdb"
    shutil.copytree(pristine, two)
    os.remove(two / "vectors.bin")
    os.remove(two / "metadata.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        VectorDatabase().load(str(two))
    message = str(excinfo.value)
    assert "manifest.json names metadata.json under files_included" in message, message
    assert "1 further file manifest.json names is also absent: vectors.bin." in message

    # And the pristine directory is untouched by any of it.
    again = VectorDatabase().load(str(pristine))
    assert len(again) == HALF_WRITTEN_N
    assert again.is_quantized()
    assert int(again.get_stats()["raw_vectors_stored"]) == HALF_WRITTEN_N


def test_load_exempts_every_graph_name_an_older_manifest_can_carry(tmp_path):
    """A directory written by 0.6.0 or earlier names two graph files, not one.

    The completeness check exempts the graph, and the exemption has to cover the
    names every release wrote or a 0.5.0 and 0.6.0 directory would start
    refusing to open. Those two releases listed hnsw_index.hnsw.graph and
    hnsw_index.hnsw.data under files_included; 0.3.0 through 0.4.1 listed the
    first and excluded the second. Neither pair is readable by this build, which
    rebuilds from the records instead, so neither is required.
    """
    index, vectors = half_written_source("raw")
    pristine = tmp_path / "legacy-graph.zdb"
    index.save(str(pristine))

    control = VectorDatabase().load(str(pristine))
    want = [hit["id"] for hit in control.search(vectors[0], top_k=5)]

    work = tmp_path / "legacy-graph-work.zdb"
    shutil.copytree(pristine, work)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    manifest["files_included"] = [
        name for name in manifest["files_included"] if name != "hnsw_index.zdbgraph"
    ] + ["hnsw_index.hnsw.graph", "hnsw_index.hnsw.data"]
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Neither legacy file is on disk, and the new one is removed too, so the
    # directory holds no graph at all under a manifest that names two.
    os.remove(work / "hnsw_index.zdbgraph")
    assert not (work / "hnsw_index.hnsw.graph").exists()

    rebuilt = VectorDatabase().load(str(work))
    assert len(rebuilt) == HALF_WRITTEN_N
    assert int(rebuilt.get_stats()["graph_nodes"]) == HALF_WRITTEN_N
    assert [hit["id"] for hit in rebuilt.search(vectors[0], top_k=5)] == want


def test_load_treats_a_never_written_file_as_one_removed_afterwards(tmp_path):
    """A manifest naming a file that was never written is the same state.

    A save writes manifest.json after every artefact it names except the graph
    dump, so an interrupted save cannot leave a manifest naming a load bearing
    file it never wrote. The only way to reach that state is to write the
    manifest by hand, and the check cannot tell it from a file removed after a
    completed save, because on disk the two are the same thing. This asserts
    that equivalence rather than assuming it.
    """
    index, _ = half_written_source("raw")
    pristine = tmp_path / "never-written.zdb"
    index.save(str(pristine))

    # A name the save never wrote, added to the inventory by hand.
    invented = tmp_path / "invented.zdb"
    shutil.copytree(pristine, invented)
    manifest = json.loads((invented / "manifest.json").read_text(encoding="utf-8"))
    assert "sidecar.bin" not in manifest["files_included"]
    manifest["files_included"].append("sidecar.bin")
    (invented / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    with pytest.raises(FileNotFoundError) as excinfo:
        VectorDatabase().load(str(invented))
    message = str(excinfo.value)
    assert "manifest.json names sidecar.bin under files_included" in message, message
    # An unrecognised name is still load bearing, and the message says so.
    assert "this build does not recognise" in message, message

    # A file that was written and then removed produces the same shape of
    # message, differing only in what the file holds.
    removed = tmp_path / "removed.zdb"
    shutil.copytree(pristine, removed)
    os.remove(removed / "vectors.bin")
    with pytest.raises(FileNotFoundError) as excinfo:
        VectorDatabase().load(str(removed))
    assert "manifest.json names vectors.bin under files_included" in str(excinfo.value)


def test_load_ignores_an_artefact_the_manifest_does_not_name(tmp_path):
    """Saving over a directory leaves files behind, and they are not read.

    A save replaces files one at a time and removes none, so a raw index saved
    over a quantized one leaves the earlier save quantization.json,
    pq_centroids.bin and pq_codes.bin in place. The record counts agree, so
    nothing caught it, and the directory reopened as a quantized index holding
    the previous save codebook and codes. The manifest is the inventory in both
    directions, so an artefact it does not name is not read.
    """
    quantized, _ = half_written_source("quantized_with_raw")
    shared = tmp_path / "shared.zdb"
    quantized.save(str(shared))
    assert (shared / "pq_codes.bin").exists()

    plain = VectorDatabase().create("hnsw", dim=HALF_WRITTEN_DIM, expected_size=5000)
    rng = np.random.default_rng(77)
    fresh = rng.standard_normal((HALF_WRITTEN_N, HALF_WRITTEN_DIM)).astype(np.float32)
    assert plain.add({
        "ids": [f"r{i}" for i in range(HALF_WRITTEN_N)],
        "embeddings": fresh,
        "metadatas": [{"tier": "gold", "n": i} for i in range(HALF_WRITTEN_N)],
    }).is_success()
    plain.save(str(shared))

    # The quantization files are still on disk and the manifest does not name
    # them, which is exactly the state that used to be read back.
    manifest = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    for name in ("quantization.json", "pq_centroids.bin", "pq_codes.bin"):
        assert (shared / name).exists(), name
        assert name not in manifest["files_included"], name

    reopened = VectorDatabase().load(str(shared))
    assert not reopened.is_quantized()
    assert not reopened.has_quantization()
    assert len(reopened) == HALF_WRITTEN_N
    assert int(reopened.get_stats()["quantized_codes_stored"]) == 0

    # It answers as the raw index that wrote it last, exactly.
    want = [(hit["id"], hit["score"]) for hit in plain.search(fresh[0], top_k=5)]
    assert [(hit["id"], hit["score"]) for hit in reopened.search(fresh[0], top_k=5)] == want
    stored = np.asarray(reopened.get_records("r0", return_vector=True)[0]["vector"],
                        dtype=np.float64)
    kept = np.asarray(plain.get_records("r0", return_vector=True)[0]["vector"],
                      dtype=np.float64)
    assert np.array_equal(stored, kept)
    # Nothing of the quantized save survives into the reopened index.
    assert reopened.get_quantization_info() is None


# ------------------------------------------------------------
# The manifest's descriptive fields
# ------------------------------------------------------------
def _directory_bytes(path):
    return sum(os.path.getsize(os.path.join(path, name)) for name in os.listdir(path))


def _read_manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_total_size_mb_counts_the_graph_dump(tmp_path):
    """The figure is the directory, graph included.

    manifest.json is written before the graph dump, so the size taken there
    misses the largest file in the directory. At 50,000 records of dimension
    1,536 that is roughly 320 MB of a roughly 630 MB directory. On a save over
    an existing directory it counted the previous save's dump, which was a
    different wrong number. The manifest is now written a second time after the
    dump, carrying nothing new but this field.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=32, expected_size=1200)
    rng = np.random.default_rng(20260820)
    saved = tmp_path / "sized.zdb"

    assert index.add({"ids": [f"a{i}" for i in range(800)],
                      "embeddings": rng.random((800, 32)).astype(np.float32)}).is_success()
    index.save(str(saved))

    manifest = _read_manifest(saved)
    on_disk_mb = _directory_bytes(str(saved)) / (1024 * 1024)
    # The graph dump is the file the old figure missed, and it is the largest
    # thing in the directory, so this is not a rounding argument.
    graph_mb = os.path.getsize(saved / "hnsw_index.zdbgraph") / (1024 * 1024)
    assert graph_mb > 0.25 * on_disk_mb
    # Writing the corrected number changes manifest.json's own length by a few
    # bytes, and the figure was taken before that write, so it is short by that
    # difference and by nothing else.
    assert abs(manifest["total_size_mb"] - on_disk_mb) * 1024 * 1024 < 64

    # Saving over the directory records the new size rather than the old one.
    assert index.add({"ids": [f"b{i}" for i in range(400)],
                      "embeddings": rng.random((400, 32)).astype(np.float32)}).is_success()
    index.save(str(saved))
    resaved = _read_manifest(saved)
    on_disk_mb = _directory_bytes(str(saved)) / (1024 * 1024)
    assert resaved["total_size_mb"] > manifest["total_size_mb"]
    assert abs(resaved["total_size_mb"] - on_disk_mb) * 1024 * 1024 < 64

    # The staging file the rewrite goes through is renamed away, so a finished
    # save leaves exactly the artefacts it always left.
    assert sorted(p.name for p in saved.glob("*")) == [
        "config.json",
        "hnsw_index.zdbgraph",
        "manifest.json",
        "mappings.bin",
        "metadata.json",
        "vectors.bin",
    ]

    # And the directory still opens.
    reopened = VectorDatabase().load(str(saved))
    assert len(reopened) == 1200


def test_total_size_mb_on_an_empty_index(tmp_path):
    """An index with no records writes no graph, and the figure still holds."""
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, expected_size=10)
    saved = tmp_path / "empty.zdb"
    index.save(str(saved))

    manifest = _read_manifest(saved)
    assert "hnsw_index.zdbgraph" in manifest["files_excluded"]
    on_disk_mb = _directory_bytes(str(saved)) / (1024 * 1024)
    assert abs(manifest["total_size_mb"] - on_disk_mb) * 1024 * 1024 < 64
    assert manifest["total_size_mb"] > 0


def test_created_at_is_the_creation_not_the_load(tmp_path):
    """A load used to restamp it, so a save afterwards recorded the load."""
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8, expected_size=20)
    index.add({"ids": ["a"], "embeddings": np.full((1, 8), 0.25, dtype=np.float32)})

    first = tmp_path / "c1.zdb"
    index.save(str(first))
    created = _read_manifest(first)["created_at"]

    # Two saves of the same live index have always agreed.
    second = tmp_path / "c2.zdb"
    index.save(str(second))
    assert _read_manifest(second)["created_at"] == created

    # A save of a loaded index now agrees with them.
    loaded = VectorDatabase().load(str(first))
    third = tmp_path / "c3.zdb"
    loaded.save(str(third))
    assert _read_manifest(third)["created_at"] == created

    # saved_at is the one that is meant to move.
    assert _read_manifest(third)["saved_at"] != _read_manifest(first)["saved_at"]


def test_training_completed_at_is_the_training_not_the_save(tmp_path):
    """It carried Utc::now() at save time, so it moved on every save."""
    vdb = VectorDatabase()
    config = {"type": "pq", "subvectors": 4, "bits": 8,
              "training_size": 1000, "storage_mode": "quantized_only"}
    index = vdb.create("hnsw", dim=8, quantization_config=config, expected_size=2000)
    rng = np.random.default_rng(20260820)

    # Untrained, so there is no completion to report.
    early = tmp_path / "t0.zdb"
    index.save(str(early))
    quant = json.loads((early / "quantization.json").read_text(encoding="utf-8"))
    assert quant["is_trained"] is False
    assert quant["training_completed_at"] is None

    assert index.add({"ids": [f"t{i}" for i in range(1000)],
                      "embeddings": rng.random((1000, 8)).astype(np.float32)}).is_success()
    assert index.is_quantized()

    first = tmp_path / "t1.zdb"
    index.save(str(first))
    completed = json.loads((first / "quantization.json").read_text(encoding="utf-8"))
    assert completed["is_trained"] is True
    stamp = completed["training_completed_at"]
    assert stamp is not None

    # A second save of the same trained index does not retrain, so the stamp
    # does not move.
    second = tmp_path / "t2.zdb"
    index.save(str(second))
    again = json.loads((second / "quantization.json").read_text(encoding="utf-8"))
    assert again["training_completed_at"] == stamp

    # Nor does a load and a save.
    loaded = VectorDatabase().load(str(first))
    third = tmp_path / "t3.zdb"
    loaded.save(str(third))
    carried = json.loads((third / "quantization.json").read_text(encoding="utf-8"))
    assert carried["training_completed_at"] == stamp


# ------------------------------------------------------------
# The raw side store of an empty quantized_with_raw index
# ------------------------------------------------------------


def _trained_with_raw(dim=8, records=1050, seed=3):
    """A trained `quantized_with_raw` index, which is the only mode with a store."""
    index = VectorDatabase().create(
        "hnsw", dim=dim, expected_size=4000,
        quantization_config={
            "type": "pq", "subvectors": 4, "bits": 4,
            "training_size": 1000, "storage_mode": "quantized_with_raw",
        },
    )
    vectors = np.random.default_rng(seed).standard_normal((records, dim)).astype(np.float32)
    index.add({"ids": [f"r{i}" for i in range(records)], "embeddings": vectors})
    assert index.is_quantized(), "the fixture must train, or there is no store to lose"
    return index


def _added_vector_comes_back_exactly(index, dim=8):
    """Add one record and report the largest error `get_records` returns on it.

    Zero under `quantized_with_raw`, because the mode keeps a raw vector for
    every record and that is what `get_records` is documented to return. A PQ
    reconstruction of a random vector is wrong in the second decimal place, so
    the two outcomes are not close to each other.
    """
    supplied = np.arange(1.0, dim + 1.0, dtype=np.float32)
    index.add({"ids": ["probe"], "embeddings": [supplied.tolist()]})
    returned = np.asarray(index.get_records("probe")[0]["vector"], dtype=np.float32)
    expected = supplied / np.linalg.norm(supplied)
    return float(np.max(np.abs(returned - expected)))


@pytest.mark.parametrize("emptied_by", ["clear", "remove_points"])
def test_a_quantized_with_raw_index_saved_empty_reopens_with_its_raw_store(
    tmp_path, emptied_by
):
    """A load has to open the raw store whether or not it has anything to put in it.

    The store was opened only when `vectors.bin` had records in it, so a trained
    `quantized_with_raw` index holding nothing at save time came back without a
    store at all. It still reported `quantized_with_raw` and `quantized_active`,
    and every record added after the load lost its raw vector permanently:
    `get_records` fell through to the PQ reconstruction and the rescoring the
    mode exists for had nothing true to rescore against. Nothing raised, and the
    record count was right, so nothing in the suite noticed.

    Two ordinary sequences reach it, and both are checked. `clear()` already
    opened the store on its own replacement graph for exactly this reason, which
    is why the same index emptied and not saved was always correct.

    Found by the random operation sequence in `test_model.py`, at
    `quantized_with_raw` sequence 8 step 278, on `clear` then `save` then `load`
    then `add`.
    """
    index = _trained_with_raw()
    if emptied_by == "clear":
        index.clear()
    else:
        index.remove_points([f"r{i}" for i in range(1050)])
    assert index.get_vector_count() == 0

    path = tmp_path / "emptied"
    index.save(str(path))
    reopened = VectorDatabase().load(str(path))

    assert reopened.get_storage_mode() == "quantized_active"
    assert _added_vector_comes_back_exactly(reopened) < 1e-6, (
        "the reopened index returned a reconstruction, so it came back without "
        "the raw store quantized_with_raw is defined by"
    )


def test_a_quantized_with_raw_index_saved_with_records_keeps_its_store(tmp_path):
    """The case that always worked, held so the fix is not the only thing tested."""
    index = _trained_with_raw()
    path = tmp_path / "populated"
    index.save(str(path))
    reopened = VectorDatabase().load(str(path))
    assert reopened.get_vector_count() == 1050
    assert _added_vector_comes_back_exactly(reopened) < 1e-6
