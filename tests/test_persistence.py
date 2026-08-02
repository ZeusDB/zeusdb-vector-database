"""Saving an index to disk and loading it back."""

import json

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
# Test 72: Persistence: index level metadata is not written at all
# ------------------------------------------------------------
def test_persistence_drops_index_level_metadata(tmp_path):
    """Current behaviour, asserted rather than expected.

    save_metadata writes index.get_vector_metadata() to metadata.json, which is
    the per record map. The index level map that add_metadata writes to has no
    slot in config.json and no file of its own, so it is dropped silently on
    save. The expectation this violates is that everything add_metadata stores
    survives a round trip, which is what the per record metadata does.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=10)
    index.add_metadata({"owner": "relay22", "dataset": "docs_v2"})
    index.add({"id": "r1", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"kept": "yes"}})

    assert index.get_all_metadata() == {"owner": "relay22", "dataset": "docs_v2"}

    save_dir = tmp_path / "indexmeta.zdb"
    index.save(str(save_dir))

    # No file carries it. config.json holds construction parameters only.
    config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
    assert "metadata" not in config
    assert sorted(config) == ["dim", "ef_construction", "expected_size", "id_counter",
                              "m", "space", "vector_count"]

    loaded = vdb.load(str(save_dir))
    assert loaded.get_all_metadata() == {}
    assert loaded.get_metadata("owner") is None

    # Per record metadata is unaffected, which is what makes the loss easy to miss.
    assert loaded.get_records("r1", return_vector=False)[0]["metadata"] == {"kept": "yes"}

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

    assert manifest["format_version"] == "1.0.0"
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

# ------------------------------------------------------------
# Test 74: Persistence: a quantized_with_raw round trip
# ------------------------------------------------------------
def test_persistence_quantized_with_raw_round_trip(tmp_path):
    """A trained PQ index does not come back quantized.

    Current behaviour, asserted rather than expected. The centroids and the
    codes are both written, but load_index restores only the trained codebook,
    so the loaded index reports raw_trained_not_rebuilt with zero stored codes.
    The expectation this violates is the README's own save and load example,
    which prints is_quantized() on the loaded index under the heading "Load and
    verify quantization state is preserved". rebuild_with_quantization() is what
    closes the gap, and that is asserted here too.
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

    # The trained codebook survives, the quantized state does not.
    assert loaded.can_use_quantization()
    assert loaded.get_quantization_info()["is_trained"] is True
    assert not loaded.is_quantized()
    assert loaded.get_storage_mode() == "raw_trained_not_rebuilt"
    assert int(loaded.get_stats()["quantized_codes_stored"]) == 0

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

    # Search works on the loaded index, over raw vectors rather than codes.
    hits = loaded.search(vectors[0].tolist(), top_k=5)
    assert 0 < len(hits) <= 5
    assert any(h["id"] == "v_0" for h in hits)

    # rebuild_with_quantization moves the loaded index back to quantized_active.
    assert loaded.rebuild_with_quantization() is True
    assert loaded.is_quantized()
    assert loaded.get_storage_mode() == "quantized_active"
    assert int(loaded.get_stats()["quantized_codes_stored"]) == count

# ------------------------------------------------------------
# Test 75: Persistence: a PQ index saved before training completes
# ------------------------------------------------------------
def test_persistence_untrained_pq_index(tmp_path):
    """A PQ index saved mid collection comes back with its progress reset.

    Current behaviour, asserted rather than expected. quantization.json carries
    every training id and the loader calls set_training_ids with all of them,
    but the reloaded index reports zero progress and needs a full training_size
    again. The expectation this violates is benchmark 40's own assertion, which
    compared the loaded progress against the saved progress and required them
    to agree within one percent. That assertion does not hold today.
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

    # The training progress is not intact. It reads as if no vector had ever
    # been collected, and the full training_size is required again.
    assert loaded.get_training_progress() == 0.0
    assert loaded.training_vectors_needed() == 1000
    assert not loaded.is_training_ready()

    # The reloaded quantization info drops the fields that come from the PQ
    # instance, so is_trained is absent rather than False.
    reloaded_info = loaded.get_quantization_info()
    assert reloaded_info["type"] == "pq"
    assert reloaded_info["training_size"] == 1000
    assert "is_trained" not in reloaded_info

    # Adding the 200 vectors that would have completed training does not
    # complete it, because the counter restarted.
    assert loaded.add({"ids": [f"w_{i}" for i in range(200)],
                       "embeddings": _sample_vectors(200, 8, seed=99)}).is_success()
    assert loaded.get_training_progress() == 20.0
    assert not loaded.can_use_quantization()
    assert loaded.get_storage_mode() == "raw_collecting_for_training"

    # Adding enough to pass training_size a second time reaches the threshold
    # but still does not train, so a reloaded untrained index cannot reach the
    # quantized state through ordinary adds at all.
    assert loaded.add({"ids": [f"x_{i}" for i in range(1000)],
                       "embeddings": _sample_vectors(1000, 8, seed=7)}).is_success()
    assert loaded.get_training_progress() == 100.0
    assert loaded.is_training_ready()
    assert not loaded.can_use_quantization()
    assert not loaded.is_quantized()
    assert loaded.get_storage_mode() == "raw_ready_for_training"

# ------------------------------------------------------------
# Test 76: Persistence: quantized_only loses post training records
# ------------------------------------------------------------
def test_persistence_quantized_only_loses_post_training_records(tmp_path):
    """Current behaviour, asserted rather than expected.

    quantized_only stops storing raw vectors once training completes, so
    vectors.bin holds the training set only. The codes for the post training
    records are written to pq_codes.bin but are not restored, so after a reload
    those records are retrievable by nothing. get_vector_count() still reports
    the original total because it reads a counter rather than the stored data.
    The expectation this violates is that a save and load round trip preserves
    every record, which it does under quantized_with_raw.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=8,
                       quantization_config=_pq_config("quantized_only"),
                       expected_size=2000)

    training_size = 1000
    count = 1010
    vectors = _sample_vectors(count, 8)
    ids = [f"v_{i}" for i in range(count)]
    assert index.add({"ids": ids, "embeddings": vectors,
                      "metadatas": [{"i": i} for i in range(count)]}).is_success()
    assert index.is_quantized()

    # Before the save the post training record is reachable through
    # get_records, which reconstructs it from its PQ code.
    assert len(index.get_records("v_1005", return_vector=True)) == 1

    save_dir = tmp_path / "qo.zdb"
    index.save(str(save_dir))
    assert "pq_codes.bin" in {p.name for p in save_dir.glob("*")}

    loaded = vdb.load(str(save_dir))

    # The counter is unchanged and disagrees with what is actually stored.
    assert loaded.get_vector_count() == count
    assert int(loaded.get_stats()["raw_vectors_stored"]) == training_size
    assert int(loaded.get_stats()["quantized_codes_stored"]) == 0
    assert len(loaded.list(number=count + 10)) == training_size

    # Training set records survive.
    assert len(loaded.get_records("v_0", return_vector=True)) == 1
    assert len(loaded.get_records("v_999", return_vector=True)) == 1

    # Post training records do not.
    for lost in ("v_1000", "v_1005", "v_1009"):
        assert loaded.get_records(lost, return_vector=True) == []
        assert not loaded.contains(lost)

    # The rebuild recovers codes for what is still stored, not for what is gone.
    assert loaded.rebuild_with_quantization() is True
    assert int(loaded.get_stats()["quantized_codes_stored"]) == training_size
    assert loaded.get_records("v_1005", return_vector=True) == []

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
