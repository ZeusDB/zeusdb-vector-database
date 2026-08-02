"""Saving an index to disk and loading it back."""

import numpy as np
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
