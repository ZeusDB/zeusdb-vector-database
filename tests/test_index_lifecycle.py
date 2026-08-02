"""Index construction, parameter validation and index type discovery."""

import pytest
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 1: Test the creation of an HNSW index with default parameters
# ------------------------------------------------------------
def test_create_index_hnsw_default():
    vdb = VectorDatabase()
    index = vdb.create()  # Uses default index_type="hnsw"
    assert index is not None
    assert index.info() is not None

# ------------------------------------------------------------
# Test 2: Test the creation of an HNSW index with custom parameters
# ------------------------------------------------------------
def test_create_index_hnsw_custom():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=10)
    assert index is not None
    stats = index.get_stats()
    assert stats["dimension"] == "4"
    assert stats["expected_size"] == "10"

# ------------------------------------------------------------
# Test 18: Test parameter validation during index creation
# ------------------------------------------------------------
def test_index_creation_validation():
    vdb = VectorDatabase()
    
    # Test invalid dimension
    with pytest.raises(RuntimeError):
        vdb.create("hnsw", dim=0)
    
    # Test invalid ef_construction
    with pytest.raises(RuntimeError):
        vdb.create("hnsw", ef_construction=0)
    
    # Test invalid expected_size
    with pytest.raises(RuntimeError):
        vdb.create("hnsw", expected_size=0)
    
    # Test invalid m
    with pytest.raises(RuntimeError):
        vdb.create("hnsw", m=300)  # > 256
    
    # Test invalid space
    with pytest.raises(RuntimeError):
        vdb.create("hnsw", space="invalid")

# ------------------------------------------------------------
# Test 20: Test case insensitive distance metrics
# ------------------------------------------------------------
def test_case_insensitive_metrics():
    vdb = VectorDatabase()
    
    # Test lowercase
    index1 = vdb.create("hnsw", dim=4, space="cosine")
    assert index1 is not None
    
    # Test uppercase
    index2 = vdb.create("hnsw", dim=4, space="COSINE")
    assert index2 is not None
    
    # Test mixed case
    index3 = vdb.create("hnsw", dim=4, space="Cosine")
    assert index3 is not None

# ------------------------------------------------------------
# Test 25: Test new create() method with various index types
# ------------------------------------------------------------
def test_new_create_method():
    vdb = VectorDatabase()
    
    # Test default (should create HNSW)
    index1 = vdb.create()
    assert index1 is not None
    assert "hnsw" in index1.info().lower()
    
    # Test explicit HNSW
    index2 = vdb.create("hnsw", dim=128)
    assert index2 is not None
    stats = index2.get_stats()
    assert stats["dimension"] == "128"
    assert stats["index_type"] == "HNSW"
    
    # Test case insensitive index type
    index3 = vdb.create("HNSW", dim=64)
    assert index3 is not None
    
    # Test invalid index type
    with pytest.raises(ValueError, match="Unknown index type"):
        vdb.create("invalid_type")

# ------------------------------------------------------------
# Test 26: Test available_index_types method
# ------------------------------------------------------------
def test_available_index_types():
    vdb = VectorDatabase()
    
    # Test class method
    available = VectorDatabase.available_index_types()
    assert isinstance(available, list)
    assert "hnsw" in available
    assert len(available) >= 1
    
    # Test instance method
    available_instance = vdb.available_index_types()
    assert available_instance == available
