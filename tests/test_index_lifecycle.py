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

# ------------------------------------------------------------
# Test 87: get_space returns the configured distance metric
# ------------------------------------------------------------
@pytest.mark.parametrize("requested,expected", [
    ("cosine", "cosine"),
    ("l2", "l2"),
    ("l1", "l1"),
    ("COSINE", "cosine"),
    ("L2", "l2"),
    ("Cosine", "cosine"),
])
def test_get_space(requested, expected):
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space=requested)

    # The space is normalized to lower case at construction, so get_space
    # reports the canonical form rather than the spelling that was passed.
    assert index.get_space() == expected
    assert index.get_stats()["space"] == expected
    assert f"space={expected}" in index.info()

# ------------------------------------------------------------
# Test 88: the dim getter property
# ------------------------------------------------------------
def test_dim_property():
    vdb = VectorDatabase()

    index = vdb.create("hnsw", dim=384)
    assert index.dim == 384
    assert isinstance(index.dim, int)

    # It is a read only getter, not a settable attribute.
    with pytest.raises(AttributeError):
        index.dim = 128

    # It agrees with the string form get_stats reports and does not move when
    # vectors are added.
    assert index.get_stats()["dimension"] == "384"
    index.add({"id": "a", "values": [0.1] * 384, "metadata": {}})
    assert index.dim == 384

    # The default dimension is 1536.
    assert vdb.create().dim == 1536

# ------------------------------------------------------------
# Test 89: version reporting
# ------------------------------------------------------------
def test_version_reporting():
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)

    number = index.get_version_number()
    assert isinstance(number, int)
    assert number > 0

    description = index.get_code_version()
    assert isinstance(description, str)
    # The two are the same constant, one bare and one wrapped in a sentence.
    assert description.startswith(f"Version: {number}, Description: ")
    assert len(description) > len(f"Version: {number}, Description: ")

    # Both are properties of the build rather than of the index, so a second
    # index with a different configuration reports the same values.
    other = vdb.create("hnsw", dim=64, space="l2")
    assert other.get_version_number() == number
    assert other.get_code_version() == description

    # This counter is not the package version. Benchmark 36 recorded 1001 as
    # the expected output and the current build reports a higher number, so it
    # is a monotonic build counter and nothing here pins its value.
    assert number != 0

# ------------------------------------------------------------
# Test 90: get_performance_info returns a mapping
# ------------------------------------------------------------
def test_get_performance_info_returns_a_dict():
    """Deliberately shallow.

    Three of the fields this returns describe a parallel insert path that does
    not exist, and they are being removed. Asserting their contents would lock
    them in, so this pins the return type only.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)

    info = index.get_performance_info()
    assert isinstance(info, dict)
    assert info
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in info.items())
