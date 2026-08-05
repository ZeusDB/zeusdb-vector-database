"""Index construction, parameter validation and index type discovery."""

from importlib.metadata import version

import pytest

import zeusdb_vector_database
from zeusdb_vector_database import HNSWIndex, VectorDatabase
from zeusdb_vector_database.zeusdb_vector_database import _create_hnsw_index


def _build_rust(dim=16, space="cosine", m=16, ef_construction=200,
                expected_size=1000, quantization_config=None):
    """Call the extension factory directly, bypassing the Python factory.

    Several rules used to live in VectorDatabase.create alone while the Rust
    constructor stayed reachable, so a rule enforced only there was not a rule.
    This is the path those rules have to hold on.
    """
    return _create_hnsw_index(
        dim=dim,
        space=space,
        m=m,
        ef_construction=ef_construction,
        expected_size=expected_size,
        quantization_config=quantization_config,
    )

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
    """The build counter is gone and the package version is the only version.

    get_version_number and get_code_version reported a hand incremented build
    counter that no release note, documentation page or consumer ever read.
    What the previous test was really asserting is that there is one version
    identifier and that it does not vary between indexes, which the package
    version satisfies.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)

    assert not hasattr(index, "get_version_number")
    assert not hasattr(index, "get_code_version")

    assert isinstance(zeusdb_vector_database.__version__, str)
    assert zeusdb_vector_database.__version__
    assert (
        version("zeusdb-vector-database") == zeusdb_vector_database.__version__
    )

# ------------------------------------------------------------
# Test 90: get_performance_info reports only what the code does
# ------------------------------------------------------------
def test_get_performance_info_reports_only_real_behaviour():
    """langchain-zeusdb calls this method, so it stays and the fields are fixed.

    The three fields that described a parallel insert path are gone, because
    add() runs one record at a time through add_single_vector on every
    unquantized index. The concurrent and batched search claims were verified
    and stay.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)

    info = index.get_performance_info()
    assert isinstance(info, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in info.items())

    assert set(info) == {
        "search_speedup_expected",
        "search_bottleneck",
        "benefits",
        "insertion_path",
    }
    assert info["insertion_path"] == "sequential"
    assert "parallel_insert" not in info["benefits"]

    for gone in ("insertion_speedup_expected", "insertion_bottleneck", "limitation"):
        assert gone not in info

    # A quantized index adds the three quantization keys and nothing else.
    quantized = vdb.create(
        "hnsw",
        dim=16,
        quantization_config={"type": "pq", "subvectors": 4, "bits": 8, "training_size": 1000},
    )
    quantized_info = quantized.get_performance_info()
    assert set(quantized_info) - set(info) == {
        "quantization_compression",
        "quantization_memory_savings",
        "quantization_accuracy_impact",
    }

# ------------------------------------------------------------
# Test 93: HNSWIndex cannot be constructed directly
# ------------------------------------------------------------
def test_hnsw_index_has_no_public_constructor():
    """The class is importable and uninstantiable.

    It carried a PyO3 #[new] reachable from three import paths, and building
    through it skipped every rule that lives in the Python factory. The class
    stays exported so isinstance checks and return annotations work.
    """
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)

    assert isinstance(index, HNSWIndex)

    with pytest.raises(TypeError, match="No constructor defined"):
        HNSWIndex(dim=4, space="cosine", m=16, ef_construction=200, expected_size=10)

    with pytest.raises(TypeError, match="No constructor defined"):
        HNSWIndex(4, "cosine", 16, 200, 10)

    # The registry no longer holds a constructor for a caller who finds it.
    assert not hasattr(VectorDatabase, "_index_constructors")
    assert VectorDatabase._index_types["hnsw"]
    assert isinstance(VectorDatabase._index_types["hnsw"], str)

# ------------------------------------------------------------
# Test 94: every exported name resolves at package level
# ------------------------------------------------------------
def test_public_exports():
    """The README logging recipe raised AttributeError on line three.

    __all__ was ["VectorDatabase"] and nothing else was bound here, so
    zeusdb_vector_database.init_logging did not resolve. The names come from
    the #[pyfunction(name = ...)] attributes in vdb-core/src/logging.rs.
    """
    assert zeusdb_vector_database.__all__ == [
        "AddResult",
        "HNSWIndex",
        "VectorDatabase",
        "__version__",
        "init_file_logging",
        "init_logging",
        "is_logging_initialized",
    ]

    for name in zeusdb_vector_database.__all__:
        assert hasattr(zeusdb_vector_database, name), name

    assert isinstance(zeusdb_vector_database.HNSWIndex, type)
    assert isinstance(zeusdb_vector_database.AddResult, type)
    assert isinstance(zeusdb_vector_database.VectorDatabase, type)
    assert isinstance(zeusdb_vector_database.__version__, str)

    for name in ("init_logging", "init_file_logging", "is_logging_initialized"):
        assert callable(getattr(zeusdb_vector_database, name)), name

    # The types are the ones the API really hands back.
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4)
    result = index.add({"id": "a", "values": [0.1, 0.2, 0.3, 0.4]})
    assert isinstance(index, zeusdb_vector_database.HNSWIndex)
    assert isinstance(result, zeusdb_vector_database.AddResult)

    # is_logging_initialized answers without side effects and reports a bool.
    assert isinstance(zeusdb_vector_database.is_logging_initialized(), bool)

# ------------------------------------------------------------
# Test 95: m has a lower bound
# ------------------------------------------------------------
def test_m_lower_bound():
    """m below 2 builds a degenerate graph and is refused.

    Zero gave every node zero neighbour capacity. One is worse than it looks:
    the layer scale is 1 / ln(m), which is infinity at 1, so every point
    overflows the layer cap and is redispatched uniformly across all 16 layers
    rather than following the exponential distribution. Measured on 3,000
    records of 32 dimensions, recall at 10 was 0.0220 at m 1 against 0.6880 at
    m 2. Both are refused, and the message says why.
    """
    vdb = VectorDatabase()

    for bad in (0, 1):
        with pytest.raises(RuntimeError, match="m must be at least 2"):
            vdb.create("hnsw", dim=4, m=bad)

    with pytest.raises(RuntimeError, match="scale of 1 / ln"):
        vdb.create("hnsw", dim=4, m=1)

    # Two is the smallest value that is accepted, and it works.
    index = vdb.create("hnsw", dim=4, m=2, expected_size=10)
    index.add({"id": "a", "values": [0.1, 0.2, 0.3, 0.4]})
    assert index.get_vector_count() == 1
    assert index.get_stats()["m"] == "2"

# ------------------------------------------------------------
# Test 96: max_training_vectors is enforced in Rust, not only in Python
# ------------------------------------------------------------
def test_max_training_vectors_floor():
    """A max below the threshold produces an index that can never train.

    The rule lived in the Python factory alone, and the Rust constructor was
    reachable directly, so it could be skipped. It is now enforced in the
    Rust builder that both paths go through, which is what _build_rust asserts.
    """
    vdb = VectorDatabase()

    with pytest.raises(ValueError, match="must be >= training_size"):
        vdb.create(
            "hnsw",
            dim=16,
            quantization_config={
                "type": "pq",
                "subvectors": 4,
                "bits": 8,
                "training_size": 2000,
                "max_training_vectors": 1500,
            },
        )

    with pytest.raises(ValueError, match="must be >= training_size"):
        _build_rust(
            dim=16,
            quantization_config={
                "type": "pq",
                "subvectors": 4,
                "bits": 8,
                "training_size": 2000,
                "max_training_vectors": 1500,
            },
        )

    # Equal is accepted, which is the boundary the rule names.
    index = vdb.create(
        "hnsw",
        dim=16,
        quantization_config={
            "type": "pq",
            "subvectors": 4,
            "bits": 8,
            "training_size": 2000,
            "max_training_vectors": 2000,
        },
    )
    assert index.get_quantization_info()["max_training_vectors"] == 2000

# ------------------------------------------------------------
# Test 97: the subvector rules are enforced in Rust as well as in Python
# ------------------------------------------------------------
@pytest.mark.parametrize("subvectors,message", [
    (0, "positive integer"),
    (32, "cannot exceed dimension"),
    (7, "must divide dimension"),
])
def test_subvector_rules(subvectors, message):
    """Python keeps the friendlier message and Rust holds the rule.

    Python rejects first, so create() reports the message it always did. The
    Rust builder is checked separately, because it is the layer that has to
    hold when the Python factory is not in the path.
    """
    vdb = VectorDatabase()

    with pytest.raises(ValueError):
        vdb.create(
            "hnsw",
            dim=16,
            quantization_config={
                "type": "pq",
                "subvectors": subvectors,
                "bits": 8,
                "training_size": 1000,
            },
        )

    with pytest.raises(ValueError, match=message):
        _build_rust(
            dim=16,
            quantization_config={
                "type": "pq",
                "subvectors": subvectors,
                "bits": 8,
                "training_size": 1000,
            },
        )

# ------------------------------------------------------------
# Test 98: the default m scales with expected_size
# ------------------------------------------------------------
@pytest.mark.parametrize("expected_size,expected_m", [
    (1, 16),
    (1_000, 16),
    (10_000, 16),
    (25_000, 16),          # last size measured adequate at m 16
    (25_001, 32),
    (100_000, 32),
    (1_000_000, 32),
    (100_000_000, 32),     # the ladder stops at 32, where the measurements stop
])
def test_default_m_scales_with_expected_size(expected_size, expected_m):
    """A fixed m of 16 capped recall on any index past about 25,000 records.

    At 100,000 records on clustered 768 dimensional data, recall at 10 at the
    default search width was 0.8025 at m 16 against 0.9870 at m 32, and no
    search width recovered the difference. m is the parameter that has to
    scale, and expected_size is the size the user already declares.

    The expression is checked directly rather than through create() for every
    size, because the graph reserves about 3KB of capacity per declared record
    at creation, so building an index at the larger sizes here would commit
    tens of gigabytes. The sizes that are built are checked below.
    """
    assert VectorDatabase._default_m(expected_size) == expected_m


@pytest.mark.parametrize("expected_size,expected_m", [
    (1_000, 16),
    (25_000, 16),
    (25_001, 32),
    (50_000, 32),
])
def test_scaled_default_m_reaches_the_built_index(expected_size, expected_m):
    """The ladder is applied by create(), not merely computed by it."""
    index = VectorDatabase().create("hnsw", dim=4, expected_size=expected_size)
    assert index.get_stats()["m"] == str(expected_m)
    assert f"m={expected_m}," in index.info()

# ------------------------------------------------------------
# Test 99: an explicit m still wins, and the ladder never leaves the valid range
# ------------------------------------------------------------
def test_explicit_m_overrides_the_scaled_default():
    """The scaled default is a default, not a policy applied over the caller."""
    vdb = VectorDatabase()

    # Below the ladder, above it, and at both ends of the valid range, on the
    # side of the threshold where the default would otherwise answer 32.
    for m in (2, 8, 16, 48, 256):
        index = vdb.create("hnsw", dim=4, m=m, expected_size=30_000)
        assert index.get_stats()["m"] == str(m)

    # The ladder never produces a value outside the bounds for any
    # expected_size the Rust layer accepts.
    with pytest.raises(RuntimeError, match="m must be at least 2"):
        vdb.create("hnsw", dim=4, m=0, expected_size=30_000)
    with pytest.raises(RuntimeError, match="less than or equal to 256"):
        vdb.create("hnsw", dim=4, m=257, expected_size=10)

    # An index created with no expected_size keeps the historical m 16, so the
    # smallest indexes see no change at all.
    assert vdb.create("hnsw", dim=4).get_stats()["m"] == "16"

    # A non-integer expected_size is left for the Rust layer to reject, rather
    # than failing inside the ladder with a worse message.
    assert VectorDatabase._default_m("large") == 16
    assert VectorDatabase._default_m(True) == 16
    with pytest.raises(RuntimeError):
        vdb.create("hnsw", dim=4, expected_size="large")
