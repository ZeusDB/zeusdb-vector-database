"""Index construction, parameter validation and index type discovery."""

import contextlib
import warnings
from importlib.metadata import version

import pytest

import zeusdb_vector_database
from zeusdb_vector_database import HNSWIndex, VectorDatabase
from zeusdb_vector_database._engine import _create_hnsw_index


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
    # index_type still defaults to hnsw. dim does not default, so it is given.
    index = vdb.create(dim=1536)
    assert index is not None
    assert index.info() is not None
    assert index.dim == 1536

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
    with pytest.raises(ValueError):
        vdb.create("hnsw", dim=0)
    
    # Test invalid ef_construction
    with pytest.raises(ValueError):
        vdb.create("hnsw", dim=4, ef_construction=0)
    
    # Test invalid expected_size
    with pytest.raises(ValueError):
        vdb.create("hnsw", dim=4, expected_size=0)
    
    # Test invalid m
    with pytest.raises(ValueError):
        vdb.create("hnsw", dim=4, m=300)  # > 256
    
    # Test invalid space
    with pytest.raises(ValueError):
        vdb.create("hnsw", dim=4, space="invalid")

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
    index1 = vdb.create(dim=1536)
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

    # dim is required as of 0.8.0. It defaulted to 1536, so vdb.create()
    # silently built an index for one vendor's model family and then rejected
    # every vector of any other width.
    with pytest.raises(TypeError, match="requires 'dim'"):
        vdb.create()
    with pytest.raises(TypeError, match="requires 'dim'"):
        vdb.create("hnsw")
    with pytest.raises(TypeError, match="requires 'dim'"):
        vdb.create("hnsw", space="l2", m=16, expected_size=100)

    # The message names the parameter and says why there is no default.
    with pytest.raises(TypeError, match="no default because dim has to equal"):
        vdb.create("hnsw")

    # An unknown index type is still reported before dim is looked at, so the
    # two mistakes do not mask each other.
    with pytest.raises(ValueError, match="Unknown index type"):
        vdb.create("ivf")

    # Given, it is what the index reports.
    assert vdb.create("hnsw", dim=1536).dim == 1536
    assert vdb.create(dim=384).dim == 384

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

    # A quantized index adds the two quantization keys and nothing else.
    #
    # quantization_memory_savings was the third. It carried
    # 1 - 1/compression_ratio as a percentage, which is the share of a vector a
    # code replaces rather than the memory the index saves, and under
    # quantized_with_raw the index saves nothing on the vectors because it keeps
    # every one of them. quantization_compression is the same number in the form
    # that is true, so the field went rather than being qualified.
    quantized = vdb.create(
        "hnsw",
        dim=16,
        quantization_config={"type": "pq", "subvectors": 4, "bits": 8, "training_size": 1000},
    )
    quantized_info = quantized.get_performance_info()
    assert set(quantized_info) - set(info) == {
        "quantization_compression",
        "quantization_accuracy_impact",
    }
    assert quantized_info["quantization_compression"] == "16.0x"

    # Product quantization scores against codes that approximate each vector,
    # so its recall is below that of the same data unquantized. Only
    # quantized_with_raw can rerank against the raw vectors, so the two modes
    # report different impacts.
    assert quantized_info["quantization_accuracy_impact"] == (
        "large_recall_loss_no_rerank_available")

    with pytest.warns(UserWarning, match="keeps a raw vector for every record"):
        with_raw = vdb.create(
            "hnsw",
            dim=16,
            quantization_config={"type": "pq", "subvectors": 4, "bits": 8,
                                 "training_size": 1000,
                                 "storage_mode": "quantized_with_raw"},
        )
    assert with_raw.get_performance_info()["quantization_accuracy_impact"] == (
        "large_recall_loss_unless_reranked")

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

    # PyO3 owns this message and changed it at 0.29. Up to 0.25 it raised its
    # own "No constructor defined". From 0.28 the module uses PEP 489 multi
    # phase initialisation and the class is built from a type spec carrying no
    # tp_new, so CPython raises the refusal itself and words it its own way.
    # The exception type, the refusal and the two call shapes are unchanged.
    with pytest.raises(TypeError, match="cannot create .*HNSWIndex"):
        HNSWIndex(dim=4, space="cosine", m=16, ef_construction=200, expected_size=10)

    with pytest.raises(TypeError, match="cannot create .*HNSWIndex"):
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
    the #[pyfunction(name = ...)] attributes in bindings/python/src/logging.rs.
    """
    assert zeusdb_vector_database.__all__ == [
        "AddResult",
        "HNSWIndex",
        "VectorDatabase",
        "__version__",
        "init_file_logging",
        "init_logging",
        "is_logging_initialized",
        "shutdown_logging",
    ]

    for name in zeusdb_vector_database.__all__:
        assert hasattr(zeusdb_vector_database, name), name

    assert isinstance(zeusdb_vector_database.HNSWIndex, type)
    assert isinstance(zeusdb_vector_database.AddResult, type)
    assert isinstance(zeusdb_vector_database.VectorDatabase, type)
    assert isinstance(zeusdb_vector_database.__version__, str)

    for name in ("init_logging", "init_file_logging", "is_logging_initialized",
                 "shutdown_logging"):
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
        with pytest.raises(ValueError, match="m must be at least 2"):
            vdb.create("hnsw", dim=4, m=bad)

    with pytest.raises(ValueError, match="scale of 1 / ln"):
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
    size. The sizes that are built are checked below.
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
    with pytest.raises(ValueError, match="m must be at least 2"):
        vdb.create("hnsw", dim=4, m=0, expected_size=30_000)
    with pytest.raises(ValueError, match="less than or equal to 256"):
        vdb.create("hnsw", dim=4, m=257, expected_size=10)

    # An index created with no expected_size keeps the historical m 16, so the
    # smallest indexes see no change at all.
    assert vdb.create("hnsw", dim=4).get_stats()["m"] == "16"

    # A non-integer expected_size is left for the Rust layer to reject, rather
    # than failing inside the ladder with a worse message.
    assert VectorDatabase._default_m("large") == 16
    assert VectorDatabase._default_m(True) == 16
    with pytest.raises(TypeError):
        vdb.create("hnsw", dim=4, expected_size="large")


# ------------------------------------------------------------
# Test 19: The neighbour selection heuristic warning
# ------------------------------------------------------------
@contextlib.contextmanager
def _no_user_warning():
    """A context in which any UserWarning becomes a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


def test_neighbour_selection_heuristic_warning():
    """`create()` warns exactly where `ef_construction` is not above `2 * m`.

    The heuristic in `select_neighbours` is skipped when the candidate list is
    no longer than the neighbour budget, which is `2 * m` at layer zero, and
    the candidate list is exactly `ef_construction` long on any index holding
    more records than that. The threshold is therefore exact and the warning
    carries no margin, which is what the pairs at 64 and 65 assert.
    """
    vdb = VectorDatabase()

    # The scaling default produces two values of m and nothing else, so those
    # two are the whole of what the default ef_construction has to clear.
    produced = {VectorDatabase._default_m(size)
                for size in (1, 10, 1_000, 24_999, 25_000, 25_001,
                             50_000, 1_000_000, 100_000_000)}
    assert produced == {16, 32}
    assert all(200 > 2 * m for m in produced)

    # And no default combination reaches it in practice either.
    with _no_user_warning():
        for expected_size in (1, 10, 10_000, 25_000, 25_001, 1_000_000):
            vdb.create("hnsw", dim=4, expected_size=expected_size)
        vdb.create("hnsw", dim=4)

    # Exactly at the budget it fires, and one above it does not.
    with pytest.warns(UserWarning,
                      match=r"ef_construction=64 is not greater than 2\*m=64"):
        vdb.create("hnsw", dim=4, m=32, ef_construction=64)
    with _no_user_warning():
        vdb.create("hnsw", dim=4, m=32, ef_construction=65)

    # The documented configuration that reaches it: a raised m with the
    # ef_construction default left alone. The message names the largest m that
    # leaves the heuristic running at that ef_construction.
    with pytest.warns(UserWarning, match=r"lower m to 99 or below"):
        vdb.create("hnsw", dim=4, m=100)
    with _no_user_warning():
        vdb.create("hnsw", dim=4, m=99)

    # Well below the budget it still fires, since the heuristic is skipped
    # there too.
    with pytest.warns(UserWarning, match=r"2\*m=512"):
        vdb.create("hnsw", dim=4, m=256)

    # Below the m floor of 2 there is no m that would run the heuristic, so
    # the message offers only the one remedy that exists.
    with pytest.warns(UserWarning) as caught:
        vdb.create("hnsw", dim=4, m=2, ef_construction=4)
    assert "Raise ef_construction above 4, to run it." in str(caught[0].message)
    assert "lower m" not in str(caught[0].message)

    # A pair the Rust layer rejects gets its real error and no warning.
    with _no_user_warning():
        with pytest.raises(ValueError, match="ef_construction must be positive"):
            vdb.create("hnsw", dim=4, ef_construction=0)
        with pytest.raises(ValueError, match="less than or equal to 256"):
            vdb.create("hnsw", dim=4, m=257)
        with pytest.raises(ValueError, match="m must be at least 2"):
            vdb.create("hnsw", dim=4, m=1)

    # The warning points at the caller's create() rather than inside the
    # factory, so a user sees their own line.
    with pytest.warns(UserWarning) as caught:
        vdb.create("hnsw", dim=4, m=100)
    assert caught[0].filename == __file__


def test_ef_construction_has_a_ceiling_and_the_ceiling_is_inclusive():
    """`ef_construction` was the last creation parameter with no upper bound.

    It sizes the two candidate heaps every insertion allocates, 8 bytes a slot,
    so create(ef_construction=2**40) succeeded and the first add() asked for
    8 TB and killed the interpreter with exit status 3221226505. The bound
    lives in the validator dim, m and expected_size share, so create(),
    rebuild() and load() refuse the same value with the same message.

    No subprocess here, unlike the dim case in test_hostile_files.py. create()
    never allocated from this value, so a broken bound creates an index rather
    than dying, and the raises below fail in-process.
    """
    vdb = VectorDatabase()
    with pytest.raises(ValueError, match="ef_construction must be at most 4096, got 4097"):
        vdb.create("hnsw", dim=4, ef_construction=4097)
    with pytest.raises(ValueError, match="ef_construction must be at most 4096, got 1099511627776"):
        vdb.create("hnsw", dim=4, ef_construction=1 << 40)
    index = vdb.create("hnsw", dim=4, ef_construction=4096, expected_size=4)
    assert index.ef_construction == 4096
    index.add({"ids": ["a"], "embeddings": [[1.0, 0.0, 0.0, 0.0]]})
    assert len(index) == 1


# ------------------------------------------------------------
# create() raises what the factory raises
# ------------------------------------------------------------
@pytest.mark.parametrize("kwargs,klass,words", [
    ({"dim": 4, "m": 1}, ValueError, "m must be at least 2, got 1."),
    ({"dim": 4, "space": "hamming"}, ValueError,
     "Unsupported space: 'hamming'. Supported spaces: 'cosine', 'l2', 'l1', 'dot'"),
    ({"dim": 0}, ValueError, "dim must be positive, got 0"),
    ({"dim": "four"}, TypeError, "'str' object cannot be interpreted as an integer"),
    ({"dim": 4, "ef_construction": 0}, ValueError, "ef_construction must be positive, got 0"),
    ({"dim": 4, "expected_size": 10**9}, ValueError,
     "expected_size must be at most 100000000, got 1000000000."),
    ({"dim": 4, "indexed_fields": ["$and"]}, ValueError, 'indexed_fields names "$and"'),
    ({"dim": 4, "sparse": {"unlink": "sometimes"}}, ValueError,
     "unknown variant `sometimes`, expected one of `strand`, `lazy`, `eager`"),
    ({"dim": 8, "space": "dot", "quantization_config": {"type": "pq", "subvectors": 2}},
     ValueError, "space='dot' cannot be product quantized."),
    ({"dim": 4, "colour": "red"}, TypeError, "unexpected keyword argument 'colour'"),
])
def test_create_raises_what_the_factory_raises(kwargs, klass, words):
    """Every refusal reaches the caller in the factory's class and words, with
    no prefix naming the factory and no exception beneath it, and an unknown
    space is a ValueError as every other rule of the declaration is."""
    with pytest.raises(klass) as raised:
        VectorDatabase().create("hnsw", **kwargs)
    assert type(raised.value) is klass
    assert words in str(raised.value)
    assert not str(raised.value).startswith("Failed to create")
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("name", ["dim", "m", "ef_construction", "expected_size"])
def test_create_refuses_a_negative_count_by_name(name):
    """A negative count is a value outside the range, refused with ValueError
    naming it."""
    kwargs = {"dim": 4, name: -3}
    with pytest.raises(ValueError) as raised:
        VectorDatabase().create("hnsw", **kwargs)
    assert str(raised.value) == f"{name} cannot be negative, got -3"
    with pytest.raises(ValueError) as raised:
        _create_hnsw_index(**{"dim": 4, "space": "l2", "m": 8, "ef_construction": 50,
                              "expected_size": 10, name: -3})
    assert str(raised.value) == f"{name} cannot be negative, got -3"


def test_the_factory_docstrings_say_what_it_raises():
    """`create()` names the classes it raises, and the field list check names
    the rules it checks rather than every rule."""
    create_doc = VectorDatabase.create.__doc__
    assert "TypeError: If dim is missing" in create_doc
    assert "RuntimeError: If index creation fails due to backend validation" not in create_doc
    fields_doc = VectorDatabase._validate_indexed_fields.__doc__
    assert "Every rule a declared field list has to satisfy" not in fields_doc
    assert "held to their rules by the Rust constructor" in fields_doc
    from zeusdb_vector_database import _engine
    assert "load(path, tokenizer=None)" not in _engine._load_index.__doc__
    assert "Useful for determining whether to set" not in _engine.is_logging_initialized.__doc__


# ------------------------------------------------------------
# An index_type that is not a string
# ------------------------------------------------------------
@pytest.mark.parametrize("index_type", [1, 1.5, b"hnsw", ["hnsw"], None, False, object()],
                         ids=["int", "float", "bytes", "list", "None", "False", "object"])
def test_an_index_type_that_is_not_a_string_raises_type_error(index_type):
    """A parameter of the wrong type raises TypeError, as the docstring says,
    and an index_type that is not a str is one, None among them."""
    with pytest.raises(TypeError) as refusal:
        VectorDatabase().create(index_type, dim=4)
    assert str(refusal.value) == (
        f"index_type must be a str such as 'hnsw', got {type(index_type).__name__}.")


# ------------------------------------------------------------
# A value the engine refuses, beside a quantization config
# ------------------------------------------------------------
def engine_refusal(**arguments):
    """What create() raises for `arguments` with no quantization config,
    being the engine's own refusal."""
    with pytest.raises((ValueError, TypeError)) as refusal:
        VectorDatabase().create("hnsw", **arguments)
    return refusal.type, str(refusal.value)


@pytest.mark.parametrize("config", [{"type": "pq"}, {"type": "pq", "subvectors": 8}],
                         ids=["derived", "given"])
@pytest.mark.parametrize("dim", [-3, 0, 65_537, "4"], ids=["negative", "zero", "above", "str"])
def test_a_dim_the_engine_refuses_raises_its_refusal_alone_beside_a_pq_config(dim, config):
    """The rules a product quantized config derives from dim stand aside for
    a dim the engine refuses, so the caller reads the engine's refusal of
    dim, and no warning, whatever the config."""
    expected = engine_refusal(dim=dim)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, TypeError)) as refusal:
            VectorDatabase().create("hnsw", dim=dim, quantization_config=dict(config))
    assert (refusal.type, str(refusal.value)) == expected
    assert [str(w.message) for w in caught] == []


@pytest.mark.parametrize("config", [{"type": "pq", "subvectors": 4}, {"type": "int8"}],
                         ids=["pq", "int8"])
@pytest.mark.parametrize("expected_size", [-5, 0], ids=["negative", "zero"])
def test_an_expected_size_the_engine_refuses_raises_its_refusal_alone(expected_size, config):
    """A training size warning stands aside for an expected_size the engine
    refuses, so the caller reads the engine's refusal of it and no warning."""
    expected = engine_refusal(dim=8, expected_size=expected_size)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as refusal:
            VectorDatabase().create("hnsw", dim=8, expected_size=expected_size,
                                    quantization_config=dict(config))
    assert (refusal.type, str(refusal.value)) == expected
    assert [str(w.message) for w in caught] == []


# ------------------------------------------------------------
# An argument of the wrong type raises TypeError
# ------------------------------------------------------------
@pytest.mark.parametrize("kwargs, words", [
    ({"indexed_fields": 5}, "indexed_fields must be a list of metadata field names, got int."),
    ({"indexed_fields": [5]},
     "indexed_fields must hold field names as strings, but it holds 5, of type int."),
    ({"indexed_fields": "lang"},
     "indexed_fields must be a list of field names, not a single string. Pass "
     "indexed_fields=['lang'] to declare one field."),
    ({"quantization_config": 5}, "quantization_config must be a dictionary"),
    ({"quantization_config": {"type": "pq", 1: 2}},
     "quantization_config must have str keys, but it holds the key 1, of type int."),
    ({"quantization_config": {"type": "int8", ("a",): 2}},
     "quantization_config must have str keys, but it holds the key ('a',), of type tuple."),
], ids=["fields-int", "fields-holding-an-int", "fields-str", "config-int", "pq-key-int",
        "int8-key-tuple"])
def test_an_argument_of_the_wrong_type_raises_type_error(kwargs, words):
    """A parameter of the wrong type raises TypeError, as the docstring says:
    a field list that is not a list of str, a quantization config that is
    not a dict and a config key that is not a str."""
    with pytest.raises(TypeError) as refusal:
        VectorDatabase().create("hnsw", dim=8, **kwargs)
    assert str(refusal.value) == words


# ------------------------------------------------------------
# A quantization count that is not an int names its type
# ------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["pq", "int8"])
@pytest.mark.parametrize("key, value", [("training_size", "2000"), ("training_size", 2000.0),
                                        ("max_training_vectors", "5000")],
                         ids=["training_size-str", "training_size-float",
                              "max_training_vectors-str"])
def test_a_quantization_count_that_is_not_an_int_names_its_type(scheme, key, value):
    """A training_size or max_training_vectors that is not an int raises
    ValueError naming the value and its type, under both schemes."""
    config = {"type": scheme, key: value}
    if scheme == "pq":
        config["subvectors"] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as refusal:
            VectorDatabase().create("hnsw", dim=8, quantization_config=config)
    assert str(refusal.value) == (
        f"{key} must be an integer, got {value!r}, of type {type(value).__name__}.")


# ------------------------------------------------------------
# None for a quantization key is the key left out
# ------------------------------------------------------------
@pytest.mark.parametrize("config", [
    {"type": "pq", "subvectors": 2, "bits": None},
    {"type": "pq", "subvectors": 2, "storage_mode": None},
    {"type": "int8", "storage_mode": None},
], ids=["pq-bits", "pq-storage_mode", "int8-storage_mode"])
def test_none_for_a_quantization_key_with_a_default_is_that_default(config):
    """None for bits or storage_mode is the key left out, as it is for every
    other key of a quantization config, so the index takes the default."""
    without = {key: value for key, value in config.items() if value is not None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        given = VectorDatabase().create("hnsw", dim=8, quantization_config=config)
        left_out = VectorDatabase().create("hnsw", dim=8, quantization_config=without)
    assert given.get_quantization_info() == left_out.get_quantization_info()


# ------------------------------------------------------------
# A quantization key the scheme does not read is refused
# ------------------------------------------------------------
@pytest.mark.parametrize("config, words", [
    ({"type": "pq", "subvector": 4},
     "quantization_config holds the key 'subvector', which type 'pq' does not read. Type 'pq' "
     "reads type, subvectors, bits, training_size, max_training_vectors and storage_mode."),
    ({"type": "int8", "trainingsize": 2000},
     "quantization_config holds the key 'trainingsize', which type 'int8' does not read. Type "
     "'int8' reads type, scale, training_size, max_training_vectors and storage_mode."),
], ids=["pq", "int8"])
def test_a_quantization_key_the_scheme_does_not_read_is_refused(config, words):
    """The engine reads the keys a scheme names alone, so a key the scheme
    does not read would change nothing, and create() raises ValueError
    naming it and the keys the scheme reads."""
    with pytest.raises(ValueError) as refusal:
        VectorDatabase().create("hnsw", dim=8, quantization_config=config)
    assert str(refusal.value) == words
