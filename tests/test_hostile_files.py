"""Every field of a saved index that sizes an allocation, given a hostile value.

An allocation sized from a field the file has not earned does not raise. It
aborts, and an abort does not unwind, so no `catch_unwind` sees it and a Python
caller gets a dead interpreter with no traceback. That is why every case here
runs the load in a **subprocess** and asserts on the child's exit status as well
as on its message, because an in-process test cannot tell a refusal from a death.

The rule these tests hold is the one `parse_dump` already draws. A length the
file's own bytes could carry is a limit, and a length nothing bounds is a
defect.

The forged files are written by hand rather than by the library, because the
library cannot write them. `mappings.bin`, `vectors.bin`, `pq_codes.bin` and
`pq_centroids.bin` are bincode, so the container length is a varint this file
encodes itself; `config.json` and `quantization.json` are JSON and are edited in
place.
"""

import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest
from helpers import artefact_digest, repair_manifest
from zeusdb_vector_database import VectorDatabase

# A length no file could carry and every unbounded container aborted on.
HUGE = 1 << 40

LOAD_TIMEOUT_S = 120


# ============================================================================
# BINCODE ON THE WIRE
# ============================================================================
#
# `bincode::config::standard()` writes lengths as a varint: a value up to 250 is
# one byte, and above that a marker byte names the width that follows. Only the
# marker for a u64 is needed here, since every forged length is 2^40.


def varint(value):
    if value <= 250:
        return bytes([value])
    if value <= 0xFFFF:
        return b"\xfb" + struct.pack("<H", value)
    if value <= 0xFFFFFFFF:
        return b"\xfc" + struct.pack("<I", value)
    return b"\xfd" + struct.pack("<Q", value)


def wire_str(text):
    raw = text.encode("utf-8")
    return varint(len(raw)) + raw


# ============================================================================
# THE DIRECTORIES UNDER TEST
# ============================================================================


@pytest.fixture(scope="module")
def raw_index(tmp_path_factory):
    """A small unquantized index, saved."""
    path = tmp_path_factory.mktemp("raw") / "index"
    index = VectorDatabase().create("hnsw", dim=4, expected_size=64)
    index.add({"ids": ["a", "b"], "embeddings": [[1.0, 0, 0, 0], [0, 1.0, 0, 0]]})
    index.save(str(path))
    return path


@pytest.fixture(scope="module")
def quantized_index(tmp_path_factory):
    """A trained `quantized_with_raw` index, saved.

    Trained, because `pq_centroids.bin` and `pq_codes.bin` are only written and
    only read once a codebook exists.
    """
    path = tmp_path_factory.mktemp("quantized") / "index"
    index = VectorDatabase().create(
        "hnsw", dim=8, expected_size=4000,
        quantization_config={
            "type": "pq", "subvectors": 4, "bits": 4,
            "training_size": 1000, "storage_mode": "quantized_with_raw",
        },
    )
    vectors = np.random.default_rng(3).standard_normal((1050, 8)).astype(np.float32)
    index.add({"ids": [f"r{i}" for i in range(1050)], "embeddings": vectors})
    assert index.is_quantized(), "the fixture must train, or the codebook files are absent"
    index.save(str(path))
    return path


CHILD = """
import sys, warnings
warnings.simplefilter("ignore")
from zeusdb_vector_database import VectorDatabase
try:
    index = VectorDatabase().load(sys.argv[1])
    print("LOADED", len(index))
except BaseException as exc:
    print("REFUSED", type(exc).__name__, str(exc).replace(chr(10), " "))
"""


def load_in_child(path, tmp_path):
    """Load a directory in a subprocess and report how the child ended.

    Returns the child's exit status and the one line it prints. A death shows up
    as a non-zero status with no line, which is what every case here produced
    before the bounds existed.
    """
    script = tmp_path / "load_probe.py"
    script.write_text(CHILD, encoding="utf-8")
    # save() and load() print a progress banner carrying non-ASCII, which the
    # default child encoding on Windows cannot decode.
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=child_env, timeout=LOAD_TIMEOUT_S,
    )
    verdicts = [
        line for line in result.stdout.splitlines()
        if line.startswith(("LOADED", "REFUSED"))
    ]
    return result.returncode, (verdicts[-1] if verdicts else ""), result


# ============================================================================
# REPAIRING THE MANIFEST DIGEST
# ============================================================================
#
# manifest.json records a length and a digest for every artefact it names, and
# the loader checks both before anything parses the file. A forged artefact
# therefore stops at the digest and never reaches the field validator the case
# is about, so every forge below repairs the entry it broke.
#
# This is the same rule the graph dump fuzzer follows. A mutator that cannot
# repair a digest proves the digest works and reaches no parsing code.
#
# The repair lives in `helpers.py` and computes the digest from the format
# rather than calling the library, and the test below holds it against every
# digest a real save wrote. Two implementations agreeing is what makes the
# repair trustworthy.

def forged(source, tmp_path, name, mutate):
    """A copy of a saved directory with one file replaced or edited."""
    target = tmp_path / "forged"
    shutil.copytree(source, target)
    mutate(target / name)
    repair_manifest(target, name)
    return target


def write_bytes(data):
    return lambda path: path.write_bytes(data)


def edit_json(**fields):
    def mutate(path):
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(fields)
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return mutate


def assert_refused(path, tmp_path, *, naming):
    """The child refused, survived, and said which field it refused on."""
    status, verdict, result = load_in_child(path, tmp_path)
    assert status == 0, (
        "the child did not survive the load, which is what an allocation sized "
        "from an unearned field does: it aborts rather than raising, so nothing "
        f"in the process can catch it. Exit status {status}.\n"
        + result.stdout[-2000:] + result.stderr[-2000:]
    )
    assert verdict.startswith("REFUSED"), f"the load was not refused: {verdict}"
    assert naming in verdict, f"the refusal does not name {naming!r}: {verdict}"


# ============================================================================
# THE BASELINE
# ============================================================================


def test_the_digest_repairer_agrees_with_the_saved_manifest(raw_index, quantized_index):
    """Every digest a real save wrote, recomputed here from the file.

    Without this the repair above could be silently wrong, and every case in
    this file would then be asserting that a wrong digest is refused rather than
    that the field validator it names does its job.
    """
    checked = 0
    for directory in (raw_index, quantized_index):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        digests = manifest["file_digests"]
        assert digests, "a save records a digest per artefact"
        for name, entry in digests.items():
            data = (directory / name).read_bytes()
            assert entry["bytes"] == len(data), name
            if "checksum" in entry:
                assert entry["checksum"] == artefact_digest(data), name
                checked += 1
    assert checked >= 8, f"only {checked} digests were compared"


def test_a_forged_file_whose_digest_is_not_repaired_is_refused(raw_index, tmp_path):
    """The digest check itself, which every other case here has to get past."""
    target = tmp_path / "unrepaired"
    shutil.copytree(raw_index, target)
    raw = bytearray((target / "vectors.bin").read_bytes())
    raw[-4] ^= 0x40
    (target / "vectors.bin").write_bytes(bytes(raw))

    status, verdict, _ = load_in_child(target, tmp_path)
    assert status == 0
    assert verdict.startswith("REFUSED"), verdict
    assert "vectors.bin" in verdict and "digest" in verdict, verdict


def test_an_untouched_directory_still_loads(raw_index, tmp_path):
    """Nothing below means anything if the unforged directory does not load."""
    status, verdict, _ = load_in_child(raw_index, tmp_path)
    assert status == 0
    assert verdict == "LOADED 2", verdict


def test_an_untouched_quantized_directory_still_loads(quantized_index, tmp_path):
    status, verdict, _ = load_in_child(quantized_index, tmp_path)
    assert status == 0
    assert verdict == "LOADED 1050", verdict


# ============================================================================
# THE BINCODE ARTEFACTS
# ============================================================================
#
# Four files, ten container lengths. `bincode::config::standard()` carries no
# byte limit, so `claim_container_read` compiled to nothing and every one of
# these went straight to the allocator. The bound is a claim budget derived from
# the file's own length; see `CLAIM_PER_WIRE_BYTE`.


@pytest.mark.parametrize(
    "name,payload",
    [
        # HashMap<String, usize>, then the key's own Vec<u8>, then the second map.
        ("mappings.bin", varint(HUGE)),
        ("mappings.bin", varint(1) + varint(HUGE) + b"x"),
        ("mappings.bin", varint(0) + varint(HUGE)),
        # HashMap<String, Vec<f32>>: the map, then one record's vector.
        ("vectors.bin", varint(HUGE)),
        ("vectors.bin", varint(1) + wire_str("a") + varint(HUGE)),
    ],
    ids=[
        "mappings-id_map-length", "mappings-key-length", "mappings-rev_map-length",
        "vectors-map-length", "vectors-record-vector-length",
    ],
)
def test_a_forged_length_in_a_raw_artefact_is_refused(raw_index, tmp_path, name, payload):
    path = forged(raw_index, tmp_path, name, write_bytes(payload))
    assert_refused(path, tmp_path, naming=name)


@pytest.mark.parametrize(
    "name,payload",
    [
        # HashMap<String, Vec<u8>>: the map, then one record's code.
        ("pq_codes.bin", varint(HUGE)),
        ("pq_codes.bin", varint(1) + wire_str("r0") + varint(HUGE)),
        # Vec<Vec<Vec<f32>>>: all three nesting levels.
        ("pq_centroids.bin", varint(HUGE)),
        ("pq_centroids.bin", varint(1) + varint(HUGE)),
        ("pq_centroids.bin", varint(1) + varint(1) + varint(HUGE)),
    ],
    ids=[
        "codes-map-length", "codes-record-code-length",
        "centroids-subvector-count", "centroids-centroid-count", "centroids-width",
    ],
)
def test_a_forged_length_in_a_quantized_artefact_is_refused(
    quantized_index, tmp_path, name, payload
):
    path = forged(quantized_index, tmp_path, name, write_bytes(payload))
    assert_refused(path, tmp_path, naming=name)


def test_the_claim_budget_admits_the_densest_file_this_build_writes(tmp_path):
    """A budget derived from a file's length has to still admit a real file.

    The densest legitimate case is short ids and narrow vectors, which is where
    the ratio of bytes claimed to bytes on the wire is largest. A budget that
    refused this would have made the bound a regression rather than a fix.
    """
    count = 2000
    ids = [f"{i:04d}" for i in range(count)]
    vectors = np.random.default_rng(5).standard_normal((count, 2)).astype(np.float32)
    index = VectorDatabase().create(
        "hnsw", dim=2, expected_size=count,
        quantization_config={
            "type": "pq", "subvectors": 1, "bits": 1,
            "training_size": 1000, "storage_mode": "quantized_only",
        },
    )
    index.add({"ids": ids, "embeddings": vectors})
    assert index.is_quantized()
    path = tmp_path / "dense"
    index.save(str(path))

    # The codebook here is twenty bytes, which is the smallest file the budget
    # is ever asked about.
    assert (path / "pq_centroids.bin").stat().st_size < 200
    assert VectorDatabase().load(str(path)).get_vector_count() == count


# ============================================================================
# quantization.json
# ============================================================================
#
# `PQ::new` allocates `subvectors * 2^bits * (dim / subvectors)` floats from two
# fields the loader never revalidated, though `create()` refuses all three of
# the values below.


@pytest.mark.parametrize(
    "fields,naming",
    [
        ({"bits": 40}, "bits is 40"),
        ({"bits": 64}, "bits is 64"),
        ({"bits": 0}, "bits is 0"),
        ({"subvectors": HUGE}, "subvectors is 1099511627776"),
        ({"subvectors": 0}, "subvectors is 0"),
        ({"subvectors": 3}, "subvectors is 3"),
    ],
    ids=["bits-40", "bits-64", "bits-0", "subvectors-huge", "subvectors-zero",
         "subvectors-does-not-divide"],
)
def test_a_hostile_quantization_field_is_refused(quantized_index, tmp_path, fields, naming):
    """Both sizing fields, at every value that used to abort or panic.

    `bits: 40` asked for 2^40 centroids and aborted. `bits: 64` shifted a usize
    by its own width, which masks to one rather than aborting, and came back as
    a codebook of a single centroid that only a later shape check happened to
    catch. `subvectors: 0` divided by zero.
    """
    path = forged(quantized_index, tmp_path, "quantization.json", edit_json(**fields))
    assert_refused(path, tmp_path, naming=naming)


# ============================================================================
# config.json
# ============================================================================


@pytest.mark.parametrize(
    "fields,naming",
    [
        ({"dim": HUGE}, "dim must be at most 65536, got 1099511627776"),
        ({"dim": 1 << 31}, "dim must be at most 65536, got 2147483648"),
        ({"id_counter": HUGE}, "id_counter is 1099511627776"),
        ({"ef_construction": HUGE}, "ef_construction must be at most 4096, got 1099511627776"),
    ],
    ids=["dim-huge", "dim-2^31", "id_counter-huge", "ef_construction-huge"],
)
def test_a_hostile_config_field_is_refused(raw_index, tmp_path, fields, naming):
    """`dim` sizes one vector buffer and had no upper bound at all.

    `id_counter` was bounded earlier and is held here so the pair stays
    together: they are the two fields of config.json that size an
    allocation rather than describe a behaviour. `ef_construction` joined
    them last. It sizes the candidate heaps of every insertion rather than
    anything the loader allocates, so a directory naming 2**40 would load,
    restore its graph from the dump, and kill the process on the first add()
    after the load. The loader is the third door to the validator that
    bounds it, and the case sits here with the other two config.json fields
    because the subprocess costs nothing and keeps the file uniform.
    """
    path = forged(raw_index, tmp_path, "config.json", edit_json(**fields))
    assert_refused(path, tmp_path, naming=naming)


CREATE_CHILD = """
import sys, warnings
warnings.simplefilter("ignore")
from zeusdb_vector_database import VectorDatabase
try:
    index = VectorDatabase().create("hnsw", dim=int(sys.argv[1]), expected_size=4)
    print("CREATED", index.dim)
except BaseException as exc:
    print("REFUSED", type(exc).__name__, str(exc).replace(chr(10), " "))
"""


def create_in_child(dim, tmp_path):
    """Create an index in a subprocess and report how the child ended."""
    script = tmp_path / "create_probe.py"
    script.write_text(CREATE_CHILD, encoding="utf-8")
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), str(dim)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=child_env, timeout=LOAD_TIMEOUT_S,
    )
    verdicts = [
        line for line in result.stdout.splitlines()
        if line.startswith(("CREATED", "REFUSED"))
    ]
    return result.returncode, (verdicts[-1] if verdicts else ""), result


@pytest.mark.parametrize(
    "dim",
    [HUGE, 1 << 31, 65_537],
    ids=["dim-huge", "dim-2^31", "dim-one-over"],
)
def test_creating_at_a_hostile_dim_is_refused(dim, tmp_path):
    """`create()` had no upper bound on `dim` at all.

    `dim` sizes one vector buffer, which is the first allocation creation makes,
    so `create(dim=2**40)` asked the allocator for 4,398,046,511,104 bytes and
    killed the interpreter with exit status 3221226505. Every other creation
    parameter that sizes an allocation was already bounded, and the loader
    bounded this one, so creation was the last door that admitted it.

    It runs in a subprocess for the reason every case in this file does: an
    abort is not an exception, and an in-process test cannot tell a refusal from
    a death.
    """
    status, verdict, result = create_in_child(dim, tmp_path)
    assert status == 0, (
        "the child did not survive the create, which is what an allocation "
        f"sized from an unbounded dim does. Exit status {status}." \
        + result.stdout[-2000:] + result.stderr[-2000:]
    )
    assert verdict.startswith("REFUSED"), f"the create was not refused: {verdict}"
    assert f"dim must be at most 65536, got {dim}" in verdict, verdict


def test_creating_at_the_dim_ceiling_still_works(tmp_path):
    """The bound admits the value it names, so the ceiling is inclusive."""
    status, verdict, _ = create_in_child(65_536, tmp_path)
    assert status == 0
    assert verdict == "CREATED 65536", verdict


def test_the_dim_ceiling_admits_the_widest_real_embedding(tmp_path):
    """The bound has to be above every width a model produces.

    3072 is the widest OpenAI embedding and the ceiling is 65,536, so this is
    a long way inside it. The test exists because a ceiling chosen too low
    would fail silently on a save nobody in this suite makes.
    """
    index = VectorDatabase().create("hnsw", dim=3072, expected_size=16)
    vector = np.random.default_rng(9).standard_normal(3072).astype(np.float32)
    index.add({"ids": ["wide"], "embeddings": [vector.tolist()]})
    path = tmp_path / "wide"
    index.save(str(path))
    assert VectorDatabase().load(str(path)).dim == 3072
