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
library cannot write them. A directory this build saves holds `mappings.bin`,
`vectors.bin`, `pq_codes.bin` and `pq_centroids.bin` inside a frame, so a forged
count or width is written into the payload and the frame is built again around
it, which carries the forgery past both of its checksums to the payload's
reader. A directory a release before 4.0.0 saved holds the same four in
bincode's wire, where a container length is a varint this file encodes itself,
and those forgeries run on a copy of a saved directory rewritten in that wire.
`config.json` and `quantization.json` are JSON and are edited in place.
"""

import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest
from helpers import FRAME_KINDS, artefact_digest, as_old_wire, frame, reframe, repair_manifest, unframe
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


@pytest.fixture(scope="module")
def old_raw_index(raw_index, tmp_path_factory):
    """The raw index as a release before 4.0.0 wrote it, at 1.1.0."""
    path = tmp_path_factory.mktemp("old-raw") / "index"
    shutil.copytree(raw_index, path)
    as_old_wire(path, "1.1.0")
    return path


@pytest.fixture(scope="module")
def old_quantized_index(quantized_index, tmp_path_factory):
    """The quantized index as a release before 4.0.0 wrote it, at 1.1.0."""
    path = tmp_path_factory.mktemp("old-quantized") / "index"
    shutil.copytree(quantized_index, path)
    as_old_wire(path, "1.1.0")
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


def test_the_digest_repairer_agrees_with_the_saved_manifest(
    raw_index, quantized_index, old_raw_index, old_quantized_index
):
    """Every digest a real save wrote, recomputed here from the file, and every
    framed artefact it wrote, framed again here from its payload.

    Without this the repairs above could be silently wrong, and every case in
    this file would then be asserting that a wrong digest or a wrong checksum
    is refused rather than that the field validator it names does its job.
    """
    checked = framed = 0
    for directory in (raw_index, quantized_index, old_raw_index, old_quantized_index):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        digests = manifest["file_digests"]
        assert digests, "a save records a length per artefact"
        for name, entry in digests.items():
            data = (directory / name).read_bytes()
            assert entry["bytes"] == len(data), name
            if "checksum" in entry:
                assert entry["checksum"] == artefact_digest(data), name
                checked += 1
            if manifest["format_version"] == "4.0.0" and name in FRAME_KINDS:
                kind, entries, payload = unframe(data)
                assert kind == FRAME_KINDS[name], name
                assert frame(kind, entries, payload) == data, name
                framed += 1
    assert checked >= 16, f"only {checked} digests were compared"
    assert framed >= 6, f"only {framed} frames were compared"


def test_a_forged_file_whose_digest_is_not_repaired_is_refused(old_raw_index, tmp_path):
    """The digest check itself, which every forgery of an old directory has to
    get past."""
    target = tmp_path / "unrepaired"
    shutil.copytree(old_raw_index, target)
    raw = bytearray((target / "vectors.bin").read_bytes())
    raw[-4] ^= 0x40
    (target / "vectors.bin").write_bytes(bytes(raw))

    status, verdict, _ = load_in_child(target, tmp_path)
    assert status == 0
    assert verdict.startswith("REFUSED"), verdict
    assert "vectors.bin" in verdict and "digest" in verdict, verdict


def test_a_forged_frame_whose_checksum_is_not_repaired_is_refused(raw_index, tmp_path):
    """The frame's payload checksum itself, which every forgery of a framed
    artefact has to get past. The manifest records a framed artefact by its
    length alone, so it is the frame that refuses."""
    target = tmp_path / "unrepaired"
    shutil.copytree(raw_index, target)
    raw = bytearray((target / "vectors.bin").read_bytes())
    raw[70] ^= 0x40
    (target / "vectors.bin").write_bytes(bytes(raw))

    status, verdict, _ = load_in_child(target, tmp_path)
    assert status == 0
    assert verdict.startswith("REFUSED"), verdict
    assert "vectors.bin: the frame's payload is corrupt" in verdict, verdict


def test_an_untouched_directory_still_loads(raw_index, old_raw_index, tmp_path):
    """Nothing below means anything if the unforged directory does not load."""
    for directory in (raw_index, old_raw_index):
        status, verdict, _ = load_in_child(directory, tmp_path)
        assert status == 0
        assert verdict == "LOADED 2", verdict


def test_an_untouched_quantized_directory_still_loads(quantized_index, old_quantized_index, tmp_path):
    for directory in (quantized_index, old_quantized_index):
        status, verdict, _ = load_in_child(directory, tmp_path)
        assert status == 0
        assert verdict == "LOADED 1050", verdict


# ============================================================================
# THE FOUR ARTEFACTS IN THE OLD WIRE
# ============================================================================
#
# Four files, ten container lengths, in a directory a release before 4.0.0
# wrote. The reader of that wire holds every length to the bytes the file has
# left, at the fewest bytes one entry can occupy, before anything is sized from
# it.


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
def test_a_forged_length_in_a_raw_artefact_is_refused(old_raw_index, tmp_path, name, payload):
    path = forged(old_raw_index, tmp_path, name, write_bytes(payload))
    assert_refused(path, tmp_path, naming=name)


def wire_vector(values):
    return varint(len(values)) + struct.pack(f"<{len(values)}f", *values)


def wire_record(ident, values):
    return wire_str(ident) + wire_vector(values)


_A = [1.0, 0.0, 0.0, 0.0]
_B = [0.0, 1.0, 0.0, 0.0]
_A_POISONED = [float("nan"), 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "payload,opens",
    [
        (varint(3) + wire_record("a", _A) + wire_record("b", _B) + wire_record("a", _A), True),
        (varint(3) + wire_record("a", _A_POISONED) + wire_record("b", _B) + wire_record("a", _A), True),
        (varint(3) + wire_record("a", _A) + wire_record("b", _B) + wire_record("a", _A_POISONED), False),
    ],
    ids=["twice-clean", "first-copy-poisoned-last-clean", "last-copy-poisoned"],
)
def test_a_vectors_bin_holding_one_id_twice_is_read_as_a_map_holds_it(
    old_raw_index, tmp_path, payload, opens
):
    """No save writes an id twice, since the map it encodes cannot hold one,
    so this is a file a hand made. Every release read the file into a map,
    which held one entry for the id, the last copy, and counted it once. The
    loader walks the file without building the map and gives the same
    answer: two records, and a refusal only when the copy that came last is
    the one that is not finite, naming the id once. A framed file names each
    record by internal id in increasing order, so it cannot hold one twice.
    """
    path = forged(old_raw_index, tmp_path, "vectors.bin", write_bytes(payload))
    status, verdict, _ = load_in_child(path, tmp_path)
    assert status == 0
    if opens:
        assert verdict == "LOADED 2", verdict
    else:
        assert verdict.startswith("REFUSED"), verdict
        assert "in 1 of 2 records" in verdict, verdict
        assert verdict.endswith("Affected records include: a"), verdict


# The forward map, then a reverse map that is not its inverse. The loader
# builds one id store from the forward map and holds the reverse map to being
# the store's exact inverse, since the two were written from one structure.
_TWO = varint(2) + wire_str("a") + varint(1) + wire_str("b") + varint(2)


@pytest.mark.parametrize(
    "payload",
    [
        _TWO + varint(2) + varint(1) + wire_str("a") + varint(2) + wire_str("c"),
        _TWO + varint(1) + varint(1) + wire_str("a"),
        varint(2) + wire_str("a") + varint(1) + wire_str("b") + varint(1) + varint(1) + varint(1) + wire_str("b"),
    ],
    ids=["reverse-names-another-id", "reverse-shorter", "two-names-one-internal-id"],
)
def test_a_mappings_file_whose_two_maps_disagree_is_refused(old_raw_index, tmp_path, payload):
    """A mappings.bin whose reverse map is not the inverse of its forward map
    describes two record sets, and the loader refuses it naming the file. The
    two maps used to be installed as they were read, and such a file loaded
    with `len` and `search` answering from different sets. A framed file holds
    one list, so it has no second map to disagree."""
    path = forged(old_raw_index, tmp_path, "mappings.bin", write_bytes(payload))
    assert_refused(path, tmp_path, naming="mappings.bin")


# One record at an internal id above the counter config.json records, with
# both maps consistent. The id store, the metadata and the columns hold an
# entry for every id up to the highest a record holds, so the loader reserved
# and filled one for every id below it before anything refused the file, and
# at 2^40 asked the allocator for eight tebibytes.
@pytest.mark.parametrize("slot", [3, 1 << 24, 1 << 40], ids=["one-above", "2^24", "2^40"])
def test_a_mappings_id_above_the_counter_is_refused(old_raw_index, tmp_path, slot):
    payload = (varint(2) + wire_str("a") + varint(1) + wire_str("b") + varint(slot)
               + varint(2) + varint(1) + wire_str("a") + varint(slot) + wire_str("b"))
    path = forged(old_raw_index, tmp_path, "mappings.bin", write_bytes(payload))
    assert_refused(
        path, tmp_path,
        naming=f"the forward map names internal id {slot} for 'b' and config.json counted 2",
    )


@pytest.mark.parametrize("slot", [3, 1 << 24, (1 << 32) - 1], ids=["one-above", "2^24", "u32-max"])
def test_a_framed_mappings_id_above_the_counter_is_refused(raw_index, tmp_path, slot):
    """The same counter, held in a framed file, whose internal ids are four
    bytes wide, so the highest it can name is the last a u32 holds."""
    payload = (struct.pack("<2I", 1, slot) + struct.pack("<2I", 1, 1) + b"ab")
    path = forged(raw_index, tmp_path, "mappings.bin", write_bytes(frame(1, 2, payload)))
    assert_refused(
        path, tmp_path,
        naming=f"the record 'b' holds internal id {slot} and config.json counted 2",
    )


def test_a_directory_whose_ids_are_sparse_still_opens(tmp_path):
    """Removals without a compaction leave the held ids sparse, and the
    highest removed, so the counter config.json records is above every id the
    mappings name. Such a directory opens and issues ids after its counter."""
    index = VectorDatabase().create("hnsw", dim=4, expected_size=64)
    vectors = np.random.default_rng(5).standard_normal((40, 4)).astype(np.float32)
    index.add({"ids": [f"r{i}" for i in range(40)], "embeddings": vectors})
    removed = [f"r{i}" for i in range(40) if i % 9 != 0]
    assert index.remove_points(removed) == [], "every id was held"
    assert len(index) == 5
    path = tmp_path / "sparse.zdb"
    index.save(str(path))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["id_counter"] == 40

    loaded = VectorDatabase().load(str(path))
    assert sorted(id for id, _ in loaded.list(number=100)) == ["r0", "r18", "r27", "r36", "r9"]
    assert loaded.add({"id": "late", "values": [0.5, 0.5, 0.5, 0.5]}).is_success()
    loaded.save(str(path))
    assert json.loads((path / "config.json").read_text(encoding="utf-8"))["id_counter"] == 41


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
    old_quantized_index, tmp_path, name, payload
):
    path = forged(old_quantized_index, tmp_path, name, write_bytes(payload))
    assert_refused(path, tmp_path, naming=name)


def test_the_densest_file_this_build_writes_still_opens(tmp_path):
    """A bound derived from a file's length has to still admit a real file.

    The densest legitimate case is short ids and narrow vectors, which is where
    an entry's bytes are fewest. A bound that refused this would have made the
    bound a regression rather than a fix, in either layout.
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

    # The codebook here is two floats inside its frame, which is the smallest
    # file the loader is ever asked about.
    assert (path / "pq_centroids.bin").stat().st_size < 200
    assert VectorDatabase().load(str(path)).get_vector_count() == count
    old = tmp_path / "dense-old"
    shutil.copytree(path, old)
    as_old_wire(old, "1.1.0")
    assert VectorDatabase().load(str(old)).get_vector_count() == count



# ============================================================================
# A LENGTH ITS FILE CANNOT CARRY
# ============================================================================
#
# Each file below is a little over 4 MiB of the old wire, and one container in
# it declares far more entries than its bytes could hold. The loader refuses
# each before anything is sized from it, and the child refuses rather than
# dying.

FILE_BYTES = (1 << 22) + 64
DECLARED_BYTES = 1 << 36


def declaring_too_many(entry_bytes):
    """A file whose outermost container declares `DECLARED_BYTES` worth of
    entries of `entry_bytes` each. The padding behind it is never read."""
    head = varint((DECLARED_BYTES - 4096) // entry_bytes)
    return head + bytes(FILE_BYTES - len(head))


def one_record_in_declaring_too_many(entry_bytes):
    """The same, declared by the first record's own container."""
    head = varint(1) + wire_str("a") + varint((DECLARED_BYTES - 4096) // entry_bytes)
    return head + bytes(FILE_BYTES - len(head))


@pytest.mark.parametrize(
    "fixture,name,payload",
    [
        # The forward map's entries.
        ("raw", "mappings.bin", declaring_too_many(32)),
        # The vector map's entries, on the walk a raw index takes and on the
        # map a quantized one reads.
        ("raw", "vectors.bin", declaring_too_many(48)),
        ("quantized", "vectors.bin", declaring_too_many(48)),
        ("quantized", "pq_codes.bin", declaring_too_many(48)),
        # The codebook's subvectors.
        ("quantized", "pq_centroids.bin", declaring_too_many(24)),
        # One record in: the vector's own length, and the code's.
        ("raw", "vectors.bin", one_record_in_declaring_too_many(4)),
        ("quantized", "vectors.bin", one_record_in_declaring_too_many(4)),
        ("quantized", "pq_codes.bin", one_record_in_declaring_too_many(1)),
    ],
    ids=[
        "mappings-id_map", "vectors-map-walked", "vectors-map-decoded",
        "codes-map", "centroids-subvector-count",
        "vectors-record-vector-walked", "vectors-record-vector-decoded",
        "codes-record-code",
    ],
)
def test_a_four_mebibyte_file_naming_a_length_it_cannot_carry_is_refused(
    old_raw_index, old_quantized_index, tmp_path, fixture, name, payload
):
    source = old_raw_index if fixture == "raw" else old_quantized_index
    path = forged(source, tmp_path, name, write_bytes(payload))
    assert_refused(path, tmp_path, naming=name)


# ============================================================================
# THE FOUR ARTEFACTS IN THE FRAME
# ============================================================================
#
# A frame holds a payload to the length its header declares, and a payload
# holds every count and width to the bytes it has. Each forgery below writes a
# hostile count or width into a well formed payload and frames it again, so
# both checksums verify and the payload's own reader is what refuses.

def edit_payload(edit):
    """A forgery that rewrites a framed artefact's payload or entry count."""
    def mutate(path):
        kind, entries, payload = unframe(path.read_bytes())
        payload, entries = edit(bytearray(payload), entries)
        path.write_bytes(frame(kind, entries, bytes(payload)))
    return mutate


def set_u32(at, value):
    def edit(payload, entries):
        payload[at:at + 4] = struct.pack("<I", value)
        return payload, entries
    return edit


def set_entries(value):
    return lambda payload, entries: (payload, value)


@pytest.mark.parametrize(
    "fixture,name,edit,naming",
    [
        ("raw", "mappings.bin", set_entries(HUGE), "records take at least"),
        ("raw", "mappings.bin", set_entries((1 << 64) - 1), "records take at least"),
        ("raw", "mappings.bin", set_u32(8, 0xFFFFFFFF), "the ids' lengths sum to"),
        ("raw", "mappings.bin", set_u32(4, 0), "and the ids are strictly increasing"),
        ("raw", "vectors.bin", set_entries(HUGE), "rows of"),
        ("raw", "vectors.bin", set_u32(0, 0xFFFFFFFF), "and config.json declares dim 4"),
        ("raw", "vectors.bin", set_u32(24, 0xFFFFFFF0), "which mappings.bin does not hold"),
        ("quantized", "vectors.bin", set_entries(HUGE), "rows of"),
        ("quantized", "pq_codes.bin", set_entries(HUGE), "rows of"),
        ("quantized", "pq_codes.bin", set_u32(0, 0xFFFFFFFF), "quantization.json declares 4 subvectors"),
        ("quantized", "pq_centroids.bin", set_entries(HUGE), "a codebook of"),
        ("quantized", "pq_centroids.bin", set_u32(0, 0xFFFFFFFF), "a codebook of"),
        ("quantized", "pq_centroids.bin", set_u32(4, 0xFFFFFFFF), "a codebook of"),
    ],
    ids=[
        "mappings-entries-2^40", "mappings-entries-u64-max", "mappings-id-length",
        "mappings-ids-not-increasing", "vectors-entries-raw", "vectors-width",
        "vectors-foreign-id", "vectors-entries-quantized", "codes-entries", "codes-width",
        "centroids-subvectors", "centroids-count", "centroids-width",
    ],
)
def test_a_forged_count_in_a_framed_artefact_is_refused(
    raw_index, quantized_index, tmp_path, fixture, name, edit, naming
):
    source = raw_index if fixture == "raw" else quantized_index
    path = forged(source, tmp_path, name, edit_payload(edit))
    assert_refused(path, tmp_path, naming=naming)


def test_a_framed_codebook_of_no_values_naming_a_huge_count_is_refused(quantized_index, tmp_path):
    """A codebook whose centroids hold no value takes no bytes however many
    subvectors it names, so its bytes agree with a count of 2^40. Its shape is
    held to quantization.json before anything is sized from it."""
    payload = struct.pack("<2I", 0, 0)
    path = forged(quantized_index, tmp_path, "pq_centroids.bin", write_bytes(frame(4, HUGE, payload)))
    assert_refused(path, tmp_path, naming="codebook is 1099511627776x0x0, expected 4x16x2")

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
