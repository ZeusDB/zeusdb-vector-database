"""Shared assertion helpers for the vector database tests."""


def normalize_vector(vector):
    """Normalize vector for cosine distance (same as Rust implementation)"""
    import math
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0.0:
        return [x / norm for x in vector]
    return vector

def assert_vectors_close(actual, expected, tolerance=1e-6, space="cosine"):
    """Assert vectors are close, accounting for normalization"""
    if space.lower() == "cosine":
        expected = normalize_vector(expected)
    
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) < tolerance, f"Vector element {i}: expected {e}, got {a}"


# ============================================================================
# REPAIRING A MANIFEST DIGEST AFTER AN EDIT
# ============================================================================
#
# manifest.json records a length and a digest for every artefact it names and
# the loader checks both before anything parses the file. A test that edits a
# saved artefact to exercise a validator therefore has to record what the file
# now holds, or the load stops at the digest and the validator never runs.
#
# The checksum below is a second implementation of the one the crate writes,
# written from the format rather than called out of the library.
# `test_hostile_files.py::test_the_digest_repairer_agrees_with_the_saved_manifest`
# holds it against every digest a real save wrote.

_MASK64 = (1 << 64) - 1
_SEED = 0xCBF29CE484222325
_PRIME = 0x100000001B3
_AVALANCHE = 0xFF51AFD7ED558CCD


def artefact_digest(data):
    """The 64 bit checksum a save records for an artefact, as sixteen hex digits."""
    state = _SEED

    def absorb(word):
        nonlocal state
        state ^= word
        state = (state * _PRIME) & _MASK64
        state ^= state >> 29

    whole = len(data) - len(data) % 8
    for offset in range(0, whole, 8):
        absorb(int.from_bytes(data[offset:offset + 8], "little"))
    tail = data[whole:]
    if tail:
        absorb(int.from_bytes(tail + bytes(8 - len(tail)), "little"))
    absorb(len(data))

    digest = state
    digest ^= digest >> 33
    digest = (digest * _AVALANCHE) & _MASK64
    digest ^= digest >> 33
    return f"{digest:016x}"


def repair_manifest(directory, *names):
    """Record what the named artefacts now hold, so a load reaches the parsers.

    Every artefact in the directory when no name is given, which is what a test
    editing several of them wants. A name the manifest does not carry a digest
    for is skipped, which covers a directory written before digests existed and
    an artefact a test has deleted outright.
    """
    import json
    import os

    manifest_path = os.path.join(str(directory), "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    digests = manifest.get("file_digests")
    if not digests:
        return

    for name in names or list(digests):
        entry = digests.get(name)
        path = os.path.join(str(directory), name)
        if entry is None or not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            data = handle.read()
        entry["bytes"] = len(data)
        if "checksum" in entry:
            entry["checksum"] = artefact_digest(data)

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
