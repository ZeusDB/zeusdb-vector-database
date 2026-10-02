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
    return f"{checksum(data):016x}"


def checksum(data):
    """The 64 bit checksum the crate takes over a buffer, as an integer."""
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
    return digest


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


# ============================================================================
# THE FRAME, AND THE WIRE BEFORE IT
# ============================================================================
#
# A directory this build saves holds mappings.bin, vectors.bin, pq_codes.bin
# and pq_centroids.bin inside a frame: a 64 byte header carrying a checksum
# over itself, the payload, and a 16 byte trailer carrying a checksum over the
# payload and the magic again. A test that edits a framed payload to reach the
# payload's reader frames it again, which repairs both checksums. A directory
# a release before 4.0.0 saved holds the same four in bincode's wire, and
# `as_old_wire` rewrites a directory this build saved into that form.
#
# Both are second implementations of the formats, written from them rather
# than called out of the library. `test_hostile_files.py` holds the frame
# against every framed artefact a real save wrote.

FRAME_MAGIC = b"ZDBFRAME"
FRAME_KINDS = {"mappings.bin": 1, "vectors.bin": 2, "pq_codes.bin": 3, "pq_centroids.bin": 4}


def frame(kind, entries, payload):
    """A payload inside a frame of `kind` naming `entries` entries."""
    header = bytearray(64)
    header[0:8] = FRAME_MAGIC
    header[8:12] = (1).to_bytes(4, "little")
    header[16] = kind
    header[17] = 2
    header[24:32] = len(payload).to_bytes(8, "little")
    header[32:40] = (len(payload) + 80).to_bytes(8, "little")
    header[40:48] = (entries % (1 << 64)).to_bytes(8, "little")
    header[56:64] = checksum(bytes(header[:56])).to_bytes(8, "little")
    return bytes(header) + bytes(payload) + checksum(bytes(payload)).to_bytes(8, "little") + FRAME_MAGIC


def unframe(data):
    """A frame's kind, its entry count and its payload."""
    assert data[:8] == FRAME_MAGIC and data[-8:] == FRAME_MAGIC, "not a frame"
    payload_bytes = int.from_bytes(data[24:32], "little")
    return data[16], int.from_bytes(data[40:48], "little"), data[64:64 + payload_bytes]


def reframe(path, payload=None, entries=None):
    """Frame a framed artefact again around an edited payload, or an edited
    entry count, so the payload's reader sees the edit."""
    kind, count, held = unframe(path.read_bytes())
    path.write_bytes(frame(kind, count if entries is None else entries,
                           held if payload is None else payload))


def _u32(data, at):
    return int.from_bytes(data[at:at + 4], "little")


def varint(value):
    """A length or an integer as bincode's standard configuration wrote it."""
    if value <= 250:
        return bytes([value])
    if value <= 0xFFFF:
        return b"\xfb" + value.to_bytes(2, "little")
    if value <= 0xFFFFFFFF:
        return b"\xfc" + value.to_bytes(4, "little")
    return b"\xfd" + value.to_bytes(8, "little")


def wire_str(text):
    raw = text.encode("utf-8")
    return varint(len(raw)) + raw


def as_old_wire(directory, version):
    """Rewrite a directory this build saved as a release before 4.0.0 wrote it.

    The four binary artefacts go into bincode's wire, keyed by external id as
    they were, the manifest records a length and a digest for each, and it
    declares `version`.
    """
    import json
    from pathlib import Path as _Path

    directory = _Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    _, entries, payload = unframe((directory / "mappings.bin").read_bytes())
    ids = [_u32(payload, 4 * i) for i in range(entries)]
    names, at = {}, 8 * entries
    for i, internal_id in enumerate(ids):
        length = _u32(payload, 4 * entries + 4 * i)
        names[internal_id] = payload[at:at + length].decode("utf-8")
        at += length
    rewritten = {"mappings.bin": (
        varint(entries) + b"".join(wire_str(names[i]) + varint(i) for i in ids)
        + varint(entries) + b"".join(varint(i) + wire_str(names[i]) for i in ids))}

    for name, unit in (("vectors.bin", 4), ("pq_codes.bin", 1)):
        path = directory / name
        if not path.exists():
            continue
        _, entries, payload = unframe(path.read_bytes())
        width = _u32(payload, 0)
        stride = 4 + unit * width
        rows = [payload[4 + stride * i:4 + stride * (i + 1)] for i in range(entries)]
        rewritten[name] = varint(entries) + b"".join(
            wire_str(names[_u32(row, 0)]) + varint(width) + row[4:] for row in rows)

    path = directory / "pq_centroids.bin"
    if path.exists():
        _, subvectors, payload = unframe(path.read_bytes())
        centroids, width = _u32(payload, 0), _u32(payload, 4)
        values, at, out = payload[8:], 0, varint(subvectors)
        for _ in range(subvectors):
            out += varint(centroids)
            for _ in range(centroids):
                out += varint(width) + values[at:at + 4 * width]
                at += 4 * width
        rewritten["pq_centroids.bin"] = out

    for name, data in rewritten.items():
        (directory / name).write_bytes(data)
        manifest["file_digests"][name] = {"bytes": len(data), "checksum": artefact_digest(data)}
    manifest["format_version"] = version
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
