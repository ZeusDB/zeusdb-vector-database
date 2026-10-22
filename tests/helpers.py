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
    an artefact a test has deleted outright. A 5.x manifest records the
    directory size its lengths add up to as well.
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
    if str(manifest.get("format_version", "")).startswith("5."):
        manifest["total_size_mb"] = sum(entry["bytes"] for entry in digests.values()) / (1024 * 1024)

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
FRAME_KIND_INT8_ROWS = 8


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


# ============================================================================
# A DIRECTORY BEFORE FORMAT 5.0.0
# ============================================================================
#
# A release before 5.0.0 held each record's row twice: in the graph dump at
# version 1, and in the separate store its graph took, `vectors.bin` for a raw
# graph, `pq_codes.bin` for a product quantized one and `int8_rows.zdbint8`
# for a scalar one. Its `metadata.json` named every record. `as_before_five`
# writes that from a directory this build saved, reading the version 2 dump.
# A version 1 entry records a distance, which the reader orders a list by, and
# each entry's is its place in the list, which orders the list as it is held.
# A test that needs a release's own bytes reads a fixture under
# tests/fixtures/formats. `reseal_dump` seals a version 2 dump again after a
# test edits inside one of its regions, `break_topology` damages the topology
# alone, and `dump_rows` reads each node's row.

GRAPH_MAGIC = b"ZDBGRAPH"
_ELEMENT_BYTES = {1: 4, 2: 1, 3: 1}


def _u64(data, at):
    return int.from_bytes(data[at:at + 8], "little")


def _dump_version_2(data):
    """A version 2 dump as its header fields, its layer table, every node's
    origin id and row, and every node's lists of node indices."""
    header = {
        "kind": data[16],
        "element": data[17],
        "dimension": _u32(data, 20),
        "m": _u32(data, 24),
        "ef_construction": _u32(data, 28),
        "nodes": _u32(data, 32),
        "entry_node": _u32(data, 36),
        "level_scale": data[40:48],
    }
    nodes = header["nodes"]
    identity, rows_at = 128, 128 + _u64(data, 48)
    topology_at = rows_at + _u64(data, 56)
    layers = [_u32(data, identity + 4 * layer) for layer in range(16)]
    origins = [_u32(data, identity + 64 + 4 * node) for node in range(nodes)]
    width = header["dimension"] * _ELEMENT_BYTES[header["element"]]
    rows = [data[rows_at + width * node:rows_at + width * (node + 1)] for node in range(nodes)]
    lists, at = [], topology_at
    for _ in range(nodes):
        count, at = data[at], at + 1
        held = []
        for _ in range(count):
            entries = int.from_bytes(data[at:at + 2], "little")
            at += 2
            held.append([_u32(data, at + 4 * k) for k in range(entries)])
            at += 4 * entries
        lists.append(held)
    return header, layers, origins, rows, lists


def dump_rows(data):
    """Each node's row in a version 2 dump, by the internal id its identity
    region names for the node."""
    _, _, origins, rows, _ = _dump_version_2(data)
    return dict(zip(origins, rows))


def break_topology(path):
    """Flip a bit of the first list count in the version 2 dump at `path`, so
    its topology fails its checksum and a load rebuilds the graph from the
    rows the dump holds."""
    data = bytearray(path.read_bytes())
    data[128 + _u64(data, 48) + _u64(data, 56)] ^= 0x01
    path.write_bytes(bytes(data))


def reseal_dump(path):
    """Seal the version 2 dump at `path` again after an edit inside a region,
    so the region's reader sees the edit: each region's checksum, then the
    header's, which covers them."""
    data = bytearray(path.read_bytes())
    at = 128
    for length_at, checksum_at in ((48, 80), (56, 88), (64, 96)):
        length = _u64(data, length_at)
        data[checksum_at:checksum_at + 8] = checksum(bytes(data[at:at + length])).to_bytes(8, "little")
        at += length
    data[120:128] = checksum(bytes(data[:120])).to_bytes(8, "little")
    path.write_bytes(bytes(data))


def _dump_version_1(header, layers, origins, rows, lists):
    """The version 1 dump of the graph a version 2 dump holds."""
    import struct

    starts, at = [], 0
    for count in layers:
        starts.append(at)
        at += count

    def place(node):
        for layer in range(16):
            if starts[layer] <= node < starts[layer] + layers[layer]:
                return layer, node - starts[layer]
        raise ValueError(f"node {node} is past the layer table")

    adjacency = bytearray()
    for held in lists:
        adjacency.append(len(held))
        for entries in held:
            adjacency += struct.pack("<I", len(entries))
            for k, target in enumerate(entries):
                layer, rank = place(target)
                adjacency += struct.pack("<BIf", layer, rank, float(k))
    body = (b"".join(struct.pack("<I", count) for count in layers)
            + b"".join(struct.pack("<Q", origin) for origin in origins)
            + bytes(adjacency) + b"".join(rows))
    entry_layer, entry_rank = place(header["entry_node"])
    head = bytearray(96)
    head[0:8] = GRAPH_MAGIC
    head[8:12] = struct.pack("<I", 1)
    head[16] = header["kind"]
    head[17] = header["element"]
    head[18] = 16
    head[20:24] = struct.pack("<I", header["dimension"])
    head[24:32] = struct.pack("<Q", header["m"])
    head[32:40] = struct.pack("<Q", header["ef_construction"])
    head[40:48] = struct.pack("<Q", header["nodes"])
    head[48:56] = header["level_scale"]
    head[56:60] = struct.pack("<I", entry_layer)
    head[60:64] = struct.pack("<I", entry_rank)
    head[64:72] = struct.pack("<Q", len(adjacency))
    head[72:80] = struct.pack("<Q", 96 + len(body) + 16)
    head[88:96] = struct.pack("<Q", checksum(bytes(head[:88])))
    return bytes(head) + body + struct.pack("<Q", checksum(body)) + GRAPH_MAGIC


def as_before_five(directory, version):
    """Rewrite a directory this build saved as a release before 5.0.0 wrote
    it, declaring `version`.

    Every live record's row goes into the separate store its graph takes, in
    increasing internal id, the graph dump is written again at version 1,
    and `metadata.json` names every record, a record with no field under an
    empty object. The manifest records each file as a save records it, and
    at 4.x lists the identity among its features. Every other field stays as
    it was.
    """
    import json
    import struct
    from pathlib import Path as _Path

    directory = _Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _, entries, payload = unframe((directory / "mappings.bin").read_bytes())
    internal = [_u32(payload, 4 * i) for i in range(entries)]
    names, at = [], 8 * entries
    for i in range(entries):
        length = _u32(payload, 4 * entries + 4 * i)
        names.append(payload[at:at + length].decode("utf-8"))
        at += length

    written = {}
    dump_path = directory / "hnsw_index.zdbgraph"
    if dump_path.exists():
        header, layers, origins, rows, lists = _dump_version_2(dump_path.read_bytes())
        node_of = {origin: node for node, origin in enumerate(origins)}
        live = [(internal_id, rows[node_of[internal_id]]) for internal_id in internal]
        element, dimension = header["element"], header["dimension"]
        if element == 1 and not (directory / "vectors.bin").exists():
            body = struct.pack("<I", dimension) + b"".join(
                struct.pack("<I", i) + row for i, row in live)
            written["vectors.bin"] = (frame(2, len(live), body), False)
        if element == 2:
            body = struct.pack("<I", dimension) + b"".join(
                struct.pack("<I", i) + row for i, row in live)
            written["pq_codes.bin"] = (frame(3, len(live), body), False)
        if element == 3:
            body = b"".join(struct.pack("<I", i) + row for i, row in live)
            written["int8_rows.zdbint8"] = (frame(FRAME_KIND_INT8_ROWS, len(live), body), False)
        written["hnsw_index.zdbgraph"] = (_dump_version_1(header, layers, origins, rows, lists), False)

    held = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    every = {name: held.get(name, {}) for name in names}
    written["metadata.json"] = (json.dumps(every, indent=2).encode("utf-8"), True)

    for name, (data, digest) in written.items():
        (directory / name).write_bytes(data)
        entry = {"bytes": len(data)}
        if digest:
            entry["checksum"] = artefact_digest(data)
        manifest["file_digests"][name] = entry
        if name not in manifest["files_included"]:
            manifest["files_included"].append(name)
        manifest["files_excluded"] = [f for f in manifest["files_excluded"] if f != name]
    manifest["format_version"] = version
    if version.startswith("4.") and manifest.get("identity") is not None:
        manifest.setdefault("features", {})["identity"] = "compatible"
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def as_old_wire(directory, version):
    """Rewrite a directory this build saved as a release before 4.0.0 wrote it.

    Every record's row goes into its separate store and the graph dump is
    written at version 1, through `as_before_five`, then the four binary
    artefacts go into bincode's wire, keyed by external id as they were, the
    manifest records a length and a digest for each, and it declares
    `version`.
    """
    import json
    from pathlib import Path as _Path

    directory = _Path(directory)
    as_before_five(directory, version)
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
