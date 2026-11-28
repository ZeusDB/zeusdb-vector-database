//! A directory this build saved, rewritten as a release before format 5.0.0
//! wrote one, for the tests that read such a directory.
//!
//! A release before 5.0.0 wrote each record's row twice: in the graph dump,
//! at version 1, and in the separate store its graph kind took, being
//! `vectors.bin` for a graph holding raw vectors, `pq_codes.bin` for a
//! product quantized graph and `int8_rows.zdbint8` for a scalar one. Its
//! `metadata.json` named every record. [`rewrite_before_five`] writes all of
//! that from the collection the directory holds, so a rewritten directory is
//! one an earlier release could hold. A test that needs a release's own
//! bytes reads a fixture under `tests/fixtures/formats`.

use super::framed;
use crate::collection::Collection;
use serde_json::{json, Map, Value};
use std::path::Path;
use zeusdb_vector_core::{
    checksum_of, frame, FrameEncoding, FrameKind, FsDir, Position, DUMP_FILENAME,
};

/// Rewrite the directory at `path`, which this build saved, as a release
/// before format 5.0.0 wrote it, declaring `version`.
///
/// The directory is read through a load of its checkpoint alone, so a
/// collection holding its lock, a journaled one among them, is dropped
/// first. Every record's row goes into the separate store its graph kind
/// takes, the graph dump is written again at version 1, and `metadata.json`
/// names every record, a record with no field under an empty object. Each
/// file is recorded in the manifest as a save records it, a length for a
/// framed file and the dump and a length and a digest for a JSON one. At 4.x
/// the identity is listed among the features, as a 4.x save lists it. Every
/// other field of the directory stays as it was, for the caller to set where
/// the version needs it.
pub(crate) fn rewrite_before_five(path: &Path, version: &str) {
    let collection = Collection::load_checkpoint_only(path.to_str().unwrap(), None)
        .unwrap_or_else(|e| panic!("{} did not load to be rewritten: {}", path.display(), e));
    let records = collection.internal_ids();
    let dim = collection.dim();
    let mut written: Vec<(&str, Vec<u8>, bool)> = Vec::new();
    {
        let index = collection.dense_index();
        let graph = index.graph();
        let node = |at: usize| {
            index
                .node_of(Position::from_slot(at))
                .expect("every record a load holds has a node")
        };
        if !records.is_empty() {
            if graph.holds_raw() {
                let rows = records.iter().enumerate().map(|(at, (internal_id, _))| {
                    let vector = graph
                        .raw_vector_at(node(at))
                        .expect("a graph holding raw vectors holds one for every node");
                    (*internal_id, vector)
                });
                let (bytes, _) = framed::write_vectors(dim, rows, records.len()).unwrap();
                written.push(("vectors.bin", bytes, false));
            }
            if graph.is_pq() {
                let width = graph
                    .codes_at(node(0))
                    .expect("a product quantized graph holds codes for every node")
                    .len();
                let rows = records.iter().enumerate().map(|(at, (internal_id, _))| {
                    let codes = graph
                        .codes_at(node(at))
                        .expect("a product quantized graph holds codes for every node");
                    (*internal_id, codes)
                });
                let (bytes, _) = framed::write_codes(width, rows, records.len()).unwrap();
                written.push(("pq_codes.bin", bytes, false));
            }
            if graph.is_int8() {
                let mut payload = Vec::new();
                for (at, (internal_id, _)) in records.iter().enumerate() {
                    let row = graph
                        .int8_row_at(node(at))
                        .expect("a scalar graph holds a row for every node");
                    payload.extend_from_slice(&(*internal_id as u32).to_le_bytes());
                    payload.extend(row.iter().map(|&value| value as u8));
                }
                let bytes = frame(
                    FrameKind::Int8Rows,
                    FrameEncoding::Engine,
                    records.len() as u64,
                    &payload,
                );
                written.push(("int8_rows.zdbint8", bytes, false));
            }
            graph
                .dump_version_1(&FsDir::new(path), DUMP_FILENAME)
                .unwrap();
            let dump = std::fs::read(path.join(DUMP_FILENAME)).unwrap();
            written.push((DUMP_FILENAME, dump, false));
        }
    }
    drop(collection);

    let read = |name: &str| -> Value {
        serde_json::from_slice(&std::fs::read(path.join(name)).unwrap()).unwrap()
    };
    let mut metadata = match read("metadata.json") {
        Value::Object(map) => map,
        other => panic!("metadata.json holds {other}"),
    };
    let mut every = Map::new();
    for (_, id) in &records {
        let fields = metadata.remove(id).unwrap_or_else(|| json!({}));
        every.insert(id.clone(), fields);
    }
    let metadata = serde_json::to_string_pretty(&Value::Object(every)).unwrap();
    written.push(("metadata.json", metadata.into_bytes(), true));

    let mut manifest = read("manifest.json");
    for (name, bytes, digest) in written {
        std::fs::write(path.join(name), &bytes).unwrap();
        manifest["file_digests"][name] = if digest {
            json!({ "bytes": bytes.len(), "checksum": format!("{:016x}", checksum_of(&bytes)) })
        } else {
            json!({ "bytes": bytes.len() })
        };
        let included = manifest["files_included"].as_array_mut().unwrap();
        if !included.iter().any(|entry| entry == name) {
            included.push(json!(name));
        }
        manifest["files_excluded"]
            .as_array_mut()
            .unwrap()
            .retain(|entry| entry != name);
    }
    manifest["format_version"] = json!(version);
    // A 4.x save lists the identity among its features where it records one.
    if version.starts_with("4.") && !manifest["identity"].is_null() {
        manifest["features"]["identity"] = json!("compatible");
    }
    std::fs::write(
        path.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}
