//! The four binary artefacts in both layouts.
//!
//! The frame this build writes, held to what it reads back, to the same bytes
//! from two saves of the same records, and to a refusal by name for each way
//! a payload can be wrong. The wire releases before 4.0.0 wrote, held to the
//! values it carries, to the last copy of a key, to a refusal for every
//! prefix and every length its bytes cannot carry, and to whole directories
//! this build saved and rewrote as an earlier release would have written
//! them.
//!
//! The format version a directory declares, held to the form every release
//! has written it in.

use super::older::rewrite_before_five;
use super::*;
use crate::collection::{Declaration, ParsedRecord};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use zeusdb_vector_core::frame_fuzz;
use zeusdb_vector_core::test_support::clustered;

// ============================================================================
// HELPERS
// ============================================================================

/// The wire releases before 4.0.0 wrote, written by hand as bincode 2's
/// standard configuration wrote it, so a test can make a file of that wire
/// in any shape, a key held twice included.
mod old {
    pub(super) fn varint(out: &mut Vec<u8>, value: u64) {
        match value {
            0..=250 => out.push(value as u8),
            251..=0xFFFF => {
                out.push(251);
                out.extend_from_slice(&(value as u16).to_le_bytes());
            }
            0x1_0000..=0xFFFF_FFFF => {
                out.push(252);
                out.extend_from_slice(&(value as u32).to_le_bytes());
            }
            _ => {
                out.push(253);
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    pub(super) fn str(out: &mut Vec<u8>, text: &str) {
        varint(out, text.len() as u64);
        out.extend_from_slice(text.as_bytes());
    }

    pub(super) fn floats(out: &mut Vec<u8>, values: &[f32]) {
        varint(out, values.len() as u64);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    pub(super) fn mappings(forward: &[(&str, usize)], reverse: &[(usize, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(&mut out, forward.len() as u64);
        for &(id, internal_id) in forward {
            str(&mut out, id);
            varint(&mut out, internal_id as u64);
        }
        varint(&mut out, reverse.len() as u64);
        for &(internal_id, id) in reverse {
            varint(&mut out, internal_id as u64);
            str(&mut out, id);
        }
        out
    }

    pub(super) fn vectors(entries: &[(&str, Vec<f32>)]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(&mut out, entries.len() as u64);
        for (id, vector) in entries {
            str(&mut out, id);
            floats(&mut out, vector);
        }
        out
    }

    pub(super) fn codes(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(&mut out, entries.len() as u64);
        for (id, code) in entries {
            str(&mut out, id);
            varint(&mut out, code.len() as u64);
            out.extend_from_slice(code);
        }
        out
    }

    pub(super) fn codebook(codebook: &[Vec<Vec<f32>>]) -> Vec<u8> {
        let mut out = Vec::new();
        varint(&mut out, codebook.len() as u64);
        for sub in codebook {
            varint(&mut out, sub.len() as u64);
            for centroid in sub {
                floats(&mut out, centroid);
            }
        }
        out
    }
}

/// A directory under the system's temporary directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zeusdb-persistence-tests-{}-{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn config(id_counter: usize) -> IndexConfig {
    IndexConfig {
        dim: 2,
        space: "l2".to_string(),
        m: 4,
        ef_construction: 50,
        expected_size: 16,
        id_counter,
        vector_count: 0,
        generated_ids: 0,
        metadata: BTreeMap::new(),
        indexed_fields: vec![],
        spaces: vec![],
    }
}

/// The records a mappings file holds, in increasing internal id, each at its
/// place among them.
fn store(records: &[(usize, &str)]) -> MappedIds {
    let mut store = IdStore::new(16);
    let mut internal = Vec::new();
    for (at, &(internal_id, id)) in records.iter().enumerate() {
        store.insert(at, id).unwrap();
        internal.push(internal_id as u32);
    }
    MappedIds { store, internal }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A refusal's words, or the word for having read.
fn refused<T>(result: Result<T, Error>) -> String {
    match result {
        Ok(_) => "read".to_string(),
        Err(e) => e.to_string(),
    }
}

/// A payload framed by hand, for a test that needs one no writer makes.
fn framed(kind: FrameKind, entries: u64, payload: &[u8]) -> Vec<u8> {
    frame(kind, FrameEncoding::Engine, entries, payload)
}

fn le(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn shape(subvectors: usize, centroids: usize, width: usize) -> framed::Shape {
    framed::Shape {
        expected: (subvectors, centroids, width),
        subvectors,
        bits: centroids.trailing_zeros() as usize,
    }
}

fn record(id: &str, vector: &[f32], cat: &str, rank: i64) -> ParsedRecord {
    ParsedRecord {
        id: id.to_string(),
        vector: vector.to_vec(),
        sparse: None,
        metadata: HashMap::from([
            ("cat".to_string(), json!(cat)),
            ("rank".to_string(), json!(rank)),
            ("tags".to_string(), json!(["x", cat])),
        ]),
    }
}

/// A collection of `n` clustered records at width 8, with a declared field,
/// under product quantization of four subvectors at four bits where `mode`
/// names a storage mode.
fn filled(quantization: Option<StorageMode>, n: usize) -> Collection {
    let dim = 8;
    let declaration =
        Declaration::validate(dim, "l2", 6, 60, 2000, vec!["cat".to_string()]).unwrap();
    let quantization =
        quantization.map(|mode| declaration.quantization(4, 4, 1000, None, mode).unwrap());
    let collection = Collection::build(declaration, quantization);
    let vectors = clustered(n, dim, 0x0018_2001);
    let records: Vec<ParsedRecord> = vectors
        .iter()
        .enumerate()
        .map(|(i, vector)| {
            record(
                &format!("r{i}\u{e9}"),
                vector,
                ["a", "b", "c"][i % 3],
                i as i64,
            )
        })
        .collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    collection
}

/// Every record, sorted by id, as its internal id, its vector's bits and its
/// metadata in key order.
fn everything(collection: &Collection) -> Vec<(String, usize, Vec<u32>, String)> {
    let names: Vec<(usize, String)> = collection.internal_ids();
    let views = collection
        .records(names.iter().map(|(_, id)| id.clone()).collect(), true, true)
        .unwrap();
    let mut out: Vec<(String, usize, Vec<u32>, String)> = names
        .iter()
        .zip(views)
        .map(|((internal_id, id), view)| {
            let metadata: BTreeMap<String, Value> = view.metadata.into_iter().collect();
            (
                id.clone(),
                *internal_id,
                bits(&view.vector.unwrap_or_default()),
                serde_json::to_string(&metadata).unwrap(),
            )
        })
        .collect();
    out.sort();
    out
}

/// One page, as an external id and a score's bits per hit.
fn page(collection: &Collection, query: &[f32]) -> Vec<(String, u32)> {
    let params = collection.search_params(10, None, false, None).unwrap();
    collection
        .search_one(query, None, params)
        .unwrap()
        .iter()
        .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
        .collect()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn write_json(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_string_pretty(value).unwrap()).unwrap();
}

/// Rewrite a directory this build saved as a release before 4.0.0 would have
/// written it: every record's row in its separate store and the graph dump
/// at version 1, through `older`, then the four binary artefacts in the old
/// wire, in the order a hash map would hand them out, and the manifest
/// declaring `version` with a length and a digest for each.
fn as_old(path: &Path, version: &str) {
    super::older::rewrite_before_five(path, version);
    let config = read_json(&path.join("config.json"));
    let dim = config["dim"].as_u64().unwrap() as usize;
    let quantization = path
        .join("quantization.json")
        .exists()
        .then(|| read_json(&path.join("quantization.json")));
    let mut manifest = read_json(&path.join("manifest.json"));

    let bytes = std::fs::read(path.join("mappings.bin")).unwrap();
    let records = framed::Mappings::read(&bytes, "mappings.bin").unwrap();
    let names: HashMap<usize, String> = records
        .iter()
        .map(|(internal_id, id)| (internal_id, id.to_string()))
        .collect();
    let forward: HashMap<&str, usize> = names
        .iter()
        .map(|(&internal_id, id)| (id.as_str(), internal_id))
        .collect();
    let forward: Vec<(&str, usize)> = forward.into_iter().collect();
    let reverse: Vec<(usize, &str)> = names
        .iter()
        .map(|(&internal_id, id)| (internal_id, id.as_str()))
        .collect();
    let mut rewritten = vec![("mappings.bin", old::mappings(&forward, &reverse))];

    if let Ok(bytes) = std::fs::read(path.join("vectors.bin")) {
        let rows = framed::Rows::read(&bytes, FrameKind::RawVectors, "v", 4, dim, |_| {
            String::new()
        })
        .unwrap();
        let entries: HashMap<&str, Vec<f32>> = rows
            .iter()
            .map(|(internal_id, row)| {
                (
                    names[&internal_id].as_str(),
                    framed::floats(row).collect::<Vec<f32>>(),
                )
            })
            .collect();
        let entries: Vec<(&str, Vec<f32>)> = entries.into_iter().collect();
        rewritten.push(("vectors.bin", old::vectors(&entries)));
    }
    if let Ok(bytes) = std::fs::read(path.join("pq_codes.bin")) {
        let width = quantization.as_ref().unwrap()["subvectors"]
            .as_u64()
            .unwrap() as usize;
        let rows = framed::Rows::read(&bytes, FrameKind::PqCodes, "c", 1, width, |_| String::new())
            .unwrap();
        let entries: HashMap<&str, Vec<u8>> = rows
            .iter()
            .map(|(internal_id, code)| (names[&internal_id].as_str(), code.to_vec()))
            .collect();
        let entries: Vec<(&str, Vec<u8>)> = entries.into_iter().collect();
        rewritten.push(("pq_codes.bin", old::codes(&entries)));
    }
    if let Ok(bytes) = std::fs::read(path.join("pq_centroids.bin")) {
        let q = quantization.as_ref().unwrap();
        let subvectors = q["subvectors"].as_u64().unwrap() as usize;
        let centroids = 1usize << q["bits"].as_u64().unwrap();
        let codebook = framed::read_codebook(
            &bytes,
            "pq_centroids.bin",
            shape(subvectors, centroids, dim / subvectors),
        )
        .unwrap();
        rewritten.push(("pq_centroids.bin", old::codebook(&codebook)));
    }

    for (name, bytes) in rewritten {
        std::fs::write(path.join(name), &bytes).unwrap();
        manifest["file_digests"][name] = json!({
            "bytes": bytes.len(),
            "checksum": format!("{:016x}", checksum_of(&bytes)),
        });
    }
    manifest["format_version"] = json!(version);
    write_json(&path.join("manifest.json"), &manifest);
}

/// Replace one artefact and record its new length, as a hand would, and
/// the directory size those lengths add up to.
fn replace(path: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(path.join(name), bytes).unwrap();
    let mut manifest = read_json(&path.join("manifest.json"));
    manifest["file_digests"][name] = json!({ "bytes": bytes.len() });
    let total: u64 = manifest["file_digests"]
        .as_object()
        .unwrap()
        .values()
        .map(|digest| digest["bytes"].as_u64().unwrap())
        .sum();
    manifest["total_size_mb"] = json!(total as f64 / (1024.0 * 1024.0));
    write_json(&path.join("manifest.json"), &manifest);
}

// ============================================================================
// THE FRAME
// ============================================================================

/// Every framed artefact reads back what was written, ids that are not
/// ASCII, an empty id and one of 65,536 bytes among them, and floats by their
/// bits, a NaN and a negative zero among them.
#[test]
fn every_framed_artefact_reads_back_what_was_written() {
    let long = "x".repeat(65_536);
    let ids: Vec<(usize, &str)> = vec![
        (1, "a"),
        (2, ""),
        (5, "\u{e9}t\u{e9}"),
        (6, "\u{1f980}"),
        (9, "a\u{0}b"),
        (12, &long),
    ];
    let text: usize = ids.iter().map(|(_, id)| id.len()).sum();
    let bytes = framed::write_mappings(ids.iter().copied(), ids.len(), text).unwrap();
    let read = framed::Mappings::read(&bytes, "mappings.bin").unwrap();
    assert_eq!(read.iter().collect::<Vec<_>>(), ids);
    assert_eq!(read.last(), Some((12, long.as_str())));
    let built = framed_ids(&bytes, &config(12)).unwrap();
    assert_eq!(built.iter().collect::<Vec<_>>(), ids);

    let none = framed::write_mappings(std::iter::empty(), 0, 0).unwrap();
    assert_eq!(none.len(), FRAME_OVERHEAD_BYTES);
    assert_eq!(framed_ids(&none, &config(0)).unwrap().len(), 0);

    let unusual = [f32::NAN, -0.0, f32::MIN_POSITIVE / 2.0, f32::INFINITY];
    let rows: Vec<(usize, Vec<f32>)> = vec![(1, unusual.to_vec()), (4, vec![1.5, -2.0, 0.0, 3.25])];
    let (bytes, entries) =
        framed::write_vectors(4, rows.iter().map(|(i, v)| (*i, v.as_slice())), 2).unwrap();
    assert_eq!(entries, 2);
    let read =
        framed::Rows::read(&bytes, FrameKind::RawVectors, "v", 4, 4, |_| String::new()).unwrap();
    let back: Vec<(usize, Vec<u32>)> = read
        .iter()
        .map(|(i, row)| (i, framed::floats(row).map(f32::to_bits).collect()))
        .collect();
    let want: Vec<(usize, Vec<u32>)> = rows.iter().map(|(i, v)| (*i, bits(v))).collect();
    assert_eq!(back, want);

    let codes: Vec<(usize, Vec<u8>)> = vec![(3, vec![0, 255, 7, 1]), (8, vec![9, 9, 9, 9])];
    let (bytes, entries) =
        framed::write_codes(4, codes.iter().map(|(i, c)| (*i, c.as_slice())), 2).unwrap();
    assert_eq!(entries, 2);
    let read =
        framed::Rows::read(&bytes, FrameKind::PqCodes, "c", 1, 4, |_| String::new()).unwrap();
    let back: Vec<(usize, Vec<u8>)> = read.iter().map(|(i, c)| (i, c.to_vec())).collect();
    assert_eq!(back, codes);

    let codebook: Vec<Vec<Vec<f32>>> = (0..4)
        .map(|s| {
            (0..16)
                .map(|c| vec![s as f32 + 0.5, -(c as f32), f32::from_bits(0x7fc0_0001)])
                .collect()
        })
        .collect();
    let bytes = framed::write_codebook(&codebook).unwrap();
    let back = framed::read_codebook(&bytes, "pq_centroids.bin", shape(4, 16, 3)).unwrap();
    let as_bits = |cb: &Vec<Vec<Vec<f32>>>| -> Vec<Vec<Vec<u32>>> {
        cb.iter()
            .map(|s| s.iter().map(|c| bits(c)).collect())
            .collect()
    };
    assert_eq!(as_bits(&back), as_bits(&codebook));
    assert_eq!(back.capacity(), 4);
    assert!(back
        .iter()
        .all(|s| s.capacity() == 16 && s.iter().all(|c| c.capacity() == 3)));
}

/// Two saves of the same records write the same bytes, the second after a
/// load has rebuilt every hash map under a seed of its own: the framed
/// artefacts, the graph dump, `metadata.json` and `config.json` with index
/// level metadata.
#[test]
fn two_saves_of_the_same_records_write_the_same_bytes() {
    let temp = TempDir::new();
    for (label, mode) in [
        ("raw", None),
        ("qraw", Some(StorageMode::QuantizedWithRaw)),
        ("qonly", Some(StorageMode::QuantizedOnly)),
    ] {
        let collection = filled(mode, 1050);
        collection
            .add_metadata(HashMap::from([
                ("owner".to_string(), "x".to_string()),
                ("purpose".to_string(), "y".to_string()),
                ("alpha".to_string(), "z".to_string()),
                ("zeta".to_string(), "w".to_string()),
            ]))
            .unwrap();
        let first = temp.at(&format!("{label}-first.zdb"));
        collection.save(first.to_str().unwrap()).unwrap();
        let loaded = Collection::load(first.to_str().unwrap()).unwrap();
        let second = temp.at(&format!("{label}-second.zdb"));
        loaded.save(second.to_str().unwrap()).unwrap();
        let mut compared = 0;
        for name in [
            "mappings.bin",
            "vectors.bin",
            "pq_codes.bin",
            "pq_centroids.bin",
            "metadata.json",
            "config.json",
            "hnsw_index.zdbgraph",
        ] {
            let a = std::fs::read(first.join(name)).ok();
            let b = std::fs::read(second.join(name)).ok();
            assert_eq!(a.is_some(), b.is_some(), "{label} {name}");
            if let (Some(a), Some(b)) = (a, b) {
                assert!(a == b, "{label} {name} moved between two saves");
                compared += 1;
            }
        }
        assert!(compared >= 4, "{label} compared {compared}");
        assert_eq!(
            read_json(&first.join("manifest.json"))["format_version"],
            json!("5.0.0")
        );
    }
}

/// Every way a framed `mappings.bin` can be wrong, refused by name.
#[test]
fn every_refusal_of_a_framed_mappings_is_named() {
    let kind = FrameKind::IdMappings;
    let file = |payload: &[u8], entries: u64| framed(kind, entries, payload);
    let say = |bytes: &[u8]| refused(framed_ids(bytes, &config(100)));

    let mut payload = le(&[1, 2]);
    payload.extend(le(&[1, 1]));
    payload.extend_from_slice(b"ab");
    assert_eq!(say(&file(&payload, 2)), "read");

    assert!(
        say(&file(&payload, 3)).ends_with("holds 18 payload bytes and 3 records take at least 24")
    );
    let mut long = payload.clone();
    long.push(b'c');
    assert!(say(&file(&long, 2))
        .ends_with("the ids' lengths sum to 2 bytes and the payload holds 3 bytes of id"));
    let mut swapped = le(&[5, 2]);
    swapped.extend(le(&[1, 1]));
    swapped.extend_from_slice(b"ab");
    assert!(say(&file(&swapped, 2))
        .ends_with("internal id 2 follows 5, and the ids are strictly increasing"));
    let mut twice = le(&[3, 3]);
    twice.extend(le(&[1, 1]));
    twice.extend_from_slice(b"ab");
    assert!(say(&file(&twice, 2))
        .ends_with("internal id 3 follows 3, and the ids are strictly increasing"));
    let mut not_utf8 = le(&[4, 7]);
    not_utf8.extend(le(&[1, 1]));
    not_utf8.extend_from_slice(&[b'a', 0xff]);
    assert!(say(&file(&not_utf8, 2)).ends_with("the id held at internal id 7 is not UTF-8"));
    let e = "\u{e9}".as_bytes();
    let mut split = le(&[4, 7]);
    split.extend(le(&[1, 1]));
    split.extend_from_slice(e);
    assert!(say(&file(&split, 2)).ends_with("the id held at internal id 4 is not UTF-8"));

    let mut high = le(&[1, 9]);
    high.extend(le(&[1, 1]));
    high.extend_from_slice(b"ab");
    assert_eq!(
        refused(framed_ids(&file(&high, 2), &config(4))),
        "Failed to parse mappings.bin: the record 'b' holds internal id 9 and config.json counted 4"
    );
    let mut same = le(&[1, 2]);
    same.extend(le(&[1, 1]));
    same.extend_from_slice(b"aa");
    assert_eq!(
        say(&file(&same, 2)),
        "Failed to parse mappings.bin: 2 records hold 1 distinct ids, so an id is held under two internal ids"
    );
    let vectors = framed(FrameKind::RawVectors, 0, &le(&[2]));
    assert!(say(&vectors).ends_with("the frame holds raw vectors where id mappings was expected"));
    assert!(say(&[1, 2, 3]).ends_with("the file holds 3 bytes and a frame is at least 80"));
}

/// Every way a framed `vectors.bin` can be wrong, refused by name, the
/// finiteness refusal in the words the old one gave.
#[test]
fn every_refusal_of_framed_vectors_is_named() {
    let ids = store(&[(1, "a"), (2, "b")]);
    let say = |bytes: &[u8]| refused(framed_vectors(bytes, ids.view(), 2));
    let row = |id: u32, values: [f32; 2]| -> Vec<u8> {
        let mut out = id.to_le_bytes().to_vec();
        out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        out
    };
    let file = |rows: &[Vec<u8>], width: u32, entries: u64| {
        let mut payload = width.to_le_bytes().to_vec();
        for r in rows {
            payload.extend_from_slice(r);
        }
        framed(FrameKind::RawVectors, entries, &payload)
    };
    let good = [row(1, [1.0, 2.0]), row(2, [3.0, 4.0])];
    assert_eq!(say(&file(&good, 2, 2)), "read");
    assert!(say(&file(&good, 3, 2))
        .ends_with("holds vectors of 3 values and config.json declares dim 2"));
    assert!(
        say(&file(&good, 2, 3)).ends_with("holds 24 bytes of rows and 3 rows of 12 bytes take 36")
    );
    assert!(say(&file(&[row(2, [0.0; 2]), row(1, [0.0; 2])], 2, 2))
        .ends_with("internal id 1 follows 2, and the ids are strictly increasing"));
    assert!(say(&file(&[row(1, [0.0; 2]), row(9, [0.0; 2])], 2, 2))
        .ends_with("names internal id 9, which mappings.bin does not hold"));
    assert!(say(&file(&[row(1, [0.0; 2])], 2, 1))
        .ends_with("holds 1 vectors and mappings.bin holds 2 records; record 'b' has no vector"));
    match framed_vectors(
        &file(&[row(1, [f32::NAN, 0.0]), row(2, [0.0; 2])], 2, 2),
        ids.view(),
        2,
    ) {
        Err(Error::VectorsNotFinite { offenders, total }) => {
            assert_eq!((offenders, total), (vec!["a".to_string()], 2))
        }
        other => panic!("{:?}", other.map(|r| r.len())),
    }
}

/// The codebook is held to its shape before it is built, and a frame naming
/// a codebook its bytes cannot hold is refused before anything is sized,
/// however large the count it names.
#[test]
fn a_framed_codebook_is_held_to_its_shape_before_it_is_built() {
    let say = |bytes: &[u8], s: framed::Shape| {
        refused(framed::read_codebook(bytes, "pq_centroids.bin", s))
    };
    let codebook = vec![vec![vec![0.5f32; 3]; 2]; 2];
    let bytes = framed::write_codebook(&codebook).unwrap();
    assert_eq!(say(&bytes, shape(2, 2, 3)), "read");
    assert_eq!(
        say(&bytes, shape(4, 2, 3)),
        Error::CodebookShapeMismatch {
            actual: (2, 2, 3),
            expected: (4, 2, 3),
            subvectors: 4,
            bits: 1
        }
        .to_string()
    );
    let mut payload = le(&[2, 3]);
    payload.extend(vec![0u8; 40]);
    assert!(
        say(&framed(FrameKind::PqCodebook, 2, &payload), shape(2, 2, 3))
            .ends_with("holds 40 bytes of values and a codebook of 2x2x3 takes 48")
    );
    // A count of 2^40 subvectors over centroids of no width is a codebook of
    // no values, so its bytes agree, and the shape refuses it before a vector
    // of 2^40 is sized.
    let empty = framed(FrameKind::PqCodebook, 1 << 40, &le(&[0, 0]));
    assert!(say(&empty, shape(2, 2, 3)).contains("codebook is 1099511627776x0x0"));
    let ragged = vec![vec![vec![0.5f32; 3], vec![0.5f32; 2]]];
    assert!(refused(framed::write_codebook(&ragged)).contains("differ in length"));
}

/// No mutation of any of the four framed artefacts panics its reader, and
/// the repairs carry most of them past the frame into the payload's reader.
#[test]
fn no_mutation_of_a_framed_artefact_panics_its_reader() {
    let ids = store(&[(1, "r1"), (3, "r3"), (4, "\u{e9}"), (7, "r7")]);
    let text: usize = ids.iter().map(|(_, id)| id.len()).sum();
    let mappings = framed::write_mappings(ids.iter(), ids.len(), text).unwrap();
    let vectors: Vec<(usize, Vec<f32>)> = ids.iter().map(|(i, _)| (i, vec![i as f32; 2])).collect();
    let (vectors, _) =
        framed::write_vectors(2, vectors.iter().map(|(i, v)| (*i, v.as_slice())), 4).unwrap();
    let codes: Vec<(usize, Vec<u8>)> = ids.iter().map(|(i, _)| (i, vec![i as u8; 2])).collect();
    let (codes, _) =
        framed::write_codes(2, codes.iter().map(|(i, c)| (*i, c.as_slice())), 4).unwrap();
    let codebook = framed::write_codebook(&vec![vec![vec![0.25f32; 1]; 4]; 2]).unwrap();

    let mut rng = frame_fuzz::Rng(0x5eed_0182_f4a3_0001);
    let cases = 4_000;
    let mut reached = [0usize; 4];
    for _ in 0..cases {
        let blob = frame_fuzz::mutate(&mut rng, &mappings, FrameKind::IdMappings);
        reached[0] += usize::from(unframe(&blob, FrameKind::IdMappings, "m").is_ok());
        let _ = framed_ids(&blob, &config(8));
        let blob = frame_fuzz::mutate(&mut rng, &vectors, FrameKind::RawVectors);
        reached[1] += usize::from(unframe(&blob, FrameKind::RawVectors, "v").is_ok());
        let _ = framed_vectors(&blob, ids.view(), 2);
        let blob = frame_fuzz::mutate(&mut rng, &codes, FrameKind::PqCodes);
        reached[2] += usize::from(unframe(&blob, FrameKind::PqCodes, "c").is_ok());
        let _ = framed::Rows::read(&blob, FrameKind::PqCodes, "c", 1, 2, |_| String::new())
            .map(|rows| rows.iter().count());
        let blob = frame_fuzz::mutate(&mut rng, &codebook, FrameKind::PqCodebook);
        reached[3] += usize::from(unframe(&blob, FrameKind::PqCodebook, "b").is_ok());
        let _ = framed::read_codebook(&blob, "b", shape(2, 4, 1));
    }
    for (kind, reached) in reached.iter().enumerate() {
        assert!(
            reached * 2 > cases,
            "kind {} reached its payload {} times in {}",
            kind + 1,
            reached,
            cases
        );
    }
}

// ============================================================================
// THE WIRE RELEASES BEFORE 4.0.0 WROTE
// ============================================================================

fn mappings_of(maps: &legacy::Maps) -> (BTreeMap<String, usize>, BTreeMap<usize, String>) {
    (
        maps.id_map.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        maps.rev_map.iter().map(|(k, v)| (*k, v.clone())).collect(),
    )
}

/// Every shape a release wrote reads back as the values it carries, at the
/// boundaries of every length marker.
#[test]
fn the_old_wire_reads_back_every_shape_a_release_wrote() {
    let long = "y".repeat(251);
    let longer = "z".repeat(65_536);
    let ids: Vec<(&str, usize)> = vec![
        ("a", 1),
        ("\u{e9}t\u{e9}", 250),
        ("\u{1f980}", 251),
        (&long, 65_536),
        (&longer, (1 << 32) + 5),
        ("a\u{0}b", 7),
    ];
    let reverse: Vec<(usize, &str)> = ids.iter().map(|&(id, i)| (i, id)).collect();
    let maps = legacy::read_mappings(&old::mappings(&ids, &reverse), "mappings.bin").unwrap();
    let (forward, back) = mappings_of(&maps);
    assert_eq!(
        forward,
        ids.iter().map(|&(id, i)| (id.to_string(), i)).collect()
    );
    assert_eq!(
        back,
        reverse.iter().map(|&(i, id)| (i, id.to_string())).collect()
    );
    let none = legacy::read_mappings(&old::mappings(&[], &[]), "mappings.bin").unwrap();
    assert!(none.id_map.is_empty() && none.rev_map.is_empty());

    let unusual = vec![
        f32::NAN,
        f32::INFINITY,
        -0.0,
        f32::MIN_POSITIVE / 2.0,
        f32::MAX,
    ];
    let entries: Vec<(&str, Vec<f32>)> = vec![
        ("a", unusual.clone()),
        ("b", vec![]),
        ("c", (0..251).map(|i| i as f32).collect()),
    ];
    let read = legacy::read_vectors(&old::vectors(&entries), "vectors.bin", 3).unwrap();
    let read: BTreeMap<&str, Vec<u32>> = read.iter().map(|(k, v)| (k.as_str(), bits(v))).collect();
    let want: BTreeMap<&str, Vec<u32>> = entries.iter().map(|(k, v)| (*k, bits(v))).collect();
    assert_eq!(read, want);

    let codes: Vec<(&str, Vec<u8>)> =
        vec![("a", vec![]), ("b", (0..300u32).map(|i| i as u8).collect())];
    let read = legacy::read_codes(&old::codes(&codes), "pq_codes.bin", 2).unwrap();
    let read: BTreeMap<&str, Vec<u8>> = read.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    assert_eq!(read, codes.iter().map(|(k, v)| (*k, v.clone())).collect());

    let codebook = vec![vec![vec![0.25f32, -1.0]; 16]; 4];
    let read = legacy::read_codebook(
        &old::codebook(&codebook),
        "pq_centroids.bin",
        shape(4, 16, 2),
    )
    .unwrap();
    assert_eq!(read, codebook);
}

/// A key a file holds twice is held once, its last copy, as the old decode's
/// map held it, on every map of the wire.
#[test]
fn a_key_held_twice_is_read_as_the_old_decode_held_it() {
    let vectors = old::vectors(&[("a", vec![1.0]), ("b", vec![2.0]), ("a", vec![3.0])]);
    let read = legacy::read_vectors(&vectors, "vectors.bin", 2).unwrap();
    assert_eq!((read.len(), read["a"].clone()), (2, vec![3.0]));
    let codes = old::codes(&[("a", vec![1]), ("a", vec![2])]);
    assert_eq!(
        legacy::read_codes(&codes, "pq_codes.bin", 1).unwrap()["a"],
        vec![2]
    );
    let maps = legacy::read_mappings(
        &old::mappings(&[("a", 1), ("b", 2), ("a", 3)], &[(3, "a"), (2, "b")]),
        "mappings.bin",
    )
    .unwrap();
    assert_eq!(
        mappings_of(&maps).0,
        BTreeMap::from([("a".to_string(), 3), ("b".to_string(), 2)])
    );
}

/// The entries of an old `vectors.bin`, an id held twice allowed.
type Entries<'a> = Vec<(&'a str, Vec<f32>)>;

/// The walk of a raw index's file counts and names what the map would hold:
/// an id held twice once and judged by its last copy, an id the mappings do
/// not hold counted.
#[test]
fn the_walk_counts_and_names_what_the_map_would_hold() {
    let clean = vec![1.0f32, 0.0];
    let poisoned = vec![f32::NAN, 0.0];
    let cases: Vec<(Entries, usize)> = vec![
        (
            vec![
                ("a", poisoned.clone()),
                ("b", vec![1.0, 2.0]),
                ("c", vec![f32::INFINITY, 1.0]),
            ],
            3,
        ),
        (vec![("a", clean.clone()), ("b", clean.clone())], 2),
        (
            vec![
                ("a", clean.clone()),
                ("b", clean.clone()),
                ("a", clean.clone()),
            ],
            2,
        ),
        (
            vec![
                ("a", poisoned.clone()),
                ("b", clean.clone()),
                ("a", clean.clone()),
            ],
            2,
        ),
        (
            vec![
                ("a", clean.clone()),
                ("b", clean.clone()),
                ("a", poisoned.clone()),
            ],
            2,
        ),
        (
            vec![
                ("a", poisoned.clone()),
                ("b", clean.clone()),
                ("a", poisoned.clone()),
            ],
            2,
        ),
        (
            vec![
                ("a", clean.clone()),
                ("z", clean.clone()),
                ("z", poisoned.clone()),
                ("z", clean.clone()),
            ],
            1,
        ),
    ];
    for (entries, records) in cases {
        let bytes = old::vectors(&entries);
        let map = legacy::read_vectors(&bytes, "vectors.bin", records).unwrap();
        let (count, offenders) =
            legacy::walk_vectors(&bytes, "vectors.bin", records, &mut |_, _| {}).unwrap();
        assert_eq!(count, map.len(), "{entries:?}");
        let named = match check_vectors_are_finite(&map) {
            Ok(()) => vec![],
            Err(Error::VectorsNotFinite { offenders, total }) => {
                assert_eq!(total, count);
                offenders
            }
            Err(other) => panic!("{other:?}"),
        };
        assert_eq!(offenders, named, "{entries:?}");
    }
}

/// Every prefix of a file a release wrote is refused, every damaged copy is
/// read or refused, nothing panics, and the walk refuses what the map refuses
/// in the same words.
#[test]
fn every_prefix_and_damaged_copy_of_an_old_file_is_read_or_refused() {
    let names: Vec<String> = (0..40).map(|i| format!("r{i}")).collect();
    let forward: Vec<(&str, usize)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i + 1))
        .collect();
    let reverse: Vec<(usize, &str)> = forward.iter().map(|&(n, i)| (i, n)).collect();
    let entries: Vec<(&str, Vec<f32>)> = names
        .iter()
        .map(|n| (n.as_str(), vec![0.5, f32::NAN]))
        .collect();
    let codes: Vec<(&str, Vec<u8>)> = names.iter().map(|n| (n.as_str(), vec![1, 2, 3])).collect();
    let files = [
        ("mappings", old::mappings(&forward, &reverse)),
        ("vectors", old::vectors(&entries)),
        ("codes", old::codes(&codes)),
        (
            "codebook",
            old::codebook(&vec![vec![vec![0.5f32; 2]; 4]; 3]),
        ),
    ];
    let read = |kind: &str, bytes: &[u8]| -> Result<(), Error> {
        match kind {
            "mappings" => legacy::read_mappings(bytes, "f").map(|_| ()),
            "vectors" => {
                let map = legacy::read_vectors(bytes, "f", 40).map(|m| m.len());
                let walk = legacy::walk_vectors(bytes, "f", 40, &mut |_, _| {}).map(|(c, _)| c);
                match (&map, &walk) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b),
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    _ => panic!("the map gave {map:?} and the walk {walk:?}"),
                }
                map.map(|_| ())
            }
            "codes" => legacy::read_codes(bytes, "f", 40).map(|_| ()),
            _ => legacy::read_codebook(bytes, "f", shape(3, 4, 2)).map(|_| ()),
        }
    };
    let mut inputs = 0usize;
    for (kind, bytes) in &files {
        assert!(read(kind, bytes).is_ok(), "{kind} whole");
        for len in 0..bytes.len() {
            assert!(read(kind, &bytes[..len]).is_err(), "{kind} prefix of {len}");
            inputs += 1;
        }
        for at in 0..bytes.len() {
            for value in [
                0x00,
                0x01,
                0xfa,
                0xfb,
                0xfc,
                0xfd,
                0xfe,
                0xff,
                bytes[at] ^ 0x80,
            ] {
                let mut damaged = bytes.clone();
                damaged[at] = value;
                let _ = read(kind, &damaged);
                inputs += 1;
            }
            let mut removed = bytes.clone();
            removed.remove(at);
            let _ = read(kind, &removed);
            let mut inserted = bytes.clone();
            inserted.insert(at, 0xfd);
            let _ = read(kind, &inserted);
            inputs += 2;
        }
    }
    assert!(inputs > 10_000, "{inputs}");
}

/// Bytes after the last entry are refused on each of the four, naming where
/// they start, which the old decode left unread.
#[test]
fn bytes_after_the_last_entry_are_refused() {
    let mut mappings = old::mappings(&[("a", 1)], &[(1, "a")]);
    mappings.push(0);
    assert_eq!(
        refused(legacy::read_mappings(&mappings, "mappings.bin")),
        "Failed to deserialize mappings.bin: the file continues past its last entry, from byte 8 to byte 9"
    );
    let mut vectors = old::vectors(&[("a", vec![1.0])]);
    vectors.extend_from_slice(&[0, 0]);
    assert!(refused(legacy::read_vectors(&vectors, "vectors.bin", 1))
        .ends_with("from byte 8 to byte 10"));
    assert!(refused(legacy::walk_vectors(
        &vectors,
        "vectors.bin",
        1,
        &mut |_, _| {}
    ))
    .ends_with("from byte 8 to byte 10"));
    let mut codes = old::codes(&[("a", vec![1])]);
    codes.push(9);
    assert!(
        refused(legacy::read_codes(&codes, "pq_codes.bin", 1)).ends_with("from byte 5 to byte 6")
    );
    let mut codebook = old::codebook(&[vec![vec![1.0]]]);
    codebook.push(0);
    assert!(refused(legacy::read_codebook(
        &codebook,
        "pq_centroids.bin",
        shape(1, 1, 1)
    ))
    .ends_with("from byte 7 to byte 8"));
}

/// A length its bytes cannot carry is refused before anything is sized from
/// it, on every container of the four. An allocation sized from one of these
/// does not unwind, so this test surviving is what it checks.
#[test]
fn a_length_its_bytes_cannot_carry_is_refused_before_anything_is_sized() {
    let huge = 1u64 << 40;
    let head = |parts: &[&[u8]]| -> Vec<u8> { parts.concat() };
    let v = |value: u64| {
        let mut out = Vec::new();
        old::varint(&mut out, value);
        out
    };
    let a = {
        let mut out = Vec::new();
        old::str(&mut out, "a");
        out
    };
    let exceeded =
        |result: Result<(), Error>| matches!(result, Err(Error::DecodeLengthExceeded { .. }));
    let pad = [0u8; 16];
    assert!(exceeded(
        legacy::read_mappings(&head(&[&v(huge), &pad]), "m").map(|_| ())
    ));
    assert!(exceeded(
        legacy::read_mappings(&head(&[&v(1), &v(huge), &pad]), "m").map(|_| ())
    ));
    assert!(exceeded(
        legacy::read_mappings(&head(&[&v(0), &v(huge), &pad]), "m").map(|_| ())
    ));
    assert!(exceeded(
        legacy::read_mappings(&head(&[&v(0), &v(1), &v(0), &v(huge), &pad]), "m").map(|_| ())
    ));
    for records in [0, usize::MAX] {
        assert!(exceeded(
            legacy::read_vectors(&head(&[&v(huge), &pad]), "v", records).map(|_| ())
        ));
        assert!(exceeded(
            legacy::read_vectors(&head(&[&v(1), &a, &v(huge), &pad]), "v", records).map(|_| ())
        ));
        assert!(exceeded(
            legacy::walk_vectors(&head(&[&v(huge), &pad]), "v", records, &mut |_, _| {})
                .map(|_| ())
        ));
        assert!(exceeded(
            legacy::walk_vectors(
                &head(&[&v(1), &a, &v(huge), &pad]),
                "v",
                records,
                &mut |_, _| {}
            )
            .map(|_| ())
        ));
        assert!(exceeded(
            legacy::read_codes(&head(&[&v(huge), &pad]), "c", records).map(|_| ())
        ));
        assert!(exceeded(
            legacy::read_codes(&head(&[&v(1), &a, &v(huge), &pad]), "c", records).map(|_| ())
        ));
    }
    let s = shape(2, 2, 2);
    assert!(exceeded(
        legacy::read_codebook(&head(&[&v(huge), &pad]), "b", s).map(|_| ())
    ));
    assert!(exceeded(
        legacy::read_codebook(&head(&[&v(1), &v(huge), &pad]), "b", s).map(|_| ())
    ));
    assert!(exceeded(
        legacy::read_codebook(&head(&[&v(1), &v(1), &v(huge), &pad]), "b", s).map(|_| ())
    ));
    assert_eq!(
        refused(legacy::read_mappings(
            &head(&[&v(huge), &pad]),
            "mappings.bin"
        )),
        Error::DecodeLengthExceeded {
            file: "mappings.bin".to_string(),
            bytes: 25
        }
        .to_string()
    );
    assert!(
        refused(legacy::read_vectors(&[254, 0, 0], "vectors.bin", 0))
            .ends_with("byte 0 is the length marker 254, which this format does not write")
    );
    assert!(
        refused(legacy::read_codes(&[1, 1, b'a', 255], "pq_codes.bin", 0))
            .ends_with("byte 3 is the length marker 255, which this format does not write")
    );
    assert!(refused(legacy::read_vectors(
        &[1, 1, b'a', 253, 0, 0],
        "vectors.bin",
        0
    ))
    .ends_with("the file ends at byte 6 and the value at byte 4 runs to byte 12"));
    assert!(refused(legacy::read_codes(
        &[1, 2, 0xc3, 0x28, 0],
        "pq_codes.bin",
        0
    ))
    .contains("the string at byte 2 is not UTF-8"));
}

/// An old codebook that is not the shape `quantization.json` describes is
/// refused in the words the install gives, before it is built, whether it is
/// of another size or ragged.
#[test]
fn an_old_codebook_of_another_shape_is_refused_before_it_is_built() {
    let say = |codebook: &[Vec<Vec<f32>>]| {
        refused(legacy::read_codebook(
            &old::codebook(codebook),
            "pq_centroids.bin",
            shape(2, 2, 2),
        ))
    };
    assert_eq!(say(&vec![vec![vec![1.0; 2]; 2]; 2]), "read");
    let mismatch = |actual| {
        Error::CodebookShapeMismatch {
            actual,
            expected: (2, 2, 2),
            subvectors: 2,
            bits: 1,
        }
        .to_string()
    };
    assert_eq!(say(&vec![vec![vec![1.0; 2]; 2]; 3]), mismatch((3, 2, 2)));
    assert_eq!(
        say(&[vec![vec![1.0; 2]; 2], vec![vec![1.0; 2]; 3]]),
        mismatch((2, 2, 2))
    );
    assert_eq!(
        say(&[vec![vec![1.0; 2], vec![1.0; 3]], vec![vec![1.0; 2]; 2]]),
        mismatch((2, 2, 2))
    );
    assert_eq!(say(&[]), mismatch((0, 0, 0)));
    assert_eq!(say(&[vec![], vec![]]), mismatch((2, 0, 0)));
}

/// The distinct keys the forward map carries are counted before it is
/// built. A prefix counts the entries it holds whole, a key held twice
/// counts once, a count the bytes do not carry counts what they do, and the
/// count stops where an entry does not read.
#[test]
fn the_distinct_keys_are_counted_before_the_forward_map_is_built() {
    let names: Vec<String> = (0..300).map(|i| format!("r{i}")).collect();
    let pairs: Vec<(&str, usize)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let bytes = old::mappings(&pairs, &[]);
    assert_eq!(legacy::count_distinct_keys(&bytes), 300);
    let twice = old::mappings(&[("a", 1), ("b", 2), ("a", 3)], &[]);
    assert_eq!(legacy::count_distinct_keys(&twice), 2);
    let mut declared = Vec::new();
    old::varint(&mut declared, 1 << 40);
    declared.extend_from_slice(&twice[1..twice.len() - 1]);
    assert_eq!(legacy::count_distinct_keys(&declared), 2);
    let mut repeated = Vec::new();
    old::varint(&mut repeated, 1 << 20);
    repeated.extend(std::iter::repeat_n([0u8, 0], 1 << 20).flatten());
    assert_eq!(legacy::count_distinct_keys(&repeated), 1);
    let marker = [2u8, 2, b'a', b'b', 7, 254, 1, b'c', 0];
    assert_eq!(legacy::count_distinct_keys(&marker), 1);
    assert_eq!(legacy::count_distinct_keys(&[]), 0);
    assert_eq!(legacy::count_distinct_keys(&[0]), 0);
}

// ============================================================================
// THE ID STORE
// ============================================================================

/// An internal id above the counter `config.json` records is refused before
/// the store is reserved, in each layout's words, and a sparse id space at or
/// below the counter, which removals without a compaction leave, is taken
/// whole.
#[test]
fn a_mappings_id_above_the_counter_is_refused_before_the_store_is_reserved() {
    let framed_file = |records: &[(usize, &str)]| {
        let text = records.iter().map(|(_, id)| id.len()).sum();
        framed::write_mappings(records.iter().copied(), records.len(), text).unwrap()
    };
    assert_eq!(
        refused(framed_ids(&framed_file(&[(1, "a"), (1 << 20, "b")]), &config(3))),
        "Failed to parse mappings.bin: the record 'b' holds internal id 1048576 and config.json counted 3"
    );
    assert_eq!(
        refused(bincode_ids(&old::mappings(&[("a", 1), ("b", 1 << 20)], &[(1, "a"), (1 << 20, "b")]), &config(3))),
        "Failed to parse mappings.bin: the forward map names internal id 1048576 for 'b' and config.json counted 3"
    );
    for counter in [1000, 4000] {
        let records = [(1, "a"), (500, "b"), (1000, "c")];
        let framed_store = framed_ids(&framed_file(&records), &config(counter)).unwrap();
        let forward: Vec<(&str, usize)> = records.iter().map(|&(i, id)| (id, i)).collect();
        let old_store = bincode_ids(&old::mappings(&forward, &records), &config(counter)).unwrap();
        for built in [framed_store, old_store] {
            assert_eq!((built.len(), built.slot_of("c")), (3, Some(1000)));
        }
    }
}

/// An old file whose two maps disagree describes two record sets and is
/// refused in the words the loader gave it.
#[test]
fn an_old_mappings_whose_two_maps_disagree_is_refused() {
    let say = |forward: &[(&str, usize)], reverse: &[(usize, &str)]| {
        refused(bincode_ids(&old::mappings(forward, reverse), &config(10)))
    };
    let two = [("a", 1), ("b", 2)];
    assert!(say(&two, &[(1, "a"), (2, "c")])
        .ends_with("the reverse map names internal id 2 as 'c' and the forward map does not"));
    assert!(
        say(&two, &[(1, "a")]).ends_with("the forward map holds 2 records and the reverse map 1")
    );
    assert!(say(&[("a", 1), ("b", 1)], &[(1, "b")])
        .ends_with("the forward map names 2 records under 1 internal ids"));
}

// ============================================================================
// WHOLE DIRECTORIES
// ============================================================================

/// A directory a release before 4.0.0 wrote opens with every record, every
/// internal id, every vector's bits, every field and every page it held, in
/// each storage mode, and saving it again writes the framed artefacts and
/// the graph dump the original save wrote, byte for byte.
#[test]
fn a_directory_written_before_format_4_opens_with_everything_it_held() {
    let temp = TempDir::new();
    let vectors = clustered(8, 8, 0x0018_2002);
    for (label, mode, version) in [
        ("raw", None, "1.1.0"),
        ("qraw", Some(StorageMode::QuantizedWithRaw), "2.0.0"),
        ("qonly", Some(StorageMode::QuantizedOnly), "3.1.0"),
    ] {
        let collection = filled(mode.clone(), 1050);
        assert_eq!(collection.is_quantized(), mode.is_some(), "{label}");
        let framed_dir = temp.at(&format!("{label}.zdb"));
        collection.save(framed_dir.to_str().unwrap()).unwrap();
        let old_dir = temp.at(&format!("{label}-old.zdb"));
        collection.save(old_dir.to_str().unwrap()).unwrap();
        as_old(&old_dir, version);
        assert_eq!(
            read_json(&old_dir.join("manifest.json"))["format_version"],
            json!(version)
        );

        let from_old = Collection::load(old_dir.to_str().unwrap()).unwrap();
        assert_eq!(everything(&from_old), everything(&collection), "{label}");
        for query in &vectors {
            assert_eq!(page(&from_old, query), page(&collection, query), "{label}");
        }
        let again = temp.at(&format!("{label}-again.zdb"));
        from_old.save(again.to_str().unwrap()).unwrap();
        for name in [
            "mappings.bin",
            "vectors.bin",
            "pq_codes.bin",
            "pq_centroids.bin",
            "hnsw_index.zdbgraph",
        ] {
            assert_eq!(
                std::fs::read(framed_dir.join(name)).ok(),
                std::fs::read(again.join(name)).ok(),
                "{label} {name}"
            );
        }
    }
}

/// A load files every record at its place among the internal ids the
/// mappings hold, counting from zero, whether `mappings.bin` is framed, and
/// so lists its records in increasing internal id, or was written before
/// format 4.0.0, and so lists them in a hash map's order. The ids are
/// scattered by removals and overwrites, and the names do not follow them.
#[test]
fn a_load_files_records_in_internal_id_order_from_either_layout() {
    let temp = TempDir::new();
    let collection = filled(None, 600);
    let gone: Vec<String> = (0..600)
        .filter(|i| i % 4 != 1)
        .map(|i| format!("r{i}\u{e9}"))
        .collect();
    assert!(collection.remove_points(&gone).unwrap().is_empty());
    let vectors = clustered(60, 8, 0x0018_2003);
    let overwrites: Vec<ParsedRecord> = (0..60)
        .map(|i| record(&format!("r{}\u{e9}", 593 - 8 * i), &vectors[i], "z", -1))
        .collect();
    let added = collection.add_records(overwrites, vec![], true);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    let expected = collection.internal_ids();
    assert!(expected.windows(2).all(|pair| pair[0].0 < pair[1].0));
    let names: Vec<String> = expected.iter().map(|(_, name)| name.clone()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_ne!(names, sorted, "the names do not follow the ids");

    for version in [None, Some("3.1.0")] {
        let path = temp.at(&format!("order-{}.zdb", version.unwrap_or("framed")));
        collection.save(path.to_str().unwrap()).unwrap();
        if let Some(version) = version {
            as_old(&path, version);
        }
        let loaded = Collection::load(path.to_str().unwrap()).unwrap();
        {
            let ids = loaded.ids();
            let filed: Vec<(usize, String)> =
                ids.iter().map(|(at, id)| (at, id.to_string())).collect();
            assert_eq!(
                filed,
                names.iter().cloned().enumerate().collect::<Vec<_>>(),
                "{version:?}: each record at its place in internal id order"
            );
        }
        assert_eq!(loaded.internal_ids(), expected, "{version:?}");
        assert!(loaded.positions_agree(), "{version:?}");
        assert_eq!(everything(&loaded), everything(&collection), "{version:?}");
        for query in &vectors[..8] {
            assert_eq!(
                page(&loaded, query),
                page(&collection, query),
                "{version:?}"
            );
        }
    }
}

/// A frame is read wherever it is found, so a directory this build saved,
/// written again as one before format 5.0.0 and relabelled to an earlier
/// major, opens. Bincode's wire is read only below 4.x, so an old directory
/// relabelled 4.0.0 is refused by the frame.
#[test]
fn the_layout_follows_mappings_bin_below_4_and_is_the_frame_from_4() {
    let temp = TempDir::new();
    let collection = filled(None, 60);
    let path = temp.at("relabelled.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    super::older::rewrite_before_five(&path, "1.1.0");
    for version in ["1.1.0", "2.0.0", "3.0.0"] {
        let mut manifest = read_json(&path.join("manifest.json"));
        manifest["format_version"] = json!(version);
        write_json(&path.join("manifest.json"), &manifest);
        let loaded = Collection::load(path.to_str().unwrap()).unwrap();
        assert_eq!(everything(&loaded), everything(&collection), "{version}");
    }
    as_old(&path, "4.0.0");
    let message = refused(Collection::load(path.to_str().unwrap()));
    assert!(
        message.starts_with("Failed to deserialize mappings.bin: the file"),
        "{message}"
    );
    assert!(message.contains("frame"), "{message}");
}

/// A format version is read only as `MAJOR.MINOR.PATCH`. Every version a
/// release wrote opens, being 1.0.0 to 4.0.0 and the 5.0.0 this build writes,
/// and so does a later minor or patch under any of those majors. A major this
/// build does not read is refused by its major, and every other shape is
/// refused as a version this build cannot interpret. A version below 5.0.0
/// is read from a directory written again as one before format 5.0.0.
#[test]
fn a_format_version_is_read_only_as_major_minor_patch() {
    let temp = TempDir::new();
    let collection = filled(None, 60);
    let path = temp.at("versioned.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let before_five = temp.at("versioned-before-five.zdb");
    collection.save(before_five.to_str().unwrap()).unwrap();
    super::older::rewrite_before_five(&before_five, "4.0.0");
    let saved = read_json(&path.join("manifest.json"));
    let saved_before_five = read_json(&before_five.join("manifest.json"));
    let load = |version: &str| {
        let below_five = version
            .split('.')
            .next()
            .and_then(|major| major.parse::<u32>().ok())
            .is_some_and(|major| (1..5).contains(&major));
        let (path, mut manifest) = if below_five {
            (&before_five, saved_before_five.clone())
        } else {
            (&path, saved.clone())
        };
        manifest["format_version"] = json!(version);
        // A 4.x save lists the identity among its features.
        if version.starts_with("4.") {
            manifest["features"]["identity"] = json!("compatible");
        }
        write_json(&path.join("manifest.json"), &manifest);
        Collection::load(path.to_str().unwrap())
    };

    for version in [
        "1.0.0", "1.1.0", "1.2.0", "2.0.0", "2.1.0", "3.0.0", "3.1.0", "4.0.0", "1.9.3", "2.3.0",
        "2.4.1", "3.2.5", "3.7.2", "4.7.1", "4.9.0", "5.0.0", "5.4.2",
    ] {
        let loaded = load(version).unwrap_or_else(|e| panic!("{version}: {e}"));
        assert_eq!(everything(&loaded), everything(&collection), "{version}");
    }

    for (version, newer) in [("6.0.0", true), ("0.9.0", false)] {
        match load(version) {
            Err(Error::FormatVersionUnsupported {
                format_version,
                newer: refused_as_newer,
                ..
            }) => assert_eq!(
                (format_version.as_str(), refused_as_newer),
                (version, newer)
            ),
            other => panic!("{version}: {:?}", other.map(|_| ())),
        }
    }

    for version in [
        "4",
        "4.banana",
        "+4.0.0",
        "04.0.0",
        "4.0.0-rc1",
        " 4.0.0",
        "abc",
        "",
        "banana",
        "4.0",
        "4.0.0.0",
        "4..0",
        "4.0.0 ",
        "4.00.0",
        "4.0.01",
        "4.+0.0",
        "\u{664}.0.0",
    ] {
        assert_eq!(
            refused(load(version)),
            format!(
                "manifest.json declares format_version '{version}', which is not a version \
                 this build can interpret. A ZeusDB index directory declares a dotted version \
                 such as 5.0.0."
            ),
            "{version:?}"
        );
    }
}

/// A framed artefact a hand assembled from another directory is refused by
/// name: codes under an internal id the mappings do not hold, in a directory
/// before format 5.0.0, which holds `pq_codes.bin`, and vectors short of a
/// record.
#[test]
fn a_framed_artefact_that_disagrees_with_the_mappings_is_refused() {
    let temp = TempDir::new();
    let collection = filled(Some(StorageMode::QuantizedWithRaw), 1050);
    let path = temp.at("spliced.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let highest = collection.internal_ids().last().unwrap().0;

    let codes_path = temp.at("codes.zdb");
    collection.save(codes_path.to_str().unwrap()).unwrap();
    super::older::rewrite_before_five(&codes_path, "4.0.0");
    let (codes, _) = framed::write_codes(4, [(highest + 7, &[1u8, 2, 3, 4][..])], 1).unwrap();
    replace(&codes_path, "pq_codes.bin", &codes);
    assert_eq!(
        refused(Collection::load(codes_path.to_str().unwrap())),
        format!(
            "Failed to deserialize pq_codes.bin: names internal id {}, which mappings.bin does not hold",
            highest + 7
        )
    );

    let ids = collection.internal_ids();
    let first: Vec<(usize, Vec<f32>)> = ids
        .iter()
        .take(3)
        .map(|&(i, _)| (i, vec![0.5; 8]))
        .collect();
    let (vectors, _) =
        framed::write_vectors(8, first.iter().map(|(i, v)| (*i, v.as_slice())), 3).unwrap();
    drop(ids);
    replace(&path, "vectors.bin", &vectors);
    let message = refused(Collection::load(path.to_str().unwrap()));
    assert!(
        message.starts_with("Failed to deserialize vectors.bin: holds 3 vectors and mappings.bin holds 1050 records; record '"),
        "{message}"
    );
}

// ============================================================================
// THE MANIFEST AT 5.x
// ============================================================================

/// Write `text` as the JSON artefact `name` and record its length and digest
/// in `manifest`, with the directory size, as a save records them.
fn record_json(path: &Path, name: &str, text: &str, manifest: &mut Value) {
    std::fs::write(path.join(name), text).unwrap();
    manifest["file_digests"][name] = json!({
        "bytes": text.len(),
        "checksum": format!("{:016x}", checksum_of(text.as_bytes())),
    });
    resize(manifest);
}

/// The directory size a manifest records, from the lengths it records.
fn resize(manifest: &mut Value) {
    let total: u64 = manifest["file_digests"]
        .as_object()
        .unwrap()
        .values()
        .map(|digest| digest["bytes"].as_u64().unwrap())
        .sum();
    manifest["total_size_mb"] = json!(total as f64 / (1024.0 * 1024.0));
}

/// A copy of `source` beside it whose manifest `edit` changes, and what a
/// load of it gives.
fn load_edited(
    source: &Path,
    name: &str,
    edit: impl FnOnce(&Path, &mut Value),
) -> Result<Collection, Error> {
    let target = source.with_file_name(name);
    let _ = std::fs::remove_dir_all(&target);
    std::fs::create_dir_all(&target).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
    }
    let mut manifest = read_json(&target.join("manifest.json"));
    edit(&target, &mut manifest);
    write_json(&target.join("manifest.json"), &manifest);
    Collection::load(target.to_str().unwrap())
}

/// The detail of a `ManifestFieldInvalid`.
fn field_invalid(result: Result<Collection, Error>, label: &str) -> String {
    match result {
        Err(Error::ManifestFieldInvalid { detail }) => detail,
        Err(other) => panic!("{label}: expected the manifest refused, got {other}"),
        Ok(_) => panic!("{label}: expected the manifest refused, and the directory opened"),
    }
}

/// An edit of a manifest's JSON.
type ManifestEdit = Box<dyn Fn(&mut Value)>;

/// Each field of a 5.x manifest a save writes from the directory is held to
/// it, one refusal a check, and the same edits on a directory before format
/// 5.0.0 load as they did.
#[test]
fn every_field_of_a_5_x_manifest_is_held_to_the_directory() {
    let temp = TempDir::new();
    let raw = temp.at("raw.zdb");
    let raw_collection = filled(None, 60);
    raw_collection.save(raw.to_str().unwrap()).unwrap();
    // A whole vectors.bin over the raw collection's records, as a release
    // before 5.0.0 wrote it.
    let (raw_vectors, _) = {
        let ids = raw_collection.internal_ids();
        let views = raw_collection
            .records(ids.iter().map(|(_, id)| id.clone()).collect(), true, true)
            .unwrap();
        let rows: Vec<(usize, Vec<f32>)> = ids
            .iter()
            .zip(views)
            .map(|((internal_id, _), view)| (*internal_id, view.vector.unwrap()))
            .collect();
        framed::write_vectors(8, rows.iter().map(|(i, v)| (*i, v.as_slice())), rows.len()).unwrap()
    };
    let quantized = temp.at("quantized.zdb");
    filled(Some(StorageMode::QuantizedOnly), 1050)
        .save(quantized.to_str().unwrap())
        .unwrap();
    let total = |manifest: &Value| {
        manifest["file_digests"]
            .as_object()
            .unwrap()
            .values()
            .map(|digest| digest["bytes"].as_u64().unwrap())
            .sum::<u64>()
    };

    // The fields read before any artefact.
    let cases: Vec<(&str, ManifestEdit, String)> = vec![
        (
            "version",
            Box::new(|m: &mut Value| m["zeusdb_version"] = json!("")),
            "records zeusdb_version as an empty string, and a save records the release that \
             wrote it"
                .to_string(),
        ),
        (
            "created",
            Box::new(|m: &mut Value| m["created_at"] = json!("yesterday")),
            "records created_at as 'yesterday', which is not an RFC 3339 time".to_string(),
        ),
        (
            "saved",
            Box::new(|m: &mut Value| m["saved_at"] = json!("2026-13-01T00:00:00Z")),
            "records saved_at as '2026-13-01T00:00:00Z', which is not an RFC 3339 time".to_string(),
        ),
        (
            "type",
            Box::new(|m: &mut Value| m["index_type"] = json!("IVF")),
            "records index_type as 'IVF', and a save records HNSW".to_string(),
        ),
        (
            "mode",
            Box::new(|m: &mut Value| m["storage_mode"] = json!("compressed")),
            "records storage_mode as 'compressed', which is not one a save records".to_string(),
        ),
        (
            "has",
            Box::new(|m: &mut Value| m["has_quantization"] = json!(true)),
            "records has_quantization as true, and files_included does not name \
             quantization.json"
                .to_string(),
        ),
        (
            "excluded",
            Box::new(|m: &mut Value| m["files_excluded"] = json!(["config.json"])),
            "names config.json under both files_included and files_excluded".to_string(),
        ),
        (
            "size",
            Box::new(|m: &mut Value| m["total_size_mb"] = json!(1.25)),
            String::new(),
        ),
        (
            "vectors",
            Box::new(|m: &mut Value| m["total_vectors"] = json!(59)),
            "records total_vectors as 59, and config.json records vector_count 60".to_string(),
        ),
        (
            "trained",
            Box::new(|m: &mut Value| m["quantization_trained"] = json!(true)),
            "records quantization_trained as true, and files_included names no \
             quantization.json"
                .to_string(),
        ),
    ];
    let older = temp.at("raw-before-five.zdb");
    filled(None, 60).save(older.to_str().unwrap()).unwrap();
    rewrite_before_five(&older, "4.0.0");
    for (label, edit, expected) in &cases {
        let detail = field_invalid(
            load_edited(&raw, &format!("raw-{label}.zdb"), |_, m| edit(m)),
            label,
        );
        if *label == "size" {
            let recorded = total(&read_json(&raw.join("manifest.json")));
            assert_eq!(
                detail,
                format!(
                    "records total_size_mb as 1.25, and the lengths file_digests records add up \
                     to {} MiB",
                    recorded as f64 / (1024.0 * 1024.0)
                ),
                "{label}"
            );
        } else {
            assert_eq!(&detail, expected, "{label}");
        }
        // The same edit before format 5.0.0 is not read.
        assert!(
            load_edited(&older, &format!("older-{label}.zdb"), |_, m| edit(m)).is_ok(),
            "{label}: a 4.0.0 directory refused"
        );
    }

    // The words of the refusal, whole.
    let message = load_edited(&raw, "raw-words.zdb", |_, m| m["total_vectors"] = json!(12))
        .err()
        .unwrap()
        .to_string();
    assert_eq!(
        message,
        "manifest.json records total_vectors as 12, and config.json records vector_count 60. A \
         save writes every field of manifest.json from the directory it describes, so this \
         manifest does not describe the directory. Restore the directory from a copy."
    );

    // files_included and file_digests name the same files, and none of them
    // one no 5.0.0 save writes.
    let detail = field_invalid(
        load_edited(&raw, "raw-undigested.zdb", |_, m| {
            m["files_included"]
                .as_array_mut()
                .unwrap()
                .push(json!("extra.bin"))
        }),
        "undigested",
    );
    assert_eq!(
        detail,
        "names extra.bin under files_included, and file_digests records no length for it"
    );
    let detail = field_invalid(
        load_edited(&raw, "raw-unlisted.zdb", |_, m| {
            m["file_digests"]["extra.bin"] = json!({ "bytes": 0 });
        }),
        "unlisted",
    );
    assert_eq!(
        detail,
        "records a length for extra.bin under file_digests, and files_included does not name it"
    );
    for name in [
        "pq_codes.bin",
        "int8_rows.zdbint8",
        "hnsw_index.hnsw.graph",
        "hnsw_index.hnsw.data",
    ] {
        let detail = field_invalid(
            load_edited(&raw, &format!("raw-{name}"), |target, m| {
                std::fs::write(target.join(name), [0u8; 8]).unwrap();
                m["files_included"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!(name));
                m["file_digests"][name] = json!({ "bytes": 8 });
                resize(m);
            }),
            name,
        );
        assert_eq!(
            detail,
            format!("names {name} under files_included, which no save at format 5.0.0 writes")
        );
    }
    let detail = field_invalid(
        load_edited(&raw, "raw-held.zdb", |target, m| {
            std::fs::write(target.join("vectors.bin"), [0u8; 8]).unwrap();
            m["files_excluded"] = json!(["vectors.bin"]);
        }),
        "held",
    );
    assert_eq!(
        detail,
        "names vectors.bin under files_excluded, and the directory holds it"
    );
    let detail = field_invalid(
        load_edited(&raw, "raw-vectors.zdb", |target, m| {
            std::fs::write(target.join("vectors.bin"), &raw_vectors).unwrap();
            m["files_included"]
                .as_array_mut()
                .unwrap()
                .push(json!("vectors.bin"));
            m["files_excluded"] = json!([]);
            m["file_digests"]["vectors.bin"] = json!({ "bytes": raw_vectors.len() });
            resize(m);
        }),
        "vectors",
    );
    assert_eq!(
        detail,
        "names vectors.bin under files_included, and the graph keeps no raw vectors beside its \
         rows"
    );

    // A trained quantizer, recorded as untrained.
    let detail = field_invalid(
        load_edited(&quantized, "quantized-trained.zdb", |_, m| {
            m["quantization_trained"] = json!(false)
        }),
        "untrained",
    );
    assert_eq!(
        detail,
        "records quantization_trained as false, and quantization.json records is_trained true"
    );

    // The untouched directories open.
    assert!(Collection::load(raw.to_str().unwrap()).is_ok());
    assert!(Collection::load(quantized.to_str().unwrap()).is_ok());
}

// ============================================================================
// THE PRODUCT QUANTIZED FIELDS OF QUANTIZATION.JSON
// ============================================================================

/// `training_size` below 1,000 and `max_training_vectors` below it are
/// refused as `create()` refuses them, at 5.0.0 and before it, and a file
/// still holding `memory_stats` and `pq_config` is read with the two
/// ignored.
#[test]
fn the_product_quantized_training_fields_are_held_to_create() {
    let temp = TempDir::new();
    let collection = filled(Some(StorageMode::QuantizedOnly), 1050);
    let current = temp.at("current.zdb");
    collection.save(current.to_str().unwrap()).unwrap();
    let older = temp.at("older.zdb");
    collection.save(older.to_str().unwrap()).unwrap();
    rewrite_before_five(&older, "4.0.0");

    for (label, source) in [("current", &current), ("older", &older)] {
        let edited = |name: &str, edit: &dyn Fn(&mut Value)| {
            load_edited(source, &format!("{label}-{name}.zdb"), |target, m| {
                let mut quantization = read_json(&target.join("quantization.json"));
                edit(&mut quantization);
                let text = serde_json::to_string_pretty(&quantization).unwrap();
                record_json(target, "quantization.json", &text, m);
            })
        };
        match edited("small", &|q| q["training_size"] = json!(999)) {
            Err(Error::QuantizationFieldInvalidInFile { file, detail }) => {
                assert!(file.ends_with("quantization.json"), "{label}: {file}");
                assert_eq!(
                    detail, "training_size is 999, and create() requires at least 1000",
                    "{label}"
                );
            }
            other => panic!("{label}: expected a refusal, got {:?}", other.map(|_| ())),
        }
        let message = edited("short", &|q| {
            q["training_size"] = json!(1000);
            q["max_training_vectors"] = json!(900);
        })
        .err()
        .unwrap();
        assert!(matches!(
            message.exception(),
            zeusdb_vector_core::Exception::Value
        ));
        let text = message.to_string();
        assert!(
            text.ends_with(
                "quantization.json: max_training_vectors is 900 and training_size is 1000, \
                 and create() requires the first to be at or above the second, so this file \
                 does not describe an index this build can rebuild."
            ),
            "{label}: {text}"
        );
        let loaded = edited("stats", &|q| {
            q["memory_stats"] = json!({
                "centroid_storage_mb": 0.0,
                "compression_ratio": 8.0,
                "centroids_per_subvector": 16,
                "total_centroids": 64,
            });
            q["pq_config"] = json!({ "dim": 8, "sub_dim": 2, "num_centroids": 16 });
        })
        .unwrap();
        assert_eq!(everything(&loaded), everything(&collection), "{label}");
    }
    let written = read_json(&current.join("quantization.json"));
    assert!(written.get("memory_stats").is_none());
    assert!(written.get("pq_config").is_none());
    let manifest = read_json(&current.join("manifest.json"));
    assert!(manifest.get("compression_info").is_none());
}

// ============================================================================
// COMPACT METADATA.JSON
// ============================================================================

/// `metadata.json` at 5.0.0 is compact and names only the records that hold
/// a field, in increasing internal id. A load gives every record it leaves
/// out an empty entry, so the collection holds what the saved one held.
#[test]
fn compact_metadata_names_only_the_records_holding_a_field() {
    let temp = TempDir::new();
    let collection = filled(None, 30);
    let bare: Vec<ParsedRecord> = (0..5)
        .map(|i| ParsedRecord {
            id: format!("bare{i}"),
            vector: vec![0.1 * i as f32; 8],
            sparse: None,
            metadata: HashMap::new(),
        })
        .collect();
    assert_eq!(collection.add_records(bare, vec![], false).total_errors, 0);
    let path = temp.at("compact.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(path.join("metadata.json")).unwrap();
    assert!(!text.contains('\n'), "{text}");
    let parsed: serde_json::Map<String, Value> = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.len(), 30);
    assert!(parsed.keys().all(|id| id.starts_with('r')));
    let positions: Vec<usize> = (0..30)
        .map(|i| text.find(&format!("\"r{i}\u{e9}\":")).unwrap())
        .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));

    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(everything(&loaded), everything(&collection));
    assert_eq!(loaded.vector_metadata().len(), 35);
}

/// A compact `metadata.json` naming a record the mappings do not hold, or
/// one record twice, is refused naming the record. Before format 5.0.0 an
/// entry the mappings do not hold is dropped, as it was.
#[test]
fn compact_metadata_naming_a_record_not_held_or_held_twice_is_refused() {
    let temp = TempDir::new();
    let collection = filled(None, 20);
    let path = temp.at("named.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(path.join("metadata.json")).unwrap();

    let ghost = format!("{{\"ghost\":{{\"x\":1}},{}", &text[1..]);
    let result = load_edited(&path, "ghost.zdb", |target, m| {
        record_json(target, "metadata.json", &ghost, m)
    });
    match result {
        Err(Error::DecodeFailed { file, error }) => {
            assert_eq!(file, "metadata.json");
            assert_eq!(
                error,
                "names record 'ghost', which mappings.bin does not hold"
            );
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }

    let twice = format!("{{\"r0\u{e9}\":{{\"x\":1}},{}", &text[1..]);
    let result = load_edited(&path, "twice.zdb", |target, m| {
        record_json(target, "metadata.json", &twice, m)
    });
    match result {
        Err(Error::DecodeFailed { file, error }) => {
            assert_eq!(file, "metadata.json");
            assert_eq!(error, "names record 'r0\u{e9}' twice");
        }
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }

    let older = temp.at("older.zdb");
    collection.save(older.to_str().unwrap()).unwrap();
    rewrite_before_five(&older, "4.0.0");
    let old_text = std::fs::read_to_string(older.join("metadata.json")).unwrap();
    let ghost = format!("{{\"ghost\":{{\"x\":1}},{}", &old_text[1..]);
    let loaded = load_edited(&older, "older-ghost.zdb", |target, m| {
        record_json(target, "metadata.json", &ghost, m)
    })
    .unwrap();
    assert_eq!(everything(&loaded), everything(&collection));
}

// ============================================================================
// WHAT A LOAD HOLDS A DIRECTORY'S STORES AND COUNTS TO
// ============================================================================

/// `config.json` rewritten through `edit`, with its length and digest
/// recorded in the manifest, as a save records them.
fn edit_config(path: &Path, edit: impl FnOnce(&mut Value)) {
    let mut config = read_json(&path.join("config.json"));
    edit(&mut config);
    let mut manifest = read_json(&path.join("manifest.json"));
    record_json(
        path,
        "config.json",
        &serde_json::to_string_pretty(&config).unwrap(),
        &mut manifest,
    );
    write_json(&path.join("manifest.json"), &manifest);
}

/// The artefact `name` named among the files the manifest says the
/// directory holds.
fn include(path: &Path, name: &str) {
    let mut manifest = read_json(&path.join("manifest.json"));
    let included = manifest["files_included"].as_array_mut().unwrap();
    if !included.iter().any(|entry| entry == name) {
        included.push(json!(name));
    }
    write_json(&path.join("manifest.json"), &manifest);
}

/// A release before 0.12.0 whose id store was full issued ids past its last
/// slot and moved its counter on with each, so a directory it wrote, below
/// format 4.0.0, can record a counter past `u32::MAX`. Such a directory
/// opens with every record, its counter held to the last id a collection
/// issues, and refuses the next record as a collection at its last id
/// refuses one. From format 4.0.0 no release wrote such a counter, and it is
/// refused.
#[test]
fn a_counter_past_the_last_id_is_held_to_it_below_format_4_and_refused_from_it() {
    let temp = TempDir::new();
    let collection = filled(None, 10);
    let expected = everything(&collection);
    let past = u32::MAX as usize + 6;
    for version in ["1.1.0", "3.0.0", "4.0.0", "5.0.0"] {
        let path = temp.at(&format!("counter-{version}.zdb"));
        collection.save(path.to_str().unwrap()).unwrap();
        match version {
            "5.0.0" => {}
            "4.0.0" => rewrite_before_five(&path, version),
            _ => as_old(&path, version),
        }
        edit_config(&path, |config| config["id_counter"] = json!(past));
        let loaded = Collection::load(path.to_str().unwrap());
        if version < "4" {
            let loaded = loaded.unwrap_or_else(|e| panic!("{version}: {e}"));
            assert_eq!(everything(&loaded), expected, "{version}");
            assert_eq!(loaded.id_counter(), MAX_INTERNAL_ID, "{version}");
            let added = loaded.add_records(vec![record("next", &[0.5; 8], "a", 0)], vec![], false);
            assert_eq!(added.total_errors, 1, "{version}");
            assert!(
                added.errors[0].contains(&format!(
                    "This collection has issued its last internal id, {}.",
                    MAX_INTERNAL_ID
                )),
                "{version}: {:?}",
                added.errors
            );
        } else {
            let message = refused(loaded);
            assert!(
                message.contains(&format!(
                    "config.json: id_counter is {past}, and an internal id is a graph node index"
                )),
                "{version}: {message}"
            );
        }
    }
}

/// A restored count that is not the one config.json records is refused
/// with where the records the load counted came from, naming no file the
/// directory does not hold: every record from the graph dump at 5.0.0, the
/// graph rebuilt from vectors.bin with its count where the dump of an
/// earlier format is gone, and nothing to rebuild from where a save at 5.0.0
/// is relabelled 4.0.0, since its reader takes a version 1 dump and the
/// save wrote no vectors.bin.
#[test]
fn the_restored_count_refusal_names_where_the_records_came_from() {
    let temp = TempDir::new();
    let collection = filled(None, 10);
    let count_eleven = |path: &Path| {
        edit_config(path, |config| config["vector_count"] = json!(11));
        let mut manifest = read_json(&path.join("manifest.json"));
        manifest["total_vectors"] = json!(11);
        write_json(&path.join("manifest.json"), &manifest);
    };
    let words = |restored: usize, expected: usize, source: &str| {
        format!(
            "Restored record count does not match config.json: the directory yields {restored} \
             records while config.json reports {expected}. {source} Refusing to load an index \
             that would report a count it cannot produce; restore the directory from a copy."
        )
    };

    let at_five = temp.at("count-5.zdb");
    collection.save(at_five.to_str().unwrap()).unwrap();
    count_eleven(&at_five);
    assert_eq!(
        refused(Collection::load(at_five.to_str().unwrap())),
        words(10, 11, "Every record came from the graph dump.")
    );

    let dump_gone = temp.at("count-4.zdb");
    collection.save(dump_gone.to_str().unwrap()).unwrap();
    rewrite_before_five(&dump_gone, "4.0.0");
    std::fs::remove_file(dump_gone.join(GRAPH_DUMP_FILENAME)).unwrap();
    count_eleven(&dump_gone);
    let message = refused(Collection::load(dump_gone.to_str().unwrap()));
    let split = |restored: usize, expected: usize| {
        let whole = words(restored, expected, "The graph dump was not read, because |");
        let (head, tail) = whole.split_once('|').unwrap();
        (head.to_string(), tail.to_string())
    };
    let (head, tail) = split(10, 11);
    assert!(message.starts_with(&head), "{message}");
    assert!(
        message.ends_with(&format!(
            ", and the graph was rebuilt from the 10 vectors vectors.bin holds.{tail}"
        )),
        "{message}"
    );

    let relabelled = temp.at("count-relabelled.zdb");
    collection.save(relabelled.to_str().unwrap()).unwrap();
    let mut manifest = read_json(&relabelled.join("manifest.json"));
    manifest["format_version"] = json!("4.0.0");
    manifest["features"]["identity"] = json!("compatible");
    write_json(&relabelled.join("manifest.json"), &manifest);
    let message = refused(Collection::load(relabelled.to_str().unwrap()));
    let (head, tail) = split(0, 10);
    assert!(message.starts_with(&head), "{message}");
    assert!(
        message.ends_with(&format!(
            ", and the directory holds no vector, code or row to rebuild the graph from.{tail}"
        )),
        "{message}"
    );
    assert!(!message.contains("pq_codes.bin"), "{message}");
}

/// A directory whose quantization.json records no trained codebook and
/// whose pq_codes.bin holds a code is refused as it is read, below format
/// 4.0.0 and at it, with every vector kept, with the graph dump read and
/// with it gone, and at 1.1.0 with the coded record's vector and the dump
/// gone as well. At 4.0.0 a record with no vector is refused by the framed
/// vectors.bin first.
#[test]
fn codes_held_where_no_codebook_was_trained_are_refused() {
    let temp = TempDir::new();
    let collection = filled(Some(StorageMode::QuantizedWithRaw), 10);
    assert!(!collection.is_quantized());
    let r0 = collection
        .internal_ids()
        .into_iter()
        .find(|(_, id)| id == "r0\u{e9}")
        .unwrap()
        .0;
    let words = "pq_codes.bin holds codes for 1 record, and quantization.json records no \
                 trained codebook to decode them with. The two files disagree about whether \
                 this index was trained, so the directory cannot be loaded without dropping \
                 those codes.";
    for (version, dump_gone, vector_gone) in [
        ("1.1.0", false, false),
        ("1.1.0", true, true),
        ("4.0.0", false, false),
        ("4.0.0", true, false),
    ] {
        let label = format!("{version}, dump gone {dump_gone}, vector gone {vector_gone}");
        let path = temp.at(&format!("codes-{version}-{dump_gone}-{vector_gone}.zdb"));
        collection.save(path.to_str().unwrap()).unwrap();
        if version == "4.0.0" {
            rewrite_before_five(&path, version);
            let (codes, _) = framed::write_codes(4, [(r0, &[1u8, 2, 3, 4][..])], 1).unwrap();
            replace(&path, "pq_codes.bin", &codes);
        } else {
            as_old(&path, version);
            replace(
                &path,
                "pq_codes.bin",
                &old::codes(&[("r0\u{e9}", vec![1u8, 2, 3, 4])]),
            );
        }
        include(&path, "pq_codes.bin");
        if dump_gone {
            std::fs::remove_file(path.join(GRAPH_DUMP_FILENAME)).unwrap();
        }
        if vector_gone {
            let bytes = std::fs::read(path.join("vectors.bin")).unwrap();
            let map = legacy::read_vectors(&bytes, "v", 10).unwrap();
            let kept: Vec<(&str, Vec<f32>)> = map
                .iter()
                .filter(|(id, _)| id.as_str() != "r0\u{e9}")
                .map(|(id, vector)| (id.as_str(), vector.clone()))
                .collect();
            replace(&path, "vectors.bin", &old::vectors(&kept));
        }
        assert_eq!(
            refused(Collection::load(path.to_str().unwrap())),
            words,
            "{label}"
        );
    }
}

/// The entries of a vectors.bin of bincode's wire, as a test edits them.
type OldVectors = Vec<(String, Vec<f32>)>;

/// vectors.bin and pq_codes.bin written before format 4.0.0 name each
/// record by its id, and a load holds them to the mappings as it holds the
/// framed ones at 4.0.0, in their words: a record the mappings do not hold
/// and an entry of the wrong width are refused in each, with the graph dump
/// read and with it gone, and a record the mappings hold with no vector is
/// refused where the directory keeps every record's raw vector. A record
/// foreign to the mappings is refused where config.json counts it as well.
/// A trained `quantized_only` directory keeping only its training records'
/// vectors opens, holding what was saved.
#[test]
fn the_old_stores_are_held_to_the_mappings_as_the_framed_ones_are() {
    let temp = TempDir::new();
    let raw = filled(None, 10);
    let edited = |name: &str, gone: bool, edit: &dyn Fn(&mut OldVectors)| {
        let path = temp.at(name);
        raw.save(path.to_str().unwrap()).unwrap();
        as_old(&path, "1.1.0");
        let bytes = std::fs::read(path.join("vectors.bin")).unwrap();
        let mut entries: OldVectors = legacy::read_vectors(&bytes, "v", 10)
            .unwrap()
            .into_iter()
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        edit(&mut entries);
        let entries: Vec<(&str, Vec<f32>)> = entries
            .iter()
            .map(|(id, vector)| (id.as_str(), vector.clone()))
            .collect();
        replace(&path, "vectors.bin", &old::vectors(&entries));
        if gone {
            std::fs::remove_file(path.join(GRAPH_DUMP_FILENAME)).unwrap();
        }
        path
    };
    let foreign = |entries: &mut OldVectors| {
        entries.push(("zz".to_string(), vec![0.25; 8]));
    };
    let narrow = |entries: &mut OldVectors| {
        let at = entries.iter().position(|(id, _)| id == "r3\u{e9}").unwrap();
        entries[at].1.pop();
    };
    let missing = |entries: &mut OldVectors| {
        entries.retain(|(id, _)| id != "r3\u{e9}");
    };
    for gone in [false, true] {
        for (name, edit, words) in [
            (
                "foreign",
                &foreign as &dyn Fn(&mut OldVectors),
                "Failed to deserialize vectors.bin: names record 'zz', which mappings.bin does \
                 not hold",
            ),
            (
                "narrow",
                &narrow,
                "Failed to deserialize vectors.bin: holds vectors of 7 values and config.json \
                 declares dim 8",
            ),
            (
                "missing",
                &missing,
                "Failed to deserialize vectors.bin: holds 9 vectors and mappings.bin holds 10 \
                 records; record 'r3\u{e9}' has no vector",
            ),
        ] {
            let path = edited(&format!("{name}-{gone}.zdb"), gone, edit);
            assert_eq!(
                refused(Collection::load(path.to_str().unwrap())),
                words,
                "{name}, dump gone {gone}"
            );
        }
    }
    let counted = edited("foreign-counted.zdb", true, &foreign);
    edit_config(&counted, |config| config["vector_count"] = json!(11));
    assert_eq!(
        refused(Collection::load(counted.to_str().unwrap())),
        "Failed to deserialize vectors.bin: names record 'zz', which mappings.bin does not hold"
    );

    let trained = filled(Some(StorageMode::QuantizedWithRaw), 1050);
    assert!(trained.is_quantized());
    for (name, code, words) in [
        (
            "codes-foreign",
            ("zz", vec![1u8, 2, 3, 4]),
            "Failed to deserialize pq_codes.bin: names record 'zz', which mappings.bin does not \
             hold",
        ),
        (
            "codes-narrow",
            ("r3\u{e9}", vec![1u8, 2, 3]),
            "Failed to deserialize pq_codes.bin: holds codes of 3 bytes and quantization.json \
             declares 4 subvectors",
        ),
    ] {
        let path = temp.at(&format!("{name}.zdb"));
        trained.save(path.to_str().unwrap()).unwrap();
        as_old(&path, "2.0.0");
        let bytes = std::fs::read(path.join("pq_codes.bin")).unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = legacy::read_codes(&bytes, "c", 1050)
            .unwrap()
            .into_iter()
            .collect();
        entries.retain(|(id, _)| id != code.0);
        entries.push((code.0.to_string(), code.1.clone()));
        let entries: Vec<(&str, Vec<u8>)> = entries
            .iter()
            .map(|(id, codes)| (id.as_str(), codes.clone()))
            .collect();
        replace(&path, "pq_codes.bin", &old::codes(&entries));
        assert_eq!(
            refused(Collection::load(path.to_str().unwrap())),
            words,
            "{name}"
        );
    }

    // A trained quantized_only directory of a release that kept its
    // training records' vectors beside the codes, and only those.
    let only = filled(Some(StorageMode::QuantizedOnly), 1050);
    let path = temp.at("only-training.zdb");
    only.save(path.to_str().unwrap()).unwrap();
    as_old(&path, "1.0.0");
    assert!(!path.join("vectors.bin").exists());
    let vectors = clustered(1050, 8, 0x0018_2001);
    let training: Vec<(String, Vec<f32>)> = (0..1000)
        .map(|i| (format!("r{i}\u{e9}"), vectors[i].clone()))
        .collect();
    let training: Vec<(&str, Vec<f32>)> = training
        .iter()
        .map(|(id, vector)| (id.as_str(), vector.clone()))
        .collect();
    replace(&path, "vectors.bin", &old::vectors(&training));
    include(&path, "vectors.bin");
    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(everything(&loaded), everything(&only));
}

/// The graph's ceilings, lowered on this thread for the length of a test
/// and put back when dropped.
struct GraphLowered(zeusdb_vector_core::GraphCeilings);

impl GraphLowered {
    fn to(ceilings: zeusdb_vector_core::GraphCeilings) -> Self {
        GraphLowered(zeusdb_vector_core::GRAPH_CEILINGS.with(|held| held.replace(ceilings)))
    }
}

impl Drop for GraphLowered {
    fn drop(&mut self) {
        zeusdb_vector_core::GRAPH_CEILINGS.with(|held| held.set(self.0));
    }
}

/// A directory whose graph would take an arena past its ceiling is refused
/// in the words a rebuild gives, naming the arena. At 5.0.0 the dump's rows come
/// back and the rebuild from them meets the room; below it the version 1
/// dump is not read and the rebuild from vectors.bin meets it. Both
/// directories open at the ceilings the layout sets.
#[test]
fn a_directory_past_the_graphs_ceiling_is_refused_naming_the_arena() {
    let temp = TempDir::new();
    let collection = filled(None, 60);
    let at_five = temp.at("ceiling-5.zdb");
    collection.save(at_five.to_str().unwrap()).unwrap();
    let at_four = temp.at("ceiling-4.zdb");
    collection.save(at_four.to_str().unwrap()).unwrap();
    rewrite_before_five(&at_four, "4.0.0");
    for path in [&at_five, &at_four] {
        let loaded = Collection::load(path.to_str().unwrap()).unwrap();
        assert_eq!(everything(&loaded), everything(&collection));
    }
    let _lowered = GraphLowered::to(zeusdb_vector_core::GraphCeilings {
        nodes: 59,
        ..zeusdb_vector_core::GraphCeilings::LAYOUT
    });
    for path in [&at_five, &at_four] {
        assert_eq!(
            refused(Collection::load(path.to_str().unwrap())),
            "The rebuilt graph is full: its nodes number 59 of the 59 it can address, and the \
             next record to go in can add up to 1.",
            "{}",
            path.display()
        );
    }
}

/// A load sizes the metadata store's entries once, to the records the
/// mappings hold or the reservation the declared size takes, whichever is
/// more, whatever order metadata.json names the records in, at 5.0.0 and
/// below it, so a directory takes the same size in every process.
#[test]
fn a_load_sizes_the_metadata_entries_once_whatever_order_the_file_names_them_in() {
    let temp = TempDir::new();
    for (records, expected_size, capacity) in [(3_000usize, 100usize, 3_000usize), (50, 100, 101)] {
        let declaration =
            Declaration::validate(8, "l2", 6, 60, expected_size, vec!["cat".to_string()]).unwrap();
        let collection = Collection::build(declaration, None);
        let vectors = clustered(records, 8, 0x0018_2004);
        let batch: Vec<ParsedRecord> = vectors
            .iter()
            .enumerate()
            .map(|(i, vector)| record(&format!("r{i}"), vector, "a", i as i64))
            .collect();
        assert_eq!(collection.add_records(batch, vec![], false).total_errors, 0);
        for version in ["5.0.0", "4.0.0"] {
            let path = temp.at(&format!("sized-{records}-{version}.zdb"));
            collection.save(path.to_str().unwrap()).unwrap();
            if version == "4.0.0" {
                rewrite_before_five(&path, version);
            }
            let text = std::fs::read_to_string(path.join("metadata.json")).unwrap();
            let named: serde_json::Map<String, Value> = serde_json::from_str(&text).unwrap();
            let mut entries: Vec<(String, Value)> = named.into_iter().collect();
            entries.sort_by_key(|(id, _)| id[1..].parse::<usize>().unwrap());
            for (order, reversed) in [("forward", false), ("reversed", true)] {
                if reversed {
                    entries.reverse();
                }
                let body: Vec<String> = entries
                    .iter()
                    .map(|(id, fields)| format!("{}:{}", json!(id), fields))
                    .collect();
                let mut manifest = read_json(&path.join("manifest.json"));
                record_json(
                    &path,
                    "metadata.json",
                    &format!("{{{}}}", body.join(",")),
                    &mut manifest,
                );
                write_json(&path.join("manifest.json"), &manifest);
                let loaded = Collection::load(path.to_str().unwrap()).unwrap();
                let layout = loaded.vector_metadata().layout();
                assert_eq!(
                    layout[0].1, capacity,
                    "{records} records at {version}, {order}"
                );
                assert_eq!(
                    everything(&loaded),
                    everything(&collection),
                    "{version}, {order}"
                );
            }
        }
    }
}
