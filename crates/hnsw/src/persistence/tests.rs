//! The four binary artefacts in both layouts.
//!
//! The frame this build writes, held to what it reads back, to the same bytes
//! from two saves of the same records, and to a refusal by name for each way
//! a payload can be wrong. The wire releases before 4.0.0 wrote, held to the
//! values it carries, to the last copy of a key, to a refusal for every
//! prefix and every length its bytes cannot carry, and to whole directories
//! this build saved and rewrote as an earlier release would have written
//! them.

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

fn store(records: &[(usize, &str)]) -> IdStore {
    let mut store = IdStore::new(16);
    for &(internal_id, id) in records {
        store.insert(internal_id, id).unwrap();
    }
    store
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
    let names: Vec<(usize, String)> = collection
        .ids()
        .iter()
        .map(|(internal_id, id)| (internal_id, id.to_string()))
        .collect();
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
/// written it: the four binary artefacts in the old wire, in the order a hash
/// map would hand them out, and the manifest declaring `version` with a
/// length and a digest for each.
fn as_old(path: &Path, version: &str) {
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

/// Replace one artefact and record its new length, as a hand would.
fn replace(path: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(path.join(name), bytes).unwrap();
    let mut manifest = read_json(&path.join("manifest.json"));
    manifest["file_digests"][name] = json!({ "bytes": bytes.len() });
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
/// load has rebuilt every hash map under a seed of its own: the four framed
/// artefacts, `metadata.json` and `config.json` with index level metadata.
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
            json!("4.0.0")
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
    let say = |bytes: &[u8]| refused(framed_vectors(bytes, &ids, 2));
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
        &ids,
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
        let _ = framed_vectors(&blob, &ids, 2);
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
        let (count, offenders) = legacy::walk_vectors(&bytes, "vectors.bin", records).unwrap();
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
                let walk = legacy::walk_vectors(bytes, "f", 40).map(|(c, _)| c);
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
    assert!(refused(legacy::walk_vectors(&vectors, "vectors.bin", 1))
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
            legacy::walk_vectors(&head(&[&v(huge), &pad]), "v", records).map(|_| ())
        ));
        assert!(exceeded(
            legacy::walk_vectors(&head(&[&v(1), &a, &v(huge), &pad]), "v", records).map(|_| ())
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
/// each storage mode, and saving it again writes the framed artefacts the
/// original save wrote, byte for byte.
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
        ] {
            assert_eq!(
                std::fs::read(framed_dir.join(name)).ok(),
                std::fs::read(again.join(name)).ok(),
                "{label} {name}"
            );
        }
    }
}

/// A frame is read wherever it is found, so a directory this build saved
/// and relabelled to an earlier major opens. Bincode's wire is read only
/// below 4.x, so an old directory relabelled 4.0.0 is refused by the frame.
#[test]
fn the_layout_follows_mappings_bin_below_4_and_is_the_frame_from_4() {
    let temp = TempDir::new();
    let collection = filled(None, 60);
    let path = temp.at("relabelled.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
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

/// A framed artefact a hand assembled from another directory is refused by
/// name: codes under an internal id the mappings do not hold, and vectors
/// short of a record.
#[test]
fn a_framed_artefact_that_disagrees_with_the_mappings_is_refused() {
    let temp = TempDir::new();
    let collection = filled(Some(StorageMode::QuantizedWithRaw), 1050);
    let path = temp.at("spliced.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    let highest = collection.ids().highest_slot().unwrap();

    let codes_path = temp.at("codes.zdb");
    collection.save(codes_path.to_str().unwrap()).unwrap();
    let (codes, _) = framed::write_codes(4, [(highest + 7, &[1u8, 2, 3, 4][..])], 1).unwrap();
    replace(&codes_path, "pq_codes.bin", &codes);
    assert_eq!(
        refused(Collection::load(codes_path.to_str().unwrap())),
        format!(
            "Failed to deserialize pq_codes.bin: names internal id {}, which mappings.bin does not hold",
            highest + 7
        )
    );

    let ids = collection.ids();
    let first: Vec<(usize, Vec<f32>)> =
        ids.iter().take(3).map(|(i, _)| (i, vec![0.5; 8])).collect();
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
