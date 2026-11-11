//! A value no record can hold, read from a saved directory.
//!
//! A load holds every float it reads from a directory finite before the
//! index is built from it. Each test copies a fixture under
//! `tests/fixtures/formats`, or a save of one at 5.0.0, sets one value of one
//! file to a NaN or an infinity, and makes the file's checksums and the
//! manifest's record of it agree with its bytes, as a damaged medium or a
//! hand edit with its digests repaired leaves it. The codebook in either
//! wire, the scalar rows artefact and a version 2 dump refuse the load,
//! naming the file and the value's place. A version 1 dump is refused naming
//! the node, and the load rebuilds the graph from the directory's own store.
//! A codebook of finite values whose squared norm overflows, and a finite
//! scale for which 254 times it overflows, refuse the load the same way.

use std::path::{Path, PathBuf};

use serde_json::Value;
use zeusdb_vector_core::{
    checksum_of, frame, restore_graph, restore_int8_graph, unframe, DumpBounds, Error, FrameKind,
    FsDir, DUMP_FILENAME,
};

use super::directory_tests::everything;
use super::journal_support::{copy_tree, fixture_folder, resized, rewrite_manifest, TempDir};
use super::Collection;
use crate::journal::{Durability, JournalPolicy};

/// The values each test sets in turn.
const NOT_FINITE: [f32; 2] = [f32::NAN, f32::INFINITY];

/// The scalar rows artefact.
const INT8_ROWS: &str = "int8_rows.zdbint8";

/// A copy of the format fixture `name` under `temp`, named `label`, being
/// the directory to load.
fn copy_of(temp: &TempDir, name: &str, label: &str) -> PathBuf {
    let target = temp.at(label);
    copy_tree(&fixture_folder(name).join("index.zdb"), &target);
    target
}

/// The directory at `path`, opened as a plain load opens it.
fn open(path: &Path) -> Result<(Collection, crate::Recovery), Error> {
    Collection::recover_with(
        path.to_str().unwrap(),
        None,
        JournalPolicy::Replay(Durability::default()),
    )
}

fn u32_at(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
}

fn u64_at(bytes: &[u8], at: usize) -> usize {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize
}

/// The manifest's record of `name` made the file's own: its length, its
/// digest where the manifest records one, and at 5.x the directory's size.
fn repair_manifest(directory: &Path, name: &str) {
    let bytes = std::fs::read(directory.join(name)).unwrap();
    rewrite_manifest(directory, |manifest| {
        if let Some(entry) = manifest["file_digests"].get_mut(name) {
            entry["bytes"] = Value::from(bytes.len() as u64);
            if entry.get("checksum").is_some() {
                entry["checksum"] = Value::from(format!("{:016x}", checksum_of(&bytes)));
            }
        }
        if manifest["format_version"]
            .as_str()
            .is_some_and(|version| version.starts_with("5."))
        {
            resized(manifest);
        }
    });
}

/// The framed artefact `name` in `directory` with the four bytes at `at` of
/// its payload set to `value`, framed again so its checksums hold, and the
/// manifest's record of it repaired.
fn set_in_frame(directory: &Path, name: &str, kind: FrameKind, at: usize, value: f32) {
    let path = directory.join(name);
    let bytes = std::fs::read(&path).unwrap();
    let framed = unframe(&bytes, kind, name).unwrap();
    let mut payload = framed.payload.to_vec();
    payload[at..at + 4].copy_from_slice(&value.to_le_bytes());
    let written = frame(kind, framed.encoding, framed.entries, &payload);
    std::fs::write(&path, written).unwrap();
    repair_manifest(directory, name);
}

/// The byte after the length at `at` in the old wire, being one byte below
/// 251, or a marker and two, four or eight bytes.
fn varint_end(bytes: &[u8], at: usize) -> usize {
    at + match bytes[at] {
        251 => 3,
        252 => 5,
        253 => 9,
        _ => 1,
    }
}

/// Where value `position` of centroid `centroid` of subvector `subvector`
/// starts in a codebook of the old wire, being a length and that many
/// subvectors, each a length and that many centroids, each a length and
/// that many floats. Every length a fixture's codebook holds is one byte.
fn old_wire_value(bytes: &[u8], subvector: usize, centroid: usize, position: usize) -> usize {
    let mut at = varint_end(bytes, 0);
    for held in 0..=subvector {
        let centroids = if held < subvector {
            usize::from(bytes[at])
        } else {
            centroid
        };
        at = varint_end(bytes, at);
        for _ in 0..centroids {
            let values = usize::from(bytes[at]);
            at = varint_end(bytes, at) + 4 * values;
        }
    }
    varint_end(bytes, at) + 4 * position
}

/// The load of `path` refused on its codebook holding `value` at value
/// `position` of centroid `centroid` of subvector `subvector`.
fn assert_codebook_refused(path: &Path, value: f32, place: (usize, usize, usize), label: &str) {
    let (subvector, centroid, position) = place;
    match open(path) {
        Err(Error::DecodeFailed { file, error }) => {
            assert_eq!(file, "pq_centroids.bin", "{label}");
            assert_eq!(
                error,
                format!(
                    "the codebook holds {value} at value {position} of centroid {centroid} of \
                     subvector {subvector}, and every value is finite"
                ),
                "{label}"
            );
        }
        Err(other) => panic!("{label}: expected the codebook refused, got {other}"),
        Ok(_) => panic!("{label}: a codebook holding {value} was accepted"),
    }
}

/// Where the row of node `node` starts in a version 1 dump, and the bytes a
/// row takes. The rows follow the 96 byte header, the layer table of 16
/// counts, every node's eight byte origin id and the adjacency region.
fn version_1_row(blob: &[u8], node: usize) -> (usize, usize) {
    let width = u32_at(blob, 20) * if blob[17] == 1 { 4 } else { 1 };
    let rows = 96 + 16 * 4 + 8 * u64_at(blob, 40) + u64_at(blob, 64);
    (rows + node * width, width)
}

/// The origin id a version 1 dump names for node `node`.
fn version_1_origin(blob: &[u8], node: usize) -> usize {
    u64_at(blob, 96 + 16 * 4 + 8 * node)
}

/// The checksum of a version 1 dump made to agree with its bytes. It covers
/// everything between the header and the trailer, and the header's own
/// checksum covers the header alone.
fn resealed_version_1(blob: &mut [u8]) {
    let end = blob.len() - 16;
    let sum = checksum_of(&blob[96..end]);
    blob[end..end + 8].copy_from_slice(&sum.to_le_bytes());
}

/// Where the row of node `node` starts in a version 2 dump, and the bytes a
/// row takes. The rows region follows the 128 byte header and the identity
/// region.
fn version_2_row(blob: &[u8], node: usize) -> (usize, usize) {
    let width = u32_at(blob, 20) * if blob[17] == 1 { 4 } else { 1 };
    (128 + u64_at(blob, 48) + node * width, width)
}

/// The origin id a version 2 dump names for node `node`, after the identity
/// region's layer table of 16 counts.
fn version_2_origin(blob: &[u8], node: usize) -> usize {
    u32_at(blob, 128 + 16 * 4 + 4 * node)
}

/// Every region's checksum of a version 2 dump made to agree with its
/// bytes, and then the header's, which covers them.
fn resealed_version_2(blob: &mut [u8]) {
    let mut start = 128;
    for (length_at, sum_at) in [(48, 80), (56, 88), (64, 96)] {
        let end = start + u64_at(blob, length_at);
        let sum = checksum_of(&blob[start..end]);
        blob[sum_at..sum_at + 8].copy_from_slice(&sum.to_le_bytes());
        start = end;
    }
    let sum = checksum_of(&blob[..120]);
    blob[120..128].copy_from_slice(&sum.to_le_bytes());
}

/// The dump in `directory` with the four bytes at `at` set to `value`,
/// sealed again by `reseal`, and the manifest's record of it repaired.
fn set_in_dump(directory: &Path, at: usize, value: f32, reseal: fn(&mut [u8])) {
    let path = directory.join(DUMP_FILENAME);
    let mut blob = std::fs::read(&path).unwrap();
    blob[at..at + 4].copy_from_slice(&value.to_le_bytes());
    reseal(&mut blob);
    std::fs::write(&path, &blob).unwrap();
    repair_manifest(directory, DUMP_FILENAME);
}

/// What the dump reader says of the version 1 dump in `directory`, read for
/// a collection shaped as `collection`, whose records are the directory's.
fn version_1_refusal(collection: &Collection, directory: &Path) -> String {
    let mut positions: Vec<u32> = collection
        .internal_ids()
        .iter()
        .map(|&(internal_id, _)| u32::try_from(internal_id).unwrap())
        .collect();
    positions.sort_unstable();
    let bounds = DumpBounds {
        min_nodes: positions.len(),
        max_origin_id: collection.id_counter(),
        positions: Some(&positions),
        version: 1,
        rows_only: false,
    };
    let dir = FsDir::new(directory);
    let dense = collection.dense();
    let restored = match dense.int8_codec() {
        Some(codec) => restore_int8_graph(
            &dir,
            DUMP_FILENAME,
            &dense.metric,
            dense.m(),
            dense.ef_construction(),
            codec.clone(),
            bounds,
        ),
        None => restore_graph(
            &dir,
            DUMP_FILENAME,
            &dense.metric,
            dense.m(),
            dense.ef_construction(),
            dense.dim,
            None,
            bounds,
        ),
    };
    match restored {
        Err(refusal) => refusal.detail,
        Ok(_) => panic!("the dump in {} was read", directory.display()),
    }
}

/// A search for every 97th record's own vector returns a full page, every
/// score finite.
fn searches(collection: &Collection, label: &str) {
    let names: Vec<String> = collection
        .internal_ids()
        .into_iter()
        .step_by(97)
        .map(|(_, id)| id)
        .collect();
    for view in collection.records(names, true, true).unwrap() {
        let query = collection.process_vector_for_space(view.vector.unwrap());
        let params = collection.search_params(10, None, false, None).unwrap();
        let page = collection.search_one(&query, None, params).unwrap();
        assert_eq!(page.len(), 10, "{label}");
        assert!(
            page.iter().all(|hit| hit.score().is_finite()),
            "{label}: a score is not finite"
        );
    }
}

/// A codebook holding a NaN or an infinity refuses the load, naming the
/// file and the value's place, in the framed wire of 4.0.0 on and in the
/// old wire before it, so no distance is taken through the value.
#[test]
fn a_codebook_value_that_is_not_finite_refuses_the_load_naming_its_place() {
    let temp = TempDir::new("finite-tests");
    for value in NOT_FINITE {
        // The framed wire: the centroid count and the width, then every
        // value, subvector by subvector and centroid by centroid.
        let label = format!("framed-{value}");
        let path = copy_of(&temp, "v4.0.0-pq-cosine", &label);
        let bytes = std::fs::read(path.join("pq_centroids.bin")).unwrap();
        let payload = unframe(&bytes, FrameKind::PqCodebook, "pq_centroids.bin")
            .unwrap()
            .payload
            .to_vec();
        let (centroids, width) = (u32_at(&payload, 0), u32_at(&payload, 4));
        let at = centroids * width + width + 1;
        set_in_frame(
            &path,
            "pq_centroids.bin",
            FrameKind::PqCodebook,
            8 + 4 * at,
            value,
        );
        assert_codebook_refused(&path, value, (1, 1, 1), &label);

        let label = format!("old-wire-{value}");
        let path = copy_of(&temp, "v1.1.0-pq", &label);
        let file = path.join("pq_centroids.bin");
        let mut bytes = std::fs::read(&file).unwrap();
        let at = old_wire_value(&bytes, 1, 1, 1);
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        std::fs::write(&file, &bytes).unwrap();
        repair_manifest(&path, "pq_centroids.bin");
        assert_codebook_refused(&path, value, (1, 1, 1), &label);
    }
}

/// A codebook whose values are finite and whose squared norm, as the
/// quantizer computes it, is not refuses the load naming the centroid, under
/// cosine and l2 in the framed wire and in the old wire before it, so no
/// distance taken through the codebook is a NaN.
#[test]
fn a_codebook_whose_tables_overflow_refuses_the_load_naming_the_place() {
    let temp = TempDir::new("finite-tests");
    let words = "the codebook's centroid 1 of subvector 1 has a squared norm of inf, and every \
                 squared norm is finite";
    for (name, value) in [
        ("v4.0.0-pq-cosine", 1e30f32),
        ("v4.0.0-pq-l2-raw", f32::MAX),
    ] {
        let label = format!("{name}-{value}");
        let path = copy_of(&temp, name, &label);
        let bytes = std::fs::read(path.join("pq_centroids.bin")).unwrap();
        let payload = unframe(&bytes, FrameKind::PqCodebook, "pq_centroids.bin")
            .unwrap()
            .payload
            .to_vec();
        let (centroids, width) = (u32_at(&payload, 0), u32_at(&payload, 4));
        let at = centroids * width + width;
        set_in_frame(
            &path,
            "pq_centroids.bin",
            FrameKind::PqCodebook,
            8 + 4 * at,
            value,
        );
        assert_codebook_tables_refused(&path, words, &label);
    }

    let path = copy_of(&temp, "v1.1.0-pq", "old-wire");
    let file = path.join("pq_centroids.bin");
    let mut bytes = std::fs::read(&file).unwrap();
    let at = old_wire_value(&bytes, 1, 1, 0);
    bytes[at..at + 4].copy_from_slice(&1e30f32.to_le_bytes());
    std::fs::write(&file, &bytes).unwrap();
    repair_manifest(&path, "pq_centroids.bin");
    assert_codebook_tables_refused(&path, words, "old-wire");
}

/// The load of `path` refused on its codebook's tables in `words`.
fn assert_codebook_tables_refused(path: &Path, words: &str, label: &str) {
    match open(path) {
        Err(Error::DecodeFailed { file, error }) => {
            assert_eq!(file, "pq_centroids.bin", "{label}");
            assert_eq!(error, words, "{label}");
        }
        Err(other) => panic!("{label}: expected the codebook refused, got {other}"),
        Ok(_) => panic!("{label}: a codebook whose tables overflow was accepted"),
    }
}

/// A scale for which 254 times it, the widest difference of two codes, is
/// not finite refuses the load naming the scale, at 4.0.0 and at 1.2.0, as
/// a scale that is not finite does.
#[test]
fn a_scale_whose_widest_difference_overflows_refuses_the_load() {
    let temp = TempDir::new("finite-tests");
    let scale = f32::MAX / 200.0;
    for name in ["v4.0.0-int8-l2", "v1.2.0-int8"] {
        let path = copy_of(&temp, name, name);
        set_in_frame(
            &path,
            "int8_scales.zdbint8",
            FrameKind::Int8Scales,
            4 * 3,
            scale,
        );
        match open(&path) {
            Err(Error::Int8ArtefactInvalid { file, detail }) => {
                assert_eq!(file, "int8_scales.zdbint8", "{name}");
                assert_eq!(
                    detail,
                    format!(
                        "scale 3 is {scale}, and 254 times it, the widest difference of two \
                         codes, is not finite"
                    ),
                    "{name}"
                );
            }
            Err(other) => panic!("{name}: expected the scales refused, got {other}"),
            Ok(_) => panic!("{name}: a scale of {scale} was accepted"),
        }
    }
}

/// A version 1 dump holding a NaN or an infinity, in a raw row or as the
/// inverse norm a scalar cosine row carries, is refused naming the node,
/// and the load rebuilds the graph from the directory's own store, which
/// holds the saved value. The collection opens with every record of the
/// untouched directory, each vector bit for bit, and its searches score
/// finite.
#[test]
fn a_version_1_dump_value_that_is_not_finite_is_refused_and_the_graph_rebuilt() {
    let temp = TempDir::new("finite-tests");
    for (name, norm) in [("v4.0.0-raw-l2", false), ("v4.0.0-int8-cosine", true)] {
        let (untouched, recovery) = open(&copy_of(&temp, name, &format!("{name}-untouched")))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!recovery.graph_rebuilt, "{name}");
        let expected = everything(&untouched);
        for value in NOT_FINITE {
            let label = format!("{name}-{value}");
            let path = copy_of(&temp, name, &label);
            let blob = std::fs::read(path.join(DUMP_FILENAME)).unwrap();
            let (row, width) = version_1_row(&blob, 2);
            let origin = version_1_origin(&blob, 2);
            let at = if norm { row + width - 4 } else { row + 4 };
            set_in_dump(&path, at, value, resealed_version_1);
            let place = if norm {
                "as the inverse norm of"
            } else {
                "in the row of"
            };
            assert_eq!(
                version_1_refusal(&untouched, &path),
                format!("its vectors hold {value} {place} node 2, internal id {origin}"),
                "{label}"
            );
            let (rebuilt, recovery) = open(&path).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(recovery.graph_rebuilt, "{label}");
            assert_eq!(everything(&rebuilt), expected, "{label}");
            assert!(rebuilt.positions_agree(), "{label}");
            searches(&rebuilt, &label);
        }
    }
}

/// A version 2 dump whose scalar cosine row carries an inverse norm that is
/// a NaN or an infinity refuses the load, naming the node and its id, as a
/// row no record can hold refuses it.
#[test]
fn a_version_2_dump_inverse_norm_that_is_not_finite_refuses_the_load() {
    let temp = TempDir::new("finite-tests");
    let (fixture, _) = open(&copy_of(&temp, "v4.0.0-int8-cosine", "fixture")).unwrap();
    let saved = temp.at("saved.zdb");
    fixture.save(saved.to_str().unwrap()).unwrap();
    drop(fixture);
    for value in NOT_FINITE {
        let label = format!("saved-{value}");
        let path = temp.at(&label);
        copy_tree(&saved, &path);
        let blob = std::fs::read(path.join(DUMP_FILENAME)).unwrap();
        let (row, width) = version_2_row(&blob, 2);
        let origin = version_2_origin(&blob, 2);
        set_in_dump(&path, row + width - 4, value, resealed_version_2);
        match open(&path) {
            Err(Error::GraphDumpInvalid { name, detail }) => {
                assert_eq!(name, DUMP_FILENAME, "{label}");
                assert_eq!(
                    detail,
                    format!(
                        "its rows region holds {value} as the inverse norm of node 2, internal \
                         id {origin}"
                    ),
                    "{label}"
                );
            }
            Err(other) => panic!("{label}: expected the dump refused, got {other}"),
            Ok(_) => panic!("{label}: an inverse norm of {value} was accepted"),
        }
    }
}

/// A scalar rows artefact whose cosine row carries an inverse norm that is
/// a NaN or an infinity refuses the load, naming the record's internal id,
/// whether or not the graph dump beside it is read.
#[test]
fn a_scalar_row_inverse_norm_that_is_not_finite_refuses_the_load() {
    let temp = TempDir::new("finite-tests");
    for value in NOT_FINITE {
        let label = format!("rows-{value}");
        let path = copy_of(&temp, "v4.0.0-int8-cosine", &label);
        let bytes = std::fs::read(path.join(INT8_ROWS)).unwrap();
        let framed = unframe(&bytes, FrameKind::Int8Rows, INT8_ROWS).unwrap();
        // Every row is a four byte internal id, then the codes, then the
        // inverse norm.
        let stride = framed.payload.len() / usize::try_from(framed.entries).unwrap();
        let id = u32_at(framed.payload, 2 * stride);
        set_in_frame(&path, INT8_ROWS, FrameKind::Int8Rows, 3 * stride - 4, value);
        match open(&path) {
            Err(Error::Int8ArtefactInvalid { file, detail }) => {
                assert_eq!(file, INT8_ROWS, "{label}");
                assert_eq!(
                    detail,
                    format!(
                        "the row of internal id {id} holds {value} as its inverse norm, and \
                         every inverse norm is finite"
                    ),
                    "{label}"
                );
            }
            Err(other) => panic!("{label}: expected the rows refused, got {other}"),
            Ok(_) => panic!("{label}: an inverse norm of {value} was accepted"),
        }
    }
}
