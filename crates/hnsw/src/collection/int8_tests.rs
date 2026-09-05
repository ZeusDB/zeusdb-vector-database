//! A scalar quantized dense space through the collection.
//!
//! What the binding reaches, exercised at the crate's seam: the declaration
//! and its refusals, training from the sample, the page against the kernel
//! and against a raw index, the mutations that keep the codec, the
//! artefacts with their bounds, the format version in this build's own
//! direction, and the rebuild fallback from the rows. The older reader's
//! direction of the version rule is held from Python against a wheel of
//! the previous release, which this crate cannot hold.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use zeusdb_vector_core::{
    compile_filter, frame_fuzz, test_support::clustered, Error, Filter, FRAME_HEADER_BYTES,
};
use zeusdb_vector_sparse::SparseConfig;

use super::{
    Collection, Declaration, Int8Scale, ParsedRecord, QuantizationConfig, QuantizationScheme,
    StorageMode,
};
use crate::{raw_distance_fn, Durability};

/// A directory under the system's temporary directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("zeusdb-int8-tests-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn sub(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn declaration(dim: usize, space: &str) -> Declaration {
    Declaration::validate(dim, space, 8, 60, 4000, vec!["cat".to_string()]).unwrap()
}

fn scalar(declaration: &Declaration, training_size: usize) -> QuantizationConfig {
    declaration
        .scalar_quantization(
            Int8Scale::PER_DIMENSION,
            training_size,
            None,
            StorageMode::QuantizedOnly,
        )
        .unwrap()
}

fn record(i: usize, vector: Vec<f32>) -> ParsedRecord {
    let mut metadata = HashMap::new();
    metadata.insert(
        "cat".to_string(),
        json!(if i.is_multiple_of(2) { "a" } else { "b" }),
    );
    ParsedRecord {
        id: format!("r{i}"),
        vector,
        sparse: None,
        metadata,
    }
}

fn add(collection: &Collection, records: Vec<ParsedRecord>) {
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

/// Every vector in `range`, processed for the collection's space.
fn records(
    collection: &Collection,
    vectors: &[Vec<f32>],
    range: std::ops::Range<usize>,
) -> Vec<ParsedRecord> {
    range
        .map(|i| record(i, collection.process_vector_for_space(vectors[i].clone())))
        .collect()
}

/// A scalar index over `vectors`, trained at `training` and holding all of
/// them.
fn build(space: &str, vectors: &[Vec<f32>], training: usize) -> Collection {
    let declaration = declaration(vectors[0].len(), space);
    let quantization = scalar(&declaration, training);
    let collection = Collection::build(declaration, Some(quantization));
    add(&collection, records(&collection, vectors, 0..vectors.len()));
    assert!(collection.is_quantized(), "{}", collection.storage_mode());
    collection
}

/// A raw index over the same vectors, for comparison.
fn build_raw(space: &str, vectors: &[Vec<f32>]) -> Collection {
    let collection = Collection::build(declaration(vectors[0].len(), space), None);
    add(&collection, records(&collection, vectors, 0..vectors.len()));
    collection
}

fn page(
    collection: &Collection,
    query: &[f32],
    k: usize,
    filter: Option<&Filter>,
) -> Vec<(String, f32)> {
    let params = collection.search_params(k, None, false, None).unwrap();
    let query = collection.validate_query(query.to_vec()).unwrap();
    collection
        .search_one(&query, filter, params)
        .unwrap()
        .into_iter()
        .map(|hit| (hit.0, hit.1))
        .collect()
}

fn decoded(collection: &Collection, id: &str) -> Vec<f32> {
    collection
        .records(vec![id.to_string()], true, true)
        .unwrap()
        .remove(0)
        .vector
        .expect("a scalar index decodes every record")
}

fn cat_filter(value: &str) -> Filter {
    let mut conditions = HashMap::new();
    conditions.insert("cat".to_string(), json!(value));
    compile_filter(&conditions).unwrap()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn write_json(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_string_pretty(value).unwrap()).unwrap();
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Every scale a codec fitted on `sample` holds, computed here as the
/// largest magnitude a dimension reaches over the sample divided by 127,
/// which is order independent and so needs no knowledge of the shuffle.
fn expected_scales(sample: &[Vec<f32>]) -> Vec<f32> {
    let dim = sample[0].len();
    (0..dim)
        .map(|j| {
            let largest = sample.iter().map(|v| v[j].abs()).fold(0f32, f32::max);
            if largest > 0.0 {
                largest / 127.0
            } else {
                1.0
            }
        })
        .collect()
}

// ============================================================================
// THE DECLARATION
// ============================================================================

/// Every rule `create()` applies to a scalar declaration refuses by name,
/// and all four spaces are admitted.
#[test]
fn the_scalar_declaration_is_held_to_its_rules() {
    let d = declaration(8, "l2");
    assert!(matches!(
        d.scalar_quantization("per_vector", 1000, None, StorageMode::QuantizedOnly),
        Err(Error::UnsupportedInt8Scale { .. })
    ));
    assert!(matches!(
        d.scalar_quantization("per_dimension", 999, None, StorageMode::QuantizedOnly),
        Err(Error::TrainingSizeTooSmall { .. })
    ));
    assert!(matches!(
        d.scalar_quantization("per_dimension", 1000, Some(999), StorageMode::QuantizedOnly),
        Err(Error::MaxTrainingBelowTrainingSize { .. })
    ));
    assert!(matches!(
        d.scalar_quantization("per_dimension", 1000, None, StorageMode::QuantizedWithRaw),
        Err(Error::Int8KeepsNoRaw)
    ));
    let message = Error::Int8KeepsNoRaw.to_string();
    assert!(message.contains("quantized_with_raw"), "{message}");
    assert!(message.contains("0.006 and 0.018"), "{message}");

    for space in ["cosine", "l2", "l1", "dot"] {
        let d = declaration(8, space);
        let q = scalar(&d, 1000);
        assert_eq!(
            q.scheme,
            QuantizationScheme::Int8 {
                scale: Int8Scale::PerDimension
            }
        );
        assert!(q.is_int8());
        let c = Collection::build(d, Some(q));
        assert!(c.has_quantization());
        assert!(!c.can_use_quantization());
        assert!(!c.is_quantized());
        assert_eq!(c.storage_mode(), "raw_collecting_for_training");
        assert_eq!(c.stats()["quantization_type"], "int8");
        assert_eq!(c.stats()["quantization_scale"], "per_dimension");
        assert_eq!(c.stats()["quantization_trained"], "false");
        assert!(c
            .info()
            .contains("quantization=int8(scale=per_dimension, untrained, inactive"));
    }
}

// ============================================================================
// TRAINING
// ============================================================================

/// Training fires on the record that fills the set, fits the scales to
/// exactly that sample, and rebuilds the graph over rows; every record after
/// it is encoded through the same scales, and the values clipped are
/// counted.
#[test]
fn training_fits_the_scales_from_the_sample_and_rebuilds_over_rows() {
    let vectors = clustered(1300, 16, 0x0157_0001);
    let d = declaration(16, "l2");
    let c = Collection::build(d.clone(), Some(scalar(&d, 1000)));

    add(&c, records(&c, &vectors, 0..999));
    assert!(!c.can_use_quantization());
    assert_eq!(c.storage_mode(), "raw_collecting_for_training");
    assert_eq!(c.stats()["training_progress"], "999/1000 (99.9%)");
    assert_eq!(c.stats()["training_vectors_needed"], "1");
    assert_eq!(c.training_vectors_needed(), 1);

    add(&c, records(&c, &vectors, 999..1000));
    assert!(c.can_use_quantization());
    assert!(c.is_quantized());
    assert_eq!(c.storage_mode(), "quantized_active");
    assert!(c.training_completed_at().is_some());
    assert_eq!(c.training_progress(), 100.0);

    let codec = c.int8_codec().expect("the codec is fitted");
    let expected = expected_scales(&vectors[..1000]);
    assert_eq!(codec.dim(), 16);
    for (j, (got, want)) in codec.scales().iter().zip(&expected).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "scale {j}");
    }
    assert_eq!(c.stats()["quantization_saturated_values"], "0");

    add(&c, records(&c, &vectors, 1000..1300));
    let clipped: usize = vectors[1000..1300].iter().map(|v| codec.saturated(v)).sum();
    let stats = c.stats();
    assert_eq!(stats["quantization_type"], "int8");
    assert_eq!(stats["quantization_trained"], "true");
    assert_eq!(stats["quantization_active"], "true");
    assert_eq!(stats["storage_mode"], "quantized_only");
    assert_eq!(stats["storage_strategy"], "memory_optimized");
    assert_eq!(stats["raw_vectors_retained"], "none_once_trained");
    assert_eq!(stats["quantized_codes_stored"], "1300");
    assert_eq!(stats["raw_vectors_stored"], "0");
    assert_eq!(stats["raw_vectors_memory_mb"], "0.00");
    assert_eq!(stats["quantized_codes_memory_mb"], "0.00");
    assert_eq!(stats["quantization_compression_ratio"], "4.0x");
    assert_eq!(stats["training_progress"], "1000/1000 (100.0%)");
    assert_eq!(stats["quantization_saturated_values"], clipped.to_string());
    assert!(
        clipped > 0,
        "the records past the sample reach past its range"
    );
    assert!(!stats.contains_key("codebook_memory_mb"));
    assert!(!stats.contains_key("rerank_calibrated"));
    assert!(stats.contains_key("scale_memory_mb"));
    assert!(c
        .info()
        .contains("quantization=int8(scale=per_dimension, trained, active, compression=4.0x)"));

    // The decoded record is the codes through the scales and nothing else.
    let vector = c.process_vector_for_space(vectors[1200].clone());
    let codes = codec.quantize(&vector).unwrap();
    assert_eq!(decoded(&c, "r1200"), codec.reconstruct(&codes).unwrap());

    // The report the binding builds from.
    let report = c.quantization_report().unwrap();
    assert!(report.scheme.is_int8());
    let quantizer = report.quantizer.unwrap();
    assert!(quantizer.is_trained);
    assert_eq!(quantizer.total_centroids, None);
    assert_eq!(quantizer.compression_ratio, 4.0);
    assert_eq!(
        quantizer.memory_mb,
        codec.memory_bytes() as f64 / (1024.0 * 1024.0)
    );
}

/// The training sample is capped at `max_training_vectors`, and a record
/// past the sample's range saturates rather than being refused.
#[test]
fn a_capped_sample_fits_fewer_records_and_a_value_past_the_range_is_clipped() {
    let vectors = clustered(1200, 8, 0x0157_0002);
    let d = declaration(8, "l2");
    let q = d
        .scalar_quantization(
            "per_dimension",
            1000,
            Some(1000),
            StorageMode::QuantizedOnly,
        )
        .unwrap();
    let c = Collection::build(d, Some(q));
    add(&c, records(&c, &vectors, 0..1000));
    assert!(c.is_quantized());
    let codec = c.int8_codec().unwrap().clone();

    // A record ten times the range of every dimension: accepted, clipped in
    // every value, and decoded to the edge of the range.
    let huge: Vec<f32> = codec.scales().iter().map(|s| s * 1270.0).collect();
    let mut wild = record(5000, huge.clone());
    wild.id = "wild".to_string();
    add(&c, vec![wild]);
    assert_eq!(c.stats()["quantization_saturated_values"], "8");
    let back = decoded(&c, "wild");
    for (value, scale) in back.iter().zip(codec.scales()) {
        assert_eq!(value.to_bits(), (127.0 * scale).to_bits());
    }
    let hits = page(&c, &huge, 1, None);
    assert_eq!(hits[0].0, "wild");
}

// ============================================================================
// THE PAGE
// ============================================================================

/// A scalar l2 page is on the scale a raw l2 index reports: every score is
/// the raw l2 distance from the query to the record's decoded vector, bit
/// for bit, and sits within the quantization error of the raw score rather
/// than at its square root. The exact scan scores through the same kernel,
/// so a filtered page agrees with the traversal bit for bit.
#[test]
fn a_scalar_l2_page_is_on_the_raw_scale_and_the_scan_agrees_with_the_traversal() {
    let vectors: Vec<Vec<f32>> = clustered(1500, 12, 0x0157_0003)
        .into_iter()
        .map(|v| v.into_iter().map(|x| x * 3.0).collect())
        .collect();
    let raw = build_raw("l2", &vectors);
    let int8 = build("l2", &vectors, 1000);
    let l2 = raw_distance_fn("l2");
    // A decoded value sits within half a scale of the stored one, so by the
    // triangle inequality the distance moves by at most half the norm of
    // the scale vector, whatever the query.
    let half_scale_norm = 0.5
        * int8
            .int8_codec()
            .unwrap()
            .scales()
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            .sqrt();

    let mut compared = 0;
    let mut roots_told_apart = 0;
    for q in 0..20 {
        // A query on the far side of the origin from a record, so every
        // distance on the page is large enough for a root to move it.
        let query: Vec<f32> = vectors[1450 + q].iter().map(|x| -x).collect();
        let query = query.as_slice();
        let traversed = page(&int8, query, 5, None);
        assert_eq!(traversed.len(), 5);
        for (id, score) in &traversed {
            let expected = l2(query, &decoded(&int8, id));
            assert_eq!(score.to_bits(), expected.to_bits(), "{id}");
            let stored = raw
                .records(vec![id.clone()], true, true)
                .unwrap()
                .remove(0)
                .vector
                .unwrap();
            let raw_score = l2(query, &stored);
            assert!(
                (score - raw_score).abs() <= half_scale_norm + 1e-4,
                "{id}: scalar {score} against raw {raw_score}, bound {half_scale_norm}"
            );
            if raw_score > 1.5 {
                assert!((score - raw_score).abs() < (score.sqrt() - raw_score).abs());
                roots_told_apart += 1;
            }
            compared += 1;
        }
        // The filtered page runs the exact scan, since the declared field
        // admits well under the threshold.
        let scanned = page(&int8, query, 5, Some(&cat_filter("a")));
        assert_eq!(scanned.len(), 5);
        for (id, score) in &scanned {
            assert!(id[1..].parse::<usize>().unwrap().is_multiple_of(2));
            let expected = l2(query, &decoded(&int8, id));
            assert_eq!(score.to_bits(), expected.to_bits(), "{id} scanned");
            if let Some((_, traversed_score)) = traversed.iter().find(|(t, _)| t == id) {
                assert_eq!(
                    score.to_bits(),
                    traversed_score.to_bits(),
                    "{id} on both paths"
                );
            }
        }
    }
    assert_eq!(compared, 100);
    assert!(
        roots_told_apart >= 50,
        "{roots_told_apart} distances above 1.5"
    );
}

/// A scalar cosine page reports cosine distances to the decoded record at
/// unit length, and the scan agrees with the traversal bit for bit.
#[test]
fn a_scalar_cosine_page_is_scored_at_unit_length() {
    let vectors = clustered(1300, 16, 0x0157_0004);
    let int8 = build("cosine", &vectors, 1000);
    let cosine = raw_distance_fn("cosine");
    for q in 0..20 {
        let query = int8.process_vector_for_space(vectors[1250 + q].clone());
        let traversed = page(&int8, &query, 5, None);
        for (id, score) in &traversed {
            assert!((0.0..=2.0).contains(score), "{id}: {score}");
            let vector = decoded(&int8, id);
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{id}: norm {norm}");
            assert!((score - cosine(&query, &vector)).abs() < 1e-5, "{id}");
        }
        let scanned = page(&int8, &query, 5, Some(&cat_filter("b")));
        for (id, score) in &scanned {
            if let Some((_, traversed_score)) = traversed.iter().find(|(t, _)| t == id) {
                assert_eq!(
                    score.to_bits(),
                    traversed_score.to_bits(),
                    "{id} on both paths"
                );
            }
        }
    }
}

/// A scalar index under `dot` and `l1` finds what a raw index under the
/// same space finds, against a brute force ranking under that space.
///
/// The bound is loose on this synthetic corpus, whose inner product ranking
/// is decided among near ties the decoding error can reorder; the
/// admission of the two spaces rests on the corpora at 100,000 records,
/// where the scalar graph sits within a few hundredths of the raw one.
#[test]
fn dot_and_l1_scalar_indexes_search_like_their_raw_counterparts() {
    let vectors = clustered(1500, 16, 0x0157_0005);
    for space in ["dot", "l1"] {
        let raw = build_raw(space, &vectors);
        let int8 = build(space, &vectors, 1000);
        assert_eq!(int8.metric(), space);
        let distance = raw_distance_fn(space);
        let mut raw_hits = 0usize;
        let mut int8_hits = 0usize;
        for q in 0..50 {
            let query = &vectors[1440 + q];
            let mut truth: Vec<(f32, usize)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (distance(query, v), i))
                .collect();
            truth.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let truth: Vec<String> = truth.iter().take(5).map(|(_, i)| format!("r{i}")).collect();
            raw_hits += page(&raw, query, 5, None)
                .iter()
                .filter(|(id, _)| truth.contains(id))
                .count();
            int8_hits += page(&int8, query, 5, None)
                .iter()
                .filter(|(id, _)| truth.contains(id))
                .count();
        }
        let raw_recall = raw_hits as f64 / 250.0;
        let int8_recall = int8_hits as f64 / 250.0;
        assert!(raw_recall >= 0.9, "{space}: raw {raw_recall}");
        let allowed = if space == "dot" { 0.2 } else { 0.1 };
        assert!(
            int8_recall >= raw_recall - allowed,
            "{space}: scalar {int8_recall} against raw {raw_recall}"
        );
    }
}

// ============================================================================
// THE MUTATIONS
// ============================================================================

/// Removal, compaction, overwrite, a rebuild at another degree, the rebuild
/// a caller asks for and a clear all keep the codec, and every one leaves
/// the index searchable over rows.
#[test]
fn every_mutation_keeps_the_codec_and_the_rows() {
    let vectors = clustered(1300, 16, 0x0157_0006);
    let c = build("l2", &vectors, 1000);
    let scales: Vec<u32> = c
        .int8_codec()
        .unwrap()
        .scales()
        .iter()
        .map(|s| s.to_bits())
        .collect();
    let same_codec = |c: &Collection| {
        let now: Vec<u32> = c
            .int8_codec()
            .unwrap()
            .scales()
            .iter()
            .map(|s| s.to_bits())
            .collect();
        assert_eq!(now, scales);
        assert!(c.is_quantized(), "{}", c.storage_mode());
    };

    assert!(c.remove_point("r10".to_string()).unwrap());
    assert_eq!(c.len(), 1299);
    assert!(page(&c, &vectors[10], 3, None)
        .iter()
        .all(|(id, _)| id != "r10"));
    same_codec(&c);

    assert!(c.compact().unwrap() > 0);
    same_codec(&c);
    assert_eq!(c.stats()["graph_nodes"], "1299");
    assert_eq!(page(&c, &vectors[20], 1, None)[0].0, "r20");

    let mut replacement = record(20, vectors[1299].clone());
    replacement.id = "r20".to_string();
    let added = c.add_records(vec![replacement], vec![], true);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(c.len(), 1299);
    let top = page(&c, &vectors[1299], 2, None);
    assert!(top.iter().any(|(id, _)| id == "r20"), "{top:?}");
    same_codec(&c);

    let plan = c.plan_rebuild(Some(6), None, Some(80)).unwrap();
    c.rebuild(plan).unwrap();
    assert_eq!(c.m(), 6);
    same_codec(&c);
    assert_eq!(page(&c, &vectors[30], 1, None)[0].0, "r30");

    assert!(c.rebuild_with_quantization().unwrap());
    same_codec(&c);
    assert_eq!(page(&c, &vectors[40], 1, None)[0].0, "r40");

    assert_eq!(c.clear().unwrap(), 1299);
    same_codec(&c);
    assert_eq!(c.storage_mode(), "quantized_active");
    assert_eq!(c.len(), 0);
    assert_eq!(c.stats()["quantization_saturated_values"], "0");
    add(&c, records(&c, &vectors, 0..5));
    assert_eq!(c.stats()["quantized_codes_stored"], "5");
    assert_eq!(page(&c, &vectors[3], 1, None)[0].0, "r3");
    same_codec(&c);
}

// ============================================================================
// THE ARTEFACTS
// ============================================================================

/// A trained scalar directory carries the scales and the rows beside the
/// dump, at the scalar minor, and comes back page for page; a second save
/// of the loaded index writes the same bytes. An untrained one carries the
/// declaration and the raw vectors, and trains after it is loaded.
#[test]
fn a_scalar_directory_round_trips_with_its_two_artefacts() {
    let vectors = clustered(1300, 16, 0x0157_0007);
    let c = build("cosine", &vectors, 1000);
    let dir = TempDir::new();
    let path = dir.sub("scalar.zdb");
    c.save(&path).unwrap();
    let path = Path::new(&path);

    let manifest = read_json(&path.join("manifest.json"));
    assert_eq!(manifest["format_version"], "1.2.0");
    let files: Vec<&str> = manifest["files_included"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(files.contains(&"int8_scales.zdbint8"), "{files:?}");
    assert!(files.contains(&"int8_rows.zdbint8"), "{files:?}");
    assert!(!files.contains(&"pq_centroids.bin"), "{files:?}");
    assert!(!files.contains(&"vectors.bin"), "{files:?}");
    assert_eq!(
        manifest["compression_info"]["compression_ratio"],
        64.0 / 20.0
    );
    assert_eq!(manifest["quantization_trained"], true);
    assert_eq!(manifest["storage_mode"], "quantized_active");

    let quantization = read_json(&path.join("quantization.json"));
    assert_eq!(quantization["type"], "int8");
    assert_eq!(quantization["scale"], "per_dimension");
    assert_eq!(quantization["is_trained"], true);
    assert_eq!(quantization["storage_mode"], "quantized_only");
    assert!(quantization.get("subvectors").is_none());
    assert!(quantization.get("bits").is_none());
    assert!(quantization.get("pq_config").is_none());
    assert_eq!(
        quantization["saturated_values"]
            .as_u64()
            .unwrap()
            .to_string(),
        c.stats()["quantization_saturated_values"]
    );

    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert!(loaded.is_quantized());
    assert_eq!(loaded.storage_mode(), "quantized_active");
    assert_eq!(loaded.len(), 1300);
    assert_eq!(loaded.vector_count(), 1300);
    let before = c.stats();
    let after = loaded.stats();
    for key in [
        "quantization_type",
        "quantization_scale",
        "quantization_trained",
        "quantization_active",
        "quantized_codes_stored",
        "quantization_compression_ratio",
        "quantization_saturated_values",
        "training_progress",
        "scale_memory_mb",
    ] {
        assert_eq!(before[key], after[key], "{key}");
    }
    assert_eq!(
        c.int8_codec().unwrap().scales(),
        loaded.int8_codec().unwrap().scales()
    );
    for q in 0..10 {
        let query = c.process_vector_for_space(vectors[1250 + q].clone());
        let want = page(&c, &query, 5, None);
        let got = page(&loaded, &query, 5, None);
        assert_eq!(
            want.iter()
                .map(|(id, s)| (id.clone(), s.to_bits()))
                .collect::<Vec<_>>(),
            got.iter()
                .map(|(id, s)| (id.clone(), s.to_bits()))
                .collect::<Vec<_>>()
        );
        let filtered_want = page(&c, &query, 5, Some(&cat_filter("a")));
        let filtered_got = page(&loaded, &query, 5, Some(&cat_filter("a")));
        assert_eq!(filtered_want, filtered_got);
    }
    assert_eq!(decoded(&c, "r7"), decoded(&loaded, "r7"));

    let again = dir.sub("again.zdb");
    loaded.save(&again).unwrap();
    let again = Path::new(&again);
    for name in [
        "hnsw_index.zdbgraph",
        "int8_scales.zdbint8",
        "int8_rows.zdbint8",
    ] {
        assert_eq!(
            std::fs::read(path.join(name)).unwrap(),
            std::fs::read(again.join(name)).unwrap(),
            "{name}"
        );
    }

    // Untrained: the declaration and the raw vectors, and training after
    // the load reaches the same scales a build without the save reaches.
    let d = declaration(16, "cosine");
    let untrained = Collection::build(d.clone(), Some(scalar(&d, 1000)));
    add(&untrained, records(&untrained, &vectors, 0..600));
    let path = dir.sub("collecting.zdb");
    untrained.save(&path).unwrap();
    let path = Path::new(&path);
    let manifest = read_json(&path.join("manifest.json"));
    assert_eq!(manifest["format_version"], "1.2.0");
    let files: Vec<&str> = manifest["files_included"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(files.contains(&"vectors.bin"), "{files:?}");
    assert!(!files.contains(&"int8_scales.zdbint8"), "{files:?}");
    let quantization = read_json(&path.join("quantization.json"));
    assert_eq!(quantization["is_trained"], false);
    assert_eq!(quantization["training_ids"].as_array().unwrap().len(), 600);
    let reopened = Collection::load(path.to_str().unwrap()).unwrap();
    assert_eq!(reopened.storage_mode(), "raw_collecting_for_training");
    assert_eq!(reopened.training_vectors_needed(), 400);
    add(&reopened, records(&reopened, &vectors, 600..1300));
    assert!(reopened.is_quantized());
    assert_eq!(
        reopened.int8_codec().unwrap().scales(),
        c.int8_codec().unwrap().scales()
    );
}

/// A dump the loader refuses is rebuilt from the rows, and the rebuilt
/// index holds every record, searches, and writes the rows back byte for
/// byte.
#[test]
fn the_rows_rebuild_the_graph_when_the_dump_is_refused() {
    let vectors = clustered(1300, 16, 0x0157_0008);
    let c = build("l2", &vectors, 1000);
    let dir = TempDir::new();
    let path = dir.sub("scalar.zdb");
    c.save(&path).unwrap();
    let path = Path::new(&path);
    let rows_written = std::fs::read(path.join("int8_rows.zdbint8")).unwrap();

    // A dump one byte short of the length the manifest records.
    let dump = path.join("hnsw_index.zdbgraph");
    let mut bytes = std::fs::read(&dump).unwrap();
    bytes.pop();
    std::fs::write(&dump, bytes).unwrap();

    let loaded = Collection::load(path.to_str().unwrap()).unwrap();
    assert!(loaded.is_quantized());
    assert_eq!(loaded.len(), 1300);
    assert_eq!(loaded.stats()["graph_nodes"], "1300");
    for q in 0..20 {
        assert_eq!(
            page(&loaded, &vectors[q * 60], 1, None)[0].0,
            format!("r{}", q * 60)
        );
    }
    for id in ["r0", "r999", "r1299"] {
        assert_eq!(decoded(&c, id), decoded(&loaded, id));
    }
    let again = dir.sub("rebuilt.zdb");
    loaded.save(&again).unwrap();
    assert_eq!(
        std::fs::read(Path::new(&again).join("int8_rows.zdbint8")).unwrap(),
        rows_written
    );
    assert_eq!(
        std::fs::read(Path::new(&again).join("int8_scales.zdbint8")).unwrap(),
        std::fs::read(path.join("int8_scales.zdbint8")).unwrap()
    );
}

/// Every bound the two artefacts and the scalar fields of quantization.json
/// are held to refuses by name, before anything is built from the file.
#[test]
fn every_scales_and_rows_bound_is_refused_by_name() {
    let vectors = clustered(1200, 16, 0x0157_0009);
    let c = build("l2", &vectors, 1000);
    let dir = TempDir::new();
    let source = dir.sub("scalar.zdb");
    c.save(&source).unwrap();
    let source = PathBuf::from(source);
    let counter = std::cell::Cell::new(0usize);

    let refused = |mutate: &dyn Fn(&Path)| -> String {
        counter.set(counter.get() + 1);
        let target = PathBuf::from(dir.sub(&format!("damaged-{}.zdb", counter.get())));
        copy_dir(&source, &target);
        mutate(&target);
        match Collection::load(target.to_str().unwrap()) {
            Ok(_) => panic!("the damaged directory opened"),
            Err(e) => e.to_string(),
        }
    };
    let edit = |target: &Path, name: &str, f: &dyn Fn(&mut Vec<u8>)| {
        let file = target.join(name);
        let mut bytes = std::fs::read(&file).unwrap();
        f(&mut bytes);
        std::fs::write(&file, bytes).unwrap();
    };
    let drop_artefact = |target: &Path, name: &str| {
        std::fs::remove_file(target.join(name)).unwrap();
        let manifest_path = target.join("manifest.json");
        let mut manifest = read_json(&manifest_path);
        let files = manifest["files_included"].as_array_mut().unwrap();
        files.retain(|v| v != name);
        manifest["file_digests"]
            .as_object_mut()
            .unwrap()
            .remove(name);
        write_json(&manifest_path, &manifest);
    };
    let edit_quantization = |target: &Path, f: &dyn Fn(&mut Value)| {
        let file = target.join("quantization.json");
        let mut value = read_json(&file);
        f(&mut value);
        std::fs::write(&file, serde_json::to_string_pretty(&value).unwrap()).unwrap();
        let manifest_path = target.join("manifest.json");
        let mut manifest = read_json(&manifest_path);
        manifest["file_digests"]
            .as_object_mut()
            .unwrap()
            .remove("quantization.json");
        write_json(&manifest_path, &manifest);
    };
    let scales = "int8_scales.zdbint8";
    let rows = "int8_rows.zdbint8";
    let payload = FRAME_HEADER_BYTES;

    // The scales: the wrong count, a scale of zero, a scale that is not
    // finite, and a payload one scale short.
    let m = refused(&|t| edit(t, scales, &|b| frame_fuzz::set_entries(b, 17)));
    assert!(
        m.contains(scales) && m.contains("17 scales") && m.contains("dim 16"),
        "{m}"
    );
    let m = refused(&|t| {
        edit(t, scales, &|b| {
            b[payload + 12..payload + 16].copy_from_slice(&0f32.to_le_bytes());
            frame_fuzz::repair_trailer(b);
        })
    });
    assert!(m.contains(scales) && m.contains("scale 3 is 0"), "{m}");
    let m = refused(&|t| {
        edit(t, scales, &|b| {
            b[payload..payload + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            frame_fuzz::repair_trailer(b);
        })
    });
    assert!(m.contains(scales) && m.contains("scale 0 is NaN"), "{m}");
    let m = refused(&|t| {
        edit(t, scales, &|b| {
            b.truncate(b.len() - 4);
            frame_fuzz::repair_header(b);
            frame_fuzz::repair_trailer(b);
        })
    });
    assert!(m.contains(scales), "{m}");
    let m = refused(&|t| drop_artefact(t, scales));
    assert!(m.contains(scales) && m.contains("missing"), "{m}");

    // The rows: ids out of order, an id the mappings do not hold, a count
    // the payload does not match, and no rows at all.
    let stride = 4 + 16;
    let m = refused(&|t| {
        edit(t, rows, &|b| {
            let mut first = [0u8; 4];
            first.copy_from_slice(&b[payload..payload + 4]);
            let mut second = [0u8; 4];
            second.copy_from_slice(&b[payload + stride..payload + stride + 4]);
            b[payload..payload + 4].copy_from_slice(&second);
            b[payload + stride..payload + stride + 4].copy_from_slice(&first);
            frame_fuzz::repair_trailer(b);
        })
    });
    assert!(m.contains(rows) && m.contains("strictly increasing"), "{m}");
    let m = refused(&|t| {
        edit(t, rows, &|b| {
            b[payload..payload + 4].copy_from_slice(&0u32.to_le_bytes());
            frame_fuzz::repair_trailer(b);
        })
    });
    assert!(
        m.contains(rows) && m.contains("internal id 0") && m.contains("mappings.bin"),
        "{m}"
    );
    let m = refused(&|t| edit(t, rows, &|b| frame_fuzz::set_entries(b, 1199)));
    assert!(m.contains(rows) && m.contains("1199 rows"), "{m}");
    let m = refused(&|t| drop_artefact(t, rows));
    assert!(m.contains(rows) && m.contains("holds 0 rows"), "{m}");

    // The fields of quantization.json under `int8`.
    let m = refused(&|t| edit_quantization(t, &|v| v["scale"] = json!("per_vector")));
    assert!(
        m.contains("quantization.json") && m.contains("per_vector"),
        "{m}"
    );
    let m = refused(&|t| edit_quantization(t, &|v| v["training_size"] = json!(10)));
    assert!(m.contains("training_size is 10"), "{m}");
    let m =
        refused(&|t| edit_quantization(t, &|v| v["storage_mode"] = json!("quantized_with_raw")));
    assert!(m.contains("quantized_with_raw"), "{m}");
    let m = refused(&|t| edit_quantization(t, &|v| v["max_training_vectors"] = json!(999)));
    assert!(m.contains("max_training_vectors is 999"), "{m}");
    let m = refused(&|t| edit_quantization(t, &|v| v["type"] = json!("opq")));
    assert!(m.contains("quantization.json"), "{m}");

    // And the untouched directory still opens.
    assert!(Collection::load(source.to_str().unwrap()).is_ok());
}

/// The scalar minor on each major, and this build's own direction of the
/// version rule: a scalar directory labelled at the older minor still opens
/// here, and a later major is refused.
#[test]
fn the_version_rule_holds_for_a_scalar_directory() {
    let vectors = clustered(1100, 8, 0x0157_000a);
    let dir = TempDir::new();

    let dense = build("l2", &vectors, 1000);
    let dense_path = dir.sub("dense.zdb");
    dense.save(&dense_path).unwrap();
    assert_eq!(
        read_json(&Path::new(&dense_path).join("manifest.json"))["format_version"],
        "1.2.0"
    );

    let d = declaration(8, "l2")
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    let q = scalar(&d, 1000);
    let spaced = Collection::build(d, Some(q));
    add(&spaced, records(&spaced, &vectors, 0..1100));
    assert!(spaced.is_quantized());
    let spaced_path = dir.sub("spaced.zdb");
    spaced.save(&spaced_path).unwrap();
    assert_eq!(
        read_json(&Path::new(&spaced_path).join("manifest.json"))["format_version"],
        "2.1.0"
    );
    assert!(Collection::load(&spaced_path).unwrap().is_quantized());

    let journaled = build("l2", &vectors, 1000);
    let journaled_path = dir.sub("journaled.zdb");
    journaled
        .journal_to(&journaled_path, Durability::default())
        .unwrap();
    assert_eq!(
        read_json(&Path::new(&journaled_path).join("manifest.json"))["format_version"],
        "3.1.0"
    );
    drop(journaled);
    assert!(Collection::load(&journaled_path).unwrap().is_quantized());

    // An untrained scalar directory carries the minor too, since its
    // quantization.json already takes the scalar layout.
    let d = declaration(8, "l2");
    let collecting = Collection::build(d.clone(), Some(scalar(&d, 1000)));
    add(&collecting, records(&collecting, &vectors, 0..10));
    let collecting_path = dir.sub("collecting.zdb");
    collecting.save(&collecting_path).unwrap();
    assert_eq!(
        read_json(&Path::new(&collecting_path).join("manifest.json"))["format_version"],
        "1.2.0"
    );

    // This build reads any 1.x, so the older minor over a scalar directory
    // opens here, and a later major is refused with the majors it reads.
    let relabel = |name: &str, version: &str| -> String {
        let target = PathBuf::from(dir.sub(name));
        copy_dir(Path::new(&dense_path), &target);
        let manifest_path = target.join("manifest.json");
        let mut manifest = read_json(&manifest_path);
        manifest["format_version"] = json!(version);
        write_json(&manifest_path, &manifest);
        target.to_string_lossy().into_owned()
    };
    assert!(Collection::load(&relabel("older-minor.zdb", "1.1.0"))
        .unwrap()
        .is_quantized());
    let message = Collection::load(&relabel("future.zdb", "4.0.0"))
        .err()
        .unwrap()
        .to_string();
    assert!(
        message.contains("format version 4.0.0 cannot be opened"),
        "{message}"
    );
    assert!(message.contains("1.x, 2.x and 3.x"), "{message}");
}
