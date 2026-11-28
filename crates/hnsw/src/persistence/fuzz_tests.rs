//! A seeded mutator over whole saved directories, run as an ordinary test.
//!
//! # The property
//!
//! **Every mutation of one file of a valid directory either loads into a
//! collection that holds together or is refused with an `Error`, and never
//! panics.** A collection that holds together has positions that agree, as
//! many records as its id store names, a vector of finite values for every
//! record that keeps one, a search and a filtered search that return, an
//! insert and a removal that keep its positions agreeing, and a save that
//! loads again with the same records and metadata. The files' readers each
//! have a fuzzer of their own. This one reads the files together, so it holds
//! the checks a load makes between them: the manifest's record of each file,
//! the counts each file names, the ids every artefact is held to, the
//! quantizer's state against its codes and the graph against the records.
//!
//! # The corpus
//!
//! Directories this build saves at format 5.0.0, being raw under each of the
//! four metrics, raw with a declared field and a sparse space under dot and
//! under BM25, raw with a text space, product quantized with and without raw
//! vectors, and scalar quantized, beside every fixture of
//! `tests/fixtures/formats`, from 1.0.0 to 4.0.0. A fixture whose manifest
//! names a journal is opened as its checkpoint alone, since the journal's
//! readers have fuzzers of their own.
//!
//! # A case
//!
//! One to three mutations of one file of one entry, drawn by the file's
//! kind. A framed file goes through the frame mutator with its repairs, the
//! graph dump through the dump fuzzer's mutator with its repairs, a JSON
//! file through edits of its parsed tree, and any other file through edits
//! of its bytes. The manifest's length and digest for the file are then
//! made the mutated file's, in seven cases of eight, so the mutation reaches
//! the file's reader rather than being refused by the manifest. Every case
//! is read from a directory in memory and written nowhere else.
//!
//! # Determinism and the budget
//!
//! The seed is [`SEED`] and every draw is splitmix64. Each corpus entry
//! draws its cases from a generator of its own, seeded by the draw of
//! [`SEED`]'s own stream at the entry's place, so a failure reproduces from
//! the commit alone and no entry's cases depend on another entry's budget.
//! `ZEUSDB_FUZZ_CASES` sets the cases of
//! each small entry for a deep run, and each quantized entry runs a tenth of
//! them. `ZEUSDB_FUZZ_TRACE=1` names each case before it runs, which is how
//! a case that aborts the process is found.

use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};
use zeusdb_vector_core::test_support::{bounded, read_bound, GRAPH_RESERVE_BYTES};
use zeusdb_vector_core::{
    checksum_of, compile_filter, dump_fuzz, frame_fuzz, Error, FrameKind, SparseVector,
    DUMP_FILENAME, FRAME_MAGIC,
};
use zeusdb_vector_sparse::{SparseConfig, Weighting};
use zeusdb_vector_text::SimpleTokenizer;

use super::load_from;
use super::memory::{Files, MemStorage};
use crate::collection::{
    generation_exhausted, Collection, Declaration, ParsedRecord, SparseHalf, StorageMode,
};
use crate::journal::{Durability, JournalPolicy, Recovery};

// ============================================================================
// THE BUDGET
// ============================================================================

/// The generator's seed, which is the whole of this harness's state.
const SEED: u64 = 0x5eed_0224_d1c7_000b;

/// Cases each small corpus entry runs, unless `ZEUSDB_FUZZ_CASES` sets
/// another number.
const CASES_SMALL: usize = 38;

/// Cases each quantized corpus entry runs, being a trained quantized
/// directory of about a thousand records, which holds a codebook or scales,
/// unless `ZEUSDB_FUZZ_CASES` sets ten times another number.
const CASES_QUANTIZED: usize = 11;

/// The cases a small and a quantized entry run in this process.
fn budgets() -> (usize, usize) {
    match std::env::var("ZEUSDB_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(cases) => (cases, (cases / 10).max(1)),
        None => (CASES_SMALL, CASES_QUANTIZED),
    }
}

fn tracing() -> bool {
    std::env::var("ZEUSDB_FUZZ_TRACE").is_ok_and(|v| v != "0")
}

// ============================================================================
// THE CORPUS
// ============================================================================

/// One saved directory the cases mutate.
struct Entry {
    name: String,
    files: Files,
    /// The policy every case of the entry loads it under.
    policy: fn() -> JournalPolicy,
    quantized: bool,
}

fn replay() -> JournalPolicy {
    JournalPolicy::Replay(Durability::default())
}

fn checkpoint_only() -> JournalPolicy {
    JournalPolicy::CheckpointOnly
}

/// Dimension of every vector the saved entries hold.
const DIM: usize = 8;

/// A record's vector, spread over the space by its index, as `space`
/// stores it.
fn vector_of(collection: &Collection, i: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..DIM)
        .map(|d| (((i * 37 + d * 11) % 97) as f32 / 97.0) - 0.4 + d as f32 * 0.01)
        .collect();
    collection.process_vector_for_space(raw)
}

/// A record's metadata, holding a value of every JSON type the store keeps.
fn metadata_of(i: usize) -> HashMap<String, Value> {
    [
        ("cat".to_string(), json!(format!("c{}", i % 5))),
        ("rank".to_string(), json!(i as i64 - 7)),
        ("score".to_string(), json!(i as f64 * 0.5)),
        ("flag".to_string(), json!(i.is_multiple_of(2))),
        ("tags".to_string(), json!([format!("t{}", i % 3), "all"])),
    ]
    .into_iter()
    .collect()
}

/// A collection of `records` records saved into memory, each record given
/// `sparse` of its index.
fn saved(
    declaration: Declaration,
    quantization: Option<crate::collection::QuantizationConfig>,
    records: usize,
    sparse: &dyn Fn(usize) -> Option<SparseHalf>,
) -> Files {
    let collection = Collection::build(declaration, quantization);
    let batch: Vec<ParsedRecord> = (0..records)
        .map(|i| ParsedRecord {
            id: format!("r{i}"),
            vector: vector_of(&collection, i),
            sparse: sparse(i),
            metadata: metadata_of(i),
        })
        .collect();
    let added = collection.add_records(batch, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    let storage = MemStorage::holding(Files::new());
    collection.save_into(&storage).unwrap();
    storage.files()
}

fn declared(space: &str) -> Declaration {
    Declaration::validate(DIM, space, 8, 40, 2_000, vec!["cat".to_string()]).unwrap()
}

fn terms_of(i: usize) -> Option<SparseHalf> {
    Some(SparseHalf::Vector(SparseVector {
        dims: vec![1 + (i % 17) as u32, 20 + (i % 5) as u32],
        values: vec![1.0, (1 + i % 3) as f32],
    }))
}

fn words_of(i: usize) -> Option<SparseHalf> {
    let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
    Some(SparseHalf::Terms(vec![
        words[i % 5].to_string(),
        words[(i / 5) % 5].to_string(),
    ]))
}

/// The directories this build saves.
fn saved_entries() -> Vec<Entry> {
    let none: &dyn Fn(usize) -> Option<SparseHalf> = &|_| None;
    let mut entries = Vec::new();
    for space in ["l2", "cosine", "dot", "l1"] {
        entries.push(Entry {
            name: format!("5.0.0 raw {space}"),
            files: saved(declared(space), None, 30, none),
            policy: replay,
            quantized: false,
        });
    }
    let dot = declared("l2")
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    entries.push(Entry {
        name: "5.0.0 raw with a sparse space under dot".to_string(),
        files: saved(dot, None, 30, &terms_of),
        policy: replay,
        quantized: false,
    });
    let bm25 = declared("l2")
        .with_sparse(
            "terms",
            SparseConfig {
                weighting: Weighting::Bm25 { k1: 1.2, b: 0.75 },
                ..SparseConfig::default()
            },
        )
        .unwrap();
    entries.push(Entry {
        name: "5.0.0 raw with a sparse space under BM25".to_string(),
        files: saved(bm25, None, 30, &terms_of),
        policy: replay,
        quantized: false,
    });
    let text = declared("cosine")
        .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
        .unwrap();
    entries.push(Entry {
        name: "5.0.0 raw with a text space".to_string(),
        files: saved(text, None, 30, &words_of),
        policy: replay,
        quantized: false,
    });
    for mode in [StorageMode::QuantizedWithRaw, StorageMode::QuantizedOnly] {
        let name = format!("5.0.0 product quantized {mode:?}");
        let declaration = Declaration::validate(DIM, "l2", 8, 40, 2_000, vec![]).unwrap();
        let quantization = declaration.quantization(4, 4, 1_000, None, mode).unwrap();
        entries.push(Entry {
            name,
            files: saved(declaration, Some(quantization), 1_100, none),
            policy: replay,
            quantized: true,
        });
    }
    let declaration = Declaration::validate(DIM, "l2", 8, 40, 2_000, vec![]).unwrap();
    let quantization = declaration
        .scalar_quantization("per_dimension", 1_000, None, StorageMode::QuantizedOnly)
        .unwrap();
    entries.push(Entry {
        name: "5.0.0 scalar quantized".to_string(),
        files: saved(declaration, Some(quantization), 1_100, none),
        policy: replay,
        quantized: true,
    });
    entries
}

/// Every file under `root`, by its name below `root` with `/` between its
/// folders.
fn read_tree(root: &Path, under: &str, files: &mut Files) {
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap())
        .collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let named = if under.is_empty() {
            name
        } else {
            format!("{under}/{name}")
        };
        if entry.file_type().unwrap().is_dir() {
            read_tree(&entry.path(), &named, files);
        } else {
            files.insert(named, std::fs::read(entry.path()).unwrap());
        }
    }
}

/// Every fixture of `tests/fixtures/formats`, its directory read whole.
fn fixture_entries() -> Vec<Entry> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/formats");
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let folder = root.join(&name);
            let (directory, journaled) = if folder.join("journaled.zdb").is_dir() {
                (folder.join("journaled.zdb"), true)
            } else {
                (folder.join("index.zdb"), false)
            };
            let mut files = Files::new();
            read_tree(&directory, "", &mut files);
            let quantized =
                files.contains_key("pq_centroids.bin") || files.contains_key("int8_scales.zdbint8");
            Entry {
                name: format!("fixture {name}"),
                files,
                policy: if journaled { checkpoint_only } else { replay },
                quantized,
            }
        })
        .collect()
}

// ============================================================================
// THE MUTATIONS
// ============================================================================

/// Values a number in a JSON file is replaced by, being the edges of every
/// width a reader converts a field to.
fn hostile_number(rng: &mut frame_fuzz::Rng) -> Value {
    match rng.below(12) {
        0 => json!(0),
        1 => json!(-1),
        2 => json!(1),
        3 => json!(u32::MAX),
        4 => json!(u32::MAX as u64 + 1),
        5 => json!(u64::MAX),
        6 => json!(i64::MIN),
        7 => json!(9_007_199_254_740_993u64),
        8 => json!(1e308),
        9 => json!(-0.0),
        10 => json!(0.5),
        _ => json!(i32::MAX as u64 + 1),
    }
}

/// A value of another type than `value`.
fn other_type(rng: &mut frame_fuzz::Rng, value: &Value) -> Value {
    let choices = [
        json!(null),
        json!(true),
        json!("x"),
        json!(7),
        json!([]),
        json!({}),
    ];
    let mut pick = choices[rng.below(choices.len())].clone();
    if std::mem::discriminant(&pick) == std::mem::discriminant(value) {
        pick = json!(null);
    }
    pick
}

/// A node of `value`'s tree, reached by descending from the root and
/// stopping at each step with odds of one in three.
fn pick_node<'a>(rng: &mut frame_fuzz::Rng, value: &'a mut Value) -> &'a mut Value {
    let children = match value {
        Value::Object(map) => map.len(),
        Value::Array(items) => items.len(),
        _ => 0,
    };
    if children == 0 || rng.below(3) == 0 {
        return value;
    }
    let at = rng.below(children);
    match value {
        Value::Object(map) => {
            let key = map.keys().nth(at).cloned().unwrap();
            pick_node(rng, map.get_mut(&key).unwrap())
        }
        Value::Array(items) => pick_node(rng, &mut items[at]),
        _ => unreachable!("a node with children is an object or an array"),
    }
}

/// One edit of a node of `value`'s tree.
fn edit_json(rng: &mut frame_fuzz::Rng, value: &mut Value) {
    let node = pick_node(rng, value);
    match rng.below(7) {
        0 => *node = hostile_number(rng),
        1 => *node = other_type(rng, node),
        2 => {
            if let Value::Object(map) = node {
                if !map.is_empty() {
                    let key = map.keys().nth(rng.below(map.len())).cloned().unwrap();
                    map.remove(&key);
                }
            }
        }
        3 => {
            if let Value::Object(map) = node {
                map.insert("zeusdb_fuzz".to_string(), hostile_number(rng));
            }
        }
        4 => {
            if let Value::Array(items) = node {
                if !items.is_empty() {
                    let at = rng.below(items.len());
                    let copy = items[at].clone();
                    items.push(copy);
                }
            }
        }
        5 => {
            if let Value::String(text) = node {
                *text = match rng.below(4) {
                    0 => String::new(),
                    1 => "\u{e9}\u{4e2d}\u{1f600}".to_string(),
                    2 => "9".repeat(40),
                    _ => format!("{text}{text}"),
                };
            }
        }
        _ => {
            if let Some(number) = node.as_i64() {
                *node = json!(number.wrapping_add(if rng.below(2) == 0 { 1 } else { -1 }));
            }
        }
    }
}

/// One to three edits of a JSON file's parsed tree, written again, and in
/// one case of eight its text cut short after. A file that does not parse
/// takes edits of its bytes instead.
fn mutate_json(rng: &mut frame_fuzz::Rng, base: &[u8]) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<Value>(base) else {
        return mutate_bytes(rng, base);
    };
    for _ in 0..1 + rng.below(3) {
        edit_json(rng, &mut value);
    }
    let mut out = serde_json::to_vec_pretty(&value).unwrap();
    if rng.below(8) == 0 {
        let cut = rng.below(out.len() + 1);
        out.truncate(cut);
    }
    out
}

/// One to three edits of a file's bytes, being a hostile value at a random
/// offset and width, a byte set, a cut, an append or a run zeroed.
fn mutate_bytes(rng: &mut frame_fuzz::Rng, base: &[u8]) -> Vec<u8> {
    let mut blob = base.to_vec();
    for _ in 0..1 + rng.below(3) {
        let len = blob.len();
        match rng.below(5) {
            0 => {
                if len > 0 {
                    let width = [1usize, 2, 4, 8][rng.below(4)];
                    let at = rng.below(len);
                    let value =
                        frame_fuzz::HOSTILE[rng.below(frame_fuzz::HOSTILE.len())].to_le_bytes();
                    let end = (at + width).min(len);
                    blob[at..end].copy_from_slice(&value[..end - at]);
                }
            }
            1 => {
                if len > 0 {
                    let at = rng.below(len);
                    blob[at] = rng.byte();
                }
            }
            2 => {
                let cut = rng.below(len + 1);
                blob.truncate(cut);
            }
            3 => {
                let extra = 1 + rng.below(32);
                let value = rng.byte();
                blob.extend(std::iter::repeat_n(value, extra));
            }
            _ => {
                if len > 0 {
                    let at = rng.below(len);
                    let end = (at + 1 + rng.below(32)).min(len);
                    blob[at..end].fill(0);
                }
            }
        }
    }
    blob
}

/// The file `name` of an entry mutated by its kind.
fn mutate_file(rng: &mut frame_fuzz::Rng, name: &str, base: &[u8]) -> Vec<u8> {
    if name == DUMP_FILENAME {
        let mut dump_rng = dump_fuzz::Rng(rng.draw());
        dump_fuzz::mutate_dump(&mut dump_rng, base)
    } else if base.starts_with(&FRAME_MAGIC.to_le_bytes()) {
        frame_fuzz::mutate(rng, base, FrameKind::IdMappings)
    } else if name.ends_with(".json") {
        mutate_json(rng, base)
    } else {
        mutate_bytes(rng, base)
    }
}

/// The manifest with its record of `name` made `blob`'s length and digest,
/// where it records `name`, written as a save writes it. A manifest that
/// does not parse is left as it is.
fn repair_manifest(manifest: &[u8], name: &str, blob: &[u8]) -> Vec<u8> {
    let Ok(mut value) = serde_json::from_slice::<Value>(manifest) else {
        return manifest.to_vec();
    };
    let Some(record) = value
        .get_mut("file_digests")
        .and_then(|digests| digests.get_mut(name))
        .and_then(Value::as_object_mut)
    else {
        return manifest.to_vec();
    };
    record.insert("bytes".to_string(), json!(blob.len()));
    if record.contains_key("checksum") {
        record.insert(
            "checksum".to_string(),
            json!(format!("{:016x}", checksum_of(blob))),
        );
    }
    serde_json::to_vec_pretty(&value).unwrap()
}

/// One case of `entry`: the file it mutates and the directory it loads.
fn draw_case(rng: &mut frame_fuzz::Rng, entry: &Entry) -> (String, Files) {
    let names: Vec<&String> = entry.files.keys().collect();
    let name = names[rng.below(names.len())].clone();
    let blob = mutate_file(rng, &name, &entry.files[&name]);
    let mut files = entry.files.clone();
    if name != "manifest.json" && rng.below(8) != 0 {
        if let Some(manifest) = files.get("manifest.json") {
            let repaired = repair_manifest(manifest, &name, &blob);
            files.insert("manifest.json".to_string(), repaired);
        }
    }
    files.insert(name.clone(), blob);
    (name, files)
}

// ============================================================================
// THE PROPERTY
// ============================================================================

/// What a load of one case gave.
enum Outcome {
    /// Loaded, and every check of an opened collection held.
    Opened,
    /// Refused with an `Error`, which names whether the manifest refused
    /// the file before its reader saw it.
    Refused { at_manifest: bool },
}

/// Hold an opened collection to everything a collection that holds together
/// answers. `label` names the case, and `recovery` what the load said,
/// which names why it opened the collection read only, where it did.
///
/// A collection that cannot record the generation after its directory's
/// refuses every save in the words of `generation_exhausted`, and one read
/// only for that reason gives those words as its reason. One read only for
/// a value past the bound on a vector's norm or on a sparse value gives that
/// value's words as its reason and refuses every save in them. Those are the refusals a case
/// may meet past the reads, each by its exact words.
fn hold_together(collection: &Collection, recovery: &Recovery, policy: JournalPolicy, label: &str) {
    assert!(
        collection.positions_agree(),
        "{label}: the positions disagree"
    );
    let ids: Vec<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    assert_eq!(
        collection.len(),
        ids.len(),
        "{label}: len against the ids held"
    );
    let records = collection
        .records(ids.clone(), true, true)
        .unwrap_or_else(|error| panic!("{label}: the records held are refused: {error}"));
    assert_eq!(
        records.len(),
        ids.len(),
        "{label}: the records against the ids"
    );
    for record in &records {
        if let Some(vector) = &record.vector {
            assert!(
                vector.iter().all(|value| value.is_finite()),
                "{label}: record {} holds a value that is not finite",
                record.id
            );
        }
    }

    let query = collection
        .validate_query(vec![0.3; collection.dim()])
        .unwrap_or_else(|error| panic!("{label}: a query is refused: {error}"));
    let params = collection.search_params(5, None, false, None).unwrap();
    collection
        .search_one(&query, None, params)
        .unwrap_or_else(|error| panic!("{label}: a search is refused: {error}"));
    let filter = compile_filter(
        &[("cat".to_string(), json!({"eq": "c1"}))]
            .into_iter()
            .collect(),
    )
    .unwrap();
    collection
        .search_one(&query, Some(&filter), params)
        .unwrap_or_else(|error| panic!("{label}: a filtered search is refused: {error}"));
    let exhausted = generation_exhausted(collection.identity().generation);
    if let Some(reason) = recovery.read_only.as_deref() {
        let words = if reason == exhausted {
            Some(exhausted)
        } else if recovery.read_only_bound() {
            Some(reason.to_string())
        } else {
            None
        };
        if let Some(words) = words {
            let refused = collection.save_into(&MemStorage::holding(Files::new()));
            assert_eq!(
                refused.err().map(|error| error.to_string()),
                Some(words),
                "{label}: the save of a collection read only for {reason}"
            );
        }
        return;
    }

    let added = collection.add_records(
        vec![ParsedRecord {
            id: "fuzz-added".to_string(),
            vector: collection.process_vector_for_space(vec![0.25; collection.dim()]),
            sparse: None,
            metadata: [("cat".to_string(), json!("c9"))].into_iter().collect(),
        }],
        vec![],
        false,
    );
    let _ = added;
    if let Some(first) = ids.first() {
        let _ = collection.remove_point(first.clone());
    }
    assert!(
        collection.positions_agree(),
        "{label}: the positions disagree after an insert and a removal"
    );

    let copy = MemStorage::holding(Files::new());
    match collection.save_into(&copy) {
        Ok(()) => {}
        Err(error) if error.to_string() == exhausted => return,
        Err(error) => panic!("{label}: the save is refused: {error}"),
    }
    let (again, _) = load_from(&copy, "memory-copy", None, policy)
        .unwrap_or_else(|error| panic!("{label}: the save does not load again: {error}"));
    let held: Vec<String> = collection
        .ids()
        .iter()
        .map(|(_, id)| id.to_string())
        .collect();
    let mut held_sorted = held.clone();
    held_sorted.sort();
    let mut again_ids: Vec<String> = again.ids().iter().map(|(_, id)| id.to_string()).collect();
    again_ids.sort();
    assert_eq!(
        again_ids, held_sorted,
        "{label}: the save loads other records"
    );
    let before = collection.records(held.clone(), false, true).unwrap();
    let after = again.records(held, false, true).unwrap();
    for (a, b) in before.iter().zip(&after) {
        assert_eq!(
            a.id, b.id,
            "{label}: the records come back in another order"
        );
        assert_eq!(
            a.metadata, b.metadata,
            "{label}: record {} comes back with other metadata",
            a.id
        );
    }
}

/// Load the case's directory and hold what opens. Returns how it ended.
fn run_case(files: Files, policy: fn() -> JournalPolicy, label: &str) -> Outcome {
    let storage = MemStorage::holding(files);
    match load_from(&storage, "memory", None, policy()) {
        Ok((collection, recovery)) => {
            hold_together(&collection, &recovery, policy(), label);
            Outcome::Opened
        }
        Err(error) => Outcome::Refused {
            at_manifest: matches!(
                error,
                Error::ArtefactLengthMismatch { .. } | Error::ArtefactDigestMismatch { .. }
            ),
        },
    }
}

/// What a panic said.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(words) = payload.downcast_ref::<&str>() {
        (*words).to_string()
    } else if let Some(words) = payload.downcast_ref::<String>() {
        words.clone()
    } else {
        "a panic that carried no message".to_string()
    }
}

/// **The property.** Every mutation of one file of a saved directory loads
/// into a collection that holds together or is refused with an `Error`, and
/// none panics. More than half the cases reach the mutated file's reader,
/// past the manifest's record of it.
#[test]
fn no_mutation_of_a_saved_directory_panics_the_load() {
    let (small, quantized) = budgets();
    let mut corpus = saved_entries();
    corpus.extend(fixture_entries());
    let mut seeds = frame_fuzz::Rng(SEED);
    let (mut cases, mut opened, mut past_manifest) = (0usize, 0usize, 0usize);
    for entry in &corpus {
        let mut rng = frame_fuzz::Rng(seeds.draw());
        let budget = if entry.quantized { quantized } else { small };
        for case in 0..budget {
            let (name, files) = draw_case(&mut rng, entry);
            let label = format!("{} case {} mutating {}", entry.name, case, name);
            if tracing() {
                eprintln!("FUZZ {label}");
            }
            let policy = entry.policy;
            // A case runs under `read_bound` of the directory's bytes and the
            // graph's creation budget, the most a load reserves from a declared
            // size it has checked.
            let bound =
                read_bound(files.values().map(Vec::len).sum()).saturating_add(GRAPH_RESERVE_BYTES);
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                bounded(bound, || run_case(files, policy, &label))
            }))
            .unwrap_or_else(|payload| panic!("{label}: {}", panic_message(payload)));
            cases += 1;
            match outcome {
                Outcome::Opened => {
                    opened += 1;
                    past_manifest += 1;
                }
                Outcome::Refused { at_manifest } => {
                    past_manifest += usize::from(!at_manifest);
                }
            }
        }
    }
    eprintln!(
        "{cases} cases over {} entries, {opened} opened, {past_manifest} past the manifest",
        corpus.len()
    );
    assert!(
        past_manifest * 2 > cases,
        "only {past_manifest} of {cases} cases reached the mutated file's reader"
    );
}
