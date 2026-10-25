//! The heap blocks a load allocates for the records' metadata, counted.
//!
//! A load reads `metadata.json` straight into the metadata store, so a
//! record's fields take one block and each of its string values one more,
//! and nothing else a record takes. This counts every block a load
//! allocates through a global allocator of its own, which is why it is a
//! test binary of its own, and holds what the fields add to a load to that,
//! at format 5.0.0 and at 4.0.0. Each directory is loaded twice, once with
//! every record holding two fields and once with none, and the difference is
//! what the fields cost.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

use serde_json::{json, Value};
use zeusdb_vector_hnsw::{Collection, Declaration, ParsedRecord};

/// The global allocator, counting every block asked for while `COUNTING`
/// is set. A block grown in place counts again, since the allocator is
/// asked again.
struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static BLOCKS: AtomicU64 = AtomicU64::new(0);

fn count() {
    if COUNTING.load(Relaxed) {
        BLOCKS.fetch_add(1, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// A directory under the system's temporary directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("zeusdb-metadata-blocks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
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

/// The blocks a load of the directory at `path` allocates.
fn blocks_to_load(path: &Path) -> u64 {
    BLOCKS.store(0, Relaxed);
    COUNTING.store(true, Relaxed);
    let loaded = Collection::load(path.to_str().unwrap());
    COUNTING.store(false, Relaxed);
    let loaded = loaded.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let blocks = BLOCKS.load(Relaxed);
    drop(loaded);
    blocks
}

/// The two fields every record of the measured directories holds: one
/// string, which is a block of its own, and one number, which is not.
fn fields(i: usize) -> Value {
    json!({ "cat": format!("c{}", i % 7), "n": i })
}

/// `metadata.json` written as `text`, its length and digest recorded in the
/// manifest with the directory's size, as a save records them.
fn write_metadata(path: &Path, text: &str) {
    std::fs::write(path.join("metadata.json"), text).unwrap();
    let manifest_path = path.join("manifest.json");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["file_digests"]["metadata.json"] = json!({
        "bytes": text.len(),
        "checksum": format!("{:016x}", zeusdb_vector_core::checksum_of(text.as_bytes())),
    });
    let total: u64 = manifest["file_digests"]
        .as_object()
        .unwrap()
        .values()
        .map(|digest| digest["bytes"].as_u64().unwrap())
        .sum();
    manifest["total_size_mb"] = json!(total as f64 / (1024.0 * 1024.0));
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

/// What the fields of `records` records add to a load: the blocks a load of
/// `with` allocates past those a load of `without` does.
fn added_by_fields(with: &Path, without: &Path) -> u64 {
    // A load before either measured one, so a block a first load alone
    // takes, such as a lazily built table, falls outside both.
    blocks_to_load(without);
    let with = blocks_to_load(with);
    let without = blocks_to_load(without);
    with.saturating_sub(without)
}

/// A load gives each record holding fields one block for them and one for
/// its string value, and nothing more, at format 5.0.0 and at 4.0.0. The
/// fixed cost of the two field names the store interns and the buffer one
/// record's fields are read into is held under a few blocks.
#[test]
fn a_load_takes_a_block_for_a_records_fields_and_one_for_each_string() {
    let temp = TempDir::new();
    let fixed = 16;

    // Format 5.0.0, from collections built and saved here.
    let records = 2_000;
    for with_fields in [true, false] {
        let declaration = Declaration::validate(8, "l2", 16, 100, records, vec![]).unwrap();
        let collection = Collection::build(declaration, None);
        let batch: Vec<ParsedRecord> = (0..records)
            .map(|i| ParsedRecord {
                id: format!("r{i}"),
                vector: (0..8).map(|j| ((i * 8 + j) as f32 * 0.37).sin()).collect(),
                sparse: None,
                metadata: if with_fields {
                    serde_json::from_value(fields(i)).unwrap()
                } else {
                    HashMap::new()
                },
            })
            .collect();
        let added = collection.add_records(batch, vec![], false);
        assert_eq!(added.total_errors, 0, "{:?}", added.errors);
        let path = temp.at(if with_fields { "fields-5" } else { "bare-5" });
        collection.save(path.to_str().unwrap()).unwrap();
    }
    let at_five = added_by_fields(&temp.at("fields-5"), &temp.at("bare-5"));
    assert!(
        at_five <= 2 * records as u64 + fixed,
        "at 5.0.0 the fields of {records} records took {at_five} blocks"
    );

    // Format 4.0.0, from a directory a release wrote, its metadata.json
    // written twice: every record with the two fields, and every record
    // with none, as a release before 5.0.0 named each record.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/formats/v4.0.0-raw-l2/index.zdb");
    let names: Vec<String> = match serde_json::from_slice::<Value>(
        &std::fs::read(fixture.join("metadata.json")).unwrap(),
    )
    .unwrap()
    {
        Value::Object(map) => map.keys().cloned().collect(),
        other => panic!("the fixture's metadata.json holds {other}"),
    };
    for with_fields in [true, false] {
        let path = temp.at(if with_fields { "fields-4" } else { "bare-4" });
        std::fs::create_dir_all(&path).unwrap();
        for entry in std::fs::read_dir(&fixture).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), path.join(entry.file_name())).unwrap();
        }
        let every: serde_json::Map<String, Value> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                (
                    name.clone(),
                    if with_fields { fields(i) } else { json!({}) },
                )
            })
            .collect();
        write_metadata(
            &path,
            &serde_json::to_string_pretty(&Value::Object(every)).unwrap(),
        );
    }
    let at_four = added_by_fields(&temp.at("fields-4"), &temp.at("bare-4"));
    assert!(
        at_four <= 2 * names.len() as u64 + fixed,
        "at 4.0.0 the fields of {} records took {at_four} blocks",
        names.len()
    );
}
