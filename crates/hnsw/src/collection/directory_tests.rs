//! A directory at format 5.0.0.
//!
//! Every graph a collection holds, saved and loaded through a version 2
//! dump, over three corpora: the four raw spaces, product quantization in
//! cosine and l2 under both storage modes, and the four scalar spaces. Each
//! record's row is held once, in the dump, beside `vectors.bin` only where a
//! product quantized graph keeps raw vectors. A dump refused on its header,
//! its identity region or its rows region refuses the load, naming the
//! check, and a dump whose topology fails is rebuilt from the rows it holds.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};
use zeusdb_vector_core::{checksum_of, test_support::clustered, Error, DUMP_FILENAME};

use super::journal_support::{break_topology, TempDir};
use super::{Collection, Declaration, Int8Scale, ParsedRecord, StorageMode};
use crate::journal::{Durability, JournalPolicy};

// ============================================================================
// THE COLLECTIONS
// ============================================================================

/// The quantization a configuration declares.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    Raw,
    PqOnly,
    PqRaw,
    Int8,
}

/// The four raw spaces.
const SPACES: [&str; 4] = ["cosine", "l2", "l1", "dot"];

/// Records in each collection, enough to train either quantizer.
const RECORDS: usize = 1050;

/// One corpus: its width, and its vectors, each corpus spread its own way.
fn corpus(which: usize) -> Vec<Vec<f32>> {
    match which {
        0 => clustered(RECORDS, 8, 0x0219_c001),
        1 => clustered(RECORDS, 12, 0x0219_c002)
            .into_iter()
            .map(|v| v.into_iter().map(|x| x * 4.0).collect())
            .collect(),
        _ => clustered(RECORDS, 16, 0x0219_c003)
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let scale = 0.5 + (i % 7) as f32;
                v.into_iter().map(|x| x * scale).collect()
            })
            .collect(),
    }
}

/// A collection of `shape` in `space` over `vectors`, each prepared for the
/// space as an insertion prepares it, every tenth record carrying a field,
/// two records removed and one overwritten, so its graph holds stranded
/// nodes.
fn collection(space: &str, shape: Shape, vectors: &[Vec<f32>]) -> Collection {
    let dim = vectors[0].len();
    let declaration = Declaration::validate(dim, space, 6, 60, 2000, vec![]).unwrap();
    let quantization = match shape {
        Shape::Raw => None,
        Shape::PqOnly | Shape::PqRaw => {
            let mode = if shape == Shape::PqRaw {
                StorageMode::QuantizedWithRaw
            } else {
                StorageMode::QuantizedOnly
            };
            Some(declaration.quantization(4, 4, 1000, None, mode).unwrap())
        }
        Shape::Int8 => Some(
            declaration
                .scalar_quantization(
                    Int8Scale::PER_DIMENSION,
                    1000,
                    None,
                    StorageMode::QuantizedOnly,
                )
                .unwrap(),
        ),
    };
    let collection = Collection::build(declaration, quantization);
    let records: Vec<ParsedRecord> = vectors
        .iter()
        .enumerate()
        .map(|(i, vector)| ParsedRecord {
            id: format!("r{i}"),
            vector: collection.process_vector_for_space(vector.clone()),
            sparse: None,
            metadata: if i % 10 == 0 {
                HashMap::from([("i".to_string(), json!(i))])
            } else {
                HashMap::new()
            },
        })
        .collect();
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert!(collection
        .remove_points(&["r3".to_string(), "r4".to_string()])
        .unwrap()
        .is_empty());
    let overwrite = ParsedRecord {
        id: "r5".to_string(),
        vector: collection.process_vector_for_space(vectors[6].clone()),
        sparse: None,
        metadata: HashMap::from([("i".to_string(), json!(-5))]),
    };
    let added = collection.add_records(vec![overwrite], vec![], true);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    assert_eq!(collection.is_quantized(), shape != Shape::Raw);
    collection
}

/// Every record, sorted by id, as its internal id, its vector's bits and its
/// fields.
fn everything(collection: &Collection) -> Vec<(String, usize, Vec<u32>, String)> {
    let names = collection.internal_ids();
    let views = collection
        .records(names.iter().map(|(_, id)| id.clone()).collect(), true, true)
        .unwrap();
    let mut out: Vec<(String, usize, Vec<u32>, String)> = names
        .iter()
        .zip(views)
        .map(|((internal_id, id), view)| {
            let fields: std::collections::BTreeMap<String, Value> =
                view.metadata.into_iter().collect();
            let bits = view
                .vector
                .unwrap_or_default()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            (
                id.clone(),
                *internal_id,
                bits,
                serde_json::to_string(&fields).unwrap(),
            )
        })
        .collect();
    out.sort();
    out
}

/// The pages of the first twelve vectors as queries, each prepared for the
/// space as a search prepares it, as an id and a score's bits per hit.
fn pages(collection: &Collection, vectors: &[Vec<f32>]) -> Vec<Vec<(String, u32)>> {
    vectors[..12]
        .iter()
        .map(|query| {
            let query = collection.process_vector_for_space(query.clone());
            let params = collection.search_params(10, None, false, None).unwrap();
            collection
                .search_one(&query, None, params)
                .unwrap()
                .iter()
                .map(|hit| (hit.id().to_string(), hit.score().to_bits()))
                .collect()
        })
        .collect()
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap()
}

/// Save `collection` into `dir`, open it, save the opened one beside it, and
/// hold the three to one another: the files a 5.0.0 directory holds, the
/// graph read and not rebuilt, the same records and the same pages, and the
/// same dump and the same artefacts from the second save.
fn round_trips(
    collection: &Collection,
    shape: Shape,
    vectors: &[Vec<f32>],
    temp: &TempDir,
    name: &str,
) {
    let first = temp.at(&format!("{name}.zdb"));
    collection.save(first.to_str().unwrap()).unwrap();
    let files: Vec<String> = manifest(&first)["files_included"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| file.as_str().unwrap().to_string())
        .collect();
    for absent in ["pq_codes.bin", "int8_rows.zdbint8"] {
        assert!(
            !files.iter().any(|file| file == absent),
            "{name}: {files:?}"
        );
        assert!(!first.join(absent).exists(), "{name}: {absent}");
    }
    let keeps_raw = shape == Shape::PqRaw;
    assert_eq!(
        files.iter().any(|file| file == "vectors.bin"),
        keeps_raw,
        "{name}: {files:?}"
    );
    assert!(files.iter().any(|file| file == DUMP_FILENAME), "{name}");
    let dump = std::fs::read(first.join(DUMP_FILENAME)).unwrap();
    assert_eq!(
        u32::from_le_bytes(dump[8..12].try_into().unwrap()),
        2,
        "{name}"
    );

    let (opened, recovery) = Collection::recover_with(
        first.to_str().unwrap(),
        None,
        JournalPolicy::Replay(Durability::default()),
    )
    .unwrap();
    assert!(!recovery.graph_rebuilt, "{name}: the graph was rebuilt");
    assert_eq!(opened.is_quantized(), collection.is_quantized(), "{name}");
    assert_eq!(everything(&opened), everything(collection), "{name}");
    assert_eq!(
        pages(&opened, vectors),
        pages(collection, vectors),
        "{name}"
    );
    assert!(opened.positions_agree(), "{name}");

    let second = temp.at(&format!("{name}-again.zdb"));
    opened.save(second.to_str().unwrap()).unwrap();
    for file in &files {
        assert_eq!(
            std::fs::read(first.join(file)).unwrap(),
            std::fs::read(second.join(file)).unwrap(),
            "{name}: {file} moved between the two saves"
        );
    }
}

/// Every configuration of `shape`'s family round trips over each corpus.
fn every_configuration(shapes: &[(&'static str, Shape)]) {
    let temp = TempDir::new("directory-tests");
    for which in 0..3 {
        let vectors = corpus(which);
        for &(space, shape) in shapes {
            let collection = collection(space, shape, &vectors);
            let name = format!("{which}-{space}-{shape:?}").to_lowercase();
            round_trips(&collection, shape, &vectors, &temp, &name);
        }
    }
}

/// The four raw spaces over the three corpora.
#[test]
fn every_raw_graph_round_trips_through_a_version_2_dump() {
    let shapes: Vec<(&str, Shape)> = SPACES.iter().map(|&space| (space, Shape::Raw)).collect();
    every_configuration(&shapes);
}

/// Product quantization in cosine and l2 under both storage modes over the
/// three corpora.
#[test]
fn every_product_quantized_graph_round_trips_through_a_version_2_dump() {
    let mut shapes = Vec::new();
    for space in ["cosine", "l2"] {
        for shape in [Shape::PqOnly, Shape::PqRaw] {
            shapes.push((space, shape));
        }
    }
    every_configuration(&shapes);
}

/// The four scalar spaces over the three corpora.
#[test]
fn every_scalar_graph_round_trips_through_a_version_2_dump() {
    let shapes: Vec<(&str, Shape)> = SPACES.iter().map(|&space| (space, Shape::Int8)).collect();
    every_configuration(&shapes);
}

// ============================================================================
// THE DUMP DAMAGED
// ============================================================================

fn read_u64(blob: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(blob[at..at + 8].try_into().unwrap())
}

/// Where each region of a version 2 dump starts: the identity, the rows,
/// the topology and the trailer.
fn regions(blob: &[u8]) -> (usize, usize, usize, usize) {
    let identity = 128;
    let rows = identity + read_u64(blob, 48) as usize;
    let topology = rows + read_u64(blob, 56) as usize;
    let trailer = topology + read_u64(blob, 64) as usize;
    (identity, rows, topology, trailer)
}

/// Every region's checksum and the header's made to agree with the bytes
/// the dump holds, so a forgery reaches the rule it breaks.
fn repaired(blob: &mut [u8]) {
    let (identity, rows, topology, trailer) = regions(blob);
    for (at, from, to) in [
        (80, identity, rows),
        (88, rows, topology),
        (96, topology, trailer),
    ] {
        let sum = checksum_of(&blob[from..to]);
        blob[at..at + 8].copy_from_slice(&sum.to_le_bytes());
    }
    let sum = checksum_of(&blob[..120]);
    blob[120..128].copy_from_slice(&sum.to_le_bytes());
}

/// A copy of `source` whose dump `damage` changes, and the refusal or the
/// rebuild the load gives.
fn load_damaged(
    temp: &TempDir,
    source: &Path,
    name: &str,
    damage: impl FnOnce(&mut Vec<u8>),
) -> Result<(Collection, crate::Recovery), Error> {
    let target = temp.at(name);
    std::fs::create_dir_all(&target).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
    }
    let path = target.join(DUMP_FILENAME);
    let mut blob = std::fs::read(&path).unwrap();
    damage(&mut blob);
    std::fs::write(&path, &blob).unwrap();
    Collection::recover_with(
        target.to_str().unwrap(),
        None,
        JournalPolicy::Replay(Durability::default()),
    )
}

/// The check a refusal names, where the load was refused on the dump.
fn refused_on_dump(result: Result<(Collection, crate::Recovery), Error>, label: &str) -> String {
    match result {
        Err(Error::GraphDumpInvalid { name, detail }) => {
            assert_eq!(name, DUMP_FILENAME, "{label}");
            detail
        }
        Err(other) => panic!("{label}: expected the dump refused, got {other}"),
        Ok(_) => panic!("{label}: expected the dump refused, and the directory opened"),
    }
}

/// A raw l2 collection of three hundred records, two removed and one
/// overwritten, saved into `temp`.
fn raw_source(temp: &TempDir) -> (Collection, std::path::PathBuf) {
    let vectors = clustered(300, 8, 0x0219_c004);
    let collection = collection("l2", Shape::Raw, &vectors);
    let path = temp.at("source.zdb");
    collection.save(path.to_str().unwrap()).unwrap();
    (collection, path)
}

/// Each region's checksum broken, and each rule of the header, the identity
/// region and the rows region broken with every checksum repaired, refuses
/// the load naming the check. The words of the refusal are the approved
/// ones.
#[test]
fn a_dump_refused_on_its_header_identity_or_rows_refuses_the_load() {
    let temp = TempDir::new("directory-tests");
    let (collection, source) = raw_source(&temp);
    let blob = std::fs::read(source.join(DUMP_FILENAME)).unwrap();
    let (identity, rows, _, _) = regions(&blob);
    let nodes = u32::from_le_bytes(blob[32..36].try_into().unwrap()) as usize;
    let counter = collection.id_counter();
    let mut n = 0usize;
    let mut refused = |damage: &dyn Fn(&mut Vec<u8>)| {
        n += 1;
        refused_on_dump(
            load_damaged(&temp, &source, &format!("refused-{n}.zdb"), |blob| {
                damage(blob)
            }),
            &format!("case {n}"),
        )
    };

    assert_eq!(refused(&|b| b[32] ^= 0x01), "its header fails its checksum");
    assert_eq!(
        refused(&|b| b[identity + 70] ^= 0x01),
        "its identity region fails its checksum"
    );
    assert_eq!(
        refused(&|b| b[rows + 9] ^= 0x01),
        "its rows region fails its checksum"
    );
    assert_eq!(
        refused(&|b| {
            let end = b.len() - 1;
            b[end] ^= 0x01;
        }),
        "its trailer does not close the file its header names"
    );
    assert!(refused(&|b| {
        b.truncate(b.len() - 1);
    })
    .starts_with("the file is"));
    assert_eq!(
        refused(&|b| {
            b[36..40].copy_from_slice(&(nodes as u32).to_le_bytes());
            repaired(b);
        }),
        format!(
            "its header names entry node {} and the dump holds {} nodes",
            nodes, nodes
        )
    );
    assert_eq!(
        refused(&|b| {
            b[16] = 1;
            repaired(b);
        }),
        "its header names the cosine graph and config.json declares l2"
    );
    assert!(refused(&|b| {
        b[40..48].copy_from_slice(&0f64.to_le_bytes());
        repaired(b);
    })
    .starts_with("its header names a level scale of 0"));
    assert_eq!(
        refused(&|b| {
            let first = identity + 64;
            b[first..first + 4].copy_from_slice(&(counter as u32 + 1).to_le_bytes());
            repaired(b);
        }),
        format!(
            "its identity region names origin id {} and config.json counted {}",
            counter + 1,
            counter
        )
    );
    let second = u32::from_le_bytes(blob[identity + 68..identity + 72].try_into().unwrap());
    assert_eq!(
        refused(&|b| {
            let first = identity + 64;
            b[first..first + 4].copy_from_slice(&second.to_le_bytes());
            repaired(b);
        }),
        format!("its identity region names origin id {second} for more than one node")
    );
    let origin = u32::from_le_bytes(
        blob[identity + 64 + 4 * 2..identity + 64 + 4 * 3]
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        refused(&|b| {
            let third = rows + 2 * 8 * 4;
            b[third..third + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            repaired(b);
        }),
        format!("its rows region holds NaN in the row of node 2, internal id {origin}")
    );

    // The words, whole.
    let message = load_damaged(&temp, &source, "words.zdb", |b| b[rows + 9] ^= 0x01)
        .err()
        .unwrap()
        .to_string();
    assert_eq!(
        message,
        "The graph dump hnsw_index.zdbgraph cannot be used, since its rows region fails its \
         checksum. At format 5.0.0 the dump holds the only copy of each record's stored vector \
         or code, so the directory cannot open without it. Restore the directory from a copy, \
         such as one backup() wrote."
    );
}

/// A named edit of a dump's bytes.
type Forgery = Box<dyn Fn(&mut Vec<u8>)>;

/// A topology that fails, on its checksum or on a rule with every checksum
/// repaired, is rebuilt from the rows the dump holds: every record and
/// every internal id is the saved one, the recovery says the graph was
/// rebuilt, and a save of the rebuilt collection opens without a rebuild.
#[test]
fn a_dump_whose_topology_fails_is_rebuilt_from_its_rows() {
    let temp = TempDir::new("directory-tests");
    let (collection, source) = raw_source(&temp);
    let blob = std::fs::read(source.join(DUMP_FILENAME)).unwrap();
    let (_, _, topology, _) = regions(&blob);
    let expected = everything(&collection);
    let forgeries: [(&str, Forgery); 2] = [
        (
            "checksum",
            Box::new(move |b: &mut Vec<u8>| b[topology] ^= 0x01),
        ),
        (
            "target",
            Box::new(move |b: &mut Vec<u8>| {
                let first = topology + 1 + 2;
                b[first..first + 4].copy_from_slice(&u32::MAX.to_le_bytes());
                repaired(b);
            }),
        ),
    ];
    for (label, damage) in forgeries {
        let (rebuilt, recovery) =
            load_damaged(&temp, &source, &format!("rebuilt-{label}.zdb"), |b| {
                damage(b)
            })
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(recovery.graph_rebuilt, "{label}");
        assert_eq!(everything(&rebuilt), expected, "{label}");
        assert_eq!(rebuilt.id_counter(), collection.id_counter(), "{label}");
        assert!(rebuilt.positions_agree(), "{label}");
        let again = temp.at(&format!("rebuilt-{label}-saved.zdb"));
        rebuilt.save(again.to_str().unwrap()).unwrap();
        let (reopened, recovery) = Collection::recover_with(
            again.to_str().unwrap(),
            None,
            JournalPolicy::Replay(Durability::default()),
        )
        .unwrap();
        assert!(!recovery.graph_rebuilt, "{label}");
        assert_eq!(everything(&reopened), expected, "{label}");
    }
}

/// A quantized graph whose topology fails is rebuilt from its rows too: the
/// codes of a product quantized graph, with the raw vectors beside them
/// under `quantized_with_raw`, and the rows of a scalar one. Every record
/// reads back as it did, and the pages a search gives are those of a graph
/// over the same records.
#[test]
fn a_quantized_dump_whose_topology_fails_is_rebuilt_from_its_rows() {
    let temp = TempDir::new("directory-tests");
    let vectors = corpus(0);
    for (label, shape) in [
        ("pq-only", Shape::PqOnly),
        ("pq-raw", Shape::PqRaw),
        ("int8", Shape::Int8),
    ] {
        let collection = collection("l2", shape, &vectors);
        let path = temp.at(&format!("{label}.zdb"));
        collection.save(path.to_str().unwrap()).unwrap();
        break_topology(&path);
        let (rebuilt, recovery) = Collection::recover_with(
            path.to_str().unwrap(),
            None,
            JournalPolicy::Replay(Durability::default()),
        )
        .unwrap();
        assert!(recovery.graph_rebuilt, "{label}");
        assert!(rebuilt.is_quantized(), "{label}");
        assert_eq!(everything(&rebuilt), everything(&collection), "{label}");
        assert_eq!(
            rebuilt.stats()["quantized_codes_stored"],
            collection.stats()["quantized_codes_stored"],
            "{label}"
        );
        for query in &vectors[..5] {
            let query = rebuilt.process_vector_for_space(query.clone());
            let params = rebuilt.search_params(1, None, false, None).unwrap();
            let page = rebuilt.search_one(&query, None, params).unwrap();
            assert_eq!(page.len(), 1, "{label}");
        }
    }
}

/// The manifest of a 5.0.0 directory names the dump, and a directory that
/// lost it is refused naming what it held.
#[test]
fn a_directory_that_lost_its_dump_is_refused() {
    let temp = TempDir::new("directory-tests");
    let (_, source) = raw_source(&temp);
    std::fs::remove_file(source.join(DUMP_FILENAME)).unwrap();
    match Collection::load(source.to_str().unwrap()) {
        Err(Error::ArtefactsMissing { missing, contents }) => {
            assert_eq!(missing, vec![DUMP_FILENAME.to_string()]);
            assert_eq!(
                contents,
                "the graph and the stored vector or code of every record"
            );
        }
        other => panic!("expected the dump missing, got {:?}", other.map(|_| ())),
    }
}
