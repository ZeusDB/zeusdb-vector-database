//! The seam and its replay.
//!
//! Every mutation hands its record to the sink before it runs, in the order
//! the mutations ran, and a collection built from a checkpoint and the
//! records past it is the collection the script built, artefact for
//! artefact: the graph dump's bytes, the postings and dictionary artefacts'
//! bytes, every internal id, every record's metadata and vector, every code
//! and the codebook's bytes, and the training stamp. Every kind of record
//! is covered across the shapes below, the quantized path with training
//! firing inside a batch and the text path with term ids replaying in issue
//! order among them.

#![allow(clippy::disallowed_types)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use zeusdb_vector_core::{
    compile_filter, test_support::clustered, Error, Fitted, IdfScope, JournalRecord, Operation,
    OperationKind, SparseVector, DUMP_FILENAME, MAX_INTERNAL_ID,
};
use zeusdb_vector_sparse::{SparseConfig, Weighting};
use zeusdb_vector_text::{SimpleTokenizer, Tokenizer};

use super::{Collection, Declaration, OperationSink, ParsedRecord, SparseHalf, StorageMode};
use crate::RerankCalibration;

// ============================================================================
// FIXTURES
// ============================================================================

/// A directory under the system's temporary directory, removed on drop.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("zeusdb-replay-tests-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }

    fn sub(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What a recording sink has been handed.
#[derive(Debug, Default)]
struct Log {
    records: Vec<(OperationKind, Vec<u8>)>,
    commits: usize,
}

/// A sink that keeps every record's bytes, shared with the test that
/// attached it.
#[derive(Clone, Debug, Default)]
struct Recording(Arc<Mutex<Log>>);

impl Recording {
    fn records(&self) -> Vec<(OperationKind, Vec<u8>)> {
        self.0.lock().unwrap().records.clone()
    }

    fn kinds(&self) -> Vec<OperationKind> {
        self.0
            .lock()
            .unwrap()
            .records
            .iter()
            .map(|(kind, _)| *kind)
            .collect()
    }

    fn len(&self) -> usize {
        self.0.lock().unwrap().records.len()
    }

    fn commits(&self) -> usize {
        self.0.lock().unwrap().commits
    }
}

impl OperationSink for Recording {
    fn append(&mut self, kind: OperationKind, payload: &[u8]) -> Result<(), Error> {
        self.0
            .lock()
            .unwrap()
            .records
            .push((kind, payload.to_vec()));
        Ok(())
    }

    fn commit(&mut self) -> Result<(), Error> {
        self.0.lock().unwrap().commits += 1;
        Ok(())
    }
}

/// A sink that refuses every record.
#[derive(Debug)]
struct Refusing;

impl OperationSink for Refusing {
    fn append(&mut self, _kind: OperationKind, _payload: &[u8]) -> Result<(), Error> {
        Err(Error::JournalIoFailed {
            path: "refused.zdbwal".into(),
            what: "append to",
            error: "the sink refuses everything".into(),
        })
    }

    fn commit(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

/// Decode one recorded record as the journal reader would hand it over.
fn decode(index: usize, kind: OperationKind, payload: &[u8], dim: usize) -> Operation {
    let record = JournalRecord {
        sequence: index as u64 + 1,
        kind,
        offset: 0,
        payload,
    };
    Operation::decode(&record, dim, "recorded").unwrap()
}

fn metadata_for(i: usize) -> HashMap<String, Value> {
    HashMap::from([
        (
            "cat".to_string(),
            json!(if i.is_multiple_of(2) { "a" } else { "b" }),
        ),
        ("rank".to_string(), json!(i)),
    ])
}

fn record(i: usize, vectors: &[Vec<f32>]) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: vectors[i].clone(),
        sparse: None,
        metadata: metadata_for(i),
    }
}

/// A small vocabulary, so terms repeat across records and new ones keep
/// arriving.
fn text_for(i: usize) -> String {
    let words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu",
    ];
    format!(
        "{} {} {} w{}",
        words[i % 12],
        words[(i * 7 + 3) % 12],
        words[(i * 5 + 1) % 12],
        i / 3
    )
}

fn text_record(collection: &Collection, i: usize, vectors: &[Vec<f32>]) -> ParsedRecord {
    ParsedRecord {
        id: format!("r{i}"),
        vector: vectors[i].clone(),
        sparse: Some(SparseHalf::Terms(
            collection.tokenize(&text_for(i)).unwrap(),
        )),
        metadata: metadata_for(i),
    }
}

fn sparse_record(i: usize, vectors: &[Vec<f32>]) -> ParsedRecord {
    let mut dims: Vec<u32> = (0..4).map(|j| ((i * 7 + j * 13) % 50) as u32).collect();
    dims.sort_unstable();
    dims.dedup();
    let values: Vec<f32> = dims.iter().map(|d| 1.0 + (*d % 5) as f32).collect();
    ParsedRecord {
        id: format!("r{i}"),
        vector: vectors[i].clone(),
        sparse: (i % 3 != 2).then_some(SparseHalf::Vector(SparseVector { dims, values })),
        metadata: metadata_for(i),
    }
}

fn unit(vectors: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    vectors
        .into_iter()
        .map(|v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn add(collection: &Collection, records: Vec<ParsedRecord>) {
    let added = collection.add_records(records, vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

fn overwrite(collection: &Collection, records: Vec<ParsedRecord>) {
    let added = collection.add_records(records, vec![], true);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
}

fn batches(range: std::ops::Range<usize>, size: usize) -> Vec<Vec<usize>> {
    range
        .collect::<Vec<_>>()
        .chunks(size)
        .map(|c| c.to_vec())
        .collect()
}

fn filter(cat: &str) -> zeusdb_vector_core::Filter {
    compile_filter(&HashMap::from([("cat".to_string(), json!(cat))])).unwrap()
}

// ============================================================================
// WHAT TWO COLLECTIONS ARE COMPARED BY
// ============================================================================

/// The queries a snapshot's pages are taken with.
struct Queries {
    dense: Vec<f32>,
    text: Option<&'static str>,
    sparse: Option<SparseVector>,
}

/// One record as a snapshot holds it: its id, its metadata in key order
/// and its vector.
type RecordSnapshot = (String, BTreeMap<String, Value>, Option<Vec<f32>>);

/// Everything rule R2 compares, plus the pages a caller sees.
#[derive(Debug, PartialEq)]
struct Snapshot {
    ids: Vec<(String, usize)>,
    id_counter: usize,
    vector_count: usize,
    generated_ids: usize,
    positions_agree: bool,
    records: Vec<RecordSnapshot>,
    codes: Vec<(String, Vec<u8>)>,
    training_ids: Vec<String>,
    threshold_reached: bool,
    stamp: Option<String>,
    /// The rerank calibration as JSON, every field but its measured time.
    calibration: Option<String>,
    /// The calibration's measured time, a clock reading like the stamp.
    calibration_millis: Option<u64>,
    /// Values a scalar graph has clipped, which every replay must reach.
    saturated: u64,
    terms: Option<Vec<String>>,
    index_metadata: BTreeMap<String, String>,
    storage_mode: String,
    m: usize,
    ef_construction: usize,
    expected_size: usize,
    artefacts: BTreeMap<String, Vec<u8>>,
    dense_page: Vec<(String, f32)>,
    filtered_page: Vec<(String, f32)>,
    text_page: Option<Vec<(String, f32)>>,
    sparse_page: Option<Vec<(String, f32)>>,
}

impl Snapshot {
    /// Take a snapshot, saving the collection under `dir` to read the
    /// artefacts whose bytes are compared.
    fn take(collection: &Collection, dir: &str, queries: &Queries) -> Snapshot {
        collection.save(dir).unwrap();
        let path = std::path::Path::new(dir);
        let mut ids: Vec<(String, usize)> = collection
            .internal_ids()
            .into_iter()
            .map(|(internal, id)| (id, internal))
            .collect();
        ids.sort();
        let views = collection
            .records(ids.iter().map(|(id, _)| id.clone()).collect(), true, true)
            .unwrap();
        let records = views
            .into_iter()
            .map(|v| (v.id, v.metadata.into_iter().collect(), v.vector))
            .collect();
        let mut codes: Vec<(String, Vec<u8>)> = collection
            .pq_codes()
            .iter()
            .map(|(id, codes)| (id.clone(), codes.clone()))
            .collect();
        codes.sort();
        let terms = collection
            .sparse()
            .and_then(|space| space.text.as_ref())
            .map(|text| {
                text.dictionary
                    .read()
                    .unwrap()
                    .terms()
                    .into_iter()
                    .map(str::to_string)
                    .collect()
            });
        let mut names = collection.space_artefact_names();
        names.push(DUMP_FILENAME.to_string());
        names.push("pq_centroids.bin".to_string());
        names.push("int8_scales.zdbint8".to_string());
        names.push("int8_rows.zdbint8".to_string());
        let artefacts = names
            .into_iter()
            .filter_map(|name| {
                std::fs::read(path.join(&name))
                    .ok()
                    .map(|bytes| (name, bytes))
            })
            .collect();
        let training_ids = collection.training_ids().clone();
        let page = |filter: Option<&zeusdb_vector_core::Filter>| {
            let params = collection.search_params(5, None, false, None).unwrap();
            collection
                .search_one(&queries.dense, filter, params)
                .unwrap()
                .iter()
                .map(|hit| (hit.id().to_string(), hit.score()))
                .collect::<Vec<_>>()
        };
        Snapshot {
            ids,
            id_counter: collection.id_counter(),
            vector_count: collection.vector_count(),
            generated_ids: collection.generated_ids(),
            positions_agree: collection.positions_agree(),
            records,
            codes,
            training_ids,
            threshold_reached: collection.training_threshold_reached(),
            stamp: collection.training_completed_at(),
            calibration: collection
                .rerank_calibration()
                .map(|c| serde_json::to_string(&RerankCalibration { millis: 0, ..c }).unwrap()),
            calibration_millis: collection.rerank_calibration().map(|c| c.millis),
            saturated: collection.int8_saturated(),
            terms,
            index_metadata: collection.all_metadata().into_iter().collect(),
            storage_mode: collection.storage_mode(),
            m: collection.m(),
            ef_construction: collection.ef_construction(),
            expected_size: collection.expected_size(),
            artefacts,
            dense_page: page(None),
            filtered_page: page(Some(&filter("a"))),
            text_page: queries.text.map(|text| {
                collection
                    .search_text(text, None, 5, IdfScope::Corpus)
                    .unwrap()
            }),
            sparse_page: queries.sparse.as_ref().map(|sparse| {
                collection
                    .search_sparse(sparse.as_ref(), None, 5, IdfScope::Corpus)
                    .unwrap()
            }),
        }
    }

    /// Every field that differs, by name.
    fn diff(&self, other: &Snapshot) -> Vec<&'static str> {
        let mut out = Vec::new();
        macro_rules! cmp {
            ($field:ident) => {
                if self.$field != other.$field {
                    out.push(stringify!($field));
                }
            };
        }
        cmp!(ids);
        cmp!(id_counter);
        cmp!(vector_count);
        cmp!(generated_ids);
        cmp!(positions_agree);
        cmp!(records);
        cmp!(codes);
        cmp!(training_ids);
        cmp!(threshold_reached);
        cmp!(stamp);
        cmp!(calibration);
        cmp!(calibration_millis);
        cmp!(saturated);
        cmp!(terms);
        cmp!(index_metadata);
        cmp!(storage_mode);
        cmp!(m);
        cmp!(ef_construction);
        cmp!(expected_size);
        cmp!(artefacts);
        cmp!(dense_page);
        cmp!(filtered_page);
        cmp!(text_page);
        cmp!(sparse_page);
        out
    }
}

/// A script's run: the original collection with a recording sink, and the
/// checkpoint it saved part way through.
struct Run {
    original: Collection,
    recording: Recording,
    dir: TempDir,
    checkpoint: Option<(String, usize)>,
}

impl Run {
    fn new(collection: Collection) -> Self {
        let recording = Recording::default();
        assert!(collection
            .attach_sink(Box::new(recording.clone()))
            .is_none());
        Run {
            original: collection,
            recording,
            dir: TempDir::new(),
            checkpoint: None,
        }
    }

    /// Save the checkpoint and note how many records precede it.
    fn checkpoint(&mut self) {
        let dir = self.dir.sub("checkpoint.zdb");
        self.original.save(&dir).unwrap();
        self.checkpoint = Some((dir, self.recording.len()));
    }

    /// Load the checkpoint, apply every record past it, and hold the result
    /// to the original under rule R2, every saved artefact among it. A
    /// second replay from the same checkpoint is held to the first, and
    /// neither fits a quantizer, since a training's record carries what it
    /// fitted. Returns the kinds replayed.
    fn prove(
        &self,
        queries: &Queries,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> HashSet<OperationKind> {
        let (checkpoint, from) = self.checkpoint.clone().expect("a checkpoint was taken");
        let original = Snapshot::take(&self.original, &self.dir.sub("original.zdb"), queries);
        let records = self.recording.records();
        assert!(
            records.len() > from,
            "the script did something after the checkpoint"
        );
        let dim = self.original.dim();
        let mut kinds = HashSet::new();
        let replay = |label: &str| {
            let replayed = Collection::load_with(&checkpoint, tokenizer.clone()).unwrap();
            for (i, (kind, payload)) in records.iter().enumerate().skip(from) {
                let operation = decode(i, *kind, payload, dim);
                replayed
                    .apply(operation)
                    .unwrap_or_else(|e| panic!("record {} ({:?}): {}", i, kind, e));
            }
            Snapshot::take(&replayed, &self.dir.sub(&format!("{label}.zdb")), queries)
        };
        for (kind, _) in records.iter().skip(from) {
            kinds.insert(*kind);
        }
        let fits = zeusdb_vector_core::test_support::quantizer_fits();
        let first = replay("replayed");
        assert_eq!(original.diff(&first), Vec::<&str>::new());
        let second = replay("replayed-again");
        assert_eq!(first.diff(&second), Vec::<&str>::new());
        assert_eq!(
            zeusdb_vector_core::test_support::quantizer_fits(),
            fits,
            "a replay fits no quantizer"
        );
        assert!(!first.records.is_empty() || first.id_counter == 0);
        kinds
    }
}

// ============================================================================
// THE ORDER THE SEAM HANDS RECORDS OVER IN
// ============================================================================

/// The records arrive in the order the seam is built to keep: a term's record at
/// the moment its id is issued and ahead of the insert that carries it, a
/// removal ahead of the inserts that replace what it removed and in the same
/// call, every insert of a call handed over and committed before any is
/// installed, and the training's record between the insert that filled the
/// set and the one after it, so that the levels the records carry are the
/// ones the graph installed at.
#[test]
fn the_seam_hands_records_over_in_the_order_the_mutations_ran() {
    let vectors = clustered(20, 4, 0x0014_7b01);
    let declaration = Declaration::validate(4, "l2", 4, 50, 100, vec!["cat".to_string()])
        .unwrap()
        .with_text(
            "text",
            SparseConfig {
                weighting: Weighting::BM25,
                ..SparseConfig::default()
            },
            Arc::new(SimpleTokenizer),
        )
        .unwrap();
    let collection = Collection::build(declaration, None);
    let recording = Recording::default();
    collection.attach_sink(Box::new(recording.clone()));

    // Two records in one call: "alpha beta gamma w0" and "beta kappa zeta w0".
    add(
        &collection,
        vec![
            text_record(&collection, 0, &vectors),
            text_record(&collection, 1, &vectors),
        ],
    );
    use OperationKind::*;
    assert_eq!(
        recording.kinds(),
        vec![Intern, Intern, Intern, Intern, Insert, Intern, Intern, Insert]
    );
    assert_eq!(recording.commits(), 1, "one commit for the call");
    let records = recording.records();
    // The interns carry the ids in issue order and the inserts the ids and
    // levels the collection installed at.
    let terms: Vec<(u32, String)> = records
        .iter()
        .enumerate()
        .filter(|(_, (kind, _))| *kind == Intern)
        .map(|(i, (kind, payload))| match decode(i, *kind, payload, 4) {
            Operation::Intern { term_id, term } => (term_id, term),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(
        terms.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        (0..6).collect::<Vec<u32>>()
    );
    assert_eq!(terms[0].1, "alpha");
    assert_eq!(terms[1].1, "delta");
    for (i, (kind, payload)) in records.iter().enumerate() {
        if let Operation::Insert {
            id,
            internal_id,
            level,
            sparse,
            ..
        } = decode(i, *kind, payload, 4)
        {
            assert_eq!(collection.internal_id_of(&id), Some(internal_id as usize));
            assert!(level < 16);
            let sparse = sparse.expect("a text record carries its counted terms");
            assert!(sparse.dims.windows(2).all(|w| w[0] < w[1]));
        }
    }

    // An overwrite of a held record admits the replacement first, which
    // hands its new term over as it is counted, then hands over the removal
    // and the replacement's insert, and commits them as one call.
    overwrite(
        &collection,
        vec![text_record(&collection, 12, &vectors).with_id("r0")],
    );
    assert_eq!(recording.kinds()[8..], [Intern, Remove, Insert]);
    assert_eq!(recording.commits(), 2, "the overwrite commits once");

    // A removal names only the ids the index holds, once each, and a
    // removal of nothing is not recorded.
    assert_eq!(
        collection
            .remove_points(&["r1".to_string(), "absent".to_string(), "r1".to_string()])
            .unwrap(),
        vec!["absent".to_string()]
    );
    assert!(!collection.remove_point("absent".to_string()).unwrap());
    assert_eq!(collection.delete_ids(&["nothing".to_string()]).unwrap(), 0);
    let records = recording.records();
    assert_eq!(records.len(), 12);
    match decode(11, records[11].0, &records[11].1, 4) {
        Operation::Remove { ids } => assert_eq!(ids, vec!["r1".to_string()]),
        other => panic!("{other:?}"),
    }

    // The small kinds, in the order the calls ran.
    assert!(collection.update_metadata("r0", metadata_for(9)).unwrap());
    assert!(!collection
        .update_metadata("absent", metadata_for(9))
        .unwrap());
    collection
        .add_metadata(HashMap::from([
            ("z".to_string(), "last".to_string()),
            ("a".to_string(), "first".to_string()),
        ]))
        .unwrap();
    collection.compact().unwrap();
    let plan = collection.plan_rebuild(Some(6), None, None).unwrap();
    collection.rebuild(plan).unwrap();
    collection.clear().unwrap();
    let kinds = recording.kinds();
    assert_eq!(
        kinds[12..],
        [UpdateMetadata, AddMetadata, Compact, Rebuild, Clear]
    );
    let records = recording.records();
    match decode(13, records[13].0, &records[13].1, 4) {
        Operation::AddMetadata { pairs } => assert_eq!(
            pairs,
            vec![
                ("a".to_string(), "first".to_string()),
                ("z".to_string(), "last".to_string())
            ]
        ),
        other => panic!("{other:?}"),
    }
    match decode(15, records[15].0, &records[15].1, 4) {
        Operation::Rebuild {
            m,
            expected_size,
            ef_construction,
        } => assert_eq!((m, expected_size, ef_construction), (6, 100, 50)),
        other => panic!("{other:?}"),
    }
    assert_eq!(recording.commits(), 8);

    // Training inside a batch: the insert that fills the set ends its
    // segment, the training's record follows it, and the rest of the batch
    // follows that, each segment committed on its own.
    let vectors = clustered(1005, 8, 0x0014_7b02);
    let declaration = Declaration::validate(8, "l2", 4, 50, 2000, vec![]).unwrap();
    let quantization = declaration
        .quantization(2, 2, 1000, None, StorageMode::QuantizedWithRaw)
        .unwrap();
    let quantized = Collection::build(declaration, Some(quantization));
    let recording = Recording::default();
    quantized.attach_sink(Box::new(recording.clone()));
    add(&quantized, (0..1005).map(|i| record(i, &vectors)).collect());
    assert!(quantized.is_quantized());
    let kinds = recording.kinds();
    assert_eq!(kinds.len(), 1006);
    assert!(kinds[..1000].iter().all(|kind| *kind == Insert));
    assert_eq!(kinds[1000], Train);
    assert!(kinds[1001..].iter().all(|kind| *kind == Insert));
    assert_eq!(recording.commits(), 3);
    let records = recording.records();
    match decode(1000, records[1000].0, &records[1000].1, 8) {
        Operation::Train {
            completed_at,
            fitted,
        } => {
            assert_eq!(Some(completed_at), quantized.training_completed_at());
            assert!(
                matches!(
                    fitted,
                    Some(Fitted::Codebook {
                        subvectors: 2,
                        centroids: 4,
                        width: 4,
                        calibration: Some(_),
                        ..
                    })
                ),
                "the record carries the codebook and its calibration"
            );
        }
        other => panic!("{other:?}"),
    }
    quantized.rebuild_with_quantization().unwrap();
    assert_eq!(recording.kinds().last(), Some(&RebuildQuantized));
}

trait WithId {
    fn with_id(self, id: &str) -> Self;
}

impl WithId for ParsedRecord {
    fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }
}

/// A sink that refuses leaves the index and every map as they were, and
/// every entry point carries the refusal out, so a caller learns that
/// nothing was recorded rather than finding a mutation the journal never
/// saw. A record refused at admission leaves the counter where it was. An
/// overwrite refused once its admission is over leaves the ids admission
/// issued issued, as a withdrawn call does, and the fault with them.
#[test]
fn a_sink_that_refuses_leaves_the_collection_as_it_was() {
    let vectors = clustered(6, 4, 0x0014_7b03);
    let declaration = Declaration::validate(4, "l2", 4, 50, 100, vec!["cat".to_string()])
        .unwrap()
        .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
        .unwrap();
    let collection = Collection::build(declaration, None);
    add(&collection, vec![record(0, &vectors), record(1, &vectors)]);
    assert!(collection.attach_sink(Box::new(Refusing)).is_none());

    let added = collection.add_records(
        vec![
            record(2, &vectors),
            text_record(&collection, 3, &vectors),
            record(4, &vectors),
        ],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 3);
    assert_eq!(added.inserted, Vec::<String>::new());
    for error in &added.errors {
        assert!(error.contains("the sink refuses everything"), "{error}");
    }
    assert_eq!(collection.len(), 2);
    assert_eq!(
        collection.id_counter(),
        2,
        "the refused records burnt no id"
    );
    // A term's record is handed over before the term is interned, so the
    // refusal of the first stopped the count with nothing interned, which
    // is the state a replay of the records before it reaches.
    assert_eq!(collection.term_count(), Some(0));

    let refused = |result: Result<(), Error>| {
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("the sink refuses everything"),
            "{err}"
        );
    };
    refused(collection.remove_point("r0".to_string()).map(|_| ()));
    refused(collection.remove_points(&["r0".to_string()]).map(|_| ()));
    refused(collection.delete_ids(&["r0".to_string()]).map(|_| ()));
    refused(collection.remove_where(&filter("a")).map(|_| ()));
    refused(
        collection
            .update_metadata("r0", metadata_for(7))
            .map(|_| ()),
    );
    refused(collection.add_metadata(HashMap::from([("k".to_string(), "v".to_string())])));
    refused(collection.compact().map(|_| ()));
    refused(
        collection
            .rebuild(collection.plan_rebuild(Some(8), None, None).unwrap())
            .map(|_| ()),
    );
    refused(collection.clear().map(|_| ()));
    refused(collection.rebuild_with_quantization().map(|_| ()));
    // An overwrite of a held record is refused at the removal's record,
    // once the replacement is admitted, and leaves the fault.
    let added = collection.add_records(vec![record(0, &vectors)], vec![], true);
    assert_eq!(added.total_errors, 1);
    assert!(
        added.errors[0].contains("the sink refuses everything"),
        "{:?}",
        added.errors
    );
    let next = collection.add_records(vec![record(5, &vectors)], vec![], false);
    assert!(
        next.errors[0].contains("The journal's commit failed"),
        "{:?}",
        next.errors
    );
    assert_eq!(collection.id_counter(), 3, "the admitted id stays issued");

    assert_eq!(collection.len(), 2);
    assert_eq!(collection.m(), 4);
    assert!(collection.all_metadata().is_empty());
    assert_eq!(
        collection
            .records(vec!["r0".to_string()], false, true)
            .unwrap()[0]
            .metadata
            .get("rank"),
        Some(&json!(0))
    );

    // Detached, the fault goes with the sink, everything runs again and the
    // next ids follow the counter.
    assert!(collection.detach_sink().is_some());
    add(&collection, vec![record(2, &vectors), record(3, &vectors)]);
    assert_eq!(collection.internal_id_of("r2"), Some(4));
    assert_eq!(collection.internal_id_of("r3"), Some(5));
}

/// A record whose journal payload would be above the ceiling is refused at
/// the door, before its internal id is issued or its level drawn, so the
/// counter does not move and the graph built afterwards is the graph built
/// without the attempt.
#[test]
fn an_oversized_record_is_refused_before_the_counter_moves() {
    let vectors = clustered(40, 4, 0x0014_7b04);
    let build = || {
        Collection::build(
            Declaration::validate(4, "l2", 4, 50, 100, vec![]).unwrap(),
            None,
        )
    };
    let recorded = build();
    let recording = Recording::default();
    recorded.attach_sink(Box::new(recording.clone()));

    let mut oversized = record(0, &vectors);
    oversized.metadata.insert(
        "blob".to_string(),
        Value::String("x".repeat(zeusdb_vector_core::JOURNAL_MAX_PAYLOAD + 1)),
    );
    let added = recorded.add_records(vec![oversized], vec![], false);
    assert_eq!(added.total_errors, 1);
    assert!(
        added.errors[0].contains("journal's record ceiling"),
        "{}",
        added.errors[0]
    );
    assert!(
        added.errors[0].contains("ValueError"),
        "{}",
        added.errors[0]
    );
    assert_eq!(recorded.id_counter(), 0);
    assert_eq!(recorded.len(), 0);
    assert_eq!(recording.len(), 0);
    assert_eq!(
        recording.commits(),
        0,
        "nothing was admitted, so nothing was committed"
    );

    add(&recorded, (0..40).map(|i| record(i, &vectors)).collect());
    let plain = build();
    add(&plain, (0..40).map(|i| record(i, &vectors)).collect());
    let dir = TempDir::new();
    recorded.save(&dir.sub("recorded.zdb")).unwrap();
    plain.save(&dir.sub("plain.zdb")).unwrap();
    assert_eq!(
        std::fs::read(dir.path().join("recorded.zdb").join(DUMP_FILENAME)).unwrap(),
        std::fs::read(dir.path().join("plain.zdb").join(DUMP_FILENAME)).unwrap(),
        "the refused record consumed no draw"
    );
    assert_eq!(recording.len(), 40);
}

/// A collection that has issued its last internal id refuses the next
/// record at the door, before the sink is reached, and an overwrite that
/// needs more ids than are left is refused before anything is removed. The
/// counter, the level stream, the dictionary and every map stay where they
/// were.
///
/// The sink refuses everything, so a record that reached it would carry the
/// sink's words rather than the ceiling's.
#[test]
fn the_last_internal_id_is_refused_before_the_sink_is_reached() {
    let vectors = clustered(40, 4, 0x0018_0a01);
    let declaration = || {
        Declaration::validate(4, "l2", 4, 50, 100, vec!["cat".to_string()])
            .unwrap()
            .with_text("text", SparseConfig::default(), Arc::new(SimpleTokenizer))
            .unwrap()
    };
    let mut collection = Collection::build(declaration(), None);
    add(&collection, vec![record(0, &vectors), record(1, &vectors)]);
    collection.set_counters(MAX_INTERNAL_ID, 2);
    assert!(collection.attach_sink(Box::new(Refusing)).is_none());

    let exhausted = |errors: &[String], count: usize| {
        assert_eq!(errors.len(), count, "{errors:?}");
        for error in errors {
            assert!(
                error.contains(
                    "RuntimeError: This collection has issued its last internal id, 4294967294."
                ),
                "{error}"
            );
            assert!(error.contains("clear()"), "{error}");
        }
    };
    let added = collection.add_records(
        vec![record(2, &vectors), text_record(&collection, 3, &vectors)],
        vec![],
        false,
    );
    assert!(added.inserted.is_empty());
    exhausted(&added.errors, 2);
    // An overwrite of a held record, refused before its removal is handed
    // to the sink.
    let added = collection.add_records(vec![record(0, &vectors)], vec![], true);
    exhausted(&added.errors, 1);
    assert_eq!(collection.len(), 2);
    assert_eq!(collection.id_counter(), MAX_INTERNAL_ID);
    assert_eq!(
        collection.term_count(),
        Some(0),
        "no term of the refused record was counted"
    );

    // One id left, and an overwrite of two held records needs two.
    collection.set_counters(MAX_INTERNAL_ID - 1, 2);
    let added =
        collection.add_records(vec![record(0, &vectors), record(1, &vectors)], vec![], true);
    assert_eq!(added.total_errors, 2);
    for error in &added.errors {
        assert!(
            error.contains(
                "This collection has 1 internal id left, ending at 4294967294, and this \
                 overwrite needs 2"
            ),
            "{error}"
        );
    }
    assert_eq!(collection.len(), 2);
    assert!(collection.contains("r0") && collection.contains("r1"));
    assert_eq!(collection.id_counter(), MAX_INTERNAL_ID - 1);

    // Detached, with the counter put back where the two records left it,
    // the records after them draw the levels a collection that never met the
    // ceiling draws, so no refusal consumed a draw.
    assert!(collection.detach_sink().is_some());
    collection.set_counters(2, 2);
    add(&collection, (2..40).map(|i| record(i, &vectors)).collect());
    let plain = Collection::build(declaration(), None);
    add(&plain, (0..40).map(|i| record(i, &vectors)).collect());
    let dir = TempDir::new();
    collection.save(&dir.sub("refused.zdb")).unwrap();
    plain.save(&dir.sub("plain.zdb")).unwrap();
    assert_eq!(
        std::fs::read(dir.path().join("refused.zdb").join(DUMP_FILENAME)).unwrap(),
        std::fs::read(dir.path().join("plain.zdb").join(DUMP_FILENAME)).unwrap(),
        "the refusals consumed no draw"
    );
}

// ============================================================================
// A RECORD REFUSED FOR ITSELF
// ============================================================================

/// The id store's two ceilings, lowered on this thread so the refusals they
/// give are reached without gigabytes of text, and put back when dropped.
struct Ceilings((u64, u64));

impl Ceilings {
    fn lower(longest: u64, arena: u64) -> Self {
        Ceilings(
            zeusdb_vector_core::ID_CEILINGS.with(|ceilings| ceilings.replace((longest, arena))),
        )
    }
}

impl Drop for Ceilings {
    fn drop(&mut self) {
        zeusdb_vector_core::ID_CEILINGS.with(|ceilings| ceilings.set(self.0));
    }
}

/// A raw collection, or a product quantized one trained at 1,000 records,
/// at width 4 with the field `cat` declared.
fn refusals_collection(pq: bool) -> Collection {
    let declaration = Declaration::validate(4, "l2", 4, 50, 100, vec!["cat".to_string()]).unwrap();
    let quantization = pq.then(|| {
        declaration
            .quantization(2, 4, 1_000, None, StorageMode::QuantizedWithRaw)
            .unwrap()
    });
    Collection::build(declaration, quantization)
}

/// A record the id store would refuse for its id's length or for its arena,
/// or the graph for its vector's width, is refused at the door, before its
/// id is issued for good, its level drawn or its record handed to a sink.
/// The records before and after it in the same call go in under the ids and
/// levels a collection that never saw it gives them, so the collection is
/// that collection, artefact for artefact, on a raw and on a product
/// quantized collection, with a sink and without, and a replay of what the
/// sink holds reaches it as well. Each of the three was refused at install,
/// with the id issued, the level drawn and the record committed, and the
/// width on a raw collection reached the vector store's assertion.
#[test]
fn a_record_refused_for_its_id_or_its_width_moves_nothing() {
    let vectors = clustered(1_100, 4, 0x0208_b001);
    let queries = Queries {
        dense: vectors[3].clone(),
        text: None,
        sparse: None,
    };
    let long = "x".repeat(41);
    for (case, pq) in [
        ("long", false),
        ("arena", false),
        ("width", false),
        ("long", true),
        ("width", true),
    ] {
        for journaled in [false, true] {
            let label = format!("{case}, pq {pq}, journaled {journaled}");
            let held = if pq { 1_010 } else { 30 };
            let bad = ParsedRecord {
                id: if case == "width" {
                    "wide".to_string()
                } else {
                    long.clone()
                },
                vector: if case == "width" {
                    vec![0.5; 5]
                } else {
                    vectors[held + 1].clone()
                },
                sparse: None,
                metadata: metadata_for(held + 1),
            };
            let words = match case {
                "long" => "an id of 41 bytes is longer than the id store can hold".to_string(),
                "arena" => "the id store's text arena is full".to_string(),
                _ => "Vector dimension mismatch: expected 4, got 5".to_string(),
            };
            let batch = |bad: Option<ParsedRecord>| {
                let mut batch = vec![record(held, &vectors)];
                batch.extend(bad);
                batch.push(record(held + 2, &vectors));
                batch
            };

            let collection = refusals_collection(pq);
            add(
                &collection,
                (0..held).map(|i| record(i, &vectors)).collect(),
            );
            let mut run = journaled.then(|| {
                let mut run = Run::new(refusals_collection(pq));
                add(
                    &run.original,
                    (0..held).map(|i| record(i, &vectors)).collect(),
                );
                run.checkpoint();
                run
            });
            let arena = collection.ids().arena_len() as u64;
            let ceilings = match case {
                "long" => Ceilings::lower(40, u64::MAX),
                "arena" => Ceilings::lower(u64::MAX, arena + format!("r{held}").len() as u64 + 40),
                _ => Ceilings::lower(u64::MAX, u64::MAX),
            };
            for target in std::iter::once(&collection).chain(run.as_ref().map(|run| &run.original))
            {
                let added = target.add_records(batch(Some(bad.clone())), vec![], false);
                assert_eq!(added.total_errors, 1, "{label}: {:?}", added.errors);
                assert!(
                    added.errors[0].contains(&words),
                    "{label}: {:?}",
                    added.errors
                );
                assert_eq!(
                    added.inserted,
                    vec![format!("r{held}"), format!("r{}", held + 2)],
                    "{label}"
                );
            }
            drop(ceilings);

            let twin = refusals_collection(pq);
            add(&twin, (0..held).map(|i| record(i, &vectors)).collect());
            add(&twin, batch(None));
            assert_eq!(collection.is_quantized(), pq, "{label}");
            let dir = TempDir::new();
            let ours = Snapshot::take(&collection, &dir.sub("ours.zdb"), &queries);
            let theirs = Snapshot::take(&twin, &dir.sub("twin.zdb"), &queries);
            // The training stamp and the calibration's measured time are each
            // collection's own clock readings.
            let differ = |snapshot: &Snapshot| -> Vec<&'static str> {
                snapshot
                    .diff(&theirs)
                    .into_iter()
                    .filter(|field| *field != "stamp" && *field != "calibration_millis")
                    .collect()
            };
            assert_eq!(differ(&ours), Vec::<&str>::new(), "{label}");
            if let Some(run) = run.as_mut() {
                let journaled = Snapshot::take(&run.original, &dir.sub("journaled.zdb"), &queries);
                assert_eq!(differ(&journaled), Vec::<&str>::new(), "{label}");
                assert!(
                    run.recording
                        .records()
                        .iter()
                        .all(|(_, payload)| !payload.windows(4).any(|w| w == b"wide")
                            && !payload.windows(41).any(|w| w == long.as_bytes())),
                    "{label}: the refused record never reached the sink"
                );
                run.prove(&queries, None);
            }
        }
    }
}

/// A collection with a sparse space under BM25, whose values must be whole,
/// at width 4, or with none where `sparse` is unset.
fn overwrite_collection(sparse: bool) -> Collection {
    let declaration = Declaration::validate(4, "l2", 4, 50, 100, vec!["cat".to_string()]).unwrap();
    let declaration = if sparse {
        declaration
            .with_sparse(
                "terms",
                SparseConfig {
                    weighting: Weighting::BM25,
                    ..SparseConfig::default()
                },
            )
            .unwrap()
    } else {
        declaration
    };
    Collection::build(declaration, None)
}

/// An overwrite holds each new record that replaces a held one to every
/// refusal of the door that depends on the record alone before it removes
/// anything, so a new record refused for itself keeps the record it would
/// have replaced, and the others of the batch replace theirs in one removal
/// and go in under the ids a collection that never saw the refused record
/// gives them. Covered: a vector of another width, an id past the store's
/// ceiling, a sparse vector on a collection with no sparse space, a sparse
/// vector against its weighting's rules, terms on a space with no text
/// layer, and, with a sink, a record past the journal's ceiling. Each used
/// to lose the record it replaced, removed and recorded as removed before
/// the new record was refused.
#[test]
fn an_overwrite_keeps_the_record_its_refused_replacement_names() {
    let vectors = clustered(40, 4, 0x0208_b002);
    let queries = Queries {
        dense: vectors[3].clone(),
        text: None,
        sparse: None,
    };
    let long = "x".repeat(41);
    for case in ["width", "long", "no space", "rules", "no text", "journal"] {
        for journaled in [false, true] {
            if case == "journal" && !journaled {
                continue;
            }
            let label = format!("{case}, journaled {journaled}");
            let sparse = matches!(case, "rules" | "no text");
            // The record the batch fails to replace, held under the id
            // the refused record names.
            let held_id = if case == "long" {
                long.clone()
            } else {
                "r2".to_string()
            };
            let held = |i: usize| ParsedRecord {
                id: if i == 2 {
                    held_id.clone()
                } else {
                    format!("r{i}")
                },
                ..record(i, &vectors)
            };
            let replacement = |i: usize| ParsedRecord {
                id: if i == 2 {
                    held_id.clone()
                } else {
                    format!("r{i}")
                },
                metadata: HashMap::from([("cat".to_string(), json!("replaced"))]),
                ..record(i + 20, &vectors)
            };
            let mut bad = replacement(2);
            let words = match case {
                "width" => {
                    bad.vector = vec![0.5; 5];
                    "Vector dimension mismatch: expected 4, got 5".to_string()
                }
                "long" => "an id of 41 bytes is longer than the id store can hold".to_string(),
                "no space" => {
                    bad.sparse = Some(SparseHalf::Vector(SparseVector {
                        dims: vec![1, 2],
                        values: vec![1.0, 2.0],
                    }));
                    Error::NoSparseSpace.to_string()
                }
                "rules" => {
                    bad.sparse = Some(SparseHalf::Vector(SparseVector {
                        dims: vec![1, 2],
                        values: vec![1.0, 1.5],
                    }));
                    Error::SparseValueNotWhole {
                        index: 1,
                        value: 1.5,
                    }
                    .to_string()
                }
                "no text" => {
                    bad.sparse = Some(SparseHalf::Terms(vec!["alpha".to_string()]));
                    Error::NoTextLayer.to_string()
                }
                _ => {
                    bad.metadata.insert(
                        "note".to_string(),
                        json!("a".repeat(zeusdb_vector_core::JOURNAL_MAX_PAYLOAD)),
                    );
                    format!(
                        "the journal's record ceiling is {} bytes",
                        zeusdb_vector_core::JOURNAL_MAX_PAYLOAD
                    )
                }
            };
            let batch = |bad: Option<ParsedRecord>| {
                let mut batch = vec![replacement(1)];
                batch.extend(bad);
                batch.push(replacement(3));
                batch.push(ParsedRecord {
                    id: "r20".to_string(),
                    ..record(30, &vectors)
                });
                batch
            };

            let mut run = Run::new(overwrite_collection(sparse));
            if !journaled {
                assert!(run.original.detach_sink().is_some());
            }
            add(&run.original, (0..10).map(held).collect());
            run.checkpoint();
            let ceilings = (case == "long").then(|| Ceilings::lower(40, u64::MAX));
            let added = run.original.add_records(batch(Some(bad)), vec![], true);
            drop(ceilings);
            assert_eq!(added.total_errors, 1, "{label}: {:?}", added.errors);
            assert!(
                added.errors[0].contains(&words),
                "{label}: {:?}",
                added.errors
            );
            assert_eq!(
                added.inserted,
                vec!["r1".to_string(), "r3".to_string(), "r20".to_string()],
                "{label}"
            );
            let kept = run
                .original
                .records(vec![held_id.clone()], true, true)
                .unwrap();
            assert_eq!(kept.len(), 1, "{label}: the replaced record is kept");
            assert_eq!(kept[0].vector.as_ref(), Some(&vectors[2]), "{label}");
            assert_eq!(kept[0].metadata, metadata_for(2), "{label}");

            let twin = overwrite_collection(sparse);
            add(&twin, (0..10).map(held).collect());
            overwrite(&twin, batch(None));
            let dir = TempDir::new();
            let ours = Snapshot::take(&run.original, &dir.sub("ours.zdb"), &queries);
            let theirs = Snapshot::take(&twin, &dir.sub("twin.zdb"), &queries);
            assert_eq!(ours.diff(&theirs), Vec::<&str>::new(), "{label}");
            if journaled {
                run.prove(&queries, None);
            }
        }
    }
}

/// An add with overwrite off that names an id the collection holds is
/// refused at the door in the duplicate's words, and the record it names,
/// the counter and the journal stay as they were for it, while a new record
/// of the same batch goes in. The same add with overwrite on replaces the
/// record.
#[test]
fn an_add_without_overwrite_refuses_a_held_id_and_keeps_its_record() {
    let vectors = clustered(30, 4, 0x0208_b003);
    let mut run = Run::new(overwrite_collection(false));
    add(
        &run.original,
        (0..10).map(|i| record(i, &vectors)).collect(),
    );
    let counter = run.original.id_counter();
    let journaled = run.recording.len();
    let again = || ParsedRecord {
        id: "r2".to_string(),
        ..record(22, &vectors)
    };

    let added = run
        .original
        .add_records(vec![again(), record(20, &vectors)], vec![], false);
    let words = Error::DuplicateId {
        id: "r2".to_string(),
    }
    .to_string();
    assert_eq!(added.total_errors, 1, "{:?}", added.errors);
    assert!(added.errors[0].contains(&words), "{:?}", added.errors);
    assert_eq!(added.inserted, vec!["r20".to_string()]);
    let kept = run
        .original
        .records(vec!["r2".to_string()], true, true)
        .unwrap();
    assert_eq!(kept.len(), 1, "the held record is kept");
    assert_eq!(kept[0].vector.as_ref(), Some(&vectors[2]));
    assert_eq!(kept[0].metadata, metadata_for(2));
    assert_eq!(run.original.len(), 11);
    assert_eq!(
        run.original.id_counter(),
        counter + 1,
        "only the new record took an id"
    );
    assert_eq!(
        run.recording.len(),
        journaled + 1,
        "only the new record was journaled"
    );

    overwrite(&run.original, vec![again()]);
    let replaced = run
        .original
        .records(vec!["r2".to_string()], true, true)
        .unwrap();
    assert_eq!(replaced[0].vector.as_ref(), Some(&vectors[22]));
    assert_eq!(run.original.len(), 11);
    run.checkpoint();
}

/// A recorded insert naming an id the id store could not name is refused
/// before its id is issued, in the words the store gives, and the collection
/// is as it was. Admission refuses such a record before it is recorded, so
/// only a journal written elsewhere can hold one.
#[test]
fn apply_refuses_a_recorded_id_the_store_could_not_name() {
    let collection = Collection::build(
        Declaration::validate(4, "l2", 4, 50, 100, vec![]).unwrap(),
        None,
    );
    let ceilings = Ceilings::lower(40, u64::MAX);
    let refused = collection.apply(Operation::Insert {
        id: "x".repeat(41),
        internal_id: 1,
        level: 0,
        vector: vec![0.0; 4],
        metadata: serde_json::Map::new(),
        sparse: None,
    });
    drop(ceilings);
    match refused {
        Err(Error::Engine(message)) => assert_eq!(
            message,
            "an id of 41 bytes is longer than the id store can hold"
        ),
        other => panic!("expected the id store's refusal, got {other:?}"),
    }
    assert_eq!(collection.len(), 0);
    assert_eq!(collection.id_counter(), 0);
}

/// A saved collection whose mappings name an id the id store could not name
/// is refused at load, in the words the store gives, for the id's length
/// and for its arena. The ceilings are lowered for the load alone, so the
/// save writes what the collection holds, and under the store's own
/// ceilings the same directory loads whole.
#[test]
fn a_load_refuses_a_mapped_id_the_store_could_not_name() {
    let dir = TempDir::new();
    let collection = Collection::build(
        Declaration::validate(4, "l2", 4, 50, 100, vec![]).unwrap(),
        None,
    );
    let named = |id: String, axis: usize| {
        let mut vector = vec![0.0; 4];
        vector[axis] = 1.0;
        ParsedRecord {
            id,
            vector,
            sparse: None,
            metadata: HashMap::new(),
        }
    };
    add(
        &collection,
        vec![named("short".to_string(), 0), named("x".repeat(41), 1)],
    );
    let path = dir.sub("named.zdb");
    collection.save(&path).unwrap();
    for (longest, arena, words) in [
        (
            40,
            u64::MAX,
            "an id of 41 bytes is longer than the id store can hold",
        ),
        (u64::MAX, 45, "the id store's text arena is full"),
    ] {
        let ceilings = Ceilings::lower(longest, arena);
        let refused = Collection::load(&path);
        drop(ceilings);
        match refused {
            Err(err) => assert!(err.to_string().contains(words), "{words}: {err}"),
            Ok(_) => panic!("{words}: the load was not refused"),
        }
    }
    assert_eq!(Collection::load(&path).unwrap().len(), 2);
}

// ============================================================================
// APPLY'S CHECKS
// ============================================================================

/// A record that does not belong to the collection is refused by name,
/// nothing panics, and the collection is as it was before the record.
#[test]
fn apply_refuses_a_record_that_does_not_belong_to_the_collection() {
    let declaration = Declaration::validate(4, "l2", 4, 50, 100, vec![])
        .unwrap()
        .with_text(
            "text",
            SparseConfig {
                weighting: Weighting::BM25,
                ..SparseConfig::default()
            },
            Arc::new(SimpleTokenizer),
        )
        .unwrap();
    let quantization = declaration
        .quantization(2, 2, 1000, None, StorageMode::QuantizedOnly)
        .unwrap();
    let collection = Collection::build(declaration, Some(quantization));
    let insert =
        |internal_id: u64, vector: Vec<f32>, sparse: Option<SparseVector>| Operation::Insert {
            id: "r0".into(),
            internal_id,
            level: 0,
            vector,
            metadata: serde_json::Map::new(),
            sparse,
        };
    let mismatch = |operation: Operation, expected: &str| match collection.apply(operation) {
        Err(Error::JournalReplayMismatch { detail }) => {
            assert!(detail.contains(expected), "{detail}")
        }
        other => panic!("expected a mismatch naming {expected}, got {:?}", other),
    };
    let unchanged = || {
        assert_eq!(collection.len(), 0);
        assert_eq!(collection.id_counter(), 0);
        assert_eq!(collection.term_count(), Some(0));
        assert!(!collection.is_quantized());
        assert!(collection.training_completed_at().is_none());
    };

    // The internal id against the counter, and under the id ceiling.
    mismatch(insert(5, vec![0.0; 4], None), "would issue 1");
    mismatch(insert(0, vec![0.0; 4], None), "would issue 1");
    mismatch(
        insert(u32::MAX as u64, vec![0.0; 4], None),
        "above the id ceiling",
    );
    mismatch(
        insert(u32::MAX as u64 + 1, vec![0.0; 4], None),
        "above the id ceiling",
    );
    unchanged();

    // The vector, as add's door holds it.
    assert!(matches!(
        collection.apply(insert(1, vec![0.0; 3], None)),
        Err(Error::VectorDimension {
            expected: 4,
            got: 3
        })
    ));
    assert!(matches!(
        collection.apply(insert(1, vec![0.0, f32::NAN, 0.0, 0.0], None)),
        Err(Error::VectorNotFinite { index: 1, .. })
    ));
    unchanged();

    // The sparse rules under the weighting: term frequency weighting takes
    // whole positive values.
    let fractional = SparseVector {
        dims: vec![0],
        values: vec![0.5],
    };
    assert!(collection
        .apply(insert(1, vec![0.0; 4], Some(fractional)))
        .is_err());
    unchanged();

    // A term id against the dictionary.
    mismatch(
        Operation::Intern {
            term_id: 1,
            term: "alpha".into(),
        },
        "would issue 0",
    );
    collection
        .apply(Operation::Intern {
            term_id: 0,
            term: "alpha".into(),
        })
        .unwrap();
    mismatch(
        Operation::Intern {
            term_id: 1,
            term: "alpha".into(),
        },
        "holds it at 0",
    );
    assert_eq!(collection.term_count(), Some(1));

    // A removal and a metadata update of a record the collection lacks.
    mismatch(
        Operation::Remove {
            ids: vec!["r0".into()],
        },
        "does not hold it",
    );
    mismatch(
        Operation::UpdateMetadata {
            id: "r0".into(),
            metadata: serde_json::Map::new(),
        },
        "does not hold it",
    );

    // The rebuild's three through the rules create() applies.
    assert!(matches!(
        collection.apply(Operation::Rebuild {
            m: 4,
            expected_size: 0,
            ef_construction: 50
        }),
        Err(Error::ExpectedSizeZero { .. })
    ));
    assert!(matches!(
        collection.apply(Operation::Rebuild {
            m: 4,
            expected_size: u64::MAX,
            ef_construction: 50
        }),
        Err(Error::ExpectedSizeTooLarge { .. }) | Err(Error::JournalReplayMismatch { .. })
    ));
    assert_eq!(collection.m(), 4);

    // A training that carries nothing it fitted, and one before the
    // collection has filled its training set.
    mismatch(
        Operation::Train {
            completed_at: "2001-02-03T04:05:06+00:00".into(),
            fitted: None,
        },
        "carries nothing it fitted",
    );
    mismatch(
        Operation::Train {
            completed_at: "2001-02-03T04:05:06+00:00".into(),
            fitted: Some(Fitted::Scales {
                values: vec![1.0; 4],
            }),
        },
        "has not filled its training set",
    );

    // Nothing is applied while a sink is attached.
    collection.attach_sink(Box::new(Recording::default()));
    match collection.apply(Operation::Clear) {
        Err(Error::Engine(message)) => assert!(message.contains("while a sink is attached")),
        other => panic!("{other:?}"),
    }
    collection.detach_sink();

    // And a record that does belong is applied, the id issued being the one
    // it names and the record installed at the level it names.
    collection.apply(insert(1, vec![0.5; 4], None)).unwrap();
    assert_eq!(collection.internal_id_of("r0"), Some(1));
    assert_eq!(collection.id_counter(), 1);
    mismatch(insert(1, vec![0.5; 4], None), "already holds it");
}

// ============================================================================
// REPLAY REPRODUCES EVERY ARTEFACT
// ============================================================================

/// A raw collection with a declared field, under cosine, through every
/// kind a raw collection can record: inserts, removals by id, by filter and
/// by overwrite, a metadata replacement, index metadata, a compaction, a
/// rebuild to another degree, and a clear.
#[test]
fn replay_reproduces_a_raw_collection_kind_for_kind() {
    let vectors = unit(clustered(400, 8, 0x0014_7b10));
    let declaration =
        Declaration::validate(8, "cosine", 8, 60, 500, vec!["cat".to_string()]).unwrap();
    let mut run = Run::new(Collection::build(declaration, None));
    let c = &run.original;
    for batch in batches(0..100, 25) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    assert!(c.remove_point("r3".to_string()).unwrap());
    assert_eq!(
        c.remove_points(&["r7".to_string(), "r8".to_string()])
            .unwrap()
            .len(),
        0
    );
    let mut edited = metadata_for(5);
    edited.insert("edited".to_string(), json!(true));
    assert!(c.update_metadata("r5", edited).unwrap());
    c.add_metadata(HashMap::from([("owner".to_string(), "test".to_string())]))
        .unwrap();
    run.checkpoint();
    let c = &run.original;

    for batch in batches(100..200, 25) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    // An overwrite of two records that exist, with new vectors.
    overwrite(
        c,
        vec![
            ParsedRecord {
                vector: vectors[390].clone(),
                ..record(10, &vectors)
            },
            ParsedRecord {
                vector: vectors[391].clone(),
                ..record(11, &vectors)
            },
        ],
    );
    assert_eq!(
        c.delete_ids(&["r150".to_string(), "r150".to_string()])
            .unwrap(),
        1
    );
    let low = compile_filter(&HashMap::from([
        ("cat".to_string(), json!("b")),
        ("rank".to_string(), json!({"lt": 20})),
    ]))
    .unwrap();
    assert!(c.remove_where(&low).unwrap() > 0);
    let mut edited = metadata_for(120);
    edited.insert("edited".to_string(), json!("twice"));
    assert!(c.update_metadata("r120", edited).unwrap());
    assert!(c.compact().unwrap() > 0);
    let plan = c.plan_rebuild(Some(12), Some(600), None).unwrap();
    c.rebuild(plan).unwrap();
    for batch in batches(200..250, 25) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    c.clear().unwrap();
    for batch in batches(250..300, 25) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    c.add_metadata(HashMap::from([(
        "phase".to_string(),
        "after clear".to_string(),
    )]))
    .unwrap();

    let queries = Queries {
        dense: vectors[395].clone(),
        text: None,
        sparse: None,
    };
    let kinds = run.prove(&queries, None);
    use OperationKind::*;
    for kind in [
        Insert,
        Remove,
        UpdateMetadata,
        AddMetadata,
        Compact,
        Rebuild,
        Clear,
    ] {
        assert!(kinds.contains(&kind), "{kind:?} was replayed");
    }
}

/// Training fires inside a batch after the checkpoint, in both storage
/// modes, and the replay installs the codebook, the calibration and the
/// stamp the training's record carries, fitting nothing, to the same codes
/// and the same graph. The quantized rebuild, a compaction over codes and an
/// overwrite into the quantized graph follow.
#[test]
fn replay_reproduces_training_inside_a_batch() {
    for (mode, seed) in [
        (StorageMode::QuantizedWithRaw, 0x0014_7b20_u64),
        (StorageMode::QuantizedOnly, 0x0014_7b21_u64),
    ] {
        let vectors = clustered(1400, 16, seed);
        let declaration = Declaration::validate(16, "l2", 8, 60, 2000, vec![]).unwrap();
        let quantization = declaration
            .quantization(4, 4, 1000, None, mode.clone())
            .unwrap();
        let mut run = Run::new(Collection::build(declaration, Some(quantization)));
        let c = &run.original;
        for batch in batches(0..600, 200) {
            add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
        }
        run.checkpoint();
        let c = &run.original;
        assert!(!c.is_quantized());

        // 600 held, the set fills at 1,000, so the third batch of this loop
        // trains part way through.
        for batch in batches(600..1200, 150) {
            add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
        }
        assert!(c.is_quantized());
        assert!(c.remove_point("r1010".to_string()).unwrap());
        assert!(c.compact().unwrap() > 0);
        overwrite(
            c,
            vec![ParsedRecord {
                vector: vectors[1399].clone(),
                ..record(1020, &vectors)
            }],
        );
        assert!(c.rebuild_with_quantization().unwrap());
        for batch in batches(1200..1300, 50) {
            add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
        }

        let queries = Queries {
            dense: vectors[1398].clone(),
            text: None,
            sparse: None,
        };
        let kinds = run.prove(&queries, None);
        use OperationKind::*;
        for kind in [Insert, Train, Remove, Compact, RebuildQuantized] {
            assert!(
                kinds.contains(&kind),
                "{kind:?} was replayed under {mode:?}"
            );
        }
        assert!(run.original.training_completed_at().is_some());
    }
}

/// The scalar counterpart of the test above: a checkpoint before the set
/// fills, training inside a batch, and the mutations after it, replayed
/// under rule R2 with the scales, the rows and the clipped count among what
/// is compared. The replay installs the scales the training's record
/// carries, and the rows are a function of the codec and the records.
#[test]
fn replay_reproduces_scalar_training_inside_a_batch() {
    let vectors = clustered(1400, 16, 0x0015_7b20);
    let declaration = Declaration::validate(16, "cosine", 8, 60, 2000, vec![]).unwrap();
    let quantization = declaration
        .scalar_quantization("per_dimension", 1000, None, StorageMode::QuantizedOnly)
        .unwrap();
    let mut run = Run::new(Collection::build(declaration, Some(quantization)));
    let c = &run.original;
    let vectors = unit(vectors);
    for batch in batches(0..600, 200) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    run.checkpoint();
    let c = &run.original;
    assert!(!c.is_quantized());

    // 600 held, the set fills at 1,000, so the third batch of this loop
    // trains part way through and the records after it encode through the
    // scales the batch itself fitted.
    for batch in batches(600..1200, 150) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    assert!(c.is_quantized());
    assert_eq!(c.stats()["quantization_type"], "int8");
    assert!(c.remove_point("r1010".to_string()).unwrap());
    assert!(c.compact().unwrap() > 0);
    overwrite(
        c,
        vec![ParsedRecord {
            vector: vectors[1399].clone(),
            ..record(1020, &vectors)
        }],
    );
    assert!(c.rebuild_with_quantization().unwrap());
    let plan = c.plan_rebuild(Some(6), None, Some(80)).unwrap();
    c.rebuild(plan).unwrap();
    for batch in batches(1200..1300, 50) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    c.clear().unwrap();
    assert!(c.is_quantized(), "a clear keeps the scales");
    for batch in batches(1300..1400, 25) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }

    let queries = Queries {
        dense: vectors[1398].clone(),
        text: None,
        sparse: None,
    };
    let kinds = run.prove(&queries, None);
    use OperationKind::*;
    for kind in [
        Insert,
        Train,
        Remove,
        Compact,
        RebuildQuantized,
        Rebuild,
        Clear,
    ] {
        assert!(kinds.contains(&kind), "{kind:?} was replayed");
    }
    assert!(run.original.training_completed_at().is_some());
    assert!(run.original.int8_codec().is_some());
}

/// A training's record is held to the collection before anything is
/// installed: the quantizer it carries is the one the collection declares,
/// at the declared shape, with every value finite and a calibration only
/// where the collection reranks, and the collection holds its quantizer
/// unfitted with its set full. Each refusal leaves the collection
/// untrained, and the record as written installs once.
#[test]
fn a_training_record_is_held_to_the_collection() {
    let vectors = clustered(1000, 8, 0x0219_b002);
    let declaration = Declaration::validate(8, "l2", 8, 60, 2000, vec![]).unwrap();
    let quantization = declaration
        .quantization(2, 4, 1000, None, StorageMode::QuantizedOnly)
        .unwrap();
    let mut run = Run::new(Collection::build(declaration, Some(quantization)));
    run.checkpoint();
    add(
        &run.original,
        (0..1000).map(|i| record(i, &vectors)).collect(),
    );
    assert!(run.original.is_quantized());
    let records = run.recording.records();
    let train_at = records
        .iter()
        .position(|(kind, _)| *kind == OperationKind::Train)
        .unwrap();
    let (checkpoint, _) = run.checkpoint.clone().unwrap();
    let replayed = Collection::load_with(&checkpoint, None).unwrap();
    for (i, (kind, payload)) in records.iter().enumerate().take(train_at) {
        replayed.apply(decode(i, *kind, payload, 8)).unwrap();
    }
    assert!(!replayed.is_quantized() && replayed.is_training_ready());
    let (completed_at, fitted) =
        match decode(train_at, records[train_at].0, &records[train_at].1, 8) {
            Operation::Train {
                completed_at,
                fitted: Some(fitted),
            } => (completed_at, fitted),
            other => panic!("{other:?}"),
        };
    let Fitted::Codebook {
        subvectors,
        centroids,
        width,
        values,
        ..
    } = fitted.clone()
    else {
        panic!("{fitted:?}")
    };
    let refused = |fitted: Fitted, expected: &str| match replayed.apply(Operation::Train {
        completed_at: completed_at.clone(),
        fitted: Some(fitted),
    }) {
        Err(Error::JournalReplayMismatch { detail }) => {
            assert!(detail.contains(expected), "{detail}")
        }
        other => panic!("{other:?}"),
    };
    refused(
        Fitted::Codebook {
            subvectors: 4,
            centroids,
            width: 2,
            values: values.clone(),
            calibration: None,
        },
        "4 subvectors",
    );
    refused(
        Fitted::Codebook {
            subvectors,
            centroids: 8,
            width,
            values: values[..2 * 8 * 4].to_vec(),
            calibration: None,
        },
        "8 centroids",
    );
    let mut not_finite = values.clone();
    not_finite[3] = f32::NAN;
    refused(
        Fitted::Codebook {
            subvectors,
            centroids,
            width,
            values: not_finite,
            calibration: None,
        },
        "value 3",
    );
    refused(
        Fitted::Codebook {
            subvectors,
            centroids,
            width,
            values: values.clone(),
            calibration: Some("{}".into()),
        },
        "keeps no raw vector",
    );
    refused(
        Fitted::Scales {
            values: vec![1.0; 8],
        },
        "declares product quantization",
    );
    assert!(!replayed.is_quantized(), "every refusal installed nothing");

    replayed
        .apply(Operation::Train {
            completed_at: completed_at.clone(),
            fitted: Some(fitted.clone()),
        })
        .unwrap();
    assert!(replayed.is_quantized());
    assert_eq!(replayed.training_completed_at(), Some(completed_at.clone()));
    refused(fitted, "already fitted");
}

/// A checkpoint taken after training, on the mode that sheds its raw
/// vectors, replayed through inserts into the quantized graph, a removal, a
/// compaction, a rebuild to another degree and a clear that keeps the
/// codebook.
#[test]
fn replay_reproduces_a_quantized_collection_from_a_checkpoint_after_training() {
    let vectors = clustered(1400, 16, 0x0014_7b30);
    let declaration = Declaration::validate(16, "l2", 8, 60, 2000, vec![]).unwrap();
    let quantization = declaration
        .quantization(4, 4, 1000, None, StorageMode::QuantizedOnly)
        .unwrap();
    let mut run = Run::new(Collection::build(declaration, Some(quantization)));
    let c = &run.original;
    for batch in batches(0..1050, 350) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    assert!(c.is_quantized());
    run.checkpoint();
    let c = &run.original;
    for batch in batches(1050..1200, 50) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    assert_eq!(
        c.remove_points(&["r1100".to_string(), "r1101".to_string()])
            .unwrap()
            .len(),
        0
    );
    assert!(c.compact().unwrap() > 0);
    let plan = c.plan_rebuild(Some(6), None, Some(80)).unwrap();
    c.rebuild(plan).unwrap();
    c.clear().unwrap();
    assert!(c.is_quantized(), "a clear keeps the codebook");
    for batch in batches(1200..1300, 25) {
        add(c, batch.iter().map(|&i| record(i, &vectors)).collect());
    }
    let queries = Queries {
        dense: vectors[1398].clone(),
        text: None,
        sparse: None,
    };
    let kinds = run.prove(&queries, None);
    use OperationKind::*;
    for kind in [Insert, Remove, Compact, Rebuild, Clear] {
        assert!(kinds.contains(&kind), "{kind:?} was replayed");
    }
}

/// A text collection, where term ids replay in issue order from the
/// interning records rather than from the order records are inserted, and
/// the postings and the dictionary come back byte for byte.
#[test]
fn replay_reproduces_a_text_collection_with_its_dictionary() {
    let vectors = clustered(400, 8, 0x0014_7b40);
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(SimpleTokenizer);
    let declaration = Declaration::validate(8, "l2", 8, 60, 500, vec!["cat".to_string()])
        .unwrap()
        .with_text(
            "text",
            SparseConfig {
                weighting: Weighting::BM25,
                ..SparseConfig::default()
            },
            tokenizer.clone(),
        )
        .unwrap();
    let mut run = Run::new(Collection::build(declaration, None));
    let c = &run.original;
    for batch in batches(0..60, 20) {
        add(
            c,
            batch.iter().map(|&i| text_record(c, i, &vectors)).collect(),
        );
    }
    run.checkpoint();
    let c = &run.original;
    let terms_at_checkpoint = c.term_count().unwrap();
    for batch in batches(60..150, 30) {
        add(
            c,
            batch.iter().map(|&i| text_record(c, i, &vectors)).collect(),
        );
    }
    assert!(
        c.term_count().unwrap() > terms_at_checkpoint,
        "new terms arrived after the checkpoint"
    );
    assert_eq!(
        c.remove_points(&["r70".to_string(), "r71".to_string()])
            .unwrap()
            .len(),
        0
    );
    assert!(c.compact().unwrap() > 0);
    overwrite(
        c,
        vec![ParsedRecord {
            sparse: Some(SparseHalf::Terms(
                c.tokenize("overwritten omega psi w99").unwrap(),
            )),
            ..record(80, &vectors)
        }],
    );
    // A batch with a duplicate inside it: the second is refused at the door
    // and its terms, interned before the refusal, stay interned.
    let added = c.add_records(
        vec![
            text_record(c, 150, &vectors),
            ParsedRecord {
                sparse: Some(SparseHalf::Terms(c.tokenize("duplicate chi w98").unwrap())),
                ..record(150, &vectors)
            },
            text_record(c, 151, &vectors),
        ],
        vec![],
        false,
    );
    assert_eq!(added.total_errors, 1);
    assert_eq!(added.inserted.len(), 2);
    for batch in batches(152..200, 24) {
        add(
            c,
            batch.iter().map(|&i| text_record(c, i, &vectors)).collect(),
        );
    }
    c.clear().unwrap();
    for batch in batches(200..230, 15) {
        add(
            c,
            batch.iter().map(|&i| text_record(c, i, &vectors)).collect(),
        );
    }

    let queries = Queries {
        dense: vectors[398].clone(),
        text: Some("alpha kappa w70"),
        sparse: None,
    };
    let kinds = run.prove(&queries, Some(tokenizer));
    use OperationKind::*;
    for kind in [Intern, Insert, Remove, Compact, Clear] {
        assert!(kinds.contains(&kind), "{kind:?} was replayed");
    }
}

/// A sparse space that takes term ids, so the insert record's sparse half
/// is what the caller gave and the weighting's rules are held on replay.
#[test]
fn replay_reproduces_a_sparse_collection() {
    let vectors = clustered(200, 8, 0x0014_7b50);
    let declaration = Declaration::validate(8, "l2", 8, 60, 300, vec!["cat".to_string()])
        .unwrap()
        .with_sparse(
            "terms",
            SparseConfig {
                weighting: Weighting::Bm25 { k1: 1.5, b: 0.6 },
                ..SparseConfig::default()
            },
        )
        .unwrap();
    let mut run = Run::new(Collection::build(declaration, None));
    let c = &run.original;
    for batch in batches(0..60, 20) {
        add(
            c,
            batch.iter().map(|&i| sparse_record(i, &vectors)).collect(),
        );
    }
    run.checkpoint();
    let c = &run.original;
    for batch in batches(60..150, 30) {
        add(
            c,
            batch.iter().map(|&i| sparse_record(i, &vectors)).collect(),
        );
    }
    assert!(c.remove_where(&filter("b")).unwrap() > 0);
    assert!(c.compact().unwrap() > 0);
    overwrite(c, vec![sparse_record(100, &vectors)]);
    for batch in batches(150..180, 15) {
        add(
            c,
            batch.iter().map(|&i| sparse_record(i, &vectors)).collect(),
        );
    }
    let queries = Queries {
        dense: vectors[199].clone(),
        text: None,
        sparse: Some(SparseVector {
            dims: vec![1, 14, 27],
            values: vec![1.0, 2.0, 1.0],
        }),
    };
    let kinds = run.prove(&queries, None);
    use OperationKind::*;
    for kind in [Insert, Remove, Compact] {
        assert!(kinds.contains(&kind), "{kind:?} was replayed");
    }
}
