//! The engine's own error type.
//!
//! Every failure the engine raises is a variant here, carrying its detail as
//! fields, and the message a caller reads is written once, in `Display`. The
//! Python exception each variant becomes is named by `exception`, as a value
//! rather than a type, so this module and everything that raises through it
//! compiles without PyO3. The one place an `Error` becomes a `PyErr` is the
//! `From` impl at the binding boundary in `lib.rs`.
//!
//! `Display` is written by hand rather than derived, because several messages
//! select a phrase from a field, count and pluralise, or truncate a list, and
//! a hand written arm keeps that in the one place the message lives.
//!
//! The messages are the ones the engine has always raised, byte for byte. A
//! message that reads badly is kept as it is; changing one is a change to what
//! a caller sees and is made on its own.

use std::fmt;
use std::path::PathBuf;

/// The Python exception class a failure is raised as.
///
/// Named here so the mapping from a failure to its class travels with the
/// failure, without this module naming PyO3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exception {
    /// `ValueError`, an argument or a stored value outside what the engine accepts
    Value,
    /// `RuntimeError`, an operation that could not be carried out
    Runtime,
    /// `KeyError`, an id the index does not hold
    Key,
    /// `FileNotFoundError`, a file or directory the loader could not find
    FileNotFound,
}

impl Exception {
    /// The class's name, as Python spells it.
    ///
    /// A `PyErr` displays as `<class>: <message>`, and the per record messages
    /// `add` collects were built from that display, so they carry the class
    /// name and keep carrying it.
    pub fn name(self) -> &'static str {
        match self {
            Exception::Value => "ValueError",
            Exception::Runtime => "RuntimeError",
            Exception::Key => "KeyError",
            Exception::FileNotFound => "FileNotFoundError",
        }
    }
}

/// Every failure the engine raises.
#[derive(Debug)]
pub enum Error {
    // ------------------------------------------------------------------
    // The index declaration. `source` is empty at `create()` and names the
    // file the value came out of at load.
    // ------------------------------------------------------------------
    /// `dim` is zero
    DimZero { source: String, dim: usize },
    /// `dim` is above the ceiling
    DimTooLarge {
        source: String,
        dim: usize,
        max: usize,
    },
    /// `ef_construction` is zero
    EfConstructionZero { source: String, value: usize },
    /// `ef_construction` is above the ceiling
    EfConstructionTooLarge {
        source: String,
        value: usize,
        max: usize,
    },
    /// `expected_size` is zero
    ExpectedSizeZero { source: String, value: usize },
    /// `expected_size` is above the ceiling
    ExpectedSizeTooLarge {
        source: String,
        value: usize,
        max: usize,
    },
    /// `m` is below 2
    MBelowMinimum { source: String, m: usize },
    /// `m` is above 256
    MTooLarge { source: String, m: usize },
    /// The distance space is not one the engine serves
    UnsupportedSpace { source: String, space: String },
    /// The space and quantization cannot be served together
    SpaceCannotBeQuantized {
        source: String,
        space: String,
        reason: &'static str,
        recovery: &'static str,
    },

    // ------------------------------------------------------------------
    // The quantization declaration at `create()`
    // ------------------------------------------------------------------
    /// `type` is not `pq`
    UnsupportedQuantizationType { qtype: String },
    /// `storage_mode` is not one of the two names, with the message the
    /// parser built
    InvalidStorageMode(String),
    /// `subvectors` is zero
    SubvectorsZero,
    /// `subvectors` exceeds `dim`
    SubvectorsExceedDim { subvectors: usize, dim: usize },
    /// `subvectors` does not divide `dim`
    SubvectorsDoNotDivideDim { subvectors: usize, dim: usize },
    /// `bits` is outside 1 to 8
    BitsOutOfRange { bits: usize },
    /// `training_size` is below 1000
    TrainingSizeTooSmall { training_size: usize },
    /// `max_training_vectors` is below `training_size`
    MaxTrainingBelowTrainingSize {
        max_training: usize,
        training_size: usize,
    },

    // ------------------------------------------------------------------
    // The indexed field declaration
    // ------------------------------------------------------------------
    /// More fields declared than the limit
    IndexedFieldsTooMany {
        source: String,
        count: usize,
        max: usize,
    },
    /// An empty name in the declaration
    IndexedFieldEmpty { source: String },
    /// A reserved filter key in the declaration
    IndexedFieldReserved { source: String, name: String },
    /// A name declared twice
    IndexedFieldRepeated { source: String, name: String },

    // ------------------------------------------------------------------
    // Filter compilation
    // ------------------------------------------------------------------
    /// An entry of a group is not a mapping
    FilterBranchNotMapping { key: String, found: &'static str },
    /// A presence operator was given something other than a boolean
    PresenceTargetNotBool {
        name: String,
        op: String,
        found: &'static str,
    },
    /// An operator name the dispatch does not know
    UnknownFilterOperation { op: String },
    /// A group would open a level past the cap
    FilterTooDeep {
        max: usize,
        key: String,
        level: usize,
    },
    /// A reserved key carrying the wrong shape
    ReservedKeyShape {
        key: String,
        shape: &'static str,
        found: &'static str,
    },

    // ------------------------------------------------------------------
    // Query and record vectors
    // ------------------------------------------------------------------
    /// A query vector with no components
    SearchVectorEmpty,
    /// A query vector of the wrong width
    SearchVectorDimension { expected: usize, got: usize },
    /// A query vector holding a NaN or an infinity
    SearchVectorNotFinite { index: usize, value: f32 },
    /// A record vector with no components
    VectorEmpty,
    /// A record vector of the wrong width
    VectorDimension { expected: usize, got: usize },
    /// A record vector holding a NaN or an infinity
    VectorNotFinite { index: usize, value: f32 },
    /// A vector in a query batch of the wrong width
    BatchVectorDimension {
        position: usize,
        expected: usize,
        got: usize,
    },
    /// A vector in a query batch holding a NaN or an infinity
    BatchVectorNotFinite {
        position: usize,
        index: usize,
        value: f32,
    },
    /// A NumPy array whose shape is not `(N, dim)`
    BatchArrayShape { dim: usize, shape: Vec<usize> },
    /// A query batch with no vectors
    BatchEmpty,
    /// A vector in a query batch with no components
    BatchVectorEmpty { position: usize },
    /// `top_k` above the ceiling
    TopKTooLarge { max: usize, top_k: usize },
    /// `ef_search` above the ceiling
    EfSearchTooLarge { max: usize, ef_search: usize },

    // ------------------------------------------------------------------
    // Records and mutation
    // ------------------------------------------------------------------
    /// An id already held, with overwrite off
    DuplicateId { id: String },
    /// Ids a strict `get_records` was asked for and the index does not hold,
    /// sorted
    RecordsAbsent { absent: Vec<String> },
    /// `list` given both a cursor and an offset
    ListAfterWithOffset { after: String, offset: usize },
    /// `list` given a cursor the index does not hold
    ListCursorMissing { after: String },
    /// `delete` given both selectors
    DeleteBothSelectors,
    /// `delete` given neither selector
    DeleteNoSelector,
    /// `remove_where` given a filter that matches every record
    RemoveWhereMatchesEverything,
    /// `rebuild` given nothing to change
    RebuildWithoutChanges,
    /// A quantized graph with no product quantizer beside it
    NoQuantizer,
    /// Live records with nothing to rebuild the graph from
    CompactRefused {
        missing: usize,
        live: usize,
        what: &'static str,
    },
    /// The quantized codes could not be re-inserted during a compaction
    ReinsertCodesFailed(String),
    /// The raw vectors could not be carried over during a compaction
    AdoptRawFailed(String),
    /// A vector could not be quantized
    QuantizeFailed(String),
    /// A failure reported by the graph or the quantizer, verbatim
    Engine(String),

    // ------------------------------------------------------------------
    // Spaces, and the index seam every space sits behind
    // ------------------------------------------------------------------
    /// A space declared with an empty name
    SpaceNameEmpty,
    /// A name no space in the collection carries
    SpaceUnknown { name: String },
    /// A name two spaces were declared under
    SpaceDeclaredTwice { name: String },
    /// More spaces declared than a collection holds
    SpacesTooMany { max: usize },
    /// A sparse vector offered to a collection that declares no sparse space
    NoSparseSpace,
    /// A vector of one family offered to a space of another
    SpaceKindMismatch {
        space: String,
        expected: &'static str,
        got: &'static str,
    },
    /// An insert under an internal id the index already holds
    RecordAlreadyHeld { id: u32 },
    /// A removal of an internal id the index does not hold
    RecordNotHeld { id: u32 },
    /// A sparse vector whose two slices differ in length
    SparseVectorShape { dims: usize, values: usize },
    /// A sparse vector whose dimensions are not strictly increasing
    SparseDimsNotIncreasing { position: usize },
    /// A sparse vector holding a NaN or an infinity
    SparseValueNotFinite { index: usize, value: f32 },
    /// A stored value at or below zero offered to a space whose scoring rule
    /// reads every value as a term frequency
    SparseValueNotPositive { index: usize, value: f32 },
    /// A stored value that is not a whole number offered to a space whose
    /// scoring rule reads every value as a term frequency
    SparseValueNotWhole { index: usize, value: f32 },
    /// A term weighting parameter outside its range
    SparseWeightingInvalid {
        parameter: &'static str,
        value: f32,
        rule: &'static str,
    },
    /// Text offered to a collection whose sparse space takes term ids alone
    NoTextLayer,
    /// A sparse vector of term ids offered to a collection whose sparse
    /// space has a text layer and takes text alone
    SparseVectorOnTextSpace,
    /// A term dictionary that has issued every id it can
    TermIdsExhausted,
    /// The caller's own tokenizer failed, carrying what it raised, so a
    /// binding can hand the caller their own failure back
    TokenizerFailed(Box<dyn std::error::Error + Send + Sync>),
    /// A directory whose text layer recorded a tokenizer of the caller's own,
    /// opened without one
    TokenizerRequired { space: String },
    /// A tokenizer handed to `load` whose declaration is not the one the
    /// directory recorded
    TokenizerMismatch {
        space: String,
        recorded: &'static str,
        handed: &'static str,
    },
    /// A tokenizer handed to `load` for a directory with no text layer
    TokenizerUnexpected,
    /// A record the sparse artefact holds and the mappings do not name
    SparseRecordUnmapped { space: String, id: u32 },
    /// A term id the postings carry beyond the dictionary's count
    TermIdBeyondDictionary {
        space: String,
        term: u32,
        terms: usize,
    },
    /// A space named in `config.json` that the collection cannot hold
    SpaceRecordInvalid { file: String, detail: String },

    // ------------------------------------------------------------------
    // A query over several arms
    // ------------------------------------------------------------------
    /// A query naming no arm
    QueryArmsEmpty,
    /// A query naming more arms than the limit
    QueryArmsTooMany { max: usize, arms: usize },
    /// `fetch` above the ceiling
    FetchTooLarge { max: usize, fetch: usize },
    /// A fusion constant outside its range
    FusionConstantInvalid { value: f32 },

    // ------------------------------------------------------------------
    // Logging
    // ------------------------------------------------------------------
    /// `log_dir` is empty
    LogDirEmpty,

    // ------------------------------------------------------------------
    // The graph rebuild at load
    // ------------------------------------------------------------------
    /// A record of the wrong width reached the rebuild
    RebuildRefusedDimension {
        id: String,
        expected: usize,
        got: usize,
    },
    /// A record holding a NaN or an infinity reached the rebuild
    RebuildRefusedNotFinite {
        id: String,
        value: f32,
        position: usize,
    },
    /// Records the rebuild refused, with their messages
    RebuildRefusedRecords { refused: Vec<String>, total: usize },
    /// The graph dump was written and could not be measured
    DumpLengthUnreadable(String),
    /// The graph dump failed
    GraphDumpFailed(String),

    // ------------------------------------------------------------------
    // The journal
    // ------------------------------------------------------------------
    /// A journal's file header cannot be read
    JournalHeaderInvalid { file: String, detail: String },
    /// A record fails and records follow it, so its bytes changed after it
    /// was written
    JournalCorrupt {
        file: String,
        sequence: u64,
        at: u64,
        detail: String,
    },
    /// A record's payload does not decode
    JournalRecordInvalid {
        file: String,
        sequence: u64,
        at: u64,
        detail: String,
    },
    /// The journal's file could not be created, written, synced or cut
    JournalIoFailed {
        path: PathBuf,
        what: &'static str,
        error: String,
    },
    /// A record's journal payload would be above the ceiling
    JournalRecordTooLarge { bytes: usize, ceiling: usize },
    /// A replayed record names a value the collection would not have
    /// issued, so the records and the collection do not belong together
    JournalReplayMismatch { detail: String },
    /// The `journal` record in `manifest.json` names a value this build
    /// cannot read
    JournalManifestInvalid { detail: String },
    /// A checkpoint of a journaled collection was asked for a directory
    /// other than the one its journal sits beside
    JournalDirectoryMismatch { journal: String, target: String },
    /// A directory's manifest names a journal that is not beside it
    JournalMissing {
        directory: String,
        file: String,
        recorded: String,
        sequence: u64,
    },
    /// The journal beside a directory belongs to another collection
    JournalNotThisCollection {
        file: String,
        journal_id: String,
        directory_id: String,
    },
    /// The journal's first record is above the one after the checkpoint's,
    /// so records the checkpoint does not hold were cut from it
    JournalStartsAfterCheckpoint {
        file: String,
        first: u64,
        checkpoint: u64,
    },
    /// A record above the checkpoint's sequence would not apply
    JournalReplayFailed {
        file: String,
        sequence: u64,
        detail: String,
    },

    // ------------------------------------------------------------------
    // The saved directory
    // ------------------------------------------------------------------
    /// A container in a bincode artefact declares more than the file holds
    DecodeLengthExceeded { file: String, bytes: usize },
    /// A bincode artefact did not decode
    DecodeFailed { file: String, error: String },
    /// The save target has no final path component
    TargetHasNoName { target: PathBuf },
    /// The staging directory could not be created
    StagingCreateFailed { staging: PathBuf, error: String },
    /// An interrupted save's directory could not be moved back
    RecoverRenameFailed {
        target: PathBuf,
        replaced: PathBuf,
        error: String,
    },
    /// The existing index could not be moved aside
    MoveAsideFailed { target: PathBuf, error: String },
    /// The staged index could not be moved into place, after the existing one
    /// was moved aside
    MoveIntoPlaceFailedAfterAside {
        target: PathBuf,
        error: String,
        restored: bool,
    },
    /// The staged index could not be moved into an empty place
    MoveIntoPlaceFailed { target: PathBuf, error: String },
    /// A directory tree could not be removed
    RemoveTreeFailed {
        path: PathBuf,
        what: &'static str,
        error: String,
    },
    /// An artefact could not be created
    ArtefactCreateFailed { name: String, error: String },
    /// An artefact could not be written
    ArtefactWriteFailed { name: String, error: String },
    /// An artefact's length is not the one the manifest records
    ArtefactLengthMismatch {
        name: String,
        actual: usize,
        recorded: u64,
        contents: &'static str,
    },
    /// An artefact's digest is not the one the manifest records
    ArtefactDigestMismatch {
        name: String,
        actual: String,
        expected: String,
        contents: &'static str,
    },
    /// An artefact could not be read
    ArtefactReadFailed { name: String, error: String },
    /// A JSON artefact is not UTF-8
    ArtefactNotUtf8 { name: String, error: String },
    /// A JSON artefact did not parse
    ArtefactParseFailed { name: &'static str, error: String },
    /// `format_version` is not a dotted version
    FormatVersionUnparsable {
        format_version: String,
        current: &'static str,
    },
    /// `format_version` names a major this build does not read
    FormatVersionUnsupported {
        format_version: String,
        supported: &'static str,
        newer: bool,
    },
    /// A 1.x manifest over a `config.json` that declares a sparse space,
    /// which no release writing 1.x could have produced
    FormatVersionSpaces { format_version: String },
    /// A manifest below 3.x that names a journal, which no release writing
    /// that format could have produced
    FormatVersionJournal { format_version: String },
    /// Files the manifest names and the directory does not hold, in manifest
    /// order, with what the first of them holds
    ArtefactsMissing {
        missing: Vec<String>,
        contents: &'static str,
    },
    /// `id_counter` above what a node index can name
    IdCounterTooLarge { file: String, id_counter: usize },
    /// Stored vectors holding a NaN or an infinity, sorted by id
    VectorsNotFinite {
        offenders: Vec<String>,
        total: usize,
    },
    /// `bits` in quantization.json outside 1 to 8
    BitsOutOfRangeInFile { file: String, bits: usize },
    /// `subvectors` in quantization.json is zero
    SubvectorsZeroInFile { file: String },
    /// `subvectors` in quantization.json does not fit `dim`
    SubvectorsInvalidInFile {
        file: String,
        subvectors: usize,
        dim: usize,
    },
    /// The raw store of a `quantized_with_raw` index could not be restored
    RestoreRawFailed(String),
    /// The restored record count is not the one config.json records
    RestoredCountMismatch {
        restored: usize,
        expected: usize,
        raw_count: usize,
        code_count: usize,
    },
    /// The codebook's shape is not the one quantization.json describes
    CodebookShapeMismatch {
        actual: (usize, usize, usize),
        expected: (usize, usize, usize),
        subvectors: usize,
        bits: usize,
    },
    /// The codebook is all zeros
    CodebookAllZero,
    /// A trained index with no pq_centroids.bin
    CentroidsMissing,
    /// Records held as codes with no codebook to reconstruct them
    CodesWithoutCodebook { count: usize },
    /// A record's codes did not reconstruct
    ReconstructFailed {
        id: String,
        codes: usize,
        error: String,
    },
    /// The directory to load is not there
    IndexDirectoryNotFound { path: String },
    /// An artefact did not serialize
    SerializeFailed { what: &'static str, error: String },
}

impl Error {
    /// The Python exception class this failure is raised as.
    pub fn exception(&self) -> Exception {
        use Error::*;
        match self {
            DimZero { .. }
            | DimTooLarge { .. }
            | EfConstructionZero { .. }
            | EfConstructionTooLarge { .. }
            | ExpectedSizeZero { .. }
            | ExpectedSizeTooLarge { .. }
            | MBelowMinimum { .. }
            | MTooLarge { .. }
            | SpaceCannotBeQuantized { .. }
            | UnsupportedQuantizationType { .. }
            | InvalidStorageMode(_)
            | SubvectorsZero
            | SubvectorsExceedDim { .. }
            | SubvectorsDoNotDivideDim { .. }
            | BitsOutOfRange { .. }
            | TrainingSizeTooSmall { .. }
            | MaxTrainingBelowTrainingSize { .. }
            | IndexedFieldsTooMany { .. }
            | IndexedFieldEmpty { .. }
            | IndexedFieldReserved { .. }
            | IndexedFieldRepeated { .. }
            | FilterBranchNotMapping { .. }
            | PresenceTargetNotBool { .. }
            | UnknownFilterOperation { .. }
            | FilterTooDeep { .. }
            | ReservedKeyShape { .. }
            | SearchVectorEmpty
            | SearchVectorDimension { .. }
            | SearchVectorNotFinite { .. }
            | VectorEmpty
            | VectorDimension { .. }
            | VectorNotFinite { .. }
            | BatchVectorDimension { .. }
            | BatchVectorNotFinite { .. }
            | BatchArrayShape { .. }
            | BatchEmpty
            | BatchVectorEmpty { .. }
            | TopKTooLarge { .. }
            | EfSearchTooLarge { .. }
            | DuplicateId { .. }
            | ListAfterWithOffset { .. }
            | DeleteBothSelectors
            | DeleteNoSelector
            | RemoveWhereMatchesEverything
            | RebuildWithoutChanges
            | LogDirEmpty
            | DecodeLengthExceeded { .. }
            | TargetHasNoName { .. }
            | IdCounterTooLarge { .. }
            | BitsOutOfRangeInFile { .. }
            | SubvectorsZeroInFile { .. }
            | SubvectorsInvalidInFile { .. }
            | SpaceNameEmpty
            | SpaceDeclaredTwice { .. }
            | SpacesTooMany { .. }
            | NoSparseSpace
            | SpaceKindMismatch { .. }
            | SparseVectorShape { .. }
            | SparseDimsNotIncreasing { .. }
            | SparseValueNotFinite { .. }
            | JournalRecordTooLarge { .. }
            | SparseValueNotPositive { .. }
            | SparseValueNotWhole { .. }
            | SparseWeightingInvalid { .. }
            | NoTextLayer
            | SparseVectorOnTextSpace
            | QueryArmsEmpty
            | QueryArmsTooMany { .. }
            | FetchTooLarge { .. }
            | FusionConstantInvalid { .. } => Exception::Value,

            RecordsAbsent { .. }
            | ListCursorMissing { .. }
            | SpaceUnknown { .. }
            | RecordNotHeld { .. } => Exception::Key,

            TermIdsExhausted
            | TokenizerFailed(_)
            | TokenizerRequired { .. }
            | TokenizerMismatch { .. }
            | TokenizerUnexpected
            | SparseRecordUnmapped { .. }
            | TermIdBeyondDictionary { .. }
            | SpaceRecordInvalid { .. }
            | FormatVersionSpaces { .. }
            | FormatVersionJournal { .. } => Exception::Runtime,

            ArtefactReadFailed { .. }
            | ArtefactsMissing { .. }
            | CentroidsMissing
            | IndexDirectoryNotFound { .. }
            | JournalMissing { .. } => Exception::FileNotFound,

            UnsupportedSpace { .. }
            | NoQuantizer
            | CompactRefused { .. }
            | ReinsertCodesFailed(_)
            | AdoptRawFailed(_)
            | QuantizeFailed(_)
            | Engine(_)
            | RecordAlreadyHeld { .. }
            | RebuildRefusedDimension { .. }
            | RebuildRefusedNotFinite { .. }
            | RebuildRefusedRecords { .. }
            | DumpLengthUnreadable(_)
            | GraphDumpFailed(_)
            | DecodeFailed { .. }
            | StagingCreateFailed { .. }
            | RecoverRenameFailed { .. }
            | MoveAsideFailed { .. }
            | MoveIntoPlaceFailedAfterAside { .. }
            | MoveIntoPlaceFailed { .. }
            | RemoveTreeFailed { .. }
            | ArtefactCreateFailed { .. }
            | ArtefactWriteFailed { .. }
            | ArtefactLengthMismatch { .. }
            | ArtefactDigestMismatch { .. }
            | ArtefactNotUtf8 { .. }
            | ArtefactParseFailed { .. }
            | FormatVersionUnparsable { .. }
            | FormatVersionUnsupported { .. }
            | VectorsNotFinite { .. }
            | RestoreRawFailed(_)
            | RestoredCountMismatch { .. }
            | CodebookShapeMismatch { .. }
            | CodebookAllZero
            | CodesWithoutCodebook { .. }
            | ReconstructFailed { .. }
            | SerializeFailed { .. }
            | JournalHeaderInvalid { .. }
            | JournalCorrupt { .. }
            | JournalRecordInvalid { .. }
            | JournalIoFailed { .. }
            | JournalReplayMismatch { .. }
            | JournalManifestInvalid { .. }
            | JournalDirectoryMismatch { .. }
            | JournalNotThisCollection { .. }
            | JournalStartsAfterCheckpoint { .. }
            | JournalReplayFailed { .. } => Exception::Runtime,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Error::*;
        match self {
            DimZero { source, dim } => write!(f, "{}dim must be positive, got {}", source, dim),
            DimTooLarge { source, dim, max } => write!(
                f,
                "{}dim must be at most {}, got {}. dim is the width of one vector \
                 buffer, so sizing that buffer from the declared width is the \
                 first allocation this index makes. That allocation is not \
                 fallible: above this bound the process aborts rather than \
                 raising. One vector at the ceiling is {} bytes, an order of \
                 magnitude above the widest embedding any published model \
                 produces.",
                source,
                max,
                dim,
                max * 4
            ),
            EfConstructionZero { source, value } => write!(
                f,
                "{}ef_construction must be positive, got {}",
                source, value
            ),
            EfConstructionTooLarge { source, value, max } => write!(
                f,
                "{}ef_construction must be at most {}, got {}. It is the width of the \
                 candidate search every insertion runs, and the graph sizes two \
                 candidate heaps from it, 8 bytes a slot, before each insertion visits \
                 a node. That allocation is not fallible: a value of 2^40 asks for 8 TB \
                 on the first add() and the process aborts rather than raising. The \
                 ceiling is eight times the neighbour budget at the largest m, being \
                 2 * 256, so the default's margin over that budget is available at \
                 every m the index allows, and a build at the ceiling runs about \
                 thirteen times longer than one at the default.",
                source, max, value
            ),
            ExpectedSizeZero { source, value } => write!(
                f,
                "{}expected_size must be positive, got {}",
                source, value
            ),
            ExpectedSizeTooLarge { source, value, max } => write!(
                f,
                "{}expected_size must be at most {}, got {}. The graph reserves one \
 slot per \
                 declared record at creation, 8 bytes each, so this \
 declaration would ask for \
                 {:.1} GB before a single record is \
 added. That allocation is not fallible: \
                 above this bound the \
 process aborts rather than raising. expected_size is a \
                 capacity \
 hint and not a limit, and under-declaring only costs some \
 \
                 reallocation, so declare what you expect to hold.",
                source,
                max,
                value,
                (*value as f64 * 8.0) / 1_000_000_000.0
            ),
            MBelowMinimum { source, m } => write!(
                f,
                "{}m must be at least 2, got {}. Layer assignment samples from a \
 scale of 1 / \
                 ln(m), which is infinity at m 1, so every point \
 overflows the layer cap and \
                 is redispatched uniformly across all \
 16 layers instead of following the \
                 exponential distribution the \
 graph depends on. Measured on 3,000 records of \
                 32 dimensions, \
 recall at 10 was 0.0220 at m 1 against 0.6880 at m 2 and \
                 1.0000 \
 at m 16.",
                source, m
            ),
            MTooLarge { source, m } => write!(
                f,
                "{}m must be less than or equal to 256, got {}",
                source, m
            ),
            UnsupportedSpace { source, space } => write!(
                f,
                "{}Unsupported space: '{}'. Supported spaces: 'cosine', 'l2', 'l1', 'dot'",
                source, space
            ),
            SpaceCannotBeQuantized {
                source,
                space,
                reason,
                recovery,
            } => write!(
                f,
                "{}space='{}' cannot be quantized. A quantized graph scores every candidate from tables of squared L2 distances to the codebook, and {}, or drop quantization_config.{}",
                source, space, reason, recovery
            ),

            UnsupportedQuantizationType { qtype } => write!(
                f,
                "Unsupported quantization type: '{}'. Only 'pq' is currently supported.",
                qtype
            ),
            InvalidStorageMode(message) => f.write_str(message),
            SubvectorsZero => f.write_str("subvectors must be a positive integer, got 0"),
            SubvectorsExceedDim { subvectors, dim } => write!(
                f,
                "subvectors ({}) cannot exceed dimension ({})",
                subvectors, dim
            ),
            SubvectorsDoNotDivideDim { subvectors, dim } => write!(
                f,
                "subvectors ({}) must divide dimension ({}) evenly",
                subvectors, dim
            ),
            BitsOutOfRange { bits } => write!(f, "bits must be between 1 and 8, got {}", bits),
            TrainingSizeTooSmall { training_size } => write!(
                f,
                "training_size must be at least 1000, got {}",
                training_size
            ),
            MaxTrainingBelowTrainingSize {
                max_training,
                training_size,
            } => write!(
                f,
                "max_training_vectors ({}) must be >= training_size ({})",
                max_training, training_size
            ),

            IndexedFieldsTooMany { source, count, max } => write!(
                f,
                "{}indexed_fields names {} fields and the limit is {}. Declare the fields \
                 you filter on; every other field is still stored and still filterable, \
                 it just costs a walk of the metadata store.",
                source, count, max
            ),
            IndexedFieldEmpty { source } => write!(
                f,
                "{}indexed_fields contains an empty name. Every entry has to be the \
                 name of a metadata field.",
                source
            ),
            IndexedFieldReserved { source, name } => write!(
                f,
                "{}indexed_fields names \"{}\", which is a reserved filter key rather \
                 than a metadata field. A field with that name cannot be filtered on, \
                 so a column for it could never be read.",
                source, name
            ),
            IndexedFieldRepeated { source, name } => write!(
                f,
                "{}indexed_fields names \"{}\" twice. Each field is declared once.",
                source, name
            ),

            FilterBranchNotMapping { key, found } => write!(
                f,
                "Every entry of \"{}\" must be a filter mapping, for example \
                 {{\"lang\": \"en\"}}, but one of them is {}.",
                key, found
            ),
            PresenceTargetNotBool { name, op, found } => write!(
                f,
                "\"{}\" takes true or false, and {{\"{}\": {{\"{}\": ...}}}} was given {}.",
                op, name, op, found
            ),
            UnknownFilterOperation { op } => write!(f, "Unknown filter operation: {}", op),
            FilterTooDeep { max, key, level } => write!(
                f,
                "Filter groups nest to {} levels and \"{}\" would open level {}.",
                max, key, level
            ),
            ReservedKeyShape { key, shape, found } => write!(
                f,
                "\"{}\" is a reserved filter key and takes {}, but it was given {}. A \
                 metadata field named \"{}\" cannot be filtered on.",
                key, shape, found, key
            ),

            SearchVectorEmpty => f.write_str("Search vector cannot be empty"),
            SearchVectorDimension { expected, got } => write!(
                f,
                "Search vector dimension mismatch: expected {}, got {}",
                expected, got
            ),
            SearchVectorNotFinite { index, value } => write!(
                f,
                "Search vector contains invalid value at index {}: {}",
                index, value
            ),
            VectorEmpty => f.write_str("Vector cannot be empty"),
            VectorDimension { expected, got } => write!(
                f,
                "Vector dimension mismatch: expected {}, got {}",
                expected, got
            ),
            VectorNotFinite { index, value } => write!(
                f,
                "Vector contains invalid value at index {}: {} (must be finite)",
                index, value
            ),
            BatchVectorDimension {
                position,
                expected,
                got,
            } => write!(
                f,
                "Vector {}: dimension mismatch: expected {}, got {}",
                position, expected, got
            ),
            BatchVectorNotFinite {
                position,
                index,
                value,
            } => write!(
                f,
                "Vector {} in batch contains invalid value at index {}: {} (must be finite)",
                position, index, value
            ),
            BatchArrayShape { dim, shape } => write!(
                f,
                "NumPy array must have shape (N, {}), got {:?}",
                dim, shape
            ),
            BatchEmpty => f.write_str("Batch cannot be empty"),
            BatchVectorEmpty { position } => {
                write!(f, "Vector {} in batch cannot be empty", position)
            }
            TopKTooLarge { max, top_k } => write!(
                f,
                "top_k must be at most {}, got {}. top_k sizes the candidate search \
                 through the default ef_search of twice top_k, and the graph sizes two \
                 candidate heaps from that width, 8 bytes a slot, before it visits a \
                 node. That allocation is not fallible: a top_k of 2^40 asks for 16 TB \
                 and the process aborts rather than raising. The ceiling is four times \
                 the largest page any comparable engine serves.",
                max, top_k
            ),
            EfSearchTooLarge { max, ef_search } => write!(
                f,
                "ef_search must be at most {}, got {}. ef_search is the width of the \
                 candidate search, and the graph sizes two candidate heaps from it, 8 \
                 bytes a slot, before it visits a node. That allocation is not \
                 fallible: an ef_search of 2^40 asks for 8 TB and the process aborts \
                 rather than raising. The ceiling is twice the top_k ceiling, which is \
                 the default ef_search at the largest page.",
                max, ef_search
            ),

            DuplicateId { id } => write!(f, "Vector with ID '{}' already exists", id),
            RecordsAbsent { absent } => {
                let named: Vec<&str> = absent.iter().take(10).map(String::as_str).collect();
                write!(
                    f,
                    "get_records(strict=True) was asked for {} id{} the index does not hold: \
                     {}{}. Call it without strict=True to receive the records that are present, \
                     or test an id with contains(id) first.",
                    absent.len(),
                    if absent.len() == 1 { "" } else { "s" },
                    named.join(", "),
                    if absent.len() > named.len() {
                        format!(", and {} more", absent.len() - named.len())
                    } else {
                        String::new()
                    }
                )
            }
            ListAfterWithOffset { after, offset } => write!(
                f,
                "list() takes after or offset, not both, and it was given after='{}' \
                 with offset={}. after names the last record of the previous page \
                 and offset counts from the start, so the two name different places.",
                after, offset
            ),
            ListCursorMissing { after } => write!(
                f,
                "list(after='{}') names a record the index does not hold, so there \
                 is no position to resume from. The cursor record was removed while \
                 the caller was paging. Resume from an id the index still holds, or \
                 start again from offset 0.",
                after
            ),
            DeleteBothSelectors => f.write_str(
                "delete takes 'ids' or 'where', not both. Two selections do not compose \
                 into one without a rule, and either rule deletes the wrong records. Call \
                 it twice, or name the records you mean with delete(ids=...).",
            ),
            DeleteNoSelector => f.write_str(
                "delete requires 'ids' or 'where'. It does not default to deleting every \
                 record. Use delete(ids=[...]) to name records, delete(where={...}) to \
                 select them by metadata, or clear() to empty the index.",
            ),
            RemoveWhereMatchesEverything => f.write_str(
                "remove_where requires a filter that selects records. An empty filter \
                 matches every record, so this would delete the whole index. Name the \
                 records with remove_points(ids) if that is what you want.",
            ),
            RebuildWithoutChanges => f.write_str(
                "rebuild() changes m, expected_size, ef_construction or any combination \
                 of them, and was given none. Rebuilding the graph as it stands is \
                 compact(), which reclaims the nodes that removals and overwrites leave \
                 behind.",
            ),
            NoQuantizer => {
                f.write_str("Index reports a quantized graph but holds no product quantizer")
            }
            CompactRefused {
                missing,
                live,
                what,
            } => write!(
                f,
                "Refusing to compact: {} of {} live records have no stored {} to rebuild \
                 the graph from, so compacting would drop them. The index is unchanged.",
                missing, live, what
            ),
            ReinsertCodesFailed(error) => write!(
                f,
                "Failed to re-insert quantized codes during compact: {}",
                error
            ),
            AdoptRawFailed(error) => write!(
                f,
                "Failed to carry the raw vectors over during compact: {}",
                error
            ),
            QuantizeFailed(error) => write!(f, "Failed to quantize vector: {}", error),
            Engine(message) => f.write_str(message),

            SpaceNameEmpty => f.write_str("A space name must not be empty"),
            SpaceUnknown { name } => write!(f, "No space is named '{}'", name),
            SpaceDeclaredTwice { name } => {
                write!(f, "Space '{}' is declared twice", name)
            }
            SpacesTooMany { max } => write!(f, "A collection holds at most {} spaces", max),
            NoSparseSpace => f.write_str("This collection declares no sparse space"),
            SpaceKindMismatch {
                space,
                expected,
                got,
            } => write!(
                f,
                "Space '{}' holds {} vectors and was given a {} vector",
                space, expected, got
            ),
            RecordAlreadyHeld { id } => write!(
                f,
                "Record {} is already held by this space. A record is removed before it is added again, and an internal id is never reused.",
                id
            ),
            RecordNotHeld { id } => write!(f, "Record {} is not held by this space", id),
            SparseVectorShape { dims, values } => write!(
                f,
                "Sparse vector has {} dims and {} values, and the two must be the same length",
                dims, values
            ),
            SparseDimsNotIncreasing { position } => write!(
                f,
                "Sparse vector dims must be strictly increasing, and the dim at position {} is not above the one before it",
                position
            ),
            SparseValueNotFinite { index, value } => write!(
                f,
                "Sparse vector contains invalid value at index {}: {} (must be finite)",
                index, value
            ),
            SparseValueNotPositive { index, value } => write!(
                f,
                "Sparse vector value at index {} is {}, and a space weighted by term \
                 frequency takes values above zero alone",
                index, value
            ),
            SparseValueNotWhole { index, value } => write!(
                f,
                "Sparse vector value at index {} is {}, and a space weighted by term \
                 frequency takes whole numbers alone",
                index, value
            ),
            SparseWeightingInvalid {
                parameter,
                value,
                rule,
            } => write!(
                f,
                "Term weighting parameter {} is {}, and it must be {}",
                parameter, value, rule
            ),
            NoTextLayer => f.write_str("This collection's sparse space takes no text"),
            SparseVectorOnTextSpace => {
                f.write_str("This collection's sparse space takes text alone")
            }
            TermIdsExhausted => f.write_str("The term dictionary has issued every id it can"),
            TokenizerFailed(inner) => write!(f, "The tokenizer raised {}", inner),
            QueryArmsEmpty => f.write_str("A query needs at least one arm"),
            QueryArmsTooMany { max, arms } => {
                write!(f, "A query names at most {} arms, got {}", max, arms)
            }
            FetchTooLarge { max, fetch } => write!(
                f,
                "fetch must be at most {}, got {}. fetch is the page each arm contributes \
                 to the fusion, and it sizes the same candidate heaps top_k does.",
                max, fetch
            ),
            FusionConstantInvalid { value } => write!(
                f,
                "Reciprocal rank constant is {}, and it must be finite and at least zero",
                value
            ),

            LogDirEmpty => f.write_str("log_dir cannot be empty"),

            RebuildRefusedDimension { id, expected, got } => write!(
                f,
                "Graph rebuild refused record '{}': vector dimension mismatch, \
                 expected {}, got {}. Refusing to load a partial graph.",
                id, expected, got
            ),
            RebuildRefusedNotFinite {
                id,
                value,
                position,
            } => write!(
                f,
                "Graph rebuild refused record '{}': vector contains {} at index {}, \
                 which is not finite. Refusing to load a partial graph.",
                id, value, position
            ),
            RebuildRefusedRecords { refused, total } => write!(
                f,
                "Graph rebuild refused {} of {} records, so the loaded index would \
                 hold records that no query can reach. Refusing to load a partial \
                 graph. Rejected records: {}",
                refused.len(),
                total,
                refused.join("; ")
            ),
            DumpLengthUnreadable(error) => write!(
                f,
                "The graph dump was written and its length could not be read: {}",
                error
            ),
            GraphDumpFailed(error) => write!(f, "HNSW graph dump failed: {}", error),

            JournalHeaderInvalid { file, detail } => write!(
                f,
                "The journal {} cannot be read: {}. Nothing in it was replayed.",
                file, detail
            ),
            JournalCorrupt {
                file,
                sequence,
                at,
                detail,
            } => write!(
                f,
                "Record {} at byte {} of the journal {} is corrupt ({}) and records \
                 follow it, so it was written whole and its bytes changed afterwards. \
                 Refusing to open, because skipping it would recover a state nothing \
                 acknowledged. Restore the journal from a copy, or repair it by name \
                 to cut it at byte {} and lose record {} and everything after it.",
                sequence, at, file, detail, at, sequence
            ),
            JournalRecordInvalid {
                file,
                sequence,
                at,
                detail,
            } => write!(
                f,
                "Record {} at byte {} of the journal {} does not decode: {}",
                sequence, at, file, detail
            ),
            JournalIoFailed { path, what, error } => write!(
                f,
                "Failed to {} the journal {}: {}",
                what,
                path.display(),
                error
            ),
            JournalRecordTooLarge { bytes, ceiling } => write!(
                f,
                "The record encodes to {} bytes and the journal's record ceiling is {} bytes. \
                 Nothing was written for it and no internal id was issued. Reduce the \
                 record's metadata.",
                bytes, ceiling
            ),
            JournalReplayMismatch { detail } => write!(
                f,
                "The journal and the checkpoint do not belong together: {}. Nothing from this \
                 record on was applied.",
                detail
            ),
            JournalDirectoryMismatch { journal, target } => write!(
                f,
                "This collection records its mutations to the journal {}, and a save of \
                 it is the checkpoint that journal replays onto, so it saves to the \
                 directory the journal sits beside and to no other. Saving to {} would \
                 write a manifest naming a journal that is not beside it, which nothing \
                 could open.",
                journal, target
            ),
            JournalManifestInvalid { detail } => write!(
                f,
                "manifest.json names a journal this build cannot read: {}. A directory \
                 saved with a journal records its file name, the collection id both it \
                 and the directory carry, and the sequence the checkpoint holds.",
                detail
            ),
            JournalMissing {
                directory,
                file,
                recorded,
                sequence,
            } => write!(
                f,
                "The directory {} was saved with a journal, which its manifest names as {}, \
                 and no journal is beside it at {}. Every mutation after sequence {} is in \
                 that file and in no other, so opening the directory without it would lose \
                 them without a word. Put the journal back beside the directory, or open \
                 the checkpoint alone and accept that loss by name.",
                directory, recorded, file, sequence
            ),
            JournalNotThisCollection {
                file,
                journal_id,
                directory_id,
            } => write!(
                f,
                "The journal {} belongs to collection {} and the directory beside it to \
                 collection {}. A journal records one collection's mutations and cannot \
                 be applied to another's.",
                file, journal_id, directory_id
            ),
            JournalStartsAfterCheckpoint {
                file,
                first,
                checkpoint,
            } => write!(
                f,
                "The journal {} starts at sequence {} and the checkpoint beside it holds \
                 sequence {}. The records between the two are in neither, so the directory \
                 and the journal are from different histories.",
                file, first, checkpoint
            ),
            JournalReplayFailed {
                file,
                sequence,
                detail,
            } => write!(
                f,
                "Record {} of the journal {} could not be applied: {} Nothing from that \
                 record on was applied and the directory was not opened.",
                sequence, file, detail
            ),
            DecodeLengthExceeded { file, bytes } => write!(
                f,
                "{} declares a length its own {} bytes could not hold. A container in \
                 this file names more entries than the file has bytes to carry, so \
                 reading it would ask the allocator for memory the file never earned, \
                 and that allocation is not fallible: the process aborts rather than \
                 raising. Refusing to read it. Restore the directory from a copy.",
                file, bytes
            ),
            DecodeFailed { file, error } => {
                write!(f, "Failed to deserialize {}: {}", file, error)
            }
            TargetHasNoName { target } => write!(
                f,
                "'{}' does not name a directory this build can save into. A save writes a \
                 sibling directory beside the target and moves it into place, so the target \
                 has to have a name of its own.",
                target.display()
            ),
            StagingCreateFailed { staging, error } => write!(
                f,
                "Failed to create the staging directory {}: {}",
                staging.display(),
                error
            ),
            RecoverRenameFailed {
                target,
                replaced,
                error,
            } => write!(
                f,
                "An earlier save was interrupted while replacing {}, and the index \
                 it was replacing is at {}. Moving it back failed: {}. Rename that \
                 directory to {} by hand before saving again.",
                target.display(),
                replaced.display(),
                error,
                target.display()
            ),
            MoveAsideFailed { target, error } => write!(
                f,
                "Failed to move the existing index at {} aside: {}. Nothing was changed \
                 and the directory is as it was.",
                target.display(),
                error
            ),
            MoveIntoPlaceFailedAfterAside {
                target,
                error,
                restored,
            } => write!(
                f,
                "Failed to move the newly saved index into {}: {}. {}",
                target.display(),
                error,
                if *restored {
                    "The index that was there is back in place and nothing was lost."
                } else {
                    "The index that was there could not be put back and is in the \
                     directory named .zdbold beside it. Rename it back by hand."
                }
            ),
            MoveIntoPlaceFailed { target, error } => write!(
                f,
                "Failed to move the newly saved index into {}: {}",
                target.display(),
                error
            ),
            RemoveTreeFailed { path, what, error } => write!(
                f,
                "Failed to remove {}, which is {}: {}",
                path.display(),
                what,
                error
            ),
            ArtefactCreateFailed { name, error } => {
                write!(f, "Failed to create {}: {}", name, error)
            }
            ArtefactWriteFailed { name, error } => {
                write!(f, "Failed to write {}: {}", name, error)
            }
            ArtefactLengthMismatch {
                name,
                actual,
                recorded,
                contents,
            } => write!(
                f,
                "{} is {} bytes and manifest.json records it as {}. {} holds {}. The file is \
                 not the one this save wrote, so the index it describes is not the index that \
                 was saved. Refusing to load it. Restore the directory from a copy.",
                name, actual, recorded, name, contents
            ),
            ArtefactDigestMismatch {
                name,
                actual,
                expected,
                contents,
            } => write!(
                f,
                "{} is the length manifest.json records and its contents do not match the \
                 digest recorded beside it, which is {} against {}. {} holds {}. The file has \
                 changed since it was written, so the index it describes is not the index that \
                 was saved. Refusing to load it. Restore the directory from a copy.",
                name, actual, expected, name, contents
            ),
            ArtefactReadFailed { name, error } => {
                write!(f, "Failed to read {}: {}", name, error)
            }
            ArtefactNotUtf8 { name, error } => {
                write!(f, "{} is not valid UTF-8: {}", name, error)
            }
            ArtefactParseFailed { name, error } => {
                write!(f, "Failed to parse {}: {}", name, error)
            }
            FormatVersionUnparsable {
                format_version,
                current,
            } => write!(
                f,
                "manifest.json declares format_version '{}', which is not a version this \
                 build can interpret. A ZeusDB index directory declares a dotted version \
                 such as {}.",
                format_version, current
            ),
            FormatVersionUnsupported {
                format_version,
                supported,
                newer,
            } => write!(
                f,
                "Index format version {} cannot be opened by this build, which reads format \
                 versions {} only. The directory was written by a {} release of \
                 zeusdb-vector-database, so {}.",
                format_version,
                supported,
                if *newer { "newer" } else { "much older" },
                if *newer {
                    "upgrade the package to open it"
                } else {
                    "open it with the release that wrote it"
                }
            ),
            FormatVersionSpaces { format_version } => write!(
                f,
                "manifest.json declares format_version {} and config.json declares a sparse \
                 space, which no release writing that format holds. A directory holding a \
                 sparse space declares format_version 2.0.0 or later.",
                format_version
            ),
            FormatVersionJournal { format_version } => write!(
                f,
                "manifest.json declares format_version {} and names a journal, which no \
                 release writing that format holds. A directory saved with a journal \
                 declares format_version 3.0.0 or later.",
                format_version
            ),
            TokenizerRequired { space } => write!(
                f,
                "The sparse space '{}' was declared with a tokenizer of the caller's own, which \
                 the directory records as external and cannot reproduce. Open it with the same \
                 implementation handed to load.",
                space
            ),
            TokenizerMismatch {
                space,
                recorded,
                handed,
            } => write!(
                f,
                "The sparse space '{}' recorded its tokenizer as {} and the one handed to load \
                 declares itself {}. A saved space is opened with the tokenizer it was declared \
                 with.",
                space, recorded, handed
            ),
            TokenizerUnexpected => f.write_str(
                "A tokenizer was handed to load and no space in the directory takes text",
            ),
            SparseRecordUnmapped { space, id } => write!(
                f,
                "The sparse space '{}' holds a record under internal id {} that mappings.bin \
                 does not name",
                space, id
            ),
            TermIdBeyondDictionary { space, term, terms } => write!(
                f,
                "The sparse space '{}' carries term id {} and its dictionary holds {} terms",
                space, term, terms
            ),
            SpaceRecordInvalid { file, detail } => {
                write!(f, "{}: the spaces declared are invalid: {}", file, detail)
            }
            ArtefactsMissing { missing, contents } => {
                let first = missing.first().map(String::as_str).unwrap_or("");
                let others = if missing.len() > 1 {
                    format!(
                        " {} further file{} manifest.json names {} also absent: {}.",
                        missing.len() - 1,
                        if missing.len() == 2 { "" } else { "s" },
                        if missing.len() == 2 { "is" } else { "are" },
                        missing[1..].join(", ")
                    )
                } else {
                    String::new()
                };
                write!(
                    f,
                    "manifest.json names {} under files_included and the index directory does \
                     not hold it. {} holds {}. manifest.json is written after every file it \
                     names except the graph dump, so a directory in this state lost the file \
                     after the save that wrote it finished, or was copied without it. \
                     Refusing to load an index assembled from the files that survived, \
                     because it would not hold what was saved. Restore the directory from a \
                     copy.{}",
                    first, first, contents, others
                )
            }
            IdCounterTooLarge { file, id_counter } => write!(
                f,
                "{}: id_counter is {}, and an internal id is a graph node index, \
                 which is a u32. No index could have issued that many, so this file \
                 does not describe an index this build can rebuild.",
                file, id_counter
            ),
            VectorsNotFinite { offenders, total } => {
                let named: Vec<&str> = offenders.iter().take(5).map(|id| id.as_str()).collect();
                write!(
                    f,
                    "vectors.bin holds a NaN or an infinity in {} of {} records, so those records \
                     would score as NaN against every query and take an arbitrary place in every \
                     result page. Refusing to load. Affected records include: {}{}",
                    offenders.len(),
                    total,
                    named.join(", "),
                    if offenders.len() > named.len() {
                        ", ..."
                    } else {
                        ""
                    }
                )
            }
            BitsOutOfRangeInFile { file, bits } => write!(
                f,
                "{}: bits is {}, and bits must be an integer between 1 and 8. The centroid count \
                 is 2 to the bits, so the codebook this names would be {} times the size of the \
                 largest one this build can build, and sizing it is not a fallible allocation: \
                 the process aborts rather than raising.",
                file,
                bits,
                1u128 << (*bits).min(96)
            ),
            SubvectorsZeroInFile { file } => write!(
                f,
                "{}: subvectors is 0. Every subvector holds dim / subvectors values, so a count \
                 of zero divides by zero.",
                file
            ),
            SubvectorsInvalidInFile {
                file,
                subvectors,
                dim,
            } => write!(
                f,
                "{}: subvectors is {} and config.json declares dim {}. subvectors must divide \
                 the dimension evenly and cannot exceed it, which is the rule create() applies, \
                 so this file does not describe an index this build can rebuild.",
                file, subvectors, dim
            ),
            RestoreRawFailed(error) => write!(
                f,
                "Failed to restore the raw vectors of a quantized_with_raw index: {}",
                error
            ),
            RestoredCountMismatch {
                restored,
                expected,
                raw_count,
                code_count,
            } => write!(
                f,
                "Restored record count does not match config.json: the directory yields {} \
                 records while config.json reports {}. vectors.bin holds {} records and \
                 pq_codes.bin holds {}, so a data file is missing or truncated. Refusing to \
                 load an index that would report a count it cannot produce; restore the \
                 directory from a copy.",
                restored, expected, raw_count, code_count
            ),
            CodebookShapeMismatch {
                actual,
                expected,
                subvectors,
                bits,
            } => write!(
                f,
                "pq_centroids.bin does not match quantization.json: codebook is {}x{}x{}, \
                 expected {}x{}x{} for {} subvectors at {} bits. The codebook belongs to a \
                 different index, so this directory cannot be loaded.",
                actual.0, actual.1, actual.2, expected.0, expected.1, expected.2, subvectors, bits
            ),
            CodebookAllZero => f.write_str(
                "pq_centroids.bin holds an all-zero codebook, so every PQ code in this \
                 directory decodes to the zero vector. This is what a save performed by \
                 zeusdb-vector-database 0.3.0 through 0.4.1 writes over a directory it has \
                 just loaded, because those versions never read the codebook back. Restore \
                 the directory from a copy taken before that save; the records cannot be \
                 recovered from this one.",
            ),
            CentroidsMissing => f.write_str(
                "quantization.json reports a trained codebook but pq_centroids.bin is \
                 missing from the index directory. The codebook cannot be rebuilt from \
                 the other files, so restore it from a copy of the saved directory.",
            ),
            CodesWithoutCodebook { count } => write!(
                f,
                "{} records are stored as PQ codes with no raw vector, but the index has \
                 no codebook to reconstruct them with. pq_codes.bin and quantization.json \
                 disagree about whether this index was trained, so the directory cannot \
                 be loaded without dropping those records.",
                count
            ),
            ReconstructFailed { id, codes, error } => write!(
                f,
                "Failed to reconstruct record '{}' from its {} PQ codes: {}. The \
                 codebook in pq_centroids.bin does not fit the codes in pq_codes.bin.",
                id, codes, error
            ),
            IndexDirectoryNotFound { path } => {
                write!(f, "Index directory not found: {}", path)
            }
            SerializeFailed { what, error } => {
                write!(f, "Failed to serialize {}: {}", what, error)
            }
        }
    }
}

impl std::error::Error for Error {}
