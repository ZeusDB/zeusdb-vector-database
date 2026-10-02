//! # ZeusDB Vector Database - Persistence Module
//!
//! This module handles all save/load operations for ZeusDB vector indexes.
//! It implements a directory-based persistence format with hybrid JSON/Binary storage.
//!
//! ## File Format:
//! ```text
//! my_index.zdb/
//! ├── manifest.json           # Index metadata and file list
//! ├── config.json             # Index configuration
//! ├── mappings.bin            # ID mappings (binary)
//! ├── metadata.json           # Vector metadata (JSON)
//! ├── vectors.bin             # Raw vectors (storage mode dependent)
//! ├── quantization.json       # quantization configuration and training state (if enabled)
//! ├── pq_codes.bin            # Quantized codes (if PQ enabled)
//! ├── pq_centroids.bin        # PQ centroids (if trained)
//! ├── int8_scales.zdbint8     # the scales of a scalar quantized index, one a dimension (if trained)
//! ├── int8_rows.zdbint8       # every record's scalar row (if trained and not empty)
//! ├── hnsw_index.zdbgraph     # HNSW graph topology and the point each node holds
//! └── spaces/                 # one directory per sparse space, where one is declared
//!     └── <name>/
//!         ├── postings.zdbsparse  # every live record's term ids and weights
//!         └── terms.zdbdict       # the term dictionary, where the space takes text
//!
//! my_index.zdb.zdbwal         # the journal, where the collection held one
//! ```
//!
//! ## The format version
//!
//! Every directory this build saves declares `4.0.0`. It is a major because
//! the four binary artefacts, `mappings.bin`, `vectors.bin`, `pq_codes.bin`
//! and `pq_centroids.bin`, are written inside the frame
//! `zeusdb_vector_core::frame` describes, where every earlier release wrote
//! them in bincode's wire and reads nothing else. A release reading 1.x to
//! 3.x alone refuses such a directory at its version check, with a message
//! naming the newer release. `framed` holds the four payloads.
//!
//! This build reads every major an earlier release wrote. A directory holding
//! a dense space alone was written at `1.1.0`, one holding a sparse space at
//! `2.0.0` and one a journaled collection saved at `3.0.0`. Each of those was
//! a major because a release reading the earlier ones alone would have found
//! every file it knew and opened the collection without its sparse space, or
//! without every acknowledged mutation since the checkpoint, and without a
//! word. A dense space declared with scalar quantization moved each to its
//! next minor, `1.2.0`, `2.1.0` or `3.1.0`, since a reader that knows the
//! product quantized fields alone refuses such a directory on the first one
//! its `quantization.json` lacks. None of the three moves the version any
//! more, because every release that reads 4.x reads all of them.
//!
//! The four binary artefacts of a directory are read in one layout, decided
//! from `mappings.bin`, which every directory holds. At 4.x it is the frame.
//! Below 4.x it is the frame where `mappings.bin` is a whole frame of its
//! kind, since a frame names what it holds and verifies its own bytes, and
//! bincode's wire otherwise, which `legacy` reads by hand.
//!
//! ## The journal
//!
//! A journaled collection records every mutation to `<name>.zdbwal`, a
//! sibling of the directory rather than a file inside it, and a save is the
//! checkpoint that journal replays onto. `manifest.json` names the file, the
//! collection id both carry and the sequence the checkpoint holds, so the two
//! are paired by content. A load replays every record above that sequence
//! before it hands the collection back. See `crate::journal` for where the
//! file lives and why, and for what is refused.
//!
//! `config.json` declares the sparse space under `spaces`, by value: the
//! space's name, its unlink policy, its lazy threshold, its weighting with
//! the weighting's parameters, and its tokenizer where it takes text. The
//! declaration is validated on load under the rules `Declaration` applies.
//! A tokenizer the engine cannot write down is recorded as `external`, and
//! a directory recording one opens only through `Collection::load_with`
//! with the same implementation handed to it.
//!
//! ## How a save lands
//!
//! Every artefact goes into `<name>.zdbtmp` beside the target and the whole
//! directory is renamed into place at the end, so a reader sees the previous
//! index or this one and never a mixture. Replacing an existing directory needs
//! two renames rather than one, with `<name>.zdbold` holding the previous index
//! between them. See `zeusdb_vector_core::FsStorage` for what that means on
//! each platform and what a killed process leaves behind.
//!
//! Every save and load here reaches the directory through
//! `zeusdb_vector_core::Storage` and `zeusdb_vector_core::Dir`, and makes no
//! filesystem call of its own. A save writes into the directory
//! `Storage::stage` opens and `Staged::commit` puts in place, and a load reads
//! the directory `Storage::dir` names after `Storage::recover`.
//!
//! `manifest.json` records a length for every artefact a save writes and a
//! digest for every JSON one, since a framed artefact and the graph dump each
//! verify their own bytes, and the loader checks what it records before
//! anything parses an artefact. See `ArtefactDigest`.
//!
//! The graph file is ZeusDB's own format, written and read by `graph::dump`,
//! and the loader restores the graph from it rather than rebuilding it by
//! re-inserting every record. See `Collection::restore_graph_from_dump`.
//!
//! This module and `collection::persist` call each other: `save` and `load`
//! on the collection reach `save_index`, `save_manifest` and `load_index`
//! here, and `load_index` builds a collection and restores it
//! through the setters `persist.rs` declares. That is a module cycle inside
//! one crate, which cargo tolerates, and it is the shape the two had in the
//! binding.
//!
//! It replaces the two files the vendored graph crate wrote,
//! `hnsw_index.hnsw.graph` and `hnsw_index.hnsw.data`. A directory saved by
//! 0.6.0 or earlier still holds those two, and opening it rebuilds the graph
//! once and writes the new file on the next save. Nothing reads the old format.

use crate::collection::{
    validate_index_parameters, validate_space_supports_quantization, Collection,
    QuantizationConfig, QuantizationScheme, SparseDeclaration, StorageMode, DEFAULT_SPACE,
};
use crate::journal::{JournalPolicy, Recovery};
use crate::RerankCalibration;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, warn};
use zeusdb_vector_core::{
    checksum_of, frame, frame_begin, frame_finish, unframe, validate_indexed_fields,
    ArtefactRecord, Bounds, Dir, Error, FrameEncoding, FrameKind, FsStorage, IdStore, Int8Codec,
    Inventory, Persist, RecordFields, Restore, SpaceName, Storage, VectorIndex,
    DUMP_FILENAME as GRAPH_DUMP_FILENAME, FRAME_OVERHEAD_BYTES, LEGACY_DUMP_FILENAMES, PQ,
};
use zeusdb_vector_sparse::{PostingsIndex, SparseConfig};
use zeusdb_vector_text::{SimpleTokenizer, TermDictionary, Tokenizer, TokenizerConfig};

mod framed;
mod legacy;

/// The target every record this file emits carries. It is the module path
/// this file had in the binding, so a filter directive naming it still
/// matches. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::persistence";

// ============================================================================
// FORMAT VERSION
// ============================================================================

/// Version written into manifest.json by every save.
///
/// A major, because the four binary artefacts are framed; see the module
/// documentation. A sparse space, a journal and scalar quantization each
/// moved the version an earlier release wrote, and none moves this one,
/// since every release that reads 4.x reads all three.
const FORMAT_VERSION: &str = "4.0.0";

/// The first major whose four binary artefacts are framed.
const FRAMED_FORMAT_MAJOR: u32 = 4;

/// The two artefacts of a trained scalar quantized index.
const INT8_SCALES_FILENAME: &str = "int8_scales.zdbint8";
const INT8_ROWS_FILENAME: &str = "int8_rows.zdbint8";
const INT8_SCALES_CONTENTS: &str =
    "the scales of a scalar quantized index, one a dimension, which every stored row decodes through";
const INT8_ROWS_CONTENTS: &str = "the scalar row of every record";

/// The majors this build reads.
///
/// A minor bump is additive by construction, so any 1.x, any 2.x, any 3.x
/// and any 4.x is read. A different major means the layout changed in a way
/// this build cannot reason about, and guessing at it would be the silent
/// truncation this format has already suffered once.
const SUPPORTED_FORMAT_MAJORS: [u32; 4] = [1, 2, 3, 4];

/// The majors this build reads, as a refusal spells them.
const SUPPORTED_FORMAT_LABEL: &str = "1.x, 2.x, 3.x and 4.x";

/// The first major whose manifest may name a journal.
const JOURNAL_FORMAT_MAJOR: u32 = 3;

// ============================================================================
// THE FOUR BINARY ARTEFACTS
// ============================================================================

// `dim` was bounded here first, at 65,536, on the reasoning that bounding it
// in `validate_index_parameters` would change `create()`'s documented
// contract. `create(dim=2**40)` aborted the process for the same reason a
// config naming it did, so the bound moved to `validate_index_parameters` as
// `MAX_DIM` and the README row changed with it. `load_config` calls that
// function, so the loader still refuses every width it refused, at the same
// number, with config.json named in the message.

/// How a directory holds its four binary artefacts, decided once from
/// `mappings.bin`; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// Each in its frame, as this build writes them.
    Framed,
    /// In bincode's wire, as releases before 4.0.0 wrote them.
    Bincode,
}

impl Layout {
    /// The layout of a directory declaring `major` whose `mappings.bin` holds
    /// `bytes`.
    fn of(major: u32, bytes: &[u8]) -> Layout {
        if major >= FRAMED_FORMAT_MAJOR
            || unframe(bytes, FrameKind::IdMappings, "mappings.bin").is_ok()
        {
            Layout::Framed
        } else {
            Layout::Bincode
        }
    }
}

/// Write a framed artefact into the staging directory and record its length
///
/// The frame verifies its own payload, so the manifest records the length
/// alone, as it does for every framed artefact; see
/// `zeusdb_vector_core::frame`. Durable before this returns, as every
/// `Dir::write` is.
fn write_framed(
    dir: &dyn Dir,
    name: &str,
    bytes: &[u8],
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    zeusdb_vector_core::write_artefact(dir, name, bytes)?;
    ledger.record_digest(name, bytes.len() as u64, None);
    Ok(())
}

// ============================================================================
// A DIGEST PER ARTEFACT
// ============================================================================

/// What the manifest records about one artefact it names
///
/// `bytes` is the file's length and `checksum` is
/// [`zeusdb_vector_core::checksum_of`] over its contents, written as sixteen hex
/// digits. Both are taken from the buffer as it is written, so neither costs a
/// read.
///
/// `checksum` is absent for the graph dump and for every framed artefact. The
/// dump is written by `graph::dump::write_dump`, which streams it and then
/// seeks back to fill the header in, so there is no single buffer to hash and
/// a digest would mean reading the largest artefact in the directory back off
/// the disk. It carries a checksum over its own header and another over its
/// own payload, both verified by `parse_dump` on every load, and a frame
/// carries the same two, verified by `unframe` on every load, so a manifest
/// digest would duplicate a check the loader already makes. Their lengths are
/// recorded and checked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ArtefactDigest {
    pub(crate) bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) checksum: Option<String>,
}

/// The length and digest of every artefact a save has written so far
///
/// Filled as each file lands and handed to `save_manifest`, which is written
/// last and therefore names what really is on disk rather than what the save
/// intended to write.
#[derive(Default)]
pub(crate) struct SaveLedger {
    digests: HashMap<String, ArtefactDigest>,
}

impl SaveLedger {
    fn record_digest(&mut self, name: &str, bytes: u64, checksum: Option<u64>) {
        self.digests.insert(
            name.to_string(),
            ArtefactDigest {
                bytes,
                checksum: checksum.map(|sum| format!("{:016x}", sum)),
            },
        );
    }

    /// The length recorded for an artefact, where one has been.
    pub(crate) fn recorded_bytes(&self, name: &str) -> Option<u64> {
        self.digests.get(name).map(|digest| digest.bytes)
    }
}

/// An index writes its artefacts through the seam's ledger, which is this
/// one, so the manifest records what the index wrote the same way it
/// records what this module wrote.
impl zeusdb_vector_core::Ledger for SaveLedger {
    fn record(&mut self, name: &str, record: zeusdb_vector_core::ArtefactRecord) {
        self.record_digest(name, record.bytes, record.checksum);
    }
}

/// Write one artefact into the staging directory and record what went in
///
/// The artefact is durable before this returns, as every `Dir::write` is.
/// Without that the rename that moves the staging directory into place can be
/// recorded while the bytes it names are still in the page cache, so a power
/// loss leaves an index directory whose manifest is complete and whose
/// artefacts are empty.
fn write_artefact(
    dir: &dyn Dir,
    name: &str,
    bytes: &[u8],
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    dir.write(name, bytes)?;
    ledger.record_digest(name, bytes.len() as u64, Some(checksum_of(bytes)));
    Ok(())
}

/// Hold an artefact to the length and digest the manifest recorded for it
///
/// `files_included` says a file should be there and `check_files_present` says
/// it is. Neither says it holds what was written. A file that is present, the
/// right length and wrong in its contents used to load in silence: an edit to
/// one byte of `metadata.json` came back as that record's metadata, and a
/// flipped byte inside `vectors.bin` came back as that record's vector.
///
/// A directory written before this field existed carries no digests, so nothing
/// is checked and it loads exactly as it did.
fn verify_artefact(name: &str, bytes: &[u8], manifest: &IndexManifest) -> Result<(), Error> {
    let Some(recorded) = manifest.file_digests.get(name) else {
        return Ok(());
    };

    if bytes.len() as u64 != recorded.bytes {
        return Err(Error::ArtefactLengthMismatch {
            name: name.to_string(),
            actual: bytes.len(),
            recorded: recorded.bytes,
            contents: artefact_contents(name),
        });
    }

    let Some(expected) = recorded.checksum.as_deref() else {
        return Ok(());
    };
    let actual = format!("{:016x}", checksum_of(bytes));
    if actual != expected {
        return Err(Error::ArtefactDigestMismatch {
            name: name.to_string(),
            actual,
            expected: expected.to_string(),
            contents: artefact_contents(name),
        });
    }

    Ok(())
}

/// Read an artefact and verify it before anything parses it
fn read_artefact(dir: &dyn Dir, name: &str, manifest: &IndexManifest) -> Result<Vec<u8>, Error> {
    let bytes = dir.read(name)?;
    verify_artefact(name, &bytes, manifest)?;
    Ok(bytes)
}

/// The same, for the artefacts that are JSON
fn read_artefact_string(
    dir: &dyn Dir,
    name: &str,
    manifest: &IndexManifest,
) -> Result<String, Error> {
    let bytes = read_artefact(dir, name, manifest)?;
    String::from_utf8(bytes).map_err(|e| Error::ArtefactNotUtf8 {
        name: name.to_string(),
        error: e.to_string(),
    })
}

/// The length manifest.json records for the graph dump, where it records one
///
/// The dump is read by `graph::dump::parse_dump` rather than through
/// `read_artefact`, because it is streamed rather than held whole in memory.
/// This is the part of the digest check that still applies to it.
pub(crate) fn recorded_dump_length(manifest: &IndexManifest, name: &str) -> Option<u64> {
    manifest.file_digests.get(name).map(|entry| entry.bytes)
}

/// Refuse a directory this build cannot interpret, and return the major it
/// declares.
fn check_format_version(format_version: &str) -> Result<u32, Error> {
    let major = format_version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
        .ok_or_else(|| Error::FormatVersionUnparsable {
            format_version: format_version.to_string(),
            current: FORMAT_VERSION,
        })?;

    if !SUPPORTED_FORMAT_MAJORS.contains(&major) {
        return Err(Error::FormatVersionUnsupported {
            format_version: format_version.to_string(),
            supported: SUPPORTED_FORMAT_LABEL,
            newer: major > SUPPORTED_FORMAT_MAJORS[SUPPORTED_FORMAT_MAJORS.len() - 1],
        });
    }

    Ok(major)
}

// ============================================================================
// DIRECTORY COMPLETENESS
// ============================================================================

/// Whether an artefact is one the loader can produce again rather than read
///
/// The graph is the only one. Every record carries what the graph is built
/// from, so a directory that lost its dump is rebuilt rather than refused, and
/// that has always been the behaviour. The list holds the name this build
/// writes and the pair 0.6.0 and earlier wrote, because a directory saved by
/// one of those names both of them under `files_included` and neither is
/// needed to reopen it.
///
/// A save now writes the dump before the manifest and moves the whole
/// directory into place afterwards, so a directory this build wrote and a
/// reader can see always holds the dump its manifest names. The exemption
/// stays for the directories that came before it, where the manifest was
/// written first and a save interrupted between the two left a manifest naming
/// a dump that was never written.
fn is_derived_artefact(name: &str) -> bool {
    name == GRAPH_DUMP_FILENAME || LEGACY_DUMP_FILENAMES.contains(&name)
}

/// What an artefact holds, for the message its absence produces
///
/// An unrecognised name is still load bearing. `files_included` has named only
/// what the save wrote since the field appeared in 0.3.0, so a name this build
/// does not know is a component a later release wrote, and absorbing its loss
/// is the failure this check exists to stop.
fn artefact_contents(name: &str) -> &'static str {
    match name {
        "config.json" => "the HNSW parameters, the saved record count and the index level metadata",
        "mappings.bin" => "the mapping from every external record id to its internal graph id",
        "metadata.json" => "the metadata of every record, which is what a filtered search reads",
        "vectors.bin" => "the raw vector of every record",
        "quantization.json" => {
            "the quantization configuration of either scheme and the training state"
        }
        "pq_centroids.bin" => "the trained PQ codebook, which every stored code decodes through",
        "pq_codes.bin" => "the quantized code of every record",
        INT8_SCALES_FILENAME => INT8_SCALES_CONTENTS,
        INT8_ROWS_FILENAME => INT8_ROWS_CONTENTS,
        _ if name.ends_with("/postings.zdbsparse") => {
            "the postings of a sparse space, being every record's term ids and weights"
        }
        _ if name.ends_with("/terms.zdbdict") => {
            "the term dictionary of a text layer, being every term and its id"
        }
        _ => "a component of the saved index that this build does not recognise",
    }
}

/// What the dictionary artefact holds, for the message its absence produces.
const DICTIONARY_CONTENTS: &str =
    "the term dictionary of a text layer, being every term and its id";

/// Refuse a directory that does not hold what its manifest says it holds
///
/// `files_included` is written from what the save actually wrote. Every entry
/// is pushed under the same condition the writer of that file tests, inside one
/// save holding the mutation lock, so the list is an inventory rather than a
/// statement about the storage mode. That has been true of every release that
/// wrote the field, which is 0.3.0 onwards, and it is what makes a named file
/// that is absent a directory that lost something rather than a directory this
/// build is reading wrongly.
///
/// This runs before any artefact is read, so a directory missing two files
/// names the first rather than failing on whichever one a partial load happens
/// to reach.
///
/// `manifest.json` is the last file a save writes and the directory is moved
/// into place whole, so an interrupted save cannot produce this state at all. A
/// directory that reaches it lost the file after a save that finished, or was
/// copied without it.
///
/// Without it a `quantized_with_raw` directory whose `vectors.bin` never landed
/// opened as a complete index built entirely from PQ reconstructions, and one
/// that lost `quantization.json` opened as an unquantized index. Both were
/// silent.
fn check_files_present(dir: &dyn Dir, manifest: &IndexManifest) -> Result<(), Error> {
    let missing: Vec<&str> = manifest
        .files_included
        .iter()
        .map(String::as_str)
        .filter(|name| !is_derived_artefact(name))
        .filter(|name| !dir.exists(name))
        .collect();

    let Some(&first) = missing.first() else {
        debug!(target: LOG_TARGET, "Every file manifest.json names is present ({} checked)",
            manifest.files_included.len()
        );
        return Ok(());
    };

    Err(Error::ArtefactsMissing {
        missing: missing.iter().map(|name| name.to_string()).collect(),
        contents: artefact_contents(first),
    })
}

/// Whether the manifest's inventory names an artefact
///
/// The optional artefacts are read only when `files_included` names them. A
/// save used to replace files one at a time and remove none, so a raw index
/// saved over a quantized one reopened as a quantized index holding the
/// previous save's codebook and codes, and the record count agreed so nothing
/// caught it. A save now builds its directory from nothing, so no artefact of
/// an earlier save survives one and this check no longer has anything to
/// exclude. It stays because a directory written by an earlier release can
/// still hold those files.
///
/// The graph dump is not gated this way. It is derived, it carries its own
/// checks on node count, distance kind and `m`, and it already falls back to
/// the rebuild when any of them disagree.
fn manifest_names(manifest: &IndexManifest, name: &str) -> bool {
    manifest.files_included.iter().any(|entry| entry == name)
}

// ============================================================================
// PERSISTENCE DATA STRUCTURES
// ============================================================================

/// Manifest file structure - tracks index metadata and included files
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexManifest {
    pub(crate) format_version: String,
    pub(crate) zeusdb_version: String,
    pub(crate) created_at: String,
    pub(crate) saved_at: String,
    pub(crate) total_vectors: usize,
    pub(crate) index_type: String,
    pub(crate) has_quantization: bool,
    pub(crate) quantization_trained: bool,
    pub(crate) storage_mode: String,
    pub(crate) files_included: Vec<String>,
    pub(crate) files_excluded: Vec<String>,

    /// The length and digest of every artefact `files_included` names
    ///
    /// A map beside the list rather than a change to the list, because
    /// `files_included` is a `Vec<String>` in every release that has read this
    /// file and turning it into a list of objects would stop those releases
    /// parsing a directory this one wrote. serde ignores a field it does not
    /// know, so an older build reads a directory written here and this build
    /// reads one written there, where the map defaults to empty and nothing is
    /// verified.
    ///
    /// See `ArtefactDigest` for what is recorded and why the graph dump carries
    /// a length alone.
    #[serde(default)]
    pub(crate) file_digests: HashMap<String, ArtefactDigest>,

    /// Every byte the directory holds except `manifest.json` itself
    ///
    /// The manifest is now the last file a save writes, so it does not exist
    /// when the figure is taken and cannot count itself. It used to be written
    /// before the graph dump and then rewritten through a temporary file to
    /// record a total it had missed the largest artefact of, which is a second
    /// write this ordering removes.
    pub(crate) total_size_mb: f64,
    pub(crate) compression_info: Option<CompressionInfo>,

    /// The journal beside the directory, where the collection that saved it
    /// held one.
    ///
    /// Absent from the file where none was held. Present, it names the
    /// journal a load replays. A release reading 1.x and 2.x alone would
    /// ignore it and open the checkpoint without the journal's mutations and
    /// without a word, which the 3.x major stopped and the 4.x major still
    /// stops; see the module documentation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) journal: Option<JournalManifest>,
}

/// How `manifest.json` names the journal beside the directory.
///
/// The three values are what pair a directory with a file that is not in it.
/// `collection_id` is drawn when a collection is built and is in the
/// journal's own header too, so a journal from another index is refused by
/// content. `sequence` is what the journal had reached when this checkpoint
/// was taken, so a recovery replays every record above it and skips every
/// record at or below it. `file` is the sibling's name as the checkpoint
/// wrote it, which a refusal quotes; the file a load actually opens is
/// derived from the directory it was handed, so a directory renamed together
/// with its journal opens under the new name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct JournalManifest {
    pub(crate) file: String,
    pub(crate) sequence: u64,
    pub(crate) collection_id: String,
}

/// The manifest is what an index reads its artefacts' recorded lengths from.
///
/// A framed artefact is recorded by its length alone, so `checksum` is
/// `None` for one, and the frame's own payload checksum is what the reader
/// verifies. A directory written before the manifest carried digests
/// records nothing, and an index restoring from one is refused before it
/// reads, since no such directory holds a space.
impl Inventory for IndexManifest {
    fn recorded(&self, name: &str) -> Option<ArtefactRecord> {
        let digest = self.file_digests.get(name)?;
        Some(ArtefactRecord {
            bytes: digest.bytes,
            checksum: digest
                .checksum
                .as_deref()
                .and_then(|hex| u64::from_str_radix(hex, 16).ok()),
        })
    }
}

/// Compression statistics for quantized indexes
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CompressionInfo {
    pub(crate) original_size_mb: f64,
    pub(crate) compressed_size_mb: f64,
    pub(crate) compression_ratio: f64,
}

/// Index configuration for reconstruction
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexConfig {
    pub(crate) dim: usize,
    pub(crate) space: String,
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) expected_size: usize,
    pub(crate) id_counter: usize,
    pub(crate) vector_count: usize,

    /// How many generated ids the index had issued, being the `N` of `vec_N`
    ///
    /// Separate from `id_counter`, which `clear` resets and this does not. See
    /// `Collection::generated_ids`. Defaulted, so a directory written before the
    /// field existed loads with a zero and takes its floor from the records it
    /// holds instead.
    #[serde(default)]
    pub(crate) generated_ids: usize,

    /// Index level metadata set through `add_metadata`
    ///
    /// Defaulted rather than required, so a directory written before this field
    /// existed loads with an empty map instead of failing to parse. Written in
    /// key order, so two saves of the same metadata write the same bytes.
    #[serde(default)]
    pub(crate) metadata: BTreeMap<String, String>,

    /// The filterable fields declared at `create()`, in declaration order.
    ///
    /// **The columns themselves are not saved.** They are derived from
    /// `metadata.json`, which is written whole and read whole, so rebuilding
    /// them at load costs one pass over the records and keeps the directory
    /// format to the files it already had. What has to survive a round trip is
    /// the declaration, because nothing else records which fields a user chose.
    ///
    /// Defaulted, so a directory written before this field existed loads with
    /// no declaration and behaves exactly as it did. `serde` ignores fields it
    /// does not know, so a directory written with one also opens in a build
    /// that predates it, at the cost of the columns rather than of the load.
    #[serde(default)]
    pub(crate) indexed_fields: Vec<String>,

    /// The sparse space declared beside the dense one, by value, where one
    /// was.
    ///
    /// Absent from the file where none was declared, so a dense-only
    /// directory's `config.json` is byte for byte what it was before the
    /// field existed. A release reading 1.x alone would ignore it and open
    /// the collection without its sparse space, which the 2.x major stopped
    /// and the 4.x major still stops; see the module documentation.
    /// A list rather than one record, so a further space is a further entry
    /// rather than a further field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) spaces: Vec<SpaceRecord>,
}

/// One sparse space as `config.json` declares it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SpaceRecord {
    /// The name the space was declared under, which is the directory under
    /// `spaces/` its artefacts are written to.
    pub(crate) name: String,
    /// The family of vector the space holds. `sparse` is the one this build
    /// writes, and a reader that is not the engine reads it before the
    /// fields below.
    pub(crate) kind: String,
    /// The index's own declaration, every field named.
    pub(crate) index: SparseConfig,
    /// The tokenizer, where the space takes text. `simple` is the built-in
    /// one and `external` is an implementation the caller keeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tokenizer: Option<TokenizerConfig>,
}

/// The one kind of space `config.json` declares under `spaces`.
const SPARSE_KIND: &str = "sparse";

/// Hold the spaces `config.json` declares to the rules `Declaration`
/// applies to a caller's own, so a hand edited file cannot build a space
/// the collection would refuse to declare.
fn validate_spaces(spaces: &[SpaceRecord], file: &str) -> Result<(), Error> {
    let invalid = |detail: String| Error::SpaceRecordInvalid {
        file: file.to_string(),
        detail,
    };
    if spaces.len() > 1 {
        return Err(invalid(format!(
            "a collection holds one sparse space and {} are declared",
            spaces.len()
        )));
    }
    for space in spaces {
        if space.kind != SPARSE_KIND {
            return Err(invalid(format!(
                "space '{}' declares kind '{}', which this build does not hold",
                space.name, space.kind
            )));
        }
        SpaceName::new(&space.name).map_err(|e| invalid(e.to_string()))?;
        if space.name == DEFAULT_SPACE {
            return Err(invalid(format!(
                "space '{}' takes the dense space's name",
                space.name
            )));
        }
        space.index.validate().map_err(|e| invalid(e.to_string()))?;
    }
    Ok(())
}

/// The tokenizer a restored text layer runs, from what `config.json`
/// recorded and what the caller handed to `load`.
///
/// The built-in tokenizer is recorded as `simple` and is built here where
/// none was handed. A caller's own is recorded as `external` and is
/// accepted only when handed. A handed tokenizer whose own declaration is
/// not the recorded one is refused, and so is one handed to a directory
/// with no text layer, since ignoring it would open the collection under a
/// tokenizer the caller did not ask for.
fn resolve_tokenizer(
    space: Option<&SpaceRecord>,
    handed: Option<Arc<dyn Tokenizer>>,
) -> Result<Option<Arc<dyn Tokenizer>>, Error> {
    let recorded = space.and_then(|space| space.tokenizer.as_ref());
    match (recorded, handed) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(Error::TokenizerUnexpected),
        (Some(TokenizerConfig::Simple), None) => Ok(Some(Arc::new(SimpleTokenizer))),
        (Some(TokenizerConfig::External), None) => Err(Error::TokenizerRequired {
            space: space.map(|s| s.name.clone()).unwrap_or_default(),
        }),
        (Some(recorded), Some(handed)) => {
            let declared = handed.config();
            if declared == *recorded {
                Ok(Some(handed))
            } else {
                Err(Error::TokenizerMismatch {
                    space: space.map(|s| s.name.clone()).unwrap_or_default(),
                    recorded: recorded.name(),
                    handed: declared.name(),
                })
            }
        }
    }
}

/// Complete quantization configuration and state
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct QuantizationPersistence {
    pub(crate) r#type: String,
    pub(crate) subvectors: usize,
    pub(crate) bits: usize,
    pub(crate) training_size: usize,
    pub(crate) max_training_vectors: Option<usize>,
    pub(crate) storage_mode: String,
    pub(crate) is_trained: bool,
    pub(crate) training_completed_at: Option<String>,
    pub(crate) memory_stats: Option<MemoryStats>,
    pub(crate) pq_config: PQConfig,
    #[serde(default)]
    pub(crate) training_ids: Vec<String>,
    #[serde(default)]
    pub(crate) training_threshold_reached: bool,
    /// What training measured about the rerank fetch on this index's own data.
    ///
    /// Absent from every directory written before the calibration existed, so
    /// it defaults to `None` and those indexes fall back to the corpus terms
    /// they were built against. See `RerankCalibration`.
    #[serde(default)]
    pub(crate) rerank_calibration: Option<RerankCalibration>,
}

/// Memory usage statistics for quantization
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MemoryStats {
    pub(crate) centroid_storage_mb: f64,
    pub(crate) compression_ratio: f64,
    pub(crate) centroids_per_subvector: usize,
    pub(crate) total_centroids: usize,
}

/// Product Quantization configuration details
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PQConfig {
    pub(crate) dim: usize,
    pub(crate) sub_dim: usize,
    pub(crate) num_centroids: usize,
}

/// quantization.json under `type: int8`: the scalar declaration and the
/// training state, and none of the product quantized fields.
///
/// A reader that knows the product quantized layout alone refuses this file
/// on its first missing field, which is what lets the format version move by
/// a minor rather than a major; see the module documentation.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Int8Persistence {
    pub(crate) r#type: String,
    pub(crate) scale: String,
    pub(crate) training_size: usize,
    pub(crate) max_training_vectors: Option<usize>,
    pub(crate) storage_mode: String,
    pub(crate) is_trained: bool,
    pub(crate) training_completed_at: Option<String>,
    #[serde(default)]
    pub(crate) training_ids: Vec<String>,
    #[serde(default)]
    pub(crate) training_threshold_reached: bool,
    /// Values the encode has clipped since the scales were fitted; see
    /// `DenseIndex::saturated`.
    #[serde(default)]
    pub(crate) saturated_values: u64,
}

/// quantization.json, as one of the two layouts it takes.
enum QuantizationFile {
    Pq(QuantizationPersistence),
    Int8(Int8Persistence),
}

/// PQ codebook laid out as [subvector][centroid][dimension within subvector]
type Centroids = Vec<Vec<Vec<f32>>>;

/// Everything the loader reads back for a quantized index
struct QuantizationArtefacts {
    config: QuantizationFile,
    centroids: Option<Centroids>,
    codes: HashMap<String, Vec<u8>>,
    /// The fitted scalar codec, from `int8_scales.zdbint8` on a trained
    /// scalar directory.
    int8: Option<Arc<Int8Codec>>,
    /// Every record's scalar row from `int8_rows.zdbint8`, by internal id,
    /// ascending.
    rows: Int8Rows,
}

/// Every row of a trained scalar directory in one block, with the internal
/// id of each beside it, ascending
///
/// `int8_rows.zdbint8` holds one row a record. The loader used to hold one
/// heap block a row on top of the file, every one of them freed at the end of
/// the load, which is what an allocator that keeps freed small blocks left
/// the process holding. One block holds them now, and a row is a slice of it.
#[derive(Default)]
pub(crate) struct Int8Rows {
    ids: Vec<u32>,
    data: Vec<i8>,
    width: usize,
}

impl Int8Rows {
    fn with_capacity(entries: usize, width: usize) -> Self {
        Int8Rows {
            ids: Vec::with_capacity(entries),
            data: Vec::with_capacity(entries.saturating_mul(width)),
            width,
        }
    }

    fn push(&mut self, id: u32, row: &[u8]) {
        self.ids.push(id);
        self.data.extend(row.iter().map(|&b| b as i8));
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// Every row with its internal id, ascending.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &[i8])> + '_ {
        self.ids
            .iter()
            .zip(self.data.chunks_exact(self.width.max(1)))
            .map(|(&id, row)| (id as usize, row))
    }
}

/// Training collection state, applied once the graph is back
///
/// The collected ids and the threshold flag are what `quantization.json`
/// recorded, and they are applied after the graph is restored or rebuilt so
/// that the flag is judged against the records the loaded collection holds.
/// Neither fallback touches the training collection: both build the graph off
/// to the side under the mapped ids and swap it in.
struct TrainingState {
    ids: Vec<String>,
    threshold_reached: bool,
    is_trained: bool,
    training_size: usize,
    /// The clipped value count a scalar directory recorded, applied once
    /// the index holds its graph, since the graph's restore builds a fresh
    /// index whose count starts at zero.
    saturated: u64,
}

impl TrainingState {
    fn from(config: &QuantizationFile) -> Self {
        match config {
            QuantizationFile::Pq(config) => TrainingState {
                ids: config.training_ids.clone(),
                threshold_reached: config.training_threshold_reached,
                is_trained: config.is_trained,
                training_size: config.training_size,
                saturated: 0,
            },
            QuantizationFile::Int8(config) => TrainingState {
                ids: config.training_ids.clone(),
                threshold_reached: config.training_threshold_reached,
                is_trained: config.is_trained,
                training_size: config.training_size,
                saturated: config.saturated_values,
            },
        }
    }

    fn apply(self, index: &mut Collection) {
        // A trained index cleared its collection when training ran, so this is
        // only ever populated for an index saved while still collecting.
        let collected = self.ids.len();
        index.set_training_ids(self.ids);

        // The saved flag is authoritative for a trained index. For an untrained
        // one it is recomputed, so a directory whose collection was truncated
        // does not come back claiming a threshold it no longer meets.
        let reached = if self.is_trained {
            self.threshold_reached
        } else {
            collected >= self.training_size
        };
        index.set_training_threshold_reached(reached);
        if self.saturated > 0 {
            index.set_int8_saturated(self.saturated);
        }

        debug!(target: LOG_TARGET, "Training state restored ({} collected ids, threshold reached: {})",
            collected, reached
        );
    }
}

// ============================================================================
// INDIVIDUAL COMPONENT LOADERS
// ============================================================================

/// Load index configuration from config.json
fn load_config(dir: &dyn Dir, manifest: &IndexManifest) -> Result<IndexConfig, Error> {
    debug!(target: LOG_TARGET, "Loading config.json...");

    let config_path = dir.locate("config.json");
    let config_data = read_artefact_string(dir, "config.json", manifest)?;

    let config: IndexConfig =
        serde_json::from_str(&config_data).map_err(|e| Error::ArtefactParseFailed {
            name: "config.json",
            error: e.to_string(),
        })?;

    // The five values `build` validates, validated here too.
    //
    // Parsing proves the file is JSON of the right shape and nothing more. Until
    // this ran, `dim`, `m`, `ef_construction`, `expected_size` and `space` went
    // straight from the file into `new_empty`, which validates none of them, and
    // then into `Backend::sized`, which clamps `dim` up to 1, `m` into 2 to 256
    // and `expected_size` up to 1 without saying so. A config naming `m: 0` came
    // back as an index at `m: 2`, and one naming an unknown `space` came back
    // scoring cosine whatever it had been saved with. A zero `dim` was refused,
    // but by a later check comparing a record against the declared width, so the
    // message named the record rather than the config.
    //
    // The file is named in the message because a caller reading `dim must be
    // positive` off a `load()` has no argument of their own to look at.
    validate_index_parameters(
        config.dim,
        &config.space,
        config.m,
        config.ef_construction,
        config.expected_size,
        &format!("{}: ", config_path),
    )?;
    // `id_counter` too, which those five do not cover and which sizes an
    // allocation rather than a behaviour.
    //
    // The internal id a record is inserted under is the index into the graph's
    // id-to-node array, so that array is `id_counter + 1` slots of four bytes.
    // A hand edited config declaring 2^40 loaded without complaint, and the
    // next `add` asked the allocator for 4,398,046,511,112 bytes and **aborted
    // the process**. An allocation failure does not unwind, so no `catch_unwind`
    // sees one and a Python caller gets a dead interpreter with no traceback.
    //
    // It is the same root cause as the graph dump's origin id, seen from the
    // other side: one dense array, two unvalidated sources for its index. The
    // dump's side is checked in `graph::dump::parse_dump` against this same
    // field, so bounding it here bounds both.
    //
    // The ceiling is `u32::MAX` because a node index is a `u32` and both graph
    // constructors refuse a graph holding more points than that. Every id was
    // issued to a record the graph then held a node for, so an index that
    // issued more ids than a node index can name has a graph it could not have
    // built. The check refuses nothing that can exist.
    if config.id_counter > u32::MAX as usize {
        return Err(Error::IdCounterTooLarge {
            file: config_path.clone(),
            id_counter: config.id_counter,
        });
    }
    // The declaration too, for the same reason. A config naming a field twice,
    // or naming a reserved filter key, would build a store the index could not
    // use, and the failure would surface as a filter that quietly walked.
    validate_indexed_fields(&config.indexed_fields, &format!("{}: ", config_path))?;
    // The spaces too, under the rules a declaration is held to.
    validate_spaces(&config.spaces, &config_path)?;

    debug!(target: LOG_TARGET, "config.json loaded");
    Ok(config)
}

/// Read the sparse space's artefacts back and install them, where the
/// collection declares one.
///
/// Runs after the id mappings are restored and before the graph is, for two
/// reasons. The largest internal id the mappings hold is what bounds the
/// space's slot table, and the graph's rebuild fallback replays every record
/// through `add`, which reads each record's sparse half out of the space and
/// carries it through; see `Collection::rebuild_from_records`.
///
/// Every record the space holds must be one the mappings name. A record may
/// leave the space empty, so the space may hold fewer, and a space holding
/// an id the mappings do not is a directory whose two artefacts came from
/// different saves. A text layer's dictionary must hold every term id the
/// postings carry, for the same reason.
fn restore_spaces(
    index: &Collection,
    dir: &dyn Dir,
    manifest: &IndexManifest,
) -> Result<(), Error> {
    let Some((name, space)) = index.sparse_named() else {
        return Ok(());
    };
    let prefix = Collection::space_prefix(name);
    let largest_id = index.ids().highest_slot().unwrap_or(0);
    let bounds = Bounds {
        min_records: 0,
        max_records: largest_id,
        max_bytes: u64::MAX,
    };
    let restored = PostingsIndex::restore(space.config(), &prefix, dir, manifest, &bounds)?;
    let unmapped = {
        let ids = index.ids();
        let mut unmapped = None;
        restored.live_set().for_each_while(|slot| {
            if ids.contains_slot(slot) {
                true
            } else {
                unmapped = Some(slot);
                false
            }
        });
        unmapped
    };
    if let Some(slot) = unmapped {
        return Err(Error::SparseRecordUnmapped {
            space: name.as_str().to_string(),
            id: u32::try_from(slot).unwrap_or(u32::MAX),
        });
    }
    if space.text.is_some() {
        let dictionary_name = Collection::dictionary_name(&prefix);
        let bytes = zeusdb_vector_core::read_artefact(
            dir,
            &dictionary_name,
            manifest,
            DICTIONARY_CONTENTS,
            u64::MAX,
        )?;
        let dictionary = TermDictionary::decode(&bytes, &dictionary_name)?;
        if let Some(term) = restored.max_dim() {
            if term as usize >= dictionary.len() {
                return Err(Error::TermIdBeyondDictionary {
                    space: name.as_str().to_string(),
                    term,
                    terms: dictionary.len(),
                });
            }
        }
        debug!(target: LOG_TARGET, "{} restored ({} terms)", dictionary_name, dictionary.len());
        index.set_dictionary(dictionary);
    }
    debug!(target: LOG_TARGET, "{}postings.zdbsparse restored ({} records, {} postings)",
        prefix,
        restored.len(),
        restored.postings_total()
    );
    index.set_sparse_index(restored);
    Ok(())
}

/// Build the id store from mappings.bin, and say how the directory holds its
/// four binary artefacts
///
/// Every internal id the artefact names is held to `id_counter`, the last id
/// `config.json` says the index issued, before the store is reserved. The
/// store, the metadata, the columns and a sparse space all hold an entry for
/// every id up to the highest a record holds, so a file naming one record at
/// a high id would have the loader reserve and fill an entry for every id
/// below it. A save holds the mutation guard for its whole run, so every id
/// the file names was issued by the counter it records. The graph dump's
/// origin ids are held to the same counter; see `Expected::max_origin_id`.
///
/// The store is built before the collection is, because the framed
/// `vectors.bin` and `pq_codes.bin` name records by internal id and are held
/// to it as they are read. The collection takes it whole.
fn load_ids(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    major: u32,
    config: &IndexConfig,
) -> Result<(IdStore, Layout), Error> {
    debug!(target: LOG_TARGET, "Loading mappings.bin...");

    let mappings_data = read_artefact(dir, "mappings.bin", manifest)?;

    let layout = Layout::of(major, &mappings_data);
    let ids = match layout {
        Layout::Framed => framed_ids(&mappings_data, config)?,
        Layout::Bincode => bincode_ids(&mappings_data, config)?,
    };

    debug!(target: LOG_TARGET, "mappings.bin loaded");
    Ok((ids, layout))
}

/// The id store a framed mappings.bin holds
///
/// One entry a record, in increasing internal id, so an internal id holds one
/// id, the highest is the last, and the store is reserved once at exactly
/// what it will hold. An id held under two internal ids is refused.
fn framed_ids(bytes: &[u8], config: &IndexConfig) -> Result<IdStore, Error> {
    let invalid = |error: String| Error::ArtefactParseFailed {
        name: "mappings.bin",
        error,
    };
    let records = framed::Mappings::read(bytes, "mappings.bin")?;
    let highest = match records.last() {
        Some((internal_id, id)) if internal_id > config.id_counter => {
            return Err(invalid(format!(
                "the record '{}' holds internal id {} and config.json counted {}",
                id, internal_id, config.id_counter
            )));
        }
        Some((internal_id, _)) => internal_id,
        None => 0,
    };
    let mut store = IdStore::new(config.expected_size);
    store.reserve(records.len(), highest);
    for (internal_id, id) in records.iter() {
        store.insert(internal_id, id)?;
    }
    if store.len() != records.len() {
        return Err(invalid(format!(
            "{} records hold {} distinct ids, so an id is held under two internal ids",
            records.len(),
            store.len()
        )));
    }
    Ok(store)
}

/// The id store bincode's two maps describe, as releases before 4.0.0 wrote
/// them
///
/// The forward map is what the store is built from, and the reverse map is
/// held to being its exact inverse, since the two were written from one
/// structure and a file whose two halves disagree describes two record sets.
fn bincode_ids(bytes: &[u8], config: &IndexConfig) -> Result<IdStore, Error> {
    let invalid = |error: String| Error::ArtefactParseFailed {
        name: "mappings.bin",
        error,
    };
    let legacy::Maps { id_map, rev_map } = legacy::read_mappings(bytes, "mappings.bin")?;
    let top = id_map
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
    if let Some((id, &internal_id)) = top {
        if internal_id > config.id_counter {
            return Err(invalid(format!(
                "the forward map names internal id {} for '{}' and config.json counted {}",
                internal_id, id, config.id_counter
            )));
        }
    }
    let highest = top.map_or(0, |(_, &internal_id)| internal_id);
    let mut store = IdStore::new(config.expected_size);
    store.reserve(id_map.len(), highest);
    for (id, &internal_id) in &id_map {
        store.insert(internal_id, id)?;
    }
    if store.len() != id_map.len() {
        return Err(invalid(format!(
            "the forward map names {} records under {} internal ids",
            id_map.len(),
            store.len()
        )));
    }
    if rev_map.len() != id_map.len() {
        return Err(invalid(format!(
            "the forward map holds {} records and the reverse map {}",
            id_map.len(),
            rev_map.len()
        )));
    }
    if let Some((internal_id, id)) = rev_map
        .iter()
        .find(|(&internal_id, id)| store.name(internal_id) != Some(id.as_str()))
    {
        return Err(invalid(format!(
            "the reverse map names internal id {} as '{}' and the forward map does not",
            internal_id, id
        )));
    }
    Ok(store)
}

/// Load vector metadata from metadata.json
fn load_metadata(
    dir: &dyn Dir,
    manifest: &IndexManifest,
) -> Result<HashMap<String, HashMap<String, Value>>, Error> {
    debug!(target: LOG_TARGET, "Loading metadata.json...");

    let metadata_data = read_artefact_string(dir, "metadata.json", manifest)?;

    let metadata: HashMap<String, HashMap<String, Value>> = serde_json::from_str(&metadata_data)
        .map_err(|e| Error::ArtefactParseFailed {
            name: "metadata.json",
            error: e.to_string(),
        })?;

    debug!(target: LOG_TARGET, "metadata.json loaded");
    Ok(metadata)
}

/// Load raw vectors from vectors.bin, keyed by external id
///
/// Read only when the manifest names it. A trained `quantized_only` index
/// writes none, and a directory saved over one that did keeps the file the
/// earlier save left. See `manifest_names`.
///
/// A framed file names each record by internal id, and its vector is filed
/// under the id `ids` holds for it, once the file is held to `ids`; see
/// `framed_vectors`.
fn load_vectors(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    ids: &IdStore,
    layout: Layout,
    dim: usize,
) -> Result<HashMap<String, Vec<f32>>, Error> {
    debug!(target: LOG_TARGET, "Loading vectors.bin...");

    if !manifest_names(manifest, "vectors.bin") {
        debug!(target: LOG_TARGET, "manifest.json does not list vectors.bin, so no raw vectors are read");
        return Ok(HashMap::new());
    }

    let vectors_data = read_artefact(dir, "vectors.bin", manifest)?;

    let vectors = match layout {
        Layout::Framed => {
            let rows = framed_vectors(&vectors_data, ids, dim)?;
            let mut vectors = HashMap::with_capacity(rows.len());
            for (internal_id, row) in rows.iter() {
                if let Some(id) = ids.name(internal_id) {
                    vectors.insert(id.to_string(), framed::floats(row).collect());
                }
            }
            vectors
        }
        Layout::Bincode => {
            let vectors = legacy::read_vectors(&vectors_data, "vectors.bin", ids.len())?;
            check_vectors_are_finite(&vectors)?;
            vectors
        }
    };

    debug!(target: LOG_TARGET, "vectors.bin loaded");
    Ok(vectors)
}

/// Refuse a stored vector holding a NaN or an infinity
///
/// `add` has always refused a non-finite value, so one can only reach
/// vectors.bin through a release that did not validate every input path or
/// through a directory edited after it was written. The check belongs to the
/// loader rather than to the graph rebuild, because the graph is now restored
/// from its dump and the rebuild that used to catch this does not run. A record
/// holding a NaN scores as NaN against every query and orders arbitrarily, so
/// the index would answer wrongly rather than visibly fail.
fn check_vectors_are_finite(vectors: &HashMap<String, Vec<f32>>) -> Result<(), Error> {
    let mut offenders: Vec<&String> = vectors
        .iter()
        .filter(|(_, vector)| vector.iter().any(|value| !value.is_finite()))
        .map(|(id, _)| id)
        .collect();

    if offenders.is_empty() {
        return Ok(());
    }

    offenders.sort();
    Err(Error::VectorsNotFinite {
        offenders: offenders.into_iter().cloned().collect(),
        total: vectors.len(),
    })
}

/// What the loader holds of vectors.bin
///
/// A raw index takes every vector back from the graph dump, which carries
/// its store, so the file is read for its record count and its finiteness
/// check and nothing of it is kept, see `count_vectors`. A quantized index
/// holds the map, since a `quantized_with_raw` index places the vectors by id
/// once the graph is back and a product quantized rebuild replays them. The
/// rebuild fallback of a raw index reads the file again, through `hold`.
///
/// Held whole, the file cost one heap block a record for the whole of the
/// load, beside the dump's own copy of every vector, and every one of those
/// blocks was freed at the end. An allocator that keeps freed small blocks
/// left the process holding them after the load returned.
enum RawVectors {
    Held(HashMap<String, Vec<f32>>),
    Counted(usize),
}

impl RawVectors {
    fn len(&self) -> usize {
        match self {
            RawVectors::Held(map) => map.len(),
            RawVectors::Counted(count) => *count,
        }
    }

    fn held(&self) -> Option<&HashMap<String, Vec<f32>>> {
        match self {
            RawVectors::Held(map) => Some(map),
            RawVectors::Counted(_) => None,
        }
    }

    fn into_held(self) -> HashMap<String, Vec<f32>> {
        match self {
            RawVectors::Held(map) => map,
            RawVectors::Counted(_) => HashMap::new(),
        }
    }

    /// Read the map where only the count was kept.
    fn hold(
        &mut self,
        dir: &dyn Dir,
        manifest: &IndexManifest,
        ids: &IdStore,
        layout: Layout,
        dim: usize,
    ) -> Result<(), Error> {
        if let RawVectors::Counted(_) = self {
            *self = RawVectors::Held(load_vectors(dir, manifest, ids, layout, dim)?);
        }
        Ok(())
    }
}

/// Read vectors.bin for its record count and its finiteness check, keeping
/// nothing
///
/// Read only when the manifest names it, as `load_vectors` is. A framed file
/// is held to the mappings as `framed_vectors` holds it, and its count is the
/// records it holds. A file of bincode's wire is walked as
/// `legacy::walk_vectors` walks it, an id the file holds twice counted once,
/// and a record whose vector is not finite is named exactly as
/// `check_vectors_are_finite` names it.
fn count_vectors(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    ids: &IdStore,
    layout: Layout,
    dim: usize,
) -> Result<usize, Error> {
    debug!(target: LOG_TARGET, "Loading vectors.bin...");

    if !manifest_names(manifest, "vectors.bin") {
        debug!(target: LOG_TARGET, "manifest.json does not list vectors.bin, so no raw vectors are read");
        return Ok(0);
    }

    let vectors_data = read_artefact(dir, "vectors.bin", manifest)?;

    let count = match layout {
        Layout::Framed => framed_vectors(&vectors_data, ids, dim)?.len(),
        Layout::Bincode => {
            let (count, offenders) = legacy::walk_vectors(&vectors_data, "vectors.bin", ids.len())?;
            if !offenders.is_empty() {
                return Err(Error::VectorsNotFinite {
                    offenders,
                    total: count,
                });
            }
            count
        }
    };

    debug!(target: LOG_TARGET, "vectors.bin walked ({} vectors, none kept)", count);
    Ok(count)
}

/// A framed vectors.bin, held to the mappings
///
/// The frame holds one vector of `dim` values for every record the mappings
/// hold and for nothing else. A record the mappings do not hold, a record
/// they hold that has no vector, and a vector holding a NaN or an infinity
/// are each refused, the last naming the records as
/// `check_vectors_are_finite` names them.
fn framed_vectors<'a>(
    bytes: &'a [u8],
    ids: &IdStore,
    dim: usize,
) -> Result<framed::Rows<'a>, Error> {
    let rows = framed::Rows::read(
        bytes,
        FrameKind::RawVectors,
        "vectors.bin",
        4,
        dim,
        |width| {
            format!(
                "holds vectors of {} values and config.json declares dim {}",
                width, dim
            )
        },
    )?;
    let corrupt = |error: String| Error::DecodeFailed {
        file: "vectors.bin".to_string(),
        error,
    };
    if let Some((internal_id, _)) = rows
        .iter()
        .find(|&(internal_id, _)| !ids.contains_slot(internal_id))
    {
        return Err(corrupt(format!(
            "names internal id {}, which mappings.bin does not hold",
            internal_id
        )));
    }
    if rows.len() != ids.len() {
        let held: std::collections::HashSet<usize> =
            rows.iter().map(|(internal_id, _)| internal_id).collect();
        let mut without: Vec<&str> = ids
            .iter()
            .filter(|(internal_id, _)| !held.contains(internal_id))
            .map(|(_, id)| id)
            .collect();
        without.sort_unstable();
        return Err(corrupt(format!(
            "holds {} vectors and mappings.bin holds {} records; record '{}' has no vector",
            rows.len(),
            ids.len(),
            without.first().copied().unwrap_or("")
        )));
    }
    let mut offenders: Vec<&str> = rows
        .iter()
        .filter(|(_, row)| framed::floats(row).any(|value| !value.is_finite()))
        .filter_map(|(internal_id, _)| ids.name(internal_id))
        .collect();
    if !offenders.is_empty() {
        offenders.sort_unstable();
        return Err(Error::VectorsNotFinite {
            offenders: offenders.into_iter().map(str::to_string).collect(),
            total: rows.len(),
        });
    }
    Ok(rows)
}

/// Load manifest for validation and metadata
fn load_manifest(dir: &dyn Dir) -> Result<IndexManifest, Error> {
    debug!(target: LOG_TARGET, "Loading manifest.json...");

    // A manifest that is not UTF-8 is refused as a read that failed, in the
    // words the standard library gives a text read of such a file.
    let manifest_data =
        String::from_utf8(dir.read("manifest.json")?).map_err(|_| Error::ArtefactReadFailed {
            name: "manifest.json".to_string(),
            error: not_utf8().to_string(),
        })?;

    let manifest: IndexManifest =
        serde_json::from_str(&manifest_data).map_err(|e| Error::ArtefactParseFailed {
            name: "manifest.json",
            error: e.to_string(),
        })?;

    debug!(target: LOG_TARGET, "manifest.json loaded");
    Ok(manifest)
}

/// What `std::fs::read_to_string` returns for a file that is not UTF-8.
fn not_utf8() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "stream did not contain valid UTF-8",
    )
}

/// Load the PQ codebook from pq_centroids.bin
///
/// Absent means the index was saved before training completed, which is a
/// legitimate state. A present but unreadable file is a hard failure, because
/// the alternative is a codebook that decodes every code to the zero vector.
///
/// The codebook is held to `shape` before anything is allocated for it, and
/// refused in the words the install gives a codebook of another shape.
fn load_pq_centroids(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    layout: Layout,
    shape: framed::Shape,
) -> Result<Option<Centroids>, Error> {
    if !manifest_names(manifest, "pq_centroids.bin") {
        return Ok(None);
    }

    debug!(target: LOG_TARGET, "Loading pq_centroids.bin...");

    let centroids_data = read_artefact(dir, "pq_centroids.bin", manifest)?;

    let centroids: Centroids = match layout {
        Layout::Framed => framed::read_codebook(&centroids_data, "pq_centroids.bin", shape)?,
        Layout::Bincode => legacy::read_codebook(&centroids_data, "pq_centroids.bin", shape)?,
    };

    debug!(target: LOG_TARGET, "pq_centroids.bin loaded ({} subvectors)", centroids.len());
    Ok(Some(centroids))
}

/// Load the quantized codes from pq_codes.bin, keyed by external id
///
/// Absent means no record has been quantized yet. In `quantized_only` these
/// codes are the only copy of every record added after training completed.
/// A framed file names each record by internal id, which `ids` must hold, and
/// each code is `subvectors` bytes.
fn load_pq_codes(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    ids: &IdStore,
    layout: Layout,
    subvectors: usize,
) -> Result<HashMap<String, Vec<u8>>, Error> {
    if !manifest_names(manifest, "pq_codes.bin") {
        return Ok(HashMap::new());
    }

    debug!(target: LOG_TARGET, "Loading pq_codes.bin...");

    let codes_data = read_artefact(dir, "pq_codes.bin", manifest)?;

    let codes = match layout {
        Layout::Framed => {
            let rows = framed::Rows::read(
                &codes_data,
                FrameKind::PqCodes,
                "pq_codes.bin",
                1,
                subvectors,
                |width| {
                    format!(
                        "holds codes of {} bytes and quantization.json declares {} subvectors",
                        width, subvectors
                    )
                },
            )?;
            let mut codes = HashMap::with_capacity(rows.len());
            for (internal_id, code) in rows.iter() {
                let Some(id) = ids.name(internal_id) else {
                    return Err(Error::DecodeFailed {
                        file: "pq_codes.bin".to_string(),
                        error: format!(
                            "names internal id {}, which mappings.bin does not hold",
                            internal_id
                        ),
                    });
                };
                codes.insert(id.to_string(), code.to_vec());
            }
            codes
        }
        Layout::Bincode => legacy::read_codes(&codes_data, "pq_codes.bin", ids.len())?,
    };

    debug!(target: LOG_TARGET, "pq_codes.bin loaded ({} records)", codes.len());
    Ok(codes)
}

/// Hold quantization.json's two sizing fields to the rules `create()` applies
///
/// `bits` fixes the centroid count at `2^bits` and `subvectors` fixes both the
/// outer dimension of the codebook and the divisor `sub_dim` comes from, so the
/// pair sizes every allocation `PQ::new` makes. The Python layer validates both
/// at `create()` and nothing revalidated them on the way back in.
///
/// The bounds are the ones `create()` already applies, so this refuses exactly
/// the configurations `create()` refuses and no more. `bits` is 1 to 8 because
/// a code is one byte a subvector. `subvectors` must be positive, must divide
/// `dim` and must not exceed it, because `sub_dim` is `dim / subvectors` and a
/// subvector of no values encodes nothing.
fn validate_quantization_fields(
    file: &str,
    config: &QuantizationPersistence,
    dim: usize,
) -> Result<(), Error> {
    if config.bits < 1 || config.bits > 8 {
        return Err(Error::BitsOutOfRangeInFile {
            file: file.to_string(),
            bits: config.bits,
        });
    }
    if config.subvectors == 0 {
        return Err(Error::SubvectorsZeroInFile {
            file: file.to_string(),
        });
    }
    if config.subvectors > dim || !dim.is_multiple_of(config.subvectors) {
        return Err(Error::SubvectorsInvalidInFile {
            file: file.to_string(),
            subvectors: config.subvectors,
            dim,
        });
    }
    Ok(())
}

/// Load quantization configuration and the codebook that goes with it
fn load_quantization(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    dim: usize,
    space: &str,
    ids: &IdStore,
    layout: Layout,
) -> Result<Option<QuantizationArtefacts>, Error> {
    debug!(target: LOG_TARGET, "Loading quantization components...");

    let quant_path = dir.locate("quantization.json");
    if !manifest_names(manifest, "quantization.json") {
        debug!(target: LOG_TARGET, "manifest.json does not list quantization.json (non-quantized index)");
        return Ok(None);
    }

    let quant_data = read_artefact_string(dir, "quantization.json", manifest)?;

    // Which of the two layouts the file takes, read off its `type` field
    // before either struct is parsed, so the product quantized struct still
    // parses a product quantized file and reports its errors word for word.
    // A file that is not an object, or names no type, is left to that
    // struct's parse, which refuses it as it always did.
    let is_int8 = serde_json::from_str::<Value>(&quant_data)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(Value::as_str)
                .map(|t| t == "int8")
        })
        .unwrap_or(false);
    if is_int8 {
        return load_int8_quantization(dir, manifest, dim, space, &quant_path, &quant_data)
            .map(Some);
    }

    // A directory whose config.json names the inner product space and whose
    // manifest names quantization.json describes an index `create()` refuses.
    // No save this build makes can produce one, so it was hand assembled, and
    // building it would give an index ranking by the wrong quantity.
    validate_space_supports_quantization(space, &format!("{}: ", dir.locate("")))?;

    let quant_config: QuantizationPersistence =
        serde_json::from_str(&quant_data).map_err(|e| Error::ArtefactParseFailed {
            name: "quantization.json",
            error: e.to_string(),
        })?;

    // The two fields that size the codebook, held to the rules `create()`
    // applies to them.
    //
    // `PQ::new` allocates `subvectors * 2^bits * (dim / subvectors)` floats
    // from these two values and nothing checked either. `bits: 40` asked for
    // 2^40 centroids and **aborted the process**, `subvectors: 2^40` aborted on
    // the outer vector, and `subvectors: 0` divided by zero. `create()` refuses
    // all three, so a directory carrying one was hand edited or written by a
    // release that did not validate its own input, and either way the file does
    // not describe an index this build can rebuild.
    validate_quantization_fields(&quant_path, &quant_config, dim)?;

    debug!(target: LOG_TARGET, "quantization.json loaded");

    let shape = framed::Shape {
        expected: (
            quant_config.subvectors,
            1usize << quant_config.bits,
            dim / quant_config.subvectors,
        ),
        subvectors: quant_config.subvectors,
        bits: quant_config.bits,
    };
    let centroids = load_pq_centroids(dir, manifest, layout, shape)?;
    let codes = load_pq_codes(dir, manifest, ids, layout, quant_config.subvectors)?;

    Ok(Some(QuantizationArtefacts {
        config: QuantizationFile::Pq(quant_config),
        centroids,
        codes,
        int8: None,
        rows: Int8Rows::default(),
    }))
}

/// quantization.json under `type: int8`, with the scales and the rows a
/// trained directory carries beside it.
///
/// The fields are held to the rules `create()` applies to a scalar
/// declaration, and each artefact to the bounds its frame states: the
/// scales artefact carries `dim` entries and exactly `dim * 4` payload
/// bytes, every scale finite and positive, and the rows artefact carries
/// `entries` records of a `u32` internal id and a row of the codec's width,
/// ids strictly increasing and none above the saved id counter, in exactly
/// `entries * (4 + width)` bytes. Every bound is checked before anything is
/// allocated from a field.
fn load_int8_quantization(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    dim: usize,
    space: &str,
    quant_path: &str,
    quant_data: &str,
) -> Result<QuantizationArtefacts, Error> {
    let file = quant_path.to_string();
    let config: Int8Persistence =
        serde_json::from_str(quant_data).map_err(|e| Error::ArtefactParseFailed {
            name: "quantization.json",
            error: e.to_string(),
        })?;
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: file.clone(),
        detail,
    };
    if crate::collection::Int8Scale::from_name(&config.scale).is_none() {
        return Err(invalid(format!(
            "scale is '{}', and this build fits '{}' alone",
            config.scale,
            crate::collection::Int8Scale::PER_DIMENSION
        )));
    }
    if config.training_size < 1000 {
        return Err(invalid(format!(
            "training_size is {}, and create() requires at least 1000",
            config.training_size
        )));
    }
    if let Some(max_training) = config.max_training_vectors {
        if max_training < config.training_size {
            return Err(invalid(format!(
                "max_training_vectors is {} and training_size is {}, and create() requires \
                 the first to be at or above the second",
                max_training, config.training_size
            )));
        }
    }
    let storage_mode = StorageMode::from_string(&config.storage_mode).map_err(Error::Engine)?;
    if storage_mode == StorageMode::QuantizedWithRaw {
        return Err(invalid(
            "storage_mode is 'quantized_with_raw', which a scalar quantized index never takes"
                .to_string(),
        ));
    }
    debug!(target: LOG_TARGET, "quantization.json loaded (int8)");

    if !config.is_trained {
        return Ok(QuantizationArtefacts {
            config: QuantizationFile::Int8(config),
            centroids: None,
            codes: HashMap::new(),
            int8: None,
            rows: Int8Rows::default(),
        });
    }

    let codec = load_int8_scales(dir, manifest, dim)?;
    let tail = if space == "cosine" { 4 } else { 0 };
    let rows = load_int8_rows(dir, manifest, codec.dim() + tail)?;
    Ok(QuantizationArtefacts {
        config: QuantizationFile::Int8(config),
        centroids: None,
        codes: HashMap::new(),
        int8: Some(codec),
        rows,
    })
}

/// The scales artefact, held to its bounds and handed back as the codec.
fn load_int8_scales(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    dim: usize,
) -> Result<Arc<Int8Codec>, Error> {
    if !manifest_names(manifest, INT8_SCALES_FILENAME) {
        return Err(Error::Int8ScalesMissing);
    }
    debug!(target: LOG_TARGET, "Loading {}...", INT8_SCALES_FILENAME);
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: INT8_SCALES_FILENAME.to_string(),
        detail,
    };
    // The bound is the frame plus `dim` floats, since the frame's own header
    // is held to the same count below.
    let max_bytes = (dim as u64)
        .saturating_mul(4)
        .saturating_add(FRAME_OVERHEAD_BYTES as u64);
    let bytes = zeusdb_vector_core::read_artefact(
        dir,
        INT8_SCALES_FILENAME,
        manifest,
        INT8_SCALES_CONTENTS,
        max_bytes,
    )?;
    let framed = unframe(&bytes, FrameKind::Int8Scales, INT8_SCALES_FILENAME)?;
    if framed.entries != dim as u64 {
        return Err(invalid(format!(
            "holds {} scales and config.json declares dim {}",
            framed.entries, dim
        )));
    }
    if framed.payload.len() != dim * 4 {
        return Err(invalid(format!(
            "holds {} payload bytes and {} scales take {}",
            framed.payload.len(),
            dim,
            dim * 4
        )));
    }
    let scales: Vec<f32> = framed
        .payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let codec = Int8Codec::from_scales(scales).map_err(invalid)?;
    debug!(target: LOG_TARGET, "{} loaded ({} scales)", INT8_SCALES_FILENAME, dim);
    Ok(Arc::new(codec))
}

/// The rows artefact, held to its bounds, as (internal id, row) ascending.
/// Empty where the manifest names none, which is what a trained index
/// holding no record writes.
fn load_int8_rows(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    row_width: usize,
) -> Result<Int8Rows, Error> {
    if !manifest_names(manifest, INT8_ROWS_FILENAME) {
        return Ok(Int8Rows::default());
    }
    debug!(target: LOG_TARGET, "Loading {}...", INT8_ROWS_FILENAME);
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: INT8_ROWS_FILENAME.to_string(),
        detail,
    };
    let stride = 4 + row_width;
    // The bound is the id ceiling's worth of rows, since an internal id is
    // a `u32` and every row below is held to that ceiling.
    let max_bytes = (u32::MAX as u64)
        .saturating_mul(stride as u64)
        .saturating_add(FRAME_OVERHEAD_BYTES as u64);
    let bytes = zeusdb_vector_core::read_artefact(
        dir,
        INT8_ROWS_FILENAME,
        manifest,
        INT8_ROWS_CONTENTS,
        max_bytes,
    )?;
    let framed = unframe(&bytes, FrameKind::Int8Rows, INT8_ROWS_FILENAME)?;
    let entries = usize::try_from(framed.entries)
        .ok()
        .filter(|&entries| entries <= u32::MAX as usize)
        .ok_or_else(|| {
            invalid(format!(
                "names {} rows, above the id ceiling",
                framed.entries
            ))
        })?;
    let expected = entries
        .checked_mul(stride)
        .ok_or_else(|| invalid(format!("names {} rows, which overflows", entries)))?;
    if framed.payload.len() != expected {
        return Err(invalid(format!(
            "holds {} payload bytes and {} rows of {} bytes take {}",
            framed.payload.len(),
            entries,
            stride,
            expected
        )));
    }
    let mut rows = Int8Rows::with_capacity(entries, row_width);
    let mut previous: Option<u32> = None;
    for chunk in framed.payload.chunks_exact(stride) {
        let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if previous.is_some_and(|last| id <= last) {
            return Err(invalid(format!(
                "internal id {} follows {}, and the ids are strictly increasing",
                id,
                previous.unwrap_or(0)
            )));
        }
        previous = Some(id);
        rows.push(id, &chunk[4..]);
    }
    debug!(target: LOG_TARGET, "{} loaded ({} rows)", INT8_ROWS_FILENAME, rows.len());
    Ok(rows)
}

// ============================================================================
// MAIN PERSISTENCE INTERFACE
// ============================================================================

/// Write every artefact except the graph dump and the manifest into `dir`
///
/// `dir` is the staging directory `Storage::stage` opened, never the target, so
/// nothing here can leave a half written file where a reader will find it.
///
/// The manifest is no longer written here. It is written last of all, after the
/// graph dump, because it now records a length and a digest per artefact and
/// cannot do that for a file that does not exist yet. That ordering used to be
/// impossible: a save that failed at the dump would have left a directory with
/// no manifest at all. Staging makes it safe, because a save that fails at any
/// point leaves the previous directory untouched and the staging directory
/// removed.
pub(crate) fn save_index(index: &Collection, dir: &dyn Dir) -> Result<SaveLedger, Error> {
    let mut ledger = SaveLedger::default();

    // Save components in order of complexity (simple -> complex)
    save_config(index, dir, &mut ledger)?;
    save_mappings(index, dir, &mut ledger)?;
    save_metadata(index, dir, &mut ledger)?;

    // Save quantization components if enabled
    if index.has_quantization() {
        save_quantization_config(index, dir, &mut ledger)?;

        if index.can_use_quantization() {
            if index.declares_int8() {
                save_int8_scales(index, dir, &mut ledger)?;
                save_int8_rows(index, dir, &mut ledger)?;
            } else {
                save_pq_centroids(index, dir, &mut ledger)?;
                save_pq_codes(index, dir, &mut ledger)?;
            }
        }
    }

    // Save vectors based on storage mode
    save_vectors(index, dir, &mut ledger)?;

    // The sparse space, where one is declared, under `spaces/<name>/`.
    save_spaces(index, dir, &mut ledger)?;

    Ok(ledger)
}

/// Write the sparse space's artefacts under `spaces/<name>/`, where the
/// collection declares one.
///
/// The postings are written by the index itself through the seam, and the
/// dictionary by the collection, since the text layer is the collection's.
/// Each is recorded in the manifest by its length alone, because both are
/// framed and the frame's payload checksum is what the reader verifies; see
/// `zeusdb_vector_core::frame`. The two guards are taken one at a time and
/// never together.
fn save_spaces(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    let Some((name, space)) = index.sparse_named() else {
        return Ok(());
    };
    let prefix = Collection::space_prefix(name);
    debug!(target: LOG_TARGET, "Saving {}postings.zdbsparse...", prefix);
    space.index.read().unwrap().write(&prefix, dir, ledger)?;
    debug!(target: LOG_TARGET, "{}postings.zdbsparse saved", prefix);
    if let Some(text) = &space.text {
        let dictionary_name = Collection::dictionary_name(&prefix);
        debug!(target: LOG_TARGET, "Saving {}...", dictionary_name);
        let bytes = text.dictionary.read().unwrap().encode();
        zeusdb_vector_core::write_artefact(dir, &dictionary_name, &bytes)?;
        ledger.record_digest(&dictionary_name, bytes.len() as u64, None);
        debug!(target: LOG_TARGET, "{} saved ({} bytes)", dictionary_name, bytes.len());
    }
    Ok(())
}

// ============================================================================
// RECONSTRUCTION FUNCTIONS
// ============================================================================

/// Reconstruct Collection using Simple Reconstruction
#[allow(clippy::too_many_arguments)]
fn reconstruct_index_simple(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    config: IndexConfig,
    ids: IdStore,
    layout: Layout,
    metadata: HashMap<String, HashMap<String, Value>>,
    mut vectors: RawVectors,
    quantization: Option<QuantizationArtefacts>,
    sparse: Option<SparseDeclaration>,
    dump_bytes: Option<u64>,
) -> Result<(Collection, bool), Error> {
    debug!(target: LOG_TARGET, "Creating empty index with loaded configuration...");

    // Step 1: Create empty index with loaded config
    let mut index = Collection::new_empty(
        config.dim,
        config.space.clone(),
        config.m,
        config.ef_construction,
        config.expected_size,
        config.indexed_fields.clone(),
        sparse,
    );

    debug!(target: LOG_TARGET, "Restoring data fields...");

    // The codes are needed twice, once to rebuild the graph for records that
    // have no raw vector and once to restore the stored codes afterwards.
    let mut quantization = quantization;
    let pq_codes = quantization
        .as_ref()
        .map(|q| q.codes.clone())
        .unwrap_or_default();
    let training_state = quantization
        .as_ref()
        .map(|q| TrainingState::from(&q.config));
    // The scalar rows, taken out rather than cloned, since the artefact is
    // the largest thing the loader holds after the dump.
    let int8_trained = quantization.as_ref().is_some_and(|q| q.int8.is_some());
    let int8_rows: Int8Rows = quantization
        .as_mut()
        .map(|q| std::mem::take(&mut q.rows))
        .unwrap_or_default();

    // Step 2: Restore all data fields directly (but not the graph)
    restore_data_fields(&mut index, ids, &config, quantization)?;

    // Step 2a: The sparse space, once the mappings are in and before the
    // graph, for the reasons `restore_spaces` gives.
    restore_spaces(&index, dir, manifest)?;

    // The rows of a trained scalar directory, held to the mappings both
    // ways before anything is built from either: every row names a record
    // the mappings hold, and every record the mappings hold has a row. A
    // directory whose rows and mappings disagree describes two indexes.
    if int8_trained {
        check_int8_rows_against_mappings(&index, &int8_rows)?;
    }

    // Step 3: Restore the graph the save wrote. Every save dumps the graph, so
    // the ordinary path is a read. A dump that is absent, damaged, or written
    // by a release this build cannot interpret falls back to the rebuild, which
    // is the path that also upgrades a graph carrying a defect the vendored
    // patches have since fixed. See `restore_graph_from_dump`.
    let graph_rebuilt = match index.restore_graph_from_dump(dir, config.id_counter, dump_bytes) {
        Ok(nodes) => {
            debug!(target: LOG_TARGET, "HNSW graph restored from the saved dump ({} nodes)", nodes);
            false
        }
        Err(reason) => {
            debug!(target: LOG_TARGET, "Rebuilding the HNSW graph, because {}", reason);
            // Both rebuilds below replay the raw vectors, so a directory
            // whose vectors were counted and not kept reads them now.
            {
                let ids = index.ids();
                vectors.hold(dir, manifest, &ids, layout, config.dim)?;
            }
            let empty = HashMap::new();
            let held = vectors.held().unwrap_or(&empty);
            if int8_trained {
                debug!(target: LOG_TARGET, "Rebuilding scalar quantized HNSW graph from the stored rows...");
                let inserted = index
                    .rebuild_graph_from_int8_rows(&int8_rows)
                    .map_err(Error::Engine)?;
                debug!(target: LOG_TARGET, "Scalar quantized graph rebuilt ({} rows inserted)", inserted);
            } else if index.can_use_quantization() {
                debug!(target: LOG_TARGET, "Rebuilding quantized HNSW graph from stored PQ codes...");
                rebuild_graph_from_codes(&mut index, &pq_codes, held)?;
            } else {
                debug!(target: LOG_TARGET, "Rebuilding HNSW graph from vectors...");
                rebuild_graph_from_data(&mut index, held, &pq_codes)?;
            }
            true
        }
    };
    // An empty directory holds no dump to read, since a save of an empty
    // index writes none, so the rebuild that runs over no records is the
    // ordinary path rather than a graph that could not be read.
    let graph_rebuilt = graph_rebuilt && index.vector_count() > 0;

    // Step 4: The stored record data, exactly as it was written, once the
    // graph is back. Neither rebuild touches the storage maps, and the
    // columns are built over the id store, which either rebuild may have added
    // an id to for a record the mappings did not name.
    let raw_count = vectors.len();
    let code_count = pq_codes.len();

    // A trained quantized_only index holds no raw vectors, but a directory
    // written before that was true carries its training records in
    // vectors.bin. They are dropped here rather than restored, so an old
    // directory sheds them on load exactly as a live index sheds them at
    // training. Only a vector whose record also has stored codes is dropped;
    // a raw vector without codes is the record's sole copy, which only a
    // directory that lost pq_codes.bin while keeping vectors.bin can contain,
    // and the count check below is what judges that case. The restored record
    // count is unaffected because every dropped vector's record keeps its
    // codes.
    let quantized_only_trained = index.can_use_quantization()
        && index
            .quantization_config()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedOnly);
    let vectors = if quantized_only_trained {
        let (kept, dropped): (HashMap<_, _>, HashMap<_, _>) = vectors
            .into_held()
            .into_iter()
            .partition(|(id, _)| !pq_codes.contains_key(id));
        if !dropped.is_empty() {
            debug!(target: LOG_TARGET, "Released {} raw training vectors quantized_only no longer keeps",
                dropped.len()
            );
        }
        RawVectors::Held(kept)
    } else {
        vectors
    };
    index.restore_storage_maps(pq_codes, metadata);

    // The raw vectors of a `quantized_with_raw` index go back into the store
    // beside the codes, addressed by the node numbering the restored graph
    // carries. A raw index needs none of this: its raw vectors came back with
    // the graph dump, which is the only place they were written.
    //
    // **Opening the store is not conditional on there being anything to put in
    // it.** This used to also require `!vectors.is_empty()`, so a trained
    // `quantized_with_raw` index holding no records at save time came back
    // without a store at all. It still reported `quantized_with_raw` and
    // `quantized_active`, and every record added after the load lost its raw
    // vector permanently: `get_records` fell through to the PQ reconstruction
    // and the rescoring the mode exists for had nothing true to rescore
    // against. Two ordinary sequences reach it, `clear()` before a save and
    // removing every record before a save.
    //
    // `clear()` already gets this right and opens the store on the replacement
    // graph for exactly this reason. This is the same rule on the load path.
    //
    // `restore_raw_store` handles the empty case itself: it sizes the store
    // from the graph's node count and pushes one vector per node, so at zero
    // nodes it opens an empty store and places nothing.
    if index.raw_store_is_expected() {
        let empty = HashMap::new();
        let placed = index
            .restore_raw_store(vectors.held().unwrap_or(&empty))
            .map_err(Error::RestoreRawFailed)?;
        debug!(target: LOG_TARGET, "{} raw vectors restored beside the codes", placed);
    }

    // Step 5: Put back the training collection the rebuild stripped
    if let Some(state) = training_state {
        state.apply(&mut index);
    }

    // Step 6: Check the saved count against the index that was actually built
    check_restored_count(&mut index, &config, raw_count, code_count)?;

    debug!(target: LOG_TARGET, "Reconstruction completed!");
    Ok((index, graph_rebuilt))
}

/// Reconcile the stored vector count with the records that were restored
///
/// `vector_count` is written to config.json and was previously restored
/// verbatim, so it could report records the directory no longer contains. The
/// count is derived here from the restored data and asserted against the saved
/// value. They agree for every directory whose files are intact, so a
/// disagreement means a file is missing or truncated and the load fails rather
/// than producing an index that misreports what it holds.
/// Hold the rows artefact to the mappings in both directions.
fn check_int8_rows_against_mappings(index: &Collection, rows: &Int8Rows) -> Result<(), Error> {
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: INT8_ROWS_FILENAME.to_string(),
        detail,
    };
    let ids = index.ids();
    if let Some((id, _)) = rows.iter().find(|&(id, _)| !ids.contains_slot(id)) {
        return Err(invalid(format!(
            "names internal id {}, which mappings.bin does not hold",
            id
        )));
    }
    if rows.len() != ids.len() {
        let held: std::collections::HashSet<usize> = rows.iter().map(|(id, _)| id).collect();
        let mut without: Vec<&str> = ids
            .iter()
            .filter(|(id, _)| !held.contains(id))
            .map(|(_, ext)| ext)
            .collect();
        without.sort_unstable();
        return Err(invalid(format!(
            "holds {} rows and mappings.bin holds {} records; record '{}' has no row",
            rows.len(),
            ids.len(),
            without.first().copied().unwrap_or("")
        )));
    }
    debug!(target: LOG_TARGET, "{} agrees with mappings.bin ({} rows)", INT8_ROWS_FILENAME, rows.len());
    Ok(())
}

fn check_restored_count(
    index: &mut Collection,
    config: &IndexConfig,
    raw_count: usize,
    code_count: usize,
) -> Result<(), Error> {
    let restored = index.count_stored_records();

    if restored != config.vector_count {
        return Err(Error::RestoredCountMismatch {
            restored,
            expected: config.vector_count,
            raw_count,
            code_count,
        });
    }

    index.set_vector_count(restored);
    debug!(target: LOG_TARGET, "Vector count verified against restored records: {}",
        restored
    );
    Ok(())
}

/// Restore all data fields to the index (everything except the HNSW graph)
fn restore_data_fields(
    index: &mut Collection,
    ids: IdStore,
    config: &IndexConfig,
    quantization: Option<QuantizationArtefacts>,
) -> Result<(), Error> {
    // Before the store moves, because this reads its ids. The floor is what
    // stops an old directory reissuing a generated id it already holds.
    let generated_floor = Collection::highest_generated_id(ids.iter().map(|(_, id)| id));
    index.set_id_store(ids);

    // The add() method will properly:
    // - Insert vectors into index.vectors
    // - Insert metadata into index.vector_metadata
    // - Update counters correctly
    // - Build the HNSW graph

    // Restore counters
    index.set_counters(config.id_counter, config.vector_count);
    index.set_generated_ids(config.generated_ids, generated_floor);

    // Restore index level metadata. Empty for a directory written before
    // config.json carried the field, which is what those directories held.
    if !config.metadata.is_empty() {
        index.add_metadata(config.metadata.clone().into_iter().collect())?;
        debug!(target: LOG_TARGET, "Index level metadata restored ({} entries)",
            config.metadata.len()
        );
    }

    // Restore quantization state if present
    if let Some(artefacts) = quantization {
        restore_quantization_state_simple(
            index,
            artefacts.config,
            artefacts.centroids,
            artefacts.int8,
        )?;
    }

    debug!(target: LOG_TARGET, "All data fields restored successfully");
    Ok(())
}

/// Install a codebook read from disk into a freshly built PQ instance
///
/// The shape check catches a codebook that belongs to a different index. The
/// all-zero check catches the one written by v0.3.0 through v0.4.1, which never
/// read pq_centroids.bin on load and so re-saved the zero codebook that
/// `PQ::new` starts with. Both fail the load rather than let the index come
/// back reporting itself trained while decoding every code to zeros.
fn install_centroids(pq: &PQ, centroids: Centroids) -> Result<(), Error> {
    let expected = (pq.subvectors(), pq.num_centroids(), pq.sub_dim());
    let actual = (
        centroids.len(),
        centroids.first().map(|s| s.len()).unwrap_or(0),
        centroids
            .first()
            .and_then(|s| s.first())
            .map(|c| c.len())
            .unwrap_or(0),
    );
    let uniform = centroids
        .iter()
        .all(|sub| sub.len() == actual.1 && sub.iter().all(|c| c.len() == actual.2));

    if actual != expected || !uniform {
        return Err(Error::CodebookShapeMismatch {
            actual,
            expected,
            subvectors: pq.subvectors(),
            bits: pq.bits(),
        });
    }

    if centroids
        .iter()
        .all(|sub| sub.iter().all(|c| c.iter().all(|&v| v == 0.0)))
    {
        return Err(Error::CodebookAllZero);
    }

    // Going through set_centroids rather than writing the field rebuilds the
    // symmetric distance table from the codebook that has just been read, so a
    // loaded index can build a graph on real distances exactly as a freshly
    // trained one does.
    pq.set_centroids(centroids).map_err(Error::Engine)
}

/// Restore quantization state (simplified for reconstruction)
fn restore_quantization_state_simple(
    index: &mut Collection,
    file: QuantizationFile,
    centroids: Option<Centroids>,
    int8: Option<Arc<Int8Codec>>,
) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Restoring quantization state...");

    let quant_data = match file {
        QuantizationFile::Pq(quant_data) => quant_data,
        QuantizationFile::Int8(quant_data) => {
            return restore_int8_state(index, quant_data, int8);
        }
    };

    // Convert QuantizationPersistence back to QuantizationConfig
    let storage_mode = StorageMode::from_string(&quant_data.storage_mode).map_err(Error::Engine)?;

    let quant_config = QuantizationConfig {
        scheme: QuantizationScheme::Pq {
            subvectors: quant_data.subvectors,
            bits: quant_data.bits,
        },
        training_size: quant_data.training_size,
        max_training_vectors: quant_data.max_training_vectors,
        storage_mode,
    };

    // Set quantization config
    index.set_quantization_config(Some(quant_config));

    // Restore what training measured about the rerank fetch. `None` here means
    // the directory was written before the calibration existed, and the search
    // falls back to the corpus terms. See `RerankCalibration`.
    index.set_rerank_calibration(quant_data.rerank_calibration);

    // Carried rather than restamped, so a load and a save do not move it. On a
    // directory written before the index held the value this is the save time
    // the old code wrote, which cannot be recovered but does at least stop
    // drifting from here on.
    index.set_training_completed_at(quant_data.training_completed_at);

    // The training ids and the threshold flag are applied after the graph
    // rebuild, which would otherwise strip them. See TrainingState.

    // Every quantized index needs a PQ instance, trained or not. Without one
    // maybe_trigger_training can never fire, so an index saved while still
    // collecting could reach the threshold again and still never train.
    let pq = Arc::new(PQ::new(
        index.dim(),
        quant_data.subvectors,
        quant_data.bits,
        quant_data.training_size,
        quant_data.max_training_vectors,
    ));

    if !quant_data.is_trained {
        index.set_pq(Some(pq));

        debug!(target: LOG_TARGET, "Quantization state restored (untrained, {} collected training IDs)",
            quant_data.training_ids.len()
        );
    } else {
        // The codebook is what makes a trained PQ trained. Without it the
        // instance would report itself trained while holding the zeros that
        // PQ::new starts with, and every reconstruction would return them.
        let centroids = centroids.ok_or(Error::CentroidsMissing)?;
        install_centroids(&pq, centroids)?;

        pq.set_trained(true);
        index.set_pq(Some(pq));

        debug!(target: LOG_TARGET, "Quantization state restored (trained, codebook loaded, {} training IDs)",
            quant_data.training_ids.len()
        );
    }

    Ok(())
}

/// The scalar counterpart of `restore_quantization_state_simple`: the
/// declaration, the stamp, and the codec where the file says it was fitted.
fn restore_int8_state(
    index: &mut Collection,
    quant_data: Int8Persistence,
    int8: Option<Arc<Int8Codec>>,
) -> Result<(), Error> {
    let storage_mode = StorageMode::from_string(&quant_data.storage_mode).map_err(Error::Engine)?;
    let scale = crate::collection::Int8Scale::from_name(&quant_data.scale).ok_or_else(|| {
        Error::Int8ArtefactInvalid {
            file: "quantization.json".to_string(),
            detail: format!("scale is '{}'", quant_data.scale),
        }
    })?;
    index.set_quantization_config(Some(QuantizationConfig {
        scheme: QuantizationScheme::Int8 { scale },
        training_size: quant_data.training_size,
        max_training_vectors: quant_data.max_training_vectors,
        storage_mode,
    }));
    index.set_training_completed_at(quant_data.training_completed_at);
    if quant_data.is_trained {
        let codec = int8.ok_or(Error::Int8ScalesMissing)?;
        index.set_int8_codec(codec)?;
        debug!(target: LOG_TARGET, "Quantization state restored (int8, trained, scales loaded, {} training IDs)",
            quant_data.training_ids.len()
        );
    } else {
        debug!(target: LOG_TARGET, "Quantization state restored (int8, untrained, {} collected training IDs)",
            quant_data.training_ids.len()
        );
    }
    Ok(())
}

/// Rebuild the graph for a trained quantized index from its stored codes
///
/// The saved graph was a PQ graph over the codes, so the rebuild inserts those
/// same codes into a fresh PQ graph rather than reconstructing vectors and
/// replaying them through the raw add() path. The loaded index therefore
/// reports `is_quantized()` true and `quantized_active`, searches through ADC
/// exactly as the saved one did, and never holds a reconstructed vector at
/// full width. The internal ids come from mappings.bin, so no id is reassigned
/// and the counters stay as saved.
fn rebuild_graph_from_codes(
    index: &mut Collection,
    pq_codes: &HashMap<String, Vec<u8>>,
    vectors: &HashMap<String, Vec<f32>>,
) -> Result<(), Error> {
    let (inserted, quantized_from_raw, remapped) = index
        .rebuild_graph_from_codes(pq_codes, vectors)
        .map_err(Error::Engine)?;

    if quantized_from_raw > 0 {
        debug!(target: LOG_TARGET, "{} records had a raw vector and no stored PQ codes and were quantized \
             through the loaded codebook",
            quantized_from_raw
        );
    }
    if remapped > 0 {
        debug!(target: LOG_TARGET, "{} records were missing from mappings.bin and were assigned fresh \
             internal ids",
            remapped
        );
    }
    debug!(target: LOG_TARGET, "Quantized graph rebuilt ({} records inserted from stored PQ codes)",
        inserted
    );
    Ok(())
}

/// Rebuild the graph from the records the directory holds, under the ids the
/// mappings name
///
/// This is the path for an index that is not trained, meaning one saved with no
/// quantization at all or one saved while still collecting training vectors.
/// A record that has a raw vector is replayed from it. A record that has only
/// PQ codes is reconstructed through the codebook, which is what `get_records`
/// already does for the same record while the index is live, so the graph is
/// built at the fidelity the storage mode already delivers rather than losing
/// the record. The codes themselves are restored as stored and are never
/// recomputed from a reconstruction. The metadata is not carried here; step 4
/// of the reconstruction writes it back from the file under the same ids.
fn rebuild_graph_from_data(
    index: &mut Collection,
    vectors: &HashMap<String, Vec<f32>>,
    pq_codes: &HashMap<String, Vec<u8>>,
) -> Result<(), Error> {
    if vectors.is_empty() && pq_codes.is_empty() {
        debug!(target: LOG_TARGET, "No records to rebuild (empty index)");
        return Ok(());
    }

    let mut records: Vec<(String, Vec<f32>)> = Vec::with_capacity(vectors.len() + pq_codes.len());
    let mut reconstructed = 0usize;

    // Every record with a raw vector, replayed from it
    for (ext_id, vector) in vectors.iter() {
        records.push((ext_id.clone(), vector.clone()));
    }

    // Every record that has codes and no raw vector, reconstructed
    let code_only: Vec<&String> = pq_codes
        .keys()
        .filter(|id| !vectors.contains_key(*id))
        .collect();

    if !code_only.is_empty() {
        let pq = index.pq().cloned().ok_or(Error::CodesWithoutCodebook {
            count: code_only.len(),
        })?;

        for ext_id in code_only {
            let codes = &pq_codes[ext_id];
            let vector = pq
                .reconstruct(codes)
                .map_err(|e| Error::ReconstructFailed {
                    id: ext_id.clone(),
                    codes: codes.len(),
                    error: e,
                })?;
            records.push((ext_id.clone(), vector));
            reconstructed += 1;
        }
    }

    debug!(target: LOG_TARGET, "Prepared {} records for the graph rebuild ({} replayed from raw vectors, {} reconstructed from PQ codes)",
        records.len(),
        records.len() - reconstructed,
        reconstructed
    );
    debug!(target: LOG_TARGET, "Rebuilding the graph from the restored records...");

    // The records are owned Rust and go straight into a fresh graph, in
    // ascending internal id, which `rebuild_from_records` orders them into
    // from the restored mappings. This used to build a PyDict holding three
    // PyLists and call add(), which parsed them back into exactly this.
    let inserted = index.rebuild_from_records(records)?;

    debug!(target: LOG_TARGET, "Graph rebuild completed: {} records inserted", inserted);
    debug!(target: LOG_TARGET, "Final vector count: {}", index.vector_count());

    Ok(())
}

// ============================================================================
// LOAD INTERFACE
// ============================================================================

/// Load a Collection from a directory structure (Approach B: Simple Reconstruction)
///
/// Reached through `Collection::load`, which the binding registers as
/// `_load_index`. `VectorDatabase.load(path)` is the documented route and is
/// a one line pass through to that.
pub(crate) fn load_index(
    path: &str,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    policy: JournalPolicy,
) -> Result<(Collection, Recovery), Error> {
    debug!(target: LOG_TARGET, "Starting index load with reconstruction from: {}", path);

    let storage = FsStorage::at(Path::new(path))?;
    load_from(&storage, path, tokenizer, policy)
}

/// The load, from the storage the directory `path` names. `path` is the
/// caller's own words for it, which the refusals and the log quote.
fn load_from(
    storage: &dyn Storage,
    path: &str,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    policy: JournalPolicy,
) -> Result<(Collection, Recovery), Error> {
    // A save killed between its two renames left the whole index beside the
    // target and nothing at it. Recovery from a killed process is a load, so
    // it is put back here, before the directory is looked for. See
    // `Storage::recover`.
    let restored_from_aside = storage.recover()?;

    // Validate directory exists
    if !storage.exists() {
        return Err(Error::IndexDirectoryNotFound {
            path: path.to_string(),
        });
    }

    // Phase 1: Load all ZeusDB components
    debug!(target: LOG_TARGET, "Phase 1: Loading ZeusDB components...");

    let dir = storage.dir();
    let manifest = load_manifest(dir)?;
    let major = check_format_version(&manifest.format_version)?;

    // A manifest below the major that names a journal is a directory
    // assembled by hand, since no release writing that format wrote the
    // field. The same rule the sparse space's declaration takes below.
    if major < JOURNAL_FORMAT_MAJOR && manifest.journal.is_some() {
        return Err(Error::FormatVersionJournal {
            format_version: manifest.format_version.clone(),
        });
    }

    // Before any artefact is read, so a directory missing two files names the
    // first rather than failing on whichever one a partial load reaches.
    check_files_present(dir, &manifest)?;
    debug!(target: LOG_TARGET, "Manifest loaded: {} vectors, format v{}",
        manifest.total_vectors, manifest.format_version
    );

    let config = load_config(dir, &manifest)?;
    debug!(target: LOG_TARGET, "Config loaded: dim={}, space={}", config.dim, config.space);

    // A 1.x directory declares no space, since no release writing 1.x held
    // one, and a 1.x manifest over a config that declares one is a
    // directory assembled by hand.
    if major < 2 && !config.spaces.is_empty() {
        return Err(Error::FormatVersionSpaces {
            format_version: manifest.format_version.clone(),
        });
    }
    let space_record = config.spaces.first();
    let sparse_tokenizer = resolve_tokenizer(space_record, tokenizer)?;
    let sparse = space_record.map(|record| SparseDeclaration {
        name: SpaceName::new(&record.name).expect("validated by load_config"),
        config: record.index.clone(),
        tokenizer: sparse_tokenizer,
    });

    // The id store, built from mappings.bin and held to config.json's
    // counter, and the layout the four binary artefacts take, which the
    // same file decides. The framed vectors.bin and pq_codes.bin name records
    // by internal id and are held to this store as they are read.
    let (ids, layout) = load_ids(dir, &manifest, major, &config)?;
    debug!(target: LOG_TARGET, "Mappings loaded: {} ID mappings", ids.len());

    let metadata = load_metadata(dir, &manifest)?;
    debug!(target: LOG_TARGET, "Metadata loaded: {} records", metadata.len());

    // A raw index takes its vectors back from the graph dump, so its
    // vectors.bin is read for its count and its check and nothing of it is
    // kept. See `RawVectors`.
    let vectors = if manifest_names(&manifest, "quantization.json") {
        RawVectors::Held(load_vectors(dir, &manifest, &ids, layout, config.dim)?)
    } else {
        RawVectors::Counted(count_vectors(dir, &manifest, &ids, layout, config.dim)?)
    };
    debug!(target: LOG_TARGET, "Vectors loaded: {} vectors", vectors.len());

    let quantization = load_quantization(dir, &manifest, config.dim, &config.space, &ids, layout)?;
    if let Some(ref quant) = quantization {
        match &quant.config {
            QuantizationFile::Pq(config) => {
                debug!(target: LOG_TARGET, "Quantization loaded: {} subvectors, trained={}, codebook={}",
                    config.subvectors,
                    config.is_trained,
                    if quant.centroids.is_some() {
                        "present"
                    } else {
                        "absent"
                    }
                );
            }
            QuantizationFile::Int8(config) => {
                debug!(target: LOG_TARGET, "Quantization loaded: int8, scale={}, trained={}, scales={}, rows={}",
                    config.scale,
                    config.is_trained,
                    if quant.int8.is_some() {
                        "present"
                    } else {
                        "absent"
                    },
                    quant.rows.len()
                );
            }
        }
    }

    // The graph dump itself is read inside the reconstruction, which needs the
    // mappings and the codebook first to judge it against.

    // Phase 2: Create empty index and restore state
    debug!(target: LOG_TARGET, "Phase 2: Creating empty index and restoring state...");
    let (mut restored_index, graph_rebuilt) = reconstruct_index_simple(
        dir,
        &manifest,
        config,
        ids,
        layout,
        metadata,
        vectors,
        quantization,
        sparse,
        recorded_dump_length(&manifest, GRAPH_DUMP_FILENAME),
    )?;

    // `new_empty` stamps the load time, because it has nothing better to start
    // from. Until this ran, a save of a loaded index wrote that load time to
    // manifest.json as the creation, so a directory that had been through one
    // load and save claimed to have been created then.
    restored_index.set_created_at(manifest.created_at);

    debug!(target: LOG_TARGET, "Index reconstruction completed successfully!");

    // Phase 3: The journal beside the directory, where the manifest names
    // one. Nothing below runs for a directory saved without one, so such a
    // directory loads exactly as it did.
    let Some(record) = manifest.journal else {
        let recovery = Recovery::unjournaled(restored_from_aside, graph_rebuilt);
        recovery.report(path);
        return Ok((restored_index, recovery));
    };

    let Some(directory_id) = crate::journal::collection_id_from_hex(&record.collection_id) else {
        return Err(Error::JournalManifestInvalid {
            detail: format!(
                "its collection_id is '{}', which is not 32 hexadecimal digits",
                record.collection_id
            ),
        });
    };
    restored_index.set_collection_id(directory_id);
    restored_index.set_journal_sequence(record.sequence);

    let durability = match policy {
        JournalPolicy::CheckpointOnly => {
            warn!(target: LOG_TARGET, operation = "load_checkpoint_only",
                directory = path,
                journal = %record.file,
                sequence = record.sequence,
                "Opened the checkpoint alone; every mutation after the sequence it holds is in \
                 the journal beside it and was not applied"
            );
            let mut recovery = Recovery::unjournaled(restored_from_aside, graph_rebuilt);
            recovery.checkpoint_sequence = record.sequence;
            recovery.report(path);
            return Ok((restored_index, recovery));
        }
        JournalPolicy::Replay(durability) => durability,
    };

    let wal = storage.journal_path();
    if !storage.journal_exists() {
        return Err(Error::JournalMissing {
            directory: path.to_string(),
            file: wal.display().to_string(),
            recorded: record.file.clone(),
            sequence: record.sequence,
        });
    }
    let file = wal.display().to_string();
    let bytes = crate::journal::read_journal_bytes(storage)?;
    let contents = zeusdb_vector_core::read_journal(&bytes, &file)?;
    crate::journal::check_contents(&contents, &file, directory_id, record.sequence)?;

    // Every record above the sequence the checkpoint holds, applied at the
    // values it names. `apply` refuses to run with a sink attached, which is
    // why the sink goes on after this loop and not before it.
    let dim = restored_index.dim();
    let mut replayed = 0usize;
    let mut skipped = 0usize;
    for journal_record in &contents.records {
        if journal_record.sequence <= record.sequence {
            skipped += 1;
            continue;
        }
        let operation = zeusdb_vector_core::Operation::decode(journal_record, dim, &file)?;
        restored_index
            .apply(operation)
            .map_err(|error| Error::JournalReplayFailed {
                file: file.clone(),
                sequence: journal_record.sequence,
                detail: error.to_string(),
            })?;
        replayed += 1;
    }

    // The journal is cut back to its last whole record and reopened for
    // append after it. Where the body is empty the header is restated at the
    // checkpoint's sequence plus one, which completes a truncation a crash
    // left half done.
    let writer = zeusdb_vector_core::JournalWriter::open_for_append(
        storage,
        &contents,
        record.sequence + 1,
    )?;
    restored_index.attach_sink(Box::new(crate::journal::JournalSink::from_writer(
        writer, durability,
    )?));

    let recovery = Recovery {
        checkpoint_sequence: record.sequence,
        first_sequence: contents.header.first_sequence,
        records_in_journal: contents.records.len(),
        replayed,
        skipped,
        damage: contents.damage.clone(),
        good_bytes: contents.good_bytes,
        restored_from_aside,
        graph_rebuilt,
        journaled: true,
    };
    recovery.report(path);
    Ok((restored_index, recovery))
}

// ============================================================================
// INDIVIDUAL COMPONENT SAVERS
// ============================================================================

/// Save index configuration as JSON
fn save_config(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving config.json...");

    let config = IndexConfig {
        dim: index.dim(),
        //space: index.get_space().to_string(),
        space: index.metric().to_string(),
        m: index.m(),
        ef_construction: index.ef_construction(),
        expected_size: index.expected_size(),
        id_counter: index.id_counter(),
        vector_count: index.vector_count(),
        generated_ids: index.generated_ids(),
        metadata: index.all_metadata().into_iter().collect(),
        indexed_fields: index.indexed_fields(),
        spaces: index
            .sparse_named()
            .map(|(name, space)| SpaceRecord {
                name: name.as_str().to_string(),
                kind: SPARSE_KIND.to_string(),
                index: space.config().clone(),
                tokenizer: space.text.as_ref().map(|text| text.tokenizer.config()),
            })
            .into_iter()
            .collect(),
    };

    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| Error::SerializeFailed {
            what: "config",
            error: e.to_string(),
        })?;

    write_artefact(dir, "config.json", config_json.as_bytes(), ledger)?;

    debug!(target: LOG_TARGET, "config.json saved");
    Ok(())
}

/// Save ID mappings, every id once inside its frame
fn save_mappings(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving mappings.bin...");

    // Every record's internal id and id, read out of the id store under its
    // guard, which ends with this block, in the order the store holds them,
    // which is increasing internal id. The frame is the one buffer, so the
    // file is written with nothing held.
    let (mappings_data, mapping_count) = {
        let ids = index.ids();
        (
            framed::write_mappings(ids.iter(), ids.len(), ids.text_bytes()),
            ids.len(),
        )
    };
    let mappings_data = mappings_data?;

    write_framed(dir, "mappings.bin", &mappings_data, ledger)?;

    debug!(target: LOG_TARGET, "mappings.bin saved ({} mappings)", mapping_count);
    Ok(())
}

/// Save vector metadata as JSON for external tool compatibility
fn save_metadata(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving metadata.json...");

    // Both guards end with the serialize, taken in the documented order,
    // the id store before vector_metadata. The file is keyed by external id
    // and the store by internal id, so every record is written under the id
    // the id store holds for it, in increasing internal id order, which is
    // the order the store walks in. `to_string_pretty` returns an owned
    // String, so the file is written with nothing held.
    let (metadata_json, record_count) = {
        let ids = index.ids();
        let vector_metadata = index.vector_metadata();
        let file = MetadataFile {
            records: ids
                .iter()
                .filter_map(|(slot, id)| vector_metadata.get(slot).map(|fields| (id, fields)))
                .collect(),
        };
        let json = serde_json::to_string_pretty(&file);
        (json, file.records.len())
    };
    let metadata_json = metadata_json.map_err(|e| Error::SerializeFailed {
        what: "metadata",
        error: e.to_string(),
    })?;

    write_artefact(dir, "metadata.json", metadata_json.as_bytes(), ledger)?;

    debug!(target: LOG_TARGET, "metadata.json saved ({} records)", record_count);
    Ok(())
}

/// `metadata.json` as it is written: one object per record under its
/// external id, holding the record's fields. The same shape the file has
/// always had, which `load_metadata` reads back into a map of maps.
struct MetadataFile<'a> {
    records: Vec<(&'a str, RecordFields<'a>)>,
}

impl Serialize for MetadataFile<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(
            self.records
                .iter()
                .map(|(id, fields)| (id, FieldsObject(*fields))),
        )
    }
}

/// One record's fields as the JSON object its mapping serialises as, in
/// name order, so two saves of the same records write the same bytes.
struct FieldsObject<'a>(RecordFields<'a>);

impl Serialize for FieldsObject<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut fields: Vec<(&str, &Value)> = self.0.iter().collect();
        fields.sort_unstable_by_key(|&(name, _)| name);
        serializer.collect_map(fields)
    }
}

/// Save quantization configuration and training state
fn save_quantization_config(
    index: &Collection,
    dir: &dyn Dir,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    if let Some(config) = index.quantization_config() {
        debug!(target: LOG_TARGET, "Saving quantization.json...");

        if let Some(scale) = config.scheme.int8_scale() {
            return save_int8_quantization_config(index, config, scale, dir, ledger);
        }
        let (subvectors, bits) = config.scheme.pq_shape().unwrap_or((1, 8));

        // When the codebook was fitted, taken from the index rather than from
        // the clock. This used to be `Utc::now()`, so the field recorded the
        // save and moved every time a trained index was saved again. `None` on
        // an index that never trained, and also on one loaded from a directory
        // written before the index carried the value; see the field on
        // `Collection`.
        let training_completed_at = index.training_completed_at();

        // CAPTURE TRAINING STATE:
        let training_ids = index.training_ids().clone();
        let training_threshold_reached = index.training_threshold_reached();

        let (memory_stats, pq_config) = if let Some(pq) = index.pq() {
            let (memory_mb, total_centroids) = pq.get_memory_stats();

            let memory_stats = MemoryStats {
                centroid_storage_mb: memory_mb,
                compression_ratio: (pq.dim() * 4) as f64 / pq.subvectors() as f64,
                centroids_per_subvector: pq.num_centroids(),
                total_centroids,
            };

            let pq_config = PQConfig {
                dim: pq.dim(),
                sub_dim: pq.sub_dim(),
                num_centroids: pq.num_centroids(),
            };

            (Some(memory_stats), pq_config)
        } else {
            let pq_config = PQConfig {
                dim: index.dim(),
                sub_dim: index.dim() / subvectors,
                num_centroids: 1 << bits,
            };
            (None, pq_config)
        };

        let quant_persistence = QuantizationPersistence {
            r#type: "pq".to_string(),
            subvectors,
            bits,
            training_size: config.training_size,
            max_training_vectors: config.max_training_vectors,
            storage_mode: config.storage_mode.to_string().to_string(),
            is_trained: index.can_use_quantization(),
            training_completed_at,
            memory_stats,
            pq_config,
            training_ids,
            training_threshold_reached,
            rerank_calibration: index.rerank_calibration(),
        };

        let quant_json = serde_json::to_string_pretty(&quant_persistence).map_err(|e| {
            Error::SerializeFailed {
                what: "quantization config",
                error: e.to_string(),
            }
        })?;

        write_artefact(dir, "quantization.json", quant_json.as_bytes(), ledger)?;

        //debug!(target: LOG_TARGET, "quantization.json saved");
        debug!(target: LOG_TARGET, "quantization.json saved with {} training IDs",
            quant_persistence.training_ids.len()
        );
    }
    Ok(())
}

/// quantization.json for a scalar quantized space: the declaration, the
/// training state and the clipped count, and none of the product quantized
/// fields.
fn save_int8_quantization_config(
    index: &Collection,
    config: &QuantizationConfig,
    scale: crate::collection::Int8Scale,
    dir: &dyn Dir,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    // Each guard is taken alone and released before the next, in the
    // declared order: the index guard for the clipped count, then the
    // training ids. A temporary guard inside the struct literal would live
    // to the end of the statement, and the index guard taken after it would
    // invert the order, which the rank registry refuses.
    let saturated_values = index.int8_saturated();
    let training_ids = index.training_ids().clone();
    let persistence = Int8Persistence {
        r#type: "int8".to_string(),
        scale: scale.name().to_string(),
        training_size: config.training_size,
        max_training_vectors: config.max_training_vectors,
        storage_mode: config.storage_mode.to_string().to_string(),
        is_trained: index.can_use_quantization(),
        training_completed_at: index.training_completed_at(),
        training_ids,
        training_threshold_reached: index.training_threshold_reached(),
        saturated_values,
    };
    let quant_json =
        serde_json::to_string_pretty(&persistence).map_err(|e| Error::SerializeFailed {
            what: "quantization config",
            error: e.to_string(),
        })?;
    write_artefact(dir, "quantization.json", quant_json.as_bytes(), ledger)?;
    debug!(target: LOG_TARGET, "quantization.json saved (int8) with {} training IDs",
        persistence.training_ids.len()
    );
    Ok(())
}

/// The scales artefact: `dim` little endian floats under a frame whose
/// entry count is `dim`. Recorded by length alone, as every framed artefact
/// is.
fn save_int8_scales(
    index: &Collection,
    dir: &dyn Dir,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    let Some(codec) = index.int8_codec() else {
        return Ok(());
    };
    debug!(target: LOG_TARGET, "Saving {}...", INT8_SCALES_FILENAME);
    let mut payload = Vec::with_capacity(codec.dim() * 4);
    for scale in codec.scales() {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    let bytes = frame(
        FrameKind::Int8Scales,
        FrameEncoding::Engine,
        codec.dim() as u64,
        &payload,
    );
    zeusdb_vector_core::write_artefact(dir, INT8_SCALES_FILENAME, &bytes)?;
    ledger.record_digest(INT8_SCALES_FILENAME, bytes.len() as u64, None);
    debug!(target: LOG_TARGET, "{} saved ({} scales)", INT8_SCALES_FILENAME, codec.dim());
    Ok(())
}

/// The rows artefact: every live record's internal id and row, ascending,
/// written straight into the frame's buffer. Nothing is written for an index
/// holding no record, and the manifest then names no rows artefact.
fn save_int8_rows(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    let stride = 4 + index.quantization_code_bytes();
    let mut out = frame_begin(
        FrameKind::Int8Rows,
        FrameEncoding::Engine,
        index.len().saturating_mul(stride),
    );
    let entries = index.write_int8_rows(&mut out);
    if entries == 0 {
        return Ok(());
    }
    debug!(target: LOG_TARGET, "Saving {}...", INT8_ROWS_FILENAME);
    let bytes = frame_finish(out, entries as u64);
    zeusdb_vector_core::write_artefact(dir, INT8_ROWS_FILENAME, &bytes)?;
    ledger.record_digest(INT8_ROWS_FILENAME, bytes.len() as u64, None);
    debug!(target: LOG_TARGET, "{} saved ({} rows)", INT8_ROWS_FILENAME, entries);
    Ok(())
}

/// Save PQ centroids for vector reconstruction
fn save_pq_centroids(
    index: &Collection,
    dir: &dyn Dir,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    if let Some(pq) = index.pq() {
        if pq.is_trained() {
            debug!(target: LOG_TARGET, "Saving pq_centroids.bin...");

            // The codebook is framed inside the closure and written outside
            // it, so the lock is held for the copy into the frame alone, which
            // is an owned buffer.
            let centroids_data =
                pq.with_centroids(|centroids| framed::write_codebook(centroids))?;

            write_framed(dir, "pq_centroids.bin", &centroids_data, ledger)?;

            debug!(target: LOG_TARGET, "pq_centroids.bin saved");
        }
    }
    Ok(())
}

/// Save quantized vector codes, each under its record's internal id
fn save_pq_codes(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    let width = index.quantization_code_bytes();
    // Both guards end with the frame, taken in the documented order, the id
    // store before the codes, so the file is written with nothing held. The
    // codes follow increasing internal id, which is the order the store
    // holds its records in.
    let (codes_data, code_count) = {
        let ids = index.ids();
        let pq_codes = index.pq_codes();
        if pq_codes.is_empty() {
            return Ok(());
        }
        debug!(target: LOG_TARGET, "Saving pq_codes.bin...");
        let records = ids.iter().filter_map(|(internal_id, id)| {
            pq_codes.get(id).map(|code| (internal_id, code.as_slice()))
        });
        framed::write_codes(width, records, pq_codes.len())?
    };

    write_framed(dir, "pq_codes.bin", &codes_data, ledger)?;

    debug!(target: LOG_TARGET, "pq_codes.bin saved ({} vectors)", code_count);
    Ok(())
}

/// Save raw vectors based on storage mode configuration
///
/// Every record's internal id and raw vector, read out of the store the graph
/// is addressed against, in increasing internal id, straight into the frame.
///
/// A trained `quantized_only` index writes none, as before, because it holds
/// none.
fn save_vectors(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    if !index.holds_raw_vectors() {
        return Ok(());
    }
    let dim = index.dim();
    let (vectors_data, vector_count) =
        index.with_raw_vectors(|records, held| framed::write_vectors(dim, records, held))?;
    if vector_count == 0 {
        return Ok(());
    }
    debug!(target: LOG_TARGET, "Saving vectors.bin...");

    write_framed(dir, "vectors.bin", &vectors_data, ledger)?;

    debug!(target: LOG_TARGET, "vectors.bin saved ({} vectors)", vector_count);
    Ok(())
}

/// Write manifest.json, which is the last file a save writes
///
/// It names every other artefact, records the length and digest of each, and
/// records the directory size. All three are facts about files that are already
/// on disk, which is why it is written last and why there is no second pass to
/// correct any of them.
pub(crate) fn save_manifest(
    index: &Collection,
    dir: &dyn Dir,
    ledger: SaveLedger,
) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving manifest.json...");

    // The manifest needs two facts about the stores rather than the stores
    // themselves, so it takes those two facts and releases both guards here.
    //
    // This used to hold the `vectors` and `pq_codes` read guards for the whole
    // function, and `get_storage_mode` below takes the graph's read guard. The
    // documented order is `hnsw < vectors < pq_codes`, so that acquired the
    // three in exactly the wrong order. It could not deadlock, because a save
    // holds the mutation lock and every path that takes the graph's write guard
    // holds it too, so no counterparty could be in flight. That is the same
    // reasoning that failed the three inversions found before this one.
    let has_raw_vectors = index.holds_raw_vectors() && index.vector_count() > 0;
    let code_count = index.pq_codes().len();

    // Determine what files are included based on what we actually saved
    let mut files_included = vec![
        "config.json".to_string(),
        "mappings.bin".to_string(),
        "metadata.json".to_string(),
    ];

    let mut files_excluded = Vec::new();

    // Add quantization files if they exist
    let int8 = index.declares_int8();
    if index.has_quantization() {
        files_included.push("quantization.json".to_string());

        if index.can_use_quantization() {
            if int8 {
                files_included.push(INT8_SCALES_FILENAME.to_string());
                if index.vector_count() > 0 {
                    files_included.push(INT8_ROWS_FILENAME.to_string());
                }
            } else {
                files_included.push("pq_centroids.bin".to_string());
                if code_count > 0 {
                    files_included.push("pq_codes.bin".to_string());
                }
            }
        }
    }

    // Add vectors.bin if it was saved
    if has_raw_vectors {
        files_included.push("vectors.bin".to_string());
    } else {
        files_excluded.push("vectors.bin".to_string());
    }

    // The sparse space's artefacts, where one is declared. Always written,
    // an empty space included, so the directory's shape says the space
    // exists.
    files_included.extend(index.space_artefact_names());

    // Phase 2: Add the HNSW graph file
    //
    // One file where there used to be two. The vendored format split the
    // topology from the points so the points could be memory mapped, which this
    // build never asked for, and the split meant the two halves could disagree
    // with each other. ZeusDB's format carries both, so the pair is now one
    // length check rather than a cross file comparison.
    let vector_count = index.vector_count();
    if vector_count > 0 {
        files_included.push(GRAPH_DUMP_FILENAME.to_string());
        debug!(target: LOG_TARGET, "Graph file in manifest:");
        debug!(target: LOG_TARGET, "Included: {}", GRAPH_DUMP_FILENAME);
    } else {
        files_excluded.push(GRAPH_DUMP_FILENAME.to_string());
        debug!(target: LOG_TARGET, "No graph file (empty index)");
    }

    // Calculate compression info for quantized indexes
    //
    // Both sizes are taken over the coded records, so the ratio is the size of
    // a code against the size of the vector it stands for. `original_size_mb`
    // used to count the raw vectors the index still holds, which under
    // quantized_only is only the training records. That put a record count in
    // the numerator and a different one in the denominator, and the ratio came
    // out as the compression ratio scaled by the share of records collected
    // before training. At 1,000 training records in 3,000 it read 10.7x where
    // the codes are 32x smaller than the vectors. Under quantized_with_raw the
    // two counts were already equal, so this changes nothing there.
    // A scalar index holds one row a live record, so the coded count is the
    // live count once it is trained.
    let code_count = if int8 && index.can_use_quantization() {
        vector_count
    } else {
        code_count
    };
    let compression_info =
        if index.has_quantization() && index.can_use_quantization() && code_count > 0 {
            let raw_size_mb = (code_count * index.dim() * 4) as f64 / (1024.0 * 1024.0);
            let compressed_size_mb =
                (code_count * index.quantization_code_bytes()) as f64 / (1024.0 * 1024.0);
            let compression_ratio = if compressed_size_mb > 0.0 {
                raw_size_mb / compressed_size_mb
            } else {
                1.0
            };

            Some(CompressionInfo {
                original_size_mb: raw_size_mb,
                compressed_size_mb,
                compression_ratio,
            })
        } else {
            None
        };

    // Every artefact, the graph dump included, because they are all on disk by
    // the time this runs. manifest.json itself is not, so the figure counts the
    // directory without it, which the field's own comment states.
    let total_size_mb = calculate_directory_size(dir).unwrap_or(0.0);

    // The journal, where the collection holds one. Its two names come from
    // the sink itself and the sequence from the collection, which the
    // checkpoint wrote there after it synced, so all three describe the
    // records the directory being written already holds.
    let journal = index
        .sink_journal_names()
        .map(|(file, collection_id)| JournalManifest {
            file,
            sequence: index.journal_sequence(),
            collection_id: crate::journal::collection_id_hex(collection_id),
        });

    let manifest = IndexManifest {
        format_version: FORMAT_VERSION.to_string(),
        zeusdb_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: index.created_at(),
        saved_at: Utc::now().to_rfc3339(),
        total_vectors: vector_count,
        index_type: "HNSW".to_string(),
        has_quantization: index.has_quantization(),
        quantization_trained: index.can_use_quantization(),
        storage_mode: index.storage_mode(),
        files_included,
        files_excluded,
        file_digests: ledger.digests,
        total_size_mb,
        compression_info,
        journal,
    };

    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| Error::SerializeFailed {
            what: "manifest",
            error: e.to_string(),
        })?;

    // Through the same writer every other artefact took, so it is fsynced
    // before the staging directory is moved into place. Its own digest is
    // discarded, since nothing can verify the file that carries the digests.
    let mut discard = SaveLedger::default();
    write_artefact(dir, "manifest.json", manifest_json.as_bytes(), &mut discard)?;

    debug!(target: LOG_TARGET, "manifest.json saved");
    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate the total size of a directory in MB, the artefacts under
/// `spaces/` included.
fn calculate_directory_size(dir: &dyn Dir) -> Result<f64, std::io::Error> {
    Ok(dir.total_bytes()? as f64 / (1024.0 * 1024.0))
}

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

/// Check if a path contains a valid ZeusDB index
///
/// Reserved surface. The body is a placeholder that reports every path invalid
/// and must be implemented before any caller is wired up, including the module
/// registration in lib.rs. The allow keeps the reservation visible instead of
/// silencing dead code across the module.
#[allow(dead_code)]
pub(crate) fn is_valid_index(_path: &str) -> bool {
    false
}

/// Get index information without full loading
///
/// Reserved surface. The body is a placeholder that reports no manifest for
/// every path and must be implemented before any caller is wired up.
#[allow(dead_code)]
pub(crate) fn get_index_info(_path: &str) -> Option<IndexManifest> {
    None
}

#[cfg(test)]
mod tests;
