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
//! ├── metadata.json           # the fields of every record that holds one (JSON)
//! ├── vectors.bin             # raw vectors beside a trained PQ graph under quantized_with_raw
//! ├── quantization.json       # quantization configuration and training state (if enabled)
//! ├── pq_centroids.bin        # PQ centroids (if trained)
//! ├── int8_scales.zdbint8     # the scales of a scalar quantized index, one a dimension (if trained)
//! ├── hnsw_index.zdbgraph     # the graph and the stored vector or code of every record
//! └── spaces/                 # one directory per sparse space, where one is declared
//!     └── <name>/
//!         ├── postings.zdbsparse  # every live record's term ids and weights
//!         └── terms.zdbdict       # the term dictionary, where the space takes text
//!
//! my_index.zdb.zdbjournal/    # the journal's segments, where the collection holds one
//! my_index.zdb.zdblock        # the lock, created by the first lock taken on the directory
//! ```
//!
//! ## The format version and the features
//!
//! Every directory this build saves declares `5.0.0` and lists the features
//! it holds under `features`, each with the mark a build that does not know
//! it reads. See `features` for what a feature is and what each mark means.
//!
//! The version is a major for the base every 5.x directory shares and for the
//! manifest's own shape, `features` included, which a reader must know before
//! it can read a feature. 5.0.0 is a major because its journal is a directory
//! of segments, each naming the checkpoint its records follow, and every
//! call's records are closed by a commit record, which a release reading 4.x
//! alone cannot read, and because its graph dump is version 2 and holds the
//! only copy of each record's row. A 5.x directory holds no `pq_codes.bin`
//! and no `int8_rows.zdbint8`, and holds `vectors.bin` only beside a trained
//! product quantized graph under `quantized_with_raw`. A release reading 1.x
//! to 4.x alone refuses such a directory at its version check, with a
//! message naming the newer release. Every field of a 5.x manifest is held
//! to what a save writes, and `metadata.json` names only the records that
//! hold a field, in increasing internal id.
//! A 5.x manifest lists its features and records its identity with the
//! journal sequence its state holds, and one missing either is refused.
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
//! its `quantization.json` lacks. `4.0.0` framed the four binary artefacts,
//! `mappings.bin`, `vectors.bin`, `pq_codes.bin` and `pq_centroids.bin`,
//! inside the frame `zeusdb_vector_core::frame` describes, where every
//! earlier release wrote them in bincode's wire, and listed features for the
//! first time. `framed` holds the four payloads. No directory below 4.0.0
//! lists features, and what one holds is read from its version and its
//! files, as it was.
//!
//! A journaled directory below 5.0.0 has its journal in one file,
//! `<name>.zdbwal`, at journal format version 1. A load replays it and then
//! upgrades the directory: a checkpoint at 5.0.0 with a journal of segments,
//! after which the file goes. Releases before 0.14.0 do not open the
//! directory afterwards.
//!
//! The four binary artefacts of a directory are read in one layout, decided
//! from `mappings.bin`, which every directory holds. At 4.x it is the frame.
//! Below 4.x it is the frame where `mappings.bin` is a whole frame of its
//! kind, since a frame names what it holds and verifies its own bytes, and
//! bincode's wire otherwise, which `legacy` reads by hand.
//!
//! ## The journal
//!
//! A journaled collection records every mutation to `<name>.zdbjournal`, a
//! directory of segments beside the directory rather than inside it, and a
//! save is the checkpoint that journal replays onto. `manifest.json` names
//! the journal, the collection id both carry and the sequence the checkpoint
//! holds, and every segment names the checkpoint it follows, so the two are
//! paired by content. A load replays every committed record above that
//! sequence before it hands the collection back. See `crate::journal` for
//! where the journal lives and why, for the lock, and for what is refused.
//!
//! ## Identity
//!
//! `manifest.json` records what the directory is under `identity`: the
//! collection it belongs to, its generation, being the saves that
//! collection has committed with this one, the snapshot it is, drawn at
//! this save, and the snapshot it was saved from where there was one. A
//! load takes all four back, so a directory loaded and saved keeps its
//! collection and moves one generation on, and a copy left behind by a
//! later save is the snapshot the later directory names as its parent. A
//! journaled directory's identity and journal record name one collection,
//! which is the one its journal's header names. A directory saved before
//! the record existed opens as it did, under the id its journal record
//! names or the one drawn at assembly, and its next save records it. See
//! `crate::Identity`.
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
//! re-inserting every record. See `Collection::restore_graph_from_dump`. At
//! 5.x the dump is version 2. A failure of its header, its identity region
//! or its rows region refuses the load with `Error::GraphDumpInvalid`, and a
//! failure of its topology rebuilds the graph from the rows it holds. Below
//! 5.x it is version 1, beside the separate row stores, and any failure
//! rebuilds the graph from those.
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
    validate_index_parameters, validate_space_supports_quantization, Collection, DumpOutcome,
    IdentityRecord, QuantizationConfig, QuantizationScheme, SparseDeclaration, StorageMode,
    DEFAULT_SPACE,
};
use crate::journal::{
    CheckpointTaken, Damage, Durability, JournalPolicy, JournalSink, ReadOnlyCause, ReadOnlySink,
    Recovery,
};
use crate::RerankCalibration;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, warn};
use zeusdb_vector_core::{
    checksum_of, frame, read_legacy_journal, read_segments, read_segments_for_repair,
    refuse_unnamable, refuses_writes, unframe, unknown_kind_at, validate_indexed_fields,
    ArtefactRecord, Bounds, Cut, Dir, DumpRows, Error, Expected, FrameEncoding, FrameKind,
    FsStorage, IdStore, Int8Codec, Inventory, JournalDamage, JournalRecord, JournalWriter,
    LiveRecord, LockFailure, LockGuard, LockMode, Locked, MetadataStore, Operation, Position,
    PositionWalk, RecordFields, RecordState, Restore, RowValues, SegmentBase, SegmentBytes,
    SpaceName, Storage, VectorIndex, DUMP_FILENAME as GRAPH_DUMP_FILENAME, FRAME_OVERHEAD_BYTES,
    LEGACY_DUMP_FILENAMES, MAX_INTERNAL_ID, PQ,
};
use zeusdb_vector_sparse::{PostingsIndex, SparseConfig};
use zeusdb_vector_text::{SimpleTokenizer, TermDictionary, Tokenizer, TokenizerConfig};

mod features;
mod framed;
mod legacy;
#[cfg(test)]
pub(crate) mod older;
mod repair;

/// The target every record this file emits carries. It is the module path
/// this file had in the binding, so a filter directive naming it still
/// matches. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::persistence";

// ============================================================================
// FORMAT VERSION
// ============================================================================

/// Version written into manifest.json by every save.
///
/// A major, because the journal is a directory of segments closed by commit
/// records and the graph dump holds the only copy of each record's row; see
/// the module documentation. A feature does not move it, since a directory
/// lists what it holds under `features`.
const FORMAT_VERSION: &str = "5.0.0";

/// The first major whose four binary artefacts are framed.
const FRAMED_FORMAT_MAJOR: u32 = 4;

/// The first major whose saves list their features. Below it no release
/// listed them, and the version decides what a directory holds.
const FEATURES_FORMAT_MAJOR: u32 = 4;

/// The first major whose journal is a directory of segments, and whose
/// manifest must list its features and record its identity with the
/// journal sequence its state holds. Below it a journal is one file at
/// journal format version 1, which a load replays and upgrades.
const SEGMENTED_FORMAT_MAJOR: u32 = 5;

/// The first major whose graph dump is version 2 and holds the only copy of
/// each record's row, whose manifest's fields are each held to what a save
/// writes, and whose `metadata.json` is compact.
const DUMP_V2_FORMAT_MAJOR: u32 = 5;

/// Bytes in a mebibyte, which `total_size_mb` counts in.
const MEBIBYTE: f64 = 1024.0 * 1024.0;

/// The storage modes a manifest records, being every value
/// `Collection::storage_mode` gives.
const STORAGE_MODES: [&str; 5] = [
    "raw_only",
    "raw_collecting_for_training",
    "raw_ready_for_training",
    "quantized_active",
    "raw_trained_not_rebuilt",
];

/// The two artefacts of a trained scalar quantized index.
const INT8_SCALES_FILENAME: &str = "int8_scales.zdbint8";
const INT8_ROWS_FILENAME: &str = "int8_rows.zdbint8";
const INT8_SCALES_CONTENTS: &str =
    "the scales of a scalar quantized index, one a dimension, which every stored row decodes through";
const INT8_ROWS_CONTENTS: &str = "the scalar row of every record";

/// The majors this build reads.
///
/// A minor bump is additive by construction, so any 1.x, any 2.x, any 3.x,
/// any 4.x and any 5.x is read. A different major means the layout changed
/// in a way this build cannot reason about, and guessing at it would be the
/// silent truncation this format has already suffered once.
const SUPPORTED_FORMAT_MAJORS: [u32; 5] = [1, 2, 3, 4, 5];

/// The majors this build reads, as a refusal spells them.
const SUPPORTED_FORMAT_LABEL: &str = "1.x, 2.x, 3.x, 4.x and 5.x";

/// The first major whose manifest may name a journal.
const JOURNAL_FORMAT_MAJOR: u32 = 3;

/// The first major whose releases refuse an internal id before they issue
/// it. Below it a release issued an id first, so one whose id store was full
/// issued ids past the store's last slot, refused each at install, and moved
/// its counter on with each.
const ID_CEILING_FORMAT_MAJOR: u32 = 4;

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
/// the disk. It carries a checksum over its own header and one over each of
/// its regions, verified on every load, and a frame carries one over its
/// header and one over its payload, verified by `unframe` on every load, so a
/// manifest digest would duplicate a check the loader already makes. Their
/// lengths are recorded and checked.
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
/// The dump is read by `graph::dump` as a stream rather than through
/// `read_artefact`, because it is not held whole in memory. This is the part
/// of the digest check that still applies to it.
pub(crate) fn recorded_dump_length(manifest: &IndexManifest, name: &str) -> Option<u64> {
    manifest.file_digests.get(name).map(|entry| entry.bytes)
}

/// Refuse a directory this build cannot interpret, and return the major it
/// declares.
///
/// A version is read only as `MAJOR.MINOR.PATCH`: three runs of ASCII digits
/// joined by dots, with no sign, no leading zero on a part of more than one
/// digit, and nothing before or after. Any other string is refused, as is a
/// major too large for a `u32`.
fn check_format_version(format_version: &str) -> Result<u32, Error> {
    let unparsable = || Error::FormatVersionUnparsable {
        format_version: format_version.to_string(),
        current: FORMAT_VERSION,
    };
    let parts: Vec<&str> = format_version.split('.').collect();
    let &[major, minor, patch] = parts.as_slice() else {
        return Err(unparsable());
    };
    if ![major, minor, patch].into_iter().all(is_version_part) {
        return Err(unparsable());
    }
    let major = major.parse::<u32>().map_err(|_| unparsable())?;

    if !SUPPORTED_FORMAT_MAJORS.contains(&major) {
        return Err(Error::FormatVersionUnsupported {
            format_version: format_version.to_string(),
            supported: SUPPORTED_FORMAT_LABEL,
            newer: major > SUPPORTED_FORMAT_MAJORS[SUPPORTED_FORMAT_MAJORS.len() - 1],
        });
    }

    Ok(major)
}

/// Whether `part` is one part of a format version: a run of ASCII digits that
/// starts with `0` only when it is `0` alone.
fn is_version_part(part: &str) -> bool {
    !part.is_empty()
        && part.bytes().all(|byte| byte.is_ascii_digit())
        && (part == "0" || !part.starts_with('0'))
}

// ============================================================================
// DIRECTORY COMPLETENESS
// ============================================================================

/// Whether an artefact is one the loader can produce again rather than read,
/// in a directory declaring `major`
///
/// Below 5.x the graph is the only one. Every record carries what the graph
/// is built from in the separate row stores, so a directory that lost its
/// dump is rebuilt rather than refused. The list holds the name this build
/// writes and the pair 0.6.0 and earlier wrote, because a directory saved by
/// one of those names both of them under `files_included` and neither is
/// needed to reopen it, and a directory saved before the manifest was
/// written last can name a dump that was never written.
///
/// At 5.x nothing is: the dump holds the only copy of each record's row, so
/// a directory that lost it lost its records.
fn is_derived_artefact(name: &str, major: u32) -> bool {
    major < DUMP_V2_FORMAT_MAJOR
        && (name == GRAPH_DUMP_FILENAME || LEGACY_DUMP_FILENAMES.contains(&name))
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
        GRAPH_DUMP_FILENAME => "the graph and the stored vector or code of every record",
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
fn check_files_present(dir: &dyn Dir, manifest: &IndexManifest, major: u32) -> Result<(), Error> {
    let missing: Vec<&str> = manifest
        .files_included
        .iter()
        .map(String::as_str)
        .filter(|name| !is_derived_artefact(name, major))
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
/// The graph dump is not gated this way. It is read wherever the collection
/// holds a record, and its own checks hold it to `config.json` and the
/// mappings.
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

    /// The features the directory holds, each with the mark a build that
    /// does not know it reads: `compatible` where such a build may open the
    /// directory without it, `incompatible` where it may not. See
    /// `features`.
    ///
    /// Written by every save, in name order. Absent from a directory saved
    /// before the record existed, which opens as it did and lists its
    /// features at its next save. serde ignores a field it does not know, so
    /// a build that predates the record opens a directory carrying it, and a
    /// save there drops it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) features: Option<BTreeMap<String, String>>,

    /// What the directory is: its collection, its generation, its snapshot
    /// and the snapshot it was saved from. See `IdentityManifest`.
    ///
    /// Written by every save. Absent from a directory saved before the
    /// record existed, which opens as it did and gains it at its next save.
    /// serde ignores a field it does not know, so a build that predates the
    /// record opens a directory carrying it, and a save there drops it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) identity: Option<IdentityManifest>,

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

    /// The lengths `file_digests` records, added and divided by a mebibyte,
    /// being every artefact the save wrote. `manifest.json` itself is the
    /// last file a save writes and counts in none.
    pub(crate) total_size_mb: f64,

    /// The journal beside the directory, where the collection that saved it
    /// held one.
    ///
    /// Absent from the file where none was held. Present, it names the
    /// journal a load replays, and `features` lists `journal`. A release
    /// reading 1.x and 2.x alone would ignore it and open the checkpoint
    /// without the journal's mutations and without a word, which the 3.x
    /// major stopped and the 4.x major still stops; see the module
    /// documentation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) journal: Option<JournalManifest>,
}

/// How `manifest.json` names the journal beside the directory.
///
/// The three values are what pair a directory with a journal that is not in
/// it, under the names and types every release from 0.11.0 parses before its
/// version check. `collection_id` is drawn when a collection is built and is
/// in every segment's header too, so a journal from another index is refused
/// by content. `sequence` is what the journal had reached when this
/// checkpoint was taken, so a recovery replays every record above it and
/// skips every record at or below it, and at 5.x it equals the identity's
/// sequence. `file` is the journal's name as the checkpoint wrote it, which
/// a refusal quotes; the journal a load actually opens is derived from the
/// directory it was handed, so a directory renamed together with its journal
/// opens under the new name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct JournalManifest {
    pub(crate) file: String,
    pub(crate) sequence: u64,
    pub(crate) collection_id: String,
}

/// How `manifest.json` names what the directory is. See `crate::Identity`.
///
/// Every id is 32 hexadecimal digits, the spelling `journal::id_hex`
/// writes. `generation` is 1 or more, since a collection's first save
/// writes 1. `parent` is absent from the file where the collection was
/// saved from no recorded snapshot. `sequence` is the journal sequence the
/// directory's state holds, 0 where the collection was never journaled,
/// which every 5.x manifest records and no earlier one does.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct IdentityManifest {
    pub(crate) collection_id: String,
    pub(crate) generation: u64,
    pub(crate) snapshot: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) sequence: Option<u64>,
}

impl IdentityManifest {
    /// The record a save writes for `identity`, which carries the snapshot
    /// the save drew, with the journal sequence the saved state holds.
    fn of(identity: &crate::Identity, sequence: u64) -> Self {
        IdentityManifest {
            collection_id: crate::journal::id_hex(identity.collection_id),
            generation: identity.generation,
            snapshot: crate::journal::id_hex(
                identity
                    .snapshot
                    .expect("the identity a save records carries the snapshot it drew"),
            ),
            parent: identity.parent.map(crate::journal::id_hex),
            sequence: Some(sequence),
        }
    }

    /// The identity this record names, held to the spelling and the range
    /// a save writes.
    fn read(&self) -> Result<crate::Identity, Error> {
        let id = |field: &str, value: &str| {
            crate::journal::id_from_hex(value).ok_or_else(|| Error::IdentityInvalid {
                detail: format!(
                    "its {} is '{}', which is not 32 hexadecimal digits",
                    field, value
                ),
            })
        };
        let collection_id = id("collection_id", &self.collection_id)?;
        let snapshot = id("snapshot", &self.snapshot)?;
        let parent = match self.parent.as_deref() {
            Some(parent) => Some(id("parent", parent)?),
            None => None,
        };
        if self.generation == 0 {
            return Err(Error::IdentityInvalid {
                detail: "its generation is 0, and a collection's first save records 1".to_string(),
            });
        }
        Ok(crate::Identity {
            collection_id,
            generation: self.generation,
            snapshot: Some(snapshot),
            parent,
        })
    }
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
    /// and the 4.x major still stops; see the module documentation. Present,
    /// `features` lists `sparse`, and `text` where the space declares a
    /// tokenizer.
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

/// quantization.json under `type: int8`: the scalar declaration and the
/// training state, and none of the product quantized fields.
///
/// A reader that knows the product quantized layout alone refuses this file
/// on its first missing field, which is what let the format version move by
/// a minor rather than a major before directories listed their features; see
/// the module documentation. A directory that lists them names `int8`, so a
/// build that does not know scalar quantization refuses it by that name
/// before this file is read.
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

    fn push_row(&mut self, id: u32, row: &[i8]) {
        self.ids.push(id);
        self.data.extend_from_slice(row);
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

/// Load index configuration from config.json, written at the format whose
/// major is `major`
fn load_config(dir: &dyn Dir, manifest: &IndexManifest, major: u32) -> Result<IndexConfig, Error> {
    debug!(target: LOG_TARGET, "Loading config.json...");

    let config_path = dir.locate("config.json");
    let config_data = read_artefact_string(dir, "config.json", manifest)?;

    let mut config: IndexConfig =
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
    // `id_counter` too, which those five do not cover. It is the last internal
    // id the index issued, and every internal id the load reads is held to it,
    // the mappings' ids in `load_ids` and the graph dump's origin ids in
    // `graph::dump::open_dump`, so bounding it here bounds both. No structure
    // is sized by it, since every per-record structure sits at the positions
    // the records hold.
    //
    // A release before 0.12.0, which wrote a format below 4.0.0, issued an id
    // before its id store refused it, so one whose store was full issued
    // `u32::MAX` and larger ids, refused each at install and moved the counter
    // on with each. Below that format a counter past the last id a collection
    // issues is one such a release left, and it is held to that id. The ids
    // the mappings name are refused past it as they are read, as every
    // directory's are, and the collection refuses every add, as one at its
    // last id does.
    //
    // From that format a release refuses an id before issuing it, so a
    // counter past `u32::MAX` was written by no release, and it is refused.
    if major < ID_CEILING_FORMAT_MAJOR && config.id_counter > MAX_INTERNAL_ID {
        debug!(target: LOG_TARGET, "config.json records id_counter {}, past the last internal id a collection issues, which a release before 0.12.0 left; held to {}",
            config.id_counter, MAX_INTERNAL_ID
        );
        config.id_counter = MAX_INTERNAL_ID;
    }
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
/// reasons. The mappings give the position each record the space holds is
/// filed at, and the graph's rebuild fallback replays every record through
/// `add`, which reads each record's sparse half out of the space and carries
/// it through; see `Collection::rebuild_from_records`.
///
/// The space's slot table is bounded by `issued`, the last internal id
/// `config.json` says the collection issued, as the graph dump's origin ids
/// are. The space records one past the largest id it ever took a record
/// under, and a removal does not lower that, so the largest id the mappings
/// hold would refuse a directory saved after the newest record went.
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
    issued: usize,
    positions: &[u32],
) -> Result<(), Error> {
    let Some((name, space)) = index.sparse_named() else {
        return Ok(());
    };
    let prefix = Collection::space_prefix(name);
    let bounds = Bounds {
        min_records: 0,
        max_records: issued,
        max_bytes: u64::MAX,
        ids_by_position: positions,
    };
    // The space files each record at the position the mappings give its
    // internal id, and refuses one they do not hold once its payload has
    // been read whole, or one its forward arena cannot take, each named here
    // with the space.
    let restored = PostingsIndex::restore(space.config(), &prefix, dir, manifest, &bounds)
        .map_err(|error| index.sparse_refusal_named(error))?;
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

/// The records `mappings.bin` holds: the id store, which files them at
/// consecutive positions from zero in increasing internal id, and the
/// internal id of the record at each position, which every other artefact
/// names a record by.
///
/// The load keeps the second until it ends, four bytes a record, and finds
/// each record an artefact names by a step forward through it, since every
/// framed artefact lists its records in increasing internal id. Positions are
/// never written, so nothing here outlives the load.
pub(crate) struct MappedIds {
    pub(crate) store: IdStore,
    pub(crate) internal: Vec<u32>,
}

impl MappedIds {
    /// The two halves together, as the readers below take them.
    pub(crate) fn view(&self) -> Mapped<'_> {
        Mapped {
            store: &self.store,
            internal: &self.internal,
        }
    }

    /// How many records the mappings hold.
    pub(crate) fn len(&self) -> usize {
        self.store.len()
    }

    /// Every record as its internal id and its external id, in increasing
    /// internal id.
    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &str)> + '_ {
        self.view().iter()
    }

    /// The internal id of the record named `id`, where the mappings hold it.
    #[cfg(test)]
    pub(crate) fn slot_of(&self, id: &str) -> Option<usize> {
        self.store.slot_of(id).map(|at| self.internal[at] as usize)
    }
}

/// The records the mappings hold, borrowed, which the load reads every other
/// artefact against: the id store by position and the internal id at each
/// position. See [`MappedIds`].
#[derive(Clone, Copy)]
pub(crate) struct Mapped<'a> {
    pub(crate) store: &'a IdStore,
    pub(crate) internal: &'a [u32],
}

impl<'a> Mapped<'a> {
    /// How many records the mappings hold.
    pub(crate) fn len(&self) -> usize {
        self.store.len()
    }

    /// A cursor finding each record an artefact names in increasing internal
    /// id by a step forward.
    pub(crate) fn walk(&self) -> PositionWalk<'a> {
        PositionWalk::new(self.internal)
    }

    /// The external id of the record holding `internal_id`, found by a
    /// search, for a lookup out of order.
    pub(crate) fn name(&self, internal_id: usize) -> Option<&'a str> {
        let id = u32::try_from(internal_id).ok()?;
        let at = self.internal.binary_search(&id).ok()?;
        self.store.name(at)
    }

    /// Every record as its internal id and its external id, in increasing
    /// internal id.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &'a str)> + 'a {
        let internal = self.internal;
        self.store
            .iter()
            .map(move |(at, id)| (internal[at] as usize, id))
    }
}

/// Build the id store from mappings.bin, and say how the directory holds its
/// four binary artefacts
///
/// Every internal id the artefact names is held to `id_counter`, the last id
/// `config.json` says the index issued, before the store is reserved, and to
/// the last id a collection issues. A save holds the mutation guard for its
/// whole run, so every id the file names was issued by the counter it
/// records. The graph dump's origin ids are held to the same counter; see
/// `Expected::max_origin_id`. The records take the positions from zero in
/// increasing internal id, so every per-record structure the load builds is
/// sized by the records the file holds and not by the ids they carry.
///
/// The store is built before the collection is, because the framed
/// `vectors.bin` and `pq_codes.bin` name records by internal id and are held
/// to it as they are read. The collection takes it whole.
fn load_ids(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    major: u32,
    config: &IndexConfig,
) -> Result<(MappedIds, Layout), Error> {
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
/// what it will hold. The record at entry `k` takes position `k`. An id held
/// under two internal ids is refused, and so is an internal id past the last
/// one a collection issues, in the words the store gave it while an entry
/// was an internal id.
fn framed_ids(bytes: &[u8], config: &IndexConfig) -> Result<MappedIds, Error> {
    let invalid = |error: String| Error::ArtefactParseFailed {
        name: "mappings.bin",
        error,
    };
    let records = framed::Mappings::read(bytes, "mappings.bin")?;
    if let Some((internal_id, id)) = records.last() {
        if internal_id > config.id_counter {
            return Err(invalid(format!(
                "the record '{}' holds internal id {} and config.json counted {}",
                id, internal_id, config.id_counter
            )));
        }
    }
    let mut store = IdStore::new(config.expected_size);
    store.reserve(records.len(), records.len().saturating_sub(1));
    let mut internal = Vec::with_capacity(records.len());
    for (at, (internal_id, id)) in records.iter().enumerate() {
        refuse_past_last_id(internal_id)?;
        // `IdStore::insert`'s refusals, being the position's ceiling, the
        // id's length and the arena's, then the record.
        refuse_past_last_id(at)?;
        refuse_unnamable(id, store.arena_len())?;
        store.insert_admitted(at, id);
        internal.push(internal_id as u32);
    }
    if store.len() != records.len() {
        return Err(invalid(format!(
            "{} records hold {} distinct ids, so an id is held under two internal ids",
            records.len(),
            store.len()
        )));
    }
    Ok(MappedIds { store, internal })
}

/// Refuse an internal id past the last one a collection issues, which no
/// save writes, in the words the id store refused it in while an entry was
/// an internal id.
pub(crate) fn refuse_past_last_id(internal_id: usize) -> Result<(), Error> {
    if internal_id > MAX_INTERNAL_ID {
        return Err(Error::Engine(format!(
            "internal id {} is above what the id store can hold",
            internal_id
        )));
    }
    Ok(())
}

/// The id store bincode's two maps describe, as releases before 4.0.0 wrote
/// them
///
/// The forward map is what the store is built from, and the reverse map is
/// held to being its exact inverse, since the two were written from one
/// structure and a file whose two halves disagree describes two record sets.
/// The forward map is sorted by internal id first, so the records take the
/// positions from zero in increasing internal id, as a framed file's do.
fn bincode_ids(bytes: &[u8], config: &IndexConfig) -> Result<MappedIds, Error> {
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
    let mut pairs: Vec<(usize, &String)> = id_map
        .iter()
        .map(|(id, &internal_id)| (internal_id, id))
        .collect();
    pairs.sort_unstable();
    for &(internal_id, _) in &pairs {
        refuse_past_last_id(internal_id)?;
    }
    // Two records under one internal id were one record to the store that
    // was keyed by it, so the file names fewer records than it lists.
    let distinct = pairs.len()
        - pairs
            .windows(2)
            .filter(|pair| pair[0].0 == pair[1].0)
            .count();
    if distinct != id_map.len() {
        return Err(invalid(format!(
            "the forward map names {} records under {} internal ids",
            id_map.len(),
            distinct
        )));
    }
    let mut store = IdStore::new(config.expected_size);
    store.reserve(pairs.len(), pairs.len().saturating_sub(1));
    let mut internal = Vec::with_capacity(pairs.len());
    for (at, &(internal_id, id)) in pairs.iter().enumerate() {
        // `IdStore::insert`'s refusals, being the position's ceiling, the
        // id's length and the arena's, then the record.
        refuse_past_last_id(at)?;
        refuse_unnamable(id, store.arena_len())?;
        store.insert_admitted(at, id);
        internal.push(internal_id as u32);
    }
    let mapped = MappedIds { store, internal };
    if rev_map.len() != id_map.len() {
        return Err(invalid(format!(
            "the forward map holds {} records and the reverse map {}",
            id_map.len(),
            rev_map.len()
        )));
    }
    if let Some((internal_id, id)) = rev_map
        .iter()
        .find(|(&internal_id, id)| mapped.view().name(internal_id) != Some(id.as_str()))
    {
        return Err(invalid(format!(
            "the reverse map names internal id {} as '{}' and the forward map does not",
            internal_id, id
        )));
    }
    Ok(mapped)
}

/// Load vector metadata from metadata.json straight into the store the
/// collection takes, each record's fields at the position the mappings give
/// its id
///
/// At 5.x the file names only the records that hold a field, and each
/// record it names must be one the mappings hold, named once. A record it
/// leaves out takes an empty entry, as a record with no field holds in
/// memory, since a filter judges an empty entry and never an absent one.
/// Below 5.x the file names every record, an entry for an id the mappings do
/// not hold is dropped and counted, and a record named twice keeps its last
/// entry.
///
/// The file is read once, with nothing between its text and the store. Each
/// id is resolved through the mappings as its key is read, borrowed from the
/// text, each field name is interned with no copy where the store holds it
/// already, and each record's fields take one block. The entry vector is
/// sized once, to the records the mappings hold or the reservation for
/// `expected_size`, whichever is more, whatever order the file names them
/// in. Text that is not the shape a metadata file takes is refused before
/// any record the file names.
fn load_metadata(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    major: u32,
    ids: Mapped<'_>,
    expected_size: usize,
) -> Result<MetadataStore, Error> {
    debug!(target: LOG_TARGET, "Loading metadata.json...");

    let metadata_data = read_artefact_string(dir, "metadata.json", manifest)?;
    let compact = major >= DUMP_V2_FORMAT_MAJOR;
    let extent = ids.store.extent();
    let mut reader = MetadataReader {
        store: MetadataStore::with_records(extent, expected_size),
        ids,
        compact,
        named: if compact {
            vec![false; extent]
        } else {
            Vec::new()
        },
        refused: None,
        unmapped: 0,
    };
    let mut deserializer = serde_json::Deserializer::from_str(&metadata_data);
    serde::de::DeserializeSeed::deserialize(&mut reader, &mut deserializer)
        .and_then(|()| deserializer.end())
        .map_err(|e| Error::ArtefactParseFailed {
            name: "metadata.json",
            error: e.to_string(),
        })?;
    let MetadataReader {
        mut store,
        refused,
        unmapped,
        ..
    } = reader;
    if let Some(refused) = refused {
        return Err(Error::DecodeFailed {
            file: "metadata.json".to_string(),
            error: match refused {
                MetadataRefused::Unmapped(id) => {
                    format!("names record '{}', which mappings.bin does not hold", id)
                }
                MetadataRefused::Twice(id) => format!("names record '{}' twice", id),
            },
        });
    }
    if compact {
        store.fill_empty(ids.store.slots());
        debug!(target: LOG_TARGET, "metadata.json loaded (compact)");
    } else {
        if unmapped > 0 {
            debug!(target: LOG_TARGET, operation = "restore_metadata",
                unmapped = unmapped,
                "metadata.json names records the mappings do not hold, and their entries were dropped"
            );
        }
        debug!(target: LOG_TARGET, "metadata.json loaded");
    }
    Ok(store)
}

/// Why a metadata.json whose text parses is refused: the first record it
/// names that the load cannot place, in the order the file names them.
enum MetadataRefused {
    /// A record the mappings do not hold.
    Unmapped(String),
    /// A record the file names a second time.
    Twice(String),
}

/// The reader [`load_metadata`] runs over the file's text, filling `store`.
///
/// A record the load refuses is noted rather than raised, and the rest of
/// the file is still read, keeping nothing, so that text which does not
/// parse is refused before any record the file names.
struct MetadataReader<'m> {
    store: MetadataStore,
    ids: Mapped<'m>,
    /// Whether the file is the compact one of 5.x.
    compact: bool,
    /// At 5.x, which positions the file has named.
    named: Vec<bool>,
    /// The first record the load refuses.
    refused: Option<MetadataRefused>,
    /// Below 5.x, the entries naming an id the mappings do not hold.
    unmapped: usize,
}

impl<'de> serde::de::DeserializeSeed<'de> for &mut MetadataReader<'_> {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> serde::de::Visitor<'de> for &mut MetadataReader<'_> {
    type Value = ();

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The words the parse each layout was read with gave.
        f.write_str(if self.compact {
            "an object of records"
        } else {
            "a map"
        })
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        // One record's fields as they are read, kept from one record to the
        // next, so the record's block is the only thing allocated for it.
        let mut pairs: Vec<(Cow<'de, str>, Value)> = Vec::new();
        while let Some(JsonKey(id)) = map.next_key::<JsonKey<'de>>()? {
            let at = self.ids.store.slot_of(&id);
            let placed = match at {
                None if self.compact => {
                    self.refused
                        .get_or_insert(MetadataRefused::Unmapped(id.into_owned()));
                    false
                }
                None => {
                    self.unmapped += 1;
                    false
                }
                Some(at) if self.compact && std::mem::replace(&mut self.named[at], true) => {
                    self.refused
                        .get_or_insert(MetadataRefused::Twice(id.into_owned()));
                    false
                }
                Some(_) => true,
            };
            let keep = placed && self.refused.is_none();
            map.next_value_seed(RecordFieldsSeed {
                pairs: &mut pairs,
                keep,
            })?;
            if let (true, Some(at)) = (keep, at) {
                self.store.insert_pairs(at, pairs.drain(..));
            }
        }
        Ok(())
    }
}

/// One record's fields, read into `pairs` where `keep` is set and read and
/// let go otherwise.
struct RecordFieldsSeed<'p, 'de> {
    pairs: &'p mut Vec<(Cow<'de, str>, Value)>,
    keep: bool,
}

impl<'de> serde::de::DeserializeSeed<'de> for RecordFieldsSeed<'_, 'de> {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> serde::de::Visitor<'de> for RecordFieldsSeed<'_, 'de> {
    type Value = ();

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The words the parse of a record's fields as a map gave.
        f.write_str("a map")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(JsonKey(name)) = map.next_key::<JsonKey<'de>>()? {
            if self.keep {
                let value: Value = map.next_value()?;
                self.pairs.push((name, value));
            } else {
                map.next_value::<serde::de::IgnoredAny>()?;
            }
        }
        Ok(())
    }
}

/// A key of a JSON object, borrowed from the text wherever it holds no
/// escape, which is every id and field name a save writes.
struct JsonKey<'de>(Cow<'de, str>);

impl<'de> Deserialize<'de> for JsonKey<'de> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Key;

        impl<'de> serde::de::Visitor<'de> for Key {
            type Value = JsonKey<'de>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a string")
            }

            fn visit_borrowed_str<E: serde::de::Error>(
                self,
                key: &'de str,
            ) -> Result<JsonKey<'de>, E> {
                Ok(JsonKey(Cow::Borrowed(key)))
            }

            fn visit_str<E: serde::de::Error>(self, key: &str) -> Result<JsonKey<'de>, E> {
                Ok(JsonKey(Cow::Owned(key.to_string())))
            }

            fn visit_string<E: serde::de::Error>(self, key: String) -> Result<JsonKey<'de>, E> {
                Ok(JsonKey(Cow::Owned(key)))
            }
        }

        deserializer.deserialize_str(Key)
    }
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
    ids: Mapped<'_>,
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
            let mut walk = ids.walk();
            for (internal_id, row) in rows.iter() {
                if let Some(id) = walk
                    .position_of(internal_id)
                    .and_then(|at| ids.store.read_name(at))
                {
                    vectors.insert(id.to_string(), framed::floats(row).collect());
                }
            }
            vectors
        }
        Layout::Bincode => {
            let vectors = legacy::read_vectors(&vectors_data, "vectors.bin", ids.len())?;
            let mut hold = LegacyHold::new(dim);
            for (id, vector) in &vectors {
                hold.see(id, vector.len(), ids);
            }
            hold.refusal("vectors.bin", |width| vectors_mismatch(width, dim))?;
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
        ids: Mapped<'_>,
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
    ids: Mapped<'_>,
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
            let mut hold = LegacyHold::new(dim);
            let (count, offenders) =
                legacy::walk_vectors(&vectors_data, "vectors.bin", ids.len(), &mut |id, width| {
                    hold.see(id, width, ids)
                })?;
            hold.refusal("vectors.bin", |width| vectors_mismatch(width, dim))?;
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

/// The refusal of a record a store names by its id and mappings.bin does not
/// hold, in the words a framed store's reader refuses one it names by
/// internal id in.
pub(crate) fn unmapped_record(file: &str, id: &str) -> Error {
    Error::DecodeFailed {
        file: file.to_string(),
        error: format!("names record '{}', which mappings.bin does not hold", id),
    }
}

/// What vectors.bin holds where its vectors are `width` values wide and
/// config.json declares `dim`.
fn vectors_mismatch(width: usize, dim: usize) -> String {
    format!(
        "holds vectors of {} values and config.json declares dim {}",
        width, dim
    )
}

/// What pq_codes.bin holds where its codes are `width` bytes wide and
/// quantization.json declares `subvectors`.
fn codes_mismatch(width: usize, subvectors: usize) -> String {
    format!(
        "holds codes of {} bytes and quantization.json declares {} subvectors",
        width, subvectors
    )
}

/// A store of bincode's wire, which names each record by its id, held to the
/// mappings and to its width entry by entry, with the rules a framed store's
/// reader holds a framed one to: every entry is `width` wide and names a
/// record the mappings hold. An id the store holds twice is judged by its
/// last copy, which is the copy its map keeps.
struct LegacyHold<'s> {
    width: usize,
    /// The ids whose last copy is not `width` wide, with that copy's width.
    wrong: BTreeMap<&'s str, usize>,
    /// The least id the mappings do not hold.
    foreign: Option<&'s str>,
}

impl<'s> LegacyHold<'s> {
    fn new(width: usize) -> Self {
        LegacyHold {
            width,
            wrong: BTreeMap::new(),
            foreign: None,
        }
    }

    /// One entry of the store, holding `held` values.
    fn see(&mut self, id: &'s str, held: usize, ids: Mapped<'_>) {
        if held == self.width {
            self.wrong.remove(id);
        } else {
            self.wrong.insert(id, held);
        }
        if ids.store.slot_of(id).is_none() && self.foreign.is_none_or(|least| id < least) {
            self.foreign = Some(id);
        }
    }

    /// The refusal a framed reader gives the store, in its order: a width
    /// that is not the declared one, in `mismatch`'s words for the least id
    /// holding one, and then the least id the mappings do not hold.
    fn refusal(self, file: &str, mismatch: impl FnOnce(usize) -> String) -> Result<(), Error> {
        if let Some(&held) = self.wrong.values().next() {
            return Err(Error::DecodeFailed {
                file: file.to_string(),
                error: mismatch(held),
            });
        }
        if let Some(id) = self.foreign {
            return Err(unmapped_record(file, id));
        }
        Ok(())
    }
}

/// Whether a directory keeps every record's raw vector in vectors.bin, being
/// one with no quantization, one not yet trained, or a trained one under
/// `quantized_with_raw`.
fn keeps_every_raw(quantization: Option<&QuantizationArtefacts>) -> bool {
    match quantization.map(|quant| &quant.config) {
        None => true,
        Some(QuantizationFile::Pq(config)) => {
            !config.is_trained || config.storage_mode == StorageMode::QuantizedWithRaw.to_string()
        }
        Some(QuantizationFile::Int8(config)) => !config.is_trained,
    }
}

/// Refuse a vectors.bin of bincode's wire that holds no vector for a record
/// the mappings hold, in the words a framed one is refused in, naming the
/// least such id.
///
/// `vectors` is what the load kept of the file, which was held to the
/// mappings as it was read, so it holds every record where it holds as many
/// as the mappings do. Where the load kept the count alone, the file is read
/// again for its ids, which only a directory short of a vector reaches.
fn refuse_missing_vectors(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    ids: Mapped<'_>,
    vectors: &RawVectors,
) -> Result<(), Error> {
    if vectors.len() == ids.len() {
        return Ok(());
    }
    let least_without = |held: &dyn Fn(&str) -> bool| {
        ids.iter()
            .map(|(_, id)| id)
            .filter(|id| !held(id))
            .min()
            .unwrap_or("")
            .to_string()
    };
    let without = match vectors {
        RawVectors::Held(map) => least_without(&|id| map.contains_key(id)),
        RawVectors::Counted(_) => {
            let bytes = read_artefact(dir, "vectors.bin", manifest)?;
            let mut held: std::collections::HashSet<&str> = std::collections::HashSet::new();
            legacy::walk_vectors(&bytes, "vectors.bin", ids.len(), &mut |id, _| {
                held.insert(id);
            })?;
            least_without(&|id| held.contains(id))
        }
    };
    Err(Error::DecodeFailed {
        file: "vectors.bin".to_string(),
        error: format!(
            "holds {} vectors and mappings.bin holds {} records; record '{}' has no vector",
            vectors.len(),
            ids.len(),
            without
        ),
    })
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
    ids: Mapped<'_>,
    dim: usize,
) -> Result<framed::Rows<'a>, Error> {
    let rows = framed::Rows::read(
        bytes,
        FrameKind::RawVectors,
        "vectors.bin",
        4,
        dim,
        |width| vectors_mismatch(width, dim),
    )?;
    let corrupt = |error: String| Error::DecodeFailed {
        file: "vectors.bin".to_string(),
        error,
    };
    let mut walk = ids.walk();
    if let Some((internal_id, _)) = rows
        .iter()
        .find(|&(internal_id, _)| walk.position_of(internal_id).is_none())
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
    ids: Mapped<'_>,
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
                |width| codes_mismatch(width, subvectors),
            )?;
            let mut codes = HashMap::with_capacity(rows.len());
            let mut walk = ids.walk();
            for (internal_id, code) in rows.iter() {
                let Some(id) = walk
                    .position_of(internal_id)
                    .and_then(|at| ids.store.read_name(at))
                else {
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
        Layout::Bincode => {
            let codes = legacy::read_codes(&codes_data, "pq_codes.bin", ids.len())?;
            let mut hold = LegacyHold::new(subvectors);
            for (id, code) in &codes {
                hold.see(id, code.len(), ids);
            }
            hold.refusal("pq_codes.bin", |width| codes_mismatch(width, subvectors))?;
            codes
        }
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
/// subvector of no values encodes nothing. `training_size` is at least 1,000
/// and `max_training_vectors`, where given, at or above it, at every format
/// version.
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
    let invalid = |detail: String| Error::QuantizationFieldInvalidInFile {
        file: file.to_string(),
        detail,
    };
    if config.training_size < 1000 {
        return Err(invalid(format!(
            "training_size is {}, and create() requires at least 1000",
            config.training_size
        )));
    }
    if let Some(max_training) = config.max_training_vectors {
        if max_training < config.training_size {
            return Err(invalid(format!(
                "max_training_vectors is {} and training_size is {}, and create() requires the \
                 first to be at or above the second",
                max_training, config.training_size
            )));
        }
    }
    Ok(())
}

/// Load quantization configuration and the codebook that goes with it
fn load_quantization(
    dir: &dyn Dir,
    manifest: &IndexManifest,
    dim: usize,
    space: &str,
    ids: Mapped<'_>,
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

    // A code is what a trained codebook encodes a record as, so a directory
    // whose quantization.json records no training holds no code, and one
    // that does holds codes nothing it carries can decode.
    if !quant_config.is_trained && !codes.is_empty() {
        return Err(Error::CodesWithoutCodebook { count: codes.len() });
    }

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

    // Save quantization components if enabled. Every record's code or row
    // is in the graph dump, which holds the only copy of each.
    if index.has_quantization() {
        save_quantization_config(index, dir, &mut ledger)?;

        if index.can_use_quantization() {
            if index.declares_int8() {
                save_int8_scales(index, dir, &mut ledger)?;
            } else {
                save_pq_centroids(index, dir, &mut ledger)?;
            }
        }
    }

    // The raw vectors a product quantized graph keeps beside its codes.
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
    // The space keys its records by position and the artefact names them by
    // internal id, being the one each record's node was inserted under, read
    // under the dense index's guard, which ranks before the space's.
    {
        let dense = index.dense_index();
        let internal = dense.ids_by_position();
        let id_of = |at: usize| internal.get(at).map_or(Position::NONE, |id| id as u32);
        space
            .index
            .read()
            .unwrap()
            .write_with(&prefix, dir, ledger, &id_of)?;
    }
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
    major: u32,
    config: IndexConfig,
    ids: MappedIds,
    layout: Layout,
    metadata: MetadataStore,
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
    // At 5.x they come from the graph once it is back, below.
    let mut quantization = quantization;
    let mut pq_codes = quantization
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

    // Step 2: Restore all data fields directly (but not the graph). The id
    // store moves into the collection, and the internal id of the record at
    // each position stays here until the load is done, since the artefacts
    // still to be read name records by it.
    let MappedIds {
        store,
        internal: positions,
    } = ids;
    restore_data_fields(&mut index, store, &config, quantization)?;

    // Step 2a: The sparse space, once the mappings are in and before the
    // graph, for the reasons `restore_spaces` gives.
    restore_spaces(&index, dir, manifest, config.id_counter, &positions)?;

    // The rows of a trained scalar directory below 5.x, held to the mappings
    // both ways before anything is built from either: every row names a
    // record the mappings hold, and every record the mappings hold has a
    // row. A directory whose rows and mappings disagree describes two
    // indexes. At 5.x the rows are in the graph dump, whose identity region
    // is held to the mappings.
    if int8_trained && major < DUMP_V2_FORMAT_MAJOR {
        check_int8_rows_against_mappings(&index, &int8_rows, &positions)?;
    }

    // Step 3: Restore the graph the save wrote. Every save dumps the graph, so
    // the ordinary path is a read. At 5.x the dump holds the only copy of each
    // record's row: a dump refused on its header, its identity or its rows
    // refuses the load, and one whose topology fails gives its rows back to
    // rebuild from. Below 5.x a dump that is absent, damaged, or written by a
    // release this build cannot interpret falls back to the rebuild from the
    // separate stores, which is the path that also upgrades a graph carrying
    // a defect the vendored patches have since fixed. See
    // `restore_graph_from_dump`.
    //
    // Each arm also says where the records came from, which the count check
    // of step 6 names if the count is not the one config.json records.
    let version = if major >= DUMP_V2_FORMAT_MAJOR { 2 } else { 1 };
    let (graph_rebuilt, source) = match index.restore_graph_from_dump(
        dir,
        config.id_counter,
        dump_bytes,
        &positions,
        version,
    )? {
        DumpOutcome::Read(nodes) => {
            debug!(target: LOG_TARGET, "HNSW graph restored from the saved dump ({} nodes)", nodes);
            (false, "Every record came from the graph dump.".to_string())
        }
        DumpOutcome::Rows {
            rows,
            reason,
            asked,
        } => {
            if asked {
                debug!(target: LOG_TARGET, "Rebuilding the HNSW graph from the dump's rows, because {}", reason);
            } else {
                warn!(target: LOG_TARGET, operation = "load_graph_rebuilt",
                    directory = %dir.locate(""),
                    dump = GRAPH_DUMP_FILENAME,
                    reason = reason.as_str(),
                    "The graph dump's topology could not be used, so the graph was rebuilt from \
                     the rows the dump holds, each region of which was verified; every record \
                     and every internal id is the saved one, and the graph's topology is not"
                );
            }
            let inserted = rebuild_graph_from_rows(&mut index, rows, &positions)?;
            debug!(target: LOG_TARGET, "Graph rebuilt from the dump's rows ({} records inserted)", inserted);
            let source = format!(
                "The graph was rebuilt from the {} rows the graph dump holds, because {}.",
                inserted, reason
            );
            (true, source)
        }
        DumpOutcome::Rebuild(reason) => {
            debug!(target: LOG_TARGET, "Rebuilding the HNSW graph, because {}", reason);
            // Both rebuilds below replay the raw vectors, so a directory
            // whose vectors were counted and not kept reads them now.
            {
                let ids = index.ids();
                let mapped = Mapped {
                    store: &ids,
                    internal: &positions,
                };
                vectors.hold(dir, manifest, mapped, layout, config.dim)?;
            }
            let empty = HashMap::new();
            let held = vectors.held().unwrap_or(&empty);
            // The stores the rebuild below reads, each with what it holds.
            let mut stores: Vec<String> = Vec::new();
            if int8_trained {
                debug!(target: LOG_TARGET, "Rebuilding scalar quantized HNSW graph from the stored rows...");
                let inserted = index.rebuild_graph_from_int8_rows(&int8_rows, &positions)?;
                debug!(target: LOG_TARGET, "Scalar quantized graph rebuilt ({} rows inserted)", inserted);
                if int8_rows.len() > 0 {
                    stores.push(format!(
                        "the {} rows {} holds",
                        int8_rows.len(),
                        INT8_ROWS_FILENAME
                    ));
                }
            } else {
                if index.can_use_quantization() {
                    debug!(target: LOG_TARGET, "Rebuilding quantized HNSW graph from stored PQ codes...");
                    rebuild_graph_from_codes(&mut index, &pq_codes, held, &positions)?;
                } else {
                    debug!(target: LOG_TARGET, "Rebuilding HNSW graph from vectors...");
                    rebuild_graph_from_data(&mut index, held, &pq_codes, &positions)?;
                }
                if !pq_codes.is_empty() {
                    stores.push(format!("the {} codes pq_codes.bin holds", pq_codes.len()));
                }
                if !held.is_empty() {
                    stores.push(format!("the {} vectors vectors.bin holds", held.len()));
                }
            }
            let source = if stores.is_empty() {
                format!(
                    "The graph dump was not read, because {}, and the directory holds no vector, \
                     code or row to rebuild the graph from.",
                    reason
                )
            } else {
                format!(
                    "The graph dump was not read, because {}, and the graph was rebuilt from {}.",
                    reason,
                    stores.join(" and ")
                )
            };
            (true, source)
        }
    };
    // An empty directory holds no dump to read, since a save of an empty
    // index writes none, so the rebuild that runs over no records is the
    // ordinary path rather than a graph that could not be read.
    let graph_rebuilt = graph_rebuilt && index.vector_count() > 0;

    // At 5.x the codes map of a product quantized graph is every live
    // record's codes, read off the graph's own store, which the dump's rows
    // filled. It holds nothing for any other graph.
    if major >= DUMP_V2_FORMAT_MAJOR {
        pq_codes = index.codes_from_graph();
    }

    // Step 4: The stored record data, exactly as it was written, once the
    // graph is back. Neither rebuild touches the storage maps, and the
    // columns are built over the id store, which holds the records the
    // mappings name and no other.

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
    check_restored_count(&mut index, &config, source)?;

    // Every structure is now at the positions the mappings gave, and the
    // internal ids by position are let go.
    drop(positions);
    debug_assert!(
        index.positions_agree(),
        "a load leaves every structure at the positions the mappings gave"
    );

    debug!(target: LOG_TARGET, "Reconstruction completed!");
    Ok((index, graph_rebuilt))
}

/// Hold the rows artefact to the mappings in both directions.
fn check_int8_rows_against_mappings(
    index: &Collection,
    rows: &Int8Rows,
    positions: &[u32],
) -> Result<(), Error> {
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: INT8_ROWS_FILENAME.to_string(),
        detail,
    };
    let store = index.ids();
    let ids = Mapped {
        store: &store,
        internal: positions,
    };
    let mut walk = ids.walk();
    if let Some((id, _)) = rows.iter().find(|&(id, _)| walk.position_of(id).is_none()) {
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

/// Reconcile the stored vector count with the records that were restored
///
/// `vector_count` is written to config.json, and a load holds it to the
/// records it restored, so the index never reports a record the directory
/// does not hold. The count is derived here from the restored data and
/// asserted against the saved value. They agree for every directory whose
/// files are intact, so where they disagree the load fails, naming in
/// `source` where the records it counted came from, rather than producing
/// an index that misreports what it holds.
fn check_restored_count(
    index: &mut Collection,
    config: &IndexConfig,
    source: String,
) -> Result<(), Error> {
    let restored = index.count_stored_records();

    if restored != config.vector_count {
        return Err(Error::RestoredCountMismatch {
            restored,
            expected: config.vector_count,
            source,
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

/// Rebuild the graph from the rows a version 2 dump holds, each region of it
/// verified, where its topology could not be used or the rebuild was asked
/// for
///
/// Every live record's row is re-inserted in increasing internal id, filed
/// at the position the mappings give its origin id, which is the order every
/// other rebuild takes, and a stranded node's row is dropped. A raw graph
/// rebuilds from the vectors through the path a raw rebuild always took, a
/// product quantized one from the codes and a scalar one from the rows,
/// under the quantizer or the codec the loader installed. The raw vectors a
/// `quantized_with_raw` graph keeps go beside the codes afterwards, as on
/// every load.
fn rebuild_graph_from_rows(
    index: &mut Collection,
    rows: DumpRows,
    positions: &[u32],
) -> Result<usize, Error> {
    let DumpRows {
        positions: record_of,
        width,
        values,
        ..
    } = rows;
    // Each live node, by the position its record holds, in increasing
    // position, which is increasing internal id.
    let mut live: Vec<(usize, usize)> = record_of
        .iter()
        .enumerate()
        .filter(|&(_, &at)| at != Position::NONE)
        .map(|(node, &at)| (at as usize, node))
        .collect();
    live.sort_unstable();
    let row = |node: usize| node * width..(node + 1) * width;
    let named = |at: usize, store: &IdStore| -> Result<String, Error> {
        store.name(at).map(str::to_string).ok_or_else(|| {
            Error::Engine(format!(
                "the dump files a node at position {}, which mappings.bin does not hold",
                at
            ))
        })
    };
    match values {
        RowValues::Raw(values) => {
            let records = {
                let ids = index.ids();
                live.iter()
                    .map(|&(at, node)| Ok((named(at, &ids)?, values[row(node)].to_vec())))
                    .collect::<Result<Vec<_>, Error>>()?
            };
            index.rebuild_from_records(records, positions)
        }
        RowValues::Codes(values) => {
            let codes = {
                let ids = index.ids();
                let mut codes = HashMap::with_capacity(live.len());
                for &(at, node) in &live {
                    codes.insert(named(at, &ids)?, values[row(node)].to_vec());
                }
                codes
            };
            let (inserted, _) =
                index.rebuild_graph_from_codes(&codes, &HashMap::new(), positions)?;
            Ok(inserted)
        }
        RowValues::Int8(values) => {
            let mut held = Int8Rows::with_capacity(live.len(), width);
            for &(at, node) in &live {
                held.push_row(positions[at], &values[row(node)]);
            }
            index.rebuild_graph_from_int8_rows(&held, positions)
        }
    }
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
    positions: &[u32],
) -> Result<(), Error> {
    let (inserted, quantized_from_raw) =
        index.rebuild_graph_from_codes(pq_codes, vectors, positions)?;

    if quantized_from_raw > 0 {
        debug!(target: LOG_TARGET, "{} records had a raw vector and no stored PQ codes and were quantized \
             through the loaded codebook",
            quantized_from_raw
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
/// PQ codes is refused, since no trained codebook can decode them here, which
/// the read of pq_codes.bin refuses first. The metadata is not carried here;
/// step 4 of the reconstruction writes it back from the file under the same
/// ids.
fn rebuild_graph_from_data(
    index: &mut Collection,
    vectors: &HashMap<String, Vec<f32>>,
    pq_codes: &HashMap<String, Vec<u8>>,
    positions: &[u32],
) -> Result<(), Error> {
    let code_only = pq_codes
        .keys()
        .filter(|id| !vectors.contains_key(*id))
        .count();
    if code_only > 0 {
        return Err(Error::CodesWithoutCodebook { count: code_only });
    }

    if vectors.is_empty() {
        debug!(target: LOG_TARGET, "No records to rebuild (empty index)");
        // The collection keeps the graph it was built with, which reserves
        // nothing until it is given the declared reservation here.
        index.reserve_dense_graph();
        return Ok(());
    }

    // Every record with a raw vector, replayed from it
    let records: Vec<(String, Vec<f32>)> = vectors
        .iter()
        .map(|(ext_id, vector)| (ext_id.clone(), vector.clone()))
        .collect();

    debug!(target: LOG_TARGET, "Prepared {} records for the graph rebuild, each replayed from its raw vector",
        records.len()
    );
    debug!(target: LOG_TARGET, "Rebuilding the graph from the restored records...");

    // The records are owned Rust and go straight into a fresh graph, in
    // ascending internal id, which `rebuild_from_records` orders them into
    // from the restored mappings. This used to build a PyDict holding three
    // PyLists and call add(), which parsed them back into exactly this.
    let inserted = index.rebuild_from_records(records, positions)?;

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

/// The lock a load holds, and why the load may not write, where it may not.
struct LoadLock {
    guard: Option<Box<dyn LockGuard>>,
    read_only: Option<String>,
}

impl LoadLock {
    /// Take the directory's lock in `mode`. A holder that excludes it
    /// refuses the load. A lock file the medium will not open for writing
    /// gives a shared lock, or none where the file does not exist, and a
    /// load that may not write. A filesystem that cannot lock gives no lock
    /// and the same, since a read corrupts nothing and a journal could not
    /// be kept safe there. A lock file that will not open for any other
    /// reason refuses the load.
    fn take(storage: &dyn Storage, mode: LockMode, path: &str) -> Result<Self, Error> {
        match storage.lock(mode, &crate::journal::holder_record(mode)) {
            Ok(Locked::Writable(guard)) => Ok(LoadLock {
                guard: Some(guard),
                read_only: None,
            }),
            Ok(Locked::ReadOnly { guard, reason }) => Ok(LoadLock {
                guard,
                read_only: Some(reason.to_string()),
            }),
            Err(LockFailure::Held { holder }) => Err(Error::CollectionLocked {
                path: path.to_string(),
                holder,
            }),
            Err(LockFailure::Failed(error)) => Err(Error::LockUnavailable {
                path: storage.lock_path().display().to_string(),
                error: error.to_string(),
            }),
            Err(LockFailure::Unavailable(error)) => {
                let lock = storage.lock_path().display().to_string();
                warn!(target: LOG_TARGET, operation = "load_unlocked",
                    directory = path,
                    lock = lock.as_str(),
                    error = %error,
                    "The lock file cannot be locked, so the load goes on without the lock and \
                     opens a journaled directory read only"
                );
                Ok(LoadLock {
                    guard: None,
                    read_only: Some(format!(
                        "its lock file {} cannot be locked ({})",
                        lock, error
                    )),
                })
            }
        }
    }
}

/// The load, from the storage the directory `path` names. `path` is the
/// caller's own words for it, which the refusals and the log quote. The
/// tests reach it with a storage of their own, whose medium refuses writes
/// or whose journal fails on purpose.
///
/// The lock comes first, shared, so no save in another process is between
/// its two renames while this one reads. A load that will replay a journal,
/// to repair it or not, holds the lock exclusive and its sink keeps it, so
/// it reads the manifest under the shared lock, lets that go and takes the
/// exclusive lock before it reads anything else, and the load proper reads
/// the manifest again.
pub(crate) fn load_from(
    storage: &dyn Storage,
    path: &str,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    policy: JournalPolicy,
) -> Result<(Collection, Recovery), Error> {
    // Nothing to open, refused before the lock file is created beside a
    // directory that is not there.
    if !storage.exists() && !storage.aside_exists() {
        return Err(Error::IndexDirectoryNotFound {
            path: path.to_string(),
        });
    }
    let mut lock = LoadLock::take(storage, LockMode::Shared, path)?;
    let mut restored_from_aside = false;
    if matches!(policy, JournalPolicy::Replay(_) | JournalPolicy::Repair(_))
        && lock.read_only.is_none()
    {
        restored_from_aside = storage.recover()?;
        // A directory holding a feature this build does not know, marked
        // read_only, is read under the shared lock, since its load writes
        // nothing.
        if storage.exists() {
            let manifest = load_manifest(storage.dir())?;
            let read_only = manifest
                .features
                .as_ref()
                .is_some_and(features::lists_read_only);
            if manifest.journal.is_some() && !read_only {
                drop(lock);
                lock = LoadLock::take(storage, LockMode::Exclusive, path)?;
            }
        }
    }
    load_locked(storage, path, tokenizer, policy, lock, restored_from_aside)
}

/// The load under `lock`. `restored_before` says whether the look at the
/// manifest before it put a directory back from `<name>.zdbold`.
fn load_locked(
    storage: &dyn Storage,
    path: &str,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    policy: JournalPolicy,
    lock: LoadLock,
    restored_before: bool,
) -> Result<(Collection, Recovery), Error> {
    // A save killed between its two renames left the whole index beside the
    // target and nothing at it. Recovery from a killed process is a load, so
    // it is put back here, before the directory is looked for. See
    // `Storage::recover`.
    let restored_from_aside = storage.recover()? || restored_before;

    // Validate directory exists
    if !storage.exists() {
        return Err(Error::IndexDirectoryNotFound {
            path: path.to_string(),
        });
    }

    let Checkpoint {
        index: mut restored_index,
        manifest,
        major,
        identity,
        graph_rebuilt,
        read_only_features,
    } = read_checkpoint(storage, tokenizer.clone())?;

    // Phase 3: The journal beside the directory, where the manifest names
    // one. Nothing below runs for a directory saved without one, so such a
    // directory loads exactly as it did.
    let format_version = manifest.format_version.clone();
    let identity_sequence = manifest
        .identity
        .as_ref()
        .and_then(|record| record.sequence)
        .filter(|_| major >= SEGMENTED_FORMAT_MAJOR);
    let Some(record) = manifest.journal else {
        // The journal sequence the directory's state holds, which a save of
        // it records again.
        if let Some(sequence) = identity_sequence {
            restored_index.set_journal_sequence(sequence);
        }
        if major >= SEGMENTED_FORMAT_MAJOR
            && (storage.journal_exists() || storage.legacy_journal_exists())
        {
            warn!(target: LOG_TARGET, operation = "load_journal_left",
                directory = path,
                "A journal sits beside a directory whose manifest names none; it was left where \
                 it is and not read"
            );
        }
        let recovery = Recovery::unjournaled(format_version, restored_from_aside, graph_rebuilt);
        if !read_only_features.is_empty() {
            let reached = identity_sequence.unwrap_or(0);
            let cause = ReadOnlyCause::Features(read_only_features);
            return Ok(read_only(restored_index, path, cause, reached, recovery));
        }
        recovery.report(path);
        return Ok((restored_index, recovery));
    };

    let Some(journal_id) = crate::journal::id_from_hex(&record.collection_id) else {
        return Err(Error::JournalManifestInvalid {
            detail: format!(
                "its collection_id is '{}', which is not 32 hexadecimal digits",
                record.collection_id
            ),
        });
    };
    // The collection the journal must belong to is the identity's where the
    // manifest records one, and the journal record's where only that does.
    let directory_id = match identity {
        Some(identity) => identity.collection_id,
        None => {
            restored_index.set_collection_id(journal_id);
            journal_id
        }
    };
    // No save writes the identity and the journal record apart, so a 5.x
    // manifest whose two name two sequences was assembled by hand.
    if major >= SEGMENTED_FORMAT_MAJOR && identity_sequence != Some(record.sequence) {
        return Err(Error::JournalManifestInvalid {
            detail: format!(
                "its sequence is {} and the directory's identity records {}",
                record.sequence,
                identity_sequence.map_or_else(|| "none".to_string(), |s| s.to_string())
            ),
        });
    }
    restored_index.set_journal_sequence(record.sequence);

    let (durability, repairing) = match policy {
        JournalPolicy::CheckpointOnly => {
            warn!(target: LOG_TARGET, operation = "load_checkpoint_only",
                directory = path,
                journal = %record.file,
                sequence = record.sequence,
                "Opened the checkpoint alone; every mutation after the sequence it holds is in \
                 the journal beside it and was not applied"
            );
            let mut recovery =
                Recovery::unjournaled(format_version, restored_from_aside, graph_rebuilt);
            recovery.checkpoint_sequence = record.sequence;
            if !read_only_features.is_empty() {
                let cause = ReadOnlyCause::Features(read_only_features);
                return Ok(read_only(
                    restored_index,
                    path,
                    cause,
                    record.sequence,
                    recovery,
                ));
            }
            recovery.report(path);
            return Ok((restored_index, recovery));
        }
        JournalPolicy::Replay(durability) => (durability, false),
        JournalPolicy::Repair(durability) => (durability, true),
    };

    let mut recovery = Recovery::unjournaled(format_version, restored_from_aside, graph_rebuilt);
    recovery.journaled = true;
    recovery.checkpoint_sequence = record.sequence;
    let opened = Opened {
        storage,
        path,
        tokenizer,
        record: &record,
        journal_id,
        directory_id,
        snapshot: identity.and_then(|identity| identity.snapshot),
        durability,
        repairing,
        lock,
        read_only_features,
    };
    if major >= SEGMENTED_FORMAT_MAJOR {
        let identity = identity.expect("a 5.x manifest was held to record an identity");
        opened.replay_segments(restored_index, &identity, recovery)
    } else {
        opened.replay_legacy(restored_index, recovery)
    }
}

/// The checkpoint a directory holds, read and built into a collection,
/// before any journal is read.
struct Checkpoint {
    index: Collection,
    manifest: IndexManifest,
    major: u32,
    identity: Option<crate::Identity>,
    graph_rebuilt: bool,
    /// The features the directory holds that this build does not know and
    /// its manifest marks `read_only`, in name order, which make the load
    /// read only.
    read_only_features: Vec<String>,
}

/// Phases 1 and 2 of a load: every artefact of the directory `storage`
/// keeps read and held to its manifest, and the collection built from them
/// at the identity the manifest records. A repair reads the checkpoint again
/// where a record it replayed would not apply, so the call that record began
/// is applied whole or not at all.
fn read_checkpoint(
    storage: &dyn Storage,
    tokenizer: Option<Arc<dyn Tokenizer>>,
) -> Result<Checkpoint, Error> {
    // Phase 1: Load all ZeusDB components
    debug!(target: LOG_TARGET, "Phase 1: Loading ZeusDB components...");

    let dir = storage.dir();
    let manifest = load_manifest(dir)?;
    let major = check_format_version(&manifest.format_version)?;

    // The features the directory lists, before any artefact is read, so a
    // directory holding one this build does not know is refused by its name
    // rather than read without it, or read only where its manifest marks it
    // so. See `features`. Every 5.x manifest lists them, possibly none, so
    // one listing nothing at all was not written by a save.
    let reading = match manifest.features.as_ref() {
        Some(record) => features::read(record, &features::THIS_BUILD)?,
        None if major >= SEGMENTED_FORMAT_MAJOR => return Err(features::missing()),
        None => features::Reading::default(),
    };

    // Every 5.x manifest records its identity with the journal sequence its
    // state holds. Below 5.x the sequence is not read, as no release reading
    // those formats reads it.
    match manifest.identity.as_ref() {
        None if major >= SEGMENTED_FORMAT_MAJOR => {
            return Err(Error::IdentityInvalid {
                detail: format!(
                    "the manifest records none, and every manifest at format {}.x records one",
                    SEGMENTED_FORMAT_MAJOR
                ),
            })
        }
        Some(record) if major >= SEGMENTED_FORMAT_MAJOR && record.sequence.is_none() => {
            return Err(Error::IdentityInvalid {
                detail: format!(
                    "it records no sequence, and every identity at format {}.x records the \
                     journal sequence its state holds",
                    SEGMENTED_FORMAT_MAJOR
                ),
            })
        }
        _ => {}
    }

    // A manifest below the major that names a journal is a directory
    // assembled by hand, since no release writing that format wrote the
    // field. The same rule the sparse space's declaration takes below.
    if major < JOURNAL_FORMAT_MAJOR && manifest.journal.is_some() {
        return Err(Error::FormatVersionJournal {
            format_version: manifest.format_version.clone(),
        });
    }

    // What the directory is, where its manifest records it, held to what a
    // save writes before any artefact is read.
    let identity = manifest
        .identity
        .as_ref()
        .map(IdentityManifest::read)
        .transpose()?;

    // Every field of a 5.x manifest a save writes from the directory, held to
    // it before any artefact is read.
    if major >= DUMP_V2_FORMAT_MAJOR {
        check_manifest_fields(dir, &manifest)?;
    }

    // Before any artefact is read, so a directory missing two files names the
    // first rather than failing on whichever one a partial load reaches.
    check_files_present(dir, &manifest, major)?;
    debug!(target: LOG_TARGET, "Manifest loaded: {} vectors, format v{}",
        manifest.total_vectors, manifest.format_version
    );

    let config = load_config(dir, &manifest, major)?;
    debug!(target: LOG_TARGET, "Config loaded: dim={}, space={}", config.dim, config.space);
    if major >= DUMP_V2_FORMAT_MAJOR && manifest.total_vectors != config.vector_count {
        return Err(Error::ManifestFieldInvalid {
            detail: format!(
                "records total_vectors as {}, and config.json records vector_count {}",
                manifest.total_vectors, config.vector_count
            ),
        });
    }

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

    let metadata = load_metadata(dir, &manifest, major, ids.view(), config.expected_size)?;
    debug!(target: LOG_TARGET, "Metadata loaded: {} records", metadata.len());

    // A raw index takes its vectors back from the graph dump, so its
    // vectors.bin is read for its count and its check and nothing of it is
    // kept. See `RawVectors`.
    let vectors = if manifest_names(&manifest, "quantization.json") {
        RawVectors::Held(load_vectors(
            dir,
            &manifest,
            ids.view(),
            layout,
            config.dim,
        )?)
    } else {
        RawVectors::Counted(count_vectors(
            dir,
            &manifest,
            ids.view(),
            layout,
            config.dim,
        )?)
    };
    debug!(target: LOG_TARGET, "Vectors loaded: {} vectors", vectors.len());

    let quantization = load_quantization(
        dir,
        &manifest,
        config.dim,
        &config.space,
        ids.view(),
        layout,
    )?;
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

    // Below 4.0.0 vectors.bin was held to the mappings and to its width as it
    // was read. Where the directory keeps every record's raw vector, it holds
    // one for every record the mappings hold as well, as a framed vectors.bin
    // is held to at 4.0.0. A trained `quantized_only` index keeps no raw
    // vector but the training records' some releases left there, so its file
    // is not held to the mappings that way.
    if matches!(layout, Layout::Bincode)
        && manifest_names(&manifest, "vectors.bin")
        && keeps_every_raw(quantization.as_ref())
    {
        refuse_missing_vectors(dir, &manifest, ids.view(), &vectors)?;
    }

    if major >= DUMP_V2_FORMAT_MAJOR {
        check_quantization_fields(&manifest, quantization.as_ref())?;
    }

    // From 4.x on the features a manifest lists are the ones the directory
    // holds, held now that every part they name has been read and before
    // anything is built from them. Below 4.x no release listed features, and
    // the version decides what a directory holds.
    if let Some(record) = manifest
        .features
        .as_ref()
        .filter(|_| major >= FEATURES_FORMAT_MAJOR)
    {
        let quantizer = quantization.as_ref().map(|quant| &quant.config);
        let held = features::Held {
            // Listed at 4.x where recorded, and part of the base at 5.x.
            identity: manifest.identity.is_some() && major < SEGMENTED_FORMAT_MAJOR,
            int8: matches!(quantizer, Some(QuantizationFile::Int8(_))),
            journal: manifest.journal.is_some(),
            pq: matches!(quantizer, Some(QuantizationFile::Pq(_))),
            sparse: !config.spaces.is_empty(),
            text: config.spaces.iter().any(|space| space.tokenizer.is_some()),
        };
        features::check_held(record, held)?;
    }

    // The graph dump itself is read inside the reconstruction, which needs the
    // mappings and the codebook first to judge it against.

    // Phase 2: Create empty index and restore state
    debug!(target: LOG_TARGET, "Phase 2: Creating empty index and restoring state...");
    let (mut restored_index, graph_rebuilt) = reconstruct_index_simple(
        dir,
        &manifest,
        major,
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
    restored_index.set_created_at(manifest.created_at.clone());

    // The collection the directory names, at the generation and snapshot it
    // records, with the time and the sequence its manifest records, so its
    // next save is the next generation of the same collection. A directory
    // recording none keeps the id drawn at assembly, or takes its journal
    // record's after this, and its lineage stays at a collection never saved.
    if let Some(identity) = identity {
        restored_index.set_collection_id(identity.collection_id);
        let sequence = manifest
            .identity
            .as_ref()
            .and_then(|record| record.sequence)
            .filter(|_| major >= SEGMENTED_FORMAT_MAJOR)
            .unwrap_or(0);
        restored_index.set_lineage(&IdentityRecord {
            identity,
            saved_at: Some(manifest.saved_at.clone()),
            sequence,
        });
    }

    debug!(target: LOG_TARGET, "Index reconstruction completed successfully!");
    Ok(Checkpoint {
        index: restored_index,
        manifest,
        major,
        identity,
        graph_rebuilt,
        read_only_features: reading.read_only,
    })
}

/// Hold a 5.x manifest's fields to what a save writes, before any artefact
/// is read
///
/// `zeusdb_version` is not empty, `created_at` and `saved_at` are RFC 3339
/// times, `index_type` is `HNSW`, and `storage_mode` is one a save records.
/// `has_quantization` says whether `files_included` names
/// `quantization.json`. `files_included` and `file_digests` name the same
/// files, none of them one no 5.0.0 save writes. `files_excluded` names
/// nothing `files_included` names and nothing the directory holds. And
/// `total_size_mb`, in whole bytes, is the lengths `file_digests` records,
/// added.
fn check_manifest_fields(dir: &dyn Dir, manifest: &IndexManifest) -> Result<(), Error> {
    let invalid = |detail: String| Err(Error::ManifestFieldInvalid { detail });
    if manifest.zeusdb_version.is_empty() {
        return invalid(
            "records zeusdb_version as an empty string, and a save records the release that \
             wrote it"
                .to_string(),
        );
    }
    for (field, value) in [
        ("created_at", &manifest.created_at),
        ("saved_at", &manifest.saved_at),
    ] {
        if chrono::DateTime::parse_from_rfc3339(value).is_err() {
            return invalid(format!(
                "records {} as '{}', which is not an RFC 3339 time",
                field, value
            ));
        }
    }
    if manifest.index_type != "HNSW" {
        return invalid(format!(
            "records index_type as '{}', and a save records HNSW",
            manifest.index_type
        ));
    }
    if !STORAGE_MODES.contains(&manifest.storage_mode.as_str()) {
        return invalid(format!(
            "records storage_mode as '{}', which is not one a save records",
            manifest.storage_mode
        ));
    }
    let names_quantization = manifest_names(manifest, "quantization.json");
    if manifest.has_quantization != names_quantization {
        return invalid(format!(
            "records has_quantization as {}, and files_included {} quantization.json",
            manifest.has_quantization,
            if names_quantization {
                "names"
            } else {
                "does not name"
            }
        ));
    }
    for name in &manifest.files_included {
        if !manifest.file_digests.contains_key(name) {
            return invalid(format!(
                "names {} under files_included, and file_digests records no length for it",
                name
            ));
        }
        if name == "pq_codes.bin"
            || name == INT8_ROWS_FILENAME
            || LEGACY_DUMP_FILENAMES.contains(&name.as_str())
        {
            return invalid(format!(
                "names {} under files_included, which no save at format 5.0.0 writes",
                name
            ));
        }
    }
    let mut recorded: Vec<&String> = manifest.file_digests.keys().collect();
    recorded.sort_unstable();
    if let Some(name) = recorded.iter().find(|name| !manifest_names(manifest, name)) {
        return invalid(format!(
            "records a length for {} under file_digests, and files_included does not name it",
            name
        ));
    }
    for name in &manifest.files_excluded {
        if manifest_names(manifest, name) {
            return invalid(format!(
                "names {} under both files_included and files_excluded",
                name
            ));
        }
        if dir.exists(name) {
            return invalid(format!(
                "names {} under files_excluded, and the directory holds it",
                name
            ));
        }
    }
    // A length in bytes divided by a power of two is exact, and a JSON
    // reader may take the decimal it reads back one unit in the last place
    // away, so the field is held to the lengths in whole bytes.
    let total = manifest
        .file_digests
        .values()
        .fold(0u64, |total, digest| total.saturating_add(digest.bytes));
    if (manifest.total_size_mb * MEBIBYTE).round() != total as f64 {
        return invalid(format!(
            "records total_size_mb as {}, and the lengths file_digests records add up to {} MiB",
            manifest.total_size_mb,
            total as f64 / MEBIBYTE
        ));
    }
    Ok(())
}

/// Hold a 5.x manifest's two quantization fields to `quantization.json`, once
/// it has been read: `quantization_trained` is its `is_trained`, false where
/// there is none, and `files_included` names `vectors.bin` only beside a
/// trained product quantized graph under `quantized_with_raw`, which is the
/// one graph whose raw vectors are not its rows.
fn check_quantization_fields(
    manifest: &IndexManifest,
    quantization: Option<&QuantizationArtefacts>,
) -> Result<(), Error> {
    let (trained, keeps_raw) = match quantization.map(|quant| &quant.config) {
        Some(QuantizationFile::Pq(config)) => (
            config.is_trained,
            config.is_trained && config.storage_mode == StorageMode::QuantizedWithRaw.to_string(),
        ),
        Some(QuantizationFile::Int8(config)) => (config.is_trained, false),
        None => (false, false),
    };
    if manifest.quantization_trained != trained {
        return Err(Error::ManifestFieldInvalid {
            detail: format!(
                "records quantization_trained as {}, and {}",
                manifest.quantization_trained,
                match quantization {
                    Some(_) => format!("quantization.json records is_trained {}", trained),
                    None => "files_included names no quantization.json".to_string(),
                }
            ),
        });
    }
    if manifest_names(manifest, "vectors.bin") && !keeps_raw {
        return Err(Error::ManifestFieldInvalid {
            detail: "names vectors.bin under files_included, and the graph keeps no raw vectors \
                     beside its rows"
                .to_string(),
        });
    }
    Ok(())
}

/// What a load of a journaled directory carries into the replay.
struct Opened<'a> {
    storage: &'a dyn Storage,
    path: &'a str,
    /// The tokenizer the load was handed, which a repair hands again where
    /// it reads the checkpoint again.
    tokenizer: Option<Arc<dyn Tokenizer>>,
    record: &'a JournalManifest,
    journal_id: u128,
    directory_id: u128,
    /// The directory's snapshot, where its manifest records one, which an
    /// unfinished repair names when its checkpoint has not landed.
    snapshot: Option<u128>,
    durability: Durability,
    /// The load was asked to repair a damaged journal.
    repairing: bool,
    lock: LoadLock,
    /// The features the directory holds that this build does not know and
    /// its manifest marks `read_only`, which make the load read only
    /// whatever the medium.
    read_only_features: Vec<String>,
}

impl Opened<'_> {
    /// The checkpoint read again into a collection of its own, at the
    /// collection and the sequence the load holds the journal to, for a
    /// repair whose replay met a record that would not apply.
    fn reopened(&self) -> Result<Collection, Error> {
        let mut index = read_checkpoint(self.storage, self.tokenizer.clone())?.index;
        index.set_collection_id(self.directory_id);
        index.set_journal_sequence(self.record.sequence);
        Ok(index)
    }

    /// Refuse a journal whose damage is a whole record of a kind this build
    /// does not know, in a directory holding features this build does not
    /// know and its manifest marks `read_only`, since the record is the
    /// feature's and not damage. `file`, `at` and `sequence` name the
    /// record, and `bytes` is the file it is in. Nothing is refused where
    /// the directory holds no such feature.
    fn refuse_feature_record(
        &self,
        bytes: &[u8],
        at: u64,
        sequence: u64,
        framing: bool,
    ) -> Result<(), Error> {
        if self.read_only_features.is_empty() {
            return Ok(());
        }
        match unknown_kind_at(bytes, at, sequence, framing) {
            Some(kind) => Err(Error::FeatureJournalUnreadable {
                path: self.path.to_string(),
                features: self.read_only_features.clone(),
                kind,
                sequence,
            }),
            None => Ok(()),
        }
    }

    /// The cause of a load read only, where it is: features the directory
    /// holds before the medium, and the medium where the lock found it
    /// cannot be written.
    fn read_only_cause(&self) -> Option<ReadOnlyCause> {
        if !self.read_only_features.is_empty() {
            return Some(ReadOnlyCause::Features(self.read_only_features.clone()));
        }
        self.lock.read_only.clone().map(ReadOnlyCause::Medium)
    }

    /// No save writes the identity and the journal record apart, so a
    /// manifest whose two name two collections was assembled by hand, and
    /// replaying would be choosing one of them. Held once the journal itself
    /// is held to the identity, so a journal of another collection is
    /// refused as that first.
    fn journal_record_names_the_directory(&self) -> Result<(), Error> {
        if self.journal_id == self.directory_id {
            return Ok(());
        }
        Err(Error::JournalManifestInvalid {
            detail: format!(
                "its collection_id is '{}' and the directory's identity names collection '{}'",
                self.record.collection_id,
                crate::journal::id_hex(self.directory_id)
            ),
        })
    }

    /// Replay the journal of segments beside a 5.x directory onto the
    /// checkpoint `index` holds, then carry the journal on, or hand the
    /// collection back read only where the medium refuses the first write.
    ///
    /// Every segment is read in name order and held to the checkpoint the
    /// directory is. Each record above it is applied in order: a committed
    /// one at the values it names, a withdrawn one as the live collection
    /// left it. Then the writer resumes, its first write deciding whether
    /// the load may write: it cuts a torn tail and completes a roll a crash
    /// interrupted. Stale segments and a covered journal of format version 1
    /// are deleted. A journal that held an abort is checkpointed before it
    /// takes another record.
    ///
    /// A repair reads the journal for a repair, which cuts at the first call
    /// any break of the journal's history touches and sets aside whole a
    /// stale segment whose header cannot be used, and cuts the same way at
    /// the call of a record that would not decode or apply, reading the
    /// checkpoint again and replaying the calls before it. What it sets
    /// aside is copied into the quarantine, which is its first write, then
    /// the journal is cut where it cuts and the stale segments set aside go,
    /// the writer resumes on the journal as it now is, and a checkpoint seals
    /// it, which the other stale segments go with. A repair that finds an unfinished one naming the directory's
    /// own snapshot takes the checkpoint that one did not, and every marker
    /// goes once the checkpoint has landed.
    fn replay_segments(
        self,
        index: Collection,
        identity: &crate::Identity,
        mut recovery: Recovery,
    ) -> Result<(Collection, Recovery), Error> {
        let storage = self.storage;
        let journal = storage.journal_path();
        let journal_name = journal.display().to_string();
        if !storage.journal_exists() {
            return Err(Error::JournalMissing {
                directory: self.path.to_string(),
                file: journal_name,
                recorded: self.record.file.clone(),
                sequence: self.record.sequence,
            });
        }
        let base = SegmentBase {
            snapshot: identity
                .snapshot
                .expect("an identity read from a manifest names its snapshot"),
            sequence: self.record.sequence,
            generation: identity.generation,
            parent: identity.parent.unwrap_or(0),
        };
        let expected = Expected {
            collection_id: self.directory_id,
            base,
        };
        let io = |what: &'static str| {
            let journal = journal.to_path_buf();
            move |error: std::io::Error| Error::JournalIoFailed {
                path: journal.clone(),
                what,
                error: error.to_string(),
            }
        };
        let dir = storage.journal_dir();
        let (numbers, files, bytes) = read_segment_files(&*dir)?;
        let segments = segment_bytes(&numbers, &files, &bytes);
        let contents = if self.repairing {
            read_segments_for_repair(&segments, &expected, &journal_name)?
        } else {
            read_segments(&segments, &expected, &journal_name)?
        };
        // A record of a read-only feature is the feature's, so it is named
        // as that before it is taken for damage.
        let damage = match (&contents.damage, &contents.cut) {
            (Some(JournalDamage::Corrupt { at, sequence, .. }), _)
            | (Some(JournalDamage::TornTail { at, sequence }), _) => contents
                .damage_file
                .as_deref()
                .and_then(|file| segments.iter().find(|segment| segment.file == file))
                .map(|segment| (segment.bytes, *at, *sequence)),
            (None, Some(cut)) => segments
                .iter()
                .find(|segment| segment.number == cut.damage_segment)
                .map(|segment| (segment.bytes, cut.damage_at, cut.damage_sequence)),
            (None, None) => None,
        };
        if let Some((bytes, at, sequence)) = damage {
            self.refuse_feature_record(bytes, at, sequence, true)?;
        }
        if !self.repairing {
            if let Some(refusal) = contents.refusal() {
                return Err(refusal);
            }
        }
        self.journal_record_names_the_directory()?;

        // `apply` refuses to run with a sink attached, which is why the sink
        // goes on after this loop and not before it.
        let mut index = index;
        let mut cut = contents.cut.clone();
        let mut kept = contents.records.len();
        if let Err((place, unapplied)) = replay_live(&index, &contents.records, &numbers, &files) {
            if !self.repairing {
                return Err(unapplied.error);
            }
            let start = contents.calls[contents.calls.partition_point(|&c| c <= place) - 1];
            cut = Some(unapplied.cut_before(&contents.records, start, place, &numbers));
            kept = start;
            index = self.reopened()?;
            replay_live(&index, &contents.records[..kept], &numbers, &files)
                .map_err(|(_, unapplied)| unapplied.error)?;
        }
        let records = &contents.records[..kept];
        recovery.replayed = records
            .iter()
            .filter(|live| live.state == RecordState::Committed)
            .count();
        recovery.withdrawn = records.len() - recovery.replayed;
        recovery.first_sequence = contents
            .live()
            .next()
            .map_or(self.record.sequence.saturating_add(1), |segment| {
                segment.number
            });
        recovery.records_in_journal = kept;
        recovery.skipped = contents.skipped;
        recovery.damage = contents
            .damage
            .as_ref()
            .map(|damage| Damage::of(damage, contents.damage_file.as_deref().unwrap_or_default()));
        recovery.good_bytes = contents.good_bytes;
        recovery.segments = numbers.len();
        let mut reached = contents.next_sequence.saturating_sub(1);

        let plan = if self.repairing {
            repair::plan_segments(
                cut.as_ref(),
                &contents.stale_set_aside,
                &repair::JournalFiles {
                    numbers: &numbers,
                    files: &files,
                    bytes: &bytes,
                },
                self.snapshot,
                self.record.sequence,
                reached,
            )
        } else {
            None
        };
        if let Some(plan) = &plan {
            recovery.damage = Some(plan.damage.clone());
            if let Some(at) = plan.cut_at() {
                recovery.good_bytes = at;
            }
            reached = plan.kept_through;
        }
        let unfinished = if self.repairing {
            repair::unfinished(&*dir)
        } else {
            Vec::new()
        };

        if let Some(cause) = self.read_only_cause() {
            recovery.repair = plan.as_ref().map(|plan| plan.report(None));
            return Ok(read_only(index, self.path, cause, reached, recovery));
        }
        // The stale segments this load deletes and the records it takes rest
        // on the directory it read, which a save in an earlier process may
        // have moved into place without its sync landing, so the directory
        // is made durable before the journal changes. A medium that refuses
        // the sync as it refuses a write opens read only, as the load's
        // first write does, since nothing is then written that rests on it.
        if let Err(error) = storage.sync_in_place() {
            if refuses_writes(&error) {
                recovery.repair = plan.as_ref().map(|plan| plan.report(None));
                let cause = ReadOnlyCause::Medium(error.to_string());
                return Ok(read_only(index, self.path, cause, reached, recovery));
            }
            return Err(Error::DirectoryNotDurable {
                directory: storage.dir().locate(""),
                error: error.to_string(),
            });
        }
        let mut sealed: Vec<String> = Vec::new();
        let writer = match &plan {
            None => JournalWriter::resume(dir, self.directory_id, base, &contents),
            Some(plan) => {
                let folder = match repair::set_aside(&*dir, plan) {
                    Ok(folder) => folder,
                    Err(repair::SetAside::Create(error)) if refuses_writes(&error) => {
                        recovery.repair = Some(plan.report(None));
                        let cause = ReadOnlyCause::Medium(error.to_string());
                        return Ok(read_only(index, self.path, cause, reached, recovery));
                    }
                    Err(repair::SetAside::Create(error)) => return Err(io("set aside")(error)),
                    Err(repair::SetAside::Write(error)) => return Err(error),
                };
                zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::RepairAfterQuarantine);
                if let Some(cut) = &cut {
                    repair::cut_segments(&*dir, &numbers, cut.segment, cut.at)
                        .map_err(io("cut"))?;
                }
                repair::remove_stale(&*dir, &plan.stale).map_err(io("set aside"))?;
                zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::RepairAfterCut);
                let folder_path = PathBuf::from(dir.quarantine().locate(&folder));
                recovery.repair = Some(plan.report(Some(folder_path)));
                sealed.push(folder);
                // The writer carries on the journal as the repair left it,
                // which reads whole, so it is read again.
                let (numbers, files, bytes) = read_segment_files(&*dir)?;
                let segments = segment_bytes(&numbers, &files, &bytes);
                let cut_journal = read_segments(&segments, &expected, &journal_name)?;
                if let Some(refusal) = cut_journal.refusal() {
                    return Err(refusal);
                }
                JournalWriter::resume(storage.journal_dir(), self.directory_id, base, &cut_journal)
            }
        };
        let writer = match writer {
            Ok(writer) => writer,
            Err(failure) if failure.refuses_writes => {
                let cause = ReadOnlyCause::Medium(failure.error.to_string());
                return Ok(read_only(index, self.path, cause, reached, recovery));
            }
            Err(failure) => return Err(failure.error),
        };
        let mut sink = JournalSink::from_writer(writer, self.durability, self.lock.guard)?;
        // A repair checkpoints where it cut, and where an unfinished repair
        // names the directory's own snapshot, since that one's checkpoint
        // never landed.
        let repaired = plan.is_some()
            || unfinished
                .iter()
                .any(|(_, snapshot)| *snapshot == self.snapshot);
        if !repaired {
            sink.remove_stale()?;
            crate::journal::remove_covered_legacy(
                storage,
                self.directory_id,
                self.record.sequence,
            )?;
        }
        index.attach_sink(Box::new(sink));
        // The fault a failed commit left lived in memory alone, so a process
        // restarted after one would take writes on top of whatever the
        // failed sync lost. A checkpoint writes the collection from memory
        // and rolls past the abort first. One checkpoint serves a repair and
        // an abort together, and its roll deletes every segment the journal
        // held, the stale ones among them.
        if repaired || contents.aborts > 0 {
            index.checkpoint()?;
            recovery.checkpoint_taken = Some(if repaired {
                CheckpointTaken::Repair
            } else {
                CheckpointTaken::Abort
            });
        }
        if repaired {
            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::RepairAfterCheckpoint);
            crate::journal::remove_covered_legacy(
                storage,
                self.directory_id,
                index.journal_sequence(),
            )?;
        }
        sealed.extend(unfinished.into_iter().map(|(folder, _)| folder));
        repair::seal(&*storage.journal_dir(), &sealed)?;
        recovery.report(self.path);
        Ok((index, recovery))
    }

    /// Replay the journal of format version 1 beside a 3.x or 4.x directory
    /// onto the checkpoint `index` holds, then upgrade the directory, or
    /// hand the collection back read only where the medium refuses the
    /// upgrade's first write.
    ///
    /// The upgrade is a checkpoint at 5.0.0 with a journal of segments: the
    /// journal's directory is created empty, a writer carrying the sequence
    /// on from the last record replayed is attached, the checkpoint commits
    /// and rolls the writer onto its first segment, and then the file goes.
    /// A crash before the commit leaves the old directory and its file, which
    /// the next load replays and upgrades again. A crash after it leaves a
    /// 5.0.0 directory holding every record of the file, which the next load
    /// deletes.
    ///
    /// A repair cuts a corrupt middle at the damaged record, since version 1
    /// frames no call, and cuts the same way at a record that would not
    /// decode or apply, reading the checkpoint again and replaying the
    /// records before it. What it sets aside is copied into the quarantine
    /// of the new journal's directory, which is its first write, the file is
    /// cut, and the upgrade's checkpoint seals it.
    fn replay_legacy(
        self,
        index: Collection,
        mut recovery: Recovery,
    ) -> Result<(Collection, Recovery), Error> {
        let storage = self.storage;
        let wal = storage.legacy_journal_path();
        let file = wal.display().to_string();
        if !storage.legacy_journal_exists() {
            return Err(Error::JournalMissing {
                directory: self.path.to_string(),
                file,
                recorded: self.record.file.clone(),
                sequence: self.record.sequence,
            });
        }
        let bytes = storage
            .read_legacy_journal()
            .map_err(|error| Error::JournalIoFailed {
                path: wal.to_path_buf(),
                what: "read",
                error: error.to_string(),
            })?;
        let contents = read_legacy_journal(&bytes, &file)?;
        if let Some(
            JournalDamage::Corrupt { at, sequence, .. } | JournalDamage::TornTail { at, sequence },
        ) = &contents.damage
        {
            self.refuse_feature_record(&bytes, *at, *sequence, false)?;
        }
        crate::journal::check_legacy_contents(
            &contents,
            &file,
            self.directory_id,
            self.record.sequence,
            self.repairing,
        )?;
        self.journal_record_names_the_directory()?;

        // Version 1 frames no call, so a repair cuts at the damaged record.
        let mut cut = match (&contents.damage, self.repairing) {
            (
                Some(JournalDamage::Corrupt {
                    at,
                    sequence,
                    detail,
                }),
                true,
            ) => Some(Cut {
                segment: 0,
                at: *at,
                sequence: *sequence,
                kind: "corrupt",
                damage_segment: 0,
                damage_at: *at,
                damage_sequence: *sequence,
                detail: detail.clone(),
            }),
            _ => None,
        };
        let mut index = index;
        let mut kept = contents.records.len();
        let checkpoint = self.record.sequence;
        if let Err((place, unapplied)) =
            replay_legacy_records(&index, &contents.records, checkpoint, &file)
        {
            if !self.repairing {
                return Err(unapplied.error);
            }
            let record = &contents.records[place];
            cut = Some(Cut {
                segment: 0,
                at: record.offset,
                sequence: record.sequence,
                kind: unapplied.kind,
                damage_segment: 0,
                damage_at: record.offset,
                damage_sequence: record.sequence,
                detail: unapplied.error.to_string(),
            });
            kept = place;
            index = self.reopened()?;
            replay_legacy_records(&index, &contents.records[..kept], checkpoint, &file)
                .map_err(|(_, unapplied)| unapplied.error)?;
        }
        let records = &contents.records[..kept];
        recovery.skipped = records.iter().filter(|r| r.sequence <= checkpoint).count();
        recovery.replayed = records.len() - recovery.skipped;
        recovery.first_sequence = contents.header.first_sequence;
        recovery.records_in_journal = kept;
        recovery.damage = contents
            .damage
            .as_ref()
            .map(|damage| Damage::of(damage, &file));
        recovery.good_bytes = contents.good_bytes;
        let mut reached = contents
            .next_sequence()
            .saturating_sub(1)
            .max(self.record.sequence);
        let plan = cut.as_ref().map(|cut| {
            repair::plan_legacy(cut, &file, &bytes, self.snapshot, self.record.sequence)
        });
        if let Some(plan) = &plan {
            recovery.damage = Some(plan.damage.clone());
            if let Some(at) = plan.cut_at() {
                recovery.good_bytes = at;
            }
            reached = plan.kept_through.max(self.record.sequence);
        }
        let unfinished = if self.repairing {
            repair::unfinished(&*storage.journal_dir())
        } else {
            Vec::new()
        };

        if let Some(cause) = self.read_only_cause() {
            recovery.repair = plan.as_ref().map(|plan| plan.report(None));
            return Ok(read_only(index, self.path, cause, reached, recovery));
        }
        zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::UpgradeAfterReplay);
        let mut sealed: Vec<String> = Vec::new();
        if let Some(plan) = &plan {
            let dir = storage.journal_dir();
            let folder = match repair::set_aside(&*dir, plan) {
                Ok(folder) => folder,
                Err(repair::SetAside::Create(error)) if refuses_writes(&error) => {
                    recovery.repair = Some(plan.report(None));
                    let cause = ReadOnlyCause::Medium(error.to_string());
                    return Ok(read_only(index, self.path, cause, reached, recovery));
                }
                Err(repair::SetAside::Create(error)) => {
                    return Err(Error::JournalIoFailed {
                        path: storage.journal_path().to_path_buf(),
                        what: "set aside",
                        error: error.to_string(),
                    })
                }
                Err(repair::SetAside::Write(error)) => return Err(error),
            };
            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::RepairAfterQuarantine);
            let at = plan
                .cut_at()
                .expect("a repair of a journal held in one file cuts it");
            storage
                .cut_legacy_journal(at)
                .map_err(|error| Error::JournalIoFailed {
                    path: wal.to_path_buf(),
                    what: "cut",
                    error: error.to_string(),
                })?;
            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::RepairAfterCut);
            let folder_path = PathBuf::from(dir.quarantine().locate(&folder));
            recovery.repair = Some(plan.report(Some(folder_path)));
            sealed.push(folder);
        }
        // The upgrade's first write, before anything has changed.
        if let Err(error) = storage.probe_writable() {
            if refuses_writes(&error) {
                return Ok(read_only(
                    index,
                    self.path,
                    ReadOnlyCause::Medium(error.to_string()),
                    reached,
                    recovery,
                ));
            }
            return Err(Error::JournalIoFailed {
                path: storage.journal_path().to_path_buf(),
                what: "upgrade",
                error: error.to_string(),
            });
        }
        let sink = JournalSink::create_at(
            storage,
            self.directory_id,
            reached,
            self.durability,
            self.lock.guard,
        )?;
        index.attach_sink(Box::new(sink));
        index.upgrade_checkpoint(self.path)?;
        match storage.remove_legacy_journal() {
            Ok(()) => crate::journal::sync_in_place(storage)?,
            Err(error) => warn!(target: LOG_TARGET, operation = "upgrade_old_journal_kept",
                journal = file.as_str(),
                error = %error,
                "The directory was upgraded and its journal of format version 1 could not be \
                 deleted; the directory holds every record in it, and the next load deletes it"
            ),
        }
        zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::UpgradeAfterOldRemoved);
        recovery.upgraded_from = Some(recovery.format_version.clone());
        // The upgrade's checkpoint seals a repair this load made, and one an
        // earlier load left unfinished under the directory's own snapshot.
        let repaired = plan.is_some()
            || unfinished
                .iter()
                .any(|(_, snapshot)| *snapshot == self.snapshot);
        recovery.checkpoint_taken = Some(if repaired {
            CheckpointTaken::Repair
        } else {
            CheckpointTaken::Upgrade
        });
        if repaired {
            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::RepairAfterCheckpoint);
        }
        sealed.extend(unfinished.into_iter().map(|(folder, _)| folder));
        repair::seal(&*storage.journal_dir(), &sealed)?;
        recovery.report(self.path);
        Ok((index, recovery))
    }
}

/// A journal's segments in name order: their numbers, their paths as a
/// message names them, and their bytes.
type SegmentFiles = (Vec<u64>, Vec<String>, Vec<Vec<u8>>);

/// Every segment of the journal `dir`, read whole.
fn read_segment_files(dir: &dyn zeusdb_vector_core::JournalDir) -> Result<SegmentFiles, Error> {
    let io = |what: &'static str| {
        move |error: std::io::Error| Error::JournalIoFailed {
            path: dir.path().to_path_buf(),
            what,
            error: error.to_string(),
        }
    };
    let numbers = dir.segments().map_err(io("list"))?;
    let mut files = Vec::with_capacity(numbers.len());
    let mut bytes = Vec::with_capacity(numbers.len());
    for &number in &numbers {
        files.push(dir.segment_path(number).display().to_string());
        bytes.push(dir.read_segment(number).map_err(io("read"))?);
    }
    Ok((numbers, files, bytes))
}

/// The segments `read_segment_files` read, as the reader borrows them.
fn segment_bytes<'a>(
    numbers: &[u64],
    files: &'a [String],
    bytes: &'a [Vec<u8>],
) -> Vec<SegmentBytes<'a>> {
    numbers
        .iter()
        .zip(files)
        .zip(bytes)
        .map(|((number, file), bytes)| SegmentBytes {
            number: *number,
            file,
            bytes,
        })
        .collect()
}

/// A record a replay could not apply: the refusal a load that does not
/// repair gives for it, and the kind of damage a repair names it, `decode`
/// or `apply`.
struct Unapplied {
    error: Error,
    kind: &'static str,
}

impl Unapplied {
    /// The cut a repair makes for the record at `place` among `records`,
    /// whose call begins at `start`, being that call's first record, so no
    /// call is applied in part. `numbers` are the segments' numbers.
    fn cut_before(
        &self,
        records: &[LiveRecord<'_>],
        start: usize,
        place: usize,
        numbers: &[u64],
    ) -> Cut {
        let segment_of = |sequence: u64| {
            numbers[numbers
                .partition_point(|&number| number <= sequence)
                .saturating_sub(1)]
        };
        let first = &records[start].record;
        let damaged = &records[place].record;
        Cut {
            segment: segment_of(first.sequence),
            at: first.offset,
            sequence: first.sequence,
            kind: self.kind,
            damage_segment: segment_of(damaged.sequence),
            damage_at: damaged.offset,
            damage_sequence: damaged.sequence,
            detail: self.error.to_string(),
        }
    }
}

/// Apply the records of a journal of segments in order, a committed one at
/// the values it names and a withdrawn one as the live collection left it.
/// Returns the place of the first record that would not decode or apply,
/// with how, and stops there.
fn replay_live(
    index: &Collection,
    records: &[LiveRecord<'_>],
    numbers: &[u64],
    files: &[String],
) -> Result<(), (usize, Unapplied)> {
    let dim = index.dim();
    for (place, live) in records.iter().enumerate() {
        let file = segment_file(numbers, files, live.record.sequence);
        let operation = Operation::decode(&live.record, dim, file).map_err(|error| {
            (
                place,
                Unapplied {
                    error,
                    kind: "decode",
                },
            )
        })?;
        let applied = match live.state {
            RecordState::Committed => index.apply(operation),
            RecordState::Withdrawn => index.apply_withdrawn(operation),
        };
        applied.map_err(|error| {
            let error = Error::JournalReplayFailed {
                file: file.to_string(),
                sequence: live.record.sequence,
                detail: error.replay_detail(),
            };
            (
                place,
                Unapplied {
                    error,
                    kind: "apply",
                },
            )
        })?;
    }
    Ok(())
}

/// Apply the records of a journal of format version 1 above `checkpoint`
/// in order, as the release that wrote them applied them. Returns the place
/// of the first record that would not decode or apply, with how, and stops
/// there.
fn replay_legacy_records(
    index: &Collection,
    records: &[JournalRecord<'_>],
    checkpoint: u64,
    file: &str,
) -> Result<(), (usize, Unapplied)> {
    let dim = index.dim();
    for (place, record) in records.iter().enumerate() {
        if record.sequence <= checkpoint {
            continue;
        }
        let operation = Operation::decode_legacy(record, dim, file).map_err(|error| {
            (
                place,
                Unapplied {
                    error,
                    kind: "decode",
                },
            )
        })?;
        index.apply_legacy(operation).map_err(|error| {
            let error = Error::JournalReplayFailed {
                file: file.to_string(),
                sequence: record.sequence,
                detail: error.replay_detail(),
            };
            (
                place,
                Unapplied {
                    error,
                    kind: "apply",
                },
            )
        })?;
    }
    Ok(())
}

/// The segment's file a record of `sequence` sits in, being the last
/// segment named at or below it.
fn segment_file<'a>(numbers: &[u64], files: &'a [String], sequence: u64) -> &'a str {
    let after = numbers.partition_point(|&number| number <= sequence);
    after
        .checked_sub(1)
        .and_then(|at| files.get(at))
        .map_or("", String::as_str)
}

/// The collection a load hands back read only, where the medium refused its
/// first write or the directory holds features this build does not know and
/// its manifest marks `read_only`: every record replayed, nothing on the
/// medium changed, and a sink attached that refuses every mutation.
/// `reached` is the last sequence the replay reached, which a save into
/// another directory records.
fn read_only(
    index: Collection,
    path: &str,
    cause: ReadOnlyCause,
    reached: u64,
    mut recovery: Recovery,
) -> (Collection, Recovery) {
    index.set_journal_sequence(reached);
    recovery.read_only = Some(cause.reason());
    if let ReadOnlyCause::Features(features) = &cause {
        recovery.read_only_features = features.clone();
    }
    index.attach_sink(Box::new(ReadOnlySink::new(
        std::path::PathBuf::from(path),
        cause,
        reached,
    )));
    recovery.report(path);
    (index, recovery)
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

    // Every record's internal id and id, read out of the id store and the
    // dense index under their guards, which end with this block, in the
    // order the store holds them, which is position order and increasing
    // internal id. A record is named by the internal id its node was inserted
    // under, since a position is never written. The frame is the one buffer,
    // so the file is written with nothing held.
    let (mappings_data, mapping_count) = {
        let ids = index.ids();
        let dense = index.dense_index();
        let internal = dense.ids_by_position();
        let records = ids
            .iter()
            .filter_map(|(at, id)| Some((internal.get(at)?, id)));
        (
            framed::write_mappings(records, ids.len(), ids.text_bytes()),
            ids.len(),
        )
    };
    let mappings_data = mappings_data?;

    write_framed(dir, "mappings.bin", &mappings_data, ledger)?;

    debug!(target: LOG_TARGET, "mappings.bin saved ({} mappings)", mapping_count);
    Ok(())
}

/// Save vector metadata as JSON for external tool compatibility
///
/// Compact, and naming only the records that hold a field, since a load
/// gives every record the file leaves out an empty entry.
fn save_metadata(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving metadata.json...");

    // Both guards end with the serialize, taken in the documented order,
    // the id store before vector_metadata. The file is keyed by external id
    // and the store by position, so every record is written under the id
    // the id store holds for it, in position order, which is increasing
    // internal id and the order the store walks in. `to_string` returns an
    // owned String, so the file is written with nothing held.
    let (metadata_json, record_count) = {
        let ids = index.ids();
        let vector_metadata = index.vector_metadata();
        let file = MetadataFile {
            records: ids
                .iter()
                .filter_map(|(slot, id)| {
                    vector_metadata
                        .get(slot)
                        .filter(|fields| !fields.is_empty())
                        .map(|fields| (id, fields))
                })
                .collect(),
        };
        let json = serde_json::to_string(&file);
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

/// `metadata.json` as it is written: one object per record holding a field,
/// under its external id, holding the record's fields, which `load_metadata`
/// reads back into a map of maps.
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

        let quant_persistence = QuantizationPersistence {
            r#type: "pq".to_string(),
            subvectors,
            bits,
            training_size: config.training_size,
            max_training_vectors: config.max_training_vectors,
            storage_mode: config.storage_mode.to_string().to_string(),
            is_trained: index.can_use_quantization(),
            training_completed_at,
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

/// Save the raw vectors a product quantized graph keeps beside its codes
///
/// Every record's internal id and raw vector, read out of the store the graph
/// is addressed against, in increasing internal id, straight into the frame.
///
/// Written only by a trained product quantized graph under
/// `quantized_with_raw`. Every other graph's rows are what it stores, which
/// the graph dump holds.
fn save_vectors(index: &Collection, dir: &dyn Dir, ledger: &mut SaveLedger) -> Result<(), Error> {
    if !index.writes_raw_vectors() {
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

/// What a manifest records that the save settles rather than reads from the
/// collection.
pub(crate) struct ManifestRecord<'a> {
    /// The identity the save records, which carries the snapshot it drew.
    pub(crate) identity: &'a crate::Identity,
    /// The journal sequence the saved state holds.
    pub(crate) sequence: u64,
    /// When the manifest was written, RFC 3339.
    pub(crate) saved_at: &'a str,
    /// Whether the manifest names the collection's journal, which a save of
    /// the collection's own directory does where it holds one and a backup's
    /// copy never does.
    pub(crate) names_journal: bool,
}

/// Write manifest.json, which is the last file a save writes
///
/// It names every other artefact, records the length and digest of each, and
/// records the directory size, being those lengths added. All three are facts
/// about files that are already on disk, which is why it is written last and
/// why there is no second pass to correct any of them. It records what
/// `record` holds too, which the save settled.
pub(crate) fn save_manifest(
    index: &Collection,
    dir: &dyn Dir,
    ledger: SaveLedger,
    record: &ManifestRecord<'_>,
) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving manifest.json...");

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
            } else {
                files_included.push("pq_centroids.bin".to_string());
            }
        }
    }

    // Add vectors.bin if it was saved
    if ledger.recorded_bytes("vectors.bin").is_some() {
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

    // Every artefact the ledger recorded, the graph dump included, because
    // they are all on disk by the time this runs. manifest.json itself is
    // not, so the figure counts the directory without it.
    let total_size_mb = ledger
        .digests
        .values()
        .fold(0u64, |total, digest| total.saturating_add(digest.bytes))
        as f64
        / MEBIBYTE;

    // The journal, where the collection holds one and the directory is its
    // own. Its two names come from the sink itself and the sequence from the
    // save, which a checkpoint took from the collection after it synced, so
    // all three describe the records the directory being written already
    // holds.
    let journal = index
        .sink_journal_names()
        .filter(|_| record.names_journal)
        .map(|(file, collection_id)| JournalManifest {
            file,
            sequence: record.sequence,
            collection_id: crate::journal::id_hex(collection_id),
        });

    // What the directory holds, each by the condition the part's own writer
    // tests: the journal record above, quantization.json in the layout its
    // writer chooses, and the space config.json declares. The identity is
    // part of the base at 5.x and is not listed.
    let space = index.sparse_named().map(|(_, space)| space);
    let held = features::Held {
        identity: false,
        int8: index.has_quantization() && int8,
        journal: journal.is_some(),
        pq: index.has_quantization() && !int8,
        sparse: space.is_some(),
        text: space.is_some_and(|space| space.text.is_some()),
    };

    let manifest = IndexManifest {
        format_version: FORMAT_VERSION.to_string(),
        features: Some(held.record()),
        identity: Some(IdentityManifest::of(record.identity, record.sequence)),
        zeusdb_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: index.created_at(),
        saved_at: record.saved_at.to_string(),
        total_vectors: vector_count,
        index_type: "HNSW".to_string(),
        has_quantization: index.has_quantization(),
        quantization_trained: index.can_use_quantization(),
        storage_mode: index.storage_mode(),
        files_included,
        files_excluded,
        file_digests: ledger.digests,
        total_size_mb,
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
