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
//! ├── quantization.json       # PQ configuration (if enabled)
//! ├── pq_codes.bin            # Quantized codes (if PQ enabled)
//! ├── pq_centroids.bin        # PQ centroids (if trained)
//! └── hnsw_index.zdbgraph     # HNSW graph topology and the point each node holds
//! ```
//!
//! ## How a save lands
//!
//! Every artefact goes into `<name>.zdbtmp` beside the target and the whole
//! directory is renamed into place at the end, so a reader sees the previous
//! index or this one and never a mixture. Replacing an existing directory needs
//! two renames rather than one, with `<name>.zdbold` holding the previous index
//! between them. See `StagingDir` for what that means on each platform and what
//! a killed process leaves behind.
//!
//! `manifest.json` records a length and a digest for every artefact it names
//! and the loader checks both before anything parses them. See
//! `ArtefactDigest`.
//!
//! The graph file is ZeusDB's own format, written and read by `graph::dump`,
//! and the loader restores the graph from it rather than rebuilding it by
//! re-inserting every record. See `Collection::restore_graph_from_dump`.
//!
//! This module and `collection::persist` call each other: `save` and `load`
//! on the collection reach `save_index`, `save_manifest`, `StagingDir` and
//! `load_index` here, and `load_index` builds a collection and restores it
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
    QuantizationConfig, StorageMode,
};
use crate::RerankCalibration;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info};
use zeusdb_vector_core::{
    checksum_of, validate_indexed_fields, Error, DUMP_FILENAME as GRAPH_DUMP_FILENAME,
    LEGACY_DUMP_FILENAMES, PQ,
};

/// The target every record this file emits carries. It is the module path
/// this file had in the binding, so a filter directive naming it still
/// matches. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::persistence";

// ============================================================================
// FORMAT VERSION
// ============================================================================

/// Version written into manifest.json by this build
///
/// Bumped from 1.0.0 because config.json now carries an index level `metadata`
/// map. The change is additive on both sides. A directory written by this build
/// still opens in 0.4.1, which ignores unknown config fields and never read the
/// version, and a directory written by 0.4.1 still opens here because the field
/// is defaulted.
const FORMAT_VERSION: &str = "1.1.0";

/// Major version this build can interpret
///
/// A minor bump is additive by construction, so any 1.x is read. A different
/// major means the layout changed in a way this build cannot reason about, and
/// guessing at it would be the silent truncation this format has already
/// suffered once.
const SUPPORTED_FORMAT_MAJOR: u32 = 1;

// ============================================================================
// THE BINCODE CLAIM BUDGET
// ============================================================================

// `dim` was bounded here first, at 65,536, on the reasoning that bounding it
// in `validate_index_parameters` would change `create()`'s documented
// contract. `create(dim=2**40)` aborted the process for the same reason a
// config naming it did, so the bound moved to `validate_index_parameters` as
// `MAX_DIM` and the README row changed with it. `load_config` calls that
// function, so the loader still refuses every width it refused, at the same
// number, with config.json named in the message.

/// The claim budget one wire byte earns, and why it is 64
///
/// `bincode::config::standard()` carries no byte limit, so `claim_container_read`
/// compiles to nothing and every container length in a decoded file goes
/// straight to the allocator. A 14 byte `mappings.bin` declaring 2^40 entries
/// asked for tens of terabytes and **aborted the process**. Measured on this
/// build, ten of the twelve container lengths across `mappings.bin`,
/// `vectors.bin`, `pq_codes.bin` and `pq_centroids.bin` abort at 2^40 and none
/// of them is bounded by anything the file has earned.
///
/// The bound has to come from the file's own length, which is the line
/// `parse_dump` draws: a header's fields are checked against the bytes the file
/// really holds, and after that every allocation is bounded by a file that
/// really is that long. Here the file is already in memory, so its length is
/// known outright.
///
/// bincode counts claimed bytes, being `len * size_of::<T>()` per container,
/// and unclaims each element as it decodes. The widest ratio of claimed bytes
/// to wire bytes across the four artefacts this build decodes is
/// `pq_centroids.bin`, a `Vec<Vec<Vec<f32>>>` whose outer container claims 24
/// bytes for an entry that costs one byte on the wire. The three maps claim 48
/// bytes for an entry that costs two. 64 is the next power of two above both,
/// so it admits every file this build writes with margin.
const CLAIM_PER_WIRE_BYTE: usize = 64;

/// Decode a bincode artefact under a budget the file's own length sets
///
/// bincode takes its limit as a const generic, so the derived budget picks a
/// rung rather than being passed. The rungs are a factor of 256 apart, so the
/// effective budget is at most 256 times the derived one. That is still a
/// multiple of the file's length rather than a free hand, and it still refuses
/// a hostile length by many orders of magnitude: the 2^40 cases above claim at
/// least a terabyte from files of a few tens of bytes.
///
/// A file long enough to need more than the top rung is one `fs::read` could
/// not have returned, so the top rung is the last arm rather than a special
/// case.
fn decode_bounded<T>(data: &[u8], file: &str) -> Result<T, Error>
where
    T: bincode::Decode<()>,
{
    use bincode::config::standard;

    let budget = data.len().saturating_mul(CLAIM_PER_WIRE_BYTE);
    let decoded = if budget <= 1 << 20 {
        bincode::decode_from_slice::<T, _>(data, standard().with_limit::<{ 1 << 20 }>())
    } else if budget <= 1 << 28 {
        bincode::decode_from_slice::<T, _>(data, standard().with_limit::<{ 1 << 28 }>())
    } else if budget <= 1 << 36 {
        bincode::decode_from_slice::<T, _>(data, standard().with_limit::<{ 1 << 36 }>())
    } else {
        bincode::decode_from_slice::<T, _>(data, standard().with_limit::<{ 1 << 44 }>())
    };

    match decoded {
        Ok((value, _)) => Ok(value),
        Err(bincode::error::DecodeError::LimitExceeded) => Err(Error::DecodeLengthExceeded {
            file: file.to_string(),
            bytes: data.len(),
        }),
        Err(e) => Err(Error::DecodeFailed {
            file: file.to_string(),
            error: e.to_string(),
        }),
    }
}

// ============================================================================
// AN ATOMIC SAVE
// ============================================================================

/// Suffix of the directory a save builds before it is moved into place.
const STAGING_SUFFIX: &str = ".zdbtmp";

/// Suffix the directory being replaced is moved aside under.
const REPLACED_SUFFIX: &str = ".zdbold";

/// A sibling of `target` carrying `suffix`, so both live on the target's volume
///
/// A rename is only cheap, and only atomic, within one volume. Staging under
/// the system temporary directory would put the new index on whichever volume
/// that is, and the move into place would then be a copy of every byte.
fn sibling(target: &Path, suffix: &str) -> Result<PathBuf, Error> {
    let name = target.file_name().ok_or_else(|| Error::TargetHasNoName {
        target: target.to_path_buf(),
    })?;
    let mut name = name.to_os_string();
    name.push(suffix);
    Ok(target.parent().unwrap_or_else(|| Path::new("")).join(name))
}

/// The directory a save builds, and the move that puts it in place
///
/// # What this buys
///
/// Every artefact used to be written straight into the target directory, one
/// `fs::write` at a time. A save interrupted part way left a directory holding
/// some of the new index and some of the old, and a save over an existing
/// directory replaced files one at a time and removed none, so a raw index
/// saved over a quantized one left `quantization.json`, `pq_centroids.bin` and
/// `pq_codes.bin` behind for ever. Only `manifest_names` kept those three from
/// being read back as part of the new index.
///
/// Here the save builds a directory from nothing and moves it in, so a stale
/// artefact cannot survive and a reader sees one whole index or the other.
///
/// # What "moves it into place" means
///
/// **It is one rename where the target does not exist, and two where it does.**
/// Neither Windows nor POSIX can rename a directory over an existing non-empty
/// directory. `rename(2)` requires the destination to be an empty directory and
/// `MoveFileExW` refuses `MOVEFILE_REPLACE_EXISTING` for directories outright,
/// so `fs::rename` fails on both platforms and there is no call in the standard
/// library that swaps two directories in one step. Linux has
/// `renameat2(RENAME_EXCHANGE)`, which std does not expose and which Windows
/// has no counterpart for.
///
/// So a save over an existing directory does this:
///
/// 1. rename the target aside to `<name>.zdbold`
/// 2. rename the staging directory to the target
/// 3. remove `<name>.zdbold`
///
/// Steps 1 and 2 are each atomic on both platforms. Between them the target
/// does not exist, which is a window of two filesystem calls with no I/O
/// between them. **A reader in that window sees no directory rather than a
/// partial one**, which is the property that matters, and a process killed in
/// it leaves the whole previous index at `<name>.zdbold`. `recover` puts that
/// back on the next save. A save to a path that holds nothing yet is step 2
/// alone, which is atomic outright.
///
/// If step 2 fails the target is renamed back from `<name>.zdbold`, so a failed
/// save leaves the previous directory where it was.
///
/// # What a killed process leaves
///
/// A leftover `<name>.zdbtmp` from a save that died before the move, and a
/// leftover `<name>.zdbold` from one that died inside the window. `recover`
/// deals with both at the start of the next save, and neither is inside the
/// index directory, so a load reads neither.
///
/// Dropping this without committing removes the staging directory, so a save
/// that fails part way cleans up after itself inside the process that started
/// it.
pub(crate) struct StagingDir {
    target: PathBuf,
    staging: PathBuf,
    replaced: PathBuf,
    committed: bool,
}

impl StagingDir {
    /// Clear what an earlier save left behind and open an empty staging
    /// directory
    pub(crate) fn open(target: &Path) -> Result<Self, Error> {
        let staging = sibling(target, STAGING_SUFFIX)?;
        let replaced = sibling(target, REPLACED_SUFFIX)?;

        Self::recover(target, &staging, &replaced)?;

        fs::create_dir_all(&staging).map_err(|e| Error::StagingCreateFailed {
            staging: staging.clone(),
            error: e.to_string(),
        })?;

        Ok(StagingDir {
            target: target.to_path_buf(),
            staging,
            replaced,
            committed: false,
        })
    }

    /// Put right whatever a killed save left behind
    ///
    /// `<name>.zdbold` present with no target is the one case that holds data:
    /// the previous save died between the two renames and that directory is the
    /// only copy of the index. It is renamed back rather than removed.
    ///
    /// `<name>.zdbold` present beside a target is the previous index after a
    /// save that finished, so it is removed.
    fn recover(target: &Path, staging: &Path, replaced: &Path) -> Result<(), Error> {
        if replaced.exists() {
            if target.exists() {
                remove_tree(replaced, "the previous index a finished save left aside")?;
            } else {
                fs::rename(replaced, target).map_err(|e| Error::RecoverRenameFailed {
                    target: target.to_path_buf(),
                    replaced: replaced.to_path_buf(),
                    error: e.to_string(),
                })?;
                info!(target: LOG_TARGET, operation = "save_recover",
                    restored = %target.display(),
                    "An interrupted save had moved the index aside; it is back in place"
                );
            }
        }
        if staging.exists() {
            remove_tree(
                staging,
                "a staging directory an interrupted save left behind",
            )?;
        }
        Ok(())
    }

    /// Where the save writes
    pub(crate) fn path(&self) -> &Path {
        &self.staging
    }

    /// Move the staged directory into place
    pub(crate) fn commit(mut self) -> Result<(), Error> {
        sync_directory(&self.staging);

        if self.target.exists() {
            fs::rename(&self.target, &self.replaced).map_err(|e| Error::MoveAsideFailed {
                target: self.target.clone(),
                error: e.to_string(),
            })?;

            if let Err(e) = fs::rename(&self.staging, &self.target) {
                // The target is empty at this point, so putting the previous
                // index back is the same rename in reverse.
                let restored = fs::rename(&self.replaced, &self.target).is_ok();
                self.committed = true;
                return Err(Error::MoveIntoPlaceFailedAfterAside {
                    target: self.target.clone(),
                    error: e.to_string(),
                    restored,
                });
            }

            remove_tree(&self.replaced, "the index this save replaced").ok();
        } else {
            fs::rename(&self.staging, &self.target).map_err(|e| Error::MoveIntoPlaceFailed {
                target: self.target.clone(),
                error: e.to_string(),
            })?;
        }

        sync_directory(self.target.parent().unwrap_or_else(|| Path::new(".")));
        self.committed = true;
        Ok(())
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}

/// Remove a directory tree, naming what it was in the failure
fn remove_tree(path: &Path, what: &'static str) -> Result<(), Error> {
    fs::remove_dir_all(path).map_err(|e| Error::RemoveTreeFailed {
        path: path.to_path_buf(),
        what,
        error: e.to_string(),
    })
}

/// Persist a directory's own entries, where the platform has a call for it
///
/// A file's bytes reaching the disk does not put its name in its directory. On
/// POSIX that needs the directory's own descriptor fsynced, which is what this
/// does, and without it a power loss can leave the renamed directory holding
/// entries that were never recorded.
///
/// **Windows has no equivalent through the standard library.** `File::open`
/// refuses a directory there, so this is a no-op, and the durability claim on
/// Windows rests on NTFS journalling the rename rather than on anything this
/// crate does. That difference is not observable from a gate that runs on
/// Windows.
///
/// Best effort on both. A filesystem that refuses the fsync is not a reason to
/// fail a save whose bytes are already written.
#[cfg(unix)]
fn sync_directory(path: &Path) {
    if let Ok(dir) = fs::File::open(path) {
        let _ = dir.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) {}

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
/// `checksum` is absent for the graph dump alone. The dump is written by
/// `graph::dump::write_dump`, which streams it and then seeks back to fill the
/// header in, so there is no single buffer to hash and a digest would mean
/// reading the largest artefact in the directory back off the disk. It carries
/// a checksum over its own header and another over its own payload, both
/// verified by `parse_dump` on every load, so a manifest digest would duplicate
/// a check the loader already makes. Its length is recorded and checked.
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
    fn record(&mut self, name: &str, bytes: u64, checksum: Option<u64>) {
        self.digests.insert(
            name.to_string(),
            ArtefactDigest {
                bytes,
                checksum: checksum.map(|sum| format!("{:016x}", sum)),
            },
        );
    }

    /// Record an artefact this module did not write, being the graph dump
    pub(crate) fn record_length(&mut self, name: &str, bytes: u64) {
        self.record(name, bytes, None);
    }
}

/// Write one artefact into the staging directory and record what went in
///
/// The file is fsynced before this returns. Without it the rename that moves
/// the staging directory into place can be recorded while the bytes it names
/// are still in the page cache, so a power loss leaves an index directory whose
/// manifest is complete and whose artefacts are empty. Every byte is already in
/// memory, so the fsync is the whole cost of that durability and it is measured
/// rather than assumed.
fn write_artefact(
    dir: &Path,
    name: &str,
    bytes: &[u8],
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    use std::io::Write;

    let path = dir.join(name);
    let mut file = fs::File::create(&path).map_err(|e| Error::ArtefactCreateFailed {
        name: name.to_string(),
        error: e.to_string(),
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| Error::ArtefactWriteFailed {
            name: name.to_string(),
            error: e.to_string(),
        })?;

    ledger.record(name, bytes.len() as u64, Some(checksum_of(bytes)));
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
fn read_artefact(path: &Path, name: &str, manifest: &IndexManifest) -> Result<Vec<u8>, Error> {
    let bytes = fs::read(path.join(name)).map_err(|e| Error::ArtefactReadFailed {
        name: name.to_string(),
        error: e.to_string(),
    })?;
    verify_artefact(name, &bytes, manifest)?;
    Ok(bytes)
}

/// The same, for the artefacts that are JSON
fn read_artefact_string(
    path: &Path,
    name: &str,
    manifest: &IndexManifest,
) -> Result<String, Error> {
    let bytes = read_artefact(path, name, manifest)?;
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

/// Refuse a directory this build cannot interpret
fn check_format_version(format_version: &str) -> Result<(), Error> {
    let major = format_version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
        .ok_or_else(|| Error::FormatVersionUnparsable {
            format_version: format_version.to_string(),
            current: FORMAT_VERSION,
        })?;

    if major != SUPPORTED_FORMAT_MAJOR {
        return Err(Error::FormatVersionUnsupported {
            format_version: format_version.to_string(),
            supported: SUPPORTED_FORMAT_MAJOR,
            newer: major > SUPPORTED_FORMAT_MAJOR,
        });
    }

    Ok(())
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
        "quantization.json" => "the product quantization configuration and the training state",
        "pq_centroids.bin" => "the trained PQ codebook, which every stored code decodes through",
        "pq_codes.bin" => "the quantized code of every record",
        _ => "a component of the saved index that this build does not recognise",
    }
}

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
fn check_files_present(path: &Path, manifest: &IndexManifest) -> Result<(), Error> {
    let missing: Vec<&str> = manifest
        .files_included
        .iter()
        .map(String::as_str)
        .filter(|name| !is_derived_artefact(name))
        .filter(|name| !path.join(name).exists())
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
    /// existed loads with an empty map instead of failing to parse.
    #[serde(default)]
    pub(crate) metadata: HashMap<String, String>,

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

/// ID mappings between external and internal IDs
#[derive(Debug, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub(crate) struct IdMappings {
    pub(crate) id_map: HashMap<String, usize>,
    pub(crate) rev_map: HashMap<usize, String>,
}

/// PQ codebook laid out as [subvector][centroid][dimension within subvector]
type Centroids = Vec<Vec<Vec<f32>>>;

/// Everything the loader reads back for a quantized index
struct QuantizationArtefacts {
    config: QuantizationPersistence,
    centroids: Option<Centroids>,
    codes: HashMap<String, Vec<u8>>,
}

/// Training collection state, held back until after the graph rebuild
///
/// The rebuild re-adds every record through `add(overwrite=true)`, and every id
/// is already in the restored mapping, so each one goes through
/// `remove_point_internal` first. That strips the id from `training_ids`, and
/// re-insertion cannot refill the list because collection is suppressed during
/// a rebuild. Applying the collected ids afterwards is what makes them survive.
struct TrainingState {
    ids: Vec<String>,
    threshold_reached: bool,
    is_trained: bool,
    training_size: usize,
}

impl TrainingState {
    fn from(config: &QuantizationPersistence) -> Self {
        TrainingState {
            ids: config.training_ids.clone(),
            threshold_reached: config.training_threshold_reached,
            is_trained: config.is_trained,
            training_size: config.training_size,
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

        debug!(target: LOG_TARGET, "Training state restored ({} collected ids, threshold reached: {})",
            collected, reached
        );
    }
}

// ============================================================================
// INDIVIDUAL COMPONENT LOADERS
// ============================================================================

/// Load index configuration from config.json
fn load_config(path: &Path, manifest: &IndexManifest) -> Result<IndexConfig, Error> {
    debug!(target: LOG_TARGET, "Loading config.json...");

    let config_path = path.join("config.json");
    let config_data = read_artefact_string(path, "config.json", manifest)?;

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
        &format!("{}: ", config_path.display()),
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
            file: config_path.display().to_string(),
            id_counter: config.id_counter,
        });
    }
    // The declaration too, for the same reason. A config naming a field twice,
    // or naming a reserved filter key, would build a store the index could not
    // use, and the failure would surface as a filter that quietly walked.
    validate_indexed_fields(
        &config.indexed_fields,
        &format!("{}: ", config_path.display()),
    )?;

    debug!(target: LOG_TARGET, "config.json loaded");
    Ok(config)
}

/// Load ID mappings from mappings.bin
fn load_mappings(path: &Path, manifest: &IndexManifest) -> Result<IdMappings, Error> {
    debug!(target: LOG_TARGET, "Loading mappings.bin...");

    let mappings_data = read_artefact(path, "mappings.bin", manifest)?;

    let mappings: IdMappings = decode_bounded(&mappings_data, "mappings.bin")?;

    debug!(target: LOG_TARGET, "mappings.bin loaded");
    Ok(mappings)
}

/// Load vector metadata from metadata.json
fn load_metadata(
    path: &Path,
    manifest: &IndexManifest,
) -> Result<HashMap<String, HashMap<String, Value>>, Error> {
    debug!(target: LOG_TARGET, "Loading metadata.json...");

    let metadata_data = read_artefact_string(path, "metadata.json", manifest)?;

    let metadata: HashMap<String, HashMap<String, Value>> = serde_json::from_str(&metadata_data)
        .map_err(|e| Error::ArtefactParseFailed {
            name: "metadata.json",
            error: e.to_string(),
        })?;

    debug!(target: LOG_TARGET, "metadata.json loaded");
    Ok(metadata)
}

/// Load raw vectors from vectors.bin
///
/// Read only when the manifest names it. A trained `quantized_only` index
/// writes none, and a directory saved over one that did keeps the file the
/// earlier save left. See `manifest_names`.
fn load_vectors(path: &Path, manifest: &IndexManifest) -> Result<HashMap<String, Vec<f32>>, Error> {
    debug!(target: LOG_TARGET, "Loading vectors.bin...");

    if !manifest_names(manifest, "vectors.bin") {
        debug!(target: LOG_TARGET, "manifest.json does not list vectors.bin, so no raw vectors are read");
        return Ok(HashMap::new());
    }

    let vectors_data = read_artefact(path, "vectors.bin", manifest)?;

    let vectors: HashMap<String, Vec<f32>> = decode_bounded(&vectors_data, "vectors.bin")?;

    check_vectors_are_finite(&vectors)?;

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

/// Load manifest for validation and metadata
fn load_manifest(path: &Path) -> Result<IndexManifest, Error> {
    debug!(target: LOG_TARGET, "Loading manifest.json...");

    let manifest_path = path.join("manifest.json");
    let manifest_data =
        fs::read_to_string(&manifest_path).map_err(|e| Error::ArtefactReadFailed {
            name: "manifest.json".to_string(),
            error: e.to_string(),
        })?;

    let manifest: IndexManifest =
        serde_json::from_str(&manifest_data).map_err(|e| Error::ArtefactParseFailed {
            name: "manifest.json",
            error: e.to_string(),
        })?;

    debug!(target: LOG_TARGET, "manifest.json loaded");
    Ok(manifest)
}

/// Load the PQ codebook from pq_centroids.bin
///
/// Absent means the index was saved before training completed, which is a
/// legitimate state. A present but unreadable file is a hard failure, because
/// the alternative is a codebook that decodes every code to the zero vector.
fn load_pq_centroids(path: &Path, manifest: &IndexManifest) -> Result<Option<Centroids>, Error> {
    if !manifest_names(manifest, "pq_centroids.bin") {
        return Ok(None);
    }

    debug!(target: LOG_TARGET, "Loading pq_centroids.bin...");

    let centroids_data = read_artefact(path, "pq_centroids.bin", manifest)?;

    let centroids: Centroids = decode_bounded(&centroids_data, "pq_centroids.bin")?;

    debug!(target: LOG_TARGET, "pq_centroids.bin loaded ({} subvectors)", centroids.len());
    Ok(Some(centroids))
}

/// Load the quantized codes from pq_codes.bin
///
/// Absent means no record has been quantized yet. In `quantized_only` these
/// codes are the only copy of every record added after training completed.
fn load_pq_codes(path: &Path, manifest: &IndexManifest) -> Result<HashMap<String, Vec<u8>>, Error> {
    if !manifest_names(manifest, "pq_codes.bin") {
        return Ok(HashMap::new());
    }

    debug!(target: LOG_TARGET, "Loading pq_codes.bin...");

    let codes_data = read_artefact(path, "pq_codes.bin", manifest)?;

    let codes: HashMap<String, Vec<u8>> = decode_bounded(&codes_data, "pq_codes.bin")?;

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
    path: &Path,
    manifest: &IndexManifest,
    dim: usize,
    space: &str,
) -> Result<Option<QuantizationArtefacts>, Error> {
    debug!(target: LOG_TARGET, "Loading quantization components...");

    let quant_path = path.join("quantization.json");
    if !manifest_names(manifest, "quantization.json") {
        debug!(target: LOG_TARGET, "manifest.json does not list quantization.json (non-quantized index)");
        return Ok(None);
    }

    // A directory whose config.json names the inner product space and whose
    // manifest names quantization.json describes an index `create()` refuses.
    // No save this build makes can produce one, so it was hand assembled, and
    // building it would give an index ranking by the wrong quantity.
    validate_space_supports_quantization(space, &format!("{}: ", path.display()))?;

    let quant_data = read_artefact_string(path, "quantization.json", manifest)?;

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
    validate_quantization_fields(&quant_path.display().to_string(), &quant_config, dim)?;

    debug!(target: LOG_TARGET, "quantization.json loaded");

    let centroids = load_pq_centroids(path, manifest)?;
    let codes = load_pq_codes(path, manifest)?;

    Ok(Some(QuantizationArtefacts {
        config: quant_config,
        centroids,
        codes,
    }))
}

// ============================================================================
// MAIN PERSISTENCE INTERFACE
// ============================================================================

/// Write every artefact except the graph dump and the manifest into `dir`
///
/// `dir` is the staging directory `StagingDir` opened, never the target, so
/// nothing here can leave a half written file where a reader will find it.
///
/// The manifest is no longer written here. It is written last of all, after the
/// graph dump, because it now records a length and a digest per artefact and
/// cannot do that for a file that does not exist yet. That ordering used to be
/// impossible: a save that failed at the dump would have left a directory with
/// no manifest at all. Staging makes it safe, because a save that fails at any
/// point leaves the previous directory untouched and the staging directory
/// removed.
pub(crate) fn save_index(index: &Collection, dir: &Path) -> Result<SaveLedger, Error> {
    let mut ledger = SaveLedger::default();

    // Save components in order of complexity (simple -> complex)
    save_config(index, dir, &mut ledger)?;
    save_mappings(index, dir, &mut ledger)?;
    save_metadata(index, dir, &mut ledger)?;

    // Save quantization components if enabled
    if index.has_quantization() {
        save_quantization_config(index, dir, &mut ledger)?;

        if index.can_use_quantization() {
            save_pq_centroids(index, dir, &mut ledger)?;
            save_pq_codes(index, dir, &mut ledger)?;
        }
    }

    // Save vectors based on storage mode
    save_vectors(index, dir, &mut ledger)?;

    Ok(ledger)
}

// ============================================================================
// RECONSTRUCTION FUNCTIONS
// ============================================================================

/// Reconstruct Collection using Simple Reconstruction
fn reconstruct_index_simple(
    path: &Path,
    config: IndexConfig,
    mappings: IdMappings,
    metadata: HashMap<String, HashMap<String, Value>>,
    vectors: HashMap<String, Vec<f32>>,
    quantization: Option<QuantizationArtefacts>,
    dump_bytes: Option<u64>,
) -> Result<Collection, Error> {
    debug!(target: LOG_TARGET, "Creating empty index with loaded configuration...");

    // Step 1: Create empty index with loaded config
    let mut index = Collection::new_empty(
        config.dim,
        config.space.clone(),
        config.m,
        config.ef_construction,
        config.expected_size,
        config.indexed_fields.clone(),
    );

    debug!(target: LOG_TARGET, "Restoring data fields...");

    // The codes are needed twice, once to rebuild the graph for records that
    // have no raw vector and once to restore the stored codes afterwards.
    let pq_codes = quantization
        .as_ref()
        .map(|q| q.codes.clone())
        .unwrap_or_default();
    let training_state = quantization
        .as_ref()
        .map(|q| TrainingState::from(&q.config));

    // Step 2: Restore all data fields directly (but not the graph)
    restore_data_fields(
        &mut index,
        mappings,
        metadata.clone(),
        vectors.clone(),
        &config,
        quantization,
    )?;

    // Step 3: Restore the graph the save wrote. Every save dumps the graph, so
    // the ordinary path is a read. A dump that is absent, damaged, or written
    // by a release this build cannot interpret falls back to the rebuild, which
    // is the path that also upgrades a graph carrying a defect the vendored
    // patches have since fixed. See `restore_graph_from_dump`.
    match index.restore_graph_from_dump(path, config.id_counter, dump_bytes) {
        Ok(nodes) => {
            debug!(target: LOG_TARGET, "HNSW graph restored from the saved dump ({} nodes)", nodes);
        }
        Err(reason) => {
            debug!(target: LOG_TARGET, "Rebuilding the HNSW graph, because {}", reason);
            if index.can_use_quantization() {
                debug!(target: LOG_TARGET, "Rebuilding quantized HNSW graph from stored PQ codes...");
                rebuild_graph_from_codes(&mut index, &pq_codes, &vectors)?;
            } else {
                debug!(target: LOG_TARGET, "Rebuilding HNSW graph from vectors...");
                rebuild_graph_from_data(&mut index, &vectors, &pq_codes, &metadata)?;
            }
        }
    }

    // Step 4: Put the stored record data back exactly as it was written. The
    // quantized rebuild never touches the storage maps, and the raw rebuild
    // routes through add(), which stores whatever vector it was given, so
    // without this a reconstructed record would be kept at full width and
    // quantized_only would lose the memory saving that is its whole purpose.
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
            .into_iter()
            .partition(|(id, _)| !pq_codes.contains_key(id));
        if !dropped.is_empty() {
            debug!(target: LOG_TARGET, "Released {} raw training vectors quantized_only no longer keeps",
                dropped.len()
            );
        }
        kept
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
        let placed = index
            .restore_raw_store(&vectors)
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
    Ok(index)
}

/// Reconcile the stored vector count with the records that were restored
///
/// `vector_count` is written to config.json and was previously restored
/// verbatim, so it could report records the directory no longer contains. The
/// count is derived here from the restored data and asserted against the saved
/// value. They agree for every directory whose files are intact, so a
/// disagreement means a file is missing or truncated and the load fails rather
/// than producing an index that misreports what it holds.
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
    mappings: IdMappings,
    _metadata: HashMap<String, HashMap<String, Value>>,
    _vectors: HashMap<String, Vec<f32>>,
    config: &IndexConfig,
    quantization: Option<QuantizationArtefacts>,
) -> Result<(), Error> {
    // Before the mappings move, because this reads their keys. The floor is
    // what stops an old directory reissuing a generated id it already holds.
    let generated_floor = Collection::highest_generated_id(mappings.id_map.keys());
    index.set_id_mappings(mappings.id_map, mappings.rev_map);

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
        index.add_metadata(config.metadata.clone());
        debug!(target: LOG_TARGET, "Index level metadata restored ({} entries)",
            config.metadata.len()
        );
    }

    // Restore quantization state if present
    if let Some(artefacts) = quantization {
        restore_quantization_state_simple(index, artefacts.config, artefacts.centroids)?;
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
    quant_data: QuantizationPersistence,
    centroids: Option<Centroids>,
) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Restoring quantization state...");

    // Convert QuantizationPersistence back to QuantizationConfig
    let storage_mode = StorageMode::from_string(&quant_data.storage_mode).map_err(Error::Engine)?;

    let quant_config = QuantizationConfig {
        subvectors: quant_data.subvectors,
        bits: quant_data.bits,
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

/// Rebuild the HNSW graph by re-inserting every record using existing add logic
///
/// This is the path for an index that is not trained, meaning one saved with no
/// quantization at all or one saved while still collecting training vectors.
/// A record that has a raw vector is replayed from it. A record that has only
/// PQ codes is reconstructed through the codebook, which is what `get_records`
/// already does for the same record while the index is live, so the graph is
/// built at the fidelity the storage mode already delivers rather than losing
/// the record. The codes themselves are restored as stored and are never
/// recomputed from a reconstruction.
fn rebuild_graph_from_data(
    index: &mut Collection,
    vectors: &HashMap<String, Vec<f32>>,
    pq_codes: &HashMap<String, Vec<u8>>,
    metadata: &HashMap<String, HashMap<String, Value>>,
) -> Result<(), Error> {
    if vectors.is_empty() && pq_codes.is_empty() {
        debug!(target: LOG_TARGET, "No records to rebuild (empty index)");
        return Ok(());
    }

    // Prepare batch data for efficient insertion
    let mut batch_vectors: Vec<Vec<f32>> = Vec::new();
    let mut batch_ids: Vec<String> = Vec::new();
    let mut batch_metadatas: Vec<HashMap<String, Value>> = Vec::new();
    let mut reconstructed = 0usize;
    let mut missing_metadata = 0usize;

    // Every record with a raw vector, replayed from it
    for (ext_id, vector) in vectors.iter() {
        if !metadata.contains_key(ext_id) {
            missing_metadata += 1;
        }
        batch_vectors.push(vector.clone());
        batch_ids.push(ext_id.clone());
        batch_metadatas.push(metadata.get(ext_id).cloned().unwrap_or_default());
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

            if !metadata.contains_key(ext_id) {
                missing_metadata += 1;
            }
            batch_vectors.push(vector);
            batch_ids.push(ext_id.clone());
            batch_metadatas.push(metadata.get(ext_id).cloned().unwrap_or_default());
            reconstructed += 1;
        }
    }

    if missing_metadata > 0 {
        debug!(target: LOG_TARGET, "{} records have no entry in metadata.json and are restored with empty metadata",
            missing_metadata
        );
    }

    // Insert in the order the records were first added rather than in the order
    // a hash map hands them out. A HashMap's iteration order varies per process,
    // so two rebuilds of one directory used to produce two differently wired
    // graphs that answered the same query differently. Internal ids are handed
    // out in arrival order, so sorting on them also puts the rebuild as close to
    // the original build as a rebuild can get.
    //
    // The sort is over the whole batch, so the records reconstructed from codes
    // are still ordered against the ones replayed from raw vectors rather than
    // appended after them. Zipping the three lists together first is what makes
    // that one sort rather than three index permutations of three clones.
    let mut records: Vec<(String, Vec<f32>, HashMap<String, Value>)> = batch_ids
        .into_iter()
        .zip(batch_vectors)
        .zip(batch_metadatas)
        .map(|((id, vector), metadata)| (id, vector, metadata))
        .collect();
    {
        let id_map = index.id_map();
        records.sort_by(|a, b| {
            let left = id_map.get(&a.0).copied().unwrap_or(usize::MAX);
            let right = id_map.get(&b.0).copied().unwrap_or(usize::MAX);
            left.cmp(&right).then_with(|| a.0.cmp(&b.0))
        });
    }

    debug!(target: LOG_TARGET, "Prepared {} records for batch insertion ({} replayed from raw vectors, {} reconstructed from PQ codes)",
        records.len(),
        records.len() - reconstructed,
        reconstructed
    );
    debug!(target: LOG_TARGET, "Rebuilding the graph from the restored records...");

    // The records are owned Rust and go straight into the insertion phase. This
    // used to build a PyDict holding three PyLists and call add(), which parsed
    // them back into exactly this. `rebuild_from_records` holds the flag
    // pairing, releases the interpreter lock and refuses a partial graph.
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
pub(crate) fn load_index(path: &str) -> Result<Collection, Error> {
    debug!(target: LOG_TARGET, "Starting index load with reconstruction from: {}", path);

    let path_buf = Path::new(path);

    // Validate directory exists
    if !path_buf.exists() {
        return Err(Error::IndexDirectoryNotFound {
            path: path.to_string(),
        });
    }

    // Phase 1: Load all ZeusDB components
    debug!(target: LOG_TARGET, "Phase 1: Loading ZeusDB components...");

    let manifest = load_manifest(path_buf)?;
    check_format_version(&manifest.format_version)?;

    // Before any artefact is read, so a directory missing two files names the
    // first rather than failing on whichever one a partial load reaches.
    check_files_present(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Manifest loaded: {} vectors, format v{}",
        manifest.total_vectors, manifest.format_version
    );

    let config = load_config(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Config loaded: dim={}, space={}", config.dim, config.space);

    let mappings = load_mappings(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Mappings loaded: {} ID mappings", mappings.id_map.len());

    let metadata = load_metadata(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Metadata loaded: {} records", metadata.len());

    let vectors = load_vectors(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Vectors loaded: {} vectors", vectors.len());

    let quantization = load_quantization(path_buf, &manifest, config.dim, &config.space)?;
    if let Some(ref quant) = quantization {
        debug!(target: LOG_TARGET, "Quantization loaded: {} subvectors, trained={}, codebook={}",
            quant.config.subvectors,
            quant.config.is_trained,
            if quant.centroids.is_some() {
                "present"
            } else {
                "absent"
            }
        );
    }

    // The graph dump itself is read inside the reconstruction, which needs the
    // mappings and the codebook first to judge it against.

    // Phase 2: Create empty index and restore state
    debug!(target: LOG_TARGET, "Phase 2: Creating empty index and restoring state...");
    let mut restored_index = reconstruct_index_simple(
        path_buf,
        config,
        mappings,
        metadata,
        vectors,
        quantization,
        recorded_dump_length(&manifest, GRAPH_DUMP_FILENAME),
    )?;

    // `new_empty` stamps the load time, because it has nothing better to start
    // from. Until this ran, a save of a loaded index wrote that load time to
    // manifest.json as the creation, so a directory that had been through one
    // load and save claimed to have been created then.
    restored_index.set_created_at(manifest.created_at);

    debug!(target: LOG_TARGET, "Index reconstruction completed successfully!");
    Ok(restored_index)
}

// ============================================================================
// INDIVIDUAL COMPONENT SAVERS
// ============================================================================

/// Save index configuration as JSON
fn save_config(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
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
        metadata: index.all_metadata(),
        indexed_fields: index.indexed_fields(),
    };

    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| Error::SerializeFailed {
            what: "config",
            error: e.to_string(),
        })?;

    write_artefact(path, "config.json", config_json.as_bytes(), ledger)?;

    debug!(target: LOG_TARGET, "config.json saved");
    Ok(())
}

/// Save ID mappings using efficient binary format
fn save_mappings(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving mappings.bin...");

    // Both guards end with this block. They are taken in the documented order,
    // id_map before rev_map, and the copy they were already making is what the
    // rest of the function works from.
    let mappings = {
        let id_map = index.id_map();
        let rev_map = index.rev_map();
        IdMappings {
            id_map: id_map.clone(),
            rev_map: rev_map.map().clone(),
        }
    };
    let mapping_count = mappings.id_map.len();

    let mappings_data =
        bincode::encode_to_vec(&mappings, bincode::config::standard()).map_err(|e| {
            Error::SerializeFailed {
                what: "mappings",
                error: e.to_string(),
            }
        })?;

    write_artefact(path, "mappings.bin", &mappings_data, ledger)?;

    debug!(target: LOG_TARGET, "mappings.bin saved ({} mappings)", mapping_count);
    Ok(())
}

/// Save vector metadata as JSON for external tool compatibility
fn save_metadata(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving metadata.json...");

    // The guard ends with the serialize. `to_string_pretty` returns an owned
    // String, so the file is written with nothing held and nothing is copied
    // that was not being copied already.
    let (metadata_json, record_count) = {
        let vector_metadata = index.vector_metadata();
        let json = serde_json::to_string_pretty(&*vector_metadata);
        (json, vector_metadata.len())
    };
    let metadata_json = metadata_json.map_err(|e| Error::SerializeFailed {
        what: "metadata",
        error: e.to_string(),
    })?;

    write_artefact(path, "metadata.json", metadata_json.as_bytes(), ledger)?;

    debug!(target: LOG_TARGET, "metadata.json saved ({} records)", record_count);
    Ok(())
}

/// Save quantization configuration and training state
fn save_quantization_config(
    index: &Collection,
    path: &Path,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    if let Some(config) = index.quantization_config() {
        debug!(target: LOG_TARGET, "Saving quantization.json...");

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
                sub_dim: index.dim() / config.subvectors,
                num_centroids: 1 << config.bits,
            };
            (None, pq_config)
        };

        let quant_persistence = QuantizationPersistence {
            r#type: "pq".to_string(),
            subvectors: config.subvectors,
            bits: config.bits,
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

        write_artefact(path, "quantization.json", quant_json.as_bytes(), ledger)?;

        //debug!(target: LOG_TARGET, "quantization.json saved");
        debug!(target: LOG_TARGET, "quantization.json saved with {} training IDs",
            quant_persistence.training_ids.len()
        );
    }
    Ok(())
}

/// Save PQ centroids for vector reconstruction
fn save_pq_centroids(
    index: &Collection,
    path: &Path,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    if let Some(pq) = index.pq() {
        if pq.is_trained() {
            debug!(target: LOG_TARGET, "Saving pq_centroids.bin...");

            // The codebook is serialized inside the closure and written outside
            // it, so the lock is held for the encode alone. `bincode` returns an
            // owned buffer, so narrowing the guard this way copies nothing.
            let centroids_data = pq
                .with_centroids(|centroids| {
                    bincode::encode_to_vec(centroids, bincode::config::standard())
                })
                .map_err(|e| Error::SerializeFailed {
                    what: "PQ centroids",
                    error: e.to_string(),
                })?;

            write_artefact(path, "pq_centroids.bin", &centroids_data, ledger)?;

            debug!(target: LOG_TARGET, "pq_centroids.bin saved");
        }
    }
    Ok(())
}

/// Save quantized vector codes
fn save_pq_codes(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    // The guard ends with the serialize, so the file is written with nothing
    // held. `encode_to_vec` was already producing an owned buffer, so this
    // narrows the guard rather than adding a copy.
    let (codes_data, code_count) = {
        let pq_codes = index.pq_codes();
        if pq_codes.is_empty() {
            return Ok(());
        }
        debug!(target: LOG_TARGET, "Saving pq_codes.bin...");
        (
            bincode::encode_to_vec(&*pq_codes, bincode::config::standard()),
            pq_codes.len(),
        )
    };

    let codes_data = codes_data.map_err(|e| Error::SerializeFailed {
        what: "PQ codes",
        error: e.to_string(),
    })?;

    write_artefact(path, "pq_codes.bin", &codes_data, ledger)?;

    debug!(target: LOG_TARGET, "pq_codes.bin saved ({} vectors)", code_count);
    Ok(())
}

/// Save raw vectors based on storage mode configuration
///
/// The file is what it always was, a `HashMap` keyed by external id, so a
/// directory this release writes loads on the reader that read the old ones.
/// What changed is where the vectors come from: there is no raw vector map any
/// more, so they are read out of the store the graph is addressed against.
///
/// A trained `quantized_only` index writes none, as before, because it holds
/// none.
fn save_vectors(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    if !index.holds_raw_vectors() {
        return Ok(());
    }
    let (vectors_data, vector_count) = {
        let vectors = index.collect_raw_vectors();
        if vectors.is_empty() {
            return Ok(());
        }
        debug!(target: LOG_TARGET, "Saving vectors.bin...");
        (
            bincode::encode_to_vec(&vectors, bincode::config::standard()),
            vectors.len(),
        )
    };

    let vectors_data = vectors_data.map_err(|e| Error::SerializeFailed {
        what: "vectors",
        error: e.to_string(),
    })?;

    write_artefact(path, "vectors.bin", &vectors_data, ledger)?;

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
    path: &Path,
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
    if index.has_quantization() {
        files_included.push("quantization.json".to_string());

        if index.can_use_quantization() {
            files_included.push("pq_centroids.bin".to_string());
            if code_count > 0 {
                files_included.push("pq_codes.bin".to_string());
            }
        }
    }

    // Add vectors.bin if it was saved
    if has_raw_vectors {
        files_included.push("vectors.bin".to_string());
    } else {
        files_excluded.push("vectors.bin".to_string());
    }

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
    let compression_info =
        if index.has_quantization() && index.can_use_quantization() && code_count > 0 {
            let raw_size_mb = (code_count * index.dim() * 4) as f64 / (1024.0 * 1024.0);
            let compressed_size_mb =
                (code_count * index.quantization_subvectors()) as f64 / (1024.0 * 1024.0);
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
    let total_size_mb = calculate_directory_size(path).unwrap_or(0.0);

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
    write_artefact(
        path,
        "manifest.json",
        manifest_json.as_bytes(),
        &mut discard,
    )?;

    debug!(target: LOG_TARGET, "manifest.json saved");
    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate the total size of a directory in MB
fn calculate_directory_size(path: &Path) -> Result<f64, std::io::Error> {
    let mut total_size = 0u64;

    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                total_size += metadata.len();
            }
        }
    }

    Ok(total_size as f64 / (1024.0 * 1024.0))
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
