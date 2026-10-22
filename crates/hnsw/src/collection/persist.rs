//! The face the persistence layer speaks to, and the two operations that
//! reach it.
//!
//! `persistence.rs` reads and writes the directory; this file is the only way it
//! reaches the index. Every private field it needs is behind an accessor here
//! and every field it restores is behind a setter, so the storage layer names no
//! field of `Collection` and cannot leave one in a state the index did not
//! choose. `save` and `load` are the two doors, and the binding calls both
//! with the interpreter lock released.

use super::{
    Collection, DenseIndex, DenseOpen, DenseRestored, Identity, IdentityRecord, QuantizationConfig,
    StorageMode, MAX_LAYER,
};
use crate::journal::{Durability, JournalPolicy, Recovery};
use crate::locks::ReadGuard;
use crate::RerankCalibration;
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, instrument};
use zeusdb_vector_core::{
    ArtefactRecord, Bounds, Dir, DumpRefusal, DumpRegion, DumpRows, Error, FsStorage, IdStore,
    Inventory, KillPoint, MetadataStore, Persist, Position, PositionWalk, SegmentBase, Storage,
    VectorGraph, DUMP_FILENAME,
};

/// What a checkpoint is for, which decides the kill points it passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CheckpointKind {
    /// A save, or the checkpoint of a journaled collection.
    Save,
    /// The checkpoint that writes a directory read at an older format again
    /// at this build's, with a journal of segments.
    Upgrade,
}

/// Whose directory a save writes, which decides the manifest's journal
/// record, the sequence its identity records, the kill points the save
/// passes, and whether the lineage moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Written {
    /// The collection's own: the manifest names the journal where the
    /// collection holds one and records the sequence its checkpoint holds,
    /// and the lineage moves to the directory once it commits.
    Own,
    /// A backup's copy: the manifest names no journal and records
    /// `sequence`, and the lineage stays where it was.
    Copy { sequence: u64 },
}

/// The points a crash test aborts a save at: after the artefacts, where the
/// save has one, then after the graph dump, after the manifest, between the
/// two renames and after the commit.
struct SavePoints {
    artefacts: Option<KillPoint>,
    dump: KillPoint,
    manifest: KillPoint,
    between: KillPoint,
    commit: KillPoint,
}

impl Written {
    fn points(self) -> SavePoints {
        match self {
            Written::Own => SavePoints {
                artefacts: Some(KillPoint::SaveAfterArtefacts),
                dump: KillPoint::SaveAfterDump,
                manifest: KillPoint::SaveAfterManifest,
                between: KillPoint::SaveBetweenRenames,
                commit: KillPoint::SaveAfterCommit,
            },
            Written::Copy { .. } => SavePoints {
                artefacts: None,
                dump: KillPoint::BackupAfterArtefacts,
                manifest: KillPoint::BackupAfterManifest,
                between: KillPoint::BackupBetweenRenames,
                commit: KillPoint::BackupAfterCommit,
            },
        }
    }
}

/// Refuse a target that is a file, or a folder holding entries and no
/// `manifest.json`, which a save would replace whole. Asked before anything
/// is written, the lock file included.
fn refuse_foreign_target(storage: &dyn Storage, path: &str) -> Result<(), Error> {
    match storage.foreign_target() {
        Some(holds) => Err(Error::TargetNotAnIndex {
            path: path.to_string(),
            holds,
        }),
        None => Ok(()),
    }
}
use zeusdb_vector_sparse::PostingsIndex;
use zeusdb_vector_text::{TermDictionary, Tokenizer};

/// The target every record this file emits carries. See the parent module.
const LOG_TARGET: &str = "zeusdb_vector_database::hnsw_index::persist";
/// Set to any non-empty value other than `0` to skip the saved graph and
/// rebuild it by re-inserting every record.
///
/// The rebuild is what upgrades an index whose graph was built by a release
/// carrying a defect the vendored patches have since fixed, since restoring the
/// dump restores the graph exactly as it was written, defects included. Without
/// this there is no way to ask for that upgrade on a directory whose dump is
/// perfectly readable.
const REBUILD_ENV: &str = "ZEUSDB_LOAD_REBUILD_GRAPH";
/// Whether the caller has asked for the rebuild rather than the saved graph
fn rebuild_requested() -> bool {
    match std::env::var(REBUILD_ENV) {
        Ok(value) => !value.is_empty() && value != "0",
        Err(_) => false,
    }
}

/// What restoring the graph from its dump gave.
pub(crate) enum DumpOutcome {
    /// The graph the dump holds, installed, with its node count.
    Read(usize),
    /// The rows of a version 2 dump, each region verified, which the caller
    /// rebuilds the graph from: its topology could not be used, or the
    /// rebuild was asked for, which `asked` says.
    Rows {
        rows: DumpRows,
        reason: String,
        asked: bool,
    },
    /// Below format 5.0.0, why the dump cannot be used, which the caller
    /// rebuilds the graph from the separate row stores for. Also the
    /// collection holding no record, at any format, which writes no dump.
    Rebuild(String),
}

/// What the manifest recorded about the graph dump, as the inventory the
/// dense index checks before it reads the dump. A directory written before
/// the manifest carried lengths records none, and nothing is checked.
struct DumpLength(Option<u64>);

impl Inventory for DumpLength {
    fn recorded(&self, name: &str) -> Option<ArtefactRecord> {
        if name != DUMP_FILENAME {
            return None;
        }
        self.0.map(|bytes| ArtefactRecord {
            bytes,
            checksum: None,
        })
    }
}
impl Collection {
    /// Count the records the index actually holds
    ///
    /// The union of the raw vectors and the PQ codes, because `quantized_only`
    /// keeps a record added after training in the codes alone. This is derived
    /// from the stored data rather than from the counter, so it is what the
    /// counter is checked against after a load. Crate private, since the
    /// only caller is the load path in `persistence`.
    pub(crate) fn count_stored_records(&self) -> usize {
        let ids = self.ids.read().unwrap();
        let index = self.dense().index.read().unwrap();
        let pq_codes = self.dense().pq_codes.read().unwrap();
        let graph = index.graph();
        let raws = index.raw_vectors();
        let holds_raw = |id: &str| ids.slot_of(id).is_some_and(|slot| raws.get(slot).is_some());
        let with_raw = ids.slots().filter(|&slot| raws.get(slot).is_some()).count();
        let code_only = pq_codes.keys().filter(|id| !holds_raw(id)).count();
        // A scalar graph holds its rows and nothing else holds them, so a
        // record with a row is a record stored.
        let with_row = ids
            .slots()
            .filter(|&slot| {
                index
                    .node_of(Position::from_slot(slot))
                    .is_some_and(|node| graph.int8_row_at(node).is_some())
            })
            .count();
        with_raw + code_only + with_row
    }

    /// Hand every raw vector the index holds to `write`, with its internal
    /// id, in increasing internal id, and the record count beside them.
    ///
    /// The id store's guard and the index's are held for the call and in that
    /// order, so `write` sees one instant of both. The records are walked in
    /// position order, which is internal id order, and each is named by the
    /// internal id its node was inserted under. The one caller is the save
    /// of a product quantized graph that keeps a raw vector beside every
    /// code; see `writes_raw_vectors`.
    pub(crate) fn with_raw_vectors<R>(
        &self,
        write: impl FnOnce(&mut dyn Iterator<Item = (usize, &[f32])>, usize) -> R,
    ) -> R {
        let ids = self.ids.read().unwrap();
        let index = self.dense().index.read().unwrap();
        let node_at = index.node_at();
        let graph = index.graph();
        let origin_ids = graph.origin_ids();
        let mut records = ids.slots().filter_map(|at| {
            let node = *node_at.get(at)?;
            let vector = graph.raw_vector_at(node)?;
            Some((*origin_ids.get(node as usize)?, vector))
        });
        write(&mut records, ids.len())
    }

    /// Whether this index should hold a raw store beside its codes.
    ///
    /// Asked by the loader after the graph is back and before the vectors are
    /// placed, which is the one moment the graph is quantized and the store is
    /// still empty, so `raws_need_their_own_file` would answer no.
    pub(crate) fn raw_store_is_expected(&self) -> bool {
        let quantized = self.dense().index.read().unwrap().graph().is_quantized();
        quantized
            && self
                .dense()
                .quantization_config
                .as_ref()
                .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw)
    }

    /// Whether a save writes `vectors.bin`, which is where the graph is a
    /// product quantized one keeping a raw vector beside every code, being a
    /// trained one under `quantized_with_raw`.
    ///
    /// Every other graph's raw vectors, where it keeps any, are the rows it
    /// stores, which the graph dump holds, and the dump holds the only copy
    /// of each.
    pub(crate) fn writes_raw_vectors(&self) -> bool {
        let index = self.dense().index.read().unwrap();
        let graph = index.graph();
        graph.is_pq() && graph.holds_raw()
    }

    /// Every live record's codes, read off a product quantized graph's own
    /// store and keyed by external id, at the capacity of the live count.
    /// Empty for any other graph.
    ///
    /// The codes map a load restores at format 5.0.0, where the graph dump
    /// holds the only copy of each record's codes.
    pub(crate) fn codes_from_graph(&self) -> HashMap<String, Vec<u8>> {
        let ids = self.ids.read().unwrap();
        let index = self.dense().index.read().unwrap();
        let graph = index.graph();
        if !graph.is_pq() {
            return HashMap::new();
        }
        let mut codes = HashMap::with_capacity(ids.len());
        for (at, id) in ids.iter() {
            let node = index.node_of(Position::from_slot(at));
            if let Some(code) = node.and_then(|node| graph.codes_at(node)) {
                codes.insert(id.to_string(), code.to_vec());
            }
        }
        codes
    }

    /// Rebuild the raw side store of a loaded quantized graph.
    ///
    /// The graph comes back from its dump with its own node numbering, which is
    /// the dump's layer major order rather than the order the records arrived
    /// in, so the vectors read from `vectors.bin` are placed by the position
    /// each node files its record at. A live record the file does not name is
    /// an error, because a `quantized_with_raw` index whose raws are partly
    /// missing would rerank some records and not others.
    ///
    /// **A stranded node takes a zero vector.** Removal and overwrite leave the
    /// node in the graph and take the record out of the id store, and the save
    /// writes the live records alone, so the file names fewer records than the
    /// graph has nodes. The store is addressed by node, so those nodes need a
    /// slot; nothing can reach it, because every reader resolves an external id
    /// through the id store first and a stranded node has none. `compact` drops the
    /// node and its slot together.
    pub(crate) fn restore_raw_store(
        &mut self,
        vectors: &HashMap<String, Vec<f32>>,
    ) -> Result<usize, String> {
        let ids = self.ids.read().unwrap();
        let mut index = self.dense().index.write().unwrap();
        let hnsw = index.graph_mut();
        if !hnsw.is_quantized() {
            return Ok(0);
        }
        let nodes = hnsw.nb_points();
        hnsw.open_raw_store(self.dense().dim, nodes)?;
        let mut by_position: HashMap<usize, &Vec<f32>> = HashMap::with_capacity(vectors.len());
        for (ext_id, vector) in vectors {
            if let Some(at) = ids.slot_of(ext_id) {
                by_position.insert(at, vector);
            }
        }
        let stranded_fill = vec![0f32; self.dense().dim];
        let mut placed = 0usize;
        let mut stranded = 0usize;
        for node in 0..nodes as u32 {
            let at = hnsw.record_of_at(node);
            match by_position.get(&(at as usize)) {
                Some(vector) if at != Position::NONE => {
                    hnsw.push_raw_vector(vector)?;
                    placed += 1;
                }
                _ if at == Position::NONE => {
                    hnsw.push_raw_vector(&stranded_fill)?;
                    stranded += 1;
                }
                _ => {
                    return Err(format!(
                        "live record with internal id {} has no raw vector in vectors.bin",
                        hnsw.origin_id_at(node)
                    ))
                }
            }
        }
        if stranded > 0 {
            debug!(target: LOG_TARGET, operation = "restore_raw_store",
                stranded_nodes = stranded,
                "Stranded graph nodes took a zero vector no reader can reach"
            );
        }
        Ok(placed)
    }
}

impl Collection {
    /// Save the index to a directory.
    ///
    /// Four phases under one hold of the mutation lock: the artefacts, the
    /// graph dump, the manifest, and the move into place. The binding calls
    /// this with the interpreter lock released, and every phase speaks only
    /// to `serde_json`, the engine's frame, `graph::dump` and the storage the
    /// directory is kept in, which is `zeusdb_vector_core::FsStorage`.
    pub fn save(&self, path: &str) -> Result<(), Error> {
        // A save reads the mappings, the metadata, the codes, the vectors and
        // the graph in five separate passes, so it needs the index to hold
        // still. PyO3's exclusive borrow used to guarantee that by keeping every
        // mutating method away from it. Relaxing the receivers removes that, and
        // a save overlapping an add would write a directory whose mappings and
        // vectors came from different instants. This takes the mutation lock
        // instead, which blocks a concurrent write and no reader at all.
        let _writers = self.writers.lock().unwrap();
        self.save_locked(path)
    }

    /// The save, and the checkpoint it is on a journaled collection, with
    /// the mutation guard already held.
    ///
    /// On a collection with no journal this is the four phases under the
    /// directory's lock, held exclusive for the save, and the directory it
    /// writes is the directory the four phases write.
    ///
    /// On a journaled collection three things wrap them, under the lock the
    /// journal already holds.
    ///
    /// The journal is synced first, whatever the commit mode the sink was
    /// given, and the sequence it has reached is written onto the collection
    /// so that phase three records it in the manifest. A sync that fails is
    /// logged and passed, since the directory is written from memory and
    /// holds every record whatever reached the device.
    ///
    /// Then the four phases, under the one hold this already has, so the
    /// directory that is committed holds exactly the mutations at or below
    /// the sequence it names.
    ///
    /// Then the journal rolls onto a segment naming the directory just
    /// committed and deletes every segment that directory holds. A crash
    /// before the roll leaves segments naming the previous directory, every
    /// record of which the new one holds, so a load finds them stale,
    /// creates the segment the roll would have and deletes them. A crash in
    /// the deletion leaves some of them, which a load deletes. The commit of
    /// the roll clears a fault a failed commit left.
    pub(crate) fn save_locked(&self, path: &str) -> Result<(), Error> {
        self.save_locked_as(path, CheckpointKind::Save)
    }

    /// [`Collection::save_locked`], as the save or as the checkpoint that
    /// upgrades a directory read at an older format, which passes three kill
    /// points of its own besides the save's.
    pub(crate) fn save_locked_as(&self, path: &str, kind: CheckpointKind) -> Result<(), Error> {
        let storage = FsStorage::at(Path::new(path))?;
        // A collection read only for features its directory holds saves
        // nowhere, since a directory this build wrote would not hold them.
        if self.sink_read_only_features() {
            return Err(self
                .sink_refusal()
                .expect("a sink naming read-only features refuses"));
        }
        // A collection read from a medium it cannot write saves into another
        // directory alone, which copies it somewhere writable.
        if let Some(directory) = self.sink_read_only_directory() {
            if crate::journal::same_directory(&directory, Path::new(path)) {
                return Err(self
                    .sink_refusal()
                    .expect("a sink naming a read-only directory refuses"));
            }
        }
        // A checkpoint saves to the directory its journal sits beside and to
        // no other, because the manifest it writes names that journal as a
        // sibling and a directory whose sibling is elsewhere is one nothing
        // can open.
        let journal = self.sink_journal_path();
        if let Some(journal) = &journal {
            if !storage.names_journal(journal) {
                return Err(Error::JournalDirectoryMismatch {
                    journal: journal.display().to_string(),
                    target: path.to_string(),
                });
            }
        }
        // The manifest names the collection twice where the sink names a
        // journal, in its identity and in its journal record, and a load
        // refuses a manifest whose two disagree. A sink naming another
        // collection is refused here, before anything is written.
        if let Some((file, journal_id)) = self.sink_journal_names() {
            if journal_id != self.collection_id {
                return Err(Error::JournalNotThisCollection {
                    file,
                    journal_id: crate::journal::id_hex(journal_id),
                    directory_id: crate::journal::id_hex(self.collection_id),
                });
            }
        }
        refuse_foreign_target(&storage, path)?;
        // The directory's lock. A journal holds it for the collection's life,
        // and any other save takes it for the save, so a second writer in
        // this process or another is refused before anything is written.
        let _lock = match journal {
            Some(_) => None,
            None => crate::journal::lock_for_save(&storage, path)?,
        };
        // Every record handed over is synced before the directory that will
        // claim them is written, where the device allows.
        if let Some(reached) = self.sync_sink_locked(false)? {
            self.set_journal_sequence(reached);
            debug!(target: LOG_TARGET, operation = "checkpoint_sync",
                sequence = reached,
                "The journal is synced and the checkpoint will name its sequence"
            );
        }
        zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::CheckpointBeforeSave);
        self.save_phases(&storage, path)?;
        zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::CheckpointAfterSaveBeforeRoll);
        if kind == CheckpointKind::Upgrade {
            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::UpgradeAfterCommit);
        }
        // The directory now holds every record at or below the sequence it
        // names, so the journal goes on after it and holds none of them.
        if journal.is_some() {
            let identity = self.identity();
            let base = SegmentBase {
                snapshot: identity
                    .snapshot
                    .expect("a committed save records the snapshot it drew"),
                sequence: self.journal_sequence(),
                generation: identity.generation,
                parent: identity.parent.unwrap_or(0),
            };
            self.roll_sink_locked(base)?;
            if kind == CheckpointKind::Upgrade {
                zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::UpgradeAfterSegment);
            }
        }
        Ok(())
    }

    /// The checkpoint that upgrades a directory read at an older format,
    /// with the mutation guard taken here. The load calls it once the sink
    /// over the new journal is attached.
    pub(crate) fn upgrade_checkpoint(&self, path: &str) -> Result<(), Error> {
        let _writers = self.writers.lock().unwrap();
        self.save_locked_as(path, CheckpointKind::Upgrade)
    }

    /// The four phases, with the mutation guard already held.
    fn save_phases(&self, storage: &FsStorage, path: &str) -> Result<(), Error> {
        let start_time = Instant::now();
        info!(target: LOG_TARGET, operation = "save_start", path = path, "Starting index save");

        self.save_to(storage, Written::Own)?;

        let duration_ms = start_time.elapsed().as_millis();
        info!(target: LOG_TARGET, operation = "save_complete",
            path = path,
            duration_ms = duration_ms,
            "Index save completed successfully"
        );
        Ok(())
    }

    /// The four phases, into the directory `storage` keeps, returning what
    /// the manifest records of the directory's identity.
    fn save_to(&self, storage: &dyn Storage, written: Written) -> Result<IdentityRecord, Error> {
        let points = written.points();
        // What the directory will be, settled before anything is written, so
        // a collection whose generation cannot move writes nothing.
        let identity = self.next_identity()?;
        let (sequence, names_journal) = match written {
            Written::Own => (self.journal_sequence(), true),
            Written::Copy { sequence } => (sequence, false),
        };

        // Every artefact goes into a directory of its own that is moved into
        // place at the end, so a reader sees the previous index or this one and
        // never a mixture of the two, and a stale artefact from an earlier save
        // cannot survive. See `Storage::stage`.
        let staging = storage.stage()?;

        // Phase 1: Save all ZeusDB components (already tested to work)
        debug!(target: LOG_TARGET, operation = "save_phase1", "Saving ZeusDB components");
        let mut ledger = crate::persistence::save_index(self, staging.dir())?;
        if let Some(point) = points.artefacts {
            zeusdb_vector_core::kill_at(point);
        }

        // Phase 2: Save the HNSW graph in ZeusDB's own format
        debug!(target: LOG_TARGET, operation = "save_phase2", "Saving HNSW graph");
        self.save_hnsw_graph(staging.dir(), &mut ledger)?;
        zeusdb_vector_core::kill_at(points.dump);

        // Phase 3: The manifest, last, because it records a length and a digest
        // for every artefact above and the directory size all of them add up
        // to.
        debug!(target: LOG_TARGET, operation = "save_phase3", "Writing the manifest");
        let saved_at = Utc::now().to_rfc3339();
        let record = crate::persistence::ManifestRecord {
            identity: &identity,
            sequence,
            saved_at: &saved_at,
            names_journal,
        };
        crate::persistence::save_manifest(self, staging.dir(), ledger, &record)?;
        zeusdb_vector_core::kill_at(points.manifest);

        // Phase 4: Two renames, or one where nothing is there yet.
        debug!(target: LOG_TARGET, operation = "save_phase4",
            "Moving the saved index into place"
        );
        staging.commit(points.between)?;
        let record = IdentityRecord {
            identity,
            saved_at: Some(saved_at),
            sequence,
        };
        // The directory is in place, so it is the one this collection last
        // committed. A save that failed before here left the lineage where
        // it was, and the next one records the same generation again. A
        // backup's copy is not the collection's directory.
        if written == Written::Own {
            self.set_lineage(&record);
        }
        zeusdb_vector_core::kill_at(points.commit);
        Ok(record)
    }

    /// Write a copy of this collection into the directory `path`, at format
    /// 5.0.0 and standalone, and return what its manifest records of the
    /// copy's identity.
    ///
    /// Under the mutation guard, so writers wait for it as they wait for a
    /// checkpoint and searches run throughout. A collection read only for
    /// features its directory holds is refused first, with
    /// [`Error::FeatureReadOnly`]. Five targets are refused
    /// before anything is written: the directory the collection's journal
    /// sits beside or a read-only collection was read from, a directory with
    /// a journal of either version beside it, a path that is not an index
    /// directory, and a directory another holder locks. The journal is
    /// synced as a checkpoint syncs it, and a sync that fails is logged and
    /// leaves the fault, since the copy is written from memory and holds
    /// every record the collection holds whatever the journal lost. Then
    /// the save's four phases under the target's lock.
    ///
    /// The copy's manifest names no journal. Its identity is the next
    /// generation under a snapshot of its own, with the collection's
    /// snapshot as its parent and, as its sequence, the sequence the
    /// journal has reached, or the collection's own where it has no
    /// journal. Neither the lineage nor the journal moves, so the next
    /// checkpoint names the same parent.
    pub fn backup(&self, path: &str) -> Result<IdentityRecord, Error> {
        let _writers = self.writers.lock().unwrap();
        // A collection read only for features its directory holds writes no
        // copy, for the reason a save into another directory gives.
        if self.sink_read_only_features() {
            return Err(self
                .sink_refusal()
                .expect("a sink naming read-only features refuses"));
        }
        let storage = FsStorage::at(Path::new(path))?;
        let own = self
            .sink_journal_path()
            .and_then(|journal| crate::journal::directory_of(&journal))
            .or_else(|| self.sink_read_only_directory());
        if own.is_some_and(|own| crate::journal::same_directory(&own, Path::new(path))) {
            return Err(Error::BackupIntoOwnDirectory {
                path: path.to_string(),
            });
        }
        let journal = if storage.journal_exists() {
            Some(storage.journal_path())
        } else if storage.legacy_journal_exists() {
            Some(storage.legacy_journal_path())
        } else {
            None
        };
        if let Some(journal) = journal {
            return Err(Error::BackupTargetJournaled {
                path: path.to_string(),
                journal: journal.display().to_string(),
            });
        }
        refuse_foreign_target(&storage, path)?;
        let _lock = crate::journal::lock_for_save(&storage, path)?;
        let sequence = match self.sync_sink_locked(true)? {
            Some(reached) => reached,
            None => self.journal_sequence(),
        };
        zeusdb_vector_core::kill_at(KillPoint::BackupAfterSync);
        let start_time = Instant::now();
        info!(target: LOG_TARGET, operation = "backup_start", path = path, "Starting backup");
        let record = self.save_to(&storage, Written::Copy { sequence })?;
        info!(target: LOG_TARGET, operation = "backup_complete",
            path = path,
            sequence = sequence,
            duration_ms = start_time.elapsed().as_millis(),
            "Backup completed successfully"
        );
        Ok(record)
    }

    /// What the next save records: this collection's id, one generation on
    /// from the directory it last committed or read, a snapshot drawn now,
    /// and that directory's snapshot as the parent.
    ///
    /// Refused only at the largest generation a manifest can hold, which no
    /// collection reaches by saving and a directory reaches only by an edit.
    fn next_identity(&self) -> Result<Identity, Error> {
        let current = self.identity();
        current.next(super::construct::draw_id()).ok_or_else(|| {
            Error::Engine(format!(
                "This collection was read from a directory recording generation {}, the \
                 largest a manifest can hold, so no save can record the one after it. No \
                 release writes that generation, so the directory's manifest was edited.",
                current.generation
            ))
        })
    }

    /// Write the HNSW graph in ZeusDB's own format. See `graph::dump`.
    #[instrument(target = LOG_TARGET, level = "info", skip(self, dir, ledger), fields(
        vector_count = self.vector_count(),
        path = %dir.locate("")
    ))]
    fn save_hnsw_graph(
        &self,
        dir: &dyn Dir,
        ledger: &mut crate::persistence::SaveLedger,
    ) -> Result<(), Error> {
        debug!(target: LOG_TARGET, operation = "save_hnsw_graph_start",
            "Starting HNSW graph save"
        );

        // EMPTY INDEX CHECK:
        let vector_count = self.vector_count();
        if vector_count == 0 {
            debug!(target: LOG_TARGET, operation = "save_hnsw_graph",
                reason = "empty_index",
                "Skipping HNSW graph dump - index is empty"
            );
            return Ok(());
        }

        // The dense index writes its one artefact, the dump, under an empty
        // prefix, which is the directory's root, and records its length
        // rather than a digest, for the reason `persistence::ArtefactDigest`
        // gives: the dump streams itself and seeks back to write its header,
        // so hashing it whole would mean reading the largest artefact in the
        // directory back off the disk for a guarantee its own two checksums
        // already give.
        let index = self.dense().index.read().unwrap();
        match index.write("", dir, ledger) {
            Ok(()) => {
                debug!(target: LOG_TARGET, operation = "save_hnsw_graph_complete",
                    file_created = DUMP_FILENAME,
                    file_bytes = ledger.recorded_bytes(DUMP_FILENAME).unwrap_or(0),
                    "HNSW graph saved successfully"
                );
                Ok(())
            }
            Err(Error::GraphDumpFailed(e)) => {
                error!(target: LOG_TARGET, operation = "save_hnsw_graph", error = %e, "HNSW graph dump failed");
                Err(Error::GraphDumpFailed(e))
            }
            Err(e) => Err(e),
        }
    }

    // 6. PERSISTENCE INTEGRATION METHODS (2 methods)

    /// Load an index from a .zdb directory. See `persistence::load_index`.
    ///
    /// The binding calls this with the interpreter lock released, so the
    /// whole load, the graph rebuild it may fall back to included, runs
    /// without it.
    ///
    /// A directory whose text layer recorded a tokenizer of the caller's own
    /// refuses to open this way, since the engine cannot reproduce one; see
    /// [`Collection::load_with`].
    pub fn load(path: &str) -> Result<Self, Error> {
        Self::load_with(path, None)
    }

    /// Load an index, handing it the tokenizer its text layer was declared
    /// with.
    ///
    /// `config.json` records the tokenizer by value. The built-in one is
    /// recorded as `simple` and needs nothing handed. A caller's own
    /// implementation is recorded as `external`, and a directory recording
    /// that opens only when the same implementation is handed here, because
    /// a query tokenized any other way would not find the terms the records
    /// were counted into. A tokenizer whose declaration is not the one
    /// recorded is refused, and so is one handed to a directory that takes
    /// no text.
    pub fn load_with(path: &str, tokenizer: Option<Arc<dyn Tokenizer>>) -> Result<Self, Error> {
        crate::persistence::load_index(
            path,
            tokenizer,
            JournalPolicy::Replay(Durability::default()),
        )
        .map(|(index, _)| index)
    }

    /// Open a directory and say what recovering it took.
    ///
    /// What [`Collection::load_with`] does, with the report it discards.
    /// The durability is the sink's from here on, so a caller that means to
    /// keep writing to the journal names the policy it wants; it is not
    /// recorded in the directory, because it is a property of the process
    /// writing and not of the data, and a bulk loader and a server open the
    /// same directory wanting different ones.
    ///
    /// The order matters and is the whole of recovery. The index a save was
    /// killed between its two renames left aside is put back first, because
    /// otherwise there is no directory to open. The checkpoint is read as it
    /// always was. The journal beside it is held to the checkpoint by
    /// collection id and by sequence. Every record above the sequence the
    /// checkpoint holds is applied, in order, at the internal id and the
    /// level it names. Only then is a sink attached, because
    /// [`Collection::apply`] refuses to run with one attached and a replay is
    /// not a new mutation.
    ///
    /// A record that will not apply refuses the open, naming the sequence,
    /// and the collection is as it was before that record. So is a corrupt
    /// middle, a journal whose header names another collection, and a
    /// journal whose first record sits above the one after the checkpoint's.
    /// A torn tail is not a refusal: the records before it are whole, they
    /// are replayed, and the journal is cut back to them.
    pub fn recover(
        path: &str,
        tokenizer: Option<Arc<dyn Tokenizer>>,
        durability: Durability,
    ) -> Result<(Self, Recovery), Error> {
        Self::recover_with(path, tokenizer, JournalPolicy::Replay(durability))
    }

    /// [`Collection::recover`] under any of the three policies, being a
    /// replay, a repair or the checkpoint alone, with the report of what the
    /// load did. See [`JournalPolicy`].
    pub fn recover_with(
        path: &str,
        tokenizer: Option<Arc<dyn Tokenizer>>,
        policy: JournalPolicy,
    ) -> Result<(Self, Recovery), Error> {
        crate::persistence::load_index(path, tokenizer, policy)
    }

    /// Open a directory's checkpoint and leave the journal beside it alone.
    ///
    /// The escape hatch for a directory copied without its sibling, which is
    /// what [`Error::JournalMissing`] tells the caller about. Every mutation
    /// after the sequence the checkpoint holds is in the journal and is not
    /// applied here, the file is neither read nor reopened, and no sink is
    /// attached, so the collection this returns records nothing until one is.
    /// A record is written to the log at `warn` saying so.
    pub fn load_checkpoint_only(
        path: &str,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> Result<Self, Error> {
        crate::persistence::load_index(path, tokenizer, JournalPolicy::CheckpointOnly)
            .map(|(index, _)| index)
    }

    /// Journal this collection to `dir`, writing the checkpoint the journal
    /// replays onto and attaching a sink over it.
    ///
    /// # Why the checkpoint is written here
    ///
    /// A journal is a list of mutations since a checkpoint, so it is worth
    /// nothing without one. A collection that has never been saved has no
    /// `config.json` to replay onto, and a journal opened beside a directory
    /// that does not exist would record a million inserts that no recovery
    /// could apply. So turning the journal on writes the empty checkpoint
    /// itself, and from that moment the directory and the file beside it are
    /// a pair. A caller who never calls [`Collection::save`] again keeps a
    /// directory holding the empty checkpoint and a journal holding every
    /// mutation, which opens and replays in full; what they pay for never
    /// checkpointing is an open that runs at insert speed, which any later
    /// save collapses.
    ///
    /// The path is named here rather than at `create()` because this is the
    /// first moment the engine has one. The declaration a collection is built
    /// from carries no directory.
    ///
    /// # The order
    ///
    /// The mutation guard is held for the whole of it, so nothing is
    /// recorded that the checkpoint does not hold and nothing is held that
    /// is not recorded. The directory's lock is taken first, exclusive, and
    /// the journal holds it from then on. The journal's directory is created
    /// empty next, because the manifest names it and a manifest naming a
    /// journal that was never created is a directory this build refuses to
    /// open. Then the sink is attached. Then the save runs as a checkpoint,
    /// which syncs a journal with no records, records sequence zero, writes
    /// a manifest naming the journal with `journal` among its features, and
    /// rolls the journal onto its first segment, which names that
    /// checkpoint.
    ///
    /// # The collection it names
    ///
    /// The journal's segments carry this collection's id, and the checkpoint
    /// records the same id in the manifest's identity and in its journal
    /// record, so the three agree by construction; a load holds the journal
    /// to the identity and refuses a manifest whose two ids disagree. The
    /// checkpoint is a save like any other, so it records the next
    /// generation, with the directory the collection last committed or read
    /// as its parent. The journal's sequence restarts at one here and the
    /// generation does not, and every segment names the checkpoint's
    /// snapshot, so a journal of an earlier history is never taken for this
    /// one.
    ///
    /// A process killed between the journal's directory and the save leaves
    /// an empty journal with no directory, which nothing opens and the next
    /// call empties again.
    pub fn journal_to(&self, dir: &str, durability: Durability) -> Result<(), Error> {
        let _writers = self.writers.lock().unwrap();
        if let Some(refusal) = self.sink_refusal() {
            return Err(refusal);
        }
        if self.sink_attached() {
            return Err(Error::Engine(
                "This collection already hands its records to a sink, so opening a journal for \
                 it would leave the records of one call in two places. A collection is \
                 journaled once."
                    .to_string(),
            ));
        }
        let storage = FsStorage::at(Path::new(dir))?;
        refuse_foreign_target(&storage, dir)?;
        let lock = crate::journal::lock_for_journal(&storage, dir)?;
        zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::LockHeld);
        info!(target: LOG_TARGET, operation = "journal_open",
            directory = dir,
            journal = %storage.journal_path().display(),
            "Opening a journal beside the directory and writing the checkpoint it replays onto"
        );
        let sink =
            crate::journal::JournalSink::create(&storage, self.collection_id(), durability, lock)?;
        self.attach_sink_locked(Box::new(sink));
        self.save_locked(dir)
    }

    /// Save into the directory the journal sits beside and roll the journal
    /// past it, which is [`Collection::save`] with the path taken from the
    /// journal rather than named again.
    ///
    /// Nothing runs this on its own. A checkpoint holds the mutation guard
    /// for the whole of a save, which is seconds at scale, and an engine
    /// that ran one inside an `add` would make one call in some thousands
    /// stall for that long at a moment the caller did not choose. The
    /// caller who opened the journal chooses when to pay it, and what they
    /// pay for not choosing is an open that replays every record above the
    /// checkpoint at the speed of an insert. [`Collection::journal_status`]
    /// reports how many that is.
    ///
    /// After a commit that failed, this is also what lets the collection
    /// take mutations again; see `SinkSlot`.
    pub fn checkpoint(&self) -> Result<(), Error> {
        let _writers = self.writers.lock().unwrap();
        if let Some(refusal) = self.sink_refusal() {
            return Err(refusal);
        }
        let Some(journal) = self.sink_journal_path() else {
            return Err(Error::NotJournaled);
        };
        let dir = crate::journal::directory_of(&journal).ok_or_else(|| Error::TargetHasNoName {
            target: journal.clone(),
        })?;
        self.save_locked(&dir.to_string_lossy())
    }

    /// Install the sparse space's index read back from its artefact, under
    /// the space's guard taken alone. For persistence loading only.
    pub(crate) fn set_sparse_index(&self, index: PostingsIndex) {
        if let Some(space) = self.sparse() {
            *space.index.write().unwrap() = index;
        }
    }

    /// Install the text layer's dictionary read back from its artefact,
    /// under its guard taken alone. For persistence loading only.
    pub(crate) fn set_dictionary(&self, dictionary: TermDictionary) {
        if let Some(text) = self.sparse().and_then(|space| space.text.as_ref()) {
            *text.dictionary.write().unwrap() = dictionary;
        }
    }

    /// Restore the graph the save wrote, instead of rebuilding it
    ///
    /// For persistence loading only, and only after `restore_data_fields` has
    /// installed the id mappings and the product quantizer, because both decide
    /// what the dump is checked against. `positions` is the internal id of the
    /// record at each position, which the mappings gave, and the reader files
    /// each node at the position of the id it carries. `version` is the dump
    /// version the directory's format requires.
    ///
    /// Below format 5.0.0 every reason the dump cannot be used is a fallback
    /// rather than a failure: the caller rebuilds from the separate row
    /// stores, so a directory whose dump is absent, was written by a release
    /// whose distance types were named differently, or is damaged still
    /// loads. At 5.0.0 the dump holds the only copy of each record's row, so a
    /// dump refused on its header, its identity or its rows refuses the load
    /// with [`Error::GraphDumpInvalid`], naming the check, and one whose
    /// topology cannot be used hands back its rows, which the caller rebuilds
    /// from. `ZEUSDB_LOAD_REBUILD_GRAPH` asks for the rebuild from the rows
    /// at 5.0.0, which still reads and verifies them, and for the rebuild from
    /// the separate stores below it.
    pub(crate) fn restore_graph_from_dump(
        &mut self,
        dir: &dyn Dir,
        max_origin_id: usize,
        recorded_bytes: Option<u64>,
        positions: &[u32],
        version: u32,
    ) -> Result<DumpOutcome, Error> {
        let asked = rebuild_requested();
        if asked && version < 2 {
            return Ok(DumpOutcome::Rebuild(format!(
                "{} asked for the rebuild",
                REBUILD_ENV
            )));
        }

        // A save skips the graph dump entirely when the index holds nothing, so
        // any dump left in an empty index's directory belongs to an earlier
        // save and describes records this one no longer holds.
        let live = self.ids.read().unwrap().len();
        if live == 0 {
            return Ok(DumpOutcome::Rebuild(
                "the index holds no records".to_string(),
            ));
        }
        let refused = |refusal: DumpRefusal| {
            if version >= 2 {
                Err(Error::GraphDumpInvalid {
                    name: DUMP_FILENAME.to_string(),
                    detail: refusal.detail,
                })
            } else {
                Ok(DumpOutcome::Rebuild(refusal.detail))
            }
        };

        // The dense index restores itself from the dump. The length
        // manifest.json recorded is checked there before the dump is read,
        // since the dump's own checksums say the file is internally
        // consistent and the length says it is the file this directory's
        // save wrote. A trained product quantizer is what makes the saved
        // graph a quantized one, so it is what decides which element type
        // the dump must carry.
        let dense = self.dense();
        let open = DenseOpen {
            metric: dense.metric.clone(),
            dim: dense.dim,
            m: dense.m(),
            ef_construction: dense.ef_construction(),
            pq: dense.pq.clone(),
            int8: dense.int8_codec().cloned(),
            keeps_raw: dense.keeps_raw(),
            version,
            rows_only: asked,
        };
        let bounds = Bounds {
            min_records: live,
            max_records: max_origin_id,
            max_bytes: u64::MAX,
            ids_by_position: positions,
        };
        let mut index =
            match DenseIndex::open_dump(&open, "", dir, &DumpLength(recorded_bytes), &bounds) {
                Ok(DenseRestored::Index(index)) => *index,
                Ok(DenseRestored::Rows { rows, reason }) => {
                    let reason = if asked {
                        format!("{} asked for the rebuild", REBUILD_ENV)
                    } else {
                        reason
                    };
                    return Ok(DumpOutcome::Rows {
                        rows,
                        reason,
                        asked,
                    });
                }
                Err(refusal) => return refused(refusal),
            };
        let nodes = index.graph().nb_points();

        // Every record the mappings hold has a node in a dump this index
        // wrote, and the reader refused a dump that names no node for one, so
        // every position takes a node here. The count is held anyway, and a
        // dump short of one is refused before it is installed, and the caller
        // rebuilds, as it does for every other refusal of a dump.
        let unplaced = {
            let ids = self.ids.read().unwrap();
            index.set_live(ids.slots())
        };
        if unplaced > 0 {
            return refused(DumpRefusal {
                region: DumpRegion::Identity,
                detail: format!(
                    "the graph dump names no node for {} of the records mappings.bin holds",
                    unplaced
                ),
            });
        }
        dense.replace_index(index);
        Ok(DumpOutcome::Read(nodes))
    }

    /// Rebuild the graph from the stored PQ codes (for persistence loading only)
    ///
    /// Requires a trained product quantizer, which the loader installs before
    /// any rebuild runs. Replaces the raw graph `new_empty` built with a fresh
    /// PQ graph and inserts every record's codes under the internal id restored
    /// from mappings.bin, so the loaded index is quantized exactly as the saved
    /// one was and no vector is reconstructed to full width on the way.
    ///
    /// A record that has a raw vector but no stored codes is quantized through
    /// the loaded codebook rather than dropped. An intact directory saved by a
    /// trained index holds codes for every record, so that path only runs on a
    /// directory that lost pq_codes.bin while keeping its raw vectors. A record
    /// missing from mappings.bin is assigned a fresh internal id for the same
    /// reason: every record must come back. It takes the next position, and
    /// its id joins `positions`, the internal id of the record at each
    /// position.
    ///
    /// Returns (records inserted, quantized from raw, remapped).
    pub(crate) fn rebuild_graph_from_codes(
        &mut self,
        pq_codes: &HashMap<String, Vec<u8>>,
        vectors: &HashMap<String, Vec<f32>>,
        positions: &mut Vec<u32>,
    ) -> Result<(usize, usize, usize), String> {
        let pq = match &self.dense().pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                return Err(
                    "the quantized graph rebuild requires a trained product quantizer".to_string(),
                )
            }
        };

        let mut extra: Vec<(String, Vec<u8>)> = Vec::new();
        for (id, vector) in vectors {
            if !pq_codes.contains_key(id) {
                let codes = pq.quantize(vector).map_err(|e| {
                    format!(
                        "record '{}' has a raw vector but no stored PQ codes, and quantizing \
                         it through the loaded codebook failed: {}",
                        id, e
                    )
                })?;
                extra.push((id.clone(), codes));
            }
        }
        // Sorted so the internal ids the missing records are about to be handed
        // are handed in a fixed order rather than in hash map order.
        extra.sort_by(|a, b| a.0.cmp(&b.0));

        let mut new_hnsw = VectorGraph::new_pq(
            &self.dense().metric,
            self.dense().m(),
            self.expected_size(),
            MAX_LAYER,
            self.dense().ef_construction(),
            pq,
        );

        let mut batch: Vec<(&Vec<u8>, usize, Position)> =
            Vec::with_capacity(pq_codes.len() + extra.len());
        let mut lost: Vec<(&String, &Vec<u8>)> = Vec::new();
        {
            let ids = self.ids.read().unwrap();
            for (id, codes) in pq_codes
                .iter()
                .chain(extra.iter().map(|(id, codes)| (id, codes)))
            {
                match ids.slot_of(id) {
                    Some(at) => {
                        batch.push((codes, positions[at] as usize, Position::from_slot(at)))
                    }
                    None => lost.push((id, codes)),
                }
            }
        }
        let remapped = lost.len();
        lost.sort_by(|a, b| a.0.cmp(b.0));
        for (id, codes) in lost {
            let internal_id = self.get_next_id();
            // Held to the ceiling before the id store is touched, as the
            // mappings' own ids are, so no id past it is cut to 32 bits.
            crate::persistence::refuse_past_last_id(internal_id).map_err(|e| e.to_string())?;
            let at = {
                let mut ids = self.ids.write().unwrap();
                let at = ids.extent();
                ids.insert(at, id).map_err(|e| e.to_string())?;
                at
            };
            positions.push(internal_id as u32);
            batch.push((codes, internal_id, Position::from_slot(at)));
        }

        // Insert in internal id order, which is arrival order, rather than in
        // the order a hash map hands the codes out. Two rebuilds of one
        // directory otherwise wire the graph differently in each process.
        batch.sort_by_key(|&(_, internal_id, _)| internal_id);

        // Filled before it is installed, and installed under one write guard, so
        // the graph the index holds is never a partly rebuilt one.
        if !batch.is_empty() {
            new_hnsw.insert_batch_pq(&batch)?;
        }
        // Every record the index holds takes a node, so a record the codes
        // did not reach refuses the rebuild before anything is swapped, rather
        // than leaving a held record outside the graph.
        let held = self.ids.read().unwrap().len();
        if new_hnsw.nb_points() != held {
            return Err(format!(
                "the rebuild from codes built {} nodes for the {} records the index holds",
                new_hnsw.nb_points(),
                held
            ));
        }
        self.dense().replace_graph(new_hnsw);
        self.sync_dense_live();

        Ok((batch.len(), extra.len(), remapped))
    }

    /// Rebuild the graph of a trained scalar index from the rows artefact,
    /// the counterpart of `rebuild_graph_from_codes`.
    ///
    /// The rows are what the saved graph held, so the rebuild inserts them
    /// into a fresh scalar graph under the codec the loader installed, in
    /// internal id order, and no vector is decoded on the way. The caller has
    /// already held every row to the mappings both ways, so every row names
    /// a live record and every live record has a row, and each row is filed
    /// at the position `positions` gives its internal id.
    pub(crate) fn rebuild_graph_from_int8_rows(
        &mut self,
        rows: &crate::persistence::Int8Rows,
        positions: &[u32],
    ) -> Result<usize, String> {
        let codec = self
            .dense()
            .int8_codec()
            .cloned()
            .ok_or_else(|| "the scalar graph rebuild requires fitted scales".to_string())?;
        let mut new_hnsw = VectorGraph::new_int8(
            &self.dense().metric,
            self.dense().m(),
            self.expected_size(),
            MAX_LAYER,
            self.dense().ef_construction(),
            codec,
        );
        let mut walk = PositionWalk::new(positions);
        let batch: Vec<(&[i8], usize, Position)> = rows
            .iter()
            .map(|(internal_id, row)| {
                let at = walk.position_of(internal_id).ok_or_else(|| {
                    format!(
                        "the row of internal id {} names a record mappings.bin does not hold",
                        internal_id
                    )
                })?;
                Ok((row, internal_id, Position::from_slot(at)))
            })
            .collect::<Result<_, String>>()?;
        if !batch.is_empty() {
            new_hnsw.insert_batch_int8(&batch)?;
        }
        self.dense().replace_graph(new_hnsw);
        self.sync_dense_live();
        Ok(batch.len())
    }

    /// Whether the dense space was declared with scalar quantization.
    pub(crate) fn declares_int8(&self) -> bool {
        self.dense().declares_int8()
    }

    /// The fitted scalar codec, where the dense space holds one.
    pub(crate) fn int8_codec(&self) -> Option<Arc<zeusdb_vector_core::Int8Codec>> {
        self.dense().int8_codec().cloned()
    }

    /// Install the codec a saved directory recorded (for persistence loading
    /// only).
    pub(crate) fn set_int8_codec(
        &mut self,
        codec: Arc<zeusdb_vector_core::Int8Codec>,
    ) -> Result<(), Error> {
        self.dense()
            .install_int8_codec(codec)
            .map_err(|_| Error::Engine("the scalar codec was already fitted".to_string()))
    }

    /// Values a scalar graph has clipped since its scales were fitted; see
    /// `DenseIndex::saturated`.
    pub(crate) fn int8_saturated(&self) -> u64 {
        self.dense().index.read().unwrap().saturated()
    }

    /// Take the clipped count a saved directory recorded (for persistence
    /// loading only).
    pub(crate) fn set_int8_saturated(&mut self, values: u64) {
        self.dense().index.read().unwrap().set_saturated(values);
    }

    /// Install the id store `mappings.bin` built (for persistence loading
    /// only).
    ///
    /// The loader builds it, holds it to the counter `config.json` records
    /// and, for a directory written before format 4.0.0, holds the reverse map
    /// to being the forward map's exact inverse, before the collection exists;
    /// see `persistence::load_ids`. It is taken whole.
    pub(crate) fn set_id_store(&mut self, store: IdStore) {
        *self.ids.write().unwrap() = store;
        // The dense index's live set is the mappings' key set, so a rebuild
        // that replays every record removes ids the index holds.
        self.sync_dense_live();
    }

    /// Set counters (for persistence loading only)
    pub(crate) fn set_counters(&mut self, id_counter: usize, vector_count: usize) {
        *self.id_counter.lock().unwrap() = id_counter;
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Set the vector count alone (for persistence loading only)
    ///
    /// Separate from `set_counters` because the id counter must keep whatever
    /// a graph rebuild advanced it to for a record the mappings did not name.
    /// Rewinding it would hand out an internal id such a record already holds.
    pub(crate) fn set_vector_count(&mut self, vector_count: usize) {
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Replace the stored record data with what was read from disk
    ///
    /// For persistence loading only, and only after the graph is back, because
    /// the columns are built over the id store and either rebuild fallback may add
    /// an id for a record the mappings did not name. Writing the two maps back
    /// and building the columns from them leaves the loaded index holding
    /// exactly what was saved.
    pub(crate) fn restore_storage_maps(
        &mut self,
        pq_codes: HashMap<String, Vec<u8>>,
        vector_metadata: HashMap<String, HashMap<String, Value>>,
    ) {
        *self.dense().pq_codes.write().unwrap() = pq_codes;
        self.restore_metadata(vector_metadata);
        self.rebuild_columns();
    }

    /// Place every record's metadata at the position the id store holds for
    /// it.
    ///
    /// `metadata.json` is keyed by external id and the store is indexed by
    /// position, so each entry is resolved through the id store, which the
    /// mappings and either graph rebuild have filled by now. An entry naming
    /// an id the mappings do not hold has no slot to go in and is dropped,
    /// counted in the log. No reader could reach such an entry by id, since
    /// every one resolved a record through the id store first; a filtered
    /// `count` on an undeclared field used to count it, and counts it no
    /// longer.
    ///
    /// The guards are taken in the declared order, the id store then
    /// `vector_metadata`.
    fn restore_metadata(&mut self, metadata: HashMap<String, HashMap<String, Value>>) {
        let ids = self.ids.read().unwrap();
        let mut store = self.vector_metadata.write().unwrap();
        store.clear(self.expected_size());
        let mut unmapped = 0usize;
        for (ext_id, fields) in metadata {
            match ids.slot_of(&ext_id) {
                Some(at) => store.insert(at, fields),
                None => unmapped += 1,
            }
        }
        if unmapped > 0 {
            debug!(target: LOG_TARGET, operation = "restore_metadata",
                unmapped = unmapped,
                "metadata.json names records the mappings do not hold, and their entries were dropped"
            );
        }
    }

    /// Build every declared column from the restored metadata.
    ///
    /// **Nothing on disk holds a column.** The declaration comes back from
    /// `config.json` and the values come back from `metadata.json`, so the
    /// columns are derived here in one pass over the records rather than read.
    /// That is what keeps the directory format unchanged, keeps a directory
    /// written by this build readable by an older one, and means a directory
    /// written before the declaration existed simply loads with no columns.
    ///
    /// Runs after the graph is restored and after `restore_storage_maps` has
    /// written the metadata back, so the id store holds the position every
    /// column entry is addressed by. It is a no-op on an index that declared
    /// nothing, which is every directory saved before this existed.
    /// It clears before it writes, so this is the one authority on what the
    /// columns hold rather than a second writer over an earlier one's
    /// leftovers.
    ///
    /// The guards are taken in the declared order, the id store then
    /// `vector_metadata` then `columns`, even though the loader holds `&mut
    /// self` and nothing else can be running.
    ///
    /// The store is cleared to the declared reservation first whether or not
    /// a field is declared, since the loader's collection starts from one that
    /// reserves nothing and a collection declaring no field keeps it.
    fn rebuild_columns(&mut self) {
        let ids = self.ids.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        let mut columns = self.columns.write().unwrap();
        columns.clear(self.expected_size());
        if !columns.is_declared() {
            return;
        }
        let empty: HashMap<String, Value> = HashMap::new();
        for at in ids.slots() {
            match vector_metadata.get(at) {
                Some(fields) => columns.write(at, &fields),
                None => columns.write(at, &empty),
            }
        }
        debug_assert!(
            columns.tracks(ids.len()),
            "the rebuild writes one column entry per record in the id store"
        );
    }

    /// Give the dense graph the reservation the declared size asks for (for
    /// persistence loading only).
    ///
    /// The loader's collection starts from a graph that reserves nothing,
    /// because a load replaces it whenever the directory holds a record. A
    /// load that restores no graph and rebuilds none keeps it, and this gives
    /// it what a created collection's graph reserves.
    pub(crate) fn reserve_dense_graph(&self) {
        let expected_size = self.expected_size();
        self.dense()
            .index
            .write()
            .unwrap()
            .reserve_declared(expected_size);
    }

    /// Set quantization config (for persistence loading only)
    pub(crate) fn set_quantization_config(&mut self, config: Option<QuantizationConfig>) {
        let keeps_raw = config
            .as_ref()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw);
        let dense = self.dense_mut();
        dense.quantization_config = config;
        dense.index.write().unwrap().set_keeps_raw(keeps_raw);
    }

    /// Set PQ instance (for persistence loading only)
    pub(crate) fn set_pq(&mut self, pq: Option<Arc<zeusdb_vector_core::PQ>>) {
        let dense = self.dense_mut();
        dense.pq = pq.clone();
        dense.index.write().unwrap().set_pq(pq);
    }

    /// Set training threshold reached flag (for persistence loading only)
    pub(crate) fn set_training_threshold_reached(&mut self, value: bool) {
        self.training_threshold_reached
            .store(value, std::sync::atomic::Ordering::Release);
    }

    /// Rebuild the graph from records read off disk, under the internal ids
    /// the mappings name.
    ///
    /// This is the load path's fallback, taken where the saved graph dump is
    /// absent, damaged, or written by a release this build cannot read. The
    /// records arrive as owned Rust, one per record the directory holds.
    ///
    /// **Every record keeps the internal id the mappings give it.** A fresh
    /// graph is built off to the side, each record inserted under its mapped
    /// id in ascending id order, and the graph swapped in under one write
    /// guard, which is the shape `rebuild_graph_from_codes` and `compact`
    /// take. The id maps, the metadata, the columns, the counters and the
    /// sparse space are left as the directory restored them, so the loaded
    /// collection holds the ids `mappings.bin` names and the counter
    /// `config.json` records. It used to route every record through `add`
    /// with overwrite, which removed each record and reissued its id above
    /// the saved counter, so a directory loaded this way disagreed with its
    /// own mappings, with the quantized fallback, and with anything outside
    /// the directory keyed on internal ids.
    ///
    /// A record the mappings do not name is given a fresh id, as the
    /// quantized fallback gives one, taken in external id order so the ids
    /// are handed out the same way in every process, and the next position,
    /// its id joining `positions`. Ascending internal id is arrival order and
    /// position order, so the rebuild is as close to the original build as a
    /// rebuild can get, and two rebuilds of one directory are one graph.
    ///
    /// What is reproduced exactly from the path this replaced is
    /// `extract_single_vector`'s validation and its call to
    /// `process_vector_for_space`. A stored vector is already normalized for
    /// a cosine index, and normalizing it a second time is what `add` did, so
    /// it is what this does. Skipping it would build a different graph, which
    /// on this path means giving a user's index different answers. A refused
    /// record fails the whole rebuild rather than being dropped, since a
    /// graph short of a record the storage maps still report is an index
    /// every query silently misses on.
    pub(crate) fn rebuild_from_records(
        &self,
        records: Vec<(String, Vec<f32>)>,
        positions: &mut Vec<u32>,
    ) -> Result<usize, Error> {
        let mut mapped: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(records.len());
        let mut unmapped: Vec<(String, Vec<f32>)> = Vec::new();
        {
            let ids = self.ids.read().unwrap();
            for (id, vector) in records {
                if vector.len() != self.dense().dim {
                    return Err(Error::RebuildRefusedDimension {
                        id,
                        expected: self.dense().dim,
                        got: vector.len(),
                    });
                }
                if let Some((position, value)) = vector
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(Error::RebuildRefusedNotFinite {
                        id,
                        value: *value,
                        position,
                    });
                }
                let vector = self.process_vector_for_space(vector);
                match ids.slot_of(&id) {
                    Some(at) => mapped.push((at, positions[at] as usize, vector)),
                    None => unmapped.push((id, vector)),
                }
            }
        }

        // The binding releases the interpreter lock around the whole load,
        // so the rebuild runs without it. The mutation guard is held for the
        // ids handed to unmapped records and for the swap, which is the
        // discipline every other mutation keeps, though nothing else can
        // reach a collection the loader still holds.
        let _writers = self.writers.lock().unwrap();
        unmapped.sort_by(|a, b| a.0.cmp(&b.0));
        for (id, vector) in unmapped {
            let internal_id = self.get_next_id();
            // Held to the ceiling before the id store is touched, as the
            // mappings' own ids are, so no id past it is cut to 32 bits.
            crate::persistence::refuse_past_last_id(internal_id)?;
            let at = {
                let mut ids = self.ids.write().unwrap();
                let at = ids.extent();
                ids.insert(at, &id)?;
                at
            };
            positions.push(internal_id as u32);
            mapped.push((at, internal_id, vector));
        }
        mapped.sort_by_key(|&(_, internal_id, _)| internal_id);

        let dense = self.dense();
        let mut new_hnsw = VectorGraph::new_raw(
            &dense.metric,
            dense.dim,
            dense.m(),
            self.expected_size(),
            MAX_LAYER,
            dense.ef_construction(),
        );
        for (at, internal_id, vector) in &mapped {
            new_hnsw.insert(vector, *internal_id, Position::from_slot(*at));
        }

        // Filled before it is installed, and installed under one write guard,
        // so the graph the index holds is never a partly rebuilt one.
        dense.replace_graph(new_hnsw);
        self.sync_dense_live();

        Ok(mapped.len())
    }

    /// Set training IDs (for persistence loading only)
    pub(crate) fn set_training_ids(&mut self, ids: Vec<String>) {
        *self.training_ids.write().unwrap() = ids;
    }

    /// Restore the creation timestamp from the saved manifest
    ///
    /// For persistence loading only. `new_empty` stamps the load time, which is
    /// what `manifest.json` used to record as `created_at` on the next save.
    pub(crate) fn set_created_at(&mut self, created_at: String) {
        *self.created_at.write().unwrap() = created_at;
    }

    /// Restore when the codebook was fitted, from the saved directory
    ///
    /// For persistence loading only. See the field for why the value is carried
    /// rather than restamped.
    pub(crate) fn set_training_completed_at(&mut self, completed_at: Option<String>) {
        *self.dense().training_completed_at.write().unwrap() = completed_at;
    }

    // ============================================================================
    // PERSISTENCE GETTERS - For accessing private fields from persistence module
    // ============================================================================

    /// How many generated ids this index has issued, for `config.json`
    pub(crate) fn generated_ids(&self) -> usize {
        *self.generated_ids.lock().unwrap()
    }

    /// Restore the generated id counter, never below what the records need
    ///
    /// `floor` is the highest N among the ids of the form `vec_N` the directory
    /// actually holds. A directory written before this field existed carries a
    /// zero and would otherwise reissue every generated id it holds, so the
    /// records themselves supply the floor. A directory that carries the field
    /// is at or above its own floor already and the max changes nothing.
    pub(crate) fn set_generated_ids(&mut self, saved: usize, floor: usize) {
        *self.generated_ids.lock().unwrap() = saved.max(floor);
    }

    /// The highest N among ids of the form `vec_N`, or zero where there is none
    ///
    /// A user is free to add a record called `vec_9` by hand, and this counts
    /// it, so the next generated id is above it. That is the point: the counter
    /// exists to stop an id being handed out twice.
    pub(crate) fn highest_generated_id(ids: impl Iterator<Item = impl AsRef<str>>) -> usize {
        ids.filter_map(|id| {
            id.as_ref()
                .strip_prefix("vec_")
                .and_then(|rest| rest.parse::<usize>().ok())
        })
        .max()
        .unwrap_or(0)
    }

    /// Get the current ID counter value (thread-safe)
    pub(crate) fn id_counter(&self) -> usize {
        *self.id_counter.lock().unwrap()
    }

    /// Get read access to the PQ codes HashMap (thread-safe), for the tests
    ///
    /// Crate private, as are the four accessors below that also hand out a
    /// guard. They return a [`ReadGuard`], and the binding reaches nothing
    /// through a guard: it takes owned values from the operations and
    /// nothing else.
    #[cfg(test)]
    pub(crate) fn pq_codes(&self) -> ReadGuard<'_, HashMap<String, Vec<u8>>> {
        self.dense().pq_codes.read().unwrap()
    }

    /// Get read access to the metadata store (thread-safe)
    pub(crate) fn vector_metadata(&self) -> ReadGuard<'_, MetadataStore> {
        self.vector_metadata.read().unwrap()
    }

    /// Get read access to the id store, being every record's external id
    /// with its position and the live set beside them
    pub(crate) fn ids(&self) -> ReadGuard<'_, IdStore> {
        self.ids.read().unwrap()
    }

    /// Get read access to the dense index, which names each record held by
    /// the internal id its node was inserted under, for a save that writes
    /// records by internal id. Taken after the id store and before any space's
    /// codes or index, which is its place in the order.
    pub(crate) fn dense_index(&self) -> ReadGuard<'_, DenseIndex> {
        self.dense().index.read().unwrap()
    }

    /// Get reference to the quantization configuration
    pub(crate) fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.dense().quantization_config.as_ref()
    }

    /// Get reference to the PQ instance
    pub(crate) fn pq(&self) -> Option<&Arc<zeusdb_vector_core::PQ>> {
        self.dense().pq.as_ref()
    }

    /// Get the index creation timestamp
    pub(crate) fn created_at(&self) -> String {
        self.created_at.read().unwrap().clone()
    }

    /// When the codebook was fitted, or `None` on an index that never trained
    pub(crate) fn training_completed_at(&self) -> Option<String> {
        self.dense().training_completed_at.read().unwrap().clone()
    }

    /// Get read access to training IDs (for persistence)
    pub(crate) fn training_ids(&self) -> ReadGuard<'_, Vec<String>> {
        self.training_ids.read().unwrap()
    }

    /// Get training threshold reached flag (for persistence)
    pub(crate) fn training_threshold_reached(&self) -> bool {
        self.training_threshold_reached
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// What training measured, where it ran. See `Space::rerank_calibration`.
    pub(crate) fn rerank_calibration(&self) -> Option<RerankCalibration> {
        self.dense().rerank_calibration()
    }

    /// Install a calibration read back from a saved index.
    pub(crate) fn set_rerank_calibration(&self, calibration: Option<RerankCalibration>) {
        self.dense().set_rerank_calibration(calibration);
    }
}
