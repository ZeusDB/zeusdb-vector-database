//! The journal beside a directory, the sink that reaches it, and the three
//! durability policies.
//!
//! [`zeusdb_vector_core`] holds what a journal is, being the file header, the
//! record framing, the reader that classifies a torn tail and a corrupt
//! middle, and the writer. The collection holds where a record comes from,
//! being an [`OperationSink`](crate::OperationSink) it hands every mutation
//! to before the mutation runs. This module is the join: a sink that is a
//! journal writer, the rule for where the file lives, what a checkpoint asks
//! of it, and what a caller is promised under each policy.
//!
//! # Where the file lives
//!
//! `<name>.zdbwal`, beside `<name>`, the sibling of the directory in the way
//! `<name>.zdbtmp` and `<name>.zdbold` already are.
//!
//! It cannot live inside the directory. A save commits by renaming the
//! target aside to `<name>.zdbold` and the staging directory into its place,
//! and Windows refuses to rename a directory that holds an open file with
//! `Access is denied`, so the first of those two renames would fail for as
//! long as the journal was open. Avoiding that rename does not help either:
//! `remove_dir_all` on a directory holding an open file succeeds, so the
//! removal of `<name>.zdbold` after the second rename would take the live
//! journal with it. Both were run against this build's `std` on the platform
//! that refuses, and a sibling survives the whole save with the file open.
//!
//! # How a directory and its journal are paired
//!
//! By content, not by name. `manifest.json` records the journal's file name,
//! the collection id both belong to, and the sequence the checkpoint holds.
//! A journal whose header names another collection is refused. A journal
//! whose first record sits above the one after the checkpoint's sequence is
//! refused, because the records between the two are in neither. A directory
//! whose manifest names a journal that is not beside it is refused by name,
//! which is the copied-without-its-sibling case, unless the caller asks for
//! the checkpoint alone.
//!
//! # What a checkpoint does
//!
//! A save of a journaled collection is a checkpoint. Under the one hold of
//! the mutation guard that the save already takes: the journal is synced
//! whatever the policy, the sequence it reached is written onto the
//! collection so the manifest records it, the four save phases run, and then
//! the journal is truncated in two steps that are each durable on their own.
//! A crash anywhere in that leaves either the previous directory with a
//! journal that replays onto it or the new directory with a journal that
//! replays onto that, and never a directory whose records are in neither.
//!
//! # The three policies
//!
//! [`Durability`] is what a journaled collection promises about a record
//! once the call that added it has returned. Each sits on one of the two
//! commit modes the writer has, and the interval policy adds a thread.
//!
//! | Policy | When the call returns | A process abort | An OS crash or power loss |
//! |---|---|---|---|
//! | [`Durability::PerCall`] | every record of the call is on the device, or the call did not return | nothing acknowledged is lost | nothing acknowledged is lost, on a device that honours the flush |
//! | [`Durability::PerInterval`] | every record is in the kernel, and a thread syncs the file within the interval | nothing acknowledged is lost | up to one interval of acknowledged calls is lost |
//! | [`Durability::NoSync`] | every record is in the kernel | nothing acknowledged is lost | everything since the kernel last wrote the file is lost |
//!
//! The middle column is the same for all three because a process abort
//! loses nothing the kernel holds, and every policy hands the bytes to the
//! kernel before the call returns. The right column is what each policy
//! syncs, and what "acknowledged" means is that the call returned without
//! an error. A record refused by the call, or a call that raised, was never
//! promised under any policy. A checkpoint syncs the journal whatever the
//! policy, so the loss window of the two weaker policies closes at every
//! save.
//!
//! The default is [`Durability::PerCall`], because it is the one a caller
//! who has not thought about durability should have, and what it costs is a
//! function of the caller's batch size rather than of the engine: one flush
//! per call, which is a millisecond or so, against the insert work the call
//! does.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::{debug, info};
use zeusdb_vector_core::{CommitMode, Error, JournalDamage, JournalWriter, OperationKind};

use crate::collection::OperationSink;
use crate::flusher::Flusher;

/// The target the log records in this module carry. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::journal";

/// The suffix a collection's journal takes beside its directory.
pub const JOURNAL_SUFFIX: &str = ".zdbwal";

/// What a journaled collection promises about a record once the call that
/// added it has returned. See the module's table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Durability {
    /// Every record of a call is on the device before the call returns, or
    /// the call did not return. One flush per call. The default.
    #[default]
    PerCall,
    /// Every record is in the kernel when the call returns, and a thread of
    /// the collection's own syncs the file within this long of the first
    /// call that dirtied it. What a caller trades is that an operating
    /// system crash or a power loss inside the window loses the calls that
    /// fell in it, and what they get is a single-record `add` that does not
    /// wait on a flush.
    PerInterval(Duration),
    /// Every record is in the kernel when the call returns and nothing syncs
    /// the file until the next checkpoint. A process abort still loses
    /// nothing. For a bulk load that will checkpoint at the end.
    NoSync,
}

impl Durability {
    /// The interval a caller who names the interval policy and no interval
    /// gets.
    pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(10);

    /// The commit mode the sink takes under this policy.
    pub fn commit_mode(self) -> CommitMode {
        match self {
            Durability::PerCall => CommitMode::Sync,
            Durability::PerInterval(_) | Durability::NoSync => CommitMode::Deferred,
        }
    }

    /// The name the binding and the stats spell the policy by.
    pub fn name(self) -> &'static str {
        match self {
            Durability::PerCall => "call",
            Durability::PerInterval(_) => "interval",
            Durability::NoSync => "none",
        }
    }

    /// The interval, where the policy has one.
    pub fn interval(self) -> Option<Duration> {
        match self {
            Durability::PerInterval(interval) => Some(interval),
            Durability::PerCall | Durability::NoSync => None,
        }
    }
}

/// The journal beside `target`, being its sibling under [`JOURNAL_SUFFIX`].
///
/// Derived from the directory the caller handed over rather than read from
/// the manifest, so a directory renamed with its journal opens under the new
/// name. The name the manifest records is what a refusal quotes.
pub fn journal_path(target: &Path) -> Result<PathBuf, Error> {
    let name = target.file_name().ok_or_else(|| Error::TargetHasNoName {
        target: target.to_path_buf(),
    })?;
    let mut name = name.to_os_string();
    name.push(JOURNAL_SUFFIX);
    Ok(target.parent().unwrap_or_else(|| Path::new("")).join(name))
}

/// The directory a journal at `journal` sits beside, being
/// [`journal_path`] undone. `None` where the file does not carry the
/// suffix.
pub fn directory_of(journal: &Path) -> Option<PathBuf> {
    let name = journal.file_name()?.to_str()?;
    let stem = name.strip_suffix(JOURNAL_SUFFIX)?;
    if stem.is_empty() {
        return None;
    }
    Some(journal.parent().unwrap_or_else(|| Path::new("")).join(stem))
}

/// A collection id as the manifest spells it, being 32 lower case
/// hexadecimal digits.
pub(crate) fn collection_id_hex(id: u128) -> String {
    format!("{:032x}", id)
}

/// A collection id read back from the manifest's spelling.
pub(crate) fn collection_id_from_hex(hex: &str) -> Option<u128> {
    u128::from_str_radix(hex, 16).ok()
}

/// The sink a journaled collection hands every record to.
///
/// One journal writer, the policy its commits keep, and the thread the
/// interval policy syncs from. `append` drops the sequence the writer
/// returns, because the collection has no use for it: what a checkpoint
/// records is the sequence the journal has reached, which
/// [`OperationSink::sequence_reached`] reports after the fact.
#[derive(Debug)]
pub struct JournalSink {
    writer: JournalWriter,
    durability: Durability,
    /// The thread, under the interval policy alone. Dropped with the sink,
    /// which joins it.
    flusher: Option<Flusher>,
    /// The name the manifest records, held here so the checkpoint writing
    /// the manifest does not derive it a second time.
    file: String,
}

impl JournalSink {
    /// Create a journal at `path` for `collection_id`, replacing any file
    /// there, with its first record at sequence one.
    pub fn create(path: &Path, collection_id: u128, durability: Durability) -> Result<Self, Error> {
        let writer = JournalWriter::create(path, collection_id, 1)?;
        Self::over(writer, durability)
    }

    /// Take a writer a recovery reopened for append.
    pub fn from_writer(writer: JournalWriter, durability: Durability) -> Result<Self, Error> {
        Self::over(writer, durability)
    }

    fn over(writer: JournalWriter, durability: Durability) -> Result<Self, Error> {
        let flusher = match durability {
            Durability::PerInterval(interval) => Some(Flusher::start(
                writer.sync_handle(),
                writer.path().to_path_buf(),
                interval,
            )?),
            Durability::PerCall | Durability::NoSync => None,
        };
        Ok(JournalSink {
            file: file_name_of(writer.path()),
            writer,
            durability,
            flusher,
        })
    }

    /// The collection the journal's header names.
    pub fn collection_id(&self) -> u128 {
        self.writer.collection_id()
    }

    /// Where the file is.
    pub fn path(&self) -> &Path {
        self.writer.path()
    }

    /// The policy every commit keeps.
    pub fn durability(&self) -> Durability {
        self.durability
    }

    /// A handle that syncs the file and does nothing else.
    pub fn sync_handle(&self) -> zeusdb_vector_core::JournalSyncHandle {
        self.writer.sync_handle()
    }

    /// The file's length, from the filesystem.
    pub fn file_len(&self) -> Result<u64, Error> {
        self.writer.file_len()
    }
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl OperationSink for JournalSink {
    fn append(&mut self, kind: OperationKind, payload: &[u8]) -> Result<(), Error> {
        // A crash test arms the process to abort with half a record on the
        // disk, which is what a process killed inside one write leaves.
        // Compiled away when `debug_assertions` are off.
        if zeusdb_vector_core::kill_armed_at(zeusdb_vector_core::KillPoint::AddMidAppend) {
            #[cfg(debug_assertions)]
            {
                let half = (zeusdb_vector_core::JOURNAL_RECORD_HEADER_BYTES + payload.len()) / 2;
                self.writer.append_torn(kind, payload, half)?;
            }
            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::AddMidAppend);
        }
        self.writer.append(kind, payload).map(|_| ())
    }

    /// Under the per-call policy, the flush. Under the other two, nothing on
    /// the device: the interval policy's thread is told the file is dirty,
    /// and a sync that thread could not complete since the last commit is
    /// this commit's failure, so a device that will not flush is reported at
    /// the next call rather than never.
    fn commit(&mut self) -> Result<(), Error> {
        match self.durability.commit_mode() {
            CommitMode::Sync => self.writer.sync(),
            CommitMode::Deferred => {
                if let Some(flusher) = &self.flusher {
                    if let Some(failure) = flusher.take_failure() {
                        return Err(failure);
                    }
                    flusher.mark_dirty();
                }
                Ok(())
            }
        }
    }

    fn sync(&mut self) -> Result<(), Error> {
        self.writer.sync()?;
        if let Some(flusher) = &self.flusher {
            flusher.mark_clean();
        }
        Ok(())
    }

    fn sequence_reached(&self) -> u64 {
        self.writer.sequence_reached()
    }

    fn truncate(&mut self) -> Result<(), Error> {
        self.writer.truncate()?;
        if let Some(flusher) = &self.flusher {
            flusher.mark_clean();
        }
        Ok(())
    }

    fn journal_file(&self) -> Option<&str> {
        Some(&self.file)
    }

    fn journal_collection_id(&self) -> Option<u128> {
        Some(self.writer.collection_id())
    }

    fn journal_path(&self) -> Option<&Path> {
        Some(self.writer.path())
    }

    fn durability(&self) -> Option<Durability> {
        Some(self.durability)
    }

    fn journal_bytes(&self) -> Option<u64> {
        self.writer.file_len().ok()
    }

    #[cfg(test)]
    fn flusher_watch(&self) -> Option<crate::flusher::Watch> {
        self.flusher.as_ref().map(Flusher::watch)
    }
}

/// What a recovery did, for a caller that wants to know rather than for the
/// engine, which needs none of it once the collection is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recovery {
    /// The sequence the checkpoint holds, from the manifest. Every record
    /// above it was replayed and every record at or below it was already in
    /// the directory.
    pub checkpoint_sequence: u64,
    /// The sequence the journal's header names as its first record's.
    pub first_sequence: u64,
    /// Records the reader accepted, damage aside.
    pub records_in_journal: usize,
    /// Records applied to the checkpoint.
    pub replayed: usize,
    /// Records at or below the checkpoint's sequence, which it already
    /// holds.
    pub skipped: usize,
    /// What stopped the reader, where anything did.
    pub damage: Option<JournalDamage>,
    /// The byte the good records end at, which the journal was cut to.
    pub good_bytes: u64,
    /// The directory was put back from `<name>.zdbold`, so the save that
    /// wrote it had been killed between its two renames.
    pub restored_from_aside: bool,
    /// The graph came from a rebuild rather than from the checkpoint's dump,
    /// so the graph the replay adds to is not the graph the crashed process
    /// held. Every record and every id is the same; the topology is not.
    pub graph_rebuilt: bool,
    /// A journal was read at all. False for a directory the manifest names
    /// none for, and for one opened as a checkpoint alone.
    pub journaled: bool,
}

impl Recovery {
    /// A directory with no journal, which is every directory saved by a
    /// collection holding none.
    pub(crate) fn unjournaled(restored_from_aside: bool, graph_rebuilt: bool) -> Self {
        Recovery {
            checkpoint_sequence: 0,
            first_sequence: 0,
            records_in_journal: 0,
            replayed: 0,
            skipped: 0,
            damage: None,
            good_bytes: 0,
            restored_from_aside,
            graph_rebuilt,
            journaled: false,
        }
    }

    /// Say what happened, once, at the level each part of it earns.
    pub(crate) fn report(&self, directory: &str) {
        if self.restored_from_aside {
            info!(target: LOG_TARGET, operation = "load_recover_aside",
                directory = directory,
                "A save had been interrupted between its two renames; the index is back in place"
            );
        }
        if !self.journaled {
            return;
        }
        match &self.damage {
            Some(JournalDamage::TornTail { at, sequence }) => {
                info!(target: LOG_TARGET, operation = "recover_torn_tail",
                    directory = directory,
                    at = at,
                    sequence = sequence,
                    "The journal ended inside a record; it was cut back to the last whole one"
                );
            }
            Some(JournalDamage::Corrupt { .. }) | None => {}
        }
        if self.graph_rebuilt {
            tracing::warn!(target: LOG_TARGET, operation = "recover_graph_rebuilt",
                directory = directory,
                replayed = self.replayed,
                "The checkpoint's graph dump was not read, so the graph was rebuilt from the \
                 records and the replayed records were added to that. Every record and every \
                 internal id is the checkpoint's; the graph's topology is not the one the \
                 process that wrote the journal held"
            );
        }
        info!(target: LOG_TARGET, operation = "recover_complete",
            directory = directory,
            checkpoint_sequence = self.checkpoint_sequence,
            records_in_journal = self.records_in_journal,
            replayed = self.replayed,
            skipped = self.skipped,
            "Recovery complete"
        );
    }
}

/// What a load does with the journal a directory's manifest names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JournalPolicy {
    /// Replay every record above the sequence the checkpoint holds, cut the
    /// journal back to its last whole record, reopen it for append under
    /// this policy and attach a sink over it. This is recovery, and it is
    /// what [`crate::Collection::load`] does under the default policy.
    Replay(Durability),
    /// Open the checkpoint alone. The journal is neither read nor reopened,
    /// no sink is attached, and the mutations it holds are not applied.
    ///
    /// For a directory copied without its sibling, which is the case the
    /// refusal exists for, and for a caller who wants the state as of the
    /// checkpoint and says so.
    CheckpointOnly,
}

/// Read the journal at `path`.
pub(crate) fn read_journal_bytes(path: &Path) -> Result<Vec<u8>, Error> {
    let bytes = std::fs::read(path).map_err(|error| Error::JournalIoFailed {
        path: path.to_path_buf(),
        what: "read",
        error: error.to_string(),
    })?;
    debug!(target: LOG_TARGET, "Read {} bytes of journal from {}", bytes.len(), path.display());
    Ok(bytes)
}

/// Hold a journal the reader has parsed to the checkpoint beside it.
///
/// Every check that can refuse the open runs here, before a record is
/// applied. The header itself the reader has already held. The collection id
/// must be the directory's, so a journal from another index is refused by
/// content. The first sequence must not sit above the one after the
/// checkpoint's, because the records between the two would be in neither.
/// And a corrupt middle is refused, since a record after it proves it landed
/// and skipping it would recover a state nothing acknowledged.
pub(crate) fn check_contents(
    contents: &zeusdb_vector_core::JournalContents<'_>,
    file: &str,
    directory_id: u128,
    checkpoint_sequence: u64,
) -> Result<(), Error> {
    if contents.header.collection_id != directory_id {
        return Err(Error::JournalNotThisCollection {
            file: file.to_string(),
            journal_id: collection_id_hex(contents.header.collection_id),
            directory_id: collection_id_hex(directory_id),
        });
    }
    if contents.header.first_sequence > checkpoint_sequence + 1 {
        return Err(Error::JournalStartsAfterCheckpoint {
            file: file.to_string(),
            first: contents.header.first_sequence,
            checkpoint: checkpoint_sequence,
        });
    }
    if let Some(refusal) = contents.refusal(file) {
        return Err(refusal);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_journal_is_a_sibling_of_the_directory() {
        let path = journal_path(Path::new("a/b/index.zdb")).unwrap();
        assert_eq!(path, Path::new("a/b/index.zdb.zdbwal"));
        assert_eq!(directory_of(&path).unwrap(), Path::new("a/b/index.zdb"));
        assert_eq!(directory_of(Path::new("a/b/index.zdb")), None);
        assert_eq!(directory_of(Path::new(".zdbwal")), None);
        assert_eq!(
            directory_of(Path::new("index.zdb.zdbwal")).unwrap(),
            Path::new("index.zdb")
        );
    }

    #[test]
    fn a_collection_id_round_trips_through_the_manifests_spelling() {
        for id in [
            0u128,
            1,
            u128::MAX,
            0x0123_4567_89ab_cdef_0123_4567_89ab_cdef,
        ] {
            let hex = collection_id_hex(id);
            assert_eq!(hex.len(), 32);
            assert_eq!(collection_id_from_hex(&hex), Some(id));
        }
        assert_eq!(collection_id_from_hex("not hexadecimal"), None);
    }

    #[test]
    fn the_three_policies_sit_on_the_two_commit_modes() {
        assert_eq!(Durability::default(), Durability::PerCall);
        assert_eq!(Durability::PerCall.commit_mode(), CommitMode::Sync);
        assert_eq!(
            Durability::PerInterval(Durability::DEFAULT_INTERVAL).commit_mode(),
            CommitMode::Deferred
        );
        assert_eq!(Durability::NoSync.commit_mode(), CommitMode::Deferred);
        assert_eq!(Durability::PerCall.name(), "call");
        assert_eq!(
            Durability::PerInterval(Duration::from_millis(3)).name(),
            "interval"
        );
        assert_eq!(Durability::NoSync.name(), "none");
        assert_eq!(Durability::PerCall.interval(), None);
        assert_eq!(
            Durability::PerInterval(Duration::from_millis(3)).interval(),
            Some(Duration::from_millis(3))
        );
        assert_eq!(Durability::DEFAULT_INTERVAL, Duration::from_millis(10));
    }
}
