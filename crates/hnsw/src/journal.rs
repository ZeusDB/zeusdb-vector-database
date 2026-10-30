//! The journal beside a directory, the sink that reaches it, the lock that
//! keeps a second writer off the collection, and the three durability
//! policies.
//!
//! [`zeusdb_vector_core`] holds what a journal is, being the segment header,
//! the record framing, the reader that classifies a torn tail, a withdrawn
//! region and a corrupt middle, and the writer. The collection holds where a
//! record comes from, being an [`OperationSink`](crate::OperationSink) it
//! hands every mutation to before the mutation runs. This module is the
//! join: a sink that is a journal writer, a sink that refuses every
//! mutation for a collection read from a medium it cannot write, the rule
//! for where the journal lives, what a checkpoint asks of it, the lock, and
//! what a caller is promised under each policy.
//!
//! # Where the journal lives
//!
//! `<name>.zdbjournal`, a directory of segments beside `<name>`, the sibling
//! of the directory in the way `<name>.zdbtmp` and `<name>.zdbold` already
//! are. A directory saved by a release from 0.11.0 to 0.13.0 has
//! `<name>.zdbwal` beside it instead, one file at journal format version 1,
//! which a load replays and then upgrades.
//!
//! It cannot live inside the directory. A save commits by renaming the
//! target aside to `<name>.zdbold` and the staging directory into its place,
//! and Windows refuses to rename a directory that holds an open file with
//! `Access is denied`, so the first of those two renames would fail for as
//! long as the journal was open. Avoiding that rename does not help either:
//! `remove_dir_all` on a directory holding an open file succeeds, so the
//! removal of `<name>.zdbold` after the second rename would take the live
//! journal with it.
//!
//! # How a directory and its journal are paired
//!
//! By content, not by name. `manifest.json` records the journal's name, the
//! collection id both belong to, and the sequence the checkpoint holds, and
//! every segment's header names the collection and the checkpoint its
//! records follow. A segment naming another collection is refused. A segment
//! naming another checkpoint is stale where the directory holds every record
//! it holds, and refused where it holds a record above the directory's
//! sequence, since it then belongs to another history of the collection. A
//! directory whose manifest names a journal that is not beside it is refused
//! by name, which is the copied-without-its-sibling case, unless the caller
//! asks for the checkpoint alone.
//!
//! # What a checkpoint does
//!
//! A save of a journaled collection is a checkpoint. Under the one hold of
//! the mutation guard that the save already takes: the journal is synced
//! whatever the policy, and a sync that fails is logged and passed, since
//! the save writes the collection from memory; the sequence the journal
//! reached is written onto the collection so the manifest records it; the
//! four save phases run; and then the journal rolls onto a new segment
//! naming the directory just committed, and every segment that directory
//! holds is deleted. A crash anywhere in that leaves either the previous
//! directory with segments that replay onto it, or the new directory with
//! segments it holds every record of, which the next load deletes.
//!
//! # A commit that fails
//!
//! The writer withdraws the call's records, by an abort record or by cutting
//! the segment back to the last boundary, and the error says which, or says
//! that neither landed. The collection then refuses every mutation until a
//! checkpoint commits, whatever the withdrawal achieved, because a failed
//! sync may also have lost records of earlier calls that the device had not
//! yet written, and only a checkpoint writes the collection from memory into
//! files the failure never touched. A load that replays an abort takes a
//! checkpoint before it journals again, for the same reason, since the fault
//! lives in memory alone.
//!
//! # The lock
//!
//! `<name>.zdblock` beside the directory, taken without waiting. A journaled
//! collection holds it exclusive for its life, a save holds it exclusive for
//! the save, and a load holds it shared while it reads, or exclusive where it
//! replays a journal, which its sink then keeps. A holder in another process,
//! or a second collection in this one, is refused with
//! [`Error::CollectionLocked`]. A filesystem that cannot lock refuses
//! journaling and saving with [`Error::LockUnavailable`], and a load there
//! goes on unlocked and opens a journaled directory read only, since it would
//! have to journal. A child forked through Python's `os.fork` closes its copy
//! of the lock file just after the fork, so the lock stays with the parent.
//! The child still shares the parent's open journal, so the sink refuses
//! every mutation there with [`Error::JournalForked`].
//!
//! # A medium that cannot be written
//!
//! A load decides at its first write, before it has changed anything: the
//! lock file opened for writing, the staging directory of an upgrade, or the
//! journal's last segment opened or created. A refusal the system gives for a
//! medium that cannot be written or for a missing permission makes a load of
//! a journaled directory read only. It replays the journal in memory and
//! changes nothing on the medium. Its collection carries a [`ReadOnlySink`],
//! which refuses every mutation with [`Error::CollectionReadOnly`], and saves
//! into another directory alone. Any other failure, and any failure after
//! the load changed something, refuses the load.
//!
//! A directory holding a feature this build does not know, which its
//! manifest marks `read_only`, opens the same way whatever the medium, journaled
//! or not and under `checkpoint_only=True` too. Its collection refuses every
//! change, every save, a checkpoint, a journal and a backup with
//! [`Error::FeatureReadOnly`], since a directory this build wrote would not
//! hold the feature.
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
//! | [`Durability::PerInterval`] | every record is in the kernel, and a thread syncs the journal within the interval | nothing acknowledged is lost | up to one interval of acknowledged calls is lost |
//! | [`Durability::NoSync`] | every record is in the kernel | nothing acknowledged is lost | everything since the kernel last wrote the journal is lost |
//!
//! The middle column is the same for all three because a process abort
//! loses nothing the kernel holds, and every policy hands the bytes to the
//! kernel before the call returns. The right column is what each policy
//! syncs, and what "acknowledged" means is that the call returned without
//! an error. A record refused by the call, or a call that raised, was never
//! promised under any policy. A checkpoint syncs the journal whatever the
//! policy, so the loss window of the two weaker policies closes at every
//! save. The table holds for a device that stores what it acknowledges.
//! After a sync fails, no system a wheel ships for says which bytes reached
//! the device, so the table holds again once a checkpoint commits, which is
//! why the collection refuses every mutation until then.
//!
//! The default is [`Durability::PerCall`], because it is the one a caller
//! who has not thought about durability should have, and what it costs is a
//! function of the caller's batch size rather than of the engine: one flush
//! per call, which is a millisecond or so, against the insert work the call
//! does.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::{info, warn};
use zeusdb_vector_core::{
    read_legacy_journal, AbortReason, CommitMode, Error, FsStorage, JournalDamage, JournalWriter,
    LegacyContents, LockFailure, LockGuard, LockMode, Locked, OperationKind, SegmentBase, Storage,
    Withdrawal,
};

use crate::collection::OperationSink;
use crate::flusher::Flusher;

/// The target the log records in this module carry. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::journal";

/// The suffix a collection's journal takes beside its directory.
pub use zeusdb_vector_core::JOURNAL_SUFFIX;

/// What a journaled collection promises about a record once the call that
/// added it has returned. See the module's table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Durability {
    /// Every record of a call is on the device before the call returns, or
    /// the call did not return. One flush per call. The default.
    #[default]
    PerCall,
    /// Every record is in the kernel when the call returns, and a thread of
    /// the collection's own syncs the journal within this long of the first
    /// call that dirtied it. What a caller trades is that an operating
    /// system crash or a power loss inside the window loses the calls that
    /// fell in it, and what they get is a single-record `add` that does not
    /// wait on a flush.
    PerInterval(Duration),
    /// Every record is in the kernel when the call returns and nothing syncs
    /// the journal until the next checkpoint. A process abort still loses
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
    Ok(FsStorage::at(target)?.journal_path().to_path_buf())
}

/// The directory a journal at `journal` sits beside, being
/// [`journal_path`] undone. `None` where the name does not carry the
/// suffix.
pub fn directory_of(journal: &Path) -> Option<PathBuf> {
    let name = journal.file_name()?.to_str()?;
    let stem = name.strip_suffix(JOURNAL_SUFFIX)?;
    if stem.is_empty() {
        return None;
    }
    Some(journal.parent().unwrap_or_else(|| Path::new("")).join(stem))
}

/// An id as the manifest spells it, being 32 lower case hexadecimal digits.
/// Every id the manifest carries takes it: a collection's, a snapshot's and
/// a parent's.
pub(crate) fn id_hex(id: u128) -> String {
    format!("{:032x}", id)
}

/// An id read back from the manifest's spelling, being 32 hexadecimal digits
/// and nothing else, so the id read back is the id written. `from_str_radix`
/// alone reads three digits, or a sign before the digits, as a whole id, and
/// a collection would then report an id its directory does not spell.
pub(crate) fn id_from_hex(hex: &str) -> Option<u128> {
    if hex.len() != 32 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u128::from_str_radix(hex, 16).ok()
}

// ============================================================================
// THE LOCK
// ============================================================================

/// What a lock file records of the holder that took it exclusive, being
/// four lines: the process, the host, the mode and the moment it was taken.
/// Nothing reads it but a refusal's message, and no decision rests on it.
pub(crate) fn holder_record(mode: LockMode) -> String {
    format!(
        "process {}\nhost {}\nmode {}\nsince {}\n",
        std::process::id(),
        host_name(),
        mode.name(),
        chrono::Utc::now().to_rfc3339()
    )
}

/// The machine's name, from what each system offers without a library:
/// `COMPUTERNAME` on Windows, the kernel's hostname on Linux and `HOSTNAME`
/// elsewhere.
fn host_name() -> String {
    #[cfg(windows)]
    let name = std::env::var("COMPUTERNAME").ok();
    #[cfg(target_os = "linux")]
    let name = std::fs::read_to_string("/proc/sys/kernel/hostname").ok();
    #[cfg(not(any(windows, target_os = "linux")))]
    let name = std::env::var("HOSTNAME").ok();
    name.map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "an unknown host".to_string())
}

/// A lock that was not taken, as the caller's refusal. `directory` is the
/// caller's own words for the directory.
pub(crate) fn lock_refusal(storage: &dyn Storage, directory: &str, failure: LockFailure) -> Error {
    match failure {
        LockFailure::Held { holder } => Error::CollectionLocked {
            path: directory.to_string(),
            holder,
        },
        LockFailure::Unavailable(error) | LockFailure::Failed(error) => Error::LockUnavailable {
            path: storage.lock_path().display().to_string(),
            error: error.to_string(),
        },
    }
}

/// The lock a save into `storage`'s directory holds for the save, taken
/// exclusive. `None` where the medium refuses writes, since the save then
/// meets that refusal at its first write and fails with it as it always has.
pub(crate) fn lock_for_save(
    storage: &dyn Storage,
    directory: &str,
) -> Result<Option<Box<dyn LockGuard>>, Error> {
    match storage.lock(LockMode::Exclusive, &holder_record(LockMode::Exclusive)) {
        Ok(Locked::Writable(guard)) => Ok(Some(guard)),
        Ok(Locked::ReadOnly { guard, .. }) => Ok(guard),
        Err(failure) => Err(lock_refusal(storage, directory, failure)),
    }
}

/// The lock a journal opened beside `storage`'s directory holds for its
/// life, taken exclusive. A medium that refuses the lock file's write
/// refuses the journal, since every record would have to be written there.
pub(crate) fn lock_for_journal(
    storage: &dyn Storage,
    directory: &str,
) -> Result<Box<dyn LockGuard>, Error> {
    match storage.lock(LockMode::Exclusive, &holder_record(LockMode::Exclusive)) {
        Ok(Locked::Writable(guard)) => Ok(guard),
        Ok(Locked::ReadOnly { reason, .. }) => Err(Error::JournalIoFailed {
            path: storage.journal_path().to_path_buf(),
            what: "create",
            error: reason.to_string(),
        }),
        Err(failure) => Err(lock_refusal(storage, directory, failure)),
    }
}

// ============================================================================
// THE SINK OVER A JOURNAL
// ============================================================================

/// The sink a journaled collection hands every record to.
///
/// One journal writer, the policy its commits keep, the thread the interval
/// policy syncs from, and the collection's lock, held exclusive for the
/// sink's life. `append` drops the sequence the writer returns, because the
/// collection has no use for it: what a checkpoint records is the sequence
/// the journal has reached, which [`OperationSink::sequence_reached`]
/// reports after the fact.
///
/// The fields drop in their order: the writer, then the thread, which syncs
/// the active segment once more through its own handle, then the lock, so
/// the lock is let go only once nothing more is written.
#[derive(Debug)]
pub struct JournalSink {
    writer: JournalWriter,
    durability: Durability,
    /// The thread, under the interval policy alone. Dropped with the sink,
    /// which joins it.
    flusher: Option<Flusher>,
    /// The journal's name, which the manifest records, held here so the
    /// checkpoint writing the manifest does not derive it a second time.
    file: String,
    /// The collection's lock, where one was taken.
    _lock: Option<Box<dyn LockGuard>>,
}

impl JournalSink {
    /// Create the journal `storage` keeps, for `collection_id`, emptying any
    /// directory of segments there, with its first record at sequence one,
    /// under `lock`. It holds no segment until the checkpoint the caller
    /// writes next rolls it onto one.
    pub fn create(
        storage: &dyn Storage,
        collection_id: u128,
        durability: Durability,
        lock: Box<dyn LockGuard>,
    ) -> Result<Self, Error> {
        Self::create_at(storage, collection_id, 0, durability, Some(lock))
    }

    /// [`JournalSink::create`], carrying the sequence on after `reached`,
    /// which is what an upgrade journals into: the records it replayed from
    /// the file it replaces took every sequence up to `reached`.
    pub fn create_at(
        storage: &dyn Storage,
        collection_id: u128,
        reached: u64,
        durability: Durability,
        lock: Option<Box<dyn LockGuard>>,
    ) -> Result<Self, Error> {
        storage
            .create_journal()
            .map_err(|error| Error::JournalIoFailed {
                path: storage.journal_path().to_path_buf(),
                what: "create",
                error: error.to_string(),
            })?;
        let next = reached
            .checked_add(1)
            .ok_or_else(|| Error::JournalIoFailed {
                path: storage.journal_path().to_path_buf(),
                what: "create",
                error: "the sequence space is exhausted".into(),
            })?;
        let writer = JournalWriter::fresh(storage.journal_dir(), collection_id, next)?;
        Self::over(writer, durability, lock)
    }

    /// Take a writer a load opened, under the lock the load took.
    pub fn from_writer(
        writer: JournalWriter,
        durability: Durability,
        lock: Option<Box<dyn LockGuard>>,
    ) -> Result<Self, Error> {
        Self::over(writer, durability, lock)
    }

    fn over(
        writer: JournalWriter,
        durability: Durability,
        lock: Option<Box<dyn LockGuard>>,
    ) -> Result<Self, Error> {
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
            _lock: lock,
        })
    }

    /// The collection the journal's segments name.
    pub fn collection_id(&self) -> u128 {
        self.writer.collection_id()
    }

    /// Where the journal is, being its directory.
    pub fn path(&self) -> &Path {
        self.writer.path()
    }

    /// The policy every commit keeps.
    pub fn durability(&self) -> Durability {
        self.durability
    }

    /// A handle that syncs the active segment and does nothing else, and
    /// follows the active segment through every roll.
    pub fn sync_handle(&self) -> zeusdb_vector_core::JournalSyncHandle {
        self.writer.sync_handle()
    }

    /// Delete the stale segments a resumed journal holds, which the
    /// directory's checkpoint holds every record of, and return a failed
    /// sync of the journal's directory after it.
    pub fn remove_stale(&mut self) -> Result<(), Error> {
        self.writer.remove_stale()
    }

    /// The writer as a test that has to reach the fork check without forking
    /// sees it, being one opened by another process.
    #[cfg(test)]
    pub(crate) fn pretend_forked(&mut self) {
        self.writer.pretend_forked();
    }

    /// Undo [`JournalSink::pretend_forked`], so the sink drops as its own
    /// process's does.
    #[cfg(test)]
    pub(crate) fn pretend_own_process(&mut self) {
        self.writer.pretend_own_process();
    }
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// A child forked from the process that opened the journal shares its open
/// files and holds no copy of its thread. Dropping the sink there does not
/// join a thread the child does not have. Its lock drops as a lock does in a
/// child, which closes the child's copy of the lock file without unlocking it
/// where the child has not closed it already, and leaves the parent's lock
/// held, since a close lets a lock go only with the last copy of its file.
impl Drop for JournalSink {
    fn drop(&mut self) {
        if !self.writer.in_its_own_process() {
            if let Some(flusher) = self.flusher.take() {
                std::mem::forget(flusher);
            }
        }
    }
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

    /// Under the per-call policy, the commit record and the flush, and a
    /// flush that fails withdraws the call. Under the other two, the commit
    /// record with nothing on the device: the interval policy's thread is
    /// told the journal is dirty, and a failed sync the thread holds
    /// withdraws this call and is its failure, so a device that will not
    /// flush is reported at the next call rather than never.
    fn commit(&mut self) -> Result<(), Error> {
        match self.durability.commit_mode() {
            CommitMode::Sync => self.writer.commit(CommitMode::Sync),
            CommitMode::Deferred => {
                if let Some(failure) = self.flusher.as_ref().and_then(Flusher::take_failure) {
                    let withdrawal = self.writer.withdraw(AbortReason::SyncFailed);
                    return Err(withdrawal.attach_to(failure));
                }
                self.writer.commit(CommitMode::Deferred)?;
                if let Some(flusher) = &self.flusher {
                    flusher.mark_dirty();
                }
                Ok(())
            }
        }
    }

    /// An abort record naming the call's first sequence, synced, or the
    /// segment cut back to the last boundary, as a failed commit withdraws.
    fn withdraw(&mut self) -> Withdrawal {
        self.writer.withdraw(AbortReason::WriteFailed)
    }

    /// Sync the active segment, which leaves the interval policy's thread
    /// nothing to sync until the next commit. A failed sync the thread holds
    /// stays held, whatever this sync says.
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

    /// The roll waits for a sync the interval policy's thread has begun to
    /// end and keeps the thread from beginning another until the active
    /// segment is replaced, so every sync of the thread is of a segment
    /// wholly before the roll or wholly after it. A roll that lands discards
    /// a failure the thread holds, since the checkpoint holds every record
    /// the replaced segments held. A child forked from the process that
    /// opened the journal has no copy of the thread, so its roll leaves the
    /// thread's state alone, and the writer refuses it.
    fn roll(&mut self, base: SegmentBase) -> Result<(), Error> {
        let rolling = match &self.flusher {
            Some(flusher) if self.writer.in_its_own_process() => Some(flusher.roll()),
            _ => None,
        };
        self.writer.roll_to(base)?;
        if let Some(rolling) = rolling {
            rolling.committed();
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
        Some(self.writer.journal_bytes())
    }

    fn journal_segments(&self) -> Option<usize> {
        Some(self.writer.segments())
    }

    fn committed_operations(&self) -> Option<u64> {
        Some(self.writer.committed_operations())
    }

    #[cfg(test)]
    fn flusher_watch(&self) -> Option<crate::flusher::Watch> {
        self.flusher.as_ref().map(Flusher::watch)
    }
}

// ============================================================================
// THE SINK OF A COLLECTION READ ONLY
// ============================================================================

/// Why a collection takes no record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReadOnlyCause {
    /// The medium refused the load's first write, for the reason the system
    /// gave.
    Medium(String),
    /// The directory holds these features, which this build does not know
    /// and its manifest marks `read_only`, in name order.
    Features(Vec<String>),
}

impl ReadOnlyCause {
    /// The reason the load's report and a medium's refusal name.
    pub(crate) fn reason(&self) -> String {
        match self {
            ReadOnlyCause::Medium(reason) => reason.clone(),
            ReadOnlyCause::Features(features) => format!(
                "it holds the {} {}, which this build does not know and its manifest marks \
                 read_only",
                if features.len() == 1 {
                    "feature"
                } else {
                    "features"
                },
                quoted(features)
            ),
        }
    }
}

/// Names quoted and joined as a sentence lists them.
fn quoted(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|name| format!("'{}'", name)).collect();
    match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {}", rest.join(", "), last),
        Some((last, _)) => last.clone(),
        None => String::new(),
    }
}

/// The sink a collection read only carries, read from a medium it cannot
/// write or from a directory holding features this build does not know,
/// marked `read_only`.
///
/// It takes no record, so every mutation is refused before it runs: with
/// [`Error::CollectionReadOnly`] for a medium, naming the directory and the
/// system's refusal, and with [`Error::FeatureReadOnly`] for features,
/// naming the directory and the features. It names no journal, so the
/// collection reports none, and the sequence it reports is the last one the
/// load replayed, which a save into another directory records as the
/// sequence its state holds.
#[derive(Debug)]
pub(crate) struct ReadOnlySink {
    directory: PathBuf,
    reason: String,
    features: Option<Vec<String>>,
    sequence: u64,
}

impl ReadOnlySink {
    pub(crate) fn new(directory: PathBuf, cause: ReadOnlyCause, sequence: u64) -> Self {
        let reason = cause.reason();
        let features = match cause {
            ReadOnlyCause::Medium(_) => None,
            ReadOnlyCause::Features(features) => Some(features),
        };
        ReadOnlySink {
            directory,
            reason,
            features,
            sequence,
        }
    }

    fn refusal(&self) -> Error {
        match &self.features {
            Some(features) => Error::FeatureReadOnly {
                path: self.directory.display().to_string(),
                features: features.clone(),
            },
            None => Error::CollectionReadOnly {
                directory: self.directory.display().to_string(),
                reason: self.reason.clone(),
            },
        }
    }
}

impl OperationSink for ReadOnlySink {
    fn append(&mut self, _kind: OperationKind, _payload: &[u8]) -> Result<(), Error> {
        Err(self.refusal())
    }

    /// Nothing was appended, since every append is refused.
    fn commit(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn sync(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn sequence_reached(&self) -> u64 {
        self.sequence
    }

    fn read_only(&self) -> Option<(&Path, &str)> {
        Some((&self.directory, &self.reason))
    }

    fn read_only_features(&self) -> Option<&[String]> {
        self.features.as_deref()
    }
}

// ============================================================================
// WHAT A RECOVERY DID
// ============================================================================

/// What stopped the reader or broke the journal's history, as a caller reads
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Damage {
    /// `torn_tail`, `corrupt`, `header`, `structure`, `decode` or `apply`.
    /// See [`Cut`](zeusdb_vector_core::Cut) for the middle four, and
    /// `decode` and `apply` are a record whose checksum holds and which
    /// would not decode or apply.
    pub kind: &'static str,
    /// The file the damage is in, by name.
    pub segment: String,
    /// The byte of that file the damage is at.
    pub at: u64,
    /// The sequence the record there carries or was expected to carry.
    pub sequence: u64,
    /// How it fails. `None` for a torn tail.
    pub detail: Option<String>,
}

impl Damage {
    /// A torn tail or a corrupt middle the reader reported, in `file`.
    pub(crate) fn of(damage: &JournalDamage, file: &str) -> Self {
        let segment = file_name_of(Path::new(file));
        match damage {
            JournalDamage::TornTail { at, sequence } => Damage {
                kind: "torn_tail",
                segment,
                at: *at,
                sequence: *sequence,
                detail: None,
            },
            JournalDamage::Corrupt {
                at,
                sequence,
                detail,
            } => Damage {
                kind: "corrupt",
                segment,
                at: *at,
                sequence: *sequence,
                detail: Some(detail.clone()),
            },
        }
    }
}

/// What a repair set aside.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repair {
    /// The last sequence kept.
    pub kept_through: u64,
    /// The first sequence set aside past the checkpoint, where the repair
    /// cut the journal, and `None` where it set aside only stale segments,
    /// which hold no record past the checkpoint.
    pub discarded_from: Option<u64>,
    /// The whole operation records the bytes set aside hold, counted by the
    /// reader's own scan.
    pub records_discarded: u64,
    /// The bytes set aside, every later segment's included.
    pub bytes_discarded: u64,
    /// The folder the bytes were copied into, or `None` where the medium
    /// refused the write and nothing was set aside.
    pub quarantine: Option<PathBuf>,
}

/// Why a load took a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointTaken {
    /// The directory was written again at 5.0.0 with a journal of segments.
    Upgrade,
    /// The journal held a call withdrawn after a failed commit.
    Abort,
    /// A repair cut the journal, or found one a killed repair left
    /// unfinished. One checkpoint serves an abort the repair also met.
    Repair,
}

impl CheckpointTaken {
    /// The name a caller reads.
    pub fn name(self) -> &'static str {
        match self {
            CheckpointTaken::Upgrade => "upgrade",
            CheckpointTaken::Abort => "abort",
            CheckpointTaken::Repair => "repair",
        }
    }
}

/// What a recovery did, for a caller that wants to know rather than for the
/// engine, which needs none of it once the collection is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recovery {
    /// The format version the directory declared when it was read.
    pub format_version: String,
    /// The sequence the checkpoint holds, from the manifest. Every record
    /// above it was replayed and every record at or below it was already in
    /// the directory.
    pub checkpoint_sequence: u64,
    /// The sequence the journal's first record above the checkpoint takes,
    /// being the first live segment's name or the first sequence of a
    /// journal held in one file.
    pub first_sequence: u64,
    /// Operation records above the checkpoint that the reader reached,
    /// damage aside.
    pub records_in_journal: usize,
    /// Records applied to the checkpoint.
    pub replayed: usize,
    /// Records of a call whose commit failed, which replay as the live
    /// collection left them: an insert's internal id consumed and a term
    /// interned, and nothing else.
    pub withdrawn: usize,
    /// Records at or below the checkpoint's sequence, which it already
    /// holds.
    pub skipped: usize,
    /// What stopped the reader, or what a repair cut the journal for, where
    /// anything did.
    pub damage: Option<Damage>,
    /// The byte the good records end at in the last segment, which the
    /// journal was cut to.
    pub good_bytes: u64,
    /// The segments the journal held when it was read, and 0 for a journal
    /// held in one file.
    pub segments: usize,
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
    /// The format the directory was upgraded from, where the load replayed a
    /// journal held in one file and wrote the directory again at 5.0.0.
    pub upgraded_from: Option<String>,
    /// Why the load took a checkpoint, where it took one.
    pub checkpoint_taken: Option<CheckpointTaken>,
    /// Why the collection is read only, where the medium refused the load's
    /// first write or the directory holds features this build does not know
    /// and its manifest marks `read_only`.
    pub read_only: Option<String>,
    /// The features that make the collection read only, in name order, and
    /// none where the medium did.
    pub read_only_features: Vec<String>,
    /// What a repair set aside, where the load was asked to repair and the
    /// journal was damaged.
    pub repair: Option<Repair>,
}

impl Recovery {
    /// A directory with no journal, which is every directory saved by a
    /// collection holding none.
    pub(crate) fn unjournaled(
        format_version: String,
        restored_from_aside: bool,
        graph_rebuilt: bool,
    ) -> Self {
        Recovery {
            format_version,
            checkpoint_sequence: 0,
            first_sequence: 0,
            records_in_journal: 0,
            replayed: 0,
            withdrawn: 0,
            skipped: 0,
            damage: None,
            good_bytes: 0,
            segments: 0,
            restored_from_aside,
            graph_rebuilt,
            journaled: false,
            upgraded_from: None,
            checkpoint_taken: None,
            read_only: None,
            read_only_features: Vec::new(),
            repair: None,
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
            self.report_read_only(directory);
            return;
        }
        if let Some(damage) = self.damage.as_ref().filter(|d| d.kind == "torn_tail") {
            info!(target: LOG_TARGET, operation = "recover_torn_tail",
                directory = directory,
                at = damage.at,
                sequence = damage.sequence,
                "The journal ended inside a record or after records no commit closed; it was cut \
                 back to the last whole call"
            );
        }
        if let (Some(repair), Some(damage)) = (&self.repair, &self.damage) {
            let quarantine = repair
                .quarantine
                .as_ref()
                .map_or_else(|| "none".to_string(), |path| path.display().to_string());
            if repair.discarded_from.is_some() {
                warn!(target: LOG_TARGET, operation = "load_repair",
                    directory = directory,
                    kind = damage.kind,
                    segment = damage.segment.as_str(),
                    at = damage.at,
                    sequence = damage.sequence,
                    detail = damage.detail.as_deref().unwrap_or(""),
                    kept_through = repair.kept_through,
                    discarded_from = repair.discarded_from,
                    records_discarded = repair.records_discarded,
                    bytes_discarded = repair.bytes_discarded,
                    quarantine = %quarantine,
                    "The journal was damaged and was repaired: every record before the damaged \
                     call was kept, and that call and every record after it were set aside in the \
                     quarantine folder"
                );
            } else {
                warn!(target: LOG_TARGET, operation = "load_repair",
                    directory = directory,
                    kind = damage.kind,
                    segment = damage.segment.as_str(),
                    detail = damage.detail.as_deref().unwrap_or(""),
                    kept_through = repair.kept_through,
                    bytes_discarded = repair.bytes_discarded,
                    quarantine = %quarantine,
                    "A stale segment of the journal, named at or below the checkpoint's sequence, \
                     could not be read and was repaired: it was set aside whole in the quarantine \
                     folder, and every record of the journal was kept"
                );
            }
        }
        if self.withdrawn > 0 {
            info!(target: LOG_TARGET, operation = "recover_withdrawn",
                directory = directory,
                withdrawn = self.withdrawn,
                "The journal held records of calls whose commit failed; they were not applied"
            );
        }
        if self.graph_rebuilt {
            warn!(target: LOG_TARGET, operation = "recover_graph_rebuilt",
                directory = directory,
                replayed = self.replayed,
                "The checkpoint's graph dump was not read, so the graph was rebuilt from the \
                 records and the replayed records were added to that. Every record and every \
                 internal id is the checkpoint's; the graph's topology is not the one the \
                 process that wrote the journal held"
            );
        }
        if let Some(from) = &self.upgraded_from {
            info!(target: LOG_TARGET, operation = "load_upgrade",
                directory = directory,
                from = from.as_str(),
                "The directory was written again at format 5.0.0 with a journal of segments, and \
                 its journal of format version 1 was replayed into it; releases before 0.14.0 do \
                 not open it now"
            );
        } else if self.checkpoint_taken == Some(CheckpointTaken::Abort) {
            info!(target: LOG_TARGET, operation = "recover_checkpoint",
                directory = directory,
                "The journal held a call withdrawn after a failed commit, so the load took a \
                 checkpoint before journaling again"
            );
        }
        self.report_read_only(directory);
        info!(target: LOG_TARGET, operation = "recover_complete",
            directory = directory,
            checkpoint_sequence = self.checkpoint_sequence,
            records_in_journal = self.records_in_journal,
            replayed = self.replayed,
            withdrawn = self.withdrawn,
            skipped = self.skipped,
            segments = self.segments,
            "Recovery complete"
        );
    }

    /// The record a load opened read only writes, at `warn`.
    fn report_read_only(&self, directory: &str) {
        let Some(reason) = &self.read_only else {
            return;
        };
        if self.read_only_features.is_empty() {
            warn!(target: LOG_TARGET, operation = "load_read_only",
                directory = directory,
                reason = reason.as_str(),
                format_version = self.format_version.as_str(),
                replayed = self.replayed,
                "The directory was opened read only, since this process cannot write it; its \
                 journal was replayed in memory, nothing on the medium changed, and every \
                 change to the collection is refused"
            );
        } else {
            warn!(target: LOG_TARGET, operation = "load_read_only",
                directory = directory,
                reason = reason.as_str(),
                features = self.read_only_features.join(", ").as_str(),
                format_version = self.format_version.as_str(),
                replayed = self.replayed,
                "The directory was opened read only, since it holds features this build does \
                 not know and its manifest marks read_only; nothing on the medium changed, and \
                 every change, save, checkpoint, journal and backup is refused"
            );
        }
    }

    /// What a caller who loaded the directory is told where the collection
    /// came back read only, being the warning the binding raises. A repair
    /// says it in [`Recovery::repair_warning`] instead.
    pub fn read_only_warning(&self, directory: &str) -> Option<String> {
        if self.repair.is_some() {
            return None;
        }
        let reason = self.read_only.as_ref()?;
        if !self.read_only_features.is_empty() {
            let replayed = if self.journaled {
                format!(
                    ", and {} records of its journal were replayed in memory",
                    self.replayed
                )
            } else {
                String::new()
            };
            return Some(format!(
                "The directory {} was opened read only, since {}. It stays at format {}{}. \
                 Every change, save, checkpoint, journal and backup is refused. Upgrade the \
                 package to change it.",
                directory, reason, self.format_version, replayed
            ));
        }
        Some(format!(
            "The directory {} was opened read only, since this process cannot write it: {}. It \
             stays at format {}, and {} records of its journal were replayed in memory. Every \
             change to the collection is refused; save it into another directory to copy it \
             somewhere writable.",
            directory, reason, self.format_version, self.replayed
        ))
    }

    /// What a caller who loaded the directory with a repair is told where
    /// the repair cut the journal, being the warning the binding raises.
    /// Where the medium refused the repair's first write, it names that
    /// refusal as well, and nothing was set aside.
    pub fn repair_warning(&self, directory: &str) -> Option<String> {
        let repair = self.repair.as_ref()?;
        let damage = self.damage.as_ref()?;
        let found = format!(
            "The journal of the directory {} was damaged at byte {} of {} ({}, at record {}{}).",
            directory,
            damage.at,
            damage.segment,
            damage.kind.replace('_', " "),
            damage.sequence,
            damage
                .detail
                .as_deref()
                .map_or_else(String::new, |detail| format!(": {detail}"))
        );
        let (kept, verb, done) = match repair.discarded_from {
            Some(from) => (
                format!(
                    "Every record through {} was kept, and record {} and the {} records from it \
                     on, {} bytes,",
                    repair.kept_through, from, repair.records_discarded, repair.bytes_discarded
                ),
                "were",
                "The journal was cut there and a checkpoint written.",
            ),
            None => (
                format!(
                    "Every record through {} was kept, and the segment {}, {} bytes, which is \
                     named at or below the checkpoint's sequence and so holds no record past it,",
                    repair.kept_through, damage.segment, repair.bytes_discarded
                ),
                "was",
                "The segment was removed from the journal and a checkpoint written.",
            ),
        };
        Some(match (&repair.quarantine, &self.read_only) {
            (Some(folder), _) => format!(
                "{found} {kept} {verb} set aside in {}. {done}",
                folder.display()
            ),
            (None, Some(reason)) if !self.read_only_features.is_empty() => format!(
                "{found} {kept} {verb} left out of the replay. The directory was opened read \
                 only, since {}. Nothing on the medium changed, and every change, save, \
                 checkpoint, journal and backup is refused.",
                reason
            ),
            (None, reason) => format!(
                "{found} {kept} {verb} left out of the replay. The directory was opened read \
                 only, since this process cannot write it: {}. Nothing on the medium changed, \
                 every change to the collection is refused, and a save into another directory \
                 copies what was kept.",
                reason.as_deref().unwrap_or("the medium refused the write")
            ),
        })
    }
}

/// What a load does with the journal a directory's manifest names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JournalPolicy {
    /// Replay every record above the sequence the checkpoint holds, cut the
    /// journal back to its last whole call, reopen it for append under this
    /// policy and attach a sink over it. This is recovery, and it is what
    /// [`crate::Collection::load`] does under the default policy.
    Replay(Durability),
    /// [`JournalPolicy::Replay`], under the exclusive lock, where damage to
    /// the journal's history is cut away rather than refused: every whole
    /// call before the damage is replayed, the damaged call and everything
    /// after it are copied into the journal's `quarantine` folder, the
    /// journal is cut and a checkpoint seals it. Where nothing is damaged
    /// and no repair is unfinished it is a replay and writes nothing more.
    Repair(Durability),
    /// Open the checkpoint alone. The journal is neither read nor reopened,
    /// no sink is attached, and the mutations it holds are not applied.
    ///
    /// For a directory copied without its sibling, which is the case the
    /// refusal exists for, and for a caller who wants the state as of the
    /// checkpoint and says so.
    CheckpointOnly,
}

/// Hold a journal of format version 1 the reader has parsed to the
/// checkpoint beside it.
///
/// Every check that can refuse the open runs here, before a record is
/// applied. The header itself the reader has already held. The collection id
/// must be the directory's, so a journal from another index is refused by
/// content. The first sequence must not sit above the one after the
/// checkpoint's, because the records between the two would be in neither.
/// And a corrupt middle is refused, since a record after it proves it landed
/// and skipping it would recover a state nothing acknowledged, unless the
/// load is `repairing` and cuts it away.
pub(crate) fn check_legacy_contents(
    contents: &LegacyContents<'_>,
    file: &str,
    directory_id: u128,
    checkpoint_sequence: u64,
    repairing: bool,
) -> Result<(), Error> {
    if contents.header.collection_id != directory_id {
        return Err(Error::JournalNotThisCollection {
            file: file.to_string(),
            journal_id: id_hex(contents.header.collection_id),
            directory_id: id_hex(directory_id),
        });
    }
    if contents.header.first_sequence > checkpoint_sequence.saturating_add(1) {
        return Err(Error::JournalStartsAfterCheckpoint {
            file: file.to_string(),
            first: contents.header.first_sequence,
            checkpoint: checkpoint_sequence,
        });
    }
    if let Some(refusal) = contents.refusal(file).filter(|_| !repairing) {
        return Err(refusal);
    }
    Ok(())
}

/// A journal of format version 1 left beside a directory that names a
/// journal of segments, which an upgrade a crash interrupted leaves.
///
/// Deleted where it names the directory's collection, reads without a
/// corrupt middle and holds no whole record above the directory's sequence,
/// since the directory then holds every record it holds. Otherwise it is
/// logged and left, and so is one the system will not delete. A deletion is
/// made durable through [`Storage::sync_in_place`], and a failed sync there
/// is returned.
pub(crate) fn remove_covered_legacy(
    storage: &dyn Storage,
    directory_id: u128,
    sequence: u64,
) -> Result<(), Error> {
    if !storage.legacy_journal_exists() {
        return Ok(());
    }
    let path = storage.legacy_journal_path().display().to_string();
    let covered = storage.read_legacy_journal().ok().is_some_and(|bytes| {
        read_legacy_journal(&bytes, &path).is_ok_and(|contents| {
            contents.header.collection_id == directory_id
                && !matches!(contents.damage, Some(JournalDamage::Corrupt { .. }))
                && contents
                    .records
                    .iter()
                    .all(|record| record.sequence <= sequence)
        })
    });
    if !covered {
        warn!(target: LOG_TARGET, operation = "journal_legacy_left",
            journal = path.as_str(),
            "A journal of format version 1 sits beside a directory whose journal is a directory \
             of segments, and the directory does not hold every record in it; it was left where \
             it is"
        );
        return Ok(());
    }
    match storage.remove_legacy_journal() {
        Ok(()) => {
            info!(target: LOG_TARGET, operation = "journal_legacy_removed",
                journal = path.as_str(),
                "A journal of format version 1 whose records the directory holds was deleted"
            );
            sync_in_place(storage)
        }
        Err(error) => {
            warn!(target: LOG_TARGET, operation = "journal_legacy_kept",
                journal = path.as_str(),
                error = %error,
                "A journal of format version 1 whose records the directory holds could not be \
                 deleted; the next load tries again"
            );
            Ok(())
        }
    }
}

/// Make the directory `storage` keeps durable, its entries and its name,
/// with a failure in the words a load raises it in.
pub(crate) fn sync_in_place(storage: &dyn Storage) -> Result<(), Error> {
    storage
        .sync_in_place()
        .map_err(|error| Error::DirectoryNotDurable {
            directory: storage.dir().locate(""),
            error: error.to_string(),
        })
}

/// Whether `a` and `b` name one directory, compared as absolute paths, since
/// `index.zdb` and `./index.zdb` are the same directory and compare unequal
/// as written.
pub(crate) fn same_directory(a: &Path, b: &Path) -> bool {
    let absolute = |path: &Path| std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    absolute(a) == absolute(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_journal_is_a_sibling_of_the_directory() {
        let path = journal_path(Path::new("a/b/index.zdb")).unwrap();
        assert_eq!(path, Path::new("a/b/index.zdb.zdbjournal"));
        assert_eq!(directory_of(&path).unwrap(), Path::new("a/b/index.zdb"));
        assert_eq!(directory_of(Path::new("a/b/index.zdb")), None);
        assert_eq!(directory_of(Path::new(".zdbjournal")), None);
        assert_eq!(directory_of(Path::new("index.zdb.zdbwal")), None);
        assert_eq!(
            directory_of(Path::new("index.zdb.zdbjournal")).unwrap(),
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
            let hex = id_hex(id);
            assert_eq!(hex.len(), 32);
            assert_eq!(id_from_hex(&hex), Some(id));
        }
        assert_eq!(id_from_hex("not hexadecimal"), None);
    }

    /// Fewer digits, more digits, a sign or anything that is not a digit
    /// is refused, so an id read back is spelled as it was written. Either
    /// case of digit is read, as the refusal's words promise.
    #[test]
    fn an_id_is_32_hexadecimal_digits_and_nothing_else() {
        let written = id_hex(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
        assert_eq!(id_from_hex("abc"), None);
        assert_eq!(id_from_hex(""), None);
        assert_eq!(id_from_hex(&written[1..]), None);
        assert_eq!(id_from_hex(&format!("{written}0")), None);
        assert_eq!(id_from_hex(&format!("+{}", &written[1..])), None);
        assert_eq!(id_from_hex(&format!("{} ", &written[1..])), None);
        assert_eq!(id_from_hex(&format!("g{}", &written[1..])), None);
        assert_eq!(
            id_from_hex(&written.to_uppercase()),
            id_from_hex(&written),
            "upper case digits are hexadecimal digits"
        );
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

    /// The holder's record is four lines naming this process, a host, the
    /// mode and the moment, and nothing a decision reads.
    #[test]
    fn the_holder_record_names_the_process_the_host_the_mode_and_the_moment() {
        let record = holder_record(LockMode::Exclusive);
        let lines: Vec<&str> = record.lines().collect();
        assert_eq!(lines.len(), 4, "{record}");
        assert_eq!(lines[0], format!("process {}", std::process::id()));
        assert!(
            lines[1].starts_with("host ") && lines[1].len() > 5,
            "{record}"
        );
        assert_eq!(lines[2], "mode exclusive");
        let since = lines[3].strip_prefix("since ").unwrap();
        assert!(
            chrono::DateTime::parse_from_rfc3339(since).is_ok(),
            "{since}"
        );
    }

    #[test]
    fn two_spellings_of_one_directory_are_one_directory() {
        assert!(same_directory(
            Path::new("index.zdb"),
            Path::new("./index.zdb")
        ));
        assert!(!same_directory(Path::new("a.zdb"), Path::new("b.zdb")));
    }
}
