//! The storage traits over the local filesystem.
//!
//! [`FsStorage`] keeps a collection at a directory path, its target, and
//! uses five siblings of it on the same volume. `<name>.zdbtmp` is the
//! directory a save writes before it is moved into place, `<name>.zdbold`
//! holds the directory being replaced while the new one moves in,
//! `<name>.zdbjournal` is the journal's directory of segments,
//! `<name>.zdbwal` is the journal of format version 1 an older release
//! wrote, and `<name>.zdblock` is the lock file. [`FsDir`] is a directory
//! of artefacts under one path.

use super::{
    refuses_writes, ArtefactWriter, Dir, JournalDir, JournalFile, LockFailure, LockGuard, LockMode,
    Locked, Staged, Storage,
};
use crate::error::Error;
use std::cell::RefCell;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{MutexGuard, PoisonError};
use tracing::info;

/// The target the records this module emits carry, being the module path
/// the staging code had in the binding. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::persistence";

/// Suffix of the directory a save builds before it is moved into place.
const STAGING_SUFFIX: &str = ".zdbtmp";

/// Suffix the directory being replaced is moved aside under.
const REPLACED_SUFFIX: &str = ".zdbold";

/// The suffix a collection's journal, a directory of segments, takes beside
/// its directory.
pub const JOURNAL_SUFFIX: &str = ".zdbjournal";

/// The suffix the journal of format version 1, one file, took beside its
/// directory. Releases from 0.11.0 to 0.13.0 wrote it, and this build reads
/// it to upgrade the directory.
pub const LEGACY_JOURNAL_SUFFIX: &str = ".zdbwal";

/// The suffix of the lock file beside a directory.
pub const LOCK_SUFFIX: &str = ".zdblock";

/// The suffix of a segment inside the journal's directory.
pub const SEGMENT_SUFFIX: &str = ".zdbseg";

/// The digits of a segment's name, so that the names sort in sequence order.
const SEGMENT_DIGITS: usize = 20;

/// The file name of the segment whose first record is `number`, being the
/// number in twenty decimal digits and the suffix.
pub(crate) fn segment_file_name(number: u64) -> String {
    format!("{number:0width$}{SEGMENT_SUFFIX}", width = SEGMENT_DIGITS)
}

/// The number a segment's file name carries, where `name` is a segment's
/// name and nothing else: twenty decimal digits and the suffix.
pub(crate) fn segment_number(name: &str) -> Option<u64> {
    let digits = name.strip_suffix(SEGMENT_SUFFIX)?;
    if digits.len() != SEGMENT_DIGITS || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

// ============================================================================
// A DIRECTORY OF ARTEFACTS
// ============================================================================

/// A directory of artefacts under one path. An artefact's name is joined to
/// the path, so a name carrying a prefix lands in the directory the prefix
/// names.
#[derive(Clone, Debug)]
pub struct FsDir {
    root: PathBuf,
}

impl FsDir {
    /// The directory at `root`. Nothing is created until an artefact is
    /// written.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        FsDir { root: root.into() }
    }

    fn at(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Create the directories `name` carries, where it carries any. A name
    /// at the root carries none and creates nothing.
    fn create_parents(&self, name: &str) -> io::Result<()> {
        match Path::new(name).parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                fs::create_dir_all(self.root.join(parent))
            }
            _ => Ok(()),
        }
    }
}

impl Dir for FsDir {
    /// The file is fsynced before this returns, so a rename that moves the
    /// directory into place cannot be recorded while the bytes it names are
    /// still in the page cache. Every byte is already in memory, so the fsync
    /// is the whole cost of that durability.
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        let create_failed = |e: io::Error| Error::ArtefactCreateFailed {
            name: name.to_string(),
            error: e.to_string(),
        };
        self.create_parents(name).map_err(create_failed)?;
        let mut file = File::create(self.at(name)).map_err(create_failed)?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|e| Error::ArtefactWriteFailed {
                name: name.to_string(),
                error: e.to_string(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        fs::read(self.at(name)).map_err(|e| Error::ArtefactReadFailed {
            name: name.to_string(),
            error: e.to_string(),
        })
    }

    fn exists(&self, name: &str) -> bool {
        self.at(name).exists()
    }

    fn length(&self, name: &str) -> io::Result<u64> {
        fs::metadata(self.at(name)).map(|meta| meta.len())
    }

    fn remove(&self, name: &str) -> io::Result<()> {
        fs::remove_file(self.at(name))
    }

    fn create(&self, name: &str) -> io::Result<Box<dyn ArtefactWriter>> {
        self.create_parents(name)?;
        let file = File::create(self.at(name))?;
        Ok(Box::new(FsArtefact { file }))
    }

    fn open(&self, name: &str) -> io::Result<Box<dyn Read>> {
        Ok(Box::new(File::open(self.at(name))?))
    }

    fn locate(&self, name: &str) -> String {
        if name.is_empty() {
            self.root.display().to_string()
        } else {
            self.at(name).display().to_string()
        }
    }
}

/// An artefact streamed into a file, which seeks back to its start to write
/// its head and is fsynced when it is finished.
struct FsArtefact {
    file: File,
}

impl Write for FsArtefact {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl ArtefactWriter for FsArtefact {
    fn write_head(&mut self, head: &[u8]) -> io::Result<()> {
        self.file
            .seek(SeekFrom::Start(0))
            .and_then(|_| self.file.write_all(head))
            .and_then(|()| self.file.flush())
    }

    fn finish(self: Box<Self>) -> io::Result<()> {
        self.file.sync_all()
    }
}

// ============================================================================
// WHERE A COLLECTION IS KEPT
// ============================================================================

/// A collection kept at a directory path on the filesystem.
///
/// # How a save lands
///
/// Every artefact goes into `<name>.zdbtmp` beside the target and the whole
/// directory is renamed into place at the end, so a reader sees the previous
/// index or this one and never a mixture, and a stale artefact of an earlier
/// save cannot survive. Before staging, every artefact went straight into
/// the target one write at a time, and a raw index saved over a quantized
/// one left `quantization.json`, `pq_centroids.bin` and `pq_codes.bin`
/// behind for ever.
///
/// # What "moves it into place" means
///
/// **It is one rename where the target does not exist, and two where it
/// does.** Neither Windows nor POSIX can rename a directory over an existing
/// non-empty directory. `rename(2)` requires the destination to be an empty
/// directory and `MoveFileExW` refuses `MOVEFILE_REPLACE_EXISTING` for
/// directories outright, so `fs::rename` fails on both platforms and there
/// is no call in the standard library that swaps two directories in one
/// step. Linux has `renameat2(RENAME_EXCHANGE)`, which std does not expose
/// and which Windows has no counterpart for.
///
/// So a commit over an existing directory does this:
///
/// 1. rename the target aside to `<name>.zdbold`
/// 2. rename the staging directory to the target
/// 3. remove `<name>.zdbold`
///
/// Steps 1 and 2 are each atomic on both platforms. Between them the target
/// does not exist, which is a window of two filesystem calls with no I/O
/// between them. **A reader in that window sees no directory rather than a
/// partial one**, which is the property that matters, and a process killed
/// in it leaves the whole previous index at `<name>.zdbold`.
/// [`Storage::recover`] puts that back, and both a load and the next save run
/// it first. A save to a path that holds nothing yet is step 2 alone, which
/// is atomic outright.
///
/// If step 2 fails the target is renamed back from `<name>.zdbold`, so a
/// failed save leaves the previous directory where it was.
///
/// A rename is only cheap, and only atomic, within one volume, which is why
/// the staging directory is a sibling of the target rather than a directory
/// under the system's temporary one.
///
/// # What a killed process leaves
///
/// A leftover `<name>.zdbtmp` from a save that died before the move, and a
/// leftover `<name>.zdbold` from one that died inside the window.
/// [`Storage::stage`] deals with both before it stages, and neither is
/// inside the index directory, so a load reads neither.
///
/// # The journal
///
/// `<name>.zdbjournal`, a directory of segments and a sibling for the same
/// reason. Windows refuses to rename a directory that holds an open file,
/// and `remove_dir_all` on a directory holding an open file succeeds, so a
/// journal inside the directory would either stop the first rename of a
/// commit or be removed with `<name>.zdbold` after the second.
///
/// A segment open for writing is opened on Windows without delete sharing,
/// so no other handle can delete or rename it while this one holds it.
///
/// # The lock
///
/// `<name>.zdblock`, an ordinary file locked whole through `File::try_lock`
/// and `File::try_lock_shared`, which are `flock` on Linux and macOS and
/// `LockFileEx` on Windows. The lock lives in a file of its own because on
/// Windows a lock blocks reads of the locked bytes through every other
/// handle, so a lock on a segment or an artefact would stop a reader. The
/// file is created on first use and never removed. A child forked through
/// Python's `os.fork` closes its copy of every lock file just after the
/// fork, through [`close_lock_files_in_forked_child`], so a lock stays with
/// the process that took it. That process unlocks the file before it closes
/// it, so the lock goes when it lets the lock go, even while a child it
/// started still holds a copy of the file.
#[derive(Clone, Debug)]
pub struct FsStorage {
    target: PathBuf,
    dir: FsDir,
    staging: PathBuf,
    replaced: PathBuf,
    journal: PathBuf,
    legacy_journal: PathBuf,
    lock: PathBuf,
}

impl FsStorage {
    /// The collection kept at `target`. Refused where the path names no
    /// directory, since each sibling is the target's name with a suffix.
    pub fn at(target: &Path) -> Result<Self, Error> {
        Ok(FsStorage {
            staging: sibling(target, STAGING_SUFFIX)?,
            replaced: sibling(target, REPLACED_SUFFIX)?,
            journal: sibling(target, JOURNAL_SUFFIX)?,
            legacy_journal: sibling(target, LEGACY_JOURNAL_SUFFIX)?,
            lock: sibling(target, LOCK_SUFFIX)?,
            dir: FsDir::new(target),
            target: target.to_path_buf(),
        })
    }
}

/// A sibling of `target` carrying `suffix`, so it lives on the target's
/// volume.
fn sibling(target: &Path, suffix: &str) -> Result<PathBuf, Error> {
    let name = target.file_name().ok_or_else(|| Error::TargetHasNoName {
        target: target.to_path_buf(),
    })?;
    let mut name = name.to_os_string();
    name.push(suffix);
    Ok(target.parent().unwrap_or_else(|| Path::new("")).join(name))
}

impl Storage for FsStorage {
    /// `<name>.zdbold` present with no target is the one case that holds
    /// data: a save died between its two renames and that directory is the
    /// only copy of the index. It is renamed back rather than removed.
    ///
    /// `<name>.zdbold` present beside a target is the previous index after a
    /// save that finished, and this leaves it where it is. Only a save
    /// removes it, because only a save knows the target beside it is the one
    /// it wrote.
    fn recover(&self) -> Result<bool, Error> {
        if !self.replaced.exists() || self.target.exists() {
            return Ok(false);
        }
        fs::rename(&self.replaced, &self.target).map_err(|e| Error::RecoverRenameFailed {
            target: self.target.clone(),
            replaced: self.replaced.clone(),
            error: e.to_string(),
        })?;
        info!(target: LOG_TARGET, operation = "save_recover",
            restored = %self.target.display(),
            "An interrupted save had moved the index aside; it is back in place"
        );
        Ok(true)
    }

    fn exists(&self) -> bool {
        self.target.exists()
    }

    fn aside_exists(&self) -> bool {
        self.replaced.exists()
    }

    fn dir(&self) -> &dyn Dir {
        &self.dir
    }

    /// Put right whatever a killed save left behind, then create the staging
    /// directory empty.
    ///
    /// `<name>.zdbold` still present after the recovery is the previous index
    /// after a save that finished, so it is removed, and so is a staging
    /// directory an interrupted save left.
    fn stage(&self) -> Result<Box<dyn Staged>, Error> {
        if !self.recover()? && self.replaced.exists() {
            remove_tree(
                &self.replaced,
                "the previous index a finished save left aside",
            )?;
        }
        if self.staging.exists() {
            remove_tree(
                &self.staging,
                "a staging directory an interrupted save left behind",
            )?;
        }
        fs::create_dir_all(&self.staging).map_err(|e| Error::StagingCreateFailed {
            staging: self.staging.clone(),
            error: e.to_string(),
        })?;
        Ok(Box::new(FsStaged {
            target: self.target.clone(),
            dir: FsDir::new(&self.staging),
            staging: self.staging.clone(),
            replaced: self.replaced.clone(),
            committed: false,
        }))
    }

    /// The target is read through its link where it is one, so a link to an
    /// index directory is an index directory.
    fn foreign_target(&self) -> Option<String> {
        let metadata = fs::metadata(&self.target).ok()?;
        if !metadata.is_dir() {
            return Some("a file".to_string());
        }
        if self.target.join("manifest.json").is_file() {
            return None;
        }
        match fs::read_dir(&self.target).map(Iterator::count) {
            Ok(0) => None,
            Ok(1) => Some("a folder holding 1 entry and no manifest.json".to_string()),
            Ok(n) => Some(format!("a folder holding {n} entries and no manifest.json")),
            Err(error) => Some(format!(
                "a folder this process cannot list ({error}) and no manifest.json"
            )),
        }
    }

    /// A staging directory an interrupted save left is removed here, as
    /// [`Storage::stage`] would remove it, so the probe asks the folder the
    /// question a checkpoint's first write asks.
    fn probe_writable(&self) -> io::Result<()> {
        if self.staging.exists() {
            fs::remove_dir_all(&self.staging)?;
        }
        fs::create_dir(&self.staging)?;
        fs::remove_dir(&self.staging)
    }

    fn journal_path(&self) -> &Path {
        &self.journal
    }

    /// Both are made absolute first, since `index.zdb` and `./index.zdb` are
    /// the same directory and compare unequal as written. Neither is
    /// canonicalised, because the target of a first save does not exist yet
    /// and canonicalising a path that does not exist fails.
    fn names_journal(&self, journal: &Path) -> bool {
        let absolute =
            |path: &Path| std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        absolute(journal) == absolute(&self.journal)
    }

    fn journal_exists(&self) -> bool {
        self.journal.is_dir()
    }

    fn create_journal(&self) -> io::Result<()> {
        fs::create_dir_all(&self.journal)?;
        let dir = FsJournalDir {
            path: self.journal.clone(),
        };
        for number in dir.segments()? {
            dir.remove_segment(number)?;
        }
        dir.sync();
        if let Some(parent) = self.journal.parent() {
            sync_directory(parent);
        }
        Ok(())
    }

    fn journal_dir(&self) -> Box<dyn JournalDir> {
        Box::new(FsJournalDir {
            path: self.journal.clone(),
        })
    }

    fn legacy_journal_path(&self) -> &Path {
        &self.legacy_journal
    }

    fn legacy_journal_exists(&self) -> bool {
        self.legacy_journal.is_file()
    }

    fn read_legacy_journal(&self) -> io::Result<Vec<u8>> {
        fs::read(&self.legacy_journal)
    }

    fn remove_legacy_journal(&self) -> io::Result<()> {
        fs::remove_file(&self.legacy_journal)?;
        if let Some(parent) = self.legacy_journal.parent() {
            sync_directory(parent);
        }
        Ok(())
    }

    fn cut_legacy_journal(&self, len: u64) -> io::Result<()> {
        let file = OpenOptions::new().write(true).open(&self.legacy_journal)?;
        file.set_len(len)?;
        file.sync_all()
    }

    fn lock_path(&self) -> &Path {
        &self.lock
    }

    /// The holder's line is written after an exclusive acquisition alone. On
    /// Windows a shared lock refuses writes to the locked bytes through
    /// every handle, its own included, so a shared holder writes nothing.
    ///
    /// The process's lock files are held from the open until the file is
    /// kept among them, so a fork never copies a lock file that its child
    /// would not know to close.
    fn lock(&self, mode: LockMode, holder: &str) -> Result<Locked, LockFailure> {
        with_lock_files(|files| self.lock_kept_in(files, mode, holder))
    }
}

impl FsStorage {
    /// [`Storage::lock`], with the lock file it opens kept in `files`.
    fn lock_kept_in(
        &self,
        files: &mut LockFileList,
        mode: LockMode,
        holder: &str,
    ) -> Result<Locked, LockFailure> {
        let open = || {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&self.lock)
        };
        let writable = match open() {
            Err(error)
                if error.kind() == io::ErrorKind::NotFound && mode == LockMode::Exclusive =>
            {
                match self
                    .lock
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    Some(parent) => fs::create_dir_all(parent).and_then(|()| open()),
                    None => Err(error),
                }
            }
            other => other,
        };
        match writable {
            Ok(file) => {
                acquire(&file, mode).map_err(|error| self.lock_failure(error))?;
                if mode == LockMode::Exclusive {
                    let _ = file
                        .set_len(0)
                        .and_then(|()| (&file).write_all(holder.as_bytes()));
                }
                Ok(Locked::Writable(Box::new(keep(files, file, mode))))
            }
            Err(reason) if refuses_writes(&reason) => match File::open(&self.lock) {
                Ok(file) => {
                    acquire(&file, LockMode::Shared).map_err(|error| self.lock_failure(error))?;
                    Ok(Locked::ReadOnly {
                        guard: Some(Box::new(keep(files, file, LockMode::Shared))),
                        reason,
                    })
                }
                Err(missing) if missing.kind() == io::ErrorKind::NotFound => Ok(Locked::ReadOnly {
                    guard: None,
                    reason,
                }),
                Err(other) => Err(LockFailure::Failed(other)),
            },
            Err(other) => Err(LockFailure::Failed(other)),
        }
    }

    /// What a lock that was not taken means. A holder that excludes this one
    /// is named by what its lock file records, its lines joined, where the
    /// system lets another handle read it, which Windows does not while the
    /// lock is held exclusive. The record is the last exclusive holder's,
    /// since a shared holder writes none.
    fn lock_failure(&self, error: std::fs::TryLockError) -> LockFailure {
        match error {
            std::fs::TryLockError::WouldBlock => {
                let recorded = fs::read_to_string(&self.lock).ok().map(|text| {
                    text.lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty())
                        .collect::<Vec<_>>()
                        .join(", ")
                });
                let holder = match recorded.filter(|text| !text.is_empty()) {
                    Some(text) => format!("another holder, its lock file recording {text}"),
                    None => "another holder, whose record this system does not let a second \
                             handle read while the lock is held"
                        .to_string(),
                };
                LockFailure::Held { holder }
            }
            std::fs::TryLockError::Error(error) => LockFailure::Unavailable(error),
        }
    }
}

/// Take `file`'s lock in `mode` without waiting.
fn acquire(file: &File, mode: LockMode) -> Result<(), std::fs::TryLockError> {
    match mode {
        LockMode::Shared => file.try_lock_shared(),
        LockMode::Exclusive => file.try_lock(),
    }
}

/// The lock files a process holds open, each beside the number its guard
/// carries.
type LockFileList = Vec<(u64, File)>;

/// [`LockFileList`] behind the lock that every opening, closing and fork of
/// a lock file takes.
#[allow(clippy::disallowed_types)]
type LockFiles = std::sync::Mutex<LockFileList>;

/// This process's lock files.
///
/// A file is kept here and not in its guard, so that a child forked from the
/// process can close its copy of every one, wherever the guard that owns it
/// lives. A lock on an open file goes only when the file is unlocked or the
/// last copy of it is closed, so the child's close leaves the parent's lock
/// held, and the lock then goes with the parent alone.
static LOCK_FILES: LockFiles = LockFiles::new(Vec::new());

/// The number the next lock file is kept under.
static NEXT_LOCK_FILE: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// This process's lock files, while the thread that forks holds them
    /// across the fork.
    static HELD_FOR_FORK: RefCell<Option<MutexGuard<'static, LockFileList>>> =
        const { RefCell::new(None) };
}

/// Hold this process's lock files from just before a fork, so that no other
/// thread opens or closes one while the process is copied. After the fork the
/// parent lets them go through [`release_lock_files_after_fork`] and the
/// child closes its copies through [`close_lock_files_in_forked_child`]. The
/// binding calls all three from `os.register_at_fork`.
pub fn hold_lock_files_for_fork() {
    HELD_FOR_FORK.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(LOCK_FILES.lock().unwrap_or_else(PoisonError::into_inner));
        }
    });
}

/// Let this process's lock files go after a fork, in the parent, as they were
/// before it.
pub fn release_lock_files_after_fork() {
    let held = HELD_FOR_FORK.with(|slot| slot.borrow_mut().take());
    drop(held);
}

/// Close, in a child just forked, the child's copy of every lock file the
/// parent held open. The parent keeps every lock, since a close releases a
/// lock only with the last copy of its file, and the child holds none, so a
/// lock goes when the parent lets it go or ends, whether or not the child
/// lives on. A guard the child drops later closes nothing.
///
/// The files are the ones held since [`hold_lock_files_for_fork`]. Where
/// nothing holds them they are taken here, unless a thread the child does not
/// have held them at the fork, and then the child keeps its copies.
pub fn close_lock_files_in_forked_child() {
    let copies = match HELD_FOR_FORK.with(|slot| slot.borrow_mut().take()) {
        Some(mut held) => std::mem::take(&mut *held),
        None => try_with_lock_files(std::mem::take).unwrap_or_default(),
    };
    drop(copies);
}

/// Run `f` through the hold this thread has taken for a fork, where it has
/// one, so that a lock taken or dropped there before the fork does not wait
/// on itself, and hand `f` back where it has none.
fn through_fork_hold<R, F>(f: F) -> Result<R, F>
where
    F: FnOnce(&mut LockFileList) -> R,
{
    HELD_FOR_FORK.with(|slot| match slot.borrow_mut().as_mut() {
        Some(held) => Ok(f(held)),
        None => Err(f),
    })
}

/// Run `f` over this process's lock files, waiting for them where another
/// thread has them.
fn with_lock_files<R>(f: impl FnOnce(&mut LockFileList) -> R) -> R {
    through_fork_hold(f)
        .unwrap_or_else(|f| f(&mut LOCK_FILES.lock().unwrap_or_else(PoisonError::into_inner)))
}

/// Run `f` over this process's lock files unless another thread has them,
/// which a child forked from the process cannot wait out, since that thread
/// is not in the child.
fn try_with_lock_files<R>(f: impl FnOnce(&mut LockFileList) -> R) -> Option<R> {
    match through_fork_hold(f) {
        Ok(result) => Some(result),
        Err(f) => match LOCK_FILES.try_lock() {
            Ok(mut held) => Some(f(&mut held)),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(f(&mut poisoned.into_inner())),
            Err(std::sync::TryLockError::WouldBlock) => None,
        },
    }
}

/// Keep `file` among this process's lock files, and give the guard whose drop
/// closes it.
fn keep(files: &mut LockFileList, file: File, mode: LockMode) -> FsLock {
    let id = NEXT_LOCK_FILE.fetch_add(1, Ordering::Relaxed);
    files.push((id, file));
    FsLock {
        id,
        mode,
        process: std::process::id(),
    }
}

/// A lock held through an open lock file, which this process's lock files
/// keep under `id`. Dropping it in the process that took it unlocks the file
/// and closes it; see [`LockGuard`].
#[derive(Debug)]
struct FsLock {
    id: u64,
    mode: LockMode,
    /// The process that took the lock.
    process: u32,
}

impl LockGuard for FsLock {
    fn mode(&self) -> LockMode {
        self.mode
    }
}

/// In the process that took the lock, the drop unlocks the file before it
/// closes it, so the lock goes even while a child process holds a copy of
/// the file, as a child does from its fork to its exec. A child never
/// unlocks, since the open file it shares with the parent carries the
/// parent's lock. A child forked from the process that took the lock has
/// closed its copy already where the fork went through `os.register_at_fork`,
/// and then the drop finds nothing to close. A child does not wait for the
/// lock files, which a thread it does not have may hold, and keeps its copy
/// where another thread had them.
impl Drop for FsLock {
    fn drop(&mut self) {
        let id = self.id;
        let take = move |files: &mut LockFileList| {
            files
                .iter()
                .position(|(kept, _)| *kept == id)
                .map(|at| files.swap_remove(at).1)
        };
        let file = if std::process::id() == self.process {
            with_lock_files(|files| {
                let file = take(files);
                if let Some(file) = &file {
                    // Where the unlock fails, the close still releases the
                    // lock once no child holds a copy.
                    let _ = file.unlock();
                }
                file
            })
        } else {
            try_with_lock_files(take).flatten()
        };
        drop(file);
    }
}

// ============================================================================
// THE JOURNAL'S DIRECTORY
// ============================================================================

/// The journal's directory of segments.
#[derive(Clone, Debug)]
struct FsJournalDir {
    path: PathBuf,
}

/// The options a segment is opened with for writing. On Windows the share
/// mode is read and write without delete, so no other handle can delete or
/// rename an open segment, which would orphan the journal under a live
/// collection. The two values are `FILE_SHARE_READ` and `FILE_SHARE_WRITE`.
fn segment_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(0x0000_0001 | 0x0000_0002);
    }
    options
}

impl JournalDir for FsJournalDir {
    fn path(&self) -> &Path {
        &self.path
    }

    /// An entry whose name is not a segment's is left where it is. One that
    /// is not the repair's `quarantine` folder is logged, once a listing.
    fn segments(&self) -> io::Result<Vec<u64>> {
        let mut numbers = Vec::new();
        let mut others = Vec::new();
        for entry in fs::read_dir(&self.path)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            match segment_number(&name) {
                Some(number) if entry.file_type()?.is_file() => numbers.push(number),
                _ if name == QUARANTINE => {}
                _ => others.push(name.into_owned()),
            }
        }
        if !others.is_empty() {
            info!(target: LOG_TARGET, operation = "journal_entries",
                journal = %self.path.display(),
                entries = ?others,
                "The journal's directory holds entries that are not segments; they were left where they are"
            );
        }
        numbers.sort_unstable();
        Ok(numbers)
    }

    fn read_segment(&self, number: u64) -> io::Result<Vec<u8>> {
        fs::read(self.segment_path(number))
    }

    fn create_segment(&self, number: u64) -> io::Result<Box<dyn JournalFile>> {
        let file = segment_options()
            .create_new(true)
            .open(self.segment_path(number))?;
        Ok(Box::new(FsJournalFile { file }))
    }

    fn open_segment(&self, number: u64) -> io::Result<Box<dyn JournalFile>> {
        let file = segment_options().open(self.segment_path(number))?;
        Ok(Box::new(FsJournalFile { file }))
    }

    fn remove_segment(&self, number: u64) -> io::Result<()> {
        fs::remove_file(self.segment_path(number))
    }

    fn sync(&self) {
        sync_directory(&self.path);
    }

    fn segment_path(&self, number: u64) -> PathBuf {
        self.path.join(segment_file_name(number))
    }

    fn create_quarantine(&self, folder: &str) -> io::Result<()> {
        fs::create_dir_all(self.path.join(QUARANTINE).join(folder))
    }

    fn quarantine(&self) -> Box<dyn Dir> {
        Box::new(FsDir::new(self.path.join(QUARANTINE)))
    }

    fn quarantine_folders(&self) -> io::Result<Vec<String>> {
        let root = self.path.join(QUARANTINE);
        if !root.is_dir() {
            return Ok(Vec::new());
        }
        let mut folders = Vec::new();
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                folders.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        folders.sort_unstable();
        Ok(folders)
    }

    fn sync_quarantine(&self, folder: &str) {
        let root = self.path.join(QUARANTINE);
        sync_directory(&root.join(folder));
        sync_directory(&root);
        sync_directory(&self.path);
    }
}

/// The folder under the journal's directory a repair copies what it sets
/// aside into, one folder a repair.
const QUARANTINE: &str = "quarantine";

/// The staging directory, and the move that puts it in place. See
/// [`FsStorage`] for what the move is on each platform.
struct FsStaged {
    target: PathBuf,
    dir: FsDir,
    staging: PathBuf,
    replaced: PathBuf,
    committed: bool,
}

impl Staged for FsStaged {
    fn dir(&self) -> &dyn Dir {
        &self.dir
    }

    fn commit(mut self: Box<Self>, between: crate::KillPoint) -> Result<(), Error> {
        sync_directory(&self.staging);

        if self.target.exists() {
            fs::rename(&self.target, &self.replaced).map_err(|e| Error::MoveAsideFailed {
                target: self.target.clone(),
                error: e.to_string(),
            })?;

            crate::kill_at(between);

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

/// A save that fails part way cleans up after itself inside the process
/// that started it.
impl Drop for FsStaged {
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
/// A file's bytes reaching the disk does not put its name in its directory.
/// On POSIX that needs the directory's own descriptor fsynced, which is what
/// this does, and without it a power loss can leave the renamed directory
/// holding entries that were never recorded.
///
/// **Windows has no equivalent through the standard library.** `File::open`
/// refuses a directory there, so this is a no-op, and the durability claim on
/// Windows rests on NTFS journalling the rename rather than on anything this
/// crate does. That difference is not observable from a gate that runs on
/// Windows.
///
/// Best effort on both. A filesystem that refuses the fsync is not a reason
/// to fail a save whose bytes are already written.
#[cfg(unix)]
fn sync_directory(path: &Path) {
    if let Ok(dir) = File::open(path) {
        let _ = dir.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) {}

// ============================================================================
// THE JOURNAL
// ============================================================================

/// One segment, shared by its writer and the thread that syncs it.
#[derive(Debug)]
struct FsJournalFile {
    file: File,
}

impl JournalFile for FsJournalFile {
    fn write_all(&self, bytes: &[u8]) -> io::Result<()> {
        (&self.file).write_all(bytes)
    }

    fn seek_end(&self) -> io::Result<()> {
        (&self.file).seek(SeekFrom::End(0)).map(|_| ())
    }

    fn set_len(&self, len: u64) -> io::Result<()> {
        self.file.set_len(len)
    }

    fn sync_data(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    fn sync_all(&self) -> io::Result<()> {
        self.file.sync_all()
    }

    fn length(&self) -> io::Result<u64> {
        self.file.metadata().map(|meta| meta.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names of every entry directly under `path`, sorted.
    fn entries(path: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// An artefact under a prefix lands in the directory the prefix names,
    /// reads back whole, and is measured, located and removed by the name it
    /// was written under.
    #[test]
    fn an_artefact_lands_under_its_prefix_and_reads_back() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = FsDir::new(tmp.path());
        dir.write("config.json", b"{}").unwrap();
        dir.write("spaces/s/postings.zdbsparse", &[7u8; 300])
            .unwrap();
        assert!(tmp.path().join("spaces").join("s").is_dir());
        assert_eq!(
            dir.read("spaces/s/postings.zdbsparse").unwrap(),
            vec![7u8; 300]
        );
        assert!(dir.exists("config.json"));
        assert!(!dir.exists("absent.bin"));
        assert_eq!(dir.length("spaces/s/postings.zdbsparse").unwrap(), 300);
        assert_eq!(dir.locate(""), tmp.path().display().to_string());
        assert_eq!(
            dir.locate("config.json"),
            tmp.path().join("config.json").display().to_string()
        );
        assert!(matches!(
            dir.read("absent.bin"),
            Err(Error::ArtefactReadFailed { .. })
        ));
        dir.remove("config.json").unwrap();
        assert!(!dir.exists("config.json"));
    }

    /// A streamed artefact has its head written over the bytes it reserved,
    /// once the rest is written, and reads back as a stream.
    #[test]
    fn a_streamed_artefact_takes_its_head_last() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = FsDir::new(tmp.path());
        let mut stream = dir.create("p/a.dump").unwrap();
        stream.write_all(&[0u8; 4]).unwrap();
        stream.write_all(b"body").unwrap();
        stream.write_head(b"HEAD").unwrap();
        stream.finish().unwrap();
        assert_eq!(
            fs::read(tmp.path().join("p").join("a.dump")).unwrap(),
            b"HEADbody"
        );
        let mut back = Vec::new();
        dir.open("p/a.dump")
            .unwrap()
            .read_to_end(&mut back)
            .unwrap();
        assert_eq!(back, b"HEADbody");
    }

    /// A commit puts the staged directory in place whole and leaves neither
    /// sibling behind, a second commit replaces the first, and a staged
    /// directory dropped without a commit is removed and the target kept.
    #[test]
    fn a_commit_replaces_the_directory_whole_and_a_drop_leaves_it() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        assert!(!storage.exists());

        let staged = storage.stage().unwrap();
        staged.dir().write("a.bin", b"first").unwrap();
        staged.commit(crate::KillPoint::SaveBetweenRenames).unwrap();
        assert!(storage.exists());
        assert_eq!(storage.dir().read("a.bin").unwrap(), b"first");

        let staged = storage.stage().unwrap();
        staged.dir().write("b.bin", b"second").unwrap();
        staged.commit(crate::KillPoint::SaveBetweenRenames).unwrap();
        assert_eq!(entries(&tmp.path().join("index.zdb")), vec!["b.bin"]);
        assert_eq!(entries(tmp.path()), vec!["index.zdb"]);

        let staged = storage.stage().unwrap();
        staged.dir().write("c.bin", b"abandoned").unwrap();
        drop(staged);
        assert_eq!(entries(tmp.path()), vec!["index.zdb"]);
        assert_eq!(entries(&tmp.path().join("index.zdb")), vec!["b.bin"]);
    }

    /// A directory a save left aside with nothing at the target is put back,
    /// and one left beside a target stays until the next save removes it.
    #[test]
    fn a_directory_left_aside_is_put_back() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("index.zdb");
        let storage = FsStorage::at(&target).unwrap();
        fs::create_dir(tmp.path().join("index.zdb.zdbold")).unwrap();
        fs::write(tmp.path().join("index.zdb.zdbold").join("a.bin"), b"kept").unwrap();
        assert!(storage.recover().unwrap());
        assert_eq!(storage.dir().read("a.bin").unwrap(), b"kept");
        assert!(!storage.recover().unwrap());

        fs::create_dir(tmp.path().join("index.zdb.zdbold")).unwrap();
        fs::create_dir(tmp.path().join("index.zdb.zdbtmp")).unwrap();
        assert!(!storage.recover().unwrap());
        assert_eq!(
            entries(tmp.path()),
            vec!["index.zdb", "index.zdb.zdbold", "index.zdb.zdbtmp"]
        );
        drop(storage.stage().unwrap());
        assert_eq!(entries(tmp.path()), vec!["index.zdb"]);
    }

    /// The journal, the version 1 journal and the lock file are the target's
    /// siblings, a path naming the journal another way names the same one,
    /// and a target with no name has no siblings.
    #[test]
    fn the_journal_and_the_lock_are_the_targets_siblings() {
        let storage = FsStorage::at(Path::new("a/b/index.zdb")).unwrap();
        assert_eq!(
            storage.journal_path(),
            Path::new("a/b/index.zdb.zdbjournal")
        );
        assert_eq!(
            storage.legacy_journal_path(),
            Path::new("a/b/index.zdb.zdbwal")
        );
        assert_eq!(storage.lock_path(), Path::new("a/b/index.zdb.zdblock"));
        assert!(storage.names_journal(Path::new("a/./b/index.zdb.zdbjournal")));
        assert!(!storage.names_journal(Path::new("a/b/other.zdb.zdbjournal")));
        assert!(!storage.names_journal(Path::new("a/b/index.zdb.zdbwal")));
        assert!(matches!(
            FsStorage::at(Path::new("")),
            Err(Error::TargetHasNoName { .. })
        ));
    }

    /// A segment is named by its first sequence in twenty digits, so the
    /// names sort in sequence order, and nothing else reads as a segment.
    #[test]
    fn a_segment_is_named_by_its_first_sequence_in_twenty_digits() {
        assert_eq!(segment_file_name(1), "00000000000000000001.zdbseg");
        assert_eq!(segment_file_name(u64::MAX), "18446744073709551615.zdbseg");
        for number in [0u64, 1, 9, 10, 4_294_967_296, u64::MAX] {
            assert_eq!(segment_number(&segment_file_name(number)), Some(number));
        }
        let mut names: Vec<String> = [100u64, 9, 1_000_000, 10]
            .iter()
            .map(|n| segment_file_name(*n))
            .collect();
        names.sort();
        let sorted: Vec<u64> = names.iter().map(|n| segment_number(n).unwrap()).collect();
        assert_eq!(sorted, vec![9, 10, 100, 1_000_000]);
        for name in [
            "1.zdbseg",
            "0000000000000000001.zdbseg",
            "000000000000000000001.zdbseg",
            "0000000000000000000a.zdbseg",
            "99999999999999999999.zdbseg",
            "00000000000000000001.zdbseg.tmp",
            "00000000000000000001",
            "quarantine",
        ] {
            assert_eq!(segment_number(name), None, "{name}");
        }
    }

    /// The journal's directory lists its segments in order and leaves every
    /// other entry where it is, a segment is created once, opened, read and
    /// removed by its number, and creating the journal again empties it of
    /// segments alone.
    #[test]
    fn the_journal_directory_holds_segments_by_number() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        assert!(!storage.journal_exists());
        storage.create_journal().unwrap();
        assert!(storage.journal_exists());
        let dir = storage.journal_dir();
        assert_eq!(dir.path(), storage.journal_path());
        assert!(dir.segments().unwrap().is_empty());
        for number in [12u64, 1, 7] {
            let file = dir.create_segment(number).unwrap();
            file.write_all(&number.to_le_bytes()).unwrap();
            file.sync_all().unwrap();
        }
        assert!(dir.create_segment(7).is_err(), "a segment is created once");
        fs::create_dir(storage.journal_path().join("quarantine")).unwrap();
        fs::write(storage.journal_path().join("notes.txt"), b"left").unwrap();
        assert_eq!(dir.segments().unwrap(), vec![1, 7, 12]);
        assert_eq!(dir.read_segment(7).unwrap(), 7u64.to_le_bytes());
        let open = dir.open_segment(7).unwrap();
        open.seek_end().unwrap();
        open.write_all(b"more").unwrap();
        assert_eq!(open.length().unwrap(), 12);
        open.set_len(8).unwrap();
        assert_eq!(dir.read_segment(7).unwrap(), 7u64.to_le_bytes());
        drop(open);
        dir.remove_segment(1).unwrap();
        assert_eq!(dir.segments().unwrap(), vec![7, 12]);
        assert_eq!(
            dir.segment_path(7),
            storage.journal_path().join("00000000000000000007.zdbseg")
        );
        storage.create_journal().unwrap();
        assert!(dir.segments().unwrap().is_empty());
        assert_eq!(
            entries(storage.journal_path()),
            vec!["notes.txt", "quarantine"]
        );
    }

    /// The version 1 journal is one file beside the directory, read whole
    /// and removed.
    #[test]
    fn the_version_1_journal_is_read_and_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        assert!(!storage.legacy_journal_exists());
        fs::write(storage.legacy_journal_path(), b"ZDBJOURN").unwrap();
        assert!(storage.legacy_journal_exists());
        assert_eq!(storage.read_legacy_journal().unwrap(), b"ZDBJOURN");
        storage.remove_legacy_journal().unwrap();
        assert!(!storage.legacy_journal_exists());
    }

    /// A cut keeps the first bytes of the journal of format version 1 and
    /// drops the rest.
    #[test]
    fn the_version_1_journal_is_cut_to_a_length() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        fs::write(storage.legacy_journal_path(), b"ZDBJOURN-and-a-tail").unwrap();
        storage.cut_legacy_journal(8).unwrap();
        assert_eq!(storage.read_legacy_journal().unwrap(), b"ZDBJOURN");
    }

    /// Nothing at the target, an empty folder and a folder holding a manifest
    /// are each a target a save may replace. A file and a folder of other
    /// entries are not, and each is named as what it is.
    #[test]
    fn a_target_that_is_not_an_index_directory_is_named() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("index.zdb");
        let storage = FsStorage::at(&target).unwrap();
        assert_eq!(storage.foreign_target(), None);
        fs::create_dir(&target).unwrap();
        assert_eq!(storage.foreign_target(), None);
        fs::write(target.join("notes.txt"), b"mine").unwrap();
        assert_eq!(
            storage.foreign_target().as_deref(),
            Some("a folder holding 1 entry and no manifest.json")
        );
        fs::create_dir(target.join("more")).unwrap();
        assert_eq!(
            storage.foreign_target().as_deref(),
            Some("a folder holding 2 entries and no manifest.json")
        );
        fs::write(target.join("manifest.json"), b"{}").unwrap();
        assert_eq!(storage.foreign_target(), None);
        let file = tmp.path().join("file.zdb");
        fs::write(&file, b"a file").unwrap();
        assert_eq!(
            FsStorage::at(&file).unwrap().foreign_target().as_deref(),
            Some("a file")
        );
    }

    /// The quarantine is folders under the journal's directory, each holding
    /// files written durably, and the listing of segments passes over it.
    #[test]
    fn the_quarantine_sits_beside_the_segments() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        storage.create_journal().unwrap();
        let journal = storage.journal_dir();
        assert!(journal.quarantine_folders().unwrap().is_empty());
        journal.create_segment(5).unwrap();
        journal.create_quarantine("b-9").unwrap();
        journal.create_quarantine("a-3").unwrap();
        journal
            .quarantine()
            .write("a-3/00000000000000000003.zdbseg", b"bytes")
            .unwrap();
        journal.sync_quarantine("a-3");
        assert_eq!(journal.quarantine_folders().unwrap(), ["a-3", "b-9"]);
        assert_eq!(
            journal
                .quarantine()
                .read("a-3/00000000000000000003.zdbseg")
                .unwrap(),
            b"bytes"
        );
        assert_eq!(journal.segments().unwrap(), [5]);
        storage.create_journal().unwrap();
        assert!(journal.segments().unwrap().is_empty());
        assert_eq!(
            journal.quarantine_folders().unwrap(),
            ["a-3", "b-9"],
            "emptying the journal of segments leaves the quarantine"
        );
    }

    /// The probe makes and removes the staging directory, and removes one an
    /// interrupted save left, so it leaves the folder as it found it.
    #[test]
    fn the_write_probe_leaves_the_folder_as_it_found_it() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        storage.probe_writable().unwrap();
        assert!(entries(tmp.path()).is_empty());
        fs::create_dir(tmp.path().join("index.zdb.zdbtmp")).unwrap();
        fs::write(tmp.path().join("index.zdb.zdbtmp").join("a.bin"), b"left").unwrap();
        storage.probe_writable().unwrap();
        assert!(entries(tmp.path()).is_empty());
    }

    /// The lock tests run one at a time, since a child's close takes every
    /// lock file of the process and a hold for a fork stops every other lock.
    #[allow(clippy::disallowed_types)]
    static ONE_LOCK_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn alone() -> MutexGuard<'static, ()> {
        ONE_LOCK_TEST.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn holder(storage: &FsStorage, mode: LockMode) -> Result<Box<dyn LockGuard>, LockFailure> {
        match storage.lock(mode, "pid 1, host test, exclusive, since now\n")? {
            Locked::Writable(guard) => Ok(guard),
            Locked::ReadOnly { .. } => panic!("the lock file is writable here"),
        }
    }

    /// What a refusal names where the system lets a second handle read the
    /// lock file, being the record [`holder`] writes.
    const RECORDED: &str =
        "another holder, its lock file recording pid 1, host test, exclusive, since now";

    /// The lock in one process, as the standard library gives it on this
    /// system. An exclusive holder refuses a second handle in either mode,
    /// shared holders sit together and refuse an exclusive one, a dropped
    /// guard releases, and the lock file is kept. A refusal names the record
    /// wherever the system lets the second handle read the lock file, which
    /// Windows does beside shared holders alone and Linux and macOS do
    /// beside any holder.
    #[test]
    fn the_lock_refuses_a_second_handle_and_releases_on_drop() {
        let _alone = alone();
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        let first = holder(&storage, LockMode::Exclusive).unwrap();
        assert_eq!(first.mode(), LockMode::Exclusive);
        let beside_exclusive = if cfg!(windows) {
            "another holder, whose record this system does not let a second handle read while \
             the lock is held"
        } else {
            RECORDED
        };
        for mode in [LockMode::Exclusive, LockMode::Shared] {
            match holder(&storage, mode) {
                Err(LockFailure::Held { holder }) => {
                    assert_eq!(holder, beside_exclusive, "{mode:?}")
                }
                other => panic!("{mode:?}: expected the lock to be held, got {other:?}"),
            }
        }
        drop(first);
        let shared = holder(&storage, LockMode::Shared).unwrap();
        let beside = holder(&storage, LockMode::Shared).unwrap();
        assert_eq!(beside.mode(), LockMode::Shared);
        match holder(&storage, LockMode::Exclusive) {
            Err(LockFailure::Held { holder }) => assert_eq!(holder, RECORDED),
            other => panic!("expected the shared holders to refuse, got {other:?}"),
        }
        drop(shared);
        drop(beside);
        let again = holder(&storage, LockMode::Exclusive).unwrap();
        drop(again);
        assert!(storage.lock_path().is_file(), "the lock file is kept");
    }

    /// A child forked from a holder closes its copy of every lock file, and a
    /// guard it drops afterwards closes nothing. Run in one process, the copy
    /// closed is the holder's own, so the lock is free once it is closed, and
    /// a holder that takes it then keeps it through the first guard's drop.
    #[test]
    fn a_forked_childs_close_leaves_its_guards_nothing_to_close() {
        let _alone = alone();
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        let first = holder(&storage, LockMode::Exclusive).unwrap();
        hold_lock_files_for_fork();
        close_lock_files_in_forked_child();
        let second = holder(&storage, LockMode::Exclusive).unwrap();
        drop(first);
        assert!(
            matches!(
                holder(&storage, LockMode::Shared),
                Err(LockFailure::Held { .. })
            ),
            "the first guard's drop closed the second holder's file"
        );
        drop(second);
        holder(&storage, LockMode::Shared).unwrap();
    }

    /// Held for a fork and let go in the parent, the lock files are as they
    /// were, and a lock taken and dropped on the thread that holds them goes
    /// through that hold rather than waiting on it.
    #[test]
    fn a_hold_for_a_fork_leaves_every_lock_as_it_was() {
        let _alone = alone();
        let tmp = tempfile::tempdir().unwrap();
        let storage = FsStorage::at(&tmp.path().join("index.zdb")).unwrap();
        let other = FsStorage::at(&tmp.path().join("other.zdb")).unwrap();
        let first = holder(&storage, LockMode::Exclusive).unwrap();
        hold_lock_files_for_fork();
        let during = holder(&other, LockMode::Exclusive).unwrap();
        drop(during);
        release_lock_files_after_fork();
        assert!(matches!(
            holder(&storage, LockMode::Exclusive),
            Err(LockFailure::Held { .. })
        ));
        drop(first);
        holder(&storage, LockMode::Exclusive).unwrap();
        holder(&other, LockMode::Exclusive).unwrap();
    }
}
