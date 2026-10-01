//! The storage traits over the local filesystem.
//!
//! [`FsStorage`] keeps a collection at a directory path, its target, and
//! uses three siblings of it on the same volume. `<name>.zdbtmp` is the
//! directory a save writes before it is moved into place, `<name>.zdbold`
//! holds the directory being replaced while the new one moves in, and
//! `<name>.zdbwal` is the journal. [`FsDir`] is a directory of artefacts
//! under one path.

use super::{ArtefactWriter, Dir, JournalFile, Staged, Storage};
use crate::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tracing::info;

/// The target the one record this module emits carries, being the module
/// path the staging code had in the binding. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::persistence";

/// Suffix of the directory a save builds before it is moved into place.
const STAGING_SUFFIX: &str = ".zdbtmp";

/// Suffix the directory being replaced is moved aside under.
const REPLACED_SUFFIX: &str = ".zdbold";

/// The suffix a collection's journal takes beside its directory.
pub const JOURNAL_SUFFIX: &str = ".zdbwal";

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

    /// Every file under the path, the directories under it included, and
    /// nothing where the path is not a directory.
    fn total_bytes(&self) -> io::Result<u64> {
        fn bytes_under(path: &Path) -> io::Result<u64> {
            let mut total = 0u64;
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    total += metadata.len();
                } else if metadata.is_dir() {
                    total += bytes_under(&entry.path())?;
                }
            }
            Ok(total)
        }

        if self.root.is_dir() {
            bytes_under(&self.root)
        } else {
            Ok(0)
        }
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
/// `<name>.zdbwal`, a sibling for the same reason. Windows refuses to rename
/// a directory that holds an open file, and `remove_dir_all` on a directory
/// holding an open file succeeds, so a journal inside the directory would
/// either stop the first rename of a commit or be removed with
/// `<name>.zdbold` after the second.
#[derive(Clone, Debug)]
pub struct FsStorage {
    target: PathBuf,
    dir: FsDir,
    staging: PathBuf,
    replaced: PathBuf,
    journal: PathBuf,
}

impl FsStorage {
    /// The collection kept at `target`. Refused where the path names no
    /// directory, since each sibling is the target's name with a suffix.
    pub fn at(target: &Path) -> Result<Self, Error> {
        Ok(FsStorage {
            staging: sibling(target, STAGING_SUFFIX)?,
            replaced: sibling(target, REPLACED_SUFFIX)?,
            journal: sibling(target, JOURNAL_SUFFIX)?,
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
        self.journal.exists()
    }

    fn read_journal(&self) -> io::Result<Vec<u8>> {
        fs::read(&self.journal)
    }

    fn create_journal(&self) -> io::Result<Box<dyn JournalFile>> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&self.journal)?;
        Ok(Box::new(FsJournalFile { file }))
    }

    fn open_journal(&self) -> io::Result<Box<dyn JournalFile>> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.journal)?;
        Ok(Box::new(FsJournalFile { file }))
    }
}

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

    fn commit(mut self: Box<Self>) -> Result<(), Error> {
        sync_directory(&self.staging);

        if self.target.exists() {
            fs::rename(&self.target, &self.replaced).map_err(|e| Error::MoveAsideFailed {
                target: self.target.clone(),
                error: e.to_string(),
            })?;

            crate::kill_at(crate::KillPoint::SaveBetweenRenames);

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

/// The journal's file, shared by its writer and the thread that syncs it.
#[derive(Debug)]
struct FsJournalFile {
    file: File,
}

impl JournalFile for FsJournalFile {
    fn write_all(&self, bytes: &[u8]) -> io::Result<()> {
        (&self.file).write_all(bytes)
    }

    fn seek_start(&self) -> io::Result<()> {
        (&self.file).seek(SeekFrom::Start(0)).map(|_| ())
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
    /// reads back whole, and is counted, measured, located and removed by
    /// the name it was written under.
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
        assert_eq!(dir.total_bytes().unwrap(), 302);
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
        assert_eq!(dir.total_bytes().unwrap(), 300);
        assert_eq!(
            FsDir::new(tmp.path().join("nowhere"))
                .total_bytes()
                .unwrap(),
            0
        );
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
        staged.commit().unwrap();
        assert!(storage.exists());
        assert_eq!(storage.dir().read("a.bin").unwrap(), b"first");

        let staged = storage.stage().unwrap();
        staged.dir().write("b.bin", b"second").unwrap();
        staged.commit().unwrap();
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

    /// The journal is the target's sibling, a path naming it another way
    /// names the same journal, and a target with no name has no siblings.
    #[test]
    fn the_journal_is_the_targets_sibling() {
        let storage = FsStorage::at(Path::new("a/b/index.zdb")).unwrap();
        assert_eq!(storage.journal_path(), Path::new("a/b/index.zdb.zdbwal"));
        assert!(storage.names_journal(Path::new("a/./b/index.zdb.zdbwal")));
        assert!(!storage.names_journal(Path::new("a/b/other.zdb.zdbwal")));
        assert!(matches!(
            FsStorage::at(Path::new("")),
            Err(Error::TargetHasNoName { .. })
        ));
    }
}
