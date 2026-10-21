//! The upgrade of a journaled directory an older release wrote.
//!
//! Releases from 0.11.0 to 0.13.0 kept a journal in one file at journal
//! format version 1. A load replays it and writes the directory again at
//! 5.0.0 with a journal of segments, then deletes the file. These tests
//! hold the upgrade to the real directories older releases wrote, kept
//! under `tests/fixtures/formats`, and to the same directories edited by
//! hand where a rule needs a case no release writes. Every test loads a
//! copy, since a load writes to the directory it opens.

#![allow(clippy::disallowed_types)]

use std::path::{Path, PathBuf};

use serde_json::json;
use zeusdb_vector_core::{Error, FsStorage, Storage};

use super::journal_support::{
    copy_fixture, expected_of, files_under, fixture_expected, holds_fixture, manifest, Segments,
    TempDir, FIXTURES,
};
use super::{Collection, ParsedRecord};
use crate::journal::{journal_path, Durability};

fn recover(path: &Path) -> Result<(Collection, crate::Recovery), Error> {
    Collection::recover(path.to_str().unwrap(), None, Durability::PerCall)
}

fn legacy_journal(path: &Path) -> PathBuf {
    FsStorage::at(path)
        .unwrap()
        .legacy_journal_path()
        .to_path_buf()
}

/// Every file of a fixture's copy but the lock file a load creates beside
/// the directory.
fn fixture_files(folder: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    files_under(&[folder])
        .into_iter()
        .filter(|(path, _)| path.extension().is_none_or(|e| e != "zdblock"))
        .collect()
}

/// Each fixture opens with every record it recorded, is written again at
/// 5.0.0 with a journal of segments carrying the sequence on, and its file
/// at journal format version 1 goes. A second open replays nothing and
/// holds the same, and a record added after the upgrade survives a third.
#[test]
fn every_fixture_upgrades_to_5_0_0_with_every_record_it_recorded() {
    for name in FIXTURES {
        let temp = TempDir::new("upgrade-tests");
        let path = copy_fixture(name, &temp.0);
        let expected = fixture_expected(name);
        let format = expected["format_version"].as_str().unwrap().to_string();
        let torn = expected["torn_sequence"].as_u64().unwrap();
        assert!(legacy_journal(&path).is_file(), "{name}");

        let (upgraded, report) = recover(&path).unwrap();
        holds_fixture(&upgraded, name).unwrap();
        assert_eq!(
            report.upgraded_from.as_deref(),
            Some(format.as_str()),
            "{name}"
        );
        assert!(report.checkpointed, "{name}");
        assert_eq!(report.format_version, format, "{name}");
        assert_eq!(
            report.replayed as u64,
            expected["records_in_journal"].as_u64().unwrap(),
            "{name}: every whole record above the checkpoint"
        );
        assert!(matches!(
            report.damage,
            Some(zeusdb_vector_core::JournalDamage::TornTail { .. })
        ));
        assert_eq!(report.read_only, None);

        let m = manifest(&path);
        assert_eq!(m["format_version"], json!("5.0.0"), "{name}");
        assert_eq!(m["journal"]["file"], json!("journaled.zdb.zdbjournal"));
        assert_eq!(m["journal"]["sequence"], json!(torn - 1), "{name}");
        assert_eq!(m["identity"]["sequence"], json!(torn - 1), "{name}");
        assert!(!legacy_journal(&path).exists(), "{name}: the file went");
        let segments = Segments::of(&path);
        assert_eq!(
            segments.numbers,
            vec![torn],
            "{name}: the sequence carries on"
        );
        assert_eq!(
            zeusdb_vector_core::read_segment_header(&segments.bytes[0], "s")
                .unwrap()
                .base,
            expected_of(&path).base
        );
        let written = upgraded.len();
        drop(upgraded);

        let (reopened, report) = recover(&path).unwrap();
        holds_fixture(&reopened, name).unwrap();
        assert_eq!(report.upgraded_from, None);
        assert_eq!(report.replayed, 0);
        assert_eq!(report.format_version, "5.0.0");
        let dim = expected["dim"].as_u64().unwrap() as usize;
        let added = reopened.add_records(
            vec![ParsedRecord {
                id: "after".to_string(),
                vector: vec![0.25; dim],
                sparse: None,
                metadata: Default::default(),
            }],
            vec![],
            false,
        );
        assert_eq!(added.total_errors, 0, "{name}: {:?}", added.errors);
        drop(reopened);

        let (third, report) = recover(&path).unwrap();
        assert_eq!(report.replayed, 1, "{name}");
        assert_eq!(third.len(), written + 1);
        assert!(third.contains("after"));
    }
}

/// The checkpoint alone opens a fixture as it stood at its last save, and
/// upgrades nothing: no file of the directory or its journal changes.
#[test]
fn a_fixtures_checkpoint_alone_upgrades_nothing() {
    for name in FIXTURES {
        let temp = TempDir::new("upgrade-tests");
        let path = copy_fixture(name, &temp.0);
        let folder = path.parent().unwrap().to_path_buf();
        let before = fixture_files(&folder);
        let checkpoint = Collection::load_checkpoint_only(path.to_str().unwrap(), None).unwrap();
        let expected = fixture_expected(name);
        assert_eq!(
            checkpoint.len() as u64,
            expected["checkpoint_records"].as_u64().unwrap(),
            "{name}"
        );
        drop(checkpoint);
        assert_eq!(fixture_files(&folder), before, "{name}");
    }
}

/// A journal of format version 1 refused for what it holds leaves the
/// directory as the release wrote it, since the upgrade begins only once
/// every record has replayed: one with a corrupt middle, one naming another
/// collection, and one that is missing.
#[test]
fn a_refused_version_1_journal_leaves_the_directory_as_it_was() {
    let name = FIXTURES[0];
    let refused = |edit: &dyn Fn(&Path)| {
        let temp = TempDir::new("upgrade-tests");
        let path = copy_fixture(name, &temp.0);
        edit(&path);
        let folder = path.parent().unwrap().to_path_buf();
        let before = fixture_files(&folder);
        let refusal = recover(&path).map(|_| ()).unwrap_err();
        assert_eq!(fixture_files(&folder), before, "nothing changed: {refusal}");
        assert_eq!(manifest(&path)["format_version"], json!("3.0.0"));
        refusal
    };

    let corrupt = refused(&|path| {
        let file = legacy_journal(path);
        let mut bytes = std::fs::read(&file).unwrap();
        bytes[64 + 40] ^= 0x5a;
        std::fs::write(&file, bytes).unwrap();
    });
    assert!(
        matches!(corrupt, Error::JournalCorrupt { sequence: 1, .. }),
        "{corrupt}"
    );

    let foreign = refused(&|path| {
        let file = legacy_journal(path);
        let bytes = std::fs::read(&file).unwrap();
        let header = zeusdb_vector_core::encode_legacy_header(7, 1);
        let mut edited = header.to_vec();
        edited.extend_from_slice(&bytes[64..]);
        std::fs::write(&file, edited).unwrap();
    });
    assert!(
        matches!(foreign, Error::JournalNotThisCollection { .. }),
        "{foreign}"
    );

    let missing = refused(&|path| std::fs::remove_file(legacy_journal(path)).unwrap());
    match missing {
        Error::JournalMissing { recorded, .. } => assert_eq!(recorded, "journaled.zdb.zdbwal"),
        other => panic!("expected the journal to be missing, got {other}"),
    }
}

/// A process killed after the upgrade created the journal's empty
/// directory and before its checkpoint committed leaves that directory
/// beside the old one and its file, and the next load replays the file and
/// upgrades again.
#[test]
fn an_empty_journal_directory_beside_an_old_directory_is_upgraded_again() {
    let name = FIXTURES[2];
    let temp = TempDir::new("upgrade-tests");
    let path = copy_fixture(name, &temp.0);
    std::fs::create_dir(journal_path(&path).unwrap()).unwrap();
    let (upgraded, report) = recover(&path).unwrap();
    assert!(report.upgraded_from.is_some());
    holds_fixture(&upgraded, name).unwrap();
}

/// A journal of format version 1 left beside the 5.0.0 directory its
/// upgrade wrote, which a crash between the commit and the file's deletion
/// leaves, is deleted at the next load, since the directory holds every
/// record it holds. One of another collection is left where it is.
#[test]
fn a_covered_version_1_journal_goes_and_a_foreign_one_stays() {
    let name = FIXTURES[1];
    let temp = TempDir::new("upgrade-tests");
    let path = copy_fixture(name, &temp.0);
    let original = std::fs::read(legacy_journal(&path)).unwrap();
    let (upgraded, _) = recover(&path).unwrap();
    drop(upgraded);

    std::fs::write(legacy_journal(&path), &original).unwrap();
    let (reopened, report) = recover(&path).unwrap();
    assert_eq!(
        report.upgraded_from, None,
        "a 5.0.0 directory is not upgraded again"
    );
    assert!(!legacy_journal(&path).exists(), "the covered file went");
    holds_fixture(&reopened, name).unwrap();
    drop(reopened);

    let mut foreign = zeusdb_vector_core::encode_legacy_header(7, 1).to_vec();
    foreign.extend_from_slice(&original[64..]);
    std::fs::write(legacy_journal(&path), &foreign).unwrap();
    let (again, _) = recover(&path).unwrap();
    assert!(legacy_journal(&path).exists(), "the foreign file stays");
    holds_fixture(&again, name).unwrap();
}

/// The fixtures themselves are what the releases wrote, read only by the
/// tests that copy them: each folder holds its directory, the file at
/// journal format version 1 and its expected records, and no load has
/// written to it.
#[test]
fn the_fixtures_are_as_the_releases_wrote_them() {
    for name in FIXTURES {
        let folder = super::journal_support::fixture_folder(name);
        let directory = folder.join("journaled.zdb");
        assert!(directory.join("manifest.json").is_file(), "{name}");
        assert!(legacy_journal(&directory).is_file(), "{name}");
        assert!(
            !journal_path(&directory).unwrap().exists(),
            "{name}: never upgraded"
        );
        assert!(
            !folder.join("journaled.zdb.zdblock").exists(),
            "{name}: never loaded"
        );
        let expected = fixture_expected(name);
        assert_eq!(
            manifest(&directory)["format_version"],
            expected["format_version"],
            "{name}"
        );
        assert_eq!(
            std::fs::metadata(legacy_journal(&directory)).unwrap().len(),
            expected["journal_bytes"].as_u64().unwrap(),
            "{name}"
        );
    }
}
