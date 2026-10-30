//! What a load asked to repair a damaged journal sets aside, and the cut.
//!
//! The load decides where the cut is, being the first record of the call
//! the damage touches, and replays every whole call before it. What this
//! module holds is the rest: the plan of what the repair sets aside, the
//! quarantine the plan is copied into, the cut itself, and the marker that
//! says a repair's checkpoint has not landed.
//!
//! A segment named at or below the directory's sequence whose header cannot
//! be used is stale by its name and holds no record past the checkpoint. A
//! repair sets it aside whole and removes it, and cuts nothing for it.
//!
//! # The quarantine
//!
//! `<name>.zdbjournal\quarantine\<UTC stamp>-<first sequence set aside>\`.
//! The stamp is written with no colon, which Windows forbids in a name. A
//! segment set aside whole keeps its name, and the part of a file the cut
//! leaves takes the file's name with the cut byte added. `repair.json`
//! records the damage, the checkpoint the journal follows, the cut where
//! there is one, the counts and every file, and an empty `UNSEALED` marks a
//! repair whose checkpoint has not landed.
//!
//! # The order
//!
//! The files, `repair.json` and `UNSEALED` are written and made durable
//! before the journal changes. Then the cut, then the stale segments set
//! aside go, then the load's checkpoint, then `UNSEALED` goes. A repair
//! killed after the quarantine leaves the journal as it was, or with only
//! its cut made, and the next repair under the same checkpoint takes the
//! folder as its own where its files hold every file it would copy. One
//! killed after the journal changed leaves a journal that reads whole, with
//! a marker naming the directory's own snapshot, so the next repair takes
//! the checkpoint. One killed after the checkpoint leaves a marker naming a
//! snapshot the directory no longer has, which the next repair removes
//! alone. A load that does not repair leaves the quarantine alone.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;
use zeusdb_vector_core::{
    whole_records_from, Cut, Error, JournalDir, StaleSegment, SEGMENT_HEADER_BYTES,
};

use crate::journal::{id_from_hex, id_hex, Damage, Repair};

/// The target the log records in this module carry, the journal's.
const LOG_TARGET: &str = "zeusdb_vector_database::journal";

/// The file that marks a repair whose checkpoint has not landed.
const UNSEALED: &str = "UNSEALED";

/// The file that records what a repair set aside.
const REPORT: &str = "repair.json";

/// One file a repair sets aside.
pub(super) struct Aside<'a> {
    /// Its name in the quarantine folder.
    name: String,
    /// The journal file it was copied from, by name.
    from: String,
    /// The byte of that file it begins at.
    at: u64,
    bytes: &'a [u8],
}

/// Where a repair cuts: the file by name, the byte, and the first sequence
/// it sets aside.
struct PlanCut {
    file: String,
    at: u64,
    sequence: u64,
}

/// What a repair sets aside and where it cuts, settled before anything is
/// written.
pub(super) struct Plan<'a> {
    /// The damage the repair names: the cut's where it cuts, and the first
    /// stale segment's header otherwise.
    pub(super) damage: Damage,
    /// The cut, where the repair makes one.
    cut: Option<PlanCut>,
    /// The stale segments set aside whole, by the number each file is named
    /// for, which the repair removes.
    pub(super) stale: Vec<u64>,
    /// The last sequence the repair keeps.
    pub(super) kept_through: u64,
    /// The snapshot of the checkpoint the journal follows, where the
    /// directory records one.
    snapshot: Option<u128>,
    /// The sequence that checkpoint holds.
    base_sequence: u64,
    files: Vec<Aside<'a>>,
    records_discarded: u64,
}

impl Plan<'_> {
    /// The bytes the plan sets aside.
    fn bytes_discarded(&self) -> u64 {
        self.files.iter().map(|file| file.bytes.len() as u64).sum()
    }

    /// The first sequence the cut sets aside, where the repair cuts.
    pub(super) fn first_discarded(&self) -> Option<u64> {
        self.cut.as_ref().map(|cut| cut.sequence)
    }

    /// The byte of its file the cut is at, where the repair cuts.
    pub(super) fn cut_at(&self) -> Option<u64> {
        self.cut.as_ref().map(|cut| cut.at)
    }

    /// The first sequence of the bytes the plan sets aside, which names the
    /// quarantine folder.
    fn first_set_aside(&self) -> u64 {
        self.stale
            .iter()
            .copied()
            .chain(self.first_discarded())
            .min()
            .unwrap_or_default()
    }

    /// What the caller reads of the repair, with the folder the bytes went
    /// into, or `None` where the medium refused the write.
    pub(super) fn report(&self, quarantine: Option<PathBuf>) -> Repair {
        Repair {
            kept_through: self.kept_through,
            discarded_from: self.first_discarded(),
            records_discarded: self.records_discarded,
            bytes_discarded: self.bytes_discarded(),
            quarantine,
        }
    }
}

/// A journal of segments as the load read it, in name order: the number
/// each file is named for, the files as a message names them, and their
/// bytes.
pub(super) struct JournalFiles<'a> {
    pub(super) numbers: &'a [u64],
    pub(super) files: &'a [String],
    pub(super) bytes: &'a [Vec<u8>],
}

/// The name a file of the journal takes in a message and in the quarantine.
fn name_of(file: &str) -> String {
    Path::new(file).file_name().map_or_else(
        || file.to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// The damage `cut` is for, as the caller reads it, in the file `file`.
fn damage_of(cut: &Cut, file: &str) -> Damage {
    Damage {
        kind: cut.kind,
        segment: name_of(file),
        at: cut.damage_at,
        sequence: cut.damage_sequence,
        detail: Some(cut.detail.clone()),
    }
}

/// The plan for a journal of segments: every segment in `stale` whole, and
/// where `cut` is given, the part of the segment the cut is in from the cut
/// on, whole where the cut takes the segment whole, and every later segment
/// whole. `reached` is the last sequence the reading kept, which the repair
/// keeps where it cuts nothing. `None` where there is nothing to set aside.
pub(super) fn plan_segments<'a>(
    cut: Option<&Cut>,
    stale: &[StaleSegment],
    journal: &JournalFiles<'a>,
    snapshot: Option<u128>,
    base_sequence: u64,
    reached: u64,
) -> Option<Plan<'a>> {
    let JournalFiles {
        numbers,
        files,
        bytes,
    } = *journal;
    let mut asides = Vec::new();
    for segment in stale {
        if let Some(i) = numbers.iter().position(|&number| number == segment.number) {
            let name = name_of(&files[i]);
            asides.push(Aside {
                name: name.clone(),
                from: name,
                at: 0,
                bytes: &bytes[i],
            });
        }
    }
    let mut damage = stale.first().map(|segment| Damage {
        kind: "header",
        segment: name_of(&segment.file),
        at: 0,
        sequence: segment.number,
        detail: Some(segment.detail.clone()),
    });
    let mut records = 0usize;
    let mut planned = None;
    if let Some(cut) = cut {
        let place = numbers.partition_point(|&number| number < cut.segment);
        let damage_place = numbers
            .partition_point(|&number| number < cut.damage_segment)
            .min(numbers.len().saturating_sub(1));
        for (i, ((&number, file), segment)) in numbers.iter().zip(files).zip(bytes).enumerate() {
            if i < place {
                continue;
            }
            let name = name_of(file);
            let from = if i == place {
                (cut.at as usize).min(segment.len())
            } else {
                0
            };
            let (first, records_from) = if i == place {
                (cut.sequence, from.max(SEGMENT_HEADER_BYTES))
            } else {
                (number, SEGMENT_HEADER_BYTES)
            };
            records += whole_records_from(segment, records_from, first, true);
            asides.push(Aside {
                name: if from == 0 {
                    name.clone()
                } else {
                    format!("{name}.{from}")
                },
                from: name,
                at: from as u64,
                bytes: &segment[from..],
            });
        }
        damage = Some(damage_of(
            cut,
            files.get(damage_place).map_or("", String::as_str),
        ));
        planned = Some(PlanCut {
            file: files
                .get(place)
                .map_or_else(String::new, |file| name_of(file)),
            at: cut.at,
            sequence: cut.sequence,
        });
    }
    Some(Plan {
        damage: damage?,
        kept_through: cut.map_or(reached, |cut| cut.sequence.saturating_sub(1)),
        cut: planned,
        stale: stale.iter().map(|segment| segment.number).collect(),
        snapshot,
        base_sequence,
        files: asides,
        records_discarded: records as u64,
    })
}

/// The plan for a journal of format version 1 cut at `cut`, being the file
/// `file` from the cut on. Version 1 frames no call, so the cut is at the
/// damaged record itself.
pub(super) fn plan_legacy<'a>(
    cut: &Cut,
    file: &str,
    bytes: &'a [u8],
    snapshot: Option<u128>,
    base_sequence: u64,
) -> Plan<'a> {
    let name = name_of(file);
    let from = (cut.at as usize).min(bytes.len());
    Plan {
        damage: damage_of(cut, file),
        cut: Some(PlanCut {
            file: name.clone(),
            at: cut.at,
            sequence: cut.sequence,
        }),
        stale: Vec::new(),
        kept_through: cut.sequence.saturating_sub(1),
        snapshot,
        base_sequence,
        files: vec![Aside {
            name: format!("{name}.{from}"),
            from: name,
            at: from as u64,
            bytes: &bytes[from..],
        }],
        records_discarded: whole_records_from(bytes, from, cut.sequence, false) as u64,
    }
}

/// `repair.json`.
#[derive(Serialize, Deserialize)]
struct Report {
    damage: ReportDamage,
    base: ReportBase,
    /// `None` where the repair set aside only stale segments.
    cut: Option<ReportCut>,
    kept_through: u64,
    records_discarded: u64,
    bytes_discarded: u64,
    files: Vec<ReportFile>,
    repaired_at: String,
}

#[derive(Serialize, Deserialize)]
struct ReportDamage {
    kind: String,
    segment: String,
    at: u64,
    sequence: u64,
    detail: Option<String>,
}

/// The checkpoint the journal followed when the repair cut it.
#[derive(Serialize, Deserialize)]
struct ReportBase {
    snapshot: Option<String>,
    sequence: u64,
}

#[derive(Serialize, Deserialize)]
struct ReportCut {
    segment: String,
    at: u64,
    sequence: u64,
}

#[derive(Serialize, Deserialize)]
struct ReportFile {
    name: String,
    from: String,
    at: u64,
    bytes: u64,
}

/// Why a quarantine was not written.
pub(super) enum SetAside {
    /// The folder could not be created, which is the repair's first write,
    /// with the system's error, for the caller to tell a medium that
    /// refuses writes from any other failure.
    Create(io::Error),
    /// A file could not be written once the folder was there.
    Write(Error),
}

/// Copy what `plan` sets aside into the quarantine of the journal `journal`
/// and return the folder's name, with every file, the report and the marker
/// made durable. The folder an unfinished repair under the same checkpoint
/// left is taken as this one's where its files hold every file this one
/// would copy, and nothing is copied again.
pub(super) fn set_aside(journal: &dyn JournalDir, plan: &Plan<'_>) -> Result<String, SetAside> {
    if let Some(folder) = reusable(journal, plan) {
        return Ok(folder);
    }
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%3fZ");
    let folder = format!("{stamp}-{}", plan.first_set_aside());
    journal
        .create_quarantine(&folder)
        .map_err(SetAside::Create)?;
    let quarantine = journal.quarantine();
    for file in &plan.files {
        quarantine
            .write(&format!("{folder}/{}", file.name), file.bytes)
            .map_err(SetAside::Write)?;
    }
    let report = Report {
        damage: ReportDamage {
            kind: plan.damage.kind.to_string(),
            segment: plan.damage.segment.clone(),
            at: plan.damage.at,
            sequence: plan.damage.sequence,
            detail: plan.damage.detail.clone(),
        },
        base: ReportBase {
            snapshot: plan.snapshot.map(id_hex),
            sequence: plan.base_sequence,
        },
        cut: plan.cut.as_ref().map(|cut| ReportCut {
            segment: cut.file.clone(),
            at: cut.at,
            sequence: cut.sequence,
        }),
        kept_through: plan.kept_through,
        records_discarded: plan.records_discarded,
        bytes_discarded: plan.bytes_discarded(),
        files: plan
            .files
            .iter()
            .map(|file| ReportFile {
                name: file.name.clone(),
                from: file.from.clone(),
                at: file.at,
                bytes: file.bytes.len() as u64,
            })
            .collect(),
        repaired_at: chrono::Utc::now().to_rfc3339(),
    };
    let json = serde_json::to_string_pretty(&report).map_err(|error| {
        SetAside::Write(Error::SerializeFailed {
            what: "repair report",
            error: error.to_string(),
        })
    })?;
    quarantine
        .write(&format!("{folder}/{REPORT}"), json.as_bytes())
        .map_err(SetAside::Write)?;
    quarantine
        .write(&format!("{folder}/{UNSEALED}"), b"")
        .map_err(SetAside::Write)?;
    journal.sync_quarantine(&folder).map_err(|error| {
        SetAside::Write(Error::JournalIoFailed {
            path: journal.path().to_path_buf(),
            what: "sync the quarantine of",
            error: error.to_string(),
        })
    })?;
    Ok(folder)
}

/// The report of the quarantine folder `folder`, where it reads.
fn read_report(journal: &dyn JournalDir, folder: &str) -> Option<Report> {
    let bytes = journal
        .quarantine()
        .read(&format!("{folder}/{REPORT}"))
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// The folder an unfinished repair under the same checkpoint left, named
/// for the same first sequence, whose files hold every file `plan` would
/// copy. A repair killed after its cut and before its stale segments went
/// leaves such a folder, and the next repair finishes in it.
fn reusable(journal: &dyn JournalDir, plan: &Plan<'_>) -> Option<String> {
    let suffix = format!("-{}", plan.first_set_aside());
    let quarantine = journal.quarantine();
    journal
        .quarantine_folders()
        .ok()?
        .into_iter()
        .filter(|folder| folder.ends_with(&suffix))
        .filter(|folder| quarantine.exists(&format!("{folder}/{UNSEALED}")))
        .find(|folder| {
            let Some(report) = read_report(journal, folder) else {
                return false;
            };
            let snapshot = report.base.snapshot.as_deref().and_then(id_from_hex);
            snapshot == plan.snapshot
                && report.base.sequence == plan.base_sequence
                && plan.files.iter().all(|file| {
                    quarantine
                        .read(&format!("{folder}/{}", file.name))
                        .is_ok_and(|held| held == file.bytes)
                })
        })
}

/// Every folder the journal's quarantine holds that an unfinished repair
/// left, with the snapshot of the checkpoint its report names, which is
/// `None` where the report names none or does not read.
pub(super) fn unfinished(journal: &dyn JournalDir) -> Vec<(String, Option<u128>)> {
    let quarantine = journal.quarantine();
    journal
        .quarantine_folders()
        .unwrap_or_default()
        .into_iter()
        .filter(|folder| quarantine.exists(&format!("{folder}/{UNSEALED}")))
        .map(|folder| {
            let snapshot = read_report(journal, &folder)
                .and_then(|report| report.base.snapshot)
                .as_deref()
                .and_then(id_from_hex);
            (folder, snapshot)
        })
        .collect()
}

/// Remove the marker from each of `folders`, once the checkpoint that seals
/// their repair has landed, and sync each folder. A marker that will not go
/// is logged and left, and the next repair removes it. A sync that fails is
/// returned, and the load fails with it.
pub(super) fn seal(journal: &dyn JournalDir, folders: &[String]) -> Result<(), Error> {
    let quarantine = journal.quarantine();
    for folder in folders {
        if let Err(error) = quarantine.remove(&format!("{folder}/{UNSEALED}")) {
            if error.kind() != io::ErrorKind::NotFound {
                warn!(target: LOG_TARGET, operation = "repair_marker_kept",
                    folder = folder.as_str(),
                    error = %error,
                    "A repair's checkpoint landed and its marker could not be removed; the next \
                     repair removes it"
                );
            }
        }
        journal
            .sync_quarantine(folder)
            .map_err(|error| Error::JournalIoFailed {
                path: journal.path().to_path_buf(),
                what: "sync the quarantine of",
                error: error.to_string(),
            })?;
    }
    Ok(())
}

/// Cut the journal of segments at the byte `at` of the segment `segment`,
/// removing it where the cut takes it whole, and remove every segment after
/// it. The later segments go first, the last of them first, so the journal
/// reads as one run of segments at every step, and the directory's entries
/// are synced last.
pub(super) fn cut_segments(
    journal: &dyn JournalDir,
    numbers: &[u64],
    segment: u64,
    at: u64,
) -> io::Result<()> {
    for &number in numbers.iter().rev().filter(|&&number| number > segment) {
        journal.remove_segment(number)?;
    }
    if at == 0 {
        journal.remove_segment(segment)?;
    } else {
        let file = journal.open_segment(segment)?;
        file.set_len(at)?;
        file.sync_all()?;
    }
    journal.sync()
}

/// Remove the stale segments `numbers` a repair set aside whole, and sync
/// the journal's directory. Each is named below every live segment, so the
/// journal reads as one run of segments at every step.
pub(super) fn remove_stale(journal: &dyn JournalDir, numbers: &[u64]) -> io::Result<()> {
    if numbers.is_empty() {
        return Ok(());
    }
    for &number in numbers {
        journal.remove_segment(number)?;
    }
    journal.sync()
}
