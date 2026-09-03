//! The write-ahead journal's file, its records, and the reader that tells a
//! torn tail from a corrupt middle.
//!
//! An append-only file beside an index directory. A 64 byte file header,
//! then records, each a 16 byte header, a payload and an 8 byte checksum
//! over both. This module knows what a record is and nothing of what its
//! payload means. The payloads are [`crate::operation`]'s, and applying one
//! is the index's.
//!
//! # Why not the artefact frame
//!
//! The frame's header carries the payload length and the file length and
//! its trailer carries a checksum over the whole payload, all three known
//! only when the file is finished. A log is never finished while the process
//! writing it lives, so a file-level frame would be a header with stale
//! lengths and no trailer, which is exactly the shape a crash leaves, and a
//! reader could not tell the two apart. Framing each record instead costs
//! 24 bytes a record and lets the reader stop at the first record that
//! fails and say where.
//!
//! # The shared first bytes
//!
//! The first seventeen bytes mean what the artefact frame's and the graph
//! dump's mean, being an eight byte magic, a `u32` format version, a `u32`
//! of reserved flags and a kind byte, so one reader dispatches on the magic
//! at offset zero and reads any of the three.
//!
//! # Layout
//!
//! ```text
//! File header, 64 bytes
//!    0   8  magic            u64   b"ZDBJOURN"
//!    8   4  format_version   u32   1
//!   12   4  flags            u32   reserved, zero
//!   16   1  kind             u8    JournalKind
//!   17   7  reserved         zero
//!   24  16  collection_id    u128  the collection the journal belongs to
//!   40   8  first_sequence   u64   the first record's sequence, at least 1
//!   48   8  reserved         u64   zero
//!   56   8  header_checksum  u64   over bytes 0 to 56
//!
//! Record
//!    0   8  sequence         u64   the previous record's plus one
//!    8   4  payload_len      u32
//!   12   2  kind             u16   OperationKind
//!   14   2  flags            u16   reserved, zero
//!   16   n  payload
//!   16+n 8  checksum         u64   over bytes 0 to 16+n
//! ```
//!
//! Every field is little-endian at a fixed width. Sequence zero is never a
//! record's. It is the value a checkpoint carries when no record has ever
//! been appended, so the first record of a fresh journal is sequence one.
//!
//! **The kind numbers are on disk for ever.** See [`OperationKind`] and
//! [`JournalKind`].
//!
//! # What bounds each length
//!
//! Every length is checked against what the file has earned before anything
//! is sized from it, which is the rule the graph dump and the artefact frame
//! follow. The file holds at least the header before any of it is read.
//! `payload_len` is held to the bytes remaining after the record header and
//! before an eight byte checksum, and to [`JOURNAL_MAX_PAYLOAD`], before the
//! payload is touched, so a header claiming a gigabyte on a kilobyte file
//! allocates nothing. `sequence` is held to the previous record's plus one,
//! `kind` to the kinds this build knows, `flags` to zero, and the checksum
//! is recomputed over the header and the payload. What is inside the
//! payload is bounded by `payload_len` and is the payload reader's business.
//!
//! # What the reader says
//!
//! A log is read one record at a time from a file that may end mid-record,
//! and the reader classifies whatever stops it as one of two things.
//!
//! A **torn tail** is a record the file ends inside, one that fails with
//! nothing after it that parses, a run of zeros a filesystem extended the
//! file with before the data landed, or a repeated sequence, being a record
//! appended twice. The reader names the byte the good records end at and
//! the caller truncates there. Nothing acknowledged is lost, because a
//! commit returns only once its records are whole.
//!
//! A **corrupt middle** is a record that fails while a record after it
//! parses at a later sequence. Appends are ordered, so a later record on
//! disk means the earlier one had landed, and what changed is the bytes
//! rather than the write. Skipping it would recover a state nobody
//! acknowledged, so the reader names the sequence and the byte, and the
//! caller refuses to open unless a repair is asked for by name.
//!
//! The record that decides between the two is any record after the failure
//! that parses whole at a sequence above the one expected, not only the
//! very next sequence. Two adjacent records damaged in the middle of a file
//! are still followed by records that prove they landed.
//!
//! # The writer
//!
//! Appends one record in one write, commits under a mode the caller chooses
//! at each call, and truncates in two steps that are each durable on their
//! own, being the body cut and synced, then the header rewritten and
//! synced. It takes no index type, holds no index lock and knows nothing of
//! a collection.

use crate::checksum::checksum_of;
use crate::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `b"ZDBJOURN"` read little-endian.
pub const JOURNAL_MAGIC: u64 = u64::from_le_bytes(*b"ZDBJOURN");

/// The only journal version this build writes, and the only one it reads.
const JOURNAL_VERSION: u32 = 1;

/// Bytes the file header occupies, checksum included.
pub const JOURNAL_HEADER_BYTES: usize = 64;

/// Bytes the header checksum is taken over, being everything before it.
const HEADER_CHECKSUM_BYTES: usize = 56;

/// Bytes a record's header occupies.
pub const JOURNAL_RECORD_HEADER_BYTES: usize = 16;

/// Bytes a record's checksum occupies.
pub const JOURNAL_RECORD_CHECKSUM_BYTES: usize = 8;

/// Bytes a record adds to its payload.
pub const JOURNAL_RECORD_OVERHEAD_BYTES: usize =
    JOURNAL_RECORD_HEADER_BYTES + JOURNAL_RECORD_CHECKSUM_BYTES;

/// The widest vector an index declares, in values. `dim` is held to it at
/// `create()`, and the index crate holds this copy to that ceiling by test.
const WIDEST_VECTOR: usize = 65_536;

/// The most centroids a codebook holds per subvector, being `bits` at its
/// ceiling of eight.
const MOST_CENTROIDS: usize = 256;

/// The widest payload a record may carry, and the ceiling every
/// `payload_len` is held to before the payload is touched.
///
/// Derived rather than chosen. A vector at the widest declaration is
/// 262,144 bytes. A codebook at that width and the most centroids, should a
/// later build record one, is the width times the centroids times four
/// bytes a value, 64 MiB, whatever the subvector count. A mebibyte of
/// metadata sits beside either with margin. It bounds a hostile length that
/// the remaining-bytes check alone would admit on a large file.
pub const JOURNAL_MAX_PAYLOAD: usize = WIDEST_VECTOR * MOST_CENTROIDS * 4 + (1 << 20);

/// What a journal file holds, at header byte 16.
///
/// **The numbers are on disk for ever. Never reuse one and never change
/// one.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JournalKind {
    /// An operation log, being every mutation since a checkpoint.
    Operations = 1,
}

impl JournalKind {
    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(JournalKind::Operations),
            _ => None,
        }
    }
}

/// What a record's payload is, at record byte 12.
///
/// **The numbers are on disk for ever. Never reuse one and never change
/// one.** A build that reads a kind it does not know refuses the record,
/// so a new kind is a new format version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationKind {
    /// One record inserted, with its whole content.
    Insert = 1,
    /// Records removed, by resolved id.
    Remove = 2,
    /// One record's metadata replaced wholesale.
    UpdateMetadata = 3,
    /// Every record removed and every counter reset.
    Clear = 4,
    /// The graph rebuilt over the live records.
    Compact = 5,
    /// The graph rebuilt under new parameters.
    Rebuild = 6,
    /// A term interned, at the id the dictionary issued.
    Intern = 7,
    /// The quantizer trained, with the stamp it took.
    Train = 8,
    /// Pairs added to the index's own metadata.
    AddMetadata = 9,
    /// The graph rebuilt over the quantized codes.
    RebuildQuantized = 10,
}

impl OperationKind {
    /// Every kind, in the order of its number.
    pub const ALL: [OperationKind; 10] = [
        OperationKind::Insert,
        OperationKind::Remove,
        OperationKind::UpdateMetadata,
        OperationKind::Clear,
        OperationKind::Compact,
        OperationKind::Rebuild,
        OperationKind::Intern,
        OperationKind::Train,
        OperationKind::AddMetadata,
        OperationKind::RebuildQuantized,
    ];

    /// The number on disk.
    pub fn code(self) -> u16 {
        self as u16
    }

    /// The kind a number names, or none.
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(OperationKind::Insert),
            2 => Some(OperationKind::Remove),
            3 => Some(OperationKind::UpdateMetadata),
            4 => Some(OperationKind::Clear),
            5 => Some(OperationKind::Compact),
            6 => Some(OperationKind::Rebuild),
            7 => Some(OperationKind::Intern),
            8 => Some(OperationKind::Train),
            9 => Some(OperationKind::AddMetadata),
            10 => Some(OperationKind::RebuildQuantized),
            _ => None,
        }
    }

    /// How the kind names itself in a message.
    pub fn label(self) -> &'static str {
        match self {
            OperationKind::Insert => "insert",
            OperationKind::Remove => "remove",
            OperationKind::UpdateMetadata => "update metadata",
            OperationKind::Clear => "clear",
            OperationKind::Compact => "compact",
            OperationKind::Rebuild => "rebuild",
            OperationKind::Intern => "intern",
            OperationKind::Train => "train",
            OperationKind::AddMetadata => "add metadata",
            OperationKind::RebuildQuantized => "rebuild quantized",
        }
    }
}

/// The file header, read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalHeader {
    /// The collection the journal belongs to, drawn when the collection is
    /// created and held in its configuration too, so a journal from another
    /// index is refused by content rather than by name.
    pub collection_id: u128,
    /// The sequence of the first record in the file. After a truncation it
    /// is the checkpoint's sequence plus one, which may be above the last
    /// record's when the body is empty.
    pub first_sequence: u64,
}

/// One record, borrowing its payload from the bytes it was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalRecord<'a> {
    pub sequence: u64,
    pub kind: OperationKind,
    /// Byte offset of the record's first byte in the file.
    pub offset: u64,
    pub payload: &'a [u8],
}

/// What stopped the reader, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JournalDamage {
    /// The file ends inside a record, or a record fails and nothing after
    /// it parses. `at` is the byte the good records end at, which is where
    /// the caller truncates, and `sequence` is the one the record at `at`
    /// was expected to carry.
    TornTail { at: u64, sequence: u64 },
    /// A record fails and a record after it parses at a later sequence, so
    /// the bytes changed after the record was written. `at` and `sequence`
    /// name the record that fails, and `detail` says how.
    Corrupt {
        at: u64,
        sequence: u64,
        detail: String,
    },
}

/// A journal read back, borrowing every payload from the bytes it was read
/// from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalContents<'a> {
    pub header: JournalHeader,
    /// Every record that parsed in sequence, in file order.
    pub records: Vec<JournalRecord<'a>>,
    /// What stopped the read, or nothing when the file ends at a record
    /// boundary.
    pub damage: Option<JournalDamage>,
    /// The byte the good records end at. The file's length when nothing is
    /// damaged, and the damage's `at` otherwise. A torn tail is cut here;
    /// a corrupt middle is cut here only under a repair asked for by name.
    pub good_bytes: u64,
}

impl JournalContents<'_> {
    /// The sequence the next record appended after `good_bytes` carries,
    /// being the last good record's plus one, or the header's first when
    /// the body is empty.
    pub fn next_sequence(&self) -> u64 {
        self.records
            .last()
            .map_or(self.header.first_sequence, |r| r.sequence + 1)
    }

    /// The refusal a corrupt middle earns, naming the file, the sequence
    /// and the byte. None for a torn tail or an undamaged file.
    pub fn refusal(&self, file: &str) -> Option<Error> {
        match &self.damage {
            Some(JournalDamage::Corrupt {
                at,
                sequence,
                detail,
            }) => Some(Error::JournalCorrupt {
                file: file.to_string(),
                sequence: *sequence,
                at: *at,
                detail: detail.clone(),
            }),
            _ => None,
        }
    }
}

// ============================================================================
// ENCODING
// ============================================================================

/// The file header for `collection_id`, with its first record at
/// `first_sequence`.
pub fn encode_journal_header(
    collection_id: u128,
    first_sequence: u64,
) -> [u8; JOURNAL_HEADER_BYTES] {
    let mut out = [0u8; JOURNAL_HEADER_BYTES];
    out[0..8].copy_from_slice(&JOURNAL_MAGIC.to_le_bytes());
    out[8..12].copy_from_slice(&JOURNAL_VERSION.to_le_bytes());
    out[16] = JournalKind::Operations.code();
    out[24..40].copy_from_slice(&collection_id.to_le_bytes());
    out[40..48].copy_from_slice(&first_sequence.to_le_bytes());
    let sum = checksum_of(&out[..HEADER_CHECKSUM_BYTES]);
    out[56..64].copy_from_slice(&sum.to_le_bytes());
    out
}

/// One record, header, payload and checksum, appended to `out`, which is
/// cleared first. The caller holds `payload` to [`JOURNAL_MAX_PAYLOAD`].
pub fn encode_journal_record(
    sequence: u64,
    kind: OperationKind,
    payload: &[u8],
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve(JOURNAL_RECORD_OVERHEAD_BYTES + payload.len());
    out.extend_from_slice(&sequence.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&kind.code().to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(payload);
    let sum = checksum_of(out);
    out.extend_from_slice(&sum.to_le_bytes());
}

// ============================================================================
// THE READER
// ============================================================================

fn take2(raw: &[u8], at: usize) -> [u8; 2] {
    [raw[at], raw[at + 1]]
}

fn take4(raw: &[u8], at: usize) -> [u8; 4] {
    let mut out = [0u8; 4];
    out.copy_from_slice(&raw[at..at + 4]);
    out
}

fn take8(raw: &[u8], at: usize) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&raw[at..at + 8]);
    out
}

fn take16(raw: &[u8], at: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(&raw[at..at + 16]);
    out
}

/// Read the file header, refusing each way it can be wrong by name, in the
/// order length, magic, version, checksum, reserved fields, kind, first
/// sequence.
fn read_header(bytes: &[u8], file: &str) -> Result<JournalHeader, Error> {
    let invalid = |detail: String| Error::JournalHeaderInvalid {
        file: file.to_string(),
        detail,
    };
    if bytes.len() < JOURNAL_HEADER_BYTES {
        return Err(invalid(format!(
            "the file holds {} bytes and the header is {}",
            bytes.len(),
            JOURNAL_HEADER_BYTES
        )));
    }
    let header = &bytes[..JOURNAL_HEADER_BYTES];
    if u64::from_le_bytes(take8(header, 0)) != JOURNAL_MAGIC {
        return Err(invalid(
            "the file does not open with the journal magic".into(),
        ));
    }
    let version = u32::from_le_bytes(take4(header, 8));
    if version != JOURNAL_VERSION {
        return Err(invalid(format!(
            "the journal is format version {version} and this build reads {JOURNAL_VERSION}"
        )));
    }
    let stored = u64::from_le_bytes(take8(header, 56));
    if stored != checksum_of(&header[..HEADER_CHECKSUM_BYTES]) {
        return Err(invalid("the header is corrupt".into()));
    }
    let flags = u32::from_le_bytes(take4(header, 12));
    let reserved_short = &header[17..24];
    let reserved_long = u64::from_le_bytes(take8(header, 48));
    if flags != 0 || reserved_short.iter().any(|&b| b != 0) || reserved_long != 0 {
        return Err(invalid(
            "the header sets a reserved field this build does not know".into(),
        ));
    }
    if JournalKind::from_code(header[16]).is_none() {
        return Err(invalid(format!(
            "the header names kind {}, which is not one this build writes",
            header[16]
        )));
    }
    let first_sequence = u64::from_le_bytes(take8(header, 40));
    if first_sequence == 0 {
        return Err(invalid(
            "the header names sequence 0 as its first record, and 0 is never a record's".into(),
        ));
    }
    Ok(JournalHeader {
        collection_id: u128::from_le_bytes(take16(header, 24)),
        first_sequence,
    })
}

/// What parsing one record at an offset found.
enum Parsed<'a> {
    /// A whole record at the expected sequence.
    Whole(JournalRecord<'a>),
    /// The file ends inside the record, or before its header is whole.
    Incomplete,
    /// The record is whole and wrong, with how.
    Failed(String),
}

/// Parse one record at `at`, expecting `expected` as its sequence. Every
/// length is held before anything is read from it, and nothing allocates.
fn parse_record(bytes: &[u8], at: usize, expected: u64) -> Parsed<'_> {
    let remaining = bytes.len().saturating_sub(at);
    if remaining < JOURNAL_RECORD_OVERHEAD_BYTES {
        return Parsed::Incomplete;
    }
    let head = &bytes[at..at + JOURNAL_RECORD_HEADER_BYTES];
    let sequence = u64::from_le_bytes(take8(head, 0));
    let payload_len = u32::from_le_bytes(take4(head, 8)) as usize;
    let kind = u16::from_le_bytes(take2(head, 12));
    let flags = u16::from_le_bytes(take2(head, 14));
    if payload_len > JOURNAL_MAX_PAYLOAD {
        return Parsed::Failed(format!(
            "a payload of {payload_len} bytes is above the ceiling of {JOURNAL_MAX_PAYLOAD}"
        ));
    }
    if payload_len > remaining - JOURNAL_RECORD_OVERHEAD_BYTES {
        return Parsed::Incomplete;
    }
    let end = at + JOURNAL_RECORD_HEADER_BYTES + payload_len;
    let stored = u64::from_le_bytes(take8(bytes, end));
    if stored != checksum_of(&bytes[at..end]) {
        return Parsed::Failed("checksum mismatch".into());
    }
    if sequence != expected {
        return Parsed::Failed(format!("sequence {sequence} where {expected} was expected"));
    }
    let Some(kind) = OperationKind::from_code(kind) else {
        return Parsed::Failed(format!("kind {kind} is not one this build knows"));
    };
    if flags != 0 {
        return Parsed::Failed("the record sets a reserved flag".into());
    }
    Parsed::Whole(JournalRecord {
        sequence,
        kind,
        offset: at as u64,
        payload: &bytes[at + JOURNAL_RECORD_HEADER_BYTES..end],
    })
}

/// Whether a whole record at a sequence above `expected` sits anywhere
/// from `from` on. This is what separates a corrupt middle from a torn
/// tail: a torn tail is followed by nothing that parses. The sequence is
/// compared before anything else is read, so a run of zeros or of ones
/// costs one comparison a byte.
fn a_later_record_parses(bytes: &[u8], from: usize, expected: u64) -> bool {
    let mut at = from;
    while at + JOURNAL_RECORD_OVERHEAD_BYTES <= bytes.len() {
        let sequence = u64::from_le_bytes(take8(bytes, at));
        if sequence > expected && matches!(parse_record(bytes, at, sequence), Parsed::Whole(_)) {
            return true;
        }
        at += 1;
    }
    false
}

/// Read a journal back, every record in sequence, classifying whatever
/// stops the read. `file` names the journal in a message. The header is
/// the only thing refused outright; damage inside the body is reported in
/// the contents and the caller decides.
pub fn read_journal<'a>(bytes: &'a [u8], file: &str) -> Result<JournalContents<'a>, Error> {
    let header = read_header(bytes, file)?;
    let mut records = Vec::new();
    let mut at = JOURNAL_HEADER_BYTES;
    let mut expected = header.first_sequence;
    let mut damage = None;
    while at < bytes.len() {
        let detail = match parse_record(bytes, at, expected) {
            Parsed::Whole(record) => {
                at += JOURNAL_RECORD_OVERHEAD_BYTES + record.payload.len();
                expected = record.sequence.wrapping_add(1);
                records.push(record);
                continue;
            }
            Parsed::Incomplete => "the record runs past the end of the file".to_string(),
            Parsed::Failed(detail) => detail,
        };
        damage = Some(if a_later_record_parses(bytes, at + 1, expected) {
            JournalDamage::Corrupt {
                at: at as u64,
                sequence: expected,
                detail,
            }
        } else {
            JournalDamage::TornTail {
                at: at as u64,
                sequence: expected,
            }
        });
        break;
    }
    let good_bytes = match &damage {
        Some(JournalDamage::TornTail { at, .. }) | Some(JournalDamage::Corrupt { at, .. }) => *at,
        None => bytes.len() as u64,
    };
    Ok(JournalContents {
        header,
        records,
        damage,
        good_bytes,
    })
}

// ============================================================================
// THE WRITER
// ============================================================================

/// What a commit does with the bytes it has appended.
///
/// The durability policies are the caller's. A per-call policy commits with
/// `Sync`; an interval policy commits with `Deferred` and syncs from its own
/// thread through a [`JournalSyncHandle`]; a policy that never syncs commits
/// with `Deferred` and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitMode {
    /// The file's data is on the device before the call returns.
    Sync,
    /// The bytes are in the kernel and the kernel writes them when it will.
    Deferred,
}

/// A handle that syncs the journal's file and does nothing else, for a
/// thread that syncs on an interval.
#[derive(Clone, Debug)]
pub struct JournalSyncHandle {
    path: PathBuf,
    file: Arc<File>,
}

impl JournalSyncHandle {
    /// Sync the file's data to the device.
    pub fn sync(&self) -> Result<(), Error> {
        self.file
            .sync_data()
            .map_err(|error| Error::JournalIoFailed {
                path: self.path.clone(),
                what: "sync",
                error: error.to_string(),
            })
    }
}

/// The appender.
///
/// One open file, appended at its end. Every append is one write call, so a
/// process killed inside one leaves a prefix of the record, which the
/// reader classifies as a torn tail.
#[derive(Debug)]
pub struct JournalWriter {
    path: PathBuf,
    file: Arc<File>,
    collection_id: u128,
    first_sequence: u64,
    next_sequence: u64,
    buffer: Vec<u8>,
}

impl JournalWriter {
    fn io<'a>(path: &'a Path, what: &'static str) -> impl Fn(std::io::Error) -> Error + 'a {
        move |error| Error::JournalIoFailed {
            path: path.to_path_buf(),
            what,
            error: error.to_string(),
        }
    }

    /// Create a journal at `path` with a header and no records, replacing
    /// any file there. Durable before it returns. `first_sequence` is at
    /// least one, since zero is never a record's.
    pub fn create(path: &Path, collection_id: u128, first_sequence: u64) -> Result<Self, Error> {
        if first_sequence == 0 {
            return Err(Error::JournalIoFailed {
                path: path.to_path_buf(),
                what: "create",
                error: "sequence 0 is never a record's".into(),
            });
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .map_err(Self::io(path, "create"))?;
        file.write_all(&encode_journal_header(collection_id, first_sequence))
            .map_err(Self::io(path, "write the header of"))?;
        file.sync_all().map_err(Self::io(path, "sync"))?;
        Ok(JournalWriter {
            path: path.to_path_buf(),
            file: Arc::new(file),
            collection_id,
            first_sequence,
            next_sequence: first_sequence,
            buffer: Vec::new(),
        })
    }

    /// Open a journal the reader has read, for appending after its good
    /// bytes. The file is cut to that length and synced first, so a torn
    /// tail is gone before the next record lands. When the body is empty
    /// the next record carries `next_if_empty`, and a header naming any
    /// other first sequence is rewritten to it, which completes a
    /// truncation a crash left half done. When the body holds records the
    /// next record follows the last one and `next_if_empty` is not read.
    pub fn open_for_append(
        path: &Path,
        contents: &JournalContents<'_>,
        next_if_empty: u64,
    ) -> Result<Self, Error> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Self::io(path, "open"))?;
        file.set_len(contents.good_bytes)
            .map_err(Self::io(path, "truncate"))?;
        file.sync_all().map_err(Self::io(path, "sync"))?;
        let mut writer = JournalWriter {
            path: path.to_path_buf(),
            file: Arc::new(file),
            collection_id: contents.header.collection_id,
            first_sequence: contents.header.first_sequence,
            next_sequence: contents.next_sequence(),
            buffer: Vec::new(),
        };
        if contents.records.is_empty() {
            if next_if_empty == 0 {
                return Err(Error::JournalIoFailed {
                    path: path.to_path_buf(),
                    what: "open",
                    error: "sequence 0 is never a record's".into(),
                });
            }
            if next_if_empty != writer.first_sequence {
                writer.restate_header(next_if_empty)?;
            }
        }
        (&*writer.file)
            .seek(SeekFrom::End(0))
            .map_err(Self::io(path, "seek"))?;
        Ok(writer)
    }

    /// Where the file is.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The collection the journal belongs to.
    pub fn collection_id(&self) -> u128 {
        self.collection_id
    }

    /// The sequence the header names as its first record's.
    pub fn first_sequence(&self) -> u64 {
        self.first_sequence
    }

    /// The sequence the next append carries.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// The sequence the journal has reached, being the last record
    /// appended, or the first sequence less one when nothing has been.
    /// What a checkpoint records after it syncs.
    pub fn sequence_reached(&self) -> u64 {
        self.next_sequence - 1
    }

    /// The file's length, from the filesystem.
    pub fn file_len(&self) -> Result<u64, Error> {
        self.file
            .metadata()
            .map(|m| m.len())
            .map_err(Self::io(&self.path, "measure"))
    }

    /// A handle for a thread that syncs on an interval.
    pub fn sync_handle(&self) -> JournalSyncHandle {
        JournalSyncHandle {
            path: self.path.clone(),
            file: self.file.clone(),
        }
    }

    /// Append one record in one write and return its sequence. The bytes
    /// are in the kernel when this returns and on the device after a
    /// [`CommitMode::Sync`] commit.
    pub fn append(&mut self, kind: OperationKind, payload: &[u8]) -> Result<u64, Error> {
        if payload.len() > JOURNAL_MAX_PAYLOAD {
            return Err(Error::JournalIoFailed {
                path: self.path.clone(),
                what: "append to",
                error: format!(
                    "a payload of {} bytes is above the ceiling of {}",
                    payload.len(),
                    JOURNAL_MAX_PAYLOAD
                ),
            });
        }
        let sequence = self.next_sequence;
        let Some(after) = sequence.checked_add(1) else {
            return Err(Error::JournalIoFailed {
                path: self.path.clone(),
                what: "append to",
                error: "the sequence space is exhausted".into(),
            });
        };
        encode_journal_record(sequence, kind, payload, &mut self.buffer);
        (&*self.file)
            .write_all(&self.buffer)
            .map_err(Self::io(&self.path, "append to"))?;
        self.next_sequence = after;
        Ok(sequence)
    }

    /// Make everything appended so far durable under `mode`.
    pub fn commit(&mut self, mode: CommitMode) -> Result<(), Error> {
        match mode {
            CommitMode::Sync => self.sync(),
            CommitMode::Deferred => Ok(()),
        }
    }

    /// Sync the file's data to the device whatever the policy, which a
    /// checkpoint does before it records the sequence it holds.
    pub fn sync(&mut self) -> Result<(), Error> {
        self.file.sync_data().map_err(Self::io(&self.path, "sync"))
    }

    /// Drop every record appended so far, which a checkpoint now holds.
    ///
    /// Two steps, each durable on its own. The body is cut back to the
    /// header and synced, then the header is rewritten to name the next
    /// sequence as its first and synced. A crash between the two leaves an
    /// empty body under a header naming the old first sequence, which the
    /// reader accepts as a journal with no records, and the next
    /// [`open_for_append`](Self::open_for_append) completes the second step.
    pub fn truncate(&mut self) -> Result<(), Error> {
        self.cut_body()?;
        self.restate_header(self.next_sequence)?;
        (&*self.file)
            .seek(SeekFrom::End(0))
            .map_err(Self::io(&self.path, "seek"))
            .map(|_| ())
    }

    /// The first step of a truncation: the body cut to the header and the
    /// cut synced.
    fn cut_body(&mut self) -> Result<(), Error> {
        self.file
            .set_len(JOURNAL_HEADER_BYTES as u64)
            .map_err(Self::io(&self.path, "truncate"))?;
        self.file.sync_all().map_err(Self::io(&self.path, "sync"))
    }

    /// The second step of a truncation: the header rewritten with
    /// `first_sequence` and synced. The position is left after the header.
    fn restate_header(&mut self, first_sequence: u64) -> Result<(), Error> {
        let mut file = &*self.file;
        file.seek(SeekFrom::Start(0))
            .map_err(Self::io(&self.path, "seek"))?;
        file.write_all(&encode_journal_header(self.collection_id, first_sequence))
            .map_err(Self::io(&self.path, "write the header of"))?;
        file.sync_all().map_err(Self::io(&self.path, "sync"))?;
        self.first_sequence = first_sequence;
        self.next_sequence = first_sequence;
        Ok(())
    }
}

// ============================================================================
// THE TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::fuzz::{Rng, HOSTILE};

    /// A record of the corpus: its offset, its payload length, its
    /// sequence and its kind.
    type Entry = (usize, usize, u64, OperationKind);

    fn header_error(bytes: &[u8]) -> String {
        match read_journal(bytes, "t") {
            Err(Error::JournalHeaderInvalid { detail, .. }) => detail,
            Err(other) => panic!("expected a header refusal, got {other:?}"),
            Ok(_) => panic!("expected a header refusal, got contents"),
        }
    }

    /// Recompute the header checksum, so a field mutation reaches the
    /// field's own check.
    fn restamp_header(blob: &mut [u8]) {
        let sum = checksum_of(&blob[..HEADER_CHECKSUM_BYTES]);
        blob[56..64].copy_from_slice(&sum.to_le_bytes());
    }

    /// Recompute the checksum of the record at `at` whose payload is
    /// `payload_len` bytes, so a field mutation reaches the field's own
    /// check.
    fn restamp_record(blob: &mut [u8], at: usize, payload_len: usize) {
        let end = at + JOURNAL_RECORD_HEADER_BYTES + payload_len;
        let sum = checksum_of(&blob[at..end]);
        blob[end..end + 8].copy_from_slice(&sum.to_le_bytes());
    }

    /// A payload of `len` bytes that is not all one value.
    fn payload(seed: u32, len: usize) -> Vec<u8> {
        (0..len as u32)
            .map(|i| (i.wrapping_mul(31).wrapping_add(seed * 7) % 251) as u8)
            .collect()
    }

    /// A valid journal of `n` records with varied kinds and sizes, and the
    /// offset and length of each record's payload.
    fn corpus(n: usize) -> (Vec<u8>, Vec<Entry>) {
        let mut bytes =
            encode_journal_header(0x1234_5678_9abc_def0_0fed_cba9_8765_4321, 1).to_vec();
        let mut table = Vec::new();
        let mut record = Vec::new();
        for i in 0..n {
            let kind = OperationKind::ALL[i % OperationKind::ALL.len()];
            let len = [0usize, 5, 40, 133, 300, 17, 64, 1][i % 8];
            let body = payload(i as u32, len);
            let sequence = (i + 1) as u64;
            encode_journal_record(sequence, kind, &body, &mut record);
            table.push((bytes.len(), len, sequence, kind));
            bytes.extend_from_slice(&record);
        }
        (bytes, table)
    }

    fn end_of(entry: &Entry) -> usize {
        entry.0 + JOURNAL_RECORD_OVERHEAD_BYTES + entry.1
    }

    // ------------------------------------------------------------------
    // The file and the record
    // ------------------------------------------------------------------

    /// The first seventeen bytes mean what the frame's and the dump's mean,
    /// and the three magics differ, so a reader dispatching on the magic
    /// tells the three apart at byte zero.
    #[test]
    fn the_header_shares_its_first_seventeen_bytes_with_the_frame_and_the_dump() {
        let bytes = encode_journal_header(7, 1);
        assert_eq!(&bytes[..8], b"ZDBJOURN");
        assert_eq!(u64::from_le_bytes(take8(&bytes, 0)), JOURNAL_MAGIC);
        assert_ne!(JOURNAL_MAGIC, crate::frame::FRAME_MAGIC);
        assert_ne!(JOURNAL_MAGIC, u64::from_le_bytes(*b"ZDBGRAPH"));
        assert_ne!(crate::frame::FRAME_MAGIC, u64::from_le_bytes(*b"ZDBGRAPH"));
        assert_eq!(u32::from_le_bytes(take4(&bytes, 8)), 1);
        assert_eq!(u32::from_le_bytes(take4(&bytes, 12)), 0);
        assert_eq!(bytes[16], JournalKind::Operations as u8);
        assert_eq!(bytes.len(), JOURNAL_HEADER_BYTES);
        assert_eq!(JOURNAL_HEADER_BYTES, crate::frame::FRAME_HEADER_BYTES);
    }

    /// The header's fields sit where the layout says, and a header reads
    /// back to what was written.
    #[test]
    fn the_header_lays_out_as_documented() {
        let id = 0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10u128;
        let bytes = encode_journal_header(id, 0x1122_3344_5566_7788);
        assert_eq!(&bytes[17..24], &[0u8; 7]);
        assert_eq!(u128::from_le_bytes(take16(&bytes, 24)), id);
        assert_eq!(u64::from_le_bytes(take8(&bytes, 40)), 0x1122_3344_5566_7788);
        assert_eq!(u64::from_le_bytes(take8(&bytes, 48)), 0);
        assert_eq!(
            u64::from_le_bytes(take8(&bytes, 56)),
            checksum_of(&bytes[..56])
        );
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(
            contents.header,
            JournalHeader {
                collection_id: id,
                first_sequence: 0x1122_3344_5566_7788
            }
        );
        assert!(contents.records.is_empty());
        assert_eq!(contents.damage, None);
        assert_eq!(contents.good_bytes, 64);
        assert_eq!(contents.next_sequence(), 0x1122_3344_5566_7788);
    }

    /// A record is a 16 byte header, the payload and an 8 byte checksum
    /// over both, with every field where the layout says.
    #[test]
    fn a_record_lays_out_as_documented() {
        let mut out = Vec::new();
        encode_journal_record(
            0x0102_0304_0506_0708,
            OperationKind::Intern,
            b"hello",
            &mut out,
        );
        assert_eq!(out.len(), JOURNAL_RECORD_OVERHEAD_BYTES + 5);
        assert_eq!(u64::from_le_bytes(take8(&out, 0)), 0x0102_0304_0506_0708);
        assert_eq!(u32::from_le_bytes(take4(&out, 8)), 5);
        assert_eq!(u16::from_le_bytes(take2(&out, 12)), 7);
        assert_eq!(u16::from_le_bytes(take2(&out, 14)), 0);
        assert_eq!(&out[16..21], b"hello");
        assert_eq!(u64::from_le_bytes(take8(&out, 21)), checksum_of(&out[..21]));
        // Encoding clears the buffer first, so a reused buffer holds one
        // record.
        encode_journal_record(9, OperationKind::Clear, &[], &mut out);
        assert_eq!(out.len(), JOURNAL_RECORD_OVERHEAD_BYTES);
    }

    /// The ten kinds carry the numbers the format fixes, every number reads
    /// back to its kind, and no other number is a kind. The numbers are on
    /// disk for ever.
    #[test]
    fn the_kind_numbers_are_fixed() {
        let expected = [
            (OperationKind::Insert, 1u16, "insert"),
            (OperationKind::Remove, 2, "remove"),
            (OperationKind::UpdateMetadata, 3, "update metadata"),
            (OperationKind::Clear, 4, "clear"),
            (OperationKind::Compact, 5, "compact"),
            (OperationKind::Rebuild, 6, "rebuild"),
            (OperationKind::Intern, 7, "intern"),
            (OperationKind::Train, 8, "train"),
            (OperationKind::AddMetadata, 9, "add metadata"),
            (OperationKind::RebuildQuantized, 10, "rebuild quantized"),
        ];
        assert_eq!(OperationKind::ALL.len(), expected.len());
        for (i, (kind, code, label)) in expected.iter().enumerate() {
            assert_eq!(OperationKind::ALL[i], *kind);
            assert_eq!(kind.code(), *code);
            assert_eq!(OperationKind::from_code(*code), Some(*kind));
            assert_eq!(kind.label(), *label);
        }
        for code in [0u16, 11, 12, 255, 256, u16::MAX] {
            assert_eq!(OperationKind::from_code(code), None);
        }
        assert_eq!(JournalKind::Operations.code(), 1);
        assert_eq!(JournalKind::from_code(1), Some(JournalKind::Operations));
        assert_eq!(JournalKind::from_code(0), None);
        assert_eq!(JournalKind::from_code(2), None);
    }

    /// The checksum over a record is the engine's own, the one the frame
    /// and the dump use, so one algorithm verifies every artefact.
    #[test]
    fn the_checksum_is_the_engines_own() {
        let mut out = Vec::new();
        encode_journal_record(1, OperationKind::Insert, &[1, 2, 3], &mut out);
        let sum = u64::from_le_bytes(take8(&out, 19));
        assert_eq!(sum, crate::checksum_of(&out[..19]));
        let header = encode_journal_header(1, 1);
        assert_eq!(
            u64::from_le_bytes(take8(&header, 56)),
            crate::checksum_of(&header[..56])
        );
    }

    /// `JOURNAL_MAX_PAYLOAD` is a codebook at the widest declaration and the
    /// most centroids, plus a mebibyte, and the widest vector sits far
    /// under it.
    #[test]
    fn the_payload_ceiling_is_derived() {
        assert_eq!(JOURNAL_MAX_PAYLOAD, 65 << 20);
        assert_eq!(WIDEST_VECTOR * MOST_CENTROIDS * 4, 64 << 20);
        const { assert!(WIDEST_VECTOR * 4 + (1 << 20) < JOURNAL_MAX_PAYLOAD) }
        const { assert!(JOURNAL_MAX_PAYLOAD < u32::MAX as usize) }
    }

    // ------------------------------------------------------------------
    // Every bound, one test a row
    // ------------------------------------------------------------------

    /// A file shorter than its header is refused before anything is read,
    /// naming what it holds, and a header alone is a journal with no
    /// records.
    #[test]
    fn a_file_shorter_than_its_header_is_refused() {
        assert_eq!(
            header_error(&[]),
            "the file holds 0 bytes and the header is 64"
        );
        let header = encode_journal_header(1, 1);
        assert_eq!(
            header_error(&header[..63]),
            "the file holds 63 bytes and the header is 64"
        );
        assert!(read_journal(&header, "t").unwrap().records.is_empty());
    }

    /// Every way the header can be wrong is refused by name, in the
    /// reader's order.
    #[test]
    fn every_header_damage_is_refused_by_name() {
        let good = encode_journal_header(5, 3);
        let mut bad = good;
        bad[0] = b'X';
        assert!(header_error(&bad).contains("journal magic"));
        let mut bad = good;
        bad[8] = 2;
        restamp_header(&mut bad);
        assert!(header_error(&bad).contains("format version 2"));
        let mut bad = good;
        bad[30] ^= 1;
        assert!(header_error(&bad).contains("header is corrupt"));
        for at in [12usize, 17, 23, 48, 55] {
            let mut bad = good;
            bad[at] = 1;
            restamp_header(&mut bad);
            assert!(header_error(&bad).contains("reserved field"), "byte {at}");
        }
        let mut bad = good;
        bad[16] = 2;
        restamp_header(&mut bad);
        assert!(header_error(&bad).contains("kind 2"));
        let mut bad = good;
        bad[40..48].copy_from_slice(&0u64.to_le_bytes());
        restamp_header(&mut bad);
        assert!(header_error(&bad).contains("sequence 0"));
        // A frame and a dump are refused at the magic rather than read as
        // a journal.
        let frame = crate::frame::frame(
            crate::frame::FrameKind::TermDictionary,
            crate::frame::FrameEncoding::Engine,
            0,
            &[],
        );
        assert!(header_error(&frame).contains("journal magic"));
    }

    /// `payload_len` is held to the bytes remaining and to the ceiling
    /// before the payload is touched. A record claiming more than the file
    /// holds is a torn tail at its own offset, one claiming more than the
    /// ceiling fails at its own offset, and a good record before either is
    /// kept.
    #[test]
    fn a_payload_length_is_held_before_anything_is_sized_from_it() {
        let (good, table) = corpus(2);
        let second = table[1].0;
        // Past the file but under the ceiling, which is a torn tail: the
        // file ends inside the record.
        let mut bad = good.clone();
        bad[second + 8..second + 12].copy_from_slice(&(1u32 << 30).to_le_bytes());
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 1);
        assert_eq!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: second as u64,
                sequence: 2
            })
        );
        assert_eq!(contents.good_bytes, second as u64);
        // Above the ceiling, on a file that could never hold it.
        let mut bad = good.clone();
        bad[second + 8..second + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 1);
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::TornTail { sequence: 2, .. })
        ));
        // Exactly the ceiling is a length the reader admits, and one above
        // it is not, even when the file is long enough.
        let mut record = Vec::new();
        let body = vec![0u8; JOURNAL_MAX_PAYLOAD];
        encode_journal_record(1, OperationKind::Insert, &body, &mut record);
        let mut file = encode_journal_header(1, 1).to_vec();
        file.extend_from_slice(&record);
        let contents = read_journal(&file, "t").unwrap();
        assert_eq!(contents.records.len(), 1);
        assert_eq!(contents.records[0].payload.len(), JOURNAL_MAX_PAYLOAD);
        file[64 + 8..64 + 12].copy_from_slice(&((JOURNAL_MAX_PAYLOAD + 1) as u32).to_le_bytes());
        file.push(0);
        let contents = read_journal(&file, "t").unwrap();
        assert!(contents.records.is_empty());
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: 64,
                sequence: 1
            })
        ));
    }

    /// A record's sequence is the previous record's plus one, or the
    /// header's first for the first record. A gap, a repeat and a first
    /// record at the wrong sequence each fail the record.
    #[test]
    fn a_sequence_is_held_to_the_previous_plus_one() {
        let (good, table) = corpus(3);
        let third = table[2].0;
        // A gap.
        let mut bad = good.clone();
        bad[third..third + 8].copy_from_slice(&4u64.to_le_bytes());
        restamp_record(&mut bad, third, table[2].1);
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::TornTail { sequence: 3, .. })
        ));
        // A repeat, being the last record appended twice.
        let mut dup = good.clone();
        let last = good[third..].to_vec();
        dup.extend_from_slice(&last);
        let contents = read_journal(&dup, "t").unwrap();
        assert_eq!(contents.records.len(), 3);
        assert_eq!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: good.len() as u64,
                sequence: 4
            })
        );
        assert_eq!(contents.good_bytes, good.len() as u64);
        // The header names a first sequence the first record does not
        // carry.
        let mut bad = good.clone();
        bad[40..48].copy_from_slice(&2u64.to_le_bytes());
        restamp_header(&mut bad);
        let contents = read_journal(&bad, "t").unwrap();
        assert!(contents.records.is_empty());
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::Corrupt {
                at: 64,
                sequence: 2,
                ..
            })
        ));
    }

    /// A kind this build does not know fails the record, and every kind it
    /// does know reads back.
    #[test]
    fn a_kind_is_held_to_the_ten() {
        let (good, table) = corpus(10);
        let contents = read_journal(&good, "t").unwrap();
        assert_eq!(contents.records.len(), 10);
        for (record, entry) in contents.records.iter().zip(&table) {
            assert_eq!(record.kind, entry.3);
            assert_eq!(record.sequence, entry.2);
            assert_eq!(record.offset, entry.0 as u64);
            assert_eq!(record.payload.len(), entry.1);
        }
        for code in [0u16, 11, u16::MAX] {
            let mut bad = good.clone();
            let at = table[4].0;
            bad[at + 12..at + 14].copy_from_slice(&code.to_le_bytes());
            restamp_record(&mut bad, at, table[4].1);
            let contents = read_journal(&bad, "t").unwrap();
            assert_eq!(contents.records.len(), 4);
            match contents.damage {
                Some(JournalDamage::Corrupt {
                    at: found,
                    sequence,
                    detail,
                }) => {
                    assert_eq!(found, at as u64);
                    assert_eq!(sequence, 5);
                    assert_eq!(detail, format!("kind {code} is not one this build knows"));
                }
                other => panic!("expected a corrupt middle, got {other:?}"),
            }
        }
    }

    /// A record setting a reserved flag fails.
    #[test]
    fn a_flag_is_held_to_zero() {
        let (good, table) = corpus(2);
        let at = table[1].0;
        let mut bad = good.clone();
        bad[at + 14] = 1;
        restamp_record(&mut bad, at, table[1].1);
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 1);
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::TornTail { sequence: 2, .. })
        ));
    }

    /// A record whose checksum does not hold fails, whether the change is
    /// in its header or its payload.
    #[test]
    fn a_checksum_is_recomputed_over_header_and_payload() {
        let (good, table) = corpus(2);
        let at = table[1].0;
        for offset in [0usize, 9, 13, 16, 16 + table[1].1 - 1, 16 + table[1].1] {
            let mut bad = good.clone();
            bad[at + offset] ^= 0x40;
            let contents = read_journal(&bad, "t").unwrap();
            assert_eq!(contents.records.len(), 1, "offset {offset}");
            assert!(
                matches!(
                    contents.damage,
                    Some(JournalDamage::TornTail { sequence: 2, .. })
                ),
                "offset {offset}"
            );
        }
    }

    // ------------------------------------------------------------------
    // The three classifications
    // ------------------------------------------------------------------

    /// An undamaged file reads back whole, with no damage and its length
    /// as the good bytes.
    #[test]
    fn a_whole_file_is_good() {
        let (good, table) = corpus(7);
        let contents = read_journal(&good, "t").unwrap();
        assert_eq!(contents.records.len(), 7);
        assert_eq!(contents.damage, None);
        assert_eq!(contents.good_bytes, good.len() as u64);
        assert_eq!(contents.next_sequence(), 8);
        assert!(contents.refusal("t").is_none());
        for (record, entry) in contents.records.iter().zip(&table) {
            assert_eq!(record.payload, &good[entry.0 + 16..end_of(entry) - 8]);
        }
    }

    /// The four torn tails a crash can leave. The file ends inside a
    /// record, the last record's checksum fails with nothing after it, a
    /// run of zeros follows the last record, and the last record is
    /// appended twice. Each is reported at the byte the good records end
    /// at.
    #[test]
    fn a_torn_tail_is_reported_at_the_byte_to_truncate_at() {
        let (good, table) = corpus(5);
        let last = &table[4];
        let end = good.len() as u64;
        // The file ends inside the last record, at every cut.
        for cut in last.0 + 1..good.len() {
            let contents = read_journal(&good[..cut], "t").unwrap();
            assert_eq!(contents.records.len(), 4, "cut {cut}");
            assert_eq!(
                contents.damage,
                Some(JournalDamage::TornTail {
                    at: last.0 as u64,
                    sequence: 5
                }),
                "cut {cut}"
            );
            assert_eq!(contents.good_bytes, last.0 as u64);
            assert_eq!(contents.next_sequence(), 5);
        }
        // The last record's checksum fails.
        let mut bad = good.clone();
        bad[last.0 + 20] ^= 0xFF;
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 4);
        assert_eq!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: last.0 as u64,
                sequence: 5
            })
        );
        // A run of zeros after the last record, shorter and longer than a
        // record header.
        for zeros in [1usize, 23, 24, 300] {
            let mut bad = good.clone();
            bad.extend(std::iter::repeat_n(0u8, zeros));
            let contents = read_journal(&bad, "t").unwrap();
            assert_eq!(contents.records.len(), 5, "zeros {zeros}");
            assert_eq!(
                contents.damage,
                Some(JournalDamage::TornTail {
                    at: end,
                    sequence: 6
                }),
                "zeros {zeros}"
            );
            assert_eq!(contents.good_bytes, end);
        }
        // The last record appended twice.
        let mut dup = good.clone();
        dup.extend_from_slice(&good[last.0..]);
        let contents = read_journal(&dup, "t").unwrap();
        assert_eq!(contents.records.len(), 5);
        assert_eq!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: end,
                sequence: 6
            })
        );
        assert!(contents.refusal("t").is_none());
    }

    /// A record that fails with a record after it that parses at a later
    /// sequence is a corrupt middle, named by sequence and byte, whether
    /// the change is one payload byte, a header field, or two adjacent
    /// records both damaged.
    #[test]
    fn a_corrupt_middle_is_told_from_a_torn_tail() {
        let (good, table) = corpus(5);
        let third = &table[2];
        // One payload byte flipped in the third record.
        let mut bad = good.clone();
        bad[third.0 + 16 + 3] ^= 0x5A;
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert_eq!(
            contents.damage,
            Some(JournalDamage::Corrupt {
                at: third.0 as u64,
                sequence: 3,
                detail: "checksum mismatch".into()
            })
        );
        assert_eq!(contents.good_bytes, third.0 as u64);
        match contents.refusal("journal.zdbwal") {
            Some(Error::JournalCorrupt {
                file,
                sequence,
                at,
                detail,
            }) => {
                assert_eq!(file, "journal.zdbwal");
                assert_eq!(sequence, 3);
                assert_eq!(at, third.0 as u64);
                assert_eq!(detail, "checksum mismatch");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // The third record's length field damaged so it claims past the
        // file: the fourth record still proves it landed.
        let mut bad = good.clone();
        bad[third.0 + 8..third.0 + 12].copy_from_slice(&(1u32 << 20).to_le_bytes());
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::Corrupt { sequence: 3, .. })
        ));
        // The third and fourth records both damaged: the fifth proves both
        // landed, so it is corrupt rather than torn.
        let mut bad = good.clone();
        bad[third.0 + 16] ^= 1;
        bad[table[3].0 + 16] ^= 1;
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::Corrupt { sequence: 3, .. })
        ));
        // The same two damaged with nothing after them is a torn tail.
        let contents = read_journal(&bad[..table[4].0], "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::TornTail { sequence: 3, .. })
        ));
        // A whole record in the middle at the wrong sequence, being a
        // duplicate the writer never produces, is refused rather than cut.
        let mut bad = good[..third.0].to_vec();
        bad.extend_from_slice(&good[table[1].0..third.0]);
        bad.extend_from_slice(&good[third.0..]);
        let contents = read_journal(&bad, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert!(matches!(
            contents.damage,
            Some(JournalDamage::Corrupt { sequence: 3, .. })
        ));
    }

    // ------------------------------------------------------------------
    // The writer
    // ------------------------------------------------------------------

    fn read_file(path: &Path) -> Vec<u8> {
        std::fs::read(path).unwrap()
    }

    /// Records appended read back whole, a commit under either mode leaves
    /// the file readable, and the writer's sequences agree with the
    /// reader's.
    #[test]
    fn records_appended_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.zdbwal");
        let mut w = JournalWriter::create(&path, 42, 1).unwrap();
        assert_eq!(w.sequence_reached(), 0);
        assert_eq!(w.file_len().unwrap(), 64);
        assert_eq!(w.append(OperationKind::Insert, b"one").unwrap(), 1);
        assert_eq!(w.append(OperationKind::Remove, b"two").unwrap(), 2);
        w.commit(CommitMode::Deferred).unwrap();
        assert_eq!(w.append(OperationKind::Clear, &[]).unwrap(), 3);
        w.commit(CommitMode::Sync).unwrap();
        assert_eq!(w.next_sequence(), 4);
        assert_eq!(w.sequence_reached(), 3);
        assert_eq!(w.collection_id(), 42);
        assert_eq!(w.first_sequence(), 1);
        assert_eq!(w.path(), path);
        w.sync_handle().sync().unwrap();
        drop(w);
        let bytes = read_file(&path);
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.header.collection_id, 42);
        assert_eq!(contents.records.len(), 3);
        assert_eq!(contents.records[0].payload, b"one");
        assert_eq!(contents.records[1].kind, OperationKind::Remove);
        assert_eq!(contents.records[2].sequence, 3);
        assert_eq!(contents.damage, None);
        assert_eq!(contents.next_sequence(), 4);
    }

    /// A payload above the ceiling is refused at the append and appends
    /// nothing, and a first sequence of zero is refused at the create.
    #[test]
    fn the_writer_refuses_what_the_reader_would() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.zdbwal");
        assert!(matches!(
            JournalWriter::create(&path, 1, 0),
            Err(Error::JournalIoFailed { what: "create", .. })
        ));
        let mut w = JournalWriter::create(&path, 1, 1).unwrap();
        let big = vec![0u8; JOURNAL_MAX_PAYLOAD + 1];
        assert!(matches!(
            w.append(OperationKind::Insert, &big),
            Err(Error::JournalIoFailed {
                what: "append to",
                ..
            })
        ));
        assert_eq!(w.next_sequence(), 1);
        assert_eq!(w.file_len().unwrap(), 64);
    }

    /// A torn tail is cut when the journal is reopened for append, and the
    /// next record follows the last good one.
    #[test]
    fn a_torn_tail_is_cut_at_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.zdbwal");
        let mut w = JournalWriter::create(&path, 9, 1).unwrap();
        w.append(OperationKind::Insert, b"hello").unwrap();
        w.append(OperationKind::Insert, b"world").unwrap();
        w.commit(CommitMode::Sync).unwrap();
        drop(w);
        // A process killed inside the third append leaves a prefix of it.
        let mut partial = Vec::new();
        encode_journal_record(3, OperationKind::Insert, b"torn record", &mut partial);
        let good_len = std::fs::metadata(&path).unwrap().len();
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&partial[..20]).unwrap();
        }
        let bytes = read_file(&path);
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert_eq!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: good_len,
                sequence: 3
            })
        );
        let mut w = JournalWriter::open_for_append(&path, &contents, 1).unwrap();
        assert_eq!(w.file_len().unwrap(), good_len);
        assert_eq!(w.next_sequence(), 3);
        w.append(OperationKind::Insert, b"again").unwrap();
        w.commit(CommitMode::Sync).unwrap();
        drop(w);
        let bytes = read_file(&path);
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.records.len(), 3);
        assert_eq!(contents.records[2].payload, b"again");
        assert_eq!(contents.damage, None);
    }

    /// A truncation cuts the body and restates the header at the next
    /// sequence, and a journal read after it holds no records and expects
    /// the next record at that sequence.
    #[test]
    fn a_truncation_restates_the_first_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.zdbwal");
        let mut w = JournalWriter::create(&path, 9, 1).unwrap();
        for i in 0..5u8 {
            w.append(OperationKind::Insert, &[i; 30]).unwrap();
        }
        w.sync().unwrap();
        assert_eq!(w.sequence_reached(), 5);
        w.truncate().unwrap();
        assert_eq!(w.file_len().unwrap(), 64);
        assert_eq!(w.first_sequence(), 6);
        assert_eq!(w.next_sequence(), 6);
        assert_eq!(w.sequence_reached(), 5);
        w.append(OperationKind::Compact, &[]).unwrap();
        w.commit(CommitMode::Sync).unwrap();
        drop(w);
        let bytes = read_file(&path);
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.header.first_sequence, 6);
        assert_eq!(contents.records.len(), 1);
        assert_eq!(contents.records[0].sequence, 6);
        assert_eq!(contents.damage, None);
    }

    /// A crash between the truncation's two steps leaves an empty body
    /// under a header naming the old first sequence, which the reader
    /// accepts as a journal with no records, and the next open for append
    /// completes the second step so the record then appended is read back.
    #[test]
    fn a_crash_between_the_truncation_steps_is_recoverable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.zdbwal");
        let mut w = JournalWriter::create(&path, 9, 1).unwrap();
        for i in 0..7u8 {
            w.append(OperationKind::Insert, &[i; 30]).unwrap();
        }
        w.sync().unwrap();
        // The first step alone, then the process is gone.
        w.cut_body().unwrap();
        drop(w);
        let bytes = read_file(&path);
        assert_eq!(bytes.len(), 64);
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.header.first_sequence, 1);
        assert!(contents.records.is_empty());
        assert_eq!(contents.damage, None);
        assert_eq!(contents.next_sequence(), 1);
        // The checkpoint holds 7, so the next record is 8, and the header
        // is restated to say so before anything is appended.
        let mut w = JournalWriter::open_for_append(&path, &contents, 8).unwrap();
        assert_eq!(w.first_sequence(), 8);
        assert_eq!(w.next_sequence(), 8);
        assert_eq!(w.sequence_reached(), 7);
        w.append(OperationKind::Insert, b"eighth").unwrap();
        w.commit(CommitMode::Sync).unwrap();
        drop(w);
        let bytes = read_file(&path);
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.header.first_sequence, 8);
        assert_eq!(contents.records.len(), 1);
        assert_eq!(contents.records[0].sequence, 8);
        assert_eq!(contents.damage, None);
        // A header restated before the first step had run would be wrong,
        // which is why the body is cut first: a reader between the steps
        // sees records at or below the checkpoint under the old first
        // sequence, and skips them by sequence.
        let mut w = JournalWriter::open_for_append(&path, &contents, 0).unwrap();
        assert_eq!(w.next_sequence(), 9);
        w.append(OperationKind::Clear, &[]).unwrap();
        drop(w);
        let bytes = read_file(&path);
        assert_eq!(read_journal(&bytes, "t").unwrap().records.len(), 2);
        // An empty body with a zero next sequence is refused.
        let header = encode_journal_header(9, 4);
        std::fs::write(&path, header).unwrap();
        let bytes = read_file(&path);
        let contents = read_journal(&bytes, "t").unwrap();
        assert!(matches!(
            JournalWriter::open_for_append(&path, &contents, 0),
            Err(Error::JournalIoFailed { what: "open", .. })
        ));
    }

    /// A repair cuts a corrupt middle where the reader said, and what
    /// follows the cut is gone.
    #[test]
    fn a_repair_cuts_at_the_corrupt_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.zdbwal");
        let mut w = JournalWriter::create(&path, 9, 1).unwrap();
        for i in 0..5u8 {
            w.append(OperationKind::Insert, &[i; 40]).unwrap();
        }
        w.sync().unwrap();
        drop(w);
        let mut bytes = read_file(&path);
        let third = 64 + 2 * (JOURNAL_RECORD_OVERHEAD_BYTES + 40);
        bytes[third + 16 + 12] ^= 0x5A;
        std::fs::write(&path, &bytes).unwrap();
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert_eq!(contents.good_bytes, third as u64);
        assert!(contents.refusal("t").is_some());
        let w = JournalWriter::open_for_append(&path, &contents, 1).unwrap();
        assert_eq!(w.file_len().unwrap(), third as u64);
        assert_eq!(w.next_sequence(), 3);
        drop(w);
        let bytes = read_file(&path);
        let contents = read_journal(&bytes, "t").unwrap();
        assert_eq!(contents.records.len(), 2);
        assert_eq!(contents.damage, None);
    }

    // ------------------------------------------------------------------
    // The fuzzer
    // ------------------------------------------------------------------

    fn budget(default: usize) -> usize {
        std::env::var("ZEUSDB_FUZZ_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// The header fields a mutation lands on, as offset and width.
    const HEADER_FIELDS: [(usize, usize); 7] = [
        (8, 4),
        (12, 4),
        (16, 1),
        (17, 7),
        (24, 16),
        (40, 8),
        (48, 8),
    ];

    /// Apply one to three mutations to a valid journal, at hostile values
    /// and random widths, over the header's fields, a record's length,
    /// kind, sequence, flags, checksum and payload, and the file's length.
    /// `body_only` keeps the header whole except for its first sequence,
    /// so the classification can be judged against the record table.
    fn mutate(
        rng: &mut Rng,
        base: &[u8],
        table: &[Entry],
        body_only: bool,
    ) -> (Vec<u8>, Vec<Whole>) {
        let mut blob = base.to_vec();
        let mut wholes = Vec::new();
        let ops = 1 + rng.below(3);
        for _ in 0..ops {
            let len = blob.len();
            let body = len.saturating_sub(JOURNAL_HEADER_BYTES);
            match rng.below(12) {
                0..=2 => {
                    // A hostile value at a random offset in the body, at a
                    // random width, not repaired.
                    if body > 0 {
                        let width = [1usize, 2, 4, 8][rng.below(4)];
                        let at = JOURNAL_HEADER_BYTES + rng.below(body);
                        let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                        let end = (at + width).min(len);
                        blob[at..end].copy_from_slice(&value[..end - at]);
                    }
                }
                3 => {
                    // One body byte set at random.
                    if body > 0 {
                        let at = JOURNAL_HEADER_BYTES + rng.below(body);
                        blob[at] = rng.byte();
                    }
                }
                4 => {
                    // The file cut short, anywhere.
                    blob.truncate(rng.below(len + 1));
                }
                5 => {
                    // Bytes appended, zeros half the time, which is what a
                    // filesystem extending the file leaves.
                    let extra = 1 + rng.below(64);
                    let value = if rng.below(2) == 0 { 0 } else { rng.byte() };
                    blob.extend(std::iter::repeat_n(value, extra));
                }
                6 => {
                    // A header field at a hostile value, restamped seven
                    // times in eight so the field's own check is reached.
                    if len >= JOURNAL_HEADER_BYTES {
                        let (at, width) = if body_only {
                            (40, 8)
                        } else {
                            HEADER_FIELDS[rng.below(HEADER_FIELDS.len())]
                        };
                        let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                        let take = width.min(8);
                        blob[at..at + take].copy_from_slice(&value[..take]);
                        if body_only || rng.below(8) != 0 {
                            restamp_header(&mut blob);
                        }
                    }
                }
                7 => {
                    // A record's sequence at a hostile value, restamped, so
                    // the record is whole at that sequence.
                    let entry = &table[rng.below(table.len())];
                    if end_of(entry) <= len {
                        let value = HOSTILE[rng.below(HOSTILE.len())];
                        blob[entry.0..entry.0 + 8].copy_from_slice(&value.to_le_bytes());
                        restamp_and_note(&mut blob, entry, &mut wholes);
                    }
                }
                8 => {
                    // A record's length at a hostile value, restamped over
                    // the original extent.
                    let entry = &table[rng.below(table.len())];
                    if end_of(entry) <= len {
                        let value = HOSTILE[rng.below(HOSTILE.len())] as u32;
                        blob[entry.0 + 8..entry.0 + 12].copy_from_slice(&value.to_le_bytes());
                        restamp_and_note(&mut blob, entry, &mut wholes);
                    }
                }
                9 => {
                    // A record's kind or flags at a value the reader must
                    // refuse, restamped. Under `body_only` the kind stays
                    // off the ten so the mutator knows the record is not
                    // whole.
                    let entry = &table[rng.below(table.len())];
                    if end_of(entry) <= len {
                        if rng.below(2) == 0 {
                            let value = if body_only {
                                [0u16, 11, 255, u16::MAX][rng.below(4)]
                            } else {
                                HOSTILE[rng.below(HOSTILE.len())] as u16
                            };
                            blob[entry.0 + 12..entry.0 + 14].copy_from_slice(&value.to_le_bytes());
                        } else {
                            let value = 1 + rng.below(u16::MAX as usize) as u16;
                            blob[entry.0 + 14..entry.0 + 16].copy_from_slice(&value.to_le_bytes());
                        }
                        restamp_and_note(&mut blob, entry, &mut wholes);
                    }
                }
                10 => {
                    // A record appended again at the end, whole, from the
                    // original bytes.
                    let entry = &table[rng.below(table.len())];
                    if end_of(entry) <= len {
                        wholes.push(Whole {
                            at: blob.len(),
                            bytes: base[entry.0..end_of(entry)].to_vec(),
                            sequence: entry.2,
                        });
                        blob.extend_from_slice(&base[entry.0..end_of(entry)]);
                    }
                }
                _ => {
                    // A run of the body zeroed.
                    if body > 0 {
                        let at = JOURNAL_HEADER_BYTES + rng.below(body);
                        let run = 1 + rng.below(48);
                        let end = (at + run).min(len);
                        blob[at..end].fill(0);
                    }
                }
            }
        }
        (blob, wholes)
    }

    /// The byte ranges where two files differ, and their lengths, for a
    /// failure message.
    fn differences(a: &[u8], b: &[u8]) -> String {
        let mut ranges = Vec::new();
        let mut open: Option<usize> = None;
        let n = a.len().max(b.len());
        for i in 0..=n {
            let same = i < a.len() && i < b.len() && a[i] == b[i];
            match (open, same) {
                (None, false) if i < n => open = Some(i),
                (Some(start), true) => {
                    ranges.push((start, i));
                    open = None;
                }
                (Some(start), false) if i == n => ranges.push((start, i)),
                _ => {}
            }
        }
        format!(
            "lengths {} and {}, differing at {:?}",
            a.len(),
            b.len(),
            ranges
        )
    }

    /// A record the mutator knows is whole, being its bytes at `at` and
    /// the sequence its header names: a copy appended at the end, or a
    /// record restamped after its header was changed.
    #[derive(Clone)]
    struct Whole {
        at: usize,
        bytes: Vec<u8>,
        sequence: u64,
    }

    /// Restamp the record at `entry` and, when its header still reads as a
    /// record, being the original length, a known kind and no flags, note
    /// it as whole at whatever sequence it now names.
    fn restamp_and_note(blob: &mut [u8], entry: &Entry, wholes: &mut Vec<Whole>) {
        restamp_record(blob, entry.0, entry.1);
        let at = entry.0;
        let len = u32::from_le_bytes(take4(blob, at + 8)) as usize;
        let kind = u16::from_le_bytes(take2(blob, at + 12));
        let flags = u16::from_le_bytes(take2(blob, at + 14));
        if len == entry.1 && OperationKind::from_code(kind).is_some() && flags == 0 {
            wholes.push(Whole {
                at,
                bytes: blob[at..end_of(entry)].to_vec(),
                sequence: u64::from_le_bytes(take8(blob, at)),
            });
        }
    }

    /// The classification the mutator's own knowledge implies, derived by
    /// byte comparison against the records it knows are whole rather than
    /// by parsing, so it is an oracle independent of the reader. The
    /// accepted records are the chain of whole records from the header
    /// whose sequences follow; the first byte with no such record fails,
    /// and it is corrupt when a whole record at a later sequence sits
    /// anywhere after it.
    fn oracle(
        base: &[u8],
        blob: &[u8],
        table: &[Entry],
        wholes: &[Whole],
    ) -> Result<(usize, Option<JournalDamage>, u64), ()> {
        if blob.len() < JOURNAL_HEADER_BYTES {
            return Err(());
        }
        // The header is refused unless it is whole, which means every
        // field the reader holds is the original's and the checksum holds,
        // since a cut below the header and an append after it rebuild the
        // header from garbage. The collection id is not held by the reader,
        // and under `body_only` the first sequence is what the mutator
        // changes.
        let stored = u64::from_le_bytes(take8(blob, 56));
        let first = u64::from_le_bytes(take8(blob, 40));
        if blob[..24] != base[..24]
            || blob[48..56] != base[48..56]
            || stored != checksum_of(&blob[..HEADER_CHECKSUM_BYTES])
            || first == 0
        {
            return Err(());
        }
        let mut all: Vec<Whole> = table
            .iter()
            .map(|e| Whole {
                at: e.0,
                bytes: base[e.0..end_of(e)].to_vec(),
                sequence: e.2,
            })
            .collect();
        all.extend(wholes.iter().cloned());
        let whole = |w: &Whole| {
            blob.len() >= w.at + w.bytes.len() && blob[w.at..w.at + w.bytes.len()] == w.bytes[..]
        };
        let mut expected = first;
        let mut accepted = 0usize;
        let mut at = JOURNAL_HEADER_BYTES;
        while let Some(w) = all
            .iter()
            .find(|w| w.at == at && w.sequence == expected && whole(w))
        {
            accepted += 1;
            at += w.bytes.len();
            expected = w.sequence.wrapping_add(1);
        }
        // A file that ends where its last good record ends is whole,
        // however many records it once held.
        if blob.len() == at {
            return Ok((accepted, None, at as u64));
        }
        let later = all
            .iter()
            .any(|w| w.at > at && w.sequence > expected && whole(w));
        let damage = if later {
            JournalDamage::Corrupt {
                at: at as u64,
                sequence: expected,
                detail: String::new(),
            }
        } else {
            JournalDamage::TornTail {
                at: at as u64,
                sequence: expected,
            }
        };
        Ok((accepted, Some(damage), at as u64))
    }

    /// **The property.** Every mutation of a valid journal reads or is
    /// refused at the header, none panics, every refusal is the reader's
    /// own error, and a useful share of cases reach past the header.
    #[test]
    fn no_mutation_of_a_valid_journal_panics_the_reader() {
        let (good, table) = corpus(12);
        assert_eq!(read_journal(&good, "t").unwrap().records.len(), 12);
        let mut rng = Rng(0x5eed_1046_e000_0146);
        let cases = budget(4_000);
        let mut past_header = 0usize;
        let mut whole = 0usize;
        let mut torn = 0usize;
        let mut corrupt = 0usize;
        for _ in 0..cases {
            let (blob, _) = mutate(&mut rng, &good, &table, false);
            match read_journal(&blob, "t") {
                Ok(contents) => {
                    past_header += 1;
                    assert!(contents.good_bytes <= blob.len() as u64);
                    assert!(contents.good_bytes >= JOURNAL_HEADER_BYTES as u64);
                    match contents.damage {
                        None => {
                            whole += 1;
                            assert_eq!(contents.good_bytes, blob.len() as u64);
                        }
                        Some(JournalDamage::TornTail { at, .. }) => {
                            torn += 1;
                            assert_eq!(at, contents.good_bytes);
                        }
                        Some(JournalDamage::Corrupt { at, .. }) => {
                            corrupt += 1;
                            assert_eq!(at, contents.good_bytes);
                        }
                    }
                }
                Err(Error::JournalHeaderInvalid { .. }) => {}
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
        assert!(
            past_header * 2 > cases,
            "only {past_header} of {cases} mutations reached past the header"
        );
        assert!(
            whole > 0 && torn > 0 && corrupt > 0,
            "{whole} {torn} {corrupt}"
        );
    }

    /// **The classification.** Every mutation of the body is classified as
    /// the record table says it should be, with the same accepted count,
    /// the same byte and the same sequence, so a mutation that turns a
    /// torn tail into a corrupt middle, or the reverse, is caught.
    #[test]
    fn the_classification_matches_an_oracle_under_mutation() {
        let (good, table) = corpus(12);
        let mut rng = Rng(0x5eed_1046_c000_0146);
        let cases = budget(4_000);
        let mut torn = 0usize;
        let mut corrupt = 0usize;
        let mut whole = 0usize;
        for case in 0..cases {
            let (blob, wholes) = mutate(&mut rng, &good, &table, true);
            let expected = oracle(&good, &blob, &table, &wholes);
            match (read_journal(&blob, "t"), expected) {
                (Err(Error::JournalHeaderInvalid { .. }), Err(())) => {}
                (Ok(contents), Ok((accepted, damage, good_bytes))) => {
                    assert_eq!(contents.records.len(), accepted, "case {case}");
                    assert_eq!(contents.good_bytes, good_bytes, "case {case}");
                    match (&contents.damage, &damage) {
                        (None, None) => whole += 1,
                        (
                            Some(JournalDamage::TornTail { at, sequence }),
                            Some(JournalDamage::TornTail {
                                at: want_at,
                                sequence: want_sequence,
                            }),
                        ) => {
                            torn += 1;
                            assert_eq!((at, sequence), (want_at, want_sequence), "case {case}");
                        }
                        (
                            Some(JournalDamage::Corrupt { at, sequence, .. }),
                            Some(JournalDamage::Corrupt {
                                at: want_at,
                                sequence: want_sequence,
                                ..
                            }),
                        ) => {
                            corrupt += 1;
                            assert_eq!((at, sequence), (want_at, want_sequence), "case {case}");
                        }
                        (found, want) => panic!(
                            "case {case}: the reader said {found:?} and the oracle {want:?}; {}",
                            differences(&good, &blob)
                        ),
                    }
                }
                (found, want) => panic!(
                    "case {case}: the reader said {:?} and the oracle {want:?}",
                    found.map(|c| c.damage)
                ),
            }
        }
        assert!(
            torn * 20 > cases && corrupt * 20 > cases && whole > 0,
            "{torn} torn, {corrupt} corrupt, {whole} whole of {cases}"
        );
    }
}
