//! The write-ahead journal: its records, its segments, the readers that tell
//! a torn tail from a corrupt middle, and the writer.
//!
//! A journal of format version 2 is a directory beside an index directory,
//! holding segments, each a file named by the sequence of its first record.
//! A segment is a 128 byte header naming the checkpoint its records follow,
//! then records, each a 16 byte header, a payload and an 8 byte checksum over
//! both. Every call's records end in a `Commit` record naming the first of
//! them, and a call whose commit failed is withdrawn by an `Abort` record.
//! This module knows what a record is and how the two framing kinds settle
//! the records before them, and nothing of what an operation's payload
//! means. The payloads are [`crate::operation`]'s, and applying one is the
//! index's.
//!
//! A journal of format version 1 is one file, a 64 byte header and records
//! with no framing, which releases from 0.11.0 to 0.13.0 wrote.
//! [`read_legacy_journal`] reads one, so that its directory can be replayed
//! and upgraded. Nothing writes one.
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
//! The first seventeen bytes of both headers mean what the artefact frame's
//! and the graph dump's mean, being an eight byte magic, a `u32` format
//! version, a `u32` of reserved flags and a kind byte, so one reader
//! dispatches on the magic at offset zero and reads any of the three.
//!
//! # Layout
//!
//! ```text
//! Segment header, 128 bytes, at format version 2
//!    0   8  magic            u64   b"ZDBJOURN"
//!    8   4  format_version   u32   2
//!   12   4  flags            u32   reserved, zero
//!   16   1  kind             u8    JournalKind
//!   17   7  reserved         zero
//!   24  16  collection_id    u128  the collection the journal belongs to
//!   40  16  base_snapshot    u128  the checkpoint the segment's records follow
//!   56   8  base_sequence    u64   that checkpoint's sequence
//!   64   8  base_generation  u64   that checkpoint's generation
//!   72  16  base_parent      u128  that checkpoint's parent, or zero
//!   88   8  first_sequence   u64   the first record's, being the number the
//!                                   file is named for, above base_sequence
//!   96  24  reserved         zero
//!  120   8  header_checksum  u64   over bytes 0 to 120
//!
//! File header, 64 bytes, at format version 1
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
//! Record, at both versions
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
//! Nor is `u64::MAX`, since no sequence follows it: the writer refuses that
//! append and the reader fails a record that carries it.
//! Sequences run on across segments and across checkpoints, and restart
//! only where a journal is opened afresh. The framing records take
//! sequences like any other.
//!
//! **The kind numbers are on disk for ever.** See [`OperationKind`] and
//! [`JournalKind`].
//!
//! # What bounds each length
//!
//! Every length is checked against what the file has earned before anything
//! is sized from it, which is the rule the graph dump and the artefact frame
//! follow. A file holds at least its header before any of it is read.
//! `payload_len` is held to the bytes remaining after the record header and
//! before an eight byte checksum, and to [`JOURNAL_MAX_PAYLOAD`], before the
//! payload is touched, so a header claiming a gigabyte on a kilobyte file
//! allocates nothing. `sequence` is held to the previous record's plus one
//! and below `u64::MAX`, `kind` to the kinds the format version knows,
//! `flags` to zero, and the checksum is recomputed over the header and the
//! payload. What is inside the payload is bounded by `payload_len` and is
//! the payload reader's business, the two framing payloads included.
//!
//! # Calls, commits and withdrawals
//!
//! A boundary is a commit record or an abort record, and before the first
//! of them the checkpoint's sequence. A call's records are appended one
//! write each, then a `Commit` naming the first sequence after the last
//! boundary. Under the per-call policy the commit is synced before the call
//! returns. A sync that fails is followed by an `Abort` naming the same
//! first sequence, which withdraws every record from it to the abort's own
//! sequence less one, the commit record among them. Where the abort cannot
//! be written or synced, the segment is cut back to the boundary and the cut
//! synced. Where that fails too, the writer takes nothing more until a
//! checkpoint rolls it onto a new segment.
//!
//! A commit is settled by the next record when that record is not an abort
//! naming the commit's first sequence, and by the end of the journal. Only a
//! settled commit's records replay. A withdrawn insert and a withdrawn
//! interning still matter to a replay, which the index decides; this module
//! says which records are withdrawn.
//!
//! # What the reader of a journal of segments holds it to
//!
//! It is handed every segment in name order and the checkpoint the
//! directory is, and holds the journal to these rules in this order.
//!
//! 1. Every segment's header reads, names the sequence its file is named
//!    for, and names the directory's collection. The last segment alone may
//!    lack a whole header where nothing follows its header's length, which
//!    is a creation a crash interrupted, and it is created again.
//! 2. A segment naming another checkpoint is stale when it holds no record
//!    above the directory's sequence, and its records are skipped, since the
//!    checkpoint holds them. One holding such a record belongs to another
//!    history and is refused, naming both checkpoints.
//! 3. The segments naming the directory's checkpoint follow every stale one,
//!    the first begins at the sequence after the checkpoint's, and each
//!    begins after the last record of the one before, with no gap and no
//!    overlap. An empty one is the last.
//! 4. Every record parses whole at the next sequence. Damage in any segment
//!    but the last is a corrupt middle, since the writer syncs a segment
//!    before it starts the next.
//! 5. A commit names the sequence after the last boundary. An abort names
//!    that sequence, or the first of a commit it directly follows. Either
//!    covers at least one record. Anything else is refused.
//! 6. In the last segment, damage followed by a whole abort, at a later
//!    sequence, that names the sequence after the last boundary or the first
//!    of the commit just before the damage, is a withdrawn region, and the
//!    reading goes on after the abort. Every byte from that sequence to the
//!    abort is withdrawn, whole records among them.
//! 7. Otherwise damage followed by a whole record at a later sequence is a
//!    corrupt middle, and damage with nothing whole after it is a torn tail.
//!    Records after the last boundary belong to the torn tail. A torn tail is
//!    cut at its first byte.
//!
//! # A reading for a repair
//!
//! [`read_segments_for_repair`] reads the same way, and refuses a journal
//! that is not the directory's as [`read_segments`] does. Every other break
//! of rules 1 to 7, and a corrupt middle, stops the reading and names a
//! [`Cut`] at the first record of the call the break touches. The records of
//! every call before the cut are handed over, and the journal cut there
//! reads under the seven rules with no damage.
//!
//! A segment named at or below the directory's sequence holds no record
//! above it, so its name makes it stale whatever its header holds. Where its
//! header does not read or names another sequence than its file, a reading
//! that is not for a repair refuses the journal and names the repair, and a
//! reading for a repair sets the segment aside whole, lists it in
//! [`JournalContents::stale_set_aside`] and reads every other segment.
//!
//! # The writer
//!
//! Appends one record in one write, and cuts the segment back where a write
//! fails, so a failed append leaves nothing a reader could take for a record.
//! Commits a call under the mode the caller chooses at each call, and
//! withdraws a call whose commit failed. Rolls onto a new segment when the
//! active one passes [`SEGMENT_ROLL_BYTES`] at a commit, and onto a segment
//! naming a new checkpoint when a checkpoint asks, deleting every segment
//! that checkpoint holds. It refuses every operation in a process other than
//! the one that opened it, which is a child forked from it, since the child
//! shares the parent's open file. It takes no index type, holds no index lock
//! and knows nothing of a collection, and it reaches its directory through
//! the [`JournalDir`] it is handed.

use crate::checksum::checksum_of;
use crate::error::Error;
use crate::operation::{AbortReason, Operation};
use crate::storage::{JournalDir, JournalFile};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::warn;

/// The target the records this module emits carry. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::journal";

/// `b"ZDBJOURN"` read little-endian.
pub const JOURNAL_MAGIC: u64 = u64::from_le_bytes(*b"ZDBJOURN");

/// The journal version this build writes, being a directory of segments.
const JOURNAL_VERSION: u32 = 2;

/// The journal version releases from 0.11.0 to 0.13.0 wrote, being one file,
/// which this build reads and does not write.
const LEGACY_JOURNAL_VERSION: u32 = 1;

/// Bytes a segment's header occupies, checksum included.
pub const SEGMENT_HEADER_BYTES: usize = 128;

/// Bytes a segment's header checksum is taken over.
const SEGMENT_CHECKSUM_BYTES: usize = 120;

/// Bytes the header of a journal of format version 1 occupies.
pub const LEGACY_HEADER_BYTES: usize = 64;

/// Bytes that header's checksum is taken over.
const LEGACY_CHECKSUM_BYTES: usize = 56;

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
pub(crate) const MOST_CENTROIDS: usize = 256;

/// The widest payload a record may carry, and the ceiling every
/// `payload_len` is held to before the payload is touched.
///
/// Derived rather than chosen. A vector at the widest declaration is
/// 262,144 bytes. A codebook at that width and the most centroids, which a
/// `Train` record carries, is the width times the centroids times four bytes
/// a value, 64 MiB, whatever the subvector count. A mebibyte of metadata, or
/// of a training record's stamp, counts and calibration, sits beside either
/// with margin. It bounds a hostile length that the remaining-bytes check
/// alone would admit on a large file.
pub const JOURNAL_MAX_PAYLOAD: usize = WIDEST_VECTOR * MOST_CENTROIDS * 4 + (1 << 20);

/// The length past which the active segment is sealed at a commit and the
/// next one started. A call's records all land in one segment, so a segment
/// may pass it by one call, and a record may be up to
/// [`JOURNAL_MAX_PAYLOAD`] on its own.
pub const SEGMENT_ROLL_BYTES: u64 = 64 << 20;

#[cfg(debug_assertions)]
thread_local! {
    static ROLL_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Set the length at which writers made on this thread roll, for a test
/// that has to reach a roll without writing 64 MiB. Zero restores
/// [`SEGMENT_ROLL_BYTES`]. Debug builds alone, since it is the tests'.
#[cfg(debug_assertions)]
pub fn set_segment_roll_bytes(bytes: u64) {
    ROLL_BYTES.with(|cell| cell.set(bytes));
}

/// The length at which a writer made on this thread rolls.
fn roll_bytes() -> u64 {
    #[cfg(debug_assertions)]
    {
        let set = ROLL_BYTES.with(std::cell::Cell::get);
        if set != 0 {
            return set;
        }
    }
    SEGMENT_ROLL_BYTES
}

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
/// one.** A reader refuses a kind its format version does not know, so a
/// new kind is a new format version or a feature of the directory.
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
    /// The records of one call, closed. Framing, at format version 2.
    Commit = 11,
    /// The records of one call whose commit failed, withdrawn. Framing, at
    /// format version 2.
    Abort = 12,
}

impl OperationKind {
    /// Every kind that is an operation, in the order of its number. A
    /// journal of format version 1 holds these and no other.
    pub const OPERATIONS: [OperationKind; 10] = [
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

    /// Every kind, in the order of its number, the two framing kinds last.
    pub const ALL: [OperationKind; 12] = [
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
        OperationKind::Commit,
        OperationKind::Abort,
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
            11 => Some(OperationKind::Commit),
            12 => Some(OperationKind::Abort),
            _ => None,
        }
    }

    /// Whether the kind is one of the two the writer frames calls with,
    /// which no collection applies.
    pub fn is_framing(self) -> bool {
        matches!(self, OperationKind::Commit | OperationKind::Abort)
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
            OperationKind::Commit => "commit",
            OperationKind::Abort => "abort",
        }
    }
}

/// One record, borrowing its payload from the bytes it was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalRecord<'a> {
    pub sequence: u64,
    pub kind: OperationKind,
    /// Byte offset of the record's first byte in its file.
    pub offset: u64,
    pub payload: &'a [u8],
}

/// What stopped a reader, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JournalDamage {
    /// The file ends inside a record, or a record fails and nothing after
    /// it parses, or records follow the last boundary. `at` is the byte the
    /// file is cut at, and `sequence` the one the record at `at` was
    /// expected to carry.
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

// ============================================================================
// RECORDS, AT BOTH VERSIONS
// ============================================================================

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

/// What parsing one record at an offset found.
enum Parsed<'a> {
    /// A whole record at the expected sequence.
    Whole(JournalRecord<'a>),
    /// The file ends inside the record, or before its header is whole.
    Incomplete,
    /// The record is whole and wrong, with how.
    Failed(String),
    /// The record is whole, carries the expected sequence and passes its
    /// checksum, and names a kind this build does not know.
    UnknownKind(u16),
}

/// Parse one record at `at`, expecting `expected` as its sequence. The
/// framing kinds are known where `framing` says so, which is at format
/// version 2. Every length is held before anything is read from it, and
/// nothing allocates.
fn parse_record(bytes: &[u8], at: usize, expected: u64, framing: bool) -> Parsed<'_> {
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
    if sequence == u64::MAX {
        return Parsed::Failed(format!(
            "sequence {sequence} is never a record's, since no sequence follows it"
        ));
    }
    if sequence != expected {
        return Parsed::Failed(format!("sequence {sequence} where {expected} was expected"));
    }
    let Some(kind) = OperationKind::from_code(kind).filter(|k| framing || !k.is_framing()) else {
        return Parsed::UnknownKind(kind);
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

/// The kind the record at `at` names, where the record is whole, carries
/// `expected` as its sequence, passes its checksum and names a kind this
/// build does not know. `None` for any other record, damaged or whole.
/// `framing` says the bytes are a segment rather than a journal of format
/// version 1.
///
/// A directory holding a `read_only` feature this build does not know reads
/// such a record as the feature's own rather than as damage.
pub fn unknown_kind_at(bytes: &[u8], at: u64, expected: u64, framing: bool) -> Option<u16> {
    match parse_record(bytes, usize::try_from(at).ok()?, expected, framing) {
        Parsed::UnknownKind(kind) => Some(kind),
        _ => None,
    }
}

/// The bytes a record occupies from its first.
fn record_len(record: &JournalRecord<'_>) -> usize {
    JOURNAL_RECORD_OVERHEAD_BYTES + record.payload.len()
}

/// Whether a whole record at a sequence above `expected` sits anywhere
/// from `from` on. This is what separates a corrupt middle from a torn
/// tail: a torn tail is followed by nothing that parses. The sequence is
/// compared before anything else is read, so a run of zeros or of ones
/// costs one comparison a byte.
fn a_later_record_parses(bytes: &[u8], from: usize, expected: u64, framing: bool) -> bool {
    let mut at = from;
    while at + JOURNAL_RECORD_OVERHEAD_BYTES <= bytes.len() {
        let sequence = u64::from_le_bytes(take8(bytes, at));
        if sequence > expected
            && matches!(parse_record(bytes, at, sequence, framing), Parsed::Whole(_))
        {
            return true;
        }
        at += 1;
    }
    false
}

/// The whole operation records in `bytes` from `from` on, the first of them
/// expected at `first`, counted as the reader's scans count them: the run of
/// whole records from `from`, then, past each record that fails, the run
/// from the next whole record at a later sequence. What a repair reports it
/// set aside. Framing records are not counted. `framing` says the bytes are
/// a journal of segments, whose framing kinds parse, rather than a journal
/// of format version 1, where they do not.
pub fn whole_records_from(bytes: &[u8], from: usize, first: u64, framing: bool) -> usize {
    let mut count = 0usize;
    let mut at = from;
    let mut expected = first;
    loop {
        let (records, failure) = scan(bytes, at, expected, framing);
        count += records.iter().filter(|r| !r.kind.is_framing()).count();
        let Some(failure) = failure else {
            return count;
        };
        let mut probe = failure.at + 1;
        let mut next = None;
        while probe + JOURNAL_RECORD_OVERHEAD_BYTES <= bytes.len() {
            let sequence = u64::from_le_bytes(take8(bytes, probe));
            if sequence > failure.expected
                && matches!(
                    parse_record(bytes, probe, sequence, framing),
                    Parsed::Whole(_)
                )
            {
                next = Some((probe, sequence));
                break;
            }
            probe += 1;
        }
        match next {
            Some((resume, sequence)) => {
                at = resume;
                expected = sequence;
            }
            None => return count,
        }
    }
}

/// Where a run of records stopped.
#[derive(Clone, Debug)]
struct Failure {
    at: usize,
    expected: u64,
    detail: String,
}

/// Every whole record from `from` on, at sequences from `first`, and what
/// stopped the run where something did.
fn scan(
    bytes: &[u8],
    from: usize,
    first: u64,
    framing: bool,
) -> (Vec<JournalRecord<'_>>, Option<Failure>) {
    let mut records = Vec::new();
    let mut at = from;
    let mut expected = first;
    while at < bytes.len() {
        let detail = match parse_record(bytes, at, expected, framing) {
            Parsed::Whole(record) => {
                at += record_len(&record);
                expected = record.sequence.wrapping_add(1);
                records.push(record);
                continue;
            }
            Parsed::Incomplete => "the record runs past the end of the file".to_string(),
            Parsed::Failed(detail) => detail,
            Parsed::UnknownKind(kind) => format!("kind {kind} is not one this build knows"),
        };
        return (
            records,
            Some(Failure {
                at,
                expected,
                detail,
            }),
        );
    }
    (records, None)
}

// ============================================================================
// THE JOURNAL OF FORMAT VERSION 1
// ============================================================================

/// The header of a journal of format version 1, read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyHeader {
    /// The collection the journal belongs to.
    pub collection_id: u128,
    /// The sequence of the first record in the file. After a truncation it
    /// is the checkpoint's sequence plus one, which may be above the last
    /// record's when the body is empty.
    pub first_sequence: u64,
}

/// A journal of format version 1 read back, borrowing every payload from
/// the bytes it was read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyContents<'a> {
    pub header: LegacyHeader,
    /// Every record that parsed in sequence, in file order.
    pub records: Vec<JournalRecord<'a>>,
    /// What stopped the read, or nothing when the file ends at a record
    /// boundary.
    pub damage: Option<JournalDamage>,
    /// The byte the good records end at. The file's length when nothing is
    /// damaged, and the damage's `at` otherwise.
    pub good_bytes: u64,
}

impl LegacyContents<'_> {
    /// The sequence after the last good record, or the header's first when
    /// the body is empty.
    pub fn next_sequence(&self) -> u64 {
        self.records
            .last()
            .map_or(self.header.first_sequence, |r| r.sequence + 1)
    }

    /// The refusal a corrupt middle earns, naming the file, the sequence
    /// and the byte. A journal of format version 1 frames no call, so a
    /// repair cuts at the damaged record itself. None for a torn tail or an
    /// undamaged file.
    pub fn refusal(&self, file: &str) -> Option<Error> {
        corrupt_refusal(self.damage.as_ref(), file, None)
    }
}

/// The refusal a corrupt middle earns, and nothing for any other damage.
/// `cut` is the sequence and the byte a repair cuts at, and the damaged
/// record's own where none is given.
fn corrupt_refusal(
    damage: Option<&JournalDamage>,
    file: &str,
    cut: Option<(u64, u64)>,
) -> Option<Error> {
    match damage {
        Some(JournalDamage::Corrupt {
            at,
            sequence,
            detail,
        }) => {
            let (cut_sequence, cut_at) = cut.unwrap_or((*sequence, *at));
            Some(Error::JournalCorrupt {
                file: file.to_string(),
                sequence: *sequence,
                at: *at,
                detail: detail.clone(),
                cut_sequence,
                cut_at,
            })
        }
        _ => None,
    }
}

/// The header of a journal of format version 1 for `collection_id`, with
/// its first record at `first_sequence`. For the tests, which build such
/// journals since nothing in this build writes one.
#[cfg(any(test, feature = "test-support"))]
pub fn encode_legacy_header(collection_id: u128, first_sequence: u64) -> [u8; LEGACY_HEADER_BYTES] {
    let mut out = [0u8; LEGACY_HEADER_BYTES];
    out[0..8].copy_from_slice(&JOURNAL_MAGIC.to_le_bytes());
    out[8..12].copy_from_slice(&LEGACY_JOURNAL_VERSION.to_le_bytes());
    out[16] = JournalKind::Operations.code();
    out[24..40].copy_from_slice(&collection_id.to_le_bytes());
    out[40..48].copy_from_slice(&first_sequence.to_le_bytes());
    let sum = checksum_of(&out[..LEGACY_CHECKSUM_BYTES]);
    out[56..64].copy_from_slice(&sum.to_le_bytes());
    out
}

/// Read the header of a journal of format version 1, refusing each way it
/// can be wrong by name, in the order length, magic, version, checksum,
/// reserved fields, kind, first sequence.
fn read_legacy_header(bytes: &[u8], file: &str) -> Result<LegacyHeader, Error> {
    let invalid = |detail: String| Error::JournalHeaderInvalid {
        file: file.to_string(),
        detail,
    };
    if bytes.len() < LEGACY_HEADER_BYTES {
        return Err(invalid(format!(
            "the file holds {} bytes and the header is {}",
            bytes.len(),
            LEGACY_HEADER_BYTES
        )));
    }
    let header = &bytes[..LEGACY_HEADER_BYTES];
    if u64::from_le_bytes(take8(header, 0)) != JOURNAL_MAGIC {
        return Err(invalid(
            "the file does not open with the journal magic".into(),
        ));
    }
    let version = u32::from_le_bytes(take4(header, 8));
    if version != LEGACY_JOURNAL_VERSION {
        return Err(invalid(format!(
            "the journal is format version {version} and a journal held in one file is \
             version {LEGACY_JOURNAL_VERSION}"
        )));
    }
    let stored = u64::from_le_bytes(take8(header, 56));
    if stored != checksum_of(&header[..LEGACY_CHECKSUM_BYTES]) {
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
    Ok(LegacyHeader {
        collection_id: u128::from_le_bytes(take16(header, 24)),
        first_sequence,
    })
}

/// Read a journal of format version 1 back, every record in sequence,
/// classifying whatever stops the read. `file` names the journal in a
/// message. The header is the only thing refused outright; damage inside
/// the body is reported in the contents and the caller decides. A framing
/// kind is not one this version knows.
pub fn read_legacy_journal<'a>(bytes: &'a [u8], file: &str) -> Result<LegacyContents<'a>, Error> {
    let header = read_legacy_header(bytes, file)?;
    let (records, failure) = scan(bytes, LEGACY_HEADER_BYTES, header.first_sequence, false);
    let damage = failure.map(|f| {
        if a_later_record_parses(bytes, f.at + 1, f.expected, false) {
            JournalDamage::Corrupt {
                at: f.at as u64,
                sequence: f.expected,
                detail: f.detail,
            }
        } else {
            JournalDamage::TornTail {
                at: f.at as u64,
                sequence: f.expected,
            }
        }
    });
    let good_bytes = match &damage {
        Some(JournalDamage::TornTail { at, .. }) | Some(JournalDamage::Corrupt { at, .. }) => *at,
        None => bytes.len() as u64,
    };
    Ok(LegacyContents {
        header,
        records,
        damage,
        good_bytes,
    })
}

// ============================================================================
// A SEGMENT'S HEADER
// ============================================================================

/// The checkpoint a segment's records follow.
///
/// A checkpoint is a committed directory: its snapshot, the journal sequence
/// its state holds, its generation and the snapshot it was saved from, zero
/// where there is none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentBase {
    pub snapshot: u128,
    pub sequence: u64,
    pub generation: u64,
    pub parent: u128,
}

/// A segment's header, read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentHeader {
    /// The collection the journal belongs to.
    pub collection_id: u128,
    /// The checkpoint the segment's records follow.
    pub base: SegmentBase,
    /// The sequence of the segment's first record, which its file is named
    /// for, above the base's sequence.
    pub first_sequence: u64,
}

/// A segment's header, laid out as the module says.
pub fn encode_segment_header(header: &SegmentHeader) -> [u8; SEGMENT_HEADER_BYTES] {
    let mut out = [0u8; SEGMENT_HEADER_BYTES];
    out[0..8].copy_from_slice(&JOURNAL_MAGIC.to_le_bytes());
    out[8..12].copy_from_slice(&JOURNAL_VERSION.to_le_bytes());
    out[16] = JournalKind::Operations.code();
    out[24..40].copy_from_slice(&header.collection_id.to_le_bytes());
    out[40..56].copy_from_slice(&header.base.snapshot.to_le_bytes());
    out[56..64].copy_from_slice(&header.base.sequence.to_le_bytes());
    out[64..72].copy_from_slice(&header.base.generation.to_le_bytes());
    out[72..88].copy_from_slice(&header.base.parent.to_le_bytes());
    out[88..96].copy_from_slice(&header.first_sequence.to_le_bytes());
    let sum = checksum_of(&out[..SEGMENT_CHECKSUM_BYTES]);
    out[120..128].copy_from_slice(&sum.to_le_bytes());
    out
}

/// Read a segment's header, refusing each way it can be wrong by name, in
/// the order length, magic, version, checksum, reserved fields, kind, first
/// sequence.
pub fn read_segment_header(bytes: &[u8], file: &str) -> Result<SegmentHeader, Error> {
    let invalid = |detail: String| Error::JournalHeaderInvalid {
        file: file.to_string(),
        detail,
    };
    if bytes.len() < SEGMENT_HEADER_BYTES {
        return Err(invalid(format!(
            "the file holds {} bytes and a segment's header is {}",
            bytes.len(),
            SEGMENT_HEADER_BYTES
        )));
    }
    let header = &bytes[..SEGMENT_HEADER_BYTES];
    if u64::from_le_bytes(take8(header, 0)) != JOURNAL_MAGIC {
        return Err(invalid(
            "the file does not open with the journal magic".into(),
        ));
    }
    let version = u32::from_le_bytes(take4(header, 8));
    if version != JOURNAL_VERSION {
        return Err(invalid(format!(
            "the segment is format version {version} and this build reads segments of \
             version {JOURNAL_VERSION}"
        )));
    }
    let stored = u64::from_le_bytes(take8(header, 120));
    if stored != checksum_of(&header[..SEGMENT_CHECKSUM_BYTES]) {
        return Err(invalid("the header is corrupt".into()));
    }
    let flags = u32::from_le_bytes(take4(header, 12));
    if flags != 0
        || header[17..24].iter().any(|&b| b != 0)
        || header[96..120].iter().any(|&b| b != 0)
    {
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
    let base = SegmentBase {
        snapshot: u128::from_le_bytes(take16(header, 40)),
        sequence: u64::from_le_bytes(take8(header, 56)),
        generation: u64::from_le_bytes(take8(header, 64)),
        parent: u128::from_le_bytes(take16(header, 72)),
    };
    let first_sequence = u64::from_le_bytes(take8(header, 88));
    if first_sequence <= base.sequence {
        return Err(invalid(format!(
            "the header names sequence {first_sequence} as its first record, at or below \
             the sequence {} of the checkpoint it follows",
            base.sequence
        )));
    }
    Ok(SegmentHeader {
        collection_id: u128::from_le_bytes(take16(header, 24)),
        base,
        first_sequence,
    })
}

// ============================================================================
// THE READER OF A JOURNAL OF SEGMENTS
// ============================================================================

/// One segment as storage handed it over.
#[derive(Clone, Copy, Debug)]
pub struct SegmentBytes<'a> {
    /// The number the segment's file is named for.
    pub number: u64,
    /// The segment's file, as a message names it.
    pub file: &'a str,
    pub bytes: &'a [u8],
}

/// What a journal must belong to: the directory's collection and the
/// checkpoint the directory is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expected {
    pub collection_id: u128,
    pub base: SegmentBase,
}

/// Whether a record above the checkpoint replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordState {
    /// Its call's commit settled, so it replays.
    Committed,
    /// Its call was withdrawn, so it does not replay as a mutation.
    Withdrawn,
}

/// One operation record above the checkpoint, and whether it replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRecord<'a> {
    pub record: JournalRecord<'a>,
    pub state: RecordState,
}

/// What the reader found in one segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentReport {
    /// The number the segment's file is named for.
    pub number: u64,
    /// The segment's length.
    pub bytes: u64,
    /// The segment names another checkpoint and holds nothing above the
    /// directory's sequence, so the directory holds it and it can go.
    pub stale: bool,
}

/// A journal of segments read back, borrowing every payload from the bytes
/// it was read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalContents<'a> {
    /// Every segment with a whole header, in order.
    pub segments: Vec<SegmentReport>,
    /// The last segment, where a crash interrupted its creation, being the
    /// number it is named for. It holds no record and is created again.
    pub torn_creation: Option<u64>,
    /// Every operation record above the checkpoint that the reader reached,
    /// in sequence order, with whether it replays. Framing records are not
    /// listed.
    pub records: Vec<LiveRecord<'a>>,
    /// The place in `records` of each call's first record, in order, so a
    /// record's call begins at the last place at or below its own.
    pub calls: Vec<usize>,
    /// Operation records in stale segments, which the checkpoint holds.
    pub skipped: usize,
    /// Abort records read in the segments naming the checkpoint.
    pub aborts: usize,
    /// What stopped the read, or nothing.
    pub damage: Option<JournalDamage>,
    /// The segment the damage is in, as its file is named, where there is
    /// damage.
    pub damage_file: Option<String>,
    /// The byte the last segment naming the checkpoint is cut at, and its
    /// length when nothing is cut. For a corrupt middle, the byte the damage
    /// begins at in the segment `damage_file` names.
    pub good_bytes: u64,
    /// The sequence the next record appended carries.
    pub next_sequence: u64,
    /// The last segment naming the checkpoint, the one appended to next,
    /// where one exists and a crash did not interrupt the creation of the
    /// one after it.
    pub active: Option<u64>,
    /// Where a repair cuts, where the journal holds a corrupt middle or, read
    /// for a repair, breaks its own history in any other way. `records` then
    /// holds only the records before the cut.
    pub cut: Option<Cut>,
    /// The segments named at or below the directory's sequence whose header
    /// does not read or names another sequence than its file, which a
    /// reading for a repair sets aside whole. A reading that is not for a
    /// repair refuses the first of them instead, so it lists none.
    pub stale_set_aside: Vec<StaleSegment>,
}

/// A segment named at or below the directory's sequence whose header the
/// reader cannot use. Every record it could hold is at or below the
/// checkpoint, so a repair sets it aside whole and cuts nothing for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaleSegment {
    /// The number its file is named for.
    pub number: u64,
    /// Its file, as a message names it.
    pub file: String,
    /// Why its header cannot be used.
    pub detail: String,
}

/// Where a repair cuts a journal of segments, and the damage that puts the
/// cut there.
///
/// The cut is at the first record of the call the damage touches, since a
/// call applies whole or not at all, and a call's records never span two
/// segments. Every byte from the cut on, and every segment after the one it
/// is in, is set aside.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cut {
    /// The segment the cut is in, by the number its file is named for.
    pub segment: u64,
    /// The byte of that segment the cut is at. Zero where the whole segment
    /// goes, being one whose header does not read or which breaks the run of
    /// segments. The header's length where its header stays and every record
    /// goes.
    pub at: u64,
    /// The sequence of the first record set aside.
    pub sequence: u64,
    /// What broke, being `corrupt` for a record that fails with a whole
    /// record after it or in a sealed segment, `header` for a segment whose
    /// header does not read or names another sequence than its file, and
    /// `structure` for segments or framing records that do not form one run
    /// of calls.
    pub kind: &'static str,
    /// The segment the damage is in, by number.
    pub damage_segment: u64,
    /// The byte of that segment the damage is at.
    pub damage_at: u64,
    /// The sequence the record at the damage was expected to carry.
    pub damage_sequence: u64,
    /// How the damage fails.
    pub detail: String,
}

impl JournalContents<'_> {
    /// Operation records whose commit settled.
    pub fn committed(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.state == RecordState::Committed)
            .count()
    }

    /// Operation records whose call was withdrawn.
    pub fn withdrawn(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.state == RecordState::Withdrawn)
            .count()
    }

    /// The segments naming the checkpoint.
    pub fn live(&self) -> impl Iterator<Item = &SegmentReport> {
        self.segments.iter().filter(|s| !s.stale)
    }

    /// The refusal a corrupt middle earns, naming the segment, the sequence
    /// and the byte, and the record and byte a repair cuts at. None for a
    /// torn tail or an undamaged journal.
    pub fn refusal(&self) -> Option<Error> {
        let cut = self.cut.as_ref().map(|cut| (cut.sequence, cut.at));
        corrupt_refusal(
            self.damage.as_ref(),
            self.damage_file.as_deref().unwrap_or_default(),
            cut,
        )
    }
}

/// The first sequence a framing record names, where its payload reads.
fn framing_first(record: &JournalRecord<'_>) -> Result<u64, String> {
    let decoded = Operation::decode(record, 0, "").map_err(|error| match error {
        Error::JournalRecordInvalid { detail, .. } => detail,
        other => other.to_string(),
    })?;
    match decoded {
        Operation::Commit { first } | Operation::Abort { first, .. } => Ok(first),
        _ => Err(format!("record {} is not framing", record.sequence)),
    }
}

/// A commit read and not yet settled, being the first sequence it names, its
/// own sequence, and the records it covers, as indexes into the reader's
/// list.
#[derive(Clone, Debug)]
struct Tentative {
    first: u64,
    commit: u64,
    records: Vec<usize>,
}

/// An abort found after damage, which withdraws the records before it.
#[derive(Clone, Copy, Debug)]
struct Withdrawing {
    at: usize,
    end: usize,
    sequence: u64,
    first: u64,
}

/// The records of the segments naming the checkpoint, settled one at a time.
struct Settle<'a> {
    records: Vec<LiveRecord<'a>>,
    /// The place in `records` of each call's first record.
    calls: Vec<usize>,
    /// Operation records after the last boundary and not under a commit.
    pending: Vec<usize>,
    /// Where the first of them sits, being its segment's place among the
    /// live segments, its offset and its sequence.
    pending_start: Option<(usize, u64, u64)>,
    tentative: Option<Tentative>,
    boundary: u64,
    aborts: usize,
    /// The last sequence read, a framing record's included.
    last: u64,
}

impl<'a> Settle<'a> {
    fn new(checkpoint: u64) -> Self {
        Settle {
            records: Vec::new(),
            calls: Vec::new(),
            pending: Vec::new(),
            pending_start: None,
            tentative: None,
            boundary: checkpoint,
            aborts: 0,
            last: checkpoint,
        }
    }

    fn withdraw(&mut self, indexes: &[usize]) {
        for &i in indexes {
            self.records[i].state = RecordState::Withdrawn;
        }
    }

    /// Take one whole record, in sequence order. A framing record that
    /// names the wrong first sequence is refused, with how.
    fn feed(&mut self, record: JournalRecord<'a>, segment: usize) -> Result<(), String> {
        self.last = record.sequence;
        if let Some(tentative) = self.tentative.take() {
            if record.kind == OperationKind::Abort && framing_first(&record)? == tentative.first {
                self.withdraw(&tentative.records);
                self.boundary = record.sequence;
                self.aborts += 1;
                return Ok(());
            }
            self.boundary = tentative.commit;
        }
        match record.kind {
            OperationKind::Commit | OperationKind::Abort => {
                let first = framing_first(&record)?;
                if first != self.boundary + 1 || first >= record.sequence {
                    return Err(format!(
                        "the {} record at sequence {} names sequence {} as its first, and the \
                         first record after the last boundary is {}",
                        record.kind.label(),
                        record.sequence,
                        first,
                        self.boundary + 1
                    ));
                }
                let covered = std::mem::take(&mut self.pending);
                self.pending_start = None;
                if record.kind == OperationKind::Commit {
                    self.tentative = Some(Tentative {
                        first,
                        commit: record.sequence,
                        records: covered,
                    });
                } else {
                    self.withdraw(&covered);
                    self.boundary = record.sequence;
                    self.aborts += 1;
                }
            }
            _ => {
                if self.pending_start.is_none() {
                    self.pending_start = Some((segment, record.offset, record.sequence));
                    self.calls.push(self.records.len());
                }
                self.records.push(LiveRecord {
                    record,
                    state: RecordState::Committed,
                });
                self.pending.push(self.records.len() - 1);
            }
        }
        Ok(())
    }

    /// The first sequences an abort after damage may name: the one after the
    /// last boundary, and the first of a commit read just before the damage.
    fn withdrawable(&self) -> [Option<u64>; 2] {
        match &self.tentative {
            Some(tentative) => [Some(tentative.first), Some(tentative.commit + 1)],
            None => [Some(self.boundary + 1), None],
        }
    }

    /// A whole abort after the damage at `failure` that withdraws back to a
    /// sequence this one may name, where there is one.
    fn find_withdrawing(&self, bytes: &[u8], failure: &Failure) -> Option<Withdrawing> {
        let names = self.withdrawable();
        let mut at = failure.at + 1;
        while at + JOURNAL_RECORD_OVERHEAD_BYTES <= bytes.len() {
            let sequence = u64::from_le_bytes(take8(bytes, at));
            if sequence >= failure.expected {
                if let Parsed::Whole(record) = parse_record(bytes, at, sequence, true) {
                    if record.kind == OperationKind::Abort {
                        if let Ok(first) = framing_first(&record) {
                            if first < record.sequence && names.contains(&Some(first)) {
                                return Some(Withdrawing {
                                    at,
                                    end: at + record_len(&record),
                                    sequence: record.sequence,
                                    first,
                                });
                            }
                        }
                    }
                }
            }
            at += 1;
        }
        None
    }

    /// Withdraw everything from the abort's first sequence to the abort,
    /// listing the whole records between the damage and the abort as
    /// withdrawn.
    fn withdraw_region(&mut self, bytes: &'a [u8], failure: &Failure, abort: &Withdrawing) {
        if let Some(tentative) = self.tentative.take() {
            if abort.first == tentative.first {
                self.withdraw(&tentative.records);
            } else {
                self.boundary = tentative.commit;
            }
        }
        let pending = std::mem::take(&mut self.pending);
        self.withdraw(&pending);
        self.pending_start = None;
        // The records found between the damage and the abort belong to the
        // call open at the damage, and begin one where none was open.
        let mut open = !pending.is_empty();
        let mut next = failure.expected.max(abort.first);
        let mut at = failure.at;
        while at < abort.at && at + JOURNAL_RECORD_OVERHEAD_BYTES <= abort.at {
            let sequence = u64::from_le_bytes(take8(bytes, at));
            if sequence >= next && sequence < abort.sequence {
                if let Parsed::Whole(record) = parse_record(bytes, at, sequence, true) {
                    if !record.kind.is_framing() {
                        if !open {
                            self.calls.push(self.records.len());
                            open = true;
                        }
                        self.records.push(LiveRecord {
                            record,
                            state: RecordState::Withdrawn,
                        });
                    }
                    next = sequence + 1;
                    at += record_len(&record);
                    continue;
                }
            }
            at += 1;
        }
        self.boundary = abort.sequence;
        self.last = abort.sequence;
        self.aborts += 1;
    }

    /// Settle what is left at the end: a commit read last settles, and the
    /// records after the last boundary are a torn tail, whose first record is
    /// returned so the caller cuts there.
    fn finish(&mut self) -> Option<(usize, u64, u64)> {
        if let Some(tentative) = self.tentative.take() {
            self.boundary = tentative.commit;
        }
        if self.pending.is_empty() {
            return None;
        }
        let first = self.pending[0];
        self.truncate(first);
        self.pending.clear();
        self.pending_start.take()
    }

    /// Keep the records before `place` and the calls they begin.
    fn truncate(&mut self, place: usize) {
        self.records.truncate(place);
        let calls = self.calls.partition_point(|&start| start < place);
        self.calls.truncate(calls);
    }
}

/// Read a journal of segments back, every segment in order and every record
/// in sequence, held to the rules of the module documentation. `segments`
/// are in name order. `journal` names the journal's directory in a message.
///
/// A header that does not read, a segment from another collection or another
/// history, a gap, an overlap or a framing record naming the wrong sequence
/// is refused outright. Damage inside a segment is reported in the contents
/// and the caller decides. A corrupt middle carries the cut a repair makes.
pub fn read_segments<'a>(
    segments: &[SegmentBytes<'a>],
    expected: &Expected,
    journal: &str,
) -> Result<JournalContents<'a>, Error> {
    read_journal(segments, expected, journal, false)
}

/// [`read_segments`] for a repair.
///
/// A segment from another collection, a stale segment holding a record
/// above the directory's sequence or following a live one, and a first live
/// segment past the one after the checkpoint are refused as
/// [`read_segments`] refuses them, since such a journal is not the
/// directory's. Every other way the journal breaks its own history stops the
/// reading there and puts a [`Cut`] in the contents: a header that does not
/// read or names another sequence than its file, an empty live segment that
/// is not the last, a gap or an overlap, a framing record naming the wrong
/// sequence, records after the last commit in a sealed segment, a torn
/// creation of the wrong number, and a corrupt middle. `records` then holds
/// the records of the calls before the cut and none after it, and
/// `next_sequence` is the cut's sequence. A segment named at or below the
/// directory's sequence whose header does not read or names another
/// sequence than its file is stale by its name, so it is listed in
/// `stale_set_aside` instead and the reading goes on without it.
pub fn read_segments_for_repair<'a>(
    segments: &[SegmentBytes<'a>],
    expected: &Expected,
    journal: &str,
) -> Result<JournalContents<'a>, Error> {
    read_journal(segments, expected, journal, true)
}

/// The first break of the journal's own history a reading for a repair met:
/// the segment, its file and the byte the break is at, the sequence the
/// record there was expected to carry, what broke and how.
struct Break {
    segment: u64,
    file: String,
    at: u64,
    sequence: u64,
    kind: &'static str,
    detail: String,
}

impl Break {
    /// A break that takes the whole segment, being one whose header does not
    /// read or which breaks the run of segments.
    fn whole(segment: &SegmentBytes<'_>, kind: &'static str, detail: String) -> Self {
        Break {
            segment: segment.number,
            file: segment.file.to_string(),
            at: 0,
            sequence: segment.number,
            kind,
            detail,
        }
    }
}

/// The cut a repair makes for `found`: the first record of the call open at
/// the break where one is open, and the break itself where none is.
fn cut_for(
    settle: &Settle<'_>,
    read: &[(SegmentBytes<'_>, SegmentHeader)],
    live: &[usize],
    found: Break,
) -> Cut {
    let (segment, at, sequence) = match settle.pending_start {
        Some((place, offset, first)) => (read[live[place]].0.number, offset, first),
        None => (found.segment, found.at, found.sequence),
    };
    Cut {
        segment,
        at,
        sequence,
        kind: found.kind,
        damage_segment: found.segment,
        damage_at: found.at,
        damage_sequence: found.sequence,
        detail: found.detail,
    }
}

/// The reader both [`read_segments`] and [`read_segments_for_repair`] run.
/// Where `repairing` is false, every break of the journal's history is the
/// refusal [`read_segments`] documents.
fn read_journal<'a>(
    segments: &[SegmentBytes<'a>],
    expected: &Expected,
    journal: &str,
    repairing: bool,
) -> Result<JournalContents<'a>, Error> {
    let invalid = |detail: String| Error::JournalSegmentsInvalid {
        journal: journal.to_string(),
        detail,
    };
    let checkpoint = expected.base.sequence;
    // The first break a reading for a repair met. A reading that is not for
    // a repair refuses at each one instead.
    let mut broken: Option<Break> = None;

    // Rule 1: every header, and a torn creation as the last alone. A segment
    // named at or below the checkpoint's sequence is stale by its name, so
    // one whose header cannot be used is set aside whole for a repair, and
    // the reading goes on.
    let mut read: Vec<(SegmentBytes<'a>, SegmentHeader)> = Vec::new();
    let mut torn_creation = None;
    let mut stale_set_aside = Vec::new();
    for (i, segment) in segments.iter().enumerate() {
        if i > 0 && segment.number <= segments[i - 1].number {
            return Err(invalid(format!(
                "the segment {} is listed after one numbered at or above it",
                segment.file
            )));
        }
        let stale_by_name = segment.number <= checkpoint;
        let header = match read_segment_header(segment.bytes, segment.file) {
            Ok(header) => header,
            Err(error) if stale_by_name => {
                let detail = header_detail(error);
                stale_set_aside.push(stale_header(segment, checkpoint, detail, repairing)?);
                continue;
            }
            Err(_) if i + 1 == segments.len() && segment.bytes.len() <= SEGMENT_HEADER_BYTES => {
                torn_creation = Some(segment.number);
                continue;
            }
            Err(error) if repairing => {
                broken = Some(Break::whole(segment, "header", error.to_string()));
                break;
            }
            Err(error) => return Err(error),
        };
        if header.first_sequence != segment.number {
            let detail = format!(
                "the segment {} is named for sequence {} and its header names {}",
                segment.file, segment.number, header.first_sequence
            );
            if stale_by_name {
                stale_set_aside.push(stale_header(segment, checkpoint, detail, repairing)?);
                continue;
            }
            let error = invalid(detail);
            if repairing {
                broken = Some(Break::whole(segment, "header", error.to_string()));
                break;
            }
            return Err(error);
        }
        if header.collection_id != expected.collection_id {
            return Err(Error::JournalNotThisCollection {
                file: segment.file.to_string(),
                journal_id: format!("{:032x}", header.collection_id),
                directory_id: format!("{:032x}", expected.collection_id),
            });
        }
        read.push((*segment, header));
    }

    // Rules 2 and 3: stale segments first, then the live ones in a run.
    let mut reports = Vec::with_capacity(read.len());
    let mut scans = Vec::with_capacity(read.len());
    let mut live: Vec<usize> = Vec::new();
    let mut skipped = 0usize;
    for (i, (segment, header)) in read.iter().enumerate() {
        let (records, failure) = scan(
            segment.bytes,
            SEGMENT_HEADER_BYTES,
            header.first_sequence,
            true,
        );
        let stale = header.base != expected.base;
        if stale {
            if !live.is_empty() {
                return Err(invalid(format!(
                    "the segment {} names another checkpoint and follows one naming the \
                     directory's",
                    segment.file
                )));
            }
            if let Some(above) = records.iter().find(|r| r.sequence > checkpoint) {
                return Err(Error::JournalNotThisCheckpoint {
                    file: segment.file.to_string(),
                    detail: format!(
                        "follows the checkpoint at sequence {} with snapshot {:032x} and holds \
                         record {}, and the directory beside it is the checkpoint at sequence \
                         {} with snapshot {:032x}",
                        header.base.sequence,
                        header.base.snapshot,
                        above.sequence,
                        checkpoint,
                        expected.base.snapshot
                    ),
                });
            }
            skipped += records.iter().filter(|r| !r.kind.is_framing()).count();
        } else {
            live.push(i);
        }
        reports.push(SegmentReport {
            number: segment.number,
            bytes: segment.bytes.len() as u64,
            stale,
        });
        scans.push((records, failure));
    }
    if let Some(&first) = live.first() {
        let (segment, header) = &read[first];
        if header.first_sequence > checkpoint + 1 {
            return Err(Error::JournalStartsAfterCheckpoint {
                file: segment.file.to_string(),
                first: header.first_sequence,
                checkpoint,
            });
        }
    }

    // Rules 4 to 7, over the live segments in order. A segment that breaks
    // the run of segments is never the last, so where a reading for a repair
    // broke at a header, the damage in every segment read before it is a
    // corrupt middle.
    let mut settle = Settle::new(checkpoint);
    let mut damage = None;
    let mut damage_file = None;
    let mut good_bytes = 0u64;
    let last_live = if torn_creation.is_some() || broken.is_some() {
        None
    } else {
        live.last().copied()
    };
    'segments: for (place, &i) in live.iter().enumerate() {
        let (segment, header) = &read[i];
        let (records, failure) = &scans[i];
        if place > 0 {
            let (previous, _) = &read[live[place - 1]];
            let before = &scans[live[place - 1]].0;
            let end = before
                .last()
                .map_or(read[live[place - 1]].1.first_sequence - 1, |r| r.sequence);
            if before.is_empty() {
                let error = invalid(format!(
                    "the segment {} holds no record and is not the last",
                    previous.file
                ));
                if repairing {
                    broken = Some(Break::whole(previous, "structure", error.to_string()));
                    break;
                }
                return Err(error);
            }
            if header.first_sequence != end + 1 {
                let error = invalid(format!(
                    "the segment {} begins at sequence {} and the segment before it ends at {}",
                    segment.file, header.first_sequence, end
                ));
                if repairing {
                    broken = Some(Break::whole(segment, "structure", error.to_string()));
                    break;
                }
                return Err(error);
            }
        }
        for record in records {
            if let Err(detail) = settle.feed(*record, place) {
                if repairing {
                    broken = Some(Break {
                        segment: segment.number,
                        file: segment.file.to_string(),
                        at: record.offset,
                        sequence: record.sequence,
                        kind: "structure",
                        detail: invalid(detail).to_string(),
                    });
                    break 'segments;
                }
                return Err(invalid(detail));
            }
        }
        good_bytes = segment.bytes.len() as u64;
        let Some(failure) = failure.clone() else {
            continue;
        };
        if Some(i) != last_live {
            damage = Some(JournalDamage::Corrupt {
                at: failure.at as u64,
                sequence: failure.expected,
                detail: failure.detail,
            });
            damage_file = Some(segment.file.to_string());
            good_bytes = failure.at as u64;
            break;
        }
        let mut stopped = Some(failure);
        while let Some(failure) = stopped.take() {
            if let Some(abort) = settle.find_withdrawing(segment.bytes, &failure) {
                settle.withdraw_region(segment.bytes, &failure, &abort);
                let (rest, after) = scan(segment.bytes, abort.end, abort.sequence + 1, true);
                for record in rest {
                    if let Err(detail) = settle.feed(record, place) {
                        if repairing {
                            broken = Some(Break {
                                segment: segment.number,
                                file: segment.file.to_string(),
                                at: record.offset,
                                sequence: record.sequence,
                                kind: "structure",
                                detail: invalid(detail).to_string(),
                            });
                            break 'segments;
                        }
                        return Err(invalid(detail));
                    }
                }
                stopped = after;
                continue;
            }
            damage_file = Some(segment.file.to_string());
            good_bytes = failure.at as u64;
            damage = Some(
                if a_later_record_parses(segment.bytes, failure.at + 1, failure.expected, true) {
                    JournalDamage::Corrupt {
                        at: failure.at as u64,
                        sequence: failure.expected,
                        detail: failure.detail,
                    }
                } else {
                    JournalDamage::TornTail {
                        at: failure.at as u64,
                        sequence: failure.expected,
                    }
                },
            );
        }
    }

    let corrupt = matches!(damage, Some(JournalDamage::Corrupt { .. }));
    // Saturating, since a checkpoint naming the last sequence leaves none
    // after it, and the writer then refuses the append.
    let mut next_sequence = match &damage {
        Some(JournalDamage::TornTail { sequence, .. }) => *sequence,
        _ => settle.last.saturating_add(1),
    };
    if !corrupt && broken.is_none() {
        if let Some(open) = settle.pending_start {
            if Some(live[open.0]) != last_live {
                let (segment, _) = &read[live[open.0]];
                let error = invalid(format!(
                    "records after the last commit sit in the segment {}, which is not the last",
                    segment.file
                ));
                if !repairing {
                    return Err(error);
                }
                // The open call is the cut, so the break names its first
                // record.
                broken = Some(Break {
                    segment: segment.number,
                    file: segment.file.to_string(),
                    at: open.1,
                    sequence: open.2,
                    kind: "structure",
                    detail: error.to_string(),
                });
            }
        }
    }
    if !corrupt && broken.is_none() {
        if let Some((place, at, sequence)) = settle.finish() {
            damage = Some(JournalDamage::TornTail { at, sequence });
            damage_file = Some(read[live[place]].0.file.to_string());
            good_bytes = at;
            next_sequence = sequence;
        }
    }
    if let (Some(number), None) = (torn_creation, broken.as_ref()) {
        if !corrupt && number != next_sequence {
            let error = invalid(format!(
                "the last segment, numbered {number}, has no whole header and is not the \
                 segment the journal would begin next, numbered {next_sequence}"
            ));
            if !repairing {
                return Err(error);
            }
            let segment = segments
                .last()
                .expect("a torn creation is the last segment listed");
            broken = Some(Break::whole(segment, "structure", error.to_string()));
        } else if corrupt && number != next_sequence && !repairing {
            return Err(invalid(format!(
                "the last segment, numbered {number}, has no whole header and is not the \
                 segment the journal would begin next, numbered {next_sequence}"
            )));
        }
    }
    if live.is_empty() {
        good_bytes = 0;
    }

    // The cut, for a corrupt middle at any reading and for every other break
    // at a reading for a repair. A repair keeps the records before it alone,
    // and the call open at the break, a tentative commit aside, goes. A
    // corrupt middle lies in a segment read before any header that broke the
    // reading, so where both were met the corrupt middle is the first break.
    let found = match (&damage, broken) {
        (
            Some(JournalDamage::Corrupt {
                at,
                sequence,
                detail,
            }),
            _,
        ) => {
            let file = damage_file.clone().unwrap_or_default();
            let number = read
                .iter()
                .find(|(segment, _)| segment.file == file)
                .map_or(0, |(segment, _)| segment.number);
            Some(Break {
                segment: number,
                file,
                at: *at,
                sequence: *sequence,
                kind: "corrupt",
                detail: detail.clone(),
            })
        }
        (_, broken) => broken,
    };
    let mut active = last_live.map(|i| read[i].0.number);
    let mut cut = None;
    if let Some(found) = found {
        let file = found.file.clone();
        let made = cut_for(&settle, &read, &live, found);
        if repairing {
            if let Some(&first) = settle.pending.first() {
                settle.truncate(first);
            }
            if !corrupt {
                damage = Some(JournalDamage::Corrupt {
                    at: made.damage_at,
                    sequence: made.damage_sequence,
                    detail: made.detail.clone(),
                });
                damage_file = Some(file);
            }
            next_sequence = made.sequence;
            good_bytes = made.at;
            active = None;
        }
        cut = Some(made);
    }
    let torn_creation = if repairing && cut.is_some() {
        None
    } else {
        torn_creation
    };
    Ok(JournalContents {
        segments: reports,
        torn_creation,
        records: settle.records,
        calls: settle.calls,
        skipped,
        aborts: settle.aborts,
        damage,
        damage_file,
        good_bytes,
        next_sequence,
        active,
        cut,
        stale_set_aside,
    })
}

/// What a header refusal says is wrong, without the file it names.
fn header_detail(error: Error) -> String {
    match error {
        Error::JournalHeaderInvalid { detail, .. } => detail,
        other => other.to_string(),
    }
}

/// A segment named at or below the directory's sequence `checkpoint` whose
/// header cannot be used, for the reason `detail`. A reading for a repair
/// sets it aside whole, and any other reading refuses the journal, naming
/// the repair.
fn stale_header(
    segment: &SegmentBytes<'_>,
    checkpoint: u64,
    detail: String,
    repairing: bool,
) -> Result<StaleSegment, Error> {
    if repairing {
        return Ok(StaleSegment {
            number: segment.number,
            file: segment.file.to_string(),
            detail,
        });
    }
    Err(Error::JournalHeaderInvalid {
        file: segment.file.to_string(),
        detail: format!(
            "{detail}. The segment is named for sequence {}, at or below the directory's \
             sequence {checkpoint}, so it holds no record the directory needs, and a load with \
             repair=True sets it aside whole in the journal's quarantine folder and keeps every \
             other segment",
            segment.number
        ),
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

/// What withdrawing a call left in the journal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Withdrawal {
    /// Nothing was appended since the last boundary, so there was nothing to
    /// withdraw.
    Nothing,
    /// An abort record withdraws the call, and its sync succeeded.
    Aborted,
    /// The call's bytes were cut from the segment, and the cut was synced.
    CutBack,
    /// Neither landed. The call's records may replay after a crash until a
    /// checkpoint, and the writer takes nothing more until then.
    Neither,
}

impl Withdrawal {
    /// What the withdrawal means for the call's records, as a sentence a
    /// refusal carries.
    pub fn sentence(self) -> &'static str {
        match self {
            Withdrawal::Nothing => "The call had no record in the journal.",
            Withdrawal::Aborted => {
                "The call's records were withdrawn by an abort record, so they never come back \
                 after a crash."
            }
            Withdrawal::CutBack => {
                "The call's records were cut from the journal, so they never come back after a \
                 crash."
            }
            Withdrawal::Neither => {
                "The call's records could be neither withdrawn nor cut from the journal, so \
                 they may come back after a crash until a checkpoint."
            }
        }
    }

    /// `error`, the failure that made the call withdraw, with what the
    /// withdrawal achieved said after it and one full stop between them.
    pub fn attach_to(self, error: Error) -> Error {
        match error {
            Error::JournalIoFailed { path, what, error } => Error::JournalIoFailed {
                path,
                what,
                error: format!(
                    "{}. {}",
                    error.strip_suffix('.').unwrap_or(&error),
                    self.sentence()
                ),
            },
            other => other,
        }
    }
}

/// The active segment's file, shared by the writer and every sync handle,
/// so a handle syncs whichever segment is active when it is asked.
///
/// A standard mutex rather than one from the index crate's rank registry,
/// which this crate cannot name. It is a leaf: it is held to clone or replace
/// the pointer and never across an I/O call or another lock.
#[allow(clippy::disallowed_types)]
type ActiveSlot = std::sync::Mutex<Option<Arc<dyn JournalFile>>>;

/// A handle that syncs the journal's active segment and does nothing else,
/// for a thread that syncs on an interval. It follows the active segment
/// through every roll.
#[derive(Clone, Debug)]
pub struct JournalSyncHandle {
    path: PathBuf,
    slot: Arc<ActiveSlot>,
}

impl JournalSyncHandle {
    /// Sync the active segment's data to the device, where there is one.
    pub fn sync(&self) -> Result<(), Error> {
        let file = self
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        match file {
            Some(file) => file.sync_data().map_err(|error| Error::JournalIoFailed {
                path: self.path.clone(),
                what: "sync",
                error: error.to_string(),
            }),
            None => Ok(()),
        }
    }
}

/// The segment records are appended to.
#[derive(Debug)]
struct Active {
    number: u64,
    file: Arc<dyn JournalFile>,
    length: u64,
}

/// A segment other than the active one, which the writer counts and a
/// checkpoint deletes.
#[derive(Clone, Copy, Debug)]
struct Other {
    number: u64,
    bytes: u64,
    stale: bool,
}

/// The appender.
///
/// One active segment, appended at its end. Every append is one write call,
/// so a process killed inside one leaves a prefix of the record, which the
/// reader classifies as a torn tail, and a write that fails is cut back, so
/// a process that carries on leaves nothing of it.
#[derive(Debug)]
pub struct JournalWriter {
    dir: Box<dyn JournalDir>,
    path: PathBuf,
    collection_id: u128,
    base: SegmentBase,
    active: Option<Active>,
    slot: Arc<ActiveSlot>,
    others: Vec<Other>,
    next_sequence: u64,
    /// The sequence of the last commit or abort record, or the base's
    /// sequence before the first.
    boundary: u64,
    /// The active segment's length at the boundary, which a withdrawal cuts
    /// back to.
    boundary_offset: u64,
    /// Operation records appended since the boundary.
    pending: u64,
    /// Operation records committed since the base.
    committed: u64,
    /// Why the writer takes nothing more, where a write it could not undo
    /// failed.
    poisoned: Option<String>,
    roll_bytes: u64,
    process: u32,
    buffer: Vec<u8>,
}

impl JournalWriter {
    fn io<'p>(path: &'p Path, what: &'static str) -> impl Fn(std::io::Error) -> Error + 'p {
        move |error| Error::JournalIoFailed {
            path: path.to_path_buf(),
            what,
            error: error.to_string(),
        }
    }

    fn with_dir(dir: Box<dyn JournalDir>, collection_id: u128) -> Self {
        JournalWriter {
            path: dir.path().to_path_buf(),
            dir,
            collection_id,
            base: SegmentBase::default(),
            active: None,
            slot: Arc::new(ActiveSlot::new(None)),
            others: Vec::new(),
            next_sequence: 1,
            boundary: 0,
            boundary_offset: 0,
            pending: 0,
            committed: 0,
            poisoned: None,
            roll_bytes: roll_bytes(),
            process: std::process::id(),
            buffer: Vec::new(),
        }
    }

    /// A writer over an empty journal directory, whose first record will be
    /// `next_sequence`. It holds no segment until the checkpoint the journal
    /// follows has committed and [`JournalWriter::roll_to`] names it, and it
    /// refuses an append until then.
    pub fn fresh(
        dir: Box<dyn JournalDir>,
        collection_id: u128,
        next_sequence: u64,
    ) -> Result<Self, Error> {
        let mut writer = Self::with_dir(dir, collection_id);
        if next_sequence == 0 {
            return Err(Error::JournalIoFailed {
                path: writer.path.clone(),
                what: "create",
                error: "sequence 0 is never a record's".into(),
            });
        }
        writer.next_sequence = next_sequence;
        writer.boundary = next_sequence - 1;
        Ok(writer)
    }

    /// A writer that carries on a journal the reader has read, appending to
    /// its active segment after its good bytes.
    ///
    /// The first write is the one that can find the medium refuses writes,
    /// so it comes before anything is changed: removing a segment whose
    /// creation a crash interrupted, or opening the active segment, or
    /// creating it where no segment names the checkpoint. The failure
    /// returned then carries the system's error whole, for the caller to
    /// classify. The torn tail is cut next and synced. Stale segments stay
    /// until [`JournalWriter::remove_stale`].
    pub fn resume(
        dir: Box<dyn JournalDir>,
        collection_id: u128,
        base: SegmentBase,
        contents: &JournalContents<'_>,
    ) -> Result<Self, ResumeFailure> {
        let mut writer = Self::with_dir(dir, collection_id);
        writer.base = base;
        writer.next_sequence = contents.next_sequence;
        writer.boundary = contents.next_sequence - 1;
        writer.committed = contents.committed() as u64;
        let failure = |path: &Path, what: &'static str| {
            let path = path.to_path_buf();
            move |error: std::io::Error| ResumeFailure {
                refuses_writes: crate::storage::refuses_writes(&error),
                error: Error::JournalIoFailed {
                    path,
                    what,
                    error: error.to_string(),
                },
            }
        };
        let active = match contents.active {
            Some(number) => {
                let path = writer.dir.segment_path(number);
                let file = writer
                    .dir
                    .open_segment(number)
                    .map_err(failure(&path, "open"))?;
                file.set_len(contents.good_bytes)
                    .and_then(|()| file.sync_all())
                    .and_then(|()| file.seek_end())
                    .map_err(failure(&path, "cut"))?;
                Active {
                    number,
                    file: Arc::from(file),
                    length: contents.good_bytes,
                }
            }
            None => {
                let number = contents.next_sequence;
                if let Some(torn) = contents.torn_creation {
                    let path = writer.dir.segment_path(torn);
                    writer
                        .dir
                        .remove_segment(torn)
                        .map_err(failure(&path, "remove"))?;
                }
                // A stale segment holding no record can carry the name the new
                // one takes, being the empty active segment of the checkpoint
                // before; it gives way.
                if contents
                    .segments
                    .iter()
                    .any(|s| s.stale && s.number == number)
                {
                    let path = writer.dir.segment_path(number);
                    writer
                        .dir
                        .remove_segment(number)
                        .map_err(failure(&path, "remove"))?;
                }
                let file = writer
                    .create_segment(number, base)
                    .map_err(|error| ResumeFailure {
                        refuses_writes: error.1,
                        error: error.0,
                    })?;
                Active {
                    number,
                    file,
                    length: SEGMENT_HEADER_BYTES as u64,
                }
            }
        };
        for segment in &contents.segments {
            if segment.number != active.number {
                writer.others.push(Other {
                    number: segment.number,
                    bytes: segment.bytes,
                    stale: segment.stale,
                });
            }
        }
        writer.boundary_offset = active.length;
        *writer.lock_slot() = Some(active.file.clone());
        writer.active = Some(active);
        Ok(writer)
    }

    fn lock_slot(&self) -> std::sync::MutexGuard<'_, Option<Arc<dyn JournalFile>>> {
        self.slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Create the segment `number` naming `base`, with its header written and
    /// synced and the directory's entries synced. A segment whose header
    /// could not be made durable is removed again. The second value of the
    /// failure says whether the medium refused the write.
    fn create_segment(
        &self,
        number: u64,
        base: SegmentBase,
    ) -> Result<Arc<dyn JournalFile>, (Error, bool)> {
        let path = self.dir.segment_path(number);
        let fail = |what: &'static str, error: std::io::Error| {
            (
                Error::JournalIoFailed {
                    path: path.clone(),
                    what,
                    error: error.to_string(),
                },
                crate::storage::refuses_writes(&error),
            )
        };
        let file = self
            .dir
            .create_segment(number)
            .map_err(|error| fail("create", error))?;
        let header = encode_segment_header(&SegmentHeader {
            collection_id: self.collection_id,
            base,
            first_sequence: number,
        });
        if let Err(error) = file.write_all(&header).and_then(|()| file.sync_all()) {
            drop(file);
            let _ = self.dir.remove_segment(number);
            return Err(fail("write the header of", error));
        }
        self.dir.sync();
        Ok(Arc::from(file))
    }

    /// Where the journal is, being its directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The collection the journal belongs to.
    pub fn collection_id(&self) -> u128 {
        self.collection_id
    }

    /// The checkpoint the active segment follows.
    pub fn base(&self) -> SegmentBase {
        self.base
    }

    /// The sequence the next append carries.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// The sequence the journal has reached, being the last record appended,
    /// framing included, or the one before the first when nothing has been.
    /// What a checkpoint records after it syncs.
    pub fn sequence_reached(&self) -> u64 {
        self.next_sequence - 1
    }

    /// Operation records committed since the checkpoint the active segment
    /// follows, which is what a replay would apply.
    pub fn committed_operations(&self) -> u64 {
        self.committed
    }

    /// Every byte of every segment the writer knows of.
    pub fn journal_bytes(&self) -> u64 {
        self.others.iter().map(|o| o.bytes).sum::<u64>()
            + self.active.as_ref().map_or(0, |a| a.length)
    }

    /// The segments the writer knows of, the active one included.
    pub fn segments(&self) -> usize {
        self.others.len() + usize::from(self.active.is_some())
    }

    /// A handle for a thread that syncs on an interval, which follows the
    /// active segment through every roll.
    pub fn sync_handle(&self) -> JournalSyncHandle {
        JournalSyncHandle {
            path: self.path.clone(),
            slot: self.slot.clone(),
        }
    }

    /// Refuse in a process that is not the one that opened the writer.
    fn check_process(&self) -> Result<(), Error> {
        let current = std::process::id();
        if current != self.process {
            return Err(Error::JournalForked {
                recorded: self.process,
                current,
            });
        }
        Ok(())
    }

    /// The writer as a test that has to reach the fork check without forking
    /// sees it, being one opened by another process.
    #[cfg(any(test, feature = "test-support"))]
    pub fn pretend_forked(&mut self) {
        self.process = self.process.wrapping_add(1);
    }

    /// The writer as its own process sees it again, undoing
    /// [`JournalWriter::pretend_forked`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn pretend_own_process(&mut self) {
        self.process = std::process::id();
    }

    /// The active segment, or the refusal of a writer that cannot append.
    fn usable(&self) -> Result<Arc<dyn JournalFile>, Error> {
        self.check_process()?;
        if let Some(reason) = &self.poisoned {
            return Err(Error::JournalIoFailed {
                path: self.path.clone(),
                what: "append to",
                error: reason.clone(),
            });
        }
        match &self.active {
            Some(active) => Ok(active.file.clone()),
            None => Err(Error::JournalIoFailed {
                path: self.path.clone(),
                what: "append to",
                error: "the journal has no segment, because the checkpoint it follows has not \
                        committed"
                    .into(),
            }),
        }
    }

    /// Write one record at the active segment's end. A write that fails is
    /// cut back to where the record began, and where the cut fails too the
    /// writer takes nothing more.
    fn write_record(&mut self, kind: OperationKind, payload: &[u8]) -> Result<u64, Error> {
        let file = self.usable()?;
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
        let start = self.active.as_ref().map_or(0, |a| a.length);
        if let Err(error) = file.write_all(&self.buffer) {
            return Err(match file.set_len(start).and_then(|()| file.seek_end()) {
                Ok(()) => Error::JournalIoFailed {
                    path: self.path.clone(),
                    what: "append to",
                    error: format!("{error}, and the bytes the write left were cut back"),
                },
                Err(cut) => {
                    let reason = format!(
                        "an earlier append failed with {error} and the bytes it left could not \
                         be cut back ({cut}), so the journal takes nothing more until \
                         checkpoint() succeeds."
                    );
                    self.poisoned = Some(reason.clone());
                    Error::JournalIoFailed {
                        path: self.path.clone(),
                        what: "append to",
                        error: reason,
                    }
                }
            });
        }
        if let Some(active) = &mut self.active {
            active.length += self.buffer.len() as u64;
        }
        self.next_sequence = after;
        if !kind.is_framing() {
            self.pending += 1;
        }
        Ok(sequence)
    }

    /// Append one operation record in one write and return its sequence.
    /// The bytes are in the kernel when this returns and on the device after
    /// a [`CommitMode::Sync`] commit.
    pub fn append(&mut self, kind: OperationKind, payload: &[u8]) -> Result<u64, Error> {
        debug_assert!(!kind.is_framing(), "the writer frames calls itself");
        self.write_record(kind, payload)
    }

    /// Write the first `bytes` of the record and no more, leaving the
    /// sequence where it was, and sync them.
    ///
    /// What a process killed inside one write leaves, reproduced on purpose
    /// so a crash test can leave it deliberately. The caller aborts straight
    /// afterwards. Debug builds only, since it is the crash tests' and
    /// nothing else's.
    #[cfg(debug_assertions)]
    pub fn append_torn(
        &mut self,
        kind: OperationKind,
        payload: &[u8],
        bytes: usize,
    ) -> Result<(), Error> {
        let file = self.usable()?;
        encode_journal_record(self.next_sequence, kind, payload, &mut self.buffer);
        let cut = bytes.min(self.buffer.len());
        file.write_all(&self.buffer[..cut])
            .and_then(|()| file.sync_data())
            .map_err(Self::io(&self.path, "append to"))
    }

    /// Close the records appended since the last boundary with a commit
    /// record, and make them durable under `mode`.
    ///
    /// A commit record that cannot be written, or a sync that fails, is
    /// followed by [`JournalWriter::withdraw`], and the error returned ends
    /// with what the withdrawal achieved. A call that appended nothing has
    /// nothing to commit.
    pub fn commit(&mut self, mode: CommitMode) -> Result<(), Error> {
        self.check_process()?;
        if self.next_sequence - 1 == self.boundary {
            return Ok(());
        }
        let first = self.boundary + 1;
        let mut payload = Vec::with_capacity(8);
        Operation::Commit { first }.encode(&mut payload);
        let commit = match self.write_record(OperationKind::Commit, &payload) {
            Ok(sequence) => sequence,
            Err(error) => {
                let withdrawal = self.withdraw(AbortReason::WriteFailed);
                return Err(withdrawal.attach_to(error));
            }
        };
        crate::kill_at(crate::KillPoint::AddAfterCommitRecord);
        if mode == CommitMode::Sync {
            let synced = self
                .usable()
                .and_then(|file| file.sync_data().map_err(Self::io(&self.path, "sync")));
            if let Err(error) = synced {
                crate::kill_at(crate::KillPoint::AddAfterFailedCommit);
                let withdrawal = self.withdraw(AbortReason::SyncFailed);
                return Err(withdrawal.attach_to(error));
            }
        }
        self.boundary = commit;
        self.boundary_offset = self.active.as_ref().map_or(0, |a| a.length);
        self.committed += self.pending;
        self.pending = 0;
        self.roll_if_full();
        Ok(())
    }

    /// Withdraw the records appended since the last boundary, being a call
    /// whose commit failed: an abort record naming the call's first sequence,
    /// synced, or where that fails the segment cut back to the boundary and
    /// the cut synced. Where both fail the writer takes nothing more until a
    /// checkpoint. Says which of these landed.
    pub fn withdraw(&mut self, reason: AbortReason) -> Withdrawal {
        let first = self.boundary + 1;
        if self.next_sequence <= first {
            return Withdrawal::Nothing;
        }
        let Some(file) = self.active.as_ref().map(|a| a.file.clone()) else {
            return Withdrawal::Nothing;
        };
        if self.poisoned.is_none() && self.check_process().is_ok() {
            let mut payload = Vec::with_capacity(9);
            Operation::Abort { first, reason }.encode(&mut payload);
            #[cfg(debug_assertions)]
            if crate::kill_armed_at(crate::KillPoint::AddMidAbort) {
                let half = (JOURNAL_RECORD_HEADER_BYTES + payload.len()) / 2;
                let _ = self.append_torn(OperationKind::Abort, &payload, half);
                crate::kill_at(crate::KillPoint::AddMidAbort);
            }
            if let Ok(abort) = self.write_record(OperationKind::Abort, &payload) {
                crate::kill_at(crate::KillPoint::AddAfterAbortBeforeSync);
                if file.sync_data().is_ok() {
                    self.boundary = abort;
                    self.boundary_offset = self.active.as_ref().map_or(0, |a| a.length);
                    self.pending = 0;
                    return Withdrawal::Aborted;
                }
            }
        }
        match file
            .set_len(self.boundary_offset)
            .and_then(|()| file.sync_all())
            .and_then(|()| file.seek_end())
        {
            Ok(()) => {
                if let Some(active) = &mut self.active {
                    active.length = self.boundary_offset;
                }
                self.next_sequence = first;
                self.pending = 0;
                self.poisoned = None;
                Withdrawal::CutBack
            }
            Err(error) => {
                self.poisoned = Some(format!(
                    "a call whose commit failed could be neither withdrawn nor cut back ({error}), \
                     so the journal takes nothing more until checkpoint() succeeds."
                ));
                Withdrawal::Neither
            }
        }
    }

    /// Seal the active segment and start the next where the active one has
    /// passed the roll length.
    ///
    /// The seal is a sync of every record the full segment holds. A seal
    /// that fails stops the writer taking anything more until a checkpoint,
    /// since a later sync does not show that those records reached the
    /// device. A next segment that cannot be created after a seal that
    /// landed leaves the writer on the full segment, whose records are
    /// committed and on the device, and the roll is tried again at the next
    /// commit.
    fn roll_if_full(&mut self) {
        let Some(active) = &self.active else {
            return;
        };
        if active.length < self.roll_bytes {
            return;
        }
        if let Err(error) = active.file.sync_data() {
            warn!(target: LOG_TARGET, operation = "journal_seal_failed",
                journal = %self.path.display(),
                error = %error,
                "The journal's full segment could not be synced before the next one was \
                 started; the journal takes nothing more until a checkpoint"
            );
            self.poisoned = Some(format!(
                "the full segment could not be synced before the next one was started \
                 ({error}), so the journal takes nothing more until checkpoint() succeeds."
            ));
            return;
        }
        crate::kill_at(crate::KillPoint::RollAfterSeal);
        if let Err(error) = self.roll_by_size() {
            warn!(target: LOG_TARGET, operation = "journal_roll_failed",
                journal = %self.path.display(),
                error = %error,
                "The journal's next segment could not be started; records go on into the \
                 full segment and the roll is tried at the next commit"
            );
        }
    }

    /// Start the next segment after a seal that landed.
    fn roll_by_size(&mut self) -> Result<(), Error> {
        let number = self.next_sequence;
        let file = self.create_segment(number, self.base).map_err(|e| e.0)?;
        crate::kill_at(crate::KillPoint::RollAfterCreate);
        let old = self.active.replace(Active {
            number,
            file: file.clone(),
            length: SEGMENT_HEADER_BYTES as u64,
        });
        *self.lock_slot() = Some(file);
        if let Some(old) = old {
            self.others.push(Other {
                number: old.number,
                bytes: old.length,
                stale: false,
            });
        }
        self.boundary_offset = SEGMENT_HEADER_BYTES as u64;
        Ok(())
    }

    /// Roll onto a segment naming the checkpoint that has just committed,
    /// and delete every segment that checkpoint holds.
    ///
    /// The checkpoint names the sequence the writer has reached, so the new
    /// segment begins at the next one. An empty active segment named for
    /// the same sequence gives way first. The new segment's header is synced
    /// before anything is deleted, and the directory's entries after. A
    /// segment that will not be deleted is left, since a later open deletes
    /// it, and logged. Where the new segment cannot be created the writer
    /// takes nothing more, since records appended under the old checkpoint
    /// would be refused by the next open, and a later checkpoint tries again.
    pub fn roll_to(&mut self, base: SegmentBase) -> Result<(), Error> {
        self.check_process()?;
        let number = self.next_sequence;
        debug_assert_eq!(
            base.sequence,
            number - 1,
            "a checkpoint names the sequence reached"
        );
        let empty_namesake = self
            .active
            .as_ref()
            .is_some_and(|a| a.number == number && a.length == SEGMENT_HEADER_BYTES as u64);
        if empty_namesake {
            *self.lock_slot() = None;
            self.active = None;
            if let Err(error) = self.dir.remove_segment(number) {
                let reason = format!(
                    "the empty segment numbered {number} could not give way to the checkpoint's \
                     ({error}), so the journal takes nothing more until checkpoint() succeeds."
                );
                self.poisoned = Some(reason.clone());
                return Err(Error::JournalIoFailed {
                    path: self.path.clone(),
                    what: "roll",
                    error: reason,
                });
            }
        }
        let file = match self.create_segment(number, base) {
            Ok(file) => file,
            Err((error, _)) => {
                self.poisoned = Some(format!(
                    "the segment naming the checkpoint at sequence {} could not be created \
                     ({error}), so the journal takes nothing more until checkpoint() succeeds.",
                    base.sequence
                ));
                return Err(error);
            }
        };
        crate::kill_at(crate::KillPoint::CheckpointAfterRoll);
        let old = self.active.replace(Active {
            number,
            file: file.clone(),
            length: SEGMENT_HEADER_BYTES as u64,
        });
        *self.lock_slot() = Some(file);
        let mut doomed: Vec<u64> = self.others.drain(..).map(|o| o.number).collect();
        if let Some(old) = old {
            doomed.push(old.number);
        }
        self.base = base;
        self.boundary = number - 1;
        self.boundary_offset = SEGMENT_HEADER_BYTES as u64;
        self.pending = 0;
        self.committed = 0;
        self.poisoned = None;
        doomed.sort_unstable();
        for (n, segment) in doomed.into_iter().enumerate() {
            if let Err(error) = self.dir.remove_segment(segment) {
                warn!(target: LOG_TARGET, operation = "journal_segment_kept",
                    journal = %self.path.display(),
                    segment = segment,
                    error = %error,
                    "A segment the checkpoint holds could not be deleted; the next open deletes it"
                );
                self.others.push(Other {
                    number: segment,
                    bytes: 0,
                    stale: true,
                });
            }
            if n == 0 {
                crate::kill_at(crate::KillPoint::CheckpointMidDelete);
            }
        }
        self.dir.sync();
        crate::kill_at(crate::KillPoint::CheckpointAfterDelete);
        Ok(())
    }

    /// Delete the stale segments a resumed journal holds, which the
    /// directory's checkpoint holds every record of. A segment that will not
    /// be deleted is left and logged.
    pub fn remove_stale(&mut self) {
        let mut kept = Vec::with_capacity(self.others.len());
        let mut removed = false;
        for other in self.others.drain(..) {
            if !other.stale {
                kept.push(other);
                continue;
            }
            match self.dir.remove_segment(other.number) {
                Ok(()) => removed = true,
                Err(error) => {
                    warn!(target: LOG_TARGET, operation = "journal_segment_kept",
                        journal = %self.path.display(),
                        segment = other.number,
                        error = %error,
                        "A stale segment could not be deleted; the next open deletes it"
                    );
                    kept.push(other);
                }
            }
        }
        self.others = kept;
        if removed {
            self.dir.sync();
        }
    }

    /// Sync the active segment's data to the device whatever the policy,
    /// which a checkpoint does before it records the sequence it holds.
    pub fn sync(&mut self) -> Result<(), Error> {
        self.check_process()?;
        match &self.active {
            Some(active) => active
                .file
                .sync_data()
                .map_err(Self::io(&self.path, "sync")),
            None => Ok(()),
        }
    }

    /// Whether the writer appends to the process it was opened in and not to
    /// a child forked from it. A sink dropped in such a child leaves the
    /// parent's thread and files alone.
    pub fn in_its_own_process(&self) -> bool {
        std::process::id() == self.process
    }
}

/// Why a writer could not carry on a journal, and whether the medium refused
/// the write, which a load reads to open the collection read only.
#[derive(Debug)]
pub struct ResumeFailure {
    pub error: Error,
    pub refuses_writes: bool,
}

// ============================================================================
// THE TESTS
// ============================================================================

#[cfg(test)]
mod tests;
