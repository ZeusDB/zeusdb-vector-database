//! The journal's tests: the records, both headers, both readers, the writer,
//! and a fuzzer for each reader.

#![allow(clippy::disallowed_types)]

use super::*;
use crate::frame::fuzz::{Rng, HOSTILE};
use crate::storage::{FsStorage, Storage};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A record of a version 1 corpus: its offset, its payload length, its
/// sequence and its kind.
type Entry = (usize, usize, u64, OperationKind);

fn header_error(result: Result<impl std::fmt::Debug, Error>) -> String {
    match result {
        Err(Error::JournalHeaderInvalid { detail, .. }) => detail,
        Err(other) => panic!("expected a header refusal, got {other:?}"),
        Ok(found) => panic!("expected a header refusal, got {found:?}"),
    }
}

/// Recompute a version 1 header's checksum, so a field mutation reaches
/// the field's own check.
fn restamp_legacy_header(blob: &mut [u8]) {
    let sum = checksum_of(&blob[..LEGACY_CHECKSUM_BYTES]);
    blob[56..64].copy_from_slice(&sum.to_le_bytes());
}

/// Recompute a segment header's checksum.
fn restamp_segment_header(blob: &mut [u8]) {
    let sum = checksum_of(&blob[..SEGMENT_CHECKSUM_BYTES]);
    blob[120..128].copy_from_slice(&sum.to_le_bytes());
}

/// Recompute the checksum of the record at `at` whose payload is
/// `payload_len` bytes, so a field mutation reaches the field's own check.
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

/// A valid version 1 journal of `n` records with varied kinds and sizes, and
/// the offset and length of each record's payload.
fn legacy_corpus(n: usize) -> (Vec<u8>, Vec<Entry>) {
    let mut bytes = encode_legacy_header(0x1234_5678_9abc_def0_0fed_cba9_8765_4321, 1).to_vec();
    let mut table = Vec::new();
    let mut record = Vec::new();
    for i in 0..n {
        let kind = OperationKind::OPERATIONS[i % OperationKind::OPERATIONS.len()];
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

// ============================================================================
// THE RECORDS AND THE KINDS
// ============================================================================

/// The first seventeen bytes of both headers mean what the frame's and the
/// dump's mean, and the three magics differ, so a reader dispatching on the
/// magic tells the three apart at byte zero.
#[test]
fn both_headers_share_their_first_seventeen_bytes_with_the_frame_and_the_dump() {
    let legacy = encode_legacy_header(7, 1);
    let segment = encode_segment_header(&SegmentHeader {
        collection_id: 7,
        base: SegmentBase::default(),
        first_sequence: 1,
    });
    for (bytes, version) in [(&legacy[..], 1u32), (&segment[..], 2)] {
        assert_eq!(&bytes[..8], b"ZDBJOURN");
        assert_eq!(u64::from_le_bytes(take8(bytes, 0)), JOURNAL_MAGIC);
        assert_eq!(u32::from_le_bytes(take4(bytes, 8)), version);
        assert_eq!(u32::from_le_bytes(take4(bytes, 12)), 0);
        assert_eq!(bytes[16], JournalKind::Operations as u8);
    }
    assert_ne!(JOURNAL_MAGIC, crate::frame::FRAME_MAGIC);
    assert_ne!(JOURNAL_MAGIC, u64::from_le_bytes(*b"ZDBGRAPH"));
    assert_eq!(legacy.len(), LEGACY_HEADER_BYTES);
    assert_eq!(segment.len(), SEGMENT_HEADER_BYTES);
    assert_eq!(LEGACY_HEADER_BYTES, crate::frame::FRAME_HEADER_BYTES);
}

/// A record is a 16 byte header, the payload and an 8 byte checksum over
/// both, with every field where the layout says.
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
    // Encoding clears the buffer first, so a reused buffer holds one record.
    encode_journal_record(9, OperationKind::Clear, &[], &mut out);
    assert_eq!(out.len(), JOURNAL_RECORD_OVERHEAD_BYTES);
}

/// The twelve kinds carry the numbers the format fixes, every number reads
/// back to its kind, the two framing kinds are the last two, and no other
/// number is a kind. The numbers are on disk for ever.
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
        (OperationKind::Commit, 11, "commit"),
        (OperationKind::Abort, 12, "abort"),
    ];
    assert_eq!(OperationKind::ALL.len(), expected.len());
    for (i, (kind, code, label)) in expected.iter().enumerate() {
        assert_eq!(OperationKind::ALL[i], *kind);
        assert_eq!(kind.code(), *code);
        assert_eq!(OperationKind::from_code(*code), Some(*kind));
        assert_eq!(kind.label(), *label);
        assert_eq!(kind.is_framing(), *code > 10, "{kind:?}");
    }
    assert_eq!(OperationKind::OPERATIONS[..], OperationKind::ALL[..10]);
    for code in [0u16, 13, 255, 256, u16::MAX] {
        assert_eq!(OperationKind::from_code(code), None);
    }
    assert_eq!(JournalKind::Operations.code(), 1);
    assert_eq!(JournalKind::from_code(1), Some(JournalKind::Operations));
    assert_eq!(JournalKind::from_code(0), None);
    assert_eq!(JournalKind::from_code(2), None);
}

/// The checksum over a record and over both headers is the engine's own,
/// the one the frame and the dump use, so one algorithm verifies every
/// artefact.
#[test]
fn the_checksum_is_the_engines_own() {
    let mut out = Vec::new();
    encode_journal_record(1, OperationKind::Insert, &[1, 2, 3], &mut out);
    assert_eq!(
        u64::from_le_bytes(take8(&out, 19)),
        crate::checksum_of(&out[..19])
    );
    let legacy = encode_legacy_header(1, 1);
    assert_eq!(
        u64::from_le_bytes(take8(&legacy, 56)),
        crate::checksum_of(&legacy[..56])
    );
    let segment = encode_segment_header(&SegmentHeader {
        collection_id: 1,
        base: SegmentBase::default(),
        first_sequence: 1,
    });
    assert_eq!(
        u64::from_le_bytes(take8(&segment, 120)),
        crate::checksum_of(&segment[..120])
    );
}

/// `JOURNAL_MAX_PAYLOAD` is a codebook at the widest declaration and the
/// most centroids, plus a mebibyte, and the widest vector sits far under
/// it. The roll length sits under the ceiling, so one call can pass it.
#[test]
fn the_payload_ceiling_is_derived() {
    assert_eq!(JOURNAL_MAX_PAYLOAD, 65 << 20);
    assert_eq!(WIDEST_VECTOR * MOST_CENTROIDS * 4, 64 << 20);
    assert_eq!(SEGMENT_ROLL_BYTES, 64 << 20);
    const { assert!(WIDEST_VECTOR * 4 + (1 << 20) < JOURNAL_MAX_PAYLOAD) }
    const { assert!(JOURNAL_MAX_PAYLOAD < u32::MAX as usize) }
}

// ============================================================================
// THE JOURNAL OF FORMAT VERSION 1, ONE TEST A ROW
// ============================================================================

/// The version 1 header's fields sit where the layout says, and a header
/// reads back to what was written.
#[test]
fn the_legacy_header_lays_out_as_documented() {
    let id = 0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10u128;
    let bytes = encode_legacy_header(id, 0x1122_3344_5566_7788);
    assert_eq!(&bytes[17..24], &[0u8; 7]);
    assert_eq!(u128::from_le_bytes(take16(&bytes, 24)), id);
    assert_eq!(u64::from_le_bytes(take8(&bytes, 40)), 0x1122_3344_5566_7788);
    assert_eq!(u64::from_le_bytes(take8(&bytes, 48)), 0);
    let contents = read_legacy_journal(&bytes, "t").unwrap();
    assert_eq!(
        contents.header,
        LegacyHeader {
            collection_id: id,
            first_sequence: 0x1122_3344_5566_7788
        }
    );
    assert!(contents.records.is_empty());
    assert_eq!(contents.damage, None);
    assert_eq!(contents.good_bytes, 64);
    assert_eq!(contents.next_sequence(), 0x1122_3344_5566_7788);
}

/// A file shorter than its header is refused before anything is read, and
/// a header alone is a journal with no records.
#[test]
fn a_legacy_file_shorter_than_its_header_is_refused() {
    assert_eq!(
        header_error(read_legacy_journal(&[], "t")),
        "the file holds 0 bytes and the header is 64"
    );
    let header = encode_legacy_header(1, 1);
    assert_eq!(
        header_error(read_legacy_journal(&header[..63], "t")),
        "the file holds 63 bytes and the header is 64"
    );
    assert!(read_legacy_journal(&header, "t")
        .unwrap()
        .records
        .is_empty());
}

/// Every way the version 1 header can be wrong is refused by name, in the
/// reader's order. A segment's header is refused as a journal of another
/// version.
#[test]
fn every_legacy_header_damage_is_refused_by_name() {
    let good = encode_legacy_header(5, 3);
    let refused = |bytes: &[u8]| header_error(read_legacy_journal(bytes, "t"));
    let mut bad = good;
    bad[0] = b'X';
    assert!(refused(&bad).contains("journal magic"));
    let mut bad = good;
    bad[8] = 2;
    restamp_legacy_header(&mut bad);
    assert!(refused(&bad).contains("format version 2"));
    let mut bad = good;
    bad[30] ^= 1;
    assert!(refused(&bad).contains("header is corrupt"));
    for at in [12usize, 17, 23, 48, 55] {
        let mut bad = good;
        bad[at] = 1;
        restamp_legacy_header(&mut bad);
        assert!(refused(&bad).contains("reserved field"), "byte {at}");
    }
    let mut bad = good;
    bad[16] = 2;
    restamp_legacy_header(&mut bad);
    assert!(refused(&bad).contains("kind 2"));
    let mut bad = good;
    bad[40..48].copy_from_slice(&0u64.to_le_bytes());
    restamp_legacy_header(&mut bad);
    assert!(refused(&bad).contains("sequence 0"));
    let frame = crate::frame::frame(
        crate::frame::FrameKind::TermDictionary,
        crate::frame::FrameEncoding::Engine,
        0,
        &[],
    );
    assert!(refused(&frame).contains("journal magic"));
    let segment = encode_segment_header(&SegmentHeader {
        collection_id: 5,
        base: SegmentBase::default(),
        first_sequence: 1,
    });
    assert!(refused(&segment).contains("format version 2"));
}

/// `payload_len` is held to the bytes remaining and to the ceiling before
/// the payload is touched.
#[test]
fn a_payload_length_is_held_before_anything_is_sized_from_it() {
    let (good, table) = legacy_corpus(2);
    let second = table[1].0;
    let mut bad = good.clone();
    bad[second + 8..second + 12].copy_from_slice(&(1u32 << 30).to_le_bytes());
    let contents = read_legacy_journal(&bad, "t").unwrap();
    assert_eq!(contents.records.len(), 1);
    assert_eq!(
        contents.damage,
        Some(JournalDamage::TornTail {
            at: second as u64,
            sequence: 2
        })
    );
    let mut bad = good.clone();
    bad[second + 8..second + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    let contents = read_legacy_journal(&bad, "t").unwrap();
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::TornTail { sequence: 2, .. })
    ));
    let mut record = Vec::new();
    let body = vec![0u8; JOURNAL_MAX_PAYLOAD];
    encode_journal_record(1, OperationKind::Insert, &body, &mut record);
    let mut file = encode_legacy_header(1, 1).to_vec();
    file.extend_from_slice(&record);
    let contents = read_legacy_journal(&file, "t").unwrap();
    assert_eq!(contents.records[0].payload.len(), JOURNAL_MAX_PAYLOAD);
    file[64 + 8..64 + 12].copy_from_slice(&((JOURNAL_MAX_PAYLOAD + 1) as u32).to_le_bytes());
    file.push(0);
    let contents = read_legacy_journal(&file, "t").unwrap();
    assert!(contents.records.is_empty());
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::TornTail {
            at: 64,
            sequence: 1
        })
    ));
}

/// A record's sequence is the previous record's plus one, or the header's
/// first for the first record.
#[test]
fn a_sequence_is_held_to_the_previous_plus_one() {
    let (good, table) = legacy_corpus(3);
    let third = table[2].0;
    let mut bad = good.clone();
    bad[third..third + 8].copy_from_slice(&4u64.to_le_bytes());
    restamp_record(&mut bad, third, table[2].1);
    let contents = read_legacy_journal(&bad, "t").unwrap();
    assert_eq!(contents.records.len(), 2);
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::TornTail { sequence: 3, .. })
    ));
    let mut dup = good.clone();
    dup.extend_from_slice(&good[third..]);
    let contents = read_legacy_journal(&dup, "t").unwrap();
    assert_eq!(contents.records.len(), 3);
    assert_eq!(
        contents.damage,
        Some(JournalDamage::TornTail {
            at: good.len() as u64,
            sequence: 4
        })
    );
    let mut bad = good.clone();
    bad[40..48].copy_from_slice(&2u64.to_le_bytes());
    restamp_legacy_header(&mut bad);
    let contents = read_legacy_journal(&bad, "t").unwrap();
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::Corrupt {
            at: 64,
            sequence: 2,
            ..
        })
    ));
}

/// A kind this version does not know fails the record, the two framing
/// kinds among them, and every operation kind reads back.
#[test]
fn a_legacy_kind_is_held_to_the_ten_operations() {
    let (good, table) = legacy_corpus(10);
    let contents = read_legacy_journal(&good, "t").unwrap();
    assert_eq!(contents.records.len(), 10);
    for (record, entry) in contents.records.iter().zip(&table) {
        assert_eq!(record.kind, entry.3);
        assert_eq!(record.sequence, entry.2);
        assert_eq!(record.offset, entry.0 as u64);
        assert_eq!(record.payload.len(), entry.1);
    }
    for code in [0u16, 11, 12, 13, u16::MAX] {
        let mut bad = good.clone();
        let at = table[4].0;
        bad[at + 12..at + 14].copy_from_slice(&code.to_le_bytes());
        restamp_record(&mut bad, at, table[4].1);
        let contents = read_legacy_journal(&bad, "t").unwrap();
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

/// A whole record naming a kind this build does not know is told apart from
/// a damaged record and from a known one, in a journal of format version 1
/// and in a segment, where the framing kinds are known.
#[test]
fn a_whole_record_of_an_unknown_kind_is_told_from_damage() {
    let (good, table) = legacy_corpus(6);
    let at = table[3].0;
    let mut unknown = good.clone();
    unknown[at + 12..at + 14].copy_from_slice(&13u16.to_le_bytes());
    restamp_record(&mut unknown, at, table[3].1);
    let contents = read_legacy_journal(&unknown, "t").unwrap();
    match contents.damage {
        Some(JournalDamage::Corrupt {
            at: found,
            sequence,
            ..
        }) => assert_eq!((found, sequence), (at as u64, 4)),
        other => panic!("expected a corrupt middle, got {other:?}"),
    }
    assert_eq!(unknown_kind_at(&unknown, at as u64, 4, false), Some(13));
    assert_eq!(unknown_kind_at(&unknown, at as u64, 5, false), None);
    assert_eq!(unknown_kind_at(&good, at as u64, 4, false), None);
    assert_eq!(
        unknown_kind_at(&unknown, unknown.len() as u64 + 3, 4, false),
        None
    );
    let mut damaged = unknown.clone();
    damaged[at + 16] ^= 0x01;
    assert_eq!(unknown_kind_at(&damaged, at as u64, 4, false), None);

    let mut built = build(1, &[(base(LIVE, 0), vec![Call::Committed(3)])]);
    let victim = built.known[1].clone();
    {
        let segment = &mut built.segments[0];
        segment[victim.at + 12..victim.at + 14].copy_from_slice(&13u16.to_le_bytes());
        restamp_record(
            segment,
            victim.at,
            victim.len - JOURNAL_RECORD_OVERHEAD_BYTES,
        );
    }
    let files = built.files();
    let contents = built.read(&files, &expected_at(0)).unwrap();
    let (found, sequence) = match contents.damage {
        Some(JournalDamage::Corrupt { at, sequence, .. })
        | Some(JournalDamage::TornTail { at, sequence }) => (at, sequence),
        other => panic!("expected damage, got {other:?}"),
    };
    assert_eq!((found, sequence), (victim.at as u64, 2));
    assert_eq!(
        unknown_kind_at(&built.segments[0], found, sequence, true),
        Some(13)
    );
    let commit = built
        .known
        .iter()
        .find(|known| known.kind == OperationKind::Commit)
        .unwrap();
    assert_eq!(
        unknown_kind_at(&built.segments[0], commit.at as u64, commit.sequence, true),
        None
    );
}

/// A record setting a reserved flag fails, and a record whose checksum
/// does not hold fails, whether the change is in its header or its payload.
#[test]
fn a_flag_and_a_checksum_are_held() {
    let (good, table) = legacy_corpus(2);
    let at = table[1].0;
    let mut bad = good.clone();
    bad[at + 14] = 1;
    restamp_record(&mut bad, at, table[1].1);
    assert!(matches!(
        read_legacy_journal(&bad, "t").unwrap().damage,
        Some(JournalDamage::TornTail { sequence: 2, .. })
    ));
    for offset in [0usize, 9, 13, 16, 16 + table[1].1 - 1, 16 + table[1].1] {
        let mut bad = good.clone();
        bad[at + offset] ^= 0x40;
        let contents = read_legacy_journal(&bad, "t").unwrap();
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

/// An undamaged file reads back whole, and the four torn tails a crash can
/// leave are each reported at the byte the good records end at.
#[test]
fn a_legacy_torn_tail_is_reported_at_the_byte_to_truncate_at() {
    let (good, table) = legacy_corpus(7);
    let contents = read_legacy_journal(&good, "t").unwrap();
    assert_eq!(contents.records.len(), 7);
    assert_eq!(contents.damage, None);
    assert_eq!(contents.good_bytes, good.len() as u64);
    assert_eq!(contents.next_sequence(), 8);
    assert!(contents.refusal("t").is_none());
    let last = &table[6];
    for cut in last.0 + 1..good.len() {
        let contents = read_legacy_journal(&good[..cut], "t").unwrap();
        assert_eq!(contents.records.len(), 6, "cut {cut}");
        assert_eq!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: last.0 as u64,
                sequence: 7
            }),
            "cut {cut}"
        );
    }
    let end = good.len() as u64;
    for zeros in [1usize, 23, 24, 300] {
        let mut bad = good.clone();
        bad.extend(std::iter::repeat_n(0u8, zeros));
        assert_eq!(
            read_legacy_journal(&bad, "t").unwrap().damage,
            Some(JournalDamage::TornTail {
                at: end,
                sequence: 8
            }),
            "zeros {zeros}"
        );
    }
    let mut dup = good.clone();
    dup.extend_from_slice(&good[last.0..]);
    assert_eq!(
        read_legacy_journal(&dup, "t").unwrap().damage,
        Some(JournalDamage::TornTail {
            at: end,
            sequence: 8
        })
    );
}

/// A record that fails with a record after it that parses at a later
/// sequence is a corrupt middle, named by sequence and byte, and the same
/// damage with nothing after it is a torn tail.
#[test]
fn a_legacy_corrupt_middle_is_told_from_a_torn_tail() {
    let (good, table) = legacy_corpus(5);
    let third = &table[2];
    let mut bad = good.clone();
    bad[third.0 + 16 + 3] ^= 0x5A;
    let contents = read_legacy_journal(&bad, "t").unwrap();
    assert_eq!(contents.records.len(), 2);
    assert_eq!(
        contents.damage,
        Some(JournalDamage::Corrupt {
            at: third.0 as u64,
            sequence: 3,
            detail: "checksum mismatch".into()
        })
    );
    match contents.refusal("journal.zdbwal") {
        Some(Error::JournalCorrupt {
            file,
            sequence,
            at,
            detail,
            cut_sequence,
            cut_at,
        }) => {
            assert_eq!(file, "journal.zdbwal");
            assert_eq!(sequence, 3);
            assert_eq!(at, third.0 as u64);
            assert_eq!(detail, "checksum mismatch");
            // Version 1 frames no call, so the cut is the damaged record.
            assert_eq!((cut_sequence, cut_at), (3, third.0 as u64));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    let mut bad = good.clone();
    bad[third.0 + 16] ^= 1;
    bad[table[3].0 + 16] ^= 1;
    assert!(matches!(
        read_legacy_journal(&bad, "t").unwrap().damage,
        Some(JournalDamage::Corrupt { sequence: 3, .. })
    ));
    assert!(matches!(
        read_legacy_journal(&bad[..table[4].0], "t").unwrap().damage,
        Some(JournalDamage::TornTail { sequence: 3, .. })
    ));
}

// ============================================================================
// THE SEGMENT'S HEADER, ONE TEST A FIELD
// ============================================================================

fn base(snapshot: u128, sequence: u64) -> SegmentBase {
    SegmentBase {
        snapshot,
        sequence,
        generation: 3,
        parent: 0x0abc,
    }
}

const COLLECTION: u128 = 0x1234_5678_9abc_def0_0fed_cba9_8765_4321;

fn segment_header(base: SegmentBase, first: u64) -> SegmentHeader {
    SegmentHeader {
        collection_id: COLLECTION,
        base,
        first_sequence: first,
    }
}

/// Every field of a segment's header sits where the layout says, and the
/// header reads back to what was written.
#[test]
fn the_segment_header_lays_out_as_documented() {
    let header = SegmentHeader {
        collection_id: 0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10,
        base: SegmentBase {
            snapshot: 0x1111_2222_3333_4444_5555_6666_7777_8888,
            sequence: 0x0a0b_0c0d,
            generation: 0x0e0f,
            parent: 0x9999_aaaa_bbbb_cccc_dddd_eeee_ffff_0000,
        },
        first_sequence: 0x0a0b_0c0e,
    };
    let bytes = encode_segment_header(&header);
    assert_eq!(
        u128::from_le_bytes(take16(&bytes, 24)),
        header.collection_id
    );
    assert_eq!(
        u128::from_le_bytes(take16(&bytes, 40)),
        header.base.snapshot
    );
    assert_eq!(u64::from_le_bytes(take8(&bytes, 56)), header.base.sequence);
    assert_eq!(
        u64::from_le_bytes(take8(&bytes, 64)),
        header.base.generation
    );
    assert_eq!(u128::from_le_bytes(take16(&bytes, 72)), header.base.parent);
    assert_eq!(u64::from_le_bytes(take8(&bytes, 88)), header.first_sequence);
    assert_eq!(&bytes[17..24], &[0u8; 7]);
    assert_eq!(&bytes[96..120], &[0u8; 24]);
    assert_eq!(
        u64::from_le_bytes(take8(&bytes, 120)),
        checksum_of(&bytes[..120])
    );
    assert_eq!(read_segment_header(&bytes, "s").unwrap(), header);
}

/// Every way a segment's header can be wrong is refused by name, in the
/// reader's order, and a version 1 file is refused as a segment.
#[test]
fn every_segment_header_damage_is_refused_by_name() {
    let good = encode_segment_header(&segment_header(base(9, 4), 5));
    let refused = |bytes: &[u8]| header_error(read_segment_header(bytes, "s"));
    assert_eq!(
        refused(&good[..127]),
        "the file holds 127 bytes and a segment's header is 128"
    );
    let mut bad = good;
    bad[0] = b'X';
    assert!(refused(&bad).contains("journal magic"));
    for version in [1u32, 3, u32::MAX] {
        let mut bad = good;
        bad[8..12].copy_from_slice(&version.to_le_bytes());
        restamp_segment_header(&mut bad);
        assert!(
            refused(&bad).contains(&format!("format version {version} ")),
            "{version}"
        );
    }
    let mut bad = good;
    bad[44] ^= 1;
    assert!(refused(&bad).contains("header is corrupt"));
    for at in [12usize, 15, 17, 23, 96, 108, 119] {
        let mut bad = good;
        bad[at] = 1;
        restamp_segment_header(&mut bad);
        assert!(refused(&bad).contains("reserved field"), "byte {at}");
    }
    let mut bad = good;
    bad[16] = 2;
    restamp_segment_header(&mut bad);
    assert!(refused(&bad).contains("kind 2"));
    for first in [0u64, 3, 4] {
        let mut bad = good;
        bad[88..96].copy_from_slice(&first.to_le_bytes());
        restamp_segment_header(&mut bad);
        assert!(
            refused(&bad).contains(&format!(
                "names sequence {first} as its first record, at or below the sequence 4"
            )),
            "{first}"
        );
    }
    let mut legacy = encode_legacy_header(COLLECTION, 1).to_vec();
    legacy.extend_from_slice(&[0u8; 64]);
    assert!(refused(&legacy).contains("format version 1 "));
}

// ============================================================================
// A JOURNAL OF SEGMENTS, BUILT BY HAND
// ============================================================================

/// One call of a hand-built journal.
#[derive(Clone, Debug)]
enum Call {
    /// Operations then a commit.
    Committed(usize),
    /// Operations, a commit and an abort naming the commit's first.
    AbortedAfterCommit(usize),
    /// Operations and an abort with no commit, as an interval policy's
    /// failure withdraws.
    AbortedWithoutCommit(usize),
    /// Operations and nothing after them.
    Open(usize),
}

/// A record of a hand-built journal: its segment, its offset, its length,
/// its sequence, its kind and, for a framing record, the first it names.
#[derive(Clone, Debug)]
struct Known {
    segment: usize,
    at: usize,
    len: usize,
    sequence: u64,
    kind: OperationKind,
    first: Option<u64>,
}

/// A hand-built journal: its segments' numbers and bytes, and every record.
#[derive(Clone, Debug)]
struct Built {
    numbers: Vec<u64>,
    segments: Vec<Vec<u8>>,
    known: Vec<Known>,
}

impl Built {
    fn files(&self) -> Vec<String> {
        self.numbers
            .iter()
            .map(|n| format!("{n:020}.zdbseg"))
            .collect()
    }

    fn read<'a>(
        &'a self,
        files: &'a [String],
        expected: &Expected,
    ) -> Result<JournalContents<'a>, Error> {
        let segments: Vec<SegmentBytes<'a>> = self
            .numbers
            .iter()
            .zip(&self.segments)
            .zip(files)
            .map(|((number, bytes), file)| SegmentBytes {
                number: *number,
                file,
                bytes,
            })
            .collect();
        read_segments(&segments, expected, "j.zdbjournal")
    }
}

/// Build a journal of segments, each a base and its calls, starting at
/// sequence `first`. A segment's calls run on from the one before.
fn build(first: u64, plan: &[(SegmentBase, Vec<Call>)]) -> Built {
    let mut built = Built {
        numbers: Vec::new(),
        segments: Vec::new(),
        known: Vec::new(),
    };
    let mut sequence = first;
    let mut record = Vec::new();
    let mut ops = 0u32;
    for (s, (segment_base, calls)) in plan.iter().enumerate() {
        built.numbers.push(sequence);
        let mut bytes = encode_segment_header(&segment_header(*segment_base, sequence)).to_vec();
        let mut push = |bytes: &mut Vec<u8>,
                        kind: OperationKind,
                        body: Vec<u8>,
                        sequence: u64,
                        first: Option<u64>| {
            encode_journal_record(sequence, kind, &body, &mut record);
            built.known.push(Known {
                segment: s,
                at: bytes.len(),
                len: record.len(),
                sequence,
                kind,
                first,
            });
            bytes.extend_from_slice(&record);
        };
        for call in calls {
            let (count, commit, abort) = match call {
                Call::Committed(n) => (*n, true, false),
                Call::AbortedAfterCommit(n) => (*n, true, true),
                Call::AbortedWithoutCommit(n) => (*n, false, true),
                Call::Open(n) => (*n, false, false),
            };
            let call_first = sequence;
            for _ in 0..count {
                let kind = OperationKind::OPERATIONS[(ops as usize) % 10];
                let len = [3usize, 0, 41, 9, 120][(ops as usize) % 5];
                push(&mut bytes, kind, payload(ops, len), sequence, None);
                ops += 1;
                sequence += 1;
            }
            let mut framing = Vec::new();
            if commit {
                Operation::Commit { first: call_first }.encode(&mut framing);
                push(
                    &mut bytes,
                    OperationKind::Commit,
                    framing.clone(),
                    sequence,
                    Some(call_first),
                );
                sequence += 1;
            }
            if abort {
                Operation::Abort {
                    first: call_first,
                    reason: AbortReason::SyncFailed,
                }
                .encode(&mut framing);
                push(
                    &mut bytes,
                    OperationKind::Abort,
                    framing.clone(),
                    sequence,
                    Some(call_first),
                );
                sequence += 1;
            }
        }
        built.segments.push(bytes);
    }
    built
}

const STALE: u128 = 0x5157;
const LIVE: u128 = 0x1157;

fn expected_at(sequence: u64) -> Expected {
    Expected {
        collection_id: COLLECTION,
        base: base(LIVE, sequence),
    }
}

fn states(contents: &JournalContents<'_>) -> Vec<(u64, RecordState)> {
    contents
        .records
        .iter()
        .map(|r| (r.record.sequence, r.state))
        .collect()
}

fn committed_seqs(range: std::ops::Range<u64>) -> Vec<(u64, RecordState)> {
    range.map(|s| (s, RecordState::Committed)).collect()
}

fn withdrawn_seqs(range: std::ops::Range<u64>) -> Vec<(u64, RecordState)> {
    range.map(|s| (s, RecordState::Withdrawn)).collect()
}

// ============================================================================
// THE READER OF SEGMENTS, ONE TEST A RULE
// ============================================================================

/// Whole calls in two live segments after a stale one read back: the stale
/// segment's records are skipped, every live operation is committed, the
/// framing records are not listed, and the journal carries on after the
/// last record of the last segment.
#[test]
fn whole_calls_across_segments_read_back() {
    let built = build(
        1,
        &[
            (base(STALE, 0), vec![Call::Committed(2), Call::Committed(1)]),
            (base(LIVE, 5), vec![Call::Committed(3)]),
            (base(LIVE, 5), vec![Call::Committed(2), Call::Committed(1)]),
        ],
    );
    let files = built.files();
    let contents = built.read(&files, &expected_at(5)).unwrap();
    assert_eq!(built.numbers, vec![1, 6, 10]);
    assert_eq!(contents.skipped, 3);
    assert_eq!(
        contents
            .segments
            .iter()
            .map(|s| (s.number, s.stale))
            .collect::<Vec<_>>(),
        vec![(1, true), (6, false), (10, false)]
    );
    let mut want = committed_seqs(6..9);
    want.extend(committed_seqs(10..12));
    want.extend(committed_seqs(13..14));
    assert_eq!(states(&contents), want);
    assert_eq!(contents.calls, vec![0, 3, 5]);
    assert_eq!(contents.committed(), 6);
    assert_eq!(contents.withdrawn(), 0);
    assert_eq!(contents.aborts, 0);
    assert_eq!(contents.damage, None);
    assert_eq!(contents.next_sequence, 15);
    assert_eq!(contents.active, Some(10));
    assert_eq!(contents.good_bytes, built.segments[2].len() as u64);
    assert_eq!(contents.torn_creation, None);
    assert_eq!(contents.live().count(), 2);
    assert!(contents.refusal().is_none());
}

/// Every call is marked at its first record, a withdrawn call's included,
/// and the call a torn tail cut goes with its records.
#[test]
fn each_call_is_marked_where_it_begins() {
    let built = build(
        1,
        &[(
            base(LIVE, 0),
            vec![
                Call::Committed(2),
                Call::AbortedWithoutCommit(1),
                Call::Committed(1),
                Call::Open(2),
            ],
        )],
    );
    let files = built.files();
    let contents = built.read(&files, &expected_at(0)).unwrap();
    let mut want = committed_seqs(1..3);
    want.extend(withdrawn_seqs(4..5));
    want.extend(committed_seqs(6..7));
    assert_eq!(states(&contents), want);
    assert_eq!(contents.calls, vec![0, 2, 3]);
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::TornTail { sequence: 8, .. })
    ));
}

/// Rule 1. A segment from another collection is refused by content, and a
/// segment whose header names another sequence than its file is refused.
#[test]
fn a_segment_from_another_collection_or_another_name_is_refused() {
    let built = build(1, &[(base(LIVE, 0), vec![Call::Committed(2)])]);
    let files = built.files();
    let other = Expected {
        collection_id: 77,
        base: base(LIVE, 0),
    };
    match built.read(&files, &other) {
        Err(Error::JournalNotThisCollection {
            journal_id,
            directory_id,
            ..
        }) => {
            assert_eq!(journal_id, format!("{COLLECTION:032x}"));
            assert_eq!(directory_id, format!("{:032x}", 77));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    let mut renamed = built.clone();
    renamed.numbers[0] = 2;
    let files = renamed.files();
    match renamed.read(&files, &expected_at(0)) {
        Err(Error::JournalSegmentsInvalid { detail, .. }) => {
            assert!(
                detail.contains("is named for sequence 2 and its header names 1"),
                "{detail}"
            )
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// Rule 2. A segment naming another checkpoint is stale when the directory
/// holds every record it holds, and refused, naming both checkpoints, when
/// it holds a record above the directory's sequence, which is what a journal
/// copied beside another checkpoint of its own collection holds.
#[test]
fn a_segment_of_another_checkpoint_is_stale_or_refused() {
    let built = build(1, &[(base(STALE, 0), vec![Call::Committed(4)])]);
    let files = built.files();
    let contents = built.read(&files, &expected_at(5)).unwrap();
    assert_eq!(contents.skipped, 4);
    assert!(contents.records.is_empty());
    assert_eq!(contents.active, None, "no segment names the checkpoint yet");
    assert_eq!(contents.next_sequence, 6);
    match built.read(&files, &expected_at(2)) {
        Err(Error::JournalNotThisCheckpoint { file, detail }) => {
            assert_eq!(file, "00000000000000000001.zdbseg");
            assert!(
                detail.contains(&format!(
                    "follows the checkpoint at sequence 0 with snapshot {STALE:032x}"
                )),
                "{detail}"
            );
            assert!(detail.contains("holds record 3"), "{detail}");
            assert!(
                detail.contains(&format!(
                    "the checkpoint at sequence 2 with snapshot {LIVE:032x}"
                )),
                "{detail}"
            );
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// Rule 3. Stale segments come first, the first live segment begins at the
/// sequence after the checkpoint's, and live segments run on with no gap,
/// no overlap and no empty segment but the last.
#[test]
fn live_segments_begin_after_the_checkpoint_and_run_on() {
    let refused = |built: &Built, expected: &Expected| match built.read(&built.files(), expected) {
        Err(Error::JournalSegmentsInvalid { detail, .. }) => detail,
        other => panic!("expected a refusal, got {other:?}"),
    };
    // A stale segment after a live one.
    let built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(2)]),
            (base(STALE, 0), vec![Call::Committed(1)]),
        ],
    );
    assert!(refused(&built, &expected_at(0)).contains("follows one naming the directory's"));
    // The first live segment above the sequence after the checkpoint's.
    let built = build(4, &[(base(LIVE, 2), vec![Call::Committed(2)])]);
    match built.read(&built.files(), &expected_at(2)) {
        Err(Error::JournalStartsAfterCheckpoint {
            first, checkpoint, ..
        }) => assert_eq!((first, checkpoint), (4, 2)),
        other => panic!("expected a refusal, got {other:?}"),
    }
    // A gap and an overlap between live segments.
    let built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(2)]),
            (base(LIVE, 0), vec![Call::Committed(2)]),
        ],
    );
    let second = built.numbers[1];
    for moved in [second + 1, second - 1] {
        let mut shifted = built.clone();
        shifted.numbers[1] = moved;
        shifted.segments[1][..SEGMENT_HEADER_BYTES].copy_from_slice(&encode_segment_header(
            &segment_header(base(LIVE, 0), moved),
        ));
        let detail = refused(&shifted, &expected_at(0));
        assert!(
            detail.contains("the segment before it ends at 3"),
            "{detail}"
        );
    }
    // Two segments listed out of their order.
    let mut swapped = built.clone();
    swapped.numbers.swap(0, 1);
    swapped.segments.swap(0, 1);
    assert!(refused(&swapped, &expected_at(0)).contains("listed after one numbered at or above it"));
    // An empty live segment that is not the last: the one after it begins
    // where the empty one does, so it carries another number than its own.
    let mut with_empty = build(2, &[(base(LIVE, 0), vec![Call::Committed(2)])]);
    with_empty.numbers.insert(0, 1);
    with_empty.segments.insert(
        0,
        encode_segment_header(&segment_header(base(LIVE, 0), 1)).to_vec(),
    );
    assert!(refused(&with_empty, &expected_at(0)).contains("holds no record and is not the last"));
}

/// Rule 4. Damage in a segment that is not the last is a corrupt middle,
/// since the writer synced it before it started the next, and nothing after
/// it is read.
#[test]
fn damage_in_a_sealed_segment_is_a_corrupt_middle() {
    let mut built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(3)]),
            (base(LIVE, 0), vec![Call::Committed(2)]),
        ],
    );
    let victim = built.known[1].clone();
    built.segments[0][victim.at + 17] ^= 0x40;
    let files = built.files();
    let contents = built.read(&files, &expected_at(0)).unwrap();
    assert_eq!(
        contents.damage,
        Some(JournalDamage::Corrupt {
            at: victim.at as u64,
            sequence: 2,
            detail: "checksum mismatch".into()
        })
    );
    assert_eq!(
        contents.damage_file.as_deref(),
        Some("00000000000000000001.zdbseg")
    );
    match contents.refusal() {
        Some(Error::JournalCorrupt { file, sequence, .. }) => {
            assert_eq!(file, "00000000000000000001.zdbseg");
            assert_eq!(sequence, 2);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// Rule 1's exception. The last segment with no whole header and nothing
/// after its header's length is a creation a crash interrupted, reported
/// to be made again. Anywhere else, or numbered otherwise, it is refused.
#[test]
fn a_torn_creation_is_the_last_segment_alone() {
    let mut built = build(1, &[(base(LIVE, 0), vec![Call::Committed(2)])]);
    for torn in [Vec::new(), vec![0u8; 40], vec![0u8; SEGMENT_HEADER_BYTES]] {
        let mut with = built.clone();
        with.numbers.push(4);
        with.segments.push(torn.clone());
        let files = with.files();
        let contents = with.read(&files, &expected_at(0)).unwrap();
        assert_eq!(contents.torn_creation, Some(4), "{} bytes", torn.len());
        assert_eq!(contents.active, None);
        assert_eq!(contents.next_sequence, 4);
        assert_eq!(states(&contents), committed_seqs(1..3));
    }
    // Numbered for another sequence than the one the journal would begin.
    let mut with = built.clone();
    with.numbers.push(9);
    with.segments.push(Vec::new());
    match with.read(&with.files(), &expected_at(0)) {
        Err(Error::JournalSegmentsInvalid { detail, .. }) => {
            assert!(
                detail.contains("numbered 9, has no whole header"),
                "{detail}"
            )
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    // With bytes after its header's length, or not the last, it is refused.
    let mut damaged = encode_segment_header(&segment_header(base(LIVE, 0), 4)).to_vec();
    damaged[50] ^= 1;
    damaged.push(0);
    built.numbers.push(4);
    built.segments.push(damaged);
    assert!(matches!(
        built.read(&built.files(), &expected_at(0)),
        Err(Error::JournalHeaderInvalid { .. })
    ));
    let mut first_torn = build(1, &[(base(LIVE, 0), vec![Call::Committed(2)])]);
    first_torn.segments[0].truncate(10);
    first_torn.numbers.push(5);
    first_torn
        .segments
        .push(encode_segment_header(&segment_header(base(LIVE, 0), 5)).to_vec());
    assert!(matches!(
        first_torn.read(&first_torn.files(), &expected_at(0)),
        Err(Error::JournalHeaderInvalid { .. })
    ));
}

/// Rules 5 and 7. Records after the last boundary form a torn tail, cut at
/// the first of them, and a commit read last settles.
#[test]
fn records_after_the_last_boundary_are_a_torn_tail() {
    let built = build(
        1,
        &[(base(LIVE, 0), vec![Call::Committed(2), Call::Open(3)])],
    );
    let files = built.files();
    let contents = built.read(&files, &expected_at(0)).unwrap();
    let open = built.known.iter().find(|k| k.sequence == 4).unwrap();
    assert_eq!(states(&contents), committed_seqs(1..3));
    assert_eq!(
        contents.damage,
        Some(JournalDamage::TornTail {
            at: open.at as u64,
            sequence: 4
        })
    );
    assert_eq!(contents.good_bytes, open.at as u64);
    assert_eq!(contents.next_sequence, 4);
    assert!(contents.refusal().is_none());
}

/// Rule 5. A commit or an abort naming another first sequence than the one
/// after the last boundary is refused, and so is one covering no record.
#[test]
fn a_framing_record_naming_the_wrong_sequence_is_refused() {
    let refused = |records: &[(OperationKind, Vec<u8>)]| {
        let mut bytes = encode_segment_header(&segment_header(base(LIVE, 0), 1)).to_vec();
        let mut record = Vec::new();
        for (i, (kind, body)) in records.iter().enumerate() {
            encode_journal_record(i as u64 + 1, *kind, body, &mut record);
            bytes.extend_from_slice(&record);
        }
        let segments = [SegmentBytes {
            number: 1,
            file: "s",
            bytes: &bytes,
        }];
        match read_segments(&segments, &expected_at(0), "j") {
            Err(Error::JournalSegmentsInvalid { detail, .. }) => detail,
            other => panic!("expected a refusal, got {other:?}"),
        }
    };
    let framing = |op: Operation| {
        let mut out = Vec::new();
        op.encode(&mut out);
        (op.kind(), out)
    };
    let op = (OperationKind::Insert, vec![1, 2, 3]);
    let detail = refused(&[op.clone(), framing(Operation::Commit { first: 2 })]);
    assert!(
        detail.contains("the commit record at sequence 2 names sequence 2 as its first"),
        "{detail}"
    );
    assert!(
        detail.contains("first record after the last boundary is 1"),
        "{detail}"
    );
    let detail = refused(&[framing(Operation::Commit { first: 1 })]);
    assert!(
        detail.contains("the commit record at sequence 1 names sequence 1 as its first"),
        "{detail}"
    );
    let detail = refused(&[
        op.clone(),
        framing(Operation::Commit { first: 1 }),
        op.clone(),
        framing(Operation::Abort {
            first: 1,
            reason: AbortReason::SyncFailed,
        }),
    ]);
    assert!(
        detail.contains("the abort record at sequence 4 names sequence 1 as its first"),
        "{detail}"
    );
    assert!(
        detail.contains("first record after the last boundary is 3"),
        "{detail}"
    );
    let detail = refused(&[op, (OperationKind::Commit, vec![1, 2, 3])]);
    assert!(detail.contains("first sequence needs 8 bytes"), "{detail}");
}

/// Rule 5. An abort withdraws the commit it directly follows, and an abort
/// with no commit withdraws the records after the last boundary, and in
/// both the journal carries on after the abort.
#[test]
fn an_abort_withdraws_its_call() {
    let built = build(
        1,
        &[(
            base(LIVE, 0),
            vec![
                Call::Committed(2),
                Call::AbortedAfterCommit(2),
                Call::Committed(1),
                Call::AbortedWithoutCommit(3),
                Call::Committed(1),
            ],
        )],
    );
    let files = built.files();
    let contents = built.read(&files, &expected_at(0)).unwrap();
    let mut want = committed_seqs(1..3);
    want.extend(withdrawn_seqs(4..6));
    want.extend(committed_seqs(8..9));
    want.extend(withdrawn_seqs(10..13));
    want.extend(committed_seqs(14..15));
    assert_eq!(states(&contents), want);
    assert_eq!(contents.aborts, 2);
    assert_eq!(contents.withdrawn(), 5);
    assert_eq!(contents.damage, None);
    assert_eq!(contents.next_sequence, 16);
}

/// Rule 6. Damage followed by an abort naming the call's first sequence is
/// a withdrawn region and not corruption, whether the call's pages are
/// missing, zeroed, half written or whole, and the region's whole records
/// are listed as withdrawn.
#[test]
fn a_withdrawn_region_is_read_past_the_abort() {
    let built = build(
        1,
        &[(
            base(LIVE, 0),
            vec![Call::Committed(2), Call::AbortedAfterCommit(4)],
        )],
    );
    let call: Vec<Known> = built
        .known
        .iter()
        .filter(|k| (4..=8).contains(&k.sequence))
        .cloned()
        .collect();
    let first_at = call[0].at;
    let abort_at = built.known.iter().find(|k| k.sequence == 9).unwrap().at;
    let check = |blob: Vec<u8>, listed: Vec<u64>| {
        let segments = [SegmentBytes {
            number: 1,
            file: "s",
            bytes: &blob,
        }];
        let contents = read_segments(&segments, &expected_at(0), "j").unwrap();
        let mut want = committed_seqs(1..3);
        want.extend(listed.iter().map(|s| (*s, RecordState::Withdrawn)));
        assert_eq!(states(&contents), want);
        assert_eq!(contents.damage, None);
        assert_eq!(contents.aborts, 1);
        assert_eq!(contents.next_sequence, 10);
        assert_eq!(contents.good_bytes, blob.len() as u64);
    };
    // The whole call, withdrawn in order.
    check(built.segments[0].clone(), vec![4, 5, 6, 7]);
    // The call's bytes zeroed, as a page that never reached the device
    // reads after a crash.
    let mut zeroed = built.segments[0].clone();
    zeroed[first_at..abort_at].fill(0);
    check(zeroed, vec![]);
    // Its second and third records lost, and its first and last whole.
    let mut holes = built.segments[0].clone();
    holes[call[1].at..call[3].at].fill(0);
    check(holes, vec![4, 7]);
    // Its first record half written.
    let mut half = built.segments[0].clone();
    half[call[0].at + call[0].len / 2..call[1].at].fill(0xA5);
    check(half, vec![5, 6, 7]);
    // An abort naming the boundary after the commit, not the commit's
    // first, settles the commit and withdraws what follows it alone.
    let mut after = build(1, &[(base(LIVE, 0), vec![Call::Committed(2)])]).segments[0].clone();
    let mut record = Vec::new();
    encode_journal_record(4, OperationKind::Insert, &[9; 30], &mut record);
    let lost = after.len();
    after.extend_from_slice(&record);
    let mut framing = Vec::new();
    Operation::Abort {
        first: 4,
        reason: AbortReason::SyncFailed,
    }
    .encode(&mut framing);
    encode_journal_record(5, OperationKind::Abort, &framing, &mut record);
    after.extend_from_slice(&record);
    after[lost + 3] ^= 0xFF;
    let segments = [SegmentBytes {
        number: 1,
        file: "s",
        bytes: &after,
    }];
    let contents = read_segments(&segments, &expected_at(0), "j").unwrap();
    assert_eq!(states(&contents), committed_seqs(1..3));
    assert_eq!(contents.damage, None);
    assert_eq!(contents.next_sequence, 6);
}

/// Rules 6 and 7. Damage followed by an abort naming another boundary is a
/// corrupt middle, since a whole record follows it, and damage with nothing
/// whole after it is a torn tail that settles the commit before it.
#[test]
fn damage_without_a_withdrawing_abort_is_corrupt_or_torn() {
    let built = build(
        1,
        &[(
            base(LIVE, 0),
            vec![Call::Committed(2), Call::AbortedAfterCommit(2)],
        )],
    );
    let victim = built
        .known
        .iter()
        .find(|k| k.sequence == 2)
        .unwrap()
        .clone();
    let mut blob = built.segments[0].clone();
    blob[victim.at + 20] ^= 1;
    let segments = [SegmentBytes {
        number: 1,
        file: "s",
        bytes: &blob,
    }];
    let contents = read_segments(&segments, &expected_at(0), "j").unwrap();
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::Corrupt { sequence: 2, .. })
    ));
    // A torn half abort after a whole commit: the commit settles.
    let commit_end = {
        let commit = built.known.iter().find(|k| k.sequence == 6).unwrap();
        commit.at + commit.len
    };
    let torn = built.segments[0][..commit_end + 7].to_vec();
    let segments = [SegmentBytes {
        number: 1,
        file: "s",
        bytes: &torn,
    }];
    let contents = read_segments(&segments, &expected_at(0), "j").unwrap();
    let mut want = committed_seqs(1..3);
    want.extend(committed_seqs(4..6));
    assert_eq!(states(&contents), want);
    assert_eq!(
        contents.damage,
        Some(JournalDamage::TornTail {
            at: commit_end as u64,
            sequence: 7
        })
    );
    assert_eq!(contents.next_sequence, 7);
}

/// Rule 7. A repeated record and a run of zeros after the last commit are a
/// torn tail at the end of the last whole call.
#[test]
fn a_repeated_record_and_a_run_of_zeros_are_a_torn_tail() {
    let built = build(1, &[(base(LIVE, 0), vec![Call::Committed(3)])]);
    let end = built.segments[0].len();
    let last = built.known.last().unwrap();
    let mut repeated = built.segments[0].clone();
    repeated.extend_from_slice(&built.segments[0][last.at..]);
    let mut zeros = built.segments[0].clone();
    zeros.extend(std::iter::repeat_n(0u8, 300));
    for blob in [repeated, zeros] {
        let segments = [SegmentBytes {
            number: 1,
            file: "s",
            bytes: &blob,
        }];
        let contents = read_segments(&segments, &expected_at(0), "j").unwrap();
        assert_eq!(states(&contents), committed_seqs(1..4));
        assert_eq!(
            contents.damage,
            Some(JournalDamage::TornTail {
                at: end as u64,
                sequence: 5
            })
        );
    }
}

// ============================================================================
// THE WRITER
// ============================================================================

/// A failure armed on one kind of call: `pass` calls go through, then the
/// `fail` calls after them fail.
#[derive(Debug, Default)]
struct Arm {
    pass: AtomicUsize,
    fail: AtomicUsize,
}

impl Arm {
    fn arm(&self, pass: usize, fail: usize) {
        self.pass.store(pass, Ordering::SeqCst);
        self.fail.store(fail, Ordering::SeqCst);
    }

    /// Whether this call fails.
    fn trips(&self) -> bool {
        let down = |n: usize| n.checked_sub(1);
        if self.fail.load(Ordering::SeqCst) == 0 {
            return false;
        }
        if self
            .pass
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, down)
            .is_ok()
        {
            return false;
        }
        self.fail
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, down)
            .is_ok()
    }
}

/// Failures a test arms on a journal directory. A failing write lands half
/// its bytes first, which is what a device that fills part way leaves.
#[derive(Debug, Default)]
struct Faults {
    writes: Arm,
    syncs: Arm,
    cuts: Arm,
    creates: Arm,
    /// The segment the last sync reached, by its number.
    last_synced: AtomicUsize,
}

fn injected() -> std::io::Error {
    std::io::Error::other("injected: the device refused it")
}

#[derive(Debug)]
struct FaultyDir {
    inner: Box<dyn JournalDir>,
    faults: Arc<Faults>,
}

#[derive(Debug)]
struct FaultyFile {
    inner: Box<dyn JournalFile>,
    faults: Arc<Faults>,
    number: u64,
}

impl JournalDir for FaultyDir {
    fn path(&self) -> &Path {
        self.inner.path()
    }
    fn segments(&self) -> std::io::Result<Vec<u64>> {
        self.inner.segments()
    }
    fn read_segment(&self, number: u64) -> std::io::Result<Vec<u8>> {
        self.inner.read_segment(number)
    }
    fn create_segment(&self, number: u64) -> std::io::Result<Box<dyn JournalFile>> {
        if self.faults.creates.trips() {
            return Err(injected());
        }
        Ok(Box::new(FaultyFile {
            inner: self.inner.create_segment(number)?,
            faults: self.faults.clone(),
            number,
        }))
    }
    fn open_segment(&self, number: u64) -> std::io::Result<Box<dyn JournalFile>> {
        Ok(Box::new(FaultyFile {
            inner: self.inner.open_segment(number)?,
            faults: self.faults.clone(),
            number,
        }))
    }
    fn remove_segment(&self, number: u64) -> std::io::Result<()> {
        self.inner.remove_segment(number)
    }
    fn sync(&self) {
        self.inner.sync()
    }
    fn segment_path(&self, number: u64) -> PathBuf {
        self.inner.segment_path(number)
    }
    fn create_quarantine(&self, folder: &str) -> std::io::Result<()> {
        self.inner.create_quarantine(folder)
    }
    fn quarantine(&self) -> Box<dyn crate::storage::Dir> {
        self.inner.quarantine()
    }
    fn quarantine_folders(&self) -> std::io::Result<Vec<String>> {
        self.inner.quarantine_folders()
    }
    fn sync_quarantine(&self, folder: &str) {
        self.inner.sync_quarantine(folder)
    }
}

impl JournalFile for FaultyFile {
    fn write_all(&self, bytes: &[u8]) -> std::io::Result<()> {
        if self.faults.writes.trips() {
            self.inner.write_all(&bytes[..bytes.len() / 2])?;
            return Err(injected());
        }
        self.inner.write_all(bytes)
    }
    fn seek_end(&self) -> std::io::Result<()> {
        self.inner.seek_end()
    }
    fn set_len(&self, len: u64) -> std::io::Result<()> {
        if self.faults.cuts.trips() {
            return Err(injected());
        }
        self.inner.set_len(len)
    }
    fn sync_data(&self) -> std::io::Result<()> {
        if self.faults.syncs.trips() {
            return Err(injected());
        }
        self.faults
            .last_synced
            .store(self.number as usize, Ordering::SeqCst);
        self.inner.sync_data()
    }
    fn sync_all(&self) -> std::io::Result<()> {
        self.inner.sync_all()
    }
    fn length(&self) -> std::io::Result<u64> {
        self.inner.length()
    }
}

/// A journal directory under a temporary folder, created empty.
fn journal_storage() -> (tempfile::TempDir, FsStorage) {
    let tmp = tempfile::tempdir().unwrap();
    let storage = FsStorage::at(&tmp.path().join("j.zdb")).unwrap();
    storage.create_journal().unwrap();
    (tmp, storage)
}

fn faulty(storage: &FsStorage) -> (Box<dyn JournalDir>, Arc<Faults>) {
    let faults = Arc::new(Faults::default());
    (
        Box::new(FaultyDir {
            inner: storage.journal_dir(),
            faults: faults.clone(),
        }),
        faults,
    )
}

/// Every segment on disk, read whole, for the reader to borrow from.
struct Loaded {
    numbers: Vec<u64>,
    files: Vec<String>,
    bytes: Vec<Vec<u8>>,
}

fn load(storage: &FsStorage) -> Loaded {
    let dir = storage.journal_dir();
    let numbers = dir.segments().unwrap();
    let files = numbers.iter().map(|n| format!("{n:020}.zdbseg")).collect();
    let bytes = numbers
        .iter()
        .map(|n| dir.read_segment(*n).unwrap())
        .collect();
    Loaded {
        numbers,
        files,
        bytes,
    }
}

impl Loaded {
    fn read(&self, expected: &Expected) -> Result<JournalContents<'_>, Error> {
        let segments: Vec<SegmentBytes<'_>> = self
            .numbers
            .iter()
            .zip(&self.files)
            .zip(&self.bytes)
            .map(|((number, file), bytes)| SegmentBytes {
                number: *number,
                file,
                bytes,
            })
            .collect();
        read_segments(&segments, expected, "j.zdbjournal")
    }
}

/// A writer over a fresh journal, rolled onto its first segment naming the
/// checkpoint at sequence zero.
fn started(dir: Box<dyn JournalDir>) -> JournalWriter {
    let mut writer = JournalWriter::fresh(dir, COLLECTION, 1).unwrap();
    writer.roll_to(base(LIVE, 0)).unwrap();
    writer
}

fn message(error: Error) -> String {
    error.to_string()
}

/// A fresh writer holds no segment and appends nothing until the checkpoint
/// it follows names one. Then a call is its records and a commit record
/// naming the first of them, read back as committed, and a call that
/// appended nothing writes nothing.
#[test]
fn a_writer_frames_each_call_with_a_commit() {
    let (_tmp, storage) = journal_storage();
    let mut writer = JournalWriter::fresh(storage.journal_dir(), COLLECTION, 1).unwrap();
    assert!(
        message(writer.append(OperationKind::Insert, b"x").unwrap_err()).contains("no segment")
    );
    assert_eq!(writer.segments(), 0);
    assert_eq!(writer.journal_bytes(), 0);
    writer.commit(CommitMode::Sync).unwrap();
    assert_eq!(
        writer.sequence_reached(),
        0,
        "a call that appended nothing commits nothing"
    );
    writer.roll_to(base(LIVE, 0)).unwrap();
    assert_eq!(writer.segments(), 1);
    assert_eq!(writer.journal_bytes(), SEGMENT_HEADER_BYTES as u64);
    assert_eq!(writer.append(OperationKind::Insert, b"one").unwrap(), 1);
    assert_eq!(writer.append(OperationKind::Remove, b"two").unwrap(), 2);
    writer.commit(CommitMode::Sync).unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    assert_eq!(writer.sequence_reached(), 3, "two records and one commit");
    assert_eq!(writer.committed_operations(), 2);
    writer.append(OperationKind::Clear, &[]).unwrap();
    writer.commit(CommitMode::Deferred).unwrap();
    assert_eq!(writer.committed_operations(), 3);
    writer.sync().unwrap();
    assert_eq!(writer.collection_id(), COLLECTION);
    assert_eq!(writer.base(), base(LIVE, 0));
    let big = vec![0u8; JOURNAL_MAX_PAYLOAD + 1];
    assert!(message(writer.append(OperationKind::Insert, &big).unwrap_err()).contains("ceiling"));
    assert_eq!(writer.next_sequence(), 6);
    drop(writer);

    let loaded = load(&storage);
    assert_eq!(loaded.numbers, vec![1]);
    assert_eq!(
        read_segment_header(&loaded.bytes[0], "s").unwrap(),
        segment_header(base(LIVE, 0), 1)
    );
    let contents = loaded.read(&expected_at(0)).unwrap();
    assert_eq!(
        states(&contents),
        vec![
            (1, RecordState::Committed),
            (2, RecordState::Committed),
            (4, RecordState::Committed)
        ]
    );
    assert_eq!(contents.next_sequence, 6);
    assert_eq!(contents.damage, None);
    assert!(JournalWriter::fresh(storage.journal_dir(), COLLECTION, 0).is_err());
}

/// An append that fails part way is cut back to where its record began, so
/// the next record follows the last whole one at the same sequence and a
/// reader finds nothing of the failed one.
#[test]
fn a_failed_append_is_cut_back() {
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = started(dir);
    writer.append(OperationKind::Insert, b"r0").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    faults.writes.arm(0, 1);
    let failed = message(
        writer
            .append(OperationKind::Insert, b"r1-lost")
            .unwrap_err(),
    );
    assert!(failed.contains("were cut back"), "{failed}");
    assert_eq!(writer.next_sequence(), 3);
    assert_eq!(writer.append(OperationKind::Insert, b"r2").unwrap(), 3);
    writer.commit(CommitMode::Sync).unwrap();
    drop(writer);
    let loaded = load(&storage);
    let contents = loaded.read(&expected_at(0)).unwrap();
    assert_eq!(
        states(&contents),
        vec![(1, RecordState::Committed), (3, RecordState::Committed)]
    );
    assert_eq!(contents.records[1].record.payload, b"r2");
    assert_eq!(contents.damage, None);
}

/// A failed append whose cut back fails too leaves the writer taking
/// nothing more, and a checkpoint's roll onto a new segment makes it whole.
#[test]
fn a_failed_cut_stops_the_writer_until_a_checkpoint_rolls() {
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = started(dir);
    writer.append(OperationKind::Insert, b"r0").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    faults.writes.arm(0, 1);
    faults.cuts.arm(0, 1);
    let failed = message(writer.append(OperationKind::Insert, b"r1").unwrap_err());
    assert!(failed.contains("could not be cut back"), "{failed}");
    let refused = message(writer.append(OperationKind::Insert, b"r2").unwrap_err());
    assert!(
        refused.ends_with("so the journal takes nothing more until checkpoint() succeeds."),
        "{refused}"
    );
    let reached = writer.sequence_reached();
    writer.roll_to(base(LIVE + 1, reached)).unwrap();
    assert_eq!(
        writer.append(OperationKind::Insert, b"r3").unwrap(),
        reached + 1
    );
    writer.commit(CommitMode::Sync).unwrap();
    drop(writer);
    let loaded = load(&storage);
    assert_eq!(loaded.numbers, vec![reached + 1]);
    let expected = Expected {
        collection_id: COLLECTION,
        base: base(LIVE + 1, reached),
    };
    let contents = loaded.read(&expected).unwrap();
    assert_eq!(
        states(&contents),
        vec![(reached + 1, RecordState::Committed)]
    );
}

/// A commit record whose write fails and whose cut fails too is withdrawn
/// by the cut of the whole call back to the last commit. The error holds
/// the refusal, which ends its sentence, and then what the withdrawal
/// achieved, with one full stop between the two.
#[test]
fn a_withdrawal_follows_a_refusal_after_one_full_stop() {
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = started(dir);
    writer.append(OperationKind::Insert, b"r0").unwrap();
    faults.writes.arm(0, 1);
    faults.cuts.arm(0, 1);
    let failed = message(writer.commit(CommitMode::Sync).unwrap_err());
    assert!(
        failed.ends_with(
            "an earlier append failed with injected: the device refused it and the bytes it \
             left could not be cut back (injected: the device refused it), so the journal \
             takes nothing more until checkpoint() succeeds. The call's records were cut from \
             the journal, so they never come back after a crash."
        ),
        "{failed}"
    );
}

/// A writer with two records appended after a committed one, which every
/// failed-commit test starts from. The call's first sequence is 3.
fn two_pending(dir: Box<dyn JournalDir>) -> JournalWriter {
    let mut writer = started(dir);
    writer.append(OperationKind::Insert, b"r0").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    writer.append(OperationKind::Insert, b"r1").unwrap();
    writer.append(OperationKind::Insert, b"r2").unwrap();
    writer
}

/// Withdraw and hold, rule 1. A commit whose sync fails is withdrawn by an
/// abort record, which the error says, and a reader finds the call's records
/// withdrawn and the calls before it committed.
#[test]
fn a_failed_sync_is_withdrawn_by_an_abort() {
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = two_pending(dir);
    faults.syncs.arm(0, 1);
    let failed = message(writer.commit(CommitMode::Sync).unwrap_err());
    assert!(failed.contains("withdrawn by an abort record"), "{failed}");
    assert_eq!(
        writer.next_sequence(),
        7,
        "two records, the commit and the abort"
    );
    assert_eq!(writer.committed_operations(), 1);
    assert_eq!(
        writer.withdraw(AbortReason::SyncFailed),
        Withdrawal::Nothing
    );
    drop(writer);
    let loaded = load(&storage);
    let contents = loaded.read(&expected_at(0)).unwrap();
    assert_eq!(
        states(&contents),
        vec![
            (1, RecordState::Committed),
            (3, RecordState::Withdrawn),
            (4, RecordState::Withdrawn)
        ]
    );
    assert_eq!(contents.aborts, 1);
    assert_eq!(contents.next_sequence, 7);
}

/// Withdraw and hold, rule 1. Where the abort's sync fails, or its write,
/// the segment is cut back to the boundary, the sequence is taken back, and
/// a reader finds no byte of the call.
#[test]
fn a_failed_abort_is_cut_back() {
    for abort_write_fails in [false, true] {
        let (_tmp, storage) = journal_storage();
        let (dir, faults) = faulty(&storage);
        let mut writer = two_pending(dir);
        let boundary = writer.journal_bytes() - 2 * (JOURNAL_RECORD_OVERHEAD_BYTES as u64 + 2);
        if abort_write_fails {
            // The commit record goes through and the abort's write fails.
            faults.writes.arm(1, 1);
            faults.syncs.arm(0, 1);
        } else {
            faults.syncs.arm(0, 2);
        }
        let failed = message(writer.commit(CommitMode::Sync).unwrap_err());
        assert!(failed.contains("cut from the journal"), "{failed}");
        assert_eq!(
            writer.next_sequence(),
            3,
            "the call's sequences are taken back"
        );
        assert_eq!(writer.journal_bytes(), boundary);
        writer.append(OperationKind::Insert, b"r3").unwrap();
        writer.commit(CommitMode::Sync).unwrap();
        drop(writer);
        let loaded = load(&storage);
        let contents = loaded.read(&expected_at(0)).unwrap();
        assert_eq!(
            states(&contents),
            vec![(1, RecordState::Committed), (3, RecordState::Committed)]
        );
        assert_eq!(contents.records[1].record.payload, b"r3");
        assert_eq!(contents.aborts, 0);
    }
}

/// Withdraw and hold, where the commit's sync, the abort and the cut back
/// all fail. The error says the records may come back after a crash until a
/// checkpoint, the writer takes nothing more, and the file holds the call
/// with its commit record, which a reader replays.
#[test]
fn a_failed_abort_and_cut_leave_neither() {
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = two_pending(dir);
    faults.syncs.arm(0, 1);
    faults.writes.arm(1, 1);
    faults.cuts.arm(0, 2);
    let failed = message(writer.commit(CommitMode::Sync).unwrap_err());
    assert!(
        failed.contains("may come back after a crash until a checkpoint"),
        "{failed}"
    );
    assert!(
        message(writer.append(OperationKind::Insert, b"x").unwrap_err())
            .ends_with("so the journal takes nothing more until checkpoint() succeeds.")
    );
    drop(writer);
    let loaded = load(&storage);
    let contents = loaded.read(&expected_at(0)).unwrap();
    assert_eq!(
        states(&contents),
        vec![
            (1, RecordState::Committed),
            (3, RecordState::Committed),
            (4, RecordState::Committed)
        ]
    );
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::TornTail { .. })
    ));
}

/// The active segment rolls at a commit once it passes the roll length, the
/// segments read back as one journal, and the sync handle follows the
/// active segment through every roll.
#[test]
fn a_full_segment_rolls_at_a_commit_and_the_sync_handle_follows() {
    set_segment_roll_bytes(400);
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = started(dir);
    set_segment_roll_bytes(0);
    let handle = writer.sync_handle();
    for i in 0..9u8 {
        writer.append(OperationKind::Insert, &[i; 100]).unwrap();
        writer.commit(CommitMode::Sync).unwrap();
        handle.sync().unwrap();
        let active = load(&storage).numbers.last().copied().unwrap();
        assert_eq!(
            faults.last_synced.load(Ordering::SeqCst) as u64,
            active,
            "the handle synced the active segment after call {i}"
        );
    }
    assert!(writer.segments() > 2, "{} segments", writer.segments());
    let loaded = load(&storage);
    assert_eq!(loaded.numbers.len(), writer.segments());
    assert_eq!(
        writer.journal_bytes(),
        loaded.bytes.iter().map(|b| b.len() as u64).sum::<u64>()
    );
    let contents = loaded.read(&expected_at(0)).unwrap();
    assert_eq!(contents.committed(), 9);
    assert_eq!(contents.active, loaded.numbers.last().copied());
    assert_eq!(contents.next_sequence, 19);
}

/// A next segment that cannot be created after the seal leaves the writer
/// on the full segment, whose records are committed, and the next commit
/// rolls.
#[test]
fn a_failed_roll_leaves_the_writer_on_the_full_segment() {
    set_segment_roll_bytes(200);
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = started(dir);
    set_segment_roll_bytes(0);
    faults.creates.arm(0, 1);
    writer.append(OperationKind::Insert, &[1; 100]).unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    assert_eq!(writer.segments(), 1);
    writer.append(OperationKind::Insert, &[2; 100]).unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    assert_eq!(writer.segments(), 2);
    drop(writer);
    let loaded = load(&storage);
    assert_eq!(loaded.read(&expected_at(0)).unwrap().committed(), 2);
}

/// A seal that fails, being the sync of the full segment before the next
/// one starts, stops the writer until a checkpoint, since a later sync does
/// not show that the records the seal covered reached the device. The
/// commit that filled the segment returns with its call committed and
/// starts no segment, every append after it is refused in words that name
/// the seal and the checkpoint, and the roll of a checkpoint lets the
/// writer append again. The seal is the commit's second sync under the
/// synced mode and its first under the deferred one.
#[test]
fn a_failed_seal_stops_the_writer_until_a_checkpoint() {
    for (mode, before_the_seal) in [(CommitMode::Sync, 1), (CommitMode::Deferred, 0)] {
        set_segment_roll_bytes(200);
        let (_tmp, storage) = journal_storage();
        let (dir, faults) = faulty(&storage);
        let mut writer = started(dir);
        set_segment_roll_bytes(0);
        faults.syncs.arm(before_the_seal, 1);
        writer.append(OperationKind::Insert, &[1; 100]).unwrap();
        writer.commit(mode).unwrap();
        assert_eq!(
            faults.syncs.fail.load(Ordering::SeqCst),
            0,
            "{mode:?}: the seal's sync failed"
        );
        assert_eq!(
            writer.segments(),
            1,
            "{mode:?}: the failed seal started no segment"
        );
        let refused = match writer.append(OperationKind::Insert, &[2; 100]) {
            Err(error) => message(error),
            Ok(sequence) => panic!(
                "{mode:?}: the append after a failed seal is refused, and it took sequence \
                 {sequence}"
            ),
        };
        assert!(
            refused.ends_with(
                "the full segment could not be synced before the next one was started \
                 (injected: the device refused it), so the journal takes nothing more until \
                 checkpoint() succeeds."
            ),
            "{mode:?}: {refused}"
        );
        let reached = writer.sequence_reached();
        writer.roll_to(base(LIVE + 1, reached)).unwrap();
        writer.append(OperationKind::Insert, &[3; 100]).unwrap();
        writer.commit(mode).unwrap();
        drop(writer);
        let loaded = load(&storage);
        let expected = Expected {
            collection_id: COLLECTION,
            base: base(LIVE + 1, reached),
        };
        assert_eq!(
            states(&loaded.read(&expected).unwrap()),
            vec![(reached + 1, RecordState::Committed)],
            "{mode:?}: the checkpoint's segment holds the call after it"
        );
    }
}

/// A checkpoint rolls onto a segment naming it, at the sequence after the
/// one it holds, and deletes every segment it holds. An empty active segment
/// named for that sequence gives way first.
#[test]
fn a_checkpoint_rolls_onto_a_segment_naming_it() {
    set_segment_roll_bytes(300);
    let (_tmp, storage) = journal_storage();
    let mut writer = started(storage.journal_dir());
    set_segment_roll_bytes(0);
    for i in 0..5u8 {
        writer.append(OperationKind::Insert, &[i; 100]).unwrap();
        writer.commit(CommitMode::Sync).unwrap();
    }
    assert!(load(&storage).numbers.len() > 1);
    let reached = writer.sequence_reached();
    let next = base(LIVE + 7, reached);
    writer.roll_to(next).unwrap();
    assert_eq!(writer.segments(), 1);
    assert_eq!(writer.committed_operations(), 0);
    let loaded = load(&storage);
    assert_eq!(loaded.numbers, vec![reached + 1]);
    assert_eq!(
        read_segment_header(&loaded.bytes[0], "s").unwrap().base,
        next
    );
    // Nothing since: the next checkpoint's segment takes the same name.
    let again = base(LIVE + 8, reached);
    writer.roll_to(again).unwrap();
    let loaded = load(&storage);
    assert_eq!(loaded.numbers, vec![reached + 1]);
    assert_eq!(
        read_segment_header(&loaded.bytes[0], "s").unwrap().base,
        again
    );
    writer.append(OperationKind::Insert, b"after").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    drop(writer);
    let loaded = load(&storage);
    let expected = Expected {
        collection_id: COLLECTION,
        base: again,
    };
    let contents = loaded.read(&expected).unwrap();
    assert_eq!(
        states(&contents),
        vec![(reached + 1, RecordState::Committed)]
    );
}

/// A checkpoint whose segment cannot be created leaves the writer taking
/// nothing more, since a record under the old checkpoint would be refused
/// by the next open, and a later checkpoint makes it whole.
#[test]
fn a_failed_checkpoint_roll_stops_the_writer() {
    let (_tmp, storage) = journal_storage();
    let (dir, faults) = faulty(&storage);
    let mut writer = started(dir);
    writer.append(OperationKind::Insert, b"r0").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    faults.creates.arm(0, 1);
    let reached = writer.sequence_reached();
    assert!(writer.roll_to(base(LIVE + 1, reached)).is_err());
    assert!(
        message(writer.append(OperationKind::Insert, b"r1").unwrap_err())
            .ends_with("so the journal takes nothing more until checkpoint() succeeds.")
    );
    writer.roll_to(base(LIVE + 1, reached)).unwrap();
    writer.append(OperationKind::Insert, b"r1").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
}

/// A resumed writer cuts the torn tail the reader found, records after the
/// last commit among it, and carries on at the sequence the reader gave.
#[test]
fn a_resumed_writer_cuts_the_torn_tail_and_carries_on() {
    let (_tmp, storage) = journal_storage();
    let mut writer = started(storage.journal_dir());
    writer.append(OperationKind::Insert, b"r0").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    writer.append(OperationKind::Insert, b"open").unwrap();
    drop(writer);
    let loaded = load(&storage);
    let contents = loaded.read(&expected_at(0)).unwrap();
    assert!(matches!(
        contents.damage,
        Some(JournalDamage::TornTail { sequence: 3, .. })
    ));
    let mut writer =
        JournalWriter::resume(storage.journal_dir(), COLLECTION, base(LIVE, 0), &contents).unwrap();
    assert_eq!(writer.next_sequence(), 3);
    assert_eq!(writer.journal_bytes(), contents.good_bytes);
    writer.append(OperationKind::Insert, b"r1").unwrap();
    writer.commit(CommitMode::Sync).unwrap();
    drop(writer);
    let loaded = load(&storage);
    let contents = loaded.read(&expected_at(0)).unwrap();
    assert_eq!(
        states(&contents),
        vec![(1, RecordState::Committed), (3, RecordState::Committed)]
    );
    assert_eq!(contents.damage, None);
}

/// A resumed writer creates the segment naming the checkpoint where no
/// segment names it, which is a crash between a checkpoint's commit and its
/// roll, and where a crash interrupted its creation, and a stale empty
/// segment of the same name gives way. The stale segments go once asked.
#[test]
fn a_resumed_writer_creates_the_segment_a_crash_left_missing() {
    for left in ["stale", "torn", "empty namesake"] {
        let (_tmp, storage) = journal_storage();
        let mut writer = started(storage.journal_dir());
        if left != "empty namesake" {
            writer.append(OperationKind::Insert, b"r0").unwrap();
            writer.commit(CommitMode::Sync).unwrap();
        }
        let reached = writer.sequence_reached();
        drop(writer);
        if left == "torn" {
            let torn = storage.journal_dir().create_segment(reached + 1).unwrap();
            torn.write_all(&[0u8; 40]).unwrap();
        }
        let checkpoint = base(LIVE + 3, reached);
        let expected = Expected {
            collection_id: COLLECTION,
            base: checkpoint,
        };
        let loaded = load(&storage);
        let contents = loaded.read(&expected).unwrap();
        assert_eq!(contents.active, None, "{left}");
        let mut writer =
            JournalWriter::resume(storage.journal_dir(), COLLECTION, checkpoint, &contents)
                .unwrap();
        writer.remove_stale();
        assert_eq!(writer.segments(), 1, "{left}");
        writer.append(OperationKind::Insert, b"r1").unwrap();
        writer.commit(CommitMode::Sync).unwrap();
        drop(writer);
        let loaded = load(&storage);
        assert_eq!(loaded.numbers, vec![reached + 1], "{left}");
        let contents = loaded.read(&expected).unwrap();
        assert_eq!(
            states(&contents),
            vec![(reached + 1, RecordState::Committed)],
            "{left}"
        );
    }
}

/// A writer refuses every operation in a process other than the one that
/// opened it, which is what a child forked from it is.
#[test]
fn a_forked_writer_refuses_every_operation() {
    let (_tmp, storage) = journal_storage();
    let mut writer = started(storage.journal_dir());
    writer.pretend_forked();
    assert!(!writer.in_its_own_process());
    let forked = |result: Result<(), Error>| {
        assert!(
            matches!(result, Err(Error::JournalForked { .. })),
            "{result:?}"
        )
    };
    forked(writer.append(OperationKind::Insert, b"x").map(|_| ()));
    forked(writer.commit(CommitMode::Sync));
    forked(writer.sync());
    let reached = writer.sequence_reached();
    forked(writer.roll_to(base(LIVE, reached)));
    assert_eq!(
        writer.journal_bytes(),
        SEGMENT_HEADER_BYTES as u64,
        "nothing was written"
    );
}

/// No record carries the last sequence a `u64` holds, since no sequence
/// follows it, so a record restamped to it fails where it stands.
#[test]
fn no_record_carries_the_last_sequence() {
    let (good, table) = legacy_corpus(2);
    let mut bad = good.clone();
    let at = table[1].0;
    bad[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    restamp_record(&mut bad, at, table[1].1);
    let contents = read_legacy_journal(&bad, "t").unwrap();
    assert_eq!(contents.records.len(), 1);
    assert_eq!(
        contents.damage,
        Some(JournalDamage::TornTail {
            at: at as u64,
            sequence: 2
        })
    );
    let mut file = encode_legacy_header(1, u64::MAX).to_vec();
    let mut record = Vec::new();
    encode_journal_record(u64::MAX, OperationKind::Insert, b"x", &mut record);
    file.extend_from_slice(&record);
    let contents = read_legacy_journal(&file, "t").unwrap();
    assert!(contents.records.is_empty());
    assert_eq!(contents.next_sequence(), u64::MAX);
}

// ============================================================================
// THE READER FOR A REPAIR
// ============================================================================

/// The journal a repair keeps: every segment but the stale ones it sets
/// aside, and where it cuts, every segment numbered below the cut's and the
/// cut's own up to the cut where it keeps any of it.
fn kept_by(built: &Built, cut: Option<&Cut>, stale: &[StaleSegment]) -> Built {
    let mut kept = Built {
        numbers: Vec::new(),
        segments: Vec::new(),
        known: Vec::new(),
    };
    for (number, bytes) in built.numbers.iter().zip(&built.segments) {
        if stale.iter().any(|segment| segment.number == *number) {
            continue;
        }
        match cut {
            Some(cut) if *number > cut.segment || (*number == cut.segment && cut.at == 0) => {}
            Some(cut) if *number == cut.segment => {
                kept.numbers.push(*number);
                kept.segments.push(bytes[..cut.at as usize].to_vec());
            }
            _ => {
                kept.numbers.push(*number);
                kept.segments.push(bytes.clone());
            }
        }
    }
    kept
}

/// `built` read for a repair.
fn read_for_repair<'a>(
    built: &'a Built,
    files: &'a [String],
    expected: &Expected,
) -> Result<JournalContents<'a>, Error> {
    let segments: Vec<SegmentBytes<'a>> = built
        .numbers
        .iter()
        .zip(&built.segments)
        .zip(files)
        .map(|((number, bytes), file)| SegmentBytes {
            number: *number,
            file,
            bytes,
        })
        .collect();
    read_segments_for_repair(&segments, expected, "j.zdbjournal")
}

/// Hold a reading of `built` for a repair to its rule: where it cuts or sets
/// a stale segment aside, the journal the repair keeps reads as an ordinary
/// load reads it and holds exactly the records the reading kept in the same
/// states. Where it cuts, that journal has no damage at all. Where it only
/// sets stale segments aside, it has the damage the reading met in the live
/// segments, which a load cuts as it would.
fn holds_its_cut(built: &Built, expected: &Expected, contents: &JournalContents<'_>) {
    let cut = contents.cut.as_ref();
    if cut.is_none() && contents.stale_set_aside.is_empty() {
        return;
    }
    let stale = &contents.stale_set_aside;
    let kept = kept_by(built, cut, stale);
    let kept_files = kept.files();
    let again = kept.read(&kept_files, expected).unwrap_or_else(|e| {
        panic!("the kept journal does not read at {cut:?} with {stale:?} set aside: {e}")
    });
    assert_eq!(again.cut, None);
    assert!(again.stale_set_aside.is_empty());
    assert_eq!(
        states(&again),
        states(contents),
        "the kept journal holds the kept records at {cut:?} with {stale:?} set aside"
    );
    match cut {
        Some(cut) => {
            assert_eq!(contents.next_sequence, cut.sequence);
            assert!(contents.damage.is_some());
            assert_eq!(
                again.damage, None,
                "the kept journal reads clean at {cut:?}"
            );
        }
        None => {
            assert_eq!(again.damage, contents.damage, "with {stale:?} set aside");
            assert_eq!(again.next_sequence, contents.next_sequence);
        }
    }
}

/// Read `built` for a repair and hold the reading to its rule. Returns the
/// reading's cut and its records.
fn repaired(built: &Built, expected: &Expected) -> (Option<Cut>, Vec<(u64, RecordState)>) {
    let files = built.files();
    let contents = read_for_repair(built, &files, expected).unwrap();
    holds_its_cut(built, expected, &contents);
    (contents.cut.clone(), states(&contents))
}

/// A segment named at or below the directory's sequence is stale by its
/// name, whatever its header holds. Where its header does not read, or names
/// another sequence than its file, a load refuses the journal and names the
/// repair, and a reading for a repair sets the segment aside whole, cuts
/// nothing for it, and keeps every record of the live segments. A corrupt
/// middle in a live segment is still cut where it falls.
#[test]
fn a_stale_segment_whose_header_cannot_be_used_is_set_aside_for_a_repair() {
    let clean = build(
        1,
        &[
            (base(STALE, 0), vec![Call::Committed(2), Call::Committed(1)]),
            (base(LIVE, 5), vec![Call::Committed(3)]),
            (base(LIVE, 5), vec![Call::Committed(2), Call::Committed(1)]),
        ],
    );
    let expected = expected_at(5);
    let files = clean.files();
    let loaded = clean.read(&files, &expected).unwrap();
    for how in ["corrupt", "renumbered"] {
        let mut built = clean.clone();
        if how == "corrupt" {
            built.segments[0][40] ^= 0x40;
        } else {
            built.segments[0][88..96].copy_from_slice(&3u64.to_le_bytes());
            restamp_segment_header(&mut built.segments[0]);
        }
        let refused = built.read(&files, &expected).unwrap_err();
        let message = refused.to_string();
        assert!(
            matches!(refused, Error::JournalHeaderInvalid { .. }),
            "{how}: {message}"
        );
        assert!(
            message.contains("at or below the directory's sequence 5")
                && message.contains("repair=True"),
            "{how}: {message}"
        );
        let contents = read_for_repair(&built, &files, &expected).unwrap();
        assert_eq!(contents.cut, None, "{how}");
        assert_eq!(
            contents
                .stale_set_aside
                .iter()
                .map(|segment| segment.number)
                .collect::<Vec<_>>(),
            vec![1],
            "{how}"
        );
        assert_eq!(states(&contents), states(&loaded), "{how}");
        assert_eq!(contents.next_sequence, loaded.next_sequence, "{how}");
        assert_eq!(contents.damage, None, "{how}");
        holds_its_cut(&built, &expected, &contents);
    }

    let mut both = clean.clone();
    both.segments[0][40] ^= 0x40;
    let seventh = both.known.iter().find(|k| k.sequence == 7).unwrap().clone();
    both.segments[seventh.segment][seventh.at + 17] ^= 0x40;
    let contents = read_for_repair(&both, &files, &expected).unwrap();
    assert_eq!(contents.stale_set_aside.len(), 1);
    let cut = contents.cut.clone().expect("the corrupt middle is cut");
    assert_eq!((cut.segment, cut.sequence, cut.kind), (6, 6, "corrupt"));
    holds_its_cut(&both, &expected, &contents);
}

/// A journal with no damage reads the same for a repair as for a load, and
/// nothing is cut.
#[test]
fn an_undamaged_journal_reads_the_same_for_a_repair() {
    let built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(3)]),
            (
                base(LIVE, 0),
                vec![Call::AbortedAfterCommit(2), Call::Committed(1)],
            ),
        ],
    );
    let files = built.files();
    let loaded = built.read(&files, &expected_at(0)).unwrap();
    let (cut, kept) = repaired(&built, &expected_at(0));
    assert_eq!(cut, None);
    assert_eq!(kept, states(&loaded));
    assert_eq!(loaded.cut, None);
}

/// A corrupt middle is cut at the first record of the call the damage is
/// in, and every record from there on is set aside. A load names the same
/// cut in its refusal.
#[test]
fn a_corrupt_middle_is_cut_at_the_first_record_of_its_call() {
    // Records 1 and 2 committed at 3, then 4, 5 and 6 committed at 7.
    let mut built = build(
        1,
        &[(base(LIVE, 0), vec![Call::Committed(2), Call::Committed(3)])],
    );
    let fourth = built
        .known
        .iter()
        .find(|k| k.sequence == 4)
        .unwrap()
        .clone();
    let fifth = built
        .known
        .iter()
        .find(|k| k.sequence == 5)
        .unwrap()
        .clone();
    built.segments[0][fifth.at + 17] ^= 0x40;
    let files = built.files();
    let loaded = built.read(&files, &expected_at(0)).unwrap();
    match loaded.refusal() {
        Some(Error::JournalCorrupt {
            sequence,
            at,
            cut_sequence,
            cut_at,
            ..
        }) => {
            assert_eq!((sequence, at), (5, fifth.at as u64));
            assert_eq!((cut_sequence, cut_at), (4, fourth.at as u64));
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    let message = loaded.refusal().unwrap().to_string();
    assert!(message.contains("repair=True"), "{message}");
    assert!(
        message.contains(&format!("byte {}", fourth.at)),
        "{message}"
    );
    assert!(message.contains("record 4"), "{message}");
    let (cut, kept) = repaired(&built, &expected_at(0));
    let cut = cut.unwrap();
    assert_eq!(
        (cut.segment, cut.at, cut.sequence, cut.kind),
        (1, fourth.at as u64, 4, "corrupt")
    );
    assert_eq!(
        (cut.damage_segment, cut.damage_at, cut.damage_sequence),
        (1, fifth.at as u64, 5)
    );
    assert_eq!(kept, committed_seqs(1..3));
}

/// Damage in a sealed segment sets aside the rest of that segment from the
/// first record of its call, and every segment after it.
#[test]
fn damage_in_a_sealed_segment_sets_aside_every_later_segment() {
    // Segment 1 holds 1, 2 and 3 committed at 4. Segment 5 holds 5 and 6
    // committed at 7.
    let mut built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(3)]),
            (base(LIVE, 0), vec![Call::Committed(2)]),
        ],
    );
    let second = built
        .known
        .iter()
        .find(|k| k.sequence == 2)
        .unwrap()
        .clone();
    built.segments[0][second.at + 17] ^= 0x40;
    let (cut, kept) = repaired(&built, &expected_at(0));
    let cut = cut.unwrap();
    assert_eq!(
        (cut.segment, cut.at, cut.sequence),
        (1, SEGMENT_HEADER_BYTES as u64, 1),
        "the call began at the segment's first record"
    );
    assert!(kept.is_empty());
}

/// A header that does not read takes its whole segment, and every segment
/// after it, while a load refuses it.
#[test]
fn a_header_that_does_not_read_cuts_its_whole_segment() {
    let mut built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(2)]),
            (base(LIVE, 0), vec![Call::Committed(2)]),
            (base(LIVE, 0), vec![Call::Committed(1)]),
        ],
    );
    built.segments[1][9] ^= 1;
    let files = built.files();
    assert!(matches!(
        built.read(&files, &expected_at(0)),
        Err(Error::JournalHeaderInvalid { .. })
    ));
    let (cut, kept) = repaired(&built, &expected_at(0));
    let cut = cut.unwrap();
    assert_eq!(
        (cut.segment, cut.at, cut.sequence, cut.kind),
        (built.numbers[1], 0, built.numbers[1], "header")
    );
    assert_eq!(kept, committed_seqs(1..3));
}

/// A segment missing from the run cuts the segment after the gap whole.
#[test]
fn a_gap_cuts_the_segment_after_it() {
    let mut built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(2)]),
            (base(LIVE, 0), vec![Call::Committed(2)]),
            (base(LIVE, 0), vec![Call::Committed(1)]),
        ],
    );
    built.numbers.remove(1);
    built.segments.remove(1);
    let files = built.files();
    assert!(matches!(
        built.read(&files, &expected_at(0)),
        Err(Error::JournalSegmentsInvalid { .. })
    ));
    let (cut, kept) = repaired(&built, &expected_at(0));
    let cut = cut.unwrap();
    assert_eq!(
        (cut.segment, cut.at, cut.kind),
        (built.numbers[1], 0, "structure")
    );
    assert_eq!(kept, committed_seqs(1..3));
}

/// A framing record naming the wrong first sequence is cut at the first
/// record of the call it would close.
#[test]
fn a_framing_record_naming_the_wrong_sequence_is_cut_at_its_call() {
    // Records 1 and 2 committed at 3, then 4 and 5 committed at 6.
    let mut built = build(
        1,
        &[(base(LIVE, 0), vec![Call::Committed(2), Call::Committed(2)])],
    );
    let commit = built
        .known
        .iter()
        .find(|k| k.sequence == 6)
        .unwrap()
        .clone();
    let fourth = built
        .known
        .iter()
        .find(|k| k.sequence == 4)
        .unwrap()
        .clone();
    let at = commit.at + JOURNAL_RECORD_HEADER_BYTES;
    built.segments[0][at..at + 8].copy_from_slice(&5u64.to_le_bytes());
    restamp_record(&mut built.segments[0], commit.at, 8);
    let files = built.files();
    assert!(matches!(
        built.read(&files, &expected_at(0)),
        Err(Error::JournalSegmentsInvalid { .. })
    ));
    let (cut, kept) = repaired(&built, &expected_at(0));
    let cut = cut.unwrap();
    assert_eq!(
        (cut.segment, cut.at, cut.sequence, cut.kind),
        (1, fourth.at as u64, 4, "structure")
    );
    assert_eq!((cut.damage_at, cut.damage_sequence), (commit.at as u64, 6));
    assert_eq!(kept, committed_seqs(1..3));
}

/// A call left open at the end of a sealed segment is cut at its first
/// record, and the segment after it goes.
#[test]
fn an_open_call_in_a_sealed_segment_is_cut_at_its_first_record() {
    // Segment 1 holds 1 committed at 2, then 3 and 4 with no commit.
    // Segment 5 holds 5 committed at 6.
    let built = build(
        1,
        &[
            (base(LIVE, 0), vec![Call::Committed(1), Call::Open(2)]),
            (base(LIVE, 0), vec![Call::Committed(1)]),
        ],
    );
    let third = built
        .known
        .iter()
        .find(|k| k.sequence == 3)
        .unwrap()
        .clone();
    let files = built.files();
    assert!(matches!(
        built.read(&files, &expected_at(0)),
        Err(Error::JournalSegmentsInvalid { .. })
    ));
    let (cut, kept) = repaired(&built, &expected_at(0));
    let cut = cut.unwrap();
    assert_eq!(
        (cut.segment, cut.at, cut.sequence, cut.kind),
        (1, third.at as u64, 3, "structure")
    );
    assert_eq!(kept, committed_seqs(1..2));
    // The same open call as the last segment's is a torn tail, which a
    // load cuts and a repair leaves to it.
    let built = build(
        1,
        &[(base(LIVE, 0), vec![Call::Committed(1), Call::Open(2)])],
    );
    let (cut, kept) = repaired(&built, &expected_at(0));
    assert_eq!(cut, None);
    assert_eq!(kept, committed_seqs(1..2));
}

/// A journal of another collection, a stale segment holding a record above
/// the directory's sequence, and a first live segment past the one after
/// the checkpoint are refused under repair as without it.
#[test]
fn a_journal_not_the_directorys_is_refused_under_repair() {
    let built = build(1, &[(base(LIVE, 0), vec![Call::Committed(2)])]);
    let files = built.files();
    let segments: Vec<SegmentBytes<'_>> = built
        .numbers
        .iter()
        .zip(&built.segments)
        .zip(&files)
        .map(|((number, bytes), file)| SegmentBytes {
            number: *number,
            file,
            bytes,
        })
        .collect();
    let other = Expected {
        collection_id: COLLECTION ^ 1,
        base: base(LIVE, 0),
    };
    assert!(matches!(
        read_segments_for_repair(&segments, &other, "j"),
        Err(Error::JournalNotThisCollection { .. })
    ));
    let elsewhere = Expected {
        collection_id: COLLECTION,
        base: base(STALE, 0),
    };
    assert!(matches!(
        read_segments_for_repair(&segments, &elsewhere, "j"),
        Err(Error::JournalNotThisCheckpoint { .. })
    ));
    let later = build(5, &[(base(LIVE, 2), vec![Call::Committed(1)])]);
    let later_files = later.files();
    let segments: Vec<SegmentBytes<'_>> = later
        .numbers
        .iter()
        .zip(&later.segments)
        .zip(&later_files)
        .map(|((number, bytes), file)| SegmentBytes {
            number: *number,
            file,
            bytes,
        })
        .collect();
    assert!(matches!(
        read_segments_for_repair(&segments, &expected_at(2), "j"),
        Err(Error::JournalStartsAfterCheckpoint { .. })
    ));
}

/// The whole records past a cut are counted by the reader's own scans, the
/// run before a damaged record and the run after it, framing aside.
#[test]
fn the_records_set_aside_are_counted_past_the_damage() {
    // Records 1 and 2 committed at 3, then 4, 5 and 6 committed at 7.
    let mut built = build(
        1,
        &[(base(LIVE, 0), vec![Call::Committed(2), Call::Committed(3)])],
    );
    let fourth = built
        .known
        .iter()
        .find(|k| k.sequence == 4)
        .unwrap()
        .clone();
    assert_eq!(
        whole_records_from(&built.segments[0], fourth.at, 4, true),
        3
    );
    let fifth = built
        .known
        .iter()
        .find(|k| k.sequence == 5)
        .unwrap()
        .clone();
    built.segments[0][fifth.at + 17] ^= 0x40;
    assert_eq!(
        whole_records_from(&built.segments[0], fourth.at, 4, true),
        2,
        "the damaged record is not counted and the ones either side of it are"
    );
    assert_eq!(
        whole_records_from(&built.segments[0], SEGMENT_HEADER_BYTES, 1, true),
        4
    );
}

// ============================================================================
// THE FUZZERS
// ============================================================================

fn budget(default: usize) -> usize {
    std::env::var("ZEUSDB_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The version 1 header fields a mutation lands on, as offset and width.
const LEGACY_HEADER_FIELDS: [(usize, usize); 7] = [
    (8, 4),
    (12, 4),
    (16, 1),
    (17, 7),
    (24, 16),
    (40, 8),
    (48, 8),
];

/// A record the mutator knows is whole, being its bytes at `at` and the
/// sequence its header names: a copy appended at the end, or a record
/// restamped after its header was changed. `first` is the sequence a
/// framing record's payload names, and none for any other record or for a
/// framing payload that does not decode.
#[derive(Clone)]
struct Whole {
    at: usize,
    bytes: Vec<u8>,
    sequence: u64,
    kind: OperationKind,
    first: Option<u64>,
}

/// The first sequence a framing payload names, read as the codec reads it:
/// a commit is the sequence alone, and an abort is the sequence and a reason
/// this build knows. None for any other kind and any other payload.
fn named_first(kind: OperationKind, payload: &[u8]) -> Option<u64> {
    let decodes = match kind {
        OperationKind::Commit => payload.len() == 8,
        OperationKind::Abort => payload.len() == 9 && AbortReason::from_code(payload[8]).is_some(),
        _ => false,
    };
    decodes.then(|| u64::from_le_bytes(take8(payload, 0)))
}

/// Restamp the record at `at` whose payload is `len` bytes and, where its
/// header still reads as a record of the original length, a kind `framing`
/// allows, no flags and a sequence some record may carry, note it as whole
/// at whatever sequence it now names, with the first its payload now names.
/// The payload is read from the bytes, since an earlier mutation of the same
/// case may have changed it.
fn restamp_and_note(
    blob: &mut [u8],
    at: usize,
    len: usize,
    framing: bool,
    wholes: &mut Vec<Whole>,
) {
    restamp_record(blob, at, len);
    let named_len = u32::from_le_bytes(take4(blob, at + 8)) as usize;
    let kind = OperationKind::from_code(u16::from_le_bytes(take2(blob, at + 12)))
        .filter(|kind| framing || !kind.is_framing());
    let flags = u16::from_le_bytes(take2(blob, at + 14));
    let sequence = u64::from_le_bytes(take8(blob, at));
    if let (true, Some(kind), 0, false) = (named_len == len, kind, flags, sequence == u64::MAX) {
        let end = at + JOURNAL_RECORD_OVERHEAD_BYTES + len;
        let payload = &blob[at + JOURNAL_RECORD_HEADER_BYTES..end - JOURNAL_RECORD_CHECKSUM_BYTES];
        wholes.push(Whole {
            at,
            bytes: blob[at..end].to_vec(),
            sequence,
            kind,
            first: named_first(kind, payload),
        });
    }
}

/// Apply one to three mutations to a valid version 1 journal, over the
/// header's fields, a record's length, kind, sequence, flags, checksum and
/// payload, and the file's length. `body_only` keeps the header whole except
/// for its first sequence, so the classification can be judged against the
/// record table.
fn mutate_legacy(
    rng: &mut Rng,
    base_bytes: &[u8],
    table: &[Entry],
    body_only: bool,
) -> (Vec<u8>, Vec<Whole>) {
    let mut blob = base_bytes.to_vec();
    let mut wholes = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let len = blob.len();
        let body = len.saturating_sub(LEGACY_HEADER_BYTES);
        match rng.below(13) {
            0..=2 => {
                if body > 0 {
                    let width = [1usize, 2, 4, 8][rng.below(4)];
                    let at = LEGACY_HEADER_BYTES + rng.below(body);
                    let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                    let end = (at + width).min(len);
                    blob[at..end].copy_from_slice(&value[..end - at]);
                }
            }
            3 => {
                if body > 0 {
                    let at = LEGACY_HEADER_BYTES + rng.below(body);
                    blob[at] = rng.byte();
                }
            }
            4 => blob.truncate(rng.below(len + 1)),
            5 => {
                let extra = 1 + rng.below(64);
                let value = if rng.below(2) == 0 { 0 } else { rng.byte() };
                blob.extend(std::iter::repeat_n(value, extra));
            }
            6 => {
                if len >= LEGACY_HEADER_BYTES {
                    let (at, width) = if body_only {
                        (40, 8)
                    } else {
                        LEGACY_HEADER_FIELDS[rng.below(LEGACY_HEADER_FIELDS.len())]
                    };
                    let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                    let take = width.min(8);
                    blob[at..at + take].copy_from_slice(&value[..take]);
                    if body_only || rng.below(8) != 0 {
                        restamp_legacy_header(&mut blob);
                    }
                }
            }
            7 | 8 => {
                let entry = &table[rng.below(table.len())];
                if end_of(entry) <= len {
                    let value = HOSTILE[rng.below(HOSTILE.len())];
                    if rng.below(2) == 0 {
                        blob[entry.0..entry.0 + 8].copy_from_slice(&value.to_le_bytes());
                    } else {
                        blob[entry.0 + 8..entry.0 + 12]
                            .copy_from_slice(&(value as u32).to_le_bytes());
                    }
                    restamp_and_note(&mut blob, entry.0, entry.1, false, &mut wholes);
                }
            }
            9 => {
                let entry = &table[rng.below(table.len())];
                if end_of(entry) <= len {
                    if rng.below(2) == 0 {
                        let value = if body_only {
                            [0u16, 11, 12, 255, u16::MAX][rng.below(5)]
                        } else {
                            HOSTILE[rng.below(HOSTILE.len())] as u16
                        };
                        blob[entry.0 + 12..entry.0 + 14].copy_from_slice(&value.to_le_bytes());
                    } else {
                        let value = 1 + rng.below(u16::MAX as usize) as u16;
                        blob[entry.0 + 14..entry.0 + 16].copy_from_slice(&value.to_le_bytes());
                    }
                    restamp_and_note(&mut blob, entry.0, entry.1, false, &mut wholes);
                }
            }
            10 => {
                let entry = &table[rng.below(table.len())];
                if end_of(entry) <= len {
                    wholes.push(Whole {
                        at: blob.len(),
                        bytes: base_bytes[entry.0..end_of(entry)].to_vec(),
                        sequence: entry.2,
                        kind: entry.3,
                        first: None,
                    });
                    blob.extend_from_slice(&base_bytes[entry.0..end_of(entry)]);
                }
            }
            11 => {
                let entry = &table[rng.below(table.len())];
                blob.truncate(entry.0.min(len));
            }
            _ => {
                if body > 0 {
                    let at = LEGACY_HEADER_BYTES + rng.below(body);
                    let end = (at + 1 + rng.below(48)).min(len);
                    blob[at..end].fill(0);
                }
            }
        }
    }
    (blob, wholes)
}

/// The classification the mutator's own knowledge implies for a version 1
/// journal, derived by byte comparison against the records it knows are
/// whole rather than by parsing, so it is an oracle independent of the
/// reader.
fn legacy_oracle(
    base_bytes: &[u8],
    blob: &[u8],
    table: &[Entry],
    wholes: &[Whole],
) -> Result<(usize, Option<JournalDamage>, u64), ()> {
    if blob.len() < LEGACY_HEADER_BYTES {
        return Err(());
    }
    let stored = u64::from_le_bytes(take8(blob, 56));
    let first = u64::from_le_bytes(take8(blob, 40));
    if blob[..24] != base_bytes[..24]
        || blob[48..56] != base_bytes[48..56]
        || stored != checksum_of(&blob[..LEGACY_CHECKSUM_BYTES])
        || first == 0
    {
        return Err(());
    }
    let mut all: Vec<Whole> = table
        .iter()
        .map(|e| Whole {
            at: e.0,
            bytes: base_bytes[e.0..end_of(e)].to_vec(),
            sequence: e.2,
            kind: e.3,
            first: None,
        })
        .collect();
    all.extend(wholes.iter().cloned());
    let whole = |w: &Whole| {
        blob.len() >= w.at + w.bytes.len() && blob[w.at..w.at + w.bytes.len()] == w.bytes[..]
    };
    let mut expected = first;
    let mut accepted = 0usize;
    let mut at = LEGACY_HEADER_BYTES;
    while let Some(w) = all
        .iter()
        .find(|w| w.at == at && w.sequence == expected && whole(w))
    {
        accepted += 1;
        at += w.bytes.len();
        expected = w.sequence.wrapping_add(1);
    }
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

/// **The property, at format version 1.** Every mutation of a valid journal
/// reads or is refused at the header, none panics, every refusal is the
/// reader's own error, and a useful share of cases reach past the header.
#[test]
fn no_mutation_of_a_valid_legacy_journal_panics_the_reader() {
    let (good, table) = legacy_corpus(12);
    assert_eq!(read_legacy_journal(&good, "t").unwrap().records.len(), 12);
    let mut rng = Rng(0x5eed_1046_e000_0146);
    let cases = budget(4_000);
    let (mut past_header, mut whole, mut torn, mut corrupt) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..cases {
        let (blob, _) = mutate_legacy(&mut rng, &good, &table, false);
        match read_legacy_journal(&blob, "t") {
            Ok(contents) => {
                past_header += 1;
                assert!(contents.good_bytes <= blob.len() as u64);
                assert!(contents.good_bytes >= LEGACY_HEADER_BYTES as u64);
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
                assert!(contents.records.iter().all(|r| !r.kind.is_framing()));
                let _ = contents.next_sequence();
            }
            Err(Error::JournalHeaderInvalid { .. }) => {}
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }
    assert!(
        past_header * 2 > cases,
        "only {past_header} of {cases} reached past the header"
    );
    assert!(
        whole > 0 && torn > 0 && corrupt > 0,
        "{whole} {torn} {corrupt}"
    );
}

/// **The classification, at format version 1.** Every mutation of the body
/// is classified as the record table says it should be.
#[test]
fn the_legacy_classification_matches_an_oracle_under_mutation() {
    let (good, table) = legacy_corpus(12);
    let mut rng = Rng(0x5eed_1046_c000_0146);
    let cases = budget(4_000);
    let (mut torn, mut corrupt, mut whole) = (0usize, 0usize, 0usize);
    for case in 0..cases {
        let (blob, wholes) = mutate_legacy(&mut rng, &good, &table, true);
        match (
            read_legacy_journal(&blob, "t"),
            legacy_oracle(&good, &blob, &table, &wholes),
        ) {
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
                    (found, want) => {
                        panic!("case {case}: the reader said {found:?} and the oracle {want:?}")
                    }
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

/// The journal of segments the second pair of fuzzers mutates: a stale
/// segment, a sealed live one, and a last live one holding a whole call, a
/// call withdrawn after its commit, a call withdrawn with no commit, two
/// whole calls and an open call.
fn segment_corpus() -> (Built, Expected) {
    let built = build(
        1,
        &[
            (base(STALE, 0), vec![Call::Committed(2), Call::Committed(1)]),
            (
                base(LIVE, 5),
                vec![Call::Committed(3), Call::AbortedAfterCommit(2)],
            ),
            (
                base(LIVE, 5),
                vec![
                    Call::Committed(2),
                    Call::AbortedAfterCommit(3),
                    Call::Committed(1),
                    Call::AbortedWithoutCommit(2),
                    Call::Committed(2),
                    Call::Open(2),
                ],
            ),
        ],
    );
    (built, expected_at(5))
}

/// What the oracle predicts for a journal of segments.
#[derive(Debug, PartialEq, Eq)]
enum Predicted {
    /// Refused at the last segment's header.
    Header,
    /// Refused for a framing record naming the wrong first sequence.
    Refused,
    /// Read, with these records and states, this damage, these good bytes,
    /// this next sequence, this many aborts, this torn creation and this
    /// many withdrawn regions. For a corrupt middle only the damage is
    /// compared.
    Read {
        records: Vec<(u64, RecordState)>,
        damage: Option<JournalDamage>,
        good_bytes: u64,
        next_sequence: u64,
        aborts: usize,
        torn_creation: Option<u64>,
        regions: usize,
    },
}

/// The oracle's model of the rule: records after the last boundary are
/// open; a commit holds them until the next record, which withdraws them
/// where it is an abort naming the commit's first and settles them where it
/// is anything else; an abort with no commit withdraws them; a framing
/// record decodes, names the sequence after the last boundary and covers a
/// record, or the journal is refused.
#[derive(Default)]
struct Model {
    boundary: u64,
    held: Option<(u64, u64, Vec<u64>)>,
    open: Vec<(u64, usize)>,
    out: Vec<(u64, RecordState)>,
    aborts: usize,
    last: u64,
    refused: bool,
    regions: usize,
}

impl Model {
    fn at_checkpoint(sequence: u64) -> Self {
        Model {
            boundary: sequence,
            last: sequence,
            ..Model::default()
        }
    }

    fn take(&mut self, sequence: u64, kind: OperationKind, first: Option<u64>, at: usize) {
        self.last = sequence;
        if let Some((held_first, commit, ops)) = self.held.take() {
            if kind == OperationKind::Abort && first == Some(held_first) {
                self.out
                    .extend(ops.iter().map(|s| (*s, RecordState::Withdrawn)));
                self.boundary = sequence;
                self.aborts += 1;
                return;
            }
            self.out
                .extend(ops.iter().map(|s| (*s, RecordState::Committed)));
            self.boundary = commit;
        }
        // A framing record whose payload does not decode is refused, as one
        // naming the wrong first is.
        if kind.is_framing() && !first.is_some_and(|f| f == self.boundary + 1 && f < sequence) {
            self.refused = true;
            return;
        }
        match kind {
            OperationKind::Commit => {
                let ops = self.open.drain(..).map(|(s, _)| s).collect();
                self.held = Some((first.unwrap(), sequence, ops));
            }
            OperationKind::Abort => {
                let ops: Vec<u64> = self.open.drain(..).map(|(s, _)| s).collect();
                self.out
                    .extend(ops.iter().map(|s| (*s, RecordState::Withdrawn)));
                self.boundary = sequence;
                self.aborts += 1;
            }
            _ => self.open.push((sequence, at)),
        }
    }

    fn names(&self) -> Vec<u64> {
        match &self.held {
            Some((first, commit, _)) => vec![*first, commit + 1],
            None => vec![self.boundary + 1],
        }
    }

    /// Settle what is left at the end: a held commit settles.
    fn settle_held(&mut self) {
        if let Some((_, commit, ops)) = self.held.take() {
            self.out
                .extend(ops.iter().map(|s| (*s, RecordState::Committed)));
            self.boundary = commit;
        }
    }
}

/// Mutate the body of the corpus's last segment alone: hostile values and
/// random bytes not repaired, the file cut short at random or at a known
/// record, bytes appended, a known record appended again, a known record's
/// sequence restamped, and a run zeroed.
fn mutate_last_segment(rng: &mut Rng, built: &Built) -> (Vec<u8>, Vec<Whole>) {
    let last = built.segments.len() - 1;
    let base_bytes = &built.segments[last];
    let table: Vec<&Known> = built.known.iter().filter(|k| k.segment == last).collect();
    let mut blob = base_bytes.clone();
    let mut wholes = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let len = blob.len();
        let body = len.saturating_sub(SEGMENT_HEADER_BYTES);
        match rng.below(11) {
            0..=2 => {
                if body > 0 {
                    let width = [1usize, 2, 4, 8][rng.below(4)];
                    let at = SEGMENT_HEADER_BYTES + rng.below(body);
                    let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                    let end = (at + width).min(len);
                    blob[at..end].copy_from_slice(&value[..end - at]);
                }
            }
            3 => {
                if body > 0 {
                    let at = SEGMENT_HEADER_BYTES + rng.below(body);
                    blob[at] = rng.byte();
                }
            }
            4 => blob.truncate(rng.below(len + 1)),
            5 => {
                let extra = 1 + rng.below(64);
                let value = if rng.below(2) == 0 { 0 } else { rng.byte() };
                blob.extend(std::iter::repeat_n(value, extra));
            }
            6 => {
                let known = table[rng.below(table.len())];
                if known.at + known.len <= len {
                    let value = HOSTILE[rng.below(HOSTILE.len())];
                    blob[known.at..known.at + 8].copy_from_slice(&value.to_le_bytes());
                    let payload_len = known.len - JOURNAL_RECORD_OVERHEAD_BYTES;
                    restamp_and_note(&mut blob, known.at, payload_len, true, &mut wholes);
                }
            }
            7 => {
                let known = table[rng.below(table.len())];
                if known.at + known.len <= len {
                    wholes.push(Whole {
                        at: blob.len(),
                        bytes: base_bytes[known.at..known.at + known.len].to_vec(),
                        sequence: known.sequence,
                        kind: known.kind,
                        first: known.first,
                    });
                    blob.extend_from_slice(&base_bytes[known.at..known.at + known.len]);
                }
            }
            8 => {
                let known = table[rng.below(table.len())];
                blob.truncate(known.at.min(len));
            }
            _ => {
                if body > 0 {
                    let at = SEGMENT_HEADER_BYTES + rng.below(body);
                    let end = (at + 1 + rng.below(96)).min(len);
                    blob[at..end].fill(0);
                }
            }
        }
    }
    (blob, wholes)
}

/// The oracle for a journal of segments whose last segment's body was
/// mutated, derived from the corpus's known records by byte comparison.
fn segment_oracle(built: &Built, expected: &Expected, blob: &[u8], wholes: &[Whole]) -> Predicted {
    let last = built.segments.len() - 1;
    let header = &built.segments[last][..SEGMENT_HEADER_BYTES];
    let mut model = Model::at_checkpoint(expected.base.sequence);
    for known in built
        .known
        .iter()
        .filter(|k| k.segment > 0 && k.segment < last)
    {
        model.take(known.sequence, known.kind, known.first, known.at);
    }
    let header_whole =
        blob.len() >= SEGMENT_HEADER_BYTES && blob[..SEGMENT_HEADER_BYTES] == header[..];
    if !header_whole {
        if blob.len() > SEGMENT_HEADER_BYTES {
            return Predicted::Header;
        }
        // A last segment no longer than its header, whose header does not
        // read, is a creation a crash interrupted, which the journal would
        // begin next.
        model.settle_held();
        return Predicted::Read {
            records: model.out,
            damage: None,
            good_bytes: built.segments[last - 1].len() as u64,
            next_sequence: built.numbers[last],
            aborts: model.aborts,
            torn_creation: Some(built.numbers[last]),
            regions: 0,
        };
    }
    let mut all: Vec<Whole> = built
        .known
        .iter()
        .filter(|k| k.segment == last)
        .map(|k| Whole {
            at: k.at,
            bytes: built.segments[last][k.at..k.at + k.len].to_vec(),
            sequence: k.sequence,
            kind: k.kind,
            first: k.first,
        })
        .collect();
    all.extend(wholes.iter().cloned());
    let intact = |w: &Whole| {
        blob.len() >= w.at + w.bytes.len() && blob[w.at..w.at + w.bytes.len()] == w.bytes[..]
    };
    let mut at = SEGMENT_HEADER_BYTES;
    let mut next = built.numbers[last];
    let mut damage = None;
    let mut good_bytes = blob.len() as u64;
    loop {
        if let Some(w) = all
            .iter()
            .find(|w| w.at == at && w.sequence == next && intact(w))
        {
            model.take(w.sequence, w.kind, w.first, w.at);
            if model.refused {
                return Predicted::Refused;
            }
            at += w.bytes.len();
            next = w.sequence + 1;
            continue;
        }
        if at >= blob.len() {
            break;
        }
        let names = model.names();
        let abort = all
            .iter()
            .filter(|w| {
                w.at > at
                    && w.kind == OperationKind::Abort
                    && w.sequence >= next
                    && w.first
                        .is_some_and(|f| names.contains(&f) && f < w.sequence)
                    && intact(w)
            })
            .min_by_key(|w| w.at)
            .cloned();
        if let Some(abort) = abort {
            let first = abort.first.unwrap();
            if let Some((held_first, commit, ops)) = model.held.take() {
                let state = if first == held_first {
                    RecordState::Withdrawn
                } else {
                    model.boundary = commit;
                    RecordState::Committed
                };
                model.out.extend(ops.iter().map(|s| (*s, state)));
            }
            let open: Vec<u64> = model.open.drain(..).map(|(s, _)| s).collect();
            model
                .out
                .extend(open.iter().map(|s| (*s, RecordState::Withdrawn)));
            let mut floor = next.max(first);
            let mut region: Vec<&Whole> = all
                .iter()
                .filter(|w| {
                    w.at >= at
                        && w.at < abort.at
                        && w.sequence >= floor
                        && w.sequence < abort.sequence
                        && intact(w)
                })
                .collect();
            region.sort_by_key(|w| w.at);
            let mut end = at;
            for w in region {
                if w.at < end || w.sequence < floor || w.at + w.bytes.len() > abort.at {
                    continue;
                }
                if !w.kind.is_framing() {
                    model.out.push((w.sequence, RecordState::Withdrawn));
                }
                floor = w.sequence + 1;
                end = w.at + w.bytes.len();
            }
            model.boundary = abort.sequence;
            model.last = abort.sequence;
            model.aborts += 1;
            model.regions += 1;
            at = abort.at + abort.bytes.len();
            next = abort.sequence + 1;
            continue;
        }
        let later = all
            .iter()
            .any(|w| w.at > at && w.sequence > next && intact(w));
        good_bytes = at as u64;
        damage = Some(if later {
            JournalDamage::Corrupt {
                at: at as u64,
                sequence: next,
                detail: String::new(),
            }
        } else {
            JournalDamage::TornTail {
                at: at as u64,
                sequence: next,
            }
        });
        break;
    }
    let corrupt = matches!(damage, Some(JournalDamage::Corrupt { .. }));
    let mut next_sequence = match &damage {
        Some(JournalDamage::TornTail { sequence, .. }) => *sequence,
        _ => model.last + 1,
    };
    if !corrupt {
        model.settle_held();
        if let Some(&(sequence, offset)) = model.open.first() {
            damage = Some(JournalDamage::TornTail {
                at: offset as u64,
                sequence,
            });
            good_bytes = offset as u64;
            next_sequence = sequence;
        }
    }
    model.out.sort_by_key(|(s, _)| *s);
    Predicted::Read {
        records: model.out,
        damage,
        good_bytes,
        next_sequence,
        aborts: model.aborts,
        torn_creation: None,
        regions: model.regions,
    }
}

/// Read the corpus with its last segment replaced by `blob`.
fn read_mutated<'a>(
    built: &'a Built,
    files: &'a [String],
    blob: &'a [u8],
    expected: &Expected,
) -> Result<JournalContents<'a>, Error> {
    let last = built.segments.len() - 1;
    let segments: Vec<SegmentBytes<'a>> = (0..built.segments.len())
        .map(|i| SegmentBytes {
            number: built.numbers[i],
            file: &files[i],
            bytes: if i == last { blob } else { &built.segments[i] },
        })
        .collect();
    read_segments(&segments, expected, "j.zdbjournal")
}

/// The calls a reading marks begin at its first record, rise, and lie
/// within its records.
fn calls_hold(contents: &JournalContents<'_>) {
    let calls = &contents.calls;
    if contents.records.is_empty() {
        assert!(calls.is_empty(), "{calls:?}");
        return;
    }
    assert_eq!(calls.first(), Some(&0), "{calls:?}");
    assert!(calls.windows(2).all(|pair| pair[0] < pair[1]), "{calls:?}");
    assert!(calls.iter().all(|&start| start < contents.records.len()));
}

/// **The property, at format version 2.** Every mutation of any segment of a
/// valid journal, its header included, reads or is refused with the reader's
/// own error, none panics, and what reads holds together: every record above
/// the checkpoint and in order, its calls marked where they begin, the next
/// sequence above the checkpoint, and the good bytes within the last segment
/// where the journal is not corrupt.
#[test]
fn no_mutation_of_a_journal_of_segments_panics_the_reader() {
    let (built, expected) = segment_corpus();
    let mut rng = Rng(0x5eed_219a_e000_0002);
    let cases = budget(4_000);
    let (mut read, mut refused, mut cuts) = (0usize, 0usize, 0usize);
    for _ in 0..cases {
        let mut mutated = built.clone();
        for _ in 0..1 + rng.below(3) {
            let s = rng.below(mutated.segments.len());
            let choice = rng.below(8);
            if choice == 7 && mutated.segments.len() > 1 && rng.below(4) == 0 {
                mutated.segments.remove(s);
                mutated.numbers.remove(s);
                continue;
            }
            let segment = &mut mutated.segments[s];
            let len = segment.len();
            match choice {
                0 | 1 => {
                    if len > 0 {
                        let at = rng.below(len);
                        let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                        let end = (at + [1usize, 2, 4, 8][rng.below(4)]).min(len);
                        segment[at..end].copy_from_slice(&value[..end - at]);
                    }
                }
                2 => {
                    if len >= SEGMENT_HEADER_BYTES {
                        let at = [8usize, 12, 16, 20, 24, 40, 56, 64, 72, 88, 100][rng.below(11)];
                        let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                        segment[at..at + 8].copy_from_slice(&value);
                        if rng.below(4) != 0 {
                            restamp_segment_header(segment);
                        }
                    }
                }
                3 => segment.truncate(rng.below(len + 1)),
                4 => {
                    let extra = 1 + rng.below(80);
                    segment.extend(std::iter::repeat_n(rng.byte(), extra));
                }
                5 | 6 => {
                    let known: Vec<&Known> =
                        built.known.iter().filter(|k| k.segment == s).collect();
                    if !known.is_empty() {
                        let k = known[rng.below(known.len())];
                        if k.at + k.len <= segment.len() {
                            let field = [0usize, 8, 12, 14, 16][rng.below(5)];
                            let room = k.len - JOURNAL_RECORD_CHECKSUM_BYTES - field;
                            let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                            let width = room.min(8);
                            segment[k.at + field..k.at + field + width]
                                .copy_from_slice(&value[..width]);
                            restamp_record(segment, k.at, k.len - JOURNAL_RECORD_OVERHEAD_BYTES);
                        }
                    }
                }
                _ => {
                    if len > 0 {
                        let at = rng.below(len);
                        let end = (at + 1 + rng.below(64)).min(len);
                        segment[at..end].fill(0);
                    }
                }
            }
        }
        let files = mutated.files();
        match mutated.read(&files, &expected) {
            Ok(contents) => {
                read += 1;
                assert!(contents.next_sequence > expected.base.sequence);
                let mut last = expected.base.sequence;
                for record in &contents.records {
                    assert!(record.record.sequence > last, "{:?}", contents.records);
                    last = record.record.sequence;
                    assert!(!record.record.kind.is_framing());
                }
                assert_eq!(
                    contents.committed() + contents.withdrawn(),
                    contents.records.len()
                );
                calls_hold(&contents);
                let corrupt = matches!(contents.damage, Some(JournalDamage::Corrupt { .. }));
                if let (Some(active), false) = (contents.active, corrupt) {
                    let i = mutated.numbers.iter().position(|n| *n == active).unwrap();
                    assert!(contents.good_bytes <= mutated.segments[i].len() as u64);
                    assert!(contents.good_bytes >= SEGMENT_HEADER_BYTES as u64);
                }
                if let Some(JournalDamage::TornTail { at, .. }) = contents.damage {
                    assert_eq!(at, contents.good_bytes);
                }
                if contents.torn_creation.is_some() {
                    assert_eq!(contents.active, None);
                }
            }
            Err(
                Error::JournalHeaderInvalid { .. }
                | Error::JournalSegmentsInvalid { .. }
                | Error::JournalNotThisCollection { .. }
                | Error::JournalNotThisCheckpoint { .. }
                | Error::JournalStartsAfterCheckpoint { .. },
            ) => refused += 1,
            Err(other) => panic!("unexpected error {other:?}"),
        }
        // The same journal read for a repair is refused only as not the
        // directory's, and where it cuts, the journal it keeps reads clean
        // with the records it kept.
        match read_for_repair(&mutated, &files, &expected) {
            Ok(contents) => {
                calls_hold(&contents);
                holds_its_cut(&mutated, &expected, &contents);
                cuts += usize::from(contents.cut.is_some());
            }
            Err(
                Error::JournalSegmentsInvalid { .. }
                | Error::JournalNotThisCollection { .. }
                | Error::JournalNotThisCheckpoint { .. }
                | Error::JournalStartsAfterCheckpoint { .. },
            ) => {}
            Err(other) => panic!("a reading for a repair gave {other:?}"),
        }
    }
    assert!(read * 4 > cases, "only {read} of {cases} read");
    assert!(refused > 0, "no case was refused");
    assert!(
        cuts * 10 > cases,
        "only {cuts} of {cases} were cut for a repair"
    );
}

/// **The classification, at format version 2.** Every mutation of the last
/// segment's body is classified as the corpus's known records say it should
/// be, withdrawn regions among the cases, with the same records in the same
/// states, the same damage at the same byte and sequence, the same good
/// bytes and the same next sequence.
#[test]
fn the_segment_classification_matches_an_oracle_under_mutation() {
    let (built, expected) = segment_corpus();
    let files = built.files();
    let original = built.read(&files, &expected).unwrap();
    assert!(original.withdrawn() > 0 && original.aborts == 3);
    assert_eq!(
        segment_oracle(&built, &expected, &built.segments[2], &[]),
        Predicted::Read {
            records: states(&original),
            damage: original.damage.clone(),
            good_bytes: original.good_bytes,
            next_sequence: original.next_sequence,
            aborts: original.aborts,
            torn_creation: None,
            regions: 0,
        }
    );
    let mut rng = Rng(0x5eed_219a_c000_0002);
    let cases = budget(4_000);
    let (mut torn, mut corrupt, mut whole, mut regions) = (0usize, 0usize, 0usize, 0usize);
    for case in 0..cases {
        let (blob, wholes) = mutate_last_segment(&mut rng, &built);
        let predicted = segment_oracle(&built, &expected, &blob, &wholes);
        let found = read_mutated(&built, &files, &blob, &expected);
        match (&predicted, found) {
            (Predicted::Header, Err(Error::JournalHeaderInvalid { .. })) => {}
            (Predicted::Refused, Err(Error::JournalSegmentsInvalid { .. })) => {}
            (
                Predicted::Read {
                    records,
                    damage,
                    good_bytes,
                    next_sequence,
                    aborts,
                    torn_creation,
                    regions: taken,
                },
                Ok(contents),
            ) => {
                match (&contents.damage, damage) {
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
                        continue;
                    }
                    (
                        Some(JournalDamage::TornTail { .. }),
                        Some(JournalDamage::TornTail { .. }),
                    ) => torn += 1,
                    (None, None) => whole += 1,
                    _ => panic!(
                        "case {case}: the reader said {:?} and the oracle {damage:?}",
                        contents.damage
                    ),
                }
                regions += usize::from(*taken > 0);
                assert_eq!(&states(&contents), records, "case {case}");
                assert_eq!(&contents.damage, damage, "case {case}");
                assert_eq!(contents.good_bytes, *good_bytes, "case {case}");
                assert_eq!(contents.next_sequence, *next_sequence, "case {case}");
                assert_eq!(contents.aborts, *aborts, "case {case}");
                assert_eq!(contents.torn_creation, *torn_creation, "case {case}");
            }
            (want, found) => panic!(
                "case {case}: the oracle said {want:?} and the reader {:?}",
                found.map(|c| (c.damage.clone(), states(&c)))
            ),
        }
    }
    assert!(
        torn * 20 > cases && corrupt * 50 > cases && whole > 0 && regions > 0,
        "{torn} torn, {corrupt} corrupt, {whole} whole, {regions} with a region, of {cases}"
    );
}

/// A journal of one segment whose calls each carry one real operation at
/// width 4 and are committed, a version 2 training record of each scheme
/// among them, with the place and length of every record.
fn operation_corpus() -> Built {
    use crate::operation::Fitted;
    let operations = vec![
        Operation::Insert {
            id: "a".into(),
            internal_id: 1,
            level: 0,
            vector: vec![1.0, 0.0, -1.0, 0.5],
            metadata: serde_json::Map::new(),
            sparse: None,
        },
        Operation::Train {
            completed_at: "2026-10-21T00:00:00+00:00".into(),
            fitted: Some(Fitted::Codebook {
                subvectors: 2,
                centroids: 4,
                width: 2,
                values: (0..16).map(|v| v as f32 * 0.25).collect(),
                calibration: Some(r#"{"fetch":12}"#.into()),
            }),
        },
        Operation::Intern {
            term_id: 0,
            term: "fox".into(),
        },
        Operation::Train {
            completed_at: "t".into(),
            fitted: Some(Fitted::Scales {
                values: vec![0.5, 1.0, 0.25, 2.0],
            }),
        },
        Operation::Remove {
            ids: vec!["a".into()],
        },
        Operation::AddMetadata {
            pairs: vec![("k".into(), "v".into())],
        },
    ];
    let mut built = Built {
        numbers: vec![1],
        segments: Vec::new(),
        known: Vec::new(),
    };
    let mut bytes = encode_segment_header(&segment_header(base(LIVE, 0), 1)).to_vec();
    let mut sequence = 1u64;
    let (mut payload, mut record) = (Vec::new(), Vec::new());
    for operation in &operations {
        operation.encode(&mut payload);
        encode_journal_record(sequence, operation.kind(), &payload, &mut record);
        built.known.push(Known {
            segment: 0,
            at: bytes.len(),
            len: record.len(),
            sequence,
            kind: operation.kind(),
            first: None,
        });
        bytes.extend_from_slice(&record);
        let first = sequence;
        sequence += 1;
        Operation::Commit { first }.encode(&mut payload);
        encode_journal_record(sequence, OperationKind::Commit, &payload, &mut record);
        built.known.push(Known {
            segment: 0,
            at: bytes.len(),
            len: record.len(),
            sequence,
            kind: OperationKind::Commit,
            first: Some(first),
        });
        bytes.extend_from_slice(&record);
        sequence += 1;
    }
    built.segments.push(bytes);
    built
}

/// **The decoder under the reader, at format version 2.** Every mutation of
/// a journal whose records carry real operations, a version 2 training
/// record of each scheme among them, reads or is refused with the reader's
/// own error, for a load and for a repair, and every record either reading
/// hands over decodes or is refused with the decoder's own error, its counts
/// held to its payload before anything is sized.
#[test]
fn no_mutation_of_a_journal_of_operations_panics_the_decoder() {
    use crate::operation::Fitted;
    let built = operation_corpus();
    let expected = expected_at(0);
    let mut rng = Rng(0x5eed_219b_d000_0002);
    let cases = budget(4_000);
    let (mut decoded, mut trained, mut refused) = (0usize, 0usize, 0usize);
    for _ in 0..cases {
        let mut mutated = built.clone();
        for _ in 0..1 + rng.below(3) {
            let segment = &mut mutated.segments[0];
            let len = segment.len();
            match rng.below(6) {
                0..=2 => {
                    // A value written inside a record's payload, the record
                    // restamped, so the reader hands the payload over.
                    let k = &built.known[rng.below(built.known.len())];
                    let payload = k.len - JOURNAL_RECORD_OVERHEAD_BYTES;
                    if payload > 0 && k.at + k.len <= segment.len() {
                        let start = k.at + JOURNAL_RECORD_HEADER_BYTES;
                        let at = start + rng.below(payload);
                        let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                        let end = (at + [1usize, 2, 4, 8][rng.below(4)]).min(start + payload);
                        segment[at..end].copy_from_slice(&value[..end - at]);
                        restamp_record(segment, k.at, payload);
                    }
                }
                3 => {
                    if len > 0 {
                        let at = rng.below(len);
                        segment[at] = rng.byte();
                    }
                }
                4 => segment.truncate(rng.below(len + 1)),
                _ => {
                    let extra = 1 + rng.below(40);
                    segment.extend(std::iter::repeat_n(rng.byte(), extra));
                }
            }
        }
        let files = mutated.files();
        for reading in [
            mutated.read(&files, &expected),
            read_for_repair(&mutated, &files, &expected),
        ] {
            let contents = match reading {
                Ok(contents) => contents,
                Err(
                    Error::JournalHeaderInvalid { .. }
                    | Error::JournalSegmentsInvalid { .. }
                    | Error::JournalNotThisCollection { .. }
                    | Error::JournalNotThisCheckpoint { .. }
                    | Error::JournalStartsAfterCheckpoint { .. },
                ) => {
                    refused += 1;
                    continue;
                }
                Err(other) => panic!("unexpected error {other:?}"),
            };
            for live in &contents.records {
                match Operation::decode(&live.record, 4, "f") {
                    Ok(Operation::Train {
                        fitted: Some(fitted),
                        ..
                    }) => {
                        decoded += 1;
                        trained += 1;
                        let values = match &fitted {
                            Fitted::Codebook { values, .. } | Fitted::Scales { values } => {
                                values.len()
                            }
                        };
                        assert!(values * 4 <= live.record.payload.len());
                    }
                    Ok(_) => decoded += 1,
                    Err(Error::JournalRecordInvalid { .. }) => {}
                    Err(other) => panic!("unexpected decode error {other:?}"),
                }
            }
        }
    }
    assert!(
        decoded > cases,
        "only {decoded} records decoded over {cases} cases"
    );
    assert!(
        trained * 4 > cases,
        "only {trained} training records decoded over {cases} cases"
    );
    assert!(refused > 0, "no case was refused");
}
