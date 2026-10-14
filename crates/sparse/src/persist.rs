//! The one artefact, live records only, ids kept.
//!
//! The payload carries every live record's id and vector in id order, so a
//! restore replays it through `insert` and the lists come back sorted without
//! a sort. Dead records and their postings are not written, which is why a
//! restored index is compact whatever the policy the saved one ran under.
//!
//! # Layout
//!
//! The payload sits inside the frame `zeusdb_vector_core::frame` describes,
//! under the kind `SparsePostings`, with the frame's `entries` field holding
//! the live record count. Every field is little-endian at a fixed width.
//!
//! ```text
//! slots    u32   record slots the index held, being one past the largest id
//! live     u32   live records, which is what follows
//! per live record, in increasing id
//!   id     u32
//!   nnz    u32
//!   dims   nnz * u32, strictly increasing
//!   values nnz * f32, finite, and positive under term frequency weighting
//! ```
//!
//! # What bounds each length
//!
//! `slots` is held to the largest internal id the collection holds, which
//! the collection reads from its own mappings, plus one. `live` is held to
//! `slots` and to the payload's length at eight bytes a record, and to the
//! frame's `entries`. Each `nnz` is held to what is left of the payload at
//! eight bytes a nonzero. Each `id` is held below `slots` and above the one
//! before it. Nothing is sized from a field before the field is held to
//! something the file has earned.
//!
//! # Ids and positions
//!
//! The index keys every record by its position and the artefact names every
//! record by its internal id, so the save is handed each position's internal
//! id and the restore each internal id's position, being the collection's
//! mappings in position order. Positions follow internal id order, so the
//! records the save writes in position order are in increasing internal id,
//! and the restore finds each record's position by a step forward.

use zeusdb_vector_core::{
    frame_begin, frame_finish, read_artefact, unframe, write_artefact, ArtefactRecord, Bounds, Dir,
    Error, FrameEncoding, FrameKind, Inventory, Ledger, Position, PositionWalk, RecordId,
    SparseRef,
};

use crate::index::{PostingsIndex, SparseConfig};

/// What the artefact holds, for a message.
const CONTENTS: &str = "the sparse postings";

/// Bytes a live record costs at least, being its id and its count.
const RECORD_HEADER_BYTES: usize = 8;

/// Bytes one nonzero costs, being its dimension and its value.
const NONZERO_BYTES: usize = 8;

pub(crate) fn artefact_name(prefix: &str) -> String {
    format!("{prefix}postings.zdbsparse")
}

/// The artefact's bytes, frame included, built in one buffer, each record
/// named by its position, as an index filled on its own keys them.
#[cfg(test)]
pub(crate) fn encode(index: &PostingsIndex) -> Vec<u8> {
    encode_with(index, &|at| at as u32)
}

/// The artefact's bytes, frame included, built in one buffer, the record at
/// each position named by the internal id `id_of` gives it.
pub(crate) fn encode_with(index: &PostingsIndex, id_of: &dyn Fn(usize) -> u32) -> Vec<u8> {
    let mut payload = frame_begin(
        FrameKind::SparsePostings,
        FrameEncoding::Engine,
        8 + index.live_nnz * NONZERO_BYTES + index.live * RECORD_HEADER_BYTES,
    );
    payload.extend((index.slot_count() as u32).to_le_bytes());
    payload.extend((index.live as u32).to_le_bytes());
    for (at, slot) in index.records.iter().enumerate() {
        if !slot.held() || index.dead.contains(at) {
            continue;
        }
        let v = index.forward(*slot);
        payload.extend(id_of(at).to_le_bytes());
        payload.extend((v.dims.len() as u32).to_le_bytes());
        for d in v.dims {
            payload.extend(d.to_le_bytes());
        }
        for w in v.values {
            payload.extend(w.to_le_bytes());
        }
    }
    frame_finish(payload, index.live as u64)
}

pub(crate) fn write(
    index: &PostingsIndex,
    prefix: &str,
    dir: &dyn Dir,
    ledger: &mut dyn Ledger,
    id_of: &dyn Fn(usize) -> u32,
) -> Result<(), Error> {
    let bytes = encode_with(index, id_of);
    let name = artefact_name(prefix);
    // The frame carries a checksum over the payload and the reader verifies
    // it, so the manifest records the length alone, as it does for the
    // graph dump.
    write_artefact(dir, &name, &bytes)?;
    ledger.record(
        &name,
        ArtefactRecord {
            bytes: bytes.len() as u64,
            checksum: None,
        },
    );
    Ok(())
}

pub(crate) fn restore(
    config: &SparseConfig,
    prefix: &str,
    dir: &dyn Dir,
    inventory: &dyn Inventory,
    bounds: &Bounds,
) -> Result<PostingsIndex, Error> {
    let name = artefact_name(prefix);
    let bytes = read_artefact(dir, &name, inventory, CONTENTS, bounds.max_bytes)?;
    decode(&bytes, config, bounds, &name)
}

/// An index from the artefact's bytes, refusing each way they can be wrong
/// by name. `file` names the artefact in the message.
///
/// Each record is filed at the position `bounds.ids_by_position` gives its
/// internal id. A record whose id the list does not hold has no position, so
/// it is held to its vector's rules as every record is and is not filed, and
/// once the whole payload has been read the first such id is refused as
/// [`Error::SparseRecordUnmapped`], with the space left for the caller to
/// name. Nothing is sized from the slot count, which is the largest id and
/// not a count of records.
pub(crate) fn decode(
    bytes: &[u8],
    config: &SparseConfig,
    bounds: &Bounds,
    file: &str,
) -> Result<PostingsIndex, Error> {
    let framed = unframe(bytes, FrameKind::SparsePostings, file)?;
    let corrupt = |detail: String| Error::DecodeFailed {
        file: file.to_string(),
        error: detail,
    };
    let payload = framed.payload;
    let mut reader = Reader {
        bytes: payload,
        at: 0,
    };
    let mut take = |n: usize| -> Result<&[u8], Error> { reader.take(n, file) };
    let u32_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let f32_of = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let slots = u32_of(take(4)?) as usize;
    let live = u32_of(take(4)?) as usize;
    if slots > bounds.max_records.saturating_add(1) {
        return Err(corrupt(format!(
            "the payload declares {} record slots and the collection has issued ids up to {}",
            slots, bounds.max_records
        )));
    }
    if live > slots {
        return Err(corrupt(format!(
            "the payload declares {} live records in {} slots",
            live, slots
        )));
    }
    if live as u64 != framed.entries {
        return Err(corrupt(format!(
            "the payload declares {} live records and the frame counts {}",
            live, framed.entries
        )));
    }
    // Each live record costs at least eight bytes, so the count is bounded by
    // the payload before anything is sized from it.
    if live.saturating_mul(RECORD_HEADER_BYTES) > payload.len().saturating_sub(8) {
        return Err(corrupt(format!(
            "the payload declares {} live records and holds {} bytes",
            live,
            payload.len()
        )));
    }
    let mut index = PostingsIndex::new(config.clone());
    // The positions of the ids below the slot count, which is as far as a
    // record of this payload can reach.
    let positions = bounds.ids_by_position;
    let reach = positions.partition_point(|&id| (id as usize) < slots);
    index.records.reserve(reach);
    index.lengths.reserve(reach);
    let mut walk = PositionWalk::new(positions);
    let mut unmapped: Option<u32> = None;
    let mut dims: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let mut last_id: Option<u32> = None;
    // Bytes the records read so far have consumed, kept beside the cursor
    // so the remaining length is known without reaching past the closure.
    let mut consumed = 8usize;
    for _ in 0..live {
        let id = u32_of(take(4)?);
        let nnz = u32_of(take(4)?) as usize;
        if id as usize >= slots {
            return Err(corrupt(format!(
                "record {} is beyond the {} slots the payload declares",
                id, slots
            )));
        }
        if last_id.is_some_and(|last| id <= last) {
            return Err(corrupt(format!(
                "record {} follows record {}, and the payload is written in increasing id order",
                id,
                last_id.unwrap_or(0)
            )));
        }
        last_id = Some(id);
        // Held to what is left rather than to the whole payload, so a run of
        // records cannot each claim the file's full length.
        consumed += RECORD_HEADER_BYTES;
        let remaining = payload.len() - consumed;
        if nnz.saturating_mul(NONZERO_BYTES) > remaining {
            return Err(corrupt(format!(
                "record {} declares {} nonzeros and {} bytes remain",
                id, nnz, remaining
            )));
        }
        dims.clear();
        values.clear();
        for _ in 0..nnz {
            dims.push(u32_of(take(4)?));
        }
        for _ in 0..nnz {
            values.push(f32_of(take(4)?));
        }
        consumed += nnz * NONZERO_BYTES;
        let vector = SparseRef {
            dims: &dims,
            values: &values,
        };
        match walk.position_of(id as usize) {
            Some(at) => {
                index
                    .insert_record(Position::from_slot(at), vector)
                    .map_err(|e| corrupt(format!("record {}: {}", id, e)))?;
                index.count_slot(RecordId(id));
            }
            None => {
                config
                    .weighting
                    .validate_record(vector)
                    .map_err(|e| corrupt(format!("record {}: {}", id, e)))?;
                unmapped.get_or_insert(id);
            }
        }
    }
    if consumed != payload.len() {
        return Err(corrupt(format!(
            "{} bytes follow the last record the payload declares",
            payload.len() - consumed
        )));
    }
    if let Some(id) = unmapped {
        return Err(Error::SparseRecordUnmapped {
            space: String::new(),
            id,
        });
    }
    index.calibrate();
    Ok(index)
}

/// A cursor over the payload's bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    /// The next `n` bytes, or the refusal where fewer than `n` remain.
    fn take(&mut self, n: usize, file: &str) -> Result<&'a [u8], Error> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| Error::DecodeFailed {
                file: file.to_string(),
                error: "the payload ends before the record it declares".to_string(),
            })?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Unlink, Weighting};
    use std::collections::HashMap;
    use zeusdb_vector_core::{
        frame_fuzz, FsDir, Persist, Prepared, Restore, SparseVector, VectorIndex,
        FRAME_HEADER_BYTES,
    };

    #[derive(Default)]
    struct Manifest(HashMap<String, ArtefactRecord>);

    impl Ledger for Manifest {
        fn record(&mut self, name: &str, record: ArtefactRecord) {
            self.0.insert(name.to_string(), record);
        }
    }

    impl Inventory for Manifest {
        fn recorded(&self, name: &str) -> Option<ArtefactRecord> {
            self.0.get(name).copied()
        }
    }

    fn bounds() -> Bounds<'static> {
        Bounds {
            min_records: 0,
            max_records: 1 << 20,
            max_bytes: 1 << 30,
            ids_by_position: crate::identity_positions(1 << 20),
        }
    }

    fn filled() -> PostingsIndex {
        let mut index = PostingsIndex::new(SparseConfig {
            unlink: Unlink::Strand,
            ..SparseConfig::default()
        });
        for id in 1..=50u32 {
            let dims: Vec<u32> = (0..5).map(|j| id * 3 + j * 11).collect();
            let values: Vec<f32> = dims.iter().map(|d| *d as f32 * 0.5).collect();
            let v = SparseVector { dims, values };
            index
                .insert(Position(id), RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        index
    }

    /// A save and a restore round trip every live record under its own id,
    /// leave the dead ones behind, and are recorded in the ledger by length
    /// alone.
    #[test]
    fn a_round_trip_keeps_every_live_record_and_drops_the_dead() {
        let mut index = filled();
        for id in [3u32, 17, 50] {
            index.remove(Position(id)).unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = Manifest::default();
        index
            .write("spaces/s/", &FsDir::new(dir.path()), &mut manifest)
            .unwrap();
        assert_eq!(
            index.artefact_names("spaces/s/"),
            vec!["spaces/s/postings.zdbsparse".to_string()]
        );
        let recorded = manifest.recorded("spaces/s/postings.zdbsparse").unwrap();
        assert!(recorded.checksum.is_none());
        assert_eq!(
            recorded.bytes as usize,
            std::fs::metadata(dir.path().join("spaces/s/postings.zdbsparse"))
                .unwrap()
                .len() as usize
        );

        let restored = PostingsIndex::restore(
            index.config(),
            "spaces/s/",
            &FsDir::new(dir.path()),
            &manifest,
            &bounds(),
        )
        .unwrap();
        assert_eq!(restored.len(), 47);
        assert_eq!(restored.stranded(), 0);
        assert_eq!(restored.postings_total(), 47 * 5);
        // The saved index kept its dead records' lists under `Strand`, so
        // its largest dimension may exceed what the live records carry.
        assert!(restored.max_dim() <= index.max_dim());
        assert_eq!(restored.max_dim(), Some(49 * 3 + 4 * 11));
        for id in 1..=50u32 {
            assert_eq!(restored.recover(Position(id)), index.recover(Position(id)));
        }
        // An empty index writes a frame around an eight byte payload.
        let empty = PostingsIndex::new(SparseConfig::default());
        let bytes = encode(&empty);
        assert_eq!(bytes.len(), 8 + zeusdb_vector_core::FRAME_OVERHEAD_BYTES);
        let back = decode(&bytes, empty.config(), &bounds(), "postings").unwrap();
        assert_eq!(back.len(), 0);
        assert_eq!(back.max_dim(), None);
    }

    /// Every way the artefact can be damaged is refused with a decode error
    /// naming the file, and a manifest that does not name it is refused
    /// before it is read.
    #[test]
    fn every_damage_is_refused_before_the_index_is_built() {
        let index = filled();
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = Manifest::default();
        index
            .write("", &FsDir::new(dir.path()), &mut manifest)
            .unwrap();
        let path = dir.path().join("postings.zdbsparse");
        let good = std::fs::read(&path).unwrap();
        let h = FRAME_HEADER_BYTES;

        let refused = |bytes: &[u8]| {
            std::fs::write(&path, bytes).unwrap();
            let mut m = Manifest::default();
            m.record(
                "postings.zdbsparse",
                ArtefactRecord {
                    bytes: bytes.len() as u64,
                    checksum: None,
                },
            );
            PostingsIndex::restore(index.config(), "", &FsDir::new(dir.path()), &m, &bounds())
        };
        let message = |bytes: &[u8]| match refused(bytes) {
            Err(Error::DecodeFailed { error, .. }) => error,
            other => panic!("expected a decode failure, got {:?}", other.map(|_| ())),
        };

        // Wrong magic, caught by the frame.
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(message(&bad).contains("frame magic"));
        // Truncated inside a record, caught by the frame's length agreement.
        assert!(message(&good[..good.len() - 3]).contains("file holds"));
        // Trailing payload bytes, with the frame repaired around them.
        let mut long = good.clone();
        long.insert(long.len() - 16, 0);
        frame_fuzz::repair_header(&mut long);
        frame_fuzz::repair_trailer(&mut long);
        assert!(message(&long).contains("bytes follow the last record"));
        // More live records than slots.
        let mut bad = good.clone();
        bad[h + 4..h + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        frame_fuzz::repair_trailer(&mut bad);
        assert!(message(&bad).contains("live records in"));
        // A live count the frame disagrees with.
        let mut bad = good.clone();
        bad[h + 4..h + 8].copy_from_slice(&49u32.to_le_bytes());
        frame_fuzz::repair_trailer(&mut bad);
        assert!(message(&bad).contains("the frame counts"));
        // A live count the payload cannot hold, with the frame and the
        // slot count agreeing.
        let mut bad = good.clone();
        bad[h..h + 4].copy_from_slice(&100_000u32.to_le_bytes());
        bad[h + 4..h + 8].copy_from_slice(&1000u32.to_le_bytes());
        frame_fuzz::set_entries(&mut bad, 1000);
        frame_fuzz::repair_trailer(&mut bad);
        assert!(message(&bad).contains("holds"));
        // A record out of order: swap the ids of the first two records.
        let mut bad = good.clone();
        bad[h + 8..h + 12].copy_from_slice(&2u32.to_le_bytes());
        let second = h + 8 + 8 + 5 * 8;
        bad[second..second + 4].copy_from_slice(&1u32.to_le_bytes());
        frame_fuzz::repair_trailer(&mut bad);
        assert!(message(&bad).contains("increasing id order"));
        // A record claiming more nonzeros than remain.
        let mut bad = good.clone();
        bad[h + 12..h + 16].copy_from_slice(&1000u32.to_le_bytes());
        frame_fuzz::repair_trailer(&mut bad);
        assert!(message(&bad).contains("bytes remain"));
        // A record beyond the slots: fifty live records whose last id is
        // fifty, in fifty slots.
        let mut bad = good.clone();
        bad[h..h + 4].copy_from_slice(&50u32.to_le_bytes());
        frame_fuzz::repair_trailer(&mut bad);
        assert!(message(&bad).contains("beyond the"));
        // A dimension repeated inside a record, refused by the vector rules.
        let mut bad = good.clone();
        let first_dims = h + 16;
        bad[first_dims + 4..first_dims + 8].copy_from_slice(&3u32.to_le_bytes());
        frame_fuzz::repair_trailer(&mut bad);
        assert!(message(&bad).contains("record 1"));
        // Slots beyond what the collection holds.
        std::fs::write(&path, &good).unwrap();
        let tight = Bounds {
            max_records: 10,
            ..bounds()
        };
        assert!(matches!(
            PostingsIndex::restore(
                index.config(),
                "",
                &FsDir::new(dir.path()),
                &manifest,
                &tight
            ),
            Err(Error::DecodeFailed { .. })
        ));
        // Not in the manifest.
        assert!(matches!(
            PostingsIndex::restore(
                index.config(),
                "",
                &FsDir::new(dir.path()),
                &Manifest::default(),
                &bounds()
            ),
            Err(Error::ArtefactsMissing { .. })
        ));
        // Larger than the byte ceiling.
        let small = Bounds {
            max_bytes: 16,
            ..bounds()
        };
        assert!(matches!(
            PostingsIndex::restore(
                index.config(),
                "",
                &FsDir::new(dir.path()),
                &manifest,
                &small
            ),
            Err(Error::DecodeLengthExceeded { .. })
        ));
    }

    /// A zero value refused under term frequency weighting names the record,
    /// and the same bytes decode under the dot product.
    #[test]
    fn a_value_the_weighting_refuses_names_its_record() {
        let mut index = PostingsIndex::new(SparseConfig::default());
        let v = SparseVector {
            dims: vec![1, 2],
            values: vec![1.0, 0.0],
        };
        index
            .insert(Position(1), RecordId(1), v.as_ref(), Prepared::none())
            .unwrap();
        let bytes = encode(&index);
        assert!(decode(&bytes, index.config(), &bounds(), "p").is_ok());
        let bm25 = SparseConfig {
            weighting: Weighting::BM25,
            ..SparseConfig::default()
        };
        match decode(&bytes, &bm25, &bounds(), "p") {
            Err(Error::DecodeFailed { error, .. }) => assert!(error.starts_with("record 1")),
            other => panic!("expected a decode failure, got {:?}", other.map(|_| ())),
        }
    }

    /// A seeded mutator over a valid artefact never panics the reader and
    /// never sizes an allocation from a field the payload has not earned,
    /// which is what the bounds above are for. The frame is repaired around
    /// each mutation so the payload's reader sees it.
    #[test]
    fn no_mutation_of_a_valid_artefact_panics_the_reader() {
        let index = filled();
        let good = encode(&index);
        let mut rng = frame_fuzz::Rng(0x5eed_5b45_e000_0138);
        let cases = std::env::var("ZEUSDB_FUZZ_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_000usize);
        let mut accepted = 0usize;
        let mut past_frame = 0usize;
        for _ in 0..cases {
            let blob = frame_fuzz::mutate(&mut rng, &good, FrameKind::SparsePostings);
            match decode(&blob, index.config(), &bounds(), "p") {
                Ok(restored) => {
                    accepted += 1;
                    // Whatever it accepted is a well formed index.
                    assert!(restored.len() <= restored.slots());
                }
                Err(Error::DecodeFailed { error, .. }) => {
                    if !error.contains("frame") && !error.contains("file") {
                        past_frame += 1;
                    }
                }
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
        assert!(
            past_frame * 4 > cases,
            "only {} of {} mutations reached the payload's reader",
            past_frame,
            cases
        );
        // A mutated artefact that still decodes is one whose mutation landed
        // on a value rather than a count or an id.
        let _ = accepted;
    }

    /// A space over scattered positions writes the slot count the scalar
    /// holds, being one past the largest internal id inserted, with its newest
    /// record kept, and reads back to the same bytes and the same records.
    #[test]
    fn a_space_over_scattered_ids_round_trips_with_its_slot_count() {
        use zeusdb_vector_core::{unframe, FrameKind};
        let far = |k: u32| 3 + k * 1_009;
        let mut index = PostingsIndex::new(SparseConfig {
            unlink: Unlink::Strand,
            ..SparseConfig::default()
        });
        for k in 0..60u32 {
            let v = SparseVector {
                dims: vec![k % 7, 10 + k % 5],
                values: vec![1.0, 2.0 + k as f32],
            };
            index
                .insert(
                    Position(far(k)),
                    RecordId(far(k)),
                    v.as_ref(),
                    Prepared::none(),
                )
                .unwrap();
        }
        index.remove(Position(far(30))).unwrap();
        assert_eq!(index.slot_count(), far(59) as usize + 1);
        let bytes = encode(&index);
        let framed = unframe(&bytes, FrameKind::SparsePostings, "postings").unwrap();
        let word = |at: usize| {
            u32::from_le_bytes([
                framed.payload[at],
                framed.payload[at + 1],
                framed.payload[at + 2],
                framed.payload[at + 3],
            ])
        };
        assert_eq!(word(0), far(59) + 1, "the slot count");
        assert_eq!(word(4), 59, "the live records");
        let bounds = Bounds {
            min_records: 0,
            max_records: far(59) as usize,
            max_bytes: 1 << 30,
            ids_by_position: crate::identity_positions(far(59) as usize),
        };
        let restored = decode(&bytes, index.config(), &bounds, "postings").unwrap();
        assert_eq!(restored.slot_count(), far(59) as usize + 1);
        assert_eq!(encode(&restored), bytes);
        for k in 0..60u32 {
            assert_eq!(restored.holds(Position(far(k))), k != 30);
        }
    }

    /// A space whose records sit at dense positions, each saved under the
    /// internal id its position names, writes the bytes a space keyed by the
    /// internal ids themselves writes, and reads them back with each record
    /// at the position the mappings give its id, to the same bytes again.
    /// A save made once the newest record with a sparse half is gone records
    /// a slot count past every id the mappings hold, and reads back under the
    /// last id the collection issued. The refusals are the ones a space keyed
    /// by internal id met: a slot count past the bound, and a record whose id
    /// the mappings do not hold, named once the payload is read whole.
    #[test]
    fn a_space_at_dense_positions_writes_and_reads_the_bytes_ids_write() {
        let far = |k: u32| 3 + k * 1_009;
        let config = SparseConfig {
            unlink: Unlink::Lazy,
            weighting: Weighting::Bm25 { k1: 1.2, b: 0.75 },
            ..SparseConfig::default()
        };
        let vector = |k: u32| SparseVector {
            dims: vec![k % 7, 10 + k % 5, 20 + k % 3],
            values: vec![1.0, 2.0 + (k % 4) as f32, 3.0],
        };
        let mut by_id = PostingsIndex::new(config.clone());
        let mut by_position = PostingsIndex::new(config.clone());
        for k in 0..60u32 {
            by_id
                .insert(
                    Position(far(k)),
                    RecordId(far(k)),
                    vector(k).as_ref(),
                    Prepared::none(),
                )
                .unwrap();
            by_position
                .insert(
                    Position(k),
                    RecordId(far(k)),
                    vector(k).as_ref(),
                    Prepared::none(),
                )
                .unwrap();
        }
        for k in [30u32, 41] {
            by_id.remove(Position(far(k))).unwrap();
            by_position.remove(Position(k)).unwrap();
        }
        let ids: Vec<u32> = (0..60u32)
            .filter(|k| *k != 30 && *k != 41)
            .map(far)
            .collect();
        let positions_of = |at: usize| far(at as u32);
        let bytes = encode(&by_id);
        assert_eq!(encode_with(&by_position, &positions_of), bytes);

        fn bounds(ids: &[u32]) -> Bounds<'_> {
            Bounds {
                min_records: 0,
                max_records: *ids.last().unwrap() as usize,
                max_bytes: 1 << 30,
                ids_by_position: ids,
            }
        }
        let restored = decode(&bytes, &config, &bounds(&ids), "postings").unwrap();
        assert_eq!(restored.len(), 58);
        assert_eq!(restored.extent(), 58);
        assert!(restored.keys_agree());
        for (at, &id) in ids.iter().enumerate() {
            let k = (id - 3) / 1_009;
            let want: f32 = vector(k).values.iter().sum();
            assert_eq!(restored.length_of(Position(at as u32)), Some(want), "{at}");
        }
        assert_eq!(encode_with(&restored, &|at| ids[at]), bytes);

        // The newest record goes, the slot count stays, and the mappings no
        // longer reach it. Held to the last id the collection issued, which
        // is what a collection passes, the payload loads, with the slot count
        // of the records it files. Held to an id below its slot count, it is
        // refused.
        by_id.remove(Position(far(59))).unwrap();
        by_position.remove(Position(59)).unwrap();
        let bytes = encode(&by_id);
        assert_eq!(encode_with(&by_position, &positions_of), bytes);
        let fewer: Vec<u32> = ids.iter().copied().filter(|&id| id != far(59)).collect();
        let issued = Bounds {
            max_records: far(59) as usize,
            ..bounds(&fewer)
        };
        let restored = decode(&bytes, &config, &issued, "postings").unwrap();
        assert_eq!(restored.len(), 57);
        assert!(restored.keys_agree());
        assert_eq!(restored.slot_count(), far(58) as usize + 1);
        let refused = decode(&bytes, &config, &bounds(&fewer), "postings")
            .err()
            .expect("a slot count past the last id issued is refused");
        assert_eq!(
            refused.to_string(),
            Error::DecodeFailed {
                file: "postings".to_string(),
                error: format!(
                    "the payload declares {} record slots and the collection has issued ids up to \
                     {}",
                    far(59) + 1,
                    far(58)
                ),
            }
            .to_string()
        );

        // A record the mappings do not hold is read, held to its rules, and
        // named once the payload is read whole.
        let bytes = encode(&by_id);
        let mut missing = fewer.clone();
        missing.push(far(59));
        missing.retain(|&id| id != far(10));
        match decode(&bytes, &config, &bounds(&missing), "postings") {
            Err(Error::SparseRecordUnmapped { id, .. }) => assert_eq!(id, far(10)),
            other => panic!("{:?}", other.map(|index| index.len())),
        }
    }
}
