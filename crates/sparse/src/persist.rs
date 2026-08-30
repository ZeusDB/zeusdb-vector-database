//! The one artefact, live records only, ids kept.
//!
//! The layout carries every live record's id and vector in id order, so a
//! restore replays it through `insert` and the lists come back sorted without
//! a sort. Dead records and their postings are not written, which is why a
//! restored index is compact whatever the policy the saved one ran under.

use std::path::Path;

use zeusdb_vector_core::{
    read_artefact, write_artefact, ArtefactRecord, Bounds, Error, Inventory, Ledger, RecordId,
    SparseRef,
};

use crate::index::{PostingsIndex, SparseConfig};

const MAGIC: &[u8; 4] = b"ZSP1";

/// What the artefact holds, for a message.
const CONTENTS: &str = "the sparse postings";

pub(crate) fn artefact_name(prefix: &str) -> String {
    format!("{prefix}postings.zdbsparse")
}

pub(crate) fn write(
    index: &PostingsIndex,
    prefix: &str,
    dir: &Path,
    ledger: &mut dyn Ledger,
) -> Result<(), Error> {
    let mut bytes = Vec::with_capacity(12 + index.live_nnz * 8 + index.live * 8);
    bytes.extend_from_slice(MAGIC);
    bytes.extend((index.records.len() as u32).to_le_bytes());
    bytes.extend((index.live as u32).to_le_bytes());
    for (id, slot) in index.records.iter().enumerate() {
        if !slot.held() || index.dead.contains(id) {
            continue;
        }
        let v = index.forward(*slot);
        bytes.extend((id as u32).to_le_bytes());
        bytes.extend((v.dims.len() as u32).to_le_bytes());
        for d in v.dims {
            bytes.extend(d.to_le_bytes());
        }
        for w in v.values {
            bytes.extend(w.to_le_bytes());
        }
    }
    let name = artefact_name(prefix);
    let checksum = write_artefact(dir, &name, &bytes)?;
    ledger.record(
        &name,
        ArtefactRecord {
            bytes: bytes.len() as u64,
            checksum: Some(checksum),
        },
    );
    Ok(())
}

pub(crate) fn restore(
    config: &SparseConfig,
    prefix: &str,
    dir: &Path,
    inventory: &dyn Inventory,
    bounds: &Bounds,
) -> Result<PostingsIndex, Error> {
    let name = artefact_name(prefix);
    let bytes = read_artefact(dir, &name, inventory, CONTENTS, bounds.max_bytes)?;
    let corrupt = |detail: String| Error::DecodeFailed {
        file: name.clone(),
        error: detail,
    };
    let mut reader = Reader {
        bytes: &bytes,
        at: 0,
    };
    let mut take = |n: usize| -> Result<&[u8], Error> {
        reader
            .take(n)
            .ok_or_else(|| corrupt("the file ends before the record it declares".into()))
    };
    if take(4)? != MAGIC {
        return Err(corrupt(
            "the file does not open with the sparse magic".into(),
        ));
    }
    let u32_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let f32_of = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let slots = u32_of(take(4)?) as usize;
    let live = u32_of(take(4)?) as usize;
    if slots > bounds.max_records.saturating_add(1) {
        return Err(corrupt(format!(
            "the file declares {} record slots and the collection has issued {} ids",
            slots, bounds.max_records
        )));
    }
    if live > slots {
        return Err(corrupt(format!(
            "the file declares {} live records in {} slots",
            live, slots
        )));
    }
    // Each live record costs at least eight bytes, so the count is bounded by
    // the file before anything is sized from it.
    if live.saturating_mul(8) > bytes.len() {
        return Err(corrupt(format!(
            "the file declares {} live records and holds {} bytes",
            live,
            bytes.len()
        )));
    }
    let mut index = PostingsIndex::new(config.clone());
    index.records.reserve(slots);
    let mut dims: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let mut last_id: Option<u32> = None;
    for _ in 0..live {
        let id = u32_of(take(4)?);
        let nnz = u32_of(take(4)?) as usize;
        if id as usize >= slots {
            return Err(corrupt(format!(
                "record {} is beyond the {} slots the file declares",
                id, slots
            )));
        }
        if last_id.is_some_and(|last| id <= last) {
            return Err(corrupt(format!(
                "record {} follows record {}, and the file is written in increasing id order",
                id,
                last_id.unwrap_or(0)
            )));
        }
        last_id = Some(id);
        if nnz.saturating_mul(8) > bytes.len() {
            return Err(corrupt(format!(
                "record {} declares {} nonzeros and the file holds {} bytes",
                id,
                nnz,
                bytes.len()
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
        index
            .insert_record(
                RecordId(id),
                SparseRef {
                    dims: &dims,
                    values: &values,
                },
            )
            .map_err(|e| corrupt(format!("record {}: {}", id, e)))?;
    }
    if let Some(rest) = take(usize::MAX).ok().filter(|rest| !rest.is_empty()) {
        return Err(corrupt(format!(
            "{} bytes follow the last record the file declares",
            rest.len()
        )));
    }
    index.calibrate();
    Ok(index)
}

/// A cursor over the artefact's bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    /// The next `n` bytes, or everything left where `n` is `usize::MAX`, or
    /// `None` where fewer than `n` remain.
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n == usize::MAX {
            let out = &self.bytes[self.at..];
            self.at = self.bytes.len();
            return Some(out);
        }
        let end = self.at.checked_add(n)?;
        let out = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Unlink;
    use std::collections::HashMap;
    use zeusdb_vector_core::{Persist, Prepared, Restore, SparseVector, VectorIndex};

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

    fn bounds() -> Bounds {
        Bounds {
            min_records: 0,
            max_records: 1 << 20,
            max_bytes: 1 << 30,
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
                .insert(RecordId(id), v.as_ref(), Prepared::none())
                .unwrap();
        }
        index
    }

    /// A save and a restore round trip every live record under its own id,
    /// leave the dead ones behind, and are recorded in the ledger.
    #[test]
    fn a_round_trip_keeps_every_live_record_and_drops_the_dead() {
        let mut index = filled();
        for id in [3u32, 17, 50] {
            index.remove(RecordId(id)).unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = Manifest::default();
        index.write("spaces/s/", dir.path(), &mut manifest).unwrap();
        assert_eq!(
            index.artefact_names("spaces/s/"),
            vec!["spaces/s/postings.zdbsparse".to_string()]
        );
        assert!(manifest.recorded("spaces/s/postings.zdbsparse").is_some());

        let restored = PostingsIndex::restore(
            index.config(),
            "spaces/s/",
            dir.path(),
            &manifest,
            &bounds(),
        )
        .unwrap();
        assert_eq!(restored.len(), 47);
        assert_eq!(restored.stranded(), 0);
        assert_eq!(restored.postings_total(), 47 * 5);
        for id in 1..=50u32 {
            assert_eq!(restored.recover(RecordId(id)), index.recover(RecordId(id)));
        }
    }

    /// Every way the artefact can be damaged is refused with a decode error
    /// naming the file, and a manifest that does not name it is refused
    /// before it is read.
    #[test]
    fn every_damage_is_refused_before_the_index_is_built() {
        let index = filled();
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = Manifest::default();
        index.write("", dir.path(), &mut manifest).unwrap();
        let path = dir.path().join("postings.zdbsparse");
        let good = std::fs::read(&path).unwrap();

        let refused = |bytes: &[u8]| {
            std::fs::write(&path, bytes).unwrap();
            let mut m = Manifest::default();
            m.record(
                "postings.zdbsparse",
                ArtefactRecord {
                    bytes: bytes.len() as u64,
                    checksum: Some(zeusdb_vector_core::checksum_of(bytes)),
                },
            );
            PostingsIndex::restore(index.config(), "", dir.path(), &m, &bounds())
        };

        // Wrong magic.
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(matches!(refused(&bad), Err(Error::DecodeFailed { .. })));
        // Truncated inside a record.
        assert!(matches!(
            refused(&good[..good.len() - 3]),
            Err(Error::DecodeFailed { .. })
        ));
        // Trailing bytes.
        let mut long = good.clone();
        long.push(0);
        assert!(matches!(refused(&long), Err(Error::DecodeFailed { .. })));
        // More live records than slots.
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(refused(&bad), Err(Error::DecodeFailed { .. })));
        // A record out of order: swap the ids of the first two records.
        let mut bad = good.clone();
        bad[12..16].copy_from_slice(&2u32.to_le_bytes());
        let second = 12 + 8 + 5 * 8;
        bad[second..second + 4].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(refused(&bad), Err(Error::DecodeFailed { .. })));
        // Slots beyond what the collection issued.
        std::fs::write(&path, &good).unwrap();
        let tight = Bounds {
            max_records: 10,
            ..bounds()
        };
        assert!(matches!(
            PostingsIndex::restore(index.config(), "", dir.path(), &manifest, &tight),
            Err(Error::DecodeFailed { .. })
        ));
        // Not in the manifest.
        assert!(matches!(
            PostingsIndex::restore(
                index.config(),
                "",
                dir.path(),
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
            PostingsIndex::restore(index.config(), "", dir.path(), &manifest, &small),
            Err(Error::DecodeLengthExceeded { .. })
        ));
    }
}
