//! What a journal record carries, and the codec that holds every length
//! inside a payload to what the payload has earned.
//!
//! [`crate::journal`] frames a record and knows nothing of its payload.
//! This module is the payload: one [`Operation`] a kind, encoded as
//! fixed-width little-endian fields whose order is stated here, and decoded
//! with every count checked against the bytes remaining before anything is
//! sized from it. Applying an operation to an index is the index's own
//! business, as are the rules that need the index's state, being the
//! internal id the counter would issue, the term id the dictionary would
//! issue, the rebuild parameters' ranges, a vector's finiteness and the
//! rules a sparse vector has to meet under the space's weighting. Those
//! run when the operation is applied, with the refusals the ordinary paths
//! raise.
//!
//! # Layouts
//!
//! ```text
//! Insert
//!    u32 id_len, id             UTF-8, at least one byte
//!    u64 internal_id            the id the record took, checked on replay
//!    u8  level                  the graph level drawn, below NB_LAYER_MAX
//!    u32 width, f32 × width     the vector, width equal to the index's dim
//!    u32 metadata_len, bytes    a JSON object
//!    u32 nnz                    u32::MAX for no sparse vector, else
//!    u32 × nnz dims, f32 × nnz values
//!
//! Remove
//!    u32 count, then count × (u32 len, id)
//!
//! UpdateMetadata
//!    u32 id_len, id
//!    u32 metadata_len, bytes    a JSON object
//!
//! Clear, Compact, RebuildQuantized
//!    empty
//!
//! Rebuild
//!    u64 m, u64 expected_size, u64 ef_construction
//!
//! Intern
//!    u32 term_id, then the term, UTF-8, at least one byte, to the end
//!
//! Train
//!    the completion stamp, UTF-8, at least one byte, to the end
//!
//! AddMetadata
//!    u32 count, then count × (u32 key_len, key, u32 value_len, value)
//! ```
//!
//! A payload ends where its last field ends, and trailing bytes fail it.
//! The encoder writes every length as a `u32`; a field wider than that
//! would make a payload wider than [`crate::journal::JOURNAL_MAX_PAYLOAD`],
//! which the writer refuses to append, so no such record reaches a file.

use crate::error::Error;
use crate::graph::dump::NB_LAYER_MAX;
use crate::journal::{JournalRecord, OperationKind};
use crate::space::SparseVector;
use serde_json::{Map, Value};

/// The value `nnz` carries for a record with no sparse vector.
const NO_SPARSE: u32 = u32::MAX;

/// One mutation, as the journal records it.
#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    /// One record inserted, with its whole content.
    Insert {
        id: String,
        /// The internal id the record took, carried as a check: replay
        /// asserts it is the one the counter would issue next.
        internal_id: u64,
        /// The graph level the record was installed at, so replay
        /// installs at the same level rather than drawing a fresh one.
        level: u8,
        vector: Vec<f32>,
        metadata: Map<String, Value>,
        sparse: Option<SparseVector>,
    },
    /// Records removed, by resolved id. Never a filter, because a filter's
    /// answer depends on the state it was asked against.
    Remove { ids: Vec<String> },
    /// One record's metadata replaced wholesale.
    UpdateMetadata {
        id: String,
        metadata: Map<String, Value>,
    },
    /// Every record removed and every counter reset.
    Clear,
    /// The graph rebuilt over the live records.
    Compact,
    /// The graph rebuilt under these parameters.
    Rebuild {
        m: u64,
        expected_size: u64,
        ef_construction: u64,
    },
    /// A term interned at the id the dictionary issued, appended at the
    /// moment of issue so log order is id order.
    Intern { term_id: u32, term: String },
    /// The quantizer trained, with the stamp it took, injected on replay
    /// rather than read from the clock.
    Train { completed_at: String },
    /// Pairs added to the index's own metadata.
    AddMetadata { pairs: Vec<(String, String)> },
    /// The graph rebuilt over the quantized codes.
    RebuildQuantized,
}

impl Operation {
    /// The kind the record header carries for this operation.
    pub fn kind(&self) -> OperationKind {
        match self {
            Operation::Insert { .. } => OperationKind::Insert,
            Operation::Remove { .. } => OperationKind::Remove,
            Operation::UpdateMetadata { .. } => OperationKind::UpdateMetadata,
            Operation::Clear => OperationKind::Clear,
            Operation::Compact => OperationKind::Compact,
            Operation::Rebuild { .. } => OperationKind::Rebuild,
            Operation::Intern { .. } => OperationKind::Intern,
            Operation::Train { .. } => OperationKind::Train,
            Operation::AddMetadata { .. } => OperationKind::AddMetadata,
            Operation::RebuildQuantized => OperationKind::RebuildQuantized,
        }
    }

    /// Encode the payload into `out`, which is cleared first.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.clear();
        match self {
            Operation::Insert {
                id,
                internal_id,
                level,
                vector,
                metadata,
                sparse,
            } => {
                put_bytes(out, id.as_bytes());
                out.extend_from_slice(&internal_id.to_le_bytes());
                out.push(*level);
                out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                for value in vector {
                    out.extend_from_slice(&value.to_le_bytes());
                }
                put_json(out, metadata);
                match sparse {
                    None => out.extend_from_slice(&NO_SPARSE.to_le_bytes()),
                    Some(sparse) => {
                        out.extend_from_slice(&(sparse.dims.len() as u32).to_le_bytes());
                        for dim in &sparse.dims {
                            out.extend_from_slice(&dim.to_le_bytes());
                        }
                        for value in &sparse.values {
                            out.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
            Operation::Remove { ids } => {
                out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
                for id in ids {
                    put_bytes(out, id.as_bytes());
                }
            }
            Operation::UpdateMetadata { id, metadata } => {
                put_bytes(out, id.as_bytes());
                put_json(out, metadata);
            }
            Operation::Clear | Operation::Compact | Operation::RebuildQuantized => {}
            Operation::Rebuild {
                m,
                expected_size,
                ef_construction,
            } => {
                out.extend_from_slice(&m.to_le_bytes());
                out.extend_from_slice(&expected_size.to_le_bytes());
                out.extend_from_slice(&ef_construction.to_le_bytes());
            }
            Operation::Intern { term_id, term } => {
                out.extend_from_slice(&term_id.to_le_bytes());
                out.extend_from_slice(term.as_bytes());
            }
            Operation::Train { completed_at } => {
                out.extend_from_slice(completed_at.as_bytes());
            }
            Operation::AddMetadata { pairs } => {
                out.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
                for (key, value) in pairs {
                    put_bytes(out, key.as_bytes());
                    put_bytes(out, value.as_bytes());
                }
            }
        }
    }

    /// Decode a record's payload, holding every length to what remains of
    /// it before anything is sized. `dim` is the index's vector width, and
    /// `file` names the journal in a message.
    pub fn decode(record: &JournalRecord<'_>, dim: usize, file: &str) -> Result<Operation, Error> {
        let mut cursor = Cursor {
            bytes: record.payload,
            at: 0,
        };
        let decoded = decode_fields(&mut cursor, record.kind, dim);
        let decoded = decoded.and_then(|op| {
            if cursor.remaining() == 0 {
                Ok(op)
            } else {
                Err(format!(
                    "{} bytes follow the last field of the {} payload",
                    cursor.remaining(),
                    record.kind.label()
                ))
            }
        });
        decoded.map_err(|detail| Error::JournalRecordInvalid {
            file: file.to_string(),
            sequence: record.sequence,
            at: record.offset,
            detail,
        })
    }
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn put_json(out: &mut Vec<u8>, metadata: &Map<String, Value>) {
    let json = serde_json::to_vec(metadata).unwrap_or_default();
    put_bytes(out, &json);
}

/// A position in a payload. Every read checks what remains first.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }

    /// `len` bytes for `what`, held to what remains.
    fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8], String> {
        if len > self.remaining() {
            return Err(format!(
                "the {what} needs {len} bytes and {} remain",
                self.remaining()
            ));
        }
        let out = &self.bytes[self.at..self.at + len];
        self.at += len;
        Ok(out)
    }

    fn u8(&mut self, what: &str) -> Result<u8, String> {
        Ok(self.take(1, what)?[0])
    }

    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let raw = self.take(4, what)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn u64(&mut self, what: &str) -> Result<u64, String> {
        let raw = self.take(8, what)?;
        let mut out = [0u8; 8];
        out.copy_from_slice(raw);
        Ok(u64::from_le_bytes(out))
    }

    /// A length-prefixed string, its length held to what remains before
    /// the bytes are read, and the bytes held to UTF-8.
    fn string(&mut self, what: &str) -> Result<String, String> {
        let len = self.u32(&format!("{what} length"))? as usize;
        let raw = self.take(len, what)?;
        std::str::from_utf8(raw)
            .map(str::to_string)
            .map_err(|e| format!("the {what} is not UTF-8: {e}"))
    }

    /// A length-prefixed JSON object.
    fn object(&mut self, what: &str) -> Result<Map<String, Value>, String> {
        let len = self.u32(&format!("{what} length"))? as usize;
        let raw = self.take(len, what)?;
        match serde_json::from_slice::<Value>(raw) {
            Ok(Value::Object(map)) => Ok(map),
            Ok(other) => Err(format!(
                "the {what} is a JSON {} where an object was expected",
                json_kind(&other)
            )),
            Err(e) => Err(format!("the {what} is not JSON: {e}")),
        }
    }

    /// `count` values of `width` bytes each, the product held to what
    /// remains before anything is sized.
    fn values<T>(
        &mut self,
        count: usize,
        width: usize,
        what: &str,
        read: impl Fn(&[u8]) -> T,
    ) -> Result<Vec<T>, String> {
        let bytes = count
            .checked_mul(width)
            .ok_or_else(|| format!("the {what} count {count} overflows"))?;
        let raw = self.take(bytes, what)?;
        Ok(raw.chunks_exact(width).map(read).collect())
    }
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn f32_at(raw: &[u8]) -> f32 {
    f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
}

fn u32_at(raw: &[u8]) -> u32 {
    u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
}

/// Read the fields of one kind. Trailing bytes are the caller's check.
fn decode_fields(c: &mut Cursor<'_>, kind: OperationKind, dim: usize) -> Result<Operation, String> {
    match kind {
        OperationKind::Insert => {
            let id = c.string("id")?;
            if id.is_empty() {
                return Err("the id is empty".into());
            }
            let internal_id = c.u64("internal id")?;
            let level = c.u8("level")?;
            if level >= NB_LAYER_MAX {
                return Err(format!("level {level} is not below {NB_LAYER_MAX}"));
            }
            let width = c.u32("width")? as usize;
            if width != dim {
                return Err(format!("the vector is {width} wide and the index is {dim}"));
            }
            let vector = c.values(width, 4, "vector", f32_at)?;
            let metadata = c.object("metadata")?;
            let nnz = c.u32("sparse count")?;
            let sparse = if nnz == NO_SPARSE {
                None
            } else {
                let nnz = nnz as usize;
                // Eight bytes a posting, held before either slice is sized.
                let needed = nnz
                    .checked_mul(8)
                    .ok_or_else(|| format!("the sparse count {nnz} overflows"))?;
                if needed > c.remaining() {
                    return Err(format!(
                        "the sparse vector needs {needed} bytes and {} remain",
                        c.remaining()
                    ));
                }
                let dims = c.values(nnz, 4, "sparse dims", u32_at)?;
                let values = c.values(nnz, 4, "sparse values", f32_at)?;
                Some(SparseVector { dims, values })
            };
            Ok(Operation::Insert {
                id,
                internal_id,
                level,
                vector,
                metadata,
                sparse,
            })
        }
        OperationKind::Remove => {
            let count = c.u32("id count")? as usize;
            // Four bytes an id at least, being its length, held before the
            // list is sized.
            let least = count
                .checked_mul(4)
                .ok_or_else(|| format!("the id count {count} overflows"))?;
            if least > c.remaining() {
                return Err(format!(
                    "{count} ids need at least {least} bytes and {} remain",
                    c.remaining()
                ));
            }
            let mut ids = Vec::with_capacity(count);
            for _ in 0..count {
                ids.push(c.string("id")?);
            }
            Ok(Operation::Remove { ids })
        }
        OperationKind::UpdateMetadata => {
            let id = c.string("id")?;
            let metadata = c.object("metadata")?;
            Ok(Operation::UpdateMetadata { id, metadata })
        }
        OperationKind::Clear => Ok(Operation::Clear),
        OperationKind::Compact => Ok(Operation::Compact),
        OperationKind::RebuildQuantized => Ok(Operation::RebuildQuantized),
        OperationKind::Rebuild => {
            let m = c.u64("m")?;
            let expected_size = c.u64("expected size")?;
            let ef_construction = c.u64("ef construction")?;
            Ok(Operation::Rebuild {
                m,
                expected_size,
                ef_construction,
            })
        }
        OperationKind::Intern => {
            let term_id = c.u32("term id")?;
            let raw = c.take(c.remaining(), "term")?;
            if raw.is_empty() {
                return Err("the term is empty".into());
            }
            let term = std::str::from_utf8(raw)
                .map_err(|e| format!("the term is not UTF-8: {e}"))?
                .to_string();
            Ok(Operation::Intern { term_id, term })
        }
        OperationKind::Train => {
            let raw = c.take(c.remaining(), "stamp")?;
            if raw.is_empty() {
                return Err("the stamp is empty".into());
            }
            let completed_at = std::str::from_utf8(raw)
                .map_err(|e| format!("the stamp is not UTF-8: {e}"))?
                .to_string();
            Ok(Operation::Train { completed_at })
        }
        OperationKind::AddMetadata => {
            let count = c.u32("pair count")? as usize;
            // Eight bytes a pair at least, being two lengths, held before
            // the list is sized.
            let least = count
                .checked_mul(8)
                .ok_or_else(|| format!("the pair count {count} overflows"))?;
            if least > c.remaining() {
                return Err(format!(
                    "{count} pairs need at least {least} bytes and {} remain",
                    c.remaining()
                ));
            }
            let mut pairs = Vec::with_capacity(count);
            for _ in 0..count {
                let key = c.string("key")?;
                let value = c.string("value")?;
                pairs.push((key, value));
            }
            Ok(Operation::AddMetadata { pairs })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::fuzz::{Rng, HOSTILE};
    use crate::journal::JOURNAL_MAX_PAYLOAD;

    const DIM: usize = 4;

    fn record(kind: OperationKind, payload: &[u8]) -> JournalRecord<'_> {
        JournalRecord {
            sequence: 17,
            kind,
            offset: 167,
            payload,
        }
    }

    fn decode(kind: OperationKind, payload: &[u8]) -> Result<Operation, Error> {
        Operation::decode(&record(kind, payload), DIM, "t.zdbwal")
    }

    fn refused(kind: OperationKind, payload: &[u8]) -> String {
        match decode(kind, payload) {
            Err(Error::JournalRecordInvalid {
                file,
                sequence,
                at,
                detail,
            }) => {
                assert_eq!(file, "t.zdbwal");
                assert_eq!(sequence, 17);
                assert_eq!(at, 167);
                detail
            }
            Err(other) => panic!("expected a record refusal, got {other:?}"),
            Ok(op) => panic!("expected a record refusal, got {op:?}"),
        }
    }

    fn metadata() -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("category".into(), Value::String("tools".into()));
        map.insert("count".into(), Value::from(3));
        map.insert(
            "tags".into(),
            Value::Array(vec![Value::from("a"), Value::from("b")]),
        );
        map
    }

    fn every_operation() -> Vec<Operation> {
        vec![
            Operation::Insert {
                id: "doc-1".into(),
                internal_id: 42,
                level: 3,
                vector: vec![1.0, -2.5, 0.0, 1e-7],
                metadata: metadata(),
                sparse: Some(SparseVector {
                    dims: vec![3, 9, 1000],
                    values: vec![1.0, 2.0, 0.5],
                }),
            },
            Operation::Insert {
                id: "ü".into(),
                internal_id: u64::MAX,
                level: 0,
                vector: vec![0.0; 4],
                metadata: Map::new(),
                sparse: None,
            },
            Operation::Remove { ids: vec![] },
            Operation::Remove {
                ids: vec!["a".into(), "".into(), "cé".into()],
            },
            Operation::UpdateMetadata {
                id: "doc-1".into(),
                metadata: metadata(),
            },
            Operation::Clear,
            Operation::Compact,
            Operation::Rebuild {
                m: 24,
                expected_size: 100_000,
                ef_construction: 400,
            },
            Operation::Intern {
                term_id: 7,
                term: "zebra".into(),
            },
            Operation::Train {
                completed_at: "2026-01-02T03:04:05.678Z".into(),
            },
            Operation::AddMetadata { pairs: vec![] },
            Operation::AddMetadata {
                pairs: vec![("k".into(), "v".into()), ("".into(), "".into())],
            },
            Operation::RebuildQuantized,
        ]
    }

    /// Every operation encodes and decodes to itself, under the kind it
    /// names.
    #[test]
    fn every_operation_round_trips() {
        let mut out = Vec::new();
        for op in every_operation() {
            op.encode(&mut out);
            let back = decode(op.kind(), &out).unwrap();
            assert_eq!(back, op);
        }
        // Encoding clears the buffer first.
        Operation::Clear.encode(&mut out);
        assert!(out.is_empty());
    }

    /// The insert layout is what the module says, byte for byte.
    #[test]
    fn an_insert_lays_out_as_documented() {
        let mut out = Vec::new();
        Operation::Insert {
            id: "ab".into(),
            internal_id: 0x0102_0304_0506_0708,
            level: 5,
            vector: vec![1.0, 2.0, 3.0, 4.0],
            metadata: Map::new(),
            sparse: Some(SparseVector {
                dims: vec![9],
                values: vec![1.5],
            }),
        }
        .encode(&mut out);
        let mut want = Vec::new();
        want.extend_from_slice(&2u32.to_le_bytes());
        want.extend_from_slice(b"ab");
        want.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        want.push(5);
        want.extend_from_slice(&4u32.to_le_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            want.extend_from_slice(&v.to_le_bytes());
        }
        want.extend_from_slice(&2u32.to_le_bytes());
        want.extend_from_slice(b"{}");
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&9u32.to_le_bytes());
        want.extend_from_slice(&1.5f32.to_le_bytes());
        assert_eq!(out, want);
        // No sparse vector is the marker alone.
        let mut out = Vec::new();
        Operation::Insert {
            id: "a".into(),
            internal_id: 1,
            level: 0,
            vector: vec![0.0; 4],
            metadata: Map::new(),
            sparse: None,
        }
        .encode(&mut out);
        assert_eq!(&out[out.len() - 4..], &u32::MAX.to_le_bytes());
    }

    /// An insert's id length is held to what remains before the id is
    /// read, and an empty id is refused.
    #[test]
    fn an_insert_id_length_is_held_to_the_payload() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(1u32 << 30).to_le_bytes());
        payload.extend_from_slice(b"ab");
        assert_eq!(
            refused(OperationKind::Insert, &payload),
            "the id needs 1073741824 bytes and 2 remain"
        );
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(refused(OperationKind::Insert, &payload), "the id is empty");
        assert!(refused(OperationKind::Insert, &[]).contains("id length"));
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u32.to_le_bytes());
        payload.extend_from_slice(&[0xFF, 0xFE]);
        assert!(refused(OperationKind::Insert, &payload).contains("not UTF-8"));
    }

    /// An insert's level is held below `NB_LAYER_MAX` before the record
    /// is applied, since the graph asserts on it.
    #[test]
    fn an_insert_level_is_held_below_the_layer_ceiling() {
        let base = Operation::Insert {
            id: "a".into(),
            internal_id: 1,
            level: NB_LAYER_MAX - 1,
            vector: vec![0.0; 4],
            metadata: Map::new(),
            sparse: None,
        };
        let mut out = Vec::new();
        base.encode(&mut out);
        assert!(decode(OperationKind::Insert, &out).is_ok());
        let level_at = 4 + 1 + 8;
        out[level_at] = NB_LAYER_MAX;
        assert_eq!(
            refused(OperationKind::Insert, &out),
            format!("level {NB_LAYER_MAX} is not below {NB_LAYER_MAX}")
        );
        out[level_at] = 255;
        assert!(refused(OperationKind::Insert, &out).contains("level 255"));
    }

    /// An insert's width is held equal to the index's `dim` before the
    /// vector is read, so a width of a billion on a short payload sizes
    /// nothing and names the width.
    #[test]
    fn an_insert_width_is_held_to_the_declared_dim() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.push(b'a');
        payload.extend_from_slice(&1u64.to_le_bytes());
        payload.push(0);
        let width_at = payload.len();
        payload.extend_from_slice(&(1u32 << 30).to_le_bytes());
        assert_eq!(
            refused(OperationKind::Insert, &payload),
            "the vector is 1073741824 wide and the index is 4"
        );
        payload[width_at..width_at + 4].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            refused(OperationKind::Insert, &payload),
            "the vector is 3 wide and the index is 4"
        );
        // The right width on a payload too short to hold it.
        payload[width_at..width_at + 4].copy_from_slice(&4u32.to_le_bytes());
        payload.extend_from_slice(&[0u8; 15]);
        assert_eq!(
            refused(OperationKind::Insert, &payload),
            "the vector needs 16 bytes and 15 remain"
        );
    }

    /// An insert's metadata length is held to what remains, and the bytes
    /// are held to a JSON object.
    #[test]
    fn an_insert_metadata_is_held_to_a_json_object() {
        let mut prefix = Vec::new();
        prefix.extend_from_slice(&1u32.to_le_bytes());
        prefix.push(b'a');
        prefix.extend_from_slice(&1u64.to_le_bytes());
        prefix.push(0);
        prefix.extend_from_slice(&4u32.to_le_bytes());
        prefix.extend_from_slice(&[0u8; 16]);
        let with = |json: &[u8], len: u32| {
            let mut p = prefix.clone();
            p.extend_from_slice(&len.to_le_bytes());
            p.extend_from_slice(json);
            p.extend_from_slice(&NO_SPARSE.to_le_bytes());
            p
        };
        assert_eq!(
            refused(OperationKind::Insert, &with(b"{}", 1 << 20)),
            "the metadata needs 1048576 bytes and 6 remain"
        );
        assert_eq!(
            refused(OperationKind::Insert, &with(b"[1]", 3)),
            "the metadata is a JSON array where an object was expected"
        );
        assert_eq!(
            refused(OperationKind::Insert, &with(b"7", 1)),
            "the metadata is a JSON number where an object was expected"
        );
        assert!(refused(OperationKind::Insert, &with(b"{", 1)).contains("not JSON"));
        match decode(OperationKind::Insert, &with(br#"{"k":[1,2]}"#, 11)).unwrap() {
            Operation::Insert { metadata, .. } => {
                assert_eq!(
                    metadata.get("k"),
                    Some(&Value::Array(vec![1.into(), 2.into()]))
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// An insert's sparse count is `u32::MAX` for none, else held to eight
    /// bytes a posting against what remains before either slice is sized.
    #[test]
    fn an_insert_sparse_count_is_held_before_anything_is_sized() {
        let mut prefix = Vec::new();
        prefix.extend_from_slice(&1u32.to_le_bytes());
        prefix.push(b'a');
        prefix.extend_from_slice(&1u64.to_le_bytes());
        prefix.push(0);
        prefix.extend_from_slice(&4u32.to_le_bytes());
        prefix.extend_from_slice(&[0u8; 16]);
        prefix.extend_from_slice(&2u32.to_le_bytes());
        prefix.extend_from_slice(b"{}");
        let with = |nnz: u32, tail: &[u8]| {
            let mut p = prefix.clone();
            p.extend_from_slice(&nnz.to_le_bytes());
            p.extend_from_slice(tail);
            p
        };
        assert!(matches!(
            decode(OperationKind::Insert, &with(NO_SPARSE, &[])).unwrap(),
            Operation::Insert { sparse: None, .. }
        ));
        assert_eq!(
            refused(OperationKind::Insert, &with(u32::MAX - 1, &[0; 8])),
            "the sparse vector needs 34359738352 bytes and 8 remain"
        );
        assert_eq!(
            refused(OperationKind::Insert, &with(2, &[0; 15])),
            "the sparse vector needs 16 bytes and 15 remain"
        );
        let mut tail = Vec::new();
        tail.extend_from_slice(&5u32.to_le_bytes());
        tail.extend_from_slice(&7u32.to_le_bytes());
        tail.extend_from_slice(&1.0f32.to_le_bytes());
        tail.extend_from_slice(&2.0f32.to_le_bytes());
        match decode(OperationKind::Insert, &with(2, &tail)).unwrap() {
            Operation::Insert {
                sparse: Some(sparse),
                ..
            } => {
                assert_eq!(sparse.dims, vec![5, 7]);
                assert_eq!(sparse.values, vec![1.0, 2.0]);
            }
            other => panic!("{other:?}"),
        }
        // A zero count is a sparse vector with no postings, which is not
        // the same as none.
        assert!(matches!(
            decode(OperationKind::Insert, &with(0, &[])).unwrap(),
            Operation::Insert {
                sparse: Some(SparseVector { .. }),
                ..
            }
        ));
    }

    /// A remove's count is held to four bytes an id against what remains
    /// before the list is sized, then each id's length against what
    /// remains.
    #[test]
    fn a_remove_count_is_held_before_the_list_is_sized() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(1u32 << 30).to_le_bytes());
        payload.extend_from_slice(&[0; 8]);
        assert_eq!(
            refused(OperationKind::Remove, &payload),
            "1073741824 ids need at least 4294967296 bytes and 8 remain"
        );
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u32.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.push(b'a');
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.push(b'b');
        assert_eq!(
            refused(OperationKind::Remove, &payload),
            "the id needs 9 bytes and 1 remain"
        );
        assert!(refused(OperationKind::Remove, &[]).contains("id count"));
    }

    /// An intern's term is the rest of the payload, non-empty UTF-8.
    #[test]
    fn an_intern_term_is_non_empty_utf8() {
        let mut payload = 3u32.to_le_bytes().to_vec();
        assert_eq!(
            refused(OperationKind::Intern, &payload),
            "the term is empty"
        );
        payload.extend_from_slice(&[0xC3, 0x28]);
        assert!(refused(OperationKind::Intern, &payload).contains("not UTF-8"));
        assert!(refused(OperationKind::Intern, &[1, 2]).contains("term id"));
        let mut payload = 3u32.to_le_bytes().to_vec();
        payload.extend_from_slice("été".as_bytes());
        assert_eq!(
            decode(OperationKind::Intern, &payload).unwrap(),
            Operation::Intern {
                term_id: 3,
                term: "été".into()
            }
        );
    }

    /// A train's stamp is the whole payload, non-empty UTF-8.
    #[test]
    fn a_train_stamp_is_non_empty_utf8() {
        assert_eq!(refused(OperationKind::Train, &[]), "the stamp is empty");
        assert!(refused(OperationKind::Train, &[0xFF]).contains("not UTF-8"));
    }

    /// A rebuild is exactly three `u64`s.
    #[test]
    fn a_rebuild_is_exactly_three_numbers() {
        assert!(refused(OperationKind::Rebuild, &[0; 23]).contains("ef construction"));
        assert_eq!(
            refused(OperationKind::Rebuild, &[0; 25]),
            "1 bytes follow the last field of the rebuild payload"
        );
        assert_eq!(
            decode(OperationKind::Rebuild, &[0; 24]).unwrap(),
            Operation::Rebuild {
                m: 0,
                expected_size: 0,
                ef_construction: 0
            }
        );
    }

    /// An add metadata's count is held to eight bytes a pair against what
    /// remains before the list is sized, then each string.
    #[test]
    fn an_add_metadata_count_is_held_before_the_list_is_sized() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        payload.extend_from_slice(&[0; 16]);
        assert_eq!(
            refused(OperationKind::AddMetadata, &payload),
            "4294967295 pairs need at least 34359738360 bytes and 16 remain"
        );
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.push(b'k');
        payload.extend_from_slice(&(1u32 << 20).to_le_bytes());
        assert_eq!(
            refused(OperationKind::AddMetadata, &payload),
            "the value needs 1048576 bytes and 0 remain"
        );
    }

    /// Trailing bytes after a payload's last field fail every kind.
    #[test]
    fn trailing_bytes_fail_every_kind() {
        let mut out = Vec::new();
        for op in every_operation() {
            let kind = op.kind();
            if matches!(kind, OperationKind::Intern | OperationKind::Train) {
                // The last field runs to the end, so nothing can trail it.
                continue;
            }
            op.encode(&mut out);
            out.push(0);
            let detail = refused(kind, &out);
            assert_eq!(
                detail,
                format!(
                    "1 bytes follow the last field of the {} payload",
                    kind.label()
                ),
                "{kind:?}"
            );
        }
        assert!(refused(OperationKind::Clear, &[1]).contains("clear payload"));
        assert!(refused(OperationKind::Compact, &[1, 2]).contains("2 bytes follow"));
        assert!(refused(OperationKind::RebuildQuantized, &[1]).contains("rebuild quantized"));
    }

    /// A seeded mutator over every valid payload never panics the decoder,
    /// every refusal is the decoder's own error, and a share of mutated
    /// payloads still decode, which is what proves the mutations land on
    /// values as well as on counts.
    #[test]
    fn no_mutation_of_a_valid_payload_panics_the_decoder() {
        let mut rng = Rng(0x5eed_0b5e_e000_0146);
        let cases = std::env::var("ZEUSDB_FUZZ_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_000usize);
        let ops = every_operation();
        let mut encoded = Vec::new();
        for op in &ops {
            let mut out = Vec::new();
            op.encode(&mut out);
            encoded.push((op.kind(), out));
        }
        let mut accepted = 0usize;
        let mut refused_past_first_field = 0usize;
        for _ in 0..cases {
            let (kind, base) = &encoded[rng.below(encoded.len())];
            let mut blob = base.clone();
            let mutations = 1 + rng.below(3);
            for _ in 0..mutations {
                let len = blob.len();
                match rng.below(5) {
                    0 | 1 => {
                        if len > 0 {
                            let width = [1usize, 2, 4, 8][rng.below(4)];
                            let at = rng.below(len);
                            let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                            let end = (at + width).min(len);
                            blob[at..end].copy_from_slice(&value[..end - at]);
                        }
                    }
                    2 => {
                        if len > 0 {
                            let at = rng.below(len);
                            blob[at] = rng.byte();
                        }
                    }
                    3 => blob.truncate(rng.below(len + 1)),
                    _ => {
                        let extra = 1 + rng.below(16);
                        let value = rng.byte();
                        blob.extend(std::iter::repeat_n(value, extra));
                    }
                }
            }
            assert!(blob.len() <= JOURNAL_MAX_PAYLOAD);
            match decode(*kind, &blob) {
                Ok(_) => accepted += 1,
                Err(Error::JournalRecordInvalid { detail, .. }) => {
                    if !detail.contains("id length") && !detail.contains("id count") {
                        refused_past_first_field += 1;
                    }
                }
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
        assert!(accepted * 20 > cases, "only {accepted} of {cases} decoded");
        assert!(
            refused_past_first_field * 4 > cases,
            "only {refused_past_first_field} of {cases} reached past the first field"
        );
    }
}
