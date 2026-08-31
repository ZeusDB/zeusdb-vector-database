//! The term dictionary, being every distinct term a space has seen and the
//! id it was given.
//!
//! # The structure
//!
//! One hash map from the term to its id, and nothing else. Ids are issued
//! from zero in the order terms first arrive, so the map's size is the next
//! id. The term is stored once, as the map's key. The id-to-term direction
//! is needed only when the dictionary is written out, and is built then by
//! sorting the entries, which costs a save one sort of the vocabulary and
//! costs every lookup nothing.
//!
//! Per term the map holds the key's pointer and length and the id, being
//! twenty four bytes with padding, plus one control byte, at a load factor
//! of seven eighths, plus the term's bytes on the heap with the allocator's
//! own header. A lookup is one hash of the term and one probe.
//!
//! # It only grows
//!
//! A term is never removed, since the postings that carried its id may be
//! gone while the id stays the term's, and a query for the term must find
//! the same id. A compaction of the space does not shrink the dictionary.
//!
//! # How it is written
//!
//! `encode` lays the terms out in id order, each as its length and its
//! bytes, behind a magic and a count, and `decode` interns them in that
//! order, so the ids come back as they were. The collection owns the
//! artefact this becomes and the checksum that guards it.

use std::collections::HashMap;

use zeusdb_vector_core::Error;

const MAGIC: &[u8; 4] = b"ZTD1";

/// Every distinct term a space has seen, by id.
#[derive(Clone, Debug, Default)]
pub struct TermDictionary {
    ids: HashMap<Box<str>, u32>,
}

impl TermDictionary {
    pub fn new() -> Self {
        Self::default()
    }

    /// The id of `term`, issued now if the term is new.
    pub fn intern(&mut self, term: &str) -> Result<u32, Error> {
        if let Some(&id) = self.ids.get(term) {
            return Ok(id);
        }
        let id = u32::try_from(self.ids.len()).map_err(|_| Error::TermIdsExhausted)?;
        if id == u32::MAX {
            return Err(Error::TermIdsExhausted);
        }
        self.ids.insert(Box::from(term), id);
        Ok(id)
    }

    /// The id of `term`, where it has one. A query asks this, so a term no
    /// record carries is not given an id by being searched for.
    pub fn id_of(&self, term: &str) -> Option<u32> {
        self.ids.get(term).copied()
    }

    /// Distinct terms, which is also the next id.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Heap bytes the structure holds, by capacity, being the map's buckets
    /// and every term's bytes. The allocator's per-term header sits outside
    /// it.
    pub fn heap_bytes(&self) -> usize {
        let buckets = self.ids.capacity() * (std::mem::size_of::<(Box<str>, u32)>() + 1);
        let terms: usize = self.ids.keys().map(|k| k.len()).sum();
        buckets + terms
    }

    /// Every term in id order.
    pub fn terms(&self) -> Vec<&str> {
        let mut entries: Vec<(&u32, &Box<str>)> = self.ids.iter().map(|(t, id)| (id, t)).collect();
        entries.sort_unstable_by_key(|(id, _)| **id);
        entries.into_iter().map(|(_, t)| &**t).collect()
    }

    /// The dictionary as bytes, in id order.
    pub fn encode(&self) -> Vec<u8> {
        let terms = self.terms();
        let mut bytes = Vec::with_capacity(8 + terms.iter().map(|t| 4 + t.len()).sum::<usize>());
        bytes.extend_from_slice(MAGIC);
        bytes.extend((terms.len() as u32).to_le_bytes());
        for term in terms {
            bytes.extend((term.len() as u32).to_le_bytes());
            bytes.extend_from_slice(term.as_bytes());
        }
        bytes
    }

    /// A dictionary from the bytes `encode` wrote, refusing each way they
    /// can be wrong by name. `file` names the artefact in the message.
    pub fn decode(bytes: &[u8], file: &str) -> Result<Self, Error> {
        let corrupt = |detail: String| Error::DecodeFailed {
            file: file.to_string(),
            error: detail,
        };
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], Error> {
            let end = at
                .checked_add(n)
                .filter(|&end| end <= bytes.len())
                .ok_or_else(|| corrupt("the file ends before the term it declares".into()))?;
            let out = &bytes[at..end];
            at = end;
            Ok(out)
        };
        if take(4)? != MAGIC {
            return Err(corrupt(
                "the file does not open with the term dictionary magic".into(),
            ));
        }
        let u32_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let count = u32_of(take(4)?) as usize;
        // Each term costs at least four bytes, so the count is bounded by
        // the file before anything is sized from it.
        if count.saturating_mul(4) > bytes.len() {
            return Err(corrupt(format!(
                "the file declares {} terms and holds {} bytes",
                count,
                bytes.len()
            )));
        }
        let mut dictionary = TermDictionary {
            ids: HashMap::with_capacity(count),
        };
        for id in 0..count {
            let len = u32_of(take(4)?) as usize;
            let term = std::str::from_utf8(take(len)?)
                .map_err(|_| corrupt(format!("term {} is not UTF-8", id)))?;
            if term.is_empty() {
                return Err(corrupt(format!("term {} is empty", id)));
            }
            if dictionary.ids.contains_key(term) {
                return Err(corrupt(format!("term {} repeats an earlier term", id)));
            }
            dictionary.ids.insert(Box::from(term), id as u32);
        }
        if at != bytes.len() {
            return Err(corrupt(format!(
                "{} bytes follow the last term the file declares",
                bytes.len() - at
            )));
        }
        Ok(dictionary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids are issued in arrival order, a repeat gets its id back, a lookup
    /// issues nothing, and the terms come back in id order.
    #[test]
    fn ids_are_issued_in_arrival_order_and_a_lookup_issues_none() {
        let mut d = TermDictionary::new();
        assert_eq!(d.intern("fox").unwrap(), 0);
        assert_eq!(d.intern("the").unwrap(), 1);
        assert_eq!(d.intern("fox").unwrap(), 0);
        assert_eq!(d.id_of("the"), Some(1));
        assert_eq!(d.id_of("dog"), None);
        assert_eq!(d.len(), 2);
        assert_eq!(d.terms(), ["fox", "the"]);
        assert!(d.heap_bytes() >= 6);
    }

    /// The bytes read back to the same ids, and every damage is refused.
    #[test]
    fn the_encoding_round_trips_and_every_damage_is_refused() {
        let mut d = TermDictionary::new();
        for term in ["b", "a", "straße", "東京"] {
            d.intern(term).unwrap();
        }
        let bytes = d.encode();
        let back = TermDictionary::decode(&bytes, "terms").unwrap();
        assert_eq!(back.terms(), d.terms());
        assert_eq!(back.id_of("straße"), Some(2));

        let refused = |bytes: &[u8]| {
            matches!(
                TermDictionary::decode(bytes, "terms"),
                Err(Error::DecodeFailed { .. })
            )
        };
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(refused(&bad));
        assert!(refused(&bytes[..bytes.len() - 1]));
        let mut long = bytes.clone();
        long.push(0);
        assert!(refused(&long));
        // A count the bytes cannot hold.
        let mut bad = bytes.clone();
        bad[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(refused(&bad));
        // A term that is not UTF-8: the first term is "b" at offset 12.
        let mut bad = bytes.clone();
        bad[12] = 0xFF;
        assert!(refused(&bad));
        // A repeated term: make the second term "b" as well.
        let mut bad = bytes.clone();
        bad[17] = b'b';
        assert!(refused(&bad));
        // An empty term.
        let mut empty = Vec::new();
        empty.extend_from_slice(MAGIC);
        empty.extend(1u32.to_le_bytes());
        empty.extend(0u32.to_le_bytes());
        assert!(refused(&empty));
        assert!(
            TermDictionary::decode(&TermDictionary::new().encode(), "terms")
                .unwrap()
                .is_empty()
        );
    }
}
