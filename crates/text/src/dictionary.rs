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
//! bytes, behind a count, inside the frame `zeusdb_vector_core::frame`
//! describes under the kind `TermDictionary`, with the frame's `entries`
//! holding the count. `decode` interns them in that order, so the ids come
//! back as they were. The count is held to the frame's `entries` and to the
//! payload's length at five bytes a term, and each term's length to what is
//! left of the payload, so nothing is sized from a field the payload has not
//! earned. The collection owns the artefact this becomes.

use std::collections::HashMap;

use zeusdb_vector_core::{frame_begin, frame_finish, unframe, Error, FrameEncoding, FrameKind};

/// Bytes a term costs at least, being its length and one byte of text.
const TERM_MIN_BYTES: usize = 5;

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

    /// The dictionary as bytes, in id order, frame included, built in one
    /// buffer.
    pub fn encode(&self) -> Vec<u8> {
        let terms = self.terms();
        let mut payload = frame_begin(
            FrameKind::TermDictionary,
            FrameEncoding::Engine,
            4 + terms.iter().map(|t| 4 + t.len()).sum::<usize>(),
        );
        payload.extend((terms.len() as u32).to_le_bytes());
        for term in &terms {
            payload.extend((term.len() as u32).to_le_bytes());
            payload.extend_from_slice(term.as_bytes());
        }
        frame_finish(payload, terms.len() as u64)
    }

    /// A dictionary from the bytes `encode` wrote, refusing each way they
    /// can be wrong by name. `file` names the artefact in the message.
    pub fn decode(bytes: &[u8], file: &str) -> Result<Self, Error> {
        let framed = unframe(bytes, FrameKind::TermDictionary, file)?;
        let payload = framed.payload;
        let corrupt = |detail: String| Error::DecodeFailed {
            file: file.to_string(),
            error: detail,
        };
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], Error> {
            let end = at
                .checked_add(n)
                .filter(|&end| end <= payload.len())
                .ok_or_else(|| corrupt("the payload ends before the term it declares".into()))?;
            let out = &payload[at..end];
            at = end;
            Ok(out)
        };
        let u32_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let count = u32_of(take(4)?) as usize;
        if count as u64 != framed.entries {
            return Err(corrupt(format!(
                "the payload declares {} terms and the frame counts {}",
                count, framed.entries
            )));
        }
        // Each term costs at least five bytes, so the count is bounded by
        // the payload before anything is sized from it.
        if count.saturating_mul(TERM_MIN_BYTES) > payload.len().saturating_sub(4) {
            return Err(corrupt(format!(
                "the payload declares {} terms and holds {} bytes",
                count,
                payload.len()
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
        if at != payload.len() {
            return Err(corrupt(format!(
                "{} bytes follow the last term the payload declares",
                payload.len() - at
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
        use zeusdb_vector_core::{frame_fuzz, FRAME_HEADER_BYTES};
        let mut d = TermDictionary::new();
        for term in ["b", "a", "straße", "東京"] {
            d.intern(term).unwrap();
        }
        let bytes = d.encode();
        let back = TermDictionary::decode(&bytes, "terms").unwrap();
        assert_eq!(back.terms(), d.terms());
        assert_eq!(back.id_of("straße"), Some(2));
        let h = FRAME_HEADER_BYTES;

        let refused = |bytes: &[u8]| {
            matches!(
                TermDictionary::decode(bytes, "terms"),
                Err(Error::DecodeFailed { .. })
            )
        };
        // Damage the frame catches.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(refused(&bad));
        assert!(refused(&bytes[..bytes.len() - 1]));
        let mut long = bytes.clone();
        long.push(0);
        assert!(refused(&long));
        // Damage the payload's reader catches, with the frame repaired
        // around it. A count the bytes cannot hold.
        let mut bad = bytes.clone();
        bad[h..h + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        frame_fuzz::set_entries(&mut bad, u32::MAX as u64);
        frame_fuzz::repair_trailer(&mut bad);
        assert!(refused(&bad));
        // A count the frame disagrees with.
        let mut bad = bytes.clone();
        bad[h..h + 4].copy_from_slice(&3u32.to_le_bytes());
        frame_fuzz::repair_trailer(&mut bad);
        assert!(refused(&bad));
        // A term that is not UTF-8: the first term is "b" at payload offset 8.
        let mut bad = bytes.clone();
        bad[h + 8] = 0xFF;
        frame_fuzz::repair_trailer(&mut bad);
        assert!(refused(&bad));
        // A repeated term: make the second term "b" as well.
        let mut bad = bytes.clone();
        bad[h + 13] = b'b';
        frame_fuzz::repair_trailer(&mut bad);
        assert!(refused(&bad));
        // Trailing payload bytes.
        let mut long = bytes.clone();
        long.insert(long.len() - 16, 0);
        frame_fuzz::repair_header(&mut long);
        frame_fuzz::repair_trailer(&mut long);
        assert!(refused(&long));
        // An empty term.
        let mut payload = Vec::new();
        payload.extend(1u32.to_le_bytes());
        payload.extend(0u32.to_le_bytes());
        let empty = zeusdb_vector_core::frame(
            FrameKind::TermDictionary,
            FrameEncoding::Engine,
            1,
            &payload,
        );
        assert!(refused(&empty));
        assert!(
            TermDictionary::decode(&TermDictionary::new().encode(), "terms")
                .unwrap()
                .is_empty()
        );
    }

    /// A seeded mutator over a valid artefact never panics the reader. The
    /// frame is repaired around each mutation so the payload's reader sees
    /// it.
    #[test]
    fn no_mutation_of_a_valid_dictionary_panics_the_reader() {
        use zeusdb_vector_core::frame_fuzz;
        let mut d = TermDictionary::new();
        for i in 0..200u32 {
            d.intern(&format!("term{}x{}", i, "é".repeat(i as usize % 4)))
                .unwrap();
        }
        let good = d.encode();
        let mut rng = frame_fuzz::Rng(0x5eed_7e47_e000_0138);
        let cases = std::env::var("ZEUSDB_FUZZ_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_000usize);
        let mut past_frame = 0usize;
        for _ in 0..cases {
            let blob = frame_fuzz::mutate(&mut rng, &good, FrameKind::TermDictionary);
            match TermDictionary::decode(&blob, "terms") {
                Ok(back) => assert!(back.len() <= 200 + 64),
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
    }
}
