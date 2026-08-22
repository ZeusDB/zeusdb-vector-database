//! The one checksum this crate computes, over any byte stream.
//!
//! It was declared inside `graph::dump`, which is where the first thing that
//! needed it lives. `manifest.json` now records a digest per artefact and reads
//! it back on load, so a second caller exists and the algorithm belongs to
//! neither of them.
//!
//! **The bytes it produces are unchanged.** The constants, the word order, the
//! carry buffer, the length fold and the final avalanche are the ones
//! `graph::dump` has always written into a dump's header and trailer, so a dump
//! written before this module existed still verifies and a dump written after
//! it is byte-identical to one written before.

/// A 64 bit checksum over a byte stream, consuming eight bytes per step.
///
/// FNV-1a's constants over whole words rather than single bytes, with a shift
/// mixed in so that a word landing in the high bits still reaches the low ones,
/// the total length folded in at the end so that appended zeros change the
/// answer, and a final avalanche. It detects corruption. It is not a signature
/// and nothing here treats it as one.
///
/// Whole words rather than bytes because the vector region of a 50,000 record
/// dump at dimension 1,536 is 307 MB and a byte at a time would cost more than
/// the read it is protecting.
///
/// A carry buffer holds a partial word between calls, so the answer depends on
/// the bytes alone and not on how they were split across writes.
pub(crate) struct Checksum {
    state: u64,
    carry: [u8; 8],
    carry_len: usize,
    len: u64,
}

/// FNV-1a's 64 bit offset basis.
const CHECKSUM_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a's 64 bit prime.
const CHECKSUM_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The multiplier of the final avalanche, taken from `splitmix64`.
const CHECKSUM_AVALANCHE: u64 = 0xff51_afd7_ed55_8ccd;

impl Checksum {
    pub(crate) fn new() -> Self {
        Checksum {
            state: CHECKSUM_SEED,
            carry: [0; 8],
            carry_len: 0,
            len: 0,
        }
    }

    fn word(&mut self, word: u64) {
        self.state ^= word;
        self.state = self.state.wrapping_mul(CHECKSUM_PRIME);
        self.state ^= self.state >> 29;
    }

    pub(crate) fn write(&mut self, bytes: &[u8]) {
        self.len = self.len.wrapping_add(bytes.len() as u64);
        let mut rest = bytes;
        if self.carry_len > 0 {
            let take = (8 - self.carry_len).min(rest.len());
            self.carry[self.carry_len..self.carry_len + take].copy_from_slice(&rest[..take]);
            self.carry_len += take;
            rest = &rest[take..];
            if self.carry_len < 8 {
                return;
            }
            let word = u64::from_le_bytes(self.carry);
            self.word(word);
            self.carry_len = 0;
        }
        let mut chunks = rest.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.word(u64::from_le_bytes(word));
        }
        let tail = chunks.remainder();
        if !tail.is_empty() {
            self.carry = [0; 8];
            self.carry[..tail.len()].copy_from_slice(tail);
            self.carry_len = tail.len();
        }
    }

    pub(crate) fn finish(mut self) -> u64 {
        if self.carry_len > 0 {
            let mut word = [0u8; 8];
            word[..self.carry_len].copy_from_slice(&self.carry[..self.carry_len]);
            self.word(u64::from_le_bytes(word));
        }
        let len = self.len;
        self.word(len);
        let mut hash = self.state;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(CHECKSUM_AVALANCHE);
        hash ^= hash >> 33;
        hash
    }
}

/// Checksum a slice that is already in memory, being the header.
pub(crate) fn checksum_of(bytes: &[u8]) -> u64 {
    let mut sum = Checksum::new();
    sum.write(bytes);
    sum.finish()
}
