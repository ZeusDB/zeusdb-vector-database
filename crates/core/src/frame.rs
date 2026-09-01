//! The frame a whole-buffer artefact is written inside.
//!
//! An artefact an index holds in memory before it writes it, which is every
//! artefact but the graph dump, goes to disk as a 64 byte header, the payload
//! and a 16 byte trailer. The header says what the payload is, how it is
//! encoded, how long it is, how many entries its outermost container holds
//! and what the whole file measures, and it carries a checksum over itself.
//! The trailer carries a checksum over the payload and the magic again.
//!
//! # Why a frame at all
//!
//! Every field a reader sizes an allocation from is checked against the
//! file's own length before the allocation happens, and the check needs the
//! length inside the file, where a manifest cannot be edited apart from it.
//! The magic and the kind let a reader that is not the engine tell one
//! artefact from another by its bytes rather than by its file name, and the
//! payload checksum lets it verify the artefact without the manifest.
//!
//! # The shared first bytes
//!
//! The graph dump's header, in `graph::dump`, opens with an eight byte magic,
//! a `u32` format version, a `u32` of reserved flags and a kind byte, and so
//! does this one, so one reader dispatches on the magic at offset zero and
//! reads either. The two layouts differ from byte 17 on, because the dump's
//! twelve fields are what let its reader derive the file's implied length
//! before allocating and none of them means anything for a payload that
//! carries its own counts.
//!
//! # Layout
//!
//! ```text
//! Header, 64 bytes
//!    0   8  magic            u64   b"ZDBFRAME"
//!    8   4  format_version   u32   1
//!   12   4  flags            u32   reserved, zero
//!   16   1  kind             u8    FrameKind
//!   17   1  encoding         u8    FrameEncoding
//!   18   6  reserved         zero
//!   24   8  payload_bytes    u64   bytes between the header and the trailer
//!   32   8  file_bytes       u64   the whole file, trailer included
//!   40   8  entries          u64   entries in the payload's outermost container
//!   48   8  reserved         u64   zero
//!   56   8  header_checksum  u64   over bytes 0 to 56
//!
//! Payload          payload_bytes
//!
//! Trailer, 16 bytes
//!    0   8  payload_checksum u64   over the payload
//!    8   8  end_magic        u64   b"ZDBFRAME"
//! ```
//!
//! The header is 64 bytes rather than 56 so the payload starts on a cache
//! line. Every field is little-endian at a fixed width.
//!
//! # What the reader checks, in order
//!
//! Length at least the overhead, magic, format version, header checksum,
//! reserved fields zero, kind, encoding, `file_bytes` equal to
//! `payload_bytes` plus the overhead, `file_bytes` equal to the bytes read,
//! end magic, payload checksum. Nothing allocates from a field. `entries` is
//! handed back for the payload's own reader to hold its count to.
//!
//! # The manifest records a length alone
//!
//! The payload checksum is verified here on every read, so a manifest digest
//! over the same bytes would be a second pass over the largest artefacts in
//! the directory for a guarantee this one already gives. A framed artefact
//! is recorded in the manifest by its length, as the graph dump is.

use crate::checksum::checksum_of;
use crate::error::Error;

/// `b"ZDBFRAME"` read little-endian.
pub const FRAME_MAGIC: u64 = u64::from_le_bytes(*b"ZDBFRAME");

/// The only frame version this build writes, and the only one it reads.
const FRAME_VERSION: u32 = 1;

/// Bytes the header occupies, checksum included.
pub const FRAME_HEADER_BYTES: usize = 64;

/// Bytes the header checksum is taken over, being everything before it.
const HEADER_CHECKSUM_BYTES: usize = 56;

/// Bytes the trailer occupies.
pub const FRAME_TRAILER_BYTES: usize = 16;

/// Bytes a frame adds to its payload.
pub const FRAME_OVERHEAD_BYTES: usize = FRAME_HEADER_BYTES + FRAME_TRAILER_BYTES;

/// What a framed payload holds.
///
/// **The numbers are on disk. Never reuse one and never change one.** One to
/// four are reserved for the id mappings, the raw vectors, the quantized
/// codes and the codebook, which are not framed by this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// A sparse space's postings, being every live record's id and vector.
    SparsePostings = 5,
    /// A text layer's term dictionary, being every term in id order.
    TermDictionary = 6,
}

impl FrameKind {
    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            5 => Some(FrameKind::SparsePostings),
            6 => Some(FrameKind::TermDictionary),
            _ => None,
        }
    }

    /// How the kind names itself in a message.
    pub fn label(self) -> &'static str {
        match self {
            FrameKind::SparsePostings => "sparse postings",
            FrameKind::TermDictionary => "term dictionary",
        }
    }
}

/// How a framed payload is encoded.
///
/// **The numbers are on disk.** One is reserved for a bincode payload, which
/// this build does not frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameEncoding {
    /// The engine's own layout, being fixed-width little-endian fields whose
    /// order the artefact's reader states.
    Engine = 2,
}

impl FrameEncoding {
    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            2 => Some(FrameEncoding::Engine),
            _ => None,
        }
    }
}

/// A frame read back, borrowing its payload from the bytes it was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Framed<'a> {
    pub encoding: FrameEncoding,
    /// Entries in the payload's outermost container, as the writer counted
    /// them. The payload's reader holds its own count to this.
    pub entries: u64,
    pub payload: &'a [u8],
}

/// Wrap `payload` in a frame.
///
/// One copy of the payload. A writer that builds its payload itself avoids
/// even that by appending it into the buffer [`begin`] returns and closing
/// the frame with [`finish`], which is what the two artefact writers do: a
/// second buffer for a 70 MB payload cost a save a quarter of its time in
/// the allocation's first touch.
pub fn frame(kind: FrameKind, encoding: FrameEncoding, entries: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = begin(kind, encoding, payload.len());
    out.extend_from_slice(payload);
    finish(out, entries)
}

/// A buffer with room reserved for the header, into which a writer appends
/// its payload before [`finish`] closes the frame. `payload_hint` sizes the
/// buffer once.
pub fn begin(kind: FrameKind, encoding: FrameEncoding, payload_hint: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload_hint + FRAME_OVERHEAD_BYTES);
    out.resize(FRAME_HEADER_BYTES, 0);
    out[0..8].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    out[8..12].copy_from_slice(&FRAME_VERSION.to_le_bytes());
    out[16] = kind.code();
    out[17] = encoding.code();
    out
}

/// Close a frame [`begin`] opened, once the payload is appended: the two
/// lengths and the entry count go into the header, the header checksum is
/// taken, and the trailer is appended.
pub fn finish(mut out: Vec<u8>, entries: u64) -> Vec<u8> {
    debug_assert!(out.len() >= FRAME_HEADER_BYTES, "a frame begun with begin");
    let payload_bytes = (out.len() - FRAME_HEADER_BYTES) as u64;
    let file_bytes = payload_bytes + FRAME_OVERHEAD_BYTES as u64;
    out[24..32].copy_from_slice(&payload_bytes.to_le_bytes());
    out[32..40].copy_from_slice(&file_bytes.to_le_bytes());
    out[40..48].copy_from_slice(&entries.to_le_bytes());
    let sum = checksum_of(&out[..HEADER_CHECKSUM_BYTES]);
    out[56..64].copy_from_slice(&sum.to_le_bytes());
    let payload_sum = checksum_of(&out[FRAME_HEADER_BYTES..]);
    out.extend_from_slice(&payload_sum.to_le_bytes());
    out.extend_from_slice(&FRAME_MAGIC.to_le_bytes());
    out
}

/// Read a frame back, refusing each way it can be wrong by name. `file`
/// names the artefact in the message.
pub fn unframe<'a>(bytes: &'a [u8], kind: FrameKind, file: &str) -> Result<Framed<'a>, Error> {
    let corrupt = |detail: String| Error::DecodeFailed {
        file: file.to_string(),
        error: detail,
    };
    if bytes.len() < FRAME_OVERHEAD_BYTES {
        return Err(corrupt(format!(
            "the file holds {} bytes and a frame is at least {}",
            bytes.len(),
            FRAME_OVERHEAD_BYTES
        )));
    }
    let header = &bytes[..FRAME_HEADER_BYTES];
    if u64::from_le_bytes(take8(header, 0)) != FRAME_MAGIC {
        return Err(corrupt(
            "the file does not open with the frame magic".into(),
        ));
    }
    let version = u32::from_le_bytes(take4(header, 8));
    if version != FRAME_VERSION {
        return Err(corrupt(format!(
            "the frame is format version {} and this build reads {}",
            version, FRAME_VERSION
        )));
    }
    let stored = u64::from_le_bytes(take8(header, 56));
    if stored != checksum_of(&header[..HEADER_CHECKSUM_BYTES]) {
        return Err(corrupt("the frame's header is corrupt".into()));
    }
    let flags = u32::from_le_bytes(take4(header, 12));
    let reserved_short = &header[18..24];
    let reserved_long = u64::from_le_bytes(take8(header, 48));
    if flags != 0 || reserved_short.iter().any(|&b| b != 0) || reserved_long != 0 {
        return Err(corrupt(
            "the frame sets a reserved field this build does not know".into(),
        ));
    }
    let found = FrameKind::from_code(header[16]).ok_or_else(|| {
        corrupt(format!(
            "the frame names kind {}, which is not one this build writes",
            header[16]
        ))
    })?;
    if found != kind {
        return Err(corrupt(format!(
            "the frame holds {} where {} was expected",
            found.label(),
            kind.label()
        )));
    }
    let encoding = FrameEncoding::from_code(header[17]).ok_or_else(|| {
        corrupt(format!(
            "the frame names encoding {}, which is not one this build reads",
            header[17]
        ))
    })?;
    let payload_bytes = u64::from_le_bytes(take8(header, 24));
    let file_bytes = u64::from_le_bytes(take8(header, 32));
    let entries = u64::from_le_bytes(take8(header, 40));
    let implied = payload_bytes
        .checked_add(FRAME_OVERHEAD_BYTES as u64)
        .ok_or_else(|| corrupt("the frame's payload length overflows".into()))?;
    if file_bytes != implied {
        return Err(corrupt(format!(
            "the frame declares {} file bytes and a payload of {} implies {}",
            file_bytes, payload_bytes, implied
        )));
    }
    if file_bytes != bytes.len() as u64 {
        return Err(corrupt(format!(
            "the frame declares {} file bytes and the file holds {}",
            file_bytes,
            bytes.len()
        )));
    }
    let payload_end = bytes.len() - FRAME_TRAILER_BYTES;
    let trailer = &bytes[payload_end..];
    if u64::from_le_bytes(take8(trailer, 8)) != FRAME_MAGIC {
        return Err(corrupt("the file does not end with the frame magic".into()));
    }
    let payload = &bytes[FRAME_HEADER_BYTES..payload_end];
    if u64::from_le_bytes(take8(trailer, 0)) != checksum_of(payload) {
        return Err(corrupt("the frame's payload is corrupt".into()));
    }
    Ok(Framed {
        encoding,
        entries,
        payload,
    })
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

/// The repairs a mutator applies after it has damaged a frame, so a mutation
/// reaches the payload's reader rather than being refused by a checksum.
/// For the fuzzers in the crates whose payloads are framed.
pub mod fuzz {
    use super::*;

    /// Recompute the header checksum and rewrite `payload_bytes` and
    /// `file_bytes` to the length the frame really has, so the size
    /// agreement holds and the header verifies.
    pub fn repair_header(blob: &mut [u8]) {
        if blob.len() < FRAME_OVERHEAD_BYTES {
            return;
        }
        let total = blob.len() as u64;
        let payload = total - FRAME_OVERHEAD_BYTES as u64;
        blob[24..32].copy_from_slice(&payload.to_le_bytes());
        blob[32..40].copy_from_slice(&total.to_le_bytes());
        let sum = checksum_of(&blob[..HEADER_CHECKSUM_BYTES]);
        blob[56..64].copy_from_slice(&sum.to_le_bytes());
    }

    /// Recompute the header checksum alone, leaving the two lengths as the
    /// mutation left them.
    pub fn restamp_header(blob: &mut [u8]) {
        if blob.len() < FRAME_HEADER_BYTES {
            return;
        }
        let sum = checksum_of(&blob[..HEADER_CHECKSUM_BYTES]);
        blob[56..64].copy_from_slice(&sum.to_le_bytes());
    }

    /// Recompute the payload checksum and restamp the end magic.
    pub fn repair_trailer(blob: &mut [u8]) {
        if blob.len() < FRAME_OVERHEAD_BYTES {
            return;
        }
        let end = blob.len() - FRAME_TRAILER_BYTES;
        let sum = checksum_of(&blob[FRAME_HEADER_BYTES..end]);
        blob[end..end + 8].copy_from_slice(&sum.to_le_bytes());
        blob[end + 8..].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    }

    /// Write `entries` into the header and restamp its checksum.
    pub fn set_entries(blob: &mut [u8], entries: u64) {
        if blob.len() < FRAME_HEADER_BYTES {
            return;
        }
        blob[40..48].copy_from_slice(&entries.to_le_bytes());
        restamp_header(blob);
    }

    /// splitmix64, which is what every fuzzer in the workspace draws from.
    pub struct Rng(pub u64);

    impl Rng {
        pub fn draw(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        /// A value below `bound`, which is zero when `bound` is zero.
        pub fn below(&mut self, bound: usize) -> usize {
            if bound == 0 {
                0
            } else {
                (self.draw() % bound as u64) as usize
            }
        }

        pub fn byte(&mut self) -> u8 {
            (self.draw() >> 24) as u8
        }
    }

    /// Values a field mutation writes, being the edges of every width.
    pub const HOSTILE: [u64; 12] = [
        0,
        1,
        2,
        255,
        256,
        u32::MAX as u64 - 1,
        u32::MAX as u64,
        u32::MAX as u64 + 1,
        i32::MAX as u64,
        i32::MAX as u64 + 1,
        u64::MAX / 2,
        u64::MAX,
    ];

    /// Apply up to three mutations to a valid framed artefact and repair the
    /// frame so the payload's reader sees them. Returns whether the frame
    /// verifies afterwards, which is what a fuzzer counts to prove its
    /// mutations reach past the checksums.
    ///
    /// Half the draws land inside the payload, where the counts and lengths
    /// the artefact's reader defends itself with live. The rest land on the
    /// header's `entries`, the frame's own fields, or the file's length.
    pub fn mutate(rng: &mut Rng, base: &[u8], kind: FrameKind) -> Vec<u8> {
        let mut blob = base.to_vec();
        let ops = 1 + rng.below(3);
        let mut header_touched = false;
        for _ in 0..ops {
            let len = blob.len();
            match rng.below(10) {
                0..=3 => {
                    // A hostile value at a random offset inside the payload,
                    // at a random width.
                    if len > FRAME_OVERHEAD_BYTES {
                        let width = [1usize, 2, 4, 8][rng.below(4)];
                        let at = FRAME_HEADER_BYTES + rng.below(len - FRAME_OVERHEAD_BYTES);
                        let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                        let end = (at + width).min(len - FRAME_TRAILER_BYTES);
                        blob[at..end].copy_from_slice(&value[..end - at]);
                    }
                }
                4 => {
                    // One byte of the payload set at random.
                    if len > FRAME_OVERHEAD_BYTES {
                        let at = FRAME_HEADER_BYTES + rng.below(len - FRAME_OVERHEAD_BYTES);
                        blob[at] = rng.byte();
                    }
                }
                5 => {
                    // Cut the file short, inside the payload or the trailer.
                    let cut = rng.below(len + 1);
                    blob.truncate(cut);
                }
                6 => {
                    // Append bytes.
                    let extra = 1 + rng.below(64);
                    let value = rng.byte();
                    blob.extend(std::iter::repeat_n(value, extra));
                }
                7 => {
                    // A hostile entry count.
                    let value = HOSTILE[rng.below(HOSTILE.len())];
                    if blob.len() >= FRAME_HEADER_BYTES {
                        blob[40..48].copy_from_slice(&value.to_le_bytes());
                    }
                    header_touched = true;
                }
                8 => {
                    // A frame field, being the version, the flags, the kind,
                    // the encoding or a reserved byte.
                    if blob.len() >= FRAME_HEADER_BYTES {
                        let (at, width) = [
                            (8usize, 4usize),
                            (12, 4),
                            (16, 1),
                            (17, 1),
                            (18, 6),
                            (48, 8),
                        ][rng.below(6)];
                        let value = HOSTILE[rng.below(HOSTILE.len())].to_le_bytes();
                        blob[at..at + width].copy_from_slice(&value[..width]);
                    }
                    header_touched = true;
                }
                _ => {
                    // A run of the payload zeroed.
                    if len > FRAME_OVERHEAD_BYTES {
                        let at = FRAME_HEADER_BYTES + rng.below(len - FRAME_OVERHEAD_BYTES);
                        let run = 1 + rng.below(32);
                        let end = (at + run).min(len - FRAME_TRAILER_BYTES);
                        blob[at..end].fill(0);
                    }
                }
            }
        }
        // Repairs, so the mutation is judged by the payload's reader rather
        // than refused at the frame. One case in eight is left unrepaired,
        // so the frame's own refusals stay in the corpus.
        if rng.below(8) != 0 {
            if header_touched {
                restamp_header(&mut blob);
                // The two lengths are made true whatever the mutation did to
                // them, unless the frame field mutation is what is being
                // tested, in which case the header stays as mutated.
                if rng.below(2) == 0 {
                    repair_header(&mut blob);
                }
            } else {
                repair_header(&mut blob);
            }
            repair_trailer(&mut blob);
        }
        let _ = kind;
        blob
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(bytes: &[u8], kind: FrameKind) -> String {
        match unframe(bytes, kind, "t") {
            Err(Error::DecodeFailed { error, .. }) => error,
            other => panic!("expected a decode failure, got {:?}", other.map(|_| ())),
        }
    }

    /// A framed payload reads back, with its entry count and encoding, and
    /// an empty payload frames to exactly the overhead.
    #[test]
    fn a_frame_round_trips_and_costs_eighty_bytes() {
        let payload: Vec<u8> = (0..1000u32).map(|i| (i % 253) as u8).collect();
        let bytes = frame(
            FrameKind::SparsePostings,
            FrameEncoding::Engine,
            42,
            &payload,
        );
        assert_eq!(bytes.len(), payload.len() + FRAME_OVERHEAD_BYTES);
        assert_eq!(&bytes[..8], b"ZDBFRAME");
        assert_eq!(&bytes[bytes.len() - 8..], b"ZDBFRAME");
        let back = unframe(&bytes, FrameKind::SparsePostings, "t").unwrap();
        assert_eq!(back.entries, 42);
        assert_eq!(back.encoding, FrameEncoding::Engine);
        assert_eq!(back.payload, &payload[..]);

        let empty = frame(FrameKind::TermDictionary, FrameEncoding::Engine, 0, &[]);
        assert_eq!(empty.len(), FRAME_OVERHEAD_BYTES);
        let back = unframe(&empty, FrameKind::TermDictionary, "t").unwrap();
        assert_eq!(back.entries, 0);
        assert!(back.payload.is_empty());
    }

    /// The first seventeen bytes mean what the graph dump's mean, so a
    /// reader dispatching on the magic tells the two apart at byte zero.
    #[test]
    fn the_header_shares_its_first_seventeen_bytes_with_the_dump() {
        let bytes = frame(FrameKind::SparsePostings, FrameEncoding::Engine, 1, &[7]);
        assert_eq!(u64::from_le_bytes(take8(&bytes, 0)), FRAME_MAGIC);
        assert_ne!(FRAME_MAGIC, u64::from_le_bytes(*b"ZDBGRAPH"));
        assert_eq!(u32::from_le_bytes(take4(&bytes, 8)), 1);
        assert_eq!(u32::from_le_bytes(take4(&bytes, 12)), 0);
        assert_eq!(bytes[16], FrameKind::SparsePostings as u8);
    }

    /// Every way a frame can be wrong is refused by name, in the reader's
    /// order, and nothing panics.
    #[test]
    fn every_damage_is_refused_by_name() {
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let good = frame(
            FrameKind::SparsePostings,
            FrameEncoding::Engine,
            3,
            &payload,
        );
        let kind = FrameKind::SparsePostings;

        assert!(refused(&good[..FRAME_OVERHEAD_BYTES - 1], kind).contains("at least"));
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(refused(&bad, kind).contains("frame magic"));
        let mut bad = good.clone();
        bad[8] = 2;
        fuzz::restamp_header(&mut bad);
        assert!(refused(&bad, kind).contains("format version 2"));
        let mut bad = good.clone();
        bad[40] ^= 1;
        assert!(refused(&bad, kind).contains("header is corrupt"));
        let mut bad = good.clone();
        bad[12] = 1;
        fuzz::restamp_header(&mut bad);
        assert!(refused(&bad, kind).contains("reserved field"));
        let mut bad = good.clone();
        bad[20] = 1;
        fuzz::restamp_header(&mut bad);
        assert!(refused(&bad, kind).contains("reserved field"));
        let mut bad = good.clone();
        bad[16] = 9;
        fuzz::restamp_header(&mut bad);
        assert!(refused(&bad, kind).contains("kind 9"));
        assert!(refused(&good, FrameKind::TermDictionary).contains("holds sparse postings"));
        let mut bad = good.clone();
        bad[17] = 1;
        fuzz::restamp_header(&mut bad);
        assert!(refused(&bad, kind).contains("encoding 1"));
        let mut bad = good.clone();
        bad[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        fuzz::restamp_header(&mut bad);
        assert!(refused(&bad, kind).contains("overflows"));
        let mut bad = good.clone();
        bad[24..32].copy_from_slice(&8u64.to_le_bytes());
        fuzz::restamp_header(&mut bad);
        assert!(refused(&bad, kind).contains("implies"));
        let mut bad = good.clone();
        bad.push(0);
        assert!(refused(&bad, kind).contains("the file holds"));
        let mut bad = good.clone();
        let end = bad.len() - 1;
        bad[end] = b'X';
        assert!(refused(&bad, kind).contains("end with"));
        let mut bad = good.clone();
        bad[FRAME_HEADER_BYTES + 4] ^= 0x10;
        assert!(refused(&bad, kind).contains("payload is corrupt"));
    }

    /// A seeded mutator over a valid frame never panics the reader, and its
    /// repairs carry mutations past the checksums into the payload, which
    /// is measured rather than assumed.
    #[test]
    fn no_mutation_of_a_valid_frame_panics_the_reader() {
        let payload: Vec<u8> = (0..4096u32)
            .map(|i| (i.wrapping_mul(31) % 251) as u8)
            .collect();
        let good = frame(
            FrameKind::TermDictionary,
            FrameEncoding::Engine,
            16,
            &payload,
        );
        let mut rng = fuzz::Rng(0x5eed_f4a3_e000_0138);
        let cases = 3_000;
        let mut verified = 0usize;
        let mut mutated_payloads = 0usize;
        for _ in 0..cases {
            let blob = fuzz::mutate(&mut rng, &good, FrameKind::TermDictionary);
            if let Ok(framed) = unframe(&blob, FrameKind::TermDictionary, "t") {
                verified += 1;
                if framed.payload != &payload[..] || framed.entries != 16 {
                    mutated_payloads += 1;
                }
            }
        }
        // The repairs are what make the fuzzer worth running. Without them a
        // mutated frame is refused by a checksum before any field is read.
        assert!(
            verified * 2 > cases,
            "only {} of {} mutated frames verified",
            verified,
            cases
        );
        assert!(
            mutated_payloads * 4 > cases,
            "only {} of {} mutations reached the payload",
            mutated_payloads,
            cases
        );
    }
}
