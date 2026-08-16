//! ZeusDB's own on-disk format for the graph.
//!
//! One file, `hnsw_index.zdbgraph`, holding everything a traversal needs and
//! nothing else. It replaces the vendored crate's two file dump, which ZeusDB
//! wrote through `Hnsw::file_dump` and read through `HnswIo`.
//!
//! # Why it exists
//!
//! Five things came with the vendored format and all five leave with it.
//!
//! The vendored header carries `std::any::type_name::<D>()` and the vendored
//! reload compares it by exact equality, so every distance type was pinned to
//! the module it was declared in and moving one stopped every saved index from
//! loading. This header carries a [`GraphKind`] discriminant instead.
//!
//! The vendored reload returns a graph whose lifetime is tied to the `HnswIo`
//! that produced it, so reaching `'static` meant leaking the loader. The reader
//! here owns no borrowed state and returns `Hnsw<'static, T, D>` outright.
//!
//! The vendored reader panics on a malformed header and reaches
//! `std::process::exit(1)` on a short data file, so the caller wrapped it in two
//! `catch_unwind` calls and measured the data file's length beforehand. Every
//! malformed input here returns an error.
//!
//! `max_nb_connection` is a `u8` in the vendored header where `Hnsw::new`
//! admits 256, so an index at `m` 256 declared 0 and rebuilt on every load. It
//! is a `u64` here.
//!
//! # Byte order and widths
//!
//! Little-endian throughout, and every width fixed. The vendored format wrote
//! `to_ne_bytes` and native `usize`, so a dump was tied to both the endianness
//! and the pointer width of the machine that wrote it, and neither fact was
//! recorded anywhere in the file. Every target ZeusDB ships is little-endian,
//! so the choice costs nothing on all of them and makes the file portable by
//! construction rather than by coincidence.
//!
//! # Layout
//!
//! ```text
//! Header, 96 bytes
//!    0   8  magic             u64   b"ZDBGRAPH"
//!    8   4  format_version    u32
//!   12   4  flags             u32   reserved, zero
//!   16   1  distance          u8    GraphKind
//!   17   1  element           u8    1 f32, 2 u8
//!   18   1  nb_layer          u8
//!   19   1  reserved0         u8    zero
//!   20   4  dimension         u32
//!   24   8  m                 u64
//!   32   8  ef_construction   u64
//!   40   8  nb_point          u64
//!   48   8  level_scale       f64
//!   56   4  entry_layer       u32
//!   60   4  entry_rank        u32
//!   64   8  adjacency_bytes   u64
//!   72   8  file_bytes        u64   the whole file, trailer included
//!   80   8  reserved1         u64   zero
//!   88   8  header_checksum   u64   over bytes 0 to 88
//!
//! Layer table      nb_layer * u32, points per layer
//! Origin ids       nb_point * u64, in (layer, rank) order
//! Adjacency        adjacency_bytes, per point in (layer, rank) order
//!                    u8  list_count, trailing empty layers trimmed
//!                    per list: u32 edge_count, then edge_count entries of
//!                      u8 target_layer, u32 target_rank, f32 distance
//! Vectors          nb_point * dimension * element width, (layer, rank) order
//!
//! Trailer, 16 bytes
//!    0   8  payload_checksum  u64   over everything from 96 to here
//!    8   8  end_magic         u64
//! ```
//!
//! A point's own `PointId` is not a field. It is the point's position, which is
//! the same identity the vendored reload asserts rather than reads. A
//! neighbour's `d_id` is not a field either, because it is the target point's
//! own origin id and the target is already named by its position. The vendored
//! format spends eight bytes per entry on it, which at the 1,482,051 edges a
//! 50,000 record dbpedia index of dimension 1,536 at `m` 16 holds is 11.9 MB.
//!
//! # How truncation is caught
//!
//! `file_bytes` sits in the header and is compared against the file's own
//! length before anything is allocated, so a header claiming a billion points
//! is rejected on a file that cannot hold a billion points. `header_checksum`
//! covers every field that comparison depends on, so a corrupted `nb_point`
//! cannot pass itself off as a consistent length. `payload_checksum` and
//! `end_magic` catch corruption inside the body and are verified as the body
//! streams past.
//!
//! The magic is written last, after the body is on disk. A save that dies part
//! way through leaves zeros at offset zero, so an incomplete dump is
//! unreadable by construction and reaches the rebuild rather than being read as
//! though it were whole.

use hnsw_rs::hnsw::{LoadedEdge, LoadedPoint, PointId, NB_LAYER_MAX};
use hnsw_rs::prelude::{Distance, Hnsw};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use tracing::{info, warn};

/// The one file a dump writes, and the one the loader reads.
pub(crate) const DUMP_FILENAME: &str = "hnsw_index.zdbgraph";

/// What 0.6.0 and earlier wrote, and what a save removes once it has replaced
/// them.
///
/// A directory saved by an earlier release keeps these two until it is saved
/// again. Leaving them behind after that save would leave a 50,000 record raw
/// index carrying 340 MB of a graph nothing will ever read, beside the 322 MB
/// of the graph that replaced it.
const LEGACY_DUMP_FILENAMES: [&str; 2] = ["hnsw_index.hnsw.graph", "hnsw_index.hnsw.data"];

/// `b"ZDBGRAPH"` read little-endian.
const MAGIC: u64 = u64::from_le_bytes(*b"ZDBGRAPH");

/// The only format version this build writes, and the only one it reads.
const FORMAT_VERSION: u32 = 1;

/// Bytes the header occupies, header checksum included.
const HEADER_BYTES: usize = 96;

/// Bytes the header checksum is taken over, being everything before it.
const HEADER_CHECKSUM_BYTES: usize = 88;

/// Bytes the trailer occupies, being the payload checksum and the end magic.
const TRAILER_BYTES: usize = 16;

/// Bytes one adjacency entry occupies, being a layer, a rank and a distance.
const EDGE_BYTES: usize = 9;

/// Bytes one point's origin id occupies.
const ORIGIN_ID_BYTES: usize = 8;

/// Bytes one entry of the layer table occupies.
const LAYER_TABLE_ENTRY_BYTES: usize = 4;

/// Bytes buffered on the way to and from the file.
const IO_BUFFER_BYTES: usize = 1 << 20;

/// Which of the six graphs a dump holds.
///
/// This is what replaces `std::any::type_name::<D>()` in the vendored header.
/// A discriminant is a value ZeusDB chose, so the distance types are free to
/// move between modules, be renamed, or be replaced outright without a saved
/// index becoming unreadable. Only the mapping from a type to its number here
/// is load bearing, and it is written down in one place.
///
/// The three quantized variants share one distance type, `DistPQ`, so the
/// vendored header could not tell them apart and the space came from
/// `config.json` alone. They are distinct here, so a dump and a `config.json`
/// that disagree are now caught rather than silently accepted.
///
/// **The numbers are on disk. Never reuse one and never change one.**
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GraphKind {
    Cosine = 1,
    L2 = 2,
    L1 = 3,
    CosinePq = 4,
    L2Pq = 5,
    L1Pq = 6,
}

impl GraphKind {
    fn code(self) -> u8 {
        self as u8
    }

    /// How the dump names itself in an error a user reads, and in a log line.
    pub(crate) fn label(self) -> &'static str {
        match self {
            GraphKind::Cosine => "cosine",
            GraphKind::L2 => "l2",
            GraphKind::L1 => "l1",
            GraphKind::CosinePq => "quantized cosine",
            GraphKind::L2Pq => "quantized l2",
            GraphKind::L1Pq => "quantized l1",
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(GraphKind::Cosine),
            2 => Some(GraphKind::L2),
            3 => Some(GraphKind::L1),
            4 => Some(GraphKind::CosinePq),
            5 => Some(GraphKind::L2Pq),
            6 => Some(GraphKind::L1Pq),
            _ => None,
        }
    }
}

/// What a point's data is made of, on disk and in memory.
///
/// The graph stores `f32` vectors when it is raw and `u8` codes when it is
/// quantized, and the dump has to encode either without the reader guessing.
/// The `KIND` byte is written into the header and checked on the way back, so a
/// quantized dump handed to the raw loader is rejected on the header rather
/// than misread as a vector a quarter the width.
pub(crate) trait DumpElement:
    Copy + Clone + Send + Sync + std::fmt::Debug + 'static
{
    /// The number the header records for this element type.
    const KIND: u8;
    /// Bytes one value occupies on disk.
    const BYTES: usize;
    /// Append `values` to `buf` little-endian.
    fn encode(values: &[Self], buf: &mut Vec<u8>);
    /// Read exactly `bytes.len() / BYTES` values out of `bytes`.
    fn decode(bytes: &[u8]) -> Vec<Self>;
}

impl DumpElement for f32 {
    const KIND: u8 = 1;
    const BYTES: usize = 4;

    fn encode(values: &[Self], buf: &mut Vec<u8>) {
        for value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn decode(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(Self::BYTES)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}

impl DumpElement for u8 {
    const KIND: u8 = 2;
    const BYTES: usize = 1;

    fn encode(values: &[Self], buf: &mut Vec<u8>) {
        buf.extend_from_slice(values);
    }

    fn decode(bytes: &[u8]) -> Vec<Self> {
        bytes.to_vec()
    }
}

// ============================================================================
// CHECKSUM
// ============================================================================

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
struct Checksum {
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
    fn new() -> Self {
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

    fn write(&mut self, bytes: &[u8]) {
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

    fn finish(mut self) -> u64 {
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
fn checksum_of(bytes: &[u8]) -> u64 {
    let mut sum = Checksum::new();
    sum.write(bytes);
    sum.finish()
}

// ============================================================================
// A CURSOR OVER THE BYTES ALREADY READ
// ============================================================================

/// A reader that checksums what it hands out and counts what it has read.
struct HashingReader<R: Read> {
    inner: R,
    sum: Checksum,
}

impl<R: Read> HashingReader<R> {
    fn new(inner: R) -> Self {
        HashingReader {
            inner,
            sum: Checksum::new(),
        }
    }

    fn read_exact_hashed(&mut self, buf: &mut [u8]) -> Result<(), String> {
        self.inner
            .read_exact(buf)
            .map_err(|e| format!("the graph dump ended early: {}", e))?;
        self.sum.write(buf);
        Ok(())
    }
}

/// A writer that checksums what it takes and counts what it has written.
struct HashingWriter<W: Write> {
    inner: W,
    sum: Checksum,
    written: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        HashingWriter {
            inner,
            sum: Checksum::new(),
            written: 0,
        }
    }

    fn put(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.inner
            .write_all(bytes)
            .map_err(|e| format!("the graph dump could not be written: {}", e))?;
        self.sum.write(bytes);
        self.written += bytes.len() as u64;
        Ok(())
    }

    fn put_u8(&mut self, value: u8) -> Result<(), String> {
        self.put(&[value])
    }

    fn put_u32(&mut self, value: u32) -> Result<(), String> {
        self.put(&value.to_le_bytes())
    }

    fn put_u64(&mut self, value: u64) -> Result<(), String> {
        self.put(&value.to_le_bytes())
    }
}

// ============================================================================
// THE HEADER
// ============================================================================

/// The header, as fields rather than as bytes.
struct Header {
    kind: GraphKind,
    element: u8,
    nb_layer: u8,
    dimension: u32,
    m: u64,
    ef_construction: u64,
    nb_point: u64,
    level_scale: f64,
    entry_layer: u32,
    entry_rank: u32,
    adjacency_bytes: u64,
    file_bytes: u64,
}

impl Header {
    /// The header as the 96 bytes that go on disk, checksum included.
    fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[0..8].copy_from_slice(&MAGIC.to_le_bytes());
        out[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        // 12..16 flags, 19 reserved0 and 80..88 reserved1 stay zero.
        out[16] = self.kind.code();
        out[17] = self.element;
        out[18] = self.nb_layer;
        out[20..24].copy_from_slice(&self.dimension.to_le_bytes());
        out[24..32].copy_from_slice(&self.m.to_le_bytes());
        out[32..40].copy_from_slice(&self.ef_construction.to_le_bytes());
        out[40..48].copy_from_slice(&self.nb_point.to_le_bytes());
        out[48..56].copy_from_slice(&self.level_scale.to_le_bytes());
        out[56..60].copy_from_slice(&self.entry_layer.to_le_bytes());
        out[60..64].copy_from_slice(&self.entry_rank.to_le_bytes());
        out[64..72].copy_from_slice(&self.adjacency_bytes.to_le_bytes());
        out[72..80].copy_from_slice(&self.file_bytes.to_le_bytes());
        let sum = checksum_of(&out[..HEADER_CHECKSUM_BYTES]);
        out[88..96].copy_from_slice(&sum.to_le_bytes());
        out
    }

    /// Read the header back, rejecting anything this build cannot interpret.
    ///
    /// Nothing here allocates from a field, so a header claiming a billion
    /// points costs the 96 bytes it was read from and nothing else. The size
    /// agreement in `read_dump` is what makes the later allocations safe.
    fn decode(raw: &[u8; HEADER_BYTES]) -> Result<Self, String> {
        let magic = u64::from_le_bytes(take8(raw, 0));
        if magic != MAGIC {
            return Err("the file is not a ZeusDB graph dump".to_string());
        }
        let version = u32::from_le_bytes(take4(raw, 8));
        if version != FORMAT_VERSION {
            return Err(format!(
                "the graph dump is format version {} and this build reads {}",
                version, FORMAT_VERSION
            ));
        }
        let stored = u64::from_le_bytes(take8(raw, 88));
        let computed = checksum_of(&raw[..HEADER_CHECKSUM_BYTES]);
        if stored != computed {
            return Err("the graph dump's header is corrupt".to_string());
        }
        let flags = u32::from_le_bytes(take4(raw, 12));
        let reserved1 = u64::from_le_bytes(take8(raw, 80));
        if flags != 0 || raw[19] != 0 || reserved1 != 0 {
            return Err(
                "the graph dump sets a reserved field this build does not know".to_string(),
            );
        }
        let kind = GraphKind::from_code(raw[16]).ok_or_else(|| {
            format!(
                "the graph dump names distance {}, which is not one this build writes",
                raw[16]
            )
        })?;
        Ok(Header {
            kind,
            element: raw[17],
            nb_layer: raw[18],
            dimension: u32::from_le_bytes(take4(raw, 20)),
            m: u64::from_le_bytes(take8(raw, 24)),
            ef_construction: u64::from_le_bytes(take8(raw, 32)),
            nb_point: u64::from_le_bytes(take8(raw, 40)),
            level_scale: f64::from_le_bytes(take8(raw, 48)),
            entry_layer: u32::from_le_bytes(take4(raw, 56)),
            entry_rank: u32::from_le_bytes(take4(raw, 60)),
            adjacency_bytes: u64::from_le_bytes(take8(raw, 64)),
            file_bytes: u64::from_le_bytes(take8(raw, 72)),
        })
    }
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

// ============================================================================
// WRITING
// ============================================================================

/// Write the graph to `dir` in ZeusDB's format.
///
/// Three passes over the layers, because the regions are laid out one after
/// another rather than interleaved per point. Only the adjacency pass
/// allocates, and it allocates one point's neighbourhood at a time, so the peak
/// is flat in the graph's size. A region layout is what lets the reader accept
/// or reject the whole topology before it touches the vectors, which are two
/// orders of magnitude larger.
///
/// The header is written twice. Once as zeros, so the file has room for it, and
/// once at the end with `adjacency_bytes` and `file_bytes` filled in and the
/// magic stamped. Nothing derives those two without a pass over the graph that
/// this pass already makes.
pub(crate) fn write_dump<T, D>(
    hnsw: &Hnsw<'_, T, D>,
    kind: GraphKind,
    dir: &Path,
) -> Result<(), String>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    let indexation = hnsw.get_point_indexation();
    let nb_point = indexation.get_nb_point();
    if nb_point == 0 {
        return Err("the graph holds no points".to_string());
    }
    let entry = indexation
        .get_entry_point_id()
        .ok_or_else(|| "the graph holds points and no entry point".to_string())?;

    let nb_layer = NB_LAYER_MAX as usize;
    let layer_counts: Vec<usize> = (0..nb_layer)
        .map(|layer| indexation.get_layer_nb_point(layer))
        .collect();
    let counted: usize = layer_counts.iter().sum();
    if counted != nb_point {
        return Err(format!(
            "the graph reports {} points and its layers hold {}",
            nb_point, counted
        ));
    }

    let dimension = indexation.get_data_dimension();
    if dimension == 0 {
        return Err("the graph holds points of no width".to_string());
    }

    let path = dir.join(DUMP_FILENAME);
    let file =
        File::create(&path).map_err(|e| format!("the graph dump could not be created: {}", e))?;
    let mut out = HashingWriter::new(BufWriter::with_capacity(IO_BUFFER_BYTES, file));

    // The header's own bytes are not part of the payload checksum, so they are
    // written straight through rather than through the hashing writer.
    out.inner
        .write_all(&[0u8; HEADER_BYTES])
        .map_err(|e| format!("the graph dump could not be written: {}", e))?;

    for count in &layer_counts {
        let count = u32::try_from(*count)
            .map_err(|_| format!("a layer holds {} points and the table is a u32", count))?;
        out.put_u32(count)?;
    }

    for layer in 0..nb_layer {
        for point in indexation.get_layer_iterator(layer) {
            out.put_u64(point.get_origin_id() as u64)?;
        }
    }

    let adjacency_start = out.written;
    for layer in 0..nb_layer {
        for point in indexation.get_layer_iterator(layer) {
            let neighbourhood = point.get_neighborhood_id();
            // Trailing empty layers are trimmed rather than written. A point
            // drawn at level zero has one non-empty list and fifteen empty
            // ones, and it is the common case, so writing sixteen counts for
            // every point spent 64 bytes each where one does.
            let list_count = neighbourhood
                .iter()
                .rposition(|list| !list.is_empty())
                .map_or(0, |last| last + 1);
            out.put_u8(
                u8::try_from(list_count)
                    .map_err(|_| format!("a point carries adjacency for {} layers", list_count))?,
            )?;
            for list in &neighbourhood[..list_count] {
                out.put_u32(u32::try_from(list.len()).map_err(|_| {
                    format!("a point carries {} neighbours at one layer", list.len())
                })?)?;
                for neighbour in list {
                    if neighbour.p_id.1 < 0 {
                        return Err(format!(
                            "a neighbour sits at rank {} of layer {}",
                            neighbour.p_id.1, neighbour.p_id.0
                        ));
                    }
                    out.put_u8(neighbour.p_id.0)?;
                    out.put_u32(neighbour.p_id.1 as u32)?;
                    out.put(&neighbour.distance.to_le_bytes())?;
                }
            }
        }
    }
    let adjacency_bytes = out.written - adjacency_start;

    let mut buffer: Vec<u8> = Vec::with_capacity(dimension * T::BYTES);
    for layer in 0..nb_layer {
        for point in indexation.get_layer_iterator(layer) {
            let values = point.get_v();
            if values.len() != dimension {
                return Err(format!(
                    "a point holds {} values where the graph holds {}",
                    values.len(),
                    dimension
                ));
            }
            buffer.clear();
            T::encode(values, &mut buffer);
            out.put(&buffer)?;
        }
    }

    let payload_checksum = out.sum.finish();
    let mut inner = out.inner;
    inner
        .write_all(&payload_checksum.to_le_bytes())
        .and_then(|()| inner.write_all(&MAGIC.to_le_bytes()))
        .map_err(|e| format!("the graph dump's trailer could not be written: {}", e))?;

    let file_bytes = HEADER_BYTES as u64 + out.written + TRAILER_BYTES as u64;
    let header = Header {
        kind,
        element: T::KIND,
        nb_layer: NB_LAYER_MAX,
        dimension: u32::try_from(dimension)
            .map_err(|_| format!("the graph holds points of {} values", dimension))?,
        m: hnsw.get_max_nb_connection_full() as u64,
        ef_construction: hnsw.get_ef_construction() as u64,
        nb_point: nb_point as u64,
        level_scale: indexation.get_level_scale(),
        entry_layer: entry.0 as u32,
        entry_rank: u32::try_from(entry.1)
            .map_err(|_| format!("the entry point sits at rank {}", entry.1))?,
        adjacency_bytes,
        file_bytes,
    };

    // The body is on disk before the header names it, and the magic arrives
    // with the header. A save interrupted before this point leaves zeros at
    // offset zero, which reads as no dump at all rather than as a short one.
    let mut file = inner
        .into_inner()
        .map_err(|e| format!("the graph dump could not be flushed: {}", e))?;
    file.seek(SeekFrom::Start(0))
        .and_then(|_| file.write_all(&header.encode()))
        .and_then(|()| file.flush())
        .map_err(|e| format!("the graph dump's header could not be written: {}", e))?;

    // Only once the new dump is whole. A save over a directory written by an
    // earlier release otherwise leaves its two files beside this one for ever,
    // and at 50,000 raw records that is 340 MB of a graph nothing will read.
    // Removing them is best effort: a directory that is readable but not
    // writable in this respect is not a reason to fail a save that succeeded.
    for legacy in LEGACY_DUMP_FILENAMES {
        let path = dir.join(legacy);
        if path.exists() {
            match std::fs::remove_file(&path) {
                Ok(()) => info!(
                    operation = "save_hnsw_graph",
                    removed = legacy,
                    "Removed a graph dump written by an earlier release"
                ),
                Err(e) => warn!(
                    operation = "save_hnsw_graph",
                    file = legacy,
                    error = %e,
                    "Could not remove a graph dump written by an earlier release"
                ),
            }
        }
    }

    Ok(())
}

// ============================================================================
// READING
// ============================================================================

/// What the reader was told to expect, from `config.json` and the loaded index.
#[derive(Clone, Copy)]
pub(crate) struct Expected {
    pub kind: GraphKind,
    pub dimension: usize,
    pub m: usize,
    pub ef_construction: usize,
    /// The live record count. The graph holds at least this many nodes and
    /// holds more whenever a removal or an overwrite has stranded one.
    pub min_nodes: usize,
}

/// A dump parsed back into the topology a graph constructor takes, with the
/// parameters the header carried.
///
/// Two consumers turn this into a graph. [`read_dump`] hands it to the vendored
/// `Hnsw::from_loaded_points`, and the parity tests hand one parse to both that
/// constructor and the flat structure in `flat.rs`, which is what lets the two
/// be compared on a single topology rather than on two reads of one file.
pub(super) struct ParsedDump<T> {
    /// `points_by_layer[l][r]` is the point at rank `r` of layer `l`.
    pub points_by_layer: Vec<Vec<LoadedPoint<T>>>,
    /// Where the traversal starts.
    pub entry: PointId,
    /// `max_nb_connection`, which the header and `config.json` agree on by the
    /// time the parse succeeds.
    pub m: usize,
    /// As the header carried it.
    pub ef_construction: usize,
    /// As the header carried it.
    pub level_scale: f64,
    /// Points the dump holds, which a constructor's answer is checked against.
    pub nb_point: usize,
}

/// Read the graph back out of `dir`.
///
/// Every failure is an error and none is a panic, an exit or an allocation from
/// a length the file has not earned. The parsing itself is [`parse_dump`], and
/// this wraps it in the one construction call ZeusDB ships.
pub(crate) fn read_dump<T, D>(
    dir: &Path,
    expected: &Expected,
    dist: D,
) -> Result<Hnsw<'static, T, D>, String>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    let parsed = parse_dump::<T>(dir, expected)?;
    let nb_point = parsed.nb_point;
    let hnsw = Hnsw::from_loaded_points(
        parsed.points_by_layer,
        parsed.entry,
        parsed.m,
        parsed.ef_construction,
        parsed.level_scale,
        dist,
    )
    .map_err(|e| format!("the graph dump could not be rebuilt: {}", e))?;

    let restored = hnsw.get_nb_point();
    if restored != nb_point {
        return Err(format!(
            "the graph dump declares {} nodes and yielded {}",
            nb_point, restored
        ));
    }
    Ok(hnsw)
}

/// Parse a dump into the topology and parameters it carries, building nothing.
///
/// The order is deliberate. The file's real length is established first, the
/// header is checked against itself second, the header is checked against the
/// index third, and only then does anything size a buffer from a field.
pub(super) fn parse_dump<T>(dir: &Path, expected: &Expected) -> Result<ParsedDump<T>, String>
where
    T: DumpElement,
{
    let path = dir.join(DUMP_FILENAME);
    let actual_bytes = match std::fs::metadata(&path) {
        Ok(meta) => meta.len(),
        Err(_) => return Err("the directory holds no ZeusDB graph dump".to_string()),
    };
    if actual_bytes < (HEADER_BYTES + TRAILER_BYTES) as u64 {
        return Err(format!(
            "the graph dump is {} bytes and the smallest one is {}",
            actual_bytes,
            HEADER_BYTES + TRAILER_BYTES
        ));
    }

    let file =
        File::open(&path).map_err(|e| format!("the graph dump could not be opened: {}", e))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);

    let mut raw = [0u8; HEADER_BYTES];
    reader
        .read_exact(&mut raw)
        .map_err(|e| format!("the graph dump's header could not be read: {}", e))?;
    let header = Header::decode(&raw)?;

    // The header now describes itself consistently. Everything below decides
    // whether it describes *this* index, and the size agreement is what makes
    // the allocations after it safe.
    if header.nb_layer != NB_LAYER_MAX {
        return Err(format!(
            "the graph dump carries {} layers where this build uses {}",
            header.nb_layer, NB_LAYER_MAX
        ));
    }
    if header.element != T::KIND {
        return Err(format!(
            "the graph dump stores element type {} where this index holds {}",
            header.element,
            T::KIND
        ));
    }
    if header.kind != expected.kind {
        return Err(format!(
            "the graph dump was written for {} and config.json declares {}",
            header.kind.label(),
            expected.kind.label()
        ));
    }
    if header.dimension as usize != expected.dimension {
        return Err(format!(
            "the graph dump stores {} values per point where this index expects {}",
            header.dimension, expected.dimension
        ));
    }
    if header.m != expected.m as u64 {
        return Err(format!(
            "the graph dump was written at m {} and config.json declares {}",
            header.m, expected.m
        ));
    }
    if header.ef_construction != expected.ef_construction as u64 {
        return Err(format!(
            "the graph dump was written at ef_construction {} and config.json declares {}",
            header.ef_construction, expected.ef_construction
        ));
    }
    if header.nb_point < expected.min_nodes as u64 {
        return Err(format!(
            "the graph dump holds {} nodes and the index holds {} records",
            header.nb_point, expected.min_nodes
        ));
    }
    if header.nb_point == 0 {
        return Err("the graph dump holds no nodes".to_string());
    }
    if !header.level_scale.is_finite() || header.level_scale <= 0. {
        return Err(format!(
            "the graph dump declares a level scale of {}",
            header.level_scale
        ));
    }

    let nb_point = usize::try_from(header.nb_point)
        .map_err(|_| format!("the graph dump declares {} nodes", header.nb_point))?;
    let dimension = header.dimension as usize;
    let nb_layer = header.nb_layer as usize;
    let adjacency_bytes = usize::try_from(header.adjacency_bytes)
        .map_err(|_| "the graph dump declares an adjacency region too large to read".to_string())?;

    // The length the header's own fields imply, against the length the file
    // actually has. This is the check that makes a header claiming a billion
    // points harmless: the arithmetic below either overflows or names a size
    // the file does not have, and either way nothing is allocated.
    let implied = (|| {
        let vectors = nb_point.checked_mul(dimension)?.checked_mul(T::BYTES)?;
        HEADER_BYTES
            .checked_add(nb_layer.checked_mul(LAYER_TABLE_ENTRY_BYTES)?)?
            .checked_add(nb_point.checked_mul(ORIGIN_ID_BYTES)?)?
            .checked_add(adjacency_bytes)?
            .checked_add(vectors)?
            .checked_add(TRAILER_BYTES)
    })()
    .ok_or_else(|| {
        format!(
            "the graph dump declares {} nodes of {} values, which no file could hold",
            nb_point, dimension
        )
    })?;
    if header.file_bytes != implied as u64 {
        return Err(format!(
            "the graph dump's header declares {} bytes and its own fields need {}",
            header.file_bytes, implied
        ));
    }
    if header.file_bytes != actual_bytes {
        return Err(format!(
            "the graph dump declares {} bytes and the file holds {}",
            header.file_bytes, actual_bytes
        ));
    }

    // Every allocation from here on is bounded by a file that really is this
    // long, so the header can no longer ask for memory it has not accounted for.
    let mut hashed = HashingReader::new(reader);

    let mut layer_counts = Vec::with_capacity(nb_layer);
    let mut table = vec![0u8; nb_layer * LAYER_TABLE_ENTRY_BYTES];
    hashed.read_exact_hashed(&mut table)?;
    let mut counted = 0usize;
    for entry in table.chunks_exact(LAYER_TABLE_ENTRY_BYTES) {
        let count = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
        counted = counted
            .checked_add(count)
            .ok_or_else(|| "the graph dump's layer table does not add up".to_string())?;
        layer_counts.push(count);
    }
    if counted != nb_point {
        return Err(format!(
            "the graph dump declares {} nodes and its layers hold {}",
            nb_point, counted
        ));
    }
    let entry_layer = header.entry_layer as usize;
    if entry_layer >= nb_layer || header.entry_rank as usize >= layer_counts[entry_layer] {
        return Err(format!(
            "the graph dump's entry point sits at layer {} rank {} and no node is there",
            header.entry_layer, header.entry_rank
        ));
    }

    let mut origin_raw = vec![0u8; nb_point * ORIGIN_ID_BYTES];
    hashed.read_exact_hashed(&mut origin_raw)?;
    let mut origin_ids = Vec::with_capacity(nb_point);
    for entry in origin_raw.chunks_exact(ORIGIN_ID_BYTES) {
        let mut word = [0u8; 8];
        word.copy_from_slice(entry);
        let id = u64::from_le_bytes(word);
        origin_ids.push(usize::try_from(id).map_err(|_| {
            format!(
                "the graph dump names origin id {}, which this target cannot hold",
                id
            )
        })?);
    }
    drop(origin_raw);

    let adjacency = read_adjacency(&mut hashed, adjacency_bytes, nb_point, &layer_counts)?;

    // The points, layer by layer, taking each one's vector as it arrives and
    // the adjacency already parsed above.
    let mut points_by_layer: Vec<Vec<LoadedPoint<T>>> = Vec::with_capacity(nb_layer);
    let mut values_raw = vec![0u8; dimension * T::BYTES];
    let mut adjacency = adjacency.into_iter();
    let mut origin_ids = origin_ids.into_iter();
    for count in &layer_counts {
        let mut layer = Vec::with_capacity(*count);
        for _ in 0..*count {
            hashed.read_exact_hashed(&mut values_raw)?;
            layer.push(LoadedPoint {
                origin_id: origin_ids
                    .next()
                    .ok_or_else(|| "the graph dump ran out of origin ids".to_string())?,
                data: T::decode(&values_raw),
                neighbours: adjacency
                    .next()
                    .ok_or_else(|| "the graph dump ran out of adjacency".to_string())?,
            });
        }
        points_by_layer.push(layer);
    }

    let mut trailer = [0u8; TRAILER_BYTES];
    hashed
        .inner
        .read_exact(&mut trailer)
        .map_err(|e| format!("the graph dump's trailer could not be read: {}", e))?;
    let stored = u64::from_le_bytes(take8(&trailer, 0));
    let end_magic = u64::from_le_bytes(take8(&trailer, 8));
    if end_magic != MAGIC {
        return Err("the graph dump does not end where it says it does".to_string());
    }
    let computed = hashed.sum.finish();
    if stored != computed {
        return Err("the graph dump's contents are corrupt".to_string());
    }

    Ok(ParsedDump {
        points_by_layer,
        entry: PointId(header.entry_layer as u8, header.entry_rank as i32),
        m: expected.m,
        ef_construction: header.ef_construction as usize,
        level_scale: header.level_scale,
        nb_point,
    })
}

/// Parse the adjacency region into one entry per point, in (layer, rank) order.
///
/// Every count is checked against the bytes the region has left before it sizes
/// anything, so a count of four billion is rejected on arithmetic rather than
/// on an allocation failure. Every target is checked against the layer table,
/// so an edge naming a node that is not there is caught here rather than inside
/// the graph constructor.
fn read_adjacency<R: Read>(
    reader: &mut HashingReader<R>,
    region_bytes: usize,
    nb_point: usize,
    layer_counts: &[usize],
) -> Result<Vec<Vec<Vec<LoadedEdge>>>, String> {
    let mut region = vec![0u8; region_bytes];
    reader.read_exact_hashed(&mut region)?;

    let mut at = 0usize;
    let mut out: Vec<Vec<Vec<LoadedEdge>>> = Vec::with_capacity(nb_point);
    for point in 0..nb_point {
        if at >= region.len() {
            return Err(format!(
                "the graph dump's adjacency ends at node {} of {}",
                point, nb_point
            ));
        }
        let list_count = region[at] as usize;
        at += 1;
        if list_count > NB_LAYER_MAX as usize {
            return Err(format!(
                "node {} carries adjacency for {} layers and a node carries at most {}",
                point, list_count, NB_LAYER_MAX
            ));
        }
        let mut lists = Vec::with_capacity(list_count);
        for layer in 0..list_count {
            if region.len() - at < 4 {
                return Err(format!(
                    "the graph dump's adjacency ends inside node {} at layer {}",
                    point, layer
                ));
            }
            let edge_count =
                u32::from_le_bytes([region[at], region[at + 1], region[at + 2], region[at + 3]])
                    as usize;
            at += 4;
            let needed = edge_count.checked_mul(EDGE_BYTES).ok_or_else(|| {
                format!(
                    "node {} declares {} neighbours at layer {}",
                    point, edge_count, layer
                )
            })?;
            if needed > region.len() - at {
                return Err(format!(
                    "node {} declares {} neighbours at layer {} and the dump has room for {}",
                    point,
                    edge_count,
                    layer,
                    (region.len() - at) / EDGE_BYTES
                ));
            }
            let mut edges = Vec::with_capacity(edge_count);
            for _ in 0..edge_count {
                let target_layer = region[at] as usize;
                let target_rank = u32::from_le_bytes([
                    region[at + 1],
                    region[at + 2],
                    region[at + 3],
                    region[at + 4],
                ]) as usize;
                let distance = f32::from_le_bytes([
                    region[at + 5],
                    region[at + 6],
                    region[at + 7],
                    region[at + 8],
                ]);
                at += EDGE_BYTES;
                if target_layer >= layer_counts.len() || target_rank >= layer_counts[target_layer] {
                    return Err(format!(
                        "node {} names layer {} rank {} and no node is there",
                        point, target_layer, target_rank
                    ));
                }
                if !distance.is_finite() {
                    return Err(format!(
                        "node {} carries a distance of {} at layer {}",
                        point, distance, layer
                    ));
                }
                if target_rank > i32::MAX as usize {
                    return Err(format!(
                        "node {} names rank {}, which is not an i32",
                        point, target_rank
                    ));
                }
                edges.push(LoadedEdge {
                    target: PointId(target_layer as u8, target_rank as i32),
                    distance,
                });
            }
            lists.push(edges);
        }
        out.push(lists);
    }
    if at != region.len() {
        return Err(format!(
            "the graph dump's adjacency holds {} bytes beyond its {} nodes",
            region.len() - at,
            nb_point
        ));
    }
    Ok(out)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::CosineDist;
    use std::collections::BTreeMap;

    /// Build a small graph the tests can round trip.
    fn sample_graph(records: usize, dim: usize, m: usize) -> Hnsw<'static, f32, CosineDist> {
        let hnsw: Hnsw<'static, f32, CosineDist> =
            Hnsw::new(m, records.max(1), NB_LAYER_MAX as usize, 64, CosineDist {});
        // A cheap deterministic spread. The graph's shape does not matter here,
        // only that it has one and that the same call makes the same one.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for id in 0..records {
            let vector: Vec<f32> = (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / 16_777_216.0 - 0.5
                })
                .collect();
            hnsw.insert((&vector, id));
        }
        hnsw
    }

    /// The whole graph as plain values, for comparing one against another.
    ///
    /// Keyed by origin id rather than by position, because two graphs that hold
    /// the same points in the same layers still have to agree on which point is
    /// where, and a position keyed comparison would pass that by. The
    /// neighbours are the targets' origin ids and the distance's exact bits, in
    /// list order.
    #[allow(clippy::type_complexity)]
    fn topology<D: Distance<f32> + Send + Sync>(
        hnsw: &Hnsw<'_, f32, D>,
    ) -> BTreeMap<usize, (u8, i32, Vec<u32>, Vec<Vec<(usize, u32)>>)> {
        let indexation = hnsw.get_point_indexation();
        let mut out = BTreeMap::new();
        for layer in 0..NB_LAYER_MAX as usize {
            for point in indexation.get_layer_iterator(layer) {
                let p_id = point.get_point_id();
                let values = point.get_v().iter().map(|v| v.to_bits()).collect();
                let adjacency = point
                    .get_neighborhood_id()
                    .iter()
                    .map(|list| {
                        list.iter()
                            .map(|n| (n.d_id, n.distance.to_bits()))
                            .collect()
                    })
                    .collect();
                out.insert(point.get_origin_id(), (p_id.0, p_id.1, values, adjacency));
            }
        }
        out
    }

    /// The reason a read was refused. `Hnsw` is not `Debug`, so `unwrap_err`
    /// and `expect_err` are both out of reach.
    fn refused<T, D>(result: Result<Hnsw<'static, T, D>, String>) -> String
    where
        T: DumpElement,
        D: Distance<T> + Send + Sync,
    {
        match result {
            Ok(_) => panic!("a dump that should have been refused was accepted"),
            Err(reason) => reason,
        }
    }

    fn expected_for(hnsw: &Hnsw<'_, f32, CosineDist>, dim: usize, nodes: usize) -> Expected {
        Expected {
            kind: GraphKind::Cosine,
            dimension: dim,
            m: hnsw.get_max_nb_connection_full(),
            ef_construction: hnsw.get_ef_construction(),
            min_nodes: nodes,
        }
    }

    #[test]
    fn a_round_trip_reproduces_every_node_and_every_edge() {
        let dir = tempfile::tempdir().unwrap();
        let built = sample_graph(600, 12, 16);
        write_dump(&built, GraphKind::Cosine, dir.path()).unwrap();
        let expected = expected_for(&built, 12, 600);
        let read: Hnsw<'static, f32, CosineDist> =
            read_dump(dir.path(), &expected, CosineDist {}).unwrap();

        assert_eq!(read.get_nb_point(), built.get_nb_point());
        assert_eq!(read.get_max_nb_connection_full(), 16);
        assert_eq!(read.get_ef_construction(), built.get_ef_construction());
        assert_eq!(
            read.get_point_indexation().get_level_scale(),
            built.get_point_indexation().get_level_scale()
        );
        assert_eq!(
            read.get_point_indexation().get_entry_point_id(),
            built.get_point_indexation().get_entry_point_id()
        );

        let before = topology(&built);
        let after = topology(&read);
        assert_eq!(before.len(), after.len());
        let mut differing_nodes = 0;
        let mut differing_edges = 0;
        for (id, left) in &before {
            let right = after.get(id).expect("a node went missing");
            if left.0 != right.0 || left.1 != right.1 || left.2 != right.2 {
                differing_nodes += 1;
            }
            for (l, list) in left.3.iter().enumerate() {
                if list != &right.3[l] {
                    differing_edges += 1;
                }
            }
        }
        assert_eq!(differing_nodes, 0);
        assert_eq!(differing_edges, 0);
    }

    #[test]
    fn writing_the_same_graph_twice_gives_the_same_bytes() {
        let one = tempfile::tempdir().unwrap();
        let two = tempfile::tempdir().unwrap();
        let built = sample_graph(300, 8, 16);
        write_dump(&built, GraphKind::Cosine, one.path()).unwrap();
        let expected = expected_for(&built, 8, 300);
        let read: Hnsw<'static, f32, CosineDist> =
            read_dump(one.path(), &expected, CosineDist {}).unwrap();
        write_dump(&read, GraphKind::Cosine, two.path()).unwrap();
        assert_eq!(
            std::fs::read(one.path().join(DUMP_FILENAME)).unwrap(),
            std::fs::read(two.path().join(DUMP_FILENAME)).unwrap()
        );
    }

    #[test]
    fn m_at_the_top_of_the_range_survives() {
        // 256 is what `Hnsw::new` admits and what the vendored header, a u8,
        // recorded as 0.
        for m in [2usize, 16, 255, 256] {
            let dir = tempfile::tempdir().unwrap();
            let built = sample_graph(120, 6, m);
            assert_eq!(built.get_max_nb_connection_full(), m);
            write_dump(&built, GraphKind::Cosine, dir.path()).unwrap();
            let expected = expected_for(&built, 6, 120);
            let read: Hnsw<'static, f32, CosineDist> =
                read_dump(dir.path(), &expected, CosineDist {}).unwrap();
            assert_eq!(read.get_max_nb_connection_full(), m);
            assert_eq!(topology(&built), topology(&read));
        }
    }

    /// Round trip one damaged file and report what the reader said.
    fn damaged(mutate: impl FnOnce(Vec<u8>) -> Vec<u8>) -> String {
        let dir = tempfile::tempdir().unwrap();
        let built = sample_graph(200, 6, 16);
        write_dump(&built, GraphKind::Cosine, dir.path()).unwrap();
        let path = dir.path().join(DUMP_FILENAME);
        let blob = std::fs::read(&path).unwrap();
        std::fs::write(&path, mutate(blob)).unwrap();
        let expected = expected_for(&built, 6, 200);
        refused(read_dump::<f32, CosineDist>(
            dir.path(),
            &expected,
            CosineDist {},
        ))
    }

    #[test]
    fn every_damaged_input_is_an_error() {
        assert!(damaged(|_| Vec::new()).contains("smallest"));
        assert!(damaged(|b| b[..96].to_vec()).contains("smallest"));
        // Long enough to clear the floor, so the header parses and the size it
        // implies is what refuses it.
        assert!(damaged(|b| b[..112].to_vec()).contains("the file holds"));
        assert!(damaged(|b| b[..b.len() / 2].to_vec()).contains("the file holds"));
        assert!(damaged(|b| b[..b.len() - 1].to_vec()).contains("the file holds"));
        assert!(damaged(|mut b| {
            b.extend_from_slice(&[0u8; 16]);
            b
        })
        .contains("the file holds"));
        assert!(damaged(|mut b| {
            b[0] ^= 0xff;
            b
        })
        .contains("not a ZeusDB graph dump"));
        assert!(damaged(|mut b| {
            b[8..12].copy_from_slice(&99u32.to_le_bytes());
            b
        })
        .contains("format version"));
        assert!(damaged(|mut b| {
            b[40] ^= 0x01;
            b
        })
        .contains("header is corrupt"));
        // A flipped bit in the vector region, where every length still agrees.
        assert!(damaged(|mut b| {
            let at = b.len() - 32;
            b[at] ^= 0x10;
            b
        })
        .contains("contents are corrupt"));
        assert!(damaged(|mut b| {
            let at = b.len() - 8;
            b[at..].copy_from_slice(&0u64.to_le_bytes());
            b
        })
        .contains("does not end where it says"));
    }

    #[test]
    fn an_absent_file_is_an_error_rather_than_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let built = sample_graph(50, 4, 16);
        let expected = expected_for(&built, 4, 50);
        let reason = refused(read_dump::<f32, CosineDist>(
            dir.path(),
            &expected,
            CosineDist {},
        ));
        assert!(reason.contains("holds no ZeusDB graph dump"), "{}", reason);
    }

    #[test]
    fn a_save_clears_the_files_an_earlier_release_left() {
        let dir = tempfile::tempdir().unwrap();
        for legacy in LEGACY_DUMP_FILENAMES {
            std::fs::write(dir.path().join(legacy), vec![7u8; 4096]).unwrap();
        }
        let built = sample_graph(100, 5, 16);
        write_dump(&built, GraphKind::Cosine, dir.path()).unwrap();

        let left: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(left, vec![DUMP_FILENAME.to_string()], "{:?}", left);

        // And a save into a directory that never held them is unbothered.
        let clean = tempfile::tempdir().unwrap();
        write_dump(&built, GraphKind::Cosine, clean.path()).unwrap();
        assert!(clean.path().join(DUMP_FILENAME).exists());
    }

    #[test]
    fn a_dump_in_the_vendored_format_is_not_recognised() {
        // MAGICDESCR_4, which is what a 0.6.0 dump opens with. There is no
        // reader for it and there is deliberately not going to be one, so the
        // only thing that has to hold is that it is rejected cleanly.
        let dir = tempfile::tempdir().unwrap();
        let built = sample_graph(50, 4, 16);
        let mut blob = 0x002a_6779u32.to_le_bytes().to_vec();
        blob.extend_from_slice(&[0u8; 4096]);
        std::fs::write(dir.path().join(DUMP_FILENAME), &blob).unwrap();
        let expected = expected_for(&built, 4, 50);
        let reason = refused(read_dump::<f32, CosineDist>(
            dir.path(),
            &expected,
            CosineDist {},
        ));
        assert!(reason.contains("not a ZeusDB graph dump"), "{}", reason);
    }

    /// A header this build wrote, carrying a node count nothing could hold.
    ///
    /// The Python damage tests cannot reach this, because rewriting the field
    /// breaks the header checksum and the reader stops there. Only the writer
    /// can produce a header that is internally consistent, so the check that
    /// the reader refuses to allocate from it is made here.
    #[test]
    fn a_header_claiming_a_billion_points_allocates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let built = sample_graph(80, 4, 16);
        write_dump(&built, GraphKind::Cosine, dir.path()).unwrap();
        let path = dir.path().join(DUMP_FILENAME);
        let blob = std::fs::read(&path).unwrap();

        let mut header = Header::decode(&{
            let mut raw = [0u8; HEADER_BYTES];
            raw.copy_from_slice(&blob[..HEADER_BYTES]);
            raw
        })
        .unwrap();
        header.nb_point = 1_000_000_000;
        let mut damaged = header.encode().to_vec();
        damaged.extend_from_slice(&blob[HEADER_BYTES..]);
        std::fs::write(&path, &damaged).unwrap();

        let expected = expected_for(&built, 4, 80);
        let reason = refused(read_dump::<f32, CosineDist>(
            dir.path(),
            &expected,
            CosineDist {},
        ));
        // Rejected on the size the fields imply, before the layer table is even
        // read, so no allocation is ever sized from the count.
        assert!(reason.contains("its own fields need"), "{}", reason);

        // And the same again with a count large enough that the arithmetic
        // itself overflows rather than merely disagreeing.
        header.nb_point = u64::MAX / 2;
        let mut damaged = header.encode().to_vec();
        damaged.extend_from_slice(&blob[HEADER_BYTES..]);
        std::fs::write(&path, &damaged).unwrap();
        let reason = refused(read_dump::<f32, CosineDist>(
            dir.path(),
            &expected,
            CosineDist {},
        ));
        assert!(reason.contains("no file could hold"), "{}", reason);
    }

    #[test]
    fn a_dump_written_for_another_configuration_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let built = sample_graph(120, 6, 16);
        write_dump(&built, GraphKind::Cosine, dir.path()).unwrap();

        let refuse = |expected: &Expected| {
            refused(read_dump::<f32, CosineDist>(
                dir.path(),
                expected,
                CosineDist {},
            ))
        };

        let base = expected_for(&built, 6, 120);
        assert!(refuse(&Expected {
            kind: GraphKind::L2,
            ..base
        })
        .contains("was written for cosine"));
        assert!(refuse(&Expected {
            dimension: 7,
            ..base
        })
        .contains("values per point"));
        assert!(refuse(&Expected { m: 17, ..base }).contains("written at m 16"));
        assert!(refuse(&Expected {
            ef_construction: 65,
            ..base
        })
        .contains("ef_construction"));
        // More live records than the dump holds nodes, which is the shape of a
        // directory whose graph belongs to an earlier, smaller save.
        assert!(refuse(&Expected {
            min_nodes: 121,
            ..base
        })
        .contains("and the index holds"));

        // The element type is checked as well, so a raw dump handed to the
        // quantized reader is refused rather than read at a quarter the width.
        assert!(
            refused(read_dump::<u8, ByteDist>(dir.path(), &base, ByteDist {}))
                .contains("element type")
        );
    }

    /// A `u8` distance, only so the element type check has something to refuse.
    struct ByteDist;

    impl Distance<u8> for ByteDist {
        fn eval(&self, _va: &[u8], _vb: &[u8]) -> f32 {
            0.0
        }
    }

    #[test]
    fn the_checksum_does_not_depend_on_how_the_bytes_were_split() {
        let payload: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let whole = checksum_of(&payload);
        for split in [1usize, 3, 7, 8, 9, 64, 511] {
            let mut sum = Checksum::new();
            for chunk in payload.chunks(split) {
                sum.write(chunk);
            }
            assert_eq!(sum.finish(), whole, "split {}", split);
        }
        // Appended zeros change the answer, which a length blind fold would not
        // catch, and a single flipped bit changes it too.
        let mut longer = payload.clone();
        longer.push(0);
        assert_ne!(checksum_of(&longer), whole);
        let mut flipped = payload.clone();
        flipped[500] ^= 0x01;
        assert_ne!(checksum_of(&flipped), whole);
    }

    #[test]
    fn the_header_is_ninety_six_bytes_and_checksums_itself() {
        let header = Header {
            kind: GraphKind::L1Pq,
            element: 2,
            nb_layer: NB_LAYER_MAX,
            dimension: 96,
            m: 256,
            ef_construction: 200,
            nb_point: 12345,
            level_scale: 0.25,
            entry_layer: 3,
            entry_rank: 7,
            adjacency_bytes: 999,
            file_bytes: 4242,
        };
        let raw = header.encode();
        assert_eq!(raw.len(), HEADER_BYTES);
        let back = Header::decode(&raw).unwrap();
        assert_eq!(back.kind, GraphKind::L1Pq);
        assert_eq!(back.m, 256);
        assert_eq!(back.nb_point, 12345);
        assert_eq!(back.level_scale, 0.25);
        assert_eq!(back.entry_rank, 7);
        assert_eq!(back.adjacency_bytes, 999);
        assert_eq!(back.file_bytes, 4242);

        // Every byte the checksum covers is actually covered by it.
        for at in 0..HEADER_CHECKSUM_BYTES {
            let mut broken = raw;
            broken[at] ^= 0x01;
            assert!(
                Header::decode(&broken).is_err(),
                "byte {} passed unchecked",
                at
            );
        }
    }
}
