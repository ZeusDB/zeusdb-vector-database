//! ZeusDB's own on-disk format for the graph.
//!
//! One file, `hnsw_index.zdbgraph`, holding the graph's topology and every
//! node's stored row. A directory at format 5.x holds version 2 of the
//! format, which this module writes and reads. A directory saved at an
//! earlier format holds version 1, which [`v1`] reads. The directory's format
//! decides which version its dump must be, through [`Expected::version`], and
//! each reader refuses the other version.
//!
//! # Byte order and widths
//!
//! Little-endian throughout, and every width fixed, so a dump is the same
//! bytes on every target ZeusDB ships for.
//!
//! # Layout of version 2
//!
//! ```text
//! Header, 128 bytes
//!    0   8  magic              u64   b"ZDBGRAPH"
//!    8   4  format_version     u32   2
//!   12   4  flags              u32   zero
//!   16   1  kind               u8    GraphKind
//!   17   1  element            u8    1 f32, 2 u8, 3 i8
//!   18   1  layers             u8    16
//!   19   1  reserved           u8    zero
//!   20   4  dimension          u32   values in one row
//!   24   4  m                  u32
//!   28   4  ef_construction    u32
//!   32   4  nodes              u32
//!   36   4  entry node         u32   a node index
//!   40   8  level_scale        f64
//!   48   8  identity bytes     u64
//!   56   8  rows bytes         u64
//!   64   8  topology bytes     u64
//!   72   8  file bytes         u64   the whole file, trailer included
//!   80   8  identity checksum  u64
//!   88   8  rows checksum      u64
//!   96   8  topology checksum  u64
//!  104  16  reserved           zero
//!  120   8  header checksum    u64   over bytes 0 to 120
//!
//! Identity   16 u32 layer counts, then one u32 origin id a node
//! Rows       one row a node, dimension values of the element's width
//! Topology   for each node, a u8 list count of at most 16, then for each
//!            list a u16 entry count and that many u32 node indices
//!
//! Trailer, 16 bytes
//!    0   8  file bytes         u64   equal to the header's
//!    8   8  magic              u64
//! ```
//!
//! Nodes are in layer then rank order: every node whose top level is layer 0
//! in the order the graph holds them, then layer 1, and so on to layer 15. A
//! node's index is its place in that order, and a list names each target by
//! its index. A node's lists run from layer 0 to the highest layer it holds an
//! entry at, an empty list below that kept and the empty lists above it left
//! out. A list is written in the order the graph holds it, which is the order
//! of each target's distance from the list's owner, and no distance is
//! stored. A list a node holds above its own level is written like any other,
//! since the graph counts its entries into the inbound counters the overflow
//! guard reads.
//!
//! # What each checksum protects
//!
//! The header checksum covers every header field, the three region lengths
//! and the three region checksums among them, so the reader trusts the
//! lengths before it sizes anything from them. Each region carries its own
//! checksum, so damage in the topology leaves the identity and the rows
//! verified. The reader checks the header, then the identity, then the rows,
//! then the trailer, then the topology. A failure in any of the first four
//! refuses the dump. A failure in the topology hands back the verified rows,
//! and the caller rebuilds the graph from them.
//!
//! # The magic is written last
//!
//! The writer streams the three regions, then the trailer, and then the header
//! with the magic over the zeros it left at offset zero, and syncs the file. A
//! save that stops part way leaves zeros where the magic goes, which no reader
//! takes for a dump.

use super::mutable::{MutableGraph, NotBuilt};
use super::store::VectorStore;
use super::traverse::LAYERS;
use super::Distance;
use crate::checksum::{checksum_of, Checksum};
use crate::storage::Dir;
use std::io::{BufReader, BufWriter, Read, Write};
use tracing::{info, warn};

pub(crate) mod v1;

/// The target the two records this module emits carry. See `LOG_TARGET` in
/// the parent module for why it is not `module_path!()`.
const LOG_TARGET: &str = "zeusdb_vector_database::graph::dump";

/// Layers a graph can hold, which is part of the on-disk contract.
///
/// Both versions record it and refuse a dump declaring any other value, so it
/// is a number the format fixes rather than a tuning knob. Every per node
/// array in the structure is sized by it and `super::traverse` walks down from
/// it.
pub const NB_LAYER_MAX: u8 = 16;

/// Where a point sits, being its layer and its rank within that layer.
///
/// The order both versions file nodes in. A version 1 list names its target
/// this way, and a version 2 list names it by its index in this order.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct PointId(pub u8, pub i32);

/// One adjacency entry as a version 1 dump holds it: the target by its layer
/// and rank, and the distance from the list's owner to the target.
///
/// A parse of a version 2 dump fills `distance` with zero for every entry,
/// since that version stores none, so a list ordered by it keeps the order it
/// was written in.
#[derive(Clone, Copy, Debug)]
pub(super) struct LoadedEdge {
    /// where the target sits, being its layer and its rank within that layer
    pub target: PointId,
    /// distance from the point holding this entry to the target
    pub distance: f32,
}

/// One point as a parse decodes it, with its row, which is the shape the
/// tests build a graph from.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(super) struct LoadedPoint<T> {
    /// the id the client inserted this point under
    pub origin_id: usize,
    /// the point's own copy of its row
    pub data: Vec<T>,
    /// adjacency by layer, lowest layer first. Fewer entries than
    /// [`NB_LAYER_MAX`] is allowed and the layers beyond are taken as empty.
    pub neighbours: Vec<Vec<LoadedEdge>>,
}

/// The one file a dump writes, and the one the loader reads.
pub const DUMP_FILENAME: &str = "hnsw_index.zdbgraph";

/// The two files the vendored graph crate wrote as its dump, which a
/// directory saved by 0.6.0 or earlier holds and no reader reads.
///
/// A save removes them where it finds them beside the dump it writes. The
/// completeness check in `persistence` reads the same two names.
pub const LEGACY_DUMP_FILENAMES: [&str; 2] = ["hnsw_index.hnsw.graph", "hnsw_index.hnsw.data"];

/// `b"ZDBGRAPH"` read little-endian, which both versions open and close with.
const MAGIC: u64 = u64::from_le_bytes(*b"ZDBGRAPH");

/// The version this module writes and [`read_dump_at`] reads.
const FORMAT_VERSION: u32 = 2;

/// Bytes the header occupies, header checksum included.
const HEADER_BYTES: usize = 128;

/// Bytes the header checksum is taken over, being everything before it.
const HEADER_CHECKSUM_BYTES: usize = 120;

/// Bytes the trailer occupies, being the file's length and the magic.
const TRAILER_BYTES: usize = 16;

/// Bytes the layer table occupies at the head of the identity region.
const LAYER_TABLE_BYTES: usize = NB_LAYER_MAX as usize * 4;

/// Bytes one node's origin id occupies.
const ORIGIN_ID_BYTES: usize = 4;

/// Bytes a list's entry count occupies.
const ENTRY_COUNT_BYTES: usize = 2;

/// Bytes one list entry occupies, being a node index.
const TARGET_BYTES: usize = 4;

/// Bytes buffered on the way to and from the file.
const IO_BUFFER_BYTES: usize = 1 << 20;

/// Which graph a dump holds.
///
/// A discriminant ZeusDB chose, so the distance types are free to move
/// between modules or be renamed without a saved index becoming unreadable.
/// The three product quantized kinds share one distance type and the four
/// scalar kinds share another, so the kind is what tells their metrics apart,
/// and a dump and a `config.json` that disagree are refused.
///
/// **The numbers are on disk. Never reuse one and never change one.**
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GraphKind {
    Cosine = 1,
    L2 = 2,
    L1 = 3,
    CosinePq = 4,
    L2Pq = 5,
    /// Quantized l1. Refused at load, so no directory this build writes
    /// carries one. The number stays reserved.
    /// `validate_space_supports_quantization` in `collection/construct.rs`
    /// records the measurement that keeps the pair refused.
    L1Pq = 6,
    /// Inner product. There is no product quantized counterpart. `create()`
    /// and `load()` refuse the pair, so no dump can carry one, and
    /// `validate_space_supports_quantization` in `collection/construct.rs`
    /// records the measurement that keeps it refused.
    Dot = 7,
    /// The four scalar quantized graphs, holding one `i8` a value and
    /// scoring on the space's own arithmetic over the decoded values.
    CosineInt8 = 8,
    L2Int8 = 9,
    L1Int8 = 10,
    DotInt8 = 11,
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
            GraphKind::Dot => "dot",
            GraphKind::CosineInt8 => "int8 cosine",
            GraphKind::L2Int8 => "int8 l2",
            GraphKind::L1Int8 => "int8 l1",
            GraphKind::DotInt8 => "int8 dot",
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
            7 => Some(GraphKind::Dot),
            8 => Some(GraphKind::CosineInt8),
            9 => Some(GraphKind::L2Int8),
            10 => Some(GraphKind::L1Int8),
            11 => Some(GraphKind::DotInt8),
            _ => None,
        }
    }
}

/// What a node's row is made of, on disk and in memory.
///
/// A raw graph stores `f32` values, a product quantized one `u8` codes and a
/// scalar quantized one `i8` codes. The `KIND` byte is written into the
/// header and checked on the way back, so a dump of one element type handed
/// to a reader of another is refused on the header rather than misread.
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
    /// The first value of `values` that is a NaN or an infinity, for the
    /// element types that can hold one.
    fn first_non_finite(_values: &[Self]) -> Option<f32> {
        None
    }
    /// Every node's row, end to end, as the rows a caller rebuilds from.
    fn into_row_values(values: Vec<Self>) -> super::RowValues;
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

    fn first_non_finite(values: &[Self]) -> Option<f32> {
        values.iter().copied().find(|value| !value.is_finite())
    }

    fn into_row_values(values: Vec<Self>) -> super::RowValues {
        super::RowValues::Raw(values)
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

    fn into_row_values(values: Vec<Self>) -> super::RowValues {
        super::RowValues::Codes(values)
    }
}

/// A scalar code, one signed byte on disk as in memory. The element kind is
/// what tells it from a product code of the same width.
impl DumpElement for i8 {
    const KIND: u8 = 3;
    const BYTES: usize = 1;

    fn encode(values: &[Self], buf: &mut Vec<u8>) {
        buf.extend(values.iter().map(|&value| value as u8));
    }

    fn decode(bytes: &[u8]) -> Vec<Self> {
        bytes.iter().map(|&byte| byte as i8).collect()
    }

    fn into_row_values(values: Vec<Self>) -> super::RowValues {
        super::RowValues::Int8(values)
    }
}

// ============================================================================
// HASHING STREAMS
// ============================================================================

/// A reader that checksums what it hands out.
///
/// Version 1 hashes everything after its header into one sum. Version 2 takes
/// the sum at the end of each region with [`HashingReader::close_region`],
/// which starts the next region's sum from nothing.
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
        self.read_hashed(buf)
            .map_err(|e| format!("the graph dump ended early: {}", e))
    }

    fn read_hashed(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.inner.read_exact(buf)?;
        self.sum.write(buf);
        Ok(())
    }

    /// The checksum over everything read since the last region closed.
    fn close_region(&mut self) -> u64 {
        std::mem::replace(&mut self.sum, Checksum::new()).finish()
    }
}

/// A writer that checksums what it takes and counts what it has written.
struct HashingWriter<W: Write> {
    inner: W,
    sum: Checksum,
    written: u64,
    /// Where the open region began, in bytes written.
    region_start: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        HashingWriter {
            inner,
            sum: Checksum::new(),
            written: 0,
            region_start: 0,
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

    fn put_u16(&mut self, value: u16) -> Result<(), String> {
        self.put(&value.to_le_bytes())
    }

    fn put_u32(&mut self, value: u32) -> Result<(), String> {
        self.put(&value.to_le_bytes())
    }

    #[cfg(any(test, feature = "test-support"))]
    fn put_u64(&mut self, value: u64) -> Result<(), String> {
        self.put(&value.to_le_bytes())
    }

    /// The checksum and the length of the region written since the last one
    /// closed.
    fn close_region(&mut self) -> (u64, u64) {
        let sum = std::mem::replace(&mut self.sum, Checksum::new()).finish();
        let bytes = self.written - self.region_start;
        self.region_start = self.written;
        (sum, bytes)
    }
}

// ============================================================================
// THE HEADER OF VERSION 2
// ============================================================================

/// The version 2 header, as fields rather than as bytes.
#[derive(Clone, Copy)]
struct Header {
    kind: GraphKind,
    element: u8,
    layers: u8,
    dimension: u32,
    m: u32,
    ef_construction: u32,
    nodes: u32,
    entry_node: u32,
    level_scale: f64,
    identity_bytes: u64,
    rows_bytes: u64,
    topology_bytes: u64,
    file_bytes: u64,
    identity_checksum: u64,
    rows_checksum: u64,
    topology_checksum: u64,
}

impl Header {
    /// The header as the 128 bytes that go on disk, checksum included.
    fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[0..8].copy_from_slice(&MAGIC.to_le_bytes());
        out[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        // 12..16 flags, 19 and 104..120 stay zero.
        out[16] = self.kind.code();
        out[17] = self.element;
        out[18] = self.layers;
        out[20..24].copy_from_slice(&self.dimension.to_le_bytes());
        out[24..28].copy_from_slice(&self.m.to_le_bytes());
        out[28..32].copy_from_slice(&self.ef_construction.to_le_bytes());
        out[32..36].copy_from_slice(&self.nodes.to_le_bytes());
        out[36..40].copy_from_slice(&self.entry_node.to_le_bytes());
        out[40..48].copy_from_slice(&self.level_scale.to_le_bytes());
        out[48..56].copy_from_slice(&self.identity_bytes.to_le_bytes());
        out[56..64].copy_from_slice(&self.rows_bytes.to_le_bytes());
        out[64..72].copy_from_slice(&self.topology_bytes.to_le_bytes());
        out[72..80].copy_from_slice(&self.file_bytes.to_le_bytes());
        out[80..88].copy_from_slice(&self.identity_checksum.to_le_bytes());
        out[88..96].copy_from_slice(&self.rows_checksum.to_le_bytes());
        out[96..104].copy_from_slice(&self.topology_checksum.to_le_bytes());
        let sum = checksum_of(&out[..HEADER_CHECKSUM_BYTES]);
        out[HEADER_CHECKSUM_BYTES..HEADER_BYTES].copy_from_slice(&sum.to_le_bytes());
        out
    }

    /// Read the header back, refusing what this build cannot interpret: the
    /// magic, the version, the checksum, the reserved fields and the kind, in
    /// that order. Nothing here allocates from a field.
    fn decode(raw: &[u8; HEADER_BYTES]) -> Result<Self, String> {
        if u64::from_le_bytes(take8(raw, 0)) != MAGIC {
            return Err("its header does not open with the dump's magic".to_string());
        }
        let version = u32::from_le_bytes(take4(raw, 8));
        if version != FORMAT_VERSION {
            return Err(format!(
                "its header names format version {}, where a directory at format 5.0.0 \
                 holds version {}",
                version, FORMAT_VERSION
            ));
        }
        let stored = u64::from_le_bytes(take8(raw, HEADER_CHECKSUM_BYTES));
        if stored != checksum_of(&raw[..HEADER_CHECKSUM_BYTES]) {
            return Err("its header fails its checksum".to_string());
        }
        let flags = u32::from_le_bytes(take4(raw, 12));
        if flags != 0 || raw[19] != 0 || raw[104..HEADER_CHECKSUM_BYTES].iter().any(|&b| b != 0) {
            return Err("its header sets a reserved field this build does not know".to_string());
        }
        let kind = GraphKind::from_code(raw[16]).ok_or_else(|| {
            format!(
                "its header names graph kind {}, which is not one this build writes",
                raw[16]
            )
        })?;
        Ok(Header {
            kind,
            element: raw[17],
            layers: raw[18],
            dimension: u32::from_le_bytes(take4(raw, 20)),
            m: u32::from_le_bytes(take4(raw, 24)),
            ef_construction: u32::from_le_bytes(take4(raw, 28)),
            nodes: u32::from_le_bytes(take4(raw, 32)),
            entry_node: u32::from_le_bytes(take4(raw, 36)),
            level_scale: f64::from_le_bytes(take8(raw, 40)),
            identity_bytes: u64::from_le_bytes(take8(raw, 48)),
            rows_bytes: u64::from_le_bytes(take8(raw, 56)),
            topology_bytes: u64::from_le_bytes(take8(raw, 64)),
            file_bytes: u64::from_le_bytes(take8(raw, 72)),
            identity_checksum: u64::from_le_bytes(take8(raw, 80)),
            rows_checksum: u64::from_le_bytes(take8(raw, 88)),
            topology_checksum: u64::from_le_bytes(take8(raw, 96)),
        })
    }
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

// ============================================================================
// WRITING
// ============================================================================

/// The callback a version 1 writer hands each point's lists to.
#[cfg(any(test, feature = "test-support"))]
pub(super) type EachNeighbourhood<'a> = &'a mut dyn FnMut(&[Vec<LoadedEdge>]) -> Result<(), String>;

/// The callback the version 2 writer hands each node's lists to, every
/// target as its node index.
pub(super) type EachLists<'a> = &'a mut dyn FnMut(&[Vec<u32>]) -> Result<(), String>;

/// What the dump writer needs of a graph, in the order the file wants it.
///
/// [`super::mutable::MutableGraph`] answers it, through the `DumpView` it hands
/// out. The writer asks for the points of a layer in rank order, once a
/// region, so each region streams.
pub(super) trait DumpSource<T> {
    /// Points the graph holds.
    fn nb_point(&self) -> usize;
    /// Where the traversal starts, or `None` on an empty graph.
    fn entry(&self) -> Option<PointId>;
    /// Points whose top level is exactly `layer`.
    fn layer_nb_point(&self, layer: usize) -> usize;
    /// Values per stored row.
    fn dimension(&self) -> usize;
    /// `max_nb_connection`, at full width.
    fn max_nb_connection(&self) -> usize;
    /// Width the insertion traversal runs at.
    fn ef_construction(&self) -> usize;
    /// The level generator's scale.
    fn level_scale(&self) -> f64;
    /// Every point of `layer`, in rank order, as its origin id.
    fn each_origin_id(
        &self,
        layer: usize,
        f: &mut dyn FnMut(usize) -> Result<(), String>,
    ) -> Result<(), String>;
    /// Every point of `layer`, in rank order, as its lists by layer, each
    /// target as its index in layer then rank order. Trailing empty lists may
    /// be present and the writer leaves them out.
    fn each_lists(&self, layer: usize, f: EachLists<'_>) -> Result<(), String>;
    /// Every point of `layer`, in rank order, as its lists by layer, each
    /// entry naming its target by layer and rank and carrying the distance,
    /// which is what a version 1 dump records.
    #[cfg(any(test, feature = "test-support"))]
    fn each_neighbourhood(&self, layer: usize, f: EachNeighbourhood<'_>) -> Result<(), String>;
    /// Every point of `layer`, in rank order, as its stored row.
    fn each_vector(
        &self,
        layer: usize,
        f: &mut dyn FnMut(&[T]) -> Result<(), String>,
    ) -> Result<(), String>;
}

/// The node index of the point at `entry`, being its place in layer then
/// rank order.
fn node_index_of(layer_counts: &[usize], entry: PointId) -> Option<u32> {
    let before: usize = layer_counts.get(..entry.0 as usize)?.iter().sum();
    let rank = usize::try_from(entry.1).ok()?;
    u32::try_from(before.checked_add(rank)?).ok()
}

/// Write the graph as the artefact `name` in `dir`, at version 2.
///
/// One pass over the layers for each region, so the writer holds one row or
/// one node's lists at a time and its peak is flat in the graph's size. The
/// header is written as zeros first and again at the end with the region
/// lengths, the region checksums and the magic, and then the file is synced.
pub(super) fn write_dump<T, S>(
    source: &S,
    kind: GraphKind,
    dir: &dyn Dir,
    name: &str,
) -> Result<(), String>
where
    T: DumpElement,
    S: DumpSource<T> + ?Sized,
{
    let nb_point = source.nb_point();
    if nb_point == 0 {
        return Err("the graph holds no points".to_string());
    }
    let nodes = u32::try_from(nb_point).map_err(|_| {
        format!(
            "the graph holds {} points and a node index is a u32",
            nb_point
        )
    })?;
    let entry = source
        .entry()
        .ok_or_else(|| "the graph holds points and no entry point".to_string())?;

    let nb_layer = NB_LAYER_MAX as usize;
    let layer_counts: Vec<usize> = (0..nb_layer)
        .map(|layer| source.layer_nb_point(layer))
        .collect();
    let counted: usize = layer_counts.iter().sum();
    if counted != nb_point {
        return Err(format!(
            "the graph reports {} points and its layers hold {}",
            nb_point, counted
        ));
    }
    let entry_node = node_index_of(&layer_counts, entry)
        .filter(|&node| node < nodes)
        .ok_or_else(|| format!("the entry point sits at layer {} rank {}", entry.0, entry.1))?;

    let dimension = source.dimension();
    if dimension == 0 {
        return Err("the graph holds points of no width".to_string());
    }
    let dimension_field = u32::try_from(dimension)
        .map_err(|_| format!("the graph holds points of {} values", dimension))?;
    let m = u32::try_from(source.max_nb_connection())
        .map_err(|_| format!("the graph declares m {}", source.max_nb_connection()))?;
    let ef_construction = u32::try_from(source.ef_construction()).map_err(|_| {
        format!(
            "the graph declares ef_construction {}",
            source.ef_construction()
        )
    })?;

    let file = dir
        .create(name)
        .map_err(|e| format!("the graph dump could not be created: {}", e))?;
    let mut out = HashingWriter::new(BufWriter::with_capacity(IO_BUFFER_BYTES, file));

    // The header is not part of any region, so its room is written straight
    // through rather than through the hashing writer.
    out.inner
        .write_all(&[0u8; HEADER_BYTES])
        .map_err(|e| format!("the graph dump could not be written: {}", e))?;

    // The identity region: the layer table, then every origin id.
    for count in &layer_counts {
        let count = u32::try_from(*count)
            .map_err(|_| format!("a layer holds {} points and the table is a u32", count))?;
        out.put_u32(count)?;
    }
    for layer in 0..nb_layer {
        source.each_origin_id(layer, &mut |origin_id| {
            let origin_id = u32::try_from(origin_id)
                .map_err(|_| format!("a point's origin id {} is not a u32", origin_id))?;
            out.put_u32(origin_id)
        })?;
    }
    let (identity_checksum, identity_bytes) = out.close_region();

    // The rows region, one row a node in the same order.
    let mut buffer: Vec<u8> = Vec::with_capacity(dimension * T::BYTES);
    for layer in 0..nb_layer {
        source.each_vector(layer, &mut |values| {
            if values.len() != dimension {
                return Err(format!(
                    "a point holds {} values where the graph holds {}",
                    values.len(),
                    dimension
                ));
            }
            buffer.clear();
            T::encode(values, &mut buffer);
            out.put(&buffer)
        })?;
    }
    let (rows_checksum, rows_bytes) = out.close_region();

    // The topology region, one node's lists at a time in the same order.
    for layer in 0..nb_layer {
        source.each_lists(layer, &mut |lists| {
            let list_count = lists
                .iter()
                .rposition(|list| !list.is_empty())
                .map_or(0, |last| last + 1);
            out.put_u8(
                u8::try_from(list_count)
                    .map_err(|_| format!("a point carries adjacency for {} layers", list_count))?,
            )?;
            for list in &lists[..list_count] {
                out.put_u16(u16::try_from(list.len()).map_err(|_| {
                    format!("a point carries {} neighbours at one layer", list.len())
                })?)?;
                for &target in list {
                    out.put_u32(target)?;
                }
            }
            Ok(())
        })?;
    }
    let (topology_checksum, topology_bytes) = out.close_region();

    let file_bytes = HEADER_BYTES as u64 + out.written + TRAILER_BYTES as u64;
    let mut inner = out.inner;
    inner
        .write_all(&file_bytes.to_le_bytes())
        .and_then(|()| inner.write_all(&MAGIC.to_le_bytes()))
        .map_err(|e| format!("the graph dump's trailer could not be written: {}", e))?;

    let header = Header {
        kind,
        element: T::KIND,
        layers: NB_LAYER_MAX,
        dimension: dimension_field,
        m,
        ef_construction,
        nodes,
        entry_node,
        level_scale: source.level_scale(),
        identity_bytes,
        rows_bytes,
        topology_bytes,
        file_bytes,
        identity_checksum,
        rows_checksum,
        topology_checksum,
    };

    // The dump is made durable once the header is in place, as every whole
    // artefact is by `Dir::write`, so the rename that moves the staging
    // directory into place never names a dump whose bytes are still in the
    // page cache.
    let mut file = inner
        .into_inner()
        .map_err(|e| format!("the graph dump could not be flushed: {}", e))?;
    file.write_head(&header.encode())
        .and_then(|()| file.finish())
        .map_err(|e| format!("the graph dump's header could not be written: {}", e))?;

    remove_legacy_dumps(dir, name);
    Ok(())
}

/// Remove the vendored dump's two files where they sit beside the dump `name`
/// just written, under whatever prefix its name carries. Best effort: a file
/// that will not go is logged and the save stands.
fn remove_legacy_dumps(dir: &dyn Dir, name: &str) {
    let prefix = name.rfind('/').map_or("", |slash| &name[..=slash]);
    for legacy in LEGACY_DUMP_FILENAMES {
        let legacy_name = format!("{prefix}{legacy}");
        if dir.exists(&legacy_name) {
            match dir.remove(&legacy_name) {
                Ok(()) => info!(
                    target: LOG_TARGET,
                    operation = "save_hnsw_graph",
                    removed = legacy,
                    "Removed a graph dump written by an earlier release"
                ),
                Err(e) => warn!(
                    target: LOG_TARGET,
                    operation = "save_hnsw_graph",
                    file = legacy,
                    error = %e,
                    "Could not remove a graph dump written by an earlier release"
                ),
            }
        }
    }
}

// ============================================================================
// READING
// ============================================================================

/// What the reader was told to expect, from `config.json`, the manifest and
/// the loaded index.
#[derive(Clone, Copy)]
pub(crate) struct Expected {
    pub kind: GraphKind,
    pub dimension: usize,
    pub m: usize,
    pub ef_construction: usize,
    /// The live record count. The graph holds at least this many nodes and
    /// holds more whenever a removal or an overwrite has stranded one.
    pub min_nodes: usize,
    /// The largest internal id the index has ever issued, from `config.json`.
    ///
    /// A node's origin id is the internal id it was inserted under, so an id
    /// past this one names no record the directory holds and is refused. No
    /// two nodes are filed under one id, so a dump holds at most one node for
    /// each id from zero to this one, and the node count is held to that
    /// before anything is sized from it. A save holds the mutation guard for
    /// its whole run, so the counter `config.json` records and the graph the
    /// same save dumped come from one instant.
    pub max_origin_id: usize,
    /// The version the directory's format requires: 1 below format 5.0.0
    /// and 2 from it.
    pub version: u32,
}

/// Which part of a version 2 dump a refusal is about, or the whole of a
/// version 1 dump, which carries one checksum over everything after its
/// header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DumpRegion {
    /// A version 1 dump, refused for any reason.
    Whole,
    /// The file as a whole, its header, the size its regions agree on, or
    /// its trailer.
    Header,
    /// The layer table and the origin ids.
    Identity,
    /// Every node's stored row.
    Rows,
}

impl DumpRegion {
    /// The region as a refusal names it.
    pub fn label(self) -> &'static str {
        match self {
            DumpRegion::Whole => "dump",
            DumpRegion::Header => "header",
            DumpRegion::Identity => "identity region",
            DumpRegion::Rows => "rows region",
        }
    }
}

/// A dump refused, naming the region and how it fails. `detail` is a clause
/// that reads after "since", such as "its rows region fails its checksum".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpRefusal {
    pub region: DumpRegion,
    pub detail: String,
}

impl DumpRefusal {
    fn header(detail: String) -> Self {
        DumpRefusal {
            region: DumpRegion::Header,
            detail,
        }
    }

    fn identity(detail: String) -> Self {
        DumpRefusal {
            region: DumpRegion::Identity,
            detail,
        }
    }

    fn rows(detail: String) -> Self {
        DumpRefusal {
            region: DumpRegion::Rows,
            detail,
        }
    }
}

/// The verified rows of a version 2 dump, with each node's origin id and
/// position, in the dump's node order.
pub(super) struct ReadRows<T> {
    pub(super) origin_ids: Vec<usize>,
    /// Each node's position among the records the mappings hold, or
    /// `Position::NONE` for a stranded node.
    pub(super) record_of: Vec<u32>,
    pub(super) store: VectorStore<T>,
}

/// What reading a dump gave.
pub(super) enum DumpRead<T, D> {
    /// The graph the dump holds, boxed so the read moves a pointer to it,
    /// with the store it scores against.
    Graph(Box<MutableGraph<T, D>>, VectorStore<T>),
    /// The rows of a version 2 dump, verified, where no graph was built from
    /// it: its topology fails, named in `reason`, or the caller asked for the
    /// rows alone.
    Rows { rows: ReadRows<T>, reason: String },
}

/// Why the rows of a dump read for a rebuild were handed back, when the
/// caller asked for them alone.
pub(super) const ROWS_ASKED_FOR: &str = "the rebuild was asked for";

/// Read the graph back out of the artefact `name` in `dir`, at the version
/// `expected` names, each node filed at the position `positions` gives its
/// origin id.
///
/// `positions` is the internal id of the record at each position, in
/// increasing order, being the records the collection's mappings hold. A node
/// whose origin id it holds takes that position, a node whose origin id it
/// does not hold is stranded, and an id it holds that no node carries refuses
/// the dump. `None` files every node at the rank of its origin id among the
/// dump's, which is a graph whose every node holds a record.
///
/// At version 2, `rows_only` reads and verifies the identity and the rows,
/// and hands the rows back without building a graph. A version 1 dump is
/// read whole or refused, and its refusal names the whole dump.
pub(super) fn read_dump_at<T, D>(
    dir: &dyn Dir,
    name: &str,
    expected: &Expected,
    dist: D,
    positions: Option<&[u32]>,
    rows_only: bool,
) -> Result<DumpRead<T, D>, DumpRefusal>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    if expected.version == 1 {
        return v1::read_dump_at(dir, name, expected, dist, positions)
            .map(|(graph, store)| DumpRead::Graph(Box::new(graph), store))
            .map_err(|detail| DumpRefusal {
                region: DumpRegion::Whole,
                detail,
            });
    }
    let opened = open_version_2::<T>(dir, name, expected, positions)?;
    let Opened {
        layer_counts,
        origin_ids,
        record_of,
        store,
        topology,
        topology_sound,
        entry,
        m,
        ef_construction,
        level_scale,
        nb_point,
        dimension,
    } = opened;
    let rows = |origin_ids, record_of, store, reason: String| {
        Ok(DumpRead::Rows {
            rows: ReadRows {
                origin_ids,
                record_of,
                store,
            },
            reason,
        })
    };
    if rows_only {
        return rows(origin_ids, record_of, store, ROWS_ASKED_FOR.to_string());
    }
    if !topology_sound {
        return rows(
            origin_ids,
            record_of,
            store,
            "its topology region fails its checksum".to_string(),
        );
    }
    let walk = match check_topology(&topology, nb_point, m) {
        Ok(walk) => walk,
        Err(reason) => return rows(origin_ids, record_of, store, reason),
    };
    match MutableGraph::from_adjacency(
        &layer_counts,
        dimension,
        entry,
        m,
        ef_construction,
        level_scale,
        dist,
        origin_ids,
        record_of,
        walk,
    ) {
        Ok(graph) => {
            // Every list is in the graph's own arrays now, so the region's
            // block goes before the graph is handed back.
            drop(topology);
            Ok(DumpRead::Graph(Box::new(graph), store))
        }
        Err(NotBuilt {
            reason,
            origin_ids,
            record_of,
        }) => rows(
            origin_ids,
            record_of,
            store,
            format!(
                "its topology region could not be built into a graph: {}",
                reason
            ),
        ),
    }
}

/// Read a dump into a graph and its store, every node holding a record, or
/// say why not, the reason for rows handed back included. For the tests.
#[cfg(test)]
pub(super) fn read_dump<T, D>(
    dir: &dyn Dir,
    name: &str,
    expected: &Expected,
    dist: D,
) -> Result<(MutableGraph<T, D>, VectorStore<T>), String>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    match read_dump_at(dir, name, expected, dist, None, false) {
        Ok(DumpRead::Graph(graph, store)) => Ok((*graph, store)),
        Ok(DumpRead::Rows { reason, .. }) => Err(reason),
        Err(refusal) => Err(refusal.detail),
    }
}

/// A version 2 dump opened and checked through its trailer, with the
/// topology region read whole and judged on its checksum alone.
struct Opened<T> {
    layer_counts: Vec<usize>,
    origin_ids: Vec<usize>,
    /// Each node's position, or `Position::NONE` for a stranded node; see
    /// [`number_nodes`].
    record_of: Vec<u32>,
    store: VectorStore<T>,
    topology: Vec<u8>,
    /// Whether the topology region's bytes match its checksum.
    topology_sound: bool,
    entry: PointId,
    m: usize,
    ef_construction: usize,
    level_scale: f64,
    nb_point: usize,
    dimension: usize,
}

/// Open a version 2 dump and check everything but the topology's structure:
/// the file's length, the header against itself and against `expected`, the
/// size the regions agree on, the identity region, the rows region, and the
/// trailer. Every refusal names its region. The topology region is read
/// whole, which the size agreement bounds by the file's real length, and
/// judged on its checksum alone.
fn open_version_2<T>(
    dir: &dyn Dir,
    name: &str,
    expected: &Expected,
    positions: Option<&[u32]>,
) -> Result<Opened<T>, DumpRefusal>
where
    T: DumpElement,
{
    let actual_bytes = dir
        .length(name)
        .map_err(|_| DumpRefusal::header("the directory holds no such file".to_string()))?;
    let smallest = (HEADER_BYTES + TRAILER_BYTES) as u64;
    if actual_bytes < smallest {
        return Err(DumpRefusal::header(format!(
            "the file is {} bytes and the smallest dump is {}",
            actual_bytes, smallest
        )));
    }

    let file = dir
        .open(name)
        .map_err(|e| DumpRefusal::header(format!("the file could not be opened: {}", e)))?;
    let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, file);
    let mut raw = [0u8; HEADER_BYTES];
    reader
        .read_exact(&mut raw)
        .map_err(|e| DumpRefusal::header(format!("its header could not be read: {}", e)))?;
    let header = Header::decode(&raw).map_err(DumpRefusal::header)?;

    // The header describes itself consistently. Everything below decides
    // whether it describes this index, and the size agreement is what makes
    // the allocations after it safe.
    let refuse = |detail: String| Err(DumpRefusal::header(detail));
    if header.layers != NB_LAYER_MAX {
        return refuse(format!(
            "its header names {} layers where this build uses {}",
            header.layers, NB_LAYER_MAX
        ));
    }
    if header.element != T::KIND {
        return refuse(format!(
            "its header names element type {} where this index holds {}",
            header.element,
            T::KIND
        ));
    }
    if header.kind != expected.kind {
        return refuse(format!(
            "its header names the {} graph and config.json declares {}",
            header.kind.label(),
            expected.kind.label()
        ));
    }
    if header.dimension as usize != expected.dimension {
        return refuse(format!(
            "its header names rows of {} values where this index holds {}",
            header.dimension, expected.dimension
        ));
    }
    if header.m as usize != expected.m {
        return refuse(format!(
            "its header names m {} and config.json declares {}",
            header.m, expected.m
        ));
    }
    if header.ef_construction as usize != expected.ef_construction {
        return refuse(format!(
            "its header names ef_construction {} and config.json declares {}",
            header.ef_construction, expected.ef_construction
        ));
    }
    if (header.nodes as usize) < expected.min_nodes {
        return refuse(format!(
            "its header names {} nodes and the index holds {} records",
            header.nodes, expected.min_nodes
        ));
    }
    if header.nodes == 0 {
        return refuse("its header names no node".to_string());
    }
    // No two nodes are filed under one id and none above the counter, so the
    // node count is held to the counter before anything is sized from it. A
    // counter at the top of the range allows any count, so the sum is checked
    // rather than wrapped.
    let nodes_the_ids_allow = (expected.max_origin_id as u64).checked_add(1);
    if nodes_the_ids_allow.is_some_and(|allowed| u64::from(header.nodes) > allowed) {
        return refuse(format!(
            "its header names {} nodes and config.json counted {}",
            header.nodes, expected.max_origin_id
        ));
    }
    if header.entry_node >= header.nodes {
        return refuse(format!(
            "its header names entry node {} and the dump holds {} nodes",
            header.entry_node, header.nodes
        ));
    }
    if !header.level_scale.is_finite() || header.level_scale <= 0. {
        return refuse(format!(
            "its header names a level scale of {}",
            header.level_scale
        ));
    }

    let nb_point = header.nodes as usize;
    let dimension = header.dimension as usize;
    let identity_needed = (LAYER_TABLE_BYTES as u64) + (nb_point as u64) * ORIGIN_ID_BYTES as u64;
    if header.identity_bytes != identity_needed {
        return refuse(format!(
            "its header names an identity region of {} bytes where {} nodes need {}",
            header.identity_bytes, nb_point, identity_needed
        ));
    }
    let rows_needed = (nb_point as u64)
        .checked_mul(dimension as u64)
        .and_then(|values| values.checked_mul(T::BYTES as u64));
    if rows_needed != Some(header.rows_bytes) {
        return refuse(format!(
            "its header names a rows region of {} bytes where {} rows of {} values need {}",
            header.rows_bytes,
            nb_point,
            dimension,
            rows_needed.map_or_else(|| "more than a u64 holds".to_string(), |n| n.to_string())
        ));
    }
    let implied = (HEADER_BYTES as u64)
        .checked_add(header.identity_bytes)
        .and_then(|n| n.checked_add(header.rows_bytes))
        .and_then(|n| n.checked_add(header.topology_bytes))
        .and_then(|n| n.checked_add(TRAILER_BYTES as u64))
        .ok_or_else(|| {
            DumpRefusal::header("its header names regions no file could hold".to_string())
        })?;
    if header.file_bytes != implied {
        return refuse(format!(
            "its header names {} bytes and its regions need {}",
            header.file_bytes, implied
        ));
    }
    if header.file_bytes != actual_bytes {
        return refuse(format!(
            "the file is {} bytes and its header names {}",
            actual_bytes, header.file_bytes
        ));
    }

    // Every allocation from here on is bounded by a file that really is this
    // long.
    let mut hashed = HashingReader::new(reader);

    // The identity region, checked against its checksum before its rules.
    let ended = |region: &str, e: std::io::Error| format!("its {} ends early: {}", region, e);
    let mut identity = vec![0u8; header.identity_bytes as usize];
    hashed
        .read_hashed(&mut identity)
        .map_err(|e| DumpRefusal::identity(ended("identity region", e)))?;
    if hashed.close_region() != header.identity_checksum {
        return Err(DumpRefusal::identity(
            "its identity region fails its checksum".to_string(),
        ));
    }
    let mut layer_counts = Vec::with_capacity(NB_LAYER_MAX as usize);
    let mut counted = 0usize;
    for entry in identity[..LAYER_TABLE_BYTES].chunks_exact(4) {
        let count = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
        counted = counted.saturating_add(count);
        layer_counts.push(count);
    }
    if counted != nb_point {
        return Err(DumpRefusal::identity(format!(
            "its identity region's layer table holds {} nodes where its header names {}",
            counted, nb_point
        )));
    }
    let mut origin_ids = Vec::with_capacity(nb_point);
    for entry in identity[LAYER_TABLE_BYTES..].chunks_exact(ORIGIN_ID_BYTES) {
        let id = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
        if id > expected.max_origin_id {
            return Err(DumpRefusal::identity(format!(
                "its identity region names origin id {} and config.json counted {}",
                id, expected.max_origin_id
            )));
        }
        origin_ids.push(id);
    }
    drop(identity);
    let record_of = number_nodes(&origin_ids, positions).map_err(|failure| {
        DumpRefusal::identity(match failure {
            Numbering::Repeated(id) => format!(
                "its identity region names origin id {} for more than one node",
                id
            ),
            Numbering::Unplaced(id) => format!(
                "its identity region names no node for internal id {}, which mappings.bin \
                 holds",
                id
            ),
        })
    })?;
    let entry = point_at(&layer_counts, header.entry_node as usize)
        .expect("the entry node is below the node count the layer table sums to");

    // The rows region, streamed into the store, against its checksum and,
    // on a raw graph, against a value no record can hold.
    let mut store = VectorStore::with_capacity(dimension, nb_point);
    let mut row = vec![0u8; dimension * T::BYTES];
    let mut non_finite: Option<(usize, f32)> = None;
    for node in 0..nb_point {
        hashed
            .read_hashed(&mut row)
            .map_err(|e| DumpRefusal::rows(ended("rows region", e)))?;
        let values = T::decode(&row);
        if non_finite.is_none() {
            if let Some(value) = T::first_non_finite(&values) {
                non_finite = Some((node, value));
            }
        }
        store.append(values);
    }
    if hashed.close_region() != header.rows_checksum {
        return Err(DumpRefusal::rows(
            "its rows region fails its checksum".to_string(),
        ));
    }
    if let Some((node, value)) = non_finite {
        return Err(DumpRefusal::rows(format!(
            "its rows region holds {} in the row of node {}, internal id {}",
            value, node, origin_ids[node]
        )));
    }

    // The topology region, read whole and judged on its checksum here. Its
    // structure is the caller's to check, since a failure there hands back
    // the rows rather than refusing them.
    let mut topology = vec![0u8; header.topology_bytes as usize];
    hashed
        .read_hashed(&mut topology)
        .map_err(|e| DumpRefusal::header(ended("topology region", e)))?;
    let topology_sound = hashed.close_region() == header.topology_checksum;

    let HashingReader { mut inner, .. } = hashed;
    let mut trailer = [0u8; TRAILER_BYTES];
    inner
        .read_exact(&mut trailer)
        .map_err(|e| DumpRefusal::header(format!("its trailer could not be read: {}", e)))?;
    if u64::from_le_bytes(take8(&trailer, 0)) != header.file_bytes
        || u64::from_le_bytes(take8(&trailer, 8)) != MAGIC
    {
        return Err(DumpRefusal::header(
            "its trailer does not close the file its header names".to_string(),
        ));
    }

    Ok(Opened {
        layer_counts,
        origin_ids,
        record_of,
        store,
        topology,
        topology_sound,
        entry,
        m: expected.m,
        ef_construction: header.ef_construction as usize,
        level_scale: header.level_scale,
        nb_point,
        dimension,
    })
}

/// The layer and rank of the node at `index` in layer then rank order.
fn point_at(layer_counts: &[usize], index: usize) -> Option<PointId> {
    let mut first = 0usize;
    for (layer, &count) in layer_counts.iter().enumerate() {
        if index < first + count {
            return Some(PointId(layer as u8, (index - first) as i32));
        }
        first += count;
    }
    None
}

/// Why the origin ids of a dump could not be numbered against the mappings.
pub(super) enum Numbering {
    /// One origin id is named for more than one node.
    Repeated(usize),
    /// An internal id the mappings hold is named for no node.
    Unplaced(u32),
}

/// Each node's position, from the origin ids the dump carries, in node order.
///
/// Pairs of origin id and node are sorted once, by origin id. Two nodes filed
/// under one id are refused, since each id names one record and the index's
/// node for each position is the inverse of the positions here. The sorted
/// pairs are then walked beside `positions`, which lists the internal ids the
/// mappings hold in increasing order. A node whose origin id the mappings hold
/// takes that id's place in the list as its position. A node whose origin id
/// they do not hold is stranded, being a record a removal or an overwrite left
/// behind, and takes `Position::NONE`. An id the mappings hold that no node
/// carries is refused.
///
/// `None` files every node at the rank of its origin id, which is the graph
/// whose every node holds a record.
pub(super) fn number_nodes(
    origin_ids: &[usize],
    positions: Option<&[u32]>,
) -> Result<Vec<u32>, Numbering> {
    let mut order: Vec<(usize, u32)> = origin_ids
        .iter()
        .enumerate()
        .map(|(node, &origin)| (origin, node as u32))
        .collect();
    // By the origin id alone. A dump that names one id twice is refused
    // below, and in any other the ids are distinct, so the node never decides
    // the order.
    order.sort_unstable_by_key(|&(origin, _)| origin);
    if let Some(pair) = order.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(Numbering::Repeated(pair[0].0));
    }
    let mut record_of = vec![crate::space::Position::NONE; origin_ids.len()];
    let Some(held) = positions else {
        for (rank, &(_, node)) in order.iter().enumerate() {
            record_of[node as usize] = rank as u32;
        }
        return Ok(record_of);
    };
    let mut next = 0usize;
    for &(origin, node) in &order {
        let Some(&id) = held.get(next) else {
            break;
        };
        if (id as usize) < origin {
            return Err(Numbering::Unplaced(id));
        }
        if id as usize == origin {
            record_of[node as usize] = next as u32;
            next += 1;
        }
    }
    if let Some(&id) = held.get(next) {
        return Err(Numbering::Unplaced(id));
    }
    Ok(record_of)
}

/// A walk over one version's lists, node by node in layer then rank order,
/// handing each node's lists to the graph's constructor as node indices in
/// the order they install in.
pub(super) trait ListWalk {
    /// The next node's lists, written into the first entries of `lists`,
    /// each cleared first, and how many lists the node carries.
    /// `layer_offsets` is where each layer's run starts in node order.
    fn next_lists(
        &mut self,
        layer_offsets: &[u32; LAYERS + 1],
        lists: &mut Vec<Vec<u32>>,
    ) -> Result<usize, String>;
}

/// Check the topology region whole, building nothing, and hand back the walk
/// the graph's constructor takes over it.
///
/// Every count is checked against the bytes the region has left before
/// anything reads past it, so no count sizes anything here or in the
/// constructor. Every count is held to the slots a list at its layer has in
/// the graph, being `2m + 1` at layer zero and `m + 1` above. Every target is
/// held below the node count, and the region must end where its last node
/// does.
fn check_topology(region: &[u8], nb_point: usize, m: usize) -> Result<TopologyWalk<'_>, String> {
    // Saturating, since `m` is held to its range by the constructor, which
    // runs after this, and an `m` it refuses must not overflow here first.
    let base_cap = m.saturating_mul(2).saturating_add(1);
    let upper_cap = m.saturating_add(1);
    let mut at = 0usize;
    for point in 0..nb_point {
        if at >= region.len() {
            return Err(format!(
                "its topology region ends at node {} of {}",
                point, nb_point
            ));
        }
        let list_count = region[at] as usize;
        at += 1;
        if list_count > NB_LAYER_MAX as usize {
            return Err(format!(
                "its topology region gives node {} lists at {} layers and a node holds at most {}",
                point, list_count, NB_LAYER_MAX
            ));
        }
        for layer in 0..list_count {
            if region.len() - at < ENTRY_COUNT_BYTES {
                return Err(format!(
                    "its topology region ends inside node {} at layer {}",
                    point, layer
                ));
            }
            let count = u16::from_le_bytes(take2(region, at)) as usize;
            at += ENTRY_COUNT_BYTES;
            let needed = count * TARGET_BYTES;
            if needed > region.len() - at {
                return Err(format!(
                    "its topology region gives node {} {} entries at layer {} and the region \
                     has room for {}",
                    point,
                    count,
                    layer,
                    (region.len() - at) / TARGET_BYTES
                ));
            }
            let cap = if layer == 0 { base_cap } else { upper_cap };
            if count > cap {
                return Err(format!(
                    "its topology region gives node {} {} entries at layer {} and a list there \
                     holds {}",
                    point, count, layer, cap
                ));
            }
            for _ in 0..count {
                let target = u32::from_le_bytes(take4(region, at)) as usize;
                at += TARGET_BYTES;
                if target >= nb_point {
                    return Err(format!(
                        "its topology region names node {} from node {} and the dump holds {} \
                         nodes",
                        target, point, nb_point
                    ));
                }
            }
        }
    }
    if at != region.len() {
        return Err(format!(
            "its topology region holds {} bytes beyond its {} nodes",
            region.len() - at,
            nb_point
        ));
    }
    Ok(TopologyWalk { region, at: 0 })
}

/// A walk over a topology region [`check_topology`] accepted, node by node
/// in layer then rank order. Only [`check_topology`] makes one, so the walk
/// reads bytes that were read and accepted in the same order.
pub(super) struct TopologyWalk<'a> {
    region: &'a [u8],
    at: usize,
}

impl ListWalk for TopologyWalk<'_> {
    fn next_lists(
        &mut self,
        _layer_offsets: &[u32; LAYERS + 1],
        lists: &mut Vec<Vec<u32>>,
    ) -> Result<usize, String> {
        let list_count = self.region[self.at] as usize;
        self.at += 1;
        while lists.len() < list_count {
            lists.push(Vec::new());
        }
        for list in lists.iter_mut().take(list_count) {
            list.clear();
            let count = u16::from_le_bytes(take2(self.region, self.at)) as usize;
            self.at += ENTRY_COUNT_BYTES;
            for _ in 0..count {
                list.push(u32::from_le_bytes(take4(self.region, self.at)));
                self.at += TARGET_BYTES;
            }
        }
        Ok(list_count)
    }
}

// ============================================================================
// THE PARSE THE TESTS BUILD FROM
// ============================================================================

/// A dump parsed back into the points a graph constructor takes, with the
/// parameters the header carried.
#[cfg(test)]
#[derive(Clone)]
pub(super) struct ParsedDump<T> {
    /// `points_by_layer[l][r]` is the point at rank `r` of layer `l`.
    pub points_by_layer: Vec<Vec<LoadedPoint<T>>>,
    /// Where the traversal starts.
    pub entry: PointId,
    /// `max_nb_connection`.
    pub m: usize,
    /// As the header carried it.
    pub ef_construction: usize,
    /// As the header carried it.
    pub level_scale: f64,
}

/// A version 2 dump parsed into points, every entry carrying a distance of
/// zero, with every refusal [`read_dump_at`] makes in the same order and the
/// same words, a topology the reader would hand rows back for included.
#[cfg(test)]
pub(super) fn parse_dump<T>(
    dir: &dyn Dir,
    name: &str,
    expected: &Expected,
) -> Result<ParsedDump<T>, String>
where
    T: DumpElement,
{
    if expected.version == 1 {
        return v1::parse_dump(dir, name, expected);
    }
    let opened = open_version_2::<T>(dir, name, expected, None).map_err(|r| r.detail)?;
    if !opened.topology_sound {
        return Err("its topology region fails its checksum".to_string());
    }
    let lists = read_topology(&opened.topology, opened.nb_point, opened.m)?;
    let point_of = |index: u32| {
        point_at(&opened.layer_counts, index as usize)
            .expect("read_topology holds every target below the node count")
    };
    let mut points_by_layer: Vec<Vec<LoadedPoint<T>>> =
        Vec::with_capacity(opened.layer_counts.len());
    let mut node = 0usize;
    for &count in &opened.layer_counts {
        let mut layer = Vec::with_capacity(count);
        for _ in 0..count {
            layer.push(LoadedPoint {
                origin_id: opened.origin_ids[node],
                data: opened.store.get(node as u32).to_vec(),
                neighbours: lists[node]
                    .iter()
                    .map(|list| {
                        list.iter()
                            .map(|&target| LoadedEdge {
                                target: point_of(target),
                                distance: 0.0,
                            })
                            .collect()
                    })
                    .collect(),
            });
            node += 1;
        }
        points_by_layer.push(layer);
    }
    Ok(ParsedDump {
        points_by_layer,
        entry: opened.entry,
        m: opened.m,
        ef_construction: opened.ef_construction,
        level_scale: opened.level_scale,
    })
}

/// The graph `MutableGraph::from_points` builds from a version 2 dump, every
/// list decoded into a block of its own, with every refusal [`read_dump`]
/// gives, in the same order and in the same words. The tests and the fuzzer
/// hold the reader to it graph for graph and refusal for refusal.
#[cfg(test)]
pub(super) fn read_dump_reference<T, D>(
    dir: &dyn Dir,
    name: &str,
    expected: &Expected,
    dist: D,
) -> Result<(MutableGraph<T, D>, VectorStore<T>), String>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    if expected.version == 1 {
        return v1::read_dump_reference(dir, name, expected, dist);
    }
    let opened = open_version_2::<T>(dir, name, expected, None).map_err(|r| r.detail)?;
    if !opened.topology_sound {
        return Err("its topology region fails its checksum".to_string());
    }
    let lists = read_topology(&opened.topology, opened.nb_point, opened.m)?;
    let point_of = |index: u32| {
        point_at(&opened.layer_counts, index as usize)
            .expect("read_topology holds every target below the node count")
    };
    let mut nodes = lists.into_iter().enumerate();
    let origin_ids = opened.origin_ids.clone();
    let store = &opened.store;
    let (graph, built_store) = MutableGraph::from_points(
        &opened.layer_counts,
        opened.dimension,
        opened.entry,
        opened.m,
        opened.ef_construction,
        opened.level_scale,
        dist,
        opened.record_of.clone(),
        || {
            let (node, node_lists) = nodes
                .next()
                .ok_or_else(|| "the graph dump ran out of points".to_string())?;
            Ok(LoadedPoint {
                origin_id: origin_ids[node],
                data: store.get(node as u32).to_vec(),
                neighbours: node_lists
                    .into_iter()
                    .map(|list| {
                        list.into_iter()
                            .map(|target| LoadedEdge {
                                target: point_of(target),
                                distance: 0.0,
                            })
                            .collect()
                    })
                    .collect(),
            })
        },
    )
    .map_err(|e| format!("its topology region could not be built into a graph: {}", e))?;
    Ok((graph, built_store))
}

/// Decode the topology region into one entry per node, every list a block of
/// its own, with the checks [`check_topology`] makes, in the same order and
/// the same words.
#[cfg(test)]
fn read_topology(region: &[u8], nb_point: usize, m: usize) -> Result<Vec<Vec<Vec<u32>>>, String> {
    check_topology(region, nb_point, m)?;
    let mut walk = TopologyWalk { region, at: 0 };
    let offsets = [0u32; LAYERS + 1];
    let mut out = Vec::with_capacity(nb_point);
    for _ in 0..nb_point {
        let mut lists: Vec<Vec<u32>> = Vec::new();
        let count = walk.next_lists(&offsets, &mut lists)?;
        lists.truncate(count);
        out.push(lists);
    }
    Ok(out)
}

/// A parsed dump as a source the writer takes, so a test can alter one thing
/// no save writes and leave every length and every checksum to the writer.
#[cfg(test)]
impl DumpSource<f32> for ParsedDump<f32> {
    fn nb_point(&self) -> usize {
        self.points_by_layer.iter().map(Vec::len).sum()
    }

    fn entry(&self) -> Option<PointId> {
        Some(self.entry)
    }

    fn layer_nb_point(&self, layer: usize) -> usize {
        self.points_by_layer.get(layer).map_or(0, Vec::len)
    }

    fn dimension(&self) -> usize {
        self.points_by_layer
            .iter()
            .flatten()
            .next()
            .map_or(0, |point| point.data.len())
    }

    fn max_nb_connection(&self) -> usize {
        self.m
    }

    fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    fn level_scale(&self) -> f64 {
        self.level_scale
    }

    fn each_origin_id(
        &self,
        layer: usize,
        f: &mut dyn FnMut(usize) -> Result<(), String>,
    ) -> Result<(), String> {
        for point in self.points_by_layer.get(layer).into_iter().flatten() {
            f(point.origin_id)?;
        }
        Ok(())
    }

    fn each_lists(&self, layer: usize, f: EachLists<'_>) -> Result<(), String> {
        let mut first = Vec::with_capacity(self.points_by_layer.len());
        let mut at = 0u32;
        for points in &self.points_by_layer {
            first.push(at);
            at += points.len() as u32;
        }
        for point in self.points_by_layer.get(layer).into_iter().flatten() {
            let lists: Vec<Vec<u32>> = point
                .neighbours
                .iter()
                .map(|list| {
                    list.iter()
                        .map(|edge| {
                            first
                                .get(edge.target.0 as usize)
                                .copied()
                                .unwrap_or(u32::MAX)
                                .wrapping_add(edge.target.1 as u32)
                        })
                        .collect()
                })
                .collect();
            f(&lists)?;
        }
        Ok(())
    }

    fn each_neighbourhood(&self, layer: usize, f: EachNeighbourhood<'_>) -> Result<(), String> {
        for point in self.points_by_layer.get(layer).into_iter().flatten() {
            f(&point.neighbours)?;
        }
        Ok(())
    }

    fn each_vector(
        &self,
        layer: usize,
        f: &mut dyn FnMut(&[f32]) -> Result<(), String>,
    ) -> Result<(), String> {
        for point in self.points_by_layer.get(layer).into_iter().flatten() {
            f(&point.data)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
