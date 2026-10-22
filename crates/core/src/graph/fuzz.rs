//! A seeded mutator over valid graph dumps of both versions, run as an
//! ordinary test.
//!
//! # The property
//!
//! **For every input, the reader builds a graph, hands back the rows or
//! refuses, and never panics, never reads out of bounds and never sizes an
//! allocation from a field the size agreement does not cover.** A refusal
//! reaches the loader's rebuild or refusal path, so any refusal is a pass
//! whatever it says. Each version has its own corpus, its own property test,
//! its own check against the reference reader and its own measure of the
//! repairs.
//!
//! # The repairs
//!
//! The checksums and the magic reject almost every random mutation, so a
//! mutator that only flips bits reaches no parsing code. The mutator repairs.
//! On version 1 it rewrites `file_bytes` to the file's length, recomputes the
//! header checksum at bytes 88 to 96, and recomputes the payload checksum and
//! the end magic in the trailer. On version 2 it rewrites `file_bytes`,
//! rewrites the trailer to the file's length and the magic, recomputes each
//! region's checksum over the bytes the header's region lengths name, and
//! recomputes the header checksum at bytes 120 to 128. Each repair is drawn
//! rather than always applied, so a run covers a mutation the checksums catch
//! and the same mutation carried past them.
//! [`the_repairs_carry_mutations_past_the_checksums`] and
//! [`the_repairs_carry_version_1_mutations_past_the_checksums`] measure how
//! far the repairs carry and fail if that collapses.
//!
//! # Reading from memory
//!
//! Every case is read from [`MemDir`], a directory holding one artefact in
//! memory, so a case costs the reader's own work. The other dump tests read
//! through files.
//!
//! # Determinism
//!
//! The seed is [`SEED`] and the generator is inline, so a failure reproduces
//! byte for byte from the commit alone. Nothing here reads the clock. The
//! environment is read for `ZEUSDB_FUZZ_CASES`, which sets the budget of the
//! property tests and the reference tests for a run by hand,
//! `ZEUSDB_FUZZ_TRACE` and `ZEUSDB_FUZZ_HISTOGRAM`.
//!
//! The harness runs in the ordinary test gate rather than under cargo-fuzz,
//! so it explores by volume rather than by coverage feedback.

use super::dump::{
    read_dump, read_dump_at, read_dump_reference, write_dump, DumpElement, DumpRead, DumpRefusal,
    Expected, GraphKind, DUMP_FILENAME, LEGACY_DUMP_FILENAMES, NB_LAYER_MAX,
};
use super::levels::LevelGenerator;
use super::mutable::MutableGraph;
use super::Distance;
use crate::checksum::checksum_of;
use crate::distance::{CosineDist, DotDist, Int8Dist, Int8Metric, L1Dist, L2Dist};
use crate::error::Error;
use crate::int8::Int8Codec;
use crate::storage::{ArtefactWriter, Dir, FsDir};
use std::cell::Cell;
use std::io::{self, Cursor, Read};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Once};

// ============================================================================
// THE BUDGET
// ============================================================================

/// The generator's seed, which is the whole of this harness's state.
const SEED: u64 = 0x5eed_0099_d00d_face;

/// Cases each property test runs, per corpus entry, unless
/// `ZEUSDB_FUZZ_CASES` sets another number.
///
/// A budget the gate runs on every commit, which makes it a regression net.
/// A deep run sets `ZEUSDB_FUZZ_CASES` for one test named with `--exact`.
const CASES_PER_ENTRY: usize = 1_000;

/// Mutations applied to one case, at most, before the repairs.
const MAX_OPS: usize = 3;

// ============================================================================
// THE GENERATOR
// ============================================================================

/// splitmix64, which is eleven lines and no dependency.
///
/// The crate's own generator is `rand_chacha` and is pinned because every
/// seeded product draw runs on it. This one draws test inputs, so it is kept
/// apart from the production stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value below `bound`, which is zero when `bound` is zero.
    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as usize
        }
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }

    fn chance(&mut self, one_in: u64) -> bool {
        self.next().is_multiple_of(one_in)
    }
}

// ============================================================================
// THE LAYOUTS
// ============================================================================

/// What the mutator needs of one version's layout.
struct Layout {
    /// The version, as [`Expected::version`] names it.
    version: u32,
    /// Bytes the header occupies.
    header: usize,
    /// Where the header checksum sits, which is also the number of bytes it
    /// covers.
    header_checksum: usize,
    /// Header fields a mutation may land on, as a name, an offset and a
    /// width. Named rather than random, since a random byte almost never
    /// lands on a field boundary. The flags and the reserved fields are here
    /// too, since the reader refuses a dump that sets one.
    fields: &'static [(&'static str, usize, usize)],
    /// The header's length fields, each eight bytes, which the size
    /// agreement is made of and which have a draw of their own.
    sizes: &'static [(&'static str, usize)],
    /// The repairs a case draws, in the order they apply.
    repairs: &'static [Op],
}

/// Version 1: a 96 byte header, one payload checksum in the trailer.
const VERSION_1: Layout = Layout {
    version: 1,
    header: 96,
    header_checksum: 88,
    fields: &[
        ("magic", 0, 8),
        ("format_version", 8, 4),
        ("flags", 12, 4),
        ("distance", 16, 1),
        ("element", 17, 1),
        ("nb_layer", 18, 1),
        ("reserved0", 19, 1),
        ("dimension", 20, 4),
        ("m", 24, 8),
        ("ef_construction", 32, 8),
        ("nb_point", 40, 8),
        ("level_scale", 48, 8),
        ("entry_layer", 56, 4),
        ("entry_rank", 60, 4),
        ("reserved1", 80, 8),
        ("header_checksum", 88, 8),
    ],
    sizes: &[("adjacency_bytes", 64), ("file_bytes", 72)],
    repairs: &[Op::RepairFileBytes, Op::RepairHeader, Op::RepairPayload],
};

/// Version 2: a 128 byte header carrying three region checksums, and a
/// trailer of the file's length and the magic.
const VERSION_2: Layout = Layout {
    version: 2,
    header: 128,
    header_checksum: 120,
    fields: &[
        ("magic", 0, 8),
        ("format_version", 8, 4),
        ("flags", 12, 4),
        ("kind", 16, 1),
        ("element", 17, 1),
        ("layers", 18, 1),
        ("reserved", 19, 1),
        ("dimension", 20, 4),
        ("m", 24, 4),
        ("ef_construction", 28, 4),
        ("nodes", 32, 4),
        ("entry_node", 36, 4),
        ("level_scale", 40, 8),
        ("identity_checksum", 80, 8),
        ("rows_checksum", 88, 8),
        ("topology_checksum", 96, 8),
        ("reserved", 104, 8),
        ("reserved", 112, 8),
        ("header_checksum", 120, 8),
    ],
    sizes: &[
        ("identity_bytes", 48),
        ("rows_bytes", 56),
        ("topology_bytes", 64),
        ("file_bytes", 72),
    ],
    repairs: &[
        Op::RepairFileBytes,
        Op::RepairTrailer,
        Op::RepairRegions,
        Op::RepairHeader,
    ],
};

/// Values a field mutation writes, being the edges of every width in the file.
const HOSTILE: [u64; 15] = [
    0,
    1,
    2,
    15,
    16,
    17,
    255,
    256,
    257,
    u32::MAX as u64,
    u32::MAX as u64 + 1,
    i32::MAX as u64,
    i32::MAX as u64 + 1,
    u64::MAX / 2,
    u64::MAX,
];

/// The magic both versions close with.
const MAGIC: u64 = u64::from_le_bytes(*b"ZDBGRAPH");

/// Bytes the trailer occupies in both versions.
const TRAILER: usize = 16;

/// Where a version 2 header holds each region's length and its checksum, in
/// file order.
const REGIONS: [(usize, usize); 3] = [(48, 80), (56, 88), (64, 96)];

// ============================================================================
// THE MUTATIONS
// ============================================================================

/// One mutation, as an offset and an operation.
///
/// A minimised failing input is a list of these plus the corpus entry it was
/// applied to, so a report names the input rather than a blob.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    /// flip one bit
    BitFlip { at: usize, bit: u8 },
    /// overwrite one byte
    SetByte { at: usize, value: u8 },
    /// cut the file short
    Truncate { len: usize },
    /// append bytes
    Extend { len: usize, value: u8 },
    /// zero a run
    ZeroRange { at: usize, len: usize },
    /// copy a run from elsewhere in the same file over another run
    Splice { from: usize, to: usize, len: usize },
    /// write a hostile value into a named header field
    HeaderField {
        name: &'static str,
        at: usize,
        width: usize,
        value: u64,
    },
    /// add `delta` to a named eight byte header field, wrapping
    NudgeField {
        name: &'static str,
        at: usize,
        delta: i64,
    },
    /// rewrite one version 2 list entry, the `pick`th the topology holds, to
    /// a node index below the header's node count
    Target { pick: usize, value: u32 },
    /// rewrite `file_bytes` to the length the file really has
    RepairFileBytes,
    /// recompute the header checksum
    RepairHeader,
    /// recompute the version 1 payload checksum and restamp the end magic
    RepairPayload,
    /// recompute each version 2 region checksum the file has room for
    RepairRegions,
    /// rewrite the version 2 trailer to the file's length and the magic
    RepairTrailer,
}

impl Op {
    fn apply(&self, blob: &mut Vec<u8>, layout: &Layout) {
        match *self {
            Op::BitFlip { at, bit } => {
                if at < blob.len() {
                    blob[at] ^= 1 << (bit & 7);
                }
            }
            Op::SetByte { at, value } => {
                if at < blob.len() {
                    blob[at] = value;
                }
            }
            Op::Truncate { len } => {
                if len < blob.len() {
                    blob.truncate(len);
                }
            }
            Op::Extend { len, value } => blob.extend(std::iter::repeat_n(value, len)),
            Op::ZeroRange { at, len } => {
                let end = (at + len).min(blob.len());
                if at < end {
                    blob[at..end].fill(0);
                }
            }
            Op::Splice { from, to, len } => {
                let len = len
                    .min(blob.len().saturating_sub(from))
                    .min(blob.len().saturating_sub(to));
                if len > 0 {
                    let taken: Vec<u8> = blob[from..from + len].to_vec();
                    blob[to..to + len].copy_from_slice(&taken);
                }
            }
            Op::HeaderField {
                at, width, value, ..
            } => {
                if at + width <= blob.len() {
                    let bytes = value.to_le_bytes();
                    blob[at..at + width].copy_from_slice(&bytes[..width]);
                }
            }
            Op::NudgeField { at, delta, .. } => {
                if let Some(value) = u64_at(blob, at) {
                    let nudged = value.wrapping_add(delta as u64);
                    blob[at..at + 8].copy_from_slice(&nudged.to_le_bytes());
                }
            }
            Op::Target { pick, value } => {
                let targets = topology_targets(blob);
                if !targets.is_empty() {
                    let nodes = u32::from_le_bytes([blob[32], blob[33], blob[34], blob[35]]);
                    let at = targets[pick % targets.len()];
                    let value = value.checked_rem(nodes).unwrap_or(value);
                    blob[at..at + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            Op::RepairFileBytes => {
                if blob.len() >= layout.header {
                    let len = blob.len() as u64;
                    blob[72..80].copy_from_slice(&len.to_le_bytes());
                }
            }
            Op::RepairHeader => {
                let at = layout.header_checksum;
                if blob.len() >= layout.header {
                    let sum = checksum_of(&blob[..at]);
                    blob[at..at + 8].copy_from_slice(&sum.to_le_bytes());
                }
            }
            Op::RepairPayload => {
                if blob.len() >= layout.header + TRAILER {
                    let end = blob.len() - TRAILER;
                    let sum = checksum_of(&blob[layout.header..end]);
                    blob[end..end + 8].copy_from_slice(&sum.to_le_bytes());
                    blob[end + 8..].copy_from_slice(&MAGIC.to_le_bytes());
                }
            }
            Op::RepairRegions => {
                if blob.len() < layout.header {
                    return;
                }
                let mut start = layout.header;
                for (length_at, sum_at) in REGIONS {
                    let end = u64_at(blob, length_at)
                        .and_then(|length| usize::try_from(length).ok())
                        .and_then(|length| start.checked_add(length))
                        .filter(|&end| end <= blob.len());
                    let Some(end) = end else {
                        break;
                    };
                    let sum = checksum_of(&blob[start..end]);
                    blob[sum_at..sum_at + 8].copy_from_slice(&sum.to_le_bytes());
                    start = end;
                }
            }
            Op::RepairTrailer => {
                if blob.len() >= layout.header + TRAILER {
                    let len = blob.len() as u64;
                    let end = blob.len() - TRAILER;
                    blob[end..end + 8].copy_from_slice(&len.to_le_bytes());
                    blob[end + 8..].copy_from_slice(&MAGIC.to_le_bytes());
                }
            }
        }
    }
}

/// The eight bytes at `at`, little-endian, where the blob holds them.
fn u64_at(blob: &[u8], at: usize) -> Option<u64> {
    let bytes = blob.get(at..at.checked_add(8)?)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

/// Where each list entry of a version 2 topology region sits, walking the
/// region the header's lengths name as far as its bytes and its node count
/// allow.
fn topology_targets(blob: &[u8]) -> Vec<usize> {
    let mut targets = Vec::new();
    let (Some(identity), Some(rows), Some(topology)) =
        (u64_at(blob, 48), u64_at(blob, 56), u64_at(blob, 64))
    else {
        return targets;
    };
    let nodes = u32::from_le_bytes([blob[32], blob[33], blob[34], blob[35]]);
    let bounds = (VERSION_2.header as u64)
        .checked_add(identity)
        .and_then(|start| start.checked_add(rows))
        .and_then(|start| Some((start, start.checked_add(topology)?)));
    let Some((start, end)) = bounds else {
        return targets;
    };
    if end > blob.len() as u64 {
        return targets;
    }
    let (mut at, end) = (start as usize, end as usize);
    for _ in 0..nodes {
        if at >= end {
            break;
        }
        let lists = blob[at] as usize;
        at += 1;
        for _ in 0..lists {
            if at + 2 > end {
                return targets;
            }
            let entries = u16::from_le_bytes([blob[at], blob[at + 1]]) as usize;
            at += 2;
            for _ in 0..entries {
                if at + 4 > end {
                    return targets;
                }
                targets.push(at);
                at += 4;
            }
        }
    }
    targets
}

/// Draw one mutation against a blob of `len` bytes.
///
/// Two fifths of every draw is a header field write, since the header is
/// where the arithmetic the reader defends itself with lives. On version 2 a
/// draw in eleven rewrites a list entry to a node the dump holds, which
/// reaches the graph's constructor with a topology every bound admits.
fn draw_op(rng: &mut Rng, len: usize, layout: &Layout) -> Op {
    let kinds = if layout.version == 1 { 10 } else { 11 };
    match rng.below(kinds) {
        0..=2 => {
            let (name, at, width) = layout.fields[rng.below(layout.fields.len())];
            Op::HeaderField {
                name,
                at,
                width,
                value: HOSTILE[rng.below(HOSTILE.len())],
            }
        }
        3 => {
            // A size field, drawn against the hostile table half the time.
            // The other half lands close to the value the size agreement
            // expects: the file's length on version 1, and the field's own
            // value nudged on version 2.
            let (name, at) = layout.sizes[rng.below(layout.sizes.len())];
            if rng.chance(2) {
                Op::HeaderField {
                    name,
                    at,
                    width: 8,
                    value: HOSTILE[rng.below(HOSTILE.len())],
                }
            } else if layout.version == 1 {
                Op::HeaderField {
                    name,
                    at,
                    width: 8,
                    value: (len as u64).wrapping_add(rng.next() % 65).wrapping_sub(32),
                }
            } else {
                Op::NudgeField {
                    name,
                    at,
                    delta: (rng.next() % 65) as i64 - 32,
                }
            }
        }
        4 => Op::BitFlip {
            at: rng.below(len),
            bit: rng.byte(),
        },
        5 => Op::SetByte {
            at: rng.below(len),
            value: rng.byte(),
        },
        6 => Op::Truncate {
            len: rng.below(len),
        },
        7 => Op::Extend {
            len: rng.below(64),
            value: rng.byte(),
        },
        8 => Op::ZeroRange {
            at: rng.below(len),
            len: rng.below(64),
        },
        9 => Op::Splice {
            from: rng.below(len),
            to: rng.below(len),
            len: rng.below(64),
        },
        _ => Op::Target {
            pick: rng.next() as usize,
            value: rng.next() as u32,
        },
    }
}

/// One case, being the mutations and the repairs drawn after them.
fn draw_case(rng: &mut Rng, len: usize, layout: &Layout) -> Vec<Op> {
    let mut ops = Vec::with_capacity(MAX_OPS + layout.repairs.len());
    for _ in 0..1 + rng.below(MAX_OPS) {
        ops.push(draw_op(rng, len, layout));
    }
    for repair in layout.repairs {
        if rng.chance(2) {
            ops.push(*repair);
        }
    }
    ops
}

fn mutate(base: &[u8], ops: &[Op], layout: &Layout) -> Vec<u8> {
    let mut blob = base.to_vec();
    for op in ops {
        op.apply(&mut blob, layout);
    }
    blob
}

// ============================================================================
// THE DIRECTORY IN MEMORY
// ============================================================================

/// A directory holding one artefact, the dump, in memory.
///
/// The reader asks a directory for a length and a stream and nothing else,
/// and every operation that writes panics, so a reader that wrote would fail
/// the property.
struct MemDir {
    bytes: Arc<[u8]>,
}

impl MemDir {
    fn holding(bytes: &[u8]) -> Self {
        MemDir {
            bytes: Arc::from(bytes),
        }
    }

    fn holds(&self, name: &str) -> io::Result<()> {
        if name == DUMP_FILENAME {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("the directory holds no {}", name),
            ))
        }
    }
}

impl Dir for MemDir {
    fn write(&self, name: &str, _bytes: &[u8]) -> Result<(), Error> {
        panic!("the reader wrote {}", name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.holds(name)
            .map(|()| self.bytes.to_vec())
            .map_err(|e| Error::ArtefactReadFailed {
                name: name.to_string(),
                error: e.to_string(),
            })
    }

    fn exists(&self, name: &str) -> bool {
        self.holds(name).is_ok()
    }

    fn length(&self, name: &str) -> io::Result<u64> {
        self.holds(name).map(|()| self.bytes.len() as u64)
    }

    fn remove(&self, name: &str) -> io::Result<()> {
        panic!("the reader removed {}", name)
    }

    fn create(&self, name: &str) -> io::Result<Box<dyn ArtefactWriter>> {
        panic!("the reader created {}", name)
    }

    fn open(&self, name: &str) -> io::Result<Box<dyn Read>> {
        self.holds(name)
            .map(|()| Box::new(Cursor::new(self.bytes.clone())) as Box<dyn Read>)
    }

    fn locate(&self, name: &str) -> String {
        format!("memory/{}", name)
    }
}

// ============================================================================
// THE CORPUS
// ============================================================================

/// A `u8` distance for the product quantized entries.
///
/// L1 over the codes, which is the shape `DistPQ` has once its table is
/// filled, so the entries carry a topology a real one could carry rather
/// than the star a constant distance builds.
struct CodeDist;

impl Distance<u8> for CodeDist {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        va.iter()
            .zip(vb)
            .map(|(a, b)| a.abs_diff(*b) as u32)
            .sum::<u32>() as f32
    }
}

/// Which of the three element types an entry is read back as.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Element {
    Raw,
    Code,
    Int8,
}

/// The distance a scalar dump is read back through. The codec's width is the
/// dump's, its scales are one over the code range so a code decodes to the
/// value it was encoded from, and the metric is one whose row is the codes
/// alone.
fn int8_dist(dim: usize) -> Int8Dist {
    Int8Dist::new(
        Arc::new(Int8Codec::from_scales(vec![1.0 / 127.0; dim]).unwrap()),
        Int8Metric::Dot,
    )
}

/// One valid dump, with what the reader must be told to expect of it.
struct Entry {
    label: &'static str,
    blob: Vec<u8>,
    expected: Expected,
    element: Element,
}

/// Deterministic values, spread enough that the graph has a shape.
fn spread(records: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed | 1;
    (0..records)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / 16_777_216.0 - 0.5
                })
                .collect()
        })
        .collect()
}

/// Build one graph, write it at the layout's version, and hand back the
/// bytes with what to expect.
#[allow(clippy::too_many_arguments)]
fn entry<T, D>(
    layout: &Layout,
    label: &'static str,
    kind: GraphKind,
    element: Element,
    records: usize,
    dim: usize,
    m: usize,
    dist: D,
    value: impl Fn(f32) -> T,
) -> Entry
where
    T: DumpElement + Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    let scale = LevelGenerator::default_scale(m);
    let mut levels = LevelGenerator::new(scale, NB_LAYER_MAX as usize);
    let (mut graph, mut store) = MutableGraph::new(dim, m, 64, scale, records, dist).unwrap();
    for (id, vector) in spread(records, dim, 0x2545_f491_4f6c_dd1d ^ records as u64)
        .into_iter()
        .enumerate()
    {
        let coded: Vec<T> = vector.into_iter().map(&value).collect();
        graph.insert(&mut store, &coded, id, &mut levels);
    }
    let dir = tempfile::tempdir().unwrap();
    let files = FsDir::new(dir.path());
    let view = graph.dump_view(&store);
    if layout.version == 1 {
        super::dump::v1::write_dump(&view, kind, &files, DUMP_FILENAME).unwrap();
    } else {
        write_dump(&view, kind, &files, DUMP_FILENAME).unwrap();
    }
    let blob = std::fs::read(dir.path().join(DUMP_FILENAME)).unwrap();
    Entry {
        label,
        blob,
        expected: Expected {
            kind,
            dimension: dim,
            m,
            ef_construction: 64,
            min_nodes: 0,
            // The corpus inserts under ids 0 to records - 1, so this is the
            // ceiling the directory would carry.
            max_origin_id: records - 1,
            version: layout.version,
        },
        element,
    }
}

/// Every graph the crate writes, at nine shapes, written at the layout's
/// version.
///
/// `min_nodes` is zero throughout, which is the loosest the reader accepts,
/// so a case is refused on the file rather than on a record count. The `m`
/// values bracket the range: 2 is the floor, 16 is the default, and 64 is
/// wide enough that a list-length mutation has somewhere to go.
fn corpus(layout: &Layout) -> Vec<Entry> {
    let code = |v: f32| ((v + 0.5) * 255.0) as u8;
    let int8 = |v: f32| (v * 254.0) as i8;
    vec![
        entry(
            layout,
            "cosine-raw-m16",
            GraphKind::Cosine,
            Element::Raw,
            220,
            6,
            16,
            CosineDist {},
            |v| v,
        ),
        entry(
            layout,
            "dot-raw-m8",
            GraphKind::Dot,
            Element::Raw,
            110,
            5,
            8,
            DotDist {},
            |v| v,
        ),
        entry(
            layout,
            "l2-raw-m2",
            GraphKind::L2,
            Element::Raw,
            90,
            3,
            2,
            L2Dist {},
            |v| v,
        ),
        entry(
            layout,
            "l1-raw-m64",
            GraphKind::L1,
            Element::Raw,
            120,
            4,
            64,
            L1Dist {},
            |v| v,
        ),
        entry(
            layout,
            "cosine-pq-m16",
            GraphKind::CosinePq,
            Element::Code,
            200,
            8,
            16,
            CodeDist,
            code,
        ),
        entry(
            layout,
            "l2-pq-m8",
            GraphKind::L2Pq,
            Element::Code,
            140,
            16,
            8,
            CodeDist,
            code,
        ),
        entry(
            layout,
            "l1-pq-m32",
            GraphKind::L1Pq,
            Element::Code,
            160,
            4,
            32,
            CodeDist,
            code,
        ),
        entry(
            layout,
            "dot-int8-m16",
            GraphKind::DotInt8,
            Element::Int8,
            200,
            8,
            16,
            int8_dist(8),
            int8,
        ),
        entry(
            layout,
            "l2-int8-m8",
            GraphKind::L2Int8,
            Element::Int8,
            140,
            16,
            8,
            Int8Dist::new(
                Arc::new(Int8Codec::from_scales(vec![1.0 / 127.0; 16]).unwrap()),
                Int8Metric::L2,
            ),
            int8,
        ),
    ]
}

// ============================================================================
// THE DRIVER
// ============================================================================

/// What one case did.
enum Outcome {
    /// The reader built a graph.
    Accepted,
    /// The reader handed back the rows of a version 2 dump, for the reason
    /// given.
    Rows(String),
    /// The reader refused, for the reason given.
    Refused(String),
    /// The property broke. The message is the panic's own.
    Panicked(String),
}

/// The panics of a run kept silent on the thread that runs it, for as long as
/// the guard lives.
///
/// Without it a caught panic still prints its message and its backtrace, and a
/// run that finds one finding prints it once per case that reproduces it.
///
/// The panic hook is the process's, and tests run on several threads at once,
/// so the hook is installed once and never put back. It passes every panic to
/// the hook it replaced unless the panicking thread holds a guard, so a test
/// that fails on another thread, during a run or after it, prints its message.
struct Quiet;

thread_local! {
    /// How many guards this thread holds.
    static QUIET: Cell<u32> = const { Cell::new(0) };
}

/// The one installation of the hook [`Quiet`] reads.
static QUIET_HOOK: Once = Once::new();

impl Quiet {
    fn new() -> Self {
        QUIET_HOOK.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if QUIET.with(Cell::get) == 0 {
                    previous(info);
                }
            }));
        });
        QUIET.with(|depth| depth.set(depth.get() + 1));
        Quiet
    }
}

impl Drop for Quiet {
    fn drop(&mut self) {
        QUIET.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Whether every case is announced on stderr before it runs.
///
/// **An allocation failure aborts the process rather than unwinding**, so
/// `catch_unwind` cannot see one and no in-process report can name the case
/// that caused it. `ZEUSDB_FUZZ_TRACE=1` announces each case before driving
/// it, unbuffered, so the last line before an abort is the case.
fn tracing() -> bool {
    std::env::var("ZEUSDB_FUZZ_TRACE").is_ok_and(|v| v != "0")
}

/// The budget a test runs per corpus entry: `ZEUSDB_FUZZ_CASES` where it is
/// set to a number, and `default` otherwise.
fn budget(default: usize) -> usize {
    std::env::var("ZEUSDB_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A read's result as an outcome, the graph and the rows dropped.
fn outcome_of<T, D>(read: Result<DumpRead<T, D>, DumpRefusal>) -> Outcome {
    match read {
        Ok(DumpRead::Graph(..)) => Outcome::Accepted,
        Ok(DumpRead::Rows { reason, .. }) => Outcome::Rows(reason),
        Err(refusal) => Outcome::Refused(refusal.detail),
    }
}

/// The message a caught panic carries.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic carrying no message".to_string())
}

/// Run the reader over `blob`, read from memory, catching a panic.
///
/// `catch_unwind` rather than a subprocess, because the reader owns no global
/// state a poisoned unwind could leave behind: it reads, allocates and
/// returns. It does not catch an abort; see [`tracing`].
fn drive(blob: &[u8], entry: &Entry) -> Outcome {
    let dir = MemDir::holding(blob);
    let expected = entry.expected;
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| match entry.element {
        Element::Raw => outcome_of(read_dump_at::<f32, CosineDist>(
            &dir,
            DUMP_FILENAME,
            &expected,
            CosineDist {},
            None,
            false,
        )),
        Element::Code => outcome_of(read_dump_at::<u8, CodeDist>(
            &dir,
            DUMP_FILENAME,
            &expected,
            CodeDist,
            None,
            false,
        )),
        Element::Int8 => outcome_of(read_dump_at::<i8, Int8Dist>(
            &dir,
            DUMP_FILENAME,
            &expected,
            int8_dist(expected.dimension),
            None,
            false,
        )),
    }));
    result.unwrap_or_else(|payload| Outcome::Panicked(panic_message(payload)))
}

/// Drop every op the failure does not need, one at a time.
///
/// A greedy delta rather than a bisection, since a case is at most seven ops.
fn minimise(entry: &Entry, ops: &[Op], layout: &Layout) -> Vec<Op> {
    let mut kept = ops.to_vec();
    let mut at = 0;
    while at < kept.len() {
        let mut trial = kept.clone();
        trial.remove(at);
        let blob = mutate(&entry.blob, &trial, layout);
        if matches!(drive(&blob, entry), Outcome::Panicked(_)) {
            kept = trial;
        } else {
            at += 1;
        }
    }
    kept
}

/// Whether the reader rejected this input without judging what was in it.
///
/// The magic, the version, the checksums, the trailer, the length floor and
/// the size agreement are integrity checks. They say the bytes are not a
/// whole file, not that the graph they hold is wrong. The property tests
/// count them and fail if they take over the run. The words of both
/// versions are here.
fn refused_on_integrity(reason: &str) -> bool {
    const VERSION_1_WORDS: [&str; 10] = [
        "not a ZeusDB graph dump",
        "header is corrupt",
        "contents are corrupt",
        "does not end where it says",
        "the smallest one is",
        "and the file holds",
        "its own fields need",
        "ended early",
        "format version",
        "holds no ZeusDB graph dump",
    ];
    const VERSION_2_WORDS: [&str; 10] = [
        "does not open with the dump's magic",
        "fails its checksum",
        "the smallest dump is",
        "the file is",
        "and its regions need",
        "regions no file could hold",
        "nodes need",
        "values need",
        "ends early",
        "its trailer",
    ];
    VERSION_1_WORDS
        .iter()
        .chain(VERSION_2_WORDS.iter())
        .any(|words| reason.contains(words))
}

// ============================================================================
// THE PROPERTY
// ============================================================================

/// **The property.** Every mutation of a valid dump of version 2 builds a
/// graph, hands back the rows or is refused, and none panics.
#[test]
fn no_mutation_of_a_valid_dump_panics_the_reader() {
    property(&VERSION_2);
}

/// **The property** over version 1, which every directory saved before
/// format 5.0.0 holds.
#[test]
fn no_mutation_of_a_version_1_dump_panics_the_reader() {
    property(&VERSION_1);
}

/// Every mutation of every corpus entry at `layout`, the budget's worth of
/// cases each.
///
/// Failures are collected rather than asserted one at a time, so one run
/// reports every distinct panic it found. A run judged only on integrity is
/// testing the checksums rather than the reader, so at least one case in
/// five is judged on its contents and at least one in twenty-five builds a
/// graph.
fn property(layout: &Layout) {
    let budget = budget(CASES_PER_ENTRY);
    let trace = tracing();
    let mut rng = Rng(SEED);
    let mut failures: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let (mut accepted, mut rows, mut refused, mut judged) = (0usize, 0usize, 0usize, 0usize);

    {
        let _quiet = Quiet::new();
        for entry in corpus(layout) {
            // The undamaged dump loads, so a refusal below is the mutation's
            // doing rather than the corpus being wrong.
            if !matches!(drive(&entry.blob, &entry), Outcome::Accepted) {
                failures.push(format!(
                    "\n  the corpus entry {} does not load",
                    entry.label
                ));
                continue;
            }

            for case in 0..budget {
                let ops = draw_case(&mut rng, entry.blob.len(), layout);
                if trace {
                    eprintln!("{} case {}: {:?}", entry.label, case, ops);
                }
                let blob = mutate(&entry.blob, &ops, layout);
                match drive(&blob, &entry) {
                    Outcome::Accepted => {
                        accepted += 1;
                        judged += 1;
                    }
                    Outcome::Rows(reason) => {
                        rows += 1;
                        if !refused_on_integrity(&reason) {
                            judged += 1;
                        }
                    }
                    Outcome::Refused(reason) => {
                        refused += 1;
                        if !refused_on_integrity(&reason) {
                            judged += 1;
                        }
                    }
                    Outcome::Panicked(message) => {
                        if !seen.contains(&message) {
                            seen.push(message.clone());
                            let small = minimise(&entry, &ops, layout);
                            failures.push(format!(
                                "\n  {} case {}: {}\n    minimised to {:?}",
                                entry.label, case, message, small
                            ));
                        }
                    }
                }
            }
        }
    }

    let total = accepted + rows + refused + seen.len();
    println!(
        "fuzz version {}: {} cases, {} loaded, {} handed back their rows, {} refused, {} judged \
         on their contents",
        layout.version, total, accepted, rows, refused, judged
    );
    assert!(
        failures.is_empty(),
        "the reader panicked on {} distinct inputs:{}",
        failures.len(),
        failures.join("")
    );
    assert!(
        judged * 5 > total,
        "only {} of {} cases were judged on their contents",
        judged,
        total
    );
    assert!(
        accepted * 25 > total,
        "only {} of {} loaded",
        accepted,
        total
    );
}

// ============================================================================
// THE REPAIRS
// ============================================================================

/// The version 2 repairs reach the parser, measured rather than assumed.
#[test]
fn the_repairs_carry_mutations_past_the_checksums() {
    repairs_carry(&VERSION_2);
}

/// The version 1 repairs reach the parser, measured rather than assumed.
#[test]
fn the_repairs_carry_version_1_mutations_past_the_checksums() {
    repairs_carry(&VERSION_1);
}

/// How often one mutation of the first corpus entry is **accepted whole**,
/// bare and with every repair of `layout` applied after it.
///
/// Acceptance is the measure because it is unambiguous: an accepted file
/// passed the magic, every checksum, the trailer, the size agreement, every
/// field check and the topology. A bare mutation is almost never accepted,
/// since any byte it touches breaks a checksum. At least one repaired
/// mutation in six is accepted, and twenty times as many as bare.
///
/// `ZEUSDB_FUZZ_HISTOGRAM=1` prints what the refusals were.
fn repairs_carry(layout: &Layout) {
    const TRIALS: usize = 1_500;
    let entries = corpus(layout);
    let entry = &entries[0];

    let mut with = 0usize;
    let mut without = 0usize;
    let mut histogram: Vec<(String, usize)> = Vec::new();
    let mut trials = 0usize;
    let keep = std::env::var("ZEUSDB_FUZZ_HISTOGRAM").is_ok();
    let mut rng = Rng(SEED ^ 0xa5a5_a5a5);

    {
        let _quiet = Quiet::new();
        while trials < TRIALS {
            let mut ops = vec![draw_op(&mut rng, entry.blob.len(), layout)];
            let bare = mutate(&entry.blob, &ops, layout);
            // A draw that changed nothing is not a case, and counting it
            // would put a no-op in the bare column.
            if bare == entry.blob {
                continue;
            }
            trials += 1;
            if matches!(drive(&bare, entry), Outcome::Accepted) {
                without += 1;
            }
            ops.extend_from_slice(layout.repairs);
            let outcome = drive(&mutate(&entry.blob, &ops, layout), entry);
            if matches!(outcome, Outcome::Accepted) {
                with += 1;
            }
            if keep {
                let key = match outcome {
                    Outcome::Accepted => "accepted".to_string(),
                    Outcome::Rows(reason) => {
                        format!("rows: {}", reason.chars().take(50).collect::<String>())
                    }
                    Outcome::Refused(reason) => reason.chars().take(56).collect(),
                    Outcome::Panicked(message) => format!("PANICKED {}", message),
                };
                match histogram.iter_mut().find(|(seen, _)| *seen == key) {
                    Some((_, count)) => *count += 1,
                    None => histogram.push((key, 1)),
                }
            }
        }
    }

    if keep {
        histogram.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (key, count) in &histogram {
            println!("fuzz: {:6} {}", count, key);
        }
    }
    println!(
        "fuzz version {}: accepted whole, {} of {TRIALS} repaired against {} of {TRIALS} bare",
        layout.version, with, without
    );
    assert!(
        with * 6 > TRIALS,
        "only {} of {} repaired mutations were accepted whole, so the fuzzer is testing the \
         checksum rather than the parser",
        with,
        TRIALS
    );
    assert!(
        with > without * 20,
        "repairing gained little: {} accepted against {} bare",
        with,
        without
    );
}

// ============================================================================
// THE CORPUS AND THE NAMES
// ============================================================================

/// A file under either name an earlier release wrote reaches no parser, at
/// either version.
///
/// A directory saved by 0.6.0 or earlier still holds the two legacy names.
/// The reader opens only the dump's own name, so with the legacy files alone
/// it finds no dump, and with the dump beside them it reads the dump.
#[test]
fn a_file_under_a_legacy_name_reaches_no_parser() {
    for (layout, absent) in [
        (&VERSION_1, "holds no ZeusDB graph dump"),
        (&VERSION_2, "the directory holds no such file"),
    ] {
        let entries = corpus(layout);
        let entry = &entries[0];
        let dir = tempfile::tempdir().unwrap();
        let mut rng = Rng(SEED ^ 0x1eaf);
        for legacy in LEGACY_DUMP_FILENAMES {
            let ops = draw_case(&mut rng, entry.blob.len(), layout);
            std::fs::write(dir.path().join(legacy), mutate(&entry.blob, &ops, layout)).unwrap();
        }
        match read_dump::<f32, CosineDist>(
            &FsDir::new(dir.path()),
            DUMP_FILENAME,
            &entry.expected,
            CosineDist {},
        ) {
            Ok(_) => panic!("a directory holding only legacy dumps loaded"),
            Err(reason) => assert!(reason.contains(absent), "{}", reason),
        }
        std::fs::write(dir.path().join(DUMP_FILENAME), &entry.blob).unwrap();
        assert!(read_dump::<f32, CosineDist>(
            &FsDir::new(dir.path()),
            DUMP_FILENAME,
            &entry.expected,
            CosineDist {}
        )
        .is_ok());
    }
}

/// The hand-written cases of version 2's own tests, expressed as this
/// mutator's ops, each refused in the words those tests hold.
#[test]
fn the_mutator_expresses_the_hand_written_cases() {
    let layout = &VERSION_2;
    let entries = corpus(layout);
    let entry = &entries[0];
    let len = entry.blob.len();
    let identity = layout.header;
    let rows = identity + u64_at(&entry.blob, 48).unwrap() as usize;
    let topology = rows + u64_at(&entry.blob, 56).unwrap() as usize;

    let case = |ops: Vec<Op>, needle: &str| match drive(&mutate(&entry.blob, &ops, layout), entry) {
        Outcome::Refused(reason) | Outcome::Rows(reason) => {
            assert!(reason.contains(needle), "{:?} gave {}", ops, reason)
        }
        Outcome::Accepted => panic!("{:?} was accepted", ops),
        Outcome::Panicked(message) => panic!("{:?} panicked: {}", ops, message),
    };

    case(vec![Op::Truncate { len: 0 }], "the smallest dump is 144");
    case(
        vec![Op::Truncate { len: layout.header }],
        "the smallest dump is 144",
    );
    case(vec![Op::Truncate { len: len / 2 }], "the file is");
    case(vec![Op::Truncate { len: len - 1 }], "the file is");
    case(vec![Op::Extend { len: 16, value: 0 }], "the file is");
    case(
        vec![Op::BitFlip { at: 0, bit: 0 }],
        "does not open with the dump's magic",
    );
    case(
        vec![Op::HeaderField {
            name: "format_version",
            at: 8,
            width: 4,
            value: 99,
        }],
        "names format version 99",
    );
    case(
        vec![Op::BitFlip { at: 32, bit: 0 }],
        "its header fails its checksum",
    );
    case(
        vec![Op::BitFlip {
            at: identity + 3,
            bit: 4,
        }],
        "its identity region fails its checksum",
    );
    case(
        vec![Op::BitFlip {
            at: rows + 5,
            bit: 4,
        }],
        "its rows region fails its checksum",
    );
    case(
        vec![Op::BitFlip {
            at: topology + 3,
            bit: 4,
        }],
        "its topology region fails its checksum",
    );
    case(
        vec![Op::SetByte {
            at: len - 1,
            value: 0,
        }],
        "its trailer does not close the file",
    );
    case(
        vec![
            Op::HeaderField {
                name: "nodes",
                at: 32,
                width: 4,
                value: 1_000_000_000,
            },
            Op::RepairHeader,
        ],
        "its header names 1000000000 nodes and config.json counted 219",
    );
    case(
        vec![
            Op::HeaderField {
                name: "topology_bytes",
                at: 64,
                width: 8,
                value: u64::MAX,
            },
            Op::RepairHeader,
        ],
        "regions no file could hold",
    );
    case(
        vec![
            Op::SetByte {
                at: topology + 1 + 2 + 3,
                value: 0xff,
            },
            Op::RepairRegions,
            Op::RepairHeader,
        ],
        "from node 0 and the dump holds 220 nodes",
    );
}

/// The hand-written cases of version 1's own tests, expressed as this
/// mutator's ops, each refused in the words those tests hold.
#[test]
fn the_mutator_expresses_the_version_1_hand_written_cases() {
    let layout = &VERSION_1;
    let entries = corpus(layout);
    let entry = &entries[0];
    let len = entry.blob.len();

    let case = |ops: Vec<Op>, needle: &str| match drive(&mutate(&entry.blob, &ops, layout), entry) {
        Outcome::Refused(reason) => {
            assert!(reason.contains(needle), "{:?} gave {}", ops, reason)
        }
        Outcome::Rows(reason) => panic!("{:?} handed back its rows: {}", ops, reason),
        Outcome::Accepted => panic!("{:?} was accepted", ops),
        Outcome::Panicked(message) => panic!("{:?} panicked: {}", ops, message),
    };

    case(vec![Op::Truncate { len: 0 }], "smallest");
    case(vec![Op::Truncate { len: layout.header }], "smallest");
    case(
        vec![Op::Truncate {
            len: layout.header + 16,
        }],
        "the file holds",
    );
    case(vec![Op::Truncate { len: len / 2 }], "the file holds");
    case(vec![Op::Truncate { len: len - 1 }], "the file holds");
    case(
        vec![Op::Extend { len: 16, value: 0 }],
        "the graph dump declares",
    );
    case(
        vec![Op::BitFlip { at: 0, bit: 0 }],
        "not a ZeusDB graph dump",
    );
    case(
        vec![Op::HeaderField {
            name: "format_version",
            at: 8,
            width: 4,
            value: 99,
        }],
        "format version",
    );
    case(vec![Op::BitFlip { at: 40, bit: 0 }], "header is corrupt");
    case(
        vec![Op::BitFlip {
            at: len - 32,
            bit: 4,
        }],
        "contents are corrupt",
    );
    case(
        vec![Op::SetByte {
            at: len - 1,
            value: 0,
        }],
        "does not end where it says",
    );
    case(
        vec![
            Op::HeaderField {
                name: "nb_point",
                at: 40,
                width: 8,
                value: 1_000_000_000,
            },
            Op::RepairHeader,
        ],
        "its own fields need",
    );
    case(
        vec![
            Op::HeaderField {
                name: "nb_point",
                at: 40,
                width: 8,
                value: u64::MAX / 2,
            },
            Op::RepairHeader,
        ],
        "no file could hold",
    );
}

/// Every corpus entry of both versions reads back whole, so a failure above
/// is the reader's rather than one of the eighteen graphs this file builds.
#[test]
fn every_corpus_entry_round_trips() {
    for layout in [&VERSION_1, &VERSION_2] {
        for entry in corpus(layout) {
            let dir = MemDir::holding(&entry.blob);
            let counts = match entry.element {
                Element::Raw => read_dump::<f32, CosineDist>(
                    &dir,
                    DUMP_FILENAME,
                    &entry.expected,
                    CosineDist {},
                )
                .map(|(graph, store)| (graph.nb_points(), store.len())),
                Element::Code => {
                    read_dump::<u8, CodeDist>(&dir, DUMP_FILENAME, &entry.expected, CodeDist)
                        .map(|(graph, store)| (graph.nb_points(), store.len()))
                }
                Element::Int8 => read_dump::<i8, Int8Dist>(
                    &dir,
                    DUMP_FILENAME,
                    &entry.expected,
                    int8_dist(entry.expected.dimension),
                )
                .map(|(graph, store)| (graph.nb_points(), store.len())),
            }
            .unwrap_or_else(|e| {
                panic!(
                    "{} at version {} did not load: {}",
                    entry.label, layout.version, e
                )
            });
            assert_eq!(counts.0, counts.1, "{} lost vectors", entry.label);
            assert_eq!(
                counts.0,
                entry.expected.max_origin_id + 1,
                "{}",
                entry.label
            );
        }
    }
}

// ============================================================================
// THE REFERENCE
// ============================================================================

/// Cases each reference test runs, per corpus entry, beside the entry as
/// written, unless `ZEUSDB_FUZZ_CASES` sets another number. A quarter of the
/// property test's budget, since every case is read twice.
const REFERENCE_CASES_PER_ENTRY: usize = 250;

/// The dump in `dir` read by the reader and by the reference, which decodes
/// every list into a block of its own and builds a point at a time.
///
/// `Ok(true)` where both load and the two graphs agree buffer for buffer and
/// capacity for capacity and the two stores value for value, `Ok(false)`
/// where both refuse in the same words, rows the reader handed back counting
/// as a refusal in their reason's words, and what differs otherwise, a panic
/// on either side included.
fn read_as_the_points<T, D>(
    dir: &MemDir,
    expected: &Expected,
    dist: impl Fn() -> D,
) -> Result<bool, String>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    let read = std::panic::catch_unwind(AssertUnwindSafe(|| {
        read_dump::<T, D>(dir, DUMP_FILENAME, expected, dist())
    }));
    let reference = std::panic::catch_unwind(AssertUnwindSafe(|| {
        read_dump_reference::<T, D>(dir, DUMP_FILENAME, expected, dist())
    }));
    match (read, reference) {
        (Ok(Ok((graph, store))), Ok(Ok((reference_graph, reference_store)))) => {
            if graph.arrays() != reference_graph.arrays() {
                return Err("the two graphs differ".to_string());
            }
            let mut values = Vec::new();
            let mut reference_values = Vec::new();
            T::encode(store.values(), &mut values);
            T::encode(reference_store.values(), &mut reference_values);
            if values != reference_values || store.memory_bytes() != reference_store.memory_bytes()
            {
                return Err("the two stores differ".to_string());
            }
            Ok(true)
        }
        (Ok(Err(reason)), Ok(Err(reference_reason))) => {
            if reason == reference_reason {
                Ok(false)
            } else {
                Err(format!(
                    "refused as {:?} where the reference refused as {:?}",
                    reason, reference_reason
                ))
            }
        }
        (Ok(Ok(_)), Ok(Err(reference_reason))) => Err(format!(
            "loaded where the reference refused as {:?}",
            reference_reason
        )),
        (Ok(Err(reason)), Ok(Ok(_))) => Err(format!(
            "refused as {:?} where the reference loaded",
            reason
        )),
        (Err(_), _) => Err("the reader panicked".to_string()),
        (_, Err(_)) => Err("the reference panicked".to_string()),
    }
}

/// The reader against the reference over version 2, outcome for outcome.
#[test]
fn every_mutation_reads_as_the_points_read_it() {
    reads_as_the_points(&VERSION_2);
}

/// The reader against the reference over version 1, outcome for outcome.
#[test]
fn every_version_1_mutation_reads_as_the_points_read_it() {
    reads_as_the_points(&VERSION_1);
}

/// Every corpus entry at `layout`, as written and mutated, either loads
/// under both readers to the same graph and the same store or is refused by
/// both in the same words. Neither the loaded share nor the refused share
/// may be empty, so the check is not made over one outcome alone.
fn reads_as_the_points(layout: &Layout) {
    let budget = budget(REFERENCE_CASES_PER_ENTRY);
    let mut rng = Rng(SEED ^ 0x0188_0188);
    let mut differing: Vec<String> = Vec::new();
    let mut loaded = 0usize;
    let mut refused = 0usize;

    {
        let _quiet = Quiet::new();
        for entry in corpus(layout) {
            for case in 0..=budget {
                // Case zero is the entry as written.
                let ops = if case == 0 {
                    Vec::new()
                } else {
                    draw_case(&mut rng, entry.blob.len(), layout)
                };
                let dir = MemDir::holding(&mutate(&entry.blob, &ops, layout));
                let outcome = match entry.element {
                    Element::Raw => {
                        read_as_the_points::<f32, _>(&dir, &entry.expected, || CosineDist {})
                    }
                    Element::Code => {
                        read_as_the_points::<u8, _>(&dir, &entry.expected, || CodeDist)
                    }
                    Element::Int8 => read_as_the_points::<i8, _>(&dir, &entry.expected, || {
                        int8_dist(entry.expected.dimension)
                    }),
                };
                match outcome {
                    Ok(true) => loaded += 1,
                    Ok(false) => refused += 1,
                    Err(difference) => differing.push(format!(
                        "\n  {} case {}: {}\n    ops {:?}",
                        entry.label, case, difference, ops
                    )),
                }
            }
        }
    }

    println!(
        "fuzz version {} against the points: {} cases, {} loaded by both, {} refused by both in \
         the same words",
        layout.version,
        loaded + refused + differing.len(),
        loaded,
        refused
    );
    assert!(
        differing.is_empty(),
        "the reader and the points parted on {} cases:{}",
        differing.len(),
        differing.join("")
    );
    assert!(
        loaded > 0 && refused > 0,
        "{} loaded and {} refused",
        loaded,
        refused
    );
}
