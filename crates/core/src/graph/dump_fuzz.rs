//! The dump fuzzer's mutator: a seeded generator, the layout of each dump
//! version, the mutations a case draws and the repairs that carry them past
//! the dump's checksums.
//!
//! Compiled under the `test-support` feature and in this crate's own tests,
//! so the wheel carries none of it. This crate's dump fuzzer draws from it,
//! and a fuzzer in another crate that mutates a saved directory's dump draws
//! from [`mutate_dump`].

use crate::checksum::checksum_of;

/// Mutations applied to one case, at most, before the repairs.
pub(crate) const MAX_OPS: usize = 3;

// ============================================================================
// THE GENERATOR
// ============================================================================

/// splitmix64, which is eleven lines and no dependency.
///
/// The crate's own generator is `rand_chacha` and is pinned because every
/// seeded product draw runs on it. This one draws test inputs, so it is kept
/// apart from the production stream.
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a draw taken modulo a usize bound, so it fits a usize"
    )]
    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.draw() % bound as u64) as usize
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the low byte of a shifted draw, which the cast takes by design"
    )]
    pub fn byte(&mut self) -> u8 {
        (self.draw() >> 24) as u8
    }

    pub fn chance(&mut self, one_in: u64) -> bool {
        self.draw().is_multiple_of(one_in)
    }
}

// ============================================================================
// THE LAYOUTS
// ============================================================================

/// What the mutator needs of one version's layout.
pub(crate) struct Layout {
    /// The version, as [`Expected::version`] names it.
    pub(crate) version: u32,
    /// Bytes the header occupies.
    pub(crate) header: usize,
    /// Where the header checksum sits, which is also the number of bytes it
    /// covers.
    pub(crate) header_checksum: usize,
    /// Header fields a mutation may land on, as a name, an offset and a
    /// width. Named rather than random, since a random byte almost never
    /// lands on a field boundary. The flags and the reserved fields are here
    /// too, since the reader refuses a dump that sets one.
    pub(crate) fields: &'static [(&'static str, usize, usize)],
    /// The header's length fields, each eight bytes, which the size
    /// agreement is made of and which have a draw of their own.
    pub(crate) sizes: &'static [(&'static str, usize)],
    /// The repairs a case draws, in the order they apply.
    pub(crate) repairs: &'static [Op],
}

/// Version 1: a 96 byte header, one payload checksum in the trailer.
pub(crate) const VERSION_1: Layout = Layout {
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
pub(crate) const VERSION_2: Layout = Layout {
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
pub(crate) enum Op {
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
    pub(crate) fn apply(&self, blob: &mut Vec<u8>, layout: &Layout) {
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
pub(crate) fn u64_at(blob: &[u8], at: usize) -> Option<u64> {
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "two offsets held within the blob's length just above"
    )]
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
pub(crate) fn draw_op(rng: &mut Rng, len: usize, layout: &Layout) -> Op {
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
                    value: (len as u64).wrapping_add(rng.draw() % 65).wrapping_sub(32),
                }
            } else {
                Op::NudgeField {
                    name,
                    at,
                    delta: (rng.draw() % 65) as i64 - 32,
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
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a pick taken modulo the entries and a value taken modulo the nodes, so any                       bits of a draw serve"
        )]
        _ => Op::Target {
            pick: rng.draw() as usize,
            value: rng.draw() as u32,
        },
    }
}

/// One case, being the mutations and the repairs drawn after them.
pub(crate) fn draw_case(rng: &mut Rng, len: usize, layout: &Layout) -> Vec<Op> {
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

pub(crate) fn mutate(base: &[u8], ops: &[Op], layout: &Layout) -> Vec<u8> {
    let mut blob = base.to_vec();
    for op in ops {
        op.apply(&mut blob, layout);
    }
    blob
}

/// One case drawn from `rng` and applied to a copy of `base`, a valid dump of
/// either version: one to three mutations, then each repair of its version
/// drawn with even odds, so a case either carries its mutations past the
/// checksums or is refused by one. The version is read from the dump's
/// format field, and a blob too short to hold one is read as version 2.
pub fn mutate_dump(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
    let version = base
        .get(8..12)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    let layout = if version == Some(1) {
        &VERSION_1
    } else {
        &VERSION_2
    };
    let ops = draw_case(rng, base.len(), layout);
    mutate(base, &ops, layout)
}
