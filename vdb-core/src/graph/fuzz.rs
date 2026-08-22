//! A seeded mutator over a valid graph dump, run as an ordinary test.
//!
//! # What it is for
//!
//! [`super::dump`] had been damaged by hand in 43 enumerated cases across three
//! relays, every one of them a specific offset with a specific expectation. The
//! general form of all of them is an arbitrary mutation of a valid dump, and
//! that is what this drives.
//!
//! The property is one sentence. **For every input, `read_dump` returns `Ok` or
//! `Err` and never panics, never reads out of bounds and never sizes an
//! allocation from a field the size agreement does not cover.** A refusal
//! reaches the loader's rebuild path, which is where every refusal reaches, so
//! an `Err` is a pass whatever it says.
//!
//! # Why it repairs checksums
//!
//! A naive byte mutator over this format is worthless. The header checksum, the
//! payload checksum and the end magic between them reject almost every random
//! mutation, so a mutator that only flips bits proves the checksums work and
//! reaches no parsing code at all.
//!
//! So the mutator **repairs**. After touching the header it recomputes bytes 88
//! to 96, after touching the body it recomputes the payload checksum and
//! restamps the end magic, and it can rewrite `file_bytes` to whatever length
//! the mutated file really has. That is the only route past `Header::decode`
//! into the arithmetic below it, and `a_header_claiming_a_billion_points_
//! allocates_nothing` had to do the same repair by hand for the same reason.
//! [`the_repairs_carry_mutations_past_the_checksums`] measures the difference
//! rather than assuming it, and fails if it collapses.
//!
//! # Why it is not cargo-fuzz
//!
//! Three reasons, any one decisive. The crate is `crate-type = ["cdylib"]`, so
//! nothing can link it as a Rust library, and reaching the reader from a
//! separate `fuzz` crate would mean publishing `graph`, `graph::dump`,
//! `parse_dump`, `Expected`, `GraphKind`, `DumpElement` and `MutableGraph` as
//! permanent API on a crate whose only consumer is a Python module. A
//! `fuzz/rust-toolchain.toml` on nightly reintroduces the two toolchain drift
//! that `rust-toolchain.toml` exists to remove. And libFuzzer is not supported
//! under MSVC, so it could not run on the machine this was written on.
//!
//! What this gives up against cargo-fuzz is coverage feedback. It explores by
//! volume rather than by novelty, so it is weaker per case, and in exchange it
//! runs in the ordinary lint and test gate on every commit, which a
//! coverage-guided fuzzer with a time budget does not.
//!
//! # Determinism
//!
//! The seed is [`SEED`] and the generator is inline, so a failure reproduces
//! byte for byte from the commit alone. Nothing here reads the clock and
//! nothing reads the environment except `ZEUSDB_FUZZ_CASES`, which raises the
//! budget for a soak run by hand and is absent in CI.

use super::dump::{
    read_dump, write_dump, DumpElement, Expected, GraphKind, DUMP_FILENAME, LEGACY_DUMP_FILENAMES,
    NB_LAYER_MAX,
};
use super::levels::LevelGenerator;
use super::mutable::MutableGraph;
use super::Distance;
use crate::distance::{CosineDist, L1Dist, L2Dist};
use std::path::Path;

// ============================================================================
// THE BUDGET
// ============================================================================

/// The generator's seed, which is the whole of this harness's state.
const SEED: u64 = 0x5eed_0099_d00d_face;

/// Cases the committed test runs, per corpus entry.
///
/// A budget rather than a target, because this runs on every commit and a gate
/// nobody can afford to run finds nothing. Measured on the machine it was
/// written on, a case costs about 6.3 ms, nearly all of it writing the mutated
/// dump and reading it back, so six entries at this budget is about forty
/// seconds against a `cargo test` that already takes two minutes.
///
/// It is not the number that finds things. The origin id defect was case 248 of
/// the first entry, and the deep runs are the soak: `ZEUSDB_FUZZ_CASES=20000`
/// takes eleven minutes and covers 120,000 cases.
const CASES_PER_ENTRY: usize = 1_000;

/// Mutations applied to one case, at most, before the repairs.
const MAX_OPS: usize = 3;

// ============================================================================
// THE GENERATOR
// ============================================================================

/// splitmix64, which is eleven lines and no dependency.
///
/// The crate's own generator is `rand_chacha` and is pinned because every
/// seeded *product* draw runs on it. This one draws test inputs, so pinning it
/// to the production stream would couple two things that have no reason to move
/// together.
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
// THE MUTATIONS
// ============================================================================

/// Header fields a mutation may land on, as a name, an offset and a width.
///
/// Named rather than random, because the header is 96 bytes holding fourteen
/// fields and a random byte almost never lands on a field boundary. `flags`,
/// `reserved0` and `reserved1` are here too, since the reader refuses a dump
/// that sets one and that refusal is part of the contract.
const HEADER_FIELDS: [(&str, usize, usize); 16] = [
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
];

/// `adjacency_bytes` and `file_bytes`, which the size agreement is made of and
/// which therefore get their own draw against the file's real length.
const SIZE_FIELDS: [(&str, usize); 2] = [("adjacency_bytes", 64), ("file_bytes", 72)];

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

/// The end magic, which the payload repair restamps.
const END_MAGIC: u64 = u64::from_le_bytes(*b"ZDBGRAPH");

/// Bytes the header occupies, restated here so the mutator does not need the
/// reader's private constants.
const HEADER: usize = 96;

/// Bytes the trailer occupies.
const TRAILER: usize = 16;

/// One mutation, as an offset and an operation.
///
/// A minimised failing input is a list of these plus the corpus entry it was
/// applied to, so a report can name the input rather than dump a blob.
#[derive(Clone, Debug, PartialEq)]
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
    /// rewrite `file_bytes` to the length the file really has
    RepairFileBytes,
    /// recompute the header checksum over bytes 0 to 88
    RepairHeader,
    /// recompute the payload checksum and restamp the end magic
    RepairPayload,
}

impl Op {
    fn apply(&self, blob: &mut Vec<u8>) {
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
            Op::RepairFileBytes => {
                if blob.len() >= HEADER {
                    let len = blob.len() as u64;
                    blob[72..80].copy_from_slice(&len.to_le_bytes());
                }
            }
            Op::RepairHeader => {
                if blob.len() >= HEADER {
                    let sum = crate::checksum::checksum_of(&blob[..88]);
                    blob[88..96].copy_from_slice(&sum.to_le_bytes());
                }
            }
            Op::RepairPayload => {
                if blob.len() >= HEADER + TRAILER {
                    let end = blob.len() - TRAILER;
                    let sum = crate::checksum::checksum_of(&blob[HEADER..end]);
                    blob[end..end + 8].copy_from_slice(&sum.to_le_bytes());
                    blob[end + 8..].copy_from_slice(&END_MAGIC.to_le_bytes());
                }
            }
        }
    }
}

/// Draw one mutation against a blob of `len` bytes.
///
/// The weights are deliberate. Two fifths of every draw is a header field
/// write, because the header is where the arithmetic the reader defends itself
/// with lives.
fn draw_op(rng: &mut Rng, len: usize) -> Op {
    match rng.below(10) {
        0..=2 => {
            let (name, at, width) = HEADER_FIELDS[rng.below(HEADER_FIELDS.len())];
            Op::HeaderField {
                name,
                at,
                width,
                value: HOSTILE[rng.below(HOSTILE.len())],
            }
        }
        3 => {
            // A size field, drawn against the file's own length as well as
            // against the hostile table, because the interesting values for
            // these two sit close to what the size agreement expects.
            let (name, at) = SIZE_FIELDS[rng.below(SIZE_FIELDS.len())];
            let value = if rng.chance(2) {
                HOSTILE[rng.below(HOSTILE.len())]
            } else {
                (len as u64).wrapping_add(rng.next() % 65).wrapping_sub(32)
            };
            Op::HeaderField {
                name,
                at,
                width: 8,
                value,
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
        _ => Op::Splice {
            from: rng.below(len),
            to: rng.below(len),
            len: rng.below(64),
        },
    }
}

/// One case, being the mutations and the repairs that follow them.
///
/// The repairs are drawn rather than always applied, so the run covers both a
/// mutation the checksums catch and the same mutation carried past them.
fn draw_case(rng: &mut Rng, len: usize) -> Vec<Op> {
    let mut ops = Vec::with_capacity(MAX_OPS + 3);
    for _ in 0..1 + rng.below(MAX_OPS) {
        ops.push(draw_op(rng, len));
    }
    if rng.chance(2) {
        ops.push(Op::RepairFileBytes);
    }
    if rng.chance(2) {
        ops.push(Op::RepairHeader);
    }
    if rng.chance(2) {
        ops.push(Op::RepairPayload);
    }
    ops
}

fn mutate(base: &[u8], ops: &[Op]) -> Vec<u8> {
    let mut blob = base.to_vec();
    for op in ops {
        op.apply(&mut blob);
    }
    blob
}

// ============================================================================
// THE CORPUS
// ============================================================================

/// A `u8` distance for the quantized half of the corpus.
///
/// L1 over the codes, which is the shape `DistPQ` has once its table is filled.
/// It is here so the quantized dumps carry a topology a real one could carry,
/// rather than the degenerate star a constant distance builds.
struct CodeDist;

impl Distance<u8> for CodeDist {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        va.iter()
            .zip(vb)
            .map(|(a, b)| a.abs_diff(*b) as u32)
            .sum::<u32>() as f32
    }
}

/// What a corpus entry is read back as.
///
/// `read_dump` is generic over the element type and the distance, and these are
/// the two shapes the crate ships, so the entry carries which of them applies
/// rather than the driver guessing.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Element {
    Raw,
    Code,
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

/// Build one graph, write it, and hand back the bytes with what to expect.
#[allow(clippy::too_many_arguments)]
fn entry<T, D>(
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
    write_dump(&graph.dump_view(&store), kind, dir.path()).unwrap();
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
            // ceiling the directory would carry. A mutation that raises an
            // origin id past it is refused rather than allocating from it.
            max_origin_id: records - 1,
        },
        element,
    }
}

/// Every graph the crate writes, at six shapes.
///
/// `min_nodes` is zero throughout, which is the loosest the reader accepts, so
/// a case is refused on the file rather than on a record count this file chose.
/// The `m` values bracket the range: 2 is the floor, 16 is the default, and 64
/// is wide enough that a list-length mutation has somewhere to go.
fn corpus() -> Vec<Entry> {
    vec![
        entry(
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
            "cosine-pq-m16",
            GraphKind::CosinePq,
            Element::Code,
            200,
            8,
            16,
            CodeDist,
            |v| ((v + 0.5) * 255.0) as u8,
        ),
        entry(
            "l2-pq-m8",
            GraphKind::L2Pq,
            Element::Code,
            140,
            16,
            8,
            CodeDist,
            |v| ((v + 0.5) * 255.0) as u8,
        ),
        entry(
            "l1-pq-m32",
            GraphKind::L1Pq,
            Element::Code,
            160,
            4,
            32,
            CodeDist,
            |v| ((v + 0.5) * 255.0) as u8,
        ),
    ]
}

// ============================================================================
// THE DRIVER
// ============================================================================

/// What one case did.
enum Outcome {
    Accepted,
    Refused(String),
    /// The property broke. The message is the panic's own.
    Panicked(String),
}

/// A silent panic hook for the length of a run, restored on drop.
///
/// Without it a caught panic still prints its message and its backtrace, and a
/// run that finds one finding prints it once per case that reproduces it.
struct Quiet(Option<PanicHook>);

/// The hook `std::panic` hands back, named so the field above reads.
type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>;

impl Quiet {
    fn new() -> Self {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        Quiet(Some(previous))
    }
}

impl Drop for Quiet {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            std::panic::set_hook(previous);
        }
    }
}

/// Whether every case is announced on stderr before it runs.
///
/// **An allocation failure aborts the process rather than unwinding**, so
/// `catch_unwind` cannot see one and no in-process report can name the case
/// that caused it. `ZEUSDB_FUZZ_TRACE=1` announces each case before driving it,
/// unbuffered, so the last line before an abort is the case. That is how the
/// origin-id finding was minimised.
fn tracing() -> bool {
    std::env::var("ZEUSDB_FUZZ_TRACE").is_ok_and(|v| v != "0")
}

/// Run the reader over `blob` written into `dir`, catching a panic.
///
/// `catch_unwind` rather than a subprocess, because the reader owns no global
/// state a poisoned unwind could leave behind: it opens a file, allocates, and
/// returns. It does not catch an abort; see [`tracing`].
fn drive(dir: &Path, blob: &[u8], entry: &Entry) -> Outcome {
    std::fs::write(dir.join(DUMP_FILENAME), blob).unwrap();
    let expected = entry.expected;
    let element = entry.element;
    let owned = dir.to_path_buf();

    let result = std::panic::catch_unwind(move || match element {
        Element::Raw => read_dump::<f32, CosineDist>(&owned, &expected, CosineDist {})
            .map(|_| ())
            .err(),
        Element::Code => read_dump::<u8, CodeDist>(&owned, &expected, CodeDist)
            .map(|_| ())
            .err(),
    });

    match result {
        Ok(None) => Outcome::Accepted,
        Ok(Some(reason)) => Outcome::Refused(reason),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic carrying no message".to_string());
            Outcome::Panicked(message)
        }
    }
}

/// Drop every op the failure does not need, one at a time.
///
/// A greedy delta rather than a bisection, because a case is at most six ops
/// and the repairs at the end are usually most of what makes it interesting, so
/// there is nothing for a bisection to save.
fn minimise(dir: &Path, entry: &Entry, ops: &[Op]) -> Vec<Op> {
    let mut kept = ops.to_vec();
    let mut at = 0;
    while at < kept.len() {
        let mut trial = kept.clone();
        trial.remove(at);
        let blob = mutate(&entry.blob, &trial);
        if matches!(drive(dir, &blob, entry), Outcome::Panicked(_)) {
            kept = trial;
        } else {
            at += 1;
        }
    }
    kept
}

/// Whether the reader rejected this input without judging what was in it.
///
/// The magic, the version, the two checksums, the end marker, the length floor
/// and the size agreement are integrity checks. They say the bytes are not a
/// whole file, not that the graph they hold is wrong. A run made only of these
/// has rotted into proving the checksums work, which is what the mutator's
/// repairs exist to prevent, so the property test counts them and fails if they
/// take over.
fn refused_on_integrity(reason: &str) -> bool {
    reason.contains("not a ZeusDB graph dump")
        || reason.contains("header is corrupt")
        || reason.contains("contents are corrupt")
        || reason.contains("does not end where it says")
        || reason.contains("the smallest one is")
        || reason.contains("and the file holds")
        || reason.contains("its own fields need")
        || reason.contains("ended early")
        || reason.contains("format version")
        || reason.contains("holds no ZeusDB graph dump")
}

// ============================================================================
// THE TESTS
// ============================================================================

/// **The property.** Every mutation of a valid dump errors or loads, and none
/// panics.
///
/// Failures are collected rather than asserted one at a time, so one run
/// reports every distinct panic it found rather than the first.
#[test]
fn no_mutation_of_a_valid_dump_panics_the_reader() {
    let budget: usize = std::env::var("ZEUSDB_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CASES_PER_ENTRY);

    let trace = tracing();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = Rng(SEED);
    let mut failures: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut accepted = 0usize;
    let mut refused = 0usize;
    let mut judged = 0usize;

    {
        let _quiet = Quiet::new();
        for entry in corpus() {
            // The undamaged dump loads, so a refusal below is the mutation's
            // doing rather than the corpus being wrong.
            if !matches!(drive(dir.path(), &entry.blob, &entry), Outcome::Accepted) {
                failures.push(format!(
                    "\n  the corpus entry {} does not load",
                    entry.label
                ));
                continue;
            }

            for case in 0..budget {
                let ops = draw_case(&mut rng, entry.blob.len());
                if trace {
                    eprintln!("{} case {}: {:?}", entry.label, case, ops);
                }
                let blob = mutate(&entry.blob, &ops);
                match drive(dir.path(), &blob, &entry) {
                    Outcome::Accepted => {
                        accepted += 1;
                        judged += 1;
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
                            let small = minimise(dir.path(), &entry, &ops);
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

    let total = accepted + refused + seen.len();
    println!(
        "fuzz: {} cases, {} loaded, {} refused, {} judged on their contents",
        total, accepted, refused, judged
    );
    assert!(
        failures.is_empty(),
        "the reader panicked on {} distinct inputs:{}",
        failures.len(),
        failures.join("")
    );
    // A run judged only on integrity has rotted into proving the checksums
    // work, which is not what this file is for.
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

/// The repairs reach the parser, measured rather than assumed.
///
/// This is the test that stops the fuzzer rotting into a checksum test. The
/// measure is how often a mutated dump is **accepted whole**, because that is
/// unambiguous: an accepted file passed the magic, both checksums, the end
/// marker, the size agreement, every field check and the topology. A bare
/// mutation cannot be accepted, since any byte it touches breaks one of the two
/// checksums. Measured on this corpus the repaired share is about three in ten
/// and the bare share is none.
///
/// `ZEUSDB_FUZZ_HISTOGRAM=1` prints what the refusals were, which is the only
/// way to see whether the run is still reaching the parsing code or has drifted
/// back into being refused on the header.
#[test]
fn the_repairs_carry_mutations_past_the_checksums() {
    const TRIALS: usize = 1_500;
    let dir = tempfile::tempdir().unwrap();
    let entries = corpus();
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
            let mut ops = vec![draw_op(&mut rng, entry.blob.len())];
            let bare = mutate(&entry.blob, &ops);
            // A draw can land on a value that is already there, on a zero run
            // that is already zero, or on a zero length range. It changed
            // nothing, so it is not a case and counting it would put a no-op in
            // the bare column and flatter the comparison.
            if bare == entry.blob {
                continue;
            }
            trials += 1;
            if matches!(drive(dir.path(), &bare, entry), Outcome::Accepted) {
                without += 1;
            }
            ops.push(Op::RepairFileBytes);
            ops.push(Op::RepairHeader);
            ops.push(Op::RepairPayload);
            let outcome = drive(dir.path(), &mutate(&entry.blob, &ops), entry);
            if matches!(outcome, Outcome::Accepted) {
                with += 1;
            }
            if keep {
                let key = match outcome {
                    Outcome::Accepted => "accepted".to_string(),
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
        "fuzz: accepted whole, {} of {TRIALS} repaired against {} of {TRIALS} bare",
        with, without
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

/// A file under either name an earlier release wrote reaches no parser.
///
/// The legacy names are part of the corpus because a directory saved by 0.6.0
/// or earlier still holds them. Nothing parses them, and the case is that a
/// hostile file under one of those names cannot reach a parser at all, whatever
/// it holds.
#[test]
fn a_file_under_a_legacy_name_reaches_no_parser() {
    let entries = corpus();
    let entry = &entries[0];
    let dir = tempfile::tempdir().unwrap();

    let mut rng = Rng(SEED ^ 0x1eaf);
    for legacy in LEGACY_DUMP_FILENAMES {
        let ops = draw_case(&mut rng, entry.blob.len());
        std::fs::write(dir.path().join(legacy), mutate(&entry.blob, &ops)).unwrap();
    }

    // No `hnsw_index.zdbgraph`, so the two legacy files are all the directory
    // holds and the reader has to say there is no dump rather than read one.
    match read_dump::<f32, CosineDist>(dir.path(), &entry.expected, CosineDist {}) {
        Ok(_) => panic!("a directory holding only legacy dumps loaded"),
        Err(reason) => assert!(reason.contains("holds no ZeusDB graph dump"), "{}", reason),
    }

    // And the current name beside them is the one that is read.
    std::fs::write(dir.path().join(DUMP_FILENAME), &entry.blob).unwrap();
    assert!(read_dump::<f32, CosineDist>(dir.path(), &entry.expected, CosineDist {}).is_ok());
}

/// The mutator reaches the cases the hand written enumeration reached.
///
/// A fuzzer that replaces an enumeration has to produce what it replaced. Each
/// entry here is one of the eleven cases from `every_damaged_input_is_an_error`
/// plus the two `a_header_claiming_a_billion_points_allocates_nothing` needed a
/// hand repaired header for, expressed as this mutator's own ops.
#[test]
fn the_mutator_expresses_the_hand_written_cases() {
    let entries = corpus();
    let entry = &entries[0];
    let dir = tempfile::tempdir().unwrap();
    let len = entry.blob.len();

    let case = |ops: Vec<Op>, needle: &str| {
        let blob = mutate(&entry.blob, &ops);
        match drive(dir.path(), &blob, entry) {
            Outcome::Refused(reason) => {
                assert!(reason.contains(needle), "{:?} gave {}", ops, reason)
            }
            Outcome::Accepted => panic!("{:?} was accepted", ops),
            Outcome::Panicked(message) => panic!("{:?} panicked: {}", ops, message),
        }
    };

    case(vec![Op::Truncate { len: 0 }], "smallest");
    case(vec![Op::Truncate { len: HEADER }], "smallest");
    case(vec![Op::Truncate { len: HEADER + 16 }], "the file holds");
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

/// Every corpus entry survives the round trip it was built from.
///
/// Cheap, and it means a failure above is the reader's rather than one of the
/// six graphs this file builds.
#[test]
fn every_corpus_entry_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    for entry in corpus() {
        std::fs::write(dir.path().join(DUMP_FILENAME), &entry.blob).unwrap();
        let counts = match entry.element {
            Element::Raw => {
                read_dump::<f32, CosineDist>(dir.path(), &entry.expected, CosineDist {})
                    .map(|(graph, store)| (graph.nb_points(), store.len()))
                    .unwrap_or_else(|e| panic!("{} did not load: {}", entry.label, e))
            }
            Element::Code => read_dump::<u8, CodeDist>(dir.path(), &entry.expected, CodeDist)
                .map(|(graph, store)| (graph.nb_points(), store.len()))
                .unwrap_or_else(|e| panic!("{} did not load: {}", entry.label, e)),
        };
        assert_eq!(counts.0, counts.1, "{} lost vectors", entry.label);
    }
}
