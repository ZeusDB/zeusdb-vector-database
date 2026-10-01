//! # ZeusDB Vector Database - Persistence Module
//!
//! This module handles all save/load operations for ZeusDB vector indexes.
//! It implements a directory-based persistence format with hybrid JSON/Binary storage.
//!
//! ## File Format:
//! ```text
//! my_index.zdb/
//! ├── manifest.json           # Index metadata and file list
//! ├── config.json             # Index configuration
//! ├── mappings.bin            # ID mappings (binary)
//! ├── metadata.json           # Vector metadata (JSON)
//! ├── vectors.bin             # Raw vectors (storage mode dependent)
//! ├── quantization.json       # quantization configuration and training state (if enabled)
//! ├── pq_codes.bin            # Quantized codes (if PQ enabled)
//! ├── pq_centroids.bin        # PQ centroids (if trained)
//! ├── int8_scales.zdbint8     # the scales of a scalar quantized index, one a dimension (if trained)
//! ├── int8_rows.zdbint8       # every record's scalar row (if trained and not empty)
//! ├── hnsw_index.zdbgraph     # HNSW graph topology and the point each node holds
//! └── spaces/                 # one directory per sparse space, where one is declared
//!     └── <name>/
//!         ├── postings.zdbsparse  # every live record's term ids and weights
//!         └── terms.zdbdict       # the term dictionary, where the space takes text
//!
//! my_index.zdb.zdbwal         # the journal, where the collection held one
//! ```
//!
//! ## The format version
//!
//! A directory holding a dense space alone is written at `1.1.0`, with the
//! flat names above and nothing under `spaces/`, and it is byte for byte
//! what 0.9.0 wrote. A directory holding a sparse space is written at
//! `2.0.0`. The bump is a major because a sparse space is not additive: a
//! release reading `1.x` alone would find every file it knows, read the
//! seven names it knows, and open the collection without its sparse space
//! and without a word, which is the failure this format has already
//! suffered once. At `2.0.0` such a release refuses at the version check
//! with a message naming the newer release.
//!
//! A directory a journaled collection saved is written at `3.0.0`, for the
//! same reason. Its manifest names the journal beside it, and a release
//! reading `1.x` and `2.x` alone would find every file the manifest names,
//! ignore the `journal` field it does not know, and open an index missing
//! every acknowledged mutation since the checkpoint, again without a word.
//! This build reads all three majors, and a directory saved by a collection
//! holding no journal keeps `1.1.0` or `2.0.0` and opens everywhere it does
//! today.
//!
//! A directory whose dense space is declared with scalar quantization is
//! written at the next minor of whichever major its other contents put it
//! in, being `1.2.0`, `2.1.0` or `3.1.0`. A minor rather than a major
//! because an older reader never opens such a directory silently: every one
//! carries a `quantization.json` naming `type: int8` and none of the product
//! quantized fields, and a reader that knows the product quantized fields
//! alone refuses it on the first missing one. The minor records that the
//! directory holds something newer without shutting out a reader that can
//! cope. A directory declared without scalar quantization keeps the version
//! it has, byte for byte.
//!
//! ## The journal
//!
//! A journaled collection records every mutation to `<name>.zdbwal`, a
//! sibling of the directory rather than a file inside it, and a save is the
//! checkpoint that journal replays onto. `manifest.json` names the file, the
//! collection id both carry and the sequence the checkpoint holds, so the two
//! are paired by content. A load replays every record above that sequence
//! before it hands the collection back. See `crate::journal` for where the
//! file lives and why, and for what is refused.
//!
//! `config.json` declares the sparse space under `spaces`, by value: the
//! space's name, its unlink policy, its lazy threshold, its weighting with
//! the weighting's parameters, and its tokenizer where it takes text. The
//! declaration is validated on load under the rules `Declaration` applies.
//! A tokenizer the engine cannot write down is recorded as `external`, and
//! a directory recording one opens only through `Collection::load_with`
//! with the same implementation handed to it.
//!
//! ## How a save lands
//!
//! Every artefact goes into `<name>.zdbtmp` beside the target and the whole
//! directory is renamed into place at the end, so a reader sees the previous
//! index or this one and never a mixture. Replacing an existing directory needs
//! two renames rather than one, with `<name>.zdbold` holding the previous index
//! between them. See `StagingDir` for what that means on each platform and what
//! a killed process leaves behind.
//!
//! `manifest.json` records a length and a digest for every artefact it names
//! and the loader checks both before anything parses them. See
//! `ArtefactDigest`.
//!
//! The graph file is ZeusDB's own format, written and read by `graph::dump`,
//! and the loader restores the graph from it rather than rebuilding it by
//! re-inserting every record. See `Collection::restore_graph_from_dump`.
//!
//! This module and `collection::persist` call each other: `save` and `load`
//! on the collection reach `save_index`, `save_manifest`, `StagingDir` and
//! `load_index` here, and `load_index` builds a collection and restores it
//! through the setters `persist.rs` declares. That is a module cycle inside
//! one crate, which cargo tolerates, and it is the shape the two had in the
//! binding.
//!
//! It replaces the two files the vendored graph crate wrote,
//! `hnsw_index.hnsw.graph` and `hnsw_index.hnsw.data`. A directory saved by
//! 0.6.0 or earlier still holds those two, and opening it rebuilds the graph
//! once and writes the new file on the next save. Nothing reads the old format.

use crate::collection::{
    validate_index_parameters, validate_space_supports_quantization, Collection,
    QuantizationConfig, QuantizationScheme, SparseDeclaration, StorageMode, DEFAULT_SPACE,
};
use crate::journal::{JournalPolicy, Recovery};
use crate::RerankCalibration;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};
use zeusdb_vector_core::{
    checksum_of, frame, frame_begin, frame_finish, unframe, validate_indexed_fields,
    ArtefactRecord, Bounds, Error, FrameEncoding, FrameKind, Int8Codec, Inventory, Persist,
    RecordFields, Restore, SpaceName, VectorIndex, DUMP_FILENAME as GRAPH_DUMP_FILENAME,
    FRAME_OVERHEAD_BYTES, LEGACY_DUMP_FILENAMES, PQ,
};
use zeusdb_vector_sparse::{PostingsIndex, SparseConfig};
use zeusdb_vector_text::{SimpleTokenizer, TermDictionary, Tokenizer, TokenizerConfig};

/// The target every record this file emits carries. It is the module path
/// this file had in the binding, so a filter directive naming it still
/// matches. See the crate root.
const LOG_TARGET: &str = "zeusdb_vector_database::persistence";

// ============================================================================
// FORMAT VERSION
// ============================================================================

/// Version written into manifest.json by a save of a collection holding a
/// dense space alone.
///
/// 1.1.0 rather than 1.0.0 because config.json carries an index level
/// `metadata` map, which is additive on both sides. A directory written at
/// this version opens on every release that reads 1.x, and one written by
/// such a release opens here.
const DENSE_FORMAT_VERSION: &str = "1.1.0";

/// Version written into manifest.json by a save of a collection holding a
/// sparse space. See the module documentation for why it is a major.
const SPACES_FORMAT_VERSION: &str = "2.0.0";

/// Version written into manifest.json by a checkpoint of a collection
/// holding a journal.
///
/// A major, by the same rule the sparse space's was. A build that reads 1.x
/// and 2.x alone opens a journaled directory without a word: it finds every
/// file the manifest names, ignores the `journal` field it does not know,
/// and hands back an index missing every acknowledged mutation since the
/// checkpoint. Nothing in the directory would tell the caller. A major stops
/// that build at the version and says why.
///
/// A directory saved by a collection holding no journal keeps 1.1.0 or
/// 2.0.0 and opens everywhere it does today, because nothing about it
/// changed.
const JOURNAL_FORMAT_VERSION: &str = "3.0.0";

/// The minor each major moves to when the dense space is declared with
/// scalar quantization. See the module documentation for why it is a minor.
const DENSE_INT8_FORMAT_VERSION: &str = "1.2.0";
const SPACES_INT8_FORMAT_VERSION: &str = "2.1.0";
const JOURNAL_INT8_FORMAT_VERSION: &str = "3.1.0";

/// The two artefacts of a trained scalar quantized index.
const INT8_SCALES_FILENAME: &str = "int8_scales.zdbint8";
const INT8_ROWS_FILENAME: &str = "int8_rows.zdbint8";
const INT8_SCALES_CONTENTS: &str =
    "the scales of a scalar quantized index, one a dimension, which every stored row decodes through";
const INT8_ROWS_CONTENTS: &str = "the scalar row of every record";

/// The majors this build reads.
///
/// A minor bump is additive by construction, so any 1.x, any 2.x and any
/// 3.x is read. A different major means the layout changed in a way this
/// build cannot reason about, and guessing at it would be the silent
/// truncation this format has already suffered once.
const SUPPORTED_FORMAT_MAJORS: [u32; 3] = [1, 2, 3];

/// The majors this build reads, as a refusal spells them.
const SUPPORTED_FORMAT_LABEL: &str = "1.x, 2.x and 3.x";

/// The first major whose manifest may name a journal.
const JOURNAL_FORMAT_MAJOR: u32 = 3;

// ============================================================================
// THE BINCODE CLAIM BUDGET
// ============================================================================

// `dim` was bounded here first, at 65,536, on the reasoning that bounding it
// in `validate_index_parameters` would change `create()`'s documented
// contract. `create(dim=2**40)` aborted the process for the same reason a
// config naming it did, so the bound moved to `validate_index_parameters` as
// `MAX_DIM` and the README row changed with it. `load_config` calls that
// function, so the loader still refuses every width it refused, at the same
// number, with config.json named in the message.

/// The claim budget one wire byte earns, and why it is 64
///
/// `bincode::config::standard()` carries no byte limit, so `claim_container_read`
/// compiles to nothing and every container length in a decoded file goes
/// straight to the allocator. A 14 byte `mappings.bin` declaring 2^40 entries
/// asked for tens of terabytes and **aborted the process**. Measured on this
/// build, ten of the twelve container lengths across `mappings.bin`,
/// `vectors.bin`, `pq_codes.bin` and `pq_centroids.bin` abort at 2^40 and none
/// of them is bounded by anything the file has earned.
///
/// The bound has to come from the file's own length, which is the line
/// `parse_dump` draws: a header's fields are checked against the bytes the file
/// really holds, and after that every allocation is bounded by a file that
/// really is that long. Here the file is already in memory, so its length is
/// known outright.
///
/// bincode counts claimed bytes, being `len * size_of::<T>()` per container,
/// and unclaims each element as it decodes. The widest ratio of claimed bytes
/// to wire bytes across the four artefacts this build decodes is
/// `pq_centroids.bin`, a `Vec<Vec<Vec<f32>>>` whose outer container claims 24
/// bytes for an entry that costs one byte on the wire. The three maps claim 48
/// bytes for an entry that costs two. 64 is the next power of two above both,
/// so it admits every file this build writes with margin.
///
/// What the budget bounds is the claims. What is allocated is held to the
/// bytes the file has left, see `HeldDecode`, so a length the budget admits
/// and the file cannot carry is not allocated.
const CLAIM_PER_WIRE_BYTE: usize = 64;

/// The claim a decode makes before it has read anything
///
/// bincode charges eight bytes for reading a length word, whatever the varint
/// that carries it occupies on the wire, and every artefact here begins with
/// one. A budget below that admits no decode at all, so a zero byte artefact
/// would be refused as a length it never declared rather than as the file
/// ending where a length was expected. The floor is that charge, and a file of
/// one byte already earns eight times it.
const LEADING_LENGTH_CLAIM: usize = std::mem::size_of::<u64>();

/// Decode a bincode artefact under a budget the file's own length sets
///
/// bincode takes its limit as a const generic, so the derived budget picks one
/// of four rungs rather than being passed. The rungs are a factor of 256
/// apart, so a file one byte above a rung's top earns 256 times the budget its
/// length gives, and every container length under that rung is sized from the
/// length the file declares. A 4 MiB file earned a 64 GiB rung, and a header
/// on one killed the process.
///
/// `claim_rung_excess` closes that gap. The rung is still what the decoder is
/// built with, because the const generic needs a constant, but whatever it
/// carries above the derived budget is claimed before anything is decoded, so
/// what a container may still claim is the budget and nothing more.
///
/// A claim is not an allocation. The decode is `HeldArtefact`'s, which makes
/// every claim bincode's own decode makes and allocates nothing ahead of the
/// bytes that carry it, so a length the budget admits and the file cannot
/// carry is refused before it is allocated. `known` is a count the loader
/// already holds for the artefact, and it sizes what is reserved and nothing
/// else.
///
/// A file long enough to need more than the top rung is one `fs::read` could
/// not have returned, so the top rung is the last arm rather than a special
/// case.
fn decode_bounded<T>(data: &[u8], file: &str, known: T::Known) -> Result<T, Error>
where
    T: HeldArtefact,
{
    decode_at(data, file, claim_budget(data.len()), known)
}

/// The decode above under a budget given outright, so a test can measure how
/// much of it a well formed artefact really claims.
fn decode_at<T>(data: &[u8], file: &str, budget: usize, known: T::Known) -> Result<T, Error>
where
    T: HeldArtefact,
{
    use bincode::config::standard;

    let decoded = if budget <= 1 << 20 {
        decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 20 }>(), budget, known)
    } else if budget <= 1 << 28 {
        decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 28 }>(), budget, known)
    } else if budget <= 1 << 36 {
        decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 36 }>(), budget, known)
    } else {
        decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 44 }>(), budget, known)
    };

    match decoded {
        Ok(value) => Ok(value),
        Err(bincode::error::DecodeError::LimitExceeded) => Err(Error::DecodeLengthExceeded {
            file: file.to_string(),
            bytes: data.len(),
        }),
        Err(e) => Err(Error::DecodeFailed {
            file: file.to_string(),
            error: e.to_string(),
        }),
    }
}

/// `T` from `data` under `budget`, whatever the rung carries above it claimed
/// first
///
/// The steps are `bincode::decode_from_slice`'s own, being a decoder over the
/// slice and one decode, so a file this refuses is refused with the message
/// that function gave. The reader is `TrackedSlice`, which reads as bincode's
/// slice reader reads and says how many bytes it has left, and the decode is
/// `HeldArtefact`'s, which claims as bincode's `Decode` claims.
fn decode_claimed<T, C>(
    data: &[u8],
    config: C,
    budget: usize,
    known: T::Known,
) -> Result<T, bincode::error::DecodeError>
where
    T: HeldArtefact,
    C: bincode::config::Config,
{
    let mut decoder = bincode::de::DecoderImpl::new(TrackedSlice::new(data), config, ());
    claim_rung_excess(&mut decoder, budget)?;
    T::decode_artefact(&mut decoder, known)
}

/// Claim whatever the decoder's rung carries above `budget`
///
/// bincode counts claimed bytes against the const limit and unclaims each
/// element as it decodes it, starting the count at zero. Starting it at the
/// rung's excess instead leaves exactly `budget` to claim, which is what a
/// limit given at run time would have left. The limit is read back from the
/// configuration rather than passed in, so the arm that built the decoder and
/// the claim that tightens it cannot name different rungs.
///
/// bincode's limit is a const generic and both its `Config` and its `Decoder`
/// are sealed, so there is no configuration and no decoder this crate can
/// write to carry a budget that is not a constant. This is the claim the
/// sealed traits still admit.
///
/// A file whose budget is above the top rung has nothing to claim and keeps
/// the rung, which needs an artefact of 256 GiB.
fn claim_rung_excess<D: bincode::de::Decoder>(
    decoder: &mut D,
    budget: usize,
) -> Result<(), bincode::error::DecodeError> {
    use bincode::config::Config;

    match decoder.config().limit() {
        Some(rung) => decoder.claim_bytes_read(rung.saturating_sub(budget)),
        None => Ok(()),
    }
}

// ============================================================================
// DECODING WITHOUT RESERVING AHEAD OF THE BYTES
// ============================================================================

/// bincode's slice reader, with the bytes it has left
///
/// `read`, `peek_read` and `consume` are `SliceReader`'s, so a read fails where
/// `SliceReader`'s would and names the same `additional`. `SliceReader` keeps
/// its slice `pub(crate)`, and `Reader` is the trait bincode's decoder takes,
/// so this is the reader a decode can ask how many bytes remain. A copy reads
/// the same bytes again from where the original stood.
#[derive(Clone, Copy)]
struct TrackedSlice<'a> {
    slice: &'a [u8],
}

impl<'a> TrackedSlice<'a> {
    fn new(slice: &'a [u8]) -> Self {
        TrackedSlice { slice }
    }

    fn bytes_left(&self) -> usize {
        self.slice.len()
    }

    /// The bytes left, as the slice they are.
    fn bytes(&self) -> &'a [u8] {
        self.slice
    }
}

impl bincode::de::read::Reader for TrackedSlice<'_> {
    #[inline(always)]
    fn read(&mut self, bytes: &mut [u8]) -> Result<(), bincode::error::DecodeError> {
        if bytes.len() > self.slice.len() {
            return Err(bincode::error::DecodeError::UnexpectedEnd {
                additional: bytes.len() - self.slice.len(),
            });
        }
        let (read_slice, remaining) = self.slice.split_at(bytes.len());
        bytes.copy_from_slice(read_slice);
        self.slice = remaining;
        Ok(())
    }

    #[inline]
    fn peek_read(&mut self, n: usize) -> Option<&[u8]> {
        self.slice.get(..n)
    }

    #[inline]
    fn consume(&mut self, n: usize) {
        self.slice = self.slice.get(n..).unwrap_or_default();
    }
}

/// A container's length as bincode reads one, a `u64` held to `usize`.
fn decode_len<D: bincode::de::Decoder>(
    decoder: &mut D,
) -> Result<usize, bincode::error::DecodeError> {
    use bincode::Decode;

    let claimed = u64::decode(decoder)?;
    usize::try_from(claimed).map_err(|_| bincode::error::DecodeError::OutsideUsizeRange(claimed))
}

/// What a container of `len` entries is reserved at, `left` bytes after its
/// length
///
/// The least of the count it declares, the entries `left` bytes could carry at
/// `least` bytes an entry, and `known`, a count the loader already holds for
/// it. With no such count nothing is reserved ahead of the entries.
fn reserved(len: usize, left: usize, least: usize, known: Option<usize>) -> usize {
    known.map_or(0, |known| len.min(left / least).min(known))
}

/// A varint as bincode writes one, being the value and the bytes it occupied,
/// or nothing where the bytes do not carry one
///
/// Read by hand for `count_distinct_keys`, which refuses nothing and claims
/// nothing, so bincode's decoder is not wanted there.
fn varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let word = |n: usize| bytes.get(1..1 + n);
    match *bytes.first()? {
        n @ 0..=250 => Some((u64::from(n), 1)),
        251 => Some((u64::from(u16::from_le_bytes(word(2)?.try_into().ok()?)), 3)),
        252 => Some((u64::from(u32::from_le_bytes(word(4)?.try_into().ok()?)), 5)),
        253 => Some((u64::from_le_bytes(word(8)?.try_into().ok()?), 9)),
        _ => None,
    }
}

/// A hash kept as it is, for a set whose entries are hashes already
#[derive(Default)]
struct Hashed(u64);

impl std::hash::Hasher for Hashed {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }

    fn write_u64(&mut self, hash: u64) {
        self.0 = hash;
    }
}

/// The distinct keys a map of strings carries in `bytes`, counted before the
/// map is built
///
/// The forward map of `mappings.bin` is the one container that no count
/// precedes in the file and none the loader holds from elsewhere. Reserved at
/// the count it declares, a file declaring a count its bytes do not carry
/// reserves that count, and grown from nothing as its entries decode, its
/// last doubling holds the old table beside the new one. The keys the bytes
/// carry are what the map will hold and nothing a declared count can inflate.
/// A key the bytes hold twice counts once, a key whose entry the bytes do not
/// hold whole does not count, and the count stops where the bytes stop, where
/// a length runs past them, or where a varint does not decode.
///
/// Each key is hashed once, and the set keeps the hash under `Hashed`, so it
/// costs eight bytes an entry and hashes nothing twice. Two keys that share a
/// hash count as one and reserve the map one entry short, which it grows for
/// as it grows for any count that is short. Nothing here refuses, since the
/// decode that follows refuses with bincode's words wherever it stops, and
/// nothing here claims, so the budget is untouched.
fn count_distinct_keys(bytes: &[u8]) -> usize {
    use std::collections::HashSet;
    use std::hash::{BuildHasher, BuildHasherDefault, RandomState};

    let Some((declared, width)) = varint(bytes) else {
        return 0;
    };
    let hasher = RandomState::new();
    let mut keys: HashSet<u64, BuildHasherDefault<Hashed>> = HashSet::default();
    let mut at = width;
    for _ in 0..declared {
        let Some((key_len, width)) = varint(&bytes[at..]) else {
            break;
        };
        let Some(end) = usize::try_from(key_len)
            .ok()
            .and_then(|key_len| (at + width).checked_add(key_len))
        else {
            break;
        };
        let Some(key) = bytes.get(at + width..end) else {
            break;
        };
        let Some((_, width)) = varint(&bytes[end..]) else {
            break;
        };
        keys.insert(hasher.hash_one(key));
        at = end + width;
    }
    keys.len()
}

/// A value decoded in bincode's steps, with nothing allocated ahead of the
/// bytes that carry it
///
/// Every claim bincode's own `Decode` makes is made here, in the same order
/// and against the same counter, so the claim budget refuses exactly the files
/// it refused. What differs is the allocation after a claim. bincode sizes a
/// container from the length the file declares as soon as the claim passes.
/// Here a string or a byte vector is allocated only once its bytes are there,
/// a vector of floats is reserved at no more floats than its bytes carry, a
/// map grows past `reserved` as its entries decode, and a vector of containers
/// past `reserved` is reserved exactly once its entries have decoded, see
/// `decode_rest`. A value that decodes is the value bincode returns, and a
/// refusal carries bincode's words.
trait HeldDecode: Sized {
    /// The fewest wire bytes one of these occupies.
    const LEAST_WIRE_BYTES: usize;

    /// What the loader knows of the values inside one of these.
    type Inner: Copy;

    fn decode_held<'a, D>(
        decoder: &mut D,
        inner: Self::Inner,
    ) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>;
}

impl HeldDecode for f32 {
    const LEAST_WIRE_BYTES: usize = 4;
    type Inner = ();

    fn decode_held<'a, D>(decoder: &mut D, _: ()) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        use bincode::Decode;

        f32::decode(decoder)
    }
}

impl HeldDecode for usize {
    const LEAST_WIRE_BYTES: usize = 1;
    type Inner = ();

    fn decode_held<'a, D>(decoder: &mut D, _: ()) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        use bincode::Decode;

        usize::decode(decoder)
    }
}

impl HeldDecode for Vec<u8> {
    const LEAST_WIRE_BYTES: usize = 1;
    type Inner = ();

    /// bincode reads a byte vector into a buffer of the declared length, so a
    /// length the bytes left cannot fill is refused here with the error that
    /// read returns, before any buffer exists.
    fn decode_held<'a, D>(decoder: &mut D, _: ()) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        use bincode::de::read::Reader;

        let len = decode_len(decoder)?;
        decoder.claim_container_read::<u8>(len)?;
        let left = decoder.reader().bytes_left();
        if len > left {
            return Err(bincode::error::DecodeError::UnexpectedEnd {
                additional: len - left,
            });
        }
        let mut bytes = vec![0u8; len];
        decoder.reader().read(&mut bytes)?;
        Ok(bytes)
    }
}

impl HeldDecode for String {
    const LEAST_WIRE_BYTES: usize = 1;
    type Inner = ();

    fn decode_held<'a, D>(decoder: &mut D, _: ()) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        let bytes = Vec::<u8>::decode_held(decoder, ())?;
        String::from_utf8(bytes).map_err(|e| bincode::error::DecodeError::Utf8 {
            inner: e.utf8_error(),
        })
    }
}

impl HeldDecode for Vec<f32> {
    const LEAST_WIRE_BYTES: usize = 1;
    type Inner = ();

    /// A float's least wire cost is its whole wire cost, so the count of floats
    /// the bytes left can carry is an exact bound and needs no known count.
    fn decode_held<'a, D>(decoder: &mut D, _: ()) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        decode_seq::<f32, D>(decoder, Some(usize::MAX), ())
    }
}

impl HeldDecode for Vec<Vec<f32>> {
    const LEAST_WIRE_BYTES: usize = 1;
    /// The centroids a subvector holds, where the loader knows it.
    type Inner = Option<usize>;

    fn decode_held<'a, D>(
        decoder: &mut D,
        centroids: Option<usize>,
    ) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        decode_seq::<Vec<f32>, D>(decoder, centroids, ())
    }
}

/// `Vec<T>` in bincode's steps, reserved at `reserved`, the entries past that
/// decoded by `decode_rest`.
fn decode_seq<'a, T, D>(
    decoder: &mut D,
    known: Option<usize>,
    inner: T::Inner,
) -> Result<Vec<T>, bincode::error::DecodeError>
where
    T: HeldDecode,
    D: bincode::de::Decoder<R = TrackedSlice<'a>>,
{
    let len = decode_len(decoder)?;
    decoder.claim_container_read::<T>(len)?;
    let left = decoder.reader().bytes_left();
    let mut decoded = Vec::with_capacity(reserved(len, left, T::LEAST_WIRE_BYTES, known));
    while decoded.len() < len {
        if decoded.len() == decoded.capacity() {
            return decode_rest(decoder, decoded, len, inner);
        }
        decoder.unclaim_bytes_read(std::mem::size_of::<T>());
        decoded.push(T::decode_held(decoder, inner)?);
    }
    Ok(decoded)
}

/// The entries of a vector past what it reserved
///
/// A vector grown as its entries decode copies them at every doubling and
/// holds the old buffer beside one twice its size. Here the rest are decoded
/// once on the claimed decoder, which refuses a file where bincode's decode
/// refuses it, and are not kept. The vector is then reserved at exactly that
/// many and the same bytes decoded again on a decoder of their own. That
/// decoder starts its count at nothing under the same rung and claims only
/// what the first pass claimed for these entries, so it refuses none of them.
fn decode_rest<'a, T, D>(
    decoder: &mut D,
    mut decoded: Vec<T>,
    len: usize,
    inner: T::Inner,
) -> Result<Vec<T>, bincode::error::DecodeError>
where
    T: HeldDecode,
    D: bincode::de::Decoder<R = TrackedSlice<'a>>,
{
    let from = *decoder.reader();
    let done = decoded.len();
    for _ in done..len {
        decoder.unclaim_bytes_read(std::mem::size_of::<T>());
        T::decode_held(decoder, inner)?;
    }
    decoded.reserve_exact(len - done);
    let mut again = bincode::de::DecoderImpl::new(from, *decoder.config(), ());
    for _ in done..len {
        decoded.push(T::decode_held(&mut again, inner)?);
    }
    Ok(decoded)
}

/// `HashMap<K, V>` in bincode's steps, reserved at `reserved` and grown past
/// it as its entries decode, the last copy of a key kept as bincode keeps it.
fn decode_map<'a, K, V, D>(
    decoder: &mut D,
    known: Option<usize>,
) -> Result<HashMap<K, V>, bincode::error::DecodeError>
where
    K: HeldDecode<Inner = ()> + Eq + std::hash::Hash,
    V: HeldDecode<Inner = ()>,
    D: bincode::de::Decoder<R = TrackedSlice<'a>>,
{
    let len = decode_len(decoder)?;
    decoder.claim_container_read::<(K, V)>(len)?;
    let least = K::LEAST_WIRE_BYTES + V::LEAST_WIRE_BYTES;
    let left = decoder.reader().bytes_left();
    let mut decoded = HashMap::with_capacity(reserved(len, left, least, known));
    for _ in 0..len {
        decoder.unclaim_bytes_read(std::mem::size_of::<(K, V)>());
        let key = K::decode_held(decoder, ())?;
        let value = V::decode_held(decoder, ())?;
        decoded.insert(key, value);
    }
    Ok(decoded)
}

/// An artefact `decode_bounded` reads, with the count the loader holds for it
trait HeldArtefact: Sized {
    /// What the loader already knows of the artefact's size. It sizes what is
    /// reserved and changes nothing a caller observes.
    type Known: Copy + Default;

    fn decode_artefact<'a, D>(
        decoder: &mut D,
        known: Self::Known,
    ) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>;
}

impl HeldArtefact for IdMappings {
    type Known = ();

    /// The forward map first, reserved at the distinct keys its bytes carry,
    /// then the reverse map, reserved at no more than the forward map holds.
    fn decode_artefact<'a, D>(decoder: &mut D, _: ()) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        let distinct = count_distinct_keys(decoder.reader().bytes());
        let id_map = decode_map(decoder, Some(distinct))?;
        let rev_map = decode_map(decoder, Some(id_map.len()))?;
        Ok(IdMappings { id_map, rev_map })
    }
}

impl HeldArtefact for HashMap<String, Vec<f32>> {
    /// The record count the mappings hold.
    type Known = usize;

    fn decode_artefact<'a, D>(
        decoder: &mut D,
        records: usize,
    ) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        decode_map(decoder, Some(records))
    }
}

impl HeldArtefact for HashMap<String, Vec<u8>> {
    /// The record count the mappings hold.
    type Known = usize;

    fn decode_artefact<'a, D>(
        decoder: &mut D,
        records: usize,
    ) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        decode_map(decoder, Some(records))
    }
}

impl HeldArtefact for Centroids {
    /// The subvector count and the centroids a subvector, from the
    /// quantization.json fields `validate_quantization_fields` has held.
    type Known = (usize, usize);

    fn decode_artefact<'a, D>(
        decoder: &mut D,
        (subvectors, centroids): (usize, usize),
    ) -> Result<Self, bincode::error::DecodeError>
    where
        D: bincode::de::Decoder<R = TrackedSlice<'a>>,
    {
        decode_seq::<Vec<Vec<f32>>, D>(decoder, Some(subvectors), Some(centroids))
    }
}

// ============================================================================
// AN ATOMIC SAVE
// ============================================================================

/// Suffix of the directory a save builds before it is moved into place.
const STAGING_SUFFIX: &str = ".zdbtmp";

/// Suffix the directory being replaced is moved aside under.
const REPLACED_SUFFIX: &str = ".zdbold";

/// A sibling of `target` carrying `suffix`, so both live on the target's volume
///
/// A rename is only cheap, and only atomic, within one volume. Staging under
/// the system temporary directory would put the new index on whichever volume
/// that is, and the move into place would then be a copy of every byte.
fn sibling(target: &Path, suffix: &str) -> Result<PathBuf, Error> {
    let name = target.file_name().ok_or_else(|| Error::TargetHasNoName {
        target: target.to_path_buf(),
    })?;
    let mut name = name.to_os_string();
    name.push(suffix);
    Ok(target.parent().unwrap_or_else(|| Path::new("")).join(name))
}

/// The directory a save builds, and the move that puts it in place
///
/// # What this buys
///
/// Every artefact used to be written straight into the target directory, one
/// `fs::write` at a time. A save interrupted part way left a directory holding
/// some of the new index and some of the old, and a save over an existing
/// directory replaced files one at a time and removed none, so a raw index
/// saved over a quantized one left `quantization.json`, `pq_centroids.bin` and
/// `pq_codes.bin` behind for ever. Only `manifest_names` kept those three from
/// being read back as part of the new index.
///
/// Here the save builds a directory from nothing and moves it in, so a stale
/// artefact cannot survive and a reader sees one whole index or the other.
///
/// # What "moves it into place" means
///
/// **It is one rename where the target does not exist, and two where it does.**
/// Neither Windows nor POSIX can rename a directory over an existing non-empty
/// directory. `rename(2)` requires the destination to be an empty directory and
/// `MoveFileExW` refuses `MOVEFILE_REPLACE_EXISTING` for directories outright,
/// so `fs::rename` fails on both platforms and there is no call in the standard
/// library that swaps two directories in one step. Linux has
/// `renameat2(RENAME_EXCHANGE)`, which std does not expose and which Windows
/// has no counterpart for.
///
/// So a save over an existing directory does this:
///
/// 1. rename the target aside to `<name>.zdbold`
/// 2. rename the staging directory to the target
/// 3. remove `<name>.zdbold`
///
/// Steps 1 and 2 are each atomic on both platforms. Between them the target
/// does not exist, which is a window of two filesystem calls with no I/O
/// between them. **A reader in that window sees no directory rather than a
/// partial one**, which is the property that matters, and a process killed in
/// it leaves the whole previous index at `<name>.zdbold`. `recover` puts that
/// back on the next save. A save to a path that holds nothing yet is step 2
/// alone, which is atomic outright.
///
/// If step 2 fails the target is renamed back from `<name>.zdbold`, so a failed
/// save leaves the previous directory where it was.
///
/// # What a killed process leaves
///
/// A leftover `<name>.zdbtmp` from a save that died before the move, and a
/// leftover `<name>.zdbold` from one that died inside the window. `recover`
/// deals with both at the start of the next save, and neither is inside the
/// index directory, so a load reads neither.
///
/// Dropping this without committing removes the staging directory, so a save
/// that fails part way cleans up after itself inside the process that started
/// it.
pub(crate) struct StagingDir {
    target: PathBuf,
    staging: PathBuf,
    replaced: PathBuf,
    committed: bool,
}

impl StagingDir {
    /// Clear what an earlier save left behind and open an empty staging
    /// directory
    pub(crate) fn open(target: &Path) -> Result<Self, Error> {
        let staging = sibling(target, STAGING_SUFFIX)?;
        let replaced = sibling(target, REPLACED_SUFFIX)?;

        Self::recover(target, &staging, &replaced)?;

        fs::create_dir_all(&staging).map_err(|e| Error::StagingCreateFailed {
            staging: staging.clone(),
            error: e.to_string(),
        })?;

        Ok(StagingDir {
            target: target.to_path_buf(),
            staging,
            replaced,
            committed: false,
        })
    }

    /// Put right whatever a killed save left behind
    ///
    /// `<name>.zdbold` present with no target is the one case that holds data:
    /// the previous save died between the two renames and that directory is the
    /// only copy of the index. `restore_replaced` renames it back, and a load
    /// now does the same before it opens a directory.
    ///
    /// `<name>.zdbold` still present after that is the previous index after a
    /// save that finished, so it is removed. Only a save removes it, because
    /// only a save knows the target beside it is the one it wrote.
    fn recover(target: &Path, staging: &Path, replaced: &Path) -> Result<(), Error> {
        if !restore_replaced(target)? && replaced.exists() {
            remove_tree(replaced, "the previous index a finished save left aside")?;
        }
        if staging.exists() {
            remove_tree(
                staging,
                "a staging directory an interrupted save left behind",
            )?;
        }
        Ok(())
    }

    /// Where the save writes
    pub(crate) fn path(&self) -> &Path {
        &self.staging
    }

    /// Move the staged directory into place
    pub(crate) fn commit(mut self) -> Result<(), Error> {
        sync_directory(&self.staging);

        if self.target.exists() {
            fs::rename(&self.target, &self.replaced).map_err(|e| Error::MoveAsideFailed {
                target: self.target.clone(),
                error: e.to_string(),
            })?;

            zeusdb_vector_core::kill_at(zeusdb_vector_core::KillPoint::SaveBetweenRenames);

            if let Err(e) = fs::rename(&self.staging, &self.target) {
                // The target is empty at this point, so putting the previous
                // index back is the same rename in reverse.
                let restored = fs::rename(&self.replaced, &self.target).is_ok();
                self.committed = true;
                return Err(Error::MoveIntoPlaceFailedAfterAside {
                    target: self.target.clone(),
                    error: e.to_string(),
                    restored,
                });
            }

            remove_tree(&self.replaced, "the index this save replaced").ok();
        } else {
            fs::rename(&self.staging, &self.target).map_err(|e| Error::MoveIntoPlaceFailed {
                target: self.target.clone(),
                error: e.to_string(),
            })?;
        }

        sync_directory(self.target.parent().unwrap_or_else(|| Path::new(".")));
        self.committed = true;
        Ok(())
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}

/// Put back an index a save was killed between its two renames
///
/// `<name>.zdbold` present with no target is the one case that holds data:
/// the save died between the two renames and that directory is the only copy
/// of the index. It is renamed back rather than removed, and the caller then
/// opens it.
///
/// `<name>.zdbold` present beside a target is the previous index after a
/// save that finished, and this leaves it where it is. Only a save removes
/// it, because only a save knows the target beside it is the one it wrote.
///
/// Reached from two places. `StagingDir::open`, so the next save puts the
/// index back before it stages a new one, which is where it has always been
/// reached from. And `load_index`, because recovery from a killed process is
/// a load: until this ran there, a process killed in that window left the
/// whole index beside the target and `load` reported the directory as not
/// found, and only another save could put it back.
///
/// Returns whether it moved anything.
pub(crate) fn restore_replaced(target: &Path) -> Result<bool, Error> {
    let replaced = sibling(target, REPLACED_SUFFIX)?;
    if !replaced.exists() || target.exists() {
        return Ok(false);
    }
    fs::rename(&replaced, target).map_err(|e| Error::RecoverRenameFailed {
        target: target.to_path_buf(),
        replaced: replaced.clone(),
        error: e.to_string(),
    })?;
    info!(target: LOG_TARGET, operation = "save_recover",
        restored = %target.display(),
        "An interrupted save had moved the index aside; it is back in place"
    );
    Ok(true)
}

/// Remove a directory tree, naming what it was in the failure
fn remove_tree(path: &Path, what: &'static str) -> Result<(), Error> {
    fs::remove_dir_all(path).map_err(|e| Error::RemoveTreeFailed {
        path: path.to_path_buf(),
        what,
        error: e.to_string(),
    })
}

/// Persist a directory's own entries, where the platform has a call for it
///
/// A file's bytes reaching the disk does not put its name in its directory. On
/// POSIX that needs the directory's own descriptor fsynced, which is what this
/// does, and without it a power loss can leave the renamed directory holding
/// entries that were never recorded.
///
/// **Windows has no equivalent through the standard library.** `File::open`
/// refuses a directory there, so this is a no-op, and the durability claim on
/// Windows rests on NTFS journalling the rename rather than on anything this
/// crate does. That difference is not observable from a gate that runs on
/// Windows.
///
/// Best effort on both. A filesystem that refuses the fsync is not a reason to
/// fail a save whose bytes are already written.
#[cfg(unix)]
fn sync_directory(path: &Path) {
    if let Ok(dir) = fs::File::open(path) {
        let _ = dir.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) {}

// ============================================================================
// A DIGEST PER ARTEFACT
// ============================================================================

/// What the manifest records about one artefact it names
///
/// `bytes` is the file's length and `checksum` is
/// [`zeusdb_vector_core::checksum_of`] over its contents, written as sixteen hex
/// digits. Both are taken from the buffer as it is written, so neither costs a
/// read.
///
/// `checksum` is absent for the graph dump alone. The dump is written by
/// `graph::dump::write_dump`, which streams it and then seeks back to fill the
/// header in, so there is no single buffer to hash and a digest would mean
/// reading the largest artefact in the directory back off the disk. It carries
/// a checksum over its own header and another over its own payload, both
/// verified by `parse_dump` on every load, so a manifest digest would duplicate
/// a check the loader already makes. Its length is recorded and checked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ArtefactDigest {
    pub(crate) bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) checksum: Option<String>,
}

/// The length and digest of every artefact a save has written so far
///
/// Filled as each file lands and handed to `save_manifest`, which is written
/// last and therefore names what really is on disk rather than what the save
/// intended to write.
#[derive(Default)]
pub(crate) struct SaveLedger {
    digests: HashMap<String, ArtefactDigest>,
}

impl SaveLedger {
    fn record_digest(&mut self, name: &str, bytes: u64, checksum: Option<u64>) {
        self.digests.insert(
            name.to_string(),
            ArtefactDigest {
                bytes,
                checksum: checksum.map(|sum| format!("{:016x}", sum)),
            },
        );
    }

    /// The length recorded for an artefact, where one has been.
    pub(crate) fn recorded_bytes(&self, name: &str) -> Option<u64> {
        self.digests.get(name).map(|digest| digest.bytes)
    }
}

/// An index writes its artefacts through the seam's ledger, which is this
/// one, so the manifest records what the index wrote the same way it
/// records what this module wrote.
impl zeusdb_vector_core::Ledger for SaveLedger {
    fn record(&mut self, name: &str, record: zeusdb_vector_core::ArtefactRecord) {
        self.record_digest(name, record.bytes, record.checksum);
    }
}

/// Write one artefact into the staging directory and record what went in
///
/// The file is fsynced before this returns. Without it the rename that moves
/// the staging directory into place can be recorded while the bytes it names
/// are still in the page cache, so a power loss leaves an index directory whose
/// manifest is complete and whose artefacts are empty. Every byte is already in
/// memory, so the fsync is the whole cost of that durability and it is measured
/// rather than assumed.
fn write_artefact(
    dir: &Path,
    name: &str,
    bytes: &[u8],
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    use std::io::Write;

    let path = dir.join(name);
    let mut file = fs::File::create(&path).map_err(|e| Error::ArtefactCreateFailed {
        name: name.to_string(),
        error: e.to_string(),
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| Error::ArtefactWriteFailed {
            name: name.to_string(),
            error: e.to_string(),
        })?;

    ledger.record_digest(name, bytes.len() as u64, Some(checksum_of(bytes)));
    Ok(())
}

/// Hold an artefact to the length and digest the manifest recorded for it
///
/// `files_included` says a file should be there and `check_files_present` says
/// it is. Neither says it holds what was written. A file that is present, the
/// right length and wrong in its contents used to load in silence: an edit to
/// one byte of `metadata.json` came back as that record's metadata, and a
/// flipped byte inside `vectors.bin` came back as that record's vector.
///
/// A directory written before this field existed carries no digests, so nothing
/// is checked and it loads exactly as it did.
fn verify_artefact(name: &str, bytes: &[u8], manifest: &IndexManifest) -> Result<(), Error> {
    let Some(recorded) = manifest.file_digests.get(name) else {
        return Ok(());
    };

    if bytes.len() as u64 != recorded.bytes {
        return Err(Error::ArtefactLengthMismatch {
            name: name.to_string(),
            actual: bytes.len(),
            recorded: recorded.bytes,
            contents: artefact_contents(name),
        });
    }

    let Some(expected) = recorded.checksum.as_deref() else {
        return Ok(());
    };
    let actual = format!("{:016x}", checksum_of(bytes));
    if actual != expected {
        return Err(Error::ArtefactDigestMismatch {
            name: name.to_string(),
            actual,
            expected: expected.to_string(),
            contents: artefact_contents(name),
        });
    }

    Ok(())
}

/// Read an artefact and verify it before anything parses it
fn read_artefact(path: &Path, name: &str, manifest: &IndexManifest) -> Result<Vec<u8>, Error> {
    let bytes = fs::read(path.join(name)).map_err(|e| Error::ArtefactReadFailed {
        name: name.to_string(),
        error: e.to_string(),
    })?;
    verify_artefact(name, &bytes, manifest)?;
    Ok(bytes)
}

/// The same, for the artefacts that are JSON
fn read_artefact_string(
    path: &Path,
    name: &str,
    manifest: &IndexManifest,
) -> Result<String, Error> {
    let bytes = read_artefact(path, name, manifest)?;
    String::from_utf8(bytes).map_err(|e| Error::ArtefactNotUtf8 {
        name: name.to_string(),
        error: e.to_string(),
    })
}

/// The length manifest.json records for the graph dump, where it records one
///
/// The dump is read by `graph::dump::parse_dump` rather than through
/// `read_artefact`, because it is streamed rather than held whole in memory.
/// This is the part of the digest check that still applies to it.
pub(crate) fn recorded_dump_length(manifest: &IndexManifest, name: &str) -> Option<u64> {
    manifest.file_digests.get(name).map(|entry| entry.bytes)
}

/// Refuse a directory this build cannot interpret, and return the major it
/// declares.
fn check_format_version(format_version: &str) -> Result<u32, Error> {
    let major = format_version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
        .ok_or_else(|| Error::FormatVersionUnparsable {
            format_version: format_version.to_string(),
            current: DENSE_FORMAT_VERSION,
        })?;

    if !SUPPORTED_FORMAT_MAJORS.contains(&major) {
        return Err(Error::FormatVersionUnsupported {
            format_version: format_version.to_string(),
            supported: SUPPORTED_FORMAT_LABEL,
            newer: major > SUPPORTED_FORMAT_MAJORS[SUPPORTED_FORMAT_MAJORS.len() - 1],
        });
    }

    Ok(major)
}

// ============================================================================
// DIRECTORY COMPLETENESS
// ============================================================================

/// Whether an artefact is one the loader can produce again rather than read
///
/// The graph is the only one. Every record carries what the graph is built
/// from, so a directory that lost its dump is rebuilt rather than refused, and
/// that has always been the behaviour. The list holds the name this build
/// writes and the pair 0.6.0 and earlier wrote, because a directory saved by
/// one of those names both of them under `files_included` and neither is
/// needed to reopen it.
///
/// A save now writes the dump before the manifest and moves the whole
/// directory into place afterwards, so a directory this build wrote and a
/// reader can see always holds the dump its manifest names. The exemption
/// stays for the directories that came before it, where the manifest was
/// written first and a save interrupted between the two left a manifest naming
/// a dump that was never written.
fn is_derived_artefact(name: &str) -> bool {
    name == GRAPH_DUMP_FILENAME || LEGACY_DUMP_FILENAMES.contains(&name)
}

/// What an artefact holds, for the message its absence produces
///
/// An unrecognised name is still load bearing. `files_included` has named only
/// what the save wrote since the field appeared in 0.3.0, so a name this build
/// does not know is a component a later release wrote, and absorbing its loss
/// is the failure this check exists to stop.
fn artefact_contents(name: &str) -> &'static str {
    match name {
        "config.json" => "the HNSW parameters, the saved record count and the index level metadata",
        "mappings.bin" => "the mapping from every external record id to its internal graph id",
        "metadata.json" => "the metadata of every record, which is what a filtered search reads",
        "vectors.bin" => "the raw vector of every record",
        "quantization.json" => {
            "the quantization configuration of either scheme and the training state"
        }
        "pq_centroids.bin" => "the trained PQ codebook, which every stored code decodes through",
        "pq_codes.bin" => "the quantized code of every record",
        INT8_SCALES_FILENAME => INT8_SCALES_CONTENTS,
        INT8_ROWS_FILENAME => INT8_ROWS_CONTENTS,
        _ if name.ends_with("/postings.zdbsparse") => {
            "the postings of a sparse space, being every record's term ids and weights"
        }
        _ if name.ends_with("/terms.zdbdict") => {
            "the term dictionary of a text layer, being every term and its id"
        }
        _ => "a component of the saved index that this build does not recognise",
    }
}

/// What the dictionary artefact holds, for the message its absence produces.
const DICTIONARY_CONTENTS: &str =
    "the term dictionary of a text layer, being every term and its id";

/// Refuse a directory that does not hold what its manifest says it holds
///
/// `files_included` is written from what the save actually wrote. Every entry
/// is pushed under the same condition the writer of that file tests, inside one
/// save holding the mutation lock, so the list is an inventory rather than a
/// statement about the storage mode. That has been true of every release that
/// wrote the field, which is 0.3.0 onwards, and it is what makes a named file
/// that is absent a directory that lost something rather than a directory this
/// build is reading wrongly.
///
/// This runs before any artefact is read, so a directory missing two files
/// names the first rather than failing on whichever one a partial load happens
/// to reach.
///
/// `manifest.json` is the last file a save writes and the directory is moved
/// into place whole, so an interrupted save cannot produce this state at all. A
/// directory that reaches it lost the file after a save that finished, or was
/// copied without it.
///
/// Without it a `quantized_with_raw` directory whose `vectors.bin` never landed
/// opened as a complete index built entirely from PQ reconstructions, and one
/// that lost `quantization.json` opened as an unquantized index. Both were
/// silent.
fn check_files_present(path: &Path, manifest: &IndexManifest) -> Result<(), Error> {
    let missing: Vec<&str> = manifest
        .files_included
        .iter()
        .map(String::as_str)
        .filter(|name| !is_derived_artefact(name))
        .filter(|name| !path.join(name).exists())
        .collect();

    let Some(&first) = missing.first() else {
        debug!(target: LOG_TARGET, "Every file manifest.json names is present ({} checked)",
            manifest.files_included.len()
        );
        return Ok(());
    };

    Err(Error::ArtefactsMissing {
        missing: missing.iter().map(|name| name.to_string()).collect(),
        contents: artefact_contents(first),
    })
}

/// Whether the manifest's inventory names an artefact
///
/// The optional artefacts are read only when `files_included` names them. A
/// save used to replace files one at a time and remove none, so a raw index
/// saved over a quantized one reopened as a quantized index holding the
/// previous save's codebook and codes, and the record count agreed so nothing
/// caught it. A save now builds its directory from nothing, so no artefact of
/// an earlier save survives one and this check no longer has anything to
/// exclude. It stays because a directory written by an earlier release can
/// still hold those files.
///
/// The graph dump is not gated this way. It is derived, it carries its own
/// checks on node count, distance kind and `m`, and it already falls back to
/// the rebuild when any of them disagree.
fn manifest_names(manifest: &IndexManifest, name: &str) -> bool {
    manifest.files_included.iter().any(|entry| entry == name)
}

// ============================================================================
// PERSISTENCE DATA STRUCTURES
// ============================================================================

/// Manifest file structure - tracks index metadata and included files
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexManifest {
    pub(crate) format_version: String,
    pub(crate) zeusdb_version: String,
    pub(crate) created_at: String,
    pub(crate) saved_at: String,
    pub(crate) total_vectors: usize,
    pub(crate) index_type: String,
    pub(crate) has_quantization: bool,
    pub(crate) quantization_trained: bool,
    pub(crate) storage_mode: String,
    pub(crate) files_included: Vec<String>,
    pub(crate) files_excluded: Vec<String>,

    /// The length and digest of every artefact `files_included` names
    ///
    /// A map beside the list rather than a change to the list, because
    /// `files_included` is a `Vec<String>` in every release that has read this
    /// file and turning it into a list of objects would stop those releases
    /// parsing a directory this one wrote. serde ignores a field it does not
    /// know, so an older build reads a directory written here and this build
    /// reads one written there, where the map defaults to empty and nothing is
    /// verified.
    ///
    /// See `ArtefactDigest` for what is recorded and why the graph dump carries
    /// a length alone.
    #[serde(default)]
    pub(crate) file_digests: HashMap<String, ArtefactDigest>,

    /// Every byte the directory holds except `manifest.json` itself
    ///
    /// The manifest is now the last file a save writes, so it does not exist
    /// when the figure is taken and cannot count itself. It used to be written
    /// before the graph dump and then rewritten through a temporary file to
    /// record a total it had missed the largest artefact of, which is a second
    /// write this ordering removes.
    pub(crate) total_size_mb: f64,
    pub(crate) compression_info: Option<CompressionInfo>,

    /// The journal beside the directory, where the collection that saved it
    /// held one.
    ///
    /// Absent from the file where none was held, so a directory saved
    /// without a journal is byte for byte the directory the same records
    /// wrote before this field existed, and its `format_version` does not
    /// move either. Present, it is what makes the directory a 3.x one; see
    /// `JOURNAL_FORMAT_VERSION`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) journal: Option<JournalManifest>,
}

/// How `manifest.json` names the journal beside the directory.
///
/// The three values are what pair a directory with a file that is not in it.
/// `collection_id` is drawn when a collection is built and is in the
/// journal's own header too, so a journal from another index is refused by
/// content. `sequence` is what the journal had reached when this checkpoint
/// was taken, so a recovery replays every record above it and skips every
/// record at or below it. `file` is the sibling's name as the checkpoint
/// wrote it, which a refusal quotes; the file a load actually opens is
/// derived from the directory it was handed, so a directory renamed together
/// with its journal opens under the new name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct JournalManifest {
    pub(crate) file: String,
    pub(crate) sequence: u64,
    pub(crate) collection_id: String,
}

/// The manifest is what an index reads its artefacts' recorded lengths from.
///
/// A framed artefact is recorded by its length alone, so `checksum` is
/// `None` for one, and the frame's own payload checksum is what the reader
/// verifies. A directory written before the manifest carried digests
/// records nothing, and an index restoring from one is refused before it
/// reads, since no such directory holds a space.
impl Inventory for IndexManifest {
    fn recorded(&self, name: &str) -> Option<ArtefactRecord> {
        let digest = self.file_digests.get(name)?;
        Some(ArtefactRecord {
            bytes: digest.bytes,
            checksum: digest
                .checksum
                .as_deref()
                .and_then(|hex| u64::from_str_radix(hex, 16).ok()),
        })
    }
}

/// Compression statistics for quantized indexes
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CompressionInfo {
    pub(crate) original_size_mb: f64,
    pub(crate) compressed_size_mb: f64,
    pub(crate) compression_ratio: f64,
}

/// Index configuration for reconstruction
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexConfig {
    pub(crate) dim: usize,
    pub(crate) space: String,
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) expected_size: usize,
    pub(crate) id_counter: usize,
    pub(crate) vector_count: usize,

    /// How many generated ids the index had issued, being the `N` of `vec_N`
    ///
    /// Separate from `id_counter`, which `clear` resets and this does not. See
    /// `Collection::generated_ids`. Defaulted, so a directory written before the
    /// field existed loads with a zero and takes its floor from the records it
    /// holds instead.
    #[serde(default)]
    pub(crate) generated_ids: usize,

    /// Index level metadata set through `add_metadata`
    ///
    /// Defaulted rather than required, so a directory written before this field
    /// existed loads with an empty map instead of failing to parse.
    #[serde(default)]
    pub(crate) metadata: HashMap<String, String>,

    /// The filterable fields declared at `create()`, in declaration order.
    ///
    /// **The columns themselves are not saved.** They are derived from
    /// `metadata.json`, which is written whole and read whole, so rebuilding
    /// them at load costs one pass over the records and keeps the directory
    /// format to the files it already had. What has to survive a round trip is
    /// the declaration, because nothing else records which fields a user chose.
    ///
    /// Defaulted, so a directory written before this field existed loads with
    /// no declaration and behaves exactly as it did. `serde` ignores fields it
    /// does not know, so a directory written with one also opens in a build
    /// that predates it, at the cost of the columns rather than of the load.
    #[serde(default)]
    pub(crate) indexed_fields: Vec<String>,

    /// The sparse space declared beside the dense one, by value, where one
    /// was.
    ///
    /// Absent from the file where none was declared, so a dense-only
    /// directory's `config.json` is byte for byte what it was before the
    /// field existed and opens on a release that predates it. Present, it
    /// is what makes the directory a 2.x one; see the module documentation.
    /// A list rather than one record, so a further space is a further entry
    /// rather than a further field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) spaces: Vec<SpaceRecord>,
}

/// One sparse space as `config.json` declares it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SpaceRecord {
    /// The name the space was declared under, which is the directory under
    /// `spaces/` its artefacts are written to.
    pub(crate) name: String,
    /// The family of vector the space holds. `sparse` is the one this build
    /// writes, and a reader that is not the engine reads it before the
    /// fields below.
    pub(crate) kind: String,
    /// The index's own declaration, every field named.
    pub(crate) index: SparseConfig,
    /// The tokenizer, where the space takes text. `simple` is the built-in
    /// one and `external` is an implementation the caller keeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tokenizer: Option<TokenizerConfig>,
}

/// The one kind of space `config.json` declares under `spaces`.
const SPARSE_KIND: &str = "sparse";

/// Hold the spaces `config.json` declares to the rules `Declaration`
/// applies to a caller's own, so a hand edited file cannot build a space
/// the collection would refuse to declare.
fn validate_spaces(spaces: &[SpaceRecord], file: &str) -> Result<(), Error> {
    let invalid = |detail: String| Error::SpaceRecordInvalid {
        file: file.to_string(),
        detail,
    };
    if spaces.len() > 1 {
        return Err(invalid(format!(
            "a collection holds one sparse space and {} are declared",
            spaces.len()
        )));
    }
    for space in spaces {
        if space.kind != SPARSE_KIND {
            return Err(invalid(format!(
                "space '{}' declares kind '{}', which this build does not hold",
                space.name, space.kind
            )));
        }
        SpaceName::new(&space.name).map_err(|e| invalid(e.to_string()))?;
        if space.name == DEFAULT_SPACE {
            return Err(invalid(format!(
                "space '{}' takes the dense space's name",
                space.name
            )));
        }
        space.index.validate().map_err(|e| invalid(e.to_string()))?;
    }
    Ok(())
}

/// The tokenizer a restored text layer runs, from what `config.json`
/// recorded and what the caller handed to `load`.
///
/// The built-in tokenizer is recorded as `simple` and is built here where
/// none was handed. A caller's own is recorded as `external` and is
/// accepted only when handed. A handed tokenizer whose own declaration is
/// not the recorded one is refused, and so is one handed to a directory
/// with no text layer, since ignoring it would open the collection under a
/// tokenizer the caller did not ask for.
fn resolve_tokenizer(
    space: Option<&SpaceRecord>,
    handed: Option<Arc<dyn Tokenizer>>,
) -> Result<Option<Arc<dyn Tokenizer>>, Error> {
    let recorded = space.and_then(|space| space.tokenizer.as_ref());
    match (recorded, handed) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(Error::TokenizerUnexpected),
        (Some(TokenizerConfig::Simple), None) => Ok(Some(Arc::new(SimpleTokenizer))),
        (Some(TokenizerConfig::External), None) => Err(Error::TokenizerRequired {
            space: space.map(|s| s.name.clone()).unwrap_or_default(),
        }),
        (Some(recorded), Some(handed)) => {
            let declared = handed.config();
            if declared == *recorded {
                Ok(Some(handed))
            } else {
                Err(Error::TokenizerMismatch {
                    space: space.map(|s| s.name.clone()).unwrap_or_default(),
                    recorded: recorded.name(),
                    handed: declared.name(),
                })
            }
        }
    }
}

/// Complete quantization configuration and state
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct QuantizationPersistence {
    pub(crate) r#type: String,
    pub(crate) subvectors: usize,
    pub(crate) bits: usize,
    pub(crate) training_size: usize,
    pub(crate) max_training_vectors: Option<usize>,
    pub(crate) storage_mode: String,
    pub(crate) is_trained: bool,
    pub(crate) training_completed_at: Option<String>,
    pub(crate) memory_stats: Option<MemoryStats>,
    pub(crate) pq_config: PQConfig,
    #[serde(default)]
    pub(crate) training_ids: Vec<String>,
    #[serde(default)]
    pub(crate) training_threshold_reached: bool,
    /// What training measured about the rerank fetch on this index's own data.
    ///
    /// Absent from every directory written before the calibration existed, so
    /// it defaults to `None` and those indexes fall back to the corpus terms
    /// they were built against. See `RerankCalibration`.
    #[serde(default)]
    pub(crate) rerank_calibration: Option<RerankCalibration>,
}

/// Memory usage statistics for quantization
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MemoryStats {
    pub(crate) centroid_storage_mb: f64,
    pub(crate) compression_ratio: f64,
    pub(crate) centroids_per_subvector: usize,
    pub(crate) total_centroids: usize,
}

/// Product Quantization configuration details
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PQConfig {
    pub(crate) dim: usize,
    pub(crate) sub_dim: usize,
    pub(crate) num_centroids: usize,
}

/// ID mappings between external and internal IDs
#[derive(Debug, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub(crate) struct IdMappings {
    pub(crate) id_map: HashMap<String, usize>,
    pub(crate) rev_map: HashMap<usize, String>,
}

/// quantization.json under `type: int8`: the scalar declaration and the
/// training state, and none of the product quantized fields.
///
/// A reader that knows the product quantized layout alone refuses this file
/// on its first missing field, which is what lets the format version move by
/// a minor rather than a major; see the module documentation.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Int8Persistence {
    pub(crate) r#type: String,
    pub(crate) scale: String,
    pub(crate) training_size: usize,
    pub(crate) max_training_vectors: Option<usize>,
    pub(crate) storage_mode: String,
    pub(crate) is_trained: bool,
    pub(crate) training_completed_at: Option<String>,
    #[serde(default)]
    pub(crate) training_ids: Vec<String>,
    #[serde(default)]
    pub(crate) training_threshold_reached: bool,
    /// Values the encode has clipped since the scales were fitted; see
    /// `DenseIndex::saturated`.
    #[serde(default)]
    pub(crate) saturated_values: u64,
}

/// quantization.json, as one of the two layouts it takes.
enum QuantizationFile {
    Pq(QuantizationPersistence),
    Int8(Int8Persistence),
}

/// PQ codebook laid out as [subvector][centroid][dimension within subvector]
type Centroids = Vec<Vec<Vec<f32>>>;

/// Everything the loader reads back for a quantized index
struct QuantizationArtefacts {
    config: QuantizationFile,
    centroids: Option<Centroids>,
    codes: HashMap<String, Vec<u8>>,
    /// The fitted scalar codec, from `int8_scales.zdbint8` on a trained
    /// scalar directory.
    int8: Option<Arc<Int8Codec>>,
    /// Every record's scalar row from `int8_rows.zdbint8`, by internal id,
    /// ascending.
    rows: Int8Rows,
}

/// Every row of a trained scalar directory in one block, with the internal
/// id of each beside it, ascending
///
/// `int8_rows.zdbint8` holds one row a record. The loader used to hold one
/// heap block a row on top of the file, every one of them freed at the end of
/// the load, which is what an allocator that keeps freed small blocks left
/// the process holding. One block holds them now, and a row is a slice of it.
#[derive(Default)]
pub(crate) struct Int8Rows {
    ids: Vec<u32>,
    data: Vec<i8>,
    width: usize,
}

impl Int8Rows {
    fn with_capacity(entries: usize, width: usize) -> Self {
        Int8Rows {
            ids: Vec::with_capacity(entries),
            data: Vec::with_capacity(entries.saturating_mul(width)),
            width,
        }
    }

    fn push(&mut self, id: u32, row: &[u8]) {
        self.ids.push(id);
        self.data.extend(row.iter().map(|&b| b as i8));
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// Every row with its internal id, ascending.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &[i8])> + '_ {
        self.ids
            .iter()
            .zip(self.data.chunks_exact(self.width.max(1)))
            .map(|(&id, row)| (id as usize, row))
    }
}

/// Training collection state, applied once the graph is back
///
/// The collected ids and the threshold flag are what `quantization.json`
/// recorded, and they are applied after the graph is restored or rebuilt so
/// that the flag is judged against the records the loaded collection holds.
/// Neither fallback touches the training collection: both build the graph off
/// to the side under the mapped ids and swap it in.
struct TrainingState {
    ids: Vec<String>,
    threshold_reached: bool,
    is_trained: bool,
    training_size: usize,
    /// The clipped value count a scalar directory recorded, applied once
    /// the index holds its graph, since the graph's restore builds a fresh
    /// index whose count starts at zero.
    saturated: u64,
}

impl TrainingState {
    fn from(config: &QuantizationFile) -> Self {
        match config {
            QuantizationFile::Pq(config) => TrainingState {
                ids: config.training_ids.clone(),
                threshold_reached: config.training_threshold_reached,
                is_trained: config.is_trained,
                training_size: config.training_size,
                saturated: 0,
            },
            QuantizationFile::Int8(config) => TrainingState {
                ids: config.training_ids.clone(),
                threshold_reached: config.training_threshold_reached,
                is_trained: config.is_trained,
                training_size: config.training_size,
                saturated: config.saturated_values,
            },
        }
    }

    fn apply(self, index: &mut Collection) {
        // A trained index cleared its collection when training ran, so this is
        // only ever populated for an index saved while still collecting.
        let collected = self.ids.len();
        index.set_training_ids(self.ids);

        // The saved flag is authoritative for a trained index. For an untrained
        // one it is recomputed, so a directory whose collection was truncated
        // does not come back claiming a threshold it no longer meets.
        let reached = if self.is_trained {
            self.threshold_reached
        } else {
            collected >= self.training_size
        };
        index.set_training_threshold_reached(reached);
        if self.saturated > 0 {
            index.set_int8_saturated(self.saturated);
        }

        debug!(target: LOG_TARGET, "Training state restored ({} collected ids, threshold reached: {})",
            collected, reached
        );
    }
}

// ============================================================================
// INDIVIDUAL COMPONENT LOADERS
// ============================================================================

/// Load index configuration from config.json
fn load_config(path: &Path, manifest: &IndexManifest) -> Result<IndexConfig, Error> {
    debug!(target: LOG_TARGET, "Loading config.json...");

    let config_path = path.join("config.json");
    let config_data = read_artefact_string(path, "config.json", manifest)?;

    let config: IndexConfig =
        serde_json::from_str(&config_data).map_err(|e| Error::ArtefactParseFailed {
            name: "config.json",
            error: e.to_string(),
        })?;

    // The five values `build` validates, validated here too.
    //
    // Parsing proves the file is JSON of the right shape and nothing more. Until
    // this ran, `dim`, `m`, `ef_construction`, `expected_size` and `space` went
    // straight from the file into `new_empty`, which validates none of them, and
    // then into `Backend::sized`, which clamps `dim` up to 1, `m` into 2 to 256
    // and `expected_size` up to 1 without saying so. A config naming `m: 0` came
    // back as an index at `m: 2`, and one naming an unknown `space` came back
    // scoring cosine whatever it had been saved with. A zero `dim` was refused,
    // but by a later check comparing a record against the declared width, so the
    // message named the record rather than the config.
    //
    // The file is named in the message because a caller reading `dim must be
    // positive` off a `load()` has no argument of their own to look at.
    validate_index_parameters(
        config.dim,
        &config.space,
        config.m,
        config.ef_construction,
        config.expected_size,
        &format!("{}: ", config_path.display()),
    )?;
    // `id_counter` too, which those five do not cover and which sizes an
    // allocation rather than a behaviour.
    //
    // The internal id a record is inserted under is the index into the graph's
    // id-to-node array, so that array is `id_counter + 1` slots of four bytes.
    // A hand edited config declaring 2^40 loaded without complaint, and the
    // next `add` asked the allocator for 4,398,046,511,112 bytes and **aborted
    // the process**. An allocation failure does not unwind, so no `catch_unwind`
    // sees one and a Python caller gets a dead interpreter with no traceback.
    //
    // It is the same root cause as the graph dump's origin id, seen from the
    // other side: one dense array, two unvalidated sources for its index. The
    // dump's side is checked in `graph::dump::parse_dump` against this same
    // field, so bounding it here bounds both.
    //
    // The ceiling is `u32::MAX` because a node index is a `u32` and both graph
    // constructors refuse a graph holding more points than that. Every id was
    // issued to a record the graph then held a node for, so an index that
    // issued more ids than a node index can name has a graph it could not have
    // built. The check refuses nothing that can exist.
    if config.id_counter > u32::MAX as usize {
        return Err(Error::IdCounterTooLarge {
            file: config_path.display().to_string(),
            id_counter: config.id_counter,
        });
    }
    // The declaration too, for the same reason. A config naming a field twice,
    // or naming a reserved filter key, would build a store the index could not
    // use, and the failure would surface as a filter that quietly walked.
    validate_indexed_fields(
        &config.indexed_fields,
        &format!("{}: ", config_path.display()),
    )?;
    // The spaces too, under the rules a declaration is held to.
    validate_spaces(&config.spaces, &config_path.display().to_string())?;

    debug!(target: LOG_TARGET, "config.json loaded");
    Ok(config)
}

/// Read the sparse space's artefacts back and install them, where the
/// collection declares one.
///
/// Runs after the id mappings are restored and before the graph is, for two
/// reasons. The largest internal id the mappings hold is what bounds the
/// space's slot table, and the graph's rebuild fallback replays every record
/// through `add`, which reads each record's sparse half out of the space and
/// carries it through; see `Collection::rebuild_from_records`.
///
/// Every record the space holds must be one the mappings name. A record may
/// leave the space empty, so the space may hold fewer, and a space holding
/// an id the mappings do not is a directory whose two artefacts came from
/// different saves. A text layer's dictionary must hold every term id the
/// postings carry, for the same reason.
fn restore_spaces(index: &Collection, path: &Path, manifest: &IndexManifest) -> Result<(), Error> {
    let Some((name, space)) = index.sparse_named() else {
        return Ok(());
    };
    let prefix = Collection::space_prefix(name);
    let largest_id = index.ids().highest_slot().unwrap_or(0);
    let bounds = Bounds {
        min_records: 0,
        max_records: largest_id,
        max_bytes: u64::MAX,
    };
    let restored = PostingsIndex::restore(space.config(), &prefix, path, manifest, &bounds)?;
    let unmapped = {
        let ids = index.ids();
        let mut unmapped = None;
        restored.live_set().for_each_while(|slot| {
            if ids.contains_slot(slot) {
                true
            } else {
                unmapped = Some(slot);
                false
            }
        });
        unmapped
    };
    if let Some(slot) = unmapped {
        return Err(Error::SparseRecordUnmapped {
            space: name.as_str().to_string(),
            id: u32::try_from(slot).unwrap_or(u32::MAX),
        });
    }
    if space.text.is_some() {
        let dictionary_name = Collection::dictionary_name(&prefix);
        let bytes = zeusdb_vector_core::read_artefact(
            path,
            &dictionary_name,
            manifest,
            DICTIONARY_CONTENTS,
            u64::MAX,
        )?;
        let dictionary = TermDictionary::decode(&bytes, &dictionary_name)?;
        if let Some(term) = restored.max_dim() {
            if term as usize >= dictionary.len() {
                return Err(Error::TermIdBeyondDictionary {
                    space: name.as_str().to_string(),
                    term,
                    terms: dictionary.len(),
                });
            }
        }
        debug!(target: LOG_TARGET, "{} restored ({} terms)", dictionary_name, dictionary.len());
        index.set_dictionary(dictionary);
    }
    debug!(target: LOG_TARGET, "{}postings.zdbsparse restored ({} records, {} postings)",
        prefix,
        restored.len(),
        restored.postings_total()
    );
    index.set_sparse_index(restored);
    Ok(())
}

/// Load ID mappings from mappings.bin
fn load_mappings(path: &Path, manifest: &IndexManifest) -> Result<IdMappings, Error> {
    debug!(target: LOG_TARGET, "Loading mappings.bin...");

    let mappings_data = read_artefact(path, "mappings.bin", manifest)?;

    let mappings: IdMappings = decode_bounded(&mappings_data, "mappings.bin", ())?;

    debug!(target: LOG_TARGET, "mappings.bin loaded");
    Ok(mappings)
}

/// Load vector metadata from metadata.json
fn load_metadata(
    path: &Path,
    manifest: &IndexManifest,
) -> Result<HashMap<String, HashMap<String, Value>>, Error> {
    debug!(target: LOG_TARGET, "Loading metadata.json...");

    let metadata_data = read_artefact_string(path, "metadata.json", manifest)?;

    let metadata: HashMap<String, HashMap<String, Value>> = serde_json::from_str(&metadata_data)
        .map_err(|e| Error::ArtefactParseFailed {
            name: "metadata.json",
            error: e.to_string(),
        })?;

    debug!(target: LOG_TARGET, "metadata.json loaded");
    Ok(metadata)
}

/// Load raw vectors from vectors.bin
///
/// Read only when the manifest names it. A trained `quantized_only` index
/// writes none, and a directory saved over one that did keeps the file the
/// earlier save left. See `manifest_names`.
fn load_vectors(
    path: &Path,
    manifest: &IndexManifest,
    records: usize,
) -> Result<HashMap<String, Vec<f32>>, Error> {
    debug!(target: LOG_TARGET, "Loading vectors.bin...");

    if !manifest_names(manifest, "vectors.bin") {
        debug!(target: LOG_TARGET, "manifest.json does not list vectors.bin, so no raw vectors are read");
        return Ok(HashMap::new());
    }

    let vectors_data = read_artefact(path, "vectors.bin", manifest)?;

    let vectors: HashMap<String, Vec<f32>> = decode_bounded(&vectors_data, "vectors.bin", records)?;

    check_vectors_are_finite(&vectors)?;

    debug!(target: LOG_TARGET, "vectors.bin loaded");
    Ok(vectors)
}

/// Refuse a stored vector holding a NaN or an infinity
///
/// `add` has always refused a non-finite value, so one can only reach
/// vectors.bin through a release that did not validate every input path or
/// through a directory edited after it was written. The check belongs to the
/// loader rather than to the graph rebuild, because the graph is now restored
/// from its dump and the rebuild that used to catch this does not run. A record
/// holding a NaN scores as NaN against every query and orders arbitrarily, so
/// the index would answer wrongly rather than visibly fail.
fn check_vectors_are_finite(vectors: &HashMap<String, Vec<f32>>) -> Result<(), Error> {
    let mut offenders: Vec<&String> = vectors
        .iter()
        .filter(|(_, vector)| vector.iter().any(|value| !value.is_finite()))
        .map(|(id, _)| id)
        .collect();

    if offenders.is_empty() {
        return Ok(());
    }

    offenders.sort();
    Err(Error::VectorsNotFinite {
        offenders: offenders.into_iter().cloned().collect(),
        total: vectors.len(),
    })
}

/// What the loader holds of vectors.bin
///
/// A raw index takes every vector back from the graph dump, which carries
/// its store, so the file is walked for its record count and its finiteness
/// check and nothing of it is kept. `Counted` decodes the file one record at
/// a time through the decoder `decode_bounded` builds, so a file
/// `load_vectors` refuses is refused with the same message. A quantized
/// index holds the map, since a `quantized_with_raw` index places the
/// vectors by id once the graph is back and a product quantized rebuild
/// replays them. The rebuild fallback of a raw index reads the file again,
/// through `hold`.
///
/// Held whole, the file cost one heap block a record for the whole of the
/// load, beside the dump's own copy of every vector, and every one of those
/// blocks was freed at the end. An allocator that keeps freed small blocks
/// left the process holding them after the load returned.
enum RawVectors {
    Held(HashMap<String, Vec<f32>>),
    Counted(usize),
}

impl RawVectors {
    fn len(&self) -> usize {
        match self {
            RawVectors::Held(map) => map.len(),
            RawVectors::Counted(count) => *count,
        }
    }

    fn held(&self) -> Option<&HashMap<String, Vec<f32>>> {
        match self {
            RawVectors::Held(map) => Some(map),
            RawVectors::Counted(_) => None,
        }
    }

    fn into_held(self) -> HashMap<String, Vec<f32>> {
        match self {
            RawVectors::Held(map) => map,
            RawVectors::Counted(_) => HashMap::new(),
        }
    }

    /// Read the map where only the count was kept.
    fn hold(&mut self, path: &Path, manifest: &IndexManifest, records: usize) -> Result<(), Error> {
        if let RawVectors::Counted(_) = self {
            *self = RawVectors::Held(load_vectors(path, manifest, records)?);
        }
        Ok(())
    }
}

/// Walk vectors.bin for its record count and its finiteness check, keeping
/// nothing
///
/// Read only when the manifest names it, as `load_vectors` is. The count is
/// the one the map decode gave, an id the file holds twice counted once,
/// and a record whose vector is not finite is named exactly as
/// `check_vectors_are_finite` names it. `records` is the count the mappings
/// hold, which sizes what the walk keeps; see `walk_vectors_with`.
fn count_vectors(path: &Path, manifest: &IndexManifest, records: usize) -> Result<usize, Error> {
    debug!(target: LOG_TARGET, "Loading vectors.bin...");

    if !manifest_names(manifest, "vectors.bin") {
        debug!(target: LOG_TARGET, "manifest.json does not list vectors.bin, so no raw vectors are read");
        return Ok(0);
    }

    let vectors_data = read_artefact(path, "vectors.bin", manifest)?;

    let (count, offenders) = walk_vectors(&vectors_data, "vectors.bin", records)?;
    if !offenders.is_empty() {
        return Err(Error::VectorsNotFinite {
            offenders,
            total: count,
        });
    }

    debug!(target: LOG_TARGET, "vectors.bin walked ({} vectors, none kept)", count);
    Ok(count)
}

/// Decode a map of vectors one record at a time, returning the record count
/// and the ids whose vector holds a value that is not finite, sorted
///
/// The budget is the one `decode_bounded` gives the same file, the rung's
/// excess claimed first and all, and the steps are the ones bincode's own map
/// decoder takes, the container claim included, so the two refuse the same
/// files for the same reasons. The count and the names are the map's as well;
/// see `walk_vectors_with`, which `records`, the count the mappings hold,
/// sizes.
fn walk_vectors(data: &[u8], file: &str, records: usize) -> Result<(usize, Vec<String>), Error> {
    walk_at(data, file, records, claim_budget(data.len()))
}

/// The claim budget a file of `bytes` earns.
fn claim_budget(bytes: usize) -> usize {
    bytes
        .saturating_mul(CLAIM_PER_WIRE_BYTE)
        .max(LEADING_LENGTH_CLAIM)
}

/// The walk above under a budget given outright, as `decode_at` is.
fn walk_at(
    data: &[u8],
    file: &str,
    records: usize,
    budget: usize,
) -> Result<(usize, Vec<String>), Error> {
    use bincode::config::standard;

    let walked = if budget <= 1 << 20 {
        walk_vectors_with(
            data,
            standard().with_limit::<{ 1 << 20 }>(),
            records,
            budget,
        )
    } else if budget <= 1 << 28 {
        walk_vectors_with(
            data,
            standard().with_limit::<{ 1 << 28 }>(),
            records,
            budget,
        )
    } else if budget <= 1 << 36 {
        walk_vectors_with(
            data,
            standard().with_limit::<{ 1 << 36 }>(),
            records,
            budget,
        )
    } else {
        walk_vectors_with(
            data,
            standard().with_limit::<{ 1 << 44 }>(),
            records,
            budget,
        )
    };

    match walked {
        Ok(value) => Ok(value),
        Err(bincode::error::DecodeError::LimitExceeded) => Err(Error::DecodeLengthExceeded {
            file: file.to_string(),
            bytes: data.len(),
        }),
        Err(e) => Err(Error::DecodeFailed {
            file: file.to_string(),
            error: e.to_string(),
        }),
    }
}

/// The steps `HashMap<String, Vec<f32>>` takes to decode, one record at a
/// time and with no map built, each id and vector read by `HeldDecode`
///
/// The map held one entry for an id the file named twice, the last copy,
/// so a file holding a duplicate counted it once and judged it by the copy
/// that came last. The walk gives the same answer without the map. Each id
/// is hashed as it is read into a set of 64 bit hashes sized by `records`,
/// the count the mappings hold, and a set as large as the header's count
/// proves every id distinct, so the count is the header's. A smaller set
/// holds a duplicate or a collision, and `count_distinct_ids` walks the
/// file again holding every id by name, which is what the map cost and
/// which no directory a save wrote reaches. The offenders are held by name
/// throughout, a finite copy releasing the name a copy before it entered,
/// so the ids named are the ids whose last copy is not finite, as the
/// map's entries were.
fn walk_vectors_with<C: bincode::config::Config>(
    data: &[u8],
    config: C,
    records: usize,
    budget: usize,
) -> Result<(usize, Vec<String>), bincode::error::DecodeError> {
    use bincode::de::Decoder;
    use bincode::Decode;
    use std::collections::{BTreeSet, HashSet};
    use std::hash::{BuildHasher, RandomState};

    let hasher = RandomState::new();
    let mut decoder = bincode::de::DecoderImpl::new(TrackedSlice::new(data), config, ());
    claim_rung_excess(&mut decoder, budget)?;
    let claimed = u64::decode(&mut decoder)?;
    let len = usize::try_from(claimed)
        .map_err(|_| bincode::error::DecodeError::OutsideUsizeRange(claimed))?;
    decoder.claim_container_read::<(String, Vec<f32>)>(len)?;
    let mut hashes: HashSet<u64> = HashSet::with_capacity(records);
    let mut offenders: BTreeSet<String> = BTreeSet::new();
    for _ in 0..len {
        decoder.unclaim_bytes_read(std::mem::size_of::<(String, Vec<f32>)>());
        let id = String::decode_held(&mut decoder, ())?;
        let vector = Vec::<f32>::decode_held(&mut decoder, ())?;
        if vector.iter().all(|value| value.is_finite()) {
            offenders.remove(&id);
        } else {
            offenders.insert(id.clone());
        }
        hashes.insert(hasher.hash_one(&id));
    }
    let count = if hashes.len() == len {
        len
    } else {
        count_distinct_ids(data, config, budget)?
    };
    Ok((count, offenders.into_iter().collect()))
}

/// The distinct ids of a file whose id hashes repeated, held by name
///
/// The same steps again over the same bytes, which the walk has already
/// taken without error, so this cannot fail where the walk did not.
fn count_distinct_ids<C: bincode::config::Config>(
    data: &[u8],
    config: C,
    budget: usize,
) -> Result<usize, bincode::error::DecodeError> {
    use bincode::de::Decoder;
    use bincode::Decode;
    use std::collections::HashSet;

    let mut decoder = bincode::de::DecoderImpl::new(TrackedSlice::new(data), config, ());
    claim_rung_excess(&mut decoder, budget)?;
    let claimed = u64::decode(&mut decoder)?;
    let len = usize::try_from(claimed)
        .map_err(|_| bincode::error::DecodeError::OutsideUsizeRange(claimed))?;
    decoder.claim_container_read::<(String, Vec<f32>)>(len)?;
    let mut ids: HashSet<String> = HashSet::new();
    for _ in 0..len {
        decoder.unclaim_bytes_read(std::mem::size_of::<(String, Vec<f32>)>());
        let id = String::decode_held(&mut decoder, ())?;
        Vec::<f32>::decode_held(&mut decoder, ())?;
        ids.insert(id);
    }
    Ok(ids.len())
}

/// Load manifest for validation and metadata
fn load_manifest(path: &Path) -> Result<IndexManifest, Error> {
    debug!(target: LOG_TARGET, "Loading manifest.json...");

    let manifest_path = path.join("manifest.json");
    let manifest_data =
        fs::read_to_string(&manifest_path).map_err(|e| Error::ArtefactReadFailed {
            name: "manifest.json".to_string(),
            error: e.to_string(),
        })?;

    let manifest: IndexManifest =
        serde_json::from_str(&manifest_data).map_err(|e| Error::ArtefactParseFailed {
            name: "manifest.json",
            error: e.to_string(),
        })?;

    debug!(target: LOG_TARGET, "manifest.json loaded");
    Ok(manifest)
}

/// Load the PQ codebook from pq_centroids.bin
///
/// Absent means the index was saved before training completed, which is a
/// legitimate state. A present but unreadable file is a hard failure, because
/// the alternative is a codebook that decodes every code to the zero vector.
fn load_pq_centroids(
    path: &Path,
    manifest: &IndexManifest,
    shape: (usize, usize),
) -> Result<Option<Centroids>, Error> {
    if !manifest_names(manifest, "pq_centroids.bin") {
        return Ok(None);
    }

    debug!(target: LOG_TARGET, "Loading pq_centroids.bin...");

    let centroids_data = read_artefact(path, "pq_centroids.bin", manifest)?;

    let centroids: Centroids = decode_bounded(&centroids_data, "pq_centroids.bin", shape)?;

    debug!(target: LOG_TARGET, "pq_centroids.bin loaded ({} subvectors)", centroids.len());
    Ok(Some(centroids))
}

/// Load the quantized codes from pq_codes.bin
///
/// Absent means no record has been quantized yet. In `quantized_only` these
/// codes are the only copy of every record added after training completed.
fn load_pq_codes(
    path: &Path,
    manifest: &IndexManifest,
    records: usize,
) -> Result<HashMap<String, Vec<u8>>, Error> {
    if !manifest_names(manifest, "pq_codes.bin") {
        return Ok(HashMap::new());
    }

    debug!(target: LOG_TARGET, "Loading pq_codes.bin...");

    let codes_data = read_artefact(path, "pq_codes.bin", manifest)?;

    let codes: HashMap<String, Vec<u8>> = decode_bounded(&codes_data, "pq_codes.bin", records)?;

    debug!(target: LOG_TARGET, "pq_codes.bin loaded ({} records)", codes.len());
    Ok(codes)
}

/// Hold quantization.json's two sizing fields to the rules `create()` applies
///
/// `bits` fixes the centroid count at `2^bits` and `subvectors` fixes both the
/// outer dimension of the codebook and the divisor `sub_dim` comes from, so the
/// pair sizes every allocation `PQ::new` makes. The Python layer validates both
/// at `create()` and nothing revalidated them on the way back in.
///
/// The bounds are the ones `create()` already applies, so this refuses exactly
/// the configurations `create()` refuses and no more. `bits` is 1 to 8 because
/// a code is one byte a subvector. `subvectors` must be positive, must divide
/// `dim` and must not exceed it, because `sub_dim` is `dim / subvectors` and a
/// subvector of no values encodes nothing.
fn validate_quantization_fields(
    file: &str,
    config: &QuantizationPersistence,
    dim: usize,
) -> Result<(), Error> {
    if config.bits < 1 || config.bits > 8 {
        return Err(Error::BitsOutOfRangeInFile {
            file: file.to_string(),
            bits: config.bits,
        });
    }
    if config.subvectors == 0 {
        return Err(Error::SubvectorsZeroInFile {
            file: file.to_string(),
        });
    }
    if config.subvectors > dim || !dim.is_multiple_of(config.subvectors) {
        return Err(Error::SubvectorsInvalidInFile {
            file: file.to_string(),
            subvectors: config.subvectors,
            dim,
        });
    }
    Ok(())
}

/// Load quantization configuration and the codebook that goes with it
fn load_quantization(
    path: &Path,
    manifest: &IndexManifest,
    dim: usize,
    space: &str,
    records: usize,
) -> Result<Option<QuantizationArtefacts>, Error> {
    debug!(target: LOG_TARGET, "Loading quantization components...");

    let quant_path = path.join("quantization.json");
    if !manifest_names(manifest, "quantization.json") {
        debug!(target: LOG_TARGET, "manifest.json does not list quantization.json (non-quantized index)");
        return Ok(None);
    }

    let quant_data = read_artefact_string(path, "quantization.json", manifest)?;

    // Which of the two layouts the file takes, read off its `type` field
    // before either struct is parsed, so the product quantized struct still
    // parses a product quantized file and reports its errors word for word.
    // A file that is not an object, or names no type, is left to that
    // struct's parse, which refuses it as it always did.
    let is_int8 = serde_json::from_str::<Value>(&quant_data)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(Value::as_str)
                .map(|t| t == "int8")
        })
        .unwrap_or(false);
    if is_int8 {
        return load_int8_quantization(path, manifest, dim, space, &quant_path, &quant_data)
            .map(Some);
    }

    // A directory whose config.json names the inner product space and whose
    // manifest names quantization.json describes an index `create()` refuses.
    // No save this build makes can produce one, so it was hand assembled, and
    // building it would give an index ranking by the wrong quantity.
    validate_space_supports_quantization(space, &format!("{}: ", path.display()))?;

    let quant_config: QuantizationPersistence =
        serde_json::from_str(&quant_data).map_err(|e| Error::ArtefactParseFailed {
            name: "quantization.json",
            error: e.to_string(),
        })?;

    // The two fields that size the codebook, held to the rules `create()`
    // applies to them.
    //
    // `PQ::new` allocates `subvectors * 2^bits * (dim / subvectors)` floats
    // from these two values and nothing checked either. `bits: 40` asked for
    // 2^40 centroids and **aborted the process**, `subvectors: 2^40` aborted on
    // the outer vector, and `subvectors: 0` divided by zero. `create()` refuses
    // all three, so a directory carrying one was hand edited or written by a
    // release that did not validate its own input, and either way the file does
    // not describe an index this build can rebuild.
    validate_quantization_fields(&quant_path.display().to_string(), &quant_config, dim)?;

    debug!(target: LOG_TARGET, "quantization.json loaded");

    let shape = (quant_config.subvectors, 1usize << quant_config.bits);
    let centroids = load_pq_centroids(path, manifest, shape)?;
    let codes = load_pq_codes(path, manifest, records)?;

    Ok(Some(QuantizationArtefacts {
        config: QuantizationFile::Pq(quant_config),
        centroids,
        codes,
        int8: None,
        rows: Int8Rows::default(),
    }))
}

/// quantization.json under `type: int8`, with the scales and the rows a
/// trained directory carries beside it.
///
/// The fields are held to the rules `create()` applies to a scalar
/// declaration, and each artefact to the bounds its frame states: the
/// scales artefact carries `dim` entries and exactly `dim * 4` payload
/// bytes, every scale finite and positive, and the rows artefact carries
/// `entries` records of a `u32` internal id and a row of the codec's width,
/// ids strictly increasing and none above the saved id counter, in exactly
/// `entries * (4 + width)` bytes. Every bound is checked before anything is
/// allocated from a field.
fn load_int8_quantization(
    path: &Path,
    manifest: &IndexManifest,
    dim: usize,
    space: &str,
    quant_path: &Path,
    quant_data: &str,
) -> Result<QuantizationArtefacts, Error> {
    let file = quant_path.display().to_string();
    let config: Int8Persistence =
        serde_json::from_str(quant_data).map_err(|e| Error::ArtefactParseFailed {
            name: "quantization.json",
            error: e.to_string(),
        })?;
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: file.clone(),
        detail,
    };
    if crate::collection::Int8Scale::from_name(&config.scale).is_none() {
        return Err(invalid(format!(
            "scale is '{}', and this build fits '{}' alone",
            config.scale,
            crate::collection::Int8Scale::PER_DIMENSION
        )));
    }
    if config.training_size < 1000 {
        return Err(invalid(format!(
            "training_size is {}, and create() requires at least 1000",
            config.training_size
        )));
    }
    if let Some(max_training) = config.max_training_vectors {
        if max_training < config.training_size {
            return Err(invalid(format!(
                "max_training_vectors is {} and training_size is {}, and create() requires \
                 the first to be at or above the second",
                max_training, config.training_size
            )));
        }
    }
    let storage_mode = StorageMode::from_string(&config.storage_mode).map_err(Error::Engine)?;
    if storage_mode == StorageMode::QuantizedWithRaw {
        return Err(invalid(
            "storage_mode is 'quantized_with_raw', which a scalar quantized index never takes"
                .to_string(),
        ));
    }
    debug!(target: LOG_TARGET, "quantization.json loaded (int8)");

    if !config.is_trained {
        return Ok(QuantizationArtefacts {
            config: QuantizationFile::Int8(config),
            centroids: None,
            codes: HashMap::new(),
            int8: None,
            rows: Int8Rows::default(),
        });
    }

    let codec = load_int8_scales(path, manifest, dim)?;
    let tail = if space == "cosine" { 4 } else { 0 };
    let rows = load_int8_rows(path, manifest, codec.dim() + tail)?;
    Ok(QuantizationArtefacts {
        config: QuantizationFile::Int8(config),
        centroids: None,
        codes: HashMap::new(),
        int8: Some(codec),
        rows,
    })
}

/// The scales artefact, held to its bounds and handed back as the codec.
fn load_int8_scales(
    path: &Path,
    manifest: &IndexManifest,
    dim: usize,
) -> Result<Arc<Int8Codec>, Error> {
    if !manifest_names(manifest, INT8_SCALES_FILENAME) {
        return Err(Error::Int8ScalesMissing);
    }
    debug!(target: LOG_TARGET, "Loading {}...", INT8_SCALES_FILENAME);
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: INT8_SCALES_FILENAME.to_string(),
        detail,
    };
    // The bound is the frame plus `dim` floats, since the frame's own header
    // is held to the same count below.
    let max_bytes = (dim as u64)
        .saturating_mul(4)
        .saturating_add(FRAME_OVERHEAD_BYTES as u64);
    let bytes = zeusdb_vector_core::read_artefact(
        path,
        INT8_SCALES_FILENAME,
        manifest,
        INT8_SCALES_CONTENTS,
        max_bytes,
    )?;
    let framed = unframe(&bytes, FrameKind::Int8Scales, INT8_SCALES_FILENAME)?;
    if framed.entries != dim as u64 {
        return Err(invalid(format!(
            "holds {} scales and config.json declares dim {}",
            framed.entries, dim
        )));
    }
    if framed.payload.len() != dim * 4 {
        return Err(invalid(format!(
            "holds {} payload bytes and {} scales take {}",
            framed.payload.len(),
            dim,
            dim * 4
        )));
    }
    let scales: Vec<f32> = framed
        .payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let codec = Int8Codec::from_scales(scales).map_err(invalid)?;
    debug!(target: LOG_TARGET, "{} loaded ({} scales)", INT8_SCALES_FILENAME, dim);
    Ok(Arc::new(codec))
}

/// The rows artefact, held to its bounds, as (internal id, row) ascending.
/// Empty where the manifest names none, which is what a trained index
/// holding no record writes.
fn load_int8_rows(
    path: &Path,
    manifest: &IndexManifest,
    row_width: usize,
) -> Result<Int8Rows, Error> {
    if !manifest_names(manifest, INT8_ROWS_FILENAME) {
        return Ok(Int8Rows::default());
    }
    debug!(target: LOG_TARGET, "Loading {}...", INT8_ROWS_FILENAME);
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: INT8_ROWS_FILENAME.to_string(),
        detail,
    };
    let stride = 4 + row_width;
    // The bound is the id ceiling's worth of rows, since an internal id is
    // a `u32` and every row below is held to that ceiling.
    let max_bytes = (u32::MAX as u64)
        .saturating_mul(stride as u64)
        .saturating_add(FRAME_OVERHEAD_BYTES as u64);
    let bytes = zeusdb_vector_core::read_artefact(
        path,
        INT8_ROWS_FILENAME,
        manifest,
        INT8_ROWS_CONTENTS,
        max_bytes,
    )?;
    let framed = unframe(&bytes, FrameKind::Int8Rows, INT8_ROWS_FILENAME)?;
    let entries = usize::try_from(framed.entries)
        .ok()
        .filter(|&entries| entries <= u32::MAX as usize)
        .ok_or_else(|| {
            invalid(format!(
                "names {} rows, above the id ceiling",
                framed.entries
            ))
        })?;
    let expected = entries
        .checked_mul(stride)
        .ok_or_else(|| invalid(format!("names {} rows, which overflows", entries)))?;
    if framed.payload.len() != expected {
        return Err(invalid(format!(
            "holds {} payload bytes and {} rows of {} bytes take {}",
            framed.payload.len(),
            entries,
            stride,
            expected
        )));
    }
    let mut rows = Int8Rows::with_capacity(entries, row_width);
    let mut previous: Option<u32> = None;
    for chunk in framed.payload.chunks_exact(stride) {
        let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if previous.is_some_and(|last| id <= last) {
            return Err(invalid(format!(
                "internal id {} follows {}, and the ids are strictly increasing",
                id,
                previous.unwrap_or(0)
            )));
        }
        previous = Some(id);
        rows.push(id, &chunk[4..]);
    }
    debug!(target: LOG_TARGET, "{} loaded ({} rows)", INT8_ROWS_FILENAME, rows.len());
    Ok(rows)
}

// ============================================================================
// MAIN PERSISTENCE INTERFACE
// ============================================================================

/// Write every artefact except the graph dump and the manifest into `dir`
///
/// `dir` is the staging directory `StagingDir` opened, never the target, so
/// nothing here can leave a half written file where a reader will find it.
///
/// The manifest is no longer written here. It is written last of all, after the
/// graph dump, because it now records a length and a digest per artefact and
/// cannot do that for a file that does not exist yet. That ordering used to be
/// impossible: a save that failed at the dump would have left a directory with
/// no manifest at all. Staging makes it safe, because a save that fails at any
/// point leaves the previous directory untouched and the staging directory
/// removed.
pub(crate) fn save_index(index: &Collection, dir: &Path) -> Result<SaveLedger, Error> {
    let mut ledger = SaveLedger::default();

    // Save components in order of complexity (simple -> complex)
    save_config(index, dir, &mut ledger)?;
    save_mappings(index, dir, &mut ledger)?;
    save_metadata(index, dir, &mut ledger)?;

    // Save quantization components if enabled
    if index.has_quantization() {
        save_quantization_config(index, dir, &mut ledger)?;

        if index.can_use_quantization() {
            if index.declares_int8() {
                save_int8_scales(index, dir, &mut ledger)?;
                save_int8_rows(index, dir, &mut ledger)?;
            } else {
                save_pq_centroids(index, dir, &mut ledger)?;
                save_pq_codes(index, dir, &mut ledger)?;
            }
        }
    }

    // Save vectors based on storage mode
    save_vectors(index, dir, &mut ledger)?;

    // The sparse space, where one is declared, under `spaces/<name>/`.
    save_spaces(index, dir, &mut ledger)?;

    Ok(ledger)
}

/// Write the sparse space's artefacts under `spaces/<name>/`, where the
/// collection declares one.
///
/// The postings are written by the index itself through the seam, and the
/// dictionary by the collection, since the text layer is the collection's.
/// Each is recorded in the manifest by its length alone, because both are
/// framed and the frame's payload checksum is what the reader verifies; see
/// `zeusdb_vector_core::frame`. The two guards are taken one at a time and
/// never together.
fn save_spaces(index: &Collection, dir: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    let Some((name, space)) = index.sparse_named() else {
        return Ok(());
    };
    let prefix = Collection::space_prefix(name);
    debug!(target: LOG_TARGET, "Saving {}postings.zdbsparse...", prefix);
    space.index.read().unwrap().write(&prefix, dir, ledger)?;
    debug!(target: LOG_TARGET, "{}postings.zdbsparse saved", prefix);
    if let Some(text) = &space.text {
        let dictionary_name = Collection::dictionary_name(&prefix);
        debug!(target: LOG_TARGET, "Saving {}...", dictionary_name);
        let bytes = text.dictionary.read().unwrap().encode();
        zeusdb_vector_core::write_artefact(dir, &dictionary_name, &bytes)?;
        ledger.record_digest(&dictionary_name, bytes.len() as u64, None);
        debug!(target: LOG_TARGET, "{} saved ({} bytes)", dictionary_name, bytes.len());
    }
    Ok(())
}

// ============================================================================
// RECONSTRUCTION FUNCTIONS
// ============================================================================

/// Reconstruct Collection using Simple Reconstruction
#[allow(clippy::too_many_arguments)]
fn reconstruct_index_simple(
    path: &Path,
    manifest: &IndexManifest,
    config: IndexConfig,
    mappings: IdMappings,
    metadata: HashMap<String, HashMap<String, Value>>,
    mut vectors: RawVectors,
    quantization: Option<QuantizationArtefacts>,
    sparse: Option<SparseDeclaration>,
    dump_bytes: Option<u64>,
) -> Result<(Collection, bool), Error> {
    debug!(target: LOG_TARGET, "Creating empty index with loaded configuration...");

    // Step 1: Create empty index with loaded config
    let mut index = Collection::new_empty(
        config.dim,
        config.space.clone(),
        config.m,
        config.ef_construction,
        config.expected_size,
        config.indexed_fields.clone(),
        sparse,
    );

    debug!(target: LOG_TARGET, "Restoring data fields...");

    // The codes are needed twice, once to rebuild the graph for records that
    // have no raw vector and once to restore the stored codes afterwards.
    let mut quantization = quantization;
    let pq_codes = quantization
        .as_ref()
        .map(|q| q.codes.clone())
        .unwrap_or_default();
    let training_state = quantization
        .as_ref()
        .map(|q| TrainingState::from(&q.config));
    // The scalar rows, taken out rather than cloned, since the artefact is
    // the largest thing the loader holds after the dump.
    let int8_trained = quantization.as_ref().is_some_and(|q| q.int8.is_some());
    let int8_rows: Int8Rows = quantization
        .as_mut()
        .map(|q| std::mem::take(&mut q.rows))
        .unwrap_or_default();

    // Step 2: Restore all data fields directly (but not the graph)
    let mapped_records = mappings.id_map.len();
    restore_data_fields(&mut index, mappings, &config, quantization)?;

    // Step 2a: The sparse space, once the mappings are in and before the
    // graph, for the reasons `restore_spaces` gives.
    restore_spaces(&index, path, manifest)?;

    // The rows of a trained scalar directory, held to the mappings both
    // ways before anything is built from either: every row names a record
    // the mappings hold, and every record the mappings hold has a row. A
    // directory whose rows and mappings disagree describes two indexes.
    if int8_trained {
        check_int8_rows_against_mappings(&index, &int8_rows)?;
    }

    // Step 3: Restore the graph the save wrote. Every save dumps the graph, so
    // the ordinary path is a read. A dump that is absent, damaged, or written
    // by a release this build cannot interpret falls back to the rebuild, which
    // is the path that also upgrades a graph carrying a defect the vendored
    // patches have since fixed. See `restore_graph_from_dump`.
    let graph_rebuilt = match index.restore_graph_from_dump(path, config.id_counter, dump_bytes) {
        Ok(nodes) => {
            debug!(target: LOG_TARGET, "HNSW graph restored from the saved dump ({} nodes)", nodes);
            false
        }
        Err(reason) => {
            debug!(target: LOG_TARGET, "Rebuilding the HNSW graph, because {}", reason);
            // Both rebuilds below replay the raw vectors, so a directory
            // whose vectors were counted and not kept reads them now.
            vectors.hold(path, manifest, mapped_records)?;
            let empty = HashMap::new();
            let held = vectors.held().unwrap_or(&empty);
            if int8_trained {
                debug!(target: LOG_TARGET, "Rebuilding scalar quantized HNSW graph from the stored rows...");
                let inserted = index
                    .rebuild_graph_from_int8_rows(&int8_rows)
                    .map_err(Error::Engine)?;
                debug!(target: LOG_TARGET, "Scalar quantized graph rebuilt ({} rows inserted)", inserted);
            } else if index.can_use_quantization() {
                debug!(target: LOG_TARGET, "Rebuilding quantized HNSW graph from stored PQ codes...");
                rebuild_graph_from_codes(&mut index, &pq_codes, held)?;
            } else {
                debug!(target: LOG_TARGET, "Rebuilding HNSW graph from vectors...");
                rebuild_graph_from_data(&mut index, held, &pq_codes)?;
            }
            true
        }
    };
    // An empty directory holds no dump to read, since a save of an empty
    // index writes none, so the rebuild that runs over no records is the
    // ordinary path rather than a graph that could not be read.
    let graph_rebuilt = graph_rebuilt && index.vector_count() > 0;

    // Step 4: The stored record data, exactly as it was written, once the
    // graph is back. Neither rebuild touches the storage maps, and the
    // columns are built over the id store, which either rebuild may have added
    // an id to for a record the mappings did not name.
    let raw_count = vectors.len();
    let code_count = pq_codes.len();

    // A trained quantized_only index holds no raw vectors, but a directory
    // written before that was true carries its training records in
    // vectors.bin. They are dropped here rather than restored, so an old
    // directory sheds them on load exactly as a live index sheds them at
    // training. Only a vector whose record also has stored codes is dropped;
    // a raw vector without codes is the record's sole copy, which only a
    // directory that lost pq_codes.bin while keeping vectors.bin can contain,
    // and the count check below is what judges that case. The restored record
    // count is unaffected because every dropped vector's record keeps its
    // codes.
    let quantized_only_trained = index.can_use_quantization()
        && index
            .quantization_config()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedOnly);
    let vectors = if quantized_only_trained {
        let (kept, dropped): (HashMap<_, _>, HashMap<_, _>) = vectors
            .into_held()
            .into_iter()
            .partition(|(id, _)| !pq_codes.contains_key(id));
        if !dropped.is_empty() {
            debug!(target: LOG_TARGET, "Released {} raw training vectors quantized_only no longer keeps",
                dropped.len()
            );
        }
        RawVectors::Held(kept)
    } else {
        vectors
    };
    index.restore_storage_maps(pq_codes, metadata);

    // The raw vectors of a `quantized_with_raw` index go back into the store
    // beside the codes, addressed by the node numbering the restored graph
    // carries. A raw index needs none of this: its raw vectors came back with
    // the graph dump, which is the only place they were written.
    //
    // **Opening the store is not conditional on there being anything to put in
    // it.** This used to also require `!vectors.is_empty()`, so a trained
    // `quantized_with_raw` index holding no records at save time came back
    // without a store at all. It still reported `quantized_with_raw` and
    // `quantized_active`, and every record added after the load lost its raw
    // vector permanently: `get_records` fell through to the PQ reconstruction
    // and the rescoring the mode exists for had nothing true to rescore
    // against. Two ordinary sequences reach it, `clear()` before a save and
    // removing every record before a save.
    //
    // `clear()` already gets this right and opens the store on the replacement
    // graph for exactly this reason. This is the same rule on the load path.
    //
    // `restore_raw_store` handles the empty case itself: it sizes the store
    // from the graph's node count and pushes one vector per node, so at zero
    // nodes it opens an empty store and places nothing.
    if index.raw_store_is_expected() {
        let empty = HashMap::new();
        let placed = index
            .restore_raw_store(vectors.held().unwrap_or(&empty))
            .map_err(Error::RestoreRawFailed)?;
        debug!(target: LOG_TARGET, "{} raw vectors restored beside the codes", placed);
    }

    // Step 5: Put back the training collection the rebuild stripped
    if let Some(state) = training_state {
        state.apply(&mut index);
    }

    // Step 6: Check the saved count against the index that was actually built
    check_restored_count(&mut index, &config, raw_count, code_count)?;

    debug!(target: LOG_TARGET, "Reconstruction completed!");
    Ok((index, graph_rebuilt))
}

/// Reconcile the stored vector count with the records that were restored
///
/// `vector_count` is written to config.json and was previously restored
/// verbatim, so it could report records the directory no longer contains. The
/// count is derived here from the restored data and asserted against the saved
/// value. They agree for every directory whose files are intact, so a
/// disagreement means a file is missing or truncated and the load fails rather
/// than producing an index that misreports what it holds.
/// Hold the rows artefact to the mappings in both directions.
fn check_int8_rows_against_mappings(index: &Collection, rows: &Int8Rows) -> Result<(), Error> {
    let invalid = |detail: String| Error::Int8ArtefactInvalid {
        file: INT8_ROWS_FILENAME.to_string(),
        detail,
    };
    let ids = index.ids();
    if let Some((id, _)) = rows.iter().find(|&(id, _)| !ids.contains_slot(id)) {
        return Err(invalid(format!(
            "names internal id {}, which mappings.bin does not hold",
            id
        )));
    }
    if rows.len() != ids.len() {
        let held: std::collections::HashSet<usize> = rows.iter().map(|(id, _)| id).collect();
        let mut without: Vec<&str> = ids
            .iter()
            .filter(|(id, _)| !held.contains(id))
            .map(|(_, ext)| ext)
            .collect();
        without.sort_unstable();
        return Err(invalid(format!(
            "holds {} rows and mappings.bin holds {} records; record '{}' has no row",
            rows.len(),
            ids.len(),
            without.first().copied().unwrap_or("")
        )));
    }
    debug!(target: LOG_TARGET, "{} agrees with mappings.bin ({} rows)", INT8_ROWS_FILENAME, rows.len());
    Ok(())
}

fn check_restored_count(
    index: &mut Collection,
    config: &IndexConfig,
    raw_count: usize,
    code_count: usize,
) -> Result<(), Error> {
    let restored = index.count_stored_records();

    if restored != config.vector_count {
        return Err(Error::RestoredCountMismatch {
            restored,
            expected: config.vector_count,
            raw_count,
            code_count,
        });
    }

    index.set_vector_count(restored);
    debug!(target: LOG_TARGET, "Vector count verified against restored records: {}",
        restored
    );
    Ok(())
}

/// Restore all data fields to the index (everything except the HNSW graph)
fn restore_data_fields(
    index: &mut Collection,
    mappings: IdMappings,
    config: &IndexConfig,
    quantization: Option<QuantizationArtefacts>,
) -> Result<(), Error> {
    // Before the mappings move, because this reads their keys. The floor is
    // what stops an old directory reissuing a generated id it already holds.
    let generated_floor = Collection::highest_generated_id(mappings.id_map.keys());
    index.set_id_mappings(mappings.id_map, mappings.rev_map, config.id_counter)?;

    // The add() method will properly:
    // - Insert vectors into index.vectors
    // - Insert metadata into index.vector_metadata
    // - Update counters correctly
    // - Build the HNSW graph

    // Restore counters
    index.set_counters(config.id_counter, config.vector_count);
    index.set_generated_ids(config.generated_ids, generated_floor);

    // Restore index level metadata. Empty for a directory written before
    // config.json carried the field, which is what those directories held.
    if !config.metadata.is_empty() {
        index.add_metadata(config.metadata.clone())?;
        debug!(target: LOG_TARGET, "Index level metadata restored ({} entries)",
            config.metadata.len()
        );
    }

    // Restore quantization state if present
    if let Some(artefacts) = quantization {
        restore_quantization_state_simple(
            index,
            artefacts.config,
            artefacts.centroids,
            artefacts.int8,
        )?;
    }

    debug!(target: LOG_TARGET, "All data fields restored successfully");
    Ok(())
}

/// Install a codebook read from disk into a freshly built PQ instance
///
/// The shape check catches a codebook that belongs to a different index. The
/// all-zero check catches the one written by v0.3.0 through v0.4.1, which never
/// read pq_centroids.bin on load and so re-saved the zero codebook that
/// `PQ::new` starts with. Both fail the load rather than let the index come
/// back reporting itself trained while decoding every code to zeros.
fn install_centroids(pq: &PQ, centroids: Centroids) -> Result<(), Error> {
    let expected = (pq.subvectors(), pq.num_centroids(), pq.sub_dim());
    let actual = (
        centroids.len(),
        centroids.first().map(|s| s.len()).unwrap_or(0),
        centroids
            .first()
            .and_then(|s| s.first())
            .map(|c| c.len())
            .unwrap_or(0),
    );
    let uniform = centroids
        .iter()
        .all(|sub| sub.len() == actual.1 && sub.iter().all(|c| c.len() == actual.2));

    if actual != expected || !uniform {
        return Err(Error::CodebookShapeMismatch {
            actual,
            expected,
            subvectors: pq.subvectors(),
            bits: pq.bits(),
        });
    }

    if centroids
        .iter()
        .all(|sub| sub.iter().all(|c| c.iter().all(|&v| v == 0.0)))
    {
        return Err(Error::CodebookAllZero);
    }

    // Going through set_centroids rather than writing the field rebuilds the
    // symmetric distance table from the codebook that has just been read, so a
    // loaded index can build a graph on real distances exactly as a freshly
    // trained one does.
    pq.set_centroids(centroids).map_err(Error::Engine)
}

/// Restore quantization state (simplified for reconstruction)
fn restore_quantization_state_simple(
    index: &mut Collection,
    file: QuantizationFile,
    centroids: Option<Centroids>,
    int8: Option<Arc<Int8Codec>>,
) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Restoring quantization state...");

    let quant_data = match file {
        QuantizationFile::Pq(quant_data) => quant_data,
        QuantizationFile::Int8(quant_data) => {
            return restore_int8_state(index, quant_data, int8);
        }
    };

    // Convert QuantizationPersistence back to QuantizationConfig
    let storage_mode = StorageMode::from_string(&quant_data.storage_mode).map_err(Error::Engine)?;

    let quant_config = QuantizationConfig {
        scheme: QuantizationScheme::Pq {
            subvectors: quant_data.subvectors,
            bits: quant_data.bits,
        },
        training_size: quant_data.training_size,
        max_training_vectors: quant_data.max_training_vectors,
        storage_mode,
    };

    // Set quantization config
    index.set_quantization_config(Some(quant_config));

    // Restore what training measured about the rerank fetch. `None` here means
    // the directory was written before the calibration existed, and the search
    // falls back to the corpus terms. See `RerankCalibration`.
    index.set_rerank_calibration(quant_data.rerank_calibration);

    // Carried rather than restamped, so a load and a save do not move it. On a
    // directory written before the index held the value this is the save time
    // the old code wrote, which cannot be recovered but does at least stop
    // drifting from here on.
    index.set_training_completed_at(quant_data.training_completed_at);

    // The training ids and the threshold flag are applied after the graph
    // rebuild, which would otherwise strip them. See TrainingState.

    // Every quantized index needs a PQ instance, trained or not. Without one
    // maybe_trigger_training can never fire, so an index saved while still
    // collecting could reach the threshold again and still never train.
    let pq = Arc::new(PQ::new(
        index.dim(),
        quant_data.subvectors,
        quant_data.bits,
        quant_data.training_size,
        quant_data.max_training_vectors,
    ));

    if !quant_data.is_trained {
        index.set_pq(Some(pq));

        debug!(target: LOG_TARGET, "Quantization state restored (untrained, {} collected training IDs)",
            quant_data.training_ids.len()
        );
    } else {
        // The codebook is what makes a trained PQ trained. Without it the
        // instance would report itself trained while holding the zeros that
        // PQ::new starts with, and every reconstruction would return them.
        let centroids = centroids.ok_or(Error::CentroidsMissing)?;
        install_centroids(&pq, centroids)?;

        pq.set_trained(true);
        index.set_pq(Some(pq));

        debug!(target: LOG_TARGET, "Quantization state restored (trained, codebook loaded, {} training IDs)",
            quant_data.training_ids.len()
        );
    }

    Ok(())
}

/// The scalar counterpart of `restore_quantization_state_simple`: the
/// declaration, the stamp, and the codec where the file says it was fitted.
fn restore_int8_state(
    index: &mut Collection,
    quant_data: Int8Persistence,
    int8: Option<Arc<Int8Codec>>,
) -> Result<(), Error> {
    let storage_mode = StorageMode::from_string(&quant_data.storage_mode).map_err(Error::Engine)?;
    let scale = crate::collection::Int8Scale::from_name(&quant_data.scale).ok_or_else(|| {
        Error::Int8ArtefactInvalid {
            file: "quantization.json".to_string(),
            detail: format!("scale is '{}'", quant_data.scale),
        }
    })?;
    index.set_quantization_config(Some(QuantizationConfig {
        scheme: QuantizationScheme::Int8 { scale },
        training_size: quant_data.training_size,
        max_training_vectors: quant_data.max_training_vectors,
        storage_mode,
    }));
    index.set_training_completed_at(quant_data.training_completed_at);
    if quant_data.is_trained {
        let codec = int8.ok_or(Error::Int8ScalesMissing)?;
        index.set_int8_codec(codec)?;
        debug!(target: LOG_TARGET, "Quantization state restored (int8, trained, scales loaded, {} training IDs)",
            quant_data.training_ids.len()
        );
    } else {
        debug!(target: LOG_TARGET, "Quantization state restored (int8, untrained, {} collected training IDs)",
            quant_data.training_ids.len()
        );
    }
    Ok(())
}

/// Rebuild the graph for a trained quantized index from its stored codes
///
/// The saved graph was a PQ graph over the codes, so the rebuild inserts those
/// same codes into a fresh PQ graph rather than reconstructing vectors and
/// replaying them through the raw add() path. The loaded index therefore
/// reports `is_quantized()` true and `quantized_active`, searches through ADC
/// exactly as the saved one did, and never holds a reconstructed vector at
/// full width. The internal ids come from mappings.bin, so no id is reassigned
/// and the counters stay as saved.
fn rebuild_graph_from_codes(
    index: &mut Collection,
    pq_codes: &HashMap<String, Vec<u8>>,
    vectors: &HashMap<String, Vec<f32>>,
) -> Result<(), Error> {
    let (inserted, quantized_from_raw, remapped) = index
        .rebuild_graph_from_codes(pq_codes, vectors)
        .map_err(Error::Engine)?;

    if quantized_from_raw > 0 {
        debug!(target: LOG_TARGET, "{} records had a raw vector and no stored PQ codes and were quantized \
             through the loaded codebook",
            quantized_from_raw
        );
    }
    if remapped > 0 {
        debug!(target: LOG_TARGET, "{} records were missing from mappings.bin and were assigned fresh \
             internal ids",
            remapped
        );
    }
    debug!(target: LOG_TARGET, "Quantized graph rebuilt ({} records inserted from stored PQ codes)",
        inserted
    );
    Ok(())
}

/// Rebuild the graph from the records the directory holds, under the ids the
/// mappings name
///
/// This is the path for an index that is not trained, meaning one saved with no
/// quantization at all or one saved while still collecting training vectors.
/// A record that has a raw vector is replayed from it. A record that has only
/// PQ codes is reconstructed through the codebook, which is what `get_records`
/// already does for the same record while the index is live, so the graph is
/// built at the fidelity the storage mode already delivers rather than losing
/// the record. The codes themselves are restored as stored and are never
/// recomputed from a reconstruction. The metadata is not carried here; step 4
/// of the reconstruction writes it back from the file under the same ids.
fn rebuild_graph_from_data(
    index: &mut Collection,
    vectors: &HashMap<String, Vec<f32>>,
    pq_codes: &HashMap<String, Vec<u8>>,
) -> Result<(), Error> {
    if vectors.is_empty() && pq_codes.is_empty() {
        debug!(target: LOG_TARGET, "No records to rebuild (empty index)");
        return Ok(());
    }

    let mut records: Vec<(String, Vec<f32>)> = Vec::with_capacity(vectors.len() + pq_codes.len());
    let mut reconstructed = 0usize;

    // Every record with a raw vector, replayed from it
    for (ext_id, vector) in vectors.iter() {
        records.push((ext_id.clone(), vector.clone()));
    }

    // Every record that has codes and no raw vector, reconstructed
    let code_only: Vec<&String> = pq_codes
        .keys()
        .filter(|id| !vectors.contains_key(*id))
        .collect();

    if !code_only.is_empty() {
        let pq = index.pq().cloned().ok_or(Error::CodesWithoutCodebook {
            count: code_only.len(),
        })?;

        for ext_id in code_only {
            let codes = &pq_codes[ext_id];
            let vector = pq
                .reconstruct(codes)
                .map_err(|e| Error::ReconstructFailed {
                    id: ext_id.clone(),
                    codes: codes.len(),
                    error: e,
                })?;
            records.push((ext_id.clone(), vector));
            reconstructed += 1;
        }
    }

    debug!(target: LOG_TARGET, "Prepared {} records for the graph rebuild ({} replayed from raw vectors, {} reconstructed from PQ codes)",
        records.len(),
        records.len() - reconstructed,
        reconstructed
    );
    debug!(target: LOG_TARGET, "Rebuilding the graph from the restored records...");

    // The records are owned Rust and go straight into a fresh graph, in
    // ascending internal id, which `rebuild_from_records` orders them into
    // from the restored mappings. This used to build a PyDict holding three
    // PyLists and call add(), which parsed them back into exactly this.
    let inserted = index.rebuild_from_records(records)?;

    debug!(target: LOG_TARGET, "Graph rebuild completed: {} records inserted", inserted);
    debug!(target: LOG_TARGET, "Final vector count: {}", index.vector_count());

    Ok(())
}

// ============================================================================
// LOAD INTERFACE
// ============================================================================

/// Load a Collection from a directory structure (Approach B: Simple Reconstruction)
///
/// Reached through `Collection::load`, which the binding registers as
/// `_load_index`. `VectorDatabase.load(path)` is the documented route and is
/// a one line pass through to that.
pub(crate) fn load_index(
    path: &str,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    policy: JournalPolicy,
) -> Result<(Collection, Recovery), Error> {
    debug!(target: LOG_TARGET, "Starting index load with reconstruction from: {}", path);

    let path_buf = Path::new(path);

    // A save killed between its two renames left the whole index beside the
    // target and nothing at it. Recovery from a killed process is a load, so
    // it is put back here, before the directory is looked for. See
    // `restore_replaced`.
    let restored_from_aside = restore_replaced(path_buf)?;

    // Validate directory exists
    if !path_buf.exists() {
        return Err(Error::IndexDirectoryNotFound {
            path: path.to_string(),
        });
    }

    // Phase 1: Load all ZeusDB components
    debug!(target: LOG_TARGET, "Phase 1: Loading ZeusDB components...");

    let manifest = load_manifest(path_buf)?;
    let major = check_format_version(&manifest.format_version)?;

    // A manifest below the major that names a journal is a directory
    // assembled by hand, since no release writing that format wrote the
    // field. The same rule the sparse space's declaration takes below.
    if major < JOURNAL_FORMAT_MAJOR && manifest.journal.is_some() {
        return Err(Error::FormatVersionJournal {
            format_version: manifest.format_version.clone(),
        });
    }

    // Before any artefact is read, so a directory missing two files names the
    // first rather than failing on whichever one a partial load reaches.
    check_files_present(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Manifest loaded: {} vectors, format v{}",
        manifest.total_vectors, manifest.format_version
    );

    let config = load_config(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Config loaded: dim={}, space={}", config.dim, config.space);

    // A 1.x directory declares no space, since no release writing 1.x held
    // one, and a 1.x manifest over a config that declares one is a
    // directory assembled by hand.
    if major < 2 && !config.spaces.is_empty() {
        return Err(Error::FormatVersionSpaces {
            format_version: manifest.format_version.clone(),
        });
    }
    let space_record = config.spaces.first();
    let sparse_tokenizer = resolve_tokenizer(space_record, tokenizer)?;
    let sparse = space_record.map(|record| SparseDeclaration {
        name: SpaceName::new(&record.name).expect("validated by load_config"),
        config: record.index.clone(),
        tokenizer: sparse_tokenizer,
    });

    let mappings = load_mappings(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Mappings loaded: {} ID mappings", mappings.id_map.len());

    let metadata = load_metadata(path_buf, &manifest)?;
    debug!(target: LOG_TARGET, "Metadata loaded: {} records", metadata.len());

    // A raw index takes its vectors back from the graph dump, so its
    // vectors.bin is walked for its count and its check and nothing of it is
    // kept. See `RawVectors`.
    let vectors = if manifest_names(&manifest, "quantization.json") {
        RawVectors::Held(load_vectors(path_buf, &manifest, mappings.id_map.len())?)
    } else {
        RawVectors::Counted(count_vectors(path_buf, &manifest, mappings.id_map.len())?)
    };
    debug!(target: LOG_TARGET, "Vectors loaded: {} vectors", vectors.len());

    let quantization = load_quantization(
        path_buf,
        &manifest,
        config.dim,
        &config.space,
        mappings.id_map.len(),
    )?;
    if let Some(ref quant) = quantization {
        match &quant.config {
            QuantizationFile::Pq(config) => {
                debug!(target: LOG_TARGET, "Quantization loaded: {} subvectors, trained={}, codebook={}",
                    config.subvectors,
                    config.is_trained,
                    if quant.centroids.is_some() {
                        "present"
                    } else {
                        "absent"
                    }
                );
            }
            QuantizationFile::Int8(config) => {
                debug!(target: LOG_TARGET, "Quantization loaded: int8, scale={}, trained={}, scales={}, rows={}",
                    config.scale,
                    config.is_trained,
                    if quant.int8.is_some() {
                        "present"
                    } else {
                        "absent"
                    },
                    quant.rows.len()
                );
            }
        }
    }

    // The graph dump itself is read inside the reconstruction, which needs the
    // mappings and the codebook first to judge it against.

    // Phase 2: Create empty index and restore state
    debug!(target: LOG_TARGET, "Phase 2: Creating empty index and restoring state...");
    let (mut restored_index, graph_rebuilt) = reconstruct_index_simple(
        path_buf,
        &manifest,
        config,
        mappings,
        metadata,
        vectors,
        quantization,
        sparse,
        recorded_dump_length(&manifest, GRAPH_DUMP_FILENAME),
    )?;

    // `new_empty` stamps the load time, because it has nothing better to start
    // from. Until this ran, a save of a loaded index wrote that load time to
    // manifest.json as the creation, so a directory that had been through one
    // load and save claimed to have been created then.
    restored_index.set_created_at(manifest.created_at);

    debug!(target: LOG_TARGET, "Index reconstruction completed successfully!");

    // Phase 3: The journal beside the directory, where the manifest names
    // one. Nothing below runs for a directory saved without one, so such a
    // directory loads exactly as it did.
    let Some(record) = manifest.journal else {
        let recovery = Recovery::unjournaled(restored_from_aside, graph_rebuilt);
        recovery.report(path);
        return Ok((restored_index, recovery));
    };

    let Some(directory_id) = crate::journal::collection_id_from_hex(&record.collection_id) else {
        return Err(Error::JournalManifestInvalid {
            detail: format!(
                "its collection_id is '{}', which is not 32 hexadecimal digits",
                record.collection_id
            ),
        });
    };
    restored_index.set_collection_id(directory_id);
    restored_index.set_journal_sequence(record.sequence);

    let durability = match policy {
        JournalPolicy::CheckpointOnly => {
            warn!(target: LOG_TARGET, operation = "load_checkpoint_only",
                directory = path,
                journal = %record.file,
                sequence = record.sequence,
                "Opened the checkpoint alone; every mutation after the sequence it holds is in \
                 the journal beside it and was not applied"
            );
            let mut recovery = Recovery::unjournaled(restored_from_aside, graph_rebuilt);
            recovery.checkpoint_sequence = record.sequence;
            recovery.report(path);
            return Ok((restored_index, recovery));
        }
        JournalPolicy::Replay(durability) => durability,
    };

    let wal = crate::journal::journal_path(path_buf)?;
    if !wal.exists() {
        return Err(Error::JournalMissing {
            directory: path.to_string(),
            file: wal.display().to_string(),
            recorded: record.file.clone(),
            sequence: record.sequence,
        });
    }
    let file = wal.display().to_string();
    let bytes = crate::journal::read_journal_bytes(&wal)?;
    let contents = zeusdb_vector_core::read_journal(&bytes, &file)?;
    crate::journal::check_contents(&contents, &file, directory_id, record.sequence)?;

    // Every record above the sequence the checkpoint holds, applied at the
    // values it names. `apply` refuses to run with a sink attached, which is
    // why the sink goes on after this loop and not before it.
    let dim = restored_index.dim();
    let mut replayed = 0usize;
    let mut skipped = 0usize;
    for journal_record in &contents.records {
        if journal_record.sequence <= record.sequence {
            skipped += 1;
            continue;
        }
        let operation = zeusdb_vector_core::Operation::decode(journal_record, dim, &file)?;
        restored_index
            .apply(operation)
            .map_err(|error| Error::JournalReplayFailed {
                file: file.clone(),
                sequence: journal_record.sequence,
                detail: error.to_string(),
            })?;
        replayed += 1;
    }

    // The journal is cut back to its last whole record and reopened for
    // append after it. Where the body is empty the header is restated at the
    // checkpoint's sequence plus one, which completes a truncation a crash
    // left half done.
    let writer =
        zeusdb_vector_core::JournalWriter::open_for_append(&wal, &contents, record.sequence + 1)?;
    restored_index.attach_sink(Box::new(crate::journal::JournalSink::from_writer(
        writer, durability,
    )?));

    let recovery = Recovery {
        checkpoint_sequence: record.sequence,
        first_sequence: contents.header.first_sequence,
        records_in_journal: contents.records.len(),
        replayed,
        skipped,
        damage: contents.damage.clone(),
        good_bytes: contents.good_bytes,
        restored_from_aside,
        graph_rebuilt,
        journaled: true,
    };
    recovery.report(path);
    Ok((restored_index, recovery))
}

// ============================================================================
// INDIVIDUAL COMPONENT SAVERS
// ============================================================================

/// Save index configuration as JSON
fn save_config(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving config.json...");

    let config = IndexConfig {
        dim: index.dim(),
        //space: index.get_space().to_string(),
        space: index.metric().to_string(),
        m: index.m(),
        ef_construction: index.ef_construction(),
        expected_size: index.expected_size(),
        id_counter: index.id_counter(),
        vector_count: index.vector_count(),
        generated_ids: index.generated_ids(),
        metadata: index.all_metadata(),
        indexed_fields: index.indexed_fields(),
        spaces: index
            .sparse_named()
            .map(|(name, space)| SpaceRecord {
                name: name.as_str().to_string(),
                kind: SPARSE_KIND.to_string(),
                index: space.config().clone(),
                tokenizer: space.text.as_ref().map(|text| text.tokenizer.config()),
            })
            .into_iter()
            .collect(),
    };

    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| Error::SerializeFailed {
            what: "config",
            error: e.to_string(),
        })?;

    write_artefact(path, "config.json", config_json.as_bytes(), ledger)?;

    debug!(target: LOG_TARGET, "config.json saved");
    Ok(())
}

/// Save ID mappings using efficient binary format
fn save_mappings(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving mappings.bin...");

    // The two maps the file holds, built from the id store under its guard,
    // which ends with this block. The file's shape is the two maps it has
    // always held, and the store hands out the same pairs both ways, so a
    // reader of either map finds what it found before. The two maps used to
    // be cloned from the two the collection held, at the same cost.
    let mappings = {
        let ids = index.ids();
        let mut id_map = HashMap::with_capacity(ids.len());
        let mut rev_map = HashMap::with_capacity(ids.len());
        for (internal_id, id) in ids.iter() {
            id_map.insert(id.to_string(), internal_id);
            rev_map.insert(internal_id, id.to_string());
        }
        IdMappings { id_map, rev_map }
    };
    let mapping_count = mappings.id_map.len();

    let mappings_data =
        bincode::encode_to_vec(&mappings, bincode::config::standard()).map_err(|e| {
            Error::SerializeFailed {
                what: "mappings",
                error: e.to_string(),
            }
        })?;

    write_artefact(path, "mappings.bin", &mappings_data, ledger)?;

    debug!(target: LOG_TARGET, "mappings.bin saved ({} mappings)", mapping_count);
    Ok(())
}

/// Save vector metadata as JSON for external tool compatibility
fn save_metadata(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving metadata.json...");

    // Both guards end with the serialize, taken in the documented order,
    // the id store before vector_metadata. The file is keyed by external id
    // and the store by internal id, so every record is written under the id
    // the id store holds for it, in increasing internal id order, which is
    // the order the store walks in. `to_string_pretty` returns an owned
    // String, so the file is written with nothing held.
    let (metadata_json, record_count) = {
        let ids = index.ids();
        let vector_metadata = index.vector_metadata();
        let file = MetadataFile {
            records: ids
                .iter()
                .filter_map(|(slot, id)| vector_metadata.get(slot).map(|fields| (id, fields)))
                .collect(),
        };
        let json = serde_json::to_string_pretty(&file);
        (json, file.records.len())
    };
    let metadata_json = metadata_json.map_err(|e| Error::SerializeFailed {
        what: "metadata",
        error: e.to_string(),
    })?;

    write_artefact(path, "metadata.json", metadata_json.as_bytes(), ledger)?;

    debug!(target: LOG_TARGET, "metadata.json saved ({} records)", record_count);
    Ok(())
}

/// `metadata.json` as it is written: one object per record under its
/// external id, holding the record's fields. The same shape the file has
/// always had, which `load_metadata` reads back into a map of maps.
struct MetadataFile<'a> {
    records: Vec<(&'a str, RecordFields<'a>)>,
}

impl Serialize for MetadataFile<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(
            self.records
                .iter()
                .map(|(id, fields)| (id, FieldsObject(*fields))),
        )
    }
}

/// One record's fields as the JSON object its mapping serialises as.
struct FieldsObject<'a>(RecordFields<'a>);

impl Serialize for FieldsObject<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter())
    }
}

/// Save quantization configuration and training state
fn save_quantization_config(
    index: &Collection,
    path: &Path,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    if let Some(config) = index.quantization_config() {
        debug!(target: LOG_TARGET, "Saving quantization.json...");

        if let Some(scale) = config.scheme.int8_scale() {
            return save_int8_quantization_config(index, config, scale, path, ledger);
        }
        let (subvectors, bits) = config.scheme.pq_shape().unwrap_or((1, 8));

        // When the codebook was fitted, taken from the index rather than from
        // the clock. This used to be `Utc::now()`, so the field recorded the
        // save and moved every time a trained index was saved again. `None` on
        // an index that never trained, and also on one loaded from a directory
        // written before the index carried the value; see the field on
        // `Collection`.
        let training_completed_at = index.training_completed_at();

        // CAPTURE TRAINING STATE:
        let training_ids = index.training_ids().clone();
        let training_threshold_reached = index.training_threshold_reached();

        let (memory_stats, pq_config) = if let Some(pq) = index.pq() {
            let (memory_mb, total_centroids) = pq.get_memory_stats();

            let memory_stats = MemoryStats {
                centroid_storage_mb: memory_mb,
                compression_ratio: (pq.dim() * 4) as f64 / pq.subvectors() as f64,
                centroids_per_subvector: pq.num_centroids(),
                total_centroids,
            };

            let pq_config = PQConfig {
                dim: pq.dim(),
                sub_dim: pq.sub_dim(),
                num_centroids: pq.num_centroids(),
            };

            (Some(memory_stats), pq_config)
        } else {
            let pq_config = PQConfig {
                dim: index.dim(),
                sub_dim: index.dim() / subvectors,
                num_centroids: 1 << bits,
            };
            (None, pq_config)
        };

        let quant_persistence = QuantizationPersistence {
            r#type: "pq".to_string(),
            subvectors,
            bits,
            training_size: config.training_size,
            max_training_vectors: config.max_training_vectors,
            storage_mode: config.storage_mode.to_string().to_string(),
            is_trained: index.can_use_quantization(),
            training_completed_at,
            memory_stats,
            pq_config,
            training_ids,
            training_threshold_reached,
            rerank_calibration: index.rerank_calibration(),
        };

        let quant_json = serde_json::to_string_pretty(&quant_persistence).map_err(|e| {
            Error::SerializeFailed {
                what: "quantization config",
                error: e.to_string(),
            }
        })?;

        write_artefact(path, "quantization.json", quant_json.as_bytes(), ledger)?;

        //debug!(target: LOG_TARGET, "quantization.json saved");
        debug!(target: LOG_TARGET, "quantization.json saved with {} training IDs",
            quant_persistence.training_ids.len()
        );
    }
    Ok(())
}

/// quantization.json for a scalar quantized space: the declaration, the
/// training state and the clipped count, and none of the product quantized
/// fields.
fn save_int8_quantization_config(
    index: &Collection,
    config: &QuantizationConfig,
    scale: crate::collection::Int8Scale,
    path: &Path,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    // Each guard is taken alone and released before the next, in the
    // declared order: the index guard for the clipped count, then the
    // training ids. A temporary guard inside the struct literal would live
    // to the end of the statement, and the index guard taken after it would
    // invert the order, which the rank registry refuses.
    let saturated_values = index.int8_saturated();
    let training_ids = index.training_ids().clone();
    let persistence = Int8Persistence {
        r#type: "int8".to_string(),
        scale: scale.name().to_string(),
        training_size: config.training_size,
        max_training_vectors: config.max_training_vectors,
        storage_mode: config.storage_mode.to_string().to_string(),
        is_trained: index.can_use_quantization(),
        training_completed_at: index.training_completed_at(),
        training_ids,
        training_threshold_reached: index.training_threshold_reached(),
        saturated_values,
    };
    let quant_json =
        serde_json::to_string_pretty(&persistence).map_err(|e| Error::SerializeFailed {
            what: "quantization config",
            error: e.to_string(),
        })?;
    write_artefact(path, "quantization.json", quant_json.as_bytes(), ledger)?;
    debug!(target: LOG_TARGET, "quantization.json saved (int8) with {} training IDs",
        persistence.training_ids.len()
    );
    Ok(())
}

/// The scales artefact: `dim` little endian floats under a frame whose
/// entry count is `dim`. Recorded by length alone, as every framed artefact
/// is.
fn save_int8_scales(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    let Some(codec) = index.int8_codec() else {
        return Ok(());
    };
    debug!(target: LOG_TARGET, "Saving {}...", INT8_SCALES_FILENAME);
    let mut payload = Vec::with_capacity(codec.dim() * 4);
    for scale in codec.scales() {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    let bytes = frame(
        FrameKind::Int8Scales,
        FrameEncoding::Engine,
        codec.dim() as u64,
        &payload,
    );
    zeusdb_vector_core::write_artefact(path, INT8_SCALES_FILENAME, &bytes)?;
    ledger.record_digest(INT8_SCALES_FILENAME, bytes.len() as u64, None);
    debug!(target: LOG_TARGET, "{} saved ({} scales)", INT8_SCALES_FILENAME, codec.dim());
    Ok(())
}

/// The rows artefact: every live record's internal id and row, ascending,
/// written straight into the frame's buffer. Nothing is written for an index
/// holding no record, and the manifest then names no rows artefact.
fn save_int8_rows(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    let stride = 4 + index.quantization_code_bytes();
    let mut out = frame_begin(
        FrameKind::Int8Rows,
        FrameEncoding::Engine,
        index.len().saturating_mul(stride),
    );
    let entries = index.write_int8_rows(&mut out);
    if entries == 0 {
        return Ok(());
    }
    debug!(target: LOG_TARGET, "Saving {}...", INT8_ROWS_FILENAME);
    let bytes = frame_finish(out, entries as u64);
    zeusdb_vector_core::write_artefact(path, INT8_ROWS_FILENAME, &bytes)?;
    ledger.record_digest(INT8_ROWS_FILENAME, bytes.len() as u64, None);
    debug!(target: LOG_TARGET, "{} saved ({} rows)", INT8_ROWS_FILENAME, entries);
    Ok(())
}

/// Save PQ centroids for vector reconstruction
fn save_pq_centroids(
    index: &Collection,
    path: &Path,
    ledger: &mut SaveLedger,
) -> Result<(), Error> {
    if let Some(pq) = index.pq() {
        if pq.is_trained() {
            debug!(target: LOG_TARGET, "Saving pq_centroids.bin...");

            // The codebook is serialized inside the closure and written outside
            // it, so the lock is held for the encode alone. `bincode` returns an
            // owned buffer, so narrowing the guard this way copies nothing.
            let centroids_data = pq
                .with_centroids(|centroids| {
                    bincode::encode_to_vec(centroids, bincode::config::standard())
                })
                .map_err(|e| Error::SerializeFailed {
                    what: "PQ centroids",
                    error: e.to_string(),
                })?;

            write_artefact(path, "pq_centroids.bin", &centroids_data, ledger)?;

            debug!(target: LOG_TARGET, "pq_centroids.bin saved");
        }
    }
    Ok(())
}

/// Save quantized vector codes
fn save_pq_codes(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    // The guard ends with the serialize, so the file is written with nothing
    // held. `encode_to_vec` was already producing an owned buffer, so this
    // narrows the guard rather than adding a copy.
    let (codes_data, code_count) = {
        let pq_codes = index.pq_codes();
        if pq_codes.is_empty() {
            return Ok(());
        }
        debug!(target: LOG_TARGET, "Saving pq_codes.bin...");
        (
            bincode::encode_to_vec(&*pq_codes, bincode::config::standard()),
            pq_codes.len(),
        )
    };

    let codes_data = codes_data.map_err(|e| Error::SerializeFailed {
        what: "PQ codes",
        error: e.to_string(),
    })?;

    write_artefact(path, "pq_codes.bin", &codes_data, ledger)?;

    debug!(target: LOG_TARGET, "pq_codes.bin saved ({} vectors)", code_count);
    Ok(())
}

/// Save raw vectors based on storage mode configuration
///
/// The file is what it always was, a `HashMap` keyed by external id, so a
/// directory this release writes loads on the reader that read the old ones.
/// What changed is where the vectors come from: there is no raw vector map any
/// more, so they are read out of the store the graph is addressed against.
///
/// A trained `quantized_only` index writes none, as before, because it holds
/// none.
fn save_vectors(index: &Collection, path: &Path, ledger: &mut SaveLedger) -> Result<(), Error> {
    if !index.holds_raw_vectors() {
        return Ok(());
    }
    let (vectors_data, vector_count) = {
        let vectors = index.collect_raw_vectors();
        if vectors.is_empty() {
            return Ok(());
        }
        debug!(target: LOG_TARGET, "Saving vectors.bin...");
        (
            bincode::encode_to_vec(&vectors, bincode::config::standard()),
            vectors.len(),
        )
    };

    let vectors_data = vectors_data.map_err(|e| Error::SerializeFailed {
        what: "vectors",
        error: e.to_string(),
    })?;

    write_artefact(path, "vectors.bin", &vectors_data, ledger)?;

    debug!(target: LOG_TARGET, "vectors.bin saved ({} vectors)", vector_count);
    Ok(())
}

/// Write manifest.json, which is the last file a save writes
///
/// It names every other artefact, records the length and digest of each, and
/// records the directory size. All three are facts about files that are already
/// on disk, which is why it is written last and why there is no second pass to
/// correct any of them.
pub(crate) fn save_manifest(
    index: &Collection,
    path: &Path,
    ledger: SaveLedger,
) -> Result<(), Error> {
    debug!(target: LOG_TARGET, "Saving manifest.json...");

    // The manifest needs two facts about the stores rather than the stores
    // themselves, so it takes those two facts and releases both guards here.
    //
    // This used to hold the `vectors` and `pq_codes` read guards for the whole
    // function, and `get_storage_mode` below takes the graph's read guard. The
    // documented order is `hnsw < vectors < pq_codes`, so that acquired the
    // three in exactly the wrong order. It could not deadlock, because a save
    // holds the mutation lock and every path that takes the graph's write guard
    // holds it too, so no counterparty could be in flight. That is the same
    // reasoning that failed the three inversions found before this one.
    let has_raw_vectors = index.holds_raw_vectors() && index.vector_count() > 0;
    let code_count = index.pq_codes().len();

    // Determine what files are included based on what we actually saved
    let mut files_included = vec![
        "config.json".to_string(),
        "mappings.bin".to_string(),
        "metadata.json".to_string(),
    ];

    let mut files_excluded = Vec::new();

    // Add quantization files if they exist
    let int8 = index.declares_int8();
    if index.has_quantization() {
        files_included.push("quantization.json".to_string());

        if index.can_use_quantization() {
            if int8 {
                files_included.push(INT8_SCALES_FILENAME.to_string());
                if index.vector_count() > 0 {
                    files_included.push(INT8_ROWS_FILENAME.to_string());
                }
            } else {
                files_included.push("pq_centroids.bin".to_string());
                if code_count > 0 {
                    files_included.push("pq_codes.bin".to_string());
                }
            }
        }
    }

    // Add vectors.bin if it was saved
    if has_raw_vectors {
        files_included.push("vectors.bin".to_string());
    } else {
        files_excluded.push("vectors.bin".to_string());
    }

    // The sparse space's artefacts, where one is declared. Always written,
    // an empty space included, so the directory's shape says the space
    // exists.
    let space_names = index.space_artefact_names();
    let holds_spaces = !space_names.is_empty();
    files_included.extend(space_names);

    // Phase 2: Add the HNSW graph file
    //
    // One file where there used to be two. The vendored format split the
    // topology from the points so the points could be memory mapped, which this
    // build never asked for, and the split meant the two halves could disagree
    // with each other. ZeusDB's format carries both, so the pair is now one
    // length check rather than a cross file comparison.
    let vector_count = index.vector_count();
    if vector_count > 0 {
        files_included.push(GRAPH_DUMP_FILENAME.to_string());
        debug!(target: LOG_TARGET, "Graph file in manifest:");
        debug!(target: LOG_TARGET, "Included: {}", GRAPH_DUMP_FILENAME);
    } else {
        files_excluded.push(GRAPH_DUMP_FILENAME.to_string());
        debug!(target: LOG_TARGET, "No graph file (empty index)");
    }

    // Calculate compression info for quantized indexes
    //
    // Both sizes are taken over the coded records, so the ratio is the size of
    // a code against the size of the vector it stands for. `original_size_mb`
    // used to count the raw vectors the index still holds, which under
    // quantized_only is only the training records. That put a record count in
    // the numerator and a different one in the denominator, and the ratio came
    // out as the compression ratio scaled by the share of records collected
    // before training. At 1,000 training records in 3,000 it read 10.7x where
    // the codes are 32x smaller than the vectors. Under quantized_with_raw the
    // two counts were already equal, so this changes nothing there.
    // A scalar index holds one row a live record, so the coded count is the
    // live count once it is trained.
    let code_count = if int8 && index.can_use_quantization() {
        vector_count
    } else {
        code_count
    };
    let compression_info =
        if index.has_quantization() && index.can_use_quantization() && code_count > 0 {
            let raw_size_mb = (code_count * index.dim() * 4) as f64 / (1024.0 * 1024.0);
            let compressed_size_mb =
                (code_count * index.quantization_code_bytes()) as f64 / (1024.0 * 1024.0);
            let compression_ratio = if compressed_size_mb > 0.0 {
                raw_size_mb / compressed_size_mb
            } else {
                1.0
            };

            Some(CompressionInfo {
                original_size_mb: raw_size_mb,
                compressed_size_mb,
                compression_ratio,
            })
        } else {
            None
        };

    // Every artefact, the graph dump included, because they are all on disk by
    // the time this runs. manifest.json itself is not, so the figure counts the
    // directory without it, which the field's own comment states.
    let total_size_mb = calculate_directory_size(path).unwrap_or(0.0);

    // The journal, where the collection holds one. Its two names come from
    // the sink itself and the sequence from the collection, which the
    // checkpoint wrote there after it synced, so all three describe the
    // records the directory being written already holds.
    let journal = index
        .sink_journal_names()
        .map(|(file, collection_id)| JournalManifest {
            file,
            sequence: index.journal_sequence(),
            collection_id: crate::journal::collection_id_hex(collection_id),
        });

    // The major from what the directory holds, and the minor from whether
    // the dense space is declared with scalar quantization; see the module
    // documentation.
    let format_version = match (journal.is_some(), holds_spaces, int8) {
        (true, _, false) => JOURNAL_FORMAT_VERSION,
        (true, _, true) => JOURNAL_INT8_FORMAT_VERSION,
        (false, true, false) => SPACES_FORMAT_VERSION,
        (false, true, true) => SPACES_INT8_FORMAT_VERSION,
        (false, false, false) => DENSE_FORMAT_VERSION,
        (false, false, true) => DENSE_INT8_FORMAT_VERSION,
    };
    let manifest = IndexManifest {
        format_version: format_version.to_string(),
        zeusdb_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: index.created_at(),
        saved_at: Utc::now().to_rfc3339(),
        total_vectors: vector_count,
        index_type: "HNSW".to_string(),
        has_quantization: index.has_quantization(),
        quantization_trained: index.can_use_quantization(),
        storage_mode: index.storage_mode(),
        files_included,
        files_excluded,
        file_digests: ledger.digests,
        total_size_mb,
        compression_info,
        journal,
    };

    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| Error::SerializeFailed {
            what: "manifest",
            error: e.to_string(),
        })?;

    // Through the same writer every other artefact took, so it is fsynced
    // before the staging directory is moved into place. Its own digest is
    // discarded, since nothing can verify the file that carries the digests.
    let mut discard = SaveLedger::default();
    write_artefact(
        path,
        "manifest.json",
        manifest_json.as_bytes(),
        &mut discard,
    )?;

    debug!(target: LOG_TARGET, "manifest.json saved");
    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate the total size of a directory in MB, the artefacts under
/// `spaces/` included.
fn calculate_directory_size(path: &Path) -> Result<f64, std::io::Error> {
    fn bytes_under(path: &Path) -> Result<u64, std::io::Error> {
        let mut total = 0u64;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                total += metadata.len();
            } else if metadata.is_dir() {
                total += bytes_under(&entry.path())?;
            }
        }
        Ok(total)
    }

    let total_size = if path.is_dir() { bytes_under(path)? } else { 0 };
    Ok(total_size as f64 / (1024.0 * 1024.0))
}

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

/// Check if a path contains a valid ZeusDB index
///
/// Reserved surface. The body is a placeholder that reports every path invalid
/// and must be implemented before any caller is wired up, including the module
/// registration in lib.rs. The allow keeps the reservation visible instead of
/// silencing dead code across the module.
#[allow(dead_code)]
pub(crate) fn is_valid_index(_path: &str) -> bool {
    false
}

/// Get index information without full loading
///
/// Reserved surface. The body is a placeholder that reports no manifest for
/// every path and must be implemented before any caller is wired up.
#[allow(dead_code)]
pub(crate) fn get_index_info(_path: &str) -> Option<IndexManifest> {
    None
}

#[cfg(test)]
mod vectors_walk_tests {
    //! The walk that counts a raw index's vectors.bin is held to the map
    //! decode it replaced, on what it counts, what it names and what it
    //! refuses, over files a save writes and files a hand has.
    use super::*;

    /// The wire a map of these entries writes, in this order. A vector of
    /// pairs encodes exactly as a map does, and unlike a map it can hold an
    /// id twice.
    fn encoded(entries: &[(&str, Vec<f32>)]) -> Vec<u8> {
        let pairs: Vec<(String, Vec<f32>)> = entries
            .iter()
            .map(|(id, vector)| (id.to_string(), vector.clone()))
            .collect();
        bincode::encode_to_vec(&pairs, bincode::config::standard()).unwrap()
    }

    /// The record count the mappings a save wrote for these entries would
    /// hold, one per distinct id, less the ids named in `without`.
    fn records(entries: &[(&str, Vec<f32>)], without: &[&str]) -> usize {
        let mut ids: Vec<&str> = entries
            .iter()
            .map(|(id, _)| *id)
            .filter(|id| !without.contains(id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// The walk and the map decode over one file give the same count, and
    /// the walk names what `check_vectors_are_finite` names over the map.
    fn assert_walk_matches_map(entries: &[(&str, Vec<f32>)], records: usize) {
        let bytes = encoded(entries);
        let map: HashMap<String, Vec<f32>> =
            decode_bounded(&bytes, "vectors.bin", records).unwrap();
        let (count, offenders) = walk_vectors(&bytes, "vectors.bin", records).unwrap();
        assert_eq!(count, map.len(), "the count over {:?}", entries);
        let named = match check_vectors_are_finite(&map) {
            Ok(()) => vec![],
            Err(Error::VectorsNotFinite { offenders, total }) => {
                assert_eq!(total, count, "the total over {:?}", entries);
                offenders
            }
            Err(other) => panic!("the finiteness check said {:?}", other),
        };
        assert_eq!(offenders, named, "the offenders over {:?}", entries);
    }

    #[test]
    fn the_walk_counts_what_the_map_holds_and_names_the_same_offenders() {
        let entries = [
            ("a", vec![f32::NAN, 0.0]),
            ("b", vec![1.0, 2.0]),
            ("c", vec![f32::INFINITY, 1.0]),
        ];
        assert_walk_matches_map(&entries, records(&entries, &[]));
    }

    #[test]
    fn the_walk_keeps_a_clean_file_and_counts_it() {
        let entries = [("a", vec![1.0, 2.0]), ("b", vec![3.0, 4.0])];
        let bytes = encoded(&entries);
        assert_eq!(
            walk_vectors(&bytes, "vectors.bin", records(&entries, &[])).unwrap(),
            (2, vec![])
        );
    }

    #[test]
    fn the_walk_counts_an_id_held_twice_once_as_the_map_did() {
        let entries = [
            ("a", vec![1.0, 0.0]),
            ("b", vec![2.0, 0.0]),
            ("a", vec![3.0, 0.0]),
        ];
        let held = records(&entries, &[]);
        assert_eq!(held, 2);
        assert_walk_matches_map(&entries, held);
        let (count, _) = walk_vectors(&encoded(&entries), "vectors.bin", held).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn the_walk_judges_an_id_held_twice_by_its_last_copy_as_the_map_did() {
        let clean = vec![1.0, 0.0];
        let poisoned = vec![f32::NAN, 0.0];
        let first_poisoned = [
            ("a", poisoned.clone()),
            ("b", clean.clone()),
            ("a", clean.clone()),
        ];
        let last_poisoned = [
            ("a", clean.clone()),
            ("b", clean.clone()),
            ("a", poisoned.clone()),
        ];
        let both_poisoned = [
            ("a", poisoned.clone()),
            ("b", clean.clone()),
            ("a", poisoned.clone()),
        ];
        for entries in [&first_poisoned, &last_poisoned, &both_poisoned] {
            assert_walk_matches_map(entries, records(entries, &[]));
        }
        let (count, offenders) = walk_vectors(&encoded(&first_poisoned), "vectors.bin", 2).unwrap();
        assert_eq!((count, offenders), (2, vec![]));
        let (count, offenders) = walk_vectors(&encoded(&both_poisoned), "vectors.bin", 2).unwrap();
        assert_eq!((count, offenders), (2, vec!["a".to_string()]));
    }

    #[test]
    fn the_walk_counts_an_id_the_mappings_do_not_hold_as_the_map_did() {
        // The set of hashes is sized by the mappings and grows past them.
        let entries = [
            ("a", vec![1.0, 0.0]),
            ("z", vec![2.0, 0.0]),
            ("z", vec![f32::NAN, 0.0]),
            ("z", vec![3.0, 0.0]),
        ];
        let held = records(&entries, &["z"]);
        assert_eq!(held, 1);
        assert_walk_matches_map(&entries, held);
        let (count, offenders) = walk_vectors(&encoded(&entries), "vectors.bin", held).unwrap();
        assert_eq!((count, offenders), (2, vec![]));
    }

    #[test]
    fn the_walk_refuses_a_truncated_file_as_the_map_does() {
        let entries = [("a", vec![1.0, 2.0, 3.0])];
        let bytes = encoded(&entries);
        let cut = &bytes[..bytes.len() - 5];
        let map = decode_bounded::<HashMap<String, Vec<f32>>>(cut, "vectors.bin", 1)
            .unwrap_err()
            .to_string();
        let walk = walk_vectors(cut, "vectors.bin", records(&entries, &[]))
            .unwrap_err()
            .to_string();
        assert_eq!(walk, map);
    }

    #[test]
    fn the_walk_refuses_a_length_the_file_has_not_earned_as_the_map_does() {
        // A few bytes declaring 2^30 records, which claim far more than the
        // file's length earns under the budget.
        let mut bytes = bincode::encode_to_vec(1u64 << 30, bincode::config::standard()).unwrap();
        bytes.extend_from_slice(&[1, b'a', 0]);
        let map = decode_bounded::<HashMap<String, Vec<f32>>>(&bytes, "vectors.bin", 0)
            .unwrap_err()
            .to_string();
        let walk = walk_vectors(&bytes, "vectors.bin", 0)
            .unwrap_err()
            .to_string();
        assert_eq!(walk, map);
    }
}

#[cfg(test)]
mod claim_budget_tests {
    //! What a bincode artefact may claim is the budget its own length earns,
    //! whatever rung the const generic built the decoder with.
    //!
    //! The claim depends on the lengths a file declares and not on the values
    //! behind them, so the well formed artefacts here are built by hand in the
    //! shapes a save writes rather than by saving an index.
    use super::*;

    /// The varint a length is written as, which is the encoder's own.
    fn varint(value: u64) -> Vec<u8> {
        bincode::encode_to_vec(value, bincode::config::standard()).unwrap()
    }

    /// `head` padded with zeros to `total` bytes, so the file sits on a
    /// chosen rung. Nothing reads the padding: the claim comes first.
    fn padded(mut head: Vec<u8>, total: usize) -> Vec<u8> {
        head.resize(total.max(head.len()), 0);
        head
    }

    /// The rung `decode_bounded` builds its decoder with for a file of
    /// `bytes`, which is what the excess claim is measured against.
    fn rung_for(bytes: usize) -> usize {
        let budget = bytes.saturating_mul(CLAIM_PER_WIRE_BYTE);
        [1 << 20, 1 << 28, 1 << 36]
            .into_iter()
            .find(|rung| budget <= *rung)
            .unwrap_or(1 << 44)
    }

    /// A length between the budget a file of `bytes` earns and the rung it is
    /// decoded at, in entries of `entry` bytes each. The rungs are a factor of
    /// 256 apart, and this is a count that fell in the gap.
    fn between_budget_and_rung(bytes: usize, entry: usize) -> u64 {
        let rung = rung_for(bytes);
        assert!(
            rung > bytes * CLAIM_PER_WIRE_BYTE,
            "the file sits on its rung exactly"
        );
        ((rung - 4096) / entry) as u64
    }

    /// The refusal a file gets, as its message, or the word for having
    /// decoded.
    fn refusal<T: HeldArtefact>(data: &[u8], file: &str) -> String {
        match decode_bounded::<T>(data, file, T::Known::default()) {
            Ok(_) => "decoded".to_string(),
            Err(e) => e.to_string(),
        }
    }

    /// A count that claims more than the file's length earns and less than
    /// the rung the decoder was built with is refused, on every artefact this
    /// build decodes. The same bytes under the rung as the budget get past
    /// the claim, so the refusal is the budget's and not the file's shape.
    #[test]
    fn a_header_above_the_budget_is_refused_where_the_rung_admitted_it() {
        let total = 1 << 20;
        for (file, entry) in [
            ("mappings.bin", std::mem::size_of::<(String, usize)>()),
            ("vectors.bin", std::mem::size_of::<(String, Vec<f32>)>()),
            ("pq_codes.bin", std::mem::size_of::<(String, Vec<u8>)>()),
            ("pq_centroids.bin", std::mem::size_of::<Vec<Vec<f32>>>()),
        ] {
            let count = between_budget_and_rung(total, entry);
            let data = padded(varint(count), total);
            let refused = match file {
                "mappings.bin" => refusal::<IdMappings>(&data, file),
                "vectors.bin" => refusal::<HashMap<String, Vec<f32>>>(&data, file),
                "pq_codes.bin" => refusal::<HashMap<String, Vec<u8>>>(&data, file),
                _ => refusal::<Centroids>(&data, file),
            };
            assert!(
                refused.contains("declares a length its own"),
                "{file} gave {refused}"
            );
        }
        let count = between_budget_and_rung(total, std::mem::size_of::<(String, Vec<f32>)>());
        let data = padded(varint(count), total);
        let at_rung =
            decode_at::<HashMap<String, Vec<f32>>>(&data, "vectors.bin", rung_for(total), 0);
        assert!(
            matches!(at_rung, Err(Error::DecodeFailed { .. })),
            "the rung as the budget gave {at_rung:?}"
        );
    }

    /// The same for a length inside one record, which is the length the rungs
    /// left unguarded once the outer container's claim had passed. The walk
    /// of a raw index's vectors.bin refuses it in the same words.
    #[test]
    fn an_inner_length_above_the_budget_is_refused_where_the_rung_admitted_it() {
        let total = 1 << 20;
        let count = between_budget_and_rung(total, std::mem::size_of::<f32>());
        let mut head = varint(1);
        head.extend(varint(1));
        head.push(b'a');
        head.extend(varint(count));
        let data = padded(head, total);
        let refused = refusal::<HashMap<String, Vec<f32>>>(&data, "vectors.bin");
        assert!(
            refused.contains("declares a length its own"),
            "the map gave {refused}"
        );
        let walked = walk_vectors(&data, "vectors.bin", 0)
            .unwrap_err()
            .to_string();
        assert_eq!(walked, refused, "the walk and the map disagree");
        let at_rung =
            decode_at::<HashMap<String, Vec<f32>>>(&data, "vectors.bin", rung_for(total), 0);
        assert!(
            matches!(at_rung, Err(Error::DecodeFailed { .. })),
            "the rung as the budget gave {at_rung:?}"
        );
    }

    /// A file whose length earns its rung outright keeps the whole of it, so
    /// the tightening takes nothing from a file already at the top of a band.
    #[test]
    fn a_file_that_earns_its_rung_keeps_all_of_it() {
        let total = (1 << 20) / CLAIM_PER_WIRE_BYTE;
        assert_eq!(rung_for(total), 1 << 20);
        let count = ((1 << 20) / std::mem::size_of::<(String, Vec<f32>)>()) as u64;
        let data = padded(varint(count), total);
        assert!(
            matches!(
                decode_bounded::<HashMap<String, Vec<f32>>>(&data, "vectors.bin", 0),
                Err(Error::DecodeFailed { .. })
            ),
            "a claim the rung admits was refused as a length"
        );
    }

    /// Every artefact shape a save writes decodes on a fraction of the budget
    /// its own length earns, so tightening the budget to that length admits
    /// every file the rungs admitted.
    #[test]
    fn every_artefact_shape_a_save_writes_claims_a_fraction_of_its_budget() {
        let standard = bincode::config::standard();
        let ids: Vec<String> = (0..1000).map(|i| format!("r{i}")).collect();
        let mappings = IdMappings {
            id_map: ids.iter().cloned().zip(0..1000usize).collect(),
            rev_map: (0..1000usize).zip(ids.iter().cloned()).collect(),
        };
        let one = IdMappings {
            id_map: [("a".to_string(), 0usize)].into_iter().collect(),
            rev_map: [(0usize, "a".to_string())].into_iter().collect(),
        };
        let none = IdMappings {
            id_map: HashMap::new(),
            rev_map: HashMap::new(),
        };
        let vectors: HashMap<String, Vec<f32>> =
            ids.iter().map(|id| (id.clone(), vec![0.5f32; 8])).collect();
        let narrow: HashMap<String, Vec<f32>> =
            ids.iter().map(|id| (id.clone(), vec![0.5f32; 1])).collect();
        let codes: HashMap<String, Vec<u8>> =
            ids.iter().map(|id| (id.clone(), vec![3u8; 4])).collect();
        let centroids: Centroids = vec![vec![vec![0.25f32; 2]; 16]; 4];
        let wide: Centroids = vec![vec![vec![0.25f32; 1]; 2]; 1024];

        /// The least multiplier of a file's own length at which it decodes.
        macro_rules! least {
            ($name:expr, $value:expr, $type:ty) => {{
                let bytes = bincode::encode_to_vec(&$value, standard).unwrap();
                let least = (1..=CLAIM_PER_WIRE_BYTE)
                    .find(|n| {
                        decode_at::<$type>(
                            &bytes,
                            "artefact",
                            bytes.len().saturating_mul(*n),
                            Default::default(),
                        )
                        .is_ok()
                    })
                    .unwrap_or(usize::MAX);
                assert!(
                    least <= CLAIM_PER_WIRE_BYTE / 4,
                    "{} needed {} bytes a wire byte where the budget gives {}",
                    $name,
                    least,
                    CLAIM_PER_WIRE_BYTE
                );
                least
            }};
        }

        let worst = [
            least!("mappings.bin, 1000 records", mappings, IdMappings),
            least!("mappings.bin, one record", one, IdMappings),
            least!("mappings.bin, no record", none, IdMappings),
            least!("vectors.bin, dim 8", vectors, HashMap<String, Vec<f32>>),
            least!("vectors.bin, dim 1", narrow, HashMap<String, Vec<f32>>),
            least!("pq_codes.bin, 4 subvectors", codes, HashMap<String, Vec<u8>>),
            least!("pq_centroids.bin, 4 by 16 by 2", centroids, Centroids),
            least!("pq_centroids.bin, 1024 by 2 by 1", wide, Centroids),
        ]
        .into_iter()
        .max()
        .unwrap();
        assert!(worst >= 1, "nothing was measured");
    }
    /// A zero byte artefact earns no budget from its length, and the floor is
    /// what keeps its refusal the one the file's shape gives rather than a
    /// length it never declared.
    #[test]
    fn a_zero_byte_artefact_is_refused_as_an_end_and_not_as_a_length() {
        assert_eq!(claim_budget(0), LEADING_LENGTH_CLAIM);
        assert_eq!(claim_budget(1), CLAIM_PER_WIRE_BYTE);
        for (file, refused) in [
            ("mappings.bin", refusal::<IdMappings>(&[], "mappings.bin")),
            (
                "vectors.bin",
                refusal::<HashMap<String, Vec<f32>>>(&[], "vectors.bin"),
            ),
            (
                "pq_codes.bin",
                refusal::<HashMap<String, Vec<u8>>>(&[], "pq_codes.bin"),
            ),
            (
                "pq_centroids.bin",
                refusal::<Centroids>(&[], "pq_centroids.bin"),
            ),
        ] {
            assert!(refused.contains("UnexpectedEnd"), "{file} gave {refused}");
        }
        let walked = walk_vectors(&[], "vectors.bin", 0).unwrap_err().to_string();
        assert!(walked.contains("UnexpectedEnd"), "the walk gave {walked}");
    }
}

#[cfg(test)]
mod held_decode_tests {
    //! The decode `decode_bounded` makes, held to bincode's own over every
    //! artefact shape a save writes, every prefix of each, and damage at every
    //! position. What a caller observes of a decode, the value field by field or
    //! the refusal word for word, has to be the same, and the count the loader
    //! passes changes nothing a caller observes.
    use super::*;
    use bincode::error::DecodeError;
    use std::collections::BTreeMap;

    /// bincode's own decode under the same budget, a slice reader with the
    /// rung's excess claimed and one `Decode::decode`.
    fn bincode_decode<T: bincode::Decode<()>>(
        data: &[u8],
        budget: usize,
    ) -> Result<T, DecodeError> {
        use bincode::config::standard;

        fn with<T: bincode::Decode<()>, C: bincode::config::Config>(
            data: &[u8],
            config: C,
            budget: usize,
        ) -> Result<T, DecodeError> {
            let mut decoder = bincode::de::DecoderImpl::new(
                bincode::de::read::SliceReader::new(data),
                config,
                (),
            );
            claim_rung_excess(&mut decoder, budget)?;
            T::decode(&mut decoder)
        }

        if budget <= 1 << 20 {
            with(data, standard().with_limit::<{ 1 << 20 }>(), budget)
        } else if budget <= 1 << 28 {
            with(data, standard().with_limit::<{ 1 << 28 }>(), budget)
        } else if budget <= 1 << 36 {
            with(data, standard().with_limit::<{ 1 << 36 }>(), budget)
        } else {
            with(data, standard().with_limit::<{ 1 << 44 }>(), budget)
        }
    }

    /// The decode `decode_bounded` makes, under the same budget.
    fn held<T: HeldArtefact>(
        data: &[u8],
        budget: usize,
        known: T::Known,
    ) -> Result<T, DecodeError> {
        use bincode::config::standard;

        if budget <= 1 << 20 {
            decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 20 }>(), budget, known)
        } else if budget <= 1 << 28 {
            decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 28 }>(), budget, known)
        } else if budget <= 1 << 36 {
            decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 36 }>(), budget, known)
        } else {
            decode_claimed::<T, _>(data, standard().with_limit::<{ 1 << 44 }>(), budget, known)
        }
    }

    /// What a caller observes of a value, in an order no hash seed sets, with
    /// every float by its bits so a NaN equals itself and -0.0 is not 0.0.
    trait Observed {
        fn observed(&self) -> String;
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    impl Observed for IdMappings {
        fn observed(&self) -> String {
            format!(
                "{:?} {:?}",
                self.id_map.iter().collect::<BTreeMap<_, _>>(),
                self.rev_map.iter().collect::<BTreeMap<_, _>>()
            )
        }
    }

    impl Observed for HashMap<String, Vec<f32>> {
        fn observed(&self) -> String {
            let by_bits: BTreeMap<_, _> =
                self.iter().map(|(id, vector)| (id, bits(vector))).collect();
            format!("{:?}", by_bits)
        }
    }

    impl Observed for HashMap<String, Vec<u8>> {
        fn observed(&self) -> String {
            format!("{:?}", self.iter().collect::<BTreeMap<_, _>>())
        }
    }

    impl Observed for Centroids {
        fn observed(&self) -> String {
            let by_bits: Vec<Vec<Vec<u32>>> = self
                .iter()
                .map(|sub| sub.iter().map(|centroid| bits(centroid)).collect())
                .collect();
            format!("{:?}", by_bits)
        }
    }

    fn seen<T: Observed>(decoded: Result<T, DecodeError>) -> String {
        match decoded {
            Ok(value) => format!("opened {}", value.observed()),
            Err(e) => format!("refused {:?}", e),
        }
    }

    fn encoded<T: bincode::Encode>(value: &T) -> Vec<u8> {
        bincode::encode_to_vec(value, bincode::config::standard()).unwrap()
    }

    /// `bytes` whole, its prefixes, and damage at its positions: each byte
    /// overwritten with every marker a length can start with and with its top
    /// bit flipped, removed, and preceded by the marker of a `u64`. Prefix
    /// lengths and positions are taken every `step` bytes.
    fn damaged(bytes: &[u8], step: usize) -> Vec<Vec<u8>> {
        let mut inputs = vec![bytes.to_vec()];
        for len in (0..bytes.len()).step_by(step) {
            inputs.push(bytes[..len].to_vec());
        }
        for at in (0..bytes.len()).step_by(step) {
            for value in [
                0x00,
                0x01,
                0xfa,
                0xfb,
                0xfc,
                0xfd,
                0xfe,
                0xff,
                bytes[at] ^ 0x80,
            ] {
                let mut input = bytes.to_vec();
                input[at] = value;
                inputs.push(input);
            }
            let mut removed = bytes.to_vec();
            removed.remove(at);
            inputs.push(removed);
            let mut inserted = bytes.to_vec();
            inserted.insert(at, 0xfd);
            inputs.push(inserted);
        }
        inputs
    }

    /// Every input decoded both ways, under the budget its length earns and
    /// under the eight byte floor, the held decode under every count in
    /// `knowns`. Returns how many decodes were compared.
    fn holds_to_bincode<T>(label: &str, bytes: &[u8], knowns: &[T::Known]) -> usize
    where
        T: HeldArtefact + bincode::Decode<()> + Observed,
    {
        let step = if bytes.len() <= 512 {
            1
        } else {
            bytes.len() / 128
        };
        let mut compared = 0;
        for input in damaged(bytes, step) {
            for budget in [claim_budget(input.len()), LEADING_LENGTH_CLAIM] {
                let expected = seen(bincode_decode::<T>(&input, budget));
                for known in knowns {
                    let got = seen(held::<T>(&input, budget, *known));
                    assert_eq!(
                        got,
                        expected,
                        "{label}, {} bytes under a budget of {budget}",
                        input.len()
                    );
                    compared += 1;
                }
            }
        }
        compared
    }

    #[test]
    fn mappings_decode_as_bincode_decodes_them() {
        let ids = |names: &[&str]| IdMappings {
            id_map: names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.to_string(), i))
                .collect(),
            rev_map: names
                .iter()
                .enumerate()
                .map(|(i, name)| (i, name.to_string()))
                .collect(),
        };
        let hundred: Vec<String> = (0..100).map(|i| format!("r{i}")).collect();
        let hundred: Vec<&str> = hundred.iter().map(String::as_str).collect();
        let long = "x".repeat(251);
        let longer = "y".repeat(65_536);
        let shapes = [
            ("no record", ids(&[])),
            ("one record", ids(&["a"])),
            ("a hundred records", ids(&hundred)),
            (
                "ids that are not ASCII",
                ids(&["\u{e9}t\u{e9}", "\u{65e5}\u{672c}", "\u{1f980}", "a\u{0}b"]),
            ),
            ("an id of 251 bytes", ids(&[&long, "b"])),
            ("an id of 65,536 bytes", ids(&[&longer])),
        ];
        let mut compared = 0;
        for (label, shape) in &shapes {
            compared += holds_to_bincode::<IdMappings>(label, &encoded(shape), &[()]);
        }
        assert!(compared > 10_000, "{compared}");
    }

    #[test]
    fn vectors_decode_as_bincode_decodes_them() {
        let map = |entries: Vec<(String, Vec<f32>)>| entries.into_iter().collect::<HashMap<_, _>>();
        let unusual = vec![
            f32::NAN,
            f32::INFINITY,
            -0.0,
            f32::MIN_POSITIVE / 2.0,
            1.5,
            f32::MAX,
            -1.0,
            0.0,
        ];
        let shapes = [
            ("no record", encoded(&map(vec![]))),
            (
                "dim 1",
                encoded(&map((0..20)
                    .map(|i| (format!("r{i}"), vec![i as f32]))
                    .collect())),
            ),
            (
                "values that are not ordinary, and an empty vector",
                encoded(&map(vec![
                    ("a".to_string(), unusual),
                    ("b".to_string(), vec![]),
                ])),
            ),
            (
                "a hundred records at dim 4",
                encoded(&map((0..100)
                    .map(|i| (format!("r{i}"), vec![0.25 * i as f32; 4]))
                    .collect())),
            ),
            // A vector of pairs writes the wire a map writes and can hold an
            // id twice.
            (
                "an id held twice",
                encoded(&vec![
                    ("a".to_string(), vec![1.0f32]),
                    ("b".to_string(), vec![2.0]),
                    ("a".to_string(), vec![3.0]),
                ]),
            ),
        ];
        let mut compared = 0;
        for (label, bytes) in &shapes {
            compared +=
                holds_to_bincode::<HashMap<String, Vec<f32>>>(label, bytes, &[0, 1, 7, usize::MAX]);
        }
        assert!(compared > 10_000, "{compared}");
    }

    #[test]
    fn codes_decode_as_bincode_decodes_them() {
        let map = |entries: Vec<(String, Vec<u8>)>| entries.into_iter().collect::<HashMap<_, _>>();
        let shapes = [
            ("no record", encoded(&map(vec![]))),
            (
                "a hundred records of four subvectors",
                encoded(&map((0..100u8)
                    .map(|i| (format!("r{i}"), vec![i, i ^ 0xff, 0, 255]))
                    .collect())),
            ),
            (
                "an empty code",
                encoded(&map(vec![("a".to_string(), vec![])])),
            ),
            (
                "a code of 300 bytes",
                encoded(&map(vec![(
                    "a".to_string(),
                    (0..300u32).map(|i| (i % 256) as u8).collect(),
                )])),
            ),
        ];
        let mut compared = 0;
        for (label, bytes) in &shapes {
            compared +=
                holds_to_bincode::<HashMap<String, Vec<u8>>>(label, bytes, &[0, 1, 7, usize::MAX]);
        }
        assert!(compared > 5_000, "{compared}");
    }

    #[test]
    fn codebooks_decode_as_bincode_decodes_them() {
        let shapes: [(&str, Centroids); 5] = [
            (
                "four subvectors of sixteen centroids at width two",
                vec![vec![vec![0.25f32; 2]; 16]; 4],
            ),
            (
                "256 subvectors of two centroids at width one",
                vec![vec![vec![-1.0f32]; 2]; 256],
            ),
            ("no subvector", vec![]),
            ("subvectors of no centroid", vec![vec![]; 3]),
            (
                "a ragged codebook",
                vec![vec![vec![1.0]], vec![vec![], vec![2.0, f32::NAN]]],
            ),
        ];
        let knowns = [(0, 0), (4, 16), (usize::MAX, usize::MAX)];
        let mut compared = 0;
        for (label, shape) in &shapes {
            compared += holds_to_bincode::<Centroids>(label, &encoded(shape), &knowns);
        }
        assert!(compared > 10_000, "{compared}");
    }

    #[test]
    fn the_walk_refuses_every_damaged_file_as_the_map_decode_does() {
        let map: HashMap<String, Vec<f32>> = (0..30)
            .map(|i| (format!("r{i}"), vec![i as f32, f32::NAN]))
            .collect();
        let twice = vec![
            ("a".to_string(), vec![f32::NAN]),
            ("a".to_string(), vec![1.0f32]),
        ];
        let mut compared = 0;
        for bytes in [encoded(&map), encoded(&twice)] {
            for input in damaged(&bytes, 1) {
                let decoded = decode_bounded::<HashMap<String, Vec<f32>>>(&input, "vectors.bin", 0);
                let walked = walk_vectors(&input, "vectors.bin", 0);
                match (decoded, walked) {
                    (Ok(decoded), Ok((count, _))) => assert_eq!(count, decoded.len()),
                    (Err(decoded), Err(walked)) => {
                        assert_eq!(walked.to_string(), decoded.to_string())
                    }
                    (decoded, walked) => panic!(
                        "the map gave {:?} and the walk {:?}",
                        decoded.map(|m| m.len()),
                        walked
                    ),
                }
                compared += 1;
            }
        }
        assert!(compared > 1_000, "{compared}");
    }

    /// A vector of containers past the count it was reserved at holds exactly
    /// its entries once they decode, whatever count the loader passed.
    #[test]
    fn a_vector_past_its_known_count_is_reserved_at_exactly_its_entries() {
        let codebook: Centroids = vec![vec![vec![0.5f32; 3]; 16]; 4];
        let bytes = encoded(&codebook);
        let budget = claim_budget(bytes.len());
        for known in [(0, 0), (1, 1), (3, 15), (4, 16), (usize::MAX, usize::MAX)] {
            let decoded = held::<Centroids>(&bytes, budget, known).unwrap();
            assert_eq!(decoded, codebook, "{known:?}");
            assert_eq!(decoded.capacity(), 4, "{known:?}");
            assert!(
                decoded
                    .iter()
                    .all(|sub| sub.capacity() == 16 && sub.iter().all(|c| c.capacity() == 3)),
                "{known:?}"
            );
        }
    }

    #[test]
    fn a_container_is_reserved_at_no_more_than_its_bytes_and_its_known_count() {
        assert_eq!(reserved(1 << 40, 10, 2, Some(usize::MAX)), 5);
        assert_eq!(reserved(1 << 40, 10, 2, Some(3)), 3);
        assert_eq!(reserved(4, 1000, 2, Some(usize::MAX)), 4);
        assert_eq!(reserved(1 << 40, 1 << 30, 2, None), 0);
        assert_eq!(reserved(9, 0, 1, Some(9)), 0);
    }

    #[test]
    fn a_varint_reads_as_bincode_writes_it() {
        let values = [
            0u64,
            1,
            250,
            251,
            255,
            256,
            65_535,
            65_536,
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            u64::MAX,
        ];
        for value in values {
            let bytes = encoded(&value);
            assert_eq!(varint(&bytes), Some((value, bytes.len())), "{value}");
            for cut in 0..bytes.len() {
                assert_eq!(varint(&bytes[..cut]), None, "{value} cut to {cut}");
            }
        }
        assert_eq!(varint(&[254, 0, 0, 0, 0, 0, 0, 0, 0]), None);
        assert_eq!(varint(&[255, 0, 0, 0, 0, 0, 0, 0, 0]), None);
        assert_eq!(varint(&[]), None);
    }

    /// The count is the keys the bytes carry. A prefix counts the entries it
    /// holds whole, a key held twice counts once, a count the bytes do not
    /// carry counts what they do, and the count stops where a length runs
    /// past the bytes or a varint does not decode.
    #[test]
    fn the_distinct_keys_are_counted_before_the_forward_map_is_built() {
        let entries: Vec<(String, usize)> = (0..300).map(|i| (format!("r{i}"), i)).collect();
        let bytes = encoded(&entries);
        assert_eq!(count_distinct_keys(&bytes), 300);
        let mappings = IdMappings {
            id_map: entries.iter().cloned().collect(),
            rev_map: entries.iter().map(|(id, i)| (*i, id.clone())).collect(),
        };
        assert_eq!(count_distinct_keys(&encoded(&mappings)), 300);
        let mut end = encoded(&300u64).len();
        for (i, entry) in entries.iter().enumerate() {
            let next = end + encoded(entry).len();
            assert_eq!(count_distinct_keys(&bytes[..next]), i + 1, "at entry {i}");
            assert_eq!(
                count_distinct_keys(&bytes[..next - 1]),
                i,
                "short of entry {i}"
            );
            end = next;
        }

        let twice = vec![
            ("a".to_string(), 1usize),
            ("b".to_string(), 2),
            ("a".to_string(), 3),
        ];
        assert_eq!(count_distinct_keys(&encoded(&twice)), 2);
        let mut repeated = encoded(&(1u64 << 20));
        repeated.extend(std::iter::repeat_n([0u8, 0], 1 << 20).flatten());
        assert_eq!(count_distinct_keys(&repeated), 1);

        let mut declared = encoded(&(1u64 << 40));
        declared.extend_from_slice(&encoded(&twice)[1..]);
        assert_eq!(count_distinct_keys(&declared), 2);

        let ab = encoded(&"ab".to_string());
        let past = [
            encoded(&2u64),
            ab.clone(),
            vec![7],
            encoded(&(1u64 << 40)),
            vec![0; 4],
        ]
        .concat();
        assert_eq!(count_distinct_keys(&past), 1);
        let marker = [
            encoded(&2u64),
            ab.clone(),
            vec![7],
            vec![254],
            encoded(&"c".to_string()),
            vec![0],
        ]
        .concat();
        assert_eq!(count_distinct_keys(&marker), 1);
        let value_marker = [encoded(&2u64), ab, vec![254], vec![0; 8]].concat();
        assert_eq!(count_distinct_keys(&value_marker), 0);
        assert_eq!(count_distinct_keys(&[vec![255], vec![0; 8]].concat()), 0);
        assert_eq!(count_distinct_keys(&[]), 0);
        assert_eq!(count_distinct_keys(&encoded(&0u64)), 0);
    }

    /// Lengths no file here carries, under the top rung's budget so every
    /// claim passes and nothing but the bytes left stands between each length
    /// and the allocator. bincode's own decode asks the allocator for up to
    /// 16 TiB on each of these, which does not unwind, so this test surviving
    /// is what it checks. The refusal is still bincode's own.
    #[test]
    fn a_length_the_bytes_left_cannot_carry_is_not_allocated() {
        let top = 1usize << 44;
        let terabyte = 1u64 << 40;
        let id = encoded(&"a".to_string());
        let words = |refused: DecodeError| format!("{refused:?}");

        let key = [encoded(&1u64), encoded(&terabyte), b"abc".to_vec()].concat();
        assert_eq!(
            words(held::<IdMappings>(&key, top, ()).unwrap_err()),
            format!("UnexpectedEnd {{ additional: {} }}", terabyte - 3)
        );
        let forward = [encoded(&(1u64 << 38)), vec![0xff; 4]].concat();
        let reverse = [encoded(&0u64), encoded(&(1u64 << 38)), vec![0xff; 4]].concat();
        for bytes in [&forward, &reverse] {
            let refused = words(held::<IdMappings>(bytes, top, ()).unwrap_err());
            assert!(refused.starts_with("InvalidIntegerType"), "{refused}");
        }

        let vector = [
            encoded(&1u64),
            id.clone(),
            encoded(&(1u64 << 38)),
            vec![0; 8],
        ]
        .concat();
        assert_eq!(
            words(held::<HashMap<String, Vec<f32>>>(&vector, top, usize::MAX).unwrap_err()),
            "UnexpectedEnd { additional: 4 }"
        );
        let records = [encoded(&(1u64 << 38)), vec![0xff; 4]].concat();
        let refused =
            words(held::<HashMap<String, Vec<f32>>>(&records, top, usize::MAX).unwrap_err());
        assert!(refused.starts_with("InvalidIntegerType"), "{refused}");

        let code = [encoded(&1u64), id, encoded(&terabyte), vec![0; 8]].concat();
        assert_eq!(
            words(held::<HashMap<String, Vec<u8>>>(&code, top, usize::MAX).unwrap_err()),
            format!("UnexpectedEnd {{ additional: {} }}", terabyte - 8)
        );
        let refused =
            words(held::<HashMap<String, Vec<u8>>>(&records, top, usize::MAX).unwrap_err());
        assert!(refused.starts_with("InvalidIntegerType"), "{refused}");

        let subvectors = [encoded(&(1u64 << 39)), vec![0xff; 4]].concat();
        let centroids = [encoded(&1u64), encoded(&(1u64 << 39)), vec![0xff; 4]].concat();
        for bytes in [&subvectors, &centroids] {
            let refused =
                words(held::<Centroids>(bytes, top, (usize::MAX, usize::MAX)).unwrap_err());
            assert!(refused.starts_with("InvalidIntegerType"), "{refused}");
        }
        let width = [
            encoded(&1u64),
            encoded(&1u64),
            encoded(&terabyte),
            vec![0; 8],
        ]
        .concat();
        assert_eq!(
            words(held::<Centroids>(&width, top, (usize::MAX, usize::MAX)).unwrap_err()),
            "UnexpectedEnd { additional: 4 }"
        );

        let walked = walk_at(&vector, "vectors.bin", 0, top)
            .unwrap_err()
            .to_string();
        assert!(
            walked.ends_with("UnexpectedEnd { additional: 4 }"),
            "{walked}"
        );
    }
}
