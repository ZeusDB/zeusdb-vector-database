//! The four binary artefacts as a directory below format 4.0.0 holds them.
//!
//! Releases 0.3.0 to 0.11.0 wrote `mappings.bin`, `vectors.bin`,
//! `pq_codes.bin` and `pq_centroids.bin` with `bincode` 2 under its standard
//! configuration, and every directory they saved holds them that way. This
//! reads that wire for exactly those four shapes. A length or an integer is
//! an unsigned varint, being one byte below 251, or the marker 251, 252 or
//! 253 followed by a little-endian `u16`, `u32` or `u64`. A string is a
//! length and that many bytes of UTF-8, a float is four little-endian bytes,
//! and a map or a vector is a length and that many entries.
//!
//! | Artefact | Shape |
//! | --- | --- |
//! | `mappings.bin` | a map of id to internal id, then a map of internal id to id |
//! | `vectors.bin` | a map of id to a vector of floats |
//! | `pq_codes.bin` | a map of id to a vector of bytes |
//! | `pq_centroids.bin` | a vector of subvectors, each a vector of centroids, each a vector of floats |
//!
//! **Every length is held to the bytes left before anything is sized from
//! it**, at the fewest bytes one entry can occupy, so no file asks for memory
//! its own bytes do not back. A map is reserved at no more than the entries
//! its bytes carry and the count the loader already holds for it, and the
//! forward map of `mappings.bin`, for which the loader holds no count, at the
//! distinct keys its bytes carry. A codebook is walked whole before anything
//! is allocated for it and refused there if it is not the shape the
//! configuration describes.
//!
//! **A value that reads is the value the old decode returned for the same
//! bytes**, a map keeping the last copy of a key it holds twice. A file the
//! old decode refused is refused here too, in this reader's own words. Bytes
//! after the last entry, which the old decode left unread and no release
//! wrote, are refused.

use super::framed::Shape;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::{BuildHasher, BuildHasherDefault, RandomState};
use zeusdb_vector_core::Error;

/// The id maps `mappings.bin` holds, as read.
pub(super) struct Maps {
    pub(super) id_map: HashMap<String, usize>,
    pub(super) rev_map: HashMap<usize, String>,
}

/// A codebook laid out as `[subvector][centroid][value]`.
pub(super) type Codebook = Vec<Vec<Vec<f32>>>;

/// Where a read stopped, before it becomes a refusal naming the file.
#[derive(Debug)]
enum Stop {
    /// The value starting at `at` runs to `to`, past the end of the file.
    End { at: usize, to: usize },
    /// A length at `at` names more entries than the bytes after it carry.
    Length,
    /// A length marker the wire does not write, at `at`.
    Marker { at: usize, marker: u8 },
    /// A value at `at` wider than a `usize` on this target.
    Wide { at: usize, value: u64 },
    /// A string at `at` that is not UTF-8.
    Utf8 {
        at: usize,
        error: std::str::Utf8Error,
    },
    /// Bytes after the last entry, starting at `at`.
    Trailing { at: usize },
}

impl Stop {
    fn refusal(self, file: &str, bytes: usize) -> Error {
        let failed = |error: String| Error::DecodeFailed {
            file: file.to_string(),
            error,
        };
        match self {
            Stop::Length => Error::DecodeLengthExceeded {
                file: file.to_string(),
                bytes,
            },
            Stop::End { at, to } => failed(format!(
                "the file ends at byte {} and the value at byte {} runs to byte {}",
                bytes, at, to
            )),
            Stop::Marker { at, marker } => failed(format!(
                "byte {} is the length marker {}, which this format does not write",
                at, marker
            )),
            Stop::Wide { at, value } => failed(format!(
                "the value at byte {} is {}, wider than this platform holds",
                at, value
            )),
            Stop::Utf8 { at, error } => {
                failed(format!("the string at byte {} is not UTF-8: {}", at, error))
            }
            Stop::Trailing { at } => failed(format!(
                "the file continues past its last entry, from byte {} to byte {}",
                at, bytes
            )),
        }
    }
}

/// A cursor over the old wire, refusing every read the bytes do not carry.
struct Wire<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Wire<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Wire { bytes, at: 0 }
    }

    fn left(&self) -> usize {
        self.bytes.len() - self.at
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Stop> {
        if n > self.left() {
            return Err(Stop::End {
                at: self.at,
                to: self.at.saturating_add(n),
            });
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    fn varint(&mut self) -> Result<u64, Stop> {
        let at = self.at;
        let marker = self.take(1)?[0];
        Ok(match marker {
            0..=250 => u64::from(marker),
            251 => u64::from(u16::from_le_bytes(word(self.take(2)?))),
            252 => u64::from(u32::from_le_bytes(word(self.take(4)?))),
            253 => u64::from_le_bytes(word(self.take(8)?)),
            _ => return Err(Stop::Marker { at, marker }),
        })
    }

    /// An integer the old wire wrote as a `usize`.
    fn usize(&mut self) -> Result<usize, Stop> {
        let at = self.at;
        let value = self.varint()?;
        usize::try_from(value).map_err(|_| Stop::Wide { at, value })
    }

    /// A container's length, refused where its entries at `least` bytes each
    /// would run past the bytes left.
    fn len(&mut self, least: usize) -> Result<usize, Stop> {
        let declared = self.varint()?;
        usize::try_from(declared)
            .ok()
            .filter(|&len| {
                len.checked_mul(least)
                    .is_some_and(|bytes| bytes <= self.left())
            })
            .ok_or(Stop::Length)
    }

    /// A string's bytes, before they are held to UTF-8.
    fn raw(&mut self) -> Result<&'a [u8], Stop> {
        let len = self.len(1)?;
        self.take(len)
    }

    fn str(&mut self) -> Result<&'a str, Stop> {
        let len = self.len(1)?;
        let at = self.at;
        std::str::from_utf8(self.take(len)?).map_err(|error| Stop::Utf8 { at, error })
    }

    /// A vector of floats, as the four bytes of each.
    fn floats(&mut self) -> Result<&'a [u8], Stop> {
        let len = self.len(4)?;
        self.take(len * 4)
    }

    fn end(&self) -> Result<(), Stop> {
        match self.left() {
            0 => Ok(()),
            _ => Err(Stop::Trailing { at: self.at }),
        }
    }
}

fn word<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(bytes);
    out
}

fn floats_of(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|value| f32::from_le_bytes(word(value)))
        .collect()
}

fn all_finite(raw: &[u8]) -> bool {
    raw.chunks_exact(4)
        .all(|value| f32::from_le_bytes(word(value)).is_finite())
}

/// `mappings.bin`, both maps.
pub(super) fn read_mappings(bytes: &[u8], file: &str) -> Result<Maps, Error> {
    mappings(bytes).map_err(|stop| stop.refusal(file, bytes.len()))
}

fn mappings(bytes: &[u8]) -> Result<Maps, Stop> {
    let distinct = count_distinct_keys(bytes);
    let mut wire = Wire::new(bytes);
    let entries = wire.len(2)?;
    let mut id_map = HashMap::with_capacity(entries.min(distinct));
    for _ in 0..entries {
        let id = wire.str()?;
        let internal_id = wire.usize()?;
        id_map.insert(id.to_string(), internal_id);
    }
    let entries = wire.len(2)?;
    let mut rev_map = HashMap::with_capacity(entries.min(id_map.len()));
    for _ in 0..entries {
        let internal_id = wire.usize()?;
        let id = wire.str()?;
        rev_map.insert(internal_id, id.to_string());
    }
    wire.end()?;
    Ok(Maps { id_map, rev_map })
}

/// A hash kept as it is, for a set whose entries are hashes already.
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

/// The distinct keys the forward map of `mappings.bin` carries, counted
/// before the map is built.
///
/// The forward map is the one container no count precedes in the file and
/// none the loader holds from elsewhere. Reserved at the count it declares, a
/// file declaring a count its bytes do not carry reserves that count, and
/// grown from nothing as its entries are read, its last doubling holds the old
/// table beside the new one. The keys the bytes carry are what the map will
/// hold and nothing a declared count can inflate. A key the bytes hold twice
/// counts once, and the count stops where an entry does not read. Each key is
/// hashed once and kept as its hash, eight bytes an entry. Two keys sharing a
/// hash count once and reserve the map one entry short, which it grows for.
pub(super) fn count_distinct_keys(bytes: &[u8]) -> usize {
    let mut wire = Wire::new(bytes);
    let Ok(declared) = wire.varint() else {
        return 0;
    };
    let hasher = RandomState::new();
    let mut keys: HashSet<u64, BuildHasherDefault<Hashed>> = HashSet::default();
    for _ in 0..declared {
        let Ok(key) = wire.raw() else {
            break;
        };
        if wire.varint().is_err() {
            break;
        }
        keys.insert(hasher.hash_one(key));
    }
    keys.len()
}

/// `vectors.bin` whole, reserved at no more than `records`, the count the
/// mappings hold.
pub(super) fn read_vectors(
    bytes: &[u8],
    file: &str,
    records: usize,
) -> Result<HashMap<String, Vec<f32>>, Error> {
    vectors(bytes, records).map_err(|stop| stop.refusal(file, bytes.len()))
}

fn vectors(bytes: &[u8], records: usize) -> Result<HashMap<String, Vec<f32>>, Stop> {
    let mut wire = Wire::new(bytes);
    let entries = wire.len(2)?;
    let mut out = HashMap::with_capacity(entries.min(records));
    for _ in 0..entries {
        let id = wire.str()?;
        let vector = floats_of(wire.floats()?);
        out.insert(id.to_string(), vector);
    }
    wire.end()?;
    Ok(out)
}

/// `vectors.bin` read for its record count and the ids of the records whose
/// vector is not finite, sorted, keeping nothing else.
///
/// The count and the ids are what the map would give. An id the file holds
/// twice is held once and judged by its last copy. Each id is hashed into a
/// set sized by `records`, the count the mappings hold, and a set as large as
/// the file's count proves every id distinct. A smaller set holds a duplicate
/// or a collision, and the ids are counted again by name, which no file a
/// save wrote reaches.
pub(super) fn walk_vectors(
    bytes: &[u8],
    file: &str,
    records: usize,
) -> Result<(usize, Vec<String>), Error> {
    walk(bytes, records).map_err(|stop| stop.refusal(file, bytes.len()))
}

fn walk(bytes: &[u8], records: usize) -> Result<(usize, Vec<String>), Stop> {
    let hasher = RandomState::new();
    let mut wire = Wire::new(bytes);
    let entries = wire.len(2)?;
    let mut hashes: HashSet<u64> = HashSet::with_capacity(records.min(entries));
    let mut offenders: BTreeSet<&str> = BTreeSet::new();
    for _ in 0..entries {
        let id = wire.str()?;
        if all_finite(wire.floats()?) {
            offenders.remove(id);
        } else {
            offenders.insert(id);
        }
        hashes.insert(hasher.hash_one(id));
    }
    wire.end()?;
    let count = if hashes.len() == entries {
        entries
    } else {
        let mut wire = Wire::new(bytes);
        let entries = wire.len(2)?;
        let mut ids: HashSet<&str> = HashSet::new();
        for _ in 0..entries {
            ids.insert(wire.str()?);
            wire.floats()?;
        }
        ids.len()
    };
    Ok((count, offenders.into_iter().map(str::to_string).collect()))
}

/// `pq_codes.bin` whole, reserved at no more than `records`.
pub(super) fn read_codes(
    bytes: &[u8],
    file: &str,
    records: usize,
) -> Result<HashMap<String, Vec<u8>>, Error> {
    codes(bytes, records).map_err(|stop| stop.refusal(file, bytes.len()))
}

fn codes(bytes: &[u8], records: usize) -> Result<HashMap<String, Vec<u8>>, Stop> {
    let mut wire = Wire::new(bytes);
    let entries = wire.len(2)?;
    let mut out = HashMap::with_capacity(entries.min(records));
    for _ in 0..entries {
        let id = wire.str()?;
        let code = wire.raw()?.to_vec();
        out.insert(id.to_string(), code);
    }
    wire.end()?;
    Ok(out)
}

/// `pq_centroids.bin` whole.
///
/// The file is walked once allocating nothing, which holds every length to
/// its bytes and measures the codebook. A codebook that is not `shape`, or
/// whose subvectors or centroids differ in length, is refused there, in the
/// words the install gives a codebook of the wrong shape, so nothing is
/// allocated for a codebook that would be refused. A codebook of the right
/// shape is then read at exactly that shape.
pub(super) fn read_codebook(bytes: &[u8], file: &str, shape: Shape) -> Result<Codebook, Error> {
    let (actual, uniform) = measure(bytes).map_err(|stop| stop.refusal(file, bytes.len()))?;
    if actual != shape.expected || !uniform {
        return Err(shape.mismatch(actual));
    }
    codebook(bytes, actual).map_err(|stop| stop.refusal(file, bytes.len()))
}

/// A codebook's shape, being its subvectors, the first subvector's centroids
/// and the first centroid's values, and whether every subvector and every
/// centroid has the first one's, as the install measures a codebook.
fn measure(bytes: &[u8]) -> Result<((usize, usize, usize), bool), Stop> {
    let mut wire = Wire::new(bytes);
    let subvectors = wire.len(1)?;
    let mut first: Option<(usize, usize)> = None;
    let mut uniform = true;
    for _ in 0..subvectors {
        let centroids = wire.len(1)?;
        let mut width: Option<usize> = None;
        for _ in 0..centroids {
            let values = wire.floats()?.len() / 4;
            uniform &= *width.get_or_insert(values) == values;
        }
        let this = (centroids, width.unwrap_or(0));
        let head = *first.get_or_insert(this);
        uniform &= this.0 == head.0 && (this.0 == 0 || this.1 == head.1);
    }
    wire.end()?;
    let (centroids, width) = first.unwrap_or((0, 0));
    Ok(((subvectors, centroids, width), uniform))
}

/// A codebook `measure` has read whole, at exactly its shape.
fn codebook(bytes: &[u8], shape: (usize, usize, usize)) -> Result<Codebook, Stop> {
    let (subvectors, centroids, _) = shape;
    let mut wire = Wire::new(bytes);
    wire.len(1)?;
    let mut out = Vec::with_capacity(subvectors);
    for _ in 0..subvectors {
        wire.len(1)?;
        let mut sub = Vec::with_capacity(centroids);
        for _ in 0..centroids {
            sub.push(floats_of(wire.floats()?));
        }
        out.push(sub);
    }
    Ok(out)
}
