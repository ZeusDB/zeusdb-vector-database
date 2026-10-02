//! The four binary artefacts as a directory at format 4.0.0 holds them, each
//! inside the frame `zeusdb_vector_core::frame` describes.
//!
//! | Artefact | Kind | `entries` | Payload |
//! | --- | --- | --- | --- |
//! | `mappings.bin` | 1, id mappings | records | `entries` u32 internal ids, strictly increasing, then `entries` u32 byte lengths, then every id's UTF-8 end to end |
//! | `vectors.bin` | 2, raw vectors | records | a u32 width, then per record a u32 internal id and `width` f32 |
//! | `pq_codes.bin` | 3, product quantized codes | records holding a code | a u32 width, then per record a u32 internal id and `width` bytes |
//! | `pq_centroids.bin` | 4, product quantized codebook | subvectors | a u32 centroid count and a u32 centroid width, then every value, subvector by subvector and centroid by centroid |
//!
//! Every field is little-endian at a fixed width. Records follow increasing
//! internal id, which is the order the id store holds them in, so two saves
//! of the same records write the same bytes. An id's text is held once, in
//! `mappings.bin`, and the other two name a record by its internal id.
//!
//! A reader here holds a payload to its frame's `entries` and to its exact
//! length before it reads a field, and hands back a view over the bytes that
//! allocates nothing. What a payload is held to outside itself, being the
//! mappings, `config.json` and `quantization.json`, is the loader's, except
//! the codebook's shape, which is held before the codebook is built.

use zeusdb_vector_core::{
    frame_begin, frame_finish, unframe, Error, FrameEncoding, FrameKind, FRAME_HEADER_BYTES,
};

/// The codebook's shape `quantization.json` describes, being the subvectors,
/// the centroids a subvector and the values a centroid, with the two fields
/// it is derived from, which the refusal names.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) expected: (usize, usize, usize),
    pub(super) subvectors: usize,
    pub(super) bits: usize,
}

impl Shape {
    /// A codebook of `actual` that is not this shape, refused.
    pub(super) fn mismatch(&self, actual: (usize, usize, usize)) -> Error {
        Error::CodebookShapeMismatch {
            actual,
            expected: self.expected,
            subvectors: self.subvectors,
            bits: self.bits,
        }
    }
}

fn word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn id32(slot: usize) -> u32 {
    u32::try_from(slot).expect("internal ids are issued below the u32 ceiling")
}

fn unwritable(what: &'static str, error: String) -> Error {
    Error::SerializeFailed { what, error }
}

/// A width or a count a payload stores as a `u32`.
fn field(value: usize, what: &'static str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| {
        unwritable(
            what,
            format!("{} does not fit the four bytes it is written in", value),
        )
    })
}

// ============================================================================
// WRITING
// ============================================================================

/// `mappings.bin` from every record's internal id and id, in increasing
/// internal id order, being `entries` records carrying `text` bytes of id
/// between them.
pub(super) fn write_mappings<'a>(
    records: impl IntoIterator<Item = (usize, &'a str)>,
    entries: usize,
    text: usize,
) -> Result<Vec<u8>, Error> {
    let lengths = FRAME_HEADER_BYTES + 4 * entries;
    let mut out = frame_begin(
        FrameKind::IdMappings,
        FrameEncoding::Engine,
        8 * entries + text,
    );
    out.resize(lengths + 4 * entries, 0);
    let mut written = 0usize;
    for (slot, id) in records {
        if written == entries {
            written += 1;
            break;
        }
        let length = field(id.len(), "mappings")?;
        let at = FRAME_HEADER_BYTES + 4 * written;
        out[at..at + 4].copy_from_slice(&id32(slot).to_le_bytes());
        let at = lengths + 4 * written;
        out[at..at + 4].copy_from_slice(&length.to_le_bytes());
        out.extend_from_slice(id.as_bytes());
        written += 1;
    }
    if written != entries {
        return Err(unwritable(
            "mappings",
            format!(
                "the id store counted {} records and yielded {}{}",
                entries,
                if written > entries { "more than " } else { "" },
                written.min(entries)
            ),
        ));
    }
    Ok(frame_finish(out, entries as u64))
}

/// `vectors.bin` from every record's internal id and raw vector, in
/// increasing internal id order. `hint` is the record count, which sizes the
/// buffer once. Returns the frame and the records it holds.
pub(super) fn write_vectors<'a>(
    width: usize,
    records: impl IntoIterator<Item = (usize, &'a [f32])>,
    hint: usize,
) -> Result<(Vec<u8>, usize), Error> {
    let stride = 4 + 4 * width;
    let mut out = frame_begin(
        FrameKind::RawVectors,
        FrameEncoding::Engine,
        4 + hint.saturating_mul(stride),
    );
    out.extend_from_slice(&field(width, "vectors")?.to_le_bytes());
    let mut entries = 0usize;
    for (slot, vector) in records {
        if vector.len() != width {
            return Err(unwritable(
                "vectors",
                format!(
                    "internal id {} holds {} values and the index is {} wide",
                    slot,
                    vector.len(),
                    width
                ),
            ));
        }
        out.extend_from_slice(&id32(slot).to_le_bytes());
        for value in vector {
            out.extend_from_slice(&value.to_le_bytes());
        }
        entries += 1;
    }
    Ok((frame_finish(out, entries as u64), entries))
}

/// `pq_codes.bin` from every coded record's internal id and code, in
/// increasing internal id order, each code `width` bytes. Returns the frame
/// and the records it holds.
pub(super) fn write_codes<'a>(
    width: usize,
    records: impl IntoIterator<Item = (usize, &'a [u8])>,
    hint: usize,
) -> Result<(Vec<u8>, usize), Error> {
    let stride = 4 + width;
    let mut out = frame_begin(
        FrameKind::PqCodes,
        FrameEncoding::Engine,
        4 + hint.saturating_mul(stride),
    );
    out.extend_from_slice(&field(width, "PQ codes")?.to_le_bytes());
    let mut entries = 0usize;
    for (slot, code) in records {
        if code.len() != width {
            return Err(unwritable(
                "PQ codes",
                format!(
                    "internal id {} holds a code of {} bytes and the index codes {}",
                    slot,
                    code.len(),
                    width
                ),
            ));
        }
        out.extend_from_slice(&id32(slot).to_le_bytes());
        out.extend_from_slice(code);
        entries += 1;
    }
    Ok((frame_finish(out, entries as u64), entries))
}

/// `pq_centroids.bin` from a codebook every subvector and centroid of which
/// has the first one's length.
pub(super) fn write_codebook(codebook: &[Vec<Vec<f32>>]) -> Result<Vec<u8>, Error> {
    let centroids = codebook.first().map_or(0, Vec::len);
    let width = codebook
        .first()
        .and_then(|sub| sub.first())
        .map_or(0, Vec::len);
    let uniform = codebook
        .iter()
        .all(|sub| sub.len() == centroids && sub.iter().all(|centroid| centroid.len() == width));
    if !uniform {
        return Err(unwritable(
            "PQ centroids",
            "the codebook's subvectors or centroids differ in length".to_string(),
        ));
    }
    let mut out = frame_begin(
        FrameKind::PqCodebook,
        FrameEncoding::Engine,
        8 + 4 * codebook.len() * centroids * width,
    );
    out.extend_from_slice(&field(centroids, "PQ centroids")?.to_le_bytes());
    out.extend_from_slice(&field(width, "PQ centroids")?.to_le_bytes());
    for value in codebook.iter().flatten().flatten() {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(frame_finish(out, codebook.len() as u64))
}

// ============================================================================
// READING
// ============================================================================

/// A product of sizes as a refusal prints it, which a hostile header can
/// carry past what any integer holds.
fn product(factors: &[u64]) -> String {
    factors
        .iter()
        .try_fold(1u128, |product, &factor| {
            product.checked_mul(u128::from(factor))
        })
        .map_or_else(
            || "more than 2^128".to_string(),
            |product| product.to_string(),
        )
}

fn corrupt(file: &str, error: String) -> Error {
    Error::DecodeFailed {
        file: file.to_string(),
        error,
    }
}

/// Refuse internal ids that do not strictly increase, each read from the
/// first four bytes of a `stride` long run of `rows`.
fn strictly_increasing(rows: &[u8], stride: usize, file: &str) -> Result<(), Error> {
    let mut previous: Option<u32> = None;
    for row in rows.chunks_exact(stride) {
        let id = word(row);
        if let Some(last) = previous.filter(|&last| id <= last) {
            return Err(corrupt(
                file,
                format!(
                    "internal id {} follows {}, and the ids are strictly increasing",
                    id, last
                ),
            ));
        }
        previous = Some(id);
    }
    Ok(())
}

/// `mappings.bin`, read: every record's internal id and id, held to its frame.
pub(super) struct Mappings<'a> {
    ids: &'a [u8],
    lengths: &'a [u8],
    text: &'a str,
}

impl<'a> Mappings<'a> {
    /// The frame, and a payload holding `entries` ids strictly increasing,
    /// `entries` lengths summing to exactly the text that follows, and every
    /// id UTF-8 on its own.
    pub(super) fn read(bytes: &'a [u8], file: &str) -> Result<Self, Error> {
        let framed = unframe(bytes, FrameKind::IdMappings, file)?;
        let payload = framed.payload;
        let entries = usize::try_from(framed.entries)
            .ok()
            .filter(|&entries| {
                entries
                    .checked_mul(8)
                    .is_some_and(|bytes| bytes <= payload.len())
            })
            .ok_or_else(|| {
                corrupt(
                    file,
                    format!(
                        "holds {} payload bytes and {} records take at least {}",
                        payload.len(),
                        framed.entries,
                        product(&[framed.entries, 8])
                    ),
                )
            })?;
        let (ids, rest) = payload.split_at(4 * entries);
        let (lengths, text) = rest.split_at(4 * entries);
        let total = lengths
            .chunks_exact(4)
            .try_fold(0u64, |sum, length| sum.checked_add(u64::from(word(length))));
        if total != Some(text.len() as u64) {
            return Err(corrupt(
                file,
                format!(
                    "the ids' lengths sum to {} bytes and the payload holds {} bytes of id",
                    total.map_or_else(|| "more than 2^64".to_string(), |t| t.to_string()),
                    text.len()
                ),
            ));
        }
        strictly_increasing(ids, 4, file)?;
        let not_utf8 = |offset: usize| {
            let mut end = 0usize;
            let held = ids
                .chunks_exact(4)
                .zip(lengths.chunks_exact(4))
                .find(|(_, length)| {
                    end += word(length) as usize;
                    end > offset
                })
                .map_or(0, |(id, _)| word(id));
            corrupt(
                file,
                format!("the id held at internal id {} is not UTF-8", held),
            )
        };
        let text = std::str::from_utf8(text).map_err(|error| not_utf8(error.valid_up_to()))?;
        let mut end = 0usize;
        for length in lengths.chunks_exact(4) {
            let start = end;
            end += word(length) as usize;
            if !text.is_char_boundary(end) {
                return Err(not_utf8(start));
            }
        }
        Ok(Mappings { ids, lengths, text })
    }

    /// How many records the artefact holds.
    pub(super) fn len(&self) -> usize {
        self.ids.len() / 4
    }

    /// The record at the highest internal id, which is the last.
    pub(super) fn last(&self) -> Option<(usize, &'a str)> {
        let id = self
            .ids
            .len()
            .checked_sub(4)
            .map(|at| word(&self.ids[at..]))?;
        let length = word(&self.lengths[self.lengths.len() - 4..]) as usize;
        Some((id as usize, &self.text[self.text.len() - length..]))
    }

    /// Every record as its internal id and id, in increasing internal id.
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, &'a str)> + '_ {
        let mut end = 0usize;
        self.ids
            .chunks_exact(4)
            .zip(self.lengths.chunks_exact(4))
            .map(move |(id, length)| {
                let start = end;
                end += word(length) as usize;
                (word(id) as usize, &self.text[start..end])
            })
    }
}

/// `vectors.bin` or `pq_codes.bin`, read: a width, then per record an
/// internal id and a row of that width, held to its frame.
pub(super) struct Rows<'a> {
    stride: usize,
    rows: &'a [u8],
}

impl<'a> Rows<'a> {
    /// The frame of `kind`, a width that is `width`, and a payload holding
    /// exactly `entries` rows of `unit` bytes a value with their internal ids
    /// strictly increasing. `mismatch` words the refusal of another width.
    pub(super) fn read(
        bytes: &'a [u8],
        kind: FrameKind,
        file: &str,
        unit: usize,
        width: usize,
        mismatch: impl FnOnce(usize) -> String,
    ) -> Result<Self, Error> {
        let framed = unframe(bytes, kind, file)?;
        let payload = framed.payload;
        if payload.len() < 4 {
            return Err(corrupt(
                file,
                format!(
                    "holds {} payload bytes and the width alone takes 4",
                    payload.len()
                ),
            ));
        }
        let held = word(payload) as usize;
        if held != width {
            return Err(corrupt(file, mismatch(held)));
        }
        let stride = 4 + unit * width;
        let rows = &payload[4..];
        let expected = usize::try_from(framed.entries)
            .ok()
            .and_then(|entries| entries.checked_mul(stride));
        if expected != Some(rows.len()) {
            return Err(corrupt(
                file,
                format!(
                    "holds {} bytes of rows and {} rows of {} bytes take {}",
                    rows.len(),
                    framed.entries,
                    stride,
                    product(&[framed.entries, stride as u64])
                ),
            ));
        }
        strictly_increasing(rows, stride, file)?;
        Ok(Rows { stride, rows })
    }

    /// How many records the artefact holds.
    pub(super) fn len(&self) -> usize {
        self.rows.len() / self.stride
    }

    /// Every record as its internal id and its row's bytes, in increasing
    /// internal id.
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, &'a [u8])> + '_ {
        self.rows
            .chunks_exact(self.stride)
            .map(|row| (word(row) as usize, &row[4..]))
    }
}

/// A row of `vectors.bin` as the floats it holds.
pub(super) fn floats(row: &[u8]) -> impl Iterator<Item = f32> + '_ {
    row.chunks_exact(4)
        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

/// `pq_centroids.bin`, read and built: the frame, a payload exactly the
/// codebook its own shape describes, and that shape `shape`, which is held
/// before anything is allocated.
pub(super) fn read_codebook(
    bytes: &[u8],
    file: &str,
    shape: Shape,
) -> Result<Vec<Vec<Vec<f32>>>, Error> {
    let framed = unframe(bytes, FrameKind::PqCodebook, file)?;
    let payload = framed.payload;
    if payload.len() < 8 {
        return Err(corrupt(
            file,
            format!(
                "holds {} payload bytes and the shape alone takes 8",
                payload.len()
            ),
        ));
    }
    let centroids = word(payload) as usize;
    let width = word(&payload[4..]) as usize;
    let values = &payload[8..];
    let subvectors = usize::try_from(framed.entries).ok();
    let expected = subvectors
        .and_then(|subvectors| subvectors.checked_mul(centroids))
        .and_then(|cells| cells.checked_mul(width))
        .and_then(|floats| floats.checked_mul(4));
    if expected != Some(values.len()) {
        return Err(corrupt(
            file,
            format!(
                "holds {} bytes of values and a codebook of {}x{}x{} takes {}",
                values.len(),
                framed.entries,
                centroids,
                width,
                product(&[framed.entries, centroids as u64, width as u64, 4])
            ),
        ));
    }
    let actual = (subvectors.unwrap_or(usize::MAX), centroids, width);
    if actual != shape.expected {
        return Err(shape.mismatch(actual));
    }
    let mut floats = values
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]));
    Ok((0..actual.0)
        .map(|_| {
            (0..centroids)
                .map(|_| floats.by_ref().take(width).collect())
                .collect()
        })
        .collect())
}
