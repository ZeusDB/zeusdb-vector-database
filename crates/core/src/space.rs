//! The seam every vector index sits behind.
//!
//! A collection holds one record set and one or more spaces over it, and each
//! space is an index implementing [`VectorIndex`] for one [`Kind`] of vector.
//! The trait returns a finished page and takes an admit set. It is not a
//! cursor, because a graph traversal has no order to seek in and a trait half
//! its implementors would have to fake is the wrong trait. What crosses the
//! crate boundary is a page and a predicate, and a postings index composes
//! internally however it likes.
//!
//! # The shapes
//!
//! The vector shape lives on the [`Kind`] marker rather than on the index
//! trait, because a generic associated type on the trait itself would cost
//! object safety. `Box<dyn VectorIndex<Dense>>` is an ordinary trait object,
//! and a new family of vector is a new marker and a new implementor with no
//! change here.
//!
//! # Ids
//!
//! A [`RecordId`] is the collection's internal id, allocated from a counter
//! that never goes backwards and never reuses a value. Every posting list a
//! sparse index keeps stays sorted by tail append because of that, and every
//! per-record structure a dense index keeps is addressed by it.
//!
//! # Cost
//!
//! [`VectorIndex::cost`] reports estimated nanoseconds rather than a count of
//! operations, because a posting visit and a distance evaluation are not the
//! same amount of work and a planner comparing raw counts across arms would
//! be wrong by the ratio between them. Each implementor multiplies its own
//! count by a unit cost it timed on itself when it was opened, with a
//! compiled-in floor for an index too small to time, and nothing persists the
//! figure, since it moves with the machine and with the build.

use std::any::Any;
use std::fmt;
use std::path::Path;

use crate::admit::Admit;
use crate::checksum::checksum_of;
use crate::error::Error;

// ============================================================================
// IDENTITY
// ============================================================================

/// A record's internal id.
///
/// A `u32` because a dense index addresses its nodes with one and a sparse
/// index keeps one in every posting, and because the record count a
/// collection may declare is capped far below what it can name. The
/// collection's own counter is a `usize`, so the two conversions below are
/// where the width is agreed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordId(pub u32);

impl RecordId {
    /// The id as the collection's slot index.
    #[inline]
    pub fn slot(self) -> usize {
        self.0 as usize
    }

    /// The id of a slot the collection allocated.
    ///
    /// Saturates rather than wrapping, so a slot no `u32` can name becomes
    /// an id no index will ever hold rather than a collision with a live one.
    /// The collection refuses an id counter that large before it reaches
    /// here.
    #[inline]
    pub fn from_slot(slot: usize) -> Self {
        RecordId(u32::try_from(slot).unwrap_or(u32::MAX))
    }
}

/// The name of one space in a collection. Never empty.
///
/// The default space is a name the binding supplies, and it is an ordinary
/// name to everything below the binding.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpaceName(String);

impl SpaceName {
    pub fn new(name: &str) -> Result<Self, Error> {
        if name.is_empty() {
            return Err(Error::SpaceNameEmpty);
        }
        Ok(SpaceName(name.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SpaceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ============================================================================
// KINDS
// ============================================================================

/// The closed set of vector families a collection can hold.
///
/// `non_exhaustive` so a crate above the collection that matches on it is
/// not broken when a family is added.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SpaceKind {
    Dense,
    Sparse,
}

impl SpaceKind {
    /// The family's name, as an error message spells it.
    pub fn name(self) -> &'static str {
        match self {
            SpaceKind::Dense => "dense",
            SpaceKind::Sparse => "sparse",
        }
    }
}

/// One family of vector, stated as the shape of what an index stores and
/// what it is asked.
///
/// The marker carries the shapes so `VectorIndex<K>` can name them without a
/// lifetime of its own and without an enum every implementor would have to
/// match on.
pub trait Kind: 'static + Send + Sync {
    const KIND: SpaceKind;
    /// What one record contributes to this space, borrowed.
    type Vector<'a>: Copy;
    /// What a search is asked with, borrowed.
    type Query<'a>: Copy;
    /// The owned form of a vector, which [`VectorIndex::recover`] hands back.
    type Owned;
}

/// One `f32` vector per record, and the query is one too.
pub struct Dense;

impl Kind for Dense {
    const KIND: SpaceKind = SpaceKind::Dense;
    type Vector<'a> = &'a [f32];
    type Query<'a> = &'a [f32];
    type Owned = Vec<f32>;
}

/// A sparse vector as parallel dimension and value slices, sorted by
/// dimension.
#[derive(Clone, Copy, Debug)]
pub struct SparseRef<'a> {
    pub dims: &'a [u32],
    pub values: &'a [f32],
}

impl SparseRef<'_> {
    /// The rules a sparse vector has to satisfy before an index takes it.
    ///
    /// The two slices are the same length, the dimensions are strictly
    /// increasing, and every value is finite. Strictly increasing rather than
    /// sorted, because a repeated dimension would put two postings for one
    /// record on one list.
    pub fn validate(&self) -> Result<(), Error> {
        if self.dims.len() != self.values.len() {
            return Err(Error::SparseVectorShape {
                dims: self.dims.len(),
                values: self.values.len(),
            });
        }
        if let Some(position) = (1..self.dims.len()).find(|&i| self.dims[i] <= self.dims[i - 1]) {
            return Err(Error::SparseDimsNotIncreasing { position });
        }
        if let Some((index, &value)) = self
            .values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(Error::SparseValueNotFinite { index, value });
        }
        Ok(())
    }

    /// The number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.dims.len()
    }

    /// The sparse dot product of two dimension-sorted vectors.
    ///
    /// Accumulated in ascending dimension order, which is the order a
    /// term-at-a-time scan adds the same contributions in, so the two agree
    /// bit for bit and a page can be checked against a brute-force one on
    /// exact equality.
    pub fn dot(&self, other: SparseRef<'_>) -> f32 {
        let (a, b) = (self, other);
        let (mut i, mut j, mut sum) = (0usize, 0usize, 0f32);
        while i < a.dims.len() && j < b.dims.len() {
            match a.dims[i].cmp(&b.dims[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += a.values[i] * b.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }
}

/// An owned sparse vector, which is what a record carries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseVector {
    pub dims: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn as_ref(&self) -> SparseRef<'_> {
        SparseRef {
            dims: &self.dims,
            values: &self.values,
        }
    }
}

/// Weighted postings per record, and the query is a weighted term list.
///
/// The weights on both sides are whatever the sparse index's scoring rule
/// decided, which is how a term weighting arrives without this type changing.
pub struct Sparse;

impl Kind for Sparse {
    const KIND: SpaceKind = SpaceKind::Sparse;
    type Vector<'a> = SparseRef<'a>;
    type Query<'a> = SparseRef<'a>;
    type Owned = SparseVector;
}

// ============================================================================
// WHAT A PLANNER REASONS ON
// ============================================================================

/// How many records an admit set is expected to hold. Bounded rather than a
/// point, so a planner compares bounds against thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selectivity {
    pub min: u32,
    pub expected: u32,
    pub max: u32,
}

impl Selectivity {
    pub fn exact(n: u32) -> Self {
        Selectivity {
            min: n,
            expected: n,
            max: n,
        }
    }
}

/// What an arm says a search will cost, in estimated nanoseconds, and whether
/// the page it would produce is exact by construction.
///
/// Nanoseconds rather than a count, for the reason the module documentation
/// gives. The figure is an estimate from a unit cost the index timed on
/// itself, and two arms' figures are comparable because both are in the same
/// unit and both were timed on the same machine in the same build.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub work_ns: f64,
    pub exact: bool,
}

/// Per-arm knobs an index reads and a planner may set.
///
/// Every field is optional or defaults to off, so an index applies its own
/// default where a caller says nothing, which is what `ef_search` does today.
/// `ef`, `fetch` and `rerank` are a graph's knobs and a postings index ignores
/// them.
///
/// `boundary_ties` asks for an exact page to keep every record tied at the
/// score of its last member rather than cutting through the tie. A caller
/// that orders equal scores by a key the index cannot see, such as an
/// external id string, needs the whole tie group to apply that rule at the
/// boundary. A traversal's page is never extended, because its order among
/// equal distances is the graph's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Budget {
    pub ef: Option<usize>,
    pub fetch: Option<usize>,
    /// Rerank depth per result. Read by an index that holds something to
    /// rerank against and ignored otherwise.
    pub rerank: Option<usize>,
    pub boundary_ties: bool,
}

/// What a score means, so a fusion can normalise and a caller can be told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreKind {
    /// Lower is better. Every dense metric reports one.
    Distance,
    /// Higher is better. A dot product over postings reports one.
    Similarity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub id: RecordId,
    pub score: f32,
}

/// One arm's page, with what its scores mean.
///
/// `exact` is true when the page is exact by construction, being that every
/// admitted record that could enter it was scored and the page is the best
/// of them under the index's own tie rule, which is score and then
/// `RecordId`. A term-at-a-time scan that never touches a record sharing no
/// term with the query is still exact, because that record's score is an
/// implicit zero. A safe early termination keeps it true. Only a give-up
/// budget would set it false, and a graph traversal, whose page is
/// approximate by nature, always does.
///
/// An exact page may be shorter than `k`, since a scan never pads with
/// records that scored nothing, and it may be longer than `k` by the tie
/// group at its boundary when [`Budget::boundary_ties`] asked for that. A
/// traversal's page is returned in the traversal's own order and holds at
/// most `k`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hits {
    pub items: Vec<Hit>,
    pub kind: ScoreKind,
    pub exact: bool,
}

/// What [`VectorIndex::prepare`] hands to [`VectorIndex::insert`].
///
/// Opaque to the collection. An associated type would say this exactly and
/// would cost object safety, so it is a box the implementor downcasts, and
/// `None` for an index that plans nothing.
pub struct Prepared(Option<Box<dyn Any + Send>>);

impl Prepared {
    pub fn none() -> Self {
        Prepared(None)
    }

    pub fn new<T: Any + Send>(value: T) -> Self {
        Prepared(Some(Box::new(value)))
    }

    pub fn take<T: Any>(self) -> Option<T> {
        self.0.and_then(|b| b.downcast::<T>().ok()).map(|b| *b)
    }
}

// ============================================================================
// THE INDEX
// ============================================================================

/// What every index implements. One family of vector per implementor, fixed
/// by `K`. Returns a finished page and takes an admit set.
///
/// The collection serialises mutation, so `prepare`, `insert` and `remove`
/// are never in flight together on one index, and a search may run beside
/// any of them under whatever guard the collection holds the index behind.
pub trait VectorIndex<K: Kind>: Persist + Send + Sync {
    /// Records this index holds a vector for. The live count, not the node
    /// or posting count, so a removed record leaves it.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this index holds a vector for `id`.
    ///
    /// A record may leave any space empty, so the collection asks this before
    /// it removes a record from a space, because `remove` refuses an id the
    /// index does not hold.
    fn holds(&self, id: RecordId) -> bool;

    /// The read-only half of an insertion. A graph descends and chooses the
    /// neighbour lists here, under the collection's read guard, which is what
    /// the two-phase graph insertion does. An index with nothing to plan
    /// returns `Prepared::none()`, which is the default.
    fn prepare(&self, id: RecordId, vector: K::Vector<'_>) -> Result<Prepared, Error> {
        let _ = (id, vector);
        Ok(Prepared::none())
    }

    /// Install one record's vector, with what `prepare` produced for it.
    /// Called under the collection's write guard, held for this call alone.
    ///
    /// An id the index already holds is refused. The collection removes a
    /// record before it re-adds one under a new id and never reuses an id,
    /// so a second insert under one id is a caller error rather than an
    /// update.
    fn insert(
        &mut self,
        id: RecordId,
        vector: K::Vector<'_>,
        prepared: Prepared,
    ) -> Result<(), Error>;

    /// Forget a record. Errs on an id the index does not hold.
    ///
    /// What the index does with the record's entries is its own business. A
    /// graph strands the node and relies on the admit set never naming the id
    /// again. A postings index may leave the postings in place and count them,
    /// and `stranded` reports what either has left behind.
    fn remove(&mut self, id: RecordId) -> Result<(), Error>;

    /// Entries removal has left behind and a compaction would reclaim, being
    /// stranded nodes for a graph and dead postings for a postings index. The
    /// collection's `compact` reads one figure across every space.
    fn stranded(&self) -> usize {
        0
    }

    /// The stored vector, where the index keeps one it can hand back. A
    /// quantized graph holding codes alone returns `None`, and a caller that
    /// needs the vector back asks `recover`.
    fn vector(&self, id: RecordId) -> Option<K::Vector<'_>>;

    /// The vector as the caller would want it back, built where `vector`
    /// cannot borrow one. A quantized graph reconstructs from its codes here.
    fn recover(&self, id: RecordId) -> Option<K::Owned> {
        let _ = id;
        None
    }

    /// The page. `admit` is always present, and the index conjoins it with
    /// its own live set, so a caller with no filter passes a set admitting
    /// everything. The index chooses its path, which is the planner's
    /// business only through `cost`.
    fn search(
        &self,
        query: K::Query<'_>,
        k: usize,
        admit: &dyn Admit,
        budget: &Budget,
    ) -> Result<Hits, Error>;

    /// What `search` would cost for this query and `k` under an admit set of
    /// this selectivity, and whether the page would be exact.
    ///
    /// The query is here because a postings index cannot price a search
    /// without it, since the postings it will visit are the lengths of the
    /// lists the query names and those range over an order of magnitude on
    /// one corpus. A graph ignores it and prices from `ef` and the record
    /// count.
    fn cost(&self, query: K::Query<'_>, k: usize, admitted: Option<&Selectivity>) -> Cost;
}

// ============================================================================
// PERSISTENCE
// ============================================================================

/// What a manifest records about one artefact.
///
/// `checksum` is absent for an artefact that streams itself and seeks back to
/// fill its header, where hashing it whole would mean reading it back off
/// the disk. Such an artefact carries its own checksums and the manifest
/// records its length alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtefactRecord {
    pub bytes: u64,
    pub checksum: Option<u64>,
}

/// Where a save records every artefact it writes, so the manifest written
/// last names what is really on disk.
pub trait Ledger {
    fn record(&mut self, name: &str, record: ArtefactRecord);
}

/// What a load reads an artefact's recorded length and digest from before it
/// parses the artefact.
pub trait Inventory {
    fn recorded(&self, name: &str) -> Option<ArtefactRecord>;
}

/// Ceilings the collection derives from its own artefacts before an index
/// sizes anything from a file it is restoring.
///
/// `min_records` is the live record count, which a graph must hold at least
/// as many nodes as. `max_records` is the largest internal id the collection
/// has ever issued, which no restored entry may exceed. `max_bytes` bounds
/// any single allocation an artefact asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub min_records: usize,
    pub max_records: usize,
    pub max_bytes: u64,
}

/// An index writes its own artefacts into a directory the collection owns,
/// under a prefix the collection chooses. Object safe, so a `dyn VectorIndex`
/// can be asked to save without the collection knowing its type.
pub trait Persist {
    /// Write every artefact under `prefix` into `dir`, recording each in
    /// `ledger`.
    fn write(&self, prefix: &str, dir: &Path, ledger: &mut dyn Ledger) -> Result<(), Error>;

    /// The artefact names this index would write under `prefix`, so the
    /// collection can check a directory against its inventory before parsing.
    fn artefact_names(&self, prefix: &str) -> Vec<String>;
}

/// An index restores itself. Not object safe by construction, because it
/// returns `Self`. The collection calls it on the concrete type, which is the
/// one place it names an index type to open one.
pub trait Restore: Sized {
    type Config;

    fn restore(
        config: &Self::Config,
        prefix: &str,
        dir: &Path,
        inventory: &dyn Inventory,
        bounds: &Bounds,
    ) -> Result<Self, Error>;
}

/// Write one artefact whole and return its checksum, for an index that holds
/// its artefact in memory before it writes it.
///
/// The file is fsynced before this returns, so a rename that moves the
/// directory into place cannot be recorded while the bytes it names are still
/// in the page cache.
pub fn write_artefact(dir: &Path, name: &str, bytes: &[u8]) -> Result<u64, Error> {
    use std::io::Write;

    // A prefix may carry directories, which are created on the way.
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::ArtefactCreateFailed {
            name: name.to_string(),
            error: e.to_string(),
        })?;
    }
    let mut file = std::fs::File::create(&path).map_err(|e| Error::ArtefactCreateFailed {
        name: name.to_string(),
        error: e.to_string(),
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| Error::ArtefactWriteFailed {
            name: name.to_string(),
            error: e.to_string(),
        })?;
    Ok(checksum_of(bytes))
}

/// Read one artefact whole and hold it to what the inventory recorded.
///
/// The length is checked first, because a file of the wrong length is not
/// the file the save wrote whatever its bytes hash to, and the digest second.
/// An artefact the inventory does not name is refused before it is read, so
/// a stray file cannot stand in for a missing one. `contents` says what the
/// artefact holds, for the message.
pub fn read_artefact(
    dir: &Path,
    name: &str,
    inventory: &dyn Inventory,
    contents: &'static str,
    max_bytes: u64,
) -> Result<Vec<u8>, Error> {
    let recorded = inventory
        .recorded(name)
        .ok_or_else(|| Error::ArtefactsMissing {
            missing: vec![name.to_string()],
            contents,
        })?;
    if recorded.bytes > max_bytes {
        return Err(Error::DecodeLengthExceeded {
            file: name.to_string(),
            bytes: usize::try_from(recorded.bytes).unwrap_or(usize::MAX),
        });
    }
    let bytes = std::fs::read(dir.join(name)).map_err(|e| Error::ArtefactReadFailed {
        name: name.to_string(),
        error: e.to_string(),
    })?;
    if bytes.len() as u64 != recorded.bytes {
        return Err(Error::ArtefactLengthMismatch {
            name: name.to_string(),
            actual: bytes.len(),
            recorded: recorded.bytes,
            contents,
        });
    }
    if let Some(expected) = recorded.checksum {
        let actual = checksum_of(&bytes);
        if actual != expected {
            return Err(Error::ArtefactDigestMismatch {
                name: name.to_string(),
                actual: format!("{:016x}", actual),
                expected: format!("{:016x}", expected),
                contents,
            });
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The sparse rules refuse each malformed shape by name and admit a
    /// well formed vector, an empty one included.
    #[test]
    fn a_sparse_vector_is_held_to_its_three_rules() {
        let ok = SparseVector {
            dims: vec![1, 5, 9],
            values: vec![0.5, 1.0, 2.0],
        };
        assert!(ok.as_ref().validate().is_ok());
        assert!(SparseVector::default().as_ref().validate().is_ok());

        let shape = SparseRef {
            dims: &[1, 2],
            values: &[1.0],
        };
        assert!(matches!(
            shape.validate(),
            Err(Error::SparseVectorShape { dims: 2, values: 1 })
        ));

        let order = SparseRef {
            dims: &[1, 3, 3],
            values: &[1.0, 1.0, 1.0],
        };
        assert!(matches!(
            order.validate(),
            Err(Error::SparseDimsNotIncreasing { position: 2 })
        ));

        let finite = SparseRef {
            dims: &[1, 3],
            values: &[1.0, f32::NAN],
        };
        assert!(matches!(
            finite.validate(),
            Err(Error::SparseValueNotFinite { index: 1, .. })
        ));
    }

    /// The merge and a lookup-based sum agree bit for bit, which is what
    /// makes a scan's page comparable with a brute-force one on exact
    /// equality.
    #[test]
    fn the_sparse_dot_product_accumulates_in_dimension_order() {
        let a = SparseVector {
            dims: vec![0, 2, 3, 7, 11],
            values: vec![0.1, 0.7, 1.3, 0.2, 5.5],
        };
        let b = SparseVector {
            dims: vec![2, 3, 5, 11, 12],
            values: vec![0.3, 0.9, 4.0, 0.25, 1.0],
        };
        let mut expected = 0f32;
        for (dim, value) in b.dims.iter().zip(&b.values) {
            if let Ok(i) = a.dims.binary_search(dim) {
                expected += a.values[i] * value;
            }
        }
        assert_eq!(a.as_ref().dot(b.as_ref()).to_bits(), expected.to_bits());
        assert_eq!(a.as_ref().dot(SparseVector::default().as_ref()), 0.0);
    }

    /// `Prepared` hands back what it was given and nothing else.
    #[test]
    fn a_prepared_plan_downcasts_to_its_own_type_alone() {
        let prepared = Prepared::new(42u64);
        assert_eq!(prepared.take::<u64>(), Some(42));
        let prepared = Prepared::new(42u64);
        assert_eq!(prepared.take::<u32>(), None);
        assert_eq!(Prepared::none().take::<u64>(), None);
    }

    /// A space name is any non-empty string.
    #[test]
    fn a_space_name_is_never_empty() {
        assert!(matches!(SpaceName::new(""), Err(Error::SpaceNameEmpty)));
        assert_eq!(SpaceName::new("default").unwrap().as_str(), "default");
    }

    struct Manifest(HashMap<String, ArtefactRecord>);

    impl Ledger for Manifest {
        fn record(&mut self, name: &str, record: ArtefactRecord) {
            self.0.insert(name.to_string(), record);
        }
    }

    impl Inventory for Manifest {
        fn recorded(&self, name: &str) -> Option<ArtefactRecord> {
            self.0.get(name).copied()
        }
    }

    /// An artefact written through the helper reads back through the other,
    /// and a byte flipped on disk, a wrong length, and a name the manifest
    /// does not carry are each refused before parsing.
    #[test]
    fn an_artefact_round_trips_and_every_damage_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let checksum = write_artefact(dir.path(), "a.bin", &bytes).unwrap();
        let mut manifest = Manifest(HashMap::new());
        manifest.record(
            "a.bin",
            ArtefactRecord {
                bytes: bytes.len() as u64,
                checksum: Some(checksum),
            },
        );
        assert_eq!(
            read_artefact(dir.path(), "a.bin", &manifest, "a test artefact", 1 << 20).unwrap(),
            bytes
        );

        assert!(matches!(
            read_artefact(dir.path(), "b.bin", &manifest, "a test artefact", 1 << 20),
            Err(Error::ArtefactsMissing { .. })
        ));
        assert!(matches!(
            read_artefact(dir.path(), "a.bin", &manifest, "a test artefact", 10),
            Err(Error::DecodeLengthExceeded { .. })
        ));

        let mut flipped = bytes.clone();
        flipped[500] ^= 0x40;
        std::fs::write(dir.path().join("a.bin"), &flipped).unwrap();
        assert!(matches!(
            read_artefact(dir.path(), "a.bin", &manifest, "a test artefact", 1 << 20),
            Err(Error::ArtefactDigestMismatch { .. })
        ));

        std::fs::write(dir.path().join("a.bin"), &bytes[..999]).unwrap();
        assert!(matches!(
            read_artefact(dir.path(), "a.bin", &manifest, "a test artefact", 1 << 20),
            Err(Error::ArtefactLengthMismatch { .. })
        ));
    }
}
