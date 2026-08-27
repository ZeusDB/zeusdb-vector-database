//! The vector store, addressed by node index.
//!
//! One contiguous allocation, node `n` at `[n * dim, (n + 1) * dim)`. It is a
//! separate object from the graph rather than a field inside it, and the graph
//! is handed a reference to it for the length of an operation.
//!
//! # Why it is not inside the graph
//!
//! The graph's own arena was measured against a `Vec<Vec<T>>` of separate
//! allocations and against a hash map keyed by external id, and the arena wins.
//! A single contiguous block the graph reaches through a pointer was measured
//! too, and costs nothing measurable at any of three dimensions.
//!
//! What separating it buys is that the block is addressable, dumpable and
//! replaceable on its own. A vector can be fetched by an integer without the
//! adjacency being present, which is the shape every comparator with a storage
//! layer has and the shape a future that pages vectors from object storage
//! needs. Nothing in this release does either, so that part is a bet rather
//! than a measurement.
//!
//! # The one copy rule
//!
//! A raw vector is held once. On a raw graph the store is the raw vectors and
//! nothing else holds them. On a quantized graph the store is the codes, and
//! the raw vectors, where the storage mode keeps them, are in a second store
//! beside it. `HNSWIndex` holds no vector map of its own.

/// A contiguous block of stored values, addressed by node index.
///
/// The width is fixed at construction and every node occupies exactly that
/// many values, so a node's slice is one multiplication away and there is no
/// per record allocation, no header and no hashing.
pub(crate) struct VectorStore<T> {
    /// Values per stored vector.
    dim: usize,
    /// Every vector, node `n` at `[n * dim, (n + 1) * dim)`.
    data: Vec<T>,
}

impl<T> VectorStore<T> {
    /// An empty store of the given width, with room reserved for `records`.
    ///
    /// **The width is at least one.** A store of no width has no node it can
    /// address, since every node would occupy zero values and every slice
    /// would be empty, so it is a state rather than a store. The two callers
    /// that reach here from outside this module both refuse a zero before they
    /// get here, and the assertion is what states that rather than leaving it
    /// to be rediscovered. See [`Self::len`].
    pub(crate) fn with_capacity(dim: usize, records: usize) -> Self {
        debug_assert!(dim > 0, "a store of no width can address no node");
        VectorStore {
            dim,
            data: Vec::with_capacity(records.saturating_mul(dim)),
        }
    }

    /// Values per stored vector.
    #[inline]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// Vectors the store holds, which is the node count of the graph it
    /// belongs to.
    ///
    /// # The width is never zero, and this is still total
    ///
    /// Every store is built by [`Self::with_capacity`], and both of its
    /// callers refuse a zero width before they call it. `MutableGraph::new`
    /// returns an error on a zero dimension and `Backend::sized` clamps to one
    /// before it, and `VectorGraph::open_raw_store` refuses one outright. A
    /// zero here would therefore be a bug in this crate rather than bad input,
    /// which is what the debug assertion on the constructor says.
    ///
    /// It is written as a checked division anyway. The cost is nothing, since
    /// the hardware division already faults on zero and `checked_div` is the
    /// same instruction with the branch the compiler was going to emit; and
    /// what it buys is that an accessor stays total, so a future constructor
    /// that gets this wrong reports a count of zero rather than panicking
    /// inside a `get_stats` call. This was an `if self.dim == 0` guard, which
    /// says the same thing and which `clippy::manual_checked_ops` flags as of
    /// Rust 1.97.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    /// One node's stored vector.
    ///
    /// This is the whole of what a distance evaluation costs to reach a
    /// vector, and it is the reason the block is contiguous and the width
    /// fixed.
    #[inline]
    pub(crate) fn get(&self, node: u32) -> &[T] {
        let at = node as usize * self.dim;
        &self.data[at..at + self.dim]
    }

    /// One node's stored vector, or `None` where the store does not reach that
    /// far.
    #[inline]
    pub(crate) fn try_get(&self, node: u32) -> Option<&[T]> {
        let at = node as usize * self.dim;
        self.data.get(at..at + self.dim)
    }

    /// Bytes the block has asked the allocator for, the header included.
    pub(crate) fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.capacity() * std::mem::size_of::<T>()
    }

    /// Return the block's spare capacity to the allocator.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }
}

impl<T: Clone> VectorStore<T> {
    /// Append one vector, which becomes the next node.
    ///
    /// The width is asserted rather than checked, because every caller has
    /// already been through the index's own validation and a mismatch here is
    /// a bug in this crate rather than bad input.
    #[inline]
    pub(crate) fn push(&mut self, values: &[T]) {
        assert_eq!(
            values.len(),
            self.dim,
            "a vector of {} values was offered to a store of {}",
            values.len(),
            self.dim
        );
        self.data.extend_from_slice(values);
    }
}

impl<T> VectorStore<T> {
    /// Append one vector by value, for the loader, which owns what it read.
    pub(crate) fn append(&mut self, values: Vec<T>) {
        assert_eq!(
            values.len(),
            self.dim,
            "a vector of {} values was offered to a store of {}",
            values.len(),
            self.dim
        );
        self.data.extend(values);
    }
}
