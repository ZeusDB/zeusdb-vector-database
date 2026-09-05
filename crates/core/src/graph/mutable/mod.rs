//! The mutable form of the graph, which is what construction writes into.
//!
//! The dump's own form is read-only by construction. Its node order is layer
//! major, so appending a node of level `l` means inserting into the middle of
//! the arena and renumbering every higher-layer node, which invalidates every
//! target. This is the same graph in a form a node can be appended to.
//!
//! # Node identity
//!
//! A node is a `u32` handed out in the order nodes arrive, and it never
//! changes. Loading adopts the dump's own order, which is layer major, so a
//! loaded graph numbers its nodes exactly as the dump does and the two can be
//! compared node for node. Insertion appends at the end whatever level it
//! drew. The level is carried per node in one byte rather than implied by the
//! node's position, which is the whole difference.
//!
//! Because the order is arbitrary rather than layer major, a save has to
//! recover the dump's (layer, rank) identity. [`DumpOrder`] does that in one
//! counting pass at save time, which is where the cost belongs: it is paid once
//! per save rather than held for the life of the index.
//!
//! # Adjacency
//!
//! Fixed-capacity slabs of targets, and no stored distance.
//!
//! Layer zero needs no indirection, because every node owns a layer zero slab:
//! node `n` holds slots `[n * base_cap, n * base_cap + base_cap)`. Only the
//! upper layers are sparse, holding `1 / (m - 1)` lists per node at the default
//! scale, so a node carries one `u32` naming its first upper list word and the
//! words of a node of level `L` are the `L` consecutive ones from there.
//!
//! **The graph holds targets alone.** The distance between a list's owner and
//! a target is the kernel's value between two stored vectors, both immutable,
//! and every kernel is symmetric to the bit, so the value is the same whichever
//! of the two is the query. The builder therefore recomputes it where it reads
//! it rather than holding an `f32` beside every target, which is what a layer
//! zero slab of `2 * m + 1` slots used to spend half its bytes on. What the
//! builder reads is bounded: a reverse link is placed into a list that is
//! already ordered, so [`MutableGraph::place_last`] evaluates the distance to
//! the entries a binary search touches and to nothing else. The dump writer
//! recomputes every edge's distance at save time, so the file carries what it
//! carried when the distances were stored. The traversal reads no stored
//! distance, since it scores every neighbour against the query.
//!
//! ## The capacity is one slot above the vendored threshold
//!
//! `2 * m + 1` at layer zero and `m + 1` above. The vendored reverse update
//! pushes into a neighbour's list first and shrinks second, testing
//! `len > threshold` after the push, so a list transiently holds one more entry
//! than the threshold while patch 3's guarded pop chooses which entry to evict,
//! and the entry just pushed is itself a candidate. Reproducing that guard
//! means the structure has to be able to hold that state, so the overflow slot
//! is part of the layout rather than an implementation detail. A push into a
//! list already at capacity is refused: it is a state the vendored algorithm
//! cannot reach, so it is a bug rather than a growth event.
//!
//! # Lists above a node's own level
//!
//! **A node owns a list at every layer up to its span, and its span can exceed
//! its level.** This is what the vendored structure has always done, and what
//! the layout here reproduces: `Point::new` gives every point
//! sixteen layer vectors whatever level it was drawn at, and the builder writes
//! into the ones above the level from two directions.
//!
//! The descent writes there. Install site 1 records the entry point it passed at
//! each layer above the new point's own level into the new point's list there,
//! so a level zero point carries entries at layer five. That is the descent
//! residue.
//!
//! The reverse update writes there too, and that is what an append-only region
//! could not hold. `search_layer` seeds its result heap with the point it is
//! entered at whatever that point's level, so a search at layer `l` can return a
//! point of level below `l`. Every time a new point draws a level above every
//! level drawn before it, its own search at that layer is entered at the old
//! entry point and returns it, so the old entry point takes a reverse link at a
//! layer above its own level. From there it is reachable at that layer, so the
//! link is not a one-off: measured on a 2,000 point cosine graph at `m` 16,
//! 4,289 above-level lists exist, 4,287 of them holding one entry and two
//! holding ten and sixteen. **An above-level list grows, is ordered, is evicted
//! from and is counted into, exactly as a list at or below the level is.**
//!
//! So the layout carries them as ordinary lists, in two forms. A list at or
//! below its owner's level is **wide**: a descriptor holding an offset, a
//! length and an inbound counter, and `m + 1` slots in the target arena,
//! because install site 2 fills it wholesale. A list above the level is one
//! **word**, which is what 99.95 percent of them ever hold. The word is the
//! target itself where the list holds one entry, [`WORD_EMPTY`] where it holds
//! none, and otherwise the index of the wide descriptor the list was promoted
//! to. A word is promoted on the push that gives its list a second entry and
//! on the first time its owner is named at that layer, because the inbound
//! counter lives on the wide descriptor. Promotion fires on the order of the
//! entry level per graph, being the old entry points' lists. A descent residue
//! list is never pushed into and its owner is never named at that layer, since
//! no traversal reaches a node above its level except through the entry point
//! chain, so it stays one word for the life of the graph.
//!
//! Keeping these lists is not a choice. The vendored
//! descent increments the pivot's inbound counter at the layer it files the edge
//! at, and the overflow pop guard reads that counter, so a build that skips them
//! evicts different entries. And the traversal reads them, so a build that
//! cannot see them chooses different neighbours. See [`insert`].
//!
//! # The inbound counters
//!
//! One `u32` per node at layer zero and one per wide upper list, which is what
//! the guarded overflow pop reads. Every counter the vendored crate touches is
//! at a layer the target owns a list at, so the counters need no table of their
//! own: they sit in step with the lists they count into, and a list held as one
//! word is promoted the first time it is counted into.
//!
//! The vendored equivalent is a `[AtomicU32; 16]` on every point whatever its
//! level, which is 64 bytes per node against about four here at `m` 16, and
//! which carries a recorded read-then-act race on the parallel insertion path.
//! These are plain integers because the mutator is serialised.

use super::dump::EachNeighbourhood;
use super::dump::{LoadedEdge, LoadedPoint, PointId};
use super::store::VectorStore;
use super::traverse::{self, Topology, LAYERS};
use super::{Distance, GraphHit};

mod insert;

/// What phase one hands phase two, re-exported so the seam can name it without
/// the arena's own module being visible outside `mutable`.
pub use insert::Planned;

/// Naming no upper list, which is every node whose span is zero.
const NO_UPPER: u32 = u32::MAX;

/// Naming no node, which is every internal id this graph never took.
const NO_NODE: u32 = u32::MAX;

/// An upper list word holding no entry.
const WORD_EMPTY: u32 = u32::MAX;

/// The bit that marks an upper list word as the index of a wide descriptor
/// rather than a target.
///
/// A target is a node index, so a node index has to stay below this bit and
/// away from [`WORD_EMPTY`]. [`MutableGraph::append_node`] refuses the node
/// that would reach it, which is the 2,147,483,648th; at the smallest layer
/// zero slab a graph of that many nodes holds twenty gigabytes of targets.
const WORD_WIDE: u32 = 1 << 31;

/// Bytes the creation-time reservation may ask the allocator for.
///
/// `Vec::with_capacity` aborts the process on allocation failure rather than
/// unwinding, so a reservation too large for the machine cannot be turned into a
/// Python exception after the fact. `expected_size` is a hint that bounds no
/// behaviour, so capping what it reserves costs a caller nothing beyond the
/// reallocations of a build larger than this budget. See [`MutableGraph::new`].
///
/// 128 mebibytes holds 21,224 records at dimension 1,536 and `m` 16, 199,728 at
/// dimension 128, and 663,956 at the dimension 8 that
/// `test_empty_index_at_a_large_declared_size_stays_under_the_bound` declares
/// five million records at. That test is what fixes the order of magnitude: it
/// requires an empty index to commit under 256 MB and under 64 bytes per
/// declared record whatever it declares, and this budget is what keeps both true
/// now that a declared record reserves the graph's own copy of its vector rather
/// than one pointer.
///
/// What it costs is reallocation on a build past the budget. The arenas grow
/// geometrically, so a 50,000 record build at dimension 1,536 crosses it twice
/// and moves roughly 400 MB in total, against a build that takes 70 seconds.
pub(super) const RESERVE_BYTES: usize = 1 << 27;

/// Records the reservation is taken for, being `expected_size` or as many as
/// [`RESERVE_BYTES`] holds, whichever is smaller.
///
/// The per record cost is the whole of what [`MutableGraph::new`] reserves, in
/// step with what [`MutableGraph::memory_bytes`] prices: the six per node
/// arrays, the graph's own copy of the vector, the layer zero slab, one upper
/// list word per expected span, and the share of a wide list a node is expected
/// to own, which is one list of `m + 1` slots and a ten byte descriptor for
/// every `m - 1` nodes.
pub(super) fn reserved_records<T>(
    dim: usize,
    m: usize,
    span: usize,
    expected_size: usize,
) -> usize {
    const PER_NODE_ARRAYS: usize = 8 + 1 + 2 + 4 + 4 + 1;
    let per_record = PER_NODE_ARRAYS
        + dim * std::mem::size_of::<T>()
        + (2 * m + 1) * std::mem::size_of::<u32>()
        + span * std::mem::size_of::<u32>()
        + wide_list_bytes(m).div_ceil(wide_lists_per(m));
    expected_size.min((RESERVE_BYTES / per_record.max(1)).max(1))
}

/// Bytes one wide upper list costs, being its descriptor and its `m + 1`
/// slots.
fn wide_list_bytes(m: usize) -> usize {
    4 + 2 + 4 + (m + 1) * std::mem::size_of::<u32>()
}

/// Nodes per wide upper list a graph at `m` is expected to hold.
///
/// A node of level `L` owns `L` lists at or below its level, and the expected
/// level under the exponential law at scale `1 / ln(m)` is `1 / (m - 1)`. So
/// one node in `m - 1` owns a wide list, before the handful the entry point
/// chain promotes.
fn wide_lists_per(m: usize) -> usize {
    m.saturating_sub(1).max(1)
}

/// Upper lists a node of a graph this size is expected to own, which is the
/// entry level such a graph reaches rather than the expected level of one node.
///
/// The descent files an edge into the new point's list at every layer from the
/// entry level down to its own level plus one, so a node's span is the entry
/// level at the moment it arrived. `log(n) / log(m)` is where the top of an
/// exponential level distribution at scale `1 / ln(m)` lands.
fn expected_span(m: usize, expected_size: usize) -> usize {
    if expected_size < 2 || m < 2 {
        return 0;
    }
    let span = (expected_size as f64).ln() / (m as f64).ln();
    (span.ceil() as usize).min(LAYERS - 1)
}

/// One chosen edge, as phase one hands it to phase two and as the sort orders
/// it.
///
/// This type used to carry no derives at all, because the sorts were
/// `sort_unstable` and that dispatches on the element type: a `Copy` pair of
/// eight bytes takes a different path from a non-`Copy` one and orders equal
/// keys differently from length 21 up. Reproducing the vendored builder's
/// permutation meant matching its `Vec<Arc<PointWithOrder>>` on both counts.
///
/// The sorts are stable now, so the permutation over equal keys is the
/// insertion order whatever the element is, and the constraint is gone with the
/// sort that imposed it. `the_entry_sort_keeps_its_insertion_order` holds the
/// property that replaced it.
///
/// The width still matters, for a different reason. A stable sort takes scratch
/// space, and the standard library serves it from a 4 KiB stack buffer whenever
/// the request fits, reaching the allocator only above that. At eight bytes a
/// list would have to run past 512 entries to allocate, and the longest one the
/// index can build is `2 * m + 1`, which is 129 at the largest `m` accepted.
/// `an_entry_is_eight_bytes` holds the width so that argument stays checkable.
#[derive(Clone, Copy)]
pub(super) struct Entry {
    /// Distance from the new point to the target, as the traversal computed it.
    pub(super) dist: f32,
    /// The node the edge points at.
    pub(super) target: u32,
}

/// The order both sorts impose, which is the vendored `PointWithOrder` order:
/// the distance alone, with the same panic on a NaN.
fn by_distance(a: &Entry, b: &Entry) -> std::cmp::Ordering {
    if !a.dist.is_nan() && !b.dist.is_nan() {
        a.dist.partial_cmp(&b.dist).unwrap()
    } else {
        panic!("got a NaN in a distance");
    }
}

/// A list of entries at the given distances, for the sort order test.
#[cfg(test)]
pub(super) fn entries_for_test(dists: &[f32]) -> Vec<Entry> {
    dists
        .iter()
        .enumerate()
        .map(|(i, &dist)| Entry {
            dist,
            target: i as u32,
        })
        .collect()
}

/// The sort every list maintenance site runs, for the sort order test.
#[cfg(test)]
pub(super) fn sort_entries_for_test(entries: &mut [Entry]) {
    entries.sort_by(by_distance);
}

/// Which arena one list's slots sit in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slab {
    /// The layer zero slab, `2 * m + 1` slots a node.
    Base,
    /// One upper list word, holding the target itself or nothing.
    Inline,
    /// A wide upper list, `m + 1` slots in the upper target arena.
    Wide,
}

/// The mutable graph: an arena of nodes, a vector slab, and slab adjacency.
///
/// Generic exactly as the vendored `Hnsw` is generic, over the element type the
/// vectors hold and the distance evaluated between a query and a stored vector.
pub(super) struct MutableGraph<T, D> {
    /// Values per stored vector.
    dim: usize,
    /// `max_nb_connection`.
    m: usize,
    /// Width the insertion traversal runs at.
    ef_construction: usize,
    /// The level generator's scale, as a dump records it.
    level_scale: f64,
    /// Node the traversal starts from.
    entry: u32,
    /// The entry node's level, which is the highest occupied layer.
    entry_level: u8,
    /// `layer_counts[l]` is the number of nodes whose level is exactly `l`,
    /// which is what the dump's layer table records.
    layer_counts: [u32; LAYERS],

    /// The id each node was inserted under, which hits report.
    origin_ids: Vec<usize>,
    /// Each node's top level.
    levels: Vec<u8>,
    /// The node each internal id sits at, indexed by that id, and
    /// [`NO_NODE`] where the graph never took one.
    ///
    /// The inverse of [`Self::origin_ids`], written by the same append that
    /// writes it, so the two cannot drift. It is what turns the external id a
    /// caller asks about into the node index the store is addressed by. Held
    /// here rather than beside the index because it changes exactly when the
    /// graph changes, which means it is covered by the graph's own lock and
    /// there is no second structure to keep in step.
    node_of: Vec<u32>,

    /// Layer zero targets, node `n` at `[n * base_cap, ..)`.
    base_targets: Vec<u32>,
    /// Entries in each node's layer zero list.
    base_len: Vec<u16>,
    /// Lists at layer zero naming this node, which the overflow pop guard
    /// reads before it evicts.
    base_in_degree: Vec<u32>,

    /// The first upper list word a node owns, or [`NO_UPPER`] where it owns
    /// none. A node's words for layers `1..=span` are the consecutive ones
    /// from there.
    upper_first: Vec<u32>,
    /// The highest layer each node owns a list at, which is at least its level
    /// and can exceed it. Zero means it owns no upper list.
    upper_span: Vec<u8>,

    /// One word per upper list: the target where the list holds one entry,
    /// [`WORD_EMPTY`] where it holds none, and the index of a wide
    /// descriptor under [`WORD_WIDE`] where it has been promoted.
    upper_word: Vec<u32>,

    /// Where each wide list's `m + 1` slots start in [`Self::upper_targets`].
    wide_at: Vec<u32>,
    /// Entries in each wide list.
    wide_len: Vec<u16>,
    /// Lists naming this wide list's owner at this list's layer, in step
    /// with [`Self::wide_len`].
    wide_in_degree: Vec<u32>,

    /// Upper layer targets, one wide list's slots at `[wide_at[i], ..)`.
    upper_targets: Vec<u32>,

    /// Times the reverse update pushed an entry past its layer's cap and had to
    /// remove one. The vendored counterpart is a process-wide static, which
    /// cannot be attributed to one graph; this is per graph.
    overflows: u64,
    /// Times the guard skipped the farthest entry because removing it would
    /// have taken its target to zero inbound links.
    saves: u64,
    /// Times no candidate qualified and the guard removed the farthest entry,
    /// which is the unmodified crate behaviour.
    fallbacks: u64,

    /// The distance, evaluated between the query and a stored vector.
    dist_f: D,

    /// The element type the store beside this graph holds.
    ///
    /// The graph itself holds no vector, so nothing else here mentions `T`.
    /// Keeping the parameter is what stops a graph of one element type being
    /// bound to a store of another, which is the mistake the type system was
    /// catching before the vectors moved out.
    _elem: std::marker::PhantomData<T>,
}

impl<T, D> MutableGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// Build the structure from the topology the dump reader produces.
    ///
    /// The same acceptance rules as the vendored `Hnsw::from_loaded_points`,
    /// because both are fed the same parsed dump and have to agree on what a
    /// malformed one is, plus one rule the other does not need. A neighbour
    /// list longer than the cap its layer allows is
    /// refused, since the vendored builder cannot produce one and this layout
    /// caps a list at what the guarded pop needs. A refusal reaches the loader's
    /// rebuild path, which is what every other refusal reaches too.
    ///
    /// The file's distances order each list and are then dropped, since the
    /// structure holds targets alone. Every one of them is checked finite
    /// first, which is the rule the vendored constructor applied to them.
    ///
    /// Nodes are numbered in the order the dump streams them, which is layer
    /// major, so a loaded graph's node indices match the dump's exactly.
    pub(super) fn from_loaded(
        points_by_layer: Vec<Vec<LoadedPoint<T>>>,
        entry_point: PointId,
        m: usize,
        ef_construction: usize,
        level_scale: f64,
        dist_f: D,
    ) -> Result<(Self, VectorStore<T>), String> {
        let nb_layer = points_by_layer.len();
        if nb_layer == 0 || nb_layer > LAYERS {
            return Err(format!(
                "a graph carries between 1 and {} layers and this one carries {}",
                LAYERS, nb_layer
            ));
        }
        if m == 0 || m > 256 {
            return Err(format!(
                "max_nb_connection is between 1 and 256 and this graph declares {}",
                m
            ));
        }
        if !level_scale.is_finite() || level_scale <= 0. {
            return Err(format!(
                "the level scale is a positive finite number and this graph declares {}",
                level_scale
            ));
        }

        let mut layer_counts = [0u32; LAYERS];
        let mut layer_offsets = [0u32; LAYERS + 1];
        let mut nb_point: usize = 0;
        for (layer, points) in points_by_layer.iter().enumerate() {
            if points.len() > i32::MAX as usize {
                return Err(format!(
                    "layer {} holds {} points and a rank is an i32",
                    layer,
                    points.len()
                ));
            }
            nb_point += points.len();
        }
        if nb_point == 0 {
            return Err("a graph holding no points has no entry point".to_string());
        }
        if nb_point > u32::MAX as usize {
            return Err(format!(
                "the graph holds {} points and a node index is a u32",
                nb_point
            ));
        }
        if nb_point >= WORD_WIDE as usize {
            return Err(format!(
                "the graph holds {} points and an upper list word names a node below {}",
                nb_point, WORD_WIDE
            ));
        }
        for layer in 0..LAYERS {
            let count = points_by_layer.get(layer).map_or(0, Vec::len) as u32;
            layer_counts[layer] = count;
            layer_offsets[layer + 1] = layer_offsets[layer] + count;
        }

        let dim = points_by_layer
            .iter()
            .find_map(|layer| layer.first())
            .map(|point| point.data.len())
            .expect("nb_point is positive, so some layer holds a point");

        let base_cap = 2 * m + 1;

        let mut graph = MutableGraph {
            dim,
            m,
            ef_construction,
            level_scale,
            entry: 0,
            entry_level: 0,
            layer_counts,
            origin_ids: Vec::with_capacity(nb_point),
            levels: Vec::with_capacity(nb_point),
            node_of: Vec::new(),
            base_targets: vec![0u32; nb_point * base_cap],
            base_len: vec![0u16; nb_point],
            base_in_degree: vec![0u32; nb_point],
            upper_first: Vec::with_capacity(nb_point),
            upper_span: Vec::with_capacity(nb_point),
            upper_word: Vec::new(),
            wide_at: Vec::new(),
            wide_len: Vec::new(),
            wide_in_degree: Vec::new(),
            upper_targets: Vec::new(),
            overflows: 0,
            saves: 0,
            fallbacks: 0,
            dist_f,
            _elem: std::marker::PhantomData,
        };

        let mut store = VectorStore::with_capacity(dim, nb_point);
        let mut lists: Vec<Vec<(f32, u32)>> = Vec::new();
        let mut targets: Vec<u32> = Vec::new();
        for (layer, points) in points_by_layer.into_iter().enumerate() {
            for (rank, point) in points.into_iter().enumerate() {
                let node = layer_offsets[layer] + rank as u32;
                if point.data.len() != dim {
                    return Err(format!(
                        "the point at layer {} rank {} holds {} values where the graph holds {}",
                        layer,
                        rank,
                        point.data.len(),
                        dim
                    ));
                }
                if point.neighbours.len() > LAYERS {
                    return Err(format!(
                        "the point at layer {} rank {} carries adjacency for {} layers and \
                         a point carries at most {}",
                        layer,
                        rank,
                        point.neighbours.len(),
                        LAYERS
                    ));
                }
                graph.origin_ids.push(point.origin_id);
                graph.levels.push(layer as u8);
                graph.note_origin(point.origin_id, node);
                store.append(point.data);

                // Every edge is checked wherever it sits, and each list is
                // ordered by its stored distances with the same stable sort the
                // vendored constructor applies, so ties keep their dump order
                // there and here.
                lists.clear();
                for list in point.neighbours.into_iter() {
                    let mut edges: Vec<(f32, u32)> = Vec::with_capacity(list.len());
                    for edge in &list {
                        edges.push((edge.distance, edge_target(edge, &layer_offsets)?));
                    }
                    edges.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0)
                            .expect("every distance was checked finite by edge_target")
                    });
                    lists.push(edges);
                }

                // The span the dump asks for, which is the highest layer the
                // point carries anything at and at least its own level. The
                // vendored point carries sixteen lists whatever its level, so a
                // span above the level is a property of the file rather than a
                // malformation.
                let mut span = layer;
                for (list_layer, list) in lists.iter().enumerate() {
                    if !list.is_empty() {
                        span = span.max(list_layer);
                    }
                }
                graph.open_node(span, layer);

                for (list_layer, list) in lists.iter().enumerate() {
                    if list.is_empty() {
                        continue;
                    }
                    // A list above its owner's level is one word, which holds
                    // one entry, so a loaded one carrying more is promoted
                    // rather than refused. Refusal is for a list past the cap
                    // its layer allows at all.
                    let cap = if list_layer == 0 {
                        base_cap
                    } else {
                        graph.upper_cap_full()
                    };
                    if list.len() > cap {
                        return Err(format!(
                            "the point at layer {} rank {} carries {} neighbours at layer {} \
                             and a list holds {}",
                            layer,
                            rank,
                            list.len(),
                            list_layer,
                            cap
                        ));
                    }
                    targets.clear();
                    targets.extend(list.iter().map(|&(_, target)| target));
                    graph.install_loaded_list(node, list_layer, &targets);
                }
            }
        }
        // The inbound counters, rebuilt from the edges just installed exactly as
        // `Hnsw::from_loaded_points` rebuilds its own. Every edge counts at the
        // layer its own list sits at, the lists above a node's level included,
        // because the vendored install site counts those too and the overflow
        // pop guard then reads the total.
        //
        // A second pass rather than a bump beside each install, because an edge
        // may name a node the walk has not reached yet. A target named at a
        // layer it owns no list at takes one, which is a state the vendored
        // builder does not reach and which the fixed sixteen slot counter array
        // on a vendored point would have absorbed silently.
        let mut inbound: Vec<(u32, usize)> = Vec::new();
        for node in 0..graph.origin_ids.len() as u32 {
            for layer in 0..=graph.span(node) {
                for slot in 0..graph.list_len(node, layer) {
                    inbound.push((graph.target_at(node, layer, slot), layer));
                }
            }
        }
        for &(target, layer) in &inbound {
            graph.grow_span(target, layer);
            graph.bump_in_degree(target, layer, 1);
        }

        // Trimming makes every buffer's capacity its length, which is what lets
        // `memory_bytes` be checked against the counts rather than believed. The
        // upper regions are the ones whose size is not known before the walk,
        // because a span is a property of the file, and the counters' pass
        // above promotes the lists the entry point chain named, so the trim
        // comes after it.
        graph.node_of.shrink_to_fit();
        graph.upper_word.shrink_to_fit();
        graph.wide_at.shrink_to_fit();
        graph.wide_len.shrink_to_fit();
        graph.wide_in_degree.shrink_to_fit();
        graph.upper_targets.shrink_to_fit();

        let entry_layer = entry_point.0 as usize;
        let entry_span = if entry_layer < LAYERS {
            layer_counts[entry_layer]
        } else {
            0
        };
        if entry_layer >= LAYERS || entry_point.1 < 0 || entry_point.1 as u32 >= entry_span {
            return Err(format!(
                "the entry point sits at layer {} rank {} and no point is there",
                entry_point.0, entry_point.1
            ));
        }
        graph.entry = layer_offsets[entry_layer] + entry_point.1 as u32;
        graph.entry_level = entry_point.0;

        Ok((graph, store))
    }

    /// An empty graph, ready to be built by insertion.
    ///
    /// `expected_size` sizes the reservation and nothing else. It is the same
    /// hint `Hnsw::new` takes and it bounds no behaviour, so a graph that
    /// outgrows it grows geometrically from there. Overshooting it costs the
    /// per node arithmetic in `memory_bytes` per unused record, of which the
    /// layer zero slab at `(2m + 1) * 4` is nearly all; the unpatched vendored
    /// reservation this replaces cost 3,025 bytes per unused record.
    ///
    /// The upper words are reserved at one per expected span per node, which
    /// is the entry level a graph this size reaches rather than a bound, and
    /// the wide lists at one per `m - 1` nodes, which is the expected level.
    /// Both are expectations rather than bounds, so they are the regions a
    /// build can grow past.
    ///
    /// # The reservation is capped in bytes
    ///
    /// A declared record costs far more here than it did in the structure this
    /// replaces, because the reservation covers the graph's own copy of the
    /// vector and its layer zero slab where the vendored one covered a single
    /// `Arc` slot. That was measured at 8.02 bytes per declared record, and
    /// `expected_size` is capped at 100 million on the strength of it, which put
    /// the creation-time reservation at 764 MB. The same declaration here at
    /// dimension 1,536 asks for 632 GB, and `Vec::with_capacity` aborts the
    /// process on allocation failure rather than unwinding, so that is a
    /// declaration a caller could make and never see an exception for.
    ///
    /// [`RESERVE_BYTES`] is what stops it. The reservation is taken for as many
    /// records as fit in that budget and no more, and a build that passes the
    /// budget grows geometrically from there exactly as one that passed
    /// `expected_size` would. A caller cannot observe the difference except in
    /// the reallocations of a build larger than the budget, which is the trade
    /// made in place of aborting.
    pub(super) fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        level_scale: f64,
        expected_size: usize,
        dist_f: D,
    ) -> Result<(Self, VectorStore<T>), String> {
        if dim == 0 {
            return Err("a graph holds vectors of at least one value".to_string());
        }
        if m == 0 || m > 256 {
            return Err(format!(
                "max_nb_connection is between 1 and 256 and this graph declares {}",
                m
            ));
        }
        if !level_scale.is_finite() || level_scale <= 0. {
            return Err(format!(
                "the level scale is a positive finite number and this graph declares {}",
                level_scale
            ));
        }
        let base_cap = 2 * m + 1;
        // The expected span, which is what sizes the word arena. Almost every
        // node ends up owning a list at every layer from one up to the entry
        // level, because the descent files one there, so the arena is sized by
        // the entry level a graph of `expected_size` points reaches rather than
        // by the far smaller expected level. It is a property of the declared
        // size rather than of the reservation, so it is taken before the cap.
        let span = expected_span(m, expected_size);
        let reserved = reserved_records::<T>(dim, m, span, expected_size);
        let words = reserved * span;
        let wide = reserved.div_ceil(wide_lists_per(m));
        let store = VectorStore::with_capacity(dim, reserved);
        Ok((
            MutableGraph {
                dim,
                m,
                ef_construction,
                level_scale,
                entry: 0,
                entry_level: 0,
                layer_counts: [0u32; LAYERS],
                origin_ids: Vec::with_capacity(reserved),
                levels: Vec::with_capacity(reserved),
                node_of: Vec::with_capacity(reserved + 1),
                base_targets: Vec::with_capacity(reserved * base_cap),
                base_len: Vec::with_capacity(reserved),
                base_in_degree: Vec::with_capacity(reserved),
                upper_first: Vec::with_capacity(reserved),
                upper_span: Vec::with_capacity(reserved),
                upper_word: Vec::with_capacity(words),
                wide_at: Vec::with_capacity(wide),
                wide_len: Vec::with_capacity(wide),
                wide_in_degree: Vec::with_capacity(wide),
                upper_targets: Vec::with_capacity(wide * (m + 1)),
                overflows: 0,
                saves: 0,
                fallbacks: 0,
                dist_f,
                _elem: std::marker::PhantomData,
            },
            store,
        ))
    }

    /// Record where an internal id landed.
    ///
    /// Written by the same append that pushes to `origin_ids`, so the map and
    /// its inverse are one operation and cannot disagree. Indexed by the id
    /// itself rather than by the id less one, which costs one slot and spares
    /// every caller an off-by-one about an id space that starts at one.
    fn note_origin(&mut self, origin_id: usize, node: u32) {
        if self.node_of.len() <= origin_id {
            self.node_of.resize(origin_id + 1, NO_NODE);
        }
        self.node_of[origin_id] = node;
    }

    /// The node one internal id sits at, or `None` where this graph never took
    /// that id.
    ///
    /// A removed record keeps its entry until the graph is replaced, because
    /// removal strands a node rather than deleting it. `id_map` is the record
    /// set and every caller consults it first, so a stranded entry is
    /// unreachable rather than wrong.
    pub(super) fn node_of(&self, origin_id: usize) -> Option<u32> {
        match self.node_of.get(origin_id) {
            None | Some(&NO_NODE) => None,
            Some(&node) => Some(node),
        }
    }

    /// Slots one layer zero list holds, being the vendored threshold plus the
    /// overflow slot the guarded pop needs.
    #[inline]
    fn base_cap(&self) -> usize {
        2 * self.m + 1
    }

    /// Slots one wide upper list holds, on the same rule.
    #[inline]
    fn upper_cap_full(&self) -> usize {
        self.m + 1
    }

    /// The highest layer one node owns a list at, which is at least its level.
    #[inline]
    fn span(&self, node: u32) -> usize {
        self.upper_span[node as usize] as usize
    }

    /// Which word holds one node's list at one layer, and `None` where the
    /// node owns none there.
    #[inline]
    fn upper_list(&self, node: u32, layer: usize) -> Option<usize> {
        if layer == 0 || layer > self.span(node) {
            return None;
        }
        Some(self.upper_first[node as usize] as usize + layer - 1)
    }

    /// The wide descriptor one word names, or `None` where the word holds the
    /// list itself.
    #[inline]
    fn wide_of(&self, list: usize) -> Option<usize> {
        let word = self.upper_word[list];
        if word & WORD_WIDE != 0 && word != WORD_EMPTY {
            Some((word & !WORD_WIDE) as usize)
        } else {
            None
        }
    }

    /// Nodes the graph holds.
    pub(super) fn nb_points(&self) -> usize {
        self.origin_ids.len()
    }

    /// The distance this graph was built with.
    #[inline]
    pub(super) fn distance(&self) -> &D {
        &self.dist_f
    }

    /// Where the traversal starts.
    #[inline]
    pub(super) fn entry(&self) -> u32 {
        self.entry
    }

    /// The entry node's top level, which is the highest occupied layer.
    #[inline]
    pub(super) fn entry_level(&self) -> u8 {
        self.entry_level
    }

    /// Nodes whose top level is exactly `layer`.
    #[inline]
    pub(super) fn layer_len(&self, layer: usize) -> usize {
        self.layer_counts[layer] as usize
    }

    /// One node's neighbour list at one layer, empty where it has none.
    ///
    /// Named apart from the trait method because the trait is implemented on
    /// [`Bound`] rather than here, and the traversal reaches this through that.
    #[inline]
    pub(super) fn neighbours_at(&self, node: u32, layer: usize) -> &[u32] {
        match self.slice_of(node, layer) {
            Some((at, len, Slab::Base)) => &self.base_targets[at..at + len],
            Some((at, len, Slab::Wide)) => &self.upper_targets[at..at + len],
            Some((at, len, Slab::Inline)) => &self.upper_word[at..at + len],
            None => &[],
        }
    }

    /// Edges the graph holds in its slabs, over every layer. The descent
    /// residue is not counted here; see [`Self::above_level_edges`].
    pub(super) fn nb_edges(&self) -> usize {
        let base: usize = self.base_len.iter().map(|&len| len as usize).sum();
        // The lists a node's own level reaches, and only those. What sits above
        // is `above_level_edges`. Walking the nodes rather than summing the
        // wide lengths is also what keeps an abandoned word run, which
        // `grow_span` leaves behind, out of the count.
        let upper: usize = (0..self.origin_ids.len() as u32)
            .map(|node| {
                (1..=self.levels[node as usize] as usize)
                    .map(|layer| self.list_len(node, layer))
                    .sum::<usize>()
            })
            .sum();
        base + upper
    }

    /// Edges sitting at layers above their owner's level, which the vendored
    /// descent files and the vendored reverse update adds to.
    pub(super) fn above_level_edges(&self) -> usize {
        (0..self.origin_ids.len() as u32)
            .map(|node| {
                let level = self.levels[node as usize] as usize;
                ((level + 1)..=self.span(node))
                    .map(|layer| self.list_len(node, layer))
                    .sum::<usize>()
            })
            .sum()
    }

    /// Upper list words the structure holds, which is what its memory figure
    /// counts. Includes any run `grow_span` abandoned.
    pub(super) fn nb_upper_lists(&self) -> usize {
        self.upper_word.len()
    }

    /// Wide upper lists the structure holds, each with `m + 1` slots.
    pub(super) fn nb_wide_lists(&self) -> usize {
        self.wide_at.len()
    }

    /// Slots the upper target arena holds, allocated rather than filled.
    pub(super) fn upper_slots(&self) -> usize {
        self.upper_targets.len()
    }

    /// Upper list words by what they hold, as (empty, one entry, wide), for
    /// the memory test. A word `grow_span` left behind is counted as well.
    #[cfg(test)]
    pub(super) fn word_census(&self) -> (usize, usize, usize) {
        let mut out = (0usize, 0usize, 0usize);
        for &word in &self.upper_word {
            if word == WORD_EMPTY {
                out.0 += 1;
            } else if word & WORD_WIDE != 0 {
                out.2 += 1;
            } else {
                out.1 += 1;
            }
        }
        out
    }

    /// Values per stored vector.
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// `max_nb_connection`.
    pub(super) fn m(&self) -> usize {
        self.m
    }

    /// Width the insertion traversal runs at.
    pub(super) fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// The level generator's scale.
    pub(super) fn level_scale(&self) -> f64 {
        self.level_scale
    }

    /// One node's top level.
    pub(super) fn level(&self, node: u32) -> u8 {
        self.levels[node as usize]
    }

    /// The id one node was inserted under, which is what two structures holding
    /// the same graph agree on where their node indices do not.
    pub(super) fn origin_id_of(&self, node: u32) -> usize {
        self.origin_ids[node as usize]
    }

    /// Where the traversal starts, as (layer, rank), which is what a dump
    /// records.
    pub(super) fn entry_point_id(&self) -> PointId {
        let order = DumpOrder::of(self);
        PointId(self.entry_level, order.rank[self.entry as usize] as i32)
    }

    /// Where every node sits, as a dump names it, in node order.
    ///
    /// The structure's own node order is the order nodes arrived, so this is
    /// the one place the dump's positional identity is readable from outside a
    /// save. It exists so a round trip can be compared on where each point
    /// landed and not only on what each point holds.
    pub(super) fn point_ids(&self) -> Vec<PointId> {
        let order = DumpOrder::of(self);
        (0..self.origin_ids.len() as u32)
            .map(|node| order.point_id(self, node))
            .collect()
    }

    /// The distance from one node to one target, which is what a stored
    /// distance was.
    ///
    /// The kernel is symmetric to the bit, so this is the value the traversal
    /// computed when it found `target` for the query that became `node`, and
    /// the value it computed when it found `node` for the query that became
    /// `target`, whichever way round the edge was filed.
    #[inline]
    fn edge_distance(&self, store: &VectorStore<T>, node: u32, target: u32) -> f32 {
        self.dist_f.eval(store.get(node), store.get(target))
    }

    /// Where one node's list sits, as (first slot, entries, which slab), or
    /// `None` where the node owns no list at that layer.
    #[inline]
    fn slice_of(&self, node: u32, layer: usize) -> Option<(usize, usize, Slab)> {
        if layer == 0 {
            let node = node as usize;
            return Some((
                node * self.base_cap(),
                self.base_len[node] as usize,
                Slab::Base,
            ));
        }
        let list = self.upper_list(node, layer)?;
        let word = self.upper_word[list];
        if word == WORD_EMPTY {
            return Some((list, 0, Slab::Inline));
        }
        if word & WORD_WIDE == 0 {
            return Some((list, 1, Slab::Inline));
        }
        let wide = (word & !WORD_WIDE) as usize;
        Some((
            self.wide_at[wide] as usize,
            self.wide_len[wide] as usize,
            Slab::Wide,
        ))
    }

    // ========================================================================
    // LIST MAINTENANCE
    //
    // What insertion does to a list, as operations on the layout rather than as
    // algorithm. Each of these is the slab form of one line of the vendored
    // insert, and each panics rather than growing where the vendored algorithm
    // cannot reach the state: a list at capacity is the transient the guarded
    // pop works in, and a push past it is a bug rather than a growth event.
    // ========================================================================

    /// Entries in one node's list at one layer, and zero where it owns none.
    #[inline]
    fn list_len(&self, node: u32, layer: usize) -> usize {
        self.slice_of(node, layer).map_or(0, |(_, len, _)| len)
    }

    /// Slots one node's list at one layer holds right now. A list held as one
    /// word holds one, and is promoted on the push that needs a second, so
    /// this is not a function of the layer alone.
    #[inline]
    fn list_cap(&self, node: u32, layer: usize) -> usize {
        match self.slice_of(node, layer) {
            Some((_, _, Slab::Base)) => self.base_cap(),
            Some((_, _, Slab::Wide)) => self.upper_cap_full(),
            Some((_, _, Slab::Inline)) => 1,
            None => 0,
        }
    }

    /// Open one node's upper lists, giving it a word at every layer from one up
    /// to `span`. A list at or below `level` is opened wide, because install
    /// site 2 fills it wholesale; one above `level` is an empty word, which is
    /// what 99.95 percent of them ever need.
    fn open_node(&mut self, span: usize, level: usize) {
        if span == 0 {
            self.upper_first.push(NO_UPPER);
            self.upper_span.push(0);
            return;
        }
        self.upper_first.push(self.upper_word.len() as u32);
        self.upper_span.push(span as u8);
        for layer in 1..=span {
            let word = if layer <= level {
                WORD_WIDE | self.open_wide()
            } else {
                WORD_EMPTY
            };
            self.upper_word.push(word);
        }
    }

    /// Append one wide descriptor with its `m + 1` slots, and hand back its
    /// index.
    fn open_wide(&mut self) -> u32 {
        let wide = u32::try_from(self.wide_at.len())
            .expect("a wide list index is a u32 and the arena is checked on every open");
        assert!(
            wide < WORD_WIDE,
            "the graph holds as many wide upper lists as a word names"
        );
        self.wide_at.push(self.upper_targets.len() as u32);
        self.wide_len.push(0);
        self.wide_in_degree.push(0);
        let cap = self.upper_cap_full();
        self.upper_targets.resize(self.upper_targets.len() + cap, 0);
        wide
    }

    /// Raise one node's span so that it owns a list at `layer`.
    ///
    /// A node's words are consecutive, so extending the run means moving it to
    /// the end of the word arena and leaving the old one behind. A wide list's
    /// slots do not move, because its word names them. This fires only where a
    /// node is named at a layer above its own level, which happens to the point
    /// that was the entry point when a higher one arrived, so it is on the order
    /// of the entry level per graph rather than per node.
    fn grow_span(&mut self, node: u32, layer: usize) {
        let old_span = self.span(node);
        if layer <= old_span {
            return;
        }
        assert!(
            layer < LAYERS,
            "a list was asked for at layer {} and a graph carries {} layers",
            layer,
            LAYERS
        );
        let level = self.levels[node as usize] as usize;
        let old_first = self.upper_first[node as usize] as usize;
        let new_first = self.upper_word.len() as u32;
        for slot in 0..old_span {
            let word = self.upper_word[old_first + slot];
            self.upper_word.push(word);
        }
        for new_layer in (old_span + 1)..=layer {
            let word = if new_layer <= level {
                WORD_WIDE | self.open_wide()
            } else {
                WORD_EMPTY
            };
            self.upper_word.push(word);
        }
        self.upper_first[node as usize] = new_first;
        self.upper_span[node as usize] = layer as u8;
    }

    /// Promote one list held as a word to a wide list, carrying its entry if it
    /// holds one, and hand back the wide descriptor's index.
    ///
    /// This is what a second entry needs and what an inbound counter needs,
    /// since both live on the wide descriptor. It fires for the old entry
    /// points' lists above their level and for nothing else on a build.
    fn promote(&mut self, list: usize) -> usize {
        let word = self.upper_word[list];
        assert!(
            word & WORD_WIDE == 0 || word == WORD_EMPTY,
            "a list is promoted once"
        );
        let wide = self.open_wide();
        if word != WORD_EMPTY {
            let at = self.wide_at[wide as usize] as usize;
            self.upper_targets[at] = word;
            self.wide_len[wide as usize] = 1;
        }
        self.upper_word[list] = WORD_WIDE | wide;
        wide as usize
    }

    /// One entry of one list.
    #[inline]
    fn target_at(&self, node: u32, layer: usize, slot: usize) -> u32 {
        let list = self.neighbours_at(node, layer);
        assert!(
            slot < list.len(),
            "slot {} of a list holding {}",
            slot,
            list.len()
        );
        list[slot]
    }

    /// Lists at `layer` naming `node`, which is what the guarded pop reads.
    ///
    /// A list held as one word has never been counted into, since the first
    /// count promotes it, so its owner's inbound count at that layer is zero.
    #[inline]
    fn in_degree(&self, node: u32, layer: usize) -> u32 {
        if layer == 0 {
            return self.base_in_degree[node as usize];
        }
        match self
            .upper_list(node, layer)
            .and_then(|list| self.wide_of(list))
        {
            Some(wide) => self.wide_in_degree[wide],
            None => 0,
        }
    }

    /// Move one inbound counter, which every edge install and every eviction
    /// does exactly once.
    ///
    /// A node can be named at a layer above its own level, so this opens a list
    /// there rather than assuming one, and promotes a list held as one word,
    /// since the counter lives on the wide descriptor. The vendored counterpart
    /// never has to, since it carries sixteen counters on every point whatever
    /// its level.
    #[inline]
    fn bump_in_degree(&mut self, node: u32, layer: usize, delta: i32) {
        let slot = if layer == 0 {
            &mut self.base_in_degree[node as usize]
        } else {
            self.grow_span(node, layer);
            let list = self
                .upper_list(node, layer)
                .expect("the span was just grown to cover this layer");
            let wide = match self.wide_of(list) {
                Some(wide) => wide,
                None => {
                    assert!(
                        delta > 0,
                        "an inbound counter is lowered only on a list that was raised"
                    );
                    self.promote(list)
                }
            };
            &mut self.wide_in_degree[wide]
        };
        *slot = slot
            .checked_add_signed(delta)
            .expect("an inbound counter is moved once per edge installed or evicted");
    }

    /// Whether one list already names a node, which the reverse update asks
    /// before it pushes.
    fn list_names(&self, node: u32, layer: usize, target: u32) -> bool {
        self.neighbours_at(node, layer).contains(&target)
    }

    /// Replace one list wholesale, which is what install site 2 does.
    ///
    /// Site 2 writes at or below the new node's level, where every list was
    /// opened wide, so a list held as one word is refused here rather than
    /// promoted; the loader promotes before it writes.
    fn write_list(&mut self, node: u32, layer: usize, targets: &[u32]) {
        let (at, _, slab) = match self.slice_of(node, layer) {
            Some(found) => found,
            None => panic!("node {} owns no list at layer {}", node, layer),
        };
        let cap = self.list_cap(node, layer);
        assert!(
            targets.len() <= cap,
            "a list at layer {} holds {} slots and {} entries were selected",
            layer,
            cap,
            targets.len()
        );
        match slab {
            Slab::Base => {
                self.base_targets[at..at + targets.len()].copy_from_slice(targets);
                self.base_len[node as usize] = targets.len() as u16;
            }
            Slab::Wide => {
                self.upper_targets[at..at + targets.len()].copy_from_slice(targets);
                let list = self
                    .upper_list(node, layer)
                    .expect("the slice was found through this list");
                let wide = self.wide_of(list).expect("the slab is wide");
                self.wide_len[wide] = targets.len() as u16;
            }
            Slab::Inline => panic!(
                "a list at layer {} is held as one word and is not written wholesale",
                layer
            ),
        }
    }

    /// Install one list the dump carried, in whichever form its layer and its
    /// owner's level give it.
    ///
    /// The caller has checked the length against the cap of the layer. A list
    /// above the owner's level holding one entry is the word itself, and one
    /// holding more is promoted first, which is the state a built graph holds
    /// an old entry point's list in.
    fn install_loaded_list(&mut self, node: u32, layer: usize, targets: &[u32]) {
        if layer > 0 {
            let list = self
                .upper_list(node, layer)
                .expect("the span was opened to cover every layer the point carries a list at");
            if self.wide_of(list).is_none() {
                if targets.len() == 1 {
                    self.upper_word[list] = targets[0];
                    return;
                }
                self.promote(list);
            }
        }
        self.write_list(node, layer, targets);
    }

    /// Append one edge to one list, which is what the reverse update and the
    /// descent do.
    ///
    /// The entry lands at the end of the list, which is where the vendored
    /// push put it; [`Self::place_last`] then moves it to where the order has
    /// it. A list held as one word takes its first entry in the word and is
    /// promoted for its second.
    fn push_edge(&mut self, node: u32, layer: usize, target: u32) {
        self.grow_span(node, layer);
        let (at, len, slab) = match self.slice_of(node, layer) {
            Some(found) => found,
            None => panic!("node {} owns no list at layer {}", node, layer),
        };
        match slab {
            Slab::Base => {
                assert!(
                    len < self.base_cap(),
                    "a layer zero list already holds its {} slots, which the vendored \
                     reverse update cannot reach because it shrinks whenever it exceeds \
                     the threshold",
                    self.base_cap()
                );
                self.base_targets[at + len] = target;
                self.base_len[node as usize] = (len + 1) as u16;
            }
            Slab::Inline => {
                if len == 0 {
                    self.upper_word[at] = target;
                } else {
                    let wide = self.promote(at);
                    let slot = self.wide_at[wide] as usize + 1;
                    self.upper_targets[slot] = target;
                    self.wide_len[wide] = 2;
                }
            }
            Slab::Wide => {
                let cap = self.upper_cap_full();
                assert!(
                    len < cap,
                    "a list at layer {} already holds its {} slots, which the vendored reverse \
                     update cannot reach because it shrinks whenever it exceeds the threshold",
                    layer,
                    cap
                );
                self.upper_targets[at + len] = target;
                let list = self
                    .upper_list(node, layer)
                    .expect("the slice was found through this list");
                let wide = self.wide_of(list).expect("the slab is wide");
                self.wide_len[wide] = (len + 1) as u16;
            }
        }
    }

    /// Move the entry a push just appended to where the list's order has it,
    /// which is what the vendored sort after every reverse push did.
    ///
    /// Every list is ordered by the distance from its owner to each target
    /// whenever no push is in flight: install site 2 writes a list phase one
    /// sorted, the descent writes one entry, the loader sorts by the file's
    /// distances, and an eviction shifts entries without reordering them. So
    /// the entries before the last are in order and the last is the one whose
    /// place is not known. A binary search over the ordered entries finds it,
    /// evaluating the distance to each entry it probes and to nothing else,
    /// and the rotation moves the last entry there.
    ///
    /// `dist` is the distance the traversal computed between the new point and
    /// this list's owner, which is the distance from the owner to the new
    /// point, since the kernel is symmetric. The entry goes after every entry
    /// at that distance or nearer, which is where a stable sort put it as the
    /// last element of its input, so two targets at an equal distance keep the
    /// order they were filed in. A NaN panics as the sort's comparator did.
    ///
    /// This runs once per reverse link per layer, which is the hottest list
    /// operation in the insert.
    fn place_last(&mut self, node: u32, layer: usize, store: &VectorStore<T>, dist: f32) {
        assert!(!dist.is_nan(), "got a NaN in a distance");
        let (at, len, slab) = match self.slice_of(node, layer) {
            Some(found) => found,
            None => panic!("node {} owns no list at layer {}", node, layer),
        };
        if len < 2 {
            return;
        }
        let owner = store.get(node);
        let dist_f = &self.dist_f;
        let ordered = &self.neighbours_at(node, layer)[..len - 1];
        let place = ordered.partition_point(|&target| {
            let d = dist_f.eval(owner, store.get(target));
            assert!(!d.is_nan(), "got a NaN in a distance");
            d <= dist
        });
        let targets = match slab {
            Slab::Base => &mut self.base_targets[at..at + len],
            Slab::Wide => &mut self.upper_targets[at..at + len],
            Slab::Inline => unreachable!("a list held as one word holds at most one entry"),
        };
        targets[place..].rotate_right(1);
    }

    /// Take one entry out of one list, shifting the rest down, and hand back
    /// the node it named. `Vec::remove` over the slab.
    ///
    /// Only a list that overflowed is evicted from, and a list held as one
    /// word cannot overflow, so the slab here is base or wide.
    fn remove_edge_at(&mut self, node: u32, layer: usize, slot: usize) -> u32 {
        let (at, len, slab) = match self.slice_of(node, layer) {
            Some(found) => found,
            None => panic!("node {} owns no list at layer {}", node, layer),
        };
        assert!(slot < len, "slot {} of a list holding {}", slot, len);
        match slab {
            Slab::Base => {
                let removed = self.base_targets[at + slot];
                self.base_targets
                    .copy_within(at + slot + 1..at + len, at + slot);
                self.base_len[node as usize] = (len - 1) as u16;
                removed
            }
            Slab::Wide => {
                let removed = self.upper_targets[at + slot];
                self.upper_targets
                    .copy_within(at + slot + 1..at + len, at + slot);
                let list = self
                    .upper_list(node, layer)
                    .expect("the slice was found through this list");
                let wide = self.wide_of(list).expect("the slab is wide");
                self.wide_len[wide] = (len - 1) as u16;
                removed
            }
            Slab::Inline => panic!(
                "a list at layer {} is held as one word, which cannot overflow and is \
                 not evicted from",
                layer
            ),
        }
    }

    /// Append one node with its vector, its level and the lists the descent
    /// left above that level, and hand back the index it took.
    ///
    /// This is the vendored `generate_new_point`: it files the point in its
    /// layer and gives it empty lists, and it runs before any edge is chosen.
    /// The regions that grow do so here and nowhere else on the ordinary path.
    /// The per node arrays and the layer zero slab take one entry and
    /// `2 * m + 1` slots. The node's upper lists are opened at the same moment,
    /// from layer one up to its span, so they stay consecutive and no later node
    /// disturbs them.
    ///
    /// The span is the higher of the level and the highest layer the descent
    /// filed an edge at, which is the entry level for almost every node. Later
    /// growth past it is possible and rare; see [`Self::grow_span`].
    fn append_node(
        &mut self,
        store: &mut VectorStore<T>,
        data: &[T],
        origin_id: usize,
        level: usize,
        above: &[(u8, u32)],
    ) -> u32 {
        assert_eq!(
            data.len(),
            self.dim,
            "a point of {} values was offered to a graph of {}",
            data.len(),
            self.dim
        );
        assert!(
            level < LAYERS,
            "a level of {} was drawn and a graph carries {} layers",
            level,
            LAYERS
        );
        let node = u32::try_from(self.origin_ids.len())
            .expect("a node index is a u32 and the arena is checked on every append");
        assert!(
            node < WORD_WIDE,
            "the graph holds as many nodes as an upper list word names"
        );

        debug_assert_eq!(
            store.len(),
            self.origin_ids.len(),
            "the store is addressed by node index, so it holds one vector per node"
        );
        self.origin_ids.push(origin_id);
        self.levels.push(level as u8);
        self.note_origin(origin_id, node);
        store.push(data);
        self.base_len.push(0);
        self.base_in_degree.push(0);
        self.base_targets
            .resize(self.base_targets.len() + self.base_cap(), 0);

        let mut span = level;
        for &(layer, _) in above {
            span = span.max(layer as usize);
        }
        self.open_node(span, level);

        for &(layer, target) in above {
            self.push_edge(node, layer as usize, target);
        }

        self.layer_counts[level] += 1;
        node
    }

    /// Move the traversal's starting point, which insertion does when it draws
    /// a level above every level drawn before it.
    fn set_entry(&mut self, node: u32, level: usize) {
        self.entry = node;
        self.entry_level = level as u8;
    }

    /// Record one overflow event and what the guard did with it.
    fn note_overflow(&mut self, victim: Option<usize>, farthest: usize) {
        self.overflows += 1;
        match victim {
            Some(slot) if slot != farthest => self.saves += 1,
            Some(_) => {}
            None => self.fallbacks += 1,
        }
    }

    /// The overflow pop counters, as (overflows, saves, fallbacks). The same
    /// three the vendored crate reported, so a fixture can be checked for
    /// actually reaching the guard rather than assumed to.
    pub(super) fn guard_stats(&self) -> (u64, u64, u64) {
        (self.overflows, self.saves, self.fallbacks)
    }

    /// One node's adjacency by layer, as the vendored `get_neighborhood_id`
    /// reports the same point's: every layer the structure carries, each entry
    /// naming the id its target was inserted under and the distance from the
    /// node to it, recomputed. The descent residue sits at the layer it was
    /// filed at, which is where the vendored point carries it too.
    ///
    /// This is the shape two builds are compared in. It resolves targets to
    /// origin ids rather than to node indices, because two structures may
    /// number their nodes differently and the id is what both agree on.
    pub(super) fn neighbourhood_ids(
        &self,
        store: &VectorStore<T>,
        node: u32,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut out = vec![Vec::new(); LAYERS];
        for (layer, list) in out.iter_mut().enumerate().take(self.span(node) + 1) {
            for &target in self.neighbours_at(node, layer) {
                list.push((
                    self.origin_ids[target as usize],
                    self.edge_distance(store, node, target),
                ));
            }
        }
        out
    }

    /// Lists at `layer` naming `node`, counted from the adjacency rather than
    /// read from the counters, so a test can hold the graph itself to the
    /// property and stay meaningful if the counters are ever wrong.
    pub(super) fn counted_in_degree(&self, layer: usize) -> Vec<u32> {
        let mut counts = vec![0u32; self.origin_ids.len()];
        for node in 0..self.origin_ids.len() as u32 {
            for &target in self.neighbours_at(node, layer) {
                counts[target as usize] += 1;
            }
        }
        counts
    }

    /// Return every buffer's spare capacity to the allocator.
    ///
    /// A graph built by insertion grows its arenas geometrically from whatever
    /// [`MutableGraph::new`] reserved, so the last growth leaves the largest of
    /// them holding close to twice what it uses. A graph produced by
    /// [`MutableGraph::from_loaded`] has no such slack, because the node count
    /// is known before the first write, which is why the same index measures
    /// smaller after a save and load round trip than it did when it was built.
    ///
    /// Thirteen reallocations, one per buffer, each copying the live bytes. No
    /// node is touched, no edge is read and no distance is evaluated, so the
    /// topology after this call is the topology before it and every search
    /// returns exactly what it returned.
    ///
    /// **The graph stays mutable.** Every buffer is a `Vec` and a push past a
    /// full `Vec` grows it, so the next insertion after this reallocates the
    /// per node arenas once and then proceeds as before. Shrinking a graph that
    /// is still being built therefore trades that one regrowth for the memory,
    /// which is the caller's decision and is why nothing here is automatic.
    ///
    /// Returns the bytes released, being the drop in
    /// [`MutableGraph::memory_bytes`].
    pub(super) fn shrink_to_fit(&mut self) -> usize {
        let before = self.memory_bytes();
        self.origin_ids.shrink_to_fit();
        self.levels.shrink_to_fit();
        self.node_of.shrink_to_fit();
        self.base_targets.shrink_to_fit();
        self.base_len.shrink_to_fit();
        self.base_in_degree.shrink_to_fit();
        self.upper_first.shrink_to_fit();
        self.upper_span.shrink_to_fit();
        self.upper_word.shrink_to_fit();
        self.wide_at.shrink_to_fit();
        self.wide_len.shrink_to_fit();
        self.wide_in_degree.shrink_to_fit();
        self.upper_targets.shrink_to_fit();
        before.saturating_sub(self.memory_bytes())
    }

    /// Bytes the structure has asked the allocator for.
    ///
    /// Exact rather than sampled: every buffer's capacity is known and there
    /// is no per-node or per-edge allocation to estimate. What the figure does
    /// not see is how much of that capacity has been written, which is
    /// [`MutableGraph::reserved_bytes`].
    pub(super) fn memory_bytes(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.origin_ids.capacity() * std::mem::size_of::<usize>();
        total += self.levels.capacity();
        total += self.node_of.capacity() * std::mem::size_of::<u32>();
        total += self.base_targets.capacity() * std::mem::size_of::<u32>();
        total += self.base_len.capacity() * std::mem::size_of::<u16>();
        total += self.base_in_degree.capacity() * std::mem::size_of::<u32>();
        total += self.upper_first.capacity() * std::mem::size_of::<u32>();
        total += self.upper_span.capacity();
        total += self.upper_word.capacity() * std::mem::size_of::<u32>();
        total += self.wide_at.capacity() * std::mem::size_of::<u32>();
        total += self.wide_len.capacity() * std::mem::size_of::<u16>();
        total += self.wide_in_degree.capacity() * std::mem::size_of::<u32>();
        total += self.upper_targets.capacity() * std::mem::size_of::<u32>();
        total
    }

    /// Bytes of that request no node has been written into.
    ///
    /// The same thirteen buffers priced at the gap between their capacity and
    /// their length, plus nothing for the struct itself, which is written in
    /// full the moment it exists. A graph built by insertion grows its arenas
    /// geometrically past whatever the creation-time reservation held, so this
    /// is the slack of the last growth of each of them, and it is exactly what
    /// [`MutableGraph::shrink_to_fit`] returns to the allocator.
    ///
    /// It is committed rather than resident. A page a `Vec` reserved and never
    /// wrote is charged against the pagefile and is not in the working set,
    /// which is why the figure is reported beside the request rather than
    /// subtracted from it.
    pub(super) fn reserved_bytes(&self) -> usize {
        fn spare<T>(v: &[T], capacity: usize) -> usize {
            capacity.saturating_sub(v.len()) * std::mem::size_of::<T>()
        }
        let mut total = 0;
        total += spare(&self.origin_ids, self.origin_ids.capacity());
        total += spare(&self.levels, self.levels.capacity());
        total += spare(&self.node_of, self.node_of.capacity());
        total += spare(&self.base_targets, self.base_targets.capacity());
        total += spare(&self.base_len, self.base_len.capacity());
        total += spare(&self.base_in_degree, self.base_in_degree.capacity());
        total += spare(&self.upper_first, self.upper_first.capacity());
        total += spare(&self.upper_span, self.upper_span.capacity());
        total += spare(&self.upper_word, self.upper_word.capacity());
        total += spare(&self.wide_at, self.wide_at.capacity());
        total += spare(&self.wide_len, self.wide_len.capacity());
        total += spare(&self.wide_in_degree, self.wide_in_degree.capacity());
        total += spare(&self.upper_targets, self.upper_targets.capacity());
        total
    }

    /// Search the graph, which is [`traverse::search`] over this layout.
    ///
    /// The empty case is answered here rather than in the traversal, which has
    /// no early return for it because neither loading constructor accepts a
    /// graph of no points. [`Self::new`] does, since a graph being built by
    /// insertion starts empty, and the vendored `search_filter` answers the same
    /// question the same way.
    pub(super) fn search<F>(
        &self,
        store: &VectorStore<T>,
        data: &[T],
        knbn: usize,
        ef_arg: usize,
        filter: Option<&F>,
    ) -> Vec<GraphHit>
    where
        F: Fn(&usize) -> bool,
    {
        if self.origin_ids.is_empty() {
            return Vec::new();
        }
        traverse::search(&self.bound(store), data, knbn, ef_arg, filter)
    }

    /// This graph with its store, which is what a traversal reads.
    #[inline]
    pub(in crate::graph) fn bound<'a>(&'a self, store: &'a VectorStore<T>) -> Bound<'a, T, D> {
        Bound { graph: self, store }
    }

    /// One node's adjacency as a dump records it, being every list from layer
    /// zero up to the highest layer the node carries anything at, each edge
    /// carrying the distance from the node to its target, recomputed.
    ///
    /// Empty lists in the middle are kept, because a node can carry residue at
    /// a layer above one it carries nothing at, and the file records lists by
    /// position. Trailing empty lists are the writer's business to trim.
    fn neighbourhood_into(
        &self,
        node: u32,
        order: &DumpOrder,
        store: &VectorStore<T>,
        out: &mut Vec<Vec<LoadedEdge>>,
    ) {
        for list in out.iter_mut() {
            list.clear();
        }
        while out.len() < LAYERS {
            out.push(Vec::new());
        }
        let lists = self.span(node) + 1;
        for (layer, list) in out.iter_mut().enumerate().take(lists) {
            let targets = self.neighbours_at(node, layer);
            list.reserve(targets.len());
            for &target in targets {
                list.push(LoadedEdge {
                    target: order.point_id(self, target),
                    distance: self.edge_distance(store, node, target),
                });
            }
        }
    }
}

/// The graph together with the store it is addressed against.
///
/// The graph holds links, levels and origin ids and no vectors at all, so a
/// traversal needs both. This pairs them for the length of one operation, which
/// is the whole of what design B costs at a distance evaluation: one load of
/// the store's base pointer, which the optimiser hoists out of the neighbour
/// loop, and then the same multiply and slice the arena used.
pub(in crate::graph) struct Bound<'a, T, D> {
    pub(in crate::graph) graph: &'a MutableGraph<T, D>,
    pub(in crate::graph) store: &'a VectorStore<T>,
}

impl<T, D> Topology for Bound<'_, T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    type Elem = T;
    type Dist = D;

    fn distance(&self) -> &D {
        self.graph.distance()
    }

    fn nb_points(&self) -> usize {
        self.graph.nb_points()
    }

    fn entry(&self) -> u32 {
        self.graph.entry()
    }

    fn entry_level(&self) -> u8 {
        self.graph.entry_level()
    }

    #[inline]
    fn layer_len(&self, layer: usize) -> usize {
        self.graph.layer_len(layer)
    }

    #[inline]
    fn vector(&self, node: u32) -> &[T] {
        self.store.get(node)
    }

    #[inline]
    fn origin_id(&self, node: u32) -> usize {
        self.graph.origin_ids[node as usize]
    }

    /// The neighbour list of one node at one layer.
    ///
    /// The whole list, at every layer the node owns one at, which is what the
    /// vendored `Point::neighbours[layer]` returns. That includes the layers
    /// above the node's own level.
    ///
    /// An earlier layout dropped those edges from the read-only form and held
    /// them out of this accessor, on the reasoning that no traversal reaches a
    /// node at a layer above its level. That reasoning has a hole, which
    /// porting the insert found: `search_layer` seeds its result heap with
    /// the point it was entered at whatever that point's level, so a point below
    /// the layer can enter a list there and be reached from it afterwards. The
    /// insertion traversal reads these lists on the vendored path, so it has to
    /// read them here.
    #[inline]
    fn neighbours(&self, node: u32, layer: usize) -> &[u32] {
        self.graph.neighbours_at(node, layer)
    }
}

/// The dump's (layer, rank) identity for every node, recovered in one pass.
///
/// The structure's own node order is the order nodes arrived, which after any
/// insertion is not layer major. A dump is written layer major, so a save needs
/// each node's rank within its own layer and each layer's members in rank
/// order. Both come from one counting sort over the levels, which is stable, so
/// a graph loaded from a dump and written straight back out reproduces the
/// order it arrived in.
pub(super) struct DumpOrder {
    /// Nodes grouped by level, in node order within a level.
    by_layer: Vec<u32>,
    /// Where each layer's run starts in [`Self::by_layer`], with a sentinel.
    offsets: [u32; LAYERS + 1],
    /// Each node's rank within its own layer.
    rank: Vec<u32>,
}

impl DumpOrder {
    pub(super) fn of<T, D>(graph: &MutableGraph<T, D>) -> Self
    where
        T: Clone + Send + Sync,
        D: Distance<T> + Send + Sync,
    {
        let mut offsets = [0u32; LAYERS + 1];
        for layer in 0..LAYERS {
            offsets[layer + 1] = offsets[layer] + graph.layer_counts[layer];
        }
        let mut cursor = offsets;
        let mut by_layer = vec![0u32; graph.origin_ids.len()];
        let mut rank = vec![0u32; graph.origin_ids.len()];
        for (node, &level) in graph.levels.iter().enumerate() {
            let level = level as usize;
            let at = cursor[level];
            by_layer[at as usize] = node as u32;
            rank[node] = at - offsets[level];
            cursor[level] = at + 1;
        }
        DumpOrder {
            by_layer,
            offsets,
            rank,
        }
    }

    /// Where one node sits, as the dump names it.
    fn point_id<T, D>(&self, graph: &MutableGraph<T, D>, node: u32) -> PointId
    where
        T: Clone + Send + Sync,
        D: Distance<T> + Send + Sync,
    {
        PointId(graph.levels[node as usize], self.rank[node as usize] as i32)
    }

    /// The nodes of one layer, in rank order.
    fn layer(&self, layer: usize) -> &[u32] {
        &self.by_layer[self.offsets[layer] as usize..self.offsets[layer + 1] as usize]
    }
}

/// The mutable graph seen as something a dump can be written from.
///
/// Holding the order alongside the graph means the three passes the writer
/// makes share one counting sort rather than repeating it.
pub(super) struct DumpView<'a, T, D> {
    graph: &'a MutableGraph<T, D>,
    store: &'a VectorStore<T>,
    order: DumpOrder,
}

impl<T, D> MutableGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// This graph as a dump source.
    pub(super) fn dump_view<'a>(&'a self, store: &'a VectorStore<T>) -> DumpView<'a, T, D> {
        DumpView {
            order: DumpOrder::of(self),
            graph: self,
            store,
        }
    }
}

impl<T, D> super::dump::DumpSource<T> for DumpView<'_, T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    fn nb_point(&self) -> usize {
        self.graph.origin_ids.len()
    }

    fn entry(&self) -> Option<PointId> {
        if self.graph.origin_ids.is_empty() {
            return None;
        }
        Some(PointId(
            self.graph.entry_level,
            self.order.rank[self.graph.entry as usize] as i32,
        ))
    }

    fn layer_nb_point(&self, layer: usize) -> usize {
        self.graph.layer_counts[layer] as usize
    }

    fn dimension(&self) -> usize {
        self.graph.dim
    }

    fn max_nb_connection(&self) -> usize {
        self.graph.m
    }

    fn ef_construction(&self) -> usize {
        self.graph.ef_construction
    }

    fn level_scale(&self) -> f64 {
        self.graph.level_scale
    }

    fn each_origin_id(
        &self,
        layer: usize,
        f: &mut dyn FnMut(usize) -> Result<(), String>,
    ) -> Result<(), String> {
        for &node in self.order.layer(layer) {
            f(self.graph.origin_ids[node as usize])?;
        }
        Ok(())
    }

    fn each_neighbourhood(&self, layer: usize, f: EachNeighbourhood<'_>) -> Result<(), String> {
        let mut scratch: Vec<Vec<LoadedEdge>> = Vec::new();
        for &node in self.order.layer(layer) {
            self.graph
                .neighbourhood_into(node, &self.order, self.store, &mut scratch);
            f(&scratch)?;
        }
        Ok(())
    }

    fn each_vector(
        &self,
        layer: usize,
        f: &mut dyn FnMut(&[T]) -> Result<(), String>,
    ) -> Result<(), String> {
        for &node in self.order.layer(layer) {
            f(self.store.get(node))?;
        }
        Ok(())
    }
}

/// Check one edge against the layer table and hand back its node index.
///
/// The same three rules `Hnsw::from_loaded_points` applies to every edge: a
/// finite distance, a target layer the graph carries, and a target rank the
/// layer holds. Applied to every edge whatever layer its list sits at.
fn edge_target(edge: &LoadedEdge, layer_offsets: &[u32; LAYERS + 1]) -> Result<u32, String> {
    if !edge.distance.is_finite() {
        return Err(format!("an edge carries a distance of {}", edge.distance));
    }
    let target_layer = edge.target.0 as usize;
    if target_layer >= LAYERS {
        return Err(format!(
            "an edge names a target at layer {} and the graph carries {}",
            target_layer, LAYERS
        ));
    }
    let span = layer_offsets[target_layer + 1] - layer_offsets[target_layer];
    if edge.target.1 < 0 || edge.target.1 as u32 >= span {
        return Err(format!(
            "an edge names rank {} of layer {} and that layer holds {} points",
            edge.target.1, target_layer, span
        ));
    }
    Ok(layer_offsets[target_layer] + edge.target.1 as u32)
}
