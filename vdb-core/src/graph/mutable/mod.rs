//! The mutable form of the graph, which is what construction writes into.
//!
//! [`super::flat::FlatGraph`] is read-only by construction. Its node order is
//! layer major, so appending a node of level `l` means inserting into the
//! middle of the arena and renumbering every higher-layer node, which
//! invalidates every CSR target. This is the same graph in a form a node can be
//! appended to.
//!
//! # Node identity
//!
//! A node is a `u32` handed out in the order nodes arrive, and it never
//! changes. Loading adopts the dump's own order, which is layer major, so a
//! loaded graph numbers its nodes exactly as `FlatGraph` does and the two can
//! be compared node for node. Insertion appends at the end whatever level it
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
//! Fixed-capacity slabs, targets and distances in separate arrays.
//!
//! Layer zero needs no indirection, because every node owns a layer zero slab:
//! node `n` holds slots `[n * base_cap, n * base_cap + base_cap)`. Only the
//! upper layers are sparse, holding `1 / (m - 1)` lists per node at the default
//! scale, so a node carries one `u32` naming its first upper list and the lists
//! of a node of level `L` are the `L` consecutive ones from there.
//!
//! Targets and distances are separate arrays because the traversal reads only
//! targets. A layer zero list at `m` 16 touches 132 bytes of the target arena
//! rather than the 264 an interleaved layout would. Construction reads both,
//! and construction is not latency critical.
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
//! its level.** This is the correction relay 79 made to the layout, and it is
//! what the vendored structure has always done: `Point::new` gives every point
//! sixteen layer vectors whatever level it was drawn at, and the builder writes
//! into the ones above the level from two directions.
//!
//! The descent writes there. Install site 1 records the entry point it passed at
//! each layer above the new point's own level into the new point's list there,
//! so a level zero point carries entries at layer five. Relay 77 called this the
//! descent residue and dropped it; relay 78 kept it in an append-only region.
//!
//! The reverse update writes there too, and that is what the append-only region
//! could not hold. `search_layer` seeds its result heap with the point it is
//! entered at whatever that point's level, so a search at layer `l` can return a
//! point of level below `l`. Every time a new point draws a level above every
//! level drawn before it, its own search at that layer is entered at the old
//! entry point and returns it, so the old entry point takes a reverse link at a
//! layer above its own level. From there it is reachable at that layer, so the
//! link is not a one-off: measured on a 2,000 point cosine graph at `m` 16,
//! 4,289 above-level lists exist, 4,287 of them holding one entry and two
//! holding ten and sixteen. **An above-level list grows, is sorted, is evicted
//! from and is counted into, exactly as a list at or below the level is.**
//!
//! So the layout carries them as ordinary lists, at two initial capacities. A
//! list at or below its owner's level starts at `m + 1` because install site 2
//! fills it wholesale. A list above the level starts at one slot, which is what
//! 99.95 percent of them ever hold, and is reallocated to `m + 1` the first time
//! a reverse push overflows it. That is why an upper list carries its own offset
//! rather than being addressed by multiplication.
//!
//! Keeping these lists is not a choice the way relay 78 framed it. The vendored
//! descent increments the pivot's inbound counter at the layer it files the edge
//! at, and the overflow pop guard reads that counter, so a build that skips them
//! evicts different entries. And the traversal reads them, so a build that
//! cannot see them chooses different neighbours. See [`insert`].
//!
//! # The inbound counters
//!
//! One `u32` per node at layer zero and one per upper list, which is what the
//! guarded overflow pop reads. Every counter the vendored crate touches is at a
//! layer the target owns, so the counters need no table of their own: they sit
//! in step with the lists they count into.
//!
//! The vendored equivalent is a `[AtomicU32; 16]` on every point whatever its
//! level, which is 64 bytes per node against 4.27 here at `m` 16, and which
//! carries a recorded read-then-act race on the parallel insertion path. These
//! are plain integers because the mutator is serialised.

use super::dump::EachNeighbourhood;
use super::traverse::{self, Topology, LAYERS};
use super::{Distance, GraphHit};
use hnsw_rs::hnsw::{LoadedEdge, LoadedPoint, PointId};

mod insert;

/// Naming no upper list, which is every node whose span is zero.
const NO_UPPER: u32 = u32::MAX;

/// Slots a list above its owner's level opens with.
///
/// Measured on a 2,000 point cosine graph at `m` 16, 4,287 of 4,289 such lists
/// hold exactly one entry for the life of the graph. The two that grow are
/// widened to `m + 1` on the push that overflows them.
const ABOVE_LEVEL_CAP: usize = 1;

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

/// One stored edge, as a list holds it and as the two sorts order it.
///
/// **This type is deliberately not `Copy` and not `Clone`.** `sort_unstable`
/// dispatches on the element type, and the permutation it produces over equal
/// keys differs between the two dispatch paths. The vendored insert sorts
/// `Vec<Arc<PointWithOrder>>`, which is 8 bytes and not `Copy`; a `Copy` pair of
/// the same width takes the other path and orders ties differently from length
/// 21 up. Layer zero lists run to `2 * m + 1`, which is 33 at the shipped `m`,
/// and quantized codes tie constantly, so the difference is reachable rather
/// than theoretical. `the_entry_sort_matches_the_vendored_tie_order` pins it.
pub(super) struct Entry {
    /// Distance from the list's owner to the target, as it is stored.
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
    entries.sort_unstable_by(by_distance);
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
    /// Every vector, node `n` at `[n * dim, (n + 1) * dim)`.
    data: Vec<T>,

    /// Layer zero targets, node `n` at `[n * base_cap, ..)`.
    base_targets: Vec<u32>,
    /// Layer zero distances, in step with [`Self::base_targets`].
    base_dists: Vec<f32>,
    /// Entries in each node's layer zero list.
    base_len: Vec<u16>,
    /// Lists at layer zero naming this node, which the overflow pop guard
    /// reads before it evicts.
    base_in_degree: Vec<u32>,

    /// The first upper list a node owns, or [`NO_UPPER`] where it owns none.
    /// A node's lists for layers `1..=span` are the consecutive descriptors
    /// from there.
    upper_first: Vec<u32>,
    /// The highest layer each node owns a list at, which is at least its level
    /// and can exceed it. Zero means it owns no upper list.
    upper_span: Vec<u8>,

    /// Where each upper list's slots start in [`Self::upper_targets`]. Held per
    /// list rather than derived, because the two initial capacities and the
    /// reallocation that joins them make the offsets irregular.
    upper_at: Vec<u32>,
    /// Entries in each upper list.
    upper_len: Vec<u16>,
    /// Slots each upper list holds, being `m + 1` at or below its owner's level
    /// and one above it until a push reallocates it.
    upper_cap: Vec<u16>,
    /// Lists naming this list's owner at this list's layer, in step with
    /// [`Self::upper_len`].
    upper_in_degree: Vec<u32>,

    /// Upper layer targets, one list's slots at `[upper_at[i], ..)`.
    upper_targets: Vec<u32>,
    /// Upper layer distances, in step with [`Self::upper_targets`].
    upper_dists: Vec<f32>,

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
}

impl<T, D> MutableGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// Build the structure from the topology the dump reader produces.
    ///
    /// The same acceptance rules as `Hnsw::from_loaded_points` and
    /// `FlatGraph::from_loaded`, because all three are fed the same parsed dump
    /// and have to agree on what a malformed one is, plus one rule the others
    /// do not need. A neighbour list longer than the cap its layer allows is
    /// refused, since the vendored builder cannot produce one and this layout
    /// caps a list at what the guarded pop needs. A refusal reaches the loader's
    /// rebuild path, which is what every other refusal reaches too.
    ///
    /// Nodes are numbered in the order the dump streams them, which is layer
    /// major, so a loaded graph's node indices match `FlatGraph`'s exactly.
    pub(super) fn from_loaded(
        points_by_layer: Vec<Vec<LoadedPoint<T>>>,
        entry_point: PointId,
        m: usize,
        ef_construction: usize,
        level_scale: f64,
        dist_f: D,
    ) -> Result<Self, String> {
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
            data: Vec::with_capacity(nb_point * dim),
            base_targets: vec![0u32; nb_point * base_cap],
            base_dists: vec![0f32; nb_point * base_cap],
            base_len: vec![0u16; nb_point],
            base_in_degree: vec![0u32; nb_point],
            upper_first: Vec::with_capacity(nb_point),
            upper_span: Vec::with_capacity(nb_point),
            upper_at: Vec::new(),
            upper_len: Vec::new(),
            upper_cap: Vec::new(),
            upper_in_degree: Vec::new(),
            upper_targets: Vec::new(),
            upper_dists: Vec::new(),
            overflows: 0,
            saves: 0,
            fallbacks: 0,
            dist_f,
        };

        let mut lists: Vec<Vec<(f32, u32)>> = Vec::new();
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
                graph.data.extend(point.data);

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
                    // A list above its owner's level opens at a single slot,
                    // which is what almost all of them ever hold, so a loaded
                    // one carrying more is widened rather than refused. Refusal
                    // is for a list past the cap its layer allows at all.
                    if list_layer > 0 && list.len() > graph.list_cap(node, list_layer) {
                        graph.widen_list(node, list_layer);
                    }
                    let cap = graph.list_cap(node, list_layer);
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
                    let (at, upper) = graph
                        .slice_of(node, list_layer)
                        .map(|(at, _, upper)| (at, upper))
                        .expect("the span covers every layer the point carries a list at");
                    let (targets, dists) = if upper {
                        (&mut graph.upper_targets, &mut graph.upper_dists)
                    } else {
                        (&mut graph.base_targets, &mut graph.base_dists)
                    };
                    for (slot, &(distance, target)) in list.iter().enumerate() {
                        targets[at + slot] = target;
                        dists[at + slot] = distance;
                    }
                    graph.set_list_len(node, list_layer, list.len());
                }
            }
        }
        // Trimming makes every buffer's capacity its length, which is what lets
        // `memory_bytes` be checked against the counts rather than believed. The
        // upper regions are the ones whose size is not known before the walk,
        // because a span is a property of the file.
        graph.upper_at.shrink_to_fit();
        graph.upper_len.shrink_to_fit();
        graph.upper_cap.shrink_to_fit();
        graph.upper_in_degree.shrink_to_fit();
        graph.upper_targets.shrink_to_fit();
        graph.upper_dists.shrink_to_fit();

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

        Ok(graph)
    }

    /// An empty graph, ready to be built by insertion.
    ///
    /// `expected_size` sizes the reservation and nothing else. It is the same
    /// hint `Hnsw::new` takes and it bounds no behaviour, so a graph that
    /// outgrows it grows geometrically from there. Overshooting it costs the
    /// per node arithmetic in `memory_bytes` per unused record, of which the
    /// layer zero slab at `(2m + 1) * 8` is nearly all; the unpatched vendored
    /// reservation this replaces cost 3,025 bytes per unused record.
    ///
    /// The upper arena is reserved at `1 / (m - 1)` lists per node, which is
    /// the expected level count under the default scale rather than a bound, so
    /// it is the one region a build routinely grows past.
    pub(super) fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        level_scale: f64,
        expected_size: usize,
        dist_f: D,
    ) -> Result<Self, String> {
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
        // The expected span, which is what sizes the upper arena. Almost every
        // node ends up owning a list at every layer from one up to the entry
        // level, because the descent files one there, so the arena is sized by
        // the entry level a graph of `expected_size` points reaches rather than
        // by the far smaller expected level.
        let upper_lists = expected_size * expected_span(m, expected_size);
        Ok(MutableGraph {
            dim,
            m,
            ef_construction,
            level_scale,
            entry: 0,
            entry_level: 0,
            layer_counts: [0u32; LAYERS],
            origin_ids: Vec::with_capacity(expected_size),
            levels: Vec::with_capacity(expected_size),
            data: Vec::with_capacity(expected_size * dim),
            base_targets: Vec::with_capacity(expected_size * base_cap),
            base_dists: Vec::with_capacity(expected_size * base_cap),
            base_len: Vec::with_capacity(expected_size),
            base_in_degree: Vec::with_capacity(expected_size),
            upper_first: Vec::with_capacity(expected_size),
            upper_span: Vec::with_capacity(expected_size),
            upper_at: Vec::with_capacity(upper_lists),
            upper_len: Vec::with_capacity(upper_lists),
            upper_cap: Vec::with_capacity(upper_lists),
            upper_in_degree: Vec::with_capacity(upper_lists),
            upper_targets: Vec::with_capacity(upper_lists),
            upper_dists: Vec::with_capacity(upper_lists),
            overflows: 0,
            saves: 0,
            fallbacks: 0,
            dist_f,
        })
    }

    /// Slots one layer zero list holds, being the vendored threshold plus the
    /// overflow slot the guarded pop needs.
    #[inline]
    fn base_cap(&self) -> usize {
        2 * self.m + 1
    }

    /// Slots one upper list holds once it has been widened, on the same rule.
    #[inline]
    fn upper_cap_full(&self) -> usize {
        self.m + 1
    }

    /// The highest layer one node owns a list at, which is at least its level.
    #[inline]
    fn span(&self, node: u32) -> usize {
        self.upper_span[node as usize] as usize
    }

    /// Which descriptor holds one node's list at one layer, and `None` where the
    /// node owns none there.
    #[inline]
    fn upper_list(&self, node: u32, layer: usize) -> Option<usize> {
        if layer == 0 || layer > self.span(node) {
            return None;
        }
        Some(self.upper_first[node as usize] as usize + layer - 1)
    }

    /// Nodes the graph holds.
    pub(super) fn nb_points(&self) -> usize {
        self.origin_ids.len()
    }

    /// Edges the graph holds in its slabs, over every layer. The descent
    /// residue is not counted here; see [`Self::above_level_edges`].
    pub(super) fn nb_edges(&self) -> usize {
        let base: usize = self.base_len.iter().map(|&len| len as usize).sum();
        // The lists a node's own level reaches, and only those. What sits above
        // is `above_level_edges`. Walking the nodes rather than summing
        // `upper_len` is also what keeps an abandoned descriptor run, which
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

    /// Upper list descriptors the structure holds, which is what its memory
    /// figure counts. Includes any run `grow_span` abandoned.
    pub(super) fn nb_upper_lists(&self) -> usize {
        self.upper_at.len()
    }

    /// Slots the upper arena holds, allocated rather than filled.
    pub(super) fn upper_slots(&self) -> usize {
        self.upper_targets.len()
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

    /// Where the traversal starts, as (layer, rank), which is what a dump
    /// records.
    pub(super) fn entry_point_id(&self) -> PointId {
        let order = DumpOrder::of(self);
        PointId(self.entry_level, order.rank[self.entry as usize] as i32)
    }

    /// The stored vector of one node.
    #[inline]
    pub(super) fn vector(&self, node: u32) -> &[T] {
        let at = node as usize * self.dim;
        &self.data[at..at + self.dim]
    }

    /// The stored distances of one node's list, in step with its targets.
    ///
    /// The traversal never reads these; list maintenance does, and so does the
    /// dump writer.
    #[inline]
    pub(super) fn distances(&self, node: u32, layer: usize) -> &[f32] {
        match self.slice_of(node, layer) {
            Some((at, len, false)) => &self.base_dists[at..at + len],
            Some((at, len, true)) => &self.upper_dists[at..at + len],
            None => &[],
        }
    }

    /// Where one node's list sits, as (first slot, entries, is upper), or
    /// `None` where the node owns no list at that layer.
    #[inline]
    fn slice_of(&self, node: u32, layer: usize) -> Option<(usize, usize, bool)> {
        if layer == 0 {
            let node = node as usize;
            return Some((node * self.base_cap(), self.base_len[node] as usize, false));
        }
        let list = self.upper_list(node, layer)?;
        Some((
            self.upper_at[list] as usize,
            self.upper_len[list] as usize,
            true,
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

    /// Slots one node's list at one layer holds right now. An upper list above
    /// its owner's level starts at one and is widened on the first push that
    /// overflows it, so this is not a function of the layer alone.
    #[inline]
    fn list_cap(&self, node: u32, layer: usize) -> usize {
        if layer == 0 {
            return self.base_cap();
        }
        match self.upper_list(node, layer) {
            Some(list) => self.upper_cap[list] as usize,
            None => 0,
        }
    }

    /// Open one node's upper arena, giving it a list at every layer from one up
    /// to `span`. A list at or below `level` starts at the full `m + 1` slots
    /// because install site 2 fills it wholesale; one above `level` starts at a
    /// single slot, which is what 99.95 percent of them ever hold.
    fn open_node(&mut self, span: usize, level: usize) {
        if span == 0 {
            self.upper_first.push(NO_UPPER);
            self.upper_span.push(0);
            return;
        }
        self.upper_first.push(self.upper_at.len() as u32);
        self.upper_span.push(span as u8);
        for layer in 1..=span {
            let cap = if layer <= level {
                self.upper_cap_full()
            } else {
                ABOVE_LEVEL_CAP
            };
            self.open_list(cap);
        }
    }

    /// Append one upper list descriptor with its slots.
    fn open_list(&mut self, cap: usize) {
        self.upper_at.push(self.upper_targets.len() as u32);
        self.upper_len.push(0);
        self.upper_cap.push(cap as u16);
        self.upper_in_degree.push(0);
        self.upper_targets.resize(self.upper_targets.len() + cap, 0);
        self.upper_dists.resize(self.upper_dists.len() + cap, 0.);
    }

    /// Raise one node's span so that it owns a list at `layer`.
    ///
    /// A node's descriptors are consecutive, so extending the run means moving
    /// it to the end of the descriptor arena and leaving the old one behind. The
    /// slots do not move, because a descriptor names them. This fires only where
    /// a node is named at a layer above its own level, which happens to the
    /// point that was the entry point when a higher one arrived, so it is on the
    /// order of the entry level per graph rather than per node.
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
        let new_first = self.upper_at.len() as u32;
        for slot in 0..old_span {
            let (at, len, cap, in_degree) = (
                self.upper_at[old_first + slot],
                self.upper_len[old_first + slot],
                self.upper_cap[old_first + slot],
                self.upper_in_degree[old_first + slot],
            );
            self.upper_at.push(at);
            self.upper_len.push(len);
            self.upper_cap.push(cap);
            self.upper_in_degree.push(in_degree);
        }
        for new_layer in (old_span + 1)..=layer {
            let cap = if new_layer <= level {
                self.upper_cap_full()
            } else {
                ABOVE_LEVEL_CAP
            };
            self.open_list(cap);
        }
        self.upper_first[node as usize] = new_first;
        self.upper_span[node as usize] = layer as u8;
    }

    /// Widen one upper list to the full `m + 1` slots, which is what a push into
    /// a list that started above its owner's level needs the first time a second
    /// entry arrives. The old slots are left behind.
    fn widen_list(&mut self, node: u32, layer: usize) {
        let full = self.upper_cap_full();
        let list = self
            .upper_list(node, layer)
            .expect("a list is widened only where it exists");
        let (old_at, len, cap) = (
            self.upper_at[list] as usize,
            self.upper_len[list] as usize,
            self.upper_cap[list] as usize,
        );
        assert!(
            cap < full,
            "a list at layer {} already holds its {} slots, which the vendored reverse \
             update cannot exceed because it shrinks whenever it passes the threshold",
            layer,
            full
        );
        let new_at = self.upper_targets.len();
        self.upper_targets.resize(new_at + full, 0);
        self.upper_dists.resize(new_at + full, 0.);
        self.upper_targets.copy_within(old_at..old_at + len, new_at);
        self.upper_dists.copy_within(old_at..old_at + len, new_at);
        self.upper_at[list] = new_at as u32;
        self.upper_cap[list] = full as u16;
    }

    /// One entry of one list.
    #[inline]
    fn target_at(&self, node: u32, layer: usize, slot: usize) -> u32 {
        match self.slice_of(node, layer) {
            Some((at, len, upper)) => {
                assert!(slot < len, "slot {} of a list holding {}", slot, len);
                if upper {
                    self.upper_targets[at + slot]
                } else {
                    self.base_targets[at + slot]
                }
            }
            None => panic!("node {} owns no list at layer {}", node, layer),
        }
    }

    /// Lists at `layer` naming `node`, which is what the guarded pop reads.
    #[inline]
    fn in_degree(&self, node: u32, layer: usize) -> u32 {
        if layer == 0 {
            return self.base_in_degree[node as usize];
        }
        match self.upper_list(node, layer) {
            Some(list) => self.upper_in_degree[list],
            None => 0,
        }
    }

    /// Move one inbound counter, which every edge install and every eviction
    /// does exactly once.
    ///
    /// A node can be named at a layer above its own level, so this opens a list
    /// there rather than assuming one. The vendored counterpart never has to,
    /// since it carries sixteen counters on every point whatever its level.
    #[inline]
    fn bump_in_degree(&mut self, node: u32, layer: usize, delta: i32) {
        if layer > 0 {
            self.grow_span(node, layer);
        }
        let slot = if layer == 0 {
            &mut self.base_in_degree[node as usize]
        } else {
            let list = self
                .upper_list(node, layer)
                .expect("the span was just grown to cover this layer");
            &mut self.upper_in_degree[list]
        };
        *slot = slot
            .checked_add_signed(delta)
            .expect("an inbound counter is moved once per edge installed or evicted");
    }

    /// Whether one list already names a node, which the reverse update asks
    /// before it pushes.
    fn list_names(&self, node: u32, layer: usize, target: u32) -> bool {
        self.neighbours(node, layer).contains(&target)
    }

    /// One list as a vector of entries, which is how both sorts see it.
    fn copy_list(&self, node: u32, layer: usize, out: &mut Vec<Entry>) {
        out.clear();
        let targets = self.neighbours(node, layer);
        let dists = self.distances(node, layer);
        out.reserve(targets.len());
        for (&target, &dist) in targets.iter().zip(dists) {
            out.push(Entry { dist, target });
        }
    }

    /// Replace one list wholesale, which is what install site 2 does.
    fn write_list(&mut self, node: u32, layer: usize, list: &[Entry]) {
        if list.len() > self.list_cap(node, layer) {
            self.widen_list(node, layer);
        }
        let cap = self.list_cap(node, layer);
        assert!(
            list.len() <= cap,
            "a list at layer {} holds {} slots and {} entries were selected",
            layer,
            cap,
            list.len()
        );
        let (at, upper) = match self.slice_of(node, layer) {
            Some((at, _, upper)) => (at, upper),
            None => panic!("node {} owns no list at layer {}", node, layer),
        };
        let (targets, dists) = if upper {
            (&mut self.upper_targets, &mut self.upper_dists)
        } else {
            (&mut self.base_targets, &mut self.base_dists)
        };
        for (slot, entry) in list.iter().enumerate() {
            targets[at + slot] = entry.target;
            dists[at + slot] = entry.dist;
        }
        self.set_list_len(node, layer, list.len());
    }

    /// Append one edge to one list, which is what the reverse update does.
    fn push_edge(&mut self, node: u32, layer: usize, target: u32, dist: f32) {
        self.grow_span(node, layer);
        if self.list_len(node, layer) == self.list_cap(node, layer) {
            assert!(
                layer > 0,
                "a layer zero list cannot pass its threshold twice"
            );
            self.widen_list(node, layer);
        }
        let cap = self.list_cap(node, layer);
        let (at, len, upper) = match self.slice_of(node, layer) {
            Some(found) => found,
            None => panic!("node {} owns no list at layer {}", node, layer),
        };
        assert!(
            len < cap,
            "a list at layer {} already holds its {} slots, which the vendored reverse \
             update cannot reach because it shrinks whenever it exceeds the threshold",
            layer,
            cap
        );
        if upper {
            self.upper_targets[at + len] = target;
            self.upper_dists[at + len] = dist;
        } else {
            self.base_targets[at + len] = target;
            self.base_dists[at + len] = dist;
        }
        self.set_list_len(node, layer, len + 1);
    }

    /// Order one list by its stored distances, through the scratch buffer both
    /// sorts share. See [`Entry`] for why the sort runs over a contiguous slice
    /// of that type and not over the two arenas in place.
    fn sort_list(&mut self, node: u32, layer: usize, scratch: &mut Vec<Entry>) {
        self.copy_list(node, layer, scratch);
        scratch.sort_unstable_by(by_distance);
        self.write_list(node, layer, scratch);
    }

    /// Take one entry out of one list, shifting the rest down, and hand back
    /// the node it named. `Vec::remove` over the slab.
    fn remove_edge_at(&mut self, node: u32, layer: usize, slot: usize) -> u32 {
        let (at, len, upper) = match self.slice_of(node, layer) {
            Some(found) => found,
            None => panic!("node {} owns no list at layer {}", node, layer),
        };
        assert!(slot < len, "slot {} of a list holding {}", slot, len);
        let (targets, dists) = if upper {
            (&mut self.upper_targets, &mut self.upper_dists)
        } else {
            (&mut self.base_targets, &mut self.base_dists)
        };
        let removed = targets[at + slot];
        targets.copy_within(at + slot + 1..at + len, at + slot);
        dists.copy_within(at + slot + 1..at + len, at + slot);
        self.set_list_len(node, layer, len - 1);
        removed
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
        data: &[T],
        origin_id: usize,
        level: usize,
        above: &[(u8, u32, f32)],
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
            node < u32::MAX,
            "the graph holds as many nodes as a u32 counts"
        );

        self.origin_ids.push(origin_id);
        self.levels.push(level as u8);
        self.data.extend_from_slice(data);
        self.base_len.push(0);
        self.base_in_degree.push(0);
        self.base_targets
            .resize(self.base_targets.len() + self.base_cap(), 0);
        self.base_dists
            .resize(self.base_dists.len() + self.base_cap(), 0.);

        let mut span = level;
        for &(layer, _, _) in above {
            span = span.max(layer as usize);
        }
        self.open_node(span, level);

        for &(layer, target, dist) in above {
            self.push_edge(node, layer as usize, target, dist);
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
    /// three the vendored crate reports from `hnsw_rs::hnsw::guard_stats`, so
    /// the two builds can be compared on how often the guard fired and how
    /// often it changed the outcome.
    pub(super) fn guard_stats(&self) -> (u64, u64, u64) {
        (self.overflows, self.saves, self.fallbacks)
    }

    /// One node's adjacency by layer, as the vendored `get_neighborhood_id`
    /// reports the same point's: every layer the structure carries, each entry
    /// naming the id its target was inserted under and the distance stored
    /// beside it. The descent residue sits at the layer it was filed at, which
    /// is where the vendored point carries it too.
    ///
    /// This is the shape the two builders are compared in. It resolves targets
    /// to origin ids rather than to node indices, because the two structures
    /// number their nodes differently and the id is what both agree on.
    pub(super) fn neighbourhood_ids(&self, node: u32) -> Vec<Vec<(usize, f32)>> {
        let mut out = vec![Vec::new(); LAYERS];
        for (layer, list) in out.iter_mut().enumerate().take(self.span(node) + 1) {
            let targets = self.neighbours(node, layer);
            let dists = self.distances(node, layer);
            for (&target, &dist) in targets.iter().zip(dists) {
                list.push((self.origin_ids[target as usize], dist));
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
            for &target in self.neighbours(node, layer) {
                counts[target as usize] += 1;
            }
        }
        counts
    }

    /// Record how many entries one list now holds.
    #[inline]
    fn set_list_len(&mut self, node: u32, layer: usize, len: usize) {
        if layer == 0 {
            self.base_len[node as usize] = len as u16;
        } else {
            let list = self
                .upper_list(node, layer)
                .expect("a length is set only on a list that exists");
            self.upper_len[list] = len as u16;
        }
    }

    /// Bytes the structure has asked the allocator for.
    ///
    /// Exact rather than sampled, for the same reason `FlatGraph`'s figure is:
    /// every buffer's capacity is known and there is no per-node or per-edge
    /// allocation to estimate.
    pub(super) fn memory_bytes(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.origin_ids.capacity() * std::mem::size_of::<usize>();
        total += self.levels.capacity();
        total += self.data.capacity() * std::mem::size_of::<T>();
        total += self.base_targets.capacity() * std::mem::size_of::<u32>();
        total += self.base_dists.capacity() * std::mem::size_of::<f32>();
        total += self.base_len.capacity() * std::mem::size_of::<u16>();
        total += self.base_in_degree.capacity() * std::mem::size_of::<u32>();
        total += self.upper_first.capacity() * std::mem::size_of::<u32>();
        total += self.upper_span.capacity();
        total += self.upper_at.capacity() * std::mem::size_of::<u32>();
        total += self.upper_len.capacity() * std::mem::size_of::<u16>();
        total += self.upper_cap.capacity() * std::mem::size_of::<u16>();
        total += self.upper_in_degree.capacity() * std::mem::size_of::<u32>();
        total += self.upper_targets.capacity() * std::mem::size_of::<u32>();
        total += self.upper_dists.capacity() * std::mem::size_of::<f32>();
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
        traverse::search(self, data, knbn, ef_arg, filter)
    }

    /// One node's adjacency as a dump records it, being every list from layer
    /// zero up to the highest layer the node carries anything at.
    ///
    /// Empty lists in the middle are kept, because a node can carry residue at
    /// a layer above one it carries nothing at, and the file records lists by
    /// position. Trailing empty lists are the writer's business to trim.
    fn neighbourhood_into(&self, node: u32, order: &DumpOrder, out: &mut Vec<Vec<LoadedEdge>>) {
        for list in out.iter_mut() {
            list.clear();
        }
        while out.len() < LAYERS {
            out.push(Vec::new());
        }
        let lists = self.span(node) + 1;
        for (layer, list) in out.iter_mut().enumerate().take(lists) {
            let targets = self.neighbours(node, layer);
            let dists = self.distances(node, layer);
            list.reserve(targets.len());
            for (&target, &distance) in targets.iter().zip(dists) {
                list.push(LoadedEdge {
                    target: order.point_id(self, target),
                    distance,
                });
            }
        }
    }
}

impl<T, D> Topology for MutableGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    type Elem = T;
    type Dist = D;

    fn distance(&self) -> &D {
        &self.dist_f
    }

    fn nb_points(&self) -> usize {
        self.origin_ids.len()
    }

    fn entry(&self) -> u32 {
        self.entry
    }

    fn entry_level(&self) -> u8 {
        self.entry_level
    }

    #[inline]
    fn layer_len(&self, layer: usize) -> usize {
        self.layer_counts[layer] as usize
    }

    #[inline]
    fn vector(&self, node: u32) -> &[T] {
        let at = node as usize * self.dim;
        &self.data[at..at + self.dim]
    }

    #[inline]
    fn origin_id(&self, node: u32) -> usize {
        self.origin_ids[node as usize]
    }

    /// The neighbour list of one node at one layer.
    ///
    /// The whole list, at every layer the node owns one at, which is what the
    /// vendored `Point::neighbours[layer]` returns. That includes the layers
    /// above the node's own level.
    ///
    /// Relay 77 dropped those edges from `FlatGraph` and relay 78 held them out
    /// of this accessor, both on the reasoning that no traversal reaches a node
    /// at a layer above its level. That reasoning has a hole, which relay 79
    /// found while porting the insert: `search_layer` seeds its result heap with
    /// the point it was entered at whatever that point's level, so a point below
    /// the layer can enter a list there and be reached from it afterwards. The
    /// insertion traversal reads these lists on the vendored path, so it has to
    /// read them here.
    #[inline]
    fn neighbours(&self, node: u32, layer: usize) -> &[u32] {
        match self.slice_of(node, layer) {
            Some((at, len, false)) => &self.base_targets[at..at + len],
            Some((at, len, true)) => &self.upper_targets[at..at + len],
            None => &[],
        }
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
    order: DumpOrder,
}

impl<T, D> MutableGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// This graph as a dump source.
    pub(super) fn dump_view(&self) -> DumpView<'_, T, D> {
        DumpView {
            order: DumpOrder::of(self),
            graph: self,
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
                .neighbourhood_into(node, &self.order, &mut scratch);
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
            f(self.graph.vector(node))?;
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
