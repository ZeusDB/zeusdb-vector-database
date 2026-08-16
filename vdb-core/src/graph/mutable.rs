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
//! # The descent residue
//!
//! The vendored insert's descent records the entry point it passed at each
//! layer above the new point's own level into the new point's list there, so a
//! level zero point can carry entries at layer five. `FlatGraph` validates
//! those edges and drops them, because no traversal can reach them. This keeps
//! them, in an append-only region with one `u32` start per node.
//!
//! The region is append-only because the residue is written only during the
//! owner's own insertion and never afterwards. Install site 1 writes only into
//! the point being inserted, once per layer above its level; install site 2
//! writes only at layers up to the point's own level; install site 3 writes
//! into a neighbour chosen by a layer `l` traversal, whose level is therefore
//! at least `l`. So no later insertion can touch a residue list.
//!
//! Keeping it costs 4 bytes per node when it is empty and about 34 more when it
//! is not, measured on real dumps. It buys two things. A dump round trips
//! through this structure byte for byte, so the structure is a lossless image
//! of the file rather than a lossy one. And the construction relay keeps the
//! option of reproducing a vendored-built graph edge for edge, which is the
//! strongest test available to it.

use super::dump::EachNeighbourhood;
use super::traverse::{self, Topology, LAYERS};
use super::{Distance, GraphHit};
use hnsw_rs::hnsw::{LoadedEdge, LoadedPoint, PointId};

/// Naming no upper list, which is every node of level zero.
const NO_UPPER: u32 = u32::MAX;

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

    /// The first upper list a node owns, or [`NO_UPPER`] at level zero.
    upper_first: Vec<u32>,
    /// Upper layer targets, list `i` at `[i * upper_cap, ..)`.
    upper_targets: Vec<u32>,
    /// Upper layer distances, in step with [`Self::upper_targets`].
    upper_dists: Vec<f32>,
    /// Entries in each upper list.
    upper_len: Vec<u16>,

    /// Where each node's descent residue starts, with a final sentinel.
    residue_start: Vec<u32>,
    /// The layer each residue edge sits at, above its owner's level.
    residue_layer: Vec<u8>,
    /// The target of each residue edge.
    residue_target: Vec<u32>,
    /// The distance each residue edge carries.
    residue_dist: Vec<f32>,

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
    /// do not need. A neighbour list longer than its slab is refused, since the
    /// vendored builder cannot produce one and this layout cannot hold one. A
    /// refusal reaches the loader's rebuild path, which is what every other
    /// refusal reaches too.
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
        let upper_cap = m + 1;
        // One upper list per layer above zero that a node's level reaches.
        let upper_lists: usize = (1..LAYERS)
            .map(|layer| layer_counts[layer] as usize * layer)
            .sum();

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
            upper_first: Vec::with_capacity(nb_point),
            upper_targets: vec![0u32; upper_lists * upper_cap],
            upper_dists: vec![0f32; upper_lists * upper_cap],
            upper_len: vec![0u16; upper_lists],
            residue_start: Vec::with_capacity(nb_point + 1),
            residue_layer: Vec::new(),
            residue_target: Vec::new(),
            residue_dist: Vec::new(),
            dist_f,
        };

        let mut next_upper = 0u32;
        let mut edges: Vec<(f32, u32)> = Vec::new();
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
                graph.residue_start.push(
                    u32::try_from(graph.residue_target.len())
                        .map_err(|_| "the descent residue holds more edges than a u32 counts")?,
                );
                if layer == 0 {
                    graph.upper_first.push(NO_UPPER);
                } else {
                    graph.upper_first.push(next_upper);
                    next_upper += layer as u32;
                }

                for (list_layer, list) in point.neighbours.into_iter().enumerate() {
                    // Every edge is checked wherever it sits, and each list is
                    // ordered by its stored distances with the same stable sort
                    // the vendored constructor applies, so ties keep their dump
                    // order there and here.
                    edges.clear();
                    for edge in &list {
                        edges.push((edge.distance, edge_target(edge, &layer_offsets)?));
                    }
                    edges.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0)
                            .expect("every distance was checked finite by edge_target")
                    });
                    if list_layer > layer {
                        for &(distance, target) in &edges {
                            graph.residue_layer.push(list_layer as u8);
                            graph.residue_target.push(target);
                            graph.residue_dist.push(distance);
                        }
                        continue;
                    }
                    let cap = if list_layer == 0 { base_cap } else { upper_cap };
                    if edges.len() > cap {
                        return Err(format!(
                            "the point at layer {} rank {} carries {} neighbours at layer {} \
                             and a list holds {}",
                            layer,
                            rank,
                            edges.len(),
                            list_layer,
                            cap
                        ));
                    }
                    let (targets, dists, at) = if list_layer == 0 {
                        graph.base_len[node as usize] = edges.len() as u16;
                        (
                            &mut graph.base_targets,
                            &mut graph.base_dists,
                            node as usize * base_cap,
                        )
                    } else {
                        let list = graph.upper_first[node as usize] as usize + list_layer - 1;
                        graph.upper_len[list] = edges.len() as u16;
                        (
                            &mut graph.upper_targets,
                            &mut graph.upper_dists,
                            list * upper_cap,
                        )
                    };
                    for (slot, &(distance, target)) in edges.iter().enumerate() {
                        targets[at + slot] = target;
                        dists[at + slot] = distance;
                    }
                }
            }
        }
        graph.residue_start.push(
            u32::try_from(graph.residue_target.len())
                .map_err(|_| "the descent residue holds more edges than a u32 counts")?,
        );
        // The residue is the one region whose size is not known before the walk,
        // so it is the one region grown geometrically. Trimming it makes every
        // buffer's capacity its length, which is what lets `memory_bytes` be
        // checked against the counts rather than believed.
        graph.residue_layer.shrink_to_fit();
        graph.residue_target.shrink_to_fit();
        graph.residue_dist.shrink_to_fit();

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

    /// Slots one layer zero list holds, being the vendored threshold plus the
    /// overflow slot the guarded pop needs.
    #[inline]
    fn base_cap(&self) -> usize {
        2 * self.m + 1
    }

    /// Slots one upper list holds, on the same rule.
    #[inline]
    fn upper_cap(&self) -> usize {
        self.m + 1
    }

    /// Nodes the graph holds.
    pub(super) fn nb_points(&self) -> usize {
        self.origin_ids.len()
    }

    /// Edges the graph holds in its slabs, over every layer. The descent
    /// residue is not counted here; see [`Self::above_level_edges`].
    pub(super) fn nb_edges(&self) -> usize {
        let base: usize = self.base_len.iter().map(|&len| len as usize).sum();
        let upper: usize = self.upper_len.iter().map(|&len| len as usize).sum();
        base + upper
    }

    /// Edges sitting at layers above their owner's level, which the vendored
    /// descent leaves behind and this structure keeps.
    pub(super) fn above_level_edges(&self) -> usize {
        self.residue_target.len()
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
        let node = node as usize;
        if layer == 0 {
            return Some((node * self.base_cap(), self.base_len[node] as usize, false));
        }
        if layer > self.levels[node] as usize {
            return None;
        }
        let list = self.upper_first[node] as usize + layer - 1;
        Some((list * self.upper_cap(), self.upper_len[list] as usize, true))
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
        total += self.upper_first.capacity() * std::mem::size_of::<u32>();
        total += self.upper_targets.capacity() * std::mem::size_of::<u32>();
        total += self.upper_dists.capacity() * std::mem::size_of::<f32>();
        total += self.upper_len.capacity() * std::mem::size_of::<u16>();
        total += self.residue_start.capacity() * std::mem::size_of::<u32>();
        total += self.residue_layer.capacity();
        total += self.residue_target.capacity() * std::mem::size_of::<u32>();
        total += self.residue_dist.capacity() * std::mem::size_of::<f32>();
        total
    }

    /// Search the graph, which is [`traverse::search`] over this layout.
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
        let lists = self.levels[node as usize] as usize + 1;
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
        let from = self.residue_start[node as usize] as usize;
        let to = self.residue_start[node as usize + 1] as usize;
        for at in from..to {
            out[self.residue_layer[at] as usize].push(LoadedEdge {
                target: order.point_id(self, self.residue_target[at]),
                distance: self.residue_dist[at],
            });
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
    /// Empty above the node's own level, which is how the vendored structure
    /// answers the same question for a node the traversal reaches there, and
    /// which is what makes the descent residue unreadable by any search. The
    /// residue is stored and it is deliberately not returned here.
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
