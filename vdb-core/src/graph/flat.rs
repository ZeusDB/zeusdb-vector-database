//! ZeusDB's own graph structure, and the traversal over it.
//!
//! This is the replacement for the vendored `Hnsw`, built ahead of the 0.7.0
//! cutover. It exists in the tree fully tested and reached by nothing outside
//! `graph/`, because a graph is one object and cannot be half replaced: either
//! the vendored crate holds the points and edges or this does. Construction has
//! not been written yet, so wiring this in would put reads on one structure
//! while writes still go to the other. The parity tests in `parity.rs` are what
//! hold it to the vendored traversal until the cutover lands.
//!
//! # The shape
//!
//! One flat arena. A node is a `u32` index in the same (layer, rank) order the
//! dump streams points in, layer major, so the node at rank `r` of layer `l`
//! sits at `layer_offsets[l] + r`. The vendored `points_by_layer` is a
//! partition, each point living only in the layer of its own top level, so this
//! order is total and a `PointId` converts to a flat index with one table
//! lookup.
//!
//! Around that index everything is struct-of-arrays: the origin ids in one
//! `Vec`, every vector in one contiguous slab, and the adjacency in one CSR
//! pair per layer. The vendored crate spends 112 bytes and six allocations per
//! point on an `Arc<Point>` carrying a `PointData` enum, a `PointId`, an
//! `Arc<RwLock<Vec<Vec<Arc<PointWithOrder>>>>>` and 64 bytes of in-degree
//! counters, plus a 32 byte `Arc` allocation per edge. Here a node costs its
//! origin id and its CSR start slots, about 12 bytes beside the vector, and an
//! edge costs the 4 bytes of its target index. The in-degree counters are not
//! stored at all: they are patch 3's overflow-guard bookkeeping, a pure
//! function of the adjacency that search never reads, and construction can
//! rebuild them by one walk when it needs them.
//!
//! # The adjacency
//!
//! The CSR for layer `l` covers the nodes whose top level is at least `l`,
//! which in flat order are exactly the suffix starting at `layer_offsets[l]`.
//! So each layer carries a `starts` array over that suffix and one `targets`
//! array, and reaching a neighbour list is a subtraction and two loads.
//!
//! Two facts about the vendored graph make that coverage sufficient.
//!
//! First, every **target** in a layer `l` list has top level at least `l`.
//! The three edge install sites all take their targets from a layer `l`
//! traversal, whose reachable set is the descent pivot and the targets of
//! layer `l` lists, both of which hold the property inductively, and the
//! reverse update installs the new point into layers up to its own level
//! only. So the traversal, which reads the lists of nodes it reached at layer
//! `l`, only ever asks about nodes the suffix covers. The
//! `node < layer_offsets[layer]` guard answers an edge below the layer with
//! an empty list, which is what the vendored traversal reads for the same
//! node, since every vendored point carries all sixteen list slots.
//!
//! Second, an **owner** can be below its list. The vendored insert's descent
//! records the entry point it passed at each layer above the new point's own
//! level into the new point's list there, so a point of level zero can carry
//! entries at layer five. Nothing ever reads them: the traversal reads the
//! lists of reached nodes only, and by the first fact a node is only reached
//! at layers up to its own level. They are write-only residue of the descent.
//! The loader validates them like every other edge, counts them, and drops
//! them, and [`FlatGraph::above_level_edges`] reports the count. The parity
//! harness confirms on real dumps that the pages do not move.
//!
//! Stored per-edge distances are used to order each list at load time, exactly
//! as `Hnsw::from_loaded_points` orders it with a stable sort, and then
//! dropped. The traversal evaluates every candidate against the query and
//! never reads a stored distance. Construction will need them back; that is
//! recorded in the relay report rather than paid for here.
//!
//! # Concurrency
//!
//! Plain data. No lock, no atomic, no interior mutability, no `Python` token,
//! and `search` takes `&self`, so the structure is `Send + Sync` whenever the
//! element and distance types are. At cutover it sits where `VectorGraph`
//! sits, behind the index's `RwLock`, and concurrent searches share read
//! guards exactly as they do today. Nothing a search returns borrows from the
//! structure: a [`GraphHit`] owns its two fields.
//!
//! # The traversal
//!
//! [`FlatGraph::search`] is a line-for-line port of the vendored
//! `Hnsw::search_filter` with patch 4 applied, over indices instead of `Arc`s.
//! The parity contract it holds, and the places it deliberately differs, are
//! written on the function.

use super::{Distance, GraphHit};
use hnsw_rs::hnsw::{LoadedEdge, LoadedPoint, PointId, NB_LAYER_MAX};
use std::collections::BinaryHeap;

/// Layers every graph carries, which is the vendored crate's fixed count.
const LAYERS: usize = NB_LAYER_MAX as usize;

/// One layer's adjacency, in compressed sparse row form.
///
/// `starts` has one entry per node of the layer's suffix plus a sentinel, and
/// `targets` holds the neighbour lists back to back. The list of suffix node
/// `i` is `targets[starts[i] .. starts[i + 1]]`, in the stored order, which is
/// ascending stored distance with ties left in dump order.
struct LayerAdjacency {
    starts: Vec<u32>,
    targets: Vec<u32>,
}

/// The flat graph: an arena of nodes, a vector slab, and CSR adjacency.
///
/// Generic exactly as the vendored `Hnsw` is generic, over the element type
/// the vectors hold and the distance evaluated between a query and a stored
/// vector. The six `VectorGraph` variants map onto `FlatGraph<f32, _>` for the
/// three raw spaces and `FlatGraph<u8, DistPQ>` for the three quantized ones.
pub(super) struct FlatGraph<T, D> {
    /// Values per stored vector.
    dim: usize,
    /// `max_nb_connection`, kept for construction.
    m: usize,
    /// Kept for construction.
    ef_construction: usize,
    /// The level generator's scale, kept for construction.
    level_scale: f64,
    /// Flat index of the node the traversal starts from.
    entry: u32,
    /// The entry node's top level, which is the highest occupied layer.
    entry_level: u8,
    /// `layer_offsets[l]` is the flat index of the first node of layer `l`,
    /// and the final entry is the node count.
    layer_offsets: [u32; LAYERS + 1],
    /// The id each node was inserted under, which hits report.
    origin_ids: Vec<usize>,
    /// Every vector, node `n` at `[n * dim, (n + 1) * dim)`.
    data: Vec<T>,
    /// Adjacency per layer.
    layers: [LayerAdjacency; LAYERS],
    /// Edges the dump carried at layers above their owner's level, validated
    /// and dropped at load; see the module documentation.
    above_level_edges: usize,
    /// The distance, evaluated between the query and a stored vector.
    dist_f: D,
}

/// A heap entry: one node ordered by its distance to the query.
///
/// The ordering is the vendored `PointWithOrder` ordering exactly, being the
/// distance alone with the same panic on a NaN, so `std::BinaryHeap` evolves
/// through the identical sequence of comparisons and equal distances resolve
/// identically. The node index takes no part in the order.
#[derive(Clone, Copy, Debug)]
struct OrderedNode {
    dist_to_ref: f32,
    node: u32,
}

impl PartialEq for OrderedNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist_to_ref == other.dist_to_ref
    }
}

impl Eq for OrderedNode {}

impl PartialOrd for OrderedNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if !self.dist_to_ref.is_nan() && !other.dist_to_ref.is_nan() {
            self.dist_to_ref.partial_cmp(&other.dist_to_ref).unwrap()
        } else {
            // The vendored ordering's exact behaviour, message included, so a
            // non-finite distance that reaches a heap fails the same way on
            // both structures.
            panic!("got a NaN in a distance");
        }
    }
}

/// Nodes already seen by one traversal, as one bit each.
///
/// The vendored `search_layer` keeps a `HashMap<PointId, Arc<Point>>` and asks
/// `contains_key` before inserting. Only the set membership is ever read, so
/// this carries the membership and nothing else: no hash, no allocation per
/// visit, one allocation per search.
struct Visited {
    words: Vec<u64>,
}

impl Visited {
    fn new(nodes: usize) -> Self {
        Visited {
            words: vec![0u64; nodes.div_ceil(64)],
        }
    }

    /// Mark `node` visited, answering whether it already was.
    #[inline]
    fn test_and_set(&mut self, node: u32) -> bool {
        let word = &mut self.words[(node >> 6) as usize];
        let mask = 1u64 << (node & 63);
        let was = *word & mask != 0;
        *word |= mask;
        was
    }
}

impl<T, D> FlatGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// Build the structure from the topology the dump reader produces.
    ///
    /// The same signature and the same acceptance rules as the vendored
    /// `Hnsw::from_loaded_points`, because the two are fed the same parsed
    /// dump and must agree on what a malformed one is. Every edge is
    /// validated, wherever it sits. Edges at layers above their owner's level
    /// are then dropped rather than stored, because the traversal cannot
    /// reach them; the module documentation carries the argument and the
    /// count survives in [`Self::above_level_edges`].
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
                "the graph holds {} points and a flat index is a u32",
                nb_point
            ));
        }
        for layer in 0..LAYERS {
            let count = points_by_layer.get(layer).map_or(0, Vec::len);
            layer_offsets[layer + 1] = layer_offsets[layer] + count as u32;
        }

        let dim = points_by_layer
            .iter()
            .find_map(|layer| layer.first())
            .map(|point| point.data.len())
            .expect("nb_point is positive, so some layer holds a point");

        // First pass: move every point's id and vector into the arena and set
        // its adjacency aside, indexed by flat node. Lists above the point's
        // own level are validated here and counted, because the CSR pass below
        // never visits them; see the module documentation for why they exist
        // and why dropping them moves no search result.
        let mut origin_ids = Vec::with_capacity(nb_point);
        let mut data = Vec::with_capacity(nb_point * dim);
        let mut adjacency = Vec::with_capacity(nb_point);
        let mut above_level_edges = 0usize;
        for (layer, points) in points_by_layer.into_iter().enumerate() {
            for (rank, point) in points.into_iter().enumerate() {
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
                for list in point.neighbours.iter().skip(layer + 1) {
                    for edge in list {
                        edge_target(edge, &layer_offsets)?;
                        above_level_edges += 1;
                    }
                }
                origin_ids.push(point.origin_id);
                data.extend(point.data);
                adjacency.push(point.neighbours);
            }
        }

        // Second pass: one CSR per layer, walking that layer's suffix of the
        // arena. Each list is checked, converted to flat indices, and ordered
        // by its stored distances with the same stable sort the vendored
        // constructor applies, so ties keep their dump order there and here.
        let mut edges: Vec<(f32, u32)> = Vec::new();
        let layers = try_array(|layer| {
            let base = layer_offsets[layer] as usize;
            let mut starts = Vec::with_capacity(nb_point - base + 1);
            let mut targets: Vec<u32> = Vec::new();
            starts.push(0u32);
            for lists in adjacency[base..].iter_mut() {
                if let Some(list) = lists.get_mut(layer) {
                    edges.clear();
                    for edge in list.drain(..) {
                        let flat = edge_target(&edge, &layer_offsets)?;
                        edges.push((edge.distance, flat));
                    }
                    edges.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0)
                            .expect("every distance was checked finite by edge_target")
                    });
                    targets.extend(edges.iter().map(|&(_, flat)| flat));
                }
                let len = u32::try_from(targets.len())
                    .map_err(|_| "a layer holds more edges than a u32 counts".to_string())?;
                starts.push(len);
            }
            targets.shrink_to_fit();
            Ok(LayerAdjacency { starts, targets })
        })?;

        let entry_layer = entry_point.0 as usize;
        let entry_span = if entry_layer < LAYERS {
            layer_offsets[entry_layer + 1] - layer_offsets[entry_layer]
        } else {
            0
        };
        if entry_layer >= LAYERS || entry_point.1 < 0 || entry_point.1 as u32 >= entry_span {
            return Err(format!(
                "the entry point sits at layer {} rank {} and no point is there",
                entry_point.0, entry_point.1
            ));
        }

        Ok(FlatGraph {
            dim,
            m,
            ef_construction,
            level_scale,
            entry: layer_offsets[entry_layer] + entry_point.1 as u32,
            entry_level: entry_point.0,
            layer_offsets,
            origin_ids,
            data,
            layers,
            above_level_edges,
            dist_f,
        })
    }

    /// Edges the dump carried at layers above their owners' levels, dropped
    /// at load because no traversal can reach them.
    pub(super) fn above_level_edges(&self) -> usize {
        self.above_level_edges
    }

    /// Nodes the graph holds.
    pub(super) fn nb_points(&self) -> usize {
        self.origin_ids.len()
    }

    /// Edges the graph holds, over every layer.
    pub(super) fn nb_edges(&self) -> usize {
        self.layers.iter().map(|layer| layer.targets.len()).sum()
    }

    /// Values per stored vector.
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// `max_nb_connection`, as construction will need it.
    pub(super) fn m(&self) -> usize {
        self.m
    }

    /// As construction will need it.
    pub(super) fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// The level generator's scale, as construction will need it.
    pub(super) fn level_scale(&self) -> f64 {
        self.level_scale
    }

    /// Where the traversal starts, as (layer, rank), which is what a dump
    /// records.
    pub(super) fn entry_point_id(&self) -> PointId {
        PointId(
            self.entry_level,
            (self.entry - self.layer_offsets[self.entry_level as usize]) as i32,
        )
    }

    /// Points in one layer.
    #[inline]
    fn layer_len(&self, layer: usize) -> usize {
        (self.layer_offsets[layer + 1] - self.layer_offsets[layer]) as usize
    }

    /// The stored vector of one node.
    #[inline]
    pub(super) fn vector(&self, node: u32) -> &[T] {
        let at = node as usize * self.dim;
        &self.data[at..at + self.dim]
    }

    /// The id the node was inserted under, which is what hits report and what
    /// a filter is asked about.
    pub(super) fn origin_id(&self, node: u32) -> usize {
        self.origin_ids[node as usize]
    }

    /// The neighbour list of one node at one layer.
    ///
    /// Empty for a node below the layer, which is how the vendored structure
    /// answers the same question, since every vendored point carries all
    /// sixteen lists and only fills the ones up to its own level.
    #[inline]
    fn neighbours(&self, node: u32, layer: usize) -> &[u32] {
        let base = self.layer_offsets[layer];
        if node < base {
            return &[];
        }
        let adjacency = &self.layers[layer];
        let at = (node - base) as usize;
        let start = adjacency.starts[at] as usize;
        let end = adjacency.starts[at + 1] as usize;
        &adjacency.targets[start..end]
    }

    /// Bytes the structure has asked the allocator for.
    ///
    /// Exact rather than sampled, which is what the flat layout buys: every
    /// buffer's capacity is known, and there is no per-node or per-edge
    /// allocation to estimate. The vendored counterpart `graph_memory_bytes`
    /// samples 4,096 points and scales.
    pub(super) fn memory_bytes(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.origin_ids.capacity() * std::mem::size_of::<usize>();
        total += self.data.capacity() * std::mem::size_of::<T>();
        for layer in &self.layers {
            total += layer.starts.capacity() * std::mem::size_of::<u32>();
            total += layer.targets.capacity() * std::mem::size_of::<u32>();
        }
        total
    }

    /// Search the graph: the vendored `Hnsw::search_filter` over flat indices.
    ///
    /// The port is line for line, so that on identical topology the two return
    /// identical pages, ids and score bits both. The descent scans the pivot's
    /// list once per layer from the entry level down to one, taking the first
    /// strict improvement scan order finds. The bottom search is
    /// [`Self::search_layer`]. The width is `ef_arg.max(knbn)`, the cut is
    /// `knbn.min(ef)`, and the filtered arm re-tests the predicate over the
    /// cut page exactly as the vendored function does, although patch 4 made
    /// that re-test a no-op by keeping only admitted points in the heap.
    ///
    /// Four deliberate differences, none of which moves a result.
    ///
    /// A neighbour is a `u32` read from the CSR, so the traversal allocates
    /// nothing per edge where the vendored one allocates an `Arc` per heap
    /// entry. The visited set is a bitset rather than a `HashMap` of `Arc`s.
    /// The predicate is a monomorphised `Fn` rather than a `&dyn FilterT`.
    /// And two vendored checks have no equivalent because the states they
    /// answer cannot exist here: the empty-graph early return, since
    /// `from_loaded` refuses a graph of no points, and `search_layer`'s
    /// negative-rank check, since a flat index is unsigned.
    ///
    /// A non-finite query is rejected by every ZeusDB entry point before the
    /// seam. One that reaches the traversal anyway behaves as it does on the
    /// vendored path: a NaN distance panics with the vendored message the
    /// moment it enters a heap comparison, and an infinite distance traverses
    /// normally and scores the page it returns.
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
        let mut dist_to_entry = self.dist_f.eval(data, self.vector(self.entry));
        let mut pivot = self.entry;
        let mut new_pivot = None;

        for layer in (1..=self.entry_level).rev() {
            let mut has_changed = false;
            for &neighbour in self.neighbours(pivot, layer as usize) {
                let tmp_dist = self.dist_f.eval(data, self.vector(neighbour));
                if tmp_dist < dist_to_entry {
                    new_pivot = Some(neighbour);
                    has_changed = true;
                    dist_to_entry = tmp_dist;
                }
            }
            if has_changed {
                pivot = new_pivot.expect("has_changed is only set beside new_pivot");
            }
        }

        let ef = ef_arg.max(knbn);
        let layer_to_search = (0..LAYERS)
            .find(|&layer| self.layer_len(layer) > 0)
            .expect("a loaded graph holds at least one point");

        let neighbours_heap = self.search_layer(data, pivot, ef, layer_to_search, filter);
        let neighbours = neighbours_heap.into_sorted_vec();
        let last = knbn.min(ef).min(neighbours.len());

        let mut hits = Vec::with_capacity(last);
        match filter {
            Some(admits) => {
                for point in &neighbours[..last] {
                    let origin_id = self.origin_ids[point.node as usize];
                    if admits(&origin_id) {
                        hits.push(GraphHit {
                            internal_id: origin_id,
                            distance: point.dist_to_ref,
                        });
                    }
                }
            }
            None => {
                for point in &neighbours[..last] {
                    hits.push(GraphHit {
                        internal_id: self.origin_ids[point.node as usize],
                        distance: point.dist_to_ref,
                    });
                }
            }
        }
        hits
    }

    /// The bottom-layer traversal: the vendored `search_layer` with patch 4,
    /// over flat indices.
    ///
    /// Positive distances in the result heap, negated distances in the
    /// candidate heap, the entry admitted to the results only where the filter
    /// admits it, `INFINITY` standing for the bound while the result heap is
    /// empty, the candidate pushed before the filter is consulted, and the
    /// result heap trimmed above `ef`. The two vendored assertions are kept,
    /// because they are behaviour.
    fn search_layer<F>(
        &self,
        point: &[T],
        entry: u32,
        ef: usize,
        layer: usize,
        filter: Option<&F>,
    ) -> BinaryHeap<OrderedNode>
    where
        F: Fn(&usize) -> bool,
    {
        let skiplist_size = ef.max(2);
        // Patch 4: one slot is the smallest width that terminates under a
        // filter; see the vendored function.
        let ef = ef.max(1);
        let mut return_points = BinaryHeap::with_capacity(skiplist_size);
        if self.layer_len(layer) == 0 {
            return return_points;
        }

        let dist_to_entry_point = self.dist_f.eval(point, self.vector(entry));
        let mut visited = Visited::new(self.origin_ids.len());
        visited.test_and_set(entry);

        let mut candidate_points = BinaryHeap::with_capacity(skiplist_size);
        candidate_points.push(OrderedNode {
            dist_to_ref: -dist_to_entry_point,
            node: entry,
        });
        let entry_admitted = match filter {
            None => true,
            Some(admits) => admits(&self.origin_ids[entry as usize]),
        };
        if entry_admitted {
            return_points.push(OrderedNode {
                dist_to_ref: dist_to_entry_point,
                node: entry,
            });
        }

        while let Some(c) = candidate_points.pop() {
            assert!(c.dist_to_ref <= 0.);
            let f_dist_to_ref = match return_points.peek() {
                Some(f) => {
                    assert!(f.dist_to_ref >= 0.);
                    f.dist_to_ref
                }
                None => f32::INFINITY,
            };
            if -(c.dist_to_ref) > f_dist_to_ref {
                return return_points;
            }
            for &e in self.neighbours(c.node, layer) {
                if !visited.test_and_set(e) {
                    let f_dist_to_p = match return_points.peek() {
                        Some(f) => f.dist_to_ref,
                        None => f32::INFINITY,
                    };
                    let e_dist_to_p = self.dist_f.eval(point, self.vector(e));
                    if e_dist_to_p < f_dist_to_p || return_points.len() < ef {
                        candidate_points.push(OrderedNode {
                            dist_to_ref: -e_dist_to_p,
                            node: e,
                        });
                        let admitted = match filter {
                            None => true,
                            Some(admits) => admits(&self.origin_ids[e as usize]),
                        };
                        if admitted {
                            return_points.push(OrderedNode {
                                dist_to_ref: e_dist_to_p,
                                node: e,
                            });
                            if return_points.len() > ef {
                                return_points.pop();
                            }
                        }
                    }
                }
            }
        }
        return_points
    }
}

/// Check one edge against the layer table and hand back its flat target.
///
/// The same three rules `Hnsw::from_loaded_points` applies to every edge: a
/// finite distance, a target layer the graph carries, and a target rank the
/// layer holds. Applied to every edge whatever layer its list sits at, so an
/// edge the CSR drops is still an edge the loader vouched for.
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

/// Build a `[T; LAYERS]` from a fallible constructor, in index order.
///
/// `std::array::try_from_fn` is unstable, and collecting into a `Vec` first
/// then converting is the stable route. The `Vec` is exactly `LAYERS` long by
/// construction, so the conversion cannot fail.
fn try_array<A>(mut build: impl FnMut(usize) -> Result<A, String>) -> Result<[A; LAYERS], String> {
    let mut built = Vec::with_capacity(LAYERS);
    for index in 0..LAYERS {
        built.push(build(index)?);
    }
    Ok(built
        .try_into()
        .unwrap_or_else(|_| unreachable!("the loop above pushed exactly LAYERS items")))
}
