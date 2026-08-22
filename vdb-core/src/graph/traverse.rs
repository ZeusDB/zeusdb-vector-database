//! The traversal, over whatever structure holds the topology.
//!
//! This is the vendored `Hnsw::search_filter` with patch 4 applied, ported once
//! and written against an accessor rather than against a layout. Two structures
//! implement that accessor: [`super::flat::FlatGraph`], whose frozen CSR is the
//! right shape for a loaded read-only index, and
//! [`super::mutable::MutableGraph`], whose fixed-capacity slabs are the shape
//! construction needs. Both return the same page from the same topology,
//! because both run this code.
//!
//! The port was proved against the vendored traversal over 45,465 pages of real
//! data. Lifting it out of `flat.rs` unchanged is what carries that proof
//! onto the second structure: the parity tests that held the CSR to the
//! vendored page still run, and the new ones differ only in which accessor is
//! passed in.

use super::dump::NB_LAYER_MAX;
use super::{Distance, GraphHit};
use std::collections::BinaryHeap;

/// Layers every graph carries, which is the vendored crate's fixed count.
pub(super) const LAYERS: usize = NB_LAYER_MAX as usize;

/// What the traversal needs of a graph, and nothing more.
///
/// A node is a `u32` index into whatever arena the implementor holds. The
/// traversal never learns how a neighbour list is stored, only that asking for
/// one by node and layer hands back a slice of node indices in the graph's own
/// order.
pub(super) trait Topology {
    /// The element type stored vectors hold. The bounds are the vendored
    /// `Distance` trait's own, since it is what evaluates them.
    type Elem: Send + Sync;
    /// The distance evaluated between a query and a stored vector.
    type Dist: Distance<Self::Elem>;

    /// The distance, which the traversal evaluates rather than reads.
    fn distance(&self) -> &Self::Dist;

    /// Nodes the graph holds, which bounds every node index.
    fn nb_points(&self) -> usize;

    /// Where the descent starts.
    fn entry(&self) -> u32;

    /// The entry node's top level, which is the highest occupied layer.
    fn entry_level(&self) -> u8;

    /// Nodes whose top level is exactly `layer`.
    ///
    /// The vendored `points_by_layer` is a partition rather than a nesting, so
    /// this counts the points the layer owns and not the points that carry
    /// adjacency there.
    fn layer_len(&self, layer: usize) -> usize;

    /// One node's stored vector.
    fn vector(&self, node: u32) -> &[Self::Elem];

    /// The id the node was inserted under, which hits report and a filter is
    /// asked about.
    fn origin_id(&self, node: u32) -> usize;

    /// One node's neighbour list at one layer, empty where it has none.
    fn neighbours(&self, node: u32, layer: usize) -> &[u32];
}

/// A heap entry: one node ordered by its distance to the query.
///
/// The ordering is the vendored `PointWithOrder` ordering exactly, being the
/// distance alone with the same panic on a NaN, so `std::BinaryHeap` evolves
/// through the identical sequence of comparisons and equal distances resolve
/// identically. The node index takes no part in the order.
#[derive(Clone, Copy, Debug)]
pub(super) struct OrderedNode {
    pub(super) dist_to_ref: f32,
    pub(super) node: u32,
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

/// Bytes one cache line holds on every target this ships to.
const CACHE_LINE: usize = 64;

/// Leading cache lines of a neighbour's vector to hint before it is scored.
///
/// This was swept on the three real sets at 100,000 records against a build
/// with the hint absent, alternating the builds over six rounds and
/// taking the minimum of every pass. The figure is the whole unfiltered search
/// through the Python entry point, so it prices the traversal plus the fixed
/// cost of the call around it rather than the traversal alone.
///
/// | Lines | Bytes hinted | sift-128 | glove-100 | dbpedia-1536 |
/// |---:|---:|---:|---:|---:|
/// | no hint at all | 0 | 376.9 us | 358.4 us | 1,154.5 us |
/// | 0 | 0 | 382.0 us | 352.6 us | 1,160.1 us |
/// | 2 | 128 | 316.5 us | 340.1 us | 1,158.0 us |
/// | 4 | 256 | 324.8 us | 310.6 us | 1,159.1 us |
/// | 8 | 512 | 304.1 us | 290.7 us | 1,155.4 us |
/// | 16 | 1,024 | 299.4 us | 293.9 us | 1,174.3 us |
/// | 32 | 2,048 | 293.5 us | 261.2 us | 1,204.9 us |
///
/// The zero row is this function compiled with the hint clamped away, which
/// leaves the hoisted neighbour slice and a call that inlines to nothing. It
/// lands within one percent of the no-hint build at all three dimensions, so
/// the restructuring costs nothing and every difference below is the hint.
///
/// **Eight, sixteen and thirty-two are the same code at 128 and 100
/// dimensions.** A sift vector is 512 bytes, being exactly eight lines, and a
/// glove one is 400 bytes, being seven, so the clamp against the vector's own
/// length makes all three issue an identical number of hints. Their measured
/// figures differ by 3.5 percent on sift and 12.6 percent on glove, and that is
/// this machine's noise floor on a minimum of sixty passes rather than an
/// effect. Read against it, the hint is worth **1.24 times on sift and 1.23
/// times on glove**, and two and four lines are worth less because they cover
/// part of a row.
///
/// **At 1,536 dimensions no line count is worth anything.** A dbpedia vector is
/// 6,144 bytes, being 96 lines, so two, four, eight, sixteen and thirty-two are
/// five genuinely different amounts of hinting and all five land inside 4.4
/// percent of the no-hint build. An earlier sweep measured 1.01 to 1.08 times
/// there and flagged that its null result might be an artefact of its own eight
/// line cap. **It is not.** The cause is in the bandwidth: at 1,536
/// dimensions the hardware streamer has a 96 line run to work with once the
/// first line lands and the search is already within 1.43 times the sequential
/// floor, where at 128 dimensions a row is eight lines, there is no run to
/// stream, and the search pays four times that floor.
///
/// Eight is the value chosen. It is the smallest count that covers a whole row
/// at both dimensions the hint is worth anything at, and nothing above it can
/// add there because the clamp discards it.
const PREFETCH_LINES: usize = 8;

/// Hint the memory system for the vectors a neighbour list is about to be
/// scored against.
///
/// A prefetch is a hint and nothing else. It moves no data the program can
/// observe, sets no flag, and the architecture defines it as unable to fault,
/// so this cannot change which page `search_layer` returns and cannot fail. The
/// proof is not left to that argument: recall at 10 was compared to four decimal
/// places, and a fixed query set's ids and score bits with the hint present and
/// absent, on all three sets, and every figure is identical.
///
/// The `unsafe` is `_mm_prefetch`'s own, which `core::arch` marks unsafe
/// because it is a target feature intrinsic rather than because the operation
/// can go wrong. What this function has to get right is the pointer it forms,
/// since `ptr::add` outside an allocation is undefined whatever the instruction
/// does with the result. The line count is clamped to the lines the vector
/// itself occupies, so the last offset is strictly inside the slice and every
/// pointer formed here is in bounds of the arena the vector lives in.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
fn prefetch_vectors<G>(graph: &G, neighbours: &[u32])
where
    G: Topology + ?Sized,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_prefetch, _MM_HINT_T0};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

    for &node in neighbours {
        let vector = graph.vector(node);
        let bytes = std::mem::size_of_val(vector);
        let base = vector.as_ptr() as *const i8;
        let lines = bytes.div_ceil(CACHE_LINE).min(PREFETCH_LINES);
        for line in 0..lines {
            // SAFETY: `line` is below `bytes.div_ceil(CACHE_LINE)`, so
            // `line * CACHE_LINE` is below `bytes` and the offset pointer is
            // inside the vector's own bytes rather than one past them. The
            // intrinsic is baseline on both of these targets, needs no runtime
            // detection, and issues a hint that reads nothing.
            unsafe { _mm_prefetch(base.add(line * CACHE_LINE), _MM_HINT_T0) };
        }
    }
}

/// The no-op every other target compiles.
///
/// `_mm_prefetch` is x86 only. There is no portable prefetch intrinsic on
/// stable Rust, so aarch64 and everything else get an empty function that
/// inlines away, and the traversal they run is the one that shipped before this
/// hint existed. Nothing about the page depends on which of the two is
/// compiled.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[inline]
fn prefetch_vectors<G>(_graph: &G, _neighbours: &[u32])
where
    G: Topology + ?Sized,
{
}

/// Search the graph: the vendored `Hnsw::search_filter` over node indices.
///
/// The port is line for line, so that on identical topology this and the
/// vendored function return identical pages, ids and score bits both. The
/// descent scans the pivot's list once per layer from the entry level down to
/// one, taking the first strict improvement scan order finds. The bottom search
/// is [`search_layer`]. The width is `ef_arg.max(knbn)`, the cut is
/// `knbn.min(ef)`, and the filtered arm re-tests the predicate over the cut
/// page exactly as the vendored function does, although patch 4 made that
/// re-test a no-op by keeping only admitted points in the heap.
///
/// Four deliberate differences, none of which moves a result.
///
/// A neighbour is a `u32` read from the graph's own arena, so the traversal
/// allocates nothing per edge where the vendored one allocates an `Arc` per
/// heap entry. The visited set is a bitset rather than a `HashMap` of `Arc`s.
/// The predicate is a monomorphised `Fn` rather than a `&dyn FilterT`. And two
/// vendored checks have no equivalent because the states they answer cannot
/// exist here: the empty-graph early return, since neither constructor accepts
/// a graph of no points, and `search_layer`'s negative-rank check, since a node
/// index is unsigned.
///
/// A non-finite query is rejected by every ZeusDB entry point before the seam.
/// One that reaches the traversal anyway behaves as it does on the vendored
/// path: a NaN distance panics with the vendored message the moment it enters a
/// heap comparison, and an infinite distance traverses normally and scores the
/// page it returns.
pub(super) fn search<G, F>(
    graph: &G,
    data: &[G::Elem],
    knbn: usize,
    ef_arg: usize,
    filter: Option<&F>,
) -> Vec<GraphHit>
where
    G: Topology + ?Sized,
    F: Fn(&usize) -> bool,
{
    let dist_f = graph.distance();
    let entry = graph.entry();
    let mut dist_to_entry = dist_f.eval(data, graph.vector(entry));
    let mut pivot = entry;
    let mut new_pivot = None;

    for layer in (1..=graph.entry_level()).rev() {
        let mut has_changed = false;
        for &neighbour in graph.neighbours(pivot, layer as usize) {
            let tmp_dist = dist_f.eval(data, graph.vector(neighbour));
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
        .find(|&layer| graph.layer_len(layer) > 0)
        .expect("a built graph holds at least one point");

    let neighbours_heap = search_layer(graph, data, pivot, ef, layer_to_search, filter);
    let neighbours = neighbours_heap.into_sorted_vec();
    let last = knbn.min(ef).min(neighbours.len());

    let mut hits = Vec::with_capacity(last);
    match filter {
        Some(admits) => {
            for point in &neighbours[..last] {
                let origin_id = graph.origin_id(point.node);
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
                    internal_id: graph.origin_id(point.node),
                    distance: point.dist_to_ref,
                });
            }
        }
    }
    hits
}

/// The bottom-layer traversal: the vendored `search_layer` with patch 4, over
/// node indices.
///
/// Positive distances in the result heap, negated distances in the candidate
/// heap, the entry admitted to the results only where the filter admits it,
/// `INFINITY` standing for the bound while the result heap is empty, the
/// candidate pushed before the filter is consulted, and the result heap trimmed
/// above `ef`. The two vendored assertions are kept, because they are
/// behaviour.
pub(super) fn search_layer<G, F>(
    graph: &G,
    point: &[G::Elem],
    entry: u32,
    ef: usize,
    layer: usize,
    filter: Option<&F>,
) -> BinaryHeap<OrderedNode>
where
    G: Topology + ?Sized,
    F: Fn(&usize) -> bool,
{
    let dist_f = graph.distance();
    let skiplist_size = ef.max(2);
    // Patch 4: one slot is the smallest width that terminates under a filter;
    // see the vendored function.
    let ef = ef.max(1);
    let mut return_points = BinaryHeap::with_capacity(skiplist_size);
    if graph.layer_len(layer) == 0 {
        return return_points;
    }

    let dist_to_entry_point = dist_f.eval(point, graph.vector(entry));
    let mut visited = Visited::new(graph.nb_points());
    visited.test_and_set(entry);

    let mut candidate_points = BinaryHeap::with_capacity(skiplist_size);
    candidate_points.push(OrderedNode {
        dist_to_ref: -dist_to_entry_point,
        node: entry,
    });
    let entry_admitted = match filter {
        None => true,
        Some(admits) => admits(&graph.origin_id(entry)),
    };
    if entry_admitted {
        return_points.push(OrderedNode {
            dist_to_ref: dist_to_entry_point,
            node: entry,
        });
    }

    while let Some(c) = candidate_points.pop() {
        // `!is_nan` where the vendored assertion was `<= 0.`, and the one below
        // where it was `>= 0.`.
        //
        // Both were the same claim written two ways, that the distance function
        // never returns a negative number. `DotDist` does: it returns
        // `1 - dot`, and any inner product above one gives one, which
        // unnormalised input reaches routinely. Nothing in the traversal needs
        // the sign. The candidate heap is ordered by the negated distance and
        // the result heap by the distance, `-(c.dist_to_ref) > f_dist_to_ref`
        // is a comparison between two of them, and none of that reads a sign
        // bit.
        //
        // **What the assertions really catch is a NaN**, because every
        // comparison against a NaN is false, and that is kept exactly. A NaN
        // query still stops here rather than reaching the heap, which is what
        // `an_infinite_query_still_answers_and_a_nan_query_stops` asserts.
        assert!(!c.dist_to_ref.is_nan());
        let f_dist_to_ref = match return_points.peek() {
            Some(f) => {
                assert!(!f.dist_to_ref.is_nan());
                f.dist_to_ref
            }
            None => f32::INFINITY,
        };
        if -(c.dist_to_ref) > f_dist_to_ref {
            return return_points;
        }
        // The vectors this list is about to be scored against, hinted before
        // the first of them is read. See `prefetch_vectors`.
        let neighbours = graph.neighbours(c.node, layer);
        prefetch_vectors(graph, neighbours);
        for &e in neighbours {
            if !visited.test_and_set(e) {
                let f_dist_to_p = match return_points.peek() {
                    Some(f) => f.dist_to_ref,
                    None => f32::INFINITY,
                };
                let e_dist_to_p = dist_f.eval(point, graph.vector(e));
                if e_dist_to_p < f_dist_to_p || return_points.len() < ef {
                    candidate_points.push(OrderedNode {
                        dist_to_ref: -e_dist_to_p,
                        node: e,
                    });
                    let admitted = match filter {
                        None => true,
                        Some(admits) => admits(&graph.origin_id(e)),
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
