//! Insertion: the vendored `Hnsw::insert_slice` written out over the slabs.
//!
//! This is the algorithm that decides what the graph is. Two builds that agree
//! on the distribution of edges but not on the edges are two different graphs,
//! and every reproducibility guarantee in the suite is a statement about the
//! edges. So this is not a reimplementation of the idea. It is
//! the vendored `Hnsw::insert_data` written out again, calling the same
//! traversal at the same widths in the same order, and it is proved by building
//! both graphs from the same data and comparing them edge for edge rather than
//! by comparing what they return.
//!
//! # The two phases
//!
//! [`MutableGraph::plan`] takes `&self` and computes: it draws nothing, reads
//! the entry point, descends, runs `search_layer` at `ef_construction` per layer
//! and selects the neighbours. [`MutableGraph::install`] takes `&mut self` and
//! writes: it appends the node, installs its lists, runs the reverse link update
//! with the guarded overflow pop, and checks the entry point.
//!
//! [`Insertion`] is what crosses between them and it owns everything it holds.
//! Nothing in it borrows the graph, so the read guard phase one runs under is
//! dropped before phase two takes the write guard. The split is sound because
//! the index's `writers` mutex already serialises mutators, so nothing can
//! change the graph in the gap, and a search running there does not mutate.
//!
//! The level is drawn outside both phases, once per insertion, before either
//! runs. That is where the vendored `generate_new_point` draws it, and the
//! order of the draws is the whole of what makes two builds of the same data
//! the same graph.
//!
//! # What the vendored insert does that is not obvious from reading it
//!
//! `generate_new_point` files the new point in `points_by_layer[level]` before
//! any search runs. `search_layer` returns an empty heap when the layer it is
//! asked about is empty, and that check reads the same table, so at `l == level`
//! it reads the layer as occupied even when the new point is the only thing in
//! it. That is the only way the registration is visible to the traversal, and
//! [`Pending`] is what reproduces it.
//!
//! The check is over points whose level is exactly `l`, not over points
//! carrying adjacency at `l`, so at a layer below the new point's level that no
//! point has been drawn at, the vendored insert installs no list at all even
//! though the traversal could have found neighbours there. That is reproduced by
//! running the same check rather than by special casing it.
//!
//! # The four patches
//!
//! Patch 1 files a reverse link at the layer being processed. Here the layer is
//! the loop variable and there is no second candidate value in scope, so it is
//! right by construction. `insertion_reproduces_the_guard_tests` is what holds
//! it, over the same fixture the vendored guard test uses.
//!
//! Patch 2 seeds the level stream, which is [`super::super::levels`].
//!
//! Patch 3 guards the overflow pop. The transient state it works in is
//! representable by design, at `2 * m + 1` and `m + 1`, so the push then
//! evaluate then evict sequence needs no special case, and the counters are
//! plain integers under the write lock where the vendored ones are atomics
//! carrying a recorded race.
//!
//! Patch 5 corrected a per layer `Vec<Arc>` reservation. There is no such thing
//! here to mis-size; see [`MutableGraph::new`].

use super::{by_distance, Entry, MutableGraph};
use crate::graph::levels::LevelGenerator;
use crate::graph::store::VectorStore;
use crate::graph::traverse::{self, OrderedNode, Topology};
use crate::graph::Distance;
use std::collections::BinaryHeap;

/// The filter the insertion traversal runs under, which is none of one.
///
/// `search_layer` is generic over the predicate so that the shipped search
/// monomorphises the caller's closure into it. Insertion admits everything, so
/// it needs a concrete type to name for the `None`.
type NoFilter = fn(&usize) -> bool;

/// Admitting everything, which is what every insertion traversal does.
const NO_FILTER: Option<&NoFilter> = None;

/// What phase one computed, owning everything it holds.
///
/// Nothing here borrows the graph. That is the property that lets the read
/// guard drop between the phases, and it is why the lists are node indices and
/// distances rather than slices into the arenas.
pub(crate) struct Insertion {
    /// The level drawn for this point, before either phase ran.
    level: usize,
    /// Nodes the graph held when the plan was made.
    ///
    /// Every node index the plan names was handed out against that arena, so a
    /// plan installed into a graph holding a different number of nodes names
    /// something other than what it chose. Nothing can put the graph in that
    /// state, because the index's `writers` mutex holds the two phases together;
    /// this is what says so at the moment it matters rather than in a comment.
    nb_points: usize,
    /// The descent residue, as (layer, target, distance), at layers above
    /// `level`. See the module documentation of [`super`] for why it is kept.
    residue: Vec<(u8, u32, f32)>,
    /// The chosen neighbours, one list per layer from `level` down to zero, in
    /// the order the install writes them. A layer whose search came back empty
    /// contributes no entry, exactly as the vendored insert installs nothing
    /// there.
    lists: Vec<(usize, Vec<Entry>)>,
}

/// What phase one decided, whichever of the two shapes an insertion takes.
///
/// The first insertion into an empty graph has no phase one at all, because the
/// vendored insert reads its entry point, finds `None` and returns before it
/// descends. Which of the two an insertion is is therefore decided under the
/// read guard along with everything else, and phase two carries it out.
pub(crate) enum Planned {
    /// The graph held no point, so the insertion files a node and takes no
    /// edges.
    First { level: usize },
    /// The descent ran and chose the lists to install.
    Descended(Insertion),
}

/// The graph as the vendored insert sees it partway through its own insertion.
///
/// One difference from the graph itself and no more: the layer the new point
/// was drawn at counts one extra point, because `generate_new_point` filed it
/// there before any search ran. Nothing else about that registration is visible
/// to a traversal. No list names the new point until the reverse update, which
/// runs after every search this insertion makes, and the point the traversal
/// starts from is passed in rather than looked up.
struct Pending<'a, T, D> {
    graph: &'a MutableGraph<T, D>,
    store: &'a VectorStore<T>,
    level: usize,
}

impl<T, D> Topology for Pending<'_, T, D>
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
        // The new point is not reachable, so no traversal can name it and the
        // visited set does not need a bit for it.
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
        self.graph.layer_len(layer) + usize::from(layer == self.level)
    }

    #[inline]
    fn vector(&self, node: u32) -> &[T] {
        self.store.get(node)
    }

    #[inline]
    fn origin_id(&self, node: u32) -> usize {
        self.graph.origin_id_of(node)
    }

    #[inline]
    fn neighbours(&self, node: u32, layer: usize) -> &[u32] {
        self.graph.neighbours_at(node, layer)
    }
}

impl<T, D> MutableGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// Insert one point, drawing its level from `levels`.
    ///
    /// The whole insertion, both phases and the draw, for a caller holding the
    /// graph outright. The cutover splits this at the two phase boundary and
    /// puts the index locks around each; until then this is what the tests
    /// build with and what the measurement times.
    pub(in crate::graph) fn insert(
        &mut self,
        store: &mut VectorStore<T>,
        data: &[T],
        origin_id: usize,
        levels: &mut LevelGenerator,
    ) {
        let level = levels.generate();
        let planned = self.plan_insertion(store, data, level);
        self.install_insertion(store, data, origin_id, planned);
    }

    /// Phase one, whichever shape the insertion takes.
    ///
    /// The empty case is settled here rather than by the caller, because
    /// whether the graph holds a point is itself something read under the read
    /// guard, and the answer cannot change before phase two runs.
    pub(in crate::graph) fn plan_insertion(
        &self,
        store: &VectorStore<T>,
        data: &[T],
        level: usize,
    ) -> Planned {
        if self.nb_points() == 0 {
            Planned::First { level }
        } else {
            Planned::Descended(self.plan(store, data, level))
        }
    }

    /// Phase two, whichever shape the insertion takes.
    pub(in crate::graph) fn install_insertion(
        &mut self,
        store: &mut VectorStore<T>,
        data: &[T],
        origin_id: usize,
        planned: Planned,
    ) {
        match planned {
            Planned::First { level } => self.insert_first(store, data, origin_id, level),
            Planned::Descended(plan) => self.install(store, data, origin_id, plan),
        }
    }

    /// The first insertion into an empty graph, which has no phase one.
    ///
    /// The vendored insert reads its entry point, finds `None`, calls
    /// `check_entry_point` and returns before it descends. The point is filed
    /// and takes no edges, because there is nothing to link it to.
    ///
    /// The vendored `point_rank == 1` early return above that is unreachable.
    /// It fires only where the entry point is already set as the first point
    /// arrives, and the two constructors that set one both start from a
    /// non-empty topology.
    pub(in crate::graph) fn insert_first(
        &mut self,
        store: &mut VectorStore<T>,
        data: &[T],
        origin_id: usize,
        level: usize,
    ) {
        assert_eq!(
            self.nb_points(),
            0,
            "the first insertion is into an empty graph"
        );
        let node = self.append_node(store, data, origin_id, level, &[]);
        self.set_entry(node, level);
    }

    /// Phase one. Everything the insertion decides, under nothing but a read.
    ///
    /// The graph must hold at least one point, which [`Self::insert`] settles
    /// before it gets here. The vendored equivalent settles it by reading
    /// `entry_point` and taking the `None` arm.
    pub(in crate::graph) fn plan(
        &self,
        store: &VectorStore<T>,
        data: &[T],
        level: usize,
    ) -> Insertion {
        debug_assert!(
            self.nb_points() > 0,
            "phase one descends from an entry point, so the graph holds one"
        );
        let view = Pending {
            graph: self,
            store,
            level,
        };
        let dist_f = self.distance();
        let mut pivot = self.entry();
        let mut dist_to_entry = dist_f.eval(data, store.get(pivot));
        let max_level_observed = self.entry_level() as usize;

        // The descent. From the entry level down to the point's own level plus
        // one, at width one, taking the first strict improvement.
        let mut residue: Vec<(u8, u32, f32)> = Vec::new();
        for l in ((level + 1)..=max_level_observed).rev() {
            let mut sorted_points = traverse::search_layer(&view, data, pivot, 1, l, NO_FILTER);
            // The vendored panic, kept because it is behaviour. A width of one
            // trims the result heap to one entry, so it cannot fire.
            assert!(
                sorted_points.len() <= 1,
                "in insert : search_layer layer {:?}, returned {:?} points ",
                l,
                sorted_points.len()
            );
            if let Some(ep) = sorted_points.pop() {
                // Install site 1, the descent residue. The bound is the
                // vendored one and it never binds: the descent visits each layer
                // once and nothing else writes here, so the count is zero every
                // time.
                let held = residue
                    .iter()
                    .filter(|&&(layer, _, _)| layer as usize == l)
                    .count();
                if held < self.m() {
                    residue.push((l as u8, ep.node, ep.dist_to_ref));
                }
                if ep.dist_to_ref < dist_to_entry {
                    pivot = ep.node;
                    dist_to_entry = ep.dist_to_ref;
                }
            }
        }

        // The neighbour selection. From the point's own level down to zero, at
        // `ef_construction`.
        let mut lists: Vec<(usize, Vec<Entry>)> = Vec::with_capacity(level + 1);
        let mut neighbours: Vec<Entry> = Vec::new();
        for l in (0..=level).rev() {
            let sorted_points =
                traverse::search_layer(&view, data, pivot, self.ef_construction(), l, NO_FILTER);
            if sorted_points.is_empty() {
                continue;
            }
            let mut candidates = negated(&sorted_points);
            let nb_conn = if l == 0 { 2 * self.m() } else { self.m() };
            neighbours.reserve(nb_conn);
            self.select_neighbours(store, &mut candidates, nb_conn, &mut neighbours);
            neighbours.sort_by(by_distance);
            // The nearest chosen neighbour carries the descent into the next
            // layer down. The vendored insert reads it after installing the
            // list, which reads the same entry.
            if let Some(nearest) = neighbours.first() {
                pivot = nearest.target;
            }
            lists.push((l, std::mem::take(&mut neighbours)));
        }

        Insertion {
            level,
            nb_points: self.nb_points(),
            residue,
            lists,
        }
    }

    /// Navarro's rule, as the vendored `select_neighbours` implements it.
    ///
    /// Pop candidates nearest first and keep one only if no already kept
    /// neighbour is at least as close to it as it is to the query.
    ///
    /// The early transfer branch above is live behaviour rather than a corner.
    /// With `extend_candidates` false, a candidate list no longer than the
    /// budget is copied out whole and the heuristic never runs, so at
    /// `ef_construction` at or below `2 * m` every node sits at the degree cap.
    /// `VectorDatabase._warn_if_selection_disabled` warns at exactly that
    /// threshold and `neighbour_selection_threshold_is_twice_m` measures it.
    ///
    /// Two vendored parameters have no counterpart. `extend_candidates` and
    /// `keep_pruned` are both false at every ZeusDB construction site: `new` and
    /// `from_loaded_points` both set them false and nothing in the crate calls
    /// either setter. The branches they guard are not ported, rather than ported
    /// behind a flag nothing can set.
    ///
    /// The query is not a parameter for the same reason. The vendored signature
    /// takes it, and the only line that reads it is inside the
    /// `extend_candidates` branch. Every distance this rule evaluates is
    /// between two stored points, and the distances to the query arrive already
    /// computed on the candidate heap.
    fn select_neighbours(
        &self,
        store: &VectorStore<T>,
        candidates: &mut BinaryHeap<OrderedNode>,
        nb_neighbours_asked: usize,
        neighbours_vec: &mut Vec<Entry>,
    ) {
        let dist_f = self.distance();
        neighbours_vec.clear();

        if candidates.len() <= nb_neighbours_asked {
            while let Some(p) = candidates.pop() {
                assert!(-p.dist_to_ref >= 0.);
                neighbours_vec.push(Entry {
                    dist: -p.dist_to_ref,
                    target: p.node,
                });
            }
            return;
        }

        while !candidates.is_empty() && neighbours_vec.len() < nb_neighbours_asked {
            if let Some(e_p) = candidates.pop() {
                let mut e_to_insert = true;
                let e_point_v = store.get(e_p.node);
                assert!(e_p.dist_to_ref <= 0.);
                if !neighbours_vec.is_empty() {
                    e_to_insert = !neighbours_vec
                        .iter()
                        .any(|d| dist_f.eval(e_point_v, store.get(d.target)) <= -e_p.dist_to_ref);
                }
                if e_to_insert {
                    neighbours_vec.push(Entry {
                        dist: -e_p.dist_to_ref,
                        target: e_p.node,
                    });
                }
            }
        }
    }

    /// Phase two. Everything the insertion writes, under the write guard.
    pub(in crate::graph) fn install(
        &mut self,
        store: &mut VectorStore<T>,
        data: &[T],
        origin_id: usize,
        plan: Insertion,
    ) {
        let Insertion {
            level,
            nb_points,
            residue,
            lists,
        } = plan;

        // Every node index the plan holds was handed out against the arena
        // phase one read, so a graph that has taken or lost a node since is a
        // graph this plan does not describe. See [`Insertion::nb_points`].
        assert_eq!(
            self.nb_points(),
            nb_points,
            "the graph held {} nodes when this insertion was planned and holds {} now",
            nb_points,
            self.nb_points()
        );

        let node = self.append_node(store, data, origin_id, level, &residue);

        // Install site 1's bookkeeping. The edges themselves went in with the
        // node, because the residue region is append only and this is the one
        // moment its owner is at the end of it. The counters are moved here
        // rather than during the descent because the descent runs under a read
        // guard and nothing reads them until the reverse update below.
        for &(layer, target, _) in &residue {
            self.bump_in_degree(target, layer as usize, 1);
        }

        // Install site 2. The vendored line is a `clone_from` over whatever the
        // list already held, and it decrements the inbound counter of every
        // target it discards. On a sequential insert there is nothing to
        // discard: the layers this loop touches are at or below the new point's
        // level, install site 1 writes only above it, and the reverse update has
        // not run yet. The vendored decrement loop exists for the parallel path,
        // where another thread's reverse update can push into this list while
        // the selection is still running. The assertion is what holds that.
        for (layer, list) in &lists {
            assert_eq!(
                self.list_len(node, *layer),
                0,
                "the list of a new node at layer {} already holds {} entries",
                layer,
                self.list_len(node, *layer)
            );
            for entry in list {
                self.bump_in_degree(entry.target, *layer, 1);
            }
            self.write_list(node, *layer, list);
        }

        // Install site 3, the reverse link update, with the guarded overflow
        // pop. Patch 1 is the `layer` here: the link is filed at the layer being
        // processed and not at the new point's own top level.
        //
        // The vendored loop holds a read guard on the new point's own list for
        // its whole length, which is sound because the one writer that could
        // touch it is the branch this loop excludes. Copying the list out is the
        // same thing without the guard, and the reason it is the same is that
        // nothing this loop does writes back into it.
        let mut own: Vec<Entry> = Vec::new();
        let mut scratch: Vec<Entry> = Vec::new();
        for layer in (0..=level).rev() {
            self.copy_list(node, layer, &mut own);
            for entry in &own {
                let (target, dist) = (entry.target, entry.dist);
                // A point is never its own neighbour on this path, since the
                // traversal cannot reach a node nothing links to. The vendored
                // guard is against deadlocking on its own lock.
                if target == node {
                    continue;
                }
                // Nor can the new point already be in a neighbour's list, for
                // the same reason. The vendored check is against another thread
                // having pushed it.
                if self.list_names(target, layer, node) {
                    continue;
                }
                self.push_edge(target, layer, node, dist);
                self.bump_in_degree(node, layer, 1);

                let nbn_at_l = self.list_len(target, layer);
                let threshold_shrinking = if layer > 0 { self.m() } else { 2 * self.m() };
                let shrink = nbn_at_l > threshold_shrinking;

                self.sort_list(target, layer, &mut scratch);
                if shrink {
                    self.guarded_pop(target, layer);
                }
            }
        }

        // The entry point check, which is the vendored `check_entry_point`. The
        // comparison is strict, so the first node drawn at a level keeps the
        // entry against every later node drawn at the same one.
        if level as u8 > self.entry_level() {
            self.set_entry(node, level);
        }
    }

    /// Patch 3. The single edge removal site.
    ///
    /// The list is sorted ascending by distance to its owner, so the farthest
    /// entry is last. Walk from the farthest inward and remove the first entry
    /// whose target would still hold at least one inbound link at this layer
    /// afterwards. If no candidate qualifies, remove the farthest, which is the
    /// unmodified crate behaviour.
    ///
    /// Reported by `layer_zero_in_degree`, which counts the orphans a build
    /// leaves. The unmodified crate strands 24 of 5,000 points on that fixture.
    fn guarded_pop(&mut self, node: u32, layer: usize) {
        let len = self.list_len(node, layer);
        let last = len - 1;
        let mut victim = None;
        for slot in (0..len).rev() {
            if self.in_degree(self.target_at(node, layer, slot), layer) >= 2 {
                victim = Some(slot);
                break;
            }
        }
        self.note_overflow(victim, last);
        let slot = victim.unwrap_or(last);
        let removed = self.remove_edge_at(node, layer, slot);
        self.bump_in_degree(removed, layer, -1);
    }
}

/// The vendored `from_positive_binaryheap_to_negative_binary_heap`.
///
/// The candidate heap `select_neighbours` pops from is built by walking the
/// result heap's own backing array and pushing each entry negated. The walk
/// order is the array order rather than the sorted order, so the heap that
/// comes out is a function of the heap that went in and not only of its
/// contents. Reproducing the walk is what makes the two agree on which of two
/// entries at an equal distance is popped first.
fn negated(positive_heap: &BinaryHeap<OrderedNode>) -> BinaryHeap<OrderedNode> {
    let mut negative_heap = BinaryHeap::with_capacity(positive_heap.len());
    for p in positive_heap.iter() {
        assert!(p.dist_to_ref >= 0.);
        negative_heap.push(OrderedNode {
            dist_to_ref: -p.dist_to_ref,
            node: p.node,
        });
    }
    negative_heap
}
