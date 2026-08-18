//! A graph a test outside this module can build.
//!
//! The seam is [`super::VectorGraph`], which takes a space by name and so
//! cannot be handed an arbitrary distance. Two test modules need exactly that:
//! `distance` builds one graph per kernel and compares the two, and
//! `hnsw_index::graph_guard_tests` builds one per fixture and holds it to a
//! property of its adjacency. Both did it through the vendored `Hnsw`, which
//! was `pub` and so reachable from anywhere in the crate.
//!
//! [`super::mutable::MutableGraph`] is `pub(super)` and its methods are too,
//! which is deliberate: the index reaches the graph through the seam and
//! nothing else. Rather than widen that for the sake of two test modules, this
//! is the small surface those two actually use, compiled only under
//! `cfg(test)`. It adds nothing to the shipped extension and it leaves the seam
//! exactly as narrow as it was.

use super::levels::LevelGenerator;
use super::mutable::MutableGraph;
use super::traverse::LAYERS;
use super::Distance;

/// One built graph, and the level stream it was built from.
pub(crate) struct TestGraph<T, D> {
    graph: MutableGraph<T, D>,
}

impl<T, D> TestGraph<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    /// Build a graph by inserting every record in order, as `add()` does.
    ///
    /// The level stream starts at the default seed and the insertion is
    /// sequential, so the graph is a function of the data and the parameters
    /// alone.
    pub(crate) fn build(m: usize, ef_construction: usize, data: &[Vec<T>], dist: D) -> Self {
        let dim = data.first().map_or(1, Vec::len);
        let scale = LevelGenerator::default_scale(m);
        let mut levels = LevelGenerator::new(scale, LAYERS);
        let mut graph = MutableGraph::new(dim, m, ef_construction, scale, data.len().max(1), dist)
            .expect("the test parameters are inside the range MutableGraph::new accepts");
        for (id, values) in data.iter().enumerate() {
            graph.insert(values.as_slice(), id, &mut levels);
        }
        TestGraph { graph }
    }

    /// Points in the graph.
    pub(crate) fn nb_points(&self) -> usize {
        self.graph.nb_points()
    }

    /// A page, as (origin id, distance) pairs in the order the traversal
    /// returned them.
    pub(crate) fn page(&self, query: &[T], knbn: usize, ef: usize) -> Vec<(usize, f32)> {
        self.graph
            .search(query, knbn, ef, None::<&fn(&usize) -> bool>)
            .into_iter()
            .map(|hit| (hit.internal_id, hit.distance))
            .collect()
    }

    /// Per layer adjacency, keyed by the id the caller inserted under, each
    /// entry naming its target's origin id.
    pub(crate) fn adjacency(&self) -> Vec<Vec<Vec<usize>>> {
        let n = self.graph.nb_points();
        let mut out = vec![Vec::new(); n];
        for node in 0..n as u32 {
            out[self.graph.origin_id_of(node)] = self
                .graph
                .neighbourhood_ids(node)
                .into_iter()
                .map(|list| list.into_iter().map(|(id, _)| id).collect())
                .collect();
        }
        out
    }

    /// Layer zero adjacency keyed by origin id, each list sorted, which is the
    /// shape the quantized graph guards compare in.
    pub(crate) fn layer_zero_adjacency(&self) -> Vec<(usize, Vec<usize>)> {
        let mut adj: Vec<(usize, Vec<usize>)> = (0..self.graph.nb_points() as u32)
            .map(|node| {
                let mut ids: Vec<usize> = self.graph.neighbourhood_ids(node)[0]
                    .iter()
                    .map(|&(id, _)| id)
                    .collect();
                ids.sort_unstable();
                (self.graph.origin_id_of(node), ids)
            })
            .collect();
        adj.sort_unstable_by_key(|(id, _)| *id);
        adj
    }

    /// The overflow pop counters, as (overflows, saves, fallbacks).
    ///
    /// A fixture that never overflows a layer zero list never reaches the guard
    /// that stops the pop evicting a point's last inbound link, so a test of
    /// that guard has to be able to say its fixture got there.
    pub(crate) fn guard_stats(&self) -> (u64, u64, u64) {
        self.graph.guard_stats()
    }

    /// Lists at layer zero naming each node, counted from the adjacency and
    /// returned by origin id rather than by node index.
    pub(crate) fn layer_zero_in_degree(&self) -> Vec<usize> {
        let counted = self.graph.counted_in_degree(0);
        let mut out = vec![0usize; counted.len()];
        for (node, &count) in counted.iter().enumerate() {
            out[self.graph.origin_id_of(node as u32)] = count as usize;
        }
        out
    }
}
