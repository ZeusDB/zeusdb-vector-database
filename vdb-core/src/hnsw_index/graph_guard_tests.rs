//! Tests that guard the graph the index builds, and the ADC distance.
//!
//! These build a graph directly rather than through an index, because what they
//! assert is a property of the graph rather than of anything around it. Two of
//! them, `self_query_reachability` and `layer_zero_in_degree`, hold the two
//! defects the vendored crate had and ZeusDB patched: reverse links filed at
//! the wrong layer, and an overflow pop that could evict a point's last inbound
//! link. Both patches are behaviour the ZeusDB insert reproduces, so these now
//! guard the shipped builder rather than a reference. The other five assert
//! that a quantized graph is built on the codes at all, which no release before
//! the symmetric distance existed managed.
//!
//! They have their own file because they belong to `DistPQ`, which is used
//! beside them, rather than to any one part of the index.
//!
//! The graph is built through `crate::graph::test_graph`, which is the small
//! `cfg(test)` surface the graph module offers a caller outside it. The seam
//! takes a space by name and so cannot be handed a distance directly.

use crate::distance::CosineDist;
use crate::distance::DistPQ;
use crate::graph::test_graph::TestGraph;
use crate::pq::PQ;
use crate::rng::SeededRng;
use crate::test_vectors::clustered;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
// Scale for the quantized graph tests. Small enough for CI and large
// enough that the neighbour selection heuristic runs, which needs
// `search_layer` to return more than `2 * M` candidates.
//
// Eight bits is the setting the README recommends and it is not
// negotiable here for speed. Six was tried, and on data with this cluster
// structure a 64 centroid codebook is coarse enough that every record in a
// cluster quantizes to the same codes in every subvector. Their distance is
// then genuinely zero, the diversity heuristic ties for real, and roughly
// 45 percent of nodes come out with one neighbour. That is the quantizer
// being too coarse for the data rather than the defect these tests guard,
// but it is indistinguishable from it at the assertion, so the tests run at
// the width real indexes use. k-means over 256 centroids is what makes them
// the slowest in the crate.
const PQ_N: usize = 1200;
const PQ_NQ: usize = 100;
const PQ_DIM: usize = 32;
const PQ_SUBVECTORS: usize = 8;
const PQ_BITS: usize = 8;
const PQ_M: usize = 16;
const PQ_EF_C: usize = 200;
struct PqFixture {
    data: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    pq: Arc<PQ>,
    codes: Vec<Vec<u8>>,
}

/// One trained codebook shared by every quantized graph test, because
/// k-means over 256 centroids is the expensive part and it is the same
/// codebook each time. The graphs themselves are built per test, since
/// that is what is under assertion.
fn fixture() -> &'static PqFixture {
    static FIXTURE: OnceLock<PqFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let all = clustered(PQ_N + PQ_NQ, PQ_DIM, 42);
        let data: Vec<Vec<f32>> = all[..PQ_N].to_vec();
        let queries: Vec<Vec<f32>> = all[PQ_N..].to_vec();

        let pq = Arc::new(PQ::new(PQ_DIM, PQ_SUBVECTORS, PQ_BITS, 1000, None));
        pq.train(&data).expect("pq training");
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let codes = pq.quantize_batch(&refs).expect("quantization");

        PqFixture {
            data,
            queries,
            pq,
            codes,
        }
    })
}

/// Uniform vectors, unit normalised, which is what the cosine path holds.
///
/// The spread is deliberately structureless, unlike [`clustered`], because the
/// two raw guards below want a graph whose neighbour lists fill rather than one
/// whose clusters let the diversity heuristic prune.
fn unit_vectors(records: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = SeededRng::seed_from_u64(42);
    (0..records)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect()
}

/// Build the quantized graph exactly as the insertion path does, one code
/// vector at a time with no query table set.
fn build_pq_graph(pq: Arc<PQ>, codes: &[Vec<u8>]) -> TestGraph<u8, DistPQ> {
    TestGraph::build(PQ_M, PQ_EF_C, codes, DistPQ::new(pq))
}

/// The strongest assertion available about the quantized graph: that the
/// data it is built from has any effect on it at all.
///
/// Build the same graph twice, in the same insertion order so the level
/// sequence is identical, once with each id holding its own codes and once
/// with every id holding a different record's codes. A graph built on real
/// distances comes out different. A graph built on a constant comes out
/// byte for byte identical, which is what `DistPQ::eval` produced for every
/// release that shipped quantization: it returned infinity whenever no
/// query table was set, and no insertion path sets one.
///
/// Measured on the shipped v0.4.1 code at 10,000 records, layer zero
/// adjacency was identical for 10,000 of 10,000 nodes.
#[test]
fn quantized_graph_depends_on_the_data() {
    let f = fixture();

    let own = build_pq_graph(f.pq.clone(), &f.codes).layer_zero_adjacency();

    let mut shuffled = f.codes.clone();
    shuffled.reverse();
    let other = build_pq_graph(f.pq.clone(), &shuffled).layer_zero_adjacency();

    assert_eq!(own.len(), PQ_N);
    assert_eq!(other.len(), PQ_N);

    let identical = own
        .iter()
        .zip(other.iter())
        .filter(|((id_a, a), (id_b, b))| id_a == id_b && a == b)
        .count();

    assert!(
        identical * 20 < PQ_N,
        "layer zero adjacency is identical for {} of {} nodes when every id is given a \
         different record's codes, so the graph is not being built on the codes. \
         DistPQ::eval is returning a constant on the insertion path.",
        identical,
        PQ_N
    );
}

/// A graph whose distances all tie leaves every node with one neighbour,
/// because the diversity heuristic in `select_neighbours` rejects a
/// candidate that is at least as close to an already chosen neighbour as it
/// is to the new point, and under a total tie that is every candidate after
/// the first. Measured on the shipped code, layer zero out-degree was
/// exactly one for 99.64 percent of nodes and a traversal reached 33 of
/// 10,000.
#[test]
fn quantized_graph_layer_zero_out_degree() {
    let f = fixture();
    let adj = build_pq_graph(f.pq.clone(), &f.codes).layer_zero_adjacency();

    let degenerate = adj.iter().filter(|(_, n)| n.len() <= 1).count();
    assert!(
        degenerate * 100 < PQ_N,
        "{} of {} nodes have layer zero out-degree of one or less; the quantized graph \
         has collapsed to a star",
        degenerate,
        PQ_N
    );

    let total: usize = adj.iter().map(|(_, n)| n.len()).sum();
    let mean = total as f64 / PQ_N as f64;
    assert!(
        mean > (PQ_M as f64) / 2.0,
        "mean layer zero out-degree is {:.2}, expected well above {} for m = {}",
        mean,
        PQ_M / 2,
        PQ_M
    );
}

/// Quantized search has to find the right answers, not merely return the
/// right number of them. Measured on the shipped code at this scale the
/// graph reached 33 nodes of 1,200 and recall was under one percent, so
/// the threshold below fails against the old behaviour by a wide margin.
///
/// The ceiling is the quantizer rather than the graph, so this asserts a
/// floor and not equality with the raw path.
#[test]
fn quantized_graph_recall_against_brute_force() {
    const K: usize = 10;

    let f = fixture();
    let (data, queries) = (&f.data, &f.queries);
    let dist = DistPQ::new(f.pq.clone());
    let graph = build_pq_graph(f.pq.clone(), &f.codes);

    let mut hits = 0usize;
    let mut returned = usize::MAX;
    for q in queries.iter() {
        let found = {
            let _lut = dist.install_query_lut(q).expect("query lut");
            let dummy = vec![0u8; PQ_SUBVECTORS];
            graph.page(&dummy, K, 100)
        };
        returned = returned.min(found.len());

        let mut truth: Vec<(f32, usize)> = data
            .iter()
            .enumerate()
            .map(|(j, v)| {
                (
                    v.iter().zip(q.iter()).map(|(x, y)| (x - y) * (x - y)).sum(),
                    j,
                )
            })
            .collect();
        truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let truth: HashSet<usize> = truth[..K].iter().map(|x| x.1).collect();

        hits += found.iter().filter(|(id, _)| truth.contains(id)).count();
    }

    assert_eq!(returned, K, "a top {} request came back short", K);

    let recall = hits as f64 / (K * PQ_NQ) as f64;
    assert!(
        recall > 0.30,
        "quantized recall at top {} is {:.4}, which is far below what these codes support",
        K,
        recall
    );
}

/// The whole graph has to be reachable, which is the property the shipped
/// code lost most visibly. Asking for more results than the graph can reach
/// is what exposed it: a request for 1,000 came back with 34.
#[test]
fn quantized_graph_is_fully_reachable() {
    let f = fixture();
    let dist = DistPQ::new(f.pq.clone());
    let graph = build_pq_graph(f.pq.clone(), &f.codes);

    let found = {
        let _lut = dist.install_query_lut(&f.data[0]).expect("query lut");
        let dummy = vec![0u8; PQ_SUBVECTORS];
        graph.page(&dummy, PQ_N, PQ_N)
    };

    assert_eq!(
        found.len(),
        PQ_N,
        "a request for all {} records returned {}, so the traversal cannot reach the \
         whole graph",
        PQ_N,
        found.len()
    );
}

/// The symmetric distance must not leak into search. With a query table set
/// the distance has to be the asymmetric one, byte for byte as before, so
/// a quantized graph and a raw graph rank a query's own record the same way.
#[test]
fn quantized_search_still_uses_the_query_table() {
    let f = fixture();
    let dist = DistPQ::new(f.pq.clone());
    let graph = build_pq_graph(f.pq.clone(), &f.codes);

    // The ADC distance from a query to a stored code, computed directly.
    let query = &f.data[7];
    let lut = f.pq.compute_adc_lut(query).expect("adc lut");
    let expected: f32 = f.codes[7]
        .iter()
        .enumerate()
        .map(|(sv, &c)| lut[sv][c as usize])
        .sum();

    let found = {
        let _lut = dist.install_query_lut(query).expect("query lut");
        let dummy = vec![0u8; PQ_SUBVECTORS];
        graph.page(&dummy, 10, 200)
    };

    assert!(
        found.iter().any(|&(id, _)| id == 7),
        "a record could not find itself"
    );

    // Every distance the search reports must be the asymmetric one. The
    // symmetric table would give a different number here, since it compares
    // the dummy query's all-zero codes rather than the query itself.
    for &(id, distance) in found.iter() {
        let adc: f32 = f.codes[id]
            .iter()
            .enumerate()
            .map(|(sv, &c)| lut[sv][c as usize])
            .sum();
        assert!(
            (distance - adc).abs() <= 1e-4 * adc.max(1.0),
            "search reported {} for record {} where its asymmetric distance is {}, \
             so the query path is no longer using the query table",
            distance,
            id,
            adc
        );
    }

    // The record's own ADC distance is the smallest of the ten returned,
    // which is what the ranking has to produce.
    assert!(
        found[0].1 <= expected + 1e-4,
        "the top hit scored {} against the query's own record at {}",
        found[0].1,
        expected
    );
}

/// Holds the reverse link filing the graph depends on.
///
/// The crate ZeusDB was built on filed a reverse link at the inserting point's
/// own top layer instead of at the layer being processed. Points assigned a
/// level above zero then lost their layer-zero inbound adjacency and could
/// become unreachable to similarity search, and at this index size roughly one
/// to two percent of self-queries failed. ZeusDB patched that, and the insert
/// that replaced it files reverse links the corrected way, so this now holds
/// the shipped builder rather than a reference.
///
/// The distance is ZeusDB's `CosineDist` and the data is unit normalised, which
/// is what `process_vector_for_space` hands every cosine insertion and every
/// cosine query. `CosineDist` is `1 - dot` and carries normalisation as a
/// precondition rather than applying it, so unnormalised data would make the
/// nearest point the longest one rather than the closest and no point would
/// find itself. Insertion is sequential, which is the path every index takes
/// through `add()`.
#[test]
fn self_query_reachability() {
    const N: usize = 3000;
    const DIM: usize = 32;

    let data = unit_vectors(N, DIM);
    let graph = TestGraph::build(16, 200, &data, CosineDist {});
    assert_eq!(graph.nb_points(), N);

    let failures: Vec<usize> = (0..N)
        .filter(|&i| graph.page(&data[i], 1, 64).first().map(|&(id, _)| id) != Some(i))
        .collect();

    assert!(
        failures.is_empty(),
        "{} of {} points cannot find themselves by self-query (first: {:?}); \
         reverse links are no longer filed at the layer being processed",
        failures.len(),
        N,
        &failures[..failures.len().min(10)]
    );
}

/// Holds the guard on the layer-zero overflow pop.
///
/// The crate ZeusDB was built on always discarded the farthest entry, so a
/// point whose only inbound link happened to be that entry was left with no
/// layer-zero in-edge and became an orphan no search could reach through the
/// graph. The unpatched crate stranded 24 of these 5,000 points. ZeusDB
/// patched it and the insert that replaced it carries the same guard, so this
/// now holds the shipped builder rather than a reference.
///
/// In-degree is counted from the adjacency lists rather than from the guard's
/// own counters, so the assertion holds against the graph itself and stays
/// meaningful if the counters are ever wrong.
///
/// `M` is 4 rather than the shipped 16 on purpose. The layer-zero
/// neighbour cap is `2 * M`, so a small `M` fills lists early and makes
/// the overflow pop frequent, which is the only site that can strand a
/// point. Whether the shipped `M` of 16 strands points depends on the
/// data model rather than on index size. On uniform vectors like these
/// it strands none up to 30,000 points, while on clustered data, 50
/// Gaussian clusters at sigma 0.15 in 768 dimensions, it strands 6 of
/// 10,000. A small `M` lets this uniform generator fail fast instead.
#[test]
fn layer_zero_in_degree() {
    const N: usize = 5000;
    const DIM: usize = 128;
    const M: usize = 4;

    let data = unit_vectors(N, DIM);
    let graph = TestGraph::build(M, 200, &data, CosineDist {});
    assert_eq!(graph.nb_points(), N);

    // The fixture has to reach the guard, or this asserts nothing. A small `M`
    // fills lists early, so the pop fires far more often than once per node and
    // the guard skips the farthest entry thousands of times.
    let (overflows, saves, fallbacks) = graph.guard_stats();
    println!(
        "overflow pop over {} nodes: {} events, {} saves, {} fallbacks",
        N, overflows, saves, fallbacks
    );
    assert!(
        overflows > N as u64,
        "the overflow pop fired {} times over {} nodes, which is not often enough          to exercise the guard",
        overflows,
        N
    );
    assert!(
        saves > 0,
        "the guard never skipped the farthest entry, so it changed no outcome          here and this fixture does not test it"
    );

    let in_degree = graph.layer_zero_in_degree();
    assert_eq!(in_degree.len(), N);
    let orphans: Vec<usize> = (0..N).filter(|&i| in_degree[i] == 0).collect();

    assert!(
        orphans.is_empty(),
        "{} of {} points have zero layer-zero in-degree (first: {:?}); \
         the overflow pop guard is no longer saving a point's last inbound link",
        orphans.len(),
        N,
        &orphans[..orphans.len().min(10)]
    );
}

/// The `ef_construction` at which the neighbour selection heuristic stops
/// running, which `VectorDatabase._warn_if_selection_disabled` warns at.
///
/// `select_neighbours` opens with `if candidates.len() <= nb_neighbours_asked`
/// and, with `extend_candidates` false, copies every candidate into the
/// neighbour list and returns from inside that branch. The budget is
/// `2 * max_nb_connection` at layer zero. The candidate list is exactly
/// `ef_construction` long once the graph holds more points than that, because
/// `search_layer` pushes every point it visits into the result heap as well as
/// the candidate heap and trims the result heap only above `ef`.
///
/// So at `ef_construction == 2 * m` every node takes exactly `2 * m`
/// neighbours and sits at the degree cap, and one above it the heuristic runs
/// and prunes. The gap is what makes the threshold exact rather than
/// approximate, and it is why the warning carries no margin.
#[test]
fn neighbour_selection_threshold_is_twice_m() {
    const N: usize = 2000;
    const DIM: usize = 16;
    const M: usize = 8;

    let data = clustered(N, DIM, 7);

    let at_cap_fraction = |efc: usize| -> (f64, f64) {
        let graph = TestGraph::build(M, efc, &data, CosineDist {});
        let degrees: Vec<usize> = graph
            .layer_zero_adjacency()
            .into_iter()
            .map(|(_, list)| list.len())
            .collect();
        assert_eq!(degrees.len(), N);
        let at_cap = degrees.iter().filter(|d| **d == 2 * M).count();
        let mean = degrees.iter().sum::<usize>() as f64 / N as f64;
        (at_cap as f64 / N as f64, mean)
    };

    let (off_at_cap, off_mean) = at_cap_fraction(2 * M);
    assert!(
        off_at_cap > 0.99,
        "at ef_construction = 2*m = {} only {:.1}% of nodes reach the degree cap and the \
         mean out-degree is {:.3}; the heuristic is running where the source says it is \
         skipped",
        2 * M,
        off_at_cap * 100.0,
        off_mean
    );

    let (on_at_cap, on_mean) = at_cap_fraction(2 * M + 1);
    assert!(
        on_at_cap < 0.5,
        "at ef_construction = 2*m + 1 = {} {:.1}% of nodes still reach the degree cap and \
         the mean out-degree is {:.3}; the heuristic is not pruning, so the threshold is \
         not 2*m",
        2 * M + 1,
        on_at_cap * 100.0,
        on_mean
    );
}
