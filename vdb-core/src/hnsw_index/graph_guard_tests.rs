//! Tests that guard the vendored `hnsw_rs` patches and the ADC distance.
//!
//! These build a graph directly rather than through an index, because what they
//! assert is a property of the graph the vendored crate wires. Two of them,
//! `self_query_reachability` and `layer_zero_in_degree`, fail if a patch
//! recorded in `vendor/hnsw_rs/ZEUSDB-PATCH.md` is ever lost, most likely during
//! an upgrade. The other five assert that a quantized graph is built on the
//! codes at all, which no release before the symmetric distance existed managed.
//!
//! They have their own file because they belong to `DistPQ` and to the vendored
//! crate rather than to any one part of the index, and because `graph.rs` is a
//! seam whose job is naming `hnsw_rs`, not holding index tests.

use super::DistPQ;
use crate::pq::PQ;
use crate::test_vectors::clustered;
// `DistCosine` is the `anndists` implementation this crate's distances
// replaced. The two graph guard tests keep it on purpose. They guard patches in
// the vendored crate rather than anything about the distance, their data is
// deliberately unnormalised, and holding the distance fixed keeps the orphan
// counts their comments record comparable across relays.
use hnsw_rs::prelude::{DistCosine, Hnsw};
use rand::rngs::StdRng;
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

/// Build the quantized graph exactly as `insert_pq_codes` does, one code
/// vector at a time with no query table set.
fn build_pq_graph(pq: Arc<PQ>, codes: &[Vec<u8>]) -> Hnsw<'static, u8, DistPQ> {
    let hnsw = Hnsw::new(PQ_M, codes.len(), 16, PQ_EF_C, DistPQ::new(pq));
    for (i, c) in codes.iter().enumerate() {
        hnsw.insert((c.as_slice(), i));
    }
    hnsw
}

/// Layer zero adjacency keyed by origin id, each list sorted.
fn layer_zero_adjacency(hnsw: &Hnsw<'static, u8, DistPQ>) -> Vec<(usize, Vec<usize>)> {
    let mut adj: Vec<(usize, Vec<usize>)> = hnsw
        .get_point_indexation()
        .into_iter()
        .map(|p| {
            let mut v: Vec<usize> = p.get_neighborhood_id()[0].iter().map(|x| x.d_id).collect();
            v.sort_unstable();
            (p.get_origin_id(), v)
        })
        .collect();
    adj.sort_unstable_by_key(|(id, _)| *id);
    adj
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

    let own = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &f.codes));

    let mut shuffled = f.codes.clone();
    shuffled.reverse();
    let other = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &shuffled));

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
    let adj = layer_zero_adjacency(&build_pq_graph(f.pq.clone(), &f.codes));

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
    let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

    let mut hits = 0usize;
    let mut returned = usize::MAX;
    for q in queries.iter() {
        let found = {
            let _lut = hnsw.get_distance().install_query_lut(q).expect("query lut");
            let dummy = vec![0u8; PQ_SUBVECTORS];
            hnsw.search(&dummy, K, 100)
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

        hits += found.iter().filter(|n| truth.contains(&n.d_id)).count();
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
    let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

    let found = {
        let _lut = hnsw
            .get_distance()
            .install_query_lut(&f.data[0])
            .expect("query lut");
        let dummy = vec![0u8; PQ_SUBVECTORS];
        hnsw.search(&dummy, PQ_N, PQ_N)
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
    let hnsw = build_pq_graph(f.pq.clone(), &f.codes);

    // The ADC distance from a query to a stored code, computed directly.
    let query = &f.data[7];
    let lut = f.pq.compute_adc_lut(query).expect("adc lut");
    let expected: f32 = f.codes[7]
        .iter()
        .enumerate()
        .map(|(sv, &c)| lut[sv][c as usize])
        .sum();

    let found = {
        let _lut = hnsw
            .get_distance()
            .install_query_lut(query)
            .expect("query lut");
        let dummy = vec![0u8; PQ_SUBVECTORS];
        hnsw.search(&dummy, 10, 200)
    };

    assert!(
        found.iter().any(|n| n.d_id == 7),
        "a record could not find itself"
    );

    // Every distance the search reports must be the asymmetric one. The
    // symmetric table would give a different number here, since it compares
    // the dummy query's all-zero codes rather than the query itself.
    for n in found.iter() {
        let adc: f32 = f.codes[n.d_id]
            .iter()
            .enumerate()
            .map(|(sv, &c)| lut[sv][c as usize])
            .sum();
        assert!(
            (n.distance - adc).abs() <= 1e-4 * adc.max(1.0),
            "search reported {} for record {} where its asymmetric distance is {}, \
             so the query path is no longer using the query table",
            n.distance,
            n.d_id,
            adc
        );
    }

    // The record's own ADC distance is the smallest of the ten returned,
    // which is what the ranking has to produce.
    assert!(
        found[0].distance <= expected + 1e-4,
        "the top hit scored {} against the query's own record at {}",
        found[0].distance,
        expected
    );
}

/// Guards the vendored hnsw_rs patch that files reverse links at the
/// layer being processed instead of at the inserting point's own top
/// layer. Without the patch, points assigned a level above zero lose
/// their layer-zero inbound adjacency and can become unreachable to
/// similarity search, and at this index size roughly one to two
/// percent of self-queries fail. A failure here means the patch was
/// lost, most likely during an hnsw_rs upgrade. See
/// vendor/hnsw_rs/ZEUSDB-PATCH.md.
///
/// Insertion is sequential on purpose. `parallel_insert` assigns levels
/// in whatever order threads reach the level generator, so the graph
/// varies between runs even under the fixed seed and the test was
/// intermittently red. Building one vector at a time makes the graph a
/// function of the data and the parameters alone, which is also the
/// path every non-quantized index takes through `add()`. The defect
/// this test guards is not specific to the parallel path.
#[test]
fn self_query_reachability() {
    const N: usize = 3000;
    const DIM: usize = 32;

    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect())
        .collect();

    let hnsw = Hnsw::new(16, N, 16, 200, DistCosine {});
    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v.as_slice(), i));
    }

    let failures: Vec<usize> = (0..N)
        .filter(|&i| hnsw.search(&data[i], 1, 64).first().map(|n| n.d_id) != Some(i))
        .collect();

    assert!(
        failures.is_empty(),
        "{} of {} points cannot find themselves by self-query (first: {:?}); \
         the hnsw_rs reverse link layer patch is missing",
        failures.len(),
        N,
        &failures[..failures.len().min(10)]
    );
}

/// Guards the vendored hnsw_rs patch that stops the layer-zero overflow
/// pop from evicting a point's last inbound link. Without the patch the
/// pop always discards the farthest entry, so a point whose only inbound
/// link happens to be that entry is left with no layer-zero in-edge and
/// becomes an orphan that no search can reach through the graph. A
/// failure here means the patch was lost, most likely during an hnsw_rs
/// upgrade. See vendor/hnsw_rs/ZEUSDB-PATCH.md.
///
/// In-degree is counted from the adjacency lists rather than from the
/// patch's own counters, so the assertion holds against the graph itself
/// and stays meaningful if the counters are ever wrong.
///
/// `M` is 4 rather than the shipped 16 on purpose. The layer-zero
/// neighbour cap is `2 * M`, so a small `M` fills lists early and makes
/// the overflow pop frequent, which is the only site that can strand a
/// point. Whether the shipped `M` of 16 strands points depends on the
/// data model rather than on index size. On uniform vectors like these
/// it strands none up to 30,000 points, while on clustered data, 50
/// Gaussian clusters at sigma 0.15 in 768 dimensions, it strands 6 of
/// 10,000. A small `M` lets this uniform generator fail fast instead,
/// and the unpatched crate strands 24 of these 5,000 points.
///
/// Insertion is sequential for the same reason as `self_query_reachability`.
#[test]
fn layer_zero_in_degree() {
    const N: usize = 5000;
    const DIM: usize = 128;
    const M: usize = 4;

    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect())
        .collect();

    let hnsw = Hnsw::new(M, N, 16, 200, DistCosine {});
    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v.as_slice(), i));
    }

    let mut in_degree = vec![0usize; N];
    let mut nb_seen = 0usize;
    for point in hnsw.get_point_indexation() {
        nb_seen += 1;
        for neighbour in &point.get_neighborhood_id()[0] {
            in_degree[neighbour.d_id] += 1;
        }
    }
    assert_eq!(nb_seen, N, "walked {} points, expected {}", nb_seen, N);

    let orphans: Vec<usize> = (0..N).filter(|&i| in_degree[i] == 0).collect();

    assert!(
        orphans.is_empty(),
        "{} of {} points have zero layer-zero in-degree (first: {:?}); \
         the hnsw_rs overflow pop guard is missing",
        orphans.len(),
        N,
        &orphans[..orphans.len().min(10)]
    );
}
