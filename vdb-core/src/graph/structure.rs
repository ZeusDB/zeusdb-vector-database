//! What holds ZeusDB's graph to its own properties.
//!
//! Until 0.7.0 this file was `parity.rs` and every test in it compared a ZeusDB
//! structure against the vendored `Hnsw` on one topology, over the traversal in
//! relay 77, the insert in relay 79 and the real data harness in both. That
//! comparison is gone with the crate it compared against, and what is left is
//! the part that never needed a second implementation.
//!
//! **The strongest evidence in the 0.7.0 arc does not survive here.** Relay 79
//! compared the two builders edge for edge over 18,400 nodes and 352,323 edges,
//! and relay 77 compared the two traversals over 45,465 pages of real data.
//! Neither can be written without a reference. What holds the graph now is the
//! round trip, the loader's rejections, the memory arithmetic, the sort tie
//! order, the level stream against a recorded golden vector, the reservation
//! cap, the seam wiring, the eight graph guards in `hnsw_index`, and the
//! Python-level suite over real data. A change to the insert is provable
//! against ZeusDB's own recorded behaviour and not against an independent
//! builder.

use super::dump::{
    parse_dump, write_dump, DumpElement, Expected, GraphKind, ParsedDump, DUMP_FILENAME,
};
use super::dump::{LoadedEdge, LoadedPoint, PointId, NB_LAYER_MAX};
use super::levels::{LevelGenerator, DEFAULT_LEVEL_SEED};
use super::mutable::{reserved_records, MutableGraph, RESERVE_BYTES};
use super::store::VectorStore;
use super::traverse::LAYERS;
use super::{Distance, VectorGraph};
use crate::distance::DistPQ;
use crate::distance::{CosineDist, L1Dist, L2Dist};
use crate::pq::PQ;
use std::sync::Arc;

type BoxedFilter = Box<dyn Fn(&usize) -> bool>;

// ============================================================================
// SMALL GRAPH FIXTURES
// ============================================================================

/// Deterministic pseudo-random vectors, the generator the dump tests use.
fn sample_vectors(records: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed | 1;
    (0..records)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / 16_777_216.0 - 0.5
                })
                .collect()
        })
        .collect()
}

/// The same, unit normalised, which is what the cosine path holds.
fn unit_sample_vectors(records: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    sample_vectors(records, dim, seed)
        .into_iter()
        .map(|mut v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect()
}

/// Build a graph by sequential insertion, which is the shipped build path.
fn build<T, D>(
    data: &[Vec<T>],
    m: usize,
    ef_construction: usize,
    dist: D,
) -> (MutableGraph<T, D>, VectorStore<T>)
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    let dim = data.first().map_or(1, Vec::len);
    let scale = LevelGenerator::default_scale(m);
    let mut levels = LevelGenerator::new(scale, LAYERS);
    let (mut graph, mut store) =
        MutableGraph::new(dim, m, ef_construction, scale, data.len().max(1), dist)
            .expect("the fixture parameters are inside the accepted range");
    for (id, values) in data.iter().enumerate() {
        graph.insert(&mut store, values.as_slice(), id, &mut levels);
    }
    (graph, store)
}

/// Round a built graph through the dump writer and reader, which is the
/// topology source the production loader uses, and hand the parse back.
fn parsed_topology<T, D>(
    graph: &MutableGraph<T, D>,
    store: &VectorStore<T>,
    kind: GraphKind,
) -> ParsedDump<T>
where
    T: Clone + Send + Sync + DumpElement,
    D: Distance<T> + Send + Sync,
{
    let dir = tempfile::tempdir().unwrap();
    write_dump(&graph.dump_view(store), kind, dir.path()).unwrap();
    let expected = Expected {
        kind,
        dimension: graph.dim(),
        m: graph.m(),
        ef_construction: graph.ef_construction(),
        min_nodes: 0,
    };
    parse_dump::<T>(dir.path(), &expected).unwrap()
}

// ============================================================================
// THE STRUCTURE
// ============================================================================

/// The parameters construction needs survive the load unchanged.
#[test]
fn the_mutable_graph_keeps_the_construction_parameters() {
    let data = sample_vectors(300, 8, 0x78_0a);
    let (built, built_store) = build(&data, 24, 80, CosineDist {});
    let parsed = parsed_topology(&built, &built_store, GraphKind::Cosine);
    let entry = parsed.entry;
    let scale = parsed.level_scale;
    let mutable = MutableGraph::from_loaded(
        parsed.points_by_layer,
        entry,
        parsed.m,
        parsed.ef_construction,
        scale,
        CosineDist {},
    )
    .unwrap()
    .0;
    assert_eq!(mutable.nb_points(), 300);
    assert_eq!(mutable.m(), 24);
    assert_eq!(mutable.ef_construction(), 80);
    assert_eq!(mutable.level_scale(), scale);
    assert_eq!(mutable.entry_point_id(), entry);
    assert_eq!(mutable.dim(), 8);
    // A loaded graph numbers its nodes in dump order, which is layer major, so
    // every node's level is the layer it arrived in and the levels are
    // non-decreasing across the arena.
    let mut previous = 0u8;
    for node in 0..mutable.nb_points() as u32 {
        let level = mutable.level(node);
        assert!(level >= previous);
        previous = level;
    }
}

/// Every rejection the loading constructor makes, made here, plus the one rule
/// this layout adds and the residue rule it keeps.
#[test]
fn the_mutable_loader_refuses_malformed_topology() {
    let point = |neighbours: Vec<Vec<LoadedEdge>>| LoadedPoint {
        origin_id: 0,
        data: vec![0.0f32, 1.0],
        neighbours,
    };
    let refuse = |points: Vec<Vec<LoadedPoint<f32>>>, entry: PointId, needle: &str| {
        let error = MutableGraph::from_loaded(points, entry, 16, 64, 0.36, CosineDist {})
            .err()
            .expect("a malformed topology must be refused");
        assert!(error.contains(needle), "{}", error);
    };

    refuse(vec![], PointId(0, 0), "between 1 and");
    refuse(vec![vec![]], PointId(0, 0), "no entry point");
    refuse(
        vec![vec![point(vec![])]],
        PointId(0, 5),
        "no point is there",
    );
    refuse(
        vec![vec![point(vec![])]],
        PointId(3, 0),
        "no point is there",
    );
    refuse(
        vec![vec![
            point(vec![]),
            LoadedPoint {
                origin_id: 1,
                data: vec![0.0f32; 3],
                neighbours: vec![],
            },
        ]],
        PointId(0, 0),
        "holds 3 values",
    );
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(9, 0),
            distance: 0.5,
        }]])]],
        PointId(0, 0),
        "that layer holds",
    );
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(0, 7),
            distance: 0.5,
        }]])]],
        PointId(0, 0),
        "that layer holds",
    );
    refuse(
        vec![vec![point(vec![vec![LoadedEdge {
            target: PointId(0, 0),
            distance: f32::NAN,
        }]])]],
        PointId(0, 0),
        "carries a distance",
    );

    // The rule this layout adds: a list longer than its slab. The insert never
    // produces one, since it shrinks past the cap under the same guard that
    // grew it, so this is a dump no ZeusDB save wrote.
    let long: Vec<Vec<LoadedPoint<f32>>> = vec![vec![LoadedPoint {
        origin_id: 0,
        data: vec![0.0f32, 1.0],
        neighbours: vec![(0..6)
            .map(|_| LoadedEdge {
                target: PointId(0, 0),
                distance: 0.5,
            })
            .collect()],
    }]];
    let error = MutableGraph::from_loaded(long, PointId(0, 0), 2, 64, 0.36, CosineDist {})
        .err()
        .expect("a list longer than its slab must be refused");
    assert!(error.contains("and a list holds 5"), "{}", error);

    // And the overflow slot itself is representable, which is the state the
    // guarded pop works in: `2 * m + 1` entries at layer zero.
    let full: Vec<Vec<LoadedPoint<f32>>> = vec![vec![LoadedPoint {
        origin_id: 0,
        data: vec![0.0f32, 1.0],
        neighbours: vec![(0..5)
            .map(|_| LoadedEdge {
                target: PointId(0, 0),
                distance: 0.5,
            })
            .collect()],
    }]];
    let held = MutableGraph::from_loaded(full, PointId(0, 0), 2, 64, 0.36, CosineDist {})
        .unwrap()
        .0;
    assert_eq!(held.nb_edges(), 5);

    // A list above its owner's level is kept rather than dropped, which is what
    // makes the structure a lossless image of the file.
    let above = vec![vec![point(vec![
        vec![],
        vec![LoadedEdge {
            target: PointId(0, 0),
            distance: 0.5,
        }],
    ])]];
    let mutable = MutableGraph::from_loaded(above, PointId(0, 0), 16, 64, 0.36, CosineDist {})
        .unwrap()
        .0;
    assert_eq!(mutable.nb_edges(), 0);
    assert_eq!(mutable.above_level_edges(), 1);

    let ok = || vec![vec![point(vec![])]];
    assert!(MutableGraph::from_loaded(ok(), PointId(0, 0), 0, 64, 0.36, CosineDist {}).is_err());
    assert!(MutableGraph::from_loaded(ok(), PointId(0, 0), 300, 64, 0.36, CosineDist {}).is_err());
    assert!(
        MutableGraph::from_loaded(ok(), PointId(0, 0), 16, 64, f64::NAN, CosineDist {}).is_err()
    );
    assert!(MutableGraph::from_loaded(ok(), PointId(0, 0), 16, 64, -1.0, CosineDist {}).is_err());
}

/// The memory figure is exact arithmetic over the buffers, so the per-node
/// arithmetic the relays state can be checked rather than believed.
#[test]
fn the_mutable_memory_figure_is_exact() {
    const M: usize = 8;
    const DIM: usize = 12;
    let data = sample_vectors(500, DIM, 0x78_0b);
    let (built, built_store) = build(&data, M, 32, CosineDist {});
    let parsed = parsed_topology(&built, &built_store, GraphKind::Cosine);
    let layer_lens: Vec<usize> = parsed.points_by_layer.iter().map(Vec::len).collect();
    let mutable = MutableGraph::from_loaded(
        parsed.points_by_layer,
        parsed.entry,
        parsed.m,
        parsed.ef_construction,
        parsed.level_scale,
        CosineDist {},
    )
    .unwrap()
    .0;

    let nodes = mutable.nb_points();
    let mut expected = std::mem::size_of_val(&mutable);
    // Per node, beside the vector: the origin id, the level, the layer zero
    // length, the layer zero inbound counter, the first upper list and the span.
    expected += nodes * std::mem::size_of::<usize>();
    expected += nodes;
    expected += nodes * std::mem::size_of::<u16>();
    expected += nodes * std::mem::size_of::<u32>();
    expected += nodes * std::mem::size_of::<u32>();
    expected += nodes;
    // The inverse of the origin ids, one node index per internal id. The
    // fixture inserts under ids zero upward, so it holds one slot per node.
    expected += nodes * std::mem::size_of::<u32>();
    // No vector. The store is a separate object and its bytes are its own; see
    // `VectorStore::memory_bytes`.
    // The layer zero slabs, `2 * m + 1` targets and distances each.
    expected += nodes * (2 * M + 1) * 2 * std::mem::size_of::<u32>();
    // Per upper list: its offset, its length, its capacity and its inbound
    // counter, being twelve bytes whatever the list holds.
    let lists = mutable.nb_upper_lists();
    expected += lists * std::mem::size_of::<u32>();
    expected += lists * std::mem::size_of::<u16>();
    expected += lists * std::mem::size_of::<u16>();
    expected += lists * std::mem::size_of::<u32>();
    // The upper slots, a target and a distance each.
    expected += mutable.upper_slots() * 2 * std::mem::size_of::<u32>();
    assert_eq!(mutable.memory_bytes(), expected);

    // Every layer holds the points drawn at exactly that level, and the span a
    // node needs runs above it, so the list count exceeds what the levels alone
    // predict. That gap is the correction relay 79 made to the layout.
    let by_level: usize = layer_lens
        .iter()
        .enumerate()
        .map(|(layer, len)| layer * len)
        .sum();
    println!(
        "mutable memory nodes {} edges {} above-level {} upper lists {} (levels alone \
         predict {}) slots {} bytes {} per_node {:.2}",
        nodes,
        mutable.nb_edges(),
        mutable.above_level_edges(),
        lists,
        by_level,
        mutable.upper_slots(),
        mutable.memory_bytes(),
        (mutable.memory_bytes() - nodes * DIM * 4) as f64 / nodes as f64
    );
}

/// The concurrency shape the design states: shareable across the threads a
/// released-GIL search runs on.
#[test]
fn the_mutable_graph_is_send_and_sync() {
    fn assert_send_sync<X: Send + Sync>() {}
    assert_send_sync::<MutableGraph<f32, CosineDist>>();
    assert_send_sync::<MutableGraph<f32, L2Dist>>();
    assert_send_sync::<MutableGraph<f32, L1Dist>>();
    assert_send_sync::<MutableGraph<u8, DistPQ>>();
}

/// What a non-finite query does if one ever reaches the traversal.
///
/// Every ZeusDB entry point rejects it first, so this documents the behaviour
/// rather than supporting it. An infinite query traverses and returns a page,
/// and a NaN query panics at the candidate assertion rather than at the heap
/// ordering, because the first heap holds one element and a one element heap
/// never compares, so `NaN <= 0.` fails first.
#[test]
fn a_non_finite_query_traverses_or_panics_at_the_candidate_assertion() {
    let data = sample_vectors(400, 8, 0x77_09);
    let (graph, store) = build(&data, 16, 48, L2Dist {});

    let infinite = vec![f32::INFINITY; 8];
    let page = graph.search(&store, &infinite, 10, 20, None::<&BoxedFilter>);
    assert_eq!(page.len(), 10, "an infinite query still returns a page");
    assert!(
        page.iter().all(|hit| hit.distance.is_infinite()),
        "every score of an infinite query is infinite"
    );

    let nan = vec![f32::NAN; 8];
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        graph.search(&store, &nan, 10, 20, None::<&BoxedFilter>)
    }));
    match panicked {
        Err(any) => {
            let text = any
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| any.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            assert_eq!(text, "assertion failed: c.dist_to_ref <= 0.");
        }
        Ok(_) => panic!("a NaN query should panic on the traversal"),
    }
}

// ============================================================================
// THE DUMP ROUND TRIP
// ============================================================================

/// Write a graph out, load it into the structure, write it out again, and
/// compare the two files byte for byte.
fn round_trip<T, D>(
    built: &MutableGraph<T, D>,
    built_store: &VectorStore<T>,
    kind: GraphKind,
    dist: impl Fn() -> D,
) -> (usize, usize)
where
    T: Clone + Send + Sync + DumpElement,
    D: Distance<T> + Send + Sync,
{
    let first = tempfile::tempdir().unwrap();
    write_dump(&built.dump_view(built_store), kind, first.path()).unwrap();
    let expected = Expected {
        kind,
        dimension: built.dim(),
        m: built.m(),
        ef_construction: built.ef_construction(),
        min_nodes: 0,
    };
    let parsed = parse_dump::<T>(first.path(), &expected).unwrap();
    let (mutable, mutable_store) = MutableGraph::from_loaded(
        parsed.points_by_layer,
        parsed.entry,
        parsed.m,
        parsed.ef_construction,
        parsed.level_scale,
        dist(),
    )
    .unwrap();
    let second = tempfile::tempdir().unwrap();
    write_dump(&mutable.dump_view(&mutable_store), kind, second.path()).unwrap();

    let before = std::fs::read(first.path().join(DUMP_FILENAME)).unwrap();
    let after = std::fs::read(second.path().join(DUMP_FILENAME)).unwrap();
    assert_eq!(
        before.len(),
        after.len(),
        "the dump changed length, {} against {}",
        before.len(),
        after.len()
    );
    let differing = before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count();
    let first_difference = before.iter().zip(after.iter()).position(|(a, b)| a != b);
    assert_eq!(
        differing, 0,
        "{} bytes differ, first at offset {:?}",
        differing, first_difference
    );
    (before.len(), mutable.above_level_edges())
}

/// A dump loaded into the structure and written back out is the same file, byte
/// for byte, on every configuration the small tests cover.
///
/// This is what makes the structure a lossless image of the file rather than a
/// lossy one, and it is why the descent residue is stored. Without it the
/// second file would be shorter by exactly the residue edges.
#[test]
fn a_dump_round_trips_through_the_mutable_graph() {
    let (cosine, cosine_store) = build(&sample_vectors(1500, 24, 0x78_01), 16, 64, CosineDist {});
    let (bytes, residue) = round_trip(&cosine, &cosine_store, GraphKind::Cosine, || CosineDist {});
    println!("round trip cosine bytes {} residue {}", bytes, residue);
    assert!(residue > 0, "the fixture must carry descent residue");

    let (l2, l2_store) = build(&sample_vectors(900, 16, 0x78_02), 16, 48, L2Dist {});
    let (bytes, residue) = round_trip(&l2, &l2_store, GraphKind::L2, || L2Dist {});
    println!("round trip l2 bytes {} residue {}", bytes, residue);

    let (l1, l1_store) = build(&sample_vectors(700, 12, 0x78_03), 8, 48, L1Dist {});
    let (bytes, residue) = round_trip(&l1, &l1_store, GraphKind::L1, || L1Dist {});
    println!("round trip l1 bytes {} residue {}", bytes, residue);

    // Ties, where a list holds runs of equal stored distances and only a stable
    // ordering reproduces the file.
    let distinct = sample_vectors(25, 16, 0x78_04);
    let repeated: Vec<Vec<f32>> = (0..600).map(|i| distinct[i % 25].clone()).collect();
    let (ties, ties_store) = build(&repeated, 16, 48, CosineDist {});
    let (bytes, residue) = round_trip(&ties, &ties_store, GraphKind::Cosine, || CosineDist {});
    println!("round trip ties bytes {} residue {}", bytes, residue);

    // The quantized graph, whose elements are `u8` codes.
    const N: usize = 800;
    const DIM: usize = 32;
    const SUBVECTORS: usize = 8;
    let all = crate::test_vectors::clustered(N, DIM, 0x78_05);
    let pq = Arc::new(PQ::new(DIM, SUBVECTORS, 8, 500, None));
    pq.train(&all).unwrap();
    let refs: Vec<&[f32]> = all.iter().map(|v| v.as_slice()).collect();
    let codes = pq.quantize_batch(&refs).unwrap();
    let (quantized, quantized_store) = build(&codes, 16, 100, DistPQ::new(pq.clone()));
    let (bytes, residue) = round_trip(&quantized, &quantized_store, GraphKind::CosinePq, || {
        DistPQ::new(pq.clone())
    });
    println!("round trip quantized bytes {} residue {}", bytes, residue);
}

// ============================================================================
// THE NEIGHBOUR LIST SORT
// ============================================================================

/// Two edges at an equal distance keep the order they were filed in.
///
/// **This replaces a comparison against a live `Vec<Arc<_>>` sort, and the
/// reference it held has no meaning now.** The sorts were `sort_unstable`, which
/// dispatches on the element type and orders equal keys differently on the two
/// paths, so reproducing the vendored builder's permutation meant matching the
/// element it sorted as well as the order. That comparison said nothing about
/// what the list ought to hold. It said only that two arbitrary permutations
/// agreed, and it would have gone on agreeing if both were wrong.
///
/// The sorts are stable now, so the permutation over equal keys is a stated
/// property rather than a coincidence of dispatch, and this holds the property.
/// A list carries its entries in the order the reverse update filed them, and
/// ties are common rather than exotic: quantized codes make whole clusters
/// equidistant, and the fixture below ties on purpose.
///
/// The consequence is that a sort no longer depends on the element type, the
/// toolchain's dispatch thresholds, or the width of [`Entry`]. What it does
/// depend on is the comparator being consistent, which `by_distance` is because
/// it panics on the one value that would break it.
#[test]
fn the_entry_sort_keeps_its_insertion_order() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut cases = 0usize;
    let mut tie_bearing = 0usize;
    let mut tied_pairs = 0usize;
    // A layer zero list runs to `2 * m + 1`, which is 33 at the shipped `m` and
    // 129 at the largest `m` the index accepts. Lengths above 20 are the ones
    // that matter most, since that is where the standard library stops running
    // a plain insertion sort and starts merging.
    for len in 1..=130usize {
        for _ in 0..40 {
            // A small number of distinct distances, so ties are the rule.
            let distinct = 1 + (next() % 6) as usize;
            let dists: Vec<f32> = (0..len)
                .map(|_| (next() % distinct as u64) as f32)
                .collect();
            if dists.len() > distinct {
                tie_bearing += 1;
            }

            // `entries_for_test` tags entry `i` with target `i`, so the target
            // is the position it was filed at.
            let mut entries = super::mutable::entries_for_test(&dists);
            super::mutable::sort_entries_for_test(&mut entries);
            assert_eq!(entries.len(), len);

            for pair in entries.windows(2) {
                let (a, b) = (&pair[0], &pair[1]);
                assert!(
                    a.dist <= b.dist,
                    "the list came out unsorted at length {} on {:?}",
                    len,
                    dists
                );
                if a.dist == b.dist {
                    tied_pairs += 1;
                    assert!(
                        a.target < b.target,
                        "two entries at distance {} came out as {} then {} at \
                         length {}, so the sort reordered a tie",
                        a.dist,
                        a.target,
                        b.target,
                        len
                    );
                }
            }
            cases += 1;
        }
    }
    // The fixture has to reach the property, or this asserts only that the sort
    // sorts. Roughly nine adjacent pairs in ten are tied at these widths.
    assert!(
        tied_pairs > 100_000,
        "only {} tied pairs over {} lists, which is not enough to hold the \
         property",
        tied_pairs,
        cases
    );
    println!(
        "entry sort: {} lists, {} of them holding ties, {} tied adjacent pairs, \
         0 reordered",
        cases, tie_bearing, tied_pairs
    );
}

/// [`Entry`] is eight bytes, which is what keeps the stable sort off the
/// allocator.
///
/// The standard library's stable sort serves its scratch space from a 4 KiB
/// stack buffer whenever the request fits and calls the allocator only above
/// that, so an eight byte element allocates only past 512 entries. The longest
/// list the index can build is `2 * m + 1`, which is 129 at the largest `m`
/// accepted and 33 at the shipped one. A wider `Entry` narrows that margin, and
/// this is where it would be noticed.
#[test]
fn an_entry_is_eight_bytes() {
    assert_eq!(std::mem::size_of::<super::mutable::Entry>(), 8);
    assert_eq!(4096 / std::mem::size_of::<super::mutable::Entry>(), 512);
}

// ============================================================================
// THE LEVEL GENERATOR
// ============================================================================

/// The stream a generator produces, as a checksum over every draw, the first
/// sixty four draws written out, and the redraws it took.
///
/// A checksum rather than the whole sequence because a hundred thousand draws
/// The default scale's stream at `m` 16, over 100,000 draws.
const DEFAULT_STREAM_HASH: u64 = 0x9a98_4bab_5466_7eb7;
#[rustfmt::skip]
const DEFAULT_STREAM_PREFIX: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
];
/// The same at a scale of 4, where the cap binds and the redraw fires.
const CAPPED_STREAM_HASH: u64 = 0x5ea3_633e_ae11_6818;
#[rustfmt::skip]
const CAPPED_STREAM_PREFIX: &[u8] = &[
     0,  1,  2,  1,  2,  0,  1,  0,  2,  4,  1,  1,  3,  5,  8, 14,
     0,  3,  1,  1,  3,  2, 13,  1, 10,  7,  1,  3,  2,  2,  1,  1,
     6,  6,  5,  7,  3,  7,  5,  7,  1,  2,  1,  8,  4,  2,  2,  5,
     8,  0,  0,  3,  2,  0,  0,  4,  1,  5,  0,  0,  0,  0,  0,  0,
];
/// Draws the capped stream sent back for a second value.
const CAPPED_STREAM_REDRAWS: usize = 1_855;
/// A stream reseeded part way through, at a scale of 4 over 5,000 draws.
const RESEEDED_STREAM_HASH: u64 = 0x96f5_f517_51a7_c751;

fn stream_signature(
    scale: f64,
    draws: usize,
    warmup: usize,
    reseed: Option<u64>,
) -> (u64, Vec<u8>, usize) {
    let mut generator = LevelGenerator::new(scale, NB_LAYER_MAX as usize);
    assert_eq!(generator.scale(), scale, "the scale is installed as given");
    for _ in 0..warmup {
        generator.generate();
    }
    if let Some(seed) = reseed {
        generator.set_seed(seed);
    }
    let before = generator.redraws();
    let levels: Vec<u8> = (0..draws).map(|_| generator.generate() as u8).collect();
    // FNV-1a over the draws, which is enough to detect any drift and is written
    // out here so the value does not depend on a hasher the standard library
    // may change.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &level in &levels {
        hash ^= level as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash, levels[..64].to_vec(), generator.redraws() - before)
}

/// The level stream is a particular sequence, and this is that sequence.
///
/// **This replaces a comparison against the vendored generator, and it is
/// stronger on the one hazard that matters.** Both generators drew from
/// `rand::rngs::StdRng`, which `rand` documents as non-portable and free to
/// change algorithm in any release. A comparison between two callers of the
/// same `StdRng` moves with it and stays green while every graph a build
/// produces changes. A recorded stream does not.
///
/// The recording was taken while the generator was `StdRng`, and it is
/// unchanged. `crate::rng::SeededRng` names the algorithm `StdRng` was, so the
/// pin moved no draw and this test is the evidence, since it holds a sequence
/// recorded before the pin existed.
///
/// So a failure here means one of two things now. The draw itself changed, or
/// the scale did. The third cause it used to have, being `rand` replacing
/// `StdRng` underneath the generator, is what the pin removed.
#[test]
fn the_level_stream_matches_the_recorded_one() {
    // The default scale at `m` 16, which is what every shipped index draws
    // with. The cap never binds here: `P(level >= 16)` is `16^-16`.
    let (hash, prefix, redraws) =
        stream_signature(LevelGenerator::default_scale(16), 100_000, 0, None);
    println!(
        "levels default hash {:#018x} redraws {} prefix {:?}",
        hash, redraws, prefix
    );
    assert_eq!(hash, DEFAULT_STREAM_HASH, "the default level stream moved");
    assert_eq!(&prefix[..], DEFAULT_STREAM_PREFIX);
    assert_eq!(redraws, 0, "the cap cannot bind at the default scale");

    // A scale where the cap binds. `P(level >= 16)` is `exp(-16 / 4)`, about
    // one draw in fifty five, so the redraw path is exercised thousands of
    // times and every draw after the first one has to have consumed the same
    // amount of the stream.
    let (hash, prefix, redraws) = stream_signature(4.0, 100_000, 0, None);
    println!(
        "levels capped hash {:#018x} redraws {} prefix {:?}",
        hash, redraws, prefix
    );
    assert_eq!(hash, CAPPED_STREAM_HASH, "the capped level stream moved");
    assert_eq!(&prefix[..], CAPPED_STREAM_PREFIX);
    // The redraw consumes a second value from the same stream, so a generator
    // that redrew at a different rate would diverge from that point on. The
    // count is part of the recorded stream rather than a statistic about it.
    assert_eq!(redraws, CAPPED_STREAM_REDRAWS, "the redraw rate moved");
}

/// `set_seed` resets the stream rather than extending it, whatever it had drawn
/// before.
#[test]
fn the_level_seed_resets_the_generator() {
    // Reseeding to the default after a warm up reproduces the cold stream.
    let cold = stream_signature(LevelGenerator::default_scale(16), 2_000, 0, None);
    let warm = stream_signature(
        LevelGenerator::default_scale(16),
        2_000,
        500,
        Some(DEFAULT_LEVEL_SEED),
    );
    assert_eq!(cold, warm);

    // And a reseed to a chosen value lands on a recorded stream too, with the
    // generator advanced by a different amount first.
    let (hash, _, _) = stream_signature(4.0, 5_000, 137, Some(0x0102_0304_0506_0708));
    println!("levels reseeded hash {:#018x}", hash);
    assert_eq!(hash, RESEEDED_STREAM_HASH, "the reseeded stream moved");
}

/// The creation-time reservation is capped in bytes.
///
/// `Vec::with_capacity` aborts the process on allocation failure rather than
/// unwinding, so the largest declaration `HNSWIndex::build` admits has to be
/// bounded before it is reached rather than caught afterwards. Checked over the
/// arithmetic rather than by taking the allocation, since taking it is the thing
/// being prevented.
#[test]
fn the_reservation_is_capped_in_bytes() {
    // What `MAX_EXPECTED_SIZE` admits, at the dimension the shipped indexes run
    // at and the degree they run at, whose expected span is 5.
    let declared = 100_000_000usize;
    let per_record = 20 + 1536 * 4 + 33 * 8 + 5 * 20;
    let reserved = reserved_records::<f32>(1536, 33, 5, declared);
    let uncapped = declared * per_record;
    let capped = reserved * per_record;
    println!(
        "expected_size {} would reserve {} bytes and reserves {} for {} records",
        declared, uncapped, capped, reserved
    );
    assert!(
        uncapped > 600 * (1usize << 30),
        "the uncapped reservation is {} bytes, so there is nothing to cap",
        uncapped
    );
    assert!(reserved < declared);
    assert!(
        capped <= RESERVE_BYTES,
        "the capped reservation is {} bytes against a budget of {}",
        capped,
        RESERVE_BYTES
    );

    // A declaration inside the budget is reserved in full, so the cap bounds
    // the hint rather than replacing it.
    assert_eq!(reserved_records::<f32>(128, 33, 4, 10_000), 10_000);
    assert_eq!(reserved_records::<u8>(96, 33, 5, 50_000), 50_000);
}

/// The seam's `set_level_seed` reaches the generator its graph draws from.
///
/// Built through the shipped seam rather than through the structure, so what is
/// held is the wiring and not the generator, which the two tests above already
/// hold. Three graphs over the same records: one left at the default, one
/// reseeded to the default, and one reseeded to another value. Compared as the
/// bytes each writes, which is the whole graph and its layer table.
#[test]
fn the_seam_reseeds_the_level_stream() {
    // Unit normalised, because this builds through the seam on the cosine space
    // and `assert_unit_for_cosine` holds every vector that reaches one to what
    // `process_vector_for_space` would have handed it.
    let data = unit_sample_vectors(600, 12, 4242);
    let build = |seed: Option<u64>| {
        let mut graph = VectorGraph::new_raw("cosine", 12, 16, data.len(), LAYERS, 64);
        if let Some(seed) = seed {
            graph.set_level_seed(seed);
        }
        for (id, vector) in data.iter().enumerate() {
            graph.insert(vector, id);
        }
        let dir = tempfile::tempdir().unwrap();
        graph.dump(dir.path()).unwrap();
        std::fs::read(dir.path().join(DUMP_FILENAME)).unwrap()
    };

    let untouched = build(None);
    let reseeded_to_default = build(Some(DEFAULT_LEVEL_SEED));
    let reseeded_elsewhere = build(Some(0x0102_0304_0506_0708));

    // A graph nobody reseeded draws from the default, so the setter and the
    // constructor agree.
    assert_eq!(untouched, reseeded_to_default);
    // And the setter is doing the work rather than being a no-op.
    assert_ne!(untouched, reseeded_elsewhere);
    println!(
        "seam reseed: default {} bytes, other seed {} bytes",
        untouched.len(),
        reseeded_elsewhere.len()
    );
}
