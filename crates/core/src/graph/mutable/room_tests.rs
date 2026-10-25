//! The graph's ceilings and the room an insertion is admitted under.
//!
//! Each arena an insertion grows is held to a ceiling, and the room is the
//! most an insertion can add to each, reckoned from its level before it is
//! planned. These tests hold every insertion of every graph kind to the room
//! at three degrees, built and loaded, admit an insertion at each ceiling and
//! refuse one past it, refuse an insertion a call's earlier ones leave no
//! room for in the words that say so, and hand back the rows of a dump whose
//! graph would pass a ceiling.

use super::*;
use crate::distance::{
    CosineDist, DistPQ, DotDist, Int8Dist, Int8Metric, L1Dist, L2Dist, PqMetric,
};
use crate::graph::dump::{
    read_dump_at, write_dump, DumpElement, DumpRead, Expected, GraphKind, DUMP_FILENAME,
};
use crate::graph::levels::LevelGenerator;
use crate::int8::Int8Codec;
use crate::pq::PQ;
use crate::storage::FsDir;
use std::sync::Arc;

/// The graph's ceilings, lowered on this thread for the length of a test and
/// put back when dropped.
struct Lowered(GraphCeilings);

impl Lowered {
    fn to(ceilings: GraphCeilings) -> Self {
        Lowered(GRAPH_CEILINGS.with(|held| held.replace(ceilings)))
    }
}

impl Drop for Lowered {
    fn drop(&mut self) {
        GRAPH_CEILINGS.with(|held| held.set(self.0));
    }
}

/// Deterministic values in `[-0.5, 0.5)`.
fn spread(records: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
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

/// What each arena of `graph` holds, in the order of [`ARENAS`].
fn arenas<T, D>(graph: &MutableGraph<T, D>) -> [usize; 4]
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    [
        graph.nb_points(),
        graph.nb_upper_lists(),
        graph.nb_wide_lists(),
        graph.upper_slots(),
    ]
}

/// Every node owns a list at every layer up to its level, so a span grown at
/// a layer is always above the level and takes an empty word.
fn assert_spans_reach_levels<T, D>(graph: &MutableGraph<T, D>, label: &str)
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    for node in 0..graph.nb_points() as u32 {
        assert!(
            graph.span(node) >= graph.level(node) as usize,
            "{label}: node {node} spans {} layers at level {}",
            graph.span(node),
            graph.level(node)
        );
    }
}

/// Insert `rows` into `graph` one at a time at the levels `level` gives, each
/// insertion held to the bound the room gives for it before it is planned,
/// arena by arena, and the room admitting it.
fn insert_within_the_room<T, D>(
    graph: &mut MutableGraph<T, D>,
    store: &mut VectorStore<T>,
    rows: &[Vec<T>],
    first_id: usize,
    mut level: impl FnMut(usize) -> usize,
    label: &str,
) where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    for (k, row) in rows.iter().enumerate() {
        let id = first_id + k;
        let drawn = level(k);
        let mut room = graph.room();
        let bound = room.bound(drawn, room.top);
        room.take(drawn)
            .unwrap_or_else(|e| panic!("{label}: insertion {id} at level {drawn}: {e}"));
        let before = arenas(graph);
        let planned = graph.plan_insertion(store, row, drawn);
        graph.install_insertion(store, row, id, id as u32, planned);
        let after = arenas(graph);
        for arena in 0..ARENAS.len() {
            assert!(
                after[arena] - before[arena] <= bound[arena],
                "{label}: insertion {id} at level {drawn} added {} {} and its bound is {}",
                after[arena] - before[arena],
                ARENAS[arena],
                bound[arena]
            );
        }
    }
}

/// `rows` built into a graph at `m` and held to the room on every insertion,
/// then the first `kept` of them written as a dump, read back, and the rest
/// inserted into the loaded graph under the same check. The levels are the
/// stream's own on the build, and on the loaded graph a sweep that reaches
/// three layers above the top, so a level past the entry level is taken
/// again and again. Every node of both graphs spans at least its level.
fn held_to_the_room<T, D>(
    kind: GraphKind,
    m: usize,
    rows: &[Vec<T>],
    kept: usize,
    dist: impl Fn() -> D,
) where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    let label = format!("{kind:?} at m {m}");
    let width = rows[0].len();
    let scale = LevelGenerator::default_scale(m);
    let mut levels = LevelGenerator::new(scale, LAYERS);
    let (mut built, mut built_store) =
        MutableGraph::new(width, m, 32, scale, rows.len(), dist()).unwrap();
    insert_within_the_room(
        &mut built,
        &mut built_store,
        rows,
        0,
        |_| levels.generate(),
        &format!("{label}, built"),
    );
    assert_spans_reach_levels(&built, &format!("{label}, built"));

    let (mut first, mut first_store) =
        MutableGraph::new(width, m, 32, scale, rows.len(), dist()).unwrap();
    let mut levels = LevelGenerator::new(scale, LAYERS);
    for (id, row) in rows[..kept].iter().enumerate() {
        first.insert(&mut first_store, row, id, &mut levels);
    }
    let dir = tempfile::tempdir().unwrap();
    write_dump(
        &first.dump_view(&first_store),
        kind,
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
    )
    .unwrap();
    let expected = Expected {
        kind,
        dimension: width,
        m,
        ef_construction: 32,
        min_nodes: kept,
        max_origin_id: kept - 1,
        version: 2,
    };
    let (mut read, mut read_store) = match read_dump_at::<T, D>(
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
        &expected,
        dist(),
        None,
        false,
    ) {
        Ok(DumpRead::Graph(graph, store)) => (*graph, store),
        Ok(DumpRead::Rows { reason, .. }) => panic!("{label} gave its rows: {reason}"),
        Err(refusal) => panic!("{label} was refused: {refusal:?}"),
    };
    assert_spans_reach_levels(&read, &format!("{label}, loaded"));
    let top = read.entry_level() as usize;
    insert_within_the_room(
        &mut read,
        &mut read_store,
        &rows[kept..],
        kept,
        |k| (k * 5) % (top + 4),
        &format!("{label}, loaded"),
    );
    assert_spans_reach_levels(&read, &format!("{label}, loaded and grown"));
}

/// Every insertion of every graph kind, at `m` 2, 16 and 256, built and
/// loaded, adds no more to any arena than the room's bound for its level,
/// and every node spans at least its level. The bound is what the door
/// admits a record under, so a record it admits is installed within every
/// ceiling.
#[test]
fn every_insertion_grows_each_arena_within_the_room_its_level_leaves() {
    let dim = 8;
    let unit = |v: &[f32]| {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<f32>>()
    };
    for (m, records) in [(2usize, 1_500usize), (16, 900), (256, 600)] {
        let kept = records * 2 / 3;
        let corpus = spread(records, dim, 0x9e37_79b9_7f4a_7c15 ^ m as u64);
        let units: Vec<Vec<f32>> = corpus.iter().map(|v| unit(v)).collect();
        held_to_the_room(GraphKind::Cosine, m, &units, kept, || CosineDist {});
        held_to_the_room(GraphKind::L2, m, &corpus, kept, || L2Dist {});
        held_to_the_room(GraphKind::L1, m, &corpus, kept, || L1Dist {});
        held_to_the_room(GraphKind::Dot, m, &corpus, kept, || DotDist {});

        let pq = Arc::new(PQ::new(dim, 4, 4, records, None));
        pq.train(&corpus).unwrap();
        let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
        let codes = pq.quantize_batch(&refs).unwrap();
        held_to_the_room(GraphKind::L2Pq, m, &codes, kept, || {
            DistPQ::new(pq.clone(), PqMetric::SquaredL2)
        });
        held_to_the_room(GraphKind::CosinePq, m, &codes, kept, || {
            DistPQ::new(pq.clone(), PqMetric::Cosine)
        });

        for (kind, metric, data) in [
            (GraphKind::CosineInt8, Int8Metric::Cosine, &units),
            (GraphKind::L2Int8, Int8Metric::L2, &corpus),
            (GraphKind::L1Int8, Int8Metric::L1, &corpus),
            (GraphKind::DotInt8, Int8Metric::Dot, &corpus),
        ] {
            let codec = Arc::new(Int8Codec::fit(dim, data).unwrap());
            let dist = Int8Dist::new(codec.clone(), metric);
            let rows: Vec<Vec<i8>> = data.iter().map(|v| dist.encode(v).unwrap()).collect();
            held_to_the_room(kind, m, &rows, kept, || {
                Int8Dist::new(codec.clone(), metric)
            });
        }
    }
}

/// A graph of 400 points at `m` 4, built by insertion.
fn sample() -> (MutableGraph<f32, L2Dist>, VectorStore<f32>) {
    let m = 4;
    let scale = LevelGenerator::default_scale(m);
    let mut levels = LevelGenerator::new(scale, LAYERS);
    let (mut graph, mut store) = MutableGraph::new(4, m, 32, scale, 400, L2Dist {}).unwrap();
    for (id, row) in spread(400, 4, 0x2545_f491_4f6c_dd1d).iter().enumerate() {
        graph.insert(&mut store, row, id, &mut levels);
    }
    (graph, store)
}

/// The layout's ceilings with one arena's set to `ceiling`.
fn with_ceiling(arena: usize, ceiling: usize) -> GraphCeilings {
    let mut ceilings = GraphCeilings::LAYOUT;
    match arena {
        NODES => ceilings.nodes = ceiling,
        WORDS => ceilings.words = ceiling,
        WIDE_LISTS => ceilings.wide_lists = ceiling,
        _ => ceilings.slots = ceiling,
    }
    ceilings
}

/// An insertion whose bound takes an arena exactly to its ceiling is
/// admitted, and the same insertion with the ceiling one lower is refused,
/// naming that arena, what it holds, what the insertion could add and the
/// ceiling, as a graph that is full. Each arena, at a level below the entry
/// level and at one above it.
#[test]
fn an_insertion_at_each_ceiling_is_admitted_and_one_past_it_refused() {
    let (graph, _) = sample();
    let top = graph.entry_level() as usize;
    for level in [0, 1, top + 2] {
        let room = graph.room();
        let held = arenas(&graph);
        let bound = room.bound(level, room.top);
        for arena in 0..ARENAS.len() {
            let at = held[arena] + bound[arena];
            {
                let _lowered = Lowered::to(with_ceiling(arena, at));
                assert!(
                    graph.room().take(level).is_ok(),
                    "level {level}, {} at {at}",
                    ARENAS[arena]
                );
            }
            let _lowered = Lowered::to(with_ceiling(arena, at - 1));
            let refused = graph.room().take(level).unwrap_err();
            match &refused {
                Error::GraphFull {
                    arena: named,
                    held: named_held,
                    admitted,
                    needed,
                    ceiling,
                    call,
                    rebuild,
                } => {
                    assert_eq!(*named, ARENAS[arena]);
                    assert_eq!(*named_held, held[arena]);
                    assert_eq!(*admitted, 0);
                    assert_eq!(*needed, bound[arena]);
                    assert_eq!(*ceiling, at - 1);
                    assert!(!call, "nothing was admitted before it");
                    assert!(!rebuild, "the room refuses for an insertion");
                }
                other => panic!("level {level}: refused as {other:?}"),
            }
            assert_eq!(
                refused.to_string(),
                format!(
                    "The graph is full: its {} number {} of the {} it can address, and one \
                     more record can add up to {}. compact() rebuilds the graph without the \
                     nodes that removed and replaced records leave in it.",
                    ARENAS[arena],
                    held[arena],
                    at - 1,
                    bound[arena]
                )
            );
        }
    }
}

/// The room carries what the insertions it admitted may add, and the entry
/// level the highest of them leaves, to each insertion after them. One the
/// graph has room for alone, refused because of what was admitted before
/// it, is refused in the words that say the call is too large to place at
/// once, and adds nothing to the room, so the next insertion meets the room
/// as it was.
#[test]
fn an_insertion_the_call_leaves_no_room_for_is_refused_as_the_call_too_large() {
    let (graph, _) = sample();
    let top = graph.entry_level() as usize;
    let held = arenas(&graph);
    let room = graph.room();
    // Three insertions above the top, the first raising the entry level the
    // next two meet.
    let first = room.bound(top + 1, Some(top));
    let second = room.bound(top + 1, Some(top + 1));
    let alone = room.bound(top + 1, Some(top));
    let words = held[WORDS] + first[WORDS] + second[WORDS] + second[WORDS] - 1;
    let _lowered = Lowered::to(with_ceiling(WORDS, words));
    let mut room = graph.room();
    room.take(top + 1).unwrap();
    room.take(top + 1).unwrap();
    let refused = room.take(top + 1).unwrap_err();
    match &refused {
        Error::GraphFull {
            arena,
            held: named_held,
            admitted,
            needed,
            ceiling,
            call,
            rebuild,
        } => {
            assert_eq!(*arena, "upper list words");
            assert_eq!(*named_held, held[WORDS]);
            assert_eq!(*admitted, first[WORDS] + second[WORDS]);
            assert_eq!(*needed, second[WORDS]);
            assert_eq!(*ceiling, words);
            assert!(call, "the graph has room for the insertion alone");
            assert!(!rebuild, "the room refuses for an insertion");
        }
        other => panic!("refused as {other:?}"),
    }
    assert!(held[WORDS] + alone[WORDS] <= words);
    assert_eq!(
        refused.to_string(),
        format!(
            "This call is too large to place at once: the graph's upper list words number \
             {}, the records the call admitted before this one can add up to {} more and \
             this one up to {}, and the graph can address {}. The graph has room for the \
             record on its own, so add the records across smaller calls.",
            held[WORDS],
            first[WORDS] + second[WORDS],
            second[WORDS],
            words
        )
    );
    // The refusal took nothing: the room holds what the two before it may
    // add and the entry level they leave.
    assert_eq!(room.admitted[WORDS], first[WORDS] + second[WORDS]);
    assert_eq!(room.admitted[NODES], 2);
    assert_eq!(room.top, Some(top + 1));
}

/// A dump whose graph would take an arena past its ceiling builds no graph
/// and hands its rows back, naming the arena, for each of the four, and the
/// same dump reads at the ceilings the layout sets.
#[test]
fn a_dump_past_a_ceiling_hands_its_rows_back_naming_the_arena() {
    let (graph, store) = sample();
    let dir = tempfile::tempdir().unwrap();
    write_dump(
        &graph.dump_view(&store),
        GraphKind::L2,
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
    )
    .unwrap();
    let expected = Expected {
        kind: GraphKind::L2,
        dimension: 4,
        m: 4,
        ef_construction: 32,
        min_nodes: 400,
        max_origin_id: 399,
        version: 2,
    };
    let read = || {
        read_dump_at::<f32, L2Dist>(
            &FsDir::new(dir.path()),
            DUMP_FILENAME,
            &expected,
            L2Dist {},
            None,
            false,
        )
    };
    let loaded = match read() {
        Ok(DumpRead::Graph(graph, _)) => arenas(&graph),
        _ => panic!("the dump reads at the layout's ceilings"),
    };
    for arena in 0..ARENAS.len() {
        let _lowered = Lowered::to(with_ceiling(arena, loaded[arena] - 1));
        match read() {
            Ok(DumpRead::Rows { reason, .. }) => assert!(
                reason.starts_with("its topology region could not be built into a graph: ")
                    && reason.contains(&format!("the graph's {} would number ", ARENAS[arena]))
                    && reason
                        .ends_with(&format!(" and the graph can address {}", loaded[arena] - 1)),
                "{}: {reason}",
                ARENAS[arena]
            ),
            Ok(DumpRead::Graph(..)) => panic!("{}: the dump built a graph", ARENAS[arena]),
            Err(refusal) => panic!("{}: refused as {refusal:?}", ARENAS[arena]),
        }
    }
}
