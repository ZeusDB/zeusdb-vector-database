//! The tests of version 2 of the graph dump. The version 1 reader's tests are
//! in `v1.rs`.

use super::*;
use crate::distance::{
    CosineDist, DistPQ, DotDist, Int8Dist, Int8Metric, L1Dist, L2Dist, PqMetric,
};
use crate::graph::levels::LevelGenerator;
use crate::int8::Int8Codec;
use crate::pq::PQ;
use crate::space::Position;
use crate::storage::FsDir;
use std::collections::BTreeMap;
use std::sync::Arc;

type Graph = MutableGraph<f32, CosineDist>;
type Store = VectorStore<f32>;

/// Deterministic values in `[-0.5, 0.5)`, spread enough that a graph has a
/// shape.
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

/// A graph of `records` points inserted under ids 0 upward.
fn sample_graph(records: usize, dim: usize, m: usize) -> (Graph, Store) {
    let scale = LevelGenerator::default_scale(m);
    let mut levels = LevelGenerator::new(scale, NB_LAYER_MAX as usize);
    let (mut graph, mut store) =
        MutableGraph::new(dim, m, 64, scale, records.max(1), CosineDist {}).unwrap();
    for (id, vector) in spread(records, dim, 0x2545_f491_4f6c_dd1d)
        .into_iter()
        .enumerate()
    {
        graph.insert(&mut store, &vector, id, &mut levels);
    }
    (graph, store)
}

/// What the reader is told of a graph `sample_graph` built: every node holds
/// a record and the largest id issued is the last.
fn expected_for(graph: &Graph, nodes: usize) -> Expected {
    Expected {
        kind: GraphKind::Cosine,
        dimension: graph.dim(),
        m: graph.m(),
        ef_construction: graph.ef_construction(),
        min_nodes: nodes,
        max_origin_id: nodes.saturating_sub(1),
        version: 2,
    }
}

/// The dump `graph` writes, as bytes.
fn dump_of<T, D>(graph: &MutableGraph<T, D>, store: &VectorStore<T>, kind: GraphKind) -> Vec<u8>
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    let dir = tempfile::tempdir().unwrap();
    write_dump(
        &graph.dump_view(store),
        kind,
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
    )
    .unwrap();
    std::fs::read(dir.path().join(DUMP_FILENAME)).unwrap()
}

/// `blob` written as the dump of a directory of its own.
fn holding(blob: &[u8]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(DUMP_FILENAME), blob).unwrap();
    dir
}

/// What reading the dump in `dir` gave, as the graph, the reason rows were
/// handed back, or the refusal.
fn outcome(dir: &std::path::Path, expected: &Expected, rows_only: bool) -> Outcome {
    match read_dump_at::<f32, CosineDist>(
        &FsDir::new(dir),
        DUMP_FILENAME,
        expected,
        CosineDist {},
        None,
        rows_only,
    ) {
        Ok(DumpRead::Graph(graph, store)) => Outcome::Graph(graph, store),
        Ok(DumpRead::Rows { rows, reason }) => Outcome::Rows(rows, reason),
        Err(refusal) => Outcome::Refused(refusal),
    }
}

enum Outcome {
    Graph(Box<Graph>, Store),
    Rows(ReadRows<f32>, String),
    Refused(DumpRefusal),
}

impl Outcome {
    fn refusal(self) -> DumpRefusal {
        match self {
            Outcome::Refused(refusal) => refusal,
            Outcome::Graph(..) => panic!("a dump that should have been refused was read"),
            Outcome::Rows(_, reason) => {
                panic!("a dump that should have been refused gave rows: {reason}")
            }
        }
    }

    fn rows(self) -> (ReadRows<f32>, String) {
        match self {
            Outcome::Rows(rows, reason) => (rows, reason),
            Outcome::Graph(..) => panic!("a dump that should have given its rows was read"),
            Outcome::Refused(refusal) => {
                panic!("a dump that should have given its rows was refused: {refusal:?}")
            }
        }
    }

    fn graph(self) -> (Graph, Store) {
        match self {
            Outcome::Graph(graph, store) => (*graph, store),
            Outcome::Rows(_, reason) => panic!("the dump gave its rows: {reason}"),
            Outcome::Refused(refusal) => panic!("the dump was refused: {refusal:?}"),
        }
    }
}

/// The whole graph as plain values, keyed by origin id: each node's layer,
/// rank, row bits and lists, every entry as its target's origin id and the
/// distance's bits, in list order.
#[allow(clippy::type_complexity)]
fn topology(
    graph: &Graph,
    store: &Store,
) -> BTreeMap<usize, (u8, i32, Vec<u32>, Vec<Vec<(usize, u32)>>)> {
    let point_ids = graph.point_ids();
    let mut out = BTreeMap::new();
    for node in 0..graph.nb_points() as u32 {
        let p_id = point_ids[node as usize];
        let values = store.get(node).iter().map(|v| v.to_bits()).collect();
        let adjacency = graph
            .neighbourhood_ids(store, node)
            .iter()
            .map(|list| list.iter().map(|&(id, d)| (id, d.to_bits())).collect())
            .collect();
        out.insert(
            graph.origin_id_of(node),
            (p_id.0, p_id.1, values, adjacency),
        );
    }
    out
}

/// Where each region of a version 2 dump starts, from its own header: the
/// identity, the rows, the topology and the trailer.
fn regions(blob: &[u8]) -> (usize, usize, usize, usize) {
    let identity = u64::from_le_bytes(take8(blob, 48)) as usize;
    let rows = u64::from_le_bytes(take8(blob, 56)) as usize;
    let topology = u64::from_le_bytes(take8(blob, 64)) as usize;
    let start = HEADER_BYTES;
    (
        start,
        start + identity,
        start + identity + rows,
        start + identity + rows + topology,
    )
}

/// Every region's checksum and the header's own made to agree with the bytes
/// the file now holds, which is what a random corruption cannot do and what
/// makes a forgery a question for the reader's rules rather than its sums.
fn repaired(mut blob: Vec<u8>) -> Vec<u8> {
    let (identity, rows, topology, trailer) = regions(&blob);
    let identity_sum = checksum_of(&blob[identity..rows]);
    let rows_sum = checksum_of(&blob[rows..topology]);
    let topology_sum = checksum_of(&blob[topology..trailer]);
    blob[80..88].copy_from_slice(&identity_sum.to_le_bytes());
    blob[88..96].copy_from_slice(&rows_sum.to_le_bytes());
    blob[96..104].copy_from_slice(&topology_sum.to_le_bytes());
    let header_sum = checksum_of(&blob[..HEADER_CHECKSUM_BYTES]);
    blob[HEADER_CHECKSUM_BYTES..HEADER_BYTES].copy_from_slice(&header_sum.to_le_bytes());
    blob
}

#[test]
fn a_round_trip_reproduces_every_node_and_every_edge() {
    let (built, built_store) = sample_graph(600, 12, 16);
    let dir = holding(&dump_of(&built, &built_store, GraphKind::Cosine));
    let (read, read_store) = outcome(dir.path(), &expected_for(&built, 600), false).graph();
    assert_eq!(read.nb_points(), built.nb_points());
    assert_eq!(read.m(), 16);
    assert_eq!(read.ef_construction(), built.ef_construction());
    assert_eq!(read.level_scale(), built.level_scale());
    assert_eq!(read.entry_point_id(), built.entry_point_id());
    assert_eq!(topology(&built, &built_store), topology(&read, &read_store));
    assert_eq!(read.in_degrees(), built.in_degrees());
}

#[test]
fn writing_the_same_graph_twice_gives_the_same_bytes() {
    let (built, built_store) = sample_graph(300, 8, 16);
    let first = dump_of(&built, &built_store, GraphKind::Cosine);
    let dir = holding(&first);
    let (read, read_store) = outcome(dir.path(), &expected_for(&built, 300), false).graph();
    assert_eq!(first, dump_of(&read, &read_store, GraphKind::Cosine));
}

#[test]
fn m_at_the_top_of_the_range_survives() {
    for m in [2usize, 16, 255, 256] {
        let (built, built_store) = sample_graph(120, 6, m);
        let dir = holding(&dump_of(&built, &built_store, GraphKind::Cosine));
        let (read, read_store) = outcome(dir.path(), &expected_for(&built, 120), false).graph();
        assert_eq!(read.m(), m);
        assert_eq!(topology(&built, &built_store), topology(&read, &read_store));
    }
}

/// The regions lie where the header says, each the length its contents take,
/// and the topology holds a two byte count a list and a four byte node index
/// an entry, with no distance.
#[test]
fn the_regions_lie_where_the_header_says_and_hold_no_distance() {
    let (built, built_store) = sample_graph(400, 7, 8);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let (identity, rows, topology_at, trailer) = regions(&blob);
    let nodes = built.nb_points();
    assert_eq!(rows - identity, LAYER_TABLE_BYTES + 4 * nodes);
    assert_eq!(topology_at - rows, nodes * 7 * 4);
    let mut topology_bytes = 0usize;
    for node in 0..nodes as u32 {
        let lists = built.neighbourhood_ids(&built_store, node);
        let count = lists
            .iter()
            .rposition(|list| !list.is_empty())
            .map_or(0, |last| last + 1);
        topology_bytes += 1 + lists[..count]
            .iter()
            .map(|list| 2 + 4 * list.len())
            .sum::<usize>();
    }
    assert_eq!(trailer - topology_at, topology_bytes);
    assert_eq!(blob.len(), trailer + TRAILER_BYTES);
    assert_eq!(u64::from_le_bytes(take8(&blob, trailer)), blob.len() as u64);
    assert_eq!(u64::from_le_bytes(take8(&blob, trailer + 8)), MAGIC);
    assert_eq!(u32::from_le_bytes(take4(&blob, 8)), 2);
}

#[test]
fn the_header_is_128_bytes_and_checksums_every_field() {
    let header = Header {
        kind: GraphKind::L1Pq,
        element: 2,
        layers: NB_LAYER_MAX,
        dimension: 96,
        m: 256,
        ef_construction: 200,
        nodes: 12345,
        entry_node: 7,
        level_scale: 0.25,
        identity_bytes: 999,
        rows_bytes: 77,
        topology_bytes: 5,
        file_bytes: 4242,
        identity_checksum: 1,
        rows_checksum: 2,
        topology_checksum: 3,
    };
    let raw = header.encode();
    assert_eq!(raw.len(), HEADER_BYTES);
    let back = Header::decode(&raw).unwrap();
    assert_eq!(back.kind, GraphKind::L1Pq);
    assert_eq!((back.m, back.nodes, back.entry_node), (256, 12345, 7));
    assert_eq!(back.level_scale, 0.25);
    assert_eq!(
        (back.identity_bytes, back.rows_bytes, back.topology_bytes),
        (999, 77, 5)
    );
    assert_eq!(back.file_bytes, 4242);
    assert_eq!(
        (
            back.identity_checksum,
            back.rows_checksum,
            back.topology_checksum
        ),
        (1, 2, 3)
    );
    for at in 0..HEADER_CHECKSUM_BYTES {
        let mut broken = raw;
        broken[at] ^= 0x01;
        assert!(
            Header::decode(&broken).is_err(),
            "byte {} passed unchecked",
            at
        );
    }
}

/// Every damage a file can take is refused naming the region it falls in, or,
/// in the topology, hands the rows back naming the topology.
#[test]
fn every_damaged_input_is_refused_naming_its_region() {
    let (built, built_store) = sample_graph(200, 6, 16);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let expected = expected_for(&built, 200);
    let (identity, rows, topology_at, _) = regions(&blob);
    let refused = |damaged: Vec<u8>| outcome(holding(&damaged).path(), &expected, false).refusal();
    let header = |r: DumpRefusal, needle: &str| {
        assert_eq!(r.region, DumpRegion::Header, "{:?}", r);
        assert!(r.detail.contains(needle), "{:?}", r);
    };
    header(refused(Vec::new()), "the smallest dump is 144");
    header(
        refused(blob[..HEADER_BYTES].to_vec()),
        "the smallest dump is 144",
    );
    header(refused(blob[..blob.len() / 2].to_vec()), "the file is");
    header(refused(blob[..blob.len() - 1].to_vec()), "the file is");
    let mut longer = blob.clone();
    longer.extend_from_slice(&[0u8; 16]);
    header(refused(longer), "the file is");
    let mut magic = blob.clone();
    magic[0] ^= 0xff;
    header(refused(magic), "does not open with the dump's magic");
    let mut version = blob.clone();
    version[8..12].copy_from_slice(&1u32.to_le_bytes());
    header(
        refused(version),
        "names format version 1, where a directory at format 5.0.0 holds version 2",
    );
    let mut field = blob.clone();
    field[32] ^= 0x01;
    header(refused(field), "its header fails its checksum");
    let mut trailing = blob.clone();
    let end = trailing.len() - 1;
    trailing[end] ^= 0x01;
    header(refused(trailing), "its trailer does not close the file");

    let mut layer_table = blob.clone();
    layer_table[identity] ^= 0x10;
    let r = refused(layer_table);
    assert_eq!(r.region, DumpRegion::Identity);
    assert_eq!(r.detail, "its identity region fails its checksum");

    let mut row = blob.clone();
    row[rows + 5] ^= 0x10;
    let r = refused(row);
    assert_eq!(r.region, DumpRegion::Rows);
    assert_eq!(r.detail, "its rows region fails its checksum");

    let mut list = blob.clone();
    list[topology_at + 3] ^= 0x10;
    let (handed, reason) = outcome(holding(&list).path(), &expected, false).rows();
    assert_eq!(reason, "its topology region fails its checksum");
    assert_eq!(
        handed.store.values(),
        built_store_in_dump_order(&built, &built_store).as_slice()
    );
}

/// The rows the dump of `graph` holds, in its layer then rank order.
fn built_store_in_dump_order(graph: &Graph, store: &Store) -> Vec<f32> {
    let point_ids = graph.point_ids();
    let mut nodes: Vec<u32> = (0..graph.nb_points() as u32).collect();
    nodes.sort_by_key(|&node| point_ids[node as usize]);
    nodes
        .iter()
        .flat_map(|&node| store.get(node).to_vec())
        .collect()
}

/// A damaged topology hands back every row and every origin id verified, in
/// the dump's order, and a rows-only read hands them back without reading
/// the topology's structure. Damage in the identity or the rows refuses both.
#[test]
fn a_damaged_topology_hands_back_the_rows_it_holds() {
    let (built, built_store) = sample_graph(300, 5, 8);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let expected = expected_for(&built, 300);
    let (_, _, topology_at, trailer) = regions(&blob);
    let in_order = built_store_in_dump_order(&built, &built_store);
    let point_ids = built.point_ids();
    let mut nodes: Vec<u32> = (0..built.nb_points() as u32).collect();
    nodes.sort_by_key(|&node| point_ids[node as usize]);
    let origins: Vec<usize> = nodes.iter().map(|&node| built.origin_id_of(node)).collect();

    // A target past the last node, with every checksum made to agree.
    let mut forged = blob.clone();
    let first_target = topology_at + 1 + 2;
    forged[first_target..first_target + 4].copy_from_slice(&5_000u32.to_le_bytes());
    let (rows, reason) = outcome(holding(&repaired(forged)).path(), &expected, false).rows();
    assert!(
        reason.starts_with("its topology region names node 5000 from node 0"),
        "{reason}"
    );
    assert_eq!(rows.store.values(), in_order.as_slice());
    assert_eq!(rows.origin_ids, origins);
    assert_eq!(rows.record_of.len(), 300);

    // Bytes past the last node.
    let mut longer = blob[..trailer].to_vec();
    longer.extend_from_slice(&[0u8; 3]);
    let topology_bytes = (trailer - topology_at + 3) as u64;
    longer[64..72].copy_from_slice(&topology_bytes.to_le_bytes());
    let file_bytes = (longer.len() + TRAILER_BYTES) as u64;
    longer[72..80].copy_from_slice(&file_bytes.to_le_bytes());
    longer.extend_from_slice(&file_bytes.to_le_bytes());
    longer.extend_from_slice(&MAGIC.to_le_bytes());
    let (_, reason) = outcome(holding(&repaired(longer)).path(), &expected, false).rows();
    assert_eq!(
        reason,
        "its topology region holds 3 bytes beyond its 300 nodes"
    );

    // A rows-only read of the whole dump, and of one whose topology fails.
    let (rows, reason) = outcome(holding(&blob).path(), &expected, true).rows();
    assert_eq!(reason, ROWS_ASKED_FOR);
    assert_eq!(rows.store.values(), in_order.as_slice());
    let mut broken = blob.clone();
    broken[topology_at] ^= 0x01;
    let (_, reason) = outcome(holding(&broken).path(), &expected, true).rows();
    assert_eq!(reason, ROWS_ASKED_FOR);
}

#[test]
fn an_absent_file_is_refused() {
    let (built, _) = sample_graph(50, 4, 16);
    let dir = tempfile::tempdir().unwrap();
    let r = outcome(dir.path(), &expected_for(&built, 50), false).refusal();
    assert_eq!(
        r,
        DumpRefusal::header("the directory holds no such file".to_string())
    );
}

#[test]
fn a_save_clears_the_files_an_earlier_release_left() {
    let dir = tempfile::tempdir().unwrap();
    for legacy in LEGACY_DUMP_FILENAMES {
        std::fs::write(dir.path().join(legacy), vec![7u8; 4096]).unwrap();
    }
    let (built, built_store) = sample_graph(100, 5, 16);
    write_dump(
        &built.dump_view(&built_store),
        GraphKind::Cosine,
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
    )
    .unwrap();
    let left: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(left, vec![DUMP_FILENAME.to_string()]);
}

/// A header this build wrote, carrying a node count nothing could hold, is
/// refused on the size its fields imply, before anything is sized from it.
#[test]
fn a_header_claiming_a_billion_points_allocates_nothing() {
    let (built, built_store) = sample_graph(80, 4, 16);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let expected = Expected {
        max_origin_id: usize::MAX,
        ..expected_for(&built, 80)
    };
    let forged = |edit: &dyn Fn(&mut Header)| {
        let mut raw = [0u8; HEADER_BYTES];
        raw.copy_from_slice(&blob[..HEADER_BYTES]);
        let mut header = Header::decode(&raw).unwrap();
        edit(&mut header);
        let mut damaged = header.encode().to_vec();
        damaged.extend_from_slice(&blob[HEADER_BYTES..]);
        outcome(holding(&damaged).path(), &expected, false)
            .refusal()
            .detail
    };
    let reason = forged(&|h| h.nodes = 1_000_000_000);
    assert!(
        reason.contains("where 1000000000 nodes need 4000000064"),
        "{reason}"
    );
    let reason = forged(&|h| {
        h.nodes = u32::MAX;
        h.identity_bytes = LAYER_TABLE_BYTES as u64 + 4 * u32::MAX as u64;
    });
    assert!(reason.contains("a rows region of"), "{reason}");
    let reason = forged(&|h| {
        h.topology_bytes = u64::MAX - 10;
    });
    assert!(reason.contains("regions no file could hold"), "{reason}");
}

/// An origin id past the counter, or one named for two nodes, is refused in
/// the identity region's words, every checksum made to agree.
#[test]
fn the_identity_region_is_held_to_the_counter_and_to_one_node_an_id() {
    let (built, built_store) = sample_graph(80, 4, 16);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let expected = expected_for(&built, 80);
    let (identity, ..) = regions(&blob);
    let first_id = identity + LAYER_TABLE_BYTES;
    let with_first_id = |id: u32| {
        let mut damaged = blob.clone();
        damaged[first_id..first_id + 4].copy_from_slice(&id.to_le_bytes());
        outcome(holding(&repaired(damaged)).path(), &expected, false).refusal()
    };
    let r = with_first_id(80);
    assert_eq!(r.region, DumpRegion::Identity);
    assert_eq!(
        r.detail,
        "its identity region names origin id 80 and config.json counted 79"
    );
    let second = u32::from_le_bytes(take4(&blob, first_id + 4));
    let r = with_first_id(second);
    assert_eq!(
        r.detail,
        format!(
            "its identity region names origin id {} for more than one node",
            second
        )
    );
    // The largest id the index issued is accepted.
    assert!(matches!(
        outcome(holding(&blob).path(), &expected, false),
        Outcome::Graph(..)
    ));
}

/// An id the mappings hold that no node carries is refused in the identity
/// region's words, and a node whose id they do not hold is stranded.
#[test]
fn the_identity_region_is_numbered_against_the_mappings() {
    let (built, built_store) = sample_graph(60, 4, 8);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let held: Vec<u32> = (0..60u32).filter(|id| id % 6 != 1).collect();
    let expected = Expected {
        min_nodes: held.len(),
        ..expected_for(&built, 60)
    };
    let dir = holding(&blob);
    let read_at = |held: &[u32]| {
        read_dump_at::<f32, CosineDist>(
            &FsDir::new(dir.path()),
            DUMP_FILENAME,
            &expected,
            CosineDist {},
            Some(held),
            false,
        )
    };
    match read_at(&held) {
        Ok(DumpRead::Graph(graph, _)) => {
            for node in 0..graph.nb_points() as u32 {
                let id = graph.origin_id_of(node) as u32;
                let want = match held.binary_search(&id) {
                    Ok(at) => at as u32,
                    Err(_) => Position::NONE,
                };
                assert_eq!(graph.record_of(node), want);
            }
        }
        _ => panic!("the dump did not read"),
    }
    let mut more = held.clone();
    more.push(61);
    match read_at(&more) {
        Err(r) => assert_eq!(
            r,
            DumpRefusal::identity(
                "its identity region names no node for internal id 61, which mappings.bin holds"
                    .to_string()
            )
        ),
        _ => panic!("an id no node carries was accepted"),
    }
}

/// A row holding a NaN or an infinity is refused naming the node and its id,
/// with every checksum made to agree, since no record can hold one.
#[test]
fn a_row_no_record_can_hold_is_refused_naming_its_node() {
    let (built, built_store) = sample_graph(40, 4, 8);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let expected = expected_for(&built, 40);
    let (identity, rows, ..) = regions(&blob);
    let third_row = rows + 2 * 4 * 4;
    let origin = u32::from_le_bytes(take4(&blob, identity + LAYER_TABLE_BYTES + 2 * 4));
    for value in [f32::NAN, f32::INFINITY] {
        let mut damaged = blob.clone();
        damaged[third_row + 4..third_row + 8].copy_from_slice(&value.to_le_bytes());
        let r = outcome(holding(&repaired(damaged)).path(), &expected, false).refusal();
        assert_eq!(r.region, DumpRegion::Rows);
        assert_eq!(
            r.detail,
            format!(
                "its rows region holds {} in the row of node 2, internal id {}",
                value, origin
            )
        );
    }
}

#[test]
fn a_dump_written_for_another_configuration_is_refused() {
    let (built, built_store) = sample_graph(120, 6, 16);
    let dir = holding(&dump_of(&built, &built_store, GraphKind::Cosine));
    let base = expected_for(&built, 120);
    let refuse = |expected: Expected| outcome(dir.path(), &expected, false).refusal().detail;
    assert!(refuse(Expected {
        kind: GraphKind::L2,
        ..base
    })
    .contains("the cosine graph and config.json declares l2"));
    assert!(refuse(Expected {
        dimension: 7,
        ..base
    })
    .contains("rows of 6 values where this index holds 7"));
    assert!(refuse(Expected { m: 17, ..base }).contains("m 16 and config.json declares 17"));
    assert!(refuse(Expected {
        ef_construction: 65,
        ..base
    })
    .contains("ef_construction 64"));
    assert!(refuse(Expected {
        min_nodes: 121,
        ..base
    })
    .contains("120 nodes and the index holds 121 records"));
    assert!(refuse(Expected {
        max_origin_id: 118,
        ..base
    })
    .contains("120 nodes and config.json counted 118"));
    let as_codes = read_dump_at::<u8, ByteDist>(
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
        &base,
        ByteDist,
        None,
        false,
    );
    match as_codes {
        Err(r) => assert!(
            r.detail.contains("element type 1 where this index holds 2"),
            "{r:?}"
        ),
        _ => panic!("a raw dump was read as codes"),
    }
}

/// A `u8` distance, only so the element type check has something to refuse.
struct ByteDist;

impl Distance<u8> for ByteDist {
    fn eval(&self, _va: &[u8], _vb: &[u8]) -> f32 {
        0.0
    }
}

/// The dump in `dir` read by the reader and by the reference.
fn read_both(dir: &std::path::Path, expected: &Expected) -> ((Graph, Store), (Graph, Store)) {
    let read = read_dump(&FsDir::new(dir), DUMP_FILENAME, expected, CosineDist {})
        .unwrap_or_else(|e| panic!("the reader refused: {}", e));
    let reference = read_dump_reference(&FsDir::new(dir), DUMP_FILENAME, expected, CosineDist {})
        .unwrap_or_else(|e| panic!("the reference refused: {}", e));
    (read, reference)
}

/// A store as its request and its values' bytes.
fn store_bytes(store: &Store) -> (usize, Vec<u8>) {
    let mut bytes = Vec::new();
    f32::encode(store.values(), &mut bytes);
    (store.memory_bytes(), bytes)
}

/// The reader builds the graph the points build, buffer for buffer, over
/// graphs of three layers and more carrying lists above their owners' levels.
#[test]
fn the_reader_builds_the_graph_the_points_build() {
    for (records, dim, m) in [(1_500usize, 6usize, 2usize), (2_000, 8, 4), (1_200, 12, 16)] {
        let (built, built_store) = sample_graph(records, dim, m);
        let dir = holding(&dump_of(&built, &built_store, GraphKind::Cosine));
        let ((read, read_store), (reference, reference_store)) =
            read_both(dir.path(), &expected_for(&built, records));
        assert!(read.above_level_edges() > 0, "m {}", m);
        assert_eq!(read.arrays(), reference.arrays(), "m {}", m);
        assert_eq!(
            store_bytes(&read_store),
            store_bytes(&reference_store),
            "m {}",
            m
        );
        assert_eq!(read.memory_bytes(), reference.memory_bytes(), "m {}", m);
        assert_eq!(
            topology(&built, &built_store),
            topology(&read, &read_store),
            "m {}",
            m
        );
    }
}

/// `parsed` written into a directory of its own.
fn written(parsed: &ParsedDump<f32>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_dump::<f32, _>(
        parsed,
        GraphKind::Cosine,
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
    )
    .unwrap();
    dir
}

/// The dump `graph` writes, parsed back, which written back is the same file.
fn parse_written(graph: &Graph, store: &Store, expected: &Expected) -> ParsedDump<f32> {
    let dir = holding(&dump_of(graph, store, GraphKind::Cosine));
    let parsed = parse_dump::<f32>(&FsDir::new(dir.path()), DUMP_FILENAME, expected).unwrap();
    let again = written(&parsed);
    assert_eq!(
        std::fs::read(dir.path().join(DUMP_FILENAME)).unwrap(),
        std::fs::read(again.path().join(DUMP_FILENAME)).unwrap(),
        "a parse written back is not the file it was parsed from"
    );
    parsed
}

/// A list longer than its layer's slots hands the rows back in the topology's
/// words, which the reference gives too, at layer zero and above.
#[test]
fn a_list_longer_than_its_layer_holds_hands_the_rows_back() {
    let (built, built_store) = sample_graph(300, 4, 2);
    let expected = expected_for(&built, 300);
    let parsed = parse_written(&built, &built_store, &expected);
    let layer_zero = parsed.points_by_layer[0].len();
    assert!(parsed.points_by_layer[1].len() > 4);
    let with_list = |layer: usize, count: usize| {
        let mut forged = parsed.clone();
        let point = &mut forged.points_by_layer[layer][0];
        let lists = point.neighbours.len().max(layer + 1);
        point.neighbours.resize(lists, Vec::new());
        point.neighbours[layer] = (1..=count)
            .map(|rank| LoadedEdge {
                target: PointId(layer as u8, rank as i32),
                distance: 0.0,
            })
            .collect();
        written(&forged)
    };
    for (layer, node, slots) in [(0usize, 0usize, 5usize), (1, layer_zero, 3)] {
        let full = with_list(layer, slots);
        let ((read, _), (reference, _)) = read_both(full.path(), &expected);
        assert_eq!(read.arrays(), reference.arrays(), "layer {}", layer);
        assert_eq!(read.neighbours_at(node as u32, layer).len(), slots);
        let past = with_list(layer, slots + 1);
        let want = format!(
            "its topology region gives node {} {} entries at layer {} and a list there holds {}",
            node,
            slots + 1,
            layer,
            slots
        );
        let (_, reason) = outcome(past.path(), &expected, false).rows();
        assert_eq!(reason, want);
        match read_dump_reference::<f32, CosineDist>(
            &FsDir::new(past.path()),
            DUMP_FILENAME,
            &expected,
            CosineDist {},
        ) {
            Err(reference) => assert_eq!(reference, want),
            Ok(_) => panic!("the reference read a list past its slots"),
        }
    }

    // A graph the constructor refuses hands the rows back in the topology's
    // words, and the reference refuses it in the same words.
    let mut wide = parsed.clone();
    wide.m = 257;
    let wide_dir = written(&wide);
    let at_257 = Expected { m: 257, ..expected };
    let (_, reason) = outcome(wide_dir.path(), &at_257, false).rows();
    assert!(
        reason.starts_with(
            "its topology region could not be built into a graph: max_nb_connection is between \
             1 and 256"
        ),
        "{reason}"
    );
    match read_dump_reference::<f32, CosineDist>(
        &FsDir::new(wide_dir.path()),
        DUMP_FILENAME,
        &at_257,
        CosineDist {},
    ) {
        Err(reference) => assert_eq!(reference, reason),
        Ok(_) => panic!("the reference built a graph of m 257"),
    }
}

/// One node inserted alone at the top layer, whose lists nothing fills,
/// loads to the graph its save wrote, holding a word a layer and no wide
/// list, and writes the same bytes again.
#[test]
fn a_node_alone_at_the_top_layer_loads_to_the_graph_its_save_wrote() {
    let top = NB_LAYER_MAX as usize - 1;
    let scale = LevelGenerator::default_scale(16);
    let (mut built, mut built_store) =
        MutableGraph::new(4, 16, 64, scale, 1, CosineDist {}).unwrap();
    let vector = [0.1f32, 0.2, 0.3, 0.4];
    let planned = built.plan_insertion(&built_store, &vector, top);
    built.install_insertion(&mut built_store, &vector, 1, 1, planned);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let dir = holding(&blob);
    let expected = Expected {
        max_origin_id: 1,
        ..expected_for(&built, 1)
    };
    let ((read, read_store), (reference, reference_store)) = read_both(dir.path(), &expected);
    assert_eq!(read.arrays(), reference.arrays());
    assert_eq!(store_bytes(&read_store), store_bytes(&reference_store));
    assert_eq!(read.memory_bytes(), reference.memory_bytes());
    assert_eq!(topology(&built, &built_store), topology(&read, &read_store));
    assert_eq!(read.entry_level() as usize, top);
    assert_eq!((read.nb_upper_lists(), read.nb_wide_lists()), (top, 0));
    assert_eq!(blob, dump_of(&read, &read_store, GraphKind::Cosine));
}

/// Lists at or below their owners' levels that the dump leaves empty load
/// as the reference loads them, whether or not an entry names their owner,
/// and the graph writes the same bytes again.
#[test]
fn empty_lists_at_or_below_a_level_load_as_the_reference_loads_them() {
    let (built, built_store) = sample_graph(300, 4, 2);
    let (mut named, mut unnamed) = (0usize, 0usize);
    for layer in 1..NB_LAYER_MAX as usize {
        let counted = built.counted_in_degree(layer);
        for node in 0..built.nb_points() as u32 {
            if layer <= built.level(node) as usize && built.neighbours_at(node, layer).is_empty() {
                if counted[node as usize] > 0 {
                    named += 1;
                } else {
                    unnamed += 1;
                }
            }
        }
    }
    assert!(
        named > 0 && unnamed > 0,
        "{} named and {} unnamed",
        named,
        unnamed
    );
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let dir = holding(&blob);
    let ((read, read_store), (reference, _)) = read_both(dir.path(), &expected_for(&built, 300));
    assert_eq!(read.arrays(), reference.arrays());
    assert_eq!(read.memory_bytes(), reference.memory_bytes());
    assert!(read.nb_wide_lists() < built.nb_wide_lists());
    assert_eq!(topology(&built, &built_store), topology(&read, &read_store));
    assert_eq!(blob, dump_of(&read, &read_store, GraphKind::Cosine));
}

/// One point, which carries no list, reads back to the graph the reference
/// builds and writes back to the same bytes.
#[test]
fn a_dump_of_one_point_round_trips() {
    let (built, built_store) = sample_graph(1, 5, 16);
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let dir = holding(&blob);
    let ((read, read_store), (reference, reference_store)) =
        read_both(dir.path(), &expected_for(&built, 1));
    assert_eq!((read.nb_points(), read.nb_edges()), (1, 0));
    assert_eq!(read.arrays(), reference.arrays());
    assert_eq!(store_bytes(&read_store), store_bytes(&reference_store));
    assert_eq!(blob, dump_of(&read, &read_store, GraphKind::Cosine));
}

/// A graph holding an empty list at or below its owner's level, and the
/// largest `m` at the smallest dimension, read back to the reference's graph
/// and write the same bytes.
#[test]
fn shapes_at_the_edges_read_back_and_write_the_same_bytes() {
    for (records, dim, m) in [(300usize, 4usize, 2usize), (300, 1, 256)] {
        let (built, built_store) = sample_graph(records, dim, m);
        let blob = dump_of(&built, &built_store, GraphKind::Cosine);
        let dir = holding(&blob);
        let ((read, read_store), (reference, reference_store)) =
            read_both(dir.path(), &expected_for(&built, records));
        assert_eq!(read.arrays(), reference.arrays());
        assert_eq!(store_bytes(&read_store), store_bytes(&reference_store));
        assert_eq!(topology(&built, &built_store), topology(&read, &read_store));
        assert_eq!(blob, dump_of(&read, &read_store, GraphKind::Cosine));
    }
}

/// A graph over scattered internal ids reads back with each node at the place
/// its origin id holds among the ids the mappings hold, and writes the same
/// bytes again.
#[test]
fn a_graph_over_scattered_ids_reads_back_at_the_positions_the_mappings_give() {
    let (dim, m, nodes) = (6, 8, 400usize);
    let far = |k: usize| 3 + k * 70_001;
    let scale = LevelGenerator::default_scale(m);
    let mut levels = LevelGenerator::new(scale, NB_LAYER_MAX as usize);
    let (mut built, mut built_store) =
        MutableGraph::new(dim, m, 64, scale, 16, CosineDist {}).unwrap();
    for (k, vector) in spread(nodes, dim, 0x9e37_79b9_7f4a_7c15).iter().enumerate() {
        built.insert(&mut built_store, vector, far(k), &mut levels);
    }
    let blob = dump_of(&built, &built_store, GraphKind::Cosine);
    let dir = holding(&blob);
    let expected = Expected {
        max_origin_id: far(nodes - 1),
        ..expected_for(&built, nodes)
    };
    let every: Vec<u32> = (0..nodes).map(|k| far(k) as u32).collect();
    let numbered = match read_dump_at::<f32, CosineDist>(
        &FsDir::new(dir.path()),
        DUMP_FILENAME,
        &expected,
        CosineDist {},
        Some(&every),
        false,
    ) {
        Ok(DumpRead::Graph(graph, store)) => (*graph, store),
        _ => panic!("the dump did not read"),
    };
    for node in 0..nodes as u32 {
        let k = (numbered.0.origin_id_of(node) - 3) / 70_001;
        assert_eq!(numbered.0.record_of(node) as usize, k, "node {node}");
    }
    assert_eq!(
        topology(&built, &built_store),
        topology(&numbered.0, &numbered.1)
    );
    assert_eq!(blob, dump_of(&numbered.0, &numbered.1, GraphKind::Cosine));
}

/// A graph of any kind written at version 2 and read back takes the same
/// insertions at the same levels as the graph that never left memory, and the
/// two write the same bytes after each.
#[test]
fn a_loaded_graph_of_every_kind_takes_insertions_as_the_unsaved_one_does() {
    let dim = 8;
    let unit = |v: &[f32]| {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<f32>>()
    };
    let corpus = spread(260, dim, 0x6a09_e667_f3bc_c909);
    let units: Vec<Vec<f32>> = corpus.iter().map(|v| unit(v)).collect();

    insertions_match(GraphKind::Cosine, &units, || CosineDist {});
    insertions_match(GraphKind::L2, &corpus, || L2Dist {});
    insertions_match(GraphKind::L1, &corpus, || L1Dist {});
    insertions_match(GraphKind::Dot, &corpus, || DotDist {});

    let pq = Arc::new(PQ::new(dim, 4, 4, 200, None));
    pq.train(&corpus).unwrap();
    let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
    let codes = pq.quantize_batch(&refs).unwrap();
    insertions_match(GraphKind::L2Pq, &codes, || {
        DistPQ::new(pq.clone(), PqMetric::SquaredL2)
    });
    insertions_match(GraphKind::CosinePq, &codes, || {
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
        insertions_match(kind, &rows, || Int8Dist::new(codec.clone(), metric));
    }
}

/// The first 200 of `rows` built into a graph and written, the dump read
/// back, and the other 60 inserted into both at chosen levels, with the two
/// dumps the same bytes after each insertion, the same inbound counts, and
/// the same evictions counted from the load on.
fn insertions_match<T, D>(kind: GraphKind, rows: &[Vec<T>], dist: impl Fn() -> D)
where
    T: DumpElement,
    D: Distance<T> + Send + Sync,
{
    let m = 4;
    let scale = LevelGenerator::default_scale(m);
    let mut levels = LevelGenerator::new(scale, NB_LAYER_MAX as usize);
    let width = rows[0].len();
    let (mut built, mut built_store) = MutableGraph::new(width, m, 48, scale, 300, dist()).unwrap();
    for (id, row) in rows[..200].iter().enumerate() {
        built.insert(&mut built_store, row, id, &mut levels);
    }
    let dir = holding(&dump_of(&built, &built_store, kind));
    let expected = Expected {
        kind,
        dimension: width,
        m,
        ef_construction: 48,
        min_nodes: 200,
        max_origin_id: 199,
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
        Ok(DumpRead::Rows { reason, .. }) => panic!("{:?} gave its rows: {}", kind, reason),
        Err(refusal) => panic!("{:?} was refused: {:?}", kind, refusal),
    };
    assert_eq!(read.in_degrees(), built.in_degrees(), "{:?}", kind);
    let (built_guard, read_guard) = (built.guard_stats(), read.guard_stats());
    let top = built.entry_level() as usize;
    for (k, row) in rows[200..].iter().enumerate() {
        let level = (k * 5) % (top + 3);
        for (graph, store) in [(&mut built, &mut built_store), (&mut read, &mut read_store)] {
            let planned = graph.plan_insertion(store, row, level);
            graph.install_insertion(store, row, 200 + k, (200 + k) as u32, planned);
        }
        assert_eq!(
            dump_of(&built, &built_store, kind),
            dump_of(&read, &read_store, kind),
            "{:?} after insertion {} at level {}",
            kind,
            k,
            level
        );
    }
    assert_eq!(read.in_degrees(), built.in_degrees(), "{:?}", kind);
    let evicted = |now: (u64, u64, u64), then: (u64, u64, u64)| {
        (now.0 - then.0, now.1 - then.1, now.2 - then.2)
    };
    assert_eq!(
        evicted(read.guard_stats(), read_guard),
        evicted(built.guard_stats(), built_guard),
        "{:?}",
        kind
    );
}
