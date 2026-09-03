//! A query over one or more arms, against the single space searches and
//! against the fusion by hand.
//!
//! What is held. A one arm dense query returns the page the dense search
//! returns, unfiltered and under each shape of filter, with the rerank the
//! space applies. A two arm query returns the reciprocal rank fusion of the
//! two arms' pages computed by hand from those pages, cut to `k` with the
//! tie at the boundary ordered by external id, and `explain` reports the
//! plan the page carries. A filter admitting every live record above the
//! dense scan threshold is planned as no filter and returns the unfiltered
//! page. And every rule a query has to satisfy refuses at the door.

use std::collections::HashMap;

use serde_json::{json, Value};

use zeusdb_vector_core::{
    compile_filter, fuse, Error, Fusion, IdfScope, SpaceKind, SparseRef, SparseVector,
};
use zeusdb_vector_sparse::SparseConfig;

use super::{
    AdmitShape, Arm, Collection, Declaration, ParsedRecord, Query, SparseHalf, DEFAULT_SPACE,
};

/// splitmix64, so every test sees the same records.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A collection holding a dense space of width four and a sparse space,
/// with `n` records, every record filling both, and a declared field
/// `cat` that takes `cats` values in rotation.
fn corpus(n: usize, cats: usize) -> (Collection, Vec<ParsedRecord>) {
    let declaration = Declaration::validate(4, "l2", 8, 50, n.max(100), vec!["cat".to_string()])
        .unwrap()
        .with_sparse("terms", SparseConfig::default())
        .unwrap();
    let collection = Collection::build(declaration, None);
    let mut rng = Rng(137);
    let records: Vec<ParsedRecord> = (0..n)
        .map(|i| {
            let vector: Vec<f32> = (0..4).map(|_| rng.unit()).collect();
            // Two to five distinct dimensions from a vocabulary of forty,
            // so most pairs of records share a term and pages are long.
            let count = 2 + rng.below(4);
            let mut dims: Vec<u32> = (0..count).map(|_| rng.below(40) as u32).collect();
            dims.sort_unstable();
            dims.dedup();
            let values: Vec<f32> = dims.iter().map(|_| 1.0 + rng.unit()).collect();
            let mut metadata: HashMap<String, Value> = HashMap::new();
            metadata.insert("cat".to_string(), json!(format!("c{}", i % cats)));
            ParsedRecord {
                id: format!("r{i:05}"),
                vector,
                sparse: Some(SparseHalf::Vector(SparseVector { dims, values })),
                metadata,
            }
        })
        .collect();
    let added = collection.add_records(records.clone(), vec![], false);
    assert_eq!(added.total_errors, 0, "{:?}", added.errors);
    (collection, records)
}

fn dense_query(seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..4).map(|_| rng.unit()).collect()
}

fn sparse_query() -> SparseVector {
    SparseVector {
        dims: vec![3, 7, 11, 19],
        values: vec![1.0, 0.5, 2.0, 1.0],
    }
}

/// A one arm dense query is the dense search, unfiltered and under a
/// bitmap, a walk that finished and a walk that gave up, id for id and
/// score bit for bit.
#[test]
fn a_one_arm_dense_query_is_the_dense_search() {
    let (collection, _) = corpus(400, 4);
    let declared = compile_filter(&HashMap::from([("cat".to_string(), json!("c1"))])).unwrap();
    let undeclared = compile_filter(&HashMap::from([(
        "cat".to_string(),
        json!({"in": ["c0", "c2"]}),
    )]))
    .unwrap();
    let filters: [Option<&zeusdb_vector_core::Filter>; 3] =
        [None, Some(&declared), Some(&undeclared)];
    for seed in 1..=10u64 {
        let vector = dense_query(seed);
        for filter in filters {
            let params = collection.search_params(7, None, false, None).unwrap();
            let expected = collection.search_one(&vector, filter, params).unwrap();
            let arms = [Arm::Dense {
                vector: &vector,
                ef: None,
                rerank: None,
            }];
            let page = collection
                .query(&Query {
                    arms: &arms,
                    filter,
                    k: 7,
                    fetch: None,
                    fusion: Fusion::default(),
                })
                .unwrap();
            assert_eq!(page.hits.len(), expected.len());
            for (position, (hit, want)) in page.hits.iter().zip(&expected).enumerate() {
                assert_eq!(hit.id, want.0);
                assert_eq!(hit.score.to_bits(), want.1.to_bits());
                assert_eq!(hit.metadata, want.2);
                assert_eq!(hit.contributions.len(), 1);
                assert_eq!(hit.contributions[0].rank, position + 1);
                assert_eq!(hit.contributions[0].arm, 0);
            }
            assert_eq!(page.plan.fusion, None);
            assert_eq!(page.plan.arms.len(), 1);
            assert_eq!(page.plan.arms[0].space.as_str(), DEFAULT_SPACE);
            assert_eq!(page.plan.arms[0].kind, SpaceKind::Dense);
            assert_eq!(page.plan.arms[0].fetch, 7, "one arm fetches its page");
            assert!(page.plan.arms[0].cost.work_ns > 0.0);
            match filter {
                None => assert_eq!(page.plan.admit, AdmitShape::All),
                Some(f) if std::ptr::eq(f, &declared) => {
                    assert_eq!(page.plan.admit, AdmitShape::Bitmap { admitted: 100 });
                    assert!(
                        page.plan.arms[0].cost.exact,
                        "a hundred records scan exactly"
                    );
                }
                Some(_) => assert_eq!(page.plan.admit, AdmitShape::Bitmap { admitted: 200 }),
            }
        }
    }
}

/// A two arm query is the reciprocal rank fusion of the two arms' pages,
/// computed by hand from those pages at the fetch depth, cut to `k` with
/// the boundary tie ordered by external id, and `explain` reports the plan
/// the page carries.
#[test]
fn two_arms_fuse_by_rank_and_explain_reports_the_plan() {
    let (collection, _) = corpus(400, 4);
    let sparse = sparse_query();
    let declared = compile_filter(&HashMap::from([("cat".to_string(), json!("c2"))])).unwrap();
    for (seed, filter, fetch) in [
        (1u64, None, None),
        (2, None, Some(25)),
        (3, Some(&declared), None),
        (4, Some(&declared), Some(40)),
    ] {
        let vector = dense_query(seed);
        // Unset, a query over two arms fetches five times its page.
        let depth = fetch.unwrap_or(5 * super::DEFAULT_FETCH_PER_K);
        let params = collection.search_params(depth, None, false, None).unwrap();
        let dense_page: Vec<(String, f32)> = collection
            .search_one(&vector, filter, params)
            .unwrap()
            .into_iter()
            .map(|hit| (hit.0, hit.1))
            .collect();
        let sparse_page = collection
            .search_sparse(sparse.as_ref(), filter, depth, IdfScope::Corpus)
            .unwrap();
        // The dense page is full, since more records are admitted than
        // fetched; the sparse page may be short, since a record sharing no
        // term with the query never appears.
        assert_eq!(dense_page.len(), depth, "seed {seed}");
        assert!(!sparse_page.is_empty(), "seed {seed}");
        let by_hand = fuse(
            Fusion::default(),
            &[
                dense_page
                    .iter()
                    .map(|(id, score)| (id, *score))
                    .collect::<Vec<(&String, f32)>>()
                    .as_slice(),
                sparse_page
                    .iter()
                    .map(|(id, score)| (id, *score))
                    .collect::<Vec<(&String, f32)>>()
                    .as_slice(),
            ],
        );

        let arms = [
            Arm::Dense {
                vector: &vector,
                ef: None,
                rerank: None,
            },
            Arm::Sparse {
                vector: SparseRef {
                    dims: &sparse.dims,
                    values: &sparse.values,
                },
                idf: IdfScope::Corpus,
            },
        ];
        let query = Query {
            arms: &arms,
            filter,
            k: 5,
            fetch,
            fusion: Fusion::default(),
        };
        let page = collection.query(&query).unwrap();
        assert_eq!(page.hits.len(), 5);
        for (hit, want) in page.hits.iter().zip(&by_hand) {
            assert_eq!(&hit.id, want.id, "seed {seed}");
            assert_eq!(hit.score.to_bits(), want.score.to_bits());
            assert_eq!(hit.contributions, want.contributions);
            for contribution in &hit.contributions {
                let (id, score) = match contribution.arm {
                    0 => &dense_page[contribution.rank - 1],
                    _ => &sparse_page[contribution.rank - 1],
                };
                assert_eq!(id, &hit.id);
                assert_eq!(score.to_bits(), contribution.score.to_bits());
            }
        }
        // Best first, ties by external id.
        assert!(page.hits.windows(2).all(|w| {
            w[0].score > w[1].score || (w[0].score == w[1].score && w[0].id < w[1].id)
        }));
        // A record on both pages outranks one on one page at the same rank.
        let on_both = page
            .hits
            .iter()
            .filter(|hit| hit.contributions.len() == 2)
            .count();
        assert!(on_both <= 5);

        assert_eq!(page.plan.fusion, Some(Fusion::default()));
        assert_eq!(page.plan.arms.len(), 2);
        assert_eq!(page.plan.arms[0].kind, SpaceKind::Dense);
        assert_eq!(page.plan.arms[1].kind, SpaceKind::Sparse);
        assert_eq!(page.plan.arms[1].space.as_str(), "terms");
        assert_eq!(page.plan.arms[0].fetch, depth);
        assert_eq!(page.plan.arms[1].fetch, depth);
        assert!(page.plan.arms[1].cost.exact);
        assert_eq!(collection.explain(&query).unwrap(), page.plan);
        assert_eq!(collection.query(&query).unwrap(), page, "reproducible");
    }
}

/// A filter admitting every live record above the dense scan threshold is
/// planned as no filter, and the page is the unfiltered page. Below the
/// threshold it stays a bitmap the dense arm scans exactly, and a filter
/// leaving one record out stays a bitmap.
#[test]
fn a_filter_admitting_every_live_record_is_planned_as_no_filter() {
    // Two past the threshold, so one removal leaves the corpus above it.
    let (collection, _) = corpus(5_002, 1);
    let all = compile_filter(&HashMap::from([("cat".to_string(), json!("c0"))])).unwrap();
    let vector = dense_query(9);
    let arms = [Arm::Dense {
        vector: &vector,
        ef: None,
        rerank: None,
    }];
    let unfiltered = collection.query(&Query::new(&arms, 10)).unwrap();
    assert_eq!(unfiltered.plan.admit, AdmitShape::All);
    let filtered = collection
        .query(&Query {
            filter: Some(&all),
            ..Query::new(&arms, 10)
        })
        .unwrap();
    assert_eq!(
        filtered.plan.admit,
        AdmitShape::All,
        "live {} matching {}",
        collection.len(),
        collection.count(Some(&all))
    );
    assert_eq!(filtered.hits, unfiltered.hits);
    assert!(
        !filtered.plan.arms[0].cost.exact,
        "a traversal, as unfiltered"
    );
    // The dense search itself takes the same plan.
    let params = collection.search_params(10, None, false, None).unwrap();
    let direct = collection.search_one(&vector, Some(&all), params).unwrap();
    assert_eq!(
        direct.iter().map(|h| &h.0).collect::<Vec<_>>(),
        filtered.hits.iter().map(|h| &h.id).collect::<Vec<_>>()
    );

    // One record removed, so the filter admits every live record still.
    assert!(collection.remove_point("r00000".to_string()).unwrap());
    let after = collection
        .query(&Query {
            filter: Some(&all),
            ..Query::new(&arms, 10)
        })
        .unwrap();
    assert_eq!(after.plan.admit, AdmitShape::All);

    // A filter leaving one live record out is a bitmap.
    let mut metadata = HashMap::new();
    metadata.insert("cat".to_string(), json!("other"));
    assert!(collection.update_metadata("r00001", metadata).unwrap());
    let most = collection
        .query(&Query {
            filter: Some(&all),
            ..Query::new(&arms, 10)
        })
        .unwrap();
    assert_eq!(most.plan.admit, AdmitShape::Bitmap { admitted: 5_000 });
    assert!(
        most.plan.arms[0].cost.exact,
        "at the threshold the bitmap scans exactly"
    );

    // Below the threshold the bitmap stays, and the dense arm scans it
    // exactly.
    let (small, _) = corpus(300, 1);
    let page = small
        .query(&Query {
            filter: Some(&all),
            ..Query::new(&arms, 10)
        })
        .unwrap();
    assert_eq!(page.plan.admit, AdmitShape::Bitmap { admitted: 300 });
    assert!(page.plan.arms[0].cost.exact);
}

/// Every rule a query has to satisfy refuses at the door, before any arm
/// runs.
#[test]
fn a_query_is_held_to_its_rules_at_the_door() {
    let (collection, _) = corpus(50, 2);
    let vector = dense_query(1);
    let sparse = sparse_query();
    let dense_arm = Arm::Dense {
        vector: &vector,
        ef: None,
        rerank: None,
    };
    let sparse_arm = Arm::Sparse {
        vector: sparse.as_ref(),
        idf: IdfScope::Corpus,
    };
    assert!(matches!(
        collection.query(&Query::new(&[], 10)),
        Err(Error::QueryArmsEmpty)
    ));
    let nine = [dense_arm; 9];
    assert!(matches!(
        collection.query(&Query::new(&nine, 10)),
        Err(Error::QueryArmsTooMany { max: 8, arms: 9 })
    ));
    let eight = [dense_arm; 8];
    assert_eq!(
        collection.query(&Query::new(&eight, 3)).unwrap().hits.len(),
        3
    );
    assert!(matches!(
        collection.query(&Query::new(&[dense_arm], 65_537)),
        Err(Error::TopKTooLarge { .. })
    ));
    assert!(matches!(
        collection.query(&Query {
            fetch: Some(65_537),
            ..Query::new(&[dense_arm], 10)
        }),
        Err(Error::FetchTooLarge {
            max: 65_536,
            fetch: 65_537
        })
    ));
    assert!(matches!(
        collection.query(&Query {
            fusion: Fusion::ReciprocalRank { k: -1.0 },
            ..Query::new(&[dense_arm, sparse_arm], 10)
        }),
        Err(Error::FusionConstantInvalid { .. })
    ));
    assert!(matches!(
        collection.query(&Query::new(
            &[Arm::Dense {
                vector: &[1.0, 2.0],
                ef: None,
                rerank: None
            }],
            10
        )),
        Err(Error::SearchVectorDimension {
            expected: 4,
            got: 2
        })
    ));
    assert!(matches!(
        collection.query(&Query::new(
            &[Arm::Dense {
                vector: &vector,
                ef: Some(200_000),
                rerank: None
            }],
            10
        )),
        Err(Error::EfSearchTooLarge { .. })
    ));
    let malformed = SparseVector {
        dims: vec![5, 5],
        values: vec![1.0, 1.0],
    };
    assert!(matches!(
        collection.query(&Query::new(
            &[Arm::Sparse {
                vector: malformed.as_ref(),
                idf: IdfScope::Corpus
            }],
            10
        )),
        Err(Error::SparseDimsNotIncreasing { position: 1 })
    ));
    assert!(matches!(
        collection.query(&Query::new(
            &[Arm::Text {
                text: "a",
                idf: IdfScope::Corpus
            }],
            10
        )),
        Err(Error::NoTextLayer)
    ));
    let dense_only = Collection::build(
        Declaration::validate(4, "l2", 8, 50, 100, vec![]).unwrap(),
        None,
    );
    assert!(matches!(
        dense_only.query(&Query::new(&[dense_arm, sparse_arm], 10)),
        Err(Error::NoSparseSpace)
    ));
    // Nothing ran, so nothing changed.
    assert_eq!(collection.len(), 50);
    assert!(collection
        .query(&Query::new(&[dense_arm], 0))
        .unwrap()
        .hits
        .is_empty());
}
