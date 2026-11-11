//! The bound on a vector's norm, and a cosine vector whose squares leave the
//! range of an `f32`.
//!
//! Under l2, dot or l1 a record or a query whose Euclidean norm passes
//! `NORM_BOUND` is refused naming its norm, a query alone and in a batch, and
//! one at the bound is taken. Under cosine nothing is held to the bound, and a
//! vector whose squares overflow or underflow in `f32` is normalized in `f64`
//! to unit length, where every other vector takes the `f32` norm and division
//! it took before.

use zeusdb_vector_core::{norm_f64, Error, Screen};

use super::{Collection, Declaration, ParsedRecord};

const DIM: usize = 4;

/// 2^62 as an `f32`, and the next `f32` above it.
const AT_BOUND: f32 = f32::from_bits(0x5E80_0000);
const PAST_BOUND: f32 = f32::from_bits(0x5E80_0001);

/// An empty raw collection under `metric`.
fn collection(metric: &str) -> Collection {
    Collection::build(
        Declaration::validate(DIM, metric, 8, 40, 100, Vec::new()).unwrap(),
        None,
    )
}

fn bits(vector: &[f32]) -> Vec<u32> {
    vector.iter().map(|value| value.to_bits()).collect()
}

/// A vector whose `f32` norm is finite and not zero normalizes to the bits
/// the `f32` norm and division give. One whose squares overflow or underflow
/// normalizes to unit length in the direction it had, and an all-zero vector
/// is kept as it is, its signs of zero with it.
#[test]
fn a_cosine_vector_whose_squares_leave_the_f32_range_is_stored_at_unit_length() {
    let cosine = collection("cosine");
    let plain = vec![3.0f32, -4.0, 0.5, 1e-3];
    let norm: f32 = plain.iter().map(|x| x * x).sum::<f32>().sqrt();
    let expected: Vec<f32> = plain.iter().map(|x| x / norm).collect();
    assert_eq!(
        bits(&cosine.process_vector_for_space(plain)),
        bits(&expected)
    );

    for (label, vector) in [
        ("the largest", vec![f32::MAX; DIM]),
        ("one at the largest", vec![f32::MAX, 0.0, 0.0, 0.0]),
        ("1e30 of either sign", vec![1e30, -1e30, 1e30, -1e30]),
        ("1e-30", vec![1e-30, 2e-30, 0.0, -1e-30]),
        (
            "the least subnormal",
            vec![0.0, f32::from_bits(1), 0.0, 0.0],
        ),
    ] {
        let unit = cosine.process_vector_for_space(vector.clone());
        let length = norm_f64(&unit);
        assert!((length - 1.0).abs() < 1e-6, "{label}: length {length}");
        for (given, stored) in vector.iter().zip(&unit) {
            assert_eq!(
                (*given > 0.0, *given < 0.0),
                (*stored > 0.0, *stored < 0.0),
                "{label}: a value changed its sign"
            );
        }
        let query = cosine
            .validate_query(vector)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(bits(&query), bits(&unit), "{label}: the query");
    }

    let zeros = vec![0.0f32, -0.0, 0.0, 0.0];
    assert_eq!(
        bits(&cosine.process_vector_for_space(zeros.clone())),
        bits(&zeros)
    );
}

/// Under l2, dot and l1 a record or a query whose norm passes the bound is
/// refused naming its norm, a query alone and in a batch, and one at the
/// bound is taken and searched. Under cosine the same vectors are taken.
#[test]
fn a_vector_past_the_bound_is_refused_naming_its_norm() {
    let at = vec![AT_BOUND, 0.0, 0.0, 0.0];
    let past = vec![0.0, 0.0, -PAST_BOUND, 0.0];
    for metric in ["l2", "dot", "l1"] {
        let index = collection(metric);
        let records: Vec<ParsedRecord> = (0..8u8)
            .map(|i| ParsedRecord {
                id: format!("r{i}"),
                vector: vec![f32::from(i), 1.0, -1.0, 0.5],
                sparse: None,
                metadata: Default::default(),
            })
            .collect();
        let added = index.add_records(records, Vec::new(), false);
        assert_eq!(added.total_errors, 0, "{metric}: {:?}", added.errors);

        assert_eq!(index.screen_vector(&at), Screen::Admitted, "{metric}");
        assert_eq!(
            index.screen_vector(&past),
            Screen::PastBound {
                norm: f64::from(PAST_BOUND)
            },
            "{metric}"
        );

        let query = index.validate_query(at.clone()).unwrap();
        let params = index.search_params(3, None, false, None).unwrap();
        let page = index.search_one(&query, None, params).unwrap();
        assert_eq!(page.len(), 3, "{metric}");
        assert!(
            page.iter().all(|hit| hit.score().is_finite()),
            "{metric}: a score at the bound is not finite"
        );

        match index.validate_query(past.clone()) {
            Err(Error::SearchVectorPastBound { norm }) => {
                assert_eq!(norm, f64::from(PAST_BOUND), "{metric}")
            }
            other => panic!("{metric}: a query past the bound gave {other:?}"),
        }
        match index.search_batch(&[at.clone(), past.clone()], None, params) {
            Err(Error::BatchVectorPastBound { position, norm }) => {
                assert_eq!((position, norm), (1, f64::from(PAST_BOUND)), "{metric}")
            }
            other => panic!("{metric}: a batch past the bound gave {other:?}"),
        }
    }
    assert_eq!(
        Error::SearchVectorPastBound { norm: 1e19 }.to_string(),
        "Search vector has a Euclidean norm of 1e19 (must be at most 2^62, \
         4.611686018427388e18, under l2, dot or l1)"
    );
    assert_eq!(
        Error::BatchVectorPastBound {
            position: 2,
            norm: 1e19
        }
        .to_string(),
        "Vector 2 in batch has a Euclidean norm of 1e19 (must be at most 2^62, \
         4.611686018427388e18, under l2, dot or l1)"
    );

    let cosine = collection("cosine");
    assert_eq!(cosine.screen_vector(&past), Screen::Admitted);
    cosine.validate_query(past).unwrap();
}
