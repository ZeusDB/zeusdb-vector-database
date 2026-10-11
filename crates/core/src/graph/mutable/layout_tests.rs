//! The layout of the struct the traversal borrows, pinned.
//!
//! The traversal reads a fixed set of fields of [`MutableGraph`] through one
//! reference, on every candidate and every neighbour entry, and how the
//! compiled loop holds that reference depends on where those fields sit. A
//! change to the struct's field types or their order moves them, and the cost
//! of that move shows in no test of what a search returns. This test holds the
//! size of the struct and the offset of every field the traversal reads to the
//! figures recorded for each of the eight graphs, so such a change fails here.
//!
//! Three of the eight share one type, being the product quantized graphs, and
//! each is listed so that the table reads one row a graph.

use super::MutableGraph;
use crate::distance::{CosineDist, DistPQ, DotDist, Int8Dist, L1Dist, L2Dist};
use std::mem::{offset_of, size_of};

/// One graph's figures: the struct's size, then the offsets of `m`, `entry`,
/// `entry_level`, `layer_counts`, `origin_ids`, `record_of`, `base_targets`,
/// `base_len`, `upper_first`, `upper_span`, `upper_word`, `wide_at`,
/// `wide_len`, `upper_targets` and `dist_f`.
type Figures = (usize, [usize; 15]);

macro_rules! figures {
    ($t:ty) => {
        (
            size_of::<$t>(),
            [
                offset_of!($t, m),
                offset_of!($t, entry),
                offset_of!($t, entry_level),
                offset_of!($t, layer_counts),
                offset_of!($t, origin_ids),
                offset_of!($t, record_of),
                offset_of!($t, base_targets),
                offset_of!($t, base_len),
                offset_of!($t, upper_first),
                offset_of!($t, upper_span),
                offset_of!($t, upper_word),
                offset_of!($t, wide_at),
                offset_of!($t, wide_len),
                offset_of!($t, upper_targets),
                offset_of!($t, dist_f),
            ],
        )
    };
}

/// A raw graph, whose distance holds nothing.
const RAW: Figures = (
    440,
    [
        320, 432, 436, 344, 0, 48, 72, 96, 144, 168, 192, 216, 240, 288, 437,
    ],
);

/// A quantized graph, whose distance holds a reference counted codebook or
/// codec and the metric.
const QUANTIZED: Figures = (
    456,
    [
        336, 448, 452, 360, 0, 48, 72, 96, 144, 168, 192, 216, 240, 288, 312,
    ],
);

/// The struct the traversal borrows has the layout recorded for every graph.
#[test]
fn the_struct_the_traversal_borrows_keeps_its_layout() {
    let graphs: [(&str, Figures, Figures); 8] = [
        ("cosine", figures!(MutableGraph<f32, CosineDist>), RAW),
        ("l2", figures!(MutableGraph<f32, L2Dist>), RAW),
        ("l1", figures!(MutableGraph<f32, L1Dist>), RAW),
        ("dot", figures!(MutableGraph<f32, DotDist>), RAW),
        ("cosine pq", figures!(MutableGraph<u8, DistPQ>), QUANTIZED),
        ("l2 pq", figures!(MutableGraph<u8, DistPQ>), QUANTIZED),
        ("l1 pq", figures!(MutableGraph<u8, DistPQ>), QUANTIZED),
        ("int8", figures!(MutableGraph<i8, Int8Dist>), QUANTIZED),
    ];
    for (name, found, recorded) in graphs {
        assert_eq!(
            found, recorded,
            "the {name} graph's struct has moved: size and offsets {found:?}, recorded {recorded:?}"
        );
    }
}
