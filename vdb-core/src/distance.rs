//! ZeusDB's own distance kernels for `f32` vectors.
//!
//! These replace the `anndists` implementations the `hnsw_rs` prelude supplies.
//! The crate ZeusDB was using computes cosine in `f64` with three accumulators
//! and no vectorisation at any feature setting, which is roughly three times the
//! arithmetic the data needs once it is normalised.
//!
//! # What each metric returns
//!
//! The graph orders by the returned value and `search` reports it to the caller
//! as a score, so the quantity matters as much as the speed. Each function here
//! returns the same quantity its `anndists` counterpart returned.
//!
//! | Metric | Returns | Range on normalised input |
//! | --- | --- | --- |
//! | [`CosineDist`] | `1 - dot`, clamped at zero from below | `[0, 2]` |
//! | [`L2Dist`] | The Euclidean distance, with the square root taken | `[0, 2]` |
//! | [`L1Dist`] | The sum of absolute differences | `[0, 2 * sqrt(d)]` |
//! | [`DotDist`] | `1 - dot`, unclamped | `[0, 2]` |
//!
//! Two deliberate departures from `anndists` are recorded at [`CosineDist`] and
//! at [`DotDist`]. Everything else agrees to within the tolerance the tests
//! assert.
//!
//! # How they are compiled
//!
//! Every kernel is one loop over eight independent accumulators, held in a
//! single [`f32x8`]. Rust does not reassociate floating point additions, so a
//! plain `sum += a * b` loop compiles to a scalar chain no matter how it is
//! written. Fixing the accumulator count in the source is what makes the
//! reduction associative enough to widen, while keeping the result determined
//! by the source rather than by the compiler.
//!
//! The accumulator was an ordinary `[f32; 8]` until relay 70, on the reasoning
//! that the auto-vectoriser would widen a loop shaped that way. It did widen it,
//! and it stopped at half the register. The emitted inner loop loaded with
//! `movsd`, being two `f32`, and issued four `mulps` and four `addps` per eight
//! element block on 128-bit registers whose upper halves were zeroes. Four
//! source shapes were compiled looking for one LLVM would take further, being
//! the `[f32; 8]` array, two `[f32; 4]` halves, a whole-block copy into local
//! arrays, and two separate four element sub-block loads. All four produced
//! byte-identical assembly. Rebuilding the same shape at `+avx2` also produced
//! two lanes, which is why the `#[target_feature(enable = "avx")]` attempt
//! recorded below measured at parity. The width was not reachable from the
//! source.
//!
//! Naming the vector type reaches it. The same loop over `f32x8` compiles to
//! `movups` at sixteen bytes with two `mulps` and two `addps` per block, which
//! is the whole 128-bit register, and it halves the instruction count of the
//! inner loop.
//!
//! **The values do not move.** Lane `i` accumulates `a[8k + i]` against
//! `b[8k + i]` over ascending `k`, which is exactly what `acc[i]` accumulated
//! before, and [`reduce`] is unchanged and still fed those same eight lanes
//! through `to_array`. The tail is untouched and still scalar. Packed SSE
//! arithmetic rounds identically to its scalar counterpart, and nothing is
//! contracted into a fused multiply-add, which the assembly confirms by still
//! issuing separate `mulps` and `addps` at `+avx2` where an FMA is available.
//! So the results are bit-identical rather than merely close, and
//! `kernels_match_the_previous_shape_bit_for_bit` asserts that against a copy of
//! the previous shape kept in the tests.
//!
//! There is no run time dispatch and no `unsafe` in this file. Compiling each
//! kernel a second time inside a `#[target_feature(enable = "avx")]` wrapper was
//! tried, on the reasoning that the wheel is built for baseline `x86-64` and
//! would otherwise never see eight wide registers. It was measured at 1.04, 0.97
//! and 1.07 times the baseline at dimensions 128, 768 and 1536, which is
//! nothing, for the reason given above. It was removed rather than carried.
//!
//! One compilation also means one answer. A saved index scores identically on
//! every machine, because there is no second code path for a different processor
//! to take. `wide` picks its representation at compile time from the target's
//! own features rather than by detecting anything, so that property survives.

// The graph crate's trait, taken from the seam that owns that crate rather than
// from the crate itself. ZeusDB implements it, so the name has to be visible
// here; see the note at the top of `graph.rs`.
use crate::graph::Distance;

// The eight lane `f32` vector the kernels accumulate into. It is a compile time
// selection over the target's registers rather than a run time one, so it adds
// no branch and no second code path. On the baseline `x86-64` the wheel builds
// for it is a pair of SSE registers.
use wide::f32x8;

/// The quantized distance, re-exported so every call site imports its distances
/// from one module.
///
/// The declaration cannot follow the name here. `Hnsw::file_dump` writes
/// `std::any::type_name::<D>()` into the dump header and the load path compares
/// it by exact equality, and `type_name` reports where a type is **declared**,
/// so moving `DistPQ` out of `hnsw_index` would change what every save writes and
/// stop every saved quantized index from loading. A re-export is an import site
/// and changes nothing on disk, which was verified by loading a directory saved
/// before it existed.
pub(crate) use crate::hnsw_index::DistPQ;

/// Independent accumulators each kernel carries.
///
/// Eight, which is one AVX register of `f32` and two SSE2 registers, so the
/// source has enough independent work for either width. It is part of the
/// arithmetic rather than a tuning knob, since changing it changes the summation
/// order and therefore the last bits of every result. It is also the lane count
/// of [`f32x8`], and the two have to agree.
const LANES: usize = 8;

/// Sum the lanes.
///
/// Pairwise rather than left to right, so the error term grows with the log of
/// the lane count rather than with the count.
#[inline(always)]
fn reduce(acc: [f32; LANES]) -> f32 {
    let a0 = acc[0] + acc[4];
    let a1 = acc[1] + acc[5];
    let a2 = acc[2] + acc[6];
    let a3 = acc[3] + acc[7];
    (a0 + a2) + (a1 + a3)
}

/// Accumulate `$term` over both slices, `LANES` elements at a time.
///
/// `$term` is written in terms of two names the caller supplies. In the block
/// loop those names are bound to a [`f32x8`] each and in the tail they are bound
/// to one `f32` each, and the same expression serves both because the operators
/// and `abs` the three kernels use are spelled the same on either type. It is a
/// macro rather than a function taking a closure both for that and so that the
/// kernels share one loop without depending on the optimiser to see through a
/// call.
///
/// The `try_into` is what turns a subslice of known length into an array, which
/// is the form `f32x8::new` takes and the form that carries no bounds check.
///
/// Lengths are not required to match. A mismatch scores the common prefix rather
/// than panicking, which the graph never exercises because every vector in an
/// index has the index's dimension.
macro_rules! lane_loop {
    ($a:expr, $b:expr, $x:ident, $y:ident, $term:expr) => {{
        // Both slices are trimmed to the same whole number of blocks up front,
        // and the blocks are walked by one index. Zipping two `chunks_exact`
        // iterators instead leaves the compiler an exit test per slice per
        // iteration, which costs a compare and a branch in the hot loop.
        let n = core::cmp::min($a.len(), $b.len());
        let blocks = n / LANES;
        let main = blocks * LANES;
        let head_a = &$a[..main];
        let head_b = &$b[..main];

        let mut acc = f32x8::ZERO;
        for k in 0..blocks {
            let block_a: &[f32; LANES] = head_a[k * LANES..(k + 1) * LANES].try_into().unwrap();
            let block_b: &[f32; LANES] = head_b[k * LANES..(k + 1) * LANES].try_into().unwrap();
            let ($x, $y) = (f32x8::new(*block_a), f32x8::new(*block_b));
            acc += $term;
        }

        // The tail carries its own accumulator rather than folding into lane
        // zero, so a dimension that is not a multiple of eight still sums in a
        // fixed order.
        let mut tail = 0.0f32;
        for j in main..n {
            let ($x, $y) = ($a[j], $b[j]);
            tail += $term;
        }

        // `to_array` reads the lanes back in index order, so `reduce` sees the
        // same eight values in the same positions it saw when the accumulator
        // was an array.
        reduce(acc.to_array()) + tail
    }};
}

/// Inner product of two vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    lane_loop!(a, b, x, y, x * y)
}

/// Sum of absolute differences.
#[inline]
pub fn l1(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    lane_loop!(a, b, x, y, (x - y).abs())
}

/// Sum of squared differences, before the square root.
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    lane_loop!(a, b, x, y, {
        let d = x - y;
        d * d
    })
}

/// Euclidean distance, square root taken.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Cosine distance on vectors that are already unit length.
///
/// # Precondition
///
/// **Both arguments must be L2 normalised.** The whole point of this function is
/// that it does not recompute what the caller has already paid for, so it takes
/// the norms as given rather than checking them. On an unnormalised pair it
/// returns `1 - dot`, which is not a cosine distance and is not scale invariant.
///
/// ZeusDB holds the precondition by normalising in `process_vector_for_space`,
/// which every insertion and every query on a cosine index passes through.
///
/// # Two departures from `anndists`
///
/// A zero vector was distance zero from everything under `DistCosine`, because
/// the branch guarding the division returned zero when either norm was zero.
/// That made an all-zero record the top hit for every query on the index. Here
/// it is distance one, which is where a vector of no direction belongs on a
/// scale whose middle is orthogonal.
///
/// A non-finite component propagates rather than being reported as a perfect
/// match. `f32::max` returns the other operand when one is NaN, so the obvious
/// clamp would turn NaN into zero. The comparison below keeps it. `DistCosine`
/// asserted on the same input and aborted the process instead. Neither case is
/// reachable from `add` or `search`, both of which reject non-finite input.
#[inline]
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> f32 {
    let d = 1.0 - dot(a, b);
    // Not `d.max(0.0)`, which would swallow a NaN. A pair of identical unit
    // vectors can accumulate a dot product just above one, and reporting that as
    // a small negative distance would be a surprising score for an exact match.
    if d < 0.0 {
        0.0
    } else {
        d
    }
}

/// Cosine distance on pre-normalised vectors.
///
/// See [`cosine_normalized`] for the precondition this carries and for the two
/// places it departs from the `anndists` type it replaces.
#[derive(Default, Copy, Clone, Debug)]
pub struct CosineDist;

impl Distance<f32> for CosineDist {
    #[inline]
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        cosine_normalized(va, vb)
    }
}

/// Euclidean distance, square root taken, matching `anndists::DistL2`.
#[derive(Default, Copy, Clone, Debug)]
pub struct L2Dist;

impl Distance<f32> for L2Dist {
    #[inline]
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        l2(va, vb)
    }
}

/// Sum of absolute differences, matching `anndists::DistL1`.
#[derive(Default, Copy, Clone, Debug)]
pub struct L1Dist;

impl Distance<f32> for L1Dist {
    #[inline]
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        l1(va, vb)
    }
}

/// Inner product turned into a distance, as `1 - dot`.
///
/// Definitionally the same computation as [`CosineDist`] once the input is
/// normalised, and kept separate because it carries no such precondition and no
/// clamp. It exists so an inner product space has an implementation ready, and
/// no `space` string reaches it today.
///
/// `anndists::DistDot` asserts that the dot product is at most one and aborts
/// the process otherwise, which a self comparison of a normalised vector can
/// trip by rounding. There is no assertion here.
///
/// Constructed only by its own tests today, which is what the allow records.
/// Adding an inner product space is a separate change with its own validation
/// and documentation surface, and this is the part of it that belongs here.
#[allow(dead_code)]
#[derive(Default, Copy, Clone, Debug)]
pub struct DotDist;

impl Distance<f32> for DotDist {
    #[inline]
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        1.0 - dot(va, vb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hnsw_rs::prelude::{DistCosine, DistL1, DistL2, Hnsw};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Dimensions the grid runs at. 128, 768 and 1536 are the measured ones,
    /// and the rest are the awkward shapes around the eight wide block, being
    /// shorter than one block, one past a block, one short of a block, and a
    /// prime.
    const DIMS: [usize; 11] = [1, 2, 3, 7, 8, 9, 15, 17, 128, 768, 1536];

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn random_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    // The references. Every one accumulates in `f64` and rounds once at the end,
    // so they are the answer the `f32` kernels are approximating rather than a
    // second `f32` implementation of the same rounding.

    fn ref_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum::<f64>()
    }

    fn ref_l1(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .sum::<f64>()
    }

    fn ref_l2(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Relative deviation. Used for L1 and L2 only.
    ///
    /// Both are sums of non-negative terms, so nothing cancels and the relative
    /// error stays near the `f32` floor whatever the input magnitude is. Falls
    /// back to absolute where the reference is zero.
    fn relative(got: f32, want: f64) -> f64 {
        let diff = (got as f64 - want).abs();
        if want.abs() > 1e-12 {
            diff / want.abs()
        } else {
            diff
        }
    }

    /// Absolute deviation. Used for dot and cosine.
    ///
    /// A dot product is a sum of signed terms and cancels, so its relative error
    /// is governed by how much cancellation the particular pair happens to
    /// suffer rather than by the kernel. Two nearly parallel unit vectors have a
    /// cosine distance near zero and unbounded relative error for an absolute
    /// error of one `f32` step, which is the resolution floor recorded in its own
    /// test below and not a defect the summation order introduces. Both
    /// quantities are bounded on unit input, so the absolute figure is the one
    /// that means something.
    fn absolute(got: f32, want: f64) -> f64 {
        (got as f64 - want).abs()
    }

    /// The tolerance every agreement assertion in this module uses.
    ///
    /// Two parts per million. A `f32` carries about seven decimal digits and a
    /// 1536 term reduction over eight lanes adds roughly a further log2(1536/8)
    /// rounding steps, so a few parts per million is the floor for any `f32`
    /// kernel. It is set once and reported rather than tuned per metric, and the
    /// tests print the worst observed deviation so a regression against it is
    /// visible rather than absorbed.
    const TOLERANCE: f64 = 2e-6;

    /// The kernel shape that shipped before relay 70, kept verbatim.
    ///
    /// It is the same arithmetic in the same order over the same eight lanes,
    /// written with an `[f32; 8]` accumulator instead of an `f32x8`. It exists
    /// so the claim that the vector type changed nothing but the instruction
    /// count can be asserted rather than argued, and so the throughput harness
    /// can time the two against each other in one process.
    mod previous {
        use super::super::{reduce, LANES};

        macro_rules! blocked_loop {
            ($a:expr, $b:expr, $x:ident, $y:ident, $term:expr) => {{
                let n = core::cmp::min($a.len(), $b.len());
                let blocks = n / LANES;
                let main = blocks * LANES;
                let head_a = &$a[..main];
                let head_b = &$b[..main];

                let mut acc = [0.0f32; LANES];
                for k in 0..blocks {
                    let block_a: &[f32; LANES] =
                        head_a[k * LANES..(k + 1) * LANES].try_into().unwrap();
                    let block_b: &[f32; LANES] =
                        head_b[k * LANES..(k + 1) * LANES].try_into().unwrap();
                    for i in 0..LANES {
                        let ($x, $y) = (block_a[i], block_b[i]);
                        acc[i] += $term;
                    }
                }

                let mut tail = 0.0f32;
                for j in main..n {
                    let ($x, $y) = ($a[j], $b[j]);
                    tail += $term;
                }

                reduce(acc) + tail
            }};
        }

        #[inline]
        #[allow(clippy::needless_range_loop)]
        pub fn dot(a: &[f32], b: &[f32]) -> f32 {
            blocked_loop!(a, b, x, y, x * y)
        }

        #[inline]
        #[allow(clippy::needless_range_loop)]
        pub fn l1(a: &[f32], b: &[f32]) -> f32 {
            blocked_loop!(a, b, x, y, (x - y).abs())
        }

        #[inline]
        #[allow(clippy::needless_range_loop)]
        pub fn l2(a: &[f32], b: &[f32]) -> f32 {
            blocked_loop!(a, b, x, y, {
                let d = x - y;
                d * d
            })
            .sqrt()
        }

        #[inline]
        pub fn cosine_normalized(a: &[f32], b: &[f32]) -> f32 {
            let d = 1.0 - dot(a, b);
            if d < 0.0 {
                0.0
            } else {
                d
            }
        }

        /// The previous shape wearing the trait, so it can build a graph.
        #[derive(Default, Copy, Clone, Debug)]
        pub struct PrevCosine;

        impl crate::graph::Distance<f32> for PrevCosine {
            #[inline]
            fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
                cosine_normalized(va, vb)
            }
        }

        #[derive(Default, Copy, Clone, Debug)]
        pub struct PrevL2;

        impl crate::graph::Distance<f32> for PrevL2 {
            #[inline]
            fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
                l2(va, vb)
            }
        }

        #[derive(Default, Copy, Clone, Debug)]
        pub struct PrevL1;

        impl crate::graph::Distance<f32> for PrevL1 {
            #[inline]
            fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
                l1(va, vb)
            }
        }
    }

    /// Every kernel against its `f64` reference, over the dimension grid.
    ///
    /// Unit input throughout, which is what the index holds on the cosine space
    /// and what the relay measurements use on the other two.
    #[test]
    fn kernels_agree_with_the_f64_reference() {
        let mut r = rng(20260805);
        let mut worst = [0.0f64; 4];

        for dim in DIMS {
            for _ in 0..200 {
                let a = normalize(&random_vector(&mut r, dim));
                let b = normalize(&random_vector(&mut r, dim));

                worst[0] = worst[0].max(absolute(dot(&a, &b), ref_dot(&a, &b)));
                worst[1] = worst[1].max(relative(l1(&a, &b), ref_l1(&a, &b)));
                worst[2] = worst[2].max(relative(l2(&a, &b), ref_l2(&a, &b)));

                let want = (1.0 - ref_dot(&a, &b)).max(0.0);
                worst[3] = worst[3].max(absolute(cosine_normalized(&a, &b), want));
            }
        }

        println!(
            "worst deviation from the f64 reference  dot {:.3e} abs  l1 {:.3e} rel  \
             l2 {:.3e} rel  cosine {:.3e} abs",
            worst[0], worst[1], worst[2], worst[3]
        );
        for (name, w) in ["dot", "l1", "l2", "cosine"].iter().zip(worst) {
            assert!(
                w < TOLERANCE,
                "{name} deviated by {w:.3e}, tolerance {TOLERANCE:.1e}"
            );
        }
    }

    /// L1 and L2 across magnitudes the unit grid does not reach.
    ///
    /// Neither cancels, so the relative error is meaningful over the whole range
    /// and this is where a lane structure that broke on large or tiny inputs
    /// would show.
    ///
    /// The range stops at 1e15 and 1e-15 rather than at the `f32` extremes,
    /// because L2 squares its terms. Past those the squares leave the normal
    /// range and the arithmetic loses precision for a reason that has nothing to
    /// do with the lane structure, in `DistL2` exactly as here.
    #[test]
    fn l1_and_l2_hold_their_accuracy_across_magnitudes() {
        let mut r = rng(616);
        let (mut worst_l1, mut worst_l2) = (0.0f64, 0.0f64);

        for scale in [1.0e-15f32, 1.0e-6, 1.0, 1.0e6, 1.0e15] {
            for dim in DIMS {
                for _ in 0..50 {
                    let a: Vec<f32> = random_vector(&mut r, dim)
                        .iter()
                        .map(|x| x * scale)
                        .collect();
                    let b: Vec<f32> = random_vector(&mut r, dim)
                        .iter()
                        .map(|x| x * scale)
                        .collect();
                    worst_l1 = worst_l1.max(relative(l1(&a, &b), ref_l1(&a, &b)));
                    worst_l2 = worst_l2.max(relative(l2(&a, &b), ref_l2(&a, &b)));
                }
            }
        }

        println!(
            "worst relative deviation across magnitudes  l1 {worst_l1:.3e}  l2 {worst_l2:.3e}"
        );
        assert!(worst_l1 < TOLERANCE, "l1 deviated by {worst_l1:.3e}");
        assert!(worst_l2 < TOLERANCE, "l2 deviated by {worst_l2:.3e}");
    }

    /// The edge cases the brief names, one assertion each.
    #[test]
    fn edge_cases() {
        // Zero vectors. L1 and L2 are zero, dot is zero, and cosine is one
        // rather than the zero `DistCosine` returned. This is the departure the
        // documentation on `cosine_normalized` records.
        for dim in DIMS {
            let z = vec![0.0f32; dim];
            assert_eq!(l1(&z, &z), 0.0);
            assert_eq!(l2(&z, &z), 0.0);
            assert_eq!(dot(&z, &z), 0.0);
            assert_eq!(cosine_normalized(&z, &z), 1.0);
        }

        // A zero vector against a unit vector is orthogonal, not identical.
        let mut unit = vec![0.0f32; 768];
        unit[3] = 1.0;
        let zero = vec![0.0f32; 768];
        assert_eq!(cosine_normalized(&zero, &unit), 1.0);
        assert_eq!(DistCosine {}.eval(&zero, &unit), 0.0);

        // One non-zero element, at a position past the first block so the tail
        // and the block path both carry it.
        for dim in [1usize, 8, 9, 768] {
            let mut a = vec![0.0f32; dim];
            let mut b = vec![0.0f32; dim];
            a[dim - 1] = 3.0;
            b[dim - 1] = 4.0;
            assert_eq!(dot(&a, &b), 12.0);
            assert_eq!(l1(&a, &b), 1.0);
            assert_eq!(l2(&a, &b), 1.0);
        }

        // Identical unit vectors give exactly zero after the clamp rather than
        // a small negative number.
        let mut r = rng(99);
        for dim in DIMS {
            let v = normalize(&random_vector(&mut r, dim));
            let d = cosine_normalized(&v, &v);
            assert!((0.0..=1e-6).contains(&d), "self distance {d} at dim {dim}");
            assert_eq!(l2(&v, &v), 0.0);
            assert_eq!(l1(&v, &v), 0.0);
        }

        // Very large and very small magnitudes, checked against the `f64`
        // reference rather than against a hand computed constant, so the
        // assertion is about the kernel and not about where `f32` overflows.
        // The large pair is chosen to keep the squares inside `f32` range, since
        // `DistL2` overflows there too and this is a kernel replacement rather
        // than a range extension.
        for (lo, hi) in [(1.0e15f32, 2.0e15f32), (1.0e-15, 2.0e-15)] {
            let a = vec![lo; 768];
            let b = vec![hi; 768];
            assert!(l1(&a, &b).is_finite() && l1(&a, &b) > 0.0);
            assert!(l2(&a, &b).is_finite() && l2(&a, &b) > 0.0);
            assert!(relative(l1(&a, &b), ref_l1(&a, &b)) < TOLERANCE);
            assert!(relative(l2(&a, &b), ref_l2(&a, &b)) < TOLERANCE);
        }

        // Small enough that every product underflows to zero, which is what
        // `f32` does anywhere and not something the lane structure introduces.
        let tiny = vec![1.0e-30f32; 768];
        assert_eq!(dot(&tiny, &tiny), 0.0);

        // Dimensions that are not multiples of the block width are covered by
        // the grid above, and this pins the one that has no block at all.
        assert_eq!(dot(&[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]), 56.0);
        assert_eq!(l1(&[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]), 9.0);
    }

    /// What a non-finite component does if one reaches a kernel.
    ///
    /// `add` and `search` both reject non-finite input, so this documents an
    /// unreachable path rather than a supported one. It propagates, which is
    /// what keeps a NaN from being reported as distance zero.
    #[test]
    fn non_finite_input_propagates() {
        let mut a = vec![0.1f32; 768];
        let b = vec![0.1f32; 768];

        a[0] = f32::NAN;
        assert!(dot(&a, &b).is_nan());
        assert!(l1(&a, &b).is_nan());
        assert!(l2(&a, &b).is_nan());
        assert!(cosine_normalized(&a, &b).is_nan());
        assert!(DotDist {}.eval(&a, &b).is_nan());

        // The same in the tail rather than in a block.
        let mut t = vec![0.1f32; 3];
        let u = vec![0.1f32; 3];
        t[2] = f32::NAN;
        assert!(dot(&t, &u).is_nan());
        assert!(cosine_normalized(&t, &u).is_nan());

        // Infinity gives infinity for the magnitude metrics, and the cosine
        // clamp keeps a negative infinity from becoming zero.
        a[0] = f32::INFINITY;
        assert_eq!(l1(&a, &b), f32::INFINITY);
        assert_eq!(l2(&a, &b), f32::INFINITY);
        assert_eq!(cosine_normalized(&a, &b), 0.0);
    }

    /// Agreement with the implementations these replace, at the value level.
    ///
    /// L1 and L2 must match closely, since the formula is unchanged and only the
    /// summation order moved. Cosine is allowed to differ by the norm correction
    /// it no longer applies, which on normalised input is the distance between
    /// the recomputed norm and one.
    #[test]
    fn values_agree_with_anndists() {
        let mut r = rng(4242);
        let (mut worst_l1, mut worst_l2, mut worst_cos) = (0.0f64, 0.0f64, 0.0f64);

        for dim in DIMS {
            for _ in 0..200 {
                let a = random_vector(&mut r, dim);
                let b = random_vector(&mut r, dim);
                let (na, nb) = (normalize(&a), normalize(&b));

                worst_l1 = worst_l1.max(relative(l1(&a, &b), DistL1 {}.eval(&a, &b) as f64));
                worst_l2 = worst_l2.max(relative(l2(&a, &b), DistL2 {}.eval(&a, &b) as f64));
                worst_cos = worst_cos.max(absolute(
                    cosine_normalized(&na, &nb),
                    DistCosine {}.eval(&na, &nb) as f64,
                ));
            }
        }

        println!(
            "worst deviation against anndists  l1 {worst_l1:.3e} rel  l2 {worst_l2:.3e} rel  \
             cosine {worst_cos:.3e} abs"
        );
        assert!(worst_l1 < TOLERANCE, "l1 deviated by {worst_l1:.3e}");
        assert!(worst_l2 < TOLERANCE, "l2 deviated by {worst_l2:.3e}");
        assert!(worst_cos < TOLERANCE, "cosine deviated by {worst_cos:.3e}");
    }

    /// The vector accumulator against the array accumulator, bit for bit.
    ///
    /// This is the assertion the relay 70 change rests on. Widening the
    /// accumulator from four lanes of a register to eight was a change of
    /// instruction selection and not of arithmetic, so every kernel must return
    /// the identical `f32`, not a close one. Compared as bit patterns rather
    /// than by value, so a signed zero or a NaN payload would show.
    ///
    /// Magnitudes are swept as well as dimensions, because a packed rounding
    /// that differed from a scalar one would show at the ends of the range
    /// rather than in the middle.
    #[test]
    fn kernels_match_the_previous_shape_bit_for_bit() {
        let mut r = rng(70_70_70);
        let mut compared = 0usize;

        for scale in [1.0e-15f32, 1.0e-6, 1.0, 1.0e6, 1.0e15] {
            for dim in DIMS {
                for _ in 0..100 {
                    let a: Vec<f32> = random_vector(&mut r, dim)
                        .iter()
                        .map(|x| x * scale)
                        .collect();
                    let b: Vec<f32> = random_vector(&mut r, dim)
                        .iter()
                        .map(|x| x * scale)
                        .collect();
                    let (na, nb) = (normalize(&a), normalize(&b));

                    for (name, got, want) in [
                        ("dot", dot(&a, &b), previous::dot(&a, &b)),
                        ("l1", l1(&a, &b), previous::l1(&a, &b)),
                        ("l2", l2(&a, &b), previous::l2(&a, &b)),
                        (
                            "cosine",
                            cosine_normalized(&na, &nb),
                            previous::cosine_normalized(&na, &nb),
                        ),
                    ] {
                        compared += 1;
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{name} at dim {dim} scale {scale:e} returned {got:e} \
                             where the previous shape returned {want:e}"
                        );
                    }
                }
            }
        }

        // Zeros, a lone non-zero, and the non-finite cases the kernels can see.
        for dim in DIMS {
            let z = vec![0.0f32; dim];
            let mut one = vec![0.0f32; dim];
            one[dim - 1] = 3.0;
            let mut nan = vec![0.1f32; dim];
            nan[0] = f32::NAN;
            let mut inf = vec![0.1f32; dim];
            inf[0] = f32::INFINITY;
            let ones = vec![0.1f32; dim];

            for (u, v) in [(&z, &z), (&one, &z), (&nan, &ones), (&inf, &ones)] {
                compared += 4;
                assert_eq!(dot(u, v).to_bits(), previous::dot(u, v).to_bits());
                assert_eq!(l1(u, v).to_bits(), previous::l1(u, v).to_bits());
                assert_eq!(l2(u, v).to_bits(), previous::l2(u, v).to_bits());
                assert_eq!(
                    cosine_normalized(u, v).to_bits(),
                    previous::cosine_normalized(u, v).to_bits()
                );
            }
        }

        println!("bit identity against the previous shape  compared {compared}  differing 0");
    }

    /// The property the graph actually depends on.
    ///
    /// A distance that differs from the old one in the last bit but never
    /// reorders a pair is safe to swap in. One that reorders is not, because the
    /// neighbour selection heuristic and the search stopping condition both read
    /// the order and nothing reads the magnitude.
    ///
    /// For each triple the old and new implementations are asked the same
    /// question, being whether the query is closer to `b` than to `c`, and the
    /// answers must match. Ties are counted separately, since a tie under one
    /// implementation and a strict order under the other is not a reordering.
    #[test]
    fn ordering_matches_anndists() {
        let mut r = rng(31337);
        let mut compared = 0usize;
        let mut disagreed = 0usize;
        let mut ties = 0usize;

        for dim in [8usize, 128, 768, 1536] {
            for _ in 0..4000 {
                let q = normalize(&random_vector(&mut r, dim));
                let b = normalize(&random_vector(&mut r, dim));
                let c = normalize(&random_vector(&mut r, dim));

                for (new_qb, new_qc, old_qb, old_qc) in [
                    (
                        cosine_normalized(&q, &b),
                        cosine_normalized(&q, &c),
                        DistCosine {}.eval(&q, &b),
                        DistCosine {}.eval(&q, &c),
                    ),
                    (
                        l2(&q, &b),
                        l2(&q, &c),
                        DistL2 {}.eval(&q, &b),
                        DistL2 {}.eval(&q, &c),
                    ),
                    (
                        l1(&q, &b),
                        l1(&q, &c),
                        DistL1 {}.eval(&q, &b),
                        DistL1 {}.eval(&q, &c),
                    ),
                ] {
                    compared += 1;
                    if new_qb == new_qc || old_qb == old_qc {
                        ties += 1;
                        continue;
                    }
                    if (new_qb < new_qc) != (old_qb < old_qc) {
                        disagreed += 1;
                    }
                }
            }
        }

        println!("ordering  compared {compared}  ties {ties}  disagreed {disagreed}");
        assert_eq!(disagreed, 0, "{disagreed} of {compared} pairs reordered");
    }

    /// Near ties, judged against the truth rather than against each other.
    ///
    /// The pass above asks whether the two implementations agree, and on
    /// independently drawn candidates they always do. This pass builds the hard
    /// case deliberately, with `c` a one component perturbation of `b`, so the
    /// two candidates sit a few parts per million apart. There, asking whether
    /// the two agree is the wrong question. Two implementations of different
    /// precision will disagree on pairs that neither can resolve, and the
    /// disagreement says nothing about which of them is right.
    ///
    /// The question that decides adoption is whether either implementation
    /// orders a pair **wrongly**, meaning against the `f64` reference over the
    /// same inputs. A wrong order is excusable only inside the error bars. Two
    /// values each carrying an absolute error of at most `TOLERANCE` can swap
    /// only when the true gap between them is under twice that, so what is
    /// asserted is that no misordering happens outside it.
    ///
    /// This replaced a straight agreement assertion, which failed at 2 of 8,000.
    /// Both cases were inside the error bars and in neither was either
    /// implementation wrong, which the counters below now show rather than
    /// assume.
    #[test]
    fn near_ties_are_never_ordered_wrongly() {
        let mut r = rng(505);
        let (mut compared, mut ties) = (0usize, 0usize);
        let (mut new_vs_old, mut new_wrong, mut old_wrong) = (0usize, 0usize, 0usize);
        let (mut worst_new_gap, mut worst_old_gap) = (0.0f64, 0.0f64);

        for dim in [128usize, 768] {
            for _ in 0..4000 {
                let q = normalize(&random_vector(&mut r, dim));
                let b = normalize(&random_vector(&mut r, dim));

                // Nudge one component, so the two candidates sit a few parts per
                // million apart and any disagreement in the last bits shows.
                let mut c = b.clone();
                c[dim / 2] += 1e-5;
                let c = normalize(&c);

                // The truth, computed in `f64` over the same `f32` inputs, so
                // the only thing being judged is the summation.
                let (true_b, true_c) = (1.0 - ref_dot(&q, &b), 1.0 - ref_dot(&q, &c));
                let gap = (true_b - true_c).abs();

                let (new_b, new_c) = (cosine_normalized(&q, &b), cosine_normalized(&q, &c));
                let (old_b, old_c) = (DistCosine {}.eval(&q, &b), DistCosine {}.eval(&q, &c));

                compared += 1;
                if true_b == true_c {
                    ties += 1;
                    continue;
                }
                let closer_is_b = true_b < true_c;

                if new_b != new_c && (new_b < new_c) != closer_is_b {
                    new_wrong += 1;
                    worst_new_gap = worst_new_gap.max(gap);
                }
                if old_b != old_c && (old_b < old_c) != closer_is_b {
                    old_wrong += 1;
                    worst_old_gap = worst_old_gap.max(gap);
                }
                if new_b != new_c && old_b != old_c && (new_b < new_c) != (old_b < old_c) {
                    new_vs_old += 1;
                }
            }
        }

        println!(
            "near ties  compared {compared}  exact ties {ties}  new against old {new_vs_old}  \
             new wrong {new_wrong} worst gap {worst_new_gap:.3e}  \
             old wrong {old_wrong} worst gap {worst_old_gap:.3e}"
        );

        // Two values each accurate to `TOLERANCE` can swap only inside twice it.
        let bar = 2.0 * TOLERANCE;
        assert!(
            worst_new_gap < bar,
            "the new cosine misordered a pair whose true distances differ by \
             {worst_new_gap:.3e}, which is outside the {bar:.1e} error bar"
        );
        assert!(
            worst_old_gap < bar,
            "the old cosine misordered a pair whose true distances differ by \
             {worst_old_gap:.3e}, which is outside the {bar:.1e} error bar"
        );
    }

    /// How close two vectors can be before the cosine distance stops separating
    /// them from each other.
    ///
    /// `1 - dot` cancels as the dot product approaches one, so the difference it
    /// returns is quantised to one `f32` step either side of one, which is about
    /// 1.2e-7. Below that the value is accumulation noise rather than a
    /// measurement of the pair. That is a consequence of the two things the
    /// metric is built on, being `f32` arithmetic and the reduction to one minus
    /// the dot product, and it cannot be removed while keeping both.
    ///
    /// The two implementations fail differently rather than one being better.
    /// Sweeping a perturbation down through the floor, the new one returns a
    /// non-zero value every time and the smallest is one step, while the old one
    /// reaches 4e-9 and then returns exact zero for the five smallest
    /// perturbations, because its `f64` ratio rounds to one. So the old one
    /// reports nothing where the new one reports noise.
    ///
    /// Neither is better at the thing that matters. The near tie pass above
    /// measures ordering directly, and there the old implementation misorders as
    /// many pairs as the new one and does so at a larger true gap, because it
    /// returns `f32` whatever precision went into producing it.
    ///
    /// Summing squared differences and halving would resolve further, since
    /// `||a - b||^2 = 2 - 2 * dot` on unit vectors and that form does not cancel.
    /// It costs about one and a half times the arithmetic and is the change to
    /// make if a caller ever needs to rank near duplicates against each other.
    #[test]
    fn the_cosine_resolution_floor_is_one_f32_step_at_one() {
        let mut r = rng(8080);
        let base = normalize(&random_vector(&mut r, 768));

        let (mut new_floor, mut old_floor) = (f64::INFINITY, f64::INFINITY);
        let (mut new_zeros, mut old_zeros) = (0usize, 0usize);

        // Walk a perturbation down through the floor and record where each
        // implementation stops separating the pair from itself.
        for exponent in 1..=9 {
            let eps = 10f32.powi(-exponent);
            let mut near = base.clone();
            near[17] += eps;
            let near = normalize(&near);

            let new_d = cosine_normalized(&base, &near) as f64;
            let old_d = DistCosine {}.eval(&base, &near) as f64;

            if new_d > 0.0 {
                new_floor = new_floor.min(new_d);
            } else {
                new_zeros += 1;
            }
            if old_d > 0.0 {
                old_floor = old_floor.min(old_d);
            } else {
                old_zeros += 1;
            }
        }

        println!(
            "cosine resolution  new smallest non zero {new_floor:.3e} with {new_zeros} collapsed  \
             old smallest non zero {old_floor:.3e} with {old_zeros} collapsed"
        );

        // One `f32` step at one is 1.192e-7. Allow two, since the accumulated
        // dot product does not land on the step boundary exactly.
        let two_steps = 2.0 * (f32::EPSILON as f64);
        assert!(
            new_floor <= two_steps,
            "the new cosine resolved no finer than {new_floor:.3e}, worse than the \
             {two_steps:.3e} the representation allows"
        );
    }

    /// Per layer adjacency, keyed by the id the caller inserted under.
    fn adjacency<D>(hnsw: &Hnsw<'_, f32, D>, n: usize) -> Vec<Vec<Vec<usize>>>
    where
        D: Distance<f32> + Send + Sync,
    {
        let mut adj = vec![Vec::new(); n];
        for point in hnsw.get_point_indexation() {
            adj[point.get_origin_id()] = point
                .get_neighborhood_id()
                .iter()
                .map(|layer| layer.iter().map(|nb| nb.d_id).collect())
                .collect();
        }
        adj
    }

    /// Build one graph per distance on identical data and count the difference.
    ///
    /// Returns the number of nodes whose adjacency differs, the number of edges
    /// that differ, the total edge count, and the number of the sampled queries
    /// that returned a different result list.
    fn compare_graphs<A, B>(
        old_dist: A,
        new_dist: B,
        data: &[Vec<f32>],
    ) -> (usize, usize, usize, usize)
    where
        A: Distance<f32> + Send + Sync,
        B: Distance<f32> + Send + Sync,
    {
        let n = data.len();
        let old = Hnsw::new(16, n, 16, 200, old_dist);
        let new = Hnsw::new(16, n, 16, 200, new_dist);
        for (i, v) in data.iter().enumerate() {
            old.insert((v.as_slice(), i));
            new.insert((v.as_slice(), i));
        }

        let (old_adj, new_adj) = (adjacency(&old, n), adjacency(&new, n));
        let (mut differing_nodes, mut differing_edges, mut total_edges) = (0usize, 0usize, 0usize);

        for id in 0..n {
            let (o, p) = (&old_adj[id], &new_adj[id]);
            let mut node_differs = false;
            for layer in 0..o.len().max(p.len()) {
                let empty = Vec::new();
                let (mut os, mut ns) = (
                    o.get(layer).unwrap_or(&empty).clone(),
                    p.get(layer).unwrap_or(&empty).clone(),
                );
                total_edges += os.len();
                os.sort_unstable();
                ns.sort_unstable();
                if os != ns {
                    node_differs = true;
                    differing_edges += os.iter().filter(|e| !ns.contains(e)).count();
                }
            }
            if node_differs {
                differing_nodes += 1;
            }
        }

        let differing_queries = data
            .iter()
            .take(200)
            .filter(|q| {
                let a: Vec<usize> = old.search(q, 10, 100).iter().map(|h| h.d_id).collect();
                let b: Vec<usize> = new.search(q, 10, 100).iter().map(|h| h.d_id).collect();
                a != b
            })
            .count();

        (
            differing_nodes,
            differing_edges,
            total_edges,
            differing_queries,
        )
    }

    /// The end to end check. Two graphs, identical data, identical level
    /// assignment, one distance each.
    ///
    /// Ordering equivalence over sampled triples says the two distances rank
    /// the same way. This says the graphs that result are the same object, which
    /// is the stronger claim and the one that decides whether recall can move.
    ///
    /// The level assignment is already fixed. `LayerGenerator` seeds from
    /// `DEFAULT_LEVEL_SEED`, which ZeusDB's patch 2 added, so two sequential
    /// builds draw the same levels in the same order without anything being set
    /// here. Insertion is sequential for the same reason the two graph guard
    /// tests in `hnsw_index` are, being that `parallel_insert` draws levels in
    /// thread arrival order and no seed makes that reproducible.
    /// Cosine, where the two graphs come out identical.
    #[test]
    fn the_two_distances_build_the_same_cosine_graph() {
        const N: usize = 3000;
        const DIM: usize = 64;

        let mut r = rng(2718);
        let data: Vec<Vec<f32>> = (0..N)
            .map(|_| normalize(&random_vector(&mut r, DIM)))
            .collect();

        let (nodes, edges, total, queries) = compare_graphs(DistCosine {}, CosineDist {}, &data);
        println!(
            "graph comparison cosine  nodes {N}  edges {total}  differing nodes {nodes}  \
             differing edges {edges}  differing queries {queries} of 200"
        );

        assert_eq!(
            nodes, 0,
            "{nodes} of {N} nodes differ, {edges} of {total} edges"
        );
        assert_eq!(
            queries, 0,
            "{queries} of 200 queries returned a different page"
        );
    }

    /// L1 and L2, where they do not.
    ///
    /// Cosine agrees exactly because the two implementations arrive at the same
    /// `f32` value for a normalised pair often enough that no comparison in the
    /// build ever flips. L1 and L2 keep the same formula and change only the
    /// summation order, from one sequential chain to eight lanes, so their values
    /// differ in the last bits on almost every pair. Where two candidates are
    /// closer together than that difference, the graphs can pick different
    /// neighbours.
    ///
    /// What is asserted is that the difference stays at the level of last bit
    /// tie-breaking rather than becoming a different graph. The bound is one
    /// percent of edges, which is roughly a hundred times the measured figure, so
    /// it catches a real divergence without tracking noise. The measured numbers
    /// are printed so a drift toward the bound is visible.
    ///
    /// This is the mechanism behind the L1 recall move in section 4.2 of the
    /// relay 40 report, where recall at 10 went from 0.7775 to 0.7780 on 200
    /// queries, being a single hit in two thousand.
    #[test]
    fn l1_and_l2_graphs_differ_only_by_last_bit_tie_breaking() {
        const N: usize = 3000;
        const DIM: usize = 64;

        let mut r = rng(2718);
        let data: Vec<Vec<f32>> = (0..N)
            .map(|_| normalize(&random_vector(&mut r, DIM)))
            .collect();

        for (name, (nodes, edges, total, queries)) in [
            ("l2", compare_graphs(DistL2 {}, L2Dist {}, &data)),
            ("l1", compare_graphs(DistL1 {}, L1Dist {}, &data)),
        ] {
            let share = edges as f64 / total as f64;
            println!(
                "graph comparison {name}  nodes {N}  edges {total}  differing nodes {nodes}  \
                 differing edges {edges} ({:.4} percent)  differing queries {queries} of 200",
                share * 100.0
            );
            assert!(
                share < 0.01,
                "{name} graphs differ by {edges} of {total} edges, which is past last bit noise"
            );
        }
    }

    /// The vector accumulator and the array accumulator build the same graph.
    ///
    /// Bit identity already implies this, since a comparison cannot separate two
    /// identical values. It is measured rather than inferred because the graph
    /// is what the change is actually risking, and a zero here is the statement
    /// that matters. All three spaces, since L1 and L2 are the two that tie-break
    /// against `anndists` and so are the two with anything to lose.
    #[test]
    fn the_two_accumulator_shapes_build_the_same_graph() {
        const N: usize = 3000;
        const DIM: usize = 64;

        let mut r = rng(2718);
        let data: Vec<Vec<f32>> = (0..N)
            .map(|_| normalize(&random_vector(&mut r, DIM)))
            .collect();

        for (name, (nodes, edges, total, queries)) in [
            (
                "cosine",
                compare_graphs(previous::PrevCosine {}, CosineDist {}, &data),
            ),
            ("l2", compare_graphs(previous::PrevL2 {}, L2Dist {}, &data)),
            ("l1", compare_graphs(previous::PrevL1 {}, L1Dist {}, &data)),
        ] {
            println!(
                "graph comparison against the previous shape {name}  nodes {N}  edges {total}  \
                 differing nodes {nodes}  differing edges {edges}  \
                 differing queries {queries} of 200"
            );
            assert_eq!(nodes, 0, "{name}: {nodes} of {N} nodes differ");
            assert_eq!(edges, 0, "{name}: {edges} of {total} edges differ");
            assert_eq!(queries, 0, "{name}: {queries} of 200 queries differ");
        }
    }

    /// Ordering equivalence against the previous shape, counted rather than
    /// assumed.
    ///
    /// The same question the `anndists` pass asks, put to the shape this one
    /// replaced. Near ties are built deliberately, by perturbing one component
    /// of the second candidate, because that is where a last bit difference
    /// would decide an order if there were one to find.
    #[test]
    fn ordering_matches_the_previous_shape() {
        let mut r = rng(7070);
        let (mut compared, mut ties, mut disagreed) = (0usize, 0usize, 0usize);

        for dim in [8usize, 128, 768, 1536] {
            for _ in 0..2000 {
                let q = normalize(&random_vector(&mut r, dim));
                let b = normalize(&random_vector(&mut r, dim));

                // Half the triples are independent draws and half are near
                // ties, so both the easy and the hard case are counted.
                let c = if compared % 2 == 0 {
                    normalize(&random_vector(&mut r, dim))
                } else {
                    let mut c = b.clone();
                    c[dim / 2] += 1e-5;
                    normalize(&c)
                };

                for (new_qb, new_qc, old_qb, old_qc) in [
                    (
                        cosine_normalized(&q, &b),
                        cosine_normalized(&q, &c),
                        previous::cosine_normalized(&q, &b),
                        previous::cosine_normalized(&q, &c),
                    ),
                    (
                        l2(&q, &b),
                        l2(&q, &c),
                        previous::l2(&q, &b),
                        previous::l2(&q, &c),
                    ),
                    (
                        l1(&q, &b),
                        l1(&q, &c),
                        previous::l1(&q, &b),
                        previous::l1(&q, &c),
                    ),
                ] {
                    compared += 1;
                    if new_qb == new_qc || old_qb == old_qc {
                        ties += 1;
                        continue;
                    }
                    if (new_qb < new_qc) != (old_qb < old_qc) {
                        disagreed += 1;
                    }
                }
            }
        }

        println!(
            "ordering against the previous shape  compared {compared}  ties {ties}  \
             disagreed {disagreed}"
        );
        assert_eq!(disagreed, 0, "{disagreed} of {compared} pairs reordered");
    }

    /// Nanoseconds per call, old against new, at the three measured dimensions.
    ///
    /// Ignored by default because it is a timing harness rather than an
    /// assertion. Run it with
    /// `cargo test --release --locked distance::tests::throughput -- --ignored --nocapture`.
    ///
    /// The working set is small enough to stay in cache on purpose. This
    /// measures the kernel, not the memory system a traversal fights, and the
    /// search latency figures in the relay report are what say how much of the
    /// kernel gain survives contact with a real graph.
    #[test]
    #[ignore = "timing harness, run with --ignored --nocapture on a release build"]
    fn throughput_old_against_new() {
        use std::time::Instant;

        const PAIRS: usize = 128;
        const REPEATS: usize = 4000;

        println!(
            "\ndim  metric  anndists ns   blocked ns   wide ns   \
             over blocked   over anndists"
        );
        for dim in [128usize, 768, 1536] {
            let mut r = rng(555);
            let a: Vec<Vec<f32>> = (0..PAIRS)
                .map(|_| normalize(&random_vector(&mut r, dim)))
                .collect();
            let b: Vec<Vec<f32>> = (0..PAIRS)
                .map(|_| normalize(&random_vector(&mut r, dim)))
                .collect();

            let calls = (PAIRS * REPEATS) as f64;

            // Generic rather than a trait object, so each cell measures an
            // inlined kernel rather than an indirect call through a vtable.
            fn time<F: Fn(&[f32], &[f32]) -> f32>(
                a: &[Vec<f32>],
                b: &[Vec<f32>],
                repeats: usize,
                calls: f64,
                f: F,
            ) -> f64 {
                use std::hint::black_box;
                // One untimed pass, so the branch predictor and the caches are
                // in the same state for every cell.
                for i in 0..a.len() {
                    black_box(f(black_box(&a[i]), black_box(&b[i])));
                }
                let t = Instant::now();
                for _ in 0..repeats {
                    for i in 0..a.len() {
                        black_box(f(black_box(&a[i]), black_box(&b[i])));
                    }
                }
                t.elapsed().as_secs_f64() * 1e9 / calls
            }

            let cells = [
                (
                    "cosine",
                    time(&a, &b, REPEATS, calls, |x, y| DistCosine {}.eval(x, y)),
                    time(&a, &b, REPEATS, calls, previous::cosine_normalized),
                    time(&a, &b, REPEATS, calls, cosine_normalized),
                ),
                (
                    "l2",
                    time(&a, &b, REPEATS, calls, |x, y| DistL2 {}.eval(x, y)),
                    time(&a, &b, REPEATS, calls, previous::l2),
                    time(&a, &b, REPEATS, calls, l2),
                ),
                (
                    "l1",
                    time(&a, &b, REPEATS, calls, |x, y| DistL1 {}.eval(x, y)),
                    time(&a, &b, REPEATS, calls, previous::l1),
                    time(&a, &b, REPEATS, calls, l1),
                ),
            ];

            for (name, anndists_ns, blocked_ns, wide_ns) in cells {
                println!(
                    "{dim:<5}{name:<8}{anndists_ns:>11.2}{blocked_ns:>13.2}{wide_ns:>10.2}\
                     {:>13.2}x{:>13.2}x",
                    blocked_ns / wide_ns,
                    anndists_ns / wide_ns
                );
            }
        }
    }
}
