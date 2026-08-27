//! Test data shared by the modules that measure against it.
//!
//! `clustered` is the specification every quantized measurement in this project
//! uses, so the recorded figures and the thresholds in the tests describe the
//! same data. One definition rather than one per test module, for
//! that reason.

use crate::rng::SeededRng;
use rand::{Rng, SeedableRng};
/// Clustered unit vectors. Fifty Gaussian centres, points drawn as a centre
/// plus 0.15 times a Gaussian perturbation, then L2 normalised, with the
/// centres deliberately left unnormalised. This is the shape real
/// embeddings have and the specification every quantized measurement in
/// this project uses, so the recorded figures and the thresholds below
/// describe the same data.
///
/// Uniform noise was tried and rejected. Its spread is too small relative
/// to the centre separation, so records within a cluster quantize to the
/// same codes, their distance is genuinely zero, and the graph partially
/// collapses for a reason that has nothing to do with what these tests
/// assert.
pub fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SeededRng::seed_from_u64(seed);
    let gauss = |rng: &mut SeededRng| {
        let u: f32 = rng.random::<f32>().max(1e-12);
        let v: f32 = rng.random::<f32>();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    };
    let centres: Vec<Vec<f32>> = (0..50)
        .map(|_| (0..dim).map(|_| gauss(&mut rng)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centres[i % 50];
            let mut v: Vec<f32> = (0..dim).map(|d| c[d] + 0.15 * gauss(&mut rng)).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect()
}
