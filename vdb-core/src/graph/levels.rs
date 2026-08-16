//! The level generator, reproducing the vendored stream draw for draw.
//!
//! A new point's top level is drawn from an exponential law, and the graph the
//! insertions build is a function of that sequence. Two generators that agree
//! on the distribution but not on the sequence build two different graphs from
//! the same data, and every reproducibility guarantee in the suite is a
//! statement about the sequence. So this is not a reimplementation of the idea.
//! It is the vendored `LayerGenerator::generate` written out again, calling the
//! same distributions on the same generator in the same order.
//!
//! # The draw
//!
//! One sample from `Uniform<f64>` over `[0, 1)`, which consumes exactly one
//! `u64` from the stream. The level is `floor(-ln(u) * scale)`. Above the cap
//! it is redrawn from `Uniform<usize>` over `[0, maxlevel)`, which consumes
//! from the same generator and so shifts everything after it. That redraw is
//! why a generator cannot be written as a pure function of a draw counter.
//!
//! # The scale is installed absolutely
//!
//! The vendored crate has two constructors, one taking a modification factor
//! that it multiplies into the default `1 / ln(m)` and one taking the value
//! itself. Patch 6 exists because the reload used the first with a value, so a
//! saved scale came back squared. This takes the value, because a dump records
//! what the generator held rather than how it was derived.
//!
//! # What `rand` promises
//!
//! Nothing, and that is worth writing down. `StdRng` is documented as
//! non-portable: any future library version may replace the algorithm, and
//! results may be platform dependent. Today it is ChaCha12 through
//! `rand_chacha`, and the crate points a caller who needs a stable stream at
//! `rand_chacha` directly. Calling the same `rand` the vendored crate calls is
//! what makes this generator's stream identical to the vendored one, and it
//! inherits the same absence of a guarantee. See the relay report.

use rand::distr::Uniform;
use rand::prelude::*;

/// Seed used for level assignment unless another one is set.
///
/// The same value the vendored patch installs, byte for byte, because a
/// replacement generator that reseeded differently would build a different
/// graph from the same data on the first insertion.
pub(super) const DEFAULT_LEVEL_SEED: u64 = 0x5A45_5553_4442_5F30;

/// Draws a point's top level from an exponential law of parameter `scale`,
/// constrained to `[0, maxlevel)`.
///
/// The vendored counterpart wraps its generator in an `Arc<Mutex<_>>` and draws
/// through `&self`, because its insert runs under a read guard and several
/// threads reach the generator at once. This draws through `&mut self`, because
/// the structure it feeds is mutated under the index write lock and the mutator
/// is serialised. The stream is the same either way; only the lock leaves.
pub(super) struct LevelGenerator {
    rng: rand::rngs::StdRng,
    unif: Uniform<f64>,
    /// Drives the number of levels generated.
    scale: f64,
    maxlevel: usize,
    /// Draws that landed at or above the cap and were redispatched. Kept
    /// because the redraw is the part of the stream that is easiest to get
    /// wrong and hardest to see: it consumes a second value, so a generator
    /// that redraws at a different rate diverges from that point on rather
    /// than at the draw itself.
    redraws: usize,
}

impl LevelGenerator {
    /// A generator whose scale is the given value itself, seeded at
    /// [`DEFAULT_LEVEL_SEED`].
    ///
    /// There is deliberately no constructor taking a modification factor. The
    /// only value ZeusDB ever installs is either `1 / ln(m)` at creation or the
    /// scale a dump recorded, and both are values rather than factors.
    pub(super) fn new(scale: f64, maxlevel: usize) -> Self {
        LevelGenerator {
            rng: StdRng::seed_from_u64(DEFAULT_LEVEL_SEED),
            unif: Uniform::<f64>::new(0., 1.).expect("zero is below one"),
            scale,
            maxlevel,
            redraws: 0,
        }
    }

    /// The scale a fresh graph at `max_nb_connection` draws with, which is the
    /// vendored `1 / ln(max_nb_connection)`.
    pub(super) fn default_scale(max_nb_connection: usize) -> f64 {
        1. / (max_nb_connection as f64).ln()
    }

    /// Reseed the stream. Resets it rather than extending it, so a caller that
    /// wants a chosen seed calls this before the first insertion.
    pub(super) fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// The scale, which a dump records.
    pub(super) fn scale(&self) -> f64 {
        self.scale
    }

    /// Draws so far that hit the cap and were redispatched.
    pub(super) fn redraws(&self) -> usize {
        self.redraws
    }

    /// Draw one level.
    ///
    /// `P(l = n) = exp(-n / S) - exp(-(n + 1) / S)` for scale `S`, and a draw
    /// at or above `maxlevel` is redispatched uniformly over the range. The
    /// redraw consumes from the same stream, which is behaviour rather than an
    /// implementation detail: it advances the sequence for every later point.
    pub(super) fn generate(&mut self) -> usize {
        let xsi = self.rng.sample(self.unif);
        let level = -xsi.ln() * self.scale;
        let mut ulevel = level.floor() as usize;
        // Very low probability at the default scale. Cf the law above.
        if ulevel >= self.maxlevel {
            self.redraws += 1;
            ulevel = self
                .rng
                .sample(Uniform::<usize>::new(0, self.maxlevel).expect("maxlevel is at least one"));
        }
        ulevel
    }
}
