//! The generator every seeded draw in the crate runs on.
//!
//! # Why it is not `StdRng`
//!
//! `rand::rngs::StdRng` is documented as non-portable, being "any future
//! library version may replace the algorithm and results may be
//! platform-dependent", and `rand` points a caller who needs a stable stream at
//! `rand_chacha` directly. Three things this crate builds are functions of a
//! seeded stream rather than of the data alone. The level sequence every graph
//! is built from, the sample the product quantizer trains on, and the k-means
//! initialisation of each subvector's codebook. A `rand` release that replaced
//! `StdRng` would move all three at once, so a user who rebuilt an index after
//! a routine dependency bump would get a different graph from the one they had
//! before it, with nothing in any release note to say so.
//!
//! So the algorithm is named here rather than taken from whatever `StdRng`
//! happens to be. `rand_chacha` documents its generators as "deterministic and
//! portable ... with testing against reference vectors", which is the opposite
//! of the `StdRng` wording.
//!
//! # What it costs
//!
//! Nothing measurable and nothing to the stream. `StdRng` in `rand` 0.9 is a
//! newtype over `rand_chacha::ChaCha12Rng` that forwards `next_u32`,
//! `next_u64`, `fill_bytes` and `from_seed` to it unchanged, so naming
//! `ChaCha12Rng` reproduces the stream `StdRng` produced word for word.
//! [`the_pin_reproduces_the_std_rng_stream`] holds that directly, and
//! `graph::structure::the_level_stream_matches_the_recorded_one` holds it end
//! to end against a stream recorded before the pin existed. `rand` already
//! depends on `rand_chacha` to build `StdRng` at all, so the direct dependency
//! adds no crate to the lockfile.
//!
//! # What is still not pinned
//!
//! Two things, and both are narrower than the algorithm was.
//!
//! `seed_from_u64` is a provided method on `rand_core::SeedableRng` rather than
//! anything `rand_chacha` writes, so the expansion from a `u64` to the thirty
//! two byte seed comes from `rand_core`. It carries no portability warning, and
//! it carries the opposite one, being "*Changing* the implementation of this
//! function should be considered a value-breaking change". A value-breaking
//! change to a `0.9` crate needs a `0.10`, which the caret constraint on `rand`
//! will not resolve to. [`the_seed_expansion_is_the_recorded_one`] records the
//! bytes regardless, so a move would be a named failure rather than a silent
//! one. The expansion is PCG32 rather than the SplitMix64 an earlier comment in
//! `pq` claimed, which changes nothing about the streams and is written down
//! here because the claim was checked.
//!
//! The distributions are `rand`'s. `Uniform<f64>`, `Uniform<usize>` and
//! `SliceRandom::shuffle` decide how stream words become a sample, and `rand`
//! promises no more about those than it does about `StdRng`. Writing them out
//! is the only way to pin them, which is a larger change than this one and buys
//! less, because the recorded level stream already fails loudly if the one that
//! builds the graph moves.

/// The generator every seeded draw in the crate runs on. See the module doc for
/// why it is named rather than taken from `StdRng`.
pub type SeededRng = rand_chacha::ChaCha12Rng;

#[cfg(test)]
mod tests {
    use super::SeededRng;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    /// The seeds the crate installs, so the two tests below cover the streams
    /// that actually build something rather than an arbitrary value.
    ///
    /// `TRAINING_SAMPLE_SEED` and the third subvector's k-means stream are the
    /// same value, since `PQ_TRAINING_SEED ^ (2 + 1)` lands on it. The two draw
    /// for unrelated purposes from separate generators, so nothing is shared
    /// between them beyond the words, and it is written down here because it
    /// looks deliberate and is not.
    const INSTALLED_SEEDS: &[u64] = &[
        // `graph::levels::DEFAULT_LEVEL_SEED`.
        0x5A45_5553_4442_5F30,
        // `pq::PQ_TRAINING_SEED`, which the sampling shuffle takes, and the
        // subvector streams derived from it as `PQ_TRAINING_SEED ^ (s + 1)`.
        0x5A_EE_5D_B0_5E_ED_57_02,
        0x5A_EE_5D_B0_5E_ED_57_03,
        0x5A_EE_5D_B0_5E_ED_57_00,
        // `collection::training::TRAINING_SAMPLE_SEED`, and `s == 2`.
        0x5A_EE_5D_B0_5E_ED_57_01,
        // Low Hamming weight values, which the expansion exists to spread.
        0,
        1,
        u64::MAX,
    ];

    /// Naming `ChaCha12Rng` reproduces what `StdRng` produced, word for word.
    ///
    /// This is the whole argument that the pin preserved every graph the crate
    /// had already built. It is written against `StdRng` deliberately, so it
    /// keeps checking the claim while `rand` still resolves to a version whose
    /// `StdRng` is ChaCha12. On the release that changes the algorithm this
    /// fails and the recorded streams elsewhere do not, which is the correct
    /// pair of outcomes: the pin held, and the thing it was pinned against
    /// moved.
    #[test]
    fn the_pin_reproduces_the_std_rng_stream() {
        for &seed in INSTALLED_SEEDS {
            let mut pinned = SeededRng::seed_from_u64(seed);
            let mut standard = StdRng::seed_from_u64(seed);
            // Past one ChaCha block, which is sixteen `u64`, so the buffer
            // refill is covered rather than only the first block.
            for word in 0..64 {
                assert_eq!(
                    pinned.next_u64(),
                    standard.next_u64(),
                    "the pinned stream diverges from StdRng at seed {:#x} word {}",
                    seed,
                    word
                );
            }
            // `fill_bytes` takes a different path through the buffer than
            // `next_u64` does, and `Uniform` reaches both.
            let mut a = [0u8; 97];
            let mut b = [0u8; 97];
            pinned.fill_bytes(&mut a);
            standard.fill_bytes(&mut b);
            assert_eq!(a, b, "fill_bytes diverges at seed {:#x}", seed);
        }
    }

    /// The bytes `seed_from_u64` expands each installed seed into.
    ///
    /// Recorded because the expansion lives in `rand_core` rather than in the
    /// crate this pin names, so it is the one step between a seed and a stream
    /// that the pin does not cover. See the module doc.
    #[test]
    fn the_seed_expansion_is_the_recorded_one() {
        /// Captures what `seed_from_u64` expanded to, since a generator does
        /// not hand its seed back. `SeedableRng` requires only `Sized`.
        struct Expansion([u8; 32]);
        impl SeedableRng for Expansion {
            type Seed = [u8; 32];
            fn from_seed(seed: Self::Seed) -> Self {
                Expansion(seed)
            }
        }

        // FNV-1a over the expansion of every installed seed in order, written
        // out here so the value does not depend on a hasher the standard
        // library may change.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &seed in INSTALLED_SEEDS {
            for byte in Expansion::seed_from_u64(seed).0 {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        println!("seed expansion hash {:#018x}", hash);
        assert_eq!(
            hash, 0xb8a5_75cd_0562_a9ef,
            "rand_core's seed_from_u64 expansion moved, so every seeded stream \
             in the crate moved with it"
        );

        // And the first one written out, so a failure above says what changed
        // rather than only that something did.
        assert_eq!(
            Expansion::seed_from_u64(INSTALLED_SEEDS[0]).0,
            [
                0xd4, 0x08, 0x93, 0xce, 0xf0, 0x2f, 0x68, 0xe7, 0x61, 0xef, 0x1f, 0x0c, 0x26, 0x12,
                0x33, 0x54, 0x4b, 0x68, 0xb8, 0x32, 0xa9, 0x40, 0xb4, 0x61, 0xf4, 0x3d, 0x02, 0x80,
                0x5a, 0x63, 0xb7, 0x3b,
            ],
            "the default level seed expands differently"
        );
    }
}
