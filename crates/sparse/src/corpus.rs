//! Synthetic sparse corpora at realistic term counts, for the tests.
//!
//! Two regimes, both assumptions about what a real corpus looks like and
//! labelled as such wherever a figure from them is quoted.
//!
//! - `text`: a tokenised passage corpus. Vocabulary 100,000, term popularity
//!   Zipf with exponent 1.0, about 45 distinct terms per record, lognormal and
//!   clipped to 5 to 400, weights small integers standing for term frequency,
//!   queries of 3 to 10 terms with unit weights.
//! - `splade`: a learned sparse encoder's output. Vocabulary 30,522, term
//!   popularity Zipf with exponent 0.7, about 180 nonzeros per record, normal
//!   and clipped to 20 to 500, positive lognormal weights, queries of about
//!   40 nonzeros.

use zeusdb_vector_core::SparseVector;

/// splitmix64. Seeded, so every run sees the same corpus.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.f64() * n as f64) as usize
    }

    /// Standard normal, by Box-Muller.
    pub(crate) fn gauss(&mut self) -> f64 {
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Zipf over `v` items by inverse CDF. Item `i` has weight `1 / (i + 1)^s`.
pub(crate) struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    pub(crate) fn new(v: usize, s: f64) -> Self {
        let mut cdf = Vec::with_capacity(v);
        let mut acc = 0f64;
        for i in 0..v {
            acc += 1.0 / ((i + 1) as f64).powf(s);
            cdf.push(acc);
        }
        let total = acc;
        for c in cdf.iter_mut() {
            *c /= total;
        }
        Self { cdf }
    }

    pub(crate) fn sample(&self, rng: &mut Rng) -> u32 {
        let u = rng.f64();
        self.cdf.partition_point(|c| *c < u).min(self.cdf.len() - 1) as u32
    }
}

pub(crate) struct Corpus {
    #[allow(dead_code)]
    pub(crate) name: &'static str,
    pub(crate) docs: Vec<SparseVector>,
    pub(crate) queries: Vec<SparseVector>,
}

fn draw(
    zipf: &Zipf,
    rng: &mut Rng,
    target: usize,
    weight: &mut dyn FnMut(&mut Rng) -> f32,
) -> SparseVector {
    let mut dims: Vec<u32> = (0..target).map(|_| zipf.sample(rng)).collect();
    dims.sort_unstable();
    dims.dedup();
    let values: Vec<f32> = dims.iter().map(|_| weight(rng)).collect();
    SparseVector { dims, values }
}

pub(crate) fn text_like(n: usize, nq: usize, seed: u64) -> Corpus {
    let mut rng = Rng::new(seed);
    let zipf = Zipf::new(100_000, 1.0);
    let mut tf = |rng: &mut Rng| -> f32 {
        // Geometric, mostly 1, sometimes 2 to 5.
        let mut t = 1;
        while rng.f64() < 0.3 && t < 5 {
            t += 1;
        }
        t as f32
    };
    let docs = (0..n)
        .map(|_| {
            let len = (45f64.ln() + 0.5 * rng.gauss())
                .exp()
                .round()
                .clamp(5.0, 400.0) as usize;
            draw(&zipf, &mut rng, len, &mut tf)
        })
        .collect();
    let queries = (0..nq)
        .map(|_| {
            let len = 3 + rng.below(8);
            draw(&zipf, &mut rng, len, &mut |_| 1.0)
        })
        .collect();
    Corpus {
        name: "text",
        docs,
        queries,
    }
}

pub(crate) fn splade_like(n: usize, nq: usize, seed: u64) -> Corpus {
    let mut rng = Rng::new(seed);
    let zipf = Zipf::new(30_522, 0.7);
    let mut w = |rng: &mut Rng| -> f32 { (0.6 * rng.gauss()).exp() as f32 };
    let docs = (0..n)
        .map(|_| {
            let len = (180.0 + 40.0 * rng.gauss()).round().clamp(20.0, 500.0) as usize;
            draw(&zipf, &mut rng, len, &mut w)
        })
        .collect();
    let queries = (0..nq)
        .map(|_| {
            let len = (40.0 + 10.0 * rng.gauss()).round().clamp(5.0, 120.0) as usize;
            draw(&zipf, &mut rng, len, &mut w)
        })
        .collect();
    Corpus {
        name: "splade",
        docs,
        queries,
    }
}

pub(crate) fn corpus(regime: &str, n: usize, nq: usize) -> Corpus {
    match regime {
        "text" => text_like(n, nq, 133),
        "splade" => splade_like(n, nq, 133),
        other => panic!("unknown regime {other}"),
    }
}
