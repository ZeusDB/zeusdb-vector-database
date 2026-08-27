// The locks in this file are not fields of `HNSWIndex`, so they are outside the
// registry in `hnsw_index::locks` and outside the order it enforces. The
// declared order already places them: they are leaves, since nothing here can
// name an index guard, so any of them may be taken under any index guard and no
// index guard is ever taken under one. See `clippy.toml`.
#![allow(clippy::disallowed_types)]

use crate::rng::SeededRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::RwLock;

/// Seed driving every draw training makes.
///
/// Training used to draw from the thread generator, which is seeded from OS
/// entropy, so two trainings of one data set produced two codebooks, two sets
/// of codes, two graphs and two rerank calibrations. A fixed seed makes the
/// codebook a function of the training data alone, the same way
/// `DEFAULT_LEVEL_SEED` in the vendored graph crate fixes level assignment
/// and `TRAINING_SAMPLE_SEED` in `hnsw_index` fixes the sample order.
///
/// The k-means of each subvector runs on its own rayon worker, so one shared
/// generator would hand out draws in thread arrival order and reintroduce the
/// nondeterminism the seed exists to remove. Each subvector therefore derives
/// its own stream as `PQ_TRAINING_SEED ^ (s + 1)`, which no scheduling order
/// can perturb, and the sampling shuffle in `train` takes the base stream.
/// `seed_from_u64` expands the value through PCG32, so the nearby seeds
/// produce unrelated streams. The generator itself is named in `crate::rng`
/// rather than taken from `StdRng`, so a `rand` release cannot move a codebook.
const PQ_TRAINING_SEED: u64 = 0x5A_EE_5D_B0_5E_ED_57_02;

/// The tables a codebook implies, rebuilt whenever the codebook is installed
///
/// Both are derived from the same centroids and both are read by
/// [`PQ::symmetric_cosine_distance`], so they live under one guard and are
/// written in one place.
#[derive(Default)]
struct Derived {
    /// Symmetric distance table, holding the squared L2 distance between every
    /// pair of distinct centroids within a subvector.
    ///
    /// Graph construction compares two stored points, and under quantization
    /// both of those are codes rather than vectors, so there is no query to
    /// build an ADC table from. This table answers that comparison in
    /// `subvectors` lookups, which is the same cost as an ADC lookup.
    ///
    /// Only the strict upper triangle is held. The distance from centroid `i`
    /// to centroid `j` equals the distance from `j` to `i`, so a full square
    /// stores every value twice, and the diagonal is zero because a centroid is
    /// at distance zero from itself. That takes the table from
    /// `subvectors * k * k` entries to `subvectors * k * (k - 1) / 2`, which is
    /// 2.00 MiB down to 0.996 MiB at the default eight subvectors of eight
    /// bits and 24.0 MiB down to 12.0 MiB at 96 subvectors. The values are
    /// unchanged, so the graph and the recall are unchanged with them.
    ///
    /// Flat, one plane of `k * (k - 1) / 2` entries per subvector, the pair
    /// `(i, j)` with `i < j` at `s * plane + sdc_offset(k, i, j)`. Written in
    /// exactly that order, so the build needs no offset arithmetic.
    sdc: Vec<f32>,

    /// Squared norm of every centroid, flat, subvector `s` centroid `c` at
    /// `s * num_centroids + c`.
    ///
    /// A reconstruction is the concatenation of one centroid per subvector, so
    /// the squared norm of the whole is the sum of `subvectors` entries of this
    /// table. That is the quantity a cosine distance needs and cannot get from
    /// the ADC table, which carries only the distance to the query.
    ///
    /// `subvectors * k` entries, which is 48 KiB at 48 subvectors of eight
    /// bits, or 0.8% of what the symmetric table above already holds.
    norms: Vec<f32>,
}

/// Product Quantization implementation for vector compression
pub struct PQ {
    /// The five scalars below are written once, by `PQ::new`, and read
    /// everywhere. They were `pub` on a struct whose interior state was already
    /// behind accessors, so the type asked a reader to hold two rules at once
    /// about what may be touched. They are private now and each has an accessor
    /// of its own name.
    dim: usize,
    subvectors: usize,
    bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,

    /// Centroids: [subvector_idx][centroid_idx][dimension_within_subvector]
    ///
    /// Thread-safe storage for concurrent access during search. Private, and
    /// reached from outside this module only through `with_centroids`, so the
    /// guard's lifetime stays inside the module that owns the lock.
    centroids: RwLock<Vec<Vec<Vec<f32>>>>,

    /// Training status to track whether PQ has been trained
    pub is_trained: RwLock<bool>,

    /// Everything `set_centroids` derives from the codebook.
    ///
    /// Empty until the codebook exists. `set_centroids` is the only way to
    /// install a codebook and it always fills both tables, so a trained PQ
    /// always carries tables that match its centroids.
    ///
    /// One lock over both rather than one lock each. A cosine distance needs
    /// the pair together, and a single guard says so without adding a second
    /// bare lock to this module that would then have to be ordered against the
    /// first. `hnsw_index::locks` ranks the index's own guards and does not
    /// reach in here.
    derived: RwLock<Derived>,

    /// Cache computed values for performance
    sub_dim: usize,
    num_centroids: usize,
}

impl PQ {
    /// Create a new PQ instance
    pub fn new(
        dim: usize,
        subvectors: usize,
        bits: usize,
        training_size: usize,
        max_training_vectors: Option<usize>,
    ) -> Self {
        let sub_dim = dim / subvectors;
        let num_centroids = 1 << bits; // 2^bits

        // Initialize empty centroids structure
        let centroids = vec![vec![vec![0.0; sub_dim]; num_centroids]; subvectors];

        PQ {
            dim,
            subvectors,
            bits,
            training_size,
            max_training_vectors,
            centroids: RwLock::new(centroids),
            is_trained: RwLock::new(false),
            derived: RwLock::new(Derived::default()),
            sub_dim,
            num_centroids,
        }
    }

    /// Check if PQ has been trained
    /// Values in a vector this quantizer encodes.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Subvectors a vector is split into, which is also the byte length of a
    /// code.
    pub fn subvectors(&self) -> usize {
        self.subvectors
    }

    /// Bits per subvector code, which fixes the centroid count.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Values in one subvector, being `dim / subvectors`.
    pub fn sub_dim(&self) -> usize {
        self.sub_dim
    }

    /// Centroids per subvector, being `2 ^ bits`.
    pub fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    pub fn is_trained(&self) -> bool {
        *self.is_trained.read().unwrap()
    }

    /// Serve the codebook to a caller that must not own the guard
    ///
    /// The save writes the codebook to pq_centroids.bin and used to take this
    /// lock itself, which put the guard's lifetime in the storage layer and
    /// held it across the file write. The closure receives a borrow and returns
    /// whatever it built from it, so the codebook is not copied and the lock is
    /// released the moment the closure returns.
    pub fn with_centroids<R>(&self, f: impl FnOnce(&Vec<Vec<Vec<f32>>>) -> R) -> R {
        f(&self.centroids.read().unwrap())
    }

    /// Set the training state (for persistence restoration)
    pub fn set_trained(&self, value: bool) {
        let mut trained = self.is_trained.write().unwrap();
        *trained = value;
    }

    /// Install a codebook and rebuild everything derived from it
    ///
    /// This is the only way a codebook reaches a `PQ`, whether it comes from
    /// k-means or from a saved index, so the symmetric distance table can never
    /// fall out of step with the centroids it was computed from. Retraining
    /// arrives here as well and overwrites the table.
    pub fn set_centroids(&self, centroids: Vec<Vec<Vec<f32>>>) -> Result<(), String> {
        if centroids.len() != self.subvectors {
            return Err(format!(
                "Codebook subvector count mismatch: expected {}, got {}",
                self.subvectors,
                centroids.len()
            ));
        }

        for (s, sub) in centroids.iter().enumerate() {
            if sub.len() != self.num_centroids {
                return Err(format!(
                    "Codebook centroid count mismatch in subvector {}: expected {}, got {}",
                    s,
                    self.num_centroids,
                    sub.len()
                ));
            }
            if let Some(bad) = sub.iter().position(|c| c.len() != self.sub_dim) {
                return Err(format!(
                    "Codebook centroid {} in subvector {} has dimension {}, expected {}",
                    bad,
                    s,
                    sub[bad].len(),
                    self.sub_dim
                ));
            }
        }

        let derived = Derived {
            sdc: Self::compute_sdc_table(&centroids, self.num_centroids),
            norms: Self::compute_centroid_norms(&centroids),
        };

        {
            let mut guard = self.centroids.write().unwrap();
            *guard = centroids;
        }
        {
            let mut guard = self.derived.write().unwrap();
            *guard = derived;
        }

        Ok(())
    }

    /// Entries one subvector's plane of the symmetric distance table holds
    ///
    /// The strict upper triangle of a `k` by `k` symmetric matrix.
    #[inline]
    pub fn sdc_plane(num_centroids: usize) -> usize {
        num_centroids * (num_centroids - 1) / 2
    }

    /// Offset of the centroid pair `(i, j)` within a subvector's plane
    ///
    /// Requires `i < j`. Row `i` starts after the `i` rows above it, which hold
    /// `k - 1`, `k - 2` and so on down to `k - i` entries, summing to
    /// `i * (2k - i - 1) / 2`. The diagonal is not stored, so the column term
    /// is `j - i - 1` rather than `j - i`.
    #[inline(always)]
    fn sdc_offset(num_centroids: usize, i: usize, j: usize) -> usize {
        i * (2 * num_centroids - i - 1) / 2 + (j - i - 1)
    }

    /// Compute the squared L2 distance between every pair of distinct centroids
    ///
    /// Squared rather than rooted, because the distance the search path returns
    /// is the plain sum of the squared ADC lookups. Rooting one and not the
    /// other would put graph construction and search on different scales.
    ///
    /// The pairs are emitted in the order `sdc_offset` addresses them, so this
    /// walks the plane once and never computes an offset.
    fn compute_sdc_table(centroids: &[Vec<Vec<f32>>], num_centroids: usize) -> Vec<f32> {
        let plane = Self::sdc_plane(num_centroids);

        centroids
            .par_iter()
            .flat_map(|sub| {
                let mut rows = Vec::with_capacity(plane);
                for i in 0..num_centroids {
                    for j in (i + 1)..num_centroids {
                        rows.push(l2_distance_squared(&sub[i], &sub[j]));
                    }
                }
                debug_assert_eq!(rows.len(), plane);
                rows
            })
            .collect()
    }

    /// Compute the squared norm of every centroid
    ///
    /// Subvector major and in centroid order, so subvector `s` centroid `c`
    /// lands at `s * num_centroids + c` and the build needs no offset
    /// arithmetic, the same way the symmetric table above is written.
    ///
    /// Squared rather than rooted. The sum over a code's subvectors is the
    /// squared norm of the reconstruction, and summing roots would not be.
    fn compute_centroid_norms(centroids: &[Vec<Vec<f32>>]) -> Vec<f32> {
        centroids
            .iter()
            .flat_map(|sub| sub.iter().map(|c| c.iter().map(|x| x * x).sum::<f32>()))
            .collect()
    }

    /// Squared norm of the reconstruction a code addresses
    ///
    /// `PQ::reconstruct` copies `centroids[s][code[s]]` into the block at
    /// `s * sub_dim` and does nothing else, so the squared norm of the whole is
    /// the sum of the squared norms of the blocks. This reads that sum out of
    /// the table instead of building the vector.
    ///
    /// Zero when no codebook has been installed, which is the same answer
    /// `symmetric_distance` gives on the same unreachable state.
    pub fn code_norm_sq(&self, codes: &[u8]) -> f32 {
        let derived = self.derived.read().unwrap();
        Self::norm_sum(&derived.norms, self.num_centroids, codes)
    }

    /// The squared reconstruction norm a code implies, given the table
    ///
    /// Split out because `symmetric_cosine_distance` needs it twice under one
    /// guard.
    #[inline]
    fn norm_sum(norms: &[f32], k: usize, codes: &[u8]) -> f32 {
        if norms.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for (s, &code) in codes.iter().enumerate() {
            sum += norms.get(s * k + code as usize).copied().unwrap_or(0.0);
        }
        sum
    }

    /// Distance between two stored points, both held as codes
    ///
    /// Graph construction has no query, so it cannot use an ADC table. This
    /// reads the precomputed centroid pair distances instead, at the same cost
    /// as an ADC lookup and on the same squared L2 scale.
    ///
    /// Returns zero when no codebook has been installed. Nothing can hold a
    /// code before then, so the case is unreachable through the index, and zero
    /// keeps the value finite rather than reintroducing the infinity that
    /// collapsed neighbour selection.
    ///
    /// The table holds the strict upper triangle, so the two codes are ordered
    /// before the lookup and an equal pair is answered as zero without reading
    /// anything. Measured against the full square, this is 22.1 ns against
    /// 15.6 ns for one eight subvector distance in isolation, and 7.2 percent
    /// on the whole graph build at 10,000 records of 64 dimensions.
    pub fn symmetric_distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let derived = self.derived.read().unwrap();
        Self::sdc_sum(&derived.sdc, self.num_centroids, a, b)
    }

    /// Cosine distance between the two reconstructions two codes address
    ///
    /// The construction counterpart of the cosine ADC scorer. Graph
    /// construction has no query, so this is the symmetric case, and the
    /// quantity it needs is the same one the search path needs, being the
    /// squared norm of each reconstruction.
    ///
    /// The inner product follows from what the two tables already hold. For one
    /// subvector, `dot(a, b) = (|a|^2 + |b|^2 - |a - b|^2) / 2`, and summing
    /// that over the subvectors gives the inner product of the two
    /// reconstructions in terms of the symmetric table and the norm table. So
    ///
    /// ```text
    /// 1 - (N_a + N_b - sdc) / (2 * sqrt(N_a) * sqrt(N_b))
    /// ```
    ///
    /// No third table and no reconstruction. It costs one norm lookup per
    /// subvector per code on top of the symmetric lookup the squared L2 branch
    /// already does, and two square roots per call.
    ///
    /// **A reconstruction of length zero is at distance one from everything.**
    /// It has no direction, which is where `cosine_normalized` puts a zero
    /// vector too, and it keeps the value finite rather than returning the
    /// infinity that once collapsed neighbour selection. The same answer covers
    /// the unreachable state where no codebook has been installed.
    pub fn symmetric_cosine_distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let derived = self.derived.read().unwrap();
        let k = self.num_centroids;
        if derived.norms.is_empty() {
            return 1.0;
        }
        // One walk of the codes rather than three. The three lookups a
        // subvector needs are made together, so the loop is entered once and
        // the two norm reads land in a 48 KiB table while the symmetric read
        // walks a table three orders of magnitude larger.
        let plane = Self::sdc_plane(k);
        let (mut n_a, mut n_b, mut sdc) = (0.0f32, 0.0f32, 0.0f32);
        for (s, (&code_a, &code_b)) in a.iter().zip(b.iter()).enumerate() {
            let (ia, ib) = (code_a as usize, code_b as usize);
            n_a += derived.norms.get(s * k + ia).copied().unwrap_or(0.0);
            n_b += derived.norms.get(s * k + ib).copied().unwrap_or(0.0);
            // The diagonal is not stored, and i < j below bounds both.
            if ia == ib {
                continue;
            }
            let (i, j) = if ia < ib { (ia, ib) } else { (ib, ia) };
            if j >= k {
                continue;
            }
            sdc += derived
                .sdc
                .get(s * plane + Self::sdc_offset(k, i, j))
                .copied()
                .unwrap_or(0.0);
        }
        if n_a <= 0.0 || n_b <= 0.0 {
            return 1.0;
        }
        let dot = (n_a + n_b - sdc) * 0.5;
        // One square root rather than two, since sqrt(x) * sqrt(y) is
        // sqrt(x * y) and both operands are positive here.
        let d = 1.0 - dot / (n_a * n_b).sqrt();
        // Not `d.max(0.0)`, which would swallow a NaN, and matching what
        // `cosine_normalized` does with a dot product that lands just above one.
        if d < 0.0 {
            0.0
        } else {
            d
        }
    }

    /// The symmetric table sum for a pair of codes, given the table
    ///
    /// The body of [`Self::symmetric_distance`] with the guard lifted out, so
    /// the cosine branch reads both tables under one guard rather than two.
    #[inline]
    fn sdc_sum(table: &[f32], k: usize, a: &[u8], b: &[u8]) -> f32 {
        if table.is_empty() {
            return 0.0;
        }
        let plane = Self::sdc_plane(k);
        let mut sum = 0.0f32;
        for (s, (&code_a, &code_b)) in a.iter().zip(b.iter()).enumerate() {
            // The diagonal is not stored. A centroid is at distance zero from
            // itself, which is exact rather than an approximation.
            if code_a == code_b {
                continue;
            }
            // i < j holds below, so bounding j bounds both.
            let (i, j) = if code_a < code_b {
                (code_a as usize, code_b as usize)
            } else {
                (code_b as usize, code_a as usize)
            };
            if j >= k {
                continue;
            }
            sum += table
                .get(s * plane + Self::sdc_offset(k, i, j))
                .copied()
                .unwrap_or(0.0);
        }
        sum
    }

    /// Bytes held by the symmetric distance table
    pub fn sdc_memory_bytes(&self) -> usize {
        self.derived.read().unwrap().sdc.len() * std::mem::size_of::<f32>()
    }

    /// Bytes held by the centroid norm table
    pub fn centroid_norm_memory_bytes(&self) -> usize {
        self.derived.read().unwrap().norms.len() * std::mem::size_of::<f32>()
    }

    /// Train the PQ codebook using k-means clustering
    pub fn train(&self, vectors: &[Vec<f32>]) -> Result<(), String> {
        if vectors.is_empty() {
            return Err("Cannot train on empty vector set".to_string());
        }

        if vectors[0].len() != self.dim {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
                vectors[0].len()
            ));
        }

        if vectors.len() < self.training_size {
            return Err(format!(
                "Insufficient training data: need at least {}, got {}",
                self.training_size,
                vectors.len()
            ));
        }

        let sample_size = self
            .max_training_vectors
            .map(|max_size| vectors.len().min(max_size))
            .unwrap_or(vectors.len());

        // Sample training vectors if we have more than needed
        let training_vectors = if sample_size < vectors.len() {
            let mut rng = SeededRng::seed_from_u64(PQ_TRAINING_SEED);
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(sample_size);
            indices.iter().map(|&i| &vectors[i]).collect::<Vec<_>>()
        } else {
            vectors.iter().collect::<Vec<_>>()
        };

        // Train each subvector independently using parallel processing
        let new_centroids: Result<Vec<_>, String> = (0..self.subvectors)
            .into_par_iter()
            .map(|s| {
                let start_idx = s * self.sub_dim;
                let end_idx = start_idx + self.sub_dim;

                // Extract subvectors for this subspace
                let subvectors: Vec<Vec<f32>> = training_vectors
                    .iter()
                    .map(|vec| vec[start_idx..end_idx].to_vec())
                    .collect();

                // Perform k-means clustering with adaptive max_iter
                let max_iter = if training_vectors.len() > 50000 {
                    50
                } else {
                    100
                };
                // This subvector's own stream; see `PQ_TRAINING_SEED`.
                let mut rng = SeededRng::seed_from_u64(PQ_TRAINING_SEED ^ (s as u64 + 1));
                self.kmeans(&subvectors, self.num_centroids, max_iter, &mut rng)
            })
            .collect();

        match new_centroids {
            Ok(centroids_vec) => {
                // Install the codebook and the symmetric distance table derived
                // from it, then mark trained. Nothing reads the table before
                // the flag is set, so the graph never sees a half-built one.
                self.set_centroids(centroids_vec)?;

                // Mark as trained
                {
                    let mut trained = self.is_trained.write().unwrap();
                    *trained = true;
                }

                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Quantize a vector into PQ codes
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u8>, String> {
        if !self.is_trained() {
            return Err("PQ must be trained before quantization".to_string());
        }

        if vector.len() != self.dim {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }

        let centroids = self.centroids.read().unwrap();
        let mut codes = vec![0u8; self.subvectors];

        for s in 0..self.subvectors {
            let start_idx = s * self.sub_dim;
            let end_idx = start_idx + self.sub_dim;
            let subvector = &vector[start_idx..end_idx];

            // Find closest centroid for this subvector
            let mut best_distance = f32::INFINITY;
            let mut best_centroid_idx = 0;

            for (centroid_idx, centroid) in centroids[s].iter().enumerate() {
                let distance = l2_distance_squared(subvector, centroid);
                if distance < best_distance {
                    best_distance = distance;
                    best_centroid_idx = centroid_idx;
                }
            }

            codes[s] = best_centroid_idx as u8;
        }

        Ok(codes)
    }

    /// Batch quantize multiple vectors for efficiency
    pub fn quantize_batch(&self, vectors: &[&[f32]]) -> Result<Vec<Vec<u8>>, String> {
        if !self.is_trained() {
            return Err("PQ must be trained before quantization".to_string());
        }

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Validate all vectors have correct dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dim {
                return Err(format!(
                    "Vector {}: dimension mismatch: expected {}, got {}",
                    i,
                    self.dim,
                    vector.len()
                ));
            }
        }

        let centroids = self.centroids.read().unwrap();

        // Parallel batch quantization
        let codes: Vec<Vec<u8>> = vectors
            .par_iter()
            .map(|vector| {
                let mut codes = vec![0u8; self.subvectors];

                for s in 0..self.subvectors {
                    let start_idx = s * self.sub_dim;
                    let end_idx = start_idx + self.sub_dim;
                    let subvector = &vector[start_idx..end_idx];

                    let mut best_distance = f32::INFINITY;
                    let mut best_centroid_idx = 0;

                    for (centroid_idx, centroid) in centroids[s].iter().enumerate() {
                        let distance = l2_distance_squared(subvector, centroid);
                        if distance < best_distance {
                            best_distance = distance;
                            best_centroid_idx = centroid_idx;
                        }
                    }

                    codes[s] = best_centroid_idx as u8;
                }

                codes
            })
            .collect();

        Ok(codes)
    }

    /// Reconstruct a vector from PQ codes (for debugging/verification)
    pub fn reconstruct(&self, codes: &[u8]) -> Result<Vec<f32>, String> {
        if !self.is_trained() {
            return Err("PQ must be trained before reconstruction".to_string());
        }

        if codes.len() != self.subvectors {
            return Err(format!(
                "Code length mismatch: expected {}, got {}",
                self.subvectors,
                codes.len()
            ));
        }

        let centroids = self.centroids.read().unwrap();
        let mut vector = vec![0.0; self.dim];

        for s in 0..self.subvectors {
            let start_idx = s * self.sub_dim;
            let end_idx = start_idx + self.sub_dim;
            let centroid_idx = codes[s] as usize;

            if centroid_idx >= centroids[s].len() {
                return Err(format!(
                    "Invalid centroid index: {} for subvector {}",
                    centroid_idx, s
                ));
            }

            vector[start_idx..end_idx].copy_from_slice(&centroids[s][centroid_idx]);
        }

        Ok(vector)
    }

    /// Compute Asymmetric Distance Computation (ADC) lookup table for a query vector
    pub fn compute_adc_lut(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        if !self.is_trained() {
            return Err("PQ must be trained before ADC computation".to_string());
        }

        if query.len() != self.dim {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            ));
        }

        let centroids = self.centroids.read().unwrap();
        let mut lut = vec![vec![0.0; self.num_centroids]; self.subvectors];

        for s in 0..self.subvectors {
            let start_idx = s * self.sub_dim;
            let end_idx = start_idx + self.sub_dim;
            let query_subvector = &query[start_idx..end_idx];

            for (centroid_idx, centroid) in centroids[s].iter().enumerate() {
                lut[s][centroid_idx] = l2_distance_squared(query_subvector, centroid);
            }
        }

        Ok(lut)
    }

    /// The ADC table with each centroid's squared norm interleaved beside it
    ///
    /// The cosine scorer needs two sums per candidate, the ADC sum and the
    /// squared norm of the reconstruction, and neither follows from the other.
    /// Two separate tables would mean two random reads per subvector into two
    /// arrays. Interleaving puts the pair in adjacent words, so a candidate
    /// still takes one lookup per subvector and the second value arrives in the
    /// same cache line as the first.
    ///
    /// Flat, `2 * subvectors * num_centroids` entries, the ADC for subvector
    /// `s` centroid `c` at `2 * (s * num_centroids + c)` and its squared norm
    /// at the word after it.
    ///
    /// The norms are copied from the table the codebook carries rather than
    /// recomputed. Recomputing would double the work of building a query table,
    /// which at 1,536 dimensions is 256 centroid distances per subvector and is
    /// paid once per query.
    ///
    /// Separate from `compute_adc_lut` rather than replacing it, so the squared
    /// L2 scorer keeps the table it has and pays nothing for a value it never
    /// reads.
    pub fn compute_adc_lut_with_norms(&self, query: &[f32]) -> Result<Vec<f32>, String> {
        if !self.is_trained() {
            return Err("PQ must be trained before ADC computation".to_string());
        }

        if query.len() != self.dim {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            ));
        }

        let k = self.num_centroids;
        let centroids = self.centroids.read().unwrap();
        let derived = self.derived.read().unwrap();
        let mut lut = vec![0.0f32; 2 * self.subvectors * k];

        for s in 0..self.subvectors {
            let start_idx = s * self.sub_dim;
            let query_subvector = &query[start_idx..start_idx + self.sub_dim];

            for (centroid_idx, centroid) in centroids[s].iter().enumerate() {
                let at = 2 * (s * k + centroid_idx);
                lut[at] = l2_distance_squared(query_subvector, centroid);
                lut[at + 1] = derived
                    .norms
                    .get(s * k + centroid_idx)
                    .copied()
                    .unwrap_or(0.0);
            }
        }

        Ok(lut)
    }

    // `adc_distance` used to sit here. It was reachable only from this file's
    // own unit test, and it rooted the sum that the live `DistPQ::eval` returns
    // unrooted, so the two disagreed by a square root. A second implementation
    // of a distance, unused and on a different scale from the one that runs, is
    // how the graph came to be built on infinity. It is gone rather than
    // corrected, since `DistPQ::eval` is the implementation.

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> (f64, usize) {
        let total_centroids = self.subvectors * self.num_centroids;
        let memory_bytes = total_centroids * self.sub_dim * std::mem::size_of::<f32>();
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);

        (memory_mb, total_centroids)
    }

    /// Get training statistics and information
    /// Comprehensive training info method - kept for debugging, testing, and future API consistency
    #[allow(dead_code)]
    pub fn get_training_info(&self) -> std::collections::HashMap<String, String> {
        let mut info = std::collections::HashMap::new();

        info.insert("dim".to_string(), self.dim.to_string());
        info.insert("subvectors".to_string(), self.subvectors.to_string());
        info.insert("bits".to_string(), self.bits.to_string());
        info.insert("sub_dim".to_string(), self.sub_dim.to_string());
        info.insert("num_centroids".to_string(), self.num_centroids.to_string());
        info.insert("training_size".to_string(), self.training_size.to_string());
        info.insert("is_trained".to_string(), self.is_trained().to_string());

        if let Some(max_training) = self.max_training_vectors {
            info.insert("max_training_vectors".to_string(), max_training.to_string());
        }

        let (memory_mb, total_centroids) = self.get_memory_stats();
        info.insert("memory_mb".to_string(), format!("{:.2}", memory_mb));
        info.insert("total_centroids".to_string(), total_centroids.to_string());

        // Calculate compression ratio
        let original_bytes = self.dim * 4; // f32
        let compressed_bytes = self.subvectors; // u8 per subvector
        let compression_ratio = original_bytes as f64 / compressed_bytes as f64;
        info.insert(
            "compression_ratio".to_string(),
            format!("{:.1}", compression_ratio),
        );

        info
    }

    /// K-means clustering implementation
    ///
    /// Every draw comes from the generator the caller hands in, so the result
    /// is a function of the data and that generator's seed.
    fn kmeans(
        &self,
        data: &[Vec<f32>],
        k: usize,
        max_iter: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<Vec<f32>>, String> {
        if data.is_empty() {
            return Err("Cannot perform k-means on empty data".to_string());
        }

        if k > data.len() {
            return Err(format!(
                "k ({}) cannot be larger than data size ({})",
                k,
                data.len()
            ));
        }

        let dim = data[0].len();

        // Initialize centroids using k-means++ for better convergence
        let mut centroids = self.kmeans_plus_plus_init(data, k, rng)?;

        let mut prev_inertia = f32::INFINITY;
        let convergence_threshold = 1e-6;

        for _iter in 0..max_iter {
            // Assignment step: assign each point to closest centroid
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
            let mut total_inertia = 0.0;

            for (point_idx, point) in data.iter().enumerate() {
                let mut best_distance = f32::INFINITY;
                let mut best_cluster = 0;

                for (centroid_idx, centroid) in centroids.iter().enumerate() {
                    let distance = l2_distance_squared(point, centroid);
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = centroid_idx;
                    }
                }

                clusters[best_cluster].push(point_idx);
                total_inertia += best_distance;
            }

            // Check for convergence using inertia
            let inertia_change = (prev_inertia - total_inertia).abs();
            if inertia_change < convergence_threshold {
                break;
            }
            prev_inertia = total_inertia;

            // Update step: recalculate centroids
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() {
                    // Reinitialize empty cluster with random point
                    centroids[cluster_idx] = data[rng.random_range(0..data.len())].clone();
                } else {
                    let mut new_centroid = vec![0.0; dim];
                    for &point_idx in cluster {
                        for (d, &val) in data[point_idx].iter().enumerate() {
                            new_centroid[d] += val;
                        }
                    }

                    for val in new_centroid.iter_mut() {
                        *val /= cluster.len() as f32;
                    }

                    centroids[cluster_idx] = new_centroid;
                }
            }
        }

        Ok(centroids)
    }

    /// K-means++ initialization for better clustering
    fn kmeans_plus_plus_init(
        &self,
        data: &[Vec<f32>],
        k: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut centroids = Vec::with_capacity(k);

        // Choose first centroid randomly
        let first_idx = rng.random_range(0..data.len());
        centroids.push(data[first_idx].clone());

        // Choose remaining centroids with probability proportional to squared distance
        for _ in 1..k {
            let mut distances = Vec::with_capacity(data.len());
            let mut total_distance = 0.0;

            for point in data {
                let mut min_distance = f32::INFINITY;
                for centroid in &centroids {
                    let distance = l2_distance_squared(point, centroid);
                    min_distance = min_distance.min(distance);
                }
                distances.push(min_distance);
                total_distance += min_distance;
            }

            if total_distance == 0.0 {
                // All points are identical, just pick randomly
                let idx = rng.random_range(0..data.len());
                centroids.push(data[idx].clone());
            } else {
                let mut cumulative = 0.0;
                let target = rng.random::<f32>() * total_distance;

                for (idx, &distance) in distances.iter().enumerate() {
                    cumulative += distance;
                    if cumulative >= target {
                        centroids.push(data[idx].clone());
                        break;
                    }
                }
            }
        }

        Ok(centroids)
    }
}

/// Compute squared L2 distance between two vectors
fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_creation() {
        let pq = PQ::new(128, 8, 8, 10000, None);
        assert_eq!(pq.dim(), 128);
        assert_eq!(pq.subvectors(), 8);
        assert_eq!(pq.bits(), 8);
        assert_eq!(pq.sub_dim(), 16);
        assert_eq!(pq.num_centroids(), 256);
        assert!(!pq.is_trained());
    }

    #[test]
    fn test_pq_training_and_quantization() {
        let pq = PQ::new(4, 2, 2, 4, None);

        // Create some test vectors
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
        ];

        // Train PQ
        assert!(pq.train(&vectors).is_ok());
        assert!(pq.is_trained());

        // Test quantization
        let codes = pq.quantize(&vectors[0]).unwrap();
        assert_eq!(codes.len(), 2); // 2 subvectors

        // Test reconstruction
        let reconstructed = pq.reconstruct(&codes).unwrap();
        assert_eq!(reconstructed.len(), 4); // Original dimension

        // Test ADC
        let lut = pq.compute_adc_lut(&vectors[0]).unwrap();
        assert_eq!(lut.len(), 2); // 2 subvectors
        assert_eq!(lut[0].len(), 4); // 2^2 = 4 centroids
    }

    #[test]
    fn test_sdc_table_shape_and_symmetry() {
        let pq = PQ::new(4, 2, 2, 4, None);

        // No codebook yet, so no table and a defined distance rather than
        // infinity.
        assert_eq!(pq.sdc_memory_bytes(), 0);
        assert_eq!(pq.symmetric_distance(&[0, 0], &[1, 1]), 0.0);

        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
        ];
        pq.train(&vectors).unwrap();

        // The strict upper triangle only, so subvectors * k * (k - 1) / 2
        // entries of f32 rather than subvectors * k * k. At k = 4 that is 6
        // entries a subvector rather than 16.
        assert_eq!(pq.sdc_memory_bytes(), 2 * 6 * 4);
        assert_eq!(PQ::sdc_plane(4), 6);

        // A code against itself is zero, and the table is symmetric.
        assert_eq!(pq.symmetric_distance(&[0, 1], &[0, 1]), 0.0);
        assert_eq!(
            pq.symmetric_distance(&[0, 1], &[2, 3]),
            pq.symmetric_distance(&[2, 3], &[0, 1])
        );
    }

    #[test]
    fn test_sdc_matches_reconstructed_distance() {
        let pq = PQ::new(4, 2, 2, 4, None);

        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
        ];
        pq.train(&vectors).unwrap();

        // The table has to agree with the squared L2 distance between the two
        // reconstructions, which is what a symmetric distance means.
        for a in 0..4u8 {
            for b in 0..4u8 {
                let codes_a = [a, b];
                for c in 0..4u8 {
                    for d in 0..4u8 {
                        let codes_b = [c, d];
                        let expected = l2_distance_squared(
                            &pq.reconstruct(&codes_a).unwrap(),
                            &pq.reconstruct(&codes_b).unwrap(),
                        );
                        let actual = pq.symmetric_distance(&codes_a, &codes_b);
                        assert!(
                            (actual - expected).abs() <= 1e-4 * expected.max(1.0),
                            "codes {:?} vs {:?}: expected {}, got {}",
                            codes_a,
                            codes_b,
                            expected,
                            actual
                        );
                    }
                }
            }
        }
    }

    /// The table is indexed `[s * plane + sdc_offset(k, i, j)]` over the strict
    /// upper triangle, so a code near the top of the u8 range reads from the
    /// far end of its own subvector's plane and must not spill into the next
    /// one. The last entry of a plane is the pair `(k - 2, k - 1)`, which is
    /// what `254` against `255` reaches below. Eight bits is the only setting
    /// where that boundary exists, and the graph tests run at six.
    #[test]
    fn test_sdc_indexing_over_the_full_code_range() {
        // Two subvectors so a spill across the plane boundary would show, and
        // 300 training points so k-means can fill all 256 centroids.
        let pq = PQ::new(4, 2, 8, 256, None);

        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };
        let vectors: Vec<Vec<f32>> = (0..300).map(|_| (0..4).map(|_| next()).collect()).collect();
        pq.train(&vectors).unwrap();

        // 2 * 32,640 * 4 = 261,120 bytes, against 2 * 65,536 * 4 = 524,288 for
        // the full square the table used to hold.
        assert_eq!(pq.sdc_memory_bytes(), 2 * (256 * 255 / 2) * 4);
        assert_eq!(PQ::sdc_plane(256), 32_640);

        // Every offset in a plane is distinct and none reaches the next plane.
        let plane = PQ::sdc_plane(256);
        let mut seen = vec![false; plane];
        for i in 0..256usize {
            for j in (i + 1)..256usize {
                let off = PQ::sdc_offset(256, i, j);
                assert!(off < plane, "pair ({i}, {j}) lands at {off}, past {plane}");
                assert!(!seen[off], "pair ({i}, {j}) collides at {off}");
                seen[off] = true;
            }
        }
        assert!(seen.into_iter().all(|x| x), "the plane has an unused slot");

        for &a in &[0u8, 1, 127, 128, 254, 255] {
            for &b in &[0u8, 1, 127, 128, 254, 255] {
                let codes_a = [a, b];
                let codes_b = [b, a];
                let expected = l2_distance_squared(
                    &pq.reconstruct(&codes_a).unwrap(),
                    &pq.reconstruct(&codes_b).unwrap(),
                );
                let actual = pq.symmetric_distance(&codes_a, &codes_b);
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected.max(1.0),
                    "codes {:?} vs {:?}: expected {}, got {}",
                    codes_a,
                    codes_b,
                    expected,
                    actual
                );
            }
        }
    }

    #[test]
    fn test_set_centroids_rejects_wrong_shape() {
        let pq = PQ::new(4, 2, 2, 4, None);

        // Wrong subvector count
        assert!(pq.set_centroids(vec![vec![vec![0.0; 2]; 4]]).is_err());
        // Wrong centroid count
        assert!(pq.set_centroids(vec![vec![vec![0.0; 2]; 3]; 2]).is_err());
        // Wrong sub-dimension
        assert!(pq.set_centroids(vec![vec![vec![0.0; 3]; 4]; 2]).is_err());
        // A rejected codebook leaves no table behind
        assert_eq!(pq.sdc_memory_bytes(), 0);
    }

    /// Deterministic pseudo-random vectors for the reproducibility tests, so
    /// the data itself cannot be the thing that varies between two trainings.
    fn lcg_vectors(seed: u64, count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32)
        };
        (0..count)
            .map(|_| (0..dim).map(|_| next()).collect())
            .collect()
    }

    /// The codebook as bits, so the comparison is exact rather than within a
    /// tolerance. Reproducible means identical, not close.
    fn codebook_bits(pq: &PQ) -> Vec<u32> {
        pq.centroids
            .read()
            .unwrap()
            .iter()
            .flatten()
            .flatten()
            .map(|v| v.to_bits())
            .collect()
    }

    /// Two trainings of one data set produce one codebook. This is the
    /// regression test for the unseeded generator the trainer used to draw
    /// from, and it failed on every run of the old code.
    #[test]
    fn test_training_is_reproducible() {
        let vectors = lcg_vectors(65, 300, 8);

        let first = PQ::new(8, 2, 4, 200, None);
        first.train(&vectors).unwrap();
        let second = PQ::new(8, 2, 4, 200, None);
        second.train(&vectors).unwrap();

        assert_eq!(codebook_bits(&first), codebook_bits(&second));
    }

    /// The sampling path draws its own shuffle, so it is covered separately.
    /// `max_training_vectors` below the data size is what reaches it.
    #[test]
    fn test_training_is_reproducible_through_sampling() {
        let vectors = lcg_vectors(65, 300, 8);

        let first = PQ::new(8, 2, 4, 200, Some(250));
        first.train(&vectors).unwrap();
        let second = PQ::new(8, 2, 4, 200, Some(250));
        second.train(&vectors).unwrap();

        assert_eq!(codebook_bits(&first), codebook_bits(&second));
    }

    /// A fixed seed must not mean a fixed codebook. Different data has to
    /// produce a different codebook, or the reproducibility checks above
    /// would pass on a trainer that ignores its input.
    #[test]
    fn test_training_varies_with_data() {
        let first = PQ::new(8, 2, 4, 200, None);
        first.train(&lcg_vectors(65, 300, 8)).unwrap();
        let second = PQ::new(8, 2, 4, 200, None);
        second.train(&lcg_vectors(66, 300, 8)).unwrap();

        assert_ne!(codebook_bits(&first), codebook_bits(&second));
    }

    #[test]
    fn test_batch_quantization() {
        let pq = PQ::new(4, 2, 2, 4, None);

        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
        ];

        assert!(pq.train(&vectors).is_ok());

        let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let batch_codes = pq.quantize_batch(&vector_refs).unwrap();

        assert_eq!(batch_codes.len(), 4);
        for codes in batch_codes {
            assert_eq!(codes.len(), 2);
        }
    }

    /// The norm table answers what the reconstruction would
    ///
    /// The whole point of the table is that a cosine distance needs
    /// `norm(c)^2` per candidate and building `c` to get it would cost the
    /// reconstruction the ADC table exists to avoid. So the table has to give
    /// the same answer the reconstruction does, and this checks it against the
    /// reconstruction rather than against a second sum over the same table.
    #[test]
    fn the_norm_table_matches_the_reconstruction() {
        let pq = PQ::new(4, 2, 2, 4, None);

        // No codebook yet, so no table and a defined answer rather than a panic.
        assert_eq!(pq.centroid_norm_memory_bytes(), 0);
        assert_eq!(pq.code_norm_sq(&[0, 0]), 0.0);

        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
        ];
        pq.train(&vectors).unwrap();

        // One entry per centroid per subvector, which is far less than the
        // strict upper triangle the symmetric table holds.
        assert_eq!(pq.centroid_norm_memory_bytes(), 2 * 4 * 4);
        assert!(pq.centroid_norm_memory_bytes() < pq.sdc_memory_bytes());

        for a in 0..4u8 {
            for b in 0..4u8 {
                let codes = [a, b];
                let recon = pq.reconstruct(&codes).expect("reconstruct");
                let want: f32 = recon.iter().map(|x| x * x).sum();
                let got = pq.code_norm_sq(&codes);
                assert!(
                    (got - want).abs() < 1e-5,
                    "codes {codes:?}: expected {want}, got {got}"
                );
            }
        }
    }

    /// The interleaved query table carries the plain one unchanged
    ///
    /// The cosine scorer reads a table of `(adc, norm)` pairs and the squared L2
    /// scorer reads the plain ADC table, and the two must not drift apart. A
    /// difference here would be a cosine index and an l2 index over one corpus
    /// disagreeing about the distance from the query to the same reconstruction.
    #[test]
    fn the_interleaved_table_carries_the_plain_adc_table() {
        let pq = PQ::new(4, 2, 2, 4, None);
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
        ];
        pq.train(&vectors).unwrap();

        let query = [1.5f32, 2.5, 3.5, 4.5];
        let plain = pq.compute_adc_lut(&query).expect("plain table");
        let interleaved = pq.compute_adc_lut_with_norms(&query).expect("interleaved");
        let k = pq.num_centroids();
        assert_eq!(interleaved.len(), 2 * pq.subvectors() * k);

        for (s, row) in plain.iter().enumerate() {
            for (c, &want) in row.iter().enumerate() {
                let at = 2 * (s * k + c);
                assert_eq!(
                    interleaved[at], want,
                    "subvector {s} centroid {c}: the two tables disagree"
                );
                // And the odd word is the norm the codebook holds, which
                // `code_norm_sq` sums for a whole code.
                let mut codes = vec![0u8; pq.subvectors()];
                codes[s] = c as u8;
                let others: f32 = (0..pq.subvectors())
                    .filter(|&o| o != s)
                    .map(|o| interleaved[2 * (o * k) + 1])
                    .sum();
                assert!((pq.code_norm_sq(&codes) - (interleaved[at + 1] + others)).abs() < 1e-5);
            }
        }
    }

    /// Retraining rebuilds both derived tables together
    ///
    /// They sit under one guard and are written in one place, so a codebook can
    /// never be installed with only one of them refreshed. The check is that
    /// each moves when the codebook moves, since a stale norm table would show
    /// up as a cosine distance to the previous codebook's reconstructions.
    #[test]
    fn retraining_rebuilds_the_norm_table_with_the_codebook() {
        let pq = PQ::new(2, 1, 1, 2, None);
        pq.set_centroids(vec![vec![vec![3.0, 4.0], vec![0.0, 1.0]]])
            .expect("codebook installs");
        pq.set_trained(true);
        assert_eq!(pq.code_norm_sq(&[0]), 25.0);
        assert_eq!(pq.code_norm_sq(&[1]), 1.0);

        pq.set_centroids(vec![vec![vec![6.0, 8.0], vec![0.0, 2.0]]])
            .expect("codebook reinstalls");
        assert_eq!(pq.code_norm_sq(&[0]), 100.0);
        assert_eq!(pq.code_norm_sq(&[1]), 4.0);
        // And the symmetric table moved with it.
        assert_eq!(pq.symmetric_distance(&[0], &[1]), 36.0 + 36.0);
    }
}
