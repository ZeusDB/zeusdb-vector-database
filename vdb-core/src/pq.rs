use rand::{rng, seq::SliceRandom, Rng};
use rayon::prelude::*;
use std::sync::RwLock;

/// Product Quantization implementation for vector compression
pub struct PQ {
    pub dim: usize,
    pub subvectors: usize,
    pub bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,

    /// Centroids: [subvector_idx][centroid_idx][dimension_within_subvector]
    /// Thread-safe storage for concurrent access during search
    pub centroids: RwLock<Vec<Vec<Vec<f32>>>>,

    /// Training status to track whether PQ has been trained
    pub is_trained: RwLock<bool>,

    /// Symmetric distance table, holding the squared L2 distance between every
    /// pair of centroids within a subvector. Flat and row major, indexed
    /// `[s * num_centroids * num_centroids + i * num_centroids + j]`.
    ///
    /// Graph construction compares two stored points, and under quantization
    /// both of those are codes rather than vectors, so there is no query to
    /// build an ADC table from. This table answers that comparison in
    /// `subvectors` lookups, which is the same cost as an ADC lookup.
    ///
    /// Empty until the codebook exists. `set_centroids` is the only way to
    /// install a codebook and it always fills this, so a trained PQ always
    /// carries a table that matches its centroids.
    sdc_table: RwLock<Vec<f32>>,

    /// Cache computed values for performance
    pub sub_dim: usize,
    pub num_centroids: usize,
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
            sdc_table: RwLock::new(Vec::new()),
            sub_dim,
            num_centroids,
        }
    }

    /// Check if PQ has been trained
    pub fn is_trained(&self) -> bool {
        *self.is_trained.read().unwrap()
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

        let table = Self::compute_sdc_table(&centroids, self.num_centroids);

        {
            let mut guard = self.centroids.write().unwrap();
            *guard = centroids;
        }
        {
            let mut guard = self.sdc_table.write().unwrap();
            *guard = table;
        }

        Ok(())
    }

    /// Compute the squared L2 distance between every pair of centroids
    ///
    /// Squared rather than rooted, because the distance the search path returns
    /// is the plain sum of the squared ADC lookups. Rooting one and not the
    /// other would put graph construction and search on different scales.
    fn compute_sdc_table(centroids: &[Vec<Vec<f32>>], num_centroids: usize) -> Vec<f32> {
        let plane = num_centroids * num_centroids;

        centroids
            .par_iter()
            .flat_map(|sub| {
                let mut rows = vec![0.0f32; plane];
                for i in 0..num_centroids {
                    for j in (i + 1)..num_centroids {
                        let d = l2_distance_squared(&sub[i], &sub[j]);
                        rows[i * num_centroids + j] = d;
                        rows[j * num_centroids + i] = d;
                    }
                }
                rows
            })
            .collect()
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
    pub fn symmetric_distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let table = self.sdc_table.read().unwrap();
        if table.is_empty() {
            return 0.0;
        }

        let k = self.num_centroids;
        let plane = k * k;
        let mut sum = 0.0f32;

        for (s, (&code_a, &code_b)) in a.iter().zip(b.iter()).enumerate() {
            let (i, j) = (code_a as usize, code_b as usize);
            if i >= k || j >= k {
                continue;
            }
            sum += table.get(s * plane + i * k + j).copied().unwrap_or(0.0);
        }

        sum
    }

    /// Bytes held by the symmetric distance table
    pub fn sdc_memory_bytes(&self) -> usize {
        self.sdc_table.read().unwrap().len() * std::mem::size_of::<f32>()
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
            let mut rng = rng();
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
                self.kmeans(&subvectors, self.num_centroids, max_iter)
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
    fn kmeans(
        &self,
        data: &[Vec<f32>],
        k: usize,
        max_iter: usize,
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
        let mut rng = rng();

        // Initialize centroids using k-means++ for better convergence
        let mut centroids = self.kmeans_plus_plus_init(data, k, &mut rng)?;

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
                // let target = rng.gen::<f32>() * total_distance;
                let target = rand::random::<f32>() * total_distance;

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
        assert_eq!(pq.dim, 128);
        assert_eq!(pq.subvectors, 8);
        assert_eq!(pq.bits, 8);
        assert_eq!(pq.sub_dim, 16);
        assert_eq!(pq.num_centroids, 256);
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

        // subvectors * 2^bits * 2^bits entries of f32
        assert_eq!(pq.sdc_memory_bytes(), 2 * 4 * 4 * 4);

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

    /// The table is indexed `[s * k * k + i * k + j]`, so a code near the top
    /// of the u8 range reads from the far end of its own subvector's plane and
    /// must not spill into the next one. Eight bits is the only setting where
    /// that boundary is reachable, and the graph tests run at six.
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

        assert_eq!(pq.sdc_memory_bytes(), 2 * 256 * 256 * 4);

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
}
