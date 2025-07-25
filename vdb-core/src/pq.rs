use std::sync::RwLock;
use rayon::prelude::*;
use rand::{seq::SliceRandom, rng, Rng};

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
        max_training_vectors: Option<usize>
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
            sub_dim,
            num_centroids,
        }
    }
    
    /// Check if PQ has been trained
    pub fn is_trained(&self) -> bool {
        *self.is_trained.read().unwrap()
    }
    
    /// Train the PQ codebook using k-means clustering
    pub fn train(&self, vectors: &[Vec<f32>]) -> Result<(), String> {
        if vectors.is_empty() {
            return Err("Cannot train on empty vector set".to_string());
        }
        
        if vectors[0].len() != self.dim {
            return Err(format!("Vector dimension mismatch: expected {}, got {}", 
                             self.dim, vectors[0].len()));
        }
        
        if vectors.len() < self.training_size {
            return Err(format!("Insufficient training data: need at least {}, got {}", 
                             self.training_size, vectors.len()));
        }
        
        let sample_size = self.max_training_vectors
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
                let max_iter = if training_vectors.len() > 50000 { 50 } else { 100 };
                self.kmeans(&subvectors, self.num_centroids, max_iter)
            })
            .collect();
        
        match new_centroids {
            Ok(centroids_vec) => {
                // Update centroids atomically
                {
                    let mut centroids = self.centroids.write().unwrap();
                    *centroids = centroids_vec;
                }
                
                // Mark as trained
                {
                    let mut trained = self.is_trained.write().unwrap();
                    *trained = true;
                }
                
                Ok(())
            }
            Err(e) => Err(e)
        }
    }
    
    /// Quantize a vector into PQ codes
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u8>, String> {
        if !self.is_trained() {
            return Err("PQ must be trained before quantization".to_string());
        }
        
        if vector.len() != self.dim {
            return Err(format!("Vector dimension mismatch: expected {}, got {}", 
                             self.dim, vector.len()));
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
                return Err(format!("Vector {}: dimension mismatch: expected {}, got {}", 
                                 i, self.dim, vector.len()));
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
            return Err(format!("Code length mismatch: expected {}, got {}", 
                             self.subvectors, codes.len()));
        }
        
        let centroids = self.centroids.read().unwrap();
        let mut vector = vec![0.0; self.dim];
        
        for s in 0..self.subvectors {
            let start_idx = s * self.sub_dim;
            let end_idx = start_idx + self.sub_dim;
            let centroid_idx = codes[s] as usize;
            
            if centroid_idx >= centroids[s].len() {
                return Err(format!("Invalid centroid index: {} for subvector {}", 
                                 centroid_idx, s));
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
            return Err(format!("Query dimension mismatch: expected {}, got {}", 
                             self.dim, query.len()));
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
    
    /// Compute ADC distance using precomputed lookup table
    pub fn adc_distance(&self, codes: &[u8], lut: &[Vec<f32>]) -> Result<f32, String> {
        if codes.len() != self.subvectors {
            return Err(format!("Code length mismatch: expected {}, got {}", 
                             self.subvectors, codes.len()));
        }
        
        if lut.len() != self.subvectors {
            return Err(format!("LUT length mismatch: expected {}, got {}", 
                             self.subvectors, lut.len()));
        }
        
        let mut distance = 0.0;
        for s in 0..self.subvectors {
            let centroid_idx = codes[s] as usize;
            if centroid_idx >= lut[s].len() {
                return Err(format!("Invalid centroid index: {} for subvector {}", 
                                 centroid_idx, s));
            }
            distance += lut[s][centroid_idx];
        }
        
        Ok(distance.sqrt()) // Convert squared distance to distance
    }
    
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
        info.insert("compression_ratio".to_string(), format!("{:.1}", compression_ratio));
        
        info
    }
    
    /// K-means clustering implementation
    fn kmeans(&self, data: &[Vec<f32>], k: usize, max_iter: usize) -> Result<Vec<Vec<f32>>, String> {
        if data.is_empty() {
            return Err("Cannot perform k-means on empty data".to_string());
        }
        
        if k > data.len() {
            return Err(format!("k ({}) cannot be larger than data size ({})", k, data.len()));
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
    fn kmeans_plus_plus_init(&self, data: &[Vec<f32>], k: usize, rng: &mut impl Rng) -> Result<Vec<Vec<f32>>, String> {
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
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum()
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
        
        let distance = pq.adc_distance(&codes, &lut).unwrap();
        assert!(distance >= 0.0);
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
