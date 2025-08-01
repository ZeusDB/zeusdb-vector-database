//! # ZeusDB Vector Database - Persistence Module
//!
//! This module handles all save/load operations for ZeusDB vector indexes.
//! It implements a directory-based persistence format with hybrid JSON/Binary storage.
//!
//! ## File Format:
//! ```
//! my_index.zdb/
//! ├── manifest.json           # Index metadata and file list
//! ├── config.json             # Index configuration
//! ├── mappings.bin            # ID mappings (binary)
//! ├── metadata.json           # Vector metadata (JSON)
//! ├── vectors.bin             # Raw vectors (storage mode dependent)
//! ├── quantization.json       # PQ configuration (if enabled)
//! ├── pq_codes.bin            # Quantized codes (if PQ enabled)
//! ├── pq_centroids.bin        # PQ centroids (if trained)
//! └── hnsw_index.hnsw.graph   # HNSW graph (Phase 2)
//! ```

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
//use chrono::{DateTime, Utc};
use chrono::Utc;
use serde_json::Value;

use crate::hnsw_index::{HNSWIndex, StorageMode};

// ============================================================================
// PERSISTENCE DATA STRUCTURES
// ============================================================================

/// Manifest file structure - tracks index metadata and included files
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexManifest {
    pub format_version: String,
    pub zeusdb_version: String,
    pub created_at: String,
    pub saved_at: String,
    pub total_vectors: usize,
    pub index_type: String,
    pub has_quantization: bool,
    pub quantization_trained: bool,
    pub storage_mode: String,
    pub files_included: Vec<String>,
    pub files_excluded: Vec<String>,
    pub total_size_mb: f64,
    pub compression_info: Option<CompressionInfo>,
}

/// Compression statistics for quantized indexes
#[derive(Debug, Serialize, Deserialize)]
pub struct CompressionInfo {
    pub original_size_mb: f64,
    pub compressed_size_mb: f64,
    pub compression_ratio: f64,
}

/// Index configuration for reconstruction
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexConfig {
    pub dim: usize,
    pub space: String,
    pub m: usize,
    pub ef_construction: usize,
    pub expected_size: usize,
    pub id_counter: usize,
    pub vector_count: usize,
}

/// Complete quantization configuration and state
#[derive(Debug, Serialize, Deserialize)]
pub struct QuantizationPersistence {
    pub r#type: String,
    pub subvectors: usize,
    pub bits: usize,
    pub training_size: usize,
    pub max_training_vectors: Option<usize>,
    pub storage_mode: String,
    pub is_trained: bool,
    pub training_completed_at: Option<String>,
    pub memory_stats: Option<MemoryStats>,
    pub pq_config: PQConfig,
}

/// Memory usage statistics for quantization
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryStats {
    pub centroid_storage_mb: f64,
    pub compression_ratio: f64,
    pub centroids_per_subvector: usize,
    pub total_centroids: usize,
}

/// Product Quantization configuration details
#[derive(Debug, Serialize, Deserialize)]
pub struct PQConfig {
    pub dim: usize,
    pub sub_dim: usize,
    pub num_centroids: usize,
}

/// ID mappings between external and internal IDs
#[derive(Debug, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct IdMappings {
    pub id_map: HashMap<String, usize>,
    pub rev_map: HashMap<usize, String>,
}

// ============================================================================
// MAIN PERSISTENCE INTERFACE
// ============================================================================

/// Save an HNSWIndex to a directory structure
pub fn save_index(index: &HNSWIndex, path: &str) -> PyResult<()> {
    println!("🚀 Starting index save to: {}", path);
    
    // Create the directory structure
    let path_buf = Path::new(path);
    fs::create_dir_all(path_buf).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create directory {}: {}", path, e)
        )
    })?;
    
    // Save components in order of complexity (simple -> complex)
    save_config(index, path_buf)?;
    save_mappings(index, path_buf)?;
    save_metadata(index, path_buf)?;
    
    // Save quantization components if enabled
    if index.has_quantization() {
        save_quantization_config(index, path_buf)?;
        
        if index.can_use_quantization() {
            save_pq_centroids(index, path_buf)?;
            save_pq_codes(index, path_buf)?;
        }
    }
    
    // Save vectors based on storage mode
    save_vectors(index, path_buf)?;
    
    // Save manifest last (references all other files)
    save_manifest(index, path_buf)?;
    
    println!("✅ Index save completed successfully!");
    Ok(())
}

/// Load an HNSWIndex from a directory structure (Phase 2 implementation)
pub fn load_index(_path: &str) -> PyResult<HNSWIndex> {
    // TODO: Implement in Phase 2
    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "Index loading not yet implemented - coming in Phase 2!"
    ))
}

// ============================================================================
// INDIVIDUAL COMPONENT SAVERS
// ============================================================================

/// Save index configuration as JSON
fn save_config(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    println!("⚙️  Saving config.json...");
    
    let config = IndexConfig {
        dim: index.get_dim(),
        space: index.get_space().to_string(),
        m: index.get_m(),
        ef_construction: index.get_ef_construction(),
        expected_size: index.get_expected_size(),
        id_counter: index.get_id_counter(),
        vector_count: index.get_vector_count(),
    };
    
    let config_path = path.join("config.json");
    let config_json = serde_json::to_string_pretty(&config).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to serialize config: {}", e)
        )
    })?;
    
    fs::write(&config_path, config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to write config.json: {}", e)
        )
    })?;
    
    println!("✅ config.json saved");
    Ok(())
}

/// Save ID mappings using efficient binary format
fn save_mappings(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    println!("🗂️  Saving mappings.bin...");
    
    let id_map = index.get_id_map();
    let rev_map = index.get_rev_map();
    
    let mappings = IdMappings {
        id_map: id_map.clone(),
        rev_map: rev_map.clone(),
    };
    
    let mappings_data = bincode::encode_to_vec(&mappings, bincode::config::standard())
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to serialize mappings: {}", e)
            )
        })?;
    
    let mappings_path = path.join("mappings.bin");
    fs::write(&mappings_path, mappings_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to write mappings.bin: {}", e)
        )
    })?;
    
    println!("✅ mappings.bin saved ({} mappings)", id_map.len());
    Ok(())
}

/// Save vector metadata as JSON for external tool compatibility
fn save_metadata(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    println!("📋 Saving metadata.json...");
    
    let vector_metadata = index.get_vector_metadata();
    
    let metadata_path = path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&*vector_metadata).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to serialize metadata: {}", e)
        )
    })?;
    
    fs::write(&metadata_path, metadata_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to write metadata.json: {}", e)
        )
    })?;
    
    println!("✅ metadata.json saved ({} records)", vector_metadata.len());
    Ok(())
}

/// Save quantization configuration and training state
fn save_quantization_config(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    if let Some(config) = index.get_quantization_config() {
        println!("🔧 Saving quantization.json...");
        
        let training_completed_at = if index.can_use_quantization() {
            Some(Utc::now().to_rfc3339()) // TODO: Get actual training completion time
        } else {
            None
        };
        
        let (memory_stats, pq_config) = if let Some(pq) = index.get_pq() {
            let (memory_mb, total_centroids) = pq.get_memory_stats();
            
            let memory_stats = MemoryStats {
                centroid_storage_mb: memory_mb,
                compression_ratio: (pq.dim * 4) as f64 / pq.subvectors as f64,
                centroids_per_subvector: pq.num_centroids,
                total_centroids,
            };
            
            let pq_config = PQConfig {
                dim: pq.dim,
                sub_dim: pq.sub_dim,
                num_centroids: pq.num_centroids,
            };
            
            (Some(memory_stats), pq_config)
        } else {
            let pq_config = PQConfig {
                dim: index.get_dim(),
                sub_dim: index.get_dim() / config.subvectors,
                num_centroids: 1 << config.bits,
            };
            (None, pq_config)
        };
        
        let quant_persistence = QuantizationPersistence {
            r#type: "pq".to_string(),
            subvectors: config.subvectors,
            bits: config.bits,
            training_size: config.training_size,
            max_training_vectors: config.max_training_vectors,
            storage_mode: config.storage_mode.to_string().to_string(),
            is_trained: index.can_use_quantization(),
            training_completed_at,
            memory_stats,
            pq_config,
        };
        
        let quant_path = path.join("quantization.json");
        let quant_json = serde_json::to_string_pretty(&quant_persistence).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to serialize quantization config: {}", e)
            )
        })?;
        
        fs::write(&quant_path, quant_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to write quantization.json: {}", e)
            )
        })?;
        
        println!("✅ quantization.json saved");
    }
    Ok(())
}

/// Save PQ centroids for vector reconstruction
fn save_pq_centroids(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    if let Some(pq) = index.get_pq() {
        if pq.is_trained() {
            println!("🎯 Saving pq_centroids.bin...");
            
            let centroids = pq.centroids.read().unwrap();
            let centroids_data = bincode::encode_to_vec(&*centroids, bincode::config::standard())
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Failed to serialize PQ centroids: {}", e)
                    )
                })?;
            
            let centroids_path = path.join("pq_centroids.bin");
            fs::write(&centroids_path, centroids_data).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to write pq_centroids.bin: {}", e)
                )
            })?;
            
            println!("✅ pq_centroids.bin saved");
        }
    }
    Ok(())
}

/// Save quantized vector codes
fn save_pq_codes(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    let pq_codes = index.get_pq_codes();
    if !pq_codes.is_empty() {
        println!("📦 Saving pq_codes.bin...");
        
        let codes_data = bincode::encode_to_vec(&*pq_codes, bincode::config::standard())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to serialize PQ codes: {}", e)
                )
            })?;
        
        let codes_path = path.join("pq_codes.bin");
        fs::write(&codes_path, codes_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to write pq_codes.bin: {}", e)
            )
        })?;
        
        println!("✅ pq_codes.bin saved ({} vectors)", pq_codes.len());
    }
    Ok(())
}

/// Save raw vectors based on storage mode configuration
fn save_vectors(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    let vectors = index.get_vectors();
    if !vectors.is_empty() {
        println!("📊 Saving vectors.bin...");
        
        let vectors_data = bincode::encode_to_vec(&*vectors, bincode::config::standard())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to serialize vectors: {}", e)
                )
            })?;
        
        let vectors_path = path.join("vectors.bin");
        fs::write(&vectors_path, vectors_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to write vectors.bin: {}", e)
            )
        })?;
        
        println!("✅ vectors.bin saved ({} vectors)", vectors.len());
    }
    Ok(())
}

/// Save manifest file (must be last - references all other files)
fn save_manifest(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    println!("📝 Saving manifest.json...");
    
    let vectors = index.get_vectors();
    let pq_codes = index.get_pq_codes();
    let vector_count = index.get_vector_count();
    
    // Determine what files are included based on what we actually saved
    let mut files_included = vec![
        "config.json".to_string(),
        "mappings.bin".to_string(),
        "metadata.json".to_string(),
    ];
    
    let mut files_excluded = Vec::new();
    
    // Add quantization files if they exist
    if index.has_quantization() {
        files_included.push("quantization.json".to_string());
        
        if index.can_use_quantization() {
            files_included.push("pq_centroids.bin".to_string());
            if !pq_codes.is_empty() {
                files_included.push("pq_codes.bin".to_string());
            }
        }
    }
    
    // Add vectors.bin if it was saved
    if !vectors.is_empty() {
        files_included.push("vectors.bin".to_string());
    } else {
        files_excluded.push("vectors.bin".to_string());
    }
    
    // Calculate compression info for quantized indexes
    let compression_info = if index.has_quantization() && index.can_use_quantization() && !pq_codes.is_empty() {
        let raw_size_mb = (vectors.len() * index.get_dim() * 4) as f64 / (1024.0 * 1024.0);
        let compressed_size_mb = (pq_codes.len() * index.get_quantization_subvectors()) as f64 / (1024.0 * 1024.0);
        let compression_ratio = if compressed_size_mb > 0.0 {
            raw_size_mb / compressed_size_mb
        } else {
            1.0
        };
        
        Some(CompressionInfo {
            original_size_mb: raw_size_mb,
            compressed_size_mb,
            compression_ratio,
        })
    } else {
        None
    };
    
    // Calculate total directory size
    let total_size_mb = calculate_directory_size(path).unwrap_or(0.0);
    
    let manifest = IndexManifest {
        format_version: "1.0.0".to_string(),
        zeusdb_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: index.get_created_at().to_string(),
        saved_at: Utc::now().to_rfc3339(),
        total_vectors: vector_count,
        index_type: "HNSW".to_string(),
        has_quantization: index.has_quantization(),
        quantization_trained: index.can_use_quantization(),
        storage_mode: index.get_storage_mode(),
        files_included,
        files_excluded,
        total_size_mb,
        compression_info,
    };
    
    let manifest_path = path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to serialize manifest: {}", e)
        )
    })?;
    
    fs::write(&manifest_path, manifest_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to write manifest.json: {}", e)
        )
    })?;
    
    println!("✅ manifest.json saved");
    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate the total size of a directory in MB
fn calculate_directory_size(path: &Path) -> Result<f64, std::io::Error> {
    let mut total_size = 0u64;
    
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            
            if metadata.is_file() {
                total_size += metadata.len();
            }
        }
    }
    
    Ok(total_size as f64 / (1024.0 * 1024.0))
}

// ============================================================================
// VALIDATION HELPERS (for Phase 3)
// ============================================================================

/// Check if a path contains a valid ZeusDB index (Phase 3)
pub fn is_valid_index(_path: &str) -> bool {
    // TODO: Implement in Phase 3
    false
}

/// Get index information without full loading (Phase 3)
pub fn get_index_info(_path: &str) -> Option<IndexManifest> {
    // TODO: Implement in Phase 3
    None
}
