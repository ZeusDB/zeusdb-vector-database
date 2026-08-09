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

use crate::hnsw_index::{HNSWIndex, QuantizationConfig, RerankCalibration, StorageMode};
use crate::pq::PQ;
use chrono::Utc;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// FORMAT VERSION
// ============================================================================

/// Version written into manifest.json by this build
///
/// Bumped from 1.0.0 because config.json now carries an index level `metadata`
/// map. The change is additive on both sides. A directory written by this build
/// still opens in 0.4.1, which ignores unknown config fields and never read the
/// version, and a directory written by 0.4.1 still opens here because the field
/// is defaulted.
const FORMAT_VERSION: &str = "1.1.0";

/// Major version this build can interpret
///
/// A minor bump is additive by construction, so any 1.x is read. A different
/// major means the layout changed in a way this build cannot reason about, and
/// guessing at it would be the silent truncation this format has already
/// suffered once.
const SUPPORTED_FORMAT_MAJOR: u32 = 1;

/// Refuse a directory this build cannot interpret
fn check_format_version(format_version: &str) -> PyResult<()> {
    let major = format_version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "manifest.json declares format_version '{}', which is not a version this \
                 build can interpret. A ZeusDB index directory declares a dotted version \
                 such as {}.",
                format_version, FORMAT_VERSION
            ))
        })?;

    if major != SUPPORTED_FORMAT_MAJOR {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Index format version {} cannot be opened by this build, which reads format \
             version {}.x only. The directory was written by a {} release of \
             zeusdb-vector-database, so {}.",
            format_version,
            SUPPORTED_FORMAT_MAJOR,
            if major > SUPPORTED_FORMAT_MAJOR {
                "newer"
            } else {
                "much older"
            },
            if major > SUPPORTED_FORMAT_MAJOR {
                "upgrade the package to open it"
            } else {
                "open it with the release that wrote it"
            }
        )));
    }

    Ok(())
}

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

    /// Index level metadata set through `add_metadata`
    ///
    /// Defaulted rather than required, so a directory written before this field
    /// existed loads with an empty map instead of failing to parse.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
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
    #[serde(default)]
    pub training_ids: Vec<String>,
    #[serde(default)]
    pub training_threshold_reached: bool,
    /// What training measured about the rerank fetch on this index's own data.
    ///
    /// Absent from every directory written before the calibration existed, so
    /// it defaults to `None` and those indexes fall back to the corpus terms
    /// they were built against. See `RerankCalibration`.
    #[serde(default)]
    pub rerank_calibration: Option<RerankCalibration>,
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

/// PQ codebook laid out as [subvector][centroid][dimension within subvector]
type Centroids = Vec<Vec<Vec<f32>>>;

/// Everything the loader reads back for a quantized index
struct QuantizationArtefacts {
    config: QuantizationPersistence,
    centroids: Option<Centroids>,
    codes: HashMap<String, Vec<u8>>,
}

/// Training collection state, held back until after the graph rebuild
///
/// The rebuild re-adds every record through `add(overwrite=true)`, and every id
/// is already in the restored mapping, so each one goes through
/// `remove_point_internal` first. That strips the id from `training_ids`, and
/// re-insertion cannot refill the list because collection is suppressed during
/// a rebuild. Applying the collected ids afterwards is what makes them survive.
struct TrainingState {
    ids: Vec<String>,
    threshold_reached: bool,
    is_trained: bool,
    training_size: usize,
}

impl TrainingState {
    fn from(config: &QuantizationPersistence) -> Self {
        TrainingState {
            ids: config.training_ids.clone(),
            threshold_reached: config.training_threshold_reached,
            is_trained: config.is_trained,
            training_size: config.training_size,
        }
    }

    fn apply(self, index: &mut HNSWIndex) {
        // A trained index cleared its collection when training ran, so this is
        // only ever populated for an index saved while still collecting.
        let collected = self.ids.len();
        index.set_training_ids(self.ids);

        // The saved flag is authoritative for a trained index. For an untrained
        // one it is recomputed, so a directory whose collection was truncated
        // does not come back claiming a threshold it no longer meets.
        let reached = if self.is_trained {
            self.threshold_reached
        } else {
            collected >= self.training_size
        };
        index.set_training_threshold_reached(reached);

        println!(
            "✅ Training state restored ({} collected ids, threshold reached: {})",
            collected, reached
        );
    }
}

// ============================================================================
// INDIVIDUAL COMPONENT LOADERS
// ============================================================================

/// Load index configuration from config.json
fn load_config(path: &Path) -> PyResult<IndexConfig> {
    println!("⚙️  Loading config.json...");

    let config_path = path.join("config.json");
    let config_data = fs::read_to_string(&config_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read config.json: {}",
            e
        ))
    })?;

    let config: IndexConfig = serde_json::from_str(&config_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to parse config.json: {}",
            e
        ))
    })?;

    println!("✅ config.json loaded");
    Ok(config)
}

/// Load ID mappings from mappings.bin
fn load_mappings(path: &Path) -> PyResult<IdMappings> {
    println!("🗂️  Loading mappings.bin...");

    let mappings_path = path.join("mappings.bin");
    let mappings_data = fs::read(&mappings_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read mappings.bin: {}",
            e
        ))
    })?;

    let (mappings, _): (IdMappings, usize) =
        bincode::decode_from_slice(&mappings_data, bincode::config::standard()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize mappings.bin: {}",
                e
            ))
        })?;

    println!("✅ mappings.bin loaded");
    Ok(mappings)
}

/// Load vector metadata from metadata.json
fn load_metadata(path: &Path) -> PyResult<HashMap<String, HashMap<String, Value>>> {
    println!("📋 Loading metadata.json...");

    let metadata_path = path.join("metadata.json");
    let metadata_data = fs::read_to_string(&metadata_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read metadata.json: {}",
            e
        ))
    })?;

    let metadata: HashMap<String, HashMap<String, Value>> = serde_json::from_str(&metadata_data)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to parse metadata.json: {}",
                e
            ))
        })?;

    println!("✅ metadata.json loaded");
    Ok(metadata)
}

/// Load raw vectors from vectors.bin
fn load_vectors(path: &Path) -> PyResult<HashMap<String, Vec<f32>>> {
    println!("📊 Loading vectors.bin...");

    let vectors_path = path.join("vectors.bin");

    // Check if vectors.bin exists (might not exist in quantized_only mode)
    if !vectors_path.exists() {
        println!("ℹ️  vectors.bin not found (quantized_only storage mode)");
        return Ok(HashMap::new());
    }

    let vectors_data = fs::read(&vectors_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read vectors.bin: {}",
            e
        ))
    })?;

    let (vectors, _): (HashMap<String, Vec<f32>>, usize) =
        bincode::decode_from_slice(&vectors_data, bincode::config::standard()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize vectors.bin: {}",
                e
            ))
        })?;

    println!("✅ vectors.bin loaded");
    Ok(vectors)
}

/// Load manifest for validation and metadata
fn load_manifest(path: &Path) -> PyResult<IndexManifest> {
    println!("📝 Loading manifest.json...");

    let manifest_path = path.join("manifest.json");
    let manifest_data = fs::read_to_string(&manifest_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read manifest.json: {}",
            e
        ))
    })?;

    let manifest: IndexManifest = serde_json::from_str(&manifest_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to parse manifest.json: {}",
            e
        ))
    })?;

    println!("✅ manifest.json loaded");
    Ok(manifest)
}

/// Load the PQ codebook from pq_centroids.bin
///
/// Absent means the index was saved before training completed, which is a
/// legitimate state. A present but unreadable file is a hard failure, because
/// the alternative is a codebook that decodes every code to the zero vector.
fn load_pq_centroids(path: &Path) -> PyResult<Option<Centroids>> {
    let centroids_path = path.join("pq_centroids.bin");
    if !centroids_path.exists() {
        return Ok(None);
    }

    println!("🎯 Loading pq_centroids.bin...");

    let centroids_data = fs::read(&centroids_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read pq_centroids.bin: {}",
            e
        ))
    })?;

    let (centroids, _): (Centroids, usize) =
        bincode::decode_from_slice(&centroids_data, bincode::config::standard()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize pq_centroids.bin: {}",
                e
            ))
        })?;

    println!(
        "✅ pq_centroids.bin loaded ({} subvectors)",
        centroids.len()
    );
    Ok(Some(centroids))
}

/// Load the quantized codes from pq_codes.bin
///
/// Absent means no record has been quantized yet. In `quantized_only` these
/// codes are the only copy of every record added after training completed.
fn load_pq_codes(path: &Path) -> PyResult<HashMap<String, Vec<u8>>> {
    let codes_path = path.join("pq_codes.bin");
    if !codes_path.exists() {
        return Ok(HashMap::new());
    }

    println!("📦 Loading pq_codes.bin...");

    let codes_data = fs::read(&codes_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read pq_codes.bin: {}",
            e
        ))
    })?;

    let (codes, _): (HashMap<String, Vec<u8>>, usize) =
        bincode::decode_from_slice(&codes_data, bincode::config::standard()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize pq_codes.bin: {}",
                e
            ))
        })?;

    println!("✅ pq_codes.bin loaded ({} records)", codes.len());
    Ok(codes)
}

/// Load quantization configuration and the codebook that goes with it
fn load_quantization(path: &Path) -> PyResult<Option<QuantizationArtefacts>> {
    println!("🔧 Loading quantization components...");

    let quant_path = path.join("quantization.json");
    if !quant_path.exists() {
        println!("ℹ️  No quantization.json found (non-quantized index)");
        return Ok(None);
    }

    let quant_data = fs::read_to_string(&quant_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
            "Failed to read quantization.json: {}",
            e
        ))
    })?;

    let quant_config: QuantizationPersistence = serde_json::from_str(&quant_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to parse quantization.json: {}",
            e
        ))
    })?;

    println!("✅ quantization.json loaded");

    let centroids = load_pq_centroids(path)?;
    let codes = load_pq_codes(path)?;

    Ok(Some(QuantizationArtefacts {
        config: quant_config,
        centroids,
        codes,
    }))
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
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create directory {}: {}",
            path, e
        ))
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

// ============================================================================
// RECONSTRUCTION FUNCTIONS
// ============================================================================

/// Reconstruct HNSWIndex using Simple Reconstruction
fn reconstruct_index_simple(
    config: IndexConfig,
    mappings: IdMappings,
    metadata: HashMap<String, HashMap<String, Value>>,
    vectors: HashMap<String, Vec<f32>>,
    quantization: Option<QuantizationArtefacts>,
) -> PyResult<HNSWIndex> {
    println!("🔧 Creating empty index with loaded configuration...");

    // Step 1: Create empty index with loaded config
    let mut index = HNSWIndex::new_empty(
        config.dim,
        config.space.clone(),
        config.m,
        config.ef_construction,
        config.expected_size,
    );

    println!("📝 Restoring data fields...");

    // The codes are needed twice, once to rebuild the graph for records that
    // have no raw vector and once to restore the stored codes afterwards.
    let pq_codes = quantization
        .as_ref()
        .map(|q| q.codes.clone())
        .unwrap_or_default();
    let training_state = quantization
        .as_ref()
        .map(|q| TrainingState::from(&q.config));

    // Step 2: Restore all data fields directly (but not the graph)
    restore_data_fields(
        &mut index,
        mappings,
        metadata.clone(),
        vectors.clone(),
        &config,
        quantization,
    )?;

    // Step 3: Rebuild the graph. A trained quantized index rebuilds through
    // the quantized path, inserting the stored codes into a fresh PQ graph, so
    // it comes back quantized and nothing is materialised at full width. An
    // index saved mid-collection or one with no quantization at all replays
    // its raw vectors through add() exactly as before.
    if index.can_use_quantization() {
        println!("🔄 Rebuilding quantized HNSW graph from stored PQ codes...");
        rebuild_graph_from_codes(&mut index, &pq_codes, &vectors)?;
    } else {
        println!("🔄 Rebuilding HNSW graph from vectors...");
        rebuild_graph_from_data(&mut index, &vectors, &pq_codes, &metadata)?;
    }

    // Step 4: Put the stored record data back exactly as it was written. The
    // quantized rebuild never touches the storage maps, and the raw rebuild
    // routes through add(), which stores whatever vector it was given, so
    // without this a reconstructed record would be kept at full width and
    // quantized_only would lose the memory saving that is its whole purpose.
    let raw_count = vectors.len();
    let code_count = pq_codes.len();

    // A trained quantized_only index holds no raw vectors, but a directory
    // written before that was true carries its training records in
    // vectors.bin. They are dropped here rather than restored, so an old
    // directory sheds them on load exactly as a live index sheds them at
    // training. Only a vector whose record also has stored codes is dropped;
    // a raw vector without codes is the record's sole copy, which only a
    // directory that lost pq_codes.bin while keeping vectors.bin can contain,
    // and the count check below is what judges that case. The restored record
    // count is unaffected because every dropped vector's record keeps its
    // codes.
    let quantized_only_trained = index.can_use_quantization()
        && index
            .get_quantization_config()
            .is_some_and(|config| config.storage_mode == StorageMode::QuantizedOnly);
    let vectors = if quantized_only_trained {
        let (kept, dropped): (HashMap<_, _>, HashMap<_, _>) = vectors
            .into_iter()
            .partition(|(id, _)| !pq_codes.contains_key(id));
        if !dropped.is_empty() {
            println!(
                "📉 Released {} raw training vectors quantized_only no longer keeps",
                dropped.len()
            );
        }
        kept
    } else {
        vectors
    };
    index.restore_storage_maps(vectors, pq_codes, metadata);

    // Step 5: Put back the training collection the rebuild stripped
    if let Some(state) = training_state {
        state.apply(&mut index);
    }

    // Step 6: Check the saved count against the index that was actually built
    check_restored_count(&mut index, &config, raw_count, code_count)?;

    println!("✅ Reconstruction completed!");
    Ok(index)
}

/// Reconcile the stored vector count with the records that were restored
///
/// `vector_count` is written to config.json and was previously restored
/// verbatim, so it could report records the directory no longer contains. The
/// count is derived here from the restored data and asserted against the saved
/// value. They agree for every directory whose files are intact, so a
/// disagreement means a file is missing or truncated and the load fails rather
/// than producing an index that misreports what it holds.
fn check_restored_count(
    index: &mut HNSWIndex,
    config: &IndexConfig,
    raw_count: usize,
    code_count: usize,
) -> PyResult<()> {
    let restored = index.count_stored_records();

    if restored != config.vector_count {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Restored record count does not match config.json: the directory yields {} \
             records while config.json reports {}. vectors.bin holds {} records and \
             pq_codes.bin holds {}, so a data file is missing or truncated. Refusing to \
             load an index that would report a count it cannot produce; restore the \
             directory from a copy.",
            restored, config.vector_count, raw_count, code_count
        )));
    }

    index.set_vector_count(restored);
    println!(
        "✅ Vector count verified against restored records: {}",
        restored
    );
    Ok(())
}

/// Restore all data fields to the index (everything except the HNSW graph)
fn restore_data_fields(
    index: &mut HNSWIndex,
    mappings: IdMappings,
    _metadata: HashMap<String, HashMap<String, Value>>,
    _vectors: HashMap<String, Vec<f32>>,
    config: &IndexConfig,
    quantization: Option<QuantizationArtefacts>,
) -> PyResult<()> {
    index.set_id_mappings(mappings.id_map, mappings.rev_map);

    // The add() method will properly:
    // - Insert vectors into index.vectors
    // - Insert metadata into index.vector_metadata
    // - Update counters correctly
    // - Build the HNSW graph

    // Restore counters
    index.set_counters(config.id_counter, config.vector_count);

    // Restore index level metadata. Empty for a directory written before
    // config.json carried the field, which is what those directories held.
    if !config.metadata.is_empty() {
        index.add_metadata(config.metadata.clone());
        println!(
            "✅ Index level metadata restored ({} entries)",
            config.metadata.len()
        );
    }

    // Restore quantization state if present
    if let Some(artefacts) = quantization {
        restore_quantization_state_simple(index, artefacts.config, artefacts.centroids)?;
    }

    println!("✅ All data fields restored successfully");
    Ok(())
}

/// Install a codebook read from disk into a freshly built PQ instance
///
/// The shape check catches a codebook that belongs to a different index. The
/// all-zero check catches the one written by v0.3.0 through v0.4.1, which never
/// read pq_centroids.bin on load and so re-saved the zero codebook that
/// `PQ::new` starts with. Both fail the load rather than let the index come
/// back reporting itself trained while decoding every code to zeros.
fn install_centroids(pq: &PQ, centroids: Centroids) -> PyResult<()> {
    let expected = (pq.subvectors, pq.num_centroids, pq.sub_dim);
    let actual = (
        centroids.len(),
        centroids.first().map(|s| s.len()).unwrap_or(0),
        centroids
            .first()
            .and_then(|s| s.first())
            .map(|c| c.len())
            .unwrap_or(0),
    );
    let uniform = centroids
        .iter()
        .all(|sub| sub.len() == actual.1 && sub.iter().all(|c| c.len() == actual.2));

    if actual != expected || !uniform {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "pq_centroids.bin does not match quantization.json: codebook is {}x{}x{}, \
             expected {}x{}x{} for {} subvectors at {} bits. The codebook belongs to a \
             different index, so this directory cannot be loaded.",
            actual.0,
            actual.1,
            actual.2,
            expected.0,
            expected.1,
            expected.2,
            pq.subvectors,
            pq.bits
        )));
    }

    if centroids
        .iter()
        .all(|sub| sub.iter().all(|c| c.iter().all(|&v| v == 0.0)))
    {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "pq_centroids.bin holds an all-zero codebook, so every PQ code in this \
             directory decodes to the zero vector. This is what a save performed by \
             zeusdb-vector-database 0.3.0 through 0.4.1 writes over a directory it has \
             just loaded, because those versions never read the codebook back. Restore \
             the directory from a copy taken before that save; the records cannot be \
             recovered from this one."
                .to_string(),
        ));
    }

    // Going through set_centroids rather than writing the field rebuilds the
    // symmetric distance table from the codebook that has just been read, so a
    // loaded index can build a graph on real distances exactly as a freshly
    // trained one does.
    pq.set_centroids(centroids)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
}

/// Restore quantization state (simplified for reconstruction)
fn restore_quantization_state_simple(
    index: &mut HNSWIndex,
    quant_data: QuantizationPersistence,
    centroids: Option<Centroids>,
) -> PyResult<()> {
    println!("🔧 Restoring quantization state...");

    // Convert QuantizationPersistence back to QuantizationConfig
    let storage_mode = StorageMode::from_string(&quant_data.storage_mode)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    let quant_config = QuantizationConfig {
        subvectors: quant_data.subvectors,
        bits: quant_data.bits,
        training_size: quant_data.training_size,
        max_training_vectors: quant_data.max_training_vectors,
        storage_mode,
    };

    // Set quantization config
    index.set_quantization_config(Some(quant_config));

    // Restore what training measured about the rerank fetch. `None` here means
    // the directory was written before the calibration existed, and the search
    // falls back to the corpus terms. See `RerankCalibration`.
    index.set_rerank_calibration(quant_data.rerank_calibration);

    // The training ids and the threshold flag are applied after the graph
    // rebuild, which would otherwise strip them. See TrainingState.

    // Every quantized index needs a PQ instance, trained or not. Without one
    // maybe_trigger_training can never fire, so an index saved while still
    // collecting could reach the threshold again and still never train.
    let pq = Arc::new(PQ::new(
        index.get_dim(),
        quant_data.subvectors,
        quant_data.bits,
        quant_data.training_size,
        quant_data.max_training_vectors,
    ));

    if !quant_data.is_trained {
        index.set_pq(Some(pq));

        println!(
            "✅ Quantization state restored (untrained, {} collected training IDs)",
            quant_data.training_ids.len()
        );
    } else {
        // The codebook is what makes a trained PQ trained. Without it the
        // instance would report itself trained while holding the zeros that
        // PQ::new starts with, and every reconstruction would return them.
        let centroids = centroids.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                "quantization.json reports a trained codebook but pq_centroids.bin is \
                 missing from the index directory. The codebook cannot be rebuilt from \
                 the other files, so restore it from a copy of the saved directory."
                    .to_string(),
            )
        })?;
        install_centroids(&pq, centroids)?;

        pq.set_trained(true);
        index.set_pq(Some(pq));

        println!(
            "✅ Quantization state restored (trained, codebook loaded, {} training IDs)",
            quant_data.training_ids.len()
        );
    }

    Ok(())
}

/// Rebuild the graph for a trained quantized index from its stored codes
///
/// The saved graph was a PQ graph over the codes, so the rebuild inserts those
/// same codes into a fresh PQ graph rather than reconstructing vectors and
/// replaying them through the raw add() path. The loaded index therefore
/// reports `is_quantized()` true and `quantized_active`, searches through ADC
/// exactly as the saved one did, and never holds a reconstructed vector at
/// full width. The internal ids come from mappings.bin, so no id is reassigned
/// and the counters stay as saved.
fn rebuild_graph_from_codes(
    index: &mut HNSWIndex,
    pq_codes: &HashMap<String, Vec<u8>>,
    vectors: &HashMap<String, Vec<f32>>,
) -> PyResult<()> {
    let (inserted, quantized_from_raw, remapped) = index
        .rebuild_graph_from_codes(pq_codes, vectors)
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    if quantized_from_raw > 0 {
        println!(
            "⚠️  {} records had a raw vector and no stored PQ codes and were quantized \
             through the loaded codebook",
            quantized_from_raw
        );
    }
    if remapped > 0 {
        println!(
            "⚠️  {} records were missing from mappings.bin and were assigned fresh \
             internal ids",
            remapped
        );
    }
    println!(
        "✅ Quantized graph rebuilt ({} records inserted from stored PQ codes)",
        inserted
    );
    Ok(())
}

/// Rebuild the HNSW graph by re-inserting every record using existing add logic
///
/// This is the path for an index that is not trained, meaning one saved with no
/// quantization at all or one saved while still collecting training vectors.
/// A record that has a raw vector is replayed from it. A record that has only
/// PQ codes is reconstructed through the codebook, which is what `get_records`
/// already does for the same record while the index is live, so the graph is
/// built at the fidelity the storage mode already delivers rather than losing
/// the record. The codes themselves are restored as stored and are never
/// recomputed from a reconstruction.
fn rebuild_graph_from_data(
    index: &mut HNSWIndex,
    vectors: &HashMap<String, Vec<f32>>,
    pq_codes: &HashMap<String, Vec<u8>>,
    metadata: &HashMap<String, HashMap<String, Value>>,
) -> PyResult<()> {
    if vectors.is_empty() && pq_codes.is_empty() {
        println!("ℹ️  No records to rebuild (empty index)");
        return Ok(());
    }

    // Prepare batch data for efficient insertion
    let mut batch_vectors = Vec::new();
    let mut batch_ids = Vec::new();
    let mut batch_metadatas = Vec::new();
    let mut reconstructed = 0usize;
    let mut missing_metadata = 0usize;

    // Every record with a raw vector, replayed from it
    for (ext_id, vector) in vectors.iter() {
        if !metadata.contains_key(ext_id) {
            missing_metadata += 1;
        }
        batch_vectors.push(vector.clone());
        batch_ids.push(ext_id.clone());
        batch_metadatas.push(metadata.get(ext_id).cloned().unwrap_or_default());
    }

    // Every record that has codes and no raw vector, reconstructed
    let code_only: Vec<&String> = pq_codes
        .keys()
        .filter(|id| !vectors.contains_key(*id))
        .collect();

    if !code_only.is_empty() {
        let pq = index.get_pq().cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "{} records are stored as PQ codes with no raw vector, but the index has \
                 no codebook to reconstruct them with. pq_codes.bin and quantization.json \
                 disagree about whether this index was trained, so the directory cannot \
                 be loaded without dropping those records.",
                code_only.len()
            ))
        })?;

        for ext_id in code_only {
            let codes = &pq_codes[ext_id];
            let vector = pq.reconstruct(codes).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to reconstruct record '{}' from its {} PQ codes: {}. The \
                     codebook in pq_centroids.bin does not fit the codes in pq_codes.bin.",
                    ext_id,
                    codes.len(),
                    e
                ))
            })?;

            if !metadata.contains_key(ext_id) {
                missing_metadata += 1;
            }
            batch_vectors.push(vector);
            batch_ids.push(ext_id.clone());
            batch_metadatas.push(metadata.get(ext_id).cloned().unwrap_or_default());
            reconstructed += 1;
        }
    }

    if missing_metadata > 0 {
        println!(
            "⚠️  {} records have no entry in metadata.json and are restored with empty metadata",
            missing_metadata
        );
    }

    println!(
        "📦 Prepared {} records for batch insertion ({} replayed from raw vectors, {} reconstructed from PQ codes)",
        batch_vectors.len(),
        batch_vectors.len() - reconstructed,
        reconstructed
    );

    // SET FLAG: Prevent training ID collection during rebuild
    index
        .rebuilding_from_persistence
        .store(true, std::sync::atomic::Ordering::Release);

    // Use the existing add() method to rebuild the graph
    Python::with_gil(|py| {
        rebuild_using_add_method(index, batch_vectors, batch_ids, batch_metadatas, py)
    })?;

    // 🔥 CLEAR FLAG: Resume normal operation
    index
        .rebuilding_from_persistence
        .store(false, std::sync::atomic::Ordering::Release);

    Ok(())
}

/// Helper function to rebuild using the existing add() method
fn rebuild_using_add_method(
    index: &mut HNSWIndex,
    batch_vectors: Vec<Vec<f32>>,
    batch_ids: Vec<String>,
    batch_metadatas: Vec<HashMap<String, Value>>,
    py: Python<'_>,
) -> PyResult<()> {
    use pyo3::types::{PyDict, PyList};

    // Convert to Python objects
    let vectors_list = PyList::new(py, &batch_vectors)?;
    let ids_list = PyList::new(py, &batch_ids)?;

    // Convert metadata to Python objects
    let metadatas_vec: PyResult<Vec<_>> = batch_metadatas
        .iter()
        .map(|m| {
            let dict = PyDict::new(py);
            for (k, v) in m {
                dict.set_item(k, convert_json_value_to_python(v, py)?)?;
            }
            Ok(dict)
        })
        .collect();
    let metadatas_list = PyList::new(py, &metadatas_vec?)?;

    // Create batch dictionary
    let batch_dict = PyDict::new(py);
    batch_dict.set_item("vectors", vectors_list)?;
    batch_dict.set_item("ids", ids_list)?;
    batch_dict.set_item("metadatas", metadatas_list)?;

    println!("🔄 Calling add() method to rebuild graph...");

    // Call the existing add method - this rebuilds the graph automatically
    let result = index.add(batch_dict.into_any(), true)?; // overwrite=true

    // add() reports per record and does not raise, so a record it refused
    // would otherwise leave the graph short while the storage maps, which are
    // written back afterwards, still report the full count. The load would
    // succeed and every query would miss the records that never reached the
    // graph. A saved directory holding a non-finite value is the case that
    // reaches here, because add() has always refused those on the list path.
    if result.total_errors > 0 {
        let detail = result.errors.join("; ");
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Graph rebuild refused {} of {} records, so the loaded index would \
             hold records that no query can reach. Refusing to load a partial \
             graph. Rejected records: {}",
            result.total_errors,
            batch_ids.len(),
            detail
        )));
    }

    println!("✅ Graph rebuild completed: {}", result.summary());

    // Verify the rebuild
    let final_vector_count = index.get_vector_count();
    println!("📊 Final vector count: {}", final_vector_count);

    Ok(())
}

/// Convert JSON Value to Python object (same as before)
fn convert_json_value_to_python(value: &Value, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => {
            let bound = b.into_pyobject(py)?;
            //Ok(bound.unbind().into())
            Ok(bound.to_owned().into())
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.unbind().into())
            } else {
                Ok(n.to_string().into_pyobject(py)?.unbind().into())
            }
        }
        Value::String(s) => Ok(s.clone().into_pyobject(py)?.unbind().into()),
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(convert_json_value_to_python(item, py)?)?;
            }
            Ok(py_list.into_pyobject(py)?.unbind().into())
        }
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (k, v) in obj {
                py_dict.set_item(k, convert_json_value_to_python(v, py)?)?;
            }
            Ok(py_dict.into_pyobject(py)?.unbind().into())
        }
    }
}

// ============================================================================
// LOAD INTERFACE
// ============================================================================

/// Load an HNSWIndex from a directory structure (Approach B: Simple Reconstruction)
///
/// Registered as `_load_index`. `VectorDatabase.load(path)` is the documented
/// route and is a one line pass through to this.
#[pyfunction]
#[pyo3(name = "_load_index")]
pub fn load_index(path: &str) -> PyResult<HNSWIndex> {
    println!("🚀 Starting index load with reconstruction from: {}", path);

    let path_buf = Path::new(path);

    // Validate directory exists
    if !path_buf.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("Index directory not found: {}", path),
        ));
    }

    // Phase 1: Load all ZeusDB components
    println!("📋 Phase 1: Loading ZeusDB components...");

    let manifest = load_manifest(path_buf)?;
    check_format_version(&manifest.format_version)?;
    println!(
        "✅ Manifest loaded: {} vectors, format v{}",
        manifest.total_vectors, manifest.format_version
    );

    let config = load_config(path_buf)?;
    println!(
        "✅ Config loaded: dim={}, space={}",
        config.dim, config.space
    );

    let mappings = load_mappings(path_buf)?;
    println!("✅ Mappings loaded: {} ID mappings", mappings.id_map.len());

    let metadata = load_metadata(path_buf)?;
    println!("✅ Metadata loaded: {} records", metadata.len());

    let vectors = load_vectors(path_buf)?;
    println!("✅ Vectors loaded: {} vectors", vectors.len());

    let quantization = load_quantization(path_buf)?;
    if let Some(ref quant) = quantization {
        println!(
            "✅ Quantization loaded: {} subvectors, trained={}, codebook={}",
            quant.config.subvectors,
            quant.config.is_trained,
            if quant.centroids.is_some() {
                "present"
            } else {
                "absent"
            }
        );
    }

    // Skip HNSW graph loading - we'll rebuild it

    // Phase 2: Create empty index and restore state
    println!("🔧 Phase 2: Creating empty index and restoring state...");
    let restored_index =
        reconstruct_index_simple(config, mappings, metadata, vectors, quantization)?;

    println!("✅ Index reconstruction with graph rebuild completed successfully!");
    Ok(restored_index)
}

// ============================================================================
// INDIVIDUAL COMPONENT SAVERS
// ============================================================================

/// Save index configuration as JSON
fn save_config(index: &HNSWIndex, path: &Path) -> PyResult<()> {
    println!("⚙️  Saving config.json...");

    let config = IndexConfig {
        dim: index.get_dim(),
        //space: index.get_space().to_string(),
        space: index.space().to_string(), // Changed from get_space()
        m: index.get_m(),
        ef_construction: index.get_ef_construction(),
        expected_size: index.get_expected_size(),
        id_counter: index.get_id_counter(),
        vector_count: index.get_vector_count(),
        metadata: index.get_all_metadata(),
    };

    let config_path = path.join("config.json");
    let config_json = serde_json::to_string_pretty(&config).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to serialize config: {}",
            e
        ))
    })?;

    fs::write(&config_path, config_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to write config.json: {}",
            e
        ))
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

    let mappings_data =
        bincode::encode_to_vec(&mappings, bincode::config::standard()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to serialize mappings: {}",
                e
            ))
        })?;

    let mappings_path = path.join("mappings.bin");
    fs::write(&mappings_path, mappings_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to write mappings.bin: {}",
            e
        ))
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
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to serialize metadata: {}",
            e
        ))
    })?;

    fs::write(&metadata_path, metadata_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to write metadata.json: {}",
            e
        ))
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

        // CAPTURE TRAINING STATE:
        let training_ids = index.get_training_ids().clone();
        let training_threshold_reached = index.get_training_threshold_reached();

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
            training_ids,
            training_threshold_reached,
            rerank_calibration: index.get_rerank_calibration(),
        };

        let quant_path = path.join("quantization.json");
        let quant_json = serde_json::to_string_pretty(&quant_persistence).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to serialize quantization config: {}",
                e
            ))
        })?;

        fs::write(&quant_path, quant_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to write quantization.json: {}",
                e
            ))
        })?;

        //println!("✅ quantization.json saved");
        println!(
            "✅ quantization.json saved with {} training IDs",
            quant_persistence.training_ids.len()
        );
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
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize PQ centroids: {}",
                    e
                ))
            })?;

            let centroids_path = path.join("pq_centroids.bin");
            fs::write(&centroids_path, centroids_data).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to write pq_centroids.bin: {}",
                    e
                ))
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

        let codes_data =
            bincode::encode_to_vec(&*pq_codes, bincode::config::standard()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize PQ codes: {}",
                    e
                ))
            })?;

        let codes_path = path.join("pq_codes.bin");
        fs::write(&codes_path, codes_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to write pq_codes.bin: {}",
                e
            ))
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

        let vectors_data =
            bincode::encode_to_vec(&*vectors, bincode::config::standard()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize vectors: {}",
                    e
                ))
            })?;

        let vectors_path = path.join("vectors.bin");
        fs::write(&vectors_path, vectors_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to write vectors.bin: {}",
                e
            ))
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

    // Phase 2: Add HNSW graph files
    // REPLACE WITH THIS CONDITIONAL LOGIC:
    let vector_count = index.get_vector_count();
    if vector_count > 0 {
        files_included.push("hnsw_index.hnsw.graph".to_string());
        files_excluded.push("hnsw_index.hnsw.data".to_string());
        println!("📋 Graph files in manifest:");
        println!("   ✅ Included: hnsw_index.hnsw.graph");
        println!("   ❌ Excluded: hnsw_index.hnsw.data (we use our own data files)");
    } else {
        files_excluded.push("hnsw_index.hnsw.graph".to_string());
        files_excluded.push("hnsw_index.hnsw.data".to_string());
        println!("ℹ️  No graph files (empty index)");
    }

    // Calculate compression info for quantized indexes
    //
    // Both sizes are taken over the coded records, so the ratio is the size of
    // a code against the size of the vector it stands for. `original_size_mb`
    // used to count the raw vectors the index still holds, which under
    // quantized_only is only the training records. That put a record count in
    // the numerator and a different one in the denominator, and the ratio came
    // out as the compression ratio scaled by the share of records collected
    // before training. At 1,000 training records in 3,000 it read 10.7x where
    // the codes are 32x smaller than the vectors. Under quantized_with_raw the
    // two counts were already equal, so this changes nothing there.
    let compression_info =
        if index.has_quantization() && index.can_use_quantization() && !pq_codes.is_empty() {
            let raw_size_mb = (pq_codes.len() * index.get_dim() * 4) as f64 / (1024.0 * 1024.0);
            let compressed_size_mb =
                (pq_codes.len() * index.get_quantization_subvectors()) as f64 / (1024.0 * 1024.0);
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
        format_version: FORMAT_VERSION.to_string(),
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
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to serialize manifest: {}",
            e
        ))
    })?;

    fs::write(&manifest_path, manifest_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to write manifest.json: {}",
            e
        ))
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
///
/// Reserved surface. The body is a placeholder that reports every path invalid
/// and must be implemented before any caller is wired up, including the module
/// registration in lib.rs. The allow keeps the reservation visible instead of
/// silencing dead code across the module.
#[allow(dead_code)]
pub fn is_valid_index(_path: &str) -> bool {
    // TODO: Implement in Phase 3
    false
}

/// Get index information without full loading (Phase 3)
///
/// Reserved surface. The body is a placeholder that reports no manifest for
/// every path and must be implemented before any caller is wired up.
#[allow(dead_code)]
pub fn get_index_info(_path: &str) -> Option<IndexManifest> {
    // TODO: Implement in Phase 3
    None
}
