//! The face the persistence layer speaks to.
//!
//! `persistence.rs` reads and writes the directory; this file is the only way it
//! reaches the index. Every private field it needs is behind an accessor here
//! and every field it restores is behind a setter, so the storage layer names no
//! field of `HNSWIndex` and cannot leave one in a state the index did not
//! choose.

use super::locks::{order, ReadGuard};
use super::{HNSWIndex, QuantizationConfig, StorageMode, MAX_LAYER};
use crate::graph::{restore_graph, DumpBounds, VectorGraph};
use crate::rerank::RawVectors;
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, instrument};
/// Set to any non-empty value other than `0` to skip the saved graph and
/// rebuild it by re-inserting every record.
///
/// The rebuild is what upgrades an index whose graph was built by a release
/// carrying a defect the vendored patches have since fixed, since restoring the
/// dump restores the graph exactly as it was written, defects included. Without
/// this there is no way to ask for that upgrade on a directory whose dump is
/// perfectly readable.
const REBUILD_ENV: &str = "ZEUSDB_LOAD_REBUILD_GRAPH";
/// Whether the caller has asked for the rebuild rather than the saved graph
fn rebuild_requested() -> bool {
    match std::env::var(REBUILD_ENV) {
        Ok(value) => !value.is_empty() && value != "0",
        Err(_) => false,
    }
}
impl HNSWIndex {
    /// Count the records the index actually holds
    ///
    /// The union of the raw vectors and the PQ codes, because `quantized_only`
    /// keeps a record added after training in the codes alone. This is derived
    /// from the stored data rather than from the counter, so it is what the
    /// counter is checked against after a load. Not exposed to Python, since
    /// the only caller is the load path in `persistence`.
    pub fn count_stored_records(&self) -> usize {
        let id_map = self.id_map.read().unwrap();
        let hnsw = self.hnsw.read().unwrap();
        let pq_codes = self.pq_codes.read().unwrap();
        let raws = RawVectors {
            id_map: &id_map,
            graph: &hnsw,
        };
        let with_raw = id_map
            .keys()
            .filter(|id| raws.contains(id.as_str()))
            .count();
        let code_only = pq_codes
            .keys()
            .filter(|id| !raws.contains(id.as_str()))
            .count();
        with_raw + code_only
    }

    /// Every raw vector the index holds, keyed by external id.
    ///
    /// Built rather than held, because there is no map of raw vectors any more.
    /// The one caller is the save of a quantized index that keeps its raws:
    /// a raw index writes no `vectors.bin` at all, since the graph dump already
    /// carries its store.
    pub(crate) fn collect_raw_vectors(&self) -> HashMap<String, Vec<f32>> {
        let id_map = self.id_map.read().unwrap();
        let hnsw = self.hnsw.read().unwrap();
        let mut out = HashMap::with_capacity(id_map.len());
        for (ext_id, &internal_id) in id_map.iter() {
            if let Some(vector) = hnsw.raw_vector(internal_id) {
                out.insert(ext_id.clone(), vector.to_vec());
            }
        }
        out
    }

    /// Whether this index should hold a raw store beside its codes.
    ///
    /// Asked by the loader after the graph is back and before the vectors are
    /// placed, which is the one moment the graph is quantized and the store is
    /// still empty, so `raws_need_their_own_file` would answer no.
    pub(crate) fn raw_store_is_expected(&self) -> bool {
        let quantized = self.hnsw.read().unwrap().is_quantized();
        quantized
            && self
                .quantization_config
                .as_ref()
                .is_some_and(|config| config.storage_mode == StorageMode::QuantizedWithRaw)
    }

    /// Whether the index holds a raw vector for every record it holds.
    ///
    /// True on every raw graph and on a `quantized_with_raw` one, false on a
    /// trained `quantized_only` one. It is what decides whether a save writes
    /// `vectors.bin`.
    ///
    /// **The saved directory still holds two copies of every raw vector on a
    /// raw index**, one in `vectors.bin` and one in the graph dump's vector
    /// region. Removing the map did not remove that, and the reason is the
    /// rebuild fallback: `vectors.bin` is keyed by external id and a damaged
    /// graph dump is exactly the case where the node ordering that addresses
    /// the store is unavailable. Dropping the file would make a raw index with
    /// a damaged dump unloadable, which is a recovery path the loader
    /// documents and thirty-three tests hold.
    pub(crate) fn holds_raw_vectors(&self) -> bool {
        self.hnsw.read().unwrap().holds_raw()
    }

    /// Rebuild the raw side store of a loaded quantized graph.
    ///
    /// The graph comes back from its dump with its own node numbering, which is
    /// the dump's layer major order rather than the order the records arrived
    /// in, so the vectors read from `vectors.bin` are placed by internal id
    /// rather than by position. A live record the file does not name is an
    /// error, because a `quantized_with_raw` index whose raws are partly
    /// missing would rerank some records and not others.
    ///
    /// **A stranded node takes a zero vector.** Removal and overwrite leave the
    /// node in the graph and take the record out of `id_map`, and the save
    /// writes the live records alone, so the file names fewer records than the
    /// graph has nodes. The store is addressed by node, so those nodes need a
    /// slot; nothing can reach it, because every reader resolves an external id
    /// through `id_map` first and a stranded node has none. `compact` drops the
    /// node and its slot together.
    pub(crate) fn restore_raw_store(
        &mut self,
        vectors: &HashMap<String, Vec<f32>>,
    ) -> Result<usize, String> {
        let id_map = self.id_map.read().unwrap();
        let mut hnsw = self.hnsw.write().unwrap();
        if !hnsw.is_quantized() {
            return Ok(0);
        }
        let nodes = hnsw.nb_points();
        hnsw.open_raw_store(self.dim, nodes)?;
        let mut by_internal: HashMap<usize, &Vec<f32>> = HashMap::with_capacity(vectors.len());
        for (ext_id, vector) in vectors {
            if let Some(&internal_id) = id_map.get(ext_id) {
                by_internal.insert(internal_id, vector);
            }
        }
        let live: std::collections::HashSet<usize> = id_map.values().copied().collect();
        let stranded_fill = vec![0f32; self.dim];
        let mut placed = 0usize;
        let mut stranded = 0usize;
        for node in 0..nodes as u32 {
            let internal_id = hnsw.origin_id_at(node);
            match by_internal.get(&internal_id) {
                Some(vector) => {
                    hnsw.push_raw_vector(vector)?;
                    placed += 1;
                }
                None if !live.contains(&internal_id) => {
                    hnsw.push_raw_vector(&stranded_fill)?;
                    stranded += 1;
                }
                None => {
                    return Err(format!(
                        "live record with internal id {} has no raw vector in vectors.bin",
                        internal_id
                    ))
                }
            }
        }
        if stranded > 0 {
            debug!(
                operation = "restore_raw_store",
                stranded_nodes = stranded,
                "Stranded graph nodes took a zero vector no reader can reach"
            );
        }
        Ok(placed)
    }

    /// The body of `save`, with the interpreter lock already released
    pub(super) fn save_locked(&self, path: &str) -> PyResult<()> {
        // A save reads the mappings, the metadata, the codes, the vectors and
        // the graph in five separate passes, so it needs the index to hold
        // still. PyO3's exclusive borrow used to guarantee that by keeping every
        // mutating method away from it. Relaxing the receivers removes that, and
        // a save overlapping an add would write a directory whose mappings and
        // vectors came from different instants. This takes the mutation lock
        // instead, which blocks a concurrent write and no reader at all.
        let _writers = self.writers.lock().unwrap();
        let start_time = Instant::now();
        info!(operation = "save_start", path = path, "Starting index save");

        let target = Path::new(path);

        // Every artefact goes into a directory of its own that is moved into
        // place at the end, so a reader sees the previous index or this one and
        // never a mixture of the two, and a stale artefact from an earlier save
        // cannot survive. See `persistence::StagingDir`.
        let staging = crate::persistence::StagingDir::open(target)?;

        // Phase 1: Save all ZeusDB components (already tested to work)
        debug!(operation = "save_phase1", "Saving ZeusDB components");
        let mut ledger = crate::persistence::save_index(self, staging.path())?;

        // Phase 2: Save the HNSW graph in ZeusDB's own format
        debug!(operation = "save_phase2", "Saving HNSW graph");
        self.save_hnsw_graph(staging.path(), &mut ledger)?;

        // Phase 3: The manifest, last, because it records a length and a digest
        // for every artefact above and the directory size all of them add up
        // to.
        debug!(operation = "save_phase3", "Writing the manifest");
        crate::persistence::save_manifest(self, staging.path(), ledger)?;

        // Phase 4: Two renames, or one where nothing is there yet.
        debug!(
            operation = "save_phase4",
            "Moving the saved index into place"
        );
        staging.commit()?;

        let duration_ms = start_time.elapsed().as_millis();
        info!(
            operation = "save_complete",
            path = path,
            duration_ms = duration_ms,
            "Index save completed successfully"
        );
        Ok(())
    }

    /// Write the HNSW graph in ZeusDB's own format. See `graph::dump`.
    #[instrument(level = "info", skip(self, ledger), fields(
        vector_count = self.get_vector_count(),
        path = %path.display()
    ))]
    fn save_hnsw_graph(
        &self,
        path: &Path,
        ledger: &mut crate::persistence::SaveLedger,
    ) -> PyResult<()> {
        debug!(
            operation = "save_hnsw_graph_start",
            "Starting HNSW graph save"
        );

        // EMPTY INDEX CHECK:
        let vector_count = self.get_vector_count();
        if vector_count == 0 {
            debug!(
                operation = "save_hnsw_graph",
                reason = "empty_index",
                "Skipping HNSW graph dump - index is empty"
            );
            return Ok(());
        }

        let hnsw_guard = self.hnsw.read().unwrap();

        let dump_result = hnsw_guard.dump(path);

        match dump_result {
            Ok(filename) => {
                // Its length rather than a digest, for the reason
                // `persistence::ArtefactDigest` gives: the dump streams itself
                // and seeks back to write its header, so hashing it whole would
                // mean reading the largest artefact in the directory back off
                // the disk for a guarantee its own two checksums already give.
                let bytes = std::fs::metadata(path.join(&filename))
                    .map(|meta| meta.len())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "The graph dump was written and its length could not be read: {}",
                            e
                        ))
                    })?;
                ledger.record_length(&filename, bytes);
                debug!(
                    operation = "save_hnsw_graph_complete",
                    file_created = %filename,
                    file_bytes = bytes,
                    "HNSW graph saved successfully"
                );
                Ok(())
            }
            Err(e) => {
                error!(operation = "save_hnsw_graph", error = %e, "HNSW graph dump failed");
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "HNSW graph dump failed: {}",
                    e
                )))
            }
        }
    }

    // 6. PERSISTENCE INTEGRATION METHODS (2 methods)

    /// Load an index from a .zdb directory structure (Phase 2)
    pub fn load(path: &str) -> PyResult<Self> {
        crate::persistence::load_index(path)
    }

    /// Restore the graph the save wrote, instead of rebuilding it
    ///
    /// For persistence loading only, and only after `restore_data_fields` has
    /// installed the id mappings and the product quantizer, because both decide
    /// what the dump is checked against.
    ///
    /// Returns the number of graph nodes restored, or the reason the dump
    /// cannot be used. Every reason is a fallback rather than a failure: the
    /// caller rebuilds instead, so a directory whose dump is absent, was written
    /// by a release whose distance types were named differently, or is damaged
    /// still loads.
    pub(crate) fn restore_graph_from_dump(
        &mut self,
        dir: &Path,
        max_origin_id: usize,
        recorded_bytes: Option<u64>,
    ) -> Result<usize, String> {
        if rebuild_requested() {
            return Err(format!("{} asked for the rebuild", REBUILD_ENV));
        }

        // The dump's own header and payload checksums say the file is
        // internally consistent. The length manifest.json recorded says it is
        // the file this directory's save wrote, which a self-consistent dump
        // from a different save of an index of the same shape would otherwise
        // pass. A directory written before the manifest carried lengths records
        // none, and nothing is checked.
        if let Some(expected) = recorded_bytes {
            let found = std::fs::metadata(dir.join(crate::graph::dump::DUMP_FILENAME))
                .map(|meta| meta.len())
                .unwrap_or(0);
            if found != expected {
                return Err(format!(
                    "the graph dump is {} bytes and manifest.json records it as {}",
                    found, expected
                ));
            }
        }

        // A save skips the graph dump entirely when the index holds nothing, so
        // any dump left in an empty index's directory belongs to an earlier
        // save and describes records this one no longer holds.
        let live = self.id_map.read().unwrap().len();
        if live == 0 {
            return Err("the index holds no records".to_string());
        }

        // A trained product quantizer is what makes the saved graph a quantized
        // one, so it is what decides which element type the dump must carry.
        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => Some(pq.clone()),
            _ => None,
        };

        let (graph, nodes) = restore_graph(
            dir,
            &self.space,
            self.get_m(),
            self.get_ef_construction(),
            self.dim,
            pq,
            DumpBounds {
                min_nodes: live,
                max_origin_id,
            },
        )?;

        self.replace_graph(graph);
        Ok(nodes)
    }

    /// Rebuild the graph from the stored PQ codes (for persistence loading only)
    ///
    /// Requires a trained product quantizer, which the loader installs before
    /// any rebuild runs. Replaces the raw graph `new_empty` built with a fresh
    /// PQ graph and inserts every record's codes under the internal id restored
    /// from mappings.bin, so the loaded index is quantized exactly as the saved
    /// one was and no vector is reconstructed to full width on the way.
    ///
    /// A record that has a raw vector but no stored codes is quantized through
    /// the loaded codebook rather than dropped. An intact directory saved by a
    /// trained index holds codes for every record, so that path only runs on a
    /// directory that lost pq_codes.bin while keeping its raw vectors. A record
    /// missing from mappings.bin is assigned a fresh internal id for the same
    /// reason: every record must come back.
    ///
    /// Returns (records inserted, quantized from raw, remapped).
    pub(crate) fn rebuild_graph_from_codes(
        &mut self,
        pq_codes: &HashMap<String, Vec<u8>>,
        vectors: &HashMap<String, Vec<f32>>,
    ) -> Result<(usize, usize, usize), String> {
        let pq = match &self.pq {
            Some(pq) if pq.is_trained() => pq.clone(),
            _ => {
                return Err(
                    "the quantized graph rebuild requires a trained product quantizer".to_string(),
                )
            }
        };

        let mut extra: Vec<(String, Vec<u8>)> = Vec::new();
        for (id, vector) in vectors {
            if !pq_codes.contains_key(id) {
                let codes = pq.quantize(vector).map_err(|e| {
                    format!(
                        "record '{}' has a raw vector but no stored PQ codes, and quantizing \
                         it through the loaded codebook failed: {}",
                        id, e
                    )
                })?;
                extra.push((id.clone(), codes));
            }
        }
        // Sorted so the internal ids the missing records are about to be handed
        // are handed in a fixed order rather than in hash map order.
        extra.sort_by(|a, b| a.0.cmp(&b.0));

        let mut new_hnsw = VectorGraph::new_pq(
            &self.space,
            self.get_m(),
            self.get_expected_size(),
            MAX_LAYER,
            self.get_ef_construction(),
            pq,
        );

        let mut batch: Vec<(&Vec<u8>, usize)> = Vec::with_capacity(pq_codes.len() + extra.len());
        let mut lost: Vec<(&String, &Vec<u8>)> = Vec::new();
        {
            let id_map = self.id_map.read().unwrap();
            for (id, codes) in pq_codes
                .iter()
                .chain(extra.iter().map(|(id, codes)| (id, codes)))
            {
                match id_map.get(id) {
                    Some(&internal_id) => batch.push((codes, internal_id)),
                    None => lost.push((id, codes)),
                }
            }
        }
        let remapped = lost.len();
        lost.sort_by(|a, b| a.0.cmp(b.0));
        for (id, codes) in lost {
            let internal_id = self.get_next_id();
            self.id_map.write().unwrap().insert(id.clone(), internal_id);
            self.rev_map
                .write()
                .unwrap()
                .insert(internal_id, id.clone());
            batch.push((codes, internal_id));
        }

        // Insert in internal id order, which is arrival order, rather than in
        // the order a hash map hands the codes out. Two rebuilds of one
        // directory otherwise wire the graph differently in each process.
        batch.sort_by_key(|&(_, internal_id)| internal_id);

        // Filled before it is installed, and installed under one write guard, so
        // the graph the index holds is never a partly rebuilt one.
        if !batch.is_empty() {
            new_hnsw.insert_batch_pq(&batch)?;
        }
        self.replace_graph(new_hnsw);

        Ok((batch.len(), extra.len(), remapped))
    }

    /// Set ID mappings (for persistence loading only)
    pub(crate) fn set_id_mappings(
        &mut self,
        id_map: HashMap<String, usize>,
        rev_map: HashMap<usize, String>,
    ) {
        *self.id_map.write().unwrap() = id_map;
        *self.rev_map.write().unwrap() = rev_map;
    }

    /// Set counters (for persistence loading only)
    pub(crate) fn set_counters(&mut self, id_counter: usize, vector_count: usize) {
        *self.id_counter.lock().unwrap() = id_counter;
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Set the vector count alone (for persistence loading only)
    ///
    /// Separate from `set_counters` because the id counter must keep whatever
    /// the graph rebuild advanced it to. Rewinding it would hand out internal
    /// ids the rebuild has already used.
    pub(crate) fn set_vector_count(&mut self, vector_count: usize) {
        *self.vector_count.lock().unwrap() = vector_count;
    }

    /// Replace the stored record data with what was read from disk
    ///
    /// For persistence loading only, and only after the graph rebuild. The
    /// rebuild routes every record through add(), which stores whatever vector
    /// it was handed, so a record that was reconstructed from PQ codes would
    /// otherwise be kept at full width. Writing the three maps back leaves the
    /// loaded index holding exactly what was saved.
    pub(crate) fn restore_storage_maps(
        &mut self,
        pq_codes: HashMap<String, Vec<u8>>,
        vector_metadata: HashMap<String, HashMap<String, Value>>,
    ) {
        *self.pq_codes.write().unwrap() = pq_codes;
        *self.vector_metadata.write().unwrap() = vector_metadata;
        self.rebuild_columns();
    }

    /// Build every declared column from the restored metadata.
    ///
    /// **Nothing on disk holds a column.** The declaration comes back from
    /// `config.json` and the values come back from `metadata.json`, so the
    /// columns are derived here in one pass over the records rather than read.
    /// That is what keeps the directory format unchanged, keeps a directory
    /// written by this build readable by an older one, and means a directory
    /// written before the declaration existed simply loads with no columns.
    ///
    /// Runs after the graph is restored and after `restore_storage_maps` has
    /// written the metadata back, so `id_map` holds the internal id every
    /// column entry is addressed by. It is a no-op on an index that declared
    /// nothing, which is every directory saved before this existed.
    /// It clears before it writes. The graph rebuild path routes every record
    /// through `add`, which writes a column entry under whatever internal id it
    /// allocated, and those are not the ids the restored mappings carry. Wiping
    /// first is what makes this the one authority on what the columns hold
    /// rather than a second writer over the first one's leftovers.
    ///
    /// The guards are taken in the declared order, `id_map` then
    /// `vector_metadata` then `columns`, even though the loader holds `&mut
    /// self` and nothing else can be running.
    fn rebuild_columns(&mut self) {
        let id_map = self.id_map.read().unwrap();
        let vector_metadata = self.vector_metadata.read().unwrap();
        let mut columns = self.columns.write().unwrap();
        if !columns.is_declared() {
            return;
        }
        columns.clear(self.get_expected_size());
        let empty = HashMap::new();
        for (ext_id, &internal_id) in id_map.iter() {
            columns.write(internal_id, vector_metadata.get(ext_id).unwrap_or(&empty));
        }
        debug_assert!(
            columns.tracks(id_map.len()),
            "the rebuild writes one column entry per record in id_map"
        );
    }

    /// Set quantization config (for persistence loading only)
    pub(crate) fn set_quantization_config(&mut self, config: Option<QuantizationConfig>) {
        self.quantization_config = config;
    }

    /// Set PQ instance (for persistence loading only)
    pub(crate) fn set_pq(&mut self, pq: Option<Arc<crate::pq::PQ>>) {
        self.pq = pq;
    }

    /// Set training threshold reached flag (for persistence loading only)
    pub(crate) fn set_training_threshold_reached(&mut self, value: bool) {
        self.training_threshold_reached
            .store(value, std::sync::atomic::Ordering::Release);
    }

    /// Suppress or resume training id collection (for persistence loading only)
    ///
    /// Wraps the flag the graph rebuild sets while it replays every record.
    /// Every id being replayed is already in the restored collection, so
    /// collecting them again would double the list. Private to this file now
    /// that `rebuild_from_records` is the only thing that sets it, and it is
    /// set and cleared in one place rather than around a call in the storage
    /// layer.
    fn set_rebuilding_from_persistence(&self, value: bool) {
        self.rebuilding_from_persistence
            .store(value, std::sync::atomic::Ordering::Release);
    }

    /// Replay records read off disk into the graph.
    ///
    /// This is the load path's fallback, taken where the saved graph dump is
    /// absent, damaged, or written by a release this build cannot read. The
    /// records arrive as owned Rust, one per record the directory holds, in the
    /// order the caller sorted them into.
    ///
    /// **It used to go through `add`.** The loader built a `PyDict` holding a
    /// `PyList` of vectors, a `PyList` of ids and a `PyList` of per-record
    /// metadata dicts, called `add`, and `add` parsed the whole thing straight
    /// back into the `Vec<(String, Vec<f32>, HashMap<String, Value>)>` the
    /// loader already had. Every record made a round trip through the
    /// interpreter for nothing, and the interpreter lock was held for the whole
    /// rebuild, which is the graph build itself and not a short window. Every
    /// other Python thread in the process was stopped for the duration.
    ///
    /// What that round trip did do, and what is reproduced here exactly, is
    /// `extract_single_vector`'s validation and its call to
    /// `process_vector_for_space`. A stored vector is already normalized for a
    /// cosine index, and normalizing it a second time is what `add` did, so it
    /// is what this does. **The graph a rebuild produces is unchanged, bit for
    /// bit, and that is the point.** Skipping the second normalization would be
    /// defensible and would build a different graph, which on this path means
    /// giving a user's index different answers.
    ///
    /// `overwrite` is true because the loader restores the id mappings before
    /// the rebuild runs, so every record being replayed is already named in
    /// `id_map` and has to be removed before it is inserted.
    pub(crate) fn rebuild_from_records(
        &self,
        records: Vec<(String, Vec<f32>, HashMap<String, Value>)>,
    ) -> PyResult<usize> {
        let total = records.len();

        // The validation `extract_single_vector` ran on the way through Python.
        // It fails the whole rebuild rather than dropping the record, which is
        // what the round trip did: a refused record left the graph short while
        // the storage maps, written back afterwards, still reported the full
        // count, so every query missed it. A saved directory holding a
        // non-finite value is the case that reaches here.
        let mut validated = Vec::with_capacity(total);
        for (id, vector, metadata) in records {
            if vector.len() != self.dim {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Graph rebuild refused record '{}': vector dimension mismatch, \
                     expected {}, got {}. Refusing to load a partial graph.",
                    id,
                    self.dim,
                    vector.len()
                )));
            }
            if let Some((position, value)) = vector
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Graph rebuild refused record '{}': vector contains {} at index {}, \
                     which is not finite. Refusing to load a partial graph.",
                    id, value, position
                )));
            }
            validated.push((id, self.process_vector_for_space(vector), metadata));
        }

        self.set_rebuilding_from_persistence(true);

        // The interpreter lock is released across the whole rebuild, which is
        // the k-means, the graph inserts and nothing else. There is no Python
        // work left inside it to hold the lock for.
        //
        // The mutation guard is taken inside the released region rather than
        // around it, which is the shape `add` uses, so a loader waiting on
        // another writer waits without the lock.
        let (inserted, errors) = Python::attach(|py| {
            py.detach(|| {
                let _writers = self.writers.lock().unwrap();
                self.insert_parsed_records(validated, true)
            })
        });

        // Cleared before the result is judged rather than after, so the flag is
        // not left set on the way out of a failed rebuild.
        self.set_rebuilding_from_persistence(false);

        // A `PyErr`'s `Display` acquires the interpreter lock, so the messages
        // are formatted here and not above.
        let refused: Vec<String> = errors
            .into_iter()
            .filter_map(|error| error.into_counted_message())
            .collect();
        if !refused.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Graph rebuild refused {} of {} records, so the loaded index would \
                 hold records that no query can reach. Refusing to load a partial \
                 graph. Rejected records: {}",
                refused.len(),
                total,
                refused.join("; ")
            )));
        }

        Ok(inserted.len())
    }

    /// Set training IDs (for persistence loading only)
    pub(crate) fn set_training_ids(&mut self, ids: Vec<String>) {
        *self.training_ids.write().unwrap() = ids;
    }

    /// Restore the creation timestamp from the saved manifest
    ///
    /// For persistence loading only. `new_empty` stamps the load time, which is
    /// what `manifest.json` used to record as `created_at` on the next save.
    pub(crate) fn set_created_at(&mut self, created_at: String) {
        *self.created_at.write().unwrap() = created_at;
    }

    /// Restore when the codebook was fitted, from the saved directory
    ///
    /// For persistence loading only. See the field for why the value is carried
    /// rather than restamped.
    pub(crate) fn set_training_completed_at(&mut self, completed_at: Option<String>) {
        *self.training_completed_at.write().unwrap() = completed_at;
    }

    // ============================================================================
    // PERSISTENCE GETTERS - For accessing private fields from persistence module
    // ============================================================================

    /// Get the vector dimension
    pub fn get_dim(&self) -> usize {
        self.dim
    }

    /// Get the distance space (cosine, l2, l1) without cloning it
    ///
    /// Named `space_str` rather than `space` because the Python property that
    /// serves `index.space` is a `#[getter]` in the `#[pymethods]` block, and
    /// PyO3 takes the property name from the Rust method name. Two methods of
    /// one name on one type do not coexist across impl blocks, so the internal
    /// accessor is the one that moves.
    pub fn space_str(&self) -> &str {
        &self.space
    }

    /// Get the maximum number of bidirectional links per node
    ///
    /// **The one read of `m` in the crate**, so that `rebuild` writing it has a
    /// single place to be seen from. Acquire against the release the write
    /// makes, though every writer also holds `writers` and every reader that
    /// matters runs after it.
    pub fn get_m(&self) -> usize {
        self.m.load(Ordering::Acquire)
    }

    /// Get the construction parameter ef_construction. The one read of it, for
    /// the reason `get_m` gives.
    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction.load(Ordering::Acquire)
    }

    /// Get the expected size parameter. The one read of it, for the reason
    /// `get_m` gives.
    pub fn get_expected_size(&self) -> usize {
        self.expected_size.load(Ordering::Acquire)
    }

    /// How many generated ids this index has issued, for `config.json`
    pub(crate) fn get_generated_ids(&self) -> usize {
        *self.generated_ids.lock().unwrap()
    }

    /// Restore the generated id counter, never below what the records need
    ///
    /// `floor` is the highest N among the ids of the form `vec_N` the directory
    /// actually holds. A directory written before this field existed carries a
    /// zero and would otherwise reissue every generated id it holds, so the
    /// records themselves supply the floor. A directory that carries the field
    /// is at or above its own floor already and the max changes nothing.
    pub(crate) fn set_generated_ids(&mut self, saved: usize, floor: usize) {
        *self.generated_ids.lock().unwrap() = saved.max(floor);
    }

    /// The highest N among ids of the form `vec_N`, or zero where there is none
    ///
    /// A user is free to add a record called `vec_9` by hand, and this counts
    /// it, so the next generated id is above it. That is the point: the counter
    /// exists to stop an id being handed out twice.
    pub(crate) fn highest_generated_id(ids: impl Iterator<Item = impl AsRef<str>>) -> usize {
        ids.filter_map(|id| {
            id.as_ref()
                .strip_prefix("vec_")
                .and_then(|rest| rest.parse::<usize>().ok())
        })
        .max()
        .unwrap_or(0)
    }

    /// Get the current ID counter value (thread-safe)
    pub fn get_id_counter(&self) -> usize {
        *self.id_counter.lock().unwrap()
    }

    /// Get read access to the PQ codes HashMap (thread-safe)
    ///
    /// `pub(crate)` rather than `pub`, as are the four accessors below that
    /// also hand out a guard. They return a [`super::locks::ReadGuard`], which
    /// is a crate type, and a `pub` item returning one is a private type in a
    /// public interface. Nothing outside the crate could call them in any case,
    /// since `hnsw_index` is a private module.
    pub(crate) fn get_pq_codes(
        &self,
    ) -> ReadGuard<'_, { order::PQ_CODES }, HashMap<String, Vec<u8>>> {
        self.pq_codes.read().unwrap()
    }

    /// Get read access to the vector metadata HashMap (thread-safe)
    pub(crate) fn get_vector_metadata(
        &self,
    ) -> ReadGuard<'_, { order::VECTOR_METADATA }, HashMap<String, HashMap<String, Value>>> {
        self.vector_metadata.read().unwrap()
    }

    /// Get read access to the ID map (external ID -> internal ID)
    pub(crate) fn get_id_map(&self) -> ReadGuard<'_, { order::ID_MAP }, HashMap<String, usize>> {
        self.id_map.read().unwrap()
    }

    /// Get read access to the reverse ID map (internal ID -> external ID)
    pub(crate) fn get_rev_map(&self) -> ReadGuard<'_, { order::REV_MAP }, HashMap<usize, String>> {
        self.rev_map.read().unwrap()
    }

    /// Get reference to the quantization configuration
    pub fn get_quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    /// Get reference to the PQ instance
    pub fn get_pq(&self) -> Option<&Arc<crate::pq::PQ>> {
        self.pq.as_ref()
    }

    /// Helper to get quantization subvectors count
    pub fn get_quantization_subvectors(&self) -> usize {
        self.quantization_config
            .as_ref()
            .map(|config| config.subvectors)
            .unwrap_or(1)
    }

    /// Get the index creation timestamp
    pub fn get_created_at(&self) -> String {
        self.created_at.read().unwrap().clone()
    }

    /// When the codebook was fitted, or `None` on an index that never trained
    pub fn get_training_completed_at(&self) -> Option<String> {
        self.training_completed_at.read().unwrap().clone()
    }

    /// Get read access to training IDs (for persistence)
    pub(crate) fn get_training_ids(&self) -> ReadGuard<'_, { order::TRAINING_IDS }, Vec<String>> {
        self.training_ids.read().unwrap()
    }

    /// Get training threshold reached flag (for persistence)
    pub fn get_training_threshold_reached(&self) -> bool {
        self.training_threshold_reached
            .load(std::sync::atomic::Ordering::Acquire)
    }
}
