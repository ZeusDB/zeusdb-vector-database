// lib.rs
mod create_index_hnsw;

use pyo3::prelude::*;

#[pymodule]
fn zeusdb_vector_database(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<create_index_hnsw::HNSWIndex>()?;
    Ok(())
}
