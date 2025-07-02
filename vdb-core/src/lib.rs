// lib.rs
mod hnsw_index;

use pyo3::prelude::*;

#[pymodule]
fn zeusdb_vector_database(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hnsw_index::HNSWIndex>()?;
    m.add_class::<hnsw_index::BatchResult>()?;
    Ok(())
}
