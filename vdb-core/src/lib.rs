// lib.rs
mod hnsw_index;
mod pq;
mod persistence;

use pyo3::prelude::*;

#[pymodule]
fn zeusdb_vector_database(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hnsw_index::HNSWIndex>()?;
    m.add_class::<hnsw_index::AddResult>()?;

    // Add the load_index function directly
    m.add_function(wrap_pyfunction!(persistence::load_index, m)?)?;
    
    Ok(())
}
