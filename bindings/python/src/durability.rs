//! Reading the durability a caller names.
//!
//! `journal_to` and `load` take the policy as one of three strings and the
//! interval as a count of milliseconds, and this is the one place the
//! strings are read, so the two calls refuse the same mistakes in the same
//! words.

use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeusdb_vector_hnsw::Durability;

/// The three names, as the two calls take them.
const NAMES: [&str; 3] = ["call", "interval", "none"];

/// The policy `name` and `interval_ms` describe.
///
/// `interval_ms` belongs to `"interval"` alone and is refused under either
/// other name, since a caller who wrote it meant the interval policy or
/// meant nothing, and either way the call should say so. Left out under
/// `"interval"` it takes the engine's default.
pub(crate) fn parse_durability(name: &str, interval_ms: Option<u64>) -> PyResult<Durability> {
    if !NAMES.contains(&name) {
        return Err(PyValueError::new_err(format!(
            "durability must be 'call', 'interval' or 'none', got '{}'. 'call' puts every \
             record of a call on the device before the call returns, 'interval' does that \
             from a thread within interval_ms milliseconds of the call, and 'none' leaves it \
             to the next checkpoint.",
            name
        )));
    }
    match interval_ms {
        Some(_) if name != "interval" => {
            return Err(PyValueError::new_err(format!(
                "interval_ms applies to durability='interval' alone, and durability is '{}'",
                name
            )));
        }
        Some(0) => {
            return Err(PyValueError::new_err(
                "interval_ms must be at least 1, being the milliseconds between the journal's \
                 flushes",
            ));
        }
        _ => {}
    }
    Ok(match name {
        "call" => Durability::PerCall,
        "interval" => Durability::PerInterval(match interval_ms {
            Some(ms) => Duration::from_millis(ms),
            None => Durability::DEFAULT_INTERVAL,
        }),
        _ => Durability::NoSync,
    })
}
