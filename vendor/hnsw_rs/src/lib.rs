#![cfg_attr(feature = "stdsimd", feature(portable_simd))]
//
// for logging (debug mostly, switched at compile time in cargo.toml)
use env_logger::Builder;

use lazy_static::lazy_static;

pub mod api;
// ZEUSDB PATCH 7. Three modules are compiled but never reached from ZeusDB, and
// two of them are the sole reason `mmap-rs` and `indexmap` enter the resolution.
// Each is now behind a default-off feature. See ZEUSDB-PATCH.md.
#[cfg(feature = "mmap")]
pub mod datamap;
pub mod filter;
#[cfg(feature = "flatten")]
pub mod flatten;
pub mod hnsw;
pub mod hnswio;
#[cfg(feature = "libext")]
pub mod libext;
pub mod prelude;

// we impose our version of anndists
pub use anndists;

lazy_static! {
    static ref LOG: u64 = init_log();
}

// install a logger facility
#[allow(unused)]
fn init_log() -> u64 {
    Builder::from_default_env().init();
    println!("\n ************** initializing logger *****************\n");
    1
}
