//! # ZeusDB Vector Database - Rust Logging Module
//!
//! This module provides structured logging for the Rust backend with automatic
//! initialization on module import and optional programmatic overrides.
//!
//! ## Important Notes
//! 
//! - **Global and immutable**: After any initialization (auto or manual), the logging 
//!   configuration is process-global and cannot be changed. Subsequent `init_*` calls 
//!   will return `False` and have no effect.
//! - **File rotation**: `ZEUSDB_LOG_FILE` is treated as "directory + base filename" 
//!   for daily rotation (e.g., `logs/zeusdb.log` creates `logs/zeusdb.log.2024-01-15`).
//! - **Programmatic control**: To take control programmatically, set 
//!   `ZEUSDB_DISABLE_AUTOLOG=1` before import, then call Python `init_*` functions.
//!
//! ## Environment Variables
//! 
//! - `RUST_LOG`: Controls log filtering (takes precedence over ZEUSDB_LOG_LEVEL); format/target still come from ZEUSDB_*
//! - `ZEUSDB_LOG_LEVEL`: trace, debug, info, warn, error (default: warn)
//! - `ZEUSDB_LOG_FORMAT`: human, json (default: human) 
//! - `ZEUSDB_LOG_TARGET`: stdout, stderr, file (default: stderr)
//! - `ZEUSDB_LOG_FILE`: log file path (default: zeusdb.log)
//! - `ZEUSDB_DISABLE_AUTOLOG`: Set to disable auto-init (for programmatic control)
//! - `NO_COLOR`: Disable colored output (respects standard)
//!
//! ## Usage Examples
//!
//! ```rust
//! use tracing::{info, debug, warn, error, trace};
//!
//! // Simple structured logging
//! info!("Index created successfully");
//! 
//! // Structured logging with fields
//! info!(
//!     operation = "vector_add",
//!     vector_count = 1000,
//!     duration_ms = 150,
//!     "Batch operation completed"
//! );
//! ```
//!
//! ## Configuration Examples
//!
//! ```bash
//! # JSON to console (matches Python init defaults)
//! export ZEUSDB_LOG_FORMAT=json
//! export ZEUSDB_LOG_TARGET=stdout
//!
//! # Human-readable to file with daily rotation
//! export ZEUSDB_LOG_FORMAT=human
//! export ZEUSDB_LOG_TARGET=file
//! export ZEUSDB_LOG_FILE=logs/zeusdb.log  # Creates logs/zeusdb.log.2024-01-15
//! ```

use tracing_subscriber::{
    fmt::{self, time::UtcTime},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
    Registry,
    Layer,
};
use tracing::Subscriber;
use tracing_subscriber::registry::LookupSpan;
use std::sync::{Once, OnceLock};
use std::io;
use tracing_appender::non_blocking::WorkerGuard;
use pyo3::prelude::*;

use tracing_subscriber::fmt::format::FmtSpan;

static INIT: Once = Once::new();
static WORKER_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

/// Initialize logging automatically on module import
/// 
/// Respects ZEUSDB_DISABLE_AUTOLOG for power users who want programmatic control.
/// Uses RUST_LOG if set; otherwise uses ZEUSDB_* environment variables.
/// 
/// Called automatically from lib.rs - users don't need to call this directly.
pub(crate) fn init_from_env_if_unset() {
    // Allow power users to opt out of auto-init
    if std::env::var("ZEUSDB_DISABLE_AUTOLOG").is_ok() {
        return;
    }

    INIT.call_once(|| {
        // Level configuration - RUST_LOG takes precedence
        let filter = if let Ok(rust_log) = std::env::var("RUST_LOG") {
            EnvFilter::new(rust_log)
        } else {
            let log_level = std::env::var("ZEUSDB_LOG_LEVEL")
                .unwrap_or_else(|_| "warn".to_string())
                .to_lowercase();
            create_env_filter(&log_level)
        };

        // Format and target configuration
        let log_format = std::env::var("ZEUSDB_LOG_FORMAT")
            .unwrap_or_else(|_| "human".to_string())
            .to_lowercase();
        
        let log_target = std::env::var("ZEUSDB_LOG_TARGET")
            .unwrap_or_else(|_| "stderr".to_string())
            .to_lowercase();

        // Create base subscriber and consume it in the match
        let subscriber = Registry::default().with(filter);
        
        // Initialize with appropriate layer, preserving format on file fallback
        match (log_format.as_str(), log_target.as_str()) {
            ("json", "stdout") => { 
                let _ = subscriber.with(create_json_stdout_layer::<_>()).try_init(); 
            }
            ("json", "stderr") => { 
                let _ = subscriber.with(create_json_stderr_layer::<_>()).try_init(); 
            }
            ("json", "file") => {
                if let Some(layer) = create_json_file_layer::<_>() {
                    let _ = subscriber.with(layer).try_init();
                } else {
                    // Fallback: preserve JSON format, use stderr
                    let _ = subscriber.with(create_json_stderr_layer::<_>()).try_init();
                }
            }
            ("human", "stdout") => { 
                let _ = subscriber.with(create_human_stdout_layer::<_>()).try_init(); 
            }
            ("human", "stderr") => { 
                let _ = subscriber.with(create_human_stderr_layer::<_>()).try_init(); 
            }
            ("human", "file") => {
                if let Some(layer) = create_human_file_layer::<_>() {
                    let _ = subscriber.with(layer).try_init();
                } else {
                    // Fallback: preserve human format, use stderr
                    let _ = subscriber.with(create_human_stderr_layer::<_>()).try_init();
                }
            }
            _ => { 
                // Unknown format/target - safe fallback
                let _ = subscriber.with(create_human_stderr_layer::<_>()).try_init(); 
            }
        }

        // Log a breadcrumb to confirm initialization (visible only if level allows)
        tracing::trace!(
            operation = "logging_init",
            format = %log_format,
            target = %log_target,
            rust_log_set = std::env::var("RUST_LOG").is_ok(),
            "ZeusDB logging initialized successfully"
        );
    });
}

/// Simplified public interface for lib.rs integration
pub fn init_logging() {
    init_from_env_if_unset()
}

/// Python-exposed logging initialization (JSON to console)
/// 
/// Returns true if initialization occurred, false if already initialized.
/// Forces JSON to stdout regardless of ZEUSDB_LOG_TARGET; use env vars for other formats.
#[pyfunction(name = "init_logging")]
pub fn py_init_logging(level: Option<String>) -> PyResult<bool> {
    let mut took_init = false;
    INIT.call_once(|| {
        took_init = true;
        let filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(level.as_deref().unwrap_or("info")))
            .unwrap();

        let registry = Registry::default()
            .with(filter)
            .with(
                fmt::layer()
                    .json()
                    .with_timer(UtcTime::rfc_3339())
                    .with_current_span(true)
                    .with_span_list(true)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(false)
                    .with_file(true)
                    .with_line_number(true)
                    .with_level(true)
                    .with_ansi(false)
                    .with_writer(io::stdout)
            );

        let _ = registry.try_init();
    });
    Ok(took_init)
}

/// Python-exposed file logging initialization (JSON to rotating files)
/// 
/// Returns true if initialization occurred, false if already initialized.
#[pyfunction(name = "init_file_logging")]
pub fn py_init_file_logging(
    log_dir: String, 
    level: Option<String>, 
    file_prefix: Option<String>
) -> PyResult<bool> {
    // Input validation
    if log_dir.trim().is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "log_dir cannot be empty"
        ));
    }

    let mut took_init = false;
    INIT.call_once(|| {
        took_init = true;
        
        let filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(level.as_deref().unwrap_or("info")))
            .unwrap();

        // Try to create log directory
        if let Err(e) = std::fs::create_dir_all(&log_dir) {
            // Install fallback subscriber first, then warn
            let _ = Registry::default()
                .with(filter)
                .with(create_json_stderr_layer::<_>())
                .try_init();
            
            tracing::warn!(
                operation = "create_log_dir",
                error = ?e,
                path = %log_dir,
                "Failed to create log directory, using stderr instead"
            );
            return;
        }
        
        let appender = tracing_appender::rolling::daily(
            log_dir, 
            file_prefix.unwrap_or_else(|| "zeusdb".to_string())
        );
        let (non_blocking, guard) = tracing_appender::non_blocking(appender);
        let _ = WORKER_GUARD.set(guard);

        let registry = Registry::default()
            .with(filter)
            .with(
                fmt::layer()
                    .json()
                    .with_timer(UtcTime::rfc_3339())
                    .with_current_span(true)
                    .with_span_list(true)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(false)
                    .with_file(true)
                    .with_line_number(true)
                    .with_level(true)
                    .with_ansi(false)
                    .with_writer(non_blocking)
            );

        let _ = registry.try_init();
    });
    Ok(took_init)
}

/// Check if logging has been initialized
/// 
/// Returns true if logging initialization has occurred (either auto or manual).
/// Useful for determining whether to set ZEUSDB_DISABLE_AUTOLOG or not.
#[pyfunction]
pub fn is_logging_initialized() -> bool {
    INIT.is_completed()
}

/// Create environment filter with intelligent defaults for dependencies
fn create_env_filter(log_level: &str) -> EnvFilter {
    let base = format!(
        "zeusdb_vector_database={level},\
         hnsw_rs=warn,rayon=warn,pyo3=warn,bincode=warn,serde_json=warn,\
         mio=warn,tokio=warn",
        level = log_level
    );
    EnvFilter::new(base)
}

/// Create JSON formatter for stdout output
fn create_json_stdout_layer<S>() -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    Box::new(
        fmt::layer()
            .json()
            .with_timer(UtcTime::rfc_3339())
            .with_current_span(true)
            .with_span_list(true)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(false)
            .with_file(true)
            .with_line_number(true)
            .with_level(true)
            .with_ansi(false)
            .with_writer(io::stdout),
    )
}

/// Create JSON formatter for stderr output
fn create_json_stderr_layer<S>() -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    Box::new(
        fmt::layer()
            .json()
            .with_timer(UtcTime::rfc_3339())
            .with_current_span(true)
            .with_span_list(true)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(false)
            .with_file(true)
            .with_line_number(true)
            .with_level(true)
            .with_ansi(false)
            .with_writer(io::stderr),
    )
}


/// Create human-readable formatter for stdout output
fn create_human_stdout_layer<S>() -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let use_ansi = is_tty_with_color("stdout");
    Box::new(
        fmt::layer()
            .compact()
            .with_timer(UtcTime::rfc_3339())
            .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT) // <-- Fix here
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .with_file(false)
            .with_line_number(false)
            .with_level(true)
            .with_ansi(use_ansi)
            .with_writer(io::stdout),
    )
}





/// Create human-readable formatter for stderr output
fn create_human_stderr_layer<S>() -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let use_ansi = is_tty_with_color("stderr");
    Box::new(
        fmt::layer()
            .compact()
            .with_timer(UtcTime::rfc_3339())
            .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT) // <-- Fix here
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .with_file(false)
            .with_line_number(false)
            .with_level(true)
            .with_ansi(use_ansi)
            .with_writer(io::stderr),
    )
}



/// Create JSON formatter for file output with rotation
fn create_json_file_layer<S>() -> Option<Box<dyn Layer<S> + Send + Sync + 'static>>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let log_file = std::env::var("ZEUSDB_LOG_FILE")
        .unwrap_or_else(|_| "zeusdb.log".to_string());

    create_file_appender(&log_file).map(|(non_blocking, guard)| {
        let _ = WORKER_GUARD.set(guard);
        Box::new(
            fmt::layer()
                .json()
                .with_timer(UtcTime::rfc_3339())
                .with_current_span(true)
                .with_span_list(true)
                .with_writer(non_blocking)
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(false)
                .with_file(true)
                .with_line_number(true)
                .with_level(true)
                .with_ansi(false),
        ) as Box<dyn Layer<S> + Send + Sync + 'static>
    })
}

/// Create human-readable formatter for file output with rotation
fn create_human_file_layer<S>() -> Option<Box<dyn Layer<S> + Send + Sync + 'static>>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let log_file = std::env::var("ZEUSDB_LOG_FILE")
        .unwrap_or_else(|_| "zeusdb.log".to_string());

    create_file_appender(&log_file).map(|(non_blocking, guard)| {
        let _ = WORKER_GUARD.set(guard);
        Box::new(
            fmt::layer()
                .compact()
                .with_timer(UtcTime::rfc_3339())
                .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
                .with_writer(non_blocking)
                .with_target(false)
                .with_thread_ids(false)
                .with_thread_names(false)
                .with_file(false)
                .with_line_number(false)
                .with_level(true)
                .with_ansi(false),
        ) as Box<dyn Layer<S> + Send + Sync + 'static>
    })
}

/// Create file appender with intelligent path handling and daily rotation
fn create_file_appender(log_file_path: &str) -> Option<(tracing_appender::non_blocking::NonBlocking, tracing_appender::non_blocking::WorkerGuard)> {
    use std::path::Path;
    use std::borrow::Cow;
    
    let path = Path::new(log_file_path);
    
    let (directory, filename) = match (path.parent(), path.file_name()) {
        (Some(dir), Some(name)) if !dir.as_os_str().is_empty() => (dir, name.to_string_lossy()),
        (_, Some(name)) => (Path::new("."), name.to_string_lossy()),
        _ => (Path::new("."), Cow::from("zeusdb.log")),
    };
    
    // Silent failure for graceful degradation
    if let Err(_) = std::fs::create_dir_all(directory) {
        return None;
    }
    
    let file_appender = tracing_appender::rolling::daily(directory, &*filename);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    
    Some((non_blocking, guard))
}

/// Check if output is a TTY and colors should be used
fn is_tty_with_color(target: &str) -> bool {
    // Respect NO_COLOR environment variable
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    
    #[cfg(feature = "atty")]
    {
        match target {
            "stderr" => atty::is(atty::Stream::Stderr),
            _ => atty::is(atty::Stream::Stdout),
        }
    }
    
    #[cfg(not(feature = "atty"))]
    {
        let _ = target;
        // Simple fallback: check if TERM is set and we're not in CI
        std::env::var("TERM").is_ok() &&
        !std::env::var("CI").is_ok() &&
        !std::env::var("GITHUB_ACTIONS").is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_auto_init_idempotent() {
        init_from_env_if_unset();
        init_from_env_if_unset(); // Should not panic
    }
    
    #[test]
    fn test_public_init_alias() {
        init_logging(); // Should work without panic
    }
    
    #[test]
    fn test_env_filter_creation() {
        let filter = create_env_filter("debug");
        assert!(format!("{:?}", filter).contains("debug"));
    }
    
    #[test]
    fn test_init_status_check() {
        // Before any init
        let _was_initialized = is_logging_initialized();
        
        // After init
        init_logging();
        let now_initialized = is_logging_initialized();
        
        // Should show state change (or already was initialized)
        assert!(now_initialized);
    }
}
