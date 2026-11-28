//! # ZeusDB Vector Database - Rust Logging Module
//!
//! This module provides structured logging for the Rust backend with automatic
//! initialization on module import and optional programmatic overrides.
//!
//! ## Important Notes
//!
//! - **Global and immutable**: After any initialization (auto or manual), the logging
//!   configuration is process-global and cannot be changed. Subsequent `init_*` calls
//!   will return `False` and have no effect, or raise `ValueError` for an empty
//!   `log_dir` or, where `RUST_LOG` is unset or does not parse, for a level they
//!   cannot read.
//! - **File target**: `ZEUSDB_LOG_FILE` names the file that is written, as given
//!   (e.g. `app.log` writes `app.log`). A path that names a folder, such as `.` or
//!   `logs/`, writes no file, and the target falls back to stderr as for a file it
//!   cannot open. Set `ZEUSDB_LOG_ROTATION=daily` to
//!   bound disk growth, which appends a date (`app.log.2026-08-03`). The resolved
//!   path is logged at startup either way. Rotation is also available through
//!   `init_file_logging(log_dir, level, file_prefix)`, whose `file_prefix` is a
//!   prefix by name and by behaviour.
//! - **Programmatic control**: To take control programmatically, set
//!   `ZEUSDB_DISABLE_AUTO_LOGGING=1` before import, then call Python `init_*` functions.
//! - **Drained at exit**: the `file` target writes from a worker thread fed by
//!   a channel, so a record emitted just before the process ends is still in
//!   flight when it ends. Importing the module registers `shutdown_logging`
//!   with `atexit`, which drains it. `os._exit` and an abort run no hook, and
//!   records still queued at either are lost. `stdout` and `stderr` do not go
//!   through the appender and are unaffected.
//!
//! ## Environment Variables
//!
//! - `RUST_LOG`: Controls log filtering (takes precedence over ZEUSDB_LOG_LEVEL); format/target still come from ZEUSDB_*
//! - `ZEUSDB_LOG_LEVEL`: trace, debug, info, warn, error (default: warn)
//! - `ZEUSDB_LOG_FORMAT`: human, json (default: human)
//! - `ZEUSDB_LOG_TARGET`: stdout, stderr, file (default: stderr)
//! - `ZEUSDB_LOG_FILE`: log file path, written as given, and a path that names a folder writes no file (default, where unset or empty: zeusdb.log)
//! - `ZEUSDB_LOG_ROTATION`: daily, never (default: never). daily appends a date to the file name
//! - `ZEUSDB_DISABLE_AUTO_LOGGING`: true, 1 or yes to disable auto-init (for programmatic control)
//! - `ZEUSDB_DISABLE_AUTOLOG`: deprecated alias for the above, honoured for compatibility
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
//! # Human-readable to a named file
//! export ZEUSDB_LOG_FORMAT=human
//! export ZEUSDB_LOG_TARGET=file
//! export ZEUSDB_LOG_FILE=logs/zeusdb.log  # Writes logs/zeusdb.log
//!
//! # The same, rotated daily to bound disk growth
//! export ZEUSDB_LOG_ROTATION=daily        # Writes logs/zeusdb.log.2026-08-03
//! ```

use pyo3::prelude::*;
use std::ffi::OsStr;
use std::io;
use std::io::IsTerminal;
use std::path::Path;
use std::sync::Once;
use tracing::Subscriber;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{
    fmt::{self, time::UtcTime},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};
use zeusdb_vector_core::Error;

use crate::PyEngineError;
use tracing_subscriber::fmt::format::FmtSpan;

static INIT: Once = Once::new();

/// The `tracing_appender` worker guard for the `file` target
///
/// The non-blocking appender hands each formatted record to a worker thread
/// over a channel and returns. Dropping this guard is the only drain that
/// channel has. The drop sends the worker a shutdown message behind whatever is
/// already queued and then waits for the acknowledgement, so everything queued
/// is written before the drop returns. `NonBlocking::flush` is a no-op, so
/// there is no other way to empty it.
///
/// It is a `Mutex<Option<_>>` rather than a `OnceLock` because a value held in
/// a `static` is never dropped, and the drop is the whole point. The guard has
/// to be taken back out to be dropped, and `OnceLock` hands out a mutable
/// borrow only through `&mut self`, which a `static` cannot give.
/// `shutdown_logging` takes it out and drops it, and the module registers that
/// with `atexit` so a normally exiting process needs no call.
///
/// `clippy.toml` disallows the standard `Mutex` so that a lock added to
/// the index cannot bypass the rank registry in `zeusdb_vector_hnsw::locks`. This is
/// not a lock on the index and is not reachable from one. It is taken twice
/// in the life of a process, once at import and once at exit, and no index
/// guard is held at either point, so it has no place in the declared order.
/// `pq` and `graph` carry the same exemption at module scope for the same
/// reason; this one is on the item, because the rest of this module holds no
/// lock at all.
#[allow(clippy::disallowed_types)]
static WORKER_GUARD: std::sync::Mutex<Option<WorkerGuard>> = std::sync::Mutex::new(None);

/// Hold the worker guard for the life of the process
///
/// The first guard wins, which is what `OnceLock::set` did. Only one can ever
/// arrive, because every initialization path runs inside `INIT.call_once` and
/// installs exactly one subscriber.
fn store_worker_guard(guard: WorkerGuard) {
    let mut slot = WORKER_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if slot.is_none() {
        *slot = Some(guard);
    }
    // Release the lock before `guard` goes out of scope, because in the
    // already-set case dropping it blocks on that appender's worker.
    drop(slot);
}

/// Drain the file appender and stop its worker
///
/// A worker thread is killed at process exit wherever it happens to be, so
/// without this whatever is still in the appender's channel is never written.
/// The loss is a tail, and it is as long as the backlog: a process that logs
/// faster than the worker writes loses most of its log, not its last line.
///
/// Registered with `atexit` when the module is imported. It is exposed as well,
/// for a caller that wants the file complete at a point of its own choosing.
///
/// After this returns the file appender is closed. A record emitted later is
/// still formatted and still handed to a channel with no reader, so it is
/// discarded. That is the right trade at exit, and it is why this is named a
/// shutdown rather than a flush. The `stdout` and `stderr` targets are
/// unaffected either way, because neither goes through the appender.
///
/// Returns true if a guard was taken and dropped, and false when there was
/// nothing to drain, which is every target other than `file`.
pub fn shutdown_logging() -> bool {
    // The temporary lock guard is released at the end of this statement, so the
    // blocking drop below happens with the mutex free.
    let guard = WORKER_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    let had_guard = guard.is_some();
    drop(guard);
    had_guard
}

/// Python-exposed shutdown, registered with `atexit` on import
///
/// Returns true if there was a file appender to drain.
#[pyfunction(name = "shutdown_logging")]
pub fn py_shutdown_logging(py: Python<'_>) -> bool {
    // The drop blocks on the appender's worker for as long as the backlog
    // takes. Nothing below touches a Python object, so the GIL is released for
    // it rather than held across a wait.
    py.detach(shutdown_logging)
}

/// Names of the auto-logging disable flag
///
/// `ZEUSDB_DISABLE_AUTO_LOGGING` is the published name. It is what the
/// documentation site, the package README and the Python layer all use, and it
/// is the one this reads. `ZEUSDB_DISABLE_AUTOLOG` is what the Rust read before
/// and is honoured as a deprecated alias so a process that set it keeps working.
const DISABLE_AUTOLOG_VAR: &str = "ZEUSDB_DISABLE_AUTO_LOGGING";
const DISABLE_AUTOLOG_VAR_DEPRECATED: &str = "ZEUSDB_DISABLE_AUTOLOG";

/// Read a boolean environment variable the way the Python layer reads it
///
/// A bare name with no value, or a value outside the truthy set, does not
/// disable anything. The Python layer has always required `true`, `1` or `yes`,
/// and the two layers disagreeing about what counts as set is the defect this
/// closes.
fn env_flag_is_set(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str(),
        "true" | "1" | "yes"
    )
}

/// Whether auto-initialization has been disabled by the caller
fn autolog_disabled() -> bool {
    env_flag_is_set(DISABLE_AUTOLOG_VAR) || env_flag_is_set(DISABLE_AUTOLOG_VAR_DEPRECATED)
}

/// Initialize logging automatically on module import
///
/// Respects ZEUSDB_DISABLE_AUTO_LOGGING for power users who want programmatic
/// control. Uses RUST_LOG if set; otherwise uses ZEUSDB_* environment variables.
///
/// Called automatically from lib.rs - users don't need to call this directly.
pub(crate) fn init_from_env_if_unset() {
    // Allow power users to opt out of auto-init
    if autolog_disabled() {
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

        // Initialize with appropriate layer, preserving format on file fallback.
        // What became of the file the `file` target names: written, or the
        // cause it could not be opened, and nothing for any other target.
        let mut file_layer: Option<Result<(), String>> = None;
        match (log_format.as_str(), log_target.as_str()) {
            ("json", "stdout") => {
                let _ = subscriber.with(create_json_stdout_layer::<_>()).try_init();
            }
            ("json", "stderr") => {
                let _ = subscriber.with(create_json_stderr_layer::<_>()).try_init();
            }
            ("json", "file") => match create_json_file_layer::<_>() {
                Ok(layer) => {
                    file_layer = Some(Ok(()));
                    let _ = subscriber.with(layer).try_init();
                }
                Err(cause) => {
                    // Fallback: preserve JSON format, use stderr
                    file_layer = Some(Err(cause));
                    let _ = subscriber.with(create_json_stderr_layer::<_>()).try_init();
                }
            },
            ("human", "stdout") => {
                let _ = subscriber.with(create_human_stdout_layer::<_>()).try_init();
            }
            ("human", "stderr") => {
                let _ = subscriber.with(create_human_stderr_layer::<_>()).try_init();
            }
            ("human", "file") => match create_human_file_layer::<_>() {
                Ok(layer) => {
                    file_layer = Some(Ok(()));
                    let _ = subscriber.with(layer).try_init();
                }
                Err(cause) => {
                    // Fallback: preserve human format, use stderr
                    file_layer = Some(Err(cause));
                    let _ = subscriber.with(create_human_stderr_layer::<_>()).try_init();
                }
            },
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

        // Name the destination when the target is a file, so the resolved path
        // is discoverable rather than inferred. Under daily rotation the
        // resolved name carries a date suffix and is not the name the caller
        // typed, which is the case this record exists for. Where the file
        // could not be opened, a warning names it, the cause and stderr.
        if log_target == "file" {
            let requested_rotation = std::env::var("ZEUSDB_LOG_ROTATION").unwrap_or_default();
            if parse_rotation(&requested_rotation).is_none() {
                tracing::warn!(
                    operation = "logging_init_file",
                    value = %requested_rotation,
                    "Unrecognised ZEUSDB_LOG_ROTATION, using never. Valid values: daily, never"
                );
            }

            match &file_layer {
                Some(Ok(())) => tracing::info!(
                    operation = "logging_init_file",
                    rotation = %configured_rotation().as_str(),
                    log_file = %resolved_log_file_path(),
                    "ZeusDB file logging writing to resolved path"
                ),
                Some(Err(cause)) => tracing::warn!(
                    operation = "logging_init_file",
                    log_file = %resolved_log_file_path(),
                    error = %cause,
                    "ZeusDB file logging cannot open the resolved path, so the engine's \
                     records go to stderr. Set ZEUSDB_LOG_FILE to a file this process can \
                     write"
                ),
                None => {}
            }
        }
    });
}

/// Simplified public interface for lib.rs integration
pub fn init_logging() {
    init_from_env_if_unset()
}

/// The filter `call` installs: `RUST_LOG` where it parses, and `level`
/// otherwise, `info` where it is left out. A `level` that does not parse,
/// or that is one word naming no level and no target of the engine's,
/// raises `ValueError` naming it, before the one initialization runs, so a
/// call that raises leaves logging as it was and a later call can still
/// initialize it.
fn requested_filter(call: &str, level: Option<&str>) -> PyResult<EnvFilter> {
    if let Ok(filter) = EnvFilter::try_from_default_env() {
        return Ok(filter);
    }
    let level = level.unwrap_or("info");
    let refused = |cause: &dyn std::fmt::Display| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{} takes a level such as 'info' or a filter such as \
             'zeusdb_vector_database=debug', and cannot read '{}': {}.",
            call, level, cause
        ))
    };
    // A name of the table `ZEUSDB_LOG_LEVEL` is read through, such as
    // Python's `warning`, is that level. `EnvFilter` reads one word that is
    // no level as a target at every level, as it would read a misspelt
    // level, so such a word is refused unless it is the engine's own target
    // or one under it. Any other text is a filter.
    let text = known_level(level).unwrap_or(level);
    if names_no_level(text) {
        return Err(refused(&"it names no level"));
    }
    EnvFilter::try_new(text).map_err(|error| refused(&error))
}

/// The target of the engine's records, which each record carries or one
/// under it.
const ENGINE_TARGET: &str = "zeusdb_vector_database";

/// Whether `text` is one word that `EnvFilter` reads as a target and not
/// as a level, being no level name and no target of the engine's own.
fn names_no_level(text: &str) -> bool {
    let word = text.trim();
    !word.is_empty()
        && !word.contains(['=', ',', '['])
        && word
            .parse::<tracing_subscriber::filter::LevelFilter>()
            .is_err()
        && word != ENGINE_TARGET
        && !word
            .strip_prefix(ENGINE_TARGET)
            .is_some_and(|rest| rest.starts_with("::"))
}

/// Python-exposed logging initialization (JSON to console)
///
/// Returns true if initialization occurred, false if already initialized,
/// and raises ValueError for a level it cannot read.
/// Forces JSON to stdout regardless of ZEUSDB_LOG_TARGET; use env vars for other formats.
/// `level` left out or `None` is `info`, under a `RUST_LOG` that takes precedence
/// where it parses.
#[pyfunction(name = "init_logging")]
#[pyo3(signature = (level = None))]
pub fn py_init_logging(level: Option<String>) -> PyResult<bool> {
    let filter = requested_filter("init_logging()", level.as_deref())?;
    let mut took_init = false;
    INIT.call_once(|| {
        took_init = true;
        let registry = Registry::default().with(filter).with(
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
        );

        let _ = registry.try_init();
    });
    Ok(took_init)
}

/// Python-exposed file logging initialization (JSON to rotating files)
///
/// Returns true if initialization occurred, false if already initialized,
/// and raises ValueError for an empty `log_dir` or a level it cannot read
/// and, before logging is set up, OSError for a folder it cannot make or a
/// log file it cannot open.
/// `level` left out or `None` is `info`, under a `RUST_LOG` that takes
/// precedence where it parses, and `file_prefix` left out or `None` is
/// `zeusdb`.
#[pyfunction(name = "init_file_logging")]
#[pyo3(signature = (log_dir, level = None, file_prefix = None))]
pub fn py_init_file_logging(
    log_dir: String,
    level: Option<String>,
    file_prefix: Option<String>,
) -> Result<bool, PyEngineError> {
    // Input validation
    if log_dir.trim().is_empty() {
        return Err(Error::LogDirEmpty.into());
    }
    let filter = requested_filter("init_file_logging()", level.as_deref())?;

    // Where the one setup has run, nothing is made or opened. Otherwise the
    // folder and the file come before the setup, so a call that cannot make
    // the one or open the other raises OSError and leaves the setup to a
    // later call.
    if INIT.is_completed() {
        return Ok(false);
    }
    std::fs::create_dir_all(&log_dir).map_err(|error| {
        pyo3::exceptions::PyOSError::new_err(format!(
            "init_file_logging() cannot make the folder {}: {}.",
            log_dir, error
        ))
    })?;
    let appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix(file_prefix.unwrap_or_else(|| "zeusdb".to_string()))
        .build(&log_dir)
        .map_err(|error| {
            let cause = std::error::Error::source(&error)
                .map_or_else(|| error.to_string(), |source| source.to_string());
            pyo3::exceptions::PyOSError::new_err(format!(
                "init_file_logging() cannot open a log file in {}: {}.",
                log_dir, cause
            ))
        })?;

    let mut took_init = false;
    INIT.call_once(|| {
        took_init = true;

        let (non_blocking, guard) = tracing_appender::non_blocking(appender);
        store_worker_guard(guard);

        let registry = Registry::default().with(filter).with(
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
                .with_writer(non_blocking),
        );

        let _ = registry.try_init();
    });
    Ok(took_init)
}

/// Check if logging has been initialized
///
/// Returns true if logging initialization has occurred (either auto or manual).
#[pyfunction]
pub fn is_logging_initialized() -> bool {
    INIT.is_completed()
}

/// The dependencies whose records the default filter holds at `warn`,
/// whatever level the engine's own records run at. Each is a package in the
/// workspace's lock file; see `every_quieted_crate_is_in_the_lock_file`.
const QUIETED_CRATES: [&str; 3] = ["rayon", "pyo3", "serde_json"];

/// Create environment filter with intelligent defaults for dependencies
fn create_env_filter(log_level: &str) -> EnvFilter {
    let mut directives = format!("zeusdb_vector_database={}", canonical_level(log_level));
    for name in QUIETED_CRATES {
        directives.push_str(&format!(",{}=warn", name));
    }
    EnvFilter::new(directives)
}

/// The one place a level name becomes a level, for both layers
///
/// The two layers used to disagree about the spelling. Python's `logging` names
/// the level `WARNING` and `CRITICAL`, and `tracing`'s `EnvFilter` names it
/// `warn` and has no critical at all. So `ZEUSDB_LOG_LEVEL=warning` was accepted
/// by the Python side and rejected by this one, which printed
/// `ignoring zeusdb_vector_database=warning` to stderr and then filtered nothing
/// through the directive, and `ZEUSDB_LOG_LEVEL=warn` was the reverse: accepted
/// here and recorded as invalid by the Python side, which fell back to its own
/// default.
///
/// Both spellings of each level are now one level, resolved here and by
/// `_LEVEL_ALIASES` in `logging_config.py`, which is the Python half of the same
/// table.
///
/// An unrecognised name falls back to `warn`, which is the default this module
/// already used, rather than being passed through for `EnvFilter` to reject.
fn canonical_level(log_level: &str) -> &'static str {
    known_level(log_level).unwrap_or("warn")
}

/// The level a name of the table above names, whatever its case and the
/// space around it, or `None` for any other text.
fn known_level(log_level: &str) -> Option<&'static str> {
    match log_level.trim().to_lowercase().as_str() {
        "trace" => Some("trace"),
        "debug" => Some("debug"),
        "info" => Some("info"),
        "warn" | "warning" => Some("warn"),
        "error" | "err" | "critical" | "fatal" => Some("error"),
        "off" | "none" | "silent" => Some("off"),
        _ => None,
    }
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

/// Create JSON formatter for file output with rotation, or the cause the
/// file cannot be opened
fn create_json_file_layer<S>() -> Result<Box<dyn Layer<S> + Send + Sync + 'static>, String>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let log_file = log_file_setting();

    create_file_appender(&log_file).map(|(non_blocking, guard)| {
        store_worker_guard(guard);
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

/// Create human-readable formatter for file output with rotation, or the
/// cause the file cannot be opened
fn create_human_file_layer<S>() -> Result<Box<dyn Layer<S> + Send + Sync + 'static>, String>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let log_file = log_file_setting();

    create_file_appender(&log_file).map(|(non_blocking, guard)| {
        store_worker_guard(guard);
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

/// How the `file` target rotates
///
/// `never` is the default, so `ZEUSDB_LOG_FILE` names the file that is written
/// and the Python and Rust layers write the same one. `daily` routes to the
/// rolling appender, which reads the value as a directory plus a base name and
/// appends the date, and is the way to bound disk growth without an external
/// log rotator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogRotation {
    Never,
    Daily,
}

impl LogRotation {
    fn as_str(self) -> &'static str {
        match self {
            LogRotation::Never => "never",
            LogRotation::Daily => "daily",
        }
    }

    /// The suffix the appender appends to the base filename, if any
    fn filename_suffix(self) -> String {
        match self {
            LogRotation::Never => String::new(),
            // tracing-appender formats a DAILY rotation as `{base}.{YYYY-MM-DD}`
            // against the UTC date.
            LogRotation::Daily => format!(".{}", zeusdb_vector_core::Timestamp::now().to_date()),
        }
    }
}

/// Parse a `ZEUSDB_LOG_ROTATION` value, returning None for anything unrecognised
///
/// An unset or empty value is `never`, which is what makes the file target write
/// the file the caller named unless rotation is asked for.
fn parse_rotation(raw: &str) -> Option<LogRotation> {
    match raw.trim().to_lowercase().as_str() {
        "" | "never" => Some(LogRotation::Never),
        "daily" => Some(LogRotation::Daily),
        _ => None,
    }
}

/// The rotation the environment asks for, falling back to `never`
fn configured_rotation() -> LogRotation {
    let raw = std::env::var("ZEUSDB_LOG_ROTATION").unwrap_or_default();
    parse_rotation(&raw).unwrap_or(LogRotation::Never)
}

/// The value of `ZEUSDB_LOG_FILE`, or `zeusdb.log` where it is unset, empty
/// or not Unicode
///
/// The Python layer reads an empty value as unset as well, so neither layer
/// reads it as a path.
fn log_file_setting() -> String {
    std::env::var("ZEUSDB_LOG_FILE")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "zeusdb.log".to_string())
}

/// The folder and the name of the file `path` names, or `None` where it
/// names a folder and so no file
///
/// A path names a folder where its last part is empty, `.` or `..`, read
/// from its text before `Path::components` drops a trailing separator and a
/// trailing `.`, where it holds no name, being a root or a drive, or where
/// it is a folder that exists, beside which daily rotation would otherwise
/// write a dated file. A bare name is in the working folder.
/// `_names_a_folder` in `logging_config.py` keeps the same rule, so the two
/// layers refuse the same paths.
fn file_in_folder(path: &Path) -> Option<(&Path, &OsStr)> {
    let text = path.as_os_str().as_encoded_bytes();
    let last = text
        .rsplit(|&byte| std::path::is_separator(char::from(byte)))
        .next()
        .unwrap_or_default();
    if matches!(last, b"" | b"." | b"..") || path.is_dir() {
        return None;
    }
    let name = path.file_name()?;
    let directory = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Some((directory, name))
}

/// Create a file appender for the file the caller named
///
/// `ZEUSDB_LOG_FILE=app.log` writes `app.log` under the default rotation of
/// `never`. It previously always went through `tracing_appender::rolling::daily`,
/// which reads the path as a directory plus a base name and appends the date, so
/// the file the caller named was never written and nothing said where the output
/// had gone. Three things point at a file and none pointed at a prefix. The
/// published documentation calls it a log file path with a default of
/// `zeusdb.log`, the Python layer opens the same value with
/// `logging.FileHandler` and writes it verbatim, and the variable is named
/// `_FILE`. Rotation stays available for the callers who need disk growth
/// bounded, under `ZEUSDB_LOG_ROTATION=daily`, where the date suffix is asked
/// for rather than imposed.
///
/// A path that names a folder, by `file_in_folder`, names no file, so it is
/// refused with the cause `is a directory` before any folder is made or file
/// opened, as the Python layer refuses it.
fn create_file_appender(
    log_file_path: &str,
) -> Result<
    (
        tracing_appender::non_blocking::NonBlocking,
        tracing_appender::non_blocking::WorkerGuard,
    ),
    String,
> {
    let Some((directory, filename)) = file_in_folder(Path::new(log_file_path)) else {
        return Err(io::ErrorKind::IsADirectory.to_string());
    };

    // A folder this process cannot make is returned as the cause the setup
    // warns of before it falls back to stderr.
    std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;

    // Built through the builder, whose `build` returns the failure that
    // `rolling::never` and `rolling::daily` panic with, so a file this
    // process cannot open falls back to stderr with its cause, as a folder
    // it cannot make does.
    let rotation = match configured_rotation() {
        LogRotation::Never => tracing_appender::rolling::Rotation::NEVER,
        LogRotation::Daily => tracing_appender::rolling::Rotation::DAILY,
    };
    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(rotation)
        .filename_prefix(filename.to_string_lossy().into_owned())
        .build(directory)
        .map_err(|error| {
            std::error::Error::source(&error)
                .map_or_else(|| error.to_string(), |source| source.to_string())
        })?;
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    Ok((non_blocking, guard))
}

/// The file the `file` target resolves to, absolute where that can be worked out
///
/// Emitted at startup so the destination is discoverable from the logs rather
/// than by guessing. Under `daily` this carries the date suffix, which is the
/// case where the resolved name differs from the one the caller typed. A path
/// that names a folder is given as that folder, absolute where it exists,
/// since no file is written for it.
fn resolved_log_file_path() -> String {
    let configured = log_file_setting();
    let path = Path::new(&configured);

    let Some((directory, name)) = file_in_folder(path) else {
        return match std::fs::canonicalize(path) {
            Ok(folder) => folder.to_string_lossy().into_owned(),
            Err(_) => configured,
        };
    };
    let mut filename = name.to_os_string();
    filename.push(configured_rotation().filename_suffix());

    match std::fs::canonicalize(directory) {
        Ok(dir) => dir.join(&filename).to_string_lossy().into_owned(),
        // The directory could not be canonicalized, which is not a reason to
        // report the name the caller typed when rotation has changed it.
        Err(_) => path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(|parent| parent.join(&filename))
            .unwrap_or_else(|| std::path::PathBuf::from(&filename))
            .to_string_lossy()
            .into_owned(),
    }
}

/// Check if output is a TTY and colors should be used
///
/// `std::io::IsTerminal` rather than the `atty` crate, which is unmaintained
/// and carries RUSTSEC-2021-0145. The standard library has answered this
/// question since Rust 1.70 and does so through the same system calls, being
/// `isatty` on unix and `GetConsoleMode` plus the msys named pipe check on
/// Windows. A redirected stream fails that check and reports false, which is
/// what `atty` reported for the same handle.
fn is_tty_with_color(target: &str) -> bool {
    // Respect NO_COLOR environment variable
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }

    match target {
        "stderr" => io::stderr().is_terminal(),
        _ => io::stdout().is_terminal(),
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
        use tracing_subscriber::prelude::*;

        // Asserts on what the filter admits, not on its Debug rendering. The
        // Debug form of EnvFilter is a derived, unstable representation and
        // carries no compatibility promise across tracing-subscriber releases.
        let subscriber = tracing_subscriber::registry().with(create_env_filter("debug"));

        tracing::subscriber::with_default(subscriber, || {
            // The crate target is configured at the requested level, so debug
            // and everything above it is admitted.
            assert!(tracing::event_enabled!(
                target: "zeusdb_vector_database",
                tracing::Level::DEBUG
            ));
            assert!(tracing::event_enabled!(
                target: "zeusdb_vector_database",
                tracing::Level::ERROR
            ));

            // Trace sits below the requested level and is rejected.
            assert!(!tracing::event_enabled!(
                target: "zeusdb_vector_database",
                tracing::Level::TRACE
            ));

            // Dependency targets stay pinned at warn regardless of the level
            // asked for, which is the point of the noise-suppressing directives.
            assert!(!tracing::event_enabled!(
                target: "rayon",
                tracing::Level::INFO
            ));
            assert!(tracing::event_enabled!(
                target: "rayon",
                tracing::Level::WARN
            ));
        });
    }

    /// Every crate the default filter quiets is a package the workspace's
    /// lock file holds, so no directive names a crate the engine does not
    /// depend on.
    #[test]
    fn every_quieted_crate_is_in_the_lock_file() {
        let lock = include_str!("../../../Cargo.lock");
        for name in QUIETED_CRATES {
            let entry = format!("name = \"{}\"", name);
            assert!(
                lock.lines().any(|line| line == entry),
                "{name} is not a package in Cargo.lock"
            );
        }
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

    /// A path whose last part is empty, `.` or `..`, a root, a drive and a
    /// folder that exists name a folder and so no file, and a path that ends
    /// in a name names that file in its parent, or in the working folder
    /// where it has none, on every system.
    #[test]
    fn a_path_that_names_a_folder_names_no_file() {
        let folder = std::env::temp_dir().join(format!("zeusdb-log-folder-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let existing = folder.to_string_lossy().into_owned();
        for text in [
            ".",
            "..",
            "logs/",
            "logs/.",
            "logs/..",
            "a/b/..",
            "/",
            existing.as_str(),
        ] {
            assert_eq!(file_in_folder(Path::new(text)), None, "{text}");
        }
        assert_eq!(
            file_in_folder(Path::new("app.log")),
            Some((Path::new("."), OsStr::new("app.log")))
        );
        assert_eq!(
            file_in_folder(Path::new("logs/app.log")),
            Some((Path::new("logs"), OsStr::new("app.log")))
        );
        // A backslash separates on Windows alone, and a drive is Windows' own.
        if cfg!(windows) {
            for text in ["logs\\", "logs\\.", "C:\\", "C:", "\\\\server\\share\\"] {
                assert_eq!(file_in_folder(Path::new(text)), None, "{text}");
            }
        } else {
            assert_eq!(
                file_in_folder(Path::new("logs\\")),
                Some((Path::new("."), OsStr::new("logs\\")))
            );
        }
        std::fs::remove_dir(&folder).unwrap();
    }
}
