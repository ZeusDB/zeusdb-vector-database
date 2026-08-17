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
//! - **File target**: `ZEUSDB_LOG_FILE` names the file that is written, exactly as
//!   given (e.g. `app.log` writes `app.log`). Set `ZEUSDB_LOG_ROTATION=daily` to
//!   bound disk growth, which appends a date (`app.log.2026-08-03`). The resolved
//!   path is logged at startup either way. Rotation is also available through
//!   `init_file_logging(log_dir, level, file_prefix)`, whose `file_prefix` is a
//!   prefix by name and by behaviour.
//! - **Programmatic control**: To take control programmatically, set
//!   `ZEUSDB_DISABLE_AUTO_LOGGING=1` before import, then call Python `init_*` functions.
//!
//! ## Environment Variables
//!
//! - `RUST_LOG`: Controls log filtering (takes precedence over ZEUSDB_LOG_LEVEL); format/target still come from ZEUSDB_*
//! - `ZEUSDB_LOG_LEVEL`: trace, debug, info, warn, error (default: warn)
//! - `ZEUSDB_LOG_FORMAT`: human, json (default: human)
//! - `ZEUSDB_LOG_TARGET`: stdout, stderr, file (default: stderr)
//! - `ZEUSDB_LOG_FILE`: log file path, written exactly as given (default: zeusdb.log)
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
use std::io;
use std::io::IsTerminal;
use std::sync::{Once, OnceLock};
use tracing::Subscriber;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{
    fmt::{self, time::UtcTime},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};

use tracing_subscriber::fmt::format::FmtSpan;

static INIT: Once = Once::new();
static WORKER_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

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

        // Name the destination when the target is a file, so the resolved path
        // is discoverable rather than inferred. Under daily rotation the
        // resolved name carries a date suffix and is not the name the caller
        // typed, which is the case this record exists for.
        if log_target == "file" {
            let requested_rotation = std::env::var("ZEUSDB_LOG_ROTATION").unwrap_or_default();
            if parse_rotation(&requested_rotation).is_none() {
                tracing::warn!(
                    operation = "logging_init_file",
                    value = %requested_rotation,
                    "Unrecognised ZEUSDB_LOG_ROTATION, using never. Valid values: daily, never"
                );
            }

            tracing::info!(
                operation = "logging_init_file",
                rotation = %configured_rotation().as_str(),
                log_file = %resolved_log_file_path(),
                "ZeusDB file logging writing to resolved path"
            );
        }
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
/// Returns true if initialization occurred, false if already initialized.
#[pyfunction(name = "init_file_logging")]
pub fn py_init_file_logging(
    log_dir: String,
    level: Option<String>,
    file_prefix: Option<String>,
) -> PyResult<bool> {
    // Input validation
    if log_dir.trim().is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "log_dir cannot be empty",
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
            file_prefix.unwrap_or_else(|| "zeusdb".to_string()),
        );
        let (non_blocking, guard) = tracing_appender::non_blocking(appender);
        let _ = WORKER_GUARD.set(guard);

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
/// Useful for determining whether to set ZEUSDB_DISABLE_AUTOLOG or not.
#[pyfunction]
pub fn is_logging_initialized() -> bool {
    INIT.is_completed()
}

/// Create environment filter with intelligent defaults for dependencies
fn create_env_filter(log_level: &str) -> EnvFilter {
    let base = format!(
        "zeusdb_vector_database={level},\
         rayon=warn,pyo3=warn,bincode=warn,serde_json=warn,\
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
    let log_file = std::env::var("ZEUSDB_LOG_FILE").unwrap_or_else(|_| "zeusdb.log".to_string());

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
    let log_file = std::env::var("ZEUSDB_LOG_FILE").unwrap_or_else(|_| "zeusdb.log".to_string());

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
            LogRotation::Daily => format!(".{}", chrono::Utc::now().format("%Y-%m-%d")),
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
fn create_file_appender(
    log_file_path: &str,
) -> Option<(
    tracing_appender::non_blocking::NonBlocking,
    tracing_appender::non_blocking::WorkerGuard,
)> {
    use std::borrow::Cow;
    use std::path::Path;

    let path = Path::new(log_file_path);

    let (directory, filename) = match (path.parent(), path.file_name()) {
        (Some(dir), Some(name)) if !dir.as_os_str().is_empty() => (dir, name.to_string_lossy()),
        (_, Some(name)) => (Path::new("."), name.to_string_lossy()),
        _ => (Path::new("."), Cow::from("zeusdb.log")),
    };

    // Silent failure for graceful degradation
    if std::fs::create_dir_all(directory).is_err() {
        return None;
    }

    let file_appender = match configured_rotation() {
        LogRotation::Never => tracing_appender::rolling::never(directory, &*filename),
        LogRotation::Daily => tracing_appender::rolling::daily(directory, &*filename),
    };
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    Some((non_blocking, guard))
}

/// The file the `file` target resolves to, absolute where that can be worked out
///
/// Emitted at startup so the destination is discoverable from the logs rather
/// than by guessing. Under `daily` this carries the date suffix, which is the
/// case where the resolved name differs from the one the caller typed.
fn resolved_log_file_path() -> String {
    use std::ffi::OsString;
    use std::path::Path;

    let configured = std::env::var("ZEUSDB_LOG_FILE").unwrap_or_else(|_| "zeusdb.log".to_string());
    let path = Path::new(&configured);

    let mut filename = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| OsString::from("zeusdb.log"));
    filename.push(configured_rotation().filename_suffix());

    match std::fs::canonicalize(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    ) {
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
