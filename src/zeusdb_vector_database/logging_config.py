# ============================================================================
# zeusdb_vector_database/logging_config.py
# Smart Auto-Configuration for ZeusDB Vector Database Logging
# ============================================================================

import logging
import os
import sys
import json
import datetime
from typing import Dict, Any
from threading import Lock
import shutil

# Module-level flag for clean state management
_logging_configured = False
_config_lock = Lock()

def auto_configure_logging():
    """
    Smart auto-configuration that handles the 3 key scenarios:
    1. Silent by default (minimal logging)
    2. Easy debugging when needed  
    3. Production-ready logging in deployments
    
    Also coordinates Python + Rust logging for consistency.
    """
    global _logging_configured
    
    with _config_lock:
        if _logging_configured:
            return
        
        # Respect disable flag early
        disabled = os.getenv('ZEUSDB_DISABLE_AUTO_LOGGING', '').lower() in ('true', '1', 'yes')
        
        # Always compute environment and config (needed for Rust coordination)
        env_context = _detect_environment()
        config = _get_smart_defaults(env_context)
        
        # CRITICAL: Always prep Rust so it initializes correctly when extension loads
        # This ensures consistent Python+Rust logging even in enterprise environments
        if not disabled:
            _configure_rust_logging(config)
        
        # Only attach Python handlers if we really should (respects enterprise setups)
        if not disabled and _should_configure_logging():
            _configure_python_logging(config)
        
        _logging_configured = True

def _detect_environment() -> str:
    """
    Detect the runtime environment to apply appropriate defaults.
    
    Returns:
        str: 'production', 'development', 'testing', 'jupyter', or 'ci'
    """
    # Check for explicit environment setting first
    env = os.getenv('ENVIRONMENT', '').lower()
    if env in ('prod', 'production'):
        return 'production'
    elif env in ('dev', 'development'):
        return 'development'
    elif env in ('test', 'testing'):
        return 'testing'
    
    # Auto-detect based on runtime indicators
    if os.getenv('PYTEST_CURRENT_TEST'):
        return 'testing'
    elif os.getenv('JUPYTER_SERVER_ROOT') or os.getenv('JPY_PARENT_PID'):
        return 'jupyter'
    elif os.getenv('CI') or os.getenv('GITHUB_ACTIONS') or os.getenv('GITLAB_CI'):
        return 'ci'
    elif os.getenv('KUBERNETES_SERVICE_HOST') or os.getenv('DOCKER_CONTAINER'):
        return 'production'
    elif 'pytest' in sys.modules:
        return 'testing'
    elif 'IPython' in sys.modules or 'get_ipython' in globals():
        return 'jupyter'
    else:
        # Default to development for interactive use
        return 'development'

def _get_smart_defaults(env_context: str) -> Dict[str, Any]:
    """
    Get smart defaults based on environment context.
    Environment variables always override defaults.
    """
    # Base configuration for each environment
    defaults = {
        'production': {
            'level': 'ERROR',        # 1. Silent by default in production
            'format': 'json',        # 3. Production-ready structured logs
            'include_timestamp': True,
            'include_process_info': True,
            'console_output': False,  # Typically log to file/external systems
        },
        'development': {
            'level': 'WARNING',      # 1. Mostly silent, but show important issues
            'format': 'human',       # 2. Easy debugging - human readable
            'include_timestamp': True,
            'include_process_info': False,
            'console_output': True,
        },
        'testing': {
            'level': 'CRITICAL',     # 1. Very silent during tests
            'format': 'human',
            'include_timestamp': False,
            'include_process_info': False,
            'console_output': False,  # Don't pollute test output
        },
        'jupyter': {
            'level': 'INFO',         # 2. Easy debugging in notebooks
            'format': 'human',       # 2. Readable for exploration
            'include_timestamp': False,  # Cleaner in notebooks
            'include_process_info': False,
            'console_output': True,
        },
        'ci': {
            'level': 'WARNING',      # Show issues but not too verbose
            'format': 'human',       # Readable in CI logs
            'include_timestamp': True,
            'include_process_info': False,
            'console_output': True,
        }
    }
    
    config = defaults.get(env_context, defaults['development']).copy()
    
    # Environment variable overrides with validation
    env_level = os.getenv('ZEUSDB_LOG_LEVEL', config['level']).upper()
    
    # ENHANCEMENT: Validate log level
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'TRACE'}
    if env_level not in valid_levels:
        # Fallback to default - we can't log the warning yet since logging isn't configured
        # But we can store it for later logging once the system is set up
        config['_invalid_log_level'] = env_level
        env_level = config['level']
    
    # Handle TRACE level: Keep it for Rust, but use DEBUG for Python (since Python doesn't have TRACE)
    if env_level == 'TRACE':
        config['level'] = 'DEBUG'  # Python side uses DEBUG
        config['_rust_level'] = 'TRACE'  # Store original for Rust
    else:
        config['level'] = env_level
    
    # Validate log format
    env_format = os.getenv('ZEUSDB_LOG_FORMAT', config['format']).lower()
    valid_formats = {'human', 'json'}
    if env_format not in valid_formats:
        config['_invalid_log_format'] = env_format
        env_format = config['format']
    config['format'] = env_format
    
    config['log_file'] = os.getenv('ZEUSDB_LOG_FILE')
    
    # Special handling for boolean environment variables with validation
    console_env = os.getenv('ZEUSDB_LOG_CONSOLE')
    if console_env is not None:
        console_value = console_env.lower()
        if console_value in ('true', '1', 'yes', 'on'):
            config['console_output'] = True
        elif console_value in ('false', '0', 'no', 'off'):
            config['console_output'] = False
        else:
            # Invalid boolean value
            config['_invalid_console_setting'] = console_env
            # Keep default value
    
    return config


def _should_configure_logging() -> bool:
    """
    Determine if we should configure logging or respect existing setup.
    
    Returns:
        bool: True if we should configure, False if existing setup should be preserved
    """
    # Check if explicit disable flag is set
    if os.getenv('ZEUSDB_DISABLE_AUTO_LOGGING', '').lower() in ('true', '1', 'yes'):
        return False
    
    # Check if root logger already has handlers (enterprise setup)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return False
    
    # Check if any zeusdb loggers already configured
    zeusdb_logger = logging.getLogger('zeusdb')
    if zeusdb_logger.handlers:
        return False
    
    return True

def _configure_rust_logging(config: Dict[str, Any]) -> None:
    """
    Configure Rust tracing by setting environment variables.
    
    This ensures Rust and Python use consistent logging levels and formats.
    MUST be called BEFORE any Rust imports.
    """
    
    # Map Python log levels to Rust tracing levels (including TRACE support)
    py2rs = {
        "CRITICAL": "error",
        "ERROR": "error", 
        "WARNING": "warn",
        "INFO": "info",
        "DEBUG": "debug",
        "TRACE": "trace"  # Support TRACE level for Rust
    }
    
    # Use stored Rust level if available (for TRACE support), otherwise map from Python level
    if '_rust_level' in config:
        rust_level = config['_rust_level'].lower()
    else:
        rust_level = py2rs.get(config['level'], "warn")
    os.environ.setdefault("ZEUSDB_LOG_LEVEL", rust_level)
    os.environ.setdefault("ZEUSDB_LOG_FORMAT", config['format'])
    
    # Configure Rust log destination to match Python
    if config.get('log_file'):
        os.environ.setdefault("ZEUSDB_LOG_TARGET", "file")
        os.environ.setdefault("ZEUSDB_LOG_FILE", config['log_file'])
    else:
        os.environ.setdefault("ZEUSDB_LOG_TARGET", "stderr")



def _configure_python_logging(config: Dict[str, Any]):
    """Configure Python logging with immediate duplicate prevention and enhanced error handling."""
    
    # Get our hierarchical logger
    logger = logging.getLogger('zeusdb.vector')
    logger.setLevel(getattr(logging, config['level']))
    
    # Prevent duplicate logs immediately when we add handlers
    logger.propagate = False
    logger.handlers.clear()  # Avoid dupes if init runs twice
    
    # Create appropriate handlers
    handlers = []
    
    # Console handler (if enabled)
    if config['console_output']:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, config['level']))
        handlers.append(console_handler)
    
    # File handler (if specified) with enhanced error context and logging
    if config.get('log_file'):
        log_file_path = config['log_file']
        log_dir = os.path.dirname(log_file_path)
        
        try:
            # Ensure directory exists
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(getattr(logging, config['level']))
            handlers.append(file_handler)
            
        except (OSError, PermissionError) as e:
            # ENHANCEMENT: Gather detailed error context
            error_context = {
                'attempted_path': log_file_path,
                'absolute_path': os.path.abspath(log_file_path),
                'parent_dir': log_dir if log_dir else os.getcwd(),
                'parent_dir_exists': os.path.exists(log_dir) if log_dir else True,
                'parent_dir_writable': None,
                'disk_space_available': None,
                'error_type': type(e).__name__,
                'error_code': getattr(e, 'errno', None)
            }
            
            # Check parent directory permissions if it exists
            if log_dir and os.path.exists(log_dir):
                try:
                    error_context['parent_dir_writable'] = os.access(log_dir, os.W_OK)
                except OSError:
                    error_context['parent_dir_writable'] = False
            
            # Check available disk space (cross-platform)
            try:
                target_dir = log_dir if log_dir and os.path.exists(log_dir) else os.getcwd()
                usage = shutil.disk_usage(target_dir)
                error_context['disk_space_available'] = usage.free
            except Exception:
                pass  # Disk space check failed, but that's not critical
            
            # Fallback to console if file creation fails
            if not config['console_output']:
                # No console output configured, so we need to create an emergency console handler
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setLevel(logging.ERROR)  # Only errors for fallback
                console_handler.setFormatter(_create_formatter(config))
                logger.addHandler(console_handler)
                
                # Log the detailed error through the emergency handler
                logger.error(
                    "Failed to create log file '%s': %s. Error context: parent_dir='%s', "
                    "parent_exists=%s, parent_writable=%s, disk_space=%s, error_code=%s",
                    error_context['attempted_path'],
                    str(e),
                    error_context['parent_dir'],
                    error_context['parent_dir_exists'],
                    error_context['parent_dir_writable'],
                    f"{error_context['disk_space_available']:,} bytes" if error_context['disk_space_available'] else "unknown",
                    error_context['error_code']
                )
                return  # Early return since we've set up the fallback handler
            else:
                # Console output was already enabled, store error context for later logging
                config['_file_handler_error'] = {
                    'exception': str(e),
                    'context': error_context
                }
    
    # Configure formatters and attach handlers
    for handler in handlers:
        formatter = _create_formatter(config)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Log any validation warnings or file errors that occurred during setup
    _log_setup_warnings(logger, config)
    
    # Configure warning capture for development environments
    if config['level'] in ('DEBUG', 'INFO') and config['console_output']:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger('py.warnings')
        warnings_logger.setLevel(logging.WARNING)


def _log_setup_warnings(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """Log any warnings or errors that occurred during configuration setup."""
    
    # Log invalid environment variable warnings
    if config.get('_invalid_log_level'):
        logger.warning(
            "Invalid ZEUSDB_LOG_LEVEL='%s'. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE. Using default: %s",
            config['_invalid_log_level'], config['level']
        )
    
    if config.get('_invalid_log_format'):
        logger.warning(
            "Invalid ZEUSDB_LOG_FORMAT='%s'. Valid formats: human, json. Using default: %s",
            config['_invalid_log_format'], config['format']
        )
    
    if config.get('_invalid_console_setting'):
        logger.warning(
            "Invalid ZEUSDB_LOG_CONSOLE='%s'. Valid values: true/false, 1/0, yes/no, on/off. Using default: %s",
            config['_invalid_console_setting'], config['console_output']
        )
    
    # Log file handler error with detailed context (if console output was already enabled)
    if config.get('_file_handler_error'):
        error_info = config['_file_handler_error']
        context = error_info['context']
        
        logger.error(
            "Could not create log file '%s': %s. Using console logging only. "
            "Diagnostic info - Parent dir: '%s' (exists: %s, writable: %s), "
            "Available space: %s, Error code: %s",
            context['attempted_path'],
            error_info['exception'],
            context['parent_dir'],
            context['parent_dir_exists'],
            context['parent_dir_writable'],
            f"{context['disk_space_available']:,} bytes" if context['disk_space_available'] else "unknown",
            context['error_code']
        )


def _create_formatter(config: Dict[str, Any]) -> logging.Formatter:
    """Create appropriate formatter based on configuration."""
    
    if config['format'] == 'json':
        return JSONFormatter(
            include_timestamp=config['include_timestamp'],
            include_process_info=config['include_process_info']
        )
    else:
        # Human-readable format
        format_parts = []
        
        if config['include_timestamp']:
            format_parts.append('%(asctime)s')
        
        if config['include_process_info']:
            format_parts.append('%(process)d')
        
        format_parts.extend(['%(name)s', '%(levelname)s', '%(message)s'])
        
        format_string = ' - '.join(format_parts)
        
        return logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

class JSONFormatter(logging.Formatter):
    """JSON formatter for production logging with proper timestamp handling."""
    
    def __init__(self, include_timestamp=True, include_process_info=True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_process_info = include_process_info
    
    def formatTime(self, record, datefmt=None):
        """Override to get proper microsecond precision (time.strftime ignores %f)."""
        dt = datetime.datetime.utcfromtimestamp(record.created)
        return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond:06d}Z"
    
    def format(self, record):
        log_entry = {
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if self.include_timestamp:
            # Use the same key Rust emits in tracing JSON
            log_entry['timestamp'] = self.formatTime(record)  # RFC 3339 string
            # Optional: keep epoch seconds under a different name if you still want it
            log_entry['timestamp_epoch'] = record.created
        
        if self.include_process_info:
            log_entry['process'] = record.process
            log_entry['thread'] = record.thread
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info'):
                log_entry[key] = value
        
        return json.dumps(log_entry, separators=(',', ':'))

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for ZeusDB components.
    
    Args:
        name: Optional name suffix (e.g., 'VectorDatabase')
              If None, returns the base zeusdb.vector logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if name:
        return logging.getLogger(f'zeusdb.vector.{name}')
    else:
        return logging.getLogger('zeusdb.vector')

# ============================================================================
# ENVIRONMENT VARIABLE DOCUMENTATION
# ============================================================================

"""
SUPPORTED ENVIRONMENT VARIABLES:

ZEUSDB_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL
    Default: WARNING (development), ERROR (production), CRITICAL (testing)

ZEUSDB_LOG_FORMAT: human, json  
    Default: human (development/testing), json (production)

ZEUSDB_LOG_FILE: /path/to/logfile.log
    Default: None (console only)

ZEUSDB_LOG_CONSOLE: true, false
    Default: true (development), false (production)

ZEUSDB_DISABLE_AUTO_LOGGING: true, false
    Default: false (set to true to disable all auto-configuration)

ENVIRONMENT: production, development, testing
    Used for auto-detection if set explicitly

USAGE EXAMPLES:

# Silent by default
python app.py

# Easy debugging  
ZEUSDB_LOG_LEVEL=DEBUG python app.py

# Production ready
ENVIRONMENT=production ZEUSDB_LOG_FORMAT=json ZEUSDB_LOG_FILE=/var/log/app.log python app.py

# Disable auto-configuration (use your own logging setup)
ZEUSDB_DISABLE_AUTO_LOGGING=true python app.py
"""