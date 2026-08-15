# ============================================================================
# zeusdb_vector_database/__init__.py
# ============================================================================

"""
ZeusDB Vector Database Module
"""
__version__ = "0.6.0"

# STEP 1: Configure logging FIRST, before importing anything that uses Rust.
from .logging_config import _auto_configure_logging
_auto_configure_logging()  # Sets env vars for Rust BEFORE the PyO3 module is imported

# Step 2: THEN import the Python shim that pulls in the Rust extension.
# imports the VectorDatabase class from the vector_database.py file
from .vector_database import VectorDatabase # noqa: E402

# Step 3: Re-export the types a caller receives and the logging controls.
#
# HNSWIndex and AddResult are exported so a caller can name the types create()
# and add() hand back, in an isinstance check or a return annotation. Neither
# can be constructed directly; an index comes from VectorDatabase.create or
# VectorDatabase.load.
#
# The three logging functions are exported because the documented programmatic
# logging recipe calls them at package level and there was nothing here to
# resolve. Their names come from the #[pyfunction(name = ...)] attributes in
# vdb-core/src/logging.rs, not from the Rust function names.
from .zeusdb_vector_database import (  # noqa: E402
    AddResult,
    HNSWIndex,
    init_file_logging,
    init_logging,
    is_logging_initialized,
)

__all__ = [
    "AddResult",
    "HNSWIndex",
    "VectorDatabase",
    "__version__",
    "init_file_logging",
    "init_logging",
    "is_logging_initialized",
]
