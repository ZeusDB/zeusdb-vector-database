# ============================================================================
# zeusdb_vector_database/__init__.py
# ============================================================================

"""
ZeusDB Vector Database Module
"""
__version__ = "0.3.1"

# STEP 1: Configure logging FIRST, before importing anything that uses Rust.
from .logging_config import auto_configure_logging as _auto_configure_logging
_auto_configure_logging()  # Sets env vars for Rust BEFORE the PyO3 module is imported

# Step 2: THEN import the Python shim that pulls in the Rust extension.
# imports the VectorDatabase class from the vector_database.py file
from .vector_database import VectorDatabase # noqa: E402

__all__ = ["VectorDatabase"]