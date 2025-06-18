#import pytest
from zeusdb_vector_database import VectorDatabase

# ------------------------------------------------------------
# Test 1
# Test the creation of an HNSW index with default parameters.
# ------------------------------------------------------------
def test_create_index_hnsw():
    vdb = VectorDatabase()
    result = vdb.hnsw()
    assert result == "HNSW index created."
