"""
vector_database.py

Pure factory for creating vector indexes using Rust backend.
Currently supports HNSW (Hierarchical Navigable Small World).
"""
from .zeusdb_vector_database import HNSWIndex

class VectorDatabase:
    """
    Pure factory for creating vector indexes.
    No state management - just creates and returns indexes.
    """
    
    def __init__(self):
        """Initialize VectorDatabase factory."""
        pass

    def create_index_hnsw(
        self,
        dim: int = 1536,
        space: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        expected_size: int = 10000
    ) -> HNSWIndex:
        """
        Creates a new HNSW (Hierarchical Navigable Small World) index.

        Args:
            dim: Vector dimension (default: 1536)
            space: Distance metric, only 'cosine' supported (default: 'cosine')
            M: Bidirectional links per node (default: 16, max: 256)
            ef_construction: Construction candidate list size (default: 200)
            expected_size: Expected number of vectors (default: 10000)

        Returns:
            HNSWIndex: Use this index directly for all operations

        Example:
            vdb = VectorDatabase()
            index = vdb.create_index_hnsw(dim=1536, expected_size=10000)
            index.add_point("doc1", vector, metadata)
            results = index.query(query_vector, k=10)
        """
        try:
            return HNSWIndex(dim, space, M, ef_construction, expected_size)
        except Exception as e:
            raise RuntimeError(f"Failed to create HNSW index: {e}") from e
    