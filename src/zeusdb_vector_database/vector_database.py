"""
VectorDatabase - Python wrapper for Rust implementation
"""
# Import the Rust implementation directly
from .zeusdb_vector_database import HNSWIndex

class VectorDatabase:
    """
    VectorDatabase class - thin Python wrapper around Rust implementation.
    All heavy computation is done in Rust for performance.
    """
    def __init__(self):
        """Initialize VectorDatabase"""
        self.index: HNSWIndex | None = None
        self.index_type: str | None = None  # Tracks index backend used
        self._stats = {"total_vectors": 0, "total_queries": 0}

    def create_index_hnsw(
            self, 
            dim: int = 1536, 
            space: str = "cosine", 
            M: int = 16, 
            ef_construction: int = 200,
            expected_size: int = 10000  # Default capacity
            ) -> HNSWIndex:
        """
        Creates a new HNSW (Hierarchical Navigable Small World) index using the specified configuration.

        This method initializes the index for approximate nearest neighbor search using the HNSW algorithm.
        It supports configuration of vector dimension, distance metric, connectivity, and construction parameters.

        Args:
            dim (int): The number of dimensions for each vector in the index (default is 1536).
            space (str): The distance metric to use for similarity, currently only 'cosine' is supported.
            M (int): The number of bidirectional links each node maintains in the graph (higher = more accuracy).
            ef_construction (int): Size of the dynamic candidate list during index construction (higher = better recall).
            expected_size (int): Estimated number of vectors to store; used to preallocate internal data structures (default is 10,000).

        Returns:
            HNSWIndex: An initialized HNSWIndex object ready for vector insertion and similarity search.

        Raises:
            ValueError: If an unsupported distance metric is given.
            RuntimeError: If index creation fails internally.
        """
        #if space not in {"cosine", "l2", "dot"}:
        if space not in {"cosine"}:
            raise ValueError(f"Unsupported space: {space}")
        if M > 256:
            raise ValueError("M (max_nb_connection) must be less than or equal to 256")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be positive")
        if expected_size <= 0:
            raise ValueError("expected_size must be positive")

        try:
            index = HNSWIndex(dim, space, M, ef_construction, expected_size)
            self.index = index
            self.index_type = "hnsw"
            return index
        except Exception as e:
            raise RuntimeError(
                f"Failed to create HNSW index with dim={dim}, space={space}, "
                f"M={M}, ef={ef_construction}, expected_size={expected_size}"
            ) from e



