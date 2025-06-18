from .create_index_hnsw import HNSWIndex, create_index_hnsw

class VectorDatabase:
    def __init__(self):
        self.index = None

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
        """
        return create_index_hnsw(dim, space, M, ef_construction, expected_size)


