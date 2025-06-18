from .zeusdb_vector_database import HNSWIndex

def create_index_hnsw(dim: int, space: str, M: int, ef_construction: int, expected_size: int) -> HNSWIndex:
    """
    Create a new HNSW (Hierarchical Navigable Small World) index using the Rust backend with expected capacity.

    Args:
        dim (int): Dimension of the vectors to be indexed.
        space (str): Distance metric to use. Only 'cosine' is currently supported.
        M (int): Number of bi-directional links created for every new element.
        ef_construction (int): Size of the dynamic list for the nearest neighbors during index construction.

    Returns:
        HNSWIndex: An instance of the HNSWIndex class representing the created index.

    Raises:
        ValueError: If an unsupported distance metric is provided.
    """
    #if space not in {"cosine", "l2", "dot"}:
    if space not in {"cosine"}:
        raise ValueError(f"Unsupported space: {space}")
    if M > 256:
        raise ValueError("M (max_nb_connection) must be less than or equal to 256")
    return HNSWIndex(dim, space, M, ef_construction, expected_size)
