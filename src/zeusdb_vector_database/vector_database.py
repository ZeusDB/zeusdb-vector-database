"""
vector_database.py

Factory for creating vector indexes with support for multiple types.
Currently supports HNSW (Hierarchical Navigable Small World) with extensible design.
"""
from typing import Callable, Dict, Any
from .zeusdb_vector_database import HNSWIndex
# from .zeusdb_vector_database import HNSWIndex, IVFIndex, LSHIndex, AnnoyIndex, FlatIndex # Future support planned

class VectorDatabase:
    """
    Factory for creating various types of vector indexes.
    Each index type is registered via _index_constructors.
    """

    _index_constructors: Dict[str, Callable[..., Any]] = {
        "hnsw": HNSWIndex,
        # "ivf": IVFIndex,      # Future support planned
        # "lsh": LSHIndex,      # Future support planned
        # "annoy": AnnoyIndex,  # Future support planned
        # "flat": FlatIndex,    # Future support planned
    }
    
    def __init__(self):
        """Initialize the vector database factory."""
        pass

    def create(self, index_type: str = "hnsw", **kwargs) -> Any:
        """
        Create a vector index of the specified type.

        Args:
            index_type: The type of index to create (case-insensitive: "hnsw", "ivf", etc.)
            **kwargs: Parameters specific to the chosen index type (validated by Rust backend)

            For "hnsw", supported parameters are:
                - dim (int): Vector dimension (default: 1536)
                - space (str): Distance metric — supports 'cosine', 'l2', or 'l1' (default: 'cosine')
                - m (int): Bidirectional links per node (default: 16, max: 256)
                - ef_construction (int): Construction candidate list size (default: 200)
                - expected_size (int): Expected number of vectors (default: 10000)

        Returns:
            An instance of the created vector index.

        Examples:
            # HNSW index with defaults
            vdb = VectorDatabase()
            index = vdb.create("hnsw", dim=1536)
            
            # HNSW index with custom parameters
            index = vdb.create("hnsw", dim=768, m=16, ef_construction=200, space="cosine", expected_size=10000)
            
            # Future IVF index
            # index = vdb.create("ivf", dim=1536, nlist=100, nprobe=10)

        Raises:
            ValueError: If index_type is not supported.
            RuntimeError: If index creation fails due to backend validation.
        """
        index_type = (index_type or "").strip().lower()

        if index_type not in self._index_constructors:
            available = ', '.join(sorted(self._index_constructors.keys()))
            raise ValueError(f"Unknown index type '{index_type}'. Available: {available}")
        
        # Apply index-specific defaults
        if index_type == "hnsw":
            kwargs.setdefault("dim", 1536)
            kwargs.setdefault("space", "cosine")
            kwargs.setdefault("m", 16)
            kwargs.setdefault("ef_construction", 200)
            kwargs.setdefault("expected_size", 10000)
        
        constructor = self._index_constructors[index_type]
        
        try:
            return constructor(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create {index_type.upper()} index: {e}") from e
            
    @classmethod
    def available_index_types(cls) -> list[str]:
        """Return list of all supported index types."""
        return sorted(cls._index_constructors.keys())
