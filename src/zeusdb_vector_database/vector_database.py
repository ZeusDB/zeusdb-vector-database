"""
VectorDatabase - Python wrapper for Rust implementation
"""
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

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
        self.index_type: str | None = None

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

        All validation is performed in Rust for maximum performance.

        Args:
            dim: Vector dimension (default: 1536)
            space: Distance metric, only 'cosine' supported (default: 'cosine')
            M: Bidirectional links per node (default: 16, max: 256)
            ef_construction: Construction candidate list size (default: 200)
            expected_size: Expected number of vectors (default: 10000)

        Returns:
            HNSWIndex: Initialized index ready for use

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If index creation fails
        """
        try:
            # All validation happens in Rust
            index = HNSWIndex(dim, space, M, ef_construction, expected_size)
            self.index = index
            self.index_type = "hnsw"
            return index
        except Exception as e:
            # Re-raise with additional context if needed
            raise RuntimeError(f"Failed to create HNSW index: {e}") from e

    # Convenience methods that delegate to the index
    def add_point(
        self, 
        id: str, 
        vector: Union[List[float], np.ndarray], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single vector (convenience method)."""
        if self.index is None:
            raise RuntimeError("No index created. Call create_index_hnsw() first.")
        
        # Convert numpy to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        # Convert metadata values to strings for Rust compatibility
        rust_metadata = None
        if metadata:
            rust_metadata = {k: str(v) for k, v in metadata.items()}
        
        return self.index.add_point(id, vector, rust_metadata)

    def add_batch(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple vectors in batch (convenience method)."""
        if self.index is None:
            raise RuntimeError("No index created. Call create_index_hnsw() first.")
        
        # Convert to Rust format
        rust_points = []
        for point in points:
            vector = point["vector"]
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            metadata = point.get("metadata")
            rust_metadata = None
            if metadata:
                rust_metadata = {k: str(v) for k, v in metadata.items()}
            
            rust_points.append((point["id"], vector, rust_metadata))
        
        return self.index.add_batch(rust_points)

    def query(
        self, 
        vector: Union[List[float], np.ndarray], 
        k: int = 10,
        filter: Optional[Dict[str, str]] = None,
        ef_search: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Query for similar vectors (convenience method)."""
        if self.index is None:
            raise RuntimeError("No index created. Call create_index_hnsw() first.")
        
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        return self.index.query(vector, filter, k, ef_search)

    def search_with_metadata(
        self,
        vector: Union[List[float], np.ndarray],
        k: int = 10,
        filter: Optional[Dict[str, str]] = None,
        ef_search: Optional[int] = None,
        include_metadata: bool = True
    ) -> List[Tuple[str, float, Optional[Dict[str, str]]]]:
        """Search with metadata included (convenience method)."""
        if self.index is None:
            raise RuntimeError("No index created. Call create_index_hnsw() first.")
        
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        return self.index.search_with_metadata(vector, filter, k, ef_search, include_metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        if self.index is None:
            return {"has_index": False, "index_type": None}
        
        stats = self.index.get_stats()
        # Convert string values back to appropriate types
        result = {
            "has_index": True,
            "index_type": self.index_type,
            "total_vectors": int(stats.get("total_vectors", "0")),
            "dimension": int(stats.get("dimension", "0")),
            "space": stats.get("space", "unknown"),
            "M": int(stats.get("M", "0")),
            "ef_construction": int(stats.get("ef_construction", "0")),
            "expected_size": int(stats.get("expected_size", "0"))
        }
        return result

    # Simple delegation methods
    def info(self) -> str:
        """Get index information."""
        if self.index is None:
            return "VectorDatabase(no index created)"
        return self.index.info()

    def list(self, number: int = 10) -> List[Tuple[str, Optional[Dict[str, str]]]]:
        """List vectors."""
        if self.index is None:
            raise RuntimeError("No index created. Call create_index_hnsw() first.")
        return self.index.list(number)

    def contains(self, id: str) -> bool:
        """Check if vector ID exists."""
        if self.index is None:
            return False
        return self.index.contains(id)

    def get_vector(self, id: str) -> Optional[List[float]]:
        """Get vector by ID."""
        if self.index is None:
            return None
        return self.index.get_vector(id)

    def get_vector_metadata(self, id: str) -> Optional[Dict[str, str]]:
        """Get vector metadata by ID."""
        if self.index is None:
            return None
        return self.index.get_vector_metadata(id)
    
    def remove_point(self, id: str) -> bool:
        """Remove a point by ID."""
        if self.index is None:
            raise RuntimeError("No index created. Call create_index_hnsw() first.")
        return self.index.remove_point(id)