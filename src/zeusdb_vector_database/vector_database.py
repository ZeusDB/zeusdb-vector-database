"""
vector_database.py

Factory for creating vector indexes with support for multiple types and quantization.
Currently supports HNSW (Hierarchical Navigable Small World) with extensible design.
"""
from typing import Dict, Any, Optional, TypedDict
from .zeusdb_vector_database import _create_hnsw_index
# Future index types are registered in _index_types and dispatched in _build_index.

class _MemoryInfo(TypedDict):
    """Type definition for quantization memory information.

    Private. The only consumer is _check_memory_usage, and the value it
    produces is stripped from the config before it reaches Rust.
    """
    centroid_storage_mb: float
    compression_ratio: float
    centroids_per_subvector: int
    total_centroids: int
    calculated_training_size: int

class VectorDatabase:
    """
    Factory for creating various types of vector indexes with optional quantization.
    Each index type is registered in _index_types and built in _build_index.
    """

    # Index type name to a one line description. Deliberately not a mapping to
    # a constructor. A registry that held one would hand a user a way to build
    # an index without the defaults create() applies, which is the second
    # construction path this factory exists to prevent.
    _index_types: Dict[str, str] = {
        "hnsw": "Hierarchical Navigable Small World graph",
        # "ivf": "Inverted file index",    # Future support planned
        # "lsh": "Locality sensitive hashing",  # Future support planned
    }

    @staticmethod
    def _build_index(index_type: str, **kwargs) -> Any:
        """Dispatch to the extension factory for the requested index type."""
        if index_type == "hnsw":
            return _create_hnsw_index(**kwargs)
        raise ValueError(f"No builder registered for index type '{index_type}'")

    @staticmethod
    def _default_m(expected_size: Any) -> int:
        """Graph degree for an index of this declared size.

        A fixed m of 16 was adequate up to about 25,000 records and clearly
        inadequate above it. On clustered 768 dimensional data at the default
        search width, recall at 10 was 1.0000 at 10,000 records, 0.9475 at
        50,000 and 0.8025 at 100,000. At m 32 the same three cells are 1.0000,
        0.9985 and 0.9870, and no search width recovers what m 16 loses.
        Raising m is not free, so the ladder only pays where it buys something:
        at 10,000 records m 16 already returns 1.0000 and m 32 would cost 1.10x
        the memory and 1.30x the build time for nothing.

        The ladder stops at 32 because that is where the measurements stop. At
        100,000 records m 64 bought 0.0060 recall over m 32 for 1.55x the build
        time, and it raises the capacity the graph reserves at creation from
        3,787 to 4,542 bytes per declared record, which matters most to exactly
        the large indexes a third step would be aimed at. An index well past
        250,000 records may want 64, but nothing here measured that, so it is
        left to the caller rather than guessed at.

        expected_size is a declaration rather than a limit, so understating it
        leaves the graph under-provisioned for the data that actually arrives.
        m is fixed at construction and no later add() revises it.

        A non-integer expected_size returns 16 so that the Rust layer reports
        the real validation error instead of this function raising a worse one.
        """
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            return 16
        return 32 if expected_size > 25_000 else 16


    def __init__(self):
        """Initialize the vector database factory."""
        pass

    def create(self, index_type: str = "hnsw", quantization_config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Create a vector index of the specified type with optional quantization.

        Args:
            index_type: The type of index to create (case-insensitive: "hnsw", "ivf", etc.)
            quantization_config: Optional quantization configuration dictionary
            **kwargs: Parameters specific to the chosen index type (validated by Rust backend)

            For "hnsw", supported parameters are:
                - dim (int): Vector dimension (default: 1536)
                - space (str): Distance metric — supports 'cosine', 'l2', or 'l1' (default: 'cosine')
                - m (int): Bidirectional links per node (min: 1, max: 256). Defaults
                  to 16 up to an expected_size of 25,000 and 32 above it. A graph
                  too sparse for the record count loses recall that no search
                  width recovers, and m is fixed at construction, so set
                  expected_size honestly or set m directly.
                - ef_construction (int): Construction candidate list size (default: 200)
                - expected_size (int): Expected number of vectors (default: 10000).
                  Also selects the default m, see above.

            Quantization config format:
                {
                    'type': 'pq',              # Currently only 'pq' (Product Quantization) supported
                    'subvectors': 8,           # Number of subvectors (must divide dim evenly, default: 8)
                    'bits': 8,                 # Bits per subvector (1-8, controls centroids, default: 8)
                    'training_size': None,     # Auto-calculated based on subvectors & bits (or specify manually)
                    'max_training_vectors': None,  # Optional limit on training vectors used
                    'storage_mode': 'quantized_only' # Storage mode for quantized vectors (or 'quantized_with_raw')  
                }

            Note: Quantization reduces memory usage (typically 4-32x compression) but may 
            slightly degrade recall accuracy. Training triggers automatically on the first 
            .add() call that reaches the training_size threshold.

        Returns:
            An instance of the created vector index.

        Examples:
            # HNSW index with defaults (no quantization)
            vdb = VectorDatabase()
            index = vdb.create("hnsw", dim=1536)
            
            # HNSW index with Product Quantization (auto-calculated training size)
            quantization_config = {
                'type': 'pq',
                'subvectors': 8,
                'bits': 8
            }
            index = vdb.create(
                index_type="hnsw", 
                dim=1536, 
                quantization_config=quantization_config
            )
            
            # Memory-optimized configuration with manual training size
            memory_optimized_config = {
                'type': 'pq',
                'subvectors': 16,         # More subvectors = better compression
                'bits': 6,                # Fewer bits = less memory per centroid
                'training_size': 75000,    # Override auto-calculation
                'storage_mode': 'quantized_only'  # Only store quantized vectors
            }
            index = vdb.create(
                index_type="hnsw",
                dim=1536,
                quantization_config=memory_optimized_config,
                expected_size=1000000     # Large dataset
            )

        Raises:
            ValueError: If index_type is not supported or quantization config is invalid.
            RuntimeError: If index creation fails due to backend validation.
        """
        index_type = (index_type or "").strip().lower()

        if index_type not in self._index_types:
            available = ', '.join(sorted(self._index_types.keys()))
            raise ValueError(f"Unknown index type '{index_type}'. Available: {available}")
        
        # Centralize dim early to ensure consistency
        dim = kwargs.get('dim', 1536)
        
        # Validate and process quantization config
        if quantization_config is not None:
            quantization_config = self._validate_quantization_config(quantization_config, dim)
        
        # Apply index-specific defaults
        if index_type == "hnsw":
            kwargs.setdefault("dim", dim)
            kwargs.setdefault("space", "cosine")
            kwargs.setdefault("ef_construction", 200)
            kwargs.setdefault("expected_size", 10000)
            # After expected_size, since the graph degree is derived from it.
            kwargs.setdefault("m", self._default_m(kwargs["expected_size"]))
        
        try:
            # Always pass quantization_config parameter
            if quantization_config is not None:
                # Remove keys with None values and internal keys
                clean_config = {k: v for k, v in quantization_config.items() if not k.startswith('_') and v is not None}
            else:
                clean_config = None

            return self._build_index(index_type, quantization_config=clean_config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create {index_type.upper()} index: {e}") from e


    def load(self, path: str) -> Any:
        """
        Load a previously saved ZeusDB index from disk.
    
        Args:
            path: Path to the .zdb directory containing the saved index
        
        Returns:
            HNSWIndex: The loaded index ready for use
        
        Example:
            >>> vdb = VectorDatabase()
            >>> loaded_index = vdb.load("my_index.zdb")
            >>> results = loaded_index.search(query_vector, top_k=5)
        """
        from .zeusdb_vector_database import _load_index  # Direct function import
        return _load_index(path)


    def _validate_quantization_config(self, config: Dict[str, Any], dim: int) -> Dict[str, Any]:
        """
        Validate and normalize quantization configuration.
        
        Args:
            config: Raw quantization configuration
            dim: Vector dimension for validation
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("quantization_config must be a dictionary")
        
        # Create a copy to avoid modifying the original
        validated_config = config.copy()
        
        # Validate quantization type
        qtype = validated_config.get('type', '').lower()
        if qtype != 'pq':
            raise ValueError(f"Unsupported quantization type: '{qtype}'. Currently only 'pq' is supported.")
        
        validated_config['type'] = 'pq'
        
        # Validate subvectors
        subvectors = validated_config.get('subvectors', 8)
        if not isinstance(subvectors, int) or subvectors <= 0:
            raise ValueError(f"subvectors must be a positive integer, got {subvectors}")
        
        if dim % subvectors != 0:
            raise ValueError(
                f"subvectors ({subvectors}) must divide dimension ({dim}) evenly. "
                f"Consider using subvectors: {', '.join(map(str, self._suggest_subvector_divisors(dim)))}"
            )
        
        if subvectors > dim:
            raise ValueError(f"subvectors ({subvectors}) cannot exceed dimension ({dim})")
        
        validated_config['subvectors'] = subvectors
        
        # Validate bits per subvector
        bits = validated_config.get('bits', 8)
        if not isinstance(bits, int) or bits < 1 or bits > 8:
            raise ValueError(f"bits must be an integer between 1 and 8, got {bits}")
        
        validated_config['bits'] = bits
        
        # Calculate smart training size if not provided
        training_size = validated_config.get('training_size')
        if training_size is None:
            training_size = self._calculate_smart_training_size(subvectors, bits)
        else:
            if not isinstance(training_size, int) or training_size < 1000:
                raise ValueError(f"training_size must be at least 1000 for stable k-means clustering, got {training_size}")
        
        validated_config['training_size'] = training_size
        
        # Validate max training vectors if provided
        max_training_vectors = validated_config.get('max_training_vectors')
        if max_training_vectors is not None:
            if not isinstance(max_training_vectors, int) or max_training_vectors < training_size:
                raise ValueError(
                    f"max_training_vectors ({max_training_vectors}) must be >= training_size ({training_size})"
                )
            validated_config['max_training_vectors'] = max_training_vectors
        
        # Validate storage mode
        storage_mode = str(validated_config.get('storage_mode', 'quantized_only')).lower()
        valid_modes = {'quantized_only', 'quantized_with_raw'}
        if storage_mode not in valid_modes:
            raise ValueError(
                f"Invalid storage_mode: '{storage_mode}'. Supported modes: {', '.join(sorted(valid_modes))}"
            )
        
        validated_config['storage_mode'] = storage_mode

        # Calculate and warn about memory usage
        self._check_memory_usage(validated_config, dim)

        # Add helpful warnings about storage mode
        if storage_mode == 'quantized_with_raw':
            import warnings
            compression_ratio = validated_config.get('__memory_info__', {}).get('compression_ratio', 1.0)
            warnings.warn(
                f"storage_mode='quantized_with_raw' will use ~{compression_ratio:.1f}x more memory "
                f"than 'quantized_only' but enables exact vector reconstruction.",
                UserWarning,
                stacklevel=2
            )
        
        # Final safety check: ensure all expected keys are present
        # This is a final defensive programming - all the keys should already be set above, but added just in case
        validated_config.setdefault('type', 'pq')
        validated_config.setdefault('subvectors', 8)
        validated_config.setdefault('bits', 8)
        validated_config.setdefault('max_training_vectors', None)
        validated_config.setdefault('storage_mode', 'quantized_only')

        return validated_config

    def _calculate_smart_training_size(self, subvectors: int, bits: int) -> int:
        """
        Calculate optimal training size based on quantization parameters.
        
        Args:
            subvectors: Number of subvectors
            bits: Bits per subvector
            
        Returns:
            Recommended training size for stable k-means clustering
        """
        # Statistical requirement: need enough samples per centroid for stable clustering
        # Training is done per subvector, so we need (2^bits * min_samples) total
        centroids_per_subvector = 2 ** bits
        min_samples_per_centroid = 20  # Statistical guideline for k-means stability
        
        # Calculate minimum samples needed for stable clustering across all subvectors
        statistical_minimum = centroids_per_subvector * min_samples_per_centroid
        
        # Practical bounds
        reasonable_minimum = 10000    # Always need at least this for diversity
        reasonable_maximum = 200000   # Diminishing returns beyond this point
        
        return min(max(statistical_minimum, reasonable_minimum), reasonable_maximum)

    
    def _suggest_subvector_divisors(self, dim: int) -> list[int]:
        """Return valid subvector counts that divide the dimension evenly (up to 32)."""
        return [i for i in range(1, min(33, dim + 1)) if dim % i == 0]
    




    def _check_memory_usage(self, config: Dict[str, Any], dim: int) -> None:
        """
        Calculate and warn about memory usage for the quantization configuration.
        
        Args:
            config: Validated quantization configuration
            dim: Vector dimension
        """
        subvectors = config['subvectors']
        bits = config['bits']
        sub_dim = dim // subvectors
        
        # Calculate centroid storage requirements
        num_centroids_per_subvector = 2 ** bits
        total_centroids = subvectors * num_centroids_per_subvector
        centroid_memory_mb = (total_centroids * sub_dim * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Calculate compression ratio
        original_bytes_per_vector = dim * 4  # float32
        compressed_bytes_per_vector = subvectors  # 1 byte per subvector code
        compression_ratio = original_bytes_per_vector / compressed_bytes_per_vector
        
        # Add memory info to config for user reference (internal)
        memory_info: _MemoryInfo = {
            'centroid_storage_mb': round(centroid_memory_mb, 2),
            'compression_ratio': round(compression_ratio, 1),
            'centroids_per_subvector': num_centroids_per_subvector,
            'total_centroids': total_centroids,
            'calculated_training_size': config['training_size']
        }
        config['__memory_info__'] = memory_info        
        # Warn about large memory usage
        if centroid_memory_mb > 100:
            import warnings
            warnings.warn(
                f"Large centroid storage required: {centroid_memory_mb:.1f}MB. "
                f"Consider reducing bits ({bits}) or subvectors ({subvectors}) for memory efficiency.",
                UserWarning,
                stacklevel=2
            )
        
        # Warn about low compression
        if compression_ratio < 4:
            import warnings
            warnings.warn(
                f"Low compression ratio: {compression_ratio:.1f}x. "
                f"Consider increasing subvectors ({subvectors}) or reducing bits ({bits}) for better compression.",
                UserWarning,
                stacklevel=2
            )
        
        # Warn about extremely high compression
        if compression_ratio > 50:
            import warnings
            warnings.warn(
                f"Very high compression ratio: {compression_ratio:.1f}x may significantly impact recall quality. "
                f"Consider reducing subvectors ({subvectors}) or increasing bits ({bits}) for better accuracy.",
                UserWarning,
                stacklevel=2
            )

    @classmethod
    def available_index_types(cls) -> list[str]:
        """Return list of all supported index types."""
        return sorted(cls._index_types.keys())
    