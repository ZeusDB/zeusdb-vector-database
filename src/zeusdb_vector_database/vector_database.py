"""
vector_database.py

Factory for creating vector indexes with support for multiple types and quantization.
Currently supports HNSW (Hierarchical Navigable Small World) with extensible design.
"""
from typing import Dict, Any, Optional
from .zeusdb_vector_database import _create_hnsw_index
# Future index types are registered in _index_types and dispatched in _build_index.

# A _MemoryInfo TypedDict used to sit here, describing a __memory_info__ entry
# _check_memory_usage wrote into the config. Its only reader took the
# compression ratio out of it and printed it as a memory multiplier, which it
# is not. With that reader gone the entry was written and never read, and
# create() stripped it before the config reached Rust, so it is gone too.

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
                - space (str): Distance metric, one of 'cosine', 'l2', or 'l1' (default: 'cosine')
                - m (int): Bidirectional links per node (min: 2, max: 256). Defaults
                  to 16 up to an expected_size of 25,000 and 32 above it. A graph
                  too sparse for the record count loses recall that no search
                  width recovers, and m is fixed at construction, so set
                  expected_size honestly or set m directly. The floor is 2
                  because at 1 the layer scale is infinite and the graph is
                  degenerate.
                - ef_construction (int): Construction candidate list size (default: 200)
                - expected_size (int): Expected number of vectors (default: 10000,
                  max: 100,000,000). Also selects the default m, see above. It is
                  a capacity hint rather than a limit, and an index that grows
                  past twice its declaration logs a warning once.

            Quantization config format:
                {
                    'type': 'pq',              # Currently only 'pq' (Product Quantization) supported
                    'subvectors': 8,           # Number of subvectors (must divide dim evenly, default: 8)
                    'bits': 8,                 # Bits per subvector (1-8, controls centroids, default: 8)
                    'training_size': None,     # Auto-calculated based on subvectors & bits (or specify manually)
                    'max_training_vectors': None,  # Optional limit on training vectors used
                    'storage_mode': 'quantized_only' # Storage mode for quantized vectors (or 'quantized_with_raw')  
                }

            Note: Quantization replaces each vector with a code of one byte per
            subvector, so the code is dim * 4 / subvectors smaller than the vector.
            The memory an index saves is smaller than that, because the codebook,
            the centroid distance table, the graph and the training records kept at
            full width are all unaffected. get_stats() reports the actual figures.
            Recall falls sharply, not slightly. Only 'quantized_with_raw' can
            recover it, by reranking against the raw vectors it keeps. Training
            triggers automatically on the first .add() call that reaches the
            training_size threshold.

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
            
            # Accuracy-weighted configuration with manual training size
            longer_code_config = {
                'type': 'pq',
                'subvectors': 16,         # More subvectors = longer code, lower ratio, better accuracy
                'bits': 6,                # Fewer bits = fewer centroids, smaller codebook and table
                'training_size': 75000,    # Override auto-calculation
                'storage_mode': 'quantized_only'  # Drop raw vectors once training completes
            }
            index = vdb.create(
                index_type="hnsw",
                dim=1536,
                quantization_config=longer_code_config,
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
        
        # Apply index-specific defaults
        #
        # Before the quantization validation rather than after it, because that
        # validation now needs expected_size to decide whether the fixed memory
        # quantization costs can be repaid. The default of 10,000 has to be in
        # place by then, since an unset expected_size is the case where the
        # answer is most often no.
        if index_type == "hnsw":
            kwargs.setdefault("dim", dim)
            kwargs.setdefault("space", "cosine")
            kwargs.setdefault("ef_construction", 200)
            kwargs.setdefault("expected_size", 10000)
            # After expected_size, since the graph degree is derived from it.
            kwargs.setdefault("m", self._default_m(kwargs["expected_size"]))

        # Validate and process quantization config
        if quantization_config is not None:
            quantization_config = self._validate_quantization_config(
                quantization_config, dim, kwargs.get("expected_size")
            )

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


    def _validate_quantization_config(self, config: Dict[str, Any], dim: int,
                                      expected_size: Any = None) -> Dict[str, Any]:
        """
        Validate and normalize quantization configuration.

        Args:
            config: Raw quantization configuration
            dim: Vector dimension for validation
            expected_size: The declared record count, used only to decide
                whether the fixed memory quantization costs can be repaid.
                None or a non-integer skips that check and leaves the Rust
                layer to report the real validation error.

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
        self._check_memory_usage(validated_config, dim, expected_size)

        # Warn about the memory cost of keeping raw vectors.
        #
        # This used to quote the compression ratio as a memory multiplier. The
        # two are different quantities. The compression ratio is the size of a
        # code against the size of the vector it replaces, while the ratio
        # between the modes also depends on the training records quantized_only
        # keeps at full width, on the codebook, on the centroid distance table
        # and on the graph. Measured over record counts from 1,000 to 50,000,
        # dimensions from 64 to 768 and 4 to 96 subvectors, the true multiplier
        # ran 1.0x to 20x on vectors and codes and 1.0x to 1.6x on the whole
        # resident index, against ratios of 16x to 384x. It moves with the
        # record count, which is not known here.
        #
        # The second sentence is this relay's addition. quantized_with_raw adds
        # the codes, the codebook and the centroid distance table on top of
        # every raw vector, and removes nothing, so it is above an unquantized
        # index at every record count. That is why _check_memory_usage runs its
        # break even arithmetic on quantized_only alone.
        if storage_mode == 'quantized_with_raw':
            import warnings
            warnings.warn(
                "storage_mode='quantized_with_raw' keeps a raw vector for every record "
                "as well as its code, so it uses more memory than 'quantized_only'. It "
                "also uses more than an unquantized index at every record count, since "
                "it adds the codes, the codebook and the centroid distance table and "
                "drops nothing. How much more depends on the final record count, which "
                "is not known at creation. get_stats() reports the memory each part "
                "holds once records are loaded. This mode is required for rerank and "
                "for exact vector reconstruction.",
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
    




    def _check_memory_usage(self, config: Dict[str, Any], dim: int,
                            expected_size: Any = None) -> None:
        """
        Warn about the fixed memory and the compression the configuration implies.

        Every figure here is fixed by the configuration and by the declared
        expected_size. Nothing that depends on the record count the index
        actually reaches belongs here, because create() does not know it.
        get_stats() reports the record dependent figures.

        Args:
            config: Validated quantization configuration
            dim: Vector dimension
            expected_size: The declared record count, or None to skip the break
                even check
        """
        import warnings

        subvectors = config['subvectors']
        bits = config['bits']
        training_size = config['training_size']
        storage_mode = config.get('storage_mode', 'quantized_only')
        sub_dim = dim // subvectors

        # The codebook is one centroid set per subvector, each of 2^bits
        # centroids of sub_dim float32. Since sub_dim is dim // subvectors,
        # subvectors cancels and the size is 2^bits * dim * 4 bytes.
        num_centroids_per_subvector = 2 ** bits
        total_centroids = subvectors * num_centroids_per_subvector
        centroid_bytes = total_centroids * sub_dim * 4
        centroid_memory_mb = centroid_bytes / (1024 * 1024)

        # Graph construction reads a table of the squared distance between every
        # pair of centroids within a subvector. The matrix is symmetric and its
        # diagonal is zero, so only the strict upper triangle is held, being
        # subvectors * k * (k - 1) / 2 float32 for k centroids. That is 0.996MB
        # at the default 8 subvectors of 8 bits, where the full square was
        # 2.00MB. It is built at training and it is usually the larger of the
        # two fixed costs, so a warning that counted only the codebook was
        # looking at the smaller number.
        k = num_centroids_per_subvector
        sdc_bytes = subvectors * (k * (k - 1) // 2) * 4
        sdc_memory_mb = sdc_bytes / (1024 * 1024)
        fixed_bytes = centroid_bytes + sdc_bytes
        fixed_memory_mb = fixed_bytes / (1024 * 1024)

        # A code is one byte per subvector whatever bits is, so this is the size
        # of a code against the size of the vector it replaces. It is not the
        # ratio between the two storage modes and it is not the memory an index
        # saves, both of which depend on the record count.
        original_bytes_per_vector = dim * 4  # float32
        compressed_bytes_per_vector = subvectors  # 1 byte per subvector code
        compression_ratio = original_bytes_per_vector / compressed_bytes_per_vector

        # Whether the configuration can repay its fixed cost at the size the
        # caller declared.
        #
        # Under quantized_only the index keeps a raw vector for each of the
        # training_size records and a code for every record. So against an
        # unquantized index of N records it saves (N - training_size) whole
        # vectors and pays a code for each of the training_size records it still
        # holds at full width, on top of the fixed cost:
        #
        #   saved(N) = (N - training_size) * (dim * 4 - subvectors)
        #              - training_size * subvectors
        #              - fixed_bytes
        #
        # Setting that to zero gives the record count above which quantization
        # starts saving. quantized_with_raw is excluded because it drops no raw
        # vector at all, so saved(N) is negative at every N and the advice to
        # raise expected_size would be false. The storage mode warning covers
        # that case instead.
        #
        # The check runs whether or not expected_size was set. The default is
        # 10,000 and the default training_size is also 10,000, so the default
        # configuration cannot save anything at the default size, and that is
        # precisely what a caller needs told.
        break_even = None
        if storage_mode == 'quantized_only' and original_bytes_per_vector > subvectors:
            per_record = original_bytes_per_vector - subvectors
            break_even = training_size + -(
                -(fixed_bytes + training_size * subvectors) // per_record
            )

        cannot_pay = (
            break_even is not None
            and isinstance(expected_size, int)
            and not isinstance(expected_size, bool)
            and expected_size < break_even
        )

        if cannot_pay:
            warnings.warn(
                f"This quantization configuration will use more memory than an "
                f"unquantized index at expected_size={expected_size}. It holds "
                f"{fixed_memory_mb:.2f}MB of codebook and centroid distance table "
                f"whatever the record count, and keeps the first {training_size} "
                f"records at full width. It starts saving above {break_even} records. "
                f"Raise expected_size if the estimate is low, or drop "
                f"quantization_config.",
                UserWarning,
                stacklevel=2
            )

        # Warn about large fixed memory, but only where the configuration does
        # pay, since the warning above already names the same figure. Exactly
        # one of the two fires.
        #
        # The threshold was 100MB while the table was a full square. Halving the
        # table halved the quantity the threshold measures, so it halves with
        # it, keeping the same configurations in scope. At dim 1536 with 512
        # subvectors of 8 bits the fixed memory is 65.2MB, which fired at 129.5MB
        # against 100 before this change and fires at 65.2MB against 50 after it.
        if fixed_memory_mb > 50 and not cannot_pay:
            warnings.warn(
                f"Large fixed quantization memory: {fixed_memory_mb:.1f}MB, being "
                f"{centroid_memory_mb:.1f}MB of centroids and {sdc_memory_mb:.1f}MB of "
                f"centroid distance table. This is held whatever the record count. "
                f"Reduce bits ({bits}) to lower both, or subvectors ({subvectors}) to "
                f"lower the table.",
                UserWarning,
                stacklevel=2
            )

        # A `compression_ratio < 4` branch used to sit here, advising a larger
        # subvectors count for better compression. It could not fire and the
        # advice was backwards. subvectors is validated at no more than dim, so
        # the ratio is at least dim * 4 / dim, which is exactly 4 and never
        # below it. Raising subvectors lowers the ratio rather than raising it.

        # Warn about extremely high compression. The ratio is dim * 4 /
        # subvectors, so only subvectors moves it. bits sets the centroid count
        # and leaves the code at one byte per subvector, so it does not.
        if compression_ratio > 50:
            warnings.warn(
                f"Very high compression ratio: {compression_ratio:.1f}x may significantly impact recall quality. "
                f"Increase subvectors ({subvectors}) to lower it, at the cost of memory and build time.",
                UserWarning,
                stacklevel=2
            )

    @classmethod
    def available_index_types(cls) -> list[str]:
        """Return list of all supported index types."""
        return sorted(cls._index_types.keys())
    