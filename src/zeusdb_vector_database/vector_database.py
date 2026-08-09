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
                    'subvectors': 24,          # Number of subvectors (must divide dim evenly).
                                               # Default: dim / 32 clamped to 8..192 and snapped to
                                               # a divisor of dim, so 128x compression from dim 256 up
                    'bits': 8,                 # Bits per subvector (1-8, controls centroids, default: 8)
                    'training_size': None,     # Auto-calculated based on subvectors & bits (or specify manually)
                    'max_training_vectors': None,  # Optional limit on training vectors used
                    'storage_mode': 'quantized_only' # Storage mode for quantized vectors (or 'quantized_with_raw')  
                }

            Note: Quantization replaces each vector with a code of one byte per
            subvector, so the code is dim * 4 / subvectors smaller than the vector.
            The memory an index saves is smaller than that, because the codebook
            and the centroid distance table are held whatever the record count.
            A record is held twice, once in a storage map and once inside the
            graph, and both storage modes replace the graph's copy, so both hold
            less than an unquantized index above their break even record count.
            Under 'quantized_only' the records collected for training are held at
            full width only until training completes, then released. get_stats()
            reports the codes, the raw vectors, the codebook and the centroid
            distance table, and does not report the graph. Recall falls sharply,
            not slightly. Only 'quantized_with_raw' can recover it, by reranking
            against the raw vectors it keeps. Training triggers automatically on
            the first .add() call that reaches the training_size threshold, and
            for 'quantized_with_raw' it also measures how far a search has to
            over-fetch before reranking, on the training records themselves.
            get_stats() reports that measurement under the rerank_ keys and the
            fetch it produces at the current record count.

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
        #
        # An unset subvectors is derived from the dimension rather than fixed at
        # 8. See _default_subvectors for what 8 produced and why the quantity
        # that has to be held steady is the compression ratio.
        subvectors = validated_config.get('subvectors')
        subvectors_was_derived = subvectors is None
        if subvectors_was_derived:
            subvectors = self._default_subvectors(dim)
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
        self._check_memory_usage(validated_config, dim, expected_size,
                                 subvectors_was_derived)

        # Warn about the memory cost of keeping raw vectors.
        #
        # This used to quote the compression ratio as a memory multiplier. The
        # two are different quantities. The compression ratio is the size of a
        # code against the size of the vector it replaces, while the ratio
        # between the modes also depends on the codebook, on the centroid
        # distance table and on the graph, which both modes hold identically.
        # It moves with the record count, which is not known here.
        #
        # The mode does not remove nothing. The HNSW graph owns its own copy of
        # every point, and under quantization that copy is subvectors bytes
        # where an unquantized graph holds dim * 4. quantized_with_raw drops
        # that copy exactly as quantized_only does, and keeps only the vectors
        # map at full width, so above its break even it holds less than an
        # unquantized index rather than more. Measured at dimension 768 and the
        # derived subvectors it is 0.69 times an unquantized index at 10,000
        # records and 0.59 times at 100,000, and at 8 subvectors 0.60 and 0.47.
        # _check_memory_usage now runs the break even arithmetic for both modes.
        if storage_mode == 'quantized_with_raw':
            import warnings
            warnings.warn(
                "storage_mode='quantized_with_raw' keeps a raw vector for every record "
                "as well as its code, so it uses more memory than 'quantized_only'. It "
                "uses less than an unquantized index above the break even record count, "
                "because the graph holds a code of "
                f"{subvectors} bytes per record where an unquantized graph holds "
                f"{dim * 4}. Measured at dimension 768 and the derived subvectors it "
                "holds 0.69 times an unquantized index at 10,000 records and 0.59 "
                "times at 100,000. "
                "get_stats() reports the memory the codes, the raw vectors, the codebook "
                "and the centroid distance table hold once records are loaded, and does "
                "not report the graph. This mode is required for rerank and for exact "
                "vector reconstruction.",
                UserWarning,
                stacklevel=2
            )
        
        # Final safety check: ensure all expected keys are present
        # This is a final defensive programming - all the keys should already be set above, but added just in case
        validated_config.setdefault('type', 'pq')
        validated_config.setdefault('subvectors', self._default_subvectors(dim))
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
        # The rate is stated per centroid because the codebook is one k-means
        # per subvector over 2^bits centroids, so what decides whether a
        # centroid is fitted or guessed is how many points fall to it. At the
        # default of 8 bits the floor governs and the effective rate is 39
        # points per centroid.
        #
        # faiss targets 256 points per centroid and warns below 39, so this
        # sits exactly at its warning floor. Raising it to faiss's target was
        # measured rather than assumed, at dim 768 over 100,000 records on an
        # anisotropic embedding-like corpus, 100 queries:
        #
        #   training_size   per centroid   recall@10   build     90th pct depth
        #        1,000            3.9        0.987      64 s          1,722
        #       10,000           39.1        0.989     121 s          1,506
        #       65,536          256.0        0.980     408 s          1,841
        #
        # 6.5 times the training set costs 3.4 times the build and returns
        # slightly less recall, which is inside the 0.013 a codebook draw moves
        # a quantized recall figure by. Training alone costs 3.1 s at 1,000
        # vectors, 40.6 s at 10,000 and 306.2 s at 65,536, being linear in the
        # training set. The default stays where it is because the measurement
        # found nothing to buy.
        centroids_per_subvector = 2 ** bits
        min_samples_per_centroid = 20

        statistical_minimum = centroids_per_subvector * min_samples_per_centroid

        # Practical bounds
        reasonable_minimum = 10000    # Always need at least this for diversity
        reasonable_maximum = 200000   # Diminishing returns beyond this point

        return min(max(statistical_minimum, reasonable_minimum), reasonable_maximum)

    
    def _suggest_subvector_divisors(self, dim: int) -> list[int]:
        """Return valid subvector counts that divide the dimension evenly (up to 32)."""
        return [i for i in range(1, min(33, dim + 1)) if dim % i == 0]

    # Dimensions covered by one subvector at the default.
    #
    # The compression ratio is dim * 4 / subvectors, and dim / subvectors is the
    # subvector width, so the ratio is exactly four times the width. A fixed
    # subvector count is therefore a fixed code length and a ratio that moves
    # with the dimension. The old default of 8 gave 32x at dim 64, 128x at 256,
    # 384x at 768 and 768x at 1,536, so one configuration landed a caller in a
    # different accuracy regime for every embedding model.
    #
    # Recall follows the ratio and not the dimension, which is what makes a
    # fixed width the right shape for the default. Measured on the realistic
    # corpus at 10,000 records, recall at 10 with no rerank reads 0.187, 0.182
    # and 0.184 at 128x for dims 256, 768 and 1,536, and 0.405 and 0.406 at 32x
    # for dims 256 and 768. Two indexes at the same ratio return the same recall
    # at different dimensions and two at the same subvector count do not.
    #
    # A width of 32 is 128x. That is the highest ratio that returns recall at 10
    # above 0.99 at the fetch the default rerank produces, at every corpus size
    # measured. Recall at 10 at the default fetch, dim 768, 200 queries:
    #
    #   ratio   10,000 records   100,000 records
    #   384x    0.9800           0.9935
    #   192x    0.9850           0.9980
    #   128x    0.9925           0.9980
    #    64x    0.9995           1.0000
    #
    # The binding size is the smaller one, which is not where it was expected.
    # At 100,000 records the fetch is 2 percent of the corpus and the deepest
    # true neighbour sits at 1.92 percent, so even 384x clears the bar. At
    # 10,000 records the fetch was the rerank floor of 200 when this was
    # measured, and 384x needs 208.
    #
    # Going further down the ratio costs memory and build time and returns
    # nothing on recall. At 100,000 records of dim 768, 32x holds 111MB more
    # resident memory and builds in 521s against 170 for the same recall, and
    # 16x holds 160MB more and builds in 846s. At 10,000 records 16x holds
    # 93.8MB where an unquantized index holds 89.6, so the memory case is gone
    # by then and a default cannot sit there.
    #
    # Query time is the one axis where a lower ratio can win, and it wins only
    # at the bottom of the range and only at scale. A lower ratio needs a
    # shallower fetch and pays more per candidate. Between 384x and 32x the two
    # cancel and the latency at matched recall is flat. At 16x the fetch
    # collapses, from 1,921 candidates at 128x to 222 at 100,000 records, and
    # the query falls to 4.32ms against 13.08. That is a real trade and it is
    # documented in the README rather than taken by default, because it costs
    # more memory than not quantizing at 10,000 records and five times the build
    # at 100,000.
    #
    # The field runs lower still, LanceDB at 32x and Weaviate at 24x, and
    # neither holds the raw vectors alongside the codes as this mode does, so
    # neither has reranking to lean on.
    DEFAULT_SUBVECTOR_WIDTH = 32

    # Floor on the derived count. A code is one byte per subvector, so a small
    # count is a small code space, and at 2 subvectors of 8 bits there are only
    # 65,536 distinct codes for the whole corpus. Eight is 256^8, which is far
    # beyond any corpus, so the floor costs nothing above it. It binds below dim
    # 256, where it holds the ratio under 128x rather than at it.
    DEFAULT_SUBVECTOR_FLOOR = 8

    # Ceiling on the derived count, because the centroid distance table is
    # subvectors * 2^bits * (2^bits - 1) / 2 * 4 bytes and the graph build does
    # one table lookup per subvector per comparison, so both grow linearly in
    # the subvector count. It binds above dim 6,144.
    DEFAULT_SUBVECTOR_CEILING = 192

    # Bytes a record costs an index beyond the two copies of its vector
    #
    # Fitted from a controlled sweep, one dimension per process, 25,000 records
    # of clustered data at m 32, an unquantized index and a quantized_with_raw
    # one built over the same records in the same process, resident set delta
    # for each. An unquantized index holds every vector twice, once in the
    # storage map and once inside the graph, so the rest is what this names.
    #
    #   dim    unquantized  quantized_with_raw  ratio   bytes/record  less 8*dim
    #    64       75.71 MiB          73.44 MiB  0.970          3,176       2,664
    #    96       83.36              73.77      0.885          3,496       2,728
    #   128       89.74              74.96      0.835          3,764       2,740
    #   192      102.57              82.73      0.807          4,302       2,766
    #   256      115.47              88.19      0.764          4,843       2,795
    #   384      140.53             105.82      0.753          5,894       2,822
    #   768      216.10             151.54      0.701          9,064       2,920
    # 1,536      369.59             236.02      0.639         15,502       3,214
    #
    # The last column is flat across a 24 fold range in the dimension, which is
    # what makes it a per record constant rather than a fit with a slope in it.
    # It is the graph's neighbour lists at m 32, the two id maps, the metadata
    # map and the hash map keys.
    MEASURED_RECORD_OVERHEAD_BYTES = 2740

    # The dimension below which quantization does not repay
    #
    # Quantization removes the graph's copy of the vector and adds a code in its
    # place, so under quantized_with_raw the memory it saves per record is
    # `dim * 4 - 2 * subvectors` bytes against the `dim * 8 + 2,740` an
    # unquantized index holds. That share is what the table above measures and
    # it is the whole of what quantization buys at this storage mode.
    #
    # The bar is one fifth. Below it a caller pays a rerank fetch of several
    # hundred candidates on every search for less than a fifth of the memory.
    #
    #   saving share = (4 * dim - 2 * subvectors) / (8 * dim + 2,740)
    #
    # At the derived subvectors of 8, that reaches one fifth at dim 235.
    # Interpolating the measured column instead, between 19.34 percent at dim
    # 192 and 23.63 percent at dim 256, puts it at dim 202. Both sit between
    # two measured points rather than inside one, which is what makes the
    # threshold a reading of the sweep rather than a round number.
    #
    # quantized_only replaces both copies of the vector rather than one, so its
    # share is `(8 * dim - 2 * subvectors) / (8 * dim + 2,740)` and it reaches
    # one fifth at dim 88. The warning is therefore driven by the share and not
    # by the dimension, and the dimension it fires below is 235 for
    # quantized_with_raw and 88 for quantized_only.
    QUANTIZATION_REPAYS_SAVING_SHARE = 0.20

    def _memory_saving_share(self, dim: int, subvectors: int,
                             storage_mode: str) -> float:
        """The share of an unquantized index's memory quantization removes.

        From `MEASURED_RECORD_OVERHEAD_BYTES` and the arithmetic recorded on
        `LOW_DIMENSION_QUANTIZATION_THRESHOLD`. `quantized_only` drops both
        copies of the vector rather than one, so it saves `dim * 8 - 2 *
        subvectors` where `quantized_with_raw` saves `dim * 4 - 2 * subvectors`.
        Neither figure counts the codebook or the centroid distance table,
        which are fixed and which `_check_memory_usage` prices separately.
        """
        held = dim * 8 + self.MEASURED_RECORD_OVERHEAD_BYTES
        replaced = dim * 8 if storage_mode == 'quantized_only' else dim * 4
        return (replaced - 2 * subvectors) / held

    def _default_subvectors(self, dim: int) -> int:
        """The subvector count an unset `subvectors` resolves to.

        `dim / DEFAULT_SUBVECTOR_WIDTH`, clamped to the floor and the ceiling,
        and snapped to a divisor of `dim` because the validator requires one.
        Divisors outside the clamp are excluded rather than merely made
        unattractive, since a tie could otherwise land outside it. A tie inside
        it takes the larger of the two, which is the more accurate.
        """
        target = min(max(round(dim / self.DEFAULT_SUBVECTOR_WIDTH),
                         self.DEFAULT_SUBVECTOR_FLOOR),
                     self.DEFAULT_SUBVECTOR_CEILING)

        divisors = set()
        i = 1
        while i * i <= dim:
            if dim % i == 0:
                divisors.add(i)
                divisors.add(dim // i)
            i += 1

        allowed = [d for d in sorted(divisors)
                   if self.DEFAULT_SUBVECTOR_FLOOR <= d <= self.DEFAULT_SUBVECTOR_CEILING]

        # A dimension below the floor, or one whose only divisors fall outside
        # the clamp, has nothing to pick from. Fall back to every divisor.
        if not allowed:
            allowed = [d for d in sorted(divisors) if d <= self.DEFAULT_SUBVECTOR_CEILING]

        return min(allowed, key=lambda d: (abs(d - target), -d))
    




    def _check_memory_usage(self, config: Dict[str, Any], dim: int,
                            expected_size: Any = None,
                            subvectors_was_derived: bool = False) -> None:
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

        # An index whose declared size never reaches training_size never
        # trains, so quantization never engages at all. It holds every record
        # raw exactly as an unquantized index does, pays no fixed cost, and
        # gains nothing. The memory warnings below describe the trained state,
        # which this configuration never reaches at its declared size, so this
        # one replaces them.
        declared_size = (
            expected_size
            if isinstance(expected_size, int) and not isinstance(expected_size, bool)
            else None
        )
        never_trains = declared_size is not None and declared_size < training_size
        if never_trains:
            warnings.warn(
                f"expected_size={expected_size} is below "
                f"training_size={training_size}, so training will never trigger "
                f"at the declared size and quantization will never engage. Raise "
                f"expected_size if the estimate is low, lower training_size, or "
                f"drop quantization_config.",
                UserWarning,
                stacklevel=2
            )

        # Whether the configuration can repay its fixed cost at the size the
        # caller declared.
        #
        # Both modes hold a record twice, once in a storage map and once inside
        # the HNSW graph, which owns its own copy of every point. Quantization
        # replaces a copy of dim * 4 bytes with a code of subvectors bytes, and
        # the two modes differ in how many of the two copies they replace.
        #
        # Under quantized_only both copies become codes, because the raw
        # vectors collected for training are released the moment their codes
        # are stored. Counting one of the two, and the code it pays for:
        #
        #   saved(N) = N * (dim * 4 - subvectors) - fixed_bytes
        #
        # That is deliberately conservative. It ignores the second copy the
        # mode also drops, so it names a record count above the true one and
        # warns where it need not rather than failing to warn.
        #
        # Under quantized_with_raw the storage map keeps every raw vector and
        # only the graph's copy becomes a code, and the mode pays for two codes
        # while dropping one copy:
        #
        #   saved(N) = N * (dim * 4 - 2 * subvectors) - fixed_bytes
        #
        # There is no second copy to leave out of that one, so it is the steady
        # state figure rather than a conservative one. This mode used to be
        # excluded here on the claim that it drops nothing and is above an
        # unquantized index at every record count. It drops the graph's copy,
        # which at dim 768 is 3,072 bytes per record, and it has a break even
        # like the other mode.
        break_even = None
        if storage_mode == 'quantized_only':
            per_record = original_bytes_per_vector - subvectors
        else:
            per_record = original_bytes_per_vector - 2 * subvectors
        if per_record > 0:
            break_even = -(-fixed_bytes // per_record)

        cannot_pay = (
            not never_trains
            and break_even is not None
            and declared_size is not None
            and declared_size < break_even
        )

        if cannot_pay:
            warnings.warn(
                f"This quantization configuration will use more memory than an "
                f"unquantized index at expected_size={expected_size}. It holds "
                f"{fixed_memory_mb:.2f}MB of codebook and centroid distance table "
                f"whatever the record count, and starts saving above {break_even} "
                f"records. Raise expected_size if the estimate is low, or drop "
                f"quantization_config.",
                UserWarning,
                stacklevel=2
            )

        # Warn about large fixed memory, but only where the configuration does
        # pay and does train, since the warnings above already cover the other
        # cases. At most one of the three fires.
        #
        # The threshold was 100MB while the table was a full square. Halving the
        # table halved the quantity the threshold measures, so it halves with
        # it, keeping the same configurations in scope. At dim 1536 with 512
        # subvectors of 8 bits the fixed memory is 65.2MB, which fired at 129.5MB
        # against 100 before this change and fires at 65.2MB against 50 after it.
        if fixed_memory_mb > 50 and not cannot_pay and not never_trains:
            warnings.warn(
                f"Large fixed quantization memory: {fixed_memory_mb:.1f}MB, being "
                f"{centroid_memory_mb:.1f}MB of centroids and {sdc_memory_mb:.1f}MB of "
                f"centroid distance table. This is held whatever the record count. "
                f"Reduce bits ({bits}) to lower both, or subvectors ({subvectors}) to "
                f"lower the table.",
                UserWarning,
                stacklevel=2
            )

        # Warn where the dimension is too low for quantization to repay. The
        # bar and the arithmetic are on QUANTIZATION_REPAYS_SAVING_SHARE. This
        # fires alongside the compression warning below rather than instead of
        # it, because they describe different things, and it is suppressed
        # where the configuration never trains or cannot pay its fixed cost at
        # all, both of which are already stated above.
        saving_share = self._memory_saving_share(dim, subvectors, storage_mode)
        if (saving_share < self.QUANTIZATION_REPAYS_SAVING_SHARE
                and not cannot_pay and not never_trains):
            warnings.warn(
                f"At dim={dim} a trained {storage_mode} index holds about "
                f"{saving_share * 100:.0f}% less memory than an unquantized index "
                f"over the same records, and every search on it fetches and "
                f"rescores hundreds of candidates. Measured at 25,000 records and "
                f"m=32, quantized_with_raw holds 0.97 times an unquantized index "
                f"at dim=64, 0.84 at dim=128, 0.81 at dim=192, 0.76 at dim=256 "
                f"and 0.64 at dim=1536.",
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
        #
        # A derived subvectors is exempt. This warning tells a caller that the
        # value they chose looks unbalanced, and the derived value is 128x from
        # dim 256 upward, which is above the threshold and is the measured
        # choice. Warning about it would have create() warn about its own
        # default at every common embedding dimension.
        if compression_ratio > 50 and not subvectors_was_derived:
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
    