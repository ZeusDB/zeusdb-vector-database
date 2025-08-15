"""
ZeusDB LangChain Integration - Enterprise Edition

LangChain-compatible VectorStore wrapper for ZeusDB vector indices with
enterprise-grade structured logging and monitoring.

Features:
- Full LangChain VectorStore API compliance
- Enterprise structured logging with performance monitoring
- Automatic score normalization for all distance metrics
- Document ID handling and retrieval
- Maximal Marginal Relevance (MMR) search
- Async support for all operations
- ZeusDB-specific features (quantization, persistence, stats)

Usage:
    >>> from zeusdb import VectorDatabase, ZeusDBLangChain
    >>> from langchain_openai import OpenAIEmbeddings
    >>>
    >>> vdb = VectorDatabase()
    >>> index = vdb.create("hnsw", dim=1536, space="cosine")
    >>>
    >>> vectorstore = ZeusDBLangChain(zeusdb_index=index, embedding=OpenAIEmbeddings())
    >>> docs = vectorstore.similarity_search("hello world", k=5)
    >>> retriever = vectorstore.as_retriever(search_type="mmr")

Note: This module requires langchain-core to be installed for full functionality.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    cast,
)
import math
import uuid
import warnings
from time import perf_counter
from contextlib import contextmanager

# -----------------------------------------------------------------------------
# Enterprise Logging Integration with Safe Fallback
# -----------------------------------------------------------------------------
try:
    from .logging_config import get_logger as _get_logger  # type: ignore[import]
except Exception:  # fallback for OSS/dev environments
    import logging
    
    class _StructuredAdapter(logging.LoggerAdapter):
        """Adapter that moves arbitrary kwargs into 'extra' for stdlib logging compatibility."""
        def process(self, msg, kwargs):
            allowed = {"exc_info", "stack_info", "stacklevel", "extra"}
            extra = kwargs.get("extra", {}) or {}
            if not isinstance(extra, dict):
                extra = {"_extra": repr(extra)}  # defensive: keep logging robust
            # Move arbitrary kwargs into .extra for stdlib logging
            fields = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in allowed}
            if fields:
                extra.update(fields)
                kwargs["extra"] = extra
            return msg, kwargs
    
    def _get_logger(name: str) -> logging.LoggerAdapter:
        base = logging.getLogger(name)
        if not base.handlers:
            base.addHandler(logging.NullHandler())
        return _StructuredAdapter(base, {})

# Initialize module logger (no configuration here - central config owns that)
# Expose a single name with a single (loose) type for static checkers
get_logger: Callable[[str], Any] = cast(Callable[[str], Any], _get_logger)
logger = get_logger("zeusdb.langchain")

# Operation context manager for structured tracing
try:
    #from .logging_config import operation_context
    from .logging_config import operation_context as _operation_context  # type: ignore[import]
except Exception:
    @contextmanager
    #def operation_context(operation_name: str, **context):
    def _operation_context(operation_name: str, **context: Any):
        logger.debug(f"{operation_name} started", operation=operation_name, **context)
        start = perf_counter()
        try:
            yield
            duration_ms = (perf_counter() - start) * 1000
            logger.info(f"{operation_name} completed", operation=operation_name, duration_ms=duration_ms, **context)
        except Exception as e:
            duration_ms = (perf_counter() - start) * 1000
            logger.error(f"{operation_name} failed", operation=operation_name, duration_ms=duration_ms, error=str(e), **context)
            raise

# Expose unified symbol (loosen the type to quiet static analyzers)
operation_context = cast(Callable[..., Any], _operation_context)

# -----------------------------------------------------------------------------
# Import LangChain types for static checking; provide runtime shims if missing.
# -----------------------------------------------------------------------------
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore as _LCVectorStore
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.runnables.config import run_in_executor
    LANGCHAIN_AVAILABLE: bool
else:
    try:
        from langchain_core.vectorstores import VectorStore as _LCVectorStore
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.runnables.config import run_in_executor
        LANGCHAIN_AVAILABLE = True
    except Exception:  # ImportError in most cases
        LANGCHAIN_AVAILABLE = False

        class _LCVectorStore:  # minimal shim, not exported
            @staticmethod
            def _cosine_relevance_score_fn(distance: float) -> float:
                return 1.0 - distance

            @staticmethod
            def _euclidean_relevance_score_fn(distance: float) -> float:
                return 1.0 - distance / math.sqrt(2)

        class Document:  # type: ignore[no-redef]
            def __init__(
                self,
                page_content: str = "",
                metadata: Optional[dict] = None,
                id: Optional[str] = None,
            ):
                self.page_content = page_content
                self.metadata = metadata or {}
                self.id = id

        class Embeddings:  # type: ignore[no-redef]
            def embed_documents(self, _texts: List[str]) -> List[List[float]]:
                raise NotImplementedError

            def embed_query(self, _text: str) -> List[float]:
                raise NotImplementedError

        async def run_in_executor(*_a: Any, **_k: Any):  # type: ignore[no-redef]
            raise ImportError(
                "ZeusDBLangChain requires `langchain-core`. "
                "Install with: pip/uv install 'zeusdb-vector-database[langchain]'"
            )


class ZeusDBVectorStore(_LCVectorStore):
    """LangChain-compatible wrapper for ZeusDB vector indices with enterprise logging."""

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __init__(
        self,
        zeusdb_index: Any,  # expected to expose: add, search, get_records, remove_point, save, etc.
        embedding: Embeddings,
        text_key: str = "text",
        **kwargs: Any,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            logger.error(
                "LangChain dependency missing",
                operation="init",
                available_langchain=False
            )
            raise ImportError(
                "ZeusDBLangChain requires langchain-core package.\n"
                "Install with one of:\n"
                "  uv pip install langchain-core\n"
                "  pip install langchain-core\n"
                "  pip/uv install 'zeusdb-vector-database[langchain]'"
            )
        
        super().__init__(**kwargs)
        self.zeusdb_index = zeusdb_index
        self.embedding = embedding
        self.text_key = text_key

        logger.info(
            "ZeusDB VectorStore initialized",
            operation="init",
            text_key=text_key,
            has_index=zeusdb_index is not None,
            has_embedding=embedding is not None
        )

    # For LangChain's retriever tags
    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access to embedding function (LangChain compatibility)."""
        return self.embedding

    # -------------------------------------------------------------------------
    # Relevance score normalization selection (used by LC base helpers)
    # -------------------------------------------------------------------------
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Choose correct normalization for ZeusDB's distance metric."""
        space = "cosine"
        try:
            space_attr = getattr(self.zeusdb_index, "space", None)
            space = space_attr() if callable(space_attr) else (space_attr or space)
            space = str(space).lower()
        except Exception as e:
            logger.debug(
                "Failed to detect space, defaulting to cosine",
                operation="relevance_score_selection",
                error=str(e)
            )
            space = "cosine"

        logger.debug(
            "Selected relevance score function",
            operation="relevance_score_selection",
            space=space
        )

        if space == "cosine":
            return self._cosine_relevance_score_fn
        elif space in {"l2", "euclidean"}:
            return self._euclidean_relevance_score_fn
        elif space == "l1":
            return self._euclidean_relevance_score_fn  # approximation
        return self._cosine_relevance_score_fn

    # -------------------------------------------------------------------------
    # Conversions
    # -------------------------------------------------------------------------
    def _zeusdb_result_to_document(self, result: Dict[str, Any]) -> Document:
        """Convert a ZeusDB search record to an LC Document with ID."""
        metadata = (result.get("metadata") or {}).copy()
        page_content = metadata.pop(self.text_key, result.get("id", ""))
        zeusdb_id = result.get("id")

        if "score" in result:
            metadata["zeusdb_score"] = result["score"]

        return Document(id=zeusdb_id, page_content=str(page_content), metadata=metadata)

    def _langchain_docs_to_zeusdb_format(self, documents: List[Document]) -> Dict[str, Any]:
        """Convert LangChain Documents to ZeusDB batch format."""
        if not documents:
            return {"ids": [], "metadatas": [], "texts": []}

        ids: List[str] = []
        metadatas: List[dict] = []
        texts: List[str] = []

        for doc in documents:
            did = str(doc.id) if doc.id is not None else str(uuid.uuid4())
            ids.append(did)
            texts.append(doc.page_content)
            md = dict(doc.metadata or {})
            md[self.text_key] = doc.page_content
            metadatas.append(md)

        logger.debug(
            "Converted documents to ZeusDB format",
            operation="docs_to_zeusdb_format",
            doc_count=len(documents),
            has_ids=all(d.id is not None for d in documents)
        )

        return {"ids": ids, "metadatas": metadatas, "texts": texts}

    def _langchain_texts_to_zeusdb_format(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Convert LangChain text format to ZeusDB batch format."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts, metadatas, and ids must have the same length")

        enriched_mds = []
        for i, md in enumerate(metadatas):
            md2 = dict(md or {})
            md2[self.text_key] = texts[i]
            enriched_mds.append(md2)

        logger.debug(
            "Converted texts to ZeusDB format",
            operation="texts_to_zeusdb_format",
            text_count=len(texts),
            has_metadata=metadatas is not None,
            has_ids=ids is not None
        )

        return {"ids": ids, "metadatas": enriched_mds, "texts": texts}

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------
    def _embed_doc_text(self, text: str) -> List[float]:
        """Embed document text using the document embedding pathway."""
        return self.embedding.embed_documents([text])[0]

    @staticmethod
    def _extract_index_dim(idx: Any) -> Optional[int]:
        """Best-effort way to read the index dimension from common attrs/methods."""
        for name in ("dim", "dimension", "dimensions", "vector_dim", "get_dim"):
            if hasattr(idx, name):
                raw = getattr(idx, name)
                try:
                    val = raw() if callable(raw) else raw  # may be int / float / str / numpy scalar / etc.

                    # Fast path: already an int
                    if isinstance(val, int):
                        return val
                    
                    # Float -> int
                    if isinstance(val, float):
                        return int(val)
                    
                    # String of digits -> int
                    if isinstance(val, str):
                        return int(val) if val.isdigit() else None
                    
                    # NumPy scalar or objects exposing .item()
                    item = getattr(val, "item", None)
                    if callable(item):
                        v2 = item()
                        if isinstance(v2, int):
                            return v2
                        if isinstance(v2, float):
                            return int(v2)
                        
                    # Objects that implement __int__
                    if hasattr(val, "__int__"):
                        try:
                            return int(val)  # type: ignore[arg-type]
                        except Exception:
                            pass
                except Exception:
                    pass
        return None

    @staticmethod
    def _validate_embedding_dims(
        embedding: "Embeddings",
        texts: Sequence[str],
        expected_dim: int,
        sample_size: int = 3,
    ) -> None:
        """Embed a small sample and ensure each vector has expected_dim."""
        if not texts:
            return
        sample = list(texts)[:sample_size]
        vecs = embedding.embed_documents(sample)
        for i, v in enumerate(vecs):
            if len(v) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {len(v)} for sample[{i}] "
                    f"but expected {expected_dim}"
                )

    @staticmethod
    def _cosine_distance(v1: List[float], v2: List[float]) -> float:
        """Cosine distance (0 = identical, 2 = opposite)."""
        try:
            dot = sum(a * b for a, b in zip(v1, v2))
            n1 = math.sqrt(sum(a * a for a in v1))
            n2 = math.sqrt(sum(b * b for b in v2))
            if n1 == 0 or n2 == 0:
                return 1.0
            sim = dot / (n1 * n2)
            sim = max(-1.0, min(1.0, sim))
            return 1.0 - sim
        except Exception:
            return 1.0

    def _get_index_dim(self) -> Optional[int]:
        """Best-effort: read the index dimension if exposed by the backend."""
        return self._extract_index_dim(self.zeusdb_index)

    def _coerce_vector(self, embedding: Sequence[float], where: str) -> Optional[List[float]]:
        """Coerce input vector (list/tuple/np.ndarray) to List[float] and dimension-check."""
        try:
            emb_any = cast(Any, embedding)
            tolist_maybe = getattr(emb_any, "tolist", None)
            if callable(tolist_maybe):  # numpy-style
                seq = cast(Sequence[float], tolist_maybe())
                vec = [float(x) for x in seq]
            else:
                seq2 = cast(Iterable[float], embedding)
                vec = [float(x) for x in seq2]
        except (TypeError, ValueError, AttributeError) as e:
            # Dual log+warn for user-visible coercion failures
            logger.warning(
                "Vector coercion failed",
                operation=where,
                error=str(e),
                error_type=type(e).__name__
            )
            warnings.warn(
                f"Could not coerce embedding to floats: {e}",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return None

        if not vec:
            # Dual log+warn for empty embeddings
            logger.warning(
                "Empty embedding received",
                operation=where
            )
            warnings.warn(
                "Received an empty embedding vector",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Optional dimension check with dual log+warn for dimension mismatches
        expected = self._get_index_dim()
        if expected is not None and len(vec) != expected:
            logger.warning(
                "Embedding dimension mismatch",
                operation=where,
                embedding_dim=len(vec),
                index_dim=expected
            )
            warnings.warn(
                "Embedding dimensions don't match index; results may be degraded.",
                category=RuntimeWarning,
                stacklevel=2
            )
            # Continue to let backend handle or raise if it can't

        return vec

    # -------------------------------------------------------------------------
    # Core VectorStore (sync)
    # -------------------------------------------------------------------------
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        texts_list = list(texts)
        if not texts_list:
            logger.debug("No texts to add", operation="add_texts")
            return []

        logger.info(
            "Adding texts to vector store",
            operation="add_texts",
            text_count=len(texts_list),
            has_metadata=metadatas is not None,
            has_ids=ids is not None
        )

        with operation_context("add_texts", text_count=len(texts_list)):
            try:
                t0 = perf_counter()
                vectors = self.embedding.embed_documents(texts_list)
                embed_ms = (perf_counter() - t0) * 1000
            except Exception:
                logger.error(
                    "Embedding failed",
                    operation="add_texts",
                    text_count=len(texts_list),
                    exc_info=True
                )
                return []

            batch = self._langchain_texts_to_zeusdb_format(texts_list, metadatas, ids)
            batch["vectors"] = vectors
            batch.pop("texts", None)

            t1 = perf_counter()
            try:
                result = self.zeusdb_index.add(batch, overwrite=kwargs.get("overwrite", True))
                add_ms = (perf_counter() - t1) * 1000

                # Handle result feedback with structured logging
                if hasattr(result, "is_success") and hasattr(result, "total_errors"):
                    if not result.is_success() and result.total_errors > 0:
                        logger.warning(
                            "Some texts failed to add",
                            operation="add_texts",
                            total_errors=result.total_errors,
                            success_count=getattr(result, "total_inserted", 0)
                        )
                        
                        # Verbose error details at DEBUG level to avoid noise
                        if kwargs.get("verbose", False):
                            errors = getattr(result, "errors", [])
                            if isinstance(errors, list):
                                for i, err in enumerate(errors[:5]):
                                    logger.debug(
                                        "Add error item",
                                        operation="add_texts",
                                        error_index=i,
                                        error_message=str(err)
                                    )

                inserted = getattr(result, "total_inserted", None)
                logger.info(
                    "Add texts completed",
                    operation="add_texts",
                    embed_ms=embed_ms,
                    add_ms=add_ms,
                    total_inserted=inserted if inserted is not None else len(texts_list)
                )

            except Exception as e:
                add_ms = (perf_counter() - t1) * 1000
                logger.error(
                    "ZeusDB add_texts failed",
                    operation="add_texts",
                    text_count=len(texts_list),
                    embed_ms=embed_ms,
                    add_ms=add_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )

        return batch["ids"]

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add LangChain Documents to the vector store."""
        if not documents:
            logger.debug("No documents to add", operation="add_documents")
            return []

        logger.info(
            "Adding documents to vector store",
            operation="add_documents",
            doc_count=len(documents),
            has_ids=all(d.id is not None for d in documents)
        )

        with operation_context("add_documents", doc_count=len(documents)):
            batch = self._langchain_docs_to_zeusdb_format(documents)
            
            try:
                t0 = perf_counter()
                vectors = self.embedding.embed_documents(batch["texts"])
                embed_ms = (perf_counter() - t0) * 1000
            except Exception:
                logger.error(
                    "Embedding failed",
                    operation="add_documents",
                    doc_count=len(documents),
                    exc_info=True
                )
                return []

            batch["vectors"] = vectors
            batch.pop("texts", None)

            t1 = perf_counter()
            try:
                result = self.zeusdb_index.add(batch, overwrite=kwargs.get("overwrite", True))
                add_ms = (perf_counter() - t1) * 1000

                if hasattr(result, "is_success") and hasattr(result, "total_errors"):
                    if not result.is_success() and result.total_errors > 0:
                        logger.warning(
                            "Some documents failed to add",
                            operation="add_documents",
                            total_errors=result.total_errors,
                            success_count=getattr(result, "total_inserted", 0)
                        )

                inserted = getattr(result, "total_inserted", None)
                logger.info(
                    "Add documents completed",
                    operation="add_documents",
                    embed_ms=embed_ms,
                    add_ms=add_ms,
                    total_inserted=inserted if inserted is not None else len(documents)
                )

            except Exception as e:
                add_ms = (perf_counter() - t1) * 1000
                logger.error(
                    "ZeusDB add_documents failed",
                    operation="add_documents",
                    doc_count=len(documents),
                    embed_ms=embed_ms,
                    add_ms=add_ms,
                    error=str(e), 
                    error_type=type(e).__name__,
                    exc_info=True
                )

        return batch["ids"]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Search for similar documents by query text."""
        has_filter = kwargs.get("filter") is not None
        ef_search = kwargs.get("ef_search")
        
        logger.debug(
            "Performing similarity search",
            operation="similarity_search",
            query_length=len(query),
            top_k=k,
            has_filter=has_filter,
            ef_search=ef_search
        )

        with operation_context("similarity_search", top_k=k, has_filter=has_filter, ef_search=ef_search):
            t0 = perf_counter()
            q_vec = self.embedding.embed_query(query)
            embed_ms = (perf_counter() - t0) * 1000

            t1 = perf_counter()
            try:
                results = self.zeusdb_index.search(
                    vector=q_vec,
                    filter=kwargs.get("filter"),
                    top_k=k,
                    ef_search=kwargs.get("ef_search"),
                    return_vector=False,
                )
                search_ms = (perf_counter() - t1) * 1000

                docs = [self._zeusdb_result_to_document(r) for r in results]

                logger.info(
                    "Similarity search completed",
                    operation="similarity_search",
                    results_count=len(results),
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    top_k=k
                )

                return docs

            except Exception as e:
                search_ms = (perf_counter() - t1) * 1000
                logger.error(
                    "ZeusDB similarity_search failed",
                    operation="similarity_search",
                    top_k=k,
                    has_filter=has_filter,
                    ef_search=ef_search,
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                return []

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Search with raw distance scores (backend distances)."""
        logger.debug(
            "Performing similarity search with scores",
            operation="similarity_search_with_score",
            query_length=len(query),
            top_k=k,
            has_filter=kwargs.get("filter") is not None
        )

        with operation_context("similarity_search_with_score", top_k=k):
            t0 = perf_counter()
            q_vec = self.embedding.embed_query(query)
            embed_ms = (perf_counter() - t0) * 1000

            t1 = perf_counter()
            try:
                results = self.zeusdb_index.search(
                    vector=q_vec,
                    filter=kwargs.get("filter"),
                    top_k=k,
                    ef_search=kwargs.get("ef_search"),
                    return_vector=False,
                )
                search_ms = (perf_counter() - t1) * 1000

                out: List[Tuple[Document, float]] = []
                for r in results:
                    out.append((self._zeusdb_result_to_document(r), float(r.get("score", 0.0))))

                logger.info(
                    "Similarity search with score completed",
                    operation="similarity_search_with_score",
                    results_count=len(results),
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    top_k=k
                )

                return out

            except Exception as e:
                search_ms = (perf_counter() - t1) * 1000
                logger.error(
                    "ZeusDB similarity_search_with_score failed",
                    operation="similarity_search_with_score",
                    top_k=k,
                    has_filter=kwargs.get("filter") is not None,
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                return []

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs and normalized relevance scores in [0, 1]."""
        rel_fn = self._select_relevance_score_fn()
        pairs = self.similarity_search_with_score(query, k=k, **kwargs)
        return [(doc, rel_fn(raw_distance)) for doc, raw_distance in pairs]

    def similarity_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search using a vector directly (list/tuple/np.ndarray/etc.)."""
        vec = self._coerce_vector(embedding, where="similarity_search_by_vector")
        if vec is None:
            return []

        logger.debug(
            "Performing similarity search by vector",
            operation="similarity_search_by_vector",
            vector_dim=len(vec),
            top_k=k,
            has_filter=kwargs.get("filter") is not None
        )

        with operation_context("similarity_search_by_vector", top_k=k):
            t0 = perf_counter()
            try:
                results = self.zeusdb_index.search(
                    vector=vec,
                    filter=kwargs.get("filter"),
                    top_k=k,
                    ef_search=kwargs.get("ef_search"),
                    return_vector=False,
                )
                search_ms = (perf_counter() - t0) * 1000

                docs = [self._zeusdb_result_to_document(r) for r in results]

                logger.info(
                    "Similarity search by vector completed",
                    operation="similarity_search_by_vector",
                    results_count=len(results),
                    search_ms=search_ms,
                    top_k=k
                )

                return docs

            except Exception as e:
                search_ms = (perf_counter() - t0) * 1000
                logger.error(
                    "ZeusDB similarity_search_by_vector failed",
                    operation="similarity_search_by_vector",
                    top_k=k,
                    has_filter=kwargs.get("filter") is not None,
                    search_ms=search_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                return []

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Get documents by IDs, preserving the input order."""
        if not ids:
            return []

        logger.debug(
            "Getting documents by IDs",
            operation="get_by_ids",
            id_count=len(ids)
        )

        try:
            records = self.zeusdb_index.get_records(list(ids), return_vector=False)
        except Exception as e:
            logger.error(
                "ZeusDB get_by_ids failed",
                operation="get_by_ids",
                id_count=len(ids),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return []

        by_id: Dict[str, Document] = {}
        for rec in records:
            metadata = (rec.get("metadata") or {}).copy()
            page_content = metadata.pop(self.text_key, rec.get("id", ""))
            did = rec.get("id")
            by_id[str(did)] = Document(id=did, page_content=str(page_content), metadata=metadata)

        result_docs = [by_id[str(i)] for i in ids if str(i) in by_id]

        logger.debug(
            "Get by IDs completed",
            operation="get_by_ids",
            requested_count=len(ids),
            found_count=len(result_docs)
        )

        return result_docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by IDs."""
        if not ids:
            return False

        logger.info(
            "Deleting documents",
            operation="delete",
            id_count=len(ids)
        )

        success_count = 0
        for did in ids:
            try:
                res = self.zeusdb_index.remove_point(did)
                if bool(res):
                    success_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to delete document",
                    operation="delete",
                    document_id=did,
                    error=str(e)
                )
                if kwargs.get("verbose"):
                    logger.debug(
                        "Delete error details",
                        operation="delete",
                        document_id=did,
                        error_type=type(e).__name__,
                        exc_info=True
                    )

        all_successful = success_count == len(ids)
        logger.info(
            "Delete operation completed",
            operation="delete",
            requested_count=len(ids),
            success_count=success_count,
            all_successful=all_successful
        )

        return all_successful

    # -------------------------------------------------------------------------
    # MMR (sync)
    # -------------------------------------------------------------------------
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Maximal Marginal Relevance search with proper diversity scoring."""
        if k >= fetch_k:
            return self.similarity_search(query, k=k, **kwargs)

        logger.debug(
            "Performing MMR search",
            operation="mmr_search",
            query_length=len(query),
            top_k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            has_filter=kwargs.get("filter") is not None
        )

        with operation_context("mmr_search", top_k=k, fetch_k=fetch_k):
            t0 = perf_counter()
            q_vec = self.embedding.embed_query(query)
            embed_ms = (perf_counter() - t0) * 1000

            t1 = perf_counter()
            try:
                results = self.zeusdb_index.search(
                    vector=q_vec,
                    filter=kwargs.get("filter"),
                    top_k=fetch_k,
                    ef_search=kwargs.get("ef_search"),
                    return_vector=True,
                )
                search_ms = (perf_counter() - t1) * 1000

                candidates: List[Tuple[Document, float, Optional[List[float]]]] = []
                for r in results:
                    doc = self._zeusdb_result_to_document(r)
                    raw_distance = float(r.get("score", 0.0))
                    vector = r.get("vector")
                    candidates.append((doc, raw_distance, vector))

                if len(candidates) <= k:
                    logger.debug(
                        "MMR search: candidates <= k, returning all",
                        operation="mmr_search",
                        candidates_count=len(candidates),
                        requested_k=k
                    )
                    return [d for d, _, _ in candidates]

                # MMR selection algorithm with proper diversity scoring
                t2 = perf_counter()
                rel_fn = self._select_relevance_score_fn()
                selected: List[Tuple[Document, List[float]]] = []
                remaining = candidates.copy()

                # Seed with best by relevance (lowest distance)
                best_doc, _, best_vec = min(remaining, key=lambda ds: ds[1])
                if best_vec is None:
                    best_vec = self._embed_doc_text(best_doc.page_content)
                selected.append((best_doc, best_vec))
                remaining = [(d, s, v) for d, s, v in remaining if d.id != best_doc.id]

                while len(selected) < k and remaining:
                    scores: List[Tuple[Document, float]] = []
                    for doc, raw_distance, doc_vec in remaining:
                        sim = rel_fn(raw_distance)  # [0,1]

                        if doc_vec is None:
                            doc_vec = self._embed_doc_text(doc.page_content)

                        # Diversity: min cosine distance to any already selected doc, normalize [0,2] -> [0,1]
                        min_div01 = float("inf")
                        for _, sel_vec in selected:
                            div = self._cosine_distance(doc_vec, sel_vec)  # [0,2]
                            div01 = div * 0.5
                            if div01 < min_div01:
                                min_div01 = div01

                        mmr = lambda_mult * sim + (1.0 - lambda_mult) * min_div01
                        scores.append((doc, mmr))

                    # Pick highest MMR
                    best_doc, _ = max(scores, key=lambda dm: dm[1])
                    best_vec = None
                    for d, _, v in remaining:
                        if d.id == best_doc.id:
                            best_vec = v
                            break
                    if best_vec is None:
                        best_vec = self._embed_doc_text(best_doc.page_content)

                    selected.append((best_doc, best_vec))
                    remaining = [(d, s, v) for d, s, v in remaining if d.id != best_doc.id]

                mmr_ms = (perf_counter() - t2) * 1000
                result_docs = [doc for doc, _ in selected]

                logger.info(
                    "MMR search completed",
                    operation="mmr_search",
                    results_count=len(result_docs),
                    candidates_count=len(candidates),
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    mmr_ms=mmr_ms,
                    top_k=k,
                    fetch_k=fetch_k
                )

                return result_docs

            except Exception as e:
                search_ms = (perf_counter() - t1) * 1000
                logger.error(
                    "ZeusDB MMR search failed",
                    operation="mmr_search",
                    top_k=k,
                    fetch_k=fetch_k,
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                return []

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """MMR search using embedding vector directly."""
        if k >= fetch_k:
            return self.similarity_search_by_vector(embedding, k=k, **kwargs)

        vec = self._coerce_vector(embedding, where="max_marginal_relevance_search_by_vector")
        if vec is None:
            return []

        logger.debug(
            "Performing MMR search by vector",
            operation="mmr_search_by_vector",
            vector_dim=len(vec),
            top_k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            has_filter=kwargs.get("filter") is not None
        )

        with operation_context("mmr_search_by_vector", top_k=k, fetch_k=fetch_k):
            t0 = perf_counter()
            try:
                results = self.zeusdb_index.search(
                    vector=vec,
                    filter=kwargs.get("filter"),
                    top_k=fetch_k,
                    ef_search=kwargs.get("ef_search"),
                    return_vector=True,
                )
                search_ms = (perf_counter() - t0) * 1000

                candidates: List[Tuple[Document, float, Optional[List[float]]]] = []
                for r in results:
                    doc = self._zeusdb_result_to_document(r)
                    raw_distance = float(r.get("score", 0.0))
                    vector = r.get("vector")
                    candidates.append((doc, raw_distance, vector))

                if len(candidates) <= k:
                    return [d for d, _, _ in candidates]

                # MMR selection algorithm
                t1 = perf_counter()
                rel_fn = self._select_relevance_score_fn()
                selected: List[Tuple[Document, List[float]]] = []
                remaining = candidates.copy()

                # Seed with best by relevance
                best_doc, _, best_vec = min(remaining, key=lambda ds: ds[1])
                if best_vec is None:
                    best_vec = self._embed_doc_text(best_doc.page_content)
                selected.append((best_doc, best_vec))
                remaining = [(d, s, v) for d, s, v in remaining if d.id != best_doc.id]

                while len(selected) < k and remaining:
                    scores: List[Tuple[Document, float]] = []
                    for doc, raw_distance, doc_vec in remaining:
                        sim = rel_fn(raw_distance)

                        if doc_vec is None:
                            doc_vec = self._embed_doc_text(doc.page_content)

                        min_div01 = float("inf")
                        for _, sel_vec in selected:
                            div = self._cosine_distance(doc_vec, sel_vec)
                            div01 = div * 0.5
                            if div01 < min_div01:
                                min_div01 = div01

                        mmr = lambda_mult * sim + (1.0 - lambda_mult) * min_div01
                        scores.append((doc, mmr))

                    best_doc, _ = max(scores, key=lambda dm: dm[1])
                    best_vec = None
                    for d, _, v in remaining:
                        if d.id == best_doc.id:
                            best_vec = v
                            break
                    if best_vec is None:
                        best_vec = self._embed_doc_text(best_doc.page_content)

                    selected.append((best_doc, best_vec))
                    remaining = [(d, s, v) for d, s, v in remaining if d.id != best_doc.id]

                mmr_ms = (perf_counter() - t1) * 1000
                result_docs = [doc for doc, _ in selected]

                logger.info(
                    "MMR search by vector completed",
                    operation="mmr_search_by_vector",
                    results_count=len(result_docs),
                    candidates_count=len(candidates),
                    search_ms=search_ms,
                    mmr_ms=mmr_ms,
                    top_k=k,
                    fetch_k=fetch_k
                )

                return result_docs

            except Exception as e:
                search_ms = (perf_counter() - t0) * 1000
                logger.error(
                    "ZeusDB MMR search by vector failed",
                    operation="mmr_search_by_vector",
                    top_k=k,
                    fetch_k=fetch_k,
                    search_ms=search_ms,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                return []

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str], 
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        zeusdb_index: Optional[Any] = None,
        validate_dims: bool = True, 
        **kwargs: Any,
    ) -> "ZeusDBVectorStore":
        """Create VectorStore from texts."""
        from .vector_database import VectorDatabase

        # Coerce once (avoids exhausting generators and re-iterating)
        texts_list = list(texts)

        logger.info(
            "Creating ZeusDB VectorStore from texts",
            operation="from_texts",
            text_count=len(texts_list),
            has_existing_index=zeusdb_index is not None,
            validate_dims=validate_dims
        )

        # Determine embedding dimension from a cheap single embed
        test_vec = embedding.embed_query("test")
        dim = len(test_vec)

        # If an index is provided, sanity-check it; otherwise create one with `dim`
        if zeusdb_index is not None:
            idx_dim = cls._extract_index_dim(zeusdb_index)
            if idx_dim is not None and idx_dim != dim:
                logger.error(
                    "Index dimension mismatch",
                    operation="from_texts",
                    embedding_dim=dim,
                    index_dim=idx_dim
                )
                raise ValueError(
                    f"Embedding dimension ({dim}) does not match index dimension ({idx_dim}). "
                    "Use a compatible embedding model or recreate the index."
                )
        else:
            # Create a fresh index using the derived dimension
            logger.debug(
                "Creating new ZeusDB index",
                operation="from_texts",
                dim=dim,
                index_type=kwargs.get("index_type", "hnsw"),
                space=kwargs.get("space", "cosine")
            )
            
            vdb = VectorDatabase()
            zeusdb_index = vdb.create(
                index_type=kwargs.get("index_type", "hnsw"),
                dim=dim,
                space=kwargs.get("space", "cosine"),
                m=kwargs.get("m", 16),
                ef_construction=kwargs.get("ef_construction", 200),
                expected_size=kwargs.get("expected_size", max(10000, len(texts_list) * 2)),
                quantization_config=kwargs.get("quantization_config"),
            )
            
        # Validate a small sample of document embeddings match `dim`
        if validate_dims:
            logger.debug(
                "Validating embedding dimensions",
                operation="from_texts",
                expected_dim=dim
            )
            cls._validate_embedding_dims(embedding, texts_list, expected_dim=dim)

        store = cls(zeusdb_index=zeusdb_index, embedding=embedding, text_key=kwargs.get("text_key", "text"))
        if texts_list:
            store.add_texts(texts_list, metadatas, ids=ids)
        
        logger.info(
            "ZeusDB VectorStore from texts completed",
            operation="from_texts",
            final_text_count=len(texts_list)
        )
        
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        *,
        zeusdb_index: Optional[Any] = None,
        **kwargs: Any,
    ) -> "ZeusDBVectorStore":
        """Create VectorStore from Documents."""
        logger.info(
            "Creating ZeusDB VectorStore from documents",
            operation="from_documents",
            doc_count=len(documents),
            has_existing_index=zeusdb_index is not None
        )

        texts = [d.page_content for d in documents]
        metadatas = [dict(d.metadata or {}) for d in documents]

        if all(d.id is not None for d in documents):
            ids_arg: Optional[List[str]] = [str(d.id) for d in documents]
        else:
            ids_arg = None

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids_arg,
            zeusdb_index=zeusdb_index,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Async wrappers (use keyword args so signatures line up for Pylance)
    # -------------------------------------------------------------------------
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of add_texts."""
        return await run_in_executor(None, self.add_texts, texts, metadatas, ids=ids, **kwargs)

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Async version of add_documents."""
        return await run_in_executor(None, self.add_documents, documents, **kwargs)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Async version of similarity_search."""
        return await run_in_executor(None, self.similarity_search, query, k=k, **kwargs)

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Async version of similarity_search_with_score."""
        return await run_in_executor(None, self.similarity_search_with_score, query, k=k, **kwargs)

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Async version of similarity_search_with_relevance_scores."""
        return await run_in_executor(None, self.similarity_search_with_relevance_scores, query, k=k, **kwargs)

    async def asimilarity_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of similarity_search_by_vector."""
        return await run_in_executor(None, self.similarity_search_by_vector, embedding, k=k, **kwargs)

    async def aget_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Async version of get_by_ids."""
        return await run_in_executor(None, self.get_by_ids, ids)

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Async version of delete."""
        return await run_in_executor(None, self.delete, ids, **kwargs)

    async def amax_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> List[Document]:
        """Async version of max_marginal_relevance_search."""
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of max_marginal_relevance_search_by_vector."""
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # ZeusDB-specific utility methods
    # -------------------------------------------------------------------------
    def get_zeusdb_stats(self) -> Dict[str, Any]:
        """Get ZeusDB index statistics."""
        try:
            stats = self.zeusdb_index.get_stats()
            logger.debug(
                "Retrieved ZeusDB stats",
                operation="get_zeusdb_stats",
                stats_keys=list(stats.keys()) if stats else []
            )
            return stats
        except Exception as e:
            logger.error(
                "Failed to get ZeusDB stats",
                operation="get_zeusdb_stats",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return {}

    def get_quantization_info(self) -> Optional[Dict[str, Any]]:
        """Get quantization information if enabled."""
        try:
            info = self.zeusdb_index.get_quantization_info()
            logger.debug(
                "Retrieved quantization info",
                operation="get_quantization_info",
                has_quantization=info is not None
            )
            return info
        except Exception as e:
            logger.error(
                "Failed to get quantization info",
                operation="get_quantization_info",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return None

    def get_zeusdb_performance_info(self) -> Dict[str, Any]:
        """Get ZeusDB performance characteristics."""
        try:
            info = self.zeusdb_index.get_performance_info()
            logger.debug(
                "Retrieved performance info",
                operation="get_zeusdb_performance_info",
                info_keys=list(info.keys()) if info else []
            )
            return info
        except Exception as e:
            logger.error(
                "Failed to get performance info",
                operation="get_zeusdb_performance_info",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return {}

    def save_index(self, path: str) -> bool:
        """Save ZeusDB index to disk with complete state preservation."""
        logger.info(
            "Saving ZeusDB index",
            operation="save_index",
            path=path
        )
        
        try:
            self.zeusdb_index.save(path)
            logger.info(
                "ZeusDB index saved successfully",
                operation="save_index",
                path=path
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to save ZeusDB index",
                operation="save_index",
                path=path,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return False

    @classmethod
    def load_index(
        cls,
        path: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "ZeusDBVectorStore":
        """Load ZeusDB index from disk."""
        from .vector_database import VectorDatabase

        logger.info(
            "Loading ZeusDB index",
            operation="load_index",
            path=path
        )

        try:
            vdb = VectorDatabase()
            zeusdb_index = vdb.load(path)

            text_key = kwargs.pop("text_key", "text")
            store = cls(
                zeusdb_index=zeusdb_index,
                embedding=embedding,
                text_key=text_key,
                **kwargs,
            )
            
            logger.info(
                "ZeusDB index loaded successfully",
                operation="load_index",
                path=path,
                vector_count=store.get_vector_count()
            )
            
            return store
        except Exception as e:
            logger.error(
                "Failed to load ZeusDB index",
                operation="load_index",
                path=path,
                exc_info=True
            )
            raise RuntimeError(f"Failed to load ZeusDB index from {path}: {e}") from e

    def benchmark_search_performance(
        self,
        query_count: int = 100,
        max_threads: Optional[int] = None,
    ) -> Dict[str, float]:
        """Benchmark search performance with concurrent operations."""
        logger.info(
            "Starting search performance benchmark",
            operation="benchmark_search_performance",
            query_count=query_count,
            max_threads=max_threads
        )
        
        try:
            results = self.zeusdb_index.benchmark_concurrent_reads(query_count, max_threads)
            logger.info(
                "Search performance benchmark completed",
                operation="benchmark_search_performance",
                query_count=query_count,
                sequential_qps=results.get("sequential_qps", 0),
                parallel_qps=results.get("parallel_qps", 0),
                speedup=results.get("speedup", 0)
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to benchmark search performance",
                operation="benchmark_search_performance",
                query_count=query_count,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return {}

    def get_vector_count(self) -> int:
        """Get total number of vectors in the index."""
        try:
            count = self.zeusdb_index.get_vector_count()
            logger.debug(
                "Retrieved vector count",
                operation="get_vector_count",
                count=count
            )
            return count
        except Exception as e:
            logger.error(
                "Failed to get vector count",
                operation="get_vector_count",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return 0

    def get_training_progress(self) -> float:
        """Get quantization training progress percentage."""
        try:
            progress = self.zeusdb_index.get_training_progress()
            logger.debug(
                "Retrieved training progress",
                operation="get_training_progress",
                progress_percent=progress
            )
            return progress
        except Exception as e:
            logger.error(
                "Failed to get training progress",
                operation="get_training_progress",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return 0.0

    def is_quantized(self) -> bool:
        """Whether quantized search is active."""
        try:
            quantized = self.zeusdb_index.is_quantized()
            logger.debug(
                "Retrieved quantization status",
                operation="is_quantized",
                is_quantized=quantized
            )
            return quantized
        except Exception as e:
            logger.error(
                "Failed to check quantization status",
                operation="is_quantized",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return False

    def can_use_quantization(self) -> bool:
        """Whether quantization is available (e.g., PQ trained)."""
        try:
            available = self.zeusdb_index.can_use_quantization()
            logger.debug(
                "Retrieved quantization availability",
                operation="can_use_quantization",
                can_use_quantization=available
            )
            return available
        except Exception as e:
            logger.error(
                "Failed to check quantization availability",
                operation="can_use_quantization",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return False

    def get_storage_mode(self) -> str:
        """Get current storage mode string (e.g., 'raw_only', 'quantized_active')."""
        try:
            mode = self.zeusdb_index.get_storage_mode()
            logger.debug(
                "Retrieved storage mode",
                operation="get_storage_mode",
                storage_mode=mode
            )
            return mode
        except Exception as e:
            logger.error(
                "Failed to get storage mode",
                operation="get_storage_mode",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            return "unknown"

    def info(self) -> str:
        """Get a human-readable info string about the index."""
        try:
            info_str = self.zeusdb_index.info()
            logger.debug(
                "Retrieved index info",
                operation="info",
                info_length=len(info_str)
            )
            return info_str
        except Exception as e:
            logger.error(
                "Failed to get index info",
                operation="info",
                exc_info=True
            )
            return f"ZeusDBVectorStore(error: {e})"


# Optional async-optimized subclass (placeholder for future native async impls)
class AsyncZeusDBVectorStore(ZeusDBVectorStore):
    """Async-optimized version of ZeusDBVectorStore."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(
            "AsyncZeusDBVectorStore initialized",
            operation="async_init"
        )
        # Reserved for future native-async optimizations


# -------------------------------------------------------------------------
# Utility functions and convenience creators
# -------------------------------------------------------------------------
def is_langchain_available() -> bool:
    """True if langchain-core is importable."""
    return LANGCHAIN_AVAILABLE


def get_langchain_requirements() -> List[str]:
    """Package requirements for LangChain integration."""
    return ["langchain-core>=0.3.74,<0.4"]


def create_zeusdb_vectorstore(
    texts: Optional[List[str]] = None,
    documents: Optional[List[Document]] = None, 
    embedding: Optional[Embeddings] = None,
    **zeusdb_kwargs: Any,
) -> ZeusDBVectorStore:
    """Convenience function to create ZeusDB vector store."""
    if embedding is None:
        logger.error(
            "Missing embedding function",
            operation="create_zeusdb_vectorstore"
        )
        raise ValueError("embedding function is required")

    logger.info(
        "Creating ZeusDB vectorstore via convenience function",
        operation="create_zeusdb_vectorstore",
        has_texts=texts is not None,
        has_documents=documents is not None,
        text_count=len(texts) if texts else 0,
        doc_count=len(documents) if documents else 0
    )

    if documents is not None:
        return ZeusDBVectorStore.from_documents(
            documents=documents,
            embedding=embedding,
            **zeusdb_kwargs,
        )
    elif texts is not None:
        return ZeusDBVectorStore.from_texts(
            texts=texts,
            embedding=embedding,
            **zeusdb_kwargs,
        )
    else:
        from .vector_database import VectorDatabase

        test_embedding = embedding.embed_query("test")
        dim = len(test_embedding)

        zeusdb_kwargs.setdefault("index_type", "hnsw")
        zeusdb_kwargs.setdefault("space", "cosine")
        zeusdb_kwargs.setdefault("m", 16)
        zeusdb_kwargs.setdefault("ef_construction", 200)
        zeusdb_kwargs.setdefault("expected_size", 10000)

        vdb = VectorDatabase()
        zeusdb_index = vdb.create(dim=dim, **zeusdb_kwargs)

        return ZeusDBVectorStore(zeusdb_index=zeusdb_index, embedding=embedding)


# Backwards compatibility alias
ZeusDBLangChain = ZeusDBVectorStore  # Primary alias for user imports
