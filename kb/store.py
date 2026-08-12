"""
WC KB — Pinecone Vector Store Manager

Single Pinecone index with namespaces per tenant:
  KB:    namespace = "{tenant_id}_kb"
  Cache: namespace = "{tenant_id}_cache"

Requires env vars:
  PINECONE_API_KEY   — from https://app.pinecone.io
  PINECONE_INDEX     — index name you create (e.g. "wc-kb")
"""
from __future__ import annotations
import os, logging, uuid
from typing import List, Optional

log = logging.getLogger("wc.kb.store")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "wc-kb")
DEFAULT_DIM      = int(os.getenv("EMBEDDING_DIM", "1536"))


class VectorStore:
    def __init__(self):
        from pinecone import Pinecone, ServerlessSpec

        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file.")

        self._pc    = Pinecone(api_key=PINECONE_API_KEY)
        self._index = self._get_or_create_index()
        log.info(f"[store] Pinecone index: {PINECONE_INDEX}")

    def _get_or_create_index(self):
        from pinecone import ServerlessSpec
        existing = [i.name for i in self._pc.list_indexes()]
        if PINECONE_INDEX not in existing:
            log.info(f"[store] Creating Pinecone index: {PINECONE_INDEX}")
            self._pc.create_index(
                name=PINECONE_INDEX,
                dimension=DEFAULT_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            log.info(f"[store] Index {PINECONE_INDEX} created.")
        return self._pc.Index(PINECONE_INDEX)

    # ── Namespace helpers ─────────────────────────────────────────────────────

    def kb_collection(self, tenant_id: str) -> str:
        return f"{tenant_id}_kb"

    def cache_collection(self, tenant_id: str) -> str:
        return f"{tenant_id}_cache"

    def ensure_collection(self, name: str, dim: int = DEFAULT_DIM):
        """No-op for Pinecone — namespaces are created on first upsert."""
        pass

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, collection: str, points: list):
        """
        Upsert vectors into a Pinecone namespace.
        `points` is a list of dicts: {id, values, metadata}
        """
        # Batch in chunks of 100 (Pinecone limit per request)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._index.upsert(vectors=batch, namespace=collection)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        score_threshold: float = 0.0,
    ) -> list:
        """
        Semantic search. Returns list of result objects with .id, .score, .payload.
        """
        pinecone_filter = None
        if filters:
            pinecone_filter = {k: {"$eq": v} for k, v in filters.items() if v}

        try:
            resp = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=collection,
                filter=pinecone_filter,
                include_metadata=True,
            )
            # Wrap Pinecone matches into objects that look like Qdrant ScoredPoints
            results = []
            for match in resp.matches:
                if match.score >= score_threshold:
                    results.append(_PineconeResult(match.id, match.score, match.metadata))
            return results
        except Exception as e:
            log.warning(f"[store] Pinecone search error in {collection}: {e}")
            return []

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_doc(self, collection: str, doc_name: str):
        """Delete all vectors for a specific document."""
        try:
            # Pinecone supports filter-based delete on serverless
            self._index.delete(
                filter={"doc_name": {"$eq": doc_name}},
                namespace=collection,
            )
        except Exception as e:
            log.warning(f"[store] delete_by_doc error: {e}")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self, collection: str) -> int:
        try:
            stats = self._index.describe_index_stats()
            ns    = stats.namespaces.get(collection)
            return ns.vector_count if ns else 0
        except Exception:
            return 0

    def list_docs(self, collection: str) -> list[str]:
        """
        Pinecone doesn't support full metadata scans on free tier.
        We store doc names in a lightweight set via a meta-vector trick.
        Returns empty list on free tier — use /kb/stats for chunk counts instead.
        """
        try:
            # Query with a zero vector to get top results and extract doc names
            zero = [0.0] * DEFAULT_DIM
            resp = self._index.query(
                vector=zero,
                top_k=10000,
                namespace=collection,
                include_metadata=True,
            )
            return list({m.metadata.get("doc_name") for m in resp.matches if m.metadata.get("doc_name")})
        except Exception:
            return []

    def set_payload(self, collection: str, point_id: str, payload: dict):
        """Update metadata on an existing vector (used by cache hit counter)."""
        try:
            self._index.update(id=point_id, set_metadata=payload, namespace=collection)
        except Exception as e:
            log.warning(f"[store] set_payload error: {e}")


class _PineconeResult:
    """Thin wrapper to match the Qdrant ScoredPoint interface used in retriever.py."""
    def __init__(self, id: str, score: float, metadata: dict):
        self.id      = id
        self.score   = score
        self.payload = metadata or {}


# Singleton
_store: VectorStore | None = None

def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
