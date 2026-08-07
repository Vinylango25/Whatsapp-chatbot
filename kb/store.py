"""
WC KB — Qdrant Vector Store Manager
Uses local embedded Qdrant (no Docker/server needed).
Each tenant gets its own collection: wc_{tenant_id}_kb
"""
from __future__ import annotations
import os, logging
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("wc.kb.store")

QDRANT_PATH    = os.getenv("QDRANT_PATH", "./qdrant_storage")
CACHE_SUFFIX   = "_cache"
DEFAULT_DIM    = int(os.getenv("EMBEDDING_DIM", "1536"))


class VectorStore:
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        storage = Path(QDRANT_PATH).resolve()
        storage.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(storage))
        log.info(f"[store] Qdrant local storage: {storage}")

    def kb_collection(self, tenant_id: str) -> str:
        return f"wc_{tenant_id}_kb"

    def cache_collection(self, tenant_id: str) -> str:
        return f"wc_{tenant_id}_cache"

    def ensure_collection(self, name: str, dim: int = DEFAULT_DIM):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self.client.get_collections().collections]
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info(f"[store] Created collection: {name}")

    def upsert(self, collection: str, points: list):
        """Upsert a list of PointStruct into the collection."""
        from qdrant_client.models import PointStruct
        self.client.upsert(collection_name=collection, points=points)

    def search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        score_threshold: float = 0.0,
    ) -> list:
        """Semantic search. Returns list of ScoredPoint."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = []
            for key, val in filters.items():
                if val:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            if conditions:
                from qdrant_client.models import Filter as QFilter
                qdrant_filter = QFilter(must=conditions)

        try:
            return self.client.search(
                collection_name=collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as e:
            log.warning(f"[store] search error in {collection}: {e}")
            return []

    def delete_by_doc(self, collection: str, doc_name: str):
        """Delete all chunks from a specific document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self.client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
            ),
        )

    def count(self, collection: str) -> int:
        try:
            return self.client.get_collection(collection).points_count or 0
        except Exception:
            return 0

    def list_docs(self, collection: str) -> list[str]:
        """List unique document names in a collection."""
        try:
            result, _ = self.client.scroll(
                collection_name=collection,
                limit=10000,
                with_payload=["doc_name"],
            )
            return list({p.payload.get("doc_name") for p in result if p.payload.get("doc_name")})
        except Exception:
            return []


# Singleton
_store: VectorStore | None = None

def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
