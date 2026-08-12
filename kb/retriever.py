"""
WC KB — Retriever
Semantic search with semantic cache (avoids repeated LLM calls for similar queries).
"""
from __future__ import annotations
import os, uuid, time, logging
from typing import List, Optional

from kb.embedder import get_embedder
from kb.store    import get_store

log = logging.getLogger("wc.kb.retriever")

SCORE_THRESHOLD   = float(os.getenv("SCORE_THRESHOLD",   "0.3"))
CACHE_THRESHOLD   = float(os.getenv("CACHE_THRESHOLD",   "0.92"))
TOP_K_DEFAULT     = int(os.getenv("TOP_K", "4"))


async def retrieve(
    query: str,
    tenant_id: str,
    category: str  = "",
    top_k: int     = TOP_K_DEFAULT,
    use_cache: bool = True,
) -> dict:
    """
    Retrieve relevant chunks for a query.
    Returns: {context, answer (if cached), source, score}
    """
    store    = get_store()
    embedder = get_embedder()
    kb_col   = store.kb_collection(tenant_id)
    cache_col = store.cache_collection(tenant_id)

    # Ensure collections exist
    store.ensure_collection(kb_col,    dim=embedder.dim)
    store.ensure_collection(cache_col, dim=embedder.dim)

    # 1. Embed the query
    query_vector = await embedder.embed(query)

    # 2. Check semantic cache first
    if use_cache:
        cache_hits = store.search(
            collection=cache_col,
            vector=query_vector,
            top_k=1,
            score_threshold=CACHE_THRESHOLD,
        )
        if cache_hits:
            hit     = cache_hits[0]
            payload = hit.payload
            log.info(f"[retriever] Cache hit (score={hit.score:.3f}) for: {query[:60]}")
            # Update hit count via Pinecone metadata update
            try:
                store.set_payload(cache_col, hit.id, {
                    "hit_count":     payload.get("hit_count", 0) + 1,
                    "last_accessed": int(time.time()),
                })
            except Exception:
                pass
            # context was serialised as JSON string on write — deserialise it
            import json
            raw_context = payload.get("context", "[]")
            try:
                context_list = json.loads(raw_context) if isinstance(raw_context, str) else raw_context
            except Exception:
                context_list = []
            return {
                "query":            query,
                "query_vector":     query_vector,
                "context":          context_list,
                "answer":           payload.get("answer"),
                "source":           "semantic_cache_hit",
                "similarity_score": hit.score,
            }

    # 3. Search the knowledge base
    filters = {"tenant_id": tenant_id}
    if category:
        filters["category"] = category

    results = store.search(
        collection=kb_col,
        vector=query_vector,
        top_k=top_k,
        filters=filters,
        score_threshold=SCORE_THRESHOLD,
    )

    context = [
        {
            "text":     r.payload.get("text", ""),
            "doc_name": r.payload.get("doc_name", ""),
            "page_num": r.payload.get("page_num"),
            "score":    round(r.score, 4),
        }
        for r in results
    ]

    log.info(f"[retriever] {len(context)} chunks retrieved for: {query[:60]}")

    return {
        "query":            query,
        "query_vector":     query_vector,
        "context":          context,
        "answer":           None,
        "source":           "kb_retrieval",
        "similarity_score": context[0]["score"] if context else 0.0,
    }


async def cache_answer(
    query: str,
    query_vector: List[float],
    tenant_id: str,
    context: list,
    answer: str,
):
    """Store a query+answer in the semantic cache for future reuse."""
    store     = get_store()
    embedder  = get_embedder()
    cache_col = store.cache_collection(tenant_id)
    store.ensure_collection(cache_col, dim=embedder.dim)

    import json
    store.upsert(cache_col, [{
        "id":     str(uuid.uuid4()),
        "values": query_vector,
        "metadata": {
            "query":         query,
            "context":       json.dumps(context),  # Pinecone metadata must be scalar — serialise list
            "answer":        answer,
            "tenant_id":     tenant_id,
            "hit_count":     0,
            "created_at":    int(time.time()),
            "last_accessed": int(time.time()),
        },
    }])
    log.info(f"[retriever] Cached answer for: {query[:60]}")
