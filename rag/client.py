"""
WC RAG Client — Internal KB
Uses the built-in Qdrant KB (no external Karisma needed).
Falls back to external Karisma if KARISMA_URL is set and internal KB is empty.
"""
from __future__ import annotations
import os, logging

log = logging.getLogger("wc.rag")

KARISMA_URL   = os.getenv("KARISMA_URL", "")   # set this to use external Karisma as fallback
KARISMA_TOKEN = os.getenv("KARISMA_TOKEN", "")


async def karisma_retrieve(
    query: str,
    tenant_id: str,
    category: str = "",
    top_k: int    = 4,
    generate: bool = False,
) -> dict:
    """
    Primary: use internal KB (Qdrant embedded).
    Fallback: call external Karisma if KARISMA_URL is set and internal KB is empty.
    """
    from kb.retriever import retrieve
    from kb.store     import get_store

    store  = get_store()
    kb_col = store.kb_collection(tenant_id)

    # Check if internal KB has content
    kb_count = store.count(kb_col)

    if kb_count > 0:
        # Use internal KB
        return await retrieve(
            query=query,
            tenant_id=tenant_id,
            category=category,
            top_k=top_k,
            use_cache=True,
        )

    # Fallback to external Karisma if configured
    if KARISMA_URL:
        log.info(f"[rag] Internal KB empty for {tenant_id} — falling back to Karisma at {KARISMA_URL}")
        return await _karisma_external(query, tenant_id, category, top_k, generate)

    # No KB content at all
    log.warning(f"[rag] No KB content for tenant {tenant_id}. Upload documents via /kb/upload")
    return {"context": [], "answer": None, "source": "empty_kb"}


async def _karisma_external(
    query: str,
    tenant_id: str,
    category: str,
    top_k: int,
    generate: bool,
) -> dict:
    """Call external Karisma as fallback."""
    import httpx
    payload = {
        "query": query, "tenant_id": tenant_id,
        "category": category, "top_k": top_k, "generate": generate,
    }
    headers = {}
    if KARISMA_TOKEN:
        headers["Authorization"] = f"Bearer {KARISMA_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{KARISMA_URL}/retrieve", json=payload, headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        log.warning(f"[rag] External Karisma error: {e}")
        return {"context": [], "answer": None, "source": "error"}


def format_context_for_llm(chunks: list[dict]) -> str:
    """Format KB chunks into a clean string for the LLM prompt."""
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        doc  = chunk.get("doc_name", "Unknown")
        page = chunk.get("page_num")
        text = chunk.get("text", "").strip()
        ref  = f"{doc}" + (f", p.{page}" if page else "")
        parts.append(f"[{i}] Source: {ref}\n{text}")
    return "\n\n".join(parts)
