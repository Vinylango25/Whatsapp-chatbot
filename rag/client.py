"""
WC — Karisma RAG Client
Calls the Karisma /retrieve endpoint to get relevant KB context.
"""
from __future__ import annotations
import os, logging
import httpx

log = logging.getLogger("wc.rag")

KARISMA_URL   = os.getenv("KARISMA_URL", "http://localhost:8001")
KARISMA_TOKEN = os.getenv("KARISMA_TOKEN", "")


async def karisma_retrieve(
    query: str,
    tenant_id: str,
    category: str = "",
    top_k: int = 4,
    generate: bool = False,
) -> dict:
    """
    Call Karisma /retrieve and return the result dict.
    Falls back to empty context on any error.
    """
    payload = {
        "query":     query,
        "tenant_id": tenant_id,
        "category":  category,
        "top_k":     top_k,
        "generate":  generate,
    }
    headers = {}
    if KARISMA_TOKEN:
        headers["Authorization"] = f"Bearer {KARISMA_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{KARISMA_URL}/retrieve",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        log.warning(f"[rag] Karisma HTTP error {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        log.warning(f"[rag] Karisma unreachable: {e}")

    return {"context": [], "answer": None, "source": "error"}


def format_context_for_llm(chunks: list[dict]) -> str:
    """Format Karisma context chunks into a clean string for the LLM prompt."""
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
