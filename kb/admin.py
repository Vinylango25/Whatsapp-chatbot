"""
WC KB — Admin API Routes
Upload documents, manage knowledge base, view stats.
"""
from __future__ import annotations
import os, tempfile, logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from kb.ingest    import ingest_file
from kb.store     import get_store
from kb.embedder  import get_embedder

log    = logging.getLogger("wc.kb.admin")
router = APIRouter()

ADMIN_KEY = os.getenv("ADMIN_API_KEY", "")


def _check_admin(api_key: str = Query(default="")):
    if ADMIN_KEY and api_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin API key")


# ── Upload document ───────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    file:      UploadFile = File(...),
    tenant_id: str        = Form(...),
    category:  str        = Form(default=""),
    tags:      str        = Form(default=""),
    api_key:   str        = Form(default=""),
):
    """Upload and ingest a document into the tenant's knowledge base."""
    _check_admin(api_key)

    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")

    allowed = {".pdf", ".docx", ".txt", ".md"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed)}"
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        result   = await ingest_file(
            file_path=tmp_path,
            tenant_id=tenant_id,
            category=category,
            tags=tag_list,
        )
        # Use original filename in result
        result["doc_name"] = file.filename
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception(f"[admin] ingest error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── KB stats ──────────────────────────────────────────────────────────────────

@router.get("/stats/{tenant_id}")
async def kb_stats(tenant_id: str, api_key: str = Query(default="")):
    """Get KB stats for a tenant."""
    _check_admin(api_key)
    store     = get_store()
    embedder  = get_embedder()
    kb_col    = store.kb_collection(tenant_id)
    cache_col = store.cache_collection(tenant_id)

    store.ensure_collection(kb_col,    dim=embedder.dim)
    store.ensure_collection(cache_col, dim=embedder.dim)

    return {
        "tenant_id":    tenant_id,
        "kb_chunks":    store.count(kb_col),
        "cache_entries": store.count(cache_col),
        "documents":    store.list_docs(kb_col),
        "embedding_provider": embedder.provider,
        "embedding_model":    embedder.model,
        "embedding_dim":      embedder.dim,
    }


# ── List documents ────────────────────────────────────────────────────────────

@router.get("/documents/{tenant_id}")
async def list_documents(tenant_id: str, api_key: str = Query(default="")):
    """List all documents in a tenant's KB."""
    _check_admin(api_key)
    store  = get_store()
    kb_col = store.kb_collection(tenant_id)
    return {"tenant_id": tenant_id, "documents": store.list_docs(kb_col)}


# ── Delete document ───────────────────────────────────────────────────────────

@router.delete("/documents/{tenant_id}/{doc_name}")
async def delete_document(
    tenant_id: str,
    doc_name:  str,
    api_key:   str = Query(default=""),
):
    """Delete a document and all its chunks from the KB."""
    _check_admin(api_key)
    store  = get_store()
    kb_col = store.kb_collection(tenant_id)
    store.delete_by_doc(kb_col, doc_name)
    return {"status": "deleted", "tenant_id": tenant_id, "doc_name": doc_name}


# ── Clear cache ───────────────────────────────────────────────────────────────

@router.delete("/cache/{tenant_id}")
async def clear_cache(tenant_id: str, api_key: str = Query(default="")):
    """Clear semantic cache for a tenant."""
    _check_admin(api_key)
    store     = get_store()
    cache_col = store.cache_collection(tenant_id)
    try:
        store._index.delete(delete_all=True, namespace=cache_col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")
    return {"status": "cleared", "tenant_id": tenant_id, "collection": cache_col}


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def kb_health():
    store    = get_store()
    embedder = get_embedder()
    return {
        "status":             "ok",
        "pinecone_index":     os.getenv("PINECONE_INDEX", "wc-kb"),
        "embedding_provider": embedder.provider,
        "embedding_model":    embedder.model,
    }
