"""
WC KB — Ingestion Pipeline
Extract → Chunk → Embed → Store in Qdrant
"""
from __future__ import annotations
import uuid, logging, time
from pathlib import Path
from typing import List

from kb.chunker  import extract_text, chunk_text
from kb.embedder import get_embedder
from kb.store    import get_store

log = logging.getLogger("wc.kb.ingest")


async def ingest_file(
    file_path: str | Path,
    tenant_id: str,
    category: str  = "",
    tags: List[str] = None,
    replace: bool   = True,
) -> dict:
    """
    Ingest a single file into the tenant's KB.
    Returns: {doc_name, chunks_created, elapsed_ms}
    """
    path     = Path(file_path)
    doc_name = path.name
    store    = get_store()
    embedder = get_embedder()
    col      = store.kb_collection(tenant_id)

    # Ensure collection exists
    store.ensure_collection(col, dim=embedder.dim)

    # Delete existing chunks for this doc if replacing
    if replace:
        store.delete_by_doc(col, doc_name)
        log.info(f"[ingest] Replaced existing chunks for {doc_name}")

    t0 = time.monotonic()

    # 1. Extract text
    log.info(f"[ingest] Extracting text from {doc_name}")
    text = extract_text(path)
    if not text.strip():
        raise ValueError(f"No text extracted from {doc_name}")

    # 2. Chunk
    chunks = chunk_text(text, doc_name)
    if not chunks:
        raise ValueError(f"No chunks produced from {doc_name}")

    # 3. Embed all chunks
    log.info(f"[ingest] Embedding {len(chunks)} chunks...")
    texts    = [c["text"] for c in chunks]
    vectors  = await embedder.embed_batch(texts)

    # 4. Build Pinecone points (plain dicts: id, values, metadata)
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append({
            "id":     str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "text":        chunk["text"],
                "doc_name":    doc_name,
                "chunk_index": chunk["chunk_index"],
                "page_num":    chunk["page_num"],
                "tenant_id":   tenant_id,
                "category":    category,
                "tags":        ",".join(tags or []),  # Pinecone metadata must be str/int/float
                "ingested_at": int(time.time()),
            },
        })

    # 5. Upsert into Qdrant
    store.upsert(col, points)
    elapsed = int((time.monotonic() - t0) * 1000)

    log.info(f"[ingest] ✓ {doc_name}: {len(points)} chunks in {elapsed}ms")
    return {
        "doc_name":      doc_name,
        "chunks_created": len(points),
        "elapsed_ms":    elapsed,
        "tenant_id":     tenant_id,
        "collection":    col,
    }


async def ingest_folder(
    folder_path: str | Path,
    tenant_id: str,
    category: str  = "",
    tags: List[str] = None,
) -> List[dict]:
    """Ingest all supported files in a folder."""
    folder  = Path(folder_path)
    results = []
    exts    = {".pdf", ".docx", ".txt", ".md"}

    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in exts:
            try:
                result = await ingest_file(f, tenant_id, category, tags)
                results.append(result)
            except Exception as e:
                log.warning(f"[ingest] Failed {f.name}: {e}")
                results.append({"doc_name": f.name, "error": str(e)})

    return results
