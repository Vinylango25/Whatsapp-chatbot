"""
Standalone local ingestion script.
- Embeds docs using sentence-transformers (fully offline, no API needed)
- Pushes vectors directly to Pinecone
- Use this whenever you need to re-ingest without an OpenAI key

Usage:
    python ingest_local.py
"""
import os, uuid, pathlib, textwrap
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env", override=True)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "wc-kb")
TENANT_ID        = os.getenv("DEFAULT_TENANT_ID", "fademasters")
DOCS_DIR         = pathlib.Path(__file__).parent / "demo" / "docs"
MODEL_NAME       = "BAAI/bge-small-en-v1.5"
DIM              = 384
CHUNK_SIZE       = 400   # words per chunk
CHUNK_OVERLAP    = 40

print(f"Loading model {MODEL_NAME}...")
from fastembed import TextEmbedding
model = TextEmbedding(MODEL_NAME)
print("Model loaded.")

# ── Pinecone setup ────────────────────────────────────────────────────────────
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)

existing = [i.name for i in pc.list_indexes()]
if PINECONE_INDEX not in existing:
    print(f"Creating index {PINECONE_INDEX} with {DIM} dims...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    import time; time.sleep(10)

index = pc.Index(PINECONE_INDEX)
namespace = f"{TENANT_ID}_kb"
print(f"Using namespace: {namespace}")

# ── Chunker ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, doc_name: str):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk_text  = " ".join(chunk_words)
        chunks.append({
            "id":       str(uuid.uuid4()),
            "text":     chunk_text,
            "doc_name": doc_name,
            "category": "general",
            "tenant_id": TENANT_ID,
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── Ingest all docs ───────────────────────────────────────────────────────────
all_chunks = []
for doc in sorted(DOCS_DIR.glob("*.txt")):
    text   = doc.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text, doc.name)
    all_chunks.extend(chunks)
    print(f"  {doc.name}: {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")
print("Embedding...")

texts = [c["text"] for c in all_chunks]
vectors = list(model.embed(texts))

# ── Upload to Pinecone in batches of 100 ─────────────────────────────────────
print("Uploading to Pinecone...")
batch_size = 100
for i in range(0, len(all_chunks), batch_size):
    batch_chunks  = all_chunks[i:i + batch_size]
    batch_vectors = vectors[i:i + batch_size]
    pinecone_batch = [
        {
            "id":       c["id"],
            "values":   v.tolist(),
            "metadata": {
                "text":      c["text"],
                "doc_name":  c["doc_name"],
                "category":  c["category"],
                "tenant_id": c["tenant_id"],
            }
        }
        for c, v in zip(batch_chunks, batch_vectors)
    ]
    index.upsert(vectors=pinecone_batch, namespace=namespace)
    print(f"  Uploaded batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")

print("\nDone! Verifying...")
import time; time.sleep(5)
stats = index.describe_index_stats()
ns    = stats.namespaces.get(namespace)
print(f"Vectors in {namespace}: {ns.vector_count if ns else 0}")
print("Ingestion complete.")
