# WC — WhatsApp AI Chatbot

A multi-tenant WhatsApp business chatbot with a built-in RAG knowledge base (Qdrant), OpenAI, and M-Pesa payments.

## Architecture

```
WhatsApp User → Twilio → WC Gateway (FastAPI :8100)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Intent Router    Internal KB     M-Pesa Daraja
              │         (Qdrant RAG)     (STK Push)
              ▼               │
         OpenAI LLM  ←────────┘
              │
         WhatsApp Reply
```

**No external RAG service needed** — Qdrant runs embedded (in-process), zero Docker/server setup.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your Twilio, OpenAI, and M-Pesa credentials
```

### 3. Start WC
```bash
uvicorn main:app --reload --port 8100
```

### 4. Upload your knowledge base documents
```bash
# Via HTTP (multipart form-data)
curl -X POST http://localhost:8100/kb/upload \
  -F "file=@your-document.pdf" \
  -F "tenant_id=demo" \
  -F "category=general"
```
Or open the Swagger UI at http://localhost:8100/docs and use the `/kb/upload` endpoint.

### 5. Expose locally with ngrok (for Twilio webhook)
```bash
ngrok http 8100
```
Copy the HTTPS URL and set it in Twilio console:
- Twilio Console → Messaging → Sandbox → When a message comes in:
  `https://your-ngrok-url.ngrok.io/webhook/whatsapp`

### 6. Test on WhatsApp
- Join Twilio sandbox: send `join <your-sandbox-code>` to `+1 415 523 8886`
- Send any message to start chatting

## Knowledge Base API

All KB endpoints are at `/kb/*` — protected by `ADMIN_API_KEY` if set.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/kb/upload` | Upload & ingest a PDF/DOCX/TXT/MD document |
| `GET`  | `/kb/stats/{tenant_id}` | KB stats (chunks, docs, cache) |
| `GET`  | `/kb/documents/{tenant_id}` | List ingested documents |
| `DELETE` | `/kb/documents/{tenant_id}/{doc_name}` | Delete a document |
| `DELETE` | `/kb/cache/{tenant_id}` | Clear semantic cache |
| `GET`  | `/kb/health` | KB health check |

## Supported Embedding Providers

| Provider | Model | Notes |
|----------|-------|-------|
| `openai` (default) | `text-embedding-3-small` | Requires `OPENAI_API_KEY` |
| `openai` | `text-embedding-3-large` | Higher quality, 3072 dims |
| `ollama` | `nomic-embed-text` | Free, local, set `OLLAMA_URL` |

Set via `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIM` in `.env`.

## Features

- **Built-in RAG** — upload PDFs/DOCX/TXT and get instant semantic search answers
- **Semantic cache** — similar queries reuse cached answers (saves OpenAI tokens)
- **M-Pesa payments** — STK Push payment collection via WhatsApp
- **Multi-turn flows** — guided payment collection (amount → phone → confirm → pay)
- **Demo mode** — Twilio sandbox = instant demo for LinkedIn/sales
- **Session management** — Redis-backed (falls back to in-memory)
- **Human escalation** — detects when customer needs a human agent
- **Karisma fallback** — optional external Karisma RAG if internal KB is empty

## Project Structure

```
WC/
├── main.py              ← FastAPI gateway + startup + Twilio webhook
├── processor/
│   └── handler.py       ← Message processing + intent routing
├── rag/
│   └── client.py        ← Internal KB retriever (with Karisma fallback)
├── llm/
│   └── openai_client.py ← OpenAI answer generation
├── state/
│   └── session.py       ← Redis session manager
├── payments/
│   └── mpesa.py         ← M-Pesa Daraja STK Push
├── demo/
│   └── sandbox.py       ← Demo/LinkedIn sandbox mode
├── kb/
│   ├── embedder.py      ← OpenAI / Ollama embedder
│   ├── store.py         ← Qdrant vector store manager
│   ├── chunker.py       ← PDF/DOCX/TXT text extraction + chunking
│   ├── ingest.py        ← Ingestion pipeline (extract → chunk → embed → store)
│   ├── retriever.py     ← Semantic search + semantic cache
│   └── admin.py         ← FastAPI admin routes (/kb/*)
├── .env.example
├── requirements.txt
└── README.md
```

## Environment Variables (Key Ones)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI embeddings + LLM |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `ollama` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_DIM` | `1536` | Embedding dimensions |
| `QDRANT_PATH` | `./qdrant_storage` | Local Qdrant storage path |
| `ADMIN_API_KEY` | *(empty)* | KB admin endpoint protection |
| `DEFAULT_TENANT_ID` | `demo` | Default tenant for single-tenant mode |
| `SCORE_THRESHOLD` | `0.3` | Minimum similarity for KB results |
| `CACHE_THRESHOLD` | `0.92` | Minimum similarity for cache hits |
| `KARISMA_URL` | *(empty)* | Optional external Karisma fallback URL |

## Deployment

### Fly.io / Railway / Render

1. Set all environment variables in the dashboard
2. Deploy — Qdrant storage persists to a mounted volume (set `QDRANT_PATH=/data/qdrant`)
3. Update Twilio and M-Pesa callback URLs

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
```
