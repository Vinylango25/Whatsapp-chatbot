# WC — WhatsApp AI Chatbot Platform

> A production-ready, multi-tenant WhatsApp AI chatbot with RAG knowledge base, M-Pesa payments, and dual webhook support (Twilio + Meta Cloud API). Built for African businesses.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Features](#3-features)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [Prerequisites](#6-prerequisites)
7. [Installation & Local Setup](#7-installation--local-setup)
8. [Environment Variables Reference](#8-environment-variables-reference)
9. [Knowledge Base (RAG System)](#9-knowledge-base-rag-system)
10. [Webhook Integrations](#10-webhook-integrations)
11. [M-Pesa Daraja Integration](#11-m-pesa-daraja-integration)
12. [Session Management](#12-session-management)
13. [LLM & Intent Routing](#13-llm--intent-routing)
14. [API Reference](#14-api-reference)
15. [Deployment (Render)](#15-deployment-render)
16. [Multi-Tenant Support](#16-multi-tenant-support)
17. [Demo Mode](#17-demo-mode)
18. [Troubleshooting](#18-troubleshooting)
19. [Roadmap](#19-roadmap)
20. [License](#20-license)

---

## 1. Project Overview

**WC** (WhatsApp Chatbot) is a fully production-ready AI-powered WhatsApp chatbot platform built specifically for African businesses. It enables any business — barbershops, salons, clinics, restaurants, real estate agencies — to deploy a 24/7 AI assistant on WhatsApp that answers customer questions, handles bookings, and processes M-Pesa payments, all without requiring customers to download any app.

### Who Is This For?

- **Business owners** who receive repetitive WhatsApp inquiries and want to automate responses
- **Developers** building WhatsApp chatbot products for clients
- **Agencies** looking to white-label and resell a chatbot SaaS product
- **Startups** in the African market needing M-Pesa + WhatsApp integration out of the box

### What Problem Does It Solve?

Most Kenyan and African businesses receive the same questions on WhatsApp every day:
- "Where are you located?"
- "What are your prices?"
- "What time do you open?"
- "How do I book?"

Staff spend hours answering these manually, miss messages after hours, and lose customers. WC answers all of these instantly, 24/7, using the business's own documents as the knowledge base — with zero hallucination risk because answers come only from uploaded content.

### Live Demo

Send **"Hi"** to `+1 415 523 8886` on WhatsApp (Twilio sandbox — send `join <sandbox-code>` first).

Currently demoing for **Fade Masters Barbershop, Nairobi**.

---

## 2. Architecture

```
WhatsApp User
      │
      ├──── Twilio Sandbox/API ──────────────────────┐
      │                                              ▼
      └──── Meta Cloud API ──────────► WC Gateway (FastAPI :8100)
                                              │
                    ┌─────────────────────────┼──────────────────────┐
                    ▼                         ▼                      ▼
             Intent Router            Internal KB              M-Pesa Daraja
                    │               (Pinecone RAG)             (STK Push)
                    │                         │
         ┌──────────┴──────────┐              │
         ▼                     ▼              │
    Payment Flow          Groq LLM ◄──────────┘
    (multi-turn)        (LLaMA 3.3)
         │                     │
         └──────────┬──────────┘
                    ▼
             WhatsApp Reply
             (Twilio TwiML or
              Meta Graph API)
```

### Request Flow (Detailed)

```
1. Customer sends WhatsApp message
2. Twilio/Meta forwards POST to /webhook/whatsapp or /webhook/meta
3. Gateway extracts: sender, message body, sender name
4. Intent Router classifies the message:
   a. Greeting    → welcome message
   b. Payment     → start multi-turn payment flow
   c. Human agent → escalation message
   d. Question    → RAG + LLM pipeline
5. RAG Pipeline:
   a. Embed query using fastembed (BAAI/bge-small-en-v1.5, ONNX)
   b. Check semantic cache (Pinecone) for similar past queries
   c. Vector search against knowledge base (Pinecone)
   d. Pass top-K chunks to Groq LLaMA 3.3 for answer generation
6. Answer returned to customer via Twilio TwiML or Meta Graph API
7. Session updated (Redis or in-memory fallback)
```

---

## 3. Features

### Core Features

| Feature | Description |
|---|---|
| **RAG Knowledge Base** | Upload PDF, DOCX, TXT docs — bot answers from them instantly |
| **Semantic Cache** | Similar queries reuse cached answers, saving LLM API calls |
| **M-Pesa STK Push** | Multi-turn payment collection entirely via WhatsApp |
| **Multi-turn Flows** | Guided conversations for payments: amount → phone → confirm → pay |
| **Session Management** | Redis-backed (falls back to in-memory) per-user conversation state |
| **Human Escalation** | Detects escalation requests and routes to human agents |
| **Dual Webhooks** | Supports both Twilio and Meta WhatsApp Cloud API simultaneously |
| **Multi-tenant** | Single deployment serves multiple businesses with isolated KBs |
| **Demo Mode** | Sandbox numbers get demo tenant config automatically |

### AI Features

| Feature | Description |
|---|---|
| **Intent Classification** | Fast regex pre-classifier before RAG (greetings, payments, escalation) |
| **Semantic Search** | BAAI/bge-small-en-v1.5 via fastembed — 384-dim ONNX embeddings |
| **Answer Generation** | Groq LLaMA 3.3-70b-versatile — free, fast, accurate |
| **Context-only Answers** | Bot only answers from uploaded docs — zero hallucination |
| **Conversation History** | Last 10 turns kept per session for context-aware replies |
| **WhatsApp Formatting** | LLM output cleaned and formatted for WhatsApp rendering |

### Integration Features

| Feature | Description |
|---|---|
| **Twilio Sandbox** | Instant demo via Twilio WhatsApp sandbox |
| **Meta Cloud API** | Production webhook for real business numbers (free, 1000 conv/month) |
| **Pinecone Vector DB** | Serverless, scales automatically, free tier available |
| **M-Pesa Daraja** | Safaricom STK Push for Kenya payments |
| **Redis Sessions** | Optional Redis for persistent sessions across restarts |

---

## 4. Technology Stack

### Backend

| Component | Technology | Version | Why |
|---|---|---|---|
| Web Framework | FastAPI | 0.115.5 | Async, fast, auto-docs |
| ASGI Server | Uvicorn | 0.32.1 | Production-grade ASGI |
| HTTP Client | httpx | 0.27.2 | Async HTTP for external APIs |
| WhatsApp (Twilio) | twilio | 9.3.7 | Twilio webhook handling + TwiML |

### AI / ML

| Component | Technology | Version | Why |
|---|---|---|---|
| LLM | Groq (LLaMA 3.3-70b) | groq 0.10.0 | Free, fast, high quality |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5) | 0.8.0 | ONNX, ~100MB RAM, no PyTorch |
| Vector DB | Pinecone Serverless | pinecone 5.4.2 | Managed, free tier, scales |
| Fallback LLM | OpenAI GPT-4o-mini | openai 1.57.0 | Optional paid fallback |

### Data & Storage

| Component | Technology | Why |
|---|---|---|
| Session Store | Redis (+ in-memory fallback) | Fast, TTL-based session management |
| Vector Store | Pinecone | Cloud-native, serverless, no ops |
| Document Parsing | pypdf, pdfplumber, python-docx | Multi-format document ingestion |

### Infrastructure

| Component | Technology | Why |
|---|---|---|
| Deployment | Render | Free tier, auto-deploy from GitHub |
| Environment | python-dotenv | Local + production env management |
| Config | render.yaml | Infrastructure as code |

---

## 5. Project Structure

```
WC/
├── main.py                    ← FastAPI gateway, startup, Twilio + Meta webhooks
│
├── processor/
│   └── handler.py             ← Intent router + message orchestrator
│
├── rag/
│   └── client.py              ← RAG pipeline coordinator + Karisma fallback
│
├── llm/
│   └── openai_client.py       ← Groq LLM client (answer generation)
│
├── kb/
│   ├── embedder.py            ← Multi-provider embedder (fastembed/openai/ollama)
│   ├── store.py               ← Pinecone vector store manager
│   ├── retriever.py           ← Semantic search + semantic cache
│   ├── chunker.py             ← PDF/DOCX/TXT text extraction + chunking
│   ├── ingest.py              ← Ingestion pipeline (extract → chunk → embed → store)
│   └── admin.py               ← FastAPI admin routes (/kb/*)
│
├── state/
│   └── session.py             ← Redis session manager (with in-memory fallback)
│
├── payments/
│   └── mpesa.py               ← M-Pesa Daraja STK Push + callback handler
│
├── demo/
│   ├── sandbox.py             ← Demo/sandbox mode detection
│   └── docs/                  ← Sample knowledge base documents
│       ├── services_and_pricing.txt
│       ├── location_and_contact.txt
│       ├── opening_hours.txt
│       ├── booking_and_appointments.txt
│       ├── barbers_and_team.txt
│       ├── faqs.txt
│       ├── haircare_aftercare.txt
│       ├── loyalty_program.txt
│       ├── policies.txt
│       └── products_for_sale.txt
│
├── api/
│   └── index.py               ← Vercel/serverless handler (alternative deploy)
│
├── ingest_local.py            ← Standalone local ingestion script (fastembed)
├── diagnose_kb.py             ← KB diagnostics and search testing script
├── render.yaml                ← Render deployment config
├── requirements.txt           ← Python dependencies
├── .env.example               ← Environment variable template
└── README.md                  ← This file
```

---

## 6. Prerequisites

Before setting up WC, ensure you have:

### Required

- **Python 3.10+** (3.12 recommended locally; Render runs 3.14)
- **Git**
- **Pinecone account** — free at https://app.pinecone.io
- **Groq account** — free at https://console.groq.com (no credit card)
- **Twilio account** — free trial at https://twilio.com

### Optional

- **Redis** — for persistent sessions (falls back to in-memory if unavailable)
- **OpenAI account** — only if you want OpenAI embeddings (paid)
- **Safaricom Daraja account** — for M-Pesa payments (free sandbox at https://developer.safaricom.co.ke)
- **Meta Developer account** — for production WhatsApp number (free)

### For local development

- `conda` or `venv` for virtual environment management
- `ngrok` for exposing local server to Twilio/Meta webhooks

---

## 7. Installation & Local Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/Vinylango25/Whatsapp-chatbot.git
cd Whatsapp-chatbot
```

### Step 2 — Create virtual environment

```bash
# Using conda (recommended)
conda create -n wc python=3.12
conda activate wc

# Or using venv
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Environment Variables Reference](#8-environment-variables-reference)).

Minimum required for basic operation:
```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
```

### Step 5 — Start the server

```bash
uvicorn main:app --reload --port 8100
```

Visit http://localhost:8100/docs for the interactive API documentation.

### Step 6 — Ingest your knowledge base

Upload documents via the admin API:

```bash
curl -X POST http://localhost:8100/kb/upload \
  -F "file=@your-document.pdf" \
  -F "tenant_id=your_business" \
  -F "category=general" \
  -F "api_key=your_admin_key"
```

Or use the standalone ingestion script for bulk upload:

```bash
# Place .txt files in demo/docs/ and run:
python ingest_local.py
```

### Step 7 — Expose locally with ngrok

```bash
ngrok http 8100
```

Copy the HTTPS URL (e.g. `https://abc123.ngrok.io`) and configure:

**Twilio sandbox webhook:**
- Console → Messaging → Try it out → Send a WhatsApp message → Sandbox Settings
- "When a message comes in": `https://abc123.ngrok.io/webhook/whatsapp`

**Meta webhook:**
- Developer dashboard → WhatsApp → Configuration → Webhook
- Callback URL: `https://abc123.ngrok.io/webhook/meta`
- Verify token: value of `META_VERIFY_TOKEN` in your `.env`

### Step 8 — Test

Send `Hi` to your Twilio sandbox number (+1 415 523 8886) or your Meta-connected number.

---

## 8. Environment Variables Reference

### Core Settings

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key (free at console.groq.com) |
| `GROQ_MODEL` | ❌ | `llama-3.3-70b-versatile` | Groq model name |
| `OPENAI_API_KEY` | ❌ | — | OpenAI key (only needed for OpenAI embeddings) |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | OpenAI model (if using OpenAI LLM) |
| `LLM_TEMPERATURE` | ❌ | `0.3` | LLM response randomness (0.0–1.0) |
| `LLM_MAX_TOKENS` | ❌ | `600` | Maximum tokens per LLM response |

### Embedding Settings

| Variable | Required | Default | Description |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | ❌ | `fastembed` | `fastembed`, `openai`, `ollama`, `huggingface` |
| `EMBEDDING_MODEL` | ❌ | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `EMBEDDING_DIM` | ❌ | `384` | Embedding dimensions (must match model + Pinecone index) |
| `HF_API_KEY` | ❌ | — | HuggingFace token (only for `huggingface` provider) |
| `OLLAMA_URL` | ❌ | `http://localhost:11434` | Ollama server URL (only for `ollama` provider) |

### Vector Database (Pinecone)

| Variable | Required | Default | Description |
|---|---|---|---|
| `PINECONE_API_KEY` | ✅ | — | Pinecone API key |
| `PINECONE_INDEX` | ❌ | `wc-kb` | Pinecone index name |
| `SCORE_THRESHOLD` | ❌ | `0.15` | Minimum similarity score to include a result |
| `CACHE_THRESHOLD` | ❌ | `0.92` | Minimum similarity to treat as semantic cache hit |
| `TOP_K` | ❌ | `4` | Number of KB chunks to retrieve per query |

### Twilio

| Variable | Required | Default | Description |
|---|---|---|---|
| `TWILIO_ACCOUNT_SID` | ✅ | — | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | ✅ | — | Twilio Auth Token |
| `TWILIO_WHATSAPP_NUMBER` | ❌ | `whatsapp:+14155238886` | Twilio WhatsApp sender number |

### Meta WhatsApp Cloud API

| Variable | Required | Default | Description |
|---|---|---|---|
| `META_VERIFY_TOKEN` | ❌ | — | Webhook verification token (you choose this) |
| `META_ACCESS_TOKEN` | ❌ | — | Meta Graph API access token |
| `META_PHONE_NUMBER_ID` | ❌ | — | Meta phone number ID from developer dashboard |

### Business Settings

| Variable | Required | Default | Description |
|---|---|---|---|
| `DEFAULT_TENANT_ID` | ❌ | `demo` | Default tenant ID for single-tenant mode |
| `DEFAULT_CATEGORY` | ❌ | `general` | Default KB category for searches |
| `BUSINESS_NAME` | ❌ | `WC Demo Business` | Business display name in bot responses |
| `ADMIN_API_KEY` | ❌ | *(empty)* | Protects KB admin endpoints |
| `SESSION_TTL_SECONDS` | ❌ | `1800` | Session expiry in seconds (default: 30 min) |
| `CONTACT_EMAIL` | ❌ | — | Business contact email |

### M-Pesa Daraja

| Variable | Required | Default | Description |
|---|---|---|---|
| `MPESA_DARAJA_URL` | ❌ | `https://sandbox.safaricom.co.ke` | Daraja base URL (change to production URL when live) |
| `MPESA_CONSUMER_KEY` | ✅ for payments | — | Daraja app consumer key |
| `MPESA_CONSUMER_SECRET` | ✅ for payments | — | Daraja app consumer secret |
| `MPESA_SHORTCODE` | ❌ | `174379` | M-Pesa business shortcode |
| `MPESA_PASSKEY` | ❌ | sandbox default | Daraja STK Push passkey |
| `MPESA_CALLBACK_URL` | ✅ for payments | — | Public URL for payment callbacks |
| `MPESA_ACCOUNT_REF` | ❌ | `WC` | Payment account reference (shown on customer's M-Pesa) |

### Session / Redis

| Variable | Required | Default | Description |
|---|---|---|---|
| `REDIS_URL` | ❌ | `redis://localhost:6379` | Redis connection URL (falls back to in-memory if unavailable) |

---

## 9. Knowledge Base (RAG System)

The RAG (Retrieval-Augmented Generation) system is the core intelligence of WC. It allows the bot to answer questions based exclusively on your business's own documents, eliminating hallucination.

### How It Works

```
Document Upload
      │
      ▼
  Text Extraction          ← supports PDF, DOCX, TXT, MD
      │
      ▼
  Text Chunking            ← splits into ~400-word chunks with 40-word overlap
      │
      ▼
  Embedding                ← BAAI/bge-small-en-v1.5 via fastembed (384 dims)
      │
      ▼
  Pinecone Upsert          ← stored in namespace "{tenant_id}_kb"
      │
      ▼
  Ready for queries ✓

Query Time:
  User message
      │
      ▼
  Embed query              ← same model, same dims
      │
      ▼
  Check semantic cache     ← Pinecone namespace "{tenant_id}_cache"
      │
      ├── Cache HIT  ──────► return cached answer (no LLM call needed)
      │
      └── Cache MISS ──────► Vector search in KB namespace
                                    │
                                    ▼
                             Top-K chunks retrieved
                                    │
                                    ▼
                             Groq LLaMA generates answer
                                    │
                                    ▼
                             Answer cached for future similar queries
```

### Supported Document Formats

| Format | Parser | Notes |
|---|---|---|
| PDF | pdfplumber + pypdf | Multi-page, text extraction |
| DOCX | python-docx | Microsoft Word documents |
| TXT | built-in | Plain text files |
| MD | built-in | Markdown files |

### Embedding Providers

| Provider | Model | Dims | RAM | Cost | Internet Required |
|---|---|---|---|---|---|
| `fastembed` (default) | BAAI/bge-small-en-v1.5 | 384 | ~100MB | Free | No (ONNX in-process) |
| `openai` | text-embedding-3-small | 1536 | minimal | Paid | Yes |
| `openai` | text-embedding-3-large | 3072 | minimal | Paid | Yes |
| `ollama` | nomic-embed-text | 768 | ~500MB | Free | No (local) |
| `huggingface` | any HF model | varies | minimal | Free | Yes |

> **Note:** Changing the embedding provider requires recreating the Pinecone index (different dimensions) and re-ingesting all documents.

### Semantic Cache

The semantic cache prevents duplicate LLM calls for similar questions. When a customer asks "What are your opening hours?" and another asks "When do you open?", the second query hits the cache instead of calling Groq again.

- Cache threshold: `CACHE_THRESHOLD=0.92` (92% similarity triggers a cache hit)
- Cache stored in Pinecone namespace `{tenant_id}_cache`
- Each cache entry stores: query, answer, hit count, timestamps

### Knowledge Base Admin API

All admin endpoints are protected by `ADMIN_API_KEY` (pass as form field `api_key`).

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/kb/upload` | Upload and ingest a document |
| `GET` | `/kb/stats/{tenant_id}` | KB stats (chunk count, document count, cache size) |
| `GET` | `/kb/documents/{tenant_id}` | List all ingested documents |
| `DELETE` | `/kb/documents/{tenant_id}/{doc_name}` | Delete a specific document |
| `DELETE` | `/kb/cache/{tenant_id}` | Clear the semantic cache |
| `GET` | `/kb/health` | KB health check |

### Local Ingestion Script

For bulk ingestion without running the server, use `ingest_local.py`:

```bash
# Place .txt files in demo/docs/ directory
python ingest_local.py
```

This script:
1. Loads fastembed model locally (no API calls)
2. Chunks all `.txt` files in `demo/docs/`
3. Embeds all chunks in one batch
4. Uploads directly to Pinecone
5. Verifies the upload

Useful when:
- Initial setup / first-time ingestion
- Re-ingesting after switching embedding models
- Ingesting from a machine with no internet for HF API (fastembed runs offline)

---

## 10. Webhook Integrations

WC supports two WhatsApp webhook providers simultaneously. Both route to the same `process_message()` function.

### Twilio Webhook (`/webhook/whatsapp`)

Twilio sends a `POST` with `application/x-www-form-urlencoded` form data when a message is received.

**Request fields used:**
| Field | Description |
|---|---|
| `From` | Sender's WhatsApp number (`whatsapp:+254712345678`) |
| `Body` | Message text |
| `ProfileName` | Sender's WhatsApp display name |
| `NumMedia` | Number of media attachments |
| `MediaUrl0` | URL of first media attachment |

**Response:** TwiML XML (`application/xml`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Message>Your answer here</Message>
</Response>
```

**Setup:**
1. Twilio Console → Messaging → Try it out → Send a WhatsApp message → Sandbox Settings
2. "When a message comes in": `https://your-domain.com/webhook/whatsapp`
3. Method: `HTTP POST`

### Meta Cloud API Webhook (`/webhook/meta`)

Meta sends a `POST` with JSON body. The `GET` endpoint handles webhook verification.

**Verification (GET `/webhook/meta`):**
Meta sends `hub.mode`, `hub.verify_token`, `hub.challenge` as query params.
We return `hub.challenge` as plain text if `hub.verify_token` matches `META_VERIFY_TOKEN`.

**Message payload (POST `/webhook/meta`):**
```json
{
  "entry": [{
    "changes": [{
      "value": {
        "messages": [{
          "from": "254712345678",
          "type": "text",
          "text": { "body": "Hello!" }
        }],
        "contacts": [{
          "profile": { "name": "John Doe" }
        }]
      }
    }]
  }]
}
```

**Response:** JSON `{"status": "ok"}` — Meta requires HTTP 200 within 20 seconds.

**Reply via Graph API:**
```
POST https://graph.facebook.com/v19.0/{META_PHONE_NUMBER_ID}/messages
Authorization: Bearer {META_ACCESS_TOKEN}
{
  "messaging_product": "whatsapp",
  "to": "254712345678",
  "type": "text",
  "text": { "body": "Your answer here" }
}
```

**Setup:**
1. Meta Developer Dashboard → WhatsApp → Configuration → Webhook
2. Callback URL: `https://your-domain.com/webhook/meta`
3. Verify token: value of `META_VERIFY_TOKEN`
4. Subscribe to `messages` field

---

## 11. M-Pesa Daraja Integration

WC includes a complete M-Pesa STK Push integration via Safaricom's Daraja API, enabling businesses to collect payments entirely through WhatsApp.

### Payment Flow

```
Customer: "I want to pay"
      │
      ▼
Bot: "How much would you like to pay? (in KES)"
      │
Customer: "500"
      │
      ▼
Bot: "Which phone number should receive the M-Pesa prompt?
      (Press 1 to use +254712345678, or type another number)"
      │
Customer: "1"
      │
      ▼
Bot: "✅ Confirm Payment
      Amount: KES 500
      Phone: +254712345678
      To: Fade Masters Barbershop
      Reply YES to pay or NO to cancel."
      │
Customer: "YES"
      │
      ▼
Daraja STK Push → Customer's phone gets M-Pesa PIN prompt
      │
Customer enters PIN
      │
      ▼
Daraja callback → /mpesa/callback
      │
      ▼
Payment status updated
Customer can reply "STATUS" to check
```

### Configuration

**Sandbox (testing):**
```env
MPESA_DARAJA_URL=https://sandbox.safaricom.co.ke
MPESA_SHORTCODE=174379
MPESA_PASSKEY=bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919
MPESA_CONSUMER_KEY=your_sandbox_consumer_key
MPESA_CONSUMER_SECRET=your_sandbox_consumer_secret
MPESA_CALLBACK_URL=https://your-domain.com/mpesa/callback
```

**Production:**
```env
MPESA_DARAJA_URL=https://api.safaricom.co.ke
MPESA_SHORTCODE=your_actual_shortcode
MPESA_PASSKEY=your_actual_passkey
MPESA_CONSUMER_KEY=your_production_consumer_key
MPESA_CONSUMER_SECRET=your_production_consumer_secret
```

### Daraja Endpoints Used

| Endpoint | Purpose |
|---|---|
| `GET /oauth/v1/generate` | OAuth token generation |
| `POST /mpesa/stkpush/v1/processrequest` | Initiate STK Push |

### M-Pesa API Endpoints (WC)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/mpesa/callback` | Daraja payment result callback |
| `GET` | `/mpesa/status/{checkout_id}` | Check payment status |

> **Note:** In production, replace the in-memory `_payments` dict in `payments/mpesa.py` with a persistent database (PostgreSQL, MongoDB, etc.) to survive server restarts.

---

## 12. Session Management

WC maintains per-user conversation state to support multi-turn flows (payments, bookings) and conversation history for context-aware replies.

### Session Data Structure

```json
{
  "sender": "whatsapp:+254712345678",
  "tenant_id": "fademasters",
  "created_at": 1723890000,
  "turn_count": 5,
  "flow": "payment",
  "flow_step": "confirm",
  "amount": 500,
  "phone": "+254712345678",
  "last_checkout_id": "ws_1234567890",
  "history": [
    {"role": "user", "content": "What are your prices?"},
    {"role": "assistant", "content": "Our prices start from KES 300..."}
  ]
}
```

### Storage Backends

**Redis (preferred for production):**
- Automatic TTL expiry (`SESSION_TTL_SECONDS`)
- Persists across server restarts
- Configure: `REDIS_URL=redis://your-redis-host:6379`

**In-memory (default fallback):**
- Works out of the box, no configuration needed
- Lost on server restart
- Fine for Render free tier (single instance)

### Session Lifecycle

```
First message → create session
      │
Each message → get session → update history + flow state → save
      │
Payment complete / 30min inactivity → session TTL expires
      │
Next message → create new session
```

---

## 13. LLM & Intent Routing

### Intent Classification

Before calling the RAG pipeline, the handler runs fast regex-based intent detection:

| Intent | Trigger Keywords | Action |
|---|---|---|
| Greeting | hi, hello, hey, habari, sasa, help, menu | Welcome message + menu |
| Payment | pay, mpesa, lipa, buy, order, nunua | Start payment flow |
| Payment status | status, check, confirm, receipt, paid | Check last payment status |
| Human escalation | agent, human, person, speak to someone | Escalation message |
| Question (default) | anything else | RAG + LLM pipeline |

This pre-classification handles ~40% of messages without any LLM call, keeping costs near zero and response times fast.

### LLM Configuration

WC uses **Groq** as the primary LLM provider — completely free with no credit card required.

**System Prompt:**
The LLM is instructed to:
- Answer only from the provided context (no hallucination)
- Use friendly WhatsApp-appropriate tone
- Use WhatsApp formatting (`*bold*`, bullet points with `•`)
- Keep responses concise (max 3-4 paragraphs)
- Offer to escalate to a human if the answer isn't in the KB

**Groq Models Available (free):**

| Model | Speed | Quality | Context |
|---|---|---|---|
| `llama-3.3-70b-versatile` | Fast | Best | 128k tokens |
| `llama-3.1-8b-instant` | Very fast | Good | 128k tokens |
| `mixtral-8x7b-32768` | Fast | Good | 32k tokens |

### WhatsApp Output Formatting

LLM output is post-processed before sending:
- `## Heading` → `*Heading*` (WhatsApp bold)
- `**bold**` → `*bold*` (WhatsApp bold)
- Code blocks removed
- Response truncated to 4000 chars (WhatsApp limit)

---

## 14. API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info and version |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI (interactive API docs) |

### WhatsApp Webhooks

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/webhook/whatsapp` | Twilio WhatsApp message webhook |
| `GET` | `/webhook/whatsapp` | Twilio webhook status check |
| `GET` | `/webhook/meta` | Meta webhook verification |
| `POST` | `/webhook/meta` | Meta WhatsApp message webhook |

### Knowledge Base Admin

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/kb/upload` | api_key | Upload and ingest a document |
| `GET` | `/kb/stats/{tenant_id}` | api_key | KB statistics |
| `GET` | `/kb/documents/{tenant_id}` | api_key | List documents |
| `DELETE` | `/kb/documents/{tenant_id}/{doc_name}` | api_key | Delete a document |
| `DELETE` | `/kb/cache/{tenant_id}` | api_key | Clear semantic cache |
| `GET` | `/kb/health` | — | KB health check |

### M-Pesa

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/mpesa/callback` | Daraja payment callback (called by Safaricom) |
| `GET` | `/mpesa/status/{checkout_id}` | Check payment status |

### Upload Example

```bash
curl -X POST https://whatsapp-chatbot-932j.onrender.com/kb/upload \
  -F "file=@services.pdf" \
  -F "tenant_id=mybusiness" \
  -F "category=general" \
  -F "api_key=your_admin_key"
```

Response:
```json
{
  "status": "success",
  "doc_name": "services.pdf",
  "chunks_added": 12,
  "tenant_id": "mybusiness"
}
```

---

## 15. Deployment (Render)

WC is deployed on Render's free tier. The `render.yaml` file contains the full deployment config.

### Initial Deployment

1. **Fork/push the repository to GitHub**

2. **Create a Render account** at https://render.com

3. **Create a new Web Service:**
   - Connect GitHub repository
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Add environment variables** in Render dashboard → Environment tab

5. **Deploy** — Render auto-deploys on every `git push` to main

### Render Environment Variables

Paste the following block into Render → Secret Files → filename `.env`:

```env
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=600
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=wc-kb
SCORE_THRESHOLD=0.15
CACHE_THRESHOLD=0.92
TOP_K=4
ADMIN_API_KEY=your_admin_key
DEFAULT_TENANT_ID=your_tenant
DEFAULT_CATEGORY=general
BUSINESS_NAME=Your Business Name
SESSION_TTL_SECONDS=1800
MPESA_DARAJA_URL=https://sandbox.safaricom.co.ke
MPESA_SHORTCODE=174379
MPESA_PASSKEY=your_passkey
MPESA_CALLBACK_URL=https://your-service.onrender.com/mpesa/callback
MPESA_ACCOUNT_REF=YourBusiness
CONTACT_EMAIL=hello@yourbusiness.com
META_VERIFY_TOKEN=your_meta_verify_token
META_ACCESS_TOKEN=your_meta_access_token
META_PHONE_NUMBER_ID=your_phone_number_id
```

### Render Free Tier Constraints

| Resource | Free Tier Limit | Impact |
|---|---|---|
| RAM | 512MB | Why we use fastembed (not sentence-transformers/PyTorch) |
| CPU | Shared | Response times ~1-3s |
| Sleep | Spins down after 15min inactivity | First message after sleep takes ~30s |
| Bandwidth | 100GB/month | ~1M WhatsApp messages |

### Preventing Cold Starts

Render free tier spins down after 15 minutes of inactivity. To prevent this:

Option 1 — Use a free uptime monitor like **UptimeRobot** (https://uptimerobot.com):
- Create a monitor for `https://your-service.onrender.com/health`
- Set interval: every 14 minutes
- This keeps the service warm 24/7

Option 2 — Upgrade to Render Starter ($7/month) for always-on service.

### Custom Domain

1. Render dashboard → your service → Settings → Custom Domain
2. Add your domain (e.g. `api.yourbusiness.com`)
3. Update your DNS with the CNAME Render provides
4. Update Twilio and Meta webhook URLs to use your custom domain

---

## 16. Multi-Tenant Support

WC is designed from the ground up to serve multiple businesses from a single deployment.

### Tenant Isolation

Each tenant gets isolated:
- **Pinecone namespace:** `{tenant_id}_kb` for knowledge, `{tenant_id}_cache` for cache
- **Session keys:** `wc:session:{sender}` with tenant_id stored in session
- **Business config:** name, M-Pesa shortcode, system prompt additions

### Adding a New Tenant

1. **Ingest their documents:**
```bash
curl -X POST https://your-service.onrender.com/kb/upload \
  -F "file=@client_docs.pdf" \
  -F "tenant_id=new_client" \
  -F "category=general" \
  -F "api_key=your_admin_key"
```

2. **Map their phone number to their tenant** (edit `processor/handler.py`):
```python
def _resolve_tenant(sender: str) -> dict:
    # Add routing logic here
    if sender == "whatsapp:+254712345678":  # client's number
        return {
            "tenant_id": "new_client",
            "business_name": "New Client Business",
            ...
        }
    # Default tenant
    return {
        "tenant_id": os.getenv("DEFAULT_TENANT_ID", "demo"),
        ...
    }
```

3. **Configure their Twilio/Meta webhook** to point to your server

### Billing Model for Resellers

If you're using WC as a SaaS platform:

| Tier | Price (KES/month) | Features |
|---|---|---|
| Starter | 5,000 | 1 number, 1 KB, basic support |
| Business | 10,000 | 1 number, unlimited KB, M-Pesa, analytics |
| Enterprise | 20,000+ | Multiple numbers, custom flows, priority support |

Your cost per client on Render free tier: ~KES 0 (within free tier limits).

---

## 17. Demo Mode

The demo mode automatically serves a pre-configured Fade Masters Barbershop demo to users connecting from the Twilio sandbox number.

### How It Works

`demo/sandbox.py` contains `is_demo_number()` which detects sandbox connections:

```python
def is_demo_number(sender: str) -> bool:
    # Twilio sandbox senders get demo tenant
    return True  # or check against a list of demo numbers
```

### Demo Knowledge Base

The `demo/docs/` directory contains 10 pre-built knowledge base documents for Fade Masters Barbershop:

| File | Contents |
|---|---|
| `services_and_pricing.txt` | All services with prices (haircuts, fades, beard) |
| `location_and_contact.txt` | Address, directions, contact details |
| `opening_hours.txt` | Hours by day of week |
| `booking_and_appointments.txt` | How to book, walk-in policy |
| `barbers_and_team.txt` | Team profiles and specialties |
| `faqs.txt` | Common customer questions |
| `haircare_aftercare.txt` | Post-haircut care instructions |
| `loyalty_program.txt` | Points system and rewards |
| `policies.txt` | Cancellation, refund, child policies |
| `products_for_sale.txt` | Hair products available for purchase |

These serve as a template for onboarding new business clients — just replace the content with their information.

---

## 18. Troubleshooting

### Bot not responding to messages

1. Check Render logs for errors
2. Verify Twilio webhook URL is set correctly in Twilio console
3. Check `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are correct
4. Ensure Twilio sandbox membership is active (expires every 72 hours)

### "I couldn't find an answer to that"

The bot is returning the fallback message, meaning KB search returned no results above the threshold.

**Diagnose:**
```bash
python diagnose_kb.py
```

**Common causes:**
- Pinecone index dimension mismatch — re-ingest with `python ingest_local.py`
- `SCORE_THRESHOLD` too high — lower it (try `0.10`)
- No documents ingested — check `GET /kb/stats/{tenant_id}`
- Wrong namespace — check `DEFAULT_TENANT_ID` matches what was ingested

### Out of memory on Render

Render free tier has 512MB RAM. If you see OOM errors:
- Ensure `EMBEDDING_PROVIDER=fastembed` (not `sentence_transformers` which pulls PyTorch ~500MB)
- Check for memory leaks in long-running sessions

### Pinecone filter errors

```
illegal condition for field tenant_id: unsupported operator
```

Pinecone free tier does not support metadata filtering. The current code uses namespaces for tenant isolation instead. If you see this error, check `kb/store.py` — the filter should have been removed.

### M-Pesa STK Push fails

1. Check `MPESA_CONSUMER_KEY` and `MPESA_CONSUMER_SECRET` are set
2. For sandbox: use sandbox credentials, not production
3. Phone number must be in format `254XXXXXXXXX` (no + prefix)
4. `MPESA_CALLBACK_URL` must be a publicly accessible HTTPS URL

### fastembed model download fails on first start

fastembed downloads the ONNX model from HuggingFace on first run (~23MB). If the download fails:
- Check internet connectivity on the server
- The model is cached after first download — subsequent starts are instant

### Semantic cache returning wrong answers

Lower `CACHE_THRESHOLD` to be more strict:
```env
CACHE_THRESHOLD=0.95
```

Or clear the cache:
```bash
curl -X DELETE https://your-service.onrender.com/kb/cache/your_tenant_id \
  -F "api_key=your_admin_key"
```

---

## 19. Roadmap

### Planned Features

- [ ] **Booking system** — actual appointment booking with calendar integration (Google Calendar, Calendly)
- [ ] **Analytics dashboard** — conversation volume, top questions, unanswered queries
- [ ] **Multi-language support** — Swahili, French for broader African market
- [ ] **Voice messages** — transcribe audio messages and answer them
- [ ] **Image handling** — process product photos, receipts
- [ ] **Broadcast messages** — send promotions to opted-in customers
- [ ] **CRM integration** — HubSpot, Salesforce customer data sync
- [ ] **Persistent payment store** — PostgreSQL/MongoDB instead of in-memory dict
- [ ] **Admin web dashboard** — upload docs, view conversations, configure bot
- [ ] **Conversation rating** — thumbs up/down feedback per response
- [ ] **LangFuse integration** — full observability, prompt tracking, cost monitoring
- [ ] **WhatsApp Flows** — rich interactive forms for booking, payments
- [ ] **Catalog integration** — product catalog from WhatsApp Business Catalog

### Performance Improvements

- [ ] Connection pooling for Pinecone queries
- [ ] Async batch embedding for large document uploads
- [ ] Response streaming for faster perceived latency
- [ ] CDN for media file serving

---

## 20. License

MIT License

Copyright (c) 2026 WC WhatsApp Chatbot

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Quick Reference Card

```
Start server:        uvicorn main:app --reload --port 8100
Ingest documents:    python ingest_local.py
Diagnose KB:         python diagnose_kb.py
API docs:            http://localhost:8100/docs
Health check:        http://localhost:8100/health
KB stats:            GET /kb/stats/{tenant_id}

Twilio webhook:      POST /webhook/whatsapp
Meta webhook:        POST /webhook/meta  (GET for verification)
M-Pesa callback:     POST /mpesa/callback

Pinecone namespaces: {tenant_id}_kb      (knowledge base)
                     {tenant_id}_cache   (semantic cache)

Session key format:  wc:session:{sender_number}
```

---

*Built with ❤️ for African businesses. If this helps your business, consider starring the repository.*
