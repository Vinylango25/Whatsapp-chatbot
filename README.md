# WC — WhatsApp AI Chatbot

A multi-tenant WhatsApp business chatbot powered by RAG (Karisma), OpenAI, and M-Pesa payments.

## Architecture

```
WhatsApp User → Twilio → WC Gateway (FastAPI :8100)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Intent Router    RAG Client      M-Pesa Daraja
              │         (Karisma)        (STK Push)
              ▼               │
         OpenAI LLM  ←────────┘
              │
         WhatsApp Reply
```

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

### 3. Start Karisma RAG backend (in separate terminal)
```bash
cd ../RAGV2/Karisma
python -m uvicorn app:app --reload --port 8001
```

### 4. Start WC
```bash
python -m uvicorn main:app --reload --port 8100
```

### 5. Expose locally with ngrok (for Twilio webhook)
```bash
ngrok http 8100
```
Copy the HTTPS URL and set it in Twilio console:
- Twilio Console → Messaging → Sandbox → Webhook URL:
  `https://your-ngrok-url.ngrok.io/webhook/whatsapp`

### 6. Test on WhatsApp
- Join Twilio sandbox: send `join <your-sandbox-code>` to `+1 415 523 8886`
- Send any message to start chatting

## Features

- **RAG answers** — accurate responses from your business knowledge base
- **M-Pesa payments** — STK Push payment collection via WhatsApp
- **Multi-turn flows** — guided payment collection (amount → phone → confirm → pay)
- **Demo mode** — Twilio sandbox = instant demo for LinkedIn/sales
- **Session management** — Redis-backed (falls back to in-memory)
- **Human escalation** — detects when customer needs a human agent

## Project Structure

```
WC/
├── main.py              ← FastAPI gateway + Twilio webhook
├── processor/
│   └── handler.py       ← Message processing + intent routing
├── rag/
│   └── client.py        ← Karisma RAG HTTP client
├── llm/
│   └── openai_client.py ← OpenAI answer generation
├── state/
│   └── session.py       ← Redis session manager
├── payments/
│   └── mpesa.py         ← M-Pesa Daraja STK Push
├── demo/
│   └── sandbox.py       ← Demo/LinkedIn sandbox mode
├── .env.example
├── requirements.txt
└── README.md
```

## Deploying to Vercel

1. Push to GitHub
2. Import in Vercel
3. Set environment variables in Vercel dashboard
4. Update Twilio webhook URL to your Vercel domain
5. Update `MPESA_CALLBACK_URL` to your Vercel domain

> Note: Vercel runs serverless functions. For Redis sessions, use Upstash Redis (free tier available).
