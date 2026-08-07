"""
WC — WhatsApp AI Chatbot Gateway
FastAPI app — receives Twilio WhatsApp webhooks and orchestrates responses.

Run:  uvicorn main:app --reload --port 8100
Docs: http://localhost:8100/docs
"""
from __future__ import annotations
import os, logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from twilio.twiml.messaging_response import MessagingResponse

from processor.handler import process_message
from payments.mpesa    import router as mpesa_router
from kb.admin          import router as kb_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wc.gateway")

app = FastAPI(
    title="WC WhatsApp AI Chatbot",
    version="1.0.0",
    description="Multi-tenant WhatsApp chatbot with RAG, OpenAI and M-Pesa payments.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(mpesa_router, prefix="/mpesa",  tags=["M-Pesa"])
app.include_router(kb_router,    prefix="/kb",     tags=["Knowledge Base"])


@app.on_event("startup")
async def startup():
    """Initialize KB store and embedder on startup."""
    log.info("[startup] Initializing WC...")
    try:
        from kb.store    import get_store
        from kb.embedder import get_embedder
        store    = get_store()
        embedder = get_embedder()
        # Pre-create demo collection
        demo_tenant = os.getenv("DEFAULT_TENANT_ID", "demo")
        store.ensure_collection(store.kb_collection(demo_tenant),    dim=embedder.dim)
        store.ensure_collection(store.cache_collection(demo_tenant), dim=embedder.dim)
        log.info(f"[startup] KB ready — provider={embedder.provider} model={embedder.model}")
    except Exception as e:
        log.warning(f"[startup] KB init warning: {e}")


@app.get("/")
def root():
    return {
        "status":  "ok",
        "service": "WC WhatsApp AI Chatbot v1.0",
        "docs":    "/docs",
        "kb":      "/kb/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request:     Request,
    From:        str = Form(...),
    Body:        str = Form(...),
    NumMedia:    str = Form(default="0"),
    MediaUrl0:   str = Form(default=""),
    ProfileName: str = Form(default=""),
):
    """
    Twilio WhatsApp webhook.
    Twilio sends a POST with form fields when a message is received.
    We respond with TwiML XML.
    """
    sender  = From.strip()
    message = Body.strip()
    name    = ProfileName.strip() or "there"

    log.info(f"[webhook] from={sender} name={name!r} msg={message!r}")

    try:
        reply = await process_message(
            sender=sender,
            message=message,
            name=name,
            has_media=int(NumMedia) > 0,
            media_url=MediaUrl0,
        )
    except Exception as exc:
        log.exception(f"[webhook] processing error: {exc}")
        reply = "Sorry, I'm having trouble right now. Please try again in a moment. 🙏"

    twiml = MessagingResponse()
    twiml.message(reply)
    return Response(content=str(twiml), media_type="application/xml")


@app.get("/webhook/whatsapp")
def whatsapp_verify():
    return {"status": "ok", "webhook": "whatsapp"}
