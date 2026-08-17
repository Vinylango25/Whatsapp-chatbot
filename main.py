"""
WC — WhatsApp AI Chatbot Gateway
FastAPI app — receives Twilio AND Meta WhatsApp Cloud API webhooks.

Run:  uvicorn main:app --reload --port 8100
Docs: http://localhost:8100/docs
"""
from __future__ import annotations
import os, logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, Request, Form, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
import httpx

from processor.handler import process_message
from payments.mpesa    import router as mpesa_router
from kb.admin          import router as kb_router

# ── Meta Cloud API credentials ────────────────────────────────────────────────
META_VERIFY_TOKEN    = os.getenv("META_VERIFY_TOKEN", "")
META_ACCESS_TOKEN    = os.getenv("META_ACCESS_TOKEN", "")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID", "")

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


# ── Meta WhatsApp Cloud API webhook ──────────────────────────────────────────

@app.get("/webhook/meta")
async def meta_verify(request: Request):
    """
    Meta webhook verification.
    Meta sends a GET with hub.mode, hub.verify_token, hub.challenge.
    We must return hub.challenge as plain text if the token matches.
    """
    mode      = request.query_params.get("hub.mode")
    token     = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        log.info("[meta] Webhook verified successfully.")
        return PlainTextResponse(content=challenge)

    log.warning(f"[meta] Webhook verification failed. token={token!r}")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhook/meta")
async def meta_webhook(request: Request):
    """
    Meta WhatsApp Cloud API webhook.
    Receives messages, processes them and replies via Meta Graph API.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Meta sends a nested structure — dig out the message
    try:
        entry   = body["entry"][0]
        changes = entry["changes"][0]
        value   = changes["value"]

        # Ignore status updates (delivered, read receipts)
        if "messages" not in value:
            return JSONResponse({"status": "ignored"})

        msg_obj  = value["messages"][0]
        msg_type = msg_obj.get("type", "")

        # Only handle text messages for now
        if msg_type != "text":
            log.info(f"[meta] Ignoring non-text message type: {msg_type}")
            return JSONResponse({"status": "ignored"})

        sender  = msg_obj["from"]           # e.g. "254712345678"
        msg_id  = msg_obj["id"]
        message = msg_obj["text"]["body"].strip()

        # Get sender name from contacts if available
        contacts = value.get("contacts", [])
        name = contacts[0].get("profile", {}).get("name", "there") if contacts else "there"

        log.info(f"[meta] from={sender} name={name!r} msg={message!r}")

        # Process the message
        reply = await process_message(
            sender=f"whatsapp:{sender}",
            message=message,
            name=name,
        )

        # Send reply back via Meta Graph API
        await _meta_send(sender, reply)

        # Acknowledge receipt to Meta (must return 200 quickly)
        return JSONResponse({"status": "ok"})

    except (KeyError, IndexError) as e:
        # Malformed payload — return 200 so Meta doesn't retry endlessly
        log.warning(f"[meta] Unexpected payload shape: {e}")
        return JSONResponse({"status": "ignored"})
    except Exception as exc:
        log.exception(f"[meta] Error processing message: {exc}")
        return JSONResponse({"status": "error"})


async def _meta_send(to: str, text: str):
    """Send a text message via Meta WhatsApp Cloud API."""
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        log.warning("[meta] META_ACCESS_TOKEN or META_PHONE_NUMBER_ID not set — cannot send reply.")
        return

    url = f"https://graph.facebook.com/v19.0/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to":                to,
        "type":              "text",
        "text":              {"body": text},
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                log.warning(f"[meta] Send failed: {resp.status_code} {resp.text}")
            else:
                log.info(f"[meta] Message sent to {to}")
    except Exception as e:
        log.warning(f"[meta] Send error: {e}")
