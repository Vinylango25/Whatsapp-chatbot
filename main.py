"""
WC — WhatsApp AI Chatbot Gateway
FastAPI app — receives Twilio WhatsApp webhooks and orchestrates responses.

Run:  uvicorn main:app --reload --port 8100
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
from payments.mpesa import router as mpesa_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wc.gateway")

app = FastAPI(title="WC WhatsApp Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount M-Pesa callback router
app.include_router(mpesa_router, prefix="/mpesa")


@app.get("/")
def root():
    return {"status": "ok", "service": "WC WhatsApp Chatbot v1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
    NumMedia: str = Form(default="0"),
    MediaUrl0: str = Form(default=""),
    ProfileName: str = Form(default=""),
):
    """
    Twilio WhatsApp webhook.
    Twilio sends a POST with form fields when a message is received.
    We respond with TwiML XML.
    """
    sender   = From.strip()   # e.g. whatsapp:+254712345678
    message  = Body.strip()
    name     = ProfileName.strip() or "there"

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

    # Build TwiML response
    twiml = MessagingResponse()
    twiml.message(reply)
    return Response(content=str(twiml), media_type="application/xml")


@app.get("/webhook/whatsapp")
def whatsapp_verify():
    """Health check for Twilio webhook URL verification."""
    return {"status": "ok", "webhook": "whatsapp"}
