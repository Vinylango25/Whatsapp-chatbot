"""
WC — Message Processor & Intent Router
Orchestrates: session → intent → RAG/LLM/payment → response
"""
from __future__ import annotations
import os, logging, re
from state.session import SessionManager
from rag.client import karisma_retrieve
from llm.openai_client import generate_answer
from payments.mpesa import initiate_stk_push, get_payment_status
from demo.sandbox import is_demo_number, get_demo_tenant
from kb.retriever import cache_answer

log = logging.getLogger("wc.processor")

session_mgr = SessionManager()

# ── Intent keywords (fast pre-classifier before RAG) ─────────────────────────
_PAYMENT_PATTERNS = re.compile(
    r"\b(pay|payment|lipa|pesa|mpesa|m-pesa|premium|bill|invoice|bei|bei yake|"
    r"buy|purchase|order|nunua|checkout)\b", re.I
)
_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hujambo|habari|sasa|mambo|niaje|good morning|good afternoon|"
    r"good evening|start|menu|help|msaada)\b", re.I
)
_HUMAN_PATTERNS = re.compile(
    r"\b(human|agent|person|rep|representative|speak to someone|talk to someone|"
    r"operator|supervisor|escalate|complaint)\b", re.I
)
_PAYMENT_STATUS_PATTERNS = re.compile(
    r"\b(status|check|confirm|did it go|imetumwa|imeenda|receipt|paid)\b", re.I
)


async def process_message(
    sender: str,
    message: str,
    name: str = "there",
    has_media: bool = False,
    media_url: str = "",
) -> str:
    """
    Main entry point. Returns a reply string to send back via WhatsApp.
    """
    # Resolve tenant
    tenant = _resolve_tenant(sender)

    # Load or create session
    session = await session_mgr.get(sender)
    if not session:
        session = await session_mgr.create(sender, tenant["tenant_id"])

    # Handle media messages
    if has_media:
        return "Thanks for the image! For now I can only process text messages. How can I help you? 😊"

    msg = message.strip()

    # ── Greeting / reset ──────────────────────────────────────────────────────
    if _GREETING_PATTERNS.match(msg):
        await session_mgr.clear_flow(sender)
        return _greeting(name, tenant)

    # ── Active payment flow ───────────────────────────────────────────────────
    if session.get("flow") == "payment":
        return await _handle_payment_flow(sender, msg, session, tenant)

    # ── Payment status check ──────────────────────────────────────────────────
    if _PAYMENT_STATUS_PATTERNS.search(msg) and session.get("last_checkout_id"):
        return await _check_payment_status(session["last_checkout_id"])

    # ── Payment initiation ────────────────────────────────────────────────────
    if _PAYMENT_PATTERNS.search(msg):
        await session_mgr.set_flow(sender, "payment", step="ask_amount")
        return (
            f"💳 *M-Pesa Payment*\n\n"
            f"How much would you like to pay? (in KES)\n\n"
            f"_Reply with the amount e.g. 500_"
        )

    # ── Human escalation ─────────────────────────────────────────────────────
    if _HUMAN_PATTERNS.search(msg):
        return _escalation_message(tenant)

    # ── RAG + LLM answer ─────────────────────────────────────────────────────
    return await _answer_question(sender, msg, session, tenant, name)


async def _answer_question(
    sender: str, msg: str, session: dict, tenant: dict, name: str
) -> str:
    """Retrieve from KB and generate an answer with OpenAI."""
    try:
        # 1. Retrieve relevant context from internal KB
        rag_result = await karisma_retrieve(
            query=msg,
            tenant_id=tenant["tenant_id"],
            category=tenant.get("category", ""),
            top_k=tenant.get("top_k", 4),
        )

        context_chunks = rag_result.get("context", [])
        cached_answer  = rag_result.get("answer")
        query_vector   = rag_result.get("query_vector")

        # 2. If semantic cache hit, use it directly (no OpenAI call needed)
        if cached_answer:
            answer = cached_answer
        elif context_chunks:
            # 3. Generate answer with OpenAI using retrieved context
            answer = await generate_answer(
                query=msg,
                context=context_chunks,
                tenant=tenant,
                name=name,
            )
            # 4. Store in semantic cache for future similar queries
            if query_vector:
                try:
                    await cache_answer(
                        query=msg,
                        query_vector=query_vector,
                        tenant_id=tenant["tenant_id"],
                        context=context_chunks,
                        answer=answer,
                    )
                except Exception as cache_err:
                    log.warning(f"[processor] cache_answer failed (non-fatal): {cache_err}")
        else:
            answer = _fallback_message(tenant)

        # 5. Track in session
        await session_mgr.add_turn(sender, msg, answer)
        return _format_whatsapp(answer)

    except Exception as exc:
        log.exception(f"[processor] RAG/LLM error: {exc}")
        return _fallback_message(tenant)


async def _handle_payment_flow(
    sender: str, msg: str, session: dict, tenant: dict
) -> str:
    """Multi-turn payment flow: collect amount → phone → confirm → STK push."""
    step = session.get("flow_step", "ask_amount")

    if step == "ask_amount":
        # Validate amount
        amount = _extract_number(msg)
        if not amount or amount < 1:
            return "Please enter a valid amount in KES (numbers only). E.g. *500*"
        await session_mgr.update_flow(sender, step="ask_phone", amount=amount)
        phone = sender.replace("whatsapp:", "").replace("+", "")
        return (
            f"💳 Amount: *KES {amount:,}*\n\n"
            f"Which phone number should receive the M-Pesa prompt?\n"
            f"_(Press 1 to use {sender.replace('whatsapp:','')}, or type another number)_"
        )

    elif step == "ask_phone":
        stored_phone = sender.replace("whatsapp:", "")
        if msg.strip() == "1":
            phone = stored_phone
        else:
            phone = _normalize_phone(msg)
            if not phone:
                return "Please enter a valid Kenyan phone number. E.g. *0712345678*"
        await session_mgr.update_flow(sender, step="confirm", phone=phone)
        amount = session.get("amount", 0)
        return (
            f"✅ *Confirm Payment*\n\n"
            f"Amount: *KES {amount:,}*\n"
            f"Phone: *{phone}*\n"
            f"To: *{tenant.get('business_name', 'Business')}*\n\n"
            f"Reply *YES* to pay or *NO* to cancel."
        )

    elif step == "confirm":
        if msg.upper() in ("YES", "Y", "NDIO", "OK", "CONFIRM"):
            amount = session.get("amount", 0)
            phone  = session.get("phone", "")
            result = await initiate_stk_push(
                phone=phone,
                amount=amount,
                account_ref=tenant.get("mpesa_account_ref", "WC"),
                description=f"Payment to {tenant.get('business_name', 'Business')}",
                tenant=tenant,
            )
            if result.get("success"):
                checkout_id = result.get("checkout_request_id")
                await session_mgr.update_flow(
                    sender, step="waiting", checkout_id=checkout_id
                )
                await session_mgr.set_last_checkout(sender, checkout_id)
                return (
                    f"📱 *M-Pesa prompt sent!*\n\n"
                    f"Check your phone *{phone}* for the M-Pesa PIN prompt.\n"
                    f"Enter your PIN to complete the payment of *KES {amount:,}*.\n\n"
                    f"_Reply *STATUS* to check payment status._"
                )
            else:
                await session_mgr.clear_flow(sender)
                return f"❌ Payment failed: {result.get('message', 'Please try again.')} "
        else:
            await session_mgr.clear_flow(sender)
            return "Payment cancelled. How else can I help you? 😊"

    # Unknown step — reset
    await session_mgr.clear_flow(sender)
    return "Something went wrong with the payment flow. Please type *pay* to start again."


async def _check_payment_status(checkout_id: str) -> str:
    result = await get_payment_status(checkout_id)
    if result.get("status") == "completed":
        return f"✅ *Payment confirmed!* Your M-Pesa payment was received successfully."
    elif result.get("status") == "pending":
        return f"⏳ *Payment pending.* Please check your phone and enter your M-Pesa PIN."
    else:
        return f"❌ *Payment not completed.* {result.get('message', 'Please try again.')}"


def _resolve_tenant(sender: str) -> dict:
    """Resolve tenant config from sender number. Falls back to demo tenant."""
    if is_demo_number(sender):
        return get_demo_tenant()
    # In production this would look up a DB — for now use env config
    return {
        "tenant_id":       os.getenv("DEFAULT_TENANT_ID", "demo"),
        "category":        os.getenv("DEFAULT_CATEGORY", "general"),
        "business_name":   os.getenv("BUSINESS_NAME", "WC Demo Business"),
        "top_k":           int(os.getenv("TOP_K", "4")),
        "mpesa_shortcode": os.getenv("MPESA_SHORTCODE", ""),
        "mpesa_passkey":   os.getenv("MPESA_PASSKEY", ""),
        "mpesa_account_ref": os.getenv("MPESA_ACCOUNT_REF", "WC"),
        "human_agent_msg": os.getenv(
            "HUMAN_AGENT_MSG",
            "I'll connect you to a human agent shortly. Please hold. 🙏"
        ),
    }


def _greeting(name: str, tenant: dict) -> str:
    biz = tenant.get("business_name", "us")
    return (
        f"👋 Hello *{name}*! Welcome to *{biz}*.\n\n"
        f"I'm your AI assistant. I can help you with:\n"
        f"• 🔍 Questions & information\n"
        f"• 💳 M-Pesa payments\n"
        f"• 📞 Connecting to an agent\n\n"
        f"What can I help you with today?"
    )


def _escalation_message(tenant: dict) -> str:
    return tenant.get(
        "human_agent_msg",
        "I'll connect you to a human agent shortly. Please hold. 🙏"
    )


def _fallback_message(tenant: dict) -> str:
    return (
        f"I'm sorry, I couldn't find an answer to that. "
        f"Would you like me to connect you to a human agent? "
        f"Reply *AGENT* to be connected. 🙏"
    )


def _format_whatsapp(text: str) -> str:
    """Clean up LLM output for WhatsApp formatting."""
    # Remove markdown that doesn't render in WhatsApp
    text = re.sub(r"#{1,6}\s+", "*", text)          # headings → bold
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)  # ** → * (WhatsApp bold)
    text = re.sub(r"__(.+?)__", r"_\1_", text)       # __ → _ (WhatsApp italic)
    text = re.sub(r"`{3}.*?`{3}", "", text, flags=re.DOTALL)  # remove code blocks
    # Trim to WhatsApp limit (4096 chars)
    if len(text) > 4000:
        text = text[:3997] + "..."
    return text.strip()


def _extract_number(text: str) -> float | None:
    m = re.search(r"[\d,]+\.?\d*", text.replace(",", ""))
    return float(m.group().replace(",", "")) if m else None


def _normalize_phone(phone: str) -> str | None:
    """Normalize to international format +2547XXXXXXXX."""
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("0") and len(digits) == 10:
        return "+254" + digits[1:]
    if digits.startswith("254") and len(digits) == 12:
        return "+" + digits
    if digits.startswith("7") and len(digits) == 9:
        return "+254" + digits
    return None
