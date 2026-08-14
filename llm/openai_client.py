"""
WC — Groq LLM Client (Free)
Generates WhatsApp-friendly answers using Groq's free API (LLaMA 3).
Sign up free at https://console.groq.com — no credit card needed.
"""
from __future__ import annotations
import os, logging
from groq import AsyncGroq
from rag.client import format_context_for_llm

log = logging.getLogger("wc.llm")

_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", ""))

MODEL       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")   # free, fast
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "600"))

_SYSTEM_PROMPT = """You are a helpful WhatsApp AI assistant for a business.
Your job is to answer customer questions accurately and concisely based ONLY on the provided context.

RULES:
- Answer in a friendly, conversational tone suitable for WhatsApp
- Keep answers concise (max 3-4 short paragraphs)
- Use simple language, avoid jargon
- Use WhatsApp formatting: *bold* for emphasis, bullet points with •
- If the answer is not in the context, say: "I don't have that information. Would you like to speak to a human agent?"
- NEVER make up information not in the context
- If asked about prices, policies, or procedures — quote directly from the context
- End with a helpful follow-up question when appropriate"""


async def generate_answer(
    query: str,
    context: list[dict],
    tenant: dict,
    name: str = "there",
) -> str:
    """
    Generate a WhatsApp-friendly answer using Groq (free LLaMA 3).
    """
    if not context:
        return "I couldn't find relevant information for that. Can you rephrase your question?"

    context_text = format_context_for_llm(context)
    business     = tenant.get("business_name", "us")

    system = _SYSTEM_PROMPT
    custom = tenant.get("system_prompt_addition", "")
    if custom:
        system += f"\n\nBusiness context: {custom}"

    user_msg = (
        f"Customer name: {name}\n"
        f"Business: {business}\n\n"
        f"Context from knowledge base:\n{context_text}\n\n"
        f"Customer question: {query}\n\n"
        f"Answer the question based on the context above."
    )

    try:
        response = await _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        log.exception(f"[llm] Groq error: {e}")
        return "I'm having trouble generating a response right now. Please try again in a moment. 🙏"
