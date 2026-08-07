"""
WC — Demo Sandbox
Pre-configured demo tenant for LinkedIn/sales demonstrations.
Anyone who messages the Twilio sandbox gets this demo experience.
"""
from __future__ import annotations
import os

# Demo numbers — add Twilio sandbox number here
_DEMO_NUMBERS = set(
    os.getenv("DEMO_NUMBERS", "").split(",")
)
# Twilio sandbox always starts with whatsapp:+14155238886 by default
_DEMO_NUMBERS.add("whatsapp:+14155238886")


def is_demo_number(sender: str) -> bool:
    """Check if sender is using the demo/sandbox number."""
    # If no specific tenant configured, everyone is demo
    if not os.getenv("DEFAULT_TENANT_ID"):
        return True
    return sender in _DEMO_NUMBERS or any(sender.endswith(n) for n in _DEMO_NUMBERS if n)


def get_demo_tenant() -> dict:
    """Return demo tenant configuration."""
    return {
        "tenant_id":       "demo",
        "category":        os.getenv("DEMO_CATEGORY", "general"),
        "business_name":   os.getenv("DEMO_BUSINESS_NAME", "WC AI Demo"),
        "top_k":           4,
        "mpesa_shortcode": "174379",        # Safaricom sandbox shortcode
        "mpesa_passkey":   "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919",
        "mpesa_account_ref": "DEMO",
        "mpesa_callback_url": os.getenv("MPESA_CALLBACK_URL", ""),
        "human_agent_msg": (
            "🎯 *Demo Mode*\n\n"
            "This is a live demo of WC — your AI-powered WhatsApp business assistant.\n\n"
            "In a real deployment, this would connect to your human agent team.\n\n"
            "Interested in deploying WC for your business? "
            "Contact us at: " + os.getenv("CONTACT_EMAIL", "hello@wc.ai")
        ),
        "system_prompt_addition": (
            "This is a demo chatbot. The user is evaluating WC for their business. "
            "Be impressive, helpful, and demonstrate the capabilities clearly."
        ),
    }
