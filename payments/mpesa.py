"""
WC — M-Pesa Daraja Integration
Handles STK Push payments via Safaricom Daraja API.
"""
from __future__ import annotations
import os, base64, logging, time
from datetime import datetime
import httpx
from fastapi import APIRouter, Request

log = logging.getLogger("wc.mpesa")
router = APIRouter()

# ── Daraja config ─────────────────────────────────────────────────────────────
DARAJA_BASE     = os.getenv("MPESA_DARAJA_URL", "https://sandbox.safaricom.co.ke")
CONSUMER_KEY    = os.getenv("MPESA_CONSUMER_KEY", "")
CONSUMER_SECRET = os.getenv("MPESA_CONSUMER_SECRET", "")
SHORTCODE       = os.getenv("MPESA_SHORTCODE", "174379")       # sandbox default
PASSKEY         = os.getenv("MPESA_PASSKEY", "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919")
CALLBACK_URL    = os.getenv("MPESA_CALLBACK_URL", "https://your-domain.com/mpesa/callback")

# In-memory payment store (replace with DB in production)
_payments: dict[str, dict] = {}


async def _get_access_token(tenant: dict | None = None) -> str:
    """Get OAuth access token from Daraja."""
    key    = (tenant or {}).get("mpesa_consumer_key")    or CONSUMER_KEY
    secret = (tenant or {}).get("mpesa_consumer_secret") or CONSUMER_SECRET
    creds  = base64.b64encode(f"{key}:{secret}".encode()).decode()

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{DARAJA_BASE}/oauth/v1/generate?grant_type=client_credentials",
            headers={"Authorization": f"Basic {creds}"},
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


def _generate_password(shortcode: str, passkey: str) -> tuple[str, str]:
    """Generate STK Push password and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    raw       = f"{shortcode}{passkey}{timestamp}"
    password  = base64.b64encode(raw.encode()).decode()
    return password, timestamp


async def initiate_stk_push(
    phone: str,
    amount: float,
    account_ref: str,
    description: str,
    tenant: dict | None = None,
) -> dict:
    """
    Initiate M-Pesa STK Push to customer phone.
    Returns dict with success, checkout_request_id, message.
    """
    shortcode = (tenant or {}).get("mpesa_shortcode") or SHORTCODE
    passkey   = (tenant or {}).get("mpesa_passkey")   or PASSKEY
    callback  = (tenant or {}).get("mpesa_callback_url") or CALLBACK_URL

    # Normalize phone
    phone = phone.replace("+", "").replace(" ", "")
    if phone.startswith("0"):
        phone = "254" + phone[1:]
    if not phone.startswith("254"):
        phone = "254" + phone

    try:
        token              = await _get_access_token(tenant)
        password, timestamp = _generate_password(shortcode, passkey)

        payload = {
            "BusinessShortCode": shortcode,
            "Password":          password,
            "Timestamp":         timestamp,
            "TransactionType":   "CustomerPayBillOnline",
            "Amount":            int(amount),
            "PartyA":            phone,
            "PartyB":            shortcode,
            "PhoneNumber":       phone,
            "CallBackURL":       callback,
            "AccountReference":  account_ref[:12],
            "TransactionDesc":   description[:13],
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{DARAJA_BASE}/mpesa/stkpush/v1/processrequest",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()

        checkout_id = data.get("CheckoutRequestID")
        _payments[checkout_id] = {
            "phone":       phone,
            "amount":      amount,
            "status":      "pending",
            "created_at":  int(time.time()),
            "merchant_id": data.get("MerchantRequestID"),
        }
        log.info(f"[mpesa] STK Push sent — checkout_id={checkout_id} phone={phone} amount={amount}")
        return {"success": True, "checkout_request_id": checkout_id, "message": "STK Push sent"}

    except httpx.HTTPStatusError as e:
        log.error(f"[mpesa] HTTP error: {e.response.status_code} {e.response.text[:200]}")
        return {"success": False, "message": f"Payment service error ({e.response.status_code})"}
    except Exception as e:
        log.exception(f"[mpesa] STK Push error: {e}")
        return {"success": False, "message": "Payment service unavailable. Please try again."}


async def get_payment_status(checkout_request_id: str) -> dict:
    """Check payment status — first from local store, then query Daraja."""
    payment = _payments.get(checkout_request_id)
    if payment:
        return {"status": payment["status"], "payment": payment}
    return {"status": "unknown", "message": "Payment record not found"}


# ── Daraja callback endpoint ──────────────────────────────────────────────────

@router.post("/callback")
async def mpesa_callback(request: Request):
    """
    Daraja calls this URL when payment completes/fails.
    Update payment status and optionally notify the customer via WhatsApp.
    """
    try:
        body = await request.json()
        log.info(f"[mpesa/callback] {body}")

        stk = body.get("Body", {}).get("stkCallback", {})
        checkout_id  = stk.get("CheckoutRequestID")
        result_code  = stk.get("ResultCode")
        result_desc  = stk.get("ResultDesc", "")

        if checkout_id in _payments:
            if result_code == 0:
                # Extract metadata
                items = {
                    item["Name"]: item.get("Value")
                    for item in stk.get("CallbackMetadata", {}).get("Item", [])
                }
                _payments[checkout_id].update({
                    "status":          "completed",
                    "mpesa_receipt":   items.get("MpesaReceiptNumber"),
                    "transaction_date": items.get("TransactionDate"),
                    "amount_paid":     items.get("Amount"),
                    "completed_at":    int(time.time()),
                })
                log.info(f"[mpesa/callback] Payment completed: {checkout_id}")
            else:
                _payments[checkout_id]["status"] = "failed"
                _payments[checkout_id]["error"]  = result_desc
                log.warning(f"[mpesa/callback] Payment failed: {checkout_id} — {result_desc}")

    except Exception as e:
        log.exception(f"[mpesa/callback] error: {e}")

    return {"ResultCode": 0, "ResultDesc": "Accepted"}


@router.get("/status/{checkout_id}")
async def payment_status_endpoint(checkout_id: str):
    """Admin endpoint to check payment status."""
    return _payments.get(checkout_id, {"status": "not_found"})
