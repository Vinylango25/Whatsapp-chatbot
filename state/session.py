"""
WC — Session Manager
Redis-backed conversation state per WhatsApp sender.
Falls back to in-memory dict if Redis is unavailable.
"""
from __future__ import annotations
import os, json, logging, time
from typing import Optional

log = logging.getLogger("wc.session")

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TTL  = int(os.getenv("SESSION_TTL_SECONDS", "1800"))   # 30 min


class SessionManager:
    def __init__(self):
        self._redis = None
        self._memory: dict[str, dict] = {}   # fallback in-memory store
        self._try_connect()

    def _try_connect(self):
        try:
            import redis.asyncio as aioredis
            self._redis_cls = aioredis
            log.info(f"[session] Redis configured at {REDIS_URL}")
        except ImportError:
            log.warning("[session] redis package not installed — using in-memory sessions")

    async def _get_redis(self):
        if not hasattr(self, "_redis_cls"):
            return None
        if self._redis is None:
            try:
                self._redis = self._redis_cls.from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
                log.info("[session] Redis connected")
            except Exception as e:
                log.warning(f"[session] Redis unavailable ({e}) — using in-memory")
                self._redis = None
        return self._redis

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(self, sender: str) -> Optional[dict]:
        key = f"wc:session:{sender}"
        r = await self._get_redis()
        if r:
            try:
                raw = await r.get(key)
                return json.loads(raw) if raw else None
            except Exception:
                pass
        return self._memory.get(key)

    async def create(self, sender: str, tenant_id: str) -> dict:
        session = {
            "sender":    sender,
            "tenant_id": tenant_id,
            "created_at": int(time.time()),
            "turn_count": 0,
            "flow":       None,
            "flow_step":  None,
            "history":    [],
        }
        await self._save(sender, session)
        return session

    async def add_turn(self, sender: str, user_msg: str, bot_reply: str):
        session = await self.get(sender) or {}
        history = session.get("history", [])
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_reply})
        # Keep last 10 turns only
        session["history"]    = history[-20:]
        session["turn_count"] = session.get("turn_count", 0) + 1
        await self._save(sender, session)

    async def set_flow(self, sender: str, flow: str, step: str = "", **kwargs):
        session = await self.get(sender) or {}
        session["flow"]      = flow
        session["flow_step"] = step
        session.update(kwargs)
        await self._save(sender, session)

    async def update_flow(self, sender: str, step: str, **kwargs):
        session = await self.get(sender) or {}
        session["flow_step"] = step
        session.update(kwargs)
        await self._save(sender, session)

    async def clear_flow(self, sender: str):
        session = await self.get(sender) or {}
        session["flow"]      = None
        session["flow_step"] = None
        # Clear flow-specific keys
        for k in ("amount", "phone", "checkout_id"):
            session.pop(k, None)
        await self._save(sender, session)

    async def set_last_checkout(self, sender: str, checkout_id: str):
        session = await self.get(sender) or {}
        session["last_checkout_id"] = checkout_id
        await self._save(sender, session)

    async def delete(self, sender: str):
        key = f"wc:session:{sender}"
        r = await self._get_redis()
        if r:
            try:
                await r.delete(key)
                return
            except Exception:
                pass
        self._memory.pop(key, None)

    # ── Private ───────────────────────────────────────────────────────────────

    async def _save(self, sender: str, session: dict):
        key = f"wc:session:{sender}"
        r = await self._get_redis()
        if r:
            try:
                await r.setex(key, SESSION_TTL, json.dumps(session))
                return
            except Exception:
                pass
        self._memory[key] = session
