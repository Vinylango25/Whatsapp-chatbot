"""
WC KB — Embedder
Supports OpenAI embeddings (default) and Ollama nomic-embed-text (local/free).
Auto-detects based on EMBEDDING_PROVIDER env var.
"""
from __future__ import annotations
import os, logging, numpy as np
from typing import List

log = logging.getLogger("wc.kb.embedder")

PROVIDER   = os.getenv("EMBEDDING_PROVIDER", "openai")   # openai | ollama
MODEL      = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DIM        = int(os.getenv("EMBEDDING_DIM", "1536"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


class Embedder:
    def __init__(self):
        self.provider = PROVIDER
        self.model    = MODEL
        self.dim      = DIM
        log.info(f"[embedder] provider={self.provider} model={self.model} dim={self.dim}")

    async def embed(self, text: str) -> List[float]:
        """Embed a single text string. Returns a list of floats."""
        if not text or not text.strip():
            return [0.0] * self.dim
        if self.provider == "openai":
            return await self._openai_embed(text)
        elif self.provider == "ollama":
            return await self._ollama_embed(text)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        if self.provider == "openai":
            return await self._openai_embed_batch(texts)
        return [await self.embed(t) for t in texts]

    # ── OpenAI ────────────────────────────────────────────────────────────────

    async def _openai_embed(self, text: str) -> List[float]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp   = await client.embeddings.create(
            input=text[:8000],
            model=self.model,
        )
        return resp.data[0].embedding

    async def _openai_embed_batch(self, texts: List[str]) -> List[List[float]]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        # OpenAI batch limit = 2048 inputs
        results = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = [t[:8000] for t in texts[i:i+batch_size]]
            resp  = await client.embeddings.create(input=batch, model=self.model)
            results.extend([d.embedding for d in resp.data])
        return results

    # ── Ollama ────────────────────────────────────────────────────────────────

    async def _ollama_embed(self, text: str) -> List[float]:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            emb = resp.json()["embedding"]
        arr = np.array(emb, dtype=np.float32)
        n   = np.linalg.norm(arr)
        return (arr / n).tolist() if n > 0 else arr.tolist()


# Singleton
_embedder: Embedder | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
