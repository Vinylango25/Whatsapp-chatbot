"""
WC KB — Embedder
Supports multiple embedding providers:
  - huggingface (default, FREE) — uses HuggingFace Inference API
  - ollama (FREE, local)        — requires Ollama running locally
  - openai (paid)               — requires OPENAI_API_KEY with credits
  - groq (FREE)                 — uses Groq API (free, fast)

Set EMBEDDING_PROVIDER in .env to choose.
"""
from __future__ import annotations
import os, logging, numpy as np
from typing import List

log = logging.getLogger("wc.kb.embedder")

PROVIDER      = os.getenv("EMBEDDING_PROVIDER", "huggingface")
MODEL         = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DIM           = int(os.getenv("EMBEDDING_DIM", "384"))   # all-MiniLM-L6-v2 = 384 dims
HF_API_KEY    = os.getenv("HF_API_KEY", "")             # free at huggingface.co
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")           # free at console.groq.com


class Embedder:
    def __init__(self):
        self.provider = PROVIDER
        self.model    = MODEL
        self.dim      = DIM
        log.info(f"[embedder] provider={self.provider} model={self.model} dim={self.dim}")

    async def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dim
        if self.provider == "huggingface":
            return await self._hf_embed(text)
        elif self.provider == "openai":
            return await self._openai_embed(text)
        elif self.provider == "ollama":
            return await self._ollama_embed(text)
        elif self.provider == "groq":
            return await self._groq_embed(text)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "huggingface":
            return await self._hf_embed_batch(texts)
        elif self.provider in ("openai", "groq"):
            return await self._openai_embed_batch(texts)
        return [await self.embed(t) for t in texts]

    # ── HuggingFace Inference API (FREE) ──────────────────────────────────────

    async def _hf_embed(self, text: str) -> List[float]:
        import httpx
        headers = {"Content-Type": "application/json"}
        if HF_API_KEY:
            headers["Authorization"] = f"Bearer {HF_API_KEY}"

        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json={"inputs": text[:512]}, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        # HF returns list of lists for sentences — take first (or mean pool)
        if isinstance(result[0], list):
            # sentence-transformers models return [[vec]] — take first
            emb = result[0] if isinstance(result[0][0], float) else result[0][0]
        else:
            emb = result

        arr = np.array(emb, dtype=np.float32)
        n   = np.linalg.norm(arr)
        return (arr / n).tolist() if n > 0 else arr.tolist()

    async def _hf_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch via HF — process in chunks to avoid timeouts."""
        results = []
        # HF free tier: process one at a time to avoid rate limits
        for text in texts:
            vec = await self._hf_embed(text)
            results.append(vec)
        return results

    # ── OpenAI (paid) ─────────────────────────────────────────────────────────

    async def _openai_embed(self, text: str) -> List[float]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp   = await client.embeddings.create(input=text[:8000], model=self.model)
        return resp.data[0].embedding

    async def _openai_embed_batch(self, texts: List[str]) -> List[List[float]]:
        from openai import AsyncOpenAI
        
        # Use Groq if that's the provider, otherwise OpenAI
        if self.provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
            api_key = os.getenv("GROQ_API_KEY", "")
        else:
            base_url = "https://api.openai.com/v1"
            api_key = os.getenv("OPENAI_API_KEY", "")
        
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        results = []
        for i in range(0, len(texts), 100):
            batch = [t[:8000] for t in texts[i:i+100]]
            resp  = await client.embeddings.create(input=batch, model=self.model)
            results.extend([d.embedding for d in resp.data])
        return results

    # ── Ollama (free, local) ──────────────────────────────────────────────────

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

    # ── Groq (FREE) ───────────────────────────────────────────────────────────

    async def _groq_embed(self, text: str) -> List[float]:
        """Embed using Groq's inference API (free tier)."""
        import httpx
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        # Groq uses OpenAI-compatible embedding endpoint
        url = "https://api.groq.com/openai/v1/embeddings"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json={"input": text[:8000], "model": self.model},
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]


# Singleton
_embedder: Embedder | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
