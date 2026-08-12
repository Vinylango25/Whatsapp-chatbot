"""
Vercel ASGI entrypoint.
Vercel serverless functions require a handler — Mangum wraps FastAPI for this.
"""
import sys
import os

# Ensure project root is on the path so all imports (kb, rag, etc.) resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from mangum import Mangum

# Vercel invokes this `handler` object
handler = Mangum(app, lifespan="off")
