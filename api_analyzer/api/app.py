"""FastAPI application factory for the API Attack Path Analyzer REST API.

Start with:
    uvicorn api_analyzer.api.app:app --reload

Environment variables
---------------------
  NEO4J_URI       bolt URI (default bolt://localhost:7687)
  NEO4J_USER      username (default neo4j)
  NEO4J_PASSWORD  password (default password)
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from neo4j import GraphDatabase

from api_analyzer import __version__
from api_analyzer.api.routes import router

# Auto-load .env from the project root so credentials are always available
load_dotenv(Path(__file__).parent.parent.parent / ".env")

_STATIC_DIR = Path(__file__).parent / "static"
_VALIDATION_GUIDE = Path(__file__).parent.parent.parent / "validation_guide.html"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Create the Neo4j driver once at startup; close it at shutdown."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    # .env uses NEO4J_USERNAME; also accept NEO4J_USER for CLI compatibility
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "attackpath")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    app.state.driver = driver
    try:
        yield
    finally:
        driver.close()


app = FastAPI(
    title="API Attack Path Analyzer",
    description=(
        "AI-powered API security analysis combining Neo4j knowledge graph traversal "
        "with Claude LLM reasoning to discover multi-hop exploit chains."
    ),
    version=__version__,
    lifespan=_lifespan,
)

# ── Optional API-key guard ─────────────────────────────────────────────────────
# Set ANALYZER_API_KEY in .env (or the host environment) to require callers to
# send   X-API-Key: <value>   on every request except GET /health and GET /.
# When the variable is not set the server is open (development default).
_ANALYZER_API_KEY: str | None = os.getenv("ANALYZER_API_KEY")

_OPEN_PATHS: frozenset[str] = frozenset({"/", "/health"})


@app.middleware("http")
async def _api_key_middleware(request: Request, call_next):
    if _ANALYZER_API_KEY and request.url.path not in _OPEN_PATHS:
        provided = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(provided, _ANALYZER_API_KEY):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-API-Key header."},
            )
    return await call_next(request)


app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def frontend():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/validation_guide.html", response_class=HTMLResponse, include_in_schema=False)
async def validation_guide():
    if _VALIDATION_GUIDE.exists():
        return FileResponse(_VALIDATION_GUIDE)
    return HTMLResponse("<h1>Validation guide not found</h1>", status_code=404)
