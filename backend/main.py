"""
GARUDA Backend — FastAPI Application Entry Point
=================================================
Run with:
    uvicorn backend.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
ReDoc     : http://localhost:8000/redoc
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.config import get_settings
from .core.database import init_db, close_db
from .api import violations, cameras, vehicles, analytics, stream, debug, auth, jobs, reviews, evidence, users, audit_logs, agent

logger = logging.getLogger(__name__)
settings = get_settings()

_startup_time = time.time()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("GARUDA API starting up…")
    await init_db()
    logger.info("Database ready")
    yield
    # Shutdown
    await close_db()
    logger.info("GARUDA API shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GARUDA — Traffic Violation Detection API",
    description=(
        "Backend API for the GARUDA automated traffic violation detection system. "
        "Handles violation ingestion, officer review, vehicle registry, "
        "real-time WebSocket feed, and analytics.\n\n"
        "**Frontend developer**: See `BACKEND_REFERENCE.md` for full integration guide."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.0.114:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.debug("%s %s → %d (%.0fms)", request.method, request.url.path,
                 response.status_code, elapsed)
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(violations.router, prefix=API_PREFIX, tags=["Violations"])
app.include_router(cameras.router,    prefix=API_PREFIX, tags=["Cameras"])
app.include_router(vehicles.router,   prefix=API_PREFIX, tags=["Vehicles"])
app.include_router(analytics.router,  prefix=API_PREFIX, tags=["Analytics"])
app.include_router(stream.router,     prefix="",         tags=["Live Feed"])
app.include_router(debug.router,      prefix="/debug",   tags=["Debug"])
app.include_router(auth.router,       prefix=API_PREFIX, tags=["Authentication"])
app.include_router(jobs.router,       prefix=API_PREFIX, tags=["Jobs"])
app.include_router(reviews.router,    prefix=API_PREFIX, tags=["Reviews"])
app.include_router(evidence.router,   prefix=API_PREFIX, tags=["Evidence"])
app.include_router(users.router,      prefix=API_PREFIX, tags=["Users"])
app.include_router(audit_logs.router, prefix=API_PREFIX, tags=["Audit Logs"])
app.include_router(agent.router,      prefix=API_PREFIX, tags=["Gemma AI Agent"])

# Serve evidence images as static files
import os
os.makedirs("evidence/annotated", exist_ok=True)
app.mount("/evidence", StaticFiles(directory="evidence"), name="evidence")

# Serve test folder as static files
os.makedirs("test", exist_ok=True)
app.mount("/test-images", StaticFiles(directory="test"), name="test-images")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "GARUDA Traffic Violation Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "websocket": "/ws/feed",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime_sec": round(time.time() - _startup_time, 1),
    }


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
