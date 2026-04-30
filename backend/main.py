"""
MedVision – FastAPI application entry point.

Why this file exists:
  This is the top-level app factory.  It creates the FastAPI instance,
  configures CORS (so the local frontend on a different port can reach the
  backend without browser security errors), mounts a static-file directory
  so processed images can be served directly by URL, and registers one
  APIRouter per lab.

  Adding a new lab later means:
    1. Create  backend/labs/<new_lab>/routes.py  with an APIRouter.
    2. Import and include it here (see the "Future labs" block below).
  Nothing else needs to change.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Core Imaging Lab – the only active lab in v0.1
from labs.core_imaging.routes import router as core_imaging_router

# ── App factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedVision API",
    version="0.1.0",
    description="Modular medical-imaging platform – Core Imaging Lab (v0.1)",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow the local frontend (any localhost port) to call the backend during
# development.  Restrict this list in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        # Allow file:// origins for opening index.html directly in a browser
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
# Serve the uploads folder so processed images can be referenced by URL.
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ── Routers ───────────────────────────────────────────────────────────────────

# Active lab – Core Imaging Lab
app.include_router(core_imaging_router, prefix="/api/core-imaging", tags=["Core Imaging Lab"])

# ── Future labs (placeholders – uncomment when each lab is implemented) ───────
# from labs.dicom_lab.routes        import router as dicom_router
# from labs.classical_pipeline.routes import router as classical_router
# from labs.pattern_insight.routes  import router as pattern_router
# from labs.ai_lab.routes           import router as ai_router
#
# app.include_router(dicom_router,     prefix="/api/dicom",     tags=["DICOM Lab"])
# app.include_router(classical_router, prefix="/api/classical", tags=["Classical Pipeline"])
# app.include_router(pattern_router,   prefix="/api/pattern",   tags=["Pattern Insight Lab"])
# app.include_router(ai_router,        prefix="/api/ai",        tags=["AI/Deep Learning Lab"])


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Return a simple health-check payload so operators can verify the API is up."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "active_labs": ["core_imaging"],
        "future_labs": ["dicom_lab", "classical_pipeline", "pattern_insight", "ai_lab"],
    }
