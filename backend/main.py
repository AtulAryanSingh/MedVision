"""
MedVision – FastAPI application entry point  (v1.0 production build)

Why this file exists:
  Top-level app factory.  Creates the FastAPI instance, configures CORS,
  mounts a static-file directory for uploads, and registers every API router.

  Adding a new router later:
    1. Create  backend/api/<feature>.py  with an APIRouter.
    2. Import and include it here – nothing else needs to change.
"""

import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import config

# ── Authentication ─────────────────────────────────────────────────────────
import db as _db
from api.auth import router as auth_router
from api.deps import get_current_user

# ── New production API routers ─────────────────────────────────────────────
from api.upload   import router as upload_router
from api.preview  import router as preview_router
from api.mpr      import router as mpr_router
from api.process  import router as process_router
from api.features import router as features_router
from api.cluster  import router as cluster_router
from api.report   import router as report_router
from api.patchify import router as patchify_router
from api.export    import router as export_router
from api.register  import router as register_router

# ── Legacy Core Imaging Lab router (v0.1, kept for backward compatibility) ─
from labs.core_imaging.routes import router as core_imaging_router

# ── App factory ────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedVision API",
    version="1.0.0",
    description=(
        "Modular medical-imaging intelligence platform — "
        "Core Imaging · Processing · Features · ML · Analysis"
    ),
)

# Initialise the users database (creates table + default admin if empty)
_db.init_db()

# ── CORS ───────────────────────────────────────────────────────────────────
# Allow all common localhost variants for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ────────────────────────────────────────────────────
UPLOAD_DIR = config.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ── Auth router (unprotected – issues tokens) ──────────────────────────────
app.include_router(auth_router, prefix="/api", tags=["Auth"])

# ── Protected API routers (require valid JWT) ──────────────────────────────
_auth = [Depends(get_current_user)]

app.include_router(upload_router,   prefix="/api", tags=["Upload"],             dependencies=_auth)
app.include_router(preview_router,  prefix="/api", tags=["Preview"],            dependencies=_auth)
app.include_router(mpr_router,      prefix="/api", tags=["MPR"],                dependencies=_auth)
app.include_router(process_router,  prefix="/api", tags=["Processing"],         dependencies=_auth)
app.include_router(features_router, prefix="/api", tags=["Features"],           dependencies=_auth)
app.include_router(cluster_router,  prefix="/api", tags=["ML / Clustering"],    dependencies=_auth)
app.include_router(report_router,   prefix="/api", tags=["Analysis Report"],    dependencies=_auth)
app.include_router(patchify_router, prefix="/api", tags=["Patchify"],           dependencies=_auth)
app.include_router(export_router,   prefix="/api", tags=["Export"],             dependencies=_auth)
app.include_router(register_router, prefix="/api", tags=["Registration"],       dependencies=_auth)

# Legacy v0.1 endpoints (still functional, protected)
app.include_router(
    core_imaging_router,
    prefix="/api/core-imaging",
    tags=["Core Imaging Lab (legacy)"],
    dependencies=_auth,
)

# ── Health check (unprotected) ─────────────────────────────────────────────

@app.get("/health", tags=["Health"])
@app.get("/", tags=["Health"])
def root():
    """Return a simple health-check payload so operators can verify the API is up."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "modules": ["upload", "preview", "processing", "features", "ml", "analysis"],
    }
