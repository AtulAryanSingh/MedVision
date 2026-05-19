# MedVision 🔬

> **v2 – Multi-page Modular Medical-Imaging Workbench**  
> FastAPI backend · React + Vite frontend · White-background clinical UI

A modular imaging workflow platform for clinical data operations (DICOM, NIfTI, PNG/JPEG)
that bridges raw imaging handling and ML experimentation.
Every module is **independent** — run Gaussian blur, Sobel edges, morphology, segmentation,
patchify, registration, clustering, reporting, or export in **any order, on demand**.

---

## Pages

| Page | What it does |
|---|---|
| 🏠 **Home** | Landing page with feature cards linking to each section |
| 📂 **Data Manager** | Upload DICOM / NIfTI / PNG / JPEG; displays shape, spacing, FOV |
| 🔬 **Workspace** | Multi-Planar Reconstruction (Axial · Coronal · Sagittal) with geometrically-correct aspect ratios using voxel spacing; window/level presets |
| 📊 **QC & Plots** | Intensity histogram, CDF, statistical features (mean/std/entropy/skewness/kurtosis), percentile strip, W/L reference table |
| ⚗️ **Tools** | Modular independent tools: Gaussian, Median, Sobel, CDF-threshold, Erosion/Dilation/Opening/Closing, Connected Components, Bounding Boxes |
| 🧩 **Patchify 3D** | Dice a 3-D volume into cubic patches; configurable size & stride; download .npz (patches + coords + spacing) |
| 📐 **Registration** | Apply geometric transforms (Translation · Rotation · Zoom · Affine) with bicubic or nearest-neighbour interpolation |
| 🧬 **Deep Learning** | Recommended DL workflow + downloadable Google Colab notebook (3-D CNN / MONAI U-Net skeleton) |
| 📥 **Downloads** | Export as PNG, NumPy array (.npy), component metrics CSV (.csv) |

---

## Architecture

```
MedVision/
├── backend/
│   ├── main.py                  # FastAPI app: CORS, routers, static files
│   ├── requirements.txt
│   ├── data/
│   │   ├── uploads/             # auto-created; not committed
│   │   └── cache/               # per-image JSON metadata; not committed
│   ├── api/
│   │   ├── upload.py            # POST /api/upload
│   │   ├── preview.py           # GET  /api/preview/{id}
│   │   ├── mpr.py               # GET  /api/mpr/{id}  ← spacing-correct MPR + FOV + W/L
│   │   ├── process.py           # POST /api/process   ← 10 independent tools
│   │   ├── features.py          # POST /api/features
│   │   ├── cluster.py           # POST /api/cluster   ← KMeans + PCA
│   │   ├── report.py            # GET  /api/report/{id}
│   │   ├── patchify.py          # POST /api/patchify  ← 3-D patch extraction → NPZ
│   │   └── export.py            # GET  /api/export/{id}/png|npy|csv
│   ├── core/
│   │   └── loader.py            # DICOM · NIfTI · PNG/JPG → float32 NumPy + metadata
│   ├── processing/
│   │   ├── filters.py           # Gaussian, Median, Sobel
│   │   ├── histogram.py         # Histogram, CDF, CDF-threshold
│   │   └── morphology.py        # Erosion, Dilation, Opening, Closing, CC-labeling
│   └── features/
│       └── extractor.py         # Statistical feature vector
│
└── frontend/
    ├── src/
    │   ├── App.jsx              # Sidebar layout + page router
    │   ├── api/client.js        # Centralised fetch wrapper
    │   ├── pages/
    │   │   ├── Home.jsx
    │   │   ├── DataManager.jsx
    │   │   ├── Workspace.jsx    # MPR viewer
    │   │   ├── QCPlots.jsx      # Histogram + CDF + stats
    │   │   ├── Tools.jsx        # Modular toolbox
    │   │   ├── Patchify.jsx     # 3-D patch extraction
│   │   │   ├── Registration.jsx # Translation · Rotation · Zoom · Affine
    │   │   ├── DeepLearning.jsx # DL workflow + Colab export
    │   │   └── Downloads.jsx    # Export centre
    │   └── styles/index.css     # White clinical theme
    └── package.json
```

---

## Run Instructions

### Prerequisites

```
Python 3.10+
Node.js 18+
```

### 1 – Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
# API docs: http://127.0.0.1:8000/docs
```

### 2 – Frontend

```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

Frontend now includes built-in **Signup** and **Login** screens.  
After login, the JWT is persisted in `localStorage` and automatically sent as
`Authorization: Bearer <token>` on API calls.

### 3 – Production build

```bash
cd frontend
npm run build          # outputs to frontend/dist/
npm run preview        # serve the production build locally
```

### 4 – Docker (backend + frontend with compose)

```bash
# from repository root
cp .env.example .env
docker compose up --build

# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
```

Backend runs in production mode with Gunicorn + Uvicorn workers.
See `docs/DEPLOYMENT.md` for deployment details.

### 5 – Backend tests

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest -q

# Optional coverage
pytest --cov=. --cov-report=term-missing
```

---

## Supported Formats

| Format | Extension | Notes |
|---|---|---|
| DICOM | `.dcm` | HU rescaling, PixelSpacing, SliceThickness extracted |
| NIfTI | `.nii`, `.nii.gz` | Full voxel spacing from header |
| PNG / JPEG | `.png`, `.jpg`, `.jpeg` | 2-D images, any size |

---

## Backend API Reference

The API now supports both compatibility synchronous routes and async orchestration routes
for long-running imaging workloads.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a file; returns `image_id` + metadata |
| `GET` | `/api/preview/{id}` | Raw axial/coronal/sagittal slices (no aspect correction) |
| `GET` | `/api/mpr/{id}` | Spacing-corrected MPR + FOV + optional W/L |
| `POST` | `/api/process` | Run one processing tool (gaussian, median, sobel, …) |
| `POST` | `/api/features` | Extract statistical feature vector |
| `POST` | `/api/cluster` | KMeans + PCA segmentation |
| `GET` | `/api/report/{id}` | Full analysis report JSON |
| `POST` | `/api/patchify` | Extract 3-D patches → NPZ |
| `POST` | `/api/register` | Apply geometric transform (translate/rotate/zoom/affine) |
| `GET` | `/api/export/{id}/png` | Download middle slice as PNG |
| `GET` | `/api/export/{id}/npy` | Download full array as .npy (base64) |
| `GET` | `/api/export/{id}/npy/stream` | Stream full array as binary `.npy` |
| `GET` | `/api/export/{id}/csv` | Download component metrics as CSV |
| `POST` | `/api/jobs/patchify` | Queue async patchify job |
| `POST` | `/api/jobs/cluster` | Queue async cluster job |
| `POST` | `/api/jobs/report/{id}` | Queue async report-generation job |
| `POST` | `/api/jobs/register` | Queue async registration job |
| `GET` | `/api/jobs/{job_id}` | Query job status and progress |
| `POST` | `/api/jobs/{job_id}/cancel` | Request job cancellation |
| `GET` | `/api/jobs/{job_id}/result` | Fetch completed job result |
| `GET` | `/api/jobs/{job_id}/result/stream` | Stream artifact result (when available) |

Full interactive docs at **`http://127.0.0.1:8000/docs`** (Swagger UI).

---

## Processing Tools (`POST /api/process`)

| `processing_type` | Parameters | Description |
|---|---|---|
| `gaussian` | `sigma` (0.1–20) | Gaussian blur |
| `median` | `kernel_size` (3–21) | Median filter |
| `sobel` | — | Sobel edge magnitude |
| `cdf_threshold` | `percentile` (1–99.9) | CDF-based binarisation |
| `erosion` | `kernel_size` | Morphological erosion |
| `dilation` | `kernel_size` | Morphological dilation |
| `opening` | `kernel_size` | Morphological opening |
| `closing` | `kernel_size` | Morphological closing |
| `connected_components` | `threshold` (0–255) | CC labelling + colour image |
| `bounding_boxes` | `threshold` (0–255) | Bounding-box overlay |

Each tool returns `result_image` (base64 PNG) + `histogram_before` + `histogram_after`.
For production workflows, prefer async job routes for heavier operations.

---

## Patch Export Format (`.npz`)

```python
import numpy as np

data    = np.load("patches_*.npz")
patches = data["patches"]   # float32  (N, P, P, P)
coords  = data["coords"]    # int32    (N, 3)   [z, y, x] top-left corner
spacing = data["spacing"]   # float32  (3,)     mm/voxel
```

---

## Backend Testing

The backend has a comprehensive pytest-based test suite covering unit tests for
core modules and integration tests for all FastAPI endpoints.

### Test Layout

```
backend/
└── tests/
    ├── conftest.py                # shared fixtures (TestClient, synthetic data, auth)
    ├── unit/
    │   ├── test_loader.py         # core/loader.py
    │   ├── test_filters.py        # processing/filters.py
    │   ├── test_histogram.py      # processing/histogram.py
    │   ├── test_morphology.py     # processing/morphology.py
    │   ├── test_extractor.py      # features/extractor.py
    │   └── test_image_cache.py    # core/image_cache.py (LRU cache)
    └── integration/
        ├── test_health.py         # GET /  and  GET /health
        ├── test_auth.py           # POST /api/auth/login  + JWT validation
        ├── test_upload.py         # POST /api/upload  (PNG, NIfTI, error cases)
        └── test_process.py        # POST /api/process (all 10 processing tools)
```

### Install Test Dependencies

```bash
cd backend
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | random (dev only) | HMAC-SHA256 signing secret. Set for stable tokens. |
| `BCRYPT_ROUNDS` | `12` | BCrypt work factor used for password hashing. |

> **Tip:** `conftest.py` sets `JWT_SECRET` via `os.environ.setdefault` so you
> don't need to export it manually for tests; the value in your environment
> (if any) takes priority.

### Run All Tests

```bash
cd backend
pytest
```

### Run Only Unit Tests

```bash
cd backend
pytest tests/unit/ -v
```

### Run Only Integration Tests

```bash
cd backend
pytest tests/integration/ -v
```

### Run with Coverage (optional)

```bash
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

### CI Notes

- No GPU required — all tests run on CPU with synthetic data.
- No external services — NIfTI/PNG test data is generated in-memory or in `tmp_path`.
- Integration tests redirect upload/cache directories to pytest's temporary
  directories so the real `backend/data/` folder is never written to during tests.

---

## Security Notes

- `image_id` is validated as a UUID (parsed with `uuid.UUID()` and re-serialised before any path operation) to prevent path-traversal attacks.
- No credentials or patient data are committed. The `data/` directory is git-ignored.
- Copy `.env.example` to `.env` and set a stable `JWT_SECRET` before production deployment.
- Frontend JWT persistence currently uses `localStorage` for simplicity. This is acceptable for local/dev use, but in production a cookie-based flow with `HttpOnly` + `Secure` + `SameSite` should be preferred to reduce XSS token exposure risk.

### Auth API quick call (registration)

`POST /api/auth/register`

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "email": "alice@example.com",
    "password": "StrongPass123!"
  }'
```

Success (`201`):

```json
{
  "id": 2,
  "username": "alice",
  "email": "alice@example.com",
  "role": "guest"
}
```

Duplicate username/email returns `409`.

### Auth API quick call (login)

`POST /api/auth/login` (expects form-encoded fields via `OAuth2PasswordRequestForm`)

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=alice&password=StrongPass123!"
```

Success (`200`):

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer"
}
```

---

## Deep Learning Integration

Use **Patchify 3D** to export `.npz` patches, annotate externally, then download the
auto-generated **Google Colab notebook** from the Deep Learning page. The notebook contains:

- Patch loader for `.npz`
- Simple 3-D CNN classifier (PyTorch)
- MONAI 3-D U-Net skeleton for segmentation
- Training loop with loss logging

No heavy computation runs on the web server.
