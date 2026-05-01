# MedVision 🔬

> **v1.0 – Modular Medical-Imaging Intelligence Platform**  
> FastAPI backend · React + Vite frontend · OpenCV / scikit-learn / NumPy

---

## Architecture

```
MedVision/
├── backend/
│   ├── main.py                    # FastAPI app factory – CORS, static files, router registration
│   ├── requirements.txt           # Python dependencies
│   ├── data/
│   │   ├── uploads/               # Uploaded originals (auto-created, git-ignored)
│   │   └── cache/                 # Per-image JSON metadata cache (auto-created)
│   │
│   ├── core/
│   │   └── loader.py              # Unified loader: DICOM · NIfTI · PNG/JPG → NumPy float32
│   │                              #   + slice helpers (axial/coronal/sagittal)
│   │                              #   + base64-PNG encoder
│   │
│   ├── processing/
│   │   ├── filters.py             # Gaussian blur · Sobel edge detection
│   │   ├── histogram.py           # Histogram · CDF · CDF-based threshold
│   │   └── morphology.py          # Erosion · Dilation · Opening · Closing
│   │                              #   + Connected components · Bounding boxes · Centre of mass
│   │
│   ├── features/
│   │   └── extractor.py           # Statistical feature vector:
│   │                              #   mean, std, min/max, skewness, kurtosis,
│   │                              #   percentiles, entropy, shape descriptors
│   │
│   ├── ml/
│   │   ├── clustering.py          # KMeans intensity segmentation → colour image + stats
│   │   └── reduction.py           # PCA 2-D projection of pixel feature space
│   │
│   ├── analysis/
│   │   ├── statistics.py          # Normalisation (min-max, z-score, histogram equalisation)
│   │   │                          #   + array comparison (Pearson r, MAD)
│   │   └── report.py              # Structured JSON report assembly
│   │
│   ├── api/
│   │   ├── __init__.py            # Shared disk-store helpers (upload dir, cache CRUD)
│   │   ├── upload.py              # POST /api/upload
│   │   ├── preview.py             # GET  /api/preview/{image_id}
│   │   ├── process.py             # POST /api/process
│   │   ├── features.py            # POST /api/features
│   │   ├── cluster.py             # POST /api/cluster
│   │   └── report.py              # GET  /api/report/{image_id}
│   │
│   └── labs/
│       └── core_imaging/          # Legacy v0.1 endpoints (kept for backward compatibility)
│           ├── routes.py
│           ├── processing.py
│           └── features.py
│
└── frontend/                      # React + Vite SPA
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx               # ReactDOM root
        ├── App.jsx                # Shell: header, tab nav, tab routing
        ├── api/
        │   └── client.js          # Thin fetch() wrapper (base URL, error handling)
        ├── styles/
        │   └── index.css          # Dark theme, CSS variables, all component styles
        └── components/tabs/
            ├── UploadViewer.jsx   # Tab 1 – Drag-drop upload + axial/sagittal/coronal viewer
            ├── ProcessingLab.jsx  # Tab 2 – Filter selector, before/after, histograms
            ├── FeatureExplorer.jsx# Tab 3 – Feature cards + intensity distribution chart
            ├── MLLab.jsx          # Tab 4 – KMeans segmentation + PCA scatter plot
            └── AnalysisReport.jsx # Tab 5 – Structured report + JSON download
```

---

## Request / response flow

```
Browser (React)
    │
    │  multipart upload / JSON body
    ▼
FastAPI (uvicorn)
    │
    ├─ POST /api/upload   → core/loader.py → data/uploads/ + data/cache/
    ├─ GET  /api/preview  → core/loader.py → get_slice_2d → base64 PNG
    ├─ POST /api/process  → processing/{filters,histogram,morphology}
    ├─ POST /api/features → features/extractor.py → cache
    ├─ POST /api/cluster  → ml/{clustering,reduction} → cache
    └─ GET  /api/report   → analysis/report.py ← cache
```

---

## Quick start

### 1 – Backend

**Prerequisites:** Python 3.10+

```bash
cd backend

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API base: **http://127.0.0.1:8000**
- Swagger UI: **http://127.0.0.1:8000/docs**

### 2 – Frontend

**Prerequisites:** Node 18+

```bash
cd frontend
npm install
npm run dev
# Visit http://localhost:5173
```

> Make sure the backend is already running on port 8000.

---

## API reference

All endpoints are prefixed with `/api`.

---

### `POST /api/upload`

Upload a medical image (PNG, JPG, DICOM `.dcm`, NIfTI `.nii` / `.nii.gz`).

**Request** – `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Medical image file |

**Response** `200`

```json
{
  "image_id":      "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename":      "brain.dcm",
  "file_type":     "dicom",
  "modality":      "CT",
  "shape":         [512, 512],
  "ndim":          2,
  "is_3d":         false,
  "intensity_min": -1024.0,
  "intensity_max": 3071.0,
  "spacing":       [0.48, 0.48, 1.0],
  "size_bytes":    524288,
  "extra_meta":    { "patient_id": "…", "study_date": "…" }
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@brain.dcm"
```

---

### `GET /api/preview/{image_id}`

Return base64 PNG slices for axial, sagittal, and coronal orientations.

**Query params** (all optional, default = middle slice)

| Param | Type | Description |
|-------|------|-------------|
| `axial_idx` | int | Axial slice index |
| `coronal_idx` | int | Coronal slice index |
| `sagittal_idx` | int | Sagittal slice index |

**Response** `200`

```json
{
  "image_id": "3fa85f64…",
  "is_3d": false,
  "shape": [512, 512],
  "slice_indices": { "axial": 0, "coronal": 0, "sagittal": 0 },
  "axial":    "data:image/png;base64,…",
  "coronal":  "data:image/png;base64,…",
  "sagittal": "data:image/png;base64,…"
}
```

```bash
curl "http://127.0.0.1:8000/api/preview/3fa85f64-5717-4562-b3fc-2c963f66afa6"
```

---

### `POST /api/process`

Apply a processing operation and return the result image + histograms.

**Request body** (JSON)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_id` | string | — | ID from `/upload` |
| `processing_type` | string | — | See table below |
| `sigma` | float | 2.0 | Gaussian sigma (0.1–20) |
| `kernel_size` | int | 5 | Morphology kernel (1–31) |
| `percentile` | float | 95.0 | CDF threshold percentile |
| `threshold` | float | 128.0 | Binary threshold (0–255) |

**`processing_type` values**

| Value | Description |
|-------|-------------|
| `gaussian` | Gaussian blur |
| `sobel` | Sobel edge detection |
| `cdf_threshold` | CDF-based binarisation |
| `erosion` | Morphological erosion |
| `dilation` | Morphological dilation |
| `opening` | Morphological opening |
| `closing` | Morphological closing |
| `connected_components` | Label and colour-code connected regions |
| `bounding_boxes` | Draw bounding boxes around regions |

**Response** `200`

```json
{
  "image_id": "3fa85f64…",
  "processing_type": "sobel",
  "result_image": "data:image/png;base64,…",
  "histogram_before": { "bins": […], "counts": […] },
  "histogram_after":  { "bins": […], "counts": […] },
  "extra_meta": {}
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"image_id":"3fa85f64-…","processing_type":"gaussian","sigma":3.0}'
```

---

### `POST /api/features`

Extract the statistical feature vector for an image.

**Request body** (JSON)

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | string | ID from `/upload` |

**Response** `200`

```json
{
  "image_id":        "3fa85f64…",
  "mean":            127.43,
  "std_dev":         52.18,
  "intensity_min":   0.0,
  "intensity_max":   255.0,
  "skewness":        0.12,
  "kurtosis":       -0.84,
  "percentile_10":   42.0,
  "percentile_25":   89.0,
  "percentile_50":  128.0,
  "percentile_75":  172.0,
  "percentile_90":  214.0,
  "entropy":         7.21,
  "nonzero_fraction": 0.98,
  "histogram": { "bins": […], "counts": […] },
  "shape_descriptors": {
    "foreground_pixels": 198432,
    "total_pixels":      262144,
    "foreground_coverage": 0.757,
    "effective_radius_px": 251.3
  }
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/features \
  -H "Content-Type: application/json" \
  -d '{"image_id":"3fa85f64-…"}'
```

---

### `POST /api/cluster`

Run KMeans segmentation and PCA dimensionality reduction.

**Request body** (JSON)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_id` | string | — | ID from `/upload` |
| `k` | int | 4 | Number of KMeans clusters (2–16) |
| `n_samples` | int | 5000 | PCA pixel sample size (100–20 000) |

**Response** `200`

```json
{
  "image_id": "3fa85f64…",
  "k": 4,
  "segmented_image": "data:image/png;base64,…",
  "centers": [12.4, 78.2, 148.6, 221.0],
  "cluster_counts": [14823, 63210, 88741, 95370],
  "pca": {
    "points": [{ "x": 1.23, "y": -0.45, "cluster": 2 }, "…"],
    "explained_variance": [0.612, 0.271],
    "n_samples_used": 5000
  }
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/cluster \
  -H "Content-Type: application/json" \
  -d '{"image_id":"3fa85f64-…","k":5}'
```

---

### `GET /api/report/{image_id}`

Aggregate all cached analysis results into a structured report.

**Response** `200`

```json
{
  "report_id":    "rpt-3fa85f64",
  "image_id":     "3fa85f64…",
  "generated_at": "2026-05-01T03:00:00+00:00",
  "version":      "1.0",
  "image_info": {
    "filename":    "brain.dcm",
    "file_type":   "dicom",
    "modality":    "CT",
    "shape":       [512, 512],
    "is_3d":       false,
    "spacing_mm":  [0.48, 0.48, 1.0],
    "intensity_range": { "min": -1024.0, "max": 3071.0 }
  },
  "processing_pipeline": [ "…" ],
  "feature_summary": {
    "mean_intensity":    127.4,
    "std_deviation":     52.2,
    "entropy_bits":       7.2,
    "foreground_coverage": 0.76
  },
  "cluster_summary": {
    "k": 4,
    "cluster_centers":  [12.4, 78.2, 148.6, 221.0],
    "cluster_counts":   [14823, 63210, 88741, 95370],
    "dominant_cluster": 3
  },
  "interpretation": [
    "High entropy — image is rich in detail.",
    "KMeans segmented the image into 4 clusters."
  ]
}
```

```bash
curl http://127.0.0.1:8000/api/report/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI + Uvicorn |
| Image I/O | OpenCV (`opencv-python-headless`), pydicom, nibabel |
| Numerics | NumPy, SciPy |
| Machine learning | scikit-learn (KMeans, PCA, StandardScaler) |
| Frontend | React 18 + Vite 5 |
| Charts | Recharts |

---

## Legacy endpoints (v0.1)

The original Core Imaging Lab endpoints are still active under the prefix `/api/core-imaging` for backward compatibility:

- `POST /api/core-imaging/upload`
- `POST /api/core-imaging/process`
- `GET  /api/core-imaging/features/{image_id}`
