# MedVision 🔬

> **v2 – Multi-page Modular Medical-Imaging Workbench**  
> FastAPI backend · React + Vite frontend · White-background clinical UI

A clinician-friendly platform for exploring medical images (DICOM, NIfTI, PNG/JPEG).
Every tool is **independent** — run Gaussian blur, Sobel edges, morphology, segmentation,
patchify, or export in **any order, on demand**. No forced pipeline.

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

### 3 – Production build

```bash
cd frontend
npm run build          # outputs to frontend/dist/
npm run preview        # serve the production build locally
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
| `GET` | `/api/export/{id}/csv` | Download component metrics as CSV |

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

## Security Notes

- `image_id` is validated as a UUID (parsed with `uuid.UUID()` and re-serialised before any path operation) to prevent path-traversal attacks.
- No credentials or patient data are committed. The `data/` directory is git-ignored.

---

## Deep Learning Integration

Use **Patchify 3D** to export `.npz` patches, annotate externally, then download the
auto-generated **Google Colab notebook** from the Deep Learning page. The notebook contains:

- Patch loader for `.npz`
- Simple 3-D CNN classifier (PyTorch)
- MONAI 3-D U-Net skeleton for segmentation
- Training loop with loss logging

No heavy computation runs on the web server.
