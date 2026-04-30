# MedVision 🔬

> **v0.1 – Module 1: Core Imaging Lab**  
> A modular medical-imaging platform built with FastAPI + OpenCV.

---

## Project structure

```
MedVision/
├── backend/
│   ├── main.py                        # FastAPI app + CORS + router registration
│   ├── requirements.txt               # Python dependencies
│   ├── data/uploads/                  # Uploaded originals (auto-created)
│   └── labs/
│       ├── core_imaging/              # ✅ Active – Core Imaging Lab
│       │   ├── routes.py              #    HTTP endpoints (upload / process / features)
│       │   ├── processing.py          #    K-Means, Gaussian, Sobel implementations
│       │   └── features.py            #    Mean, std-dev, histogram extraction
│       ├── dicom_lab/                 # 🔲 Placeholder – DICOM Processing Lab
│       ├── classical_pipeline/        # 🔲 Placeholder – Classical Pipeline
│       ├── pattern_insight/           # 🔲 Placeholder – Pattern Insight Lab
│       └── ai_lab/                    # 🔲 Placeholder – AI / Deep Learning Lab
└── frontend/
    ├── index.html                     # Single-page UI
    └── style.css
```

---

## How to run the backend

### Prerequisites
- Python 3.10+

### Steps

```bash
cd backend

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be live at **http://127.0.0.1:8000**.  
Interactive docs (Swagger UI): **http://127.0.0.1:8000/docs**

---

## How to run the frontend

The frontend is a plain HTML file – no build step required.

**Option A – open directly in the browser**

```bash
open frontend/index.html          # macOS
xdg-open frontend/index.html      # Linux
start frontend/index.html         # Windows
```

**Option B – serve with a local HTTP server** (avoids any file:// quirks)

```bash
cd frontend
python -m http.server 5500
# then visit http://localhost:5500
```

> Make sure the backend is already running on port 8000 before using the UI.

---

## API reference

### Base URL
```
http://127.0.0.1:8000/api/core-imaging
```

---

### POST `/upload`
Upload a PNG or JPG image.

**Request** – `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | PNG or JPG image |

**Response** – `200 OK`

```json
{
  "image_id":    "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename":    "brain_scan.png",
  "content_type":"image/png",
  "size_bytes":  204800,
  "width":       512,
  "height":      512,
  "channels":    3
}
```

**curl example**

```bash
curl -X POST http://127.0.0.1:8000/api/core-imaging/upload \
  -F "file=@/path/to/image.png"
```

---

### POST `/process`
Apply a processing algorithm and receive the result as PNG bytes.

**Request** – `multipart/form-data`

| Field | Type | Allowed values |
|-------|------|----------------|
| `image_id` | string | ID returned by `/upload` |
| `processing_type` | string | `kmeans` · `gaussian` · `sobel` |

**Response** – `200 OK` with `Content-Type: image/png`

**curl example**

```bash
# Save processed image to disk
curl -X POST http://127.0.0.1:8000/api/core-imaging/process \
  -F "image_id=3fa85f64-5717-4562-b3fc-2c963f66afa6" \
  -F "processing_type=sobel" \
  --output processed.png
```

---

### GET `/features/{image_id}`
Extract mean, standard deviation, and intensity histogram.

**Response** – `200 OK`

```json
{
  "image_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "mean":     127.43,
  "std_dev":  52.18,
  "histogram": {
    "bins":   [0.5, 1.5, "…", 254.5],
    "counts": [120, 85,  "…", 43]
  }
}
```

**curl example**

```bash
curl http://127.0.0.1:8000/api/core-imaging/features/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

---

## Adding a new lab (future)

1. Create `backend/labs/<lab_name>/` with `__init__.py` and `routes.py`.
2. Define an `APIRouter` in `routes.py`.
3. Uncomment (or add) the corresponding import + `app.include_router(…)` block in `backend/main.py`.

Nothing else in the existing code needs to change.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI |
| Image processing | OpenCV (`opencv-python-headless`), NumPy, SciPy |
| ASGI server | Uvicorn |
| Frontend | Plain HTML + CSS + vanilla JS |
