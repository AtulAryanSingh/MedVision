# Deployment

## 1) Configure environment

```bash
cp .env.example .env
```

Set at least:
- `JWT_SECRET` (required for stable auth tokens)
- `ALLOWED_ORIGINS` (must include your frontend origin)
- `UPLOAD_DIR`, `CACHE_DIR`, `USERS_DB_PATH` (storage paths)

## 2) Run with Docker Compose

```bash
docker compose up --build
```

Services:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

Backend process (inside container):

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 main:app
```
