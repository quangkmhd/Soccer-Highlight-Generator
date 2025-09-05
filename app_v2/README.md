# App v2 – Soccer Highlight API & UI

AI-powered service to upload a match video, run end-to-end detection (ball-actions + camera segmentation + rules), and preview/export highlight clips via REST API and Gradio UI.

---

## 1. Quick Start

```bash
# (Optional) Create venv & install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r app_v2/requirements.txt

# A) Start FastAPI server (REST)
uvicorn app_v2.main_api:app --host 0.0.0.0 --port 8000 --reload
# → Docs: http://localhost:8000/docs

# B) Start Gradio UI (Web)
python3 app_v2/gradio_app.py
# → Default: http://localhost:7860
```

The service reads settings from `app_v2/config.yaml` and creates the upload directory automatically.

---

## 2. API Overview (base: `/api/v2`)

- `POST /upload/file` – upload a video file (multipart). Returns `video_id`.
- `POST /upload/path` – register a local file path. Form field: `input_path`. Returns `video_id`.
- `GET  /video/{video_id}` – get metadata (duration, fps, resolution, size, format).
- `POST /process/{video_id}` – start AI pipeline for that video. Returns `job_id`.
- `GET  /status/{job_id}` – polling status: `queued|processing|completed|failed`.
- `GET  /queue/status` – queue length, current job, etc.
- `GET  /results/{job_id}` – list detected highlight clips (timestamps, labels, scores).
- `POST /select_clips` – save user-selected clips for export/download.
- `GET  /download/metadata?job_id=...&mode=selected&format=srt|xml` – export SRT/XML.
- `GET  /clips/{job_id}/{clip_filename}.mp4` – stream clip preview.

See implementations in `app_v2/api/routes.py`.

---

## 3. Typical Flow (cURL)

```bash
# 1) Register a local video (or POST /upload/file with multipart)
curl -X POST http://localhost:8000/api/v2/upload/path \
     -F "input_path=/absolute/path/to/match.mp4"
# → { "video_id": "..." }

# 2) Start processing
curl -X POST http://localhost:8000/api/v2/process/<video_id>
# → { "job_id": "...", "status": "queued" }

# 3) Poll status
curl http://localhost:8000/api/v2/status/<job_id>
# → { "status": "processing" | "completed" | ... }

# 4) Get results when completed
curl http://localhost:8000/api/v2/results/<job_id>
# → { "clips": [ {"start": 745.0, "end": 773.0, "label": "Goal", "score": 92.3}, ... ] }

# 5) Export metadata (SRT/XML)
curl -L "http://localhost:8000/api/v2/download/metadata?job_id=<job_id>&mode=selected&format=srt" -o highlights.srt
```

---

## 4. Configuration (`app_v2/config.yaml`)

- `api.base_url` – base URL used by the UI client.
- `video.upload_dir` – local upload folder (auto-created).
- `video.supported_formats` – allowed extensions (e.g., `[mp4, mov]`).
- `gradio.{server_name, server_port, share, debug}` – Gradio server options.

Utilities in `app_v2/api/config.py` ensure directories exist and expose helpers like `get_api_base_url()` and `get_gradio_config()`.

---

## 5. Notes

- Model weights and traditional pipelines live in sibling folders (`ball_action_spotting/`, `CALF_segmentation/`, `rules/`). This app orchestrates them through async jobs and a queue in `app_v2/api/job_manager.py`.
- Only one video is processed at a time; queue endpoints provide visibility.
- When processing completes, results include previewable MP4 clips and exportable SRT/XML.

---

## 6. Project Layout

- `app_v2/main_api.py` – FastAPI entrypoint and lifespan.
- `app_v2/gradio_app.py` – Gradio web UI entrypoint.
- `app_v2/api/` – routes, job queue, services, models, config.
- `app_v2/ui/` – UI client and interface factory.
- `app_v2/uploads/` – uploaded/registered videos.
