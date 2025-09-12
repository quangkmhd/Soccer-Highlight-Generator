# Soccer Action – End-to-End Highlight Clips Generation

This repository detects football (soccer) events in a single match video and automatically creates **ranked highlights**.

Modules:

1. **ball_action_spotting/** – classifies on-ball actions (pass, shot…) & general actions.
2. **CALF_segmentation/** – segments camera views (main, close-up, goal-line…).
3. **rules/** – merges predictions, applies rule-based windows & scoring to pick best moments.
4. **create_clips/** – cuts and (optionally) merges highlight clips.
5. **inference/** – orchestration & parallel processing helpers.
6. **app_v2/** – REST API server & Gradio web interface for video upload and processing v2.
7. **main.py** – one-liner CLI entry-point.

---

## Quick Start

```bash
# 1.  Install Python 3.10.12
python3 -m venv sh_venv && source sh_venv/bin/activate   
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
pip install -rrequirements.txt  

# 2.  Download / place model weights
weight/
  ├─ action/             # Action spotting model weights
  ├─ ball/               # Ball detection model weights  
  ├─ camera/             # Camera segmentation model weights
  └─ resnet/             # ResNet feature extraction weights

# 3.  Run full pipeline on one video
python3 main.py path/to/match.mp4 
```

Outputs:

* `pipeline_output/<video>/` – raw model predictions and merged JSON.
* `highlight_results/<video>_highlights.json` – ranked highlight list.
* `clips/` – extracted video snippets (if `create_clips` enabled in config).

---

## Docker (GPU) Quick Start

> Requires: Docker ≥ 20.10, NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
>
> This will build the image, pass through all GPUs, and launch both the REST API (8000) and Gradio UI (7860).

```bash
# From repository root
#  Build image + run container (foreground)
docker compose up --build
```



## Web Application (`app_v2`)

The repository includes a comprehensive web application with REST API and modern UI for video processing and highlight generation.

### Features
- **FastAPI REST API** with async job processing and queue management
- **Gradio Web Interface** for easy video upload and result visualization
- **Video Management** supporting multiple formats (mp4, avi, mov, mkv, etc.)
- **Real-time Processing Status** with job queue monitoring
- **Clip Preview & Export** with SRT/XML subtitle generation
- **Configurable Settings** via YAML configuration

### Quick Start

```bash
# 1. Install dependencies
pip install -r app_v2/requirements.txt

# 2. Start API server (port 8000)
python3 app_v2/main_api.py
# or: uvicorn app_v2.main_api:app --host 0.0.0.0 --port 8000

# 3. Launch Gradio UI (port 7860)
python3 app_v2/gradio_app.py
```

**Access Points:**
- Gradio UI: [http://localhost:7860](http://localhost:7860)
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- API Base: `http://localhost:8000/api/v2`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload/file` | Upload video file (multipart) |
| `POST` | `/upload/path` | Register local file path |
| `GET` | `/video/{video_id}` | Get video metadata |
| `POST` | `/process/{video_id}` | Start AI processing |
| `GET` | `/status/{job_id}` | Check processing status |
| `GET` | `/results/{job_id}` | Get highlight results |
| `GET` | `/clips/{job_id}/{filename}` | Stream video clips |
| `GET` | `/download/metadata` | Export SRT/XML subtitles |

### Configuration

Edit `app_v2/config.yaml` to customize:
- Server ports and host settings
- Video upload limits and supported formats
- Processing queue configuration
- Storage paths and metadata handling

---

## Project Structure

| Path | Purpose |
|------|----------|
| `ball_action_spotting/` | PyTorch models for on-ball & general actions. See its `README.md`. |
| `CALF_segmentation/` | Camera view segmentation (ResNet + CALF). See its `README.md`. |
| `rules/` | Rule engine & scoring; tune behaviour in `rules/config.yaml`. |
| `create_clips/` | FFmpeg helpers to cut & merge highlight clips. |
| `inference/` | Parallel pipeline (`parallel_inference.py`) & YAML config. |
| `app_v2/` | **Web application with REST API & Gradio UI** |
| `├── api/` | FastAPI routes, job management, database, services |
| `├── ui/` | Gradio interface, API client, event handlers |
| `├── config.yaml` | Application configuration (ports, limits, paths) |
| `├── main_api.py` | FastAPI server entrypoint |
| `├── gradio_app.py` | Gradio UI entrypoint |
| `weight/` | **Model weights directory** |
| `├── action/` | Action spotting model files |
| `├── ball/` | Ball detection model files |
| `├── camera/` | Camera segmentation model files |
| `├── resnet/` | ResNet feature extraction weights |
| `main.py` | CLI entry-point for direct pipeline execution |

---

## Configuration

### Pipeline Configuration
Edit `inference/inference_config.yaml` to point to your weight files and tweak GPU, batch sizes, FPS, etc. Each sub-section (`ball_action_params`, `camera_params`, `rules`) matches arguments of underlying scripts.

`rules/config.yaml` controls time windows, confidence thresholds, scoring weights. See `rules/README.md` for details.

### Web Application Configuration
Edit `app_v2/config.yaml` to customize:
- **Server Settings**: Host, ports, debug mode
- **Video Processing**: Supported formats, file size limits, upload directory
- **API Configuration**: Base URLs, concurrent job limits
- **Gradio Settings**: UI server options, sharing, debug mode

---

## Development Notes

* Codebase uses **PyTorch** + **CUDA**  and **FFmpeg** for video I/O.
* Multiprocessing start method is set to `spawn` in `main.py` for Windows compatibility.
* Large intermediate npy/JSON files are stored under each video’s folder to avoid recomputation.

---

## Module Documentation

For detailed module-specific instructions:

* `ball_action_spotting/README.md` - Action detection models and training
* `CALF_segmentation/README.md` - Camera view segmentation
* `rules/README.md` - Rule engine and scoring system
* `app_v2/README.md` - Web application API and UI details
