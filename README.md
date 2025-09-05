# Soccer Action – End-to-End Highlight Clips Generation

This repository detects football (soccer) events in a single match video and automatically creates **ranked highlights**.

Modules:

1. **ball_action_spotting/** – classifies on-ball actions (pass, shot…) & general actions.
2. **CALF_segmentation/** – segments camera views (main, close-up, goal-line…).
3. **rules/** – merges predictions, applies rule-based windows & scoring to pick best moments.
4. **create_clips/** – cuts and (optionally) merges highlight clips.
5. **inference/** – orchestration & parallel processing helpers.
6. **main.py** – one-liner CLI entry-point.

---

## Quick Start

```bash
# 1.  Install Python 3.10.12
python3 -m venv sh_venv && source sh_venv/bin/activate   
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
pip install -rrequirements.txt  

# 2.  Download / place model weights
weight/
  ├─ action/             model-019-0.797827.pth
  ├─ ball/               model-006-0.864002.pth
  └─ camera/             model_last.pth.tar

# 3.  Run full pipeline on one video
python3 main.py path/to/match.mp4 \
        --config inference/inference_config.yaml
```

Outputs:

* `pipeline_output/<video>/` – raw model predictions and merged JSON.
* `highlight_results/<video>_highlights.json` – ranked highlight list.
* `clips/` – extracted video snippets (if `create_clips` enabled in config).

---

## Web Demo (`app_v2`)

The repository also ships with a small web application located in `app_v2/` that wraps the full pipeline behind a FastAPI REST API and a Gradio-powered front-end.

```bash
# 1.  Install extra dependencies
pip install -r app_v2/requirements.txt

# 2.  Start the API server (port 8000)
python3 -m uvicorn app_v2.main_api:app --reload --port 8000
#     or simply
python3 app_v2/main_api.py

# 3.  Launch the Gradio UI in a separate terminal (port 7860)
python3 app_v2/gradio_app.py
```

The Gradio interface will print a local URL (default: http://localhost:7860). Open it in your browser, upload a match video and wait while the server processes the file. Once finished, the ranked highlight list together with preview clips will be displayed.

The underlying REST endpoints are documented at `http://localhost:8000/docs` (FastAPI Swagger UI).

---

## Project Structure

| Path                      | Purpose                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| `ball_action_spotting/` | PyTorch models for on-ball & general actions. See its `README.md`. |
| `CALF_segmentation/`    | Camera view segmentation (ResNet + CALF). See its `README.md`.     |
| `rules/`                | Rule engine & scoring; tune behaviour in `rules/config.yaml`.      |
| `create_clips/`         | FFmpeg helpers to cut & merge highlight clips.                       |
| `inference/`            | Parallel pipeline (`parallel_inference.py`) & YAML config.         |
| `main.py`               | Thin CLI wrapper calling parallel inference.                         |

---

## Configuration

Edit `inference/inference_config.yaml` to point to your weight files and tweak GPU, batch sizes, FPS, etc. Each sub-section (`ball_action_params`, `camera_params`, `rules`) matches arguments of underlying scripts.

`rules/config.yaml` controls time windows, confidence thresholds, scoring weights. See `rules/README.md` for details.

---

## Development Notes

* Codebase uses **PyTorch** + **CUDA**  and **FFmpeg** for video I/O.
* Multiprocessing start method is set to `spawn` in `main.py` for Windows compatibility.
* Large intermediate npy/JSON files are stored under each video’s folder to avoid recomputation.

---

For module-specific instructions read:

* `ball_action_spotting/README.md`
* `CALF_segmentation/README.md`
* `rules/README.md`
