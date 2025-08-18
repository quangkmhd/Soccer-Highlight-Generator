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
