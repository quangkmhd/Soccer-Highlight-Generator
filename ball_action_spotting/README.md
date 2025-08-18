# Ball-Action Spotting

Detect football (soccer) actions and ball-related events in a single video file using pretrained PyTorch models. Using for key moment and start clips.

---

## 1.  Quick Start

```bash
# Run prediction
python3 ball_action_spotting/ball_action_predict.py \
    --video-path           path/to/match.mp4 \
    --action-model-path    weight/action/model.pth \
    --ball-action-model-path weight/ball/model.pth \
    --output-dir           results/ \
    --gpu-id               0            # use -1 for CPU
```

The script will create `results/<video_name>_ball_action.json` containing a chronologically-sorted list of detected events:

---

## 2.  Arguments

| Flag                         | Description                                           | Default      |
| ---------------------------- | ----------------------------------------------------- | ------------ |
| `--video-path`             | Input video (any codec readable by FFmpeg)            | *required* |
| `--action-model-path`      | Path to**action** model weights (`.pth`)      | *required* |
| `--ball-action-model-path` | Path to**ball-action** model weights (`.pth`) | *required* |
| `--output-dir`             | Folder for prediction JSON                            | *required* |
| `--gpu-id`                 | CUDA device index,`-1` for CPU                      | `0`        |
| `--batch-size`             | Frames per inference batch                            | `3`        |
| `--target-fps`             | Resampled FPS for inference                           | `25.0`     |

---

## 3.  Notes

* Weights are **not** included. Place your models in `weight/action/` and `weight/ball/` or anywhere else and pass their paths.
* The script automatically resamples frames to the target FPS and handles GPU memory efficiently.
* For custom post-processing thresholds, edit `src/action/constants.py` and `src/ball_action/constants.py`.
* Evaluation / training pipelines live under `scripts/` and `src/`; see code comments for details.
