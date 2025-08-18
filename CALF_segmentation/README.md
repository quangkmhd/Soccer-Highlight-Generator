# CALF Segmentation – Camera View Classification

Identify camera view segments (e.g., main, close-up, goal-line…) in football videos and output timestamps for each change. Using for end clips.

---

## 1.  Quick Start

```bash
# Run prediction
python3 CALF_segmentation/camera_predict.py \
    --video_path   path/to/match.mp4 \
    --model_path   weight/camera/model.pth.tar \
    --pca_file     CALF_segmentation/Features/pca_512_PT.pkl \
    --scaler_file  CALF_segmentation/Features/average_512_PT.pkl \
    --confidence_threshold  0.8 \
    --output_dir   results/ \
    --backend      PT  \
    --GPU          0       # -1 for CPU
```

The script creates `results/<video>_camera.json` containing a list of detected view changes with timestamps and confidences.

---

## 2.  Arguments

| Flag                              | Description                                      | Default      |
| --------------------------------- | ------------------------------------------------ | ------------ |
| `--video_path`                  | Input video file                                 | *required* |
| `--model_path`                  | Path to CALF segmentation weights (`.pth.tar`) | *required* |
| `--output_dir`                  | Folder for JSON output                           | `output/`  |
| `--backend`                     | Feature extractor backend                       | `PT`       |
| `--fps`                         | Frame rate for feature extraction                | `2.0`      |
| `--batch_size_feat`             | Batch size when extracting features              | `128`      |
| `--GPU`                         | CUDA device index,`-1` for CPU                 | `0`        |
| `--confidence_threshold`        | Min conf. to emit an event                       | `0.2`      |
| `--pca_file`, `--scaler_file` | Paths to 512-D PCA & scaler pickle files         | pre-set      |

---

## 3.  Notes

* Model weights and PCA/scaler files are **not included**. Place them under `weight/` or supply custom paths.
* Feature extraction uses ResNet layer2 (512-D) and optional PCA for speed.
* Class labels are defined inside `camera_predict.py` (see `LABELS`). Adjust them if you train new classes.
* Training & evaluation code sits in `Features/` and `src/` sub-folders.
