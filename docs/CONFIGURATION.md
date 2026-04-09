# Configuration Guide for Soccer Highlight Generator

This document details how to configure the deep learning inference parameters, rule engine weights, and Docker environments for the Soccer Highlight Generator.

## 1. Deep Learning Inference Config (`inference/config.yaml`)

This YAML file dictates the behavior of the heavy neural networks. Adjusting these can resolve hardware limitations (like Out Of Memory errors) or improve accuracy at the cost of speed.

### ResNet Feature Extraction
```yaml
feature_extraction:
  batch_size: 64
  num_workers: 4
  fps: 2.0
```
- **`batch_size`**: (Default `64`). Decrease to `32` or `16` if your GPU has less than 8GB of VRAM. A lower batch size slows down the initial video parsing.
- **`fps`**: (Default `2.0`). The rate at which frames are sampled. 2 FPS is generally sufficient to capture soccer actions. Increasing to `5.0` will yield slightly more accurate timestamps but will increase memory usage and processing time by 2.5x.

### Action Spotting Network
```yaml
action_spotting:
  confidence_threshold: 0.65
  nms_window_seconds: 4.0
```
- **`confidence_threshold`**: Actions predicted with a probability below this number are discarded. Lower this if the system is missing obvious shots/passes.
- **`nms_window_seconds`**: Non-Maximum Suppression window. If the model detects a "shot" at `T=45.0` and another "shot" at `T=46.5`, it merges them into a single event if they fall within this 4.0-second window, preventing duplicate highlight clips of the exact same event.

## 2. Rule Engine Configuration (`rules/weights.json`)

This is the most crucial configuration file for editors. It defines the point values assigned to different events. By tweaking this, you change the "personality" of the highlight reel.

```json
{
  "action_scores": {
    "goal": 1000,
    "shot": 100,
    "cross": 30,
    "pass": 10,
    "tackle": 20,
    "foul": -50
  },
  "camera_multipliers": {
    "main_camera": 1.0,
    "replay": 0.2,
    "close_up": 0.1
  },
  "sequence_bonus": 1.5
}
```

### Explanation of Rules:
- **`action_scores`**: Base points awarded when an action is detected. Giving "goal" a massive score ensures it overrides any other sequence. Giving "foul" a negative score means fouls actively reduce a clip's chance of making the reel.
- **`camera_multipliers`**: Applied to the total score of the clip. A clip occurring entirely during a `close_up` has its score reduced to 10% of its original value, virtually guaranteeing it won't be selected over wide-angle, live-play footage.
- **`sequence_bonus`**: If multiple positive actions occur within a short window (e.g., `pass` -> `cross` -> `shot`), the combined score is multiplied by this factor to reward fluid, attacking plays.

## 3. Docker Environment Configuration (`docker-compose.yml`)

The system relies on Docker to manage FFmpeg, OpenCV, and PyTorch CUDA dependencies.

### GPU Passthrough
To utilize GPU acceleration, your host machine must have the NVIDIA Container Toolkit installed, and the `docker-compose.yml` must include:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Port Mappings
The compose file exposes two critical ports to the host:
- `8000:8000`: The FastAPI backend for REST requests.
- `7860:7860`: The Gradio Web UI for user interaction.
If these ports conflict with existing services on your host, modify the left side of the mapping (e.g., `8080:8000`).

### Persistent Volume Caching
```yaml
    volumes:
      - ./pipeline_output:/workspace/pipeline_output
      - ./data:/workspace/data
```
This configuration is critical. Because extracting ResNet features takes a long time, the system caches the `.npy` feature files in `pipeline_output/cache/`. By mounting this volume to the host, you preserve the cache even if the container is destroyed. If you rerun the pipeline on the same video, it will skip extraction and finish in seconds.
