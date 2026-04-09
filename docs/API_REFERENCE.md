# Soccer Highlight Generator API & Code Reference

This document provides a detailed reference for developers wishing to interact with the Soccer Highlight Generator via CLI, Python modules, or the REST API.

## 1. CLI Entry Point (`main.py`)

The primary script for executing the full end-to-end pipeline locally.

```bash
python3 main.py <video_path> [OPTIONS]
```

**Arguments:**
- `video_path` (str, required): Absolute or relative path to the input soccer match video (`.mp4`, `.avi`, `.mkv`).

**Options:**
- `--top-k` (int, optional): The number of highlight clips to generate. Default: `5`. The system will take the 5 highest-scoring moments.
- `--merge-clips` (flag): If passed, the system will not only generate individual clip files but will concatenate them chronologically into a single `final_highlights.mp4` video.
- `--skip-extraction` (flag): Bypasses the heavy ResNet feature extraction phase. *Only use this if you have previously processed this exact video and the `.npy` feature files already exist in the cache directory.*
- `--output-dir` (str, optional): Base directory for all generated artifacts (JSONs, individual clips, merged reel). Default: `./pipeline_output/`.

## 2. REST API Reference (FastAPI)

When running the FastAPI server (`uvicorn server:app --host 0.0.0.0 --port 8000`), the following endpoints are exposed.

### `POST /process`

Initiates the highlight generation pipeline for a given video.

**Request Body (JSON):**
```json
{
  "video_path": "/workspace/data/full_match_2023.mp4",
  "clip_count": 10,
  "prioritize_goals": true
}
```
- `video_path`: Must be a path accessible to the server/container.
- `clip_count`: Equivalent to `--top-k`.
- `prioritize_goals`: Boolean. If true, the rule engine applies a massive multiplier to any sequence containing a "goal" action, practically guaranteeing its inclusion in the reel.

**Response (JSON):**
```json
{
  "status": "success",
  "job_id": "job_98765abc",
  "processing_time_seconds": 452.3,
  "artifacts": {
    "metadata": "/workspace/pipeline_output/match/highlights.json",
    "clips": [
      "/workspace/pipeline_output/match/clips/rank_1_goal.mp4",
      "/workspace/pipeline_output/match/clips/rank_2_shot.mp4"
    ],
    "merged_video": "/workspace/pipeline_output/match/final_highlights.mp4"
  }
}
```

## 3. Internal Python Modules

For developers looking to integrate specific pieces of the pipeline into larger systems.

### `inference.feature_extractor`

```python
from inference.feature_extractor import ResNetExtractor

extractor = ResNetExtractor(batch_size=32, device="cuda")
feature_tensor = extractor.process_video("match.mp4")
# Returns a numpy array of shape (num_frames, 2048)
```

### `rules.scoring_engine`

The heart of the highlight selection logic.

```python
from rules.scoring_engine import rank_highlights

# action_list: List of dicts [{"type": "shot", "time": 450.2, "confidence": 0.95}, ...]
# camera_timeline: List of strings ["main", "main", "close-up", ...] mapped to seconds
top_clips = rank_highlights(action_list, camera_timeline, top_k=5)

for clip in top_clips:
    print(f"Clip from {clip['start_time']} to {clip['end_time']} - Score: {clip['score']}")
```

### `create_clips.trimmer`

A wrapper around FFmpeg for exact video slicing.

```python
from create_clips.trimmer import extract_clip

# Extracts a segment from exactly 12m30s to 12m45s without re-encoding video (fast)
extract_clip(
    input_video="match.mp4",
    start_time=750.0,
    end_time=765.0,
    output_path="./clips/highlight_1.mp4"
)
```
