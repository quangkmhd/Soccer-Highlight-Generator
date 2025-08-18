# Rules – Highlight Extraction Pipeline

Post-process model predictions to produce a ranked list of football highlights.

This module

1. merges ball-action + camera predictions into one JSON
2. applies rule-based windows & scoring (see `config.yaml`)
3. outputs the top events for clipping.

---

## 1.  Quick Start

```bash
# 1 ) Combine predictions from models
python3 rules/combine_predictions.py \
    path/to/<video>_ball_action.json \
    path/to/<video>_camera.json \
    pipeline_output/         # output dir
# → pipeline_output/<video>_predicted.json

# 2 ) Run rule engine to get highlights
python3 rules/rules_main.py \
    --predictions-json-path pipeline_output/<video>_predicted.json \
    --output-dir            highlight_results/ \
    --config                rules/config.yaml
# → highlight_results/<video>_highlights.json
```

---

## 2.  Key Scripts

* **`combine_predictions.py`** – merges two prediction JSONs into unified format.
* **`engine.py`** – core logic: groups events, applies windows, builds highlight candidates.
* **`rank_score.py`** – assigns scores & selects best segments.
* **`rules_main.py`** – CLI wrapper around engine + ranking.
* **`config.yaml`** – tune time windows, confidence thresholds, scoring weights.

---

## 3.  Config Overview (`config.yaml`)

* **grouping.time_threshold** – max gap (s) to cluster consecutive same-label events.
* **events.*** – lists of semantic event groups (S, G, K, F, C, E).
* **windows.*** – pre/post windows used to build clips (goal, penalty, foul-card,…).
* **filtering.confidence_thresholds** – per-label min confidence.
* **ranking.priority_order** & **scoring.base_scores / bonuses** – influence final ordering.

Adjust these values to match your highlight definition.

---

## 4.  Output Format

`*_highlights.json` is an array of highlight objects:

```json
[
  {
    "start": 745.0,     // seconds
    "end": 773.0,
    "label": "Goal",
    "score": 92.3
  },
  ...
]
```

You can feed this into your clipping script to render highlight videos.
