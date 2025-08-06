import argparse
import json
import os
import sys
import yaml
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

def load_scoring_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load scoring configuration from YAML file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["scoring"]

# Load default scoring config
_scoring = load_scoring_config()
BASE_SCORES = _scoring["base_scores"]
DEFAULT_BASE_SCORE = _scoring["default_base_score"]
CAMERA_BONUS = _scoring["camera_bonus"]
COMBO_BONUSES = _scoring["combo_bonuses"]
MAX_SCORE = _scoring["max_score"]

def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def calculate_highlight_score(
    title: str, 
    confidence: float, 
    events: List[Dict[str, Any]], 
    scoring_config: Optional[Dict[str, Any]] = None
) -> int:
    """Calculate highlight score based on title, confidence and camera events.
    
    Args:
        title: Event title (e.g., 'Goal', 'Penalty', 'Corner')
        confidence: Confidence score (0.0 to 1.0)
        events: List of camera/context events with labels
        scoring_config: Optional custom scoring configuration
    
    Returns:
        Calculated score (0 to MAX_SCORE)
    """
    if scoring_config is None:
        scoring_config = _scoring
    
    base_scores = scoring_config["base_scores"]
    default_base = scoring_config["default_base_score"]
    camera_bonus = scoring_config["camera_bonus"]
    combo_bonuses = scoring_config["combo_bonuses"]
    max_score = scoring_config["max_score"]
    
    # Calculate base score
    base = base_scores.get(title, default_base) * confidence
    
    # Calculate camera bonus
    labels = {e["label"] for e in events}
    bonus = sum(camera_bonus.get(lbl, 0) for lbl in labels)
    
    # Extra bonus for Goal/Penalty
    if title in {"Goal", "Penalty"}:
        if {"Inside the goal", "Public"}.issubset(labels):
            bonus += combo_bonuses["spectacular_view_bonus"]
        if "Goal line technology camera" in labels:
            bonus += combo_bonuses["tech_confirmation_bonus"]
    
    return int(min(max_score, base + bonus))


def load_highlights(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load highlights from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def default_output_path(input_path: Union[str, Path]) -> str:
    """Generate default output path for scored highlights"""
    name = os.path.basename(str(input_path)).replace(".json", "_scores.json")
    return os.path.join("pipeline_output", name)

def process_single_highlight(
    highlight: Dict[str, Any], 
    highlight_id: int = 1,
    scoring_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a single highlight and calculate its score.
    
    Args:
        highlight: Highlight data with title, confidence, start_time, end_time, events
        highlight_id: ID for the highlight
        scoring_config: Optional custom scoring configuration
    
    Returns:
        Processed highlight with score and formatted times
    """
    title = highlight.get("title", "Unknown")
    conf = float(highlight.get("confidence", 0))
    events = highlight.get("events", [])
    score = calculate_highlight_score(title, conf, events, scoring_config)
    
    start, end = float(highlight["start_time"]), float(highlight["end_time"])
    
    # Event summary for detailed analysis
    summary = defaultdict(lambda: {"count": 0, "max_confidence": 0.0})
    for ev in events:
        lbl = ev["label"]
        summary[lbl]["count"] += 1
        summary[lbl]["max_confidence"] = max(
            summary[lbl]["max_confidence"], 
            ev.get("confidence", 1.0)
        )
    
    return {
        "highlight_id": highlight_id,
        "start_time": format_time(start),
        "end_time": format_time(end), 
        "duration": format_time(end - start),
        "score": score,
        "events_summary": summary,
        "title": title,
        "primary_confidence": conf,
        "events": events  # Keep original events for reference
    }


def write_report(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Write scored results to JSON file in the format expected by cut_clips.py"""
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
    
    # Prepare JSON output structure
    json_output = []
    
    for i, item in enumerate(results, 1):
        json_item = {
            "event": item["title"],
            "rank": i,
            "score": item["score"],
            "start": item["start_time"],
            "end": item["end_time"]
        }
        json_output.append(json_item)
    
    # Write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

def score_highlights_from_file(
    input_path: Union[str, Path], 
    output_path: Optional[Union[str, Path]] = None,
    scoring_config: Optional[Dict[str, Any]] = None,
    sort_by_score: bool = True
) -> Tuple[List[Dict[str, Any]], Path]:
    """Score highlights from a JSON file and optionally save results.
    
    Args:
        input_path: Path to input highlights JSON file
        output_path: Path to output scored highlights file (optional)
        scoring_config: Custom scoring configuration (optional)
        sort_by_score: Whether to sort results by score (default: True)
    
    Returns:
        Tuple of (scored_results, output_file_path)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_scores.json"
    else:
        output_path = Path(output_path)
    
    # Load highlights
    highlights = load_highlights(input_path)
    
    # Process each highlight
    results = []
    for i, hl in enumerate(highlights, 1):
        processed = process_single_highlight(hl, i, scoring_config)
        results.append(processed)
    
    # Sort by score if requested
    if sort_by_score:
        results.sort(key=lambda x: x["score"], reverse=True)
    
    # Save results
    write_report(results, output_path)
    
    return results, output_path

def score_highlights_from_data(
    highlights_data: List[Dict[str, Any]],
    scoring_config: Optional[Dict[str, Any]] = None,
    sort_by_score: bool = True
) -> List[Dict[str, Any]]:
    """Score highlights from data directly without file I/O.
    
    Args:
        highlights_data: List of highlight dictionaries
        scoring_config: Custom scoring configuration (optional)
        sort_by_score: Whether to sort results by score (default: True)
    
    Returns:
        List of scored highlights
    """
    results = []
    for i, hl in enumerate(highlights_data, 1):
        processed = process_single_highlight(hl, i, scoring_config)
        results.append(processed)
    
    if sort_by_score:
        results.sort(key=lambda x: x["score"], reverse=True)
    
    return results


def main(argv: List[str] | None = None) -> None:
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Smart highlight scoring")
    parser.add_argument("input_json", help="Input JSON highlights path")
    parser.add_argument("-o", "--output", help="Output report path")
    parser.add_argument("--config", help="Custom scoring config path")
    args = parser.parse_args(argv or sys.argv[1:])
    
    # Load custom config if provided
    scoring_config = None
    if args.config:
        scoring_config = load_scoring_config(args.config)
    
    # Use the utility function
    output_path = args.output or default_output_path(args.input_json)
    results, final_output_path = score_highlights_from_file(
        input_path=args.input_json,
        output_path=output_path,
        scoring_config=scoring_config,
        sort_by_score=True
    )
    
    print(f"✅ Report: {final_output_path}")
    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"🏆 Top: {results[0]['score']} | 📊 Avg: {avg:.1f}")
        print(f"📋 Total highlights: {len(results)}")

if __name__ == "__main__":
    main()
