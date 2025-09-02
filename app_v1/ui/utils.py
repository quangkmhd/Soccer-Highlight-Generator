"""Utility helpers for the Gradio demo (filename parsing, sorting).

Follow user's code style: short, typed, reusable.
"""
from __future__ import annotations

from typing import Any, Dict, List


def extract_score_from_filename(filename: str) -> str:
    """Extract a readable *event label* and *score* from a clip filename.

    Examples
    --------
    '47_01-27-43_01-28-42_goal.mp4' -> 'Goal Score: 47'
    '47_01-27-43_01-28-42_foul-card.mp4' -> 'Foul -> Card Score: 47'
    """
    try:
        parts = filename.split("_")
        score_part = parts[0]
        if not score_part.isdigit():
            return filename  # Unexpected pattern

        # Strip extension and derive event label
        label_raw = parts[-1].split(".")[0]
        
        # Event names are now unsanitized, so use them directly
        label_text = label_raw

        return f"{label_text} Score: {score_part}" if label_text else f"Score: {score_part}"
    except Exception:
        return filename


def get_numeric_score(filename: str) -> int:
    """Numeric score for sorting, fallback 0."""
    try:
        score = filename.split("_")[0]
        return int(score) if score.isdigit() else 0
    except Exception:
        return 0


def sort_clips_by_score(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort clips by score descending."""
    return sorted(clips, key=lambda c: get_numeric_score(c.get("filename", "")), reverse=True)
