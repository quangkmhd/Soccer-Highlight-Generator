"""Utility helpers for the Gradio demo (filename parsing, sorting).

Follow user's code style: short, typed, reusable.
"""
from __future__ import annotations

from typing import Any, Dict, List


def extract_score_from_filename(filename: str) -> str:
    """Extract score text from a clip filename.
    Example: '47_01-27-43_01-28-42_goal.mp4' -> 'Score: 47'
    """
    try:
        score = filename.split("_")[0]
        return f"Score: {score}" if score.isdigit() else filename
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
