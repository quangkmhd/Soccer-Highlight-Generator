"""SRT Export Utility Functions

This module provides functions to parse clip filenames and generate SRT subtitle files
from soccer action clips with embedded timing and event information.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def parse_clip_filename(filename: str) -> Dict:
    """Parse clip filename to extract timing and event information.
    
    Expected format: "score_H-MM-SS_H-MM-SS_event.mp4"
    Example: "100_02-33-16_02-34-50_Penalty → Goal.mp4"
    
    Args:
        filename: The clip filename to parse
        
    Returns:
        Dict with keys: score, start_seconds, end_seconds, event
        Returns fallback values if parsing fails
    """
    # Remove file extension
    name = Path(filename).stem
    
    # Pattern to match: score_H-MM-SS_H-MM-SS_event
    pattern = r'^(\d+)_(\d{1,2})-(\d{2})-(\d{2})_(\d{1,2})-(\d{2})-(\d{2})_(.+)$'
    match = re.match(pattern, name)
    
    if not match:
        logger.warning(f"Could not parse filename: {filename}")
        return {
            "score": 0,
            "start_seconds": 0.0,
            "end_seconds": 5.0,  # Default 5 second duration
            "event": "Soccer Event"
        }
    
    try:
        score, start_h, start_m, start_s, end_h, end_m, end_s, event = match.groups()
        
        start_seconds = int(start_h) * 3600 + int(start_m) * 60 + int(start_s)
        end_seconds = int(end_h) * 3600 + int(end_m) * 60 + int(end_s)
        
        return {
            "score": int(score),
            "start_seconds": float(start_seconds),
            "end_seconds": float(end_seconds),
            "event": event
        }
        
    except ValueError as e:
        logger.warning(f"Error parsing time values in filename {filename}: {e}")
        return {
            "score": 0,
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "event": "Soccer Event"
        }


def clips_to_srt_events(selected_filenames: List[str]) -> List[Dict]:
    """Convert selected clip filenames to sorted events list by start time.
    
    Args:
        selected_filenames: List of clip filenames to process
        
    Returns:
        List of event dictionaries sorted by start_seconds
    """
    events = []
    
    for filename in selected_filenames:
        parsed = parse_clip_filename(filename)
        events.append({
            "filename": filename,
            "start_seconds": parsed["start_seconds"],
            "end_seconds": parsed["end_seconds"],
            "score": parsed["score"],
            "event": parsed["event"]
        })
    
    # Sort by start time
    events.sort(key=lambda x: x["start_seconds"])
    return events


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT format timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt_file(events: List[Dict], output_path: Path) -> bool:
    """Write events to SRT file format.
    
    Args:
        events: List of event dictionaries with timing and content
        output_path: Path where to write the SRT file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, event in enumerate(events, 1):
                # SRT entry number
                f.write(f"{i}\n")
                
                # Timestamp line
                start_time = seconds_to_srt_time(event["start_seconds"])
                end_time = seconds_to_srt_time(event["end_seconds"])
                f.write(f"{start_time} --> {end_time}\n")
                
                # Caption text with score and event
                caption = f"[{event['score']}] {event['event']}"
                f.write(f"{caption}\n")
                
                # Empty line separator
                f.write("\n")
        
        logger.info(f"SRT file written successfully to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing SRT file to {output_path}: {e}")
        return False


def generate_srt_from_clips(selected_filenames: List[str], video_name: str, output_dir: Path) -> Optional[Path]:
    """Generate SRT file from selected clip filenames.
    
    Args:
        selected_filenames: List of clip filenames to include
        video_name: Name of the source video (for output filename)
        output_dir: Directory to save the SRT file
        
    Returns:
        Path to generated SRT file, or None if failed
    """
    if not selected_filenames:
        logger.warning("No clips provided for SRT generation")
        return None
    
    try:
        # Parse clips to events
        events = clips_to_srt_events(selected_filenames)
        
        if not events:
            logger.warning("No valid events found after parsing clips")
            return None
        
        # Generate output filename
        safe_video_name = re.sub(r'[^\w\-_\.]', '_', video_name)
        output_filename = f"{safe_video_name}_highlights.srt"
        output_path = output_dir / output_filename
        
        # Write SRT file
        if write_srt_file(events, output_path):
            return output_path
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error generating SRT from clips: {e}")
        return None
