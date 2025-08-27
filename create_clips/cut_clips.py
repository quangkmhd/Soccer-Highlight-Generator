import json
import os
import subprocess
import argparse
import logging
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def time_str_to_seconds(time_str):
    """Converts a H:M:S(.sss) string to seconds (float)."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)

def load_rules_config(rules_config_path):
    """Load rules configuration file"""
    if not rules_config_path or not os.path.exists(rules_config_path):
        return None
    
    with open(rules_config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def determine_clip_event_name(highlight, rules_config):
    """Determine the appropriate event name based on Goal confidence and pre-events"""
    if not rules_config:
        return highlight.get("event", "Unknown")
    
    confidence_thresholds = rules_config.get('filtering', {}).get('confidence_thresholds', {})
    priority_order = rules_config.get('ranking', {}).get('priority_order', [])
    
    # Get all events in this clip
    clip_events = highlight.get('events', [])
    if not clip_events:
        return highlight.get("event", "Unknown")
    
    # Find Goal events and their confidence
    goal_events = [e for e in clip_events if e.get('label') == 'Goal']
    goal_confidence = max([e.get('confidence', 0) for e in goal_events]) if goal_events else 0
    # For naming purposes, use a fixed 0.75 threshold for Goal vs. Shot decision
    goal_naming_threshold = 0.75
    
    # Check for pre-events in priority order - only use if they meet confidence thresholds
    pre_events = ['Penalty', 'Direct free-kick', 'Corner']
    qualifying_pre_event = None
    
    for pre_event in pre_events:
        if pre_event not in priority_order:
            continue
            
        pre_event_instances = [e for e in clip_events if e.get('label') == pre_event]
        if not pre_event_instances:
            continue
            
        pre_event_confidence = max([e.get('confidence', 0) for e in pre_event_instances])
        pre_event_threshold = confidence_thresholds.get(pre_event, 0.75)
        
        # Only consider pre-event if it meets confidence threshold
        if pre_event_confidence >= pre_event_threshold:
            qualifying_pre_event = pre_event
            break  # Use first qualifying pre-event based on priority order
    
    # Determine naming based on goal confidence and qualifying pre-events
    if qualifying_pre_event:
        if goal_confidence >= goal_naming_threshold:
            return f"{qualifying_pre_event} → Goal"
        else:
            return qualifying_pre_event
    
    # No qualifying pre-events found, check Goal alone
    if goal_events: # Check if a goal event exists
        if goal_confidence >= goal_naming_threshold:
            return "Goal"
        else:
            return "Shot"
    
    # Fallback to original event name
    return highlight.get("event", "Unknown")

def create_clips_from_json(json_path, video_path, output_dir, rules_config_path=None):


    with open(json_path, 'r', encoding='utf-8') as f:
        highlights = json.load(f)

    # Load rules config for intelligent naming
    rules_config = load_rules_config(rules_config_path)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Use a subdirectory within the main output dir for clips of this video
    video_clips_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_clips_dir, exist_ok=True)

    for highlight in highlights:
        # Parse the new JSON format from rank_score.py
        rank = highlight.get("rank")
        # Use intelligent event naming based on Goal confidence and pre-events
        event = determine_clip_event_name(highlight, rules_config)
        score = highlight.get("score")
        start_time_str = highlight.get("start")  # Already in HH:MM:SS format
        end_time_str = highlight.get("end")      # Already in HH:MM:SS format
        
        # Convert time strings to seconds for ffmpeg
        start_time = time_str_to_seconds(start_time_str)
        end_time = time_str_to_seconds(end_time_str)
        
        # Sanitize time strings for filename (replace ':' with '-')
        def sanitize_time_str(time_str):
            return time_str.replace(':', '-') if time_str else ''
        
        start_time_str_sanitized = sanitize_time_str(start_time_str)
        end_time_str_sanitized = sanitize_time_str(end_time_str)
        
        # Sanitize event name for filename
        event_sanitized = event
        
        # Create filename: rank_start_end_event_score.mp4
        output_filename = f"{score}_{start_time_str_sanitized}_{end_time_str_sanitized}_{event_sanitized}.mp4"

        # Save directly in video folder (no event subfolders)
        output_filepath = os.path.join(video_clips_dir, output_filename)

        if os.path.exists(output_filepath):
            # logging.info(f"Clip already exists, skipping: {output_filepath}")
            continue

        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',  # Use stream copy for much faster processing
            output_filepath
        ]
        
        # ffmpeg_command = [
        #     'ffmpeg',
        #     '-i', video_path,
        #     '-ss', str(start_time),
        #     '-to', str(end_time),
        #     '-c:v', 'libx264',  # Re-encode for precise cuts
        #     '-c:a', 'aac',
        #     output_filepath
        # ]




        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"Successfully created clip: {os.path.basename(output_filepath)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut video clips based on a JSON file of highlights.")
    parser.add_argument("-j", "--json_path", required=True, help="Path to the highlights JSON file.")    
    parser.add_argument("-v", "--video_path", required=True, help="Path to the source video file.")
    
    parser.add_argument("-o", "--output_dir", required=True )
    parser.add_argument("-rc", "--rules_config")
    
    args = parser.parse_args()  

    create_clips_from_json(args.json_path, args.video_path, args.output_dir, args.rules_config) 