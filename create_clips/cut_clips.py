import json
import os
import subprocess
import argparse
import logging
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

def create_clips_from_json(json_path, video_path, output_dir, rules_config_path=None):

    if not os.path.exists(video_path):
        logging.error(f"Video file not found at: {video_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            highlights = json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found at: {json_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Use a subdirectory within the main output dir for clips of this video
    video_clips_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_clips_dir, exist_ok=True)

    for highlight in highlights:
        # Parse the new JSON format from rank_score.py
        rank = highlight.get("rank")
        event = highlight.get("event")
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
        event_sanitized = event.lower().replace(' ', '-').replace('->', '-')
        
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



        try:
            # logging.info(f"Executing command: {' '.join(ffmpeg_command)}")
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"Successfully created clip: {os.path.basename(output_filepath)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating clip {output_filepath}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut video clips based on a JSON file of highlights.")
    parser.add_argument("-j", "--json_path", required=True, help="Path to the highlights JSON file.")    
    parser.add_argument("-v", "--video_path", required=True, help="Path to the source video file.")
    
    parser.add_argument("-o", "--output_dir", required=True )
    parser.add_argument("-rc", "--rules_config")
    
    args = parser.parse_args()  

    create_clips_from_json(args.json_path, args.video_path, args.output_dir, args.rules_config) 