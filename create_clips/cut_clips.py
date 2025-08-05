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

def create_clips_from_json(json_path, video_path, output_dir, rules_config_path):

    if not os.path.exists(video_path):
        logging.error(f"Video file not found at: {video_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            highlights = json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found at: {json_path}")
        return
        
    try:
        with open(rules_config_path, 'r', encoding='utf-8') as f:
            rules_config = yaml.safe_load(f)
        priority_order = rules_config['ranking']['priority_order']
    except FileNotFoundError:
        logging.error(f"Rules config file not found at: {rules_config_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Use a subdirectory within the main output dir for clips of this video
    video_clips_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_clips_dir, exist_ok=True)


    for i, highlight in enumerate(highlights):
        start_time = highlight.get("start_time")
        end_time = highlight.get("end_time")
        # Nếu start_time là chuỗi, chuyển sang float (giây)
        if isinstance(start_time, str):
            try:
                start_time = time_str_to_seconds(start_time)
            except Exception:
                pass
        if isinstance(end_time, str):
            try:
                end_time = time_str_to_seconds(end_time)
            except Exception:
                pass
        start_time_str = highlight.get("start_time_str")
        end_time_str = highlight.get("end_time_str")
        if not start_time_str and start_time is not None:
            # Tạo chuỗi HH:MM:SS từ giây
            h = int(start_time // 3600)
            m = int((start_time % 3600) // 60)
            s = start_time % 60
            start_time_str = f"{h:02}:{m:02}:{s:06.3f}" if isinstance(s, float) else f"{h:02}:{m:02}:{s:02}"
        if not end_time_str and end_time is not None:
            h = int(end_time // 3600)
            m = int((end_time % 3600) // 60)
            s = end_time % 60
            end_time_str = f"{h:02}:{m:02}:{s:06.3f}" if isinstance(s, float) else f"{h:02}:{m:02}:{s:02}"
        
        # --- Sanitize time strings for filename (replace ':' with '-') ---
        def sanitize_time_str(time_str):
            return time_str.replace(':', '-') if time_str else ''
        
        start_time_str_sanitized = sanitize_time_str(start_time_str)
        end_time_str_sanitized = sanitize_time_str(end_time_str)
        
        # --- End of sanitize ---
        
        # --- New filename logic ---

        action_labels = []
        if 'merged_info' in highlight and 'primary_actions_merged' in highlight['merged_info']:
            merged_actions = highlight['merged_info']['primary_actions_merged']
            merged_actions.sort(key=lambda x: priority_order.index(x['label']) if x['label'] in priority_order else 99)
            action_labels = [action['label'] for action in merged_actions]
        else:
            present_priority_events = {}
            for event in highlight.get('events', []):
                if event.get('label') in priority_order:
                    label = event['label']
                    if label not in present_priority_events or event.get('confidence', 0) > present_priority_events[label].get('confidence', 0):
                        present_priority_events[label] = event
            
            sorted_events = sorted(present_priority_events.values(), key=lambda x: priority_order.index(x['label']))
            action_labels = [event['label'] for event in sorted_events]

        sanitized_action_labels = [label.lower().replace(' ', '-').replace('->', '-') for label in action_labels if label]
        actions_str = "-".join(sanitized_action_labels)

        title = highlight.get('title', 'Unknown Event')
        main_event_label = title.split(' (')[0]
        
        primary_event = None
        for event in highlight.get('events', []):
            if event.get('label') == main_event_label:
                if primary_event is None or event.get('confidence', 0) > primary_event.get('confidence', 0):
                    primary_event = event
        
        conf = primary_event.get('confidence', 0) if primary_event else 0
        
        output_filename = f"{start_time_str_sanitized}--{end_time_str_sanitized}_{actions_str}_{conf:.2f}.mp4"

        # --- End of new filename logic ---

        # Extract event name from title, e.g., "Penalty" from "Penalty (at 12:45)"
        event_folder_name = title.split(' (')[0].strip()
        
        # Sanitize folder name
        event_folder_name = "".join([c for c in event_folder_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        
        event_dir = os.path.join(video_clips_dir, event_folder_name)
        os.makedirs(event_dir, exist_ok=True)

        output_filepath = os.path.join(event_dir, output_filename)

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
    
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the output clips.")
    parser.add_argument("-rc", "--rules_config", default="rules/config.yaml", help="Path to the rules config YAML file.")
    
    args = parser.parse_args()  

    create_clips_from_json(args.json_path, args.video_path, args.output_dir, args.rules_config) 