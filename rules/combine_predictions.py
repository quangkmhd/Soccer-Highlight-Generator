import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def combine_json_predictions(ball_action_json: Path, camera_json: Path, output_dir: Path) -> Path:
    """Combine the predictions from ball action and camera models into a single JSON file
    
    Args:
        ball_action_json: Path to the JSON file with ball action predictions
        camera_json: Path to the JSON file with camera predictions
        output_dir: Directory to save the combined JSON file
        
    Returns:
        Path to the combined JSON file
    """
    logger.info("Combining ball-action and camera predictions...")
    
    with open(ball_action_json, 'r', encoding='utf-8') as f:
        ball_action_data = json.load(f)
 
    with open(camera_json, 'r', encoding='utf-8') as f:
        camera_data = json.load(f)
    
    video_path = ball_action_data.get('UrlLocal', ball_action_json.stem)
    video_basename = Path(video_path).stem
    
    camera_events = camera_data if isinstance(camera_data, list) else camera_data.get('predictions', [])
    
    combined_data = {
        "predictions": ball_action_data.get('predictions', []) + camera_events,
        "UrlLocal": video_path
    }
    
    combined_json_path = output_dir / f"{video_basename}_predicted.json"
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Combined predictions saved to {combined_json_path}")
    return combined_json_path