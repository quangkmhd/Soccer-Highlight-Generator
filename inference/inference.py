#!/usr/bin/env python3
# Soccer Action Spotting - End-to-end Inference Pipeline
import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
import tempfile
import shutil
import time
import torch
from datetime import datetime

# Thiết lập đường dẫn để tìm các module
root_dir = Path(__file__).parent.parent  # Đi lên một cấp từ thư mục inference
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "ball_action_spotting"))
sys.path.append(str(root_dir / "CALF_segmentation"))
sys.path.append(str(root_dir / "create_clips"))
sys.path.append(str(root_dir / "rules"))

# Import các module sau khi đã thiết lập đường dẫn
from ball_action_spotting.ball_action_predict import predict_on_video
from CALF_segmentation.camera_predict import camera_predict_on_video
from rules.rules_main import load_config as load_rules_config
from rules.rules_main import load_prediction_data, process_highlights, save_top_highlights
from rules.combine_predictions import combine_json_predictions
from create_clips.cut_clips import create_clips_from_json
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_output_directories(output_base: Path, video_name: str) -> dict:
    """Set up output directories for each step in the pipeline"""
    
    # Tạo thư mục chính với tên video
    video_dir = output_base / video_name
    
    # Đơn giản hóa cấu trúc thư mục
    dirs = {
        "base": video_dir,
        "predictions": video_dir,  # Để file JSON ở thư mục gốc của video
        "merged": video_dir,       # Để file highlights ở thư mục gốc của video
        "clips": video_dir         # Thư mục clips sẽ là thư mục con tự động tạo bởi cut_clips.py
    }
    
    # Tạo thư mục video nếu chưa tồn tại
    video_dir.mkdir(parents=True, exist_ok=True)
    
    return dirs

def run_ball_action_prediction(video_path: Path, config: dict, output_dir: Path) -> Path:
    """Run the ball action and action models prediction"""
    
    # Xác định đường dẫn file output
    output_file = output_dir / f"{video_path.stem}_ball_action.json"
    
    # Kiểm tra nếu file đã tồn tại
    if output_file.exists():
        logger.info(f"Ball action prediction file already exists: {output_file}. Skipping prediction.")
        return output_file
    
    logger.info("Running Ball Action and Action models prediction...")
    
    action_model_path = Path(config['models']['action_model'])
    ball_action_model_path = Path(config['models']['ball_action_model'])
    params = config['ball_action_params']
    
    try:
        # Run prediction
        predict_on_video(
            video_path=video_path,
            action_model_path=action_model_path,
            ball_action_model_path=ball_action_model_path,
            output_dir=output_dir,
            gpu_id=params['gpu_id'],
            batch_size=params['batch_size'],
            target_fps=params['target_fps']
        )
        
        # Verify file was created
        if not output_file.exists():
            logger.warning(f"Ball action prediction completed but file not found: {output_file}")
    except Exception as e:
        logger.error(f"Error during ball action prediction: {e}")
        # If file was partially created, we'll still try to use it
        if output_file.exists():
            logger.info(f"Using existing ball action prediction file despite error: {output_file}")
        else:
            raise  # Re-raise if no file was created
    
    return output_file

def run_camera_prediction(video_path: Path, config: dict, output_dir: Path) -> Path:
    """Run the camera model prediction using the dedicated function"""
    
    # Xác định đường dẫn file output
    expected_output = output_dir / f"{video_path.stem}_camera.json"
    
    # Kiểm tra nếu file đã tồn tại
    if expected_output.exists():
        logger.info(f"Camera prediction file already exists: {expected_output}. Skipping prediction.")
        return expected_output
    
    logger.info("Running Camera model prediction...")
    
    camera_model_path = Path(config['models']['camera_model'])
    params = config['camera_params']
    
    try:
        # Đường dẫn đến file PCA và scaler
        pca_file = Path("CALF_segmentation/Features/pca_512_TF2.pkl")
        scaler_file = Path("CALF_segmentation/Features/average_512_TF2.pkl")
        
        # Kiểm tra tồn tại của file PCA và scaler
        if not pca_file.exists():
            logger.warning(f"PCA file not found: {pca_file}")
        if not scaler_file.exists():
            logger.warning(f"Scaler file not found: {scaler_file}")
            
        # Call the camera prediction function
        output_file = camera_predict_on_video(
            video_path=str(video_path),
            model_path=str(camera_model_path),
            output_dir=str(output_dir),
            gpu_id=params['gpu_id'],
            fps=params['fps'],
            backend=params['backend'],
            batch_size_feat=params['batch_size_feat'],
            confidence_threshold=params['confidence_threshold'],
            pca_file=str(pca_file),
            scaler_file=str(scaler_file),
            num_classes_type=13  
        )
        
        output_path = Path(output_file)
        if not output_path.exists():
            logger.warning(f"Camera prediction completed but file not found: {output_path}")
    except Exception as e:
        logger.error(f"Error during camera prediction: {e}")
        # If file was created despite the error, use it
        if expected_output.exists():
            logger.info(f"Using existing camera prediction file despite error: {expected_output}")
            return expected_output
        else:
            raise  # Re-raise if no file was created
    
    return Path(output_file)



def apply_rules(combined_json: Path, rules_config: Path, output_dir: Path) -> Path:
    """Apply rules to combined predictions using rules_main"""
    
    # Load combined predictions để xác định video_basename
    prediction_data = load_prediction_data(combined_json)
    video_basename = Path(prediction_data.get('UrlLocal', combined_json.stem)).stem
    
    # Xác định tên file output (không thể biết trước số lượng highlight)
    # Nên ta sẽ kiểm tra bất kỳ file nào có dạng video_basename_top_*_highlights.json
    existing_files = list(output_dir.glob(f"{video_basename}_top_*_highlights.json"))
    if existing_files:
        logger.info(f"Found existing highlights file: {existing_files[0]}. Skipping rule application.")
        return existing_files[0]
    
    logger.info("Applying rules to combined predictions...")
    
    try:
        # Load rules configuration
        rules_config_data = load_rules_config(rules_config)
        
        # Process highlights
        top_highlights = process_highlights(prediction_data, rules_config_data)
        
        # Save highlights
        limit = len(top_highlights)
        output_file = output_dir / f"{video_basename}_top_{limit}_highlights.json"
        
        save_top_highlights(top_highlights, output_file)
        logger.info(f"Top highlights saved to {output_file}")
        
        if not output_file.exists():
            logger.warning(f"Rules applied but output file not found: {output_file}")
    except Exception as e:
        logger.error(f"Error during rules application: {e}")
        # Check if any file was created despite error
        new_files = list(output_dir.glob(f"{video_basename}_top_*_highlights.json"))
        if new_files:
            logger.info(f"Using existing highlights file despite error: {new_files[0]}")
            return new_files[0]
        else:
            raise  # Re-raise if no file was created
    
    return output_file

def cut_video_clips(highlights_json: Path, video_path: Path, config: dict, output_dir: Path) -> Path:
    """Cut video into clips based on the highlights JSON"""
    
    # Tạo thư mục video clips nếu chưa tồn tại
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Trong cấu trúc mới, thư mục clip sẽ được tạo tự động bởi cut_clips.py
    # Nó sẽ là output_dir/video_name
    video_name = video_path.stem
    
    # Kiểm tra xem đã có video clips trong thư mục chưa
    # cut_clips.py sẽ tạo thư mục clips với tên video_name
    if any(output_dir.glob(f"{video_name}/**/*.mp4")):
        logger.info(f"Found existing video clips in {output_dir}/{video_name}. Skipping clip cutting.")
        return output_dir
    
    logger.info("Cutting video into clips...")
    
    rules_config_path = Path(config['rules']['config_path'])
    if not rules_config_path.exists():
        raise FileNotFoundError(f"Rules config not found at: {rules_config_path}")
    
    try:
        create_clips_from_json(
            json_path=str(highlights_json),
            video_path=str(video_path),
            output_dir=str(output_dir),
            rules_config_path=str(rules_config_path)
        )
        
        # Kiểm tra xem có clip nào được tạo ra không
        clips_path = output_dir / video_name
        if not clips_path.exists() or not any(clips_path.glob("**/*.mp4")):
            logger.warning(f"No clips were created in {clips_path}")
    except Exception as e:
        logger.error(f"Error during video clip cutting: {e}")
        # Nếu đã có clip được tạo ra, vẫn trả về thư mục
        clips_path = output_dir / video_name
        if clips_path.exists() and any(clips_path.glob("**/*.mp4")):
            logger.info(f"Some clips were created despite error. Using existing clips.")
            return output_dir
        else:
            raise  # Re-raise nếu không có clip nào được tạo
    
    return output_dir

def run_inference_pipeline(video_path: Path, config_path: Path = Path("inference/inference_config.yaml")) -> Path:
    """Run the complete inference pipeline"""
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories
    output_base = Path(config['paths']['output_dir'])
    output_dirs = setup_output_directories(output_base, video_path.stem)
    
    # Run Ball Action and Action models prediction
    ball_action_json = run_ball_action_prediction(
        video_path=video_path,
        config=config,
        output_dir=output_dirs["predictions"]
    )
    
    # Run Camera model prediction
    camera_json = run_camera_prediction(
        video_path=video_path,
        config=config,
        output_dir=output_dirs["predictions"]
    )
    
    # Step 1: Combine predictions from both models into a single JSON file
    try:
        video_basename = video_path.stem
        expected_combined_json = output_dirs["predictions"] / f"{video_basename}_predicted.json"
        
        # Kiểm tra nếu file đã tồn tại
        if expected_combined_json.exists():
            logger.info(f"Combined predictions file already exists: {expected_combined_json}. Skipping combination.")
            combined_json = expected_combined_json
        else:
            logger.info("Combining ball-action and camera predictions...")
            combined_json = combine_json_predictions(
                ball_action_json=ball_action_json,
                camera_json=camera_json,
                output_dir=output_dirs["predictions"]
            )
    except Exception as e:
        logger.error(f"Error during prediction combination: {e}")
        # Nếu file đã được tạo, sử dụng nó
        if expected_combined_json.exists():
            logger.info(f"Using existing combined file despite error: {expected_combined_json}")
            combined_json = expected_combined_json
        else:
            raise
    
    # Step 2: Apply rules to the combined predictions
    rules_config_path = Path(config['rules']['config_path'])
    highlights_json = apply_rules(
        combined_json=combined_json,
        rules_config=rules_config_path,
        output_dir=output_dirs["merged"]
    )
    
    # Cut video into clips
    clips_dir = cut_video_clips(
        highlights_json=highlights_json,
        video_path=video_path,
        config=config,
        output_dir=output_dirs["clips"]
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Complete inference pipeline finished in {elapsed_time:.2f} seconds")
    logger.info(f"Output clips available at: {clips_dir}")
    
    return clips_dir

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Soccer Action Spotting - End-to-end Inference Pipeline")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video file")
    parser.add_argument("--config", type=Path, default=Path("infrence/inference_config.yaml"), help="Path to the inference configuration file")
    args = parser.parse_args()
    

    output_dir = run_inference_pipeline(args.video, args.config)
    logger.info(f"Successfully processed video. Output available at: {output_dir}")

if __name__ == "__main__":
    sys.exit(main())
