#!/usr/bin/env python3
# Soccer Action Spotting - End-to-end Inference Pipeline with Parallel Processing
import argparse
import json
import logging
import os
import sys
import yaml
import gc
from pathlib import Path
import tempfile
import shutil
import time
import torch
from datetime import datetime
import multiprocessing
from multiprocessing import Process, Queue

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
from rules.rules_main import load_config as load_rules_config, load_prediction_data, process_highlights
from rules.combine_predictions import combine_json_predictions
from create_clips.cut_clips import create_clips_from_json
from rules.rank_score import score_highlights_from_data, write_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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

def run_ball_action_prediction(video_path: Path, config: dict, output_dir: Path, result_queue=None) -> Path:
    """Run the ball action and action models prediction"""
    
    # Xác định đường dẫn file output
    output_file = output_dir / f"{video_path.stem}_ball_action.json"
    
    # Kiểm tra nếu file đã tồn tại
    if output_file.exists():
        logger.info(f"Ball action prediction file already exists: {output_file}. Skipping prediction.")
        if result_queue is not None:
            result_queue.put(output_file)
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
            if result_queue is not None:
                result_queue.put(Exception(f"Error during ball action prediction: {e}"))
            raise  # Re-raise if no file was created
    
    if result_queue is not None:
        result_queue.put(output_file)
    return output_file

def run_camera_prediction(video_path: Path, config: dict, output_dir: Path, result_queue=None) -> Path:
    """Run the camera model prediction using PyTorch-based function"""
    
    # Xác định đường dẫn file output
    expected_output = output_dir / f"{video_path.stem}_camera.json"
    
    # Kiểm tra nếu file đã tồn tại
    if expected_output.exists():
        logger.info(f"Camera prediction file already exists: {expected_output}. Skipping prediction.")
        if result_queue is not None:
            result_queue.put(expected_output)
        return expected_output
    
    logger.info("Running Camera model prediction with PyTorch...")
    
    camera_model_path = Path(config['models']['camera_model'])
    params = config['camera_params']
    
    try:
        # Đường dẫn đến file PCA và scaler
        pca_file = Path("CALF_segmentation/Features/pca_512_PT.pkl")
        scaler_file = Path("CALF_segmentation/Features/average_512_PT.pkl")

        # Kiểm tra tồn tại của file PCA và scaler - dừng luôn nếu không có
        if not pca_file.exists():
            error_msg = f"PCA file not found: {pca_file}. Camera prediction cannot proceed without PCA file."
            logger.error(error_msg)
            if result_queue is not None:
                result_queue.put(Exception(error_msg))
            raise FileNotFoundError(error_msg)
        if not scaler_file.exists():
            error_msg = f"Scaler file not found: {scaler_file}. Camera prediction cannot proceed without scaler file."
            logger.error(error_msg)
            if result_queue is not None:
                result_queue.put(Exception(error_msg))
            raise FileNotFoundError(error_msg)
            
        # Call the PyTorch camera prediction function
        output_file = camera_predict_on_video(
            video_path=str(video_path),
            model_path=str(camera_model_path),
            output_dir=str(output_dir),
            gpu_id=params['gpu_id'],
            fps=params['fps'],
            batch_size_feat=params['batch_size_feat'],
            confidence_threshold=params['confidence_threshold'],
            num_classes_type=13,
            chunk_size=params.get('chunk_size', 48),
            receptive_field=params.get('receptive_field', 16),
            num_detections=params.get('num_detections', 45),
            overwrite=params.get('overwrite', True),
            pca_file=pca_file,
            scaler_file=scaler_file
        )
        
        output_path = Path(output_file)
        if not output_path.exists():
            logger.warning(f"Camera prediction completed but file not found: {output_path}")
    except Exception as e:
        logger.error(f"Error during PyTorch camera prediction: {e}")
        # If file was created despite the error, use it
        if expected_output.exists():
            logger.info(f"Using existing camera prediction file despite error: {expected_output}")
            if result_queue is not None:
                result_queue.put(expected_output)
            return expected_output
        else:
            if result_queue is not None:
                result_queue.put(Exception(f"Error during PyTorch camera prediction: {e}"))
            raise  # Re-raise if no file was created
    
    if result_queue is not None:
        result_queue.put(Path(output_file))
    return Path(output_file)

def apply_rules_and_scoring(combined_json: Path, rules_config: Path, output_dir: Path) -> Path:
    """Apply rules and scoring system to the combined predictions"""
    prediction_data = load_prediction_data(combined_json)
    video_basename = Path(prediction_data.get('UrlLocal', combined_json.stem)).stem
    
    scored_output_file = output_dir / f"{video_basename}_scores.json"
    if scored_output_file.exists():
        logger.info(f"Found existing scored highlights file: {scored_output_file}. Skipping rules and scoring.")
        return scored_output_file
    
    logger.info("Applying rules and scoring to combined predictions...")
    
    try:
        rules_config_data = load_rules_config(rules_config)
        top_highlights = process_highlights(prediction_data, rules_config_data)
        
        logger.info("Applying scoring system to highlights...")
        scored_results = score_highlights_from_data(
            highlights_data=top_highlights,
            sort_by_score=True
        )
        
        write_report(scored_results, str(scored_output_file))
        logger.info(f"Scored highlights saved to {scored_output_file}")
        
    except Exception as e:
        logger.error(f"Error during rules application and scoring: {e}")
        if scored_output_file.exists():
            return scored_output_file
        raise
        
    return scored_output_file

def cut_video_clips(highlights_json: Path, video_path: Path, config: dict, output_dir: Path) -> Path:
    """Cut video into clips based on the highlights JSON"""
    
    # Tạo thư mục video clips nếu chưa tồn tại
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = video_path.stem
    
    if any(output_dir.glob(f"{video_name}/**/*.mp4")):
        logger.info(f"Found existing video clips in {output_dir}/{video_name}. Skipping clip cutting.")
        return output_dir

    
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

def run_inference_pipeline_parallel(video_path: Path, config_path: Path = Path("inference/inference_config.yaml")) -> Path:
    """Run the complete inference pipeline with parallel processing"""
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories
    output_base = Path(config['paths']['output_dir'])
    output_dirs = setup_output_directories(output_base, video_path.stem)
    
    # Run Ball Action and Camera predictions in parallel
    logger.info("Starting parallel predictions for ball action and camera models")
    
    # Use queues to get results from processes
    ball_action_queue = multiprocessing.Queue()
    camera_queue = multiprocessing.Queue()
    
    # Create processes for each prediction
    ball_action_process = Process(
        target=run_ball_action_prediction,
        args=(video_path, config, output_dirs["predictions"], ball_action_queue)
    )
    
    camera_process = Process(
        target=run_camera_prediction,
        args=(video_path, config, output_dirs["predictions"], camera_queue)
    )
    
    # Start both processes
    ball_action_process.start()
    camera_process.start()
    
    # Wait for results
    try:
        ball_action_result = ball_action_queue.get()
        camera_result = camera_queue.get()
        
        # Check for exceptions
        if isinstance(ball_action_result, Exception):
            raise ball_action_result
        if isinstance(camera_result, Exception):
            raise camera_result
        
        ball_action_json = ball_action_result
        camera_json = camera_result
        
        logger.info("Both prediction processes completed successfully")
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise
    finally:
        # Clean up processes
        if ball_action_process.is_alive():
            ball_action_process.terminate()
        if camera_process.is_alive():
            camera_process.terminate()
            
        ball_action_process.join()
        camera_process.join()
    
    # Force garbage collection after model predictions
    gc.collect()
    
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
    
    # Step 2: Apply rules and scoring to the combined predictions
    rules_config_path = Path(config['rules']['config_path'])
    highlights_json = apply_rules_and_scoring(
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
    logger.info(f"Complete parallel inference pipeline finished in {elapsed_time:.2f} seconds")
    logger.info(f"Output clips available at: {clips_dir}")
    
    return clips_dir

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Soccer Action Spotting - End-to-end Parallel Inference Pipeline")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video file")
    parser.add_argument("--config", type=Path, default=Path("inference/inference_config.yaml"), help="Path to the inference configuration file")
    args = parser.parse_args()
    
    # On Linux, must use 'spawn' method for safe parallel processing with GPU
    multiprocessing.set_start_method('spawn')
    
    output_dir = run_inference_pipeline_parallel(args.video, args.config)
    logger.info(f"Successfully processed video. Output available at: {output_dir}")

if __name__ == "__main__":
    sys.exit(main())