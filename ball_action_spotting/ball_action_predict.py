
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import timedelta
import torch
import numpy as np
from tqdm import tqdm

# Thiết lập đường dẫn để tìm các module trong ball_action_spotting
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "src"))  # Thêm đường dẫn đến thư mục src

# Import sau khi đã thiết lập đường dẫn
from src.frame_fetchers.nvdec import NvDecFrameFetcher
from src.predictors import MultiDimStackerPredictor
from src.utils import post_processing, get_video_info, frame_index_to_timestamp
import src.action.constants as action_constants
import src.ball_action.constants as ball_action_constants

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
INDEX_SAVE_ZONE = 1


def calculate_frame_indices_for_target_fps(frame_count: int, source_fps: float, target_fps: float = 25.0) -> tuple[list[int], dict[int, int]]:
    duration = frame_count / source_fps
    target_frame_count = int(duration * target_fps)
    original_frame_indices = []
    predict_index2original_index_map = {}
    
    for i in range(target_frame_count):
        time_i = i / target_fps
        original_idx = round(time_i * source_fps)
        if original_idx >= frame_count:
            break
        original_frame_indices.append(original_idx)
        predict_index2original_index_map[i] = original_idx
    
    return original_frame_indices, predict_index2original_index_map

def get_raw_predictions(predictor: MultiDimStackerPredictor,
                                  video_path: Path,
                                  frame_count: int,
                                  source_fps: float,
                                  model_name: str,
                                  batch_size: int,
                                  target_fps: float) -> tuple[list[int], np.ndarray, dict[int, int]]:
    
    frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=predictor.device.index)
    frame_fetcher.num_frames = frame_count

    frame_indices_to_process, original_mapping = calculate_frame_indices_for_target_fps(
        frame_count, source_fps, target_fps)
    
    indexes_generator = predictor.indexes_generator
    
    min_frame_index = indexes_generator.clip_index(0, len(frame_indices_to_process), INDEX_SAVE_ZONE)
    max_frame_index = indexes_generator.clip_index(len(frame_indices_to_process), len(frame_indices_to_process), INDEX_SAVE_ZONE)
    
    frame_index2prediction = dict()
    predict_index2original_frame_index = dict()  
    predictor.reset_buffers()
    
    with tqdm(desc=f"Processing {model_name}", total=len(frame_indices_to_process)) as t:
        i = 0
        predict_idx = 0  
        while i < len(frame_indices_to_process):
            original_indices_batch = frame_indices_to_process[i:i + batch_size]
            batch_frames = []
            valid_original_indices = []
            
            for frame_index in original_indices_batch:
                frame = frame_fetcher.fetch_frame(frame_index)
                if frame is None:
                    logger.info(f"Cannot read frame {frame_index}. Maybe reached the end of video.")
                    break
                batch_frames.append(frame)
                valid_original_indices.append(frame_index)
            
            if not batch_frames:
                break
            
            predict_indices_batch = list(range(predict_idx, predict_idx + len(batch_frames)))
        
            batch_results = predictor.predict_batch(batch_frames, predict_indices_batch)
            
            has_reached_max = False
            for result_idx, (prediction, predict_index) in enumerate(batch_results):
                if predict_index < min_frame_index:
                    continue
                    
                if prediction is not None:
                    frame_index2prediction[predict_index] = prediction.cpu().numpy()
                    
                    if predict_index in original_mapping:
                        original_frame_idx = original_mapping[predict_index]
                        predict_index2original_frame_index[predict_index] = original_frame_idx
                    else:
                        original_frame_idx = valid_original_indices[result_idx] if result_idx < len(valid_original_indices) else predict_index
                        predict_index2original_frame_index[predict_index] = original_frame_idx
                
                if predict_index >= max_frame_index:
                    has_reached_max = True
                    logger.info(f"Reached max_frame_index ({max_frame_index}). Stop processing.")
                    break
            
            t.update(len(batch_frames))
            i += len(batch_frames)
            predict_idx += len(batch_frames)
    
            if has_reached_max:
                break

    predictor.reset_buffers()
    frame_indexes = sorted(frame_index2prediction.keys())
    
    if not frame_indexes:
        return [], np.array([]), {}
    raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
    
    return frame_indexes, raw_predictions, predict_index2original_frame_index

def process_predictions(frame_indexes: list[int], raw_predictions: np.ndarray, class_map: dict, postprocess_params: dict,source_fps: float,model_type: str,predict_index2original_frame_index: dict[int, int]) -> list:

    all_events = []
    for class_name, class_index in class_map.items():
        if class_name == 'EMPTY':
            continue
        action_frame_indexes, confidences = post_processing(frame_indexes, raw_predictions[:, class_index], **postprocess_params)

        for frame_idx, conf in zip(action_frame_indexes, confidences):
            original_frame_idx = predict_index2original_frame_index.get(frame_idx, frame_idx)
            event = {
                "event": class_name,
                "timestamp": frame_index_to_timestamp(original_frame_idx, source_fps),
                "confidence": float(conf),
                "type": model_type
            }
            all_events.append(event)
    return all_events

def predict_on_video(video_path: Path,action_model_path: Path,ball_action_model_path: Path,output_dir: Path,gpu_id: int,batch_size: int,target_fps: float):

    if not video_path.exists():
        logger.error(f"File video does not exist: {video_path}")
        return
    
    video_info = get_video_info(video_path)
    source_fps = video_info['fps']
    num_frames = video_info['frame_count']

    action_predictor = MultiDimStackerPredictor(action_model_path, device=f"cuda:{gpu_id}")
    ball_action_predictor = MultiDimStackerPredictor(ball_action_model_path, device=f"cuda:{gpu_id}")

    logger.info(f"Processing video with {source_fps}fps, targeting {target_fps}fps for predictions")
    
    action_frame_indexes, action_raw_preds, action_predict_index2original = get_raw_predictions(
        action_predictor, video_path, num_frames, source_fps, "Action Model", batch_size, target_fps)

    ball_action_frame_indexes, ball_action_raw_preds, ball_action_predict_index2original = get_raw_predictions(
        ball_action_predictor, video_path, num_frames, source_fps, "Ball-Action Model", batch_size, target_fps)

    all_predictions = []
    
    if len(action_frame_indexes) > 0 and action_raw_preds.size > 0:
        action_events = process_predictions(
            action_frame_indexes,
            action_raw_preds,
            action_constants.class2target,
            action_constants.postprocess_params,
            source_fps,
            "action",
            action_predict_index2original
        )
        all_predictions.extend(action_events)
        logger.info(f"Found {len(action_events)} action events")

    if len(ball_action_frame_indexes) > 0 and ball_action_raw_preds.size > 0:
        ball_action_events = process_predictions(
            ball_action_frame_indexes,
            ball_action_raw_preds,
            ball_action_constants.class2target,
            ball_action_constants.postprocess_params,
            source_fps,
            "ball_action",
            ball_action_predict_index2original
        )
        all_predictions.extend(ball_action_events)
        logger.info(f"Found {len(ball_action_events)} ball-action events")
    
    all_predictions.sort(key=lambda x: x["timestamp"])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_ball_action.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": all_predictions, "UrlLocal": str(video_path)}, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Saved {len(all_predictions)} events to {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Predict Action and Ball-Action on real video.")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--action-model-path", type=Path, required=True)
    parser.add_argument("--ball-action-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--target-fps", type=float, default=25.0)

    args = parser.parse_args()

    predict_on_video(
        args.video_path,
        args.action_model_path,
        args.ball_action_model_path,
        args.output_dir,
        args.gpu_id,
        args.batch_size,
        args.target_fps
    )


if __name__ == "__main__":
    main() 