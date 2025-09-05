
import sys
import os
import argparse
from pathlib import Path
import warnings
import logging

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "ball_action_spotting"))
sys.path.append(str(current_dir / "CALF_segmentation"))
sys.path.append(str(current_dir / "create_clips"))
sys.path.append(str(current_dir / "rules"))

# Import sau khi đã thiết lập đường dẫn
from inference.parallel_inference import run_inference_pipeline_parallel

def main():
    parser = argparse.ArgumentParser(description="Soccer Action Spotting - Simple Interface with Parallel Processing")
    parser.add_argument("video", type=str, nargs="?", help="Path to the video file")
    parser.add_argument("--config", type=str, default="inference/inference_config.yaml", help="Path to the inference configuration file")
    args = parser.parse_args()
    
    if not args.video:
        args.video = input("Nhập đường dẫn đến file video: ")
    
    video_path = Path(args.video)
    config_path = Path(args.config)
    
    if not video_path.exists():
        print(f"Lỗi: File video không tồn tại: {video_path}")
        return 1
    
    if not config_path.exists():
        print(f"Lỗi: File config không tồn tại: {config_path}")
        return 1
        
    print(f"Đang xử lý video với parallel processing: {video_path}")
    print(f"Sử dụng config: {config_path}")
    
    # Set multiprocessing start method for compatibility
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    run_inference_pipeline_parallel(video_path, config_path)

if __name__ == "__main__":
    main()
