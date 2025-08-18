
import sys
import os
import argparse
from pathlib import Path
import warnings
import logging

# Silence TensorFlow/absl logs and all Python warnings as early as possible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 1=filter INFO, 2=+WARN, 3=+ERROR
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
warnings.filterwarnings("ignore")

# Reduce Python logging noise from all libraries
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for noisy_logger in [
    "tensorflow",
    "absl",
    "keras",
    "matplotlib",
    "PIL",
    "numba",
    "sklearn",
    "urllib3",
    "werkzeug",
    "torch",
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

try:
    # Further quiet absl pre-init stderr warnings if available
    import absl.logging as absl_logging  # type: ignore

    absl_logging.set_verbosity("error")
    absl_logging.set_stderrthreshold("error")
    if hasattr(absl_logging, "_warn_preinit_stderr"):
        absl_logging._warn_preinit_stderr = 0  # type: ignore[attr-defined]
except Exception:
    pass

# Thêm đường dẫn của các module vào sys.path
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
    
    try:
        # Gọi hàm parallel processing với đối tượng Path
        output_dir = run_inference_pipeline_parallel(video_path, config_path)
        print(f"Xử lý hoàn tất với parallel processing. Kết quả được lưu tại: {output_dir}")
        return 0
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
        return 1
if __name__ == "__main__":
    sys.exit(main())
