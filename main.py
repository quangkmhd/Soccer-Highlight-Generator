import sys
import os
import argparse
from pathlib import Path

# Thêm đường dẫn của các module vào sys.path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "ball_action_spotting"))
sys.path.append(str(current_dir / "CALF_segmentation"))
sys.path.append(str(current_dir / "create_clips"))
sys.path.append(str(current_dir / "rules"))

# Import sau khi đã thiết lập đường dẫn
from inference.inference import run_inference_pipeline

def main():
    parser = argparse.ArgumentParser(description="Soccer Action Spotting - Simple Interface")
    parser.add_argument("video", type=str, nargs="?", help="Path to the video file")
    args = parser.parse_args()
    
    if not args.video:
        args.video = input("Nhập đường dẫn đến file video: ")
    
    video_path = Path(args.video)

        
    print(f"Đang xử lý video: {video_path}")
    
    # Gọi hàm với đối tượng Path
    output_dir = run_inference_pipeline(video_path)
    print(f"Xử lý hoàn tất. Kết quả được lưu tại: {output_dir}")
if __name__ == "__main__":
    sys.exit(main())
