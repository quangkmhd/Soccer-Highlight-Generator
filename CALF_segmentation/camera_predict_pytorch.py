import argparse
import json
import logging
import os
import time
import tempfile
import sys
from pathlib import Path
import numpy as np
import torch

# Thiết lập đường dẫn để tìm các module trong CALF_segmentation
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "src"))  # Thêm đường dẫn đến thư mục src
sys.path.append(str(current_dir / "Features"))  # Thêm đường dẫn đến thư mục Features

# TensorFlow no longer needed - using PyTorch backend exclusively

# Import sau khi đã thiết lập đường dẫn
from Features.VideoFeatureExtractor_pytorch import VideoFeatureExtractor, PCAReducer
from src.model import Model  

# -----------------------------------------------------------------------------
LABELS = {
    0: "Main camera center", 1: "Close-up player or field referee",
    2: "Main camera left", 3: "Main camera right",
    4: "Goal line technology camera", 5: "Main behind the goal",
    6: "Spider camera", 7: "Close-up side staff",
    8: "Close-up corner", 9: "Close-up behind the goal",
    10: "Inside the goal", 11: "Public", 12: "Other/Unknown"
}

def format_time(frame_idx: int, fps: float) -> str:
    total_sec = frame_idx / fps
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"


def extract_features(video_path: str,
                     feature_path: str,
                     fps: float,
                     backend: str,
                     overwrite: bool,
                     batch_size_feat: int,
                     device: str = "cuda") -> np.ndarray:
    if os.path.exists(feature_path) and not overwrite:
        logging.info("File feature already exists, load directly: %s", feature_path)
        return np.load(feature_path)

    # Use PyTorch VideoFeatureExtractor without PCA (features already 512 dims from layer2)
    extractor = VideoFeatureExtractor(
        feature="ResNET", 
        back_end="PT",  # Force PyTorch backend
        overwrite=overwrite, 
        transform="crop", 
        grabber="opencv", 
        FPS=fps,
        device=device,
        apply_pca=False,  # Disable PCA - features are already 512 dims
        pca_file=None,
        scaler_file=None
    )

    # Extract features (512 dims from ResNet layer2)
    extractor.extractFeatures(video_path, feature_path, overwrite=overwrite)
    return np.load(feature_path)

def run_inference(features: np.ndarray,model_path: str,device: torch.device,fps: float,num_classes_type: int = 13,chunk_size: int = 48,receptive_field: int = 16,num_detections: int = 45) -> np.ndarray:

    # Use fixed input_size=512 since features are extracted from ResNet layer2 (512 dims)
    input_size = 512
    logging.info(f"Using input_size={input_size}, actual feature dimensions: {features.shape[1]}")
    
    # Check if feature dimensions match expected input_size
    if features.shape[1] != input_size:
        raise ValueError(f"Feature dimensions {features.shape[1]} don't match expected input_size {input_size}. ResNet layer2 should output 512 dims.")

    model = Model(
        input_size=input_size,
        num_classes_type=num_classes_type,
        chunk_size=chunk_size,
        receptive_field=receptive_field,
        num_detections=num_detections,
        framerate=int(fps),
    ).to(device)

    logging.info(f"Inference will run on device: {device} (model parameters on {next(model.parameters()).device})")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), chunk_size):
            chunk = features[start:start + chunk_size]
            # pad with 0 to avoid repeating last vector
            if len(chunk) < chunk_size:
                pad_len = chunk_size - len(chunk)
                chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode='constant')
            tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            seg, _ = model(tensor)
            seg = seg.squeeze(0).cpu().numpy()
            if len(chunk) < chunk_size:
                seg = seg[:len(chunk)]
            outputs.append(seg)
    return np.concatenate(outputs, axis=0)


def post_process_and_save(output: np.ndarray,
                           confidence_threshold: float,
                           output_dir: str,
                           video_name: str,
                           fps: float):
    
    preds = output.argmax(axis=1)
    confs = output.max(axis=1)

    events = []
    last_cls = preds[0]
    last_added = 0
    min_gap = int(1 * fps)  # minimum 1 second

    for i, (cls, conf) in enumerate(zip(preds, confs)):
        changed = cls != last_cls
        high_conf = conf >= confidence_threshold
        gap_ok = (i - last_added) >= min_gap
        if ((changed and gap_ok) or (high_conf and gap_ok)):
            events.append({
                "event": LABELS.get(cls, "Unknown"),
                "timestamp": format_time(i, fps),
                "confidence": round(float(conf), 4),
                "type": "camera_change",
            })
            last_cls, last_added = cls, i

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{video_name}_camera.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=4, ensure_ascii=False)
    logging.info("Saved events → %s (%d events)", path, len(events))


def camera_predict_on_video(
    video_path: str,
    model_path: str,
    output_dir: str,
    gpu_id: int = 0,
    fps: float = 2.0,
    backend: str = 'PT',
    batch_size_feat: int = 64,
    confidence_threshold: float = 0.2,
    num_classes_type: int = 13,
    chunk_size: int = 48,
    receptive_field: int = 16,
    num_detections: int = 45,
    overwrite: bool = True
) -> str:
    
    logging.info(f"Running camera prediction on {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device for PyTorch
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    device_str = f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    logging.info(f"Using device: {device}")
    
    # Get video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create temporary directory for features
    with tempfile.TemporaryDirectory() as temp_dir:
        feature_path = os.path.join(temp_dir, f"{video_name}_ResNET_{backend}_layer2_512.npy")
        
        # Extract features
        logging.info("Extracting features...")
        features = extract_features(
            video_path=video_path, 
            feature_path=feature_path, 
            fps=fps, 
            backend=backend, 
            overwrite=overwrite, 
            batch_size_feat=batch_size_feat,

            device=device_str
        )
        
        # Run inference
        logging.info("Running inference...")
        output = run_inference(
            features=features,
            model_path=model_path,
            device=device,
            fps=fps,
            num_classes_type=num_classes_type,
            chunk_size=chunk_size,
            receptive_field=receptive_field,
            num_detections=num_detections
        )
        
        # Post-process and save
        logging.info("Post-processing and saving results...")
        output_file = os.path.join(output_dir, f"{video_name}_camera.json")
        post_process_and_save(
            output=output,
            confidence_threshold=confidence_threshold,
            output_dir=output_dir,
            video_name=video_name,
            fps=fps
        )
    
    logging.info(f"Camera prediction completed successfully. Results saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--model_path', default='/home/qmask_quangnh58/soccer_highlight/CALF-segmentation/model/CALF/model_last.pth.tar')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--backend', default='PT')
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--batch_size_feat', type=int, default=128)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--loglevel', default='INFO')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--confidence_threshold', type=float, default=0.2)
    parser.add_argument('--num_classes_type', type=int, default=13)
    parser.add_argument('--chunk_size', type=int, default=48)
    parser.add_argument('--receptive_field', type=int, default=16)
    parser.add_argument('--num_detections', type=int, default=45)

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), None),
                        format="%(asctime)s [%(levelname)-5.5s] %(message)s")

    # Call the new function with all the arguments
    camera_predict_on_video(
        video_path=args.video_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        gpu_id=args.GPU,
        fps=args.fps,
        backend=args.backend,
        batch_size_feat=args.batch_size_feat,
        confidence_threshold=args.confidence_threshold,
        num_classes_type=args.num_classes_type,
        chunk_size=args.chunk_size,
        receptive_field=args.receptive_field,
        num_detections=args.num_detections,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    main()
