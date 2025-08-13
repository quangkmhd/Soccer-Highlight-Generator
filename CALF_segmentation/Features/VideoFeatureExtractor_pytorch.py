import argparse
import os
import logging
from typing import Optional, Generator

import numpy as np
import cv2
import pickle as pkl
from SoccerNet.DataLoader import Frame, FrameCV

# PyTorch dependencies
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.models import ResNet152_Weights
except Exception:
    torch = None
    logging.warning("PyTorch not available; PyTorch backend will be disabled.")

# Import NVDEC frame loader
try:
    from FrameNVDEC import FrameNVDEC, is_nvdec_available, create_frame_loader, get_frames_from_loader
    NVDEC_AVAILABLE = True
except ImportError:
    FrameNVDEC = None
    is_nvdec_available = lambda: False
    create_frame_loader = None
    get_frames_from_loader = None
    NVDEC_AVAILABLE = False
    logging.warning("FrameNVDEC not available; NVDEC grabber will be disabled.")

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# FrameNVDEC class moved to separate file FrameNVDEC.py


class VideoFeatureExtractor:
    def __init__(self,
                 feature: str = "ResNET",
                 back_end: str = "PT",
                 overwrite: bool = False,
                 transform: str = "crop",
                 grabber: str = "opencv",
                 FPS: float = 2.0,
                 split: str = "all",
                 device: str = "cuda",
                 apply_pca: bool = True,
                 pca_file: str = None,
                 scaler_file: str = None):

        self.feature = feature
        self.back_end = back_end
        self.verbose = True
        self.transform = transform
        self.overwrite = overwrite
        self.grabber = grabber
        self.FPS = FPS
        self.split = split
        self.apply_pca = apply_pca
        # Initialize PCA reducer only if requested or files provided
        self.pca_reducer = PCAReducer(pca_file=pca_file, scaler_file=scaler_file) if (apply_pca or pca_file is not None or scaler_file is not None) else None

        if self.back_end == "TF2":
            try:
                import tensorflow as tf
                import tensorflow.keras as keras
                from tensorflow.keras import Model as TFModel
            except Exception:
                tf = None
                keras = None
                TFModel = None
                logging.warning("TensorFlow not available; TF2 backend will fail.")

            if tf is None or keras is None or TFModel is None:
                raise ImportError("TensorFlow and Keras are required for the TF2 backend.")

            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logging.warning("TF GPU config error: %s", e)

            # Resolve weights path: prefer host path, fallback to container path; else use 'imagenet'
            weights_candidates = [
                '/home/qmask_quangnh58/soccer_highlight/CALF_segmentation/Features/resnet152_weights_tf_dim_ordering_tf_kernels.h5',
                '/app/CALF_segmentation/Features/resnet152_weights_tf_dim_ordering_tf_kernels.h5',
            ]

            resolved_weights_path = None
            for candidate in weights_candidates:
                if os.path.exists(candidate):
                    resolved_weights_path = candidate
                    break

            if resolved_weights_path is None:
                logging.warning(
                    "ResNet152 weights not found at expected paths; falling back to 'imagenet'."
                )
                weights_arg = 'imagenet'
            else:
                logging.info("Using ResNet152 weights from: %s", resolved_weights_path)
                weights_arg = resolved_weights_path
            # keep for potential CPU rebuild
            self._weights_arg = weights_arg

            try:
                base_model = keras.applications.resnet.ResNet152(
                    include_top=True,
                    weights=weights_arg,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000)
            except Exception as e:
                logging.warning("ResNet152 init failed on default device (%s). Retrying on CPU...", e)
                with tf.device('/CPU:0'):
                    base_model = keras.applications.resnet.ResNet152(
                        include_top=True,
                        weights=weights_arg,
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        classes=1000)
                self.model = TFModel(base_model.input,
                                     outputs=[base_model.get_layer("avg_pool").output])
                self.model.trainable = False

        elif self.back_end == "PT":
            if models is None or transforms is None:
                raise ImportError("PyTorch/Torchvision is required for the PT backend.")
            # Respect requested device if provided
            try:
                self.device = torch.device(device) if isinstance(device, str) else device
            except Exception:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Base model with ImageNet weights
            self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

            # Build a truncated model up to layer2 to obtain 512-dim features
            self.model_layer2 = nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu,
                self.model.maxpool,
                self.model.layer1,
                self.model.layer2,
            ).to(self.device)
            self.model_layer2.eval()

            # Preprocessing consistent with ImageNet (RGB)
            self.transform_PT = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported backend: {self.back_end}")

    def preprocess_frames(self, frames):
        """Convert preprocessed frames to PyTorch tensor format (frames already resized by process_frames)"""
        processed_frames = []
        for frame in frames:
            # Convert BGR (cv2) to RGB before transforms to match TF2 preprocessing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.transform_PT(rgb_frame)
            processed_frames.append(processed_frame)
        # Stack into batch tensor
        batch_tensor = torch.stack(processed_frames)
        return batch_tensor

    def _predict_with_fallback(self, frames_tensor, batch_size: int = 64):
        """Run prediction; on GPU failure, retry on CPU transparently."""
        try:
            features_list = []
            with torch.no_grad():
                for i in range(0, frames_tensor.shape[0], batch_size):
                    batch = frames_tensor[i:i+batch_size].to(self.device)
                    # Forward through truncated model up to layer2
                    feat_map = self.model_layer2(batch)
                    # Global average pooling to obtain 512-dim features
                    pooled = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
                    batch_features = pooled.view(pooled.shape[0], -1)
                    features_list.append(batch_features.cpu().numpy())
            return np.vstack(features_list)
        except Exception as e:
            logging.warning("Prediction on %s failed (%s). Retrying on CPU...", self.device, e)
            try:
                # Rebuild truncated model on CPU
                self.model_layer2 = self.model_layer2.cpu()
                self.device = torch.device("cpu")
                features_list = []
                with torch.no_grad():
                    for i in range(0, frames_tensor.shape[0], batch_size):
                        batch = frames_tensor[i:i+batch_size].to(self.device)
                        feat_map = self.model_layer2(batch)
                        pooled = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
                        batch_features = pooled.view(pooled.shape[0], -1)
                        features_list.append(batch_features.cpu().numpy())
                return np.vstack(features_list)
            except Exception as e2:
                logging.error("Prediction failed after CPU fallback: %s", e2)
                raise

    def resize_with_padding(self, frame, target_size=(224, 224)):
        h, w = frame.shape[:2]
        target_h, target_w = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return canvas

    def process_frames(self, frames, target_size=(224, 224)):
        processed = []
        for f in frames:
            processed.append(self.resize_with_padding(f, target_size))
        return np.array(processed)

    def extractFeatures(self, path_video_input, path_features_output,
                        start=None, duration=None, overwrite=False):
        logging.info(f"Extracting features for video {path_video_input}")
        if os.path.exists(path_features_output) and not overwrite:
            logging.info("Features already exist, use overwrite=True to overwrite.")
            return
        logging.info("FrameCV...")
        print(self.grabber)
        # Create video loader using factory function
        if NVDEC_AVAILABLE and create_frame_loader:
            videoLoader = create_frame_loader(path_video_input, grabber=self.grabber,
                                            FPS=self.FPS, transform=self.transform,
                                            start=start, duration=duration)
        else:
            # Fallback to traditional loaders
            if self.grabber == "skvideo":
                videoLoader = Frame(path_video_input, FPS=self.FPS,
                                    transform=self.transform, start=start, duration=duration)
            elif self.grabber == "opencv":
                videoLoader = FrameCV(path_video_input, FPS=self.FPS,
                                      transform=self.transform, start=start, duration=duration)
            else:
                logging.warning(f"Grabber {self.grabber} not supported without NVDEC; falling back to OpenCV.")
                videoLoader = FrameCV(path_video_input, FPS=self.FPS,
                                      transform=self.transform, start=start, duration=duration)

        logging.info("process_frames...")

        # Get frames using unified interface
        try:
            if NVDEC_AVAILABLE and get_frames_from_loader:
                raw_frames = get_frames_from_loader(videoLoader)
            else:
                # Traditional interface
                raw_frames = videoLoader.frames if hasattr(videoLoader, 'frames') else []
        except Exception as e:
            # Ensure cleanup even if error occurs
            if hasattr(videoLoader, 'cleanup'):
                videoLoader.cleanup()
            raise e
            
        # Process frames (resize with padding) - same as TF version
        processed = self.process_frames(raw_frames, target_size=(224, 224))
        
        # Apply PyTorch preprocessing
        frames_tensor = self.preprocess_frames(processed)
        logging.info("Predicting...")
        features = self._predict_with_fallback(frames_tensor, batch_size=64)
        
        # Apply PCA reduction if enabled
        if self.apply_pca and self.pca_reducer is not None:
            original_dim = features.shape[1]
            pca_applied = False
            
            # Apply average centering
            if self.pca_reducer.average is not None:
                logging.info("Applying feature centering...")
                features = features - self.pca_reducer.average
                
            # Apply PCA transformation
            if self.pca_reducer.pca is not None:
                logging.info("Applying PCA reduction...")
                try:
                    if isinstance(self.pca_reducer.pca, dict):
                        # Our simplified dictionary format - manual transformation
                        components = self.pca_reducer.pca['components_']
                        features = np.dot(features, components.T)
                        pca_applied = True
                        logging.info(f"Features reduced from {original_dim} to {features.shape[1]} dimensions (dict format)")
                    else:
                        # Standard scikit-learn PCA object
                        features = self.pca_reducer.pca.transform(features)
                        pca_applied = True
                        logging.info(f"Features reduced from {original_dim} to {features.shape[1]} dimensions (sklearn format)")
                except Exception as e:
                    logging.warning(f"PCA transformation failed: {e}")
                    logging.warning("Continuing with raw features")
            
            if not pca_applied and self.pca_reducer.average is None:
                logging.warning("No PCA or centering applied - using raw features")
        
        np.save(path_features_output, features)

        logging.info(f"Saved features to {path_features_output} with shape {features.shape}")


class PCAReducer:
    def __init__(self, pca_file=None, scaler_file=None):
        self.pca_file = pca_file
        self.scaler_file = scaler_file
        self.loadPCA()

    def loadPCA(self):
        self.pca = None
        if self.pca_file is not None and os.path.exists(self.pca_file):
            try:
                with open(self.pca_file, "rb") as fobj:
                    pca_data = pkl.load(fobj)
                
                # Handle both scikit-learn PCA objects and our simplified dictionary format
                if isinstance(pca_data, dict):
                    # Our simplified dictionary format
                    self.pca = pca_data
                    logging.info(f"Loaded simplified PCA with {pca_data['n_components_']} components")
                elif hasattr(pca_data, 'components_'):
                    # Standard scikit-learn PCA object
                    self.pca = pca_data
                    logging.info(f"Loaded scikit-learn PCA with {pca_data.n_components_} components")
                else:
                    logging.warning(f"Unknown PCA format in {self.pca_file}")
                    self.pca = None
                    
            except Exception as e:
                logging.warning(f"Failed to load PCA file {self.pca_file}: {e}")
                logging.warning("PCA reduction will be disabled")
                self.pca = None
        elif self.pca_file is not None:
            logging.warning(f"PCA file not found: {self.pca_file}")
            
        self.average = None
        if self.scaler_file is not None and os.path.exists(self.scaler_file):
            try:
                with open(self.scaler_file, "rb") as fobj:
                    self.average = pkl.load(fobj)
                logging.info(f"Successfully loaded scaler from {self.scaler_file}")
            except Exception as e:
                logging.warning(f"Failed to load scaler file {self.scaler_file}: {e}")
                logging.warning("Feature centering will be disabled")
                self.average = None
        elif self.scaler_file is not None:
            logging.warning(f"Scaler file not found: {self.scaler_file}")

    def reduceFeatures(self, input_features, output_features, overwrite=False):
        logging.info(f"Reducing features {input_features}")
        if os.path.exists(output_features) and not overwrite:
            logging.info("Features already exist, use overwrite=True to overwrite.")
            return
        feat = np.load(input_features)
        if self.average is not None:
            feat = feat - self.average
        if self.pca is not None:
            if isinstance(self.pca, dict):
                # Our simplified dictionary format - manual transformation
                components = self.pca['components_']
                feat = np.dot(feat, components.T)
            else:
                # Standard scikit-learn PCA object
                feat = self.pca.transform(feat)
        np.save(output_features, feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature from a video (PyTorch backend).')

    parser.add_argument('--path_video', type=str, required=True,
                        help="Path of the Input Video")
    parser.add_argument('--path_features', type=str, required=True,
                        help="Path of the Output Features")
    parser.add_argument('--start', type=float, default=None,
                        help="time of the video to start extracting features [default:None]")
    parser.add_argument('--duration', type=float, default=None,
                        help="duration of the video before finishing extracting features [default:None]")
    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features.")

    parser.add_argument('--GPU', type=int, default=0,
                        help="ID of the GPU to use [default:0]")
    parser.add_argument('--loglevel', type=str, default="INFO",
                        help="loglevel for logging [default:INFO]")

    parser.add_argument('--back_end', type=str, default="PT",
                        help="PyTorch backend [default:PT]")
    parser.add_argument('--features', type=str, default="ResNET",
                        help="ResNET or R25D [default:ResNET]")
    parser.add_argument('--transform', type=str, default="crop",
                        help="crop or resize? [default:crop]")
    parser.add_argument('--video', type=str, default="LQ",
                        help="LQ or HQ? [default:LQ]")
    parser.add_argument('--grabber', type=str, default="opencv",
                        help="skvideo or opencv? [default:opencv]")
    parser.add_argument('--FPS', type=float, default=2.0,
                        help="FPS for the features [default:2.0]")

    parser.add_argument('--apply_pca', action="store_true", default=True,
                        help="Apply PCA reduction to features [default:True]")
    parser.add_argument('--no_pca', action="store_true",
                        help="Disable PCA reduction")
    parser.add_argument('--PCA', type=str, default=None,
                        help="Path to PCA components pickle file [default: auto-detect in Features dir]")
    parser.add_argument('--PCA_scaler', type=str, default=None,
                        help="Path to PCA scaler pickle file [default: auto-detect in Features dir]")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), None),
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.StreamHandler()])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
        device = f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    # Handle PCA configuration
    apply_pca = args.apply_pca and not args.no_pca

    myFeatureExtractor = VideoFeatureExtractor(
        feature=args.features,
        back_end=args.back_end,
        transform=args.transform,
        grabber=args.grabber,
        FPS=args.FPS,
        device=device,
        apply_pca=apply_pca,
        pca_file=args.PCA,
        scaler_file=args.PCA_scaler)

    myFeatureExtractor.extractFeatures(path_video_input=args.path_video,
                                       path_features_output=args.path_features,
                                       start=args.start,
                                       duration=args.duration,
                                       overwrite=args.overwrite)

    logging.info("Feature extraction completed successfully!")
