#!/usr/bin/env python3
"""
FrameNVDEC: GPU-accelerated video frame loader using NVIDIA Video Codec SDK
Compatible with SoccerNet FrameCV interface
"""

import os
import cv2
import numpy as np
import logging
import imutils
from typing import Optional, Generator

# Optional dependencies
try:
    import torch
except ImportError:
    torch = None
    logging.warning("PyTorch not available; NVDEC will be disabled.")

try:
    import PyNvVideoCodec as nvc
except ImportError:
    nvc = None
    logging.warning("PyNvVideoCodec not available; NVDEC will be disabled.")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import moviepy.editor
except ImportError:
    moviepy = None
    logging.warning("MoviePy not available; will use OpenCV for duration.")

def getDuration(video_path):
    """Get the duration (in seconds) for a video - compatible with FrameCV"""
    if moviepy is not None:
        try:
            return moviepy.editor.VideoFileClip(video_path).duration
        except:
            pass

    # Fallback to OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return 0.0


class FrameNVDEC:
    """
    NVDEC-based frame loader for GPU-accelerated video decoding
    Compatible with SoccerNet FrameCV interface
    """
    
    def __init__(
        self,
        video_path: str,
        FPS: float = 2.0,
        transform: str = "crop",
        start: Optional[float] = None,
        duration: Optional[float] = None,
        gpu_id: int = 0
    ):
        """
        Initialize NVDEC frame loader - Compatible with FrameCV interface

        Args:
            video_path: Path to video file (compatible with FrameCV)
            FPS: Target frames per second for extraction
            transform: Transform type ("crop", "resize", "resize256crop224")
            start: Start time in seconds (None for beginning)
            duration: Duration in seconds (None for full video)
            gpu_id: GPU device ID
        """
        if nvc is None or torch is None:
            raise RuntimeError("NVDEC dependencies missing (PyNvCodec or PyTorch).")

        self.path = video_path
        self.FPS = FPS
        self.transform = transform
        self.start = start
        self.duration = duration
        self.gpu_id = gpu_id

        # Probe native FPS & frame count via OpenCV (compatible with FrameCV)
        cap = cv2.VideoCapture(self.path)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if fps_video <= 0:
            fps_video = 25.0
        self.fps_video = fps_video  # Compatible with FrameCV attribute name

        # Compute frame range
        start_frame = int(self.start * fps_video) if self.start is not None else 0
        if self.duration is not None:
            end_frame = int((self.start or 0) * fps_video + self.duration * fps_video)
        else:
            end_frame = frame_count if frame_count > 0 else None

        self.start_frame = start_frame
        self.end_frame = end_frame

        # Compute stride for downsampling
        self.stride = max(1, round(fps_video / max(1e-6, FPS)))

        # Progress bar
        if frame_count > 0 and end_frame is not None:
            total_raw = max(0, end_frame - start_frame)
            selected_total = total_raw // self.stride + (1 if (total_raw % self.stride) > 0 else 0)
        else:
            selected_total = None
        self.pbar = tqdm(total=selected_total, desc=f"NVDEC: {os.path.basename(self.path)}") if tqdm and selected_total else None

        # NVDEC setup
        self.demuxer = nvc.CreateDemuxer(str(self.path))
        try:
            width = self.demuxer.Width()
            height = self.demuxer.Height()
            codec = self.demuxer.GetNvCodecId()
        except Exception:
            width = height = None
            codec = None

        self.width = width
        self.height = height
        self.decoder = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=codec if codec is not None else 0,
            usedevicememory=True,
        )

        # Compute time_second for compatibility with FrameCV
        if self.duration is not None:
            self.time_second = self.duration
        elif frame_count > 0 and fps_video > 0:
            self.time_second = frame_count / fps_video
        else:
            self.time_second = 0.0

        # Store frame count for compatibility
        self.numframe = frame_count

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def cleanup(self):
        """Explicitly clean up GPU memory and NVDEC resources."""
        try:
            # Close progress bar if exists
            if hasattr(self, 'pbar') and self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            
            # Clean up NVDEC decoder
            if hasattr(self, 'decoder') and self.decoder is not None:
                del self.decoder
                self.decoder = None
            
            # Clean up NVDEC demuxer  
            if hasattr(self, 'demuxer') and self.demuxer is not None:
                del self.demuxer
                self.demuxer = None
                
            # Force GPU memory cleanup
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.warning("Error during FrameNVDEC cleanup: %s", e)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield frames as uint8 numpy arrays (224x224 RGB).
        Generator method for memory efficiency.
        """
        frame_index = -1
        for packet in self.demuxer:
            try:
                surfaces = self.decoder.Decode(packet)
            except Exception as e:
                logging.warning("NVDEC: skipping bad packet: %s", e)
                continue

            for surface in surfaces:
                frame_index += 1
                # Frame range filter
                if frame_index < self.start_frame:
                    continue
                if self.end_frame is not None and frame_index >= self.end_frame:
                    break

                # Downsampling stride
                if (frame_index - self.start_frame) % self.stride != 0:
                    continue

                try:
                    # GPU tensor from NVDEC surface
                    t_gpu = torch.from_dlpack(surface)  # (H*1.5, W) for NV12

                    # Extract Y plane (grayscale)
                    if t_gpu.dim() == 2 and self.height and self.width:
                        t_gpu = t_gpu[:self.height, :]
                    elif t_gpu.dim() == 3:
                        t_gpu = t_gpu[0]
                    if t_gpu.dim() != 2:
                        t_gpu = t_gpu.view(self.height, self.width)

                    # Apply transform similar to FrameCV
                    frame_processed = self._apply_transform(t_gpu)

                    # Convert to BGR format (compatible with FrameCV)
                    if frame_processed.shape[2] == 3:  # RGB
                        bgr = cv2.cvtColor(frame_processed, cv2.COLOR_RGB2BGR)
                    else:  # Grayscale
                        bgr = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)

                    # Explicit GPU memory cleanup
                    del t_gpu
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if self.pbar:
                        self.pbar.update(1)

                    yield bgr

                except Exception as e:
                    logging.warning("NVDEC: failed to convert surface: %s", e)
                    continue

            if self.end_frame is not None and frame_index >= self.end_frame:
                break

        # Final cleanup
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    def _apply_transform(self, t_gpu):
        """
        Apply transform to GPU tensor similar to FrameCV

        Args:
            t_gpu: GPU tensor from NVDEC surface

        Returns:
            Processed frame as numpy array (224x224x3)
        """
        # Normalize to [0,1] float for GPU processing
        y_gpu = t_gpu.to(dtype=torch.float32) / 255.0
        y_gpu = y_gpu.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        if self.transform == "resize256crop224":
            # Resize to height 256 keeping aspect ratio, then center crop to 224x224
            h, w = y_gpu.shape[2], y_gpu.shape[3]
            new_h = 256
            new_w = int(w * new_h / h)

            # Resize
            resized = torch.nn.functional.interpolate(
                y_gpu, size=(new_h, new_w), mode='bilinear', align_corners=False
            )

            # Center crop to 224x224
            off_h = (new_h - 224) // 2
            off_w = (new_w - 224) // 2
            cropped = resized[:, :, off_h:off_h+224, off_w:off_w+224]

        elif self.transform == "crop":
            # Resize to height 224 keeping aspect ratio, then center crop width to 224
            h, w = y_gpu.shape[2], y_gpu.shape[3]
            new_h = 224
            new_w = int(w * new_h / h)

            # Resize
            resized = torch.nn.functional.interpolate(
                y_gpu, size=(new_h, new_w), mode='bilinear', align_corners=False
            )

            # Center crop width to 224
            if new_w > 224:
                off_w = (new_w - 224) // 2
                cropped = resized[:, :, :, off_w:off_w+224]
            else:
                cropped = resized

        elif self.transform == "resize":
            # Direct resize to 224x224 (lose aspect ratio)
            cropped = torch.nn.functional.interpolate(
                y_gpu, size=(224, 224), mode='bilinear', align_corners=False
            )

        else:
            # Default: resize to 224x224
            cropped = torch.nn.functional.interpolate(
                y_gpu, size=(224, 224), mode='bilinear', align_corners=False
            )

        # Convert back to uint8 and CPU
        frame_cpu = (cropped.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # Convert grayscale to RGB by duplicating channels
        if frame_cpu.ndim == 2:
            frame_rgb = np.stack([frame_cpu, frame_cpu, frame_cpu], axis=2)
        else:
            frame_rgb = frame_cpu

        return frame_rgb

    @property
    def frames_list(self):
        """
        Convert generator to list for compatibility with existing code.
        Note: This loads all frames into memory, use with caution.
        """
        return list(self.frames())

    @property
    def frames_array(self):
        """
        Property to mimic FrameCV.frames attribute
        Returns numpy array of all frames (loads into memory)
        """
        if not hasattr(self, '_frames_cache'):
            self._frames_cache = np.array(list(self.frames()))
        return self._frames_cache


def is_nvdec_available() -> bool:
    """Check if NVDEC dependencies are available"""
    return nvc is not None and torch is not None


def create_frame_loader(video_path: str, grabber: str = "opencv", **kwargs):
    """
    Factory function to create appropriate frame loader

    Args:
        video_path: Path to video file (compatible with FrameCV)
        grabber: "opencv", "skvideo", or "nvdec"
        **kwargs: Additional arguments for frame loader

    Returns:
        Frame loader instance (FrameCV, Frame, or FrameNVDEC)
    """
    from SoccerNet.DataLoader import Frame, FrameCV

    if grabber.lower() == "nvdec":
        if is_nvdec_available():
            return FrameNVDEC(video_path, **kwargs)
        else:
            logging.warning("NVDEC requested but not available; falling back to OpenCV.")
            return FrameCV(video_path, **kwargs)
    elif grabber.lower() == "opencv":
        return FrameCV(video_path, **kwargs)
    elif grabber.lower() == "skvideo":
        return Frame(video_path, **kwargs)
    else:
        raise ValueError(f"Unsupported grabber: {grabber}")


def get_frames_from_loader(video_loader):
    """
    Get frames from any frame loader (FrameCV, Frame, or FrameNVDEC)
    Handles different interfaces uniformly

    Args:
        video_loader: Frame loader instance

    Returns:
        List of frames (numpy arrays)
    """
    try:
        if isinstance(video_loader, FrameNVDEC):
            # FrameNVDEC: use generator method with context manager for cleanup
            with video_loader:
                return list(video_loader.frames())
        elif hasattr(video_loader, 'frames') and callable(video_loader.frames):
            # Other loaders with callable frames method
            return list(video_loader.frames())
        else:
            # Traditional FrameCV/Frame with frames attribute (numpy array)
            frames = video_loader.frames if hasattr(video_loader, 'frames') else []
            # Convert numpy array to list if needed
            if isinstance(frames, np.ndarray):
                return [frames[i] for i in range(len(frames))]
            return frames
    except Exception as e:
        # Ensure cleanup even if error occurs
        if hasattr(video_loader, 'cleanup'):
            video_loader.cleanup()
        raise e
