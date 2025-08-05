import abc
from typing import Type

import torch
import torch.nn.functional as F


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    frames = frames.to(torch.float32) / 255.0
    return frames


def pad_to_frames(frames: torch.Tensor,
                  size: tuple[int, int],
                  pad_mode: str = "constant",
                  fill_value: int = 0) -> torch.Tensor:
    height, width = frames.shape[-2:]
    height_pad = size[1] - height
    width_pad = size[0] - width
    assert height_pad >= 0 and width_pad >= 0

    top_height_pad: int = height_pad // 2
    bottom_height_pad: int = height_pad - top_height_pad
    left_width_pad: int = width_pad // 2
    right_width_pad: int = width_pad - left_width_pad
    frames = torch.nn.functional.pad(
        frames,
        [left_width_pad, right_width_pad, top_height_pad, bottom_height_pad],
        mode=pad_mode,
        value=fill_value,
    )
    return frames


class FramesProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        pass


class PadNormalizeFramesProcessor(FramesProcessor):
    def __init__(self,
                 size: tuple[int, int],
                 pad_mode: str = "constant",
                 fill_value: int = 0):
        self.size = size
        self.pad_mode = pad_mode
        self.fill_value = fill_value

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        frames = pad_to_frames(frames, self.size,
                               pad_mode=self.pad_mode,
                               fill_value=self.fill_value)
        frames = normalize_frames(frames)
        return frames


class ResizeAndPadNormalizeFramesProcessor(FramesProcessor):
    """
    Resizes the frame to fit within the target size while maintaining aspect ratio,
    then pads it to the exact size before normalization.
    """
    def __init__(self,
                 size: tuple[int, int],
                 pad_mode: str = "constant",
                 fill_value: int = 0):
        self.size = size  # (width, height)
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self._printed_size_info = False

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames shape: (B, C, H, W)
        _b, _c, h, w = frames.shape
        target_w, target_h = self.size

        # Calculate new size to maintain aspect ratio
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)

        new_h, new_w = int(h * scale), int(w * scale)

        if not self._printed_size_info:
            self._printed_size_info = True

        # Resize using bilinear interpolation
        # Note: interpolate works on float tensors
        resized_frames = F.interpolate(
            frames.float(),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).to(frames.dtype)  # Convert back to original dtype (e.g., uint8)

        # Pad the resized frames to the target size
        padded_frames = pad_to_frames(resized_frames, self.size,
                                      pad_mode=self.pad_mode,
                                      fill_value=self.fill_value)

        # Normalize the final frames
        normalized_frames = normalize_frames(padded_frames)

        return normalized_frames


_FRAME_PROCESSOR_REGISTRY: dict[str, Type[FramesProcessor]] = dict(
    pad_normalize=PadNormalizeFramesProcessor,
    resize_and_pad_normalize=ResizeAndPadNormalizeFramesProcessor,
)


def get_frames_processor(name: str, processor_params: dict) -> FramesProcessor:
    assert name in _FRAME_PROCESSOR_REGISTRY
    return _FRAME_PROCESSOR_REGISTRY[name](**processor_params)
