import re
import time
import random
from pathlib import Path

import numpy as np
import cv2  # type: ignore
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from datetime import timedelta
from torch import nn


def get_video_info(video_path: str | Path) -> dict[str, int | float]:
    video = cv2.VideoCapture(str(video_path))
    video_info = dict(
        frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(video.get(cv2.CAP_PROP_FPS)),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return video_info


def set_random_seed(index: int):
    seed = int(time.time() * 1000.0) + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def get_best_model_path(dir_path, return_score=False, more_better=True):
    dir_path = Path(dir_path)
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = float(score.group(0)[1:-4])
            model_scores.append((model_path, score))

    if not model_scores:
        if return_score:
            return None, -np.inf
        else:
            return None

    model_score = sorted(model_scores, key=lambda x: x[1], reverse=more_better)
    best_model_path = model_score[0][0]
    if return_score:
        best_score = model_score[0][1]
        return best_model_path, best_score
    else:
        return best_model_path


def post_processing(frame_indexes: list[int],
                         predictions: np.ndarray,
                         gauss_sigma: float,
                         height: float,
                         distance: int) -> tuple[list[int], list[float]]:

    if len(frame_indexes) == 0 or len(predictions) == 0:
        return [], []
    
    predictions = gaussian_filter(predictions, gauss_sigma)
    peaks, _ = find_peaks(predictions, height=height, distance=distance)
    
    # Kiểm tra bounds để tránh index out of range
    valid_peaks = [peak for peak in peaks if 0 <= peak < len(frame_indexes)]
    
    confidences = predictions[valid_peaks].tolist()
    # FIX: Sử dụng frame_indexes[peaks] thay vì peaks + frame_indexes[0]
    action_frame_indexes = [frame_indexes[peak] for peak in valid_peaks]
    return action_frame_indexes, confidences


def load_weights_from_pretrain(nn_module: nn.Module, pretrain_nn_module: nn.Module):
    state_dict = nn_module.state_dict()
    pretrain_state_dict = pretrain_nn_module.state_dict()

    assert state_dict.keys() == pretrain_state_dict.keys()

    load_state_dict = dict()
    for name, pretrain_weights in pretrain_state_dict.items():
        weights = state_dict[name]
        if weights.shape == pretrain_weights.shape:
            load_state_dict[name] = pretrain_weights
        else:
            print(f"Layer '{name}' has different shape in pretrain "
                  f"{weights.shape} != {pretrain_weights.shape}. Skip loading.")
            load_state_dict[name] = weights

    nn_module.load_state_dict(load_state_dict)


def get_lr(base_lr: float, batch_size: int, base_batch_size: int = 4) -> float:
    return base_lr * (batch_size / base_batch_size)


def frame_index_to_timestamp(frame_index: int, fps: float) -> str:
    total_seconds = frame_index / fps
    td = timedelta(seconds=total_seconds)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"