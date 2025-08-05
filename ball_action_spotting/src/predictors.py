from pathlib import Path
from itertools import islice
from typing import Optional, Iterable

import torch
from kornia.geometry.transform import hflip

import argus

from src.indexes import StackIndexesGenerator
from src.frames import get_frames_processor


def batched(iterable: Iterable, size: int):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, size)):
        yield batch


class MultiDimStackerPredictor:
    def __init__(self, model_path: Path, device: str = "cuda:0", tta: bool = False):
        self.model = argus.load_model(model_path, device=device, optimizer=None, loss=None)
        self.model.eval()
        self.device = self.model.device
        self.tta = tta
        assert self.model.params["nn_module"][0] == "multidim_stacker"
        
        # Always use the flexible resizing processor for prediction
        name, params = self.model.params["frames_processor"]
        self.frames_processor = get_frames_processor("resize_and_pad_normalize", params)

        self.frame_stack_size = self.model.params["frame_stack_size"]
        self.frame_stack_step = self.model.params["frame_stack_step"]
        self.indexes_generator = StackIndexesGenerator(self.frame_stack_size,
                                                       self.frame_stack_step)
        self.model_stack_size = self.model.params["nn_module"][1]["stack_size"]

        self._frame_index2frame: dict[int, torch.Tensor] = dict()
        self._stack_indexes2features: dict[tuple[int], torch.Tensor] = dict()
        self._predict_offset: int = self.indexes_generator.make_stack_indexes(0)[-1]

    def reset_buffers(self):
        self._frame_index2frame = dict()
        self._stack_indexes2features = dict()

    def _clear_old(self, minimum_index: int):
        for index in list(self._frame_index2frame.keys()):
            if index < minimum_index:
                del self._frame_index2frame[index]
        for stack_indexes in list(self._stack_indexes2features.keys()):
            if any([i < minimum_index for i in stack_indexes]):
                del self._stack_indexes2features[stack_indexes]

    @torch.no_grad()
    def predict(self, frame: torch.Tensor, index: int) -> tuple[Optional[torch.Tensor], int]:
        frame = frame.to(device=self.model.device)
        processed = self.frames_processor(frame[None, None, ...])[0, 0].half()  # store in FP16 to save memory
        self._frame_index2frame[index] = processed
        predict_index = index - self._predict_offset
        predict_indexes = self.indexes_generator.make_stack_indexes(predict_index)
        self._clear_old(predict_indexes[0])
        if set(predict_indexes) <= set(self._frame_index2frame.keys()):
            stacks_indexes = list(batched(predict_indexes, self.model_stack_size))
            for stack_indexes in stacks_indexes:
                if stack_indexes not in self._stack_indexes2features:
                    frames = torch.stack([self._frame_index2frame[i] for i in stack_indexes], dim=0).float()
                    if self.tta:
                        frames = torch.stack([frames, hflip(frames)], dim=0)
                    else:
                        frames = frames.unsqueeze(0)
                    features = self.model.nn_module.forward_2d(frames)
                    self._stack_indexes2features[stack_indexes] = features.half()
            features = torch.cat([self._stack_indexes2features[s].float() for s in stacks_indexes], dim=1)
            features = self.model.nn_module.forward_3d(features)
            prediction = self.model.nn_module.forward_head(features)
            prediction = self.model.prediction_transform(prediction)
            prediction = torch.mean(prediction, dim=0)
            return prediction, predict_index
        else:
            return None, predict_index
        
        
    @torch.no_grad()
    def predict_batch(self, frames: list[torch.Tensor], indices: list[int]) -> list[tuple[Optional[torch.Tensor], int]]:
        """
        Predict batch of frames.
        Args:
            frames (list[Tensor]): List of frame tensors.
            indices (list[int]): Corresponding indices for frames.
        Returns:
            list[tuple[Optional[Tensor], int]]: List of (prediction, predict_index) or (None, predict_index).
        """
        assert len(frames) == len(indices), "Frames and indices must match"

        # Handle empty batch
        if len(frames) == 0:
            return []

        # Check for duplicate indices
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate indices found")

        # Step 1: Sort by indices to ensure sequential order
        sorted_pairs = sorted(zip(frames, indices), key=lambda x: x[1])
        sorted_frames, sorted_indices = zip(*sorted_pairs)

        # Step 2: Update buffer sequentially
        for frame, index in zip(sorted_frames, sorted_indices):
            frame = frame.to(device=self.model.device)
            processed = self.frames_processor(frame[None, None, ...])[0, 0].half()
            self._frame_index2frame[index] = processed

        # Step 3: Collect all required stacks for the batch
        all_stack_frames = []  # To batch forward_2d
        all_stack_keys = []    # To map back features to stacks
        frame_to_stacks = {idx: [] for idx in sorted_indices}  # Per-frame stacks

        # Clear old buffers once using the earliest (smallest) stack index in this batch
        first_predict_index = sorted_indices[0] - self._predict_offset
        first_predict_indexes = self.indexes_generator.make_stack_indexes(first_predict_index)
        self._clear_old(first_predict_indexes[0])

        for index in sorted_indices:
            predict_index = index - self._predict_offset
            predict_indexes = self.indexes_generator.make_stack_indexes(predict_index)

            if set(predict_indexes) <= set(self._frame_index2frame.keys()):
                stacks_indexes = list(batched(predict_indexes, self.model_stack_size))
                frame_to_stacks[index] = stacks_indexes
                for stack_indexes in stacks_indexes:
                    if stack_indexes not in self._stack_indexes2features and stack_indexes not in all_stack_keys:
                        stack_frames = torch.stack([self._frame_index2frame[i] for i in stack_indexes], dim=0).float()
                        if self.tta:
                            stack_frames = torch.stack([stack_frames, hflip(stack_frames)], dim=0)
                        else:
                            stack_frames = stack_frames.unsqueeze(0)
                        all_stack_frames.append(stack_frames)
                        all_stack_keys.append(stack_indexes)

        # Step 4: Batch forward_2d if there are stacks to process
        if all_stack_frames:
            batched_stack_frames = torch.cat(all_stack_frames, dim=0)
            batched_features = self.model.nn_module.forward_2d(batched_stack_frames)
            batched_features = batched_features.half()
            cursor = 0
            for i, key in enumerate(all_stack_keys):
                slice_size = all_stack_frames[i].size(0)
                self._stack_indexes2features[key] = batched_features[cursor:cursor + slice_size]
                cursor += slice_size

        # Step 5: Process per-frame: cat features, forward_3d, forward_head in batch
        all_3d_inputs = []
        valid_indices_for_3d = []

        # Preinitialize results with None for all
        results = [(None, idx - self._predict_offset) for idx in sorted_indices]

        for res_idx, idx in enumerate(sorted_indices):
            stacks_indexes = frame_to_stacks.get(idx, [])
            if stacks_indexes and all(s in self._stack_indexes2features for s in stacks_indexes):
                features_3d_input = torch.cat([self._stack_indexes2features[s].float() for s in stacks_indexes], dim=1)
                all_3d_inputs.append(features_3d_input)
                valid_indices_for_3d.append(res_idx)  # Use position in results

        if all_3d_inputs:
            batched_3d_input = torch.cat(all_3d_inputs, dim=0)
            batched_3d_output = self.model.nn_module.forward_3d(batched_3d_input)
            batched_prediction = self.model.nn_module.forward_head(batched_3d_output)
            batched_prediction = self.model.prediction_transform(batched_prediction)
            b_tta = 2 if self.tta else 1
            num_frames = len(valid_indices_for_3d)
            batched_prediction = batched_prediction.view(num_frames, b_tta, -1).mean(dim=1)

            # Overwrite results at valid positions
            for pos, pred in zip(valid_indices_for_3d, batched_prediction):
                predict_index = results[pos][1]  # Keep the predict_index
                results[pos] = (pred, predict_index)

        # Step 6: Return results in original input order
        original_order_results = [None] * len(indices)
        index_to_position = {idx: pos for pos, idx in enumerate(indices)}
        for res in results:
            original_pos = index_to_position[res[1] + self._predict_offset]
            original_order_results[original_pos] = res

        return original_order_results