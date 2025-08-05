from typing import Any, Generator
from pathlib import Path
import logging

import torch
import PyNvVideoCodec as nvc

from src.frame_fetchers.abstract import AbstractFrameFetcher

logger = logging.getLogger(__name__)


class NvDecFrameFetcher(AbstractFrameFetcher):
    """
    Sử dụng Demuxer và Decoder cấp thấp của PyNvVideoCodec để tăng tính ổn định
    với các video có frame bị hỏng. Các đối tượng decoder và demuxer được giữ
    trong suốt vòng đời của fetcher để tránh lỗi bộ nhớ.

    Một generator nội bộ quản lý vòng lặp giải mã, cho phép bắt và bỏ qua
    các gói tin (packets) bị lỗi, tránh được lỗi Segmentation Fault.
    """

    def __init__(self, video_path: str | Path, gpu_id: int, cuda_stream: int = 0, cuda_context: int = 0):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self.cuda_stream = cuda_stream
        self.cuda_context = cuda_context

        try:
            # Thăm dò video một lần để lấy metadata
            probe_demuxer = nvc.CreateDemuxer(str(self.video_path))
            self.width = probe_demuxer.Width()
            self.height = probe_demuxer.Height()
            self.codec = probe_demuxer.GetNvCodecId()
            del probe_demuxer
        except Exception as e:
            logger.error(f"Không thể thăm dò video '{self.video_path}' để lấy metadata: {e}", exc_info=True)
            raise

        # Khởi tạo generator để nạp frame
        self._frame_idx = -1
        self._frame_generator: Generator | None = None
        self._reset_generator()

    def _reset_generator(self):
        """Tạo hoặc tạo lại generator giải mã."""
        logger.debug("Tạo lại generator giải mã.")
        self.demuxer = nvc.CreateDemuxer(str(self.video_path))
        self.decoder = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=self.codec,
            usedevicememory=True,  # Fixed: changed from use_device_memory to usedevicememory
            cudacontext=self.cuda_context,
            cudastream=self.cuda_stream,
            maxwidth=self.width,
            maxheight=self.height
        )
        self._frame_generator = self._create_frame_generator()
        self._frame_idx = -1

    def _create_frame_generator(self) -> Generator:
        """Generator để giải mã và yield từng frame."""
        for packet in self.demuxer:
            try:
                surfaces = self.decoder.Decode(packet)
                for surface in surfaces:
                    yield surface
            except Exception as e:
                logger.warning(f"Bỏ qua packet bị lỗi trong quá trình giải mã: {e}. Video có thể bị hỏng.")
                continue

    def _next_decode(self) -> Any:
        """Lấy frame tiếp theo từ generator."""
        if self._frame_generator is None:
            return None
        try:
            frame = next(self._frame_generator)
            self._frame_idx += 1
            return frame
        except StopIteration:
            return None  # Kết thúc video

    def _seek_and_decode(self, index: int) -> Any:
        """Tìm và giải mã một frame cụ thể."""
        # Nếu seek về phía sau hoặc quá xa, hãy tạo lại generator
        if index <= self._frame_idx:
            self._reset_generator()

        # Bỏ qua các frame cho đến khi đến đúng index
        while self._frame_idx < index:
            frame = self._next_decode()
            if frame is None:  # Hết video trước khi đến index
                logger.warning(f"Không thể seek đến frame {index}, video có ít hơn {index + 1} frames.")
                return None
        return frame

    def _convert(self, frame: Any) -> torch.Tensor:
        """
        Chuyển đổi frame thô (từ decoder) thành grayscale tensor.
        """
        if frame is None:
            # Trả về frame đen nếu giải mã thất bại
            return torch.zeros(self.height, self.width,
                               dtype=torch.uint8,
                               device=f"cuda:{self.gpu_id}")

        try:
            # Sử dụng DLPack để chuyển đổi zero-copy hiệu quả.
            frame_tensor = torch.from_dlpack(frame)

            # Giả định định dạng là YUV (ví dụ: NV12), kênh đầu tiên (Y) là kênh grayscale.
            if frame_tensor.dim() == 2 and frame_tensor.shape[0] == int(self.height * 1.5):
                frame_tensor = frame_tensor[:self.height, :]  # Lấy Y plane
            elif frame_tensor.dim() == 3:  # YUV format, just take Y channel
                frame_tensor = frame_tensor[0]  # Kênh Y là grayscale

            # Đảm bảo tensor cuối cùng là 2D
            if frame_tensor.dim() != 2:
                logger.warning(f"Frame tensor có chiều không mong muốn: {frame_tensor.shape}. Cố gắng reshape.")
                frame_tensor = frame_tensor.view(self.height, self.width)

            return frame_tensor.to(torch.uint8)

        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi frame sang tensor: {e}", exc_info=True)
            return torch.zeros(self.height, self.width,
                               dtype=torch.uint8,
                               device=f"cuda:{self.gpu_id}")