"""Minimal NVCodec helpers used by the video upscaler pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pycuda.driver as cuda


@dataclass
class _CudaArrayInterface:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    typestr: str
    data: int

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "strides": self.strides,
            "data": (self.data, False),
            "typestr": self.typestr,
            "version": 3,
        }


class AppFrame:
    """Simple GPU frame wrapper compatible with PyNvVideoCodec."""

    def __init__(self, width: int, height: int, fmt: str = "NV12") -> None:
        fmt = fmt.upper()
        self.width = int(width)
        self.height = int(height)

        if fmt != "NV12":
            raise NotImplementedError(f"Only NV12 is supported, got {fmt}")

        frame_size = int(self.width * self.height * 3 / 2)
        self.gpu_alloc = cuda.mem_alloc(frame_size)
        self.frameSize = frame_size
        luma = _CudaArrayInterface(
            (self.height, self.width, 1), (self.width, 1, 1), "|u1", int(self.gpu_alloc)
        )
        chroma = _CudaArrayInterface(
            (self.height // 2, self.width // 2, 2),
            (self.width, 2, 1),
            "|u1",
            int(self.gpu_alloc) + self.width * self.height,
        )
        self._cai = [luma, chroma]

    def cuda(self) -> list[_CudaArrayInterface]:
        """Expose CUDA array interface planes."""
        return self._cai

    @property
    def gpuAlloc(self) -> int:
        """Readable alias used by upstream code."""
        return int(self.gpu_alloc)
