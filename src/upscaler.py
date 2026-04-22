"""
ONNX-based video frame upscaler with TensorRT acceleration.
Supports PureVision and PureScale 2x models.
"""

import os
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pycuda.driver as cuda

log = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path.home() / "purevision-service" / "models"))

MODEL_CONFIG = {
    "purevision": {
        "filename": "2x_PureVision.onnx",
        "architecture": "ESRGAN (pixel-unshuffle)",
        "description": "2x upscale + denoise + artifact removal",
    },
    "purescale": {
        "filename": "2x_PureScale_fp16.onnx",
        "architecture": "RealPLKSR (pixelshuffle)",
        "description": "2x upscale with fine detail preservation",
    },
}


def memcpy2d_dtod(dst, dst_pitch, src, src_pitch, row_bytes, rows, stream):
    """Pitch-aware device-to-device 2D copy."""
    copy = cuda.Memcpy2D()
    copy.set_src_device(int(src))
    copy.set_dst_device(int(dst))
    copy.src_pitch = int(src_pitch)
    copy.dst_pitch = int(dst_pitch)
    copy.width_in_bytes = int(row_bytes)
    copy.height = int(rows)
    copy(stream)


class OnnxUpscaler:
    """ONNX upscaler with TensorRT acceleration and tiling support."""

    def __init__(
        self,
        model_name: str = "purevision",
        models_dir: Path | None = None,
        tile_size: int = 512,
        overlap: int = 16,
    ):
        self.model_name = model_name
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale = 2

        if models_dir is None:
            models_dir = DEFAULT_MODELS_DIR
        models_dir = Path(models_dir)

        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIG.keys())}")

        config = MODEL_CONFIG[model_name]
        model_path = models_dir / config["filename"]
        trt_cache_path = models_dir / "trt_cache" / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        log.info(f"Loading model [{model_name}]: {model_path}")
        self.session = self._load_session(model_path, trt_cache_path)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_dtype = self._get_input_dtype()

        log.info(f"Model loaded: input={self.input_name}, dtype={self.input_dtype}")
        log.info(f"Providers: {self.session.get_providers()}")

        self._setup_buffers()

    def _load_session(self, model_path: Path, trt_cache_path: Path) -> ort.InferenceSession:
        """Load ONNX model with TensorRT/CUDA acceleration."""
        providers = []

        available = ort.get_available_providers()
        log.info(f"Available providers: {available}")

        if "TensorrtExecutionProvider" in available:
            trt_cache_path.mkdir(parents=True, exist_ok=True)
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "trt_max_workspace_size": 4 * 1024 ** 3,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(trt_cache_path),
                }
            ))
            log.info(f"TensorRT enabled, cache: {trt_cache_path}")

        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}))
            log.info("CUDA enabled")

        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = False

        return ort.InferenceSession(str(model_path), sess_options, providers=providers)

    def _get_input_dtype(self) -> np.dtype:
        """Get numpy dtype from model input type."""
        input_type = self.session.get_inputs()[0].type
        if "float16" in input_type:
            return np.float16
        return np.float32

    def _setup_buffers(self):
        """Allocate GPU buffers for tiled inference."""
        tile_elements = 3 * self.tile_size * self.tile_size
        out_tile_size = self.tile_size * self.scale
        out_tile_elements = 3 * out_tile_size * out_tile_size

        self.d_tile_input_f32 = cuda.mem_alloc(tile_elements * 4)
        self.d_tile_input_f16 = cuda.mem_alloc(tile_elements * 2)
        self.d_tile_output_f16 = cuda.mem_alloc(out_tile_elements * 2)
        self.d_tile_output_f32 = cuda.mem_alloc(out_tile_elements * 4)

        self.tile_input_elements = tile_elements
        self.tile_output_elements = out_tile_elements

    def process_nchw(
        self,
        d_nchw_in,
        width: int,
        height: int,
        d_nchw_out,
        stream: cuda.Stream,
        color_converter=None,
    ) -> bool:
        """
        Process an NCHW float32 frame on GPU, tiling as needed.

        Args:
            d_nchw_in: Device pointer to input NCHW float32 buffer
            width: Frame width
            height: Frame height
            d_nchw_out: Device pointer to output NCHW float32 buffer
            stream: CUDA stream
            color_converter: PitchAwareGPUColorConverter for fp16 conversion

        Returns:
            True if successful
        """
        out_width = width * self.scale
        out_height = height * self.scale

        cuda.memset_d32_async(d_nchw_out, 0, 3 * out_width * out_height, stream=stream)

        if width <= self.tile_size and height <= self.tile_size:
            return self._process_single_tile(
                d_nchw_in, width, height, d_nchw_out, stream, color_converter
            )

        return self._process_multi_tile(
            d_nchw_in, width, height, d_nchw_out, stream, color_converter
        )

    def _process_single_tile(
        self,
        d_nchw_in,
        width: int,
        height: int,
        d_nchw_out,
        stream: cuda.Stream,
        color_converter=None,
    ) -> bool:
        """Process a single tile that fits within tile_size."""
        stream.synchronize()

        tile_elements = 3 * width * height
        out_width = width * self.scale
        out_height = height * self.scale
        out_elements = 3 * out_width * out_height

        h_input = np.zeros((1, 3, height, width), dtype=np.float32)
        cuda.memcpy_dtoh(h_input.ravel(), d_nchw_in)

        if self.input_dtype == np.float16:
            h_input = h_input.astype(np.float16)

        h_output = self.session.run([self.output_name], {self.input_name: h_input})[0]

        if h_output.dtype == np.float16:
            h_output = h_output.astype(np.float32)

        cuda.memcpy_htod(d_nchw_out, h_output.ravel())
        return True

    def _process_multi_tile(
        self,
        d_nchw_in,
        width: int,
        height: int,
        d_nchw_out,
        stream: cuda.Stream,
        color_converter=None,
    ) -> bool:
        """Process frame using multiple tiles with overlap blending."""
        stream.synchronize()

        h_frame = np.zeros((1, 3, height, width), dtype=np.float32)
        cuda.memcpy_dtoh(h_frame.ravel(), d_nchw_in)

        out_width = width * self.scale
        out_height = height * self.scale
        h_out_frame = np.zeros((1, 3, out_height, out_width), dtype=np.float32)

        x_origins = self._compute_tile_origins(width)
        y_origins = self._compute_tile_origins(height)

        for origin_y in y_origins:
            for origin_x in x_origins:
                tile_w = min(self.tile_size, width - origin_x)
                tile_h = min(self.tile_size, height - origin_y)

                pad_left = 0 if origin_x == 0 else self.overlap
                pad_right = 0 if origin_x + tile_w >= width else self.overlap
                pad_top = 0 if origin_y == 0 else self.overlap
                pad_bottom = 0 if origin_y + tile_h >= height else self.overlap

                h_tile = h_frame[:, :, origin_y:origin_y+tile_h, origin_x:origin_x+tile_w].copy()

                if self.input_dtype == np.float16:
                    h_tile = h_tile.astype(np.float16)

                h_tile_out = self.session.run([self.output_name], {self.input_name: h_tile})[0]

                if h_tile_out.dtype == np.float16:
                    h_tile_out = h_tile_out.astype(np.float32)

                crop_w = tile_w - pad_left - pad_right
                crop_h = tile_h - pad_top - pad_bottom

                if crop_w <= 0 or crop_h <= 0:
                    continue

                crop_x_out = pad_left * self.scale
                crop_y_out = pad_top * self.scale
                crop_w_out = crop_w * self.scale
                crop_h_out = crop_h * self.scale

                dst_x_out = (origin_x + pad_left) * self.scale
                dst_y_out = (origin_y + pad_top) * self.scale

                h_out_frame[
                    :, :,
                    dst_y_out:dst_y_out+crop_h_out,
                    dst_x_out:dst_x_out+crop_w_out
                ] = h_tile_out[
                    :, :,
                    crop_y_out:crop_y_out+crop_h_out,
                    crop_x_out:crop_x_out+crop_w_out
                ]

        cuda.memcpy_htod(d_nchw_out, h_out_frame.ravel())
        return True

    def _compute_tile_origins(self, length: int) -> list[int]:
        """Generate tile start positions with overlap coverage."""
        if length <= self.tile_size:
            return [0]

        stride = max(1, self.tile_size - 2 * self.overlap)
        origins = [0]

        while True:
            last = origins[-1]
            if last + self.tile_size >= length:
                break

            next_origin = last + stride
            if next_origin + self.tile_size >= length:
                tail = max(length - self.tile_size, 0)
                if tail != origins[-1]:
                    origins.append(tail)
                break

            origins.append(next_origin)

        return origins
