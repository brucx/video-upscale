"""
Pitch-aware, zero-copy NV12 <-> NCHW(float32) GPU converter
"""

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

KERNEL_SRC = r"""
// NV12(Y + interleaved UV) -> NCHW(float32, normalized 0..1)
extern "C" __global__ void nv12_to_nchw_f32(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    float* __restrict__ out_nchw,
    int W, int H, int pitchY, int pitchUV)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float Y = float(y_plane[y * pitchY + x]) - 16.0f;

    int uvx = (x >> 1) << 1;
    int uvy = (y >> 1);
    float U = float(uv_plane[uvy * pitchUV + uvx    ]) - 128.0f;
    float V = float(uv_plane[uvy * pitchUV + uvx + 1]) - 128.0f;

    float C = 1.164383f * Y;
    float R = (C + 1.596027f * V) * (1.0f/255.0f);
    float G = (C - 0.391762f * U - 0.812968f * V) * (1.0f/255.0f);
    float B = (C + 2.017232f * U) * (1.0f/255.0f);

    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    int hw = W * H;
    int idx = y * W + x;
    out_nchw[idx]        = R;
    out_nchw[idx + hw]   = G;
    out_nchw[idx + 2*hw] = B;
}

// NCHW(float32, normalized 0..1) -> NV12(Y + interleaved UV)
extern "C" __global__ void nchw_f32_to_nv12(
    const float* __restrict__ in_nchw,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int W, int H, int pitchY, int pitchUV)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int hw = W * H;
    int idx = y * W + x;

    float R = fminf(fmaxf(in_nchw[idx]              * 255.0f, 0.0f), 255.0f);
    float G = fminf(fmaxf(in_nchw[idx + hw]         * 255.0f, 0.0f), 255.0f);
    float B = fminf(fmaxf(in_nchw[idx + 2*hw]       * 255.0f, 0.0f), 255.0f);

    float Yf =  0.257f*R + 0.504f*G + 0.098f*B + 16.0f;
    unsigned char Y = (unsigned char) (Yf < 0.f ? 0.f : (Yf > 255.f ? 255.f : Yf));
    y_plane[y * pitchY + x] = Y;

    if ((x % 2 == 0) && (y % 2 == 0))
    {
        float Uf = -0.148f*R - 0.291f*G + 0.439f*B + 128.0f;
        float Vf =  0.439f*R - 0.368f*G - 0.071f*B + 128.0f;
        unsigned char U = (unsigned char) (Uf < 0.f ? 0.f : (Uf > 255.f ? 255.f : Uf));
        unsigned char V = (unsigned char) (Vf < 0.f ? 0.f : (Vf > 255.f ? 255.f : Vf));

        int uvx = x;
        int uvy = y / 2;
        uv_plane[uvy * pitchUV + uvx    ] = U;
        uv_plane[uvy * pitchUV + uvx + 1] = V;
    }
}
"""


FP16_CONVERT_KERNEL_SRC = r"""
#include <cuda_fp16.h>

extern "C" __global__ void convert_f32_to_f16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = __float2half(src[idx]);
}

extern "C" __global__ void convert_f16_to_f32(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = __half2float(src[idx]);
}
"""


class PitchAwareGPUColorConverter:
    """Pitch-aware, zero-copy NV12 <-> NCHW(float32) on GPU"""

    def __init__(self):
        self.module = SourceModule(KERNEL_SRC, no_extern_c=True,
                                   options=["-lineinfo", "-use_fast_math"])
        self.k_nv12_to_nchw = self.module.get_function("nv12_to_nchw_f32")
        self.k_nchw_to_nv12 = self.module.get_function("nchw_f32_to_nv12")

        self._fp16_module = None
        self._f32_to_f16_kernel = None
        self._f16_to_f32_kernel = None

    def _init_fp16_kernels(self):
        if self._fp16_module is None:
            self._fp16_module = SourceModule(FP16_CONVERT_KERNEL_SRC, options=["-std=c++14"])
            self._f32_to_f16_kernel = self._fp16_module.get_function("convert_f32_to_f16")
            self._f16_to_f32_kernel = self._fp16_module.get_function("convert_f16_to_f32")

    @staticmethod
    def _grid(block, W, H):
        return ((W + block[0] - 1) // block[0], (H + block[1] - 1) // block[1], 1)

    def nv12_to_nchw_f32(self, dY_ptr, dUV_ptr, W, H, pitchY, pitchUV,
                         d_out_nchw, stream=None):
        """NV12(Y/UV planes with pitch) -> NCHW(float32, 0..1)"""
        if hasattr(d_out_nchw, "ptr"):
            out_ptr = np.uintp(d_out_nchw.ptr)
        else:
            out_ptr = np.uintp(int(d_out_nchw))

        if hasattr(dY_ptr, "ptr"):
            y_ptr = np.uintp(dY_ptr.ptr)
        else:
            y_ptr = np.uintp(int(dY_ptr))

        if hasattr(dUV_ptr, "ptr"):
            uv_ptr = np.uintp(dUV_ptr.ptr)
        else:
            uv_ptr = np.uintp(int(dUV_ptr))

        block = (32, 16, 1)
        grid = self._grid(block, W, H)

        self.k_nv12_to_nchw(
            y_ptr, uv_ptr, out_ptr,
            np.int32(W), np.int32(H),
            np.int32(pitchY), np.int32(pitchUV),
            block=block, grid=grid, stream=stream
        )

    def nchw_f32_to_nv12(self, d_in_nchw, dY_ptr, dUV_ptr, W, H,
                         pitchY, pitchUV, stream=None):
        """NCHW(float32, 0..1) -> NV12(Y/UV planes with pitch)"""
        if hasattr(d_in_nchw, "ptr"):
            in_ptr = np.uintp(d_in_nchw.ptr)
        else:
            in_ptr = np.uintp(int(d_in_nchw))

        if hasattr(dY_ptr, "ptr"):
            y_ptr = np.uintp(dY_ptr.ptr)
        else:
            y_ptr = np.uintp(int(dY_ptr))

        if hasattr(dUV_ptr, "ptr"):
            uv_ptr = np.uintp(dUV_ptr.ptr)
        else:
            uv_ptr = np.uintp(int(dUV_ptr))

        block = (32, 16, 1)
        grid = self._grid(block, W, H)

        self.k_nchw_to_nv12(
            in_ptr, y_ptr, uv_ptr,
            np.int32(W), np.int32(H),
            np.int32(pitchY), np.int32(pitchUV),
            block=block, grid=grid, stream=stream
        )

    def convert_f32_to_f16_gpu(self, src_ptr, dst_ptr, numel: int, stream):
        """Convert float32 to float16 on GPU"""
        self._init_fp16_kernels()
        block = (256, 1, 1)
        grid = ((numel + block[0] - 1) // block[0], 1, 1)
        self._f32_to_f16_kernel(
            np.intp(src_ptr),
            np.intp(dst_ptr),
            np.int32(numel),
            block=block,
            grid=grid,
            stream=stream,
        )

    def convert_f16_to_f32_gpu(self, src_ptr, dst_ptr, numel: int, stream):
        """Convert float16 to float32 on GPU"""
        self._init_fp16_kernels()
        block = (256, 1, 1)
        grid = ((numel + block[0] - 1) // block[0], 1, 1)
        self._f16_to_f32_kernel(
            np.intp(src_ptr),
            np.intp(dst_ptr),
            np.int32(numel),
            block=block,
            grid=grid,
            stream=stream,
        )
