"""
GPU video processing pipeline: decode -> upscale -> encode
Uses PyNvVideoCodec for hardware-accelerated video I/O.
"""

import logging
import math
import os
import subprocess
import time

import numpy as np
import PyNvVideoCodec as nvc
import pycuda.driver as cuda

from .color import PitchAwareGPUColorConverter
from .nvcodec import AppFrame
from .upscaler import OnnxUpscaler

log = logging.getLogger(__name__)

MAX_NVENC_DIM = 4096


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


def process_video(
    input_path: str,
    output_path: str,
    model_name: str = "purevision",
    models_dir: str | None = None,
    gpu_id: int = 0,
    tile_size: int = 512,
    overlap: int = 16,
    scale: int = 2,
    codec: str = "hevc",
    preserve_audio: bool = True,
):
    """
    End-to-end GPU video upscaling pipeline.

    Args:
        input_path: Input video file path
        output_path: Output video file path
        model_name: Model to use (purevision or purescale)
        models_dir: Directory containing ONNX models
        gpu_id: GPU device ID
        tile_size: Tile size for inference
        overlap: Overlap pixels between tiles
        scale: Upscale factor (2 or 4)
        codec: Output codec (h264, hevc, av1)
        preserve_audio: Whether to copy audio from input
    """
    print("=" * 60)
    print("GPU Video Upscaling Pipeline")
    print("=" * 60)

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
    cuda.init()
    dev = cuda.Device(gpu_id)
    ctx = dev.make_context()

    try:
        if preserve_audio:
            temp_output = output_path.replace(".mp4", "_temp_video_only.mp4")
            video_only_output = temp_output
        else:
            video_only_output = output_path

        print(f"\n[1] Initializing on GPU {gpu_id}...")

        decode_stream = cuda.Stream()
        process_stream = cuda.Stream()
        encode_stream = cuda.Stream()
        print("   - CUDA streams ready")

        nv_dmx = nvc.CreateDemuxer(filename=input_path)
        fps = nv_dmx.FrameRate()
        width = nv_dmx.Width()
        height = nv_dmx.Height()
        codec_id = nv_dmx.GetNvCodecId()

        print(f"\n[2] Input video:")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Codec: {codec_id}")

        nv_dec = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=codec_id,
            cudacontext=int(ctx.handle),
            cudastream=int(decode_stream.handle),
            usedevicememory=1,
        )
        print("   - Hardware decoder ready")

        up_width = width * scale
        up_height = height * scale

        encode_width = up_width
        encode_height = up_height
        resize_required = encode_width > MAX_NVENC_DIM or encode_height > MAX_NVENC_DIM

        if resize_required:
            clamp_ratio = MAX_NVENC_DIM / float(max(encode_width, encode_height))
            target_width = encode_width * clamp_ratio
            target_height = encode_height * clamp_ratio

            def _even_floor(value: float) -> int:
                val = int(math.floor(value))
                if val % 2 != 0:
                    val -= 1
                return max(val, 2)

            encode_width = _even_floor(target_width)
            encode_height = _even_floor(target_height)

        print(f"\n[3] Output video:")
        print(f"   - Upscaled: {up_width}x{up_height} ({scale}x)")
        if resize_required:
            print(f"   - NVENC target: {encode_width}x{encode_height} (clamped)")
        else:
            print(f"   - NVENC target: {encode_width}x{encode_height}")

        codec = codec.lower()
        supported_codecs = {"h264", "hevc", "av1"}
        if codec not in supported_codecs:
            raise ValueError(f"Unsupported codec '{codec}'. Choose from {supported_codecs}.")

        if codec == "h264" and (encode_width > MAX_NVENC_DIM or encode_height > MAX_NVENC_DIM):
            print("[INFO] Switching to HEVC due to resolution limits")
            codec = "hevc"

        encoder_config = {
            "codec": codec,
            "preset": "P2",
            "rc": "cbr",
            "bitrate": 15000000,
            "fps": int(fps),
            "bframes": 0,
            "lookahead": 0,
            "aq": 0,
        }

        nv_enc = nvc.CreateEncoder(
            encode_width, encode_height, "NV12", True, **encoder_config
        )
        print(f"   - Hardware encoder ready ({codec.upper()})")

        NUM_BUFFERS = 8
        app_frames = [AppFrame(encode_width, encode_height, "NV12") for _ in range(NUM_BUFFERS)]
        decode_events = [cuda.Event() for _ in range(NUM_BUFFERS)]
        process_events = [cuda.Event() for _ in range(NUM_BUFFERS)]

        d_nchw_in = cuda.mem_alloc(3 * width * height * 4)
        d_nchw_out = cuda.mem_alloc(3 * up_width * up_height * 4)
        d_y_out = cuda.mem_alloc(encode_width * encode_height)
        d_uv_out = cuda.mem_alloc(encode_width * encode_height // 2)

        if scale == 4:
            d_nchw_intermediate = cuda.mem_alloc(3 * (width * 2) * (height * 2) * 4)
        else:
            d_nchw_intermediate = None

        upscaler = OnnxUpscaler(
            model_name=model_name,
            models_dir=models_dir,
            tile_size=tile_size,
            overlap=overlap,
            device_id=gpu_id,
            cuda_context=ctx,
        )
        print(f"   - Upscaler ready: {model_name}")

        try:
            cuda.Context.pop()
        except cuda.LogicError:
            pass
        ctx.push()

        color_converter = PitchAwareGPUColorConverter()
        print("   - Color converter ready")

        frame_count = 0
        processing_times = []
        start_time = time.time()

        print(f"\n[4] Processing frames...")

        with open(video_only_output, "wb") as out_file:
            buffer_idx = 0

            for packet in nv_dmx:
                for decoded_frame in nv_dec.Decode(packet):
                    frame_start = time.perf_counter()

                    app_frame = app_frames[buffer_idx]
                    decode_event = decode_events[buffer_idx]
                    process_event = process_events[buffer_idx]

                    try:
                        luma_ptr = decoded_frame.GetPtrToPlane(0)
                        uv_ptr = decoded_frame.GetPtrToPlane(1)

                        frame_size = decoded_frame.framesize()
                        expected_compact = width * height * 3 // 2
                        uv_offset = int(uv_ptr) - int(luma_ptr)

                        if frame_size == expected_compact and uv_offset == width * height:
                            pitchY = width
                            pitchUV = width
                        else:
                            pitchY = uv_offset // height
                            pitchUV = pitchY
                    except Exception as exc:
                        log.error(f"Failed to decode NV12 planes: {exc}")
                        continue

                    decode_event.record(decode_stream)
                    process_stream.wait_for_event(decode_event)

                    color_converter.nv12_to_nchw_f32(
                        luma_ptr,
                        uv_ptr,
                        width,
                        height,
                        pitchY,
                        pitchUV,
                        d_nchw_in,
                        stream=process_stream,
                    )

                    if scale == 4:
                        upscaler.process_nchw(
                            d_nchw_in, width, height, d_nchw_intermediate, process_stream
                        )
                        upscaler.process_nchw(
                            d_nchw_intermediate, width * 2, height * 2, d_nchw_out, process_stream
                        )
                    else:
                        upscaler.process_nchw(
                            d_nchw_in, width, height, d_nchw_out, process_stream
                        )

                    out_pitchY = encode_width
                    out_pitchUV = encode_width

                    color_converter.nchw_f32_to_nv12(
                        d_nchw_out,
                        d_y_out,
                        d_uv_out,
                        encode_width,
                        encode_height,
                        out_pitchY,
                        out_pitchUV,
                        stream=process_stream,
                    )

                    cuda.memcpy_dtod_async(
                        app_frame.gpuAlloc,
                        d_y_out,
                        out_pitchY * encode_height,
                        stream=process_stream,
                    )
                    cuda.memcpy_dtod_async(
                        int(app_frame.gpuAlloc) + out_pitchY * encode_height,
                        d_uv_out,
                        out_pitchUV * encode_height // 2,
                        stream=process_stream,
                    )

                    process_event.record(process_stream)
                    encode_stream.wait_for_event(process_event)
                    process_event.synchronize()

                    bitstream = nv_enc.Encode(app_frame)
                    if bitstream:
                        out_file.write(bytearray(bitstream))

                    frame_time = time.perf_counter() - frame_start
                    processing_times.append(frame_time)
                    frame_count += 1

                    buffer_idx = (buffer_idx + 1) % NUM_BUFFERS

                    if frame_count % 60 == 0:
                        avg_time = np.mean(processing_times[-60:])
                        current_fps = 1.0 / avg_time if avg_time > 0 else 0
                        print(f"   Frame {frame_count}: {current_fps:.1f} FPS")

            ctx.synchronize()
            bitstream = nv_enc.EndEncode()
            if bitstream:
                out_file.write(bytearray(bitstream))

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n[5] Performance:")
        print("=" * 60)
        print(f"   Frames: {frame_count}")
        print(f"   Time: {total_time:.2f}s")
        print(f"   FPS: {avg_fps:.2f}")

        if processing_times:
            p50 = np.percentile(processing_times, 50) * 1000
            p90 = np.percentile(processing_times, 90) * 1000
            p99 = np.percentile(processing_times, 99) * 1000
            print(f"   Latency: P50={p50:.1f}ms, P90={p90:.1f}ms, P99={p99:.1f}ms")

        if preserve_audio:
            print(f"\n[6] Adding audio...")
            _add_audio(input_path, video_only_output, output_path)

        print(f"\n[7] Output: {output_path}")
        print("=" * 60)

    finally:
        ctx.pop()


def _add_audio(input_path: str, video_only_path: str, output_path: str):
    """Mux audio from input to output video."""
    check_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]

    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
        has_audio = result.stdout.strip() == "audio"
    except Exception:
        has_audio = False

    if has_audio:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", video_only_path,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=300)
            print("   - Audio added")
            os.remove(video_only_path)
        except Exception as e:
            log.warning(f"Failed to add audio: {e}")
            if video_only_path != output_path:
                os.rename(video_only_path, output_path)
    else:
        if video_only_path != output_path:
            os.rename(video_only_path, output_path)
