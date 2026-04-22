#!/usr/bin/env python3
"""
Video Super-Resolution CLI Tool

2x/4x video upscaling using PureVision/PureScale ONNX models with TensorRT acceleration.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="GPU video upscaler with ONNX + TensorRT acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2x upscale with PureVision (default)
  python upscale.py --input video.mp4 --output video_2x.mp4

  # 4x upscale with PureScale
  python upscale.py --input video.mp4 --output video_4x.mp4 --model purescale --scale 4

  # Custom tile size for low VRAM
  python upscale.py --input video.mp4 --output out.mp4 --tile-size 256

Models:
  - purevision: ESRGAN architecture, 2x upscale + denoise + artifact removal
  - purescale: RealPLKSR architecture, 2x upscale with fine detail preservation
"""
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output video path"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="purevision",
        choices=["purevision", "purescale"],
        help="Model to use (default: purevision)"
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=2,
        choices=[2, 4],
        help="Upscale factor: 2 or 4 (4 = two passes of 2x)"
    )
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=int,
        default=512,
        help="Tile size for inference (default: 512)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=16,
        help="Overlap pixels between tiles (default: 16)"
    )
    parser.add_argument(
        "--codec", "-c",
        type=str,
        default="hevc",
        choices=["h264", "hevc", "av1"],
        help="Output video codec (default: hevc)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to models directory (default: ~/purevision-service/models or MODELS_DIR env)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio processing"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    models_dir = args.models_dir
    if models_dir is None:
        models_dir = os.environ.get("MODELS_DIR")
    if models_dir is None:
        models_dir = Path.home() / "purevision-service" / "models"
    models_dir = Path(models_dir)

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        print("Please download models from: https://github.com/brucx/purevision-service/releases")
        sys.exit(1)

    from src.pipeline import process_video

    process_video(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        models_dir=models_dir,
        gpu_id=args.gpu,
        tile_size=args.tile_size,
        overlap=args.overlap,
        scale=args.scale,
        codec=args.codec,
        preserve_audio=not args.no_audio,
    )


if __name__ == "__main__":
    main()
