# Video Upscale

2x/4x Video Super-Resolution using PureVision/PureScale ONNX models with GPU acceleration (TensorRT + PyNvVideoCodec).

## Features

- **2x/4x upscaling** with state-of-the-art models
- **GPU-accelerated** video decode/encode (PyNvVideoCodec / NVENC)
- **TensorRT optimization** for fast inference
- **Tiled inference** for arbitrary resolution input
- **Audio preservation** via FFmpeg

## Models

| Model | Architecture | Description |
|-------|-------------|-------------|
| `purevision` | ESRGAN (pixel-unshuffle) | 2x upscale + denoise + artifact removal |
| `purescale` | RealPLKSR (pixelshuffle) | 2x upscale with fine detail preservation |

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- FFmpeg (for audio processing)

### Install with uv

```bash
# Clone the repository
git clone https://github.com/brucx/video-upscale.git
cd video-upscale

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Download Models

Download ONNX models from [purevision-service releases](https://github.com/brucx/purevision-service/releases):

```bash
# Create models directory
mkdir -p ~/purevision-service/models

# Download models (example)
wget -O ~/purevision-service/models/2x_PureVision.onnx \
  https://github.com/brucx/purevision-service/releases/download/v1.0/2x_PureVision.onnx

wget -O ~/purevision-service/models/2x_PureScale_fp16.onnx \
  https://github.com/brucx/purevision-service/releases/download/v1.0/2x_PureScale_fp16.onnx
```

Or set a custom models directory:
```bash
export MODELS_DIR=/path/to/your/models
```

## Usage

### Basic 2x Upscale

```bash
python upscale.py --input video.mp4 --output video_2x.mp4
```

### 4x Upscale with PureScale

```bash
python upscale.py \
  --input video.mp4 \
  --output video_4x.mp4 \
  --model purescale \
  --scale 4
```

### Full Options

```bash
python upscale.py \
  --input input.mp4 \
  --output output_2x.mp4 \
  --model purevision \     # purevision | purescale
  --scale 2 \              # 2 | 4 (4 = two passes of 2x)
  --gpu 0 \                # GPU device ID
  --tile-size 512 \        # Tile size for inference
  --overlap 16 \           # Overlap pixels between tiles
  --codec hevc             # h264 | hevc | av1
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input, -i` | (required) | Input video path |
| `--output, -o` | (required) | Output video path |
| `--model, -m` | `purevision` | Model: `purevision` or `purescale` |
| `--scale, -s` | `2` | Upscale factor: `2` or `4` |
| `--gpu, -g` | `0` | GPU device ID |
| `--tile-size, -t` | `512` | Tile size for inference |
| `--overlap` | `16` | Overlap pixels between tiles |
| `--codec, -c` | `hevc` | Output codec: `h264`, `hevc`, `av1` |
| `--models-dir` | auto | Path to models directory |
| `--no-audio` | false | Skip audio processing |

## Performance

Tested on RTX 4090:

| Resolution | Model | Scale | FPS |
|------------|-------|-------|-----|
| 1080p | PureVision | 2x | ~15 |
| 1080p | PureScale | 2x | ~12 |
| 720p | PureVision | 4x | ~8 |

## License

MIT
