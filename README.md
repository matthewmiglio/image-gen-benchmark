# image-gen-benchmark

Benchmark harness for comparing open-weight text-to-video models under controlled conditions. Generates multiple videos per prompt across every model using identical parameters (resolution, seeds, CFG, steps, frame count), then organizes outputs for side-by-side comparison.

## Models tested

Currently configured to benchmark these top-tier local models:

- LTX-Video 2 (Lightricks)
- Hunyuan Video (Tencent)
- Open-Sora 2.5
- Wan 2.2
- SkyReels V1
- Mochi 1 (Genmo)
- CogVideoX-5B

Models can be toggled on/off in `config/models.json`.

## How it works

1. Same set of prompts is fed to every enabled model
2. Each prompt is generated 3 times with fixed seeds for reproducibility
3. Shared params (resolution, frame count, CFG, steps) are held constant
4. Videos and metadata are written to the configured output directory

## Setup

Requires Python 3.11+, a CUDA-capable GPU (RTX 4090+ recommended), and [Poetry](https://python-poetry.org/).

```bash
git clone <repo-url> && cd image-gen-benchmark
poetry install
```

### HuggingFace authentication

Some model repos are gated and require authentication. Create a token at https://huggingface.co/settings/tokens (read access), then log in once:

```bash
poetry run python -c "from huggingface_hub import login; login(token='hf_your_token_here')"
```

This stores your token locally at `~/.cache/huggingface/token`. Downloads use `hf_transfer` (a Rust-based parallel downloader) automatically for significantly faster speeds.

## Configure

- `config/prompts.json` -- test prompts and shared generation parameters (seeds, resolution, CFG, etc.)
- `config/models.json` -- which models to run, toggle with `enabled`
- `config/settings.json` -- where to store model weights and output (defaults to `D:/`, change to suit your disk)

## Run

```bash
# Run full benchmark (downloads missing models automatically)
poetry run python benchmark.py

# Run specific models only
poetry run python benchmark.py --models ltx-video-2 hunyuan-video

# Download weights without generating
poetry run python benchmark.py --download-only
```

## Output

Videos and metadata land in the directory set by `config/settings.json`:

```
<output_dir>/
  ltx-video-2/
    campfire/
      seed_0.mp4
      seed_1.mp4
      seed_2.mp4
      metadata.json
  hunyuan-video/
    campfire/
      ...
```

Generation is resumable -- existing outputs are skipped on re-run.
