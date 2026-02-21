"""
Benchmark harness for open-weight text-to-video models.

Downloads model weights if missing, then generates videos for every
enabled model x prompt x seed combination. Outputs land in output/.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Enable fast Rust-based downloads via hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def check_hf_transfer():
    """Verify hf_transfer is installed and enabled, warn loudly if not."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    try:
        import hf_transfer  # noqa: F401
        available = True
    except ImportError:
        available = False

    enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"

    if available and enabled:
        print(f"{GREEN}{BOLD}[OK] hf_transfer is installed and enabled - fast downloads active{RESET}")
        return

    print(f"\n{RED}{BOLD}{'='*70}")
    print(f"  WARNING: FAST DOWNLOADS NOT ACTIVE")
    print(f"{'='*70}{RESET}")

    if not available:
        print(f"{YELLOW}  hf_transfer package is NOT installed.{RESET}")
        print(f"{YELLOW}  Install it with: pip install hf_transfer{RESET}")

    if not enabled:
        print(f"{YELLOW}  HF_HUB_ENABLE_HF_TRANSFER is not set to '1'.{RESET}")

    print(f"{RED}{BOLD}  Downloads will be 5-10x slower without hf_transfer!{RESET}")
    print(f"{RED}{BOLD}{'='*70}{RESET}\n")

class Spinner:
    """Background spinner that shows elapsed time."""

    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, message: str):
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        t0 = time.time()
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - t0
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {frame} {self._message} ({elapsed:.0f}s)")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.2)
        elapsed = time.time() - t0
        sys.stdout.write(f"\r  [done] {self._message} ({elapsed:.1f}s)\n")
        sys.stdout.flush()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"


def load_settings() -> dict:
    settings_path = CONFIG_DIR / "settings.json"
    if not settings_path.exists():
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        print("No settings.json found. Let's set up your directories.\n")
        while True:
            model_dir = input("Enter a folder path for storing models: ").strip()
            if model_dir and Path(model_dir).parent.exists():
                break
            print("Invalid path. Parent directory must exist. Try again.")
        settings = {
            "model_dir": model_dir,
            "output_dir": str(ROOT / "output"),
        }
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"\nSettings saved to {settings_path}\n")
    settings = load_json(settings_path)
    return {
        "model_dir": Path(settings["model_dir"]),
        "output_dir": Path(settings["output_dir"]),
    }


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_metadata(out_dir: Path, meta: dict):
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def get_hf_token() -> str | None:
    """Resolve HF token from env var or huggingface-cli login cache."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


IGNORE_PATTERNS = [
    "*.md",
    "*.txt",
    ".gitattributes",
    "assets/*",
    "*.png",
    "*.jpg",
    "*.gif",
]


def download_model(model_cfg: dict, model_dir: Path) -> Path:
    """Download model weights from HuggingFace if not already cached.

    For models with pipeline=="custom", uses snapshot_download to a local dir.
    For diffusers models, downloading is deferred to load_pipeline() which
    calls from_pretrained (handles its own caching).
    """
    if model_cfg.get("pipeline") != "custom":
        repo = model_cfg.get("diffusers_repo", model_cfg["repo"])
        print(f"[download] {model_cfg['name']} ({repo}) - will download on first load if needed")
        return model_dir / model_cfg["id"]

    from huggingface_hub import snapshot_download

    repo = model_cfg["repo"]
    local_path = model_dir / model_cfg["id"]
    print(f"[download] Checking {model_cfg['name']} ({repo}) ...")
    if local_path.exists() and any(
        p.name != ".cache" for p in local_path.iterdir()
    ):
        print(f"  [skip] Already downloaded at {local_path}")
        return local_path
    print(f"  [download] Downloading to {local_path} ...")
    token = get_hf_token()
    snapshot_download(
        repo_id=repo,
        local_dir=str(local_path),
        token=token,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"  [done] {model_cfg['name']} ready at {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _get_model_repo(model_cfg: dict) -> str:
    """Return the HF repo ID to use for from_pretrained."""
    return model_cfg.get("diffusers_repo", model_cfg["repo"])


def load_pipeline(model_cfg: dict, model_dir: Path):
    """Load and return an inference pipeline for the given model.

    Returns a dict with 'pipe' (the pipeline object), 'id' (model id),
    and any extra keys needed for generation / export.
    """
    import torch

    model_id = model_cfg["id"]
    source = _get_model_repo(model_cfg)
    token = get_hf_token()
    print(f"  [load] Loading {model_id} from {source} ...")

    cache = str(model_dir)

    if model_id == "ltx-video-2":
        from diffusers.pipelines.ltx2 import LTX2Pipeline
        with Spinner(f"Loading {model_id} weights"):
            pipe = LTX2Pipeline.from_pretrained(
                source, torch_dtype=torch.bfloat16, token=token,
                cache_dir=cache,
            )
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "id": model_id}

    if model_id == "hunyuan-video":
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        with Spinner(f"Loading {model_id} weights"):
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                source, subfolder="transformer",
                torch_dtype=torch.bfloat16, token=token,
                cache_dir=cache,
            )
            pipe = HunyuanVideoPipeline.from_pretrained(
                source, transformer=transformer,
                torch_dtype=torch.float16, token=token,
                cache_dir=cache,
            )
            pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "id": model_id}

    if model_id == "wan-2.2":
        from diffusers import AutoencoderKLWan, WanPipeline
        with Spinner(f"Loading {model_id} weights"):
            vae = AutoencoderKLWan.from_pretrained(
                source, subfolder="vae",
                torch_dtype=torch.float32, token=token,
                cache_dir=cache,
            )
            pipe = WanPipeline.from_pretrained(
                source, vae=vae,
                torch_dtype=torch.bfloat16, token=token,
                cache_dir=cache,
            )
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "id": model_id}

    if model_id == "skyreels-v1":
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        local_transformer = model_dir / model_cfg["id"]
        with Spinner(f"Loading {model_id} weights"):
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                str(local_transformer), torch_dtype=torch.bfloat16, token=token,
            )
            base_source = model_cfg.get("diffusers_repo", "hunyuanvideo-community/HunyuanVideo")
            pipe = HunyuanVideoPipeline.from_pretrained(
                base_source, transformer=transformer,
                torch_dtype=torch.float16, token=token,
                cache_dir=cache,
            )
            pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "id": model_id}

    if model_id == "mochi-1":
        from diffusers import MochiPipeline
        with Spinner(f"Loading {model_id} weights"):
            pipe = MochiPipeline.from_pretrained(
                source, variant="bf16",
                torch_dtype=torch.bfloat16, token=token,
                cache_dir=cache,
            )
            pipe.enable_vae_tiling()
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "id": model_id}

    if model_id == "cogvideox-5b":
        from diffusers import CogVideoXPipeline
        with Spinner(f"Loading {model_id} weights"):
            pipe = CogVideoXPipeline.from_pretrained(
                source, torch_dtype=torch.bfloat16, token=token,
                cache_dir=cache,
            )
            pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "id": model_id}

    if model_id == "open-sora-v2":
        return {"pipe": None, "id": model_id, "model_dir": model_dir}

    raise ValueError(f"Unknown model id: {model_id}")


# Per-model default FPS for export
MODEL_FPS = {
    "ltx-video-2": 24,
    "hunyuan-video": 15,
    "wan-2.2": 16,
    "skyreels-v1": 15,
    "mochi-1": 30,
    "cogvideox-5b": 8,
    "open-sora-v2": 24,
}


def generate_video(
    pipeline_info: dict,
    prompt_text: str,
    seed: int,
    params: dict,
    out_path: Path,
):
    """Run a single generation and save the result to out_path."""
    import torch
    from diffusers.utils import export_to_video

    model_id = pipeline_info["id"]
    pipe = pipeline_info["pipe"]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    w, h = params["resolution"]
    num_frames = params["frame_count"]
    steps = params["steps"]
    cfg = params["cfg_scale"]
    fps = MODEL_FPS[model_id]

    if model_id == "ltx-video-2":
        from diffusers.pipelines.ltx2.export_utils import encode_video
        video, audio = pipe(
            prompt=prompt_text,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            width=w, height=h,
            num_frames=num_frames,
            frame_rate=float(fps),
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            output_type="np",
            return_dict=False,
        )
        encode_video(
            video[0],
            fps=float(fps),
            audio=audio[0].float().cpu(),
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=str(out_path),
        )
        return

    if model_id == "hunyuan-video":
        output = pipe(
            prompt=prompt_text,
            height=h, width=w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).frames[0]
        export_to_video(output, str(out_path), fps=fps)
        return

    if model_id == "wan-2.2":
        output = pipe(
            prompt=prompt_text,
            height=h, width=w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).frames[0]
        export_to_video(output, str(out_path), fps=fps)
        return

    if model_id == "skyreels-v1":
        output = pipe(
            prompt=prompt_text,
            negative_prompt="Aerial view, overexposed, low quality, deformation",
            height=h, width=w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=1.0,
            true_cfg_scale=cfg,
            generator=generator,
        ).frames[0]
        export_to_video(output, str(out_path), fps=fps)
        return

    if model_id == "mochi-1":
        output = pipe(
            prompt=prompt_text,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).frames[0]
        export_to_video(output, str(out_path), fps=fps)
        return

    if model_id == "cogvideox-5b":
        output = pipe(
            prompt=prompt_text,
            height=h, width=w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).frames[0]
        export_to_video(output, str(out_path), fps=fps)
        return

    if model_id == "open-sora-v2":
        _generate_open_sora(pipeline_info, prompt_text, seed, params, out_path)
        return

    raise ValueError(f"Unknown model id: {model_id}")


def _generate_open_sora(
    pipeline_info: dict,
    prompt_text: str,
    seed: int,
    params: dict,
    out_path: Path,
):
    """Generate a video using Open-Sora v2 via its CLI scripts."""
    model_dir = pipeline_info["model_dir"]
    repo_dir = model_dir / "open-sora-v2-repo"

    if not (repo_dir / "scripts").exists():
        print("  [setup] Cloning Open-Sora v2 repo ...")
        subprocess.run(
            ["git", "clone", "https://github.com/hpcaitech/Open-Sora", str(repo_dir)],
            check=True,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "torchrun", "--nproc_per_node", "1", "--standalone",
            str(repo_dir / "scripts" / "diffusion" / "inference.py"),
            str(repo_dir / "configs" / "diffusion" / "inference" / "768px.py"),
            "--save-dir", tmp_dir,
            "--prompt", prompt_text,
        ]
        subprocess.run(cmd, check=True, cwd=str(repo_dir))

        generated = list(Path(tmp_dir).glob("*.mp4"))
        if not generated:
            raise RuntimeError("Open-Sora v2 did not produce any mp4 output")
        shutil.copy2(str(generated[0]), str(out_path))


def run_benchmark(model_ids: list[str] | None = None, download_only: bool = False):
    settings = load_settings()
    model_dir = settings["model_dir"]
    output_dir = settings["output_dir"]

    prompts_cfg = load_json(CONFIG_DIR / "prompts.json")
    models_cfg = load_json(CONFIG_DIR / "models.json")

    params = prompts_cfg["params"]
    prompts = prompts_cfg["prompts"]
    models = [m for m in models_cfg["models"] if m["enabled"]]

    if model_ids:
        models = [m for m in models if m["id"] in model_ids]

    if not models:
        print("No models selected. Check config/models.json or --models flag.")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model directory:  {model_dir}")
    print(f"Output directory: {output_dir}")

    # --- Download phase ---
    check_hf_transfer()
    for model in models:
        download_model(model, model_dir)

    if download_only:
        print("Download-only mode. Done.")
        return

    # --- Generation phase ---
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model['name']}")
        print(f"{'='*60}")

        pipeline = load_pipeline(model, model_dir)

        for prompt in prompts:
            prompt_dir = output_dir / model["id"] / prompt["id"]
            prompt_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "model": model["id"],
                "prompt_id": prompt["id"],
                "prompt_text": prompt["text"],
                "params": params,
                "runs": [],
            }

            for i, seed in enumerate(params["seeds"]):
                out_path = prompt_dir / f"seed_{i}.mp4"

                if out_path.exists():
                    print(f"  [skip] {out_path} already exists")
                    continue

                print(f"  [gen] {model['id']} / {prompt['id']} / seed={seed} ...")
                t0 = time.time()
                generate_video(pipeline, prompt["text"], seed, params, out_path)
                elapsed = time.time() - t0

                meta["runs"].append({
                    "seed": seed,
                    "file": out_path.name,
                    "generation_time_s": round(elapsed, 2),
                })
                print(f"         Done in {elapsed:.1f}s -> {out_path}")

            save_metadata(prompt_dir, meta)

    print(f"\nBenchmark complete. Results in {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run video generation benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only run these model IDs (space-separated). Default: all enabled.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download model weights but skip generation.",
    )
    args = parser.parse_args()
    run_benchmark(model_ids=args.models, download_only=args.download_only)


if __name__ == "__main__":
    main()
