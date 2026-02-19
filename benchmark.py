"""
Benchmark harness for open-weight text-to-video models.

Downloads model weights if missing, then generates videos for every
enabled model x prompt x seed combination. Outputs land in output/.
"""

import argparse
import json
import os
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

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"


def load_settings() -> dict:
    settings = load_json(CONFIG_DIR / "settings.json")
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
    """Download model weights from HuggingFace if not already cached."""
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

def load_pipeline(model_cfg: dict, model_dir: Path):
    """Load and return an inference pipeline for the given model."""
    local_path = model_dir / model_cfg["id"]
    # TODO: branch on model_cfg["pipeline"] to load diffusers / custom pipelines
    raise NotImplementedError(
        f"Pipeline loading for {model_cfg['id']} not yet implemented. "
        f"Weights at: {local_path}"
    )


def generate_video(pipeline, prompt_text: str, seed: int, params: dict) -> bytes:
    """Run a single generation and return raw video bytes."""
    # TODO: call pipeline with prompt, seed, resolution, frame_count, cfg, steps
    raise NotImplementedError("Video generation not yet implemented.")


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
                video_bytes = generate_video(pipeline, prompt["text"], seed, params)
                elapsed = time.time() - t0

                out_path.write_bytes(video_bytes)
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
