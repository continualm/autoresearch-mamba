"""
Architecture-aware data preparation entry point for autoresearch-mamba (MLX).
Reuses the fixed tokenizer, dataloader, and evaluator from prepare_mlx.py for
both Mamba-2 and Mamba-3 runs.

Usage:
    python prepare_mlx_mamba_3.py
    AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 python prepare_mlx_mamba_3.py
    AUTORESEARCH_MLX_PRESET_FILE=mlx_mamba_3_preset.local.json python prepare_mlx_mamba_3.py
"""

import argparse
import json
import os

from prepare_mlx import CACHE_DIR, DEFAULT_NUM_SHARDS, MAX_SHARD, download_data, train_tokenizer

ARCHITECTURE_ENV = "AUTORESEARCH_MLX_ARCHITECTURE"
PRESET_FILE_ENV = "AUTORESEARCH_MLX_PRESET_FILE"


def normalize_architecture(value):
    key = str(value or "mamba-3").strip().lower().replace("_", "-")
    aliases = {
        "mamba2": "mamba-2",
        "mamba-2": "mamba-2",
        "m2": "mamba-2",
        "mamba3": "mamba-3",
        "mamba-3": "mamba-3",
        "m3": "mamba-3",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported architecture {value!r}; expected mamba-2 or mamba-3")
    return aliases[key]


def load_preset_architecture():
    preset_path = os.environ.get(PRESET_FILE_ENV)
    if not preset_path:
        return None
    if not os.path.isabs(preset_path):
        preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), preset_path)
    if not os.path.exists(preset_path):
        return None
    with open(preset_path, "r", encoding="utf-8") as f:
        preset = json.load(f)
    for section in ("train_mamba_3_mlx", "train_mamba_mlx"):
        section_preset = preset.get(section)
        if isinstance(section_preset, dict) and "ARCHITECTURE" in section_preset:
            return normalize_architecture(section_preset["ARCHITECTURE"])
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for architecture-aware MLX autoresearch")
    parser.add_argument("--architecture", choices=["mamba-2", "mamba-3"], default=None, help="Architecture label for the run; prep/eval remain shared")
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    args = parser.parse_args()

    architecture = normalize_architecture(
        args.architecture
        or os.environ.get(ARCHITECTURE_ENV)
        or load_preset_architecture()
        or "mamba-3"
    )
    os.environ[ARCHITECTURE_ENV] = architecture
    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    active_preset = os.environ.get(PRESET_FILE_ENV)
    print(f"SSM architecture: {architecture}")
    print("Prep/eval harness: shared between Mamba-2 and Mamba-3")
    if active_preset:
        print(f"Preset override file: {active_preset}")
    print(f"Cache directory: {CACHE_DIR}")
    print()

    download_data(num_shards, download_workers=args.download_workers)
    print()
    train_tokenizer()
    print()
    print("Done! Ready to train.")
