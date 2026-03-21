# autoresearch-mamba

![teaser](progress.png)

Karpathy-style autoresearch for Mamba language models on Apple Silicon with MLX.

The current MLX path is architecture-aware:

- `mamba-2`
- `mamba-3`

The repo keeps the same high-level loop as `autoresearch-karpathy`: a fixed evaluator, a fixed 5 minute training budget, and one editable training surface that an agent can mutate to lower `val_bpb`.

## Status

- The canonical MLX path is now `prepare_mlx_mamba_3.py` + `train_mamba_3_mlx.py`
- `train_mamba_3_mlx.py` supports both `mamba-2` and `mamba-3`
- `train_mamba_mlx.py` remains as the original Mamba-2 MLX script
- The Mamba-3 MLX path is active work and is being tightened against the upstream reference implementation

For background on Mamba-3, see Together AI's March 17, 2026 overview: https://www.together.ai/blog/mamba-3

## Goal

The objective is simple:

- minimize `val_bpb`
- under a fixed `TIME_BUDGET = 300` seconds
- without changing the evaluation harness for the active run

## Canonical MLX Path

Use these files for new MLX autoresearch runs:

- `prepare_mlx_mamba_3.py`: fixed prep entry point for `mamba-2` and `mamba-3`
- `train_mamba_3_mlx.py`: editable architecture-aware MLX training script
- `program.md`: autonomous keep/discard loop instructions

Additional reference files remain in the repo:

- `prepare_mlx.py`
- `train_mamba_mlx.py`
- `prepare.py`
- `train_mamba.py`

## Repository Layout

- `program.md`: autoresearch instructions and experiment loop
- `prepare_mlx_mamba_3.py`: fixed architecture-aware MLX prep entry point
- `train_mamba_3_mlx.py`: editable architecture-aware MLX training surface
- `prepare_mlx.py`: shared tokenizer, dataloader, evaluator
- `train_mamba_mlx.py`: original MLX Mamba-2 training script
- `analysis.ipynb`: notebook for analyzing `results.tsv`
- `pyproject.toml`: Python dependencies

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The MLX path requires Apple Silicon and a working MLX installation.

## Quick Start

### Mamba-3 Full Baseline

```bash
python3 prepare_mlx_mamba_3.py
python3 train_mamba_3_mlx.py
```

By default, the architecture-aware path targets `mamba-3`.

### Mamba-3 With A Local Preset

```bash
AUTORESEARCH_MLX_PRESET_FILE=/path/to/your-local-mamba-3-preset.json python3 prepare_mlx_mamba_3.py
AUTORESEARCH_MLX_PRESET_FILE=/path/to/your-local-mamba-3-preset.json python3 train_mamba_3_mlx.py
```

If the preset carries `ARCHITECTURE=mamba-3`, the prep wrapper can infer the architecture directly from the preset file.

### Mamba-2 From The New MLX Path

```bash
AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 python3 prepare_mlx_mamba_3.py
AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 python3 train_mamba_3_mlx.py
```

### Mamba-2 With A Local Preset

```bash
AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 AUTORESEARCH_MLX_PRESET_FILE=/path/to/your-local-mamba-2-preset.json python3 prepare_mlx_mamba_3.py
AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 AUTORESEARCH_MLX_PRESET_FILE=/path/to/your-local-mamba-2-preset.json python3 train_mamba_3_mlx.py
```

## Presets

The MLX path has two built-in default presets from the scripts themselves:

- default Mamba-3 preset: run `prepare_mlx_mamba_3.py` and `train_mamba_3_mlx.py` without `AUTORESEARCH_MLX_PRESET_FILE`; this is the default architecture-aware setup
- default Mamba-2 preset: set `AUTORESEARCH_MLX_ARCHITECTURE=mamba-2` without `AUTORESEARCH_MLX_PRESET_FILE`; this uses the built-in Mamba-2 defaults in the same path

Local presets are fixed infrastructure for a run:

- do not edit the active preset after the run starts
- do not mix results from different presets in the same `results.tsv`
- do not compare `val_bpb` directly across different evaluation setups
- point `AUTORESEARCH_MLX_PRESET_FILE` at the local preset you want to use for the full run

Example local Mamba-3 preset:

```json
{
  "prepare_mlx": {
    "MAX_SEQ_LEN": 512,
    "EVAL_TOKENS": 524288,
    "DEFAULT_NUM_SHARDS": 4
  },
  "train_mamba_3_mlx": {
    "ARCHITECTURE": "mamba-3",
    "DEPTH": 4,
    "D_MODEL": 384,
    "D_STATE": 32,
    "HEADDIM": 32,
    "TOTAL_BATCH_SIZE": 16384,
    "DEVICE_BATCH_SIZE": 4,
    "ROPE_FRACTION": 0.5,
    "DT_MIN": 0.001,
    "DT_MAX": 0.1,
    "DT_INIT_FLOOR": 0.0001,
    "A_FLOOR": 0.0001,
    "IS_OUTPROJ_NORM": false,
    "IS_MIMO": false,
    "MIMO_RANK": 4
  }
}
```

Example local Mamba-2 preset:

```json
{
  "prepare_mlx": {
    "MAX_SEQ_LEN": 512,
    "EVAL_TOKENS": 524288,
    "DEFAULT_NUM_SHARDS": 4
  },
  "train_mamba_3_mlx": {
    "ARCHITECTURE": "mamba-2",
    "DEPTH": 6,
    "D_MODEL": 512,
    "D_STATE": 64,
    "D_CONV": 4,
    "HEADDIM": 64,
    "TOTAL_BATCH_SIZE": 16384,
    "DEVICE_BATCH_SIZE": 4
  }
}
```

## Running The Agent

Open your coding agent in this repo, point it at `program.md`, and keep the architecture and preset fixed for the full run.

Example prompt:

```text
Read program.md and start the MLX autoresearch loop with AUTORESEARCH_MLX_PRESET_FILE set to your local Mamba-3 preset. Treat the preset and architecture as fixed infrastructure, log results to results.tsv, and keep or discard changes based on val_bpb.
```

## What Changes During A Run

The intended editable file is:

- `train_mamba_3_mlx.py`

Typical experiment directions:

- model depth, width, `d_state`, `headdim`, `ngroups`
- chunk size and gradient accumulation
- optimizer, learning rate schedule, weight decay, Adam betas
- Mamba-2-specific settings like `d_conv`
- Mamba-3-specific settings like `rope_fraction`, `dt` range, `a_floor`, `is_outproj_norm`, `is_mimo`, and `mimo_rank`

The evaluation harness and active preset stay fixed for the run.

## Output

A successful run prints:

```text
architecture
val_bpb
training_seconds
total_seconds
total_tokens_M
num_steps
num_params_M
depth
```

Lower `val_bpb` is better.
