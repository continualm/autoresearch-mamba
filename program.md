# Autoresearch: Mamba-2 / Mamba-3 (MLX)

This repo runs Karpathy-style autoresearch on Apple Silicon with a fixed MLX evaluator and an editable training script.

For the current MLX path, the active architecture for a run is fixed by `AUTORESEARCH_MLX_ARCHITECTURE` or by the active preset and must not change mid-run:

- `mamba-2`
- `mamba-3`

The architecture-aware MLX entry points are:

- `prepare_mlx_mamba_3.py`
- `train_mamba_3_mlx.py`

`prepare_mlx.py` and `train_mamba_mlx.py` remain useful legacy references, but new MLX autoresearch runs should use the architecture-aware pair above.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today (for example `mar21`). The branch `autoresearch/<tag>` must be new.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Choose the fixed infrastructure**: lock the architecture and preset before the first baseline.
4. **Read the in-scope files**:
   - `prepare_mlx_mamba_3.py` — fixed prep entry point. Do not modify during a run.
   - `train_mamba_3_mlx.py` — editable architecture-aware training script.
   - `prepare_mlx.py` — shared tokenizer, dataloader, and BPB evaluator.
5. **Verify data exists**: check `~/.cache/autoresearch/`. If data or tokenizer artifacts are missing, run exactly one prep command and keep the same architecture/preset for the entire run:
   - Mamba-3 full baseline: `python prepare_mlx_mamba_3.py`
   - Mamba-3 local preset: `AUTORESEARCH_MLX_PRESET_FILE=mlx_mamba_3_preset.local.json python prepare_mlx_mamba_3.py`
   - Mamba-2 from the new path: `AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 python prepare_mlx_mamba_3.py`
6. **Initialize `results.tsv`** with only the header row. Do not mix architectures or preset/evaluation setups in the same log.
7. **Confirm and start** once the setup is coherent.

## Experimentation

Each experiment runs on Apple Silicon with a fixed **5 minute** training budget. Launch training with the same architecture and preset combination used for prep and the baseline:

- Mamba-3 full baseline: `python train_mamba_3_mlx.py`
- Mamba-3 local preset: `AUTORESEARCH_MLX_PRESET_FILE=mlx_mamba_3_preset.local.json python train_mamba_3_mlx.py`
- Mamba-2 from the new path: `AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 python train_mamba_3_mlx.py`

`mlx_mamba_3_preset.local.json` is the dedicated local Mamba-3 preset. `mlx_preset.local.json` remains available for older or legacy local MLX runs. Whichever preset you pick becomes fixed infrastructure for that run. Do not edit it mid-run, and do not compare `val_bpb` across different preset/evaluation setups.

The architecture choice is also fixed infrastructure. Do not switch between `mamba-2` and `mamba-3` inside the same branch or `results.tsv` history.

**What you CAN do:**
- Modify `train_mamba_3_mlx.py`. It is the only intended experiment surface during the loop. It supports both `mamba-2` and `mamba-3`.
- Tune model size, optimizer, schedule, state dimensions, chunking, grouping, MLP usage, and Mamba-3-specific hyperparameters.

**What you CANNOT do:**
- Modify `prepare_mlx_mamba_3.py` or `prepare_mlx.py` during an active run.
- Modify the active preset file after the baseline has started.
- Switch the architecture after baseline.
- Add dependencies or change the evaluation harness.

The objective is simple: minimize `val_bpb` under the fixed evaluator and fixed time budget.

## Output Format

A completed run prints a summary like this:

```text
---
architecture:     <mamba-2|mamba-3>
val_bpb:          <float>
training_seconds: <float>
total_seconds:    <float>
total_tokens_M:   <float>
num_steps:        <int>
num_params_M:     <float>
depth:            <int>
```

Useful extraction commands:

```bash
grep "^architecture:" run.log
grep "^val_bpb:" run.log
```

## Logging Results

Log every experiment to `results.tsv` as tab-separated text with this header:

```text
commit	val_bpb	status	description
```

Columns:
1. short git commit hash
2. `val_bpb` (`0.000000` for crashes)
3. `keep`, `discard`, or `crash`
4. short experiment description

Do not mix architectures or preset/evaluation setups in the same results log.

## Experiment Loop

Once setup is complete, the loop is:

1. Check the current branch and commit.
2. Edit `train_mamba_3_mlx.py` with one experimental idea.
3. `git add train_mamba_3_mlx.py && git commit -m "experiment: <description>"`
4. Run the same command shape used for the baseline and redirect output to `run.log`.
   - `python train_mamba_3_mlx.py > run.log 2>&1`
   - `AUTORESEARCH_MLX_PRESET_FILE=mlx_mamba_3_preset.local.json python train_mamba_3_mlx.py > run.log 2>&1`
   - `AUTORESEARCH_MLX_ARCHITECTURE=mamba-2 python train_mamba_3_mlx.py > run.log 2>&1`
5. Read the result with `grep "^val_bpb:" run.log`.
6. If no metric is printed, inspect `tail -n 50 run.log`, classify the run as a crash, and either fix the bug or revert the broken idea.
7. Append the result to `results.tsv`.
8. Keep the commit only if `val_bpb` improved.
9. Revert losing ideas.

If a run exceeds 10 minutes total, kill it and treat it as a failure.

## Experiment Ideas

### Shared
- `d_state`, `expand`, `headdim`, `ngroups`
- learning rate, weight decay, Adam betas, warmup/warmdown ratios
- batch size vs gradient accumulation
- chunk size
- residual scaling and output projection initialization
- optional `GatedMLP`

### Mamba-2
- `d_conv`
- `dt` init range
- `A` init range
- normalization and gating variations

### Mamba-3
- `rope_fraction`
- `dt_min`, `dt_max`, `dt_init_floor`, `a_floor`
- `B/C` norm and bias behavior
- `is_outproj_norm`
- `is_mimo` and `mimo_rank`

## Never Stop

After setup and baseline, do not pause the autoresearch loop unless the human interrupts it.
