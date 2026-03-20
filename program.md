# Autoresearch: Mamba2 (MLX)

This is an experiment to have the LLM do its own research, optimizing a Mamba2 language model on Apple Silicon.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare_mlx.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train_mamba_mlx.py` — the file you modify. Mamba2 model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, run one prep command and keep that preset choice fixed for the entire run:
   - Full tracked baseline: `python prepare_mlx.py`
   - Local Apple Silicon preset: `AUTORESEARCH_MLX_PRESET_FILE=mlx_preset.local.json python prepare_mlx.py`
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on Apple Silicon (MLX). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch it with the same preset command you used for prep and baseline:

- Full tracked baseline: `python train_mamba_mlx.py`
- Local Apple Silicon preset: `AUTORESEARCH_MLX_PRESET_FILE=mlx_preset.local.json python train_mamba_mlx.py`

`mlx_preset.local.json` is an opt-in local testing preset, ignored by git. If you use it, treat it as fixed infrastructure for the whole run. Do not edit it mid-run, and do not compare its `val_bpb` directly against runs produced with the full preset because `MAX_SEQ_LEN` and `EVAL_TOKENS` differ.

**What you CAN do:**
- Modify `train_mamba_mlx.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare_mlx.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- If you are using `AUTORESEARCH_MLX_PRESET_FILE`, do not modify the preset file after the run has started. It is part of the fixed setup for that run.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare_mlx.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Apple Silicon has unified memory — some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the chosen preset command as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          <float>
training_seconds: <float>
total_seconds:    <float>
total_tokens_M:   <float>
num_steps:        <int>
num_params_M:     <float>
depth:            <int>
```

Note that the script is configured to always stop after 5 minutes. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	val_bpb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	val_bpb	status	description
a1b2c3d	1.150000	keep	baseline
b2c3d4e	1.143200	keep	increase LR to 1e-3
c3d4e5f	1.165000	discard	switch to GeLU activation
d4e5f6g	0.000000	crash	double model width (OOM)
```

Do not mix results from different preset/evaluation setups in the same `results.tsv`. Use one branch/run log per preset choice.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar19`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_mamba_mlx.py` with an experimental idea by directly hacking the code.
3. `git add train_mamba_mlx.py && git commit -m "experiment: <description>"`
4. Run the experiment with the same preset command used for the baseline, redirecting everything so output does not flood your context:
   - Full tracked baseline: `python train_mamba_mlx.py > run.log 2>&1`
   - Local Apple Silicon preset: `AUTORESEARCH_MLX_PRESET_FILE=mlx_preset.local.json python train_mamba_mlx.py > run.log 2>&1`
5. Read out the results: `grep "^val_bpb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

## Experiment ideas (SSM-specific)

The tracked defaults are the full MLX baseline. `mlx_preset.local.json` is a reduced Apple Silicon local-testing preset. Pick one preset at setup time, keep it fixed for the entire run, and explore from that active baseline:

### Architecture
- State dimension (d_state): try 32, 64, 128, 256
- Expand factor: try 1.5, 2, 3
- Head dimension: try 32, 64, 128
- Conv kernel width (d_conv): try 2, 4, 8
- Number of groups (ngroups): try 1, 2, 4
- Add GatedMLP (set d_intermediate > 0) — the baseline has no MLP
- Add residual scaling (per-layer lambdas like Karpathy's GPT)
- Try different activation functions (SiLU vs ReLU² vs GELU)
- Untie embedding weights (separate lm_head)

### Initialization
- A matrix init range: try (1, 4), (1, 16), (1, 64)
- dt init range: try different (dt_min, dt_max)
- Zero-init output projections
- Different embedding initialization scales
- Remove or tune the out_proj 1/sqrt(n_layer) scaling

### Optimization
- Learning rate: try 1e-4, 3e-4, 1e-3, 3e-3
- Weight decay: try 0, 0.01, 0.1, 0.2
- Adam betas: try (0.9, 0.95), (0.9, 0.999), (0.8, 0.95)
- LR schedule (warmup/warmdown ratios)
- Gradient clipping
- Batch size vs gradient accumulation tradeoffs
- Keep dt_bias, A_log, D excluded from weight decay unless you are explicitly testing that ablation

### Training
- Model dimension vs depth tradeoff (more layers vs wider layers)
- SSD chunk size: try 32, 64, 128, 256
- Mixed precision strategies

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
