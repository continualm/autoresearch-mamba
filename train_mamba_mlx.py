"""
Autoresearch Mamba pretraining script. Apple Silicon (MLX), single-file.
Pure Mamba2 implementation with SSD (Structured State Space Duality).
Usage:
    python train_mamba_mlx.py
    AUTORESEARCH_MLX_PRESET_FILE=mlx_preset.local.json python train_mamba_mlx.py
"""

import math
import os
import time
import json
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from prepare_mlx import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

PRESET_FILE_ENV = "AUTORESEARCH_MLX_PRESET_FILE"


def load_preset(section):
    preset_path = os.environ.get(PRESET_FILE_ENV)
    if not preset_path:
        return {}
    if not os.path.isabs(preset_path):
        preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), preset_path)
    with open(preset_path, "r", encoding="utf-8") as f:
        preset = json.load(f)
    section_preset = preset.get(section, {})
    if not isinstance(section_preset, dict):
        raise ValueError(f"Preset section {section!r} must be an object")
    print(f"Preset overrides: loaded {section} from {preset_path}")
    return section_preset


PRESET = load_preset("train_mamba_mlx")

# ---------------------------------------------------------------------------
# Mamba2 Model
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    vocab_size: int = 8192
    d_model: int = 768
    n_layer: int = 8
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    d_intermediate: int = 0  # GatedMLP hidden dim (0 = no MLP, matching vanilla Mamba2)
    pad_vocab_size_multiple: int = 8


# ---------------------------------------------------------------------------
# SSD: Structured State Space Duality (Mamba2 core algorithm)
# ---------------------------------------------------------------------------

def segsum(x):
    """Stable segment sum calculation for SSD.
    x: (..., T) -> (..., T, T) lower-triangular cumulative sums.
    """
    T = x.shape[-1]
    # Expand: (..., T) -> (..., T, T)
    x = mx.repeat(mx.expand_dims(x, -1), T, axis=-1)
    # Zero out upper triangle (keep below diagonal)
    mask = mx.tril(mx.ones((T, T)), k=-1).astype(mx.bool_)
    x = mx.where(mask, x, mx.zeros_like(x))
    # Cumulative sum along rows
    x_segsum = mx.cumsum(x, axis=-2)
    # Mask: keep lower triangle (including diagonal), set rest to -inf
    mask2 = mx.tril(mx.ones((T, T)), k=0).astype(mx.bool_)
    x_segsum = mx.where(mask2, x_segsum, mx.full(x_segsum.shape, -1e9))
    return x_segsum


def ssd_scan(X, A, B, C, block_len=64):
    """
    Structured State Space Duality scan — Mamba2 core.

    Arguments:
        X: (batch, length, n_heads, d_head)    — input (already scaled by dt)
        A: (batch, length, n_heads)             — log decay (already scaled by dt)
        B: (batch, length, n_heads, d_state)    — input matrix
        C: (batch, length, n_heads, d_state)    — output matrix
        block_len: chunk size for the SSD algorithm

    Returns:
        Y: (batch, length, n_heads, d_head)
    """
    batch, seqlen, nheads, headdim = X.shape
    d_state = B.shape[-1]

    # Pad sequence to multiple of block_len if needed
    pad_len = (block_len - seqlen % block_len) % block_len
    if pad_len > 0:
        X = mx.pad(X, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        A = mx.pad(A, [(0, 0), (0, pad_len), (0, 0)])
        B = mx.pad(B, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        C = mx.pad(C, [(0, 0), (0, pad_len), (0, 0), (0, 0)])

    L = X.shape[1]
    nblocks = L // block_len

    # Rearrange into blocks: (batch, nblocks, block_len, ...)
    X = X.reshape(batch, nblocks, block_len, nheads, headdim)
    A = A.reshape(batch, nblocks, block_len, nheads)
    B = B.reshape(batch, nblocks, block_len, nheads, d_state)
    C = C.reshape(batch, nblocks, block_len, nheads, d_state)

    # Transpose A for cumsum: (batch, nheads, nblocks, block_len)
    A = mx.transpose(A, (0, 3, 1, 2))
    A_cumsum = mx.cumsum(A, axis=-1)

    # 1. Intra-chunk (diagonal blocks): attention-like within each chunk
    # L_matrix: (batch, nheads, nblocks, block_len, block_len) — lower triangular decay
    L_matrix = mx.exp(segsum(A))
    # Y_diag = einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L_matrix, X)
    # Rearrange for matmul: compute CB = C @ B^T per chunk, masked by L
    # CB: (batch, nblocks, nheads, block_len, block_len)
    CB = mx.einsum("bclhn,bcshn->bhcls", C, B)
    # Masked CB
    CB_masked = CB * L_matrix  # (batch, nheads, nblocks, block_len, block_len)
    # Rearrange X: (batch, nblocks, block_len, nheads, headdim) -> (batch, nheads, nblocks, block_len, headdim)
    X_t = mx.transpose(X, (0, 3, 1, 2, 4))
    # Y_diag: (batch, nheads, nblocks, block_len, headdim)
    Y_diag = mx.einsum("bhcls,bhcsp->bhclp", CB_masked, X_t)

    # 2. Inter-chunk states: accumulate SSM states at chunk boundaries
    # decay_states: (batch, nheads, nblocks, block_len) — decay from each position to end of chunk
    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # B_t: (batch, nblocks, block_len, nheads, d_state) -> (batch, nheads, nblocks, block_len, d_state)
    B_t = mx.transpose(B, (0, 3, 1, 2, 4))
    # states: (batch, nblocks, nheads, headdim, d_state)
    states = mx.transpose(
        mx.einsum("bhcl,bhcln,bhclp->bhcpn", decay_states, B_t, X_t),
        (0, 2, 1, 3, 4)
    )

    # 3. Propagate states across chunks via recurrence
    # Initial state is zeros
    initial = mx.zeros((batch, 1, nheads, headdim, d_state))
    states = mx.concatenate([initial, states], axis=1)  # (batch, nblocks+1, nheads, headdim, d_state)

    # Decay between chunk boundaries
    # A_cumsum[:, :, :, -1]: (batch, nheads, nblocks) — total decay per chunk
    chunk_decay = mx.pad(A_cumsum[:, :, :, -1], [(0, 0), (0, 0), (1, 0)])  # (batch, nheads, nblocks+1)
    decay_chunk = mx.exp(segsum(chunk_decay))  # (batch, nheads, nblocks+1, nblocks+1)

    # Propagate: new_states[z] = sum_c decay[z,c] * states[c]
    # states: (batch, nblocks+1, nheads, headdim, d_state) -> (batch, nheads, nblocks+1, headdim, d_state)
    states_t = mx.transpose(states, (0, 2, 1, 3, 4))
    new_states = mx.einsum("bhzc,bhcpn->bhzpn", decay_chunk, states_t)
    # Take states 0..nblocks-1 (input to each chunk)
    new_states = new_states[:, :, :-1]  # (batch, nheads, nblocks, headdim, d_state)

    # 4. State-to-output: convert accumulated states to output per position
    state_decay_out = mx.exp(A_cumsum)  # (batch, nheads, nblocks, block_len)
    # C_t: (batch, nheads, nblocks, block_len, d_state)
    C_t = mx.transpose(C, (0, 3, 1, 2, 4))
    Y_off = mx.einsum("bhcln,bhcpn,bhcl->bhclp", C_t, new_states, state_decay_out)

    # 5. Combine diagonal + off-diagonal
    Y = Y_diag + Y_off  # (batch, nheads, nblocks, block_len, headdim)

    # Rearrange back: (batch, nheads, nblocks, block_len, headdim) -> (batch, seqlen, nheads, headdim)
    Y = mx.transpose(Y, (0, 2, 3, 1, 4))  # (batch, nblocks, block_len, nheads, headdim)
    Y = Y.reshape(batch, L, nheads, headdim)

    # Remove padding
    if pad_len > 0:
        Y = Y[:, :seqlen]

    return Y


class GroupedRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5, group_size=None):
        super().__init__()
        self.eps = eps
        self.group_size = d if group_size is None else group_size
        assert d % self.group_size == 0
        self.weight = mx.ones((d,))

    def __call__(self, x):
        grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
        rms = mx.rsqrt(mx.mean(grouped * grouped, axis=-1, keepdims=True) + self.eps)
        return (grouped * rms).reshape(x.shape) * self.weight


# ---------------------------------------------------------------------------
# Mamba2 Block
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2,
                 headdim=64, ngroups=1, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        assert self.d_inner % headdim == 0
        assert self.d_inner % self.ngroups == 0
        assert self.nheads % self.ngroups == 0

        # Input projection: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # Depthwise conv on [x, B, C]
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            bias=True,
            padding=d_conv - 1,
        )

        # dt bias (initialized via inverse softplus of uniform in [dt_min, dt_max])
        dt_min, dt_max = 0.001, 0.1
        dt = np.exp(np.random.uniform(size=self.nheads) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min))
        dt = np.clip(dt, a_min=1e-4, a_max=None)
        inv_dt = dt + np.log(-np.expm1(-dt))
        self.dt_bias = mx.array(inv_dt, dtype=mx.float32)

        # A parameter (log space, negative for stability)
        A = np.random.uniform(1.0, 16.0, size=self.nheads)
        self.A_log = mx.array(np.log(A), dtype=mx.float32)

        # D skip connection
        self.D = mx.array(np.ones(self.nheads), dtype=mx.float32)

        # Output norm + projection
        self.norm = GroupedRMSNorm(self.d_inner, group_size=self.d_inner // self.ngroups)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, u):
        """
        u: (batch, length, d_model)
        Returns: (batch, length, d_model)
        """
        batch, seqlen, _ = u.shape

        # Input projection
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)

        # Split: [z | xBC | dt]
        d_inner = self.d_inner
        d_xBC = d_inner + 2 * self.ngroups * self.d_state
        z = zxbcdt[:, :, :d_inner]
        xBC = zxbcdt[:, :, d_inner:d_inner + d_xBC]
        dt = zxbcdt[:, :, d_inner + d_xBC:]  # (B, L, nheads)

        # Depthwise Conv1d on xBC
        # MLX Conv1d expects (B, L, C)
        # Transpose to (B, C, L) for standard conv, then back
        # Actually MLX Conv1d uses (B, L, C) natively — but we need causal padding
        xBC = self.conv1d(xBC)[:, :seqlen, :]  # causal: take first seqlen outputs
        xBC = nn.silu(xBC)

        # Split xBC into x, B, C
        x = xBC[:, :, :d_inner]
        B_ssm = xBC[:, :, d_inner:d_inner + self.ngroups * self.d_state]
        C_ssm = xBC[:, :, d_inner + self.ngroups * self.d_state:]

        # Reshape for multihead
        x = x.reshape(batch, seqlen, self.nheads, self.headdim)

        # For ngroups: expand B,C to match nheads
        # B: (B, L, ngroups, d_state) — if ngroups < nheads, repeat
        B_ssm = B_ssm.reshape(batch, seqlen, self.ngroups, self.d_state)
        C_ssm = C_ssm.reshape(batch, seqlen, self.ngroups, self.d_state)
        if self.ngroups < self.nheads:
            repeats = self.nheads // self.ngroups
            B_ssm = mx.repeat(B_ssm, repeats, axis=2)
            C_ssm = mx.repeat(C_ssm, repeats, axis=2)

        # dt: softplus(dt + dt_bias)
        dt = nn.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # A: negative exponential
        A = -mx.exp(self.A_log)  # (nheads,)

        # Scale inputs by dt for SSD formulation: x_scaled = x * dt, A_scaled = A * dt
        x_scaled = x * mx.expand_dims(dt, -1)  # (B, L, nheads, headdim)
        A_scaled = A[None, None, :] * dt  # (B, L, nheads)

        # SSD scan
        y = ssd_scan(x_scaled, A_scaled, B_ssm, C_ssm, block_len=self.chunk_size)
        # y: (B, L, nheads, headdim)

        # D skip connection
        y = y + mx.expand_dims(self.D, -1) * x  # (B, L, nheads, headdim)

        # Reshape back to (B, L, d_inner)
        y = y.reshape(batch, seqlen, d_inner)

        # Gate then norm (norm_before_gate=False, matching reference)
        y = self.norm(y * nn.silu(z))

        # Output projection
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# GatedMLP
# ---------------------------------------------------------------------------

class GatedMLP(nn.Module):
    def __init__(self, d_model, d_hidden=0):
        super().__init__()
        if d_hidden == 0:
            d_hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)

    def __call__(self, x):
        y = self.fc1(x)
        y, gate = mx.split(y, 2, axis=-1)
        return self.fc2(y * nn.silu(gate))


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm_ssm = nn.RMSNorm(config.d_model)
        self.ssm = Mamba2Block(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            ngroups=config.ngroups,
        )
        # MLP: only added when d_intermediate > 0 (vanilla Mamba2 has no MLP)
        self.has_mlp = config.d_intermediate > 0
        if self.has_mlp:
            self.norm_mlp = nn.RMSNorm(config.d_model)
            self.mlp = GatedMLP(config.d_model, config.d_intermediate)

    def __call__(self, x):
        x = x + self.ssm(self.norm_ssm(x))
        if self.has_mlp:
            x = x + self.mlp(self.norm_mlp(x))
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class MambaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [ResidualBlock(config, i) for i in range(config.n_layer)]
        self.norm_f = nn.RMSNorm(config.d_model)
        # No separate lm_head — weight-tied with embedding (reference default)

        # Reference init: embedding normal(std=0.02), residual projections scaled by
        # 1/sqrt(n_layer) without MLP and 1/sqrt(2 * n_layer) with MLP.
        self.embedding.weight = mx.random.normal(shape=self.embedding.weight.shape) * 0.02
        residual_scale = math.sqrt(config.n_layer * (2 if config.d_intermediate > 0 else 1))
        for layer in self.layers:
            proj = layer.ssm.out_proj.weight
            layer.ssm.out_proj.weight = proj / residual_scale
            if layer.has_mlp:
                layer.mlp.fc2.weight = layer.mlp.fc2.weight / residual_scale

    def __call__(self, idx, targets=None):
        """
        idx: (batch, seqlen) token indices
        targets: optional (batch, seqlen) for loss computation
        Returns: logits (batch, seqlen, vocab_size) or loss scalar
        """
        x = self.embedding(idx)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        # Weight-tied lm_head: logits = x @ embedding.weight.T
        logits = x @ self.embedding.weight.T

        if targets is not None:
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction='mean'
            )
            return loss
        return logits

    def count_params(self):
        nparams = sum(p.size for _, p in mlx.utils.tree_flatten(self.parameters()))
        return nparams

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = int(PRESET.get("DEPTH", 8))
D_MODEL = int(PRESET.get("D_MODEL", 768))
D_STATE = int(PRESET.get("D_STATE", 64))
D_CONV = 4               # conv kernel width
EXPAND = 2               # inner dim expansion factor
HEADDIM = int(PRESET.get("HEADDIM", 64))

# Optimization
TOTAL_BATCH_SIZE = int(PRESET.get("TOTAL_BATCH_SIZE", 2**18))
LEARNING_RATE = 3e-4      # AdamW learning rate
WEIGHT_DECAY = 0.1        # AdamW weight decay
ADAM_BETAS = (0.9, 0.95)  # Adam betas
WARMUP_RATIO = 0.05       # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3      # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.1       # final LR as fraction of initial

# Batch size
DEVICE_BATCH_SIZE = int(PRESET.get("DEVICE_BATCH_SIZE", 16))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mx.random.seed(42)
np.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Pad vocab to multiple
pad_multiple = 8
if vocab_size % pad_multiple != 0:
    vocab_size = ((vocab_size + pad_multiple - 1) // pad_multiple) * pad_multiple
    print(f"Padded vocab size: {vocab_size:,}")

config = MambaConfig(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layer=DEPTH,
    d_state=D_STATE,
    d_conv=D_CONV,
    expand=EXPAND,
    headdim=HEADDIM,
)
print(f"Model config: {asdict(config)}")

model = MambaLM(config)
mx.eval(model.parameters())

num_params = model.count_params()
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# Optimizer
no_weight_decay_names = ("dt_bias", "A_log", "D")
optimizer = optim.MultiOptimizer(
    [
        optim.AdamW(
            learning_rate=LEARNING_RATE,
            betas=ADAM_BETAS,
            weight_decay=0.0,
        ),
        optim.AdamW(
            learning_rate=LEARNING_RATE,
            betas=ADAM_BETAS,
            weight_decay=WEIGHT_DECAY,
        ),
    ],
    filters=[lambda path, _: any(path.endswith(suffix) for suffix in no_weight_decay_names)],
)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Tokens per step: {TOTAL_BATCH_SIZE:,}")

# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Training step (compiled)
# ---------------------------------------------------------------------------

def loss_fn(model, x, y):
    return model(x, y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

print("\nTraining started...")

while True:
    t0 = time.time()

    # Gradient accumulation
    total_loss = 0.0
    acc_grads = None
    for micro_step in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        loss, grads = loss_and_grad_fn(model, x, y)
        if acc_grads is None:
            acc_grads = grads
        else:
            acc_grads = mlx.utils.tree_map(lambda a, b: a + b, acc_grads, grads)
        total_loss += loss.item()
        mx.eval(loss)

    # Average gradients
    if grad_accum_steps > 1:
        acc_grads = mlx.utils.tree_map(lambda g: g / grad_accum_steps, acc_grads)

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    optimizer.learning_rate = mx.array(LEARNING_RATE * lrm)

    # Apply gradients
    model.update(optimizer.apply_gradients(acc_grads, model))
    mx.eval(model.parameters(), optimizer.state)

    t1 = time.time()
    dt = t1 - t0

    train_loss_f = total_loss / grad_accum_steps

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after training log

total_tokens = step * TOTAL_BATCH_SIZE

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

print("Evaluating...")
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
