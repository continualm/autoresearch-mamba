"""
Autoresearch Mamba pretraining script. PyTorch/CUDA, single-file.
Pure Mamba2 implementation with SSD (Structured State Space Duality).
Usage: python train_mamba.py
"""

import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

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
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask2 = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask2, -torch.inf)
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
        X = F.pad(X, (0, 0, 0, 0, 0, pad_len))
        A = F.pad(A, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))

    L = X.shape[1]
    nblocks = L // block_len

    # Rearrange into blocks: (batch, nblocks, block_len, ...)
    X = X.reshape(batch, nblocks, block_len, nheads, headdim)
    A = A.reshape(batch, nblocks, block_len, nheads)
    B = B.reshape(batch, nblocks, block_len, nheads, d_state)
    C = C.reshape(batch, nblocks, block_len, nheads, d_state)

    # Transpose A for cumsum: (batch, nheads, nblocks, block_len)
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Intra-chunk (diagonal blocks): attention-like within each chunk
    L_matrix = torch.exp(segsum(A))
    CB = torch.einsum("bclhn,bcshn->bhcls", C, B)
    CB_masked = CB * L_matrix
    X_t = X.permute(0, 3, 1, 2, 4)
    Y_diag = torch.einsum("bhcls,bhcsp->bhclp", CB_masked, X_t)

    # 2. Inter-chunk states: accumulate SSM states at chunk boundaries
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_t = B.permute(0, 3, 1, 2, 4)
    states = torch.einsum("bhcl,bhcln,bhclp->bhcpn", decay_states, B_t, X_t)
    states = states.permute(0, 2, 1, 3, 4)

    # 3. Propagate states across chunks via recurrence
    initial = torch.zeros(batch, 1, nheads, headdim, d_state, device=X.device, dtype=X.dtype)
    states = torch.cat([initial, states], dim=1)

    chunk_decay = F.pad(A_cumsum[:, :, :, -1], (1, 0))
    decay_chunk = torch.exp(segsum(chunk_decay))

    states_t = states.permute(0, 2, 1, 3, 4)
    new_states = torch.einsum("bhzc,bhcpn->bhzpn", decay_chunk, states_t)
    new_states = new_states[:, :, :-1]

    # 4. State-to-output: convert accumulated states to output per position
    state_decay_out = torch.exp(A_cumsum)
    C_t = C.permute(0, 3, 1, 2, 4)
    Y_off = torch.einsum("bhcln,bhcpn,bhcl->bhclp", C_t, new_states, state_decay_out)

    # 5. Combine diagonal + off-diagonal
    Y = Y_diag + Y_off
    Y = Y.permute(0, 2, 3, 1, 4)
    Y = Y.reshape(batch, L, nheads, headdim)

    # Remove padding
    if pad_len > 0:
        Y = Y[:, :seqlen]

    return Y


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5, group_size=None):
        super().__init__()
        self.eps = eps
        self.group_size = d if group_size is None else group_size
        assert d % self.group_size == 0
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x_grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
        x_grouped = x_grouped * torch.rsqrt(x_grouped.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_grouped.reshape_as(x) * self.weight


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
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # A parameter (log space, negative for stability)
        A = torch.empty(self.nheads).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.nheads))

        # Output norm + projection
        self.norm = RMSNorm(self.d_inner, group_size=self.d_inner // self.ngroups)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
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
        # PyTorch Conv1d: (B, C, L) — need to transpose
        xBC = F.silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
        )

        # Split xBC into x, B, C
        x = xBC[:, :, :d_inner]
        B_ssm = xBC[:, :, d_inner:d_inner + self.ngroups * self.d_state]
        C_ssm = xBC[:, :, d_inner + self.ngroups * self.d_state:]

        # Reshape for multihead
        x = x.reshape(batch, seqlen, self.nheads, self.headdim)

        # For ngroups: expand B,C to match nheads
        B_ssm = B_ssm.reshape(batch, seqlen, self.ngroups, self.d_state)
        C_ssm = C_ssm.reshape(batch, seqlen, self.ngroups, self.d_state)
        if self.ngroups < self.nheads:
            repeats = self.nheads // self.ngroups
            B_ssm = B_ssm.repeat_interleave(repeats, dim=2)
            C_ssm = C_ssm.repeat_interleave(repeats, dim=2)

        # dt: softplus(dt + dt_bias)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # A: negative exponential
        A = -torch.exp(self.A_log)  # (nheads,)

        # Scale inputs by dt for SSD formulation
        x_scaled = x * dt.unsqueeze(-1)  # (B, L, nheads, headdim)
        A_scaled = A.unsqueeze(0).unsqueeze(0) * dt  # (B, L, nheads)

        # SSD scan
        y = ssd_scan(x_scaled, A_scaled, B_ssm, C_ssm, block_len=self.chunk_size)

        # D skip connection
        y = y + self.D.unsqueeze(-1) * x

        # Reshape back to (B, L, d_inner)
        y = y.reshape(batch, seqlen, d_inner)

        # Gate then norm (norm_before_gate=False, matching reference)
        y = self.norm(y * F.silu(z))

        # Output projection
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# GatedMLP (available for experiments, not used in vanilla Mamba2 baseline)
# ---------------------------------------------------------------------------

class GatedMLP(nn.Module):
    def __init__(self, d_model, d_hidden=0):
        super().__init__()
        if d_hidden == 0:
            d_hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm_ssm = RMSNorm(config.d_model)
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
            self.norm_mlp = RMSNorm(config.d_model)
            self.mlp = GatedMLP(config.d_model, config.d_intermediate)

    def forward(self, x):
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
        self.layers = nn.ModuleList([ResidualBlock(config, i) for i in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        # No separate lm_head — weight-tied with embedding (reference default)

        # Reference init: embedding normal(std=0.02), residual projections scaled by
        # 1/sqrt(n_layer) without MLP and 1/sqrt(2 * n_layer) with MLP.
        nn.init.normal_(self.embedding.weight, std=0.02)
        residual_scale = math.sqrt(config.n_layer * (2 if config.d_intermediate > 0 else 1))
        for layer in self.layers:
            with torch.no_grad():
                layer.ssm.out_proj.weight.div_(residual_scale)
                if layer.has_mlp:
                    layer.mlp.fc2.weight.div_(residual_scale)

    def forward(self, idx, targets=None):
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
        logits = F.linear(x, self.embedding.weight)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            return loss
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 8                # number of Mamba layers
D_MODEL = 768            # model dimension
D_STATE = 64             # SSM state dimension
D_CONV = 4               # conv kernel width
EXPAND = 2               # inner dim expansion factor
HEADDIM = 64             # head dimension

# Optimization
TOTAL_BATCH_SIZE = 2**18  # ~262K tokens per step
LEARNING_RATE = 3e-4      # AdamW learning rate
WEIGHT_DECAY = 0.1        # AdamW weight decay
ADAM_BETAS = (0.9, 0.95)  # Adam betas
WARMUP_RATIO = 0.05       # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3      # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.1       # final LR as fraction of initial

# Batch size
DEVICE_BATCH_SIZE = 16    # per-device batch size (adjust for VRAM)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

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

model = MambaLM(config).to(device)

num_params = model.count_params()
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# Optimizer
no_weight_decay_names = {"dt_bias", "A_log", "D"}
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if any(name.endswith(suffix) for suffix in no_weight_decay_names):
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE,
    betas=ADAM_BETAS,
)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

print(f"Device: {device}")
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
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

print("\nTraining started...")

model.train()
while True:
    t0 = time.time()

    # Gradient accumulation
    optimizer.zero_grad()
    total_loss = 0.0
    for micro_step in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        x, y = x.to(device), y.to(device)
        loss = model(x, y)
        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()
        total_loss += loss.item()

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE * lrm

    # Step
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

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
model.eval()
with torch.no_grad():
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()

if torch.cuda.is_available():
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
