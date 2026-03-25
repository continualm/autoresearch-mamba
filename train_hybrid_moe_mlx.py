"""
Hybrid Mamba-Transformer MoE pretraining script. Apple Silicon (MLX), single-file.
Nemotron-H style hybrid architecture: Mamba SSM + Self-Attention + Mixture-of-Experts.

Usage:
    python train_hybrid_moe_mlx.py
    AUTORESEARCH_MLX_PRESET_FILE=mlx_hybrid_preset.local.json python train_hybrid_moe_mlx.py
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

# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------


def load_preset(*sections):
    preset_path = os.environ.get(PRESET_FILE_ENV)
    if not preset_path:
        return {}
    if not os.path.isabs(preset_path):
        preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), preset_path)
    with open(preset_path, "r", encoding="utf-8") as f:
        preset = json.load(f)
    for section in sections:
        section_preset = preset.get(section)
        if section_preset is None:
            continue
        if not isinstance(section_preset, dict):
            raise ValueError(f"Preset section {section!r} must be an object")
        print(f"Preset overrides: loaded {section} from {preset_path}")
        return section_preset
    print(f"Preset overrides: no matching section found in {preset_path} for {sections}")
    return {}


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "on"}:
        return True
    if key in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value {value!r}")


def normalize_mamba_type(value):
    key = str(value or "mamba-2").strip().lower().replace("_", "-")
    aliases = {
        "mamba2": "mamba-2",
        "mamba-2": "mamba-2",
        "m2": "mamba-2",
        "mamba3": "mamba-3",
        "mamba-3": "mamba-3",
        "m3": "mamba-3",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported mamba_type {value!r}; expected mamba-2 or mamba-3")
    return aliases[key]


PRESET = load_preset("train_hybrid_moe_mlx")


# ---------------------------------------------------------------------------
# Hybrid pattern parser (Nemotron-H style)
# ---------------------------------------------------------------------------

VALID_LAYER_TYPES = {"M", "*", "E", "-"}


def parse_hybrid_pattern(pattern):
    """Parse Nemotron-style hybrid_override_pattern string.

    Each character maps to one layer:
      M = Mamba SSM layer
      * = Self-Attention layer
      E = MoE (Mixture of Experts) layer
      - = Dense MLP/FFN layer

    Returns list of single-character layer type codes.
    """
    layers = list(pattern)
    for i, ch in enumerate(layers):
        if ch not in VALID_LAYER_TYPES:
            raise ValueError(
                f"Invalid layer type '{ch}' at position {i} in pattern '{pattern}'. "
                f"Valid types: {VALID_LAYER_TYPES}"
            )
    if len(layers) == 0:
        raise ValueError("Empty hybrid pattern")
    return layers


# ---------------------------------------------------------------------------
# HybridConfig
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    hybrid_pattern: str = "MEM*EME"
    mamba_type: str = "mamba-2"
    vocab_size: int = 8192
    d_model: int = 512
    # Mamba params
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 64
    # Mamba-3 specific
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    a_floor: float = 1e-4
    is_outproj_norm: bool = False
    is_mimo: bool = False
    mimo_rank: int = 4
    # Attention params
    num_attention_heads: int = 8
    num_kv_heads: int = 4
    attn_head_dim: int = 64
    rope_theta: float = 10000.0
    # MoE params
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 512
    shared_expert_hidden: int = 0
    aux_loss_coeff: float = 0.01
    # Dense MLP params (for '-' layers)
    d_intermediate: int = 0
    pad_vocab_size_multiple: int = 8


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def segsum(x):
    """Stable segment sum calculation for structured SSM scans."""
    T = x.shape[-1]
    x = mx.repeat(mx.expand_dims(x, -1), T, axis=-1)
    lower_exclusive = mx.tril(mx.ones((T, T)), k=-1).astype(mx.bool_)
    x = mx.where(lower_exclusive, x, mx.zeros_like(x))
    x_segsum = mx.cumsum(x, axis=-2)
    lower_inclusive = mx.tril(mx.ones((T, T)), k=0).astype(mx.bool_)
    return mx.where(lower_inclusive, x_segsum, mx.full(x_segsum.shape, -1e9))


def repeat_groups(x, target_heads):
    if x.shape[2] == target_heads:
        return x
    assert target_heads % x.shape[2] == 0
    return mx.repeat(x, target_heads // x.shape[2], axis=2)


def repeat_rank_groups(x, target_heads):
    if x.shape[3] == target_heads:
        return x
    assert target_heads % x.shape[3] == 0
    return mx.repeat(x, target_heads // x.shape[3], axis=3)


def wrap_two_pi(x):
    two_pi = 2.0 * math.pi
    return x - two_pi * mx.floor(x / two_pi)


def apply_rotary_pairwise(tensor, cos, sin):
    rotary_dim = cos.shape[-1] * 2
    tensor_rot = tensor[..., :rotary_dim]
    tensor_pass = tensor[..., rotary_dim:]
    left = tensor_rot[..., 0::2]
    right = tensor_rot[..., 1::2]
    rotated_left = left * cos - right * sin
    rotated_right = left * sin + right * cos
    rotated = mx.stack([rotated_left, rotated_right], axis=-1).reshape(tensor_rot.shape)
    if tensor_pass.shape[-1] == 0:
        return rotated
    return mx.concatenate([rotated, tensor_pass], axis=-1)


def apply_rotary_mimo(tensor, cos, sin, rotary_dim_divisor):
    if rotary_dim_divisor not in (2, 4):
        raise ValueError(f"Unsupported rotary_dim_divisor={rotary_dim_divisor}")
    headdim = tensor.shape[-1]
    half = headdim // 2
    first_half = tensor[..., :half]
    second_half = tensor[..., half:half + half]
    tail = tensor[..., half + half:]

    rotary_half = cos.shape[-1]
    if half > rotary_half:
        pad = half - rotary_half
        cos = mx.concatenate([cos, mx.ones((*cos.shape[:-1], pad), dtype=cos.dtype)], axis=-1)
        sin = mx.concatenate([sin, mx.zeros((*sin.shape[:-1], pad), dtype=sin.dtype)], axis=-1)

    rotated_first = first_half * cos - second_half * sin
    rotated_second = first_half * sin + second_half * cos
    rotated = mx.concatenate([rotated_first, rotated_second], axis=-1)
    if tail.shape[-1] == 0:
        return rotated
    return mx.concatenate([rotated, tail], axis=-1)


# ---------------------------------------------------------------------------
# Mamba-2 core
# ---------------------------------------------------------------------------

def ssd_scan(X, A, B, C, block_len=64):
    """Structured State Space Duality scan used by the MLX Mamba-2 baseline."""
    batch, seqlen, nheads, headdim = X.shape
    d_state = B.shape[-1]

    pad_len = (block_len - seqlen % block_len) % block_len
    if pad_len > 0:
        X = mx.pad(X, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        A = mx.pad(A, [(0, 0), (0, pad_len), (0, 0)])
        B = mx.pad(B, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        C = mx.pad(C, [(0, 0), (0, pad_len), (0, 0), (0, 0)])

    L = X.shape[1]
    nblocks = L // block_len

    X = X.reshape(batch, nblocks, block_len, nheads, headdim)
    A = A.reshape(batch, nblocks, block_len, nheads)
    B = B.reshape(batch, nblocks, block_len, nheads, d_state)
    C = C.reshape(batch, nblocks, block_len, nheads, d_state)

    A = mx.transpose(A, (0, 3, 1, 2))
    A_cumsum = mx.cumsum(A, axis=-1)

    L_matrix = mx.exp(segsum(A))
    CB = mx.einsum("bclhn,bcshn->bhcls", C, B)
    CB_masked = CB * L_matrix
    X_t = mx.transpose(X, (0, 3, 1, 2, 4))
    Y_diag = mx.einsum("bhcls,bhcsp->bhclp", CB_masked, X_t)

    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_t = mx.transpose(B, (0, 3, 1, 2, 4))
    states = mx.transpose(
        mx.einsum("bhcl,bhcln,bhclp->bhcpn", decay_states, B_t, X_t),
        (0, 2, 1, 3, 4),
    )

    initial = mx.zeros((batch, 1, nheads, headdim, d_state))
    states = mx.concatenate([initial, states], axis=1)

    chunk_decay = mx.pad(A_cumsum[:, :, :, -1], [(0, 0), (0, 0), (1, 0)])
    decay_chunk = mx.exp(segsum(chunk_decay))
    states_t = mx.transpose(states, (0, 2, 1, 3, 4))
    new_states = mx.einsum("bhzc,bhcpn->bhzpn", decay_chunk, states_t)
    new_states = new_states[:, :, :-1]

    state_decay_out = mx.exp(A_cumsum)
    C_t = mx.transpose(C, (0, 3, 1, 2, 4))
    Y_off = mx.einsum("bhcln,bhcpn,bhcl->bhclp", C_t, new_states, state_decay_out)

    Y = Y_diag + Y_off
    Y = mx.transpose(Y, (0, 2, 3, 1, 4))
    Y = Y.reshape(batch, L, nheads, headdim)

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
        grouped_f = grouped.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(grouped_f * grouped_f, axis=-1, keepdims=True) + self.eps)
        return (grouped.astype(mx.float32) * rms).reshape(x.shape) * self.weight


class RMSNormGated(nn.Module):
    def __init__(self, d, eps=1e-5, group_size=None, norm_before_gate=True):
        super().__init__()
        self.eps = eps
        self.group_size = d if group_size is None else group_size
        self.norm_before_gate = norm_before_gate
        assert d % self.group_size == 0
        self.weight = mx.ones((d,))

    def _norm(self, x):
        grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
        grouped_f = grouped.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(grouped_f * grouped_f, axis=-1, keepdims=True) + self.eps)
        return (grouped.astype(mx.float32) * rms).reshape(x.shape) * self.weight

    def __call__(self, x, z=None):
        if z is not None and not self.norm_before_gate:
            x = x * nn.silu(z)
        x = self._norm(x)
        if z is not None and self.norm_before_gate:
            x = x * nn.silu(z)
        return x


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=64,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
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

        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            bias=True,
            padding=d_conv - 1,
        )

        dt = np.exp(np.random.uniform(size=self.nheads) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min))
        dt = np.clip(dt, a_min=dt_init_floor, a_max=None)
        inv_dt = dt + np.log(-np.expm1(-dt))
        self.dt_bias = mx.array(inv_dt, dtype=mx.float32)

        A = np.random.uniform(1.0, 16.0, size=self.nheads)
        self.A_log = mx.array(np.log(A), dtype=mx.float32)
        self.D = mx.array(np.ones(self.nheads), dtype=mx.float32)

        self.norm = GroupedRMSNorm(self.d_inner, group_size=self.d_inner // self.ngroups)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, u):
        batch, seqlen, _ = u.shape
        zxbcdt = self.in_proj(u)

        d_inner = self.d_inner
        d_xbc = d_inner + 2 * self.ngroups * self.d_state
        z = zxbcdt[:, :, :d_inner]
        xbc = zxbcdt[:, :, d_inner:d_inner + d_xbc]
        dt = zxbcdt[:, :, d_inner + d_xbc:]

        xbc = self.conv1d(xbc)[:, :seqlen, :]
        xbc = nn.silu(xbc)

        x = xbc[:, :, :d_inner]
        B_ssm = xbc[:, :, d_inner:d_inner + self.ngroups * self.d_state]
        C_ssm = xbc[:, :, d_inner + self.ngroups * self.d_state:]

        x = x.reshape(batch, seqlen, self.nheads, self.headdim)
        B_ssm = B_ssm.reshape(batch, seqlen, self.ngroups, self.d_state)
        C_ssm = C_ssm.reshape(batch, seqlen, self.ngroups, self.d_state)
        B_ssm = repeat_groups(B_ssm, self.nheads)
        C_ssm = repeat_groups(C_ssm, self.nheads)

        dt = nn.softplus(dt + self.dt_bias)
        A = -mx.exp(self.A_log)
        x_scaled = x * mx.expand_dims(dt, -1)
        A_scaled = A[None, None, :] * dt

        y = ssd_scan(x_scaled, A_scaled, B_ssm, C_ssm, block_len=self.chunk_size)
        y = y + mx.expand_dims(self.D, -1) * x
        y = y.reshape(batch, seqlen, d_inner)
        y = self.norm(y * nn.silu(z))
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Mamba-3 core (SISO)
# ---------------------------------------------------------------------------

def mamba3_chunk_forward(
    q, k, v, adt, dt, trap_proj, z, D, q_bias, k_bias, angle_proj,
    prev_state, prev_k, prev_v, prev_angle,
):
    dt_f = dt.astype(mx.float32)
    trap = mx.sigmoid(trap_proj.astype(mx.float32))

    angles = mx.tanh(angle_proj.astype(mx.float32)) * math.pi
    angle_cumsum = mx.cumsum(angles * mx.expand_dims(dt_f, -1), axis=1) + mx.expand_dims(prev_angle, axis=1)
    two_pi = 2.0 * math.pi
    angle_cumsum = angle_cumsum - two_pi * mx.floor(angle_cumsum / two_pi)

    q_biased = q + q_bias[None, None, :, :]
    k_biased = k + k_bias[None, None, :, :]

    beta0 = dt_f[:, 0, :] * (1.0 - trap[:, 0, :])
    acc_state = prev_state + prev_v.astype(mx.float32)[:, :, :, None] * prev_k.astype(mx.float32)[:, :, None, :] * beta0[:, :, None, None]

    dt_shifted = mx.pad(dt_f[:, 1:, :], [(0, 0), (0, 1), (0, 0)])
    trap_shifted = mx.pad(trap[:, 1:, :], [(0, 0), (0, 1), (0, 0)])
    shifted_gamma = dt_shifted * (1.0 - trap_shifted)
    scale = dt_f * trap + shifted_gamma

    qk_dot = (q_biased.astype(mx.float32) * k_biased.astype(mx.float32)).sum(axis=-1) * shifted_gamma

    cos_angles = mx.cos(angle_cumsum).astype(q.dtype)
    sin_angles = mx.sin(angle_cumsum).astype(q.dtype)
    q_rot = apply_rotary_pairwise(q_biased, cos_angles, sin_angles)
    k_rot = apply_rotary_pairwise(k_biased, cos_angles, sin_angles)

    q_f = q_rot.astype(mx.float32)
    k_f = k_rot.astype(mx.float32)
    v_f = v.astype(mx.float32)
    k_scaled = k_f * mx.expand_dims(scale, -1)

    adt_h = mx.transpose(adt.astype(mx.float32), (0, 2, 1))
    decay = mx.exp(segsum(adt_h))
    qk = mx.einsum("bthq,bshq->bhts", q_f, k_scaled)
    causal_mask = mx.tril(mx.ones((q.shape[1], k.shape[1])), k=0).astype(qk.dtype)
    qk_causal = qk * causal_mask[None, None, :, :]
    out = mx.einsum("bhts,bshv->bthv", qk_causal * decay, v_f)

    exp_da_cs = mx.exp(mx.cumsum(adt_h, axis=-1))
    out = out + mx.einsum("bhvq,bthq,bth->bthv", acc_state, q_f, mx.transpose(exp_da_cs, (0, 2, 1)))
    out = out + D[None, None, :, None].astype(mx.float32) * v_f
    out = out - v_f * mx.expand_dims(qk_dot, -1)

    if z is not None:
        z_f = z.astype(mx.float32)
        out = out * z_f * mx.sigmoid(z_f)

    da_cs_last = mx.exp(mx.sum(adt_h, axis=-1))
    da_cs_rev = mx.exp(mx.sum(adt_h, axis=-1, keepdims=True) - mx.cumsum(adt_h, axis=-1))
    v_scaled = v_f * mx.expand_dims(mx.transpose(da_cs_rev, (0, 2, 1)), -1)
    next_state = acc_state * da_cs_last[:, :, None, None] + mx.einsum("bthq,bthv->bhvq", k_scaled, v_scaled)

    return out.astype(v.dtype), next_state, k_rot[:, -1, :, :], v[:, -1, :, :], angle_cumsum[:, -1, :, :]


def mamba3_mimo_recurrent(
    q, k, v, adt, dt, trap_proj, z, D, q_bias, k_bias, angle_proj,
    mimo_v, mimo_z, mimo_o, rotary_dim_divisor,
    prev_state=None, prev_k=None, prev_v=None, prev_angle=None, outproj_norm=None,
):
    batch, seqlen, _, nheads, d_state = q.shape
    headdim = v.shape[-1]
    dt_f = dt.astype(mx.float32)
    trap = mx.sigmoid(trap_proj.astype(mx.float32))
    angle_values = mx.tanh(angle_proj.astype(mx.float32)) * math.pi

    q_bias_r = mx.transpose(q_bias, (1, 0, 2))
    k_bias_r = mx.transpose(k_bias, (1, 0, 2))
    mimo_v_f = mimo_v.astype(mx.float32)
    mimo_z_f = mimo_z.astype(mx.float32)
    mimo_o_f = mimo_o.astype(mx.float32)
    v_proj = mx.einsum("bthp,hrp->bthrp", v.astype(mx.float32), mimo_v_f)
    z_proj = None
    if z is not None and outproj_norm is None:
        z_proj = mx.einsum("bthp,hrp->bthrp", z.astype(mx.float32), mimo_z_f)

    state = mx.zeros((batch, nheads, headdim, d_state), dtype=mx.float32) if prev_state is None else prev_state.astype(mx.float32)
    k_state = mx.zeros((batch, q.shape[2], nheads, d_state), dtype=q.dtype) if prev_k is None else prev_k
    v_state = mx.zeros((batch, nheads, headdim), dtype=v.dtype) if prev_v is None else prev_v
    angle_state = mx.zeros((batch, nheads, angle_proj.shape[-1]), dtype=mx.float32) if prev_angle is None else prev_angle.astype(mx.float32)
    k_state_h = mx.transpose(k_state, (0, 2, 1, 3))

    outputs = []
    for idx in range(seqlen):
        q_t = q[:, idx, :, :, :] + q_bias_r[None, :, :, :]
        k_t = k[:, idx, :, :, :] + k_bias_r[None, :, :, :]
        q_t = mx.transpose(q_t, (0, 2, 1, 3))
        k_t = mx.transpose(k_t, (0, 2, 1, 3))
        v_t = v_proj[:, idx, :, :, :]
        z_t = None if z_proj is None else z_proj[:, idx, :, :, :]

        angle_state = wrap_two_pi(angle_state + angle_values[:, idx, :, :] * dt_f[:, idx, :, None])
        cos_angles = mx.expand_dims(mx.cos(angle_state).astype(q.dtype), axis=2)
        sin_angles = mx.expand_dims(mx.sin(angle_state).astype(q.dtype), axis=2)
        q_rot = apply_rotary_mimo(q_t, cos_angles, sin_angles, rotary_dim_divisor)
        k_rot = apply_rotary_mimo(k_t, cos_angles, sin_angles, rotary_dim_divisor)

        alpha = mx.exp(adt[:, idx, :].astype(mx.float32))
        beta = (1.0 - trap[:, idx, :]) * dt_f[:, idx, :] * alpha
        gamma = trap[:, idx, :] * dt_f[:, idx, :]

        prev_v_proj = mx.einsum("bhp,hrp->bhrp", v_state.astype(mx.float32), mimo_v_f)
        prev_kv = mx.einsum("bhrd,bhrp->bhpd", k_state_h.astype(mx.float32), prev_v_proj)
        curr_kv = mx.einsum("bhrd,bhrp->bhpd", k_rot.astype(mx.float32), v_t)
        state = (
            alpha[:, :, None, None] * state
            + beta[:, :, None, None] * prev_kv
            + gamma[:, :, None, None] * curr_kv
        )

        out_rank = mx.einsum("bhpd,bhrd->bhrp", state, q_rot.astype(mx.float32))
        out_rank = out_rank + D[None, :, None, None].astype(mx.float32) * v_t
        if z_t is not None:
            out_rank = out_rank * nn.silu(z_t)

        if outproj_norm is None:
            outputs.append(mx.einsum("bhrp,hrp->bhp", out_rank, mimo_o_f))
        else:
            outputs.append(mx.transpose(out_rank, (0, 2, 1, 3)))

        k_state_h = k_rot.astype(q.dtype)
        v_state = v[:, idx, :, :]

    if outproj_norm is None:
        y = mx.stack(outputs, axis=1)
    else:
        if z is None:
            raise ValueError("Mamba-3 MIMO outproj norm requires z")
        y_rank = mx.stack(outputs, axis=1)
        z_rank = mx.einsum("bthp,hrp->bthrp", z.astype(mx.float32), mimo_z_f)
        z_rank = mx.transpose(z_rank, (0, 1, 3, 2, 4))
        y_rank = outproj_norm(
            y_rank.reshape(batch, seqlen, q.shape[2], nheads * headdim),
            z_rank.reshape(batch, seqlen, q.shape[2], nheads * headdim),
        ).reshape(batch, seqlen, q.shape[2], nheads, headdim)
        y = mx.einsum("btrhp,hrp->bthp", y_rank.astype(mx.float32), mimo_o_f)

    return y.astype(v.dtype), state, mx.transpose(k_state_h, (0, 2, 1, 3)), v_state, angle_state


class Mamba3Block(nn.Module):
    def __init__(
        self, d_model, d_state=64, expand=2, headdim=64, ngroups=1,
        rope_fraction=0.5, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
        a_floor=1e-4, is_outproj_norm=False, is_mimo=False, mimo_rank=4,
        chunk_size=64, layer_idx=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.d_inner = expand * d_model
        self.nheads = self.d_inner // self.headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.a_floor = a_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.layer_idx = layer_idx

        assert self.d_inner % self.headdim == 0
        assert self.nheads % self.ngroups == 0
        assert rope_fraction in (0.5, 1.0)
        self.rotary_dim_divisor = int(2 / rope_fraction)

        split_tensor_size = int(self.d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError("Mamba-3 requires an even rotary dimension greater than zero")

        d_in_proj = 2 * self.d_inner + 2 * self.mimo_rank * self.ngroups * self.d_state + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        dt = np.exp(np.random.uniform(size=self.nheads) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min))
        dt = np.clip(dt, a_min=dt_init_floor, a_max=None)
        inv_dt = dt + np.log(-np.expm1(-dt))
        self.dt_bias = mx.array(inv_dt, dtype=mx.float32)

        self.B_bias = mx.ones((self.nheads, self.mimo_rank, self.d_state), dtype=mx.float32)
        self.C_bias = mx.ones((self.nheads, self.mimo_rank, self.d_state), dtype=mx.float32)
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5)
        self.D = mx.ones((self.nheads,), dtype=mx.float32)
        if self.is_mimo:
            self.mimo_x = mx.ones((self.nheads, self.mimo_rank, self.headdim), dtype=mx.float32) / self.mimo_rank
            self.mimo_z = mx.ones((self.nheads, self.mimo_rank, self.headdim), dtype=mx.float32)
            self.mimo_o = mx.ones((self.nheads, self.mimo_rank, self.headdim), dtype=mx.float32) / self.mimo_rank
        if self.is_outproj_norm:
            self.outproj_norm = RMSNormGated(self.d_inner, eps=1e-5, group_size=self.headdim, norm_before_gate=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _get_zero_states(self, batch, dtype):
        angle_state = mx.zeros((batch, self.nheads, self.num_rope_angles), dtype=mx.float32)
        ssm_state = mx.zeros((batch, self.nheads, self.headdim, self.d_state), dtype=mx.float32)
        if self.is_mimo:
            k_state = mx.zeros((batch, self.mimo_rank, self.nheads, self.d_state), dtype=dtype)
        else:
            k_state = mx.zeros((batch, self.nheads, self.d_state), dtype=dtype)
        v_state = mx.zeros((batch, self.nheads, self.headdim), dtype=dtype)
        return angle_state, ssm_state, k_state, v_state

    def __call__(self, u):
        batch, seqlen, _ = u.shape

        proj = self.in_proj(u)
        cursor = 0
        z = proj[:, :, cursor:cursor + self.d_inner]
        cursor += self.d_inner
        x = proj[:, :, cursor:cursor + self.d_inner]
        cursor += self.d_inner
        bc_width = self.mimo_rank * self.ngroups * self.d_state
        B_proj = proj[:, :, cursor:cursor + bc_width]
        cursor += bc_width
        C_proj = proj[:, :, cursor:cursor + bc_width]
        cursor += bc_width
        dd_dt = proj[:, :, cursor:cursor + self.nheads]
        cursor += self.nheads
        dd_A = proj[:, :, cursor:cursor + self.nheads]
        cursor += self.nheads
        trap = proj[:, :, cursor:cursor + self.nheads]
        cursor += self.nheads
        angles = proj[:, :, cursor:cursor + self.num_rope_angles]

        z = z.reshape(batch, seqlen, self.nheads, self.headdim)
        x = x.reshape(batch, seqlen, self.nheads, self.headdim)
        B_proj = B_proj.reshape(batch, seqlen, self.mimo_rank, self.ngroups, self.d_state)
        C_proj = C_proj.reshape(batch, seqlen, self.mimo_rank, self.ngroups, self.d_state)
        angles = mx.expand_dims(angles, axis=2)
        angles = mx.repeat(angles, self.nheads, axis=2)

        dt = nn.softplus(dd_dt + self.dt_bias)
        a = -nn.softplus(dd_A.astype(mx.float32))
        a = mx.minimum(a, -self.a_floor)
        adt = a * dt.astype(mx.float32)

        angle_state, ssm_state, k_state, v_state = self._get_zero_states(batch, x.dtype)

        if self.is_mimo:
            k = repeat_rank_groups(self.B_norm(B_proj), self.nheads)
            q = repeat_rank_groups(self.C_norm(C_proj), self.nheads)
            y, ssm_state, k_state, v_state, angle_state = mamba3_mimo_recurrent(
                q, k, x, adt, dt, trap, z, self.D,
                self.C_bias, self.B_bias, angles,
                self.mimo_x, self.mimo_z, self.mimo_o, self.rotary_dim_divisor,
                prev_state=ssm_state, prev_k=k_state, prev_v=v_state, prev_angle=angle_state,
                outproj_norm=self.outproj_norm if self.is_outproj_norm else None,
            )
            y = y.reshape(batch, seqlen, self.d_inner)
        else:
            k = repeat_groups(mx.squeeze(self.B_norm(B_proj), axis=2), self.nheads)
            q = repeat_groups(mx.squeeze(self.C_norm(C_proj), axis=2), self.nheads)
            prev_k = k_state
            outputs = []
            for start in range(0, seqlen, self.chunk_size):
                stop = min(start + self.chunk_size, seqlen)
                chunk_out, ssm_state, prev_k, v_state, angle_state = mamba3_chunk_forward(
                    q[:, start:stop, :, :], k[:, start:stop, :, :], x[:, start:stop, :, :],
                    adt[:, start:stop, :], dt[:, start:stop, :], trap[:, start:stop, :],
                    None if self.is_outproj_norm else z[:, start:stop, :, :],
                    self.D, self.C_bias[:, 0, :], self.B_bias[:, 0, :],
                    angles[:, start:stop, :, :],
                    ssm_state, prev_k, v_state, angle_state,
                )
                outputs.append(chunk_out)
            y = mx.concatenate(outputs, axis=1).reshape(batch, seqlen, self.d_inner)
            if self.is_outproj_norm:
                y = self.outproj_norm(y, z.reshape(batch, seqlen, self.d_inner))

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
# CausalSelfAttention (GQA + RoPE)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim, rope_theta=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.rope = nn.RoPE(head_dim, base=rope_theta)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=nn.MultiHeadAttention.create_additive_causal_mask(T))
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MoE Layer (top-k router + expert dispatch)
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k, expert_hidden, shared_expert_hidden=0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = [GatedMLP(d_model, expert_hidden) for _ in range(num_experts)]
        self.shared_expert = None
        if shared_expert_hidden > 0:
            self.shared_expert = GatedMLP(d_model, shared_expert_hidden)

    def __call__(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # Router
        logits = self.gate(x_flat)
        router_probs = mx.softmax(logits, axis=-1)

        # Top-k selection
        sorted_indices = mx.argsort(-logits, axis=-1)
        top_k_indices = sorted_indices[:, :self.top_k]

        # Top-k weights (renormalized softmax over selected experts)
        bt_range = mx.expand_dims(mx.arange(B * T), -1)
        top_k_logits = logits[bt_range, top_k_indices]
        top_k_weights = mx.softmax(top_k_logits, axis=-1)

        # Auxiliary load-balancing loss (Switch Transformer style)
        top1_indices = top_k_indices[:, 0]
        all_experts_range = mx.arange(self.num_experts)
        top1_onehot = (mx.expand_dims(top1_indices, -1) == all_experts_range).astype(mx.float32)
        f = top1_onehot.mean(axis=0)
        p = router_probs.mean(axis=0)
        aux_loss = self.num_experts * mx.sum(f * p)

        # Expert dispatch: run all experts, gather top-k outputs
        expert_outputs = mx.stack(
            [expert(x_flat) for expert in self.experts], axis=1
        )  # (BT, E, D)
        bt_repeat = mx.repeat(mx.expand_dims(mx.arange(B * T), -1), self.top_k, axis=-1).reshape(-1)
        ek_flat = top_k_indices.reshape(-1)
        selected = expert_outputs[bt_repeat, ek_flat].reshape(B * T, self.top_k, D)
        output = (selected * mx.expand_dims(top_k_weights, -1)).sum(axis=1)

        # Optional shared expert
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        return output.reshape(B, T, D), aux_loss


# ---------------------------------------------------------------------------
# Hybrid residual block and model
# ---------------------------------------------------------------------------

class HybridResidualBlock(nn.Module):
    def __init__(self, config, layer_idx, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.has_aux_loss = (layer_type == "E")
        self.norm = nn.RMSNorm(config.d_model)

        if layer_type == "M":
            if config.mamba_type == "mamba-3":
                self.block = Mamba3Block(
                    d_model=config.d_model, d_state=config.d_state, expand=config.expand,
                    headdim=config.headdim, ngroups=config.ngroups,
                    rope_fraction=config.rope_fraction, dt_min=config.dt_min,
                    dt_max=config.dt_max, dt_init_floor=config.dt_init_floor,
                    a_floor=config.a_floor, is_outproj_norm=config.is_outproj_norm,
                    is_mimo=config.is_mimo, mimo_rank=config.mimo_rank,
                    chunk_size=config.chunk_size, layer_idx=layer_idx,
                )
            else:
                self.block = Mamba2Block(
                    d_model=config.d_model, d_state=config.d_state,
                    d_conv=config.d_conv, expand=config.expand,
                    headdim=config.headdim, ngroups=config.ngroups,
                    chunk_size=config.chunk_size,
                    dt_min=config.dt_min, dt_max=config.dt_max,
                    dt_init_floor=config.dt_init_floor,
                )
        elif layer_type == "*":
            self.block = CausalSelfAttention(
                d_model=config.d_model,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.attn_head_dim,
                rope_theta=config.rope_theta,
            )
        elif layer_type == "E":
            self.block = MoELayer(
                d_model=config.d_model,
                num_experts=config.num_experts,
                top_k=config.top_k,
                expert_hidden=config.expert_hidden,
                shared_expert_hidden=config.shared_expert_hidden,
            )
        elif layer_type == "-":
            d_hidden = config.d_intermediate if config.d_intermediate > 0 else 4 * config.d_model
            self.block = GatedMLP(config.d_model, d_hidden)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def __call__(self, x):
        h = self.norm(x)
        if self.has_aux_loss:
            out, aux_loss = self.block(h)
            return x + out, aux_loss
        else:
            return x + self.block(h), mx.array(0.0)


class HybridLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layer_types = parse_hybrid_pattern(config.hybrid_pattern)
        self.layer_types = layer_types
        n_layer = len(layer_types)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [
            HybridResidualBlock(config, i, lt)
            for i, lt in enumerate(layer_types)
        ]
        self.norm_f = nn.RMSNorm(config.d_model)

        # Weight initialization
        self.embedding.weight = mx.random.normal(shape=self.embedding.weight.shape) * 0.02

        # Residual scaling: output projections divided by sqrt(n_layers)
        residual_scale = math.sqrt(n_layer)
        for layer in self.layers:
            lt = layer.layer_type
            if lt == "M":
                layer.block.out_proj.weight = layer.block.out_proj.weight / residual_scale
            elif lt == "*":
                layer.block.o_proj.weight = layer.block.o_proj.weight / residual_scale
            elif lt == "E":
                for expert in layer.block.experts:
                    expert.fc2.weight = expert.fc2.weight / residual_scale
                if layer.block.shared_expert is not None:
                    layer.block.shared_expert.fc2.weight = layer.block.shared_expert.fc2.weight / residual_scale
            elif lt == "-":
                layer.block.fc2.weight = layer.block.fc2.weight / residual_scale

    def __call__(self, idx, targets=None):
        x = self.embedding(idx)
        total_aux_loss = mx.array(0.0)
        for layer in self.layers:
            x, aux = layer(x)
            total_aux_loss = total_aux_loss + aux
        x = self.norm_f(x)
        logits = x @ self.embedding.weight.T

        if targets is not None:
            ce_loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
            )
            return ce_loss, total_aux_loss
        return logits

    def count_params(self):
        return sum(p.size for _, p in mlx.utils.tree_flatten(self.parameters()))

    def count_active_params(self):
        """Active params per forward pass (accounts for MoE top-k < num_experts)."""
        total = self.count_params()
        for layer in self.layers:
            if layer.layer_type == "E":
                moe = layer.block
                expert_params = sum(p.size for _, p in mlx.utils.tree_flatten(moe.experts[0].parameters()))
                inactive = expert_params * (moe.num_experts - moe.top_k)
                total -= inactive
        return total


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HYBRID_PATTERN = str(PRESET.get("HYBRID_PATTERN", "MEM*EME"))
MAMBA_TYPE = normalize_mamba_type(PRESET.get("MAMBA_TYPE", "mamba-2"))

D_MODEL = int(PRESET.get("D_MODEL", 512))
D_STATE = int(PRESET.get("D_STATE", 64))
D_CONV = int(PRESET.get("D_CONV", 4))
EXPAND = int(PRESET.get("EXPAND", 2))
HEADDIM = int(PRESET.get("HEADDIM", 64))
NGROUPS = int(PRESET.get("NGROUPS", 1))
D_INTERMEDIATE = int(PRESET.get("D_INTERMEDIATE", 0))
CHUNK_SIZE = int(PRESET.get("CHUNK_SIZE", 64))

# Mamba-3 specific
ROPE_FRACTION = float(PRESET.get("ROPE_FRACTION", 0.5))
DT_MIN = float(PRESET.get("DT_MIN", 0.001))
DT_MAX = float(PRESET.get("DT_MAX", 0.1))
DT_INIT_FLOOR = float(PRESET.get("DT_INIT_FLOOR", 1e-4))
A_FLOOR = float(PRESET.get("A_FLOOR", 1e-4))
IS_OUTPROJ_NORM = parse_bool(PRESET.get("IS_OUTPROJ_NORM", False))
IS_MIMO = parse_bool(PRESET.get("IS_MIMO", False))
MIMO_RANK = int(PRESET.get("MIMO_RANK", 4))

# Attention params
NUM_ATTENTION_HEADS = int(PRESET.get("NUM_ATTENTION_HEADS", 8))
NUM_KV_HEADS = int(PRESET.get("NUM_KV_HEADS", 4))
ATTN_HEAD_DIM = int(PRESET.get("ATTN_HEAD_DIM", 64))
ROPE_THETA = float(PRESET.get("ROPE_THETA", 10000.0))

# MoE params
NUM_EXPERTS = int(PRESET.get("NUM_EXPERTS", 8))
TOP_K = int(PRESET.get("TOP_K", 2))
EXPERT_HIDDEN = int(PRESET.get("EXPERT_HIDDEN", 512))
SHARED_EXPERT_HIDDEN = int(PRESET.get("SHARED_EXPERT_HIDDEN", 0))
AUX_LOSS_COEFF = float(PRESET.get("AUX_LOSS_COEFF", 0.01))

# Optimization
TOTAL_BATCH_SIZE = int(PRESET.get("TOTAL_BATCH_SIZE", 2**18))
LEARNING_RATE = float(PRESET.get("LEARNING_RATE", 3e-4))
WEIGHT_DECAY = float(PRESET.get("WEIGHT_DECAY", 0.1))
ADAM_BETAS = (
    float(PRESET.get("ADAM_BETA1", 0.9)),
    float(PRESET.get("ADAM_BETA2", 0.95)),
)
WARMUP_RATIO = float(PRESET.get("WARMUP_RATIO", 0.05))
WARMDOWN_RATIO = float(PRESET.get("WARMDOWN_RATIO", 0.3))
FINAL_LR_FRAC = float(PRESET.get("FINAL_LR_FRAC", 0.1))
DEVICE_BATCH_SIZE = int(PRESET.get("DEVICE_BATCH_SIZE", 16))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mx.random.seed(42)
np.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()

layer_types = parse_hybrid_pattern(HYBRID_PATTERN)
n_mamba = sum(1 for lt in layer_types if lt == "M")
n_attn = sum(1 for lt in layer_types if lt == "*")
n_moe = sum(1 for lt in layer_types if lt == "E")
n_mlp = sum(1 for lt in layer_types if lt == "-")

print(f"Architecture: hybrid")
print(f"Hybrid pattern: {HYBRID_PATTERN} ({len(layer_types)} layers: {n_mamba}M {n_attn}* {n_moe}E {n_mlp}-)")
print(f"Mamba type: {MAMBA_TYPE}")
print(f"Vocab size: {vocab_size:,}")

pad_multiple = 8
if vocab_size % pad_multiple != 0:
    vocab_size = ((vocab_size + pad_multiple - 1) // pad_multiple) * pad_multiple
    print(f"Padded vocab size: {vocab_size:,}")

config = HybridConfig(
    hybrid_pattern=HYBRID_PATTERN,
    mamba_type=MAMBA_TYPE,
    vocab_size=vocab_size,
    d_model=D_MODEL,
    d_state=D_STATE,
    d_conv=D_CONV,
    expand=EXPAND,
    headdim=HEADDIM,
    ngroups=NGROUPS,
    chunk_size=CHUNK_SIZE,
    rope_fraction=ROPE_FRACTION,
    dt_min=DT_MIN,
    dt_max=DT_MAX,
    dt_init_floor=DT_INIT_FLOOR,
    a_floor=A_FLOOR,
    is_outproj_norm=IS_OUTPROJ_NORM,
    is_mimo=IS_MIMO,
    mimo_rank=MIMO_RANK,
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    attn_head_dim=ATTN_HEAD_DIM,
    rope_theta=ROPE_THETA,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    expert_hidden=EXPERT_HIDDEN,
    shared_expert_hidden=SHARED_EXPERT_HIDDEN,
    aux_loss_coeff=AUX_LOSS_COEFF,
    d_intermediate=D_INTERMEDIATE,
)
print(f"Model config: {asdict(config)}")

model = HybridLM(config)
mx.eval(model.parameters())

num_params = model.count_params()
active_params = model.count_active_params()
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M total, {active_params / 1e6:.1f}M active)")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

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
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def loss_fn(model, x, y):
    ce_loss, aux_loss = model(x, y)
    return ce_loss + config.aux_loss_coeff * aux_loss


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

t_start = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

print("\nTraining started...")

while True:
    t0 = time.time()
    total_loss = 0.0
    acc_grads = None

    for _ in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        loss, grads = loss_and_grad_fn(model, x, y)
        if acc_grads is None:
            acc_grads = grads
        else:
            acc_grads = mlx.utils.tree_map(lambda a, b: a + b, acc_grads, grads)
        total_loss += loss.item()
        mx.eval(loss)

    if grad_accum_steps > 1:
        acc_grads = mlx.utils.tree_map(lambda g: g / grad_accum_steps, acc_grads)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    optimizer.learning_rate = mx.array(LEARNING_RATE * lrm)

    model.update(optimizer.apply_gradients(acc_grads, model))
    mx.eval(model.parameters(), optimizer.state)

    t1 = time.time()
    dt = t1 - t0
    train_loss_f = total_loss / grad_accum_steps

    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

print("Evaluating...")
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()

print("---")
print(f"architecture:     hybrid")
print(f"hybrid_pattern:   {HYBRID_PATTERN}")
print(f"mamba_type:       {MAMBA_TYPE}")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"num_active_params_M: {active_params / 1e6:.1f}")
print(f"depth:            {len(layer_types)}")
