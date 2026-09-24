"""Trainable PyTorch mirror of the Zonos2 architecture.

The inference implementation (python/zonos2/models/zonos2.py) is built on a custom
BaseOP graph with flashinfer kernels, a paged KV cache, and an ambient batch
context — none of which supports backprop. This module re-implements the exact
same math with plain nn.Modules and SDPA so the model can be fine-tuned, while
keeping **state_dict keys identical to the checkpoint format**, so:

  * ``load_state_dict(load_checkpoint_state_dict(path))`` loads a release
    checkpoint directly, and
  * ``model.state_dict()`` (after LoRA merge) saves back to a checkpoint the
    inference server loads unchanged.

Numerics mirrored from the inference code:
  * MultiEmbedding: sum of 9 audio + 1 text embedding tables, no padding masking
    (zonos2.py:140-143 keeps padding rows deliberately).
  * Speaker conditioning: optional LDA affine, then speaker_projection, written
    over the embedding at the reserved slot position (zonos2.py:849-873).
  * emb_norm: parameterless RMSNorm applied once before layer 0 (zonos2.py:875).
  * Residual stream uses the fused-add pattern of RMSNormFused (norm.py:48-55).
  * Attention: fused wkv (k|v), per-head RMS QK-norm (eps=1e-6, no gamma) with
    learnable |temp| on Q, interleaved (non-NeoX) RoPE, per-head sigmoid output
    gating (zonos2.py:210-291).
  * Dense FFN: w_in = [h, gate] halves, y = h * silu(gate) (zonos2.py:322-331).
  * MoE: EDA router (router-state residual across MoE layers, cloned pre-norm),
    fp32 softmax, bias-aware top-k selection with pre-bias combine weights;
    experts compute silu(gate) * up with gate first (fused layout).
  * Output: single multi_output linear reshaped to (n_codebooks, audio_vocab)
    with tanh softcap at loss_softcap (zonos2.py:910-929).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


def rms_norm_fp32(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm computed in fp32 like the flashinfer kernels, cast back to x.dtype."""
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_fp32(x, self.weight, self.eps)


class Linear3D(nn.Module):
    """Linear whose checkpoint weight is stored 3D as [divisor, out/divisor, in].

    Matches ChunkedLinear (layers/linear.py:130): forward flattens to a plain 2D
    matmul; the row order of the flattened weight is [chunk0; chunk1].
    """

    def __init__(self, in_features: int, out_features: int, divisor: int):
        super().__init__()
        assert out_features % divisor == 0
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(divisor, out_features // divisor, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.flatten(0, 1))


def apply_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved (GPT-J style, is_neox=False) RoPE in fp32.

    x: (B, T, H, D); cos/sin: (T, D/2).
    """
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Embeddings and output head
# ---------------------------------------------------------------------------


class MultiEmbedding(nn.Module):
    """Sum of per-column embeddings. Keys: multi_embedder.embedders.{i}.weight"""

    def __init__(self, config):
        super().__init__()
        audio_vocab = config.codebook_size + 2
        tables = [nn.Embedding(audio_vocab, config.dim) for _ in range(config.n_codebooks)]
        if config.text_vocab is not None:
            tables.append(nn.Embedding(config.text_vocab + 1, config.dim))
        self.embedders = nn.ModuleList(tables)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (..., n_codebooks [+ 1]) int64. No padding masking, matching inference.
        result = self.embedders[0](codes[..., 0])
        for i in range(1, codes.size(-1)):
            result = result + self.embedders[i](codes[..., i])
        return result


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Keys: layers.{N}.attention.{wq,wkv,wo,gater}.weight and .temp"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_heads if config.n_heads is not None else config.dim // config.head_dim
        self.num_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else self.num_heads
        self.head_dim = config.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.wq = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.wkv = Linear3D(config.dim, kv_dim * 2, divisor=2)
        self.wo = nn.Linear(self.num_heads * self.head_dim, config.dim, bias=False)
        self.temp = nn.Parameter(torch.ones(1, self.num_heads, 1))
        self.gater = nn.Linear(config.dim, self.num_heads, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, KV, D = self.num_heads, self.num_kv_heads, self.head_dim

        gate = torch.sigmoid(self.gater(x))  # (B, T, H)

        q = self.wq(x).view(B, T, H, D)
        kv = self.wkv(x)
        k, v = kv.split(KV * D, dim=-1)
        k = k.view(B, T, KV, D)
        v = v.view(B, T, KV, D)

        # QK norm: per-head RMS without gamma (eps=1e-6), |temp| scaling on Q.
        q = rms_norm_fp32(q, None, 1e-6) * self.temp.abs().view(1, 1, H, 1).to(q.dtype)
        k = rms_norm_fp32(k, None, 1e-6)

        q = apply_rope_interleaved(q, cos, sin)
        k = apply_rope_interleaved(k, cos, sin)

        if KV < H:  # GQA: expand KV heads (works on any torch version)
            k = k.repeat_interleave(H // KV, dim=2)
            v = v.repeat_interleave(H // KV, dim=2)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)  # (B, T, H, D)

        o = o * gate.unsqueeze(-1)
        return self.wo(o.reshape(B, T, H * D))


# ---------------------------------------------------------------------------
# Feed-forward: dense and MoE
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Dense FFN. Keys: layers.{N}.feed_forward.{w_in,w_out}.weight

    NOTE: the w_in halves are [h, gate] (up first) — y = h * silu(gate) — the
    opposite order of the fused-expert layout. Mirrors zonos2.py:322-331.
    """

    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.w_in = Linear3D(config.dim, intermediate_size * 2, divisor=2)
        self.w_out = nn.Linear(intermediate_size, config.dim, bias=False)
        self._inter = intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_gate = self.w_in(x)
        h = h_gate[..., : self._inter]
        gate = h_gate[..., self._inter :]
        return self.w_out(h * F.silu(gate))


class Router(nn.Module):
    """EDA MoE router. Keys under layers.{N}.feed_forward.router.*"""

    def __init__(self, config, layer_id: int, top_k: int):
        super().__init__()
        self.num_experts = config.moe_n_experts
        self.top_k = top_k
        strategy = str(getattr(config, "moe_balancing_strategy", "legacy")).strip().lower().replace("-", "_")
        self.use_legacy_balancing = strategy in ("legacy", "old", "aux", "aux_loss")

        rd = config.moe_router_dim
        self.down_proj = nn.Linear(config.dim, rd, bias=True)
        self.router_mlp = nn.Sequential(
            nn.Linear(rd, rd, bias=True),
            nn.GELU(),
            nn.Linear(rd, rd, bias=True),
            nn.GELU(),
            nn.Linear(rd, self.num_experts, bias=False),
        )
        self.rmsnorm_eda = RMSNorm(rd, eps=config.norm_eps)
        self.use_eda = layer_id != config.moe_start_from_layer
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(rd))
        # Load-balancing biases are a training artifact updated by the original
        # trainer; kept frozen (buffer) during fine-tuning.
        self.register_buffer("balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, router_states: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.down_proj(x)
        if self.use_eda and router_states is not None:
            h = h + router_states * self.router_states_scale
        h_next = h.clone()  # EDA carries the un-normalized state (zonos2.py:618)
        hn = self.rmsnorm_eda(h)

        expert_prob = torch.softmax(self.router_mlp(hn).float(), dim=-1)
        with torch.no_grad():
            bias = self.balancing_biases.detach().float()
            scores = expert_prob + bias if self.use_legacy_balancing else expert_prob - bias
            _, expert_choice = torch.topk(scores, self.top_k, dim=-1)
        # Combine weights are the pre-bias probabilities; grads flow through gather.
        route_prob = torch.gather(expert_prob, dim=-1, index=expert_choice)
        return route_prob, expert_choice, h_next


class GroupedExperts(nn.Module):
    """MoE experts. Keys: layers.{N}.feed_forward.experts.{gate_up_proj,down_proj}

    gate_up_proj rows are [w1 (gate, silu-activated); w3 (up)] — silu(gate) * up,
    matching silu_and_mul in the fused inference path.
    """

    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.num_experts = config.moe_n_experts
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * intermediate_size, config.dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, config.dim, intermediate_size)
        )

    def forward(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        # x: (N, hidden); topk_weights/topk_ids: (N, top_k)
        out = torch.zeros_like(x)
        inter = self.intermediate_size
        for e in range(self.num_experts):
            token_idx, slot_idx = (topk_ids == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            xe = x[token_idx]
            gu = F.linear(xe, self.gate_up_proj[e])
            gate, up = gu[..., :inter], gu[..., inter:]
            ye = F.linear(F.silu(gate) * up, self.down_proj[e])
            w = topk_weights[token_idx, slot_idx].to(ye.dtype).unsqueeze(-1)
            out.index_add_(0, token_idx, ye * w)
        return out


class MoEFeedForward(nn.Module):
    """Keys: layers.{N}.feed_forward.{router,experts}.*"""

    def __init__(self, config, layer_id: int, top_k: int, intermediate_size: int):
        super().__init__()
        self.router = Router(config, layer_id, top_k)
        self.experts = GroupedExperts(config, intermediate_size)

    def forward(
        self, x: torch.Tensor, router_states: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, hidden = x.shape
        flat = x.reshape(-1, hidden)
        rs_flat = router_states.reshape(-1, router_states.shape[-1]) if router_states is not None else None
        topk_weights, topk_ids, rs_next = self.router(flat, rs_flat)
        out = self.experts(flat, topk_weights, topk_ids)
        return out.view(B, T, hidden), rs_next.view(B, T, -1)


# ---------------------------------------------------------------------------
# Transformer block and full model
# ---------------------------------------------------------------------------


def _resolve_layer_topk(config, layer_id: int) -> int:
    default_topk = max(int(getattr(config, "moe_router_topk", 1) or 1), 1)
    special = getattr(config, "special_topk_layers", None)
    if special:
        return int(special.get(layer_id, special.get(str(layer_id), default_topk)))
    return default_topk


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_id: int, intermediate_size: int):
        super().__init__()
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

        n_layers = config.n_layers
        self.is_moe = (
            config.moe_n_experts > 1
            and layer_id >= config.moe_start_from_layer
            and (n_layers - layer_id) > config.moe_end_from_layer
        )
        if self.is_moe:
            self.feed_forward = MoEFeedForward(
                config, layer_id, _resolve_layer_topk(config, layer_id), intermediate_size
            )
        else:
            self.feed_forward = FeedForward(config, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        router_states: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Fused-add RMSNorm residual pattern (norm.py:48-55).
        if residual is None:
            residual = x
            h = self.attention_norm(x)
        else:
            residual = residual + x
            h = self.attention_norm(residual)

        attn_out = self.attention(h, cos, sin)

        residual = residual + attn_out
        h = self.ffn_norm(residual)

        if self.is_moe:
            ffn_out, router_states = self.feed_forward(h, router_states)
        else:
            ffn_out = self.feed_forward(h)
            router_states = None  # dense layers reset the EDA state, as in inference
        return ffn_out, residual, router_states


class Zonos2Trainable(nn.Module):
    """Trainable Zonos2 with checkpoint-identical state_dict keys."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_codebooks = config.n_codebooks
        self.audio_vocab = config.codebook_size + 2
        self.loss_softcap = float(getattr(config, "loss_softcap", 0.0) or 0.0)

        # intermediate_size: ffn_dim_multiplier * dim rounded up to multiple_of
        ffn_dim = int(config.ffn_dim_multiplier * config.dim)
        multiple_of = int(getattr(config, "multiple_of", 256) or 256)
        intermediate_size = multiple_of * ((ffn_dim + multiple_of - 1) // multiple_of)

        self.multi_embedder = MultiEmbedding(config)
        # emb_norm has no parameters (elementwise_affine=False).

        self.speaker_lda_projection = None
        self.speaker_projection = None
        if getattr(config, "speaker_enabled", False):
            spk_in = config.speaker_embedding_dim
            if getattr(config, "speaker_lda_dim", None):
                self.speaker_lda_projection = nn.Linear(
                    spk_in, int(config.speaker_lda_dim), bias=True
                )
                spk_in = int(config.speaker_lda_dim)
            self.speaker_projection = nn.Linear(spk_in, config.dim, bias=True)

        self.layers = nn.ModuleList(
            [TransformerBlock(config, i, intermediate_size) for i in range(config.n_layers)]
        )
        self.out_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.multi_output = nn.Linear(
            config.dim, self.audio_vocab * self.n_codebooks, bias=False
        )

        # RoPE tables (interleaved, base=rope_theta, rotary_dim == head_dim).
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float) / config.head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.gradient_checkpointing = False

    # -- forward -----------------------------------------------------------

    def _rope_tables(self, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(T, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.rope_inv_freq.to(device))
        return freqs.cos(), freqs.sin()

    def forward(
        self,
        frames: torch.Tensor,
        speaker_embedding: Optional[torch.Tensor] = None,
        speaker_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """frames: (B, T, n_codebooks+1) int; returns logits (B, T, n_codebooks, audio_vocab).

        speaker_embedding: (B, speaker_embedding_dim) fp32; speaker_positions: (B,)
        position of the reserved speaker slot in each sequence (typically 0).
        """
        B, T, _ = frames.shape
        x = self.multi_embedder(frames.long())

        if self.speaker_projection is not None and speaker_embedding is not None:
            s = speaker_embedding.to(self.speaker_projection.weight.dtype)
            if self.speaker_lda_projection is not None:
                s = self.speaker_lda_projection(
                    speaker_embedding.to(self.speaker_lda_projection.weight.dtype)
                )
                s = s.to(self.speaker_projection.weight.dtype)
            projected = self.speaker_projection(s)  # (B, hidden)
            if speaker_positions is None:
                speaker_positions = torch.zeros(B, dtype=torch.long, device=frames.device)
            x = x.clone()
            x[torch.arange(B, device=x.device), speaker_positions] = projected.to(x.dtype)

        # Parameterless emb_norm, then the residual stream starts fresh (zonos2.py:875).
        x = rms_norm_fp32(x, None, self.config.norm_eps)

        cos, sin = self._rope_tables(T, x.device)
        residual: Optional[torch.Tensor] = None
        router_states: Optional[torch.Tensor] = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, residual, router_states = torch.utils.checkpoint.checkpoint(
                    layer, x, residual, router_states, cos, sin, use_reentrant=False
                )
            else:
                x, residual, router_states = layer(x, residual, router_states, cos, sin)

        h = self.out_norm(x + residual)

        logits = self.multi_output(h).view(B, T, self.n_codebooks, self.audio_vocab)
        if self.loss_softcap > 0:
            logits = self.loss_softcap * torch.tanh(logits / self.loss_softcap)
        return logits

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls, model_path: str, dtype: torch.dtype = torch.bfloat16
    ) -> "Zonos2Trainable":
        from zonos2_compat import load_checkpoint_state_dict, load_config

        config = load_config(model_path)
        model = cls(config)
        sd = load_checkpoint_state_dict(model_path)

        # Drop speaker-LDA weights if the config disabled them (mirrors the
        # engine's _check_speaker_lda_weights) and any other stale keys, but
        # report everything so silent mismatches can't slip through.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("speaker_lda_projection.")]
        if missing:
            raise RuntimeError(f"Checkpoint is missing {len(missing)} keys: {missing[:8]} ...")
        if unexpected:
            raise RuntimeError(
                f"Checkpoint has {len(unexpected)} unexpected keys: {unexpected[:8]} ..."
            )

        model.to(dtype)
        # Router math runs in fp32; keep the frozen biases exact.
        for module in model.modules():
            if isinstance(module, Router):
                module.balancing_biases.data = module.balancing_biases.data.float()
        return model

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (
            p for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def tts_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Next-frame CE over all codebooks.

    logits: (B, T, C, V) — position t predicts frame t+1.
    labels: (B, T-1, C) int64 with ignore_index at masked cells; labels[:, t]
    is the target for logits[:, t].
    """
    B, T, C, V = logits.shape
    pred = logits[:, : T - 1].reshape(-1, V).float()
    tgt = labels.reshape(-1)
    return F.cross_entropy(pred, tgt, ignore_index=ignore_index)


__all__ = ["Zonos2Trainable", "tts_cross_entropy", "Linear3D", "rms_norm_fp32"]
