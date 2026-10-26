# Port of the DSpark draft model of our `speculators` training code (speculators/models/{dflash,dspark}), so that Aurora can train it.
# Parameter names are the ones of our DSpark checkpoints (fc, hidden_norm, layers.N.*, norm, lm_head, markov_head.*, confidence_head.*, d2t, t2d; the
# embedding is the target's and is frozen), so a checkpoint loads directly and a state dict Aurora saves loads into vLLM's DSpark drafter.
# Not ported: the loss (aurora/models/dspark.py uses Aurora's forward KL), sliding-window layers, the confidence head's training.
from typing import Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import or_masks
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
)

from aurora.models.draft.base import Eagle3DraftModel
from aurora.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)


class DSparkConfig(PretrainedConfig):
    """The config.json of a DSpark checkpoint: the draft's decoder (`transformer_layer_config`, a Qwen3 config) plus the DSpark fields."""

    model_type = "dspark"

    def __init__(
        self,
        transformer_layer_config: Optional[dict] = None,
        block_size: int = 8,
        aux_hidden_state_layer_ids: Optional[list] = None,
        draft_vocab_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        sample_from_anchor: bool = True,
        sliding_window_non_causal: bool = False,
        speculators_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        tl = dict(transformer_layer_config or {})
        self.transformer_layer_config = tl
        self.block_size = block_size
        self.aux_hidden_state_layer_ids = list(aux_hidden_state_layer_ids or [])
        self.draft_vocab_size = draft_vocab_size
        self.mask_token_id = mask_token_id
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov
        self.sample_from_anchor = sample_from_anchor
        self.sliding_window_non_causal = sliding_window_non_causal
        self.speculators_config = speculators_config
        # what the rest of Aurora reads from a draft config
        self.vocab_size = tl.get("vocab_size")
        self.hidden_size = tl.get("hidden_size")
        self.rms_norm_eps = tl.get("rms_norm_eps", 1e-6)
        self.initializer_range = tl.get("initializer_range", 0.02)   # the std of HF's default init (a from-scratch draft); ours trained from 0.0099

    # The saved config.json must read like our DSpark checkpoints' (vLLM's loader goes by `speculators_model_type`): no model_type, none of the derived fields above.
    _DERIVED = ("model_type", "vocab_size", "hidden_size", "rms_norm_eps", "initializer_range")

    def to_dict(self):
        return {k: v for k, v in super().to_dict().items() if k not in self._DERIVED}

    def to_diff_dict(self):
        return {k: v for k, v in super().to_diff_dict().items() if k not in self._DERIVED}

    def qwen3_config(self) -> Qwen3Config:
        tl = dict(self.transformer_layer_config)
        rope = tl.pop("rope_parameters", None) or {}
        tl.pop("model_type", None)
        return Qwen3Config(**tl, rope_theta=rope.get("rope_theta", 10000.0))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    # q holds only the block (query) positions, k the context plus the blocks: the queries take the last q_len rotary positions
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_len = q.size(-2)
    return (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :]), (k * cos) + (_rotate_half(k) * sin)


class DSparkAttention(nn.Module):
    """Qwen3 attention whose keys and values also come from the target's projected hidden states (the context)."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_heads, self.num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, block_mask):
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        k = torch.cat([self.k_proj(target_hidden), self.k_proj(hidden_states)], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([self.v_proj(target_hidden), self.v_proj(hidden_states)], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)
        out = compile_friendly_flex_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), block_mask=block_mask, enable_gqa=self.num_heads != self.num_kv_heads, scale=self.scaling
        )
        return self.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = DSparkAttention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, block_mask):
        h = hidden_states + self.self_attn(self.input_layernorm(hidden_states), target_hidden, position_embeddings, block_mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class MarkovHead(nn.Module):
    """Low-rank sequential logit bias B = W1 @ W2: W1 is indexed by the previous token (target vocabulary), W2 projects to the draft vocabulary."""

    def __init__(self, *, verifier_vocab_size, draft_vocab_size, markov_rank, hidden_size, head_type="vanilla"):
        super().__init__()
        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        self.head_type, self.markov_rank = head_type, markov_rank
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        nn.init.normal_(self.markov_w1.weight, std=0.01)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        elif head_type == "rnn":
            self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def block_bias(self, prev_token_ids, hidden_states):
        """prev_token_ids [N, block], hidden_states [N, block, hidden] -> bias [N, block, draft_vocab]."""
        prev_emb = self.markov_w1(prev_token_ids.long()).to(self.markov_w2.weight.dtype)
        if self.head_type == "vanilla":
            return self.markov_w2(prev_emb)
        hidden_states = hidden_states.to(prev_emb.dtype)
        if self.head_type == "gated":
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1)))
            return self.markov_w2(gate * prev_emb)
        state = prev_emb.new_zeros(prev_emb.shape[0], self.markov_rank)
        outputs = []
        for k in range(prev_emb.shape[1]):
            gate_raw, cand_raw, out_raw = self.joint_proj(torch.cat([state, prev_emb[:, k], hidden_states[:, k]], dim=-1)).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
            outputs.append(self.markov_w2(torch.tanh(out_raw)))
        return torch.stack(outputs, dim=1)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (one linear layer). Frozen in Aurora for now: see DSparkTrainer."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


def anchor_block_mask_mod(document_ids, total_seq_len, anchor_positions, block_size):
    """Flex-attention mask: query block j (the block of anchor j) sees the base tokens of its own document strictly before the anchor, and its own block."""
    query_anchor = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix(_b, _h, q_idx, kv_idx):
        q_anchor = query_anchor[q_idx]
        kv_pos = torch.remainder(kv_idx, total_seq_len)
        same_doc = (document_ids[q_anchor] == document_ids[kv_pos]) & (document_ids[q_anchor] != -1)
        return (kv_idx < total_seq_len) & same_doc & (kv_pos < q_anchor)

    def same_block(_b, _h, q_idx, kv_idx):
        return (kv_idx >= total_seq_len) & (q_idx // block_size == (kv_idx - total_seq_len) // block_size)

    return or_masks(base_prefix, same_block), anchor_positions.numel() * block_size, total_seq_len + anchor_positions.numel() * block_size


class DSparkDraftModel(Eagle3DraftModel):
    """DFlash-style block drafter (anchored blocks of mask tokens attending to the target's hidden states) with a Markov logit bias."""

    config_class = DSparkConfig
    _keys_to_ignore_on_load_missing = ["embed_tokens.weight"]   # the target's embedding, loaded by load_embedding

    def __init__(self, config: DSparkConfig, attention_backend: str = "flex_attention") -> None:
        super().__init__(config)
        if attention_backend != "flex_attention":
            raise ValueError("DSpark uses flex attention only (attention_backend=flex_attention)")
        qcfg = config.qwen3_config()
        self.config = config
        self.target_vocab_size = qcfg.vocab_size
        self.vocab_size = config.draft_vocab_size or qcfg.vocab_size   # the draft vocabulary (Eagle3's convention)
        self.block_size = config.block_size
        self.sample_from_anchor = config.sample_from_anchor
        self.mask_token_id = config.mask_token_id
        hidden = qcfg.hidden_size
        self.embed_tokens = nn.Embedding(qcfg.vocab_size, hidden)
        self.layers = nn.ModuleList([DSparkDecoderLayer(qcfg) for _ in range(qcfg.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(hidden, eps=qcfg.rms_norm_eps)
        # RoPE is computed on the fly (see rotary): a module buffer for inv_freq would be cast to bf16 on ranks that build the model from a config, and Aurora's state
        # broadcast then mixes dtypes (NaN on ranks > 0)
        self.rope_theta = float(getattr(qcfg, "rope_theta", 10000.0))
        self.head_dim = getattr(qcfg, "head_dim", None) or hidden // qcfg.num_attention_heads
        self.fc = nn.Linear(len(config.aux_hidden_state_layer_ids) * hidden, hidden, bias=False)
        self.hidden_norm = Qwen3RMSNorm(hidden, eps=qcfg.rms_norm_eps)
        self.lm_head = nn.Linear(hidden, self.vocab_size, bias=False)
        self.markov_head = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(verifier_vocab_size=qcfg.vocab_size, draft_vocab_size=self.vocab_size, markov_rank=config.markov_rank,
                                          hidden_size=hidden, head_type=config.markov_head_type)
        self.confidence_head = None
        if config.enable_confidence_head:
            self.confidence_head = ConfidenceHead(hidden + (config.markov_rank if config.confidence_head_with_markov else 0))
        if self.vocab_size != self.target_vocab_size:
            self.register_buffer("t2d", torch.ones(self.target_vocab_size, dtype=torch.bool))
            self.register_buffer("d2t", torch.zeros(self.vocab_size, dtype=torch.int64))
        self.post_init()

    # --- the Eagle3DraftModel interface (the parts that make sense for a block drafter) ---
    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states):
        return self.hidden_norm(self.fc(hidden_states))

    def compute_logits(self, hidden_states):
        return self.lm_head(self.norm(hidden_states))

    def backbone(self, *args, **kwargs):
        raise NotImplementedError("DSpark has no Eagle3 train-time-test backbone: see DSparkModel.forward")

    def rotary(self, x, position_ids):
        """Default (unscaled) RoPE: cos and sin of position_ids [1, S], shape [1, S, head_dim], in x's dtype."""
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim))
        freqs = position_ids[..., None].float() * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    def forward_blocks(self, hidden_states, input_ids, document_ids, position_ids, anchor_positions):
        """hidden_states [1, L, n_aux*H] (target, packed), input_ids [1, L], document_ids [L], position_ids [1, L], anchor_positions [A].

        Returns the draft logits at the anchored blocks [A*block, draft_vocab] (Markov bias included) and the packed indices of those positions [A*block].
        """
        block = self.block_size
        total = hidden_states.shape[1]
        a = anchor_positions.numel()
        mask_mod, q_len, kv_len = anchor_block_mask_mod(document_ids, total, anchor_positions, block)
        block_mask = compile_friendly_create_block_mask(mask_mod, None, None, q_len, kv_len, hidden_states.device)

        block_idx = (anchor_positions[:, None] + torch.arange(block, device=hidden_states.device)[None]).reshape(-1)   # packed index of every block slot
        mask_ids = torch.full((1, a * block), self.mask_token_id, dtype=torch.long, device=hidden_states.device)
        mask_ids[:, ::block] = input_ids[:, anchor_positions]   # slot 0 carries the anchor token, the others the mask token
        noise = self.embed_tokens(mask_ids)
        ctx = self.project_hidden_states(hidden_states)

        block_pos = position_ids[0, anchor_positions][:, None] + torch.arange(block, device=hidden_states.device)[None]
        pos_embed = self.rotary(hidden_states, torch.cat([position_ids, block_pos.reshape(1, -1)], dim=1))
        for layer in self.layers:
            noise = layer(noise, ctx, pos_embed, block_mask)
        hidden = self.norm(noise)
        logits = self.lm_head(hidden)
        if self.markov_head is not None:
            # sample_from_anchor: slot k predicts token p+k+1 and its bias is conditioned on the token at p+k (the ground truth of the block)
            block_tokens = input_ids[0, block_idx].view(a, block)
            prev = block_tokens if self.sample_from_anchor else torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
            logits = (logits.view(a, block, -1) + self.markov_head.block_bias(prev, hidden.view(a, block, -1))).view(a * block, -1)
        return logits.view(a * block, -1), block_idx
