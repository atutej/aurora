# The module DSparkTrainer calls, in the place of Eagle3Model: same forward interface and return value (per-position forward-KL losses and accuracies),
# but the "positions" are the slots of DSpark's anchored blocks instead of the steps of EAGLE-3's train-time-test unroll.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.models.eagle3 import PrecomputedTarget


class DSparkModel(nn.Module):
    def __init__(self, draft_model, max_anchors: int = 3072, anchor_pad_multiple: int = 256):
        super().__init__()
        if not draft_model.sample_from_anchor:
            raise ValueError("DSparkModel supports sample_from_anchor=True only (the default of our DSpark drafters)")
        self.draft_model = draft_model
        self.length = draft_model.block_size   # what Eagle3Model calls the unroll length: here the number of slots per block
        self.vocab_pruning = True
        self.max_anchors = max_anchors
        self.anchor_pad_multiple = anchor_pad_multiple   # anchors are padded to a multiple of this so flex attention sees few distinct shapes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: PrecomputedTarget,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values=None,
        position_ids=None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """input_ids, loss_mask, attention_mask: (B, T); hidden_states: (B, T, n_aux*H); target: probabilities over the draft vocabulary at every position
        (the next-token distribution of the target given the prefix up to and including that position; NOT shifted).

        The batch is packed into one sequence of documents. Anchors are drawn at random among the supervised positions whose whole block lies inside its
        document; slot k of the block at anchor p predicts token p+k+1 and is trained against target[p+k].
        """
        if not isinstance(target, PrecomputedTarget):
            raise ValueError("DSpark trains on a pruned draft vocabulary: it needs a PrecomputedTarget")
        dm, block, device = self.draft_model, self.draft_model.block_size, input_ids.device
        bsz, seq = input_ids.shape
        keep = attention_mask.bool().flatten().nonzero().squeeze(-1)   # packed position -> b * T + t
        doc = torch.arange(bsz, device=device)[:, None].expand(bsz, seq).flatten()[keep]
        pos = torch.arange(seq, device=device)[None].expand(bsz, seq).flatten()[keep]
        lens = attention_mask.sum(-1)
        ids = input_ids.clamp(min=0, max=dm.target_vocab_size - 1).flatten()[keep][None]
        hs = hidden_states.reshape(-1, hidden_states.shape[-1])[keep][None]
        supervised = loss_mask.bool().flatten()[keep]
        aligned = (target.position_mask if target.position_mask is not None else loss_mask).bool().flatten()[keep]

        valid = torch.nonzero(supervised & (lens[doc] - pos >= block), as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            zero = hs.new_tensor(0.0, requires_grad=True)
            return [zero for _ in range(block)], [], [hs.new_tensor(0.0) for _ in range(block)]
        k = min(self.max_anchors, valid.numel())
        anchors = valid[torch.randperm(valid.numel(), device=device)[:k]].sort().values   # contiguous anchors keep flex attention's blocks dense
        n_pad = -k % self.anchor_pad_multiple
        anchor_valid = torch.cat([torch.ones(k, dtype=torch.bool, device=device), torch.zeros(n_pad, dtype=torch.bool, device=device)])
        anchors = torch.cat([anchors, anchors.new_zeros(n_pad)])   # padding anchors: harmless (they see only their own block), masked out of the loss

        logits, block_idx = dm.forward_blocks(hs, ids, doc, pos[None], anchors)
        with torch.no_grad():
            tp = target.target_p_padded[doc[block_idx], pos[block_idx]]   # (A*block, V_draft)
        slot_mask = (aligned[block_idx] & anchor_valid.repeat_interleave(block)).view(-1, block)

        log_p = F.log_softmax(logits.float(), dim=-1)
        kl = -(tp * log_p).sum(-1).view(-1, block)
        hit = (logits.argmax(-1) == tp.argmax(-1)).float().view(-1, block)
        plosses, acces = [], []
        for slot in range(block):
            m = slot_mask[:, slot]
            if not m.any():
                plosses.append(logits.new_tensor(0.0, requires_grad=True))
                acces.append(logits.new_tensor(0.0))
            else:
                plosses.append(kl[:, slot][m].mean())
                acces.append(hit[:, slot][m].mean())
        return plosses, [], acces
