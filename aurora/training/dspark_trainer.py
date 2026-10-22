# Trainer for a DSpark draft model (anchored-block drafter, see aurora/models/dspark.py), as a subclass of Eagle3Trainer: it reuses the trainer's init (FSDP, optimizer,
# checkpointing, the target lm_head), backward (Aurora's 0.8**i weights over the per-position losses), eval and metrics, and overrides only what differs.
#
# What is deliberately the same as Aurora's EAGLE-3 training: the loss (forward KL against the target's distribution restricted to the draft vocabulary), the
# per-position weights 0.8**i (here i = slot of the block), the metric names (train/acc_i, train/simulated_acc_len). What is left out of our own DSpark training for now:
#   * its per-position loss weights (fixed exponential decay, gamma 4), its TV term and its confidence-head loss;
#   * the confidence head: it is FROZEN here (see _build_draft_model). TODO(revisit): train it again once we bring DSpark's own loss terms back; as is, the head would
#     get no gradient from a forward-KL loss and the checkpoint's trained values are kept as they are.
from aurora.models.dspark import DSparkModel
from aurora.models.eagle3 import compute_target_p_padded
from aurora.training.eagle3_trainer import Eagle3Trainer
from aurora.utils.logging import logger


class DSparkTrainer(Eagle3Trainer):
    shard_module_key = "layers."   # FULL_SHARD shards the Linear modules of the decoder layers

    def _build_draft_model(self, draft_model_config, target_model_path: str):
        draft_model = super()._build_draft_model(draft_model_config, target_model_path)
        if draft_model.confidence_head is not None:
            # TODO(revisit): frozen on purpose, see the header of this file. Not in the optimizer (it takes requires_grad parameters only), not in the loss.
            draft_model.confidence_head.requires_grad_(False)
            logger.info("DSpark confidence head frozen (not trained by the forward-KL loss)")
        return draft_model

    def _wrap_draft_model(self, draft_model):
        return DSparkModel(draft_model, max_anchors=getattr(self.args, "dspark_max_anchors", 3072))

    def _forward(self, batch: dict):
        # No shift here (Eagle3Trainer shifts input_ids and the target by one position): DSpark's anchor block reads tokens and hidden states in place.
        input_ids = batch["input_ids"].cuda()
        loss_mask = batch["loss_mask"]
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)
        loss_mask = loss_mask.cuda()
        target = compute_target_p_padded(
            target_hidden_states=batch["last_hidden_states"].cuda(),
            target_lm_head_weight=self.target_lm_head_weight,
            t2d=self.t2d,
            loss_mask=loss_mask,
            length=0,
        )
        plosses, _, acces = self.model(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].cuda(),
            target=target,
            loss_mask=loss_mask,
            hidden_states=batch["hidden_states"].cuda(),
        )
        return plosses, acces
