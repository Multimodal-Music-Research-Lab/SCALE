from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


def compute_losses(outputs: dict, batch: dict, config) -> dict:
    width = outputs["boundary_logits"].shape[1]
    padding = outputs["padding_mask"]
    target_boundary = batch["widen_true_boundaries"][:, :width]
    target_function = batch["true_functions"][:, :width]
    boundary_valid = (~padding) & (~batch["boundary_mask"][:, :width])
    function_valid = (~padding) & (~batch["function_mask"][:, :width])
    pos_weight = torch.as_tensor(float(config.boundary_pos_weight), device=target_boundary.device)
    boundary_bce = F.binary_cross_entropy_with_logits(
        outputs["boundary_logits"], target_boundary, reduction="none", pos_weight=pos_weight
    )
    boundary_bce = masked_mean(boundary_bce, boundary_valid.float())
    local_bce = F.binary_cross_entropy_with_logits(
        outputs["local_boundary_logits"], target_boundary, reduction="none", pos_weight=pos_weight
    )
    local_bce = masked_mean(local_bce, boundary_valid.float())
    label_mask = batch["label_id_masks"][:, :1, :].bool()
    function_logits = outputs["function_logits"].masked_fill(label_mask, -1e4)
    hard_labels = target_function.argmax(dim=-1)
    label_ce = F.cross_entropy(
        function_logits.transpose(1, 2), hard_labels, reduction="none"
    )
    label_ce = masked_mean(label_ce, function_valid.float())
    audio_function_logits = outputs["audio_function_logits"].masked_fill(
        label_mask, -1e4
    )
    audio_label_ce = F.cross_entropy(
        audio_function_logits.transpose(1, 2), hard_labels, reduction="none"
    )
    audio_label_ce = masked_mean(audio_label_ce, function_valid.float())
    if "lyrics_chorus_logits" in outputs:
        chorus_labels = batch["lyrics_chorus_labels"]
        chorus_weights = batch["lyrics_chorus_weights"]
        chorus_valid = chorus_labels >= 0
        chorus_pos_weight = torch.as_tensor(
            float(config.lyrics_chorus_pos_weight), device=target_boundary.device
        )
        if chorus_valid.any():
            chorus_bce = F.binary_cross_entropy_with_logits(
                outputs["lyrics_chorus_logits"][chorus_valid],
                chorus_labels[chorus_valid].float(),
                reduction="none",
                pos_weight=chorus_pos_weight,
            )
            selected_weights = chorus_weights[chorus_valid]
            chorus_bce = (
                chorus_bce * selected_weights
            ).sum() / selected_weights.sum().clamp_min(1.0)
        else:
            chorus_bce = outputs["lyrics_chorus_logits"].sum() * 0.0
    else:
        chorus_bce = label_ce.new_zeros(())
    boundary_loss = boundary_bce + float(config.local_boundary_weight) * local_bce
    total = (
        float(config.function_weight) * label_ce
        + float(config.audio_function_weight) * audio_label_ce
        + float(config.boundary_weight) * boundary_loss
        + float(config.lyrics_chorus_weight) * chorus_bce
    )
    return {
        "loss": total,
        "boundary_bce": boundary_bce.detach(),
        "local_boundary_bce": local_bce.detach(),
        "function_ce": label_ce.detach(),
        "audio_function_ce": audio_label_ce.detach(),
        "lyrics_chorus_bce": chorus_bce.detach(),
    }
