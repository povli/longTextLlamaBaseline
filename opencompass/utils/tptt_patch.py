import importlib
from typing import Any

import torch


def patch_tptt_full_length_mix(model: Any) -> bool:
    module_name = getattr(model.__class__, "__module__", "")
    if not module_name:
        return False
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    if not hasattr(module, "LiZAttention"):
        return False

    liz_attention = module.LiZAttention
    if getattr(liz_attention, "_opencompass_full_length_mix", False):
        return True
    if not hasattr(liz_attention, "_opencompass_apply_mag_orig"):
        liz_attention._opencompass_apply_mag_orig = liz_attention._apply_mag

    def _apply_mag(
        self,
        mag_weight: torch.Tensor,
        linear_attention: torch.Tensor,
        softmax_attention: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if softmax_attention is None:
            return liz_attention._opencompass_apply_mag_orig(
                self, mag_weight, linear_attention, *args, **kwargs
            )

        if self.disable_linear_attn:
            return softmax_attention

        if linear_attention.shape[1] != softmax_attention.shape[1]:
            lin_len = linear_attention.shape[1]
            soft_len = softmax_attention.shape[1]
            if lin_len > soft_len:
                prefix = linear_attention[:, : lin_len - soft_len, :]
                mag_tail = mag_weight[:, -soft_len:, :]
                linear_tail = linear_attention[:, -soft_len:, :]
                softmax_weighted = (1 - mag_tail) * softmax_attention
                linear_weighted = mag_tail * linear_tail
                if self.cross_gate:
                    output_tail = (
                        softmax_weighted
                        + linear_weighted
                        + softmax_weighted * linear_weighted
                    )
                else:
                    output_tail = softmax_weighted + linear_weighted
                return torch.cat([prefix, output_tail], dim=1)

            prefix = softmax_attention[:, : soft_len - lin_len, :]
            mag_tail = mag_weight[:, -lin_len:, :]
            softmax_tail = softmax_attention[:, -lin_len:, :]
            softmax_weighted = (1 - mag_tail) * softmax_tail
            linear_weighted = mag_tail * linear_attention
            if self.cross_gate:
                output_tail = (
                    softmax_weighted
                    + linear_weighted
                    + softmax_weighted * linear_weighted
                )
            else:
                output_tail = softmax_weighted + linear_weighted
            return torch.cat([prefix, output_tail], dim=1)

        softmax_weighted = (1 - mag_weight) * softmax_attention
        linear_weighted = mag_weight * linear_attention
        if self.cross_gate:
            return softmax_weighted + linear_weighted + softmax_weighted * linear_weighted
        return softmax_weighted + linear_weighted

    liz_attention._apply_mag = _apply_mag
    liz_attention._opencompass_full_length_mix = True
    return True
