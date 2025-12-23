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

    def _mix_full_length(
        self,
        mag_weight: torch.Tensor,
        linear_attention: torch.Tensor,
        softmax_attention: torch.Tensor,
    ) -> torch.Tensor:
        if self.disable_linear_attn:
            if linear_attention.shape[1] == softmax_attention.shape[1]:
                return softmax_attention
            pad_len = linear_attention.shape[1] - softmax_attention.shape[1]
            if pad_len > 0:
                pad = torch.zeros(
                    softmax_attention.shape[0],
                    pad_len,
                    softmax_attention.shape[2],
                    device=softmax_attention.device,
                    dtype=softmax_attention.dtype,
                )
                return torch.cat([pad, softmax_attention], dim=1)
            return softmax_attention[:, -linear_attention.shape[1]:, :]

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
        return _mix_full_length(self, mag_weight, linear_attention, softmax_attention)

    liz_attention._apply_mag = _apply_mag

    if not getattr(liz_attention, "_opencompass_full_length_forward", False):
        liz_attention._opencompass_full_length_forward = True
        liz_attention._opencompass_forward_orig = liz_attention.forward

        def _forward(self, hidden_states, attention_mask=None, **kwargs):
            device = hidden_states.device
            dtype = hidden_states.dtype
            self.base_attn.to(device)

            if self.training:
                kwargs.pop("past_key_value", None)
                kwargs["use_cache"] = False
            elif "use_cache" not in kwargs:
                kwargs.pop("past_key_value", None)
                kwargs["use_cache"] = False

            kwargs.pop("position_ids", None)

            q, k, v, out_proj = self._apply_shared_projections(hidden_states)
            o_lin = self.linear_attn(
                x=[q, k, v], attn_mask=attention_mask, out_proj=out_proj, **kwargs
            )

            o_base, attn_weights, present_key_value, expected_attn_mode = (
                self._process_self_attn(hidden_states, attention_mask, kwargs)
            )

            o_lin, o_base = self._prepare_attn_mixin(o_lin, o_base, dtype, eps=1e-5)
            mag_weight = self.memory_gate(hidden_states)
            out = _mix_full_length(self, mag_weight, o_lin, o_base)

            if expected_attn_mode == 3:
                return out, attn_weights, present_key_value
            if expected_attn_mode == 2:
                return out, attn_weights
            return out

        liz_attention.forward = _forward

    liz_attention._opencompass_full_length_mix = True
    return True
