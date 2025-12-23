"""Utilities for lightweight GPU memory logging during generation."""

from __future__ import annotations

import torch

from opencompass.models import HuggingFaceBaseModel


def patch_hf_base_generate_for_mem() -> None:
    """Patch HuggingFaceBaseModel.generate to log per-GPU memory deltas."""
    if getattr(HuggingFaceBaseModel.generate, "_mem_patched", False):
        return

    orig_generate = HuggingFaceBaseModel.generate

    def wrapped(self, inputs, max_out_len, **kwargs):
        devices = []
        base_stats = {}
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
            for i in devices:
                torch.cuda.synchronize(i)
                base_stats[i] = (
                    torch.cuda.memory_allocated(i),
                    torch.cuda.memory_reserved(i),
                )
                torch.cuda.reset_peak_memory_stats(i)

        out = orig_generate(self, inputs, max_out_len, **kwargs)

        if devices:
            for i in devices:
                torch.cuda.synchronize(i)
            parts = []
            for i in devices:
                peak_alloc = torch.cuda.max_memory_allocated(i)
                peak_reserved = torch.cuda.max_memory_reserved(i)
                inc_alloc = max(0, peak_alloc - base_stats[i][0])
                inc_reserved = max(0, peak_reserved - base_stats[i][1])
                parts.append(
                    f"cuda:{i} inc_alloc={inc_alloc/1024**2:.1f}MB "
                    f"inc_reserved={inc_reserved/1024**2:.1f}MB "
                    f"peak_alloc={peak_alloc/1024**2:.1f}MB"
                )
            if hasattr(self, "logger"):
                self.logger.info("[mem] " + " | ".join(parts))
            else:
                print("[mem] " + " | ".join(parts))
        return out

    wrapped._mem_patched = True
    HuggingFaceBaseModel.generate = wrapped
