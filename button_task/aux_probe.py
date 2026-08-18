"""Probe heads for observing TTT memory.  Must never train the TTT module.

Usage contract:
- Input to this module must be ``features.detach()``.
- Its loss/optimizer may only update this module's own parameters.
- Never add its loss to the policy/TTT training loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AuxProbeHeads(nn.Module):
    """Observation probes over (detached) policy/TTT features.

    Three heads:
    - ``next_key``: which button comes next (2-class).
    - ``count``: presses completed so far (0..max_len, (max_len+1)-class).
    - ``progress``: normalized progress count/max_len in [0, 1] (regression).
      This is the general-purpose temporal supervision signal; in the TTT
      stages it MAY receive gradient into the TTT module (user decision),
      while next_key/count stay observation-only by default.

    This module must never be part of the main policy loss unless explicitly
    configured for TTT progress supervision.
    """

    def __init__(self, dim: int, hidden_dim: int = 128, max_len: int = 6):
        super().__init__()
        self.max_len = int(max_len)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.next_key = nn.Linear(hidden_dim, 2)
        self.count = nn.Linear(hidden_dim, self.max_len + 1)
        self.progress = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor):
        """features must already be detached. Shape (..., dim)."""
        x = self.act(self.fc(self.norm(features)))
        return self.next_key(x), self.count(x), torch.sigmoid(self.progress(x)).squeeze(-1)

    @staticmethod
    def labels_from_remaining(pw: str, remaining: str):
        """Convert remaining password to next_key index and press_count."""
        count_label = len(pw) - len(remaining)
        if count_label < len(pw):
            next_label = int(pw[count_label]) - 1
        else:
            next_label = 0  # episode already finished; metric will mask it
        return next_label, count_label


@torch.no_grad()
def probe_accuracy(
    probe: AuxProbeHeads,
    features: torch.Tensor,
    next_labels: torch.Tensor,
    count_labels: torch.Tensor,
    valid: torch.Tensor | None = None,
):
    """Return next_key and count accuracy without any graph connection."""
    logits_next, logits_count, progress = probe(features.detach())
    pred_next = logits_next.argmax(dim=-1)
    pred_count = logits_count.argmax(dim=-1)

    next_ok = pred_next == next_labels.to(pred_next.device)
    count_ok = pred_count == count_labels.to(pred_count.device)

    if valid is None:
        return float(next_ok.float().mean()), float(count_ok.float().mean())

    valid = valid.bool().to(next_ok.device)
    n = int(valid.sum().item()) or 1
    return (
        float(next_ok[valid].float().sum().item()) / n,
        float(count_ok[valid].float().sum().item()) / n,
    )
