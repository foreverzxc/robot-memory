"""Learnable condition tokens injected at the head of the GPT sequence.

Designed for the two-button password task (adapted from the workspace's
``button_task/password_tokens.py``), but kept generic: a batch of integer
indices (B, L) in [0, num_symbols), with ``PAD_IDX`` marking padded slots,
is embedded into ``(B, num_out, dim)`` tokens.

Modes:
- ``seq``: one token per position (L tokens). Main mode.
- ``sum``: one token = sum_i(char_embed[i] * pos_embed[i]). Ablation only.
- ``lookup``: one token looked up from the full-sequence table. Ablation only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAD_IDX = 2


class ConditionTokenEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        max_len: int = 6,
        num_symbols: int = 2,
        mode: str = "seq",
        init_std: float = 0.1,
    ):
        super().__init__()
        if max_len <= 0 or max_len > 6:
            raise ValueError("Condition length must be in [1, 6]")
        if num_symbols < 2:
            raise ValueError("num_symbols must be >= 2")

        self.dim = int(dim)
        self.max_len = int(max_len)
        self.num_symbols = int(num_symbols)
        self.mode = mode

        if mode == "seq":
            # shape: (position, symbol, dim); padding uses a dedicated pad token
            self.table = nn.Parameter(
                torch.randn(self.max_len, self.num_symbols, self.dim) * init_std
            )
            self.pad_token = nn.Parameter(torch.zeros(self.dim))
        elif mode == "sum":
            # 1 output token = sum_i(char_embed[char_i] * pos_embed[i])
            self.char_embed = nn.Parameter(
                torch.randn(self.num_symbols, self.dim) * init_std
            )
            self.pos_embed = nn.Parameter(torch.randn(self.max_len, self.dim) * 0.5)
        elif mode == "lookup":
            # One full-sequence token. Different lengths live in disjoint
            # offsets so prefixes of different lengths never collide.
            offsets = []
            total = 0
            for length in range(1, self.max_len + 1):
                offsets.append(total)
                total += self.num_symbols**length
            self.register_buffer(
                "lookup_offsets", torch.tensor(offsets, dtype=torch.long)
            )
            self.table = nn.Parameter(torch.randn(total, self.dim) * init_std)
            self.pad_token = nn.Parameter(torch.zeros(self.dim))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def num_out_tokens(self) -> int:
        return self.max_len if self.mode == "seq" else 1

    def forward_idx(self, idx: torch.Tensor) -> torch.Tensor:
        """Embed an integer index tensor.

        Args:
            idx: (B, L) int64, values in [0, num_symbols); PAD_IDX pads.
        Returns:
            tokens: (B, num_out, dim)
        """
        if idx.ndim != 2:
            raise ValueError(f"idx must be (B, L), got {idx.shape}")
        if idx.shape[1] > self.max_len:
            raise ValueError(
                f"condition length {idx.shape[1]} > max_len {self.max_len}"
            )
        bad = (idx < 0) | ((idx >= self.num_symbols) & (idx != PAD_IDX))
        if torch.any(bad):
            raise ValueError(f"idx contains out-of-range values: {idx.unique().tolist()}")
        idx = idx.to(next(self.parameters()).device)
        valid = idx != PAD_IDX

        if self.mode == "seq":
            B, L = idx.shape
            chars_padded = torch.where(valid, idx, torch.zeros_like(idx)).clamp(
                0, self.num_symbols - 1
            )
            positions = torch.arange(self.max_len, device=idx.device)[:L]
            real_tokens = self.table[positions, chars_padded]  # (B, L, dim)
            pad_tokens = self.pad_token.expand(B, L, -1)
            return torch.where(valid[..., None], real_tokens, pad_tokens)

        if self.mode == "sum":
            outs = []
            for b in range(idx.shape[0]):
                chars = idx[b][valid[b]]
                e = self.char_embed[chars]                       # (K, D)
                p = self.pos_embed[: chars.shape[0]]             # (K, D)
                outs.append((e * p).sum(dim=0, keepdim=True))    # (1, D)
            return torch.stack(outs, dim=0)

        if self.mode == "lookup":
            outs = []
            powers_cache = {}
            for b in range(idx.shape[0]):
                chars = idx[b][valid[b]]
                L = chars.shape[0]
                if L == 0:
                    outs.append(self.pad_token.expand(1, -1))
                    continue
                offset = int(self.lookup_offsets[L - 1].item())
                if L not in powers_cache:
                    powers_cache[L] = torch.tensor(
                        [self.num_symbols ** (L - 1 - i) for i in range(L)],
                        dtype=torch.long,
                        device=idx.device,
                    )
                index = offset + int((chars * powers_cache[L]).sum().item())
                outs.append(self.table[index : index + 1])
            return torch.stack(outs, dim=0)

        raise ValueError(f"Unknown mode: {self.mode}")
