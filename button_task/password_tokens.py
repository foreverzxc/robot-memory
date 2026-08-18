"""Learnable password tokens for the two-button task.

Password strings use characters '1' (left) and '2' (right), with maximum
length 6. This module intentionally has no dependency on TurboVLA.

Modes:
- ``seq``: 6 tokens, one per position. Main mode, recommended.
- ``sum``: 1 token, position-symbol product then sum. Ablation only.
- ``lookup``: 1 token looked up from the full-password table. Ablation only.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

PAD_IDX = 2


class PasswordTokenEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        max_len: int = 6,
        num_buttons: int = 2,
        mode: str = "seq",
        init_std: float = 0.1,
    ):
        super().__init__()
        if max_len <= 0 or max_len > 6:
            raise ValueError("Password length must be in [1, 6]")
        if num_buttons != 2:
            raise ValueError("This task assumes exactly two buttons")

        self.dim = int(dim)
        self.max_len = int(max_len)
        self.num_buttons = int(num_buttons)
        self.mode = mode

        if mode == "seq":
            # shape: (position, button, dim)
            self.table = nn.Parameter(torch.randn(self.max_len, self.num_buttons, self.dim) * init_std)
            self.pad_token = nn.Parameter(torch.zeros(self.dim))
        elif mode == "sum":
            # 1 output token = sum_i(char_embed[char_i] * pos_embed[i])
            self.char_embed = nn.Parameter(torch.randn(self.num_buttons, self.dim) * init_std)
            self.pos_embed = nn.Parameter(torch.randn(self.max_len, self.dim) * 0.5)
        elif mode == "lookup":
            # One full-password token. Supports all lengths <= max_len via
            # prefix offsets, so different lengths never collide.
            offsets = []
            total = 0
            for length in range(1, self.max_len + 1):
                offsets.append(total)
                total += self.num_buttons**length
            self.register_buffer("lookup_offsets", torch.tensor(offsets, dtype=torch.long))
            self.table = nn.Parameter(torch.randn(total, self.dim) * init_std)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _parse(self, password: str) -> tuple[torch.Tensor, torch.Tensor]:
        if not password:
            raise ValueError("password must not be empty")
        if len(password) > self.max_len:
            raise ValueError(f"password length {len(password)} > max_len {self.max_len}")
        chars = torch.tensor([int(c) - 1 for c in password], dtype=torch.long)
        if torch.any(chars < 0) or torch.any(chars >= self.num_buttons):
            raise ValueError(f"password contains unsupported symbols: {password!r}")
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[: len(password)] = True
        return chars, mask

    def forward(self, passwords: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tokens, mask)``.

        ``tokens``:
            - seq mode: (B, max_len, dim), padded positions use pad_token.
            - sum/lookup mode: (B, 1, dim).
        ``mask``:
            - seq mode: (B, max_len) bool, True for valid digits.
            - sum/lookup mode: (B, 1) all True.
        """
        if self.mode == "seq":
            outs = []
            masks = []
            for pw in passwords:
                chars, mask = self._parse(pw)
                dev = self.table.device
                chars = chars.to(dev)
                mask = mask.to(dev)
                chars_padded = torch.zeros(self.max_len, dtype=torch.long, device=dev)
                chars_padded[: chars.shape[0]] = chars
                positions = torch.arange(self.max_len, device=dev)
                real_tokens = self.table[positions, chars_padded]
                pad_tokens = self.pad_token.expand(self.max_len, -1)
                tok = torch.where(mask[:, None], real_tokens, pad_tokens)
                outs.append(tok)
                masks.append(mask)
            return torch.stack(outs, dim=0), torch.stack(masks, dim=0)

        if self.mode == "sum":
            outs = []
            for pw in passwords:
                chars, _ = self._parse(pw)
                dev = self.char_embed.device
                chars = chars.to(dev)
                e = self.char_embed[chars]                 # (L, D)
                p = self.pos_embed[: chars.shape[0]]       # (L, D)
                outs.append((e * p).sum(dim=0, keepdim=True))  # (1, D)
            tokens = torch.stack(outs, dim=0)
            mask = tokens.new_ones(tokens.shape[0], 1, dtype=torch.bool)
            return tokens, mask

        if self.mode == "lookup":
            outs = []
            for pw in passwords:
                chars, _ = self._parse(pw)
                offset = int(self.lookup_offsets[len(pw) - 1].item())
                powers = torch.tensor(
                    [self.num_buttons ** (len(pw) - 1 - i) for i in range(len(pw))],
                    dtype=torch.long,
                )
                index = offset + int((chars * powers).sum().item())
                outs.append(self.table[index : index + 1])
            tokens = torch.stack(outs, dim=0)
            mask = tokens.new_ones(tokens.shape[0], 1, dtype=torch.bool)
            return tokens, mask

        raise ValueError(f"Unknown mode: {self.mode}")
