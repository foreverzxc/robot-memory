"""可学习密码 query token：替代 BERT 语言输入。

思路（用户提出）：类似 DETR 的可学习 query——只有两个基向量分别代表左/右，
给定密码串后做带位置编码的线性组合，得到一个（或一组）token 作为条件输入。

两种模式：
- sum：1 个 token = sum_i (char_embed[c_i] * pos_embed[i])（Hadamard 乘积保留顺序信息）
- seq：6 个 token = char_embed[c_i] + pos_embed[i]（显式逐位 token）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_visual_tokens(visual, num_views, pool_per_view):
    """把每视角 HxW patch 平均池化为 grid_h x grid_w 个 token。"""
    B, L, D = visual.shape
    n = L // num_views
    H = W = int(round(n ** 0.5))
    if H * W != n:
        raise ValueError(f"cannot reshape {n} tokens into square grid")
    grid = int(round(pool_per_view ** 0.5))
    if grid * grid != pool_per_view:
        raise ValueError("pool_per_view must be a perfect square (1, 4, 9, ...)")
    x = visual.reshape(B, num_views, H, W, D).permute(0, 1, 4, 2, 3).reshape(B * num_views, D, H, W)
    x = F.adaptive_avg_pool2d(x, (grid, grid))
    return x.reshape(B, num_views, D, grid * grid).permute(0, 1, 3, 2).reshape(B, num_views * grid * grid, D)


def visual_cond_no_text(base, samples, text_len=21, pool_per_view=0):
    """无语言条件：视觉 token 过融合层时文本侧用固定零向量，输出 (B, 512, dim)。
    pool_per_view>0 时跳过融合层，直接平均池化为每视角 pool_per_view 个 token。"""
    pixel = base._normalize_samples(samples)
    with torch.no_grad():
        visual = base.encode_vision(pixel)
        if pool_per_view and pool_per_view > 0:
            return pool_visual_tokens(visual, base.num_views, pool_per_view).float()
        B = visual.shape[0]
        dummy = torch.zeros(B, text_len, visual.shape[-1], device=visual.device, dtype=visual.dtype)
        mask = torch.zeros(B, text_len, dtype=torch.bool, device=visual.device)
        self_attn = (
            torch.eye(text_len, dtype=torch.bool, device=visual.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )
        visual, _ = base.vision_language_interaction(visual, dummy, mask, self_attn)
    return visual.float()


def encode_condition_pooled_visual(base, instructions, samples, pool_per_view=0, cached_text=None):
    """TurboVLA language+vision condition, with optional per-view visual pooling.

    This is the same as ``TurboVLA.encode_condition`` except the visual tokens are pooled
    after the vision-language interaction layers, keeping language tokens intact.
    """
    pixel = base._normalize_samples(samples)
    device = pixel.device
    precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (
        base.config.interaction.compute_precision == "bf16_autocast" and device.type == "cuda"
    ) else torch.no_grad()
    with precision_context:
        if cached_text is None:
            text_tokens, text_key_padding_mask, text_self_attention_masks = base.text_encoder(
                instructions, device=device
            )
        else:
            text_tokens, text_key_padding_mask, text_self_attention_masks = cached_text
        visual_tokens = base.encode_vision(pixel)
        visual_tokens, text_tokens = base.vision_language_interaction(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_key_padding_mask=text_key_padding_mask,
            text_self_attention_masks=text_self_attention_masks,
        )
        if pool_per_view and pool_per_view > 0:
            visual_tokens = pool_visual_tokens(visual_tokens, base.num_views, pool_per_view)
        return torch.cat([visual_tokens, text_tokens], dim=1).float()


class PasswordTokenEncoder(nn.Module):
    def __init__(self, dim=256, max_len=6, mode="sum", char_std=0.1, pos_std=None, table_std=0.5):
        super().__init__()
        assert mode in {"sum", "seq"}
        self.dim = int(dim)
        self.max_len = int(max_len)
        self.mode = mode
        if mode == "seq":
            # 每个位置 x 每个字符一张独立 embedding，顺序与字符完全可区分
            self.table = nn.Parameter(torch.randn(self.max_len, 2, self.dim) * table_std)
            self.end_token = nn.Parameter(torch.zeros(self.dim))
        else:
            self.char_embed = nn.Parameter(torch.randn(2, self.dim) * char_std)
            if pos_std is None:
                pos_std = 2.0
            self.pos_embed = nn.Parameter(torch.randn(self.max_len, self.dim) * pos_std)

    def forward(self, passwords):
        """passwords: list[str]（字符为 '1'/'2'）。返回 (B, L_out, dim)。"""
        outs = []
        for pw in passwords:
            dev = self.table.device if self.mode == "seq" else self.char_embed.device
            chars = torch.tensor([int(c) - 1 for c in str(pw)], dtype=torch.long, device=dev)
            chars = chars[: self.max_len]
            if self.mode == "sum":
                e = self.char_embed[chars]  # (n, dim)
                p = self.pos_embed[: e.shape[0]]  # (n, dim)
                out = (e * p).sum(dim=0, keepdim=True)  # (1, dim)
            else:
                idx = torch.arange(chars.shape[0], device=dev)
                tokens = self.table[idx, chars]  # (n, dim)
                n = tokens.shape[0]
                if n < self.max_len:
                    pad = self.end_token.unsqueeze(0).expand(self.max_len - n, -1)
                    out = torch.cat([tokens, pad], dim=0)
                else:
                    out = tokens
            outs.append(out)
        return torch.stack(outs, dim=0)  # (B, L_out, dim)
