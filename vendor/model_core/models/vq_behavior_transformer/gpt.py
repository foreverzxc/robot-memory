"""
An adaptation of Andrej Karpathy's nanoGPT implementation in PyTorch.
Original source: https://github.com/karpathy/nanoGPT

Original License:
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Original comments:
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import torch
import einops
import torch.nn as nn
from torch.nn import functional as F


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )

def generate_mask_matrix(npatch, nwindow, per_timestep=False):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    if per_timestep:
        # block-diagonal: tokens of timestep t only attend to timestep t.
        # Cross-time information must then come from the TTT fast weights,
        # which keeps a T>1 training window equivalent to T=1 inference.
        for i in range(nwindow):
            row = torch.cat([zeros] * i + [ones] + [zeros] * (nwindow - i - 1), dim=1)
            rows.append(row)
    else:
        # original block-lower-triangular mask: later timesteps may attend to
        # all previous timesteps inside the window.
        for i in range(nwindow):
            row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
            rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
        # )
        bias = generate_mask_matrix(
            config.n_patches, config.block_size, per_timestep=config.per_timestep_attn
        )
        self.register_buffer(
            "bias",
            bias,
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, mask=None):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if mask is None:
            mask = self.bias[:, :, :T, :T]
        # scaled_dot_product_attention avoids materializing the (B, nh, T, T)
        # attention map; bool True = attend (matches our 0/1 mask convention).
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=(mask != 0),
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_dim: int = 256
    output_dim: int = 256
    n_patches: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    cond_len: int = 0  # number of prepended condition tokens (e.g. password tokens)
    per_timestep_attn: bool = False  # block-diagonal attention: TTT is the only cross-time path


class GPT(nn.Module):
    def __init__(self, config, ttt_module=None):
        """ttt_module: either a single TTTSequence applied at the ln_f output
        (legacy mode), or a list of one TTTSequence per transformer block
        (per-layer RoboTTT mode, applied after each attention block)."""
        super().__init__()
        assert config.input_dim is not None
        assert config.output_dim is not None
        assert config.block_size is not None
        self.config = config
        self.ttt = ttt_module  # None when TTT disabled
        if isinstance(ttt_module, (list, tuple)):
            assert len(ttt_module) == config.n_layer, (
                f"per-layer TTT expects one module per block "
                f"({config.n_layer}), got {len(ttt_module)}"
            )
            self.ttt_layers = nn.ModuleList(ttt_module)
            self.per_layer_ttt = True
        else:
            self.ttt_layers = None
            self.per_layer_ttt = False

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.input_dim, config.n_embd),
                wpe=nn.Embedding(
                    config.block_size * config.n_patches + config.cond_len,
                    config.n_embd,
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def ttt_modules(self):
        """All TTT modules in forward order ([] when disabled)."""
        if self.ttt is None:
            return []
        if self.per_layer_ttt:
            return list(self.ttt_layers)
        return [self.ttt]

    def init_ttt_fast_weights(self, batch: int):
        """Fresh fast weights; returns a list in per-layer mode, dict otherwise."""
        if not self.ttt_modules():
            return None
        if self.per_layer_ttt:
            return [ttt.init_fast_weights(batch) for ttt in self.ttt_layers]
        return self.ttt.init_fast_weights(batch)

    def set_ttt_tbptt_step_size(self, step_size):
        for ttt in self.ttt_modules():
            ttt.tbptt_step_size = step_size

    def _attention_mask(self, t: int, cond_len: int, device):
        """Attention mask over [cond tokens; obs tokens].

        - cond tokens attend to all cond tokens (bidirectional),
        - obs tokens attend to all cond tokens,
        - obs tokens keep the original block-causal mask among themselves.
        """
        p = self.config.n_patches
        T = t * p
        obs_mask = self.transformer.h[0].attn.bias[:, :, :T, :T].to(device)
        if cond_len <= 0:
            return obs_mask
        full = obs_mask.new_zeros((1, 1, cond_len + T, cond_len + T))
        full[:, :, :cond_len, :cond_len] = 1.0
        full[:, :, cond_len:, :cond_len] = 1.0
        full[:, :, cond_len:, cond_len:] = obs_mask
        return full

    def forward(
        self,
        input,
        cond=None,
        prev_fast_weights=None,
        return_progress=False,
        debug=False,
        return_features=False,
        targets=None,
        ttt_valid=None,
    ):
        device = input.device
        b, t, p, d = input.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        cond_len = cond.shape[1] if cond is not None else 0
        if cond is None and self.config.cond_len > 0:
            raise ValueError(
                f"GPT configured with cond_len={self.config.cond_len} but no cond was passed"
            )
        # positions: cond tokens take 0..cond_len-1, obs tokens take cond_len..
        tok_emb = self.transformer.wte(input)  # token embeddings of shape (b, t, n_embd)
        if self.config.per_timestep_attn:
            # every timestep block uses the SAME per-patch positions so a T>1
            # training window is numerically identical to T=1 inference at each
            # step (no cross-time attention, no timestep-dependent position).
            pos = torch.arange(
                cond_len, cond_len + p, dtype=torch.long, device=device
            )
            pos_emb = self.transformer.wpe(pos)  # (p, n_embd)
            pos_emb = einops.repeat(pos_emb, 'p d -> b t p d', b=b, t=t)
        else:
            pos = torch.arange(
                cond_len, cond_len + t * p, dtype=torch.long, device=device
            ).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
            pos_emb = einops.rearrange(pos_emb, 'b (t p) d -> b t p d', t=t)
        x = self.transformer.drop(tok_emb + pos_emb)
        x = einops.rearrange(x, 'b t p d -> b (t p) d')
        if cond is not None:
            cond_emb = self.transformer.wte(cond)  # (b, L, n_embd)
            cond_pos = torch.arange(
                0, cond_len, dtype=torch.long, device=device
            ).unsqueeze(0)
            cond_pos_emb = self.transformer.wpe(cond_pos)  # (1, L, n_embd)
            cond_x = self.transformer.drop(cond_emb + cond_pos_emb)
            x = torch.cat([cond_x, x], dim=1)  # (b, L + t*p, n_embd)
        mask = self._attention_mask(t, cond_len, device)

        # --- optional TTT ---
        # per-layer RoboTTT mode: after every attention block, apply that
        # block's TTT module to the per-timestep observation token stream.
        next_fast_weights = None
        ttt_stats = None
        layer_stats = []

        if self.per_layer_ttt:
            if prev_fast_weights is None:
                prev_list = [None] * len(self.ttt_layers)
            else:
                prev_list = list(prev_fast_weights)
                assert len(prev_list) == len(self.ttt_layers), (
                    f"per-layer TTT expects {len(self.ttt_layers)} fast-weight "
                    f"states, got {len(prev_list)}"
                )
            next_fast_weights = [None] * len(self.ttt_layers)
            for bi, block in enumerate(self.transformer.h):
                x = block(x, mask=mask)
                ttt = self.ttt_layers[bi]
                is_last = bi == len(self.ttt_layers) - 1
                cond_feat = x[:, :cond_len]  # (b, L, e)
                obs_feat = x[:, cond_len:].reshape(b, t, p, self.config.n_embd)
                ttt_out, next_fast_weights[bi], stats_i = ttt(
                    obs_feat,
                    prev_fast_weights=prev_list[bi],
                    return_progress=(
                        return_progress and is_last and ttt.progress_head is not None
                    ),
                    debug=debug,
                    return_features=(return_features and is_last),
                    valid=ttt_valid,
                )
                x = torch.cat(
                    [cond_feat, ttt_out.reshape(b, t * p, self.config.n_embd)],
                    dim=1,
                )
                if stats_i:
                    layer_stats.append(stats_i)
            x = self.transformer.ln_f(x)
            if layer_stats:
                ttt_stats = {}
                for stats in layer_stats:
                    for key, val in stats.items():
                        if key in ("progress", "features"):
                            ttt_stats[key] = val
                        elif isinstance(val, (int, float)):
                            ttt_stats[key] = ttt_stats.get(key, 0.0) + val / len(layer_stats)
                        else:
                            # tensor-valued auxiliary outputs (e.g. count /
                            # next-key logits) are passed through as-is
                            ttt_stats[key] = val
        else:
            for block in self.transformer.h:
                x = block(x, mask=mask)
            x = self.transformer.ln_f(x)

            # legacy single TTT at the ln_f output
            if self.ttt is not None:
                cond_feat = x[:, :cond_len]  # (b, L, e), shared across timesteps
                obs_feat = x[:, cond_len:].reshape(b, t, p, self.config.n_embd)
                ttt_out, next_fast_weights, ttt_stats = self.ttt(
                    obs_feat,
                    prev_fast_weights=prev_fast_weights,
                    return_progress=return_progress,
                    debug=debug,
                    return_features=return_features,
                    valid=ttt_valid,
                )
                x = torch.cat(
                    [cond_feat, ttt_out.reshape(b, t * p, self.config.n_embd)], dim=1
                )

        logits = self.lm_head(x)
        logits = logits[:, cond_len:]  # drop cond-token outputs
        logits = einops.rearrange(logits, 'b (t p) d -> b t p d', t=t)
        logits = logits[:, :, -1]  # keep only the last patch token per timestep
        return logits, next_fast_weights, ttt_stats

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter; anything left over
        # (e.g. raw nn.Parameter gates from an attached TTT module) gets no
        # weight decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        leftover = set(param_dict.keys()) - union_params
        no_decay |= leftover

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
