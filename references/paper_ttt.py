"""多层 TTT（RoboTTT 风格，可堆叠 N 层）。

- 每层：快模型 2 层 MLP（GELU），独立 Q/K/V 投影、tanh 门控、slow 分支；
- 每步内层更新：W ← W − η·∇_W ||f_W(K_t) − V_t||²（K/V 为该层投影）；
- 应用：h_l = slow_l(h_{l-1}) + tanh(α_l)·f_{W_l}(Q_l)，层间串联；
- W₀、η、投影、α 由外层动作损失元学习（BPTT）；
- TBPTT 分段截断，快权重数值跨段携带。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


class FastMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.empty(hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def forward(self, x, params=None):
        if params is None:
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        else:
            W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        h = F.gelu(x @ W1 + b1)
        return h @ W2 + b2


def inner_update(fast, params, k, v, lr, create_graph=True, grad_clip=1.0,
                 damp=0.0, init_params=None):
    pred = functional_call(fast, params, (k,))
    loss = F.mse_loss(pred, v)
    grads = torch.autograd.grad(loss, params.values(), create_graph=create_graph)
    flat = torch.cat([g.reshape(-1) for g in grads])
    norm = flat.norm()
    if norm > grad_clip:
        scale = grad_clip / norm.detach()
        grads = [g * scale for g in grads]
    new = {key: p - lr * g for key, (p, g) in zip(params.keys(), zip(params.values(), grads))}
    if damp > 0 and init_params is not None:
        # 向初始快权重回拨（consolidation）：每次更新后保留 damp 比例的新权重，
        # 回拨 (1-damp) 比例到 episode 初始权重，防止在线更新漂移。
        new = {
            key: damp * np_ + (1.0 - damp) * p0
            for key, (np_, p0) in zip(new.keys(), zip(new.values(), init_params.values()))
        }
    return new


class TTTLayer(nn.Module):
    """单层 TTT：输入 in_dim -> 输出 proj_dim。"""

    def __init__(self, in_dim, proj_dim, fast_hidden, base_lr):
        super().__init__()
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim)
        self.base_lr = float(base_lr)
        self.norm = nn.LayerNorm(self.in_dim)
        self.norm_kv = nn.LayerNorm(self.proj_dim)
        self.proj_q = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        self.proj_k = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        self.proj_v = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        self.fast = FastMLP(self.proj_dim, fast_hidden, self.proj_dim)
        self.slow_out = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        if self.in_dim == self.proj_dim:
            nn.init.zeros_(self.slow_out.weight)
            with torch.no_grad():
                self.slow_out.weight.copy_(torch.eye(self.in_dim))
        self.out_ln = nn.LayerNorm(self.proj_dim)
        self.alpha = nn.Parameter(torch.full((self.proj_dim,), 0.001))
        self.log_lr_scale = nn.Parameter(torch.tensor(0.0))

    def inner_lr(self):
        return self.base_lr * torch.exp(self.log_lr_scale).clamp(max=1.0)


class PaperTTT(nn.Module):
    def __init__(
        self,
        in_dim,
        proj_dim=64,
        fast_hidden=128,
        head_hidden=256,
        state_dim=0,
        action_dim=2,
        base_lr=0.1,
        seg_len=24,
        inner_grad_clip=1.0,
        num_count_classes=0,
        out_clip=10.0,
        chunk_size=1,
        num_ttt_layers=1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.seg_len = int(seg_len)
        self.inner_grad_clip = float(inner_grad_clip)
        self.out_clip = float(out_clip)
        self.num_ttt_layers = int(num_ttt_layers)

        self.ttt_layers = nn.ModuleList(
            [
                TTTLayer(in_dim if i == 0 else proj_dim, proj_dim, fast_hidden, base_lr)
                for i in range(self.num_ttt_layers)
            ]
        )
        head_in = proj_dim + state_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.chunk_size * self.action_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.num_count_classes = int(num_count_classes)
        self.count_head = None
        if self.num_count_classes > 0:
            self.count_head = nn.Sequential(
                nn.Linear(head_in, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, self.num_count_classes),
            )

    def inner_lr(self):
        return torch.stack([l.inner_lr() for l in self.ttt_layers]).mean()

    def step(self, x, state, params, create_graph=True):
        """单步：更新每层快权重并计算 head 输入。返回 (new_params, head_input)。"""
        new_params = {}
        h = x
        for i, layer in enumerate(self.ttt_layers):
            hin = layer.norm(h)
            k = layer.norm_kv(layer.proj_k(hin))
            v = layer.norm_kv(layer.proj_v(hin))
            q = layer.norm_kv(layer.proj_q(hin))
            p = inner_update(
                layer.fast, params[i], k, v, layer.inner_lr(),
                create_graph=create_graph, grad_clip=self.inner_grad_clip,
            )
            o = layer.out_ln(layer.fast(q, p))
            if self.out_clip > 0:
                o = o.clamp(-self.out_clip, self.out_clip)
            h = layer.slow_out(h) + torch.tanh(layer.alpha) * o
            new_params[i] = p
        if self.state_dim > 0:
            h = torch.cat([h, state])
        return new_params, h

    def _init_params(self):
        return {
            i: {k: v for k, v in layer.fast.named_parameters()}
            for i, layer in enumerate(self.ttt_layers)
        }

    def forward_sequence(self, x, state=None, bias=None, create_graph=True):
        """整条序列前向。返回 (T/num_chunks, chunk, action) 的动作；若有 count_head 则返回 (actions, counts)。"""
        T = x.shape[0]
        params = self._init_params()
        outs = []
        count_outs = []
        chunk_idx = 0
        for seg_start in range(0, T, self.seg_len):
            if seg_start > 0:
                params = {
                    i: {k: v.detach().requires_grad_(True) for k, v in pl.items()}
                    for i, pl in params.items()
                }
            for t in range(seg_start, min(seg_start + self.seg_len, T)):
                st = state[t] if state is not None else None
                params, head_in = self.step(x[t], st, params, create_graph=create_graph)
                if t % self.chunk_size == 0 and t + self.chunk_size <= T:
                    action = self.head(head_in).view(self.chunk_size, self.action_dim)
                    if bias is not None:
                        action = action + bias[chunk_idx]
                    outs.append(action)
                    if self.count_head is not None:
                        count_outs.append(self.count_head(head_in))
                    chunk_idx += 1
        if self.count_head is not None:
            return torch.stack(outs), torch.stack(count_outs)
        return torch.stack(outs)
