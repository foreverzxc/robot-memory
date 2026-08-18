"""RoboTTT 式 decoder-TTT：TTT 层插在 TurboVLA ACTDecoder 每层之后。

- TTT 流 = 1 个 register token（承载记忆中的世界信息）+ 12 个动作 token；
- 内层更新：对整批 token 做 K→V 的 mini-batch MSE 关联；
- 应用：O = slow(x) + tanh(α)·f_W(Q)，逐层门控；
- 快权重跨 replan 携带，TBPTT 截断训练。
"""

import torch
import torch.nn as nn

from paper_ttt import FastMLP, inner_update


class TTTDecoderLayer(nn.Module):
    def __init__(self, dim=256, fast_hidden=256, base_lr=0.1, out_clip=10.0, grad_clip=1.0,
                 legacy_slow=False):
        super().__init__()
        self.dim = int(dim)
        self.base_lr = float(base_lr)
        self.out_clip = float(out_clip)
        self.grad_clip = float(grad_clip)
        self.legacy_slow = bool(legacy_slow)
        self.norm = nn.LayerNorm(self.dim)
        self.norm_kv = nn.LayerNorm(self.dim)
        self.proj_q = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.dim, bias=False)
        self.fast = FastMLP(self.dim, fast_hidden, self.dim)
        self.slow_out = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.zeros_(self.slow_out.weight)
        with torch.no_grad():
            self.slow_out.weight.copy_(torch.eye(self.dim))
        self.out_ln = nn.LayerNorm(self.dim)
        self.alpha = nn.Parameter(torch.full((self.dim,), 0.001))
        self.log_lr_scale = nn.Parameter(torch.tensor(0.0))

    def inner_lr(self):
        return self.base_lr * torch.exp(self.log_lr_scale).clamp(max=1.0)

    def forward_stream(self, x, params, create_graph=True, update=True, debug=False,
                       damp=0.0, init_params=None, inner_kv_mode="all", n_extra=0):
        """x: (n_tokens, dim)。mini-batch 内层更新 + apply。返回 (out, params)。"""
        xn = self.norm(x)
        if inner_kv_mode == "macro":
            kv_x = xn[: 1 + n_extra]  # register + 密码/state：宏观任务状态
        elif inner_kv_mode == "register":
            kv_x = xn[:1]            # 只看 register 汇总 token
        elif inner_kv_mode == "actions":
            kv_x = xn[1 + n_extra :]  # 只看 12 个动作细节 token（对照组）
        else:
            kv_x = xn
        k = self.norm_kv(self.proj_k(kv_x))
        v = self.norm_kv(self.proj_v(kv_x))
        q = self.norm_kv(self.proj_q(xn))
        if update:
            params = inner_update(
                self.fast, params, k, v, self.inner_lr(),
                create_graph=create_graph, grad_clip=self.grad_clip,
                damp=damp, init_params=init_params,
            )
        o = self.out_ln(self.fast(q, params))
        if self.out_clip > 0:
            o = o.clamp(-self.out_clip, self.out_clip)
        if self.legacy_slow:
            base = self.slow_out(xn)
            out = base + torch.tanh(self.alpha) * o
        else:
            base = x
            out = base + torch.tanh(self.alpha) * o
        if not debug:
            return out, params
        info = {
            "base_norm": base.detach().norm(),
            "ttt_norm": (torch.tanh(self.alpha) * o).norm().detach(),
            "gate_mean": torch.tanh(self.alpha).mean().detach(),
            "reg_before": x[0].detach(),
            "reg_after": out[0].detach(),
            "fast_norm": params["W1"].detach().norm(),
            "inner_lr": self.inner_lr().detach(),
        }
        return out, params, info


class AuxHeads(nn.Module):
    """从 TTT 流输出预测任务状态的简单 MLP 头：
    - next_key：下一次该按哪个键（1/2 -> 2 分类）
    - count：已经按了几次（0..6 -> 7 分类）
    输入 = register token 输出 + 12 个 action token 输出的均值池化。
    """

    def __init__(self, dim=256, hidden=128, max_count=6):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.fc1 = nn.Linear(2 * dim, hidden)
        self.act = nn.GELU()
        self.next_key = nn.Linear(hidden, 2)
        self.count = nn.Linear(hidden, max_count + 1)

    def forward(self, last_out, n_extra):
        reg = last_out[0]
        acts = last_out[1 + n_extra :].mean(dim=0)
        x = self.act(self.fc1(self.norm(torch.cat([reg, acts], dim=-1))))
        return self.next_key(x), self.count(x)


class DecoderWithTTT(nn.Module):
    """包装 ACTDecoder：每层 decoder 后接一个 TTT 层，register token 携带 memory 世界信息。"""

    def __init__(self, decoder, num_layers=None, legacy_slow=False):
        super().__init__()
        self.decoder = decoder
        self.num_layers = len(decoder.decoder.layers) if num_layers is None else int(num_layers)
        dim = decoder.action_queries.weight.shape[1]
        self.ttt_layers = nn.ModuleList(
            [TTTDecoderLayer(dim, legacy_slow=legacy_slow) for _ in range(self.num_layers)]
        )
        self.reg_params = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(self.num_layers)])
        self.mem_proj = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim)) for _ in range(self.num_layers)]
        )
        self.aux_heads = AuxHeads(dim)

    def init_ttt_params(self):
        return {
            i: {k: v for k, v in layer.fast.named_parameters()}
            for i, layer in enumerate(self.ttt_layers)
        }

    def forward(self, memory, ttt_params, create_graph=True, update=True, debug=False,
                state_tokens=None, pw_tokens=None, return_aux=False,
                inner_damp=0.0, init_ttt_params=None, inner_kv_mode="all"):
        """memory: (1, L, dim)。返回 (actions (1,12,7), ttt_params)。"""
        hidden = self.decoder.action_queries.weight.unsqueeze(0)  # (1,12,dim)
        mem_mean = memory.mean(dim=1)  # (1,dim)
        if state_tokens is None:
            state_tokens = memory[:, -2:]  # 默认取 memory 末尾 2 个 state token
        if state_tokens.dim() == 3:
            state_tokens = state_tokens.squeeze(0)
        if pw_tokens is not None and pw_tokens.dim() == 3:
            pw_tokens = pw_tokens.squeeze(0)
        n_extra = state_tokens.shape[0] + (pw_tokens.shape[0] if pw_tokens is not None else 0)
        debug_list = []
        last_out = None
        for i, layer in enumerate(self.decoder.decoder.layers):
            hidden = layer(hidden, memory)  # (1,12,dim)
            reg = self.reg_params[i].unsqueeze(0) + self.mem_proj[i](mem_mean)  # (1,dim)
            if pw_tokens is not None:
                stream = torch.cat([reg, pw_tokens, state_tokens, hidden.squeeze(0)], dim=0)
            else:
                stream = torch.cat([reg, state_tokens, hidden.squeeze(0)], dim=0)
            if debug:
                out, ttt_params[i], info = self.ttt_layers[i].forward_stream(
                    stream, ttt_params[i], create_graph=create_graph, update=update, debug=True,
                    damp=inner_damp,
                    init_params=init_ttt_params[i] if init_ttt_params is not None else None,
                    inner_kv_mode=inner_kv_mode, n_extra=n_extra,
                )
                debug_list.append(info)
            else:
                out, ttt_params[i] = self.ttt_layers[i].forward_stream(
                    stream, ttt_params[i], create_graph=create_graph, update=update,
                    damp=inner_damp,
                    init_params=init_ttt_params[i] if init_ttt_params is not None else None,
                    inner_kv_mode=inner_kv_mode, n_extra=n_extra,
                )
            last_out = out
            hidden = out[1 + n_extra :].unsqueeze(0)
        actions = torch.tanh(self.decoder.action_projection(hidden))
        if return_aux:
            aux = self.aux_heads(last_out, n_extra)
            return actions, ttt_params, aux
        if debug:
            return actions, ttt_params, debug_list
        return actions, ttt_params
