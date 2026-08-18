"""TTT (test-time training) layer for the button task, pure torch.

Design follows the WM reference implementation (references/paper_ttt.py,
references/decoder_ttt.py) with the stabilization they validated:

- inner update: one mini-batch gradient step W <- W - lr * grad ||f_W(K) - V||^2
  over the tokens of the current timestep (K/V from per-layer projections);
- per-sample updates: every batch element owns an independent fast-weight
  state and receives its own gradient (no cross-batch averaging), matching
  the official RoboTTT ``vmap(grad(_store), in_dims=(0, 0))`` convention;
- apply: out = x + tanh(alpha) * f_W(Q) with alpha small-init (residual gating);
- learnable inner lr (softplus, initialized at ``base_lr``; can grow/shrink);
- inner grad clip, output clip, optional damp toward episode-initial weights;
- TBPTT: fast weights are detached every ``tbptt_step_size`` steps during
  training (graph management only -- the update rule itself never changes);
- train/inference consistency: both paths call the exact same
  :meth:`TTTLayer.forward_stream` with ``update=True``; fast weights are
  carried explicitly and reset per episode (inference) / per batch (training).

No dependency on the robo_ttt package (einx etc. not installed in the button
venv); the mechanism is the same K->V fast-weight association used there.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


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
        if W1.ndim == 3:  # batched fast weights (B, ...): add the token dim
            b1 = b1.unsqueeze(1)
        if W2.ndim == 3:
            b2 = b2.unsqueeze(1)
        h = F.gelu(x @ W1 + b1)
        return h @ W2 + b2


def _as_valid_bool(valid, b: int) -> bool:
    if valid is None:
        return True
    if torch.is_tensor(valid):
        return bool(valid[b].item())
    return bool(valid[b])


def inner_update(
    fast: FastMLP,
    params: Dict[str, torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    lr: torch.Tensor,
    create_graph: bool = True,
    grad_clip: float = 1.0,
    damp: float = 0.0,
    init_params: Optional[Dict[str, torch.Tensor]] = None,
    valid: Optional[torch.Tensor] = None,
    store_grad=None,
) -> Dict[str, torch.Tensor]:
    """One fast-weight SGD step on the K->V association loss, per batch element.

    Mirrors the official RoboTTT ``vmap(grad(_store), in_dims=(0, 0))``
    per-sample gradient computation (the official store returns -MSE and adds
    the delta; here the store returns +MSE and the delta is subtracted, which
    is the same update rule). Each batch element b owns an independent
    fast-weight state and receives the gradient of ``MSE(f_W(k_b), v_b)``
    computed on its own tokens only (no cross-batch averaging). Rows marked
    invalid in ``valid`` are carried over unchanged. The vmap/grad transform
    remains differentiable, so outer meta-gradients still flow through the
    update to W0, lr, and the QKV projections.
    """
    if store_grad is None:
        def _store(params_b, inputs):
            keys, values = inputs
            pred = functional_call(fast, params_b, (keys,))
            return F.mse_loss(pred, values)

        store_grad = vmap(grad(_store, argnums=0), in_dims=(0, 0))
    deltas = store_grad(params, (k, v))

    # per-sample grad clip (fp32 norm; bf16-safe). ``create_graph`` is
    # implicit: torch.func.grad is differentiable by default.
    if grad_clip > 0:
        flat = torch.cat(
            [d.reshape(d.shape[0], -1) for d in deltas.values()], dim=1
        )
        norms = flat.detach().float().norm(dim=1, keepdim=True)
        scale = torch.where(
            norms > grad_clip,
            grad_clip / norms.clamp(min=1e-8),
            torch.ones_like(norms),
        ).to(dtype=flat.dtype)
        deltas = {
            name: d * scale.view(d.shape[0], *([1] * (d.ndim - 1)))
            for name, d in deltas.items()
        }

    B = k.shape[0]
    new_rows = {name: [] for name in params}
    for b in range(B):
        if not _as_valid_bool(valid, b):
            for name, p in params.items():
                new_rows[name].append(p[b])
            continue
        for name, p in params.items():
            updated = p[b] - lr * deltas[name][b]
            if damp > 0 and init_params is not None:
                # consolidation: pull back toward the episode-initial weights
                updated = damp * updated + (1.0 - damp) * init_params[name][b]
            new_rows[name].append(updated)

    return {name: torch.stack(rows, dim=0) for name, rows in new_rows.items()}


class TTTLayer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        fast_hidden: int = 256,
        base_lr: float = 0.1,
        out_clip: float = 10.0,
        grad_clip: float = 1.0,
        gate_init: float = 0.001,
    ):
        super().__init__()
        self.dim = int(dim)
        self.base_lr = float(base_lr)
        self.out_clip = float(out_clip)
        self.grad_clip = float(grad_clip)
        self.norm = nn.LayerNorm(self.dim)
        self.norm_kv = nn.LayerNorm(self.dim)
        self.proj_q = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.dim, bias=False)
        self.fast = FastMLP(self.dim, fast_hidden, self.dim)
        # cache the vmap(grad(...)) transform; its closure only depends on
        # self.fast, while the fast-weight values are passed in per call
        def _store(params_b, inputs):
            keys, values = inputs
            pred = functional_call(self.fast, params_b, (keys,))
            return F.mse_loss(pred, values)

        self._store_grad = vmap(grad(_store, argnums=0), in_dims=(0, 0))
        self.out_ln = nn.LayerNorm(self.dim)
        # residual gate, small init: TTT output starts ~0 and opens as needed
        self.alpha = nn.Parameter(torch.full((self.dim,), gate_init))
        # learnable inner lr (official RoboTTT softplus convention: it can
        # grow or shrink; initialized at ``base_lr``)
        raw_lr = math.log(math.expm1(base_lr)) if base_lr > 0 else -10.0
        self.log_lr = nn.Parameter(torch.tensor(raw_lr))

    def inner_lr(self):
        return F.softplus(self.log_lr)

    def init_params(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.fast.named_parameters()}

    def forward_stream(
        self,
        x,
        params,
        update=True,
        create_graph=True,
        debug=False,
        damp=0.0,
        init_params=None,
        valid=None,
    ):
        """One timestep: optional inner update, then gated residual apply.

        x: (B, N, dim) tokens of the current timestep.
        Returns (out, params, info). ``info`` carries gate/update diagnostics.
        """
        xn = self.norm(x)
        k = self.norm_kv(self.proj_k(xn))
        v = self.norm_kv(self.proj_v(xn))
        q = self.norm_kv(self.proj_q(xn))
        if update:
            # Promote fast weights to fresh leaf tensors when the outer graph
            # must not flow into them: inference (no_grad) OR frozen TTT base
            # parameters (requires_grad=False views). In normal online training
            # the graph is preserved so meta-gradients reach W0 / lr / QKV.
            promote = (not torch.is_grad_enabled()) or not all(
                p.requires_grad for p in params.values()
            )
            if promote:
                params = {
                    key: p.detach().requires_grad_(True) for key, p in params.items()
                }
            # enable_grad so the inner update also works inside torch.no_grad()
            # (inference) with exactly the same numerical rule as in training.
            with torch.enable_grad():
                params = inner_update(
                    self.fast, params, k, v, self.inner_lr(),
                    create_graph=create_graph, grad_clip=self.grad_clip,
                    damp=damp, init_params=init_params, valid=valid,
                    store_grad=self._store_grad,
                )
        o = self.out_ln(self.fast(q, params))
        if self.out_clip > 0:
            o = o.clamp(-self.out_clip, self.out_clip)
        out = x + torch.tanh(self.alpha) * o
        if not debug:
            return out, params, None
        info = {
            "gate_mean": torch.tanh(self.alpha).mean().detach(),
            "gate_abs_mean": torch.tanh(self.alpha).abs().mean().detach(),
            "ttt_out_norm": (torch.tanh(self.alpha) * o).norm().detach(),
            "x_norm": x.detach().norm(),
            "fast_norm": torch.stack([p.detach().norm() for p in params.values()]).norm(),
            "inner_lr": self.inner_lr().detach(),
        }
        return out, params, info


class TTTSequence(nn.Module):
    """Process (B, T, N, dim) token streams step by step, carrying fast weights.

    The same code path serves training (chunked sequences, TBPTT detach) and
    inference (one step at a time, T=1) -- see the module docstring.
    """

    def __init__(
        self,
        dim: int,
        fast_hidden: int = 256,
        base_lr: float = 0.1,
        num_layers: int = 1,
        tbptt_step_size: Optional[int] = None,
        damp: float = 0.0,
        out_clip: float = 10.0,
        grad_clip: float = 1.0,
        gate_init: float = 0.001,
        progress_head: bool = False,
        progress_hidden: int = 128,
        count_head: bool = False,
        next_key_head: bool = False,
        count_classes: int = 7,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.tbptt_step_size = tbptt_step_size
        self.damp = float(damp)
        self.layers = nn.ModuleList(
            [
                TTTLayer(
                    dim, fast_hidden, base_lr, out_clip=out_clip,
                    grad_clip=grad_clip, gate_init=gate_init,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.progress_head = None
        if progress_head:
            self.progress_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, progress_hidden),
                nn.GELU(),
                nn.Linear(progress_hidden, 1),
                nn.Sigmoid(),
            )
        self.count_head = None
        if count_head:
            self.count_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, progress_hidden),
                nn.GELU(),
                nn.Linear(progress_hidden, int(count_classes)),
            )
        self.next_key_head = None
        if next_key_head:
            self.next_key_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, progress_hidden),
                nn.GELU(),
                nn.Linear(progress_hidden, 2),
            )

    def init_fast_weights(self, batch: int):
        """Fresh fast weights (per episode at inference, per batch element in
        training). Each parameter is repeated to (batch, ...) so every batch
        element carries its own fast-weight state (RoboTTT convention).
        ``init_layers`` snapshots the episode-initial weights so ``damp``
        always consolidates toward W0, even when chunks are carried across
        TBPTT boundaries or processed one timestep at a time."""
        layers = [
            {
                k: v.unsqueeze(0).expand(batch, *v.shape)
                for k, v in layer.fast.named_parameters()
            }
            for layer in self.layers
        ]
        init_layers = [
            {k: v.detach().clone() for k, v in pl.items()} for pl in layers
        ]
        return {"step": 0, "layers": layers, "init_layers": init_layers}

    def _detach_layers(self, fw):
        fw["layers"] = [
            {k: v.detach().requires_grad_(True) for k, v in pl.items()}
            for pl in fw["layers"]
        ]
        return fw

    def forward(
        self,
        streams: torch.Tensor,
        prev_fast_weights: Optional[dict] = None,
        return_progress: bool = False,
        debug: bool = False,
        return_features: bool = False,
        valid: Optional[torch.Tensor] = None,
    ) -> tuple:
        """streams: (B, T, N, dim).

        ``valid``: optional (B, T) bool mask. Invalid timesteps do not update
        their fast weights (they are carried unchanged) and can be masked out
        of the outer loss by the caller.

        Returns (out (B, T, N, dim), next_fast_weights, stats).
        If return_progress, stats["progress"] = (B, T) predictions from the
        progress head, computed on a *detached* copy of the input so that its
        gradients can only reach the TTT parameters, never the policy.
        If return_features, stats["features"] = detached out stream (for
        offline probe analysis).
        """
        B, T, N, D = streams.shape
        if valid is not None:
            valid = valid.bool()
            assert valid.shape == (B, T), f"valid must be (B,T), got {valid.shape}"

        if prev_fast_weights is None:
            fw = self.init_fast_weights(B)
        else:
            # fresh dict per call: never mutate the caller's fast weights
            fw = {
                "step": prev_fast_weights.get("step", 0),
                "layers": [dict(pl) for pl in prev_fast_weights["layers"]],
            }
            if "init_layers" in prev_fast_weights:
                fw["init_layers"] = [
                    dict(pl) for pl in prev_fast_weights["init_layers"]
                ]

        # episode-initial weights for optional consolidation (damp).
        # Backward compatibility: fast-weight dicts created before
        # init_layers existed fall back to the weights entering this call.
        if self.damp > 0 and "init_layers" not in fw:
            fw["init_layers"] = [
                {k: v.detach().clone() for k, v in pl.items()}
                for pl in fw["layers"]
            ]
        init_layers = fw.get("init_layers") if self.damp > 0 else None

        outs = []
        progs = []
        count_logits = []
        next_key_logits = []
        stats_accum = []
        # Auxiliary supervision runs as an independent shadow TTT pass on the
        # SAME base parameters but detached inputs: its gradients can reach
        # the TTT parameters only, never the policy backbone.
        has_aux_heads = (
            self.progress_head is not None
            or self.count_head is not None
            or self.next_key_head is not None
        )
        shadow_params = (
            [
                {
                    k: v.unsqueeze(0).expand(B, *v.shape)
                    for k, v in layer.fast.named_parameters()
                }
                for layer in self.layers
            ]
            if (has_aux_heads and return_progress)
            else None
        )
        for t in range(T):
            x_t = streams[:, t]  # (B, N, D)
            step_valid = valid[:, t] if valid is not None else None
            # --- main path: gradient allowed into the policy backbone ---
            for li, layer in enumerate(self.layers):
                init_p = init_layers[li] if init_layers is not None else None
                x_t, new_params, info = layer.forward_stream(
                    x_t, fw["layers"][li], update=True, create_graph=True,
                    debug=debug, damp=self.damp, init_params=init_p,
                    valid=step_valid,
                )
                fw["layers"][li] = new_params
                if debug and info is not None:
                    info["layer"] = li
                    stats_accum.append(info)
            outs.append(x_t)

            # --- auxiliary shadow path: gradient to TTT only ---
            if shadow_params is not None:
                xp = streams[:, t].detach()
                for li, layer in enumerate(self.layers):
                    init_p = init_layers[li] if init_layers is not None else None
                    xp, shadow_params[li], _ = layer.forward_stream(
                        xp, shadow_params[li], update=True, create_graph=True,
                        damp=self.damp, init_params=init_p, valid=step_valid,
                    )
                feat = xp.mean(dim=1)
                if self.progress_head is not None:
                    # task-agnostic progress supervision: gradient allowed
                    # into TTT through this path only
                    progs.append(self.progress_head(feat).squeeze(-1))
                if self.count_head is not None:
                    # task-related probe: detach so no gradient enters TTT
                    count_logits.append(self.count_head(feat.detach()))
                if self.next_key_head is not None:
                    # task-related probe: detach so no gradient enters TTT
                    next_key_logits.append(self.next_key_head(feat.detach()))

            # TBPTT truncation (training only): detach fast weights every K
            # steps, including the final step of a chunk, so the returned
            # fast weights are graph-free and can be carried into the next
            # chunk without stale-graph errors.
            if (
                self.training
                and self.tbptt_step_size is not None
                and (t + 1) % self.tbptt_step_size == 0
            ):
                fw = self._detach_layers(fw)
                if shadow_params is not None:
                    shadow_params = self._detach_layers({"layers": shadow_params})["layers"]

        fw["step"] = fw.get("step", 0) + T
        if not torch.is_grad_enabled():
            # inference: return graph-free fast weights (values are identical)
            fw = self._detach_layers(fw)

        out = torch.stack(outs, dim=1)  # (B, T, N, D)
        stats = {}
        if debug and stats_accum:
            for key, val in stats_accum[0].items():
                if isinstance(val, torch.Tensor):
                    stats[f"ttt/{key}"] = float(
                        torch.stack([s[key] for s in stats_accum]).mean()
                    )
        if self.progress_head is not None and return_progress:
            stats["progress"] = torch.stack(progs, dim=1)  # (B, T)
        if self.count_head is not None and return_progress:
            stats["count_logits"] = torch.stack(count_logits, dim=1)  # (B, T, C)
        if self.next_key_head is not None and return_progress:
            stats["next_key_logits"] = torch.stack(next_key_logits, dim=1)  # (B, T, 2)
        if return_features:
            stats["features"] = out.detach()  # (B, T, N, D)
        return out, fw, stats
