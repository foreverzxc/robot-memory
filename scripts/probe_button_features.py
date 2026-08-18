"""Probe whether no-TTT policy features encode task progress (next_key / count).

Loads a trained checkpoint, extracts detached GPT features for each demo frame,
then trains AuxProbeHeads on train episodes and reports next_key / press_count
accuracy on holdout episodes. The probe is observation-only: its loss never
touches the policy, matching the hard constraint.

Usage:
    E:\\WM\\turbovla\\.venv\\Scripts\\python.exe scripts\\probe_button_features.py ^
        --ckpt runs/b1_4plus4_cls/snapshot.pt --args runs/b1_4plus4_cls/args.json ^
        --h5 E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5 ^
        --labels runs/b1_4plus4_labels.npz --train-pw ... --eval-pw ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "patch_policy"))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import einops

from button_task.button_dataset import ButtonH5Dataset
from button_task.aux_probe import AuxProbeHeads
from models.encoder.dino import DinoV2Encoder

if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--args", dest="args_file", default=None)
    p.add_argument("--h5", required=True)
    p.add_argument("--labels", required=True, help="labels npz from label_button_demos.py")
    p.add_argument("--train-pw", required=True, help="comma-separated train passwords")
    p.add_argument("--eval-pw", required=True, help="comma-separated eval passwords")
    p.add_argument("--max-episodes-per-pw", type=int, default=0)
    p.add_argument("--probe-epochs", type=int, default=30)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ttt", action="store_true",
                   help="checkpoint has a TTT module; extract post-TTT stream "
                        "features (pooled over tokens) with per-episode fast "
                        "weight carry, matching the eval protocol")
    p.add_argument("--ttt-fast-hidden", type=int, default=256)
    p.add_argument("--ttt-base-lr", type=float, default=0.1)
    return p.parse_args()


def build_encoder(mode, image_size, device):
    if mode == "patch":
        return DinoV2Encoder(
            name="dinov2_vits14", feature_key="x_norm_patchtokens",
            output_dim=384, n_patches=(image_size // 14) ** 2,
        ).to(device)
    if mode == "cls":
        return DinoV2Encoder(
            name="dinov2_vits14", feature_key="x_norm_clstoken",
            output_dim=384, n_patches=1,
        ).to(device)
    return DinoV2Encoder(
        name="dinov2_vits14", feature_key="x_norm_patchtokens",
        output_dim=384, postprocess="avg_pool", n_patches=1,
    ).to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    run_args = {}
    if args.args_file and Path(args.args_file).is_file():
        with open(args.args_file, encoding="utf-8") as f:
            run_args = json.load(f)
    image_size = run_args.get("image_size", 224)
    encoder_mode = run_args.get("encoder_mode", "cls")

    encoder = build_encoder(encoder_mode, image_size, device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    from models.vq_behavior_transformer.bet import BehaviorTransformer

    snap = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_state = snap.get("model_state", snap.get("model"))
    if hasattr(model_state, "state_dict"):
        model_state = model_state.state_dict()
    n_patches = 1 if encoder_mode in ("cls", "avg_pool") else (image_size // 14) ** 2
    ttt_enabled = bool(args.ttt) or run_args.get("ttt") in ("online", "frozen")
    ttt_per_layer = any(
        ".ttt_layers." in k for k in model_state.keys()
    ) if isinstance(model_state, dict) else False
    ttt_module = None
    if ttt_enabled:
        from button_task.ttt_layer import TTTSequence

        if ttt_per_layer:
            n_ttt_blocks = int(run_args.get("n_layer", 8))
            ttt_module = [
                TTTSequence(
                    dim=run_args.get("n_embd", 512),
                    fast_hidden=run_args.get("ttt_fast_hidden", args.ttt_fast_hidden),
                    base_lr=run_args.get("ttt_base_lr", args.ttt_base_lr),
                    num_layers=1,
                    tbptt_step_size=run_args.get("tbptt") or None,
                    damp=run_args.get("damp", 0.0),
                    progress_head=(
                        run_args.get("prog_weight", 0.0) > 0 and i == n_ttt_blocks - 1
                    ),
                )
                for i in range(n_ttt_blocks)
            ]
        else:
            ttt_module = TTTSequence(
                dim=run_args.get("n_embd", 512),
                fast_hidden=run_args.get("ttt_fast_hidden", args.ttt_fast_hidden),
                base_lr=run_args.get("ttt_base_lr", args.ttt_base_lr),
                num_layers=1,
                tbptt_step_size=run_args.get("tbptt") or None,
                damp=run_args.get("damp", 0.0),
                progress_head=run_args.get("prog_weight", 0.0) > 0,
            )
    model = BehaviorTransformer(
        obs_dim=384, act_dim=7, goal_dim=0, views=2,
        vqvae_latent_dim=run_args.get("vqvae_latent_dim", 512),
        vqvae_n_embed=run_args.get("vqvae_n_embed", 16),
        vqvae_groups=run_args.get("vqvae_groups", 2),
        vqvae_fit_steps=None,
        vqvae_iters=run_args.get("vqvae_iters", 1000),
        n_patches=n_patches,
        n_layer=run_args.get("n_layer", 8),
        n_head=run_args.get("n_head", 8),
        n_embd=run_args.get("n_embd", 512),
        dropout=0.0,
        vqvae_encoder_loss_multiplier=1.0,
        vqvae_batch_size=1024,
        act_scale=run_args.get("act_scale", 1.0),
        offset_loss_multiplier=run_args.get("offset_loss_multiplier", 10.0),
        obs_window_size=1,
        act_window_size=run_args.get("action_window", 12),
        cond_len=6,
        cond_mode=run_args.get("cond_mode", "seq"),
        cond_num_symbols=2,
        gpt_block_size=run_args.get("gpt_block_size", 1),
        vqvae_max_samples=None,
        ttt_module=ttt_module,
        per_timestep_attn=run_args.get("per_step_attn", False),
    ).to(device)
    # skip shape-mismatched keys (wpe/bias sized by gpt_block_size)
    model_sd = model.state_dict()
    model_state = {
        k: v for k, v in model_state.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    model.load_state_dict(model_state, strict=False)
    model.vqvae_is_fit = True
    model.eval()
    print(f"loaded {args.ckpt} (ttt_enabled={ttt_enabled})")

    labels = np.load(args.labels, allow_pickle=False)
    train_pws = set(args.train_pw.split(","))
    eval_pws = set(args.eval_pw.split(","))

    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)

    def extract(split_pws, max_eps_per_pw):
        """Return (features (N, dim), next_key (N,), count (N,)) on CPU."""
        feats, keys, counts = [], [], []
        seen = {}
        with h5py.File(args.h5, "r") as f:
            for name in sorted(f.keys()):
                g = f[name]
                pw = str(g.attrs.get("password", ""))
                if pw not in split_pws:
                    continue
                if max_eps_per_pw and seen.get(pw, 0) >= max_eps_per_pw:
                    continue
                seen[pw] = seen.get(pw, 0) + 1
                if f"{name}/count" not in labels:
                    continue
                # labels arrays have T+1 entries (t=0 plus one per action); the
                # T demo frames are the obs BEFORE each action -> keep [:-1]
                count = torch.as_tensor(labels[f"{name}/count"], dtype=torch.long)[:-1]
                key = torch.as_tensor(labels[f"{name}/next_key"], dtype=torch.long)[:-1]
                img = torch.from_numpy(np.asarray(g["image"][()]))  # T H W C uint8
                wrist = torch.from_numpy(np.asarray(g["wrist_image"][()]))
                pw_idx = torch.as_tensor(
                    ButtonH5Dataset.encode_password(pw, 6)[0], dtype=torch.long
                ).expand(img.shape[0], -1)
                fw = None  # fresh fast weights per episode (eval protocol)
                with torch.no_grad():
                    for i in range(0, img.shape[0], 256):
                        b_obs = torch.stack(
                            [
                                F.interpolate(
                                    img[i : i + 256].permute(0, 3, 1, 2).float().div_(255.0),
                                    size=(image_size, image_size), mode="bilinear",
                                    align_corners=False,
                                ),
                                F.interpolate(
                                    wrist[i : i + 256].permute(0, 3, 1, 2).float().div_(255.0),
                                    size=(image_size, image_size), mode="bilinear",
                                    align_corners=False,
                                ),
                            ],
                            dim=1,
                        ).unsqueeze(1).to(device)  # (B,1,V,3,H,W)
                        with amp_ctx:
                            emb = encoder(b_obs)  # (B,1,V,P,E)
                        emb = einops.rearrange(emb, "B T V P E -> B T (V P) E").float()
                        b_pw = pw_idx[i : i + 256].to(device)
                        with amp_ctx:
                            out, fw_new, stats = model._gpt_model(
                                emb,
                                cond=model.cond_encoder.forward_idx(b_pw),
                                prev_fast_weights=fw,
                                return_features=ttt_enabled,
                            )
                        if ttt_enabled:
                            fw = fw_new
                            feats.append(
                                stats["features"].mean(dim=2).squeeze(1).detach().cpu().float()
                            )  # (B, D) pooled over stream tokens
                        else:
                            feats.append(out[:, -1].detach().cpu().float())  # (B, n_embd)
                keys.append(key)
                counts.append(count)
        return (
            torch.cat(feats, dim=0),
            torch.cat(keys, dim=0),
            torch.cat(counts, dim=0),
        )

    print("extracting train features...")
    tr_f, tr_k, tr_c = extract(train_pws, args.max_episodes_per_pw)
    print("extracting eval features...")
    ev_f, ev_k, ev_c = extract(eval_pws, args.max_episodes_per_pw)
    print(f"train frames={tr_f.shape[0]} eval frames={ev_f.shape[0]} dim={tr_f.shape[1]}")

    # label distribution reference (chance baseline)
    valid_tr = tr_k >= 0
    print(f"train next_key label dist: {torch.bincount(tr_k[valid_tr], minlength=2).tolist()}")
    valid_ev = ev_k >= 0
    print(f"eval  next_key label dist: {torch.bincount(ev_k[valid_ev], minlength=2).tolist()}")
    print(f"train count dist: {torch.bincount(tr_c, minlength=7).tolist()}")

    # train probe (probe params only; features detached by construction)
    probe = AuxProbeHeads(dim=tr_f.shape[1], hidden_dim=128, max_len=6).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=0.0)
    n = tr_f.shape[0]
    idx = torch.randperm(n)
    n_train = int(n * 0.8)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]
    for epoch in range(args.probe_epochs):
        probe.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        total = 0.0
        nb = 0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            feats = tr_f[b].to(device)
            v = (tr_k[b] >= 0).to(device)
            logits_k, logits_c, prog = probe(feats.detach())
            loss = (
                F.cross_entropy(logits_k[v], tr_k[b].to(device)[v])
                + F.cross_entropy(logits_c, tr_c[b].to(device))
                + F.mse_loss(prog, tr_c[b].to(device).float() / 6.0)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if epoch % 5 == 0:
            print(f"  probe epoch {epoch}: loss={total / max(1, nb):.4f}", flush=True)

    # eval
    probe.eval()
    with torch.no_grad():
        preds_k, preds_c, preds_p = [], [], []
        for i in range(0, ev_f.shape[0], args.batch_size):
            feats = ev_f[i : i + args.batch_size].to(device)
            logits_k, logits_c, prog = probe(feats)
            preds_k.append(logits_k.argmax(-1).cpu())
            preds_c.append(logits_c.argmax(-1).cpu())
            preds_p.append(prog.cpu())
        pk = torch.cat(preds_k)
        pc = torch.cat(preds_c)
        pp = torch.cat(preds_p)
    v = ev_k >= 0
    next_acc = float((pk[v] == ev_k[v]).float().mean())
    count_acc = float((pc == ev_c).float().mean())
    chance_next = float(torch.bincount(ev_k[v], minlength=2).float().max().item() / v.sum().item())
    chance_count = float(torch.bincount(ev_c, minlength=7).float().max().item() / ev_c.shape[0])
    print(f"\nPROBE next_key acc (valid frames): {next_acc:.3f}  (chance {chance_next:.3f})")
    print(f"PROBE count acc (all frames):      {count_acc:.3f}  (chance {chance_count:.3f})")

    # third signal: normalized progress = count / password_len (0..1)
    prog_labels = ev_c.float() / 6.0
    l1 = float((pp - prog_labels).abs().mean())
    print(f"PROBE progress L1 (pred vs count/6): {l1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
