"""Stage B1: train VQ-BeT (Patch Policy) with password tokens on the button task.

Standalone script — no hydra/wandb/accelerate needed. Run with the button venv:

    $env:MUJOCO_GL = "glfw"
    E:\\WM\\turbovla\\.venv\\Scripts\\python.exe train_button.py --h5 <demos.h5> \
        --out runs/b1_smoke --epochs 50 --batch-size 64

Data iterates in episode shards: frames are read once per shard, resized and
encoded on the GPU per batch (encoder is frozen). Action chunks are 12 steps
(``act_scale`` 1.0, raw 7-D OSC_POSE actions). Password conditioning uses 6
per-position learnable tokens (``seq`` mode) prepended to the GPT sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "patch_policy"))

import einops
import numpy as np
import torch
import torch.nn.functional as F

from button_task.button_dataset import ButtonH5Dataset
from button_task.train_dataset import ButtonSliceDataset
from button_task.env_wrapper import make_button_patch_env
from models.encoder.dino import DinoV2Encoder
from models.vq_behavior_transformer.bet import BehaviorTransformer

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"
# robosuite's numba JIT cache lives in the system temp dir by default; under a
# restricted environment that write is denied and numba loops forever trying
# to probe it. Redirect the cache inside the workspace.
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, help="path to demos.h5")
    p.add_argument("--split",
        default=str(ROOT / "button_task" / "password_split.json"),
        help="train/holdout password split JSON")
    p.add_argument("--pw-train", default=None,
                   help="comma-separated train passwords (overrides split)")
    p.add_argument("--pw-eval", default=None,
                   help="comma-separated eval passwords (overrides holdout)")
    p.add_argument("--out", required=True, help="output dir (created)")
    p.add_argument("--resume", default=None, help="snapshot dir to resume from")

    # data
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--views", type=int, default=2, choices=[1, 2])
    p.add_argument("--encoder-mode", default="patch",
                   choices=["patch", "cls", "avg_pool"],
                   help="patch = dense patch tokens (Patch Policy default); "
                        "cls = DINOv2 CLS token (paper baseline); "
                        "avg_pool = mean-pooled patch tokens (paper baseline)")
    p.add_argument("--action-window", type=int, default=12)
    p.add_argument("--max-train-episodes", type=int, default=0)
    p.add_argument("--max-holdout-episodes", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--epoch-fraction", type=float, default=1.0,
                   help="fraction of slices used per epoch (speed knob)")

    # model
    p.add_argument("--cond-mode", default="seq", choices=["seq", "sum", "lookup"])
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--vqvae-n-embed", type=int, default=16)
    p.add_argument("--vqvae-groups", type=int, default=2)
    p.add_argument("--vqvae-latent-dim", type=int, default=512)
    p.add_argument("--vqvae-iters", type=int, default=1000)
    p.add_argument("--vqvae-max-samples", type=int, default=65536)
    p.add_argument("--offset-loss-multiplier", type=float, default=10.0)
    p.add_argument("--act-scale", type=float, default=1.0)

    # optimization
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)

    # eval / logging
    p.add_argument("--eval-freq", type=int, default=10)
    p.add_argument("--eval-train-passwords", type=int, default=0,
                   help="also eval on N train passwords (0 = holdout only)")
    p.add_argument("--max-env-steps", type=int, default=600)
    p.add_argument("--val-max-batches", type=int, default=200)
    p.add_argument("--no-env-eval", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="run N training steps then exit (pipeline check)")
    return p.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["train"], d["holdout"]


@torch.no_grad()
def embed_obs(encoder, obs, device):
    """obs: (B, 1, V, 3, H, W) float [0,1] -> (B, 1, V*P, E)"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = encoder(obs)  # (B, 1, V, P, E)
    return einops.rearrange(emb, "B T V P E -> B T (V P) E")


@torch.no_grad()
def embed_shard_frames(encoder, shard_obs, device, image_size, chunk=256):
    """Encode every frame of a shard once; cache on GPU in fp16.

    shard_obs: (S, V, 3, H, W) uint8 on CPU.
    Returns: (S, 1, V*P, E) fp16 on ``device``.
    """
    emb_list = []
    S = shard_obs.shape[0]
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    for i in range(0, S, chunk):
        obs = shard_obs[i : i + chunk].to(device).float().div_(255.0)  # (c,V,3,H,W)
        B, V, C, H, W = obs.shape
        if obs.shape[-2:] != image_size:
            obs = obs.reshape(B * V, C, H, W)
            obs = F.interpolate(
                obs, size=image_size, mode="bilinear", align_corners=False
            )
            obs = obs.reshape(B, V, C, *image_size)
        obs = obs.unsqueeze(1)  # (c,1,V,3,H,W)
        with amp_ctx:
            emb = encoder(obs)  # (c,1,V,P,E)
        emb_list.append(einops.rearrange(emb, "B T V P E -> B T (V P) E").half())
        del obs
    return torch.cat(emb_list, dim=0)


class EvalEnv:
    """Thin wrapper around one ButtonEnv used for rollout evaluation."""

    def __init__(self, passwords, image_size, max_steps, seed):
        self.passwords = list(passwords)
        self.env = make_button_patch_env(
            seed=seed,
            password=self.passwords[0],
            passwords=self.passwords,
            cameras=("agentview", "robot0_eye_in_hand"),
            image_size=(image_size, image_size),
            horizon=1000,
            max_steps=max_steps,
        )

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def run_episode(self, goal_idx, model, encoder, device, action_window):
        env = self.env
        obs = env.reset(goal_idx=goal_idx)  # (V,3,H,W) float
        pw = env.password
        pw_idx = torch.as_tensor(
            ButtonH5Dataset.encode_password(pw, 6)[0], dtype=torch.long, device=device
        ).unsqueeze(0)
        action_list = []
        done = False
        steps = 0
        info = {}
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        while not done and steps < env.max_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_emb = embed_obs(encoder, obs_t.unsqueeze(0), device)  # (1,1,P,E)
            with amp_ctx:
                pred, _, _, _ = model(obs_emb, None, None, password_idx=pw_idx)
            chunk = pred[0, -1].cpu().numpy()  # (W, 7)
            if action_window > 1:
                action_list.append(chunk)
                if len(action_list) > action_window:
                    action_list = action_list[1:]
                curr = np.mean([c[0] for c in action_list], axis=0)
                action_list = [
                    np.concatenate([c[1:], np.zeros((1, c.shape[-1]), dtype=c.dtype)], axis=0)
                    for c in action_list
                ]
            else:
                curr = chunk[0]
            obs, reward, done, info = env.step(curr.astype(np.float32))
            steps += 1
        return {
            "password": pw,
            "success": bool(info.get("success", False)),
            "failed": bool(info.get("failed", False)),
            "press_count": int(info.get("press_count", 0)),
            "steps": steps,
        }


def eval_in_sim(model, encoder, device, passwords, image_size, max_steps, seed,
                action_window, tag=""):
    model.eval()
    eval_env = EvalEnv(passwords, image_size, max_steps, seed)
    results = []
    try:
        for k in range(len(passwords)):
            r = eval_env.run_episode(k, model, encoder, device, action_window)
            results.append(r)
            print(
                f"  [{tag}] pw={r['password']:>6} success={int(r['success'])} "
                f"failed={int(r['failed'])} press={r['press_count']} steps={r['steps']}"
            )
    finally:
        eval_env.close()
    model.train()
    n = len(results)
    success_rate = sum(r["success"] for r in results) / n
    summary = {
        f"success_rate{tag}": success_rate,
        f"failed_rate{tag}": sum(r["failed"] for r in results) / n,
        f"mean_press{tag}": float(np.mean([r["press_count"] for r in results])),
        f"mean_steps{tag}": float(np.mean([r["steps"] for r in results])),
        f"per_pw{tag}": {
            r["password"]: {"success": r["success"], "press_count": r["press_count"]}
            for r in results
        },
    }
    return summary


def make_model(args, device):
    n_patches = 1 if args.encoder_mode in ("cls", "avg_pool") else (args.image_size // 14) ** 2
    return BehaviorTransformer(
        obs_dim=384,  # DINOv2 ViT-S patch features
        act_dim=7,
        goal_dim=0,
        views=args.views,
        vqvae_latent_dim=args.vqvae_latent_dim,
        vqvae_n_embed=args.vqvae_n_embed,
        vqvae_groups=args.vqvae_groups,
        vqvae_fit_steps=None,  # fit once at end of first epoch
        vqvae_iters=args.vqvae_iters,
        n_patches=n_patches,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        vqvae_encoder_loss_multiplier=1.0,
        vqvae_batch_size=1024,
        act_scale=args.act_scale,
        offset_loss_multiplier=args.offset_loss_multiplier,
        obs_window_size=1,
        act_window_size=args.action_window,
        cond_len=6,
        cond_mode=args.cond_mode,
        cond_num_symbols=2,
        gpt_block_size=1,  # obs window is 1; keeps the attention mask small
        vqvae_max_samples=args.vqvae_max_samples,
    ).to(device)


def save_snapshot(snapshot_path, payload):
    tmp = snapshot_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, snapshot_path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_data = load_split(args.split)
    train_pws = (
        args.pw_train.split(",") if args.pw_train else split_data["train"]
    )
    holdout_pws = (
        args.pw_eval.split(",") if args.pw_eval else split_data["holdout"]
    )

    cameras = ("agentview", "robot0_eye_in_hand")[: args.views]

    train_ds = ButtonSliceDataset(
        args.h5,
        allowed_passwords=train_pws,
        cameras=cameras,
        image_size=(args.image_size, args.image_size),
        action_window=args.action_window,
        shard_size=args.shard_size,
        seed=args.seed,
        max_episodes=args.max_train_episodes,
    )
    holdout_ds = None
    if holdout_pws:
        try:
            holdout_ds = ButtonSliceDataset(
                args.h5,
                allowed_passwords=holdout_pws,
                cameras=cameras,
                image_size=(args.image_size, args.image_size),
                action_window=args.action_window,
                shard_size=args.shard_size,
                seed=args.seed,
                max_episodes=args.max_holdout_episodes,
            )
        except ValueError as e:
            print(f"warning: no holdout episodes in this HDF5 ({e}); "
                  f"validation/eval disabled")
    print(
        f"train episodes={train_ds.num_episodes} slices={train_ds.num_slices} | "
        f"holdout episodes={holdout_ds.num_episodes if holdout_ds else 0} "
        f"slices={holdout_ds.num_slices if holdout_ds else 0}"
    )

    # frozen DINOv2 encoder
    if args.encoder_mode == "patch":
        encoder = DinoV2Encoder(
            name="dinov2_vits14",
            feature_key="x_norm_patchtokens",
            output_dim=384,
            n_patches=(args.image_size // 14) ** 2,
        ).to(device)
    elif args.encoder_mode == "cls":
        encoder = DinoV2Encoder(
            name="dinov2_vits14",
            feature_key="x_norm_clstoken",
            output_dim=384,
            n_patches=1,
        ).to(device)
    else:  # avg_pool
        encoder = DinoV2Encoder(
            name="dinov2_vits14",
            feature_key="x_norm_patchtokens",
            output_dim=384,
            postprocess="avg_pool",
            n_patches=1,
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    model = make_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable model params: {n_params / 1e6:.2f}M")

    # resume or fresh
    start_epoch = 0
    best_success = -1.0
    metrics_log = []
    snapshot_path = out_dir / "snapshot.pt"
    best_path = out_dir / "best.pt"
    if args.resume:
        snap_dir = Path(args.resume)
        snap = torch.load(snap_dir / "snapshot.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(snap["model_state"])
        model.vqvae_is_fit = snap.get("vqvae_is_fit", True)
        start_epoch = snap["epoch"] + 1
        best_success = snap.get("best_success", -1.0)
        metrics_log = snap.get("metrics_log", [])
        print(f"resumed from {snap_dir}, start_epoch={start_epoch}, best_success={best_success}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(0.9, 0.999),
    )
    if args.resume:
        snap = torch.load(snap_dir / "snapshot.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(snap["optimizer_state"])

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    def log_epoch(record):
        metrics_log.append(record)
        with open(out_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    global_step = 0
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        loss_acc = {}

        for shard in train_ds.iter_shards(shuffle=True):
            t_shard = time.time()
            shard_emb = embed_shard_frames(
                encoder, shard.obs, device, (args.image_size, args.image_size)
            )
            chunks = train_ds.build_chunks(shard)
            n_slices = shard.starts.shape[0]
            n_use = max(1, int(n_slices * args.epoch_fraction))
            order = torch.randperm(n_slices)[:n_use]
            for i in range(0, n_use, args.batch_size):
                idx = order[i : i + args.batch_size]
                obs_emb = shard_emb[idx].float()
                act = chunks[idx].to(device, non_blocking=True)
                pw = shard.pw_idx[idx].to(device, non_blocking=True)
                with amp_ctx:
                    pred, loss, loss_dict, _ = model(obs_emb, None, act, password_idx=pw)
                if loss is not None and torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    for k, v in loss_dict.items():
                        if isinstance(v, (int, float)):
                            loss_acc[k] = loss_acc.get(k, 0.0) + v
                    epoch_steps += 1
                    global_step += 1
                    if global_step % 50 == 0:
                        print(
                            f"  [step {global_step}] "
                            f"shard_load={time.time() - t_shard:.1f}s "
                            f"loss={loss.item():.4f}"
                        )
                if args.smoke_steps and global_step >= args.smoke_steps:
                    print(f"smoke: reached {global_step} steps, saving and exiting")
                    save_snapshot(snapshot_path, {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "vqvae_is_fit": model.vqvae_is_fit,
                        "best_success": best_success,
                        "metrics_log": metrics_log,
                    })
                    return 0

        model.finish_epoch()  # fits VQ after the first epoch (vqvae_fit_steps=None)

        dt = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, epoch_steps),
            "steps": epoch_steps,
            "seconds": round(dt, 1),
            "vqvae_fit": model.vqvae_is_fit,
            **{f"train/{k}": v / max(1, epoch_steps) for k, v in loss_acc.items()},
        }

        # validation loss on a few holdout slices
        if holdout_ds is not None and epoch % max(1, args.eval_freq // 2) == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for shard in holdout_ds.iter_shards(shuffle=True):
                    if val_batches >= args.val_max_batches:
                        break
                    shard_emb = embed_shard_frames(
                        encoder, shard.obs, device, (args.image_size, args.image_size)
                    )
                    chunks = holdout_ds.build_chunks(shard)
                    n_slices = shard.starts.shape[0]
                    order = torch.randperm(n_slices)
                    for i in range(0, n_slices, args.batch_size):
                        if val_batches >= args.val_max_batches:
                            break
                        idx = order[i : i + args.batch_size]
                        obs_emb = shard_emb[idx].float()
                        act = chunks[idx].to(device, non_blocking=True)
                        pw = shard.pw_idx[idx].to(device, non_blocking=True)
                        with amp_ctx:
                            _, loss, _, _ = model(obs_emb, None, act, password_idx=pw)
                        if loss is not None:
                            val_loss += loss.item()
                            val_batches += 1
            record["val_loss"] = val_loss / max(1, val_batches)
            model.train()

        # sim evaluation (holdout passwords; optional train passwords for overfit check)
        if (
            not args.no_env_eval
            and holdout_ds is not None
            and model.vqvae_is_fit
            and ((epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1)
        ):
            with torch.no_grad():
                record["eval_holdout"] = eval_in_sim(
                    model, encoder, device, holdout_pws,
                    args.image_size, args.max_env_steps, args.seed,
                    args.action_window, tag="_holdout",
                )
                if args.eval_train_passwords > 0:
                    eval_pws = train_pws[: args.eval_train_passwords]
                    record["eval_train"] = eval_in_sim(
                        model, encoder, device, eval_pws,
                        args.image_size, args.max_env_steps, args.seed,
                        args.action_window, tag="_train",
                    )
            holdout_success = record["eval_holdout"]["success_rate_holdout"]
            if holdout_success > best_success:
                best_success = holdout_success
                save_snapshot(best_path, {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "vqvae_is_fit": model.vqvae_is_fit,
                    "best_success": best_success,
                    "metrics_log": metrics_log,
                })
                print(f"  new best holdout success {best_success:.3f} -> best.pt")

        save_snapshot(snapshot_path, {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "vqvae_is_fit": model.vqvae_is_fit,
            "best_success": best_success,
            "metrics_log": metrics_log,
        })
        log_epoch(record)
        print(
            f"epoch {epoch}: train_loss={record['train_loss']:.4f} "
            f"steps={epoch_steps} {dt:.0f}s"
            + (f" val_loss={record.get('val_loss', float('nan')):.4f}" if "val_loss" in record else "")
            + (f" holdout_succ={record['eval_holdout']['success_rate_holdout']:.3f}" if "eval_holdout" in record else "")
        )

    print(f"done. best holdout success = {best_success:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
