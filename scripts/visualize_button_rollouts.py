"""Visualize trained button-policy rollouts as GIFs.

Loads a checkpoint trained by train_button_ttt.py, rolls out the policy in
ButtonEnv, and writes annotated agentview GIFs. Passwords are sampled
proportionally from the train/holdout split (default 3:1, i.e. 6 train and 2
holdout GIFs for the current 48/16 split).

Usage:
    E:\\WM\\turbovla\\.venv\\Scripts\\python.exe scripts\\visualize_button_rollouts.py ^
        --ckpt runs/b4_ttt_perlayer_patch112_smoke/best.pt ^
        --args runs/b4_ttt_perlayer_patch112_smoke/args.json ^
        --out runs/b4_ttt_perlayer_patch112_smoke/gifs
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

import einops
import numpy as np
import torch
from PIL import Image, ImageDraw

from button_task.button_dataset import ButtonH5Dataset
from button_task.env_wrapper import make_button_patch_env
from models.encoder.dino import DinoV2Encoder

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--args", dest="args_file", required=True,
                   help="args.json of the training run")
    p.add_argument("--out", default=None, help="output dir (default: ckpt_dir/gifs)")
    p.add_argument("--split", default=str(ROOT / "button_task" / "password_split.json"))
    p.add_argument("--total", type=int, default=8,
                   help="total GIFs; train/holdout sampled proportionally")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="write every Nth frame to keep GIFs smaller")
    p.add_argument("--gif-width", type=int, default=256)
    p.add_argument("--passwords", default=None,
                   help="comma-separated passwords to visualize (overrides split)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_indices(n, k):
    if k <= 0 or n == 0:
        return []
    k = min(k, n)
    if k == 1:
        return [n // 2]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    run_args = json.loads(Path(args.args_file).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_pws, holdout_pws = list(split["train"]), list(split["holdout"])

    if args.passwords:
        selected_train = [p.strip() for p in args.passwords.split(",") if p.strip()]
        selected_holdout = []
        print(f"manual GIFs ({len(selected_train)}): {selected_train}")
    else:
        n_holdout = max(1, round(args.total * len(holdout_pws) / (len(train_pws) + len(holdout_pws))))
        n_train = max(1, args.total - n_holdout)
        selected_train = [train_pws[i] for i in sample_indices(len(train_pws), n_train)]
        selected_holdout = [holdout_pws[i] for i in sample_indices(len(holdout_pws), n_holdout)]
        print(f"train GIFs ({n_train}): {selected_train}")
        print(f"holdout GIFs ({n_holdout}): {selected_holdout}")

    out_dir = Path(args.out) if args.out else (ckpt.parent / "gifs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model config from the training run ----
    image_size = run_args.get("image_size", 112)
    encoder_mode = run_args.get("encoder_mode", "patch")
    action_window = run_args.get("action_window", 12)

    encoder = DinoV2Encoder(
        name="dinov2_vits14",
        feature_key="x_norm_patchtokens" if encoder_mode == "patch" else "x_norm_clstoken",
        output_dim=384,
        n_patches=(image_size // 14) ** 2 if encoder_mode == "patch" else 1,
        postprocess=None if encoder_mode == "patch" else ("avg_pool" if encoder_mode == "avg_pool" else None),
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    snap = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_state = snap.get("model_state", snap.get("model"))
    if hasattr(model_state, "state_dict"):
        model_state = model_state.state_dict()
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint has no usable state_dict")

    from models.vq_behavior_transformer.bet import BehaviorTransformer
    from button_task.ttt_layer import TTTSequence

    n_patches = (image_size // 14) ** 2 if encoder_mode == "patch" else 1
    ttt_enabled = run_args.get("ttt") in ("online", "frozen")
    ttt_per_layer = any(".ttt_layers." in k for k in model_state.keys())
    ttt_module = None
    if ttt_enabled:
        n_ttt_blocks = int(run_args.get("n_layer", 8))
        make_ttt = lambda i: TTTSequence(
            dim=run_args.get("n_embd", 512),
            fast_hidden=run_args.get("ttt_fast_hidden", 256),
            base_lr=run_args.get("ttt_base_lr", 0.01),
            num_layers=1,
            tbptt_step_size=run_args.get("tbptt") or None,
            damp=run_args.get("damp", 0.0),
            progress_head=(
                run_args.get("prog_weight", 0.0) > 0 and i == n_ttt_blocks - 1
            ),
        )
        ttt_module = [make_ttt(i) for i in range(n_ttt_blocks)] if ttt_per_layer else make_ttt(0)

    model = BehaviorTransformer(
        obs_dim=384, act_dim=7, goal_dim=0, views=2,
        vqvae_latent_dim=run_args.get("vqvae_latent_dim", 512),
        vqvae_n_embed=run_args.get("vqvae_n_embed", 16),
        vqvae_groups=run_args.get("vqvae_groups", 2),
        vqvae_fit_steps=None,
        vqvae_iters=run_args.get("vqvae_iters", 300),
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
        act_window_size=action_window,
        cond_len=6,
        cond_mode=run_args.get("cond_mode", "seq"),
        cond_num_symbols=2,
        gpt_block_size=run_args.get("gpt_block_size", 16),
        vqvae_max_samples=None,
        ttt_module=ttt_module,
        per_timestep_attn=run_args.get("per_step_attn", False),
    ).to(device)
    model_sd = model.state_dict()
    model_state = {
        k: v for k, v in model_state.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    model.load_state_dict(model_state, strict=False)
    model.vqvae_is_fit = True
    model.eval()
    print(f"loaded {ckpt} (ttt_enabled={ttt_enabled}, per_layer={ttt_per_layer})")

    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    selected = [(pw, "train") for pw in selected_train] + [(pw, "holdout") for pw in selected_holdout]
    env = make_button_patch_env(
        seed=args.seed,
        password=selected[0][0],
        passwords=[pw for pw, _ in selected],
        cameras=("agentview", "robot0_eye_in_hand"),
        image_size=(image_size, image_size),
        horizon=1000,
        max_steps=args.max_steps,
    )

    try:
        for k, (pw, tag) in enumerate(selected):
            obs = env.reset(goal_idx=k)
            pw_full = torch.as_tensor(
                ButtonH5Dataset.encode_password(pw, 6)[0],
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            frames = []
            fw = None
            action_list = []
            done, steps, info = False, 0, {}

            def add_frame(obs, info, step, press):
                if "image" in info:
                    img = np.asarray(info["image"])
                else:
                    img = np.transpose(obs[0], (1, 2, 0))
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(img).convert("RGB").resize(
                    (args.gif_width, int(args.gif_width * img.shape[0] / img.shape[1])),
                    Image.Resampling.BILINEAR,
                )
                draw = ImageDraw.Draw(img)
                text = f"{tag} {pw} | step {step} | press {press}"
                draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
                draw.text((4, 3), text, fill=(255, 255, 0))
                frames.append(img)

            add_frame(obs, {}, steps, int(info.get("press_count", 0)))
            while not done and steps < env.max_steps:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad(), amp_ctx:
                    emb = encoder(obs_t)
                emb = einops.rearrange(emb, "B V P E -> B (V P) E").unsqueeze(1)
                # Full password is constant for the whole rollout; dynamic
                # remaining-suffix conditioning is forbidden (progress leak).
                with torch.no_grad(), amp_ctx:
                    pred, _, _, ttt_out = model(
                        emb, None, None, password_idx=pw_full,
                        prev_fast_weights=fw,
                    )
                if ttt_enabled:
                    fw = ttt_out["next_fast_weights"]
                chunk = pred[0, -1].cpu().numpy()
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
                if steps % args.frame_stride == 0 or done:
                    add_frame(obs, info, steps, int(info.get("press_count", 0)))

            gif_path = out_dir / f"{tag}_{pw}.gif"
            if frames:
                frames[0].save(
                    gif_path, save_all=True, append_images=frames[1:],
                    duration=1000 // args.fps, loop=0,
                )
            print(f"{gif_path.name}: success={bool(info.get('success'))} "
                  f"press={info.get('press_count')} steps={steps} "
                  f"frames={len(frames)}", flush=True)
    finally:
        env.close()
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
