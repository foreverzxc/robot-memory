"""Offline evaluation of a trained checkpoint on the button simulator.

Loads a checkpoint saved by train_button.py (best.pt / snapshot.pt / model_*.pt)
and rolls out the policy in ButtonEnv on the given passwords.

Usage:
    $env:MUJOCO_GL = "glfw"
    E:\\WM\\turbovla\\.venv\\Scripts\\python.exe scripts\\eval_button_checkpoint.py ^
        --ckpt runs/b1_4plus4_cls/best.pt --args runs/b1_4plus4_cls/args.json ^
        --passwords 122221,211222,221221,212112 --repeats 3
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

import numpy as np
import torch
import einops

from button_task.button_dataset import ButtonH5Dataset
from button_task.env_wrapper import make_button_patch_env
from models.encoder.dino import DinoV2Encoder

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="checkpoint file")
    p.add_argument("--args", dest="args_file", default=None,
                   help="args.json from the training run (needed for model config)")
    p.add_argument("--passwords", required=True,
                   help="comma-separated passwords to evaluate")
    p.add_argument("--repeats", type=int, default=3,
                   help="rollouts per password (different env seeds)")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--image-size", type=int, default=None,
                   help="override image size (default: from args.json)")
    p.add_argument("--encoder-mode", default=None,
                   choices=["patch", "cls", "avg_pool"],
                   help="override encoder mode (default: from args.json)")
    p.add_argument("--action-window", type=int, default=None,
                   help="override action window (default: from args.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-password", default=None,
                   help="diagnostic: feed THIS password's tokens to the model "
                        "while the env runs its own password (checks whether "
                        "the model actually reads the password tokens)")
    p.add_argument("--ttt", action="store_true",
                   help="checkpoint has a TTT module; carry fast weights per "
                        "episode during rollout (train/infer-consistent updates)")
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

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    run_args = {}
    if args.args_file and Path(args.args_file).is_file():
        with open(args.args_file, encoding="utf-8") as f:
            run_args = json.load(f)

    image_size = args.image_size or run_args.get("image_size", 224)
    encoder_mode = args.encoder_mode or run_args.get("encoder_mode", "cls")
    action_window = args.action_window or run_args.get("action_window", 12)
    if args.action_window is not None and run_args.get("action_window", 12) != args.action_window:
        raise SystemExit(
            "--action-window override changes the model architecture and would "
            "load an incompatible checkpoint; remove the override (action window "
            f"must be {run_args.get('action_window', 12)})"
        )

    encoder = build_encoder(encoder_mode, image_size, device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # load model (checkpoint holds state_dict under "model_state")
    snap = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_state = snap.get("model_state", snap.get("model")) if isinstance(snap, dict) else None
    if model_state is None:
        model_state = snap  # raw state_dict
    if hasattr(model_state, "state_dict"):
        model_state = model_state.state_dict()

    from models.vq_behavior_transformer.bet import BehaviorTransformer

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
        act_window_size=action_window,
        cond_len=6,
        cond_mode=run_args.get("cond_mode", "seq"),
        cond_num_symbols=2,
        gpt_block_size=run_args.get("gpt_block_size", 1),
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
    print(f"loaded {ckpt} (epoch {snap.get('epoch') if isinstance(snap, dict) else '?'}, "
          f"ttt_enabled={ttt_enabled})")

    passwords = [p for p in args.passwords.split(",") if p]
    eval_passwords = []
    for p in passwords:
        eval_passwords.extend([p] * args.repeats)

    env = make_button_patch_env(
        seed=args.seed,
        password=eval_passwords[0],
        passwords=eval_passwords,
        cameras=("agentview", "robot0_eye_in_hand"),
        image_size=(image_size, image_size),
        horizon=1000,
        max_steps=args.max_steps,
    )

    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    results = {}
    try:
        for k in range(len(eval_passwords)):
            obs = env.reset(goal_idx=k)
            pw = env.password
            token_pw = args.token_password or pw
            # Full password is constant for the whole rollout. Dynamic
            # remaining-suffix conditioning is forbidden (progress leakage).
            pw_full = torch.as_tensor(
                ButtonH5Dataset.encode_password(token_pw, 6)[0],
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            action_list = []
            fw = None  # fresh fast weights per episode (TTT eval protocol)
            done, steps = False, 0
            info = {}
            while not done and steps < env.max_steps:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad(), amp_ctx:
                    emb = encoder(obs_t)
                emb = einops.rearrange(emb, "B V P E -> B (V P) E").unsqueeze(1)
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
            r = {
                "success": bool(info.get("success", False)),
                "failed": bool(info.get("failed", False)),
                "press_count": int(info.get("press_count", 0)),
                "steps": steps,
            }
            results.setdefault(pw, []).append(r)
            tok = f" (tokens={token_pw})" if args.token_password else ""
            print(f"pw={pw:>6}{tok} success={int(r['success'])} failed={int(r['failed'])} "
                  f"press={r['press_count']} steps={r['steps']}", flush=True)
    finally:
        env.close()

    print("\nsummary:")
    for pw, rs in results.items():
        sr = sum(x["success"] for x in rs) / len(rs)
        pr = [x["press_count"] for x in rs]
        print(f"  {pw}: success {sr:.2f} ({sum(x['success'] for x in rs)}/{len(rs)}) "
              f"press {min(pr)}-{max(pr)} steps {min(x['steps'] for x in rs)}-{max(x['steps'] for x in rs)}")
    overall = sum(x["success"] for rs in results.values() for x in rs) / (len(eval_passwords))
    print(f"overall success: {overall:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
