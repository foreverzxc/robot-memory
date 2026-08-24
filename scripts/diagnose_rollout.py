"""Diagnose a single policy rollout with per-step action and env-state logs."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "model_core"))

import einops
import numpy as np
import torch

from button_task.button_dataset import ButtonH5Dataset
from button_task.env_wrapper import make_button_patch_env
from models.encoder.dino import DinoV2Encoder

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--args", dest="args_file", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_args = json.loads(Path(args.args_file).read_text(encoding="utf-8"))
    image_size = run_args.get("image_size", 224)
    encoder_mode = run_args.get("encoder_mode", "cls")
    action_window = run_args.get("action_window", 12)

    encoder = DinoV2Encoder(
        name="dinov2_vits14",
        feature_key="x_norm_clstoken" if encoder_mode == "cls" else "x_norm_patchtokens",
        output_dim=384,
        n_patches=(image_size // 14) ** 2 if encoder_mode == "patch" else 1,
        postprocess=None if encoder_mode == "patch" else ("avg_pool" if encoder_mode == "avg_pool" else None),
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    snap = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = snap.get("model_state", snap.get("model"))
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    from models.vq_behavior_transformer.bet import BehaviorTransformer
    from button_task.ttt_layer import TTTSequence

    n_patches = (image_size // 14) ** 2 if encoder_mode == "patch" else 1
    ttt_enabled = run_args.get("ttt") in ("online", "frozen")
    ttt_per_layer = any(".ttt_layers." in k for k in state.keys())
    n_ttt_blocks = int(run_args.get("n_layer", 8))
    ttt_module = None
    if ttt_enabled:
        make_ttt = lambda i: TTTSequence(
            dim=run_args.get("n_embd", 512),
            fast_hidden=run_args.get("ttt_fast_hidden", 256),
            base_lr=run_args.get("ttt_base_lr", 0.01),
            num_layers=1,
            tbptt_step_size=run_args.get("tbptt") or None,
            damp=run_args.get("damp", 0.0),
            progress_head=run_args.get("prog_weight", 0.0) > 0 and i == n_ttt_blocks - 1,
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
        n_layer=n_ttt_blocks,
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
    state = {k: v for k, v in state.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(state, strict=False)
    model.vqvae_is_fit = True
    model.eval()

    pw = args.password
    env = make_button_patch_env(
        seed=args.seed, password=pw, passwords=[pw],
        cameras=("agentview", "robot0_eye_in_hand"),
        image_size=(image_size, image_size), horizon=1000, max_steps=args.max_steps,
    )
    obs = env.reset(goal_idx=0)
    base = env.env
    pw_idx = torch.as_tensor(
        ButtonH5Dataset.encode_password(pw, 6)[0], dtype=torch.long, device=device
    ).unsqueeze(0)
    fw = None
    action_list = []
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    done, steps, info = False, 0, {}
    print("step | press | last_pressed | display | eef_x | eef_y | eef_z | action[0:3]")
    while not done and steps < args.max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad(), amp_ctx:
            emb = encoder(obs_t)
        emb = einops.rearrange(emb, "B V P E -> B (V P) E").unsqueeze(1)
        with torch.no_grad(), amp_ctx:
            pred, _, _, ttt_out = model(emb, None, None, password_idx=pw_idx, prev_fast_weights=fw)
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
        site_id = base.robots[0].eef_site_id
        if isinstance(site_id, dict):
            site_id = next(iter(site_id.values()))
        eef = np.asarray(base.sim.data.site_xpos[site_id])
        print(
            f"{steps:4d} | {base._press_count} | {base._last_pressed} | {base._display_pressed} | "
            f"{eef[0]:+.3f} | {eef[1]:+.3f} | {eef[2]:+.3f} | "
            f"{curr[0]:+.3f} {curr[1]:+.3f} {curr[2]:+.3f}",
            flush=True,
        )
        obs, _, done, info = env.step(curr.astype(np.float32))
        steps += 1
    print("final:", info.get("success"), info.get("failed"), info.get("press_count"))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

