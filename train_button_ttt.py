"""Stage B2/B3: VQ-BeT + TTT training on the button task.

Chunked sequence training with a test-time-training layer attached to the GPT
observation token stream (see button_task/ttt_layer.py). Groups:
  --ttt none     plain VQ-BeT sequence baseline (no TTT module)
  --ttt frozen   TTT base params frozen; fast-weight inner updates still run
  --ttt online   TTT params trained jointly with the policy

Optional progress supervision (gradient reaches only the TTT module):
  --prog-weight > 0 together with --labels <npz from label_button_demos.py>

Optional curriculum (labels are used ONLY to schedule the loss; the model
always receives the full, constant password for each episode):
  --curriculum-epochs N   first N epochs only train windows/chunks whose
                     REMAINING password length is <= ramp (1..6), then all;
                     short contexts first (needs --labels)

TTT fast weights are reset per batch (or carried per episode with
--carry-windows) in training and per episode at eval; the same forward path
runs in both (train/inference consistency). By default --per-step-attn keeps
GPT attention inside each timestep so a T>1 window is equivalent to T=1
inference, with TTT as the only cross-time channel.

Usage:
    $env:MUJOCO_GL = "glfw"
    python train_button_ttt.py --h5 <demos.h5> --out runs/b3_ttt \
        --load runs/b1_4plus4_cls/best.pt --args runs/b1_4plus4_cls/args.json \
        --pw-train ... --pw-eval ... --ttt online --t-window 4 --prog-weight 1.0 \
        --labels runs/b1_4plus4_labels.npz
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
from button_task.ttt_layer import TTTSequence
from models.encoder.dino import DinoV2Encoder
from models.vq_behavior_transformer.bet import BehaviorTransformer

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--split", default=str(ROOT / "button_task" / "password_split.json"))
    p.add_argument("--pw-train", default=None)
    p.add_argument("--pw-eval", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--load", default=None, help="B1 checkpoint (snapshot.pt/best.pt)")
    p.add_argument("--args", dest="args_file", default=None,
                   help="args.json of the --load run (model config source)")

    # data / sequence
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--encoder-mode", default="cls", choices=["patch", "cls", "avg_pool"])
    p.add_argument("--action-window", type=int, default=12)
    p.add_argument("--t-window", type=int, default=4, help="sequence length T")
    p.add_argument("--carry-windows", action="store_true",
                   help="per-episode chunked training: iterate each episode in "
                        "non-overlapping T-chunks and CARRY the fast weights "
                        "across chunks (matches the inference accumulation "
                        "protocol exactly); TBPTT detaches between chunks")
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--no-embed-cache", action="store_true",
                   help="disable DINO embedding disk cache (raw images are "
                        "re-read and re-embedded every epoch)")
    p.add_argument("--max-train-episodes", type=int, default=0)
    p.add_argument("--max-holdout-episodes", type=int, default=0)

    # TTT
    p.add_argument("--ttt", default="none", choices=["none", "frozen", "online"])
    p.add_argument("--ttt-fast-hidden", type=int, default=256)
    p.add_argument("--ttt-base-lr", type=float, default=0.1)
    p.add_argument("--tbptt", type=int, default=0, help="detach fast weights every K steps (0=off)")
    p.add_argument("--damp", type=float, default=0.0)
    p.add_argument("--prog-weight", type=float, default=0.0,
                   help="progress supervision loss weight (TTT gradient only)")
    p.add_argument("--labels", default=None, help="npz from label_button_demos.py")
    p.add_argument("--curriculum-epochs", type=int, default=0,
                   help="first N epochs only train windows/chunks whose "
                        "remaining password length <= ramp (1..6); needs --labels")

    # model (used when not loading a checkpoint)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--vqvae-n-embed", type=int, default=16)
    p.add_argument("--vqvae-groups", type=int, default=2)
    p.add_argument("--vqvae-latent-dim", type=int, default=512)
    p.add_argument("--vqvae-iters", type=int, default=300)
    p.add_argument("--vqvae-max-samples", type=int, default=16384)
    p.add_argument("--offset-loss-multiplier", type=float, default=10.0)
    p.add_argument("--act-scale", type=float, default=1.0)
    p.add_argument("--gpt-block-size", type=int, default=16)
    p.add_argument("--cond-mode", default="seq", choices=["seq", "sum", "lookup"])
    p.add_argument("--per-step-attn", action=argparse.BooleanOptionalAction, default=True,
                   help="restrict GPT attention to within each timestep (each "
                        "step sees only its own image; TTT is the only "
                        "cross-time path). Disable with --no-per-step-attn.")

    # optimization
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)

    # eval / logging
    p.add_argument("--eval-freq", type=int, default=10)
    p.add_argument("--eval-train-passwords", type=int, default=0)
    p.add_argument("--max-env-steps", type=int, default=600)
    p.add_argument("--no-env-eval", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=0)
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
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = encoder(obs)
    return einops.rearrange(emb, "B T V P E -> B T (V P) E")


@torch.no_grad()
def embed_shard_frames(encoder, shard_obs, device, image_size, chunk=256):
    emb_list = []
    S = shard_obs.shape[0]
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    for i in range(0, S, chunk):
        obs = shard_obs[i : i + chunk].to(device).float().div_(255.0)
        B, V, C, H, W = obs.shape
        if obs.shape[-2:] != image_size:
            obs = obs.reshape(B * V, C, H, W)
            obs = F.interpolate(obs, size=image_size, mode="bilinear", align_corners=False)
            obs = obs.reshape(B, V, C, *image_size)
        obs = obs.unsqueeze(1)
        with amp_ctx:
            emb = encoder(obs)
        emb_list.append(einops.rearrange(emb, "B T V P E -> B T (V P) E").bfloat16())
        del obs
    return torch.cat(emb_list, dim=0)


def get_shard_embeddings(shard, encoder, device, image_size, cache_dir,
                         dataset=None, use_cache=True):
    """Return per-shard DINO embeddings, loading/saving per-episode cache.

    When ``use_cache`` is false this is identical to embed_shard_frames.
    ``dataset`` must provide load_episode()/name_to_idx on cache misses.
    """
    if not use_cache:
        if shard.obs is None:
            raise ValueError("embed cache disabled but shard has no obs")
        return embed_shard_frames(encoder, shard.obs, device, image_size)

    if shard.obs is not None:
        # caller already loaded raw obs; embed directly (no per-episode split)
        return embed_shard_frames(encoder, shard.obs, device, image_size)

    if cache_dir is None:
        raise ValueError("light shards require a cache directory")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    pos = 0
    for name, L in zip(shard.names, shard.episode_lengths):
        cache_path = cache_dir / f"{name}_img{image_size}.pt"
        if cache_path.is_file():
            ep_emb = torch.load(cache_path, map_location="cpu", weights_only=False)
            if ep_emb.shape[0] != L:
                cache_path.unlink(missing_ok=True)
            else:
                parts.append(ep_emb)
                continue
        # cache miss: load this episode's raw images once and embed them
        idx = dataset.name_to_idx[name]
        ep_shard = dataset.load_episode(idx)
        ep_emb = embed_shard_frames(
            encoder, ep_shard.obs, device, image_size
        ).detach().cpu()
        torch.save(ep_emb, cache_path)
        parts.append(ep_emb)
    return torch.cat(parts, dim=0).to(device)


class EvalEnv:
    def __init__(self, passwords, image_size, max_steps, seed):
        self.passwords = list(passwords)
        self.env = make_button_patch_env(
            seed=seed, password=self.passwords[0], passwords=self.passwords,
            cameras=("agentview", "robot0_eye_in_hand"),
            image_size=(image_size, image_size), horizon=1000, max_steps=max_steps,
        )

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def eval_in_sim(model, encoder, device, passwords, image_size, max_steps, seed,
                action_window, tag="", ttt_enabled=False):
    model.eval()
    eval_env = EvalEnv(passwords, image_size, max_steps, seed)
    results = []
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    try:
        for k in range(len(passwords)):
            obs = eval_env.env.reset(goal_idx=k)
            pw = eval_env.env.password
            # The policy ALWAYS receives the full password. Dynamic
            # remaining-suffix conditioning is forbidden (ground-truth
            # progress leakage); press_count is only used for reporting.
            pw_full = torch.as_tensor(
                ButtonH5Dataset.encode_password(pw, 6)[0],
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            action_list = []
            fw = None  # fresh fast weights per episode
            done, steps, info = False, 0, {}
            while not done and steps < eval_env.env.max_steps:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_emb = embed_obs(encoder, obs_t.unsqueeze(0), device)
                with torch.no_grad(), amp_ctx:
                    pred, _, _, ttt_out = model(
                        obs_emb, None, None, password_idx=pw_full,
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
                obs, reward, done, info = eval_env.env.step(curr.astype(np.float32))
                steps += 1
            results.append({
                "password": pw,
                "success": bool(info.get("success", False)),
                "failed": bool(info.get("failed", False)),
                "press_count": int(info.get("press_count", 0)),
                "steps": steps,
            })
            print(f"  [{tag}] pw={pw:>6} success={int(results[-1]['success'])} "
                  f"failed={int(results[-1]['failed'])} press={results[-1]['press_count']} "
                  f"steps={steps}", flush=True)
    finally:
        eval_env.close()
    model.train()
    n = len(results)
    return {
        f"success_rate{tag}": sum(r["success"] for r in results) / n,
        f"failed_rate{tag}": sum(r["failed"] for r in results) / n,
        f"mean_press{tag}": float(np.mean([r["press_count"] for r in results])),
        f"mean_steps{tag}": float(np.mean([r["steps"] for r in results])),
        f"per_pw{tag}": {
            r["password"]: {"success": r["success"], "press_count": r["press_count"]}
            for r in results
        },
    }


def build_encoder(args, device):
    if args.encoder_mode == "patch":
        return DinoV2Encoder(
            name="dinov2_vits14", feature_key="x_norm_patchtokens",
            output_dim=384, n_patches=(args.image_size // 14) ** 2,
        ).to(device)
    if args.encoder_mode == "cls":
        return DinoV2Encoder(
            name="dinov2_vits14", feature_key="x_norm_clstoken",
            output_dim=384, n_patches=1,
        ).to(device)
    return DinoV2Encoder(
        name="dinov2_vits14", feature_key="x_norm_patchtokens",
        output_dim=384, postprocess="avg_pool", n_patches=1,
    ).to(device)


def build_model(args, device, load_args=None):
    ca = load_args or vars(args)
    n_patches = 1 if args.encoder_mode in ("cls", "avg_pool") else (args.image_size // 14) ** 2
    n_gpt_layers = int(ca.get("n_layer", args.n_layer))
    ttt_module = None
    if args.ttt in ("frozen", "online"):
        # per-layer RoboTTT: one TTT module after each GPT attention block.
        # Only the last TTT module owns the three auxiliary heads:
        # progress (normalized press count), discrete count, next key.
        ttt_module = [
            TTTSequence(
                dim=ca.get("n_embd", args.n_embd),
                fast_hidden=args.ttt_fast_hidden,
                base_lr=args.ttt_base_lr,
                num_layers=1,
                tbptt_step_size=args.tbptt or None,
                damp=args.damp,
                progress_head=(args.prog_weight > 0 and i == n_gpt_layers - 1),
                count_head=(args.prog_weight > 0 and i == n_gpt_layers - 1),
                next_key_head=(args.prog_weight > 0 and i == n_gpt_layers - 1),
                count_classes=7,
            )
            for i in range(n_gpt_layers)
        ]
    model = BehaviorTransformer(
        obs_dim=384, act_dim=7, goal_dim=0, views=2,
        vqvae_latent_dim=ca.get("vqvae_latent_dim", args.vqvae_latent_dim),
        vqvae_n_embed=ca.get("vqvae_n_embed", args.vqvae_n_embed),
        vqvae_groups=ca.get("vqvae_groups", args.vqvae_groups),
        vqvae_fit_steps=None,
        vqvae_iters=ca.get("vqvae_iters", args.vqvae_iters),
        n_patches=n_patches,
        n_layer=n_gpt_layers,
        n_head=ca.get("n_head", args.n_head),
        n_embd=ca.get("n_embd", args.n_embd),
        dropout=0.0,
        vqvae_encoder_loss_multiplier=1.0,
        vqvae_batch_size=1024,
        act_scale=ca.get("act_scale", args.act_scale),
        offset_loss_multiplier=ca.get("offset_loss_multiplier", args.offset_loss_multiplier),
        obs_window_size=1,
        act_window_size=args.action_window,
        cond_len=6,
        cond_mode=args.cond_mode,
        cond_num_symbols=2,
        gpt_block_size=args.gpt_block_size,
        vqvae_max_samples=args.vqvae_max_samples,
        ttt_module=ttt_module,
        per_timestep_attn=args.per_step_attn,
    ).to(device)
    return model


def load_label_maps(labels_path):
    """Return (counts_map, next_keys_map) from a labels npz."""
    if not labels_path or not Path(labels_path).is_file():
        return None, None
    z = np.load(labels_path, allow_pickle=False)
    counts, next_keys = {}, {}
    for key in z.files:
        if key.endswith("/count"):
            counts[key[: -len("/count")]] = torch.as_tensor(z[key], dtype=torch.long)[:-1]
        elif key.endswith("/next_key"):
            next_keys[key[: -len("/next_key")]] = torch.as_tensor(z[key], dtype=torch.long)[:-1]
    print(f"labels: {len(counts)} episodes with per-frame counts, "
          f"{len(next_keys)} with next-key labels")
    return counts, next_keys


def save_snapshot(path, payload):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_data = load_split(args.split)
    default_train, default_holdout = split_data
    train_pws = args.pw_train.split(",") if args.pw_train else default_train
    holdout_pws = args.pw_eval.split(",") if args.pw_eval else default_holdout

    load_args = {}
    if args.args_file and Path(args.args_file).is_file():
        with open(args.args_file, encoding="utf-8") as f:
            load_args = json.load(f)

    counts_map, next_keys_map = load_label_maps(args.labels)
    needs_labels = args.curriculum_epochs > 0
    allowed_train_episodes = None
    if needs_labels:
        if counts_map is None:
            raise ValueError(
                "--curriculum-epochs requires --labels (per-frame press counts)"
            )
        # only train on episodes whose replay labels were generated
        # successfully (mismatched/early-terminated replays are skipped)
        allowed_train_episodes = set(counts_map.keys())
        print(f"labels cover {len(allowed_train_episodes)} episodes; "
              f"training will use only those")

    cameras = ("agentview", "robot0_eye_in_hand")
    train_ds = ButtonSliceDataset(
        args.h5, allowed_passwords=train_pws, cameras=cameras,
        image_size=(args.image_size, args.image_size),
        action_window=args.action_window, shard_size=args.shard_size,
        seed=args.seed, max_episodes=args.max_train_episodes,
        allowed_episodes=allowed_train_episodes,
    )
    holdout_ds = ButtonSliceDataset(
        args.h5, allowed_passwords=holdout_pws, cameras=cameras,
        image_size=(args.image_size, args.image_size),
        action_window=args.action_window, shard_size=args.shard_size,
        seed=args.seed, max_episodes=args.max_holdout_episodes,
    )
    print(f"train episodes={train_ds.num_episodes} | holdout episodes={holdout_ds.num_episodes}")

    encoder = build_encoder(args, device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    model = build_model(args, device, load_args)
    if args.load:
        ckpt = Path(args.load)
        snap = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = snap.get("model_state", snap)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        # strict=False still raises on shape mismatches; skip those keys
        # (e.g. wpe/attention-bias buffers sized by gpt_block_size, which
        # differs from the B1 run).
        model_sd = model.state_dict()
        state = {
            k: v for k, v in state.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.vqvae_is_fit = True  # freeze the VQ codebook from B1
        skipped = len(snap.get("model_state", snap).keys()) - len(state)
        print(f"loaded {ckpt}: loaded={len(state)} skipped_shape_mismatch={skipped} "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    if args.ttt == "frozen":
        for ttt in model._gpt_model.ttt_modules():
            for p in ttt.parameters():
                p.requires_grad = False
            # the auxiliary progress head is supervision infrastructure, not
            # the TTT base mechanism: keep it trainable when progress loss is
            # used (it lives in the last per-layer TTT module)
            if args.prog_weight > 0 and ttt.progress_head is not None:
                for p in ttt.progress_head.parameters():
                    p.requires_grad = True
        print("TTT base params frozen (fast-weight updates still run)")

    # carry-windows safety: TBPTT must detach at every chunk boundary so the
    # graph of a chunk is never carried into the next chunk's backward.
    if args.carry_windows and args.ttt != "none":
        if args.tbptt <= 0 or args.t_window % args.tbptt != 0:
            args.tbptt = args.t_window
        model._gpt_model.set_ttt_tbptt_step_size(args.tbptt)
        print(f"carry-windows: TBPTT detach every {args.tbptt} steps "
              f"(chunk boundary)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable / 1e6:.2f}M")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.lr, betas=(0.9, 0.999)
    )

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    snapshot_path = out_dir / "snapshot.pt"
    best_path = out_dir / "best.pt"
    metrics_log = []
    best_success = -1.0
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)

    def log_epoch(record):
        metrics_log.append(record)
        with open(out_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def ep_counts_for(shard):
        if counts_map is None:
            return None
        return [counts_map.get(name) for name in shard.names]

    def ep_next_keys_for(shard):
        if next_keys_map is None:
            return None
        return [next_keys_map.get(name) for name in shard.names]

    def max_remaining(epoch):
        if args.curriculum_epochs <= 0 or epoch >= args.curriculum_epochs:
            return 6
        return max(1, round((epoch + 1) / args.curriculum_epochs * 6))

    def _stack_one(f):
        out = {
            "step": max(x["step"] for x in f),
            "layers": [
                {
                    k: torch.cat([x["layers"][li][k] for x in f], dim=0)
                    for k in f[0]["layers"][li]
                }
                for li in range(len(f[0]["layers"]))
            ],
        }
        if all("init_layers" in x for x in f):
            out["init_layers"] = [
                {
                    k: torch.cat([x["init_layers"][li][k] for x in f], dim=0)
                    for k in f[0]["init_layers"][li]
                }
                for li in range(len(f[0]["init_layers"]))
            ]
        return out

    def stack_fws(fws):
        """Merge per-episode fast weights into one batch state.

        Per-layer mode: each episode state is a list of per-block dicts, and
        the result is a list of stacked per-block dicts. Legacy single-TTT
        mode: states are plain dicts.
        """
        if len(fws) == 0:
            return None
        if isinstance(fws[0], list):
            return [_stack_one([f[li] for f in fws]) for li in range(len(fws[0]))]
        return _stack_one(fws)

    def _split_one(fw, B):
        out = []
        for b in range(B):
            st = {
                "step": fw["step"],
                "layers": [
                    {k: fw["layers"][li][k][b : b + 1] for k in fw["layers"][li]}
                    for li in range(len(fw["layers"]))
                ],
            }
            if "init_layers" in fw:
                st["init_layers"] = [
                    {
                        k: fw["init_layers"][li][k][b : b + 1]
                        for k in fw["init_layers"][li]
                    }
                    for li in range(len(fw["init_layers"]))
                ]
            out.append(st)
        return out

    def split_fws(fw, B):
        """Split a batch fast-weight state back into per-episode states."""
        if isinstance(fw, list):
            per_block = [_split_one(block, B) for block in fw]
            return [list(blocks) for blocks in zip(*per_block)]
        return _split_one(fw, B)

    def train_step(obs_emb, act, pw, counts_t, fw, debug_batch, valid_mask=None,
                   next_keys_t=None):
        """One forward + auxiliary-head losses + backward.

        Auxiliary heads live in the last TTT module and receive gradient only
        through the shadow TTT pass (never the policy backbone):
          progress head  -> MSE(progress, press_count/6)
          count head     -> cross-entropy over 0..6 press counts
          next-key head  -> cross-entropy over left/right next key
        """
        with amp_ctx:
            pred, loss, loss_dict, ttt_out = model(
                obs_emb, None, act, password_idx=pw,
                prev_fast_weights=fw,
                return_progress=args.prog_weight > 0,
                debug=debug_batch,
                loss_mask=valid_mask,
            )
        next_fw = ttt_out["next_fast_weights"] if ttt_out is not None else None
        pl, pn = 0.0, 0
        if args.prog_weight > 0 and ttt_out is not None and counts_t is not None:
            stats = ttt_out["stats"]
            counts_dev = counts_t.to(device)
            mask = (counts_dev >= 0)
            if valid_mask is not None:
                mask = mask & valid_mask.to(device)

            if "progress" in stats:
                labels = counts_dev.float() / 6.0  # (B, T)
                prog = stats["progress"]  # (B, T)
                nv = mask.sum().clamp(min=1)
                prog_loss = (((prog - labels) ** 2) * mask.float()).sum() / nv
                loss = loss + args.prog_weight * prog_loss
                pl = float(prog_loss.item() * nv.item())
                pn = int(nv.item())

            if "count_logits" in stats:
                logits = stats["count_logits"]  # (B, T, 7)
                flat_logits = einops.rearrange(logits, "B T C -> (B T) C")
                flat_targets = einops.rearrange(counts_dev, "B T -> (B T)")
                flat_mask = einops.rearrange(mask, "B T -> (B T)")
                if flat_mask.any():
                    acc = (
                        flat_logits[flat_mask].argmax(-1)
                        == flat_targets[flat_mask]
                    ).float().mean()
                    loss_dict["aux_count_acc"] = float(acc.detach().cpu().item())

            if "next_key_logits" in stats:
                logits = stats["next_key_logits"]  # (B, T, 2)
                flat_logits = einops.rearrange(logits, "B T C -> (B T) C")
                flat_mask = einops.rearrange(mask, "B T -> (B T)")
                if next_keys_t is not None:
                    next_dev = next_keys_t.to(device)
                    flat_next = einops.rearrange(next_dev, "B T -> (B T)")
                    valid_next = flat_mask & (flat_next >= 0)
                    if valid_next.any():
                        acc = (
                            flat_logits[valid_next].argmax(-1)
                            == flat_next[valid_next]
                        ).float().mean()
                        loss_dict["aux_nextkey_acc"] = float(acc.detach().cpu().item())
        if loss is not None and torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item(), loss_dict, ttt_out, next_fw, pl, pn
        return None, loss_dict, ttt_out, next_fw, pl, pn

    global_step = 0
    T, W = args.t_window, args.action_window
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        loss_acc = {}
        prog_acc = 0.0
        prog_n = 0
        max_rem = max_remaining(epoch)
        t_embed = 0.0
        t_train = 0.0
        t_ctx = 0.0
        embed_cache_dir = (
            out_dir / "emb_cache" if not args.no_embed_cache else None
        )
        if embed_cache_dir is None:
            shard_iter = train_ds.iter_shards(shuffle=True)
        else:
            shard_iter = train_ds.iter_light_shards(shuffle=True)

        for shard in shard_iter:
            ta = time.perf_counter()
            shard_emb = get_shard_embeddings(
                shard, encoder, device, (args.image_size, args.image_size),
                embed_cache_dir, dataset=train_ds,
                use_cache=embed_cache_dir is not None,
            )
            t_embed += time.perf_counter() - ta
            ep_counts = ep_counts_for(shard)
            ep_next_keys = ep_next_keys_for(shard)

            if args.carry_windows:
                # per-episode chunked iteration with fast weights carried along
                # the whole episode (matches inference exactly); several
                # episodes are batched per step with per-element fast weights.
                # TBPTT boundary detach was configured on the module before
                # the epoch loop (see setup above args.json).
                fresh_fw = (
                    model._gpt_model.init_ttt_fast_weights(1)
                    if args.ttt != "none"
                    else None
                )
                ep_states = []
                pos = 0
                for ep_i, L in enumerate(shard.episode_lengths):
                    ep_states.append({
                        "pos": pos, "L": L, "ep_i": ep_i, "s": 0,
                        "fw": None,
                        "counts": ep_counts[ep_i] if ep_counts is not None else None,
                        "next_keys": ep_next_keys[ep_i] if ep_next_keys is not None else None,
                        "pw_idx": shard.pw_idx[pos],
                    })
                    pos += L
                random.shuffle(ep_states)
                queue = list(ep_states)
                while queue:
                    batch = queue[: args.batch_size]
                    queue = queue[args.batch_size :]
                    obs_idx_list, act_list, pw_list, counts_list, valid_list, next_keys_list = (
                        [], [], [], [], [], []
                    )
                    rem_list = []
                    for st in batch:
                        s, L, pos_ = st["s"], st["L"], st["pos"]
                        t_actual = min(T, L - s)
                        idx = torch.arange(s, s + t_actual).clamp(max=L - 1) + pos_
                        if t_actual < T:
                            idx = torch.cat([idx, idx[-1:].expand(T - t_actual)])
                        obs_idx_list.append(idx)
                        valid_list.append(torch.arange(T) < t_actual)
                        act_i = (torch.arange(T + W - 1) + s).clamp(max=L - 1) + pos_
                        act_list.append(shard.action[act_i])
                        c = int(st["counts"][s].item()) if st["counts"] is not None else None
                        rem_list.append(6 - c if c is not None else 0)
                        # full password ALWAYS; labels only schedule the
                        # curriculum and never modify the task conditioning
                        pw = st["pw_idx"].clone()
                        pw_list.append(pw)
                        if st["counts"] is not None:
                            counts_list.append(
                                st["counts"][torch.arange(s, s + T).clamp(max=L - 1)]
                            )
                        else:
                            counts_list.append(torch.full((T,), -1, dtype=torch.long))
                        if st["next_keys"] is not None:
                            next_keys_list.append(
                                st["next_keys"][torch.arange(s, s + T).clamp(max=L - 1)]
                            )
                        else:
                            next_keys_list.append(torch.full((T,), -1, dtype=torch.long))

                    def assemble(idxs):
                        obs_emb = shard_emb[torch.stack([obs_idx_list[i] for i in idxs])].float().squeeze(2)
                        act = torch.stack([act_list[i] for i in idxs]).to(device, non_blocking=True)
                        pw = torch.stack([pw_list[i] for i in idxs]).to(device, non_blocking=True)
                        counts_t = torch.stack([counts_list[i] for i in idxs])
                        valid_t = torch.stack([valid_list[i] for i in idxs])
                        next_keys_t = torch.stack([next_keys_list[i] for i in idxs])
                        return obs_emb, act, pw, counts_t, valid_t, next_keys_t

                    # curriculum (short->long): only chunks whose remaining
                    # length <= max_rem get gradients; others run forward-only
                    # and still carry their fast weights forward.
                    train_idx = [i for i, r in enumerate(rem_list) if r <= max_rem]
                    fwd_idx = [i for i, r in enumerate(rem_list) if r > max_rem]

                    def carry_fw(sub_batch, new_fw):
                        if new_fw is not None:
                            for st, f in zip(sub_batch, split_fws(new_fw, len(sub_batch))):
                                st["fw"] = f

                    if train_idx:
                        sub = [batch[i] for i in train_idx]
                        obs_emb, act, pw, counts_t, valid_t, next_keys_t = assemble(train_idx)
                        if args.ttt != "none":
                            fw_batch = stack_fws([
                                st["fw"] if st["fw"] is not None else fresh_fw
                                for st in sub
                            ])
                        else:
                            fw_batch = None
                        debug_batch = global_step == 0 and epoch == 0
                        ta = time.perf_counter()
                        li, ld, ttt_out, next_fw, pl, pn = train_step(
                            obs_emb, act, pw, counts_t, fw_batch, debug_batch,
                            valid_mask=valid_t, next_keys_t=next_keys_t,
                        )
                        t_train += time.perf_counter() - ta
                        if li is not None:
                            epoch_loss += li
                            for k, v in ld.items():
                                if isinstance(v, (int, float)):
                                    loss_acc[k] = loss_acc.get(k, 0.0) + v
                            epoch_steps += 1
                            global_step += 1
                            prog_acc += pl
                            prog_n += pn
                            carry_fw(sub, next_fw)
                            if debug_batch and ttt_out is not None:
                                for k, v in ttt_out["stats"].items():
                                    if k.startswith("ttt/"):
                                        print(f"  [step 0] {k}={v:.4f}", flush=True)
                    if fwd_idx:
                        ta = time.perf_counter()
                        sub = [batch[i] for i in fwd_idx]
                        obs_emb, act, pw, counts_t, valid_t, next_keys_t = assemble(fwd_idx)
                        with torch.no_grad(), amp_ctx:
                            if args.ttt != "none":
                                fw_batch = stack_fws([
                                    st["fw"] if st["fw"] is not None else fresh_fw
                                    for st in sub
                                ])
                                # context-only forward: update and carry fast
                                # weights, no outer task loss.
                                _, _, _, ttt_out = model(
                                    obs_emb, None, None, password_idx=pw,
                                    prev_fast_weights=fw_batch,
                                    loss_mask=valid_t,
                                )
                                carry_fw(sub, ttt_out["next_fast_weights"])
                        t_ctx += time.perf_counter() - ta
                    for st in batch:
                        st["s"] += T
                        if st["s"] < st["L"]:
                            queue.append(st)
                    if args.smoke_steps and global_step >= args.smoke_steps:
                        print(f"smoke: reached {global_step} steps, exiting")
                        return 0
                continue

            # fresh-per-window mode (fast weights reset every batch)
            obs_idx, act_seq, pw_win, counts, valid = train_ds.build_sequence_windows(
                shard, T, W, ep_counts=ep_counts
            )

            # remaining-length curriculum also applies here: windows whose
            # remaining suffix is still too long are skipped entirely (no
            # gradient, no forward) until later epochs.
            if (
                args.curriculum_epochs > 0
                and epoch < args.curriculum_epochs
                and counts is not None
            ):
                rem = (6 - counts[:, 0]).clamp(0, 6)
                eligible = (counts[:, 0] >= 0) & (rem <= max_rem)
                win_ids = eligible.nonzero(as_tuple=False).squeeze(1)
                if win_ids.numel() == 0:
                    continue
            else:
                win_ids = torch.arange(obs_idx.shape[0])
            n_win = win_ids.shape[0]
            order = win_ids[torch.randperm(n_win)]
            for i in range(0, n_win, args.batch_size):
                idx = order[i : i + args.batch_size]
                obs_emb = shard_emb[obs_idx[idx]].float().squeeze(2)  # (B, T, P, E)
                act = act_seq[idx].to(device, non_blocking=True)
                pw = pw_win[idx].to(device, non_blocking=True)
                counts_t = counts[idx] if counts is not None else None
                valid_t = valid[idx]
                debug_batch = global_step == 0 and epoch == 0
                li, ld, ttt_out, next_fw, pl, pn = train_step(
                    obs_emb, act, pw, counts_t, None, debug_batch,
                    valid_mask=valid_t,
                )
                if li is not None:
                    epoch_loss += li
                    for k, v in ld.items():
                        if isinstance(v, (int, float)):
                            loss_acc[k] = loss_acc.get(k, 0.0) + v
                    epoch_steps += 1
                    global_step += 1
                    prog_acc += pl
                    prog_n += pn
                    if debug_batch and ttt_out is not None:
                        for k, v in ttt_out["stats"].items():
                            if k.startswith("ttt/"):
                                print(f"  [step 0] {k}={v:.4f}", flush=True)
                if args.smoke_steps and global_step >= args.smoke_steps:
                    print(f"smoke: reached {global_step} steps, exiting")
                    return 0

        model.finish_epoch()  # VQ fit after first epoch

        dt = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, epoch_steps),
            "steps": epoch_steps,
            "seconds": round(dt, 1),
            "timing/embed_s": round(t_embed, 1),
            "timing/train_s": round(t_train, 1),
            "timing/context_s": round(t_ctx, 1),
            "timing/other_s": round(max(0.0, dt - t_embed - t_train - t_ctx), 1),
            "vqvae_fit": model.vqvae_is_fit,
            "curriculum_max_remaining": max_rem,
            **{f"train/{k}": v / max(1, epoch_steps) for k, v in loss_acc.items()},
        }
        if prog_n > 0:
            record["train/prog_mse"] = prog_acc / prog_n

        # TTT gate diagnostics on the first batch of the epoch
        if args.ttt != "none" and args.epochs > 0:
            with torch.no_grad():
                diag_iter = (
                    train_ds.iter_light_shards(shuffle=False)
                    if embed_cache_dir is not None
                    else train_ds.iter_shards(shuffle=False)
                )
                for shard in diag_iter:
                    shard_emb = get_shard_embeddings(
                        shard, encoder, device, (args.image_size, args.image_size),
                        embed_cache_dir, dataset=train_ds,
                        use_cache=embed_cache_dir is not None,
                    )
                    ep_counts = ep_counts_for(shard)
                    ep_next_keys = ep_next_keys_for(shard)
                    obs_idx, act_seq, pw_win, counts, valid = train_ds.build_sequence_windows(
                        shard, T, W, ep_counts=ep_counts
                    )
                    idx = torch.arange(min(args.batch_size, obs_idx.shape[0]))
                    obs_emb = shard_emb[obs_idx[idx]].float().squeeze(2)
                    with amp_ctx:
                        _, _, _, ttt_out = model(
                            obs_emb, None, act_seq[idx].to(device),
                            password_idx=pw_win[idx].to(device), debug=True,
                            loss_mask=valid[idx],
                        )
                    if ttt_out is not None:
                        for k, v in ttt_out["stats"].items():
                            if isinstance(v, (int, float)):
                                record[k] = v
                    break

        # sim evaluation
        if (
            not args.no_env_eval
            and model.vqvae_is_fit
            and ((epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1)
        ):
            with torch.no_grad():
                record["eval_holdout"] = eval_in_sim(
                    model, encoder, device, holdout_pws, args.image_size,
                    args.max_env_steps, args.seed, args.action_window,
                    tag="_holdout", ttt_enabled=args.ttt != "none",
                )
                if args.eval_train_passwords > 0:
                    record["eval_train"] = eval_in_sim(
                        model, encoder, device, train_pws[: args.eval_train_passwords],
                        args.image_size, args.max_env_steps, args.seed,
                        args.action_window, tag="_train", ttt_enabled=args.ttt != "none",
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
            f"epoch {epoch}: train_loss={record['train_loss']:.4f} steps={epoch_steps} {dt:.0f}s"
            + (f" holdout_succ={record['eval_holdout']['success_rate_holdout']:.3f}"
               if "eval_holdout" in record else "")
        )

    print(f"done. best holdout success = {best_success:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
