"""Offline held-out action evaluation for RoboMME decoder-TTT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "patch_policy_ttt" / "runs" / "robomme_cache"
DEFAULT_CKPT = REPO_ROOT / "patch_policy_ttt" / "runs" / "robomme_ttt"
DEFAULT_TURBO = Path(r"E:\WM\turbovla")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["PickXtimes", "SwingXtimes"], required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--turbo-root", type=Path, default=DEFAULT_TURBO)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_TURBO / "weights" / "libero" / "spatial.pth")
    parser.add_argument("--dinov3-path", type=Path, default=DEFAULT_TURBO / "weights" / "dinov3")
    parser.add_argument("--bert-path", type=Path, default=DEFAULT_TURBO / "weights" / "bert")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def detach_fast_weights(params):
    return {
        layer: {
            name: value.detach().clone().requires_grad_(True)
            for name, value in layer_params.items()
        }
        for layer, layer_params in params.items()
    }


def fresh_fast_weights(model):
    return {
        layer: {
            name: value.detach().clone().requires_grad_(True)
            for name, value in block.fast.named_parameters()
        }
        for layer, block in enumerate(model.ttt_layers)
    }


def evaluate_episode(base, ttt_model, path: Path, device):
    with np.load(path, allow_pickle=False) as data:
        cond = torch.from_numpy(data["cond"]).float().to(device)
        state = torch.from_numpy(data["state"]).float().to(device)
        target = torch.from_numpy(data["target_chunk"]).float().to(device)

    ttt_fast = fresh_fast_weights(ttt_model)
    ttt_errors = []
    base_errors = []
    ttt_first_errors = []
    base_first_errors = []
    ttt_gripper = []
    base_gripper = []

    for index in range(cond.shape[0]):
        with torch.no_grad():
            state_tokens = base.action_head.state_projection(state[index : index + 1])
        memory = torch.cat([cond[index : index + 1], state_tokens], dim=1)

        # TTT inner updates require autograd even when create_graph=False.
        with torch.enable_grad():
            ttt_pred, ttt_fast = ttt_model(
                memory, ttt_fast, create_graph=False
            )
        ttt_fast = detach_fast_weights(ttt_fast)
        with torch.no_grad():
            base_pred = ttt_model.decoder(memory)
            ttt_error = ttt_pred.detach() - target[index : index + 1]
            base_error = base_pred - target[index : index + 1]
            ttt_errors.append(ttt_error.cpu())
            base_errors.append(base_error.cpu())
            ttt_first_errors.append(ttt_error[:, 0].cpu())
            base_first_errors.append(base_error[:, 0].cpu())
            target_gripper = target[index : index + 1, :, 6] >= 0
            ttt_gripper.append(((ttt_pred[:, :, 6] >= 0) == target_gripper).cpu())
            base_gripper.append(((base_pred[:, :, 6] >= 0) == target_gripper).cpu())

    ttt_error = torch.cat(ttt_errors, dim=0)
    base_error = torch.cat(base_errors, dim=0)
    ttt_first = torch.cat(ttt_first_errors, dim=0)
    base_first = torch.cat(base_first_errors, dim=0)
    return {
        "ttt": {
            "mae": float(ttt_error.abs().mean()),
            "rmse": float(ttt_error.square().mean().sqrt()),
            "first_action_mae": float(ttt_first.abs().mean()),
            "gripper_direction_accuracy": float(torch.cat(ttt_gripper).float().mean()),
        },
        "no_ttt": {
            "mae": float(base_error.abs().mean()),
            "rmse": float(base_error.square().mean().sqrt()),
            "first_action_mae": float(base_first.abs().mean()),
            "gripper_direction_accuracy": float(torch.cat(base_gripper).float().mean()),
        },
        "chunks": int(cond.shape[0]),
    }


def aggregate(results):
    total_chunks = sum(item["chunks"] for item in results)
    output = {"episodes": len(results), "chunks": total_chunks}
    for mode in ("ttt", "no_ttt"):
        output[mode] = {
            key: float(np.average([item[mode][key] for item in results], weights=[item["chunks"] for item in results]))
            for key in results[0][mode]
        }
    return output


def main() -> int:
    args = parse_args()
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be between 0 and 1")
    args.cache_dir = args.cache_dir.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.base_ckpt = args.base_ckpt.resolve()
    args.dinov3_path = args.dinov3_path.resolve()
    args.bert_path = args.bert_path.resolve()
    if args.out is None:
        args.out = args.ckpt_dir / f"offline_eval_{args.task}.json"
    args.out = args.out.resolve()
    if args.checkpoint is None:
        args.checkpoint = args.ckpt_dir / f"decoder_ttt_{args.task.lower()}.pth"
    args.checkpoint = args.checkpoint.resolve()

    os.chdir(args.turbo_root)
    sys.path.insert(0, str(args.turbo_root))
    sys.path.insert(0, str(args.turbo_root / "TurboVLA"))
    from decoder_ttt import DecoderWithTTT
    from train_ttt_decoder import build_base_model

    cache_dir = args.cache_dir / args.task
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(manifest))
    holdout_count = max(1, int(round(len(manifest) * args.holdout_fraction)))
    holdout_ids = set(int(index) for index in order[:holdout_count])
    split_ids = {
        "holdout": sorted(holdout_ids),
        "train_seen": sorted(set(range(len(manifest))) - holdout_ids),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = build_base_model(
        str(args.base_ckpt), str(args.dinov3_path), str(args.bert_path)
    )
    base.eval()
    base.to(device)
    ttt_model = DecoderWithTTT(base.action_head.decoder).to(device)
    checkpoint = args.checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ttt_model.load_state_dict(state["model_state_dict"], strict=True)
    ttt_model.eval()
    for parameter in ttt_model.parameters():
        parameter.requires_grad_(False)

    evaluation = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "cache": str(cache_dir),
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "split": {key: len(value) for key, value in split_ids.items()},
        "metric_space": "per-task normalized eef_action; gripper uses sign",
        "note": "offline behavior-cloning metrics; this is not simulator rollout success",
    }
    for split_name, indices in split_ids.items():
        results = []
        for index in indices:
            path = cache_dir / manifest[index]["file"]
            results.append(evaluate_episode(base, ttt_model, path, device))
        evaluation[split_name] = aggregate(results)
        print(split_name, json.dumps(evaluation[split_name], sort_keys=True), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"saved: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
