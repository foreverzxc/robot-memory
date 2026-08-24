"""Build compact RoboMME feature caches for the existing decoder-TTT trainer."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "datasets" / "robomme_data_h5"
DEFAULT_OUT = REPO_ROOT / "patch_policy_ttt" / "runs" / "robomme_cache"
DEFAULT_TURBO = Path(r"E:\WM\turbovla")
CHUNK = 12
STATE_DIM = 8
ACTION_DIM = 7


def add_turbovla_path(root: Path) -> None:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "TurboVLA"))


def decode_scalar(value) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    value = array.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def timestep_keys(episode: h5py.Group) -> list[str]:
    return sorted(
        (key for key in episode.keys() if key.startswith("timestep_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )


def read_episode(episode: h5py.Group):
    images = []
    wrists = []
    states = []
    actions = []
    for key in timestep_keys(episode):
        timestep = episode[key]
        if bool(timestep["info"]["is_video_demo"][()]):
            continue
        obs = timestep["obs"]
        action = timestep["action"]
        images.append(np.asarray(obs["front_rgb"][()], dtype=np.uint8))
        wrists.append(np.asarray(obs["wrist_rgb"][()], dtype=np.uint8))
        states.append(
            np.concatenate(
                [
                    np.asarray(obs["eef_state"][()], dtype=np.float32),
                    np.asarray(obs["gripper_state"][()], dtype=np.float32),
                ]
            )
        )
        actions.append(np.asarray(action["eef_action"][()], dtype=np.float32))

    if not actions:
        return None
    return (
        np.stack(images),
        np.stack(wrists),
        np.stack(states),
        np.stack(actions),
    )


def collect_stats(h5_path: Path) -> dict[str, np.ndarray]:
    states = []
    actions = []
    with h5py.File(h5_path, "r") as data:
        for name in data:
            episode = read_episode(data[name])
            if episode is None:
                continue
            _, _, episode_states, episode_actions = episode
            states.append(episode_states)
            actions.append(episode_actions)

    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0)
    action_min = all_actions.min(axis=0)
    action_max = all_actions.max(axis=0)
    # Keep near-constant pose dimensions numerically stable.
    action_range = np.maximum(action_max - action_min, 1e-3)
    return {
        "state_mean": all_states.mean(axis=0).astype(np.float32),
        "state_std": np.maximum(all_states.std(axis=0), 1e-4).astype(np.float32),
        "action_min": action_min.astype(np.float32),
        "action_max": action_max.astype(np.float32),
        "action_range": action_range.astype(np.float32),
        "num_frames": np.asarray([len(all_states)], dtype=np.int64),
    }


def pool_visual_tokens(condition: torch.Tensor, visual_tokens: int, pool_per_view: int) -> torch.Tensor:
    if pool_per_view <= 0:
        return condition
    if visual_tokens % 2 != 0:
        raise ValueError(f"expected two-view visual token count, got {visual_tokens}")
    per_view = visual_tokens // 2
    side = math.isqrt(per_view)
    if side * side != per_view:
        raise ValueError(f"visual tokens per view must form a square, got {per_view}")
    pool_side = math.isqrt(pool_per_view)
    if pool_side * pool_side != pool_per_view:
        raise ValueError(f"pool_per_view must be a square, got {pool_per_view}")

    batch, _, dim = condition.shape
    visual = condition[:, :visual_tokens]
    text = condition[:, visual_tokens:]
    visual = visual.view(batch, 2, side, side, dim).permute(0, 1, 4, 2, 3)
    visual = F.adaptive_avg_pool2d(
        visual.reshape(batch * 2, dim, side, side), (pool_side, pool_side)
    )
    visual = visual.reshape(batch, 2, dim, pool_side * pool_side).permute(0, 1, 3, 2)
    visual = visual.reshape(batch, 2 * pool_per_view, dim)
    return torch.cat([visual, text], dim=1)


def normalize_actions(actions: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    result = 2.0 * (actions - stats["action_min"]) / stats["action_range"] - 1.0
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["PickXtimes", "SwingXtimes"], required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--turbo-root", type=Path, default=DEFAULT_TURBO)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_TURBO / "weights" / "libero" / "spatial.pth")
    parser.add_argument("--dinov3-path", type=Path, default=DEFAULT_TURBO / "weights" / "dinov3")
    parser.add_argument("--bert-path", type=Path, default=DEFAULT_TURBO / "weights" / "bert")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool-per-view", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative")

    # Resolve user-relative paths before changing cwd for the TurboVLA loader.
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.base_ckpt = args.base_ckpt.resolve()
    args.dinov3_path = args.dinov3_path.resolve()
    args.bert_path = args.bert_path.resolve()

    # The existing TurboVLA builder resolves its text-layout path relative to
    # its project root.
    os.chdir(args.turbo_root)
    add_turbovla_path(args.turbo_root)
    from eval_ttt_decoder import make_processor
    from train_ttt_decoder import build_base_model

    h5_path = args.data_dir / f"record_dataset_{args.task}.h5"
    out_dir = args.out_dir / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = collect_stats(h5_path)
    stats_json = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in stats.items()
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats_json, indent=2), encoding="utf-8"
    )

    model = build_base_model(
        str(args.base_ckpt), str(args.dinov3_path), str(args.bert_path)
    )
    model.eval()
    processor = make_processor(str(args.dinov3_path))
    device = next(model.parameters()).device
    with torch.no_grad():
        probe = torch.zeros((1, 2, 3, 256, 256), device=device)
        visual_tokens = int(model.encode_vision(probe).shape[1])
    print(
        f"task={args.task} frames={int(stats['num_frames'][0])} "
        f"visual_tokens={visual_tokens} device={device}",
        flush=True,
    )

    manifest = []
    with h5py.File(h5_path, "r") as data:
        episode_names = sorted(
            data.keys(), key=lambda key: int(key.split("_", 1)[1])
        )
        if args.max_episodes:
            episode_names = episode_names[: args.max_episodes]

        text_cache = {}
        for index, name in enumerate(episode_names):
            output_path = out_dir / f"dec_{index:03d}.npz"
            if output_path.exists() and not args.force:
                with np.load(output_path, allow_pickle=False) as cached:
                    chunks = int(cached["cond"].shape[0])
                manifest.append({"file": output_path.name, "episode": name, "chunks": chunks})
                print(f"[{index + 1}/{len(episode_names)}] {name}: cached", flush=True)
                continue

            episode = read_episode(data[name])
            if episode is None or len(episode[3]) < CHUNK:
                print(f"[{index + 1}/{len(episode_names)}] {name}: skipped", flush=True)
                continue
            images, wrists, states, actions = episode
            starts = list(range(0, len(actions) - CHUNK + 1, CHUNK))
            instruction = decode_scalar(data[name]["setup"]["task_goal"][()])
            if not instruction:
                instruction = f"complete the {args.task} manipulation task"
            if instruction not in text_cache:
                with torch.no_grad():
                    text_cache[instruction] = model.encode_text([instruction])
            cached_text = text_cache[instruction]

            cond_list = []
            for start in range(0, len(starts), args.batch_size):
                batch_starts = starts[start : start + args.batch_size]
                pixel_batches = []
                for timestep in batch_starts:
                    front = processor(
                        images=Image.fromarray(images[timestep]), return_tensors="pt"
                    )["pixel_values"].squeeze(0)
                    wrist = processor(
                        images=Image.fromarray(wrists[timestep]), return_tensors="pt"
                    )["pixel_values"].squeeze(0)
                    pixel_batches.append(torch.stack([front, wrist]))
                pixel_values = torch.stack(pixel_batches).to(device, non_blocking=True)
                batch_text = tuple(
                    value.expand(len(batch_starts), *value.shape[1:])
                    for value in cached_text
                )
                with torch.no_grad():
                    condition = model.encode_condition(
                        [instruction] * len(batch_starts),
                        {"dinov3": pixel_values},
                        cached_text=batch_text,
                    )
                    condition = pool_visual_tokens(
                        condition, visual_tokens, args.pool_per_view
                    )
                cond_list.append(condition.float().cpu().numpy().astype(np.float16))

            normalized_states = (
                (states - stats["state_mean"]) / (stats["state_std"] + 1e-6)
            ).astype(np.float16)
            normalized_actions = normalize_actions(actions, stats)
            state_chunks = np.stack([normalized_states[s] for s in starts], axis=0)
            action_chunks = np.stack(
                [normalized_actions[s : s + CHUNK] for s in starts], axis=0
            )
            valid = np.ones((len(starts), CHUNK), dtype=np.float32)
            np.savez_compressed(
                output_path,
                cond=np.concatenate(cond_list, axis=0),
                state=state_chunks,
                target_chunk=action_chunks,
                mask=valid,
            )
            manifest.append(
                {
                    "file": output_path.name,
                    "episode": name,
                    "task": args.task,
                    "instruction": instruction,
                    "steps": int(len(actions)),
                    "chunks": len(starts),
                }
            )
            print(
                f"[{index + 1}/{len(episode_names)}] {name}: "
                f"frames={len(actions)} chunks={len(starts)}",
                flush=True,
            )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved cache={out_dir} episodes={len(manifest)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
