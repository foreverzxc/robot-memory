"""Roll out a RoboMME decoder-TTT checkpoint in the official simulator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK = 12
ACTION_DIM = 7
STATE_DIM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("PickXtimes", "SwingXtimes"), required=True)
    parser.add_argument("--dataset", choices=("test", "val", "train"), default="test")
    parser.add_argument("--episode", type=int, default=0, help="Episode index; use -1 for all episodes.")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "robomme_rollouts")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--turbo-root", type=Path, required=True)
    parser.add_argument("--base-ckpt", type=Path, default=None)
    parser.add_argument("--dinov3-path", type=Path, default=None)
    parser.add_argument("--bert-path", type=Path, default=None)
    parser.add_argument("--no-online-updates", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def latest(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def latest_obs(obs: dict, key: str) -> np.ndarray:
    values = obs[key]
    if len(values) == 0:
        raise ValueError(f"RoboMME observation field is empty: {key}")
    return latest(values[-1])


def frame(obs: dict) -> np.ndarray:
    front = latest_obs(obs, "front_rgb_list").astype(np.uint8)
    wrist = latest_obs(obs, "wrist_rgb_list").astype(np.uint8)
    return np.hstack([front, wrist])


def reset_fast_weights(model):
    return {
        layer: {
            name: value.detach().clone().requires_grad_(True)
            for name, value in block.fast.named_parameters()
        }
        for layer, block in enumerate(model.ttt_layers)
    }


def detach_fast_weights(params):
    return {
        layer: {
            name: value.detach().clone().requires_grad_(True)
            for name, value in layer_params.items()
        }
        for layer, layer_params in params.items()
    }


def state_from_obs(obs: dict) -> np.ndarray:
    eef = latest_obs(obs, "eef_state_list").reshape(-1).astype(np.float32)
    gripper = latest_obs(obs, "gripper_state_list").reshape(-1).astype(np.float32)
    state = np.concatenate([eef, gripper])
    if state.shape != (STATE_DIM,):
        raise ValueError(f"expected an 8D RoboMME state, got {state.shape}")
    return state


def prepare_images(obs: dict, processor, device: torch.device) -> torch.Tensor:
    from PIL import Image

    front = Image.fromarray(latest_obs(obs, "front_rgb_list").astype(np.uint8))
    wrist = Image.fromarray(latest_obs(obs, "wrist_rgb_list").astype(np.uint8))
    front_tensor = processor(images=front, return_tensors="pt")["pixel_values"].squeeze(0)
    wrist_tensor = processor(images=wrist, return_tensors="pt")["pixel_values"].squeeze(0)
    return torch.stack([front_tensor, wrist_tensor], dim=0).unsqueeze(0).to(device)


def denormalize_actions(prediction: torch.Tensor, stats: dict) -> np.ndarray:
    values = prediction.detach().float().cpu().numpy()
    action_min = np.asarray(stats["action_min"], dtype=np.float32)
    action_max = np.asarray(stats["action_max"], dtype=np.float32)
    actions = (values + 1.0) * 0.5 * (action_max - action_min) + action_min
    actions[..., :6] = np.clip(actions[..., :6], action_min[:6], action_max[:6])
    actions[..., 6] = np.where(actions[..., 6] >= 0.0, 1.0, -1.0)
    return actions.astype(np.float32)


def add_video_demo_border(value: np.ndarray) -> np.ndarray:
    output = value.copy()
    thickness = min(10, output.shape[0] // 8, output.shape[1] // 8)
    output[:thickness] = (255, 0, 0)
    output[-thickness:] = (255, 0, 0)
    output[:, :thickness] = (255, 0, 0)
    output[:, -thickness:] = (255, 0, 0)
    return output


def initial_frames(obs: dict) -> list[np.ndarray]:
    front_values = obs["front_rgb_list"]
    wrist_values = obs["wrist_rgb_list"]
    if len(front_values) != len(wrist_values):
        raise ValueError("front and wrist conditioning videos have different lengths")
    frames = []
    for index, (front, wrist) in enumerate(zip(front_values, wrist_values)):
        front_np = latest(front).astype(np.uint8)
        wrist_np = latest(wrist).astype(np.uint8)
        combined = np.hstack([front_np, wrist_np])
        if index < len(front_values) - 1:
            combined = add_video_demo_border(combined)
        frames.append(combined)
    return frames


def load_stats(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("state_mean", "state_std", "action_min", "action_max")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"stats file is missing fields: {missing}")
    return data


def load_models(args: argparse.Namespace):
    turbo_root = args.turbo_root.resolve()
    os.chdir(turbo_root)
    sys.path.insert(0, str(turbo_root))
    sys.path.insert(0, str(turbo_root / "TurboVLA"))

    from decoder_ttt import DecoderWithTTT
    from eval_ttt_decoder import make_processor
    from train_ttt_decoder import build_base_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = build_base_model(
        str(args.base_ckpt), str(args.dinov3_path), str(args.bert_path)
    )
    base.eval()
    base.to(device)

    decoder_ttt = DecoderWithTTT(base.action_head.decoder).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = decoder_ttt.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    decoder_ttt.eval()
    for parameter in decoder_ttt.parameters():
        parameter.requires_grad_(False)
    processor = make_processor(str(args.dinov3_path))
    return base, decoder_ttt, processor, device


def run_episode(
    base,
    decoder_ttt,
    processor,
    device,
    env,
    stats: dict,
    max_steps: int,
    online_updates: bool,
):
    obs, info = env.reset()
    instruction = str(info["task_goal"][0])
    frames = initial_frames(obs)
    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    text_cache = base.encode_text([instruction], device=device)
    fast_weights = reset_fast_weights(decoder_ttt)
    action_queue = deque()
    steps = 0
    last_info = info
    done = False

    while steps < max_steps:
        if not action_queue:
            pixel_values = prepare_images(obs, processor, device)
            state = torch.tensor(state_from_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
            state = (state - state_mean) / (state_std + 1e-6)
            with torch.no_grad():
                condition = base.encode_condition(
                    [instruction], {"dinov3": pixel_values}, cached_text=text_cache
                ).float()
                state_tokens = base.action_head.state_projection(state)
                memory = torch.cat([condition, state_tokens], dim=1)
            with torch.enable_grad():
                prediction, next_fast_weights = decoder_ttt(
                    memory,
                    fast_weights,
                    create_graph=False,
                    update=online_updates,
                )
            if online_updates:
                fast_weights = detach_fast_weights(next_fast_weights)
            action_queue.extend(denormalize_actions(prediction[0], stats))

        action = action_queue.popleft()
        obs, _, terminated, truncated, last_info = env.step(action)
        frames.append(frame(obs))
        steps += 1
        if terminated or truncated:
            done = True
            break

    status = str(last_info.get("status", "timeout")) if done else "timeout"
    return status, frames, steps, last_info, instruction


def main() -> int:
    args = parse_args()
    args.turbo_root = args.turbo_root.expanduser().resolve()
    args.checkpoint = (
        args.checkpoint
        or REPO_ROOT / "weights" / f"decoder_ttt_{args.task.lower()}_split80.pth"
    ).expanduser().resolve()
    args.stats = (
        args.stats or REPO_ROOT / "weights" / f"{args.task.lower()}_stats.json"
    ).expanduser().resolve()
    args.base_ckpt = (
        args.base_ckpt or args.turbo_root / "weights" / "libero" / "spatial.pth"
    ).expanduser().resolve()
    args.dinov3_path = (
        args.dinov3_path or args.turbo_root / "weights" / "dinov3"
    ).expanduser().resolve()
    args.bert_path = (
        args.bert_path or args.turbo_root / "weights" / "bert"
    ).expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    for path in (args.checkpoint, args.stats, args.base_ckpt, args.dinov3_path, args.bert_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.num_episodes is not None and args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")

    stats = load_stats(args.stats)
    base, decoder_ttt, processor, device = load_models(args)
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id=args.task,
        dataset=args.dataset,
        action_space="ee_pose",
        gui_render=False,
        max_steps=args.max_steps,
    )
    total_episodes = builder.get_episode_num()
    if args.episode == -1:
        episode_ids = list(range(total_episodes))
    elif args.episode >= 0:
        episode_ids = [args.episode]
    else:
        raise ValueError("--episode must be >= 0 or -1")
    if args.num_episodes is not None:
        episode_ids = episode_ids[: args.num_episodes]
    if not episode_ids or max(episode_ids) >= total_episodes:
        raise ValueError(f"episode range exceeds {total_episodes} episodes in {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for episode_idx in episode_ids:
        env = builder.make_env_for_episode(episode_idx)
        try:
            status, frames, steps, final_info, instruction = run_episode(
                base,
                decoder_ttt,
                processor,
                device,
                env,
                stats,
                args.max_steps,
                online_updates=not args.no_online_updates,
            )
            video_path = args.output_dir / f"{args.task}_ep{episode_idx}_{status}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            result = {
                "task": args.task,
                "dataset": args.dataset,
                "episode": episode_idx,
                "status": status,
                "success": status == "success",
                "steps": steps,
                "task_goal": instruction,
                "video": str(video_path),
                "info": {
                    key: str(value)
                    for key, value in final_info.items()
                    if key in ("status", "error_message", "task_goal")
                },
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            env.close()

    summary = {
        "task": args.task,
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "online_updates": not args.no_online_updates,
        "episodes": len(results),
        "successes": sum(item["success"] for item in results),
        "success_rate": sum(item["success"] for item in results) / len(results),
        "status_counts": {
            status: sum(item["status"] == status for item in results)
            for status in sorted({item["status"] for item in results})
        },
        "results": results,
    }
    result_path = args.output_dir / f"{args.task}_summary.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved summary: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
