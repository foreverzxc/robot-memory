"""Replay demo episodes in ButtonEnv to recover per-frame progress labels.

The HDF5 files only store the FINAL press_count per episode. The aux memory
probes (next_key / press_count) need per-frame labels. Since each group stores
its collection seed, replaying the recorded actions under the same seed
reproduces the episode, and the env's per-step info gives exact labels.

Output: <h5 stem>_labels.npz with per-group arrays:
    group <name>/count     int64 (T+1,) press count before t=0 and after each action
    group <name>/next_key  int64 (T+1,) next key index (0/1), -1 when finished

Usage:
    E:\\WM\\turbovla\\.venv\\Scripts\\python.exe scripts\\label_button_demos.py ^
        --h5 E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5 ^
        --passwords 111222,111221,211221,222222,122221,211222,221221,212112
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--passwords", default=None,
                   help="comma-separated passwords to label (default: all)")
    p.add_argument("--out", default=None, help="output .npz (default: next to h5)")
    p.add_argument("--max-episodes", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    from button_task.button_env_v2 import make_button_env

    h5_path = Path(args.h5)
    out_path = Path(args.out) if args.out else h5_path.with_suffix("").with_name(
        h5_path.stem + "_labels.npz"
    )
    pw_filter = None
    if args.passwords:
        pw_filter = set(args.passwords.split(","))

    labels = {}
    with h5py.File(h5_path, "r") as f:
        names = sorted(f.keys())
        done = 0
        for name in names:
            g = f[name]
            password = str(g.attrs.get("password", ""))
            if pw_filter is not None and password not in pw_filter:
                continue
            seed = int(g.attrs.get("seed", 0))
            action = g["action"][()]
            success = bool(g.attrs.get("success", False))

            env = make_button_env(
                seed=seed, password=password, horizon=1000,
                camera_names=("agentview", "robot0_eye_in_hand"),
                use_camera_obs=False,  # no camera needed for labels
            )
            try:
                counts = [0]  # t=0: nothing pressed yet
                for a in action:
                    _, _, done, info = env.step(a.astype(np.float32))
                    pc = int(info.get("press_count", 0))
                    counts.append(min(pc, len(password)))
                    if done:
                        break
            finally:
                env.close()

            env_pc = int(g.attrs.get("press_count", -1))
            replay_final = counts[-1]
            # normal successful/failed demos terminate at the LAST recorded
            # action; only treat termination before the end as drift
            if len(counts) - 1 < len(action):
                print(f"{name} pw={password}: replay terminated early "
                      f"({len(counts) - 1}/{len(action)} steps), skip", flush=True)
                continue
            if replay_final != env_pc:
                print(f"{name} pw={password}: MISMATCH replay={replay_final} "
                      f"h5={env_pc}, skip", flush=True)
                continue

            counts = np.asarray(counts, dtype=np.int64)
            next_key = np.full(counts.shape, -1, dtype=np.int64)
            for i, c in enumerate(counts):
                if c < len(password):
                    next_key[i] = int(password[c]) - 1
            labels[f"{name}/count"] = counts
            labels[f"{name}/next_key"] = next_key
            print(f"{name} pw={password} T={len(action)} final_press={counts[-1]} OK", flush=True)

            done += 1
            if args.max_episodes and done >= args.max_episodes:
                break

    np.savez_compressed(out_path, **labels)
    print(f"saved {len(labels) // 2} episodes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
