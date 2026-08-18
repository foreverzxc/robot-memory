"""Best-effort labels for the 3 passwords whose demos drift on the final press.

Replay-based labeling normally rejects episodes whose replay press_count does
not match the recorded final press_count. Diagnostics show that for the
affected passwords almost all episodes replay the first 5 presses correctly
and only the 6th (final) press is missed by a few end frames of physics drift.

Policy here:
  - replaypc == h5pc: accept (pad to episode length if the env terminated
    early after success);
  - replaypc == h5pc - 1: accept, but set the FINAL frame count to h5pc, so
    the remaining-password condition is correct for the very end of the
    episode;
  - otherwise: skip.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runs" / ".numba_cache")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--passwords", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from button_task.button_env_v2 import make_button_env

    passwords = set(args.passwords.split(","))
    labels = {}
    with h5py.File(args.h5, "r") as f:
        for name in sorted(f.keys()):
            g = f[name]
            password = str(g.attrs.get("password", ""))
            if password not in passwords:
                continue
            seed = int(g.attrs.get("seed", 0))
            h5pc = int(g.attrs.get("press_count", -1))
            action = g["action"][()]
            T = len(action)

            env = make_button_env(
                seed=seed, password=password, horizon=1000,
                camera_names=("agentview", "robot0_eye_in_hand"),
                use_camera_obs=False,
            )
            try:
                counts = [0]
                for a in action:
                    _, _, done, info = env.step(a.astype(np.float32))
                    counts.append(min(int(info.get("press_count", 0)), len(password)))
                    if done:
                        break
            finally:
                env.close()

            steps = len(counts) - 1
            replaypc = counts[-1]
            if steps < T and replaypc != h5pc:
                print(f"{name} pw={password}: early termination "
                      f"({steps}/{T}) and replay={replaypc} != h5={h5pc}, skip")
                continue
            if replaypc not in (h5pc, h5pc - 1):
                print(f"{name} pw={password}: replay={replaypc} too far from "
                      f"h5={h5pc}, skip")
                continue

            # pad to T steps, then fix the final frame when only the last
            # press was missed by replay drift
            counts = counts + [counts[-1]] * (T + 1 - len(counts))
            counts = np.asarray(counts, dtype=np.int64)
            if replaypc == h5pc - 1:
                counts[-1] = h5pc
            labels[f"{name}/count"] = counts
            print(f"{name} pw={password}: best-effort OK "
                  f"(replay={replaypc}, h5={h5pc}, final marked {h5pc})")

    np.savez_compressed(args.out, **labels)
    print(f"saved {len(labels)} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
