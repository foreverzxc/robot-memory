"""Compute normalization stats from a collected button-demo HDF5 file.

Outputs the same schema consumed by build_sub_cache.py / eval scripts:

    {"button_random_pw": {"state": {"mean": [...], "std": [...]},
                          "action": {"min": [...], "max": [...]}}}

state is 8-D, action is 7-D.
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--key", default="button_random_pw")
    args = parser.parse_args()

    states = []
    actions = []
    with h5py.File(args.h5, "r") as f:
        names = sorted(f.keys())
        for name in names:
            g = f[name]
            states.append(np.asarray(g["state"][()], dtype=np.float32))
            actions.append(np.asarray(g["action"][()], dtype=np.float32))

    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)

    stats = {
        args.key: {
            "state": {
                "mean": states.mean(axis=0).astype(np.float32).tolist(),
                "std": states.std(axis=0).astype(np.float32).tolist(),
            },
            "action": {
                "min": actions.min(axis=0).astype(np.float32).tolist(),
                "max": actions.max(axis=0).astype(np.float32).tolist(),
            },
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(f"saved stats -> {args.out} ({states.shape[0]} states, {actions.shape[0]} actions)")


if __name__ == "__main__":
    main()
