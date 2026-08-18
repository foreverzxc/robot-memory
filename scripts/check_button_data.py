"""Inspect E:\\WM\\turbovla button HDF5 demos.

Usage:
    python scripts/check_button_data.py
    python scripts/check_button_data.py --h5 "E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

DEFAULT_H5 = Path(
    "E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--max_groups", type=int, default=0, help="0 = inspect all groups")
    args = parser.parse_args()

    path: Path = args.h5
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    try:
        import h5py
        import numpy as np
    except Exception as exc:
        print(f"h5py/numpy import failed: {exc}", file=sys.stderr)
        return 3

    passwords = Counter()
    with h5py.File(path, "r") as f:
        names = sorted(f.keys())
        total_steps = 0
        lengths = []
        first_keys = None
        first_attrs = None

        for i, name in enumerate(names):
            if args.max_groups and i >= args.max_groups:
                break
            g = f[name]
            pw = str(g.attrs.get("password", ""))
            passwords[pw] += 1
            image = g["image"]
            action = g["action"]
            n_steps = int(image.shape[0])
            lengths.append(n_steps)
            total_steps += n_steps
            if first_keys is None:
                first_keys = list(g.keys())
                first_attrs = dict(g.attrs)
                print(f"Example group: {name}")
                print(f"  keys: {first_keys}")
                print(f"  attrs: {first_attrs}")
                for key in first_keys:
                    print(f"  {key}: shape={g[key].shape}, dtype={g[key].dtype}")

        print("\nSummary:")
        print(f"  file: {path}")
        print(f"  groups inspected: {len(lengths)}")
        print(f"  total steps: {total_steps}")
        print(f"  episode length: min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths) / len(lengths):.1f}")
        print(f"  unique passwords: {len(passwords)}")
        print("  password counts:")
        for pw, count in sorted(passwords.items()):
            print(f"    {pw}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
