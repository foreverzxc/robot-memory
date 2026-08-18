"""Create a deterministic train/holdout split by password.

Passwords are valid strings over {1,2} with length <= 6.

Usage:
    python scripts/make_password_split.py
    python scripts/make_password_split.py `
        --h5 "E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5" `
        --out "button_task/password_split.json" `
        --holdout_count 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_H5S = [
    "E:/WM/turbovla/data/button_demos/random_pw6_lang_1000/demos.h5",
    "E:/WM/turbovla/data/button_demos/random_pw6_lang_100/demos.h5",
    "E:/WM/turbovla/data/button_demos/random_pw6_lang_small/demos.h5",
]


def is_valid_password(pw: str, max_len: int) -> bool:
    return 1 <= len(pw) <= max_len and all(c in "12" for c in pw)


def stable_order(items):
    # Reproducible ordering across Python versions.
    return sorted(items, key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", action="append", dest="h5_files", default=None)
    parser.add_argument("--out", type=Path, default=Path("button_task/password_split.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=6)
    parser.add_argument(
        "--holdout_count",
        type=int,
        default=16,
        help="number of held-out passwords; 0 means use 20 percent",
    )
    parser.add_argument(
        "--holdout",
        default="",
        help="comma separated explicit holdout passwords, overrides --holdout_count",
    )
    args = parser.parse_args()

    h5_files = args.h5_files or DEFAULT_H5S
    counts = {}
    for raw in h5_files:
        path = Path(raw)
        if not path.exists():
            print(f"WARNING: skip missing file {path}", file=sys.stderr)
            continue
        try:
            import h5py
        except Exception as exc:
            print(f"h5py import failed: {exc}", file=sys.stderr)
            return 2

        with h5py.File(path, "r") as f:
            for name in f.keys():
                pw = str(f[name].attrs.get("password", ""))
                if is_valid_password(pw, args.max_len):
                    counts[pw] = counts.get(pw, 0) + 1

    if not counts:
        print("No valid passwords found in provided files.", file=sys.stderr)
        return 3

    passwords = stable_order(counts.keys())

    if args.holdout.strip():
        holdout = stable_order(p.strip() for p in args.holdout.split(",") if p.strip())
        invalid = [p for p in holdout if p not in counts]
        if invalid:
            print(f"Explicit holdout passwords not found in data: {invalid}", file=sys.stderr)
            return 4
    else:
        rng_state = None
        try:
            import numpy as np
            rng = np.random.default_rng(args.seed)
            order = list(passwords)
            rng.shuffle(order)
        except Exception:
            # Fallback to hash-based stable order.
            order = passwords

        n = args.holdout_count
        if n <= 0:
            n = max(1, int(round(len(order) * 0.2)))
        n = min(n, max(1, len(order) - 1))
        holdout = stable_order(order[:n])

    train = [p for p in passwords if p not in set(holdout)]

    result = {
        "max_len": args.max_len,
        "num_passwords": len(passwords),
        "train": train,
        "holdout": holdout,
        "password_counts": {p: counts[p] for p in passwords},
        "source_files": h5_files,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}")
    print(f"train passwords: {len(train)}")
    print(f"holdout passwords: {len(holdout)} -> {holdout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
