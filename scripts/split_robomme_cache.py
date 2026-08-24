"""Create reproducible train/holdout cache directories from a full cache."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "holdout"], required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be between 0 and 1")

    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(manifest))
    holdout_count = max(1, int(round(len(manifest) * args.holdout_fraction)))
    holdout_ids = set(int(index) for index in order[:holdout_count])
    selected = [
        index
        for index in range(len(manifest))
        if (index in holdout_ids) == (args.split == "holdout")
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    output_manifest = []
    for index in selected:
        item = dict(manifest[index])
        source_file = args.source / item["file"]
        shutil.copy2(source_file, args.out / source_file.name)
        output_manifest.append(item)
    for name in ("stats.json",):
        source_file = args.source / name
        if source_file.exists():
            shutil.copy2(source_file, args.out / name)
    (args.out / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8"
    )
    print(f"split={args.split} episodes={len(output_manifest)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
