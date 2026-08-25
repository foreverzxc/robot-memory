"""Download the frozen base-policy assets used by RoboMME rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vendor.base_policy.downloads import ensure_base_policy_weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=REPO_ROOT / "weights")
    args = parser.parse_args()
    weights_dir = args.weights_dir.expanduser().resolve()
    ensure_base_policy_weights(
        weights_dir / "libero" / "spatial.pth",
        weights_dir / "dinov3",
        weights_dir / "bert",
    )
    print(f"base assets ready under: {weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
