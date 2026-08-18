"""Summarize train_button.py log.jsonl into a compact table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "log.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    has_ttt = any("ttt/gate_mean" in r for r in rows)
    if has_ttt:
        print(f"{'ep':>3} {'train':>8} {'holdout':>8} {'train_pw':>8} {'gate':>8} {'gate_abs':>8} {'ttt_norm':>8} {'inner_lr':>8} {'sec':>6}")
    else:
        print(f"{'ep':>3} {'train':>8} {'val':>8} {'holdout':>8} {'train_pw':>8} {'sec':>6}")
    for r in rows:
        ep = r.get("epoch", "?")
        tl = r.get("train_loss")
        vl = r.get("val_loss")
        hs = r["eval_holdout"]["success_rate_holdout"] if "eval_holdout" in r else None
        ts = r["eval_train"]["success_rate_train"] if "eval_train" in r else None
        sec = r.get("seconds")

        def f(x, w=8):
            return f"{x:.3f}".rjust(w) if isinstance(x, (int, float)) else "-".rjust(w)

        line = f"{str(ep):>3} {f(tl)}"
        if has_ttt:
            line += f" {f(hs)} {f(ts)} {f(r.get('ttt/gate_mean'))} {f(r.get('ttt/gate_abs_mean'))} {f(r.get('ttt/ttt_out_norm'))} {f(r.get('ttt/inner_lr'))} {str(sec):>6}"
        else:
            line += f" {f(vl)} {f(hs)} {f(ts)} {str(sec):>6}"
        print(line)
    # best holdout
    best = max(
        (r for r in rows if "eval_holdout" in r),
        key=lambda r: r["eval_holdout"]["success_rate_holdout"],
        default=None,
    )
    if best is not None:
        print(f"\nbest: epoch {best['epoch']} holdout={best['eval_holdout']['success_rate_holdout']:.3f}")
        for pw, info in sorted(best["eval_holdout"]["per_pw_holdout"].items()):
            print(f"  {pw}: success={int(info['success'])} press={info['press_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
