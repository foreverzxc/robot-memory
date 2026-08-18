"""批量采集按钮密码 demo 数据（HDF5），并统计单条耗时。

用法：
    $env:MUJOCO_GL = "glfw"
    # 固定密码（默认 112）
    .\\.venv\\Scripts\\python.exe collect_button_demos.py --num_demos 200
    # 随机密码：8 种 3 位密码 × 每种 25 条 = 200 条（分层、打乱顺序）
    .\\.venv\\Scripts\\python.exe collect_button_demos.py --per_password 25

输出：
    data/button_demos/demos.h5            固定密码：图像观测 + 动作 + 元数据
    data/button_demos/random_pw/demos.h5  随机密码：同上（含指令文本）
    两个目录下都有 timing.json
"""

import argparse
import json
import os
import random
import time

import h5py
import numpy as np

from button_env_v2 import make_button_env
from button_expert import ButtonExpert

PASSWORD_3DIGIT = ["111", "112", "121", "122", "211", "212", "221", "222"]
DIRECTION_MAP = {"1": "left", "2": "right"}


def password_to_instruction(password):
    """统一 L/R 格式：112 -> press password L L R（不混用其他措辞）。"""
    letters = ["left" if c == "1" else "right" for c in str(password)]
    return f"press password {' '.join(letters)}"


def collect_demo(env, expert, password, seed, horizon):
    """采集一条 demo，返回 (images, wrists, states, actions, meta)。

    每个动作对齐一条观测：image/wrist/state 是执行该动作前的观测。
    """
    np.random.seed(seed)
    obs = env.reset()
    expert.reset(env)

    images = []
    wrists = []
    states = []
    actions = []
    infos = []
    t0 = time.perf_counter()

    steps = 0
    while not env.done and steps < horizon:
        action = expert.act()
        images.append(obs["agentview_image"].copy())
        wrists.append(obs["robot0_eye_in_hand_image"].copy())
        states.append(
            np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    obs["robot0_eef_quat"],
                    [obs["robot0_gripper_qpos"][0]],
                ]
            ).astype(np.float32)
        )
        actions.append(action.copy())
        obs, reward, done, info = env.step(action)
        infos.append(info)
        steps += 1
        if info.get("success") or info.get("failed"):
            break

    wall_time = time.perf_counter() - t0
    final = infos[-1] if infos else {}
    meta = {
        "seed": seed,
        "steps": steps,
        "sim_seconds": steps * env.control_timestep,
        "wall_time": wall_time,
        "press_count": final.get("press_count", 0),
        "success": bool(final.get("success", False)),
        "failed": bool(final.get("failed", False)),
        "password": password,
        "instruction": password_to_instruction(password),
    }
    return (
        np.stack(images),
        np.stack(wrists),
        np.stack(states),
        np.stack(actions),
        meta,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=10, help="固定密码模式：采集条数")
    parser.add_argument(
        "--per_password",
        type=int,
        default=0,
        help="随机密码模式：每种密码采 N 条（8 种 3 位密码分层采集，与 --num_demos 二选一）",
    )
    parser.add_argument("--passwords", default=",".join(PASSWORD_3DIGIT))
    parser.add_argument("--random_passwords", action="store_true",
                        help="随机密码模式：每条 demo 随机取一个 --password_len 位的 L/R 密码")
    parser.add_argument("--password_len", type=int, default=3)
    parser.add_argument("--out_dir", default=os.path.join("data", "button_demos"))
    parser.add_argument("--seed_base", type=int, default=100)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3, help="单条失败后换种子重试次数")
    parser.add_argument("--password", default="112")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.02, help="初始化关节噪声幅度（0=完全确定）")
    args = parser.parse_args()

    init_noise = {"magnitude": args.noise, "type": "gaussian"}

    if args.random_passwords:
        rng = np.random.default_rng(args.shuffle_seed)
        plan = ["".join(rng.choice(list("12"), size=args.password_len)) for _ in range(args.num_demos)]
        passwords = sorted(set(plan))
        env = make_button_env(
            seed=0,
            password=plan[0],
            horizon=args.horizon,
            initialization_noise=init_noise,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        expert = None
    elif args.per_password > 0:
        # 随机密码分层模式：8 种密码 × 每种 N 条，打乱顺序
        passwords = [p.strip() for p in args.passwords.split(",") if p.strip()]
        plan = [pw for pw in passwords for _ in range(args.per_password)]
        random.Random(args.shuffle_seed).shuffle(plan)
        if args.out_dir == os.path.join("data", "button_demos"):
            args.out_dir = os.path.join("data", "button_demos", "random_pw")
        env = make_button_env(
            seed=0,
            password=plan[0],
            horizon=args.horizon,
            initialization_noise=init_noise,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        expert = None
    else:
        passwords = [args.password]
        plan = [args.password] * args.num_demos
        env = make_button_env(
            seed=0,
            password=args.password,
            horizon=args.horizon,
            initialization_noise=init_noise,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        expert = ButtonExpert(env, password=args.password)

    os.makedirs(args.out_dir, exist_ok=True)
    h5_path = os.path.join(args.out_dir, "demos.h5")
    total_t0 = time.perf_counter()
    metas = []
    pw_stats = {pw: {"total": 0, "success": 0, "failed": 0, "wall_sum": 0.0} for pw in passwords}

    with h5py.File(h5_path, "w") as f:
        f.attrs["password_mode"] = (
            f"random_pw_len{args.password_len}" if args.random_passwords
            else ("random_3digit" if args.per_password > 0 else "fixed")
        )
        f.attrs["passwords"] = ",".join(passwords)
        f.attrs["per_password"] = args.per_password
        f.attrs["num_demos"] = len(plan)
        f.attrs["init_noise"] = args.noise

        for i, pw in enumerate(plan):
            if env._password != pw:
                env.set_password(pw)
            if expert is None or expert.password != pw:
                expert = ButtonExpert(env, password=pw)
            seed = args.seed_base + i
            images = wrists = states = actions = meta = None
            for attempt in range(args.retries + 1):
                images, wrists, states, actions, meta = collect_demo(env, expert, pw, seed, args.horizon)
                if meta["success"]:
                    break
                seed += 10000  # 换一个种子重试

            g = f.create_group(f"demo_{i:03d}")
            g.create_dataset("image", data=images, dtype=np.uint8)
            g.create_dataset("wrist_image", data=wrists, dtype=np.uint8)
            g.create_dataset("state", data=states, dtype=np.float32)
            g.create_dataset("action", data=actions, dtype=np.float32)
            for k, v in meta.items():
                g.attrs[k] = v
            metas.append(meta)

            st = pw_stats[pw]
            st["total"] += 1
            st["wall_sum"] += meta["wall_time"]
            if meta["success"]:
                st["success"] += 1
            if meta["failed"]:
                st["failed"] += 1

            status = "OK " if meta["success"] else ("FAIL" if meta["failed"] else "TIME")
            print(
                f"demo {i + 1:02d}/{len(plan)} pw={pw} seed={seed} {status} "
                f"steps={meta['steps']:4d} sim={meta['sim_seconds']:.2f}s "
                f"wall={meta['wall_time']:.2f}s"
            )

    total_time = time.perf_counter() - total_t0
    ok = [m for m in metas if m["success"]]
    wall = np.array([m["wall_time"] for m in metas])
    steps = np.array([m["steps"] for m in metas])
    summary = {
        "num_demos": len(plan),
        "success": len(ok),
        "failed": sum(m["failed"] for m in metas),
        "per_password_stats": {
            pw: {
                "total": s["total"],
                "success": s["success"],
                "failed": s["failed"],
                "wall_mean": round(s["wall_sum"] / s["total"], 4) if s["total"] else None,
            }
            for pw, s in pw_stats.items()
        },
        "wall_time_per_demo_mean": float(wall.mean()),
        "wall_time_per_demo_std": float(wall.std()),
        "wall_time_per_demo_min": float(wall.min()),
        "wall_time_per_demo_max": float(wall.max()),
        "steps_per_demo_mean": float(steps.mean()),
        "total_wall_time": total_time,
        "h5_path": h5_path,
    }
    with open(os.path.join(args.out_dir, "timing.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print()
    print(f"成功 {len(ok)}/{len(plan)}，失败 {summary['failed']}")
    print(f"单条 wall time: mean {wall.mean():.2f}s, min {wall.min():.2f}s, max {wall.max():.2f}s")
    print(f"总耗时: {total_time:.2f}s  ->  约 {total_time / max(len(ok), 1):.2f}s/成功条")
    for pw, s in pw_stats.items():
        print(f"  密码 {pw}: {s['success']}/{s['total']} 成功")
    print("HDF5:", os.path.abspath(h5_path))
    env.close()


if __name__ == "__main__":
    main()
