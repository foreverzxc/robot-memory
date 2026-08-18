"""Scripted expert：按密码顺序按压按钮，输出与 TurboVLA 兼容的 7-D OSC_POSE 动作。"""

import numpy as np


class ButtonExpert:
    """状态机式 scripted expert。

    每个密码字符分六个阶段：
      hover    -> 末端移到按钮正上方
      descend  -> 缓慢下压直到接触（按钮变亮）
      press_in -> 再下压几步让接触稳定
      hold     -> 保持按压若干步（不重复计数）
      lift     -> 抬起回到 hover
      settle   -> 悬停几拍确认松开，进入下一个字符

    动作格式：OSC_POSE 7-D = [dx, dy, dz, dax, day, daz, gripper]，gripper 常闭(+1)。
    """

    def __init__(
        self,
        env,
        password="112",
        hover_z=0.13,
        descend_step=0.02,
        press_in_steps=4,
        press_in_step=0.008,
        hold_steps=8,
        settle_steps=4,
        tol=0.015,
        max_move_steps=80,
    ):
        self.env = env
        self.password = str(password)
        self.hover_z = hover_z
        self.descend_step = descend_step
        self.press_in_steps = press_in_steps
        self.press_in_step = press_in_step
        self.hold_steps = hold_steps
        self.settle_steps = settle_steps
        self.tol = tol
        self.max_move_steps = max_move_steps
        self.reset(env)

    def reset(self, env=None):
        if env is not None:
            self.env = env
        self.targets = self._get_button_positions(self.env)
        self.char_idx = 0
        self.phase = "hover"
        self.hold_count = 0
        self.press_in_count = 0
        self.settle_count = 0
        self.phase_steps = 0

    # ------------------------------------------------------------------ #

    def _get_button_positions(self, env):
        out = {}
        for name, btn in env._button_objs.items():
            body_id = env.sim.model.body_name2id(btn.root_body)
            out[name] = env.sim.data.body_xpos[body_id].copy()
        return out

    def _eef_pos(self):
        site_id = self.env.robots[0].eef_site_id
        if isinstance(site_id, dict):
            site_id = next(iter(site_id.values()))
        return self.env.sim.data.site_xpos[site_id].copy()

    def _make_action(self, delta):
        # OSC_POSE 默认把 [-1, 1] 线性映射到 [-0.05, 0.05] 米，
        # 因此把期望位移（米）除以 0.05 得到控制器输入。
        delta = np.asarray(delta, dtype=float)
        normalized = np.clip(delta / 0.05, -1.0, 1.0)
        return np.concatenate([normalized, np.zeros(3), [1.0]])

    def _current_button(self):
        ch = self.password[self.char_idx]
        return "btn_left" if ch == "1" else "btn_right"

    # ------------------------------------------------------------------ #

    def act(self):
        """返回当前阶段的一个 7-D 动作。"""
        if self.char_idx >= len(self.password):
            return self._make_action(np.zeros(3))

        env = self.env
        target = self.targets[self._current_button()]
        eef = self._eef_pos()
        hover = target + np.array([0.0, 0.0, self.hover_z])
        self.phase_steps += 1

        if self.phase == "hover":
            delta = hover - eef
            if np.linalg.norm(delta) < self.tol or self.phase_steps > self.max_move_steps:
                self.phase = "descend"
                self.phase_steps = 0
                return self._make_action(np.zeros(3))
            return self._make_action(delta)

        if self.phase == "descend":
            # 已经接触（按钮按下）则进入保持
            if env._get_pressed_buttons() is not None:
                self.phase = "press_in"
                self.press_in_count = 0
                self.phase_steps = 0
                return self._make_action(np.zeros(3))
            # 先对齐 xy，再按固定步长下压
            delta = np.array(
                [
                    (target[0] - eef[0]) * 0.3,
                    (target[1] - eef[1]) * 0.3,
                    -self.descend_step,
                ]
            )
            if self.phase_steps > self.max_move_steps:
                self.phase = "hold"
                self.hold_count = 0
                self.phase_steps = 0
            return self._make_action(delta)

        if self.phase == "press_in":
            # 再向下压几步，让接触稳定（避免刚碰到就弹开）
            self.press_in_count += 1
            if self.press_in_count >= self.press_in_steps:
                self.phase = "hold"
                self.hold_count = 0
                self.phase_steps = 0
            return self._make_action(np.array([0.0, 0.0, -self.press_in_step]))

        if self.phase == "hold":
            self.hold_count += 1
            if self.hold_count >= self.hold_steps:
                self.phase = "lift"
                self.phase_steps = 0
            return self._make_action(np.zeros(3))

        if self.phase == "lift":
            delta = hover - eef
            if np.linalg.norm(delta) < self.tol or self.phase_steps > self.max_move_steps:
                self.phase = "settle"
                self.settle_count = 0
                self.phase_steps = 0
                return self._make_action(np.zeros(3))
            return self._make_action(delta)

        if self.phase == "settle":
            # 悬停几拍，确保接触已真正松开，再进入下一个按钮
            self.settle_count += 1
            if self.settle_count >= self.settle_steps:
                self.char_idx += 1
                self.phase = "hover"
                self.phase_steps = 0
            return self._make_action(np.zeros(3))

        return self._make_action(np.zeros(3))

    def is_done(self):
        return self.char_idx >= len(self.password)


def run_expert_episode(env, password="112", **kwargs):
    """跑完一条 expert demo，返回 (obs 列表, action 列表, info 列表, 最终 info)。"""
    expert = ButtonExpert(env, password=password, **kwargs)
    obs_list, action_list, info_list = [], [], []
    obs = env.reset()
    obs_list.append(obs)

    steps = 0
    while not env.done and steps < env.horizon:
        action = expert.act()
        obs, reward, done, info = env.step(action)
        obs_list.append(obs)
        action_list.append(action)
        info_list.append(info)
        steps += 1
        if info.get("success", False) or info.get("failed", False):
            break

    return obs_list, action_list, info_list, info_list[-1] if info_list else {}
