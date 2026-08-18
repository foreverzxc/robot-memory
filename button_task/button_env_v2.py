"""按钮环境 v2 — 2 个固定按钮 + 按下变亮反馈 + 密码 112 判定。

设计要点（与 docs/BUTTON_ENV_TASK.md 一致）：
- 按钮 = 固定在桌面的 CylinderObject（joints=None，不物理移动）
- 按压检测 = Panda 指尖/指垫 geom 与按钮碰撞 geom 的接触
- 反馈 = 按下时按钮视觉 geom 变为亮绿色，松开恢复暗色（geom_rgba 即时修改）
- 判定 = 只在「未按 -> 按下」上升沿判定一次，顺序错误即失败，按满密码成功
- 观测/动作与 TurboVLA/LIBERO 兼容：agentview 256x256 图像 + OSC_POSE 7-D 动作
"""

import inspect

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.transform_utils import mat2quat


# robosuite 1.4.x uses ``mount_types`` while 1.5.x renamed it to ``base_types``.
# The project runs collection on 1.4.1 and closed-loop evaluation on 1.5.2, so
# detect the supported kwarg at import time instead of hard-coding one name.
_MANIP_ENV_PARAMS = inspect.signature(ManipulationEnv.__init__).parameters
MOUNT_KWARG = "base_types" if "base_types" in _MANIP_ENV_PARAMS else "mount_types"


# 按钮外观参数（长方体按钮：顶面是平面，手指底端压在平面上，接触更稳定）
BUTTON_SIZE = [0.035, 0.035, 0.017]   # 半尺寸 x/y/z：7cm x 7cm x 3.4cm
BUTTON_DARK_RGBA = np.array([0.13, 0.13, 0.17, 1.0])
BUTTON_LIGHT_RGBA = np.array([0.0, 0.95, 0.35, 1.0])

# 颜色显示滞回：亮起立即生效；熄灭需连续 N 个子步无接触（约 8ms，远小于一帧 50ms，
# 不会让历史状态进入观测图像）。物理接触已加固后这层只作为数值噪声兜底。
DISPLAY_HOLD_SUBSTEPS = 4

# 按钮位置：(x, y)，z 由桌面高度决定。
# (±0.15, 0.35) 会把手臂逼到关节 3 限位，按完右按钮回左按钮时路径不可达；
# 内收到 (±0.12, 0.30) 保证 8 种密码顺序都能稳定完成。
BTN_POSITIONS = {
    "btn_left": (-0.12, 0.30),
    "btn_right": (0.12, 0.30),
}

# 释放去抖：接触消失需连续 100 个子步（200ms）才认为真正松开。
# 实测按住期间接触可能断续 30~60+ 子步，去抖必须明显长于这个值，
# 否则同一次按压会被重复计数（顺序密码会因此误判失败）。
RELEASE_DEBOUNCE_SUBSTEPS = 100


class ButtonEnv(ManipulationEnv):
    """Panda 机械臂按压 2 个固定按钮，按密码顺序（如 112 = 左-左-右）完成任务。"""

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        password="112",
        agentview_camera=None,
    ):
        # 任务状态
        self._password = str(password)
        # 可选：(pos(3), quat_wxyz(4))，用于把 agentview 相机对准按钮区域
        self._agentview_camera = agentview_camera if agentview_camera is not None else DEFAULT_AGENTVIEW_CAMERA
        self._press_count = 0
        self._last_pressed = None
        self._failed = False
        # 抖动抑制：连续 3 个仿真子步确认按下、连续 5 个子步确认松开，
        # 避免指尖在按钮表面弹跳导致一次按压被重复计数。
        self._press_candidate = None
        self._press_candidate_steps = 0
        self._release_steps = 0
        self._display_pressed = None
        self._display_off_steps = 0

        # 按钮模型引用（_load_model 时填充）
        self._button_objs = {}
        self._button_gids = {}        # name -> 碰撞 geom id（接触检测用）
        self._button_vis_gids = {}    # name -> 视觉 geom id（颜色反馈用）
        self._dark_rgba = {}          # name -> 暗色 rgba

        # 桌面设置（与 Lift 一致）
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        # eef z above the button top by this margin is treated as "clearly lifted".
        self._release_z = self.table_offset[2] + 2 * BUTTON_SIZE[2] + 0.04

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            **{MOUNT_KWARG: "default"},
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    # ------------------------------------------------------------------ #
    # 模型加载 / 引用
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """按 robosuite 标准写法创建 self.model（修复旧版 self.model=None 的问题）。"""
        super()._load_model()

        # 机器人放在 table arena 的标准位置
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 桌面
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # 可选：重新对准 agentview 相机，让两个按钮都清晰可见
        if self._agentview_camera is not None:
            cam = mujoco_arena.worldbody.find("./camera[@name='agentview']")
            cam.set("pos", array_to_string(self._agentview_camera[0]))
            cam.set("quat", array_to_string(self._agentview_camera[1]))

        # 2 个固定按钮：中心 z = 桌面高度 + 半高，让底面正好贴桌
        objects = []
        for name, (bx, by) in BTN_POSITIONS.items():
            btn = BoxObject(
                name=name,
                size=BUTTON_SIZE,
                rgba=BUTTON_DARK_RGBA,
                joints=None,  # 固定物体，不移动
                obj_type="all",
            )
            pos = [bx, by, self.table_offset[2] + BUTTON_SIZE[2]]
            btn.get_obj().set("pos", array_to_string(pos))
            self._button_objs[name] = btn
            objects.append(btn)

        # 任务模型 = arena + 机器人 + 按钮
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self._button_gids = {}
        self._button_vis_gids = {}
        self._dark_rgba = {}
        for name, btn in self._button_objs.items():
            gid = self.sim.model.geom_name2id(btn.contact_geoms[0])
            vis_gid = self.sim.model.geom_name2id(btn.visual_geoms[0])
            self._button_gids[name] = gid
            self._button_vis_gids[name] = vis_gid
            self._dark_rgba[name] = self.sim.model.geom_rgba[vis_gid].copy()

    # ------------------------------------------------------------------ #
    # 接触检测 / 颜色反馈 / 密码判定
    # ------------------------------------------------------------------ #

    def _get_finger_gids(self):
        """Panda 夹爪的所有手指相关碰撞 geom id（指体 + 指垫）。"""
        gripper = self.robots[0].gripper
        if isinstance(gripper, dict):
            gripper = next(iter(gripper.values()))
        names = []
        for group in gripper.important_geoms.values():
            names.extend(group)
        return {self.sim.model.geom_name2id(n) for n in names}

    def _get_pressed_buttons(self):
        """返回当前被按住的按钮名（若有），否则 None。"""
        finger_gids = self._get_finger_gids()
        for name, gid in self._button_gids.items():
            for c in self.sim.data.contact:
                if (c.geom1 == gid and c.geom2 in finger_gids) or (
                    c.geom2 == gid and c.geom1 in finger_gids
                ):
                    return name
        return None

    def _eef_z(self):
        site_id = self.robots[0].eef_site_id
        if isinstance(site_id, dict):
            site_id = next(iter(site_id.values()))
        return float(self.sim.data.site_xpos[site_id][2])

    def _register_press(self, name):
        """上升沿判定：检查新按压是否等于密码下一项。"""
        idx = self._press_count
        if idx < len(self._password):
            expected_name = self._button_name_for_char(self._password[idx])
            if name == expected_name:
                self._press_count += 1
            else:
                self._failed = True

    def _update_button_state(self):
        """根据当前接触状态更新颜色（带显示滞回），并做去抖后的上升沿判定。"""
        pressed = self._get_pressed_buttons()
        if pressed is not None:
            # 按下确认（连续 3 个子步接触同一按钮）
            if pressed == self._press_candidate:
                self._press_candidate_steps += 1
            else:
                self._press_candidate = pressed
                self._press_candidate_steps = 1
            if self._press_candidate_steps >= 3 and pressed != self._last_pressed:
                self._register_press(pressed)
                self._last_pressed = pressed
                self._press_candidate_steps = 0
            self._release_steps = 0
        else:
            # 松开确认（连续 5 个子步无接触才视为真正松开）
            self._press_candidate = None
            self._press_candidate_steps = 0
            if self._last_pressed is not None and self._eef_z() > self._release_z:
                self._last_pressed = None
                self._release_steps = 0

        # 显示滞回：有接触立即亮；无接触时保持 DISPLAY_HOLD_SUBSTEPS 个子步再灭
        if pressed is not None:
            self._display_pressed = pressed
            self._display_off_steps = 0
        elif self._display_pressed is not None:
            self._display_off_steps += 1
            if self._display_off_steps >= DISPLAY_HOLD_SUBSTEPS:
                self._display_pressed = None
                self._display_off_steps = 0

        for name, vis_gid in self._button_vis_gids.items():
            self.sim.model.geom_rgba[vis_gid] = (
                BUTTON_LIGHT_RGBA if name == self._display_pressed else self._dark_rgba[name]
            )

    def _button_name_for_char(self, ch):
        return "btn_left" if ch == "1" else "btn_right"

    # ------------------------------------------------------------------ #
    # step / reset / reward
    # ------------------------------------------------------------------ #

    def step(self, action):
        """与 base 相同，但在每次 sim.step() 之后立即更新按钮状态，
        保证颜色反馈与接触同步，且本帧观测已包含颜色变化。"""
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        policy_step = True
        for _ in range(int(self.control_timestep / self.model_timestep)):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_button_state()  # 接触/颜色/判定都在这里
            self._update_observables()
            policy_step = False

        self.cur_time += self.control_timestep
        reward, done, info = self._post_action(action)

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()

        # 强制渲染一次，保证返回的观测与当前按钮颜色一致
        self._update_observables(force=True)
        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()

        info["press_count"] = self._press_count
        info["success"] = self._check_success()
        info["failed"] = self._failed
        if info["success"] or self._failed:
            done = True
            self.done = True

        return observations, reward, done, info

    def reset(self):
        self._press_count = 0
        self._last_pressed = None
        self._failed = False
        self._press_candidate = None
        self._press_candidate_steps = 0
        self._release_steps = 0
        self._display_pressed = None
        self._display_off_steps = 0
        obs = super().reset()
        # 硬重置后 sim 是新的，需恢复按钮暗色
        self._update_button_state()
        return obs

    def set_password(self, password):
        """切换任务密码（下次 reset 生效，状态由 reset() 清空）。"""
        self._password = str(password)

    def reward(self, action=None):
        """稀疏奖励：成功 1.0，否则 0.0。"""
        return 1.0 if self._check_success() else 0.0

    def _check_success(self):
        return self._press_count >= len(self._password) and not self._failed


def camera_look_at(pos, target):
    """生成 (camera_pos, camera_quat_wxyz)，使相机位于 pos 并看向 target。

    返回的 quat 是 MuJoCo XML 使用的 wxyz 顺序。
    """
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    d = target - pos
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, -1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    c = np.dot(z, d)
    if s < 1e-8:
        rot = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )
        rot = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    q_xyzw = mat2quat(rot)
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return pos, q_wxyz


# 默认 agentview 相机：对准两个按钮（原始 table_arena 的 agentview 会把按钮挤到画面边缘/被机械臂遮挡）
DEFAULT_AGENTVIEW_CAMERA = camera_look_at((0.75, -0.65, 1.15), (0.0, 0.30, 0.78))


def make_button_env(seed=0, password="112", horizon=1000, camera_names="agentview", **kwargs):
    """创建 ButtonEnv 并完成一次复位。"""
    np.random.seed(seed)
    if hasattr(suite, "load_composite_controller_config"):
        controller_config = suite.load_composite_controller_config(controller="BASIC")
        # ButtonExpert emits world-frame deltas, but robosuite 1.5 defaults the
        # OSC part to the robot-base frame. Align it with the world frame so the
        # same 7-D actions mean the same thing as in the 1.4 collection env.
        for part in controller_config.get("body_parts", {}).values():
            if isinstance(part, dict) and part.get("type") == "OSC_POSE":
                part["input_ref_frame"] = "world"
    elif hasattr(suite, "load_part_controller_config"):
        controller_config = suite.load_part_controller_config(default_controller="OSC_POSE")
        if isinstance(controller_config, dict) and controller_config.get("type") == "OSC_POSE":
            controller_config["input_ref_frame"] = "world"
    else:
        controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    kwargs.pop("controller_configs", None)
    camera_heights = kwargs.pop("camera_heights", 256)
    camera_widths = kwargs.pop("camera_widths", 256)
    use_camera_obs = kwargs.pop("use_camera_obs", True)
    env = ButtonEnv(
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=use_camera_obs,
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        password=password,
        horizon=horizon,
        **kwargs,
    )
    env.reset()
    return env


if __name__ == "__main__":
    env = make_button_env()
    obs = env.reset()
    print("BUTTON_ENV_OK")
    print("obs keys:", list(obs.keys()))
    print("obs shapes:", {k: np.shape(v) for k, v in obs.items()})
    print("action dim:", env.action_dim)
    print("button geoms:", {k: v.contact_geoms[0] for k, v in env._button_objs.items()})
    print("button visual geoms:", {k: v.visual_geoms[0] for k, v in env._button_objs.items()})
    print("finger geoms:", sorted(env._get_finger_gids()))
    print("button xpos:", {k: env.sim.data.body_xpos[env.sim.model.body_name2id(v.root_body)] for k, v in env._button_objs.items()})
    env.close()
