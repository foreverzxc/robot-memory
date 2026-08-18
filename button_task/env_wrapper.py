"""Patch-Policy-compatible wrapper for the WM ButtonEnv.

Expected Patch Policy eval interface:
- reset(goal_idx=...) -> np.ndarray of shape (V, 3, H, W), float [0,1]
- step(action) -> (obs, reward, done, info)
- info["image"]: HWC uint8 image for video recording
- info["success"], info["failed"], info["press_count"]
"""

from __future__ import annotations

import gym
import numpy as np
import torch
import torch.nn.functional as F

CAMERA_KEY_SUFFIX = {
    "agentview": "agentview_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
    "sideview": "sideview_image",
}


def _camera_obs(env_obs, cameras, image_size):
    views = []
    for cam in cameras:
        key = CAMERA_KEY_SUFFIX.get(cam, f"{cam}_image")
        if key not in env_obs:
            raise KeyError(f"ButtonEnv observation does not contain {key}; got {sorted(env_obs.keys())}")
        img = np.asarray(env_obs[key])  # HWC uint8
        views.append(img)

    # Resize all views on CPU as uint8 HWC first.
    resized = []
    for img in views:
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        if (int(t.shape[-2]), int(t.shape[-1])) != tuple(image_size):
            t = F.interpolate(t, size=tuple(image_size), mode="bilinear", align_corners=False)
        t = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).round().byte()
        resized.append(t.numpy())

    out = np.stack([img.astype(np.float32) / 255.0 for img in resized], axis=0)
    return out.transpose(0, 3, 1, 2)  # V,C,H,W


class ButtonPatchWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        passwords,
        cameras=("agentview", "robot0_eye_in_hand"),
        image_size=(224, 224),
        max_steps=600,
    ):
        super().__init__(env)
        self.passwords = list(passwords)
        if not self.passwords:
            raise ValueError("passwords must not be empty")
        self.cameras = tuple(cameras)
        self.image_size = tuple(image_size)
        self.max_steps = int(max_steps)
        self._password = self.passwords[0]
        self._step_count = 0

    @property
    def password(self):
        return self._password

    def set_password(self, password):
        self._password = str(password)
        self.env.set_password(self._password)

    def reset(self, goal_idx=0, **kwargs):
        self._password = self.passwords[int(goal_idx) % len(self.passwords)]
        self.env.set_password(self._password)
        env_obs = self.env.reset(**kwargs)
        self._step_count = 0
        return _camera_obs(env_obs, self.cameras, self.image_size)

    def step(self, action):
        env_obs, reward, done, info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = True

        info = dict(info)
        info["image"] = _first_view_hwc(env_obs, self.cameras)
        info["success"] = bool(info.get("success", False))
        info["failed"] = bool(info.get("failed", False))
        info["press_count"] = int(info.get("press_count", 0))
        if info["success"] or info["failed"]:
            done = True

        return _camera_obs(env_obs, self.cameras, self.image_size), reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed=seed)


def _first_view_hwc(env_obs, cameras):
    key = CAMERA_KEY_SUFFIX.get(cameras[0], f"{cameras[0]}_image")
    return np.asarray(env_obs[key])


def make_button_patch_env(
    seed=0,
    password="112",
    passwords=None,
    cameras=("agentview", "robot0_eye_in_hand"),
    image_size=(224, 224),
    horizon=1000,
    max_steps=600,
    **button_kwargs,
):
    from .button_env_v2 import make_button_env  # copied by prepare_workspace.ps1

    passwords = list(passwords or [password])
    env = make_button_env(
        seed=seed,
        password=passwords[0],
        horizon=horizon,
        camera_names=cameras,
        **button_kwargs,
    )
    return ButtonPatchWrapper(
        env,
        passwords=passwords,
        cameras=cameras,
        image_size=image_size,
        max_steps=max_steps,
    )
