"""HDF5 trajectory dataset for the two-button password task.

Input: demos.h5 produced by the WM button collection scripts.
Output follows Patch Policy's TrajectoryDataset convention so it can be used by
its slicing/embedding utilities, plus password index tensors:

    __getitem__(idx) -> (obs, action, mask, password_idx, password_len)
        obs:          (T, V, 3, H, W) float32 in [0, 1]
        action:       (T, 7) float32
        mask:         (T,) bool
        password_idx: (T, max_len) int64, padded with PAD_IDX
        password_len: (T,) int64
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .password_tokens import PAD_IDX

BUTTON_CAMERAS = ("agentview", "robot0_eye_in_hand")


class ButtonH5Dataset:
    def __init__(
        self,
        h5_path,
        cameras=("agentview", "robot0_eye_in_hand"),
        image_size=(224, 224),
        max_len=6,
        allowed_passwords=None,
        max_episodes=0,
        allowed_episodes=None,
    ):
        self.h5_path = str(h5_path)
        self.cameras = tuple(cameras)
        self.image_size = tuple(int(x) for x in image_size)
        self.max_len = int(max_len)
        self.allowed_passwords = (
            None if allowed_passwords is None else set(str(p) for p in allowed_passwords)
        )
        self.allowed_episodes = (
            None if allowed_episodes is None else set(str(n) for n in allowed_episodes)
        )

        self.episodes = []  # (group_name, password, length)
        with h5py.File(self.h5_path, "r") as f:
            for name in sorted(f.keys()):
                if self.allowed_episodes is not None and name not in self.allowed_episodes:
                    continue
                g = f[name]
                password = str(g.attrs.get("password", ""))
                if self.allowed_passwords is not None and password not in self.allowed_passwords:
                    continue
                length = int(g["action"].shape[0])
                self.episodes.append((name, password, length))
                if max_episodes and len(self.episodes) >= int(max_episodes):
                    break

        if not self.episodes:
            raise ValueError(f"No matching episodes in {self.h5_path}")

    def __len__(self):
        return len(self.episodes)

    def get_seq_length(self, idx):
        return self.episodes[idx][2]

    def get_password(self, idx):
        return self.episodes[idx][1]

    @staticmethod
    def encode_password(password, max_len):
        if len(password) > max_len:
            raise ValueError(f"password {password!r} is longer than max_len {max_len}")
        idx = torch.full((max_len,), PAD_IDX, dtype=torch.long)
        for i, ch in enumerate(password):
            idx[i] = int(ch) - 1
        return idx, len(password)

    def _load_group(self, idx):
        name, password, length = self.episodes[idx]
        f = h5py.File(self.h5_path, "r")
        g = f[name]
        image_views = []
        for cam in self.cameras:
            key = f"{cam}_image" if cam == "robot0_eye_in_hand" else "image"
            # The WM HDF5 uses "image" for agentview and "wrist_image" for
            # robot0_eye_in_hand; accept both naming schemes.
            if key not in g and cam == "agentview":
                key = "image"
            if key not in g and cam == "robot0_eye_in_hand":
                key = "wrist_image"
            if key not in g:
                f.close()
                raise KeyError(f"Camera {cam}: group {name} has no dataset {key}")
            arr = g[key][()]
            image_views.append(arr)
        action = g["action"][()].astype(np.float32)
        f.close()

        obs = self._prepare_obs(image_views)
        mask = torch.ones(obs.shape[0], dtype=torch.bool)
        pw_idx, pw_len = self.encode_password(password, self.max_len)
        pw_idx = pw_idx.unsqueeze(0).expand(obs.shape[0], -1).contiguous()
        pw_len_tensor = torch.full((obs.shape[0],), pw_len, dtype=torch.long)
        return obs, torch.from_numpy(action), mask, pw_idx, pw_len_tensor

    def _prepare_obs(self, image_views):
        """Convert list of uint8 HWC arrays to (T, V, 3, H, W) float in [0,1]."""
        views = []
        for arr in image_views:
            if arr.ndim != 4:
                raise ValueError(f"Expected THWC image array, got shape {arr.shape}")
            # Resize on GPU-free CPU path. Use torch interpolate for speed over
            # per-frame cv2 calls.
            t = torch.from_numpy(arr).float()  # (T,H,W,C)
            t = t.permute(0, 3, 1, 2)          # (T,C,H,W)
            if t.shape[-2:] != self.image_size:
                t = F.interpolate(
                    t,
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
            t = t.permute(0, 2, 3, 1).clamp(0, 255).round().byte().permute(0, 3, 1, 2)
            views.append(t / 255.0)
        return torch.stack(views, dim=1)  # (T,V,C,H,W)

    def get_frames(self, idx, frames):
        obs, action, mask, pw_idx, pw_len = self._load_group(idx)
        frames = list(frames)
        return obs[frames], action[frames], mask[frames], pw_idx[frames], pw_len[frames]

    def get_all_actions(self):
        actions = []
        for i in range(len(self)):
            _, action, _, _, _ = self._load_group(i)
            actions.append(action)
        if not actions:
            return torch.empty((0, 7))
        return torch.cat(actions, dim=0)

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
