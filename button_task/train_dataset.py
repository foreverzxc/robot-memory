"""Slice-level dataset for VQ-BeT training on the button HDF5 data.

Slicing follows Patch Policy's ``TrajectorySlicerDataset`` semantics with an
observation window of 1: for an episode of length T there are T slices with
start s in [0, T); each slice carries ``obs[s]`` (one frame) and the action
chunk ``act[s : s + action_window]``, padded by repeating the last action when
the chunk runs past the episode end.

Random HDF5 access per slice would be far too slow (340k slices over ~1000
groups), so training iterates over *episode shards* held in memory. The
training loop calls :meth:`iter_shards` to obtain a whole shard and then
samples slices from it with :meth:`sample_batch`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .button_dataset import ButtonH5Dataset, BUTTON_CAMERAS


@dataclass
class EpisodeShard:
    """All raw data of one episode, loaded and concatenated across the shard.

    ``obs`` is uint8 (S, V, 3, H, W) at the HDF5 native resolution; frames are
    resized to the encoder size on the GPU inside :meth:`ButtonSliceDataset.
    slice_obs`.
    """

    obs: torch.Tensor  # (S, V, 3, H, W) uint8
    action: torch.Tensor  # (S, 7) float32
    pw_idx: torch.Tensor  # (S, max_len) int64
    pw_len: torch.Tensor  # (S,) int64
    starts: torch.Tensor  # (S,) int64, global slice index per frame
    episode_lengths: List[int]
    passwords: List[str]
    names: List[str]


class ButtonSliceDataset:
    """Wraps :class:`ButtonH5Dataset` with shard-based slice iteration."""

    def __init__(
        self,
        h5_path,
        allowed_passwords=None,
        cameras=BUTTON_CAMERAS,
        image_size=(224, 224),
        action_window=12,
        max_len=6,
        shard_size=16,
        seed=0,
        max_episodes=0,
        h5_kwargs=None,
        allowed_episodes=None,
    ):
        self.image_size = tuple(int(x) for x in image_size)
        self.action_window = int(action_window)
        self.shard_size = int(shard_size)
        self.cameras = tuple(cameras)
        self.rng = random.Random(seed)

        self.base = ButtonH5Dataset(
            h5_path,
            cameras=self.cameras,
            image_size=self.image_size,
            max_len=max_len,
            allowed_passwords=allowed_passwords,
            max_episodes=max_episodes,
            allowed_episodes=allowed_episodes,
        )

        self.num_episodes = len(self.base)
        self.num_slices = sum(self.base.get_seq_length(i) for i in range(self.num_episodes))
        # per-episode slice start offsets in the concatenated slice space
        self.ep_slice_start = []
        total = 0
        for i in range(self.num_episodes):
            self.ep_slice_start.append(total)
            total += self.base.get_seq_length(i)
        assert total == self.num_slices

        # Pre-load light metadata/actions once (no images) so epoch iteration
        # does not re-open the HDF5 and re-read actions every epoch.
        self.meta = []
        self.name_to_idx = {}
        import h5py
        with h5py.File(self.base.h5_path, "r") as f:
            for idx, (name, password, length) in enumerate(self.base.episodes):
                g = f[name]
                action = torch.from_numpy(
                    np.asarray(g["action"][()]).astype(np.float32)
                )
                pw_idx, pw_len = self.base.encode_password(password, self.base.max_len)
                pw_idx = pw_idx.unsqueeze(0).expand(length, -1).contiguous()
                pw_len_t = torch.full((length,), pw_len, dtype=torch.long)
                starts = torch.arange(
                    self.ep_slice_start[idx], self.ep_slice_start[idx] + length,
                    dtype=torch.long,
                )
                self.meta.append({
                    "name": name,
                    "password": password,
                    "length": length,
                    "action": action,
                    "pw_idx": pw_idx,
                    "pw_len": pw_len_t,
                    "starts": starts,
                })
                self.name_to_idx[name] = idx

    def __len__(self):
        return self.num_slices

    # ------------------------------------------------------------------ #

    def load_episode(self, idx: int) -> EpisodeShard:
        """Load one episode (raw uint8 frames, no resize) from the HDF5."""
        name, password, length = self.base.episodes[idx]
        obs_raw = []
        import h5py

        with h5py.File(self.base.h5_path, "r") as f:
            g = f[name]
            for cam in self.cameras:
                if cam == "agentview":
                    key = "image"
                elif cam == "robot0_eye_in_hand":
                    key = "wrist_image"
                else:
                    key = f"{cam}_image"
                if key not in g:
                    key2 = f"{cam}_image"
                    if key2 not in g:
                        raise KeyError(f"camera {cam}: no {key!r} or {key2!r} in group {name}")
                    key = key2
                arr = np.asarray(g[key][()])
                obs_raw.append(torch.from_numpy(arr.copy()))
            action = torch.from_numpy(np.asarray(g["action"][()]).astype(np.float32))

        # obs_raw entries are (T, H, W, 3) uint8 -> stack into (T, V, H, W, 3)
        obs = torch.stack([t.permute(0, 3, 1, 2) for t in obs_raw], dim=1)

        pw_idx, pw_len = self.base.encode_password(password, self.base.max_len)
        pw_idx = pw_idx.unsqueeze(0).expand(length, -1).contiguous()
        pw_len_t = torch.full((length,), pw_len, dtype=torch.long)

        starts = torch.arange(
            self.ep_slice_start[idx], self.ep_slice_start[idx] + length, dtype=torch.long
        )
        return EpisodeShard(
            obs=obs,
            action=action,
            pw_idx=pw_idx,
            pw_len=pw_len_t,
            starts=starts,
            episode_lengths=[length],
            passwords=[password],
            names=[name],
        )

    def iter_shards(self, shuffle=True) -> Iterator[EpisodeShard]:
        """Yield concatenated episode shards.

        Concatenation happens along the frame axis so a single batch tensor
        index can address any slice in the shard.
        """
        order = list(range(self.num_episodes))
        if shuffle:
            self.rng.shuffle(order)
        for i in range(0, self.num_episodes, self.shard_size):
            idxs = order[i : i + self.shard_size]
            shards = [self.load_episode(j) for j in idxs]
            obs = torch.cat([s.obs for s in shards], dim=0)
            action = torch.cat([s.action for s in shards], dim=0)
            pw_idx = torch.cat([s.pw_idx for s in shards], dim=0)
            pw_len = torch.cat([s.pw_len for s in shards], dim=0)
            starts = torch.cat([s.starts for s in shards], dim=0)
            yield EpisodeShard(
                obs=obs,
                action=action,
                pw_idx=pw_idx,
                pw_len=pw_len,
                starts=starts,
                episode_lengths=[l for s in shards for l in s.episode_lengths],
                passwords=[p for s in shards for p in s.passwords],
                names=[n for s in shards for n in s.names],
            )

    def iter_light_shards(self, shuffle=True) -> Iterator[EpisodeShard]:
        """Yield concatenated shards WITHOUT obs images.

        Actions/password metadata come from the pre-loaded ``self.meta``.
        ``obs`` is None; the caller is expected to provide embeddings (e.g.
        from a disk cache) instead of calling embed_shard_frames on raw obs.
        """
        order = list(range(self.num_episodes))
        if shuffle:
            self.rng.shuffle(order)
        for i in range(0, self.num_episodes, self.shard_size):
            idxs = order[i : i + self.shard_size]
            metas = [self.meta[j] for j in idxs]
            action = torch.cat([m["action"] for m in metas], dim=0)
            pw_idx = torch.cat([m["pw_idx"] for m in metas], dim=0)
            pw_len = torch.cat([m["pw_len"] for m in metas], dim=0)
            starts = torch.cat([m["starts"] for m in metas], dim=0)
            yield EpisodeShard(
                obs=None,
                action=action,
                pw_idx=pw_idx,
                pw_len=pw_len,
                starts=starts,
                episode_lengths=[m["length"] for m in metas],
                passwords=[m["password"] for m in metas],
                names=[m["name"] for m in metas],
            )

    def build_chunks(self, shard: EpisodeShard) -> torch.Tensor:
        """Precompute per-slice action chunks for the shard: (n_slices, W, A).

        Chunks never cross episode boundaries: a chunk that would run past the
        episode end is padded by repeating the episode's last action.
        """
        A = self.action_window
        chunks = []
        pos = 0
        for T in shard.episode_lengths:
            idx = torch.arange(A).unsqueeze(0) + torch.arange(T).unsqueeze(1)
            idx = idx.clamp(max=T - 1) + pos  # repeat-end padding
            chunks.append(shard.action[idx])
            pos += T
        return torch.cat(chunks, dim=0)

    def slice_obs(self, shard: EpisodeShard, idx: torch.Tensor, device) -> torch.Tensor:
        """Return float obs in [0,1] at slice indices, resized on the GPU.

        idx: (B,) int64 into the shard's frame axis.
        Returns: (B, 1, V, 3, H, W) float32 on ``device``.
        """
        obs = shard.obs[idx]  # (B, V, 3, H, W) uint8
        obs = obs.to(device, non_blocking=True).float().div_(255.0)
        if obs.shape[-2:] != self.image_size:
            B, V, C, H, W = obs.shape
            obs = obs.reshape(B * V, C, H, W)
            obs = F.interpolate(
                obs, size=self.image_size, mode="bilinear", align_corners=False
            )
            obs = obs.reshape(B, V, C, *self.image_size)
        return obs.unsqueeze(1)  # (B, 1, V, 3, H, W)

    def num_slices_in_shard(self, shard: EpisodeShard) -> int:
        return shard.starts.shape[0]

    # ------------------------------------------------------------------ #
    # TTT sequence windows (B2/B3)

    def build_sequence_windows(self, shard: EpisodeShard, T: int, W: int,
                               ep_counts=None):
        """Build sequence windows for TTT chunked training.

        For every episode and start s in [0, L):
          obs indices: s..s+T-1 (repeat-end padded past the end)
          actions:     s..s+T+W-2 (repeat-end padded)
          pw tokens:   the episode's FULL password, constant for every window
                       of that episode (dynamic suffix conditioning is
                       forbidden: it leaks ground-truth progress)
          counts:      per obs step press counts when labels are available

        Returns (obs_idx (S, T), act (S, T+W-1, 7), pw_idx (S, 6),
                 counts (S, T) or None, valid (S, T)). All index into the
                 shard's concatenated frame axis.
        """
        obs_idx_list, act_list, pw_list, count_list, valid_list = [], [], [], [], []
        pos = 0
        for ep_i, L in enumerate(shard.episode_lengths):
            starts = torch.arange(L)
            obs_i = starts[:, None] + torch.arange(T)[None, :]
            valid_i = obs_i < L  # padding mask for repeat-end windows
            obs_i = obs_i.clamp(max=L - 1)
            act_i = (starts[:, None] + torch.arange(T + W - 1)[None, :]).clamp(max=L - 1)
            obs_idx_list.append(obs_i + pos)
            act_list.append(shard.action[act_i + pos])
            valid_list.append(valid_i)
            pw = shard.pw_idx[pos]  # (6,) full password of this episode
            pw_list.append(pw.expand(L, -1).contiguous())
            if ep_counts is not None and ep_counts[ep_i] is not None:
                count_list.append(ep_counts[ep_i][obs_i])  # (L, T)
            else:
                count_list.append(
                    torch.full((L, T), -1, dtype=torch.long)
                )  # unlabeled -> masked out
            pos += L

        obs_idx = torch.cat(obs_idx_list, dim=0)
        act = torch.cat(act_list, dim=0)
        pw_idx = torch.cat(pw_list, dim=0)
        counts = torch.cat(count_list, dim=0) if ep_counts is not None else None
        valid = torch.cat(valid_list, dim=0)
        return obs_idx, act, pw_idx, counts, valid
