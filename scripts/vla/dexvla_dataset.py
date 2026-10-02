# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dex-VLA PyTorch Dataset & Action Chunking Loader.

Prepares multi-modal expert demonstration trajectories for training the Dex-VLA
Diffusion Action Expert:
  - Visual Depth: 3 camera streams (Head + Right Wrist + Left Wrist)
  - Proprioception: Dual Shadow Hand 48-dim joint pos & vel
  - Tactile: 10 fingertip 3D contact force vectors (30-dim)
  - Natural Language Instructions: Raw strings & tokenized representations
  - Target Actions: Sliding action chunk horizon H (e.g. H=16 steps x 40-DoF)
"""

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DexVLADataset(Dataset):
    """PyTorch Dataset for Dex-VLA action-chunking training."""

    def __init__(
        self,
        dataset_dir: str = "datasets/vla_shadow_hand",
        chunk_size: int = 16,
        obs_horizon: int = 1,
        normalize: bool = True,
        stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ):
        """
        Args:
            dataset_dir: Directory containing .h5 episode files.
            chunk_size: Action chunk horizon H (number of future action steps to predict).
            obs_horizon: Number of past observation frames to condition on (default 1).
            normalize: Whether to normalize actions and state vectors.
            stats: Pre-computed dataset normalization stats (min/max or mean/std).
        """
        super().__init__()
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.chunk_size = chunk_size
        self.obs_horizon = obs_horizon
        self.normalize = normalize

        self.episode_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.h5")))
        if not self.episode_files:
            # Check for .npz
            self.episode_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.npz")))

        if not self.episode_files:
            raise FileNotFoundError(f"No .h5 or .npz episode files found in {self.dataset_dir}")

        print(f"[DexVLADataset] Found {len(self.episode_files)} demonstration episodes.")

        # Index valid (episode_idx, step_idx) pairs
        self.indices: List[Tuple[int, int]] = []
        self._episode_lengths: List[int] = []

        # Read metadata and calculate sample indices
        for ep_idx, ep_file in enumerate(self.episode_files):
            if ep_file.endswith(".h5"):
                with h5py.File(ep_file, "r") as f:
                    ep_len = len(f["actions_dual_hand_40d"])
            else:
                data = np.load(ep_file, allow_pickle=True)
                ep_len = len(data["actions_dual_hand_40d"])

            self._episode_lengths.append(ep_len)
            # Focus on active manipulation window (steps 0 to 200: throw, flight, catch, hold)
            active_len = min(ep_len, 200)
            for t in range(active_len):
                self.indices.append((ep_idx, t))

        print(f"[DexVLADataset] Total transition samples: {len(self.indices)} (Chunk Horizon: {self.chunk_size})")

        # Compute or assign normalization statistics
        if stats is not None:
            self.stats = stats
        else:
            self.stats = self._compute_statistics()

    def _compute_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute mean and std for continuous actions, joint states, and forces across dataset."""
        all_actions = []
        all_proprio = []
        all_forces = []

        print("[DexVLADataset] Computing dataset normalization statistics...")
        # Sample subset for fast statistics computation
        sample_files = self.episode_files[: min(len(self.episode_files), 50)]
        for ep_file in sample_files:
            if ep_file.endswith(".h5"):
                with h5py.File(ep_file, "r") as f:
                    all_actions.append(np.array(f["actions_dual_hand_40d"], dtype=np.float32))
                    
                    r_pos = np.array(f["right_hand_joint_pos"], dtype=np.float32)
                    l_pos = np.array(f["left_hand_joint_pos"], dtype=np.float32)
                    r_vel = np.array(f["right_hand_joint_vel"], dtype=np.float32)
                    l_vel = np.array(f["left_hand_joint_vel"], dtype=np.float32)
                    proprio = np.concatenate([r_pos, l_pos, r_vel, l_vel], axis=-1)
                    all_proprio.append(proprio)

                    r_f = np.array(f["right_fingertip_forces"], dtype=np.float32).reshape(len(r_pos), -1)
                    l_f = np.array(f["left_fingertip_forces"], dtype=np.float32).reshape(len(l_pos), -1)
                    forces = np.concatenate([r_f, l_f], axis=-1)
                    all_forces.append(forces)

        actions_concat = np.concatenate(all_actions, axis=0)
        proprio_concat = np.concatenate(all_proprio, axis=0)
        forces_concat = np.concatenate(all_forces, axis=0)

        stats = {
            "actions": {
                "mean": np.mean(actions_concat, axis=0),
                "std": np.clip(np.std(actions_concat, axis=0), 1e-4, None),
                "min": np.min(actions_concat, axis=0),
                "max": np.max(actions_concat, axis=0),
            },
            "proprio": {
                "mean": np.mean(proprio_concat, axis=0),
                "std": np.clip(np.std(proprio_concat, axis=0), 1e-4, None),
            },
            "forces": {
                "mean": np.mean(forces_concat, axis=0),
                "std": np.clip(np.std(forces_concat, axis=0), 1e-4, None),
            },
        }
        return stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.indices[idx]
        ep_file = self.episode_files[ep_idx]
        ep_len = self._episode_lengths[ep_idx]

        if ep_file.endswith(".h5"):
            with h5py.File(ep_file, "r") as f:
                head_depth = np.array(f["overhead_head_depth"][t], dtype=np.float32)  # (64, 64)
                r_wrist_depth = np.array(f["wrist_cam_right_depth"][t], dtype=np.float32)
                l_wrist_depth = np.array(f["wrist_cam_left_depth"][t], dtype=np.float32)

                r_pos = np.array(f["right_hand_joint_pos"][t], dtype=np.float32)
                l_pos = np.array(f["left_hand_joint_pos"][t], dtype=np.float32)
                r_vel = np.array(f["right_hand_joint_vel"][t], dtype=np.float32)
                l_vel = np.array(f["left_hand_joint_vel"][t], dtype=np.float32)

                r_force = np.array(f["right_fingertip_forces"][t], dtype=np.float32).flatten()  # 15-dim
                l_force = np.array(f["left_fingertip_forces"][t], dtype=np.float32).flatten()  # 15-dim

                # Read action chunk [t : t + chunk_size]
                act_end = min(t + self.chunk_size, ep_len)
                action_chunk = np.array(f["actions_dual_hand_40d"][t:act_end], dtype=np.float32)
                prompt = str(f.attrs.get("language_instruction", "Throw and catch the ball."))
        else:
            data = np.load(ep_file, allow_pickle=True)
            head_depth = np.array(data["overhead_head_depth"][t], dtype=np.float32)
            r_wrist_depth = np.array(data["wrist_cam_right_depth"][t], dtype=np.float32)
            l_wrist_depth = np.array(data["wrist_cam_left_depth"][t], dtype=np.float32)

            r_pos = np.array(data["right_hand_joint_pos"][t], dtype=np.float32)
            l_pos = np.array(data["left_hand_joint_pos"][t], dtype=np.float32)
            r_vel = np.array(data["right_hand_joint_vel"][t], dtype=np.float32)
            l_vel = np.array(data["left_hand_joint_vel"][t], dtype=np.float32)

            r_force = np.array(data["right_fingertip_forces"][t], dtype=np.float32).flatten()
            l_force = np.array(data["left_fingertip_forces"][t], dtype=np.float32).flatten()

            act_end = min(t + self.chunk_size, ep_len)
            action_chunk = np.array(data["actions_dual_hand_40d"][t:act_end], dtype=np.float32)
            prompt = str(data.get("language_instruction", "Throw and catch the ball."))

        # Squeeze depth cameras to clean 2D (64, 64)
        head_depth = np.squeeze(head_depth)
        r_wrist_depth = np.squeeze(r_wrist_depth)
        l_wrist_depth = np.squeeze(l_wrist_depth)

        # Clip depth and normalize to [0, 1]
        head_depth = np.clip(np.nan_to_num(head_depth, nan=2.0, posinf=2.0), 0.1, 2.0) / 2.0
        r_wrist_depth = np.clip(np.nan_to_num(r_wrist_depth, nan=1.5, posinf=1.5), 0.05, 1.5) / 1.5
        l_wrist_depth = np.clip(np.nan_to_num(l_wrist_depth, nan=1.5, posinf=1.5), 0.05, 1.5) / 1.5

        # Multi-camera tensor: (3, 64, 64)
        images = np.stack([head_depth, r_wrist_depth, l_wrist_depth], axis=0)

        # Proprioception vector (joint pos + vel): 48 + 48 = 96-dim
        proprio = np.concatenate([r_pos, l_pos, r_vel, l_vel], axis=-1)
        # Tactile vector: 15 + 15 = 30-dim
        tactile = np.concatenate([r_force, l_force], axis=-1)

        # Pad action chunk if near end of episode
        if len(action_chunk) < self.chunk_size:
            pad_len = self.chunk_size - len(action_chunk)
            last_act = action_chunk[-1:]
            pad_acts = np.repeat(last_act, pad_len, axis=0)
            action_chunk = np.concatenate([action_chunk, pad_acts], axis=0)

        # Normalize if requested
        if self.normalize:
            action_chunk = np.clip(action_chunk, -1.0, 1.0)
            proprio = (proprio - self.stats["proprio"]["mean"]) / self.stats["proprio"]["std"]
            tactile = (tactile - self.stats["forces"]["mean"]) / self.stats["forces"]["std"]

        return {
            "image": torch.from_numpy(images).float(),                # (3, 64, 64)
            "proprio": torch.from_numpy(proprio).float(),            # (96,)
            "tactile": torch.from_numpy(tactile).float(),            # (30,)
            "action_chunk": torch.from_numpy(action_chunk).float(),  # (chunk_size, 40)
            "prompt": prompt,
        }


def get_dexvla_dataloader(
    dataset_dir: str = "datasets/vla_shadow_hand",
    batch_size: int = 64,
    chunk_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, DexVLADataset]:
    """Helper to construct PyTorch DataLoader for Dex-VLA."""
    dataset = DexVLADataset(dataset_dir=dataset_dir, chunk_size=chunk_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset

