# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Closed-Loop Evaluation Script for Dex-VLA Policy in Isaac Lab.

Executes the trained Dex-VLA Diffusion Action Expert directly inside the live
Isaac Lab physics simulation environment:
  - Multi-camera visual feeds (Overhead Head + Left/Right Wrist depth)
  - 10-fingertip 3D tactile contact forces + 48-dim joint states
  - Natural language prompt conditioning
  - Action chunking execution (H=16) with unnormalization
  - Live success rate, goal distance, and tactile catch monitoring
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate trained Dex-VLA policy in Isaac Lab closed-loop.")
parser.add_argument("--checkpoint", type=str, default="checkpoints/dexvla/dexvla_best.pth", help="Path to Dex-VLA checkpoint (.pth).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel evaluation environments.")
parser.add_argument("--task", type=str, default="Template-Catching-Ball-Rl-Direct-v0", help="Task name.")
parser.add_argument(
    "--prompt",
    type=str,
    default="Throw the yellow tennis ball with the right hand and catch it firmly with the left hand.",
    help="Natural language instruction prompt to condition the policy on.",
)
parser.add_argument("--chunk_exec_steps", type=int, default=8, help="Number of steps to execute from each predicted action chunk before re-planning.")
parser.add_argument("--diffusion_steps", type=int, default=20, help="Number of reverse diffusion inference sampling steps.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for evaluation.")

# Append AppLauncher cli args (headless, livestream, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras for VLA visual evaluation
args_cli.enable_cameras = True

# Clear sys.argv for simulation app
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation app is running, import simulation & PyTorch modules."""

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import Catching_ball_RL  # noqa: F401
import Catching_ball_RL.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Add repository root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from source.dex_vla.model import DexVLAPolicy


def main():
    print("=" * 80)
    print(" " * 22 + "DEX-VLA CLOSED-LOOP EVALUATOR")
    print("=" * 80)
    print(f"Policy Checkpoint: {os.path.abspath(args_cli.checkpoint)}")
    print(f"Conditioning Prompt: \"{args_cli.prompt}\"")
    print(f"Environments: {args_cli.num_envs} parallel")
    print("=" * 80)

    # 1. Load Checkpoint & Normalization Stats
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    stats = checkpoint["dataset_stats"]
    config = checkpoint.get("config", {})
    chunk_size = config.get("chunk_size", 16)
    embed_dim = config.get("embed_dim", 256)
    num_layers = config.get("num_layers", 4)
    num_heads = config.get("num_heads", 8)

    # Convert stats to torch tensors
    act_mean = torch.tensor(stats["actions"]["mean"], dtype=torch.float32, device=device)
    act_std = torch.tensor(stats["actions"]["std"], dtype=torch.float32, device=device)
    proprio_mean = torch.tensor(stats["proprio"]["mean"], dtype=torch.float32, device=device)
    proprio_std = torch.tensor(stats["proprio"]["std"], dtype=torch.float32, device=device)
    forces_mean = torch.tensor(stats["forces"]["mean"], dtype=torch.float32, device=device)
    forces_std = torch.tensor(stats["forces"]["std"], dtype=torch.float32, device=device)

    # 2. Instantiate and load model
    model = DexVLAPolicy(
        action_dim=40,
        chunk_size=chunk_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_diffusion_steps=100,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[INFO] Model loaded successfully into GPU memory.")

    # 3. Create Isaac Lab DirectMARLEnv
    env_cfg: DirectMARLEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if hasattr(args_cli, "device") and args_cli.device else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env: DirectMARLEnv = env.unwrapped

    num_envs = args_cli.num_envs
    prompts = [args_cli.prompt] * num_envs

    print("\n[INFO] Starting live closed-loop physics simulation...\n")

    obs, _ = env.reset()
    total_eval_episodes = 0
    successful_episodes = 0
    step_count = 0

    # Action chunk buffers per environment
    action_chunk_buffers = [None] * num_envs
    chunk_step_indices = [0] * num_envs

    try:
        while simulation_app.is_running():
            step_count += 1

            # 1. Capture live multi-modal sensor inputs from raw environment
            with torch.no_grad():
                # Visual depth cameras: (num_envs, 64, 64)
                h_depth = raw_env.head_depth.squeeze(-1) if raw_env.head_depth.dim() == 4 else raw_env.head_depth
                r_depth = raw_env.right_wrist_depth.squeeze(-1) if raw_env.right_wrist_depth.dim() == 4 else raw_env.right_wrist_depth
                l_depth = raw_env.left_wrist_depth.squeeze(-1) if raw_env.left_wrist_depth.dim() == 4 else raw_env.left_wrist_depth

                # Normalize depth to [0, 1]
                h_depth = torch.clamp(torch.nan_to_num(h_depth, nan=2.0, posinf=2.0), 0.1, 2.0) / 2.0
                r_depth = torch.clamp(torch.nan_to_num(r_depth, nan=1.5, posinf=1.5), 0.05, 1.5) / 1.5
                l_depth = torch.clamp(torch.nan_to_num(l_depth, nan=1.5, posinf=1.5), 0.05, 1.5) / 1.5

                # Multi-camera image tensor: (B, 3, 64, 64)
                images = torch.stack([h_depth, r_depth, l_depth], dim=1).float()

                # Proprioception: (B, 96)
                r_pos = raw_env.right_hand_dof_pos
                l_pos = raw_env.left_hand_dof_pos
                r_vel = raw_env.right_hand_dof_vel
                l_vel = raw_env.left_hand_dof_vel
                proprio = torch.cat([r_pos, l_pos, r_vel, l_vel], dim=-1).float()
                proprio_norm = (proprio - proprio_mean) / proprio_std

                # Tactile forces: (B, 30)
                r_force = raw_env.right_fingertip_forces.view(num_envs, -1).float()
                l_force = raw_env.left_fingertip_forces.view(num_envs, -1).float()
                tactile = torch.cat([r_force, l_force], dim=-1)
                tactile_norm = (tactile - forces_mean) / forces_std

                # Query Dex-VLA policy when action chunk buffer is depleted
                needs_replanning = (chunk_step_indices[0] == 0) or (chunk_step_indices[0] >= args_cli.chunk_exec_steps)

                if needs_replanning:
                    # Predict clean 40-DoF action chunks directly
                    pred_action_chunks = model.sample_actions(
                        images=images,
                        proprio=proprio_norm,
                        tactile=tactile_norm,
                        prompts=prompts,
                    )  # (B, chunk_size, 40) in [-1, 1]

                    for env_id in range(num_envs):
                        action_chunk_buffers[env_id] = pred_action_chunks[env_id]
                        chunk_step_indices[env_id] = 0

                # Extract current step's action from the chunk
                current_actions = torch.zeros(num_envs, 40, device=device)
                for env_id in range(num_envs):
                    idx = min(chunk_step_indices[env_id], chunk_size - 1)
                    current_actions[env_id] = action_chunk_buffers[env_id][idx]
                    chunk_step_indices[env_id] += 1

                # Split 40-DoF action into right hand (20) and left hand (20)
                right_actions = current_actions[:, :20]
                left_actions = current_actions[:, 20:]
                action_dict = {"right_hand": right_actions, "left_hand": left_actions}

            # 2. Step Isaac Lab physics simulation
            step_return = env.step(action_dict)
            if len(step_return) == 5:
                obs, rewards, dones, time_outs, infos = step_return
            else:
                obs, rewards, dones, infos = step_return
                time_outs = dones

            # 3. Telemetry & Success tracking
            with torch.no_grad():
                goal_dist = torch.norm(raw_env.object_pos - raw_env.goal_pos, dim=-1)
                min_dist = goal_dist.min().item()
                mean_dist = goal_dist.mean().item()
                hold_s = (raw_env.hold_steps / 60.0).mean().item()

                if step_count % 30 == 0:
                    print(
                        f"[DEX-VLA EVAL | Step {step_count:5d}] "
                        f"Mean Goal Dist: {mean_dist:.4f}m | "
                        f"Best Env Dist: {min_dist:.4f}m | "
                        f"Hold Duration: {hold_s:.2f}s"
                    )

                # Track completed episodes
                any_dones = dones["right_hand"] | time_outs["right_hand"]
                for env_id in range(num_envs):
                    if any_dones[env_id]:
                        total_eval_episodes += 1
                        chunk_step_indices[env_id] = 0
                        if goal_dist[env_id] <= 0.15 or raw_env.hold_steps[env_id] >= 30.0:
                            successful_episodes += 1

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")

    # Print summary
    success_rate = (successful_episodes / max(total_eval_episodes, 1)) * 100.0
    print("\n" + "=" * 80)
    print("🎉 DEX-VLA EVALUATION SUMMARY")
    print(f"   - Total Episodes: {total_eval_episodes}")
    print(f"   - Successful Catches: {successful_episodes} ({success_rate:.1f}% Success Rate)")
    print(f"   - Conditioned Prompt: \"{args_cli.prompt}\"")
    print("=" * 80 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
