# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record expert demonstration rollouts for Dexterous VLA (Vision-Language-Action) fine-tuning.

This script executes the trained PPO cooperative throwing & catching policy in parallel Isaac Lab
environments and records full multi-modal trajectory datasets:
  - Multi-camera Depth / Visual feeds (Overhead Head + Left Wrist + Right Wrist)
  - Proprioception (Joint positions, velocities for both hands)
  - Tactile fingertip contact force vectors (3D forces across 10 fingertips)
  - Continuous 40-DoF dual-hand action trajectories
  - Natural language task instructions
  - Success / Completion verification metadata

Output Format:
  - Saved as HDF5 (.h5) or Compressed NumPy (.npz) per episode.
  - Ready for LeRobot, OpenPI (pi0), Octo, and OpenVLA fine-tuning pipelines.
"""

import argparse
import json
import math
import os
import random
import sys
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record expert demonstration rollouts for Dexterous VLA models.")
parser.add_argument("--num_episodes", type=int, default=50, help="Total number of successful episodes to record.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Catching-Ball-Rl-Direct-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PPO checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", default=False, help="Use the latest trained checkpoint.")
parser.add_argument("--output_dir", type=str, default="datasets/vla_shadow_hand", help="Directory to save the dataset.")
parser.add_argument("--only_successful", action="store_true", default=True, help="Only save episodes that achieved successful catch.")
parser.add_argument("--format", type=str, default="h5", choices=["h5", "npz"], help="Dataset file format: 'h5' (HDF5) or 'npz'.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for rollout environment.")

# Append AppLauncher cli args (headless, livestream, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras for VLA recording
args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation app is launched, now import simulation & ML modules."""

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Catching_ball_RL  # noqa: F401
import isaaclab_tasks  # noqa: F401

# Diverse Language Instruction Prompt Generator for Dexterous VLA training
# Supports hands in general, robotic hands, shadow hands, bimanual coordination, and various objects
def generate_diverse_language_prompt() -> str:
    """Generate diverse, highly varied natural language prompts for VLA conditioning."""
    right_hands = [
        "the right hand", "right hand", "the right robotic hand", "right robot hand",
        "the right shadow hand", "the throwing hand", "the right dexterous hand",
    ]
    left_hands = [
        "the left hand", "left hand", "the left robotic hand", "left robot hand",
        "the left shadow hand", "the receiving hand", "the left dexterous hand", "the opposite hand",
    ]
    throw_verbs = ["throw", "pass", "toss", "lob", "launch", "pitch", "fling"]
    throw_verbs_3rd = ["throws", "passes", "tosses", "lobs", "launches", "pitches", "flings"]
    
    objects = [
        "the ball", "the tennis ball", "the yellow ball", "the small ball",
        "the sphere", "the ball object", "the yellow tennis ball",
    ]
    catch_phrases = [
        "catch and hold it", "catch and secure it", "grasp it with fingertips",
        "catch it firmly", "secure it in the palm", "catch and stabilize it",
        "grasp and hold it", "cup and secure the ball",
    ]
    catch_phrases_3rd = [
        "catches and holds it", "catches and secures it", "grasps it with fingertips",
        "catches it firmly", "secures it in the palm", "catches and stabilizes it",
        "grasps and holds it", "cups and secures the ball",
    ]
    gaps = [
        "across the gap", "over the table gap", "across the 1-meter distance",
        "to the other side", "across the table", "",
    ]

    r_hand = random.choice(right_hands)
    l_hand = random.choice(left_hands)
    t_verb = random.choice(throw_verbs)
    t_verb_3rd = random.choice(throw_verbs_3rd)
    obj = random.choice(objects)
    c_phrase = random.choice(catch_phrases)
    c_phrase_3rd = random.choice(catch_phrases_3rd)
    gap = random.choice(gaps)
    gap_str = f" {gap}" if gap else ""

    templates = [
        # Imperative / Command style
        f"{t_verb.capitalize()} {obj} from {r_hand} to {l_hand}{gap_str}, and {c_phrase}.",
        f"Using {r_hand}, {t_verb} {obj}{gap_str} so that {l_hand} can {c_phrase}.",
        f"{t_verb.capitalize()} {obj}{gap_str} to {l_hand} and {c_phrase}.",
        f"Coordinate {r_hand} and {l_hand} to {t_verb} and {c_phrase} {obj}.",
        f"Pass {obj} from right to left hand and {c_phrase}.",
        f"Perform a bimanual hand-to-hand pass of {obj} and {c_phrase}.",
        # Declarative / Descriptive style
        f"{r_hand.capitalize()} {t_verb_3rd} {obj}{gap_str} to {l_hand}, which {c_phrase_3rd}.",
        f"{r_hand.capitalize()} {t_verb_3rd} {obj}, and {l_hand} {c_phrase_3rd}.",
        f"Bimanual cooperative catch: {r_hand} {t_verb_3rd} {obj}{gap_str}, {l_hand} {c_phrase_3rd}.",
        # General / Robot-agnostic style
        f"Throw {obj} with the right hand and catch it with the left hand.",
        f"Catch {obj} with the left hand after it is tossed from the right hand.",
        f"Receive and hold {obj} thrown across the gap.",
    ]

    return random.choice(templates)


def save_episode_data(file_path: str, data: dict, file_format: str):
    """Save an episode trajectory in HDF5 or NPZ format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_format == "h5":
        try:
            import h5py
            with h5py.File(file_path, "w") as f:
                for k, v in data.items():
                    if isinstance(v, str):
                        f.attrs[k] = v
                    elif isinstance(v, (int, float, bool)):
                        f.attrs[k] = v
                    elif isinstance(v, np.ndarray):
                        f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                    elif isinstance(v, torch.Tensor):
                        f.create_dataset(k, data=v.cpu().numpy(), compression="gzip", compression_opts=4)
            return
        except ImportError:
            print("[WARN] h5py not found, falling back to compressed .npz format.")
    
    # NPZ fallback / native numpy
    npz_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            npz_data[k] = v.cpu().numpy()
        elif isinstance(v, (str, int, float, bool)):
            npz_data[k] = np.array(v)
        else:
            npz_data[k] = np.array(v)
    
    npz_path = file_path if file_path.endswith(".npz") else file_path.replace(".h5", ".npz")
    np.savez_compressed(npz_path, **npz_data)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    """Main VLA demonstration rollout collection loop."""
    print("=" * 80)
    print(" " * 20 + "DEXTEROUS VLA ROLLOUT RECORDER")
    print("=" * 80)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    # Locate checkpoint
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO] Using Policy Checkpoint: {resume_path}")
    print(f"[INFO] Target Output Directory: {os.path.abspath(args_cli.output_dir)}")
    print(f"[INFO] Recording {args_cli.num_episodes} demonstration episodes across {args_cli.num_envs} envs...")

    # Create Isaac Lab environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env: DirectMARLEnv = env.unwrapped

    # Convert MARL to single-agent for RL-Games
    env_wrapped = multi_agent_to_single_agent(env)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    rl_env = RlGamesVecEnvWrapper(env_wrapped, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # Register the environment to rl-games registry
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env})

    # Load RL-Games player
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    player.restore(resume_path)
    player.reset()
    player.is_deterministic = True

    # Rollout buffers for parallel environments
    num_envs = args_cli.num_envs
    env_buffers = [
        {
            "wrist_cam_right_depth": [],
            "wrist_cam_left_depth": [],
            "overhead_head_depth": [],
            "right_hand_joint_pos": [],
            "right_hand_joint_vel": [],
            "left_hand_joint_pos": [],
            "left_hand_joint_vel": [],
            "right_fingertip_forces": [],
            "left_fingertip_forces": [],
            "actions_right_hand": [],
            "actions_left_hand": [],
            "object_pos": [],
            "object_linvel": [],
            "rewards": [],
        }
        for _ in range(num_envs)
    ]

    saved_episodes = 0
    total_attempted_episodes = 0
    start_time = time.time()

    # Reset environment
    obs_dict = rl_env.reset()
    if isinstance(obs_dict, dict):
        obs = obs_dict["obs"]
    else:
        obs = obs_dict

    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    episode_rewards = torch.zeros(num_envs, device=rl_env.device)
    episode_lengths = torch.zeros(num_envs, device=rl_env.device, dtype=torch.long)

    print("\n[INFO] Starting recording rollouts...\n")

    while saved_episodes < args_cli.num_episodes:
        # Get policy action from RL-Games player
        with torch.inference_mode():
            obs_torch = player.obs_to_torch(obs)
            actions = player.get_action(obs_torch, is_deterministic=True)

        # Step environment
        next_obs_dict, rewards, dones, infos = rl_env.step(actions)
        if isinstance(next_obs_dict, dict):
            next_obs = next_obs_dict["obs"]
        else:
            next_obs = next_obs_dict

        # Collect step data for each environment
        with torch.no_grad():
            right_depth = raw_env.right_wrist_depth.cpu().numpy()  # (num_envs, 64, 64)
            left_depth = raw_env.left_wrist_depth.cpu().numpy()    # (num_envs, 64, 64)
            head_depth = raw_env.head_depth.cpu().numpy()          # (num_envs, 64, 64)

            right_joint_pos = raw_env.right_hand_dof_pos.cpu().numpy()
            right_joint_vel = raw_env.right_hand_dof_vel.cpu().numpy()
            left_joint_pos = raw_env.left_hand_dof_pos.cpu().numpy()
            left_joint_vel = raw_env.left_hand_dof_vel.cpu().numpy()

            right_forces = raw_env.right_fingertip_forces.cpu().numpy()  # (num_envs, 5, 3)
            left_forces = raw_env.left_fingertip_forces.cpu().numpy()    # (num_envs, 5, 3)

            act_right = raw_env.actions["right_hand"].cpu().numpy()
            act_left = raw_env.actions["left_hand"].cpu().numpy()

            obj_pos = raw_env.object_pos.cpu().numpy()
            obj_linvel = raw_env.object_linvel.cpu().numpy()

            step_rewards = rewards.cpu().numpy()

        for env_id in range(num_envs):
            # Append step data
            env_buffers[env_id]["wrist_cam_right_depth"].append(right_depth[env_id])
            env_buffers[env_id]["wrist_cam_left_depth"].append(left_depth[env_id])
            env_buffers[env_id]["overhead_head_depth"].append(head_depth[env_id])
            env_buffers[env_id]["right_hand_joint_pos"].append(right_joint_pos[env_id])
            env_buffers[env_id]["right_hand_joint_vel"].append(right_joint_vel[env_id])
            env_buffers[env_id]["left_hand_joint_pos"].append(left_joint_pos[env_id])
            env_buffers[env_id]["left_hand_joint_vel"].append(left_joint_vel[env_id])
            env_buffers[env_id]["right_fingertip_forces"].append(right_forces[env_id])
            env_buffers[env_id]["left_fingertip_forces"].append(left_forces[env_id])
            env_buffers[env_id]["actions_right_hand"].append(act_right[env_id])
            env_buffers[env_id]["actions_left_hand"].append(act_left[env_id])
            env_buffers[env_id]["object_pos"].append(obj_pos[env_id])
            env_buffers[env_id]["object_linvel"].append(obj_linvel[env_id])
            env_buffers[env_id]["rewards"].append(step_rewards[env_id])

            episode_rewards[env_id] += rewards[env_id]
            episode_lengths[env_id] += 1

            # Check if this environment finished an episode
            if dones[env_id]:
                total_attempted_episodes += 1
                ep_len = episode_lengths[env_id].item()
                ep_rew = episode_rewards[env_id].item()

                # Verify success criteria using the trajectory collected BEFORE auto-reset:
                obj_pos_traj = np.array(env_buffers[env_id]["object_pos"])  # (T, 3)
                goal_target = np.array([0.0, -0.665, 0.55], dtype=np.float32)
                dists_to_goal = np.linalg.norm(obj_pos_traj - goal_target, axis=-1)
                final_goal_dist = float(dists_to_goal[-1])
                min_goal_dist = float(np.min(dists_to_goal))

                # Check if left hand made contact in second half of trajectory
                left_forces_traj = np.array(env_buffers[env_id]["left_fingertip_forces"])  # (T, 5, 3)
                left_forces_mag = np.linalg.norm(left_forces_traj, axis=-1)  # (T, 5)
                late_contact = np.any(left_forces_mag[len(left_forces_mag) // 2 :] > 0.05)

                is_successful = (ep_len >= 280) and (final_goal_dist <= 0.15 or (min_goal_dist <= 0.10 and late_contact))

                if (not args_cli.only_successful) or is_successful:
                    if saved_episodes < args_cli.num_episodes:
                        saved_episodes += 1
                        ext = args_cli.format
                        ep_filename = f"episode_{saved_episodes:06d}.{ext}"
                        ep_filepath = os.path.join(args_cli.output_dir, ep_filename)

                        # Package numpy arrays
                        prompt = generate_diverse_language_prompt()
                        compiled_episode = {
                            "wrist_cam_right_depth": np.array(env_buffers[env_id]["wrist_cam_right_depth"], dtype=np.float32),
                            "wrist_cam_left_depth": np.array(env_buffers[env_id]["wrist_cam_left_depth"], dtype=np.float32),
                            "overhead_head_depth": np.array(env_buffers[env_id]["overhead_head_depth"], dtype=np.float32),
                            "right_hand_joint_pos": np.array(env_buffers[env_id]["right_hand_joint_pos"], dtype=np.float32),
                            "right_hand_joint_vel": np.array(env_buffers[env_id]["right_hand_joint_vel"], dtype=np.float32),
                            "left_hand_joint_pos": np.array(env_buffers[env_id]["left_hand_joint_pos"], dtype=np.float32),
                            "left_hand_joint_vel": np.array(env_buffers[env_id]["left_hand_joint_vel"], dtype=np.float32),
                            "right_fingertip_forces": np.array(env_buffers[env_id]["right_fingertip_forces"], dtype=np.float32),
                            "left_fingertip_forces": np.array(env_buffers[env_id]["left_fingertip_forces"], dtype=np.float32),
                            "actions_right_hand": np.array(env_buffers[env_id]["actions_right_hand"], dtype=np.float32),
                            "actions_left_hand": np.array(env_buffers[env_id]["actions_left_hand"], dtype=np.float32),
                            "actions_dual_hand_40d": np.concatenate(
                                [
                                    np.array(env_buffers[env_id]["actions_right_hand"], dtype=np.float32),
                                    np.array(env_buffers[env_id]["actions_left_hand"], dtype=np.float32),
                                ],
                                axis=-1,
                            ),
                            "object_pos": np.array(env_buffers[env_id]["object_pos"], dtype=np.float32),
                            "object_linvel": np.array(env_buffers[env_id]["object_linvel"], dtype=np.float32),
                            "rewards": np.array(env_buffers[env_id]["rewards"], dtype=np.float32),
                            "language_instruction": prompt,
                            "episode_length": ep_len,
                            "total_reward": ep_rew,
                            "success": is_successful,
                        }

                        save_episode_data(ep_filepath, compiled_episode, args_cli.format)
                        print(
                            f"[RECORDED {saved_episodes:04d}/{args_cli.num_episodes:04d}] "
                            f"Length: {ep_len:3d} steps | Reward: {ep_rew:6.2f} | "
                            f"Goal Dist: {final_goal_dist:.3f}m | "
                            f"Saved: {ep_filename}"
                        )

                # Reset environment buffer for next episode in this env slot
                for key in env_buffers[env_id]:
                    env_buffers[env_id][key] = []
                episode_rewards[env_id] = 0.0
                episode_lengths[env_id] = 0

        obs = next_obs

    elapsed = time.time() - start_time
    success_rate = (saved_episodes / max(total_attempted_episodes, 1)) * 100.0

    # Write dataset summary JSON
    meta_path = os.path.join(args_cli.output_dir, "dataset_info.json")
    dataset_metadata = {
        "dataset_name": "shadow_hand_catching_vla",
        "total_episodes": saved_episodes,
        "total_attempted_episodes": total_attempted_episodes,
        "success_rate_pct": success_rate,
        "recording_time_sec": round(elapsed, 2),
        "task": args_cli.task,
        "checkpoint": resume_path,
        "action_dim": 40,
        "camera_modalities": ["overhead_head_depth", "wrist_cam_right_depth", "wrist_cam_left_depth"],
        "tactile_modalities": ["right_fingertip_forces", "left_fingertip_forces"],
        "language_instruction_generator": "dynamic_combinatorial_synthesis",
        "format": args_cli.format,
    }

    with open(meta_path, "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    print("\n" + "=" * 80)
    print(f"🎉 DATASET COLLECTION COMPLETE!")
    print(f"   - Saved Episodes: {saved_episodes} / {total_attempted_episodes} ({success_rate:.1f}% success rate)")
    print(f"   - Output Location: {os.path.abspath(args_cli.output_dir)}")
    print(f"   - Metadata File: {meta_path}")
    print(f"   - Total Time: {elapsed:.1f}s")
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
