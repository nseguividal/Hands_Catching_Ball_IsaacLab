# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to visualize and render video from recorded Dexterous VLA episodes.

Usage:
  python scripts/vla/visualize_episode.py --episode=datasets/vla_shadow_hand/episode_000001.h5 --output=episode_01_replay.mp4
"""

import argparse
import os
import numpy as np


def normalize_depth(depth_frame: np.ndarray, min_val: float = 0.2, max_val: float = 2.0) -> np.ndarray:
    """Convert raw metric depth to 8-bit visual grayscale / colormap."""
    # Replace infs and NaNs
    clean_depth = np.nan_to_num(depth_frame, nan=max_val, posinf=max_val, neginf=min_val)
    clipped = np.clip(clean_depth, min_val, max_val)
    # Normalize to 0-255 (inverted so closer objects appear brighter)
    norm = ((max_val - clipped) / (max_val - min_val) * 255.0).astype(np.uint8)
    return norm


def main():
    parser = argparse.ArgumentParser(description="Render video from a recorded VLA episode.")
    parser.add_argument(
        "--episode",
        type=str,
        default="datasets/vla_shadow_hand/episode_000001.h5",
        help="Path to .h5 or .npz episode file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="episode_01_replay.mp4",
        help="Output video file path (.mp4 or .gif).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second.")
    parser.add_argument(
        "--camera",
        type=str,
        default="all",
        choices=["all", "overhead_head_depth", "wrist_cam_right_depth", "wrist_cam_left_depth"],
        help="Which camera view to export ('all' creates a multi-view grid).",
    )
    args = parser.parse_args()

    episode_path = os.path.abspath(args.episode)
    if not os.path.exists(episode_path):
        print(f"[ERROR] Episode file not found: {episode_path}")
        return

    # Load data
    language_prompt = ""
    if episode_path.endswith(".h5"):
        import h5py

        with h5py.File(episode_path, "r") as f:
            head_depth = np.array(f["overhead_head_depth"])
            right_depth = np.array(f["wrist_cam_right_depth"])
            left_depth = np.array(f["wrist_cam_left_depth"])
            language_prompt = f.attrs.get("language_instruction", "")
            total_reward = f.attrs.get("total_reward", 0.0)
    elif episode_path.endswith(".npz"):
        data = np.load(episode_path, allow_pickle=True)
        head_depth = data["overhead_head_depth"]
        right_depth = data["wrist_cam_right_depth"]
        left_depth = data["wrist_cam_left_depth"]
        language_prompt = str(data.get("language_instruction", ""))
        total_reward = float(data.get("total_reward", 0.0))
    else:
        print("[ERROR] Unsupported format. Must be .h5 or .npz")
        return

    num_frames = len(head_depth)
    print(f"[INFO] Loaded episode with {num_frames} frames.")
    print(f"[INFO] Language Instruction: \"{language_prompt}\"")

    # Try importing imageio or cv2 for video writing
    try:
        import imageio
        has_imageio = True
    except ImportError:
        has_imageio = False

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    # Squeeze extra dimensions if present (e.g. (T, 64, 64, 1) -> (T, 64, 64))
    head_depth = np.squeeze(head_depth)
    right_depth = np.squeeze(right_depth)
    left_depth = np.squeeze(left_depth)

    rendered_frames = []
    
    # Optional colormap with matplotlib if available
    try:
        import matplotlib.pyplot as plt
        colormap = plt.get_cmap("inferno")
    except Exception:
        colormap = None

    for t in range(num_frames):
        h_norm = normalize_depth(head_depth[t])
        r_norm = normalize_depth(right_depth[t])
        l_norm = normalize_depth(left_depth[t])

        if colormap is not None:
            h_rgb = (colormap(h_norm.astype(np.float32) / 255.0)[:, :, :3] * 255).astype(np.uint8)
            r_rgb = (colormap(r_norm.astype(np.float32) / 255.0)[:, :, :3] * 255).astype(np.uint8)
            l_rgb = (colormap(l_norm.astype(np.float32) / 255.0)[:, :, :3] * 255).astype(np.uint8)
        else:
            h_rgb = np.stack([h_norm] * 3, axis=-1)
            r_rgb = np.stack([r_norm] * 3, axis=-1)
            l_rgb = np.stack([l_norm] * 3, axis=-1)

        if args.camera == "overhead_head_depth":
            frame = h_rgb
        elif args.camera == "wrist_cam_right_depth":
            frame = r_rgb
        elif args.camera == "wrist_cam_left_depth":
            frame = l_rgb
        else:
            # Multi-view layout: Overhead on top, wrists below
            # Upscale 64x64 to 256x256 for clear visualization
            if has_cv2:
                h_large = cv2.resize(np.ascontiguousarray(h_rgb), (256, 256), interpolation=cv2.INTER_NEAREST)
                r_large = cv2.resize(np.ascontiguousarray(r_rgb), (128, 128), interpolation=cv2.INTER_NEAREST)
                l_large = cv2.resize(np.ascontiguousarray(l_rgb), (128, 128), interpolation=cv2.INTER_NEAREST)
                wrists = np.hstack([r_large, l_large])  # 128x256
                combined = np.vstack([h_large, wrists])  # 384x256
                frame = combined
            else:
                # Basic repeat upscale
                h_large = np.repeat(np.repeat(h_rgb, 4, axis=0), 4, axis=1)  # 256x256
                r_large = np.repeat(np.repeat(r_rgb, 2, axis=0), 2, axis=1)  # 128x128
                l_large = np.repeat(np.repeat(l_rgb, 2, axis=0), 2, axis=1)  # 128x128
                wrists = np.hstack([r_large, l_large])
                frame = np.vstack([h_large, wrists])

        rendered_frames.append(frame)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if output_path.endswith(".gif") or not has_cv2:
        if not output_path.endswith(".gif"):
            output_path = output_path.replace(".mp4", ".gif")
        import imageio
        imageio.mimsave(output_path, rendered_frames, fps=args.fps)
        print(f"\n🎉 Saved replay GIF to: {output_path}")
    else:
        # Save as MP4 with OpenCV
        h, w, _ = rendered_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, args.fps, (w, h))
        for f in rendered_frames:
            # OpenCV expects BGR
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"\n🎉 Saved replay video to: {output_path}")


if __name__ == "__main__":
    main()
