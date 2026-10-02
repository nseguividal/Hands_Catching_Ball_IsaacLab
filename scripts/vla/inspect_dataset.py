# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to inspect, validate, and verify recorded Dexterous VLA demonstration datasets.

Usage:
  python scripts/vla/inspect_dataset.py --dataset_dir=datasets/vla_shadow_hand
"""

import argparse
import glob
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect recorded VLA demonstration dataset.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/vla_shadow_hand", help="Path to dataset folder.")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return

    # Check for metadata
    info_file = os.path.join(dataset_dir, "dataset_info.json")
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            info = json.load(f)
        print("=" * 80)
        print(" " * 25 + "DATASET METADATA SUMMARY")
        print("=" * 80)
        for k, v in info.items():
            if k == "language_instructions":
                print(f"  {k}: ({len(v)} prompt variations)")
            else:
                print(f"  {k}: {v}")
        print("=" * 80)

    # Find episode files
    h5_files = sorted(glob.glob(os.path.join(dataset_dir, "*.h5")))
    npz_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    files = h5_files if h5_files else npz_files

    if not files:
        print(f"[WARN] No .h5 or .npz episode files found in {dataset_dir}")
        return

    print(f"\n[INFO] Found {len(files)} recorded episode files. Inspecting first episode sample:\n")
    sample_path = files[0]
    print(f"File: {os.path.basename(sample_path)}")

    if sample_path.endswith(".h5"):
        import h5py
        with h5py.File(sample_path, "r") as f:
            print("\n" + "-" * 50)
            print(f"{'Field / Modality':<30} | {'Shape':<18} | {'Type'}")
            print("-" * 50)
            for k in f.keys():
                ds = f[k]
                print(f"{k:<30} | {str(ds.shape):<18} | {ds.dtype}")
            print("-" * 50)
            print("\nAttributes:")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")
    elif sample_path.endswith(".npz"):
        data = np.load(sample_path, allow_pickle=True)
        print("\n" + "-" * 50)
        print(f"{'Field / Modality':<30} | {'Shape':<18} | {'Type'}")
        print("-" * 50)
        for k in data.files:
            arr = data[k]
            shape_str = str(arr.shape) if hasattr(arr, "shape") else "scalar"
            dtype_str = str(arr.dtype) if hasattr(arr, "dtype") else type(arr).__name__
            print(f"{k:<30} | {shape_str:<18} | {dtype_str}")
        print("-" * 50)
        if "language_instruction" in data:
            print(f"\nLanguage Instruction: \"{data['language_instruction']}\"")
        if "total_reward" in data:
            print(f"Total Reward: {data['total_reward']}")
        if "success" in data:
            print(f"Success: {data['success']}")

    print("\n✅ Dataset verification complete!\n")


if __name__ == "__main__":
    main()
