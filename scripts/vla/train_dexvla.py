# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training Script for Dex-VLA Diffusion Action Expert.

Supports single-GPU and Cloud GPU (A100/H100/RTX 4090/L40S) training:
  - Automatic Mixed Precision (AMP - fp16 / bf16)
  - Train / Validation Split
  - Cosine Annealing Learning Rate Scheduler with Linear Warmup
  - Checkpointing (Best Validation Loss & Periodic saves)
  - Weights & Biases (WandB) & Terminal telemetry logging
"""

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add repository root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.vla.dexvla_dataset import DexVLADataset
from source.dex_vla.model import DexVLAPolicy


def save_checkpoint(
    save_path: str,
    model: DexVLAPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    stats: Dict,
    args: argparse.Namespace,
):
    """Save model checkpoint, optimizer state, and dataset normalization statistics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy stats to serializable python floats/lists
    serializable_stats = {}
    for k, sub_dict in stats.items():
        serializable_stats[k] = {}
        for sub_k, val in sub_dict.items():
            if isinstance(val, np.ndarray):
                serializable_stats[k][sub_k] = val.tolist()
            else:
                serializable_stats[k][sub_k] = val

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "dataset_stats": serializable_stats,
        "config": vars(args),
    }
    torch.save(checkpoint_data, save_path)
    
    # Also save separate json metadata for easy inspection
    stats_json_path = os.path.join(os.path.dirname(save_path), "dataset_stats.json")
    with open(stats_json_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Dex-VLA Policy on Cloud / Local GPU.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/vla_shadow_hand", help="Path to demonstration dataset folder.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/dexvla", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--chunk_size", type=int, default=16, help="Action chunk prediction horizon H.")
    parser.add_argument("--embed_dim", type=int, default=256, help="Transformer embedding dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of DiT Transformer blocks.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of diffusion training steps.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of data reserved for validation.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["none", "fp16", "bf16"], help="Automatic Mixed Precision mode.")
    parser.add_argument("--save_freq", type=int, default=10, help="Epoch frequency to save intermediate checkpoints.")
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="dex-vla-shadow-hand", help="WandB project name.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" " * 28 + "DEX-VLA TRAINING PIPELINE")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset Directory: {os.path.abspath(args.dataset_dir)}")
    print(f"Output Directory:  {os.path.abspath(args.output_dir)}")
    print(f"Hyperparameters:   Epochs={args.epochs} | Batch={args.batch_size} | LR={args.lr} | ChunkSize={args.chunk_size}")
    print("=" * 80)

    # Initialize WandB if requested
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # 1. Load Dataset
    full_dataset = DexVLADataset(dataset_dir=args.dataset_dir, chunk_size=args.chunk_size)
    total_samples = len(full_dataset)
    val_size = max(int(total_samples * args.val_ratio), 1)
    train_size = total_samples - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[INFO] Train Samples: {train_size} ({len(train_loader)} batches) | Val Samples: {val_size} ({len(val_loader)} batches)")

    # 2. Instantiate Model
    model = DexVLAPolicy(
        action_dim=40,
        chunk_size=args.chunk_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 3. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP Scaler
    amp_enabled = args.mixed_precision != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        print(f"[INFO] Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # 4. Main Training Loop
    print("\n[INFO] Starting training...\n")
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            tactile = batch["tactile"].to(device, non_blocking=True)
            prompts = batch["prompt"]
            actions = batch["action_chunk"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                loss = model.compute_loss(
                    images=images,
                    proprio=proprio,
                    tactile=tactile,
                    prompts=prompts,
                    target_action_chunk=actions,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.5f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        lr_scheduler.step()
        avg_train_loss = train_loss_accum / max(train_steps, 1)

        # Validation Loop
        model.eval()
        val_loss_accum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                proprio = batch["proprio"].to(device, non_blocking=True)
                tactile = batch["tactile"].to(device, non_blocking=True)
                prompts = batch["prompt"]
                actions = batch["action_chunk"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    val_loss = model.compute_loss(
                        images=images,
                        proprio=proprio,
                        tactile=tactile,
                        prompts=prompts,
                        target_action_chunk=actions,
                    )
                val_loss_accum += val_loss.item()
                val_steps += 1

        avg_val_loss = val_loss_accum / max(val_steps, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"LR: {current_lr:.2e}"
        )

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
            })

        # Checkpoint: Save Latest
        latest_path = os.path.join(args.output_dir, "dexvla_latest.pth")
        save_checkpoint(latest_path, model, optimizer, epoch, avg_val_loss, full_dataset.stats, args)

        # Checkpoint: Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.output_dir, "dexvla_best.pth")
            save_checkpoint(best_path, model, optimizer, epoch, avg_val_loss, full_dataset.stats, args)
            print(f"  ⭐ New best validation loss: {best_val_loss:.5f} -> Saved {best_path}")

        # Periodic Checkpoint
        if epoch % args.save_freq == 0:
            epoch_path = os.path.join(args.output_dir, f"dexvla_epoch_{epoch:03d}.pth")
            save_checkpoint(epoch_path, model, optimizer, epoch, avg_val_loss, full_dataset.stats, args)

    total_time = time.time() - training_start
    print("\n" + "=" * 80)
    print(f"🎉 DEX-VLA TRAINING COMPLETE!")
    print(f"   - Total Training Time: {total_time / 60:.1f} minutes")
    print(f"   - Best Val Loss:       {best_val_loss:.5f}")
    print(f"   - Checkpoints Saved:   {os.path.abspath(args.output_dir)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
