#!/usr/bin/env bash
# Script to bundle dataset + Dex-VLA code for upload to Cloud GPUs (RunPod, Lambda, AWS, GCP, Vast.ai)

set -e

BUNDLE_NAME="dexvla_cloud_package.tar.gz"

echo "========================================================"
echo " Packaging Dex-VLA Dataset & Training Code for Cloud"
echo "========================================================"

# Remove old bundle if exists
rm -f "${BUNDLE_NAME}"

# Create compressed archive containing only what is needed for training:
# 1. datasets/vla_shadow_hand/*.h5
# 2. scripts/vla/
# 3. source/dex_vla/
tar -czvf "${BUNDLE_NAME}" \
    datasets/vla_shadow_hand \
    scripts/vla/dexvla_dataset.py \
    scripts/vla/train_dexvla.py \
    scripts/vla/requirements_cloud.txt \
    source/dex_vla

BUNDLE_SIZE=$(du -h "${BUNDLE_NAME}" | cut -f1)

echo "========================================================"
echo "🎉 Package created successfully: ${BUNDLE_NAME} (${BUNDLE_SIZE})"
echo "========================================================"
echo ""
echo "How to use on your Cloud GPU instance:"
echo "1. Upload ${BUNDLE_NAME} to your cloud VM (via scp, rsync, Google Drive, or Web UI)"
echo "2. Extract: tar -xzvf ${BUNDLE_NAME}"
echo "3. Install requirements: pip install -r scripts/vla/requirements_cloud.txt"
echo "4. Run training: python scripts/vla/train_dexvla.py --epochs=100 --batch_size=128 --mixed_precision=fp16"
echo "5. Download the trained 'checkpoints/dexvla/dexvla_best.pth' back to your local PC."
echo "========================================================"
