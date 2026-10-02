# Hands Catching Ball - Sim-to-Real (Camera & Tactile)

Branch: `feature/vla-dexterity`

Two robotic Shadow Hands learn to dynamically throw and catch a ball across a 1.05m gap using **Wrist Depth Cameras** + **Overhead Humanoid Head Camera** + **Tactile Fingertip Sensors** (Option 1: Full Asymmetric Sim-to-Real), and export demonstration datasets for Dexterous VLA fine-tuning.

---

## ⚡ Quick Start Commands

> **Important:** Always include `--enable_cameras` because this project uses wrist and head depth cameras.

### 1. Run Dummy Agent (Sanity Check)
Test the simulation physics, camera views, and hand setup with random actions:

```bash
python scripts/random_agent.py --task=Template-Catching-Ball-Rl-Direct-v0 --num_envs=16 --enable_cameras
```

---

### 2. Train the Model

#### A. Standard Fast Headless Training:
```bash
python scripts/rl_games/train.py --task=Template-Catching-Ball-Rl-Direct-v0 --num_envs=256 --enable_cameras --headless
```

#### B. Headless Training with WebRTC Live Stream (Watch Live in Browser):
Stream the live 3D viewport to a browser on another PC/laptop without extra GPU load:

```bash
python scripts/rl_games/train.py --task=Template-Catching-Ball-Rl-Direct-v0 --num_envs=256 --enable_cameras --headless --livestream=1
```
* **To view the stream:** Open Google Chrome or Firefox on a second computer and go to:
  ```
  http://<IP_OF_TRAINING_PC>:8211/streaming/webrtc-client/
  ```

#### C. Resume Training from Saved Checkpoint:
To see all saved checkpoints sorted by newest first:
```bash
ls -lt logs/rl_games/shadow_hand_over_sim2real/*/nn/*.pth
```

Continue training an existing model run:

```bash
python scripts/rl_games/train.py --task=Template-Catching-Ball-Rl-Direct-v0 --num_envs=256 --enable_cameras --headless --livestream=1 --checkpoint=logs/rl_games/shadow_hand_over_sim2real/<RUN_DIR>/nn/shadow_hand_over_sim2real.pth
```

* Where `<RUN_DIR>` looks like: `2026-10-01_15-08-15` (`YYYY-MM-DD_HH-MM-SS`)
* Checkpoints are automatically saved to `logs/rl_games/shadow_hand_over_sim2real/<date_time>/nn/`.

---

### 3. Play / Visualize Trained Policy
Watch your trained policy throw and catch in the Omniverse GUI viewport:

```bash
# Automatically load the latest trained checkpoint:
python scripts/rl_games/play.py --task=Template-Catching-Ball-Rl-Direct-v0 --use_last_checkpoint --num_envs=16 --enable_cameras
```

Or load a specific checkpoint:
```bash
python scripts/rl_games/play.py --task=Template-Catching-Ball-Rl-Direct-v0 --checkpoint=logs/rl_games/shadow_hand_over_sim2real/<RUN_DIR>/nn/shadow_hand_over_sim2real.pth --num_envs=16 --enable_cameras
```

---

### 4. Monitor Training (TensorBoard)
Track rewards, contact forces, and ball distances in real time:

```bash
python -m tensorboard.main --logdir=logs/rl_games/shadow_hand_over_sim2real
```
Open **`http://localhost:6006`** in your browser.

---

### 5. Record Dexterous VLA (Vision-Language-Action) Dataset
Record multi-camera Depth feeds, fingertip 3D tactile forces, joint proprioception, 40-DoF dual-hand actions, and diverse natural language instructions from your trained expert policy:

```bash
# Record 50 successful demonstration episodes (running 16 parallel envs):
python scripts/vla/record_rollouts.py --use_last_checkpoint --num_episodes=50 --num_envs=16 --headless

# Or specify a custom checkpoint, episode count, and output directory:
python scripts/vla/record_rollouts.py --checkpoint=logs/rl_games/shadow_hand_over_sim2real/<RUN_DIR>/nn/shadow_hand_over_sim2real.pth --num_episodes=200 --num_envs=16 --output_dir=datasets/vla_shadow_hand --headless
```

---

### 6. Inspect & Validate Recorded Dataset
Inspect dataset structure, tensor dimensions, rewards, and language prompt metadata:

```bash
python scripts/vla/inspect_dataset.py --dataset_dir=datasets/vla_shadow_hand
```

---

### 7. Visualize & Export Episode Replays
Render video replays or animated GIFs of the recorded camera feeds (with depth colormaps) to inspect the demonstrations:

```bash
# Export Multi-View Video (Overhead Head + Left & Right Wrist Cameras):
python scripts/vla/visualize_episode.py --episode=datasets/vla_shadow_hand/episode_000001.h5 --output=episode_01_all_views.mp4

# Export Overhead Head Camera only:
python scripts/vla/visualize_episode.py --episode=datasets/vla_shadow_hand/episode_000001.h5 --camera=overhead_head_depth --output=episode_01_head_camera.mp4

# Export as Animated GIF (viewable in any browser / image viewer):
python scripts/vla/visualize_episode.py --episode=datasets/vla_shadow_hand/episode_000001.h5 --output=episode_01_replay.gif
```

---

### 8. Train Dex-VLA (Cloud or Local GPU)

#### A. Bundle Dataset & Training Code:
Package everything needed into a single compressed `.tar.gz` bundle:
```bash
bash scripts/vla/bundle_for_cloud.sh
```
This generates `dexvla_cloud_package.tar.gz`.

#### B. On Cloud GPU (RunPod / Lambda / AWS / Vast.ai):
```bash
# 1. Extract package:
tar -xzvf dexvla_cloud_package.tar.gz

# 2. Install lightweight requirements (No Isaac Sim needed on cloud):
pip install -r scripts/vla/requirements_cloud.txt

# 3. Launch Dex-VLA Diffusion Training (A100 / H100 / RTX 4090):
python scripts/vla/train_dexvla.py --epochs=100 --batch_size=128 --mixed_precision=fp16
```

#### C. Bring Model Back to Local Machine:
Download `checkpoints/dexvla/dexvla_best.pth` (~60 MB) back into your local repository to run closed-loop evaluation in Isaac Lab!

---

### 9. Closed-Loop Evaluation in Isaac Lab
Test the trained Dex-VLA Diffusion Action Expert live in the Isaac Lab 3D physics simulator conditioned on natural language prompts:

```bash
# Evaluate with GUI viewport:
python scripts/vla/eval_dexvla.py --checkpoint=checkpoints/dexvla/dexvla_best.pth --num_envs=4

# Evaluate headless (fast quantitative metrics across 16 envs):
python scripts/vla/eval_dexvla.py --checkpoint=checkpoints/dexvla/dexvla_best.pth --num_envs=16 --headless

# Evaluate with custom natural language prompt:
python scripts/vla/eval_dexvla.py --checkpoint=checkpoints/dexvla/dexvla_best.pth --prompt="Throw the ball with the right hand and catch it firmly with the left hand." --num_envs=4
```