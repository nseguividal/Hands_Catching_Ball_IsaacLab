# Hands Catching Ball - Sim-to-Real (Camera & Tactile)

Branch: `feature/camera-and-force`

Two robotic Shadow Hands learn to dynamically throw and catch a ball across a 1.0m gap using **Wrist Depth Cameras** + **Tactile Fingertip Sensors** (Option 1: Full Asymmetric Sim-to-Real).

---

## ⚡ Quick Start Commands

> **Important:** Always include `--enable_cameras` because this branch uses wrist-mounted depth cameras.

### 1. Run Dummy Agent (Sanity Check)
Test the simulation physics, camera views, and hand setup with random actions:

```bash
python scripts/random_agent.py --task=Template-Catching-Ball-Rl-Direct-v0 --num_envs=16 --enable_cameras
```

---

### 2. Train the Model
Train the policy across 128 parallel environments (headless mode for fast GPU training):

```bash
python scripts/rl_games/train.py --task=Template-Catching-Ball-Rl-Direct-v0 --num_envs=128 --enable_cameras --headless
```

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