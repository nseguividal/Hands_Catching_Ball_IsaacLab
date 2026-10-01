# Hands Catching Ball - Sim-to-Real (Camera & Tactile)

Branch: `feature/camera-and-force`

Two robotic Shadow Hands learn to dynamically throw and catch a ball across a 1.05m gap using **Wrist Depth Cameras** + **Overhead Humanoid Head Camera** + **Tactile Fingertip Sensors** (Option 1: Full Asymmetric Sim-to-Real).

---

## ⚡ Quick Start Commands

> **Important:** Always include `--enable_cameras` because this branch uses wrist and head depth cameras.

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

* Where <RUN_DIR> looks like: 2026-10-01_15-08-15  [YYYY-MM-DD_HH-MM-SS]

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