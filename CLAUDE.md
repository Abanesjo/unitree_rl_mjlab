# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unitree RL Mjlab is an RL framework for training locomotion policies on Unitree robots (Go2, A2, G1, G1-23DOF, H1_2, R1) using MuJoCo physics. Built on the **mjlab** framework (Isaac Lab API + MuJoCo backend), it uses RSL-RL (PPO) for training and exports ONNX models for real-robot deployment.

## Setup

```bash
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
pip install -e .
```

The sole Python dependency is `mjlab==1.2.0` (which pulls in MuJoCo, PyTorch, rsl_rl, tyro, etc). System requirements: Ubuntu 22.04, Nvidia GPU, driver >= 550.

## Key Commands

### Training
```bash
# Velocity tracking (single GPU)
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# Multi-GPU (uses torchrunx)
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096

# Motion imitation (requires --motion_file)
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=4096

# Resume from checkpoint
python scripts/train.py Unitree-G1-Flat --agent.resume=True
```

### Play / Visualize
```bash
# Trained policy
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<run_dir>/model_xxx.pt

# Zero or random policy (no checkpoint needed)
python scripts/play.py Unitree-G1-Flat --agent=zero

# Motion tracking playback
python scripts/play.py Unitree-G1-Tracking --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=...

# Viewer options: auto (default), native (X11/Wayland), viser (web-based)
python scripts/play.py Unitree-G1-Flat --viewer=viser --checkpoint_file=...
```

### All registered task IDs

Velocity tracking (each robot has Flat and Rough variants):
`Unitree-Go2-Flat`, `Unitree-Go2-Rough`, `Unitree-G1-Flat`, `Unitree-G1-Rough`, `Unitree-G1-23Dof-Flat`, `Unitree-G1-23Dof-Rough`, `Unitree-G1-UpperBody-Flat`, `Unitree-G1-UpperBody-Rough`, `Unitree-H1_2-Flat`, `Unitree-H1_2-Rough`, `Unitree-A2-Flat`, `Unitree-A2-Rough`, `Unitree-R1-Flat`, `Unitree-R1-Rough`

Motion tracking:
`Unitree-G1-Tracking`, `Unitree-G1-Tracking-No-State-Estimation`

Use `python scripts/list_envs.py` to list all registered tasks (supports keyword filtering).

### Motion preprocessing
```bash
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz --input-fps 30 --output-fps 50
```

### C++ deployment build (example: G1)
```bash
cd deploy/robots/g1 && mkdir build && cd build && cmake .. && make
```

### C++ simulator build
```bash
cd simulate && mkdir build && cd build && cmake .. && make -j8
```

## Architecture

### CLI pattern
Both `scripts/train.py` and `scripts/play.py` use a two-pass **tyro** CLI: the first positional arg selects a task ID from the mjlab registry, then remaining args override the task's dataclass config. All `--env.*` and `--agent.*` flags map directly to config dataclass fields. The registry is populated by importing `src.tasks`, which auto-discovers all task modules.

### Task structure (`src/tasks/`)
Two task types, each following the same pattern:

- **velocity/** — Velocity tracking (walk/run with joystick commands)
- **tracking/** — Motion imitation (BeyondMimic-style whole-body tracking)

Each task has:
- `*_env_cfg.py` — Factory function returning a `ManagerBasedRlEnvCfg` with full MDP setup (observations, actions, rewards, terminations, commands, curriculum)
- `config/<robot>/env_cfgs.py` — Per-robot overrides (actuators, sensor frames, reward parameters, action scales)
- `config/<robot>/rl_cfg.py` — Per-robot RL hyperparameters (network dims, PPO settings, experiment name)
- `config/<robot>/__init__.py` — Registers task IDs via `register_mjlab_task()` with env_cfg, play_env_cfg (with `play=True`), rl_cfg, and runner_cls
- `mdp/` — MDP components: `observations.py`, `rewards.py`, `terminations.py`, `curriculums.py`, custom commands
- `rl/runner.py` — Custom `OnPolicyRunner` subclass (`VelocityOnPolicyRunner` or `MotionTrackingOnPolicyRunner`) that handles ONNX export with metadata

### Robot assets (`src/assets/robots/<robot>/`)
Each robot directory contains:
- `*_constants.py` — Action scales, actuator configs (BuiltinPositionActuatorCfg), motor specs, XML model paths, initial state, and `get_<robot>_robot_cfg()` factory function
- `xmls/` — MuJoCo XML model files and mesh assets

`SRC_PATH` (defined in `src/__init__.py`) is used throughout to resolve asset paths.

### Task registration flow
`src/tasks/__init__.py` calls `import_packages(__name__, _BLACKLIST_PKGS)` which auto-discovers all `config/<robot>/__init__.py` files. Each of those calls `register_mjlab_task()` to populate the registry with task_id, env configs, RL configs, and runner class.

### Training output
Logs go to `logs/rsl_rl/<experiment_name>/<datetime>/` containing:
- `model_<iter>.pt` — Checkpoints (every 100 iterations)
- `policy.onnx` + `policy.onnx.data` — Deployment-ready ONNX export
- `params/env.yaml`, `params/agent.yaml` — Config dumps

### Deployment (`deploy/`)
Per-robot C++ controllers using FSM (Finite State Machine) architecture, ONNX Runtime for inference, and Unitree SDK2 (CycloneDDS) for communication. Use `--network=lo` for sim, `--network=<iface>` for real robot. Place ONNX files in `deploy/robots/<robot>/config/policy/velocity/v0/exported/`.

## Key conventions

- Simulation runs at 200 Hz (dt=0.005), policy at 50 Hz (decimation=4)
- Reward functions are defined as standalone functions in `mdp/rewards.py` and referenced by `ManagerBasedRlEnvCfg` terms
- Observation terms follow the same pattern in `mdp/observations.py`
- Robot-specific constants (action scales, motor specs) live in `*_constants.py`, not in config files
- WandB is used for experiment tracking; logs directory is gitignored
- No automated tests or CI — testing is manual via the simulation pipeline
