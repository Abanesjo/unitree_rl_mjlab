# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unitree RL Mjlab is an RL framework for training locomotion policies on Unitree robots (Go2, A2, G1, G1-23DOF, H1_2, R1) using MuJoCo physics. Built on the **mjlab** framework (Isaac Lab API + MuJoCo backend), it uses RSL-RL (PPO) for training and exports ONNX models for real-robot deployment.

## Setup

```bash
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab
pip install -e .
```

The sole Python dependency is `mjlab==1.2.0` (which pulls in MuJoCo, PyTorch, rsl_rl, tyro, etc).

## Key Commands

### Training
```bash
# Velocity tracking (single GPU)
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# Multi-GPU
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096

# Motion imitation (requires --motion_file)
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=4096

# Resume from checkpoint
python scripts/train.py Unitree-G1-Flat --agent.resume=True
```

### Available task IDs
`Unitree-Go2-Flat`, `Unitree-G1-Flat`, `Unitree-G1-23Dof-Flat`, `Unitree-H1_2-Flat`, `Unitree-A2-Flat`, `Unitree-R1-Flat`, `Unitree-G1-Tracking-No-State-Estimation`

Use `python scripts/list_envs.py` to list all registered tasks.

### Play / Visualize
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<run_dir>/model_xxx.pt
```

### Motion preprocessing
```bash
python scripts/csv_to_npz.py --input-file <csv> --output-name <name>.npz --input-fps 30 --output-fps 50
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
Both `scripts/train.py` and `scripts/play.py` use a two-pass **tyro** CLI: the first positional arg selects a task ID from the mjlab registry, then remaining args override the task's dataclass config. All `--env.*` and `--agent.*` flags map directly to config dataclass fields.

### Task structure (`src/tasks/`)
Two task types, each following the same pattern:

- **velocity/** — Velocity tracking (walk/run with joystick commands)
- **tracking/** — Motion imitation (BeyondMimic-style whole-body tracking)

Each task has:
- `*_env_cfg.py` — Factory function returning a `ManagerBasedRlEnvCfg` with full MDP setup (observations, actions, rewards, terminations, commands, curriculum)
- `config/<robot>/env_cfgs.py` — Per-robot overrides (actuators, sensor frames, reward parameters, action scales)
- `config/<robot>/rl_cfg.py` — Per-robot RL hyperparameters (network dims, PPO settings, experiment name)
- `mdp/` — MDP components: `observations.py`, `rewards.py`, `terminations.py`, `curriculums.py`, custom commands
- `rl/runner.py` — Custom `OnPolicyRunner` subclass that handles ONNX export with metadata

### Robot assets (`src/assets/robots/<robot>/`)
Each robot directory contains:
- `*_constants.py` — Action scales, actuator configs, motor specs, XML model paths
- `xmls/` — MuJoCo XML model files

### Task registration
`src/tasks/__init__.py` uses `mjlab.utils.lab_api.tasks.importer.import_packages` to auto-discover and register all task modules. Per-robot config files register tasks via mjlab's registry decorators.

### Training output
Logs go to `logs/rsl_rl/<experiment_name>/<datetime>/` containing:
- `model_<iter>.pt` — Checkpoints (every 100 iterations)
- `policy.onnx` + `policy.onnx.data` — Deployment-ready ONNX export
- `params/env.yaml`, `params/agent.yaml` — Config dumps

### Deployment (`deploy/`)
Per-robot C++ controllers using FSM (Finite State Machine) architecture, ONNX Runtime for inference, and Unitree SDK2 (CycloneDDS) for communication. Use `--network=lo` for sim, `--network=<iface>` for real robot.

## Key conventions

- Simulation runs at 200 Hz (dt=0.005), policy at 50 Hz (decimation=4)
- Reward functions are defined as standalone functions in `mdp/rewards.py` and referenced by `ManagerBasedRlEnvCfg` terms
- Observation terms follow the same pattern in `mdp/observations.py`
- Robot-specific constants (action scales, motor specs) live in `*_constants.py`, not in config files
- WandB is used for experiment tracking; logs directory is gitignored
