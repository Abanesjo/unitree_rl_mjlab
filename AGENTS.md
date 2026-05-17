# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package for Unitree robot RL training with MuJoCo, plus C++ simulation and deployment tools. Core Python code lives in `src/`: task definitions are under `src/tasks/velocity` and `src/tasks/tracking`, while robot constants, XML models, meshes, and motion assets live under `src/assets`. User-facing entry points are in `scripts/`. Deployment controllers are in `deploy/robots/<robot>/`; the bundled Unitree MuJoCo bridge is in `simulate/`. Documentation and media assets are in `doc/`. Treat `build/`, `logs/`, `wandb/`, and `*.egg-info/` as generated local artifacts.

## Build, Test, and Development Commands

Create the local virtual environment with `uv` and install the package in editable mode:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

List registered tasks with `python scripts/list_envs.py`. Train with `python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096`. Play a checkpoint with `python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt`. Convert motion CSV files with `python scripts/csv_to_npz.py --input-file src/assets/motions/g1/<file>.csv --output-name <name>.npz --input-fps 30 --output-fps 50`.

Build deployment or simulation binaries from their own directories:

```bash
cd deploy/robots/g1 && mkdir -p build && cd build && cmake .. && make
cd simulate && mkdir -p build && cd build && cmake .. && make -j8
```

## Coding Style & Naming Conventions

Use 4-space indentation for Python and follow existing `snake_case` module, function, and field names. Keep task IDs consistent with patterns such as `Unitree-G1-Flat` or `Unitree-G1-Tracking`. Put shared MDP logic in `mdp/`, robot-specific overrides in `config/<robot>/`, and hardware constants in `src/assets/robots/<robot>/*_constants.py`. Match nearby C++ style when editing deployment or simulator code.

## Testing Guidelines

There is no automated test suite or CI in this repository. Validate changes by listing tasks, running a small training or play command, and building relevant CMake targets. For robot-facing changes, test in simulation first with `--network=lo`.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, often lowercase, for example `add G1 environments`, `fix tracking deploy`, and `update setup.py`. Keep commits focused and exclude generated logs or build outputs. Pull requests should describe the affected robot/task, list validation commands, note required checkpoint or ONNX artifacts, and attach screenshots or clips when simulation behavior changes.

## Security & Configuration Tips

Do not commit private experiment logs, WandB data, network interface details, or large generated checkpoints unless they are intentional release artifacts. Keep deployable ONNX files in `deploy/robots/<robot>/config/policy/.../exported/`.
