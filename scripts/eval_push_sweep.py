"""Evaluate lower-body push recovery checkpoints with fixed Newton force presets."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


PRESETS = {
  "mild": {
    "force_magnitude_range": (70.0, 110.0),
    "duration_s": (0.12, 0.18),
    "cooldown_s": (4.0, 7.0),
  },
  "medium": {
    "force_magnitude_range": (120.0, 180.0),
    "duration_s": (0.14, 0.22),
    "cooldown_s": (3.0, 5.0),
  },
  "planar_recovery": {
    "force_magnitude_range": (220.0, 320.0),
    "duration_s": (0.16, 0.24),
    "cooldown_s": (2.0, 4.0),
  },
  "hard_planar_recovery": {
    "force_magnitude_range": (300.0, 420.0),
    "duration_s": (0.16, 0.24),
    "cooldown_s": (2.0, 4.0),
  },
  "full": {
    "force_magnitude_range": (400.0, 550.0),
    "duration_s": (0.18, 0.28),
    "cooldown_s": (2.0, 4.0),
  },
}

METRIC_KEYS = (
  "Metrics/push_force_active_frac",
  "Metrics/push_force_mean_n",
  "Metrics/pelvis_tilt_deg_mean",
  "Metrics/pelvis_ang_acc_mean",
  "Metrics/stance_width_mean",
  "Metrics/stance_fore_aft_split_mean",
  "Metrics/recovery_reach_actual_x_mean",
  "Metrics/recovery_step_height_x_mean",
  "Episode_Metrics/mean_action_acc",
)


def _apply_eval_cfg(env_cfg, preset_name: str, num_envs: int, seed: int) -> None:
  preset = PRESETS[preset_name]
  env_cfg.seed = seed
  env_cfg.scene.num_envs = num_envs
  curriculum = getattr(env_cfg, "curriculum", None)
  if curriculum is None:
    curriculum = getattr(env_cfg, "curriculums", {})

  # Use mature upper-body disturbance during eval instead of the default frozen
  # startup stage.
  upper_curriculum = curriculum.get("upper_body_disturbance")
  if upper_curriculum is not None:
    stages = upper_curriculum.params.get("stages", [])
    if stages:
      mature_stage = dict(stages[-1])
      mature_stage["step"] = -1
      upper_curriculum.params["stages"] = [mature_stage]

  # Keep the force preset fixed for the whole sweep.
  curriculum.pop("push_robot", None)
  push_event = env_cfg.events["push_robot"]
  push_event.params["force_magnitude_range"] = preset["force_magnitude_range"]
  push_event.params["force_z_range"] = (0.0, 0.0)
  push_event.params["torque_range"] = (0.0, 0.0)
  push_event.params["duration_s"] = preset["duration_s"]
  push_event.params["cooldown_s"] = preset["cooldown_s"]


def _load_policy(task_id: str, checkpoint: Path, env, agent_cfg, device: str):
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def run_one(task_id: str, checkpoint: Path, preset_name: str, num_envs: int, steps: int, seed: int) -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  _apply_eval_cfg(env_cfg, preset_name, num_envs=num_envs, seed=seed)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  policy = _load_policy(task_id, checkpoint, env, agent_cfg, device)

  obs = env.get_observations()
  episode_lengths = torch.zeros(num_envs, device=device)
  completed_lengths: list[float] = []
  fall_lengths: list[float] = []
  last_log = {}

  with torch.inference_mode():
    for _ in range(steps):
      actions = policy(obs)
      obs, _, dones, extras = env.step(actions)
      episode_lengths += 1.0
      last_log = extras.get("log", {})
      done_ids = torch.nonzero(dones, as_tuple=False).flatten()
      if done_ids.numel() == 0:
        continue
      lengths = episode_lengths[done_ids].detach().cpu().tolist()
      completed_lengths.extend(lengths)
      max_len = float(env.max_episode_length)
      fall_lengths.extend(length for length in lengths if length < max_len - 1.0)
      episode_lengths[done_ids] = 0.0

  open_lengths = episode_lengths.detach().cpu()
  completed = len(completed_lengths)
  falls = len(fall_lengths)
  mean_completed = sum(completed_lengths) / completed if completed else 0.0
  mean_fall = sum(fall_lengths) / falls if falls else 0.0
  print(f"\n=== {preset_name} ===")
  print(f"checkpoint: {checkpoint}")
  print(f"completed_episodes: {completed}")
  print(f"falls: {falls}")
  print(f"fall_rate_completed: {falls / completed:.3f}" if completed else "fall_rate_completed: n/a")
  print(f"mean_completed_length: {mean_completed:.1f}")
  print(f"mean_fall_length: {mean_fall:.1f}")
  print(f"open_survivors: {(open_lengths > 0).sum().item()}/{num_envs}")
  print(f"open_length_mean: {open_lengths.float().mean().item():.1f}")
  for key in METRIC_KEYS:
    value = last_log.get(key)
    if value is None:
      continue
    if torch.is_tensor(value):
      value = value.detach().float().mean().item()
    print(f"{key}: {float(value):.4f}")
  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  task_id = os.environ.get("TASK_ID", "Unitree-G1-LowerBody-Flat")
  checkpoint = Path(os.environ["CHECKPOINT"])
  preset_names = os.environ.get("PRESET_NAMES", "mild,medium").split(",")
  num_envs = int(os.environ.get("NUM_ENVS", "24"))
  steps = int(os.environ.get("STEPS", "800"))
  seed = int(os.environ.get("SEED", "8383"))

  configure_torch_backends()
  for index, preset_name in enumerate(p.strip() for p in preset_names if p.strip()):
    if preset_name not in PRESETS:
      raise ValueError(f"Unknown preset: {preset_name}")
    run_one(task_id, checkpoint, preset_name, num_envs, steps, seed + index)


if __name__ == "__main__":
  main()
