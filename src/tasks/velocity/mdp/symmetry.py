"""Symmetry augmentation helpers for velocity tasks."""

from __future__ import annotations

import torch


ACTOR_HISTORY_LENGTH = 6
ACTOR_OBS_DIM = 630
CRITIC_OBS_DIM = 126
POLICY_ACTION_DIM = 15
LOWER_BODY_ACTION_DIM = 12
AUX_PLANAR_VELOCITY_DIM = 3

G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

G1_UPPER_BODY_COMMAND_NAMES = (
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

G1_LOWER_BODY_ACTION_NAMES = G1_JOINT_NAMES[:12]


def _mirror_joint_name(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name


def _joint_mirror_sign(name: str) -> float:
  if any(token in name for token in ("_roll_joint", "_yaw_joint")):
    return -1.0
  return 1.0


def _index_and_sign(
  names: tuple[str, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
  name_to_id = {name: i for i, name in enumerate(names)}
  indices = [name_to_id[_mirror_joint_name(name)] for name in names]
  signs = [_joint_mirror_sign(name) for name in names]
  return (
    torch.as_tensor(indices, device=device, dtype=torch.long),
    torch.as_tensor(signs, device=device, dtype=torch.float32),
  )


def _mirror_named(values: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
  index, sign = _index_and_sign(names, values.device)
  sign = sign.to(dtype=values.dtype)
  return values.index_select(-1, index) * sign


def _mirror_polar_vectors(values: torch.Tensor) -> torch.Tensor:
  mirrored = values.clone()
  mirrored[..., 1] = -mirrored[..., 1]
  return mirrored


def _mirror_axial_vectors(values: torch.Tensor) -> torch.Tensor:
  mirrored = values.clone()
  mirrored[..., 0] = -mirrored[..., 0]
  mirrored[..., 2] = -mirrored[..., 2]
  return mirrored


def _mirror_feet_vectors(values: torch.Tensor) -> torch.Tensor:
  mirrored = values.reshape(*values.shape[:-1], 2, 3).flip(-2)
  mirrored = _mirror_polar_vectors(mirrored)
  return mirrored.reshape_as(values)


def _mirror_feet_scalars(values: torch.Tensor) -> torch.Tensor:
  return values.reshape(*values.shape[:-1], 2).flip(-1).reshape_as(values)


def _mirror_lower_body_actions(values: torch.Tensor) -> torch.Tensor:
  return _mirror_named(values, G1_LOWER_BODY_ACTION_NAMES)


def _mirror_aux_planar_velocity(values: torch.Tensor) -> torch.Tensor:
  mirrored = values.clone()
  mirrored[..., 1] = -mirrored[..., 1]
  mirrored[..., 2] = -mirrored[..., 2]
  return mirrored


def _mirror_policy_actions(actions: torch.Tensor) -> torch.Tensor:
  if actions.shape[-1] != POLICY_ACTION_DIM:
    raise ValueError(f"Unexpected G1 lower-body action dim: {actions.shape[-1]}")
  mirrored = actions.clone()
  mirrored[..., :LOWER_BODY_ACTION_DIM] = _mirror_lower_body_actions(
    mirrored[..., :LOWER_BODY_ACTION_DIM]
  )
  aux_end = LOWER_BODY_ACTION_DIM + AUX_PLANAR_VELOCITY_DIM
  mirrored[..., LOWER_BODY_ACTION_DIM:aux_end] = _mirror_aux_planar_velocity(
    mirrored[..., LOWER_BODY_ACTION_DIM:aux_end]
  )
  return mirrored


def _mirror_history(
  values: torch.Tensor,
  history: int,
  width: int,
  mirror_fn,
) -> torch.Tensor:
  reshaped = values.reshape(*values.shape[:-1], history, width)
  mirrored = mirror_fn(reshaped)
  return mirrored.reshape_as(values)


def _mirror_actor_obs(actor_obs: torch.Tensor) -> torch.Tensor:
  mirrored = actor_obs.clone()
  if mirrored.shape[-1] != ACTOR_OBS_DIM:
    raise ValueError(f"Unexpected G1 lower-body actor obs dim: {mirrored.shape[-1]}")

  offset = 0
  mirrored[..., offset : offset + 18] = _mirror_history(
    actor_obs[..., offset : offset + 18],
    ACTOR_HISTORY_LENGTH,
    3,
    _mirror_axial_vectors,
  )
  offset += 18
  mirrored[..., offset : offset + 18] = _mirror_history(
    actor_obs[..., offset : offset + 18],
    ACTOR_HISTORY_LENGTH,
    3,
    _mirror_polar_vectors,
  )
  offset += 18
  mirrored[..., offset : offset + 174] = _mirror_history(
    actor_obs[..., offset : offset + 174],
    ACTOR_HISTORY_LENGTH,
    29,
    lambda x: _mirror_named(x, G1_JOINT_NAMES),
  )
  offset += 174
  mirrored[..., offset : offset + 174] = _mirror_history(
    actor_obs[..., offset : offset + 174],
    ACTOR_HISTORY_LENGTH,
    29,
    lambda x: _mirror_named(x, G1_JOINT_NAMES),
  )
  offset += 174
  mirrored[..., offset : offset + 72] = _mirror_history(
    actor_obs[..., offset : offset + 72],
    ACTOR_HISTORY_LENGTH,
    12,
    _mirror_lower_body_actions,
  )
  offset += 72
  mirrored[..., offset : offset + 102] = _mirror_history(
    actor_obs[..., offset : offset + 102],
    ACTOR_HISTORY_LENGTH,
    17,
    lambda x: _mirror_named(x, G1_UPPER_BODY_COMMAND_NAMES),
  )
  offset += 102
  mirrored[..., offset : offset + 36] = _mirror_history(
    actor_obs[..., offset : offset + 36],
    ACTOR_HISTORY_LENGTH,
    6,
    _mirror_feet_vectors,
  )
  offset += 36
  mirrored[..., offset : offset + 36] = _mirror_history(
    actor_obs[..., offset : offset + 36],
    ACTOR_HISTORY_LENGTH,
    6,
    _mirror_feet_vectors,
  )
  offset += 36
  if offset != ACTOR_OBS_DIM:
    raise RuntimeError(f"G1 lower-body actor mirror consumed {offset} dims")
  return mirrored


def _mirror_critic_obs(critic_obs: torch.Tensor) -> torch.Tensor:
  mirrored = critic_obs.clone()
  if mirrored.shape[-1] != CRITIC_OBS_DIM:
    raise ValueError(f"Unexpected G1 lower-body critic obs dim: {mirrored.shape[-1]}")

  offset = 0
  mirrored[..., offset : offset + 3] = _mirror_axial_vectors(
    critic_obs[..., offset : offset + 3]
  )
  offset += 3
  mirrored[..., offset : offset + 3] = _mirror_polar_vectors(
    critic_obs[..., offset : offset + 3]
  )
  offset += 3
  mirrored[..., offset : offset + 29] = _mirror_named(
    critic_obs[..., offset : offset + 29], G1_JOINT_NAMES
  )
  offset += 29
  mirrored[..., offset : offset + 29] = _mirror_named(
    critic_obs[..., offset : offset + 29], G1_JOINT_NAMES
  )
  offset += 29
  mirrored[..., offset : offset + 12] = _mirror_lower_body_actions(
    critic_obs[..., offset : offset + 12]
  )
  offset += 12
  mirrored[..., offset : offset + 3] = _mirror_polar_vectors(
    critic_obs[..., offset : offset + 3]
  )
  offset += 3
  mirrored[..., offset : offset + 2] = _mirror_feet_scalars(
    critic_obs[..., offset : offset + 2]
  )
  offset += 2
  mirrored[..., offset : offset + 2] = _mirror_feet_scalars(
    critic_obs[..., offset : offset + 2]
  )
  offset += 2
  mirrored[..., offset : offset + 2] = _mirror_feet_scalars(
    critic_obs[..., offset : offset + 2]
  )
  offset += 2
  mirrored[..., offset : offset + 6] = _mirror_feet_vectors(
    critic_obs[..., offset : offset + 6]
  )
  offset += 6
  mirrored[..., offset : offset + 17] = _mirror_named(
    critic_obs[..., offset : offset + 17], G1_UPPER_BODY_COMMAND_NAMES
  )
  offset += 17
  mirrored[..., offset : offset + 6] = _mirror_feet_vectors(
    critic_obs[..., offset : offset + 6]
  )
  offset += 6
  mirrored[..., offset : offset + 6] = _mirror_feet_vectors(
    critic_obs[..., offset : offset + 6]
  )
  offset += 6
  mirrored[..., offset : offset + 3] = _mirror_polar_vectors(
    critic_obs[..., offset : offset + 3]
  )
  offset += 3
  mirrored[..., offset : offset + 3] = _mirror_aux_planar_velocity(
    critic_obs[..., offset : offset + 3]
  )
  offset += 3
  if offset != CRITIC_OBS_DIM:
    raise RuntimeError(f"G1 lower-body critic mirror consumed {offset} dims")
  return mirrored


def g1_lower_body_mirror_augmentation(env, obs=None, actions=None):
  """Return original and left/right mirrored samples for G1 lower-body PPO."""
  mirrored_obs = None
  mirrored_actions = None

  if obs is not None:
    mirrored_obs = obs.clone()
    if "actor" in mirrored_obs.keys():
      mirrored_obs["actor"] = _mirror_actor_obs(obs["actor"])
    if "critic" in mirrored_obs.keys():
      mirrored_obs["critic"] = _mirror_critic_obs(obs["critic"])
    mirrored_obs = torch.cat((obs, mirrored_obs), dim=0)

  if actions is not None:
    mirrored_action_values = _mirror_policy_actions(actions)
    mirrored_actions = torch.cat((actions, mirrored_action_values), dim=0)

  return mirrored_obs, mirrored_actions
