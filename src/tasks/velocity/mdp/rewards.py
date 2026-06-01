from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _whole_body_com_w(asset: Entity) -> torch.Tensor:
  root_body_id = asset.data.indexing.root_body_id
  com_w = asset.data.data.subtree_com[:, root_body_id, :]
  if com_w.ndim == 3:
    com_w = com_w.squeeze(1)
  return com_w


def _foot_contact_mask(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  num_feet: int,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  contact = contact_sensor.data.found > 0
  if contact.ndim == 3:
    contact = contact.squeeze(-1)
  return contact[:, :num_feet]


def _foot_positions_in_root_frame(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  rel_pos_w = foot_pos_w - asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, rel_pos_w.shape[1], -1
  )
  foot_pos_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4), rel_pos_w.reshape(-1, 3)
  )
  return foot_pos_b.reshape(asset.data.root_link_pos_w.shape[0], -1, 3)


def _balance_risk(
  asset: Entity,
  tilt_limit: float = 0.35,
  ang_vel_limit: float = 1.5,
  planar_speed_limit: float = 0.75,
) -> torch.Tensor:
  tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  ang_vel_xy = torch.norm(asset.data.root_link_ang_vel_b[:, :2], dim=1)
  planar_speed = torch.norm(
    torch.cat(
      (asset.data.root_link_lin_vel_b[:, :2], asset.data.root_link_ang_vel_b[:, 2:3]),
      dim=1,
    ),
    dim=1,
  )
  risk = torch.maximum(
    torch.clamp(tilt / tilt_limit, 0.0, 1.0),
    torch.clamp(ang_vel_xy / ang_vel_limit, 0.0, 1.0),
  )
  risk = torch.maximum(risk, torch.clamp(planar_speed / planar_speed_limit, 0.0, 1.0))
  return risk


def _support_margin_violation(
  env: ManagerBasedRlEnv,
  asset: Entity,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  point_xy_w: torch.Tensor,
  foot_half_length: float,
  foot_half_width: float,
  margin: float,
  no_contact_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute support margin violation in a pelvis-yaw support frame."""
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  num_feet = foot_pos_w.shape[1]
  contact = _foot_contact_mask(env, sensor_name, num_feet)
  has_support = torch.any(contact, dim=1)

  x_axis = quat_apply(
    foot_quat_w.reshape(-1, 4),
    torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=foot_pos_w.dtype)
    .expand(foot_quat_w.reshape(-1, 4).shape[0], -1),
  ).reshape(env.num_envs, num_feet, 3)[..., :2]
  y_axis = quat_apply(
    foot_quat_w.reshape(-1, 4),
    torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=foot_pos_w.dtype)
    .expand(foot_quat_w.reshape(-1, 4).shape[0], -1),
  ).reshape(env.num_envs, num_feet, 3)[..., :2]
  x_axis = torch.nn.functional.normalize(x_axis, dim=-1)
  y_axis = torch.nn.functional.normalize(y_axis, dim=-1)

  offsets = torch.tensor(
    (
      (foot_half_length, foot_half_width),
      (foot_half_length, -foot_half_width),
      (-foot_half_length, foot_half_width),
      (-foot_half_length, -foot_half_width),
    ),
    device=env.device,
    dtype=foot_pos_w.dtype,
  )
  corners = (
    foot_pos_w[..., :2].unsqueeze(2)
    + x_axis.unsqueeze(2) * offsets[None, None, :, 0:1]
    + y_axis.unsqueeze(2) * offsets[None, None, :, 1:2]
  )

  root_x = quat_apply(
    asset.data.root_link_quat_w,
    torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=foot_pos_w.dtype)
    .expand(env.num_envs, -1),
  )[:, :2]
  root_x = torch.nn.functional.normalize(root_x, dim=-1)
  root_y = torch.stack((-root_x[:, 1], root_x[:, 0]), dim=1)

  corner_proj_x = torch.sum(corners * root_x[:, None, None, :], dim=-1)
  corner_proj_y = torch.sum(corners * root_y[:, None, None, :], dim=-1)
  active = contact.unsqueeze(-1)
  support_min_x = torch.amin(
    torch.where(active, corner_proj_x, torch.full_like(corner_proj_x, float("inf"))),
    dim=(1, 2),
  )
  support_max_x = torch.amax(
    torch.where(active, corner_proj_x, torch.full_like(corner_proj_x, -float("inf"))),
    dim=(1, 2),
  )
  support_min_y = torch.amin(
    torch.where(active, corner_proj_y, torch.full_like(corner_proj_y, float("inf"))),
    dim=(1, 2),
  )
  support_max_y = torch.amax(
    torch.where(active, corner_proj_y, torch.full_like(corner_proj_y, -float("inf"))),
    dim=(1, 2),
  )

  point_x = torch.sum(point_xy_w * root_x, dim=1)
  point_y = torch.sum(point_xy_w * root_y, dim=1)
  lower_x = support_min_x + margin
  upper_x = support_max_x - margin
  lower_y = support_min_y + margin
  upper_y = support_max_y - margin
  outside_x = torch.relu(lower_x - point_x) + torch.relu(point_x - upper_x)
  outside_y = torch.relu(lower_y - point_y) + torch.relu(point_y - upper_y)
  violation = outside_x.square() + outside_y.square()
  violation = torch.where(
    has_support,
    violation,
    violation + torch.full_like(violation, no_contact_penalty),
  )
  distance = torch.sqrt(outside_x.square() + outside_y.square())
  return violation, distance, contact.float().sum(dim=1)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def root_xy_displacement_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize horizontal root displacement from the environment origin."""
  asset: Entity = env.scene[asset_cfg.name]
  xy_error = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  displacement = torch.norm(xy_error, dim=1)
  env.extras["log"]["Metrics/root_xy_displacement_mean"] = torch.mean(displacement)
  return torch.sum(torch.square(xy_error), dim=1)


def root_xy_drift_huber(
  env: ManagerBasedRlEnv,
  deadband: float = 0.25,
  linear_width: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize walking away while allowing local recovery steps/lunges."""
  asset: Entity = env.scene[asset_cfg.name]
  xy_error = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  displacement = torch.norm(xy_error, dim=1)
  excess = torch.relu(displacement - deadband)
  cost = torch.where(
    excess < linear_width,
    0.5 * excess.square() / max(linear_width, 1.0e-6),
    excess - 0.5 * linear_width,
  )
  env.extras["log"]["Metrics/root_xy_displacement_mean"] = torch.mean(displacement)
  return cost


def root_planar_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base-frame planar root velocity [vx, vy, wz]."""
  asset: Entity = env.scene[asset_cfg.name]
  linear_xy = asset.data.root_link_lin_vel_b[:, :2]
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2:3]
  planar_velocity = torch.cat((linear_xy, yaw_rate), dim=1)
  speed = torch.norm(planar_velocity, dim=1)
  env.extras["log"]["Metrics/root_planar_speed_mean"] = torch.mean(speed)
  return torch.sum(torch.square(planar_velocity), dim=1)


def root_planar_velocity_saturating(
  env: ManagerBasedRlEnv,
  saturation_speed: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize residual planar motion without blocking transient recovery."""
  asset: Entity = env.scene[asset_cfg.name]
  planar_velocity = torch.cat(
    (asset.data.root_link_lin_vel_b[:, :2], asset.data.root_link_ang_vel_b[:, 2:3]),
    dim=1,
  )
  speed_sq = torch.sum(torch.square(planar_velocity), dim=1)
  cost = speed_sq / (speed_sq + saturation_speed**2)
  env.extras["log"]["Metrics/root_planar_speed_mean"] = torch.mean(torch.sqrt(speed_sq))
  return cost


def com_support_box_violation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_half_length: float,
  foot_half_width: float,
  margin: float = 0.0,
  no_contact_penalty: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize projected COM outside the contacted feet's support box.

  This uses a simple axis-aligned world-XY box around the contacted foot sites.
  The COM is MuJoCo's subtree COM for the robot root body.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None

  root_body_id = asset.data.indexing.root_body_id
  com_xy = asset.data.data.subtree_com[:, root_body_id, :2]
  if com_xy.ndim == 3:
    com_xy = com_xy.squeeze(1)

  foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
  effective_half_length = max(foot_half_length - margin, 0.0)
  effective_half_width = max(foot_half_width - margin, 0.0)
  half_extents = torch.tensor(
    (effective_half_length, effective_half_width),
    device=env.device,
    dtype=foot_xy.dtype,
  )
  foot_min = foot_xy - half_extents
  foot_max = foot_xy + half_extents

  contact = contact_sensor.data.found > 0
  if contact.ndim == 3:
    contact = contact.squeeze(-1)
  contact = contact[:, : foot_xy.shape[1]]
  has_support = torch.any(contact, dim=1)

  contact_expanded = contact.unsqueeze(-1)
  support_min = torch.min(
    torch.where(contact_expanded, foot_min, torch.full_like(foot_min, float("inf"))),
    dim=1,
  ).values
  support_max = torch.max(
    torch.where(contact_expanded, foot_max, torch.full_like(foot_max, -float("inf"))),
    dim=1,
  ).values

  support_min = torch.where(has_support.unsqueeze(1), support_min, com_xy)
  support_max = torch.where(has_support.unsqueeze(1), support_max, com_xy)
  outside = torch.relu(support_min - com_xy) + torch.relu(com_xy - support_max)
  violation = torch.sum(torch.square(outside), dim=1)
  violation = torch.where(
    has_support,
    violation,
    violation + torch.full_like(violation, no_contact_penalty),
  )

  outside_distance = torch.sqrt(torch.sum(torch.square(outside), dim=1))
  env.extras["log"]["Metrics/com_support_outside_distance_mean"] = torch.mean(
    outside_distance
  )
  env.extras["log"]["Metrics/support_contact_count_mean"] = torch.mean(
    contact.float().sum(dim=1)
  )
  return violation


def com_support_margin_violation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_half_length: float,
  foot_half_width: float,
  margin: float = 0.02,
  no_contact_penalty: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize COM being outside or near the edge of active foot support."""
  asset: Entity = env.scene[asset_cfg.name]
  com_xy = _whole_body_com_w(asset)[:, :2]
  violation, distance, contact_count = _support_margin_violation(
    env,
    asset,
    sensor_name,
    asset_cfg,
    com_xy,
    foot_half_length,
    foot_half_width,
    margin,
    no_contact_penalty,
  )
  env.extras["log"]["Metrics/com_support_margin_distance_mean"] = torch.mean(distance)
  env.extras["log"]["Metrics/support_contact_count_mean"] = torch.mean(contact_count)
  return violation


class capture_point_support_margin_violation:
  """Penalize capture point being outside or near active foot support."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev_com_xy = torch.zeros(env.num_envs, 2, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_half_length: float,
    foot_half_width: float,
    margin: float = 0.02,
    no_contact_penalty: float = 1.0,
    gravity: float = 9.81,
    min_com_height: float = 0.30,
    max_capture_offset: float = 1.00,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    com_w = _whole_body_com_w(asset)
    com_xy = com_w[:, :2]
    reset_mask = (env.episode_length_buf <= 1) | (~self.initialized)
    self.prev_com_xy = torch.where(reset_mask.unsqueeze(1), com_xy, self.prev_com_xy)
    com_vel_xy = (com_xy - self.prev_com_xy) / env.step_dt
    self.prev_com_xy[:] = com_xy
    self.initialized[:] = True

    omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
    capture_offset = com_vel_xy / omega.unsqueeze(1)
    capture_offset_norm = torch.norm(capture_offset, dim=1, keepdim=True)
    capture_offset = capture_offset * torch.clamp(
      max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
      max=1.0,
    )
    capture_xy = com_xy + capture_offset

    violation, distance, _ = _support_margin_violation(
      env,
      asset,
      sensor_name,
      asset_cfg,
      capture_xy,
      foot_half_length,
      foot_half_width,
      margin,
      no_contact_penalty,
    )
    env.extras["log"]["Metrics/capture_support_margin_distance_mean"] = torch.mean(
      distance
    )
    return violation


def track_planar_velocity_estimate(
  env: ManagerBasedRlEnv,
  action_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a no-op action head for estimating [vx_b, vy_b, wz_b]."""
  asset: Entity = env.scene[asset_cfg.name]
  estimate = env.action_manager.get_term(action_name).raw_action
  true_velocity = torch.cat(
    (
      asset.data.root_link_lin_vel_b[:, :2],
      asset.data.root_link_ang_vel_b[:, 2:3],
    ),
    dim=1,
  )
  error = torch.mean(torch.square(estimate - true_velocity), dim=1)
  env.extras["log"]["Metrics/planar_velocity_estimate_error_mean"] = torch.mean(
    torch.sqrt(error)
  )
  return torch.exp(-error / std**2)


def action_term_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Penalize action rate for one named action term only."""
  start = 0
  for name, dim in zip(
    env.action_manager.active_terms, env.action_manager.action_term_dim
  ):
    end = start + dim
    if name == action_name:
      return torch.sum(
        torch.square(
          env.action_manager.action[:, start:end]
          - env.action_manager.prev_action[:, start:end]
        ),
        dim=1,
      )
    start = end
  raise KeyError(f"Action term '{action_name}' not found.")


def action_term_acc_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Penalize second difference of one named action term."""
  start = 0
  for name, dim in zip(
    env.action_manager.active_terms, env.action_manager.action_term_dim
  ):
    end = start + dim
    if name == action_name:
      acc = (
        env.action_manager.action[:, start:end]
        - 2.0 * env.action_manager.prev_action[:, start:end]
        + env.action_manager.prev_prev_action[:, start:end]
      )
      return torch.sum(torch.square(acc), dim=1)
    start = end
  raise KeyError(f"Action term '{action_name}' not found.")


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def pelvis_tilt_barrier(
  env: ManagerBasedRlEnv,
  soft_limit: float = math.radians(12.0),
  hard_limit: float = math.radians(55.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Graded pelvis tilt penalty before terminal fall angles."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
  else:
    projected_gravity_b = asset.data.projected_gravity_b
  tilt = torch.acos(torch.clamp(-projected_gravity_b[:, 2], -1.0, 1.0)).abs()
  excess = torch.relu(tilt - soft_limit)
  normalized = excess / max(hard_limit - soft_limit, 1.0e-6)
  env.extras["log"]["Metrics/pelvis_tilt_deg_mean"] = torch.mean(
    tilt * (180.0 / math.pi)
  )
  return normalized.square()


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


class body_angular_acceleration_penalty:
  """Penalize body angular acceleration in roll/pitch axes."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev_ang_vel_xy = torch.zeros(env.num_envs, 2, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    ang_vel_xy = ang_vel[:, :2]
    reset_mask = (env.episode_length_buf <= 1) | (~self.initialized)
    self.prev_ang_vel_xy = torch.where(
      reset_mask.unsqueeze(1), ang_vel_xy, self.prev_ang_vel_xy
    )
    ang_acc = (ang_vel_xy - self.prev_ang_vel_xy) / env.step_dt
    self.prev_ang_vel_xy[:] = ang_vel_xy
    self.initialized[:] = True
    env.extras["log"]["Metrics/pelvis_ang_acc_mean"] = torch.mean(
      torch.norm(ang_acc, dim=1)
    )
    return torch.sum(torch.square(ang_acc), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  active = 1.0
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


class default_joint_position:
  """Reward staying near default joint positions with per-joint tolerances."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del std  # Resolved once in __init__.

    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


class risk_gated_default_joint_position:
  """Reward nominal posture strongly at low risk and loosely during recovery."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, low_risk_std = resolve_matching_names_values(
      data=cfg.params["low_risk_std"],
      list_of_strings=joint_names,
    )
    _, _, high_risk_std = resolve_matching_names_values(
      data=cfg.params["high_risk_std"],
      list_of_strings=joint_names,
    )
    self.low_risk_std = torch.tensor(
      low_risk_std, device=env.device, dtype=torch.float32
    )
    self.high_risk_std = torch.tensor(
      high_risk_std, device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    low_risk_std,
    high_risk_std,
    asset_cfg: SceneEntityCfg,
    risk_tilt_limit: float = 0.35,
    risk_ang_vel_limit: float = 1.5,
    risk_planar_speed_limit: float = 0.75,
  ) -> torch.Tensor:
    del low_risk_std, high_risk_std
    asset: Entity = env.scene[asset_cfg.name]
    risk = _balance_risk(
      asset,
      tilt_limit=risk_tilt_limit,
      ang_vel_limit=risk_ang_vel_limit,
      planar_speed_limit=risk_planar_speed_limit,
    )
    std = (
      self.low_risk_std * (1.0 - risk).unsqueeze(1)
      + self.high_risk_std * risk.unsqueeze(1)
    )
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    env.extras["log"]["Metrics/balance_risk_mean"] = torch.mean(risk)
    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stance_geometry_penalty(
  env: ManagerBasedRlEnv,
  nominal_width: float = 0.22,
  risk_width: float = 0.34,
  max_width: float = 0.65,
  risk_split_gain: float = 0.40,
  max_split: float = 0.65,
  foot_crossing_margin: float = 0.02,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Risk-gated stance width and fore-aft split shaping."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  if foot_pos_b.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  left = foot_pos_b[:, 0, :]
  right = foot_pos_b[:, 1, :]
  width = torch.abs(left[:, 1] - right[:, 1])
  split = torch.abs(left[:, 0] - right[:, 0])
  risk = _balance_risk(asset)

  target_min_width = nominal_width * (1.0 - risk) + risk_width * risk
  width_low_cost = torch.relu(target_min_width - width).square()
  width_high_cost = torch.relu(width - max_width).square()

  com_xy = _whole_body_com_w(asset)[:, :2]
  rel_com_w = com_xy - asset.data.root_link_pos_w[:, :2]
  root_quat_w = asset.data.root_link_quat_w
  rel_com_b = quat_apply_inverse(
    root_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  desired_split = torch.clamp(torch.abs(rel_com_b[:, 0]) * risk_split_gain, max=max_split)
  split_low_cost = risk * torch.relu(desired_split - split).square()
  split_high_cost = torch.relu(split - max_split).square()

  crossing_cost = torch.relu(foot_crossing_margin - (left[:, 1] - right[:, 1])).square()
  env.extras["log"]["Metrics/stance_width_mean"] = torch.mean(width)
  env.extras["log"]["Metrics/stance_fore_aft_split_mean"] = torch.mean(split)
  return width_low_cost + width_high_cost + split_low_cost + split_high_cost + crossing_cost


def no_foot_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize states with no supporting foot contact."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  contact = contact_sensor.data.found > 0
  if contact.ndim == 3:
    contact = contact.squeeze(-1)
  no_contact = torch.sum(contact.float(), dim=1) <= 0.0
  return no_contact.float()


def track_joint_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking commanded joint positions (exponential kernel)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  current = asset.data.joint_pos[:, asset_cfg.joint_ids]
  error = torch.mean(torch.square(current - command), dim=1)
  return torch.exp(-error / std**2)


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  diff_angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  reward = torch.sum(torch.square(diff_angle), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command <= command_threshold).float()
      reward *= scale
  return reward
