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


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward each non-terminal control step."""
  return torch.ones(env.num_envs, device=env.device)


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


def _sagittal_biased_direction(
  direction: torch.Tensor,
  sagittal_bias_gain: float = 1.0,
  lateral_suppression: float = 0.0,
  sagittal_activation: float = 0.08,
) -> torch.Tensor:
  """Bias diagonal recovery directions toward fore-aft support when x demand exists."""
  if sagittal_bias_gain == 1.0 and lateral_suppression <= 0.0:
    return direction
  sagittal_gate = torch.clamp(
    torch.abs(direction[:, 0]) / max(sagittal_activation, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  lateral_scale = torch.clamp(1.0 - lateral_suppression * sagittal_gate, min=0.0)
  return torch.stack(
    (direction[:, 0] * sagittal_bias_gain, direction[:, 1] * lateral_scale),
    dim=1,
  )


def _balanced_foot_tie_bias(
  env: ManagerBasedRlEnv,
  num_feet: int,
  dtype: torch.dtype,
  scale: float = 0.0,
  period_s: float = 4.0,
) -> torch.Tensor:
  """Tiny deterministic tie-breaker that balances preferred foot across envs/time."""
  if scale <= 0.0 or num_feet < 2:
    return torch.zeros((env.num_envs, num_feet), device=env.device, dtype=dtype)
  env_ids = torch.arange(env.num_envs, device=env.device)
  phase = 0
  if period_s > 0.0:
    period_steps = max(1, int(round(period_s / max(env.step_dt, 1.0e-6))))
    common_step = getattr(env, "common_step_counter", 0)
    if torch.is_tensor(common_step):
      common_step = int(common_step.item())
    phase = int(common_step) // period_steps
  env_sign = torch.where(
    ((env_ids + phase) % 2) == 0,
    torch.full((env.num_envs,), -1.0, device=env.device, dtype=dtype),
    torch.full((env.num_envs,), 1.0, device=env.device, dtype=dtype),
  )
  foot_ids = torch.arange(num_feet, device=env.device, dtype=dtype)
  centered_foot_ids = foot_ids - 0.5 * float(num_feet - 1)
  return scale * env_sign.unsqueeze(1) * centered_foot_ids.unsqueeze(0)


def _signed_extreme_with_tie_bias(
  env: ManagerBasedRlEnv,
  values: torch.Tensor,
  positive: torch.Tensor,
  scale: float = 0.0,
  period_s: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Select max or min per env, using a balanced bias only for near-ties."""
  num_values = values.shape[1]
  tie_bias = _balanced_foot_tie_bias(
    env,
    num_values,
    values.dtype,
    scale=scale,
    period_s=period_s,
  )
  score = torch.where(positive.unsqueeze(1), values, -values)
  _, selected_idx = torch.max(score + tie_bias, dim=1)
  selected = torch.gather(values, 1, selected_idx.unsqueeze(1)).squeeze(1)
  signed_selected = torch.where(positive, selected, -selected)
  return signed_selected, selected_idx


def _directional_recovery_step_need(
  env: ManagerBasedRlEnv,
  asset: Entity,
  foot_pos_b: torch.Tensor,
  min_reach: float = 0.30,
  max_reach: float = 0.72,
  capture_reach_gain: float = 1.05,
  velocity_reach_gain: float = 0.95,
  direction_com_gain: float = 0.60,
  direction_velocity_gain: float = 0.95,
  direction_deadband: float = 0.04,
  sagittal_bias_gain: float = 1.35,
  lateral_suppression: float = 0.65,
  sagittal_activation: float = 0.08,
  risk_activation: float = 0.12,
  target_velocity: float = 0.22,
  need_scale: float = 0.08,
  dynamic_need_weight: float = 0.35,
  gravity: float = 9.81,
  min_com_height: float = 0.30,
  max_capture_offset: float = 0.90,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
) -> torch.Tensor:
  """Estimate whether support-polygon recovery actually needs a swing step."""
  com_w = _whole_body_com_w(asset)
  rel_com_w = com_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
  capture_offset_b = root_lin_vel_b / omega.unsqueeze(1)
  capture_offset_norm = torch.norm(capture_offset_b, dim=1, keepdim=True)
  capture_offset_b = capture_offset_b * torch.clamp(
    max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
    max=1.0,
  )
  rel_capture_b = rel_com_b[:, :2] + capture_offset_b
  risk = _balance_risk(
    asset,
    tilt_limit=risk_tilt_limit,
    ang_vel_limit=risk_ang_vel_limit,
    planar_speed_limit=risk_planar_speed_limit,
  )

  direction = (
    direction_com_gain * rel_capture_b
    + direction_velocity_gain * root_lin_vel_b
  )
  direction = _sagittal_biased_direction(
    direction,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
  )
  direction_norm = torch.norm(direction, dim=1)
  risk_scale = torch.clamp(
    (risk - risk_activation) / max(1.0 - risk_activation, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  active = risk_scale * (direction_norm > direction_deadband).float()
  direction_unit = direction / torch.clamp(direction_norm.unsqueeze(1), min=1.0e-6)

  foot_proj = torch.sum(foot_pos_b[:, :, :2] * direction_unit[:, None, :], dim=-1)
  leading_reach = torch.max(foot_proj, dim=1).values
  capture_need = torch.abs(torch.sum(rel_capture_b * direction_unit, dim=1))
  velocity_need = torch.relu(torch.sum(root_lin_vel_b * direction_unit, dim=1))
  target_reach = torch.clamp(
    min_reach + capture_reach_gain * capture_need + velocity_reach_gain * velocity_need,
    max=max_reach,
  )
  raw_reach_gap = torch.relu(target_reach - leading_reach)
  dynamic_need = torch.clamp(
    (
      torch.clamp(capture_need / max(min_reach, 1.0e-6), max=1.0)
      + torch.clamp(velocity_need / max(target_velocity, 1.0e-6), max=1.0)
    )
    * 0.5,
    max=1.0,
  )
  return active * torch.clamp(
    raw_reach_gap / max(need_scale, 1.0e-6)
    + dynamic_need_weight * dynamic_need,
    max=1.0,
  )


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

  # No-contact environments have no finite support bounds. Collapse the
  # interval to the point itself so the logged margin distance stays finite;
  # the explicit no-contact penalty below handles the bad state.
  support_min_x = torch.where(has_support, support_min_x, point_x)
  support_max_x = torch.where(has_support, support_max_x, point_x)
  support_min_y = torch.where(has_support, support_min_y, point_y)
  support_max_y = torch.where(has_support, support_max_y, point_y)

  # Shrinking by a margin can invert a one-foot support interval. Collapse
  # inverted intervals to their center instead of creating artificial distance.
  center_x = 0.5 * (support_min_x + support_max_x)
  center_y = 0.5 * (support_min_y + support_max_y)
  lower_x = torch.minimum(support_min_x + margin, center_x)
  upper_x = torch.maximum(support_max_x - margin, center_x)
  lower_y = torch.minimum(support_min_y + margin, center_y)
  upper_y = torch.maximum(support_max_y - margin, center_y)
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


def _support_signed_margin(
  env: ManagerBasedRlEnv,
  asset: Entity,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  point_xy_w: torch.Tensor,
  foot_half_length: float,
  foot_half_width: float,
  margin: float,
  no_contact_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return signed support margin for a point in the pelvis-yaw support frame."""
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
  support_min_x = torch.where(has_support, support_min_x, point_x)
  support_max_x = torch.where(has_support, support_max_x, point_x)
  support_min_y = torch.where(has_support, support_min_y, point_y)
  support_max_y = torch.where(has_support, support_max_y, point_y)

  center_x = 0.5 * (support_min_x + support_max_x)
  center_y = 0.5 * (support_min_y + support_max_y)
  lower_x = torch.minimum(support_min_x + margin, center_x)
  upper_x = torch.maximum(support_max_x - margin, center_x)
  lower_y = torch.minimum(support_min_y + margin, center_y)
  upper_y = torch.maximum(support_max_y - margin, center_y)

  inside_x = torch.minimum(point_x - lower_x, upper_x - point_x)
  inside_y = torch.minimum(point_y - lower_y, upper_y - point_y)
  inside_margin = torch.minimum(inside_x, inside_y)
  outside_x = torch.relu(lower_x - point_x) + torch.relu(point_x - upper_x)
  outside_y = torch.relu(lower_y - point_y) + torch.relu(point_y - upper_y)
  outside_distance = torch.sqrt(outside_x.square() + outside_y.square())
  is_inside = (outside_x <= 0.0) & (outside_y <= 0.0)
  signed_margin = torch.where(is_inside, inside_margin, -outside_distance)
  signed_margin = torch.where(
    has_support,
    signed_margin,
    torch.full_like(signed_margin, -abs(float(no_contact_margin))),
  )
  return signed_margin, contact.float().sum(dim=1)


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
  risk_reduction: float = 0.0,
  min_scale: float = 1.0,
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
  if risk_reduction > 0.0:
    risk = _balance_risk(asset)
    scale = torch.clamp(1.0 - risk_reduction * risk, min=min_scale, max=1.0)
    cost = cost * scale
    env.extras["log"]["Metrics/root_xy_drift_penalty_scale_mean"] = torch.mean(scale)
  env.extras["log"]["Metrics/root_xy_displacement_mean"] = torch.mean(displacement)
  return cost


def root_xy_return_velocity_bonus(
  env: ManagerBasedRlEnv,
  deadband: float = 0.45,
  displacement_width: float = 0.45,
  target_return_speed: float = 0.35,
  stable_risk: float = 0.35,
  max_bonus: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward moving back toward the origin only after balance risk has settled."""
  asset: Entity = env.scene[asset_cfg.name]
  xy_error = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  displacement = torch.norm(xy_error, dim=1)
  direction_from_origin = xy_error / torch.clamp(displacement.unsqueeze(1), min=1.0e-6)
  radial_velocity = torch.sum(
    asset.data.root_link_lin_vel_w[:, :2] * direction_from_origin,
    dim=1,
  )
  return_speed = torch.relu(-radial_velocity)
  displacement_scale = torch.clamp(
    (displacement - deadband) / max(displacement_width, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  risk = _balance_risk(asset)
  stable_scale = torch.clamp(
    (stable_risk - risk) / max(stable_risk, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  speed_score = torch.clamp(
    return_speed / max(target_return_speed, 1.0e-6),
    max=1.0,
  )
  bonus = max_bonus * stable_scale * displacement_scale * speed_score
  env.extras["log"]["Metrics/root_xy_return_bonus_mean"] = torch.mean(bonus)
  env.extras["log"]["Metrics/root_xy_return_stable_scale_mean"] = torch.mean(
    stable_scale
  )
  env.extras["log"]["Metrics/root_xy_return_radial_velocity_mean"] = torch.mean(
    radial_velocity
  )
  return bonus


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


def root_planar_velocity_saturating_risk_gated(
  env: ManagerBasedRlEnv,
  saturation_speed: float = 0.75,
  risk_reduction: float = 0.85,
  min_scale: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize residual planar motion less during active balance recovery."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = root_planar_velocity_saturating(
    env,
    saturation_speed=saturation_speed,
    asset_cfg=asset_cfg,
  )
  risk = _balance_risk(asset)
  scale = torch.clamp(1.0 - risk_reduction * risk, min=min_scale)
  env.extras["log"]["Metrics/root_planar_velocity_penalty_scale_mean"] = torch.mean(
    scale
  )
  return cost * scale


def supported_root_planar_velocity_brake(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  speed_deadband: float = 0.45,
  saturation_speed: float = 0.65,
  min_contacts: float = 1.35,
  risk_activation: float = 0.20,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize residual planar velocity once a recovery stance is supported."""
  asset: Entity = env.scene[asset_cfg.name]
  num_feet = len(asset_cfg.site_ids) if asset_cfg.site_ids is not None else 2
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  contact_count = torch.sum(contact, dim=1)
  support_scale = torch.clamp(
    (contact_count - min_contacts) / max(float(num_feet) - min_contacts, 1.0e-6),
    min=0.0,
    max=1.0,
  )

  planar_velocity = torch.cat(
    (asset.data.root_link_lin_vel_b[:, :2], asset.data.root_link_ang_vel_b[:, 2:3]),
    dim=1,
  )
  speed = torch.norm(planar_velocity, dim=1)
  excess = torch.relu(speed - speed_deadband)
  speed_cost = torch.square(excess) / (
    torch.square(excess) + saturation_speed**2
  )
  risk = _balance_risk(asset)
  risk_scale = torch.clamp(
    (risk - risk_activation) / max(1.0 - risk_activation, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  cost = speed_cost * support_scale * risk_scale

  env.extras["log"]["Metrics/supported_brake_cost_mean"] = torch.mean(cost)
  env.extras["log"]["Metrics/supported_brake_support_scale_mean"] = torch.mean(
    support_scale
  )
  env.extras["log"]["Metrics/supported_brake_speed_mean"] = torch.mean(speed)
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


class capture_margin_improvement_reward:
  """Reward step-to-step improvement in capture-point support margin."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev_com_xy = torch.zeros(env.num_envs, 2, device=env.device)
    self.prev_margin = torch.zeros(env.num_envs, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_half_length: float,
    foot_half_width: float,
    margin: float = 0.02,
    no_contact_margin: float = 0.25,
    gravity: float = 9.81,
    min_com_height: float = 0.30,
    max_capture_offset: float = 1.00,
    delta_scale: float = 0.04,
    max_reward: float = 1.0,
    max_penalty: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    com_w = _whole_body_com_w(asset)
    com_xy = com_w[:, :2]
    reset_mask = (env.episode_length_buf <= 1) | (~self.initialized)
    self.prev_com_xy = torch.where(reset_mask.unsqueeze(1), com_xy, self.prev_com_xy)
    com_vel_xy = (com_xy - self.prev_com_xy) / env.step_dt
    self.prev_com_xy[:] = com_xy

    omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
    capture_offset = com_vel_xy / omega.unsqueeze(1)
    capture_offset_norm = torch.norm(capture_offset, dim=1, keepdim=True)
    capture_offset = capture_offset * torch.clamp(
      max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
      max=1.0,
    )
    capture_xy = com_xy + capture_offset
    signed_margin, contact_count = _support_signed_margin(
      env,
      asset,
      sensor_name,
      asset_cfg,
      capture_xy,
      foot_half_length,
      foot_half_width,
      margin,
      no_contact_margin,
    )

    self.prev_margin = torch.where(reset_mask, signed_margin, self.prev_margin)
    delta = signed_margin - self.prev_margin
    self.prev_margin[:] = signed_margin
    self.initialized[:] = True

    risk = _balance_risk(asset)
    reward = torch.clamp(
      delta / max(delta_scale, 1.0e-6),
      min=-abs(float(max_penalty)),
      max=abs(float(max_reward)),
    )
    reward = reward * risk
    env.extras["log"]["Metrics/capture_signed_margin_mean"] = torch.mean(
      signed_margin
    )
    env.extras["log"]["Metrics/capture_margin_delta_mean"] = torch.mean(delta)
    env.extras["log"]["Metrics/capture_margin_contact_count_mean"] = torch.mean(
      contact_count
    )
    return reward


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


def action_term_rate_l2_risk_gated(
  env: ManagerBasedRlEnv,
  action_name: str,
  risk_reduction: float = 0.80,
  min_scale: float = 0.20,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize action rate less when recovery motion is necessary."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = action_term_rate_l2(env, action_name)
  risk = _balance_risk(asset)
  scale = torch.clamp(1.0 - risk_reduction * risk, min=min_scale)
  env.extras["log"]["Metrics/action_rate_penalty_scale_mean"] = torch.mean(scale)
  return cost * scale


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


def action_term_acc_l2_risk_gated(
  env: ManagerBasedRlEnv,
  action_name: str,
  risk_reduction: float = 0.85,
  min_scale: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize action acceleration less when recovery motion is necessary."""
  asset: Entity = env.scene[asset_cfg.name]
  cost = action_term_acc_l2(env, action_name)
  risk = _balance_risk(asset)
  scale = torch.clamp(1.0 - risk_reduction * risk, min=min_scale)
  env.extras["log"]["Metrics/action_acc_penalty_scale_mean"] = torch.mean(scale)
  return cost * scale


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


def low_risk_foot_motion_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_idle_risk: float = 0.22,
  airborne_velocity_weight: float = 1.0,
  airborne_height_weight: float = 12.0,
  airborne_contact_weight: float = 0.0,
  fresh_takeoff_weight: float = 0.0,
  fresh_takeoff_window_s: float = 0.06,
  height_deadband: float = 0.035,
  ground_height: float = 0.0,
  return_displacement_deadband: float = 0.0,
  return_displacement_width: float = 0.50,
  return_motion_relief: float = 0.0,
  motion_need_threshold: float = 0.0,
  motion_need_idle_weight: float = 1.0,
  min_reach: float = 0.30,
  max_reach: float = 0.72,
  capture_reach_gain: float = 1.05,
  velocity_reach_gain: float = 0.95,
  direction_com_gain: float = 0.60,
  direction_velocity_gain: float = 0.95,
  direction_deadband: float = 0.04,
  sagittal_bias_gain: float = 1.35,
  lateral_suppression: float = 0.65,
  sagittal_activation: float = 0.08,
  risk_activation: float = 0.12,
  target_velocity: float = 0.22,
  need_scale: float = 0.08,
  dynamic_need_weight: float = 0.35,
  gravity: float = 9.81,
  min_com_height: float = 0.30,
  max_capture_offset: float = 0.90,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize airborne foot motion when recovery does not need a step."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  num_feet = foot_pos_w.shape[1]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.current_air_time is not None
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  airborne = 1.0 - contact
  current_air_time = sensor_data.current_air_time[:, :num_feet]
  fresh_takeoff = airborne * (current_air_time <= fresh_takeoff_window_s).float()

  foot_vel_w = (
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    - asset.data.root_link_lin_vel_w[:, None, :]
  )
  foot_speed_sq = torch.sum(torch.square(foot_vel_w), dim=-1)
  height_excess = torch.relu(foot_pos_w[:, :, 2] - ground_height - height_deadband)

  risk = _balance_risk(asset)
  idle_scale = torch.clamp(
    (max_idle_risk - risk) / max(max_idle_risk, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  if return_motion_relief > 0.0 and return_displacement_deadband > 0.0:
    root_displacement = torch.norm(
      asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2],
      dim=1,
    )
    return_need = torch.clamp(
      (root_displacement - return_displacement_deadband)
      / max(return_displacement_width, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    idle_scale = idle_scale * torch.clamp(
      1.0 - return_motion_relief * return_need,
      min=0.0,
      max=1.0,
    )
    env.extras["log"]["Metrics/low_risk_foot_return_need_mean"] = torch.mean(
      return_need
    )
  if motion_need_threshold > 0.0:
    motion_need = _directional_recovery_step_need(
      env,
      asset,
      foot_pos_b,
      min_reach=min_reach,
      max_reach=max_reach,
      capture_reach_gain=capture_reach_gain,
      velocity_reach_gain=velocity_reach_gain,
      direction_com_gain=direction_com_gain,
      direction_velocity_gain=direction_velocity_gain,
      direction_deadband=direction_deadband,
      sagittal_bias_gain=sagittal_bias_gain,
      lateral_suppression=lateral_suppression,
      sagittal_activation=sagittal_activation,
      risk_activation=risk_activation,
      target_velocity=target_velocity,
      need_scale=need_scale,
      dynamic_need_weight=dynamic_need_weight,
      gravity=gravity,
      min_com_height=min_com_height,
      max_capture_offset=max_capture_offset,
      risk_tilt_limit=risk_tilt_limit,
      risk_ang_vel_limit=risk_ang_vel_limit,
      risk_planar_speed_limit=risk_planar_speed_limit,
    )
    need_idle_scale = motion_need_idle_weight * torch.clamp(
      (motion_need_threshold - motion_need)
      / max(motion_need_threshold, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    idle_scale = torch.maximum(idle_scale, need_idle_scale)
    env.extras["log"]["Metrics/low_risk_foot_motion_need_mean"] = torch.mean(
      motion_need
    )
    env.extras["log"]["Metrics/low_risk_foot_need_idle_scale_mean"] = torch.mean(
      need_idle_scale
    )
  cost = torch.sum(
    airborne
    * (
      airborne_velocity_weight * foot_speed_sq
      + airborne_height_weight * torch.square(height_excess)
      + airborne_contact_weight
    ),
    dim=1,
  )
  cost += fresh_takeoff_weight * torch.sum(fresh_takeoff, dim=1)
  weighted_cost = cost * idle_scale
  env.extras["log"]["Metrics/low_risk_foot_motion_cost_mean"] = torch.mean(
    weighted_cost
  )
  env.extras["log"]["Metrics/low_risk_foot_motion_idle_scale_mean"] = torch.mean(
    idle_scale
  )
  env.extras["log"]["Metrics/low_risk_foot_airborne_frac"] = torch.mean(airborne)
  env.extras["log"]["Metrics/low_risk_foot_takeoff_frac"] = torch.mean(fresh_takeoff)
  if num_feet >= 2:
    left_airborne = airborne[:, 0]
    right_airborne = airborne[:, 1]
    left_takeoff = fresh_takeoff[:, 0]
    right_takeoff = fresh_takeoff[:, 1]
    env.extras["log"]["Metrics/foot_airborne_left_frac"] = torch.mean(left_airborne)
    env.extras["log"]["Metrics/foot_airborne_right_frac"] = torch.mean(right_airborne)
    env.extras["log"]["Metrics/foot_takeoff_left_frac"] = torch.mean(left_takeoff)
    env.extras["log"]["Metrics/foot_takeoff_right_frac"] = torch.mean(right_takeoff)
    env.extras["log"]["Metrics/foot_takeoff_balance_mean"] = torch.mean(
      left_takeoff - right_takeoff
    )
  return weighted_cost


def foot_takeoff_symmetry_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  airborne_weight: float = 0.25,
  takeoff_weight: float = 1.0,
  fresh_takeoff_window_s: float = 0.06,
  imbalance_deadband: float = 0.02,
  imbalance_scale: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize whichever foot is over-used across the current vectorized batch."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  num_feet = foot_pos_w.shape[1]
  if num_feet < 2:
    return torch.zeros(env.num_envs, device=env.device)

  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.current_air_time is not None
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  airborne = 1.0 - contact
  current_air_time = sensor_data.current_air_time[:, :num_feet]
  fresh_takeoff = airborne * (current_air_time <= fresh_takeoff_window_s).float()

  usage = takeoff_weight * fresh_takeoff + airborne_weight * airborne
  left_usage = usage[:, 0]
  right_usage = usage[:, 1]
  batch_delta = torch.mean(left_usage - right_usage).detach()
  left_pressure = torch.clamp(
    (batch_delta - imbalance_deadband) / max(imbalance_scale, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  right_pressure = torch.clamp(
    (-batch_delta - imbalance_deadband) / max(imbalance_scale, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  cost = left_pressure * left_usage + right_pressure * right_usage

  env.extras["log"]["Metrics/foot_symmetry_cost_mean"] = torch.mean(cost)
  env.extras["log"]["Metrics/foot_symmetry_usage_delta_mean"] = batch_delta
  env.extras["log"]["Metrics/foot_symmetry_left_pressure_mean"] = left_pressure
  env.extras["log"]["Metrics/foot_symmetry_right_pressure_mean"] = right_pressure
  return cost


def directional_swing_foot_choice_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  airborne_weight: float = 0.25,
  takeoff_weight: float = 1.0,
  fresh_takeoff_window_s: float = 0.06,
  need_activation: float = 0.05,
  need_width: float = 0.35,
  overused_pressure_weight: float = 0.0,
  imbalance_deadband: float = 0.02,
  imbalance_scale: float = 0.08,
  raw_lateral_activation: float = 0.18,
  lateral_activation: float = 0.55,
  lateral_dominance: float = 0.85,
  balanced_period_s: float = 3.0,
  min_reach: float = 0.30,
  max_reach: float = 0.72,
  capture_reach_gain: float = 1.05,
  velocity_reach_gain: float = 0.95,
  direction_com_gain: float = 0.60,
  direction_velocity_gain: float = 0.95,
  direction_deadband: float = 0.04,
  sagittal_bias_gain: float = 1.35,
  lateral_suppression: float = 0.65,
  sagittal_activation: float = 0.08,
  risk_activation: float = 0.12,
  target_velocity: float = 0.22,
  need_scale: float = 0.08,
  dynamic_need_weight: float = 0.35,
  gravity: float = 9.81,
  min_com_height: float = 0.30,
  max_capture_offset: float = 0.90,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize recovery swing on the wrong foot for the current recovery direction."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  num_feet = foot_pos_b.shape[1]
  if num_feet < 2:
    return torch.zeros(env.num_envs, device=env.device)

  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.current_air_time is not None
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  airborne = 1.0 - contact
  current_air_time = sensor_data.current_air_time[:, :num_feet]
  fresh_takeoff = airborne * (current_air_time <= fresh_takeoff_window_s).float()
  usage = takeoff_weight * fresh_takeoff + airborne_weight * airborne

  motion_need = _directional_recovery_step_need(
    env,
    asset,
    foot_pos_b,
    min_reach=min_reach,
    max_reach=max_reach,
    capture_reach_gain=capture_reach_gain,
    velocity_reach_gain=velocity_reach_gain,
    direction_com_gain=direction_com_gain,
    direction_velocity_gain=direction_velocity_gain,
    direction_deadband=direction_deadband,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
    risk_activation=risk_activation,
    target_velocity=target_velocity,
    need_scale=need_scale,
    dynamic_need_weight=dynamic_need_weight,
    gravity=gravity,
    min_com_height=min_com_height,
    max_capture_offset=max_capture_offset,
    risk_tilt_limit=risk_tilt_limit,
    risk_ang_vel_limit=risk_ang_vel_limit,
    risk_planar_speed_limit=risk_planar_speed_limit,
  )

  com_w = _whole_body_com_w(asset)
  rel_com_w = com_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
  capture_offset_b = root_lin_vel_b / omega.unsqueeze(1)
  capture_offset_norm = torch.norm(capture_offset_b, dim=1, keepdim=True)
  capture_offset_b = capture_offset_b * torch.clamp(
    max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
    max=1.0,
  )
  rel_capture_b = rel_com_b[:, :2] + capture_offset_b
  direction = (
    direction_com_gain * rel_capture_b
    + direction_velocity_gain * root_lin_vel_b
  )
  direction = _sagittal_biased_direction(
    direction,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
  )
  direction_norm = torch.norm(direction, dim=1)
  direction_unit = direction / torch.clamp(direction_norm.unsqueeze(1), min=1.0e-6)

  raw_abs_x = torch.abs(direction[:, 0])
  raw_abs_y = torch.abs(direction[:, 1])
  unit_abs_y = torch.abs(direction_unit[:, 1])
  lateral_preferred = (
    (raw_abs_y > raw_lateral_activation)
    & (unit_abs_y > lateral_activation)
    & (raw_abs_y > lateral_dominance * raw_abs_x)
  )
  lateral_left = direction_unit[:, 1] > 0.0

  balanced_bias = _balanced_foot_tie_bias(
    env,
    num_feet,
    foot_pos_b.dtype,
    scale=1.0,
    period_s=balanced_period_s,
  )
  balanced_left = balanced_bias[:, 0] > balanced_bias[:, 1]
  prefer_left = torch.where(lateral_preferred, lateral_left, balanced_left)
  prefer_right = ~prefer_left
  preferred = torch.stack((prefer_left, prefer_right), dim=1).float()

  need_gate = torch.clamp(
    (motion_need - need_activation) / max(need_width, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  nonpreferred_usage = torch.sum(usage * (1.0 - preferred), dim=1)
  if overused_pressure_weight > 0.0:
    left_usage = usage[:, 0]
    right_usage = usage[:, 1]
    batch_delta = torch.mean(left_usage - right_usage).detach()
    left_overused = torch.clamp(
      (batch_delta - imbalance_deadband) / max(imbalance_scale, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    right_overused = torch.clamp(
      (-batch_delta - imbalance_deadband) / max(imbalance_scale, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    overused = torch.stack(
      (
        left_overused.expand_as(left_usage),
        right_overused.expand_as(right_usage),
      ),
      dim=1,
    )
    overused_nonpreferred_usage = torch.sum(
      usage * overused * (1.0 - preferred), dim=1
    )
    nonpreferred_usage = (
      nonpreferred_usage
      + overused_pressure_weight * overused_nonpreferred_usage
    )
    env.extras["log"]["Metrics/directional_foot_choice_usage_delta_mean"] = (
      batch_delta
    )
    env.extras["log"]["Metrics/directional_foot_choice_left_overused_mean"] = (
      left_overused
    )
    env.extras["log"]["Metrics/directional_foot_choice_right_overused_mean"] = (
      right_overused
    )
    env.extras["log"][
      "Metrics/directional_foot_choice_overused_nonpref_usage_mean"
    ] = torch.mean(overused_nonpreferred_usage)
  cost = need_gate * nonpreferred_usage

  env.extras["log"]["Metrics/directional_foot_choice_cost_mean"] = torch.mean(cost)
  env.extras["log"]["Metrics/directional_foot_choice_need_mean"] = torch.mean(
    motion_need
  )
  env.extras["log"]["Metrics/directional_foot_choice_gate_mean"] = torch.mean(
    need_gate
  )
  env.extras["log"]["Metrics/directional_foot_choice_lateral_frac"] = torch.mean(
    lateral_preferred.float()
  )
  env.extras["log"]["Metrics/directional_foot_choice_left_pref_frac"] = torch.mean(
    prefer_left.float()
  )
  env.extras["log"]["Metrics/directional_foot_choice_nonpref_usage_mean"] = torch.mean(
    nonpreferred_usage
  )
  return cost


def underused_recovery_foot_bonus(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  airborne_weight: float = 0.15,
  takeoff_weight: float = 1.0,
  fresh_takeoff_window_s: float = 0.06,
  imbalance_deadband: float = 0.02,
  imbalance_scale: float = 0.08,
  need_activation: float = 0.05,
  need_width: float = 0.35,
  raw_lateral_activation: float = 0.18,
  lateral_activation: float = 0.55,
  lateral_dominance: float = 0.85,
  balanced_period_s: float = 3.0,
  min_reach: float = 0.30,
  max_reach: float = 0.72,
  capture_reach_gain: float = 1.05,
  velocity_reach_gain: float = 0.95,
  direction_com_gain: float = 0.60,
  direction_velocity_gain: float = 0.95,
  direction_deadband: float = 0.04,
  sagittal_bias_gain: float = 1.35,
  lateral_suppression: float = 0.65,
  sagittal_activation: float = 0.08,
  risk_activation: float = 0.12,
  target_velocity: float = 0.22,
  need_scale: float = 0.08,
  dynamic_need_weight: float = 0.35,
  gravity: float = 9.81,
  min_com_height: float = 0.30,
  max_capture_offset: float = 0.90,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward recovery steps with the under-used foot when a step is needed."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  num_feet = foot_pos_b.shape[1]
  if num_feet < 2:
    return torch.zeros(env.num_envs, device=env.device)

  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.current_air_time is not None
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  airborne = 1.0 - contact
  current_air_time = sensor_data.current_air_time[:, :num_feet]
  fresh_takeoff = airborne * (current_air_time <= fresh_takeoff_window_s).float()
  usage = takeoff_weight * fresh_takeoff + airborne_weight * airborne

  motion_need = _directional_recovery_step_need(
    env,
    asset,
    foot_pos_b,
    min_reach=min_reach,
    max_reach=max_reach,
    capture_reach_gain=capture_reach_gain,
    velocity_reach_gain=velocity_reach_gain,
    direction_com_gain=direction_com_gain,
    direction_velocity_gain=direction_velocity_gain,
    direction_deadband=direction_deadband,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
    risk_activation=risk_activation,
    target_velocity=target_velocity,
    need_scale=need_scale,
    dynamic_need_weight=dynamic_need_weight,
    gravity=gravity,
    min_com_height=min_com_height,
    max_capture_offset=max_capture_offset,
    risk_tilt_limit=risk_tilt_limit,
    risk_ang_vel_limit=risk_ang_vel_limit,
    risk_planar_speed_limit=risk_planar_speed_limit,
  )

  com_w = _whole_body_com_w(asset)
  rel_com_w = com_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
  capture_offset_b = root_lin_vel_b / omega.unsqueeze(1)
  capture_offset_norm = torch.norm(capture_offset_b, dim=1, keepdim=True)
  capture_offset_b = capture_offset_b * torch.clamp(
    max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
    max=1.0,
  )
  rel_capture_b = rel_com_b[:, :2] + capture_offset_b
  direction = (
    direction_com_gain * rel_capture_b
    + direction_velocity_gain * root_lin_vel_b
  )
  direction = _sagittal_biased_direction(
    direction,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
  )
  direction_norm = torch.norm(direction, dim=1)
  direction_unit = direction / torch.clamp(direction_norm.unsqueeze(1), min=1.0e-6)

  raw_abs_x = torch.abs(direction[:, 0])
  raw_abs_y = torch.abs(direction[:, 1])
  unit_abs_y = torch.abs(direction_unit[:, 1])
  lateral_preferred = (
    (raw_abs_y > raw_lateral_activation)
    & (unit_abs_y > lateral_activation)
    & (raw_abs_y > lateral_dominance * raw_abs_x)
  )
  lateral_left = direction_unit[:, 1] > 0.0

  balanced_bias = _balanced_foot_tie_bias(
    env,
    num_feet,
    foot_pos_b.dtype,
    scale=1.0,
    period_s=balanced_period_s,
  )
  balanced_left = balanced_bias[:, 0] > balanced_bias[:, 1]
  prefer_left = torch.where(lateral_preferred, lateral_left, balanced_left)
  prefer_right = ~prefer_left
  preferred = torch.stack((prefer_left, prefer_right), dim=1).float()

  left_usage = usage[:, 0]
  right_usage = usage[:, 1]
  batch_delta = torch.mean(left_usage - right_usage).detach()
  left_overused = torch.clamp(
    (batch_delta - imbalance_deadband) / max(imbalance_scale, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  right_overused = torch.clamp(
    (-batch_delta - imbalance_deadband) / max(imbalance_scale, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  underused = torch.stack(
    (
      right_overused.expand_as(left_usage),
      left_overused.expand_as(right_usage),
    ),
    dim=1,
  )
  need_gate = torch.clamp(
    (motion_need - need_activation) / max(need_width, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  selected_usage = torch.sum(usage * underused * preferred, dim=1)
  bonus = need_gate * selected_usage

  env.extras["log"]["Metrics/underused_recovery_foot_bonus_mean"] = torch.mean(bonus)
  env.extras["log"]["Metrics/underused_recovery_foot_need_mean"] = torch.mean(
    motion_need
  )
  env.extras["log"]["Metrics/underused_recovery_foot_gate_mean"] = torch.mean(
    need_gate
  )
  env.extras["log"]["Metrics/underused_recovery_foot_usage_delta_mean"] = batch_delta
  env.extras["log"]["Metrics/underused_recovery_foot_left_overused_mean"] = (
    left_overused
  )
  env.extras["log"]["Metrics/underused_recovery_foot_right_overused_mean"] = (
    right_overused
  )
  env.extras["log"]["Metrics/underused_recovery_foot_selected_usage_mean"] = (
    torch.mean(selected_usage)
  )
  return bonus


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
  soft_max_width: float = 0.0,
  max_width: float = 0.65,
  soft_overwidth_weight: float = 0.0,
  soft_overwidth_risk_activation: float = 0.15,
  risk_split_gain: float = 0.40,
  split_velocity_gain: float = 0.0,
  risk_min_split: float = 0.0,
  width_lateral_velocity_gain: float = 0.0,
  max_split: float = 0.65,
  foot_crossing_margin: float = 0.02,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Risk-gated stance width and fore-aft split shaping.

  COM displacement alone is too late for force recovery: during a shove the COM
  can still be close to the pelvis while the capture point is already outside
  support. Root-frame planar velocity therefore also increases the desired
  fore-aft split and lateral stance width.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  if foot_pos_b.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  left = foot_pos_b[:, 0, :]
  right = foot_pos_b[:, 1, :]
  width = torch.abs(left[:, 1] - right[:, 1])
  split = torch.abs(left[:, 0] - right[:, 0])
  risk = _balance_risk(asset)
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]

  recovery_width = risk_width + width_lateral_velocity_gain * torch.abs(
    root_lin_vel_b[:, 1]
  )
  target_min_width = nominal_width * (1.0 - risk) + recovery_width * risk
  target_min_width = torch.clamp(target_min_width, max=max_width)
  width_low_cost = torch.relu(target_min_width - width).square()
  width_high_cost = torch.relu(width - max_width).square()
  soft_overwidth_cost = torch.zeros_like(width)
  if soft_max_width > 0.0 and soft_overwidth_weight > 0.0:
    overwidth_risk = torch.clamp(
      (risk - soft_overwidth_risk_activation)
      / max(1.0 - soft_overwidth_risk_activation, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    soft_overwidth_cost = (
      soft_overwidth_weight
      * overwidth_risk
      * torch.relu(width - soft_max_width).square()
    )

  com_xy = _whole_body_com_w(asset)[:, :2]
  rel_com_w = com_xy - asset.data.root_link_pos_w[:, :2]
  root_quat_w = asset.data.root_link_quat_w
  rel_com_b = quat_apply_inverse(
    root_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  desired_split_from_com = torch.abs(rel_com_b[:, 0]) * risk_split_gain
  desired_split_from_velocity = (
    risk * torch.abs(root_lin_vel_b[:, 0]) * split_velocity_gain
  )
  desired_split_from_risk = risk * risk_min_split
  desired_split = torch.maximum(
    desired_split_from_com,
    torch.maximum(desired_split_from_velocity, desired_split_from_risk),
  )
  desired_split = torch.clamp(desired_split, max=max_split)
  split_low_cost = risk * torch.relu(desired_split - split).square()
  split_high_cost = torch.relu(split - max_split).square()

  crossing_cost = torch.relu(foot_crossing_margin - (left[:, 1] - right[:, 1])).square()
  env.extras["log"]["Metrics/stance_width_mean"] = torch.mean(width)
  env.extras["log"]["Metrics/stance_target_width_mean"] = torch.mean(target_min_width)
  env.extras["log"]["Metrics/stance_fore_aft_split_mean"] = torch.mean(split)
  env.extras["log"]["Metrics/stance_target_fore_aft_split_mean"] = torch.mean(
    desired_split
  )
  env.extras["log"]["Metrics/stance_soft_overwidth_mean"] = torch.mean(
    soft_overwidth_cost
  )
  return (
    width_low_cost
    + width_high_cost
    + soft_overwidth_cost
    + split_low_cost
    + split_high_cost
    + crossing_cost
  )


class capture_point_reach_penalty:
  """Penalize missing leading-foot reach during high-risk recovery.

  The support-margin terms penalize an unrecoverable capture point, but they do
  not directly say which foot should move. This term adds that directional
  signal: when the capture point or planar velocity escapes forward/backward or
  sideways in the pelvis frame, the leading foot should extend in that
  direction.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev_com_xy = torch.zeros(env.num_envs, 2, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    min_fore_aft_reach: float = 0.14,
    max_fore_aft_reach: float = 0.55,
    capture_fore_aft_gain: float = 0.90,
    velocity_fore_aft_gain: float = 0.60,
    min_lateral_reach: float = 0.18,
    max_lateral_reach: float = 0.45,
    capture_lateral_gain: float = 0.75,
    velocity_lateral_gain: float = 0.45,
    direction_velocity_gain: float = 0.20,
    direction_deadband: float = 0.025,
    lateral_weight: float = 0.60,
    gravity: float = 9.81,
    min_com_height: float = 0.30,
    max_capture_offset: float = 0.90,
    risk_tilt_limit: float = 0.35,
    risk_ang_vel_limit: float = 1.5,
    risk_planar_speed_limit: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
    if foot_pos_b.shape[1] < 2:
      return torch.zeros(env.num_envs, device=env.device)

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
    rel_capture_w = torch.cat(
      (
        capture_xy - asset.data.root_link_pos_w[:, :2],
        torch.zeros(env.num_envs, 1, device=env.device),
      ),
      dim=1,
    )
    rel_capture_b = quat_apply_inverse(asset.data.root_link_quat_w, rel_capture_w)
    root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
    risk = _balance_risk(
      asset,
      tilt_limit=risk_tilt_limit,
      ang_vel_limit=risk_ang_vel_limit,
      planar_speed_limit=risk_planar_speed_limit,
    )

    direction = rel_capture_b[:, :2] + direction_velocity_gain * root_lin_vel_b
    foot_x = foot_pos_b[:, :, 0]
    foot_y = foot_pos_b[:, :, 1]
    leading_x = torch.where(
      direction[:, 0] >= 0.0,
      torch.max(foot_x, dim=1).values,
      -torch.min(foot_x, dim=1).values,
    )
    leading_y = torch.where(
      direction[:, 1] >= 0.0,
      torch.max(foot_y, dim=1).values,
      -torch.min(foot_y, dim=1).values,
    )

    fore_aft_need = torch.maximum(
      torch.abs(rel_capture_b[:, 0]) * capture_fore_aft_gain,
      torch.abs(root_lin_vel_b[:, 0]) * velocity_fore_aft_gain,
    )
    lateral_need = torch.maximum(
      torch.abs(rel_capture_b[:, 1]) * capture_lateral_gain,
      torch.abs(root_lin_vel_b[:, 1]) * velocity_lateral_gain,
    )
    fore_aft_active = (torch.abs(direction[:, 0]) > direction_deadband).float()
    lateral_active = (torch.abs(direction[:, 1]) > direction_deadband).float()
    target_fore_aft = fore_aft_active * torch.clamp(
      min_fore_aft_reach * risk + fore_aft_need,
      max=max_fore_aft_reach,
    )
    target_lateral = lateral_active * torch.clamp(
      min_lateral_reach * risk + lateral_need,
      max=max_lateral_reach,
    )

    fore_aft_cost = risk * torch.relu(target_fore_aft - leading_x).square()
    lateral_cost = risk * torch.relu(target_lateral - leading_y).square()
    cost = fore_aft_cost + lateral_weight * lateral_cost

    env.extras["log"]["Metrics/recovery_reach_target_x_mean"] = torch.mean(
      target_fore_aft
    )
    env.extras["log"]["Metrics/recovery_reach_actual_x_mean"] = torch.mean(leading_x)
    env.extras["log"]["Metrics/recovery_reach_target_y_mean"] = torch.mean(
      target_lateral
    )
    env.extras["log"]["Metrics/recovery_reach_actual_y_mean"] = torch.mean(leading_y)
    env.extras["log"]["Metrics/recovery_reach_penalty_mean"] = torch.mean(cost)
    return cost


def recovery_step_clearance_penalty(
  env: ManagerBasedRlEnv,
  min_fore_aft_reach: float = 0.18,
  max_fore_aft_reach: float = 0.60,
  com_fore_aft_gain: float = 0.80,
  velocity_fore_aft_gain: float = 0.80,
  min_lateral_reach: float = 0.16,
  max_lateral_reach: float = 0.42,
  com_lateral_gain: float = 0.55,
  velocity_lateral_gain: float = 0.45,
  direction_com_gain: float = 0.50,
  direction_velocity_gain: float = 0.40,
  direction_deadband: float = 0.025,
  foot_tie_break_scale: float = 0.0,
  foot_tie_break_period_s: float = 4.0,
  clearance_height: float = 0.055,
  ground_height: float = 0.0,
  lateral_weight: float = 0.50,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Encourage an actual swing-foot clearance when recovery reach is missing."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  if foot_pos_b.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  com_xy = _whole_body_com_w(asset)[:, :2]
  rel_com_w = com_xy - asset.data.root_link_pos_w[:, :2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  risk = _balance_risk(
    asset,
    tilt_limit=risk_tilt_limit,
    ang_vel_limit=risk_ang_vel_limit,
    planar_speed_limit=risk_planar_speed_limit,
  )

  direction = (
    direction_com_gain * rel_com_b[:, :2]
    + direction_velocity_gain * root_lin_vel_b
  )
  foot_x = foot_pos_b[:, :, 0]
  foot_y = foot_pos_b[:, :, 1]

  fore_forward = direction[:, 0] >= 0.0
  lateral_left = direction[:, 1] >= 0.0
  leading_x, leading_x_idx = _signed_extreme_with_tie_bias(
    env,
    foot_x,
    fore_forward,
    scale=foot_tie_break_scale,
    period_s=foot_tie_break_period_s,
  )
  leading_y, leading_y_idx = _signed_extreme_with_tie_bias(
    env,
    foot_y,
    lateral_left,
    scale=foot_tie_break_scale,
    period_s=foot_tie_break_period_s,
  )

  foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - ground_height
  leading_x_height = torch.gather(foot_height, 1, leading_x_idx.unsqueeze(1)).squeeze(1)
  leading_y_height = torch.gather(foot_height, 1, leading_y_idx.unsqueeze(1)).squeeze(1)

  fore_aft_need = torch.maximum(
    torch.abs(rel_com_b[:, 0]) * com_fore_aft_gain,
    torch.abs(root_lin_vel_b[:, 0]) * velocity_fore_aft_gain,
  )
  lateral_need = torch.maximum(
    torch.abs(rel_com_b[:, 1]) * com_lateral_gain,
    torch.abs(root_lin_vel_b[:, 1]) * velocity_lateral_gain,
  )
  fore_aft_active = (torch.abs(direction[:, 0]) > direction_deadband).float()
  lateral_active = (torch.abs(direction[:, 1]) > direction_deadband).float()
  target_fore_aft = fore_aft_active * torch.clamp(
    min_fore_aft_reach * risk + fore_aft_need,
    max=max_fore_aft_reach,
  )
  target_lateral = lateral_active * torch.clamp(
    min_lateral_reach * risk + lateral_need,
    max=max_lateral_reach,
  )

  fore_deficit = risk * torch.relu(target_fore_aft - leading_x)
  lateral_deficit = risk * torch.relu(target_lateral - leading_y)
  fore_clearance_deficit = torch.relu(clearance_height - leading_x_height)
  lateral_clearance_deficit = torch.relu(clearance_height - leading_y_height)
  cost = fore_deficit * fore_clearance_deficit
  cost = cost + lateral_weight * lateral_deficit * lateral_clearance_deficit

  env.extras["log"]["Metrics/recovery_step_deficit_x_mean"] = torch.mean(fore_deficit)
  env.extras["log"]["Metrics/recovery_step_height_x_mean"] = torch.mean(
    leading_x_height
  )
  env.extras["log"]["Metrics/recovery_step_deficit_y_mean"] = torch.mean(
    lateral_deficit
  )
  env.extras["log"]["Metrics/recovery_step_height_y_mean"] = torch.mean(
    leading_y_height
  )
  env.extras["log"]["Metrics/recovery_step_clearance_penalty_mean"] = torch.mean(cost)
  return cost


def recovery_step_velocity_penalty(
  env: ManagerBasedRlEnv,
  min_fore_aft_reach: float = 0.18,
  max_fore_aft_reach: float = 0.60,
  com_fore_aft_gain: float = 0.80,
  velocity_fore_aft_gain: float = 0.80,
  min_lateral_reach: float = 0.16,
  max_lateral_reach: float = 0.42,
  com_lateral_gain: float = 0.55,
  velocity_lateral_gain: float = 0.45,
  direction_com_gain: float = 0.50,
  direction_velocity_gain: float = 0.40,
  direction_deadband: float = 0.025,
  foot_tie_break_scale: float = 0.0,
  foot_tie_break_period_s: float = 4.0,
  min_step_velocity: float = 0.20,
  velocity_target_gain: float = 0.60,
  max_step_velocity: float = 0.55,
  lateral_weight: float = 0.50,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Encourage the recovery foot to move in the direction of missing reach."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  if foot_pos_b.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  foot_vel_w = (
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    - asset.data.root_link_lin_vel_w[:, None, :]
  )
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_vel_w.shape[1], -1
  )
  foot_vel_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4), foot_vel_w.reshape(-1, 3)
  )
  foot_vel_b = foot_vel_b.reshape(env.num_envs, -1, 3)

  com_xy = _whole_body_com_w(asset)[:, :2]
  rel_com_w = com_xy - asset.data.root_link_pos_w[:, :2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  risk = _balance_risk(
    asset,
    tilt_limit=risk_tilt_limit,
    ang_vel_limit=risk_ang_vel_limit,
    planar_speed_limit=risk_planar_speed_limit,
  )

  direction = (
    direction_com_gain * rel_com_b[:, :2]
    + direction_velocity_gain * root_lin_vel_b
  )
  foot_x = foot_pos_b[:, :, 0]
  foot_y = foot_pos_b[:, :, 1]

  fore_forward = direction[:, 0] >= 0.0
  lateral_left = direction[:, 1] >= 0.0
  leading_x, leading_x_idx = _signed_extreme_with_tie_bias(
    env,
    foot_x,
    fore_forward,
    scale=foot_tie_break_scale,
    period_s=foot_tie_break_period_s,
  )
  leading_y, leading_y_idx = _signed_extreme_with_tie_bias(
    env,
    foot_y,
    lateral_left,
    scale=foot_tie_break_scale,
    period_s=foot_tie_break_period_s,
  )

  selected_x_vel = torch.gather(
    foot_vel_b[:, :, 0], 1, leading_x_idx.unsqueeze(1)
  ).squeeze(1)
  selected_y_vel = torch.gather(
    foot_vel_b[:, :, 1], 1, leading_y_idx.unsqueeze(1)
  ).squeeze(1)
  leading_x_vel = torch.where(fore_forward, selected_x_vel, -selected_x_vel)
  leading_y_vel = torch.where(lateral_left, selected_y_vel, -selected_y_vel)

  fore_aft_need = torch.maximum(
    torch.abs(rel_com_b[:, 0]) * com_fore_aft_gain,
    torch.abs(root_lin_vel_b[:, 0]) * velocity_fore_aft_gain,
  )
  lateral_need = torch.maximum(
    torch.abs(rel_com_b[:, 1]) * com_lateral_gain,
    torch.abs(root_lin_vel_b[:, 1]) * velocity_lateral_gain,
  )
  fore_aft_active = (torch.abs(direction[:, 0]) > direction_deadband).float()
  lateral_active = (torch.abs(direction[:, 1]) > direction_deadband).float()
  target_fore_aft = fore_aft_active * torch.clamp(
    min_fore_aft_reach * risk + fore_aft_need,
    max=max_fore_aft_reach,
  )
  target_lateral = lateral_active * torch.clamp(
    min_lateral_reach * risk + lateral_need,
    max=max_lateral_reach,
  )

  fore_deficit = risk * torch.relu(target_fore_aft - leading_x)
  lateral_deficit = risk * torch.relu(target_lateral - leading_y)
  target_fore_velocity = torch.clamp(
    min_step_velocity + velocity_target_gain * torch.abs(root_lin_vel_b[:, 0]),
    max=max_step_velocity,
  )
  target_lateral_velocity = torch.clamp(
    min_step_velocity + velocity_target_gain * torch.abs(root_lin_vel_b[:, 1]),
    max=max_step_velocity,
  )
  fore_cost = fore_deficit * torch.relu(target_fore_velocity - leading_x_vel)
  lateral_cost = lateral_deficit * torch.relu(target_lateral_velocity - leading_y_vel)
  cost = fore_cost + lateral_weight * lateral_cost

  env.extras["log"]["Metrics/recovery_step_velocity_x_mean"] = torch.mean(
    leading_x_vel
  )
  env.extras["log"]["Metrics/recovery_step_velocity_y_mean"] = torch.mean(
    leading_y_vel
  )
  env.extras["log"]["Metrics/recovery_step_velocity_penalty_mean"] = torch.mean(cost)
  return cost


def recovery_step_contact_phase_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_reach: float = 0.28,
  max_reach: float = 0.62,
  com_reach_gain: float = 0.75,
  velocity_reach_gain: float = 0.55,
  capture_reach_gain: float = 0.80,
  direction_com_gain: float = 0.45,
  direction_velocity_gain: float = 0.55,
  direction_deadband: float = 0.04,
  sagittal_bias_gain: float = 1.0,
  lateral_suppression: float = 0.0,
  sagittal_activation: float = 0.08,
  foot_tie_break_scale: float = 0.0,
  foot_tie_break_period_s: float = 4.0,
  risk_activation: float = 0.20,
  clearance_height: float = 0.045,
  min_step_velocity: float = 0.12,
  recontact_margin: float = 0.03,
  support_contact_weight: float = 1.5,
  stuck_contact_weight: float = 0.35,
  clearance_weight: float = 1.0,
  velocity_weight: float = 0.7,
  recontact_weight: float = 0.6,
  no_support_weight: float = 2.0,
  no_swing_weight: float = 0.65,
  need_scale: float = 0.10,
  dynamic_need_weight: float = 0.30,
  ground_height: float = 0.0,
  gravity: float = 9.81,
  min_com_height: float = 0.30,
  max_capture_offset: float = 0.90,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Shape a support-swing-recontact recovery phase under high balance risk."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  if foot_pos_b.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  num_feet = foot_pos_b.shape[1]
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  contact_count = torch.sum(contact, dim=1)

  com_xy = _whole_body_com_w(asset)[:, :2]
  rel_com_w = com_xy - asset.data.root_link_pos_w[:, :2]
  com_height = _whole_body_com_w(asset)[:, 2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  omega = torch.sqrt(gravity / torch.clamp(com_height, min=min_com_height))
  capture_offset_b = root_lin_vel_b / omega.unsqueeze(1)
  capture_offset_norm = torch.norm(capture_offset_b, dim=1, keepdim=True)
  capture_offset_b = capture_offset_b * torch.clamp(
    max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
    max=1.0,
  )
  rel_capture_b = rel_com_b[:, :2] + capture_offset_b
  risk = _balance_risk(
    asset,
    tilt_limit=risk_tilt_limit,
    ang_vel_limit=risk_ang_vel_limit,
    planar_speed_limit=risk_planar_speed_limit,
  )

  direction = (
    direction_com_gain * rel_capture_b
    + direction_velocity_gain * root_lin_vel_b
  )
  direction = _sagittal_biased_direction(
    direction,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
  )
  direction_norm = torch.norm(direction, dim=1)
  risk_scale = torch.clamp(
    (risk - risk_activation) / max(1.0 - risk_activation, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  active = risk_scale * (direction_norm > direction_deadband).float()
  direction_unit = direction / torch.clamp(direction_norm.unsqueeze(1), min=1.0e-6)
  tie_bias = _balanced_foot_tie_bias(
    env,
    num_feet,
    foot_pos_b.dtype,
    scale=foot_tie_break_scale,
    period_s=foot_tie_break_period_s,
  )

  foot_proj = torch.sum(foot_pos_b[:, :, :2] * direction_unit[:, None, :], dim=-1)
  _, leading_idx = torch.max(foot_proj + tie_bias, dim=1)
  leading_reach = torch.gather(foot_proj, 1, leading_idx.unsqueeze(1)).squeeze(1)

  foot_vel_w = (
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    - asset.data.root_link_lin_vel_w[:, None, :]
  )
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_vel_w.shape[1], -1
  )
  foot_vel_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4), foot_vel_w.reshape(-1, 3)
  ).reshape(env.num_envs, num_feet, 3)
  foot_vel_proj = torch.sum(foot_vel_b[:, :, :2] * direction_unit[:, None, :], dim=-1)

  foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - ground_height

  com_need = torch.abs(torch.sum(rel_com_b[:, :2] * direction_unit, dim=1))
  capture_need = torch.abs(torch.sum(rel_capture_b * direction_unit, dim=1))
  velocity_need = torch.relu(torch.sum(root_lin_vel_b * direction_unit, dim=1))
  target_reach = torch.clamp(
    min_reach
    + com_reach_gain * com_need
    + capture_reach_gain * capture_need
    + velocity_reach_gain * velocity_need,
    max=max_reach,
  )
  raw_reach_gap = torch.relu(target_reach - leading_reach)
  dynamic_need = torch.clamp(
    (
      torch.clamp(capture_need / max(min_reach, 1.0e-6), max=1.0)
      + torch.clamp(velocity_need / max(min_step_velocity, 1.0e-6), max=1.0)
    )
    * 0.5,
    max=1.0,
  )
  step_need = active * torch.clamp(
    raw_reach_gap / max(need_scale, 1.0e-6)
    + dynamic_need_weight * dynamic_need,
    max=1.0,
  )
  reach_deficit = active * raw_reach_gap
  phase_need = torch.maximum(reach_deficit, 0.30 * step_need)
  needs_swing = (step_need > 1.0e-4).float()

  support_if_swing = contact_count.unsqueeze(1) - contact
  has_support_if_swing = (support_if_swing >= 1.0).float()
  clearance_score = torch.clamp(
    foot_height / max(clearance_height, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  velocity_score = torch.clamp(
    torch.relu(foot_vel_proj) / max(min_step_velocity, 1.0e-6),
    max=1.0,
  )
  lift_velocity_score = torch.clamp(
    torch.relu(foot_vel_b[:, :, 2]) / max(min_step_velocity, 1.0e-6),
    max=1.0,
  )
  decontact_score = 1.0 - contact
  movement_signal = torch.clamp(
    0.50 * velocity_score + 0.30 * lift_velocity_score + 0.20 * clearance_score,
    max=1.0,
  )
  swing_signal = torch.clamp(
    0.45 * decontact_score + 0.35 * lift_velocity_score + 0.20 * clearance_score,
    max=1.0,
  )
  candidate_signal = has_support_if_swing * movement_signal * swing_signal
  _, selected_idx = torch.max(candidate_signal + tie_bias, dim=1)
  best_signal = torch.gather(
    candidate_signal, 1, selected_idx.unsqueeze(1)
  ).squeeze(1)
  selected_reach = torch.gather(foot_proj, 1, selected_idx.unsqueeze(1)).squeeze(1)
  selected_contact = torch.gather(contact, 1, selected_idx.unsqueeze(1)).squeeze(1)
  support_contact_count = torch.gather(
    support_if_swing, 1, selected_idx.unsqueeze(1)
  ).squeeze(1)
  selected_height = torch.gather(foot_height, 1, selected_idx.unsqueeze(1)).squeeze(1)
  selected_velocity = torch.gather(
    foot_vel_proj, 1, selected_idx.unsqueeze(1)
  ).squeeze(1)

  reached_contact = torch.any(
    (foot_proj >= (target_reach - recontact_margin).unsqueeze(1))
    & (contact > 0.5),
    dim=1,
  ).float()
  reached_airborne = torch.any(
    (foot_proj >= (target_reach - recontact_margin).unsqueeze(1))
    & (contact <= 0.5),
    dim=1,
  ).float()

  initiation_cost = phase_need * torch.relu(0.45 - best_signal)
  support_cost = needs_swing * torch.relu(1.0 - support_contact_count)
  recontact_cost = active * reached_airborne * (1.0 - reached_contact)
  no_support_cost = active * (contact_count <= 0.0).float()
  supported_swing = torch.max((1.0 - contact) * has_support_if_swing, dim=1).values
  no_swing_cost = step_need * (1.0 - supported_swing)

  cost = (
    clearance_weight * initiation_cost
    + velocity_weight * initiation_cost
    + stuck_contact_weight * initiation_cost
    + support_contact_weight * support_cost
    + recontact_weight * recontact_cost
    + no_support_weight * no_support_cost
    + no_swing_weight * no_swing_cost
  )

  env.extras["log"]["Metrics/recovery_phase_target_reach_mean"] = torch.mean(
    target_reach
  )
  env.extras["log"]["Metrics/recovery_phase_actual_reach_mean"] = torch.mean(
    leading_reach
  )
  env.extras["log"]["Metrics/recovery_phase_candidate_reach_mean"] = torch.mean(
    selected_reach
  )
  env.extras["log"]["Metrics/recovery_phase_deficit_mean"] = torch.mean(reach_deficit)
  env.extras["log"]["Metrics/recovery_phase_step_need_mean"] = torch.mean(step_need)
  env.extras["log"]["Metrics/recovery_phase_active_mean"] = torch.mean(active)
  env.extras["log"]["Metrics/recovery_phase_selected_contact_mean"] = torch.mean(
    selected_contact
  )
  env.extras["log"]["Metrics/recovery_phase_support_contact_mean"] = torch.mean(
    support_contact_count
  )
  env.extras["log"]["Metrics/recovery_phase_selected_height_mean"] = torch.mean(
    selected_height
  )
  env.extras["log"]["Metrics/recovery_phase_selected_velocity_mean"] = torch.mean(
    selected_velocity
  )
  env.extras["log"]["Metrics/recovery_phase_reached_frac"] = torch.mean(
    reached_contact
  )
  env.extras["log"]["Metrics/recovery_phase_supported_swing_frac"] = torch.mean(
    supported_swing
  )
  env.extras["log"]["Metrics/recovery_phase_no_swing_cost_mean"] = torch.mean(
    no_swing_cost
  )
  env.extras["log"]["Metrics/recovery_phase_penalty_mean"] = torch.mean(cost)
  env.extras["log"]["Metrics/recovery_direction_x_abs_mean"] = torch.mean(
    torch.abs(direction[:, 0])
  )
  env.extras["log"]["Metrics/recovery_direction_y_abs_mean"] = torch.mean(
    torch.abs(direction[:, 1])
  )
  return cost


def recovery_swing_bonus(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_reach: float = 0.24,
  max_reach: float = 0.62,
  capture_reach_gain: float = 0.90,
  velocity_reach_gain: float = 0.70,
  direction_com_gain: float = 0.60,
  direction_velocity_gain: float = 0.75,
  direction_deadband: float = 0.04,
  sagittal_bias_gain: float = 1.0,
  lateral_suppression: float = 0.0,
  sagittal_activation: float = 0.08,
  foot_tie_break_scale: float = 0.0,
  foot_tie_break_period_s: float = 4.0,
  risk_activation: float = 0.12,
  target_clearance: float = 0.06,
  target_velocity: float = 0.22,
  target_lift_velocity: float = 0.18,
  need_scale: float = 0.10,
  dynamic_need_weight: float = 0.35,
  completion_progress_power: float = 2.0,
  ground_height: float = 0.0,
  gravity: float = 9.81,
  min_com_height: float = 0.30,
  max_capture_offset: float = 0.90,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward supported swing-foot lift and progress during recovery."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  if foot_pos_b.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  num_feet = foot_pos_b.shape[1]
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  contact_count = torch.sum(contact, dim=1)

  com_w = _whole_body_com_w(asset)
  rel_com_w = com_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
  capture_offset_b = root_lin_vel_b / omega.unsqueeze(1)
  capture_offset_norm = torch.norm(capture_offset_b, dim=1, keepdim=True)
  capture_offset_b = capture_offset_b * torch.clamp(
    max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
    max=1.0,
  )
  rel_capture_b = rel_com_b[:, :2] + capture_offset_b
  risk = _balance_risk(
    asset,
    tilt_limit=risk_tilt_limit,
    ang_vel_limit=risk_ang_vel_limit,
    planar_speed_limit=risk_planar_speed_limit,
  )

  direction = (
    direction_com_gain * rel_capture_b
    + direction_velocity_gain * root_lin_vel_b
  )
  direction = _sagittal_biased_direction(
    direction,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
  )
  direction_norm = torch.norm(direction, dim=1)
  risk_scale = torch.clamp(
    (risk - risk_activation) / max(1.0 - risk_activation, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  active = risk_scale * (direction_norm > direction_deadband).float()
  direction_unit = direction / torch.clamp(direction_norm.unsqueeze(1), min=1.0e-6)
  tie_bias = _balanced_foot_tie_bias(
    env,
    num_feet,
    foot_pos_b.dtype,
    scale=foot_tie_break_scale,
    period_s=foot_tie_break_period_s,
  )

  foot_proj = torch.sum(foot_pos_b[:, :, :2] * direction_unit[:, None, :], dim=-1)
  _, leading_idx = torch.max(foot_proj + tie_bias, dim=1)
  leading_reach = torch.gather(foot_proj, 1, leading_idx.unsqueeze(1)).squeeze(1)

  foot_vel_w = (
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    - asset.data.root_link_lin_vel_w[:, None, :]
  )
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_vel_w.shape[1], -1
  )
  foot_vel_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4), foot_vel_w.reshape(-1, 3)
  ).reshape(env.num_envs, num_feet, 3)
  foot_vel_proj = torch.sum(foot_vel_b[:, :, :2] * direction_unit[:, None, :], dim=-1)

  foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - ground_height

  capture_need = torch.abs(torch.sum(rel_capture_b * direction_unit, dim=1))
  velocity_need = torch.relu(torch.sum(root_lin_vel_b * direction_unit, dim=1))
  target_reach = torch.clamp(
    min_reach + capture_reach_gain * capture_need + velocity_reach_gain * velocity_need,
    max=max_reach,
  )
  raw_reach_gap = torch.relu(target_reach - leading_reach)
  dynamic_need = torch.clamp(
    (
      torch.clamp(capture_need / max(min_reach, 1.0e-6), max=1.0)
      + torch.clamp(velocity_need / max(target_velocity, 1.0e-6), max=1.0)
    )
    * 0.5,
    max=1.0,
  )
  step_need = active * torch.clamp(
    raw_reach_gap / max(need_scale, 1.0e-6)
    + dynamic_need_weight * dynamic_need,
    max=1.0,
  )
  deficit = active * raw_reach_gap
  support_if_swing = contact_count.unsqueeze(1) - contact
  has_support_if_swing = (support_if_swing >= 1.0).float()
  clearance_score = torch.clamp(
    foot_height / max(target_clearance, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  velocity_score = torch.clamp(
    torch.relu(foot_vel_proj) / max(target_velocity, 1.0e-6),
    max=1.0,
  )
  lift_velocity_score = torch.clamp(
    torch.relu(foot_vel_b[:, :, 2]) / max(target_lift_velocity, 1.0e-6),
    max=1.0,
  )
  progress_score = torch.clamp(
    torch.relu(foot_proj) / torch.clamp(target_reach.unsqueeze(1), min=1.0e-6),
    max=1.0,
  )
  completion_progress = torch.clamp(
    (foot_proj - min_reach)
    / torch.clamp(target_reach.unsqueeze(1) - min_reach, min=1.0e-6),
    min=0.0,
    max=1.0,
  ).pow(completion_progress_power)
  decontact_score = 1.0 - contact
  initiation_signal = has_support_if_swing * torch.clamp(
    0.55 * decontact_score
    + 0.20 * lift_velocity_score
    + 0.15 * velocity_score
    + 0.10 * clearance_score,
    max=1.0,
  )
  flight_signal = has_support_if_swing * decontact_score * torch.clamp(
    0.35 * clearance_score
    + 0.30 * velocity_score
    + 0.20 * progress_score
    + 0.15 * completion_progress,
    max=1.0,
  )
  candidate_bonus = torch.clamp(
    0.35 * initiation_signal + 0.65 * flight_signal,
    max=1.0,
  )
  _, selected_idx = torch.max(candidate_bonus + tie_bias, dim=1)
  best_bonus = torch.gather(
    candidate_bonus, 1, selected_idx.unsqueeze(1)
  ).squeeze(1)
  bonus = step_need * best_bonus
  selected_reach = torch.gather(foot_proj, 1, selected_idx.unsqueeze(1)).squeeze(1)
  selected_contact = torch.gather(contact, 1, selected_idx.unsqueeze(1)).squeeze(1)
  selected_height = torch.gather(foot_height, 1, selected_idx.unsqueeze(1)).squeeze(1)
  selected_velocity = torch.gather(
    foot_vel_proj, 1, selected_idx.unsqueeze(1)
  ).squeeze(1)
  selected_lift_velocity = torch.gather(
    foot_vel_b[:, :, 2], 1, selected_idx.unsqueeze(1)
  ).squeeze(1)

  env.extras["log"]["Metrics/recovery_swing_bonus_mean"] = torch.mean(bonus)
  env.extras["log"]["Metrics/recovery_swing_deficit_mean"] = torch.mean(deficit)
  env.extras["log"]["Metrics/recovery_swing_step_need_mean"] = torch.mean(step_need)
  env.extras["log"]["Metrics/recovery_swing_candidate_reach_mean"] = torch.mean(
    selected_reach
  )
  env.extras["log"]["Metrics/recovery_swing_contact_mean"] = torch.mean(
    selected_contact
  )
  env.extras["log"]["Metrics/recovery_swing_decontact_mean"] = torch.mean(
    decontact_score
  )
  env.extras["log"]["Metrics/recovery_swing_height_mean"] = torch.mean(
    selected_height
  )
  env.extras["log"]["Metrics/recovery_swing_velocity_mean"] = torch.mean(
    selected_velocity
  )
  env.extras["log"]["Metrics/recovery_swing_lift_velocity_mean"] = torch.mean(
    selected_lift_velocity
  )
  return bonus


def recovery_step_completion_bonus(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_reach: float = 0.26,
  max_reach: float = 0.64,
  capture_reach_gain: float = 0.85,
  velocity_reach_gain: float = 0.65,
  direction_com_gain: float = 0.60,
  direction_velocity_gain: float = 0.75,
  direction_deadband: float = 0.04,
  sagittal_bias_gain: float = 1.0,
  lateral_suppression: float = 0.0,
  sagittal_activation: float = 0.08,
  foot_tie_break_scale: float = 0.0,
  foot_tie_break_period_s: float = 4.0,
  risk_activation: float = 0.12,
  recontact_margin: float = 0.04,
  need_scale: float = 0.10,
  dynamic_need_weight: float = 0.35,
  progress_power: float = 2.0,
  complete_weight: float = 0.70,
  progress_weight: float = 0.30,
  min_air_time: float = 0.04,
  max_recontact_time: float = 0.30,
  gravity: float = 9.81,
  min_com_height: float = 0.30,
  max_capture_offset: float = 0.90,
  risk_tilt_limit: float = 0.35,
  risk_ang_vel_limit: float = 1.5,
  risk_planar_speed_limit: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward completing recovery by placing a contacted foot in the needed direction."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
  if foot_pos_b.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  num_feet = foot_pos_b.shape[1]
  contact = _foot_contact_mask(env, sensor_name, num_feet).float()
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.last_air_time is not None
  assert sensor_data.current_contact_time is not None
  last_air_time = sensor_data.last_air_time[:, :num_feet]
  current_contact_time = sensor_data.current_contact_time[:, :num_feet]
  fresh_recontact = (
    contact
    * (last_air_time >= min_air_time).float()
    * (current_contact_time <= max_recontact_time).float()
  )

  com_w = _whole_body_com_w(asset)
  rel_com_w = com_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  rel_com_b = quat_apply_inverse(
    asset.data.root_link_quat_w,
    torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
  )
  root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
  omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
  capture_offset_b = root_lin_vel_b / omega.unsqueeze(1)
  capture_offset_norm = torch.norm(capture_offset_b, dim=1, keepdim=True)
  capture_offset_b = capture_offset_b * torch.clamp(
    max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
    max=1.0,
  )
  rel_capture_b = rel_com_b[:, :2] + capture_offset_b
  risk = _balance_risk(
    asset,
    tilt_limit=risk_tilt_limit,
    ang_vel_limit=risk_ang_vel_limit,
    planar_speed_limit=risk_planar_speed_limit,
  )

  direction = (
    direction_com_gain * rel_capture_b
    + direction_velocity_gain * root_lin_vel_b
  )
  direction = _sagittal_biased_direction(
    direction,
    sagittal_bias_gain=sagittal_bias_gain,
    lateral_suppression=lateral_suppression,
    sagittal_activation=sagittal_activation,
  )
  direction_norm = torch.norm(direction, dim=1)
  risk_scale = torch.clamp(
    (risk - risk_activation) / max(1.0 - risk_activation, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  active = risk_scale * (direction_norm > direction_deadband).float()
  direction_unit = direction / torch.clamp(direction_norm.unsqueeze(1), min=1.0e-6)
  tie_bias = _balanced_foot_tie_bias(
    env,
    num_feet,
    foot_pos_b.dtype,
    scale=foot_tie_break_scale,
    period_s=foot_tie_break_period_s,
  )

  foot_proj = torch.sum(foot_pos_b[:, :, :2] * direction_unit[:, None, :], dim=-1)
  _, leading_idx = torch.max(foot_proj + tie_bias, dim=1)
  leading_reach = torch.gather(foot_proj, 1, leading_idx.unsqueeze(1)).squeeze(1)
  capture_need = torch.abs(torch.sum(rel_capture_b * direction_unit, dim=1))
  velocity_need = torch.relu(torch.sum(root_lin_vel_b * direction_unit, dim=1))
  target_reach = torch.clamp(
    min_reach + capture_reach_gain * capture_need + velocity_reach_gain * velocity_need,
    max=max_reach,
  )
  raw_reach_gap = torch.relu(target_reach - leading_reach)
  dynamic_need = torch.clamp(
    (
      torch.clamp(capture_need / max(min_reach, 1.0e-6), max=1.0)
      + torch.clamp(velocity_need / max(0.22, 1.0e-6), max=1.0)
    )
    * 0.5,
    max=1.0,
  )
  step_need = active * torch.clamp(
    raw_reach_gap / max(need_scale, 1.0e-6)
    + dynamic_need_weight * dynamic_need,
    max=1.0,
  )

  contact_progress = fresh_recontact * torch.clamp(
    (foot_proj - min_reach)
    / torch.clamp(target_reach.unsqueeze(1) - min_reach, min=1.0e-6),
    min=0.0,
    max=1.0,
  ).pow(progress_power)
  complete_contact = fresh_recontact * (
    foot_proj >= (target_reach - recontact_margin).unsqueeze(1)
  ).float()
  _, selected_idx = torch.max(contact_progress + tie_bias, dim=1)
  best_progress = torch.gather(
    contact_progress, 1, selected_idx.unsqueeze(1)
  ).squeeze(1)
  completed = torch.max(complete_contact, dim=1).values
  selected_reach = torch.gather(foot_proj, 1, selected_idx.unsqueeze(1)).squeeze(1)
  selected_contact = torch.gather(
    fresh_recontact, 1, selected_idx.unsqueeze(1)
  ).squeeze(1)

  bonus = step_need * (
    progress_weight * best_progress + complete_weight * completed
  )

  env.extras["log"]["Metrics/recovery_completion_bonus_mean"] = torch.mean(bonus)
  env.extras["log"]["Metrics/recovery_completion_step_need_mean"] = torch.mean(
    step_need
  )
  env.extras["log"]["Metrics/recovery_completion_progress_mean"] = torch.mean(
    best_progress
  )
  env.extras["log"]["Metrics/recovery_completion_contact_mean"] = torch.mean(
    selected_contact
  )
  env.extras["log"]["Metrics/recovery_completion_fresh_contact_mean"] = torch.mean(
    fresh_recontact
  )
  env.extras["log"]["Metrics/recovery_completion_reached_frac"] = torch.mean(
    completed
  )
  env.extras["log"]["Metrics/recovery_completion_selected_reach_mean"] = torch.mean(
    selected_reach
  )
  return bonus


class recovery_step_progress_bonus:
  """Dense, stateful reward for moving a swing foot into recovery support."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev_contact: torch.Tensor | None = None
    self.prev_foot_proj: torch.Tensor | None = None
    self.takeoff_reach: torch.Tensor | None = None
    self.max_airborne_reach: torch.Tensor | None = None
    self.recovery_active: torch.Tensor | None = None
    self.recovery_age: torch.Tensor | None = None
    self.latched_direction: torch.Tensor | None = None
    self.latched_target_reach: torch.Tensor | None = None
    self.latched_need: torch.Tensor | None = None
    self.post_recontact_age: torch.Tensor | None = None
    self.post_recontact_speed: torch.Tensor | None = None
    self.post_recontact_tilt: torch.Tensor | None = None
    self.post_recontact_risk: torch.Tensor | None = None
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def _ensure_buffers(
    self,
    env: ManagerBasedRlEnv,
    num_feet: int,
    dtype: torch.dtype,
  ) -> None:
    shape = (env.num_envs, num_feet)
    needs_alloc = (
      self.prev_contact is None
      or self.prev_contact.shape != shape
      or self.prev_contact.device != env.device
    )
    if not needs_alloc:
      return
    self.prev_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.prev_foot_proj = torch.zeros(shape, dtype=dtype, device=env.device)
    self.takeoff_reach = torch.zeros(shape, dtype=dtype, device=env.device)
    self.max_airborne_reach = torch.zeros(shape, dtype=dtype, device=env.device)
    self.recovery_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.recovery_age = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
    self.latched_direction = torch.zeros(env.num_envs, 2, dtype=dtype, device=env.device)
    self.latched_target_reach = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
    self.latched_need = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
    self.post_recontact_age = torch.full(
      (env.num_envs,),
      1.0e6,
      dtype=dtype,
      device=env.device,
    )
    self.post_recontact_speed = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
    self.post_recontact_tilt = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
    self.post_recontact_risk = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    min_reach: float = 0.28,
    max_reach: float = 0.72,
    capture_reach_gain: float = 1.00,
    velocity_reach_gain: float = 0.90,
    direction_com_gain: float = 0.60,
    direction_velocity_gain: float = 0.95,
    direction_deadband: float = 0.04,
    sagittal_bias_gain: float = 1.0,
    lateral_suppression: float = 0.0,
    sagittal_activation: float = 0.08,
    foot_tie_break_scale: float = 0.0,
    foot_tie_break_period_s: float = 4.0,
    risk_activation: float = 0.12,
    target_velocity: float = 0.22,
    target_clearance: float = 0.055,
    progress_scale: float = 0.035,
    advance_scale: float = 0.16,
    need_scale: float = 0.08,
    dynamic_need_weight: float = 0.35,
    reach_weight: float = 0.45,
    progress_weight: float = 0.25,
    velocity_weight: float = 0.05,
    airborne_progress_weight: float = 0.25,
    recontact_weight: float = 1.80,
    recontact_advance_weight: float = 0.70,
    recontact_target_weight: float = 0.30,
    modest_recontact_weight: float = 0.0,
    modest_recontact_margin: float = 0.015,
    modest_recontact_scale: float = 0.10,
    modest_recontact_min_support: float = 1.5,
    latch_need_threshold: float = 0.15,
    release_need_threshold: float = 0.05,
    recovery_memory_s: float = 0.70,
    recovery_retention_risk: float = 0.22,
    latched_need_memory: float = 0.995,
    useful_recontact_threshold: float = 0.02,
    stabilize_window_s: float = 0.38,
    stabilize_weight: float = 0.35,
    speed_improvement_scale: float = 0.35,
    tilt_improvement_scale: float = 0.10,
    risk_improvement_scale: float = 0.30,
    stable_state_weight: float = 0.45,
    speed_stable_scale: float = 0.35,
    tilt_stable_scale: float = 0.18,
    risk_stable_target: float = 0.35,
    min_air_time: float = 0.04,
    max_recontact_time: float = 0.30,
    recontact_margin: float = 0.04,
    ground_height: float = 0.0,
    gravity: float = 9.81,
    min_com_height: float = 0.30,
    max_capture_offset: float = 0.90,
    risk_tilt_limit: float = 0.35,
    risk_ang_vel_limit: float = 1.5,
    risk_planar_speed_limit: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    """Reward the sequence: decontact, move outward, then recontact farther out."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_b = _foot_positions_in_root_frame(asset, asset_cfg)
    if foot_pos_b.shape[1] < 2:
      return torch.zeros(env.num_envs, device=env.device)

    num_feet = foot_pos_b.shape[1]
    self._ensure_buffers(env, num_feet, foot_pos_b.dtype)
    assert self.prev_contact is not None
    assert self.prev_foot_proj is not None
    assert self.takeoff_reach is not None
    assert self.max_airborne_reach is not None
    assert self.recovery_active is not None
    assert self.recovery_age is not None
    assert self.latched_direction is not None
    assert self.latched_target_reach is not None
    assert self.latched_need is not None
    assert self.post_recontact_age is not None
    assert self.post_recontact_speed is not None
    assert self.post_recontact_tilt is not None
    assert self.post_recontact_risk is not None

    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.last_air_time is not None
    assert sensor_data.current_contact_time is not None
    current_contact_time = sensor_data.current_contact_time[:, :num_feet]
    last_air_time = sensor_data.last_air_time[:, :num_feet]
    contact_bool = current_contact_time > 0.0
    contact = contact_bool.float()
    contact_count = torch.sum(contact, dim=1)

    com_w = _whole_body_com_w(asset)
    rel_com_w = com_w[:, :2] - asset.data.root_link_pos_w[:, :2]
    rel_com_b = quat_apply_inverse(
      asset.data.root_link_quat_w,
      torch.cat((rel_com_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1),
    )
    root_lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]
    omega = torch.sqrt(gravity / torch.clamp(com_w[:, 2], min=min_com_height))
    capture_offset_b = root_lin_vel_b / omega.unsqueeze(1)
    capture_offset_norm = torch.norm(capture_offset_b, dim=1, keepdim=True)
    capture_offset_b = capture_offset_b * torch.clamp(
      max_capture_offset / torch.clamp(capture_offset_norm, min=1.0e-6),
      max=1.0,
    )
    rel_capture_b = rel_com_b[:, :2] + capture_offset_b
    risk = _balance_risk(
      asset,
      tilt_limit=risk_tilt_limit,
      ang_vel_limit=risk_ang_vel_limit,
      planar_speed_limit=risk_planar_speed_limit,
    )

    direction = (
      direction_com_gain * rel_capture_b
      + direction_velocity_gain * root_lin_vel_b
    )
    direction = _sagittal_biased_direction(
      direction,
      sagittal_bias_gain=sagittal_bias_gain,
      lateral_suppression=lateral_suppression,
      sagittal_activation=sagittal_activation,
    )
    direction_norm = torch.norm(direction, dim=1)
    risk_scale = torch.clamp(
      (risk - risk_activation) / max(1.0 - risk_activation, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    active = risk_scale * (direction_norm > direction_deadband).float()
    current_direction_unit = direction / torch.clamp(
      direction_norm.unsqueeze(1), min=1.0e-6
    )
    current_foot_proj = torch.sum(
      foot_pos_b[:, :, :2] * current_direction_unit[:, None, :], dim=-1
    )
    tie_bias = _balanced_foot_tie_bias(
      env,
      num_feet,
      foot_pos_b.dtype,
      scale=foot_tie_break_scale,
      period_s=foot_tie_break_period_s,
    )
    _, leading_idx = torch.max(current_foot_proj + tie_bias, dim=1)
    leading_reach = torch.gather(
      current_foot_proj, 1, leading_idx.unsqueeze(1)
    ).squeeze(1)

    capture_need = torch.abs(torch.sum(rel_capture_b * current_direction_unit, dim=1))
    velocity_need = torch.relu(torch.sum(root_lin_vel_b * current_direction_unit, dim=1))
    target_reach = torch.clamp(
      min_reach + capture_reach_gain * capture_need + velocity_reach_gain * velocity_need,
      max=max_reach,
    )
    raw_reach_gap = torch.relu(target_reach - leading_reach)
    dynamic_need = torch.clamp(
      (
        torch.clamp(capture_need / max(min_reach, 1.0e-6), max=1.0)
        + torch.clamp(velocity_need / max(target_velocity, 1.0e-6), max=1.0)
      )
      * 0.5,
      max=1.0,
    )
    step_need = active * torch.clamp(
      raw_reach_gap / max(need_scale, 1.0e-6)
      + dynamic_need_weight * dynamic_need,
      max=1.0,
    )

    reset_mask = (env.episode_length_buf <= 1) | (~self.initialized)
    prev_active = torch.where(
      reset_mask,
      torch.zeros_like(self.recovery_active),
      self.recovery_active,
    )
    recovery_expired = self.recovery_age >= recovery_memory_s
    start_recovery = (
      (step_need > latch_need_threshold)
      & (reset_mask | (~prev_active) | recovery_expired)
    )
    keep_recovery = (
      prev_active
      & (self.recovery_age < recovery_memory_s)
      & ((step_need > release_need_threshold) | (risk > recovery_retention_risk))
    )
    recovery_active = start_recovery | keep_recovery
    next_recovery_age = torch.where(
      reset_mask | start_recovery,
      torch.zeros_like(self.recovery_age),
      torch.where(
        recovery_active,
        self.recovery_age + env.step_dt,
        torch.zeros_like(self.recovery_age),
      ),
    )

    latch_mask = reset_mask | start_recovery
    latched_direction = torch.where(
      latch_mask.unsqueeze(1),
      current_direction_unit,
      self.latched_direction,
    )
    latched_target_reach = torch.where(
      latch_mask,
      target_reach,
      self.latched_target_reach,
    )
    latched_need = torch.where(
      latch_mask,
      step_need,
      torch.maximum(step_need, self.latched_need * latched_need_memory),
    )
    latched_need = torch.where(
      recovery_active,
      torch.clamp(latched_need, max=1.0),
      torch.zeros_like(latched_need),
    )
    episode_need = torch.where(recovery_active, latched_need, step_need)

    foot_proj = torch.sum(foot_pos_b[:, :, :2] * latched_direction[:, None, :], dim=-1)
    foot_vel_w = (
      asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
      - asset.data.root_link_lin_vel_w[:, None, :]
    )
    root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
      -1, foot_vel_w.shape[1], -1
    )
    foot_vel_b = quat_apply_inverse(
      root_quat_w.reshape(-1, 4), foot_vel_w.reshape(-1, 3)
    ).reshape(env.num_envs, num_feet, 3)
    foot_vel_proj = torch.sum(foot_vel_b[:, :, :2] * latched_direction[:, None, :], dim=-1)
    foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - ground_height

    prev_contact = torch.where(reset_mask.unsqueeze(1), contact_bool, self.prev_contact)
    prev_foot_proj = torch.where(
      reset_mask.unsqueeze(1),
      foot_proj,
      self.prev_foot_proj,
    )
    takeoff_reach = torch.where(
      reset_mask.unsqueeze(1),
      foot_proj,
      self.takeoff_reach,
    )
    max_airborne_reach = torch.where(
      reset_mask.unsqueeze(1),
      foot_proj,
      self.max_airborne_reach,
    )

    airborne = ~contact_bool
    fresh_recontact = (
      contact_bool
      & (last_air_time >= min_air_time)
      & (current_contact_time <= max_recontact_time)
    )
    active_foot_mask = recovery_active.unsqueeze(1)
    started_airborne = prev_contact & airborne & active_foot_mask
    fresh_recontact_active = fresh_recontact & active_foot_mask
    support_if_swing = contact_count.unsqueeze(1) - contact
    has_support_if_swing = (support_if_swing >= 1.0).float()

    takeoff_reach = torch.where(started_airborne, foot_proj, takeoff_reach)
    max_airborne_reach = torch.where(started_airborne, foot_proj, max_airborne_reach)
    max_airborne_reach = torch.where(
      airborne & active_foot_mask,
      torch.maximum(max_airborne_reach, foot_proj),
      max_airborne_reach,
    )

    progress_delta = torch.relu(foot_proj - prev_foot_proj)
    progress_score = torch.clamp(
      progress_delta / max(progress_scale, 1.0e-6),
      max=1.0,
    )
    velocity_score = torch.clamp(
      torch.relu(foot_vel_proj) / max(target_velocity, 1.0e-6),
      max=1.0,
    )
    clearance_score = torch.clamp(
      foot_height / max(target_clearance, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    airborne_advance = torch.relu(max_airborne_reach - takeoff_reach)
    airborne_progress_score = torch.clamp(
      airborne_advance / max(advance_scale, 1.0e-6),
      max=1.0,
    )
    target_reach_2d = torch.clamp(
      latched_target_reach.unsqueeze(1),
      min=min_reach + 1.0e-6,
    )
    reach_progress_score = torch.clamp(
      (foot_proj - min_reach) / (target_reach_2d - min_reach),
      min=0.0,
      max=1.0,
    )

    swing_signal = (
      reach_weight * reach_progress_score
      + progress_weight * progress_score
      + velocity_weight * velocity_score * reach_progress_score
      + airborne_progress_weight * airborne_progress_score
    )
    swing_reward = (
      episode_need.unsqueeze(1)
      * recovery_active.unsqueeze(1).float()
      * airborne.float()
      * has_support_if_swing
      * clearance_score
      * swing_signal
    )

    recontact_advance = torch.relu(foot_proj - takeoff_reach)
    recontact_advance_score = torch.clamp(
      recontact_advance / max(advance_scale, 1.0e-6),
      max=1.0,
    )
    recontact_target_score = (
      foot_proj >= (latched_target_reach - recontact_margin).unsqueeze(1)
    ).float()
    recontact_signal = (
      recontact_advance_weight * recontact_advance_score
      + recontact_target_weight * recontact_target_score
    )
    recontact_reward = (
      episode_need.unsqueeze(1)
      * fresh_recontact_active.float()
      * recontact_weight
      * recontact_signal
    )
    modest_recontact_score = torch.clamp(
      (recontact_advance + modest_recontact_margin)
      / max(modest_recontact_scale, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    supported_recontact = fresh_recontact_active.float() * (
      contact_count.unsqueeze(1) >= modest_recontact_min_support
    ).float()
    modest_recontact_reward = (
      episode_need.unsqueeze(1)
      * supported_recontact
      * modest_recontact_weight
      * modest_recontact_score
    )

    _, best_swing_idx = torch.max(swing_reward + tie_bias, dim=1)
    best_swing_reward = torch.gather(
      swing_reward, 1, best_swing_idx.unsqueeze(1)
    ).squeeze(1)
    total_recontact_reward = recontact_reward + modest_recontact_reward
    _, best_recontact_idx = torch.max(total_recontact_reward + tie_bias, dim=1)
    best_recontact_reward = torch.gather(
      total_recontact_reward, 1, best_recontact_idx.unsqueeze(1)
    ).squeeze(1)
    recontact_quality = torch.max(
      fresh_recontact_active.float() * recontact_signal,
      dim=1,
    ).values
    modest_recontact_quality = torch.max(
      supported_recontact * modest_recontact_score,
      dim=1,
    ).values

    planar_speed = torch.norm(
      torch.cat(
        (root_lin_vel_b, asset.data.root_link_ang_vel_b[:, 2:3]),
        dim=1,
      ),
      dim=1,
    )
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    useful_recontact = (recontact_quality > useful_recontact_threshold) & recovery_active
    modest_recontact = (modest_recontact_quality > 0.0) & recovery_active
    stabilizing_recontact = useful_recontact | modest_recontact
    expired_post_age = torch.full_like(
      self.post_recontact_age,
      stabilize_window_s + env.step_dt,
    )
    post_recontact_age = torch.where(
      reset_mask,
      expired_post_age,
      self.post_recontact_age + env.step_dt,
    )
    post_recontact_age = torch.where(
      stabilizing_recontact,
      torch.zeros_like(post_recontact_age),
      post_recontact_age,
    )
    post_recontact_speed = torch.where(
      reset_mask | stabilizing_recontact,
      planar_speed,
      self.post_recontact_speed,
    )
    post_recontact_tilt = torch.where(
      reset_mask | stabilizing_recontact,
      tilt,
      self.post_recontact_tilt,
    )
    post_recontact_risk = torch.where(
      reset_mask | stabilizing_recontact,
      risk,
      self.post_recontact_risk,
    )
    post_recontact_active = post_recontact_age <= stabilize_window_s
    speed_improvement = torch.clamp(
      torch.relu(post_recontact_speed - planar_speed)
      / max(speed_improvement_scale, 1.0e-6),
      max=1.0,
    )
    tilt_improvement = torch.clamp(
      torch.relu(post_recontact_tilt - tilt) / max(tilt_improvement_scale, 1.0e-6),
      max=1.0,
    )
    risk_improvement = torch.clamp(
      torch.relu(post_recontact_risk - risk) / max(risk_improvement_scale, 1.0e-6),
      max=1.0,
    )
    support_scale = torch.clamp(contact_count - 1.0, min=0.0, max=1.0)
    stabilize_bonus = (
      post_recontact_active.float()
      * support_scale
      * torch.clamp(
        (1.0 - stable_state_weight)
        * (
          0.45 * speed_improvement
          + 0.35 * risk_improvement
          + 0.20 * tilt_improvement
        )
        + stable_state_weight
        * (
          0.40 * torch.exp(-torch.square(planar_speed / max(speed_stable_scale, 1.0e-6)))
          + 0.35 * torch.exp(-torch.square(tilt / max(tilt_stable_scale, 1.0e-6)))
          + 0.25 * torch.clamp(
            (risk_stable_target - risk) / max(risk_stable_target, 1.0e-6),
            min=0.0,
            max=1.0,
          )
        ),
        max=1.0,
      )
    )
    bonus = best_swing_reward + best_recontact_reward + stabilize_weight * stabilize_bonus

    self.prev_contact[:] = contact_bool
    self.prev_foot_proj[:] = foot_proj
    self.takeoff_reach[:] = takeoff_reach
    self.max_airborne_reach[:] = torch.where(
      contact_bool,
      foot_proj,
      max_airborne_reach,
    )
    self.recovery_active[:] = recovery_active
    self.recovery_age[:] = next_recovery_age
    self.latched_direction[:] = latched_direction
    self.latched_target_reach[:] = latched_target_reach
    self.latched_need[:] = latched_need
    self.post_recontact_age[:] = post_recontact_age
    self.post_recontact_speed[:] = post_recontact_speed
    self.post_recontact_tilt[:] = post_recontact_tilt
    self.post_recontact_risk[:] = post_recontact_risk
    self.initialized[:] = True

    env.extras["log"]["Metrics/recovery_progress_bonus_mean"] = torch.mean(bonus)
    env.extras["log"]["Metrics/recovery_progress_swing_mean"] = torch.mean(
      best_swing_reward
    )
    env.extras["log"]["Metrics/recovery_progress_recontact_mean"] = torch.mean(
      best_recontact_reward
    )
    env.extras["log"]["Metrics/recovery_progress_delta_mean"] = torch.mean(
      progress_delta
    )
    env.extras["log"]["Metrics/recovery_progress_airborne_advance_mean"] = torch.mean(
      airborne_advance
    )
    env.extras["log"]["Metrics/recovery_progress_reach_score_mean"] = torch.mean(
      reach_progress_score
    )
    env.extras["log"]["Metrics/recovery_progress_recontact_frac"] = torch.mean(
      fresh_recontact.float()
    )
    env.extras["log"]["Metrics/recovery_progress_step_need_mean"] = torch.mean(
      step_need
    )
    env.extras["log"]["Metrics/recovery_progress_latched_active_frac"] = torch.mean(
      recovery_active.float()
    )
    env.extras["log"]["Metrics/recovery_progress_latched_need_mean"] = torch.mean(
      latched_need
    )
    env.extras["log"]["Metrics/recovery_progress_latched_target_mean"] = torch.mean(
      latched_target_reach
    )
    env.extras["log"]["Metrics/recovery_progress_recontact_quality_mean"] = torch.mean(
      recontact_quality
    )
    env.extras["log"]["Metrics/recovery_progress_modest_recontact_mean"] = torch.mean(
      torch.max(modest_recontact_reward, dim=1).values
    )
    env.extras["log"]["Metrics/recovery_progress_modest_recontact_quality_mean"] = (
      torch.mean(modest_recontact_quality)
    )
    env.extras["log"]["Metrics/recovery_progress_modest_recontact_frac"] = torch.mean(
      modest_recontact.float()
    )
    env.extras["log"]["Metrics/recovery_progress_useful_recontact_frac"] = torch.mean(
      useful_recontact.float()
    )
    env.extras["log"]["Metrics/recovery_progress_stabilizing_recontact_frac"] = torch.mean(
      stabilizing_recontact.float()
    )
    env.extras["log"]["Metrics/recovery_stabilize_bonus_mean"] = torch.mean(
      stabilize_bonus
    )
    env.extras["log"]["Metrics/recovery_stabilize_active_frac"] = torch.mean(
      post_recontact_active.float()
    )
    env.extras["log"]["Metrics/recovery_stabilize_speed_score_mean"] = torch.mean(
      torch.exp(-torch.square(planar_speed / max(speed_stable_scale, 1.0e-6)))
    )
    return bonus


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
