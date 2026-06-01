from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Foot site positions relative to the pelvis/root frame."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  rel_pos_w = foot_pos_w - asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, rel_pos_w.shape[1], -1
  )
  rel_pos_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4), rel_pos_w.reshape(-1, 3)
  )
  return rel_pos_b.reshape(env.num_envs, -1)


def foot_vel_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Foot site linear velocities relative to the pelvis/root frame."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  rel_vel_w = foot_vel_w - asset.data.root_link_lin_vel_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, rel_vel_w.shape[1], -1
  )
  rel_vel_b = quat_apply_inverse(
    root_quat_w.reshape(-1, 4), rel_vel_w.reshape(-1, 3)
  )
  return rel_vel_b.reshape(env.num_envs, -1)


def whole_body_com_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Whole-body subtree COM relative to the pelvis/root frame."""
  asset: Entity = env.scene[asset_cfg.name]
  root_body_id = asset.data.indexing.root_body_id
  com_w = asset.data.data.subtree_com[:, root_body_id, :]
  if com_w.ndim == 3:
    com_w = com_w.squeeze(1)
  rel_com_w = com_w - asset.data.root_link_pos_w
  return quat_apply_inverse(asset.data.root_link_quat_w, rel_com_w)


def root_planar_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Privileged planar root velocity [vx_b, vy_b, wz_b]."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.cat(
    (asset.data.root_link_lin_vel_b[:, :2], asset.data.root_link_ang_vel_b[:, 2:3]),
    dim=1,
  )


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
