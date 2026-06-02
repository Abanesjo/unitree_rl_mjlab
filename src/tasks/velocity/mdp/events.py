"""Task-local MDP events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class apply_planar_body_force_pulse:
  """Apply finite-duration planar force pulses to selected bodies.

  Force magnitude is sampled in Newtons, direction is sampled uniformly in the
  horizontal plane, and the resulting wrench is cleared after the sampled
  duration. This is intended for training visible balance recovery without
  using non-physical root velocity teleports.
  """

  @dataclass
  class VizCfg:
    """Arrow visualization settings for active push forces."""

    rgba: tuple[float, float, float, float] = (0.95, 0.25, 0.15, 0.9)
    scale: float = 0.004
    width: float = 0.015
    min_force: float = 1.0

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._viz_cfg: apply_planar_body_force_pulse.VizCfg = cfg.params.get(
      "viz_cfg", apply_planar_body_force_pulse.VizCfg()
    )

    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._interval_time_left = self._sample_range(
      cfg.params.get("cooldown_s", (0.0, 0.0)), (self._num_envs,)
    )
    self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._last_force_magnitude = torch.zeros(self._num_envs, device=self._device)
    self._last_cooldown_s = cfg.params.get("cooldown_s", (0.0, 0.0))
    self._body_point_offset = self._make_offset(cfg.params.get("body_point_offset"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_magnitude_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    force_z_range: tuple[float, float] = (0.0, 0.0),
    torque_range: tuple[float, float] = (0.0, 0.0),
    body_point_offset: tuple[float, float, float] | None = None,
  ) -> None:
    del env_ids, asset_cfg  # Step events operate on all environments.

    self._last_cooldown_s = cooldown_s
    self._body_point_offset = self._make_offset(body_point_offset)

    force_low, force_high = sorted(force_magnitude_range)
    force_low = max(0.0, float(force_low))
    force_high = max(0.0, float(force_high))
    if force_high <= 0.0:
      self._clear_active()
      self._tick_idle_timers(cooldown_s)
      self._log(env)
      return

    dt = self._step_dt
    self._time_remaining[self._active] -= dt

    expired = self._active & (self._time_remaining <= 0.0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      self._zero_envs(expired_ids)
      self._active[expired_ids] = False
      self._time_remaining[expired_ids] = 0.0
      self._last_force_magnitude[expired_ids] = 0.0
      self._resample_interval(expired_ids, cooldown_s)

    inactive = ~self._active
    self._interval_time_left[inactive] -= dt
    eligible = inactive & (self._interval_time_left <= 0.0)
    if not eligible.any():
      self._log(env)
      return

    trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
    n = len(trigger_ids)

    magnitudes = self._sample_range((force_low, force_high), (n,))
    angles = torch.rand(n, device=self._device) * (2.0 * torch.pi)
    force_xy = torch.stack(
      (magnitudes * torch.cos(angles), magnitudes * torch.sin(angles)), dim=-1
    )
    force_z = self._sample_range(force_z_range, (n, 1))
    total_force = torch.cat((force_xy, force_z), dim=-1)

    forces = (
      total_force[:, None, :]
      .expand(n, self._num_bodies, 3)
      .clone()
      / float(self._num_bodies)
    )
    torques = self._sample_range(torque_range, (n, self._num_bodies, 3))

    if self._body_point_offset is not None:
      body_quat = self._asset.data.body_com_quat_w[trigger_ids][:, self._body_ids]
      offset_w = quat_apply(
        body_quat.reshape(-1, 4),
        self._body_point_offset.expand(n * self._num_bodies, 3),
      ).reshape(n, self._num_bodies, 3)
      torques = torques + torch.cross(offset_w, forces, dim=-1)

    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
    )

    self._time_remaining[trigger_ids] = self._sample_range(duration_s, (n,))
    self._active[trigger_ids] = True
    self._last_force_magnitude[trigger_ids] = magnitudes
    self._log(env)

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw arrows for active push forces in rendered videos."""
    if not self._active.any():
      return

    viz = self._viz_cfg
    min_sq = viz.min_force * viz.min_force
    wrench = self._asset.data.body_external_wrench
    com_pos = self._asset.data.body_com_pos_w
    com_quat = (
      self._asset.data.body_com_quat_w if self._body_point_offset is not None else None
    )

    for env_idx in visualizer.get_env_indices(self._num_envs):
      if not self._active[env_idx]:
        continue
      for body_idx in range(wrench.shape[1]):
        force = wrench[env_idx, body_idx, :3]
        if (force * force).sum().item() < min_sq:
          continue
        start = com_pos[env_idx, body_idx]
        if self._body_point_offset is not None and com_quat is not None:
          start = start + quat_apply(com_quat[env_idx, body_idx], self._body_point_offset)
        visualizer.add_arrow(
          start=start.cpu().numpy(),
          end=(start + force * viz.scale).cpu().numpy(),
          color=viz.rgba,
          width=viz.width,
        )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
    else:
      active_ids = env_ids[self._active[env_ids]]
    if len(active_ids) > 0:
      self._zero_envs(active_ids)

    self._time_remaining[env_ids] = 0.0
    self._last_force_magnitude[env_ids] = 0.0
    self._active[env_ids] = False
    self._resample_interval(env_ids, self._last_cooldown_s)

  def _make_offset(
    self, offset: tuple[float, float, float] | None
  ) -> torch.Tensor | None:
    if offset is None:
      return None
    return torch.tensor(offset, device=self._device, dtype=torch.float32)

  def _sample_range(
    self, bounds: tuple[float, float], shape: tuple[int, ...]
  ) -> torch.Tensor:
    low, high = sorted((float(bounds[0]), float(bounds[1])))
    if high == low:
      return torch.full(shape, low, device=self._device)
    return torch.rand(shape, device=self._device) * (high - low) + low

  def _resample_interval(
    self, env_ids: torch.Tensor | slice, cooldown_s: tuple[float, float]
  ) -> None:
    if isinstance(env_ids, slice):
      shape = self._interval_time_left[env_ids].shape
    else:
      shape = (len(env_ids),)
    self._interval_time_left[env_ids] = self._sample_range(cooldown_s, shape)

  def _tick_idle_timers(self, cooldown_s: tuple[float, float]) -> None:
    self._interval_time_left -= self._step_dt
    ready = self._interval_time_left <= 0.0
    if ready.any():
      self._resample_interval(ready.nonzero(as_tuple=False).squeeze(-1), cooldown_s)

  def _clear_active(self) -> None:
    if not self._active.any():
      return
    active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
    self._zero_envs(active_ids)
    self._active[active_ids] = False
    self._time_remaining[active_ids] = 0.0
    self._last_force_magnitude[active_ids] = 0.0

  def _zero_envs(self, env_ids: torch.Tensor) -> None:
    zeros = torch.zeros((len(env_ids), self._num_bodies, 3), device=self._device)
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )

  def _log(self, env: ManagerBasedRlEnv) -> None:
    if not hasattr(env, "extras") or "log" not in env.extras:
      return
    active = self._active.float()
    active_force = self._last_force_magnitude[self._active]
    env.extras["log"]["Metrics/push_force_active_frac"] = torch.mean(active)
    env.extras["log"]["Metrics/push_force_mean_n"] = (
      torch.mean(active_force)
      if active_force.numel() > 0
      else torch.tensor(0.0, device=self._device)
    )


def reset_recovery_drill_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  probability: float,
  planar_speed_range: tuple[float, float],
  tilt_range: tuple[float, float],
  angular_speed_range: tuple[float, float],
  yaw_rate_range: tuple[float, float] = (0.0, 0.0),
  height_offset_range: tuple[float, float] = (0.0, 0.0),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Inject reset-time recovery drills for lower-body balance learning.

  The normal reset still places the robot at the stationary target. This event
  optionally overwrites a subset of just-reset root states with small, physically
  recoverable lean and planar velocity samples so the policy sees many examples
  where moving the support polygon is better than freezing in place.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  elif isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)[env_ids]

  if len(env_ids) == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  probability = float(max(0.0, min(1.0, probability)))
  selected = torch.rand(len(env_ids), device=env.device) < probability
  active_env_ids = env_ids[selected]
  if len(active_env_ids) == 0:
    _log_recovery_drill(env, 0.0)
    return

  n = len(active_env_ids)
  angles = torch.rand(n, device=env.device) * (2.0 * torch.pi)
  direction_w = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
  speeds = _sample_uniform_range(planar_speed_range, (n,), env.device)
  tilts = _sample_uniform_range(tilt_range, (n,), env.device)
  angular_speeds = _sample_uniform_range(angular_speed_range, (n,), env.device)
  yaw_rates = _sample_uniform_range(yaw_rate_range, (n,), env.device)
  height_offsets = _sample_uniform_range(height_offset_range, (n,), env.device)

  pose = torch.cat(
    (
      asset.data.root_link_pos_w[active_env_ids],
      asset.data.root_link_quat_w[active_env_ids],
    ),
    dim=1,
  ).clone()
  velocity = asset.data.root_link_vel_w[active_env_ids].clone()

  pose[:, 2] += height_offsets
  roll = -direction_w[:, 1] * tilts
  pitch = direction_w[:, 0] * tilts
  yaw = torch.zeros_like(roll)
  orientation_delta = quat_from_euler_xyz(roll, pitch, yaw)
  pose[:, 3:7] = quat_mul(pose[:, 3:7], orientation_delta)

  velocity[:, 0:2] = direction_w * speeds.unsqueeze(1)
  velocity[:, 2] = 0.0
  velocity[:, 3] = -direction_w[:, 1] * angular_speeds
  velocity[:, 4] = direction_w[:, 0] * angular_speeds
  velocity[:, 5] = yaw_rates

  asset.write_root_link_pose_to_sim(pose, env_ids=active_env_ids)
  asset.write_root_link_velocity_to_sim(velocity, env_ids=active_env_ids)

  _log_recovery_drill(
    env,
    float(n) / float(len(env_ids)),
    speed_mean=torch.mean(speeds),
    tilt_mean=torch.mean(tilts),
    angular_speed_mean=torch.mean(angular_speeds),
  )


def _sample_uniform_range(
  bounds: tuple[float, float],
  shape: tuple[int, ...],
  device: torch.device,
) -> torch.Tensor:
  low, high = sorted((float(bounds[0]), float(bounds[1])))
  if high == low:
    return torch.full(shape, low, device=device)
  return torch.rand(shape, device=device) * (high - low) + low


def _log_recovery_drill(
  env: ManagerBasedRlEnv,
  selected_frac: float,
  speed_mean: torch.Tensor | None = None,
  tilt_mean: torch.Tensor | None = None,
  angular_speed_mean: torch.Tensor | None = None,
) -> None:
  if not hasattr(env, "extras") or "log" not in env.extras:
    return
  device = env.device
  env.extras["log"]["Metrics/recovery_drill_frac"] = torch.tensor(
    selected_frac, device=device
  )
  env.extras["log"]["Metrics/recovery_drill_speed_mean"] = (
    speed_mean if speed_mean is not None else torch.tensor(0.0, device=device)
  )
  env.extras["log"]["Metrics/recovery_drill_tilt_deg_mean"] = (
    tilt_mean * (180.0 / torch.pi)
    if tilt_mean is not None
    else torch.tensor(0.0, device=device)
  )
  env.extras["log"]["Metrics/recovery_drill_ang_speed_mean"] = (
    angular_speed_mean
    if angular_speed_mean is not None
    else torch.tensor(0.0, device=device)
  )
