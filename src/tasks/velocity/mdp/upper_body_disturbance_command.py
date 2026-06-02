"""Smooth upper-body joint-position disturbance commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_MODE_STATIC = 0
_MODE_RANDOM_WALK = 1
_MODE_SINUSOID = 2
_MODE_PULSE = 3


@dataclass(kw_only=True)
class UpperBodyDisturbanceCommandCfg(CommandTermCfg):
  """Generate plausible static and dynamic upper-body target trajectories."""

  entity_name: str
  joint_names: tuple[str, ...]
  ranges: dict[str, tuple[float, float]]
  rel_default_envs: float = 0.10
  mode_probabilities: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  amplitude_scale: float = 0.35
  random_walk_velocity_range: tuple[float, float] = (0.05, 0.25)
  random_walk_acceleration_range: tuple[float, float] = (0.10, 0.80)
  sinusoid_frequency_range: tuple[float, float] = (0.25, 1.00)
  pulse_duration_range: tuple[float, float] = (0.40, 1.50)

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyDisturbanceCommand:
    return UpperBodyDisturbanceCommand(self, env)


class UpperBodyDisturbanceCommand(CommandTerm):
  """Command term for static poses, random walks, sinusoids, and smooth pulses."""

  cfg: UpperBodyDisturbanceCommandCfg

  def __init__(self, cfg: UpperBodyDisturbanceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self._joint_ids, self._joint_names = self.robot.find_joints(cfg.joint_names)
    self._num_joints = len(self._joint_ids)

    range_low = []
    range_high = []
    for name in self._joint_names:
      if name not in cfg.ranges:
        raise ValueError(
          f"Joint '{name}' not found in upper-body ranges. "
          f"Available: {list(cfg.ranges.keys())}"
        )
      lo, hi = cfg.ranges[name]
      range_low.append(lo)
      range_high.append(hi)
    self._range_low = torch.tensor(range_low, device=self.device, dtype=torch.float32)
    self._range_high = torch.tensor(range_high, device=self.device, dtype=torch.float32)
    self._range_center = 0.5 * (self._range_low + self._range_high)
    self._range_half = 0.5 * (self._range_high - self._range_low)

    default_joint_pos = self.robot.data.default_joint_pos
    assert default_joint_pos is not None
    self._default_joint_pos = default_joint_pos[:, self._joint_ids].clone()
    self._default_joint_pos = torch.clamp(
      self._default_joint_pos, self._range_low, self._range_high
    )

    self.joint_pos_command = self._default_joint_pos.clone()
    self._previous_command = self.joint_pos_command.clone()
    self._command_velocity = torch.zeros_like(self.joint_pos_command)
    self._mode = torch.full(
      (self.num_envs,), _MODE_STATIC, device=self.device, dtype=torch.long
    )
    self._target = self._default_joint_pos.clone()
    self._start = self._default_joint_pos.clone()
    self._amplitude = torch.zeros_like(self.joint_pos_command)
    self._phase = torch.zeros_like(self.joint_pos_command)
    self._frequency = torch.zeros(self.num_envs, 1, device=self.device)
    self._elapsed = torch.zeros(self.num_envs, 1, device=self.device)
    self._duration = torch.ones(self.num_envs, 1, device=self.device)

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["command_speed"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["static_frac"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["random_walk_frac"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sinusoid_frac"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["pulse_frac"] = torch.zeros(self.num_envs, device=self.device)

    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_sliders: list[viser.GuiSliderHandle] = []
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.joint_pos_command

  def _sample_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
    n = len(env_ids)
    raw_sample = self._range_low + torch.rand(
      n, self._num_joints, device=self.device
    ) * (self._range_high - self._range_low)
    default = self._default_joint_pos[env_ids]
    amp_scale = torch.clamp(
      torch.tensor(float(self.cfg.amplitude_scale), device=self.device),
      min=0.0,
      max=1.0,
    )
    sample = default + (raw_sample - default) * amp_scale
    sample = torch.clamp(sample, self._range_low, self._range_high)
    if self.cfg.rel_default_envs > 0.0:
      default_mask = torch.rand(n, device=self.device) <= self.cfg.rel_default_envs
      if default_mask.any():
        sample[default_mask] = default[default_mask]
    return sample

  def _sample_modes(self, env_ids: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor(
      self.cfg.mode_probabilities, device=self.device, dtype=torch.float32
    )
    probs = torch.clamp(probs, min=0.0)
    if torch.sum(probs) <= 0.0:
      probs[0] = 1.0
    probs = probs / torch.sum(probs)
    return torch.multinomial(probs, num_samples=len(env_ids), replacement=True)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    self._mode[env_ids] = self._sample_modes(env_ids)
    self._elapsed[env_ids] = 0.0
    self._start[env_ids] = self.joint_pos_command[env_ids]
    self._target[env_ids] = self._sample_pose(env_ids)
    self._command_velocity[env_ids] = 0.0

    amp_scale = max(float(self.cfg.amplitude_scale), 0.0)
    self._amplitude[env_ids] = (
      (torch.rand(n, self._num_joints, device=self.device) * 2.0 - 1.0)
      * self._range_half
      * amp_scale
    )
    self._phase[env_ids] = torch.rand(
      n, self._num_joints, device=self.device
    ) * (2.0 * torch.pi)
    f_min, f_max = self.cfg.sinusoid_frequency_range
    self._frequency[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
      f_min, f_max
    )
    d_min, d_max = self.cfg.pulse_duration_range
    self._duration[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
      d_min, d_max
    )

    static = self._mode[env_ids] == _MODE_STATIC
    if static.any():
      static_ids = env_ids[static]
      self.joint_pos_command[static_ids] = self._target[static_ids]

  def _update_command(self) -> None:
    dt = self._env.step_dt
    self._previous_command[:] = self.joint_pos_command
    self._elapsed += dt

    random_walk = self._mode == _MODE_RANDOM_WALK
    if random_walk.any():
      v_min, v_max = self.cfg.random_walk_velocity_range
      a_min, a_max = self.cfg.random_walk_acceleration_range
      accel_mag = torch.empty_like(self._command_velocity[random_walk]).uniform_(
        a_min, a_max
      )
      accel = torch.randn_like(self._command_velocity[random_walk]) * accel_mag
      self._command_velocity[random_walk] += accel * dt
      n_random_walk = random_walk.nonzero().shape[0]
      max_vel = torch.empty((n_random_walk, 1), device=self.device).uniform_(
        v_min, v_max
      )
      self._command_velocity[random_walk] = torch.clamp(
        self._command_velocity[random_walk], -max_vel, max_vel
      )
      next_command = (
        self.joint_pos_command[random_walk]
        + self._command_velocity[random_walk] * dt
      )
      next_command = torch.clamp(next_command, self._range_low, self._range_high)
      at_limit = (next_command <= self._range_low) | (next_command >= self._range_high)
      self._command_velocity[random_walk] = torch.where(
        at_limit,
        -0.25 * self._command_velocity[random_walk],
        self._command_velocity[random_walk],
      )
      self.joint_pos_command[random_walk] = next_command

    sinusoid = self._mode == _MODE_SINUSOID
    if sinusoid.any():
      phase = (
        2.0 * torch.pi * self._frequency[sinusoid] * self._elapsed[sinusoid]
        + self._phase[sinusoid]
      )
      self.joint_pos_command[sinusoid] = torch.clamp(
        self._default_joint_pos[sinusoid] + self._amplitude[sinusoid] * torch.sin(phase),
        self._range_low,
        self._range_high,
      )

    pulse = self._mode == _MODE_PULSE
    if pulse.any():
      tau = torch.clamp(self._elapsed[pulse] / self._duration[pulse], 0.0, 1.0)
      smooth = tau * tau * (3.0 - 2.0 * tau)
      self.joint_pos_command[pulse] = (
        self._start[pulse] * (1.0 - smooth) + self._target[pulse] * smooth
      )

    self._command_velocity = (self.joint_pos_command - self._previous_command) / dt

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max(max_command_time / self._env.step_dt, 1.0)
    current = self.robot.data.joint_pos[:, self._joint_ids]
    self.metrics["error_joint_pos"] += (
      torch.mean(torch.abs(current - self.joint_pos_command), dim=1)
      / max_command_step
    )
    self.metrics["command_speed"] += (
      torch.mean(torch.abs(self._command_velocity), dim=1) / max_command_step
    )
    self.metrics["static_frac"] += (
      (self._mode == _MODE_STATIC).float() / max_command_step
    )
    self.metrics["random_walk_frac"] += (
      (self._mode == _MODE_RANDOM_WALK).float() / max_command_step
    )
    self.metrics["sinusoid_frac"] += (
      (self._mode == _MODE_SINUSOID).float() / max_command_step
    )
    self.metrics["pulse_frac"] += (
      (self._mode == _MODE_PULSE).float() / max_command_step
    )

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    from viser import Icon

    sliders: list = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Manual upper body", initial_value=False)
      for i, jname in enumerate(self._joint_names):
        slider = server.gui.add_slider(
          jname,
          min=float(self._range_low[i]),
          max=float(self._range_high[i]),
          step=0.01,
          initial_value=float(self._default_joint_pos[0, i]),
        )
        sliders.append(slider)
      zero_btn = server.gui.add_button("Default", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for j, s in enumerate(sliders):
          s.value = float(self._default_joint_pos[0, j])

    self._gui_enabled = enabled
    self._gui_sliders = sliders
    self._gui_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._gui_enabled is not None and self._gui_enabled.value:
      assert self._gui_get_env_idx is not None
      idx = self._gui_get_env_idx()
      for i, slider in enumerate(self._gui_sliders):
        self.joint_pos_command[idx, i] = slider.value

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass
