from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformJointPositionCommand(CommandTerm):
  """Command term that generates uniform random joint position targets."""

  cfg: UniformJointPositionCommandCfg

  def __init__(self, cfg: UniformJointPositionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Resolve joint names to indices.
    self._joint_ids, self._joint_names = self.robot.find_joints(cfg.joint_names)
    self._num_joints = len(self._joint_ids)

    # Build per-joint range tensors from the config.
    range_low = []
    range_high = []
    for name in self._joint_names:
      if name in cfg.ranges:
        lo, hi = cfg.ranges[name]
      else:
        raise ValueError(
          f"Joint '{name}' not found in ranges. "
          f"Available: {list(cfg.ranges.keys())}"
        )
      range_low.append(lo)
      range_high.append(hi)
    self._range_low = torch.tensor(range_low, device=self.device, dtype=torch.float32)
    self._range_high = torch.tensor(range_high, device=self.device, dtype=torch.float32)

    # Store default joint positions for the controlled joints.
    default_joint_pos = self.robot.data.default_joint_pos
    assert default_joint_pos is not None
    self._default_joint_pos = default_joint_pos[:, self._joint_ids]

    # Command buffer.
    self.joint_pos_command = self._default_joint_pos.clone()

    # Metrics.
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)

    # GUI state (set by create_gui).
    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_sliders: list[viser.GuiSliderHandle] = []
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.joint_pos_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    current = self.robot.data.joint_pos[:, self._joint_ids]
    self.metrics["error_joint_pos"] += (
      torch.mean(torch.abs(current - self.joint_pos_command), dim=1) / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    # Uniform sample within per-joint ranges.
    r = torch.rand(n, self._num_joints, device=self.device)
    self.joint_pos_command[env_ids] = (
      self._range_low + r * (self._range_high - self._range_low)
    )
    # For a fraction of environments, use the default joint positions.
    if self.cfg.rel_default_envs > 0.0:
      default_mask = torch.rand(n, device=self.device) <= self.cfg.rel_default_envs
      default_env_ids = env_ids[default_mask]
      if len(default_env_ids) > 0:
        self.joint_pos_command[default_env_ids] = self._default_joint_pos[
          default_env_ids
        ]

  def _update_command(self) -> None:
    pass  # Targets are static until resampled.

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create joint position sliders in the Viser viewer."""
    from viser import Icon

    sliders: list = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for i, jname in enumerate(self._joint_names):
        lo = float(self._range_low[i])
        hi = float(self._range_high[i])
        default = float(self._default_joint_pos[0, i])
        slider = server.gui.add_slider(
          jname,
          min=lo,
          max=hi,
          step=0.01,
          initial_value=default,
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
      for i, s in enumerate(self._gui_sliders):
        self.joint_pos_command[idx, i] = s.value

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass  # No visualization for joint position commands.


@dataclass(kw_only=True)
class UniformJointPositionCommandCfg(CommandTermCfg):
  """Configuration for uniform joint position command generation."""

  entity_name: str
  joint_names: tuple[str, ...]
  ranges: dict[str, tuple[float, float]]
  rel_default_envs: float = 0.05

  def build(self, env: ManagerBasedRlEnv) -> UniformJointPositionCommand:
    return UniformJointPositionCommand(self, env)
