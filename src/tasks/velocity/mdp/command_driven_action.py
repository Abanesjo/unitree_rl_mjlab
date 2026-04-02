"""Action term that drives joints from command manager output (zero policy dims)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class CommandDrivenJointPositionActionCfg(ActionTermCfg):
  """Config for an action term that drives joints from command manager output.

  This term consumes zero policy output dimensions. Instead, it reads joint
  position targets from a named command term and applies them every physics
  substep via PD position control. Optionally, a second set of joints can be
  held at their default positions.
  """

  command_name: str
  """Name of the command term in the command manager."""

  commanded_joint_names: tuple[str, ...]
  """Joint names whose targets come from the command."""

  fixed_joint_names: tuple[str, ...] = ()
  """Joint names held at their default (home) positions."""

  def build(self, env: ManagerBasedRlEnv) -> CommandDrivenJointPositionAction:
    return CommandDrivenJointPositionAction(self, env)


class CommandDrivenJointPositionAction(ActionTerm):
  """Action term that drives joints from the command manager.

  This term has ``action_dim == 0`` — it does not consume any policy output.
  On every physics substep, it reads the current command and writes position
  targets to the simulation for the commanded joints (and optionally holds
  a second set of joints at their default positions).
  """

  cfg: CommandDrivenJointPositionActionCfg

  def __init__(
    self, cfg: CommandDrivenJointPositionActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg, env)

    # Resolve commanded joint IDs.
    self._commanded_ids, _ = self._entity.find_joints(cfg.commanded_joint_names)
    self._commanded_ids_t = torch.tensor(
      self._commanded_ids, device=self.device, dtype=torch.long
    )

    # Resolve fixed joint IDs.
    if cfg.fixed_joint_names:
      self._fixed_ids, _ = self._entity.find_joints(cfg.fixed_joint_names)
      self._fixed_ids_t = torch.tensor(
        self._fixed_ids, device=self.device, dtype=torch.long
      )
      self._fixed_default_pos = self._entity.data.default_joint_pos[
        :, self._fixed_ids_t
      ].clone()
    else:
      self._fixed_ids_t = None

    # Look up the command term for fast access during apply_actions.
    self._command_term = env.command_manager.get_term(cfg.command_name)

    # Empty raw action buffer (for the raw_action property).
    self._raw_actions = torch.zeros(
      self.num_envs, 0, device=self.device
    )

  # -- Properties --

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  # -- Methods --

  def process_actions(self, actions: torch.Tensor) -> None:
    pass  # Nothing to process — no policy output consumed.

  def apply_actions(self) -> None:
    # Commanded joints: drive to positions from the command manager.
    cmd_pos = self._command_term.command
    encoder_bias = self._entity.data.encoder_bias[:, self._commanded_ids_t]
    self._entity.set_joint_position_target(
      cmd_pos - encoder_bias, joint_ids=self._commanded_ids_t
    )

    # Fixed joints: hold at default (home) positions.
    if self._fixed_ids_t is not None:
      fixed_bias = self._entity.data.encoder_bias[:, self._fixed_ids_t]
      self._entity.set_joint_position_target(
        self._fixed_default_pos - fixed_bias, joint_ids=self._fixed_ids_t
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    pass  # No internal state to reset.
