"""Joint-position action with target smoothing and optional control delay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SmoothedJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with first-order smoothing in target space."""

  smoothing_alpha_range: tuple[float, float] = (0.45, 0.65)
  """Range for per-environment smoothing alpha.

  Alpha is the fraction of the newly requested target used each policy step.
  Lower values are smoother and slower.
  """

  delay_steps_range: tuple[int, int] = (0, 0)
  """Inclusive random action-target delay range in control steps."""

  def build(self, env: ManagerBasedRlEnv) -> SmoothedJointPositionAction:
    return SmoothedJointPositionAction(self, env)


class SmoothedJointPositionAction(JointPositionAction):
  """Apply delayed and low-pass filtered position targets."""

  cfg: SmoothedJointPositionActionCfg

  def __init__(
    self, cfg: SmoothedJointPositionActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg, env)

    delay_min, delay_max = cfg.delay_steps_range
    if delay_min < 0 or delay_max < delay_min:
      raise ValueError(
        f"Invalid delay_steps_range={cfg.delay_steps_range}; expected 0 <= min <= max."
      )
    self._delay_min = int(delay_min)
    self._delay_max = int(delay_max)

    self._alpha = torch.empty(self.num_envs, 1, device=self.device)
    self._delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._delay_buffer = torch.zeros(
      self.num_envs, self._delay_max + 1, self.action_dim, device=self.device
    )
    self._smoothed_targets = torch.zeros(
      self.num_envs, self.action_dim, device=self.device
    )
    self._reset_state(torch.arange(self.num_envs, device=self.device))

  def _offset_tensor(self) -> torch.Tensor:
    if isinstance(self._offset, torch.Tensor):
      return self._offset
    return torch.full_like(self._processed_actions, float(self._offset))

  def _reset_state(self, env_ids: torch.Tensor | slice) -> None:
    offset = self._offset_tensor()
    self._smoothed_targets[env_ids] = offset[env_ids]
    self._delay_buffer[env_ids] = offset[env_ids].unsqueeze(1)
    alpha_min, alpha_max = self.cfg.smoothing_alpha_range
    self._alpha[env_ids] = torch.empty_like(self._alpha[env_ids]).uniform_(
      alpha_min, alpha_max
    )
    if self._delay_max > 0:
      if isinstance(env_ids, slice):
        n = self.num_envs
      else:
        n = len(env_ids)
      self._delay_steps[env_ids] = torch.randint(
        self._delay_min,
        self._delay_max + 1,
        (n,),
        device=self.device,
        dtype=torch.long,
      )
    else:
      self._delay_steps[env_ids] = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    requested_targets = self._raw_actions * self._scale + self._offset

    if self._delay_max > 0:
      self._delay_buffer[:, 1:] = self._delay_buffer[:, :-1].clone()
      self._delay_buffer[:, 0] = requested_targets
      batch_ids = torch.arange(self.num_envs, device=self.device)
      requested_targets = self._delay_buffer[batch_ids, self._delay_steps]

    self._processed_actions = (
      self._alpha * requested_targets + (1.0 - self._alpha) * self._smoothed_targets
    )
    self._smoothed_targets[:] = self._processed_actions

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._reset_state(env_ids)
