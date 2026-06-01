"""No-op action term for auxiliary planar velocity estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class PlanarVelocityEstimateActionCfg(ActionTermCfg):
  """Config for an action term that exposes a 3D velocity-estimate output."""

  def build(self, env: ManagerBasedRlEnv) -> PlanarVelocityEstimateAction:
    return PlanarVelocityEstimateAction(self, env)


class PlanarVelocityEstimateAction(ActionTerm):
  """Consumes policy outputs for [vx_b, vy_b, wz_b] and applies nothing."""

  cfg: PlanarVelocityEstimateActionCfg

  def __init__(
    self, cfg: PlanarVelocityEstimateActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg, env)
    self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def action_dim(self) -> int:
    return 3

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

  def apply_actions(self) -> None:
    pass

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
