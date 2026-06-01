import json
import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    action_manager = self.env.unwrapped.action_manager
    metadata["action_term_names"] = list(action_manager.active_terms)
    metadata["action_term_dims"] = list(action_manager.action_term_dim)
    if "planar_velocity_estimate" in action_manager.active_terms:
      metadata["total_policy_action_dim"] = int(sum(action_manager.action_term_dim))
      metadata["actuated_policy_action_dim"] = int(
        action_manager.action_term_dim[action_manager.active_terms.index("joint_pos")]
      )
      metadata["auxiliary_action_names"] = json.dumps(
        ["estimated_vx_b", "estimated_vy_b", "estimated_wz_b"]
      )
      metadata["deployment_action_note"] = (
        "Apply only the first actuated_policy_action_dim outputs to lower-body "
        "joints; ignore auxiliary velocity-estimate outputs."
      )
    if "joint_pos" in action_manager.active_terms:
      joint_action = action_manager.get_term("joint_pos")
      if hasattr(joint_action, "target_names"):
        metadata["controlled_joint_names"] = list(joint_action.target_names)
      if hasattr(joint_action, "cfg"):
        cfg = joint_action.cfg
        if hasattr(cfg, "smoothing_alpha_range"):
          metadata["action_smoothing_alpha_range"] = list(cfg.smoothing_alpha_range)
        if hasattr(cfg, "delay_steps_range"):
          metadata["action_delay_steps_range"] = list(cfg.delay_steps_range)
    metadata["policy_control_step_dt"] = float(self.env.unwrapped.step_dt)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
