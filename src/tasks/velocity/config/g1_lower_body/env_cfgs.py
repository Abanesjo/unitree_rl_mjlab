"""Unitree G1 lower-body-only velocity environment configurations.

The policy outputs 12-DOF lower body targets (hips, knees, ankles).
During training the 8 upper body controlled joints are randomized as
disturbances and the 9 remaining upper body joints are held at default.
The policy does not observe upper body targets and receives no reward
for upper body tracking.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.velocity.mdp as mdp

from src.assets.robots import G1_ACTION_SCALE
from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_rough_env_cfg,
)
from src.tasks.velocity.mdp.command_driven_action import (
  CommandDrivenJointPositionActionCfg,
)
from src.tasks.velocity.mdp.joint_position_command import (
  UniformJointPositionCommandCfg,
)

# Filter G1_ACTION_SCALE to lower body joints only.
# resolve_matching_names_values requires every pattern to match at least one
# target, so we cannot pass the full 29-DOF scale dict to a 12-DOF action term.
G1_LOWER_BODY_ACTION_SCALE = {
  k: v for k, v in G1_ACTION_SCALE.items()
  if any(p in k for p in ("hip", "knee", "ankle"))
}

# ---------------------------------------------------------------------------
# Joint groupings
# ---------------------------------------------------------------------------

# Upper body joints whose positions are randomized as disturbances.
CONTROLLED_JOINTS = (
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_elbow_joint",
)

# Sampling ranges for each controlled joint (~80% of joint limits).
CONTROLLED_JOINT_RANGES = {
  "waist_roll_joint": (-0.40, 0.40),
  "waist_pitch_joint": (-0.40, 0.40),
  "left_shoulder_pitch_joint": (-2.0, 2.0),
  "left_shoulder_roll_joint": (-1.0, 1.8),
  "left_elbow_joint": (-0.8, 1.7),
  "right_shoulder_pitch_joint": (-2.0, 2.0),
  "right_shoulder_roll_joint": (-1.8, 1.0),
  "right_elbow_joint": (-0.8, 1.7),
}

# Upper body joints held at their default (home) position.
FIXED_UPPER_BODY_JOINTS = (
  "waist_yaw_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Lower body joint patterns (for reward filtering).
LOWER_BODY_JOINT_PATTERNS = (
  ".*_hip_.*_joint",
  ".*_knee_joint",
  ".*_ankle_.*_joint",
)

# Large std value used to effectively disable pose penalty.
_DISABLED_STD = 100.0


def unitree_g1_lower_body_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain lower-body velocity config."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # --- Replace action space: lower body from policy, upper body from command ---
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"),
      scale=G1_LOWER_BODY_ACTION_SCALE,
      use_default_offset=True,
    ),
    "upper_body_ctrl": CommandDrivenJointPositionActionCfg(
      entity_name="robot",
      command_name="upper_body",
      commanded_joint_names=CONTROLLED_JOINTS,
      fixed_joint_names=FIXED_UPPER_BODY_JOINTS,
    ),
  }

  # --- Add upper body joint position command (disturbance generator) ---
  cfg.commands["upper_body"] = UniformJointPositionCommandCfg(
    entity_name="robot",
    joint_names=CONTROLLED_JOINTS,
    resampling_time_range=(3.0, 8.0),
    rel_default_envs=0.05,
    ranges=CONTROLLED_JOINT_RANGES,
    debug_vis=False,
  )

  # --- Add upper body targets to observations (for anticipation, not control) ---
  cfg.observations["actor"].terms["upper_body_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "upper_body"},
  )
  cfg.observations["critic"].terms["upper_body_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "upper_body"},
  )

  # NOTE: track_upper_body reward is deliberately NOT added.

  # --- Pose penalty: loosen lower body to allow CoG adaptation ---
  # The upper body is randomly perturbed, so the policy needs freedom to
  # bend knees, shift hips, and adjust ankles to compensate — not just
  # during locomotion but also while standing still.
  cfg.rewards["pose"].params["std_standing"] = {
    # Lower body -- loosened to allow adaptive balance.
    r".*_hip_pitch_joint": 0.3,
    r".*_hip_roll_joint": 0.2,
    r".*_hip_yaw_joint": 0.15,
    r".*_knee_joint": 0.3,
    r".*_ankle_pitch_joint": 0.2,
    r".*_ankle_roll_joint": 0.15,
    # All upper body -- effectively disabled.
    r"waist_yaw_joint": _DISABLED_STD,
    r"waist_roll_joint": _DISABLED_STD,
    r"waist_pitch_joint": _DISABLED_STD,
    r".*_shoulder_pitch_joint": _DISABLED_STD,
    r".*_shoulder_roll_joint": _DISABLED_STD,
    r".*_shoulder_yaw_joint": _DISABLED_STD,
    r".*_elbow_joint": _DISABLED_STD,
    r".*_wrist_roll_joint": _DISABLED_STD,
    r".*_wrist_pitch_joint": _DISABLED_STD,
    r".*_wrist_yaw_joint": _DISABLED_STD,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.3,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.3,
    r".*ankle_roll.*": 0.2,
    # All upper body -- effectively disabled.
    r".*waist.*": _DISABLED_STD,
    r".*shoulder.*": _DISABLED_STD,
    r".*elbow.*": _DISABLED_STD,
    r".*wrist.*": _DISABLED_STD,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.4,
    r".*hip_yaw.*": 0.3,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.4,
    r".*ankle_roll.*": 0.2,
    # All upper body -- effectively disabled.
    r".*waist.*": _DISABLED_STD,
    r".*shoulder.*": _DISABLED_STD,
    r".*elbow.*": _DISABLED_STD,
    r".*wrist.*": _DISABLED_STD,
  }

  # --- Restrict stand_still to lower body only ---
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LOWER_BODY_JOINT_PATTERNS,
  )

  # --- Restrict joint_acc_l2 to lower body ---
  # Upper body has uncontrollable accelerations from command resampling.
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=cfg.rewards["joint_acc_l2"].func,
    weight=cfg.rewards["joint_acc_l2"].weight,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)},
  )

  # --- Restrict joint_pos_limits to lower body ---
  # Policy cannot control upper body limit violations.
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=cfg.rewards["joint_pos_limits"].func,
    weight=cfg.rewards["joint_pos_limits"].weight,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)},
  )

  # --- Use pelvis instead of torso for orientation/ang_vel rewards ---
  # The waist joints are randomized, so torso_link tilts even when the robot
  # is perfectly balanced. Pelvis is below the waist and reflects actual
  # base stability.
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

  # --- Stronger push disturbances (removed in play mode by base config) ---
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (1.0, 2.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.8, 0.8),
      "y": (-0.8, 0.8),
      "z": (-0.5, 0.5),
      "roll": (-0.8, 0.8),
      "pitch": (-0.8, 0.8),
      "yaw": (-1.0, 1.0),
    }

  # --- Reduce angular momentum penalty ---
  # Upper body motion generates more angular momentum naturally.
  cfg.rewards["angular_momentum"].weight = -0.01

  return cfg


def unitree_g1_lower_body_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain lower-body velocity config."""
  cfg = unitree_g1_lower_body_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("height_scan", None)

  # Disable terrain curriculum.
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
