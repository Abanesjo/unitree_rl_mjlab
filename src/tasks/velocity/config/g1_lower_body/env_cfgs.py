"""Unitree G1 lower-body stationary balance environment configurations.

The policy outputs 12-DOF lower body targets (hips, knees, ankles), plus
3 auxiliary planar velocity estimates. During training the 8 upper body
controlled joints are randomized as disturbances and the 9 remaining upper
body joints are held at default.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
from src.tasks.velocity.mdp.velocity_estimate_action import (
  PlanarVelocityEstimateActionCfg,
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
  """Create Unitree G1 rough terrain lower-body stationary config."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # --- Replace action space: legs from policy, plus auxiliary velocity estimate ---
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"),
      scale=G1_LOWER_BODY_ACTION_SCALE,
      use_default_offset=True,
    ),
    "planar_velocity_estimate": PlanarVelocityEstimateActionCfg(
      entity_name="robot",
    ),
    "upper_body_ctrl": CommandDrivenJointPositionActionCfg(
      entity_name="robot",
      command_name="upper_body",
      commanded_joint_names=CONTROLLED_JOINTS,
      fixed_joint_names=FIXED_UPPER_BODY_JOINTS,
    ),
  }

  # --- Replace commands: only upper-body joint position disturbances remain ---
  cfg.commands = {}
  cfg.commands["upper_body"] = UniformJointPositionCommandCfg(
    entity_name="robot",
    joint_names=CONTROLLED_JOINTS,
    resampling_time_range=(3.0, 8.0),
    rel_default_envs=0.05,
    ranges=CONTROLLED_JOINT_RANGES,
    debug_vis=False,
  )

  # --- Remove locomotion observations and expose upper-body command only ---
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms.pop("command", None)
    terms.pop("phase", None)
    terms["actions"].params = {"action_name": "joint_pos"}
    terms["upper_body_command"] = ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "upper_body"},
    )

  # --- Remove locomotion command curriculum ---
  cfg.curriculum = {}
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False

  # Spawn exactly at each environment origin so the stationary target is explicit.
  cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
  cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)

  # --- Replace locomotion rewards with stationary balance rewards ---
  for reward_name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "foot_gait",
    "foot_clearance",
  ):
    cfg.rewards.pop(reward_name, None)

  cfg.rewards["root_xy_displacement_l2"] = RewardTermCfg(
    func=mdp.root_xy_displacement_l2,
    weight=-10.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["root_planar_velocity_l2"] = RewardTermCfg(
    func=mdp.root_planar_velocity_l2,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["track_planar_velocity_estimate"] = RewardTermCfg(
    func=mdp.track_planar_velocity_estimate,
    weight=0.5,
    params={
      "action_name": "planar_velocity_estimate",
      "std": 0.5,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_term_rate_l2,
    weight=cfg.rewards["action_rate_l2"].weight,
    params={"action_name": "joint_pos"},
  )

  # --- Pose reward: loosen lower body to allow CoG adaptation ---
  # The upper body is randomly perturbed, so the policy needs freedom to
  # bend knees, shift hips, and adjust ankles to compensate.
  cfg.rewards["pose"] = RewardTermCfg(
    func=mdp.default_joint_position,
    weight=cfg.rewards["pose"].weight,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "std": {
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
      },
    },
  )

  # --- Restrict stand_still to lower body only ---
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still,
    weight=cfg.rewards["stand_still"].weight,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)
    },
  )

  # --- Restrict joint_acc_l2 to lower body ---
  # Upper body has uncontrollable accelerations from command resampling.
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=cfg.rewards["joint_acc_l2"].func,
    weight=cfg.rewards["joint_acc_l2"].weight,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)
    },
  )

  # --- Restrict joint_pos_limits to lower body ---
  # Policy cannot control upper body limit violations.
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=cfg.rewards["joint_pos_limits"].func,
    weight=cfg.rewards["joint_pos_limits"].weight,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)
    },
  )

  # --- Make foot penalties command-free ---
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=mdp.feet_slip,
    weight=cfg.rewards["foot_slip"].weight,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": cfg.rewards["foot_slip"].params["asset_cfg"],
    },
  )
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=mdp.soft_landing,
    weight=cfg.rewards["soft_landing"].weight,
    params={"sensor_name": "feet_ground_contact"},
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
  """Create Unitree G1 flat terrain lower-body stationary config."""
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

  # No terrain or velocity-command curriculum for the stationary task.
  cfg.curriculum = {}

  return cfg
