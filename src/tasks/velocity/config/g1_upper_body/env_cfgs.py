"""Unitree G1 velocity + upper body control environment configurations."""

from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.mdp.joint_position_command import (
  UniformJointPositionCommandCfg,
)

# Upper body joints to control via commands.
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

# Joint patterns for non-controlled joints (used to exclude controlled joints
# from stand_still and pose rewards).
NON_CONTROLLED_JOINT_PATTERNS = (
  ".*_hip_.*_joint",
  ".*_knee_joint",
  ".*_ankle_.*_joint",
  "waist_yaw_joint",
  ".*_shoulder_yaw_joint",
  ".*_wrist_.*_joint",
)

# Large std value used to effectively disable pose penalty for controlled joints.
_DISABLED_STD = 100.0


def unitree_g1_upper_body_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity + upper body control config."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # --- Add upper body joint position command ---
  cfg.commands["upper_body"] = UniformJointPositionCommandCfg(
    entity_name="robot",
    joint_names=CONTROLLED_JOINTS,
    resampling_time_range=(3.0, 8.0),
    rel_default_envs=0.05,
    ranges=CONTROLLED_JOINT_RANGES,
    debug_vis=False,
  )

  # --- Add upper body command to observations ---
  cfg.observations["actor"].terms["upper_body_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "upper_body"},
  )
  cfg.observations["critic"].terms["upper_body_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "upper_body"},
  )

  # --- Add upper body tracking reward ---
  cfg.rewards["track_upper_body"] = RewardTermCfg(
    func=mdp.track_joint_position,
    weight=1.0,
    params={
      "command_name": "upper_body",
      "std": 0.5,
      "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS),
    },
  )

  # --- Neutralize controlled joints in pose reward ---
  # The resolve_matching_names_values utility does not allow overlapping patterns.
  # std_standing uses a catch-all ".*" so we must replace the entire dict with
  # non-overlapping patterns. For std_walking/std_running the G1 config already
  # uses non-overlapping per-joint patterns, so we just update the values in place.

  # Replace std_standing entirely (original is {".*": 0.05}).
  cfg.rewards["pose"].params["std_standing"] = {
    # Non-controlled joints — tight standing tolerance.
    r".*_hip_pitch_joint": 0.05,
    r".*_hip_roll_joint": 0.05,
    r".*_hip_yaw_joint": 0.05,
    r".*_knee_joint": 0.05,
    r".*_ankle_pitch_joint": 0.05,
    r".*_ankle_roll_joint": 0.05,
    r"waist_yaw_joint": 0.05,
    r".*_shoulder_yaw_joint": 0.05,
    r".*_wrist_roll_joint": 0.05,
    r".*_wrist_pitch_joint": 0.05,
    r".*_wrist_yaw_joint": 0.05,
    # Controlled joints — effectively disabled.
    r"waist_roll_joint": _DISABLED_STD,
    r"waist_pitch_joint": _DISABLED_STD,
    r".*_shoulder_pitch_joint": _DISABLED_STD,
    r".*_shoulder_roll_joint": _DISABLED_STD,
    r".*_elbow_joint": _DISABLED_STD,
  }

  # Update std_walking and std_running in place (non-overlapping patterns).
  for regime in ("std_walking", "std_running"):
    std_dict = cfg.rewards["pose"].params[regime]
    std_dict[r".*waist_roll.*"] = _DISABLED_STD
    std_dict[r".*waist_pitch.*"] = _DISABLED_STD
    std_dict[r".*shoulder_pitch.*"] = _DISABLED_STD
    std_dict[r".*shoulder_roll.*"] = _DISABLED_STD
    std_dict[r".*elbow.*"] = _DISABLED_STD

  # --- Exclude controlled joints from stand_still ---
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=NON_CONTROLLED_JOINT_PATTERNS,
  )

  # --- Reduce angular momentum penalty ---
  # Upper body motion generates more angular momentum naturally.
  cfg.rewards["angular_momentum"].weight = -0.01

  return cfg


def unitree_g1_upper_body_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity + upper body control config."""
  cfg = unitree_g1_upper_body_rough_env_cfg(play=play)

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
