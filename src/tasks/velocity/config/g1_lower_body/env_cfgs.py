"""Unitree G1 lower-body balance environment configurations.

The policy outputs 12 lower-body joint targets plus 3 no-op auxiliary planar
velocity estimates. Upper-body joints are driven by a separate PD action from
smooth disturbance commands and are never controlled by the policy.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import src.tasks.velocity.mdp as mdp

from src.assets.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_rough_env_cfg,
)
from src.tasks.velocity.mdp.command_driven_action import (
  CommandDrivenJointPositionActionCfg,
)
from src.tasks.velocity.mdp.smoothed_joint_position_action import (
  SmoothedJointPositionActionCfg,
)
from src.tasks.velocity.mdp.upper_body_disturbance_command import (
  UpperBodyDisturbanceCommandCfg,
)
from src.tasks.velocity.mdp.velocity_estimate_action import (
  PlanarVelocityEstimateActionCfg,
)

G1_LOWER_BODY_ACTION_SCALE = {
  r".*_hip_pitch_joint": 0.70,
  r".*_hip_roll_joint": 0.45,
  r".*_hip_yaw_joint": 0.30,
  r".*_knee_joint": 0.90,
  r".*_ankle_pitch_joint": 0.55,
  r".*_ankle_roll_joint": 0.25,
}

# ---------------------------------------------------------------------------
# Joint groupings
# ---------------------------------------------------------------------------

# Upper body joints driven by the separate PD controller.
UPPER_BODY_JOINTS = (
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Sampling ranges for each upper-body joint. These are intentionally narrower
# than hard XML limits to avoid training mostly on self-colliding postures.
UPPER_BODY_JOINT_RANGES = {
  "waist_yaw_joint": (-0.60, 0.60),
  "waist_roll_joint": (-0.40, 0.40),
  "waist_pitch_joint": (-0.40, 0.40),
  "left_shoulder_pitch_joint": (-2.0, 2.0),
  "left_shoulder_roll_joint": (-1.1, 1.6),
  "left_shoulder_yaw_joint": (-1.2, 1.2),
  "left_elbow_joint": (-0.7, 1.8),
  "left_wrist_roll_joint": (-0.5, 0.5),
  "left_wrist_pitch_joint": (-0.5, 0.5),
  "left_wrist_yaw_joint": (-0.5, 0.5),
  "right_shoulder_pitch_joint": (-2.0, 2.0),
  "right_shoulder_roll_joint": (-1.6, 1.1),
  "right_shoulder_yaw_joint": (-1.2, 1.2),
  "right_elbow_joint": (-0.7, 1.8),
  "right_wrist_roll_joint": (-0.5, 0.5),
  "right_wrist_pitch_joint": (-0.5, 0.5),
  "right_wrist_yaw_joint": (-0.5, 0.5),
}

FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
BAD_GROUND_GEOM_NAMES = (
  "pelvis_collision",
  "torso_collision",
  "head_collision",
  "left_shoulder_yaw_collision",
  "left_elbow_yaw_collision",
  "left_wrist_collision",
  "left_hand_collision",
  "right_shoulder_yaw_collision",
  "right_elbow_yaw_collision",
  "right_wrist_collision",
  "right_hand_collision",
)

# Lower body joint patterns (for reward filtering).
LOWER_BODY_JOINT_PATTERNS = (
  ".*_hip_.*_joint",
  ".*_knee_joint",
  ".*_ankle_.*_joint",
)

# Large std value used to effectively disable pose penalty.
_DISABLED_STD = 100.0

RECOVERY_FOOT_SELECTION_PARAMS = {
  "min_reach": 0.30,
  "max_reach": 0.72,
  "capture_reach_gain": 1.05,
  "velocity_reach_gain": 0.95,
  "direction_com_gain": 0.60,
  "direction_velocity_gain": 0.95,
  "direction_deadband": 0.04,
  "sagittal_bias_gain": 1.35,
  "lateral_suppression": 0.65,
  "sagittal_activation": 0.08,
  "risk_activation": 0.12,
  "target_velocity": 0.22,
  "need_scale": 0.08,
  "dynamic_need_weight": 0.35,
}

BALANCED_FOOT_CHOICE_PARAMS = {
  "need_activation": 0.05,
  "need_width": 0.35,
  "raw_lateral_activation": 0.18,
  "lateral_activation": 0.55,
  "lateral_dominance": 0.85,
  "balanced_period_s": 3.0,
}


def _push_stage(
  step: int,
  force_magnitude_range: tuple[float, float] = (0.0, 0.0),
  duration_s: tuple[float, float] = (0.06, 0.09),
  cooldown_s: tuple[float, float] = (8.0, 12.0),
) -> dict:
  return {
    "step": step,
    "force_magnitude_range": force_magnitude_range,
    "force_z_range": (0.0, 0.0),
    "torque_range": (0.0, 0.0),
    "duration_s": duration_s,
    "cooldown_s": cooldown_s,
  }


def unitree_g1_lower_body_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain lower-body stationary config."""
  cfg = unitree_g1_rough_env_cfg(play=play)
  cfg.sim.nconmax = None
  cfg.scene.entities["robot"].init_state = KNEES_BENT_KEYFRAME

  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=BAD_GROUND_GEOM_NAMES,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (nonfoot_ground_cfg,)

  # --- Replace action space: legs from policy, plus auxiliary velocity estimate ---
  cfg.actions = {
    "joint_pos": SmoothedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"),
      scale=G1_LOWER_BODY_ACTION_SCALE,
      use_default_offset=True,
      smoothing_alpha_range=(0.65, 0.80),
      delay_steps_range=(0, 0),
    ),
    "planar_velocity_estimate": PlanarVelocityEstimateActionCfg(
      entity_name="robot",
    ),
    "upper_body_ctrl": CommandDrivenJointPositionActionCfg(
      entity_name="robot",
      command_name="upper_body",
      commanded_joint_names=UPPER_BODY_JOINTS,
    ),
  }

  # --- Replace commands: upper-body joint-position disturbances only ---
  cfg.commands = {}
  cfg.commands["upper_body"] = UpperBodyDisturbanceCommandCfg(
    entity_name="robot",
    joint_names=UPPER_BODY_JOINTS,
    resampling_time_range=(4.0, 8.0),
    rel_default_envs=1.00,
    ranges=UPPER_BODY_JOINT_RANGES,
    mode_probabilities=(1.0, 0.0, 0.0, 0.0),
    amplitude_scale=0.00,
    random_walk_velocity_range=(0.05, 0.20),
    random_walk_acceleration_range=(0.10, 0.60),
    sinusoid_frequency_range=(0.25, 0.80),
    pulse_duration_range=(0.60, 1.60),
    debug_vis=False,
  )

  if not play:
    cfg.events["push_robot"] = EventTermCfg(
      func=mdp.apply_planar_body_force_pulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force_magnitude_range": (0.0, 0.0),
        "force_z_range": (0.0, 0.0),
        "torque_range": (0.0, 0.0),
        "duration_s": (0.06, 0.09),
        "cooldown_s": (8.0, 12.0),
        "body_point_offset": (0.0, 0.0, -0.20),
      },
    )
    cfg.events["recovery_drill_reset"] = EventTermCfg(
      func=mdp.reset_recovery_drill_state,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "probability": 0.0,
        "planar_speed_range": (0.0, 0.0),
        "tilt_range": (0.0, 0.0),
        "angular_speed_range": (0.0, 0.0),
        "yaw_rate_range": (0.0, 0.0),
        "height_offset_range": (0.0, 0.0),
      },
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
    terms["foot_pos_b"] = ObservationTermCfg(
      func=mdp.foot_pos_b,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)},
    )
    terms["foot_vel_b"] = ObservationTermCfg(
      func=mdp.foot_vel_b,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)},
    )

  cfg.observations["actor"].history_length = 6
  cfg.observations["critic"].history_length = 1
  cfg.observations["critic"].terms["whole_body_com_b"] = ObservationTermCfg(
    func=mdp.whole_body_com_b,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.observations["critic"].terms["root_planar_velocity"] = ObservationTermCfg(
    func=mdp.root_planar_velocity,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False

  if play:
    cfg.curriculum = {}
  else:
    cfg.curriculum = {
      "upper_body_disturbance": CurriculumTermCfg(
        func=mdp.upper_body_disturbance,
        params={
          "command_name": "upper_body",
          "stages": [
            {
              "step": 0,
              "mode_probabilities": (1.0, 0.0, 0.0, 0.0),
              "amplitude_scale": 0.00,
              "rel_default_envs": 1.00,
              "resampling_time_range": (4.0, 8.0),
            },
            {
              "step": 120 * 48,
              "mode_probabilities": (1.0, 0.0, 0.0, 0.0),
              "amplitude_scale": 0.08,
              "rel_default_envs": 0.70,
              "resampling_time_range": (3.5, 6.0),
            },
            {
              "step": 300 * 48,
              "mode_probabilities": (1.0, 0.0, 0.0, 0.0),
              "amplitude_scale": 0.18,
              "rel_default_envs": 0.45,
              "resampling_time_range": (3.0, 6.0),
            },
            {
              "step": 700 * 48,
              "mode_probabilities": (1.0, 0.0, 0.0, 0.0),
              "amplitude_scale": 0.32,
              "rel_default_envs": 0.25,
              "resampling_time_range": (2.5, 5.0),
            },
            {
              "step": 1300 * 48,
              "mode_probabilities": (0.75, 0.20, 0.05, 0.0),
              "amplitude_scale": 0.50,
              "rel_default_envs": 0.15,
              "resampling_time_range": (2.5, 5.0),
              "random_walk_velocity_range": (0.05, 0.30),
              "random_walk_acceleration_range": (0.12, 0.80),
              "sinusoid_frequency_range": (0.25, 0.90),
            },
            {
              "step": 2400 * 48,
              "mode_probabilities": (0.45, 0.35, 0.15, 0.05),
              "amplitude_scale": 0.60,
              "rel_default_envs": 0.10,
              "random_walk_velocity_range": (0.08, 0.45),
              "random_walk_acceleration_range": (0.20, 1.20),
              "sinusoid_frequency_range": (0.25, 1.20),
            },
            {
              "step": 5200 * 48,
              "mode_probabilities": (0.35, 0.35, 0.20, 0.10),
              "amplitude_scale": 0.70,
              "rel_default_envs": 0.08,
              "random_walk_velocity_range": (0.12, 0.65),
              "random_walk_acceleration_range": (0.30, 1.60),
              "sinusoid_frequency_range": (0.40, 1.60),
              "pulse_duration_range": (0.40, 1.20),
            },
            {
              "step": 8000 * 48,
              "mode_probabilities": (0.25, 0.35, 0.25, 0.15),
              "amplitude_scale": 0.80,
              "rel_default_envs": 0.05,
              "random_walk_velocity_range": (0.15, 0.80),
              "random_walk_acceleration_range": (0.40, 2.00),
              "sinusoid_frequency_range": (0.50, 2.00),
              "pulse_duration_range": (0.35, 1.00),
            },
            {
              "step": 14600 * 48,
              "mode_probabilities": (0.75, 0.20, 0.05, 0.0),
              "amplitude_scale": 0.25,
              "rel_default_envs": 0.15,
              "random_walk_velocity_range": (0.05, 0.25),
              "random_walk_acceleration_range": (0.12, 0.70),
              "sinusoid_frequency_range": (0.25, 0.90),
              "pulse_duration_range": (0.45, 1.20),
            },
            {
              "step": 15400 * 48,
              "mode_probabilities": (0.55, 0.30, 0.12, 0.03),
              "amplitude_scale": 0.40,
              "rel_default_envs": 0.10,
              "random_walk_velocity_range": (0.08, 0.35),
              "random_walk_acceleration_range": (0.18, 1.00),
              "sinusoid_frequency_range": (0.30, 1.15),
              "pulse_duration_range": (0.40, 1.10),
            },
            {
              "step": 16500 * 48,
              "mode_probabilities": (0.50, 0.32, 0.14, 0.04),
              "amplitude_scale": 0.45,
              "rel_default_envs": 0.08,
              "random_walk_velocity_range": (0.08, 0.40),
              "random_walk_acceleration_range": (0.20, 1.10),
              "sinusoid_frequency_range": (0.30, 1.20),
              "pulse_duration_range": (0.40, 1.10),
            },
            {
              "step": 18000 * 48,
              "mode_probabilities": (0.40, 0.35, 0.18, 0.07),
              "amplitude_scale": 0.55,
              "rel_default_envs": 0.07,
              "random_walk_velocity_range": (0.10, 0.50),
              "random_walk_acceleration_range": (0.25, 1.35),
              "sinusoid_frequency_range": (0.35, 1.45),
              "pulse_duration_range": (0.38, 1.05),
            },
            {
              "step": 20500 * 48,
              "mode_probabilities": (0.30, 0.35, 0.23, 0.12),
              "amplitude_scale": 0.70,
              "rel_default_envs": 0.05,
              "random_walk_velocity_range": (0.13, 0.70),
              "random_walk_acceleration_range": (0.35, 1.80),
              "sinusoid_frequency_range": (0.45, 1.85),
              "pulse_duration_range": (0.35, 1.00),
            },
          ],
        },
      ),
      "drift_weight": CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
          "reward_name": "root_xy_drift_huber",
          "weight_stages": [
            {"step": 0, "weight": -0.8},
            {"step": 4000 * 48, "weight": -0.65},
            {"step": 14500 * 48, "weight": -0.45},
            {"step": 18000 * 48, "weight": -0.35},
          ],
        },
      ),
      "recovery_drill": CurriculumTermCfg(
        func=mdp.recovery_drill_curriculum,
        params={
          "event_name": "recovery_drill_reset",
          "stages": [
            {
              "step": 0,
              "probability": 0.0,
              "planar_speed_range": (0.0, 0.0),
              "tilt_range": (0.0, 0.0),
              "angular_speed_range": (0.0, 0.0),
              "yaw_rate_range": (0.0, 0.0),
            },
            {
              "step": 14500 * 48,
              "probability": 0.20,
              "planar_speed_range": (0.08, 0.20),
              "tilt_range": (math.radians(2.0), math.radians(5.0)),
              "angular_speed_range": (0.15, 0.35),
              "yaw_rate_range": (-0.08, 0.08),
            },
            {
              "step": 14800 * 48,
              "probability": 0.40,
              "planar_speed_range": (0.16, 0.38),
              "tilt_range": (math.radians(4.0), math.radians(10.0)),
              "angular_speed_range": (0.25, 0.75),
              "yaw_rate_range": (-0.12, 0.12),
            },
            {
              "step": 15200 * 48,
              "probability": 0.50,
              "planar_speed_range": (0.20, 0.48),
              "tilt_range": (math.radians(5.0), math.radians(12.0)),
              "angular_speed_range": (0.35, 0.95),
              "yaw_rate_range": (-0.16, 0.16),
            },
            {
              "step": 16500 * 48,
              "probability": 0.54,
              "planar_speed_range": (0.24, 0.58),
              "tilt_range": (math.radians(6.0), math.radians(14.0)),
              "angular_speed_range": (0.45, 1.10),
              "yaw_rate_range": (-0.22, 0.22),
            },
            {
              "step": 18500 * 48,
              "probability": 0.38,
              "planar_speed_range": (0.22, 0.55),
              "tilt_range": (math.radians(5.0), math.radians(13.0)),
              "angular_speed_range": (0.35, 1.10),
              "yaw_rate_range": (-0.24, 0.24),
            },
            {
              "step": 21500 * 48,
              "probability": 0.48,
              "planar_speed_range": (0.26, 0.65),
              "tilt_range": (math.radians(6.0), math.radians(16.0)),
              "angular_speed_range": (0.45, 1.30),
              "yaw_rate_range": (-0.30, 0.30),
            },
            {
              "step": 23500 * 48,
              "probability": 0.54,
              "planar_speed_range": (0.30, 0.75),
              "tilt_range": (math.radians(7.0), math.radians(18.0)),
              "angular_speed_range": (0.55, 1.50),
              "yaw_rate_range": (-0.34, 0.34),
            },
            {
              "step": 25500 * 48,
              "probability": 0.65,
              "planar_speed_range": (0.35, 0.85),
              "tilt_range": (math.radians(8.0), math.radians(20.0)),
              "angular_speed_range": (0.65, 1.65),
              "yaw_rate_range": (-0.38, 0.38),
            },
            {
              "step": 27500 * 48,
              "probability": 0.75,
              "planar_speed_range": (0.40, 1.00),
              "tilt_range": (math.radians(9.0), math.radians(23.0)),
              "angular_speed_range": (0.70, 1.90),
              "yaw_rate_range": (-0.45, 0.45),
            },
          ],
        },
      ),
      "push_robot": CurriculumTermCfg(
        func=mdp.push_robot_curriculum,
        params={
          "event_name": "push_robot",
          "stages": [
            _push_stage(0),
            _push_stage(9000 * 48),
            _push_stage(12000 * 48, (20.0, 35.0), (0.04, 0.06)),
            _push_stage(13500 * 48, (25.0, 40.0), (0.04, 0.07), (6.0, 10.0)),
            _push_stage(13700 * 48, (30.0, 45.0), (0.05, 0.08), (5.0, 8.0)),
            _push_stage(14200 * 48, (35.0, 50.0), (0.05, 0.08), (5.0, 8.0)),
            _push_stage(14600 * 48, duration_s=(0.06, 0.10), cooldown_s=(6.0, 9.0)),
            _push_stage(14800 * 48, duration_s=(0.06, 0.10), cooldown_s=(6.0, 9.0)),
            _push_stage(15400 * 48, (20.0, 35.0), (0.04, 0.07), (6.0, 9.0)),
            _push_stage(15800 * 48, (30.0, 45.0), (0.05, 0.08), (5.0, 7.0)),
            _push_stage(16200 * 48, (35.0, 50.0), (0.05, 0.08), (5.0, 7.0)),
            _push_stage(16500 * 48, (35.0, 50.0), (0.05, 0.08), (4.5, 6.5)),
            _push_stage(17500 * 48, (35.0, 50.0), (0.06, 0.09), (4.0, 6.0)),
            _push_stage(18500 * 48, (35.0, 50.0), (0.06, 0.09), (3.5, 5.5)),
            _push_stage(20500 * 48, (35.0, 50.0), (0.06, 0.08), (2.2, 3.8)),
            _push_stage(22500 * 48, (35.0, 50.0), (0.07, 0.10), (2.0, 4.0)),
            _push_stage(25000 * 48, (35.0, 50.0), (0.08, 0.12), (2.0, 4.0)),
          ],
        },
      ),
    }

  # Spawn exactly at each environment origin so the stationary target is explicit.
  cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
  cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
  if not play:
    cfg.events["reset_base"].params["velocity_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }

  # --- Replace locomotion rewards with stationary balance rewards ---
  for reward_name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "foot_gait",
    "foot_clearance",
  ):
    cfg.rewards.pop(reward_name, None)

  cfg.rewards["alive"] = RewardTermCfg(func=mdp.alive, weight=2.0)
  cfg.rewards["root_xy_drift_huber"] = RewardTermCfg(
    func=mdp.root_xy_drift_huber,
    weight=-0.8,
    params={
      "deadband": 0.45,
      "linear_width": 0.60,
      "risk_reduction": 0.90,
      "min_scale": 0.08,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["root_xy_return_velocity"] = RewardTermCfg(
    func=mdp.root_xy_return_velocity_bonus,
    weight=0.40,
    params={
      "deadband": 0.50,
      "displacement_width": 0.60,
      "target_return_speed": 0.35,
      "stable_risk": 0.32,
      "max_bonus": 1.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["root_planar_velocity_saturating"] = RewardTermCfg(
    func=mdp.root_planar_velocity_saturating_risk_gated,
    weight=-1.1,
    params={
      "saturation_speed": 0.70,
      "risk_reduction": 0.88,
      "min_scale": 0.12,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["supported_root_planar_velocity_brake"] = RewardTermCfg(
    func=mdp.supported_root_planar_velocity_brake,
    weight=-1.25,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "speed_deadband": 0.70,
      "saturation_speed": 0.85,
      "min_contacts": 1.50,
      "risk_activation": 0.45,
    },
  )
  cfg.rewards["com_support_margin_violation"] = RewardTermCfg(
    func=mdp.com_support_margin_violation,
    weight=-12.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=FOOT_SITE_NAMES, body_names=FOOT_BODY_NAMES
      ),
      "foot_half_length": 0.10,
      "foot_half_width": 0.04,
      "margin": 0.02,
    },
  )
  cfg.rewards["capture_point_support_margin_violation"] = RewardTermCfg(
    func=mdp.capture_point_support_margin_violation,
    weight=-8.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=FOOT_SITE_NAMES, body_names=FOOT_BODY_NAMES
      ),
      "foot_half_length": 0.10,
      "foot_half_width": 0.04,
      "margin": 0.03,
      "max_capture_offset": 0.80,
    },
  )
  cfg.rewards["capture_margin_improvement"] = RewardTermCfg(
    func=mdp.capture_margin_improvement_reward,
    weight=4.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=FOOT_SITE_NAMES, body_names=FOOT_BODY_NAMES
      ),
      "foot_half_length": 0.10,
      "foot_half_width": 0.04,
      "margin": 0.03,
      "max_capture_offset": 0.80,
      "delta_scale": 0.035,
      "max_reward": 0.8,
      "max_penalty": 1.0,
    },
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
  cfg.rewards["pelvis_tilt_barrier"] = RewardTermCfg(
    func=mdp.pelvis_tilt_barrier,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      "soft_limit": 0.20,
      "hard_limit": 0.95,
    },
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_term_rate_l2_risk_gated,
    weight=-0.045,
    params={
      "action_name": "joint_pos",
      "risk_reduction": 0.80,
      "min_scale": 0.20,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_term_acc_l2_risk_gated,
    weight=-0.015,
    params={
      "action_name": "joint_pos",
      "risk_reduction": 0.85,
      "min_scale": 0.15,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["pelvis_ang_acc_l2"] = RewardTermCfg(
    func=mdp.body_angular_acceleration_penalty,
    weight=-3.0e-4,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
  )

  # --- Pose reward: strict at low risk, loose during active recovery ---
  cfg.rewards["pose"] = RewardTermCfg(
    func=mdp.risk_gated_default_joint_position,
    weight=0.45,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "low_risk_std": {
        r".*_hip_pitch_joint": 0.35,
        r".*_hip_roll_joint": 0.20,
        r".*_hip_yaw_joint": 0.18,
        r".*_knee_joint": 0.40,
        r".*_ankle_pitch_joint": 0.25,
        r".*_ankle_roll_joint": 0.16,
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
      "high_risk_std": {
        r".*_hip_pitch_joint": 0.90,
        r".*_hip_roll_joint": 0.55,
        r".*_hip_yaw_joint": 0.35,
        r".*_knee_joint": 1.10,
        r".*_ankle_pitch_joint": 0.70,
        r".*_ankle_roll_joint": 0.30,
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

  cfg.rewards["stance_geometry"] = RewardTermCfg(
    func=mdp.stance_geometry_penalty,
    weight=-32.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "nominal_width": 0.22,
      "risk_width": 0.44,
      "soft_max_width": 0.54,
      "max_width": 0.62,
      "soft_overwidth_weight": 4.0,
      "soft_overwidth_risk_activation": 0.15,
      "risk_split_gain": 2.30,
      "split_velocity_gain": 2.30,
      "risk_min_split": 0.42,
      "width_lateral_velocity_gain": 0.25,
      "max_split": 0.86,
    },
  )
  cfg.rewards["capture_point_reach"] = RewardTermCfg(
    func=mdp.capture_point_reach_penalty,
    weight=-48.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_fore_aft_reach": 0.44,
      "max_fore_aft_reach": 0.78,
      "capture_fore_aft_gain": 2.30,
      "velocity_fore_aft_gain": 2.40,
      "min_lateral_reach": 0.20,
      "max_lateral_reach": 0.40,
      "capture_lateral_gain": 0.75,
      "velocity_lateral_gain": 0.50,
      "direction_velocity_gain": 0.80,
      "direction_deadband": 0.025,
      "lateral_weight": 0.30,
    },
  )
  cfg.rewards["recovery_step_clearance"] = RewardTermCfg(
    func=mdp.recovery_step_clearance_penalty,
    weight=-24.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_fore_aft_reach": 0.32,
      "max_fore_aft_reach": 0.74,
      "com_fore_aft_gain": 0.95,
      "velocity_fore_aft_gain": 1.45,
      "min_lateral_reach": 0.16,
      "max_lateral_reach": 0.36,
      "com_lateral_gain": 0.35,
      "velocity_lateral_gain": 0.45,
      "direction_com_gain": 0.45,
      "direction_velocity_gain": 0.75,
      "foot_tie_break_scale": 1.0e-2,
      "foot_tie_break_period_s": 3.0,
      "clearance_height": 0.055,
      "lateral_weight": 0.30,
    },
  )
  cfg.rewards["recovery_step_velocity"] = RewardTermCfg(
    func=mdp.recovery_step_velocity_penalty,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_fore_aft_reach": 0.32,
      "max_fore_aft_reach": 0.74,
      "com_fore_aft_gain": 0.95,
      "velocity_fore_aft_gain": 1.45,
      "min_lateral_reach": 0.16,
      "max_lateral_reach": 0.36,
      "com_lateral_gain": 0.35,
      "velocity_lateral_gain": 0.45,
      "direction_com_gain": 0.45,
      "direction_velocity_gain": 0.75,
      "direction_deadband": 0.025,
      "foot_tie_break_scale": 1.0e-2,
      "foot_tie_break_period_s": 3.0,
      "min_step_velocity": 0.20,
      "velocity_target_gain": 0.90,
      "max_step_velocity": 0.75,
      "lateral_weight": 0.30,
    },
  )
  cfg.rewards["recovery_step_contact_phase"] = RewardTermCfg(
    func=mdp.recovery_step_contact_phase_penalty,
    weight=-35.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_reach": 0.32,
      "max_reach": 0.72,
      "com_reach_gain": 0.75,
      "velocity_reach_gain": 0.90,
      "capture_reach_gain": 1.10,
      "direction_com_gain": 0.60,
      "direction_velocity_gain": 0.90,
      "direction_deadband": 0.04,
      "sagittal_bias_gain": 1.35,
      "lateral_suppression": 0.65,
      "sagittal_activation": 0.08,
      "foot_tie_break_scale": 1.0e-2,
      "foot_tie_break_period_s": 3.0,
      "risk_activation": 0.12,
      "clearance_height": 0.055,
      "min_step_velocity": 0.14,
      "recontact_margin": 0.03,
      "support_contact_weight": 1.5,
      "stuck_contact_weight": 1.2,
      "clearance_weight": 1.0,
      "velocity_weight": 0.8,
      "recontact_weight": 0.6,
      "no_support_weight": 2.0,
      "no_swing_weight": 1.00,
      "need_scale": 0.08,
      "dynamic_need_weight": 0.30,
    },
  )
  cfg.rewards["recovery_swing_bonus"] = RewardTermCfg(
    func=mdp.recovery_swing_bonus,
    weight=6.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_reach": 0.30,
      "max_reach": 0.72,
      "capture_reach_gain": 1.10,
      "velocity_reach_gain": 0.95,
      "direction_com_gain": 0.60,
      "direction_velocity_gain": 0.95,
      "sagittal_bias_gain": 1.35,
      "lateral_suppression": 0.65,
      "sagittal_activation": 0.08,
      "foot_tie_break_scale": 1.0e-2,
      "foot_tie_break_period_s": 3.0,
      "risk_activation": 0.12,
      "target_clearance": 0.06,
      "target_velocity": 0.22,
      "target_lift_velocity": 0.18,
      "need_scale": 0.08,
      "dynamic_need_weight": 0.35,
      "completion_progress_power": 2.0,
    },
  )
  cfg.rewards["recovery_step_progress"] = RewardTermCfg(
    func=mdp.recovery_step_progress_bonus,
    weight=18.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_reach": 0.28,
      "max_reach": 0.72,
      "capture_reach_gain": 1.05,
      "velocity_reach_gain": 0.95,
      "direction_com_gain": 0.60,
      "direction_velocity_gain": 0.95,
      "direction_deadband": 0.04,
      "sagittal_bias_gain": 1.35,
      "lateral_suppression": 0.65,
      "sagittal_activation": 0.08,
      "foot_tie_break_scale": 1.0e-2,
      "foot_tie_break_period_s": 3.0,
      "risk_activation": 0.12,
      "target_velocity": 0.22,
      "target_clearance": 0.055,
      "progress_scale": 0.035,
      "advance_scale": 0.16,
      "need_scale": 0.08,
      "dynamic_need_weight": 0.35,
      "reach_weight": 0.45,
      "progress_weight": 0.25,
      "velocity_weight": 0.05,
      "airborne_progress_weight": 0.25,
      "recontact_weight": 2.20,
      "recontact_advance_weight": 0.70,
      "recontact_target_weight": 0.30,
      "modest_recontact_weight": 1.20,
      "modest_recontact_margin": 0.030,
      "modest_recontact_scale": 0.14,
      "modest_recontact_min_support": 1.5,
      "latch_need_threshold": 0.15,
      "release_need_threshold": 0.05,
      "recovery_memory_s": 0.70,
      "recovery_retention_risk": 0.22,
      "latched_need_memory": 0.995,
      "useful_recontact_threshold": 0.005,
      "stabilize_window_s": 0.38,
      "stabilize_weight": 0.45,
      "stable_state_weight": 0.45,
      "speed_stable_scale": 0.35,
      "tilt_stable_scale": 0.18,
      "risk_stable_target": 0.35,
      "min_air_time": 0.04,
      "max_recontact_time": 0.30,
      "recontact_margin": 0.04,
    },
  )
  cfg.rewards["recovery_step_completion"] = RewardTermCfg(
    func=mdp.recovery_step_completion_bonus,
    weight=28.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_reach": 0.32,
      "max_reach": 0.72,
      "capture_reach_gain": 1.05,
      "velocity_reach_gain": 0.90,
      "direction_com_gain": 0.60,
      "direction_velocity_gain": 0.95,
      "direction_deadband": 0.04,
      "sagittal_bias_gain": 1.35,
      "lateral_suppression": 0.65,
      "sagittal_activation": 0.08,
      "foot_tie_break_scale": 1.0e-2,
      "foot_tie_break_period_s": 3.0,
      "risk_activation": 0.12,
      "recontact_margin": 0.04,
      "need_scale": 0.08,
      "dynamic_need_weight": 0.35,
      "progress_power": 2.0,
      "complete_weight": 0.70,
      "progress_weight": 0.30,
      "min_air_time": 0.04,
      "max_recontact_time": 0.30,
    },
  )

  # --- Restrict stand_still to lower body only ---
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still,
    weight=-0.03,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)
    },
  )
  cfg.rewards["no_foot_contact"] = RewardTermCfg(
    func=mdp.no_foot_contact_penalty,
    weight=-5.0,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["low_risk_foot_motion"] = RewardTermCfg(
    func=mdp.low_risk_foot_motion_penalty,
    weight=-9.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "max_idle_risk": 0.32,
      "airborne_velocity_weight": 1.0,
      "airborne_height_weight": 12.0,
      "airborne_contact_weight": 0.08,
      "fresh_takeoff_weight": 0.55,
      "fresh_takeoff_window_s": 0.06,
      "height_deadband": 0.025,
      "return_displacement_deadband": 0.50,
      "return_displacement_width": 0.60,
      "return_motion_relief": 0.85,
      "motion_need_threshold": 0.55,
      "motion_need_idle_weight": 0.95,
      **RECOVERY_FOOT_SELECTION_PARAMS,
    },
  )
  cfg.rewards["foot_takeoff_symmetry"] = RewardTermCfg(
    func=mdp.foot_takeoff_symmetry_penalty,
    weight=-3.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "airborne_weight": 0.25,
      "takeoff_weight": 1.0,
      "fresh_takeoff_window_s": 0.06,
      "imbalance_deadband": 0.02,
      "imbalance_scale": 0.08,
    },
  )
  cfg.rewards["directional_swing_foot_choice"] = RewardTermCfg(
    func=mdp.directional_swing_foot_choice_penalty,
    weight=-6.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "airborne_weight": 0.25,
      "takeoff_weight": 1.0,
      "fresh_takeoff_window_s": 0.06,
      "overused_pressure_weight": 4.0,
      "imbalance_deadband": 0.02,
      "imbalance_scale": 0.08,
      **BALANCED_FOOT_CHOICE_PARAMS,
      **RECOVERY_FOOT_SELECTION_PARAMS,
    },
  )
  cfg.rewards["underused_recovery_foot"] = RewardTermCfg(
    func=mdp.underused_recovery_foot_bonus,
    weight=16.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "airborne_weight": 0.15,
      "takeoff_weight": 1.0,
      "fresh_takeoff_window_s": 0.06,
      "imbalance_deadband": 0.02,
      "imbalance_scale": 0.08,
      **BALANCED_FOOT_CHOICE_PARAMS,
      **RECOVERY_FOOT_SELECTION_PARAMS,
    },
  )
  cfg.rewards["bad_body_ground_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 30.0},
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
    weight=-0.90,
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
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.rewards["body_ang_vel"].weight = -0.15

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(80.0)

  # Preserve sim-to-real randomization from the base G1 task. External pushes
  # are finite planar force pulses staged by curriculum above.
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].params["force_magnitude_range"] = (0.0, 0.0)
    cfg.events["push_robot"].params["force_z_range"] = (0.0, 0.0)
    cfg.events["push_robot"].params["torque_range"] = (0.0, 0.0)
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

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

  # Upper-body self-contact is not lower-body controllable. Keep the explicit
  # floor-contact safety term and avoid feeding PPO reward noise from the arms.
  cfg.rewards.pop("self_collisions", None)

  if not play and "upper_body_disturbance" in cfg.curriculum:
    stages = cfg.curriculum["upper_body_disturbance"].params["stages"]
    for stage in stages:
      if stage["step"] == 2400 * 48:
        stage.update(
          {
            "mode_probabilities": (0.60, 0.32, 0.08, 0.0),
            "resampling_time_range": (2.8, 5.5),
            "random_walk_velocity_range": (0.06, 0.35),
            "random_walk_acceleration_range": (0.15, 0.90),
            "sinusoid_frequency_range": (0.25, 1.00),
          }
        )
      if stage["step"] == 5200 * 48:
        stage.update(
          {
            "mode_probabilities": (0.50, 0.34, 0.14, 0.02),
            "random_walk_velocity_range": (0.08, 0.48),
            "random_walk_acceleration_range": (0.20, 1.20),
            "sinusoid_frequency_range": (0.30, 1.25),
            "pulse_duration_range": (0.70, 1.60),
          }
        )
      if stage["step"] == 8000 * 48:
        stage.update(
          {
            "step": 11000 * 48,
            "mode_probabilities": (0.32, 0.36, 0.22, 0.10),
            "amplitude_scale": 0.80,
            "rel_default_envs": 0.05,
            "random_walk_velocity_range": (0.09, 0.55),
            "random_walk_acceleration_range": (0.22, 1.30),
            "sinusoid_frequency_range": (0.32, 1.30),
            "pulse_duration_range": (0.75, 1.70),
          }
        )
    if not any(stage["step"] == 6500 * 48 for stage in stages):
      stages.append(
        {
          "step": 6500 * 48,
          "mode_probabilities": (0.42, 0.36, 0.18, 0.04),
          "amplitude_scale": 0.75,
          "rel_default_envs": 0.06,
          "random_walk_velocity_range": (0.09, 0.52),
          "random_walk_acceleration_range": (0.22, 1.30),
          "sinusoid_frequency_range": (0.32, 1.30),
          "pulse_duration_range": (0.70, 1.60),
        }
      )
    if not any(stage["step"] == 9000 * 48 for stage in stages):
      stages.append(
        {
          "step": 9000 * 48,
          "mode_probabilities": (0.38, 0.36, 0.20, 0.06),
          "amplitude_scale": 0.78,
          "rel_default_envs": 0.05,
          "random_walk_velocity_range": (0.09, 0.55),
          "random_walk_acceleration_range": (0.22, 1.35),
          "sinusoid_frequency_range": (0.33, 1.35),
          "pulse_duration_range": (0.70, 1.65),
        }
      )
    stages.sort(key=lambda stage: stage["step"])

  return cfg
