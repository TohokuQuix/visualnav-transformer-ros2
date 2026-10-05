"""Generate a field-explicit future logging contract; does not run the simulator."""
from __future__ import annotations

import json
from pathlib import Path

from eat_nav_evaluator_multihead_addendum import validate_logging_schema

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-addendum"
FIELDS = []


def field(name, shape, unit, frame, phase, sim, robot, *, order="scalar", norm="raw_no_normalization",
          clip="none", mask=None, parity="pending_physical_parity"):
    FIELDS.append({"name": name, "shape": shape, "order": order, "unit": unit, "frame": frame,
                   "timestamp_phase": phase, "normalization": norm, "clip": clip,
                   "missing_mask": mask or f"{name}_valid", "simulator_api": sim,
                   "robot_api": robot, "parity_status": parity, "role": "future_logging_only"})


def build():
    FIELDS.clear()
    R="post_step"; A="anchor_pre_path"; P="pre_command_update"; C="post_command_update_pre_step"; T="terminal_pre_reset"
    field("reward_total", "[]", "RL reward including dt", "environment", R, "RewardManager total", "not present; future policy logging", parity="sim_only")
    field("reward_component_pre_dt", "map<reward_term,float>", "term-specific before dt", "environment", R, "RewardManager raw term output", "not present; future policy logging", order="frozen_reward_term_names", mask="reward_component_valid")
    field("reward_component_post_dt", "map<reward_term,float>", "term-specific after one dt", "environment", R, "RewardManager weighted step output", "not present; future policy logging", order="frozen_reward_term_names", mask="reward_component_valid")
    field("reward_component_weight", "map<reward_term,float>", "config multiplier", "environment", R, "RewardManager term config", "not present; future policy metadata", order="frozen_reward_term_names")
    field("control_dt_s", "[]", "s", "simulation_clock", R, "env.step_dt", "controller period /clock", parity="source_equivalence_pending")
    field("reward_config_sha256", "[64]", "hex SHA256", "config", A, "frozen reward config file", "future deployed reward config if available", order="hex_byte_order")
    field("reward_term_version", "map<reward_term,string>", "source revision", "config", A, "RewardManager term source revisions", "future deployed reward source metadata", order="frozen_reward_term_names")
    field("termination_reward", "[]", "RL reward including dt", "environment", T, "RewardManager termination term", "not present; future policy logging", parity="sim_only")
    field("reward_component_valid", "map<reward_term,bool>", "boolean", "environment", R, "RewardManager term finite/available mask", "future policy reward mask", order="frozen_reward_term_names")

    field("path_active", "[]", "boolean", "follower", P, "bridge/follower status", "production follower status")
    field("path_id_sha256", "[64]", "hex SHA256", "Path base/map identity", A, "frozen candidate Path file SHA", "Path message canonical SHA", order="hex_byte_order")
    field("path_publish_stamp_ns", "[]", "ns", "ROS simulation/robot clock", A, "bridge Path publication", "ROS Path header and publication event")
    field("path_horizon_end", "[]", "boolean event", "follower", C, "bridge Path horizon event", "production follower Path end event")
    field("replanning_event", "[]", "boolean event", "planner/follower", C, "bridge replan event", "ROS planner/follower event")
    field("follower_mode", "[]", "enum", "follower", P, "bridge follower status", "production follower status", order="frozen_enum_table")
    field("command_current", "[3]", "m/s,m/s,rad/s", "robot base", C, "CommandManager current velocity command", "adopted /cmd_vel and bridge state", order="vx_vy_wz")
    field("command_last_nonzero", "[3]", "m/s,m/s,rad/s", "robot base", C, "CommandManager last nonzero command", "follower last adopted nonzero command", order="vx_vy_wz")
    field("command_source", "[]", "enum", "command manager", C, "CommandManager source/term", "follower/bridge command source", order="frozen_enum_table")
    field("command_publish_stamp_ns", "[]", "ns", "ROS simulation/robot clock", P, "bridge command publication", "ROS command publication")
    field("command_adopt_stamp_ns", "[]", "ns", "ROS simulation/robot clock", C, "CommandManager adoption event", "deployed command adoption event")
    field("pre_command_observation_stamp_ns", "[]", "ns", "logical simulation/robot clock", P, "ObservationManager pre-update capture", "future deployed pre-update observation logger")
    field("post_command_observation_stamp_ns", "[]", "ns", "logical simulation/robot clock", C, "ObservationManager Actor/Critic input capture", "future deployed post-update observation logger")
    field("exact_stamp_tf", "[7]", "m,unit quaternion", "map to base", C, "bridge exact-stamp TF", "tf2 exact-stamp map/base", order="xyz_qw_qx_qy_qz")
    field("post_path_control_state", "[3]", "hold,stop,replan flags", "follower", C, "bridge controller state", "production follower state", order="hold_stop_replan")

    field("projected_gravity", "[3]", "unit vector", "base", C, "Articulation projected gravity / Critic term", "IMU orientation plus TF", order="x_y_z", clip="[-1,1]")
    field("base_linear_velocity", "[3]", "m/s", "base", C, "Articulation base velocity / Critic term", "odometry plus TF; not in current bridge", order="vx_vy_vz")
    field("base_angular_velocity", "[3]", "rad/s", "base", C, "Articulation base angular velocity / Critic term", "IMU angular velocity plus TF", order="wx_wy_wz")
    field("full_joint_position", "[52]", "rad", "ordered joints", C, "Articulation joint_pos", "/joint_states; full order currently unverified", order="frozen_52_RL_joint_names")
    field("full_joint_velocity", "[52]", "rad/s", "ordered joints", C, "Articulation joint_vel", "/joint_states; full order currently unverified", order="frozen_52_RL_joint_names")
    field("previous_action", "[6]", "normalized RL action", "ActionManager order", C, "ActionManager previous action", "deployed Actor action API; currently unavailable", order="frozen_6_ActionManager_names")
    field("actor_action", "[6]", "normalized RL action", "ActionManager order", C, "Actor output actually sent to ActionManager", "deployed Actor action output capture", order="frozen_6_ActionManager_names")
    field("actor_command_input", "[3]", "m/s,m/s,rad/s", "base", C, "Actor observation velocity command term", "deployed Actor command input capture", order="vx_vy_wz")
    field("actor_observation", "[N_actor]", "mixed by field", "Actor term frames", C, "post-delay Actor input actually consumed", "deployed Actor input capture", order="frozen_Actor_observation_schema")
    field("critic_observation", "[267]", "mixed by field", "Critic term frames", C, "ObservationManager critic group", "offline-only on deployed robot unless critic runs", order="frozen_Critic_schema_indices_9_to_12_command", parity="sim_only_until_deployed_Critic_parity")
    field("controller_state", "map<string,value>", "mixed", "follower/controller", C, "bridge and follower state", "production follower status", order="frozen_controller_state_keys")
    field("raw_applied_effort", "[52]", "N m or actuator unit per joint", "ordered joints", R, "Articulation raw applied effort", "motor telemetry if available; not primary model input", order="frozen_52_RL_joint_names", parity="not_primary_input")
    field("effort_limit", "[52]", "matching raw effort unit", "ordered joints", R, "actuator config effective limit", "motor rated/command limit metadata", order="frozen_52_RL_joint_names", parity="not_primary_input")
    field("contact", "[N_contact,3]", "N", "contact sensor link order", R, "ContactSensor force vector", "physical contact sensor if available", order="frozen_contact_link_names", parity="sim_only_until_sensor_mapping")
    field("roll_pitch_tilt", "[3]", "rad", "base relative gravity", R, "root quaternion derived roll/pitch and max tilt", "IMU/TF derived roll/pitch and max tilt", order="roll_pitch_max_abs")
    field("root_pose", "[7]", "m,unit quaternion", "map/world", C, "Articulation root pose", "odom/TF map-to-base", order="xyz_qw_qx_qy_qz")
    field("realized_xy", "[2]", "m", "map", R, "post-step root XY corrected map frame", "tf2 map-to-base post-step XY", order="x_y")
    field("dem_height_mask_history", "[4,2,20,20]", "m,binary", "base-yaw raster", C, "transformed height scanner DEM and valid mask", "GridMap adapter DEM and valid mask", order="time_height_mask_row_column", norm="scalar_height_offset_only_then_train_only_valid_cell_stats")
    field("relative_goal", "[2]", "m", "base", A, "frozen anchor goal", "ROS goal plus exact-stamp TF", order="x_y")
    field("candidate_path", "[8,2]", "m", "base", A, "frozen candidate Path before publication", "candidate generator Path before follower", order="waypoint_x_y")

    for name, unit, phase, sim, robot in (
        ("termination_reason", "enum", T, "env termination manager", "deployed safety/controller event"),
        ("truncation", "boolean", T, "time-limit truncated flag", "deployed episode boundary"),
        ("timeout", "boolean", T, "time-limit event", "deployed timeout event"),
        ("reset", "boolean", T, "environment reset marker", "deployed reset marker"),
        ("stack_proxy", "boolean", T, "offline proxy from motion/command", "prospective monitoring only"),
        ("verified_stack", "boolean", T, "verified physical/trace adjudication", "operator or validated detector"),
        ("excessive_tilt", "boolean", T, "prospective verified tilt threshold source", "validated robot safety limit"),
        ("infrastructure_failure", "boolean", T, "runner/bridge failure marker", "ROS/system health event"),
        ("manual_stop", "boolean", T, "operator stop marker", "operator stop event"),
        ("censored", "boolean", T, "measurement censor mask", "data collection censor mask"),
        ("measurement_valid", "boolean", T, "protocol validity marker", "measurement validity marker"),
        ("failure_positive_eligible", "boolean", T, "verified label policy", "verified label policy"),
    ):
        field(name, "[]", unit, "outcome validity", phase, sim, robot,
              parity="policy_contract_pending", mask=f"{name}_recorded")
    schema = {
        "schema_version": "future_evaluator_collection_v1",
        "status": "proposed_logging_only_no_implementation_or_rollout",
        "step_timing": {
            A: "frozen pre-Path anchor; DEM/goal/Path and source state before candidate publication",
            P: "step k before CommandManager command update and before Actor/Critic input regeneration",
            C: "same logical step k after command adoption; Actor/Critic inputs actually used before env.step",
            R: "transition k complete; reward includes exactly one dt and post-step body/contact/XY",
            T: "terminal boundary before reset; preserve current episode ID and terminal Critic inputs",
        },
        "join_keys": ["source_trajectory", "environment_instance", "terrain_seed", "source_episode",
                      "StateGroup", "candidate_Path_SHA", "repeat", "logical_step", "logical_stamp_ns"],
        "fields": FIELDS,
        "safety_label_rule": "failure_positive_eligible only for independently verified physical failure; infrastructure, invalid measurement, timeout, manual stop, censoring excluded",
        "primary_input_separation": "this schema logs future outcomes; primary model input is frozen separately and only anchor-pre-Path fields may enter it",
        "reward_component_warning": "historical 63 traces contain scalar reward only; no component values reconstructed here",
    }
    validate_logging_schema(schema)
    return schema


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "future-collection-logging-schema.json"
    path.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(FIELDS)} future logging fields")
