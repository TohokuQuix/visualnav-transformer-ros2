"""Read selected MCAP bags without playback or robot communication."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

from visualnav_transformer.eat_nav.portable_input_v2 import extract_track_2s, make_fixture


JOINTS = ("chassis/fr_sprocket_axle", "chassis/rr_sprocket_axle",
          "chassis/fl_sprocket_axle", "chassis/rl_sprocket_axle")
FLIPPERS = ("chassis/rr_flipper_joint", "chassis/rl_flipper_joint",
            "chassis/fr_flipper_joint", "chassis/fl_flipper_joint")
ARMS = ("arm/grasp_left", "arm/grasp_right", "arm/joint1", "arm/joint2",
        "arm/joint3", "arm/joint4", "arm/joint5", "arm/joint6")
TOPICS = ("/joint_states", "/bno055/imu", "/livox/imu_172_16_94_141",
          "/tf", "/tf_static", "/teaching_playback/path", "/diff_drive_controller/cmd_vel")


def stamp_ns(header):
    return int(header.stamp.sec) * 10**9 + int(header.stamp.nanosec)


def timing(stamps):
    stamps = np.asarray(stamps, dtype=np.int64)
    if len(stamps) < 2:
        return {"count": int(len(stamps)), "median_interval_ms": None, "p95_interval_ms": None, "max_interval_ms": None, "backward_or_zero_count": 0}
    diff = np.diff(stamps) / 1e6
    return {"count": int(len(stamps)), "median_interval_ms": float(np.median(diff)),
            "p95_interval_ms": float(np.percentile(diff, 95)), "max_interval_ms": float(np.max(diff)),
            "backward_or_zero_count": int(np.sum(diff <= 0))}


def finite_count(values):
    try:
        return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def one_bag(uri):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(uri), storage_id="mcap"), ConverterOptions("", ""))
    all_topics = {x.name: x.type for x in reader.get_all_topics_and_types()}
    selected = [name for name in TOPICS if name in all_topics]
    reader.set_filter(StorageFilter(topics=selected))
    message_types = {name: get_message(all_topics[name]) for name in selected}
    by_topic = Counter()
    header_times = {name: [] for name in selected}
    bag_times = {name: [] for name in selected}
    joints, imus, tf_times, paths, commands = [], [], [], [], []
    while reader.has_next():
        topic, raw, bag_ns = reader.read_next()
        msg = deserialize_message(raw, message_types[topic])
        by_topic[topic] += 1
        bag_times[topic].append(int(bag_ns))
        if topic == "/joint_states":
            names = list(msg.name)
            joints.append({"bag_ns": int(bag_ns), "header_ns": stamp_ns(msg.header),
                           "names": names, "position": list(msg.position), "velocity": list(msg.velocity),
                           "finite": finite_count(msg.position) and finite_count(msg.velocity),
                           "lengths_valid": len(names) == len(msg.position) == len(msg.velocity)})
            header_times[topic].append(stamp_ns(msg.header))
        elif topic in ("/bno055/imu", "/livox/imu_172_16_94_141"):
            imus.append({"topic": topic, "bag_ns": int(bag_ns), "header_ns": stamp_ns(msg.header),
                         "frame": msg.header.frame_id, "orientation": [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                         "angular": [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                         "orientation_cov0": float(msg.orientation_covariance[0]),
                         "angular_cov0": float(msg.angular_velocity_covariance[0]),
                         "finite": finite_count([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
                                                  msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])})
            header_times[topic].append(stamp_ns(msg.header))
        elif topic in ("/tf", "/tf_static"):
            for tf in msg.transforms:
                if tf.header.frame_id.lstrip("/") == "map" and tf.child_frame_id.lstrip("/") == "chassis/body":
                    tf_times.append({"topic": topic, "bag_ns": int(bag_ns), "header_ns": stamp_ns(tf.header)})
                    header_times[topic].append(stamp_ns(tf.header))
        elif topic == "/teaching_playback/path":
            paths.append({"bag_ns": int(bag_ns), "header_ns": stamp_ns(msg.header), "pose_count": len(msg.poses)})
            header_times[topic].append(stamp_ns(msg.header))
        elif topic == "/diff_drive_controller/cmd_vel":
            commands.append({"bag_ns": int(bag_ns), "vx": float(msg.twist.linear.x), "yaw": float(msg.twist.angular.z)})
            header_times[topic].append(stamp_ns(msg.header))
    primary_imus = [m for m in imus if m["topic"] == "/bno055/imu"]
    fallback_imus = [m for m in imus if m["topic"] == "/livox/imu_172_16_94_141"]
    primary_times = np.asarray([x["bag_ns"] for x in primary_imus], dtype=np.int64)
    tf_stamp_values = np.asarray([x["bag_ns"] for x in tf_times if x["topic"] == "/tf"], dtype=np.int64)
    presence = {name: sum(name in j["names"] for j in joints) / len(joints) if joints else 0. for name in JOINTS + FLIPPERS + ARMS}
    finite_joint = sum(j["finite"] and j["lengths_valid"] for j in joints)
    body_examples = []
    synchronized = 0
    imu_age_ms = []
    tf_near = 0
    for i, joint in enumerate(joints):
        stamp = joint["header_ns"] or joint["bag_ns"]
        joint_header_stamp = joint["header_ns"] or None
        imu = None
        if len(primary_times):
            ix = int(np.clip(np.searchsorted(primary_times, joint["bag_ns"]), 0, len(primary_times)-1))
            candidates = [ix] + ([ix-1] if ix > 0 else [])
            k = min(candidates, key=lambda z: abs(primary_times[z]-joint["bag_ns"]))
            age = abs(int(primary_times[k])-joint["bag_ns"]) / 1e6
            imu_age_ms.append(age)
            if age <= 50.:
                synchronized += 1
                m = primary_imus[k]
                imu = {"angular_velocity": m["angular"], "orientation_xyzw": m["orientation"],
                       "stamp_ns": m["header_ns"] or None, "frame_id": m["frame"], "orientation_covariance_0": m["orientation_cov0"],
                       "angular_velocity_covariance_0": m["angular_cov0"], "extrinsic_verified": False}
        if len(tf_stamp_values):
            ix = int(np.clip(np.searchsorted(tf_stamp_values, joint["bag_ns"]), 0, len(tf_stamp_values)-1))
            if min(abs(int(tf_stamp_values[ix])-joint["bag_ns"]), abs(int(tf_stamp_values[ix-1])-joint["bag_ns"]) if ix else 10**20) <= 50_000_000:
                tf_near += 1
        # No candidate Path, DEM, goal or arm-stowed calibration is manufactured.
        fixture = make_fixture("physical", {"joint_names": joint["names"], "joint_velocity": joint["velocity"],
                                           "joint_position": joint["position"], "joint_stamp_ns": joint_header_stamp,
                                           "anchor_stamp_ns": stamp, "imu": imu, "common": {}}, max_age_ns=250_000_000)
        if i < 5 or i == len(joints)-1:
            body_examples.append({"joint_index": i, "bag_timestamp_ns": joint["bag_ns"], "header_timestamp_ns": joint["header_ns"],
                                  "two_sprocket": fixture["fields"]["track_sprocket_velocity"],
                                  "flipper": fixture["fields"]["flipper_position"],
                                  "angular": fixture["fields"]["body_angular_velocity"],
                                  "gravity": fixture["fields"]["projected_gravity"],
                                  "variant_status": {k:v["status"] for k,v in fixture["variants"].items()}})
    track_values = [extract_track_2s(j["names"], j["velocity"], "physical") for j in joints
                    if j["finite"] and j["lengths_valid"] and all(n in j["names"] for n in JOINTS)]
    track_array = np.asarray(track_values, dtype=np.float64).reshape((-1, 2))
    track_distribution = {side: {"min": float(np.min(track_array[:, i])), "p05": float(np.percentile(track_array[:, i], 5)),
                                  "median": float(np.median(track_array[:, i])), "p95": float(np.percentile(track_array[:, i], 95)),
                                  "max": float(np.max(track_array[:, i]))} for i, side in enumerate(("right", "left"))} if len(track_array) else None
    flipper_distribution = {}
    for name in FLIPPERS:
        values = [j["position"][j["names"].index(name)] for j in joints if j["lengths_valid"] and name in j["names"]]
        flipper_distribution[name] = {"samples": len(values), "raw_min_rad": min(values) if values else None,
                                      "raw_max_rad": max(values) if values else None}
    arm_ranges = {}
    for name in ARMS:
        values = [j["position"][j["names"].index(name)] for j in joints if j["lengths_valid"] and name in j["names"]]
        arm_ranges[name] = {"samples": len(values), "min_rad": min(values) if values else None,
                            "max_rad": max(values) if values else None}
    commanded_motion = {"stationary": 0, "straight": 0, "turning": 0, "other": 0}
    for c in commands:
        if abs(c["vx"]) < .05 and abs(c["yaw"]) < .05: mode = "stationary"
        elif abs(c["yaw"]) >= .15: mode = "turning"
        elif abs(c["vx"]) >= .05: mode = "straight"
        else: mode = "other"
        commanded_motion[mode] += 1
    return {"uri": str(uri), "read_only": True, "topic_count": len(all_topics), "topics": all_topics,
            "filtered_message_counts": dict(by_topic),
            "timing_by_topic": {t:{"bag":timing(bag_times[t]), "header":timing(header_times[t]),
                                  "zero_header_count":sum(x==0 for x in header_times[t])} for t in selected},
            "joint_sample_count": len(joints), "joint_finite_shape_valid_fraction":finite_joint/len(joints) if joints else None,
            "joint_presence_fraction": presence,
            "primary_imu_frame_counts": dict(Counter(x["frame"] for x in primary_imus)),
            "primary_imu_covariance_sentinel_counts": {"orientation_minus_one":sum(x["orientation_cov0"]==-1 for x in primary_imus),
                                                       "angular_minus_one":sum(x["angular_cov0"]==-1 for x in primary_imus)},
            "primary_imu_finite_fraction":sum(x["finite"] for x in primary_imus)/len(primary_imus) if primary_imus else None,
            "fallback_imu_frame_counts":dict(Counter(x["frame"] for x in fallback_imus)),
            "fallback_imu_covariance_sentinel_counts":{"orientation_minus_one":sum(x["orientation_cov0"]==-1 for x in fallback_imus),
                                                       "angular_minus_one":sum(x["angular_cov0"]==-1 for x in fallback_imus)},
            "joint_header_to_bag_offset_ms_p95":float(np.percentile([abs(j["bag_ns"]-j["header_ns"])/1e6 for j in joints if j["header_ns"]],95)) if joints else None,
            "primary_imu_header_to_bag_offset_ms_p95":float(np.percentile([abs(j["bag_ns"]-j["header_ns"])/1e6 for j in primary_imus if j["header_ns"]],95)) if primary_imus else None,
            "joint_to_primary_imu_within_50ms_fraction":synchronized/len(joints) if joints else None,
            "joint_to_primary_imu_bag_time_nearest_p95_ms":float(np.percentile(imu_age_ms,95)) if imu_age_ms else None,
            "joint_to_map_body_tf_within_50ms_fraction":tf_near/len(joints) if joints else None,
            "map_body_tf_count":len(tf_times), "path_topic_is_candidate_path":False,
            "teaching_path_message_count":len(paths), "teaching_path_pose_count_range": [min(x["pose_count"] for x in paths),max(x["pose_count"] for x in paths)] if paths else None,
            "candidate_path_available":False,"dem_available":False,"relative_goal_available":False,
            "arm_joint_position_ranges":arm_ranges,"arm_stowed_status":"unverified; no prospective reference/tolerance",
            "flipper_raw_position_ranges":flipper_distribution,"track_scaled_distribution":track_distribution,
            "commanded_motion_counts_not_measured_motion":commanded_motion,
            "body_fixture_examples":body_examples,"full_core_availability_fraction":0.,"full_core_2s_availability_fraction":0.,
            "sprocket_sample_examples":track_values[:5]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("bags", nargs="+", type=Path)
    args = parser.parse_args()
    result = {"status": "read_only_body_state_source_parity_only", "bags": [one_bag(p) for p in args.bags],
              "sync_note": "50 ms nearest bag-time diagnostic does not certify header-stamp synchronized observation",
              "candidate_path_note": "/teaching_playback/path is not an Evaluator candidate Path",
              "missing_rule": "No candidate anchor, DEM, relative goal, calibrated flipper zero/sign or arm-stowed tolerance is inferred"}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps([{"uri":b["uri"],"joint_count":b["joint_sample_count"],"primary_imu_count":b["filtered_message_counts"].get('/bno055/imu',0)} for b in result["bags"]]))


if __name__ == "__main__":
    main()
