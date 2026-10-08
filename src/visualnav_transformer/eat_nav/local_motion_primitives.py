"""Pure robot-local primitive geometry and offline goal labels."""

from __future__ import annotations

import math

import numpy as np

from .curvature_attainable import eight_point_curvature
from .goal_bank_offline import path_sha

BEARINGS_DEG = (-20, -10, 0, 10, 20)
DISTANCE_M = .75
CURVATURE_SCREEN_PER_M = 1.236


def _xy_at_station(stations, curvature):
    s = np.asarray(stations, dtype=float)
    if abs(curvature) < 1e-14:
        return np.stack((s, np.zeros_like(s)), axis=-1)
    return np.stack((np.sin(curvature*s)/curvature,
                     (1-np.cos(curvature*s))/curvature), axis=-1)


def constant_curvature_primitive(distance_m: float, bearing_deg: int) -> dict:
    if bearing_deg not in BEARINGS_DEG or not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError('frozen positive chord and bearing required')
    theta = math.radians(bearing_deg)
    curvature = 2*math.sin(theta)/distance_m
    length = distance_m if bearing_deg == 0 else distance_m*theta/math.sin(theta)
    stations = np.linspace(0., length, math.ceil(length/.01)+1)
    dense = _xy_at_station(stations, curvature)
    waypoints = _xy_at_station(np.linspace(length/8, length, 8), curvature)
    target = np.array([distance_m*math.cos(theta), distance_m*math.sin(theta)])
    error = float(np.linalg.norm(waypoints[-1]-target))
    if error > 1e-9:
        raise ArithmeticError('arc endpoint reconstruction failed')
    return {'family':'constant_curvature_arc','variant_id':f'b{bearing_deg:+03d}',
            'bearing_deg':bearing_deg,'chord_distance_m':distance_m,
            'endpoint':target,'endpoint_error_m':error,
            'initial_tangent_xy':np.array([1.,0.]),
            'analytic_curvature_per_m':curvature,
            'analytic_max_curvature_per_m':abs(curvature),
            'dense_measured_max_curvature_per_m':eight_point_curvature(dense),
            'eight_point_max_curvature_per_m':eight_point_curvature(waypoints),
            'arc_length_m':length,'stations_m':stations,'dense':dense,
            'waypoints':waypoints,'path_sha256':path_sha(waypoints)}


def physical_and_label_ids(anchor_id: str, primitive_bearing_deg: int, splits: list[str]) -> list[dict]:
    if primitive_bearing_deg not in BEARINGS_DEG or len(splits) != 5 or len(set(splits)) != 1:
        raise ValueError('all five goal relabels must use one physical run and split')
    run_id = f'{anchor_id}:primitive:{primitive_bearing_deg:+03d}'
    return [{'physical_run_id':run_id,'label_id':f'{run_id}:goal:{goal:+03d}',
             'goal_bearing_deg':goal,'split':splits[0]}
            for goal in BEARINGS_DEG]


def goal_labels(trajectory_world_xy, anchor_world_xy_yaw, reward_steps,
                goal_bearing_deg: int, *, local_endpoint, gamma=.99) -> dict:
    if goal_bearing_deg not in BEARINGS_DEG:
        raise ValueError('goal bearing outside frozen contract')
    trajectory = np.asarray(trajectory_world_xy, dtype=float)
    pose = np.asarray(anchor_world_xy_yaw, dtype=float)
    endpoint = np.asarray(local_endpoint, dtype=float)
    rewards = np.asarray(reward_steps, dtype=float)
    if (trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) < 2
            or pose.shape != (3,) or endpoint.shape != (2,)
            or not np.isfinite(trajectory).all() or not np.isfinite(pose).all()
            or not np.isfinite(endpoint).all() or not np.isfinite(rewards).all()):
        raise ValueError('finite trajectory, pose, endpoint and rewards required')
    c,s=math.cos(pose[2]),math.sin(pose[2])
    rotation=np.array([[c,-s],[s,c]])
    local=(trajectory-pose[:2])@rotation
    theta=math.radians(goal_bearing_deg)
    goal=DISTANCE_M*np.array([math.cos(theta),math.sin(theta)])
    along=goal/DISTANCE_M
    lateral=np.array([-along[1],along[0]])
    distance=np.linalg.norm(local-goal,axis=1)
    return {'locomotion':{'discounted_reward_2s':float(np.dot(gamma**np.arange(len(rewards)),rewards)),
                          'reward_steps':rewards.tolist()},
            'navigation':{'global_goal_base_xy_m':goal.tolist(),
                          'goal_distance_reduction_2s':float(distance[0]-distance[-1]),
                          'final_distance_to_global_goal_m':float(distance[-1]),
                          'minimum_distance_to_global_goal_m':float(distance.min()),
                          'progress_along_goal_ray_m':float((local[-1]-local[0])@along),
                          'signed_lateral_deviation_m':float(local[-1]@lateral),
                          'candidate_endpoint_residual_m':float(np.linalg.norm(endpoint-goal))}}
