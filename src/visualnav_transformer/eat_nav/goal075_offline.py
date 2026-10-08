"""Pure offline geometry and contract checks for the 0.75 m goal review."""

import math

import numpy as np


BEARINGS_DEG = (-20, -10, 0, 10, 20)
GOAL_DISTANCE_M = .75
FOLLOWER_SCREEN_PER_M = 1.236
INITIAL_TANGENT_TOLERANCE_RAD = 1e-6


def goal075(bearing_deg):
    if bearing_deg not in BEARINGS_DEG:
        raise ValueError('bearing outside frozen contract')
    angle=math.radians(bearing_deg)
    return np.array([GOAL_DISTANCE_M*math.cos(angle),GOAL_DISTANCE_M*math.sin(angle)])


def contract075():
    return {'schema':'eatnav_common_goal075_offline_v1',
            'goal_distance_m':GOAL_DISTANCE_M,'bearings_deg':list(BEARINGS_DEG),
            'future_path_points':8,'path_dimensions':2,'path_frame':'robot-local-base',
            'path_endpoint':'common_global_goal','initial_point_xy_m':[0.,0.],
            'initial_tangent_xy':[1.,0.],
            'initial_tangent_tolerance_rad':INITIAL_TANGENT_TOLERANCE_RAD,
            'endpoint_tolerance_m':.001,'curvature_screen_per_m':FOLLOWER_SCREEN_PER_M,
            'reverse_estimate_allowed':False,'turn_in_place_estimate_allowed':False,
            'teacher_steps':40,'teacher_step_s':.05,'teacher_duration_s':2.,
            'primary_terrain_arc_end_m':.6,'diagnostic_path_arc_end_m':.75,
            'nominal_speed_mps':.3,'nominal_time_to_endpoint_s':2.5,
            'candidate_count_primary':5,'candidate_sides':['direct','left','left','right','right'],
            'source_anchor_count':8,'state_group_count':40,
            'separation_frame':'goal-ray-relative',
            'no_rollout':True}


def initial_forward_tangent_valid(tangent):
    vector=np.asarray(tangent,dtype=float)
    if vector.shape!=(2,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<=1e-12:
        return False
    angle=math.atan2(vector[1],vector[0])
    return abs(angle)<=INITIAL_TANGENT_TOLERANCE_RAD
