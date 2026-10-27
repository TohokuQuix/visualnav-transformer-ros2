"""Original fourteen circular paths for training; six-path evaluation is separate."""
import numpy as np
from visualnav_transformer.eat_nav.bidirectional_bank import bank
from visualnav_transformer.eat_nav.motion_14_stage_a import candidate_role
from eat_nav_risk_navigation import pack_inputs as pack_four
from eat_nav_all_fields import rotation,compare_anchor,geometry_choice as legacy_choice

_BANK=bank()
CANDIDATE_IDS=tuple(candidate_role(c.direction,c.arc_angle_deg) for c in _BANK)
DIRECTIONS=np.asarray([c.direction for c in _BANK],dtype=np.float32)
ANGLES=tuple(np.deg2rad(c.arc_angle_deg) for c in _BANK)
OLD_SLOTS=(0,1,2,7)
EVAL_SLOTS=(0,1,2,7,8,9)
ADDED_SLOTS=tuple(i for i in range(14) if i not in OLD_SLOTS)

def candidate_paths():return np.stack([c.waypoints for c in _BANK]).astype(np.float32)

def geometry_choice(goal_body):return OLD_SLOTS[legacy_choice(goal_body)]

def pack_inputs(dem,actor_state,body):
    arrays=pack_four(dem,actor_state,body)
    arrays.update(paths=candidate_paths()[None],directions=DIRECTIONS.reshape(1,14,1).copy(),candidate_mask=np.ones((1,14),dtype=bool))
    return arrays

def extend_path(points,terminal_heading,index):
    points=np.asarray(points,dtype=float);tail=points[-1]+candidate_paths()[index].astype(float)@rotation(terminal_heading).T
    return np.concatenate((points,tail[1:])),terminal_heading+ANGLES[index]

def compare_initial(actual,reference):
    reduced=dict(actual)
    for k in ('paths','directions','candidate_mask'):
        if k not in actual or k not in reference:return ['candidate_keys']
        reduced[k]=actual[k][:,list(OLD_SLOTS)]
    errors=compare_anchor(reduced,reference)
    expected={'paths':candidate_paths()[None],'directions':DIRECTIONS.reshape(1,14,1),'candidate_mask':np.ones((1,14),dtype=bool)}
    for k,value in expected.items():
        a=np.asarray(actual[k])
        if a.shape!=value.shape or a.dtype!=value.dtype or not np.array_equal(a,value):errors.append('appended_bank_geometry' if k=='paths' else 'appended_'+k)
    return errors
