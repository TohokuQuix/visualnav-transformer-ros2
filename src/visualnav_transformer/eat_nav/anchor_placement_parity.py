"""Frozen numeric and identity gate for geometry-backed DEM reconstruction."""
from __future__ import annotations

import math

import numpy as np

PARITY_TOLERANCE = {
    'height_rms_m': 1e-5,
    'height_max_abs_m': 1e-4,
    'mask_iou': 1.0,
    'origin_max_abs_m': 1e-5,
    'yaw_abs_rad': 1e-5,
}

EXACT_KEYS = ('pose_id','env_id','terrain','terrain_seed','frame_id','axis',
              'resolution_m','shape','physics_step')


def _mask_iou(first, second):
    union = np.logical_or(first, second)
    return float(np.logical_and(first, second).sum() / union.sum()) if union.any() else 1.0


def _error(first, second, mask):
    if not mask.any():
        return None, None
    difference = first[mask].astype(np.float64)-second[mask].astype(np.float64)
    return float(np.sqrt(np.mean(difference*difference))), float(np.max(np.abs(difference)))


def compare_dem_frame(expected: dict, actual: dict, tolerance: dict = PARITY_TOLERANCE) -> dict:
    """Compare a saved frame to an independent RayCaster reconstruction."""
    mismatches=[key for key in EXACT_KEYS if expected.get(key)!=actual.get(key)]
    result={'identity_mismatches':mismatches,'tolerance':dict(tolerance)}
    try:
        e_src=np.asarray(expected['source_height'],dtype=np.float32)
        a_src=np.asarray(actual['source_height'],dtype=np.float32)
        e_smask=np.asarray(expected['source_mask'],dtype=bool)
        a_smask=np.asarray(actual['source_mask'],dtype=bool)
        e_dem=np.asarray(expected['transformed'],dtype=np.float32)
        a_dem=np.asarray(actual['transformed'],dtype=np.float32)
        if any(x.shape!=(20,20) for x in (e_src,a_src,e_smask,a_smask)) or e_dem.shape!=(2,20,20) or a_dem.shape!=(2,20,20):
            raise ValueError('grid shape mismatch')
        e_tmask=e_dem[1]>.5;a_tmask=a_dem[1]>.5
        finite=bool(np.isfinite(e_src[e_smask]).all() and np.isfinite(a_src[a_smask]).all() and
                    np.isfinite(e_dem[0][e_tmask]).all() and np.isfinite(a_dem[0][a_tmask]).all())
        src_rms,src_max=_error(e_src,a_src,e_smask&a_smask)
        dem_rms,dem_max=_error(e_dem[0],a_dem[0],e_tmask&a_tmask)
        origin=float(np.max(np.abs(np.asarray(expected['source_origin_xy'],dtype=float)-
                                   np.asarray(actual['source_origin_xy'],dtype=float))))
        env_origin=float(np.max(np.abs(np.asarray(expected['env_origin_xyz'],dtype=float)-
                                       np.asarray(actual['env_origin_xyz'],dtype=float))))
        yaw=abs(math.remainder(float(expected['yaw_rad'])-float(actual['yaw_rad']),2*math.pi))
        result.update({'source_height_rms_m':src_rms,'source_height_max_abs_m':src_max,
                       'transformed_height_rms_m':dem_rms,'transformed_height_max_abs_m':dem_max,
                       'source_mask_iou':_mask_iou(e_smask,a_smask),
                       'transformed_mask_iou':_mask_iou(e_tmask,a_tmask),
                       'source_unknown_fraction_expected':float(1-e_smask.mean()),
                       'source_unknown_fraction_actual':float(1-a_smask.mean()),
                       'transformed_unknown_fraction_expected':float(1-e_tmask.mean()),
                       'transformed_unknown_fraction_actual':float(1-a_tmask.mean()),
                       'source_origin_max_abs_m':origin,'env_origin_max_abs_m':env_origin,
                       'yaw_abs_rad':yaw,'finite':finite})
        result['pass']=(not mismatches and finite and src_rms is not None and dem_rms is not None and
            src_rms<=tolerance['height_rms_m'] and dem_rms<=tolerance['height_rms_m'] and
            src_max<=tolerance['height_max_abs_m'] and dem_max<=tolerance['height_max_abs_m'] and
            result['source_mask_iou']==tolerance['mask_iou'] and
            result['transformed_mask_iou']==tolerance['mask_iou'] and
            origin<=tolerance['origin_max_abs_m'] and env_origin<=tolerance['origin_max_abs_m'] and
            yaw<=tolerance['yaw_abs_rad'])
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        result.update({'pass':False,'error':repr(exc)})
    return result
