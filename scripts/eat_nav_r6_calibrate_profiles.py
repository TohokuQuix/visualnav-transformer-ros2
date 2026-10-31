"""Freeze >=10 validation-derived candidates and a validation-only pose margin."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from eat_nav_r6_run import read,write,sha
from visualnav_transformer.eat_nav.r6_model import load_checkpoint,predict
from visualnav_transformer.eat_nav.r6_policy import derive_profiles


def pose_underprediction(predicted,actual,mask,candidate_mask):
 predicted=np.asarray(predicted);actual=np.asarray(actual);mask=np.asarray(mask,bool);candidate_mask=np.asarray(candidate_mask,bool)
 if predicted.shape!=actual.shape or actual.shape[:-1]!=mask.shape or actual.shape[-1]!=9 or candidate_mask.shape!=mask.shape[:-1]:raise ValueError('matched pose/mask contract required')
 admitted=mask.any(axis=2)&candidate_mask;errors=[]
 for sin_i,cos_i in ((5,6),(7,8)):
  p=np.max(np.where(mask,np.abs(np.degrees(np.arctan2(predicted[...,sin_i],predicted[...,cos_i]))),-np.inf),axis=2)
  a=np.max(np.where(mask,np.abs(np.degrees(np.arctan2(actual[...,sin_i],actual[...,cos_i]))),-np.inf),axis=2)
  errors.append(a[admitted]-p[admitted])
 return np.maximum(0,np.maximum(*errors))


def calibrated_margin(profile,errors):
 q=(float(profile['quantiles']['pitch'])+float(profile['quantiles']['roll']))/2
 if not .5<=q<=.95 or len(errors)<30:raise ValueError('predeclared empirical margin grid/support required')
 return dict(quantile=q,margin_deg=float(np.quantile(errors,q)),rule='mean of pitch/roll empirical quantiles, fixed before closed-loop trials')


def calibrate(dataset,checkpoint,records_file,output):
 dataset=Path(dataset);output=Path(output)
 if output.exists():raise FileExistsError('fresh calibrated profiles required')
 records=read(records_file);result=derive_profiles(records)
 if result['status']!='derived':raise ValueError('insufficient validation support for profiles')
 with np.load(dataset/'samples.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
 ids=read(dataset/'split.json')['roles']['validation'];model,payload=load_checkpoint(checkpoint)
 predicted=np.concatenate([predict(model,payload['normalization'],arrays,ids[i:i+32])['trajectory'] for i in range(0,len(ids),32)])
 actual=arrays['trajectory'][ids];mask=arrays['trajectory_mask'][ids].astype(bool);complete=mask.all(axis=2)&arrays['candidate_mask'][ids].astype(bool)
 aligned=pose_underprediction(predicted,actual,mask,arrays['candidate_mask'][ids])
 admitted=mask.any(axis=2)&arrays['candidate_mask'][ids].astype(bool)
 aligned_map={tuple(k):float(e) for k,e in zip(np.argwhere(admitted),aligned)}
 members=read(dataset/'sample-index.json');lookup={members[original]['sample_id']:j for j,original in enumerate(ids)}
 peak_roll=np.max(np.abs(np.degrees(np.arctan2(predicted[...,5],predicted[...,6]))),axis=2)
 peak_pitch=np.max(np.abs(np.degrees(np.arctan2(predicted[...,7],predicted[...,8]))),axis=2)
 combined=[];raw20hz=0
 for record in records:
  j=lookup[record['sample_id']];k=record['candidate']
  error=max(0.,record['max_abs_roll_deg']-float(peak_roll[j,k]),record['max_abs_pitch_deg']-float(peak_pitch[j,k]),aligned_map.get((j,k),0.))
  combined.append(error);raw20hz+=record.get('source_observation_hz')==20
 underestimation=np.asarray(combined)
 if len(underestimation)<30:raise ValueError('insufficient complete validation pose errors')
 position_error=np.linalg.norm(predicted[...,:2]-actual[...,:2],axis=-1)
 position_peak=np.max(np.where(mask,position_error,0.),axis=2)[mask.any(axis=2)&arrays['candidate_mask'][ids].astype(bool)]
 position_q95=float(np.quantile(position_peak,.95))
 margin=float(np.quantile(underestimation,.95));calibration=dict(source='validation_only',method='95th quantile of max(aligned-grid underestimation, observed20Hzpeak minus full2sforecast peak); preterminal failures included',candidate_count=len(underestimation),raw20hz_records=raw20hz,full_horizon_candidates=int(complete.sum()),partial_observed_candidates=int(len(underestimation)-complete.sum()),margin_deg=margin,quantiles={str(q):float(np.quantile(underestimation,q)) for q in (0,.5,.9,.95,1)},correlation_note='fourteen candidates share source anchors; descriptive coverage, not independent safety guarantee')
 result.update(position_error_diagnostic=dict(source='validation_only',horizontal_peak_error_q95_m=position_q95,sample_count=len(position_peak),scope='bag clearance diagnostic; not silently added as a planner stopping margin'),model_sha256=sha(checkpoint),dataset_sha256=sha(dataset/'samples.npz'),records_file_sha256=sha(records_file),pose_margin_calibration=calibration,trial_execution_status='candidates only; no completed closed-loop comparison implied')
 result['pose_margin_calibration']['scope']='q95 reference; applied margin is profile-specific empirical quantile'
 result['margin_grid_rule']='For each profile use validation peak-error quantile equal to mean(pitch_quantile,roll_quantile); q95 retained as diagnostic. Predeclared before any production closed-loop outcomes.'
 for profile in result['profiles']:
  fitted=calibrated_margin(profile,underestimation)
  profile.update(position_error_q95_m=position_q95,position_error_scope='diagnostic for clearance interpretation; control footprint unchanged',tilt_margin_deg=fitted['margin_deg'],uncertainty_margin_quantile=fitted['quantile'],uncertainty_margin_status='validation_peak_underestimation_profile_quantile',model_sha256=result['model_sha256'],dataset_sha256=result['dataset_sha256'])
 result['universally_STOP_profiles_by_angular_bounds']=sum(min(p['pitch_deg'],p['roll_deg'])<=p['tilt_margin_deg'] for p in result['profiles'])
 output.mkdir(parents=True)
 for profile in result['profiles']:write(output/(profile['id']+'.json'),profile)
 result['profile_files_sha256']={p.name:sha(p) for p in output.glob('pose_*.json')};write(output/'index.json',result);return result


def main():
 torch.set_num_threads(1)
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('dataset','checkpoint','records','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();r=calibrate(a.dataset,a.checkpoint,a.records,a.output);print(json.dumps(dict(status=r['status'],profiles=len(r['profiles']),pose_margin_calibration=r['pose_margin_calibration']),indent=2))
if __name__=='__main__':main()
