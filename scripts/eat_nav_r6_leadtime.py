"""Validation-only anchor warnings vs observed events/contact phase.

Stall is a2s interval outcome: its confirmation lead is not a measured physical
stall-onset lead. Contact is diagnostic, not automatically a failure label.
"""
import argparse,collections,json
from pathlib import Path
import numpy as np
import torch
from eat_nav_r6_run import read,write,sha
from visualnav_transformer.eat_nav.r6_model import load_checkpoint,predict
from visualnav_transformer.eat_nav.r6_policy import Footprint,select_candidate,select_from_features,wilson
from visualnav_transformer.eat_nav.r6_geometry import calibrated_footprint

HEADS=('tip_over','near_fall','linear_stall','turn_stall')


def phase_of(record):
 if record.get('pre_anchor_contact') is True:return 'after_observed_contact'
 if record.get('prefix_contact_complete') and record.get('pre_anchor_contact') is False and record.get('pre_anchor_contact_scope')=='settling_and_prefix_5ms':return 'before_observed_contact'
 return 'contact_phase_unknown'


def evaluate(dataset,checkpoint,records_file,profiles_file,calibration_file,output):
 output=Path(output)
 if output.exists():raise FileExistsError('fresh leadtime output required')
 with np.load(Path(dataset)/'samples.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
 index=read(Path(dataset)/'sample-index.json');records=read(records_file);profiles=read(profiles_file)['profiles'];cal=read(calibration_file)
 if any(r['split']!='validation' for r in records):raise ValueError('validation-only diagnostic')
 bysample=collections.defaultdict(list)
 for record in records:bysample[record['sample_id']].append(record)
 model,payload=load_checkpoint(checkpoint);output.mkdir(parents=True);details=[]
 for i,member in enumerate(index):
  if member['role']!='validation':continue
  pred=predict(model,payload['normalization'],a,[i]);flippers=dict(zip(('rr','rl','fr','fl'),map(float,a['actor_state'][i,10:14])))
  fp=Footprint(**calibrated_footprint(cal,flippers,a['body_state'][i,:3]))
  base=select_candidate(pred['trajectory'][0],pred['events'][0],a['dem'][i,-1],[2,0],profiles[-1],footprint=fp,candidate_mask=a['candidate_mask'][i])
  choices={p['id']:select_from_features(base['policy_features'],p) for p in profiles}
  for r in bysample[member['sample_id']]:
   k=r['candidate'];contact=r.get('contact_metadata',{});event_records=[];pose_warnings={}
   for h,name in enumerate(HEADS):
    if not r['event_mask'][h]:continue
    positive=bool(r['event_targets'][h]);alert=bool(pred['events'][0,k,h]>=.5)
    observed=r.get('observed_onsets',{});time_sec=observed.get('first_confirmation_sec',{}).get(name)
    distance=observed.get('path_distance_to_confirmation_m',{}).get(name)
    if name in ('linear_stall','turn_stall') and positive:time_sec=2.;distance=r.get('future_displacement_m')
    event_records.append(dict(head=name,positive=positive,alert=alert,probability=float(pred['events'][0,k,h]),confirmation_lead_sec=time_sec if positive and alert else None,distance_to_confirmation_m=distance if positive and alert else None,lead_definition='2s window confirmation, physical stall onset unavailable' if name.endswith('stall') else 'first20Hzfuture confirmation from current anchor',already_tilted60_at_anchor=bool(observed.get('initial_tilt_deg',0) is not None and observed.get('initial_tilt_deg',0)>=60)))
   for profile in profiles:
    why=choices[profile['id']]['reasons'][k]
    pose_warnings[profile['id']]=dict(risk_warning=any(w.startswith('DEM_step_') or w in ('predicted_pitch','predicted_roll') for w in why),map_unknown='map_unknown' in why,reasons=why)
   details.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],field=r['field'],level=r['level'],scene=r.get('scene','field'),candidate=k,phase=phase_of(r),events=event_records,first_observed_contact_sec=contact.get('first_observed_contact_sec'),contact_fully_observed=contact.get('fully_observed',False),pose_warnings=pose_warnings))
 groups=collections.defaultdict(list)
 for r in details:groups[(r['scene'],r['phase'])].append(r)
 summaries={}
 for (scene,phase),rows in groups.items():
  heads={}
  for name in HEADS:
   xs=[e for r in rows for e in r['events'] if e['head']==name];positives=[e for e in xs if e['positive']];detected=[e for e in positives if e['alert']];leads=[e['confirmation_lead_sec'] for e in detected if e['confirmation_lead_sec'] is not None];dist=[e['distance_to_confirmation_m'] for e in detected if e['distance_to_confirmation_m'] is not None]
   heads[name]=dict(known_candidates=len(xs),positive_candidates=len(positives),alerted_positive_candidates=len(detected),recall=len(detected)/len(positives) if positives else None,recall_wilson95=wilson(len(detected),len(positives)),mean_probability=float(np.mean([e['probability'] for e in xs])) if xs else None,detected_confirmation_lead_sec_quantiles=np.quantile(leads,[0,.25,.5,.75,1]).tolist() if leads else None,detected_distance_m_quantiles=np.quantile(dist,[0,.25,.5,.75,1]).tolist() if dist else None,warning_with_at_least_half_second_count=sum(x>=.5 for x in leads),already_tilted60_at_anchor_count=sum(e['already_tilted60_at_anchor'] for e in xs))
  summaries[f'{scene}/{phase}']=dict(candidates=len(rows),source_groups=len({r['group_id'] for r in rows}),heads=heads)
 result=dict(schema='eatnav_r6_anchor_leadtime_v1',split='validation',model_sha256=sha(checkpoint),dataset_sha256=sha(Path(dataset)/'samples.npz'),source_records_sha256=sha(records_file),candidate_records=len(details),summaries=summaries,limitations=['Correlated14branches/sharedparentgroups; descriptive CIs only','One prediction per independent anchor, not earliest detection on a continuous physical episode','Stall confirmation lead is not physical stall onset lead','Contact phase remains unknown if full pre-anchor instrumentation was unavailable','Unknown map exclusion is not counted as successful risk anticipation'])
 write(output/'details.json',details);write(output/'report.json',result);return result


def main():
 torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__)
 for key in ('dataset','checkpoint','records','profiles','calibration','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();r=evaluate(a.dataset,a.checkpoint,a.records,a.profiles,a.calibration,a.output);print(json.dumps(dict(candidate_records=r['candidate_records'],groups=list(r['summaries'])),indent=2))
if __name__=='__main__':main()
