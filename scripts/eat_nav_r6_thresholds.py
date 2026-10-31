#!/usr/bin/env python3
"""Derive validation-only threshold candidates, or freeze validated top three.

This command neither reads a bag nor executes policies. It refuses overwrites.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from visualnav_transformer.eat_nav.r6_policy import derive_profiles,freeze_top_three


def read(path):
    text=Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()] if str(path).endswith('.jsonl') else json.loads(text)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    derive=sub.add_parser('derive',help='JSON/JSONL records of observed validation candidate outcomes')
    derive.add_argument('--records',required=True,type=Path)
    derive.add_argument('--output',required=True,type=Path)
    derive.add_argument('--min-samples',type=int,default=30)
    derive.add_argument('--min-success',type=int,default=10)
    derive.add_argument('--min-failure',type=int,default=5)
    freeze=sub.add_parser('freeze',help='after >=10 actual pose-profile validation rollouts; never accepts test rows')
    freeze.add_argument('--profiles',required=True,type=Path,help='derive output index.json')
    freeze.add_argument('--evaluations',required=True,type=Path,help='validation summary rows, JSON or JSONL')
    freeze.add_argument('--output',required=True,type=Path)
    freeze.add_argument('--evaluation-binding',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('refusing to replace '+str(args.output))
    if args.command=='derive':
        if min(args.min_samples,args.min_success,args.min_failure)<1:raise ValueError('positive minimum support required')
        records=read(args.records)
        if not isinstance(records,list):raise ValueError('record list required')
        result=derive_profiles(records,args.min_samples,args.min_success,args.min_failure)
        result['records_file']=str(args.records.resolve());result['records_file_sha256']=digest(args.records)
        result['trial_execution_status']='threshold candidates only; no closed-loop validation results implied'
        args.output.mkdir(parents=True)
        for config in result['profiles']:
            (args.output/(config['id']+'.json')).write_text(json.dumps(config,indent=2)+'\n')
        result['profile_files_sha256']={p.name:digest(p) for p in args.output.glob('pose_*.json')}
        (args.output/'index.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status=result['status'],profile_count=len(result['profiles']),source=result['source']),indent=2))
    else:
        profile_index=read(args.profiles)
        if profile_index.get('status')!='derived':raise ValueError('profiles must be successfully derived')
        summary=read(args.evaluations);binding=read(args.evaluation_binding)
        if summary.get('schema')!='eatnav_r6_queue_summary_v1' or summary.get('split')!='validation' or not summary.get('all_jobs_completed'):raise ValueError('completed actual validation queue required')
        cells=summary.get('matched_cell_counts',{})
        planned=summary.get('planned_cell_counts',{})
        if len(planned)!=35 or min(planned.values())<30 or len(cells)!=35 or min(cells.values())<1:raise ValueError('30planned trials per cell and positive matched valid support in all35cells required')
        if binding.get('role')!='validation' or binding.get('model_sha256')!=profile_index.get('model_sha256'):raise ValueError('validation/profile model binding mismatch')
        result=freeze_top_three(profile_index['profiles'],summary['selection_rows'])
        result['comparison_profiles']={c['id']:c['profile'] for c in binding['conditions'] if c['id'] in ('event_2s','combined')}
        result['evaluation_source_sha256']=binding['source_sha256']
        result['model_sha256']=binding['model_sha256'];result['cohort_sha256']=binding['cohorts_sha256'];result['matched_trials']=summary['matched_initial_valid_trials']
        # Freeze digest covers all selection inputs and exact profile parameters.
        result['sample_support_note']='30planned trials per field/level; initialization-invalid/missing excluded on the same paired cohort, actual denominators and CIs reported; no conclusive safety claim'
        result['input_files']={str(p.resolve()):digest(p) for p in (args.profiles,args.evaluations,args.evaluation_binding)}
        result.pop('freeze_sha256')
        result['freeze_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as file:file.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(freeze_sha256=result['freeze_sha256'],selected=[p['id'] for p in result['profiles']]),indent=2))


if __name__=='__main__':main()
