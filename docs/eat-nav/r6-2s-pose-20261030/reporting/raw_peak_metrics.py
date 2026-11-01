"""20Hz observed-peak diagnostics; never fit a model, profile or missing future."""
import argparse,collections,gzip,json,math
from pathlib import Path
import numpy as np
from eat_nav_r6_run import read,write,sha,REPO
from eat_nav_r6_storage import read_trace
from visualnav_transformer.eat_nav.fall_stall_targets import post_state,quaternion,termination_reasons
from visualnav_transformer.eat_nav.r6_provenance import verify_execution_freeze,runtime_source_hashes


def peak_record(prediction,rows):
    prediction=np.asarray(prediction,float)
    if prediction.shape!=(20,9) or not np.isfinite(prediction).all():
        raise ValueError('finite20x9 forecast required')
    pairs=prediction[:,3:].reshape(20,3,2)
    if not np.allclose(np.linalg.norm(pairs,axis=-1),1,atol=.02):
        raise ValueError('normalized rotation pairs required')
    predicted=float(np.degrees(np.arccos(np.clip(prediction[:,6]*prediction[:,8],-1,1))).max())
    tilts=[];tip=False;reason='short_recording';first=None;stamp=None
    for row in rows[:40]:
        if row.get('boundary_censor') or row.get('reset') or row.get('manual_stop_before_step') or \
                (row.get('terminal') or {}).get('termination_reason')=='parallel_env_censored_by_other_env':
            reason='boundary_reset_or_censor';break
        if row.get('goal_hold') or row.get('protective_stop'):
            reason='control_override';break
        if first is None:first=row['control_step'];stamp=row['logical_sim_time_ns']
        n=len(tilts)
        if row['control_step']!=first+n or row['logical_sim_time_ns']!=stamp+n*50_000_000 or row['post_timestamp_ns']!=stamp+(n+1)*50_000_000:
            raise ValueError('contiguous20Hz intervals required')
        root=post_state(row);q=quaternion(root)
        if q is None or len(root)!=13 or not np.isfinite(root).all():raise ValueError('finite observed root required')
        _,x,y,_=q;angle=math.degrees(math.acos(float(np.clip(1-2*(x*x+y*y),-1,1))))
        tilts.append(angle);tip=tip or angle>=85 or 'tilt_over' in termination_reasons(row)
        if row.get('terminal') or row.get('external_intervention') or row.get('manual_stop'):
            reason='terminal_or_intervention';break
    full=len(tilts)==40
    if full:reason='full_two_second_observation'
    actual=max(tilts) if tilts else None;grid=max(tilts[1::2]) if len(tilts)>1 else None
    return dict(observed_rows=len(tilts),observed_sec=len(tilts)*.05,full_window=full,observation_end=reason,
        predicted_full2s_peak_tilt_deg=predicted,observed_peak_tilt_deg=actual,observed_grid_peak_tilt_deg=grid,
        grid_peak_gap_deg=actual-grid if grid is not None else None,
        full_window_absolute_error_deg=abs(actual-predicted) if full else None,
        underprediction_lower_bound_deg=max(0.,actual-predicted) if actual is not None else None,tip_observed=bool(tip))


def summarize(rows):
    def distribution(key):
        a=[r[key] for r in rows if r[key] is not None]
        return dict(count=len(a),mean=float(np.mean(a)) if a else None,max=max(a) if a else None,q95=float(np.quantile(a,.95)) if a else None)
    return dict(windows=len(rows),full_windows=sum(r['full_window'] for r in rows),partial_windows=sum(not r['full_window'] for r in rows),
        tip_observed_windows=sum(r['tip_observed'] for r in rows),full_window_max_tilt_MAE_deg=distribution('full_window_absolute_error_deg'),
        observed_underprediction_lower_bound_deg=distribution('underprediction_lower_bound_deg'),observed_20hz_minus_10hz_peak_deg=distribution('grid_peak_gap_deg'))


def evaluate(queue_root,output,interim=False):
    queue_root=Path(queue_root);output=Path(output)
    if output.exists():raise FileExistsError('fresh diagnostic output required')
    binding=read(queue_root/'frozen-binding.json');role=binding['role']
    if role=='test':
        frozen=read(queue_root.parent/'frozen-top3.json')
        verified=verify_execution_freeze(frozen,binding['model_sha256'],runtime_source_hashes(REPO))
        if binding['frozen_selection']!=verified['freeze_sha256']:raise ValueError('test freeze mismatch')
    elif role!='validation':raise ValueError('validation or frozen test queue required')
    summary=read(queue_root/'summary.json')
    if not interim and not summary['all_jobs_completed']:raise ValueError('complete queue required for final diagnostics')
    ledger=read(queue_root/'progress.json');records=[];skipped=collections.Counter();sources={}
    for jid,record in ledger.items():
        if record.get('state')!='completed' or record.get('execution')!='simulated':continue
        job=read(queue_root/jid/'assignment.json')
        if job['condition']['mode']=='old_reference':skipped['legacy_model_has_no_future_pose_head']+=1;continue
        base=queue_root/jid/'sim-output';manifest=read(base/'manifest.json')
        if manifest['model_sha256']!=binding['model_sha256']:raise ValueError('stored prediction/model mismatch')
        storage=read(base/'trace-storage.json');sources[jid]=dict(complete_sha256=sha(base/'complete.json'),trace_storage_sha256=sha(base/'trace-storage.json'))
        for folder in sorted(base.glob('group-*')):
            group=read(folder/'group.json')
            if group['role']!=role:raise ValueError('mixed split records')
            if not group['initial_valid']:skipped['initialization_invalid']+=1;continue
            meta=storage[f'{folder.name}/trace.jsonl']
            if sha(base/meta['compressed_path'])!=meta['compressed_sha256']:raise ValueError('compressed trace changed')
            rows=read_trace(folder/'trace.jsonl')
            for f in sorted(folder.glob('decision-*.json')):
                decision=read(f);slot=decision['selected_index'];step=decision['decision_step']
                if slot is None:skipped['STOP_has_no_executed_bank_candidate']+=1;continue
                if 'predicted_trajectory' not in decision:raise ValueError('missing recorded R6 forecast')
                window=rows[step:step+40]
                if not window or decision['decision_stamp_ns']!=window[0]['logical_sim_time_ns'] or any(t>decision['decision_stamp_ns'] for t in decision['dem_stamps_ns']):raise ValueError('causal decision/window timing mismatch')
                result=peak_record(np.asarray(decision['predicted_trajectory'])[slot],window)
                records.append(dict(result,condition=job['condition']['id'],field=job['field'],level=job['level'],group_id=group['group_id'],decision_step=step,candidate=slot,actual_source=str(folder)))
    if not records:raise ValueError('no actually executed R6 forecast windows')
    bycondition=collections.defaultdict(list);bycell=collections.defaultdict(list)
    for row in records:
        bycondition[row['condition']].append(row);bycell[f"{row['condition']}/{row['field']}/level{row['level']}"].append(row)
    report=dict(schema='eatnav_r6_observed_20hz_peak_diagnostics_v1',split=role,interim=bool(interim),queue_complete=summary['all_jobs_completed'],model_sha256=binding['model_sha256'],binding_sha256=sha(queue_root/'frozen-binding.json'),script_sha256=sha(Path(__file__)),overall=summarize(records),by_condition={k:summarize(v) for k,v in sorted(bycondition.items())},by_condition_field_level={k:summarize(v) for k,v in sorted(bycell.items())},skipped=dict(skipped),actual_source_provenance=sources,
        limitations=['MAE uses only complete observed2s candidate windows; natural terminal pre-reset pose is retained','Partial observed maximum is a lower bound on the unobserved full-horizon maximum: report underprediction lower bound, never partial absolute MAE','Observed20Hz peaks may still miss substep peaks; predictions remain on their frozen0.1sgrid','STOP/goal-hold overrides have no executed bank candidate supervision','Only recorded R6 forecasts used; legacy model has no futureXYZ/posehead; policy-equivalent reuse not duplicated','Correlated windows/conditions share source cohorts; no independent safety or causal claim','No fitting, threshold changes, new inference or robot commands'])
    output.mkdir(parents=True);write(output/'report.json',report)
    with gzip.open(output/'windows.jsonl.gz','wt') as out:
        for row in records:out.write(json.dumps(row,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--interim',action='store_true');a=p.parse_args();result=evaluate(a.queue_root,a.output,a.interim);print(json.dumps(dict(split=result['split'],interim=result['interim'],overall=result['overall']),indent=2))
