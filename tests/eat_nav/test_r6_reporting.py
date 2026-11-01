import importlib.util
from pathlib import Path
import pytest
path=Path.cwd()/'docs/eat-nav/r6-2s-pose-20261030/reporting/render_report.py'
spec=importlib.util.spec_from_file_location('r6_report',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row(gid,valid=True,**kw):
 return dict(group_id=gid,initial_valid=valid,near_fall_count=1,intervention=False,timeout=True,processing_sec=2.,decision_count=4,all_candidates_excluded_decisions=1,observed_sec=10.,unknown_stall_windows=2,**kw)
def test_diagnostics_common_denominator_and_decision_frequency():
 ledger={'a':dict(state='completed',results=[row('ok'),row('invalid')]),'b':dict(state='completed',results=[row('ok'),row('invalid',False)])}
 jobs=[dict(job_id='a',condition={'id':'A'}),dict(job_id='b',condition={'id':'B'})]
 out=m.diagnostic_rows({'all_jobs_completed':True,'matched_initial_valid_trials':1},ledger,jobs)
 assert out[0]==['A',1,1,0,1,'2.000','0.500',1,'0.250',2]
 assert out[1][0]=='B' and out[1][1]==1

def test_diagnostics_refuses_incomplete_and_mismatch():
 with pytest.raises(ValueError):m.diagnostic_rows({'all_jobs_completed':False}, {}, [])
 ledger={'a':dict(state='completed',results=[row('x')])};jobs=[dict(job_id='a',condition={'id':'A'})]
 with pytest.raises(ValueError):m.diagnostic_rows({'all_jobs_completed':True,'matched_initial_valid_trials':2},ledger,jobs)

def raw_report():
 return dict(schema='eatnav_r6_observed_20hz_peak_diagnostics_v1',split='validation',interim=False,queue_complete=True,model_sha256='frozen-model',binding_sha256='frozen-queue',by_condition={'pose':dict(windows=2,full_windows=1,partial_windows=1,tip_observed_windows=1,full_window_max_tilt_MAE_deg=dict(count=1,mean=3.,max=3.,q95=3.),observed_underprediction_lower_bound_deg=dict(count=2,mean=51.,max=100.,q95=95.),observed_20hz_minus_10hz_peak_deg=dict(count=2,mean=2.,max=4.,q95=3.8))})

def test_raw_peak_table_keeps_partial_peak_failure_separate_from_full_mae():
 assert m.raw_peak_rows(raw_report(),'validation','frozen-model','frozen-queue')==[['pose',1,1,1,'3.000','100.000','4.000']]

def test_raw_peak_final_table_rejects_incomplete_wrong_binding_or_mixed_mae():
 for key,value in [('interim',True),('queue_complete',False),('split','test'),('model_sha256','other-model')]:
  report=raw_report();report[key]=value
  with pytest.raises(ValueError):m.raw_peak_rows(report,'validation','frozen-model','frozen-queue')
 report=raw_report();report['by_condition']['pose']['full_window_max_tilt_MAE_deg']['count']=2
 with pytest.raises(ValueError):m.raw_peak_rows(report,'validation','frozen-model','frozen-queue')

def test_raw_peak_final_table_rejects_same_model_wrong_queue_binding():
 report=raw_report();report['binding_sha256']='another-cohort-or-profile'
 with pytest.raises(ValueError):m.raw_peak_rows(report,'validation','frozen-model','frozen-queue')
