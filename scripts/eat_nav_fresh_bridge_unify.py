"""Build an immutable-by-convention analysis view of 100 old + 19 fresh runs.

The view contains links; original attempts, diagnostics, and equivalence runs stay separate.
"""
from __future__ import annotations
import datetime,json,pathlib
from eat_nav_fresh_bridge_run import ROOT,OLD,DOC,atomic,read,sha
from eat_nav_fresh_bridge_resume import used_bytes,verify_old_hundred

VIEW=ROOT/'unified'

def link_once(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    target=pathlib.Path(target).resolve()
    if path.is_symlink():
        if path.resolve()!=target:raise ValueError('link target changed: '+str(path))
    elif path.exists():raise FileExistsError(path)
    else:path.symlink_to(target)

def assemble():
    raw=read(ROOT/'status.json')
    if raw['stage']!='RAW_COMPLETE' or raw['completed']!=119:raise RuntimeError('119 raw runs not complete')
    if read(ROOT/'r4-equivalence-gate.json')['stage']!='PASS' or not read(ROOT/'r5-stress-gate.json')['pass']:
        raise RuntimeError('equivalence/lifecycle gate missing')
    schedule=read(DOC/'schedule.json')['runs']
    if len(schedule)!=119:raise ValueError('schedule count')
    verify_old_hundred(schedule)
    VIEW.mkdir(parents=True,exist_ok=True)
    ledger=[]
    for entry in schedule:
        ident=entry['run_id'];order=entry['order']
        source=(OLD/'runs'/ident) if order<100 else ROOT/'runs'/('measurement-'+ident)
        complete=read(source/'run-complete.json')
        if complete['identity']!=entry:raise ValueError('identity mismatch '+ident)
        for name,digest in complete['output_sha256'].items():
            if sha(source/name)!=digest:raise ValueError('output SHA mismatch '+ident+'/'+name)
        marker=(OLD if order<100 else ROOT)/'markers'/(ident+'.done.json')
        if read(marker)['run_complete_sha256']!=sha(source/'run-complete.json'):
            raise ValueError('done SHA mismatch '+ident)
        link_once(VIEW/'runs'/ident,source)
        link_once(VIEW/'markers'/marker.name,marker)
        ledger.append({'run_id':ident,'protocol':'old' if order<100 else 'fresh',
                       'complete_sha256':sha(source/'run-complete.json')})
    for n in range(7):
        ident=f'readiness-{n:03d}.json'
        source=(OLD if n<6 else ROOT)/'duplicate-gates'/ident
        if not read(source)['pass']:raise ValueError('duplicate gate failed '+ident)
        link_once(VIEW/'duplicate-gates'/ident,source)
    link_once(VIEW/'runner.log',ROOT/'resume-supervisor.log')
    old_sec=read(OLD/'status.json')['elapsed_sec'];new_sec=raw['elapsed_sec']
    size=used_bytes()['total']
    if size>1_500_000_000:raise RuntimeError('1.5GB storage cap exceeded')
    atomic(VIEW/'source-ledger.json',{'schedule_sha256':sha(DOC/'schedule.json'),
           'old_completed':100,'fresh_completed':19,'old_raw_status_sha256':sha(OLD/'status.json'),
           'fresh_raw_status_sha256':sha(ROOT/'status.json'),'runs':ledger,
           'diagnostic_and_equivalence_excluded':True})
    atomic(VIEW/'status.json',{'stage':'RAW_COMPLETE','completed_processes':119,
           'completed_rollouts':112,'elapsed_sec':old_sec+new_sec,'old_elapsed_sec':old_sec,'fresh_elapsed_sec':new_sec,
           'size_bytes':size,'storage_cap_bytes':1_500_000_000,
           'source_ledger_sha256':sha(VIEW/'source-ledger.json'),
           'protocol':'100_old_plus_19_fresh_equivalence_gated'})
    return VIEW

if __name__=='__main__':print('UNIFIED',assemble(),flush=True)
