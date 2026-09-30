"""Detached, one-shot R9 supervisor after the 119-run raw readiness gate."""
from __future__ import annotations
import datetime,os,subprocess,time
from eat_nav_fresh_bridge_run import ROOT,REPO,atomic,read,sha
from eat_nav_fresh_bridge_resume import used_bytes
from eat_nav_fresh_bridge_unify import assemble

CUTOFF=datetime.datetime(2026,9,30,14,5,tzinfo=datetime.timezone.utc).timestamp()
CAP=1_500_000_000

def launch(stage,script,timeout):
    log=ROOT/(stage+'-supervisor.log')
    env=os.environ.copy();env['PYTHONPATH']='src:scripts'
    env['EAT_READINESS_ROOT']=str(ROOT/'unified')
    with log.open('x') as stream:
        result=subprocess.run(['/usr/bin/python3','-u','scripts/'+script],cwd=REPO,
                              env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(stage+' failed with exit '+str(result.returncode))
    return {'log_sha256':sha(log),'exit_code':result.returncode}

def main():
    while time.time()<CUTOFF:
        status=read(ROOT/'status.json')
        if status['stage']=='RAW_COMPLETE':break
        if status['stage'] in ('STOPPED','MEASUREMENT_INVALID'):
            atomic(ROOT/'r9-status.json',{'stage':'BLOCKED_BY_RAW_GATE','raw_stage':status['stage'],
                                          'stop_reason':status.get('stop_reason')})
            return
        time.sleep(20)
    else:raise TimeoutError('global 4.5-hour cap before raw complete')
    if used_bytes()['total']>CAP:raise RuntimeError('storage cap before R9')
    view=assemble()
    atomic(ROOT/'r9-status.json',{'stage':'POSTPROCESS_RUNNING','unified_status_sha256':sha(view/'status.json')})
    remaining=min(3600,max(1,int(CUTOFF-time.time())))
    post=launch('r9-postprocess','eat_nav_evaluator_readiness_postprocess.py',remaining)
    if read(view/'postprocess-status.json')['stage']!='COMPLETE':raise RuntimeError('postprocess status incomplete')
    if used_bytes()['total']>CAP:raise RuntimeError('storage cap after postprocessing')
    atomic(ROOT/'r9-status.json',{'stage':'FINALIZE_RUNNING','postprocess':post})
    final=launch('r9-finalize','eat_nav_evaluator_readiness_finalize.py',min(120,max(1,int(CUTOFF-time.time()))))
    if used_bytes()['total']>CAP:raise RuntimeError('storage cap after finalize')
    atomic(ROOT/'r9-status.json',{'stage':'COMPLETE','postprocess':post,'finalize':final,
                                 'storage':used_bytes(),'unified_status_sha256':sha(view/'status.json')})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic(ROOT/'r9-status.json',{'stage':'STOPPED','reason':repr(exc),'storage':used_bytes()})
        raise
