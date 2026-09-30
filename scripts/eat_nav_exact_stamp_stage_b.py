"""Run the twenty immutable 6-second Stage B trials after every exact-stamp gate."""
from __future__ import annotations
import gzip, hashlib, json, os, time
from pathlib import Path
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120
import eat_nav_exact_stamp_run as pre
import eat_nav_sim_clock_run as base

PREFLIGHT=Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon')
ROOT=Path('/tmp/eatnav-candidate-coverage/exact-stamp-stage-b')
LIMIT_SEC=5400
LIMIT_BYTES=250_000_000

def size_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

def compress_closed(log_dir):
    files=list(log_dir.glob('reset-*-branch-*.log'))
    if not files:return []
    current=max(int(p.name.split('-')[1]) for p in files)
    compressed=[]
    for path in files:
        if int(path.name.split('-')[1])>=current:continue
        dest=path.with_suffix('.log.gz')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if dest.exists():raise ValueError('compressed log destination already exists')
        tmp=dest.with_suffix('.gz.tmp')
        with gzip.open(tmp,'wb',compresslevel=6) as stream:stream.write(raw)
        with gzip.open(tmp,'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest()!=digest:raise ValueError('log compression digest mismatch')
        tmp.replace(dest);path.unlink()
        compressed.append({'path':str(dest),'raw_sha256':digest,'compressed_sha256':base.sha(dest)})
    return compressed

def main():
    began_unix=float(os.environ.get('EAT_EXACT_STAGE_B_START_UNIX',time.time()))
    started=time.monotonic()-(time.time()-began_unix);ROOT.mkdir(parents=True,exist_ok=True)
    required=['t4b.done.json','t4c.done.json','t4d.done.json','t4e.done.json']
    if any(not (PREFLIGHT/name).exists() for name in required):
        raise RuntimeError('preflight gate incomplete')
    pre.ROOT=ROOT;base.ROOT=ROOT;base.CONTAINER=str(ROOT);base.SIM=str(ROOT/'exact_sim.py')
    identities=pre.frozen()['runs']
    if len(identities)!=20 or [r['order'] for r in identities]!=list(range(20)):
        raise ValueError('frozen twenty-run order mismatch')
    pre.prepare()
    log_dir=Path(os.environ['EAT_EXACT_FOLLOWER_LOG_DIR'])
    complete=[];durations=[];compressed=[]
    try:
        for j,identity in enumerate(identities):
            elapsed=time.monotonic()-started
            estimate=max(durations[-3:],default=150.)
            if elapsed+estimate>LIMIT_SEC:
                raise TimeoutError(f'predicted wall limit exceeded before run {j}: elapsed={elapsed:.1f} estimate={estimate:.1f}')
            if size_bytes(ROOT)+30_000_000>LIMIT_BYTES:
                raise RuntimeError(f'predicted storage limit exceeded before run {j}: bytes={size_bytes(ROOT)}')
            run_id=f'long-{j:03d}'
            pre.atomic(ROOT/'status.json',{'stage':'T5_RUNNING','run_id':run_id,'completed':j,
                'elapsed_sec':elapsed,'remaining_estimate_sec':(20-j)*estimate,
                'size_bytes':size_bytes(ROOT),'wall_limit_sec':LIMIT_SEC,'storage_limit_bytes':LIMIT_BYTES})
            tick=time.monotonic()
            marker=base.run_one('stage-b',run_id,identity,120,identity['logical_t0_ns'],
                                deadline=time.time()+LIMIT_SEC-elapsed)
            pre.inspect_run(ROOT/'stage-b/runs'/run_id,120,identity['logical_t0_ns'])
            durations.append(time.monotonic()-tick)
            complete.append({'run_id':run_id,'identity':identity,'marker_sha256':base.sha(ROOT/'stage-b/runs'/run_id/'run-complete.json')})
            compressed.extend(compress_closed(log_dir))
            pre.atomic(ROOT/'compressed-log-sha.json',compressed)
            if (j+1)%5==3:
                state=identity['state_id']
                folders=[ROOT/'stage-b/runs'/f'long-{k:03d}' for k in range(j-2,j+1)]
                dup=duplicate_120([base.read(f/'live-0.json') for f in folders],
                                  [base.trace(f/'live-0.trace.json.gz') for f in folders],.0105)
                pre.atomic(ROOT/'duplicate-gates'/f'{state}.json',dup)
                if not dup['pass'] or not dup['dem_sha_equal']:
                    raise RuntimeError(f'{state} nominal duplicate gate failed')
            pre.atomic(ROOT/'runs'/f'{run_id}.done.json',complete[-1])
            if size_bytes(ROOT)>LIMIT_BYTES:raise RuntimeError('storage limit exceeded')
        pre.atomic(ROOT/'stage-b-runs-complete.json',{'status':'complete','run_count':20,
            'elapsed_sec':time.monotonic()-started,'size_bytes':size_bytes(ROOT),'runs':complete})
        pre.atomic(ROOT/'status.json',{'stage':'T5_PASS','completed':20,
            'elapsed_sec':time.monotonic()-started,'size_bytes':size_bytes(ROOT)})
        print('EXACT_STAGE_B_COMPLETE',20,flush=True)
    except BaseException as exc:
        pre.atomic(ROOT/'status.json',{'stage':'STOPPED','reason':repr(exc),'completed':len(complete),
            'elapsed_sec':time.monotonic()-started,'size_bytes':size_bytes(ROOT),
            'next_run':identities[len(complete)]['order'] if len(complete)<20 else None})
        print('EXACT_STAGE_B_STOP',repr(exc),flush=True)
        raise

if __name__=='__main__':main()
