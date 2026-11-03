"""Run actual FAST-LIO per parent, with bounded CPU parallelism and strict provenance."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('bounded CPU parallelism 1..8')
    jobs=sorted(a.raw.glob('*/*/metadata.json'))
    if not jobs:raise ValueError('no parent raw records')
    def run(meta):
        parent=meta.parent;output=parent/'fastlio';done=output/'complete.json'
        if done.exists():
            record=json.loads(done.read_text())
            if record.get('source')!='actual_FAST_LIO_process' or hashlib.sha256((output/'odometry.npz').read_bytes()).hexdigest()!=record.get('odometry_npz_sha256'):raise ValueError('existing FAST-LIO output hash mismatch')
            return record
        if output.exists():raise ValueError('incomplete FAST-LIO output preserved for inspection: '+str(output))
        command=['bash','-c','source /home/ubuntu/onix/install/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1; exec /usr/bin/python3 "$@"','fastlio-replay',str(REPO/'scripts/eat_nav_single_pose2s_fastlio_replay.py'),'--parent',str(parent),'--output',str(output)]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        (parent/'fastlio-host.log').write_text(result.stdout)
        if result.returncode:raise RuntimeError('FAST-LIO failed '+str(parent)+'; see fastlio-host.log')
        return json.loads(done.read_text())
    report=a.raw.parent/'fastlio-processing-status.json';start=time.time();completed=0
    def status(phase,error=None):
        tmp=report.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,expected_parents=len(jobs),completed_parents=completed,workers=a.workers,elapsed_seconds=time.time()-start,error=error),indent=2)+'\n');tmp.replace(report)
    status('running')
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run,path) for path in jobs]
            try:
                for future in concurrent.futures.as_completed(futures):
                    record=future.result();completed+=1;status('running');print('FASTLIO '+str(completed)+'/'+str(len(jobs))+' '+record['parent_id'],flush=True)
            except BaseException:
                for future in futures:future.cancel()
                raise
        status('completed')
    except BaseException as exc:status('failed_preserved',repr(exc));raise

if __name__=='__main__':main()
