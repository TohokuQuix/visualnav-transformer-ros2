"""Post-conflict SIGTERM after follower evidence is durably flushed."""
from __future__ import annotations
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 conflict=ROOT/'same-stamp-command-conflict-first.json'
 if not conflict.exists():raise RuntimeError('no conflict evidence; refusing shutdown')
 record=json.loads(conflict.read_text());branch=record['branch']
 logs=[ROOT/'follower-logs'/f'reset-0001-branch-{i}.log' for i in range(8)]
 if any(not p.exists() or p.stat().st_size==0 for p in logs):raise RuntimeError('follower log missing')
 active=subprocess.run(['pgrep','-f','^/tmp/eatnav-candidate-coverage/tf-race-diagnostic/diagnostic-follower-build-r1/simple_trajectory_follower_node'],capture_output=True,text=True)
 if active.returncode==0:raise RuntimeError('follower still active; evidence may not be flushed')
 source=subprocess.check_output(['docker','exec','isaac-lab-onix','pgrep','-f','^/isaac-sim/kit/python/bin/python3 /tmp/eatnav-candidate-coverage/tf-race-diagnostic/sim.py'],text=True).split()
 if len(source)!=1:raise RuntimeError(f'expected one simulator process, got {source}')
 marker={'conflict_sha256':sha(conflict),'follower_log_sha256':{str(i):sha(p) for i,p in enumerate(logs)},
  'simulator_container_pid':int(source[0]),'signal':'SIGTERM','reason':'post-conflict graceful shutdown after callback logs flushed',
  'time_unix_ns':time.time_ns()}
 target=ROOT/'graceful-stop-request.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(marker,indent=2,sort_keys=True)+'\n');tmp.replace(target)
 subprocess.run(['docker','exec','isaac-lab-onix','kill','-TERM',source[0]],check=True)
 print('TF_RACE_GRACEFUL_STOP_REQUESTED',source[0],flush=True)
if __name__=='__main__':main()
