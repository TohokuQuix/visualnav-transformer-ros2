"""Finish the diagnostic bridge after all run outputs and follower files close."""
from __future__ import annotations
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')
def main():
 status=json.loads((ROOT/'status.json').read_text())
 if status.get('stage')!='R4_COMPLETE' or status.get('repeats')!=1:
  raise RuntimeError('diagnostic not complete; refusing cleanup')
 if not (ROOT/'repeat-00/stage-complete.json').exists():raise RuntimeError('repeat marker missing')
 if subprocess.run(['pgrep','-f','^/tmp/eatnav-candidate-coverage/tf-race-diagnostic/diagnostic-follower-build-r1/simple_trajectory_follower_node'],capture_output=True).returncode==0:
  raise RuntimeError('follower still running')
 pids=subprocess.check_output(['pgrep','-f','^/usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_tf_race_bridge.py'],text=True).split()
 if len(pids)!=1:raise RuntimeError(f'unexpected bridge processes {pids}')
 pid=int(pids[0]);record={'bridge_pid':pid,'stage':'post-completion cleanup','term_already_sent_by_detached_shell':True,
   'follower_closed':True,'run_marker_present':True,'signal':'SIGKILL after bridge failed to exit on SIGTERM',
   'time_unix_ns':time.time_ns()}
 marker=ROOT/'bridge-cleanup.json';tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');tmp.replace(marker)
 os.kill(pid,signal.SIGKILL)
 print('BRIDGE_CLEANED',pid)
if __name__=='__main__':main()
