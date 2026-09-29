"""One bounded real-follower startup probe with the frozen state 42 Path."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import time


def request(stream, action, **fields):
    stream.write((json.dumps({'action':action, **fields})+'\n').encode())
    stream.flush()
    result=json.loads(stream.readline())
    if not result.get('ok'):
        raise RuntimeError(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--anchor',type=Path,required=True)
    parser.add_argument('--map-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--external-bridge',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    anchor=json.loads(args.anchor.read_text())['anchor']
    root=anchor['root_state']
    w,x,y,z=root[3:7]
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    path=json.loads(args.map_path.read_text())
    pose=[path[0][0],path[0][1],yaw]
    log_path=args.output/'host-probe-bridge.log'
    with log_path.open('w') as log:
        process=None if args.external_bridge else subprocess.Popen(['/tmp/eatnav-phase-c-ros.sh'],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+15
            while True:
                try:
                    conn=socket.create_connection(('127.0.0.1',47819),timeout=1)
                    break
                except OSError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.1)
            conn.settimeout(30)
            with conn, conn.makefile('rwb') as stream:
                reset=request(stream,'reset',poses=[pose]*8,paths=[path]*8)
                step=request(stream,'step',poses=[pose]*8,timeout_sec=2)
                time.sleep(2)
                status=request(stream,'status')
            result={'pose':pose,'reset':reset,'step':step,'status_after_2s':status,
                    'follower_pids':reset['followers'],'bridge_pid':None if process is None else process.pid}
            tmp=args.output/'host-probe.json.tmp'
            tmp.write_text(json.dumps(result,indent=2)+'\n')
            tmp.replace(args.output/'host-probe.json')
            print(json.dumps({'step_fresh':step['fresh'],'sequence_at_step':step['sequence'],
                              'sequence_after_2s':status['sequence']}),flush=True)
        finally:
            if process is not None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            for pid in reset['followers'] if 'reset' in locals() else []:
                try:subprocess.run(['kill','-TERM',str(pid)],check=False)
                except OSError:pass

if __name__=='__main__':main()
