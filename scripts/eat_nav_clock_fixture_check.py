"""Compare logging-only follower output to frozen production fixtures."""
from __future__ import annotations
import json,os,socket,subprocess,time
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from eat_nav_clock_diagnostic_stage0 import OUT,OLD,FOLLOWER,atomic_json,sha

PORT=47829

def main():
    build=json.loads((OUT/'diagnostic-build.json').read_text())
    binary=Path(build['binary']);assert sha(binary)==build['binary_sha256']
    fixture=json.loads((Path(__file__).resolve().parents[1]/'tests/eat_nav/fixtures/command_preview_production.json').read_text())
    assert len(fixture['fixtures'])==7
    log=OUT/'fixture-bridge-2.log'
    env=os.environ.copy();env.update({'ROS_DOMAIN_ID':'77','RMW_IMPLEMENTATION':'rmw_fastrtps_cpp'})
    with log.open('x') as stream:
        proc=subprocess.Popen(['/usr/bin/python3',str(Path(__file__).resolve().parents[0]/'eat_nav_phase_c_ros_bridge.py'),
            '--host','127.0.0.1','--port',str(PORT),'--branches','8','--follower',str(binary),
            '--params',str(FOLLOWER/'config/onix.yaml'),'--diagnostic-dir',str(OUT/'fixture-follower-logs')],
            stdout=stream,stderr=subprocess.STDOUT,env=env)
        try:
            for _ in range(150):
                if proc.poll() is not None:raise RuntimeError('fixture bridge exited')
                try:
                    with socket.create_connection(('127.0.0.1',PORT),timeout=.1):break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('fixture bridge startup')
            client=FollowerBridgeClient(port=PORT,timeout=30.)
            rows=[]
            try:
                for row in fixture['fixtures']:
                    path=OLD/'preview-candidates'/f"{row['state_id']}--{row['candidate_id']}.json"
                    record=json.loads(path.read_text());pose=record['initial_map_pose'];map_path=record['map_path']
                    client.request('reset',poses=[pose]*8,paths=[map_path]*8)
                    response=client.request('step',poses=[pose]*8,timeout_sec=2.)
                    observed=np.asarray(response['commands'][0]);expected=np.asarray(row['production_command'])
                    error=float(np.max(np.abs(observed-expected)))
                    rows.append({'state_id':row['state_id'],'candidate_id':row['candidate_id'],
                        'diagnostic_command':observed.tolist(),'production_command':expected.tolist(),
                        'max_abs_error':error,'pass':bool(error<=row['tolerance']),
                        'input_sha256':sha(path)})
                    if error>row['tolerance']:raise AssertionError(f"fixture failed {row['state_id']}: {error}")
            finally:client.close()
        finally:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    atomic_json(OUT/'diagnostic-fixture-parity.json',{'status':'pass','tolerance':1e-5,
        'production_binary_sha256':build['production_binary_sha256'],
        'diagnostic_binary_sha256':build['binary_sha256'],'fixtures':rows})
    print('DIAGNOSTIC_FIXTURE_PASS',len(rows),max(x['max_abs_error'] for x in rows))

if __name__=='__main__':main()
