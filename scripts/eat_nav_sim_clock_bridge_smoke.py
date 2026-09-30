"""Live ROS/production-follower protocol smoke without Isaac Sim."""
import json,math,sys
from pathlib import Path
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient

root=Path('/tmp/eatnav-candidate-coverage')
preview=json.loads((root/'command-preview-long-horizon/long-horizon/preview-120/coverage-001--bank-nominal.json').read_text())
path=json.loads((root/'pilot-inputs/state-001/bank-nominal-map-path.json').read_text())
pose=preview['initial_map_pose']
print('pose',pose,flush=True)
client=FollowerBridgeClient(timeout=20.)
try:
    reset=client.request('reset',stamp_ns=1_000_000_000,poses=[pose],paths=[path])
    print('reset',json.dumps(reset),flush=True)
    for k in range(1,5):
        resp=client.request('step',stamp_ns=1_000_000_000+k*50_000_000,poses=[pose],timeout_sec=3)
        print('step',k,json.dumps(resp),flush=True)
    print('final',json.dumps(client.request('finish')),flush=True)
finally:client.close()
