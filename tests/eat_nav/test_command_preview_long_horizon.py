import pytest
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120,endpoint_metrics

def row(x,cmd=.3):
 return {'root_state':[x,0,0,1,0,0,0],'post_root_state':[x+.01,0,0,1,0,0,0],
  'command':[cmd,0,0],'actor_input':[0]*22,'actor_latent':[0]*256,'actor_action':[0]*6,
  'critic_observation':[0]*267,'critic_value':1.,'reward':0.,'contact_force':[[0.,0.,0.]],
  'dem':{'transformed_sha256':'abc'},'terminal':None}

def test_duplicate_gate_and_step_routing():
 traces=[[row(k*.01) for k in range(120)] for _ in range(3)]
 anchors=[{'root_state':[0]*13,'fixed_map_path_sha256':'a'} for _ in range(3)]
 result=duplicate_120(anchors,traces,.0105)
 assert result['pass'] and result['xy_rms_m']==0 and result['step_count']==120
 a=endpoint_metrics(traces[0],40,1.0,[0,0],[1,0])
 b=endpoint_metrics(traces[0],120,1.0,[0,0],[1,0])
 assert a['step_count']==40 and b['step_count']==120

def test_duplicate_detects_path_mismatch():
 traces=[[row(k*.01) for k in range(120)] for _ in range(3)]
 anchors=[{'root_state':[0]*13,'fixed_map_path_sha256':'a'},{'root_state':[0]*13,'fixed_map_path_sha256':'b'},{'root_state':[0]*13,'fixed_map_path_sha256':'a'}]
 with pytest.raises(ValueError):duplicate_120(anchors,traces,.0105)

def test_resume_marker_sha_check(tmp_path,monkeypatch):
 import hashlib,json
 from scripts import eat_nav_command_preview_long_horizon_run as runner
 monkeypatch.setattr(runner,'RUNROOT',tmp_path)
 run={'order':0,'candidate_id':'bank-nominal'}
 folder=tmp_path/'runs'/'long-000';folder.mkdir(parents=True)
 data=folder/'live-0.json';data.write_text('{}')
 marker={'frozen_run':run,'files_sha256':{'live-0.json':hashlib.sha256(data.read_bytes()).hexdigest()}}
 (folder/'run-complete.json').write_text(json.dumps(marker))
 assert runner.verify_run(run)
 data.write_text('{"changed":true}')
 with pytest.raises(AssertionError):runner.verify_run(run)
