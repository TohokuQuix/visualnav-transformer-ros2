import hashlib,json
import pytest
from visualnav_transformer.eat_nav.r6_provenance import runtime_source_hashes,verify_execution_freeze

def sealed(**updates):
 data=dict(schema='eatnav_r6_frozen_top_three_v1',source='validation_only',profiles=[{'id':str(i)} for i in range(3)],model_sha256='model',evaluation_source_sha256={'src/module.py':'source'})
 data.update(updates)
 data['freeze_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return data

def test_frozen_test_rejects_tamper_model_and_policy_before_simulation():
 good=sealed();assert verify_execution_freeze(good,'model',{'src/module.py':'source'})==good
 for bad in (dict(good,freeze_sha256='invalid'),sealed(model_sha256='wrong'),sealed(evaluation_source_sha256={'src/module.py':'changed'})):
  with pytest.raises(ValueError):verify_execution_freeze(bad,'model',{'src/module.py':'source'})

def test_staged_transitive_sources_and_reference_index_are_bound(tmp_path):
 for name in ('src/visualnav_transformer/eat_nav/dem.py','src/visualnav_transformer/eat_nav/config.py','scripts/eat_nav_r4.py','config/eat_nav.yaml','checkpoints/eat_nav/rl_actor/model_10000.pt','checkpoints/eat_nav/rl_actor/env.yaml','checkpoints/eat_nav/fall_stall_r4star_r5_index.json'):
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
 original=runtime_source_hashes(tmp_path)
 for name in ('src/visualnav_transformer/eat_nav/dem.py','src/visualnav_transformer/eat_nav/config.py','scripts/eat_nav_r4.py','checkpoints/eat_nav/fall_stall_r4star_r5_index.json'):
  p=tmp_path/name;p.write_text('changed');changed=runtime_source_hashes(tmp_path)
  assert changed[name]!=original[name]
  with pytest.raises(ValueError):verify_execution_freeze(sealed(evaluation_source_sha256=original),'model',changed)
  p.write_text('original')
 assert runtime_source_hashes(tmp_path)==original

def test_test_queue_rejects_bogus_freeze_before_writing_binding(tmp_path,monkeypatch):
 import eat_nav_r6_eval_queue as queue
 checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'model')
 sources={'src/module.py':'source'}
 monkeypatch.setattr(queue,'runtime_source_hashes',lambda repo:sources)
 profiles={'profiles':[{'id':str(i)} for i in range(12)]}
 bad=sealed(comparison_profiles={'event_2s':profiles['profiles'][-1],'combined':profiles['profiles'][-1]})
 bad['freeze_sha256']='not-a-valid-digest'
 output=tmp_path/'test'
 with pytest.raises(ValueError,match='digest'):
  queue.run_queue({'test_rollouts':[]},checkpoint,profiles,output,'test',frozen=bad,max_jobs=0)
 assert not (output/'frozen-binding.json').exists()

def test_validation_queue_rejects_changed_transitive_source_on_resume(tmp_path,monkeypatch):
 import eat_nav_r6_eval_queue as queue
 checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'model')
 sources={'src/visualnav_transformer/eat_nav/dem.py':'original'}
 monkeypatch.setattr(queue,'runtime_source_hashes',lambda repo:dict(sources))
 profiles={'profiles':[{'id':str(i)} for i in range(12)]}
 queue.run_queue({'validation_rollouts':[]},checkpoint,profiles,tmp_path/'validation','validation',max_jobs=0)
 sources['src/visualnav_transformer/eat_nav/dem.py']='changed'
 with pytest.raises(ValueError,match='binding changed'):
  queue.run_queue({'validation_rollouts':[]},checkpoint,profiles,tmp_path/'validation','validation',max_jobs=0)
