import json,hashlib
import pytest
from visualnav_transformer.eat_nav.r6_planner import verified_freeze

def test_frozen_digest_prevents_silent_profile_modification(tmp_path):
 d=dict(schema='eatnav_r6_frozen_top_three_v1',source='validation_only',profiles=[dict(pitch_deg=10)])
 d['freeze_sha256']=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 path=tmp_path/'freeze.json';path.write_text(json.dumps(d));assert verified_freeze(path)['profiles'][0]['pitch_deg']==10
 d['profiles'][0]['pitch_deg']=20;path.write_text(json.dumps(d))
 with pytest.raises(ValueError):verified_freeze(path)
