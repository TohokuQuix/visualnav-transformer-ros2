from pathlib import Path
import json,numpy as np
from rp_change_rule import C,score_components
from scipy.spatial.transform import Rotation

def test_selected_condition_weight_and_14candidate_scores():
 protocol=json.loads((Path(__file__).parents[1]/'protocol.json').read_text());assert C==protocol['rp_change_c_m_per_10deg']
 current=Rotation.from_euler('xyz',[12,-8,60],degrees=True).as_matrix();future=Rotation.from_euler('xyz',[[i-6,i%5-2,10*i] for i in range(14)],degrees=True).as_matrix();relative=current.T@future
 g=np.linspace(-.1,.3,14);x=score_components(g,relative,current,np.ones(14));now=Rotation.from_matrix(current).as_euler('xyz',degrees=True);end=Rotation.from_matrix(future).as_euler('xyz',degrees=True);d=(end[:,:2]-now[:2]+180)%360-180
 bonus=np.full(14,-.5)
 np.testing.assert_allclose(x['score_m'],g-protocol['rp_change_c_m_per_10deg']*np.linalg.norm(d,axis=1)/10+bonus,atol=1e-12)
 if C==0:np.testing.assert_array_equal(x['score_m'],g+bonus)


def test_formal_initialization_failure_does_not_retry_reduced_batch(tmp_path,monkeypatch):
 import workflow,pytest
 monkeypatch.setattr(workflow,'ROOT',tmp_path)
 monkeypatch.setattr(workflow,'run_identity',lambda mesh:{})
 parents=[dict(trial_id='t'+str(i)) for i in range(10)]
 monkeypatch.setattr(workflow,'parents_for',lambda name:parents)
 (tmp_path/'mesh').write_bytes(b'fixture');(tmp_path/'protocol.json').write_text('{}')
 batch=tmp_path/'batch'
 for p in parents:
  d=batch/p['trial_id'];d.mkdir(parents=True);(d/'run.json').write_text(json.dumps(dict(result='initialization_failure')))
 calls=[]
 def job(*args):calls.append(args);return batch
 monkeypatch.setattr(workflow,'job',job)
 with pytest.raises(RuntimeError):workflow.cohort('original',tmp_path/'mesh')
 assert len(calls)==1
