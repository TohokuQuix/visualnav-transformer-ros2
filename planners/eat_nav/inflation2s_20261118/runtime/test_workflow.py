import unittest,tempfile,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import workflow
class WorkflowTests(unittest.TestCase):
 def test_resume_identity_covers_helpers_mesh_and_physics(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);base=r/'base';source=r/'source';(r/'code').mkdir();(base/'actor').mkdir(parents=True);(base/'runs/scene1-smoke-r3').mkdir(parents=True);(source/'src/visualnav_transformer/eat_nav').mkdir(parents=True);(source/'docs/eat-nav/model2000-pose2s-20261110/models/seed123').mkdir(parents=True)
   for name in ['navigation_sim.py','nav_contract.py','pose_contract.py','rp_change_rule.py','hold_schedule.py','initial_gate.py','inflation_rule.py','feasibility.py','kinematics.py','collision_geometry.py']:(r/'code'/name).write_text(name)
   (r/'checkpoint.pt').write_bytes(b'NEWNN');(r/'protocol.json').write_text('{}');(r/'mesh.npz').write_bytes(b'mesh')
   for name in ['model_2000.pt','env.yaml','agent.yaml']:(base/'actor'/name).write_text(name)
   (source/'src/visualnav_transformer/eat_nav/model.py').write_text('a=1');(source/'docs/eat-nav/model2000-pose2s-20261110/models/seed123/checkpoint.pt').write_bytes(b'NN')
   np.savez(base/'runs/scene1-smoke-r3/physics.npz',masses=np.ones((1,1)),inertias=np.ones((1,1,9)),material=np.ones((1,1,3)))
   with patch.multiple(workflow,ROOT=r,BASE=base,SOURCE=source,EXTERNAL_ROOTS=[]):
    a=workflow.run_identity(r/'mesh.npz');b=workflow.run_identity(r/'mesh.npz');self.assertEqual(a,b)
    import hashlib
    self.assertEqual(a['nn_sha256'],hashlib.sha256(b'NEWNN').hexdigest())
    (r/'code/nav_contract.py').write_text('changed');c=workflow.run_identity(r/'mesh.npz');self.assertNotEqual(a,c)
    (r/'code/hold_schedule.py').write_text('timing changed');self.assertNotEqual(c,workflow.run_identity(r/'mesh.npz'))
    (r/'code/inflation_rule.py').write_text('geometry changed');self.assertNotEqual(c,workflow.run_identity(r/'mesh.npz'))
    (r/'mesh.npz').write_bytes(b'mesh2');self.assertNotEqual(c,workflow.run_identity(r/'mesh.npz'))
 def test_owned_process_cleanup(self):
  p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
  try:workflow.cleanup_process(p);self.assertIsNotNone(p.poll())
  finally:
   if p.poll() is None:p.kill();p.wait()
