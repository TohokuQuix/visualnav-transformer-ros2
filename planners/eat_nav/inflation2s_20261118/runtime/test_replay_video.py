import unittest,tempfile,json,subprocess
from pathlib import Path
import numpy as np
from render_video import load_trace,replay_indices,joint_permutation,VideoStream,read_frame
class ReplayTests(unittest.TestCase):
 def trace(self,p,**overrides):
  d=dict(root=np.tile([.6,1.7,.3,1,0,0,0,0,0,0,0,0,0],(11,1)),joint_position=np.zeros((11,3)),joint_velocity=np.zeros((11,3)),time_s=np.arange(11)*.05);d.update(overrides);np.savez(p,**d)
 def test_exact_past_sample_and_terminal(self):
  t=np.arange(12)*.05;idx=replay_indices(t,5);np.testing.assert_array_equal(idx,[0,4,8,11]);self.assertEqual(idx[-1],len(t)-1)
 def test_reject_nonmonotonic_or_non20hz_trace(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'t.npz';self.trace(p,time_s=np.zeros(11))
   with self.assertRaises(ValueError):load_trace(p)
   self.trace(p,time_s=np.arange(11)*.1)
   with self.assertRaises(ValueError):load_trace(p)
 def test_root_wxyz_and_shape_validated(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'t.npz';self.trace(p);self.assertEqual(load_trace(p)['root'].shape,(11,13));self.trace(p,root=np.zeros((11,13)))
   with self.assertRaises(ValueError):load_trace(p)
 def test_reject_nonfinite_joints_and_batch_dimension(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'t.npz';self.trace(p,joint_position=np.full((11,3),np.nan))
   with self.assertRaises(ValueError):load_trace(p)
   self.trace(p,root=np.zeros((11,1,13)))
   with self.assertRaises(ValueError):load_trace(p)
 def test_joint_names_remap_and_duplicates_rejected(self):
  np.testing.assert_array_equal(joint_permutation(['r','l'],['l','r'],2),[1,0])
  np.testing.assert_array_equal(joint_permutation(None,['l','r'],2),[0,1])
  with self.assertRaises(ValueError):joint_permutation(['l','l'],['l','r'],2)
  with self.assertRaises(ValueError):joint_permutation(None,['l','r'],3)
 def test_float32_timestamps_preserve20hz_decimation(self):
  t=np.arange(6001,dtype=np.float32)*np.float32(.05);np.testing.assert_array_equal(replay_indices(t,5),np.arange(0,6001,4))
 def test_truncated_frame_rejected_without_padding(self):
  import io
  self.assertEqual(bytes(read_frame(io.BytesIO(b'abc'),3)),b'abc')
  with self.assertRaises(EOFError):read_frame(io.BytesIO(b'ab'),3)
 def test_stream_is_decodable_and_frame_count_correct(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'v.mp4'
   with VideoStream(p,128,96,5) as video:
    for i in range(3):video.write(np.full((96,128,3),i*70,np.uint8),float(i)/5,'trial7')
   result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(p)]));s=result['streams'][0];self.assertEqual(int(s['nb_frames']),3);self.assertEqual(s['width'],128)
if __name__=='__main__':unittest.main()
