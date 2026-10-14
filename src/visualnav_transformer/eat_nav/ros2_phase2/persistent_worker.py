"""Bounded local CPU worker; keeps Torch outside the ROS 3.12 process."""

import json
import os
from pathlib import Path
import select
import subprocess

import numpy as np

from .predictor_client import (validate_mode, inverse_target,
                               prepare_model_input, PATHS,
                               validate_adoption_artifacts)


class ModelWorkerClient:
    def __init__(self, python, mode, checkpoint, *, results=None,
                 normalization=None, timeout_s=10):
        validate_mode(mode,7 if mode=='legacy6_demo' else 14,checkpoint)
        if mode=='mock14_transport':
            raise ValueError('mock uses no worker')
        self.mode=mode
        self.timeout_s=timeout_s
        self.norm=(json.loads(Path(normalization).read_text()) if normalization else None)
        if mode=='learned14_adopted':
            checked_results,checked_norm=validate_adoption_artifacts(results,checkpoint)
            if checked_norm!=self.norm:
                raise ValueError('Stage C normalization file mismatch')
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=''
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[3])
        self.process=subprocess.Popen([str(python),'-u','-m',
            'visualnav_transformer.eat_nav.ros2_phase2.model_worker_server'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,bufsize=1,env=env)
        try:
            init={'op':'init','mode':mode,'checkpoint':str(checkpoint)}
            if mode=='learned14_adopted':
                if results is None or self.norm is None:
                    raise ValueError('adopted worker needs results/normalization')
                init['results']=str(results)
                init['normalization']=self.norm
            result=self._request(init)
            if result.get('status')!='ready':
                raise RuntimeError('model worker not ready')
            self.checkpoint_sha256=result['checkpoint_sha256']
        except BaseException:
            self.close()
            raise

    def _request(self,payload):
        line=json.dumps(payload,separators=(',',':'))
        if len(line)>1_000_000 or self.process.poll() is not None:
            raise RuntimeError('worker unavailable/request oversized')
        self.process.stdin.write(line+'\n'); self.process.stdin.flush()
        ready,_,_=select.select([self.process.stdout],[],[],self.timeout_s)
        if not ready:
            raise TimeoutError('model worker deadline')
        response=self.process.stdout.readline(100_001)
        if not response or len(response)>=100_000:
            raise RuntimeError('model worker exited/oversized response')
        result=json.loads(response)
        if 'error' in result:
            raise RuntimeError('model worker: '+result['error'])
        return result

    def predict_legacy(self,dem,state,robot_pose_map,goal_map):
        if self.mode!='legacy6_demo': raise ValueError('wrong worker mode')
        result=self._request({'op':'predict','dem':np.asarray(dem,np.float32).tolist(),
            'state':np.asarray(state,np.float32).tolist(),
            'robot_pose_map':list(robot_pose_map),'goal_map':list(goal_map)})
        motion=np.asarray(result['motion_xy'],np.float32)
        scores=np.asarray(result['scores_m'],np.float32)
        if motion.shape!=(7,2) or scores.shape!=(7,) or not np.isfinite(motion).all() or \
                not np.isfinite(scores).all():
            raise ValueError('legacy worker result')
        return motion,scores,self.checkpoint_sha256

    def predict_adopted(self,dem,state):
        if self.mode!='learned14_adopted': raise ValueError('wrong worker mode')
        values=prepare_model_input(dem,state,PATHS,
                                   np.asarray([1]*7+[-1]*7,np.float32),self.norm)
        result=self._request({'op':'predict','arrays':[v.tolist() for v in values]})
        return inverse_target(result['scaled_xy'],self.norm),self.checkpoint_sha256

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"stop"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                self.process.terminate()
                try: self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=2)
        for stream in (self.process.stdin,self.process.stdout,self.process.stderr):
            stream.close()

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
