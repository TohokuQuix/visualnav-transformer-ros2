"""CPU-only recorded-state replay contract and bounded RGB->MP4 stream."""
from __future__ import annotations
import json,hashlib,subprocess,shutil
from pathlib import Path
import numpy as np

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_trace(path):
 with np.load(path,allow_pickle=False) as z:
  names=('root','joint_position','joint_velocity','time_s')
  if not set(names)<=set(z.files):raise ValueError('trace requires '+','.join(names))
  data={k:np.asarray(z[k]) for k in names}
 t=data['time_s'];r=data['root'];q=data['joint_position'];v=data['joint_velocity']
 if t.ndim!=1 or len(t)<1:raise ValueError('time_s must be nonempty[T]')
 if r.shape!=(len(t),13) or q.ndim!=2 or len(q)!=len(t) or v.shape!=q.shape:raise ValueError('expected root[T,13],joint_position/velocity[T,J]')
 if not all(np.isfinite(a).all() for a in data.values()):raise ValueError('nonfinite recorded state')
 if len(t)>1 and not np.allclose(np.diff(t),.05,rtol=0,atol=2e-5):raise ValueError('recorded states must be strictly increasing20Hz (.05s), with no gaps')
 if not np.allclose(np.linalg.norm(r[:,3:7],axis=1),1.,rtol=0,atol=1e-3):raise ValueError('root quaternion must be normalized wxyz')
 return data

def replay_indices(time_s,fps):
 t=np.asarray(time_s,dtype=float)
 if fps not in (5,10):raise ValueError('fps must be5or10')
 if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or (len(t)>1 and np.any(np.diff(t)<=0)):raise ValueError('invalid time vector')
 duration=t[-1]-t[0];grid=t[0]+np.minimum(np.arange(int(np.ceil(duration*fps-1e-8))+1)/fps,duration)
 #20microsecond tolerance recognizes float32 representations of exact20Hz stamps; far below a50ms acquisition step.
 idx=np.searchsorted(t,grid+2e-5,side='right')-1;idx=np.maximum(idx,0)
 if idx[-1]!=len(t)-1:idx=np.r_[idx,len(t)-1]
 return idx.astype(np.int64)

def joint_permutation(recorded_names,target_names,joint_count):
 target=list(target_names)
 if len(target)!=joint_count or len(set(target))!=len(target):raise ValueError('joint count/runtime names mismatch')
 if recorded_names is None:return np.arange(joint_count,dtype=np.int64)
 source=list(recorded_names)
 if len(source)!=joint_count or len(set(source))!=len(source) or set(source)!=set(target):raise ValueError('recorded/runtime joint identities mismatch')
 return np.array([source.index(n) for n in target],dtype=np.int64)

class VideoStream:
 """At most one RGB frame and one encoded pipe buffer; no image directory."""
 def __init__(self,path,width,height,fps,ffmpeg='ffmpeg'):
  self.path=Path(path);self.width=width;self.height=height;self.fps=fps;self.frames=0;self.process=None;self.log=None
  if self.path.exists():raise FileExistsError(self.path)
  if width<2 or height<2 or width%2 or height%2 or fps not in (5,10):raise ValueError('even image dimensions and5or10fps required')
  if not shutil.which(ffmpeg):raise FileNotFoundError(ffmpeg)
  self.path.parent.mkdir(parents=True,exist_ok=True);self.log=self.path.with_suffix('.ffmpeg.log').open('wb')
  self.command=[ffmpeg,'-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-metadata','comment=Recorded-state replay; no physics, Actor or planner integration during frames',str(self.path)]
  self.process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
 def write(self,rgb,time_s,trial):
  from PIL import Image,ImageDraw,ImageFont
  a=np.asarray(rgb)
  if a.shape!=(self.height,self.width,3) or a.dtype!=np.uint8:raise ValueError('expected uint8 RGB[height,width,3]')
  image=Image.fromarray(a,'RGB');draw=ImageDraw.Draw(image)
  try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',max(10,self.width//80))
  except OSError:font=ImageFont.load_default()
  text=f'RECORDED-STATE REPLAY | {trial} | recorded t={time_s:.2f}s\nSaved root + joints; no physics / Actor / planner rerun'
  box=draw.multiline_textbbox((10,8),text,font=font,spacing=4);draw.rectangle((0,0,self.width,box[3]+8),fill=(15,15,15));draw.multiline_text((10,8),text,font=font,fill=(255,255,255),spacing=4)
  try:self.process.stdin.write(np.asarray(image).tobytes())
  except BrokenPipeError as e:raise RuntimeError('FFmpeg pipe closed; inspect '+str(self.path.with_suffix('.ffmpeg.log'))) from e
  self.frames+=1
 def close(self):
  if self.process is None:return
  self.process.stdin.close();code=self.process.wait(timeout=120);self.log.close();self.process=None
  if code:raise RuntimeError('FFmpeg failed with exit'+str(code))
  if not self.frames:raise RuntimeError('empty video')
 def __enter__(self):return self
 def __exit__(self,typ,value,tb):
  if typ is not None:
   if self.process is not None:
    self.process.stdin.close();self.process.terminate();self.process.wait(timeout=10);self.log.close();self.process=None
   return False
  self.close()

def read_frame(stream,byte_count):
 buffer=bytearray(byte_count);view=memoryview(buffer);offset=0
 while offset<byte_count:
  size=stream.readinto(view[offset:])
  if not size:raise EOFError(f'truncated raw RGB frame: received{offset}/{byte_count} bytes')
  offset+=size
 return buffer

def main():
 import argparse,sys
 p=argparse.ArgumentParser(description='Encode clean streamed Isaac RGB as labelled recorded-state replay; CPU FFmpeg, no frame dumps.')
 p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--fps',type=int,choices=(5,10),default=5);p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
 p.add_argument('--trial-id',default='selected-trial');p.add_argument('--raw-file',default='-');p.add_argument('--ffmpeg',default='ffmpeg')
 args=p.parse_args();data=load_trace(args.trace);idx=replay_indices(data['time_s'],args.fps);stream=sys.stdin.buffer if args.raw_file=='-' else open(args.raw_file,'rb')
 try:
  with VideoStream(args.output,args.width,args.height,args.fps,args.ffmpeg) as video:
   for source_index in idx:
    frame=read_frame(stream,args.width*args.height*3);rgb=np.frombuffer(frame,np.uint8).reshape(args.height,args.width,3);video.write(rgb,float(data['time_s'][source_index]),args.trial_id)
   if stream.read(1):raise ValueError('raw stream has unexpected additional frame data')
 finally:
  if stream is not sys.stdin.buffer:stream.close()
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height,r_frame_rate,duration','-of','json',str(args.output)]))
 if int(probe['streams'][0]['nb_frames'])!=len(idx):raise RuntimeError('encoded frame count mismatch')
 summary=dict(mode='recorded-state replay, not fresh physics',trace_sha256=sha256(args.trace),video_sha256=sha256(args.output),encoder_sha256=sha256(__file__),fps=args.fps,frame_count=len(idx),recorded_duration_s=float(data['time_s'][-1]-data['time_s'][0]),encoded_duration_s=len(idx)/args.fps,terminal_frame_included=bool(idx[-1]==len(data['time_s'])-1),frame_source_indices=idx.tolist(),ffprobe=probe,encoder_command=video.command,trial_id=args.trial_id)
 args.output.with_suffix('.video.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ('mode','frame_count','encoded_duration_s','video_sha256')}))
if __name__=='__main__':main()
