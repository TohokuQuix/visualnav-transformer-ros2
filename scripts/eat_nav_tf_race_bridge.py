"""Logging-only wrapper of the frozen simulation-clock bridge."""
from __future__ import annotations
import argparse,collections,json,math,threading,time
from pathlib import Path
import rclpy
from eat_nav_sim_clock_bridge import SimClockNode,Server as BaseServer,stamp_ns
from visualnav_transformer.eat_nav.tf_race_evidence import write_first_conflict

ROOT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')

def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(path)

class TraceNode(SimClockNode):
 def __init__(self,branches):
  super().__init__(branches)
  self.trace_lock=threading.Lock();self.trace_dir=None;self.files={};self.publish_seq=collections.Counter()
  self.first_conflict=None;self.history=collections.deque(maxlen=1000)
  self.path_sha=json.loads((ROOT/'tf-race-source-audit.json').read_text())['sources']['nominal_path']['sha256']

 def start_repeat(self,repeat):
  directory=ROOT/f'repeat-{repeat:02d}';directory.mkdir(parents=True,exist_ok=True)
  with self.trace_lock:
   for f in self.files.values():f.close()
   self.trace_dir=directory
   self.files={kind:(directory/f'{kind}-trace.jsonl').open('x',buffering=1)
    for kind in ('clock-publish','tf-publish','path-publish','bridge-command')}
   self.first_conflict=None;self.history.clear();self.publish_seq.clear()

 def event(self,kind,data):
  row={'kind':kind,'steady_ns':time.monotonic_ns(),'system_ns':time.time_ns(),**data}
  with self.trace_lock:
   if self.trace_dir is not None:
    self.files[kind].write(json.dumps(row,separators=(',',':'))+'\n');self.files[kind].flush()
   self.history.append(row)
  return row

 def clock(self,ns,reset=False):
  before=time.monotonic_ns();super().clock(ns,reset=reset)
  self.publish_seq['clock']+=1
  self.event('clock-publish',{'sequence':self.publish_seq['clock'],'stamp_ns':ns,
   'publish_lower_steady_ns':before,'publish_upper_steady_ns':time.monotonic_ns(),
   'clock_now_ns':self.get_clock().now().nanoseconds,'reset':reset})

 def poses(self,poses,ns):
  before=time.monotonic_ns();super().poses(poses,ns)
  for i,pose in enumerate(poses):
   self.publish_seq[f'tf-{i}']+=1
   self.event('tf-publish',{'branch':i,'sequence':self.publish_seq[f'tf-{i}'],
    'tf_stamp_ns':ns,'pose':pose,'publish_lower_steady_ns':before,
    'publish_steady_ns':self.tf_published_steady_ns,'publish_upper_steady_ns':time.monotonic_ns()})

 def path(self,paths,ns):
  before=time.monotonic_ns();super().path(paths,ns)
  for i,xy in enumerate(paths):
   self.publish_seq[f'path-{i}']+=1
   self.event('path-publish',{'branch':i,'sequence':self.publish_seq[f'path-{i}'],
    'path_stamp_ns':ns,'path_sha256':self.path_sha,'points':xy,
    'publish_lower_steady_ns':before,'publish_steady_ns':self.path_published_steady_ns,
    'publish_upper_steady_ns':time.monotonic_ns()})

 def on_command(self,i,msg):
  stamp=stamp_ns(msg.header.stamp)
  cmd=[float(msg.twist.linear.x),float(msg.twist.linear.y),float(msg.twist.angular.z)]
  with self.cv:
   error_count=len(self.errors)
   prior=list(self.protocol.commands.get(i,[]))
   receive_seq=len(self.events)+1
  super().on_command(i,msg)  # Frozen acceptance/selection is untouched.
  with self.cv:
   accepted=self.events[-1]['accepted'];reason=self.errors[-1] if len(self.errors)>error_count else None
   recent=list(self.events);step_stamp=self.protocol.stamp_ns
  row=self.event('bridge-command',{'branch':i,'receive_seq':receive_seq,'header_stamp_ns':stamp,
   'logical_stamp_ns':step_stamp,'command':cmd,'accepted':accepted,'reject_reason':reason,
   'same_stamp_commands':[list(c) for c in prior]+[cmd]})
  if reason and reason.startswith('different commands at same stamp'):
   with self.trace_lock:
    if self.first_conflict is None:
     self.first_conflict=write_first_conflict(ROOT/'same-stamp-command-conflict-first.json',
      branch=i,stamp_ns=stamp,previous=prior,incoming=cmd,incoming_event=row,preceding=recent,
      publishers=list(self.history))

class TraceServer(BaseServer):
 def __init__(self,args):
  self.args=args;self.log_dir=args.diagnostic_dir;self.log_dir.mkdir(parents=True,exist_ok=True)
  rclpy.init();self.node=TraceNode(args.branches)
  self.spin=threading.Thread(target=rclpy.spin,args=(self.node,),daemon=True);self.spin.start()
  self.followers=[];self.logs=[];self.reset_number=0

 def stop_followers(self):
  for p in self.followers:p.terminate()
  for p in self.followers:
   try:p.wait(timeout=5)
   except Exception:p.kill();p.wait()
  for f in self.logs:f.flush();f.close()
  self.followers=[];self.logs=[]
  with self.node.trace_lock:
   self.node.trace_dir=None
   for f in self.node.files.values():f.flush();f.close()
   self.node.files={}

 def handle(self,request):
  if request['action']=='reset':
   repeat=int(request['repeat_id']);self.node.start_repeat(repeat)
   out=super().handle(request)
   atomic(ROOT/f'repeat-{repeat:02d}'/'process-map.json',{
    'repeat':repeat,'branch_pid':{str(i):pid for i,pid in enumerate(out['readiness']['follower_alive'])} if False else
      {str(i):p.pid for i,p in enumerate(self.followers)},
    'branch_frame_id':{str(i):f'branch_{i}/chassis/body' for i in range(self.node.branches)},
    'use_sim_time':True,'logical_t0_ns':request['stamp_ns']})
   return out
  if request['action']=='diagnostic_stop':
   self.stop_followers();return {'stopped':True,'first_conflict':self.node.first_conflict}
  try:return super().handle(request)
  except BaseException:
   conflict=self.node.first_conflict
   if conflict:
    n=self.node;branch=conflict['branch'];initial_count=sum(x['branch']==branch for x in n.events)
    n.cv_wait(lambda:sum(x['branch']==branch for x in n.events)>=initial_count+20,2.)
    self.stop_followers()
   raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=47819)
 p.add_argument('--branches',type=int,default=8);p.add_argument('--follower',type=Path,required=True)
 p.add_argument('--params',type=Path,required=True);p.add_argument('--diagnostic-dir',type=Path,required=True)
 TraceServer(p.parse_args()).serve()
if __name__=='__main__':main()
