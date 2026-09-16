import json,time,importlib.util,copy
from pathlib import Path
import numpy as np,torch,yaml
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import CompressedImage
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from visualnav_transformer.deployment.src.robot_interface import ImageHistory,decode_compressed,metric_path
from visualnav_transformer.deployment.src.robot_inference import NoMaDPredictor
root=Path('/tmp/krail-bag-check');out=root/'all_candidates';out.mkdir(exist_ok=True);train=Path('/home/ubuntu/visualnav_docker/visualnav-transformer/train');bag='/media/ubuntu/Extreme SSD/20260803_experiment/2026_08_03-10_21_21'
torch.set_num_threads(1)
reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));reader.set_filter(rosbag2_py.StorageFilter(topics=['/front_camera/image_raw/compressed','/tf','/tf_static']))
buf=Buffer(cache_time=Duration(seconds=120));frames=[];tf_count=0
while reader.has_next():
 topic,data,record=reader.read_next()
 if topic.endswith('compressed'):
  msg=deserialize_message(data,CompressedImage);stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec;frames.append((stamp,msg,record))
 else:
  for tr in deserialize_message(data,TFMessage).transforms:
   (buf.set_transform_static if topic=='/tf_static' else buf.set_transform)(tr,'bag');tf_count+=1
hist=ImageHistory(4,.25);samples=[];last=-1;resets=0
for stamp,msg,record in frames:
 resets+=hist.add(decode_compressed(msg.data),stamp)
 snap=hist.snapshot()
 if snap and stamp-last>=4e9:
  try:tr=buf.lookup_transform('map','chassis/body',Time(nanoseconds=snap[1]))
  except Exception:continue
  samples.append((snap,tr));last=stamp
# Last decoded observation is the fixed goal; compare beginning/middle/end.
goal=decode_compressed(frames[-1][1].data);goal.save(out/'goal.jpg')
cfg=yaml.safe_load((root/'source/config/krail_baseline_model.yaml').read_text())
pred=NoMaDPredictor(model_config=cfg,checkpoint='/home/ubuntu/visualnav-transformer-ros2/model_weights/krail_baseline_20260904_latest.pth',goal_image=str(out/'goal.jpg'),device='cpu',num_samples=8)

# Capture the full output before the existing predictor selects candidate zero.
import visualnav_transformer.deployment.src.robot_inference as inference
original_get_action=inference.get_action
captured=[]
def capture_action(*args,**kwargs):
    value=original_get_action(*args,**kwargs)
    captured.append(value.detach().cpu().numpy().copy())
    return value
inference.get_action=capture_action
previous=json.loads((root/'results/results.json').read_text())
old=[r for r in previous['results'] if r['model']=='baseline']
rows=[]
for index,((images,stamp),tr) in enumerate(samples):
    captured.clear();torch.manual_seed(42);result=pred(images,0)
    candidates=captured[0]
    assert candidates.shape==(8,8,2)
    np.testing.assert_array_equal(candidates[0],result.points)
    np.testing.assert_array_equal(candidates[0]*.05,np.asarray(old[index]['local_xy_m'],dtype=np.float32))
    paths=[]
    for candidate in candidates:
        path=metric_path(candidate,.05,tr,'map',stamp)
        paths.append([[p.pose.position.x,p.pose.position.y] for p in path.poses])
    rows.append(dict(index=index,time_sec=(stamp-frames[0][0])/1e9,local_xy_m=(candidates*.05).tolist(),map_xy=paths))
report=dict(bag=bag,model='baseline',seed=42,num_samples=8,device='cpu',results=rows,recorded_positions=previous['recorded_positions'])
(out/'results.json').write_text(json.dumps(report,indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
xy=np.asarray(report['recorded_positions'])
colors=plt.get_cmap('tab20')
fig,ax=plt.subplots(figsize=(12,7))
ax.plot(xy[:,0],xy[:,1],color='0.65',lw=2,zorder=0)
for row in rows:
    color=colors(row['index']%20)
    for j in range(7,-1,-1):
        p=np.asarray(row['map_xy'][j]);ax.plot(p[:,0],p[:,1],color=color,alpha=1 if j==0 else .45,lw=2.2 if j==0 else .8)
    ax.scatter(p[0,0],p[0,1],s=18,color=color);ax.annotate(str(row['index']),(p[0,0],p[0,1]),xytext=(4,5),textcoords='offset points',fontsize=9)
ax.set(title='Baseline: all 8 candidates at each of 11 observations',xlabel='map x [m]',ylabel='map y [m]');ax.axis('equal');ax.grid(alpha=.25)
ax.legend(handles=[Line2D([0],[0],color='0.65',lw=2,label='Recorded robot trajectory'),Line2D([0],[0],color='tab:blue',lw=2.2,label='Candidate 0 (selected)'),Line2D([0],[0],color='tab:blue',alpha=.45,lw=.8,label='Candidates 1-7')])
fig.tight_layout();fig.savefig(out/'baseline_all_candidates.png',dpi=180)
fig,axes=plt.subplots(3,4,figsize=(16,12))
for ax,row in zip(axes.flat,rows):
    paths=np.array(row['map_xy']);origin=paths[0,0]
    ax.plot(xy[:,0],xy[:,1],color='0.8',lw=1)
    for j in range(7,-1,-1):
        p=paths[j];ax.plot(p[:,0],p[:,1],color='tab:red' if j==0 else 'tab:blue',alpha=1 if j==0 else .5,lw=2 if j==0 else 1,marker='.',ms=3)
    ax.scatter(*origin,c='black',s=20,zorder=5)
    radius=max(.25,float(np.max(np.abs(paths-origin)))*1.2)
    ax.set_xlim(origin[0]-radius,origin[0]+radius);ax.set_ylim(origin[1]-radius,origin[1]+radius)
    ax.set_aspect('equal');ax.grid(alpha=.25);ax.set_title(f"Observation {row['index']} | {row['time_sec']:.1f} s");ax.set_xlabel('map x [m]');ax.set_ylabel('map y [m]')
axes.flat[-1].axis('off');axes.flat[-1].legend(handles=[Line2D([0],[0],color='tab:red',lw=2,label='Candidate 0 (selected)'),Line2D([0],[0],color='tab:blue',alpha=.5,label='Candidates 1-7'),Line2D([0],[0],color='0.8',label='Recorded trajectory')],loc='center')
fig.suptitle('Baseline: 8 candidates per observation (individual map views)');fig.tight_layout();fig.savefig(out/'baseline_candidates_detail.png',dpi=160)
print('PASS: 11 observations x 8 candidates; candidate zero exactly matches previous results',flush=True)
