import rclpy,time,json,math
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav_msgs.msg import Path
rclpy.init();n=Node('baseline_path_monitor',parameter_overrides=[Parameter('use_sim_time',value=True)])
rows=[]
def cb(m):
 age=(n.get_clock().now().nanoseconds-m.header.stamp.sec*10**9-m.header.stamp.nanosec)/1e9
 rows.append(dict(count=len(m.poses),frame=m.header.frame_id,age_sec=age,stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,finite=all(math.isfinite(v) for p in m.poses for v in [p.pose.position.x,p.pose.position.y])))
n.create_subscription(Path,'/teaching_playback/path',cb,QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
end=time.monotonic()+55
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
valid=[r for r in rows if r['count']]
report=dict(received=len(rows),nonempty=len(valid),empty=len(rows)-len(valid),point_counts=sorted(set(r['count'] for r in valid)),max_age_sec=max([r['age_sec'] for r in valid],default=None),rows=rows)
open('/tmp/krail-online/monitor.json','w').write(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}));n.destroy_node();rclpy.shutdown()
