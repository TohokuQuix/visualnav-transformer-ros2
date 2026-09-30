"""Simulation-only ROS bridge. Only the simulator client advances /clock."""
from __future__ import annotations
import argparse,json,math,socket,subprocess,threading,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import TransformStamped,TwistStamped,PoseStamped
from nav_msgs.msg import Path as RosPath
from std_msgs.msg import String
from tf2_ros import Buffer,TransformBroadcaster,TransformListener
from visualnav_transformer.eat_nav.sim_clock_protocol import ClockProtocol,STEP_NS


def stamp_ns(msg):return int(msg.sec)*1_000_000_000+int(msg.nanosec)


def make_stamp(ns):
    from builtin_interfaces.msg import Time
    return Time(sec=ns//1_000_000_000,nanosec=ns%1_000_000_000)


class SimClockNode(Node):
    def __init__(self,branches):
        super().__init__('eat_nav_sim_clock_bridge')
        self.set_parameters([Parameter('use_sim_time',Parameter.Type.BOOL,True)])
        self.branches=branches;self.cv=threading.Condition();self.protocol=ClockProtocol()
        self.sequence=[0]*branches;self.commands=[None]*branches;self.command_count=[0]*branches
        self.status_count=[0]*branches;self.rejected=[0]*branches;self.errors=[]
        self.events=[];self.active_stamp_ns=None;self.tf_published_steady_ns=None;self.path_published_steady_ns=None
        clock_qos=QoSProfile(depth=10,durability=DurabilityPolicy.VOLATILE,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.clock_pub=self.create_publisher(Clock,'/clock',clock_qos)
        self.clock_sub=self.create_subscription(Clock,'/clock',self.on_clock,clock_qos)
        path_qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.paths=[self.create_publisher(RosPath,f'/phase_c/branch_{i}/path',path_qos) for i in range(branches)]
        self.subs=[self.create_subscription(TwistStamped,f'/phase_c/branch_{i}/cmd_vel',lambda msg,i=i:self.on_command(i,msg),10) for i in range(branches)]
        self.status_subs=[self.create_subscription(String,f'/eat_nav_follower_{i}/status',lambda msg,i=i:self.on_status(i,msg),10) for i in range(branches)]
        self.tf=TransformBroadcaster(self);self.tf_buffer=Buffer();self.tf_listener=TransformListener(self.tf_buffer,self,spin_thread=False)

    def on_clock(self,msg):
        with self.cv:self.cv.notify_all()

    def on_status(self,i,msg):
        with self.cv:self.status_count[i]+=1;self.cv.notify_all()

    def on_command(self,i,msg):
        stamp=stamp_ns(msg.header.stamp)
        command=(float(msg.twist.linear.x),float(msg.twist.linear.y),float(msg.twist.angular.z))
        steady=time.monotonic_ns()
        with self.cv:
            try:
                accepted=(steady>=(self.tf_published_steady_ns or 2**63) and
                    steady>=(self.path_published_steady_ns or 2**63) and
                    self.protocol.accept(i,stamp,command))
            except ValueError as exc:
                self.errors.append(str(exc));accepted=False
            event={'branch':i,'stamp_ns':stamp,'steady_ns':steady,'command':command,'accepted':accepted,
                   'tf_stamp_ns':self.protocol.tf_stamp_ns,'path_stamp_ns':self.protocol.path_stamp_ns}
            self.events.append(event)
            if accepted:
                self.commands[i]=list(command);self.sequence[i]+=1;self.command_count[i]+=1
            else:self.rejected[i]+=1
            self.cv.notify_all()

    def clock(self,ns,reset=False):
        with self.cv:
            self.protocol.advance(ns,reset=reset);self.active_stamp_ns=ns
            self.commands=[None]*self.branches;self.command_count=[0]*self.branches
            self.tf_published_steady_ns=None
            if reset:self.path_published_steady_ns=None
        self.clock_pub.publish(Clock(clock=make_stamp(ns)))
        if not self.cv_wait(lambda:self.get_clock().now().nanoseconds==ns,2.):
            raise RuntimeError(f'bridge clock mismatch target={ns} actual={self.get_clock().now().nanoseconds}')

    def cv_wait(self,predicate,timeout):
        deadline=time.monotonic()+timeout
        with self.cv:
            while not predicate():
                remaining=deadline-time.monotonic()
                if remaining<=0:return False
                self.cv.wait(timeout=remaining)
            return True

    def poses(self,poses,ns):
        if len(poses)!=self.branches or self.get_clock().now().nanoseconds!=ns:raise ValueError('pose/clock mismatch')
        transforms=[]
        for i,(x,y,yaw) in enumerate(poses):
            msg=TransformStamped();msg.header.stamp=make_stamp(ns);msg.header.frame_id='map';msg.child_frame_id=f'branch_{i}/chassis/body'
            msg.transform.translation.x=float(x);msg.transform.translation.y=float(y)
            msg.transform.rotation.z=math.sin(float(yaw)/2);msg.transform.rotation.w=math.cos(float(yaw)/2)
            transforms.append(msg)
        with self.cv:self.protocol.set_tf(ns);self.tf_published_steady_ns=time.monotonic_ns()
        self.tf.sendTransform(transforms)

    def path(self,paths,ns):
        if len(paths)!=self.branches or self.get_clock().now().nanoseconds!=ns:raise ValueError('Path/clock mismatch')
        with self.cv:self.protocol.set_path(ns);self.path_published_steady_ns=time.monotonic_ns()
        for i,xy in enumerate(paths):
            if len(xy)<2:raise ValueError('Path needs two poses')
            msg=RosPath();msg.header.frame_id='map';msg.header.stamp=make_stamp(ns)
            for x,y in xy:
                pose=PoseStamped();pose.header=msg.header;pose.pose.position.x=float(x);pose.pose.position.y=float(y);pose.pose.orientation.w=1.
                msg.poses.append(pose)
            self.paths[i].publish(msg)

    def empty_paths(self,ns):
        for pub in self.paths:
            msg=RosPath();msg.header.frame_id='map';msg.header.stamp=make_stamp(ns);pub.publish(msg)


class Server:
    def __init__(self,args):
        self.args=args;self.log_dir=args.diagnostic_dir;self.log_dir.mkdir(parents=True,exist_ok=True)
        rclpy.init();self.node=SimClockNode(args.branches)
        self.spin=threading.Thread(target=rclpy.spin,args=(self.node,),daemon=True);self.spin.start()
        self.followers=[];self.logs=[];self.reset_number=0

    def snapshot(self):
        n=self.node
        with n.cv:
            return {'clock_now_ns':n.get_clock().now().nanoseconds,'clock_publishers':n.count_publishers('/clock'),
                'use_sim_time':n.get_parameter('use_sim_time').value,'status_counts':list(n.status_count),
                'command_sequence':list(n.sequence),'callback_count':list(n.command_count),
                'rejected':list(n.rejected),'protocol_errors':list(n.errors),
                'follower_alive':[p.poll() is None for p in self.followers],
                'path_subscribers':[p.get_subscription_count() for p in n.paths],
                'tf_ready':[bool(n.tf_buffer.can_transform('map',f'branch_{i}/chassis/body',rclpy.time.Time())) for i in range(n.branches)]}

    def launch(self):
        n=self.node
        for i in range(n.branches):
            command=[str(self.args.follower),'--ros-args','--params-file',str(self.args.params),
                '-r',f'__node:=eat_nav_follower_{i}','-p','use_sim_time:=true',
                '-p',f'path_topic:=/phase_c/branch_{i}/path','-p',f'cmd_vel_topic:=/phase_c/branch_{i}/cmd_vel',
                '-p',f'robot_frame:=branch_{i}/chassis/body']
            log=(self.log_dir/f'reset-{self.reset_number:04d}-branch-{i}.log').open('x')
            self.logs.append(log);self.followers.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))

    def reset(self,request):
        for p in self.followers:p.terminate()
        for p in self.followers:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in self.logs:f.close()
        self.followers=[];self.logs=[];self.reset_number+=1
        n=self.node;ns=int(request['stamp_ns']);poses=request['poses'];paths=request['paths']
        with n.cv:
            n.sequence=[0]*n.branches;n.commands=[None]*n.branches;n.status_count=[0]*n.branches
            n.rejected=[0]*n.branches;n.errors=[];n.events=[]
        n.clock(ns,reset=True);n.empty_paths(ns);n.poses(poses,ns);self.launch()
        deadline=time.monotonic()+10
        while True:
            snap=self.snapshot()
            if not all(snap['follower_alive']):raise RuntimeError('follower exited before Path '+repr(snap))
            if (all(c>=2 for c in snap['status_counts']) and all(x>0 for x in snap['path_subscribers'])
                and all(snap['tf_ready'])):break
            if time.monotonic()>deadline:raise TimeoutError('pre-Path readiness '+repr(snap))
            # Re-publish the *same* logical stamp. Wall timers can run, logical time cannot.
            n.clock_pub.publish(Clock(clock=make_stamp(ns)));n.poses(poses,ns)
            n.cv_wait(lambda:any(c>v for c,v in zip(n.status_count,snap['status_counts'])),.3)
        n.path(paths,ns)
        data=self.wait_command(ns,timeout=4.)
        data['readiness']=snap;data['reset_number']=self.reset_number
        return data

    def wait_command(self,ns,timeout):
        n=self.node
        if not n.cv_wait(lambda:n.protocol.ready(n.branches) or bool(n.errors),timeout):
            raise TimeoutError('fresh command timeout '+repr(self.snapshot()))
        with n.cv:
            if n.errors:raise RuntimeError('clock protocol '+repr(n.errors))
            if n.get_clock().now().nanoseconds!=ns:raise RuntimeError('clock drift')
            commands=[list(x) for x in n.commands]
            events=[x for x in n.events if x['stamp_ns']==ns]
            return {'fresh':True,'commands':commands,'sequence':list(n.sequence),'stamp_ns':ns,
                'callback_count':list(n.command_count),'events':events,'rejected':list(n.rejected),
                'tf_stamp_ns':n.protocol.tf_stamp_ns,'path_stamp_ns':n.protocol.path_stamp_ns,
                'clock_now_ns':n.get_clock().now().nanoseconds}

    def handle(self,req):
        action=req['action']
        if action=='reset':return self.reset(req)
        if action=='step':
            ns=int(req['stamp_ns']);n=self.node;n.clock(ns);n.poses(req['poses'],ns)
            return self.wait_command(ns,float(req.get('timeout_sec',3.)))
        if action=='status':return self.snapshot()
        if action=='finish':return {'snapshot':self.snapshot(),'events':list(self.node.events)}
        raise ValueError(action)

    def serve(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind((self.args.host,self.args.port));sock.listen(1)
            print(json.dumps({'stage':'listening','port':self.args.port,'use_sim_time':True}),flush=True)
            while True:
                conn,_=sock.accept()
                with conn,conn.makefile('rwb') as stream:
                    for line in stream:
                        try:response={'ok':True,**self.handle(json.loads(line))}
                        except BaseException as exc:response={'ok':False,'error':repr(exc)}
                        stream.write((json.dumps(response)+'\n').encode());stream.flush()


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=47819)
    p.add_argument('--branches',type=int,default=8);p.add_argument('--follower',type=Path,required=True)
    p.add_argument('--params',type=Path,required=True);p.add_argument('--diagnostic-dir',type=Path,required=True)
    Server(p.parse_args()).serve()
if __name__=='__main__':main()
