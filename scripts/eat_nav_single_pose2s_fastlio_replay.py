"""Feed actual simulated LiDAR/IMU to installed FAST-LIO; record real /Odometry.
Run with ROS Jazzy system Python and isolated local DDS domain. No robot commands.
The installed fork does not populate twist: downstream uses causal pose differences.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu,PointCloud2,PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from sensor_msgs_py.point_cloud2 import create_cloud


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binary',type=Path,default=Path('/home/ubuntu/onix/install/fast_lio/lib/fast_lio/fastlio_mapping'));parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/eat_nav/fastlio-sim.yaml');args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='77' or os.environ.get('RMW_IMPLEMENTATION')!='rmw_cyclonedds_cpp':raise ValueError('isolated domain77/cyclonedds required; do not use robot ROS domain')
    output=args.output.resolve();output.mkdir(exist_ok=False,parents=True)
    sensor=args.parent/'lio-sensors.npz';metadata=json.loads((args.parent/'metadata.json').read_text())
    if hashlib.sha256(sensor.read_bytes()).hexdigest()!=metadata['lio_sensor_sha256']:raise ValueError('sensor hash mismatch')
    data=np.load(sensor,allow_pickle=False);tag=metadata['parent']['parent_id'].replace('-','_');prefix='/eatnav_lio_'+tag
    node_name='lio_'+tag
    cmd=[str(args.binary),'--ros-args','--params-file',str(args.config),'-r','__node:='+node_name,'-r','/eatnav_sim/imu:='+prefix+'/imu','-r','/eatnav_sim/lidar:='+prefix+'/lidar','-r','/Odometry:='+prefix+'/Odometry','-r','/tf:='+prefix+'/tf','-r','/tf_static:='+prefix+'/tf_static']
    log=(output/'fastlio.log').open('x');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=output,env=dict(os.environ,OMP_NUM_THREADS='1'))
    rclpy.init();node=rclpy.create_node('recorder_'+tag);imu=node.create_publisher(Imu,prefix+'/imu',100);cloud=node.create_publisher(PointCloud2,prefix+'/lidar',qos_profile_sensor_data)
    records=[];released=-2000000000;start=time.monotonic()
    def callback(m):
        p=m.pose.pose;stamp=m.header.stamp.sec*10**9+m.header.stamp.nanosec-10000000000
        records.append(dict(stamp_ns=stamp,available_ns=released,received_monotonic_ns=time.monotonic_ns(),frame_id=m.header.frame_id,child_frame_id=m.child_frame_id,pose=[p.position.x,p.position.y,p.position.z,p.orientation.w,p.orientation.x,p.orientation.y,p.orientation.z],pose_covariance=list(m.pose.covariance),twist_linear=[m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z],twist_angular=[m.twist.twist.angular.x,m.twist.twist.angular.y,m.twist.twist.angular.z]))
    subscription=node.create_subscription(Odometry,prefix+'/Odometry',callback,100)
    fields=[PointField(name=n,offset=o,datatype=PointField.FLOAT32,count=1) for n,o in [('x',0),('y',4),('z',8),('intensity',12),('time',16)]]+[PointField(name='ring',offset=20,datatype=PointField.UINT16,count=1)]
    dtype=np.dtype({'names':['x','y','z','intensity','time','ring'],'formats':['f4','f4','f4','f4','f4','u2'],'offsets':[0,4,8,12,16,20],'itemsize':24})
    def stamp(t):
        ns=int(t)+10000000000
        return Time(sec=ns//10**9,nanosec=ns%10**9)
    lidar={int(t):i for i,t in enumerate(data['lidar_end_stamp_ns'])}
    error=None
    try:
        end=time.monotonic()+15
        while not imu.get_subscription_count() or not cloud.get_subscription_count():
            if proc.poll() is not None:raise RuntimeError('FAST-LIO exited during discovery '+str(proc.returncode))
            if time.monotonic()>end:raise RuntimeError('FAST-LIO discovery timeout')
            rclpy.spin_once(node,timeout_sec=.01)
        terminal_ns=None if metadata.get('terminal') is None else metadata['terminal']['step']*50000000
        for k,t in enumerate(data['imu_stamp_ns']):
            if terminal_ns is not None and t>terminal_ns:break
            released=int(t);m=Imu();m.header=Header(stamp=stamp(t),frame_id='sim_body')
            m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z=map(float,data['imu_acc'][k]);m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z=map(float,data['imu_gyro'][k]);m.orientation_covariance[0]=-1.;imu.publish(m)
            if int(t) in lidar:
                xyz=data['lidar_points'][lidar[int(t)]];valid=np.isfinite(xyz).all(1)&(np.linalg.norm(xyz,axis=1)>.15)
                pts=np.zeros(int(valid.sum()),dtype=dtype)
                for axis,index in zip(['x','y','z'],range(3)):pts[axis]=xyz[valid,index]
                pts['intensity']=1.;pts['time']=.1;pts['ring']=(np.arange(len(xyz))//512)[valid]
                cloud.publish(create_cloud(Header(stamp=stamp(int(t)-100000000),frame_id='sim_body'),fields,pts,point_step=24))
            until=time.monotonic()+.015
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.001)
            if proc.poll() is not None:raise RuntimeError('FAST-LIO exited '+str(proc.returncode))
        until=time.monotonic()+1
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.005)
        if not records:raise RuntimeError('no actual FAST-LIO /Odometry received')
        (output/'odometry.json').write_text(json.dumps(records,indent=2)+'\n')
        np.savez_compressed(output/'odometry.npz',stamps_ns=np.array([r['stamp_ns'] for r in records],np.int64),received_ns=np.array([r['available_ns'] for r in records],np.int64),poses=np.array([r['pose'] for r in records]))
        report=dict(state='completed',parent_id=metadata['parent']['parent_id'],source='actual_FAST_LIO_process',odom_source_topic='/Odometry',isolated_remapped_topic=prefix+'/Odometry',binary=str(args.binary),binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),sensor_sha256=metadata['lio_sensor_sha256'],odometry_npz_sha256=hashlib.sha256((output/'odometry.npz').read_bytes()).hexdigest(),odometry_json_sha256=hashlib.sha256((output/'odometry.json').read_bytes()).hexdigest(),odometry_count=len(records),wall_seconds=time.monotonic()-start,twist_status='unpopulated in installed fork; use causal pose difference',timing='paced offline replay; available_ns is last sensor time released on callback, no simulated latency equivalence claim',command=cmd)
        (output/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    except BaseException as exc:
        error=repr(exc);(output/'failure.json').write_text(json.dumps(dict(error=error),indent=2)+'\n');raise
    finally:
        proc.send_signal(signal.SIGINT)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        node.destroy_node();rclpy.shutdown();log.close()

if __name__=='__main__':main()
