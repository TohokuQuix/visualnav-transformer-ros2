"""URDF link-frame FK. No simulator, policy, or future data dependency."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

def matrix(pose):
 p=np.asarray(pose,float)
 if p.shape!=(7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1)>.001:raise ValueError('finite xyz/wxyz pose required')
 t=np.eye(4);t[:3,:3]=Rotation.from_quat(p[[4,5,6,3]]).as_matrix();t[:3,3]=p[:3];return t

class RobotFK:
 def __init__(self,path):
  root=ET.parse(path).getroot();links={x.attrib['name'] for x in root.findall('link')};children=set();self.joints=[]
  for j in root.findall('joint'):
   parent=j.find('parent').attrib['link'];child=j.find('child').attrib['link'];children.add(child);origin=j.find('origin');t=np.eye(4)
   if origin is not None:
    t[:3,3]=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ');t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
   axis=j.find('axis');a=np.fromstring(axis.attrib.get('xyz','1 0 0'),sep=' ') if axis is not None else np.array([1.,0,0]);a=a/np.linalg.norm(a)
   self.joints.append((j.attrib['name'],j.attrib['type'],parent,child,t,a))
  roots=links-children
  if len(roots)!=1:raise ValueError('single-root tree required')
  self.root=roots.pop()
 def frames(self,root_pose,names,q):
  q=np.asarray(q,float)
  if q.shape!=(len(names),) or len(set(names))!=len(names) or not np.isfinite(q).all():raise ValueError('joint vector')
  values=dict(zip(names,q));out={self.root:matrix(root_pose)};pending=self.joints.copy()
  while pending:
   rest=[]
   for name,kind,parent,child,t,axis in pending:
    if parent not in out:rest.append((name,kind,parent,child,t,axis));continue
    m=np.eye(4)
    if kind!='fixed':
     if name not in values:raise ValueError('missing joint '+name)
     if kind in ('revolute','continuous'):m[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix()
     elif kind=='prismatic':m[:3,3]=axis*values[name]
     else:raise ValueError('unsupported joint '+kind)
    out[child]=out[parent]@t@m
   if len(rest)==len(pending):raise ValueError('cyclic/disconnected joints')
   pending=rest
  return out
