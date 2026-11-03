"""Legacy nominal P1 geometry, with no recovery, boost, or risk gates."""
import math
import numpy as np


def wrap(x):return math.atan2(math.sin(x),math.cos(x))


class NominalFollower:
    def __init__(self,path):
        self.path=np.asarray(path,float)
        if self.path.shape!=(8,2) or not np.isfinite(self.path).all():raise ValueError('finite bank path required')
        self.closest=0;self.goal=False;self.direction=None

    def tangent(self,index):
        for forward in [True,False]:
            total=np.zeros(2);length=0.
            iterator=range(index,len(self.path)-1) if forward else range(index,0,-1)
            for i in iterator:
                delta=self.path[i+1]-self.path[i] if forward else self.path[i]-self.path[i-1]
                n=np.linalg.norm(delta)
                if n<1e-6:continue
                take=min(n,.5-length);total+=delta/n*take;length+=take
                if length>=.5:break
            if np.linalg.norm(total)>=1e-6:return math.atan2(total[1],total[0])
        raise ValueError('no tangent')

    def step(self,x,y,yaw):
        if self.goal:return (0.,0.,0.)
        robot=np.array([x,y]);best=None
        for i in range(self.closest,7):
            delta=self.path[i+1]-self.path[i];n=np.linalg.norm(delta)
            ratio=float(np.clip((robot-self.path[i])@delta/(n*n),0,1));point=self.path[i]+ratio*delta
            remaining=(1-ratio)*n+np.linalg.norm(np.diff(self.path[i+1:],axis=0),axis=1).sum()
            key=(float(np.linalg.norm(robot-point)),-remaining)
            if best is None or key<best[0]:best=(key,i,ratio,point,remaining)
        _,i,ratio,point,remaining=best;self.closest=i
        err=abs(wrap(self.tangent(i)-yaw))
        if self.direction is None:self.direction=1 if err<=math.pi/2 else -1
        elif err<=math.radians(80):self.direction=1
        elif err>=math.radians(100):self.direction=-1
        line_yaw=self.tangent(7)
        extension=remaining<=.5
        if np.linalg.norm(robot-self.path[-1])<.2 or (extension and (robot-self.path[-1])@np.array([math.cos(line_yaw),math.sin(line_yaw)])>=0):
            self.goal=True;return (0.,0.,0.)
        if extension:
            unit=np.array([math.cos(line_yaw),math.sin(line_yaw)])
            target=self.path[-1]+(((robot-self.path[-1])@unit)+.5)*unit
        else:
            current=point;look=.5;target=self.path[-1]
            for j in range(i+1,8):
                delta=self.path[j]-current;n=np.linalg.norm(delta)
                if look<=n:target=current+delta*(look/n);break
                look-=n;current=self.path[j]
        heading=wrap(math.atan2(target[1]-y,target[0]-x)-(yaw+(math.pi if self.direction<0 else 0)))
        v=self.direction*.3*max(0,1-abs(heading)/math.pi) if abs(heading)<.6 else 0.
        return (v,0.,float(np.clip(.5*heading,-1.6,1.6)))
