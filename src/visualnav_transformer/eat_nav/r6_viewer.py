"""Optional Isaac GUI overlay, imported only in explicit GUI reproductions."""
import math
import numpy as np


class R6Viewer:
 def __init__(self,field,level,mode):
  import omni.ui as ui
  self.mode=mode
  self.window=ui.Window('R6 two-second predictor | live simulator',width=600,height=250)
  with self.window.frame:
   with ui.VStack():
    ui.Label(f'LIVE {field} level{level} | {mode}',height=30)
    self.status=ui.Label('Settling; no replayed success claims',word_wrap=True,height=160)
 def update(self,bare,root,goal,bank,decision,step):
  from pxr import Gf,UsdGeom
  from .fall_stall_targets import root_yaw
  yaw=root_yaw(np.asarray(root).tolist());c,s=math.cos(yaw),math.sin(yaw);rotation=np.array([[c,s],[-s,c]])
  selected=decision['selected_index'];excluded=decision.get('excluded',[False]*14)
  for k,path in enumerate(bank):
   world=path@rotation+root[:2]
   curve=UsdGeom.BasisCurves.Define(bare.sim.stage,f'/World/R6DesiredCandidate{k}')
   curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([len(path)]);curve.CreatePointsAttr([Gf.Vec3f(float(x),float(y),float(root[2]+.15)) for x,y in world]);curve.CreateWidthsAttr([.025 if k==selected else .009]);curve.CreateDisplayColorAttr([Gf.Vec3f(0,1,.1) if k==selected else Gf.Vec3f(.8,.2,.2) if excluded[k] else Gf.Vec3f(.4,.4,.8)])
  if 'predicted_trajectory' in decision:
   for k,path in enumerate(np.asarray(decision['predicted_trajectory'])):
    xy=path[:,:2]@rotation+root[:2];z=path[:,2]+root[2]+.08
    curve=UsdGeom.BasisCurves.Define(bare.sim.stage,f'/World/R6PredictedCandidate{k}')
    curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([len(path)]);curve.CreatePointsAttr([Gf.Vec3f(float(x),float(y),float(h)) for (x,y),h in zip(xy,z)]);curve.CreateWidthsAttr([.035 if k==selected else .006]);curve.CreateDisplayColorAttr([Gf.Vec3f(1,.8,0) if k==selected else Gf.Vec3f(.5,.5,.5)])
  sphere=UsdGeom.Sphere.Define(bare.sim.stage,'/World/R6Goal');sphere.CreateRadiusAttr(.1);sphere.CreateDisplayColorAttr([Gf.Vec3f(1,0,.1)]);xform=UsdGeom.Xformable(sphere);xform.ClearXformOpOrder();xform.AddTranslateOp().Set(Gf.Vec3d(float(goal[0]),float(goal[1]),float(root[2]+.15)))
  self.status.text=f't={step*.05:.2f}s | '+('STOP' if selected is None else f'candidate={selected}')+'\n14 desired circular paths: green selected, red excluded\n'+('Yellow: selected forecast (.1..2.0s)' if 'predicted_trajectory' in decision else 'Legacy10second reference; no future-pose overlay')+'\n85deg fall labels; physical field name shown above'
