"""Conditional, versioned left/right gap closures; original mesh remains unchanged."""
import numpy as np,json,hashlib
from pathlib import Path

def create_gap_wall_variant(source,output):
 source=Path(source);output=Path(output)
 if output.exists():raise FileExistsError(output)
 z=np.load(source);vertices=z['vertices'];faces=z['faces'];h=vertices[:225*347,2].reshape(225,347);vv=[vertices];ff=[faces];changes=[];n=len(vertices)
 box_faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],int)
 for side,x,cols in [('start_side',-.025,[50,51,52]),('goal_side',4.925,[293,294,295])]:
  low=(h[50:175,cols].max(axis=1)<.8);starts=np.flatnonzero(low&~np.r_[False,low[:-1]]);ends=np.flatnonzero(low&~np.r_[low[1:],False])
  for a,b in zip(starts,ends):
   y0=max(0.,a*.02-.01);y1=min(2.48,b*.02+.01)
   if y1-y0<.019:continue
   xmin,xmax=(-.05,0.) if side=='start_side' else (4.9,4.95);v=np.array([[xmin,y0,-.1],[xmax,y0,-.1],[xmax,y1,-.1],[xmin,y1,-.1],[xmin,y0,1.187191367149353],[xmax,y0,1.187191367149353],[xmax,y1,1.187191367149353],[xmin,y1,1.187191367149353]])
   vv.append(v);ff.append(box_faces+n);n+=8;changes.append(dict(side=side,x_range=[xmin,xmax],y_range=[y0,y1],z_range=[-.1,1.187191367149353]))
 outv=np.concatenate(vv);outf=np.concatenate(ff);assert np.array_equal(outv[:len(vertices)],vertices) and np.array_equal(outf[:len(faces)],faces)
 output.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(output,vertices=outv,faces=outf)
 report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),original_vertices_unchanged=True,original_faces_unchanged=True,trigger_required='all10original trials bypass',added_walls=changes,selection='left/right perimeter gaps where existing boundary maxheight<.8; no interior terrain edits')
 output.with_suffix('.json').write_text(json.dumps(report,indent=2));return report
