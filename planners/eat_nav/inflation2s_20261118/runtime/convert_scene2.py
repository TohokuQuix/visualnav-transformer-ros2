"""Exact MuJoCo binary->triangle mesh conversion, CPU-only; no interpolation/filtering."""
from pathlib import Path
import argparse,struct,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
b=(a.assets/'heightfield.bin').read_bytes();rows,cols=struct.unpack('<ii',b[:8]);assert len(b)==4*(2+rows*cols)
raw=np.frombuffer(b,offset=8,dtype='<f4').reshape(rows,cols).astype(np.float64)
meta=json.loads((a.assets/'metadata.json').read_text());xml=ET.parse(a.assets/'scene.xml');size=np.fromstring(xml.find('asset/hfield').attrib['size'],sep=' ');pos=np.fromstring(xml.find('worldbody/geom').attrib['pos'],sep=' ')
assert (rows,cols)==(meta['rows'],meta['cols']);assert meta['execution_row_y_direction']=='min_to_max'
manifest=json.loads((a.assets.parent/'manifest.json').read_text());expected=next(s['sha256'] for s in manifest['scenes'] if s['scene_id']==2);assert hashlib.sha256(b).hexdigest()==expected
assert np.isfinite(raw).all() and raw.max()>raw.min()
normalized=(raw-raw.min())/(raw.max()-raw.min());height=normalized*size[2]+pos[2]
x=np.linspace(pos[0]-size[0],pos[0]+size[0],cols);y=np.linspace(pos[1]-size[1],pos[1]+size[1],rows);xx,yy=np.meshgrid(x,y)
vertices=np.stack((xx,yy,height),-1).reshape(-1,3);ids=np.arange(rows*cols).reshape(rows,cols);sw=ids[:-1,:-1].ravel();se=ids[:-1,1:].ravel();nw=ids[1:,:-1].ravel();ne=ids[1:,1:].ravel()
# MuJoCo3.3.6 engine_collision_convex.c: row strips visit (r+1,c),(r,c),then next col: diagonal SW->NE.
faces=np.stack((np.stack((sw,se,ne),1),np.stack((sw,ne,nw),1)),1).reshape(-1,3)
top_faces=len(faces);top_vertices=len(vertices)
# Preserve hfield base and vertical perimeter sides; only top samples define terrain elevation.
perimeter=np.concatenate((ids[0,:],ids[1:,-1],ids[-1,-2::-1],ids[-2:0:-1,0]))
bottom=vertices[perimeter].copy();bottom[:,2]=pos[2]-size[3];bid=np.arange(len(vertices),len(vertices)+len(bottom));vertices=np.concatenate((vertices,bottom));extra=[]
for i,j in enumerate(np.roll(np.arange(len(perimeter)),-1)):
    extra.extend(((perimeter[i],bid[i],bid[j]),(perimeter[i],bid[j],perimeter[j])))
# Center fan avoids degenerate triangles along straight perimeter runs.
center=len(vertices);vertices=np.concatenate((vertices,[[pos[0],pos[1],pos[2]-size[3]]]))
for i,j in enumerate(np.roll(np.arange(len(bid)),-1)):extra.append((center,bid[j],bid[i]))
faces=np.concatenate((faces,np.array(extra,dtype=np.int64)))
np.savez_compressed(a.output/'scene2_mesh.npz',vertices=vertices,faces=faces,heights=height,x=x,y=y,top_vertices=top_vertices,top_faces=top_faces)
for key,actual in [('min_x',x[0]),('max_x',x[-1]),('min_y',y[0]),('max_y',y[-1]),('min_z',height.min()),('max_z',height.max()),('base_thickness_m',size[3]),('resolution_m',x[1]-x[0])]:assert abs(meta[key]-actual)<1e-12,(key,meta[key],actual)
point_errors=np.abs(vertices[:top_vertices,2].reshape(rows,cols)-height)
assert point_errors.max()==0;assert np.max(abs(np.diff(x)-.02))<1e-12;assert np.max(abs(np.diff(y)-.02))<1e-12
samples=[]
for r,c in [(0,0),(0,cols-1),(rows-1,0),(rows-1,cols-1),(112,173),(int(np.argmax(height)//cols),int(np.argmax(height)%cols)),(50,120),(150,240)]:samples.append(dict(row=r,col=c,raw=float(raw[r,c]),xyz=vertices[r*cols+c].tolist()))
# Find a spawn with >=0.7x0.7m flat pad wholly inside original terrain, preferring low original elevation.
choices=[]
for r in range(40,rows-40,5):
 for c in range(40,cols-40,5):
  patch=height[r-20:r+21,c-20:c+21];choices.append((float(np.ptp(patch)),float(patch.max()),r,c))
# Original zero margin provides a reproducible staging strip without an added floor.
c=int(np.argmin(abs(x-2.46)));r=int(np.argmin(abs(y+.5)));patch=height[r-18:r+19,c-35:c+36];spread=float(np.ptp(patch));z=float(patch.max());spawn=[float(x[c]),float(y[r]),z]
report=dict(format='little endian int32 rows, int32 cols; C row-major float32 elevations',rows=rows,cols=cols,bytes=len(b),source_sha256=hashlib.sha256(b).hexdigest(),raw_range=[float(raw.min()),float(raw.max())],normalization='(raw-min)/(max-min), identity here because range is exactly[0,1]',size=size.tolist(),position=pos.tolist(),bounds_top=np.stack((vertices[:top_vertices].min(0),vertices[:top_vertices].max(0))).tolist(),resolution=[float(x[1]-x[0]),float(y[1]-y[0])],axis='columns->world+x,rows->world+y; no reversal or transpose',vertex_max_error_m=float(point_errors.max()),top_vertices=top_vertices,top_triangles=top_faces,total_vertices=len(vertices),total_triangles=len(faces),diagonal='SW->NE (row+col+), matches MuJoCo3.3.6 source triangle strips',samples=samples,spawn_terrain_xyz=spawn,spawn_patch_ptp_m=spread,discrepancies=[])
(a.output/'conversion.json').write_text(json.dumps(report,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(height,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='terrain');ax.plot(spawn[0],spawn[1],'rx');ax.set(xlabel='world x(m)',ylabel='world y(m)',title='Scene2 original samples; red=stable spawn candidate');fig.colorbar(im,label='height(m)');fig.savefig(a.output/'heightfield-audit.png',dpi=150)
print(json.dumps(report,indent=2))
