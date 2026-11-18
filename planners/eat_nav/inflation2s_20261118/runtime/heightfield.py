import numpy as np
def triangle_height(mesh,xy):
 h=np.asarray(mesh['heights']);xs=np.asarray(mesh['x']);ys=np.asarray(mesh['y']);x,y=np.asarray(xy,float)
 if h.shape!=(len(ys),len(xs)) or min(h.shape)<2:raise ValueError('heightfield dimensions differ')
 if not(xs[0]-1e-10<=x<=xs[-1]+1e-10 and ys[0]-1e-10<=y<=ys[-1]+1e-10):raise ValueError('spawn footprint outside original terrain')
 j=min(max(int(np.searchsorted(xs,x,side='right'))-1,0),len(xs)-2);i=min(max(int(np.searchsorted(ys,y,side='right'))-1,0),len(ys)-2)
 u=(x-xs[j])/(xs[j+1]-xs[j]);v=(y-ys[i])/(ys[i+1]-ys[i])
 return float((1-u)*h[i,j]+(u-v)*h[i,j+1]+v*h[i+1,j+1]) if u>=v else float((1-v)*h[i,j]+(v-u)*h[i+1,j]+u*h[i+1,j+1])
