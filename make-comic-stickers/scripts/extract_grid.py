"""Split a regular transparent grid; always inspect extracted assets visually."""
import numpy as np
from PIL import Image
from scipy import ndimage
def extract(source,heading_shift=0,rows=4,cols=3):
 rgba=np.asarray(Image.open(source).convert('RGBA'));a=rgba[:,:,3];h,w=a.shape
 if a.min()==255:raise ValueError('Source grid needs a transparent background')
 lab,n=ndimage.label(a>=128);counts=np.bincount(lab.ravel());slices=ndimage.find_objects(lab)
 anchors=[]
 for row in range(rows):
  for col in range(cols):
   cell=lab[round(row*h/rows):round((row+1)*h/rows),round(col*w/cols):round((col+1)*w/cols)]
   ns=np.bincount(cell.ravel(),minlength=n+1);ns[0]=0
   for old in anchors:ns[old]=0
   k=int(ns.argmax());assert ns[k]>max(20,h*w/(rows*cols)*.002),(row,col,ns[k]);anchors.append(k)
 owner=np.full(n+1,-1,int)
 for i,k in enumerate(anchors):owner[k]=i
 centers=ndimage.center_of_mass(a>=128,lab,list(range(1,n+1)))
 for row in range(rows):
  subset=anchors[row*cols:row*cols+cols];main=np.isin(lab,subset)
  _,near=ndimage.distance_transform_edt(~main,return_indices=True)
  for k in range(1,n+1):
   if owner[k]>=0 or counts[k]<5:continue
   cy,cx=centers[k-1]
   if min(rows-1,int((cy+heading_shift)*rows/h))!=row:continue
   sl=slices[k-1]
   if cy-row*h/rows<.32*h/rows:
    idx=row*cols+min(cols-1,int(cx*cols/w))
   else:
    y=int(cy);x=int(cx);idx=int(owner[lab[near[0,y,x],near[1,y,x]]])
   owner[k]=idx
 # Recover antialiased fringes immediately adjacent to the opaque artwork.
 dist,near=ndimage.distance_transform_edt(lab==0,return_indices=True)
 owned=owner[lab];fringe=(lab==0)&(a>0)&(dist<=2.5)
 owned[fringe]=owner[lab[near[0,fringe],near[1,fringe]]]
 assets=[]
 for i in range(rows*cols):
  m=owned==i;yy,xx=np.where(m);assert len(xx)>20
  lo=[max(0,int(xx.min())-3),max(0,int(yy.min())-3)];hi=[min(w,int(xx.max())+4),min(h,int(yy.max())+4)]
  tile=rgba[lo[1]:hi[1],lo[0]:hi[0]].copy();tile[:,:,3]=np.where(m[lo[1]:hi[1],lo[0]:hi[0]],tile[:,:,3],0)
  assets.append(Image.fromarray(tile))
 return assets
