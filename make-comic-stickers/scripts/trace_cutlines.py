import ctypes as C
from ctypes.util import find_library
import numpy as np
class P(C.Structure):_fields_=[('x',C.c_double),('y',C.c_double)]
class Curve(C.Structure):_fields_=[('n',C.c_int),('tag',C.POINTER(C.c_int)),('c',C.POINTER(P*3))]
class Path(C.Structure):pass
Path._fields_=[('area',C.c_int),('sign',C.c_int),('curve',Curve),('next',C.POINTER(Path)),('childlist',C.POINTER(Path)),('sibling',C.POINTER(Path)),('priv',C.c_void_p)]
class State(C.Structure):_fields_=[('status',C.c_int),('plist',C.POINTER(Path)),('priv',C.c_void_p)]
class Bitmap(C.Structure):_fields_=[('w',C.c_int),('h',C.c_int),('dy',C.c_int),('map',C.POINTER(C.c_ulong))]
class Param(C.Structure):_fields_=[('turdsize',C.c_int),('turnpolicy',C.c_int),('alphamax',C.c_double),('opticurve',C.c_int),('opttolerance',C.c_double)]
libname=find_library('potrace')
if not libname:raise RuntimeError('Install the system libpotrace shared library before running this script.')
if C.sizeof(C.c_ulong)!=8:raise RuntimeError('This Potrace binding requires a 64-bit unsigned-long ABI (Linux/macOS 64-bit).')
L=C.CDLL(libname);L.potrace_param_default.restype=C.POINTER(Param);L.potrace_param_free.argtypes=[C.POINTER(Param)];L.potrace_trace.argtypes=[C.POINTER(Param),C.POINTER(Bitmap)];L.potrace_trace.restype=C.POINTER(State);L.potrace_state_free.argtypes=[C.POINTER(State)]
def trace(mask):
 h,w=mask.shape;dy=(w+63)//64;a=np.zeros((h,dy*64),np.uint8);a[:,:w]=mask
 words=np.ascontiguousarray(np.packbits(a,axis=1,bitorder='big').view('>u8').astype(np.uint64))
 bm=Bitmap(w,h,dy,words.ctypes.data_as(C.POINTER(C.c_ulong)));pa=L.potrace_param_default();pa.contents.turdsize=2;pa.contents.opttolerance=.12
 st=L.potrace_trace(pa,C.byref(bm));assert st and st.contents.status==0
 p=st.contents.plist;parts=[];loops=0
 def xy(v):return f'{v.x:.3f},{v.y:.3f}'
 while p:
  v=p.contents.curve
  if v.n:
   q=['M'+xy(v.c[v.n-1][2])]
   for k in range(v.n):
    c=v.c[k]
    if v.tag[k]==2:q.extend(['L'+xy(c[1]),'L'+xy(c[2])])
    else:q.append('C'+xy(c[0])+' '+xy(c[1])+' '+xy(c[2]))
   parts.append(' '.join(q)+' Z');loops+=1
  p=p.contents.next
 L.potrace_state_free(st);L.potrace_param_free(pa)
 return ' '.join(parts),loops
