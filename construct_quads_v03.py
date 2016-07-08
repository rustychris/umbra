from __future__ import print_function
import time
import utils
import pdb
import plot_utils
import unstructured_grid
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin, fmin_powell
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator

##  

stencil=unstructured_grid.UnstructuredGrid(extra_node_fields=[('ij','i4',2)])

for xi,x in enumerate(np.linspace(0,20,5)):
    omega=0.25
    theta=omega*x
    amp=5
    y=amp*np.cos(theta)
    dx=omega*amp*np.sin(theta)
    dy=1
    mag=np.sqrt(dx**2+dy**2)
    dx/=mag
    dy/=mag

    n1=stencil.add_node(x=[x-dx,y-dy],ij=[5*x,0])
    #stencil.add_node(x=[x,y],ij=[5*x,5])
    n2=stencil.add_node(x=[x+dx,y+dy],ij=[5*x,10])
    #e=stencil.add_edge(nodes=[n1,n2])

# figure out some quads based on proximity in index space
from matplotlib import tri
t=tri.Triangulation(x=stencil.nodes['ij'][:,0],
                    y=stencil.nodes['ij'][:,1])

for abc in t.triangles:
    stencil.add_cell(nodes=abc)
stencil.make_edges_from_cells()

# so for each of the domain locations, evaluate the average of the parameter
# fits, then for each node domain, interpolate those in ij space, evaluate local
# image.

# build a patch for the whole area - 
imax,jmax = stencil.nodes['ij'].max(axis=0)
imin,jmin = stencil.nodes['ij'].min(axis=0)

i_samps=np.arange(imin,imax+1)
j_samps=np.arange(jmin,jmax+1)
I,J=np.meshgrid(i_samps,j_samps)
I=I.ravel()
J=J.ravel()

if 'ij' not in stencil.cells.dtype.names:
    ij_centers=[stencil.nodes['ij'][stencil.cell_to_nodes(c)].astype('f8').mean(axis=0)
                 for c in range(stencil.Ncells())]
    ij_centers=np.array(ij_centers)
    stencil.add_cell_field('ij',ij_centers)

# maybe just form the triangulation in ij space, and apply a linear
# or cubic interpolation.
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
stencil.plot_nodes(labeler=lambda n,rec: "%d,%d"%(rec['ij'][0],rec['ij'][1]))
stencil.plot_edges().set_color('0.5')

ax.axis('equal')
try:
    ax.axis(zoom)
except NameError:
    pass

# have some triangulation t -
pnts=stencil.nodes['ij']
values=stencil.nodes['x']

if 0:
    # does about what you'd expect
    interper=LinearNDInterpolator(pnts,values)
else:
    # nice!
    interper=CloughTocher2DInterpolator(pnts, values)

imin,jmin = stencil.nodes['ij'].min(axis=0)
imax,jmax = stencil.nodes['ij'].max(axis=0)

patch=unstructured_grid.UnstructuredGrid(extra_node_fields=[('ij','i4',2)])

cn_map= patch.add_rectilinear(p0=[0,0],p1=[1,1],
                              nx=1+imax-imin,ny=1+jmax-jmin)

for i,j in np.ndindex(cn_map['nodes'].shape):
    n=cn_map['nodes'][i,j]

    patch.nodes['ij'][n]=[i,j]
    p_ij=patch.nodes['ij'][n]

    patch.nodes['x'][n]=interper(p_ij[0],p_ij[1])


plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
stencil.plot_nodes(labeler=lambda n,rec: "%d,%d"%(rec['ij'][0],rec['ij'][1]))
stencil.plot_edges().set_color('0.5')
patch.plot_edges()

ax.axis('equal')

C_NONE=0 # freely movable
C_FIXED=1 # static
g=patch
if 'constrained' not in g.nodes.dtype.names:
    g.add_node_field('constrained',np.zeros(g.Nnodes(),'i4'))


for n in stencil.valid_node_iter():
    # print( stencil.nodes['ij'][n] )
    s_ij=stencil.nodes['ij'][n]
    idx=np.nonzero( (g.nodes['ij'][:,0]==s_ij[0]) & (g.nodes['ij'][:,1]==s_ij[1]) )[0][0]
    g.nodes['constrained'][idx]=C_FIXED


## 



