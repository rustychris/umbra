# Try out some methods for laying out contiguous quads

# workflow:
#  Will start out with a "friendly" way of getting to the node/ij
#  inputs below.  For initial proof-of-concept, go straight to that
#  ...
#  Have a collection of fixed nodes with associated i,j indices
#  For nearby/consecutive groups of those nodes, calculate a local
#  conformal map.
#  interpolate that map between groups calculating the intervening
#  node locations.


import unstructured_grid
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin, fmin_powell
##  

#
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
## 


# figure out some quads based on proximity in index space
from matplotlib import tri
t=tri.Triangulation(x=stencil.nodes['ij'][:,0],
                    y=stencil.nodes['ij'][:,1])

for abc in t.triangles:
    stencil.add_cell(nodes=abc)

##     
nparams=7

def fwd7(v):
    z0=v[0]+1j*v[1]
    a=v[2]+1j*v[3]
    b=v[4]+1j*v[5]
    c=v[6]
    def s(z):
        return np.real(z)+1j*c*np.imag(z)
    return lambda z: z0+a*s(z)+b*s(z)**2 

def fwd6(v):
    return fwd7(np.concatenate( (v,[1])))

def fit_doms_imgs(doms,imgs,aspect=1):
    def cost(v):
        v=np.concatenate( (v,[aspect]))
        projs=fwd7(v)(doms[:,0]+1j*doms[:,1])
        err=np.abs(imgs[:,0]+1j*imgs[:,1] - projs)
        return (err**2).sum()

    def cost2(v):
        projs=fwd7(v)(doms[:,0]+1j*doms[:,1])
        err=np.abs(imgs[:,0]+1j*imgs[:,1] - projs)
        return (err**2).sum()

    v0=np.array( [-1,-1,1,0,1,0,1] )

    # seems like it needs to fit without aspect ratio first
    # maybe.  
    # but if we fit without aspect ratio first, then have to
    # be sure that at least the sign is correct (i.e. dom and
    # img are both right-handed coordinate systems)
    # with 3 nodes, there can be multiple exact answers.
    # with 4 nodes, it came close but wasn't exact.
    print("Starting f: ",cost(v0[:6]))
    vopt=fmin_powell(cost,v0[:6])
    print("midway f: ",cost(vopt))
    vopt=np.concatenate( (vopt,[aspect]) )
    vopt=fmin_powell(cost2,vopt)
    print("final f: ",cost2(vopt))
    print("optimized parameters: ",vopt)
    return vopt
# # 

fits=np.zeros( (stencil.Ncells(),nparams) )

for c in range(stencil.Ncells()):
    if stencil.cells['deleted'][c]:
        fits[c][:]=np.nan
    else:
        nodes=stencil.cell_to_nodes(c)
    
        doms=stencil.nodes['ij'][nodes]
        imgs=stencil.nodes['x'][nodes]
        v=fit_doms_imgs(doms,imgs,aspect=0.8)
        fits[c,:]=v # order makes a difference?!

if 'fit' not in stencil.cells.dtype.names:
    stencil.add_cell_field('fit',fits)
stencil.make_edges_from_cells()
## 

i_samps=np.arange(doms[:,0].min(),doms[:,0].max()+1)
j_samps=np.arange(doms[:,1].min(),doms[:,1].max()+1)
I,J=np.meshgrid(i_samps,j_samps)
I=I.ravel()
J=J.ravel()

patch=unstructured_grid.UnstructuredGrid()

xform=fwd7(vopt)

for i,j in zip(I,J):
    z=xform(i+1j*j) 
    patch.add_node(x=[np.real(z),np.imag(z)] )

# # 


plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
stencil.plot_nodes(labeler=lambda n,rec: "%d,%d"%(rec['ij'][0],rec['ij'][1]))
stencil.plot_edges().set_color('0.5')
# stencil.plot_cells() # it's a thing of beauty!
patch.plot_nodes(ax=ax)

ax.axis('equal')
ax.axis(zoom)
# print cost2(v)

# not sure if there's a better formulation to use which
# would allow for a circular arc.
# exp(z) might give a circle?

## 

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
## 
patch=unstructured_grid.UnstructuredGrid(extra_node_fields=[('ij','i4',2),
                                                            ('fit','f8',nparams)])

cn_map= patch.add_rectilinear(p0=[0,0],p1=[1,1],
                              nx=1+imax-imin,ny=1+jmax-jmin)

# # 
for i,j in np.ndindex(cn_map['nodes'].shape): # zip(I,J):
    n=cn_map['nodes'][i,j]

    ij_dist=(i-stencil.cells['ij'][:,0])**2 + (j-stencil.cells['ij'][:,1])**2
    if 0:# Choose a cell:
        c=np.argmin(ij_dist)
        fit=stencil.cells['fit'][c,:]
    else: # inverse distance
        cbest=np.argsort(ij_dist)[:4]
        weights=(1+ij_dist[cbest])**(-2)
        weights=weights / weights.sum()
        fit=(stencil.cells['fit'][cbest,:] * weights[:,None]).sum(axis=0)

    z=fwd7(fit)(i+1j*j)
    #patch.add_node(x=[np.real(z),np.imag(z)],ij=[i,j])
    patch.nodes['x'][n]=[np.real(z),np.imag(z)]
    patch.nodes['ij'][n]=[i,j]
# # 


plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
stencil.plot_nodes(labeler=lambda n,rec: "%d,%d"%(rec['ij'][0],rec['ij'][1]))
# stencil.plot_edges().set_color('0.5')
# stencil.plot_cells() # it's a thing of beauty!

# patch.plot_nodes(ax=ax)
patch.plot_edges(ax=ax)
plt.draw()


# HERE:
#  there's a tendency for the grid to expand a it gets away from
#  the more-closely bunched control points.
#  this is probably related to how the parameterization is written
#  could maybe get around this by fitting groups of 4 at once.
#  bigger picture, probably not using the best parameterization of
#  the transform.

# the result could likely be improved slightly by interpolating
# between nodes, with each node getting a mean of the fits it participated
# in.

# would be worth looking into other parameterizations, especially parameterizations
# which more easily handle arc'ing segments.
