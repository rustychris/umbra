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

    n1=stencil.add_node(x=[x+dx,y+dy],ij=[5*x,0])
    #stencil.add_node(x=[x,y],ij=[5*x,5])
    n2=stencil.add_node(x=[x-dx,y-dy],ij=[5*x,10])
    #e=stencil.add_edge(nodes=[n1,n2])
## 


# figure out some quads based on proximity in index space
from matplotlib import tri
t=tri.Triangulation(x=stencil.nodes['ij'][:,0],
                    y=stencil.nodes['ij'][:,1])

for abc in t.triangles:
    stencil.add_cell(nodes=abc)

##     


## 

for c in stencil.valid_cell_iter():
    print c
    nodes=stencil.cell_to_nodes(c)
    break

doms=stencil.nodes['ij'][nodes]
imgs=stencil.nodes['x'][nodes]

def cost(v):
    z0=v[0]+1j*v[1]
    a=v[2]+1j*v[3]
    b=v[4]+1j*v[5]
    xform=lambda z: z0+a*z+b*z**2 
    projs=xform(doms)
    err=np.abs(imgs - projs)
    return (err**2).sum()

def cost2(v):
    z0=v[0]+1j*v[1]
    a=v[2]+1j*v[3]
    b=v[4]+1j*v[5]
    c=v[6]
    def s(z):
        return np.real(z)+1j*c*np.imag(z)
    xform=lambda z: z0+a*s(z)+b*s(z)**2 

    projs=xform(doms)
    err=np.abs(imgs - projs)
    return (err**2).sum()

v0=np.array( [-1,-1,1,0,1,0,1] )

# seems like it needs to fit without aspect ratio first
vopt=fmin_powell(cost,v0[:6])
vopt=np.concatenate( (vopt,[1]) )
#vopt=fmin_powell(cost2,vopt)

## 

i_samps=np.arange(doms[:,0].min(),doms[:,0].max()+1)
j_samps=np.arange(doms[:,1].min(),doms[:,1].max()+1)
I,J=np.meshgrid(i_samps,j_samps)
I=I.ravel()
J=J.ravel()

patch=unstructured_grid.UnstructuredGrid()

v=vopt
v[0]=0
v[1]=6
v[2]=.25
v[3]=-0.02
v[4]=0.00
v[5]=-0.005
v[6]=-1
z0=v[0]+1j*v[1]
a=v[2]+1j*v[3]
b=v[4]+1j*v[5]
c=v[6]
def s(z):
    return np.real(z)+1j*c*np.imag(z)
xform=lambda z: z0+a*s(z)+b*s(z)**2


for i,j in zip(I,J):
    z=xform(i+1j*j) 
    patch.add_node(x=[np.real(z),np.imag(z)] )

# # 


plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
stencil.plot_nodes(labeler=lambda n,rec: "%d,%d"%(rec['ij'][0],rec['ij'][1]))
stencil.plot_edges()
stencil.plot_cells() # it's a thing of beauty!
patch.plot_nodes(ax=ax)

ax.axis('equal')
ax.axis(zoom)
print cost2(v)

# not sure if there's a better formulation to use which
# would allow for a circular arc.
# exp(z) might give a circle?
