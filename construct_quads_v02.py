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
from __future__ import print_function
import time
import utils
import plot_utils
import unstructured_grid
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin, fmin_powell
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

## 

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

## 
# have some triangulation t -
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator
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


movable=g.nodes['constrained']==0
idx_movable=np.nonzero(movable)[0]


def node_to_triples(g,n,full=True):
    """ full=False: only include triples with n in the middle
    """
    trips=[]
    cells=g.node_to_cells(n)
    for c in cells:
        c_nodes=list(g.cell_to_nodes(c))
        idx=c_nodes.index(n)
        N=len(c_nodes)
        if full:
            offsets=[-1,0,1]
        else:
            offsets=[0]
        for offset in offsets:
            trips.append( [c_nodes[(idx-1+offset)%N],
                           c_nodes[(idx  +offset)%N],
                           c_nodes[(idx+1+offset)%N] ] )
    return np.array(trips)



            

## 

# use the angles, but do chunks of nodes at a time

# node_to_triples(g,n,full=True)

def multincostf(g,nodes,verbose=0):
    def modify_grid(x):
        x=x.reshape([len(nodes),2])
        g.nodes['x'][nodes_to_optimize] = x
        g.cells['_center']=np.nan
        g.cells['_area']=np.nan
        return g

    x0=np.array([g.nodes['x'][n] for n in nodes]).ravel()

    cells = np.unique(np.concatenate( [g.node_to_cells(n) for n in nodes]) )

    #edges=[set(g.cell_to_edges(c))
    #       for c in cells]
    #edges=list( edges[0].union( *edges[1:] ) )
    #links=[j for j in edges
    #       if np.all(g.edges['cells'][j]>0)]

    triples=np.concatenate( [node_to_triples(g,n)
                             for n in nodes] )
    # keep just the unique triples
    triples=np.array( list(set( [tuple(r) for r in triples] ) ) )

    def costf(x,verbose=verbose):
        g_mod=modify_grid(x)
        cost=0

        w_cangle=0.0 # slows things down considerably.
        w_angle=0.1
        w_cons_length=1000.
 
        if w_angle>0:
            # interior angle of cells
            deltas=np.diff(g_mod.nodes['x'][triples],axis=1)
            angles=np.diff( np.arctan2( deltas[:,:,1], deltas[:,:,0] ),axis=1) % (2*np.pi) * 180/np.pi
            angle_cost=np.sum( (angles-90)**2 )
            if verbose:
                print("   Angle cost: %f"%angle_cost)
            cost+=w_angle*angle_cost
            if w_cons_length>0:
                Ls=utils.dist(np.diff(g_mod.nodes['x'][triples],axis=1))
                cost_cons_length=np.sum( ((Ls[:,0]-Ls[:,1])/(Ls[:,0]+Ls[:,1]))**4 )
                if verbose:
                    print("   Con length cost: %f"%(w_cons_length*cost_cons_length))
                cost+=w_cons_length*cost_cons_length
        if w_cangle>0:
            ctrs=g_mod.cells_center(refresh=True)
            cangle_cost=0
            for c in cells:
                nodes=g.cell_to_nodes(c)
                deltas=g_mod.nodes['x'][nodes] - ctrs[c]
                angles=np.arctan2(deltas[:,1],deltas[:,0]) * 180/np.pi
                cds=utils.cdiff(angles) % 360.
                cangle_cost+=np.sum( (cds-90)**4 )
            if verbose:
                print "   CAngle cost: %f"%( w_cangle*cangle_cost )
            cost+=w_cangle*cangle_cost

        return cost
    return costf,x0,modify_grid

# 229 nodes
nodes_to_optimize=[n for n in idx_movable]
#                    if 17<g.nodes['ij'][n,0]<40]

costf,x0,modify_grid = multincostf(g,nodes_to_optimize)

print("Initial cost: %f"%costf(x0,verbose=True))


c0=costf(x0)
t=time.time()
xopt=fmin_powell(costf,x0,disp=False,xtol=0.01,maxiter=15)
print("Optimized:")
copt=costf(xopt,verbose=True)
if copt>c0:
    print "F",
    g=modify_grid(x0)
elif copt==c0:
    print "~",
    g=modify_grid(x0)
else:
    g=modify_grid(xopt)
    g.cells_center(refresh=True)
    elapsed=time.time()-t
    print( "Elapsed: %.3fs - cost delta=%f"%(elapsed,c0-copt) )

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
g.plot_edges()
g.plot_nodes(mask=g.nodes['constrained']>0)
ax.axis('equal')
ax.axis(zoom)
print("-"*80)

## 

angle_errs=180/np.pi * np.abs(g.angle_errors())
angle_errs[np.isnan(angle_errs)]=0
circ_errs=g.circum_errors()


plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
colle=g.plot_edges(values=angle_errs,lw=2)
collc=g.plot_cells(values=circ_errs,zorder=-5)
g.plot_nodes(mask=g.nodes['constrained']>0)

plot_utils.cbar(colle,label='angles')
plot_utils.cbar(collc,label='circ.')
ax.axis('equal')
ax.axis(zoom)


# bottom line - the method above does a decent job of getting
# the errors down to the 1 degree, 1% circumcenter error.
# but it does not make significant improvements at scales
# larger than 15 cells or so.  
# seems like a good candidate for a multigrid-ish approach.
# or some sort of reduced resolution warping - 
# maybe choose half the nodes, then define an interpolation scheme
# to modify the remaining nodes.  could cascade that up. 
# it's not clear that 
