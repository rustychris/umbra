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
import pdb
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
nodes_to_optimize=[n for n in idx_movable
                   if g.nodes['ij'][n,0]<25]

# costf,x0,modify_grid = multincostf(g,nodes_to_optimize)

print("Initial cost: %f"%costf(x0,verbose=True))


c0=costf(x0)
t=time.time()
xopt=fmin_powell(costf,x0,disp=False,xtol=0.01,maxiter=4)
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
    
## 
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
g.plot_edges()
g.plot_nodes(mask=g.nodes['constrained']>0)
ax.axis('equal')
try:
    ax.axis(zoom)
except NameError:
    pass
print("-"*80)

## 

zoom=(-0.81531300962113595,
      6.2923657070774617,
      -0.37078323300472316,
      7.1762028852923372)

# reduced basis approach - choose a few nodes, extrapolate movement
# of other nodes based on the handful chosen
nsub=[6,33,43,66,76,121,131] # hand-picked, small subset.
nsub=[55,65,88,98,132,142,198,208] 
# nsub=nodes_to_optimize[::18] 

# include bordering, non-optimized nodes, to maintain
# a smooth transition
nbrs=[g.node_to_nodes(n)
      for n in nodes_to_optimize]
nbrs=np.unique( np.concatenate(nbrs) )
nbrs=np.setdiff1d(nbrs,nodes_to_optimize)

g=modify_grid(x0)

weighters=np.concatenate( (nsub,nbrs) )
weightx=g.nodes['x'][weighters]


def expand_1d_to_2d(expand):
    """ 
    for node:node weights, double each dimension
    to handle x,y
    """
    expand=expand.repeat(2,0).repeat(2,1)
    expand[::2,1::2]=0
    expand[1::2,::2]=0
    return expand
    
def expand_by_inv_dist():
    # generate a matrix which can be multiplied by
    expand=np.zeros( (len(nodes_to_optimize),
                      len(nsub)), 'f8')

    for ni,n in enumerate(nodes_to_optimize):
        nx=g.nodes['x'][n]
        dists=utils.dist(nx-weightx)
        # something more like the cubic triangulation based
        # approach above would be better here.
        weights=(dists+0.00001)**(-3)
        weights /= weights.sum()
        # goofy - weights are by node, but applied to coordinates,
        # so double up
        expand[ni]=weights[:len(nsub)]
    
    return expand_1d_to_2d(expand)

def expand_by_linear_tri(ax=None):
    # can a triangulation of the weighters give better distribution
    # of the influence?

    xmin,ymin=weightx.min(axis=0)
    xmax,ymax=weightx.max(axis=0)
    dx=xmax-xmin
    dy=ymax-ymin
    halo = np.array( [ [xmin-10*dx,ymin-10*dy],
                       [xmin-10*dx,ymax+10*dy],
                       [xmax+10*dx,ymin-10*dy],
                       [xmax+10*dy,ymax+10*dy] ] )
    weightx_and_halo = np.concatenate( [weightx,halo] )
    weighters_and_halo=np.concatenate( [weighters,[-1]*4] )
    t2=tri.Triangulation(x=weightx_and_halo[:,0],
                         y=weightx_and_halo[:,1])

    # amass the weights for each of the nodes_to_optimize:
    x_to_opt=g.nodes['x'][nodes_to_optimize]

    weights=[]
    for i in range(len(nsub)):
        vals=np.zeros(len(weighters_and_halo))
        vals[ weighters_and_halo==nsub[i] ]=1.0
        if 1:
            tinterp=tri.LinearTriInterpolator(triangulation=t2,z=vals)
        else:
            tinterp=tri.CubicTriInterpolator(triangulation=t2,z=vals)
        # okay - but some nodes fall outside - and get masked.
        weights.append( tinterp(x_to_opt[:,0],x_to_opt[:,1]) )
        #if i==1:
        #    # at this point weights[-1][63] is 0.83 -
        #    # reasonable!
        #    pdb.set_trace()

    weights=np.ma.array(weights) # [nsub, nodes_to_optimize]

    assert weights.mask.sum() == 0
    weights=np.array(weights)

    if ax:
        ax.triplot(t2)
        for i,xy in enumerate(weightx_and_halo):
            ax.text( xy[0],xy[1],str(i))
        for i,xy in enumerate(x_to_opt):
            ax.text( xy[0],xy[1],str(i),color='r')
                     
    return expand_1d_to_2d(weights.T)

g=modify_grid(x0)
expand=expand_by_linear_tri()
# expand=expand_by_inv_dist()

def expanded(xin):
    dxy=np.dot(expand,xin)
    return x0 + dxy

def exp_costf(xin):
    return costf(expanded(xin))

# inv_dist copies x coordinates to both x and y output

xin=0.01*np.zeros(2*len(nsub))
xexp_opt=fmin_powell(exp_costf,xin,xtol=0.01,maxiter=50)
#xexp_opt=0*xin
#xexp_opt[2]=0.0
xexp = expanded(xexp_opt)

g=modify_grid(xexp)
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
colle=g.plot_edges()
g.plot_nodes(mask=nsub)
#ax.plot(weightx[:,0],
#        weightx[:,1],'go')

# expand_by_linear_tri(ax)

# something wrong with the linear tri weights
base_weights=expand[::2,::2]

# show that the ones out side the convex hull
# have low-ish weights - is something doubled up?
#ax.scatter(g.nodes['x'][nodes_to_optimize,0],
#           g.nodes['x'][nodes_to_optimize,1],
#           50,base_weights[:,1],lw=0)

ax.axis(zoom)

# one of the control points is 74 in nodes_to_optimize, and
# which is node 76.
# its neighbor to the west is index 63 in nodes_to_optimize, or
# node 65.
if 0:
    for ni,n in enumerate(nodes_to_optimize):
        ax.text( g.nodes['x'][n,0],
                 g.nodes['x'][n,1],
                 "[%d] %d]"%(ni,n))

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
