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

import time
import utils
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

# figure out some quads based on proximity in index space
from matplotlib import tri
t=tri.Triangulation(x=stencil.nodes['ij'][:,0],
                    y=stencil.nodes['ij'][:,1])

for abc in t.triangles:
    stencil.add_cell(nodes=abc)
stencil.make_edges_from_cells()

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
## 

i_samps=np.arange(doms[:,0].min(),doms[:,0].max()+1)
j_samps=np.arange(doms[:,1].min(),doms[:,1].max()+1)
I,J=np.meshgrid(i_samps,j_samps)
I=I.ravel()
J=J.ravel()

patch=unstructured_grid.UnstructuredGrid()
## 
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

# Mobius transformations might be useful.

## 

patch=unstructured_grid.UnstructuredGrid(extra_node_fields=[('ij','i4',2)])

cn_map= patch.add_rectilinear(p0=[0,0],p1=[1,1],
                              nx=30,ny=30)

for i,j in np.ndindex(cn_map['nodes'].shape): 
    n=cn_map['nodes'][i,j]

    # typ. rectangular
    z=(0.5*(i-15)+2j*(j-15))
    try:
        #a=1+1j
        #b=1j
        #c=-.1j
        #d=1
        #z=(a*z+b)/(c*z+d)
        #z=np.cos(0.1*(z+10j))
        # z=np.exp(0.1*z)
        z=1+0.1*z+(0.1*z)**2/2+(0.1*z)**3/6
        patch.nodes['x'][n]=[np.real(z),np.imag(z)]
    except ZeroDivisionError:
        patch.nodes['x'][n]=np.nan

    patch.nodes['ij'][n]=[i,j]

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

# patch.plot_nodes(ax=ax)
patch.plot_edges(ax=ax)
plt.draw()

## 

# could be something promising - but after looking at the grid
# that Ed sent, I'm wondering if a brute-force approach would be
# better.
# something where we track local grid geometry, and interpolate
# that.
# so if each node carried a local estimate of dx,dy, and an angle?
# or something along the lines of spline fitting?
#
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


# clumping - optimize a single node at a time
def ncostf(g,n,w_area=0,w_angle=0,w_length=0,w_cangle=0,verbose=0):
    def modified_node_grid(g_src,n,x):
        g_test=g_src.copy()
        g_test.nodes['x'][n]=x
        g_test.cells_center(refresh=True) # ,mode='sequential')
        g_test.cells['_area']=np.nan
        g_test.cells['edges']=g.cells['edges']

        return g_test

    cells=g.node_to_cells(n)
    edges=[set(g.cell_to_edges(c))
           for c in cells]
    edges=list( edges[0].union( *edges[1:] ) )
    links=[j for j in edges
           if np.all(g.edges['cells'][j]>0)]

    triples=node_to_triples(g,n)

    if verbose:
        print "COST: n=%d cells %s  links %s"%(n,cells,links)
        print "   triples: ",triples

    def cost(x):
        g_mod=modified_node_grid(g,n,x)
        # This one sucked:
        if 0:
            cost=np.sum([link_cost(g_mod,j)
                         for j in links])

        # mimic report_orthogonality()
        cost=0
        if 1:
            nsi=4 # quads only right now...
            centers=g_mod.cells_center(mode='sequential')
            # this is somehow making the mean and max circumcenter error larger!
            offsets = g_mod.nodes['x'][g_mod.cells['nodes'][cells,:nsi]] - centers[cells,None,:]
            dists = utils.mag(offsets)

            # maybe lets small cells take over too much
            # cost += np.mean( np.std(dists,axis=1) / np.mean(dists,axis=1) )
            # different, but not much better
            # cost += np.mean( np.std(dists,axis=1) )
            base_cost=np.max( np.std(dists,axis=1)  )
            if verbose:
                print "   Circum. cost: %f"%(base_cost)
            cost += base_cost
            
        if w_area>0 and len(links):
            # this helps a bit, but sometimes getting good areas
            # means distorting the shapes
            A=g_mod.cells_area()
            pairs=A[ g.edges['cells'][links] ]
            diffs=(pairs[:,0]-pairs[:,1])/pairs.mean(axis=1)
            area_cost=w_area*np.sum(diffs**2)
            if verbose:
                print "   Area cost: %f"%area_cost
            cost+=area_cost
        if w_length>0:
            # if it's not too far off, this makes it look nicer at
            # the expense of circ. errors
            l_cost=0
            g_mod.update_cell_edges()
            lengths=g_mod.edges_length()
            for c in cells:
                e=g_mod.cell_to_edges(c)
                if 0:
                    # maybe that's too much leeway, and
                    # the user should have to specify aspect ratios
                    if len(e)==4:
                        a,b,c,d=lengths[e]
                        l_cost+=( ((a-c)/(a+c))**2 +
                                  ((b-d)/(b+d))**2 )
                else:
                    lmean=lengths.mean()
                    l_cost+= np.sum( (lengths-lmean)**2 )
            if verbose:
                print "   Length cost: %f"%( w_length*l_cost )
            cost+=w_length*l_cost
        if w_angle>0:
            # interior angle of cells
            deltas=np.diff(g_mod.nodes['x'][triples],axis=1)
            angles=np.diff( np.arctan2( deltas[:,:,1], deltas[:,:,0] ),axis=1) % (2*np.pi) * 180/np.pi
            angle_cost=w_angle*np.sum( (angles-90)**2 )
            if verbose:
                print "   Angle cost: %f"%angle_cost
            cost+=angle_cost
        if w_cangle>0:
            cangle_cost=0
            for c in cells:
                nodes=g.cell_to_nodes(c)
                deltas=g_mod.nodes['x'][nodes] - g_mod.cells_center()[c]
                angles=np.arctan2(deltas[:,1],deltas[:,0]) * 180/np.pi
                cds=utils.cdiff(angles) % 360.
                cangle_cost+=np.sum( (cds-90)**4 )
            if verbose:
                print "   CAngle cost: %f"%( w_cangle*cangle_cost )
            cost+=w_cangle*cangle_cost
                
        return cost
    return cost


for it in range(5):
    reordered=np.argsort( np.random.random(len(idx_movable)) )
    for n in idx_movable[reordered]:
        if g.nodes['ij'][n,0]>20:
            continue
        t=time.time()
        x0=g.nodes['x'][n].copy()
        costf=ncostf(g,n,w_length=0.0,w_area=0,w_angle=0.1,w_cangle=0.000001)
        c0=costf(x0)
        nxopt=fmin(costf,x0,disp=False)
        copt=costf(nxopt)
        if copt>c0:
            print "F",
        elif copt==c0:
            print "~",
        else:
            g.nodes['x'][n]=nxopt
            g.cells_center(refresh=True)
            elapsed=time.time()-t
            print( "Node=%d elapsed: %.3fs - delta %s, cost delta=%f"%(n,elapsed,nxopt-x0, c0-copt) )
    plt.figure(1).clf()
    fig,ax=plt.subplots(1,1,num=1)
    g.plot_edges()
    g.plot_nodes(mask=g.nodes['constrained']>0)
    ax.axis('equal')
    ax.axis(zoom)
    print("-"*80)
    plt.pause(0.1)

            

## 
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
stencil.plot_nodes(labeler=lambda n,rec: "%d,%d"%(rec['ij'][0],rec['ij'][1]))
stencil.plot_edges().set_color('0.5')
g.plot_edges()
g.plot_nodes(mask=g.nodes['constrained'])

ax.axis('equal')


patch.report_orthogonality()

# And can that be optimized to something reasonably orthogonal?
# starts off with:
#    Mean circumcenter error: 0.0953189
#    Max circumcenter error: 0.342121
#  Recalculating edge to cells
#    Mean angle error: 6.96 deg
#    Max angle error: 69.47 deg

# after one round -
#   Mean circumcenter error: 0.0591299
#   Max circumcenter error: 0.4069
# Recalculating edge to cells
#   Mean angle error: 4.42 deg
#   Max angle error: 36.60 deg

# pretty slow - could probably do something multigrid-ish here.
