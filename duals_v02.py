import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin,fmin_cg,fmin_bfgs,fmin_powell
import time
import utils
reload(utils)

import unstructured_grid
reload(unstructured_grid)
## 

# constraint names:
C_NONE=0 # freely movable
C_FIXED=1 # static

# next steps in that approach would be to
# try it on a larger test case, and start accelerating
# the evaluation of the cost function.

# other approaches:
#  - infer a local conformal mapping from 1 or 2
#    cells, then use that to shift the locations of
#    vertices
#  - where enough nodes are constrained, propagate that
#    constraint across the mesh.  probably only works in
#    a small set of situations
#  - reformulate the problem in terms of circumcenters and
#    position relative to those circumcenters
#    For the test case of 3x3 nodes, which 4 constrained:
#      currently, have 5 free nodes, for 10 dof.
#      for a perfectly orthogonal grid, nodes 1,7 have 1 dof
#      then cells 0 and 2 have a radius degree of freedom
#      which fully determines node 3, and leaves 0 and 6 with
#      1 dof.
#    total actual dofs: 6
#    if the original problem were specified in terms of circumcenters
#    and radii - cells 1 and 3 are fully constrained.  then there
#    are 3 remaining dofs for circumcenters of cells 0 and 2, plus
#    1 dofs each for 0 and 6, less the radius constraint from 4, so back
#    to 6.

#  - do more of this on the dual?
#  - focus on the edges and nodes, making angles approach 90.
#    relatively quick to get a nice output, though the errors
#    are not necessarily minimized

#  any way to use a 2d spline of some sort?

# in the conformal map world:
# F(eta,zeta) => x,y
#  such that (dx/deta,dy/deta ) . (dx/dzeta,dy/dzeta) = 0
#  and at certain locations F(eta_i,zeta_a) = x_i,y_i

# is there some way to do a taylor expansion from a cell?
#  x(e,z) = x0+a*(e-e0) + b*(z-z0) + c*(e-e0)**2 + d*(z-z0)**2 + g*(e-e0)*(z-z0)

# for a 2nd order poly, get 10 coefficients.
#   if it's defined by the 4 vertices and the perpendicularity requirement..
#   one vertex defines x0.
#   3 vertices, or 6 equations, plus
#   4 constraints on perpendicularity

# does any of this get clearer using complex numbers?


nx=7
ny=5

p1=np.array([-2.,-2])
p2=np.array([2.,2])

g= unstructured_grid.UnstructuredGrid()
gp_map = g.add_rectilinear(p1,p2,nx,ny)
nmap=gp_map['nodes']

g.add_node_field('eta',np.zeros(g.Nnodes()))
g.add_node_field('zeta',np.zeros(g.Nnodes()))
g.add_node_field('ez',np.zeros(g.Nnodes(),np.complex128))

for eta,zeta in np.ndindex(nmap.shape):
    n=nmap[eta,zeta] 
    g.nodes['eta'][n] = eta
    g.nodes['zeta'][n] = zeta
    g.nodes['ez'][n]=eta+1j*zeta


def apply_xform(f):
    C=f(g.nodes['ez'])
    g.nodes['x'][:,0] = np.real(C)
    g.nodes['x'][:,1] = np.imag(C)

# regular 2nd degree, 2-d polynomial: a+bx +cy +dxy + e x^2 + fy^2
# that's 5 for just one dimensions
# so total 10.
# but 2 of those are translation
# at least one is perpendicularity, probably 2?
# hmm.
# so with something this simple,
# have 6 dofs, with which to try and fit to 4 vertices.
# 
z0=0+0j
a=1+.1j
b=.01+0.1j
c=1
def s(z):
    return np.real(z)+1j*c*np.imag(z)

apply_xform( lambda z: z0+a*s(z)+b*s(z)**2 )

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
g.plot_edges(ax=ax)
ax.axis('equal')
plt.draw()


if 0:
    targets=np.array( [ [0+0j,2+0j],
                        [10+0j,8+0j],
                        [0+10j,0+10j],
                        [10+10j,10+10j]] )
if 0:
    targets=np.array( [ [0+0j,2+0j],
                        [5+0j,5+0j],
                        [0+10j,0+10j],
                        [10+10j,10+10j]] )
if 0:
    targets=np.array( [ [0+4j,0+10j],
                        [5+4j,5+11j],
                        [10+4j,10+10j],
                        [5+2j,5+9j]])
if 1:
    targets=np.array( [ [0+2j,0+5j],
                        [(nx-1)//2+2j,5+7j],
                        [(nx-1)+2j,10+5j]])
doms=targets[:,0]
imgs=targets[:,1]


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
vopt=fmin_powell(cost2,np.concatenate( (vopt,[1]) ))

# # 

v=vopt

z0=v[0]+1j*v[1]
a=v[2]+1j*v[3]
b=v[4]+1j*v[5]
c=v[6]
def s(z):
    return np.real(z)+1j*c*np.imag(z)
apply_xform(lambda z: z0+a*s(z)+b*s(z)**2 )

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
g.plot_edges(ax=ax)

ax.plot(np.real(imgs),np.imag(imgs),'ro')
ax.axis('equal')
ax.axis(xmin=-2,xmax=12,ymin=-2,ymax=12)
plt.draw()

#-# 

g.cells_center(refresh=True) # ,mode='sequential')

g.report_orthogonality()

# So a rough algorithm would be:
#  a non-orthogonal grid is generated based on linear
#  interpolation, completely defining the topology of
#  the grid.

# small (4?) groups of 4 constrained nodes are considered
# at a time, and a conformal map based on the above
# scaling is calculated.

# the coefficients can be shifted in zeta/phi space,
g_orig=g.copy()

## 

# even with all of that, still need a reasonably solid
# way of optimizing in the absence of regular quads.
g2=g.copy()

# constrain the corners:
constrained=[int(len(g2.node_to_edges(n))==2)
             for n in g2.valid_node_iter() ]
g2.add_node_field('constrained',np.array(constrained))

# # 

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
g2.plot_edges(ax=ax)

ax.scatter(g2.nodes['x'][:,0],
           g2.nodes['x'][:,1],
           30,g2.nodes['constrained'],lw=0)

plt.draw()

## 

# with the nearly optimal grid as starting point, but 70 dofs (35*2)
# 8800 function evaluations
# but it took a good 2 minutes

# 1. Can sympy help with speeding this up?
#    This gets tricky with the circumcenters - it gets involved
#    to write the function for circumcenters in terms of the inputs
#    1a: replicate the cost function in sympy
#    1b: have sympy calculate jacobian
# 2. Can this be applied to clumps of cells, treating the problem
#    more locally?
# 3. Can the cost functions be streamlined at all?
# 4. Is there a way of constructing better locations for nodes more
#    directly?

## 

g_test=g # modified_grid(g2,xopt)
g_test.report_orthogonality()
plt.figure(1).clf()
ax=plt.gca()
g_test.plot_edges()
g2.plot_edges(color='k')
    
vc=g_test.cells_center()
for c in g2.valid_cell_iter():
    ax.text(vc[c,0],vc[c,1],str(c),color='b')
# 
nx=g_test.nodes['x']
for n in g_test.valid_node_iter():
    if g2.nodes['constrained'][n]==C_FIXED:
        color='r'
    else:
        color='b'
    ax.text(nx[n,0],nx[n,1],str(n),color=color)


## 

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

# optimize based on the angles formed with the circumcenter

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

g.update_cell_edges()
n=29
nx0=g2.nodes['x'][n]
print ncostf(g,n,w_area=0.0,w_length=0,w_angle=0.01,w_cangle=0.0,verbose=1)(nx0)

## 

idx_movable=np.nonzero(movable)[0]

g_perturb=g_orig.copy()

g_perturb.nodes['x'][movable] += 0.5 * (np.random.random( g_perturb.nodes['x'][movable].shape )-0.5)

#g.nodes['x'][1,1]-=0.2
## 
g=g_perturb.copy()
## 
# the perturbed grid doesn't correct very well, though.
# max error in cell 10 of 0.2
# as a cell gets smaller, it weights the movement of a node
# greater since errors are normalized by radii
# one approach is to include area ratios for each link
# another is to just avoid normalization.  This means that it's
# not the same metric as in report_orthogonality

g_errs=g.circumcenter_errors()
g_orig_err=g_orig.circumcenter_errors()

for it in range(1):
    # does it even out more quickly with some noise?
    # g.nodes['x'][movable] += 0.1 * (np.random.random( g.nodes['x'][movable].shape ) - 0.5)

    reordered=np.argsort( np.random.random(len(idx_movable)) )
    for n in idx_movable[reordered]:
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
            print "Node=%d elapsed: %.3fs - delta %s, cost delta=%f"%(n,elapsed,nxopt-x0, c0-copt)
    g_errs_new=g.circumcenter_errors()

# # 

ax.cla()
g.plot_edges(ax=ax)
#coll=g_perturb.plot_nodes(ax=ax)
#coll.set_color('g')
#coll.set_lw(1)
#coll=g_orig.plot_edges(ax=ax)
#coll.set_color('k')
#coll.set_lw(0.2)

if 1:
    vc=g.cells_center()
    for c in g.valid_cell_iter():
        ax.text(vc[c,0],vc[c,1],str(c),color='b',size=7,ha='center',va='center')
    nx=g.nodes['x']
    for n in g.valid_node_iter():
        if g2.nodes['constrained'][n]==C_FIXED:
            color='r'
        else:
            color='b'
        ax.text(nx[n,0],nx[n,1],str(n),color=color,size=7)
# try:
#     ax.axis(z)
# except NameError:
#     pass
plt.draw()

# # 

circ_errs=g_orig.circumcenter_errors()
print "Original: max circ %f  mean circ %f"%(circ_errs.max(),circ_errs.mean())

circ_errs=g_perturb.circumcenter_errors()
print "Perturb: max circ %f  mean circ %f"%(circ_errs.max(),circ_errs.mean())

circ_errs=g.circumcenter_errors()
print "Updated: max circ %f  mean circ %f"%(circ_errs.max(),circ_errs.mean())

# so we're moving node 1, but the error in cell 4 is
# increasing??
# verified that only the position of node 1 has changed.
# oddly only the cell center for cell 1 has changed??
# probably because it's calculating centers based only on
# the first 3 nodes - node 1 is the 4th in cell 0, but
# the 1st in cell 1.

# comparing offsets - predictably, only the last offset of cell 0
# and all 4 offsets of cell 1 changed.

# manually going through this, they both emerge with the same max
# error of 0.03589805499655066, and the updated grid having a mean
# error slightly lower than the original grid
# so maybe the original grid wasn't getting its centers updated correctly?

# okay - boiled down to the different cell center methods

# Excellent - the local optimization is working well - can go from the
# nearly ortho original grid to something well within tolerance with
# 3 loops.  takes a few seconds.

# Hmm -
#  so the circum-angle is okay -- hard to really get it dialed in,
# though with some switching on/off, it gets reasonably close and
# at least it doesn't appear to have wild local minima - more that
# it fights a bit with the circumcenter error when things get close

# Some progress - the angle cost was only including 1/3 of the relevant
# angles.  Fixing that makes it work much better - such that with just
# the circumcenter cost and the angle cost it can get back to something
# better than the original grid after 2-3 iterations

# an occasional cangle step is useful for expanding the shapes - i.e.
# towards some consistency in aspect ratio.
# works okay to have it as a constant very low level factor, too.
# this gets the grid down to tolerable errors - max angle error of
# ~ 1.7deg, mean of 0.35.  and max around 2% circumcenter error.
# would be nice to be at 0.1deg and 0.1%.
