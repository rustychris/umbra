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

## 

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


## 

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
g2.plot_edges(ax=ax)

ax.scatter(g2.nodes['x'][:,0],
           g2.nodes['x'][:,1],
           30,g2.nodes['constrained'],lw=0)

plt.draw()


## 

alpha=0.5 # low means favor area, high means favor angle
beta=0.25 # weighting for cell-centered costs
gamma=0.5 # weighting for circumcenter vs. distance from edge

def link_cost_angle(g,j):
    vc=g.cells_center()

    c1,c2=g.edges['cells'][j]
    n1,n2=g.edges['nodes'][j]

    assert c1>=0
    assert c2>=0

    vec_norm=vc[c2]-vc[c1]
    vec_tan =g.nodes['x'][n1]- g.nodes['x'][n2]

    vec_norm=vec_norm/utils.mag(vec_norm)
    vec_tan =vec_tan/utils.mag(vec_tan)

    parallel=(vec_norm*vec_tan).sum()
    #return np.abs(parallel) # |cos(theta)|
    # maybe this is better behaved?
    return parallel**2

def link_cost_area(g,j):
    Ac=g.cells_area()[g.edges['cells'][j]]
    return np.abs(Ac[0] - Ac[1]) / Ac.mean()

def link_cost(g,j):
    cost=0
    if alpha>0:
        cost+=alpha*link_cost_angle(g,j)
    if alpha<1:
        cost+= (1-alpha)*link_cost_area(g,j)
    return cost

def all_links_cost(g):
    internal_edges=np.nonzero( np.all(g.edges['cells']>=0,axis=1) )[0]
    costs=[link_cost(g,j) for j in internal_edges]
    return np.sum(costs)

def cell_edges_signed_distance(g,c):
    vc=g.cells_center()[c]
    nodes=g.cell_to_nodes(c)
    L=np.sqrt(g.cells_area()[c])
    dists=[]
    for a,b in utils.circular_pairs(g.nodes['x'][nodes]):
        ab=b-a
        ab/=utils.mag(ab)
        norm=[-ab[1],ab[0]]
        c_ab=np.dot(vc-a,norm)
        dists.append( c_ab/L )
    return np.array(dists)

def cell_clearance_cost(g,c):
    return (
        cell_edges_signed_distance(g,c)**(-2)
    ).mean()

def cell_circumcenter_cost(g,c):
    nodes=g.cell_to_nodes(c)
    nx=g.nodes['x'][nodes]
    vc=g.cells_center()[c]
    # ad-hoc choice of powers here, skipping the sqrt
    dists2=((vc-nx)**2).sum(axis=1)
    return np.std(dists2)/np.mean(dists2)

def cell_cost(g,c):
    return gamma*cell_circumcenter_cost(g,c) + (1-gamma)*cell_clearance_cost(g,c)

def all_cells_cost(g):
    costs=[cell_cost(g,c) for c in g.valid_cell_iter()]
    return np.sum(costs)

def modified_grid(g,x):
    g_test=g.copy()
    xys=x.reshape((-1,2))
    g_test.nodes['x'][movable]=xys
    g_test.cells_center(refresh=True,mode='sequential')
    g_test.cells['_area']=np.nan
    return g_test

def cost(x):
    gmod=modified_grid(g2,x)
    cost=0
    if beta<1:
        cost+=(1-beta)*all_links_cost(gmod)
    if beta>0:
        cost+=beta*all_cells_cost(gmod) 
    return cost


# angle cost on the larger test case needs better control
# on aspect ratio
movable=g2.nodes['constrained']==C_NONE
x0=g2.nodes['x'][movable].ravel()
print cost(x0)
xopt=x0


## 

if 0:
    # maybe the best, including with powell
    alpha=0.75 # low means favor area, high means favor angle
    beta=0.25 # weighting for cell-centered costs
    gamma=0.5 # weighting for circumcenter vs. distance from edge
if 0:
    # if the topology is dialed in, don't really need the
    # area cost
    # this takes 76s, 6192 evaluations
    # cost of 12.56 -> 12.23
    # achieves max circ error 0.001 and angle of 0.04
    alpha=1.0 # low means favor area, high means favor angle
    beta=0.25 # weighting for cell-centered costs
    gamma=0.5 # weighting for circumcenter vs. distance from edge
if 1:
    # this took 340s, with 26k function evaluations
    # cost went from 0.39 to 0.00035, and ended with 0.0004 max
    # circum error and 0.00 deg angle error
    # I made some change, and then it took 565s and 45k evals.
    # then changed the abs to be square...
    # that made starting cost 0.00631, down to 1e-21
    # with 26k evaluations over 132s
    alpha=1.0 # low means favor area, high means favor angle
    beta=0.0 # weighting for cell-centered costs
    gamma=1.0 # weighting for circumcenter vs. distance from edge

print "Starting cost: ",cost(x0)
t=time.time()
xopt=fmin_powell(cost,xopt) # slow, but works very well in one go.
elapsed=time.time()-t
print "Ending cost: ",cost(xopt)
print "%.3fs"%elapsed

g_test=modified_grid(g2,xopt)
g_test.report_orthogonality()

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


# clumping - optimize a single node at a time
def ncostf(g,n):
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
            centers=g_mod.cells_center()
            # this is somehow making the mean and max circumcenter error larger!
            offsets = g_mod.nodes['x'][g_mod.cells['nodes'][cells,:nsi]] - centers[cells,None,:]
            dists = utils.mag(offsets)
            error = np.std(dists,axis=1) / np.mean(dists,axis=1)
            cost+= error.mean()
        
        return cost
    return cost

g.update_cell_edges()
nx0=g2.nodes['x'][n]
print ncostf(g,n)(nx0)

## 

idx_movable=np.nonzero(movable)[0]

g=g_orig.copy()

#g.nodes['x'][1,1]-=0.2

g.report_orthogonality()
g_orig.report_orthogonality()
for it in range(3):
    for n in idx_movable:
        t=time.time()
        x0=g.nodes['x'][n].copy()
        costf=ncostf(g,n)
        c0=costf(x0)
        nxopt=fmin(ncostf(g,n),x0)
        copt=costf(nxopt)
        if copt>=c0:
            print "--------- OPT FAILED ---------"
        else:
            g.nodes['x'][n]=nxopt
            g.cells_center(refresh=True)
        elapsed=time.time()-t
        print "Node=%d elapsed: %.3fs - delta %s, cost delta=%f"%(n,elapsed,nxopt-x0, c0-copt)

## 

ax.cla()
g.plot_edges(ax=ax)
coll=g_orig.plot_edges(ax=ax)
coll.set_color('k')
coll.set_lw(0.2)


vc=g.cells_center()
for c in g.valid_cell_iter():
    ax.text(vc[c,0],vc[c,1],str(c),color='b')
nx=g.nodes['x']
for n in g.valid_node_iter():
    if g2.nodes['constrained'][n]==C_FIXED:
        color='r'
    else:
        color='b'
    ax.text(nx[n,0],nx[n,1],str(n),color=color)


print "Original ---"
g_orig.report_orthogonality()
print "Updated ---"
g.report_orthogonality()

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
