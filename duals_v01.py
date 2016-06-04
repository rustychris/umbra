import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin,fmin_cg,fmin_bfgs,fmin_powell

import utils
reload(utils)

import unstructured_grid
reload(unstructured_grid)
## 

# how about a cost function, like the triangle case?
# if possible, make it continuous

p1=np.array([0.,0.])
p2=np.array([10.,10.])
nx=3
ny=3

g_prime= unstructured_grid.UnstructuredGrid()
gp_map = g_prime.add_rectilinear(p1,p2,nx,ny)

# perturb the original grid
mid_col_nodes=gp_map['nodes'][1,:]
g_prime.nodes['x'][mid_col_nodes,1]+=2
g_prime.cells_center(refresh=True,mode='sequential')
g_prime.add_node_field('constrained',np.zeros(g_prime.Nnodes(),'i4'))

g_prime.nodes['constrained'][gp_map['nodes'][:,-1]]=C_FIXED
g_prime.nodes['constrained'][gp_map['nodes'][1,1]]=C_FIXED
g_prime.update_cell_edges()
g_prime.edge_to_cells(recalc=True)

## 

plt.figure(1).clf()
ax=plt.gca()
g_prime.plot_edges()

ecp=g_prime.edges_center()
for j in g_prime.valid_edge_iter():
    ax.text(ecp[j,0],ecp[j,1],str(j),color='b')

vc=g_prime.cells_center()
for c in g_prime.valid_cell_iter():
    ax.text(vc[c,0],vc[c,1],str(c),color='b')

nx=g_prime.nodes['x']
for n in g_prime.valid_node_iter():
    ax.text(nx[n,0],nx[n,1],str(n),color='b')
    
## 

# NOTES:
#  The orthogonality condition is local, while the conformal
#  condition is... less local.

# cost function -
# for each link, how close to perpendicular?
# and what is the ratio of the cell sizes?
g=g_prime
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
    return np.abs(parallel) # |cos(theta)|

def link_cost_area(g,j):
    Ac=g.cells_area()[g.edges['cells'][j]]
    return np.abs(Ac[0] - Ac[1]) / Ac.mean()

def link_cost(g,j):
    return alpha*link_cost_angle(g,j) + (1-alpha)*link_cost_area(g,j)

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

all_links_cost(g_prime)


movable=g_prime.nodes['constrained']==C_NONE

def modified_grid(g,x):
    g_test=g.copy()
    xys=x.reshape((-1,2))
    g_test.nodes['x'][movable]=xys
    g_test.cells_center(refresh=True,mode='sequential')
    g_test.cells['_area']=np.nan
    return g_test


def cost(x):
    gmod=modified_grid(g_prime,x)
    return ( (1-beta)*all_links_cost(gmod)
             + beta*all_cells_cost(gmod) )

x0=g_prime.nodes['x'][movable].ravel()

xx=x0.copy()
xx[0]+=1


print all_links_cost(g_prime)
print cost(x0)
print cost(xx)
xopt=x0
## 

if 0:
    # maybe the best, including with powell
    alpha=0.75 # low means favor area, high means favor angle
    beta=0.25 # weighting for cell-centered costs
    gamma=0.5 # weighting for circumcenter vs. distance from edge
if 1:
    # maybe the best, including with powell
    alpha=0.75 # low means favor area, high means favor angle
    beta=0.25 # weighting for cell-centered costs
    gamma=0.5 # weighting for circumcenter vs. distance from edge

#xopt=fmin_cg(cost,xopt) # didn't work at all. likewise bfgs
# xopt=fmin(cost,xopt) # slow, but eventually
xopt=fmin_powell(cost,xopt) # slow, but works very well in one go.


## 
g_test=modified_grid(g_prime,xopt)
g_test.report_orthogonality()
plt.figure(1).clf()
ax=plt.gca()
g_test.plot_edges()

# ecp=g_prime.edges_center()
# for j in g_prime.valid_edge_iter():
#     ax.text(ecp[j,0],ecp[j,1],str(j),color='b')
# 
vc=g_test.cells_center()
for c in g_prime.valid_cell_iter():
    ax.text(vc[c,0],vc[c,1],str(c),color='b')
# 
nx=g_test.nodes['x']
for n in g_test.valid_node_iter():
    if g_prime.nodes['constrained'][n]==C_FIXED:
        color='r'
    else:
        color='b'
    ax.text(nx[n,0],nx[n,1],str(n),color=color)

##   

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
#  - do more of this on the dual?
#  - focus on the edges and nodes, making angles approach 90.

