# v03: revisiting, since I skipped the centroid part of
# the original algorithm.
#  using the centroids of the prime to set nodes of the dual
#  at least converges
#  at the moment, seems like the centers of the dual aren't doing
#  much to alter the prime.

import numpy as np
import matplotlib.pyplot as plt

import utils
reload(utils)

import unstructured_grid
reload(unstructured_grid)
## 

p1=np.array([0.,0.])
p2=np.array([10.,10.])
nx=3
ny=3

g_prime= unstructured_grid.UnstructuredGrid()
gp_map = g_prime.add_rectilinear(p1,p2,nx,ny)

dxy=(p2-p1) / np.array([nx-1,ny-1])

g_dual=unstructured_grid.UnstructuredGrid()
gd_map = g_dual.add_rectilinear(p1-0.5*dxy,p2+0.5*dxy,nx+1,ny+1)

# perturb the original grid
mid_col_nodes=gp_map['nodes'][1,:]
g_prime.nodes['x'][mid_col_nodes,1]+=2

g_prime.cells_center(refresh=True,mode='sequential')

g_dual.add_node_field('constrained',np.zeros(g_dual.Nnodes(),'i4'))
g_prime.add_node_field('constrained',np.zeros(g_prime.Nnodes(),'i4'))

# constraint names:
C_NONE=0 # freely movable
C_FIXED=1 # static

# make the corners into triangles - this preserves cell numbering, though.
for n in [0,3,15,12]:
    g_dual.elide_node(n)

g_prime.nodes['constrained'][gp_map['nodes'][:-1,-1]]=C_FIXED
# g_prime.nodes['constrained'][gp_map['nodes'][1,1]]=C_FIXED
# g_prime.nodes['constrained'][gp_map['nodes'][:,0]]=C_FIXED

## 

def centers_to_nodes(src_grid,src_cell_map,
                     dst_grid,dst_node_map,
                     method='circumcenter'):
    # Update the dual based on circumcenters of the prime.
    for cx in range(src_cell_map.shape[0]): # nx-1 for prime->dual
        for cy in range(src_cell_map.shape[1]): # ny-1 for prime->dual
            p_c=src_cell_map[cx,cy]
            d_n=dst_node_map[cx,cy]
            if (p_c<0) or (d_n<0):
                continue
            if method=='circumcenter':
                dst_grid.nodes['x'][d_n]=src_grid.cells_center()[p_c]
            elif method=='centroid':
                poly=src_grid.cell_polygon(p_c)
                dst_grid.nodes['x'][d_n]=np.array(poly.centroid)
            else:
                assert False


def prime_to_dual():
    centers_to_nodes(g_prime,gp_map['cells'],
                     g_dual,gd_map['nodes'][1:-1,1:-1],
                     method='centroid')
    # update the centers of the dual:
    g_dual.cells_center(refresh=True,mode='sequential')

def dual_to_prime():
    nodes=gp_map['nodes'].copy()
    fixed=(g_prime.nodes['constrained'][nodes]==C_FIXED)
    nodes[fixed]=-1 # don't update some of them.

    centers_to_nodes(g_dual,gd_map['cells'],
                     g_prime,nodes)

    # g_prime.cells_center(refresh=True,mode='sequential')
    constrained_cells_center(g_prime)

# the problem seems to creep in where there are nodes of the
# prime which cannot be updated. but the dual keeps trucking
# along assuming that it's center is true.  so really we need
# some way to come back to nodes in the dual which are adjacent
# to a fixed circumcenter, and nudge them.

def update_dual_roaming(j,n_fixed,n_roam,j_prime):
    jp_seg=g_prime.nodes['x'][g_prime.edges['nodes'][j_prime]]
    jp_vec=jp_seg[1,:] - jp_seg[0,:]
    jp_vec=jp_vec / utils.mag(jp_vec)
    vec_perp=np.array([jp_vec[1],-jp_vec[0]])

    f_roam=lambda d: g_dual.nodes['x'][n_fixed] + d*vec_perp
    d0=np.dot(vec_perp,g_dual.nodes['x'][n_roam]-g_dual.nodes['x'][n_fixed])
    if d0<0: # clarify orientation
        d0*=-1
        vec_perp*=-1

    # rather than maintaining length of the original edge, better
    # to relationship with prime's nodes.
    # I think this equivalent to making the length double the
    # distance from n_fixed to the edge j_prime (or distance
    # to one of j_prime's nodes, projected onto vec_perp
    if 1:
        # preserve prime's nodes
        delta=jp_seg[0,:] - g_dual.nodes['x'][n_fixed]
        d_target=2*np.dot(delta,vec_perp)
        new_roam=f_roam(d_target)
    else:
        # preserves length of the edge:
        new_roam=f_roam(d0)

    # just move it halfway - seems like going all the way causes
    # oscillations
    new_roam=0.5*(new_roam+g_dual.nodes['x'][n_roam])
    g_dual.modify_node(n_roam,x=new_roam)

def update_all_roaming():
    update_dual_roaming(j=2,n_fixed=5,n_roam=1,j_prime=3)
    update_dual_roaming(j=19,n_fixed=9,n_roam=13,j_prime=8)
    update_dual_roaming(j=11,n_fixed=9,n_roam=8,j_prime=7)
    update_dual_roaming(j=1,n_fixed=5,n_roam=4,j_prime=0)
     
    update_dual_roaming(j=5,n_fixed=6,n_roam=2,j_prime=6)
    update_dual_roaming(j=7,n_fixed=6,n_roam=7,j_prime=5)
    update_dual_roaming(j=15,n_fixed=10,n_roam=11,j_prime=11)
    update_dual_roaming(j=21,n_fixed=10,n_roam=14,j_prime=10)


def constrained_cells_center(self):
    """ calculate cell centers giving preference to constrained
    nodes 
    """
    for c in np.arange(self.Ncells()):
        nodes=self.cell_to_nodes(c)
        constrained=(self.nodes['constrained'][nodes]==C_FIXED)
        points=self.nodes['x'][nodes]
        # for starters, only invoke the special logic when there
        # at least three constrained nodes
        ncon=np.sum(constrained)
        if ncon==0: # regular
            self.cells['_center'][c] = utils.poly_circumcenter(points)
        elif ncon==1:
            ci=np.nonzero(constrained)[0][0]
            nn=len(constrained)
            constrained[(ci+1)%nn]=True
            constrained[(ci-1)%nn]=True
            self.cells['_center'][c] = utils.poly_circumcenter(points[constrained])
        else:
            print ".",
            self.cells['_center'][c] = utils.poly_circumcenter(points[constrained])
    print
    return self.cells['_center']

# adjusting nodes of the dual to deal with constrained centers:
def nudge_dual_to_constrained(dual_c,prime_n):
    dual_ns=g_dual.cell_to_nodes(dual_c)
    center=g_prime.nodes['x'][prime_n]

    vecs=g_dual.nodes['x'][dual_ns] - center
    radii=utils.mag(vecs)
    mean_r=radii.mean()

    for n,rad,vec in zip(dual_ns,radii,vecs):
        print n,rad
        new_x=center+vec*mean_r/rad
        g_dual.nodes['x'][n]=new_x
    g_dual.cells_center(refresh=True,mode='sequential')

        
## 


if 1:
    for i in range(10):
        prime_to_dual()
        nudge_dual_to_constrained(dual_c=4,prime_n=4)
        dual_to_prime()
        update_all_roaming()

g_prime.report_orthogonality()

# # 
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g_prime.plot_edges(ax=ax)
g_dual.plot_edges(ax=ax,color='g')

vc=g_prime.cells_center(refresh=True,mode='sequential')
ax.plot(vc[:,0],vc[:,1],'b.')

vc=g_dual.cells_center()
ax.plot(vc[:,0],vc[:,1],'go')

sel=(g_prime.nodes['constrained']!=0)
ax.scatter(g_prime.nodes['x'][sel,0],
           g_prime.nodes['x'][sel,1],
           20,g_prime.nodes['constrained'][sel])


# Doesn't seem to converge to better than a 4.24deg max error.
# would it be better with more updates to the dual?

# doesn't converge when the length of the dual segments is
# kept constant.
# there is the possibility of adjusting the length to keep
# the dual's circumcenter coincident with the prime's corner
# vertex
# that's with the update sequence being prime_to_dual, update_roaming,
# dual_to_prime

# that at least converges.
# but the solution is not unique, and has some oscillations along
# the way.

# Next step: constrain more nodes

## 
ecp=g_prime.edges_center()
for j in g_prime.valid_edge_iter():
    ax.text(ecp[j,0],ecp[j,1],str(j),color='b')

    
ecd=g_dual.edges_center()
for j in g_dual.valid_edge_iter():
    ax.text(ecd[j,0],ecd[j,1],str(j),color='g')
    
for n in g_dual.valid_node_iter():
    ax.text(g_dual.nodes['x'][n,0],
            g_dual.nodes['x'][n,1],
            str(n),color='g')

for n in g_prime.valid_node_iter():
    ax.text(g_prime.nodes['x'][n,0],
            g_prime.nodes['x'][n,1],
            str(n),color='b')

vc=g_dual.cells_center()
for c in g_dual.valid_cell_iter():
    ax.text(vc[c,0],vc[c,1],str(c),color='g')


## first do it manually for a single edge
# update dual edges to be perpendicular to the prime
def update_dual_roaming(j,n_fixed,n_roam,j_prime):
    jp_seg=g_prime.nodes['x'][g_prime.edges['nodes'][j_prime]]
    jp_vec=jp_seg[1,:] - jp_seg[0,:]
    jp_vec=jp_vec / utils.mag(jp_vec)
    vec_perp=np.array([jp_vec[1],-jp_vec[0]])

    f_roam=lambda d: g_dual.nodes['x'][n_fixed] + d*vec_perp
    d0=np.dot(vec_perp,g_dual.nodes['x'][n_roam]-g_dual.nodes['x'][n_fixed])
    if d0<0: # clarify orientation
        d0*=-1
        vec_perp*=-1

    # rather than maintaining length of the original edge, better
    # to relationship with prime's nodes.
    # I think this equivalent to making the length double the
    # distance from n_fixed to the edge j_prime (or distance
    # to one of j_prime's nodes, projected onto vec_perp
    if 1:
        # preserve prime's nodes
        delta=g_dual.nodes['x'][n_fixed] - jp_seg[0,:]
        d_target=2*np.dot(delta,vec_perp)
        print d_target
        new_roam=f_roam(d_target)
    else:
        # preserves length of the edge:
        new_roam=f_roam(d0)

    g_dual.modify_node(n_roam,x=new_roam)

def update_all_roaming():
    update_dual_roaming(j=2,n_fixed=5,n_roam=1,j_prime=3)
    update_dual_roaming(j=19,n_fixed=9,n_roam=13,j_prime=8)
    update_dual_roaming(j=11,n_fixed=9,n_roam=8,j_prime=7)
    update_dual_roaming(j=1,n_fixed=5,n_roam=4,j_prime=0)

    update_dual_roaming(j=5,n_fixed=6,n_roam=2,j_prime=6)
    update_dual_roaming(j=7,n_fixed=6,n_roam=7,j_prime=5)
    update_dual_roaming(j=15,n_fixed=10,n_roam=11,j_prime=11)
    update_dual_roaming(j=21,n_fixed=10,n_roam=14,j_prime=10)

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
from scipy.optimize import fmin,fmin_cg,fmin_bfgs,fmin_powell

if 0:
    # not so great
    alpha=0.5 # low means favor area, high means favor angle
    beta=0.5 # weighting for cell-centered costs
    gamma=0.5 # weighting for circumcenter vs. distance from edge
if 0:
    # okay, but not with powell
    alpha=0.5 # low means favor area, high means favor angle
    beta=0.25 # weighting for cell-centered costs
    gamma=0.5 # weighting for circumcenter vs. distance from edge
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
if 0:
    # looks good, though angle error not quashed very effectively
    alpha=0.75 # low means favor area, high means favor angle
    beta=0.75 # weighting for cell-centered costs
    gamma=0.5 # weighting for circumcenter vs. distance from edge
if 0:
    # similar, but worse with the angles
    alpha=0.75 # low means favor area, high means favor angle
    beta=0.75 # weighting for cell-centered costs
    gamma=0.1 # weighting for circumcenter vs. distance from edge
if 0:
    # really bad
    alpha=0.75 # low means favor area, high means favor angle
    beta=0.75 # weighting for cell-centered costs
    gamma=0.9 # weighting for circumcenter vs. distance from edge
if 0:
    # not so great either
    alpha=0.5 # low means favor area, high means favor angle
    beta=0.75 # weighting for cell-centered costs
    gamma=0.0 # weighting for circumcenter vs. distance from edge
if 0:
    # not so great either
    alpha=0.5 # low means favor area, high means favor angle
    beta=0.25 # weighting for cell-centered costs
    gamma=0.0 # weighting for circumcenter vs. distance from edge

#xopt=fmin_cg(cost,xopt) # didn't work at all. likewise bfgs
# xopt=fmin(cost,xopt) # slow, but eventually
xopt=fmin_powell(cost,xopt) # slow, but works very well in one go.


# # 
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
# nx=g_prime.nodes['x']
# for n in g_prime.valid_node_iter():
#     ax.text(nx[n,0],nx[n,1],str(n),color='b')

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
