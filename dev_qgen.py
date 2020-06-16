# Building up a quad grid from patches
from stompy.grid import unstructured_grid
from stompy import utils
from stompy.spatial import field
from shapely import geometry
from scipy.interpolate import griddata, Rbf
import numpy as np
from stompy.grid import orthogonalize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import six
##

six.moves.reload_module(unstructured_grid)

# similar codes as in front.py
FREE=0 # default
RIGID=1
SLIDE=2

# generating grid
gen=unstructured_grid.UnstructuredGrid(max_sides=150,
                                       extra_node_fields=[('ij',np.int32,2)],
                                       extra_edge_fields=[('dij',np.int32,2)])

if 0:
    # Simple test case - rectangle
    gen.add_rectilinear([0,0],[100,200],2,2)
    gen.nodes['ij']= (gen.nodes['x']/10).astype(np.int32)

    # test slight shift in xy:
    sel=gen.nodes['x'][:,1]>100
    gen.nodes['x'][sel, 0] +=30
    # and in ij
    gen.nodes['ij'][sel,0] += 2

    # specifying ij on nodes is a convenience, but dij on edegs
    # dictates actual grid layout
    gen.edges['dij']=(gen.nodes['ij'][gen.edges['nodes'][:,1]]
                      - gen.nodes['ij'][gen.edges['nodes'][:,0]])
else:
    # Slightly more complex: convex polygon
    # 6 - 4
    # | 2 3
    # 0 1
    n0=gen.add_node(x=[0,0]    ,ij=[0,0])
    n1=gen.add_node(x=[100,0]  ,ij=[10,0])
    n2=gen.add_node(x=[100,205],ij=[10,20])
    n3=gen.add_node(x=[155,190],ij=[15,20])
    n4=gen.add_node(x=[175,265],ij=[15,30])
    n6=gen.add_node(x=[0,300]  ,ij=[0,30])
    
    gen.add_cell_and_edges(nodes=[n0,n1,n2,n3,n4,n6])

    # specifying ij on nodes is a convenience, but dij on edegs
    # dictates actual grid layout
    gen.edges['dij']=(gen.nodes['ij'][gen.edges['nodes'][:,1]]
                      - gen.nodes['ij'][gen.edges['nodes'][:,0]])
    
# target grid
g=unstructured_grid.UnstructuredGrid(max_sides=4,
                                     extra_node_fields=[('ij',np.int32,2),
                                                        ('rigid',np.int32)])

for c in gen.valid_cell_iter():
    ijs=[]
    ns=gen.cell_to_nodes(c)
    xys=gen.nodes['x'][ns]
    ijs=[ np.array([0,0]) ]
    for j in gen.cell_to_edges(c,ordered=True):
        if gen.edges['cells'][j,0]==c:
            s=1
        elif gen.edges['cells'][j,1]==c:
            s=-1
        else: assert 0
        ijs.append( ijs[-1]+s*gen.edges['dij'][j] )
    assert np.all( ijs[0]==ijs[-1] )

    ijs=np.array(ijs[:-1])
    ijs-=ijs.min(axis=0)

    # Create in ij space
    patch=g.add_rectilinear(p0=[0,0],
                            p1=ijs.max(axis=0),
                            nx=1+ijs[:,0].max(),
                            ny=1+ijs[:,1].max())
    pnodes=patch['nodes'].ravel()

    # Copy xy to ij, then remap xy
    g.nodes['ij'][pnodes] = g.nodes['x'][pnodes]

    Extrap=utils.LinearNDExtrapolator
    
    int_x=Extrap(ijs,xys[:,0])
    node_x=int_x(g.nodes['x'][pnodes,:])

    int_y=Extrap(ijs,xys[:,1])
    node_y=int_y(g.nodes['x'][pnodes,:])

    g.nodes['x'][pnodes]=np.c_[node_x,node_y]
    
    # delete cells that fall outside of the ij
    for n in pnodes[ np.isnan(node_x) ]:
        g.delete_node_cascade(n)

    # Mark nodes as rigid if they match a point in the generator
    for n in g.valid_node_iter():
        match0=gen.nodes['ij'][:,0]==g.nodes['ij'][n,0]
        match1=gen.nodes['ij'][:,1]==g.nodes['ij'][n,1]
        match=np.nonzero(match0&match1)[0]
        if len(match):
            g.nodes['rigid'][n]=RIGID
            
    ij_poly=geometry.Polygon(ijs)
    for c in patch['cells'].ravel():
        if g.cells['deleted'][c]: continue
        cn=g.cell_to_nodes(c)
        c_ij=np.mean(g.nodes['ij'][cn],axis=0)
        if not ij_poly.contains(geometry.Point(c_ij)):
            g.delete_cell(c)

    # This part will need to get smarter when there are multiple patches:
    g.delete_orphan_edges()
    g.delete_orphan_nodes()
    
g.renumber()

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
gen.plot_edges(lw=1.5,color='b',ax=ax)
g.plot_edges(lw=0.5,color='k',ax=ax)
g.plot_nodes(mask=g.nodes['rigid']>0) # good
ax.axis('equal')

##

# Need to force the corners to be 90deg angles, otherwise
# there's no hope of getting orthogonal cells in the interior.

# Generate bezier control points for each edge.

def add_bezier(gen):
    order=3 # cubic bezier curves
    bez=np.zeros( (gen.Nedges(),order+1,2) )
    bez[:,0,:] = gen.nodes['x'][gen.edges['nodes'][:,0]]
    bez[:,order,:] = gen.nodes['x'][gen.edges['nodes'][:,1]]

    gen.add_edge_field( 'bez',bez,on_exists='replace' )

    for n in gen.valid_node_iter():
        js=gen.node_to_edges(n)
        assert len(js)==2
        # orient the edges
        njs=[]
        deltas=[]
        dijs=[]
        flips=[]
        for j in js:
            nj=gen.edges['nodes'][j]
            dij=gen.edges['dij'][j]
            flip=0
            if nj[0]!=n:
                nj=nj[::-1]
                dij=-dij
                flip=1
            assert nj[0]==n
            njs.append(nj)
            dijs.append(dij)
            flips.append(flip)
            deltas.append( gen.nodes['x'][nj[1]] - gen.nodes['x'][nj[0]] )
        # the angle in ij space tells us what it *should* be
        theta0_ij=np.arctan2(dijs[0][1],dijs[0][0])
        theta1_ij=np.arctan2(dijs[1][1],dijs[1][0])
        dtheta_ij=(theta1_ij - theta0_ij + np.pi) % (2*np.pi) - np.pi

        theta0=np.arctan2(deltas[0][1],deltas[0][0])
        theta1=np.arctan2(deltas[1][1],deltas[1][0])
        dtheta=(theta1 - theta0 + np.pi) % (2*np.pi) - np.pi

        theta_err=dtheta-dtheta_ij
        theta0_adj = theta0+theta_err/2
        theta1_adj = theta1-theta_err/2

        cp0 = gen.nodes['x'][n] + utils.rot( theta_err/2, 1./3 * deltas[0] )
        cp1 = gen.nodes['x'][n] + utils.rot( -theta_err/2, 1./3 * deltas[1] )

        # save to the edge
        gen.edges['bez'][js[0],1+flips[0]] = cp0
        gen.edges['bez'][js[1],1+flips[1]] = cp1

add_bezier(gen)        
## 

# HERE:
# the bezier boundary looks decent.
#  - I think I can use this directly to clip and set BCs
#    for the diffusion problem.

# potential recipe:
#   upsample

gen_up=gen.copy()

# 0 is fine
for j in gen.valid_edge_iter():
    n0=gen.edges['nodes'][j,0]
    nN=gen.edges['nodes'][j,1]
    bez=gen.edges['bez'][j]
    dij=gen.edges['dij'][j]
    steps=int(utils.mag(dij))
    t=np.linspace(0,1,1+steps)

    B0=(1-t)**3
    B1=3*(1-t)**2 * t
    B2=3*(1-t)*t**2
    B3=t**3
    
    points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]
    dij_up=dij/steps

    j_split=j
    for i in range(1,len(points)-1):
        j_new,n_new,next_split = gen_up.split_edge(j_split,x=points[i],split_cells=False)
        # Is the next edge to split j_split or j_new?
        # The bezier curve goes in the direction of the edge
        if nN in gen_up.edges['nodes'][j_new]:
            j_split=j_new
        elif nN in gen_up.edges['nodes'][j_split]:
            pass
        else:
            raise Exception('what?')


plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
#gen.plot_edges(lw=0.5,color='b',ax=ax,alpha=0.2)
gen_up.plot_edges(lw=0.5,color='b',ax=ax,alpha=0.5)
gen_up.plot_nodes()
gen_up.plot_cells(color='0.7') 

paths=[]

for bz in gen.edges['bez']:
    p=Path(bz,[Path.MOVETO,Path.CURVE4, Path.CURVE4, Path.CURVE4])
    pp=PathPatch(p,transform=ax.transData,fc='none',ec='r',lw=2.0)
    ax.add_patch(pp)

ax.axis('equal')

g=gen_up
##

# What would it look like to solve a pair of laplacians
# on the grid to get a psi/phi coordinate system?
# And does that help?

# Create a rectilinear grid covering the distorted grid
six.moves.reload_module(unstructured_grid)

dx=2.0

xmin,xmax,ymin,ymax = g.bounds()
gr=unstructured_grid.UnstructuredGrid(max_sides=4)
gr.add_rectilinear(p0=[xmin,ymin],p1=[xmax,ymax],
                   nx=int( (xmax-xmin)/dx )+1,
                   ny=int( (ymax-ymin)/dx )+1)

sel=gr.select_cells_intersecting( g.boundary_polygon() )

for c in np.nonzero(~sel)[0]:
    gr.delete_cell(c)

gr.delete_orphan_edges()
gr.delete_orphan_nodes()
gr.renumber()

##

gd=gr.create_dual(create_cells=True)

##

from stompy.model import unstructured_diffuser

diff_i=unstructured_diffuser.Diffuser(gr)
diff_j=unstructured_diffuser.Diffuser(gr)

def to_cells(nodes):
    return np.unique( [ c for n in nodes for c in gr.node_to_cells(n) ] )

for e in gen.valid_edge_iter():
    sel_nodes=gr.select_nodes_boundary_segment( coords=gen.nodes['x'][gen.edges['nodes'][e]] )
    sel_cells=to_cells(sel_nodes)

    dij=gen.edges['dij'][e]

    if dij[0]==0: # i is constant
        i=gen.nodes['ij'][e,0]
        for c in sel_cells:
            diff_i.set_dirichlet(i,cell=c)
    elif dij[1]==0: # j is constant 
        j=gen.nodes['ij'][e,1]
        for c in sel_cells:
            diff_j.set_dirichlet(j,cell=c)

for diff in [diff_i,diff_j]:            
    diff.construct_linear_system()
    diff.solve_linear_system(animate=False)

i_soln=diff_i.C_solved
j_soln=diff_j.C_solved

i_node=i_soln[ gd.nodes['dual_cell'] ]
j_node=j_soln[ gd.nodes['dual_cell'] ]

## 
plt.figure(10).clf()
fig,ax=plt.subplots(1,1,num=10)

gd.contour_node_values(i_node,20,linewidths=1.5,colors='orange')
gd.contour_node_values(j_node,30,linewidths=1.5,colors='red')

ax.axis('equal')

# Basic idea, but need to rethink the boundary conditions, and
# how to better deal with corners.  I.e. the edges should be 
# more like bezier curves, where the angles at corners respect
# the dij of the two edges.
# And may have to figure out how to solve with specified normal

gr.plot_edges(ax=ax,color='0.5',lw=0.5,alpha=0.5)

##

# There must be some additional constraints on how the the BCs
# are chosen that dictate whether it works.  Thinking back to
# the HOR case, I had to tweak the velocity potential in order
# to get a reasonable grid.

# One approach would be
# 1. calculate the psi field,
# 2. calculate u and v throughout the domain from psi
# 3. calculate phi by breadth first search on u,v
# 4. In theory then we're done.  But could further use that to
#    get BCs and solve phi again.

##

# Get the rectilinear field:
j_fld=field.XYZField(X=gr.cells_center(), F=j_soln).rectify()
i_fld=field.XYZField(X=gr.cells_center(), F=i_soln).rectify()

##
# integrate

ij_to_c=field.XYZField(gr.cells_center(),F=np.arange(gr.Ncells())).rectify()
C=ij_to_c.F
dx=ij_to_c.dx
dy=ij_to_c.dy
valid=np.isfinite(C)
C=C.astype(np.int32)
C[~valid]=-1

cells=[]
xrows=[]
yrows=[]

from scipy import sparse

Nc=gr.Ncells()

Mx=sparse.dok_matrix( (Nc,Nc), np.float64 )
My=sparse.dok_matrix( (Nc,Nc), np.float64 )

i_recon=np.zeros(Nc) # looks fine - i increasing w/ +y
j_recon=np.zeros(Nc) # ditto. +x


for i,j in zip(*np.nonzero(valid)):
    c=C[i,j]
    assert c>=0
    i_recon[c]=i
    j_recon[c]=j
    # y derivative
    im=max(0,i-1)
    ip=min(C.shape[0]-1,i+1)
    if C[im,j]<0: im=i
    if C[ip,j]<0: ip=i
    if im<ip:
        My[c,C[ip,j]]=1./(dy*(ip-im))
        My[c,C[im,j]]=-1./(dy*(ip-im))
    else:
        print("!")
    # x derivative
    jm=max(0,j-1)
    jp=min(C.shape[1]-1,j+1)
    if C[i,jm]<0: jm=j
    if C[i,jp]<0: jp=j
    if jm<jp:
        Mx[c,C[i,jp]]=1./(dx*(jp-jm))
        Mx[c,C[i,jm]]=-1./(dx*(jp-jm))
    else:
        print("!")
    
u2=-My.dot(i_soln)
v2=Mx.dot(i_soln)

##

uv=np.concatenate( [u2,v2] )

Mxy=sparse.vstack( [Mx,My] )

phi,*rest=sparse.linalg.lsqr(Mxy,uv)


phi_fld=field.XYZField(gr.cells_center(),F=phi).rectify()

## 

plt.figure(10).clf()
i_fld.contour(20)

phi_fld.contour(20)
plt.axis('equal')

# Calculate velocities on F
psi=i_fld.F
# This isn't going to do well at boundaries
dpsi_dx,dpsi_dy = i_fld.gradient()

u=dpsi_dy ; u.F*=-1
v=dpsi_dx

X,Y=i_fld.XY()
valid=np.isfinite(u.F + v.F)

valid[::2]=False
valid[:,::2]=False

#plt.quiver( X[valid], Y[valid], u.F[valid], v.F[valid],scale=2)

#cc=gr.cells_center()
#plt.quiver( cc[:,0], cc[:,1], u2,v2,color='b', scale=2)
# gr.plot_cells(values=phi,cmap=turbo)





##
# try the conformal approaches.
import dev_smooth
six.moves.reload_module(dev_smooth)

rigid=g.nodes['rigid']==RIGID

nodes=np.array(list(g.valid_node_iter()))

# local fits were not great

if 1: # global fit just to rigid nodes
    node_idxs=nodes
    fixed=rigid & (g.nodes['x'][:,1]>100)
    #fixed=rigid & (g.nodes['x'][:,0]<110)
    #fixed=rigid
    free_nodes=nodes[ ~fixed ]
    fit_nodes= nodes[  fixed ] 

    dev_smooth.conformal_smooth(g,node_idxs=node_idxs,
                                free_nodes=free_nodes,
                                fit_nodes=fit_nodes,
                                ij=g.nodes['ij'][node_idxs],
                                halo=[0,2],
                                max_update=1.0)

plt.figure(1).clf()
gen.plot_edges(lw=1.5,color='b')
g.plot_edges(lw=0.5,color='k')
plt.axis('equal')


##
opt_vec=np.zeros( (len(free_nodes),2), np.float64)
free_nodes=np.nonzero( g.nodes['rigid']==FREE )[0]
tweaker=orthogonalize.Tweaker(g)

##
for i in range(1):
    x0=g.nodes['x'][free_nodes].copy()
    for ni,n in enumerate(free_nodes):
        g.modify_node(n,x=g.nodes['x'][n]+0.5*opt_vec[ni])

    tweaker.local_smooth(node_idxs=nodes,
                         free_nodes=free_nodes,
                         ij=g.nodes['ij'][nodes],
                         stencil_radius=2,
                         n_iter=2,
                         min_halo=0)

    for _ in range(5):
        for ni,n in enumerate(free_nodes):
            tweaker.nudge_node_orthogonal(n)

    opt_vec[:]=g.nodes['x'][free_nodes] - x0

plt.figure(1).clf()
gen.plot_edges(lw=1.5,color='b')
g.plot_edges(lw=0.5,color='k')
plt.axis('equal')

print("Max delta: ",utils.mag(opt_vec).max())
