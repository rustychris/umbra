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
#gen_up.plot_cells(color='0.7') 

paths=[]

for bz in gen.edges['bez']:
    p=Path(bz,[Path.MOVETO,Path.CURVE4, Path.CURVE4, Path.CURVE4])
    pp=PathPatch(p,transform=ax.transData,fc='none',ec='r',lw=2.0)
    ax.add_patch(pp)

ax.axis('equal')

g=gen_up
##

# Like gen_up, but modifying g
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

    for i in range(1,len(points)-1):
        ij0=gen.nodes['ij'][n0]
        ijN=gen.nodes['ij'][nN]
        ij_i=ij0+i*dij_up
        
        # j_new,n_new,next_split = gen_up.split_edge(j_split,x=points[i],split_cells=False)
        n=np.nonzero( np.all( g.nodes['ij']==ij_i, axis=1))[0][0]
        g.modify_node(n,x=points[i])

def smooth_interior_quads(g):
    # redistribute interior nodes evenly
    # by solving laplacian
    M=sparse.dok_matrix( (g.Nnodes(),g.Nnodes()), np.float64)
    rhs_nodes=-1*np.ones(g.Nnodes(),np.int32)
    for n in g.valid_node_iter():
        if g.is_boundary_node(n):
            M[n,n]=1
            rhs_nodes[n]=n
        else:
            nbrs=g.node_to_nodes(n)
            M[n,n]=-len(nbrs)
            for nbr in nbrs:
                M[n,nbr]=1
    rhs_x=np.where( rhs_nodes<0, 0, g.nodes['x'][rhs_nodes,0])
    new_x=sparse.linalg.spsolve(M.tocsr(),rhs_x)
    rhs_y=np.where( rhs_nodes<0, 0, g.nodes['x'][rhs_nodes,1])
    new_y=sparse.linalg.spsolve(M.tocsr(),rhs_y)

    g.nodes['x'][:,0]=new_x
    g.nodes['x'][:,1]=new_y

smooth_interior_quads(g)    
plt.figure(3).clf()
g.plot_edges()
plt.axis('equal')

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


##

# Regroup:
#  The flow-net approach seems good
#  - it does not support self-joins that are not themselves conformal,
#    that would have to be done patch by patch.
#  - likewise, no support at this point for holes in the polygon, would
#    have to split up into patches
#  - the solution method is slow.

# Two thoughts on improving solution method:
#  - construct the rough quad grid as I'm already doing, solve
#    the laplace eqns on that grid with a finite-difference approach
#    that makes no assumptions on cell geometry.  Probably solve on
#    nodes and edges so we have a full field
#  - still solve on regular cartesian grid, but directly via numpy,
#    not through unstructured.

# What would the math look like to do this finite difference on the
# unstructued grid?

# Does it help to construct the dual?
# Normal dual construction assumes starting grid is orthogonal.
ggd=g.create_dual(create_cells=False)


##


from scipy.spatial import Voronoi,voronoi_plot_2d

## 
vor=Voronoi(g.nodes['x'])
#voronoi_plot_2d(vor,ax=ax)

# Consider vertices outside the original grid to all be infinite,

from shapely import geometry
boundary=g.boundary_polygon()

fin_vertices=np.array([boundary.contains(geometry.Point(v[0],v[1]))
                       for v in vor.vertices])
max_sides=max([len(c) for c in vor_cells])
# mark empty with -2 to distinguish from infinite==-1
cells=-2*np.ones( (len(vor.regions),max_sides), np.int32)
for ci,c in enumerate(vor_cells):
    cells[ci,:len(c)]=c

cells=np.where( (cells>=0) & (~fin_vertices[cells.clip(0)]),
                -1, cells)
valid=np.all(cells!=-1,axis=1) & (cells.max(axis=1)>=0)
cells=cells[valid]

##     
ggd=unstructured_grid.UnstructuredGrid(max_sides=max_sides,
                                       points=vor.vertices,
                                       cells=cells.clip(-1))
ggd.make_edges_from_cells()

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
g.plot_edges(lw=0.5,color='k',ax=ax)
g.plot_nodes(labeler='id')
ax.axis('equal')

#ggd.plot_edges()
#ggd.plot_cells(alpha=0.4)

# HERE
#  Do I know how to set the BCs properly?
#  the laplacian is easy enough with the orthogonal
#  grid.
##

# try this old  paper:
# https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19890000871.pdf

# select a sample node
# this operates on the nodes of g
# for starters use the existing edges, but in general will calculate
# a proper tinagulation

#n0=189
#is_boundary=0

from scipy import sparse
class NodeDiscretization(object):
    def __init__(self,g):
        self.g=g
        # self.dirichlet_nodes={}
    def construct_matrix(self,op='laplacian',dirichlet_nodes={}):
        B=np.zeros(g.Nnodes(),np.float64)
        M=sparse.dok_matrix( (g.Nnodes(),g.Nnodes()),np.float64)

        for n in range(g.Nnodes()):
            if n in dirichlet_nodes:
                nodes=[n]
                alphas=[1]
                rhs=dirichlet_nodes[n]
            else:
                nodes,alphas,rhs=self.node_discretization(n,op=op)
                # could add to rhs here
            B[n]=rhs
            for node,alpha in zip(nodes,alphas):
                M[n,node]=alpha
        return M,B
    def node_laplacian(self,n0):
        return self.node_discretization(n0,'laplacian')

    def node_dx(self,n0):
        return self.node_discretization(n0,'dx')
    
    def node_discretization(self,n0,op='laplacian'):
        def beta(c):
            return 1.0
        
        N=self.g.angle_sort_adjacent_nodes(n0)
        P=len(N)
        is_boundary=int(self.g.is_boundary_node(n0))
        M=len(N) - is_boundary

        if is_boundary:
            # roll N to start and end on boundary nodes:
            nbr_boundary=[self.g.is_boundary_node(n)
                          for n in N]
            while not (nbr_boundary[0] and nbr_boundary[-1]):
                N=np.roll(N,1)
                nbr_boundary=np.roll(nbr_boundary,1)
        
        # area of the triangles
        A=[] 
        for m in range(M):
            tri=[n0,N[m],N[(m+1)%P]]
            Am=utils.signed_area( self.g.nodes['x'][tri] )
            A.append(Am)
        AT=np.sum(A)

        alphas=[]
        x=self.g.nodes['x'][N,0]
        y=self.g.nodes['x'][N,1]
        x0,y0=self.g.nodes['x'][n0]
        
        for n in range(P):
            n_m_e=(n-1)%M
            n_m=(n-1)%P
            n_p=(n+1)%P
            a=0
            if op=='laplacian':
                if n>0 or P==M: # nm<M
                    a+=-beta(n_m_e)/(4*A[n_m_e]) * ( (y[n_m]-y[n])*(y0-y[n_m]) + (x[n] -x[n_m])*(x[n_m]-x0))
                if n<M:
                    a+= -beta(n)/(4*A[n])  * ( (y[n]-y[n_p])*(y[n_p]-y0) + (x[n_p]-x[n ])*(x0 - x[n_p]))
            elif op=='dx':
                if n>0 or P==M: # nm<M
                    a+= beta(n_m_e)/(2*AT) * (y0-y[n_m])
                if n<M:
                    a+= beta(n)/(2*AT) * (y[n_p]-y0)
            elif op=='dy':
                if n>0 or P==M: # nm<M
                    a+= beta(n_m_e)/(2*AT) * (x[n_m]-x0)
                if n<M:
                    a+= beta(n)/(2*AT) * (x0 - x[n_p])
            else:
                raise Exception('bad op')
                
            alphas.append(a)

        alpha0=0
        for e in range(M):
            ep=(e+1)%P
            if op=='laplacian':
                alpha0+= - beta(e)/(4*A[e]) * ( (y[e]-y[ep])**2 + (x[ep]-x[e])**2 )
            elif op=='dx':
                alpha0+= beta(e)/(2*AT)*(y[e]-y[ep])
            elif op=='dy':
                alpha0+= beta(e)/(2*AT)*(x[ep]-x[e])
            else:
                raise Exception('bad op')
                
        if op=='laplacian' and P>M:
            norm_grad=0 # no flux bc
            L01=np.sqrt( (x[0]-x0)**2 + (y0-y[0])**2 )
            L0P=np.sqrt( (x[0]-x[-1])**2 + (y0-y[-1])**2 )

            gamma=3/AT * ( beta(0) * norm_grad * L01/2
                           + beta(P-1) * norm_grad * L0P/2 )
        else:
            gamma=0
        assert np.isfinite(alpha0)
        return ([n0]+list(N),
                [alpha0]+list(alphas),
                -gamma)

nd=NodeDiscretization(g)

e2c=g.edge_to_cells()

boundary=e2c.min(axis=1)<0
dirichlet_nodes={}
for e in np.nonzero(boundary)[0]:
    n1,n2=g.edges['nodes'][e]
    i1=g.nodes['ij'][n1,0]
    i2=g.nodes['ij'][n2,0]
    if i1==i2:
        dirichlet_nodes[n1]=i1
        dirichlet_nodes[n2]=i2

M,B=nd.construct_matrix(op='laplacian',dirichlet_nodes=dirichlet_nodes)

psi=sparse.linalg.spsolve(M.tocsr(),B)

## 


Mdx,Bdx=nd.construct_matrix(op='dx')
Mdy,Bdy=nd.construct_matrix(op='dy')

dpsi_dx=Mdx.dot(psi)
dpsi_dy=Mdy.dot(psi)

u=dpsi_dy
v=-dpsi_dx
# solve Mdx*phi = u
#       Mdy*phi = v
Mdxdy=sparse.vstack( (Mdx,Mdy) )
Buv=np.concatenate( (u,v))

phi,*rest=sparse.linalg.lsqr(Mdxdy,Buv)

##

plt.figure(2).clf()
g.plot_boundary(color='k',lw=0.5)
g.contour_node_values(psi,20,linewidths=1.5,colors='orange')
g.contour_node_values(phi,20,linewidths=1.5,colors='blue')

##

# NEXT:
#   solve the above with proper triangulation
#   
