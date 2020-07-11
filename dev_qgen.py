# Building up a quad grid from patches
from stompy.grid import unstructured_grid
from stompy.grid import exact_delaunay

from stompy import utils
from stompy.spatial import field
from shapely import geometry
from scipy.interpolate import griddata, Rbf
from scipy import sparse

import numpy as np
from stompy.grid import orthogonalize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from stompy.grid import triangulate_hole,front

import six
##

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(front)

##

# similar codes as in front.py
FREE=0 # default
RIGID=1
SLIDE=2

# generating grid
gen=unstructured_grid.UnstructuredGrid(max_sides=150,
                                       extra_node_fields=[('ij',np.int32,2)],
                                       extra_edge_fields=[('dij',np.int32,2)])
def node_ij_to_edge(g):
    dij=(g.nodes['ij'][g.edges['nodes'][:,1]]
         - g.nodes['ij'][g.edges['nodes'][:,0]])
    g.add_edge_field('dij',dij,on_exists='replace')
    

if 0:
    # Simple test case - rectangle
    gen.add_rectilinear([0,0],[100,200],2,2)
    gen.nodes['ij']= (gen.nodes['x']/10).astype(np.int32)

    # test slight shift in xy:
    sel=gen.nodes['x'][:,1]>100
    gen.nodes['x'][sel, 0] +=30
    # and in ij
    gen.nodes['ij'][sel,0] += 2

    node_ij_to_edge(gen)
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

    node_ij_to_edge(gen)

##


class NodeDiscretization(object):
    def __init__(self,g):
        self.g=g
    def construct_matrix(self,op='laplacian',dirichlet_nodes={}):
        g=self.g
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


class QuadGen(object):
    def __init__(self,gen,execute=True):
        self.gen=gen
        if execute:
            self.add_bezier(gen)
            self.create_intermediate_grid()
            self.adjust_intermediate_bounds()
            self.smooth_interior_quads(self.g_int)
            self.calc_psi_phi()
            self.adjust_intermediate_by_psi_phi()
        
    def create_intermediate_grid(self):
        # target grid
        self.g_int=g=unstructured_grid.UnstructuredGrid(max_sides=4,
                                                        extra_node_fields=[('ij',np.float64,2),
                                                                           ('gen_j',np.int32),
                                                                           ('rigid',np.int32)])
        gen=self.gen
        for c in gen.valid_cell_iter():
            ijs=[]
            ns=gen.cell_to_nodes(c)
            xys=gen.nodes['x'][ns]
            ijs=[ np.array([0,0]) ]
            local_edges=gen.cell_to_edges(c,ordered=True)
            for j in local_edges:
                if gen.edges['cells'][j,0]==c:
                    s=1
                elif gen.edges['cells'][j,1]==c:
                    s=-1
                else: assert 0
                ijs.append( ijs[-1]+s*gen.edges['dij'][j] )
            assert np.allclose( ijs[0],ijs[-1] )

            ijs=np.array(ijs[:-1])
            ijs-=ijs.min(axis=0) # force to have ll corner at (0,0)

            # Create in ij space
            patch=g.add_rectilinear(p0=[0,0],
                                    p1=ijs.max(axis=0),
                                    nx=int(1+ijs[:,0].max()),
                                    ny=int(1+ijs[:,1].max()))
            pnodes=patch['nodes'].ravel()

            g.nodes['gen_j'][pnodes]=-1

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
            
            # Fill in generating edges for boundary nodes
            boundary_nodes=g.boundary_cycle()
            # hmm -
            # each boundary node in g sits at either a node or
            # edge of gen.
            # For any non-rigid node in g, it should sit on
            # an edge of gen.  ties can go either way, doesn't
            # matter (for bezier purposes)
            # Can int_x/y help here?
            # or just brute force it
            
            local_edge_ijs=np.array( [ ijs, np.roll(ijs,-1,axis=0)] )
            lower_ij=local_edge_ijs.min(axis=0)
            upper_ij=local_edge_ijs.max(axis=0)
            
            for n in boundary_nodes:
                n_ij=g.nodes['ij'][n]

                # [ {nA,nB}, n_local_edges, {i,j}]
                candidates=np.all( (n_ij>=lower_ij) & (n_ij<=upper_ij),
                                   axis=1)
                for lj in np.nonzero(candidates)[0]:
                    # is n_ij approximately on the line
                    # local_edge_ijs[lj] ?
                    offset=utils.point_line_distance(n_ij,local_edge_ijs[:,lj,:])
                    if offset<0.1:
                        g.nodes['gen_j'][n]=local_edges[lj]
                        break
                else:
                    raise Exception("Failed to match up a boundary node")
                
        g.renumber()

    def plot_intermediate(self,num=1):
        plt.figure(num).clf()
        fig,ax=plt.subplots(num=num)
        self.gen.plot_edges(lw=1.5,color='b',ax=ax)
        self.g_int.plot_edges(lw=0.5,color='k',ax=ax)
        self.g_int.plot_nodes(mask=self.g_int.nodes['rigid']>0) # good
        ax.axis('equal')

    def add_bezier(self,gen):
        """
        Generate bezier control points for each edge.
        """
        # Need to force the corners to be 90deg angles, otherwise
        # there's no hope of getting orthogonal cells in the interior.
        
        order=3 # cubic bezier curves
        bez=np.zeros( (gen.Nedges(),order+1,2) )
        bez[:,0,:] = gen.nodes['x'][gen.edges['nodes'][:,0]]
        bez[:,order,:] = gen.nodes['x'][gen.edges['nodes'][:,1]]

        gen.add_edge_field('bez', bez, on_exists='replace')

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
            # now node n's two edges are in njs, as node pairs, with the first
            # in each pair being n
            # dij is the ij delta along that edge
            # flip records whether it was necessary to flip the edge
            # and deltas records the geometry delta
            
            # the angle in ij space tells us what it *should* be
            # these are angles going away from n
            # How does this work out when it's a straight line in ij space?
            theta0_ij=np.arctan2( -dijs[0][1], -dijs[0][0])
            theta1_ij=np.arctan2(dijs[1][1],dijs[1][0]) 
            dtheta_ij=(theta1_ij - theta0_ij + np.pi) % (2*np.pi) - np.pi

            theta0=np.arctan2(-deltas[0][1],-deltas[0][0])
            theta1=np.arctan2(deltas[1][1],deltas[1][0])
            dtheta=(theta1 - theta0 + np.pi) % (2*np.pi) - np.pi

            theta_err=dtheta-dtheta_ij # 103: -0.346, slight right but should be straight
            #theta0_adj = theta0+theta_err/2
            #theta1_adj = theta1-theta_err/2

            # not sure about signs here.
            cp0 = gen.nodes['x'][n] + utils.rot( theta_err/2, 1./3 * deltas[0] )
            cp1 = gen.nodes['x'][n] + utils.rot( -theta_err/2, 1./3 * deltas[1] )

            # save to the edge
            gen.edges['bez'][js[0],1+flips[0]] = cp0
            gen.edges['bez'][js[1],1+flips[1]] = cp1

    def plot_gen_bezier(self,num=10):
        fig=plt.figure(num)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        self.gen.plot_edges(lw=0.3,color='k',alpha=0.5,ax=ax)
        self.gen.plot_nodes(alpha=0.5,ax=ax,zorder=3,color='orange')
        
        for j in gen.valid_edge_iter():
            n0=gen.edges['nodes'][j,0]
            nN=gen.edges['nodes'][j,1]
            bez=gen.edges['bez'][j]
            
            t=np.linspace(0,1,21)

            B0=(1-t)**3
            B1=3*(1-t)**2 * t
            B2=3*(1-t)*t**2
            B3=t**3
            points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]

            ax.plot(points[:,0],points[:,1],'r-')
            ax.plot(bez[:,0],bez[:,1],'b-o')

    def gen_bezier_curve(self,samples_per_edge=10):
        points=self.gen_bezier_linestring(samples_per_edge=samples_per_edge)
        return front.Curve(points,closed=True)
        
    def gen_bezier_linestring(self,samples_per_edge=10):
        """
        Calculate an up-sampled linestring for the bezier boundary of self.gen
        """
        bound_nodes=self.gen.boundary_cycle()

        points=[]
        for a,b in zip(bound_nodes,np.roll(bound_nodes,-1)):
            j=gen.nodes_to_edge(a,b)
            
            n0=gen.edges['nodes'][j,0]
            nN=gen.edges['nodes'][j,1]
            bez=gen.edges['bez'][j]
            
            t=np.linspace(0,1,1+samples_per_edge)
            if n0==b: # have to flip order
                t=t[::-1]

            B0=(1-t)**3
            B1=3*(1-t)**2 * t
            B2=3*(1-t)*t**2
            B3=t**3
            edge_points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]

            points.append(edge_points[:-1])
        return np.concatenate(points,axis=0)

    def adjust_intermediate_bounds(self):
        """
        Adjust exterior of intermediate grid with bezier
        curves
        """
        gen=self.gen
        g=self.g_int

        # This one gets tricky with the floating-point ij values.
        # gen.nodes['ij'] may be float valued.
        # The original code iterates over gen edges, assumes that
        # Each gen edge divides to an exact number of nodes, then
        # we know the exact ij of those nodes,
        # pre-evaluate the spline and then just find the corresponding
        # nodes.

        # With float-valued gen.nodes['ij'], though, we still have
        # a bezier curve, but it's ends may not be on integer values.
        # The main hurdle is that we need a different way of associating
        # nodes in self.g to a generating edge
        
        for j in gen.valid_edge_iter():
            n0=gen.edges['nodes'][j,0]
            nN=gen.edges['nodes'][j,1]
            bez=gen.edges['bez'][j]
            
            # dij=gen.edges['dij'][j]
            # steps=int(utils.mag(dij))
            # t=np.linspace(0,1,1+steps)

            g_nodes=np.nonzero( g.nodes['gen_j']==j )[0]

            p0=gen.nodes['x'][n0]
            pN=gen.nodes['x'][nN]

            T=utils.dist(pN-p0)
            t=utils.dist( g.nodes['x'][g_nodes] - p0 ) / T

            too_low=(t<0)
            too_high=(t>1)
            if np.any(too_low):
                print("Some low")
            if np.any(too_high):
                print("Some high")
                
            t=t.clip(0,1)

            if 1: # the intended bezier way:
                B0=(1-t)**3
                B1=3*(1-t)**2 * t
                B2=3*(1-t)*t**2
                B3=t**3
                points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]
            else: # debugging linear way
                print("Debugging - no bezier boundary")
                points=(1-t)[:,None]*p0 + t[:,None]*pN

            for n,point in zip(g_nodes,points):
                g.modify_node(n,x=point)
            # dij_up=dij/steps
            # 
            # for i in range(1,len(points)-1):
            #     ij0=gen.nodes['ij'][n0]
            #     ijN=gen.nodes['ij'][nN]
            #     ij_i=ij0+i*dij_up
            # 
            #     n=np.nonzero( np.all( g.nodes['ij']==ij_i, axis=1))[0][0]
            #     g.modify_node(n,x=points[i])

    def smooth_interior_quads_basic(self,g):
        # redistribute interior nodes evenly
        # by solving laplacian
        # Still have an issue where tight bends will allow nodes to
        # move outside the boundary.
        M=sparse.dok_matrix( (g.Nnodes(),g.Nnodes()), np.float64)
        rhs_nodes=-1*np.ones(g.Nnodes(),np.int32)
        for n in g.valid_node_iter():
            if g.is_boundary_node(n):
                M[n,n]=1
                rhs_nodes[n]=n
            else:
                nbrs=g.node_to_nodes(n)
                if 0: # isotropic
                    M[n,n]=-len(nbrs)
                    for nbr in nbrs:
                        M[n,nbr]=1
                else: # see about allowing some anisotropy
                    # In the weighting, want to normalize by distances
                    i_length=0
                    j_length=0
                    dists=utils.dist(g.nodes['x'][n],g.nodes['x'][nbrs])
                    ij_deltas=np.abs(g.nodes['ij'][n] - g.nodes['ij'][nbrs])
                    # length scales for i and j
                    ij_scales=1./( (ij_deltas*dists[:,None]).sum(axis=0) )
                    
                    for nbr,ij_delta in zip(nbrs,ij_deltas):
                        fac=(ij_delta*ij_scales).sum()
                        M[n,nbr]=fac
                        M[n,n]-=fac
                    
        rhs_x=np.where( rhs_nodes<0, 0, g.nodes['x'][rhs_nodes,0])
        new_x=sparse.linalg.spsolve(M.tocsr(),rhs_x)
        rhs_y=np.where( rhs_nodes<0, 0, g.nodes['x'][rhs_nodes,1])
        new_y=sparse.linalg.spsolve(M.tocsr(),rhs_y)

        g.nodes['x'][:,0]=new_x
        g.nodes['x'][:,1]=new_y

    def smooth_interior_quads_sliding(self,g,iterations=5):
        """
        Smooth quad grid by allowing boundary nodes to slide, and
        imparting a normal constraint at the boundary.
        """
        curve=self.gen_bezier_curve()

        N=g.Nnodes()

        for slide_it in utils.progress(range(iterations)):
            M=sparse.dok_matrix( (2*N,2*N), np.float64)

            rhs=np.zeros(2*N,np.float64)

            for n in g.valid_node_iter():
                if g.is_boundary_node(n):
                    dirichlet=g.nodes['rigid'][n]
                    #dirichlet=True
                    if dirichlet:
                        M[n,n]=1
                        rhs[n]=g.nodes['x'][n,0]
                        M[N+n,N+n]=1
                        rhs[N+n]=g.nodes['x'][n,1]
                    else:
                        # figure out the normal from neighbors.
                        boundary_nbrs=[]
                        interior_nbr=[]
                        for nbr in g.node_to_nodes(n):
                            if g.nodes['gen_j'][nbr]>=0:
                                boundary_nbrs.append(nbr)
                            else:
                                interior_nbr.append(nbr)
                        assert len(boundary_nbrs)==2
                        assert len(interior_nbr)==1

                        vec=np.diff( g.nodes['x'][boundary_nbrs], axis=0)[0]
                        nrm=utils.to_unit( np.array([vec[1],-vec[0]]) )
                        tng=utils.to_unit( np.array(vec) )
                        c3=np.dot(nrm,g.nodes['x'][n])
                        # n-equation puts it on the linen
                        M[n,n]=nrm[0]
                        M[n,N+n]=nrm[1]
                        rhs[n]=c3
                        # N+n equation set the normal
                        # the edge to interior neighbor (xi,yi) perpendicular to that line.
                        # (xb-xi)*c1 + (yb-yi)*c2 = 0
                        # c1*xb - c1*xi + c2*yb - c2*yi = 0
                        inbr=interior_nbr[0]
                        M[N+n,n]=tng[0]
                        M[N+n,inbr]=-tng[0]
                        M[N+n,N+n]=tng[1]
                        M[N+n,N+inbr]=-tng[1]
                        rhs[N+n]=0.0
                else:
                    nbrs=g.node_to_nodes(n)
                    if 0: # isotropic
                        M[n,n]=-len(nbrs)
                        M[N+n,N+n]=-len(nbrs)
                        for nbr in nbrs:
                            M[n,nbr]=1
                            M[N+n,N+nbr]=1
                    else:
                        # In the weighting, want to normalize by distances
                        i_length=0
                        j_length=0
                        dists=utils.dist(g.nodes['x'][n],g.nodes['x'][nbrs])
                        ij_deltas=np.abs(g.nodes['ij'][n] - g.nodes['ij'][nbrs])
                        # length scales for i and j
                        ij_scales=1./( (ij_deltas*dists[:,None]).sum(axis=0) )

                        assert np.all( np.isfinite(ij_scales) )

                        for nbr,ij_delta in zip(nbrs,ij_deltas):
                            fac=(ij_delta*ij_scales).sum()
                            M[n,nbr]=fac
                            M[n,n]-=fac
                            M[N+n,N+nbr]=fac
                            M[N+n,N+n]-=fac

            new_xy=sparse.linalg.spsolve(M.tocsr(),rhs)

            g.nodes['x'][:,0]=new_xy[:N]
            g.nodes['x'][:,1]=new_xy[N:]

            # And nudge the boundary nodes back onto the boundary
            for n in g.valid_node_iter():
                if g.nodes['gen_j'][n]>=0:
                    new_f=curve.point_to_f(g.nodes['x'][n],rel_tol='best')
                    g.nodes['x'][n] = c(new_f)

        return g
        

    def bezier_boundary_polygon(self):
        """
        For trimming nodes that got shifted outside the proper boundary
        """
        # This would be more efficient if unstructured_grid just provided
        # some linestring methods that accepted a node mask
        
        g_tri=self.g_int.copy()
        internal_nodes=g_tri.nodes['gen_j']<0
        for n in np.nonzero(internal_nodes)[0]:
            g_tri.delete_node_cascade(n)
        boundary_linestring = g_tri.extract_linear_strings()[0]
        boundary=g_tri.nodes['x'][boundary_linestring]
        return geometry.Polygon(boundary)
        
    def get_intermediate_triangular(self):
        # Nodes, bounds and nodes['ij'] of g_int, but made of only triangles.
        if 0:
            gtri=self.g_int.copy()
            gtri.make_triangular()
        elif 0:
            # Alternative approach to get the triangulated grid:
            poly=self.bezier_boundary_polygon()

            # drop internal nodes outside that boundary
            gtri=self.g_int.copy()
            internal_nodes=gtri.nodes['gen_j']<0
            inside_nodes=gtri.select_nodes_intersecting(geom=poly)
            bad_nodes=internal_nodes & (~inside_nodes)
            for n in np.nonzero(bad_nodes)[0]:
                gtri.delete_node_cascade(n)
            gtri.renumber(reorient_edges=False)

            # Could just delete all of the internal edges
            g_dt=exact_delaunay.Triangulation()
            g_dt.bulk_init(gtri.nodes['x'])

            # Constrain outside boundary:
            # (depending on details, could reuse whatever was
            #  used in generating poly above)
            for j in gtri.valid_edge_iter():
                ns=gtri.edges['nodes'][j]
                gen_js=gtri.nodes['gen_j'][ns]

                if (gen_js[0]>=0) and (gen_js[1]>=0):
                    g_dt.add_constraint(ns[0],ns[1])
                else:
                    # clear out so internal edges can be replaced by
                    # delaunay edges
                    gtri.delete_edge_cascade(j)

            # By retaining the boundary edges in gtri, avoid roundoff
            # issues testing edge centers on the boundary
            # against the boundary.
            ec=g_dt.edges_center()
            for j in g_dt.valid_edge_iter():
                if not poly.contains( geometry.Point(ec[j]) ):
                    continue
                dt_nodes=g_dt.edges['nodes'][j]
                j_exist=gtri.nodes_to_edge(dt_nodes)
                if j_exist is None:
                    gtri.add_edge(nodes=dt_nodes)
            gtri.make_cells_from_edges(max_sides=3)

        else:
            gtri=self.g_int.copy()
            
            # Delete all internal edges:
            e2c=gtri.edge_to_cells(recalc=True)
            seed_point=gtri.cells_centroid()[0]
            for j in np.nonzero(e2c.min(axis=1)>=0)[0]:
                gtri.delete_edge_cascade(j)
            gtri.delete_orphan_nodes()
            gtri.renumber(reorient_edges=False)


            gnew=triangulate_hole.triangulate_hole(gtri,seed_point,hole_rigidity='all-nodes',splice=False)

            # For starters, just see how the psi/phi solution looks on gnew
            # So I need ij for the boundary nodes of gnew
            node_ij=np.zeros( (gnew.Nnodes(),2), np.float64)
            node_ij[:]=np.nan

            c=0 # the cell I'm working in from gen
            for jcount,j in enumerate(gen.cell_to_edges(c)):
                gen_n=gen.edges['nodes'][j,:]
                if gen.edges['cells'][j,1]==c:
                    gen_n=gen_n[::-1]
                else:
                    assert gen.edges['cells'][j,0]==c

                gnew_a=gnew.select_nodes_nearest( gen.nodes['x'][gen_n[0]] )
                gnew_b=gnew.select_nodes_nearest( gen.nodes['x'][gen_n[1]] )

                gnew_string=gnew.select_nodes_boundary_segment( gen.nodes['x'][gen_n] )

                dist_along=utils.dist_along(gnew.nodes['x'][gnew_string])
                alpha=dist_along/dist_along[-1]

                node_ij[gnew_string,:]=( (1-alpha)[:,None]*gen.nodes['ij'][gen_n[0]]
                                         + alpha[:,None]*gen.nodes['ij'][gen_n[1]] )

            gnew.add_node_field('ij',node_ij,on_exists='overwrite')
            gtri=gnew
            
        self.gtri=gtri            
        return gtri
        
    def calc_psi_phi(self):
        # gtri=self.get_intermediate_triangular()
        self.gtri=gtri=self.g_int

        self.nd=nd=NodeDiscretization(gtri)

        e2c=gtri.edge_to_cells()

        if 1: # PSI
            boundary=e2c.min(axis=1)<0
            dirichlet_nodes={}
            for e in np.nonzero(boundary)[0]:
                n1,n2=gtri.edges['nodes'][e]
                i1=gtri.nodes['ij'][n1,0]
                i2=gtri.nodes['ij'][n2,0]
                if i1==i2:
                    dirichlet_nodes[n1]=i1
                    dirichlet_nodes[n2]=i2

            M,B=nd.construct_matrix(op='laplacian',dirichlet_nodes=dirichlet_nodes)

            psi=sparse.linalg.spsolve(M.tocsr(),B)
            self.psi=psi

        if 1: # PHI
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
            self.phi=phi

    def plot_psi_phi(self,num=4):
        plt.figure(num).clf()
        self.gtri.plot_edges(color='k',lw=0.5,alpha=0.2)
        self.gtri.contour_node_values(self.psi,20,linewidths=1.5,colors='orange')
        self.gtri.contour_node_values(self.phi,20,linewidths=1.5,colors='blue')
        
    def adjust_intermediate_by_psi_phi(self):
        """
        Move internal nodes of g_int according to phi and psi fields
        """
        gtri=self.gtri
        gen=self.gen
        g=self.g_int

        for coord in [0,1]: # i,j
            gen_valid=(~gen.nodes['deleted'])&(gen.nodes['ij_fixed'][:,coord])
            gen_to_gtri_nodes=[gtri.select_nodes_nearest(x)
                               for x in gen.nodes['x'][gen_valid]]

            # i or j coord:
            all_coord=gen.nodes['ij'][gen_valid,coord]
            if coord==0:
                all_field=self.psi[gen_to_gtri_nodes]
            else:
                all_field=self.phi[gen_to_gtri_nodes]

            # will have coords
            coord_to_field=np.array( [ [k,np.mean(all_field[elts])]
                                       for k,elts in utils.enumerate_groups(all_coord)] )

            if coord==0:
                i_psi=coord_to_field
            else:
                j_phi=coord_to_field

        # the mapping isn't necessarily monotonic at this point, but it
        # needs to be..  so force it.
        # enumerate_groups will put k in order, but not the field values
        i_psi[:,1] = np.sort(i_psi[:,1])
        j_phi[:,1] = np.sort(j_phi[:,1])[::-1]

        g_psi=np.interp( g.nodes['ij'][:,0],
                         i_psi[:,0],i_psi[:,1])
        g_phi=np.interp( g.nodes['ij'][:,1],
                         j_phi[:,0], j_phi[:,1])

        # Use gtri to go from phi/psi to x,y
        interp_xy=utils.LinearNDExtrapolator( np.c_[self.psi,self.phi],
                                              gtri.nodes['x'],eps=0.5 )
        new_xy=interp_xy( np.c_[g_psi,g_phi] )

        g.nodes['x']=new_xy
        g.refresh_metadata()

    def plot_result(self,num=5):
        plt.figure(num).clf()
        self.g_int.plot_edges()
        plt.axis('equal')

if 0:
    qg=QuadGen(gen)
    qg.plot_result()

# HERE:
#  Think about how usage would actually work, esp.
#    compared to how it might be done in janet.
#   In Janet the quad tools that I've used are tailored
#   to river reaches. Define a line (typ. centerline),
#   lateral cell sizes to each side, and optionally a bounding contour.
#
#   To replicate that workflow with the code here...
#    Limited to the case where the sections conform to the bounding
#    contour.
#    Currently I only process a single, simple polygon, so the spacing
#    in the lateral
#    HERE HERE

#  See how higher order interp works for phi,psi

gen=unstructured_grid.UnstructuredGrid.from_pickle("../pescadero/bathy/cbec-survey-interp-grid.pkl")

## 
gen.renumber(reorient_edges=False)

gen.plot_nodes(labeler='id')
gen.plot_edges(labeler='id')

# For now, have have a cell.
gen.modify_max_sides(1000)

cell_nodes=gen.extract_linear_strings()[0][:-1]
gen.add_cell(nodes=cell_nodes)

NA=-9999

ij=np.zeros((gen.Nnodes(),2),np.float64)
ij[:,:]=NA

gen.add_node_field('ij',ij,on_exists='replace')

corners=[  (552203.2836289784, 4124587.264231968),
           (552178.9600882703, 4124531.13298418),
           (552475.2083404858, 4124249.853064708),
           (552468.9715351759, 4124292.2633408145)]
 
# Just set the corners:
for val,pnt in zip( [ [20,80],
                      [0,80],
                      [0,0],
                      [20,0] ],
                    corners ):
    n=gen.select_nodes_nearest(pnt)
    gen.nodes['ij'][n] = val

def fill_ij_interp(gen):
    # the rest are filled by linear interpolation
    gen.add_node_field( 'ij_fixed', gen.nodes['ij']!=NA, on_exists='replace')
    
    for idx in [0,1]:
        node_vals=gen.nodes['ij'][:,idx] 
        strings=gen.extract_linear_strings()
        for s in strings:
            if s[0]==s[-1]:
                # cycle, so we can roll
                has_val=np.nonzero( node_vals[s]!=NA )[0]
                if len(has_val):
                    s=np.roll(s[:-1],-has_val[0])
                    s=np.r_[s,s[0]]
            s_vals=node_vals[s]
            dists=utils.dist_along( gen.nodes['x'][s] )
            valid=s_vals!=NA
            fill_vals=np.interp( dists[~valid],
                                 dists[valid], s_vals[valid] )
            node_vals[s[~valid]]=fill_vals

fill_ij_interp(gen)

# plt.figure(6).clf()
# gen.plot_edges(lw=0.5)
# gen.plot_nodes(labeler='ij')

node_ij_to_edge(gen)

qg=QuadGen(gen,execute=False)

qg.add_bezier(gen)
# qg.plot_gen_bezier()

qg.create_intermediate_grid()
qg.adjust_intermediate_bounds()

qg.smooth_interior_quads_sliding(qg.g_int)

qg.plot_intermediate()

g_int_preadjust=qg.g_int.copy()

##

# Seems that the internal quad spacing code is letting
# nodes bunch up on an inside curve.
qg.calc_psi_phi()

qg.plot_psi_phi()

qg.adjust_intermediate_by_psi_phi()

qg.plot_result()
plt.figure(6).clf()
g_int_preadjust.plot_edges(color='green')

# As much as the preadjust grid looks pretty nice, the post-adjust grid
# does have better orthogonality metrics (better than half the angle error,
# same for circumcenter error).
# That said, the cell size distribution is greater with the adjusted grid.
# Would need to rewrite the sampling to take into account local scale of
# psi/phi.

##

# Step back for a moment:
# What would it look like to instead create the rough grid, and
# then try to solve for the layout of the original nodes in a way
# that satisfies orthogonality?

# Go back to that paper about morphing a grid towards orthogonal.

# One paper:
# Numerical conformal mapping and mesh generation for
# polygonal and multiply-connected regions
# B. Lin and S. N. Chandler-Wilde

# But that other paper allowed more arbitrary grids

# A Method for Orthogonal Grid Generation
# Mehmet Ali Akinlar1, Stephen Salako2 and Guojun Liao3
# grid deformation approach, "nearly" orthogonal. But the results
# look pretty bad.

# So the anisotropic smoothing has a weakness where the spacing
# of boundary nodes warps the interior.
# Currently I smooth x and y independently, using the same matrix.

# But is there a way to locally linearize where slidable boundary nodes
# can fall, forcing their internal edge to be perpendicular to the boundary?

# For a sliding boundary node [xb,yb] , it has to fall on a line, so
# c1*xb + c2*yb = c3
# where [c1,c2] is a normal vector of the line

# And I want the edge to its interior neighbor (xi,yi) perpendicular to that line.
# (xb-xi)*c1 + (yb-yi)*c2 = 0

# So all of that seems legit.
# I'm a little wary that the system is still square -- may mean that it reaches
# a crazy solution


# HERE:
#   I think it's worth trying to solve phi/psi on the intermediate quad grid.
#   to see different the result is.  That means refactoring and adapting the
#   front.py approach to triangular, so the code that puts ij back on the grid
#   can 
