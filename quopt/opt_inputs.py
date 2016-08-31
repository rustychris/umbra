import unstructured_grid
import os
import utils

# node constraint values
FREE=0
FIXED=1

## 

try:
    cwd=os.path.dirname( __file__ )
except NameError:
    cwd=os.getcwd()
    
##     

def quad_reach_3wide():
    g=unstructured_grid.UnstructuredGrid.from_pickle( os.path.join(cwd,
                                                                   "../Umbra/sample_data",
                                                                   "three-quads2.pkl"))
    g.add_node_field('constrained',np.zeros(g.Nnodes(),'i2'))
    fixed=[0,3,48,60]
    g.nodes['constrained'][ fixed ] = FIXED
    return g
## 

g=quad_reach_3wide()

plt.clf()
#ncoll=g.plot_nodes(values=g.nodes['constrained'], #labeler=lambda n,node: str(n),
#                   lw=0,cmap='BrBG').set_zorder(10)
# g.plot_cells(lw=0).set_zorder(-5)
g.plot_edges(color='k').set_zorder(5)

## 

def constrained_circumcenter(points,select):
    nconstrained=np.sum(select)
    if nconstrained==0:
        return utils.poly_circumcenter(points)
    if nconstrained==3:
        return utils.circumcenter(points[select])
    if nconstrained>3:
        return utils.poly_circumcenter(points[select])
    # the interesting cases:
    if nconstrained==1:
        con=points[select]
        uncon=points[~select]
        ctrs=[]
        for ab in list(utils.circular_n(uncon,2)):
            abc=np.concatenate( (ab,con) )
            ctrs.append( utils.circumcenter(ab[0],ab[1],con[0]) )
        return np.mean( np.array(ctrs),axis=0 )
    else:
        # not that hard, but too lazy..
        assert False
        

def ortho_move_nodes(g,plot=False):
    radius_errors=[]
    nudges=np.zeros( (g.Nnodes(),2), 'f8')
    ncounts=np.zeros( g.Nnodes(), 'i4')

    for c in g.valid_cell_iter():
        nodes=g.cell_to_nodes(c)
        points=g.nodes['x'][nodes] 
        #ctr = utils.poly_circumcenter(points)
        con= (g.nodes['constrained'][nodes]==FIXED)
        ctr=constrained_circumcenter(points, con)
        diffs=points - ctr
        dists=utils.dist(diffs)
        if np.any(con):
            radius=np.mean(dists[con])
        else:
            radius=np.mean(dists)
        radius_errors.append( radius-dists )
        this_nudge=((radius-dists)/radius)[:,None] * diffs
        nudges[nodes] += this_nudge
        ncounts[nodes] += 1

    sel=ncounts>0
    nudges[sel] /= ncounts[sel][:,None]
    sel=(ncounts>0)&(g.nodes['constrained']==0)
    for n in np.nonzero(sel):
        g.modify_node(n,x=g.nodes['x'][n]+1.*nudges[n])

    g.cells_center(refresh=True)
    
    radius_errors=np.concatenate( radius_errors)
    print "Max radius error: %f  mean: %f"%( np.abs(radius_errors).max(),
                                             utils.rms(radius_errors) )

    if plot:
        plt.cla()
        g.plot_edges(color='k').set_zorder(5)
        #quiv=plt.quiver(g.nodes['x'][:,0],
        #                g.nodes['x'][:,1],
        #                nudges[:,0],nudges[:,1])
        plt.draw()
        plt.pause(0.05)

        g.cells_center(refresh=True)
        g.report_orthogonality()


for loop in range(50):
    ortho_move_nodes(g,plot=True)

# HERE:
# look into ways of including a smoothing step
#   possibly giving each edge a target length, and nudge a bit
#   towards that
#   or shoot for evenly distributed angles
#   or from the perspective of the numerics, try to make successive
#   distances between centers equal? 

## 
plt.clf()
ncoll=g.plot_nodes(values=g.nodes['constrained'], #labeler=lambda n,node: str(n),
                   lw=0,cmap='BrBG').set_zorder(10)
g.plot_edges(color='k').set_zorder(5)
#quiv=plt.quiver(g.nodes['x'][:,0],
#                g.nodes['x'][:,1],
#                nudges[:,0],nudges[:,1])
g.report_orthogonality()

## 

# same, but add a nudge to making the angles equal within each cell
for loop in range(200):
    radius_errors=[]
    nudges=np.zeros( (g.Nnodes(),2), 'f8')
    ncounts=np.zeros( g.Nnodes(), 'i4')

    for c in g.valid_cell_iter():
        nodes=g.cell_to_nodes(c)
        points=g.nodes['x'][nodes] 
        #ctr = utils.poly_circumcenter(points)
        con= (g.nodes['constrained'][nodes]==FIXED)
        ctr=constrained_circumcenter(points, con)
        diffs=points - ctr
        dists=utils.dist(diffs)
        if np.any(con):
            radius=np.mean(dists[con])
        else:
            radius=np.mean(dists)
        radius_errors.append( radius-dists )
        this_nudge=((radius-dists)/radius)[:,None] * diffs

        if 1: # nudge to equal angles
            # first edge vec is in the direction of n0 to n1
            edge_vecs=utils.to_unit( utils.cdiff(points,axis=0) )
            edge_angles=180/np.pi * np.arctan2(edge_vecs[:,1],edge_vecs[:,0])
            # first interior angle is formed by n0n1 and n1n2
            # so int_angles[0] actually corresponds to nodes[1]
            int_angles=utils.cdiff(edge_angles) % 360.0
            target=int_angles.mean() # aka 360/nsides
            errors=int_angles-target

            # => array([ 5.47620555, -0.99873301, -5.47620555,  0.99873301])
            #             n1            n2           n3          n0

            ip1=np.arange(len(nodes))
            im1=(ip1-2)%(len(nodes))

            # optimistic - likely false association between internal angles
            # and how much to rotate around circumcenter
            angle_nudges=errors[im1] - errors[ip1]

            # produces CCW tangent vectors scaled by radius
            tangents=utils.rot(np.pi/2,diffs)
            node_nudges=0.02*angle_nudges[:,None]*np.pi/180.0 * tangents
            this_nudge -= node_nudges
            # consider n0
            # next CCW, it has n1, with an internal angle 5deg too large
            # next CW,  it has n3, internal angle is 5 deg too small.
            # so we want to move n0 towards n3, CW
            # it's nudge should be -10 degrees
            # n2 should have the opposite +10degree nudge

        nudges[nodes] += this_nudge
        ncounts[nodes] += 1

    sel=ncounts>0
    nudges[sel] /= ncounts[sel][:,None]
    sel=(ncounts>0)&(g.nodes['constrained']==0)
    for n in np.nonzero(sel):
        g.modify_node(n,x=g.nodes['x'][n]+1.*nudges[n])

    g.cells_center(refresh=True)

    radius_errors=np.concatenate( radius_errors)
    print "Max radius error: %f  mean: %f"%( np.abs(radius_errors).max(),
                                             utils.rms(radius_errors) )

    if loop%10==0:
        plt.cla()
        g.plot_edges(color='k').set_zorder(5)
        #quiv=plt.quiver(g.nodes['x'][:,0],
        #                g.nodes['x'][:,1],
        #                nudges[:,0],nudges[:,1])
        plt.pause(0.01)

        g.report_orthogonality()


# that smooths the grid out significantly.
# too slow and local to do anything beyond
# a 2 cell neighborhood, though.
