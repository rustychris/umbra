# Building up a quad grid from patches
from stompy.grid import unstructured_grid
from stompy import utils
from shapely import geometry
from scipy.interpolate import griddata, Rbf

##

# generating grid
gen=unstructured_grid.UnstructuredGrid(max_sides=50,
                                       extra_node_fields=[('ij',np.int32,2)],
                                       extra_edge_fields=[('dij',np.int32,2)])


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

# target grid
g=unstructured_grid.UnstructuredGrid(max_sides=4,
                                     extra_node_fields=[('ij',np.int32,2)])




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
                            nx=ijs[:,0].max(),
                            ny=ijs[:,1].max())
    pnodes=patch['nodes'].ravel()

    # Copy xy to ij, then remap xy
    g.nodes['ij'][pnodes] = g.nodes['x'][pnodes]

    int_x=utils.LinearNDExtrapolator(ijs,xys[:,0])
    node_x=int_x(g.nodes['x'][pnodes,:])

    int_y=utils.LinearNDExtrapolator(ijs,xys[:,1])
    node_y=int_y(g.nodes['x'][pnodes,:])

    g.nodes['x'][pnodes]=np.c_[node_x,node_y]
    
    # delete cells that fall outside of the ij
    
    for n in pnodes[ np.isnan(node_x) ]:
        g.delete_node_cascade(n)

    ij_poly=geometry.Polygon(ijs)
    for c in patch['cells'].ravel():
        cn=g.cell_to_nodes(c)
        c_ij=np.mean(g.nodes['ij'][cn],axis=0)
        if not ij_poly.contains(geometry.Point(c_ij)):
            g.delete_cell(c)

    # This part will need to get smarter when there are multiple patches:
    g.delete_orphan_edges()
    g.delete_orphan_nodes()

plt.figure(1).clf()
gen.plot_edges(lw=1.5,color='b')
g.plot_edges(lw=0.5,color='k')
plt.axis('equal')
##


# Not very useful yet.
from stompy.grid import orthogonalize
tweaker=orthogonalize.Tweaker(g)

tweak_nodes=[n for n in g.valid_node_iter() if g.node_degree(n)>2]

for i in range(1):
    tweaker.local_smooth(node_idxs=tweak_nodes,
                         ij=g.nodes['ij'][tweak_nodes],
                         min_halo=1)
    # for n in tweak_nodes:
    #     # In the future, instead of using degree to figure this out
    #     # pinned nodes should get marked based on the nodes in gen.
    #     tweaker.nudge_node_orthogonal(n)

plt.figure(1).clf()
gen.plot_edges(lw=1.5,color='b')
g.plot_edges(lw=0.5,color='k')
plt.axis('equal')
