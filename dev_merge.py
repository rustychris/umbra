import matplotlib.pyplot as plt
import numpy as np
from stompy.grid import unstructured_grid
import six
six.moves.reload_module(unstructured_grid)
## 

g=unstructured_grid.UnstructuredGrid.read_ugrid("/home/rusty/src/hor_flow_and_salmon/model/grid/snubby_junction/snubby-07-edit09.nc")

p=np.array([647796.4, 4185586.5])
j=g.select_edges_nearest(p)

zoom=(647784.8687127546, 647807.529949435, 4185579.9619691074, 4185593.4337877515)

plt.figure(1).clf()
fig,axs=plt.subplots(2,1,sharex=True,sharey=True,num=1)

ax=axs[0]
g.plot_edges(lw=0.4,color='k',ax=ax,clip=zoom,labeler='id')
g.plot_edges(lw=1.4,color='r',mask=[j],ax=ax)
g.plot_nodes(labeler='id',clip=zoom,ax=ax)
ax.axis(zoom)

# Goal is an operation that is the reverse of split
# split takes an edge, and
# splits it into two edges
# neighborig quads are split into 3 triangles
# and neighboring triangles are split in two,
# with an optional, heuristic for merging triangles
# into quads.

# The reverse, join, operation, might...
# take an edge, and merge the adjacent
# cells.
# For two triangles, this creates a quad, and we're done.
# For two quads, the default would be to create a hex.
# But we can test whether the two quads can instead be merged
# into a larger quad, by also merging the nodes at the end of
# the edge.


## 
jn=g.edges['nodes'][j].copy()

c=g.merge_cells(j=j)

j_next=[]

# Check the two nodes:
for n in jn:
    ncs=g.node_to_cells(n)
    nc_sides=[g.cell_Nsides(nc) for nc in ncs if nc!=c]

    if nc_sides==[4,4]:
        j=g.cells_to_edge(c,ncs[0])
        assert j is not None
        he=g.halfedge(j,0)
        if he.cell()!=c: he=he.opposite()
        # now c is on our left
        if he.node_rev()!=n: he=he.fwd()
        assert he.cell()==c
        assert he.node_rev()==n
        # delete the edge between the two quads
        he_opp=he.opposite()
        g.delete_edge_cascade(he_opp.fwd().j)
        trav=he_opp
        A=trav.node_rev() ; trav=trav.rev()
        B=trav.node_rev() ; trav=trav.rev()
        jC1=trav.opposite().fwd().j
        C=trav.node_rev() ; trav=trav.rev()
        jC2=trav.opposite().rev().j
        D=trav.node_rev() ; trav=trav.rev()
        E=trav.node_rev() ; trav=trav.rev()

        g.add_edge(nodes=[A,C])
        g.add_cell( nodes=[A,C,B] )
        g.add_edge(nodes=[C,E])
        g.add_cell( nodes=[C,E,D] )
        g.merge_edges(node=n)
        g.add_cell( nodes=[A,E,C] )
        # and find a potential edge to merge next
        if jC1==jC2:
            j_next.append(jC1)
    elif nc_sides==[3,3,3]:
        seed=g.cells_center()[ncs[0]]
        n_nbrs=[]
        for jnbr in list(g.node_to_edges(n)):
            print("Checking j=%d e2c %s against %d"%(jnbr,g.edge_to_cells(jnbr),c))
            if c not in g.edge_to_cells(jnbr):
                # This is finding a cell that is already deleted?
                print("to delete")
                g.delete_edge_cascade(jnbr) 
        n_nbrs=g.node_to_nodes(n)
        assert len(n_nbrs)==2,"n_nbrs should be two, but it's %s"%( str(n_nbrs) ) 
        g.merge_edges(node=n)
        g.add_cell_at_point(seed)
    else:
        pass

# Still to fix:
#  Is it possible for j_next to include an edge w/ only 1 neighbor?
#   yes - the 2Q code doesn't check for cells existing


                    
# First, could get the two sides of c bordering n, and
# see if they are "straightish"

# But the fun part is the topology
# Possibilities:
#  (a) 3 triangles, which could be merged into a single
#      quad
#  (b) 2 quads, which could be semi-split into 3 triangles.
#  (c) some mix of quads and triangles in which case we do nothing.
#  (d) 4 triangles -> do nothing.
# 
#
ax=axs[1]
g.plot_edges(lw=0.4,color='k',ax=ax,clip=zoom)
if j_next:
    g.plot_edges(lw=1.4,color='m',ax=ax,mask=j_next)
#g.plot_edges(lw=1.4,color='y',ax=ax,mask=[jC1,jC2])
    

g.plot_cells(ax=ax,clip=zoom)
g.plot_nodes(ax=ax,labeler='id',clip=zoom)
ax.axis(zoom)
fig.tight_layout()
