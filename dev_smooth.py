import matplotlib.pyplot as plt
import numpy as np
from stompy.grid import unstructured_grid
import six
from stompy import filters

six.moves.reload_module(unstructured_grid)
## 

g=unstructured_grid.UnstructuredGrid.read_ugrid("/home/rusty/src/hor_flow_and_salmon/model/grid/snubby_junction/snubby-07-edit30.nc")

#zoom=(647300.308797298, 647345.1556059818, 4185781.740257725, 4185822.8274726057)
#ctr=np.array([647325., 4185800.])
ctr=np.array([647340.9995215593, 4185815.11691633])
zoom=(647272.002689184, 647366.8838680824, 4185758.0535712815, 4185844.980679361)


# Select some nodes around ctr:
node_idxs=g.select_nodes_nearest(ctr,count=500)
node_idxs=np.asarray(node_idxs)

# Starting from an arbitrary node, start figuring out
# i,j indices.

# Will need to weed out nodes that aren't surrounded
# by quads, and be careful about borders

to_loc_idx={n:i for i,n in enumerate(node_idxs)}

visited=np.zeros( len(node_idxs), np.bool8)
ij=np.zeros( (len(node_idxs),2), np.int32)

n0=n=node_idxs[0]

nloc=to_loc_idx[n]
ij[nloc,:]=[0,0]
visited[nloc]=True

for nbr in g.node_to_nodes(n):
    if nbr in to_loc_idx:
        break
else:
    # disconnected? or very small selection.
    # in this case, ideal would be to choose the largest connected
    # component.  Short of that, probably just print a message that
    # it was not connected and return.
    assert False 
    
he=g.nodes_to_halfedge(n,nbr)
assert he.node_rev()==n

rotL=np.array( [[0,-1],[1,0]] )
rotR=np.array( [[0,1],[-1,0]] )

stack=[]

# This step defines the +x direction.
stack.append( [he,np.array([1,0])] )

# The actual visit pattern:
while len(stack):
    he,incr=stack.pop(0)
    n_from=he.node_rev()
    
    n=he.node_fwd()
    if n not in to_loc_idx:
        continue 
    nloc=to_loc_idx[n]
    
    if visited[nloc]:
        continue
    visited[nloc]=True

    new_ij=ij[to_loc_idx[n_from]] + incr
    ij[nloc]=new_ij

    valid=True
    cells=g.node_to_cells(n)
    if len(cells) != 4:
        # This could be smarter about boundaries, but we're not there yet.
        continue
    
    for c in cells:
        if g.cell_Nsides(c)!=4:
            valid=False
            break
        
    if not valid:
        continue

    # queue up neighbors:
    # names here are as if the incoming edge is +x
    he_py=he.fwd()
    he_px=he_py.opposite().fwd()
    he_my=he_px.opposite().fwd()

    stack.append( [he_py,rotL.dot(incr)] )
    stack.append( [he_px,incr] )
    stack.append( [he_my,rotR.dot(incr)] )

# Did we visit everyone?
disconnected=(~visited).sum()
if disconnected:
    print("%d of %d nodes were not connected"%(disconnected,len(node_idxs)))

    plt.figure(1).clf()
    g.plot_edges(lw=0.5,color='k')
    g.plot_nodes(mask=node_idxs,ax=ax,masked_values=visited)
    plt.axis(zoom)

    ij=ij[visited,:]
    orig_node_idxs=node_idxs
    node_idxs=node_idxs[visited]
    
##
# Explicit approach:
#  Build out the XY matrix, with nan for non selected nodes.
#  Apply lowpass fir in both directions, maybe with a lower
#  nan threshold than usual.

# Update cells that have non-nan result get updated, maybe
# scaled by the largest update.

nij=1 + ij.max(axis=0) - ij.min(axis=0)
ij-=ij.min(axis=0)

XY=np.zeros( (nij[0],nij[1],2), np.float64)
XY[...]=np.nan
XY[ ij[:,0], ij[:,1], 0] = g.nodes['x'][node_idxs,0]
XY[ ij[:,0], ij[:,1], 1] = g.nodes['x'][node_idxs,1]

XYlp=XY
XYlp=filters.lowpass_fir(XYlp,winsize=5,axis=0,nan_weight_threshold=0.5)
XYlp=filters.lowpass_fir(XYlp,winsize=5,axis=1,nan_weight_threshold=0.5)

# copy back to nodes:
g2=g.copy()
g2.nodes=g2.nodes.copy()

# valid nodes have to be in the original set and have a valid x and y
# after filtering
valid=np.isfinite(XYlp[...,0] + XY[...,0])
valid_nodes=valid[ij[:,0],ij[:,1]]

g2.nodes['x'][node_idxs[valid_nodes],0] = XYlp[ ij[valid_nodes,0], ij[valid_nodes,1], 0]
g2.nodes['x'][node_idxs[valid_nodes],1] = XYlp[ ij[valid_nodes,0], ij[valid_nodes,1], 1] 


plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
g.plot_edges(lw=0.4,color='0.5',ax=ax)
g2.plot_edges(lw=0.4,color='k',ax=ax)
# g.plot_nodes(mask=node_idxs,ax=axs[0])
ax.axis(zoom)
fig.tight_layout()

# First cut:
#   Terrible.
#   contracts quites a bit.
#   clearly need a way to more strongly anchor the anchor points.
#   moving to 1st derivative would help some

# That gets better after fixing some indexing problems.
# Just feathering in means that we can't do boundaries, which
# is a problem.
# But not feathering means near the edge of the stencil contracts
# a lot.

## 
# Would filtering on the difference grid help?

X=XY[...,0]
plt.figure(3).clf()

fig,axs=plt.subplots(2,1,num=3,sharex=True,sharey=True)

img0=axs[0].imshow(X)
plt.colorbar(img0,ax=axs[0])

dXYdi=np.diff(XY,axis=0)
dXYdj=np.diff(XY,axis=1)

img1=axs[1].imshow(dXYdj[:,:,0]) # dX/di
plt.colorbar(img1,ax=axs[1])

# use the location of n0 to anchor everything.

# Hmm -
# dXYd{i,j} has double the degrees of freedom (almost).
# there is an implicit constraint that dXY/dij
# is irrotational, or equivalently that it is the gradient of
# a potential.  In this case, obviously the gradient of the initial
# coordinates.
# wolog, consider just the x coordinate.

# Briefly, what about other types of smoothing.
# Like a diffusion operation?
#  that's great where there is a front between movable and
#  immovable nodes.
# But what about at a boundary?
#  Say each node averages the location of its neighbors?
#  One approach would be that nodes can only be updated who
#  have both neighbors in the direction of the filter.
#  So a straighline border can be smoother only in the direction
#  parallel to the boundary.
#  An inside corner can still be smoothed both ways.
#  and outside corners cannot be smoothed at all.

# a little weird that outside corners end up being a special case.
# Ah - but if boundary nodes are treated with a neumann BC, roughly
# by adding a ghost node, I think it would work.

# I'd rather something that operates on the differences, so they can
# smoothed just the same.

# What sort of smoothing operation on a gradient preserves its
# irrotational status?
# seems like we could get into some annoying details
