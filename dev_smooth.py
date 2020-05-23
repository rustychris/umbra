import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from stompy.grid import unstructured_grid
import six
from stompy import filters

six.moves.reload_module(unstructured_grid)
## 

g=unstructured_grid.UnstructuredGrid.read_ugrid("/home/rusty/src/hor_flow_and_salmon/model/grid/snubby_junction/snubby-07-edit45.nc")

#zoom=(647300.308797298, 647345.1556059818, 4185781.740257725, 4185822.8274726057)
#ctr=np.array([647325., 4185800.])
ctr=np.array([647340.9995215593, 4185815.11691633])
zoom=(647272.002689184, 647366.8838680824, 4185758.0535712815, 4185844.980679361)


##

rotL=np.array( [[0,-1],[1,0]] )
rotR=np.array( [[0,1],[-1,0]] )

## 

stack=[]

node_ij={} # map nodes to their ij index
visited_cells={} # cell index => the index, 0..3, of its node that has the min i and min j.

j=g.select_edges_nearest(ctr)
he=g.halfedge(j,0)
node_ij[ he.node_rev() ] = np.array([0,0])

# stack is a half edge, meaning visit the cell the half edge is facing.
# node_ij[node_rev()] is guaranteed to be populated.
# and dir gives the ij vector for the edge normal (into the new cell)

stack.append( (he, np.array([1,0]) ) )

# offsets=np.array( [ [0,0],
#                     [1,0],
#                     [1,1],
#                     [0,1]] )

g.edge_to_cells(recalc=True)
cc=g.cells_center()

max_cells=150

while stack:
    he,vecnorm = stack.pop(0)
    c=he.cell()
    if (c in visited_cells) or (c<0):
        continue
    visited_cells[c]=True

    # Be sure search is breadth first, and we stop
    # with a given count.
    if len(visited_cells)>max_cells:
        break
    
    assert he.node_rev() in node_ij
    
    if g.cell_Nsides(c)!=4:
        continue

    he_trav=he
    ij_norm=vecnorm
    
    for i in range(4):
        # Update node_fwd:
        nrev=he_trav.node_rev()
        ij_rev=node_ij[nrev]
        nfwd=he_trav.node_fwd()
        ij_fwd=ij_rev + rotR.dot(ij_norm)
        if nfwd in node_ij:
            assert np.all(node_ij[nfwd]==ij_fwd)
        else:
            node_ij[nfwd]=ij_fwd

        # So both ends of the half edge have ij.
        # queue a visit to trav's opposite
        he_opp=he_trav.opposite()
        stack.append( (he_opp,-ij_norm) )

        # And move to next face of quad
        he_trav=he_trav.fwd()
        ij_norm=rotL.dot(ij_norm)

node_idxs=np.array( list(node_ij.keys()) )
ij=np.array( [node_ij[n] for n in node_idxs] )


# This is the half-edge approach.
# A cell-based approach may be simpler.

#   # Select some nodes around ctr:
#   node_idxs=g.select_nodes_nearest(ctr,count=500)
#   node_idxs=np.asarray(node_idxs)
#   
#   # Starting from an arbitrary node, start figuring out
#   # i,j indices.
#   
#   # Will need to weed out nodes that aren't surrounded
#   # by quads, and be careful about borders
#   
#   to_loc_idx={n:i for i,n in enumerate(node_idxs)}
#   
#   visited=np.zeros( len(node_idxs), np.bool8)
#   ij=np.zeros( (len(node_idxs),2), np.int32)
#   
#   n0=n=node_idxs[0]
#   
#   nloc=to_loc_idx[n]
#   ij[nloc,:]=[0,0]
#   visited[nloc]=True
#   
#   for nbr in g.node_to_nodes(n):
#       if nbr in to_loc_idx:
#           break
#   else:
#       # disconnected? or very small selection.
#       # in this case, ideal would be to choose the largest connected
#       # component.  Short of that, probably just print a message that
#       # it was not connected and return.
#       assert False 
#       
#   he=g.nodes_to_halfedge(n,nbr)
#   assert he.node_rev()==n
#   
#   stack=[]
#   
#   # This step defines the +x direction.
#   stack.append( [he,np.array([1,0])] )
#   
#   # The actual visit pattern:
#   while len(stack):
#       he,incr=stack.pop(0)
#       n_from=he.node_rev()
#       
#       n=he.node_fwd()
#       if n not in to_loc_idx:
#           continue 
#       nloc=to_loc_idx[n]
#       
#       if visited[nloc]:
#           continue
#       visited[nloc]=True
#   
#       new_ij=ij[to_loc_idx[n_from]] + incr
#       ij[nloc]=new_ij
#   
#       # valid=True
#       # cells=g.node_to_cells(n)
#       # if len(cells) != 4:
#       #     # This could be smarter about boundaries, but we're not there yet.
#       #     # currently this will skip over outside corners because
#       #     # there isn't a way to reach them
#       #     # If I omit this check, the half-edge code is more complicated.
#       #     continue
#       # 
#       # for c in cells:
#       #     if g.cell_Nsides(c)!=4:
#       #         valid=False
#       #         break
#       #     
#       # if not valid:
#       #     continue
#   
#       # if len(cells)==4:
#       # queue up neighbors:
#       # names here are as if the incoming edge is +x
#   
#       # invariant: cell that he faces is a quad
#       # if that holds for he, then it holds for he.fwd()
#       def valid_he(h):
#           c=h.cell()
#           return (c>=0) and (g.cell_Nsides(c)==4)
#       assert valid_he(he)
#       he_py=he.fwd()
#       assert valid_he(he_py)
#       stack.append( [he_py,rotL.dot(incr)] )
#   
#       # Since we can always get he_py, try for he_px
#       # first from that side.
#       he_tmp=he_py.opposite()
#       if valid_he(he_tmp):
#           he_px=he_tmp.fwd()
#           stack.append( [he_px,incr] )
#       else:
#           he_px=None
#       
#       # Option 1 to get to he_my:
#       he_opp=he.opposite()
#       if valid_he(he_opp):
#           he_my=he_opp.rev()
#       elif he_px is not None:
#           # Option 2 come around from he_px:
#           he_tmp=he_px.opposite()
#           if valid_he(he_tmp):
#               he_my=he_tmp.fwd()
#           else:
#               he_my=None
#               
#       if he_my is not None:
#           stack.append( [he_my,rotR.dot(incr)] )
#   
#       if (he_my is not None) and (he_px is None):
#           # Try to get to he_px via he_my.


# ij[:,0] looks okay.

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(lw=0.5,color='k',ax=ax)
g.plot_nodes(mask=node_idxs,ax=ax,masked_values=ij[:,1])
zoom=(647283.4119939702, 647401.4629277434, 4185757.0395007157, 4185865.6048835167)
ax.axis(zoom)

##
# Explicit approach:
#  Build out the XY matrix, with nan for non selected nodes.
#  Apply lowpass fir in both directions, maybe with a lower
#  nan threshold than usual.

# Update cells that have non-nan result get updated, maybe
# scaled by the largest update.

nghost=1
nij=1 + ij.max(axis=0) - ij.min(axis=0) + 2*np.array([nghost,nghost])
ij-= ij.min(axis=0) - nghost

XY=np.zeros( (nij[0],nij[1],2), np.float64)
XY[...]=np.nan
XY[ ij[:,0], ij[:,1], 0] = g.nodes['x'][node_idxs,0]
XY[ ij[:,0], ij[:,1], 1] = g.nodes['x'][node_idxs,1]

def smooth(XYin,iters=5):
    for _ in range(iters):
        # One iteration of local smoothing:
        XYlp=XYin.copy()
        for comp in [0,1]:
            d_in=XYin[:,:,comp]
            d_out=XYlp[:,:,comp]

            im=np.roll(d_in,1,axis=0) ; im[0,:]=np.nan
            ip=np.roll(d_in,-1,axis=0) ; ip[-1,:]=np.nan
            jm=np.roll(d_in,1,axis=1) ; jm[0,:]=np.nan
            jp=np.roll(d_in,-1,axis=1) ; jp[-1,:]=np.nan

            # add ghosts.
            # These ghosts ensure a zero 2nd derivative
            # across the boundary, but constructing a straight
            # line between internal - boundary - ghost.
            # But what I want is a zero slope condition.
            # normally this would be accomplished by copying
            # the interior value to the outside.
            # But what I want is to mirror the interior point
            # across the normal
            # interior.
            # What if I just copy the delta from the next node
            # in?
            # That would be like saying, for an outside
            # corner, something like
            #              
            # d_out = d_in + (d_out_ip - ip)
            # solution to stencil at ip:
            # d_out_ip = 0.2*( ip + ipp + d_in + jp_ip_g + jm_ip_g)
            # sub back in:
            # d_out = d_in + ( 0.2*( ip + ipp + d_out + jp_ip_g + jm_ip_g) - ip)
            # And rewrite with the usual notation, but now solve for the im_g ghost
            # usual: 
            # d_out =
            # 0.2*d_in + 0.2*ip + 0.2*im_g + 0.2*jp + 0.2*jm
            #       = d_in + 0.2*ip + 0.2*ipp + 0.2*d_in + 0.2*jp_ip + 0.2*jm_ip - ip
            #
            # d_in + ip + im_g + jp + jm
            #       = 5*d_in + ip + ipp + d_in + jp_ip + jm_ip - 5*ip
            # 
            # im_g = 5*(d_in - ip) + ipp + jp_ip + jm_ip - jp - jm
            
            missing=np.isnan(im)
            im_g=np.where(np.isnan(im), 2*d_in - ip, im)
            ip_g=np.where(np.isnan(ip), 2*d_in - im, ip)
            jm_g=np.where(np.isnan(jm), 2*d_in - jp, jm)
            jp_g=np.where(np.isnan(jp), 2*d_in - jm, jp)

            updated= 0.2*(d_out + ip_g + im_g + jp_g + jm_g )
            deltas=updated - d_out
            d_out[:,:]=updated

            # There is probably a more clever way to do this.
            # beyond me right now.
            # special handling of outside corners:
            delta_im=np.roll(deltas,1,axis=0)
            delta_ip=np.roll(deltas,-1,axis=0)
            delta_jm=np.roll(deltas,1,axis=1)
            delta_jp=np.roll(deltas,-1,axis=1)

            # Okay, this kind of does the intended thing,
            # but it's not very effective.
            # I think that it boils down to where the ghost
            # data comes from.  ghosts are mirrored
            # from inside the domain, and thus have no knowledge
            # of a parallel line.
            out_pp = np.isnan(ip) & np.isnan(jp)
            d_out[out_pp] += 0.5*(delta_im + delta_jm)[out_pp]
            out_mm = np.isnan(im) & np.isnan(jm)
            d_out[out_mm] += 0.5*(delta_ip + delta_jp)[out_mm]
            out_pm = np.isnan(ip) & np.isnan(jm)
            d_out[out_pm] += 0.5*(delta_im + delta_jp)[out_pm]
            out_mp = np.isnan(im) & np.isnan(jp)
            d_out[out_mp] += 0.5*(delta_ip + delta_jm)[out_mp]
            
        missing=np.isnan(XYlp)
        XYlp[missing]=XYin[missing]
        XYin=XYlp
        
    return XYin

XYlp=smooth(XY,5)

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
g2.plot_edges(lw=1.0,color='k',ax=ax)
g.plot_nodes(mask=node_idxs,ax=ax)
zoom=(647303.9082227673, 647367.4675893171, 4185791.442893589, 4185849.8951797294)
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


# The cell-based traversal seems good.
# Now the issues are
# (a) feathering
#     I think it's reasonable to have a feathering
#     radius defined by how many edges you are from
#     a nonselected node.  Then taper off the updates.
#     in proportion to that radius.
# (b) allowing outside corners to move.
#  Ghost cells means these don't move at all.
#  But I think I'd rather something that just has
#  a larger stencil in-board of the point?
#  just deal with it??
#   Maybe worth checking to see if this issue is real, or if
#   if there is some nan contamination.

##

# The straight up filtering on the original locations is no
# good, as it contracts near borders.
# Operating on differences is no good, as it doubles the degrees
# of freedom, and there is the possibility of introducing
# curl.
# Is it possible to take the differences, lowpass them,
# hmmm... not sure where I was going.

# What would it take to make sure that the gradient remains
# curl free?

# central differencing.
S=XY[:,:,0]
dSdi=(S[2:,:] - S[0:-2,:])/2.0
dSdj=(S[:,2:] - S[:,:-2])/2.0

dXdj=
dXYdj=np.diff(XY,axis=1)

gradX= np.dstack( [ dXYdi[...,0],
                    dXYdj[...,0] ])


##

# Local conformal fit:

# node coordinates in complex grid space
Z=(ij - ij.mean(axis=0)).dot( np.array([1,1j]) )

# node coordinates in real space.
X=g.nodes['x'][node_idxs]
X0=X-X.mean(axis=0)

# and edges:
local_j=np.unique( [j for n in node_idxs for j in g.node_to_edges(n)] )
real_local_j=[j for j in local_j
              if ( (g.edges['nodes'][j,0] in node_idxs)
                   and (g.edges['nodes'][j,1] in node_idxs) )]
seg_nodes=[ [np.nonzero(node_idxs==n1)[0][0],
             np.nonzero(node_idxs==n2)[0][0]]
            for n1,n2 in g.edges['nodes'][real_local_j] ]
seg_nodes=np.array(seg_nodes)
## 

def calc_halo(node_idxs,g):
    # Come up with weights based on rings
    node_insets=np.zeros( len(node_idxs), np.int32) - 1

    # Outer ring:
    stack=[]
    for ni,n in enumerate(node_idxs):
        for nbr in g.node_to_nodes(n):
            if nbr not in node_idxs:
                node_insets[ni]=0 # on the outer ring.
                stack.append(ni)

    while stack:
        ni=stack.pop(0)
        n=node_idxs[ni]

        for nbr in g.node_to_nodes(n):
            nbri=np.nonzero(node_idxs==nbr)[0]
            if nbri.size==0: continue
            nbri=nbri[0]
            if node_insets[nbri]<0:
                node_insets[nbri]=1+node_insets[ni]
                stack.append(nbri)
    return node_insets

halos=calc_halo(node_idxs,g)

# how much a node will be updated
# This leaves the outer two rings in place, partially updates
# the next ring, and fully updates anybody inside of there
update_weights=np.interp(halos, [1,3],[0,1])
error_weights=1-update_weights

##             
aspect=0.6 # Aspect ratio. smaller smushes in the j direction
tele_j=0.0 # 
tele_i=0.0
center_i=0 # Location of center of curvature in the ij plane
center_j=2000
scale=3.5 # isotropic scaling
theta=-0.76 # rotation
# translation is solved directly

def fwd_transform(vec,Z,error_weights):
    # the parameters being optimized
    # Optimize over inverse center to avoid singularity with zero curvature
    aspect,inv_center_i,inv_center_j,scale,theta,tele_i,tele_j = vec
    inv_eps=0.0001

    Ztran=Z

    y=np.imag(Ztran)
    if np.abs(tele_j)>1e-4:
        y=(np.exp(tele_j*y)-1)/tele_j
    y=y*aspect
    x=np.real(Ztran)
    if np.abs(tele_i)>1e-4:
        x=(np.exp(tele_i*x)-1)/tele_i

    Ztran=x + 1j*y

    # Curvature can be done with a single
    # center, complex valued.  But for optimization, use the
    # inverse, and flip around here.
    inv_center=inv_center_i + 1j*inv_center_j
    if np.abs(inv_center) > inv_eps:
        center=1./inv_center
        Ztran=np.exp(Ztran/center)*center

    Ztran=scale*Ztran

    Ztran=Ztran*np.exp(1j*theta)

    # move back to R2 plane
    Xz=np.c_[ np.real(Ztran), np.imag(Ztran)]
    
    # make the offset match where we can't move nodes
    offset=((Xz-X0)*error_weights[:,None]).sum(axis=0) / error_weights.sum()
    Xz-=offset

    return Xz

def cost(vec):
    Xtran=fwd_transform(vec,Z,error_weights)
    err=  (((Xtran-X0)**2).sum(axis=1)*error_weights).sum() / error_weights.sum()
    return err

from scipy.optimize import fmin

vec_init=[1.0,0.001,0.001,5,1.0,0.0,0.0]
best=fmin(cost,vec_init)

fit=fwd_transform(best,Z,error_weights)

##
plt.figure(3).clf()
fig,ax=plt.subplots(1,1,num=3)

#ax.plot(X0[:,0], X0[:,1],'r.')
#ax.plot(Xz[:,0], Xz[:,1],'b.')
ax.add_collection(LineCollection(X0[seg_nodes],color='r',lw=0.5))
# scat=ax.scatter(Xz[:,0],Xz[:,1],20,ij[:,0])
# plt.colorbar(scat,label='i')
# scat=ax.scatter(Xz[:,0],Xz[:,1],20,weights)

ax.add_collection(LineCollection(fit[seg_nodes],color='b',lw=0.5))
ax.axis('equal')

##

                               
x=np.linspace(-10,10,100)
plt.figure(4).clf()
plt.plot(x,x,'k-')
plt.plot(x,0.5*x,'g-')
tele=-0.001
aspect=2.

plt.plot(x,aspect*(np.exp(tele*x)-1)/tele,'b-')
