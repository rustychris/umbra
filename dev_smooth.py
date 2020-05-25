import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from stompy.grid import unstructured_grid
import six
from scipy.optimize import fmin
from stompy import filters

six.moves.reload_module(unstructured_grid)
## 

g=unstructured_grid.UnstructuredGrid.read_ugrid("/home/rusty/src/hor_flow_and_salmon/model/grid/snubby_junction/snubby-07-edit45.nc")

def calc_halo(g, node_idxs):
    """
    calculate how many steps each node in node_idxs is away
    from a node *not* in node_idxs.
    """
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

def fwd_transform(vec,Z,X0,error_weights):
    """
    The Z=complex ij => real X transform.
    vec: parameters for the transform:
       aspect: how much narrow cells are in the j dimension than i dimension
       inv_center_i/j: the inverse of the center of curvature in the complex ij plane.
         or 0,0 for no curvature
      scale: isotropic scaling
      tele_i,j: telescoping factors in i,j directions
    """
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

def conformal_smooth(g,ctr,max_cells=250,max_radius=None,halo=[0,5]):
    node_idxs,ij=g.select_quad_subset(ctr,max_cells=max_cells,max_radius=max_radius)

    print("Found %d nodes"%len(node_idxs))

    # node coordinates in complex grid space
    Z=(ij - ij.mean(axis=0)).dot( np.array([1,1j]) )

    # node coordinates in real space.
    X=g.nodes['x'][node_idxs]
    Xoff=X.mean(axis=0)
    X0=X-Xoff

    halos=calc_halo(g,node_idxs)

    # how much a node will be updated
    # This leaves the outer two rings in place, partially updates
    # the next ring, and fully updates anybody inside of there
    update_weights=np.interp(halos, halo,[0,1])
    error_weights=1-update_weights

    def cost(vec):
        Xtran=fwd_transform(vec,Z,X0,error_weights)
        err=  (((Xtran-X0)**2).sum(axis=1)*error_weights).sum() / error_weights.sum()
        return err

    vec_init=[1.0,0.001,0.001,5,1.0,0.0,0.0]
    best=fmin(cost,vec_init)

    fit=fwd_transform(best,Z,X0,error_weights) + Xoff

    new_node_x=( (1-update_weights)[:,None]*g.nodes['x'][node_idxs]
                 + update_weights[:,None]*fit )
    g.nodes['x'][node_idxs]=new_node_x
    return node_idxs

##

g2=g.copy()
g2.nodes=g2.nodes.copy()

## 
zoom=(647272.002689184, 647366.8838680824, 4185758.0535712815, 4185844.980679361)
#zoom=(647338.5356397176, 647353.926604037, 4185825.9776436556, 4185835.2842944413)
plt.figure(2).clf()
fig,axs=plt.subplots(2,1,num=2,sharex=True,sharey=True)
axs[0].axis(zoom)

while 1:
    axs[0].collections=[]
    g.plot_edges(lw=0.5,color='k',ax=axs[0])
    axs[1].collections=[]
    g2.plot_edges(lw=0.5,color='k',ax=axs[1])
    
    plt.draw()
    fig.tight_layout()

    ctr=plt.ginput(1)
    if not ctr:
        break
    ctr=np.array(ctr[0])

    # This could be reused for multiple specific nodes
    node_idxs,ij=g2.select_quad_subset(ctr,max_cells=100)
    halos=calc_halo(g2,node_idxs)
    
    pad=2
    ij-=ij.min(axis=0) - pad
    XY=np.nan*np.zeros( (pad+1+ij[:,0].max(),
                         pad+1+ij[:,1].max(),
                         2), np.float64)
    XY[ij[:,0],ij[:,1]]=g2.nodes['x'][node_idxs]

    # Grab a specific node
    # n=g2.select_nodes_nearest(ctr)
    # or iterate through sufficiently internal nodes
    n_iter=3
    for count in range(n_iter):
        for ni,n in enumerate(node_idxs):
            if halos[ni]<2: continue

            # Find that node in
            ni=np.nonzero(node_idxs==n)[0]
            assert len(ni)>0,"Somehow n wasn't in the quad subset"
            ni=ni[0]

            # Query XY to estimate where n "should" be.
            i,j=ij[ni]

            
            dXY_di = np.diff(XY,axis=0)
            dXY_dj = np.diff(XY,axis=1)

            # Local means
            d_di=np.nanmean( [ dXY_di[i,j], dXY_di[i-1,j]],axis=0)
            d_dj=np.nanmean( [ dXY_dj[i,j], dXY_dj[i,j-1]],axis=0)

            delta_i = np.nanmean(  [d_di - dXY_di[i-1,j],
                                    dXY_di[i,j] - d_di],axis=0)
            delta_j = np.nanmean( [d_dj - dXY_dj[i,j-1],
                                   dXY_dj[i,j] - d_dj ],axis=0)

            delta= 0.9*np.nanmean( [delta_i,delta_j],axis=0)
            new_x=XY[i,j] + delta
            if np.isfinite(new_x[0]):
                if count+1==n_iter:
                    g2.modify_node(n,x=new_x)
                XY[i,j]=new_x
                print(f"moved {n} by {delta}")
            else:
                print("Hit nans.")

##

local_j=np.unique( [j for n in node_idxs for j in g.node_to_edges(n)] )
real_local_j=[j for j in local_j
              if ( (g.edges['nodes'][j,0] in node_idxs)
                   and (g.edges['nodes'][j,1] in node_idxs) )]
seg_nodes=[ [np.nonzero(node_idxs==n1)[0][0],
             np.nonzero(node_idxs==n2)[0][0]]
            for n1,n2 in g.edges['nodes'][real_local_j] ]
seg_nodes=np.array(seg_nodes)

plt.figure(3).clf()
fig,ax=plt.subplots(1,1,num=3)

#ax.plot(X0[:,0], X0[:,1],'r.')
#ax.plot(Xz[:,0], Xz[:,1],'b.')
#ax.add_collection(LineCollection(X0[seg_nodes],color='r',lw=0.5))
# scat=ax.scatter(Xz[:,0],Xz[:,1],20,ij[:,0])
# plt.colorbar(scat,label='i')
# scat=ax.scatter(Xz[:,0],Xz[:,1],20,weights)

ax.add_collection(LineCollection(fit[seg_nodes],color='b',lw=0.5))
ax.axis('equal')

##
