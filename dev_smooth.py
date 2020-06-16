import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from stompy import utils
from stompy.grid import unstructured_grid, orthogonalize
import six
from scipy.optimize import fmin
from stompy import filters

## 

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

def conformal_smooth(g,node_idxs,ij=None,halo=[0,5],
                     free_nodes=None,fit_nodes=None,
                     max_update=1.0):
    tweaker=orthogonalize.Tweaker(g)
    
    if ij is None:
        node_idxs,ij=g.select_quad_subset(ctr=None,max_cells=None,node_set=node_idxs)

    print("Found %d nodes"%len(node_idxs))

    # node coordinates in complex grid space
    Z=(ij - ij.mean(axis=0)).dot( np.array([1,1j]) )

    # node coordinates in real space.
    X=g.nodes['x'][node_idxs]
    Xoff=X.mean(axis=0)
    X0=X-Xoff

    halos=tweaker.calc_halo(node_idxs)

    if free_nodes is not None:
        # use dict for faster tests
        free_nodes={n:True for n in free_nodes}
        for ni,n in enumerate(node_idxs):
            if n not in free_nodes:
                halos[ni]=0

    # how much a node will be updated
    # This leaves the outer two rings in place, partially updates
    # the next ring, and fully updates anybody inside of there
    update_weights=np.interp(halos, halo,[0,1])
    error_weights=1-update_weights
    update_weights*=max_update

    if fit_nodes is not None:
        fit_node={n:True for n in fit_nodes}
        for ni,n in enumerate(node_idxs):
            if n not in fit_nodes:
                error_weights[ni]=0
        
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
