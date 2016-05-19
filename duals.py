import numpy as np
import matplotlib.pyplot as plt
import utils
reload(utils)

import unstructured_grid
reload(unstructured_grid)
## 

p1=np.array([0.,0.])
p2=np.array([10.,10.])
nx=3
ny=3

g_prime= unstructured_grid.UnstructuredGrid()
gp_map = g_prime.add_rectilinear(p1,p2,nx,ny)

dxy=(p2-p1) / np.array([nx-1,ny-1])

g_dual=unstructured_grid.UnstructuredGrid()
gd_map = g_dual.add_rectilinear(p1-0.5*dxy,p2+0.5*dxy,nx+1,ny+1)

# perturb the original grid
mid_col_nodes=gp_map['nodes'][1,:]
g_prime.nodes['x'][mid_col_nodes,1]+=2

g_prime.cells_center(refresh=True,mode='sequential')

g_dual.add_node_field('constrained',np.zeros(g_dual.Nnodes(),'i4'))
g_prime.add_node_field('constrained',np.zeros(g_prime.Nnodes(),'i4'))

# constraint names:
C_NONE=0 # freely movable
C_FIXED=1 # static


g_prime.nodes['constrained'][gp_map['nodes'][:,-1]]=C_FIXED
# g_prime.nodes['constrained'][gp_map['nodes'][:,0]]=C_FIXED

## 

def centers_to_nodes(src_grid,src_cell_map,
                     dst_grid,dst_node_map):
    # Update the dual based on circumcenters of the prime.
    for cx in range(src_cell_map.shape[0]): # nx-1 for prime->dual
        for cy in range(src_cell_map.shape[1]): # ny-1 for prime->dual
            p_c=src_cell_map[cx,cy]
            d_n=dst_node_map[cx,cy]
            if (p_c<0) or (d_n<0):
                continue
            dst_grid.nodes['x'][d_n]=src_grid.cells_center()[p_c]


def prime_to_dual():
    centers_to_nodes(g_prime,gp_map['cells'],
                     g_dual,gd_map['nodes'][1:-1,1:-1])
    # update the centers of the dual:
    g_dual.cells_center(refresh=True,mode='sequential')

def dual_to_prime():
    nodes=gp_map['nodes'].copy()
    fixed=(g_prime.nodes['constrained'][nodes]==C_FIXED)
    nodes[fixed]=-1 # don't update some of them.
    centers_to_nodes(g_dual,gd_map['cells'],
                     g_prime,nodes)
    g_prime.cells_center(refresh=True,mode='sequential')


## 
g_prime.report_orthogonality()
prime_to_dual() ; dual_to_prime()

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g_prime.plot_edges(ax=ax)
g_dual.plot_edges(ax=ax,color='g')

vc=g_prime.cells_center()
ax.plot(vc[:,0],vc[:,1],'b.')

vc=g_dual.cells_center()
ax.plot(vc[:,0],vc[:,1],'g.')

sel=(g_prime.nodes['constrained']!=0)
ax.scatter(g_prime.nodes['x'][sel,0],
           g_prime.nodes['x'][sel,1],
           20,g_prime.nodes['constrained'][sel])


# Doesn't seem to converge to better than a 4.24deg max error.
# not sure if that would be better with more updates to the
# dual.
