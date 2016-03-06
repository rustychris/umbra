import unstructured_grid
import wkb2shp
from delft import dfm_grid

## 

reload(unstructured_grid)
reload(dfm_grid)

## 

ug=dfm_grid.DFMGrid(fn='/home/rusty/models/delft/nms/nms_03/small_links_fixed_net.nc')

## 

#
poly=wkb2shp.shp2geom('/home/rusty/models/delft/nms/nms_03/clip-v00.shp')[0]['geom']


## 

from shapely import prepared

to_del=ug.select_nodes_intersecting(prepared.prep(poly),as_type='index')

## 

print "Starting counts: %d nodes, %d edges, %d cells"%(ug.Nnodes_valid(),
                                                       ug.Nedges_valid(),
                                                       ug.Ncells_valid() )

for ni,n in enumerate(to_del):
    if ni%1000==0:
        print "%d / %d"%(ni,len(to_del))
    ug.delete_node_cascade(n)

# Most of the time spent in edge_to_cells(), over 27k calls
# secondly is nodes_to_edge
# edge_to_cells always checks all of self.edges['cells'] against UNKNOWN
# that's where all the time goes.

print "Ending count: %d nodes, %d edges, %d cells"%(ug.Nnodes_valid(),
                                                    ug.Nedges_valid(),
                                                    ug.Ncells_valid() )

## 

ug2=dfm_grid.DFMGrid(fn='/home/rusty/models/delft/nms/nms_03/spinupdate_net.nc')

print "Starting counts: %d nodes, %d edges, %d cells"%(ug2.Nnodes_valid(),
                                                       ug2.Nedges_valid(),
                                                       ug2.Ncells_valid() )
## 

sel=ug2.select_nodes_intersecting(prepared.prep(poly))

# # 
to_del2=np.nonzero(~sel)[0]

for ni,n in enumerate(to_del2):
    if ni%1000==0:
        print "%d / %d"%(ni,len(to_del2))
    ug2.delete_node_cascade(n)

print "Ending counts: %d nodes, %d edges, %d cells"%(ug2.Nnodes_valid(),
                                                     ug2.Nedges_valid(),
                                                     ug2.Ncells_valid() )


## 
if 0:
    plt.clf()
    ax=plt.gca()
    coll1=ug.plot_edges(ax=ax)
    coll2=ug2.plot_edges(ax=ax)

    coll1.set_edgecolor('b')
    coll2.set_edgecolor('g')
## 

# Get those into one consolidated grid
# start with a very slow way:

def merge_grid(self,other):
    node_map=np.zeros(other.Nnodes(),'i4')-1
    edge_map=np.zeros(other.Nedges(),'i4')-1
    cell_map=np.zeros(other.Ncells(),'i4')-1
    
    # deal with topology first
    for n in other.valid_node_iter():
        node_map[n] = self.add_node( x=other.nodes['x'][n] )
    for e in other.valid_edge_iter():
        edge_map[e] = self.add_edge( nodes=node_map[ other.edges['nodes'][e] ] )
    for c in other.valid_cell_iter():
        cell_map[c] = self.add_cell( nodes=node_map[ other.cell_to_nodes(c) ])

    for fname in other.nodes.dtype.names:
        if fname not in ['x','deleted'] and fname in self.nodes.dtype.names:
            for n in other.valid_node_iter():
                self.nodes[fname][node_map[n]]=other.nodes[fname][n]

    for fname in other.edges.dtype.names:
        if fname not in ['nodes','deleted'] and fname in self.nodes.dtype.names:
            for e in other.valid_edge_iter():
                old_val=other.edges[fname][e]
                if fname=='cells':
                    old_val=old_val.copy()
                    for i in [0,1]:
                        if old_val[i]>=0:
                            old_val[i]=cell_map[ old_val[i] ]
                self.edges[fname][edge_map[n]]=old_val

    for fname in other.cells.dtype.names:
        if (fname not in ['nodes','deleted'] and 
            fname in self.nodes.dtype.names  and
            fname[0]!='_'):
            for c in other.valid_cell_iter():
                old_val=other.cells[fname][c]
                if fname=='edges':
                    for i in range(len(old_val)):
                        if old_val[i]>=0:
                            old_val[i]=edge_map[old_val[i]]
                self.cells[fname][node_map[n]]=old_val
    
        

merge_grid(ug,ug2)

## 

plt.clf()
#ug.plot_edges()
ug.plot_cells()
plt.axis('equal')

## 

# 
ug.renumber()


## 


dfm_grid.write_dfm(ug,'/home/rusty/models/delft/nms/nms_03/merge_to_stitch_net.nc',overwrite=True)
