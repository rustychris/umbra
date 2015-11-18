import unstructured_grid
import trigrid
import os
import numpy as np
import matplotlib.pyplot as plt

## 
reload(unstructured_grid)
try:
    __file__
except NameError:
    __file__="./test_unstructured_grid.py"

tg=trigrid.TriGrid(suntans_path=os.path.join( os.path.dirname(__file__),
                                              "Umbra/sample_data/sfbay" ) )
ug=unstructured_grid.UnstructuredGrid.from_trigrid(tg)

# # 

z=(546849.79838709673, 582787.29838709673, 4081854.8387096776, 4109677.4193548388)

plt.clf()
ug.plot_edges()
ug.plot_nodes()
ug.plot_cells(centers=True)
plt.axis(z)

## 

# check methods for adding a node, adding an edge, delete a node, delete an edge
# 
# # 
n1=ug.add_node(x=[577208.29133064509, 4089969.7580645154])

plt.clf()
ug.plot_edges()
ug.plot_nodes()
ug.plot_cells(centers=True)
plt.axis(z)

# # 
nbr=ug.select_nodes_nearest([573295.74092741928, 4089100.3024193542])
ug.delete_node_cascade(nbr)

plt.clf()
ug.plot_edges()
ug.plot_nodes()
ug.plot_cells(centers=True)
plt.axis(z)

## 

# add edge:
nbrs=ug.select_nodes_nearest(ug.nodes['x'][n1],count=5)
# ug.add_edge

for n in nbrs:
    try:
        ug.add_edge(nodes=[n1,n])
    except ug.InvalidEdge:
        print "Skipping %d - %d"%(n1,n)
        
plt.clf()
ug.plot_edges()
ug.plot_nodes()
ug.plot_cells(centers=True)
plt.axis(z)
