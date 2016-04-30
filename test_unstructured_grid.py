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
ug1=unstructured_grid.UnstructuredGrid.from_trigrid(tg)

#-# 

z=(546849.79838709673, 582787.29838709673, 4081854.8387096776, 4109677.4193548388)

def on_new_node(grid,func_name,**kwargs):
    print "Got signal %s!"%func_name
    print kwargs

ug.subscribe_after('add_node',on_new_node)

n1=ug.add_node(x=[577208.29133064509, 4089969.7580645154])

## 
plt.clf()
ug.plot_edges()
ug.plot_nodes()
ug.plot_cells(centers=True)
plt.axis(z)

## 
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

## 

ug.modify_node(n1,x=[580178.93145161285, 4093520.0352822575])

# move a node
plt.clf()
ug.plot_edges()
ug.plot_nodes()
ug.plot_cells(centers=True)
plt.axis(z)

## 


reload(unstructured_grid)

ug=unstructured_grid.SuntansGrid(os.path.join( os.path.dirname(__file__),
                                              "Umbra/sample_data/sfbay" ))


nbr=ug.select_nodes_nearest([573295.74092741928, 4089100.3024193542])
print "N nodes: ",ug.Nnodes()
ug.delete_node_cascade(nbr)
print "N deleted nodes: ",np.sum(ug.nodes['deleted'])
print "N deleted edges: ",np.sum(ug.edges['deleted'])
print "N nodes after delete: ",ug.Nnodes()

ug.renumber()
print "N nodes after renumber: ",ug.Nnodes()

plt.clf()
coll=ug.plot_edges()

# was a point cache problem.
ug.write_suntans('/home/rusty/test2')

ug2=unstructured_grid.SuntansGrid('/home/rusty/test2')
coll2=ug2.plot_edges()
coll2.set_color('g')


## 

reload(unstructured_grid)
ug2=unstructured_grid.SuntansGrid('/home/rusty/test2')

ug2.write_pickle('/home/rusty/pickle')
#-# 
ug3=unstructured_grid.UnstructuredGrid.from_pickle('/home/rusty/pickle')
print ug3._Listenable__post_listeners
