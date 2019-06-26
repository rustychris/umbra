from stompy.grid import unstructured_grid
six.moves.reload_module(unstructured_grid)

## 
g=unstructured_grid.UnstructuredGrid.from_ugrid('/home/rusty/mirrors/ucd-X/Arc_Hydro/CSC_Project/MODELING/1_Hydro_Model_Files/Geometry/IntermediateCombinedGrids/CacheSloughComplex_v102.nc')

##

zoom=(607049.3789851875, 607112.602248394, 4235362.0737852305, 4235409.18531362)
plt.figure(1).clf()

g.plot_edges(lw=4,color='k',clip=zoom,alpha=0.5)
g.plot_nodes(labeler='id',clip=zoom)
g.plot_cells(lw=0,color='0.5',alpha=0.3,clip=zoom,zorder=-3,
             labeler=lambda i,r: str(i))
plt.axis(zoom)

##

g.split_edge(x=np.array([607073.725, 4235389.887]),merge_thresh=0.2)
g.split_edge(x=[607078.353, 4235387.52],merge_thresh=0.2)
g.split_edge(x=[607088.845, 4235380.44],merge_thresh=0.2)
g.split_edge(x=[607083.686, 4235384.11],merge_thresh=0.2)

## 

# And merge_cells

g.merge_cells(x=[607077.828, 4235391.108])

g.merge_cells(x=[607074.2437854839, 4235385.862130942])
g.merge_cells(x=[607079.3151117861, 4235383.064157809])
g.merge_cells(x=[607082.2879582392, 4235387.873174131])
g.merge_cells(x=[607087.3592845413, 4235383.938524413])
g.merge_cells(x=[607085.4356780129, 4235379.741564715])
