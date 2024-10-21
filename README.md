# umbra
Unstructured Mesh Generator / Editor for QGIS

## Layers
A mesh is internally stored as a collection of nodes, a collection of edges connecting those nodes, and a collection of cells formed from cycles
of nodes (equivalently, cycles of edges). In QGIS these are represented as three distinct layers placed in a group. The plugin ensures that the geometry across
those layers is kept in sync. Meshes are selected by selecting any of the respective layers -- it doesn't matter if you select the nodes layer, edges
layer, or cells layer. Selecting the group is not robust, though.

## Mouse / Keyboard
To edit a mesh select one of the corresponding layers in the layer window (cells, edges, nodes), select the Umbra editor tool (the red/orange/yellow/green mesh button), and click in the 
map window. 

* Shift-click creates a node
* Multiple clicks while holding shift creates a sequence of nodes and connecting edges. Clicks near an existing node will connect to the existing node.
* Right-click deletes nodes or edges. Only the *midpoint* of an edge will delete the edge, which can be annoying when the edge is long.
* Ctrl-left click will toggle a cell in the region around the mouse. If the resulting cell would have more nodes than the mesh's max edges the click is ignored.
* Click-and-drag moves existing nodes.

* 'r' applies orthogonalization in the neighborhood around the current mouse position
* 'R' nudges the neighborhood towards a regular curvilinear mesh. When the neighborhood is not topologically a curvilinear mesh this may fail
* 'z' undoes the last operation. For bulk operations this may only undo the individual operations (if you smooth 1000 nodes, 'z' might only undo the last move). 'z' can be applied repeatedly.
* 'Z' will redo the last operation that was undone.
* 'm' merge nodes of an ede
* 's' or 'S' split an edge into two edges, splitting cells
* 'i' insert a node into an edge, no splitting cells
* 'Q' at an inside corner, add an orthogonal quad
* 'q' at an inside corner, add a perpendicular quad
* 'j' join cells
* 't', 'T' triangulate a region
* 'g' grow quads in a region
* Backspace or Delete: delete selected items

If elements of the mesh are selected, some operations will affect the selected elements rather than
an element nearest the mouse. To select elements choose the specific mesh layer (cells/edges/nodes) in 
the layer pane, and use standard Qgis tools to select features.

## Generating Quads
