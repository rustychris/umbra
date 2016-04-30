import unstructured_grid
import numpy as np
import os

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry)

from PyQt4.QtCore import QVariant

g=unstructured_grid.SuntansGrid(os.path.join( os.environ['HOME'],"src/umbra/Umbra/sample_data/sfbay/"))


g.add_node_field('feat_id',np.zeros(g.Nnodes(),'i4')-1)
g.add_edge_field('feat_id',np.zeros(g.Nedges(),'i4')-1)
g.add_cell_field('feat_id',np.zeros(g.Ncells(),'i4')-1)

# instrument the grid to propagate changes back to the UI
def on_modify_node(g,func_name,n,**k):
    if 'x' in k:
        for j in g.node_to_edges(n):
            print "update geometry of edge %d"%j
            
g.subscribe_after('modify_node',on_modify_node)


def populate_nodes(layer):
    layer.dataProvider().deleteFeatures(layer.allFeatureIds())
    
    # takes an existing point memory layer, adds in nodes from g
    feats=[]
    for n in range(g.Nnodes()):
        geom = QgsGeometry.fromPoint(QgsPoint(g.nodes['x'][n,0],g.nodes['x'][n,1]))
        # doesn't work to set the feature id...
        feat = QgsFeature()
        feat.setGeometry(geom)
        feats.append(feat)
    (res, outFeats) = layer.dataProvider().addFeatures(feats)

    g.nodes['feat_id'] = [f.id() for f in outFeats]

    if 1:
        def on_node_geometry_changed(feat_id,geom):
            xy=geom.asPoint()
            n=np.nonzero( g.nodes['feat_id']==feat_id )[0][0]
            g.modify_node(n,x=xy)
        layer.geometryChanged.connect(on_node_geometry_changed)

    return res


def edge_geometry(g,j):
    seg=g.nodes['x'][g.edges['nodes'][j]]
    pnts=[QgsPoint(segs[0,0],segs[0,1]),
          QgsPoint(segs[1,0],segs[1,1])]
    return QgsGeometry.fromPolyline(pnts)
    
def populate_edges(layer):
    layer.dataProvider().deleteFeatures(layer.allFeatureIds())

    # takes an existing line memory layer, adds in nodes from g
    feats=[]
    for j in g.valid_edge_iter():
        geom=edge_geometry(g,j)
        feat = QgsFeature()
        feat.initAttributes(4)
        feat.setAttribute(0,j) 
        # QGIS doesn't know about numpy types
        feat.setAttribute(3,int(g.edges['mark'][j]))
        feat.setGeometry(geom)
        feats.append(feat)
    (res, outFeats) = layer.dataProvider().addFeatures(feats)
    g.edges['feat_id']=[f.id() for f in outFeats]

    return res

# HERE:
#   need to store a reference to the edge layer so that update_edge_geometry
#   can push a new edge_geometry(g,j) to the layer.
#   nearing time to wrap this into a class.

def populate_cells(layer):
    layer.dataProvider().deleteFeatures(layer.allFeatureIds())

    # takes an existing line memory layer, adds in nodes from g
    feats=[]
    for i in g.valid_cell_iter():
        pnts=[QgsPoint(g.nodes['x'][n,0],g.nodes['x'][n,1])
              for n in g.cell_to_nodes(i)]
        geom = QgsGeometry.fromPolygon([pnts])
        feat = QgsFeature()
        feat.setGeometry(geom)
        feats.append(feat)
    (res, outFeats) = layer.dataProvider().addFeatures(feats)
    return res

def clear_layers(canvas):
    # could use some code like 
    # layers = QgsMapLayerRegistry.instance().mapLayersByName('my_line')
    lr=QgsMapLayerRegistry.instance()
    layers=lr.mapLayers()
    lr.removeMapLayers( layers.keys() )
    

# create the memory layers and populate accordingly
def populate_all(canvas):
    clear_layers(canvas)

    crs="?crs=epsg:4326"
    # create layer
    nl = QgsVectorLayer("Point"+crs, "nodes", "memory")
    el = QgsVectorLayer("LineString"+crs,"edges","memory")
    cl = QgsVectorLayer("Polygon"+crs,"cells","memory")

    pr = el.dataProvider()

    # add fields - eventually would be tied in with g.edge_dtype
    pr.addAttributes([QgsField("edge_id",QVariant.Int),
                      QgsField("c0", QVariant.Int),
                      QgsField("c1",  QVariant.Int),
                      QgsField("mark", QVariant.Int)])
    el.updateFields() # tell the vector layer to fetch changes from the provider
    
    populate_nodes(nl)
    populate_edges(el)
    populate_cells(cl)

    for layer in [cl,el,nl]:
        # add layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(layer)

    # set extent to the extent of our layer
    # skip while developing
    # canvas.setExtent(layer.extent())

# remarkably, this all works and is relatively fast.
# Next: 
# 1. populate attributes dynamically from dtype attrs.
# 2. add code to group the layers
# 3. allow for editing nodes and propagating that to edges and cells.
# more involved editing modes: 
#   adding nodes, sync back to unstructured grid.
#   dragging edge endpoints which update cells and nodes
#   drawing polygons in the cell layer to then match and/or create nodes/edges on demand.

# keep as a standalone script for a while - much easier to develop than plugin.

# invoke this in the python console as:
# ug_qgis.populate_all(iface.mapCanvas())

# Editing a node -
#   start with simplest
#    1. moving a node is recognized by the grid
#    2. that propagates updates to the edge and cell layers

# via the geometryChanged slot?
# QgsMapLayerRegistry.instance().addMapLayer(layer)

