import unstructured_grid

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry)

from PyQt4.QtCore import QVariant

g=unstructured_grid.SuntansGrid("/home/rusty/models/suntans/spinupdated/rundata/")

def populate_nodes(layer):
    layer.dataProvider().deleteFeatures(layer.allFeatureIds())
    
    # takes an existing point memory layer, adds in nodes from g
    feats=[]
    for n in range(g.Nnodes()):
        geom = QgsGeometry.fromPoint(QgsPoint(g.nodes['x'][n,0],g.nodes['x'][n,1]))
        feat = QgsFeature()
        feat.setGeometry(geom)
        feats.append(feat)
    (res, outFeats) = layer.dataProvider().addFeatures(feats)
    return res


def populate_edges(layer):
    layer.dataProvider().deleteFeatures(layer.allFeatureIds())

    # takes an existing line memory layer, adds in nodes from g
    feats=[]
    segs=g.nodes['x'][g.edges['nodes']]
    for j in g.valid_edge_iter():
        pnts=[QgsPoint(segs[j,0,0],segs[j,0,1]),
              QgsPoint(segs[j,1,0],segs[j,1,1])]
        geom = QgsGeometry.fromPolyline(pnts)
        feat = QgsFeature()
        feat.initAttributes(4)
        feat.setAttribute(0,j) 
        # QGIS doesn't know about numpy types
        feat.setAttribute(3,int(g.edges['mark'][j]))
        feat.setGeometry(geom)
        feats.append(feat)
    (res, outFeats) = layer.dataProvider().addFeatures(feats)
    return res

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
    canvas.setExtent(layer.extent())

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
