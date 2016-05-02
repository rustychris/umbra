import unstructured_grid
import numpy as np
import os

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry, QgsFeatureRequest)

from PyQt4.QtCore import QVariant

g=unstructured_grid.SuntansGrid(os.path.join( os.environ['HOME'],"src/umbra/Umbra/sample_data/sfbay/"))


class UgQgis(object):
    """
    interfaces an UnstructuredGrid object and Qgis layers
    """
    def __init__(self,g):
        self.g = self.extend_grid(g)

    def extend_grid(self,g):
        g.add_node_field('feat_id',np.zeros(g.Nnodes(),'i4')-1)
        g.add_edge_field('feat_id',np.zeros(g.Nedges(),'i4')-1)
        g.add_cell_field('feat_id',np.zeros(g.Ncells(),'i4')-1)

        # install grid callbacks:
        g.subscribe_after('modify_node',self.on_modify_node)

        return g

    # Callbacks installed on the grid
    # instrument the grid to propagate changes back to the UI
    def on_modify_node(self,g,func_name,n,**k):
        if 'x' in k:
            edge_changes={}
            for j in self.g.node_to_edges(n):
                fid=self.g.edges[j]['feat_id']
                geom=self.edge_geometry(j)
                edge_changes[fid]=geom
            self.el.dataProvider().changeGeometryValues(edge_changes)
            self.el.triggerRepaint()

            cell_changes={}
            for i in self.g.node_to_cells(n):
                fid=self.g.nodes[i]['feat_id']
                geom=self.cell_geometry(i)
                cell_changes[fid]=geom
            self.cl.dataProvider().changeGeometryValues(cell_changes)
            self.cl.triggerRepaint()
                
    def populate_nodes(self):
        layer=self.nl
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing point memory layer, adds in nodes from g
        feats=[]
        for n in range(self.g.Nnodes()):
            geom = QgsGeometry.fromPoint(QgsPoint(self.g.nodes['x'][n,0],
                                                  self.g.nodes['x'][n,1]))
            feat = QgsFeature() # can't set feature_ids
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)

        self.g.nodes['feat_id'] = [f.id() for f in outFeats]

        layer.geometryChanged.connect(self.on_node_geometry_changed)

        return res

    # callbacks from Qgis layers
    def on_node_geometry_changed(self,feat_id,geom):
        xy=geom.asPoint()
        # this should be sped up with a hash table
        n=np.nonzero( self.g.nodes['feat_id']==feat_id )[0][0]
        self.g.modify_node(n,x=xy)

    def edge_geometry(self,j):
        seg=self.g.nodes['x'][self.g.edges['nodes'][j]]
        pnts=[QgsPoint(seg[0,0],seg[0,1]),
              QgsPoint(seg[1,0],seg[1,1])]
        return QgsGeometry.fromPolyline(pnts)

    def populate_edges(self):
        layer=self.el
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing line memory layer, adds in nodes from g
        feats=[]
        for j in self.g.valid_edge_iter():
            geom=self.edge_geometry(j)
            feat = QgsFeature()
            feat.initAttributes(4)
            feat.setAttribute(0,j) 
            # QGIS doesn't know about numpy types
            feat.setAttribute(3,int(self.g.edges['mark'][j]))
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.g.edges['feat_id']=[f.id() for f in outFeats]

        return res

    def cell_geometry(self,i):
        pnts=[QgsPoint(self.g.nodes['x'][n,0],
                       self.g.nodes['x'][n,1])
              for n in self.g.cell_to_nodes(i)]
        return QgsGeometry.fromPolygon([pnts])
        
    def populate_cells(self):
        layer=self.cl
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing line memory layer, adds in nodes from g
        feats=[]
        for i in self.g.valid_cell_iter():
            geom=self.cell_geometry(i)
            feat = QgsFeature()
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        return res

    def clear_layers(self,canvas):
        # bad code - just for dev
        # could use some code like 
        # layers = QgsMapLayerRegistry.instance().mapLayersByName('my_line')
        lr=QgsMapLayerRegistry.instance()
        layers=lr.mapLayers()
        lr.removeMapLayers( layers.keys() )

    # create the memory layers and populate accordingly
    def populate_all(self,canvas):
        self.clear_layers(canvas)

        crs="?crs=epsg:26910" # was 4326
        # create layer
        self.nl = QgsVectorLayer("Point"+crs, "nodes", "memory")
        self.el = QgsVectorLayer("LineString"+crs,"edges","memory")
        self.cl = QgsVectorLayer("Polygon"+crs,"cells","memory")

        pr = self.el.dataProvider()

        # add fields - eventually would be tied in with g.edge_dtype
        pr.addAttributes([QgsField("edge_id",QVariant.Int),
                          QgsField("c0", QVariant.Int),
                          QgsField("c1",  QVariant.Int),
                          QgsField("mark", QVariant.Int)])
        self.el.updateFields() # tell the vector layer to fetch changes from the provider

        self.populate_nodes()
        self.populate_edges()
        self.populate_cells()

        for layer in [self.cl,self.el,self.nl]:
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
uq=UgQgis(g)

