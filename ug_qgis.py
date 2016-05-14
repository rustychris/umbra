import unstructured_grid
import numpy as np
import os

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry, QgsFeatureRequest )
from qgis.gui import QgsMapTool
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

    direct_edits=False # edits come through map tool
    # direct_edits=True # edits come through layer edit operations

    # Callbacks installed on the grid
    # instrument the grid to propagate changes back to the UI
    def on_modify_node(self,g,func_name,n,**k):
        if 'x' in k:
            if not self.direct_edits:
                fid=self.g.nodes[n]['feat_id']
                geom=self.node_geometry(n)
                self.nl.dataProvider().changeGeometryValues({fid:geom})
                self.nl.triggerRepaint()
            
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

    def node_geometry(self,n):
        return QgsGeometry.fromPoint(QgsPoint(self.g.nodes['x'][n,0],
                                              self.g.nodes['x'][n,1]))
        
    def populate_nodes(self):
        layer=self.nl
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing point memory layer, adds in nodes from g
        feats=[]
        for n in range(self.g.Nnodes()):
            geom = self.node_geometry(n)
            feat = QgsFeature() # can't set feature_ids
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)

        self.g.nodes['feat_id'] = [f.id() for f in outFeats]

        if self.direct_edits:
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
    def populate_all(self,iface):
        canvas=iface.mapCanvas()
        self.iface=iface
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

        # Create a group for the layers -
        li=iface.legendInterface()

        grp_name="grid001"
        for gidx,gname in enumerate(li.groups()):
            if gname==grp_name:
                li.removeGroup(gidx)
        
        group_index=li.addGroup("grid001")

        for layer in [self.cl,self.el,self.nl]:
            # add layer to the registry
            QgsMapLayerRegistry.instance().addMapLayer(layer)
            li.moveLayer(layer,group_index)

        li.currentLayerChanged.connect(self.on_layer_changed)
            
        # set extent to the extent of our layer
        # skip while developing
        # canvas.setExtent(layer.extent())
        
        self.tool=UgTool(canvas,self)
        canvas.setMapTool(self.tool)

    def __del__(self):
        print "UgQgis deleted"
        li=self.iface.legendInterface()
        li.currentLayerChanged.disconnect(self.on_layer_changed)

    def on_layer_changed(self,layer):
        if layer == self.nl:
            # this much works..
            print "Setting map tool to ours"
            self.iface.mapCanvas().setMapTool(self.tool)
        
# remarkably, this all works and is relatively fast.
# Next: 
# 3. allow for editing nodes and propagating that to edges and cells.
# more involved editing modes: 
#   adding nodes, sync back to unstructured grid.
#   dragging edge endpoints which update cells and nodes
#   drawing polygons in the cell layer to then match and/or create nodes/edges on demand.

# keep as a standalone script for a while - much easier to develop than plugin.

# invoke this in the python console as:
# ug_qgis.populate_all(iface)

# Editing a node -
#   start with simplest
#    1. moving a node is recognized by the grid DONE
#    2. that propagates updates to the edge and cell layers DONE


# Editing design:
#   Option A: edits go through the respective layers.
#    -- less custom machinery
#    -- mostly just wiring up the modification events to percolate from one
#       layer to the others
#    -- no extra work on the UI side

#   Option B: edits go through an view/controller which forwards
#     the modifications to the layers via the grid.
#    -- more flexible
#    -- avoids awkward modification routing
#    -- more work early on, possibly have to transition to plugin architecture sooner.


# already have a pilot test for Option A
# try a pilot test for option B - can we listen in on some mouse events?
# canvas=iface.mapCanvas() # => QgsMapCanvas
# can get keyPress and keyRelease from its signals.
# changes in mapTools
# mouseLastXY

# better to go with the maptool approach, like here:
# http://gis.stackexchange.com/questions/45094/how-to-programatically-check-for-a-mouse-click-in-qgis

class UgTool(QgsMapTool):
    def __init__(self, canvas, ug_qgis):
        QgsMapTool.__init__(self, canvas)
        self.canvas = canvas
        self.ug_qgis=ug_qgis
        self.select_node=None

    def canvasPressEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()

        point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)
        xy=[point.x(),point.y()]

        n=self.ug_qgis.g.select_nodes_nearest(xy)
        print "Nearest node is ",n
        self.select_node=n

    def canvasMoveEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)

    def canvasReleaseEvent(self, event):
        #Get the click
        x = event.pos().x()
        y = event.pos().y()

        point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)
        xy=[point.x(),point.y()]
        if self.select_node is not None:
            n=self.select_node
            self.select_node=None
            print "Modifying location of node %d"%n
            self.ug_qgis.g.modify_node(n,x=xy)

    def activate(self):
        print "active"

    def deactivate(self):
        print "inactive"

    def isZoomTool(self):
        return False

    def isTransient(self):
        return False

    def isEditTool(self):
        return True

    
uq=UgQgis(g)


