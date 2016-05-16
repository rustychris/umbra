import unstructured_grid
import numpy as np
from collections import defaultdict
import os

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry, QgsFeatureRequest )
from qgis.gui import QgsMapTool
from qgis import utils
from PyQt4.QtCore import QVariant, Qt
from PyQt4.QtGui import QCursor, QPixmap

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
        g.subscribe_after('add_node',self.on_add_node)

        g.subscribe_after('add_edge',self.on_add_edge)
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

    def on_add_node(self,g,func_name,return_value,**k):
        n=return_value
        geom=self.node_geometry(n)
        feat = QgsFeature() # can't set feature_ids
        feat.setGeometry(geom)
        (res, outFeats) = self.nl.dataProvider().addFeatures([feat])

        self.g.nodes['feat_id'][n] = outFeats[0].id()
        self.nl.triggerRepaint()

    def on_add_edge(self,g,func_name,return_value,**k):
        j=return_value
        feat=QgsFeature()
        feat.setGeometry(self.edge_geometry(j))
        (res,outFeats) = self.el.dataProvider().addFeatures([feat])

        self.g.edges['feat_id'][j] = outFeats[0].id()
        self.el.triggerRepaint()
         
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
        
        self.tool=UgEditTool(canvas,self)
        canvas.setMapTool(self.tool)

    def __del__(self):
        print "UgQgis deleted"
        li=self.iface.legendInterface()
        li.currentLayerChanged.disconnect(self.on_layer_changed)

    def on_layer_changed(self,layer):
        if layer is not None:
            # Not sure why, but using layer.id() was triggering an error
            # about the layer being deleted
            if layer.name()==self.nl.name():
                # doesn't ever happen
                print "Setting map tool to ours"
                self.iface.mapCanvas().setMapTool(self.tool)
            else:
                self.log("on_layer_changed, but to id=%s"%layer.name())
                self.log("My node name=%s"%self.nl.name())

    def log(self,s):
        with open(os.path.join(os.path.dirname(__file__),'log'),'a') as fp:
            fp.write(s+"\n")

class UgEditTool(QgsMapTool):
    # All geometric edits go through this tool
    # Edit operations:
    #   add node
    #   delete node
    #   move node
    #   add edge between existing nodes
    #   add edge with one existing node
    #   add edge with two new nodes
    #   delete edge
    #   add cell
    #   delete cell


    # All of those should be possible with mouse actions on the canvas combined
    # with key modifiers, and should not interfere with panning or zooming the canvas

    # Scheme 1:
    #   Click and drag moves a node
    #   Shift engages edge operations:
    #     shift-click starts drawing edges, with nodes selected or created based on radius
    #     so you can draw a linked bunch of edges by holding down shift and clicking along the path
    #   Creating an isolated node is shift click, then release shift before clicking again
    #   Right click signifies delete, choosing the nearest node or edge center within a radius
    #   Cells are toggled with the spacebar or 'c' key, based on current mouse position

    def __init__(self, canvas, ug_qgis):
        QgsMapTool.__init__(self, canvas)
        self.canvas = canvas
        self.ug_qgis=ug_qgis

        # track state of ongoing operations
        self.op_action=None
        self.op_node=None
        
        self.cursor = QCursor(Qt.CrossCursor)

    node_click_pixels=10
    edge_click_pixels=10

    def event_to_item(self,event,types=['node','edge']):
        self.log("Start of event_to_item self=%s"%id(self))
        pix_x = event.pos().x()
        pix_y = event.pos().y()

        map_to_pixel=self.canvas.getCoordinateTransform()
        
        map_point = map_to_pixel.toMapCoordinates(pix_x,pix_y)
        map_xy=[map_point.x(),map_point.y()]
        res={}
        if 'node' in types:
            n=self.ug_qgis.g.select_nodes_nearest(map_xy)
            node_xy=self.ug_qgis.g.nodes['x'][n]
            node_pix_point = map_to_pixel.transform(node_xy[0],node_xy[1])
            dist2= (pix_x-node_pix_point.x())**2 + (pix_y-node_pix_point.y())**2 
            self.log("Distance^2 is %s"%dist2)
            if dist2<=self.node_click_pixels**2:
                # back to pixel space to calculate distance
                res['node']=n
            else:
                res['node']=None
        if 'edge' in types:
            self.log("No edge support yet")
            assert False
        self.log( "End of event_to_item self=%s"%id(self) )
        return res
        
    def canvasPressEvent(self, event):
        super(UgEditTool,self).canvasPressEvent(event)

        if event.button() == Qt.LeftButton:
            # or Qt.ControlModifier
            if event.modifiers() == Qt.NoModifier:
                self.start_move_node(event)
            elif event.modifiers() == Qt.ShiftModifier:
                self.add_edge_or_node(event)
        else:
            self.log("Press event, but not the left button")
            self.clear_op()

        self.log("Press event end")

    def start_move_node(self,event):
        items=self.event_to_item(event,types=['node'])
        if items['node'] is None:
            self.log("Didn't hit a node")
            self.clear_op() # just to be safe
        else:
            n=items['node']
            self.log("canvas press is %s"%n)
            self.op_node=n
            self.op_action='move_node'

    def add_edge_or_node(self,event):
        if self.op_action=='add_edge':
            self.log("Continuing add_edge_or_node")
            last_node=self.op_node
        else:
            self.log("Starting add_edge_or_node")
            last_node=None
            
        self.op_action='add_edge'
        self.op_node=self.select_or_add_node(event)
        if last_node is not None:
            j=self.ug_qgis.g.add_edge(nodes=[last_node,self.op_node])
            self.log("Adding an edge! j=%d"%j)

    def select_or_add_node(self,event):
        items=self.event_to_item(event,types=['node'])
        if items['node'] is None:
            map_point=event.mapPoint()
            map_xy=[map_point.x(),map_point.y()]

            self.log("Creating new node")
            return self.ug_qgis.g.add_node(x=map_xy) # reaching a little deep
        else:
            return items['node']
            
    def log(self,s):
        with open(os.path.join(os.path.dirname(__file__),'log'),'a') as fp:
            fp.write(s+"\n")
    
    # def canvasMoveEvent(self, event):
    #     x = event.pos().x()
    #     y = event.pos().y()
    #     point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)

    def canvasReleaseEvent(self, event):
        super(UgEditTool,self).canvasReleaseEvent(event)
        self.log("Release event top, type=%s"%event.type())

        self.log("Release with op_node=%s self=%s"%(self.op_node,id(self)))
        if self.op_action=='move_node' and self.op_node is not None:
            x = event.pos().x()
            y = event.pos().y()
            point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)
            xy=[point.x(),point.y()]
            self.log( "Modifying location of node %d self=%s"%(self.op_node,id(self)) )
            self.ug_qgis.g.modify_node(self.op_node,x=xy)
            self.clear_op()
        elif self.op_action=='add_edge':
            # all the action happens on the press, and releasing the shift key
            # triggers the end of the add_edge mode
            pass 
        else:
            # think safety, act safely.
            self.clear_op()
        self.log("Release event end")

    #def keyPressEvent(self,event):
    #    super(UgEditTool,self).keyPressEvent(event)
    #    self.log("keyPress")

    def keyReleaseEvent(self,event):
        super(UgEditTool,self).keyReleaseEvent(event)
        # do we check for modifiers, or they key?
        if event.key() == Qt.Key_Shift:
            self.log("released shift")
            if self.op_action=='add_edge':
                self.clear_op()
        # seems like intercepting a shift could get us into trouble
        # ignore() lets it percolate to other interested parties
        event.ignore()

    def clear_op(self):
        self.op_node=None
        self.op_action=None 

    def activate(self):
        self.log("active")
        self.canvas.setCursor(self.cursor)

    def deactivate(self):
        print "inactive"

    def isZoomTool(self):
        return False

    def isTransient(self):
        return False

    def isEditTool(self):
        return True

    
uq=UgQgis(g)
 
uq.populate_all(utils.iface)
 
 
# Next:
# having trouble with the layer-select logic tripping the maptool activation.
# more involved editing modes: 
#   drawing polygons in the cell layer to then match and/or create nodes/edges on demand.

# keep as a standalone script for a while - much easier to develop than plugin.

# invoke this in the python console as:
# ug_qgis.uq.populate_all(iface)

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

# better to go with the maptool approach, like here:
# http://gis.stackexchange.com/questions/45094/how-to-programatically-check-for-a-mouse-click-in-qgis
