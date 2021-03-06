import unstructured_grid
from delft import dfm_grid
reload(unstructured_grid)
reload(dfm_grid)

import utils

import numpy as np
from collections import defaultdict
import os

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry, QgsFeatureRequest )
from qgis.gui import QgsMapTool
import qgis.utils 
from PyQt4.QtCore import QVariant, Qt
from PyQt4.QtGui import QCursor, QPixmap, QKeyEvent

def dist(a,b=None):
    if b is None:
        b=(a*0)
    return np.sqrt(np.sum((a-b)**2,axis=-1))



class UgQgis(object):
    """
    interfaces an UnstructuredGrid object and Qgis layers
    """
    def __init__(self,g):
        self.g = self.extend_grid(g)
        self.install_edge_quality()

    def install_edge_quality(self):
        if 'edge_quality' not in self.g.edges.dtype.names:
            edge_q=np.zeros(self.g.Nedges(),'f8')
            self.g.add_edge_field('edge_quality',edge_q)
            self.update_edge_quality()

    def update_edge_quality(self,edges=None):
        # First part is calculating the values
        if edges is None:
            edges=slice(None)
        g=self.g
        vc=g.cells_center()
        ec=g.edges_center()
        g.edge_to_cells()

        c2c=dist( vc[g.edges['cells'][edges,0]] - vc[g.edges['cells'][edges,1]] )
        A=g.cells_area()
        Acc= A[g.edges['cells'][edges,:]].sum(axis=1)
        c2c=c2c / np.sqrt(Acc) # normalized
        c2c[ np.any(g.edges['cells'][edges,:]<0,axis=1) ] = np.inf
        g.edges['edge_quality'][edges]=c2c
        
    def extend_grid(self,g):
        g.add_node_field('feat_id',np.zeros(g.Nnodes(),'i4')-1)
        g.add_edge_field('feat_id',np.zeros(g.Nedges(),'i4')-1)
        g.add_cell_field('feat_id',np.zeros(g.Ncells(),'i4')-1)

        # install grid callbacks:
        if 1: # re-enabled. DBG - temp. disabled
            g.subscribe_after('modify_node',self.on_modify_node)
            g.subscribe_after('add_node',self.on_add_node)
            g.subscribe_after('add_edge',self.on_add_edge)

            g.subscribe_after('add_cell',self.on_add_cell)

            g.subscribe_before('delete_edge',self.on_delete_edge)
            g.subscribe_before('delete_node',self.on_delete_node)
            g.subscribe_before('delete_cell',self.on_delete_cell)

        return g

    def on_delete_node(self,g,func_name,n,**k):
        # self.log('got signal for delete node')
        self.nl.dataProvider().deleteFeatures([self.g.nodes['feat_id'][n]])
        self.nl.triggerRepaint()

    def on_delete_edge(self,g,func_name,j,**k):
        # self.log('got signal for delete edge')
        self.el.dataProvider().deleteFeatures([self.g.edges['feat_id'][j]])
        self.el.triggerRepaint()

    def on_delete_cell(self,g,func_name,c,**k):
        # this one isn't working, while the others are...
        feat_id=self.g.cells['feat_id'][c]
        self.log('got signal for delete cell %d, feat_id %s'%(c,feat_id))
        self.cl.dataProvider().deleteFeatures([feat_id])
        self.cl.triggerRepaint()

    # Callbacks installed on the grid
    # instrument the grid to propagate changes back to the UI
    def on_modify_node(self,g,func_name,n,**k):
        if 'x' in k:
            fid=self.g.nodes[n]['feat_id']
            geom=self.node_geometry(n)
            self.nl.dataProvider().changeGeometryValues({fid:geom})
            self.nl.triggerRepaint()

            # update cells first, so that edge_quality has fresh
            # cell center and area information
            cell_changes={}
            cells=self.g.node_to_cells(n)
            self.g.cells_center(refresh=cells)
            self.g.cells['_area'][cells]=np.nan # trigger recalc.
            cell_edges=set()
            for i in cells:
                # this was all sorts of messed up - don't understand how
                # it was working at all before...
                fid=self.g.cells[i]['feat_id']
                geom=self.cell_geometry(i)
                cell_changes[fid]=geom
                cell_edges.update(self.g.cell_to_edges(i))
            self.cl.dataProvider().changeGeometryValues(cell_changes)
            self.cl.triggerRepaint()
                
            edge_geom_changes={}
            # edge centers are not cached at this point, so don't
            # need to update them...
            for j in self.g.node_to_edges(n):
                fid=self.g.edges[j]['feat_id']
                geom=self.edge_geometry(j)
                edge_geom_changes[fid]=geom
            self.el.dataProvider().changeGeometryValues(edge_geom_changes)

            # Edges for which a node or cell has changed:
            # this doesn't seem to be working now.
            edge_attr_changes={}
            edge_quality_idx=[i
                              for i,attr in enumerate(self.e_attrs)
                              if attr.name()=='edge_quality'][0]
            js=list(cell_edges)
            self.update_edge_quality(js)
            for j in js:
                # and update edge quality field - would be nice
                # to come up with a nice abstraction here...
                fid=self.g.edges[j]['feat_id']
                edge_attr_changes[fid]={edge_quality_idx:float(self.g.edges[j]['edge_quality'])}
            self.el.dataProvider().changeAttributeValues(edge_attr_changes)

            self.el.triggerRepaint()

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

    def on_add_cell(self,g,func_name,return_value,**k):
        c=return_value
        self.log('got signal for add cell')
        geom=self.cell_geometry(c)
        feat = QgsFeature() # can't set feature_ids
        feat.setGeometry(geom)
        (res, outFeats) = self.cl.dataProvider().addFeatures([feat])

        self.g.cells['feat_id'][c] = outFeats[0].id()
        self.cl.triggerRepaint()
         
    def node_geometry(self,n):
        return QgsGeometry.fromPoint(QgsPoint(self.g.nodes['x'][n,0],
                                              self.g.nodes['x'][n,1]))
        
    def populate_nodes(self):
        layer=self.nl
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing point memory layer, adds in nodes from g
        feats=[]
        valid=[]
        for n in range(self.g.Nnodes()):
            valid.append(n)
            geom = self.node_geometry(n)
            feat = QgsFeature() # can't set feature_ids
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.g.nodes['feat_id'][valid] = [f.id() for f in outFeats]

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
        valid=[]
        for j in self.g.valid_edge_iter():
            geom=self.edge_geometry(j)
            valid.append(j)
            feat = QgsFeature()
            feat.initAttributes(len(self.e_attrs))
            for idx,eattr in enumerate(self.e_attrs):
                name=eattr.name()
                typecode=eattr.type()
                if name=='edge_id':
                    feat.setAttribute(idx,j) 
                elif name=='c0':
                    feat.setAttribute(idx,int(self.g.edges['cells'][j,0]))
                elif name=='c1':
                    feat.setAttribute(idx,int(self.g.edges['cells'][j,1]))
                elif typecode==2: # integer
                    feat.setAttribute(idx,int(self.g.edges[name][j]))
                elif typecode==6: # double
                    feat.setAttribute(idx,float(self.g.edges[name][j]))
                else:
                    continue
                # QGIS doesn't know about numpy types
            
            # feat.setAttribute(3,int(self.g.edges['mark'][j]))
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.g.edges['feat_id'][valid]=[f.id() for f in outFeats]

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
        valid=[]
        for i in self.g.valid_cell_iter():
            geom=self.cell_geometry(i)
            feat = QgsFeature()
            feat.setGeometry(geom)
            feats.append(feat)
            valid.append(i)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.g.cells['feat_id'][valid]=[f.id() for f in outFeats]
        return res

    def clear_layers(self,canvas):
        # more for development - clear out old layers
        lr=QgsMapLayerRegistry.instance()
        layers=lr.mapLayers()
        to_remove=[]
        for layer_key in layers.keys():
            if ( layer_key.startswith('edges') or
                 layer_key.startswith('cells') or
                 layer_key.startswith('nodes') ):
                to_remove.append(layer_key)
        lr.removeMapLayers( to_remove )

    def on_node_layer_deleted(self):
        self.nl=None
        self.log("signal received: node layer deleted")
        # to avoid confusion, let the node layer removal trigger
        # everything, at least for now.  
        self.graceful_deactivate()
    def on_edge_layer_deleted(self):
        self.el=None
        self.log("signal received: edge layer deleted")
    def on_cell_layer_deleted(self):
        self.cl=None
        self.log("signal received: cell layer deleted")

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

        self.nl.layerDeleted.connect(self.on_node_layer_deleted)
        self.el.layerDeleted.connect(self.on_edge_layer_deleted)
        self.cl.layerDeleted.connect(self.on_cell_layer_deleted)

        pr = self.el.dataProvider()

        # add fields - eventually would be tied in with g.edge_dtype
        e_attrs=[QgsField("edge_id",QVariant.Int)]

        for fidx,fdesc in enumerate(self.g.edge_dtype.descr):
            # descr gives string reprs of the types, use dtype
            # to get back to an object.
            fname=fdesc[0] ; ftype=np.dtype(fdesc[1])
            if len(fdesc)>2:
                fshape=fdesc[2]
            else:
                fshape=None

            if fname=='nodes':
                continue
            elif fname=='cells':
                e_attrs += [QgsField("c0", QVariant.Int),
                            QgsField("c1", QVariant.Int)]
            else:
                if np.issubdtype(ftype,np.int):
                    e_attrs.append( QgsField(fname,QVariant.Int) )
                elif np.issubdtype(ftype,np.float):
                    e_attrs.append( QgsField(fname,QVariant.Double) )
                else:
                    self.log("Not read other datatypes")
        self.e_attrs=e_attrs            
        pr.addAttributes(e_attrs)
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

        # if this is omitted, be sure to also skip over
        # the disconnects.
        li.currentLayerChanged.connect(self.on_layer_changed)
            
        # set extent to the extent of our layer
        # skip while developing
        # canvas.setExtent(layer.extent())

        self.tool=UgEditTool(canvas,self)
        canvas.setMapTool(self.tool)

    def graceful_deactivate(self):
        self.log("UgQgis deactivating")
        li=self.iface.legendInterface()
        li.currentLayerChanged.disconnect(self.on_layer_changed)
        self.log("currentLayerChanged callback disconnected")
        # remove map tool from the canvas
        canvas=self.iface.mapCanvas()
        if canvas.mapTool() == self.tool:
            self.log("active map tool is ours - how to remove??")
            self.tool=None
        # remove callbacks from the grid:
        self.log("Not ready for removing callbacks from the grid")

    def on_layer_changed(self,layer):
        if layer is not None:
            # Not sure why, but using layer.id() was triggering an error
            # about the layer being deleted.  Hmm - this is still somehow
            # related to an error about the layer being deleted.
            self.log("About to check names")
            if layer.name()==self.nl.name():
                # this happens, but doesn't seem to succeed
                self.log("Setting map tool to ours")
                self.iface.mapCanvas().setMapTool(self.tool)
                self.log("Done with setting map tool to ours")
            else:
                self.log("on_layer_changed, but to id=%s"%layer.name())
                self.log("My node name=%s"%self.nl.name())

    def log(self,s):
        with open(os.path.join(os.path.dirname(__file__),'log'),'a') as fp:
            fp.write(s+"\n")
            fp.flush()

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
    #   Click and drag moves a node - done
    #   Shift engages edge operations:
    #     shift-click starts drawing edges, with nodes selected or created based on radius
    #     so you can draw a linked bunch of edges by holding down shift and clicking along the path
    #     done.
    #   Creating an isolated node is shift click, then release shift before clicking again
    #     done.
    #   Right click signifies delete, choosing the nearest node or edge center within a radius
    #     done.
    #   Cells are toggled with the spacebar or 'c' key, based on current mouse position

    def __init__(self, canvas, ug_qgis):
        # maybe this is safer??
        super(UgEditTool,self).__init__(canvas)
        self.canvas = canvas
        self._g=ug_qgis.g # should do this more based on the active layer...

        # track state of ongoing operations
        self.op_action=None
        self.op_node=None
        
        self.cursor = QCursor(Qt.CrossCursor)

    node_click_pixels=10
    edge_click_pixels=10

    def grid(self):
        # in the future this will be more dynamic, based on selected layer, e.g.
        return self._g

    def event_to_item(self,event,types=['node','edge']):
        self.log("Start of event_to_item self=%s"%id(self))
        pix_x = event.pos().x()
        pix_y = event.pos().y()

        map_to_pixel=self.canvas.getCoordinateTransform()
        
        map_point = map_to_pixel.toMapCoordinates(pix_x,pix_y)
        map_xy=[map_point.x(),map_point.y()]
        res={}
        g=self.grid()
        if 'node' in types:
            n=g.select_nodes_nearest(map_xy)
            node_xy=g.nodes['x'][n]
            node_pix_point = map_to_pixel.transform(node_xy[0],node_xy[1])
            dist2= (pix_x-node_pix_point.x())**2 + (pix_y-node_pix_point.y())**2 
            self.log("Distance^2 is %s"%dist2)
            if dist2<=self.node_click_pixels**2:
                # back to pixel space to calculate distance
                res['node']=n
            else:
                res['node']=None
        if 'edge' in types:
            j=g.select_edges_nearest(map_xy)
            edge_xy=g.edges_center()[j]
            edge_pix_point = map_to_pixel.transform(edge_xy[0],edge_xy[1])
            dist2= (pix_x-edge_pix_point.x())**2 + (pix_y-edge_pix_point.y())**2 
            self.log("Distance^2 is %s"%dist2)
            if dist2<=self.edge_click_pixels**2:
                # back to pixel space to calculate distance
                res['edge']=j
            else:
                res['edge']=None
        self.log( "End of event_to_item self=%s"%id(self) )
        return res
        
    def canvasPressEvent(self, event):
        super(UgEditTool,self).canvasPressEvent(event)

        if event.button()==Qt.LeftButton:
            # or Qt.ControlModifier
            if event.modifiers() == Qt.NoModifier:
                self.start_move_node(event)
            elif event.modifiers() == Qt.ShiftModifier:
                self.add_edge_or_node(event)
            elif event.modifiers() == Qt.ControlModifier:
                self.toggle_cell(event)
        elif (event.button()==Qt.RightButton) and (event.modifiers()==Qt.NoModifier):
            # mostly delete operations
            self.delete_edge_or_node(event)
        else:
            self.log("Press event, but not the left button")
            self.clear_op()

        self.log("Press event end")

    def toggle_cell(self,event):
        self.log("Got a toggle cell event")
        if isinstance(event,QKeyEvent): 
            # if called with a keypress event, but that wasn't working...
            # A little trickier, because event here is a keypress (space bar)
            mouse_pnt=self.canvas.mouseLastXY()
        else:
            # from a mouse event
            mouse_pnt=event.pos()

        pix_x=mouse_pnt.x()
        pix_y=mouse_pnt.y()
        map_to_pixel=self.canvas.getCoordinateTransform()
        map_point = map_to_pixel.toMapCoordinates(pix_x,pix_y)
        map_xy=[map_point.x(),map_point.y()]
        
        # First, does it fall within an existing cell?
        self.grid().toggle_cell_at_point(map_xy)

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

    def delete_edge_or_node(self,event):
        items=self.event_to_item(event,types=['node','edge'])
        if items['node'] is not None:
            self.grid().delete_node_cascade(items['node'])
            self.clear_op() # just to be safe
        elif items['edge'] is not None:
            self.grid().delete_edge_cascade(items['edge'])
            self.clear_op() # safety first
        else:
            self.log("Delete press event, but no feature hits")

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
            j=self.grid().add_edge(nodes=[last_node,self.op_node])
            self.log("Adding an edge! j=%d"%j)

    def select_or_add_node(self,event):
        items=self.event_to_item(event,types=['node'])
        if items['node'] is None:
            map_point=event.mapPoint()
            map_xy=[map_point.x(),map_point.y()]

            self.log("Creating new node")
            return self.grid().add_node(x=map_xy) # reaching a little deep
        else:
            return items['node']
            
    def log(self,s):
        with open(os.path.join(os.path.dirname(__file__),'log'),'a') as fp:
            fp.write(s+"\n")
            fp.flush()
    
    # def canvasMoveEvent(self, event):
    #     x = event.pos().x()
    #     y = event.pos().y()
    #     point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)

    def canvasReleaseEvent(self, event):
        super(UgEditTool,self).canvasReleaseEvent(event)
        self.log("Release event top, type=%s"%event.type())

        g=self.grid()

        self.log("Release with op_node=%s self=%s"%(self.op_node,id(self)))
        if self.op_action=='move_node' and self.op_node is not None:
            x = event.pos().x()
            y = event.pos().y()
            point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)
            xy=[point.x(),point.y()]
            self.log( "Modifying location of node %d self=%s"%(self.op_node,id(self)) )
            g.modify_node(self.op_node,x=xy)
            self.clear_op()
        elif self.op_action=='add_edge':
            # all the action happens on the press, and releasing the shift key
            # triggers the end of the add_edge mode
            pass 
        else:
            # think safety, act safely.
            self.clear_op()
        self.log("Release event end")

    # def keyPressEvent(self,event):
    #     super(UgEditTool,self).keyPressEvent(event)
    #     self.log("keyPress %s"%event.key() )
    #     # weird, but seems that shift comes through, but not 
    #     # space??  doesn't even show up.
    #     if event.key() == Qt.Key_Space:
    #         self.toggle_cell(event)

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
        self.log("inactive")

    def isZoomTool(self):
        return False

    def isTransient(self):
        return False

    def isEditTool(self):
        return True


## 

if 1:
    ugrid_fn="grid-ugrid.nc"
    if not os.path.exists(ugrid_fn):
        dfm_fn=os.path.join( os.environ['HOME'],"models/grids/sfbd-grid-southbay/SFEI_SSFB_fo_dwrbathy_net.nc")
        g=dfm_grid.DFMGrid(dfm_fn)
        g.write_ugrid(ugrid_fn)
    g=unstructured_grid.UnstructuredGrid.from_ugrid(ugrid_fn)
else:
    g=unstructured_grid.SuntansGrid(os.path.join( os.environ['HOME'],"src/umbra/Umbra/sample_data/sfbay/"))

## 
uq=UgQgis(g)
 
uq.populate_all(qgis.utils.iface)
 
 
# Next:
#   possible bug: create closed cycle of edges, add as cell, remove an edge
#         => doesn't cascade to the cell, and the cell can continue to get edits.

# Editing design:
#   Option B: edits go through an view/controller which forwards
#     the modifications to the layers via the grid.
#    -- more flexible
#    -- avoids awkward modification routing
#    -- more work early on, possibly have to transition to plugin architecture sooner.

# better to go with the maptool approach, like here:
# http://gis.stackexchange.com/questions/45094/how-to-programatically-check-for-a-mouse-click-in-qgis


# issue of map tool getting unset -- stopgap:
#   manually calling iface.mapCanvas().setMapTool(ug_qgis.uq.tool)
# long-term - deal with this when transitioning to real plugin/tool.


## 
if 0:
    reload(unstructured_grid)
    g=unstructured_grid.UnstructuredGrid()


    n1=g.add_node(x=[0,0])
    n2=g.add_node(x=[1,2])
    n3=g.add_node(x=[0,5])

    j1=g.add_edge(nodes=[n1,n2])
    j2=g.add_edge(nodes=[n2,n3])
    j3=g.add_edge(nodes=[n3,n1])

    plt.figure(1).clf()
    g.plot_nodes()
    g.plot_edges()
    g.plot_cells()

    c=g.toggle_cell_at_point(x=[0.5,2])
    c1=g.toggle_cell_at_point(x=[3,2])
    c2=g.toggle_cell_at_point(x=[0.5,2])

    ## 
    x=[0.5,2]

    # lame stand-in for a true bounding polygon test
    edges_near = g.select_edges_nearest(x,count=6)

    # HERE: refactor unstructured_grid.find_cycles()
    # so that it can find cycles with a given subset of either
    # edges or nodes.  
    # ultimately, we'll do a select_cells_nearest(inside=True),
    # to see whether the clicked point is already in a cell.
    potential_cells=g.find_cycles(max_cycle_len=7,starting_edges=edges_near)

