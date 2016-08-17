from PyQt4.QtGui import *
from PyQt4.QtCore import *
# from qgis.core import 
from qgis.gui import QgsMapTool

import logging
log=logging.getLogger('umbra.editor')

# Copied / boilerplated from cadtools/singlesegmentfindertoolpy
import umbra_layer 

import numpy as np
# import dock_tools

class UmbraEditorTool(QgsMapTool):
    node_click_pixels=10
    edge_click_pixels=10

    #nomod = 0x00000000       # Qt::NoModifier	0x00000000	No modifier key is pressed.
    #shiftmod = 0x02000000    # Qt::ShiftModifier	0x02000000	A Shift key on the keyboard is pressed.
    #ctrlmod = 0x04000000     # Qt::ControlModifier	0x04000000	A Ctrl key on the keyboard is pressed.
    #leftbutton = 1
    #rightbutton = 2
    #midbutton = 4
    #edit_mode = 'edgenode'
    
    def __init__(self, iface, umbra):
        self.iface=iface
        self.canvas = iface.mapCanvas()
        self.umbra=umbra # to get references to the grid
        self.log=log
        
        super(UmbraEditorTool,self).__init__(self.canvas)

        #our own fancy cursor
        self.cursor = QCursor(QPixmap(["16 16 3 1",
                                      "      c None",
                                      ".     c #FF0000",
                                      "+     c #FFFFFF",
                                      "                ",
                                      "       +.+      ",
                                      "      ++.++     ",
                                      "     +.....+    ",
                                      "    +.     .+   ",
                                      "   +.   .   .+  ",
                                      "  +.    .    .+ ",
                                      " ++.    .    .++",
                                      " ... ...+... ...",
                                      " ++.    .    .++",
                                      "  +.    .    .+ ",
                                      "   +.   .   .+  ",
                                      "   ++.     .+   ",
                                      "    ++.....+    ",
                                      "      ++.++     ",
                                      "       +.+      "]))
        # Create actions 
        self.action_coverage_editor = QAction(QIcon(":/plugins/Umbra/icon.png"),
                                              "Edit unstructured mesh geometry",
                                              self.iface.mainWindow())
        self.action_coverage_editor.setCheckable(True) 
        self.action_coverage_editor.setEnabled(True)
        QObject.connect(self.action_coverage_editor, SIGNAL("triggered()"), self.handle_tool_select)
        QObject.connect(self.iface, SIGNAL("currentLayerChanged(QgsMapLayer*)"), self.handle_layer_changed)
        
        # self.state = []
        # track state of ongoing operations
        self.op_action=None
        self.op_node=None
        
    def handle_tool_select(self):
        print "handle_tool_select - probably shouldn't be calling this now"
        if self.action_coverage_editor.isChecked():
            self.canvas.setMapTool(self)
            # And popup the dock?
            self.umbra.dockwidget_show()
        else:
            self.canvas.unsetMapTool(self)
            # self.umbra.dockwidget_hide() # untested

    #def set_edit_mode(self,mode):
    #    self.edit_mode = mode
        
    def activate(self):
        print "Call to activate for the editor tool"
        self.canvas.setCursor(self.cursor)

    def deactivate(self):
        print "Tool deactivated"

    def handle_layer_changed(self):
        """ 
        set the tool to enabled if the new layer is an UmbraLayer
        it's possible that this would be better off in umbra
        """
        enabled=self.umbra.current_layer_is_umbra()

        self.action_coverage_editor.setEnabled( enabled )
        if enabled:
            self.canvas.setMapTool(self)
        else:
            # not sure that this does anything:
            self.canvas.unsetMapTool(self)

    def grid(self):
        # in the future this will be more dynamic, based on selected layer, e.g.
        return self.umbra.current_grid()

    def event_to_item(self,event,types=['node','edge']):
        self.log.info("Start of event_to_item self=%s"%id(self))
        pix_x = event.pos().x()
        pix_y = event.pos().y()

        map_to_pixel=self.canvas.getCoordinateTransform()
        
        map_point = map_to_pixel.toMapCoordinates(pix_x,pix_y)
        map_xy=[map_point.x(),map_point.y()]
        res={}
        g=self.grid()
        if g is None:
            log.info("event_to_item: no grid available")
            return res

        if 'node' in types:
            n=g.select_nodes_nearest(map_xy)
            node_xy=g.nodes['x'][n]
            node_pix_point = map_to_pixel.transform(node_xy[0],node_xy[1])
            dist2= (pix_x-node_pix_point.x())**2 + (pix_y-node_pix_point.y())**2 
            self.log.info("Distance^2 is %s"%dist2)
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
            self.log.info("Distance^2 is %s"%dist2)
            if dist2<=self.edge_click_pixels**2:
                # back to pixel space to calculate distance
                res['edge']=j
            else:
                res['edge']=None
        self.log.info( "End of event_to_item self=%s"%id(self) )
        return res
        
    def canvasPressEvent(self, event):
        super(UmbraEditorTool,self).canvasPressEvent(event)

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
            self.log.info("Press event, but not the left button (%s)"%event.button())
            self.log.info(" with modifiers %s"%( int(event.modifiers())) )
            self.clear_op()

        self.log.info("Press event end")

    def toggle_cell(self,event):
        self.log.info("Got a toggle cell event")
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
            self.log.info("Didn't hit a node")
            self.clear_op() # just to be safe
        else:
            n=items['node']
            self.log.info("canvas press is %s"%n)
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
            self.log.info("Delete press event, but no feature hits")

    def add_edge_or_node(self,event):
        if self.op_action=='add_edge':
            self.log.info("Continuing add_edge_or_node")
            last_node=self.op_node
        else:
            self.log.info("Starting add_edge_or_node")
            last_node=None
            
        self.op_action='add_edge'
        self.op_node=self.select_or_add_node(event)
        if last_node is not None:
            j=self.grid().add_edge(nodes=[last_node,self.op_node])
            self.log.info("Adding an edge! j=%d"%j)

    def select_or_add_node(self,event):
        items=self.event_to_item(event,types=['node'])
        if items['node'] is None:
            map_point=event.mapPoint()
            map_xy=[map_point.x(),map_point.y()]

            self.log.info("Creating new node")
            return self.grid().add_node(x=map_xy) # reaching a little deep
        else:
            return items['node']
            
    # def canvasMoveEvent(self, event):
    #     x = event.pos().x()
    #     y = event.pos().y()
    #     point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)

    def canvasReleaseEvent(self, event):
        super(UmbraEditorTool,self).canvasReleaseEvent(event)
        self.log.info("Release event top, type=%s"%event.type())

        g=self.grid()

        self.log.info("Release with op_node=%s self=%s"%(self.op_node,id(self)))
        if self.op_action=='move_node' and self.op_node is not None:
            x = event.pos().x()
            y = event.pos().y()
            point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)
            xy=[point.x(),point.y()]
            self.log.info( "Modifying location of node %d self=%s"%(self.op_node,id(self)) )
            g.modify_node(self.op_node,x=xy)
            self.clear_op()
        elif self.op_action=='add_edge':
            # all the action happens on the press, and releasing the shift key
            # triggers the end of the add_edge mode
            pass 
        else:
            # think safety, act safely.
            self.clear_op()
        self.log.info("Release event end")

    # def keyPressEvent(self,event):
    #     super(UmbraEditorTool,self).keyPressEvent(event)
    #     self.log.info("keyPress %s"%event.key() )
    #     # weird, but seems that shift comes through, but not 
    #     # space??  doesn't even show up.
    #     if event.key() == Qt.Key_Space:
    #         self.toggle_cell(event)

    def keyReleaseEvent(self,event):
        super(UmbraEditorTool,self).keyReleaseEvent(event)
        # do we check for modifiers, or they key?
        if event.key() == Qt.Key_Shift:
            self.log.info("released shift")
            if self.op_action=='add_edge':
                self.clear_op()
        # seems like intercepting a shift could get us into trouble
        # ignore() lets it percolate to other interested parties
        event.ignore()

    def clear_op(self):
        self.op_node=None
        self.op_action=None 

    def isZoomTool(self):
        return False

    def isTransient(self):
        return False

    def isEditTool(self):
        # not sure about this..
        # return True

        # it /is/ an edit tool, but isEditTool() is only useful when the layer in 
        # question is a proper vector layer.
        return False
