# First try of a MapTool to support editing the coverage.
# steps:
# 1. Get it to print out the point that was clicked
# 2. Select a vertex based on the click
# 3. Drag a vertex
# 4. Drag a vertex with live feedback


from PyQt4.QtGui import *
from PyQt4.QtCore import *
# from qgis.core import 
from qgis.gui import QgsMapTool

# Copied / boilerplated from cadtools/singlesegmentfindertoolpy
from umbra_layer import UmbraLayer

import numpy as np
# import dock_tools

class UmbraEditorTool(QgsMapTool):
    px_tolerance = 8
    nomod = 0x00000000       # Qt::NoModifier	0x00000000	No modifier key is pressed.
    shiftmod = 0x02000000    # Qt::ShiftModifier	0x02000000	A Shift key on the keyboard is pressed.
    ctrlmod = 0x04000000     # Qt::ControlModifier	0x04000000	A Ctrl key on the keyboard is pressed.
    leftbutton = 1
    rightbutton = 2
    midbutton = 4
    edit_mode = 'edgenode'
    
    def __init__(self, iface, toolBar=None):
        self.iface = iface
        self.canvas=iface.mapCanvas()
        super(UmbraEditorTool,self).__init__(self.canvas)
        #self.rb1 = QgsRubberBand(self.canvas,  False)
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
        
        # toolBar.addAction(self.action_coverage_editor)

        self.state = []

    dock_widget = None
    def handle_tool_select(self):
        print "handle_tool_select - probably shouldn't be calling this now"
        if self.action_coverage_editor.isChecked():
            self.canvas.setMapTool(self)
            # And popup the dock?
            if self.dock_widget is None:
                self.dock_widget = dock_tools.GridEditDockWidget(self.iface,self)
            self.dock_widget.show()
            
        else:
            self.canvas.unsetMapTool(self)
            self.dock_widget.hide()

    def set_edit_mode(self,mode):
        self.edit_mode = mode
        
    def activate(self):
        print "Call to activate for the editor tool"
        
        self.canvas.setCursor(self.cursor)

    def deactivate(self):
        print "Tool deactivated"

    def handle_layer_changed(self):
        """ set the tool to enabled if the new layer is an UmbraLayer """

        clayer = self.canvas.currentLayer()

        # occasionally happens, probaly related to stale code after a reload
        try:
            enabled = (UmbraLayer is not None) and isinstance(clayer,UmbraLayer)
        except TypeError:
            print "What - UmbraLayer is",UmbraLayer
            enabled=False

        print "Enabled is now set to",enabled
        self.action_coverage_editor.setEnabled( enabled )
        if enabled:
            self.canvas.setMapTool(self)
        else:
            self.canvas.unsetMapTool(self)

    def event_to_target(self,event,px_tolerance=None,target_type='node'):
        if px_tolerance is None:
            px_tolerance = self.px_tolerance
            
        xy = [event.pos().x(), event.pos().y()]
        map2pixel = self.canvas.getCoordinateTransform()
        
        qxy = map2pixel.toMapCoordinates(xy[0],xy[1])
        qxy2 = map2pixel.toMapCoordinates(xy[0]+1,xy[1]+1)
        
        map_x = qxy.x()
        map_y = qxy.y()

        # approx geographic distance associated with one pixel
        pxscale = abs(map_x - qxy2.x())
        
        # is it close to an existing node?
        # how to get native coordinates from pixel?
        layer = self.canvas.currentLayer()

        if target_type == 'node':
            i = layer.find_closest_node([map_x,map_y])
            print "Layer says closest node is ",i
            # how far away is it?
            px_dist = layer.distance_to_node([map_x,map_y],i) / pxscale
            print "And pixel distance is ",px_dist
            result = ('node',i)
        elif target_type == 'cell':
            i = layer.find_closest_cell([map_x,map_y])
            # how far away is it?
            px_dist = layer.distance_to_cell([map_x,map_y],i) / pxscale
            result = ('cell',i)

        print "px_tolerance is ",px_tolerance
        if px_dist < px_tolerance:
            return result
        else:
            return ('point',(map_x,map_y))

    def keyPressEvent(self,event):
        txt = event.text()
        print "Got a key press - ",txt
        if txt == "m":
            print "Would display menu"
        elif txt == 's':
            print "Will save..."
            self.canvas.currentLayer().saveGui()
        elif txt == 'r':
            print "Relaxing"
            # need to query to find out where the cursor is
            # self.canvas.currentLayer().relax()
            p = self.canvas.mouseLastXY()
            map2pixel = self.canvas.getCoordinateTransform()
            qxy = map2pixel.toMapCoordinates(p.x(),p.y())
            map_xy = [qxy.x(), qxy.y()]
            self.canvas.currentLayer().relax_at_point(map_xy)
            
    def canvasPressEvent(self,event):
        #Get the click
        
        # target = ('node',i)  or ('point',(x,y))
        if self.edit_mode == 'optimize':
            target_type,target_id = self.event_to_target(event,px_tolerance=np.inf,target_type='cell')
        else:
            target_type,target_id = self.event_to_target(event)

        mods = event.modifiers()
        button = event.button()

        if self.edit_mode == 'edgenode': # for now the one and only.
            if button == self.rightbutton:
                print "Delete"
                cmd = 'del'
            elif mods & self.shiftmod:
                print "Move"
                cmd = 'move'
            else:
                print "Add"
                cmd = 'add'
        elif self.edit_mode == 'cell_toggle':
            print "Cell toggle"
            cmd = 'cell_toggle'
        elif self.edit_mode == 'optimize':
            print "Optimize"
            if button ==self.leftbutton and (mods & self.shiftmod):
                cmd = 'repave'
            else:
                cmd = 'relax'
        else:
            print "Unknown mouse action"
            return

        self.state.append( (cmd,target_type,target_id) )
                                  
    # def canvasMoveEvent(self,event):
    #     print "canvasMoveEvent"

    def canvasReleaseEvent(self,event):
        if self.edit_mode =='optimize':
            # optimize always takes a node.
            target_type,target_id = self.event_to_target(event,px_tolerance=np.inf,target_type='cell')
        else:
            target_type,target_id = self.event_to_target(event)
        
        mods = event.modifiers()        
        button = event.button()

        if self.edit_mode == 'edgenode':
            if button == self.rightbutton:
                cmd = 'del'
            elif mods & self.shiftmod:
                cmd = 'move'
            else:
                cmd = 'add'
        elif self.edit_mode == 'cell_toggle':
            cmd = 'cell_toggle'
        elif self.edit_mode == 'optimize':
            # we always want a node for the target  -
            if button ==self.leftbutton and (mods & self.shiftmod):
                cmd = 'repave'
            else:
                cmd = 'relax'
        print "Command: ",cmd
        self.state.append( (cmd,target_type,target_id) )

        # Process whatever is in state
        layer = self.canvas.currentLayer()
        grid = layer.grid

        layer.freeze_repaints()
        
        if len(self.state) == 2:
            print "mouse state:"
            print self.state

            if self.state[0][0] == 'add' and self.state[1][0] == 'add':
                nodes = []

                # this is where the old code checked to see if the line 
                # is safe.

                for j in [0,1]:
                    if self.state[j][1] == 'point':
                        pnt = np.array( self.state[j][2] )
                        n = grid.add_node(x=pnt)
                        nodes.append( n )
                    elif self.state[j][1] == 'node':
                        nodes.append( self.state[j][2] )
                    else:
                        raise Exception,"How is it neither a point nor a node?"
                if nodes[0] == nodes[1]:
                    print "That's the same node!"
                    nodes = None
                else:
                    print "Checking on edge betwen %s and %s"%(nodes[0],nodes[1])
                    j = grid.nodes_to_edge( *nodes )
                    if j is not None:
                        print "That edge already exists"
                        nodes = None

                if nodes is not None:
                    grid.add_edge(nodes=nodes)
                    
            elif self.state[0][0] == 'del' and self.state[1][0] == 'del':
                print "Two dels - great."
                if self.state[0][1] == 'node' and self.state[1][1] == 'node':
                    if self.state[0][2] == self.state[1][2]:
                        print "Okay - will delete that node"
                        grid.delete_node_cascade( self.state[0][2] )
                    else:
                        print "And both are nodes.  Will delete that edge if it exists"
                        j = grid.nodes_to_edge( self.state[0][2],self.state[1][2] )
                        if j:
                            grid.delete_edge_cascade(j)
                        else:
                            print "Didn't find that edge"
                else:
                    print "Two deletes, but not both nodes.  Forget it"
            elif self.state[0][0] == 'move' and self.state[0][1] == 'node':
                if self.state[1][1] == 'node':
                    print "Not ready to move one node onto another node"
                elif self.state[1][1] == 'point':
                    grid.modify_node(self.state[0][2],x=np.array(self.state[1][2]))
            elif self.state[0][0] == 'cell_toggle' and self.state[0][1] == 'point':
                print "Would be toggle cell at point ",self.state[0][2]
                grid.toggle_cell(p=self.state[0][2])
            elif self.state[0][0] == 'relax':
                if self.state[0][1] != 'cell':
                    print "RELAX, but target isn't a cell"
                else:
                    c = self.state[0][2]
                    nbr_size = self.dock_widget.settings['neighborhood size']
                    grid.relax_neighborhood(c,nbr_size)
            elif self.state[0][0] == 'repave':
                if self.state[0][1] != 'cell':
                    print "REPAVE, but target isn't a cell"
                else:
                    c = self.state[0][2]
                    nbr_size = self.dock_widget.settings['neighborhood size']
                    scale_factor = self.dock_widget.settings['scale factor']
                    grid.repave_neighborhood(c,neighborhood_size=nbr_size,scale_factor=scale_factor)
                print "REPAVE"
                
        # 
        self.state = []
        layer.thaw_repaints()

    def isZoomTool(self):
        return False

    def isTransient(self):
        return False

    def isEditTool(self):
        # it /is/ an edit tool, but isEditTool() is only useful when the layer in 
        # question is a proper vector layer.
        return False
