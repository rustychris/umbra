from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
# from qgis.core import 
from qgis.gui import QgsMapTool

import logging
log=logging.getLogger('umbra.editor')

# Copied / boilerplated from cadtools/singlesegmentfindertoolpy
import umbra_layer

import numpy as np
import traceback

class Worker(QObject):
    '''
    worker for various long-running grid operations.
    See https://snorfalorpagus.net/blog/2013/12/07/multithreading-in-qgis-python-plugins/
    from which this was copied.
    '''
    def __init__(self, thunk):
        QObject.__init__(self)
        self.thunk = thunk
        self.killed = False
    def run(self):
        ret = None
        try:
            self.progress.emit(0.1)
            self.thunk()
            self.progress.emit(0.9)

            # if self.killed is False:
            #     self.progress.emit(100)
            #     ret = (self.layer, total_area,)
            ret="some return value"
        except Exception, e:
            # forward the exception upstream
            self.error.emit(e, traceback.format_exc())
        self.finished.emit(ret)
    def kill(self):
        self.killed = True
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(Exception, basestring)
    progress = QtCore.pyqtSignal(float)


class SlowOpDialog(QDialog):
    def __init__(self,parent=None,iface=None,umbra=None):
        super(SlowOpDialog,self).__init__(parent)
        self.umbra=umbra
        self.iface=iface
        # log.info("Calling setupUI")
        # self.setupUi(self)
        self.iface=iface

    def startWorker(self, thunk):
        # create a new worker instance
        worker = Worker(thunk)

        # configure the QgsMessageBar
        messageBar = self.iface.messageBar().createMessage('Doing something time consuming...', )
        progressBar = QtGui.QProgressBar()
        progressBar.setAlignment(QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        cancelButton = QtGui.QPushButton()
        cancelButton.setText('Cancel')
        cancelButton.clicked.connect(worker.kill)
        messageBar.layout().addWidget(progressBar)
        messageBar.layout().addWidget(cancelButton)
        self.iface.messageBar().pushWidget(messageBar, self.iface.messageBar().INFO)
        self.messageBar = messageBar

        # start the worker in a new thread
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.finished.connect(self.workerFinished)
        worker.error.connect(self.workerError)
        worker.progress.connect(progressBar.setValue)
        thread.started.connect(worker.run)
        thread.start()
        self.thread = thread
        self.worker = worker
    def workerError(self,*a):
        log.error("Worker is unhappy")
        self.workerFinished(ret=None)
    def workerFinished(self, ret):
        # clean up the worker and thread
        log.info("worker finished!")
        self.worker.deleteLater()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        # remove widget from message bar
        self.iface.messageBar().popWidget(self.messageBar)
        # close the dialog:
        self.close()

        if ret is not None:
            # report the result
            layer, total_area = ret
            self.iface.messageBar().pushMessage('The total area of {name} is {area}.'.format(name=layer.name(), area=total_area))
        else:
            # notify the user that something went wrong
            self.iface.messageBar().pushMessage( ('Something went wrong! '
                                                  'See the message log for more information.' ),
                                                  level=QgsMessageBar.CRITICAL, duration=3)

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
        # handle_tool_select has been dummied for a while.  probably this is ready
        # to be removed, too.
        # QObject.connect(self.action_coverage_editor, SIGNAL("triggered()"), self.handle_tool_select)

        #QObject.connect(self.iface, SIGNAL("currentLayerChanged(QgsMapLayer*)"), self.handle_layer_changed)
        # can we use the newer syntax?
        self.iface.currentLayerChanged.connect(self.handle_layer_changed)

        # track state of ongoing operations
        self.op_action=None
        self.op_node=None

    def unload(self):
        """
        Reverse the actions of __init__
        """
        self.iface.currentLayerChanged.disconnect(self.handle_layer_changed)

    def activate(self):
        print "Call to activate for the editor tool"
        self.canvas.setCursor(self.cursor)

    def deactivate(self):
        self.log.info("Tool deactivated")

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
        return self.umbra.current_grid()
    def gridlayer(self):
        return self.umbra.active_gridlayer()

    def event_to_item(self,event,types=['node','edge']):
        self.log.info("Start of event_to_item self=%s"%id(self))
        map_xy,pix_xy=self.event_to_map_xy(event)

        map_to_pixel=self.canvas.getCoordinateTransform()

        res=dict(node=None,edge=None,cell=None)
        g=self.grid()
        if g is None:
            log.info("event_to_item: no grid available")
            return res

        if 'node' in types:
            n=g.select_nodes_nearest(map_xy)
            res['node']=None
            if n is not None:
                node_xy=g.nodes['x'][n]
                node_pix_point = map_to_pixel.transform(node_xy[0],node_xy[1])
                dist2= (pix_xy[0]-node_pix_point.x())**2 + (pix_xy[1]-node_pix_point.y())**2 
                self.log.info("Distance^2 is %s"%dist2)
                if dist2<=self.node_click_pixels**2:
                    # back to pixel space to calculate distance
                    res['node']=n
        if 'edge' in types:
            j=g.select_edges_nearest(map_xy)
            res['edge']=None
            if j is not None:
                edge_xy=g.edges_center()[j]
                edge_pix_point = map_to_pixel.transform(edge_xy[0],edge_xy[1])
                dist2= (pix_xy[0]-edge_pix_point.x())**2 + (pix_xy[1]-edge_pix_point.y())**2 
                self.log.info("Distance^2 is %s"%dist2)
                if dist2<=self.edge_click_pixels**2:
                    # back to pixel space to calculate distance
                    res['edge']=j
        self.log.info( "End of event_to_item self=%s"%id(self) )
        return res

    # Bindings:
    # left mouse drag => move node
    # left mouse with shift => add edge and/or node
    # left mouse with control => toggle cell
    # Right mouse => delete edge/node
    # space: supposedly toggle cell
    # 'z': undo
    # 'Z': redo
    # 'm': try to merge nodes of selected edge
    # 's': split an edge and any adjacent cells
    # 'j': join cells sharing an edge
    # Delete or backspace: delete selected elements
    # What would be good for merging nodes?
    #  could draw an edge, then have a key for merging nodes
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

    def event_to_map_xy(self,event):
        """
        Return map_xy,pix_xy from event
        """
        if isinstance(event,QKeyEvent): 
            mouse_pnt=self.canvas.mouseLastXY()
        else:
            # from a mouse event
            mouse_pnt=event.pos()

        pix_x=mouse_pnt.x()
        pix_y=mouse_pnt.y()
        map_to_pixel=self.canvas.getCoordinateTransform()
        map_point = map_to_pixel.toMapCoordinates(pix_x,pix_y)
        map_xy=[map_point.x(),map_point.y()]
        return map_xy,[pix_x,pix_y]

    def merge_nodes_of_edge(self,event):
        gl=self.gridlayer()
        if gl is None:
            return

        self.log.info("Merging nodes of edge?")

        items=self.event_to_item(event,types=['edge'])
        if items['edge'] is not None:
            gl.merge_nodes_of_edge(items['edge'])
            self.clear_op() # safety first
        else:
            self.log.info("no feature hits")

    def split_edge(self,event):
        gl=self.gridlayer()
        if gl is None:
            return
        items=self.event_to_item(event,types=['edge'])
        if items['edge'] is not None:
            gl.split_edge(e=items['edge'])
            self.clear_op() # safety first
        else:
            self.log.info("no feature hits")

    def merge_cells(self,event):
        gl=self.gridlayer()
        if gl is None:
            return
        items=self.event_to_item(event,types=['edge'])
        if items['edge'] is not None:
            gl.merge_cells(e=items['edge'])
            self.clear_op() # safety first
        else:
            self.log.info("no feature hits")

    def optimize_local(self,event):
        gl=self.gridlayer()
        if gl is None:
            return

        self.log.info("Trying a local optimize")
        map_xy,_=self.event_to_map_xy(event)

        n_iters=self.umbra.dockwidget.orthogNIters.value()

        if 1:
            # serial approach
            gl.orthogonalize_local(map_xy,iterations=n_iters)
        else:
            # threaded!?  this crashes
            thunk=( lambda gl=gl,map_xy=map_xy,n_iters=n_iters:
                    gl.orthogonalize_local(map_xy,iterations=n_iters) )
            self.log.info("Creating SlowOpDialog")
            slow_dlg=SlowOpDialog(parent=self.iface.mainWindow(),
                                  iface=self.iface,
                                  umbra=self.umbra)
            # I don't think we ever actually exec the dialog.  it's
            # not really needed..
            # slow_dlg.exec_()
            self.log.info("exec_ on SlowOpDialog")
            
            slow_dlg.startWorker(thunk)

            self.log.info("Called startWorker")
            
    def toggle_cell(self,event):
        gl=self.gridlayer()
        if gl is None:
            return
        
        self.log.info("Got a toggle cell event")
        map_xy,_=self.event_to_map_xy(event)
        
        gl.toggle_cell_at_point(map_xy)

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
        gl=self.gridlayer()
        if gl is None:
            self.log.warning("Got to delete_edge_or_node, but gridlayer is None")
            return
        
        items=self.event_to_item(event,types=['node','edge'])
        if items['node'] is not None:
            gl.delete_node(items['node'])
            self.clear_op() # just to be safe
        elif items['edge'] is not None:
            gl.delete_edge(items['edge'])
            self.clear_op() # safety first
        else:
            self.log.info("Delete press event, but no feature hits")

    def add_edge_or_node(self,event):
        gl=self.gridlayer()
        if gl is None:
            return None
        
        if self.op_action=='add_edge':
            self.log.info("Continuing add_edge_or_node")
            last_node=self.op_node
        else:
            self.log.info("Starting add_edge_or_node")
            last_node=None
            
        self.op_action='add_edge'
        self.op_node=self.select_or_add_node(event)
        if last_node is not None:
            gl.add_edge(nodes=[last_node,self.op_node])

    def select_or_add_node(self,event):
        gl=self.gridlayer()
        if gl is None:
            return None
        
        items=self.event_to_item(event,types=['node'])
        if items['node'] is None:
            map_point=event.mapPoint()
            map_xy=[map_point.x(),map_point.y()]

            self.log.info("Creating new node")
            return gl.add_node(x=map_xy) 
        else:
            return items['node']

    # def canvasMoveEvent(self, event):
    #     x = event.pos().x()
    #     y = event.pos().y()
    #     point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)

    def canvasReleaseEvent(self, event):
        super(UmbraEditorTool,self).canvasReleaseEvent(event)
        self.log.info("Release event top, type=%s"%event.type())

        gl=self.gridlayer()
        if gl is None:
            self.log.info("canvasReleaseEvent in editor tool, but no grid layer!")
            self.clear_op()
            return

        self.log.info("Release with op_node=%s self=%s"%(self.op_node,id(self)))
        if self.op_action=='move_node' and self.op_node is not None:
            x = event.pos().x()
            y = event.pos().y()
            point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)
            xy=[point.x(),point.y()]
            self.log.info( "Modifying location of node %d self=%s"%(self.op_node,id(self)) )
            gl.modify_node(self.op_node,x=xy)
            self.clear_op()
        elif self.op_action=='add_edge':
            # all the action happens on the press, and releasing the shift key
            # triggers the end of the add_edge mode
            pass 
        else:
            # think safety, act safely.
            self.clear_op()
        self.log.info("Release event end")

    def keyPressEvent(self,event):
        super(UmbraEditorTool,self).keyPressEvent(event)
        key=event.key()
        txt=event.text()

        self.log.info("keyPress %r %s"%(key,txt) )
        # weird, but seems that shift comes through, but not 
        # space??  doesn't even show up.
        # Pressing 'delete' on mac seems to give 16777219, "^H"
        # that sounds like backspace rather than delete.

        # Might want to add 'm' for merge selected nodes?
        if txt == ' ':
            self.toggle_cell(event)
        elif txt == 'z':
            self.undo()
        elif txt == 'Z':
            self.redo()
        elif txt=='r':
            self.optimize_local(event)
        elif txt == 'm':
            self.merge_nodes_of_edge(event)
        elif txt == 's':
            self.split_edge(event)
        elif txt == 'j':
            self.merge_cells(event)
        elif key == Qt.Key_Delete or key == Qt.Key_Backspace:
            # A little shaky here, but I think the idea is that
            # we accept it if we handle it, which is good b/c
            # otherwise qgis will complain that the layer isn't editable.
            if self.delete(event):
                self.log.info("Trying to accept this Delete event")
                event.accept()
            else:
                self.log.info("Ignoring this Delete event")
                event.ignore()

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

    def delete(self,event):
        gl=self.gridlayer()
        if gl is not None:
            self.log.info('calling delete_selected')
            gl.delete_selected()
            return True
        else:
            self.log.info('delete(), but no gridlayer so ignore')
        return False

    def undo(self):
        self.log.info('got request for undo')
        gl=self.gridlayer()
        if gl is not None:
            self.log.info('sending undo request to umbra layer %r'%gl)
            gl.undo()
        else:
            self.log.info('editor got undo request, but has no grid layer')

    def redo(self):
        self.log.info('got request for redo')
        gl=self.gridlayer()
        if gl is not None:
            self.log.info('sending redo request to umbra layer %r'%gl)
            gl.redo()
        else:
            self.log.info('editor got redo request, but has no grid layer')

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
