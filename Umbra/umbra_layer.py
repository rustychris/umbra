import os
import sys
sys.path.append( os.path.join(os.environ["HOME"],"python") )

import unstructured_grid
import trigrid
import utils
import numpy as np

from PyQt4.QtCore import QLineF, SIGNAL
import PyQt4.QtGui as QtGui

from qgis.core import *

class UmbraLayer(QgsPluginLayer):
    LAYER_TYPE="umbra"

    def __init__(self,iface=None,format=None,path=None):
        self.repaint_freezes = 0
        self.repaint_queued = False

        self.grid=None
        self.iface=iface
        if path is not None:
            name=path
        else:
            name="Umbra layer"

        super(UmbraLayer,self).__init__(self.LAYER_TYPE,name)
        self.load_grid(format=format,path=path)
        self.prepare_grid()
        self.setValid(True)

    def load_grid(self,format=None,path=None):
        if path is None:
            # for development, load sample data:
            suntans_path=os.path.join( os.path.dirname(__file__),
                                       "sample_data/sfbay" )
            # tg=trigrid.TriGrid( )
            #sms_fname='/Users/rusty/research/meshing/delta_1.grd')        
            self.grid=unstructured_grid.SuntansGrid(suntans_path)
        else:
            if format=='SUNTANS':
                # tg=trigrid.TriGrid(suntans_path=path)
                self.grid=unstructured_grid.SuntansGrid(path)
            elif format=='pickle':
                self.grid=unstructured_grid.UnstructuredGrid.from_pickle(path)
            else:
                raise Exception("Need to add other grid types!")
            
    def prepare_grid(self):
        """ after loading grid, set up any callbacks, cached data, bounds, etc.
        """
        self.prepare_lines()
    
        # hardwired EPSG:26910
        self.coordRS = QgsCoordinateReferenceSystem()
        self.coordRS.createFromUserInput("EPSG:26910")
        self.setCrs(self.coordRS)
    
        # try setting extent field:
        self.setExtent(self.extent())
    
        # wire up some callbacks to the grid model
        # self.grid.listen('delete_edge',self.on_delete_edge)
        self.grid.subscribe_after('add_edge',self.on_add_edge)
        self.grid.subscribe_after('modify_node',self.on_modify_node)
        self.grid.subscribe_after('delete_edge',self.on_delete_edge)
        self.grid.subscribe_after('delete_node',self.on_delete_node)
  
    # 0: no queued repaints, and repaints will be handled synchronously
    def my_repaint(self):
        print "my_repaint, freezes=%s on "%self.repaint_freezes,self
        self.repaint_queued = True
        if self.repaint_freezes <= 0:
            self.emit(SIGNAL("repaintRequested()"))
        
    def freeze_repaints(self):
        self.repaint_freezes+=1
  
    def thaw_repaints(self):
        self.repaint_freezes-=1
        if self.repaint_freezes<=0 and self.repaint_queued:
            self.emit(SIGNAL("repaintRequested()"))
        
    def distance_to_node(self,pnt,i):
        """ compute distance from the given point to the given node, returned in
        physical distance units [meters]"""
        if not self.grid:
            return 1e6
        return utils.dist( self.grid.nodes['x'][i] - np.asarray(pnt) )
    
    def distance_to_cell(self,pnt,c):
        if not self.grid:
            return 1e6
        return utils.dist( self.grid.cells_center()[c] - np.asarray(pnt) )
    
    def find_closest_node(self,xy):
        # xy: [x,y]
        print "Finding closest node to ",xy
        return self.grid.select_nodes_nearest(xy)
    
    def find_closest_cell(self,xy):
        # xy: [x,y]
        return self.grid.select_cells_nearest(xy)

    def visible_edges(self):
        """ index array of edges which should be drawn (i.e. skipping deleted edges)
        """
        return np.nonzero( ~self.grid.edges['deleted'] )[0]

    def visible_nodes(self):
        return np.nonzero( ~self.grid.nodes['deleted'] )[0]
        
    def prepare_lines(self):
        # maintain an array of QLineF objects, in the original CRS
        self.null_line = QLineF(0,0,0,0)
        
        self.qlines = [self.null_line]*self.grid.Nedges()
    
        for j in self.visible_edges():
            segment = self.grid.nodes['x'][self.grid.edges['nodes'][j]]
            self.qlines[j] = QLineF(segment[0,0],segment[0,1],
                                    segment[1,0],segment[1,1])
    
    def extent(self):
        xmin,xmax,ymin,ymax = self.grid.bounds()
        print "extent() called on UmbraLayer"
        return QgsRectangle(xmin,ymin,xmax,ymax)
      
    def draw(self, rendererContext):
        print "Call to draw on ",self
        
        try:
            self.draw_edges(rendererContext)
        except Exception as exc:
            print exc
            import sys
            print sys.exc_info()

        self.repaint_queued = False
        self.repaint_freezes = 0
        print "End of Call to draw"
        return False # ?True

    def renumber(self):
        self.freeze_repaints()
        self.grid.renumber()
        self.prepare_lines()
        self.thaw_repaints()
    
    def on_delete_edge(self,grid,func_name,j,**kwargs):
        self.qlines[j] = self.null_line
        self.my_repaint()
    def on_delete_node(self,grid,func_name,n,**kwargs):
        self.my_repaint()
      
    def on_add_edge(self,grid,func_name,**kws):
        print "Got word of a new edge"
        j=kws['return_value']
        # For now, we just recreate all of the edges
        if len(self.qlines) != j:
            print "ERROR: Length of qlines is not right..."
            return
        self.qlines.append(None)
        self.update_edge(j)

    def on_modify_node(self,grid,func_name,*args,**kwargs):
        n=args[0]
        if 'x' not in kwargs:
            return # ignore attribute changes for now.
        map(self.update_edge,grid.node_to_edges(n))
        self.my_repaint()
    def update_edge(self,j):
        segment = self.grid.nodes['x'][self.grid.edges['nodes'][j],:]
        self.qlines[j] = QLineF(segment[0,0],segment[0,1],
                                segment[1,0],segment[1,1])
        self.my_repaint()

    # def render_edgemarks(self,painter,mx):
    #     """ draw edges with nonzero markers in a different color:
    #     mx should be the scaling of the canvas
    #     """
    #     marker1 = nonzero(self.grid.edges[:,2] == 1)[0]
    #     marker2 = nonzero(self.grid.edges[:,2] == 2)[0]
    #     marker3 = nonzero(self.grid.edges[:,2] == 3)[0]
    #     unmeshed = nonzero( any(self.grid.edges[:,3:5] == -2,axis=1 ))[0]
    # 
    #     marker_bad = self.bad_edge_list
    # 
    #     for indices,color in zip( [marker1,marker2,marker3,unmeshed,marker_bad],
    #                               ['red','blue','orange','yellow','magenta'] ):
    #         painter.setPen( QtGui.QColor(color) )
    #         pen = painter.pen()
    #         pen.setWidth(3/abs(mx))
    #         painter.setPen(pen)
    #         
    #         for m in indices:
    #             painter.drawLine( self.qlines[m] )

    def render_nodes(self,painter,extent):
        # not ready for index here
        # visible_nodes = self.grid.index.qsi.intersects(extent)

        # print "Call to render_nodes"

        xform = painter.combinedTransform()
        dx = xform.m11()
        dy = xform.m22()
        color = QtGui.QColor("green")
        dx = 2.0/dx
        dy = 2.0/dy
        # print "dx: %s  dy: %s "%(dx,dy)
        for n in self.visible_nodes():
            painter.fillRect( self.grid.nodes['x'][n,0]-dx,
                              self.grid.nodes['x'][n,1]-dy, 
                              2*dx,2*dy,color)
        
    # max_edgeneighbors_to_render = 100
    # def render_edgeneighbors(self,painter,extent):
    #     # Flip coordinates so that text is written right-side-up
    #     painter.scale(1,-1)
    #     painter.setPen( QtGui.QColor("black") )
    # 
    #     try:
    #         # a bit tricky - have to do an index lookup -
    #         # This will only work if we're using the wrapper around a QgsSpatialIndex..
    #         visible_nodes = self.grid.index.qsi.intersects(extent)
    # 
    #         # cheap pre-limiting on the number of edges to show:
    #         if len(visible_nodes) > 2*self.max_edgeneighbors_to_render:
    #             visible_nodes = visible_nodes[:self.max_edgeneighbors_to_render]
    # 
    #         edges = []
    #         for vn in visible_nodes:
    #             edges.append( self.grid.pnt2edges(vn) )
    #         edges = concatenate(edges)
    #         edges = unique1d(edges)
    # 
    #         if len(edges) > self.max_edgeneighbors_to_render:
    #             edges = edges[:self.max_edgeneighbors_to_render]
    # 
    #         ecenters = self.grid.edge_centers()
    #         for e in edges:
    #             edge = self.grid.edges[e]
    #             A = self.grid.points[edge[0]]
    #             B = self.grid.points[edge[1]]
    #             normAB = 0.02*(B-A)[::-1]
    #             normAB[0] *= -1
    #             ec = ecenters[e]
    #             p1 = ec + normAB
    #             p2 = ec - normAB
    #             painter.drawText( p1[0],-p1[1], "%d"%edge[3] )
    #             painter.drawText( p2[0],-p2[1], "%d"%edge[4] )
    #     finally:
    #         painter.scale(1,-1)
            
    def draw_edges(self,rendererContext,boundary_only=False):
        # print "Top of draw edges"
        painter = rendererContext.painter()
        painter.save()

        #-# Prep the painter object:
        # figure out the visible area, and a linear transform to get pixels from
        # grid units
        map2pixel = rendererContext.mapToPixel()
        extent = rendererContext.extent()
        geo_topleft = np.array( [extent.xMinimum(), extent.yMaximum()])
        geo_bottomright = np.array( [extent.xMaximum(), extent.yMinimum()])
        
        pix_topleft = map2pixel.transform(extent.xMinimum(), extent.yMaximum())
        pix_topleft = np.array( [pix_topleft.x(), pix_topleft.y()])
        pix_bottomright = map2pixel.transform(extent.xMaximum(), extent.yMinimum())
        pix_bottomright = np.array( [pix_bottomright.x(), pix_bottomright.y()])

        # print "Middle of draw edges"
        x0 = geo_topleft[0]
        mx = (pix_bottomright[0] - pix_topleft[0]) / (geo_bottomright[0] - geo_topleft[0])
        bx = pix_topleft[0]
    
        y0 = geo_bottomright[1]
        my = (pix_topleft[1] - pix_bottomright[1]) / ( geo_topleft[1] - geo_bottomright[1])
        by = pix_bottomright[1]
    
        painter.translate(bx-mx*x0,by-my*y0)
        painter.scale(mx,my)

        # print "About to call render_edges"
        #-# draw features...
        self.render_edges(painter)
        # if self.paint_edgemarks:
        #     # print "Scaling is ",mx,my
        #     self.render_edgemarks(painter,mx)
        # if self.paint_edgeneighbors:
        #     self.render_edgeneighbors(painter,extent=extent)
        
        #print "About to call render_nodes"
        #if self.paint_nodes:
        self.render_nodes(painter,extent=extent)
        
        painter.restore()
    
    def render_edges(self,painter):
        #print "Call to render_edges"
        painter.drawLines( self.qlines )
  
    def readXml(self, node):
        print "call to readXML"
    
    def writeXml(self, node, doc):
        print "call to writeXml"
        
    # Editing - this doesn't seem to work...
    def isEditable(self):
        return True


class UmbraPluginLayerType(QgsPluginLayerType):
    def __init__(self, iface):
        QgsPluginLayerType.__init__(self, UmbraLayer.LAYER_TYPE)
        self.iface = iface
  
    def createLayer(self):
        layer = UmbraLayer(iface=self.iface)
        return layer
  
    def showLayerProperties(self,layer):
        print "Request for layer properties"
  


def is_umbra_layer(l):
    return isinstance(l,UmbraLayer)
