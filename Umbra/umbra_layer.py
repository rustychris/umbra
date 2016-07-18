import os
import sys
sys.path.append( os.path.join(os.environ["HOME"],"python") )

from   unstructured_grid import mag
import unstructured_grid
import numpy as np
from delft import dfm_grid

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry )
from PyQt4.QtCore import QVariant
import logging
log=logging.getLogger('umbra.layer')

# Used to be a QgsPluginLayer, but no more.
# now it manages the layers and data specific to a grid

class UmbraLayer(object):
    count=0
    
    def __init__(self,umbra,grid):
        """
        Does not add the layers to the GUI - call register_layers
        for that.
        """
        # having some trouble getting reliable output from the log...
        if 0:
            self.log=log
        else:
            class DumbLog(object):
                def info(self,*a):
                    print " ".join(a)
            log=DumbLog()
            log.debug=log.info
            log.warning=log.info
            log.error=log.info
            self.log=log
            
        self.umbra=umbra

        # modifications internal to the grid
        self.grid=self.extend_grid(grid)

        # modifications which involve the UmbraLayer instance
        self.install_edge_quality()
        self.iface=None # gets set in register_layers
        
        self.layers=[] # the actual qgs layers associated with this grid.
        self.umbra.register_grid(self)

    def grid_name(self):
        """ used to label the group
        """
        UmbraLayer.count+=1
        return "grid%4d"%UmbraLayer.count
        
    @classmethod
    def open_layer(klass,umbra,grid_format,path):
        g=klass.load_grid(path=path,grid_format=grid_format)
        return klass(umbra=umbra,grid=g)

    @classmethod
    def load_grid(klass,grid_format=None,path=None):
        if path is None:
            # for development, load sample data:
            suntans_path=os.path.join( os.path.dirname(__file__),
                                       "sample_data/sfbay" )
            grid=unstructured_grid.SuntansGrid(suntans_path)
        else:
            if grid_format=='SUNTANS':
                grid=unstructured_grid.SuntansGrid(path)
            elif grid_format=='pickle':
                grid=unstructured_grid.UnstructuredGrid.from_pickle(path)
            elif grid_format=='DFM':
                grid=dfm_grid.DFMGrid(fn=path)
            else:
                raise Exception("Need to add other grid types, like %s!"%grid_format)
        return grid

    def install_edge_quality(self):
        if 'edge_quality' not in self.grid.edges.dtype.names:
            edge_q=np.zeros(self.grid.Nedges(),'f8')
            self.grid.add_edge_field('edge_quality',edge_q)
            self.update_edge_quality()

    def update_edge_quality(self,edges=None):
        # First part is calculating the values
        if edges is None:
            edges=slice(None)
        g=self.grid
        vc=g.cells_center()
        ec=g.edges_center()
        g.edge_to_cells()

        c2c=mag( vc[g.edges['cells'][edges,0]] - vc[g.edges['cells'][edges,1]] )
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
        self.nl.dataProvider().deleteFeatures([self.grid.nodes['feat_id'][n]])
        self.nl.triggerRepaint()

    def on_delete_edge(self,g,func_name,j,**k):
        self.el.dataProvider().deleteFeatures([self.grid.edges['feat_id'][j]])
        self.el.triggerRepaint()

    def on_delete_cell(self,g,func_name,c,**k):
        # this one isn't working, while the others are...
        feat_id=self.grid.cells['feat_id'][c]
        self.log.info('got signal for delete cell %d, feat_id %s'%(c,feat_id))
        self.cl.dataProvider().deleteFeatures([feat_id])
        self.cl.triggerRepaint()

    # Callbacks installed on the grid
    # instrument the grid to propagate changes back to the UI
    def on_modify_node(self,g,func_name,n,**k):
        if 'x' in k:
            fid=self.grid.nodes[n]['feat_id']
            geom=self.node_geometry(n)
            self.nl.dataProvider().changeGeometryValues({fid:geom})
            self.nl.triggerRepaint()

            # update cells first, so that edge_quality has fresh
            # cell center and area information
            cell_changes={}
            cells=self.grid.node_to_cells(n)
            self.grid.cells_center(refresh=cells)
            self.grid.cells['_area'][cells]=np.nan # trigger recalc.
            cell_edges=set()
            for i in cells:
                # this was all sorts of messed up - don't understand how
                # it was working at all before...
                fid=self.grid.cells[i]['feat_id']
                geom=self.cell_geometry(i)
                cell_changes[fid]=geom
                cell_edges.update(self.grid.cell_to_edges(i))
            self.cl.dataProvider().changeGeometryValues(cell_changes)
            self.cl.triggerRepaint()
                
            edge_geom_changes={}
            # edge centers are not cached at this point, so don't
            # need to update them...
            for j in self.grid.node_to_edges(n):
                fid=self.grid.edges[j]['feat_id']
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
                fid=self.grid.edges[j]['feat_id']
                edge_attr_changes[fid]={edge_quality_idx:float(self.grid.edges[j]['edge_quality'])}
            self.el.dataProvider().changeAttributeValues(edge_attr_changes)

            self.el.triggerRepaint()

    def on_add_node(self,g,func_name,return_value,**k):
        n=return_value
        geom=self.node_geometry(n)
        feat = QgsFeature() # can't set feature_ids
        feat.setGeometry(geom)
        (res, outFeats) = self.nl.dataProvider().addFeatures([feat])

        self.grid.nodes['feat_id'][n] = outFeats[0].id()
        self.nl.triggerRepaint()

    def on_add_edge(self,g,func_name,return_value,**k):
        j=return_value
        feat=QgsFeature()
        feat.setGeometry(self.edge_geometry(j))
        (res,outFeats) = self.el.dataProvider().addFeatures([feat])

        self.grid.edges['feat_id'][j] = outFeats[0].id()
        self.el.triggerRepaint()

    def on_add_cell(self,g,func_name,return_value,**k):
        c=return_value
        self.log.info('got signal for add cell')
        geom=self.cell_geometry(c)
        feat = QgsFeature() # can't set feature_ids
        feat.setGeometry(geom)
        (res, outFeats) = self.cl.dataProvider().addFeatures([feat])

        self.grid.cells['feat_id'][c] = outFeats[0].id()
        self.cl.triggerRepaint()
         
    def node_geometry(self,n):
        return QgsGeometry.fromPoint(QgsPoint(self.grid.nodes['x'][n,0],
                                              self.grid.nodes['x'][n,1]))
        
    def populate_nodes(self):
        layer=self.nl
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing point memory layer, adds in nodes from g
        feats=[]
        valid=[]
        for n in range(self.grid.Nnodes()):
            valid.append(n)
            geom = self.node_geometry(n)
            feat = QgsFeature() # can't set feature_ids
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.nodes['feat_id'][valid] = [f.id() for f in outFeats]

        return res

    # callbacks from Qgis layers
    def on_node_geometry_changed(self,feat_id,geom):
        xy=geom.asPoint()
        # this should be sped up with a hash table
        n=np.nonzero( self.grid.nodes['feat_id']==feat_id )[0][0]
        self.grid.modify_node(n,x=xy)

    def edge_geometry(self,j):
        seg=self.grid.nodes['x'][self.grid.edges['nodes'][j]]
        pnts=[QgsPoint(seg[0,0],seg[0,1]),
              QgsPoint(seg[1,0],seg[1,1])]
        return QgsGeometry.fromPolyline(pnts)

    def populate_edges(self):
        layer=self.el
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing line memory layer, adds in nodes from g
        feats=[]
        valid=[]
        for j in self.grid.valid_edge_iter():
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
                    feat.setAttribute(idx,int(self.grid.edges['cells'][j,0]))
                elif name=='c1':
                    feat.setAttribute(idx,int(self.grid.edges['cells'][j,1]))
                elif typecode==2: # integer
                    feat.setAttribute(idx,int(self.grid.edges[name][j]))
                elif typecode==6: # double
                    feat.setAttribute(idx,float(self.grid.edges[name][j]))
                else:
                    continue
                # QGIS doesn't know about numpy types
            
            # feat.setAttribute(3,int(self.grid.edges['mark'][j]))
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.edges['feat_id'][valid]=[f.id() for f in outFeats]

        return res

    def cell_geometry(self,i):
        pnts=[QgsPoint(self.grid.nodes['x'][n,0],
                       self.grid.nodes['x'][n,1])
              for n in self.grid.cell_to_nodes(i)]
        return QgsGeometry.fromPolygon([pnts])
        
    def populate_cells(self):
        layer=self.cl
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing line memory layer, adds in nodes from g
        feats=[]
        valid=[]
        for i in self.grid.valid_cell_iter():
            geom=self.cell_geometry(i)
            feat = QgsFeature()
            feat.setGeometry(geom)
            feats.append(feat)
            valid.append(i)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.cells['feat_id'][valid]=[f.id() for f in outFeats]
        return res

    def on_node_layer_deleted(self):
        self.nl=None
        self.log.info("signal received: node layer deleted")
        # to avoid confusion, let the node layer removal trigger
        # everything, at least for now.  
        self.graceful_remove()
    def on_edge_layer_deleted(self):
        self.el=None
        self.log.info("signal received: edge layer deleted")
    def on_cell_layer_deleted(self):
        self.cl=None
        self.log.info("signal received: cell layer deleted")

    def graceful_remove(self):
        """ when the node layer is deleted, cascade to the others
        """
        reg=QgsMapLayerRegistry.instance()
        for l in [self.nl,self.el,self.cl]:
            try:
                if l is not None:
                    reg.removeMapLayers([l])
            except Exception as exc:
                self.log.error(exc)
                
    # create the memory layers and populate accordingly
    def register_layers(self):
        iface=self.umbra.iface
        canvas=iface.mapCanvas()
        self.iface=iface

        crs="?crs=epsg:26910" # was 4326
        # create layer
        self.nl = QgsVectorLayer("Point"+crs, "nodes", "memory")
        self.el = QgsVectorLayer("LineString"+crs,"edges","memory")
        self.cl = QgsVectorLayer("Polygon"+crs,"cells","memory")

        self.layers += [self.nl,self.el,self.cl]
        
        self.nl.layerDeleted.connect(self.on_node_layer_deleted)
        self.el.layerDeleted.connect(self.on_edge_layer_deleted)
        self.cl.layerDeleted.connect(self.on_cell_layer_deleted)

        pr = self.el.dataProvider()

        # add fields - eventually would be tied in with g.edge_dtype
        e_attrs=[QgsField("edge_id",QVariant.Int)]

        for fidx,fdesc in enumerate(self.grid.edge_dtype.descr):
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
                    self.log.info("Not read other datatypes")
        self.e_attrs=e_attrs            
        pr.addAttributes(e_attrs)
        self.el.updateFields() # tell the vector layer to fetch changes from the provider

        self.populate_nodes()
        self.populate_edges()
        self.populate_cells()

        # Create a group for the layers -
        li=iface.legendInterface()

        grp_name=self.grid_name()
        self.group_index=group_index=li.addGroup(grp_name)

        for layer in [self.cl,self.el,self.nl]:
            # add layer to the registry
            QgsMapLayerRegistry.instance().addMapLayer(layer)
            li.moveLayer(layer,group_index)

        # set extent to the extent of our layer
        # skip while developing
        # canvas.setExtent(layer.extent())

        # this is handled in Umbra, not here
        #self.tool=UgEditTool(canvas,self)
        #canvas.setMapTool(self.tool)

    def log(self,s):
        with open(os.path.join(os.path.dirname(__file__),'log'),'a') as fp:
            fp.write(s+"\n")
            fp.flush()

    def distance_to_node(self,pnt,i):
        """ compute distance from the given point to the given node, returned in
        physical distance units [meters]"""
        if not self.grid:
            return 1e6
        return mag( self.grid.nodes['x'][i] - np.asarray(pnt) )
    
    def distance_to_cell(self,pnt,c):
        if not self.grid:
            return 1e6
        return mag( self.grid.cells_center()[c] - np.asarray(pnt) )
    
    def find_closest_node(self,xy):
        # xy: [x,y]
        print "Finding closest node to ",xy
        return self.grid.select_nodes_nearest(xy)
    
    def find_closest_cell(self,xy):
        # xy: [x,y]
        return self.grid.select_cells_nearest(xy)

    def extent(self):
        xmin,xmax,ymin,ymax = self.grid.bounds()
        print "extent() called on UmbraLayer"
        return QgsRectangle(xmin,ymin,xmax,ymax)
      
    def renumber(self):
        self.grid.renumber()
    


# def is_umbra_layer(l):
#     return isinstance(l,UmbraLayer)
