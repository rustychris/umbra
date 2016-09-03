import os
import sys
sys.path.append( os.path.join(os.environ["HOME"],"python") )

from   unstructured_grid import mag
import unstructured_grid
import numpy as np
from delft import dfm_grid

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry, 
                        QgsMarkerSymbolV2, QgsLineSymbolV2, QgsFillSymbolV2 )
from PyQt4.QtCore import QVariant

import logging
log=logging.getLogger('umbra.layer')

# Used to be a QgsPluginLayer, but no more.
# now it manages the layers and data specific to a grid

# Maybe the Undo handling will live here?
from PyQt4.QtGui import QUndoCommand,QUndoStack
class GridCommand(QUndoCommand):
    def __init__(self,g,description="edit grid",redo=None):
        super(GridCommand,self).__init__(description)
        self.grid = g
        self.checkpoint = g.checkpoint()
        self.redo_count=0
        if redo:
            self.redo_thunk=redo
        else:
            self.redo_thunk=None

    def undo(self):
        self.grid.revert(self.checkpoint)
        
    def redo(self):
        # might be used in the future to allow exactly one redo()
        if self.redo_thunk is not None:
            self.redo_thunk()
        else:
            self.redo_count+=1
        

class UmbraSubLayer(object):
    def __init__(self,log,grid,crs,prefix,tag=None):
        self.log=log
        self.grid=grid
        self.tag=tag
        self.extend_grid()
        self.crs=crs
        self.prefix=prefix
        self.qlayer=self.create_qlayer()
        self.populate_qlayer()

    def create_qlayer(self):
        return None
        
    def extend_grid(self):
        """ 
        install callbacks or additional fields as
        needed
        """
        pass
    def unextend_grid(self):
        """ 
        remove those callbacks or additional fields
        """
        pass
    def selection(self):
        return []
        
class UmbraNodeLayer(UmbraSubLayer):
    def extend_grid(self):
        g=self.grid
        if 'feat_id' not in g.nodes.dtype.names:
            g.add_node_field('feat_id',
                             np.zeros(g.Nnodes(),'i4')-1)

        g.subscribe_after('modify_node',self.on_modify_node)
        g.subscribe_after('add_node',self.on_add_node)
        g.subscribe_before('delete_node',self.on_delete_node)
    def unextend_grid(self):
        g=self.grid
        g.delete_node_field('feat_id')
        g.unsubscribe_after('modify_node',self.on_modify_node)
        g.unsubscribe_after('add_node',self.on_add_node)
        g.unsubscribe_before('delete_node',self.on_delete_node)
        
    def create_qlayer(self):
        layer= QgsVectorLayer("Point"+self.crs, self.prefix+"-nodes", "memory")
        # nice clean black dot
        symbol = QgsMarkerSymbolV2.createSimple({'outline_style':'no',
                                                 'name': 'circle', 
                                                 'size_unit':'MM',
                                                 'size':'1',
                                                 'color': 'black'})
        layer.rendererV2().setSymbol(symbol)
        return layer

    def populate_qlayer(self):
        layer=self.qlayer
        # shouldn't be necessary..
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing point memory layer, adds in nodes from g
        feats=[]
        valid=[]
        #for n in range(self.grid.Nnodes()): # why was I using this???
        for n in self.grid.valid_node_iter():
            valid.append(n)
            geom = self.node_geometry(n)
            feat = QgsFeature() # can't set feature_ids
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.nodes['feat_id'][valid] = [f.id() for f in outFeats]

    def on_modify_node(self,g,func_name,n,**k):
        if 'x' not in k:
            return
        fid=self.grid.nodes[n]['feat_id']
        geom=self.node_geometry(n)
        self.qlayer.dataProvider().changeGeometryValues({fid:geom})
        self.qlayer.triggerRepaint()
        
    def on_add_node(self,g,func_name,return_value,**k):
        n=return_value
        geom=self.node_geometry(n)
        feat = QgsFeature() # can't set feature_ids
        feat.setGeometry(geom)
        (res, outFeats) = self.qlayer.dataProvider().addFeatures([feat])

        self.grid.nodes['feat_id'][n] = outFeats[0].id()
        self.qlayer.triggerRepaint()
        
    def on_delete_node(self,g,func_name,n,**k):
        self.qlayer.dataProvider().deleteFeatures([self.grid.nodes['feat_id'][n]])
        self.qlayer.triggerRepaint()

    def node_geometry(self,n):
        return QgsGeometry.fromPoint(QgsPoint(self.grid.nodes['x'][n,0],
                                              self.grid.nodes['x'][n,1]))

class UmbraEdgeLayer(UmbraSubLayer):
    def create_qlayer(self):
        qlayer=QgsVectorLayer("LineString"+self.crs,self.prefix+"-edges","memory")
        
        pr = qlayer.dataProvider()

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
                    self.log.info("Not ready for other datatypes")
        self.e_attrs=e_attrs # violates functional nature ...
        pr.addAttributes(e_attrs)
        qlayer.updateFields() # tell the vector layer to fetch changes from the provider
        
        # clean, thin black style
        symbol = QgsLineSymbolV2.createSimple({'line_style':'solid',
                                                 'line_width':'0.2',
                                                 'line_width_unit':'MM',
                                                 'line_color': 'black'})
        qlayer.rendererV2().setSymbol(symbol)
        return qlayer

    def populate_qlayer(self):
        layer=self.qlayer
        # shouldn't be necessary
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
    
    def extend_grid(self):
        self.install_edge_quality()
        g=self.grid
        if 'feat_id' not in g.edges.dtype.names:
            g.add_edge_field('feat_id',
                             np.zeros(g.Nedges(),'i4')-1)
        g.subscribe_after('add_edge',self.on_add_edge)
        g.subscribe_before('delete_edge',self.on_delete_edge)
        g.subscribe_after('modify_node',self.on_modify_node)

    def unextend_grid(self):
        g=self.grid
        g.delete_edge_field('feat_id','edge_quality')
        g.unsubscribe_after('add_edge',self.on_add_edge)
        g.unsubscribe_before('delete_edge',self.on_delete_edge)
        g.unsubscribe_after('modify_node',self.on_modify_node)

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

    def on_add_edge(self,g,func_name,return_value,**k):
        j=return_value
        feat=QgsFeature()
        feat.setGeometry(self.edge_geometry(j))
        (res,outFeats) = self.qlayer.dataProvider().addFeatures([feat])

        self.grid.edges['feat_id'][j] = outFeats[0].id()
        self.qlayer.triggerRepaint()
        
    def on_delete_edge(self,g,func_name,j,**k):
        self.qlayer.dataProvider().deleteFeatures([self.grid.edges['feat_id'][j]])
        self.qlayer.triggerRepaint()
    def on_modify_node(self,g,func_name,n,**k):
        if 'x' not in k:
            return
        
        edge_geom_changes={}
        # edge centers are not cached at this point, so don't
        # need to update them...
        for j in self.grid.node_to_edges(n):
            fid=self.grid.edges[j]['feat_id']
            geom=self.edge_geometry(j)
            edge_geom_changes[fid]=geom
        self.qlayer.dataProvider().changeGeometryValues(edge_geom_changes)

        # Edges for which a node or cell has changed:
        # this doesn't seem to be working now.
        edge_attr_changes={}
        edge_quality_idx=[i
                          for i,attr in enumerate(self.e_attrs)
                          if attr.name()=='edge_quality'][0]

        # js=list(cell_edges)
        # self.update_edge_quality(js)
        # for j in js:
        #     # and update edge quality field - would be nice
        #     # to come up with a nice abstraction here...
        #     fid=self.grid.edges[j]['feat_id']
        #     edge_attr_changes[fid]={edge_quality_idx:float(self.grid.edges[j]['edge_quality'])}
        # self.qlayer.dataProvider().changeAttributeValues(edge_attr_changes)
        self.qlayer.triggerRepaint()

    def edge_geometry(self,j):
        seg=self.grid.nodes['x'][self.grid.edges['nodes'][j]]
        pnts=[QgsPoint(seg[0,0],seg[0,1]),
              QgsPoint(seg[1,0],seg[1,1])]
        return QgsGeometry.fromPolyline(pnts)


class UmbraCellLayer(UmbraSubLayer):
    def create_qlayer(self):
        layer=QgsVectorLayer("Polygon"+self.crs,self.prefix+"-cells","memory")
        # transparent red, no border
        # but this is the wrong class...
        symbol = QgsFillSymbolV2.createSimple({'outline_style':'no',
                                               'style':'solid',
                                               'color': '249,0,0,78'})
        layer.rendererV2().setSymbol(symbol)
        return layer
        
    def extend_grid(self):
        g=self.grid
        if 'feat_id' not in g.cells.dtype.names:
            g.add_cell_field('feat_id',
                             np.zeros(g.Ncells(),'i4')-1)

        g.subscribe_after('add_cell',self.on_add_cell)
        g.subscribe_before('delete_cell',self.on_delete_cell)
        g.subscribe_after('modify_node',self.on_modify_node)
    def unextend_grid(self):
        g=self.grid
        g.delete_cell_field('feat_id')
        g.unsubscribe_after('add_cell',self.on_add_cell)
        g.unsubscribe_before('delete_cell',self.on_delete_cell)
        g.unsubscribe_after('modify_node',self.on_modify_node)
        

    def on_delete_cell(self,g,func_name,c,**k):
        # this one isn't working, while the others are...
        # actually I think it is working, or was before the refactor
        feat_id=self.grid.cells['feat_id'][c]
        self.log.info('got signal for delete cell %d, feat_id %s'%(c,feat_id))
        self.qlayer.dataProvider().deleteFeatures([feat_id])
        self.qlayer.triggerRepaint()

    def on_modify_node(self,g,func_name,n,**k):
        if 'x' not in k:
            return
        
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
        self.qlayer.dataProvider().changeGeometryValues(cell_changes)
        self.qlayer.triggerRepaint()

    def on_add_cell(self,g,func_name,return_value,**k):
        c=return_value
        self.log.info('got signal for add cell')
        geom=self.cell_geometry(c)
        feat = QgsFeature() # can't set feature_ids
        feat.setGeometry(geom)
        (res, outFeats) = self.qlayer.dataProvider().addFeatures([feat])

        self.grid.cells['feat_id'][c] = outFeats[0].id()
        self.qlayer.triggerRepaint()
         
    def cell_geometry(self,i):
        pnts=[QgsPoint(self.grid.nodes['x'][n,0],
                       self.grid.nodes['x'][n,1])
              for n in self.grid.cell_to_nodes(i)]
        return QgsGeometry.fromPolygon([pnts])

    def populate_qlayer(self):
        layer=self.qlayer
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

    def selection(self):
        # these are feature ids...
        cell_feat_ids=[feat.id()
                       for feat in self.qlayer.selectedFeatures()]
        cell_feat_ids=set(cell_feat_ids)

        selected=[]
        for c in range(self.grid.Ncells()):
            if self.grid.cells['feat_id'][c] in cell_feat_ids:
                selected.append(c)
        return selected
    
class UmbraLayer(object):
    count=0

    layer_count=0
    
    def __init__(self,umbra,grid,name=None):
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
                    s=" ".join(a)
                    with open(os.path.join(os.path.dirname(__file__),'log'),'a') as fp:
                        fp.write(s+"\n")
                        fp.flush()
            log=DumbLog()
            log.debug=log.info
            log.warning=log.info
            log.error=log.info
            self.log=log
            
        self.umbra=umbra

        self.grid=grid

        self.iface=None # gets set in register_layers

        UmbraLayer.layer_count+=1

        if name is None:
            name="grid%d"%UmbraLayer.layer_count
        self.name=name 

        self.layers=[] # SubLayer objects associated with this grid.
        self.umbra.register_grid(self)
        
        self.undo_stack=QUndoStack()

    def match_to_qlayer(self,ql):
        # need to either make the names unique, or find a better test.
        # since we can't really enforce the grouped layout, better to make the names
        # unique.
        for layer in self.layers:
            if layer.qlayer.name() == ql.name():
                return True
        return False
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


    def on_layer_deleted(self,sublayer):
        self.layers.remove(sublayer)
        self.log.info("signal received: node layer deleted")
        sublayer.unextend_grid()

        # these are the steps which lead to this callback.  don't do them
        # again.
        
        # reg=QgsMapLayerRegistry.instance()
        # reg.removeMapLayers([sublayer.qlayer])
        
    # create the memory layers and populate accordingly
    def register_layer(self,sublayer):
        self.layers.append( sublayer )
        def callback(sublayer=sublayer):
            self.on_layer_deleted(sublayer)
        sublayer.qlayer.layerDeleted.connect(callback)
        
        QgsMapLayerRegistry.instance().addMapLayer(sublayer.qlayer)
        li=self.iface.legendInterface()
        li.moveLayer(sublayer.qlayer,self.group_index)

    def layer_by_tag(self,tag):
        for layer in self.layers:
            if layer.tag == tag:
                return layer
        return None

    def create_group(self):
        # Create a group for the layers -
        li=self.iface.legendInterface()
        grp_name=self.grid_name()
        self.group_index=li.addGroup(grp_name)
        
    def register_layers(self):
        crs="?crs=epsg:26910" # was 4326
        self.iface=self.umbra.iface
        self.create_group()

        self.register_layer( UmbraCellLayer(self.log,
                                            self.grid,
                                            crs=crs,
                                            prefix=self.name,
                                            tag='cells') )

        self.register_layer( UmbraEdgeLayer(self.log,
                                            self.grid,
                                            crs=crs,
                                            prefix=self.name,
                                            tag='edges' ) )

        self.register_layer( UmbraNodeLayer(self.log,
                                            self.grid,
                                            crs=crs,
                                            prefix=self.name,
                                            tag='nodes') )
        
        # set extent to the extent of our layer
        # skip while developing
        # canvas.setExtent(layer.extent())

    def remove_all_qlayers(self):
        layers=[]
        for sublayer in self.layers:
            layers.append( sublayer.qlayer.name() )
        reg=QgsMapLayerRegistry.instance()
        self.log.info("Found %d layers to remove"%len(layers))
        
        reg.removeMapLayers(layers)
        
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
    
    # thin wrapper to grid editing calls
    # channel them through here to (a) keep consistent interface
    # and (b) track undo stacks at the umbra layer level.

    def undo(self):
        if self.undo_stack.canUndo():
            self.undo_stack.undo()
        else:
            self.log.warning("request for undo, but stack cannot")
    def redo(self):
        if self.undo_stack.canRedo():
            self.undo_stack.redo()
        else:
            self.log.warning("request for undo, but stack cannot")
            
    def modify_node(self,n,**kw):
        cmd=GridCommand(self.grid,
                        "Modify node",
                        lambda: self.grid.modify_node(n,**kw))
        self.undo_stack.push(cmd)

    def toggle_cell_at_point(self,xy):
        def do_toggle():
            self.log.info("umbra_layer: toggle cell at %s"%xy)
            self.grid.toggle_cell_at_point(xy)
        cmd=GridCommand(self.grid,
                        "Toggle cell",
                        lambda: self.grid.toggle_cell_at_point(xy))
        self.undo_stack.push(cmd)
        
    def delete_node(self,n):
        cmd=GridCommand(self.grid,
                        "Delete node",
                        lambda: self.grid.delete_node_cascade(n))
        self.undo_stack.push(cmd)
    def delete_edge(self,e):
        cmd=GridCommand(self.grid,
                        "Delete edge",
                        lambda: self.grid.delete_edge_cascade(e))
        self.undo_stack.push(cmd)
        
    def add_edge(self,nodes):
        if self.grid.nodes_to_edge(*nodes) is not None:
            self.log.info("Edge already existed, probably")
            return
        
        self.add_edge_last_id=None
        def redo():
            j=self.grid.add_edge(nodes=nodes)
            self.add_edge_last_id=j
            
        cmd=GridCommand(self.grid,"Add edge",redo)
        self.undo_stack.push(cmd)

        assert self.add_edge_last_id is not None
        self.log.info("Adding an edge! j=%d"%self.add_edge_last_id)

        return self.add_edge_last_id
    
    def add_node(self,x):
        # awkward jumping through hoops to both use the undo stack
        # and get the id of a node which was just added
        self.add_node_last_id=None
        def redo():
            n=self.grid.add_node(x=x)
            self.add_node_last_id=n
            
        cmd=GridCommand(self.grid,"Add node",redo)
        self.undo_stack.push(cmd)
        
        assert self.add_node_last_id is not None
        return self.add_node_last_id

    def delete_selected(self):
        cell_layer=self.layer_by_tag('cells')
        if cell_layer is not None:
            selected_cells = cell_layer.selection()
            self.log.info("Found %d selected cells"%len(selected_cells))
            def redo(cells=selected_cells): # pass this way b/c of python bindings weirdness
                for c in cells:
                    self.grid.delete_cell(c)
            cmd=GridCommand(self.grid,"Delete cells",redo)
            self.undo_stack.push(cmd)
