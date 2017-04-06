import os
import sys
import numpy as np
import time

try:
    from six import iteritems
except ImportError:
    def iteritems(d):
        return d.iteritems()

sys.path.append( os.path.join(os.environ["HOME"],"python") )

from qgis.core import ( QgsGeometry, QgsPoint, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMapLayerRegistry, 
                        QgsMarkerSymbolV2, QgsLineSymbolV2, QgsFillSymbolV2 )
from PyQt4.QtCore import QVariant

import logging
log=logging.getLogger('umbra.layer')

from stompy.grid import unstructured_grid, orthogonalize
from stompy.utils import mag
from stompy.model.delft import dfm_grid

here=os.path.dirname(__file__)

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

def update_cell_quality(grid,cells=None,with_callback=True):
    """
    Recalculate cell_quality for the given list/array of cells, or
    for all cells if cells==None.
    with_callback: modify fields via modify_node, otherwise will
    directly modify grid.cells['cell_quality']
    """
    errors=grid.circumcenter_errors(cells=cells,radius_normalized=True)
    if with_callback:
        if cells is None:
            cells_errors=enumerate(errors)
        else:
            cells_errors=zip(cells,errors)
            
        for c,err in cells_errors:
            if not grid.cells['deleted'][c]:
                grid.modify_cell(c,cell_quality=err)
    else:
        grid.cells['cell_quality'][cells]=errors

# The edge quality assigned to an edge without two neighbors.
lonely_edge_quality=1.0
def update_edge_quality(g,edges=None,with_callback=True):
    # First part is calculating the values
    if edges is None:
        edges=slice(None)
    vc=g.cells_center()
    ec=g.edges_center()
    # this had been recalculating for all edges
    g.edge_to_cells(e=edges)

    diffs=vc[g.edges['cells'][edges,1]] - vc[g.edges['cells'][edges,0]] 
    if 0: # unsigned distance:
        c2c=mag(diffs)
    else: # signed distance
        normals=g.edges_normals(edges)
        c2c=normals[...,0]*diffs[...,0] + normals[...,1]*diffs[...,1]
    A=g.cells_area()
    Acc= A[g.edges['cells'][edges,:]].sum(axis=1)
    c2c=c2c / np.sqrt(Acc) # normalized - should be 1.0 for square grid.
    # used to set inf on borders, but that means auto-scaling color limits
    # don't work. 
    c2c[ np.any(g.edges['cells'][edges,:]<0,axis=1) ] = lonely_edge_quality
    if with_callback:
        if isinstance(edges,slice):
            edges=np.arange(g.Nedges())[slice]
        for e,quality in zip(edges,c2c):
            g.modify_edge(e,edge_quality=quality)
    else:
        g.edges['edge_quality'][edges]=c2c

class UmbraSubLayer(object):
    def __init__(self,log,grid,crs,prefix,tag=None):
        self.log=log
        self.grid=grid
        self.tag=tag
        self.extend_grid()
        self.crs=crs
        self.prefix=prefix
        self.frozen=False
        self.qlayer=self.create_qlayer()
        self.populate_qlayer()

    # This is a somewhat extreme version --
    def freeze(self):
        self.frozen=True
        self.disconnect_grid()
        
    def thaw(self):
        self.frozen=False
        self.connect_grid()
        # Makes use of the fact that populate_qlayer() drops all existing features
        self.populate_qlayer()
        
    def create_qlayer(self):
        return None
        
    def extend_grid(self):
        """ 
        install callbacks or additional fields as
        needed
        """
        pass
    def connect_grid(self):
        """ 
        part of extend grid - just the adding callbacks part
        """
        pass
    def unextend_grid(self):
        """ 
        remove those callbacks or additional fields
        """
        pass
    def disconnect_grid(self):
        """
        part of unextend grid, just the callback disconnects
        """
    def selection(self):
        return []

    def predefined_style(self,name):
        sld_path=os.path.join(here,'styles',name+'.sld')
        self.qlayer.loadSldStyle(sld_path)
        
class UmbraNodeLayer(UmbraSubLayer):
    def extend_grid(self):
        g=self.grid
        if 'feat_id' not in g.nodes.dtype.names:
            g.add_node_field('feat_id',
                             np.zeros(g.Nnodes(),'i4')-1)
        self.connect_grid()
    def connect_grid(self):
        self.grid.subscribe_after('modify_node',self.on_modify_node)
        self.grid.subscribe_after('add_node',self.on_add_node)
        self.grid.subscribe_before('delete_node',self.on_delete_node)
        
    def unextend_grid(self):
        g=self.grid
        g.delete_node_field('feat_id')
        self.disconnect_grid()
    def disconnect_grid(self):
        self.grid.unsubscribe_after('modify_node',self.on_modify_node)
        self.grid.unsubscribe_after('add_node',self.on_add_node)
        self.grid.unsubscribe_before('delete_node',self.on_delete_node)
        
    def create_qlayer(self):
        qlayer= QgsVectorLayer("Point"+self.crs, self.prefix+"-nodes", "memory")

        attrs=[QgsField("node_id",QVariant.Int)]
        casters=[int]

        for fidx,fdesc in enumerate(self.grid.node_dtype.descr):
            # descr gives string reprs of the types, use dtype
            # to get back to an object.
            fname=fdesc[0] ; ftype=np.dtype(fdesc[1])
            if len(fdesc)>2:
                fshape=fdesc[2]
            else:
                fshape=None

            if fname in ['x','feat_id']:
                continue
            
            if np.issubdtype(ftype,np.int):
                attrs.append( QgsField(fname,QVariant.Int) )
                casters.append(int)
            elif np.issubdtype(ftype,np.float):
                attrs.append( QgsField(fname,QVariant.Double) )
                casters.append(float)
            else:
                self.log.info("Not ready for other datatypes")
        self.attrs=attrs 
        self.casters=casters
        
        pr = qlayer.dataProvider()
        pr.addAttributes(attrs)
        qlayer.updateFields() # tell the vector layer to fetch changes from the provider
        
        # nice clean black dot
        symbol = QgsMarkerSymbolV2.createSimple({'outline_style':'no',
                                                 'name': 'circle', 
                                                 'size_unit':'MM',
                                                 'size':'1',
                                                 'color': 'black'})
        qlayer.rendererV2().setSymbol(symbol)
        return qlayer

    def populate_qlayer(self,clear=True):
        layer=self.qlayer
        # sometimes unnecessary, but there is now code which requires this.
        if clear:
            layer.dataProvider().deleteFeatures(layer.allFeatureIds())

        # takes an existing point memory layer, adds in nodes from g
        feats=[]
        valid=[]
        #for n in range(self.grid.Nnodes()): # why was I using this???
        for n in self.grid.valid_node_iter():
            valid.append(n)
            geom = self.node_geometry(n)
            feat = QgsFeature() # can't set feature_ids
            self.set_feature_attributes(n,feat)
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.nodes['feat_id'][valid] = [f.id() for f in outFeats]

    def set_feature_attributes(self,n,feat):
        feat.initAttributes(len(self.attrs))
        for idx,attr in enumerate(self.attrs):
            name=attr.name()
            typecode=attr.type()
            caster=self.casters[idx]
            item=self.grid.nodes[n]
            
            if name=='node_id':
                feat.setAttribute(idx,caster(n))
            else:
                feat.setAttribute(idx,caster(item[name]))

    def on_modify_node(self,g,func_name,n,**k):
        fid=self.grid.nodes[n]['feat_id']
        changed=False

        attr_changes={fid:{}}
            
        for idx,attr in enumerate(self.attrs):
            name=attr.name()
            if name in k:
                attr_changes[fid][idx]=self.casters[idx](k[name])

        if attr_changes[fid]:
            provider=self.qlayer.dataProvider()
            provider.changeAttributeValues(attr_changes)
            changed=True

        if 'x' in k:
            geom=self.node_geometry(n)
            self.qlayer.dataProvider().changeGeometryValues({fid:geom})
            changed=True
            
        if changed:
            self.qlayer.triggerRepaint()
        
    def on_add_node(self,g,func_name,return_value,**k):
        n=return_value
        geom=self.node_geometry(n)
        feat = QgsFeature() # can't set feature_ids
        self.set_feature_attributes(n,feat)
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

    def selection(self):
        # these are feature ids...
        node_feat_ids=[feat.id()
                       for feat in self.qlayer.selectedFeatures()]
        node_feat_ids=set(node_feat_ids)

        selected=[]
        for n in range(self.grid.Nnodes()):
            if self.grid.nodes['feat_id'][n] in node_feat_ids:
                selected.append(n)
        return selected

    def join_selected(self):
        return
        nodes=self.selection()
        # work out the kinks before dealing with more than two.
        # Seems that unstructured_grid isn't actually ready for this,
        # but has many of the pieces
        self.grid.merge_nodes(nodes[0],nodes[1])
        


class UmbraEdgeLayer(UmbraSubLayer):
    def create_qlayer(self):
        qlayer=QgsVectorLayer("LineString"+self.crs,self.prefix+"-edges","memory")
        
        pr = qlayer.dataProvider()

        e_attrs=[QgsField("edge_id",QVariant.Int)]
        casters=[int]

        for fidx,fdesc in enumerate(self.grid.edge_dtype.descr):
            # descr gives string reprs of the types, use dtype
            # to get back to an object.
            fname=fdesc[0] ; ftype=np.dtype(fdesc[1])
            if len(fdesc)>2:
                fshape=fdesc[2]
            else:
                fshape=None

            if fname in ['nodes','feat_id']:
                continue
            
            if fname=='cells':
                e_attrs += [QgsField("c0", QVariant.Int),
                            QgsField("c1", QVariant.Int)]
                casters += [int,int]
            else:
                if np.issubdtype(ftype,np.int):
                    e_attrs.append( QgsField(fname,QVariant.Int) )
                    casters.append(int)
                elif np.issubdtype(ftype,np.float):
                    e_attrs.append( QgsField(fname,QVariant.Double) )
                    casters.append(float)
                else:
                    self.log.info("Not ready for other datatypes")
        self.e_attrs=e_attrs 
        self.casters=casters
        
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
            self.set_feature_attributes(j,feat)
            
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.edges['feat_id'][valid]=[f.id() for f in outFeats]

    def set_feature_attributes(self,j,feat):
        feat.initAttributes(len(self.e_attrs))
        for idx,eattr in enumerate(self.e_attrs):
            name=eattr.name()
            typecode=eattr.type()
            caster=self.casters[idx]
            edge=self.grid.edges[j]
            
            if name=='edge_id':
                feat.setAttribute(idx,caster(j))
            elif name=='c0':
                feat.setAttribute(idx,caster(edge['cells'][0]))
            elif name=='c1':
                feat.setAttribute(idx,caster(edge['cells'][1]))
            else:
                feat.setAttribute(idx,caster(edge[name]))
            # elif typecode==2: # integer
            #     feat.setAttribute(idx,int(self.grid.edges[name][j]))
            # elif typecode==6: # double
            #     feat.setAttribute(idx,float(self.grid.edges[name][j]))
            # else:
            #     continue
        
    def extend_grid(self):
        g=self.grid
        if 'feat_id' not in g.edges.dtype.names:
            g.add_edge_field('feat_id',
                             np.zeros(g.Nedges(),'i4')-1)
        self.connect_grid()
    def unextend_grid(self):
        g=self.grid
        g.delete_edge_field('feat_id','edge_quality')
        self.disconnect_grid()
        
    def connect_grid(self):
        self.grid.subscribe_after('add_edge',self.on_add_edge)
        self.grid.subscribe_before('delete_edge',self.on_delete_edge)
        self.grid.subscribe_after('modify_node',self.on_modify_node)
        self.grid.subscribe_after('modify_edge',self.on_modify_edge)

    def disconnect_grid(self):
        self.grid.unsubscribe_after('add_edge',self.on_add_edge)
        self.grid.unsubscribe_before('delete_edge',self.on_delete_edge)
        self.grid.unsubscribe_after('modify_node',self.on_modify_node)
        self.grid.unsubscribe_after('modify_edge',self.on_modify_edge)

    def on_add_edge(self,g,func_name,return_value,**k):
        j=return_value
        feat=QgsFeature()

        self.set_feature_attributes(j,feat)
        feat.setGeometry(self.edge_geometry(j))
        (res,outFeats) = self.qlayer.dataProvider().addFeatures([feat])

        self.grid.edges['feat_id'][j] = outFeats[0].id()
        self.qlayer.triggerRepaint()
        self.log.info("Just added an edge")
        
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

        self.qlayer.triggerRepaint()

    def on_modify_edge(self,g,func_name,j,**k):
        fid=g.edges['feat_id'][j]
        attr_changes={fid:{}}
            
        for idx,attr in enumerate(self.e_attrs):
            name=attr.name()
            if name in k:
                attr_changes[fid][idx]=self.casters[idx](k[name])
        if not attr_changes[fid]:
            return
        
        provider=self.qlayer.dataProvider()
        provider.changeAttributeValues(attr_changes)
        
        self.qlayer.triggerRepaint()
                       
    def edge_geometry(self,j):
        seg=self.grid.nodes['x'][self.grid.edges['nodes'][j]]
        pnts=[QgsPoint(seg[0,0],seg[0,1]),
              QgsPoint(seg[1,0],seg[1,1])]
        return QgsGeometry.fromPolyline(pnts)


class UmbraCellLayer(UmbraSubLayer):
    # the name of the field added to the grid to track the features here
    feat_id_name='feat_id'
    
    def create_qlayer(self):
        qlayer=QgsVectorLayer("Polygon"+self.crs,self.prefix+"-cells","memory")

        c_attrs=[QgsField("cell_id",QVariant.Int)]
        casters=[int]
        
        for fidx,fdesc in enumerate(self.grid.cell_dtype.descr):
            # descr gives string reprs of the types, use dtype
            # to get back to an object.
            fname=fdesc[0] ; ftype=np.dtype(fdesc[1])
            if len(fdesc)>2:
                fshape=fdesc[2]
            else:
                fshape=None

            if fname.startswith('_') or fname in ['nodes','edges','deleted','feat_id']:
                continue
            else:
                self.log.info("Trying to add field for %s"%fname)
                
                if np.issubdtype(ftype,np.int):
                    c_attrs.append( QgsField(fname,QVariant.Int) )
                    casters.append(int)
                elif np.issubdtype(ftype,np.float):
                    c_attrs.append( QgsField(fname,QVariant.Double) )
                    casters.append(float)
                else:
                    self.log.info("Not ready for other datatypes")
        self.c_attrs=c_attrs
        self.casters=casters
        
        pr = qlayer.dataProvider()
        pr.addAttributes(c_attrs)
        qlayer.updateFields() # tell the vector layer to fetch changes from the provider
        
        # transparent red, no border
        # but this is the wrong class...
        symbol = QgsFillSymbolV2.createSimple({'outline_style':'no',
                                               'style':'solid',
                                               'color': '249,0,0,78'})
        qlayer.rendererV2().setSymbol(symbol)
        return qlayer
        
    def extend_grid(self):
        g=self.grid
        if self.feat_id_name not in g.cells.dtype.names:
            g.add_cell_field(self.feat_id_name,
                             np.zeros(g.Ncells(),'i4')-1)
        self.connect_grid()
    def unextend_grid(self):
        g=self.grid
        if self.feat_id_name in g.cells.dtype.names:
            g.delete_cell_field(self.feat_id_name)
        self.disconnect_grid()
      
    def connect_grid(self):
        self.grid.subscribe_after('add_cell',self.on_add_cell)
        self.grid.subscribe_before('delete_cell',self.on_delete_cell)
        self.grid.subscribe_after('modify_node',self.on_modify_node)
        self.grid.subscribe_after('modify_cell',self.on_modify_cell)
    def disconnect_grid(self):
        self.grid.unsubscribe_after('add_cell',self.on_add_cell)
        self.grid.unsubscribe_before('delete_cell',self.on_delete_cell)
        self.grid.unsubscribe_after('modify_node',self.on_modify_node)
        self.grid.unsubscribe_after('modify_cell',self.on_modify_cell)
        
    def on_delete_cell(self,g,func_name,c,**k):
        feat_id=self.grid.cells[self.feat_id_name][c]
        self.log.info('got signal for delete cell %d, feat_id %s'%(c,feat_id))
        self.qlayer.dataProvider().deleteFeatures([feat_id])
        self.qlayer.triggerRepaint()

    def on_modify_node(self,g,func_name,n,**k):
        if 'x' not in k:
            return
        
        cell_changes={}
        cells=self.grid.node_to_cells(n)
        # These are handled in UmbraLayer now.
        # self.grid.cells_center(refresh=cells)
        # self.grid.cells['_area'][cells]=np.nan # trigger recalc.
        # cell_edges=set()
        for i in cells:
            # this was all sorts of messed up - don't understand how
            # it was working at all before...
            fid=self.grid.cells[i][self.feat_id_name]
            geom=self.cell_geometry(i)
            cell_changes[fid]=geom
            # cell_edges.update(self.grid.cell_to_edges(i))

        if len(cell_changes):
            provider=self.qlayer.dataProvider()
            provider.changeGeometryValues(cell_changes)
            self.qlayer.triggerRepaint()
        
    def on_modify_cell(self,g,func_name,c,**k):
        fid=g.cells['feat_id'][c]
        attr_changes={fid:{}}
            
        for idx,attr in enumerate(self.c_attrs):
            name=attr.name()
            if name in k:
                attr_changes[fid][idx]=self.casters[idx](k[name])
        if not attr_changes[fid]:
            return
        
        provider=self.qlayer.dataProvider()
        provider.changeAttributeValues(attr_changes)
        
        self.qlayer.triggerRepaint()
        self.log.info("UmbraCellLayer:on_modify_node for cell %s"%c)
        self.log.info("attr_changes: %s"%str(attr_changes))

    def on_add_cell(self,g,func_name,return_value,**k):
        c=return_value
        self.log.info('got signal for add cell')
        geom=self.cell_geometry(c)
        feat = QgsFeature() # can't set feature_ids
        self.set_feature_attributes(c,feat)
        
        feat.setGeometry(geom)
        (res, outFeats) = self.qlayer.dataProvider().addFeatures([feat])

        self.grid.cells[self.feat_id_name][c] = outFeats[0].id()
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
            valid.append(i)
            
            feat = QgsFeature()
            self.set_feature_attributes(i,feat)
            
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.cells[self.feat_id_name][valid]=[f.id() for f in outFeats]

    def set_feature_attributes(self,i,feat):
        feat.initAttributes(len(self.c_attrs))
        for idx,cattr in enumerate(self.c_attrs):
            name=cattr.name()
            typecode=cattr.type()
            if name=='cell_id':
                feat.setAttribute(idx,i)
            else:
                feat.setAttribute(idx,self.casters[idx](self.grid.cells[name][i]))
            # older, manual way
            # elif typecode==2: # integer
            #     feat.setAttribute(idx,int(self.grid.cells[name][i]))
            # elif typecode==6: # double
            #     feat.setAttribute(idx,float(self.grid.cells[name][i]))
            # else:
            #     continue
            # QGIS doesn't know about numpy types

    def selection(self):
        # these are feature ids...
        cell_feat_ids=[feat.id()
                       for feat in self.qlayer.selectedFeatures()]
        cell_feat_ids=set(cell_feat_ids)

        selected=[]
        for c in range(self.grid.Ncells()):
            if self.grid.cells[self.feat_id_name][c] in cell_feat_ids:
                selected.append(c)
        return selected


class UmbraCellCenterLayer(UmbraCellLayer):
    """
    First try at additional derived layers
    """
    feat_id_name='center_feat_id'
    def create_qlayer(self):
        layer=QgsVectorLayer("Point"+self.crs,self.prefix+"-centers","memory")

        # for now, cell centers don't carry all of the fields like cell polygons do.
        self.c_attrs=[]
        self.casters=[]
        
        symbol = QgsMarkerSymbolV2.createSimple({'outline_style':'no',
                                                 'name': 'circle', 
                                                 'size_unit':'MM',
                                                 'size':'1',
                                                 'color': 'green'})
        layer.rendererV2().setSymbol(symbol)
        return layer

    # extend, unextend, on_delete_cell: use inherited method of UmbraCellLayer
       
    def cell_geometry(self,i):
        pnts=[QgsPoint(self.grid.nodes['x'][n,0],
                       self.grid.nodes['x'][n,1])
              for n in self.grid.cell_to_nodes(i)]
        cc=self.grid.cells_center()
        return QgsGeometry.fromPoint( QgsPoint(cc[i,0],cc[i,1]) )

    # def populate_qlayer(self): # same as cell Layer

    # def selection(self): # same as Cell Layer

    
class UmbraLayer(object):
    count=0

    layer_count=0
    crs="?crs=epsg:26910" # was 4326
    
    def __init__(self,umbra,grid,name=None,path=None,grid_format=None):
        """
        Does not add the layers to the GUI - call register_layers
        for that.
        """
        # having some trouble getting reliable output from the log...
        if 1:
            self.log=log
        else:
            class DumbLog(object):
                def info(self,*a):
                    s=" ".join(a)
                    with open(os.path.join(here,'log'),'a') as fp:
                        fp.write(s+"\n")
                        fp.flush()
            dlog=DumbLog()
            dlog.debug=dlog.info
            dlog.warning=dlog.info
            dlog.error=dlog.info
            self.log=dlog

        self.frozen=False
        self.umbra=umbra
        # for writing metadata to project files to reload the layer
        self.grid_format=grid_format
        self.path=path

        self.grid=grid

        self.connect_grid()
        
        self.iface=None # gets set in register_layers

        UmbraLayer.layer_count+=1

        if name is None:
            name="grid%d"%UmbraLayer.layer_count
        self.name=name 

        self.layers=[] # SubLayer objects associated with this grid.
        self.umbra.register_grid(self)
        
        self.undo_stack=QUndoStack()

    def write_to_project(self,prj,scope,doc,tag):
        if not scope.endswith('/'):
            scope=scope+'/'

        prj.writeEntry(scope,tag+'name',self.name)
        prj.writeEntry(scope,tag+'grid_format',str(self.grid_format))
        prj.writeEntry(scope,tag+'path',str(self.path))

    @classmethod
    def load_from_project(klass,umbra,prj,scope,tag):
        name,_=prj.readEntry(scope,tag+'name','')
        grid_format,_=prj.readEntry(scope,tag+'grid_format',"")
        path,_=prj.readEntry(scope,tag+'path',"")

        if '' in [name,grid_format,path]:
            self.log.error("Project file missing requisite data to load an umbra layer")
            return

        return klass.open_layer(umbra=umbra,grid_format=grid_format,path=path,name=name)

        
    def connect_grid(self):
        """
        Install callbacks on the grid to update cell and edge quality fields.
        The individual layers will be later than these callbacks.
        """
        #self.grid.subscribe_after('add_cell',self.on_add_cell)
        #self.grid.subscribe_before('delete_cell',self.on_delete_cell)

        g=self.grid
        if 'cell_quality' not in g.cells.dtype.names:
            g.add_cell_field('cell_quality',
                             np.zeros(g.Ncells(),'f8'))

        if 'edge_quality' not in g.edges.dtype.names:
            edge_q=np.zeros(self.grid.Nedges(),'f8')
            g.add_edge_field('edge_quality',edge_q)
            
        update_edge_quality(g,edges=None,with_callback=False)
            
        # shortcut the callbacks 
        update_cell_quality(g,with_callback=False)

        self.grid.subscribe_after('modify_node',self.on_modify_node)
        self.grid.subscribe_after('add_cell',self.on_add_cell)
        self.grid.subscribe_after('add_edge',self.on_add_edge)

    def on_modify_node(self,g,func_name,n,**k):
        if 'x' not in k:
            return
        
        cell_changes={}
        cells=self.grid.node_to_cells(n)
        self.grid.cells_center(refresh=cells)
        self.grid.cells['_area'][cells]=np.nan # trigger recalc.
        cell_edges=set()
        for i in cells:
            cell_edges.update(self.grid.cell_to_edges(i))

        self.log.info("UmbraLayer:on_modify_node update cells %s"%cells)
        
        update_cell_quality(self.grid,cells,with_callback=True)
        update_edge_quality(self.grid,list(cell_edges),with_callback=True)

    def on_add_cell(self,g,func_name,return_value,**k):
        if 'cell_quality' not in k:
            update_cell_quality(g,cells=[return_value],
                                with_callback=False)
    def on_add_edge(self,g,func_name,return_value,**k):
        if 'edge_quality' not in k:
            update_edge_quality(g,edges=[return_value],
                                with_callback=False)
            # .edges['edge_quality'][j]=lonely_edge_quality
        
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
    def open_layer(klass,umbra,grid_format,path,name=None):
        g=klass.load_grid(path=path,grid_format=grid_format)
        return klass(umbra=umbra,grid=g,path=path,grid_format=grid_format,name=name)

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
        self.iface=self.umbra.iface
        self.create_group()

        self.register_layer( UmbraCellLayer(self.log,
                                            self.grid,
                                            crs=self.crs,
                                            prefix=self.name,
                                            tag='cells') )

        self.register_layer( UmbraEdgeLayer(self.log,
                                            self.grid,
                                            crs=self.crs,
                                            prefix=self.name,
                                            tag='edges' ) )

        self.register_layer( UmbraNodeLayer(self.log,
                                            self.grid,
                                            crs=self.crs,
                                            prefix=self.name,
                                            tag='nodes') )
        
        # set extent to the extent of our layer
        # skip while developing
        # canvas.setExtent(layer.extent())
        
    def add_centers_layer(self):
        self.register_layer( UmbraCellCenterLayer(self.log,
                                                  self.grid,
                                                  crs=self.crs,
                                                  prefix=self.name,
                                                  tag='centers') )

    def set_edge_quality_style(self):
        sl = self.layer_by_tag('edges')
        if sl is not None:
            sl.predefined_style('edge-quality')
        
    def set_cell_quality_style(self):
        sl=self.layer_by_tag('cells')
        if sl is not None:
            sl.predefined_style('cell-quality')
        
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

    def orthogonalize_local(self,xy,iterations=1):
        """
        If nodes are selected, then adjust the position of those nodes.
        If no nodes are selected, choose the nearest cell and adjust all
        of its nodes.
        """
        tweaker=orthogonalize.Tweaker(self.grid)
        
        nl=self.layer_by_tag('nodes')
        if nl is not None:
            nl_sel=nl.selection()
            if len(nl_sel)>0:
                for it in range(iterations):
                    for n in nl_sel:
                        tweaker.nudge_node_orthogonal(n)
                return
        
        c=self.grid.select_cells_nearest(xy)

        for it in range(iterations):
            tweaker.nudge_cell_orthogonal(c)
    
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

    def merge_nodes_of_edge(self,e):
        n0,n1 = self.grid.edges['nodes'][e]
        
        cmd=GridCommand(self.grid,
                        "Merge nodes",
                        lambda n0=n0,n1=n1: self.grid.merge_nodes(n0,n1))

        self.undo_stack.push(cmd)
    
    def delete_selected(self):
        layer=self.layer_by_tag('cells')
        if layer is not None:
            selected = layer.selection()
            self.log.info("Found %d selected cells"%len(selected))
            def redo(selected=selected): # pass this way b/c of python bindings weirdness
                for c in selected:
                    self.grid.delete_cell(c)
            cmd=GridCommand(self.grid,"Delete items",redo)
            self.undo_stack.push(cmd)
        else:
            self.log.info("delete_selected: didn't find layer!")

        layer=self.layer_by_tag('edges')
        if layer is not None:
            selected = layer.selection()
            self.log.info("Found %d selected cells"%len(selected))
            def redo(selected=selected): # pass this way b/c of python bindings weirdness
                for c in selected:
                    self.grid.delete_edge_cascade(c)
            cmd=GridCommand(self.grid,"Delete edges",redo)
            self.undo_stack.push(cmd)
        else:
            self.log.info("delete_selected: didn't find edge layer!")
        
        layer=self.layer_by_tag('nodes')
        if layer is not None:
            selected = layer.selection()
            self.log.info("Found %d selected cells"%len(selected))
            def redo(selected=selected): # pass this way b/c of python bindings weirdness
                for c in selected:
                    self.grid.delete_node_cascade(c)
            cmd=GridCommand(self.grid,"Delete nodes",redo)
            self.undo_stack.push(cmd)
        else:
            self.log.info("delete_selected: didn't find node layer!")
    def combine_with(self,src_layer):
        self.freeze()
        self.grid.add_grid(src_layer.grid)
        self.thaw()
    def freeze(self):
        self.frozen=True
        for sublayer in self.layers:
            sublayer.freeze()
    def thaw(self):
        self.frozen=False
        for sublayer in self.layers:
            sublayer.thaw()
