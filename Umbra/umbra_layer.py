import os
import copy
import sys
import numpy as np
import time

# HERE: https://gis.stackexchange.com/questions/75384/add-layer-to-a-qgis-group-using-python

try:
    from six import iteritems
except ImportError:
    def iteritems(d):
        return iter(d.items())

sys.path.append( os.path.join(os.environ["HOME"],"python") )

from qgis.core import ( QgsGeometry, QgsPointXY, QgsFeature,QgsVectorLayer, QgsField,
                        QgsMarkerSymbol, QgsLineSymbol, QgsFillSymbol, QgsProject,
                        QgsLayerTreeLayer, Qgis )
from qgis.PyQt.QtCore import QVariant

import logging
log=logging.getLogger('umbra.layer')

from stompy.grid import unstructured_grid, orthogonalize
from stompy.utils import mag
from stompy.model.delft import dfm_grid

here=os.path.dirname(__file__)

# Used to be a QgsPluginLayer, but no more.
# now it manages the layers and data specific to a grid

# Maybe the Undo handling will live here?
from qgis.PyQt.QtWidgets import QUndoCommand, QUndoStack
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
            cells_errors=list(zip(cells,errors))

        for c,err in cells_errors:
            if not grid.cells['deleted'][c]:
                grid.modify_cell(c,cell_quality=err)
    else:
        grid.cells['cell_quality'][cells]=errors


def update_node_degree(grid,nodes=None,with_callback=True):
    if nodes is None:
        nodes=np.nonzero(~grid.nodes['deleted'])[0]

    for n in nodes:
        new_degree=grid.node_degree(n)
        if grid.nodes['degree'][n]!=new_degree:
            if with_callback:
                grid.modify_node(n,degree=new_degree)
            else:
                grid.nodes['degree'][n]=new_degree

# The edge quality assigned to an edge without two neighbors.
lonely_edge_quality=1.0

# Test both for center spacing, and for center-edge spacing
one_sided_test=True

def update_edge_quality(g,edges=None,with_callback=True):
    # First part is calculating the values
    # print("Edges came in as ",edges)

    if edges is None:
        # edges=slice(None)
        # This code will usually get called with specific indices,
        # so don't worry about optimizing the code for slices
        # or global evaluation
        edges=np.nonzero( ~g.edges['deleted'] )[0]
    elif isinstance(edges,slice):
        edges=np.arange(g.Nedges())[edges]
    elif isinstance(edges,list):
        edges=np.asarray(edges,np.int32)

    # so now edges is guaranteed to be an array of indices
    # print("edges dtype is ",edges.dtype)

    # these could become performance bottle necks.
    vc=g.cells_center()

    # this had been recalculating for all edges
    g.edge_to_cells(e=edges)

    # need special treatment for edges that don't have two cells.
    external=np.any(g.edges['cells'][edges,:]<0,axis=1)
    c2c=np.zeros( len(external), np.float64)

    int_edges=edges[~external]

    # ec_int=g.edges_center(int_edges) # YES
    # use direct approach, where we can skip some sanity checks that are in
    # unstructured_grid.edges_center()
    ec_int = g.nodes['x'][g.edges['nodes'][int_edges,:]].mean(axis=1)

    # this did not before have the int_edges bit, but that
    # I think was a problem because d is coming in just for int_edges,
    # but normals was not.
    normals=g.edges_normals(int_edges)
    dist_signed=lambda d: normals[...,0]*d[...,0] + normals[...,1]*d[...,1]

    if not one_sided_test:
        diffs=vc[g.edges['cells'][int_edges,1]] - vc[g.edges['cells'][int_edges,0]]
        c2c[~external]=dist_signed(diffs)
    else:
        diff_left =vc[g.edges['cells'][int_edges,1]] - ec_int
        diff_right=ec_int - vc[g.edges['cells'][int_edges,0]]
        # Factor of 2 so that overall range is similar to double-sided test
        c2cl=2*dist_signed(diff_left)
        c2cr=2*dist_signed(diff_right)

        c2c[~external]=np.minimum(c2cl,c2cr)

    cell_select=g.edges['cells'][int_edges,:]
    Acc=g.cells_area(cell_select).sum(axis=1)

    c2c[~external] /= np.sqrt(Acc) # normalized - should be 1.0 for square grid.
    # used to set inf on borders, but that means auto-scaling color limits
    # don't work.
    c2c[ external ] = lonely_edge_quality
    if with_callback:
        for e,quality in zip(edges,c2c):
            g.modify_edge(e,edge_quality=quality)
    else:
        g.edges['edge_quality'][edges]=c2c


def float_or_null(v):
    if np.isnan(v):
        return None
    else:
        return float(v)
    
class UmbraSubLayer(object):
    int_nan=-9999 # how to convert NULL into an int.
    
    def __init__(self,parent,prefix,tag=None,qlayer=None):
        """
        qlayer: not robust. for the case where a layer
        already exists, and we just populate it.  But maybe too fragile to assume
        that the existing layer is defined in the same way as a new layer
        would be?
        """
        self.parent=parent
        self.log=parent.log
        self.grid=parent.grid
        self.tag=tag
        self.extend_grid()
        self.crs=parent.crs
        self.prefix=prefix
        self.frozen=False

        self.log.info("SubLayer: qlayer=%s"%qlayer)
        
        if qlayer is not None:
            self.log.info("Clearing old fields from layer")
            pr = qlayer.dataProvider()
            attr_idxs = pr.attributeIndexes() #  list of ints
            pr.deleteAttributes(attr_idxs)
            qlayer.updateFields() # tell the vector layer to fetch changes from the provider

        self.qlayer=self.create_qlayer(existing=qlayer)

        self.populate_qlayer()
        # connect signals to see if attributes are modified
        self.connect_qlayer()

    # This is a somewhat extreme version --
    def freeze(self):
        self.frozen=True
        self.disconnect_grid()

    def thaw(self):
        self.frozen=False
        self.connect_grid()
        # Makes use of the fact that populate_qlayer() drops all existing features
        self.populate_qlayer()

    def create_qlayer(self,existing=None):
        return None

    def connect_qlayer(self):
        self.qlayer.beforeCommitChanges.connect(self.on_beforeCommitChanges)

    def update_title_abstract(self):
        """
        Set or update the qlayer title and abstract, which generally
        show the file path
        """
        if self.qlayer is None: return
        if self.parent is None: return
        if self.parent.path is None: return

        self.qlayer.setTitle(os.path.basename(self.parent.path))
        self.qlayer.setAbstract(self.parent.path)
        
        
    edits_to_commit=None
    
    def on_beforeCommitChanges(self):
        """
        Ideally these would get copied, and then when the 
        edits were known to be successfully committed these
        changes would get mirrored to the grid. However, the
        commit process isn't that simple, and since it 
        is going to a memory layer the likelihood of the commit
        failing is minimal.
        """
        log.info("beforeCommitChanges: applying edit buffer")
        eb=self.qlayer.editBuffer()
        
        for field in eb.addedAttributes():
            # QgsField object
            name=field.name()
            if field.typeName()=='double':
                np_type=np.float64
            elif field.typeName()=='integer':
                np_type=np.int32
            else:
                np_type=object
            self.add_field(field, field.name(), np_type)

        # { feat_id: {attr_id: new_value, ... }, ... }
        changes=eb.changedAttributeValues()

        for feat_id in changes:
            for attr_id in changes[feat_id]:
                new_val=changes[feat_id][attr_id]
                self.update_values(feat_id,attr_id,new_val)
            
    def add_field(self,field,name,np_type):
        log.warning("add_field not implemented for %s"%str(self))

    def add_field_generic(self,field,name,np_type,adder,n_items):
        """
        Common code for sublcass add_field methods.
        field,name,np_type: same as passed to add_field
        adder: self.grid.add_{node,cell,edge}_field
        n_items: self.grid.{Nnodes,Ncells,Nedges}()
        """
        values=np.zeros(n_items,np_type)
        adder(name,values)
        self.attrs.append(field)
        
        if np_type==np.float64:
            self.casters.append(float_or_null)
        elif np_type==np.int32:
            self.casters.append(int)
        else:
            self.casters.append(lambda x:x)
        
    def update_values(self,feat_id,attr_id,value):
        """
        TODO: currently if there are multiple layers for the same feature
        type, an attribute edit on one layer won't be seen in the other.
        Maybe should just have a single editable feature?  for example,
        you can't edit cell fields on a CellCenterLayer?
        """
        log.warning("update_values not implemented for %s"%str(self))

    def update_values_generic(self,target,feat_id,attr_id,value):
        """
        Handle a GUI update to an attribute value. update_values() is
        defined on sublcasses, to specify the details which are then
        utilized here in a more generic process.

        target: the numpy array (e.g. grid.cells) to update
        feat_id: the QGIS feature id being updated.  this will get mapped
         to an unstructured_grid id.
        """
        grid_name=self.attrs[attr_id].name()
        items=np.nonzero( (~target['deleted'])
                          & (target['feat_id']==feat_id) )[0] # SLOW! (maybe)
        if len(items)!=1:
            msg="feat_id %s matched items %s"%( feat_id,
                                                items )
            log.error(msg)
            if self.parent.iface is not None: 
                self.parent.iface.messageBar().pushMessage("Error", msg, level=Qgis.Warning)
            return 
            
        # Casters are intended for the other way around, but used here
        # to guide some special checks.
        caster=self.casters[attr_id]

        # Losing track a bit -- seems that 
        # value is coming in as float, None, or QVariant
        if (value is None) or (isinstance(value,QVariant) and value.isNull()):
            if (caster==float) or (caster==float_or_null):
                np_value=np.nan
            elif caster==int:
                np_value=self.int_nan
            else:
                log.warning("Not sure how to handle NULL field and caster=%s"%caster)
                np_value=value
        else:
            # Hmm - value might be a QVariant?  Or not.
            # Not sure what's up. when in doubt, sit on hands.
            # np_value=self.casters[attr_id]( value )
            np_value=value
            
        target[grid_name][items]=np_value
        log.info("update_values generic: grid_name %s value %s => np_value %s"%(grid_name,
                                                                                value,np_value))

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
    track_node_degree=False
    def extend_grid(self):
        g=self.grid
        if 'feat_id' not in g.nodes.dtype.names:
            g.add_node_field('feat_id',
                             np.zeros(g.Nnodes(),'i4')-1)

        if self.track_node_degree:
            g.add_node_field('degree',-np.ones(g.Nnodes(),np.int8),on_exists='pass')
            update_node_degree(g,with_callback=False)

        self.connect_grid()
    def connect_grid(self):
        self.grid.subscribe_after('modify_node',self.on_modify_node)
        self.grid.subscribe_after('add_node',self.on_add_node)
        self.grid.subscribe_before('delete_node',self.on_delete_node)

        if self.track_node_degree:
            self.grid.subscribe_after('add_edge',self.on_add_edge)
            self.grid.subscribe_before('delete_edge',self.on_delete_edge)

    def unextend_grid(self):
        g=self.grid
        g.delete_node_field('feat_id')
        self.disconnect_grid()
    def disconnect_grid(self):
        self.grid.unsubscribe_after('modify_node',self.on_modify_node)
        self.grid.unsubscribe_after('add_node',self.on_add_node)
        self.grid.unsubscribe_before('delete_node',self.on_delete_node)

        if self.track_node_degree:
            self.grid.unsubscribe_after('add_edge',self.on_add_edge)
            self.grid.unsubscribe_before('delete_edge',self.on_delete_edge)

    def create_qlayer(self,existing=None):
        if existing:
            # assumes that caller has cleared out the data table
            # and we can start from a clean slate.
            qlayer=existing
        else:
            qlayer= QgsVectorLayer("Point"+self.crs, self.prefix+"-nodes", "memory")

        attrs=[QgsField("node_id",QVariant.Int)]

        # The primary point of casters is to get from numpy types to plain python types.
        casters=[int]

        for fidx,fdesc in enumerate(self.grid.nodes.dtype.descr):
            # descr gives string reprs of the types, use dtype
            # to get back to an object.
            fname=fdesc[0] ; ftype=np.dtype(fdesc[1])
            if len(fdesc)>2:
                fshape=fdesc[2]
            else:
                fshape=None

            if fname in ['x','feat_id']:
                continue

            if fshape is not None:
                self.log.warning("Not ready for non-scalar fields (%s: %s)"%(fname,str(ftype)))
                continue
            
            if np.issubdtype(ftype,np.int):
                attrs.append( QgsField(fname,QVariant.Int) )
                casters.append(int)
            elif np.issubdtype(ftype,np.floating):
                attrs.append( QgsField(fname,QVariant.Double) )
                casters.append(float_or_null)
            else:
                self.log.info("Not ready for other datatypes (%s: %s)"%(fname,str(ftype)))
        self.attrs=attrs
        self.casters=casters

        pr = qlayer.dataProvider()
        pr.addAttributes(attrs)
        qlayer.updateFields() # tell the vector layer to fetch changes from the provider

        if not existing:
            # nice clean black dot
            symbol = QgsMarkerSymbol.createSimple({'outline_style':'no',
                                                   'name': 'circle',
                                                   'size_unit':'MM',
                                                   'size':'1',
                                                   'color': 'black'})
            qlayer.renderer().setSymbol(symbol)
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
            geom = self.node_geometry(n)
            valid.append(n)
            feat = QgsFeature() # can't set feature_ids
            self.set_feature_attributes(n,feat)
            feat.setGeometry(geom)
            feats.append(feat)
        (res, outFeats) = layer.dataProvider().addFeatures(feats)
        self.grid.nodes['feat_id'][valid] = [f.id() for f in outFeats]

    def set_feature_attributes(self,n,feat):
        """
        push attributes for grid.nodes[n] to the provide 
        QGIS feature feat.
        """
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

    def update_values(self,feat_id,attr_id,value):
        """
        Copy QGIS changes in attribute values to the grid
        """
        self.update_values_generic(self.grid.nodes,feat_id,attr_id,value)

    def add_field(self,field,name,np_type):
        self.add_field_generic(field,name,np_type,
                               self.grid.add_node_field,self.grid.Nnodes())
        
                
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
        self.log.info("Just added a node - %s"%geom.asWkt())
        self.log.info("in the grid, node %d has x=%s"%(n,self.grid.nodes['x'][n]))

        (res, outFeats) = self.qlayer.dataProvider().addFeatures([feat])

        self.grid.nodes['feat_id'][n] = outFeats[0].id()
        self.qlayer.triggerRepaint()

        self.log.info("on_add_node: requested repaint from qlayer=%s"%self.qlayer)

    def on_delete_node(self,g,func_name,n,**k):
        self.log.info("on_delete_node")
        self.qlayer.dataProvider().deleteFeatures([self.grid.nodes['feat_id'][n]])
        self.qlayer.triggerRepaint()

    def on_delete_edge(self,g,func_name,j,**k):
        if self.track_node_degree:
            update_node_degree(self.grid,self.grid.edges['nodes'][j],with_callback=True)
    def on_add_edge(self,g,func_name,return_value,**k):
        if self.track_node_degree:
            update_node_degree(self.grid,k['nodes'],with_callback=True)

    def node_geometry(self,n):
        return QgsGeometry.fromPointXY(QgsPointXY(self.grid.nodes['x'][n,0],
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
    def create_qlayer(self,existing=None):
        if existing:
            qlayer=existing
        else:
            qlayer=QgsVectorLayer("LineString"+self.crs,self.prefix+"-edges","memory")

        pr = qlayer.dataProvider()

        attrs=[QgsField("edge_id",QVariant.Int)]
        casters=[int]

        for fidx,fdesc in enumerate(self.grid.edges.dtype.descr):
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
                # Not sure when it would be useful to have this, and the machinery
                # is not robust for vector-valued fields, or even pushing updates
                # from UnstructuredGrid.
                continue 
                #attrs += [QgsField("c0", QVariant.Int),
                #          QgsField("c1", QVariant.Int)]
                #casters += [int,int]
            else:
                if np.issubdtype(ftype,np.int):
                    attrs.append( QgsField(fname,QVariant.Int) )
                    casters.append(int)
                elif np.issubdtype(ftype,np.float):
                    attrs.append( QgsField(fname,QVariant.Double) )
                    casters.append(float_or_null)
                else:
                    self.log.info("Not ready for other datatypes (%s: %s)"%(fname,str(ftype)))

        self.attrs=attrs
        self.casters=casters

        pr.addAttributes(attrs)
        qlayer.updateFields() # tell the vector layer to fetch changes from the provider

        if not existing:
            # clean, thin black style
            symbol = QgsLineSymbol.createSimple({'line_style':'solid',
                                                 'line_width':'0.2',
                                                 'line_width_unit':'MM',
                                                 'line_color': 'black'})
            qlayer.renderer().setSymbol(symbol)
        return qlayer

    def populate_qlayer(self):
        layer=self.qlayer
        # shouldn't be necessary
        layer.dataProvider().deleteFeatures(layer.allFeatureIds())

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
        feat.initAttributes(len(self.attrs))
        for idx,eattr in enumerate(self.attrs):
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
            
    def add_field(self,field,name,np_type):
        self.add_field_generic(field,name,np_type,
                               self.grid.add_edge_field,self.grid.Nedges())
        
    def update_values(self,feat_id,attr_id,value):
        """
        Copy QGIS changes in attribute values to the grid
        """
        self.update_values_generic(self.grid.edges,feat_id,attr_id,value)

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
        geom=self.edge_geometry(j)
        feat.setGeometry(geom)
        (res,outFeats) = self.qlayer.dataProvider().addFeatures([feat])

        self.grid.edges['feat_id'][j] = outFeats[0].id()
        self.qlayer.triggerRepaint()
        self.log.info("Just added an edge - %s"%geom.asWkt())

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

        for idx,attr in enumerate(self.attrs):
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
        if np.any(np.isnan(seg)):
            self.log.warning("Edge geometry for j=%d, nodes=%s, has coords %s"%
                             (j,self.grid.edges['nodes'][j],str(seg)))
        pnts=[QgsPointXY(seg[0,0],seg[0,1]),
              QgsPointXY(seg[1,0],seg[1,1])]
        return QgsGeometry.fromPolylineXY(pnts)

    def selection(self):
        feat_ids=[feat.id()
                  for feat in self.qlayer.selectedFeatures()]
        feat_ids=set(feat_ids)

        selected=[]
        for n in range(self.grid.Nedges()):
            if self.grid.edges['feat_id'][n] in feat_ids:
                selected.append(n)
        return selected
        
    def set_selection(self,js):
        """
        Update QGIS edge selection.  
        js: sequence of edge indexes
        """
        # qgis api doesn't know ndarray
        feat_ids=list(self.grid.edges['feat_id'][js])
        ql=self.qlayer
        ql.modifySelection(feat_ids, # to select
                           ql.selectedFeatureIds()) # to deselect
                           
class UmbraCellLayer(UmbraSubLayer):
    # the name of the field added to the grid to track the features here
    feat_id_name='feat_id'

    def create_qlayer(self,existing=None):
        self.log.info("CellLayer: existing=%s"%existing)
        if existing:
            qlayer=existing
        else:
            qlayer=QgsVectorLayer("Polygon"+self.crs,self.prefix+"-cells","memory")

        attrs=[QgsField("cell_id",QVariant.Int)]
        casters=[int]

        for fidx,fdesc in enumerate(self.grid.cells.dtype.descr):
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
                    attrs.append( QgsField(fname,QVariant.Int) )
                    casters.append(int)
                elif np.issubdtype(ftype,np.float):
                    attrs.append( QgsField(fname,QVariant.Double) )
                    casters.append(float_or_null)
                else:
                    self.log.info("Not ready for other datatypes (%s: %s)"%(fname,str(ftype)))

        self.attrs=attrs
        self.casters=casters

        pr = qlayer.dataProvider()
        pr.addAttributes(attrs)
        qlayer.updateFields() # tell the vector layer to fetch changes from the provider

        if not existing:
            # transparent red, no border
            # but this is the wrong class...
            symbol = QgsFillSymbol.createSimple({'outline_style':'no',
                                                 'style':'solid',
                                                 'color': '249,0,0,78'})
            qlayer.renderer().setSymbol(symbol)
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
            self.log.info("CellLayer:on_modify_node, calling triggerRepaint")
            self.qlayer.triggerRepaint()

    def on_modify_cell(self,g,func_name,c,**k):
        fid=g.cells['feat_id'][c]
        attr_changes={fid:{}}

        for idx,attr in enumerate(self.attrs):
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
        pnts=[QgsPointXY(self.grid.nodes['x'][n,0],
                         self.grid.nodes['x'][n,1])
              for n in self.grid.cell_to_nodes(i)]
        return QgsGeometry.fromPolygonXY([pnts])

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
        feat.initAttributes(len(self.attrs))
        for idx,cattr in enumerate(self.attrs):
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

    def add_field(self,field,name,np_type):
        self.add_field_generic(field,name,np_type,
                               self.grid.add_cell_field,self.grid.Ncells())
            
    def update_values(self,feat_id,attr_id,value):
        """
        Copy QGIS changes in attribute values to the grid
        """
        self.update_values_generic(self.grid.cells,feat_id,attr_id,value)
            
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
    def create_qlayer(self,existing=None):
        if existing:
            qlayer=existing
        else:
            qlayer=QgsVectorLayer("Point"+self.crs,self.prefix+"-centers","memory")

        # for now, cell centers don't carry all of the fields like cell polygons do.
        self.attrs=[]
        self.casters=[]

        if not existing:
            symbol = QgsMarkerSymbol.createSimple({'outline_style':'no',
                                                   'name': 'circle', 
                                                   'size_unit':'MM',
                                                   'size':'1',
                                                   'color': 'green'})
            qlayer.renderer().setSymbol(symbol)
        return qlayer

    # extend, unextend, on_delete_cell: use inherited method of UmbraCellLayer
    # populate_qlayer(self), selection() also same as cell Layer

    def cell_geometry(self,i):
        cc=self.grid.cells_center()
        return QgsGeometry.fromPointXY( QgsPointXY(cc[i,0],cc[i,1]) )


class UmbraLayer(object):
    count=0

    layer_count=0
    crs="?crs=epsg:26910" # was 4326

    def __init__(self,umbra,grid,name=None,path=None,grid_format=None):
        """
        Does not add the layers to the GUI - call register_layers
        for that.
        """
        self.log=log

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
            name=umbra.generate_grid_name()
        self.name=name

        self.layers=[] # SubLayer objects associated with this grid.
        self.umbra.register_grid(self)

        self.undo_stack=QUndoStack()

    def update_savestate(self,**kws):
        """
        Called after saving the grid to a new file, to update the layer's information
        for future saves.
        """
        if 'grid_format' in kws:
            self.grid_format=kws['grid_format']
        if 'path' in kws:
            self.path=kws['path']
            for sl in self.layers:
                sl.update_title_abstract()

    def write_to_project(self,prj,scope,doc,tag):
        # had been scope here, but pretty sure it should be tag.
        if not tag.endswith('/'):
            tag=tag+'/'

        prj.writeEntry(scope,tag+'name',self.name)
        prj.writeEntry(scope,tag+'grid_format',str(self.grid_format))
        prj.writeEntry(scope,tag+'path',str(self.path))

    @classmethod
    def load_from_project(cls,umbra,prj,scope,tag):
        name,_=prj.readEntry(scope,tag+'name','')
        grid_format,_=prj.readEntry(scope,tag+'grid_format',"")
        path,_=prj.readEntry(scope,tag+'path',"")

        if '' in [name,grid_format,path]:
            self.log.error("Project file missing requisite data to load an umbra layer")
            return

        log.info('load_from_project: name=%s  grid_format=%s  path=%s'%(name,grid_format,path))

        new_layer=cls.open_layer(umbra=umbra,grid_format=grid_format,path=path,name=name)
        new_layer.register_layers(use_existing=True)
        return new_layer

    def connect_grid(self):
        """
        Install callbacks on the grid to update cell and edge quality fields.
        The individual layers will be later than these callbacks.
        """
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
            if ql==layer.qlayer:
                log.info("Match on ==, not name?!")
                return True
        return False

    @classmethod
    def open_layer(cls,umbra,grid_format,path,name=None):
        g=cls.load_grid(path=path,grid_format=grid_format)
        return cls(umbra=umbra,grid=g,path=path,grid_format=grid_format,name=name)

    @classmethod
    def load_grid(cls,grid_format=None,path=None):
        if path is None:
            # for development, load sample data:
            suntans_path=os.path.join( os.path.dirname(__file__),
                                       "sample_data/sfbay" )
            grid=unstructured_grid.UnstructuredGrid.read_suntans(suntans_path)
        else:
            if grid_format=='SUNTANS':
                grid=unstructured_grid.UnstructuredGrid.read_suntans(path)
            elif grid_format=='pickle':
                grid=unstructured_grid.UnstructuredGrid.from_pickle(path)
            elif grid_format=='DFM':
                grid=dfm_grid.DFMGrid(fn=path)
            elif grid_format=='UGRID':
                grid=unstructured_grid.UnstructuredGrid.from_ugrid(nc=path)
            elif grid_format=='UnTRIM':
                grid=unstructured_grid.UnTRIM08Grid(grd_fn=path)
            elif grid_format=='SHP':
                grid=unstructured_grid.UnstructuredGrid.from_shp(grd_fn=path)
            else:
                raise Exception("Need to add other grid types, like %s!"%grid_format)
        return grid

    def on_layer_deleted(self,sublayer):
        if sublayer in self.layers:
            # on shutdown, sublayer may not be in self.layers (?)
            self.layers.remove(sublayer)
        self.log.info("signal received: layer deleted")
        self.log.info("that leaves %d more sublayers for this grid"%len(self.layers))
        try:
            sublayer.unextend_grid()
        except AttributeError:
            self.log.error("on_layer_deleted got called with a weird object: %s"%sublayer)

        # not sure what the proper behavior should be when all of our layers 
        # have been removed.  The layer groups are unfortunately not very
        # robust, so it's a bit difficult to let the layer group play a proxy
        # role for the umbra layer.  Probably at the point that all the sublayers 
        # are removed, we should remove the umbra layer entirely.
        if len(self.layers)==0:
            self.unload_layer()

        # these are the steps which led to this callback.  don't do them
        # again.

        # QgsProject.instance().removeMapLayers([sublayer.qlayer])

    def unload_layer(self):
        """ Remove this layer.  Should only be called once the qgis layers
        have already been removed, and with current implementation, will be
        called automatically when that happens.
        """
        # remove the group
        self.log.info("All layers gone - will remove group '%s'"%self.name)
        # then drop this umbra_layer entirely
        self.remove_group()
        self.umbra.unregister_grid(self)    
        
    # create the memory layers and populate accordingly
    def register_layer(self,sublayer,preexisting=False):
        """
        Given a sublayer object, hook up some callbacks so this parent
        UmbraLayer can keep track, and add it to the group.
        preexisting: if true, assume the qlayer has already been added to
        the layer registry group.
        """
        self.layers.append( sublayer )

        self.log.info("registering destroyed callback with sublayer=%s"%sublayer)

        def callback(*args,sublayer=sublayer):
            self.log.info("delete on sublayer=%s"%sublayer)
            self.on_layer_deleted(sublayer)
        # this might not be the right thing -- this may only be getting called
        # when the object is destructored, but I want to know when it is removed
        # from the project.
        sublayer.qlayer.destroyed.connect(callback)

        if not preexisting:
            project=QgsProject.instance()
            # is layer==sublayer.qlayer?  dunno.
            new_layer=project.addMapLayer( sublayer.qlayer, addToLegend=False)
            node_layer=QgsLayerTreeLayer(new_layer)
            
            group=self.group() # assumes already created
            group.addChildNode(node_layer)

        sublayer.update_title_abstract()

    def layer_by_tag(self,tag):
        for layer in self.layers:
            # layer.tag not getting set.
            self.log.info("Looking for layer by tag: %s vs %s"%(layer.tag,tag))
            if layer.tag == tag:
                return layer
        self.log.info("Looking for layer by tag no hits")
        return None

    def create_group(self,use_existing=False):
        log.info("UmbraLayer:create_group, grp_name=%s use_existing=%s"%(self.group_name,use_existing))

        # The logic below is weak -- will need to decide whether to use a
        # different group name when use_existing is False and the group already
        # exists, or to just use the existing group.  
        root=QgsProject.instance().layerTreeRoot()

        if use_existing:
            group=root.findGroup(self.group_name)
            if group is None:
                self.log.warning("Failed to find group '%s', will create it"%self.group_name)
                use_existing=False

        if not use_existing:
            group=root.findGroup(self.group_name)
            if group is not None:
                self.log.warning("use_existing is false, but %s yields %s"%(self.group_name,group))
            else:
                # Create a group for the layers -
                group=root.addGroup(self.group_name)
                self.log.info("Created group '%s', value is %s"%
                              (self.group_name,group))

        assert root.findGroup(self.group_name) is not None
        
    def remove_group(self):
        root=QgsProject.instance().layerTreeRoot()
        grp=root.findGroup(self.group_name)
        if grp is not None:
            root.removeChildNode(grp)
        else:
            self.log.warning("Group '%s' not found to remove"%self.group_name)
    @property
    def group_name(self):
        return self.name
    def group(self):
        return QgsProject.instance().layerTreeRoot().findGroup(self.group_name)
    

    # tags are used to map between names and the sublayer 
    # class implementation:
    tag_map=dict(cells=UmbraCellLayer,
                 edges=UmbraEdgeLayer,
                 nodes=UmbraNodeLayer,
                 centers=UmbraCellCenterLayer)

    def register_layers(self,use_existing=False):
        """
        Add individual feature layers for this grid.
        use_existing: defaults to creating new layers.  If true,
         look for existing layers by name, and replace the data 
         in those layers rather than creating new layers. 
        """
        self.iface=self.umbra.iface
        self.create_group(use_existing=use_existing)

        self.log.info("register_layers(), use_existing=%s"%use_existing)
        
        group=self.group()
        # project=QgsProject.instance()
            
        if use_existing:
            # Try to scan the group to find which layers to instantiate
            # but if no layers are there, fall through and create
            # default set of layers.  alternatively, could trigger
            # our own removal, since there shouldn't be situations
            # where the group exists but has no layers
            group_layers=group.children()

            # Still, allow a fall through if no layers can be found
            use_existing=False
            for layer_tree_layer in group_layers:
                # Need to get qlayer from the layer_tree_layer
                qlayer=layer_tree_layer.layer()
                
                # should be <umbra_layer.name>-<tag>
                self.log.info("Checking on layer %s"%qlayer.name())
                try:
                    grid_name,tag = qlayer.name().split('-')
                except ValueError:
                    self.log.error("Found misnamed layer name '%s' in umbra."%
                                   (qlayer.name()) )
                    continue

                if tag not in self.tag_map:
                    self.log.error("Tag %s does not map to known grid layer class"%tag)
                    continue

                cls=self.tag_map[tag]

                sublayer=cls(self,prefix=self.name,qlayer=qlayer,tag=tag)
                self.register_layer(sublayer,preexisting=True)
                use_existing=True # got at least one layer, so prevent fall-through

        if not use_existing:
            for tag in ['nodes','edges','cells']:
                self.add_layer_by_tag(tag)

    def add_layer_by_tag(self,tag):
        cls=self.tag_map[tag]
        self.register_layer( cls(self,
                                 prefix=self.name,
                                 tag=tag) )

    def add_centers_layer(self):
        self.add_layer_by_tag('centers')

    def set_edge_quality_style(self):
        sl = self.layer_by_tag('edges')
        if sl is not None:
            sl.predefined_style('edge-quality')

    def set_cell_quality_style(self):
        sl=self.layer_by_tag('cells')
        self.log.info("Trying to set cell quality style, and sl is %s"%sl)
        if sl is not None:
            sl.predefined_style('cell-quality')

    def remove_all_qlayers(self):
        """
        Removes the QGIS layers for this Umbra layer.
        Currently, removing the last QGIS layer triggers
        removal/unregistering of the whole umbra layer.
        """
        layers=[]
        for sublayer in self.layers:
            try:
                layers.append( sublayer.qlayer.name() )
            except RuntimeError:
                # There is an issue with the destroyed callback, where it doesn't receive the
                # right layer, so it can't remove it from self.layers, and then at the
                # bitter end we try to remove it but it's already gone/deleted.
                self.log.error("qlayer was still in self.layers, but it was already deleted")
            self.log.info("Found %d layers to remove"%len(layers))
        project=QgsProject.instance().removeMapLayers(layers)

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
        # print "Finding closest node to ",xy
        return self.grid.select_nodes_nearest(xy)
    
    def find_closest_cell(self,xy):
        # xy: [x,y]
        return self.grid.select_cells_nearest(xy)

    def extent(self):
        xmin,xmax,ymin,ymax = self.grid.bounds()
        # print "extent() called on UmbraLayer"
        return QgsRectangle(xmin,ymin,xmax,ymax)
      
    def renumber(self):
        self.grid.renumber(reorient_edges=False)
        # Also recalculate edge->cells and similar
        self.grid.edge_to_cells(recalc=True)
        self.grid.update_cell_edges(select='all') # maybe less important.
        bare=self.grid.orient_edges(on_bare_edge='return')
        if len(bare):
            txt=" ".join([str(j) for j in bare])
            msg="Edges with no cell: %s"%txt
            if self.iface is not None:
                self.iface.messageBar().pushMessage("Info", msg, level=Qgis.Warning)
    
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
        self.log.info("orthogonalize_local: xy=%s,  iterations=%s"%(xy,iterations))
        
        tweaker=orthogonalize.Tweaker(self.grid)
        
        nl=self.layer_by_tag('nodes')
        if nl is not None:
            nl_sel=nl.selection()
            if len(nl_sel)>0:
                self.log.info("orthogonalize_local: node_selection=%s,  iterations=%s"%(nl_sel,iterations))
                for it in range(iterations):
                    self.log.info("orthogonalize_local: node_iteration %s"%it)
                    for n in nl_sel:
                        tweaker.nudge_node_orthogonal(n)
                return
        
        c=self.grid.select_cells_nearest(xy)
        if c is None:
            # this is maybe from stale indices?
            self.log.warning("orthogonalize_local: cell came up None")
            return 

        for it in range(iterations):
            self.log.info("orthogonalize_local: cell_iteration %s, c=%s"%(it,c))
            tweaker.nudge_cell_orthogonal(c)

    def smooth_local(self,node_idxs,min_halo=1,n_iter=1,stencil_radius=1,max_cells=75):
        def do_smooth(node_idxs=node_idxs,min_halo=min_halo,grid=self.grid,n_iter=n_iter,
                      stencil_radius=stencil_radius):
            tweaker=orthogonalize.Tweaker(grid)
            if len(node_idxs)==1:
                # Search to get more
                node_idxs,ij=grid.select_quad_subset(ctr=grid.nodes['x'][node_idxs[0]],
                                                     max_cells=max_cells)
            else:
                ij=None
            tweaker.local_smooth(node_idxs=node_idxs,ij=ij,min_halo=min_halo,n_iter=n_iter,
                                 stencil_radius=stencil_radius)
        cmd=GridCommand(self.grid,
                        "Smooth local",
                        do_smooth)
        self.undo_stack.push(cmd)
            
    def toggle_cell_at_point(self,xy):
        # this might be safer in capturing xy, and
        # helps with logging.
        self.log.info("umbra_layer: toggle cell at %s"%xy)
        
        def do_toggle(xy=xy):
            self.log.info("umbra_layer: do_toggle cell at %s"%xy)
            self.grid.toggle_cell_at_point(xy)
        cmd=GridCommand(self.grid,
                        "Toggle cell",
                        do_toggle)
        # self.grid.toggle_cell_at_point(xy))
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
            self.log.info("Edge already existed (nodes=%s), probably"%str(nodes))
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

    def split_edge(self,e,merge_thresh=-1):
        """
        e: list of edge indices
        merge_thresh: threshold for merging tri=>quad after split
        """
        def do_split_edges(e=e):
            all_js_next=[]
            for j in e:
                j_new,n_new,js_next=self.grid.split_edge(j,merge_thresh=merge_thresh)
                # update the edge selection to be the next obvious edge
                all_js_next += list(js_next)
                
            if len(all_js_next):
                self.log.debug("Will update edge selection with next edges to split")
                el=self.layer_by_tag('edges')
                if el is not None:
                    el.set_selection(all_js_next)
            
        cmd=GridCommand(self.grid,
                        "Split edge",
                        do_split_edges)
        self.undo_stack.push(cmd)

    def triangulate_hole(self,seed,**kwargs):
        # make sure it's legal:
        c=self.grid.select_cells_nearest(seed,inside=True)
        if c is not None:
            self.iface.messageBar().pushMessage("Error", "Point is inside an existing cell", level=Qgis.Warning,
                                                duration=3)
            
        def do_triangulate_hole(seed=seed):
            from stompy.grid import triangulate_hole
            gnew=triangulate_hole.triangulate_hole(self.grid,seed,**kwargs)

            if not kwargs['splice']:
                UmbraLayer(umbra=self.umbra,grid=gnew,path=None,
                           grid_format='UGRID', # default. maybe doesn't matter.
                           name=None)
                my_layer = UmbraLayer.open_layer(umbra=self.umbra,
                                                 grid_format='UGRID',# default?
                                                 path=path)
                my_layer.register_layers()
            
        cmd=GridCommand(self.grid,
                        "Triangulate hole",
                        do_triangulate_hole)
        self.iface.messageBar().pushMessage("Info","Starting triangulate_hole",level=Qgis.Info)
        self.undo_stack.push(cmd)
        
    def add_quad_from_edge(self,e,orthogonal='edge'):
        def do_extend_edge(e=e):
            try:
                res=self.grid.add_quad_from_edge(e,orthogonal=orthogonal)
            except self.grid.GridException as exc:
                self.log.warning("Could not extend edge %d"%e)
                return
            
            # update the edge selection to be the next obvious edge
            if 'j_next' in res:
                self.log.info("Will update edge selection with next edges to extend")
                el=self.layer_by_tag('edges')
                if el is not None:
                    el.set_selection([res['j_next']])
            
        cmd=GridCommand(self.grid,
                        "Extend quad from edge",
                        do_extend_edge)
        self.undo_stack.push(cmd)

    def merge_cells(self,e,chain=True):
        # allow chaining merge operations
        # e: list of edges to merge

        def do_merges(e=e,self=self,chain=chain):
            g=self.grid
            j_next=[]

            for j in e:
                jn=g.edges['nodes'][j].copy()

                c=g.merge_cells(j=j)
                if not chain: return

                # Check the two nodes:
                for n in jn:
                    ncs=g.node_to_cells(n)
                    nc_sides=[g.cell_Nsides(nc) for nc in ncs if nc!=c]

                    if nc_sides==[4,4]:
                        j=g.cells_to_edge(c,ncs[0])
                        assert j is not None
                        he=g.halfedge(j,0)
                        if he.cell()!=c: he=he.opposite()
                        # now c is on our left
                        if he.node_rev()!=n: he=he.fwd()
                        assert he.cell()==c
                        assert he.node_rev()==n
                        # delete the edge between the two quads
                        he_opp=he.opposite()
                        g.delete_edge_cascade(he_opp.fwd().j)
                        trav=he_opp
                        A=trav.node_rev() ; trav=trav.rev()
                        B=trav.node_rev() ; trav=trav.rev()
                        jC1=trav.opposite().fwd().j
                        C=trav.node_rev() ; trav=trav.rev()
                        jC2=trav.opposite().rev().j
                        D=trav.node_rev() ; trav=trav.rev()
                        E=trav.node_rev() ; trav=trav.rev()

                        g.add_edge(nodes=[A,C])
                        g.add_cell( nodes=[A,C,B] )
                        g.add_edge(nodes=[C,E])
                        g.add_cell( nodes=[C,E,D] )
                        g.merge_edges(node=n)
                        g.add_cell( nodes=[A,E,C] )
                        # and find a potential edge to merge next
                        if jC1==jC2:
                            j_next.append(jC1)
                    elif nc_sides==[3,3,3]:
                        seed=g.cells_center()[ncs[0]]
                        for jnbr in list(g.node_to_edges(n)):
                            print("Checking j=%d e2c %s against %d"%(jnbr,g.edge_to_cells(jnbr),c))
                            if c not in g.edge_to_cells(jnbr):
                                # This is finding a cell that is already deleted?
                                g.delete_edge_cascade(jnbr) 
                        n_nbrs=g.node_to_nodes(n)
                        assert len(n_nbrs)==2,"n_nbrs should be two, but it's %s"%( str(n_nbrs) ) # fails HERE
                        g.merge_edges(node=n)
                        g.add_cell_at_point(seed)
                    else:
                        pass

                if j_next:
                    self.log.info("Will update edge selection with next edges to merge")
                    el=self.layer_by_tag('edges')
                    if el is not None:
                        el.set_selection(j_next)
        
        cmd=GridCommand(self.grid,
                        "Merge cells",
                        do_merges)
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
