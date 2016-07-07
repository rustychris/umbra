import unstructured_grid
from delft import dfm_grid
reload(unstructured_grid)
reload(dfm_grid)
import qnc
## 

nc_path="/Users/rusty/models/sfbd-grid-southbay/SFEI_SSFB_fo_dwrbathy_net.nc"
# nc=qnc.QDataset(nc_path)
g=dfm_grid.DFMGrid(nc_path)
g.write_ugrid('test_ugrid.nc',overwrite=True)

## 

# g.plot_edges()

# ugrid writing:


## 

nc=qnc.QDataset('test_ugrid.nc')
g=unstructured_grid.UnstructuredGrid.from_ugrid(nc)

## 

# For the moment, not sure how much functionality will be put
# in UnstructuredGrid, a Paving-like subclass of it, outside
# classes, or outside methods.
# easiest to work with some discrete methods for now.

def set_marks_on_full_grid(self):
    """
    getting edge marks set up for grid generation
    uses cells to define the half-edge marks
    """
    # make sure that edges['cells'] is set
    e2c=self.edge_to_cells()
    g.edges['mark'][:]=self.INTERNAL
    g.edges['mark'][e2c[:,1]<0] = self.LAND


set_marks_on_full_grid(g)    

## 

zoom0=(580841., 587152., 4143806, 4148726)
zoom1=(583471., 584479, 4146240, 4147026)
sel_nodes=g.select_nodes_intersecting(xxyy=zoom1)

## 

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

coll=g.plot_edges(values=g.edges['mark'])
colln=g.plot_nodes(mask=sel_nodes)

ax.axis(zoom0)

## 

# screwed up - this should be starting with construct_quads_v03.py,
# and on to ...
# for both the next step of quad generation, and for any
# sort of port of tom to unstructured_grid, need support
# for nodes which are constrained to a curve.

# class NodeConstraints(object):

