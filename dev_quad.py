"""
Develop a 1-quad expansion tool.

"""
from stompy.grid import unstructured_grid

g=unstructured_grid.UnstructuredGrid.from_ugrid('/home/rusty/src/sfb_ocean/suntans/grid-merged/suisun_main-v05-edit11.nc')

##
zoom=(577526.0330947199, 579209.116101901, 4211084.380445508, 4212246.5484448485)
map_xy=np.array([578239, 4211660])
#map_xy=np.array([578443., 4211697.])
j=g.select_edges_nearest(map_xy)



plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g.plot_cells(ax=ax,edgecolor='k',lw=0.4)

g.plot_edges(mask=[j],color='y',ax=ax,lw=3)

ax.axis('equal')
ax.axis(zoom)

##
self=g

def quad_step(self,j):
    nodes=self.edges['nodes'][j]
    cells=self.edge_to_cells(j)
    if (cells<0).sum()!=1:
        raise self.GridException("Must have exactly one side edge unpaved")

    he=g.halfedge(j,0)
    if he.cell()>=0:
        he=he.opposite()
    assert he.cell()<0

    he_fwd=he.fwd()
    he_rev=he.rev()

    abcd=[he_rev.node_rev(),
          he.node_rev(),
          he.node_fwd(),
          he_fwd.node_fwd()]

    pnts=self.nodes['x'][abcd]
    dpnts=np.diff(pnts,axis=0)
    angles=np.arctan2(dpnts[:,1],dpnts[:,0])
    int_angles=np.diff(angles)

    # In creating the new point, the most obvious choices are to
    # make a trapezoid and the remaining faces are symmetric.
    # there is a choice of which edges to make parallel.  The 
    # more common usage is probably adding a row along the
    # length of a channel, so the selected edge is one of the symmetric
    # edges, not the parallel edge

    j_next=None

    min_int_angle=60*np.pi/180.
    if (int_angles[0]>int_angles[1]) and (int_angles[0]>min_int_angle):
        # quad will be a,b,c,N
        # calculate new 'd'
        # start with symmetric trapezoid
        new_x_d=pnts[2]+pnts[0]-pnts[1] # parallelogram
        para=utils.to_unit(pnts[0]-pnts[1])
        new_x_d-= 2 * para*np.dot(para,pnts[2]-pnts[1])
        new_x=new_x_d
        new_n=self.add_node(x=new_x)
        self.add_edge(nodes=[new_n,abcd[2]])
        j_next=self.add_edge(nodes=[new_n,abcd[0]])
        self.add_cell(nodes=[abcd[0],abcd[1],abcd[2],new_n])

    elif (int_angles[1]>int_angles[0]) and (int_angles[1]>min_int_angle):
        # quad will be N,b,c,d
        # calculate new 'a'
        new_x_a=pnts[1]+pnts[3]-pnts[2] # parallelogram
        para=utils.to_unit(pnts[3]-pnts[2])
        new_x_a-=2 * para*np.dot(para,pnts[1]-pnts[2])
        new_x=new_x_a
        new_n=self.add_node(x=new_x)
        self.add_edge(nodes=[new_n,abcd[1]])
        j_next=self.add_edge(nodes=[new_n,abcd[3]])
        self.add_cell(nodes=[new_n,abcd[1],abcd[2],abcd[3]])

    return dict(j_next=j_next)

##
j=j_next
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g.plot_cells(ax=ax,edgecolor='k',lw=0.4)

g.plot_edges(mask=[j],color='y',ax=ax,lw=3)

ax.axis('equal')
ax.axis( (577865.8396766097, 582306.0895312515, 4210916.932012921, 4213982.92238186) )
