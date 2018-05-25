import numpy as np
import matplotlib.pyplot as plt

from stompy.grid import unstructured_grid

##

g=unstructured_grid.UnstructuredGrid(max_sides=4)

g.add_rectilinear( [0,0],[100,200],11,21 )

x0=g.nodes['x'].copy()

##

def apply_map(f):
    c0=x0[:,0]+1j*x0[:,1]

    c_prime=f(c0)
    
    g.nodes['x'][:,0]=np.real(c_prime)
    g.nodes['x'][:,1]=np.imag(c_prime)


# 90 degree bend
k=np.pi/200.0
# apply_map(lambda x: np.exp(k*x))

# How many terms are needed for a taylor expansion of exp(x)?
def f_map(x):

    t0=50+50j # center point for the expansion
    scal0=100
    f0= (t0
         + scal0 * (x-t0)*k
         + scal0 * 1/2. * (x-t0)**2*k**2
         + scal0 * 1/6. * (x-t0)**3*k**3
         + scal0 * 1/24. * (x-t0)**4*k**4 )
    
    f1= 1.85 * (0.81+0.5j)*x + ( 40 - 20j)
    
    # transition:
    # select=np.imag(x)<100
    # return np.where(select,f0,f1)
    width=60.
    center=100

    # HERE: Linear interpolation of the transformation loses the conformal
    # properties.  Need a smarter way to do this, maybe local SC mapping?
    alpha=((np.imag(x) - (center-width/2.))/width).clip(0,1)
    return (1-alpha)*f0 + alpha*f1
    

# 4th order is where it starts to look very nice, for a 90 degree bend.
apply_map( f_map )

    
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g.plot_edges(ax=ax)

ax.axis('equal')

##

# How would it work to interpolate between local transforms like that?
# The original space is just a long rectangular strip, x0.
# For a given point in rectangular space
