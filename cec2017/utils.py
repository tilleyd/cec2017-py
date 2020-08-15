# cec2017.utils
# Author: Duncan Tilley
# Additional functions for graphing and benchmarking

def surface_plot(function, domain=(-100,100), points=30, dimension=2, ax=None):
    """
    Creates a surface plot of a function.

    Args:
        function (function): The objective function to be called at each point.
        domain (num, num): The inclusive (min, max) domain for each dimension.
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed.
        dimension (int): The dimension to pass to the function. If this is more
            than 2, the elements after the first 2 will simply be zero,
            providing a slice at x_3 = 0, ..., x_n = 0.
        ax (matplotlib axes): Optional axes to use (must have projection='3d').
            Note, if specified plt.show() will not be called.
    """
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np

    # create points^2 tuples of (x,y) and populate z
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])
    zs = np.zeros(points*points)

    if dimension > 2:
        # concatenate remaining zeros
        tail = np.zeros(dimension - 2)
        for i in range(0, xys.shape[0]):
            zs[i] = function(np.concatenate([xys[i], tail]))
    else:
        for i in range(0, xys.shape[0]):
            zs[i] = function(xys[i])

    # create the plot
    ax_in = ax
    if ax is None:
        ax = plt.axes(projection='3d')

    X = xys[:,0].reshape((points, points))
    Y = xys[:,1].reshape((points, points))
    Z = zs.reshape((points, points))
    ax.plot_surface(X, Y, Z, cmap='gist_ncar', edgecolor='none')
    ax.set_title(function.__name__)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if ax_in is None:
        plt.show()

def time(function, domain=(-100,100), points=30):
    """
    Returns the time in seconds to calculate points^2 evaluations of the
    given function.

    function
        The objective function to be called at each point.
    domain
        The inclusive (min, max) domain for each dimension.
    points
        The number of points to collect on each dimension. A total of points^2
        function evaluations will be performed.
    """
    from time import time
    import numpy as np

    # create points^2 tuples of (x,y) and populate z
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])
    zs = np.zeros(points*points)

    before = time()
    for i in range(0, xys.shape[0]):
        zs[i] = function(xys[i])
    return time() - before
