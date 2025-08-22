"""
=============================================
3.4.3  Inverse Laplace operator
=============================================

The inverse Laplace operator is a useful tool in many fluid mechanics,
image processing, signal filtering etc. In the CodPy framework, this operator is implemented
as the pseudo-inverse of the Laplacian matrix built from a kernel.

Overview
--------

The Laplacian operator $\\Delta_k$ is defined in CodPy via kernel methods
as a matrix $\\Delta_k(X, Y) \\in \\mathbb{R}^{N_x \\times N_x}$ constructed
from a set of input points $X$ (and optionally $Y$). It generalizes the
classical Laplacian to the setting of RKHS.

The **inverse Laplacian** is simply the matrix inverse (or pseudo-inverse)
of this operator:

$$
\\Delta_k^{-1}(X, Y) =
\\left( \\Delta_k(X, Y) \\right)^{-1}
\\in \\mathbb{R}^{N_x \\times N_x}
$$

This operator provides a way to "undo" the effect of the Laplacian on a
function. It is particularly useful when solving elliptic partial differential
equations.

CodPy Implementation
--------------------
"""

# Importing necessary modules
import os
import sys

from matplotlib import pyplot as plt

curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)


import numpy as np

# from codpy.plotting import plot1D
# Lets import multi_plot function from codpy utils
from codpy.plot_utils import multi_plot


# Define the sinusoidal function
def periodic_fun(x):
    """
    A sinusoidal function that generates a sum of sines based on the input ``x``.
    """
    from math import pi

    sinss = np.cos(2 * x * pi)
    if x.ndim == 1:
        sinss = np.prod(sinss, axis=0)
        ress = np.sum(x, axis=0)
    else:
        sinss = np.prod(sinss, axis=1)
        ress = np.sum(x, axis=1)
    return ress + sinss


def nabla_my_fun(x):
    from math import pi

    import numpy as np

    sinss = np.cos(2 * x * pi)
    if x.ndim == 1:
        sinss = np.prod(sinss, axis=0)
        D = len(x)
        out = np.ones((D))

        def helper(d):
            out[d] += 2.0 * sinss * pi * np.sin(2 * x[d] * pi) / np.cos(2 * x[d] * pi)

        [helper(d) for d in range(0, D)]
    else:
        sinss = np.prod(sinss, axis=1)
        N = x.shape[0]
        D = x.shape[1]
        out = np.ones((N, D))

        def helper(d):
            out[:, d] += (
                2.0 * sinss * pi * np.sin(2 * x[:, d] * pi) / np.cos(2 * x[:, d] * pi)
            )

        [helper(d) for d in range(0, D)]
    return out


# Function to generate periodic data
def generate_periodic_data_cartesian(size_x, size_z, fun=None, nabla_fun=None):
    """
    Generates 2D structured Cartesian grid data for x and z domains,
    and evaluates a given function and optionally its gradient.

    Parameters:
    - size_x: number of points per axis for x (grid will be size_x^2)
    - size_z: number of points per axis for z (grid will be size_z^2)
    - fun: function to evaluate at each point
    - nabla_fun: optional gradient function to evaluate

    Returns:
    - x, z: 2D Cartesian grids of shape (N, 2)
    - fx, fz: function values at x and z
    - nabla_fx, nabla_fz (if nabla_fun is provided)
    """

    def cartesian_grid(size, box):
        lin = [np.linspace(box[0, d], box[1, d], size) for d in range(2)]
        X, Y = np.meshgrid(*lin)
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    # Define domain boxes
    X_box = np.array([[-1, -1], [1, 1]])
    Z_box = np.array([[-1.5, -1.5], [1.5, 1.5]])

    # Generate Cartesian grids
    x = cartesian_grid(size_x, X_box)
    z = cartesian_grid(size_z, Z_box)

    # Function evaluations
    fx = fun(x).reshape(-1, 1) if fun else None
    fz = fun(z).reshape(-1, 1) if fun else None

    if nabla_fun:
        nabla_fx = nabla_fun(x)
        nabla_fz = nabla_fun(z)
        return x, fx, z, fz, nabla_fx, nabla_fz

    return x, fx, z, fz


# Lets define helper function to plot 3D projection of the function
def plot_trisurf(xfx, ax, legend="", elev=90, azim=-100, **kwargs):
    from matplotlib import cm

    """
    Helper function to plot a 3D surface using a trisurf plot.

    Parameters:
    - xfx: A tuple containing the x-coordinates (2D points) and their 
      corresponding function values.
    - ax: The matplotlib axis object for plotting.
    - legend: The legend/title for the plot.
    - elev, azim: Elevation and azimuth angles for the 3D view.
    - kwargs: Additional keyword arguments for further customization.
    """

    xp, fxp = xfx[0], xfx[1]
    x, fx = xp, fxp

    X, Y = x[:, 0], x[:, 1]
    Z = fx.flatten()
    ax.plot_trisurf(X, Y, Z, antialiased=False, cmap=cm.jet)
    ax.view_init(azim=azim, elev=elev)
    ax.title.set_text(legend)


# import CodPy's core module and Kernel class
from codpy import core
from codpy.kernel import Kernel

#########################################################################
# Inverse Laplace operator
# In the experiments below, we will compare the original function with the
# result of applying the inverse Laplace operator to the function and then
# applying the Laplace operator again. This should yield the original function
# back, demonstrating the operator's effectiveness:
# $$
# \\Delta_k^{-1}(\\Delta_k(f)) = f
# $$
#
#########################################################################


def fun_Delta1(size_x=50, size_y=50):
    x, fx, z, fz, _, nabla_fz = generate_periodic_data_cartesian(
        size_x, size_y, periodic_fun, nabla_fun=nabla_my_fun
    )

    nabla_fz = nabla_fz.reshape(-1, 2, 1)

    kernel_ptr = Kernel(
        x=x, fx=fx, set_kernel=core.kernel_setter("tensornorm", "scale_to_unitcube")
    ).get_kernel()

    temp = core.DiffOps.nabla_t_nabla_inv(
        x=x,
        y=x,
        fx=core.DiffOps.nabla_t_nabla(x=x, y=x, fx=fx, kernel_ptr=kernel_ptr),
        kernel_ptr=kernel_ptr,
        order=2,
        regularization=1e-8,
    )
    multi_plot(
        [(x, fx), (x, temp)],
        plot_trisurf,
        projection="3d",
        mp_nrows=1,
        mp_figsize=(12, 3),
        mp_title=[
            "Comparison between original function to the product of Laplace and its inverse"
        ],
    )
    plt.show()


fun_Delta1()


#########################################################################
# Inverse Laplace operator
# In the experiments below, we will compare the original function with the
# result of applying the inverse Laplace operator to the function and then
# applying the Laplace operator again. This should yield the original function
# back, demonstrating the operator's effectiveness:
# $$
# \\Delta_k(\\Delta_k^{-1}(f)) = f
# $$
#
# Since the Laplace operator is associative, they should be equal.
#
#########################################################################


def fun_Delta2(size_x=50, size_y=50):
    x, fx, z, fz, _, nabla_fz = generate_periodic_data_cartesian(
        size_x, size_y, periodic_fun, nabla_fun=nabla_my_fun
    )

    nabla_fz = nabla_fz.reshape(-1, 2, 1)

    kernel_ptr = Kernel(
        x=x, fx=fx, set_kernel=core.kernel_setter("tensornorm", "scale_to_unitcube")
    ).get_kernel()

    temp = core.DiffOps.nabla_t_nabla_inv(
        x=x, y=x, fx=fx, kernel_ptr=kernel_ptr, order=2, regularization=1e-8
    )
    temp = core.DiffOps.nabla_t_nabla(
        x=x, y=x, fx=temp, kernel_ptr=kernel_ptr, order=2, regularization=1e-8
    )
    multi_plot(
        [(x, fx), (x, temp)],
        plot_trisurf,
        projection="3d",
        mp_nrows=1,
        mp_figsize=(12, 3),
        mp_title=[
            "Comparison between original function and the product of the inverse of the Laplace operator and the Laplace operator"
        ],
    )
    plt.show()


fun_Delta2()
