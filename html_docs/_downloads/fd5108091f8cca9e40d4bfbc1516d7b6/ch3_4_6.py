"""
3.4.6 Leray-orthogonal operator
===============================

The Leray-orthogonal operator plays a fundamental role in fluid dynamics,
particularly in the mathematical formulation and numerical modeling of
incompressible flows governed by the Euler or Navier–Stokes equations.

Overview
--------

The Leray projection is used to decompose a vector field into divergence-free
(incompressible) and curl-free components. In CodPy, this is achieved using
kernel-based differential operators.

We define the **Leray-orthogonal operator** as:

$$
L_k(X, Y)^{\\perp} =
\\nabla_k(X, Y) \\cdot
\\Delta_k(X, Y)^{-1} \\cdot
\\nabla_k^T(X, Y)
$$

or alternatively:

$$
L_k(\cdot)^{\\perp} =
\\nabla_k(\cdot) \\cdot
\\nabla_k^{-1}(\cdot)
$$

This operator projects any vector field onto the space orthogonal to the
gradient field—i.e., the divergence-free subspace.

Action on a Vector Field
------------------------

Given a vector field $f(Z) \\in \\mathbb{R}^{D \\times N_z \\times D_f}$,
the Leray-orthogonal operator acts as:

$$
L_k(Z)^{\\perp} f(Z)
\\in \\mathbb{R}^{D \\times N_z \\times D_f}
$$

That is, it returns the projection of $f(Z)$ onto the orthogonal complement
of the gradient image in RKHS, which corresponds to the divergence-free component.

Use Case in Fluid Dynamics
--------------------------

This operator enables **Helmholtz–Hodge decomposition** of a vector field
into divergence-free and gradient parts:

$$
f(Z) = f^{\\perp}(Z) + \\nabla_k h(Z)
$$

where $f^{\\perp} = L_k^{\\perp} f$ is divergence-free and $\\nabla_k h$
is the gradient component.

CodPy Implementation
--------------------

To compute the Leray projection operator in CodPy:
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
# Leray operator and Helmholtz-Hodge decomposition
#
#########################################################################


def fun_LerayT(size_x=50, size_y=50):
    x, fx, z, fz, _, nabla_fz = generate_periodic_data_cartesian(
        size_x, size_y, periodic_fun, nabla_fun=nabla_my_fun
    )

    nabla_fz = nabla_fz.reshape(-1, 2, 1)

    kernel_ptr = Kernel(
        x=x, fx=fx, 
    ).get_kernel()
    LerayT_fz = core.DiffOps.nabla(
        x=x,
        y=x,
        z=z,
        fx=core.DiffOps.nabla_inv(x=x, y=x, z=x, fz=nabla_fz, kernel_ptr=kernel_ptr),
        kernel_ptr=kernel_ptr,
        order=2,
        regularization=1e-8,
    )
    multi_plot(
        [
            (z, nabla_fz[:, 0, :]),
            (z, LerayT_fz[:, 0, :]),
            (z, nabla_fz[:, 1, :]),
            (z, LerayT_fz[:, 1, :]),
        ],
        plot_trisurf,
        projection="3d",
        mp_nrows=1,
        mp_figsize=(12, 3),
        mp_title=[
            "Comparing f(z) and the transpose of the Leray operator on each direction"
        ],
    )
    plt.show()


fun_LerayT()


def fun_LerayT_nabla_nablainv(size_x=50, size_y=50):
    x, fx, z, fz, _, nabla_fz = generate_periodic_data_cartesian(
        size_x, size_y, periodic_fun, nabla_fun=nabla_my_fun
    )

    kernel_ptr = Kernel(
        x=x, fx=fx, set_kernel=core.kernel_setter("tensornorm", "scale_to_unitcube")
    ).get_kernel()
    nabla_fx = core.DiffOps.nabla(
        x=x,
        y=x,
        z=z,
        fx=fx,
        kernel_ptr=kernel_ptr,
        order=2,
        regularization=1e-8,
    )
    nabla_fz = core.DiffOps.leray_t(x=x, y=x, fx=nabla_fx, kernel_ptr=kernel_ptr)

    LerayT_fz = core.DiffOps.nabla(
        x=x,
        y=x,
        z=z,
        fx=core.DiffOps.nabla_inv(x=x, y=x, z=z, fz=nabla_fz, kernel_ptr=kernel_ptr),
        kernel_ptr=kernel_ptr,
        order=2,
        regularization=1e-8,
    )
    multi_plot(
        [
            (z, nabla_fz[:, 0, :]),
            (z, LerayT_fz[:, 0, :]),
            (z, nabla_fz[:, 1, :]),
            (z, LerayT_fz[:, 1, :]),
        ],
        plot_trisurf,
        projection="3d",
        mp_nrows=1,
        mp_figsize=(12, 3),
        mp_title=[
            "Comparing $\nabla \nabla^{-1}f(z)$ and the transpose of the Leray operator $L_k \nabla f(z)$ on each direction"
        ],
    )
    plt.show()


fun_LerayT_nabla_nablainv()
