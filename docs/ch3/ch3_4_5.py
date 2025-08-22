"""
=======================================================
3.4.5  Integral operator - inverse divergence operator
=======================================================


This section describes the Leray operator and its role in the
Helmholtz–Hodge decomposition, which is foundational in the analysis of
incompressible fluid flows, turbulence, and vector field analysis.

Overview
--------

The **Helmholtz–Hodge decomposition** expresses any vector field
as the orthogonal sum of a **gradient component** and a
**divergence-free component**:

$$
v = \\nabla h + \\zeta,
\\quad \\nabla \\cdot \\zeta = 0,
\\quad h = \\Delta^{-1} \\nabla \\cdot v
$$

In the context of kernel methods, CodPy provides a numerical
approximation of this decomposition via the **Leray operator**.

Leray Operator Definition
-------------------------

The Leray operator is defined by subtracting the Leray-orthogonal projection
from the identity:

$$
L_k(\cdot) = I_d - L_k(\cdot)^{\\perp}
= I_d - \\nabla_k(\cdot)
\\cdot \\Delta_k(\cdot)^{-1}
\\cdot \\nabla_k(\cdot)^T
$$

This operator projects any field onto the **divergence-free** subspace.
For any vector field $v_z \\in \\mathbb{R}^{D \\times N_z \\times D_v}$,
we obtain the orthogonal decomposition:

$$
v_z = L_k(Z) v_z + L_k(Z)^{\\perp} v_z
$$

with the orthogonality condition:

$$
\\left< L_k v_z, L_k^{\\perp} v_z \\right>_{D, N_z, D_v} = 0
$$

Numerical Helmholtz–Hodge Decomposition
---------------------------------------

Using the Leray operator, we can numerically approximate the decomposition:

$$
v_z = \\nabla_k(Z) h_x + \\zeta_z
$$

where:
    - $h_x = \\nabla_k^{-1}(X, Y, Z) v_z$ is the scalar potential,
    - $\\zeta_z = L_k(Z) v_z$ is the divergence-free component.
    - $\\nabla_k(\\cdot) = (\\nabla K)(\\cdot,X)( K(X,X) + \\epsilon R(X,X) )^{-1}$

This decomposition satisfies the orthogonality relations:

$$
\\nabla_k(Z)^T \\zeta_z = 0,
\\quad
\\left< \\zeta_z, \\nabla_k h_x \\right>_{D, N_z, D_f} = 0
$$

These conditions mirror the classical Hodge decomposition and provide
a powerful framework for the simulation and analysis of fluid flows.

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
# CodPy Implementation using gradient operator
#
# We use TensorNorm kernel function defined as:
#
#
# $$
# k(x, y) = \prod_{d} \max(1 - \|x_d - y_d\|, 0)
# $$
#
# and the unit cube map $S$:
#
# $$
# S(X) = \frac{x - \min_n{x^n} + \frac{0.5}{N_x}}{\alpha},
# \quad \alpha = \max_n{x^n} - \min_n{x^n}
# $$
#
# To compute the gradient of a function $f(x)$ numerically using CodPy, we need:
# to import CodPy's core module and Kernel class and initialize kernel pointer.
#
#########################################################################


def fun_Integral(size_x=50, size_y=50):
    x, fx, z, fz, _, nabla_fz = generate_periodic_data_cartesian(
        size_x, size_y, periodic_fun, nabla_fun=nabla_my_fun
    )

    nabla_fz = nabla_fz.reshape(-1, 2, 1)

    # kernel_ptr = Kernel(
    #     x=x, fx=fx, set_kernel=core.kernel_setter("gaussianper", None,0,1e-9)
    # ).get_kernel()

    kernel_ptr = Kernel(
        x=x, fx=fx, 
    ).get_kernel()
    temp = core.DiffOps.nabla_t(
        x=x,
        y=x,
        z=x,
        fz=core.DiffOps.nabla_t_inv(x=x, y=x, z=x, fx=fx, kernel_ptr=kernel_ptr),
        kernel_ptr=kernel_ptr,
        order=0,
        regularization=1e-8,
    )
    multi_plot(
        [(x, fx), (x, temp)],
        plot_trisurf,
        projection="3d",
        elev=30,
        mp_nrows=1,
        mp_figsize=(12, 3),
        mp_title=[
            "Comparison between original function to the product of the inverse of the gradient operator and the gradient operator"
        ],
    )
    plt.show()


fun_Integral()


def fun_nabla_inv(size_x=50, size_y=50):
    x, fx, z, fz, _, nabla_fz = generate_periodic_data_cartesian(
        size_x, size_y, periodic_fun, nabla_fun=nabla_my_fun
    )

    nabla_fz = nabla_fz.reshape(-1, 2, 1)

    kernel_ptr = Kernel(
        x=x, fx=fx, set_kernel=core.kernel_setter("gaussianper", None,0, 1e-8)
    ).get_kernel()

    nabla_f_x = core.DiffOps.nabla(
        x=x,
        z=x,
        fx=fx,
        kernel_ptr=kernel_ptr,
        order=0,
        regularization=1e-8,
    )
    fz_inv = core.DiffOps.nabla_inv(
        x, x, x, fx=nabla_f_x.squeeze(), kernel_ptr=kernel_ptr
    )
    multi_plot(
        [(x, fx), (x, fz_inv)],
        plot_trisurf,
        projection="3d",
        mp_nrows=1,
        mp_figsize=(12, 3),
        mp_title=[
            "Comparison between the product of the divergence operator and its inverse and the product of Laplace operator and its inverse"
        ],
    )
    plt.show()


fun_nabla_inv()
pass
