"""
===============================================
2.2 Reproducing kernels and transformation maps
===============================================

In this experiment,
"""

# Importing necessary modules
import os
import sys

curr_f = os.path.join(os.getcwd(), "codpy-book", "utils")
sys.path.insert(0, curr_f)


import matplotlib.pyplot as plt
import numpy as np

# import CodPy's core module and Kernel class
from codpy import core
from codpy.kernel import Kernel

# from codpy.plotting import plot1D
# Lets import multi_plot function from codpy utils
from codpy.plot_utils import multi_plot, plot1D


def kernel_fun(x=None, kernel_name=None, D=1):
    if x is None:
        x = np.linspace(-3, 3, num=100)
    y = np.zeros([1, D])
    kernel = Kernel(
        set_kernel=core.kernel_setter(kernel_name, None),
        x=x,
        order=1,
    )
    out = kernel.knm(x, y)
    return out


def kernel_funs_plot():
    kernel_list = [
        "gaussian",
        "tensornorm",
        "absnorm",
        "matern",
        "multiquadricnorm",
        # "multiquadrictensor",
        "sincardtensor",
        # "sincardsquaretensor",
        # "dotproduct",
        "gaussianper",
        "maternnorm",
        "scalarproduct",
    ]

    # Prepare the results for plotting each kernel
    results = [
        (np.linspace(-3, 3, num=100), kernel_fun(kernel_name=kernel_name).flatten())
        for kernel_name in kernel_list
    ]

    # Legends for each kernel in the plot
    legends = kernel_list

    # Plot all kernels using multi_plot in a 4x4 grid
    multi_plot(
        results,
        plot1D,
        f_names=legends,
        mp_nrows=3,
        mp_ncols=3,
        mp_figsize=(16, 12),
    )


# Run the experiment with CodPy and SciPy models
# core.KerInterface.set_verbose()
kernel_funs_plot()
plt.show()

######################################################
# **Kernel Gram matrix**
# **Positive definite kernels and kernel matrices**. A *kernel*, denoted by $k: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$, is a symmetric real-valued function, that is, satisfying $k(x, y)=k(y, x)$. Given two collections of points in $\mathbb{R}^D$, namely $X = (x^1, \cdots, x^{N_x})$ and $Y = (y^1, \cdots, y^{N_y})$, we define the associated *kernel matrix* $K(X,Y) = \big(k(x^n,y^m) \big) \in \mathbb{R}^{N_x, N_y}$ by
# $$K(X, Y) =\left( \begin{array}{ccc} k(x^1,y^1) & \cdots & k(x^1,y^{N_y}) \\ \ddots & \ddots & \ddots \\ k(x^{N_x},y^1) & \cdots & k(x^{N_x},y^{N_y}) \end{array}\right).$$
#
import pandas as pd

x = np.random.randn(10, 1)
kernel = Kernel(
    set_kernel=core.kernel_setter("gaussian", None),
    x=x,
    order=1,
)

# Kernel Gram matrix
print(pd.DataFrame(kernel.knm(x, x)))

#########################################################################
# **MMD Matrix**
#
# MMD matrices provide a very useful tool in order to evaluate the accuracy of a computation. To any positive kernel $k : \mathbb{R}^D, \mathbb{R}^D \mapsto \mathbb{R}$, we associate the *discrepancy function* $d_k(x,y)$ defined (for $x,y\in\mathbb{R}^D$) by
# $$d_k(x,y) = k(x,x) + k(y,y) - 2k(x,y)$$.
# For positive kernels, $d_k(\cdot,\cdot)$ is continuous, non-negative, and satisfies the condition $d_k(x,x) = 0$ (for all relevant $x$)


print(pd.DataFrame(kernel.kernel_distance(x)))


#########################################################################
# Kernels 2D visualisation


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


def generate2Ddata(sizes_x):
    data_x = np.random.uniform(-1, 1, (sizes_x, 2))

    kernel_list = [
        "gaussian",
        "tensornorm",
        "absnorm",
        "matern",
        "multiquadricnorm",
        # "multiquadrictensor",
        "sincardtensor",
        # "sincardsquaretensor",
        # "dotproduct",
        "gaussianper",
        "maternnorm",
        "scalarproduct",
    ]

    # Prepare the results for plotting each kernel
    results = [
        (data_x, kernel_fun(data_x, kernel_name, D=2).flatten())
        for kernel_name in kernel_list
    ]

    # Legends for each kernel in the plot
    legends = kernel_list

    # Plot all kernels using multi_plot in a 4x4 grid
    multi_plot(
        results,
        plot_trisurf,
        f_names=legends,
        mp_nrows=3,
        mp_ncols=3,
        mp_figsize=(12, 16),
        elev=30,
        projection="3d",
    )


generate2Ddata(400)
plt.show()
pass