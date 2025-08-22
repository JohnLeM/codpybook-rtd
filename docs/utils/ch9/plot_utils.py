import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from codpy.data_conversion import get_date, get_float
from codpy.kernel import Kernel, Sampler
from codpy.plot_utils import multi_plot, multi_plot_figs
from codpy.utils import get_matrix

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

PARENT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.ch9.data_utils import get_datetime

my_len_switchDict = {list: lambda x: len(x),
                    pd.core.indexes.base.Index  : lambda x: len(x),
                    np.array : lambda x: len(x),
                    np.ndarray : lambda x: x.size,
                    pd.DataFrame  : lambda x: my_len(x.values),
                    pd.core.groupby.generic.DataFrameGroupBy : lambda x: x.ngroups
                    }

primitive = (int, str, bool,np.float32,np.float64,float)
def is_primitive(thing):
    debug = type(thing)
    return isinstance(thing, primitive)

def my_len(x):
    if is_primitive(x):return 0
    type_ = type(x)
    method = my_len_switchDict.get(type_,lambda x : 0)
    return method(x)

def lexicographical_permutation(x,fx=[],**kwargs):
    # x = get_data(x)
    if x.ndim==1: index_array = np.argsort(x)
    else:
        indexfx = kwargs.get("indexfx",0)
        index_array = np.argsort(a = x[:,indexfx])
    x_sorted = x[index_array]
    if (my_len(fx) != my_len(x_sorted)): return (x_sorted,index_array)
    if type(fx) == type([]): out = [s[index_array] for s in fx]
    # else: out = np.take_along_axis(arr = x,indices = index_array,axis = indexx)
    else: out = fx[index_array]
    return (x_sorted,out,index_array)

get_data_switchDict = { pd.DataFrame: lambda x :  x.values,
                        pd.Series: lambda x : np.array(x.array, dtype= 'float'),
                        tuple: lambda xs : [get_data(x) for x in xs],
                    }
def get_data(x):
    type_debug = type(x)
    method = get_data_switchDict.get(type_debug,lambda x: np.asarray(x,dtype='float'))
    return method(x)

def compare_plot_lists_ax(listxs, listfxs, ax, **kwargs):
    index = kwargs.get("index",0)
    labelx=kwargs.get("labelx",'x-units')
    fun_x=kwargs.get("fun_x",get_data)
    extra_plot_fun=kwargs.get("extra_plot_fun",None)
    labely=kwargs.get("labely",'f(x)-units')
    listlabels=kwargs.get("listlabels",[None for n in range(len(listxs))])
    listalphas=kwargs.get("alphas",np.repeat(1.,len(listxs)))
    xscale =kwargs.get("xscale",None)
    yscale =kwargs.get("yscale",None)
    figsize =kwargs.get("figsize",(2,2))
    loc =kwargs.get("loc",'upper left')
    prop =kwargs.get("prop",{'size': 6})
    ax.tick_params(axis='both', which='major', labelsize=kwargs.get('fontsize',10))
    ax.tick_params(axis='both', which='minor', labelsize=kwargs.get('fontsize',10))

    for x,fx,label,alpha in zip(listxs, listfxs,listlabels,listalphas):
        plotx = fun_x(x)
        plotfx = get_data(fx)
        plotx,plotfx,permutation = lexicographical_permutation(plotx,plotfx,**kwargs)
        if extra_plot_fun is not None: extra_plot_fun(plotx,plotfx)
        ax.plot(plotx,plotfx,marker = 'o',ls='-',label= label, markersize=12 / len(plotx),alpha = alpha)
        ax.legend(prop={'size': 6})
    title = kwargs.get("title",'')
    ax.title.set_text(title)
    if yscale is not None: ax.set_yscale(yscale)
    if yscale is not None: ax.set_xscale(xscale)
    plt.legend(prop={'size': kwargs.get('fontsize',10)})
    title = kwargs.get("title",'')
    plt.title(title,fontsize = kwargs.get('fontsize',10))
    ax.set_xlabel(labelx,fontsize = kwargs.get('fontsize',10))
    ax.set_ylabel(labely,fontsize = kwargs.get('fontsize',10))

def plot_trajectories(kwargs):
    fx, fz, data = kwargs["fx"], kwargs["fz"], kwargs["data"]
    colormap, legend = kwargs.get("colormap", "tab20b"), kwargs.get("legend", False)
    symbol = list(data.columns)

    def fun_plot(param, ax=None, **kwargs):
        indice, paths = param[0], param[1]
        limit = kwargs.get("nmax", 200)
        paths = paths[:limit]
        index = data.index[: paths.shape[1]]
        index = kwargs.get("get_index", get_date)((list(index)))
        paths = pd.DataFrame(paths.T, index=index)
        title = kwargs.get("title", None)
        legend = kwargs.get("legend", False)
        # paths = path.T
        paths.plot(
            colormap=colormap,
            legend=legend,
            ax=ax,
            linewidth=kwargs.get("linewidth", 1),
            title=title,
        )
        name = "Ref:" + str(symbol[indice - 1])
        paths[name] = fx[0, indice, : paths.shape[0]].T
        # paths.plot(colormap='tab20b',y=[i for i in range(fz.shape[0]+1)])
        paths[name].plot(color="red", legend=name, linewidth=3, rot=45, ax=ax)

    indices = kwargs.get("plot_indice", [1])
    if "ax" in kwargs:
        for i in indices:
            fun_plot((i, fz[:, i, :]), **kwargs)
    else:
        plots = [(i, fz[:, i, :]) for i in indices]
        params = {**{"mp_nrows": 1, "mp_figsize": (6, int(6 / len(indices))), **kwargs}}
        multi_plot(plots, fun_plot, **params, get_index=get_datetime)


def scatter_hist(x, y, **kwargs):
    # plt.ioff()  # turn off interactive mode
    figsize = kwargs.get("figsize", (9, 9))
    width_ratios = kwargs.get("width_ratios", (7, 3))
    height_ratios = kwargs.get("height_ratios", (3, 7))
    left = kwargs.get("left", 0.1)
    right = kwargs.get("right", 0.9)
    bottom = kwargs.get("bottom", 0.1)
    top = kwargs.get("top", 0.9)
    wspace = kwargs.get("wspace", 0.05)
    hspace = kwargs.get("hspace", 0.05)
    labelx = kwargs.get("labelx", "axis1")
    labely = kwargs.get("labely", "axis2")
    nmax = kwargs.get("nmax", 5000)
    x, y = x[:nmax], y[:nmax]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    # ax = kwargs.get('ax',fig.add_subplot(gs[1, 0]))
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    s = kwargs.get("s", 2.5)
    c = kwargs.get("c", "blue")
    ax.scatter(x, y, s=s, c=c)

    n_std = kwargs.get("n_std", 3)
    edgecolor = kwargs.get("edgecolor", "red")
    linewidth = kwargs.get("linewidth", 3)
    label = r"$" + str(n_std) + r"\sigma$"
    confidence_ellipse(
        x, y, ax, n_std=n_std, edgecolor=edgecolor, label=label, linewidth=linewidth
    )
    # ax.scatter(np.mean(x), np.mean(y), c=s, s=5)
    ax.legend()
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    # bins = int(len(x)/100)
    kde_x = stats.gaussian_kde(x)
    xx = np.linspace(np.min(x), np.max(x), 10000)
    yy = np.linspace(np.min(y), np.max(y), 10000)
    kde_y = stats.gaussian_kde(y)
    bins = kwargs.get("bins", 100)
    ax_histx.hist(x, bins=bins, color=c, density=True)
    ax_histx.plot(xx, kde_x(xx))
    ax_histy.hist(y, bins=bins, color=c, density=True, orientation="horizontal")
    ax_histy.plot(kde_y(yy), yy)
    return fig


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    import matplotlib.transforms as transforms
    from matplotlib.patches import Ellipse

    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def hist_plot(param, **kwargs) -> None:
    """
    author: SMI
    outputs the histogram of list: param.
    """
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from scipy import stats

    ax = kwargs.get("ax", None)
    nmax = kwargs.get("nmax", 5000)
    if ax is None:
        ax = plt.figure()

    labels = kwargs.get("labels", ["labelx", "labely"])
    label_size = kwargs.get("label_size", 5)
    bins = kwargs.get("bins", 100)
    num_colors = kwargs.get("num_colors", len(param))
    colors = cm.rainbow(np.linspace(0, 1, num_colors))

    for i, label, color in zip(param, labels, colors):
        x = get_matrix(i)[:nmax]
        kde_x = stats.gaussian_kde(x.flatten())
        xx = np.linspace(np.min(x), np.max(x), 10000)
        plt.hist(x, bins=bins, density=True, color=color, label=label)
        plt.plot(xx, kde_x(xx))
        plt.legend(loc="upper right", prop={"size": label_size})


def display_historical_vs_generated_distribution(kwargs):
    def graphic(params):
        transform_h = params["transform_h"]
        transform_g = params["transform_g"]
        dim = transform_h.shape[1]

        def fun_plot(param, **kwargs):
            import io

            from PIL import Image

            if dim == 1:
                return hist_plot([param], ax=kwargs["ax"])
            fig = scatter_hist(
                param[:, 0],
                param[:, 1],
                figsize=(3, 3),
                labelx=params["symbols"][0],
                labely=params["symbols"][1],
                **kwargs,
            )
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            kwargs["ax"].imshow(Image.open(img_buf))
            plt.close(fig)
            pass

        fig = multi_plot_figs(
            [transform_h.values, transform_g.values],
            mp_nrows=1,
            mp_figsize=(6, 3),
            fun_plot=fun_plot,
            mp_ncols=2,
            **params,
        )

    if "transform_h" not in kwargs:
        kwargs["transform_h"] = kwargs["transform"](kwargs["data"], **kwargs)
    sample_times = kwargs.get("timelist", kwargs["times"])
    Nz = kwargs.get("Nz", 10)

    sampler = kwargs.get("sampler", None)
    if sampler is None:
        generator = lambda n: np.random.uniform(
            size=(n, kwargs["transform_h"].shape[1])
        )
        sampler = Sampler(kwargs["transform_h"].values, latent_generator=generator)

    transform_g = sampler.sample(Nz * len(sample_times))
    transform_g = pd.DataFrame(transform_g, columns=kwargs["transform_h"].columns)
    kwargs["transform_g"] = transform_g
    kwargs["graphic"] = graphic

    return kwargs


def get_box(x):
    if isinstance(x, list):
        temp = [get_matrix(np.asarray(get_box(y))) for y in x]
        temp = np.concatenate(temp, axis=1)
        return temp
        # out = np.zeros((temp.shape[0],2))
        # for d in range(0,temp.shape[0]):
        #     out[d][0],out[d][1] = np.min(temp[d,:]),np.max(temp[d,:])
        # return out[0,:],out[1,:]

    x = get_matrix(x)
    if x.shape[1] == 1:
        return [np.min(x), np.max(x)]
    out = [get_box(x[:, d]) for d in range(0, x.shape[1])]
    return out


def graph_dates(strings, np_dates, ax=None):
    if (
        isinstance(np_dates, np.ndarray)
        and np_dates.ndim == 1
        and np.all(np.isfinite(np_dates))
    ):
        pass
    else:
        np_dates = np.concatenate(np_dates)
    idx, np_idx = strings, np_dates
    idx_step = int(len(idx) / 4)
    idx_four = np_idx[::idx_step]
    idx_labels_four = idx[::idx_step]
    if ax is None:
        fig, ax = plt.subplots()
    plt.xticks(idx_four, idx_labels_four, rotation=20)
    return ax


def scatter_plot(param, **kwargs) -> None:
    x, y = get_matrix(param[0]), get_matrix(param[1])
    if x.shape[0] * x.shape[1] * y.shape[0] * y.shape[1] == 0:
        return
    color = kwargs.get("color", ["blue", "red"])
    label = kwargs.get("label", ["first", "second"])
    type = kwargs.get("type", ["o", "o"])
    markersize = kwargs.get("markersize", [2, 2])
    plt.plot(
        x[:, 0], x[:, 1], "o", color=color[0], label=label[0], markersize=markersize[0]
    )
    plt.plot(
        y[:, 0], y[:, 1], "o", color=color[1], label=label[1], markersize=markersize[1]
    )
    # plt.plot(y[:,0], y[:,1],'o',color = 'red', label = "Sampling", markersize=2, alpha = 0.5)
    plt.legend()


def plot_confidence_levels(param, **kwargs):
    import itertools

    Nx = kwargs.get("Nx", 10)

    def fun(x, **kwargs):
        out = np.ndarray([x.shape[0], 2])
        out[:, 0], out[:, 1] = (
            get_float(x.index),
            kwargs["getter"].get_closed_formula().get_spot_basket(x, **kwargs),
        )
        return out

    def get_index(param):
        x_test, y_test = list(param[0].index), list(param[1].index)
        x_list, y_list = get_date(x_test), get_date(y_test)
        idx, idy = (
            [d.strftime("%Y-%m-%d") for d in x_list],
            [d.strftime("%Y-%m-%d") for d in y_list],
        )
        return list(pd.DatetimeIndex(np.concatenate([idx, idy])).strftime("%Y/%m/%d"))

    fun = kwargs.get("fun", fun)
    training_set, test_set = fun(param[0], **kwargs), fun(param[1], **kwargs)

    idx = get_index(param)
    idx_num = np.concatenate([training_set[:, 0], test_set[:, 0]])
    # ax = get_ax_helper(**kwargs)

    box = get_box([np.concatenate([test_set, training_set])])
    mean = box.mean(axis=1)
    for d in range(box.shape[1]):
        min_, max_ = box[d, 0], box[d, 1]
        if min_ == max_:
            box[d, 0] -= 1
            box[d, 1] += 1
        else:
            box[d, 0] -= abs(box[d, 0] - mean[d]) * 0.1
            box[d, 1] += abs(box[d, 1] - mean[d]) * 0.1

    xlist, ylist = (
        np.linspace(box[0, 0], box[0, 1], Nx),
        np.linspace(box[1, 0], box[1, 1], Nx),
    )
    x = np.asarray([e for e in itertools.product(xlist, ylist)])
    X, Y = np.meshgrid(xlist, ylist)
    kernel = Kernel(x=x, y=training_set)
    Dxy = kernel.dnm()
    # Dxy = kernel.dnm(x =x,y=x)
    Dxy = Dxy.min(axis=1)
    mat = np.zeros([Nx, Nx])

    def helper(n):
        m, k = np.argmin(abs(x[n, 0] - xlist)), np.argmin(abs(x[n, 1] - ylist))
        mat[k, m] = Dxy[n]

    [helper(n) for n in range(0, x.shape[0])]
    ax = kwargs.get("ax", None)
    if ax is None:
        ax = graph_dates(idx, idx_num)
    contour_filled = ax.contour(X, Y, mat)
    ax.clabel(contour_filled, fmt="%2.1f", fontsize=6)
    plt.colorbar(contour_filled)
    plt.xlabel("time")
    plt.ylabel("basket values")
    # plt.xticks(idx_four, idx_labels_four, rotation=20)

    # plt.scatter((training_set,test_set), label = ['training set', 'test set'])
    scatter_plot((training_set, test_set), label=["training set", "test set"])


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
