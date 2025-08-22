"""
==========================
9.03 ARMA(p,1) model
==========================
We reproduce here the figure 9.4 of the book.
Utilitary functions can be found next to this file. Here, we only define codpy-related functions.
"""

#########################################################################
# Necessary Imports
# ------------------------
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from codpy.kernel import Sampler

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
data_path = os.path.join(CURRENT_DIR, "data")
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

import utils.ch9.mapping as maps
from utils.ch9.data_utils import stats_df
from utils.ch9.market_data import retrieve_market_data
from utils.ch9.path_generation import generate_paths
from utils.ch9.plot_utils import display_historical_vs_generated_distribution

#########################################################################
# Parameter definition
# ------------------------

def get_cdpres_param():
    return {
        "rescale_kernel": {"max": 2000, "seed": None},
        "rescale": True,
        "grid_projection": True,
        "reproductibility": False,
        "date_format": "%d/%m/%Y",
        "begin_date": "01/06/2020",
        "end_date": "01/06/2022",
        "today_date": "01/06/2022",
        "symbols": ["AAPL", "GOOGL", "AMZN"],
    }


#########################################################################
# Get the market data
# ------------------------
params = retrieve_market_data()

#########################################################################
# Defining the map
# ------------------------
# The ARMA(p,q) map is defined as:
# $$X^k = \mu + \sum_{i=1}^{p} a_i X^{k-i} + \sum_{j=1}^{q} b_j \epsilon^{k-j}$$
# where $\mu \in \mathbb{R}$ is a mean parameter,
# $\{a_i\}$ and $\{b_j\}$ are model coefficients,
# and $\{\epsilon^k\}$ is a sequence of i.i.d. white noise variables with zero mean and finite variance $\sigma^2$.
# This map requires coefficients p and q to be defined. As described in the book, we suppose
# them to be given.


def arma_pq(x, p, q):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(x, order=(p, 0, q), trend="ct")
    results = model.fit()

    # Return the AR and MA parameters separately
    return results.arparams, results.maparams


p, q = 1, 1


def estimate_coeff():
    asx = []
    amx = []
    for i in range(params["data"].values.shape[1]):
        ar_params, ma_params = arma_pq(params["data"].values[:, i], p, q)
        asx += [ar_params]
        amx += [ma_params]
        # ARMA_stationarity_test(np.asarray(asx).T[:,0], np.asarray(amx).T[:,0])
    return np.asarray(asx).T, np.asarray(amx).T


params["a"], params["b"] = estimate_coeff()
params["map"] = maps.composition_map(
    [maps.arma_map(), maps.mean_map(), maps.remove_time()]
)
params = maps.apply_map(params)

#########################################################################
# We define our sampler on the mapped data using codpy's Sampler
# ------------------------------------------------------------------------
# You can define your own latent generator function, here we use a simple uniform distribution.
# But if not provided, a default one will be used by the Sampler class.
mapped_data = params["transform_h"].values
generator = lambda n: np.random.uniform(size=(n, mapped_data.shape[1]))
sampler = Sampler(mapped_data, latent_generator=generator)
params["sampler"] = sampler

#########################################################################
# We plot the original distribution vs the generated one
# ------------------------------------------------------------------------
params = display_historical_vs_generated_distribution(params)
params["graphic"](params)
plt.show()

#########################################################################
# Reproductibility test
# ------------------------
# We regenerate the same path by generating from the latent representation
# We make sure we get the original data back.
params["reproductibility"] = True
params = generate_paths(params)
params["graphic"](params)
plt.show()

#########################################################################
# We now generate a new set of 10 paths
# ------------------------------------------------
params["reproductibility"] = False
params["Nz"] = 10
params = generate_paths(params)
params["graphic"](params)
plt.show()

stats = stats_df(params["transform_h"], params["transform_g"]).T
print(stats)
