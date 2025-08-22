"""
=================================================================
9.04 GARCH(1,1) model
=================================================================
We reproduce here the figure 9.5 of the book.
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
# The GARCH(p,q) map is defined as:
# $$\begin{aligned}
# X^k &= \mu + \sigma^k \epsilon^k, \\
# (\sigma^k)^2 &= \alpha_0 + \sum_{i=1}^{p} \alpha_i (X^{k-i})^2 + \sum_{j=1}^{q} \beta_j (\sigma^{k-j})^2,
# \end{aligned}$$
# where $\mu \in \mathbb{R}$, $\{\epsilon^k\}$ is a white noise sequence with unit variance, and $\sigma^k$ is a stochastic volatility term determined recursively. 
# The parameters $\alpha_i$ and $\beta_i$ denote the GARCH parameters.
q = 1


def garch_pq(x):
    import numpy as np
    from arch import arch_model

    p_values = range(1, 6)
    q_values = range(1, 6)
    aic_values = np.full((len(p_values), len(q_values)), np.inf)

    for i in p_values:
        for j in q_values:
            try:
                model = arch_model(x, vol="Garch", p=i, q=j)
                results = model.fit(disp="off")  # suppress convergence messages
                aic_values[i - 1, j - 1] = results.aic
            except:
                continue

    p, q = np.unravel_index(np.argmin(aic_values, axis=None), aic_values.shape)
    print(f"The smallest AIC is {aic_values[p, q]} for model GARCH({p}, {q})")

    best_p, best_q = p + 1, q + 1
    model = arch_model(x, vol="Garch", p=1, q=1)
    results = model.fit()
    print(results.summary())
    a0 = results.params["omega"]
    a = [
        results.params[f"alpha[{i+1}]"]
        for i in range(best_p)
        if f"alpha[{i+1}]" in results.params
    ]
    b = [
        results.params[f"beta[{i+1}]"]
        for i in range(best_q)
        if f"beta[{i+1}]" in results.params
    ]
    return a, b


def estimate_coeff():
    asx, bsx = [], []
    for i in range(params["data"].values.shape[1]):
        ax, bx = garch_pq(params["data"].values[:, i])
        asx += [ax]
        bsx += [bx]
    return np.asarray(asx).T, np.asarray(bsx).T


params["a"], params["b"] = estimate_coeff()
params["q"] = q
params["map"] = maps.composition_map(
    [maps.garch_map(), maps.mean_map(), maps.diff(), maps.log_map, maps.remove_time()]
)
params = maps.apply_map(params)

#########################################################################
# We define our sampler on the mapped data using codpy's Sampler
# ------------------------------------------------------------------------------------------------
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
# ------------------------------------------------
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
