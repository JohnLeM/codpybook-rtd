
#########################################################################
# Necessary Imports
# ------------------------
#########################################################################
import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql

import codpy.AAD as AAD
from codpy import core
from codpy.conditioning import Sampler
from codpy.data_conversion import get_date, get_float
from codpy.kernel import Kernel, get_matrix
from codpy.metrics import get_relative_mean_squared_error
from codpy.plot_utils import compare_plot_lists, multi_plot, multi_plot_figs, plot1D
from codpy.sampling import get_uniforms

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
data_path = os.path.join(CURRENT_DIR, "data")
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

import utils.ch9.mapping as maps
from utils.ch9.data_utils import csv_to_np, get_time, stats_df
from utils.ch9.market_data import calibrate_Heston, retrieve_market_data
from utils.ch9.montecarlo import *
from utils.ch9.path_generation import (
    anais_path_generator,
    data_random_generator,
    generate_from_samples_np,
    generate_paths,
)
from utils.ch9.plot_utils import (
    hist_plot,
    plot_confidence_levels,
    plot_trajectories,
    plot_trisurf,
    scatter_hist,
)
from utils.ch9.ql_tools import (
    BSM,
    Heston,
    Heston_param_getter,
    HestonClosedFormula,
    HestonOption,
    QL_params,
    QL_path_generator,
    basket_option_param_getter,
    date_to_quantlib,
    option_param_getter,
)
from utils.pytorch_operators import *


#########################################################################
# Parameter definition
# ------------------------
#########################################################################
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


def set_fun(x, **kwargs):
    return kwargs["getter"].set_time_spot(x, **kwargs)


def get_anais_bsm_params(kw=None):
    if kw is None:
        kw = QL_params()

    params = {
        "N": kw.get("N", 100),
        "Nz": kw.get("N", 100),
        "getter": option_param_getter(),
        "today_date": "01/06/2020",
        "maturity_date": "01/06/2022",
        "option_type": ql.Option.Call,
        "option": {
            "maturity_date": "01/06/2022",  # ?
            "timesteps": 1,
            "today_date": "01/06/2020",
            "strike_price": 50,
            "spot": 50,
            "option_type": ql.Option.Call,
        },
        "Stats": False,
        "BSM": {
            "spot_price": 100,
            "risk_free_rate": 0.00,
            "dividend_rate": 0,
            "volatility": 0.25,
        },
        "BasketOption": {
            "option_type": ql.Option.Call,
            "today_date": datetime.date(2020, 6, 1),
            "maturity_date": datetime.date(2022, 6, 1),
            "strike_price": 125,
            "risk_free_rate": 0.0,
            "dividend_rate": 0,
            "number_of_assets": 2,
            "weights": [0.5, 0.5],
        },
        "BSMultiple": {
            #'spot_price' : [100., 120.],
            #'risk_free_rate' : 0.0,
            "dividend_rate": 0,
            "spot_price": [100.0, 110.0],
            "volatility": [0.025, 0.025],
            "correlation_matrix": [
                [1, 0],
                [
                    0,
                    1,
                ],
            ],
            "BSMs": [
                {
                    "BSM": {
                        "spot_price": 100,
                        "risk_free_rate": 0.00,
                        "dividend_rate": 0,
                        "volatility": 0.025,
                    }
                },
                {
                    "BSM": {
                        "spot_price": 150,
                        "risk_free_rate": 0.00,
                        "dividend_rate": 0,
                        "volatility": 0.025,
                    }
                },
            ],
        },
    }
    return {**kw, **params}


def get_instrument_param_Heston_option(kwargs=None):
    if kwargs is None:
        kwargs = retrieve_market_data()

    dim = kwargs["data"].shape[1]
    strike_price = np.mean(kwargs["data"].values[:, 0])
    params = {
        "getter": Heston_param_getter(),
        "HestonOption": {
            "option_type": ql.Option.Call,
            "maturity_date": get_date(kwargs["end_date"]),
            "strike_price": strike_price,
        },
    }
    out = {**kwargs, **params}

    out["payoff"] = HestonOption(**out)

    return out


#########################################################################
# Get the market data
# ------------------------
#########################################################################


def calibrate_bsm(kw):
    times = kw["times"]
    # times=(times - times[0])/365.
    x = get_matrix(kw["data"])
    spot_price = x[0, 0]
    x = maps.composition_map([maps.diff(), maps.log_map])(x.T, {**kw, "times": times}).T
    times = 252.0
    # x *= 252.
    sigma = np.std(x[:-1]) * np.sqrt(times)
    # sigma=0.00001
    mu = np.mean(x[:-1]) * times
    BSM_ = {
        "spot_price": spot_price,
        "risk_free_rate": mu,
        "dividend_rate": 0,
        "volatility": sigma,
    }
    BSMultiple = {"correlation_matrix": [[1]], "BSMs": [{"BSM": BSM_}]}

    kw["BSM"] = BSM_
    kw["path_generator"] = {"process": BSM().get_process(**kw)}
    kw["BSMultiple"] = BSMultiple

    return kw


def retrieve_market_data_BSM(kw=None):
    if kw is None:
        kw = {**get_anais_bsm_params(), **retrieve_market_data()}
    times = kw["times"]
    params = {
        "path_generator": {"process": BSM().get_process(**kw)},
        "today_date": times[0],
        "maturity_date": times[-1],
        "generator": anais_path_generator(kw),
        "plot_indice": [0],
        "time_list": times,
        # 'data':pd.DataFrame(kw['data'].iloc[:,0]),
        "Nz": kw["N"],
    }
    times = (times - times[0]) / 365.0
    data = QL_path_generator().generate(**{**params, "time_list": times, "N": 1})
    params["data"] = pd.DataFrame(data[0].T, index=kw["times"])

    out = {**kw, **params}
    out = calibrate_bsm(
        out
    )  ## kwargs avec le mu et sigma modifé dans le champs BSM issu de retrieve market data
    # out['payoff'] = option(**out)
    x = QL_path_generator().generate(**{**out, "time_list": times})
    out = {**out, "paths": x}
    return out


def get_instrument_param_basket_option(kwargs=None):
    if kwargs is None:
        kwargs = retrieve_market_data()

    dim = kwargs["data"].shape[1]

    def basket_values(values, **kwargs):
        basket_values = (
            basket_option_param_getter()
            .get_instrument(**kwargs)
            .basket_value(
                values=values, weights=kwargs["BasketOption"]["weights"], **kwargs
            )
        )
        return basket_values

    strike_price = np.mean(kwargs["data"].values)
    strike_price = np.mean(strike_price)
    params = {
        "getter": basket_option_param_getter(),
        "BasketOption": {
            "weights": np.repeat(1.0 / dim, dim),
            "option_type": ql.Option.Call,
            "maturity_date": get_date(kwargs["end_date"]),
            "strike_price": strike_price,
        },
    }
    out = {**kwargs, **params}

    out["basket_values"] = basket_values
    out["payoff"] = basket_option_param_getter().get_instrument(**out)

    return out


class trivial_last_path_generator2(path_generator):
    def generate(self, payoff, time_list=None, **kwargs):
        paths_keys = kwargs.get("paths_key", "paths")
        out = kwargs[paths_keys][..., 1:, :]
        return out[..., -1]


class get_pricer_param_BSM:
    def get_zero_coupon(kw):
        getter = kw["getter"]
        risk_free_rate = np.mean(getter.get_risk_free_rate(**kw))
        today_date, maturity_date = (
            getter.get_today_date(**kw),
            getter.get_maturity_date(**kw),
        )
        time_to_maturity = ql.Actual365Fixed().yearFraction(
            date_to_quantlib(today_date), date_to_quantlib(maturity_date)
        )
        return np.exp(-risk_free_rate * time_to_maturity)

    def cf_pricer(self, values, **kwargs):
        if isinstance(values, pd.DataFrame):
            values = csv_to_np(values, **kwargs).squeeze().T
        else:
            values = get_matrix(values)
        ql_process = path_generator.get_param(**kwargs).get("process")
        out = (
            kwargs["getter"]
            .get_closed_formula()
            .prices(
                ql_process=ql_process,
                values=values,
                set_fun=kwargs["getter"].set_spot,
                **kwargs,
            )
        )
        if kwargs.get("Stats", False):
            return {"closed price": pd.Series([out], index=[0])}
        else:
            return out

    def MC_pricer(self, **kw):
        out = MonteCarloPricer().price(
            **{**kw, "generator": trivial_last_path_generator()}
        )
        zero_coupon = get_pricer_param_BSM.get_zero_coupon(kw)
        if type(kw["getter"]) != type(Heston_param_getter()):
            out = out * zero_coupon
        # if kw.get('Stats',False): return {'MC pricer': out}
        else:
            return out

    def GEN_pricer(self, **kw):
        out = MonteCarloPricer().price(
            **{**kw, "generator": trivial_last_path_generator2(), "paths_key": "fz"}
        )
        zero_coupon = get_pricer_param_BSM.get_zero_coupon(kw)
        if type(kw["getter"]) != type(Heston_param_getter()):
            out = zero_coupon * out
        # if kw.get('Stats',False): return {'GEN pricer': out}
        else:
            return out

    def __call__(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs["cf_pricer"], kwargs["mc_pricer"], kwargs["gen_pricer"] = (
            self.cf_pricer,
            self.MC_pricer,
            self.GEN_pricer,
        )
        kwargs["today_date"] = kwargs["begin_date"]
        return kwargs


def get_pricer_param(kwargs=None):
    if kwargs is None:
        kwargs = get_instrument_param_basket_option()
    spots = kwargs["data"].values[-1, :]
    params = {
        "BSMultiple": {
            "BSMs": [
                {
                    "BSM": {
                        "spot_price": spots[n],
                        "risk_free_rate": 0,
                        "dividend_rate": 0,
                        "volatility": 0.25,
                    }
                }
                for n in range(len(spots))
            ],
            "correlation_matrix": kwargs["data"].corr(),
        }
    }

    def pricer(values, **kwargs):
        if isinstance(values, pd.DataFrame):
            values = csv_to_np(values, **kwargs).squeeze().T
        else:
            values = get_matrix(values)
        # out= kwargs['getter'].get_closed_formula().prices(values = values[:,1:],set_fun = kwargs['getter'].set_spot,**kwargs)
        set_fun_ = kwargs.get('set_fun', set_fun)
        kwargs['set_fun'] = set_fun_
        out = (
            kwargs["getter"]
            .get_closed_formula()
            .prices(values=values, **kwargs)
        )
        return out

    out = {**params, **kwargs}
    out["pricer"] = pricer

    def graphic_payoff(out):
        payoff_values = out["payoff"](x=out["data"], **out)
        basket_values_ = out["basket_values"](values=out["data"], **out)
        strike = out["getter"].get_strike(**out)
        basket_values_ = (basket_values_ / strike - 1.0) * 100
        ax = out.get("ax", None)
        if ax is None:
            fig, ax = plt.subplots(figsize=out.get("figsize", (6, 6)))
        figure2 = pd.DataFrame(
            np.array([basket_values_, payoff_values]).T,
            columns=["basket values", "payoff values"],
        )
        plot1D(figure2, labelx="basket values (%K)", labely="payoff values", ax=ax)
        #

    def graphic_pricer(out):
        values = out["data"]
        pricer_values = get_matrix(out["pricer"](values=values, **out))
        times = get_matrix(np.array(out["times"]))
        values = np.concatenate([times, pricer_values], axis=1)
        figure2 = pd.DataFrame(values, columns=["basket values", "pricer values"])
        ax = out.get("ax", None)
        ax = graph_dates(out["data"].index, times, ax=ax)
        plot1D(figure2, ax=ax, labelx="times days", labely="pricer values")
        #

    def graphic(kw):
        multi_plot(
            [kw, kw],
            fun_plot=[graphic_payoff, graphic_pricer],
            mp_ncols=2,
            mp_figsize=(6, 3),
        )

    out["graphic"] = graphic

    return out


def get_model_param(kwargs=None):
    if kwargs is None:
        kwargs = get_pricer_param()
    map_ = kwargs.get(
        "map", maps.composition_map([maps.diff(), maps.log_map, maps.remove_time()])
    )
    params = {
        "map": map_,
        "seed": None,
    }

    def transform(csv, **kwargs):
        values_ = csv_to_np(csv)
        out = kwargs["map"](values_, kwargs)
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])
        if len(csv.columns) == out.shape[0]:
            out = pd.DataFrame(out.T, columns=csv.columns)
        else:
            out = pd.DataFrame(out.T)
        return out

    out = {**params, **kwargs}
    out["transform"] = transform
    out["transform_h"] = transform(out["data"], **out)

    def graphic(params):
        transform_h = params["transform"](params["data"], **params)
        print("mean:", transform_h.mean(), " cov:", transform_h.cov())

        dim = len(params["symbols"])
        if dim == 1:
            multi_plot(
                [[transform_h.values[:, 0]]],
                mp_figsize=(3, 3),
                fun_plot=hist_plot,
                labelx=params["symbols"][0],
                mp_ncols=1,
                **kwargs,
            )
        else:
            scatter_hist(
                transform_h.values[:, 0],
                transform_h.values[:, 1],
                labelx=params["symbols"][0],
                labely=params["symbols"][1],
                figsize=(5, 5),
                **params,
            )

    out["graphic"] = graphic

    return out


def retrieve_market_data_Heston(kw=None):
    if kw is None:
        kw = {**QL_params(), **retrieve_market_data()}
    # kappa = getter.get_kappa(**kw)
    times = kw["times"]
    seed = kw.get("seed", 42)
    params = {
        "N": kw.get("N", 100),
        "Nz": kw.get("Nz", 1000),
        "path_generator": {"process": Heston().get_process(**kw)},
        "today_date": times[0],
        "maturity_date": times[-1],
        "generator": anais_path_generator(kw),
        "plot_indice": [0],
        "time_list": times,
        "getter": Heston_param_getter(),
    }
    times = (times - times[0]) / 365.0
    data = QL_path_generator().generate(
        **{**params, "time_list": times, "N": 1, "seed": seed}
    )
    # plt.plot(data[0,1,:])
    params["data"] = pd.DataFrame(
        data[0, 0, :].T, index=kw["times"], columns=["Heston gen."]
    )

    out = {**kw, **params}
    out = calibrate_Heston(
        out
    )  ## kwargs avec le mu et sigma modifé dans le champs BSM issu de retrieve market data
    # out['payoff'] = option(**out)
    x = QL_path_generator().generate(
        **{**out, "time_list": times, "N": out["Nz"], "seed": seed}
    )
    out = {**out, "paths": x[:, [0], :]}
    return out


class test_reproductibility_BSM:
    def get_env_param(self):
        generated_paths_params = retrieve_market_data_BSM()
        generated_paths_params = get_instrument_param_basket_option(
            generated_paths_params
        )
        generated_paths_params = get_pricer_param_BSM()(generated_paths_params)
        return generated_paths_params

    def get_params(self, params=None):
        if params is None:
            params = self.get_env_param()
        generated_paths_params = get_model_param(params)
        generated_paths_params = generate_paths(
            {**generated_paths_params, **{"reproductibility": True}}
        )
        return generated_paths_params

    def __init__(self, params=None) -> None:
        self.generated_paths_params = self.get_params(params)

    def graphic(generated_paths_params):
        if isinstance(generated_paths_params, list):
            #    list_ = [cls.generated_paths_params for cls in generated_paths_params]
            multi_plot(
                generated_paths_params,
                fun_plot=test_reproductibility_BSM.graphic,
                mp_figsize=(4, 3),
            )

            return

        initial_path = get_matrix(generated_paths_params["data"])
        reproduced_path = generated_paths_params["fz"][:, 1:, :][0, 0, :]
        plt.plot(initial_path, color="red", label="Initial path")
        plt.plot(reproduced_path, color="blue", label="Reproduced path", linestyle="--")
        plt.legend()

    def __call__(self, generated_paths_params=None):
        if generated_paths_params is None:
            generated_paths_params = self.generated_paths_params
        generated_paths_params["graphic"] = test_reproductibility_BSM.graphic
        return generated_paths_params
        pass


class test_reproductibility_BSM_diff_log(test_reproductibility_BSM):
    def get_map(self, kwargs):
        return maps.composition_map([maps.diff(), maps.log_map, maps.remove_time()])

    def get_params(self, params=None):
        if params is None:
            params = self.get_env_param()
        generated_paths_params = get_model_param(
            {**params, **{"map": self.get_map(params)}}
        )
        generated_paths_params = generate_paths(
            {**generated_paths_params, **{"reproductibility": True}}
        )
        return generated_paths_params


class get_pricer_param_Heston(get_pricer_param_BSM):
    def cf_pricer(self, values, **kwargs):
        if isinstance(values, pd.DataFrame):
            values = csv_to_np(values, **kwargs).squeeze().T
        else:
            values = get_matrix(values).T
        ql_process = path_generator.get_param(**kwargs).get("process")
        set_fun = HestonOption(**kwargs).set_spot
        out = HestonClosedFormula().prices(
            ql_process=ql_process, values=values[:, 0], set_fun=set_fun, **kwargs
        )
        return out


class test_reproductibility_Heston_diff_log(test_reproductibility_BSM_diff_log):
    def get_env_param(self):
        generated_paths_params = retrieve_market_data_Heston()
        generated_paths_params = get_instrument_param_Heston_option(
            generated_paths_params
        )
        generated_paths_params = get_pricer_param_Heston()(generated_paths_params)
        return generated_paths_params


class test_reproductibility_BSM_cond_map(test_reproductibility_BSM_diff_log):
    def get_map(self, kwargs):
        return maps.composition_map(
            (
                maps.VarConditioner_map(kwargs),
                maps.add_variance_map(var_q=10),
                maps.log_map,
                maps.remove_time(),
            )
        )


class test_reproductibility_Heston_cond_map(test_reproductibility_BSM_cond_map):
    def get_env_param(self):
        generated_paths_params = retrieve_market_data_Heston()
        generated_paths_params = get_instrument_param_Heston_option(
            generated_paths_params
        )
        generated_paths_params = get_pricer_param_Heston()(generated_paths_params)
        return generated_paths_params


#########################################################################
# Reproducibility tests
# ------------------------
#########################################################################


def fig_test_reproductibility_Heston():
    test_reproductibility_Heston_diff_log1 = test_reproductibility_Heston_diff_log()()
    test_reproductibility_Heston_diff_log2 = test_reproductibility_Heston_cond_map(
        test_reproductibility_Heston_diff_log1
    )()
    test_reproductibility_Heston_diff_log1["graphic"](
        [test_reproductibility_Heston_diff_log1, test_reproductibility_Heston_diff_log2]
    )
    return test_reproductibility_Heston_diff_log1


#########################################################################
# Comparing distributions
# ------------------------
#########################################################################
class compare_distributions_BSM(test_reproductibility_BSM):
    def get_params(self, params=None):
        if params is None:
            params = self.get_env_param()
        generated_paths_params = get_model_param(params)
        generated_paths_params = generate_paths(generated_paths_params)
        return generated_paths_params

    def graphic(params, kwargs={}):
        import io

        from PIL import Image

        def get_list(params):
            if isinstance(params, list):
                out = [
                    [params[n]["transform_h"], params[n]["transform_g"]]
                    for n in range(0, len(params))
                ]
                return out
            else:
                return get_list([params])

        transforms = get_list(params)
        dim = transforms[0][0].shape[1]

        def fun_plot(param, **kwargs):
            param1, param2 = param[0], param[1]
            dim1, dim2 = param1.shape[1], param2.shape[1]
            if dim1 == 1 and dim2 == 1:
                kwargs["ax"] = hist_plot(param, **kwargs)
                return
            fig = scatter_hist(
                x=param1.iloc[:, 0],
                y=param1.iloc[:, 1],
                figsize=(3, 3),
                labelx=str(param1.columns[0]),
                labely=str(param1.columns[1]),
                **kwargs,
            )
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            kwargs["ax"].imshow(Image.open(img_buf))
            plt.close(fig)
            pass

        labels = ["gen. noise", "hist. noise"]
        multi_plot_figs(
            transforms,
            mp_nrows=1,
            mp_figsize=(6, 3),
            fun_plot=fun_plot,
            mp_ncols=len(transforms),
            **{**kwargs, **{"labels": labels}},
        )

    def __call__(self, kwargs=None):
        if kwargs is None:
            kwargs = self.generated_paths_params
        kwargs["graphic"] = compare_trajectories_BSM.graphic
        # if 'transform_h' not in kwargs:
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
        # transform_g = pd.DataFrame(transform_g,columns = kwargs['transform_h'].columns)
        kwargs["transform_g"] = transform_g
        kwargs["graphic"] = compare_distributions_BSM.graphic
        D = kwargs["transform_g"].shape[1]
        f_name = ["Cal. Model(Gen. Model):" + str(d) for d in range(D)]
        kwargs["df_stats"] = stats_df(
            kwargs["transform_h"],
            kwargs["transform_g"],
            f_names=f_name,
            format="{:.1e}",
        ).T

        return kwargs


class compare_trajectories_BSM(compare_distributions_BSM):
    def graphic(cls):
        # def fun_plot(param,**kwargs) :
        #     import io
        #     from PIL import Image
        #     fig = plot_trajectories(**param)
        #     img_buf = io.BytesIO()
        #     plt.savefig(img_buf, format='png')
        #     plt.close(fig)
        #     plt.imshow(Image.open(img_buf))
        #     pass
        def get_list(call_cls):
            if isinstance(call_cls, list):
                out = [
                    {
                        **call_cls[0],
                        **{
                            "fx": call_cls[0]["fx"][:, 1:, :],
                            "fz": call_cls[0]["paths"],
                            "title": "Heston generator",
                        },
                    }
                ]
                out += [
                    {
                        **call_cls[n],
                        **{
                            "fx": call_cls[0]["fx"][:, 1:, :],
                            "fz": call_cls[n]["fz"][:, 1:, :],
                            "title": call_cls[n].get("name", None),
                        },
                    }
                    for n in range(0, len(call_cls))
                ]
                return out
            else:
                return get_list([cls], [call_cls])

        paths = get_list(cls)
        multi_plot(
            paths,
            mp_nrows=1,
            mp_figsize=(6, 3),
            fun_plot=plot_trajectories,
            mp_ncols=len(paths),
            set_visible=False,
        )

    def __call__(self, generated_paths_params=None):
        if generated_paths_params is None:
            generated_paths_params = self.generated_paths_params
        generated_paths_params["graphic"] = compare_trajectories_BSM.graphic
        return generated_paths_params
        pass


class compare_distributions_BSM_diff_log(compare_distributions_BSM):
    def get_map(self, params):
        return maps.composition_map([maps.diff(), maps.log_map, maps.remove_time()])

    def get_params(self, params=None):
        if params is None:
            params = self.get_env_param()
        generated_paths_params = get_model_param(
            {**params, **{"map": self.get_map(params)}}
        )
        generated_paths_params = generate_paths(generated_paths_params)
        return generated_paths_params


class HestonDiffLog(compare_distributions_BSM_diff_log):
    def set_name(self, kwargs):
        kwargs["name"] = "Heston log diff"
        return kwargs

    def get_env_param(self):
        generated_paths_params = retrieve_market_data_Heston()
        generated_paths_params = get_instrument_param_Heston_option(
            generated_paths_params
        )
        generated_paths_params = get_pricer_param_Heston()(generated_paths_params)
        generated_paths_params = self.set_name(generated_paths_params)
        return generated_paths_params


class HestonCondMap(HestonDiffLog):
    def get_map(self, kwargs):
        return maps.composition_map(
            (
                maps.VarConditioner_map(kwargs),
                maps.add_variance_map(var_q=10),
                maps.log_map,
                maps.remove_time(),
            )
        )

    # def get_map(self,kwargs): return composition_map([QuantileConditioner_map(kwargs),add_variance_map(var_q=5),log_map,remove_time()])
    def get_params(self, params=None):
        if params is None:
            params = self.get_env_param()
        # params['Denoiser'] = None
        return super().get_params(params)

    def set_name(self, kwargs):
        kwargs["name"] = "Heston VolLoc"
        return kwargs


def fig_compare_distributions_Heston(params=None):
    a_instance = HestonDiffLog(params)
    a = a_instance()
    index = [
        a_instance.__class__.__name__ + " lat.:" + str(n)
        for n in range(a["df_stats"].shape[1])
    ]
    b_instance = HestonCondMap(a)
    b = b_instance()
    index += [
        b_instance.__class__.__name__ + " lat.:" + str(n)
        for n in range(b["df_stats"].shape[1])
    ]
    a["graphic"]([a, b])
    out = pd.concat([a["df_stats"].T, b["df_stats"].T])
    out.index = index
    return out


#########################################################################
# Comparing trajectories
# ------------------------
#########################################################################


class compare_trajectories_BSM_diff_log(compare_trajectories_BSM):
    def get_map(self, params):
        return maps.composition_map([maps.diff(), maps.log_map, maps.remove_time()])

    def get_params(self, params=None):
        if params is None:
            params = self.get_env_param()
        params = self.set_name(params)
        generated_paths_params = get_model_param(
            {**params, **{"map": self.get_map(params)}}
        )
        generated_paths_params = generate_paths(generated_paths_params)
        return generated_paths_params


class compare_trajectories_Heston_diff_log(compare_trajectories_BSM_diff_log):
    def get_env_param(self):
        generated_paths_params = retrieve_market_data_Heston()
        generated_paths_params = get_instrument_param_Heston_option(
            generated_paths_params
        )
        generated_paths_params = get_pricer_param_Heston()(generated_paths_params)
        return self.set_name(generated_paths_params)

    def set_name(self, kwargs):
        kwargs["name"] = "Heston Diff Log map"
        return kwargs


class compare_trajectories_Heston_cond_map(compare_trajectories_Heston_diff_log):
    def __init__(self, params=None) -> None:
        super().__init__(params)

    def get_params(self, params=None):
        if params is None:
            params = self.get_env_param()
        # params['Denoiser'] = None
        return super().get_params(params)

    def get_map(self, kwargs):
        return maps.composition_map(
            (
                maps.VarConditioner_map(kwargs),
                maps.add_variance_map(var_q=5),
                maps.log_map,
                maps.remove_time(),
            )
        )

    # def get_map(self,kwargs): return composition_map([QuantileConditioner_map(kwargs),add_variance_map(var_q=5),log_map,remove_time()])

    def set_name(self, kwargs):
        kwargs["name"] = "Heston Cond map"
        kwargs["coeff"] = 0.98  # ...
        # kwargs['Denoiser'] = None

        return kwargs


def fig_compare_trajectories_Heston(kw=None):
    if kw is None:
        kw = compare_trajectories_Heston_diff_log()()
    HestonLogDiff = compare_trajectories_Heston_diff_log(kw)
    HestonCond = compare_trajectories_Heston_cond_map(kw)
    compare_trajectories_BSM.graphic(
        [HestonLogDiff.generated_paths_params, HestonCond.generated_paths_params]
    )
    return kw


#########################################################################
# Hundred examples of generated paths with conditioned covariance map
# ------------------------
#########################################################################


def get_divergence_param(kwargs=None):
    if kwargs is None:
        kwargs = get_pricer_param()

    def get_x(fx, **kwargs):
        return fx

    def get_z(x, **kwargs):
        Nz = int(kwargs["Nz"])
        Dx = x.shape[1]
        Tx = x.shape[0]
        Ez = int(Nz / x.shape[0])
        z = np.ndarray([Ez, Tx, Dx])
        # fixed_dist = np.random.random_sample([Ez,Dx])
        fixed_dist = get_uniforms(Ez, Dx, **kwargs)

        def helpern(n):
            z[:, n, :] = fixed_dist

        [helpern(n) for n in range(0, Tx)]
        return z.reshape([z.shape[0] * z.shape[1], z.shape[2]])

    kwargs["get_x"], kwargs["get_z"] = get_x, get_z
    kwargs["map"] = maps.composition_map(
        (maps.diffusive_map(kwargs), maps.log_map, maps.remove_time())
    )
    # kwargs['map'] = composition_map((diffusive_map(kwargs),remove_time()))
    kwargs["Nz"], kwargs["legend"], kwargs["colormap"] = 100, False, "binary"
    # kwargs['get_rand'] = alg.get_uniform_like
    return kwargs


#########################################################################
# Pricing as a function of time
# ------------------------
#########################################################################
def get_instrument_param_basket_option_mat(kwargs=None):
    from dateutil.relativedelta import relativedelta

    if kwargs is None:
        kwargs = get_instrument_param_basket_option()
    spot_ = kwargs["data"].values.T[:, -1]
    spot_ = np.mean(spot_)
    kwargs["BasketOption"]["strike_price"] = spot_
    maturity_date = get_date(kwargs["end_date"]) + relativedelta(days=30)
    kwargs["BasketOption"]["maturity_date"] = maturity_date
    kwargs["payoff"] = basket_option_param_getter().get_instrument(**kwargs)
    return kwargs


#########################################################################
# Training and test set
# ------------------------
#########################################################################


def get_var_param(kwargs):
    return {
        ##codpy specific
        "rescale_kernel": {"max": 2000, "seed": None},
        # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 0,0.,map_setters.set_unitcube_map),
        # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel,2,1e-8,None),
        # 'set_kernel' : core.kernel_setter(kernel_string='matern', polynomial_order=2, regularization=1e-8,map_string="standardmean"),
        "set_kernel": core.kernel_setter(
            kernel_string="matern",
            polynomial_order=2,
            regularization=1e-8,
            map_string="standardmean",
        ),
        # 'set_kernel': core.kernel_setter('matern', None, 2, 1e-8),
        "distance": "norm22",
        "rescale": True,
        "grid_projection": True,
        "map": maps.composition_map(
            [maps.normal_return(kwargs), maps.log_map, maps.remove_time()]
        ),
    }


def basket_values(values, **kwargs):
    basket_values = (
        basket_option_param_getter()
        .get_instrument(**kwargs)
        .basket_value(
            values=values, weights=kwargs["BasketOption"]["weights"], **kwargs
        )
    )
    return basket_values


def get_var_data(kwargs=None):
    if kwargs is None:
        kwargs = get_model_param()
    kwargs = {
        **kwargs,
        **get_var_param(kwargs),
    }  # need a 2 order polynomial kernel, need a normal return map

    # kwargs = maps.apply_map(kwargs)

    horizon = kwargs.get("horizon", 10)
    getter = kwargs["getter"]
    params = {
        "Nz": 500,
        "H": 1,
    }
    out = {**kwargs, **params}
    columns = out["data"].columns

    def np_to_csv(np, columns=columns, **kwargs):
        values = np.squeeze()
        values, index = values[:, 1:], values[:, 0]
        out = pd.DataFrame(values, columns=columns, index=index)
        return out

    def random_sample(shape, **kwargs):
        import math

        from scipy.stats import qmc

        sampler = qmc.Sobol(d=shape[1], scramble=False)
        m = int(math.log(shape[0], 2)) + 1
        out = sampler.random_base2(m)[: shape[0]]
        return out

    samples = csv_to_np(out["data"], **out)
    # last day
    today_date = get_float(getter.get_today_date(**out))
    sample_times = today_date + horizon

    # VarData: We generate paths of length t+horizon.
    # Up to time t, this is historical data
    # From t to t+horizon, this is generated data
    # Shape is historical data (Nz,D,t) + generated data (Nz,D,horizon) stacked on the last dimension

    # This is to make sure we set the initial conditions correctly
    # i.e we put them to be the last value of the historical data
    out["sample_times"] = [sample_times]
    out = maps.apply_map(out)

    # Here we generate new data
    # This will generate from the initial conditions, setup above
    SyntheticData = generate_from_samples_np(
        samples=samples,
        initial_values=getter.get_spot(**out),
        time_start=getter.get_today_date(**out),
        random_sample=random_sample,
        **out,
    )
    # SyntheticData = historical_path_generator().generate_from_samples(
    #     samples = samples,
    #     sample_times=[sample_times],
    #     initial_values = getter.get_spot(**out),
    #     time_start=getter.get_today_date(**out),
    #     random_sample = random_sample,
    #     **out)

    # What are we trying to get exactly here?
    # Generate a test set at time+H only, (500,3,1)
    sample_times = today_date + params["H"]

    out["sample_times"] = [sample_times]
    out = maps.apply_map(out)

    TestData = generate_from_samples_np(
        samples=samples,
        initial_values=getter.get_spot(**out),
        time_start=getter.get_today_date(**out),
        **out,
    )
    # TestData = historical_path_generator().generate_from_samples(
    #     samples = samples,
    #     sample_times=[sample_times],
    #     initial_values = getter.get_spot(**out),
    #     time_start=getter.get_today_date(**out),**out)

    out["TestData"] = np_to_csv(TestData[..., -1], **out)
    values = SyntheticData[:, 1:, -1]
    index = np.repeat(sample_times, values.shape[0])
    SyntheticData = pd.DataFrame(values, columns=columns, index=index)
    index = index + 1.0
    VarDataPlus = pd.DataFrame(values, columns=columns, index=index)
    index = index - 2.0
    VarDataMinus = pd.DataFrame(values, columns=columns, index=index)
    out["SyntheticData"] = pd.concat([VarDataMinus, SyntheticData, VarDataPlus])

    def fun(x, **kwargs):
        out = np.ndarray([x.shape[0], 2])
        out[:, 0], out[:, 1] = (
            get_time(x),
            kwargs["getter"].get_closed_formula().get_spot_basket(x, **kwargs),
        )
        return out

    def graphic(out):
        strike = getter.get_strike(**out)
        baskets = basket_values(values=TestData[:, 1:], **kwargs)
        baskets = (baskets / strike - 1.0) * 100
        multi_plot(
            [[out["data"], out["TestData"]], [out["SyntheticData"], out["TestData"]]],
            plot_confidence_levels,
            f_names=["Hist. training / test set", "Synthetic training / test set"],
            labelx="basket values",
            labely="time",
            mp_figsize=(6, 3),
            fun=fun,
            loc="upper left",
            prop={"size": 3},
            mp_ncols=2,
            **out,
        )

    out["graphic"] = graphic
    return out


#########################################################################
# Prices output
# ------------------------
#########################################################################
def predict_prices(kwargs=None):
    if kwargs is None:
        kwargs = get_var_data()
    getter = kwargs["getter"]
    strike_price = kwargs["data"].values[-1, :]
    kwargs["BasketOption"]["strike_price"] = np.mean(strike_price)
    kwargs["BasketOption"]["maturity_date"] = get_date(
        kwargs["end_date"]
    ) + datetime.timedelta(days=30)

    def taylor(values, **kwargs):
        time_spot_ = kwargs["getter"].get_time_spot(**kwargs)
        fx = kwargs["pricer"]  # (values = time_spot_,**kwargs)
        taylor_explanation = {}
        # out = AAD.AAD.taylor_expansion(x=time_spot_, y=time_spot_,z=values, fx=fx, nabla = getter.get_closed_formula(**kwargs).nabla, hessian = getter.get_closed_formula(**kwargs).hessian,taylor_order = 2,taylor_explanation=taylor_explanation,**kwargs)
        fx = fx(time_spot_, **kwargs)
        out = pde.taylor_expansion(
            x=time_spot_,
            y=time_spot_,
            z=values,
            fx=fx,
            nabla=getter.get_closed_formula(**kwargs).nabla,
            hessian=getter.get_closed_formula(**kwargs).hessian,
            taylor_order=2,
            taylor_explanation=taylor_explanation,
            **kwargs,
        )
        return out

    def plot_helper(xfx, **kwargs):
        spot_baskets, pnls = xfx[0], xfx[1]
        compare_plot_lists({**kwargs, **{"listxs": spot_baskets, "listfxs": pnls}})

    def codpy(values, **kwargs):
        x = csv_to_np(kwargs["SyntheticData"]).squeeze().T
        y = kwargs["getter"].get_time_spot(**kwargs)
        # Pricer get option values from the 3 stocks and time
        fx = get_matrix(kwargs["pricer"](x, **kwargs))
        fx_variate = get_matrix(kwargs["payoff"](x[:, 1:], **kwargs))
        fz_variate = get_matrix(kwargs["payoff"](values[:, 1:], **kwargs))
        fx -= fx_variate
        k = Kernel(
            x=x,
            y=x, 
            fx=fx, 
            # set_kernel=core.kernel_setter("tensornorm", "standardmean", 0, 1e-8),
            # order=2,
        )
        out = k(z=values)
        # out = core.KerOp.projection(x=x, y=x, z=values, fx=fx, kernel_ptr=k, **kwargs)
        out += fz_variate
        return out

    params = {
        "taylor": taylor,
        "codpy": codpy,
    }
    out = {**params, **kwargs}

    def graphic(out):
        labelx = "Basket Values (% K)"
        labely = "Option Values (USD)"
        maturity_date = [
            get_float(getter.get_maturity_date(**out))
            - get_float(getter.get_today_date(**out))
        ]
        f_names = ["exact-Taylor-codpy"]
        TestData = csv_to_np(out["TestData"], **out).squeeze().T
        strike = getter.get_strike(**out)
        baskets = basket_values(values=TestData[:, 1:], **kwargs)
        baskets = (baskets / strike - 1.0) * 100
        plot_datas = [
            [
                [baskets, baskets, baskets],
                [
                    out["pricer"](TestData, **out),
                    out["taylor"](TestData, **out),
                    out["codpy"](TestData, **out),
                ],
            ]
        ]
        title_fig = ["Exact", "Taylor", "codpy"]
        multi_plot(
            plot_datas,
            plot_helper,
            f_names=f_names,
            labelx=labelx,
            labely=labely,
            listlabels=title_fig,
            loc="upper left",
            prop={"size": 3},
            mp_nrows=1,
            **out,
        )

    out["graphic"] = graphic
    return out


#########################################################################
# Predict greeks
# ------------------------
#########################################################################
def codpy_nabla(values, **k):
    x = csv_to_np(k["SyntheticData"]).squeeze().T
    fx = get_matrix(k["pricer"](x, **k))
    out = core.DiffOps.nabla(x=x, y=x, z=values, fx=fx, **k).squeeze()
    return out


def codpy_nabla_corrected(values, **k):
    x = csv_to_np(k["SyntheticData"]).squeeze().T
    fx = get_matrix(k["pricer"](x, **k))
    xbaskets, zbaskets = np.zeros([x.shape[0], 2]), np.zeros([values.shape[0], 2])
    xbaskets[:, 1], xbaskets[:, 0] = basket_values(values=x[:, 1:], **k), x[:, 0]
    zbaskets[:, 1], zbaskets[:, 0] = (
        basket_values(values=values[:, 1:], **k),
        values[:, 0],
    )
    ker = Kernel(
        x=xbaskets, 
        y=xbaskets, 
        fx=fx,
        set_kernel=core.kernel_setter("tensornorm", "scale_to_unitcube", 0, 1e-8),
        # order=0,
    )
    # kpt = Kernel(set_kernel=core.kernel_setter("matern", None, 0, 1e-8)).get_kernel()
    out = ker.grad(z = zbaskets, **k).squeeze()
    # out = core.DiffOps.nabla(
    #     x=xbaskets, y=xbaskets, z=zbaskets, fx=fx, kernel_ptr=kpt, **k
    # ).squeeze()
    out = np.concatenate(
        [out[:, [0]], out[:, [1]] @ get_matrix(k["getter"].get_weights(**k)).T], axis=1
    )
    return out


def predict_greeks(kwargs=None):
    # We basically get SyntheticData and TestData from get_var_data
    # Those are generated paths from the Sampler
    if kwargs is None:
        kwargs = predict_prices()

    def plot_helper(xfx, **kwargs):
        x, fx = xfx[0], xfx[1]
        args = xfx[2]
        compare_plot_lists({**kwargs, **args, **{"listxs": x, "listfxs": fx}})

    def taylor_delta(self, **k):
        getter = k["getter"]
        x = getter.get_time_spot(**k)
        z = csv_to_np(k["TestData"]).T.squeeze()
        delta = k["getter"].get_closed_formula().nablas(values=x, set_fun=set_fun, **k)
        gamma = (
            k["getter"].get_closed_formula().hessians(values=x, set_fun=set_fun, **k)
        )
        indices = np.repeat(0, z.shape[0])
        deltas = delta[indices]
        gammas = gamma[indices]
        product_ = np.reshape(
            [gammas[n] @ deltas[n] for n in range(gammas.shape[0])], deltas.shape
        )
        f_z = delta + product_
        return f_z

    def graphic(nabla=codpy_nabla, **out):
        dim = len(out["symbols"])
        getter = out["getter"]
        labelx = "Basket Values (% K)"
        labely = "Values"
        # listlabels = ['Exact','Codpy','Taylor']
        TestData = csv_to_np(out["TestData"], **out).squeeze().T
        exact_nabla_ = getter.get_closed_formula(**kwargs).nablas(
            TestData, set_fun=set_fun, **out
        )
        codpy_nabla_ = nabla(TestData, **out).squeeze()
        taylor_delta_ = taylor_delta(TestData, **out).squeeze()

        f_names = ["Theta"] + ["Delta-" + out["symbols"][n] for n in range(0, dim)]
        strike = getter.get_strike(**out)
        baskets = basket_values(values=TestData[:, 1:], **kwargs)
        baskets = (baskets / strike - 1.0) * 100
        plot_datas = [
            [
                [baskets, baskets, baskets],
                [exact_nabla_[:, n], codpy_nabla_[:, n], taylor_delta_[:, n]],
                {"listlabels": ["Exact", "Codpy", "Taylor"]},
            ]
            for n in range(0, dim + 1)
        ]
        title_fig = ["Exact-codpy-taylor"]
        multi_plot(
            plot_datas,
            plot_helper,
            f_names=f_names,
            labelx=labelx,
            labely=labely,
            loc="upper left",
            prop={"size": 4},
            mp_ncols=2,
            mp_nrows=2,
            **out,
        )

        pass

    kwargs["graphic"] = graphic
    kwargs["codpy_nabla"] = codpy_nabla
    kwargs["codpy_nabla_corrected"] = codpy_nabla_corrected
    return kwargs


# fig_test_reproductibility_Heston()
# plt.show()

# table = fig_compare_distributions_Heston()
# print(table)
# plt.show()

# fig_compare_trajectories_Heston()
# plt.show()

# TODO: fix this
# params = generate_paths(get_divergence_param())
# params['graphic'](params)
# plt.show()

# params = maps.composition_map([get_pricer_param,get_instrument_param_basket_option_mat])()
# params['graphic'](params)
# plt.show()

# Repro test
# params = get_model_param()
# params = {**params,**get_var_param(params)}
# params['reproductibility'] = True
# params = maps.apply_map(params)
# params = generate_paths(params)
# params['graphic'](params)
# plt.show()

# params = get_var_data()
# params['graphic'](params)
# plt.show()

import torch
from torch import nn

from codpy.parallel import elapsed_time


def get_param21():
    return {
        "PytorchRegressor": {
            "epochs": 500,
            "layers": [128, 128, 128, 128],
            "loss": nn.MSELoss(),
            "activations": [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
            "optimizer": torch.optim.Adam,
        },
        "codpy_param": {
            #'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map),
            # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, 3 ,1e-8 ,None),
            #'set_codpy_kernel':  kernel_setters.kernel_helper(kernel_setters.set_multiquadricnorm_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map),
            # 'set_codpy_kernel':kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None),
            "set_kernel": core.kernel_setter("gaussianper", None, 3, 1e-8),
            # 'rescale' : True,
        },
    }


def benchmark_gradient(x, z, fx, **kwargs):
    fun = kwargs.get("fun")
    nabla_fz = AAD.AAD.gradient(fun, z)
    fz = fun(z).detach().numpy()
    D = x.shape[1]
    # x,y,z,fx,fy,fz,Nx,Ny,Nz = data_random_generator(**kwargs).get_raw_data(D=1,Nx=500,Ny=500,Nz=500)
    pytorch_grad1 = elapsed_time(
        lambda: pytorch.nabla(x=x, y=x, z=z, fx=fx, **kwargs), msg="torch nabla:"
    )
    pytorch_grad2 = elapsed_time(
        lambda: pytorch.nabla(x=x, y=x, z=z, fx=fx, **kwargs), msg="torch nabla:"
    )
    pytorch_f_z = elapsed_time(
        lambda: pytorch.projection(x=x, y=x, z=z, fx=fx, **kwargs),
        msg="torch projection:",
    )

    cp_ker = Kernel(x=x, fx=fx, order=2, **kwargs["codpy_param"])
    codpy_grad = elapsed_time(lambda: cp_ker.grad(z=z), msg="codpy nabla:")
    codpy_f_z = elapsed_time(lambda: cp_ker(z=z), msg="codpy projection:")

    rmse_delta_codpy = get_relative_mean_squared_error(codpy_grad, nabla_fz)
    rmse_delta_pytorch = get_relative_mean_squared_error(pytorch_grad1, nabla_fz)
    print("RMSE delta codpy:", rmse_delta_codpy)
    print("RMSE delta pytorch:", rmse_delta_pytorch)

    list_results = [
        (x, fx),
        (z, fz),
        (z, pytorch_f_z),
        (z, codpy_f_z),
        (z, nabla_fz),
        (z, codpy_grad[:, 0, :]),
        (z, pytorch_grad1),
        (z, pytorch_grad2),
    ]
    f_names = [
        "training set",
        "ground truth values",
        "Pytorch f",
        "Codpy f",
        "exact grad",
        "codpy grad",
        "pytorch grad-1",
        "pytorch grad-2",
    ]
    if D == 1:
        multi_plot(
            list_results,
            plot1D,
            mp_max_items=len(list_results),
            f_names=f_names,
            mp_nrows=2,
            mp_ncols=4,
            mp_figsize=(8, 4),
        )
    else:
        multi_plot(
            list_results,
            plot_trisurf,
            projection="3d",
            mp_max_items=len(list_results),
            f_names=f_names,
            elev=25,
            azim=-80,
            linewidth=0.2,
            antialiased=True,
            mp_nrows=2,
            mp_ncols=4,
            mp_figsize=(8, 4),
        )


def my_fun_torch(x):
    from math import pi

    type_ = type(x)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, requires_grad=True)
    sinss = torch.cos(2 * x * pi)
    if x.dim() == 1:
        sinss = torch.prod(sinss, dim=0)
        ress = torch.sum(x, dim=0)
    else:
        sinss = torch.prod(sinss, dim=1)
        ress = torch.sum(x, dim=1)
    return ress + sinss


def differentialMlBenchmarks(D, N):
    set_kernel = core.kernel_setter("gaussianper", None, 2, 1e-8)
    x, y, z, fx, fy, fz, Nx, Ny, Nz = data_random_generator(
        fun=my_fun_torch, types=["cart", "cart", "cart"]
    ).get_raw_data(D=D, Nx=N, Ny=N, Nz=N)
    benchmark_gradient(
        x=x,
        z=z,
        fx=fx,
        set_codpy_kernel=set_kernel,
        rescale=True,
        **get_param21(),
        fun=my_fun_torch,
        types=["cart", "cart", "cart"],
    )


def taylor_test(**kwargs):
    numbers = kwargs.get("numbers", {"D": 1, "Nx": 500, "Ny": 500, "Nz": 500})
    scenarios = ts_scenario_generator()
    function = kwargs.get("function", my_fun)
    random_generator = kwargs.get("random_generator", AAD_data_random_generator)
    generator_ = [
        random_generator(fun=function, types=["sto", "sto", "sto"], **kwargs),
        random_generator(fun=function, types=["sto", "sto", "sto"], **kwargs),
    ]
    predictor_ = [
        pytorch_taylor(**kwargs, set_data=False),
        codpy_taylor(**kwargs, set_data=False),
    ]
    dic_kwargs = [
        {
            **numbers,
            "generator": generator,
            "predictor": predictor,
            **predictor.get_new_params(**kwargs),
        }
        for generator, predictor in zip(generator_, predictor_)
    ]
    scenarios.run_scenarios(dic_kwargs, data_accumulator())
    # return scenarios
    x, z, fx, f_z, fz = (
        scenarios.accumulator.get_xs()[0],
        scenarios.accumulator.get_zs()[0],
        scenarios.accumulator.get_fxs()[0],
        scenarios.accumulator.get_f_zs(),
        scenarios.accumulator.get_fzs()[0],
    )
    f_zaad = f_z[0]
    f_zk = f_z[1]
    order = kwargs.get("taylor_order", [])
    D = numbers["D"]
    list_results, f_names = (
        [(z, fz), (z, f_zk), (z, f_zaad)],
        [
            "z, fz (AAD ord.)" + str(order),
            "Codpy ord." + str(order),
            "Pytorch ord." + str(order),
        ],
    )
    if D == 1:
        multi_plot(
            list_results,
            plot1D,
            mp_max_items=len(list_results),
            mp_ncols=3,
            f_names=f_names,
            mp_nrows=1,
            mp_figsize=(12, 4),
        )
    else:
        multi_plot(
            list_results,
            plot_trisurf,
            projection="3d",
            mp_ncols=4,
            mp_max_items=len(list_results),
            f_names=f_names,
            elev=25,
            azim=-80,
            linewidth=0.2,
            antialiased=True,
            mp_nrows=1,
        )


# # TEST repro AAD
# differentialMlBenchmarks(D=1,N=500)
# plt.show()

# # Test Repro taylor
# taylor_test(**get_param21(),taylor_order = 2)
# plt.show()

# params = predict_prices()
# params["graphic"](params)
# plt.show()

# params = predict_greeks()
# params["graphic"](nabla=codpy_nabla_corrected, **params)
# plt.show()
# pass
