import os
import sys
import copy 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codpy.data_conversion import get_date, get_float
from codpy.kernel import get_matrix

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

PARENT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
data_path = os.path.join(PARENT_DIR, "data")
if not os.path.exists(data_path):
    os.makedirs(data_path)
PARENT_DIR = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, PARENT_DIR)

import utils.ch9.mapping as maps
from utils.ch9.data_utils import df_summary, get_time
from utils.ch9.ql_tools import Heston
import utils.ch9.plot_utils as plot_utils

DEFAULT_ARGS = {
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


def interpolate(x, fx, z, **kwargs):
    from scipy import interpolate

    x, fx, z = get_float(x), get_float(fx), get_float(z)
    if len(x) == 1:
        x.append(x[0] + 1.0), fx.append(fx[0])
    return interpolate.interp1d(x, fx, **kwargs)(z)


def interpolate_nulls(data, **kwargs):
    kind = str(kwargs.get("kind", "linear"))
    bounds_error = bool(kwargs.get("bounds_error", False))
    copy = bool(kwargs.get("copy", False))
    var_col = kwargs.get("var_col", None)
    float_fun = kwargs.get("float_fun", None)

    nulls = [col for col in data.columns if data[col].isnull().sum()]
    for col in nulls:
        fx = data.loc[data[col].notnull()][col].values
        if var_col is None:
            x = data.loc[data[col].notnull()].index.values
            z = data.index.values
        else:
            x = data.loc[data[col].notnull()][var_col].values
            z = data[var_col].values
        if float_fun is not None:
            x, z = float_fun(x), float_fun(z)
        data[col] = interpolate(
            x,
            fx,
            z,
            kind=kind,
            bounds_error=bounds_error,
            fill_value=(fx[0], fx[-1]),
            copy=copy,
        )
        pass
    return data


def retrieve_market_data(kwargs=None):
    if kwargs is None:
        kwargs = DEFAULT_ARGS
    symbols = kwargs["symbols"]
    begin_date = kwargs["begin_date"]
    end_date = kwargs["end_date"]
    date_format = kwargs["date_format"]
    params = {
        "yf_param": {
            "symbols": symbols,
            "begin_date": begin_date,
            "end_date": end_date,
            "yf_begin_date": begin_date,
            "yahoo_columns": ["Close"],
            "date_format": date_format,
            "yahoo_date_format": "%Y-%m-%d",
            "csv_date_format": date_format,
            "csv_file": os.path.join(
                data_path,
                "-".join(symbols)
                + "-"
                + begin_date.replace("/", "-")
                + "-"
                + end_date.replace("/", "-")
                + ".csv",
            ),
        },
    }

    params["data"] = ts_data.get_yf_ts_data(**params["yf_param"])
    params["table1"] = df_summary(params["data"])
    params["times"] = get_time(params["data"])

    def graphic(params):
        params["data"].plot(rot=20, fontsize="10")
        plt.legend(list(params["data"].columns), fontsize="10")

    params["graphic"] = graphic

    return {**params, **kwargs}


class ts_data:
    def get_param(**kwargs):
        sep = kwargs.get("sep", ";")
        csv_file = kwargs.get("csv_file", None)
        begin_date = kwargs.get("begin_date", None)
        end_date = kwargs.get("end_date", None)
        date_format = kwargs.get("date_format", "%d/%m/%Y")
        csv_date_format = kwargs.get("csv_date_format", "%m/%d/%Y %H:%M")
        time_col = str(kwargs.get("time_col", "Date"))
        select_columns = kwargs.get("select_columns", None)
        return (
            sep,
            csv_file,
            begin_date,
            end_date,
            date_format,
            csv_date_format,
            time_col,
            select_columns,
        )

    def get_sep(**kwargs):
        return kwargs.get("sep", ";")

    def get_csv_file(**kwargs):
        return kwargs.get("csv_file", None)

    def get_begin_date(**kwargs):
        return kwargs.get("begin_date", None)

    def get_end_date(**kwargs):
        return kwargs.get("end_date", None)

    def get_date_format(**kwargs):
        return kwargs.get("csv_format", "%d/%m/%Y")

    def get_csv_date_format(**kwargs):
        return kwargs.get("csv_date_format", "%m/%d/%Y %H:%M")

    def get_time_col(**kwargs):
        return kwargs.get("time_col", "Date")

    def get_select_columns(**kwargs):
        return kwargs.get("select_columns", None)

    def get_sep(**kwargs):
        return kwargs.get("sep", ";")

    def get_yf_ts_data(**kwargs):
        data = ts_data.get_csv_ts_data(**kwargs)
        if data is not None :
            return data

        import yfinance as yf

        (
            sep,
            csv_file,
            begin_date,
            end_date,
            date_format,
            csv_date_format,
            time_col,
            select_columns,
        ) = ts_data.get_param(**kwargs)

        symbols = kwargs.get(
            "symbols", AssertionError("get_yf_data: symbols must be input")
        )
        yahoo_date_format = kwargs.get("yahoo_date_format", "%Y-%m-%d")
        # yf_symbols = " ".join(symbols)
        yf_begin_date = get_date(
            kwargs.get("yf_begin_date"), date_format=date_format
        ).strftime(yahoo_date_format)
        yahoo_columns = kwargs.get("yahoo_columns", None)

        data = yf.download(symbols, yf_begin_date)
        if yahoo_columns is not None:
            data = data[yahoo_columns]
        if len(symbols) > 1:
            data = data.droplevel(level=0, axis=1)

        index = [
            get_date(x, date_format=csv_date_format).strftime(date_format)
            for x in data.index
        ]
        data = pd.DataFrame(data=data.values, columns=data.columns, index=index)

        if begin_date is not None:
            begin_date = get_float(begin_date, date_format=date_format)
            data = data.loc[
                [
                    (get_float(x, date_format=date_format) >= begin_date)
                    for x in data.index
                ]
            ]
        if end_date is not None:
            end_date = get_float(end_date, date_format=date_format)
            data = data.loc[
                [
                    (get_float(x, date_format=date_format) <= end_date)
                    for x in data.index
                ]
            ]

        if len(symbols) == 1:
            data.rename(columns={"Close": symbols[0]}, inplace=True)
        if csv_file is not None:
            data.to_csv(csv_file, sep=sep, index=True)

        return data

    def get_csv_ts_data(**kwargs):
        (
            sep,
            csv_file,
            begin_date,
            end_date,
            date_format,
            csv_date_format,
            time_col,
            select_columns,
        ) = ts_data.get_param(**kwargs)
        if csv_file is not None and os.path.exists(csv_file):
            data = pd.read_csv(csv_file, sep=sep, index_col=0)
        else:
            return None

        # Convert the index to datetime format
        data.index = pd.to_datetime(data.index, format=csv_date_format)

        # Filter by date range
        if begin_date is not None:
            begin_date = pd.to_datetime(begin_date, format=date_format)
            data = data[data.index >= begin_date]
        if end_date is not None:
            end_date = pd.to_datetime(end_date, format=date_format)
            data = data[data.index <= end_date]

        # Convert index to desired output format
        data.index = data.index.strftime(date_format)

        return data

    def interpolate(data, **kwargs):
        return interpolate_nulls(data, **kwargs)


def calibrate_Heston(kw):
    times = kw["times"]
    # times=(times - times[0])/365.
    x = get_matrix(kw["data"]).T
    spot_price = x[0, 0]
    X = maps.composition_map([maps.diff(), maps.log_map])(x, {**kw, "times": times})
    factor = 252.0
    mu = np.mean(X) * factor
    # CIR = diff()(x[1,:],{**kw,'times':times})
    # sigma=np.std(CIR)*np.sqrt(times)
    # kappa=np.mean(CIR)*factor
    kw["Heston"]["risk_free_rate"] = mu
    kw["Heston"]["spot_price"] = spot_price
    kw["path_generator"] = {"process": Heston().get_process(**kw)}
    return kw

class scenario_generator:
    gpa=[]
    results =[]
    data_generator,predictor,accumulator = [],[],[]
    def set_data(self,data_generator,predictor,accumulator,**kwargs):
        self.gpa.append((data_generator,predictor,accumulator))
    def __init__(self,data_generator = None,predictor= None,accumulator= None,**kwargs):
        if data_generator:self.set_data(data_generator,predictor,accumulator,**kwargs)
    def run_scenarios_cube(self,Ds, Nxs,Nys,Nzs,**kwargs):
        for d in Ds:
            for nx in Nxs:
                for ny in Nys:
                    for nz in Nzs:
                        self.data_generator.set_data(int(d),int(nx),int(ny),int(nz),**kwargs)
                        self.predictor.set_data(self.data_generator,**kwargs)
                        print("  predictor:",self.predictor.id()," d:", d," nx:",nx," ny:",ny," nz:",nz)
                        self.accumulator.accumulate(self.predictor,self.data_generator,**kwargs)
    def run_scenarios(self,list_scenarios,data_generator,predictor,accumulator,**kwargs):
        for scenario in list_scenarios:
            self.data_generator,self.predictor,self.accumulator = data_generator,predictor,accumulator
            data_generator.set_data(**{**scenario,**kwargs})
            predictor.set_data(data_generator,**scenario,**kwargs)
            # print("predictor:",self.predictor.id()," d:", d," nx:",nx," ny:",ny," nz:",nz)
            accumulator.accumulate(predictor,data_generator,**kwargs)
        if not len(self.results): self.results = accumulator.get_output_datas()
        else: self.results = pd.concat((self.results,accumulator.get_output_datas()))
        # print(self.results)
    def run_all(self,list_scenarios,**kwargs):
        self.results = []
        for scenario in list_scenarios:
            d,nx,ny,nz = scenario
            d,nx,ny,nz = int(d),int(nx),int(ny),int(nz)
            # print(" d:", d," nx:",nx," ny:",ny," nz:",nz)
            for data_generator,predictor,accumulator in self.gpa:
                run_scenarios(self,list_scenarios,data_generator,predictor,accumulator,**kwargs)

    def compare_plot(self,axis_label, field_label, **kwargs):
        xs=[]
        fxs=[]
        # self.results.to_excel("results.xlsx")
        groups = self.results.groupby("predictor_id")
        predictor_ids = list(groups.groups.keys())
        groups = groups[(axis_label,field_label)]
        for name, group in groups:
            xs.append(group[axis_label].values.astype(float))
            fxs.append(group[field_label].values.astype(float))
            pass
        plot_utils.compare_plot_lists(listxs = xs, listfxs = fxs, listlabels=predictor_ids, xscale ="linear",yscale ="linear", Show = True,**kwargs)

    def compare_plot_ax(self,axis_field_label, ax,**kwargs):
        xs=[]
        fxs=[]
        axis_label,field_label = axis_field_label[0],axis_field_label[1]
        # self.results.to_excel("results.xlsx")
        groups = self.results.groupby("predictor_id")
        predictor_ids = list(groups.groups.keys())
        groups = groups[list((axis_label,field_label))]
        for name, group in groups:
            xs.append(group[axis_label].values.astype(float))
            fxs.append(group[field_label].values.astype(float))
        pass
        plot_utils.compare_plot_lists({'listxs' : xs, 'listfxs' : fxs, 'ax':ax,'listlabels':predictor_ids, 'labelx':axis_label,'labely':field_label,**kwargs}
        )


    def compare_plots(self,axis_field_labels, **kwargs):
        multi_plot(axis_field_labels,self.compare_plot_ax, **kwargs)


class ts_scenario_generator(scenario_generator):
    def run_scenarios(self,list_scenarios,accumulator):
        for scenario in list_scenarios:
            data_generator,predictor = scenario.get("generator",None),scenario.get("predictor",None)
            self.data_generator,self.predictor,self.accumulator = data_generator,predictor,accumulator
            data_generator.set_data(**scenario)
            predictor.set_data(**scenario)
            # print("predictor:",self.predictor.id()," d:", d," nx:",nx," ny:",ny," nz:",nz)
            accumulator.accumulate(**scenario)
        if not len(self.results): self.results = accumulator.get_output_datas()
        else: self.results = pd.concat((self.results,accumulator.get_output_datas()))


    def plot_output(self,**kwargs):
        results = [{"fz":generator.format_output(predictor.fz,**kwargs),"f_z":generator.format_output(predictor.f_z,**kwargs),**kwargs} for predictor,generator in zip(self.accumulator.predictors,self.accumulator.generators)]
        results = [{"fz":generator.format_output(predictor.fz,**kwargs),"f_z":generator.format_output(predictor.f_z,**kwargs),**kwargs} for predictor,generator in zip(self.accumulator.predictors,self.accumulator.generators)]
        listxs, listfxs = {},{}
        fixed_columns =  kwargs.get('plot_columns',list(results[-1]["fz"].columns))
        for result in results :
            fz,f_z = result["fz"],result["f_z"]
            plot_columns =  kwargs.get('plot_columns',list(fz.columns))
            xs = [list(fz.index),list(f_z.index)]

            for fixed_col,col in zip(fixed_columns,plot_columns):
                if fixed_col not in listxs.keys():listxs[fixed_col]=[]
                if fixed_col not in listfxs.keys():listfxs[fixed_col]=[]
                fzcol,f_zcol = list(fz[col]),list(f_z[col])
                [listxs[fixed_col].append(x) for x in xs]
                listfxs[fixed_col].append(list(fz[col].values)), listfxs[fixed_col].append(list(f_z[col].values))

        for key in listxs.keys():
            xs,fxs = listxs[key], listfxs[key]
            plot_utils.compare_plot_lists(xs, fxs, ax = None, labely = key, **kwargs)


    def plot_output(self,**kwargs):
        listxs, listfxs = {},{}
        results = kwargs.get('results',[])
        for result in results :
            fz = result
            plot_columns =  kwargs.get('plot_columns',list(fz.columns))

            for col in plot_columns:
                if col not in listxs.keys():listxs[col]=[]
                if col not in listfxs.keys():listfxs[col]=[]
                listxs[col].append(list(fz.index))
                listfxs[col].append(list(fz[col].values))

        for key in listxs.keys():
            xs,fxs = listxs[key], listfxs[key]
            plot_utils.compare_plot_lists(xs, fxs, ax = None, labely = key, **kwargs)

class data_accumulator:
    def __init__(self,**kwargs):
        self.set_data(generators = [],predictors= [],**kwargs)

    def set_data(self,generators =[], predictors = [],**kwargs):
        self.generators,  self.predictors = generators,predictors

    def accumulate_reshape_helper(self,x):
        if len(x) == 0: return []
        newshape = np.insert(x.shape,0,1)
        return x.reshape(newshape)

    def accumulate(self,predictor,generator,**kwargs):
        self.generators.append(copy.copy(generator))
        self.predictors.append(copy.copy(predictor))


    def plot_learning_and_train_sets(self,xs=[],zs=[],title="training (red) vs test (green) variables and values",labelx='variables ',labely=' values'):
        D = len(self.generators)
        fig = plt.figure()
        d=0
        for d in range(0,D):
            g = self.generators[d]
            if (len(xs)): x = xs[d]
            else: x = self.generators[d].x[:,0]
            ax=fig.add_subplot(1,D,d+1)
            plotx,plotfx,permutation = lexicographical_permutation(x.flatten(),g.fx.flatten())
            ax.plot(plotx,plotfx,color = 'red')
            if (len(zs)): z = zs[d]
            else: z = self.generators[d].z[:,0]
            plotz,plotfz,permutation = lexicographical_permutation(z.flatten(),g.fz.flatten())
            ax.plot(plotz,plotfz,color = 'green')
            plt.xlabel(labelx)
            plt.ylabel(labely+self.predictors[d].id())
            d = d+1
        plt.title(title)

    def plot_predicted_values(self,zs=[],title="predicted (red) vs test (green) variables and values",labelx='z',labely='predicted values'):
        d = 0
        D = len(self.predictors)
        fig = plt.figure()
        for d in range(0,D):
            p = self.predictors[d]
            if (len(zs)): z = zs[d]
            else: z = self.generators[d].z[:,0]
            ax=fig.add_subplot(1,D,d+1)
            plotx,plotfx,permutation = lexicographical_permutation(z.flatten(),p.f_z.flatten())
            ax.plot(plotx,plotfx,color = 'red')
            plotx,plotfx,permutation = lexicographical_permutation(z.flatten(),p.fz.flatten())
            ax.plot(plotx,plotfx,color = 'green')
            plt.xlabel(labelx)
            plt.ylabel(labely+p.id())
            d = d+1
        plt.title(title)

    def plot_errors(self,fzs=[],title="error on predicted set ",labelx='f(z)',labely='error:'):
        d = 0
        D = len(self.predictors)
        fig = plt.figure()
        for p in self.predictors:
            ax=fig.add_subplot(1,D,d+1)
            if (len(fzs)): fz = fzs[d]
            else: fz = p.fz
            plotx,plotfx,permutation = lexicographical_permutation(get_matrix(fz).flatten(),get_matrix(p.f_z).flatten()-get_matrix(p.fz).flatten())
            ax.plot(plotx,plotfx, color= "red")
            ax.plot(plotx,get_matrix(p.fz).flatten()-get_matrix(p.fz).flatten(), color= "green")
            plt.xlabel(labelx)
            plt.ylabel(labely+p.id())
            d = d+1

        plt.title(title)

    def format_helper(self,x):
        return x.reshape(len(x),1)
    def get_elapsed_predict_times(self):
        return  np.asarray([np.round(s.elapsed_predict_time,2) for s in self.predictors])
    def get_discrepancy_errors(self):
        return  np.asarray([np.round(s.discrepancy_error,4) for s in self.predictors])
    def get_norm_functions(self):
        return  np.asarray([np.round(s.norm_function,2) for s in self.predictors])
    def get_accuracy_score(self):
        return  np.asarray([np.round(s.accuracy_score,4) for s in self.predictors])
    def get_numbers(self):
        return  np.asarray([s.get_numbers() for s in self.predictors])
    def get_Nxs(self):
        return  np.asarray([s.Nx for s in self.predictors])
    def get_Nys(self):
        return  np.asarray([s.Ny for s in self.predictors])
    def get_Nzs(self):
        return  np.asarray([s.Nz for s in self.predictors])
    def get_predictor_ids(self):
        return  np.asarray([s.id() for s in self.predictors])
    def get_xs(self):
        return  [s.x for s in self.predictors]
    def get_ys(self):
        return  [s.y for s in self.predictors]
    def get_zs(self):
        return  [s.z for s in self.predictors]
    def get_fxs(self):
        return  [s.fx for s in self.predictors]
    def get_fys(self):
        return  [s.fy for s in self.predictors]
    def get_fzs(self):
        return  [s.fz for s in self.predictors]
    def get_f_zs(self):
        return  [s.f_z for s in self.predictors]

    def confusion_matrices(self):
        return  [s.confusion_matrix() for s in self.predictors]

    def plot_clusters(self, **kwargs):
        multi_plot(self.predictors ,graphical_cluster_utilities.plot_clusters, **kwargs)

    def plot_confusion_matrices(self, **kwargs):
        multi_plot(self.predictors ,add_confusion_matrix.plot_confusion_matrix, **kwargs)

    def get_maps_cluster_indices(self,cluster_indices=[],element_indices=[],**kwargs):
        out = []
        for predictor in self.predictors:
            out.append(predictor.get_map_cluster_indices(cluster_indices=cluster_indices,element_indices=element_indices,**kwargs))
        return out


    def get_output_datas(self):
        execution_time = self.format_helper(self.get_elapsed_predict_times())
        discrepancy_errors = self.format_helper(self.get_discrepancy_errors())
        norm_function = self.format_helper(self.get_norm_functions())
        scores = self.format_helper(self.get_accuracy_score())
        numbers = self.get_numbers()
        indices = self.format_helper(self.get_predictor_ids())
        indices = pd.DataFrame(data=indices,columns=["predictor_id"])
        numbers = np.concatenate((numbers,execution_time,scores,norm_function,discrepancy_errors), axis=1)
        numbers = pd.DataFrame(data=numbers,columns=["D", "Nx","Ny","Nz","Df","execution_time","scores","norm_function","discrepancy_errors"])
        numbers = pd.concat((indices,numbers),axis=1)
        return  numbers
