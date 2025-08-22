
"""
====================================================================================================
9.14 Crypto management portfolio
====================================================================================================
Utilitary functions can be found next to this file. Here, we only define codpy-related functions.
"""

#########################################################################
# Necessary Imports
# ------------------------
#########################################################################
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from codpy.kernel import Sampler, Kernel, kernel_setter
from codpy.plot_utils import multi_plot, compare_plot_lists
from codpy.data_conversion import get_float, get_matrix
from codpy.conditioning import ConditionerKernel 
from codpy.file import files_indir 
from codpy.utils import get_closest_index 

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
data_path = os.path.join(PARENT_DIR, "data")
sys.path.insert(0, PARENT_DIR)

from utils.ch9.plot_utils import plot_trajectories, hist_plot, scatter_hist, multi_plot_figs, multi_plot
from utils.ch9.heston import get_model_param
import utils.ch9.mapping as maps
from utils.ch9.data_utils import get_datetime, csv_to_np 
from utils.ch9.path_generation import raw_data_generator, RegenerateHistory
from utils.ch9.market_data import interpolate_nulls, ts_data

cryptofigs_path = os.path.join(PARENT_DIR,"cryptoFigs")

def get_cryptoconditionner_map(**kwargs):
    kwargs['coeff'] = 1.
    return maps.composition_map([cryptoConditioner_map(kwargs) ,maps.pct_change()])



def get_crypto_params():
    kwargs = {
    'rescale_kernel':{'max': 1000, 'seed':42},
    'discrepancy:xmax':500,
    'discrepancy:ymax':500,
    'discrepancy:zmax':500,
    'discrepancy:nmax':1000,
    # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, 0,1e-8,map_setters.set_unitcube_map),
    'set_codpy_kernel': kernel_setter("maternnorm", "standardmean", 0, 0),
    # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_absnorm_kernel, 0,0 ,map_setters.set_unitcube_map),
    # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 1,1e-8,map_setters.set_unitcube_map),
    # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel, 0,1e-8 ,map_setters.set_unitcube_map),
    # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 0,0 ,map_setters.set_standard_mean_map),
    # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8 ,map_setters.set_standard_min_map),
    # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2,1e-8 ,map_setters.set_standard_min_map),
    'rescale':True,
    'grid_projection':True,
    'seed':43,
    'date_format' : '%d/%m/%Y %h:%m:%s',
    'freq':24,
    # 'Dx':3,
    'latent_dim':20,
    'reproductibility':False,
    'iter':10,
    'Denoiser':None,
    # "estimate_returns":[get_data(),get_standard_returns(),get_crypto_returns(),get_capm_returns()],
    "estimate_returns":[get_crypto_returns()],
    "strats" : [get_efficient_portfolio],
    # "strats" : [get_equiweighted_portfolio,get_efficient_portfolio,get_efficient_portfolio,get_efficient_portfolio],
    # "estimate_returns":[get_standard_returns(),get_capm_returns()],
    # "strats" : [get_efficient_portfolio,get_efficient_portfolio],
    "weights_file_path" : os.path.join(get_crypto_path(),"weights_file.pkl"),
    "strat_names":["Sharpe 1D","Long Short","Cond. crypto","Cond. CAPM"],
    "benchmark_pic_path" : os.path.join(cryptofigs_path,"benchmark.png"),
    "weights_pic_path" : os.path.join(cryptofigs_path,"weights.png"),
    "transaction_costs":0.0000
    }
    return kwargs

def get_crypto_params_transaction_costs():
    kwargs = {**get_crypto_params(),
    "benchmark_pic_path" : os.path.join(cryptofigs_path,"benchmark_with_tc.png"),
    "weights_pic_path" : os.path.join(cryptofigs_path,"weights_with_tc.png"),
    "transaction_costs":0.0007
    }
    return kwargs

def get_crypto_params_transation_strategy():
    kwargs = {**get_crypto_params_transaction_costs(),
    "estimate_returns":[get_data(),get_standard_returns()],#,get_crypto_returns(),get_capm_returns()],
    "strats" : [get_equiweighted_portfolio,get_efficient_portfolio_with_transation_costs],#,get_efficient_portfolio_with_transation_costs,get_efficient_portfolio_with_transation_costs],
    "strat_names":["Index","Long Short (LS)"],#,"LS-Cond liq","LS-Cond CAPM"],
    "weights_file_path" : os.path.join(get_crypto_path(),"weights_file_tc.pkl"),
    "benchmark_pic_path" : os.path.join(cryptofigs_path,"benchmark_tc_strat.png"),
    "weights_pic_path" : os.path.join(cryptofigs_path,"weights_tc_strat.png")
    }
    return kwargs


def get_yahoo_params(kwargs=None):
    if kwargs is None: kwargs = crypto_generator.get_data()

    data = kwargs['data']
    times = get_datetime(list(data.index))
    global_param = {
        'symbols' : ['^GSPC'],
        'begin_date' : times[0].strftime('%d/%m/%Y'),
        'end_date' :  times[-1].strftime('%d/%m/%Y'),
        'date_format' :'%d/%m/%Y',
        'yahoo_date_format': '%Y-%m-%d',            
    }

    yf_param = {
        'symbols':global_param["symbols"],
        'begin_date':global_param.get("begin_date",None),
        'end_date':global_param.get("end_date",None),
        'yf_begin_date': global_param.get("begin_date",None),
        'yahoo_columns': ['Close'],
        'date_format' : global_param["date_format"],
        'yahoo_date_format': '%Y-%m-%d',            
        'csv_date_format': '%d/%m/%Y',            
        'csv_file' : os.path.join(data_path,'-'.join(global_param["symbols"])+'-'+global_param["begin_date"].replace('/','-')+"-"+global_param["end_date"].replace('/','-')+".csv"),
    }      
    return yf_param


def get_crypto_path(**kwargs):
    return os.path.join(data_path,"crypto")
def get_crypto_data_dir_path(**kwargs):
    return os.path.join(get_crypto_path(**kwargs),"data")
def get_crypto_data_path(**kwargs):
    return os.path.join(get_crypto_path(**kwargs),"crypto_data.gz")
def get_crypto_values_path(**kwargs):
    return os.path.join(get_crypto_path(**kwargs),"crypto_values.gz")
def get_values_path(**kwargs):
    return os.path.join(get_crypto_path(**kwargs),"values.csv")
def get_data_path(**kwargs):
    return os.path.join(get_crypto_path(**kwargs),"data.csv")
def get_weights_path(**kwargs):
    # return os.path.join(get_crypto_path(**kwargs),"weights_file_test.pkl")
    return os.path.join(get_crypto_path(**kwargs),"weights_file.pkl")

def test_df(df):
    debug = list(df.index)
    for d in debug:
        if debug.count(d) > 1: 
            print(d,",datetime:",get_datetime(d))
    pass


class crypto_generator(raw_data_generator):
    def transform(csv,**kwargs):
        values_ = csv_to_np(csv)
        out = kwargs['map'](values_,kwargs)[1]
        out = out.reshape(out.shape[0]*out.shape[1],out.shape[2])
        if len(csv.columns) == out.shape[0]: out = pd.DataFrame(out.T,columns = csv.columns)
        else: out = pd.DataFrame(out.T)
        return out
    def change_extension(filename,extension=".csv"):
        split_tup = os.path.splitext(filename)
        print(split_tup)
        filepath = split_tup[0]
        file_extension_debug = split_tup[1]
        return filepath+extension
    def get_file_paths(**kwargs):
        extension = kwargs.get('extension',".csv")
        path = kwargs.get('path',get_crypto_data_dir_path(**kwargs))
        file_paths = files_indir(path,extension=extension)
        return file_paths

    def get_raw_data(kwargs=None):
        if kwargs is None: kwargs = get_crypto_params()
        crypto_data_path = kwargs.get('crypto_data_path',get_crypto_data_path(**kwargs))
        crypto_values_path = kwargs.get('crypto_values_path',get_crypto_values_path(**kwargs))
        if not os.path.isfile(crypto_data_path) or not os.path.isfile(crypto_values_path):
            crypto_generator.merge_csv(crypto_data_path,crypto_values_path,**kwargs)
        returns =  pd.read_csv(crypto_data_path,sep=";",index_col=0)
        values =  pd.read_csv(crypto_values_path,sep=";",index_col=0,compression='gzip')
        return returns,values
    def get_data(kwargs=None):
        from sklearn.model_selection import train_test_split
        if kwargs is None: kwargs = get_crypto_params()

        def filter_return(data):
            if "index" in data: 
                data.drop(columns="index",inplace=True)
            date_beg = get_float(kwargs.get("date_beg",data.index[0]),**kwargs)
            date_end = get_float(kwargs.get("date_end",data.index[-1]),**kwargs)
            freq = kwargs.get("freq",1)
            mask = [data.index[i] >= date_beg and data.index[i] <= date_end and i%freq == 0 for i in range(len(data.index))]
            data = data[mask]
            data.dropna(axis=1,inplace=True)
            data = data[np.logical_not(data.index.duplicated())]
            data.to_csv(get_data_path(),sep=";")
            return data
            return values
        def filter_col_values(values,prices):
            def filter(col):
                test = values[col].unique()
                if len(test) <=1 or col.split(":")[0] not in prices: 
                    values.drop(columns=col,inplace=True)
            [filter(col) for col in values]
            return values
        def filter_values(values,prices):
            values = filter_col_values(values,prices)
            nan = float("nan")
            values = values.replace(0.,nan)
            values = interpolate_nulls(values)
            mask = [ind in prices.index for ind in values.index]
            values = values[mask]
            values = values.shift(1).dropna()
            values = values[np.logical_not(values.index.duplicated())]
            values.to_csv(get_values_path(),sep=";")
            return values
        
        data_path,values_path = kwargs.get('data_file_path',get_data_path(**kwargs)),kwargs.get('values_file_path',get_values_path(**kwargs))
        if not os.path.isfile(data_path) or not os.path.isfile(values_path):
            prices,values = crypto_generator.get_raw_data(kwargs)
            prices = filter_return(prices)
            values  = filter_values(values,prices)
        prices,values =  pd.read_csv(data_path,sep=";",index_col=0),pd.read_csv(values_path,sep=";",index_col=0)
        times = values.index #should be the intersection of both
        out = {**kwargs,**{
            'data':prices,
            'returns':prices.pct_change().dropna(),
            'values':values,
            'times':times,
            "transform":crypto_generator.transform}}
        # return out
        map_ = get_cryptoconditionner_map(**out)
        return {**out,**{'map':map_}}

    class transform_pkl:
        def __init__(self,**kwargs):
            pkl_file_paths = crypto_generator.get_file_paths(**kwargs,extension=".pkl")
            crypto_generator.transform_pkl.save_csv_file(pkl_file_paths,**kwargs)
        def get_pkl_file_data(file_paths,**kwargs):
            if isinstance(file_paths,list):return[crypto_generator.transform_pkl.get_pkl_file_data(f,**kwargs) for f in file_paths]
            return pd.read_pickle(file_paths)
        
        def save_csv_file(pkl_file_paths,**kwargs):
            if isinstance(pkl_file_paths,list):return[crypto_generator.transform_pkl.save_csv_file(f,**kwargs) for f in pkl_file_paths]
            if os.path.isfile(pkl_file_paths): 
                csv_path = crypto_generator.change_extension(pkl_file_paths)
                if not os.path.isfile(csv_path): 
                    crypto_generator.transform_pkl.get_pkl_file_data(pkl_file_paths,**kwargs).to_csv(csv_path,sep = ';')
    

    def transform_csv(pds,**kwargs):
        if isinstance(pds,list):return[crypto_generator.transform_csv(p,**kwargs) for p in pds]
        times = get_float(pds.index,**kwargs)
        # for i,d in enumerate(times): # possible problem due to hour change on 26 March
        #     if times.count(d) > 1: 
        #         print("i:",i,",index:",pds.index[i],",times:",get_float(pds.index[i],**kwargs),",datetime:",get_datetime(d))

        out = pd.DataFrame(pds.values,index = times,columns = pds.columns)
        out['index'] = pds.index
        # test_df(out)

        return out
    def get_struct(df,**kwargs):
        fz=np.expand_dims(df.values.T, axis=0)
        fz = df.values
        fz=np.array([fz[:,[i]].T for i in range(fz.shape[1])])
        return fz
    
    def graphic(df,**kwargs):
        fz=crypto_generator.get_struct(df)
        plot_trajectories({'fx':fz,'fz':fz,'data':df,'plot_indice':[0]})
    def save_data_csv(crypto_data_path,returns):
        ncols = returns.select_dtypes(include=float).columns.tolist()
        returns[ncols] +=1.
        returns[ncols] = returns[ncols].cumprod(axis='index')
        returns.to_csv(crypto_data_path,sep = ';',compression='gzip')
        crypto_generator.graphic(returns)
    def save_cond_csv(crypto_values_path,conds):
        def filter(c,cond):
            if c == "index": take=False
            else:
                test = cond[c].unique()
                if len(test) <= 1: take=False
                else:take=True
            if take==False: cond.drop(columns=c,inplace=True)
            return take

        def get_columns(i,cond): return [c+":"+str(i) for c in list(cond.columns) if filter(c,cond)],cond
        def helper(i,cond): 
            columns,cond = get_columns(i,cond)
            out= pd.DataFrame(cond.values, columns = columns, index = cond.index)
            # test_df(out)
            return out
        conds = [helper(i,c) for i,c in enumerate(conds)]
        merged = pd.concat(conds, axis=1, ignore_index = False)

        # merged = pd.concat(conds,ignore_index=False)
        merged.to_csv(crypto_values_path,sep = ';',compression='gzip')
    def save_csv(crypto_data_path,crypto_values_path,**kwargs):
        pkl_file_paths = crypto_generator.get_file_paths(**kwargs,extension=".pkl")
        out = crypto_generator.transform_pkl.get_pkl_file_data(pkl_file_paths,**kwargs)
        out = crypto_generator.transform_csv(out,**kwargs)
        returns,conds = out[-1],out[:-1]
        # conds.index = conds.index.shift(-1,freq='D')
        # for c in conds: test_df(c)

        crypto_generator.save_data_csv(crypto_data_path,returns)
        crypto_generator.save_cond_csv(crypto_values_path,conds)
    def merge_csv(crypto_data_path,crypto_values_path,**kwargs):
        if not os.path.isfile(crypto_data_path) or not os.path.isfile(crypto_values_path):
            crypto_generator.save_csv(crypto_data_path,crypto_values_path,**kwargs)
        data,values =  pd.read_csv(crypto_data_path,sep=";",index_col=0,compression='gzip'),pd.read_csv(crypto_values_path,sep=";",index_col=0,compression='gzip')
        if "index" in data.columns: 
            data.drop(columns=["index"],inplace=True)
        if "index" in values.columns: 
            values.drop(columns=["index"],inplace=True)
        return data,values

from functools import cache
@cache
def cache_normals(N,D):
    kernel.init(**get_crypto_params())
    out = cd.alg.get_normals(N = N,D = D,nmax = 100)
    return out

# class cryptoConditioner(maps.QuantileConditioner):
#     def __init__(self,**kwargs):
#         self.x,self.y = get_matrix(kwargs['x'].copy()),get_matrix(kwargs['y'].copy())
#         self.latent_x,self.latent_y, self.latent = None,None,None
#         self.y_original = self.y.copy()
#         # self.Denoiser = op.Denoiser(**{**kwargs,**{'x':self.x,'fx':self.y}})
#         self.Denoiser = kwargs.get('Denoiser',self.get_denoiser())
#         if self.Denoiser is not None:
#             self.fx = self.Denoiser(z=self.x)
#             self.y -= self.fx
#         self.latent_x,self.latent_y = self.get_latent(**{**kwargs,**{'cond_val':None,'N':None}})
#         self.encoder_y = gen.encoder(**{**kwargs,**{'x':self.latent,'y':self.y,'reordering':self.reordering,'permut':'source'}})
#         permutation = self.encoder_y.permutation
#         # test = self.encoder_y.operator.x - latent[permutation]
#         # test = self.encoder_y.operator.x - self.encoder_y.latent
#         self.latent = self.encoder_y.latent
#         # self.latent_x,self.latent_y = self.latent_x[permutation],self.latent_y[permutation]
#         self.latent_x,self.latent_y = self.latent[:,:self.latent_x.shape[1]],self.latent[:,self.latent_y.shape[1]:]
#         self.encoder_x = gen.encoder(**{**kwargs,**{'x':self.latent_x,'y':self.x,'reordering':self.reordering,'permut':'target'}})

#         params= self.encoder_x.params.copy()
#         # self.permutation_x = map_invertion(self.encoder_x.permutation, type_in = np.ndarray)
#         # params['x'],params['y'],params['fx']= params['fx'][self.permutation_x],params['fx'][self.permutation_x],params['x'][self.permutation_x]
#         params['x'],params['y'],params['fx'] = self.x,self.x,self.latent_x
#         self.decoder_x = op.Cache(**params)

##check reproductibility        
        # test = self.decoder_x.fx - self.latent_x
        # test = self.decoder_x(z=self.x)
        # error = test - self.latent_x
        # test =  np.concatenate([test,self.latent_y],axis=1)
        # error = test - self.latent
        # test = self.encoder_y(z=test)
        # error = test -  self.y
    #     pass
    # def __call__(self, **kwargs):
    #     self.x_cond,N,z = get_matrix(kwargs['x']),kwargs.get('N',None),kwargs.get('z',None)
    #     x = self.decoder_x(z=self.x_cond)
    #     z = np.concatenate([x,z],axis=1)
    #     out = self.encoder_y(z=z)
    #     if self.Denoiser is not None:
    #         debug = self.Denoiser(z=self.x_cond) 
    #         out += debug
    #     return out
    # def get_latent(self,**kwargs) : # you can pick any random variable as latent variable
    #     if self.latent_x is None:
    #         latent_dim = kwargs.get('latent_dim',self.x.shape[1])
    #         if latent_dim < self.x.shape[1]:
    #             self.latent_x = cache_normals(self.x.shape[0],latent_dim)
    #             # np.random.shuffle(self.latent_x)
    #         else: 
    #             self.latent_x = self.x.copy()
    #     if self.latent_y is None:
    #         latent_dim = self.latent_x.shape[1]
    #         if latent_dim < self.y.shape[1]:
    #             self.latent_y = cache_normals(self.y.shape[0],latent_dim).copy()
    #         else: 
    #             self.latent_y = self.y.copy()
    #         np.random.seed(42)
    #         np.random.shuffle(self.latent_y)
    #     if self.latent is None: self.latent = np.concatenate([self.latent_x,self.latent_y],axis=1)
    #     self.reordering = kwargs.get('reordering',True)
    #     return self.latent_x.copy(),self.latent_y.copy()

   
class cryptoConditioner_map:
    def __init__(self,kwargs = None):
        pass
    def __call__(self, y,kw):
        data = kw["data"]
        # x is the "expert knowledge" and y is the noise
        # we want to condition y on x____
        x,dist = get_matrix(kw['values'].loc[data.index]),y.reshape([y.shape[0]*y.shape[1],y.shape[2]]).T
        # TODO: to replace with the Conditioner on Codpy - see conditionnermap

        latent_generator_y = kw.get('latent_generator_y',None)
        self.conditionner = ConditionerKernel(x=x, y=dist, latent_generator_y=latent_generator_y) 
        self.conditionner.set_maps()

        latent_noise = self.conditionner.sampler_xy.get_x()[:, x.shape[1]:]
        latent_noise = latent_noise.T
        latent_noise = latent_noise.reshape(y.shape)

        # Reproductibility test
        # if kw.get('reproductibility', False):
        #     xy = self.conditionner.sampler_xy(self.conditionner.sampler_xy.get_x())
        #     shape = (y.shape[0], self.x.shape[1]+y.shape[1], y.shape[2])
        #     xy = (xy.T).reshape(shape)
        #     xy_og = np.concatenate([x, z], axis=1)
        #     xy_og = (xy_og.T).reshape(shape)
        #     assert np.allclose(xy, xy_og), "The sampled xy does not match"

        return np.expand_dims(x.T, axis=0), latent_noise



        self.conditionner = cryptoConditioner(**{**kw,**{'x':x,'y':dist}})
        x,latent_y = self.conditionner.x,self.conditionner.latent_y
        return  np.expand_dims(x.T, axis=0),np.expand_dims(latent_y.T, axis=0)
        # return  np.expand_dims(latent_y.T, axis=0)

class inv_cryptoConditioner_map:
    def __init__(self,call_back):
        self.call_back = call_back
    def __call__(self, y,kw):
        x,latent_y = y[0],y[1]
        def helper(i) :
            x_cond,z = x[...,i],latent_y[...,i]
            x_cond = np.concatenate([x_cond for n in range(z.shape[0])],axis=0)
            out = self.call_back.conditionner(**{**kw,**{'x':x_cond,'z':z}})
            return np.expand_dims(out,axis=-1)
        
        out = np.concatenate([helper(i) for i in range(latent_y.shape[2])],axis=-1)
        return out


maps.inverse_class_switchDict["cryptoConditioner_map"] =   inv_cryptoConditioner_map

def generated_paths(params = None):
    if params is None: params = get_model_param()
    params['fz'],params['data'],params['fx']=RegenerateHistory(**params)
    def graphic(params):
        params['Nz'] = params.get('Nz',10)
        D = params['fz'].shape[1]
        # mp_nrows = int(np.sqrt(D))
        mp_nrows = 5
        plot_indice = list(range(1,mp_nrows*mp_nrows))
        plot_trajectories({**params,**{'plot_indice':plot_indice,'mp_nrows':mp_nrows,'mp_ncols':mp_nrows,'mp_figsize':(10,10)}})
        
    params['graphic'] = graphic
    # params['stats_mean'] = stats_df(params['data'], np.mean(f_z, axis=0).T[:,1:]).T
    # plt.plot(f_z[0,1,:])
    # plt.plot(params['data'].values[:,0])
    # 
    return params

def get_generated_param(kwargs = None):
    if kwargs is None: kwargs = get_model_param()
    def graphic(params):
        transform_h = params['transform_h']
        transform_g = params['transform_g']
        dim = transform_h.shape[1]
        
        def fun_plot(param,**kwargs) :
            import io
            from PIL import Image
            if dim == 1: return hist_plot([param],ax=kwargs['ax'])
            symbols = params.get('symbols',["first","second"])
            labelx,labely=symbols[0],symbols[1]
            fig = scatter_hist(param[:,0], param[:,1],figsize=(3,3), labelx=labelx,labely = labely,**kwargs)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            kwargs['ax'].imshow(Image.open(img_buf))
            plt.close(fig)
            pass
        fig = multi_plot_figs([transform_h.values,transform_g.values],mp_nrows=1,mp_figsize=(6,3),fun_plot = fun_plot, mp_ncols = 2, **params) 
        
        # else : 

    if 'transform_h' not in kwargs:
        kwargs['transform_h'] = kwargs['transform'](kwargs['data'],**kwargs)
    sample_times = kwargs.get("timelist",kwargs['times'])
    Nz = kwargs.get('Nz',10)
    sampler = Sampler(x=kwargs['transform_h'].values, **kwargs)
    transform_g = sampler.sample(N=Nz*len(sample_times))
    # transform_g = gen.sampler(**{**kwargs,'y':kwargs['transform_h']})(N=Nz*len(sample_times))
    transform_g = pd.DataFrame(transform_g,columns = kwargs['transform_h'].columns)
    kwargs['transform_g'] = transform_g
    kwargs['graphic'] = graphic

    return kwargs

def plot_correlation(kwargs=None):
    from pypfopt import risk_models

    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.plotting import plot_covariance
    if kwargs is None: kwargs = crypto_generator.get_data()
    data = kwargs["data"]
    mu = mean_historical_return(data)
    S = CovarianceShrinkage(data).ledoit_wolf()
    S = risk_models.cov_to_corr(S)
    fig, ax = plt.subplots(figsize=(3, 3))
    cax = ax.imshow(S)
    fig.colorbar(cax)

    # plot_covariance(S, plot_correlation=True, show_tickers=False, figsize=(2,2))
    pass

def plot_riskvsexpectation(kwargs=None):
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.plotting import plot_covariance
    if kwargs is None: kwargs = crypto_generator.get_data()
    data = kwargs["data"].pct_change()
    mu,sigma = data.mean(),data.std()
    weights = kwargs["weights"]
    plt.scatter(mu, sigma,alpha = 0.5)
    plt.xlabel('Expected returns')
    plt.ylabel('Risk')
    plt.title("Risk vs. Expected Returns")
    for label, x, y in zip(range(data.shape[1]), mu, sigma):
        plt.annotate(
            str(label), 
            xy = (x, y), xytext = (0, 0),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
    pass


def plot_efficient_frontier (kwargs=None):
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.plotting import plot_efficient_frontier
    if kwargs is None: kwargs = plot_efficient_portfolio()
    ef = get_efficient_frontier(kwargs)
    ef_max_sharpe = EfficientFrontier(ef.mu, ef.sigma)
    data = kwargs["data"]
    columns= list(data.columns)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_efficient_frontier(ef_max_sharpe, ax=ax, show_assets=True)
    weights = ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance(risk_free_rate=0.0)
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    def helper(key):
        return columns.index(key)
    opt = [helper(key) for key in weights.keys()]

    sigma =kwargs["data"].pct_change().std()
    mu,sigma = ef.EfficientFrontier_.expected_returns,np.sqrt(np.diag(ef.EfficientFrontier_.cov_matrix))
    ax.scatter(sigma[opt], mu[opt], marker="*", s=50, c="y", label="Eff. port.")
    ax.scatter(sigma, mu, marker="*", s=50, c="y", label="Eff. port.")
    for label, x, y in zip(range(data.shape[1]), sigma, mu):
        ax.annotate(
            str(label), 
            xy = (x, y), xytext = (x,y),color="purple",
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0.', color="black"))
    ax.set_title("Efficient Frontier with assets")
    ax.legend()
    # plt.show()
    pass

class efficient_frontier:
    def __init__(self,**kwargs):
        from pypfopt.efficient_frontier import EfficientFrontier
        if kwargs is None: kwargs = crypto_generator.get_data()
        self.sigma = self.get_sigma(**kwargs)
        self.mu = self.get_mu(**kwargs)
        self.EfficientFrontier_= EfficientFrontier(self.mu, self.sigma,weight_bounds=(-1,1))
    def get_sigma(self,**kwargs):
        from pypfopt.risk_models import CovarianceShrinkage
        data,returns_data,frequency,compounding = kwargs["returns"], kwargs.get("returns_data",False),kwargs.get("frequency",1),kwargs.get("compounding",True)
        res = CovarianceShrinkage(prices=data,returns_data=returns_data,frequency=frequency)
        res = res.ledoit_wolf()
        return res
    def get_mu(self,**kwargs):
        from pypfopt.expected_returns import mean_historical_return
        data,returns_data,frequency,compounding = kwargs["returns"], kwargs.get("returns_data",False),kwargs.get("frequency",1),kwargs.get("compounding",True)
        return mean_historical_return(data,returns_data=returns_data,compounding=compounding,frequency=frequency)

    def __call__(self,**kwargs):
        data = kwargs["data"]
        target_volatility = kwargs.get("target_volatility",np.sqrt(self.sigma.values.mean()))
        weights = self.EfficientFrontier_.efficient_risk(target_volatility=target_volatility, market_neutral = True)
        out = {}
        def helper(i,w):out[data.columns[i]] = w
        [helper(i,w) for i,w in enumerate(weights)]
        return weights
  
class efficient_frontier_with_transation_costs(efficient_frontier):
    def __init__(self,**kwargs):
        from pypfopt.objective_functions import transaction_cost
        super().__init__(**kwargs)
        weight = kwargs["weight"]
        if not weight.empty:
            sample_times = kwargs["sample_times"]
            key = min(weight.index, key = lambda key: abs(key-sample_times[0]))
            self.EfficientFrontier_.add_objective(transaction_cost, w_prev = weight.loc[key], k=kwargs.get("transaction_costs",0.0007))

def get_efficient_frontier(kwargs=None):return efficient_frontier(**kwargs)
def get_efficient_frontier_with_transation_costs(kwargs=None):return efficient_frontier_with_transation_costs(**kwargs)


def get_index(kwargs = None):
    if kwargs is None: kwargs = crypto_generator.get_data()
    times = kwargs['times']
    data = kwargs["data"]
    data /= data.iloc[0]
    return data.mean(axis=1)

class get_crypto_returns:
    def __init__(self):
        self.mat = None

    def get_params(self,kwargs):
        return {**kwargs,**{'returns_data':True,"compounding":True,"frequency":1}}
        # return {**kwargs,**{'returns_data':True}}


    def __call__(self,kwargs = None):
        if kwargs is None: kwargs = crypto_generator.get_data()
        data = kwargs["data"]
        # last_times = data.index[-1]
        times = kwargs["times"]
        sample_times = kwargs["sample_times"]
        index = get_closest_index(times,sample_times)
        def testing(mat=None):
            if mat is None:
                if self.mat is None:
                    self.mat = np.random.normal(size=data.shape)
                    self.mat = pd.DataFrame(self.mat,columns=data.columns)
                mat = self.mat.copy()
            out = mat-np.mean(mat,axis=0)
            weights = np.array(data.iloc[index[1]] / data.iloc[index[0]]-1.)
            def helper(i):out[data.columns[i]] += weights[i]
            [helper(i) for i in range(out.shape[1])]
            return pd.DataFrame(out,columns=data.columns)
        # return testing()
        samples = get_matrix(data).T[np.newaxis,...] # Normalized underlying values
        cond,latent_y = kwargs["map"](samples,kwargs) # corresponding conditionned noise
        # if index[-1] >= latent_y.shape[-1]:
        #     raise IndexError('list index out of range')
        values = np.expand_dims(latent_y[0].T,axis=-1)
        values = np.concatenate([values,values],axis=2)
        # cond_values = cond[...,index]
        cond_values = np.expand_dims(kwargs["values"].iloc[index].T,axis=0)

        # values = values.reshape((values.shape[2], values.shape[1], values.shape[0]))
        out = maps.inverse(kwargs["map"])([cond_values,values],kwargs)
        # out = inverse(kwargs["map"])([cond_values,latent_y],kwargs)
        out = out[...,1]/out[...,0] - 1.
        out = pd.DataFrame(out,columns=data.columns)
        # mat = np.random.normal(size=data.shape)
        # mat += out.values-np.mean(mat,axis=0)
        # return pd.DataFrame(mat,columns=data.columns)
        return out
        return testing(out)

class get_capm_returns(get_crypto_returns):
    def __init__(self):
        self.mat = None
    def get_capm(self,index, data):
        set_codpy_kernel = kernel_setter("linear_regressor", "scale_to_unitcube", 2, 1e-8)
        ker = Kernel(
            x=get_matrix(index.pct_change().dropna()),
            fx=get_matrix(data.pct_change().dropna()),
            set_kernel=set_codpy_kernel
        )
        out = ker.grad(z=get_matrix(index.pct_change().dropna())).mean(axis=0)
        # out = op.nabla(x=index.pct_change().dropna(),y=index.pct_change().dropna(),z=index.pct_change().dropna(),fx=data.pct_change().dropna(),set_codpy_kernel=set_codpy_kernel,rescale=True).mean(axis=0)
        return out


    def __call__(self,kwargs = None):
        if kwargs is None: kwargs = crypto_generator.get_data()
        index = get_index(kwargs)
        data = kwargs["data"]
        windows = kwargs.get("windows",300)
        beg = kwargs.get("beg",0)
        end = beg+windows
        values = []
        bound = int(not kwargs.get("reproductibility",False))
        while end <  kwargs["data"].shape[0]:
            values.append(pd.DataFrame(self.get_capm(index.iloc[beg:end],data.iloc[beg:end])[-1,:]).T)
            beg,end = beg+1,end+1
        values = pd.concat(values,axis=0)
        values.columns = data.columns
        index_ = get_float(data.index)
        values.index = index_[kwargs.get("beg",0)+windows:end]
        # if bound > 0 : data = data.loc[values.index[:-bound]]
        # else: data = kwargs["data"].loc[values.index]
        return super().__call__({**kwargs,**{"data":data.loc[values.index],
                                             "values":values,
                                             "times" : get_float(values.index)
                                             }})

class get_standard_returns(get_crypto_returns):
    def __call__(self,kwargs = None):
        if kwargs is None: kwargs = crypto_generator.get_data()
        out = kwargs['data'].pct_change().dropna()
        # bound = int(not kwargs.get("reproductibility",False))
        # if bound > 0:return out.iloc[:-bound]
        return out

class get_data(get_crypto_returns):
    def get_params(self,kwargs):
        return {**kwargs,**{'returns_data':False,"compounding":True,"frequency":1}}
    def __call__(self,kwargs = None):
        if kwargs is None: kwargs = crypto_generator.get_data()
        out = kwargs['data'].pct_change().dropna()
        bound = int(not kwargs.get("reproductibility",False))
        if bound > 0:return out.iloc[:-bound]
        return out

def get_efficient_portfolio (kwargs=None):
    if kwargs is None: kwargs = crypto_generator.get_data()
    # kwargs["returns"]=kwargs["data"]
    weights = dict(get_efficient_frontier(kwargs)(**kwargs))
    return weights

def get_efficient_portfolio_with_transation_costs(kwargs=None):
    if kwargs is None: kwargs = crypto_generator.get_data()
    # kwargs["returns"]=kwargs["data"]
    weights = dict(get_efficient_frontier_with_transation_costs(kwargs)(**kwargs))
    return weights


def test_strategy (kwargs=None):
    if kwargs is None: kwargs = crypto_generator.get_data()
    data = kwargs["data"]
    times = kwargs["times"]
    sample_times = kwargs["sample_times"]
    index = get_closest_index(times,sample_times)
    weights = np.array(data.iloc[index[1]] / data.iloc[index[0]]-1.)
    out = {}
    def helper(i,w):out[data.columns[i]] = w
    [helper(i,w) for i,w in enumerate(weights)]
    return weights


def get_equiweighted_portfolio (kwargs=None):
    import ffn
    from pypfopt.expected_returns import mean_historical_return
    data = kwargs["data"].to_log_returns().dropna()
    data = data.calc_mean_var_weights()
    # print(data[abs(data > 1e-8)])
    weights = {}
    for key in data.index: weights[key] = data[key] #
    # if N%2 != 0: 
    #     weights[mu.index[mid+1]]=0.
    return weights


def filtered_weights(weights):
    def helper(pair):
        return np.abs(pair[1]) > 0.01
    filtered_weight = dict(filter(helper,weights.items()))
    return filtered_weight


def plot_efficient_portfolio (kwargs=None):
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.plotting import plot_weights
    if kwargs is None: kwargs = crypto_generator.get_data()
    weights = get_efficient_portfolio(kwargs)
    filtered_weight = filtered_weights(weights)
    fig, ax = plt.subplots(figsize=(4, 2))
    plot_weights(filtered_weight, ax=ax, show_assets=True)
    ax.set_title("Optimal Portfolio for Sharpe Ratio")
    plt.show()
    return {**kwargs,"weights":weights}

def get_windows(beg, end,kwargs=None):
    if kwargs is None: kwargs = crypto_generator.get_data()
    returns,values =  kwargs["data"],kwargs["values"]
    returns,values = returns.iloc[beg:end],values.iloc[beg:end]
    times = returns.index
    out = {**kwargs,**{'data':returns,'values':values,'times':times}}
    return out
    pass


def plot_weights(weights,**kwargs):

    def plot_helper(weight,**kwargs):
        ax= kwargs['ax']
        means = weight.mean()
        index = np.argsort(np.array(np.abs(means.values)))[-10:]
        cols = weight.columns[index]
        times = list(weight.index)
        date_times = get_datetime(times)

        means = means.values[index]
        weight = weight.values[:,index]
        print(weight)
        weight = pd.DataFrame(weight, index = date_times, columns=cols)
        weight.plot(ax=ax)
        plt.legend(loc='upper left', title='weights',fontsize=4)

    def get_list(weights,**kwargs):
        return weights
    
    multi_plot(get_list(weights),fun_plot = plot_helper,**kwargs)
    if kwargs.get("path_to_figure",False):
        plt.savefig(kwargs["path_to_figure"])


def back_test(weights,**kwargs):
    def get_reference_indices(results):
        test = get_yahoo_params()
        csv = ts_data.get_yf_ts_data(**test)
        csv_index = get_float(get_datetime(list(csv.index)))
        time_index = get_float(results[0].index)
        csv_index = get_closest_index(csv_index,time_index)
        csv = csv.iloc[csv_index]
        csv.index = results[0].index
        csv = csv.diff(axis=0)
        csv = (csv+1.).cumprod(axis=0)
        return csv
    def get_reference_indices(results):
        csv= pd.DataFrame(get_index(kwargs),columns=["index"])
        csv_index = get_float(list(csv.index))
        time_index = get_float(results[0].index)
        csv_index = get_closest_index(csv_index,time_index)
        csv = csv.iloc[csv_index]
        csv.index = results[0].index
        csv = csv.diff(axis=0)
        csv = (csv+1.).cumprod(axis=0)
        return csv
        

    if isinstance(weights,list):
        strat_names = kwargs.get("strat_names",None)
        if strat_names is None: strat_names = ["strat:"+str(n) for n in range(len(weights))]
        results = [back_test(weight,**{**kwargs,**{"strat_names":strat_name}}) for weight,strat_name in zip(weights,strat_names)]
        # results.append(get_reference_indices(results))
        values = pd.concat(results,axis=1)
        values.plot(fontsize=15)
        if kwargs.get("path_to_figure",False):
            plt.savefig(kwargs["path_to_figure"])
        return values

    data,times = kwargs["data"],list(kwargs["times"])
    pos_change = abs(weights.sub(weights.shift(1), fill_value=0))
    # test = data.iloc[index-1:]
    # data.plot(legend=False)
    # test.plot(legend=False)
    strat_names = kwargs.get("strat_names","ptf_values")
    values = pd.DataFrame(columns=[strat_names])
    cost_rate = kwargs.get("transaction_costs",0.0007)
    for row in weights.iterrows():
        time_ = row[0]
        date_time = get_datetime(time_)
        index = times.index(time_)
        weight = row[1]
        if index+1 < data.shape[0] and index >= 0:
            costs = 0.
            time_next = times[index+1]
            keys = list(weight.index)
            PnL = np.array(weight*(data.loc[time_next][keys]-data.loc[time_][keys]))
            if cost_rate is not None: 
                costs = cost_rate*pos_change.loc[time_].sum()
            values.loc[date_time] = PnL.sum()-costs
        pass
    values = (values+1.).cumprod(axis=0)
    return values


def ffn_test(kwargs = None):
    import ffn
    if kwargs is None: kwargs = crypto_generator.get_data()
    returns_csvdata_path,values_csvdata_path = kwargs.get('returns_file_path',get_data_path(**kwargs)),kwargs.get('values_file_path',get_values_path(**kwargs))
    data = kwargs["data"]
    data.index = get_datetime(list(data.index))
    returns = data.to_log_returns().dropna()
    perf = data.calc_stats()
    perf.plot()
    print(perf.display())
    perf["underlying0"].display_monthly_returns()        
    pass


def get_weights(kwargs=None):    
    import pickle
    if kwargs is None: kwargs = crypto_generator.get_data()
    weights_file_path = kwargs.get("weights_file_path",get_weights_path(**kwargs))
    times = list(kwargs["times"])
    if not os.path.exists(weights_file_path) or "test" in weights_file_path:
        windows = kwargs.get("windows",.95)
        weights = []
        estimate_returns = kwargs.get("estimate_returns",[get_data(),get_standard_returns(),get_crypto_returns(),get_crypto_returns(),get_capm_returns()])
        strats = kwargs.get("strats",[get_equiweighted_portfolio,get_efficient_portfolio,get_efficient_portfolio,get_efficient_portfolio_with_transation_costs,get_efficient_portfolio])
        for estimate_return,strat in zip(estimate_returns,strats):
            weight = pd.DataFrame(columns=kwargs["data"].columns)
            beg = kwargs.get("beg",0)
            end=int(kwargs["data"].shape[0]*windows)+beg
            params = estimate_return.get_params(kwargs)
            while end <  kwargs["data"].shape[0]-1:
                local_time, sample_times = times[beg:end+1], times[end-1:end+1]
                data = kwargs["data"].loc[local_time] # We filter on the window
                values = kwargs["values"].loc[local_time] # Same here for the conditionned values
                params = {**params,**{"times":local_time,"weight":weight,"sample_times":sample_times,"data":data,"values":values}}                
                params["returns"] = estimate_return(params)
                w = strat(params)
                # try :
                #     params["returns"] = estimate_return(params)
                #     w = strat(params)
                # except Exception as error:
                #     print("allocation failed, weights nullified:",error)
                #     w={}
                #     def helper(col): w[col]= 0.
                #     [helper(col) for col in params["data"].columns]
                
                # print(filtered_weights(w))
                print("time:",get_datetime(sample_times[-1]),"portfolio:",np.sum(list(w.values())))
                weight.loc[sample_times[0]] = w
                beg,end = beg+1,end+1
            weights.append(weight)

        with open(weights_file_path, 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    weights = []
    with (open(weights_file_path, "rb")) as openfile:
        while True:
            try:
                weights.append(pickle.load(openfile))
            except EOFError:
                break    
    return weights[0]

def benchmark(kwargs=None):    
    if kwargs is None: kwargs = crypto_generator.get_data()
    benchmark_pic_path = kwargs.get("benchmark_pic_path",os.path.join(cryptofigs_path,"benchmark.png"))
    weights_pic_path = kwargs.get("weights_pic_path",os.path.join(cryptofigs_path,"weights.png"))
    weights = get_weights(kwargs)
    plot_weights(weights,path_to_figure=weights_pic_path,**kwargs)
    ptf_values = back_test(weights,path_to_figure=benchmark_pic_path,**kwargs)
    import matplotlib.image as mpimg
    img1 = mpimg.imread(benchmark_pic_path)
    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(18.5, 10.5)
    ax1.imshow(img1)
    ax1.axis('off')
    pass



def sandbox(kwargs=None):    
    if kwargs is None: kwargs = crypto_generator.get_data()
    windows = kwargs.get("windows",.8)
    beg = kwargs.get("beg",0)
    end=int(kwargs["data"].shape[0]*windows)+beg
    weights = pd.DataFrame(columns=kwargs["data"].columns)
    while end <  kwargs["data"].shape[0]:
        returns,values = kwargs["data"].iloc[beg:end],kwargs["values"].iloc[beg:end]
        times = kwargs["data"].index
        # returns = get_standard_returns({**kwargs,**{'data':returns,'values':values,'times':times}})
        returns = get_crypto_returns(**{**kwargs,**{'data':returns,'values':values,'times':list(times)}})
        w = get_efficient_portfolio({**kwargs,**{'data':returns}})
        print(filtered_weights(w))
        weights.loc[times[end]] = w
        beg,end = beg+1,end+1
    ptf_values = back_test([weights],**kwargs)
    print(ptf_values)
    crypto_generator.graphic(kwargs["data"])
    plot_correlation(kwargs)
    out = get_generated_param(get_model_param({**kwargs,**{'map' : maps.composition_map([maps.diff(),maps.log_map,maps.remove_time()])}}))
    out["graphic"](out)
    out = generated_paths({**get_model_param(out),**{'reproductibility':True}} )
    out["graphic"](out)
    out = generated_paths({**get_model_param(out),**{'reproductibility':False}} )
    out["graphic"](out)
    pass

if __name__ == "__main__":
    # out = crypto_generator.get_data()
    # crypto_generator.graphic(out["data"])

    benchmark()

    # NOT WORKING    
    # out = plot_efficient_portfolio()
    # out = generated_paths({**get_model_param(out),**{'reproductibility':False}} )
    # out["graphic"](out)
    # plt.show()

    # NOT WORKING
    # err0 = np.random.random(size = [1,4,4])
    # print(err0)
    # pct_change_ = maps.pct_change()
    # err = pct_change_(err0,{})
    # print(err)
    # err = maps.inv_pct_change(pct_change_)(y=err)-err0

    # ffn_test()

    # out = crypto_generator.get_data()
    # kw = get_generated_param(get_model_param({**out,**{'map' : maps.composition_map([maps.diff(),maps.log_map,maps.remove_time()])}}))
    # kw["graphic"](kw)
    # plt.show()

    # NOT WORKING    
    # out = crypto_generator.get_data()
    # out = get_generated_param(get_model_param({**out,**{'map' : maps.composition_map([cryptoConditioner_map(out),maps.pct_change()])}}))

    # NOT WORKING
    # out = generated_paths({**get_model_param(crypto_generator.get_data()),**{'reproductibility':False}} )
    # out["graphic"](out)
    # plt.show()

    # NOT WORKING
    # benchmark(get_crypto_params())
    
    # NOT WORKING
    # plot_efficient_frontier()
    
    # benchmark(crypto_generator.get_data(get_crypto_params_transaction_costs()))
    # benchmark(crypto_generator.get_data(get_crypto_params_transation_strategy()))

    # NOT WORKING
    # plot_correlation()
    # out = plot_efficient_portfolio()
    # plot_efficient_frontier(out)
    # out = plot_efficient_frontier()
    pass
