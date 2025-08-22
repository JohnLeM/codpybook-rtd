import abc
from cmath import isinf
import numpy as np
import scipy.stats as sps
import os 
import sys 
import pandas as pd 
import copy 

# import openturns as ot

import codpy.AAD as AAD
from codpy import core 

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

PARENT_DIR = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, PARENT_DIR)

class FD:
    def nabla(x,fun,set_fun=None,**kwargs):
        import numdifftools.nd_scipy as nd
        if isinstance(x,list): return [FD.nabla(x=x[n],fun=fun,**kwargs) for n in range(0,x.shape[0])]
        if set_fun is None: set_fun = lambda x,**k : k
        if x.ndim == 2 :
            return np.concatenate([FD.nabla(x=x[n],fun=fun,set_fun=set_fun,**kwargs) for n in range(0,x.shape[0])], axis = 0)
        def lambda_helper(x,**kwargs):
            k = set_fun(x=x,**kwargs)
            out = fun(None, **k)
            return out
        out = nd.Gradient(lambda_helper)(x,**kwargs)
        # print("grad:",get_matrix(out).T)
        return core.get_matrix(out).T

    def hessian(x,fun,**kwargs):
        import numdifftools.nd_scipy as nd
        # if fun == None: fun = option_param.price
        if isinstance(x,list): return [FD.hessian(x=x[n],fun=fun,**kwargs) for n in range(0,len(x))]
        if x.ndim == 2 :
            out = np.array([FD.hessian(x=x[n],fun=fun,**kwargs) for n in range(0,x.shape[0])])
            return out
        def hessian_helper(x,**kwargs):
            out = FD.nabla(x=x,fun=fun,**kwargs)
            out = np.squeeze(out)
            return out

        out = nd.Jacobian(hessian_helper)(x=x,**kwargs)
        # print("grad:",get_matrix(out).T)
        return out.T

    def nablas(fun, x,**kwargs):
        copy_kwargs = copy.deepcopy(kwargs)
        # copy_kwargs = kwargs.copy()
        def helper(v): return FD.nabla(fun = fun, x = v,**copy_kwargs)
        if isinstance(x,list): out = [helper(v) for v in core.get_matrix(x)]
        elif isinstance(x,np.ndarray):
            if x.ndim == 1: return helper(x)
            out = [helper(x[n]) for n in range(0,x.shape[0]) ]
        elif isinstance(x,pd.DataFrame): return FD.nablas(fun=fun,x= x.values,**copy_kwargs)
        out = np.array(out).reshape(values.shape)
        return out

    def hessians(fun, x,**kwargs):
        copy_kwargs = copy.deepcopy(kwargs)
        # copy_kwargs = kwargs.copy()
        def helper(v): return FD.hessian(fun = fun, x = v,**copy_kwargs)
        if isinstance(x,list): out = [helper(v) for v in core.get_matrix(x)]
        elif isinstance(x,np.ndarray):
            if x.ndim == 1: return helper(x)
            out = [helper(x[n]) for n in range(0,x.shape[0]) ]
        elif isinstance(x,pd.DataFrame): return FD.hessians(fun=fun,x= x.values,**copy_kwargs)
        out = np.array(out).reshape(x.shape[0],x.shape[1],x.shape[1])
        return out
    
class StochasticProcess(abc.ABC):
    @abc.abstractmethod
    def get_process(self,**kwargs):
        pass


class path_generator(abc.ABC):
    def __init__(self,**kwargs):
        pass
    def get_param(**kwargs): return kwargs.get('path_generator')
    @abc.abstractmethod
    def generate(self,N,payoff,**kwargs):
        pass

    def get_process(**kwargs) : return path_generator.get_param(**kwargs).get('process')
    def get_D(**kwargs) : return path_generator.get_process(**kwargs).factors() #.get_process(**kwargs) enlevé



class payoff(abc.ABC):
    @abc.abstractmethod
    def f(self,x,**kwargs):
        pass
    @abc.abstractmethod
    def get_times(self,**kwargs):
        pass
    def __call__(self,x,**kwargs):return self.f(x,**kwargs)

class pricer(abc.ABC):
    @abc.abstractmethod
    def price(self,**kwargs):pass
    def nabla(self,**kwargs):return FD.nabla(fun = self.price, **kwargs)
    def hessian(self,**kwargs):return FD.hessian(fun = self.price, **kwargs)

    def nablas(self, values,**kwargs): 
        out = get_matrix(FD.nablas(x = values,fun = self.price, **kwargs))
        return out

    def hessians(self, values,**kwargs): 
        # out = FD.hessians(fun = self.price,x = values,**kwargs)
        out = AAD.AAD.hessian(fx = self.price,x = values,**kwargs)
        return out
    def prices(self,set_fun, values,**kwargs): 
        copy_kwargs = kwargs.copy()
        values = values.copy() #??
        def helper(v,**k): 
            k = set_fun(v,**k)
            return self.price(None, **k) 
        if isinstance(values,list): out = [helper(v,**copy_kwargs) for v in get_matrix(values)]
        elif isinstance(values,np.ndarray): out = [helper(values[n],**copy_kwargs) for n in range(0,values.shape[0]) ]
        elif isinstance(values,pd.DataFrame): return self.prices(set_fun, values.values,**copy_kwargs)
        out = np.array(out)

        return out
    def pnls(self,set_fun, x,z,**kwargs): 
        left = self.prices(set_fun, x,**kwargs)
        right = self.prices(set_fun, z,**kwargs)
        pnls = right[:min(len(left),len(right))] - left[:min(len(left),len(right))]

        # from QL_tools import BasketOptionClosedFormula
        # spot = BasketOptionClosedFormula.get_spot_basket(x= z[:,1:],**kwargs)
        # spot,toto,p = lexicographical_permutation(spot,pnls.copy())
        # plt.plot(spot,toto)
        # 

        # grad = op.nabla(x=x[:,:,0], y=x[:,:,0], z=x[:,:,0], fx=pnls, rescale = True,**kwargs)

        return pnls

class MonteCarloPricer(pricer):

    def get_params(**kwargs) :  return kwargs.get('MonteCarloPricer',{})
    def get_N(**kwargs): return MonteCarloPricer.get_params(**kwargs)['N']

    def price(self,payoff,generator,**kwargs):
        initial_values = kwargs['getter'].get_spot(**kwargs)
        x=generator.generate(payoff=payoff,initial_values = initial_values,**kwargs)
        # if type(kwargs['getter'])== type(qlib.Heston_param_getter()): y=payoff.f(x) 
        # else: y=payoff(x,**kwargs)
        y=payoff(x,**kwargs)
        out = np.mean(y)
        if kwargs.get('Stats',False)==True:# Confidence Interval : 95%
            var = np.var(y, ddof=1)
            quantile = sps.norm.ppf(1 - (1-0.95)/2)
            ci_size = quantile * np.sqrt(var / y.size)    
            result = pd.DataFrame(
                [out, var, out - ci_size, out + ci_size], 
                index=("Mean","Var","Lower bound", "Upper bound"), 
                columns=['MC'])
            # result = [out, var, [round(out - ci_size,6), round(out + ci_size,6)]]
            return result
        else: return out

class trivial_path_generator(path_generator):
    
    def generate(self,payoff,time_list=None,**kwargs):
        paths_keys = kwargs.get('paths_key','paths')
        return kwargs[paths_keys]

class trivial_last_path_generator(path_generator):
    
    def generate(self,payoff,time_list=None,**kwargs):
        paths_keys = kwargs.get('paths_key','paths')
        out = kwargs[paths_keys]
        return out[...,-1]

class historical_path_generator(path_generator):

    def generate_from_samples(self,**kwargs):
        def generate_from_samples_dataframe(**kwargs):
            samples = kwargs['samples']
            pd_samples = np.ndarray((1,samples.shape[1]+1,samples.shape[0]))
            pd_samples[0,0,:] = get_float(samples.index)
            pd_samples[0,1:,:] = samples.values.T
            out = generate_from_samples_np(**kwargs)
            out = pd.DataFrame(out[:,1:,0],columns = samples.columns,index = out[:,0,0])
            return out
        
        def generate_from_samples_np(**kwargs):
            out = generate_distribution_from_samples_np(kwargs)
            mapping = kwargs.get("map",None)
            if mapping is not None: 
                out = inverse(mapping)(out,kwargs)
            return out
            
        import pandas 
        samples = kwargs['samples']
        generate_from_samples_switchDict = { pandas.core.frame.DataFrame: generate_from_samples_dataframe }
        type_debug = type(samples)
        method = generate_from_samples_switchDict.get(type_debug,generate_from_samples_np)
        return method(**kwargs)

    def generate(self,payoff,time_list=None,**kwargs):
        historical_generator = kwargs.get("historical_generator",None)
        samples = historical_generator.generate(**kwargs)
        if time_list is None: time_list = payoff.get_times(**kwargs)
        # return historical_generator.generate(**kwargs,time_list=time_list)
        return self.generate_from_samples(samples=samples,sample_times=time_list,**kwargs)



class Recurrent_historical_path_generator(historical_path_generator):
    def generate_from_samples(self,samples,sample_times,**kwargs):
        mapping = kwargs.get("map",None)
        if mapping is not None: 
            mapped_samples=np.zeros((samples.shape[0],samples.shape[1],samples.shape[2]-1))
            time_list = list(set(samples[:,0,:].flatten()))
            time_list.sort()
            mapped_samples= mapping(samples,**kwargs, times = time_list)
            x,fx=ts_format(mapped_samples,**kwargs)
        else: x,fx = ts_format(samples,**kwargs)

        return super().generate_from_samples(samples,sample_times,x=x,fx=fx,**kwargs)

def remove_mean(A):
    mean = np.mean(A[:,1,:])
    A[:,1,:] = A[:,1,:] - mean
    return A,mean

def split(A,h):
    out=np.zeros((int(A.shape[2]/h),A.shape[1],h))
    for i in range(int(A.shape[2]/h)):
        out[i,:,:]=A[0,:,i*h:(i+1)*h]
    return out

