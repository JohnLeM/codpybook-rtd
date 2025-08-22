# utils1.py
# This file contains utility functions used in the Chapter 9 experiments.
# These are helper functions for data processing.
import os 
import sys 

import pandas as pd
import numpy as np
import abc
from typing import Any

from codpy.plot_utils import multi_plot, get_matrix
from codpydll import * 
from codpy.kernel import Kernel
from codpy.conditioning import ConditionerKernel
from codpy.permutation import map_invertion
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

PARENT_DIR = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, PARENT_DIR)

from utils.ch9.data_utils import csv_to_np
from utils.ch9.plot_utils import hist_plot, scatter_hist

def apply_map(kwargs = None):
    # if kwargs is None: kwargs = get_pricer_param()
    map_ = kwargs.get('map',composition_map([diff(),log_map,remove_time()]))
    params = {
        'map':map_,
        'seed':None,
    }
    def transform(csv,kwargs):
        values_ = csv_to_np(csv)
        out = kwargs['map'](values_,kwargs)
        out = out.reshape(out.shape[0]*out.shape[1],out.shape[2])
        if len(csv.columns) == out.shape[0]: out = pd.DataFrame(out.T,columns = csv.columns)
        else: out = pd.DataFrame(out.T)
        return out
    out = {**params,**kwargs}
    out['transform'] = transform
    out['transform_h'] = transform(out['data'],out)

    def graphic(params):
        transform_h = params['transform'](params['data'],**params)
        print("mean:",transform_h.mean()," cov:",transform_h.cov())

        dim = len(params['symbols'])
        if dim == 1: multi_plot([[transform_h.values[:,0]]],mp_figsize=(3,3),fun_plot = hist_plot, labelx=params['symbols'][0], mp_ncols = 1, **kwargs) 
        else : scatter_hist(transform_h.values[:,0], transform_h.values[:,1], labelx=params['symbols'][0],labely = params['symbols'][1],figsize=(5,5),**params)
        
    out['graphic'] = graphic
    return out

get_data_switchDict = { pd.DataFrame: lambda x :  x.values,
                        pd.Series: lambda x : np.array(x.array, dtype= 'float'),
                        tuple: lambda xs : [get_data(x) for x in xs],
                    }
def get_data(x):
    type_debug = type(x)
    method = get_data_switchDict.get(type_debug,lambda x: np.asarray(x,dtype='float'))
    return method(x)
    
def get_closest_index(mySortedList, myNumber):
    from bisect import bisect_left
    if isinstance(myNumber,list):return [get_closest_index(mySortedList,n) for n in myNumber]
    pos = bisect_left(mySortedList, myNumber)
    if pos == len(mySortedList):
        return pos-1
    return pos

def inverse(fun):
    import inspect
    import types
    if inspect.isclass(type(fun)) and not isinstance(fun, types.FunctionType):
        method = inverse_class_switchDict[fun.__class__.__name__]
        return method(fun)
    def default_inverse(fun):
        assert(False)
    method = inverse_fun_switchDict.get(fun.__name__,default_inverse)
    return method

def VanDerMonde(x,orders,**kwargs):
    x = get_matrix(x)
    if x.ndim==1: return cd.tools.VanDerMonde(x,orders)
    out = np.array([cd.tools.VanDerMonde(x[n],orders) for n in range(0,x.shape[0])])
    return out

def get_slice(my_array, dimension, index):
    items = [slice(None, None, None)] * my_array.ndim
    def helper(d,ind):
        items[d] = ind
    if isinstance(dimension,list): [helper(d,ind) for d,ind in zip(dimension,index)]
    else:helper(dimension,index)
    array_slice = my_array[tuple(items)]
    return array_slice

def ts_format_np(A,h,p=0, **kwargs):
    A = np.array(A)
    dim = A.ndim
    if dim == 1:
        (first,second) = cd.tools.ts_format(get_matrix(A),h,p)
        if p ==0: return first
        return (first,second)
    if dim == 2:
        if A.shape[0] > A.shape[1]: (first,second) = cd.tools.ts_format(get_matrix(A),h,p)
        else:
            (first,second) = cd.tools.ts_format(get_matrix(A.T),h,p)
            (first,second) = (first.T,second.T)
        if p ==0: return first
        return (first,second)
    along_axis = kwargs.get('along_axis',None)
    if along_axis is not None :
        out = [ts_format_np(get_slice(A,along_axis,ind),h,p) for ind in range(0,A.shape[along_axis])]
        if p == 0 : return np.array(out)
        return np.array([s[0] for s in out]),np.array([s[1] for s in out])

class composition_map2:
    def __init__(self,mapl,mapr):
        self.mapl,self.mapr=mapl,mapr
    def __call__(self, a=None,kw=None):
        if a is None and kw is None: return self.mapl(self.mapr())
        if a is None : return self.mapl(self.mapr(kw))
        if kw is None : return self.mapl(self.mapr(a))
        return self.mapl(self.mapr(a, kw),kw)   

class composition_map:
    def __init__(self,maps):
        if len(maps) == 1:self.maps = maps[0]
        else: 
            self.maps = composition_map2(maps[-2],maps[-1])
            def helper(map): 
                self.maps = composition_map2(map,self.maps)
            [helper(map) for map in reversed(maps[:-2])]
    def __call__(self, a=None,kw=None): 
        if a is None and kw is None: return self.maps()
        if a is None : return self.maps(kw)
        if kw is None : return self.maps(a)
        return self.maps(a,kw)


class inv_composition_map2:
    def __init__(self,comp_map):
        self.mapl,self.mapr=comp_map.mapl,comp_map.mapr
    def __call__(self, a,kw):
        return inverse(self.mapr)(inverse(self.mapl)(a, kw),kw)   

class inv_composition_map:
    def __init__(self,comp_map):
        self.maps = comp_map.maps
    def __call__(self, a,kw):
        return inverse(self.maps)(a,kw) 

class diff:
    def __init__(self,kwargs = None):
        if kwargs is not None: self.zero = kwargs.get('zero',True)
        else: self.zero = True

    def get_sample_times(kwargs):
        times = list(kwargs.get('times',None))
        sample_times = list(kwargs.get('sample_times',times))
        closest_index =get_closest_index(times,sample_times[0])
        sample_times.insert(0,times[closest_index])
        return sample_times,closest_index

    def __call__(self, x,kwargs):
        times,axis = list(kwargs.get('times',None)),kwargs.get("axis",-1)
        self.sample_times = list(kwargs.get('sample_times',times))
        self.closest_index =get_closest_index(times,self.sample_times[0])
        self.axis = kwargs.get("axis",-1)
        self.x = x.copy()
        self.ic = x[...,self.closest_index]
        if self.zero:
            out = np.zeros_like(x)
            out[...,:-1] = np.diff(x,axis = kwargs.get("axis",-1))
        else:
            out = np.diff(x,axis = axis)
        return out

class inv_diff:
    def __init__(self,call_back):
        self.call_back = call_back

    def one_step(self,x,y,kw):
        return x+y

    # def integrate(self,y,kw):
    #     ic = self.call_back.ic
    #     out = np.zeros([ic.shape[1],y.shape[1]])
    #     out[...,0] = kw.get('initial_conditions',ic)
    #     def one_step(n):
    #         if n > 0: out[...,[n]] = self.one_step(out[...,[n-1]],y[...,[n]],kw)
    #         else: 
    #             debug = self.one_step(ic.T,y[...,[n]],kw)
    #             out[...,[0]] = debug
    #         # error = out[...,n] - self.call_back.x[...,n]
    #         pass
    #     [one_step(n) for n in range(0,y.shape[1])]
    #     return out
    def integrate(self,y,kw):
        ic = self.call_back.ic
        out = np.zeros_like(y)
        out[...,0] = kw.get('initial_conditions',ic)
        def one_step(n):
            if n > 0: out[...,n] = self.one_step(out[...,n-1],y[...,n-1],{**kw,**{'n':n-1}})
            # else: 
            #     ic = np.array([self.call_back.ic for n in range(y.shape[0])]).reshape([y.shape[0],y.shape[1]])
            #     debug = self.one_step(ic,y[...,n],{**kw,**{'n':n}})
            #     out[...,0] = debug
            # error = out[...,n] - self.call_back.x[...,n] # reproductibility test for debug purposes
            pass
        [one_step(n) for n in range(0,y.shape[2])]
        return out

    def __call__(self, y,kw):
        y = get_data(y)
        # out = self.integrate(y,kw)
        assert y.ndim == 3, "input y must be formatted as a tensor N,D,T"
        out = self.integrate(y,kw) 
        return out



class random_variable_sum(diff):
    pass

class inv_random_variable_sum(inv_diff):
    def one_step(self,x,y,kw):
        out = alg.transported_sum(**{**kw,**{'x':x,'y':y}})
        print(y)
        print(out)
        return out
        # return x+y
    

    def __call__(self, y,kw):
        out = np.zeros([y.shape[0],self.call_back.ic.shape[1],y.shape[2]])
        out[...,0] = kw.get('initial_conditions',self.call_back.ic)
        def integrate(n):
            if n > 0: out[...,n] = self.one_step(out[...,n-1],y[...,n],kw)
            pass
        [integrate(n) for n in range(0,y.shape[2])]
        return out



class pct_change(diff):
    def __init__(self,kwargs = None):
        if kwargs is not None: self.zero = kwargs.get('zero',True)
        else: self.zero = True

    def __call__(self, x,kwargs):
        out = super().__call__(x,kwargs)
        out = out / self.x
        return out

class inv_pct_change(inv_diff):

    def __call__(self, y,kw):
        y = get_data(y)*self.call_back.x
        out = super().__call__(y,kw)
        return out

class mean_map(diff):
    def __call__(self, x,kwargs):
        self.mean = np.mean(x,axis=-1)
        out = x.copy()
        def helper(n):out[...,n] -= self.mean
        [ helper(n) for n in range(0,x.shape[2])]
        return out

class inv_mean_map(inv_diff):

    def __call__(self, y,kw):
        out = y
        def helper(n):out[...,n] += self.call_back.mean
        [ helper(n) for n in range(0,y.shape[2])]
        return out


class remove_time:
    def __call__(self, x,kwargs):
        self.times = x[:,[0],:]
        return x[:,1:,:]

class add_time : 
    def __init__(self,normal_return_):
        self.call_back = normal_return_
    def __call__(self, x,kwargs):
        out = np.ndarray([x.shape[0],x.shape[1]+1,x.shape[2]])
        out[:,1:,:] = x
        sample_times = kwargs.get('sample_times',None)
        if sample_times is None:
            out[:,0,:] = self.call_back.times[...,:out.shape[2]]
        else:
            out[:,0,:] = sample_times[:out.shape[2]]
        return out

class normal_return(diff):
    def __call__(self, x,kwargs):
        # Take the diff and then divide by sqrt of the time difference
        out = super().__call__(x,kwargs)
        times = np.array(kwargs.get('times',None))
        times = np.sqrt(np.diff(times))
        def helper(n): out[...,n] /= times[n]
        [helper(n) for n in range(0,out.shape[-1]-1)]
        return out

class inv_normal_return(inv_diff):
    def __init__(self,normal_return_):
        self.call_back = normal_return_

    def __call__(self, x,kwargs):
        sample_times = list(kwargs.get('sample_times',None))
        if sample_times is None:# or len(sample_times) != x.shape[2]: 
            sample_times = kwargs['times']
        times = np.array(kwargs.get('times',None))[self.call_back.closest_index]
        self.zero = sample_times[0] != times
        if self.zero: sample_times.insert(0,times)
        # sqrt_sample_times = np.sqrt(np.abs(np.diff(sample_times)))
        sqrt_sample_times = np.sqrt(np.diff(sample_times))
        def helper(n): x[...,n] *= sqrt_sample_times[n]
        if self.zero: 
            stop = x.shape[2]
        else: 
            stop = x.shape[2]-1
        [helper(n) for n in range(0,stop)]
        if self.zero: 
            zeros = np.zeros([x.shape[0],x.shape[1],1])
            x=np.concatenate([x,zeros],axis=2)
        out = super().__call__(x,kwargs)
        if self.zero: out = out[...,1:]
        return out

class tstar_operator:
    def __init__(self,kwargs):
        pass
    def __call__(self, y,kw):
        self.q = kw.get("q",1)
        D = y.shape[1]
        if self.q < 2: return y 
        self.times, self.timestar = kw['times'],kw.get('timestar',None)
        if self.timestar is None: 
            self.timestar = self.get_timestar(kw)
            kw['timestar'] = self.timestar
        a,self.operator = self.get_operator(y,kw = kw)
        out = np.zeros((y.shape[0],y.shape[1],len(kw['timestar'])))
        def helper(n): 
            def helperd(d): 
                c = np.array([np.multiply(a[n,k*D+d,:],self.operator[k,:]) for k in range(0,self.q)]).sum(axis=0)
                out[n,d,...] = c
                pass
            [helperd(d) for d in range(0,D)]
        [ helper(n) for n in range(y.shape[0])]
        return out
    
    def get_formatted_times(self,q,times,times_star,kwargs):
        times, times_star = ts_format_np(np.array(times),h=q,along_axis=[0]),np.array(times_star)
        ind =int((times.shape[1])/2)
        indices = np.array(get_closest_index(times[:,ind],list(times_star)))
        times = np.array([times[indices[n]] -times_star[n]  for n in range(0,len(indices))])
        return np.array(times),indices

    def get_timestar(self,kwargs):
        times = np.array(kwargs['times'])
        times = ts_format_np(times,h=self.q,along_axis=[0])
        q = kwargs.get("q",1)
        if q < 2: return times
        q = kwargs.get("q",1)
        def tstar(x): 
            mid = int((len(x)-1)/2)
            return x[mid] + (x[mid+1] - x[mid])/2.
        timestar = [tstar(times[n]) for n in range(0,times.shape[0])]
        return np.array(timestar)

    def get_operator(self,y,kw):
        orders = kw.get('orders',None)
        q = kw.get('q',1)
        D = y.shape[1]

        if orders is None:
            orders = np.zeros([q])
            orders[0] = 1
        times,self.indices = self.get_formatted_times(q,self.times, self.timestar,kw)
        out = np.zeros([y.shape[0],y.shape[1]*q,len(self.timestar)])
        y = ts_format_np(y,h=q,along_axis = 0)
        def helper(n):
            out[...,n] = y[...,self.indices[n]]
        [helper(n) for n in range(0,out.shape[-1])]
        return out,VanDerMonde(times,orders).T

  

class multi_time_interpolate(abc.ABC):
    @abc.abstractmethod
    def get_operator(self,kw): ...
    def __init__(self,kwargs):
        self.operator = self.get_operator(kwargs)

    def integrate(self,y,ic,kw):
        def one_step(n,ic,y):
            test = ic[...,n:n+self.call_back.q-1]
            op =  self.call_back.operator[n,:-1]
            test = test * op
            test = test.sum(axis=-1)
            test = (y[...,n] - test)/self.call_back.operator[n,-1]
            ic[...,n+self.call_back.q-1]  = test
        [one_step(n,ic,y) for n in range(0,ic.shape[2]-self.call_back.q)]
        return ic

    def apply_operator(self,operator,y,kw):
        D_axis = kw.get('D_axis',1)
        q = operator.shape[1]
        def helper(n): 
            s = [ts_format_np(y[n,ind,:].T,h=q) for ind in range(0,y.shape[D_axis])]
            s = [np.multiply(i,operator).sum(axis=1) for i in s]
            return s
        out = [ helper(n) for n in range(y.shape[0]) ]
        out = np.array(out)
        return out

    def __call__(self, y,kw):
        if isinstance(y,pd.DataFrame):
            values,times = q_interpolate(y.values,**kw)
            return pd.DataFrame(values.T,index = self.timestar,columns = y.columns)

        if self.q < 2: return y
        self.initial_conditions = y.copy()
        out = self.apply_operator(self.operator,y,kw)
        shape_ = list(y.shape)
        shape_[-1] = out.shape[-1]
        out = out.reshape(shape_)
        return out


class q_interpolate(multi_time_interpolate):
    @classmethod
    def get_operator(self,kw): return tstar_operator(kw)
    def __init__(self,kwargs): pass

    def __call__(self, y,kw):
        if isinstance(y,pd.DataFrame):
            values,times = q_interpolate(y.values,**kw)
            return pd.DataFrame(values.T,index = self.timestar,columns = y.columns)

        self.initial_conditions = y.copy()
        self.operator = self.get_operator(kw)
        out = self.operator(y,kw)
        if self.operator.q < 2: return y
        shape_ = list(y.shape)
        shape_[-1] = out.shape[-1]
        out = out.reshape(shape_)
        kw['timestar'] = self.operator.timestar
        kw['sample_times'] = self.operator.times
        return out

class inv_q_interpolate(q_interpolate):
    from bisect import bisect_left
    def __init__(self,q_interpolate_):
        self.call_back = q_interpolate_

    def integrate(self,y,ic,kw):
        times = self.call_back.operator.times
        timestar = self.call_back.operator.timestar
        min_,max_ = min(timestar),max(timestar)
        times = np.array([ min(max(times[n],min_),max_) for n in range(0,len(times))])
        out = self.get_operator(kw)(y,{**kw,**{'times':timestar,'timestar':times}})
        return out

    def __call__(self, y,kw):
        if isinstance(y,pd.DataFrame):
            values = self.__call__(y.values,**kw)
            return pd.DataFrame(values.T,index = self.call_back.timestar,columns = y.columns)

        if self.call_back.operator.q < 2: return y
        out = np.zeros(y.shape)
        out[...,:self.call_back.operator.q-1] = kw.get('initial_conditions',self.call_back.initial_conditions[...,:self.call_back.operator.q-1])
        out = self.integrate(y,out,kw)
        return out


class additive_noise_map(diff):
    def __init__(self,kwargs = None):
        if kwargs is not None: self.call_back = kwargs.get('call_back',diff({'zero':True}))
        else: self.call_back = diff({'zero':True})
        pass

    def __call__(self, y,kw):
        out = self.call_back(y,kw)
        self.ic = self.call_back.ic
        self.x = self.call_back.x
        x,z = self.x.reshape([self.x.shape[0]*self.x.shape[1],self.x.shape[2]]).T,out.reshape([out.shape[0]*out.shape[1],out.shape[2]]).T
        # fx = Kernel(x=x, fx=z, reg=1e-1)()
        # fx = op.Denoiser(**{**kw,**{'x':x,'fx':z}})()

        # We map the process to its noise
        # But we add some regularization to model extra noise
        # $ ln(X^{k+1}) = ln(X^{k}) + G(ln(X^k)) + \epsilon^k $
        # so $ G(ln(X^k)) $ is only part of the total noise, a function of the process itself
        self.operator = Kernel(x=x, fx=z, reg=0.1) #op.Cache(**{**kw,**{'x':x,'y':x,'fx':fx}})
        fx = self.operator(x)

        # The constant $\epsilon^k$ part is the difference remaining
        f_x = out-fx.T.reshape(out.shape)
        return f_x
    
class inv_additive_noise_map(inv_diff):

    def one_step(self,x,y,kw):
        x = get_matrix(x)
        z = self.call_back.operator(**{**kw,**{'z':x}})
        return x+z+y

class QuantileConditioner_map(diff):
    def __init__(self,kwargs):
        super().__init__({'zero':True})
        pass

    def __call__(self, y,kw):
        out = super().__call__(y,kw)
        
        x = self.x.reshape((self.x.shape[1], self.x.shape[2]*self.x.shape[0])).T # (506*1, 3)
        z = out.reshape((out.shape[1], out.shape[0]*out.shape[2])).T # (506*1, 3)

        # Conditionning the noise by the process
        # $L(\epislon) = \epsilon | X$
        latent_generator_y = kw.get('latent_generator_y',None)
        self.conditionner = ConditionerKernel(x=x, y=z, latent_generator_y=latent_generator_y) 
        self.conditionner.set_maps()

        latent_noise = self.conditionner.sampler_xy.get_x()[:, x.shape[1]:]
        latent_noise = latent_noise.T
        latent_noise = latent_noise.reshape(y.shape)

        # Reproductibility test
        if kw.get('reproductibility', False):
            xy = self.conditionner.sampler_xy(self.conditionner.sampler_xy.get_x())
            shape = (y.shape[0], self.x.shape[1]+y.shape[1], y.shape[2])
            xy = (xy.T).reshape(shape)
            xy_og = np.concatenate([x, z], axis=1)
            xy_og = (xy_og.T).reshape(shape)
            assert np.allclose(xy, xy_og), "The sampled xy does not match"

        return latent_noise

class VarConditioner_map(diff):
    def __init__(self,kwargs):
        super().__init__({'zero':True})
        pass

    def __call__(self, y,kw):
        self.noise = super().__call__(y,kw)
        variance = kw['vars']
        x = variance.reshape([variance.shape[1], variance.shape[0]*variance.shape[2]]).T
        y = self.noise.reshape([self.noise.shape[1], self.noise.shape[0]*self.noise.shape[2]]).T
        
        # We condition the noise by the variance
        # Noise is y, variance is x
        # This is $$\epsilon_X | \sigma^{k}$$
        self.conditionner = ConditionerKernel(x=x, y=y)
        self.conditionner.set_maps()

        # In this model, the variance is also a stochastic process
        # $$\sigma^{k+1} = \sigma^{k} + ( \epsilon_{\sigma} | \sigma^{k} )$$ 
        latent_gen = lambda n: np.random.normal(size=(n, x.shape[1]))
        kw['latent_generator_y'] = latent_gen
        self.map = QuantileConditioner_map(kw)
        latent_eps_sigma = self.map(variance, kw) # this is the latent representation of $$( \epsilon_{\sigma} | \sigma^{k} )$$

        # We get the latent representation of $$(\epsilon_x | \sigma^{k})$$
        latent_eps_x = self.conditionner.sampler_xy.get_x()[:, x.shape[1]:]
        latent_eps_x = latent_eps_x.T
        latent_eps_x = latent_eps_x.reshape(latent_eps_sigma.shape)
        
        # We return both latent_epsilon_x and latent_epsilon_sigma
        out = np.concatenate([latent_eps_sigma, latent_eps_x],axis=1)
        return  out

class inv_QuantileConditioner_map(inv_diff):

    def one_step(self,x,y,kw):
        x, y = get_matrix(x),get_matrix(y)

        # We want to get back the original noise from the latent representation
        # And y is already the latent representation
        latent_xy = np.concatenate([x,y],axis=1)
        xy = self.call_back.conditionner.sampler_xy(latent_xy)
        z = xy[:, x.shape[1]:]

        return super().one_step(x,z,kw)
    def __call__(self, y,kw):
        return super().__call__(y,kw)

class inv_VarConditioner_map(inv_diff):

    def one_step(self,x,y,kw):
        # Here x is ln(X) - the process
        # and y is reconstructed epsilon_sigma
        x,y = get_matrix(x),get_matrix(y)

        # We get back the corresponding latent_eps_x
        n = kw['n']
        latent_eps_x = self.latent_eps_x[:,:,n]

        # This is to get back $$\epsilon_X | \sigma^{k}$$
        latent_xy = np.concatenate([y, latent_eps_x], axis=1)
        xy = self.call_back.conditionner.sampler_xy(latent_xy)
        eps_x = xy[:, x.shape[1]:]

        return super().one_step(x,eps_x,kw)
    def __call__(self, y,kw):
        # y is generated latent epsilon_sigma and epsilon_x concat
        latent_eps_sigma = y[:,:self.call_back.conditionner.latent_y.shape[1],:]
        self.latent_eps_x = y[:,self.call_back.conditionner.latent_y.shape[1]:,:]
        
        # We first integrate the latent_eps_sigma
        map_ = composition_map([self.call_back.map])
        eps_sigma = inverse(map_)(latent_eps_sigma, kw)
        # Now we integrate the latent_eps_x with the eps_sigma and the process 
        out = super().__call__(eps_sigma,kw)

        # This basically hijacks and get the variances back directly
        # orvars = kw['vars'].copy()
        # test = [orvars for n in range(y.shape[0])]
        # test = np.concatenate(test,axis=0)
        # out = super().__call__(test,kw)
        # meanvars = np.mean(orvars)
        # vars_noise -= np.mean(vars_noise)+meanvars
       
        return out

    
class diffusive_map(diff):
    def __init__(self,kwargs):
        super().__init__({'zero':True})
        pass

    def __call__(self, y,kw):
        out = super().__call__(y,kw)
        self.noise = out.reshape(out.shape[0]*out.shape[1],out.shape[2]).T
        self.transported_sum_ = gen.transported_sum(**{**kw,**{'y':self.noise}})
        return out
    
class inv_diffusive_map(inv_diff):

    def __call__(self, y,kw):
        out = np.zeros([y.shape[0],self.call_back.ic.shape[1],y.shape[2]])
        out[...,0] = kw.get('initial_conditions',self.call_back.ic)
        transported_sum_ = self.call_back.transported_sum_
        def integrate(n):
            if n == 0:return
            last = out[...,n-1]
            new = transported_sum_(x = last,shuffle=False)
            out[...,n] = new
            # out[...,n] = out[...,n-1] + Bs
            pass
        [integrate(n) for n in range(0,y.shape[2])]
        out= super().__call__(y,kw)
        return out
    


class add_variance_map:
    def __init__(self,**kwargs):
        self.var_q = kwargs.get('var_q',10)
    def __call__(self, y,kw):
        D = y.shape[1]
        along_axis=0
        z = diff()(y,kw)
        x = ts_format_np(np.array(z),h=self.var_q,along_axis=along_axis)
        out = np.zeros_like(y)
        def helpern(n,d,k):
            ktilde = min(k,x.shape[2]-1)
            sample = x[n,range(d, D * self.var_q, D),ktilde]
            sqrvar = np.sqrt(np.var(sample))
            out[n,d,k] = sqrvar
        [helpern(n,d,k) for n in range(0,out.shape[0]) for k in range(out.shape[2]) for d in range(0,D)]
        kw['vars'] = out
        # plt.plot(out[0,0,:])
        return y

class inv_add_variance_map:
    def __init__(self,call_back):
        self.call_back = call_back
    def __call__(self, y,kw):
        return y



class ar_map(diff):
    def __call__(self, X, kwargs):
        '''
        parameters:
            X: signal pd.DataFrame format
        output:
            res: residuals in pd.DataFrame format
        '''
        a = kwargs['a']
        order = kwargs['a'].shape[0]
        N,D,T = X.shape[0],X.shape[1],X.shape[2]-order
        self.ic = X
        out = np.ndarray([N,D,T])
        def helper(n):
            def integratet(t):
                temp = np.array([a[o]*X[n,:,t-o-1] for o in range(order)])
                temp = np.sum(temp,axis=0)
                temp = X[n,:,t] - temp
                out[n,:,t-order] = temp
                pass
            [integratet(t) for t in range(order,X.shape[2])]
        [ helper(n) for n in  range(N) ] #X^n - sum_{1<=i<=order } a_i x_{n-i}
        # test = X - inv_ar_map(self)(out,kwargs)
           
        return np.array(out)
class inv_ar_map(inv_diff):
    def __init__(self, call_back) -> None:
        self.call_back = call_back
    def __call__(self, X, kwargs) -> Any:
        a = kwargs['a']
        order = kwargs['a'].shape[0]
        N,D,T = X.shape[0],X.shape[1],X.shape[2]+order
        sample_times = kwargs.get('sample_times',False)
        if sample_times is not None : T = min(kwargs['sample_times'].shape[0],X.shape[2]+order)
        out = np.ndarray([N,D,T])
        out[...,:order] = self.call_back.ic[...,:order] 
        def integraten(n):
            def integratet(t):
                temp = np.array([a[o]*out[n,:,t-o-1] for o in range(order)])
                temp = np.sum(temp,axis=0)
                temp = X[n,:,t-order] + temp
                # test = self.call_back.ic[...,t]
                out[n,:,t] = temp
                pass
            [integratet(t) for t in range(order,T)]
        [integraten(n) for n in range(N)]
         
        return np.array(out)

class ma_map(ar_map):
    def __call__(self, X, kwargs):
        if 'b' not in kwargs: return X
        return super().__call__(X, {**kwargs,**{'a':-kwargs['b']}})
class inv_ma_map(inv_ar_map):
    def __init__(self, call_back) -> None:
        self.call_back = call_back
    def __call__(self, X, kwargs) :
        if 'b' not in kwargs: return X
        return super().__call__(X, {**kwargs,**{'a':-kwargs['b']}})         

class arma_map(ar_map):
     
    map_ = composition_map([ma_map(),ar_map()])
    def __call__(self, X, kwargs):
        return self.map_(X, kwargs)

class inv_arma_map:
    def __init__(self, call_back) :
        self.call_back = call_back
    def __call__(self, x, kwargs):
        return inverse(self.call_back.map_)(x, kwargs)


class garch_map(diff):
    def __call__(self, X, kwargs):
        if len(X.shape) == 3:
            X = X[0].T 
        data = np.zeros((1, kwargs['data'].shape[0], kwargs['data'].shape[1])) 
        data[0, : , :] = kwargs['data']
        p, q = kwargs['a'].shape[0], kwargs['b'].shape[0]
        self.ic = X[:max(p,q),:].T.tolist()  
        res = []
        for i in range(X.shape[1]):
            Xtmp = self.direct_map(X[:,i], kwargs['a'][:,i], kwargs['b'][:,i], kwargs)
            res += [Xtmp]
        res = pd.DataFrame(res).values
        return np.reshape(res,(1,res.shape[0],res.shape[1]))
    def direct_map(self, x, a, b, kwargs):
        p = len(a) - 1
        q = len(b)
        residuals = [0] * p  # initialize residuals with zeros
        variances = [1] * q  # initialize variances with ones
        for t in range(max(p, q), len(x)):
            alpha_component = sum(a[i+1] * residuals[t-i-1]**2 for i in range(min(t, p)))
            beta_component = sum(b[i] * variances[t-i-1] for i in range(min(t, q)))
            variance_t = a[0] + alpha_component + beta_component
            epsilon_t = x[t] / (variance_t ** 0.5)
            residuals.append(epsilon_t)
            variances.append(variance_t)
        return residuals[p:]






class inv_garch_map(inv_diff):
    def __init__(self, call_back) -> None:
        self.call_back = call_back
    def __call__(self, res, kwargs) -> Any:
        '''
        parameters:
            res: residuals in pd.DataFrame format
            kwargs['ic']: p initial conditions of the signal
        output:
            reconstruted signal in pd.DataFrame format
        '''
        out = np.zeros([res.shape[0],res.shape[1],res.shape[2]])
        ic = np.array(self.call_back.ic)
        def invert(n):
            p = len(kwargs['a'][:,0])
            q = len(kwargs['b'][:,0])
            out[n,0,:] = np.asarray(self.inverse_map(res[n,0,p:], ic[0,:max(p,q)], kwargs['a'][:,0], kwargs['b'][:,0], kwargs))
            for i in range(1, res.shape[1]):  # Start from the second element
                p = len(kwargs['a'][:,i])
                q = len(kwargs['b'][:,i])
                out[n, i, :] = np.array(self.inverse_map(res[n,i,p:], ic[i,:max(p,q)], kwargs['a'][:,i], kwargs['b'][:,i], kwargs))
        [invert(n) for n in range(0,res.shape[0])]
        return np.array(out)
    
    def inverse_map(self, res, x_0, a, b, kwargs) -> list:
        '''
        parameters:
            p: the order of GARCH(p,q) process
            res: residual of the stochastic process in the form of NDarray
            a: autoregressive parameteres 
        output:
            X: reconstructed signal
        '''
        X = x_0.tolist()
        p = len(a) - 1
        q = len(b)
        res = [0]*max(p, q) + res.tolist()  # pad residuals at the beginning with zeros
        variances = [0] * max(p, q)  # initialize variances with zeros
        for t in range(max(p, q), len(res)):
            variance_t = a[0] + sum(a[i] * res[t-i]**2 for i in range(1, min(t, p)+1)) + sum(b[i-1] * variances[t-i] for i in range(1, min(t, q)+1))
            variances.append(variance_t)
            X_t = res[t] * (variance_t ** 0.5)
            X.append(X_t)
            
        return X

def Id(x,kwargs): return x

def cartesian_map(x,kwargs):
    out=x.copy()
    list_maps=kwargs.get('list_maps')[:x.shape[1]]
    for i in range(0,len(list_maps)):
        out[:,i,:]=list_maps[i](x[:,i,:],kwargs)   
    return out

def moment_cartesian_map(x,kwargs):
    out = cartesian_map(x,kwargs)
    return out

def inv_cartesian_map(x,kwargs):
    out=x.copy()
    list_maps=kwargs.get('list_maps')[:x.shape[1]]
    D = len(list_maps)
    for i in range(0,len(list_maps)):
        out[:,i,:]=inverse(list_maps[i])(x[:,i,:],kwargs)         
    return out

def inv_moment_cartesian_map(x,kwargs):
    out = inv_cartesian_map(x,kwargs)
    return out

#################################################################################################################
################################################ Normal returns #################################################
#################################################################################################################

def log_map(x,kwargs=None): return  np.log(x)
def exp_map(x,kwargs=None): 
    # return x
    return  np.exp(x)

def log_ret(x,kwargs):
    if (x.ndim == 3):
        out = np.concatenate( [log_ret(x[n],**kwargs).reshape([1,x.shape[1],x.shape[2]]) for n in range(0,x.shape[0])],axis = 0)
        return out
    x = np.log(x)
    out = normal_return()(x,kwargs)
    return get_matrix(out)    

def inv_log_ret(x,kwargs):
    times = list(kwargs.get('times',None))
    mean = kwargs.get('mean',None)
    # linear_kernel = kernel_setters.kernel_helper(setter = kernel_setters.set_linear_regressor_kernel,polynomial_order = 2,regularization = 1e-8)
    # y = op.projection(x=x,y=x,z=x,fx=x, set_codpy_kernel=linear_kernel)
    out = inv_normal_return(x,times = times,mean=mean,initial_values = 0.)
    initial_values = kwargs.get('initial_values',None)
    T = x.shape[1]
    out = initial_values*np.exp(out)
    return out

#################################################################################################################

#################################################################################################################
inverse_class_switchDict = {
    'composition_map2':inv_composition_map2,
    'composition_map':inv_composition_map,
    'q_interpolate':inv_q_interpolate,
    'additive_noise_map':inv_additive_noise_map,
    'diff':inv_diff,
    'normal_return':inv_normal_return,
    'QuantileConditioner_map':inv_QuantileConditioner_map,
    'VarConditioner_map':inv_VarConditioner_map,
    'diffusive_map':inv_diffusive_map,
    'add_variance_map':inv_add_variance_map,
    'pct_change':inv_pct_change,
    'random_variable_sum':inv_random_variable_sum,
    'mean_map': inv_mean_map,
    'ar_map': inv_ar_map,
    'ma_map': inv_ma_map,
    'garch_map': inv_garch_map,
    'remove_time':add_time,
    'arma_map': inv_arma_map
}
inverse_fun_switchDict = {
    'moment_cartesian_map':inv_moment_cartesian_map,
    'cartesian_map':inv_cartesian_map,
    'log_ret':inv_log_ret,
    'log_map':exp_map,
    'Id':Id
}
