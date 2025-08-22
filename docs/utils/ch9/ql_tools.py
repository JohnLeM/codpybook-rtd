import abc

import QuantLib as ql
import numpy as np
import os,sys
import copy 

from codpy.data_conversion import get_date, get_float
from codpy.kernel import get_matrix 

import math,datetime
import pandas as pd
import torch

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

PARENT_DIR = os.path.abspath(os.path.join(current_dir, '..'))
data_path = os.path.join(PARENT_DIR, "data")
if not os.path.exists(data_path):
    os.makedirs(data_path)
PARENT_DIR = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, PARENT_DIR)

from utils.ch9.montecarlo import * 
import utils.ch9.mapping as maps

def date_to_quantlib(date):
    date_to_quantlib_switchDict = { 
                                    ql.Date: lambda x : x,
                                    datetime.date: lambda x : ql.Date(x.day,x.month,x.year),
                                    datetime.datetime: lambda x : ql.Date(x.day,x.month,x.year)
                                    }
    def default_date_to_quantlib(x) : return date_to_quantlib(get_date(x))
    debug = type(date)
    return date_to_quantlib_switchDict.get(debug,default_date_to_quantlib)(date)
    return ql.Date(date.day,date.month,date.year)

class codpy_param_getter:
    def get_params(**kwargs) : return kwargs.get('codpy',{})
    def get_kernel_fun(**kwargs): return codpy_param_getter.get_params(**kwargs)['set_kernel']

#################################################################################################################
################################################ Vanilla Option #################################################
################################################################################################################# 

def VanillaOption(today_date, spot_price, risk_free_rate, dividend_rate, strike_price, volatility,  maturity_date, day_count = ql.Actual365Fixed(), calendar = ql.UnitedKingdom(),
                    option_type = ql.Option.Call):

    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    ql.Settings.instance().evaluationDate = today_date
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today_date, risk_free_rate, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(today_date, dividend_rate, day_count))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today_date, calendar, volatility, day_count))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle,  dividend_yield, flat_ts,  flat_vol_ts)  
    european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    out = european_option.NPV()
    return out

def PNLVanillaOption(PricesDates, Spotprices, ShiftPricesDates, ShiftSpotprices, risk_free_rate  = 0.0, dividend_rate  = 0.01, strike_price  = 3000, volatility = .1, maturity_date = ql.Date(1, 1, 2023), notional = 10000.):
    out = np.zeros(len(PricesDates))

    for i in range(len(PricesDates)) :
        datei = PricesDates[i]
        shiftdatei = ShiftPricesDates[i]
        spoti = Spotprices[i]
        shiftspoti = ShiftSpotprices[i]
        shiftvaluei = VanillaOption(shiftdatei, shiftspoti, risk_free_rate, dividend_rate, strike_price, volatility, maturity_date)
        valuei = VanillaOption(datei, spoti, risk_free_rate, dividend_rate, strike_price, volatility, maturity_date)
        out[i] = shiftvaluei - valuei
    return out*notional


class option(payoff):
    def get_param(**kwargs):
        return kwargs.get('option')
    
    def __init__(self,**kwargs):
        getter = option_param_getter()
        self.today_date= getter.get_today_date(**kwargs)
        self.maturity_date= getter.get_maturity_date(**kwargs)
        self.strike_price= getter.get_strike(**kwargs)
        self.spot_price= getter.get_spot(**kwargs)
        self.risk_free_rate= getter.get_risk_free_rate(**kwargs)
        self.time_to_maturity = ql.Actual365Fixed().yearFraction(date_to_quantlib(self.today_date), date_to_quantlib(self.maturity_date))
        self.option_type = getter.get_option_type(**kwargs)

    def set_spot(self,spot):
        self.spot_price=spot
    
    def f(self,x):
        if self.option_type == ql.Option.Call:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*np.maximum(0,x[:,-1,-1] - self.strike_price)
        else:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*np.maximum(0,self.strike_price- x[:,-1,-1])

    def get_times(self,**kwargs):
        return [self.today_date,self.maturity_date]

class OptionClosedFormula(pricer):
    def getter(self): return option_param_getter()

    def get_ql_payoff(self,**kwargs):
        getter = self.getter()
        strike = getter.get_strike(**kwargs)
        option_type = getter.get_option_type(**kwargs)
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = getter.get_exercise(**kwargs)
        return ql.VanillaOption(payoff, exercise) 

    def get_ql_process(self,**kwargs):
        process = self.getter().get_process(**kwargs)
        return process 

    def price(self,**kwargs):
        today_date = option_param_getter().get_today_date(**kwargs)
        european_option = self.get_ql_payoff(**kwargs) 
        ql.Settings.instance().evaluationDate = date_to_quantlib(today_date)
        ql_process = self.get_ql_process(**kwargs) 
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(ql_process))
        price=european_option.NPV()
        return price

class BarrierClosedFormula(pricer):
    def getter(self): return option_param_getter()

    def get_ql_payoff(self,**kwargs):
        getter = self.getter()
        strike = getter.get_strike(**kwargs)
        option_type = getter.get_option_type(**kwargs)
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = getter.get_exercise(**kwargs)
        return ql.VanillaOption(payoff, exercise) 

    def get_ql_process(self,**kwargs):
        process = self.getter().get_process(**kwargs)
        return process  

    def price(self,**kwargs):
        today_date = option_param_getter().get_today_date(**kwargs)
        european_option = self.get_ql_payoff(**kwargs) 
        ql.Settings.instance().evaluationDate = date_to_quantlib(today_date)
        ql_process = self.get_ql_process(**kwargs) 
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(ql_process))
        price=european_option.NPV()
        return price

class yf_param_getter(codpy_param_getter):
    def get_D(self,**kwargs): 
        yf_param = kwargs['yf_param']
        return len(yf_param['symbols'])

    def get_today_date(self,**kwargs): 
        today_date = kwargs.get('today_date')
        return get_date(today_date)
    def set_today_date(self,value,**kwargs): 
        kwargs['today_date'] = value
        return kwargs
    def set_spot(self,value,**kwargs): 
        self.get_process_param(**kwargs)['spot_price'] = float(value)
        return kwargs
    def get_spot(self,**kwargs): return get_float(self.get_process_param(**kwargs)['spot_price'])


class option_param_getter(codpy_param_getter):
    def get_process(self,**kwargs): return QL_path_generator.get_process(**kwargs)
    def get_D(self,**kwargs): return QL_path_generator.get_D(**kwargs)
    def get_N(self,**kwargs): return MonteCarloPricer.get_N(**kwargs)
    def get_H(self,**kwargs): return MC_time_series_generator.get_H(**kwargs)

    def get_today_date(self,**kwargs): 
        today_date = kwargs.get('today_date')
        return get_date(today_date)
    def get_closed_formula(self,**kwargs): return BasketOptionClosedFormula()
    def set_today_date(self,value,**kwargs): 
        kwargs['today_date'] = value
        return kwargs
    def set_spot(self,value,**kwargs): 
        self.get_process_param(**kwargs)['spot_price'] = float(value)
        return kwargs
    def get_process_param(self,**kwargs): return kwargs['BSM']
    def get_param(self,**kwargs): return kwargs['option']
    def get_spot(self,**kwargs): return get_float(self.get_process_param(**kwargs)['spot_price'])
    def get_risk_free_rate(self,**kwargs):return get_float(self.get_process_param(**kwargs)['risk_free_rate'])
    def get_dividend_rate(self,**kwargs): return get_float(self.get_process_param(**kwargs)['dividend_rate'])
    def get_volatility(self,**kwargs): return get_float(self.get_process_param(**kwargs)['volatility'])
    def get_exercise(self,**kwargs): 
        debug = self.get_maturity_date(**kwargs)
        return ql.EuropeanExercise(date_to_quantlib(debug))
    def get_strike(self,**kwargs): return self.get_param(**kwargs)['strike_price']
    def get_maturity_date(self,**kwargs): return get_date(self.get_param(**kwargs)['maturity_date'])
    def get_option_type(self,**kwargs): return self.get_param(**kwargs)['option_type']

#################################################################################################################
################################################# Basket Option #################################################
################################################################################################################# 

class basket_option_param_getter(option_param_getter):
    def get_process(self,**kwargs): return BSMultiple().get_process(**kwargs)
    def get_D(self,**kwargs) : return self.get_process(**kwargs).factors()
    def get_instrument(self,**kwargs): return BasketOption(**kwargs)
    def get_param(self,**kwargs): return kwargs['BasketOption']
    def get_process_param(self,**kwargs): return kwargs['BSMultiple']
    def get_processes_param(self,**kwargs): return self.get_process_param(**kwargs)['BSMs']
    def get_weights(self,**kwargs): return self.get_param(**kwargs)['weights']
    def get_spot(self,**kwargs): 
        processes_param = self.get_processes_param(**kwargs)
        spots = [option_param_getter().get_spot(**k) for k in processes_param]
        result = np.array(spots)
        return result
    def get_time_spot(self,**kwargs): 
        today_date=[get_float(self.get_today_date(**kwargs),**kwargs)]
        spots = self.get_spot(**kwargs)
        return get_matrix(np.concatenate([ today_date,spots])).T


    def set_spot(self,x,**kwargs): 
        new_spots = [option_param_getter().set_spot(**k,value = v) for k,v in zip(self.get_processes_param(**kwargs),x)]
        kwargs['BSMultiple']['BSMs'] = new_spots
        return kwargs
    def set_time_spot(self,x,**kwargs): 
        # Check if we have enough elements for time and spots
        if len(x) <= 1:
            # Only update time if we have at least one element and it looks like a valid date
            if len(x) >= 1 and x[0] > 700000:  # Reasonable date range check
                kwargs = self.set_today_date(x[0],**kwargs)
            return kwargs
        kwargs = self.set_spot(x[1:],**kwargs) 
        kwargs = self.set_today_date(x[0],**kwargs) 
        return kwargs
    def set_spot_today_date(self,values,time,**kwargs): 
        kwargs = self.set_spot(values,**kwargs) 
        kwargs = self.set_today_date(time,**kwargs) 
        return kwargs
    def get_risk_free_rate(self,**kwargs): return [option_param_getter().get_risk_free_rate(**k) for k in self.get_processes_param(**kwargs)]
    def get_dividend_rate(self,**kwargs): return [option_param_getter().get_dividend_rate(**k) for k in self.get_processes_param(**kwargs)]
    def get_volatility(self,**kwargs): return [option_param_getter().get_volatility(**k) for k in self.get_processes_param(**kwargs)]

class basket_binary_param_getter(basket_option_param_getter):
    def get_instrument(self,**kwargs): return Basketbinary(**kwargs)
    def get_param(self,**kwargs): return kwargs['BasketBinary']
    def get_closed_formula(self,**kwargs): return BasketbinaryClosedFormula()



class BasketOption(payoff):
    def get_param(**kwargs):
        return kwargs.get('BasketOption')

    def set_spot(self,spot):
        self.spot_price=spot

    def __init__(self,**kwargs):
        getter = basket_option_param_getter()
        self.today_date= getter.get_today_date(**kwargs)
        self.maturity_date= getter.get_maturity_date(**kwargs)
        self.weights= getter.get_weights(**kwargs)
        self.option_type= getter.get_option_type(**kwargs)
        self.time_to_maturity = ql.Actual365Fixed().yearFraction(date_to_quantlib(self.today_date), date_to_quantlib(self.maturity_date))

    def basket_value(self,weights,values,**kwargs):
        import itertools
        # return get_data(np.squeeze(values)).dot(weights)
        if values.ndim == 3: 
            a = np.arange(values.shape[0])
            b = np.arange(values.shape[2])
            test = [(i,j) for i in a for j in b]
            out = np.array([values[e[0],:,e[1]].dot(weights) for e in test])
            return out
        return maps.get_data(values).dot(weights)
    def hessian_f(self,x):
        f_switchDict = { 
                                pd.DataFrame: lambda x : self.nabla_f(x.values),
                                torch.tensor: lambda x : self.nabla_f(float(x)),
                                }
        def default_fun(x): return np.zeros([x.shape[0],x.shape[1],x.shape[1],1])
        debug = type(x)
        return f_switchDict.get(debug,default_fun)(x)

    def nabla_f(self,x):
        f_switchDict = { 
                                pd.DataFrame: lambda x : self.nabla_f(x.values),
                                torch.tensor: lambda x : self.nabla_f(float(x)),
                                }
        def default_fun(x):
            # if x.ndim ==2 : return np.concatenate([default_fun(x[n]) for n in range(x.shape[0])])
            out = np.zeros(x.shape)
            if x.ndim==1 : z = self.basket_value(np.array(self.weights),x[1:])- self.strike_price
            else : z = self.basket_value(np.array(self.weights),x[:, 1:])- self.strike_price
            if self.option_type == ql.Option.Call: 
                out[z >= 0,1:] = self.weights
            else:
                out[z < 0,1:] = -self.weights
            return out[:,:,np.newaxis]
        debug = type(x)
        return f_switchDict.get(debug,default_fun)(x)

    def f(self,x, **kwargs):
        f_switchDict = { 
                                pd.DataFrame: lambda x,**kwargs : self.f(x.values, **kwargs),
                                torch.tensor: lambda x,**kwargs : self.f(float(x), **kwargs),
                                }
        def default_fun(x,**kwargs):
            
            self.strike_price= basket_option_param_getter().get_strike(**kwargs)

            if x.ndim==1 : z = self.basket_value(np.array(self.weights),x[1:])- self.strike_price
            else : 
                means = kwargs.get('means',None)
                if means is not None:
                    meanx = x[:,:,0].mean(axis=0)
                    x[:,:,0] *= means / meanx
                temp = self.basket_value(np.array(self.weights),x)
                z = temp- self.strike_price
            if self.option_type == ql.Option.Call:
                return np.maximum(0.,z)
            else:
                return np.maximum(0.,-z)
        debug = type(x)
        return f_switchDict.get(debug,default_fun)(x,**kwargs)

    def get_times(self,**kwargs):
        today_date= kwargs['getter'].get_today_date(**kwargs)
        return [today_date,self.maturity_date]

class Basketbinary(BasketOption):
    def get_param(**kwargs):
        return kwargs.get('Basketbinary')

    def __init__(self,**kwargs):
        getter = basket_binary_param_getter()
        self.today_date= getter.get_today_date(**kwargs)
        self.maturity_date= getter.get_maturity_date(**kwargs)
        self.strike_price= getter.get_strike(**kwargs)
        self.weights= getter.get_weights(**kwargs)
        self.option_type= getter.get_option_type(**kwargs)
        self.time_to_maturity = ql.Actual365Fixed().yearFraction(date_to_quantlib(self.today_date), date_to_quantlib(self.maturity_date))

    def f(self,x):
        f_switchDict = { 
                                pd.DataFrame: lambda x : self.f(x.values),
                                torch.tensor: lambda x : self.f(float(x)),
                                }
        def default_fun(x):
            if x.ndim==1 : z = self.basket_value(np.array(self.weights),x[1:])- self.strike_price
            else : z = self.basket_value(np.array(self.weights),x[:, 1:])- self.strike_price
            out = np.zeros(z.shape)
            if self.option_type == ql.Option.Call:
                out[z >= 0] = 1.
            else:
                out[z < 0] = 1.
            return out
        debug = type(x)
        return f_switchDict.get(debug,default_fun)(x)
    def get_closed_formula(self,**kwargs):  return BasketbinaryClosedFormula()
    def nabla_f(self,x):
        f_switchDict = { 
                                pd.DataFrame: lambda x : self.nabla_f(x.values),
                                torch.tensor: lambda x : self.nabla_f(float(x)),
                                }
        def default_fun(x): return np.zeros([x.shape[0],x.shape[1],1])
        debug = type(x)
        return f_switchDict.get(debug,default_fun)(x)





class BasketOptionClosedFormula(OptionClosedFormula):

    def get_spot_basket(self,x,**kwargs):
        getter = self.getter()
        # x = kwargs.get("x")
        if isinstance(x,pd.DataFrame):x = x.loc[:,x.columns != "Date"].values
        if x.ndim == 2:
            out = [self.get_spot_basket(x=get_matrix(x)[n],**kwargs) for n in range(0,x.shape[0])]
            return np.array(out)
        if x.ndim == 3:
            out = np.zeros((x.shape[0],x.shape[2]))
            def helper(n) : out[n] = self.get_spot_basket(x=x[n],**kwargs)
            [helper(n) for n in range(0,x.shape[0])]
            return out
        weights = kwargs.get('weights',np.array(getter.get_weights(**kwargs)) )
        
        # Check if x is empty - this should no longer happen with our fix, but keep as safeguard
        if x.size == 0:
            return 0.0
            
        if x.ndim == 1:return x.dot(np.array(weights))

    def get_ql_payoff(self,**kwargs):
        getter = self.getter()
        strike = getter.get_strike(**kwargs)
        option_type = getter.get_option_type(**kwargs)
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = getter.get_exercise(**kwargs)
        return ql.VanillaOption(payoff, exercise) 
        

    def getter(self): return basket_option_param_getter()
    def get_ql_process(self,**kwargs):
        getter = self.getter()
        spot = self.get_spot_basket(getter.get_spot(**kwargs),**kwargs)
        process = getter.get_process(**kwargs)
        return process.get_process(**kwargs)

    def get_option(self,**kwargs):
        getter = self.getter()
        european_option = self.get_ql_payoff(**kwargs) 
        ql_process = kwargs.get('ql_process',None)
        if ql_process is None:
            dic = kwargs.get('BSM',None)
            if dic is None:
                spot = self.get_spot_basket(**{**kwargs,**{'x':np.asarray(getter.get_spot(**kwargs))}})
                volatility = getter.get_volatility(**kwargs)
                dic = {'BSM':{
                                'spot_price' : spot,
                                'risk_free_rate' : kwargs.get('risk_free_rate',0.),
                                'dividend_rate' : kwargs.get('dividend_rate',0.),
                                'volatility' : np.mean(volatility),
                    }}
                ql_process = BSM().get_process(**{**kwargs,**dic})
            else: ql_process = BSM().get_process(**kwargs)
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(ql_process))
        return european_option
    def nabla(self,**kwargs):
        getter = self.getter()
        european_option =self.get_option(**kwargs)
        getter = self.getter()
        today_date = getter.get_today_date(**kwargs)
        fl_today_date = get_float(today_date)
        ql_today_date = date_to_quantlib(fl_today_date)
        ql_other_date = date_to_quantlib(fl_today_date-365.)
        weights = np.array(getter.get_weights(**kwargs))
        ql.Settings.instance().evaluationDate = ql_today_date
        deltas = european_option.delta()*weights
        days = ql.Thirty360(ql.Thirty360.BondBasis).dayCount(ql_other_date,ql_today_date)
        theta = np.array([european_option.theta()])/days
        nabla = np.concatenate((theta, deltas))
        return np.array(nabla)

    def price(self, t=None, **kwargs):
        getter = self.getter()
        today_date = getter.get_today_date(**kwargs)
        ql.Settings.instance().setEvaluationDate( date_to_quantlib(today_date) )
        # test = ql.Settings.instance().evaluationDate
        price=self.get_option(**kwargs).NPV()
        return price
    def hessian(self,**kwargs):
        hessian = super().hessian(set_fun = basket_option_param_getter().set_time_spot,**kwargs)
        european_option =self.get_option(**kwargs)
        getter = self.getter()
        today_date = getter.get_today_date(**kwargs)
        ql.Settings.instance().evaluationDate = date_to_quantlib(today_date)
        gamma = european_option.gamma()
        hessian[0,1,1] = gamma
        return hessian

    def nablas(self, values,**kwargs): 
        copy_kwargs = copy.deepcopy(kwargs)
        set_fun = kwargs["set_fun"]
        # copy_kwargs = kwargs.copy()
        def helper(v): 
            k = set_fun(v,**copy_kwargs)
            return self.nabla(**copy_kwargs) 
        if isinstance(values,list): out = [helper(v) for v in get_matrix(values)]
        elif isinstance(values,np.ndarray): 
            if values.ndim == 1: return helper(values)
            out = [helper(values[n]) for n in range(0,values.shape[0]) ]
        elif isinstance(values,pd.DataFrame): return FD.nablas(fun=fun,values= values.values,**copy_kwargs)
        out = np.array(out).reshape(values.shape)
        return out

class BasketbinaryClosedFormula(BasketOptionClosedFormula):

    def getter(self): return basket_binary_param_getter()
    def nabla_f(self,x):
            f_switchDict = { 
                                    pd.DataFrame: lambda x : self.nabla_f(x.values),
                                    torch.tensor: lambda x : self.nabla_f(float(x)),
                                    }
            def default_fun(x): return  np.zeros(x.shape)
            debug = type(x)
            return f_switchDict.get(debug,default_fun)(x)

    def get_ql_payoff(self,**kwargs):
        getter = self.getter()
        strike = getter.get_strike(**kwargs)
        option_type = getter.get_option_type(**kwargs)
        binaryPayoff = ql.CashOrNothingPayoff(option_type, strike, 1)
        exercise = getter.get_exercise(**kwargs)
        return ql.VanillaOption(binaryPayoff, exercise)


#################################################################################################################
################################################# Heston Option #################################################
################################################################################################################# 

class HestonOption(payoff):
    def get_param(**kwargs):
        return kwargs.get('HestonOption')
    
    def __init__(self,**kwargs):
        getter = Heston_param_getter()
        self.today_date = getter.get_today_date(**kwargs)
        self.maturity_date = getter.get_maturity_date(**kwargs)
        self.strike_price = getter.get_strike(**kwargs)
        self.spot_price = getter.get_spot(**kwargs)
        self.risk_free_rate = getter.get_risk_free_rate(**kwargs)
        self.dividend_rate = getter.get_dividend_rate(**kwargs)
        self.volatility = getter.get_volatility(**kwargs)
        self.timesteps = getter.get_timesteps(**kwargs)
        self.time_to_maturity = ql.Actual365Fixed().yearFraction(date_to_quantlib(self.today_date), date_to_quantlib(self.maturity_date))
        self.option_type = getter.get_option_type(**kwargs)

    def set_spot(self,v,**kwargs): 
        kwargs['Heston']['spot_price'] = v
        return kwargs
    
    def f(self,x,**kwargs):
        if self.option_type == ql.Option.Call:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*np.maximum(0,x[:,0] - self.strike_price)
        else:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*np.maximum(0,self.strike_price- x[:,0])

    def get_times(self,**kwargs):
        return [self.today_date,self.maturity_date]
        return list(ql.TimeGrid(self.time_to_maturity, self.timesteps))

class Heston_param_getter(option_param_getter):
    def get_process_param(self,**kwargs): return kwargs['Heston']
    def get_maturity_date(self,**kwargs): return get_date(kwargs['maturity_date'])
    def get_param(self,**kwargs): return kwargs['HestonOption']
    def get_risk_free_rate(self, **kwargs): return self.get_process_param(**kwargs)['risk_free_rate']
    def get_volatility(self,**kwargs): return get_float(self.get_process_param(**kwargs)['v0'])
    def get_kappa(self,**kwargs): return self.get_process_param(**kwargs)['kappa']
    def get_theta(self,**kwargs): return self.get_process_param(**kwargs)['theta']
    def get_sigma(self,**kwargs):return self.get_process_param(**kwargs)['sigma']
    def get_rho(self,**kwargs): return self.get_process_param(**kwargs)['rho']
    def get_timesteps(self,**kwargs):return self.get_param(**kwargs).get('timesteps',None)

class HestonClosedFormula(pricer):
    def price(self,**kwargs):
        getter = Heston_param_getter()
        today_date = date_to_quantlib(getter.get_today_date(**kwargs))
        risk_free_rate = getter.get_risk_free_rate(**kwargs)
        dividend_rate = getter.get_dividend_rate(**kwargs)
        strike = getter.get_strike(**kwargs)
        option_type = getter.get_option_type(**kwargs)
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        spot = getter.get_spot(**kwargs)
        v0 = getter.get_volatility(**kwargs)
        kappa=getter.get_kappa(**kwargs)
        theta=getter.get_theta(**kwargs)
        rho=getter.get_rho(**kwargs)
        sigma=getter.get_sigma(**kwargs)
                                                                           
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today_date, risk_free_rate, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today_date, dividend_rate, ql.Actual365Fixed()))
        exercise = Heston_param_getter().get_exercise(**kwargs)
        ql.Settings.instance().evaluationDate = today_date

        european_option = ql.VanillaOption(payoff, exercise) 
        hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, ql.QuoteHandle(ql.SimpleQuote(spot)), v0, kappa, theta, sigma, rho)
        european_option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(hestonProcess)))
        price=european_option.NPV()
        return price

#################################################################################################################
################################################## Asian Option #################################################
################################################################################################################# 
class Asian_param_getter(option_param_getter):
    def get_fixing_dates(**kwargs): return kwargs['GeometricAsianOption']['fixingDates']
    def get_maturity_date(**kwargs): return get_date(kwargs['GeometricAsianOption']['maturity_date'])
    def get_today_date(**kwargs): return get_date(kwargs['GeometricAsianOption']['today_date'])
    def get_strike(**kwargs): return kwargs['GeometricAsianOption']['strike_price']        
        
class GeometricAsianOption(payoff):
    def get_param(**kwargs):
        return kwargs.get('GeometricAsianOption')

    def set_spot(self,spot):
        self.spot_price=spot

    def __init__(self,**kwargs):
        param=GeometricAsianOption.get_param(**kwargs)
        self.today_date= date_to_quantlib(param.get("today_date"))
        self.maturity_date= date_to_quantlib(param.get("maturity_date"))
        self.strike_price= param.get("strike_price")
        self.spot_price = param.get("spot_price")
        self.risk_free_rate = param.get("risk_free_rate")
        self.dividend_rate = param.get("dividend_rate")
        self.volatility = param.get("volatility")
        if param.get("option_type")=='call':
            self.option_type= ql.Option.Call
        else:
            self.option_type= ql.Option.Put
        self.fixingDates= param.get("fixingDates")
        self.time_to_maturity = ql.Actual365Fixed().yearFraction(self.today_date, self.maturity_date)

    def f(self,x):
        if self.option_type == ql.Option.Call:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*np.maximum(0,np.exp(np.log(x[:,0,:]).mean(axis=1))-self.strike_price)
        else:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*np.maximum(0,self.strike_price-np.exp(np.log(x[:,0,:]).mean(axis=1)))
    
    def get_times(self,**kwargs):
        all_dates=[self.today_date]+[date_to_quantlib(i) for i in self.fixingDates]+[self.maturity_date]
        return [ql.Actual365Fixed().yearFraction(self.today_date, i) for i in all_dates]    

class AsianClosedFormula(pricer):
    def price(self,**kwargs):
        today_date = Asian_param_getter.get_today_date(**kwargs)
        risk_free_rate = Asian_param_getter.get_risk_free_rate(**kwargs)
        dividend_rate = Asian_param_getter.get_dividend_rate(**kwargs)
        strike = Asian_param_getter.get_strike(**kwargs)
        option_type = Asian_param_getter.get_option_type(**kwargs)
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = Asian_param_getter.get_option_type(**kwargs)
        maturity_date = Asian_param_getter.get_maturity_date(**kwargs)
        volatility = Asian_param_getter.get_volatility(**kwargs)
        spot = Asian_param_getter.get_spot(**kwargs)
        fixingdates= Asian_param_getter.get_fixing_dates(**kwargs)
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today_date, risk_free_rate, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today_date, dividend_rate, ql.Actual365Fixed()))
        exercise = ql.EuropeanExercise(maturity_date)
        ql.Settings.instance().evaluationDate = today_date

        volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today_date, ql.UnitedKingdom(), volatility, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(spot)), dividendTS, riskFreeTS, volTS)
        geometricRunningAccumulator = 1.0
        fixingdates=[today_date]+[date_to_quantlib(i) for i in fixingdates]+[maturity_date]
        discreteGeometricAsianOption = ql.DiscreteAveragingAsianOption(ql.Average().Geometric, geometricRunningAccumulator, 0,fixingdates , payoff, exercise)
        discreteGeometricAsianOption.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))
        return discreteGeometricAsianOption.NPV()

#################################################################################################################
################################################ Binary Option #################################################
################################################################################################################# 
class Binary_param_getter(option_param_getter):
    def get_strike(**kwargs): return kwargs['BinaryOption']['strike_price']
    def get_maturity_date(**kwargs): return get_date(kwargs['BinaryOption']['maturity_date'])
    def get_profit(**kwargs): return kwargs['BinaryOption']['option_profit']
    def get_today_date(**kwargs): return get_date(kwargs['BinaryOption']['today_date'])

class BinaryOption(payoff):

    def get_param(**kwargs):
        return kwargs.get('BinaryOption')

    def set_spot(self,spot):
        self.spot_price=spot

    def __init__(self,**kwargs):
        param=BinaryOption.get_param(**kwargs)
        self.today_date= date_to_quantlib(param.get("today_date"))
        self.maturity_date= date_to_quantlib(param.get("maturity_date"))
        self.strike_price= param.get("strike_price")
        self.spot_price = param.get("spot_price")
        self.risk_free_rate = param.get("risk_free_rate")
        if param.get("option_type")=='call':
            self.option_type= ql.Option.Call
        else:
            self.option_type= ql.Option.Put
        self.option_profit=param.get("option_profit")
        self.time_to_maturity = ql.Actual365Fixed().yearFraction(self.today_date, self.maturity_date)

    def f(self,x):
        if self.option_type == ql.Option.Call:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*self.option_profit*np.heaviside(x[:,0,-1]-self.strike_price,1)
        else:
            return math.exp(-self.risk_free_rate*self.time_to_maturity)*self.option_profit*np.heaviside(self.strike_price-x[:,0,-1],1)
    
    def get_times(self,**kwargs):
        return [0,self.time_to_maturity]  

class BinaryClosedFormula(pricer):
    def price(self,**kwargs):
        getter = Binary_param_getter()
        today_date = getter.get_today_date(**kwargs)
        risk_free_rate = getter.get_risk_free_rate(**kwargs)
        dividend_rate = getter.get_dividend_rate(**kwargs)
        strike = getter.get_strike(**kwargs)
        option_type = getter.get_option_type(**kwargs)
        binaryPayoff = ql.CashOrNothingPayoff(option_type, strike, 1)
        exercise = getter.get_option_type(**kwargs)
        maturity_date = getter.get_maturity_date(**kwargs)
        volatility = getter.get_volatility(**kwargs)
        spot = getter.get_spot(**kwargs)
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today_date, risk_free_rate, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today_date, dividend_rate, ql.Actual365Fixed()))
        european_exercise = ql.EuropeanExercise(maturity_date)
        ql.Settings.instance().evaluationDate = today_date
        time_to_maturity=ql.Actual365Fixed().yearFraction(today_date, maturity_date)
        volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today_date, ql.UnitedKingdom(), volatility, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(spot)), dividendTS, riskFreeTS, volTS)
    
        BinaryOption = ql.VanillaOption(binaryPayoff, european_exercise)
        BinaryOption.setPricingEngine(ql.AnalyticbinaryEngine(process))
        return BinaryOption.NPV()    


#################################################################################################################
################################################ binary Option #################################################
#################################################################################################################
class Binary_param_getter(option_param_getter):
    def get_maturity_date(**kwargs): return get_date(kwargs['BinaryOption']['maturity_date'])
    def get_today_date(**kwargs): return get_date(kwargs['BinaryOption']['today_date'])
    def get_strike(**kwargs): return kwargs['BinaryOption']['strike_price']

class BinaryOption(payoff):
    def get_param(**kwargs):
        return kwargs.get('BinaryOption')

    def set_spot(self,spot):
        self.spot_price=spot

    def __init__(self,**kwargs):
        param=BinaryOption.get_param(**kwargs)
        if param.get("option_type")=='call':
            self.option_type= ql.Option.Call
        else:
            self.option_type= ql.Option.Put
        self.timesteps = param.get("timesteps")        
        self.today_date= date_to_quantlib(param.get("today_date"))
        self.maturity_date= date_to_quantlib(param.get("maturity_date"))
        self.strike_price= param.get("strike_price")
        self.spot_price = param.get("spot_price")
        self.risk_free_rate = param.get("risk_free_rate")
        self.dividend_rate = param.get("dividend_rate")
        self.volatility = param.get('volatility')
        self.time_to_maturity = ql.Actual365Fixed().yearFraction(self.today_date, self.maturity_date)
       

    def f(self,x):
        discount=math.exp(-self.risk_free_rate*self.time_to_maturity)
        if self.option_type == ql.Option.Call:
            return discount*np.heaviside(x[:,0,-1],self.strike_price)
        else:
            return discount*np.heaviside(self.strike_price, x[:,0,-1])
    
    def get_times(self,**kwargs): 
        return [0,self.time_to_maturity] 

#################################################################################################################
############################################# Stochastic processes ##############################################
#################################################################################################################

class BSM(StochasticProcess):
    def get_param(**kwargs):
        return kwargs.get('BSM')

    def get_process(self,**kwargs):
        getter = option_param_getter()
        initialValue = getter.get_spot(**kwargs)
        initialValue = get_float(initialValue)
        risk_free_rate = getter.get_risk_free_rate(**kwargs)
        dividend_rate= getter.get_dividend_rate(**kwargs)
        volatility= getter.get_volatility(**kwargs)
        today = date_to_quantlib(getter.get_today_date(**kwargs))
        # today = date_to_quantlib(getter.get_today_date(**kwargs))
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
        volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), volatility, ql.Actual365Fixed()))
        BSMprocess = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(initialValue)), dividendTS, riskFreeTS, volTS)
        return BSMprocess

class BSMultiple(BSM):
    def get_param(**kwargs): return kwargs.get('BSMultiple')
    def get_processes_param(**kwargs): return BSMultiple.get_param(**kwargs)['BSMs']
    def get_processes(self,**kwargs): 
        # return [super().get_process(**k) for k in BSMultiple.get_processes_param(**kwargs)]
        test = []
        for k in BSMultiple.get_processes_param(**kwargs) :
            test.append(super().get_process(**{**kwargs,**k}))
        return test
    def get_correlation(**kwargs): return BSMultiple.get_param(**kwargs)['correlation_matrix']
    def get_process(self,**kwargs):
        matrice_correlation=BSMultiple.get_correlation(**kwargs)
        multiProcess = ql.StochasticProcessArray(self.get_processes(**kwargs), matrice_correlation)
        return multiProcess 


class Heston(StochasticProcess):
    def get_param(**kwargs):
        return kwargs.get('Heston')

    def get_process(self,**kwargs):
        param=Heston.get_param(**kwargs)
        initialValue = param.get("spot_price")
        risk_free_rate = param.get("risk_free_rate")
        dividend_rate= param.get("dividend_rate")
        v0= param.get("v0")
        kappa = param.get('kappa')
        theta = param.get('theta')
        rho = param.get('rho')
        sigma = param.get('sigma')
        today = ql.Date().todaysDate()
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
        hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, ql.QuoteHandle(ql.SimpleQuote(initialValue)), v0, kappa, theta, sigma, rho)
        return hestonProcess


#################################################################################################################
##########################"##################### Path generator #################################################
#################################################################################################################

class QL_path_generator(path_generator):

    def get_date_fraction(list):
        out=[]
        start=list[0]
        for i in list:
            out.append(ql.Actual365Fixed().yearFraction(date_to_quantlib(start), date_to_quantlib(i)))
        return out



    def generate(self,N,payoff=None,time_list=None,**kwargs): #N= nb of trajectories
        process=path_generator.get_param(**kwargs).get('process') # utiliser la classe Heston process dans kwargs du type process: // 'paths_generator': {'process':Heston()} ((dictionnaire du type QL_params()))
        seed=kwargs.get('seed',None)
        if seed==None:
            seed=np.random.randint(0,1000000)
        antithetic=path_generator.get_param(**kwargs).get('antithetic')
        D = QL_path_generator.get_D(**kwargs)
        if time_list is None:
            time_list= payoff.get_times(**kwargs)
        if not(isinstance(time_list[0],float)):
            time_list=QL_path_generator.get_date_fraction(time_list)
        T=len(time_list)
        out=np.zeros((N,D,T))
        if seed is not None: rng = ql.UniformRandomSequenceGenerator(D * (T-1), ql.UniformRandomGenerator(seed=seed))
        else: rng = ql.UniformRandomSequenceGenerator(D * (T-1), ql.UniformRandomGenerator())
        # if seed is not None: rng = ql.UniformRandomSequenceGenerator(D * T, ql.UniformRandomGenerator(seed=seed))
        # else: rng = ql.UniformRandomSequenceGenerator(D * T, ql.UniformRandomGenerator())
        sequenceGenerator = ql.GaussianRandomSequenceGenerator(rng)
        process=path_generator.get_process(**kwargs)
        pathGenerator = ql.GaussianMultiPathGenerator(process,time_list, sequenceGenerator, False)
        if antithetic==True:
            for i in range(0,N,2):
                samplePath = pathGenerator.next()
                values = samplePath.value()
                for j in range(D):
                    out[i,j]=np.array(values[j])

                antitheticSamplePath = pathGenerator.antithetic()
                antitheticValues = antitheticSamplePath.value()
                for j in range(D):
                    out[i+1,j]=np.array(antitheticValues[j])
        else:
             for i in range(0,N,1):
                samplePath = pathGenerator.next()
                values = samplePath.value()
                for j in range(D):
                    out[i,j]=np.array(values[j])
        return out

def QL_params():
    params = {
    'GeometricAsianOption':{
                'option_type' : 'call',
                'today_date': datetime.date.today(),
                'maturity_date' : datetime.date(2022,5,15),
                'strike_price' : 102,
                'risk_free_rate' : 0,
                'dividend_rate' : 0,
                'number_of_assets':1,
                'spot_price' : 100,
                'volatility' : 0.1,
                'fixingDates': [datetime.date(2022,3,15),datetime.date(2022,4,15)]
    },
    'BinaryOption':{
                        'option_type' : 'call',
                        'today_date': datetime.date.today(),
                        'maturity_date' : datetime.date(2022,4,15),
                        'option_profit' : 2,
                        'strike_price' : 105,
                        'risk_free_rate' : 0,
                        'dividend_rate' : 0,
                        'number_of_assets':1,
                        'spot_price' : 100,
                        'volatility' : 0.1,
                    },
    'BarrierOption':{
                        'option_type' : 'call',
                        'barrier_type' :'UpOut',
                        'barrier' : 109,
                        'rebate':0.,
                        'today_date': datetime.date.today(),
                        'maturity_date' : datetime.date(2022,4,15),
                        'strike_price' : 105,
                        'risk_free_rate' : 0.,
                        'dividend_rate' : 0.,
                        'number_of_assets':1,
                        'spot_price' : 100,
                        'volatility' : 0.1,
                    },

    'BasketOption' :{
                    'option_type' : 'call',
                    'today_date': datetime.date.today(),
                    'maturity_date' : datetime.date(2022,4,15),
                    'strike_price' : 105,
                    'risk_free_rate' : 0,
                    'dividend_rate' : 0,
                    'number_of_assets':2,
                    'weights':[0.5,0.5],
                    'spot_price' : [100., 100.],
                    'volatility' : [0.1, 0.1],
                    'matrice_correlation' : [[1, 1 ], 
                                            [1, 1, ]]
                    },
    'BSMultiple':{
                'spot_price' : [100., 100.],
                'risk_free_rate' : 0,
                'dividend_rate' : 0,
                'volatility' : [0.1, 0.1],
                'matrice_correlation' : [[1, 1, ], 
                                        [1, 1, ]]
                },
    'BSM':{
                'spot_price' : 50,
                'risk_free_rate' : 0.0,
                'dividend_rate' : 0,
                'volatility' : 0.25,
                },
    'Heston':{
                # 'spot_price' : 100,
                # 'risk_free_rate' : 0.02,
                # 'dividend_rate' : 0,        
                # 'v0':0.025,
                # 'kappa':1.5,
                # 'theta':0.0225,
                # 'rho':-0.5,
                # 'sigma':0.25

                #### DAX PARAMS ####
                'spot_price' : 50,
                'risk_free_rate' : 0.025,
                'dividend_rate' : 0,        
                'v0':0.025,
                'kappa':1.22,
                'theta':0.025,
                'rho':-0.66,
                'sigma':0.025               
            },                         
    }
    return params









