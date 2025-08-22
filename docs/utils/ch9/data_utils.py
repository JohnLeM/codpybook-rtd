import numpy as np
import pandas as pd
import datetime
import torch
import scipy 

from codpy.data_conversion import get_float
from codpy.kernel import get_matrix
from codpy.core import Misc
def get_time(csv,**kwargs): 
    return np.asarray(get_float(csv.index))


def df_summary(df: pd.DataFrame, **kwargs) :
    """
    author: SMI
    The function outputs summuary statistics.
    """
    out = pd.DataFrame(columns = ['Mean', 'Variance', 'Skewness', 'Kurtosis'], index = df.columns)
    F = kwargs.get('format', "{:.2g}")
    assert isinstance(df, pd.DataFrame), "Data is not pandas data frame."
    out['Mean'] = df.mean(axis=0)
    out['Variance'] = df.skew(axis=0)
    out['Skewness'] = df.var(axis=0)
    out['Kurtosis'] = df.kurtosis(axis=0)
    # format = kwargs.get('format', "{:.2g}")
    format = kwargs.get('format', None)
    if F is not None: out.style.format(format)
    return out


def csv_to_np(csv,**kwargs):
    out = csv.copy()
    out.insert(loc=0, column='Date', value=get_time(out,**kwargs))
    out = out.values.T
    out = out.reshape(1,out.shape[0],out.shape[1])
    return out

get_datetime_switchDict = { str: lambda x,**kwargs :  datetime.datetime.strptime(x, kwargs.get('date_format','%d/%m/%Y')),
                        datetime.date:lambda x,**kwargs : datetime.combine(x, datetime.min.time()),
                        int : lambda x,**kwargs : get_datetime(float(x)),
                        float : lambda x,**kwargs : datetime.datetime.fromtimestamp(x),
                        np.float64 : lambda x,**kwargs : get_datetime(float(x)),
                        pd._libs.tslibs.timestamps.Timestamp : lambda x,**kwargs : pd.to_datetime(x),
                        torch.Tensor : lambda x,**kwargs: get_datetime(get_float(x)),
                        np.ndarray : lambda x,**kwargs: get_datetime(get_float(x)),
                        datetime.datetime: lambda x,**kwargs: x,
                    }
def get_datetime(x,**kwargs):
    if isinstance(x,list): return [get_datetime(n,**kwargs) for n in x]
    type_debug = type(x)
    method = get_datetime_switchDict.get(type_debug,None)
    return method(x,**kwargs)

def ks_testD(x,y,**kwargs):
    x,y, = get_matrix(x),get_matrix(y)
    alpha = kwargs.get("alpha",.05)
    Nx,Ny,D = x.shape[0],y.shape[0],x.shape[1]
    ks = []
    thsld = []
    for i in range(0,D):
        z = scipy.stats.ks_2samp(x[:,i],y[:,i])
        ks    += [z[1]]
        # thsld += [np.sqrt(-np.log(alpha/2)*(1+Nx/Ny)/(2*Ny))]
        thsld += [0.05]
    return ks, thsld

def hellinger(x, y):
    if scipy.sparse.issparse(x) and scipy.sparse.issparse(y):
        x = x.toarray()
        y = y.toarray()
    return np.sqrt(0.5 * ((np.sqrt(x) - np.sqrt(y))**2).sum())

def codpy_distance(p,q, type, rescale = True):
    if(type == 'H'):
        return hellinger(p,q)
    elif(type == 'D'):
        return Misc.discrepancy(p,q, rescale = rescale)

def kl_div(x,y):
    import torch
    import torch.nn.functional as F
    return F.kl_div(torch.Tensor(x).log_softmax(0), torch.Tensor(y).softmax(0)).detach().numpy()

def compare_distances(p,q):
    import pandas as pd
    KL = kl_div(p,q)
    D = codpy_distance(p,q,'D') 
    df = {'KL': [KL], 'MMD': [D]} 
    df = pd.DataFrame(df)     
    return df

get_stats_df_switchDict = {
                        np.ndarray : lambda x, **kwargs: np.array([get_float(z, **kwargs) for z in x]),
                        }
def stats_df(dfx, dfy, **kwargs):
    def transf(df1, df2,**kwargs):
        if isinstance(df1,list):return [transf(x,y) for x,y in zip(df1,df2)]
        format = kwargs.get('format', "{:.2g}")
        return str(format.format(df1))+'(' + format.format(df2) + ')'
    def distance(xy):
        dist = []
        for i in xy:
            temp = compare_distances(i[0],i[1])
            dist+= [temp.to_numpy().flatten()]
        dist = pd.DataFrame(dist, columns= temp.columns).T
        dist.columns = df.columns
        return dist.T
    test = type(dfx)
    if isinstance(dfx,list): 
        out = pd.concat([stats_df(fx, fy, **kwargs) for (fx,fy) in zip(dfx, dfy)])
        if kwargs.get("f_names",False): out.index = kwargs["f_names"]
        return out
    if not isinstance(dfx,pd.DataFrame): dfx=pd.DataFrame(dfx)
    if not isinstance(dfy,pd.DataFrame): dfy=pd.DataFrame(dfy)
    out = pd.DataFrame()
    summaryx = df_summary(dfx)
    summaryy = df_summary(dfy)
    ks_df, thrsld = ks_testD(dfx, dfy)

    out['Mean'] = transf(summaryx.Mean.to_list(), summaryy.Mean.to_list())
    out['Variance'] = transf(summaryx.Variance.to_list(), summaryy.Variance.to_list())
    out['Skewness'] = transf(summaryx.Skewness.to_list(), summaryy.Skewness.to_list())
    out['Kurtosis'] = transf(summaryx.Kurtosis.to_list(), summaryy.Kurtosis.to_list())
    out['KS test'] = transf(ks_df, thrsld)
    if kwargs.get("f_name",False): out.index = kwargs["f_name"]
    return out