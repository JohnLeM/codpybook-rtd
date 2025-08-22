import os 
import abc 

import numpy as np
import pandas as pd 

from codpy import core
from codpy.kernel import Sampler, kernel_setter
from codpy.permutation import map_invertion
from codpy.utils import get_matrix
from codpy.data_conversion import get_float
from codpy.plot_utils import multi_plot
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

from utils.ch9.mapping import apply_map, inverse
from utils.ch9.market_data import ts_data
from utils.ch9.plot_utils import plot_trajectories
from utils.ch9.ql_tools import QL_path_generator 


def get_time(csv,**kwargs): 
    return np.asarray(get_float(csv.index))

def generate_distribution_from_samples_np(kwargs):
    """
        Generate noise induced by the model (mapping) from a learned distribution. 
    """
    samples = kwargs['samples']
    sample_times = kwargs.get('sample_times',None)
    mapping = kwargs.get("map",None)
    
    if kwargs.get('transform_h', None) is None:
        mapped_ = mapping(samples,kwargs)
        xm = mapped_.reshape((mapped_.shape[0]*mapped_.shape[1],mapped_.shape[2])).T
        T = mapped_.shape[2]
    else:
        mapped_ = kwargs['transform_h'].values
        xm = mapped_
        T = mapped_.shape[0] // samples.shape[0]
    
    if sample_times is not None:
        T = min(len(sample_times), T)

    sampler = kwargs.get('sampler',None)
    if sampler is None:
        generator = lambda n: np.random.uniform(size=(n, xm.shape[1]))
        sampler = Sampler(xm, latent_generator=generator)
    
    assert sampler.get_x().shape == xm.shape, f"Sampler's x shape {sampler.get_x().shape} does not match the mapped data shape {xm.shape}."

    if kwargs.get('reproductibility',False) :
        Nz = 1
        generated_noise = np.zeros(shape=(Nz, mapped_.shape[1], T))
        sample = sampler(sampler.get_x()) 
        generated_noise[0] = sample[map_invertion(sampler.permutation)].T
        return generated_noise 
    
    Nz = kwargs['Nz']
    generated_noise = sampler.sample(Nz*T)
    generated_noise = generated_noise.reshape(Nz, -1, T)
    # temp_mean = np.mean(generated_noise, axis=0)
    # generated_noise -= temp_mean[np.newaxis, :, :]  
    # generated_noise += np.mean(mapped_, axis=0)[np.newaxis, :, :]
    return generated_noise

def generate_from_samples_np(**kwargs):
    """
        Generate noise induced by the model and reconstruct the paths using the model (mapping) definition.
    """
    out = generate_distribution_from_samples_np(kwargs)
    mapping = kwargs.get("map",None)
    if mapping is not None: 
        out = inverse(mapping)(out,kwargs)
    return out
    
def RegenerateHistory(**kwargs):
    data = kwargs.get("data",None)
    if data is None: data = ts_data.get_yf_ts_data(**kwargs['yf_param'])
    timelist = kwargs.get("timelist",None)
    if timelist is None: timelist = kwargs['times']

    samples=np.zeros((1,data.shape[1]+1,data.shape[0]))
    samples[0,1:,:]= get_matrix(data).T
    samples[0,0,:] =timelist
    initial_values = samples[0,1:,0]
    time_start= timelist[0]
    kwargs['Nz'] = kwargs.get('Nz',10)
    params={**kwargs,**{'initial_values': initial_values }}
    params['sample_times'] = timelist
    f_z = generate_from_samples_np(samples = samples,time_start=time_start,**params)

    return f_z,data,samples

def generate_paths(params = None):
    if params is None: params = apply_map()
    params['fz'],params['data'],params['fx']=RegenerateHistory(**params)
    def graphic(params):
        params['Nz'] = params.get('Nz',10)
        plot_indice = [1,2,3]
        plot_trajectories({**params,**{'plot_indice':plot_indice}})
        
    params['graphic'] = graphic
    # params['stats_mean'] = stats_df(params['data'], np.mean(f_z, axis=0).T[:,1:]).T
    # plt.plot(f_z[0,1,:])
    # plt.plot(params['data'].values[:,0])
    # 
    return params

class anais_path_generator(QL_path_generator):

    def __init__(self,kwargs=None):
        super().__init__()
        if kwargs is None: self.param=get_model_param()
        else: self.param = kwargs
        self.param['data']=np.squeeze(self.param['data'],axis=0)
    def generate(self,N,payoff,time_list=None,**kwargs):
        time_list=get_time(self.param['data'])
        time_list-= get_float(payoff.today_date)
        time_list/=365.

        self.path=super().generate(1,payoff,time_list,**kwargs)
        kwargs['data']=np.squeeze(self.path,axis=0).T
        
        f_z,data,f_x=RegenerateHistory(**{**self.param,**kwargs}) 
        return f_z

class data_generator(abc.ABC):
    index = []
    def get_index(self):
        return self.index
    index_z = []
    def get_index_z(self):
        if not len(self.index_z):
            self.index_z = [str(i) for i in range(0,len(self.fz))]
        return self.index_z

    @abc.abstractmethod
    def get_data(self,**kwargs):
        pass

    def id(self,name = "data_generator"):
        return name


    def set_data(self,**kwargs):
        """
        Make sure everything is on the right format & length
        """
        self.D,self.Nx,self.Ny,self.Nz,self.Df = int(kwargs.get("D",0)),int(kwargs.get("Nx",0)),int(kwargs.get("Ny",0)),int(kwargs.get("Nz",0)),int(kwargs.get("Df",0))
        self.x,self.y,self.z,self.fx,self.fy,self.fz,self.dfx,self.dfz = [],[],[],[],[],[],[],[]
        self.map_ids = []
        if self.Ny >0 & self.Nx >0 : self.Ny = min(self.Ny,self.Nx)
        def crop(x,Nx):
            if isinstance(x,list):return [crop(y,Nx) for y in x]
            if (Nx>0 and Nx < x.shape[0]):return x[:Nx]
            return x

        if abs(self.Nx)*abs(self.Ny)*abs(self.Nz) >0:
            self.x, self.fx, self.y, self.fy, self.z, self.fz = self.get_data(**kwargs)
            self.column_selector(**kwargs)
            if bool(kwargs.get('data_generator_crop',True)):
                # TODO make sure this isn't croping twice since we might do it in get_data? 
                # Or just redundant cuz Nx = x 
                self.x,self.fx,self.y,self.fy,self.z,self.fz = crop(self.x,self.Nx),crop(self.fx,self.Nx),crop(self.y,self.Ny),crop(self.fy,self.Ny),crop(self.z,self.Nz),crop(self.fz,self.Nz)
            self.Ny = core.get_matrix(self.y).shape[0]
            if  not isinstance(self.z,list): self.Nz = core.get_matrix(self.z).shape[0]
            self.Nx = core.get_matrix(self.x).shape[0]
            self.D = core.get_matrix(self.x).shape[1]
            if (len(core.get_matrix(self.fx))):
                if self.fx.ndim == 1: self.Df = 1
                else:self.Df = self.fx.shape[1]
    def get_nb_features(self):
        return self.fx.shape[1]

    def copy_data(self,out):
        out.x,out.y,out.z,out.fx,out.fy,out.fz, out.dfx,out.dfz = self.x.copy(),self.y.copy(),self.z.copy(),self.fx.copy(),self.fy.copy(),self.fz.copy(),self.dfx.copy(),self.dfz.copy()
        out.D,out.Nx,out.Ny,out.Nz = self.D,self.Nx,self.Ny,self.Nz
        return out

    def get_input_data(self):
        return self.D,self.Nx,self.Ny,self.Nz,self.Df
    def get_output_data(self):
        # print(self.x)
        # print(self.fx)
        return self.x,self.y,self.z,self.fx,self.fy,self.fz,self.dfx,self.dfz

    def get_params(**kwargs) :
        return kwargs.get('data_generator',None)

    def column_selector(self,**kwargs):
        """If we require feature engineering
        """
        params = data_generator.get_params(**kwargs)
        if params is None : return
        params = params.get('variables_selector',None)
        if params is None : return
        variables_cols_drop = params.get('variables_cols_drop',[])
        variables_cols_keep = params.get('variables_cols_keep',[])
        values_cols_drop = params.get('values_cols_drop',[])
        values_cols_keep = params.get('values_cols_keep',[])

        if len(variables_cols_drop) or len(variables_cols_keep):
            self.x = column_selector(self.x,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.y = column_selector(self.y,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.z = column_selector(self.z,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
        if len(values_cols_drop) or len(values_cols_keep):
            self.fx = column_selector(self.fx,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            self.fy = column_selector(self.fy,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            self.fz = column_selector(self.fz,cols_drop = values_cols_drop, cols_keep = values_cols_keep)

    def save_cd_data(object,**params):
        if params is not None:
            save_cv_data = params.get('save_cv_data',None)
            if save_cv_data is not None:
                index = save_cv_data.get('index',False)
                header = save_cv_data.get('header',False)
                x_csv,y_csv,z_csv,fx_csv,fz_csv,f_z_csv = save_cv_data.get('x_csv',None),save_cv_data.get('y_csv',None),save_cv_data.get('z_csv',None),save_cv_data.get('fx_csv',None),save_cv_data.get('fz_csv',None),save_cv_data.get('f_z_csv',None)
                if x_csv is not None :  save_to_file(object.x, file_name=x_csv,sep = ';', index=index, header=header)
                if y_csv is not None :  save_to_file(object.y, file_name=y_csv,sep = ';', index=index, header=header)
                if z_csv is not None :  save_to_file(object.z, file_name=z_csv,sep = ';', index=index, header=header)
                if fx_csv is not None :  save_to_file(object.fx, file_name=fx_csv,sep = ';', index=index, header=header)
                if fz_csv is not None :  save_to_file(object.fz, file_name=fz_csv,sep = ';', index=index, header=header)
                if f_z_csv is not None :  save_to_file(object.f_z, file_name=f_z_csv,sep = ';', index=index, header=header)

    def __init__(self,**kwargs):
        self.set_data(**kwargs)

class California_data_generator(data_generator):
    x_raw, fx_raw, z_raw, fz_raw = [],[],[],[]
    def set_raw_data(self, **kwargs):
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        x, fx = housing.data, housing.target
        a = np.arange(len(x))
        np.random.seed(42)
        np.random.shuffle(a)
        x = x[a]
        fx = fx.reshape((len(fx),1))
        fx = fx[a]
        California_data_generator.x_raw, California_data_generator.fx_raw = pd.DataFrame(x),pd.DataFrame(fx)

    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        if len(California_data_generator.x_raw) == 0:
            self.set_raw_data()
        length = len(California_data_generator.x_raw)
        return California_data_generator.x_raw[0:Nx], California_data_generator.fx_raw[0:Nx], California_data_generator.x_raw[0:Nx], California_data_generator.fx_raw[0:Nx], California_data_generator.x_raw, California_data_generator.fx_raw

    def get_feature_names(self):
        from sklearn import datasets
        return datasets.load_boston().feature_names
    def copy(self):
        return self.copy_data(Boston_data_generator())
    
class Boston_data_generator(data_generator):
    x_raw, fx_raw, z_raw, fz_raw = [],[],[],[]
    def set_raw_data(self, **kwargs):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        x, fx = datasets.load_boston(return_X_y=True)
        a = np.arange(len(x))
        np.random.seed(42)
        np.random.shuffle(a)
        x = x[a]
        fx = fx.reshape((len(fx),1))
        fx = fx[a]
        Boston_data_generator.x_raw, Boston_data_generator.fx_raw = pd.DataFrame(x),pd.DataFrame(fx)

    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        if len(Boston_data_generator.x_raw) == 0:
            self.set_raw_data()
        length = len(Boston_data_generator.x_raw)
        return Boston_data_generator.x_raw[0:Nx], Boston_data_generator.fx_raw[0:Nx], Boston_data_generator.x_raw[0:Nx], Boston_data_generator.fx_raw[0:Nx], Boston_data_generator.x_raw, Boston_data_generator.fx_raw

    def get_feature_names(self):
        from sklearn import datasets
        return datasets.load_boston().feature_names
    def copy(self):
        return self.copy_data(Boston_data_generator())
    
def tensor_vectorize(fun, x):
    N,D = x.shape[0],x.shape[1]
    E = len(fun(x[0]))
    out = np.zeros((N, E))
    for n in range(0, N):
        out[n] = fun(x[n])
    return out.reshape(N,E,1)

class data_random_generator(data_generator):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.fun = kwargs.get("fun",None)
        self.nabla_fun = kwargs.get("nabla_fun",None)
        self.type = kwargs.get("type","sto")
        self.X_box = kwargs.get("X_box",None)
        self.Y_box,self.Z_box = kwargs.get("Y_box",self.X_box),kwargs.get("Z_box",self.X_box)
        self.D = 0
        if self.X_box is not None:self.D = self.X_box.shape[1]
        self.X_min = kwargs.get("X_min",-1.)
        self.X_max = kwargs.get("X_max",1.)
        self.Y_min = kwargs.get("Y_min",-1.)
        self.Y_max = kwargs.get("Y_max",1.)
        self.Z_min = kwargs.get("Z_min",-1.5)
        self.Z_max = kwargs.get("Z_max",1.5)
        self.types = kwargs.get("types",["sto","sto","sto"])
        self.seeds = kwargs.get("seeds",[42,35,52])


    def get_raw_data(self,**kwargs):
        import numpy as np
        D,Nx,Ny,Nz,Df = int(kwargs.get("D",self.D)),int(kwargs.get("Nx",0)),int(kwargs.get("Ny",0)),int(kwargs.get("Nz",0)),int(kwargs.get("Df",0))
        def cartesian_product(*arrays):
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[...,i] = a
            return arr.reshape(-1, la)

        def get_array(**kwargs):
            N,D = kwargs['N'],kwargs['D']
            if N==0: return []
            import itertools as it
            type = kwargs.get("type","sto")
            seed = kwargs.get("seed",0)
            box = kwargs.get("box",None)
            if box is None:
                min_,max_ = kwargs.get("min",None),kwargs.get("max",None)
                box = np.zeros((2,D))
                box[0,:],box[1,:] = min_, max_
            if type == "sto" or D>=5:
                if seed : np.random.seed(seed)
                out = np.zeros((N,D))
                def helper(d): out[:,d] = np.random.uniform(box[0,d],box[1,d],size=N)
                [helper(d) for d in range(0,D)]
                return out
                # return np.random.uniform(min, max, size = (N,D))
            if D>1:
                N = int(pow(N,1./D))+1
                v = [np.arange(start=box[0,d],stop=box[1,d]+0.00001,step = (box[1,d]-box[0,d]) / N ) for d in range(0,D)]
                v = cartesian_product(*(v))
                if D==1: return np.asarray(v).reshape(len(v),D)
                v = np.asarray(v)
                return v
            return get_matrix(np.asarray(np.arange(start=box[0,0],stop=box[1,0]+0.00001,step = (box[1,0]-box[0,0]) / N )))

        types = kwargs.get('types',self.types)
        x = get_array(**{**kwargs,**{'type':types[0],'seed':self.seeds[0],'box':self.X_box,'min':self.X_min,'max':self.X_max,'N':Nx}})
        y = get_array(**{**kwargs,**{'type':types[1],'seed':self.seeds[1],'box':self.Y_box,'min':self.Y_min,'max':self.Y_max,'N':Ny}})
        z = get_array(**{**kwargs,**{'type':types[2],'seed':self.seeds[2],'box':self.Z_box,'min':self.Z_min,'max':self.Z_max,'N':Nz}})
        if kwargs.get("sort",False):
            x,y,z = x[np.argsort(a = x[:,0])],y[np.argsort(a = y[:,0])],z[np.argsort(a = z[:,0])]

        Nx,Ny,Nz = len(x), len(y), len(z)
        if self.fun != None:
            # fy = matrix_vectorize(self.fun,y)
            # fx = matrix_vectorize(self.fun,x)
            # fz = matrix_vectorize(self.fun,z)
            fy = get_matrix(self.fun(y))
            fx = get_matrix(self.fun(x))
            fz = get_matrix(self.fun(z))
            if self.nabla_fun != None:
                nabla_fx = tensor_vectorize(self.nabla_fun,x)
                nabla_fz = tensor_vectorize(self.nabla_fun,z)
                return(x,y,z,fx,fy,fz,nabla_fx,nabla_fz,Nx,Ny,Nz)
            else:
                return(x,y,z,fx,fy,fz,Nx, Ny, Nz)
        else:
            return(x,y,z,Nx,Ny,Nz)

    def get_data(self,**kwargs):
        D,Nx,Ny,Nz,Df = int(kwargs.get("D",self.D)),int(kwargs.get("Nx",0)),int(kwargs.get("Ny",0)),int(kwargs.get("Nz",0)),int(kwargs.get("Df",0))
        if (D*Nx) or (D*Ny) or (D*Nz):
            x,y,z,fx,fy,fz,Nx, Ny, Nz = self.get_raw_data(**kwargs)
            return x,fx,y,fy,z,fz

    def copy(self):
        return self.copy_data(data_random_generator())

def normal_wrapper(center,size,**kwargs):
    scale=kwargs.get('scale',1.0)
    out = np.random.normal(scale = scale,size=size)
    out += center
    return out


def multimodal_datas(random_variable = normal_wrapper,**kwargs):
    N = kwargs.get('N', 500)
    D = kwargs.get('D', 1)
    columns = kwargs.get('columns', [str(n) for n in range(D)])
    C = kwargs.get('C', 2)

    radius = kwargs.get('radius', np.random.normal(size = (C)))
    centers = kwargs.get('centers', np.random.normal(size = (C, D)))[:C]
    if C > 1 : centers -= np.mean(centers, axis=0)
    weights = kwargs.get('weights', np.repeat(1./C,C))
    x = pd.DataFrame(columns = columns)
    fx = pd.DataFrame(columns = ["values"])
    for c,r,ind in zip(centers, radius,range(C)):
        size = int(N * weights[int(ind)])
        temp = pd.DataFrame(random_variable(center=c,size = (size,D),radius=r,**kwargs), columns = x.columns)
        x = pd.concat([x,temp])
        temp = pd.DataFrame([str(ind) for n in range(size)], columns = fx.columns)
        fx = pd.concat([fx,temp])
    return x,fx

DEFAULT_PDE = {
        'rescale_kernel':{'max': 1000, 'seed':42},
        'discrepancy:xmax':500,
        'discrepancy:ymax':500,
        'discrepancy:zmax':500,
        'discrepancy:nmax':1000,
        # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, 0,1e-8,map_setters.set_unitcube_map),
        # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_matern_tensor_kernel, 0,0 ,map_setters.set_standard_mean_map),
        # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_absnorm_kernel, 0,0 ,map_setters.set_unitcube_map),
        'set_codpy_kernel': kernel_setter("tensornorm", "scale_to_unitcube", 0, 0),
        # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel, 0,1e-8 ,map_setters.set_unitcube_map),
        # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 2,1e-8 ,map_setters.set_standard_mean_map),
        # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 2,1e-8 ,map_setters.set_scale_factor_map),
        # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8 ,map_setters.set_standard_min_map),
        # 'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2,1e-8 ,map_setters.set_standard_min_map),
        'rescale':True,
        'Nx':400,
        'D':2,
        'Ny':400,
        'Nz':400,
        'seed':43,
        'elev':20    
    }
def multimodal_mesh_generator(params = None):
    if params is None: params = DEFAULT_PDE
    epsilon = params.get('epsilon',10.)
    params['C'] = params.get('C',2)
    params['Dx'] = params.get('Dx',2)
    params['D'] = params.get('D',2)
   
    centers = params.get('centers',np.random.normal(size=[params['C'],params['D']]))
    centers -= np.mean(centers, axis=0)
    params['centers'] = centers
    params['N'] = params.get('N',500)
    params['random_variable'] = params.get('random_variable',normal_wrapper)
    params['distance'] = params.get('distance','norm22')
    x,labels = multimodal_datas(**params)
    x = get_matrix(x)
    def my_fun(x): return np.asarray([np.exp(-epsilon*np.linalg.norm(x[n]-centers[0])) for n in range(0,x.shape[0])])
    labels = params.get('fun',my_fun)(x)
    def scatterplot(params):
        import seaborn as sns
        datas = params.get('datas',[[params['x'],params['fx']],[params['x'],params['sol']]])
        def plot_helper(test,**params):
            x,fx = get_matrix(test[0]),get_matrix(test[1])
            if len(fx): test = pd.DataFrame(np.concatenate([x,fx],axis=1),columns = ['0','1','label'])
            else :
                test = pd.DataFrame(x[:,0:2],columns = ['0','1'])
                test['label'] = 0.
            if test.values.shape[1] == 2:test['1'] = test['label']
            sns.scatterplot(test, x=test['0'], y=test['1'], size='label',hue="label")
        multi_plot(datas,fun_plot=plot_helper,**params)
    params['x'],params['y'],params['z'],params['fx'] = get_matrix(x),get_matrix(x),get_matrix(x),get_matrix(labels)
    params['sol'] = params['fx'].copy()
    params['graphic'] = scatterplot

    return params

class raw_data_generator(data_generator):
    x_data  = []
    @abc.abstractmethod
    def get_raw_data(self,**kwargs):
        pass
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        if len(self.x_data) == 0:
            self.x_data = self.get_raw_data(**kwargs)
        return self.x_data, [], self.x_data, [], self.x_data, []
    