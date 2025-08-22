import os 
import sys 

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

import torch 
import torch.nn as nn 

from codpy.core import kernel_setter, DiffOps, KerOp
from codpy.kernel import Kernel, get_matrix
from codpy.AAD import AAD 
from codpy.plot_utils import multi_plot, plot1D
from codpy.pde import CrankNicolson
from codpy.parallel import elapsed_time 
from codpy.metrics import get_relative_mean_squared_error 

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
data_path = os.path.join(CURRENT_DIR, "data")
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.ch7.pde import get_pde_params, pde_solver, regular_mesh_generator
from utils.ch9.path_generation import multimodal_mesh_generator, data_random_generator
from utils.ch9.plot_utils import plot_trisurf
from utils.pytorch_operators import * 

def my_fun_torch(x):
    import numpy as np
    from math import pi
    type_ = type(x)
    if not isinstance(x,torch.Tensor) : 
        x = torch.tensor(x, requires_grad=True)
    sinss = torch.cos(2 * x * pi)
    if x.dim() == 1 : 
        sinss = torch.prod(sinss, dim=0)
        ress = torch.sum(x, dim=0)
    else : 
        sinss = torch.prod(sinss, dim=1)
        ress = torch.sum(x, dim=1)
    return ress+sinss


# def set_global_chap4():
#     D,Nx,Ny,Nz=2,500,500,500
#     data_random_generator_ = data_random_generator(fun = my_fun,nabla_fun = nabla_my_fun, types=["cart","cart","cart"])
#     x,y,z,fx,fy,fz,nabla_fx,nabla_fz,Nx,Ny,Nz =  data_random_generator_.get_raw_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)

def testAAD(f,x):
    y = f(x)
    gradf=AAD.AAD.gradient(f,x)
    grad1 = gradf[:,0]
    gradf = AAD.AAD.hessian(f,x)
    grad2 = gradf[:,0]
    aa = x.detach().numpy()[:,0]
    yy = y.detach().numpy()[:,0]
    title_list = ["cubic", "1st derivative", "2nd derivative"]
    multi_plot([(aa,yy),(aa,grad1), (aa,grad2)], plot1D,mp_ncols = 3, f_names=title_list, mp_max_items = 4, mp_nrows = 1)
    
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
            "set_kernel": kernel_setter("gaussianper", None, 2, 1e-8),
        },
    }
def get_param22():
    return {
        "PytorchRegressor": {
            "epochs": 500,
            "layers": [128, 128, 128, 128],
            "loss": nn.MSELoss(),
            "activations": [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
            "optimizer": torch.optim.Adam,
        },
        "codpy_param": {
            "set_kernel": kernel_setter("gaussianper", None, 0, 1e-8),
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
    set_kernel = kernel_setter("gaussianper", None, 2, 1e-8)
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


def multimodal_lagrangian_mesh_generator(params):
    params =  multimodal_mesh_generator(params)
    params['sol'] = params['x'].copy()    
    return params
def heat_lagrangian_step(**kwargs):
    N,D = kwargs['x'].shape[0],kwargs['x'].shape[1]
    params = kwargs.copy()
    D = KerOp.dnm(x=kwargs['sol'],y=kwargs['sol'], distance = 'norm2', kernel_ptr=kwargs.get('kernel_ptr',None))
    def helper(n): D[n,n] = 1e+20
    [helper(n) for n in range(0,N)]
    dt = np.min(D)
    params['x'],params['y'],params['fx'] = params['sol'],params['sol'],[]
    params['dt'] = params.get('dt',dt)
    A = params['operator'](**params)
    kwargs['sol'] = CrankNicolson(A = A, u0=kwargs['sol'],**params)
    return kwargs

def lagrangian(**kwargs):
        return  -DiffOps.nabla_t_nabla(**kwargs)

def CHA_generator(kwargs):
    # kwargs= multimodal_mesh_generator(kwargs)
    kwargs= regular_mesh_generator(kwargs)
    kwargs['sol'] = kwargs['x']
    def scatterplot(**params):
        import seaborn as sns
        if 'datas' not in params:
            params['datas'] = [[params['x'],params['fx']],[params['char'],params['fx']],[params['sol'],params['fsol']]]
        D = params['x'].shape[1]
        def plot_helper(test,**params):
            x,fx = get_matrix(test[0]),get_matrix(test[1])
            if len(fx): test = pd.DataFrame(np.concatenate([x,fx],axis=1),columns = ['0','1','label'])
            else :
                test = pd.DataFrame(x[:,0:2],columns = ['0','1'])
                test['label'] = 0.
            if test.values.shape[1] == 2:test['1'] = test['label']
            sns.scatterplot(test, x=test['0'], y=test['1'], size='label',hue="label", legend = False)
        if D == 1:multi_plot(params['datas'],fun_plot=plot1D,**params)
        # else :multi_plot(params['datas'],fun_plot=plot_helper,**params)
        else :multi_plot(params['datas'],plot_trisurf,projection='3d',**params)
        
    kwargs['graphic'] = scatterplot
    # kwargs['graphic'] = standard_graphic
    
    return kwargs

def one_convexhull_step(**kwargs):
    # kwargs['graphic'](**kwargs,f_names=['initial condition','time evolution'])
    def get_V(**kwargs):
        shape_ = kwargs['x'].shape
        out = np.zeros(shape_)
        for d in range(shape_[1]): 
            out[:,d] = -kwargs['fx'].flatten()
        return out
    def characteristic_method(sol,dt,V=[],**kwargs):
        out = sol.copy()
        if len(V) == 0: V = get_V(**kwargs)
        for d in range(sol.shape[1]): 
            fx = V[:,d]*dt
            out[:,d] += fx.flatten()
        return out

    def CHA(char,P=[],**kwargs):
        from scipy.spatial import ConvexHull
        out = np.expand_dims(char.copy(),axis=2)
        Pout =P.copy()
        N = char.shape[0]
        kwargs['fz'] = out
        # plot1D([kwargs['x'],kwargs['fx']])
        # plot1D([kwargs['x'],sol])
        # standard_graphic(datas=[[kwargs['x'],kwargs['fx']]])
        # standard_graphic(datas=[[sol,kwargs['fx']]])
        # kwargs['graphic'](datas=[[out.squeeze(),P]],**kwargs)
        h = DiffOps.nabla_inv(**kwargs)
        # standard_graphic(datas=[[kwargs['x'],kwargs['fx']]])
        # standard_graphic(datas=[[char,kwargs['fx']]])
        # >plot1D([kwargs['x'],h])
        # kwargs['graphic'](datas=[[kwargs['x'],h]],**kwargs)
        # standard_graphic(datas=[[kwargs['x'],h]])
        points = np.concatenate([kwargs['x'],h],axis=1)
        # standard_graphic(datas=[[kwargs['x'],h]])
        kwargs['fx'] = h
        # out = op.nabla(**kwargs).squeeze()
        # kwargs['graphic'](datas=[[out,P]],**kwargs)
        hull = ConvexHull(points)
        x = kwargs['x']
        convex_hull = kwargs['x'][hull.vertices]
        inside_hull = list(set(range(0,N)).difference(hull.vertices))
        h_convex = h[hull.vertices]
        # standard_graphic(datas=[[convex_hull,h_convex]])
        # plot1D([convex_hull,h_convex])
        kwargs['x'],kwargs['y'],kwargs['z'],kwargs['fx'] = convex_hull,convex_hull,x,h_convex
        out = DiffOps.nabla(**kwargs)
        out = out.squeeze()
        # standard_graphic(datas=[[out,Pout]])
        # plot1D([x,out])
        # kwargs['graphic'](**kwargs,f_names=['initial condition','time evolution'])

        if len(P) == 0: return out
        kwargs['fx'] = P[hull.vertices]
        Pout = KerOp.projection(**kwargs)
        # standard_graphic(datas=[[out,Pout]])
        return out,Pout
        pass
    kwargs['sol'] = kwargs.get('sol',kwargs['x'])
    kwargs['sol'] = characteristic_method(**kwargs)
    char = kwargs['sol'].copy()
    kwargs['sol'],kwargs['fsol'] = CHA(char = char, P=kwargs['fx'],**kwargs)
    kwargs['char'] = char
    # kwargs['fsol'] = kwargs['fx']
    return kwargs


def output_lagrangian_datas(kwargs):
    from sklearn.preprocessing import StandardScaler
    normalized = StandardScaler().fit(kwargs['sol']).transform(kwargs['sol'])
    kwargs['datas'] = [(kwargs['x'],[]),(kwargs['sol'],[]),(normalized,[])]
    return kwargs

def my_fun(x):
    import numpy as np
    from math import pi
    coss = np.cos(2 * x * pi)
    if x.ndim == 1 : 
        coss = np.prod(coss, axis=0)
        ress = np.sum(x, axis=0)
    else : 
        coss = np.prod(coss, axis=1)
        ress = np.sum(x, axis=1)
    return ress+coss

def Denoising(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    D, Nx,Ny,Nz,eps,types=kwargs.get('D',2),kwargs.get('Nx',500),kwargs.get('Ny',500),kwargs.get('Nz',500),kwargs.get('eps',0.4),kwargs.get('types',["cart","cart","cart"])
    kwargs['set_codpy_kernel'] = kernel_setter("maternnorm", "scale_to_unitcube", 0, 1e-8)

    def your_fun(x,**kwargs):
        out = my_fun(x)
        out += eps* np.random.normal(size=out.shape)
        return out
    data_random_generator_ = data_random_generator(fun = your_fun,nabla_fun = None, types=types)
    x,y,z,fx,*_ =  data_random_generator_.get_raw_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    ker = Kernel(
        x=x, 
        y=y, 
        fx=fx, 
        set_kernel=kwargs['set_codpy_kernel']
    ).get_kernel()
    kwargs['kernel_ptr'] = ker    
    f_x = KerOp.gradient_denoiser(**{**kwargs,**{'x':x,'z': x,'fx':fx}}) 

    multi_plot([(x,fx),(x,f_x)],plot_trisurf, projection='3d',f_names=['Noisy signal','Denoised signal'], mp_nrows = 1, mp_figsize = (8,4),**kwargs)
    
    pass

def solve(**kwargs):
    kwargs['sol'] = kwargs.get('operator')(**kwargs)
    return kwargs

def Delta_inv(**kwargs):
    order,regularization=kwargs['set_codpy_kernel'].polynomial_order, kwargs['set_codpy_kernel'].regularization
    return DiffOps.nabla_t_nabla_inv(**kwargs,order=order, regularization=regularization)

def PDE2(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    kwargs['set_codpy_kernel'] = kernel_setter("tensornorm", "scale_to_unitcube", 1, 1e-8)
    out = pde_solver({**kwargs,'C':2,'centers':[[-3.,0.],[3.,0.]],'one_step':solve,'operator' : Delta_inv, 'data_generator' : multimodal_mesh_generator })
    toto = {'f_names':['f(x)','solution'], 'mp_nrows' : 1, 'mp_figsize' : (8,4)}
    out['graphic']({**out,**toto})
    
    pass

def PDE1(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    kwargs['set_codpy_kernel'] = kernel_setter("gaussian", "standardmean", 2, 1e-8)
    out = pde_solver({**kwargs,'epsilon':10,'one_step':solve, 'operator' : Delta_inv})
    out['graphic'](**out,f_names=['f(x)','solution'], mp_nrows = 1, mp_figsize = (8,4))
    pass

def PDE3(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    out =pde_solver({'epsilon':5.,'nbiter' : 10,'dt':.1,**get_pde_params()})
    out['graphic'](**out,f_names=['initial condition','time evolution'], mp_nrows = 1, mp_figsize = (8,4))
    
    pass

def PDE4(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    out =pde_solver({'C':2,'centers':[[-3.,0.],[3.,0.]],'nbiter' : 10,'dt':.1,'data_generator' : multimodal_mesh_generator,**get_pde_params()})
    out['graphic']({**out,**{'f_names':['initial condition','time evolution'], 'mp_nrows' : 1}})
    
    pass

def Lagrangian(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    out =pde_solver({**get_pde_params(),'dt':0.01,'N':200,'operator':lagrangian,'scale':1.,'C':1,'centers':[[0.,0.]], 'theta':.0,'nbiter':100,
        'data_generator' : multimodal_lagrangian_mesh_generator,'one_step':heat_lagrangian_step,'output_datas':output_lagrangian_datas })
    out['graphic']({**out,**{'f_names':['initial condition','time evolution','sharp sequences'], 'mp_nrows' : 1}})
    

def CHA1D(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    kwargs['set_codpy_kernel'] = kernel_setter("tensornorm", "scale_to_unitcube", 0, 1e-8)
    out = pde_solver({**kwargs,'D':1,'C':1,'epsilon':5.,'dt':.6,'nbiter' : 1,'one_step':one_convexhull_step,'data_generator' : CHA_generator})
    out['graphic'](**out,f_names=['initial cond.','conservative sol.','entropy sol.'], mp_nrows = 1)
    

def CHA2D(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    kwargs['set_codpy_kernel'] = kernel_setter("tensornorm", "scale_to_unitcube", 0, 1e-8)
    # kwargs['set_codpy_kernel'] = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2,1e-8,map_setters.set_unitcube_map)
    # kwargs['set_codpy_kernel'] = kernel_setters.kernel_helper(kernel_setters.set_matern_norm_kernel, 0,0 ,map_setters.set_standard_mean_map)
    out = pde_solver({**kwargs,'D':2,'C':1,'epsilon':5.,'dt':.7,'nbiter' : 1,'one_step':one_convexhull_step,'data_generator' : CHA_generator})
    out['graphic'](**out,f_names=['initial cond.','conservative sol.','entropy sol.'], mp_nrows = 1)
    


if __name__ == "__main__":
    PDE1()
    # Denoising()
    # PDE3()
    # PDE4()
    # Lagrangian()
    # test_sampler({**get_celebA_params(),**{'C':2,'Dx':1,'D':2,'Nx':500,'random_variable':sphere_sampling,'centers':[[0.,1.],[0.,.5]],'distance':'norm22'}})    
    # taylor_test(**get_param21(),taylor_order = 2)
    # CHA1D()
    # CHA2D()
    # Lagrangian()
    # PDE1()
    # PDE2()
    # Denoising()
    # PDE3()
    # PDE4()
    # Lagrangian()
    # CHA1D()
    # differentialMlBenchmarks(D=1,N=500)
    # taylor_test(**get_param21(),taylor_order = 2,numbers = {"D":2, "Nx":500,"Ny":500,"Nz":500 }, Z_min=-1.1,Z_max=1.1,)

    pass
