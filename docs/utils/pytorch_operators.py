import numpy as np
import abc
from matplotlib import pyplot as plt
import torch # GPU + autodiff library
from torch.autograd import grad, Variable
import os, sys
import time 
import numpy as np
from pathlib import Path
parent_path = os.path.dirname(__file__)
parent_path = os.path.dirname(parent_path)
if parent_path not in sys.path: sys.path.append(parent_path)
import torch
import torch.nn as nn
from include import *
from codpy.core import get_matrix
import codpy.AAD as AAD
import codpy.pde as pde
from codpy import core 
from codpy.kernel import Kernel
# from codpy_tools import *
# from stat_tools import *
# from data_generators import * 
# from predictors import * 
# from scikit_tools import * 
# from mnist_codpy import * 
# from time_series import*

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

import utils.codpy_predictors as cp_preds 
import utils.ch9.path_generation as path_gen
import utils.ch9.plot_utils as plot_utils
from utils.ch9.market_data import ts_scenario_generator, data_accumulator

class PytorchRegressor(cp_preds.data_predictor):
    def get_params(**kwargs): return kwargs.get('PytorchRegressor',{})
    def get_model(self, **kwargs):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        FX = torch.tensor(get_matrix(kwargs.get('FX',self.fx)),dtype=torch.float)
        self.epochs = kwargs.get('epochs',500)
        batch_size = kwargs.get('batch_size',128)
        layers = kwargs.get('layers', [128,64])
        out_layer =  kwargs.get('out_layer', FX.shape[1])
        self.loss = kwargs.get('loss',nn.CrossEntropyLoss())
        #dataset=TensorDataset(torch.tensor(get_matrix(self.x),dtype=torch.float),torch.tensor(get_matrix(self.fx),dtype=torch.float))
        #self.dataloader=DataLoader(dataset,batch_size = batch_size, shuffle=False)
        return PytorchNet(input_dim = self.D ,hidden_dim = layers, out_dim = out_layer, **kwargs)    
    def predictor(self,**kwargs):
        params = PytorchRegressor.get_params(**kwargs)
        self.model = self.fit(**params)
        self.f_z = self.predict(self.z)
    def fit(self, **kwargs):  
        import torch.jit as jit
        #model = torch.jit.script(self.get_model(**kwargs))
        optimizer = kwargs.get('optimizer', torch.optim.Adam)
        criterion = kwargs.get('loss', nn.MSELoss())
        model = self.get_model(**kwargs)
        optimizer = optimizer(model.parameters())
        log_each = 500
        x,fx = torch.FloatTensor(get_matrix(self.x)),torch.FloatTensor(get_matrix(self.fx))
        x.requires_grad_(True)
        for step in range(self.epochs):
            #for batch_x,batch_fx in dataloader:
            y_hat = model(x) 
            loss = criterion(fx, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % log_each == 0:
                print(f'{step}/{self.epochs} loss {loss.item():.5f}')
        return model
    def predict(self,x):
        self.model.eval()
        x = torch.FloatTensor(get_matrix(x))
        with torch.no_grad():
            p = self.model(x)
        return p.numpy().ravel()   
    def gradient(self,**kwargs):
        self.predictor(**kwargs)
        z = torch.FloatTensor(get_matrix(self.z)).requires_grad_(True)
        y = self.model(z)
        grad_outputs = torch.ones_like(y)
        try:
            self.grad = torch.autograd.grad(y, [z], grad_outputs = grad_outputs, create_graph=True)[0]
        except:
            self.model.eval()
            self.grad = torch.autograd.grad(y, [z], grad_outputs = grad_outputs, create_graph=True)[0]
        self.grad = self.grad.detach().numpy()
        return self.grad

    def id(self):
        return "PytorchRegressor"


class PytorchClassifier(cp_preds.codpyprClassifier, cp_preds.add_confusion_matrix):
    def get_params(**kwargs): return kwargs.get('PytorchClassifier',{})
    def get_model(self, **kwargs):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        import os
        params = PytorchClassifier.get_params(**kwargs)
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        self.epochs = params.get('epochs',5)
        batch_size = params.get('batch_size',128)
        layers = params.get('layers', [128,64])
        out_layer =  params.get('out_layer', [10])
        self.loss = params.get('loss',nn.CrossEntropyLoss())
        activation = params.get('activation',nn.ReLU())
        data_type = params.get("datatype", "float")
        dataset=TensorDataset(torch.tensor(self.x,dtype=torch.float),torch.tensor(self.fx,dtype=torch.long))
        self.dataloader=DataLoader(dataset,batch_size = batch_size, shuffle=False)
        return PytorchNet(input_dim = self.D ,hidden_dim = layers, out_dim = out_layer, activation = activation)

    def predictor(self,**kwargs):
        import torch
        optimizer = kwargs.get('optimizer', torch.optim.Adam)
        model = self.get_model(**kwargs)
        optimizer = optimizer(PytorchNet.parameters(model), lr=0.001)
        self.fit(**kwargs)
        z = torch.FloatTensor(self.z)
        self.f_z = model.predict(z, **kwargs)

    def fit(self,epochs, **kwargs):  
        import torch.jit as jit
        #model = torch.jit.script(self.get_model(**kwargs))
        optimizer = kwargs.get('optimizer', torch.optim.Adam)
        criterion = kwargs.get('loss', nn.MSELoss())
        model = self.get_model(**kwargs)
        optimizer = optimizer(model.parameters(), lr=0.001)
        log_each = 500
        x,fx = torch.FloatTensor(self.x),torch.FloatTensor(self.fx)
        x.requires_grad_(True)
        for step in range(self.epochs):
            for x,fx in self.dataloader:
                y_hat = model(x) 
                loss = criterion(fx, y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if step % log_each == 0:
                print(f'{step}/{self.epochs} loss {loss.item():.5f}')
        return model

    def predict(self, z, **kwargs): 
        get_proba = kwargs.get('get_proba',False)
        f_z = []
        with torch.no_grad():
            if get_proba:
                for i,data in enumerate(z):
                    y_pred = self.model(data).clone().detach().numpy()
                    f_z.append(y_pred)
                f_z = np.array(f_z)
            else:
                for i,data in enumerate(z):
                    y_pred = self.model(data)
                    f_z.append(y_pred.argmax().item())
                f_z = np.asarray(f_z).reshape((len(f_z), 1))
        return f_z

    def id(self,name = "Pytorch"):
        return "Pytorch"

class PytorchNet(nn.Module):
    def __init__(self,input_dim, hidden_dim, out_dim, **kwargs):
        super(PytorchNet, self).__init__()
        activations = kwargs.get('activations',np.repeat(nn.ReLU(),len(hidden_dim)))
        dropout = 0.0
        current_dim = input_dim
        layers = []
        for hdim,activation in zip(hidden_dim,activations):
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(activation)
            layers.append(nn.Dropout(p = dropout))
            current_dim = hdim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)
        print(self.net)

    def forward(self, input:torch.FloatTensor):
        return self.net(input)


if __name__ == "__main__":

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

    def test_pytorch(**kwargs):
        fun = kwargs.get("fun",my_fun)
        pytorch_regressor = kwargs.get("Pytorch_regressor",PytorchRegressor)(**kwargs)
        scenarios_list = [ (1, 100*i, 50,100*i ) for i in np.arange(1,5,1)]
        scenario_generator_ = scenario_generator()
        scenario_generator_.run_scenarios(scenarios_list,
        path_gen.data_random_generator(fun = fun),
        pytorch_regressor,
        data_accumulator(), **kwargs)
        results = scenario_generator_.accumulator.get_output_datas().dropna(axis=1).T
        print(results)
        list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
        multi_plot(list_results,plot1D,mp_max_items = len(scenarios_list))
        pass
   
    def get_torch_param():
        return {'PytorchRegressor':
            {'epochs': 500,
            'layers': [128,128,128,128],
            'loss': nn.MSELoss(),
            'activations': [nn.SiLU(),nn.SiLU()],
            'optimizer': torch.optim.Adam,
            }}
    test_pytorch(**get_torch_param())
    


    def quad(x):
        return x ** 2
    
    def get_torch_param():
        return {'PytorchRegressor':
            {'epochs': 500,
            'layers': [128,128,128,128],
            'loss': nn.MSELoss(),
            # 'activations': [nn.SiLU(),nn.SiLU()],
            'optimizer': torch.optim.Adam
            }, 
            'fun': quad}

    test_pytorch(**get_torch_param())
    pass

class pytorch:
    def nabla(x,y,z,fx,**kwargs):
        Pyt = PytorchRegressor()
        Pyt.x,Pyt.y,Pyt.z,Pyt.fx,Pyt.D = x,y,z,fx,x.shape[-1]
        return Pyt.gradient(**kwargs)
    def projection(x,y,z,fx,**kwargs):
        Pyt = PytorchRegressor()
        Pyt.x,Pyt.y,Pyt.z,Pyt.fx,Pyt.D = x,y,z,fx,x.shape[-1]
        Pyt.predictor(**kwargs)
        return Pyt.f_z
    def divergence(x,y,z,fx,**kwargs):
        Pyt = PytorchRegressor()
        Pyt.x,Pyt.y,Pyt.z,Pyt.fx,Pyt.D = x,y,z,fx,x.shape[-1]
        return Pyt.divergence(**kwargs)

    def laplace(x,y,z,fx,**kwargs):
        return pytorch.divergence(x,y,z,pytorch.nabla(x,y,x,fx,**kwargs),**kwargs)

    # def hessian(x,y,z,fx,**kwargs):
    #     Pyt = PytorchRegressor()
    #     x = torch.FloatTensor(x).requires_grad_(True)
    #     y = torch.FloatTensor(y).requires_grad_(True)
    #     z = torch.FloatTensor(z).requires_grad_(True)
    #     Pyt.x,Pyt.y,Pyt.z,Pyt.fx,Pyt.D = x,y,z,fx,x.shape[-1]
    #     fx = Pyt.fit(**kwargs)
    #     out = AAD.AAD.hessian(fx=fx,x=x)
    #     return out
    def hessian(x,z,fx,**kwargs):
        Pyt = PytorchRegressor()
        x = torch.FloatTensor(x).requires_grad_(True)
        y = torch.FloatTensor(x).requires_grad_(True)
        z = torch.FloatTensor(z).requires_grad_(True)
        Pyt.x,Pyt.y,Pyt.z,Pyt.fx,Pyt.D = x,x,z,fx,x.shape[-1]
        fx = Pyt.fit(**kwargs)
        out = AAD.AAD.hessian(fx=fx,x=x)
        return out
    
def my_fun(x, **kwargs):
    import numpy as np
    from math import pi
    sinss = np.cos(2 * x * pi)
    if x.ndim == 1 : 
        sinss = np.prod(sinss, axis=0)
        ress = np.sum(x, axis=0)
    else : 
        sinss = np.prod(sinss, axis=1)
        ress = np.sum(x, axis=1)
    return ress+sinss

def my_fun_torch(x, **kwargs):
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

def quad(x):
    if not isinstance(x,torch.Tensor) : 
        x = torch.tensor(x, requires_grad=True)
    return x ** 2


def cubic(x):
    if not isinstance(x,torch.Tensor) : 
        x = torch.tensor(x, requires_grad=True)
    return x ** 3

def benchmark_gradient(x,z,fx,**kwargs):
    fun = kwargs.get('fun')
    nabla_fz = AAD.gradient(fun,z)
    # fx = fun(x)
    # x,y,z,fx,fy,fz,Nx,Ny,Nz = path_gen.data_random_generator(**kwargs).get_raw_data(D=1,Nx=500,Ny=500,Nz=500)    
    pytorch_grad1 = elapsed_time(lambda : pytorch.nabla(x=x,y=x,z=z,fx=fx,**kwargs),msg="torch nabla:")
    pytorch_grad2 = elapsed_time(lambda : pytorch.nabla(x=x,y=x,z=z,fx=fx,**kwargs),msg="torch nabla:")
    pytorch_f_z =  elapsed_time(lambda : pytorch.projection(x=x,y=x,z=z,fx=fx,**kwargs),msg="torch projection:")

    codpy_grad = elapsed_time(lambda : op.nabla(x=x,y=x,z=z,fx=fx,**kwargs),msg="codpy nabla:")
    codpy_f_z = elapsed_time(lambda : op.projection(x=x,y=x,z=z,fx=fx,**kwargs),msg="codpy projection:")

    rmse_delta_codpy = get_relative_mean_squared_error(codpy_grad,nabla_fz)
    rmse_delta_pytorch = get_relative_mean_squared_error(pytorch_grad1,nabla_fz)

    list_results = [(z,fz),(z,pytorch_f_z),(z,codpy_f_z)]
    f_names=["exact f","Pytorch f","Codpy f"]
    if D==1: multi_plot(list_results,plot1D,mp_max_items = len(list_results),f_names=f_names)
    else: multi_plot(list_results,plot_trisurf,projection='3d', mp_ncols = len(list_results), mp_max_items = len(list_results),f_names=["exact","pytorch","codpy"],elev = 25,azim=-80,linewidth = 0.2,antialiased = True,figsize=(9,9))
    #else: multi_plot(list_results,plot_trisurf,projection='3d', mp_ncols = len(list_results), mp_max_items = len(list_results),f_names=["exact","pytorch","codpy"],antialiased = True,figsize=(9,9))

    f_names=["exact grad","codpy grad","pytorch grad-1","pytorch grad-2"]
    list_results = [(z,nabla_fz[:,0,:]),(z,codpy_grad[:,0,:]),(z,pytorch_grad1[:,0]),(z,pytorch_grad2[:,0])]
    if D==1: multi_plot(list_results,plot1D,mp_max_items = len(list_results),f_names=f_names)
    #else: multi_plot(list_results,plot_trisurf,projection='3d', mp_ncols = len(list_results), mp_max_items = len(list_results),f_names=f_names,elev = 25,azim=-80,linewidth = 0.2,antialiased = True,figsize=(9,9))
    else: multi_plot(list_results,plot_trisurf,projection='3d', mp_ncols = len(list_results), mp_max_items = len(list_results),f_names=f_names,antialiased = True,figsize=(9,9))

class AAD_data_random_generator(path_gen.data_random_generator):
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        function_torch = kwargs.get('function_torch',my_fun_torch)
        if (D*Nx*Ny*Nz):
            if self.fun != None: 
                if self.nabla_fun != None: x,y,z,fx,fz,nabla_fx,nabla_fz,Nx,Ny,Nz = self.get_raw_data(Nx,Ny,Nz,D,**kwargs)
                else: x,y,z,fx,fy,fz,Nx, Ny, Nz = self.get_raw_data(Nx=Nx,Ny=Ny,Nz=Nz,D=D,**kwargs)
        torch_x = torch.tensor(x, requires_grad=True, dtype=torch.float64)
        torch_z = torch.tensor(z, requires_grad=True, dtype=torch.float64)    
        fz = AAD.AAD.taylor_expansion(x=torch_x,z=torch_z, fx= function_torch, order = True,distance = 'norm2',**kwargs)
        return  x, fx, y, fy, z, fz

class pytorch_taylor(PytorchRegressor):
    def get_params(**kwargs) :
        return kwargs.get('PytorchRegressor',{})
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            self.f_z = pde.taylor_expansion(self.x,self.x,self.z,self.fx, order = True,distance = 'norm2', nabla = pytorch.nabla, hessian = pytorch.hessian, **kwargs)
            self.save_cd_data(**kwargs)
            pass
    def id(self,name = ""):
        return "AAD Pytorch Taylor"

class codpy_taylor(cp_preds.codpyprRegressor):
    def get_params(**kwargs) :
        return kwargs.get('codpy_param',{})
    def predictor(self,**kwargs):
        params = codpy_taylor.get_params(**kwargs)
        if (self.D*self.Nx*self.Ny*self.Nz ):
            kernel_ptr = Kernel(
                x=self.x, fx=self.fx, set_kernel=params.get("set_kernel",core.kernel_setter("gaussianper", None))
            ).get_kernel()
            self.f_z = pde.taylor_expansion(self.x,self.x,self.z,self.fx, distance = 'norm2',kernel_ptr=kernel_ptr, **params, **kwargs)
            self.save_cd_data(**kwargs)
            pass
    def id(self,name = ""):
        return "Codpy Taylor"


def taylor_test(**kwargs):
    numbers = kwargs.get("numbers",(1, 500, 500, 500 ))
    scenarios = ts_scenario_generator()
    function = kwargs.get('function',my_fun)
    random_generator = kwargs.get('random_generator',AAD_data_random_generator)
    generator_ = [random_generator(fun = function, types=["sto","sto","sto"], **kwargs),random_generator(fun = function, types=["sto","sto","sto"], **kwargs)]
    predictor_ = [pytorch_taylor(**kwargs,set_data = False), codpy_taylor(**kwargs,set_data = False)]
    dic_kwargs = [{ "numbers":numbers, "generator":generator,"predictor":predictor, **predictor.get_new_params(**kwargs)} for generator,predictor in zip(generator_,predictor_)]
    scenarios.run_scenarios(dic_kwargs, data_accumulator())
    results = scenarios.accumulator.get_output_datas().dropna(axis=1).T
    print(results)
    return scenarios

def taylor_plots(scenarios, **kwargs):
    numbers = kwargs.get("numbers",(1, 500, 500, 500 ))
    function_torch = kwargs.get('function_torch',my_fun_torch)
    x,z,fx,f_z, fz = scenarios.accumulator.get_xs()[0], scenarios.accumulator.get_zs()[0], scenarios.accumulator.get_fxs()[0], scenarios.accumulator.get_f_zs(), scenarios.accumulator.get_fzs()[0]
    f_zaad = f_z[0]
    f_zk = f_z[1]
    order = kwargs.get('taylor_order', [])
    D = numbers[0]
    list_results,f_names = [ (z,fz),(z,f_zk),(z,f_zaad)], ["z, fz (AAD ord.)" +str(order),"Codpy ord." + str(order),"Pytorch ord."  + str(order)]
    if D==1: multi_plot(list_results,plot1D,mp_max_items = len(list_results),mp_ncols = 3,f_names=f_names)
    else: multi_plot(list_results,plot_utils.plot_trisurf,projection='3d', mp_ncols = 4, mp_max_items = len(list_results),f_names=f_names,antialiased = True,figsize=(9,9))

def get_torch_param():
    return {
        'PytorchRegressor':
        {'epochs': 500,
        'layers': [128,128,128,128],
        'loss': nn.MSELoss(),
        'activations': [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
        'optimizer': torch.optim.Adam,
        },
        'codpy_param':
        {
        #'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map),
        'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, 3 ,1e-8 ,None),
        #'set_codpy_kernel':  kernel_setters.kernel_helper(kernel_setters.set_multiquadricnorm_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map),
        'rescale' : True,
        }}

def test_AAD():
    n = 100
    def f(x):
        return 1/6 * x**3
    torch.manual_seed(0)
    a = torch.randn((n,1), requires_grad=True)
    y = f(a)
    gradf=AAD.gradient(f,a)
    grad1 = gradf[:,0]
    gradf = AAD.hessian(f,a)
    grad2 = gradf[:,0]
    aa = a.detach().numpy()[:,0]
    yy = y.detach().numpy()[:,0]
    title_list = ["cubic", "1st derivative", "2nd derivative"]
    multi_plot([(aa,yy),(aa,grad1), (aa,grad2)], plot1D,mp_ncols = 3, f_names=title_list, mp_max_items = 4)

if __name__ == "__main__":

    # set_kernel = kernel_setters.kernel_helper(kernel_setters.set_multiquadricnorm_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map)
    #set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map)
    # set_kernel = kernel_setters.kernel_helper(kernel_setters.set_tensornorm_kernel, 2,0 ,map_setters.set_unitcube_map)    
    set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None)

    # test_AAD()
    x,y,z,fx,fy,fz,Nx,Ny,Nz = path_gen.data_random_generator(fun = my_fun_torch, types=["cart","cart","cart"]).get_raw_data(D=1,Nx=500,Ny=500,Nz=500)    
    # x,y,z,Nx,Ny,Nz = data_random_generator().get_raw_data(D=1,Nx=500,Ny=500,Nz=500,Z_min=-1.1,Z_max=1.1)
    benchmark_gradient(x = x, z=z, fx=fx,set_codpy_kernel = set_kernel, rescale = True,**get_torch_param(),fun = my_fun_torch, types=["cart","cart","cart"])
    # benchmark_gradient(D=2,Nx=500,Ny=500,Nz=500,set_codpy_kernel = set_kernel, rescale = True,Z_min=-1.1,Z_max=1.1,**get_torch_param(),fun = my_fun, types=["cart","cart","cart"])
    # benchmark_gradient(D=100,Nx=500,Ny=500,Nz=500,set_codpy_kernel = set_kernel, rescale = True,Z_min=-1.1,Z_max=1.1,**get_torch_param(),fun = my_fun, types=["sto","sto","sto"])

    # def get_torch_param():
    #     return {'PytorchRegressor':
    #         {'epochs': 500,
    #         'layers': [128,128,128,128],
    #         'loss': nn.MSELoss(),
    #         'activations': [nn.ReLU()],
    #         'optimizer': torch.optim.Adam
    #         },
    #         'codpy_param':
    #         {
    #             'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussianper_kernel,2,1e-8,None),
    #             'rescale' : True,
    #         }}
            
    # params = get_torch_param()

    # # global_param = {'numbers':(1, 500, 500, 500 ), 'random_generator':data_random_generator}
    # # scenarios = taylor_test(**global_param,**params,taylor_order = 1)
    # # taylor_plots(scenarios, **global_param,**params,taylor_order = 1)
    # # scenarios = taylor_test(**global_param,**params,taylor_order = 2)
    # # taylor_plots(scenarios, **global_param,**params,taylor_order = 2)

    # global_param = {'numbers':(1, 500, 500, 500 )}
    # scenarios = taylor_test(**global_param,**params,taylor_order = 1)
    # taylor_plots(scenarios, **global_param,**params,taylor_order = 1)
    # scenarios = taylor_test(**global_param,**params,taylor_order = 2)
    # taylor_plots(scenarios, **global_param,**params,taylor_order = 2)

    def get_torch_param():
        return {
            'PytorchRegressor':
            {'epochs': 500,
            'layers': [128,128,128,128],
            'loss': nn.MSELoss(),
            'activations': [nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],
            'optimizer': torch.optim.Adam,
            },
            'codpy_param':
            {
            #'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map),
            'set_codpy_kernel': kernel_setters.kernel_helper(kernel_setters.set_linear_regressor_kernel, 3 ,1e-8 ,None),
            #'set_codpy_kernel':  kernel_setters.kernel_helper(kernel_setters.set_multiquadricnorm_kernel, 2 ,1e-8 ,map_setters.set_unitcube_mean_map),
            'rescale' : True,
            }}
    params = get_torch_param()

    global_param = {'function':quad, 'function_torch': quad, 'numbers':(1, 500, 500, 500 )}
    scenarios = taylor_test(**global_param,**params,Z_min=-3,Z_max=3,taylor_order = 1)
    taylor_plots(scenarios, **global_param,**params, taylor_order = 1)
    scenarios = taylor_test(**global_param,**params,Z_min=-3,Z_max=3, taylor_order = 2)
    taylor_plots(scenarios, **global_param,**params,taylor_order = 2)

    global_param = {'function':np.exp, 'function_torch': torch.exp, 'numbers':(1, 500, 500, 500 )}
    scenarios = taylor_test(**global_param,**params,Z_min=-3,Z_max=3,taylor_order = 1)
    taylor_plots(scenarios, **global_param,**params, taylor_order = 1)
    scenarios = taylor_test(**global_param,**params,Z_min=-3,Z_max=3, taylor_order = 2)
    taylor_plots(scenarios, **global_param,**params,taylor_order = 2)
    pass



