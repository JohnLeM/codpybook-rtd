import numpy as np
import os
import sys
parent_path = os.path.dirname(__file__)
parent_path = os.path.dirname(parent_path)
if parent_path not in sys.path: sys.path.append(parent_path)

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
data_path = os.path.join(CURRENT_DIR, "data")
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.ch9.path_generation import data_random_generator 
from utils.ch9.plot_utils import plot_trisurf 

from codpy.core import kernel_setter, DiffOps
from codpy.kernel import get_matrix, Kernel
from codpy.plot_utils import *
from codpy.pde import CrankNicolson 

def get_pde_params():

    kwargs = {
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
        'elev':20,
    }
    return kwargs

def standard_graphic(**kwargs):
    if 'datas' not in kwargs: 
        kwargs['datas'] = kwargs.get('datas',[[kwargs['x'],kwargs['fx']],[kwargs['x'],kwargs['sol']]])
    D = get_matrix(kwargs['datas'][0][0]).shape[1]
    if D==1:multi_plot(kwargs['datas'],plot1D,**kwargs)
    else: multi_plot(kwargs['datas'],plot_trisurf,projection='3d',**kwargs)

def one_eulerian_step(**kwargs):
    if 'generator' in kwargs :
        kwargs['sol'] = kwargs.get('sol',kwargs['fx'])
        kwargs['sol'] = kwargs['generator'] @ kwargs['sol']
        return kwargs
    dt,theta,iter_ = kwargs.get('dt',.01),kwargs.get('theta',1.),kwargs.get('iter',10)
    kwargs['operator'] = kwargs.get('operator')(**{**kwargs,'fx':[]})
    kwargs['generator'] = CrankNicolson(kwargs['operator'], u0=[], dt=dt,theta =theta)
    return one_eulerian_step(**kwargs)

def one_lagrangian_step(**kwargs):
    operator = kwargs.get('operator')(**{**kwargs,'fx':[]})
    operator2 = CrankNicolson(A = operator, u0=[], **kwargs)
    sol = operator2 @ kwargs['sol']
    kwargs['sol'] = sol
    return kwargs

def pde_solver(kwargs=None):
    if kwargs is None: kwargs=get_pde_params()
    time_grid = kwargs.get('time_grid',[])
    kwargs['operator'] = kwargs.get('operator', DiffOps.nabla_t_nabla)
    kwargs['Nx'],kwargs['Ny'],kwargs['Nz'],kwargs['D'] = kwargs.get('Nx',400),kwargs.get('Ny',400),kwargs.get('Nz',400),kwargs.get('D',2)
    kwargs['data_generator'] = kwargs.get('data_generator', regular_mesh_generator)
    kwargs = kwargs['data_generator'](kwargs)
    kwargs['one_step'] = kwargs.get('one_step',one_eulerian_step)
    kwargs['graphic'] = kwargs.get('graphic',standard_graphic)
    iter_ = kwargs.get('nbiter',1)
    time = 0.
    ker = Kernel(
        x=kwargs['x'], 
        y=kwargs['y'], 
        fx=kwargs.get('fx',None), 
        set_kernel=kwargs['set_codpy_kernel']
    ).get_kernel()
    kwargs['kernel_ptr'] = ker
    for n in range(0,iter_) : 
        kwargs = kwargs['one_step'](**kwargs)
    if 'output_datas' in kwargs: kwargs = kwargs['output_datas'](kwargs)
    return kwargs

class differential_operator:
    axis = -1
    grid = []
    def __init__(self,**kwargs):
        self.signature = kwargs.get("signature",[0.,1.])
        self.order = kwargs.get("order",len(self.signature))
        if self.order != len(self.signature):
            signature = np.zeros([self.order])
            signature[0:len(self.signature)] = self.signature
            self.signature = signature
        pass
    def __call__(self, y,axis=0,**kwargs):
        N = y.shape[axis]
        shape_out = np.array(y.shape)
        shape_out[axis] -= self.order #better to match the input data ?
        grid = np.array(kwargs.get("grid",range(0,N)))
        out = np.zeros(shape_out)
        out_view = np.swapaxes(out, axis, 0)
        for n in range(0,N-self.order):
            grid_expand = grid[n:n+self.order]-grid[n+self.order-1]
            operator = VanDerMonde(grid_expand,self.signature)
            values = []
            for s in range(n,n+self.order):
                val = y.take(indices = s,axis = axis)*operator[s-n]
                if len(values):values += val
                else: values = val
            out_view[n] = values
            pass
        return out
        pass


def regular_mesh_generator(kwargs): 
    epsilon =  kwargs.get('epsilon',1.)
    def my_fun(x): return np.asarray([np.exp(-epsilon*np.linalg.norm(x[n])) for n in range(0,x.shape[0])])
    kwargs['D'],kwargs['Nx'],kwargs['Ny'],kwargs['Nz'] = kwargs.get('D',2),kwargs.get('Nx',400),kwargs.get('Ny',400),kwargs.get('Nz',400)
    kwargs['x'],kwargs['y'],kwargs['z'],kwargs['fx'],kwargs['fy'],kwargs['fz'],kwargs['Nx'],kwargs['Ny'],kwargs['Nz'] = data_random_generator(fun = my_fun,types=["cart","cart","cart"]).get_raw_data(**kwargs)
    kwargs['z'],kwargs['fz'],kwargs['Nz'] = kwargs['x'],kwargs['fx'],kwargs['Nx']
    return kwargs

if __name__ == "__main__":
    pde_solver()
    pde_solver({'dt':.1,'data_generator' : multimodal_mesh_generator,**get_pde_params()})
