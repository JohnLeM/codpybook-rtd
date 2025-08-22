"""
====================================
7.06 Automatic Differentiation
====================================
We reproduce here the figures 7.9, 7.10 & 7.11 of the book.
Utilitary functions can be found next to this file. Here, we only define codpy-related functions.
"""

#########################################################################
# Necessary Imports
# ------------------------
import os 
import sys 

import matplotlib.pyplot as plt 
import torch 

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
data_path = os.path.join(CURRENT_DIR, "data")
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.ch7.ch7_utils import testAAD, differentialMlBenchmarks, taylor_test, get_param21,get_param22

#########################################################################
# AAD 
# ------------------------
# The first figure displays the computation of first- and second-order derivatives of a function $f(X)=\frac{1}{6}X^3$ using AAD.
testAAD(lambda x: 1/6 * x**3, torch.randn((100,1), requires_grad=True))
plt.show()

#########################################################################
# 1 dimensional differential machines
# ------------------------------------------------
# Here, we illustrate a general multi-dimensional benchmark of two differential machines methods. 
# The first one uses the kernel gradient operator. 
# The second one uses a neural network defined with Pytorch together with AAD tools.
differentialMlBenchmarks(D=1,N=500)
plt.show()

#########################################################################
# 2 dimensional differential machines
# ------------------------------------------------
# We perform the same benchmark in 2 dimensions below: 
differentialMlBenchmarks(D=2,N=500)
plt.show()

#########################################################################
# Taylor expansions and differential machines
# ------------------------------------------------
# We approximate up to second order $\nabla f(x)$,$\nabla^2 f(x)$ with
# $$\nabla f_x = \nabla_Z \mathcal{P}_m\big(X,Y,Z=x,f(X)\big), \nabla^2 f_x = \nabla^2_Z \mathcal{P}_m\big(X,Y,Z=x,f(X)\big)$$
taylor_test(**get_param22(),taylor_order = 2)
plt.show()
