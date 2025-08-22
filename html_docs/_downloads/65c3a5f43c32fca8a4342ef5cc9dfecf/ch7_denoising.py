"""
====================================
7.02 Denoising
====================================
We reproduce here the figure 7.3 of the book.
Utilitary functions can be found next to this file. Here, we only define codpy-related functions.
"""

#########################################################################
# Necessary Imports
# ------------------------
import os 
import sys 

import matplotlib.pyplot as plt 

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
data_path = os.path.join(CURRENT_DIR, "data")
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.ch7.ch7_utils import Denoising

#########################################################################
# Problem statement
# ------------------------
# We consider the denoiser procedure introduced in the book, which aims to solve: 
# $$\inf_{G \in \mathcal{H}_k} \|G-F\|_{L^2}^2 + \epsilon \|\nabla G\|_{L^2}^2.$$
# The noisy signal (left image) is given by $F_\eta(x) = F(x) + \eta$, where $\eta$ is a white noise, and $f$ is a cosine function. The regularized solution is plotted on the right.

Denoising() 
plt.show()
