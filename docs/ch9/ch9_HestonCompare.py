"""
====================================================================================================
9.10 Heston Process - Path comparison
====================================================================================================
We reproduce here the figure 9.12 of the book.
We show a comparison of generated paths with a generative method, diff log map and Heston process.
Utilitary functions can be found next to this file.
"""
    
#########################################################################
# Necessary Imports
# ------------------------

import os 
import sys 

from matplotlib import pyplot as plt

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()
data_path = os.path.join(CURRENT_DIR, "data")
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from utils.ch9.heston import *

fig_compare_trajectories_Heston()
plt.show()