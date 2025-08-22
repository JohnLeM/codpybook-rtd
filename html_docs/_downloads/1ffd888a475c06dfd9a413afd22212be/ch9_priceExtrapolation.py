"""
====================================================================================================
9.12 Price extrapolation using KRR and Taylor
====================================================================================================
We reproduce here the figure 9.15 of the book.
We show price extrapolation using Kernel Ridge Regression (KRR) and Taylor series.
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

params = predict_prices()
params["graphic"](params)
plt.show()