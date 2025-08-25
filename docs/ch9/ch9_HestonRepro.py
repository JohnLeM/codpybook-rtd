"""
====================================================================================================
9.09 Heston Process - Reproducibility
====================================================================================================
We reproduce here the figure 9.10 and 9.11 of the book.
We show a reproducibility test for a Heston process as well as a comparison of the generated noise with a generative method.
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

fig_test_reproductibility_Heston()
plt.show()

table = fig_compare_distributions_Heston()
print(table)
plt.show()