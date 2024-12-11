# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:56:36 2024

@author: herbi
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import MZ,MW,stheta,ctheta,MGAMMA, ME,MMU,g,vtheta,ztheta,MNU
from Core.Functions import xfunc,yfunc,zfunc,tau
from PVFunctions.BFunctions import B0,B1,dbl_deriv_B0,dbl_deriv_B1

def B_0P(ks,m1,m2):
    return -dbl_deriv_B0(ks, m1, m2)



