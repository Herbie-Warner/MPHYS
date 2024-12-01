# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:08:26 2024

@author: herbi
"""


import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.Utilities import g,MZ,stheta,ctheta,vtheta

alpha_W = g**2/(4*np.pi)

def LO_GAMMA_Total(s):
    return 4*np.pi*alpha_W**2/(s*3) * stheta**4

def LO_Z_Total(s):
    pref = np.pi * alpha_W**2/(192*ctheta**4)
    num = s * (1+vtheta**2)**2/(s-MZ**2)**2
    return pref*num

def LO_Z_GAMMA_Total(s):
    pref = np.pi * alpha_W**2*stheta**2*vtheta**2/(6*ctheta**2*(s-MZ**2))   
    return pref

def LO_Total(s):
    return LO_GAMMA_Total(s)+ LO_Z_Total(s)+ LO_Z_GAMMA_Total(s)