# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:03:51 2024

@author: herbi
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.Utilities import g,MZ,xfunc,stheta,ctheta,vtheta



def LO_GAMMA(s,t):
    return g**4*stheta**4/(32*np.pi**2) * xfunc(s,t)/s


def LO_Z(s,t):
    pref = g**4/(128*np.pi)
    num = s*((vtheta**4+6*vtheta**2+1)*xfunc(s,t) - 8*vtheta**2*t**2/(s**2))
    return pref * num/(32*ctheta**4*((s-MZ**2)**2))

def LO_inter(s,t):
    pref = g**4/(128*np.pi)
    num = stheta**2*vtheta**2/ctheta**2 * s*xfunc(s,t) + stheta**2/ctheta**2 * (s+2*t)
    return pref*num/(s*(s-MZ**2))

def LO(s,t):
    return LO_GAMMA(s,t) + LO_Z(s,t)+LO_inter(s,t)