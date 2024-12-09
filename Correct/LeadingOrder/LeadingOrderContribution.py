# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:40:01 2024

@author: herbi
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Core.Quantities import MZ,stheta,ctheta,vtheta,g,ztheta
from Core.Functions import xfunc,yfunc

def sig_Z(s,t):
    pref = g**4/(32*np.pi)
    num = ztheta*s**2*xfunc(s,t)-8*vtheta**2*t**2
    return pref*num/(256*ctheta**4*s*(-s+MZ**2)**2)

def sig_P(s,t):
    pref = g**4/(32*np.pi)
    return pref * 2*stheta**4*xfunc(s,t)/s

def sig_ZP(s,t):
    pref = -g**4/(32*np.pi)
    num = stheta**2*(vtheta**2*xfunc(s,t)+yfunc(s,t))
    return pref*num/(4*ctheta**2*(-s+MZ**2))

def LeadingOrderTotal(s,t):
    return sig_Z(s,t)+sig_P(s,t)+sig_ZP(s,t)



def their_sig_z(s,t):
    pref = -g**4/(128*np.pi*s*(-s+MZ**2))
    
    AI = vtheta**2*stheta**2/ctheta**2 *s *xfunc(s,t) + stheta**2/(ctheta**2) * (s+2*t)
    return pref*AI

