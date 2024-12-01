# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:35:24 2024

@author: herbi
"""


import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PVFunctions.BFunctions import B0,B1, B0P,B1P
from Utilities.Utilities import ME,MGAMMA,MMU,ctheta,stheta,vtheta,MZ,MNU,MW,g,xfunc
from Lo.LOContribution import LO


prefactor = g**6/((2*np.pi)**3 * 256)

def WV_f(mf):
    term1 = 2*stheta**2*(B1(-mf**2,mf,MGAMMA) + 2*mf**2*B1P(-mf**2,mf,MGAMMA)+1/2)
    term2 = 8*stheta**2*mf**2*B0P(-mf**2,mf,MGAMMA)
    term3 = (vtheta**2+1)/(8*ctheta**2) * (B1(-mf**2,mf,MZ)+1/2)
    term4 = 1/2 * (B1(-mf**2,MNU,MW)+1/2)
    
    IR_term = 2*stheta**2*np.log(mf**2/MGAMMA**2)
    return term1+term2+term3+term4 -IR_term

def WV_A(mf):
    term1 = vtheta/(4*ctheta**2) * (B1(-mf**2,mf,MZ)+1/2)
    term2 = -1/2 * (B1(-mf**2,MNU,MW)+1/2)
    return term1+term2



ZV = 1/2 * (WV_f(ME)+WV_f(MMU))
ZA = 1/2 * (WV_A(ME)+WV_A(MMU))

def dsigma_WR_E(s,t):
    V_f = 32*stheta**4*xfunc(s,t)/s + stheta**2*(xfunc(s,t)*vtheta**2+(s+2*t)/s)/(ctheta**2*(s-MZ**2))
    
    V_A = 4*stheta**2*vtheta*(s+t)**2/(ctheta**2*s**2*(s-MZ**2))
    return prefactor*(ZV * V_f + ZA*V_A)

def dsigma_WR_W(s,t):
    V_f1 = 16*ctheta**2*stheta**2*(s-MZ**2)*(vtheta**2*s**2*xfunc(s,t) + s*(s+2*t))
    V_f2 = s*(s**2*xfunc(s,t)*(1+vtheta**4) + 2*vtheta**2*(3*s**2+6*s*t+2*t**2))
    
    V_f = (V_f1+V_f2)/(8*ctheta**4*s**2*(-s+MZ**2)**2)
    
    V_A = vtheta*(s+t)**2 * (8*ctheta**2*stheta**2*(s-MZ**2)+s*(vtheta**2+1))/(2*ctheta**4*s**2*(s-MZ**2)**2)
    
    return prefactor*(ZV*V_f + ZA*V_A)

def sigma_WR_tot(s,t):
    return dsigma_WR_W(s,t)+dsigma_WR_E(s, t)