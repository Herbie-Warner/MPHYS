# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:52:23 2024

@author: herbi
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import MZ,stheta,ctheta,vtheta,g
from Core.Functions import xfunc,yfunc,zfunc
from LeadingOrder.LeadingOrderContribution import sig_P,sig_Z,sig_ZP
from VacuumPolarisation.VacBackEnd import F_Z,F_P,F_PZ

def Z_Self_Energy(s,t):
    LO_Z = sig_Z(s,t)
    LO_ZP = sig_ZP(s,t)
    
    deltaZ = g**2/(16*np.pi**2) * (F_Z(-s)-F_Z(-MZ**2))/(-s+MZ**2) 
    return deltaZ * (2*LO_Z + LO_ZP)




def P_Self_Energy(s,t):
    LO_P = sig_P(s,t)
    LO_ZP = sig_ZP(s,t)
    deltaP = g**2/(16*np.pi**2) * F_P(-s)/(-s)
    
    return deltaP * (2*LO_P + LO_ZP)

def ZP_Self_Energy(s,t):
    
    F_val = F_PZ(-s)
    pref = g**6/(64*(2*np.pi)**3)
    
    term1 = -stheta**3*vtheta*xfunc(s,t) /(ctheta*s*(-s+MZ**2))
    term2 = stheta*vtheta*(vtheta**2*xfunc(s,t)+2*zfunc(s,t)+yfunc(s,t))/(16*ctheta**2*(-s+MZ**2)**2)
     
    
    return pref*F_val * (term1+term2)

