# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:07:49 2024

@author: herbi
"""
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Lo.LOTotal import LO_Z_Total, LO_GAMMA_Total, LO_Z_GAMMA_Total
from Vacuum import DELTA_P, DELTA_0, A10P
from Utilities.Utilities import g,stheta,vtheta,ctheta,MZ

def photon_self_energy_Total(s):
    sigma_E = LO_GAMMA_Total(s)
    sigme_Cross = LO_Z_GAMMA_Total(s)
    return -DELTA_P(-s)*(2*sigma_E+sigme_Cross)

def Z_self_energy_Total(s):
    sigma_Z = LO_Z_Total(s)
    sigme_Cross = LO_Z_GAMMA_Total(s)
    return -DELTA_0(-s)*(2*sigma_Z+sigme_Cross)

def mix_self_energy_Total(s):
    pref = g**6/(64*(2*np.pi)**3)
    num = stheta**3*vtheta*(-4/3)/(ctheta*s*(-s+MZ**2))
    AGMZ = A10P(-s)   
    term1 = -pref*num*AGMZ   
    num2 = stheta*vtheta/(16*ctheta**3*s**2*(-s+MZ**2)**2)*AGMZ
    int2 = -s**2*(2*vtheta**4+4)/3   
    term2 = pref*num2*int2
    return term1+term2
    
