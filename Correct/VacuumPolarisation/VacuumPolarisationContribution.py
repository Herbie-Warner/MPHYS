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
from LeadingOrder.LeadingOrderContribution import sig_P,sig_Z,sig_ZP
from VacuumPolarisation.VacBackEnd import F_Z

def Z_Self_Energy(s,t):
    LO_Z = sig_Z(s,t)
    LO_ZP = sig_ZP(s,t)
    
    deltaZ = g**2/(16*np.pi**2) * (F_Z(-s)-F_Z(-MZ**2))/(-s+MZ**2)
    return deltaZ * (2*LO_Z + LO_ZP)

def P_Self_Energy(s,t):
    return 0

def ZP_Self_Energy(s,t):
    return 0

