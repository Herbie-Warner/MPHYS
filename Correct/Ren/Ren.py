# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:22:57 2024

@author: herbi
"""
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import MZ,stheta,g,MW,ctheta
from Core.Functions import xfunc,yfunc

from VacuumPolarisation.VacBackEnd import F_W,F_Z


def get_delta_stheta():
    
    pre = g**6/(16*np.pi**2)
    
    FW = F_W(-MW**2)
    FZ = F_Z(-MZ**2)
    
    del_W = pre*FW
    del_Z = pre*FZ

    term = del_Z/MZ**2 - del_W/MW**2
    print(term)
    
    return ctheta**2*term
    
    
    
    
get_delta_stheta()

def sigRen(s, t):
    numerator = (g**4 * stheta * (
        t**2 * (-1 + 6 * stheta - 12 * stheta**2 + 16 * stheta**3) +
        s * (
            -4 * MZ**2 * (-1 + stheta) * stheta * (
                yfunc(s, t) * (-1 + 2 * stheta) + 
                xfunc(s, t) * (-1 + 2 * stheta + 8 * stheta**2)
            ) + 
            s * (
                4 * yfunc(s, t) * stheta * (1 - 3 * stheta + 2 * stheta**2) + 
                xfunc(s, t) * (1 - 2 * stheta - 32 * stheta**3 + 16 * stheta**4)
            )
        )
    ))
    
    denominator = 512 * np.pi * (MZ**2 - s)**2 * s * (-1 + stheta)**3
    
    result = numerator / denominator
    return result
