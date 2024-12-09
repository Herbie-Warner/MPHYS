# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:14:36 2024

@author: herbi
"""

import numpy as np
import gmpy2

from Core.Precision import HPC

MW = 80.377 + 0j
MZ = 91.1876 + 0j

ME = 0.511 * 1e-3 + 0j
MMU = 105.6583755 * 1e-3 + 0j

MH = 125 + 0j



SMALLPARAM = 1e-1

MGAMMA = SMALLPARAM + 0j
MNU = SMALLPARAM + 0j

DIMENSIONAL_SMALL_PARAM = 1e-2

euler_const = 0.5772156649015328606065120900824

#DELTA = HPC(-2/(4+DIMENSIONAL_SMALL_PARAM-4)) + HPC(euler_const) - HPC(gmpy2.log(np.pi))
DELTA = 0

sin_sq = (1-(MW/MZ)**2).real

stheta = np.sqrt(sin_sq)
ctheta = np.sqrt(1-sin_sq)
vtheta = 4*stheta**2-1
ztheta = vtheta**4 + 6*vtheta**2 + 1


g = 0.30282212088/stheta


CONVERSION_GEV_TO_mBarns = 0.3894*1e9