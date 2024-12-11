# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:14:36 2024

@author: herbi
"""

import numpy as np
import gmpy2

from Core.Precision import HPC

MW = 80.377 + 0j
ZWIDTH = 0
MZ = 91.1876 + ZWIDTH*1j


ME = 0.511 * 1e-3 + 0j
MMU = 105.6583755 * 1e-3 + 0j

MH = 125 + 0j


MTAU = 1.77693 + 0j

MUP = 2.16  * 1e-3 + 0j
MCHARM = 1.273 + 0j
MTOP = 172.57 + 0j
MDOWN = 4.7 * 1e-3 + 0J
MSTRANGE = 93.5 *1e-3 + 0j
MBOTTOM = 4.183 + 0j




SMALLPARAM = 1e-1

MGAMMA = 1e-3 + 0j
MNU = SMALLPARAM + 0j

DIMENSIONAL_SMALL_PARAM = 1e-6

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


ckm_matrix = np.array([
    [0.974, 0.226, 0.003],
    [0.226, 0.973, 0.041],
    [0.008, 0.041, 0.999]
])