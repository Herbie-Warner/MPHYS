# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:17:19 2024

@author: herbi
"""


def xfunc(s,t):
    return 1 + 2*(t/s + t**2/s**2)

def yfunc(s,t):
    return 1 + 2*t/s

def zfunc(s,t):
    return (s+t)**2/(s**2)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gmpy2
from Core.Precision import behaved_quadratic_routes, HPC

def tau(sij,mi,mj):
    sij  = HPC(sij)
    mi = HPC(mi)
    mj = HPC(mj)
    
    alpha, _ = behaved_quadratic_routes(mj**2, sij, mi**2)

    return alpha/(alpha**2*mj**2-mi**2) * gmpy2.log(mj**2 * alpha**2/mi**2)
