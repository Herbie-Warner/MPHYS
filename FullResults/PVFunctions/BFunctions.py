# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:42:10 2024

@author: herbi
"""

import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utilities.Utilities import DELTA
from PVFunctions.AFunctions import A

correctiveB0 = -0.5675138858494044
correctiveB1 = 0.033756945424700646


smalll = 1e-25

def F1(x):
    x += smalll
    return -1 - x*np.log(1-x)+x*np.log(-x)

def F2(x):
    x += smalll
    return -1/2 - x -x**2*np.log(-1+x)+x**2*np.log(x)

def F3(x):
    inside = 6*x**3 * np.log(-x) - 3*(2*x**3+1)*np.log(1-x) - 6*x**2-3*x-2
    return 1/9 * inside


def find_xs(psquared,m1,m2):
    a = -psquared
    b = psquared+m2**2-m1**2
    c = m1**2
    det = np.sqrt(b**2-4*a*c)
    #print(a,b,c,det)
    return (-b+det)/(2*a), (-b-det)/(2*a)

def B0(psquared,m1,m2):
    term1 = np.log(-psquared)
    x1,x2 = find_xs(psquared,m1,m2)
    term2 = np.log(1-x1)+F1(x1)
    term3 = np.log(1-x2) +F1(x2)
    return DELTA - (term1+term2+term3) - correctiveB0

def B1(psquared,m1,m2):
    term1 = np.log(-psquared)
    x1,x2 = find_xs(psquared,m1,m2)
    term2 = np.log(1-x1)+F2(x1)
    term3 = np.log(1-x2) +F2(x2)
    return -0.5*DELTA + 0.5*(term1+term2+term3) - correctiveB1

def B22(psquared,m1,m2):
    term1 = 1/2*A(m2) - m1**2*B0(psquared,m1,m2)
    term2 = -1/2 * (m1**2-m2**2-psquared)*B1(psquared,m1,m2)
    term3 = -1/2*(m1**2+m2**2+1/3 * psquared)
    return 1/3* (term1+term2+term3)

def B21(psquared,m1,m2):
    term1 = A(m2)-m1**2*B0(psquared,m1,m2)
    term2 = -4*B22(psquared,m1,m2)
    term3 = -1/2 * (m1**2+m2**2+1/3*psquared)
    return 1/psquared * (term1+term2+term3)

def B21_alt(psquared,m1,m2):
    
    term1 = -B22(psquared,m1,m2)
    term2 = 1/2 * (A(m2)+(m1**2-m2**2-psquared)*B1(psquared,m1,m2))
   
    return 1/psquared * (term1+term2)

def B0P(psquared,m1,m2):
    x1,x2 = find_xs(psquared,m1,m2)
    pref = -1/(psquared*(x1-x2))
    fac = (1-x1)*F1(x1) - (1-x2)*F1(x2)
    return -pref*fac

def B1P(psquared,m1,m2):
    x1,x2 = find_xs(psquared,m1,m2)
    pref = 1/(psquared*(x1-x2))
    fac = (1-x1)*F2(x1) - (1-x2)*F2(x2)
    return -pref*fac