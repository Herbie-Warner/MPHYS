# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:44:39 2024

@author: herbi
"""

import gmpy2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Precision import HPC,behaved_quadratic_routes
from Core.Quantities import DELTA

def F1(x):
   
    if abs(x) == 0:
        return 0
    return -1 - x*gmpy2.log(1-x)+x*gmpy2.log(-x+0)

def F2(x):
   
    if abs(x) == 0:
        return -1/2
    
    return -1/2 - x -x**2*gmpy2.log(-1+x+0)+x**2*gmpy2.log(0+x)

def F3(x):
    x += 0j
    
    term1 = 6*x**3*gmpy2.log(x+0j+0)
    term2 = x**3 * (6*gmpy2.log(-1+0j)-6*gmpy2.log(1-x+0))
    term3 = -6*x**3-3*x+6*gmpy2.log(1-x)-2
    inside = 1/6 * (term1+term2+term3)   
    return inside - gmpy2.log(1-x+0)

def B0(psquared,m1,m2):
    psquared = HPC(psquared)
    m1 = HPC(m1)
    m2 = HPC(m2)  
    x1, x2 = behaved_quadratic_routes(-psquared, psquared+m2**2-m1**2, m1**2)

    
    term1 = gmpy2.log(-psquared)
    term2 = gmpy2.log(1-x1)+F1(x1)
    term3 = gmpy2.log(1-x2)+F1(x2)
    return (DELTA - (term1+term2+term3)).real

def B1(psquared,m1,m2):
    psquared = HPC(psquared)
    m1 = HPC(m1)
    m2 = HPC(m2)  
    x1, x2 = behaved_quadratic_routes(-psquared, psquared+m2**2-m1**2, m1**2)
    #print(x1,x2)
    term1 = gmpy2.log(-psquared)
    term2 = gmpy2.log(1-x1)+F2(x1)
    term3 = gmpy2.log(1-x2)+F2(x2)
    return (-1/2*DELTA + 1/2 * (term1+term2+term3)).real

def B21(psquared,m1,m2):
    psquared = HPC(psquared)
    m1 = HPC(m1)
    m2 = HPC(m2)  
    x1, x2 = behaved_quadratic_routes(-psquared, psquared+m2**2-m1**2, m1**2)
  
    term1 = gmpy2.log(-psquared)
    term2 = gmpy2.log(1-x1)+F3(x1)
    term3 = gmpy2.log(1-x2)+F3(x2)
    return (1/3*DELTA - 1/3 * (term1+term2+term3)).real

def B22(psquared,m1,m2):
    psquared = HPC(psquared)
    m1 = HPC(m1)
    m2 = HPC(m2)  
    x1, x2 = behaved_quadratic_routes(-psquared, psquared+m2**2-m1**2, m1**2)
    
    term1 = -(m1**2+m2**2+1/3*psquared)*(DELTA+1/2)
    term2 = -m2**2+m2**2*gmpy2.log(m2**2)+(1/3 * psquared+m1**2)*gmpy2.log(-psquared)
    
    term3 = m1**2*(1-x1)*gmpy2.log(1-x1)+m1**2*x1*gmpy2.log(-x1)-m1**2 + 1/3 * psquared * (gmpy2.log(1-x1)+F3(x1))
    term4 = m1**2*(1-x2)*gmpy2.log(1-x2)+m1**2*x2*gmpy2.log(-x2)-m1**2 + 1/3 * psquared * (gmpy2.log(1-x2)+F3(x2))
    return (1/4 * (term1+term2+term3+term4)).real

def dbl_deriv_B0(psquared,m1,m2):
    psquared = HPC(psquared)
    m1 = HPC(m1)
    m2 = HPC(m2)  
    x1, x2 = behaved_quadratic_routes(-psquared, psquared+m2**2-m1**2, m1**2)
    
    pref = -1/(psquared*(x1-x2))
    val = (1-x1)*F1(x1)-(1-x2)*F1(x2)
    return pref * val

def dbl_deriv_B1(psquared,m1,m2):
    psquared = HPC(psquared)
    m1 = HPC(m1)
    m2 = HPC(m2)  
    x1, x2 = behaved_quadratic_routes(-psquared, psquared+m2**2-m1**2, m1**2)
    
    pref = 1/(psquared*(x1-x2))
    val = (1-x1)*F2(x1)-(1-x2)*F2(x2)
    return pref * val
