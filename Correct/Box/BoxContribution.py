# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:22:36 2024

@author: herbi
"""

import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import MZ,MW,stheta,ctheta,MGAMMA,g,vtheta,ME,MMU,MNU,CONVERSION_GEV_TO_mBarns,ztheta
from Core.Functions import xfunc,yfunc,zfunc
from PVFunctions.DFunctions import D11,D12,D13,D24,D25,D26,D27


prefactor = g**6/(2*np.pi)**3 * 1/256


def get_13(s_12):
    return -s_12

def get_24(s_11,s_12):
    return s_11 + s_12

def get_25(s_11,sig_12_over_t,s,t):
    return -s_11 - (t+2*s)*sig_12_over_t

def get_26(s_12_over_t,s):
    return 2*s * s_12_over_t


def get_27(s_11_over_t,s_12_over_t):
    return -2*(s_11_over_t+3*s_12_over_t)


def box1(s,t):
    sigma_E_11 = 32*t*xfunc(s,t)
    sigma_E_12 = 32*t**3/s**2
    sigma_E_12_over_t = 32*t**2/s**2
    sigma_E_11_over_t = 32*xfunc(s,t)
    
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = -2*t*stheta**4*(s*xfunc(s,t)*vtheta**2+(s+2*t))
    sigma_W_12 = -2*t**3*stheta**4/s * (vtheta**2-1)
    sigma_W_12_over_t = -2*t**2*stheta**4/s * (vtheta**2-1)
    sigma_W_11_over_t =  -2*stheta**4*(s*xfunc(s,t)*vtheta**2+(s+2*t))
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = -t/2
    p1p3 = (s+t)/2
    p2p3 = -s/2
    
    m1 = ME
    m2 = MGAMMA
    m3 = MMU
    m4 = MGAMMA
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))


def box2(s,t):
    sigma_E_11 = 2*t*stheta**4/ctheta**2 * (xfunc(s,t)*vtheta**2 + yfunc(s,t))
    sigma_E_12 = 2*t**3*stheta**4 /(s**2*ctheta**2) * (vtheta**2-1)
    
    sigma_E_11_over_t = 2*stheta**4/ctheta**2 * (xfunc(s,t)*vtheta**2 + yfunc(s,t)) 
    sigma_E_12_over_t =  2*t**2*stheta**4 /(s**2*ctheta**2) * (vtheta**2-1)
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = -t*stheta**2*s/8 * (xfunc(s,t)*ztheta - 8*t**2/s**2 * vtheta**2)
    sigma_W_12 = -t**3*stheta**2/(8*s*ctheta**2) * (vtheta**2-1)**2
    sigma_W_12_over_t = -stheta**2*s/8 * (xfunc(s,t)*ztheta - 8*t**2/s**2 * vtheta**2)
    sigma_W_11_over_t =  -t**2*stheta**2/(8*s*ctheta**2) * (vtheta**2-1)**2
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = -t/2
    p1p3 = (s+t)/2
    p2p3 = -s/2
    
    m1 = ME
    m2 = MGAMMA
    m3 = MMU
    m4 = MZ
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))


def box3(s,t):
    sigma_E_11 = 2*t*stheta**4/ctheta**2 * (xfunc(s,t)*vtheta**2 + yfunc(s,t))
    sigma_E_12 = 2*t**3*stheta**4 /(s**2*ctheta**2) * (vtheta**2-1)
    
    sigma_E_11_over_t = 2*stheta**4/ctheta**2 * (xfunc(s,t)*vtheta**2 + yfunc(s,t)) 
    sigma_E_12_over_t =  2*t**2*stheta**4 /(s**2*ctheta**2) * (vtheta**2-1)
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = -t*stheta**2*s/8 * (xfunc(s,t)*ztheta - 8*t**2/s**2 * vtheta**2)
    sigma_W_12 = -t**3*stheta**2/(8*s*ctheta**2) * (vtheta**2-1)**2
    sigma_W_12_over_t = -stheta**2*s/8 * (xfunc(s,t)*ztheta - 8*t**2/s**2 * vtheta**2)
    sigma_W_11_over_t =  -t**2*stheta**2/(8*s*ctheta**2) * (vtheta**2-1)**2
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = -t/2
    p1p3 = (s+t)/2
    p2p3 = -s/2
    
    m1 = ME
    m2 = MZ
    m3 = MMU
    m4 = MGAMMA
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))



def box4(s,t):
    sigma_E_11 = t*stheta**2/(8*ctheta**4) * (xfunc(s,t)*ztheta - 8*t**2/s**2 * vtheta**2)
    sigma_E_12 = t**3*stheta**2/(8*s*ctheta**4) * (vtheta**2-1)**2
    
    sigma_E_11_over_t = stheta**2/(8*ctheta**4) * (xfunc(s,t)*ztheta - 8*t**2/s**2 * vtheta**2)
    sigma_E_12_over_t =  t**2*stheta**2/(8*s*ctheta**4) * (vtheta**2-1)**2
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = -s*t/(128*ctheta**4) * (xfunc(s,t)*(vtheta**6+15*vtheta**2*(vtheta**2+1)) + yfunc(s,t)-6*t**2*vtheta**2/s**2 * (3*vtheta**2+2))
    sigma_W_12 = -t**3/(128*ctheta**4 * s) * (vtheta**2-1)**3
    sigma_W_12_over_t =  -s/(128*ctheta**4) * (xfunc(s,t)*(vtheta**6+15*vtheta**2*(vtheta**2+1)) + yfunc(s,t)-6*t**2*vtheta**2/s**2 * (3*vtheta**2+2))
    sigma_W_11_over_t =  -t**2/(128*ctheta**4 * s) * (vtheta**2-1)**3
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = -t/2
    p1p3 = (s+t)/2
    p2p3 = -s/2
    
    m1 = ME
    m2 = MZ
    m3 = MMU
    m4 = MZ
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))

def box5(s,t):
    sigma_E_11 = 4*t*stheta**2*zfunc(s,t)
    sigma_E_12 = 0
    
    sigma_E_11_over_t =4*stheta**2*zfunc(s,t)
    sigma_E_12_over_t =  0
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = -t/(4*s) * (s+t)**2 * (vtheta-1)**2
    sigma_W_12 = 0
    sigma_W_12_over_t =  -1/(4*s) * (s+t)**2 * (vtheta-1)**2
    sigma_W_11_over_t =  0
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = -t/2
    p1p3 = (s+t)/2
    p2p3 = -s/2
    
    m1 = ME
    m2 = MW
    m3 = MMU
    m4 = MW
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))

def box6(s,t):
    u = -s-t
    sigma_E_11 = -32*u*xfunc(s,u)
    sigma_E_12 = -32*u**3/s**2
    
    sigma_E_11_over_t =-32*xfunc(s,u)
    sigma_E_12_over_t =  -32*u**2/s**2
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = 2*u*stheta**4*s*(xfunc(s,u)*vtheta**2-yfunc(s,u))
    sigma_W_12 = 2*u**3*stheta**4/s * (vtheta**2+1)
    sigma_W_12_over_t = 2*stheta**4*s*(xfunc(s,u)*vtheta**2-yfunc(s,u))
    sigma_W_11_over_t =  2*u**2*stheta**4/s * (vtheta**2+1)
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = u/2
    p1p3 = -t
    p2p3 = -s/2
    
    m1 = ME
    m2 = MGAMMA
    m3 = MMU
    m4 = MGAMMA
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))

def box7(s,t):
    u = -s-t
    sigma_E_11 = -2*u*stheta*84/ctheta**2 * (xfunc(s,u)*vtheta**2-yfunc(s,u))
    sigma_E_12 = -2*u**3*stheta**3/(s**2*ctheta**2) * (vtheta**2+1)
    
    sigma_E_11_over_t =-2*stheta*84/ctheta**2 * (xfunc(s,u)*vtheta**2-yfunc(s,u))
    sigma_E_12_over_t =  -2*u**2*stheta**3/(s**2*ctheta**2) * (vtheta**2+1)
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = u*stheta**2*s/(8*ctheta**2) * xfunc(s,u) * (vtheta**4-2*vtheta**2+1)
    sigma_W_12 = u**3*stheta**2/(8*s*ctheta**2) * ztheta
    sigma_W_12_over_t = stheta**2*s/(8*ctheta**2) * xfunc(s,u) * (vtheta**4-2*vtheta**2+1)
    sigma_W_11_over_t =  u**2*stheta**2/(8*s*ctheta**2) * ztheta
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = u/2
    p1p3 = -t
    p2p3 = -s/2
    
    m1 = ME
    m2 = MZ
    m3 = MMU
    m4 = MGAMMA
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))


def box8(s,t):
    u = -s-t
    sigma_E_11 = -2*u*stheta*84/ctheta**2 * (xfunc(s,u)*vtheta**2-yfunc(s,u))
    sigma_E_12 = -2*u**3*stheta**3/(s**2*ctheta**2) * (vtheta**2+1)
    
    sigma_E_11_over_t =-2*stheta*84/ctheta**2 * (xfunc(s,u)*vtheta**2-yfunc(s,u))
    sigma_E_12_over_t =  -2*u**2*stheta**3/(s**2*ctheta**2) * (vtheta**2+1)
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = u*stheta**2*s/(8*ctheta**2) * xfunc(s,u) * (vtheta**4-2*vtheta**2+1)
    sigma_W_12 = u**3*stheta**2/(8*s*ctheta**2) * ztheta
    sigma_W_12_over_t = stheta**2*s/(8*ctheta**2) * xfunc(s,u) * (vtheta**4-2*vtheta**2+1)
    sigma_W_11_over_t =  u**2*stheta**2/(8*s*ctheta**2) * ztheta
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = u/2
    p1p3 = -t
    p2p3 = -s/2
    
    m1 = ME
    m2 = MGAMMA
    m3 = MMU
    m4 = MZ
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))

def box9(s,t):
    u = -s-t
    sigma_E_11 = -u*stheta**2/(8*ctheta**4) *xfunc(s,u) * (vtheta**4-2*vtheta**2+1)
    sigma_E_12 = -u**3*stheta**2/(8*s**2*ctheta**4) * ztheta
    
    sigma_E_11_over_t =-stheta**2/(8*ctheta**4) *xfunc(s,u) * (vtheta**4-2*vtheta**2+1)
    sigma_E_12_over_t = -u**2*stheta**2/(8*s**2*ctheta**4) * ztheta
    
    sigma_E_13 = get_13(sigma_E_12)
    sigma_E_24 = get_24(sigma_E_11,sigma_E_12)
    sigma_E_25 = get_25(sigma_E_11,sigma_E_12_over_t,s,t)
    sigma_E_26 = get_26(sigma_E_12_over_t, s)
    sigma_E_27 = get_27(sigma_E_11_over_t,sigma_E_12_over_t)
    
    
    sigma_W_11 = u*s/(128*ctheta**4) * (xfunc(s,u)*(vtheta**6-3*vtheta**4+3*vtheta**2)-yfunc(s,u)+6*u**2/s**2 * vtheta**2*(3*vtheta**2+2))
    sigma_W_12 = u**3/(128*s*ctheta**4) * (vtheta**6+15*vtheta**4+15*vtheta**2+1)
    sigma_W_12_over_t =s/(128*ctheta**4) * (xfunc(s,u)*(vtheta**6-3*vtheta**4+3*vtheta**2)-yfunc(s,u)+6*u**2/s**2 * vtheta**2*(3*vtheta**2+2))
    sigma_W_11_over_t =  u**2/(128*s*ctheta**4) * (vtheta**6+15*vtheta**4+15*vtheta**2+1)
    
    
    sigma_W_13 = get_13(sigma_W_12)
    sigma_W_24 = get_24(sigma_W_11,sigma_W_12)
    sigma_W_25 = get_25(sigma_W_11,sigma_W_12_over_t,s,t)
    sigma_W_26 = get_26(sigma_W_12_over_t, s)
    sigma_W_27 = get_27(sigma_W_11_over_t,sigma_W_12_over_t)
    
    p1p2 = u/2
    p1p3 = -t
    p2p3 = -s/2
    
    m1 = ME
    m2 = MZ
    m3 = MMU
    m4 = MZ
    
    D11_val = D11(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    D12_val = D12(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D13_val = D13(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)
    D24_val = D24(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)    
    D25_val = D25(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)   
    D26_val = D26(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3,m1,m2,m3,m4)   
    D27_val = D27(-ME**2, -MMU**2, -MMU**2, p1p2, p1p3, p2p3, m1,m2,m3,m4)
    
    
  
    sigE = sigma_E_11*D11_val +sigma_E_12*D12_val +sigma_E_13*D13_val +sigma_E_24*D24_val +sigma_E_25*D25_val +sigma_E_26*D26_val +sigma_E_27*D27_val 
    sigW = sigma_W_11*D11_val +sigma_W_12*D12_val +sigma_W_13*D13_val +sigma_W_24*D24_val +sigma_W_25*D25_val +sigma_W_26*D26_val +sigma_W_27*D27_val 
     
    return prefactor * (sigE + sigW/(ctheta**2 * (-s+MZ**2)))


def boxTotal(s,t):
    return float((box3(s, t)+box6(s, t)+box7(s, t)+box8(s, t)).real)

    return float((box1(s, t)+box2(s, t)+box3(s, t)+box4(s, t)+box5(s, t)+box6(s, t)+box7(s, t)+box8(s, t)+box9(s, t)).real)

#print(box1(20**2,20**2 /2)*CONVERSION_GEV_TO_mBarns)
    
    
    
    
    