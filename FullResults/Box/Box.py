# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:10:45 2024

@author: herbi
"""


import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from Utilities.Utilities import MZ,ME,MMU,stheta,vtheta,ctheta,MNU,MW,g,xfunc,MGAMMA
from PVFunctions.DFunctions import D11,D12,D13,D24,D25,D26,D27

prefactor = g**6/((2*np.pi)**3 * 256)

small_imag_param = 1j*1e-30
    
def sigma_13(sigma_12):
    return -sigma_12
    
def sigma_24(sigma_11,sigma_12):
    return sigma_11+sigma_12
    
def sigma_25(sigma_11,sigma_12,s,t):
    t += small_imag_param
    return (-sigma_11 - (1+2*s/t)*sigma_12).real
    
def sigma_26(sigma_12, s,t):
    t += small_imag_param
    return (2*s/t * sigma_12).real
    
def sigma_27(sigma_11,sigma_12,t):
    t+=small_imag_param
    return (-2/t * (sigma_11+3*sigma_12)).real


def box1(s,t):
    sigma_11_E = 32*t*xfunc(s,t)
    sigma_12_E = 32*t**3/s**2
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    
    
    sigma_11_W = -4*stheta**4*(8*stheta**4-vtheta)*s*t*xfunc(s,t) + 4*stheta**4*t**3/s
    sigma_12_W = -2*t**3*stheta**4/s * (vtheta**2-1)
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
    D11V = D11(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MGAMMA)
    D12V = D12(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MGAMMA)
    D13V = D13(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MGAMMA)
    D24V = D24(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MGAMMA)
    D25V = D25(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MGAMMA)
    D26V = D26(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MGAMMA)
    D27V = D27(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MGAMMA)
   
    E_tot = D11V*sigma_11_E + D12V*sigma_12_E + D13V*sigma_13_E + D24V*sigma_24_E + D25V*sigma_25_E + D26V*sigma_26_E + D27V*sigma_27_E
    W_tot = 1/(ctheta**2*(-s+MZ**2))*(D11V*sigma_11_W + D12V*sigma_12_W + D13V*sigma_13_W + D24V*sigma_24_W + D25V*sigma_25_W + D26V*sigma_26_W + D27V*sigma_27_W)
    return prefactor * (0*E_tot+W_tot)
    
   
def box2(s,t):
    sigma_11_E = 2*t*stheta**4/ctheta**2 * (xfunc(s,t)*vtheta**2+1/s * (s+2*t))
    sigma_12_E = 2*t**3*stheta**4/(s**2*ctheta**2) * (vtheta**2-1)
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = -t*stheta**2*s/8 *(xfunc(s,t)*(vtheta**4+6*vtheta**2+1) - 8*vtheta**2*t**2/s**2)
    sigma_12_W = -t**3*stheta**2*(vtheta**2-1)**2 /(8*s*ctheta**2)
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
    D11V = D11(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MZ)
    D12V = D12(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MZ)
    D13V = D13(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MZ)
    D24V = D24(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MZ)
    D25V = D25(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MZ)
    D26V = D26(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MZ)
    D27V = D27(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MGAMMA,MMU, MZ)
   
    E_tot = D11V*sigma_11_E + D12V*sigma_12_E + D13V*sigma_13_E + D24V*sigma_24_E + D25V*sigma_25_E + D26V*sigma_26_E + D27V*sigma_27_E
    W_tot = 1/(ctheta**2*(-s+MZ**2))*(D11V*sigma_11_W + D12V*sigma_12_W + D13V*sigma_13_W + D24V*sigma_24_W + D25V*sigma_25_W + D26V*sigma_26_W + D27V*sigma_27_W)
    return prefactor * (E_tot+W_tot)
  
def box3(s,t):
    sigma_11_E = 2*t*stheta**4/ctheta**2 * (xfunc(s,t)*vtheta**2+1/s * (s+2*t))
    sigma_12_E = 2*t**3*stheta**4/(s**2*ctheta**2) * (vtheta**2-1)
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = -t*stheta**2*s/8 *(xfunc(s,t)*(vtheta**4+6*vtheta**2+1) - 8*vtheta**2*t**2/s**2)
    sigma_12_W = -t**3*stheta**2*(vtheta**2-1)**2 /(8*s*ctheta**2)
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
    D11V = D11(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MGAMMA)
    D12V = D12(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MGAMMA)
    D13V = D13(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MGAMMA)
    D24V = D24(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MGAMMA)
    D25V = D25(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MGAMMA)
    D26V = D26(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MGAMMA)
    D27V = D27(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MGAMMA)
   
    E_tot = D11V*sigma_11_E + D12V*sigma_12_E + D13V*sigma_13_E + D24V*sigma_24_E + D25V*sigma_25_E + D26V*sigma_26_E + D27V*sigma_27_E
    W_tot = 1/(ctheta**2*(-s+MZ**2))*(D11V*sigma_11_W + D12V*sigma_12_W + D13V*sigma_13_W + D24V*sigma_24_W + D25V*sigma_25_W + D26V*sigma_26_W + D27V*sigma_27_W)
    return prefactor * (E_tot+W_tot)
    
def box4(s,t):
    sigma_11_E = t*stheta**2/(8*ctheta**4) * (xfunc(s,t)*(vtheta**4+6*vtheta*2+1)-8*t**2/s**2 * vtheta**2)
    sigma_12_E = 8*stheta**6/ctheta**4 * (4*stheta**4-vtheta)*t**3/s**2
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = -s*t/(128*ctheta**4)*(xfunc(s,t)*(vtheta**6+15*vtheta**2*(vtheta**2+1)) + 1/s * (s+2*t)-6*t**2*vtheta**2/s**2 * (3*vtheta**2+2))
    sigma_12_W = -t**3*(vtheta**2-1)**3 / (128*s*ctheta**4)
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
    D11V = D11(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MZ)
    D12V = D12(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MZ)
    D13V = D13(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MZ)
    D24V = D24(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MZ)
    D25V = D25(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MZ)
    D26V = D26(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MZ)
    D27V = D27(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MZ,MMU, MZ)
   
    E_tot = D11V*sigma_11_E + D12V*sigma_12_E + D13V*sigma_13_E + D24V*sigma_24_E + D25V*sigma_25_E + D26V*sigma_26_E + D27V*sigma_27_E
    W_tot = 1/(ctheta**2*(-s+MZ**2))*(D11V*sigma_11_W + D12V*sigma_12_W + D13V*sigma_13_W + D24V*sigma_24_W + D25V*sigma_25_W + D26V*sigma_26_W + D27V*sigma_27_W)
    return prefactor * (E_tot+W_tot)
    

    
def box5(s,t):
    sigma_11_E = 4*t*stheta**2*(s+t)**2/s**2
    sigma_12_E = 0
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = -t/(4*s) * (s+t)**2 * (vtheta-1)**2
    sigma_12_W = 0
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
    D11V = D11(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MW,MMU, MW)
    D12V = D12(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MW,MMU, MW)
    D13V = D13(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MW,MMU, MW)
    D24V = D24(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MW,MMU, MW)
    D25V = D25(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MW,MMU, MW)
    D26V = D26(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MW,MMU, MW)
    D27V = D27(-ME**2, -MMU**2, -MMU*2, -t/2, (s+t)/2, -s/2, ME, MW,MMU, MW)
   
    E_tot = D11V*sigma_11_E + D12V*sigma_12_E + D13V*sigma_13_E + D24V*sigma_24_E + D25V*sigma_25_E + D26V*sigma_26_E + D27V*sigma_27_E
    W_tot = 1/(ctheta**2*(-s+MZ**2))*(D11V*sigma_11_W + D12V*sigma_12_W + D13V*sigma_13_W + D24V*sigma_24_W + D25V*sigma_25_W + D26V*sigma_26_W + D27V*sigma_27_W)
    return prefactor * (E_tot+W_tot)
    
    
    
def box6(s,t):
    u = -s-t
    sigma_11_E = -32*u*xfunc(s,t)
    sigma_12_E = -32*u**3/s**2
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = 2*u*stheta**4*s*(xfunc(s,u)*vtheta**2-1/s*(s+2*u))
    sigma_12_W = 2*u**3*stheta**4/s * (vtheta**2+1)
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
def box7(s,t):
    u = -s-t
    sigma_11_E = -2*u*stheta**4/ctheta**2 * (xfunc(s,u)*vtheta**2-1/2*(s+2*u))
    sigma_12_E =-2*u**3*stheta**4*(vtheta**2+1)/(s**2*ctheta**2)
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = u*stheta**2*s*xfunc(s,u)*(vtheta**4-2*vtheta**2+1)/(8*ctheta**2)
    sigma_12_W = u**3*stheta**2*(vtheta**4+6*vtheta**2+1)/(8*s*ctheta**2)
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
def box8(s,t):
    None #is box7
    
    
def box9(s,t):
    u = -s-t
    sigma_11_E = -u*stheta**2/(8*ctheta**4) *xfunc(s,u)*(vtheta**4-2*vtheta**2+1)
    sigma_12_E = -u**3*stheta**2/(8*s**2*ctheta**4) * (vtheta**4+6*vtheta**2+2)
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_26(sigma_12_E,s,t)
    sigma_27_E = sigma_27(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = u*s/(128*ctheta**4) * (xfunc(s,u)*(vtheta**6-3*vtheta**4+3*vtheta**2)-(s+2*u)/s + 6*u**2*vtheta**2*(3*vtheta**2+2)/s**2)
    sigma_12_W = u**3*(vtheta**6+15*vtheta**4+15*vtheta**2+1)/(128*s*ctheta**4)
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_26(sigma_12_W,s,t)
    sigma_27_W = sigma_27(sigma_11_W,sigma_12_W,t)
    
def box5(s,t):
    sigma_11_E = 4*t*stheta**2*(s+t)**2 / s**2
    sigma_12_E = 0
    sigma_13_E = sigma_13(sigma_12_E)
    sigma_24_E = sigma_24(sigma_11_E,sigma_12_E)
    sigma_25_E = sigma_25(sigma_11_E,sigma_12_E,s,t)
    sigma_26_E = sigma_24(sigma_12_E,s,t)
    sigma_27_E = sigma_24(sigma_11_E,sigma_12_E,t)
    
    sigma_11_W = (vtheta-4*stheta**4)*(s*t*xfunc(s,t)-t**3/s)
    sigma_12_W = 0
    sigma_13_W = sigma_13(sigma_12_W)
    sigma_24_W = sigma_24(sigma_11_W,sigma_12_W)
    sigma_25_W = sigma_25(sigma_11_W,sigma_12_W,s,t)
    sigma_26_W = sigma_24(sigma_12_W,s,t)
    sigma_27_W = sigma_24(sigma_11_W,sigma_12_W,t)
  
    
    
 
import matplotlib.pyplot as plt
figure = plt.figure()
ax = figure.add_subplot()
ecm = 200
sval = ecm**2

theta = np.linspace(0,2*np.pi,100)

tval = sval/2 * (np.cos(theta)-1)

#ax.plot(theta,box1(sval+0j, tval+0j))

box1(sval+0j,sval/2*(np.cos(np.pi)-1)+0j)

    