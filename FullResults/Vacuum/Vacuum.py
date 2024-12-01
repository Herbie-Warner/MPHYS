# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:37:45 2024

@author: herbi
"""

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from PVFunctions.BFunctions import B0,B21,B1,B22
from Utilities.Utilities import MZ,ME,MMU,stheta,vtheta,ctheta,MNU,MW,g,xfunc
from PVFunctions.AFunctions import A
from Lo.LOContribution import LO_Z, LO_GAMMA, LO_inter

def BTILDE(psquared,mf):
    inside = 8*B21(psquared,mf,mf)+4*B1(psquared,mf,mf)-2*B0(psquared,mf,mf)
    return inside
    

def ALP(psquared,mf):
    inside = BTILDE(psquared,mf)
    return psquared*stheta**2*inside

def AL0(psquared,mf):
    NU = BTILDE(psquared,MNU)
    ferm = BTILDE(psquared,mf)
    extra = B0(psquared,mf,mf)
    return psquared/(8*ctheta**2)*NU + psquared/(16*ctheta**2) * (vtheta**2+1)*ferm - mf**2/(2*ctheta**2)*extra

def ALOP(psquared,mf):
    BT = BTILDE(psquared,mf)
    return -psquared*stheta/(4*ctheta) * vtheta* BT

def A1(psquared,m1,m2):
    term1 = (m1**2+m2**2-4*psquared)*B0(psquared,m1,m2)
    term2 = -A(m1)
    term3 = -A(m2)
    term4 = -10*B22(psquared,m1,m2)
    term5 = -2*(m1**2+m2**2+1/3*psquared)
    return -(term1+term2+term3+term4+term5)

def A10(psquared):
    return ctheta**2*A1(psquared,MW,MW)


def A1P(psquared):
    return stheta**2*A1(psquared,MW,MW)


def A10P(psquared):
    return ctheta*stheta*A1(psquared,MW,MW)

def DELTA_P(psquared):
    s = -psquared
    pref = g**2/(16*np.pi**2*(-s))
    
    AP = ALP(psquared,ME)+ALP(psquared,MMU)+A1P(psquared)
    return AP*pref

def DELTA_0(psquared):
    s = -psquared
    pref = g**2/(16*np.pi**2*(-s+MZ**2))
    
    AP = AL0(psquared,ME)+AL0(psquared,MMU)+A10(psquared)
    APR = AL0(-MZ**2,ME)+AL0(-MZ**2,MMU)+A10(-MZ**2)
    return (AP-APR)*pref


def photon_self_energy(s,t):

    pref = DELTA_P(-s)
    return pref*(2*LO_GAMMA(s, t)+LO_inter(s, t))

def Z_self_energy(s,t):

    pref = DELTA_0(-s)
    return pref*(2*LO_Z(s, t)+LO_inter(s, t))


def mix_self_energy(s,t):
    pref = g**6/(64*((2*np.pi)**3))
    
    term1 = -stheta**3*vtheta*xfunc(s,t)/(ctheta*s*(-s+MZ**2))
    term3 = -stheta*vtheta*(vtheta**2*s**2*xfunc(s,t)-s**2-2*s*t+2*t**2)/(16*ctheta**3*s**2*(-s+MZ**2)**2)
       
    fac = A10P(-s)
    
    return pref*fac*(term1+term3)

def vacuum(s,t):
    return photon_self_energy(s, t) + Z_self_energy(s, t) + mix_self_energy(s, t)