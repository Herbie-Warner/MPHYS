# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:17:32 2024

@author: herbi
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import MZ,MW,stheta,ctheta,MGAMMA, MH,vtheta,MNU
from Core.Quantities import MUP,MDOWN,MCHARM,MSTRANGE,MTOP,MBOTTOM,ME,MMU,MTAU,ckm_matrix
from PVFunctions.AFunctions import A
from PVFunctions.BFunctions import B0,B22,B1,B21

def F1(ks,m1,m2):
    term1 = A(m1)
    term2 = A(m2)
    term3 = 10*B22(ks,m1,m2)
    term4 = (4*ks-m1**2-m2**2)*B0(ks,m1,m2)
    term5 = 2*(m1**2+m2**2+1/3 * ks)
    return term1+term2+term3+term4+term5

def F2(m):
    return -(3*A(m)+2*m**2)

def F1_P(ks):
    if -ks.real > 4*abs(MW**2):
        return stheta**2 * F1(ks,MW,MW)
    return 0

def F1_Z(ks):
    if -ks.real > 4*abs(MW**2):
        return ctheta**2 * F1(ks,MW,MW)
    return 0

def F1_PZ(ks):
    if -ks.real > 4*abs(MW**2):
        return stheta*ctheta * F1(ks,MW,MW)
    return 0

def F1_W(ks):
    return stheta**2 * F1(ks,MGAMMA,MW)+ctheta**2 * F1(ks,MZ,MW)



def F2_P(ks):
    if -ks.real > 4*abs(MW**2): return 2*stheta**2*F2(MW)
    return 0

def F2_Z(ks):
    if -ks.real > 4*abs(MW**2): return 2*ctheta**2*F2(MW)
    return 0

def F2_PZ(ks):
    if -ks.real > 4*abs(MW**2): return 2*stheta*ctheta*F2(MW)
    return 0

def F2_W(ks):
    return ctheta**2 * F2(MW)+F2(MW)



def F3_P(ks):
    if -ks.real > 4*abs(MW**2): return 2*stheta**2*MW**2*B0(ks,MW,MW)
    return 0

def F3_Z(ks):
    if -ks.real > 4*abs(MW**2): return 2*stheta**4*MW**2*B0(ks,MW,MW)/ctheta**2
    return 0

def F3_PZ(ks):
    if -ks.real > 4*abs(MW**2): return -2*stheta**3*MW**2*B0(ks,MW,MW)/ctheta
    return 0

def F3_W(ks):
    return stheta**4/ctheta**2 * MW**2*B0(ks,MZ,MW) + stheta**2*MW**2 * B0(ks,MGAMMA,MW)



def F4_P(ks):
    return 0

def F4_Z(ks):
    if -ks.real > abs(MZ+MH)**2: return MZ**2/ctheta**4 * B0(ks,MH,MZ)
    return 0

def F4_PZ(ks):
    return 0

def F4_W(ks):
    return MW**2*B0(ks,MH,MW)



def F5_P(ks):
    return 0

def F5_Z(ks):
    
    if -ks.real > abs(MZ+MH)**2: return 1/ctheta**2 * B22(ks,MZ,MH)
    return 0

def F5_PZ(ks):
    return 0

def F5_W(ks):
    return B22(ks,MW,MH)



def F6_P(ks):
    if -ks.real > 4*abs(MW**2): return 4*stheta**2*B22(ks,MW,MW)
    return 0

def F6_Z(ks):
    if -ks.real > 4*abs(MW**2): return (ctheta**2-stheta**2)**2/ctheta**2 *B22(ks,MW,MW)
    return 0

def F6_PZ(ks):
    if -ks.real > 4*abs(MW**2): return 2*stheta/ctheta *(ctheta**2-stheta**2)*B22(ks,MW,MW)
    return 0
    
def F6_W(ks):
    return B22(ks,MZ,MW)



def F7_P(ks):
    if -ks.real > 4*abs(MW**2): return - 2*stheta**2*B22(ks,MW,MW)
    return 0

def F7_Z(ks):
   if -ks.real > 4*abs(MW**2): return - 2*ctheta**2*B22(ks,MW,MW)
   return 0

def F7_PZ(ks):
    if -ks > 4*abs(MW**2): return - 2*stheta*ctheta*B22(ks,MW,MW)
    return 0

def F7_W(ks):
    return - 2*ctheta**2*B22(ks,MZ,MW)- 2*stheta**2*B22(ks,MGAMMA,MW)



def F8_P(ks):
    if -ks.real > 4*abs(MW**2): return -2*stheta**2*A(MW)
    return 0

def F8_Z(ks):
    tot = 0
    if -ks.real > 4*abs(MZ**2): tot += -1/(2*ctheta**2)*(A(MZ))
    if -ks.real > 4*abs(MH**2): tot += -1/(2*ctheta**2)*(A(MH))
    if -ks.real > 4*abs(MW**2): tot += -1/2 * (stheta**2-ctheta**2)**2/ctheta**2 * A(MW)
    
    
    return tot

def F8_PZ(ks):
    if -ks.real > 4*abs(MZ**2): return stheta/ctheta *(stheta**2-ctheta**2)*A(MW)
    return 0

def F8_W():
    return -1/4 * (A(MZ)+A(MH)+2*A(MW))


def B_tilde(ks,m1,m2):
    inside = 8*B21(ks,m1,m2)+4*B1(ks,m1,m2)-2*B0(ks,m1,m2)
    return ks*inside

def F9_P_lep(ks):
    
    def interior(mlep):
        if np.sqrt(-ks) > 2* abs(mlep):
            return B_tilde(ks, mlep, mlep)
        return 0
    
    return stheta**2 * (interior(ME)+interior(MMU)+interior(MTAU))
    
def F9_P_had(ks):
    
    def interior(mlep,charge):
        if np.sqrt(-ks) > 2*abs(mlep):
            return 3*B_tilde(ks, mlep, mlep) * charge**2
        return 0
    
    u = interior(MUP,2/3)
    d = interior(MDOWN,2/3)
    c = interior(MCHARM,2/3)
    s = interior(MSTRANGE,2/3)
    t = interior(MTOP,2/3)
    b = interior(MBOTTOM,2/3)
    
    return stheta**2 * (u+d+c+s+t+b)

def F9_P(ks):
    return F9_P_had(ks) + F9_P_lep(ks)


def F9_Z_lep(ks):
    def interior(mlep):
        summed = 0
        if np.sqrt(-ks) > 2*abs(mlep):
            summed += B_tilde(ks, mlep, mlep) *(vtheta**2+1)/(16*ctheta**2) - mlep**2/(2*ctheta**2) *B0(ks,mlep,mlep)

        return summed + B_tilde(ks, MNU, MNU)/(8*ctheta**2)
    
    return interior(ME) + interior(MMU) + interior(MTAU)

def F9_Z_had(ks):
    l_p = ((8*stheta**2/3 +1 )/(4*ctheta))**2 + (1/(4*ctheta))**2
    l_m = ((4*stheta**2/3 -1 )/(4*ctheta))**2 + (1/(4*ctheta))**2
    
    def interior(mass,pm):
        if np.sqrt(-ks) < 2*abs(mass):
            return 0      
        return pm*B_tilde(ks, mass, mass)-mass**2/(2*ctheta**2)*B0(ks,mass,mass)
    
    u = interior(MUP, l_p)
    d = interior(MDOWN, l_m)    
    s = interior(MSTRANGE, l_m)    
    t = interior(MTOP, l_p)
    b = interior(MBOTTOM, l_m)
    c = interior(MCHARM, l_p)
    return u+d+s+t+b+c
   
    
def F9_Z(ks):
    return F9_Z_had(ks)+F9_Z_lep(ks)

def F9_PZ_lep(ks):
    def interior(mass):
        if np.sqrt(-ks) < 2*abs(mass):
            return 0      
        return B_tilde(ks, mass, mass) * (-stheta/(4*ctheta))
    
    return interior(ME)+interior(MMU)+interior(MTAU)


def F9_PZ_had(ks):
    l_p =stheta*(8*stheta**2+3)/(18*ctheta)
    l_m = stheta*(4*stheta**2-3)/(36*ctheta)
    
    def interior(mass,pm):
        if np.sqrt(-ks) < 2*abs(mass):
            return 0      
        return pm*B_tilde(ks, mass, mass)
    
    u = interior(MUP, l_p)
    d = interior(MDOWN, l_m)    
    s = interior(MSTRANGE, l_m)    
    t = interior(MTOP, l_p)
    b = interior(MBOTTOM, l_m)
    c = interior(MCHARM, l_p)
    return u+d+s+t+b+c
    
def F9_PZ(ks):
    return F9_PZ_lep(ks) + F9_PZ_had(ks)



def F9_W_lep(ks):
    
    def interior(mlep):
        if np.sqrt(-ks) < abs(mlep):
            return 0   
        return -2*B22(ks,MNU,mlep)+1/2 * A(mlep)-1/2 * (mlep**2+ks)*B0(ks,MNU,mlep)
    
    return interior(ME)+interior(MMU)+interior(MTAU)


def F9_W_had(ks):
    
    
    def interior(m1,m2,ck1,ck2):
        if np.sqrt(-ks) < abs(m1)+abs(m2):
            return 0   
        inside =  -2*B22(ks,m1,m2)+1/2 * A(m1) + 1/2*A(m2) -(2*m1*m2-ks-m1**2-m2**2)*B0(ks,m1,m2)
        return inside*(ckm_matrix[ck1,ck2]**2)
    
    ud = interior(MUP, MDOWN, 0, 0)
    us = interior(MUP, MSTRANGE, 0, 1)
    ub = interior(MUP, MBOTTOM, 0, 2)
    
    cd = interior(MCHARM, MDOWN, 1, 0)
    cs = interior(MCHARM, MSTRANGE,1, 1)
    cb = interior(MCHARM, MBOTTOM, 1, 2)
    
    td = interior(MTOP, MDOWN, 2, 0)
    ts = interior(MTOP, MSTRANGE, 2, 1)
    tb = interior(MTOP, MBOTTOM, 2, 2)
    
    return ud+us+ub+cd+cs+cb+td+ts+tb
    
 
    
   
def F9_W(ks):
    return F9_W_had(ks)+F9_W_lep(ks)


def F_PZ(ks):
    return F1_PZ(ks)+F2_PZ(ks)+F3_PZ(ks)+F4_PZ(ks)+F5_PZ(ks)+F6_PZ(ks)+F7_PZ(ks)+F8_PZ(ks) + F9_PZ(ks)

def F_P(ks):
    print(F1_P(ks))
    print(F2_P(ks))
    print(F3_P(ks))
    print(F4_P(ks))
    print(F5_P(ks))
    print(F6_P(ks))
    print(F7_P(ks))
    print(F8_P(ks))
    print(F9_P(ks))
    return F1_P(ks) + F2_P(ks)+ F3_P(ks)+ F4_P(ks)+ F5_P(ks)+ F6_P(ks)+ F7_P(ks)+ F8_P(ks) + F9_P(ks)

def F_Z(ks):
    return F1_Z(ks) + F2_Z(ks)+ F3_Z(ks)+ F4_Z(ks)+ F5_Z(ks)+ F6_Z(ks)+ F7_Z(ks)+ F8_Z(ks)+F9_Z(ks)

def F_W(ks):
    #strictly W+
    return F1_W(ks) + F2_W(ks)+ F3_W(ks)+ F4_W(ks)+ F5_W(ks)+ F6_W(ks)+ F7_W(ks)+ F8_W()+F9_W(ks)


"""


import matplotlib.pyplot as plt

figure = plt.figure()
ax = figure.add_subplot()

ecm = np.linspace(10,20,100)
svals = np.square(ecm)
vals = []
for s in svals:
    vals.append( F9_P_had(-s))



plottable = np.array([float(c.real) for c in vals])
ax.plot(ecm,plottable)

#print(F9_P(-1000**2))

"""

