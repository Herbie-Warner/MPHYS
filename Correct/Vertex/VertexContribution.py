# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:03:50 2024

@author: herbi
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import MZ,MW,stheta,ctheta,MGAMMA, ME,MMU,g,vtheta,ztheta,MNU
from Core.Functions import xfunc,yfunc,zfunc,tau
from PVFunctions.CFunctions import C0,C11,C23,C24,C0_reg,C11_reg,C23_reg,C24_reg

prefactor = g**6/(2048*(np.pi**3))

def V1(p1squared,p2squared,m1,m2,m3,p1dotp2):
    C11_val = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C23_val = C23(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C24_val = C24(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 2*(-2*p1dotp2) * (C11_val+C23_val) - 4*(C24_val-1/2)

def V2(p1squared,p2squared,m1,m2,m3,p1dotp2):
    C0_val = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C11_val = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C23_val = C23(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C24_val = C24(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return -2*(-2*p1dotp2) * (C0_val+C11_val+C23_val) + 12*(C24_val-1/6)

def V1_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    C11_val = C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C23_val = C23_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C24_val = C24_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 2*(-2*p1dotp2) * (C11_val+C23_val) - 4*(C24_val-1/2)

def V2_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    C0_val = C0_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C11_val = C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C23_val = C23_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C24_val = C24_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return -2*(-2*p1dotp2) * (C0_val+C11_val+C23_val) + 12*(C24_val-1/6)



def sum_12345(M_boson, p1dotp2,lep1,lep2): 
    one = V1(-ME**2,-ME**2,lep1,M_boson,lep1,p1dotp2)
    two = V1(-MMU**2,-MMU**2,lep2,M_boson,lep2,p1dotp2)
   
    return -(one + two)



def sum_12345_reg(M_boson, p1dotp2,lep1,lep2): 
    one = V1_reg(-ME**2,-ME**2,lep1,M_boson,lep1,p1dotp2)
    two = V1_reg(-MMU**2,-MMU**2,lep2,M_boson,lep2,p1dotp2)
   
    return -(one + two)

def sum_67(M_boson, p1dotp2,lep1,lep2):
    one = V2(-ME**2,-ME**2,lep1,M_boson,lep1,p1dotp2)
    two = V2(-MMU**2,-MMU**2,lep2,M_boson,lep2,p1dotp2)
   
    return -(one + two)

def F1(s,t):
    interior1 = 16*stheta**6*xfunc(s,t)/(-s)
    interior2 = stheta**4/(ctheta**2) * (vtheta**2*xfunc(s,t)+yfunc(s,t))/(-s+MZ**2)
    return (interior1+interior2)*sum_12345_reg(MGAMMA, -s/2,ME,MMU)*prefactor

def F2(s,t):
    interior1 = 16*stheta**4*(1+vtheta**2)*xfunc(s,t)/(16*ctheta**2*(-s))
    interior2 = stheta**2/(16*ctheta**4) * (vtheta**4*xfunc(s,t)+6*vtheta**2*zfunc(s,t)+yfunc(s,t))/(-s+MZ**2)
    return (interior1+interior2)*sum_12345(MZ, -s/2,ME,MMU)*prefactor

def F3(s,t):
    interior1 = stheta**4/(ctheta**2) * (vtheta**2*xfunc(s,t)+yfunc(s,t))/(-s+MZ**2)
    interior2 = -stheta**2/(16*ctheta**4) * (ztheta*s*xfunc(s,t)-8*vtheta**2*t**2/s)/(-s+MZ**2)**2
    return (interior1+interior2)*sum_12345_reg(MGAMMA, -s/2,ME,MMU)*prefactor

def F4(s,t):
    interior1 = stheta**2/(16*ctheta**4) * (vtheta**4*xfunc(s,t)+6*vtheta**2*zfunc(s,t)+yfunc(s,t))/(-s+MZ**2)
    interior2 =-(vtheta**2+1)/(256*ctheta**6) * (ztheta*s*xfunc(s,t)+vtheta**2*s*yfunc(s,t))/(-s+MZ**2)**2
    return (interior1+interior2)*sum_12345(MZ, -s/2,ME,MMU)*prefactor


def F5(s,t):
    interior1 = stheta**2/(2*ctheta**2) * (vtheta*xfunc(s,t)-yfunc(s,t))/(-s+MZ**2)
    interior2 = -(vtheta-1)/(32*ctheta**4) * (vtheta**2*s*xfunc(s,t)-2*s*yfunc(s,t)+s*xfunc(s,t))/(-s+MZ**2)**2
    return (interior1+interior2)*sum_12345(MW, -s/2,MNU,MNU)*prefactor


def F6(s,t):
    interior1 = -stheta**2 * (vtheta*xfunc(s,t)-yfunc(s,t))/(-s+MZ**2)
    interior2 = (vtheta-1)/(16*ctheta**4) * (vtheta**2*s*xfunc(s,t)-2*s*yfunc(s,t)+s*xfunc(s,t))/(-s+MZ**2)**2
    return (interior1+interior2)*sum_67(MNU, -s/2,MW,MW)*prefactor

def F7(s,t):
    interior1 = 4*stheta**4*xfunc(s,t)/(-s)
    interior2 = stheta**2*(vtheta-1)/(4*ctheta**2) * (vtheta*xfunc(s,t)-yfunc(s,t))/(-s+MZ**2)
    return (interior1+interior2)*sum_67(MNU, -s/2,MW,MW)*prefactor


from LeadingOrder.LeadingOrderContribution import LeadingOrderTotal
import gmpy2

def VertexBrem(s,t):
    pre = g**2/(16*np.pi**2) * stheta**2*s * LeadingOrderTotal(s, t)
    inside1 = tau(s,ME,ME) * gmpy2.log(ME**2/MGAMMA**2)
    inside2 = tau(s,MMU,MMU) * gmpy2.log(MMU**2/MGAMMA**2)
    return 2*pre*(inside1+inside2)


def VertexBremZ(s,t):
    #pre = g**2/(16*np.pi**2) * stheta**2*s * LeadingOrderTotal(s, t)/(ctheta**2*16*stheta**2) * vtheta
    #inside1 = tau(s,ME,ME) * gmpy2.log(s/MZ**2)
    #inside2 = tau(s,MMU,MMU) * gmpy2.log(s/MZ**2)
    
    alpha = g**2/(4*np.pi)
    
    pre = alpha/(4*np.pi) * np.log(s/MZ**2)**2
    
    
    return 2*pre*LeadingOrderTotal(s, t) 



def VertexCorrectionTotal(s,t):
    

    return F2(s,t)+F4(s,t)+F5(s,t)+F6(s,t)+F7(s,t) #+F3(s,t)+F1(s,t) #-VertexBremZ(s, t)#-VertexBrem(s, t)


