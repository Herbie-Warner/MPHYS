# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:54:09 2024

@author: herbi
"""
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Utilities.Utilities import stheta,vtheta,ctheta,ME,MMU,MNU,MZ,MW,xfunc,MGAMMA,g
from PVFunctions.CFunctions import C0,C11,C24,C23, C0_reg,C11_reg, C24_reg,C23_reg

prefactor = g**6 /(2048*np.pi**3)

def sum_term_12345(p1,p2,m1,m2,m3,p1dotp2):
    
    sval = -2*p1dotp2
    p1squared = -p1**2
    p2squared = -p2**2
    first = sval*(2*C11(p1squared,p2squared,m1,m2,m3,p1dotp2)  + 2*C23(p1squared,p2squared,m1,m2,m3,p1dotp2))
    second = -4*( C24(p1squared,p2squared,m1,m2,m3,p1dotp2)-0.5)
    
    return 1/(2*sval) * (first+second)


def sum_term_67(p1,p2,m1,m2,m3,p1dotp2):
    
    sval = -2*p1dotp2
    
    p1squared = -p1**2
    p2squared = -p2**2
    
    first = -2*sval*(C11(p1squared,p2squared,m1,m2,m3,p1dotp2) + C0(p1squared,p2squared,m1,m2,m3,p1dotp2) + C23(p1squared,p2squared,m1,m2,m3,p1dotp2))
    second = 12 * (C24(p1squared,p2squared,m1,m2,m3,p1dotp2)-1/6)
    return 1/(2*sval) * (first+second)

def sum_term_12345_reg(p1,p2,m1,m2,m3,p1dotp2):
    
    sval = -2*p1dotp2
    p1squared = -p1**2
    p2squared = -p2**2
    first = sval*(2*C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)  + 2*C23_reg(p1squared,p2squared,m1,m2,m3,p1dotp2))
    second = -4*( C24_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)-0.5)
    
    return 1/(2*sval) * (first+second)


def sum_term_67_reg(p1,p2,m1,m2,m3,p1dotp2):
    
    sval = -2*p1dotp2
    
    p1squared = -p1**2
    p2squared = -p2**2
    
    first = -2*sval*(C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2) + C0_reg(p1squared,p2squared,m1,m2,m3,p1dotp2) + C23_reg(p1squared,p2squared,m1,m2,m3,p1dotp2))
    second = 12 * (C24_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)-1/6)
    return 1/(2*sval) * (first+second)

def Dia1E(s,t):
    p1dotp2 = -s/2
    finite = -32*stheta**6*xfunc(s,t)
    summede = sum_term_12345_reg(ME, ME, ME, MGAMMA, ME, p1dotp2)
    summedmu = sum_term_12345_reg(MMU, MMU, MMU, MGAMMA, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia2E(s,t):
    p1dotp2 = -s/2
    finite = -2*stheta**4*(vtheta**2+1)*xfunc(s,t)/ctheta**2
    summede = sum_term_12345(ME, ME, ME, MZ, ME, p1dotp2)
    summedmu = sum_term_12345(MMU, MMU, MMU, MZ, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia3E(s,t):
    p1dotp2 = -s/2
    finite = 2*stheta**4/ctheta**2 * (s+2*t+s*vtheta**4*xfunc(s,t))/(-s+MZ**2)
    summede = sum_term_12345_reg(ME, ME, ME, MGAMMA, ME, p1dotp2)
    summedmu = sum_term_12345_reg(MMU, MMU, MMU, MGAMMA, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia4E(s,t):
    p1dotp2 = -s/2
    finite = stheta**2/(8*ctheta**4)*(vtheta**4*s*xfunc(s,t) + 6*(s+t)**2*vtheta**2 /s + s + 2*t)/(-s+MZ**2)
    summede = sum_term_12345(ME, ME, ME, MZ, ME, p1dotp2)
    summedmu = sum_term_12345(MMU, MMU, MMU, MZ, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia5E(s,t):
    p1dotp2 = -s/2
    finite = stheta**2/ctheta**2 * (vtheta*s*xfunc(s,t)-s-2*t)/(-s+MZ**2)
    summede = sum_term_12345(ME, ME, MNU, MW, MNU, p1dotp2)
    summedmu = sum_term_12345(MMU, MMU, MNU, MW, MNU, p1dotp2)
    return finite*(summede+summedmu)

def Dia6E(s,t):
    p1dotp2 = -s/2
    finite = 2*stheta**2/ctheta * (vtheta*s*xfunc(s,t)-s-2*t)/(-s+MZ**2)
    summede = sum_term_67(ME, ME, MW, MNU, MW, p1dotp2)
    summedmu =sum_term_67(MMU, MMU, MW, MNU, MW, p1dotp2)
    return finite*(summede+summedmu)

def Dia7E(s,t):
    p1dotp2 = -s/2
    finite = 8*stheta**4*xfunc(s,t)
    summede = sum_term_67(ME, ME, MW, MNU, MW, p1dotp2)
    summedmu =sum_term_67(MMU, MMU, MW, MNU, MW, p1dotp2)
    return finite*(summede+summedmu)

def totalVertexE(s,t):
    #print("here")
    """
    print(Dia1E(s, t).real )
    print(Dia2E(s, t).real )
    print(Dia3E(s, t).real )
    print(Dia4E(s, t).real )
    print(Dia5E(s, t).real )
    print(Dia6E(s, t).real )
    print(Dia7E(s, t).real )
    """
    
    val =  Dia1E(s, t) + Dia2E(s, t)+ Dia3E(s, t)+ Dia4E(s, t)+ Dia5E(s, t)+ Dia6E(s, t)+ Dia7E(s, t)
    #val =   Dia2E(s, t)+ Dia4E(s, t)+ Dia7E(s, t)
   # val =  Dia3E(s, t)+  Dia4E(s, t)+ Dia5E(s, t)+ Dia6E(s, t)
    
    return prefactor *val





def Dia1W(s,t):
    p1dotp2 = -s/2
    finite = 2*stheta**4/ctheta**2 * (vtheta**2*s*xfunc(s,t)+s+2*t)/(-s+MZ**2)
    summede = sum_term_12345_reg(ME, ME, ME, MGAMMA, ME, p1dotp2)
    summedmu = sum_term_12345_reg(MMU, MMU, MMU, MGAMMA, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia2W(s,t):
    p1dotp2 = -s/2
    finite = stheta**2/(8*ctheta**4) * (vtheta**4*s*xfunc(s,t)+6*(s+t)**2*vtheta**2/s + s + 2*t)/(-s+MZ**2)
    summede = sum_term_12345(ME, ME, ME, MZ, ME, p1dotp2)
    summedmu = sum_term_12345(MMU, MMU, MMU, MZ, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia3W(s,t):
    p1dotp2 = -s/2
    finite = stheta**2/(8*ctheta**4) * (vtheta**4*s**2*xfunc(s,t)+2*vtheta**2*(3*s**2+6*s*t+2*t**2) + s**2*xfunc(s,t))/(-s+MZ**2)**2
    summede = sum_term_12345_reg(ME, ME, ME, MGAMMA, ME, p1dotp2)
    summedmu = sum_term_12345_reg(MMU, MMU, MMU, MGAMMA, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia4W(s,t):
    p1dotp2 = -s/2
    finite = -(vtheta**2+1)/(128*ctheta**6) * (vtheta**4*s**2*xfunc(s,t)+2*(7*s**2+14*s*t+6*t**2)*vtheta**2+s**2*xfunc(s,t))/(-s+MZ**2)**2
    summede = sum_term_12345(ME, ME, ME, MZ, ME, p1dotp2)
    summedmu = sum_term_12345(MMU, MMU, MMU, MZ, MMU, p1dotp2)
    return finite*(summede+summedmu)

def Dia5W(s,t):
    p1dotp2 = -s/2
    finite = (1-vtheta)/(16*ctheta**4) * (vtheta**2*s**2*xfunc(s,t)-2*s*(s+2*t)*vtheta+s**2*xfunc(s,t))/(-s+MZ**2)**2
    summede = sum_term_12345(ME, ME, MNU, MW, MNU, p1dotp2)
    summedmu = sum_term_12345(MMU, MMU, MNU, MW, MNU, p1dotp2)
    return finite*(summede+summedmu)

def Dia6W(s,t):
    p1dotp2 = -s/2
    finite = (1-vtheta)/(8*ctheta**3) * (vtheta**2*s**2*xfunc(s,t)-2*s*(s+2*t)*vtheta+s**2*xfunc(s,t))/(-s+MZ**2)**2
    summede = sum_term_67(ME, ME, MW, MNU, MW, p1dotp2)
    summedmu =sum_term_67(MMU, MMU, MW, MNU, MW, p1dotp2)
    return finite*(summede+summedmu)

def Dia7W(s,t):
    p1dotp2 = -s/2
    finite = stheta*(vtheta-1)/(2*ctheta**2) * (vtheta*s*xfunc(s,t)-s-2*t)/(-s+MZ**2)
    summede = sum_term_67(ME, ME, MW, MNU, MW, p1dotp2)
    summedmu =sum_term_67(MMU, MMU, MW, MNU, MW, p1dotp2)
    return finite*(summede+summedmu)

def totalVertexW(s,t):
    val =  Dia1W(s, t) + Dia2W(s, t)+ Dia3W(s, t)+ Dia4W(s, t)+ Dia5W(s, t)+ Dia6W(s, t)+ Dia7W(s, t)
   
    """
    print(Dia1W(s, t).real )
    print(Dia2W(s, t).real )
    print(Dia3W(s, t).real )
    print(Dia4W(s, t).real )
    print(Dia5W(s, t).real )
    print(Dia6W(s, t).real )
    print(Dia7W(s, t).real ) 
    """
    
    
    #val =  Dia1W(s, t) + Dia2W(s, t)+ Dia3W(s, t)+ Dia4W(s, t)+ Dia5W(s, t)+ Dia6W(s, t)
   
  
    return prefactor *val


def totalVertex(s,t):
    return totalVertexE(s, t) + totalVertexW(s, t)


"""
ECM = 10
sval = ECM**2

tval = sval/2 * (np.cos(np.pi/2)-1)
VVal = totalVertexW(sval,tval)
"""
