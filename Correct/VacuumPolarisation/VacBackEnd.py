# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:17:32 2024

@author: herbi
"""


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import MZ,MW,stheta,ctheta,MGAMMA, MH
from PVFunctions.AFunctions import A
from PVFunctions.BFunctions import B0,B22

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
    return stheta**2 * F1(ks,MW,MW)

def F1_Z(ks):
    return ctheta**2 * F1(ks,MW,MW)

def F1_PZ(ks):
    return stheta*ctheta * F1(ks,MW,MW)

def F1_W(ks):
    return stheta**2 * F1(ks,MGAMMA,MW)+ctheta**2 * F1(ks,MZ,MW)



def F2_P(ks):
    return 2*stheta**2*F2(MW)

def F2_Z(ks):
    return 2*ctheta**2*F2(MW)

def F2_PZ(ks):
    return 2*stheta*ctheta*F2(MW)

def F2_W(ks):
    return ctheta**2 * F2(MW)+F2(MW)



def F3_P(ks):
    return 2*stheta**2*MW**2*B0(ks,MW,MW)

def F3_Z(ks):
    return 2*stheta**4*MW**2*B0(ks,MW,MW)/ctheta**2

def F3_PZ(ks):
    return -2*stheta**3*MW**2*B0(ks,MW,MW)/ctheta

def F3_W(ks):
    return stheta**4/ctheta**2 * MW**2*B0(ks,MZ,MW) + stheta**2*MW**2 * B0(ks,MGAMMA,MW)



def F4_P(ks):
    return 0

def F4_Z(ks):
    return MZ**2/ctheta**4 * B0(ks,MH,MZ)

def F4_PZ(ks):
    return 0

def F4_W(ks):
    return MW**2*B0(ks,MH,MW)



def F5_P(ks):
    return 0

def F5_Z(ks):
    return 1/ctheta**2 * B22(ks,MZ,MH)

def F5_PZ(ks):
    return 0

def F5_W(ks):
    return B22(ks,MW,MH)



def F6_P(ks):
    return 4*stheta**2*B22(ks,MW,MW)

def F6_Z(ks):
    return (ctheta**2-stheta**2)**2/ctheta**2 *B22(ks,MW,MW)

def F6_PZ(ks):
    return 2*stheta/ctheta *(ctheta**2-stheta**2)*B22(ks,MW,MW)

def F6_W(ks):
    return B22(ks,MZ,MW)



def F7_P(ks):
    return - 2*stheta**2*B22(ks,MW,MW)

def F7_Z(ks):
    return - 2*ctheta**2*B22(ks,MW,MW)

def F7_PZ(ks):
    return - 2*stheta*ctheta*B22(ks,MW,MW)

def F7_W(ks):
    return - 2*ctheta**2*B22(ks,MZ,MW)- 2*stheta**2*B22(ks,MGAMMA,MW)



def F8_P():
    return -2*stheta**2*A(MW)

def F8_Z():
    return -1/(2*ctheta**2)*(A(MZ)+A(MH)) + -1/2 * (stheta**2-ctheta**2)**2/ctheta**2 * A(MW)

def F8_PZ():
    return stheta/ctheta *(stheta**2-ctheta**2)*A(MW)

def F8_W():
    return -1/4 * (A(MZ)+A(MH)+2*A(MW))


def F_PZ(ks):
    return F1_PZ(ks)+F2_PZ(ks)+F3_PZ(ks)+F4_PZ(ks)+F5_PZ(ks)+F6_PZ(ks)+F7_PZ(ks)+F8_PZ()

def F_P(ks):
    return F1_P(ks) + F2_P(ks)+ F3_P(ks)+ F4_P(ks)+ F5_P(ks)+ F6_P(ks)+ F7_P(ks)+ F8_P()

def F_Z(ks):
    return F2_Z(ks)+ F3_Z(ks)+ F4_Z(ks)+ F5_Z(ks)+ F6_Z(ks)+ F7_Z(ks)+ F8_Z()

"""
val1 = -6000
val2 = -5000

print(float(F2_Z(val1).real))
print(float(F2_Z(val2).real))
print(float(F2_Z(val1).real-F2_Z(val2).real))
"""
