# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:16:51 2024

@author: herbi
"""

#CHECK F3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PVFunctions.CFunctions import C0, C11, C12



def D0(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    return 0

def get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared):
  
    a = p1squared
    b = p1dotp2
    c = p1dotp3
    e = p2squared
    f = p2dotp3
    i = p3squared
    determinant = -c**2 * e + 2 * b * c * f - a * f**2 - b**2 * i + a * e * i
    
    inverse_matrix = [
        [(-f**2 + e * i) / determinant, (c * f - b * i) / determinant, (-c * e + b * f) / determinant],
        [(c * f - b * i) / determinant, (-c**2 + a * i) / determinant, (b * c - a * f) / determinant],
        [(-c * e + b * f) / determinant, (b * c - a * f) / determinant, (-b**2 + a * e) / determinant]
    ]   
    return inverse_matrix

def R_20(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f1 = m1**2-m2**2-p1squared
    
    D0V = D0(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C0134 = C0(p1squared,p3squared,m1,m3,m4,p1dotp3)
    C0234 = C0(p2squared,p3squared,m2,m3,m4,p2dotp3)
    return 1/2 * (f1*D0V + C0134 - C0234)
    
def R_21(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    
    D0V = D0(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C0124 = C0(p1squared,p2squared,m1,m2,m4,p1dotp2)
    C0134 = C0(p1squared,p3squared,m1,m3,m4,p1dotp3)
    return 1/2 * (f2*D0V + C0124-C0134)

def R_22(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f3 = m3**2-m4**2+p2squared+2*p2dotp3 #REALLY NOT SURE HERE!?
    
    D0V = D0(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C0123 = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C0124 = C0(p1squared,p2squared,m1,m2,m4,p1dotp2)
    return 1/2 * (f3*D0V + C0123-C0124)

def D11(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R20V = R_20(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R21V = R_21(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R22V = R_22(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[0]  
    return X0[0]*R20V + X0[1]*R21V + X0[2]*R22V

def D12(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R20V = R_20(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R21V = R_21(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R22V = R_22(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[1]  
    return X0[0]*R20V + X0[1]*R21V + X0[2]*R22V

def D13(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R20V = R_20(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R21V = R_21(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R22V = R_22(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[2]  
    return X0[0]*R20V + X0[1]*R21V + X0[2]*R22V

def D27(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f1 = m1**2-m2**2-p1squared
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    f3 = m3**2-m4**2+p2squared+2*p2dotp3
    
    D0V = D0(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    D11V = D11(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    D12V = D12(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    D13V = D13(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C0234 = C0(p2squared,p3squared,m2,m3,m4,p2dotp3)
    return -m1**2*D0V -1/2*(f1*D11V + f2*D12V + f3*D13V - C0234)

def R_30(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f1 = m1**2-m2**2-p1squared
    D11V = D11(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C11_134 = C11(p1squared,p3squared,m1,m3,m4,p1dotp3)
    C0_234 = C0(p2squared,p3squared,m2,m3,m4,p2dotp3)
    D27V = D27(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    return 1/2*(f1*D11V + C11_134+C0_234) - D27V

def R_31(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    D11V = D11(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C11_124 = C11(p1squared,p2squared,m1,m2,m4,p1dotp2)
    C11_134 = C11(p1squared,p3squared,m1,m3,m4,p1dotp3)
    
    return 1/2* (f2*D11V+C11_124-C11_134)

def R_32(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f3 = m3**2-m4**2+p2squared+2*p2dotp3
    D11V = D11(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C11_123 = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C11_124 = C11(p1squared,p2squared,m1,m2,m4,p1dotp2)
    return 1/2*(f3*D11V + C11_123-C11_124)

def R_33(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f1 = m1**2-m2**2-p1squared
    D12V = D12(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C11_134 = C11(p1squared,p3squared,m1,m3,m4,p1dotp3)
    C11_234 = C11(p2squared,p3squared,m2,m3,m4,p2dotp3)
    return 1/2*(f1*D12V + C11_134-C11_234)

def R_34(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    D12V = D12(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C12_124 = C12(p1squared,p2squared,m1,m2,m4,p1dotp2)
    C11_134 = C11(p1squared,p3squared,m1,m3,m4,p1dotp3)
    D27V = D27(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    return 1/2*(f2*D12V + C12_124-C11_134) - D27V


def R_35(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    f3 = m3**2-m4**2+p2squared+2*p2dotp3
    D12V = D12(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    C12_123 = C12(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C12_124 = C12(p1squared,p2squared,m1,m2,m4,p1dotp2)
    return 1/2*(f3*D12V + C12_123-C12_124)

def D21(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R30V = R_30(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R31V = R_31(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R32V = R_32(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[0]  
    return X0[0]*R30V + X0[1]*R31V + X0[2]*R32V

def D24(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R30V = R_30(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R31V = R_31(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R32V = R_32(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[1]  
    return X0[0]*R30V + X0[1]*R31V + X0[2]*R32V

def D25(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R30V = R_30(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R31V = R_31(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R32V = R_32(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[2]  
    return X0[0]*R30V + X0[1]*R31V + X0[2]*R32V


def D24_ALT(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R33V = R_33(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R34V = R_34(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R35V = R_35(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[0]  
    return X0[0]*R33V + X0[1]*R34V + X0[2]*R35V

def D22(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R33V = R_33(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R34V = R_34(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R35V = R_35(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[1]  
    return X0[0]*R33V + X0[1]*R34V + X0[2]*R35V

def D26(p1squared,p2squared,p3squared,p1dotp2,p1dotp3,p2dotp3,m1,m2,m3,m4):
    X = get_X_inverse(p1squared, p1dotp2, p1dotp3, p2squared, p2dotp3, p3squared)
    R33V = R_33(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R34V = R_34(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)
    R35V = R_35(p1squared, p2squared, p3squared, p1dotp2, p1dotp3, p2dotp3, m1, m2, m3, m4)   
    X0 = X[2]  
    return X0[0]*R33V + X0[1]*R34V + X0[2]*R35V