# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:27:43 2024

@author: herbi
"""
import gmpy2
import sys
import os
from scipy.special import spence as spence0
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Precision import HPC,behaved_quadratic_routes,convert_gmpy_type
from PVFunctions.BFunctions import B0,B1


infinitessimal = HPC('1e-20')

def spence(x):
    
    x = convert_gmpy_type(x)
    
    return spence0(x)


def theta_fun(x):
    if x.real >= 0:
        return HPC(1)
    return HPC(0)

def get_sign(x):
    if x.real >= 0:
        return +1
    return -1

def eta(a,b):
    pref = 2*np.pi*1j
    #print(theta_fun(-a.imag))
    term1 = theta_fun(-a.imag)*theta_fun(-b.imag)*theta_fun((a*b).imag)
    term2 = theta_fun(a.imag)*theta_fun(b.imag)*theta_fun(-(a*b).imag)
    return pref * (term1-term2)
    

def R(y0,y1):
    
    eta1 = eta(-y1,1/(y0-y1))
    eta2 = eta(1-y1,1/(y0-y1))
    
    #print(eta1,eta2)
    
    term1 = spence(y0/(y0-y1))-spence((y0-1)/(y0-y1))
    
    if eta1 != 0 and eta2 != 0:
        term2 = eta1*gmpy2.log((y0)/(y0-y1))       
        term3 = -eta2*gmpy2.log((y0-1)/(y0-y1))  
        return term1 + term2 + term3
    
    if eta1 == 0 and eta2 != 0:
        return term1 -eta2*gmpy2.log((y0-1)/(y0-y1)) 
    
        
    if eta2 == 0 and eta1 != 0:
        return term1 + eta1*gmpy2.log((y0)/(y0-y1))       
    
    if eta1 == 0 and eta2 == 0:
        return term1
    else:
        
        print("oh no")
        sys.exit()
    

def s3(a,b,c,y0):
    y1,y2 = behaved_quadratic_routes(a,b,c)
    #print(a)
    #print(b)
    #print(c)
    #print(y1,y2)
    #print(a,b,c)
    
    term1 = R(y1,y0)
    #sys.exit()
    term2 = R(y2,y0)
   
    
    epsilon = -infinitessimal*get_sign(c.imag)
    sigma = -infinitessimal*get_sign((a*y0**2+b*y0+c).imag)
    
    inside = (eta(-y1,-y2)-eta(y0-y1,y0-y2)-eta(a-1j*epsilon,1/(a-1j*sigma)))*gmpy2.log((y0-1)/y0)
    
    return term1+term2 + inside

import sys
def C0(p1squared,p2squared,m1,m2,m3,p1dotp2):

    a = -HPC(p2squared)
   
 
    b = -HPC(p1squared)
    c = -HPC(2*p1dotp2)
  
   
    
    d = HPC(m2**2)-HPC(m3**2)-a
    e = HPC(m1**2)-HPC(m2**2)+HPC(p1squared)+HPC(2*p1dotp2)
    f = HPC(m3**2)
  
    a1,b1  =  behaved_quadratic_routes(b, c, a) 
    
    
    alpha = a1
    #y0 = -(d+e*alpha)/(c+2*alpha*b)
    y1 = -(d+e*alpha+2*a+c*alpha)/(c+2*alpha*b)
    y2 = -(d+e*alpha)/((c+2*alpha*b)*(1-alpha))
    y3 = (d+e*alpha)/(c+2*alpha*b)
    
    term1 = s3(b,c+e,a+d+f,y1) * 1/(c+2*alpha*b)
    #sys.exit()
    term2 = -s3(a+b+c,e+d,f,y2) * 1/((c+2*alpha*b))
    term3 = s3(a,d,f,y3) * 1/(c+2*alpha*b)
    
    #print(term1)
    
    #print(term1)
   
    
    return term1+term2+term3

def R1(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    p1dotp2 = HPC(p1dotp2)
    
    f1 = m1**2-m2**2-p1squared
    
    C0_val = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B013 = B0(p1squared,m1,m3)
    B023 = B0(p2squared,m2,m3)
    return 1/2 * (f1*C0_val + B013-B023)

def R2(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    p1dotp2 = HPC(p1dotp2)
    
    f2 = m2**2-m3**2-p2squared - 2*p1dotp2
    
    C0_val = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B012 = B0(p1squared,m1,m2)
    B013 = B0(p1squared,m1,m3)
    return 1/2 * (f2*C0_val + B012-B013)

def C11(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    p1dotp2 = HPC(p1dotp2)
    
    det = p1squared*p2squared - p1dotp2**2
    
    R1_val = R1(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R2_val = R2(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    return (1/det * (p2squared*R1_val-p1dotp2*R2_val)).real
    

def C12(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    p1dotp2 = HPC(p1dotp2)
    
    det = p1squared*p2squared - p1dotp2**2
    
    R1_val = R1(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R2_val = R2(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    return (1/det * (p1squared*R2_val-p1dotp2*R1_val)).real

def C24(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    p1dotp2 = HPC(p1dotp2)
    
    f2 = m2**2-m3**2-p2squared - 2*p1dotp2
    f1 = m1**2-m2**2-p1squared
    
    C0_val = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C11_val = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C12_val = C12(p1squared,p2squared,m1,m2,m3,p1dotp2)    
    B0_23 = B0(p2squared,m2,m3)
    
    return (1/4 - 1/2*m1**2*C0_val + 1/4 * (B0_23-f1*C11_val-f2*C12_val)).real


def R3(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    f1 = m1**2-m2**2-p1squared
     
    C11_val = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C24_val = C24(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    
    B113 = B1(p1squared,m1,m3)
    B023 = B0(p2squared,m2,m3)

    return 1/2*(f1*C11_val+B113+B023)-C24_val


def R4(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    f1 = m1**2-m2**2-p1squared
    
    C12_val = C12(p1squared,p2squared,m1,m2,m3,p1dotp2)
       
    B113 = B1(p1squared,m1,m3)
    B123 = B1(p2squared,m2,m3)

    return 1/2*(f1*C12_val+B113-B123)


def R5(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    
    f2 = m2**2-m3**2-p2squared - 2*p1dotp2  
    C11_val = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)   
    B112 = B1(p1squared,m1,m2)
    B113 = B1(p1squared,m1,m3)
    return 1/2*(f2*C11_val+B112-B113)


def R6(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    
    f2 = m2**2-m3**2-p2squared - 2*p1dotp2
    
    
    C12_val = C12(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C24_val = C24(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    
    B113 = B1(p1squared,m1,m3)

    return 1/2*(f2*C12_val-B113)-C24_val

def C21(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    R3_val = R3(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R5_val = R5(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    det = p1squared*p2squared - p1dotp2**2
    
    return 1/det * (p2squared*R3_val-p1dotp2*R5_val)


def C23(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    R3_val = R3(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R5_val = R5(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    det = p1squared*p2squared - p1dotp2**2
    
    return 1/det * (p1squared*R5_val-p1dotp2*R3_val)

def C22(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    R4_val = R4(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R6_val = R6(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    det = p1squared*p2squared - p1dotp2**2
    
    return 1/det * (p2squared*R4_val-p1dotp2*R6_val)

def C23_alt(p1squared,p2squared,m1,m2,m3,p1dotp2):
    p1squared = HPC(p1squared)
    p2squared = HPC(p2squared)
    m1 = HPC(m1)
    m2 = HPC(m2) 
    m3 = HPC(m3) 
    p1dotp2 = HPC(p1dotp2)
    
    R4_val = R4(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R6_val = R6(p1squared,p2squared,m1,m2,m3,p1dotp2)
    
    det = p1squared*p2squared - p1dotp2**2
    
    return 1/det * (p1squared*R6_val-p1dotp2*R4_val)




"""
def G_func(A, B, C, D, E, y_i):
    
    def interior(x):
        term1 = gmpy2.log(y_i*(C*y_i+D)+E)
        term2 = -gmpy2.log(A*(gmpy2.sqrt(D**2-4*C*E)+D)-2*B*C)
        term3 = -gmpy2.log(A*(gmpy2.sqrt(D**2-4*C*E)-D)+2*B*C)
        term4 = gmpy2.log(C)
        term5 = 2*gmpy2.log(A)
        term6 = -gmpy2.log(HPC(-1/4))
        term7 = spence(-2*C*(A*x+B)/(A*(gmpy2.sqrt(D**2-4*C*E)+D)-2*B*C))
        term8 = spence(2*C*(A*x+B)/(A*(gmpy2.sqrt(D**2-4*C*E)-D)+2*B*C))
        tot = -1/A * ((term1 + term2 + term3 + term4 + term5 + term6)*gmpy2.log(abs(A*x+B)) + term7 + term8)
        return tot
    
    total = interior(1) - interior(0)
    return total.real

import sys
def C0(p1squared,p2squared,m1,m2,m3,p1dotp2):

    a = -HPC(p2squared)
   
 
    b = -HPC(p1squared)
    c = -HPC(2*p1dotp2)
  
   
    
    d = HPC(m2**2)-HPC(m3**2)-a
    e = HPC(m1**2)-HPC(m2**2)+HPC(p1squared)+HPC(2*p1dotp2)
    f = HPC(m3**2)
  
    a1,b1  =  behaved_quadratic_routes(b, c, a) 
    
    #print(b,c,a)
    #print(a,b)
    alpha = b1
    y0 = -(d+e*alpha)/(c+2*alpha*b)
    
    
    y1 = y0+alpha
    y2 = y0/(1-alpha)
    y3 = -y0/alpha

    term1 = G_func(c+2*alpha*b,d+e*alpha+2*a+c*alpha,b,c+e,a+d+f,y1)
    term2 = G_func((c+2*alpha*b)*(1-alpha),d+e*alpha,a+b+c,e+d,f,y2)
    term3 = G_func(-(c+2*alpha*b)*alpha,d+e*alpha,a,d,f,y3)
    #print(term1,(1-alpha)*term2,alpha*term3)
    return (term1 - (1-alpha)*term2 - alpha*term3).real

"""
from Core.Quantities import MZ,ME,MGAMMA

print(C24(-ME**2, -ME**2, MGAMMA*1e-20, MZ, MGAMMA*1e-20, 1e-5))

print(-1/4 * (np.log(MZ**2)-0.5))