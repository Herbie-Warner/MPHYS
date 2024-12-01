# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:55:25 2024

@author: herbi
"""
import numpy as np
from scipy.special import spence

threshold_for_taylor = 1e-10
ininitessimal = 1e-20

def theta_func(x):
    if x >= 0:
        return 1
    return 0

def eta(a,b):
    term1 = theta_func(-a.imag)*theta_func(-b.imag)*theta_func((a*b).imag)
    term2 = theta_func(a.imag)*theta_func(b.imag)*theta_func(-(a*b).imag)
    return 0
    return 2*np.pi*1j*(term1-term2)


def R(y0,y1):
    spe = spence(y0/(y0-y1))-spence((y0-1)/(y0-y1))
    et = eta(-y1,1/(y0-y1))*np.log(y0/(y0-y1)) - eta(1-y1,1/(y0-y1))*np.log((y0-1)/(y0-y1))
    return spe + et

def S3(y0,a,b,c):
    epsilon = ininitessimal * -np.sign(c.imag)
    delta = ininitessimal * -np.sign((a*y0**2+b*y0+c).imag)
    
    #print(a,b,c)
    
    y1 = b/(2*a) * (-1+np.sqrt(1-4*a*c/b**2))
    y2 = b/(2*a) * (-1-np.sqrt(1-4*a*c/b**2))
    
    
    if (a*c/b**2).real < threshold_for_taylor:
        y1 = b/(2*a) * (-2*a*c/b**2)
        
    finite = -(eta(-y1,-y2)-eta(y0-y1,y0-y2)-eta(a-1j*epsilon,1/(a-1j*delta)))*np.log((y0-1)/(y0))
    finite = 0
    return R(y0,y1)+R(y0,y2) + finite
    
    
    
    

def C0_NEW(p1squared,p2squared,m1,m2,m3,p1dotp2):
    a = -p2squared
    b = -p1squared
    c = -2*p1dotp2
    d = m2**2-m3**2+p2squared
    e = m1**2-m2**2+p1squared + 2*p1dotp2
    f = m3**2
    
    cpluse = m1**2-m2**2 + p1squared
    if p1squared == -m1**2:
        cpluse = -m2**2
  
    
    sqrt_term = np.sqrt(1-p1squared*p2squared/(p1dotp2**2))
    general_alpha_plus = p1dotp2/p1squared * (-1+sqrt_term)
    general_alpha_minus = p1dotp2/p1squared * (-1-sqrt_term)
    
    if abs(p1squared*p2squared/(p1dotp2**2)) < threshold_for_taylor:
        general_alpha_plus = p1dotp2/p1squared * (- 0.5 * p1squared*p2squared/(p1dotp2**2))
    
    alpha = general_alpha_plus
    
    y0 = -(d+e*alpha)/(c+2*alpha*b)
    y1 = y0+alpha
    y2 = y0/(1-alpha)
    y3 = -y0/alpha
  
    term1 = S3(y1,b,cpluse,a+d+f)
    term2 = -(1-alpha)*S3(y2,a+b+c,e+d,f)
    term3 = -alpha*S3(y3,a,d,f)
    return term1+term2+term3
   
    
    
from Utilities import ME,MGAMMA,MZ



from CFunctions import C0

sval = 100000
print(C0(-ME**2, -ME**2, ME, MGAMMA, ME, -sval/2))


alpha_plusr = -sval/(2*ME**2) * (1+1-2*ME**4/sval**2)
alpha_minsr = -sval/(2*ME**2) * (2*ME**4/sval**2)



alpha = alpha_minsr

C0IRVAL = -1/2 * alpha/((alpha**2-1)*ME**2) * np.log(alpha**2)*np.log(MGAMMA**2)
print(C0IRVAL)