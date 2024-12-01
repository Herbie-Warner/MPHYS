# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:42:10 2024

@author: herbi
"""

import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Utilities.Utilities import find_quadratic_routes
from scipy.special import spence
from PVFunctions.BFunctions import B0,B1


smallpar = 1e-10
threshold_for_taylor = 1e-10
#USED IN S3, C0
#Approx alpha in C0, C0_IR for numerical (VALID FOR P1.P2 >>> P1^2,P2^2)

def theta_func(x):
    if x >= 0:
        return 1
    return 0

def eta(a,b):
    term1 = theta_func(-a.imag)*theta_func(-b.imag)*theta_func((a*b).imag)
    term2 = theta_func(a.imag)*theta_func(b.imag)*theta_func(-(a*b).imag)
    return 2*np.pi*1j*(term1-term2)

def R(y0,y1):
    spe = spence(y0/(y0-y1)) -spence((y0-1)/(y0-y1))
    et = eta(-y1,(1/(y0-y1)))*np.log(y0/(y0-y1)) - eta(1-y1,1/(y0-y1))*np.log((y0-1)/(y0-y1))
    return spe + et
    
def s3(y0, a,b,c):
    y1,y2 = find_quadratic_routes(a, b, c)
    
    epsilon = smallpar*np.sign(-c.imag) #CAREFUL?
    delta = smallpar*np.sign(-(a*y0**2+b*y0+c).imag) 
    integ = R(y0,y1)+R(y0,y2)
    finite = (eta(-y1,-y2)-eta(y0-y1,y0-y2)-eta(a-1j*epsilon,1/(a-1j*delta)))*np.log((y0-1)/y0)
    return integ - finite

def each_term(a,b,c,d,e,y):
    

    def interior(x):
        # Calculate intermediate values
        discriminant = np.sqrt(b**2 - 4 * a * c)
        term1 = np.log(y * (a * y + b) + c)
        term2 = np.log((discriminant + b) * d - 2 * e * a)
        term3 = np.log((discriminant - b) * d + 2 * e * a)
        term4 = 2 * np.log(d)
        term5 = np.log(a)
        term6 = np.log(-1 / 4 + 0j)
        ln_part = term1 - term2 - term3 + term4 + term5 - term6
    
        abs_dx_plus_e = np.abs(d * x + e)
        ln_abs_dx_plus_e = np.log(abs_dx_plus_e)
    
        li2_arg1 = -2 * a * abs_dx_plus_e / ((discriminant + b) * d - 2 * e * a)
        li2_arg2 = 2 * a * abs_dx_plus_e / ((discriminant - b) * d + 2 * e * a)
    
        dilog_part1 = spence(1 - li2_arg1)
        dilog_part2 = spence(1 - li2_arg2)
    
        result = -((ln_part * ln_abs_dx_plus_e) + dilog_part1 + dilog_part2) / d
        return result


    return (interior(1)-interior(0)).real


def C0_IR(mlep,mprop,sval):
    
    alpha_plusr = -sval/(2*mlep**2) * (1+1-2*mlep**4/sval**2)
    alpha_minsr = -sval/(2*mlep**2) * (2*mlep**4/sval**2)
    
    
    
    alpha = alpha_minsr
    
    C0IRVAL = -1/2 * alpha/((alpha**2-1)*mlep**2) * np.log(alpha**2)*np.log(mprop**2)
    return C0IRVAL


def C0(p1squared,p2squared,m1,m2,m3,p1dotp2):
    a = -p2squared
    b = -p1squared
    c = -2*p1dotp2
    d = m2**2-m3**2+p2squared
    e = m1**2-m2**2+p1squared + 2*p1dotp2
    f = m3**2
    
    cpluse = m1**2-m2**2 + p1squared
    if p1squared == -m1**2:
        cpluse = -m2**2
  
    
   # Compute the common term
    sqrt_term = np.sqrt(1 - p1squared * p2squared / (p1dotp2**2))
    
    # Compute general_alpha_plus and general_alpha_minus without conditions
    general_alpha_plus = p1dotp2 / p1squared * (-1 + sqrt_term)
    general_alpha_minus = p1dotp2 / p1squared * (-1 - sqrt_term)
    
    # Apply the condition for the threshold
    condition = abs(p1squared * p2squared / (p1dotp2**2)) < threshold_for_taylor
    
    # Update general_alpha_plus where the condition is True
    general_alpha_plus = np.where(
        condition,
        p1dotp2 / p1squared * (-0.5 * p1squared * p2squared / (p1dotp2**2)),
        general_alpha_plus
    )
    alpha = general_alpha_plus
    
    y0 = -(d+e*alpha)/(c+2*alpha*b)
    y1 = y0+alpha
    y2 = y0/(1-alpha)
    y3 = -y0/alpha
    
    term1 = each_term(b, cpluse, a+d+f, c+2*alpha*b, d+e*alpha+2*a+c*alpha, y1)
    term2 = each_term(a+b+c, e+d, f, (c+2*alpha*b)*(1-alpha), d+e*alpha, y2)
    term3 = each_term(a, d,f, -(c+2*alpha*b), d+e*alpha, y3)
    
    return term1 - (1-alpha)*term2 - alpha*term3

def C0_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    a = -p2squared
    b = -p1squared
    c = -2*p1dotp2
    d = m2**2-m3**2+p2squared
    e = m1**2-m2**2+p1squared + 2*p1dotp2
    f = m3**2
    
    cpluse = m1**2-m2**2 + p1squared
    if p1squared == -m1**2:
        cpluse = -m2**2
  
    
   
    # Compute the common term
    sqrt_term = np.sqrt(1 - p1squared * p2squared / (p1dotp2**2))
    
    # Compute general_alpha_plus and general_alpha_minus without conditions
    general_alpha_plus = p1dotp2 / p1squared * (-1 + sqrt_term)
    general_alpha_minus = p1dotp2 / p1squared * (-1 - sqrt_term)
    
    # Apply the condition for the threshold
    condition = abs(p1squared * p2squared / (p1dotp2**2)) < threshold_for_taylor
    
    # Update general_alpha_plus where the condition is True
    general_alpha_plus = np.where(
        condition,
        p1dotp2 / p1squared * (-0.5 * p1squared * p2squared / (p1dotp2**2)),
        general_alpha_plus
    )
    
    # general_alpha_minus remains unchanged, as there’s no condition affecting it

    alpha = general_alpha_plus
    
    y0 = -(d+e*alpha)/(c+2*alpha*b)
    y1 = y0+alpha
    y2 = y0/(1-alpha)
    y3 = -y0/alpha
    
    term1 = each_term(b, cpluse, a+d+f, c+2*alpha*b, d+e*alpha+2*a+c*alpha, y1)
    term2 = each_term(a+b+c, e+d, f, (c+2*alpha*b)*(1-alpha), d+e*alpha, y2)
    term3 = each_term(a, d,f, -(c+2*alpha*b), d+e*alpha, y3)
    
    full = term1 - (1-alpha)*term2 - alpha*term3
    IR = C0_IR(m1, m2, -2*p1dotp2)
    
    return full-IR


    
def R1(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f1 = m1**2 -m2**2-p1squared
    C0VAL = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B013 = B0(p1squared, m1, m3)
    B023 = B0(p2squared, m2, m3)
    return 1/2 * (f1*C0VAL + B013 - B023)

def R2(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C0VAL = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B012 = B0(p1squared, m1, m2)
    B013 = B0(p1squared, m1, m3)
    return 1/2 * (f2*C0VAL + B012 - B013)

def C11(p1squared,p2squared,m1,m2,m3,p1dotp2):
    R1VAL = R1(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R2VAL = R2(p1squared,p2squared,m1,m2,m3,p1dotp2)  
    det = p1squared*p2squared - p1dotp2**2
    return 1/det * (p2squared*R1VAL-p1dotp2*R2VAL)

def C12(p1squared,p2squared,m1,m2,m3,p1dotp2):
    R1VAL = R1(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R2VAL = R2(p1squared,p2squared,m1,m2,m3,p1dotp2)   
    det = p1squared*p2squared - p1dotp2**2
    return 1/det * (p1squared*R2VAL-p1dotp2*R1VAL)
    
def C24(p1squared,p2squared,m1,m2,m3,p1dotp2):
    C0VAL = C0(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B023 = B0(p2squared,m2,m3)
    f1 = m1**2 -m2**2-p1squared
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C11VAL = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C12VAL = C12(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/4 -1/2*m1**2*C0VAL + 1/4*(B023-f1*C11VAL-f2*C12VAL)

def R3(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f1 = m1**2 -m2**2-p1squared
    C11VAL = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B113 = B1(p1squared,m1,m3)
    B023 = B0(p2squared,m2,m3)
    C24VAL = C24(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/2 * (f1*C11VAL+B113+B023) - C24VAL

def R4(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f1 = m1**2 -m2**2-p1squared
    C12VAL = C12(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B113 = B1(p1squared,m1,m3)
    B123 = B1(p2squared,m2,m3)
    return 1/2 * (f1*C12VAL+B113-B123) 

def R5(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C11VAL = C11(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B112 = B1(p1squared,m1,m2)
    B113 = B1(p2squared,m1,m3)
    return 1/2 * (f2*C11VAL+B112-B113)

def R6(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C12VAL = C12(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B113 = B1(p1squared,m1,m3)
    C24VAL = C24(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/2 * (f2*C12VAL-B113) - C24VAL

def C21(p1squared,p2squared,m1,m2,m3,p1dotp2):
    det = p1squared*p2squared-p1dotp2**2
    R3VAL = R3(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R5VAL = R5(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/det * (p2squared*R3VAL-p1dotp2*R5VAL)

def C23(p1squared,p2squared,m1,m2,m3,p1dotp2):
    det = p1squared*p2squared-p1dotp2**2
    R3VAL = R3(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R5VAL = R5(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/det * (p1squared*R5VAL-p1dotp2*R3VAL)

def C22(p1squared,p2squared,m1,m2,m3,p1dotp2):
    det = p1squared*p2squared-p1dotp2**2
    R4VAL = R4(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R6VAL = R6(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/det * (p2squared*R4VAL-p1dotp2*R6VAL)



def R1_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f1 = m1**2 -m2**2-p1squared
    C0VAL = C0_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B013 = B0(p1squared, m1, m3)
    B023 = B0(p2squared, m2, m3)
    return 1/2 * (f1*C0VAL + B013 - B023)

def R2_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C0VAL = C0_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B012 = B0(p1squared, m1, m2)
    B013 = B0(p1squared, m1, m3)
    return 1/2 * (f2*C0VAL + B012 - B013)

def C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    R1VAL = R1_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R2VAL = R2_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)  
    det = p1squared*p2squared - p1dotp2**2
    return 1/det * (p2squared*R1VAL-p1dotp2*R2VAL)

def C12_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    R1VAL = R1_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R2VAL = R2_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)   
    det = p1squared*p2squared - p1dotp2**2
    return 1/det * (p1squared*R2VAL-p1dotp2*R1VAL)
    
def C24_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    C0VAL = C0_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B023 = B0(p2squared,m2,m3)
    f1 = m1**2 -m2**2-p1squared
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C11VAL = C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    C12VAL = C12_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/4 -1/2*m1**2*C0VAL + 1/4*(B023-f1*C11VAL-f2*C12VAL)

def R3_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f1 = m1**2 -m2**2-p1squared
    C11VAL = C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B113 = B1(p1squared,m1,m3)
    B023 = B0(p2squared,m2,m3)
    C24VAL = C24_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/2 * (f1*C11VAL+B113+B023) - C24VAL

def R4_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f1 = m1**2 -m2**2-p1squared
    C12VAL = C12_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B113 = B1(p1squared,m1,m3)
    B123 = B1(p2squared,m2,m3)
    return 1/2 * (f1*C12VAL+B113-B123) 

def R5_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C11VAL = C11_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B112 = B1(p1squared,m1,m2)
    B113 = B1(p2squared,m1,m3)
    return 1/2 * (f2*C11VAL+B112-B113)

def R6_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    f2 = m2**2-m3**2-p2squared-2*p1dotp2
    C12VAL = C12_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    B113 = B1(p1squared,m1,m3)
    C24VAL = C24_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/2 * (f2*C12VAL-B113) - C24VAL

def C21_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    det = p1squared*p2squared-p1dotp2**2
    R3VAL = R3_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R5VAL = R5_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/det * (p2squared*R3VAL-p1dotp2*R5VAL)

def C23_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    det = p1squared*p2squared-p1dotp2**2
    R3VAL = R3_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R5VAL = R5_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/det * (p1squared*R5VAL-p1dotp2*R3VAL)

def C22_reg(p1squared,p2squared,m1,m2,m3,p1dotp2):
    det = p1squared*p2squared-p1dotp2**2
    R4VAL = R4_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    R6VAL = R6_reg(p1squared,p2squared,m1,m2,m3,p1dotp2)
    return 1/det * (p2squared*R4VAL-p1dotp2*R6VAL)







