# -*- coding: utf-8 -*-
import gmpy2
import sys
import os
from scipy.special import spence as spence0

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Precision import HPC,convert_gmpy_type
from PVFunctions.CFunctions import C0,C11,C12

import numpy as np




# Replacing the small constants with HPC
smallpar = HPC('1e-10')
threshold_for_taylor = HPC('1e-6')


def spence(x):
    
    x = convert_gmpy_type(x)
    
    return spence0(x)
    
    

def theta_fun(x):
    if x.real >= 0:
        return HPC(1)
    return HPC(0)

def epsilon(x):
    if x.real >= 0:
        return HPC(1)
    return HPC(-1)

def behaved_alpha_plus(a, b, c):
    alpha_plus = (-b+gmpy2.sqrt(b**2-4*a*c))/(2*a)
    
    if abs(b**2) > abs(4*a*c):
        condition = abs((4*a*c)/(b**2))< abs(threshold_for_taylor)
        
        alpha_plus = np.where(condition,
                              -c/b, alpha_plus)
        alpha = alpha_plus.item()
    else:
        condition = abs(b**2/(4*a*c)) < abs(threshold_for_taylor)
        
        alpha_plus = np.where(condition,
                              (-b+HPC(1j)*gmpy2.sqrt(4*a*c)*(1-b**2/(8*a*c)))/(2*a),
                              alpha_plus)
        alpha = alpha_plus.item()
        
    if abs(a/b) < 1e-10:
        alpha = -c/b
    elif abs(b/a)< 1e-10:
        alpha = gmpy2.sqrt(-c/a)
    else:
        pass
    
    return alpha

def behaved_alpha_minus(a, b, c):
    alpha_minus = (-b-gmpy2.sqrt(b**2-4*a*c))/(2*a)
    
    if abs(b**2) > abs(4*a*c):
        condition = abs((4*a*c)/(b**2))< abs(threshold_for_taylor)
        
        alpha_minus = np.where(condition,
                              (c/b - b/a), alpha_minus)
        alpha = alpha_minus.item()
    else:
        condition = abs(b**2/(4*a*c)) < abs(threshold_for_taylor)
        
        alpha_minus = np.where(condition,
                              (-b-HPC(1j)*gmpy2.sqrt(4*a*c)*(1-b**2/(8*a*c)))/(2*a),
                              alpha_minus)
        alpha = alpha_minus
        
    if abs(a/b) < 1e-10:
        alpha = -c/b
    elif abs(b/a)< 1e-10:
        alpha = -gmpy2.sqrt(-c/a)
    else:
        pass
    
    return alpha

def R_func(y_k, y_plus, y_minus, A_i, A_j, c, alpha, b):
    R = -((theta_fun(-A_i*A_j))/(c + HPC(2) * alpha * b)) * (HPC(np.pi)**2 +
        HPC(1j)*HPC(np.pi) * theta_fun(y_k.imag) * (HPC(2) * gmpy2.log(y_k - y_minus) - gmpy2.log((y_k - y_minus) * (y_k - y_plus))) -
        HPC(1j) * HPC(np.pi) * theta_fun(-y_k.imag) * (HPC(2) * gmpy2.log(y_k - y_plus) -
        gmpy2.log((y_k - y_minus) * (y_k - y_plus))))
    
    return R

def S_fun(A_1, A_2, A_3, A_4, l_12, m_1, S_34, S_24, S_23):
    S = (epsilon(A_1 / A_2 - l_12 / (HPC(2) * m_1**2)) * (theta_fun(A_3 * A_4) * S_34
        - theta_fun(-A_3 * A_4) * (S_24 + S_23)))
    
    return S

def S_ij(q_ij_squared, x_1, x_2):
    x_1 = HPC(1)
    x_2 = HPC(1)
    
    res = (-1j * HPC(np.pi)) / (q_ij_squared * (x_1 - x_2) + HPC(1j)*smallpar) * (gmpy2.log((x_1 - HPC(1)) / x_1 + HPC(1j)*smallpar) - gmpy2.log((x_2 - HPC(1)) / x_2 + HPC(1j)*smallpar))
    return res

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

def C0_IR_box(mlep1, mlep2, mprop, sval):
    
    a = mlep2**2 
    b = sval
    c = mlep1**2
    
    #alpha = behaved_alpha_plus(a, b, c)
    alpha = behaved_alpha_minus(a, b, c)
    
    C0IRVAL = gmpy2.log((mlep1*mlep2)/mprop**2) * HPC(1) / HPC(2) * (alpha / (alpha**2*mlep2**2 - mlep1**2)) * gmpy2.log((mlep2/mlep1)**2*alpha**2)
    return C0IRVAL

def y_plus_ij(l_ij, A_i, A_j, m_i, m_j):
    a = -l_ij * A_i * A_j + m_i**2 * A_i**2 + m_j**2 * A_j**2
    b = l_ij * A_i * A_j - HPC(2) * m_j**2 * A_j**2
    c = m_j**2 * A_j**2 - HPC(1j)*smallpar
    
    y = behaved_alpha_plus(a, b, c)
    
    #y = (-b + gmpy2.sqrt(b**2 - HPC(4) * a * c)) / (HPC(2) * a)
    return y

def y_minus_ij(l_ij, A_i, A_j, m_i, m_j):
    a = -l_ij * A_i * A_j + m_i**2 * A_i**2 + m_j**2 * A_j**2
    b = l_ij * A_i * A_j - HPC(2) * m_j**2 * A_j**2
    c = m_j**2 * A_j**2
    
    y = behaved_alpha_minus(a, b, c)
    
    #y = (-b - gmpy2.sqrt(b**2 - HPC(4) * a * c)) / (HPC(2) * a)
    return y

def D0(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):

    l_12 = p1_squared + m1**2 + m2**2
    
    l_13 = p1_squared + p2_squared + HPC(2) * p1p2 + m1**2 + m3**2

    l_14 = p1_squared + p2_squared + p3_squared + HPC(2) * p1p2 + HPC(2) * p1p3 + HPC(2) * p2p3 + m1**2 + m4**2

    l_23 = p2_squared + m2**2 + m3**2

    l_24 = p2_squared + p3_squared + HPC(2) * p2p3 + m2**2 + m4**2

    l_34 = p3_squared + m3**2 + m4**2

    
    beta = behaved_alpha_plus(-p1_squared, (p1_squared+m2**2-m1**2), m1**2)
  
    A_1 = HPC(1)/(p1_squared*beta**2 + m1**2)
    A_2 = HPC(1)/(p1_squared*(1-beta)**2 + m2**2)
   
    A_3 = HPC(1)/((HPC(1)-beta)*l_13 + beta*l_23)
    A_4 = HPC(1)/((HPC(1)-beta)*l_14+beta*l_24)
    
    q_12_squared = l_12*A_1*A_2 - m1**2*A_1**2 - m2**2*A_2**2 
    q_13_squared = l_13*A_1*A_3 - m1**2*A_1**2 - m3**2*A_3**2 
    q_14_squared = l_14*A_1*A_4 - m1**2*A_1**2 - m4**2*A_4**2 
    q_34_squared = l_34 * A_3 * A_4 - m3**2 * A_3**2 - m4**2 * A_4**2
    q_24_squared = l_24 * A_2 * A_4 - m2**2 * A_2**2 - m4**2 * A_4**2
    q_23_squared = l_23 * A_2 * A_3 - m2**2 * A_2**2 - m3**2 * A_3**2
    
    a = -q_34_squared 
    b = -q_23_squared
    c = -q_24_squared + q_23_squared + q_34_squared
    d = A_3**2*m3**2 - A_4**2*m4**2 + q_34_squared
    e = A_2**2*m2**2 - A_3**2*m3**2 + q_24_squared - q_34_squared
   
    k = A_1**2*m1**2 - A_2**2*m2**2 + q_14_squared - q_24_squared
    
  
    alpha = behaved_alpha_minus(b, c, a)
    
    y_1 = -(d + e * alpha) / ((c + HPC(2) * alpha * b) * (HPC(1) - alpha))
    y_2 = (d + e * alpha) / ((c + HPC(2) * alpha * b) * alpha)
    y_3 = -(d + e * alpha + c * alpha + HPC(2) * a) / (c + HPC(2) * alpha * b)
    
    y_4 = -(d + (e + k) * alpha) / ((c + HPC(2) * alpha * b) * (HPC(1) - alpha))
    y_5 = (d + (e + k) * alpha) / (alpha * (c + HPC(2) * alpha * b))
    y_6 = -(d + (e + k) * alpha + c * alpha + HPC(2) * a) / (c + HPC(2) * alpha * b)

    y_plus_1 = y_plus_ij(l_24, A_2, A_4, m2, m4)
    y_minus_1 = y_minus_ij(l_24, A_2, A_4, m2, m4)
    
    y_plus_2 = y_plus_ij(l_34, A_3, A_4, m3, m4)
    y_minus_2 = y_minus_ij(l_34, A_3, A_4, m3, m4)
    
    y_plus_3 = y_plus_ij(l_23, A_2, A_3, m2, m3)
    y_minus_3 = y_minus_ij(l_23, A_2, A_3, m2, m3)
    
    y_plus_4 = y_plus_ij(l_14, A_1, A_4, m1, m4)
    y_minus_4 = y_minus_ij(l_14, A_1, A_4, m1, m4)
    
    y_plus_5 = y_plus_ij(l_34, A_3, A_4, m3, m4)
    y_minus_5 = y_minus_ij(l_34, A_3, A_4, m3, m4)
    
    y_plus_6 = y_plus_ij(l_13, A_1, A_3, m1, m3)
    y_minus_6 = y_minus_ij(l_13, A_1, A_3, m1, m3)

    x_1 = behaved_alpha_plus(-q_34_squared, q_34_squared+q_24_squared-q_23_squared, -q_24_squared-1j*smallpar)
    x_2 = behaved_alpha_minus(-q_34_squared, q_34_squared+q_24_squared-q_23_squared, -q_24_squared-1j*smallpar)
  
    term1 = (-(1 - alpha) * G_func((c + HPC(2) * alpha * b) * (HPC(1) - alpha), d + e * alpha,
                                   -l_24 * A_2 * A_4 + m2**2 * A_2**2 + m4**2 * A_4**2,
                                   l_24 * A_2 * A_4 - HPC(2) * m4**2 * A_4**2, m4**2 * A_4**2, y_1) -
             R_func(y_1, y_plus_1, y_minus_1, A_2, A_4, c, alpha, b))


    term2 = (-(alpha) * G_func(-(c + 2 * alpha * b) * (alpha), d + e * alpha,
                               -l_34 * A_3 * A_4 + m3**2 * A_3**2 + m4**2 * A_4**2,
                               l_34 * A_3 * A_4 - 2 * m4**2 * A_4**2,
                               m4**2 * A_4**2, y_2) +
             R_func(y_2, y_plus_2, y_minus_2, A_3, A_4, c, alpha, b))


    term3 = (G_func((c + 2 * alpha * b), d + e * alpha + c * alpha + 2 * a,
                    -l_23 * A_2 * A_3 + m2**2 * A_2**2 + m3**2 * A_3**2,
                    l_23 * A_2 * A_3 - 2 * m3**2 * A_3**2,
                    m3**2 * A_3**2, y_3) +
             R_func(y_3, y_plus_3, y_minus_3, A_2, A_3, c, alpha, b))


    term4 = ((1 - alpha) * G_func((c + 2 * alpha * b) * (1 - alpha), d + (e + k) * alpha,
                                  -l_14 * A_1 * A_4 + m1**2 * A_1**2 + m4**2 * A_4**2,
                                  l_14 * A_1 * A_4 - 2 * m4**2 * A_4**2,
                                  m4**2 * A_4**2, y_4) +
             R_func(y_4, y_plus_4, y_minus_4, A_1, A_4, c, alpha, b))

    term5 = ((alpha) * G_func(-(c + 2 * alpha * b) * (alpha), d + (e + k) * alpha,
                              -l_34 * A_3 * A_4 + m3**2 * A_3**2 + m4**2 * A_4**2,
                              l_34 * A_3 * A_4 - 2 * m4**2 * A_4**2,
                              m4**2 * A_4**2, y_5) -
             R_func(y_5, y_plus_5, y_minus_5, A_3, A_4, c, alpha, b))

    term6 = (-G_func((c + 2 * alpha * b), d + (e + k) * alpha + c * alpha + 2 * a,
                     -l_13 * A_1 * A_3 + m1**2 * A_1**2 + m3**2 * A_3**2,
                     l_13 * A_1 * A_3 - 2 * m3**2 * A_3**2,
                     m3**2 * A_3, y_6) -
             R_func(y_6, y_plus_6, y_minus_6, A_1, A_3, c, alpha, b))

    term7 = theta_fun(-A_1 * A_2) * S_fun(A_1, A_2, A_3, A_4, l_12, m1,
                                           S_ij(q_34_squared, x_1, x_2), S_ij(q_24_squared, x_1, x_2), S_ij(q_23_squared, x_1, x_2))


    total = ((A_1 * A_2 * A_3 * A_4) / k) * (term1 + term2 + term3 + term4 + term5 + term6 + term7)

    return convert_gmpy_type(total)

def D0reg(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):

    D0_tot = D0(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    D0_IR = HPC('1') / (HPC('2') * p2p3) * C0_IR_box(m1, m3, m2, -2*p1p2)

    #print(D0_tot)
    #print(D0_IR.real)
    #print((D0_tot - D0_IR).real)

    return HPC('0')

def get_row_1(a,b,c,d,e,f):
    det = -c**2*d + 2*b*c*e-a*e**2-b**2*f+a*d*f
    row1 = [(d*f-e**2)/det,(c*e-b*f)/det,(b*e-c*d)/det]
    return row1

def get_row_2(a,b,c,d,e,f):
    det = -c**2*d + 2*b*c*e-a*e**2-b**2*f+a*d*f
    row2 = [(c*e-b*f)/det,(a*f-c**2)/det,(b*c-a*e)/det]
    return row2

def get_row_3(a,b,c,d,e,f):
    det = -c**2*d + 2*b*c*e-a*e**2-b**2*f+a*d*f
    row3 = [(b*e-c*d)/det,(b*c-a*e)/det,(a*d-b**2)/det]

    return row3


def R20(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    #print(p1p2)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
    
    f1 = m1**2-m2**2-p1_squared
  
    D0val = D0(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    C0_134 = C0(p1_squared,p3_squared,m1,m3,m4,p1p3)
    C0_234 = C0(p2_squared,p3_squared,m2,m3,m4,p2p3)
    return 1/2 * (f1*D0val + C0_134-C0_234)


def R21(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)

    f2 = m2**2-m3**2-p2_squared-2*p1p2
   
    D0val = D0(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    C0_124 = C0(p1_squared,p2_squared,m1,m2,m4,p1p2)
    C0_134 = C0(p1_squared,p3_squared,m1,m3,m4,p1p3)
    return 1/2 * (f2*D0val + C0_124-C0_134)


def R22(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
    
    f3 = m3**2-m4**2 -p3_squared + p2_squared+p3_squared+2*p2p3
    
    D0val = D0(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    C0_123 = C0(p1_squared,p2_squared,m1,m2,m3,p1p2)
    C0_124 = C0(p1_squared,p2_squared,m1,m2,m4,p1p2)
    return 1/2 * (f3*D0val + C0_123-C0_124)

def D11(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_1(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R20(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R21(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R22(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val

def D12(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_2(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R20(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R21(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R22(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val


def D13(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_3(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R20(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R21(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R22(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val

def D27(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
    
    f1 = m1**2-m2**2-p1_squared
    f2 = m2**2-m3**2-p2_squared-2*p1p2
    f3 = m3**2-m4**2 -p3_squared + p2_squared+p3_squared+2*p2p3
    
    D0_val = D0(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    D11_val = D11(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    D13_val = D13(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    D12_val = D12(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)  
    C0_234 = C0(p2_squared,p3_squared,m2,m3,m4,p2p3)  
    return -m1**2*D0_val -1/2 * (f1*D11_val+f2*D12_val+f3*D13_val-C0_234)


def R30(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
    
    f1 = m1**2-m2**2-p1_squared
  
    D11val = D11(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    D27val = D27(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    C11_134 = C11(p1_squared,p3_squared,m1,m3,m4,p1p3)
    C0_234 = C0(p2_squared,p3_squared,m2,m3,m4,p2p3)
    return 1/2 * (f1*D11val + C11_134 + C0_234)-D27val

def R33(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
    
    f1 = m1**2-m2**2-p1_squared
   
    D12val = D12(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    
    C11_134 = C11(p1_squared,p3_squared,m1,m3,m4,p1p3)
    C11_234 = C11(p2_squared,p3_squared,m2,m3,m4,p2p3)
    return 1/2 * (f1*D12val + C11_134 - C11_234)


def R36(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
    
    f1 = m1**2-m2**2-p1_squared
    
    
    D13val = D13(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    
    C12_134 = C12(p1_squared,p3_squared,m1,m3,m4,p1p3)
    C12_234 = C12(p2_squared,p3_squared,m2,m3,m4,p2p3)
    return 1/2 * (f1*D13val+C12_134-C12_234)


def R31(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
    f2 = m2**2-m3**2-p2_squared-2*p1p2
   
    D11val = D11(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    
    C11_124 = C11(p1_squared,p2_squared,m1,m2,m4,p1p2)
    C11_134 = C11(p1_squared,p3_squared,m1,m3,m4,p1p3)
    return 1/2 * (f2*D11val+C11_124-C11_134)

def R34(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    
 
    f2 = m2**2-m3**2-p2_squared-2*p1p2
 
    D12val = D12(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    D27val = D27(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    
    
    C12_124 = C12(p1_squared,p2_squared,m1,m2,m4,p1p2)
    C11_134 = C11(p1_squared,p3_squared,m1,m3,m4,p1p3)
    return 1/2 * (f2*D12val+C12_124-C11_134)-D27val


def R32(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    

    f3 = m3**2-m4**2 -p3_squared + p2_squared+p3_squared+2*p2p3
    
    D11val = D11(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    
    C11_123 = C11(p1_squared,p2_squared,m1,m2,m3,p1p2)
    C11_124 = C11(p1_squared,p2_squared,m1,m2,m4,p1p2)
    return 1/2 * (f3*D11val+C11_123-C11_124)

def R35(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    p1_squared= HPC(p1_squared)
    p2_squared= HPC(p2_squared)
    p3_squared= HPC(p3_squared)
    p1p2= HPC(p1p2)
    p1p3= HPC(p1p3)
    p2p3= HPC(p2p3)
    m1 = HPC(m1)
    m2 = HPC(m2)
    m3 = HPC(m3)
    m4 = HPC(m4)
    

    f3 = m3**2-m4**2 -p3_squared + p2_squared+p3_squared+2*p2p3
    
    D12val = D12(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)   
    
    C12_123 = C12(p1_squared,p2_squared,m1,m2,m3,p1p2)
    C12_124 = C12(p1_squared,p2_squared,m1,m2,m4,p1p2)
    return 1/2 * (f3*D12val + C12_123-C12_124)

def D21(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_1(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R30(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R31(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R32(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val

def D24_alt(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_2(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R30(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R31(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R32(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val


def D25(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_3(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R30(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R31(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R32(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val

def D24(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_1(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R33(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R34(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R35(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val


def D22(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_2(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R33(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R34(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R35(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val


def D26(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4):
    row1 = get_row_3(p1_squared, p1p2,p1p3, p2_squared, p2p3, p3_squared)
    
    R20_val = R33(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R21_val = R34(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    R22_val = R35(p1_squared, p2_squared, p3_squared, p1p2, p1p3, p2p3, m1, m2, m3, m4)
    
    return row1[0]*R20_val + row1[1]*R21_val + row1[2]*R22_val


#print(D24_alt(-304,-34,-324,-20,-40,-30,450,234,12,345))
from PVFunctions.CFunctions import C0


from Core.Quantities import ME,MZ,MGAMMA,MMU,MW


p1s = -ME**2
p2s = -ME**2
p3s = -MMU**2
m1 = ME
m2 = MZ
m3 = MMU

m4 = MZ

ecm = 10000
s = ecm**2

p1dotp2 = -s/2

theta = np.linspace(0,2*np.pi,100)

t = s/2 * (np.cos(theta)-1)

p2dotp4 = (s+t)/2

p1dotp3 = -t/2

#print(C0(p1s, p2s, m1, m2, m3, p1dotp3))

import sys
#sys.exit()
C0_vals = []
i = 0
for tval in t:
    p1dotp3 = -tval/2
    C0_vals.append(float(C0(p1s, p2s, m1, m2, m3, p1dotp3).real))
    i += 1
    print(i)

#print(C0(p1s, p2s, m1, m2, m3, p1dotp3))
#print(float(D24(p1s,p2s,p3s,p1dotp2,p1dotp3,p2dotp4,m1,m2,m3,m4).real))
import matplotlib.pyplot as plt
figure = plt.figure()
ax = figure.add_subplot()


ax.plot(theta,C0_vals)
