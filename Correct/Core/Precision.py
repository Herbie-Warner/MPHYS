# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:41:33 2024

@author: herbi
"""


from decimal import getcontext
import gmpy2
import numpy as np
getcontext().prec = 50




def HPC(x, precision=200):
    """
    Creates a complex number with arbitrary precision from a single input x.
    
    If x is real, the imaginary part will be set to 0.
    If x is already complex, it will preserve the real and imaginary parts.
    
    Parameters:
        x (int, float, str, or complex): The input, which can be a real or complex number.
        precision (int): The precision (in bits) for the real and imaginary parts.
        
    Returns:
        gmpy2.mpc: The complex number with arbitrary precision.
    """
    # Set the precision context
    gmpy2.get_context().precision = precision
    if isinstance(x, gmpy2.mpc):
        return x
    
    
    
    
   
    
    # If x is a string representation of a complex number
    if isinstance(x, str):
        return gmpy2.mpc(x, precision=precision)
    
    # If x is a complex number, handle real and imaginary parts separately
    if isinstance(x, complex):
        real_part = gmpy2.mpfr(x.real, precision)
        imag_part = gmpy2.mpfr(x.imag, precision)
        return gmpy2.mpc(real_part, imag_part)

    # If x is a real number (either int or float), create a complex number with the real part and imaginary part as 0
    real_part = gmpy2.mpfr(x, precision)
    imag_part = gmpy2.mpfr(0, precision)  # Imaginary part is 0 for real numbers
    return gmpy2.mpc(real_part, imag_part)

smallpar = HPC('1e-10')
threshold_for_taylor = HPC('1e-6')


def convert_gmpy_type(x):
    # Check if x is an instance of gmpy2.mpc (complex number)
    if isinstance(x, gmpy2.mpc):
        # Convert to Python complex number
        py_complex = complex(x.real, x.imag)

        return py_complex

    # Check if x is an instance of gmpy2.mpfr (real number)
    elif isinstance(x, gmpy2.mpfr):
        # Convert to Python float
        py_float = float(x)
        return py_float

    else:
        # If it's neither, print an error


        return None
    
def behaved_quadratic_routes(a_n, b_n, c_n):
    a = HPC(a_n)
    b = HPC(b_n)
    c = HPC(c_n)
    
    
     
    alpha_plus = (-b+gmpy2.sqrt(b**2-4*a*c))/(2*a)
    
    alpha_plus_fin = 0
    alpha_minus_fin = 0
    
    if abs(b**2) > abs(4*a*c):
        condition = abs((4*a*c)/(b**2))< abs(threshold_for_taylor)
        
        alpha_plus = np.where(condition,
                              -c/b, alpha_plus)
        alpha_plus_fin = alpha_plus.item()
    else:
        condition = abs(b**2/(4*a*c)) < abs(threshold_for_taylor)
        
        alpha_plus = np.where(condition,
                              (-b+HPC(1j)*gmpy2.sqrt(4*a*c)*(1-b**2/(8*a*c)))/(2*a),
                              alpha_plus)
        alpha_plus_fin = alpha_plus.item()
        
  
    if abs(a/b) < 1e-10:
        alpha_plus_fin = -c/b
    elif abs(b/a)< 1e-10:
        alpha_plus_fin = gmpy2.sqrt(-c/a)
    else:
        pass
    
    
    alpha_minus = (-b-gmpy2.sqrt(b**2-4*a*c))/(2*a)
    
    if abs(b**2) > abs(4*a*c):
        condition = abs((4*a*c)/(b**2))< abs(threshold_for_taylor)
        
        alpha_minus = np.where(condition,
                              (c/b - b/a), alpha_minus)
        alpha_minus_fin = alpha_minus.item()
    else:
        condition = abs(b**2/(4*a*c)) < abs(threshold_for_taylor)
        
        alpha_minus = np.where(condition,
                              (-b-HPC(1j)*gmpy2.sqrt(4*a*c)*(1-b**2/(8*a*c)))/(2*a),
                              alpha_minus)
        alpha_minus_fin = alpha_minus
        
    if abs(a/b) < 1e-10:
        alpha_minus_fin = -c/b
    elif abs(b/a)< 1e-10:
        alpha_minus_fin = -gmpy2.sqrt(-c/a)
    else:
        pass
    
    if abs(c) == 0:
        return 0,-b/a
    
    
    return alpha_plus_fin,alpha_minus_fin