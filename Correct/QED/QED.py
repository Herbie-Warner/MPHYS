# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:49:27 2024

@author: herbi
"""
import numpy as np
alpha_0 = 1/137

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Core.Functions import xfunc
from Core.Quantities import ME,MMU,MTOP, MUP,MDOWN,MCHARM,MSTRANGE,MTOP,MBOTTOM,MTAU

def get_new_alpha(s):
    
    def interior(qj,mj,scale):
        if s < 2*abs(mj):
            return 0
        interior = 1 + alpha_0/(3*np.pi) * np.log(s/(mj**2))
        return scale* (qj**2*4*np.pi*alpha_0/s * interior)
    
    
    e = interior(1, ME,1)
    m = interior(1, MMU,1)
    tau = interior(1, MTAU,1)
    
    u = interior(2/3, MUP,3)
    d = interior(1/3, MDOWN,3)
    c = interior(2/3, MCHARM,3)
    s = interior(1/3, MSTRANGE,3)
    t = interior(2/3, MTOP,3)
    b = interior(1/3, MBOTTOM,3)
    
    print(e,m,tau)
    print(u,d,c)
    print(s,t,b)
    
    return 1/137
    return e+m+tau+u+d+c+s+t+b
       
def sigm_0_bakend(s,t):
    alpha = get_new_alpha(s)
    
    term = alpha**2 * np.pi*xfunc(s,t)/s
    return term*(1+3*alpha/(4*np.pi))

def QED(s,t):
    return sigm_0_bakend(s, t)


get_new_alpha(100**2)