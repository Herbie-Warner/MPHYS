# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:41:02 2024

@author: herbi
"""

import numpy as np

epsilon = 1e-5 + 0j

THETAW = 13.6*np.pi/180
stheta = np.sin(THETAW)
ctheta = np.cos(THETAW)
vtheta = 4*stheta**2 -1

MZ = 91 + 0j
MW = 80 + 0j

#For regularisation
ME = 0.511*(1e-3) + 0j
MMU = 105*(1e-3) + 0j
MGAMMA = epsilon
MNU = epsilon


g = 0.65

GAMMA =  0.577216


dimension = 4 - epsilon
DELTA =  GAMMA - np.log(np.pi) #-2/(dimension-4)


Gev_minus_2_to_mbarns = 0.3894*1e9


def xfunc(s,t):
    return 1 + 2*(t/s + t**2/s**2)

def find_quadratic_routes(a,b,c):
    det = np.sqrt(b**2-4*a*c)
    return (-b+det)/(2*a), (-b-det)/(2*a)




"""
import matplotlib.pyplot as  plt
from Utilities import ME,MGAMMA,MZ
vals = np.linspace(50,10000,1000)

figure = plt.figure()
ax = figure.add_subplot()
yvals = []
for s in vals:
    yvals.append(C0(-ME**2, -ME**2, ME, MZ, ME, -s))

ax.plot(vals,yvals)
"""