# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:38:21 2024

@author: herbi
"""

from Utilities import MZ,xfunc,vtheta,ctheta,stheta
import numpy as np

sval = 1000
tval = 500

def y1(s,t):
    return -4*vtheta/ctheta**2 * stheta**2 * ((xfunc(s,t)-t**2/s**2))

their1 = y1(sval,tval)/(-sval+MZ**2)

their2 = their1
their2 = 0

z1 = -1+8*stheta**2-24*stheta**4+32*stheta**6
their3 = z1/ctheta**4 * (sval*xfunc(sval,tval)-tval**2/2) * (1/(-sval+MZ**2)**2)

print(their3)

their = their1+ their2+their3

mine = vtheta*tval**2*(sval*(vtheta**2+1) - 16*ctheta**2*stheta**2*(sval-MZ**2))/(2*sval**2*ctheta**4*(sval-MZ**2))

print(their)
print(mine)
import matplotlib.pyplot as plt

figure = plt.figure()

ax = figure.add_subplot()

svals = np.linspace(50,80**2,1000)
tval = 500
their1 = y1(svals,tval)/(-svals+MZ**2)

their2 = their1
their2 = 0

z1 = -1+8*stheta**2-24*stheta**4+32*stheta**6
their3 = z1/ctheta**4 * (svals*xfunc(svals,tval)-tval**2/2) * (1/(-svals+MZ**2)**2)

their = their1+their2+their3
ax.plot(svals,their)


mine = vtheta*tval**2*(svals*(vtheta**2+1) - 16*ctheta**2*stheta**2*(svals-MZ**2))/(2*svals**2*ctheta**4*(svals-MZ**2))
ax.plot(svals,mine,label="mine")
ax.legend()